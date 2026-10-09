"""``overrides.json`` — the owner's optional edits, folded in by the engine (issue #241).

Fixture provenance (the producer of every shape used here): the playbook under
test is a real ``mine`` -> ``judge`` -> ``judge-apply`` -> ``project`` ->
``posture interview`` run over the synthetic NDA example, so its precedent
labels, verdict store (``judge/verdicts.jsonl``), Posture, Floor invariants,
digest, dossiers and ``precedent.jsonl`` are the ones the engine writes.
Verdicts are the dicts ``playbook judge-apply --verdicts`` accepts, with the four
labels cycled so every branch of the label vocabulary is present. The
``overrides.json`` files are the shape the page's Review tab writes
(``document_renderer`` / ``index_page_assets``), built here from the page's own
row builder (``review_rows.build_review_rows``).

SECURITY NOTE: synthetic text only (examples/nda).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner
from click.testing import Result as CliResult

from playbook_engine.agent_judge import VerdictStore
from playbook_engine.canonicalize import content_hash
from playbook_engine.cli import cli
from playbook_engine.digest import build_digest_v4
from playbook_engine.equivalence import LABELS, iter_slots, slot_key
from playbook_engine.opf_accessors import (
    PRECEDENT_SIDECAR,
    perspective_party,
    verify_precedent_sidecar,
)
from playbook_engine.overrides import (
    OVERRIDES_FILENAME,
    OverridesError,
    load_overrides,
    parse_overrides,
)
from playbook_engine.review_rows import build_review_rows

_ROOT = Path(__file__).resolve().parent.parent
_NDA_DIR = _ROOT / "examples" / "nda"
_CORPUS_DIR = _NDA_DIR / "corpus"
_SMOKE_CONFIG = _NDA_DIR / "config.smoke.yaml"
_CANNED = _NDA_DIR / "canned-verdicts.jsonl"
_POSTURE_ANSWERS = _NDA_DIR / "posture-answers.json"


def _invoke(args: list[str]) -> CliResult:
    return CliRunner().invoke(cli, args)


def _load(out_dir: Path) -> dict[str, Any]:
    return json.loads((out_dir / "playbook.opf.json").read_text(encoding="utf-8"))


def _write_overrides(out_dir: Path, entries: list[dict[str, Any]]) -> Path:
    path = out_dir / OVERRIDES_FILENAME
    path.write_text(
        json.dumps({"overrides_version": 1, "overrides": entries}, indent=2), encoding="utf-8"
    )
    return path


def _vs(key: str, label: str, **extra: Any) -> dict[str, Any]:
    return {
        "target": "vs_standard",
        "id": key,
        "value": {"label": label},
        "basis": "owner",
        **extra,
    }


# ---------------------------------------------------------------------------
# The producer: a real run
# ---------------------------------------------------------------------------


def _non_equivalence_canned() -> list[str]:
    return [
        line
        for line in _CANNED.read_text(encoding="utf-8").splitlines()
        if line.strip() and "label" not in json.loads(line)["verdict"]
    ]


@pytest.fixture(scope="module")
def derived_nda(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """mine -> judge -> judge-apply (labels cycled) -> project -> posture interview."""
    base = tmp_path_factory.mktemp("overrides")
    out_dir = base / "out"
    out_dir.mkdir()
    canned = base / "canned.jsonl"
    canned.write_text("\n".join(_non_equivalence_canned()) + "\n", encoding="utf-8")
    assert _invoke(["judge-apply", str(out_dir), "--verdicts", str(canned)]).exit_code == 0
    mine_args = ["--config", str(_SMOKE_CONFIG), "--out", str(out_dir)]
    assert _invoke(["mine", str(_CORPUS_DIR), *mine_args]).exit_code == 0
    assert _invoke(["judge", str(_CORPUS_DIR), *mine_args]).exit_code == 0
    pending = [
        json.loads(line)
        for line in (out_dir / "judge" / "pending.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    items = [p for p in pending if p["kind"] == "equivalence"]
    assert len(items) >= 8, "premise: the NDA example queues several equivalence questions"
    drafts = out_dir / "drafts.jsonl"
    drafts.write_text(
        "\n".join(
            json.dumps(
                {
                    "key": item["key"],
                    "verdict": {
                        "label": LABELS[i % len(LABELS)],
                        "reason": f"Model reason {i}.",
                        "basis": "agent",
                    },
                }
            )
            for i, item in enumerate(items)
        )
        + "\n",
        encoding="utf-8",
    )
    assert _invoke(["judge-apply", str(out_dir), "--verdicts", str(drafts)]).exit_code == 0
    project = _invoke(["project", str(out_dir), "--config", str(_SMOKE_CONFIG)])
    assert project.exit_code == 0, project.output
    interview = _invoke(
        ["posture", "interview", str(out_dir), "--answers-file", str(_POSTURE_ANSWERS)]
    )
    assert interview.exit_code == 0, interview.output
    assert _invoke(["validate", str(out_dir / "playbook.opf.json")]).exit_code == 0
    return out_dir


@pytest.fixture
def out_dir(derived_nda: Path, tmp_path: Path) -> Path:
    dest = tmp_path / "out"
    shutil.copytree(derived_nda, dest)
    return dest


def _first_row(doc: dict[str, Any], label: str) -> dict[str, Any]:
    rows = [r for r in build_review_rows(doc).rows if r["label"] == label]
    assert rows, f"premise: the derived playbook has a Review row labelled {label}"
    return rows[0]


def _entries_for(doc: dict[str, Any], key: str) -> list[dict[str, Any]]:
    """The ``vs_standard`` objects of every text (any record, any role) carrying verdict *key*."""
    party = perspective_party(doc)
    type_id = doc["agreement_type"]["id"]
    return [
        slot.entry["vs_standard"]
        for slot in iter_slots(doc["evidence"])
        if isinstance(slot.entry.get("vs_standard"), dict) and slot_key(slot, type_id, party) == key
    ]


# ---------------------------------------------------------------------------
# Round trip: apply-overrides
# ---------------------------------------------------------------------------


def test_the_derived_playbook_has_every_label_and_a_posture_and_floor(derived_nda: Path) -> None:
    doc = _load(derived_nda)
    labels = {r["label"] for r in build_review_rows(doc).rows}
    assert labels == set(LABELS), labels
    assert doc["posture"]["system_prompt"] and doc["floor"]["invariants"]


def test_a_vs_standard_and_a_posture_override_change_the_playbook_and_restamp_owner(
    out_dir: Path,
) -> None:
    before = _load(out_dir)
    old_hash = before["identity"]["content_hash"]
    row = _first_row(before, "less_protective")
    posture = "Owner-edited posture: hold the line on survival and exclusions."
    invariant = before["floor"]["invariants"][0]
    _write_overrides(
        out_dir,
        [
            _vs(row["key"], "equivalent", note="Reviewed by counsel."),
            {"target": "posture", "id": "system_prompt", "value": posture, "basis": "owner"},
            {
                "target": "floor",
                "id": invariant["id"],
                "field": "statement",
                "value": "Do not concede on this clause, ever.",
                "basis": "owner",
            },
        ],
    )

    result = _invoke(["apply-overrides", str(out_dir)])
    assert result.exit_code == 0, result.output

    after = _load(out_dir)
    # the label changed and was re-stamped owner, with no check
    assert _entries_for(after, row["key"]), "premise: the key still addresses a labelled text"
    for vs in _entries_for(after, row["key"]):
        assert vs["label"] == "equivalent"
        assert vs["basis"] == "owner"
        assert vs["check"] is None
        assert vs["reason"]
    # the Posture text and the Floor statement changed, the Posture version moved
    assert after["posture"]["system_prompt"] == posture
    assert after["posture"]["version"] == before["posture"]["version"] + 1
    assert after["floor"]["invariants"][0]["statement"] == "Do not concede on this clause, ever."
    # everything derived was recomputed by the engine, never hand-edited
    assert after["identity"]["content_hash"] != old_hash
    assert after["identity"]["content_hash"] == content_hash(after)
    assert after["digest"] == build_digest_v4(after)
    assert verify_precedent_sidecar(after, out_dir / PRECEDENT_SIDECAR)
    # and the result validates, against the verdict store that now holds the owner verdict
    assert _invoke(["validate", str(out_dir / "playbook.opf.json")]).exit_code == 0
    stored = VerdictStore(out_dir / "judge" / "verdicts.jsonl").get_by_key(row["key"])
    assert stored is not None and stored["basis"] == "owner" and stored["label"] == "equivalent"


def test_confirming_a_label_restamps_the_basis_and_keeps_the_models_reason(
    out_dir: Path,
) -> None:
    before = _load(out_dir)
    row = _first_row(before, "different_concept")
    assert _entries_for(before, row["key"])[0]["basis"] == "agent"
    _write_overrides(out_dir, [_vs(row["key"], "different_concept")])

    assert _invoke(["apply-overrides", str(out_dir)]).exit_code == 0

    after = _load(out_dir)
    for vs in _entries_for(after, row["key"]):
        assert vs["label"] == "different_concept" and vs["basis"] == "owner"
        assert vs["reason"] == row["reason"], "a confirmation keeps the model's reason"
    assert after["identity"]["content_hash"] != before["identity"]["content_hash"]


def test_changing_the_label_replaces_the_reason_that_would_now_contradict_it(
    out_dir: Path,
) -> None:
    before = _load(out_dir)
    row = _first_row(before, "more_protective")
    _write_overrides(out_dir, [_vs(row["key"], "less_protective")])
    assert _invoke(["apply-overrides", str(out_dir)]).exit_code == 0
    for vs in _entries_for(_load(out_dir), row["key"]):
        assert vs["label"] == "less_protective"
        assert vs["reason"] != row["reason"]


def test_reapplying_an_applied_file_changes_nothing(out_dir: Path) -> None:
    doc = _load(out_dir)
    row = _first_row(doc, "less_protective")
    _write_overrides(
        out_dir,
        [
            _vs(row["key"], "equivalent"),
            {"target": "posture", "id": "system_prompt", "value": "New posture.", "basis": "owner"},
        ],
    )
    assert _invoke(["apply-overrides", str(out_dir)]).exit_code == 0
    snapshot = _snapshot(out_dir)
    again = _invoke(["apply-overrides", str(out_dir)])
    assert again.exit_code == 0 and "no change" in again.output
    assert _snapshot(out_dir) == snapshot, "a second apply changed a file"
    assert _load(out_dir)["posture"]["version"] == doc["posture"]["version"] + 1


def test_an_override_restores_a_text_the_playbook_already_carries_without_a_store(
    tmp_path: Path,
) -> None:
    """A published playbook next to no verdict store: the labels are edited in the
    document alone (there is nothing to replay)."""
    out = tmp_path / "out"
    out.mkdir()
    shutil.copy(_NDA_DIR / "playbook.opf.json", out / "playbook.opf.json")
    shutil.copy(_NDA_DIR / PRECEDENT_SIDECAR, out / PRECEDENT_SIDECAR)
    doc = _load(out)
    row = _first_row(doc, "less_protective")
    _write_overrides(out, [_vs(row["key"], "more_protective")])
    assert _invoke(["apply-overrides", str(out)]).exit_code == 0
    after = _load(out)
    assert {vs["label"] for vs in _entries_for(after, row["key"])} == {"more_protective"}
    assert not (out / "judge").exists(), "no store existed, so none is invented"
    assert _invoke(["validate", str(out / "playbook.opf.json")]).exit_code == 0


# ---------------------------------------------------------------------------
# Rejection: malformed and unknown-id entries change nothing
# ---------------------------------------------------------------------------


def _snapshot(out_dir: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(out_dir)): p.read_bytes()
        for p in sorted(out_dir.rglob("*"))
        if p.is_file() and ".cache" not in p.parts
    }


_KEY = "a" * 64


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ("{not json", "not valid JSON"),
        ('{"overrides_version": 2, "overrides": []}', "overrides_version must be 1"),
        ('{"overrides_version": 1}', "overrides must be a list"),
        ('{"overrides_version": 1, "overrides": [], "extra": 1}', "unknown top-level key"),
        (
            json.dumps({"overrides_version": 1, "overrides": [_vs("short", "equivalent")]}),
            "64-hex verdict cache key",
        ),
        (
            json.dumps({"overrides_version": 1, "overrides": [_vs(_KEY, "very_protective")]}),
            "label must be one of",
        ),
        (
            json.dumps(
                {
                    "overrides_version": 1,
                    "overrides": [{**_vs(_KEY, "equivalent"), "basis": "judge"}],
                }
            ),
            'basis must be "owner"',
        ),
        (
            json.dumps(
                {
                    "overrides_version": 1,
                    "overrides": [_vs(_KEY, "equivalent"), _vs(_KEY, "less_protective")],
                }
            ),
            "duplicates overrides[0]",
        ),
        (
            json.dumps(
                {
                    "overrides_version": 1,
                    "overrides": [
                        {
                            "target": "floor",
                            "id": "x",
                            "field": "id",
                            "value": "y",
                            "basis": "owner",
                        }
                    ],
                }
            ),
            "floor field must be one of",
        ),
        (
            json.dumps(
                {
                    "overrides_version": 1,
                    "overrides": [
                        {"target": "posture", "id": "version", "value": "2", "basis": "owner"}
                    ],
                }
            ),
            'the only posture field is "system_prompt"',
        ),
        (
            json.dumps(
                {
                    "overrides_version": 1,
                    "overrides": [
                        {"target": "posture", "id": "system_prompt", "value": " ", "basis": "owner"}
                    ],
                }
            ),
            "non-empty text",
        ),
    ],
)
def test_a_malformed_file_is_rejected_with_a_clear_error_and_changes_nothing(
    out_dir: Path, raw: str, message: str
) -> None:
    (out_dir / OVERRIDES_FILENAME).write_text(raw, encoding="utf-8")
    before = _snapshot(out_dir)
    result = _invoke(["apply-overrides", str(out_dir)])
    assert result.exit_code == 1
    assert "was rejected; nothing was changed" in result.output
    assert message in result.output
    assert _snapshot(out_dir) == before


def test_an_unknown_id_is_rejected_and_nothing_else_in_the_file_is_applied(
    out_dir: Path,
) -> None:
    doc = _load(out_dir)
    row = _first_row(doc, "less_protective")
    _write_overrides(
        out_dir,
        [
            _vs(row["key"], "equivalent"),  # valid, and must NOT be applied alone
            _vs(_KEY, "equivalent"),
            {
                "target": "floor",
                "id": "no-such-invariant",
                "field": "statement",
                "value": "x",
                "basis": "owner",
            },
        ],
    )
    before = _snapshot(out_dir)
    result = _invoke(["apply-overrides", str(out_dir)])
    assert result.exit_code == 1
    assert "no judged text in this playbook has that verdict key" in result.output
    assert "the Floor has no invariant with that id" in result.output
    assert _snapshot(out_dir) == before, "one bad entry must leave every file untouched"


def test_apply_overrides_needs_both_files(tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    result = _invoke(["apply-overrides", str(out)])
    assert result.exit_code == 1 and "playbook.opf.json not found" in result.output
    (out / "playbook.opf.json").write_text("{}", encoding="utf-8")
    result = _invoke(["apply-overrides", str(out)])
    assert result.exit_code == 1 and "overrides.json not found" in result.output


def test_load_overrides_names_the_file_and_parse_returns_entries(tmp_path: Path) -> None:
    path = tmp_path / OVERRIDES_FILENAME
    path.write_text('{"overrides_version": 1, "overrides": [], "x": 1}', encoding="utf-8")
    with pytest.raises(OverridesError, match=r"overrides\.json: unknown top-level key"):
        load_overrides(path)
    entries = parse_overrides(
        {
            "overrides_version": 1,
            "based_on": "sha256:abc",
            "overrides": [
                _vs(_KEY, "equivalent", note="n"),
                {
                    "target": "floor",
                    "id": "i",
                    "field": "rationale",
                    "value": "r",
                    "basis": "owner",
                },
            ],
        }
    )
    assert [e.target for e in entries] == ["vs_standard", "floor"]
    assert entries[0].note == "n" and entries[1].field == "rationale"


# ---------------------------------------------------------------------------
# project folds the file in
# ---------------------------------------------------------------------------


def test_project_folds_overrides_in_and_matches_apply_overrides(
    out_dir: Path, tmp_path: Path
) -> None:
    doc = _load(out_dir)
    row = _first_row(doc, "less_protective")
    entries = [
        _vs(row["key"], "equivalent"),
        {"target": "posture", "id": "system_prompt", "value": "Folded posture.", "basis": "owner"},
    ]
    applied = tmp_path / "applied"
    shutil.copytree(out_dir, applied)
    _write_overrides(applied, entries)
    assert _invoke(["apply-overrides", str(applied)]).exit_code == 0

    _write_overrides(out_dir, entries)
    projected = _invoke(["project", str(out_dir), "--config", str(_SMOKE_CONFIG)])
    assert projected.exit_code == 0, projected.output
    assert "overrides: 2 entries" in projected.output

    from_project = _load(out_dir)
    from_apply = _load(applied)
    assert {vs["label"] for vs in _entries_for(from_project, row["key"])} == {"equivalent"}
    assert from_project["posture"]["system_prompt"] == "Folded posture."
    assert from_project["identity"]["content_hash"] == from_apply["identity"]["content_hash"], (
        "project and apply-overrides fold the same file into the same playbook"
    )
    assert _invoke(["validate", str(out_dir / "playbook.opf.json")]).exit_code == 0


def test_an_owner_label_survives_a_reprojection_without_the_file(out_dir: Path) -> None:
    """The owner verdict was written into the verdict store, so a later project replays it."""
    doc = _load(out_dir)
    row = _first_row(doc, "less_protective")
    path = _write_overrides(out_dir, [_vs(row["key"], "equivalent")])
    assert _invoke(["apply-overrides", str(out_dir)]).exit_code == 0
    path.unlink()
    assert _invoke(["project", str(out_dir), "--config", str(_SMOKE_CONFIG)]).exit_code == 0
    for vs in _entries_for(_load(out_dir), row["key"]):
        assert vs["label"] == "equivalent" and vs["basis"] == "owner"


def test_project_skips_a_stale_override_reports_it_and_applies_the_rest(out_dir: Path) -> None:
    """An optional review file never gates a run (a stale entry is normal after a re-derivation)."""
    doc = _load(out_dir)
    row = _first_row(doc, "less_protective")
    _write_overrides(
        out_dir,
        [
            _vs(_KEY, "equivalent"),
            _vs(row["key"], "equivalent"),
            {
                "target": "floor",
                "id": "no-such-invariant",
                "field": "statement",
                "value": "x",
                "basis": "owner",
            },
        ],
    )
    result = _invoke(["project", str(out_dir), "--config", str(_SMOKE_CONFIG)])
    assert result.exit_code == 0, result.output
    assert "overrides: 3 entries (1 changed the playbook, 0 already in effect, 2 skipped)" in (
        result.output
    )
    assert result.output.count("overrides skipped:") == 2
    assert "remove the entry from overrides.json" in result.output
    assert {vs["label"] for vs in _entries_for(_load(out_dir), row["key"])} == {"equivalent"}
    assert _invoke(["validate", str(out_dir / "playbook.opf.json")]).exit_code == 0
    # the strict command still refuses the same file, and changes nothing
    before = _snapshot(out_dir)
    strict = _invoke(["apply-overrides", str(out_dir)])
    assert strict.exit_code != 0
    assert _snapshot(out_dir) == before


def test_project_skips_a_malformed_overrides_file_and_still_projects(out_dir: Path) -> None:
    (out_dir / OVERRIDES_FILENAME).write_text("{not json", encoding="utf-8")
    before = _load(out_dir)["identity"]["content_hash"]
    result = _invoke(["project", str(out_dir), "--config", str(_SMOKE_CONFIG)])
    assert result.exit_code == 0, result.output
    assert "could not be used, none applied" in result.output
    assert "overrides skipped:" in result.output
    assert _load(out_dir)["identity"]["content_hash"] == before


def test_project_without_overrides_is_unchanged(out_dir: Path) -> None:
    """No file, no step: the projection's content hash is what it was."""
    before = _load(out_dir)["identity"]["content_hash"]
    # the posture interview ran after the first projection; the next one carries it forward
    assert _invoke(["project", str(out_dir), "--config", str(_SMOKE_CONFIG)]).exit_code == 0
    assert _load(out_dir)["identity"]["content_hash"] == before


# ---------------------------------------------------------------------------
# Floor overrides keep the engine's attribution invariants
# ---------------------------------------------------------------------------


def _sign(out_dir: Path) -> str:
    result = _invoke(
        [
            "floor", "sign", str(out_dir),
            "--statement", "Liability caps must never reach a confidentiality breach.",
            "--signed-by", "Legal Owner",
            "--id", "liability-carveout",
        ]
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    return "liability-carveout"


def _floor_override(inv_id: str, field: str, value: str) -> dict[str, Any]:
    return {"target": "floor", "id": inv_id, "field": field, "value": value, "basis": "owner"}


@pytest.mark.parametrize("field", ["statement", "rationale"])
def test_a_floor_override_of_a_signed_invariant_is_refused_and_changes_nothing(
    out_dir: Path, field: str
) -> None:
    inv_id = _sign(out_dir)
    _write_overrides(out_dir, [_floor_override(inv_id, field, "Rewritten by someone else.")])
    before = _snapshot(out_dir)
    result = _invoke(["apply-overrides", str(out_dir)])
    assert result.exit_code != 0
    assert "x_signed_by" in result.output and "floor sign" in result.output
    assert _snapshot(out_dir) == before
    signed = next(i for i in _load(out_dir)["floor"]["invariants"] if i["id"] == inv_id)
    assert signed["x_signed_by"] == "Legal Owner" and "x_signed_at" in signed
    assert "Rewritten" not in json.dumps(signed)


def test_a_rationale_override_of_a_q4_promoted_invariant_is_refused(out_dir: Path) -> None:
    inv = _load(out_dir)["floor"]["invariants"][0]
    assert inv["rationale"].startswith("Authored by the legal owner in posture interview")
    _write_overrides(out_dir, [_floor_override(inv["id"], "rationale", "Because I said so.")])
    before = _snapshot(out_dir)
    result = _invoke(["apply-overrides", str(out_dir)])
    assert result.exit_code != 0
    assert "attribution marker" in result.output
    assert _snapshot(out_dir) == before


def test_a_statement_override_of_a_q4_invariant_round_trips_through_the_interview(
    out_dir: Path,
) -> None:
    """Edit the statement, re-run the interview with the same answers, re-apply: all valid."""
    inv_id = _load(out_dir)["floor"]["invariants"][0]["id"]
    _write_overrides(out_dir, [_floor_override(inv_id, "statement", "Owner-worded hard line.")])
    assert _invoke(["apply-overrides", str(out_dir)]).exit_code == 0
    edited = next(i for i in _load(out_dir)["floor"]["invariants"] if i["id"] == inv_id)
    assert edited["statement"] == "Owner-worded hard line." and edited["x_basis"] == "owner"
    # attribution still holds: validate raises no "no structural attribution" warning
    validated = _invoke(["validate", str(out_dir / "playbook.opf.json")])
    assert validated.exit_code == 0 and "no structural attribution" not in validated.output
    # the interview re-run with the same answers is not refused
    rerun = _invoke(["posture", "interview", str(out_dir), "--answers-file", str(_POSTURE_ANSWERS)])
    assert rerun.exit_code == 0, rerun.output
    # and re-applying the overrides (a no-op when already in effect) puts the edit back
    assert _invoke(["apply-overrides", str(out_dir)]).exit_code == 0
    again = next(i for i in _load(out_dir)["floor"]["invariants"] if i["id"] == inv_id)
    assert again["statement"] == "Owner-worded hard line."
    validated = _invoke(["validate", str(out_dir / "playbook.opf.json")])
    assert validated.exit_code == 0 and "no structural attribution" not in validated.output


def test_the_page_offers_no_edit_box_where_the_engine_refuses_the_edit(out_dir: Path) -> None:
    from playbook_engine.document_renderer import render_index_html  # noqa: PLC0415

    inv_id = _sign(out_dir)
    doc = _load(out_dir)
    page = render_index_html(out_dir)
    q4 = next(i for i in doc["floor"]["invariants"] if i["id"] != inv_id)
    assert f'data-target="floor" data-id="{inv_id}"' not in page
    assert f'data-target="floor" data-id="{q4["id"]}" data-field="statement"' in page
    assert f'data-target="floor" data-id="{q4["id"]}" data-field="rationale"' not in page
    assert "Not editable here" in page
