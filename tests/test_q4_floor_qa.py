"""Q4 / Floor fixes from the 2026-11-03 QA sweep (track D): findings 6, 7, 17
(docs only), 30, 31, 32 and the separator note. Keyless, synthetic names only."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from playbook_engine import floor_candidates as fc
from playbook_engine.canonicalize import compute_section_digests, content_hash
from playbook_engine.cli import cli
from playbook_engine.floor_candidates import (
    FloorCandidateError,
    _q4_items,
    dropped_q4_invariants,
    floor_evidence_conflicts,
    floor_invariant_attribution,
    is_q4_null_like,
    match_taxonomy_entry,
    promote_interview_q4_invariants,
    resolve_q4_items,
    retire_floor_invariant,
)
from playbook_engine.posture import apply_posture_interview, preview_q4_statements

_TAX = [
    {"id": "survival_period", "label": "Survival of Obligations", "status": "active"},
    {"id": "limitation_of_liability", "label": "Limitation of Liability", "status": "active"},
    {"id": "residuals", "label": "Residuals", "status": "active"},
    {"id": "old_clause", "label": "Old Clause", "status": "inactive"},
]
_ANSWERS = {
    "rounds": "Two rounds.",
    "leverage": "Collaborative.",
    "risk_appetite": "Accept to close on non-material changes.",
    "sacred_clauses": "Survival of Obligations; Residuals",
    "flexible_clauses": "Governing law.",
    "audience": "Terse.",
}


def _write(tmp_path: Path, **sections: Any) -> Path:
    doc: dict[str, Any] = {
        "opf_version": "0.6",
        "agreement_type": {"id": "t", "name": "T"},
        "baseline": {"has_canonical_template": False},
        "perspective": {"party": "Acme Widgets LLC", "counterparty_type": "Customer"},
        "taxonomy": {"source": "custom", "entries": _TAX},
        "evidence": {"clauses": [], "precedent": [], "absence": {"guard": 0.1, "records": []}},
        "posture": {},
        "floor": {},
        "corpus": {"documents": [], "stats": {}},
        "compiler": {"name": "playbook-engine", "version": "0", "run_id": "r", "generated_at": "x"},
    }
    doc.update(sections)
    doc["identity"] = {
        "content_hash": content_hash(doc),
        "section_digests": compute_section_digests(doc),
    }
    path = tmp_path / "playbook.opf.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


# --- #6: null-like, separators, taxonomy, polarity -----------------------------


@pytest.mark.parametrize(
    "item", ["None", "N/A", "n/a", "not sure", "I'm not sure yet", "TBD", "-", "N/A - nothing", ""]
)
def test_null_like_items_are_recognised(item: str) -> None:
    assert is_q4_null_like(item)


@pytest.mark.parametrize("item", ["Residuals", "No limitation of liability", "Term"])
def test_real_names_are_not_null_like(item: str) -> None:
    assert not is_q4_null_like(item)


def test_null_answer_promotes_nothing() -> None:
    for answer in ("None", "N/A", "I'm not sure yet", "none; n/a"):
        assert (
            promote_interview_q4_invariants(
                {"sacred_clauses": answer}, posture_version=1, existing_invariants=[]
            )
            == []
        )


def test_commas_and_semicolons_both_split() -> None:
    assert _q4_items("A, B and C; D") == ["A", "B and C", "D"]
    assert _q4_items("A, B, and C") == ["A", "B", "C"]
    assert _q4_items("Confidentiality (incl. residuals, survival), Term") == [
        "Confidentiality (incl. residuals, survival)",
        "Term",
    ]
    # a conditional sentence is never shredded on its commas
    sentence = "limitation of liability, if present, must not be unilateral"
    assert _q4_items(sentence) == [sentence]


def test_taxonomy_match_exact_and_normalized() -> None:
    assert match_taxonomy_entry("Survival of Obligations", _TAX) == "survival_period"
    assert match_taxonomy_entry("limitation of liability", _TAX) == "limitation_of_liability"
    assert match_taxonomy_entry("residual", _TAX) == "residuals"
    assert match_taxonomy_entry("Old Clause", _TAX) is None  # inactive
    assert match_taxonomy_entry("Something else", _TAX) is None
    assert match_taxonomy_entry("Residuals", None) is None


def test_promoted_item_gets_taxonomy_id_and_hold_direction() -> None:
    result = promote_interview_q4_invariants(
        {"sacred_clauses": "Survival of Obligations"},
        posture_version=1,
        existing_invariants=[],
        taxonomy_entries=_TAX,
        signed_by="J. Placeholder",
    )
    inv = result[0]
    assert inv["statement"] == "Do not concede on Survival of Obligations."
    assert inv["x_taxonomy_id"] == "survival_period"
    assert inv["x_direction"] == "hold"
    assert inv["x_signed_by"] == "J. Placeholder"


def test_refused_items_read_never_accept() -> None:
    items = {
        i.text: i for i in resolve_q4_items("Uncapped liability; no limitation of liability", _TAX)
    }
    assert items["Uncapped liability"].statement == "Never accept uncapped liability."
    assert items["Uncapped liability"].direction == "refuse"
    no_lol = items["no limitation of liability"]
    assert no_lol.statement == "Never accept limitation of liability."
    assert no_lol.taxonomy_id == "limitation_of_liability"


@pytest.mark.parametrize(
    "item", ["never assign", "don't let them weaken confidentiality", "without a survival period"]
)
def test_unclear_direction_is_held_not_guessed(item: str) -> None:
    (resolved,) = resolve_q4_items(item, _TAX)
    assert resolved.kind == "ambiguous"
    assert (
        promote_interview_q4_invariants(
            {"sacred_clauses": item}, posture_version=1, existing_invariants=[]
        )
        == []
    )


def test_apply_interview_warns_for_nulls_and_ambiguous_items(tmp_path: Path) -> None:
    _write(tmp_path)
    answers = {**_ANSWERS, "sacred_clauses": "Residuals; none; never assign"}
    result = apply_posture_interview(tmp_path, answers, generated_at="2026-11-03T00:00:00Z")
    text = "\n".join(result.warnings)
    assert "'none'" in text and "not a clause name" in text
    assert "'never assign'" in text and "playbook floor sign" in text
    ids = [i["id"] for i in _load(tmp_path / "playbook.opf.json")["floor"]["invariants"]]
    assert ids == ["residuals"]


def test_preview_echoes_exact_statements(tmp_path: Path) -> None:
    _write(tmp_path)
    assert preview_q4_statements(tmp_path, {"sacred_clauses": "Uncapped liability, Residuals"}) == [
        "Never accept uncapped liability.",
        "Do not concede on Residuals.",
    ]


def test_cli_interview_echoes_statements_before_writing(tmp_path: Path) -> None:
    _write(tmp_path)
    answers = tmp_path / "answers.json"
    answers.write_text(json.dumps({**_ANSWERS, "sacred_clauses": "Residuals"}), encoding="utf-8")
    res = CliRunner().invoke(
        cli,
        [
            "posture",
            "interview",
            str(tmp_path),
            "--answers-file",
            str(answers),
            "--signed-by",
            "J. P.",
        ],
    )
    assert res.exit_code == 0, res.output
    assert res.output.index("Do not concede on Residuals.") < res.output.index("posture.version=")
    inv = _load(tmp_path / "playbook.opf.json")["floor"]["invariants"][0]
    assert inv["x_signed_by"] == "J. P." and inv["x_taxonomy_id"] == "residuals"


def test_questions_binding_note_mentions_separators() -> None:
    from playbook_engine.posture import INTERVIEW_QUESTIONS

    note = next(q.binding_note for q in INTERVIEW_QUESTIONS if q.q == "sacred_clauses")
    assert "';'" in note and "','" in note


# --- #30: posture prose and signer ---------------------------------------------


def test_prose_lists_only_signed_items_as_hard_lines(tmp_path: Path) -> None:
    path = _write(tmp_path)
    answers = {**_ANSWERS, "sacred_clauses": "Residuals; never assign; none"}
    apply_posture_interview(tmp_path, answers, generated_at="2026-11-03T00:00:00Z")
    prompt = _load(path)["posture"]["system_prompt"]
    assert "Hard lines named in interview (see Floor): Residuals." in prompt
    assert "NOT signed into the Floor" in prompt and "never assign" in prompt
    assert "none" not in prompt.lower().replace("non-material", "")
    # the recorded interview keeps the raw answer
    entry = next(
        e for e in _load(path)["posture"]["generation"]["interview"] if e["q"] == "sacred_clauses"
    )
    assert entry["answer"] == "Residuals; never assign; none"


def test_declining_promotion_signs_nothing_and_prose_says_so(tmp_path: Path) -> None:
    path = _write(tmp_path)
    apply_posture_interview(
        tmp_path, _ANSWERS, generated_at="2026-11-03T00:00:00Z", promote_q4=False
    )
    doc = _load(path)
    assert doc["floor"] == {}
    assert "Hard lines named in interview" not in doc["posture"]["system_prompt"]


def test_q4_marker_records_signer_or_admits_none() -> None:
    signed = promote_interview_q4_invariants(
        {"sacred_clauses": "Residuals"}, posture_version=1, signed_by="J. P."
    )[0]
    unsigned = promote_interview_q4_invariants({"sacred_clauses": "Residuals"}, posture_version=1)[
        0
    ]
    assert signed["rationale"].startswith("Authored by the legal owner")
    assert "x_signed_by" in signed
    assert "legal owner" not in unsigned["rationale"] and "no signer" in unsigned["rationale"]
    assert "x_signed_by" not in unsigned
    assert floor_invariant_attribution(signed) == "posture_interview"
    assert floor_invariant_attribution(unsigned) == "posture_interview"


def test_rerun_with_signer_stamps_an_unsigned_line_without_rewriting_it() -> None:
    first = promote_interview_q4_invariants({"sacred_clauses": "Residuals"}, posture_version=1)
    second = promote_interview_q4_invariants(
        {"sacred_clauses": "Residuals"},
        posture_version=2,
        existing_invariants=first,
        signed_by="J. P.",
    )
    assert second[0]["statement"] == first[0]["statement"]
    assert second[0]["rationale"] == first[0]["rationale"]
    assert second[0]["x_signed_by"] == "J. P."


# --- #7: dropped Q4 lines warn; retire is explicit -----------------------------


def test_dropped_q4_line_stays_and_warns_with_retire_command(tmp_path: Path) -> None:
    path = _write(tmp_path)
    apply_posture_interview(tmp_path, _ANSWERS, generated_at="2026-11-03T00:00:00Z")
    narrowed = {**_ANSWERS, "sacred_clauses": "Residuals"}
    result = apply_posture_interview(tmp_path, narrowed, generated_at="2026-11-03T01:00:00Z")
    ids = [i["id"] for i in _load(path)["floor"]["invariants"]]
    assert "survival-of-obligations" in ids  # nothing binding removed silently
    text = "\n".join(result.warnings)
    assert "STILL BINDING" in text
    assert "playbook floor retire" in text and "--id survival-of-obligations" in text


def test_dropped_helper_ignores_hand_signed_lines() -> None:
    inv = [{"id": "x", "statement": "s", "rationale": "r", "x_signed_by": "J. P."}]
    assert dropped_q4_invariants({"sacred_clauses": "Residuals"}, inv) == []


def test_floor_retire_cli_removes_records_and_blocks_repromotion(tmp_path: Path) -> None:
    path = _write(tmp_path)
    apply_posture_interview(tmp_path, _ANSWERS, generated_at="2026-11-03T00:00:00Z")
    runner = CliRunner()
    res = runner.invoke(
        cli,
        ["floor", "retire", str(tmp_path), "--id", "residuals", "--signed-by", "J. P."],
    )
    assert res.exit_code == 0, res.output
    doc = _load(path)
    assert [i["id"] for i in doc["floor"]["invariants"]] == ["survival-of-obligations"]
    assert doc["floor"]["x_retired"][0]["x_retired_by"] == "J. P."
    assert doc["identity"]["content_hash"] == content_hash(doc)
    # the same Q4 answer does not bring it back
    result = apply_posture_interview(tmp_path, _ANSWERS, generated_at="2026-11-03T02:00:00Z")
    assert [i["id"] for i in _load(path)["floor"]["invariants"]] == ["survival-of-obligations"]
    assert any("retired" in w for w in result.warnings)
    # validates
    from playbook_engine.validator import validate_document

    assert not [e for e in validate_document(_load(path)).errors if e.blocking]


def test_floor_retire_requires_signer_and_known_id() -> None:
    floor = {"invariants": [{"id": "a", "statement": "s"}]}
    with pytest.raises(FloorCandidateError):
        retire_floor_invariant(floor, "a", signed_by=" ")
    with pytest.raises(FloorCandidateError):
        retire_floor_invariant(floor, "zzz", signed_by="J. P.")
    out = retire_floor_invariant(
        floor, "a", signed_by="J. P.", retired_at="2026-11-03T00:00:00+00:00"
    )
    assert out["invariants"] == [] and out["x_retired"][0]["statement"] == "s"


def test_sign_collision_error_points_at_retire(tmp_path: Path) -> None:
    from playbook_engine.floor_candidates import sign_floor_invariant

    existing = [{"id": "a", "statement": "old"}]
    with pytest.raises(FloorCandidateError, match="floor retire"):
        sign_floor_invariant(
            "new", invariant_id="a", signed_by="J. P.", existing_invariants=existing
        )


# --- #31: Floor vs signed Evidence ---------------------------------------------


def _evidence_doc() -> dict[str, Any]:
    return {
        "evidence": {
            "clauses": [
                {
                    "taxonomy_id": "survival_period",
                    "n_deals": 6,
                    "n_signed_standard": 4,
                    "n_variants": 2,
                    "n_refused": 1,
                },
                {"taxonomy_id": "limitation_of_liability", "n_deals": 3, "n_variants": 0},
            ],
            "absence": {
                "records": [
                    {"taxonomy_id": "residuals", "document_id": "d1", "status": "absent_verified"}
                ]
            },
        },
        "floor": {
            "invariants": [
                {
                    "id": "h",
                    "statement": "Do not concede on Survival.",
                    "x_taxonomy_id": "survival_period",
                    "x_direction": "hold",
                },
                {
                    "id": "r",
                    "statement": "Never accept LoL.",
                    "x_taxonomy_id": "limitation_of_liability",
                    "x_direction": "refuse",
                },
                {
                    "id": "p",
                    "statement": "Residuals required.",
                    "x_taxonomy_id": "residuals",
                    "x_required_presence": True,
                },
                {"id": "free", "statement": "No clause tie."},
            ]
        },
    }


def test_floor_evidence_conflicts_flags_each_kind() -> None:
    warnings = floor_evidence_conflicts(_evidence_doc())
    text = "\n".join(warnings)
    assert len(warnings) == 3
    assert "'h'" in text and "non-standard" in text and "refused in 1" in text
    assert "'r'" in text and "3 signed deal(s) contain" in text
    assert "'p'" in text and "verified to lack" in text
    assert "Warning only" in text


def test_interview_surfaces_evidence_conflict_as_warning_only(tmp_path: Path) -> None:
    ev = _evidence_doc()["evidence"]
    _write(tmp_path, evidence=ev)
    result = apply_posture_interview(tmp_path, _ANSWERS, generated_at="2026-11-03T00:00:00Z")
    assert any("contradicts signed Evidence" in w for w in result.warnings)  # survival_period


def test_signed_required_presence_line_conflicts_with_absence_evidence(tmp_path: Path) -> None:
    ev = _evidence_doc()["evidence"]
    _write(tmp_path, evidence=ev)
    from playbook_engine.floor_candidates import sign_floor_invariant

    invs = sign_floor_invariant(
        "Residuals must be present.",
        invariant_id="res",
        taxonomy_id="residuals",
        signed_by="J. P.",
        required_presence=True,
    )
    doc = _load(tmp_path / "playbook.opf.json")
    doc["floor"] = {"invariants": invs}
    assert floor_evidence_conflicts(doc)


# --- #32: no silent revert, no version churn ------------------------------------


def test_owner_edited_q4_statement_survives_rerun() -> None:
    first = promote_interview_q4_invariants(
        {"sacred_clauses": "Residuals"}, posture_version=1, signed_by="J. P."
    )
    edited = [{**first[0], "statement": "Never accept any residuals clause."}]
    again = promote_interview_q4_invariants(
        {"sacred_clauses": "Residuals"},
        posture_version=2,
        existing_invariants=edited,
        signed_by="J. P.",
    )
    assert again == edited


def test_rerun_after_page_edits_is_a_true_noop(tmp_path: Path) -> None:
    path = _write(tmp_path)
    apply_posture_interview(
        tmp_path, _ANSWERS, generated_at="2026-11-03T00:00:00Z", signed_by="J. P."
    )
    doc = _load(path)
    doc["posture"]["system_prompt"] = "Owner rewrote this."
    doc["floor"]["invariants"][0]["statement"] = "Owner edited statement."
    doc["identity"]["content_hash"] = content_hash(doc)
    doc["identity"]["section_digests"] = compute_section_digests(doc)
    path.write_text(json.dumps(doc), encoding="utf-8")
    before = path.read_text(encoding="utf-8")
    version = doc["posture"]["version"]

    for stamp in ("2026-11-03T01:00:00Z", "2026-11-03T02:00:00Z"):
        result = apply_posture_interview(tmp_path, _ANSWERS, generated_at=stamp, signed_by="J. P.")
        assert result.changed is False and result.version == version
    assert path.read_text(encoding="utf-8") == before


def test_real_clause_names_that_open_with_a_null_word_are_not_non_answers() -> None:
    for item in ("Pass-through costs", "Nothing-to-disclose certificate", "Skip tracing"):
        assert not fc.is_q4_null_like(item), item
    for item in ("None", "none yet", "N/A", "TBD", "pass", "not sure yet", "I don't know"):
        assert fc.is_q4_null_like(item), item


def test_a_signed_q4_line_is_bound_to_its_text() -> None:
    merged = fc.promote_interview_q4_invariants(
        {fc.INTERVIEW_Q4_ID: "uncapped liability"},
        posture_version=1,
        signed_by="Legal Owner",
    )
    assert fc.signed_binding_status(merged[0]) == "bound"
    assert fc.signed_binding_status({**merged[0], "x_direction": "hold"}) == "altered"


def test_a_page_edit_of_a_signed_q4_line_drops_the_sign_off() -> None:
    from playbook_engine.overrides import Override, apply_overrides  # noqa: PLC0415

    merged = fc.promote_interview_q4_invariants(
        {fc.INTERVIEW_Q4_ID: "uncapped liability"},
        posture_version=1,
        signed_by="Legal Owner",
    )
    doc = {"floor": {"invariants": merged}}
    edit = Override(target="floor", id=merged[0]["id"], field="statement", value="Owner wording.")
    apply_overrides(doc, [edit])
    line = doc["floor"]["invariants"][0]
    assert line["statement"] == "Owner wording."
    assert fc.signed_binding_status(line) is None
