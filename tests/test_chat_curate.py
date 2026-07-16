"""Tests for chat_curate.py — issue #159.

SECURITY NOTE: All fixtures use synthetic text and fictional party/institution
names only. No real agreement text or real document paths are used.
"""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from playbook_engine.canonicalize import compute_section_digests, content_hash
from playbook_engine.chat_curate import apply_curate_commands
from playbook_engine.cli import cli

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _make_opf(tmp_path: Path, clauses: list[dict] | None = None) -> dict:
    """Build a minimal valid OPF v0.1-shaped dict and write playbook.opf.json.

    Clauses sort by (taxonomy_id, id) in ``_build_index`` — governing_law <
    indemnification — so C1 is always governing_law and C2 indemnification.
    """
    if clauses is None:
        clauses = [
            {
                "id": "clause.indemnification",
                "taxonomy_id": "indemnification",
                "title": "Indemnification",
                "our_standard": None,
                "observed_positions": [],
                "rollup": {"position": "usually_held", "confidence": {"score": 0.6}},
            },
            {
                "id": "clause.governing_law",
                "taxonomy_id": "governing_law",
                "title": "Governing Law",
                "our_standard": None,
                "observed_positions": [],
                "rollup": {"position": "no_signal", "confidence": {"score": 0.1}},
            },
        ]

    doc = {
        "opf_version": "0.1",
        "agreement_type": {"id": "educational-affiliation", "name": "Educational Affiliation"},
        "baseline": {"has_canonical_template": True},
        "taxonomy": {
            "source": "custom",
            "entries": [
                {"id": "indemnification", "label": "Indemnification", "status": "active"},
                {"id": "governing_law", "label": "Governing Law", "status": "active"},
            ],
        },
        "clauses": clauses,
        "corpus": {"documents": [], "stats": {}},
        "compiler": {
            "name": "playbook-engine",
            "version": "0.1.0",
            "generated_at": "2026-01-01T00:00:00Z",
        },
    }
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "playbook.opf.json").write_text(json.dumps(doc), encoding="utf-8")
    return doc


def _load_opf(tmp_path: Path) -> dict:
    return json.loads((tmp_path / "out" / "playbook.opf.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# apply_curate_commands — pin + note
# ---------------------------------------------------------------------------


def test_pin_and_note_write_expected_embedded_state(tmp_path: Path) -> None:
    """A fixture sequence of pin + note instructions produces the expected
    embedded pin (with comment/pinned_by/baseline_stance) and note."""
    _make_opf(tmp_path)
    out_dir = tmp_path / "out"

    commands = [
        "pin governing_law to usually_conceded: keep as filed",
        "note indemnification: check with GC next cycle",
    ]
    result = apply_curate_commands(out_dir, commands, pinned_by="marc")

    assert result.pins_written == 1
    assert result.notes_written == 1
    assert result.conflicts == []
    assert [o.applied for o in result.outcomes] == [True, True]

    doc = _load_opf(tmp_path)
    pins = doc["curation"]["pins"]
    assert len(pins) == 1
    pin = pins[0]
    assert pin["clause_id"] == "clause.governing_law"
    assert pin["position"] == "usually_conceded"
    assert pin["comment"] == "keep as filed"
    assert pin["pinned_by"] == "marc"
    # baseline_stance records what the pin overrides FROM: governing_law's
    # rollup.position ("no_signal") at pin time.
    assert pin["baseline_stance"] == "no_signal"
    assert "conflict" not in pin

    notes_path = out_dir / "viewer_notes.md"
    assert notes_path.exists()
    content = notes_path.read_text(encoding="utf-8")
    assert "check with GC next cycle" in content
    assert "Indemnification" in content


def test_pin_accepts_optional_clause_and_stance_filler_words(tmp_path: Path) -> None:
    """'pin clause X to stance Y' and 'pin X to Y' parse identically."""
    _make_opf(tmp_path)
    out_dir = tmp_path / "out"

    result = apply_curate_commands(out_dir, ["pin clause governing_law to stance usually_conceded"])
    assert result.pins_written == 1
    assert result.outcomes[0].applied is True

    doc = _load_opf(tmp_path)
    assert doc["curation"]["pins"][0]["position"] == "usually_conceded"


def test_clause_reference_resolves_by_item_number_taxonomy_id_or_title(tmp_path: Path) -> None:
    """A clause reference resolves case-insensitively via C-number, id,
    taxonomy_id, or title — all four address the same clause."""
    _make_opf(tmp_path)
    out_dir = tmp_path / "out"

    for ref in ("C1", "clause.governing_law", "governing_law", "Governing Law", "GOVERNING_LAW"):
        result = apply_curate_commands(out_dir, [f"pin {ref} to usually_conceded"])
        assert result.pins_written == 1, f"ref={ref!r} failed to resolve"
        doc = _load_opf(tmp_path)
        assert doc["curation"]["pins"][0]["clause_id"] == "clause.governing_law"


def test_unresolvable_clause_reference_is_reported_not_fatal(tmp_path: Path) -> None:
    """An unresolvable clause reference is reported as unapplied; other
    instructions in the same batch still apply (issue #138 — no false OK,
    no losing the rest of the batch to one typo)."""
    _make_opf(tmp_path)
    out_dir = tmp_path / "out"

    commands = [
        "pin nonexistent_clause to usually_held",
        "pin governing_law to usually_conceded",
    ]
    result = apply_curate_commands(out_dir, commands)

    assert result.pins_written == 1
    bad, good = result.outcomes
    assert bad.applied is False
    assert bad.action == "error"
    assert "nonexistent_clause" in bad.detail
    assert good.applied is True


def test_unparseable_line_is_reported_not_fatal(tmp_path: Path) -> None:
    """A line matching neither grammar form is reported as unapplied, not raised."""
    _make_opf(tmp_path)
    out_dir = tmp_path / "out"

    result = apply_curate_commands(out_dir, ["frobnicate the governing law clause"])
    assert result.pins_written == 0
    assert result.notes_written == 0
    assert result.outcomes[0].applied is False
    assert result.outcomes[0].action == "error"


def test_blank_and_comment_lines_are_skipped(tmp_path: Path) -> None:
    _make_opf(tmp_path)
    out_dir = tmp_path / "out"

    result = apply_curate_commands(
        out_dir,
        ["", "   ", "# a comment line", "pin governing_law to usually_conceded"],
    )
    assert len(result.outcomes) == 1
    assert result.pins_written == 1


def test_missing_playbook_raises_file_not_found(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    try:
        apply_curate_commands(out_dir, ["pin governing_law to usually_conceded"])
        raise AssertionError("expected FileNotFoundError")
    except FileNotFoundError:
        pass


def test_identity_digests_refresh_when_present(tmp_path: Path) -> None:
    """A pin refreshes identity.content_hash/section_digests when the stored
    OPF already carries an identity block, mirroring apply_feedback (#147)."""
    doc = _make_opf(tmp_path)
    out_dir = tmp_path / "out"

    doc["identity"] = {
        "content_hash": content_hash(doc),
        "section_digests": compute_section_digests(doc),
    }
    (out_dir / "playbook.opf.json").write_text(json.dumps(doc), encoding="utf-8")
    hash_before = doc["identity"]["content_hash"]
    curation_digest_before = doc["identity"]["section_digests"]["curation"]

    apply_curate_commands(out_dir, ["pin governing_law to usually_conceded"])

    after = _load_opf(tmp_path)
    assert after["identity"]["content_hash"] == hash_before
    assert after["identity"]["section_digests"]["curation"] != curation_digest_before


# ---------------------------------------------------------------------------
# Conflict detection on recompiled/changed evidence
# ---------------------------------------------------------------------------


def test_pin_conflicting_with_new_evidence_is_reported(tmp_path: Path) -> None:
    """A pin made against one historical_stance, whose clause's stance later
    moves (simulating a recompile / fresh evidence), is flagged as a
    conflict the next time curate runs — even via an unrelated instruction."""
    _make_opf(tmp_path)
    out_dir = tmp_path / "out"

    first = apply_curate_commands(out_dir, ["pin governing_law to usually_conceded"])
    assert first.conflicts == []
    doc = _load_opf(tmp_path)
    assert doc["curation"]["pins"][0]["baseline_stance"] == "no_signal"
    assert (
        "conflict" not in doc["curation"]["pins"][0]
        or doc["curation"]["pins"][0]["conflict"] is None
    )

    # Simulate new evidence: the clause's recomputed stance has since moved
    # (e.g. via a pipeline recompile) without touching curation.pins.
    doc["clauses"][1]["rollup"]["position"] = "usually_held"
    assert doc["clauses"][1]["id"] == "clause.governing_law"
    (out_dir / "playbook.opf.json").write_text(json.dumps(doc), encoding="utf-8")

    second = apply_curate_commands(out_dir, ["note indemnification: unrelated instruction"])

    assert len(second.conflicts) == 1
    conflict = second.conflicts[0]
    assert conflict.clause_id == "clause.governing_law"
    assert conflict.action == "conflict"
    assert "no_signal" in conflict.detail
    assert "usually_held" in conflict.detail

    after = _load_opf(tmp_path)
    persisted_pin = after["curation"]["pins"][0]
    assert persisted_pin["conflict"]["recomputed_historical_stance"] == "usually_held"


def test_repinning_a_conflicted_clause_clears_the_conflict(tmp_path: Path) -> None:
    """Re-pinning a clause resets baseline_stance to the CURRENT stance, so
    the just-created pin is never itself in conflict."""
    _make_opf(tmp_path)
    out_dir = tmp_path / "out"

    apply_curate_commands(out_dir, ["pin governing_law to usually_conceded"])
    doc = _load_opf(tmp_path)
    doc["clauses"][1]["rollup"]["position"] = "usually_held"
    (out_dir / "playbook.opf.json").write_text(json.dumps(doc), encoding="utf-8")

    # Confirm the conflict exists before re-pinning.
    pre = apply_curate_commands(out_dir, ["note indemnification: noop"])
    assert len(pre.conflicts) == 1

    # Re-pin against the now-current stance.
    result = apply_curate_commands(out_dir, ["pin governing_law to usually_conceded"])
    assert result.conflicts == []

    after = _load_opf(tmp_path)
    pin = after["curation"]["pins"][0]
    assert pin["baseline_stance"] == "usually_held"
    assert pin.get("conflict") is None


# ---------------------------------------------------------------------------
# CLI — playbook curate
# ---------------------------------------------------------------------------


def test_curate_cmd_success(tmp_path: Path) -> None:
    _make_opf(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["curate", str(tmp_path / "out"), "--command", "pin governing_law to usually_conceded"],
    )
    assert result.exit_code == 0, result.output
    assert "OK  pin" in result.output
    assert "applied 1 pin(s), 0 note(s)" in result.output


def test_curate_cmd_from_file(tmp_path: Path) -> None:
    _make_opf(tmp_path)
    commands_file = tmp_path / "commands.txt"
    commands_file.write_text(
        "pin governing_law to usually_conceded\nnote indemnification: check next cycle\n",
        encoding="utf-8",
    )
    runner = CliRunner()
    result = runner.invoke(cli, ["curate", str(tmp_path / "out"), "--file", str(commands_file)])
    assert result.exit_code == 0, result.output
    assert "applied 1 pin(s), 1 note(s)" in result.output


def test_curate_cmd_missing_opf_exits_nonzero(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    runner = CliRunner()
    result = runner.invoke(
        cli, ["curate", str(out_dir), "--command", "pin governing_law to usually_conceded"]
    )
    assert result.exit_code != 0


def test_curate_cmd_no_instructions_exits_nonzero(tmp_path: Path) -> None:
    _make_opf(tmp_path)
    runner = CliRunner()
    result = runner.invoke(cli, ["curate", str(tmp_path / "out")])
    assert result.exit_code != 0


def test_curate_cmd_reports_conflict(tmp_path: Path) -> None:
    _make_opf(tmp_path)
    out_dir = tmp_path / "out"
    runner = CliRunner()

    runner.invoke(
        cli, ["curate", str(out_dir), "--command", "pin governing_law to usually_conceded"]
    )

    doc = _load_opf(tmp_path)
    doc["clauses"][1]["rollup"]["position"] = "usually_held"
    (out_dir / "playbook.opf.json").write_text(json.dumps(doc), encoding="utf-8")

    result = runner.invoke(
        cli, ["curate", str(out_dir), "--command", "note indemnification: unrelated"]
    )
    assert result.exit_code == 0, result.output
    assert "CONFLICT" in result.output
    assert "conflict(s) flagged" in result.output
