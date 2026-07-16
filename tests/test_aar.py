"""Tests for the after-action report (aar.py).

SECURITY NOTE: All fixtures are programmatically constructed with synthetic text.
No real agreement files are committed or referenced.  Fictional party names only.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from playbook_engine.aar import (
    build_after_action_data,
    build_after_action_report,
    write_after_action_report,
)
from playbook_engine.cli import cli

# ---------------------------------------------------------------------------
# Fixture helpers — mirrors test_inspection_report.py conventions
# ---------------------------------------------------------------------------


def _write_trail(out_dir: Path, doc_id: str, **kwargs: object) -> None:
    trail_dir = out_dir / "trail"
    trail_dir.mkdir(parents=True, exist_ok=True)
    data = {"document_id": doc_id, **kwargs}
    (trail_dir / f"{doc_id}.json").write_text(json.dumps(data), encoding="utf-8")


def _write_scope(out_dir: Path, docs: list[dict]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "scope.json").write_text(json.dumps({"documents": docs}), encoding="utf-8")


def _write_observations(out_dir: Path, obs: list[dict]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = "\n".join(json.dumps(o) for o in obs)
    (out_dir / "observations.jsonl").write_text(lines, encoding="utf-8")


def _write_playbook(out_dir: Path, playbook: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "playbook.opf.json").write_text(json.dumps(playbook), encoding="utf-8")


def _make_obs(
    doc_id: str,
    version: str,
    text: str,
    tid: str | None = "TERM",
    deviation: str = "none",
    direction: str = "neutral",
    magnitude: str = "none",
    outcome: str = "signed",
    provenance: str = "our_paper",
) -> dict:
    return {
        "observation_id": f"{doc_id}/{version}/1",
        "taxonomy_id": tid,
        "text_summary": text,
        "citation": {
            "document_id": doc_id,
            "version": version,
            "clause_path": "1",
            "char_span": None,
        },
        "deviation": deviation,
        "risk_delta": {"direction": direction, "magnitude": magnitude},
        "provenance": provenance,
        "outcome": outcome,
    }


def _make_minimal_playbook(generated_at: str = "2026-01-15T09:00:00Z") -> dict:
    """Build a minimal valid-enough playbook dict for testing."""
    return {
        "opf_version": "0.1",
        "agreement_type": {"id": "test", "name": "Test Agreement"},
        "baseline": {"has_canonical_template": False},
        "taxonomy": {"source": "test", "entries": []},
        "clauses": [
            {
                "id": "clause.term",
                "taxonomy_id": "TERM",
                "title": "Term",
                "observed_positions": [],
                "rollup": {
                    "position": "standard",
                    "confidence": {
                        "score": 0.9,
                        "basis": "test",
                        "n_our_paper": 5,
                        "n_counterparty_paper": 0,
                    },
                },
            },
            {
                "id": "clause.governing_law",
                "taxonomy_id": "GOVERNING_LAW",
                "title": "Governing Law",
                "observed_positions": [],
                "rollup": {
                    "position": "negotiable",
                    "confidence": {
                        "score": 0.4,
                        "basis": "test",
                        "n_our_paper": 2,
                        "n_counterparty_paper": 1,
                    },
                },
            },
        ],
        "corpus": {
            "documents": [
                {
                    "document_id": "deal-alice",
                    "provenance": "our_paper",
                    "in_scope": True,
                    "scope_rationale": "In scope for testing.",
                    "versions": 2,
                }
            ],
            "stats": {},
        },
        "compiler": {
            "name": "playbook-engine",
            "version": "0.1.0",
            "generated_at": generated_at,
        },
    }


def _make_out_dir(tmp_path: Path) -> Path:
    """Build a minimal synthetic out/ directory with two documents."""
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    _write_trail(
        out_dir,
        "deal-alice",
        ordered_versions=["v1", "v2"],
        signed_version="v2",
        provenance="our_paper",
        basis="greedy",
    )
    _write_trail(
        out_dir,
        "deal-bob",
        ordered_versions=["v1"],
        signed_version="v1",
        provenance="counterparty_paper",
        basis="single",
    )
    _write_scope(
        out_dir,
        [
            {
                "document_id": "deal-alice",
                "in_scope": True,
                "scope_rationale": "Accepted without LLM judgment (stub mode).",
                "scope_confidence": 0.5,
            },
            {
                "document_id": "deal-bob",
                "in_scope": True,
                "scope_rationale": "Accepted without LLM judgment (stub mode).",
                "scope_confidence": 0.5,
            },
        ],
    )
    _write_observations(
        out_dir,
        [
            _make_obs("deal-alice", "v2", "Alice Corp shall indemnify Beta University."),
            _make_obs(
                "deal-alice", "v2", "Governing law: State of California.", tid="GOVERNING_LAW"
            ),
            _make_obs("deal-bob", "v1", "Beta University shall not be liable.", tid=None),
        ],
    )
    _write_playbook(out_dir, _make_minimal_playbook())
    return out_dir


# ---------------------------------------------------------------------------
# Section heading presence tests (acceptance criteria)
# ---------------------------------------------------------------------------


def test_report_contains_corpus_coverage_heading(tmp_path: Path) -> None:
    out_dir = _make_out_dir(tmp_path)
    report = build_after_action_report(out_dir)
    assert "## Corpus Coverage" in report


def test_report_contains_backbone_health_heading(tmp_path: Path) -> None:
    out_dir = _make_out_dir(tmp_path)
    report = build_after_action_report(out_dir)
    assert "## Backbone Health" in report


def test_report_contains_judgment_economics_heading(tmp_path: Path) -> None:
    out_dir = _make_out_dir(tmp_path)
    report = build_after_action_report(out_dir)
    assert "## Judgment Economics" in report


def test_report_contains_semantic_coverage_heading(tmp_path: Path) -> None:
    out_dir = _make_out_dir(tmp_path)
    report = build_after_action_report(out_dir)
    assert "## Semantic Coverage" in report


def test_report_contains_needs_attention_heading(tmp_path: Path) -> None:
    out_dir = _make_out_dir(tmp_path)
    report = build_after_action_report(out_dir)
    assert "## Needs Attention" in report


def test_report_contains_honesty_heading(tmp_path: Path) -> None:
    out_dir = _make_out_dir(tmp_path)
    report = build_after_action_report(out_dir)
    assert "## Honesty" in report


# ---------------------------------------------------------------------------
# Corpus coverage accuracy
# ---------------------------------------------------------------------------


def test_corpus_coverage_in_scope_count(tmp_path: Path) -> None:
    """Acceptance: counts in-scope correctly from scope.json."""
    out_dir = _make_out_dir(tmp_path)
    report = build_after_action_report(out_dir)
    assert "2 in scope" in report


def test_corpus_coverage_out_of_scope_shown(tmp_path: Path) -> None:
    """Out-of-scope documents appear with rationale."""
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    _write_trail(out_dir, "deal-alice", ordered_versions=["v1"], provenance="our_paper")
    _write_scope(
        out_dir,
        [
            {"document_id": "deal-alice", "in_scope": True, "scope_rationale": "Looks good."},
            {
                "document_id": "deal-eve",
                "in_scope": False,
                "scope_rationale": "Not an affiliation agreement.",
            },
        ],
    )
    report = build_after_action_report(out_dir)
    assert "Not an affiliation agreement." in report


# ---------------------------------------------------------------------------
# Classification coverage % (acceptance criterion)
# ---------------------------------------------------------------------------


def test_classification_coverage_pct_correct(tmp_path: Path) -> None:
    """Acceptance: classification % is computed correctly from observations.jsonl."""
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    _write_trail(out_dir, "deal-alice", ordered_versions=["v1"], provenance="our_paper")
    _write_scope(out_dir, [{"document_id": "deal-alice", "in_scope": True}])
    # 3 classified, 1 unclassified = 75%
    _write_observations(
        out_dir,
        [
            _make_obs("deal-alice", "v1", "Term A.", tid="TERM"),
            _make_obs("deal-alice", "v1", "Term B.", tid="GOVERNING_LAW"),
            _make_obs("deal-alice", "v1", "Term C.", tid="INDEMNIFICATION"),
            _make_obs("deal-alice", "v1", "Unclassified clause.", tid=None),
        ],
    )
    data = build_after_action_data(out_dir)
    assert data["semantic_coverage"]["classification_pct"] == 75.0
    assert data["semantic_coverage"]["classified_count"] == 3
    assert data["semantic_coverage"]["unclassified_count"] == 1


def test_classification_coverage_all_classified(tmp_path: Path) -> None:
    """100% coverage when all observations have taxonomy_id."""
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    _write_trail(out_dir, "deal-alice", ordered_versions=["v1"], provenance="our_paper")
    _write_scope(out_dir, [{"document_id": "deal-alice", "in_scope": True}])
    _write_observations(
        out_dir,
        [
            _make_obs("deal-alice", "v1", "Term A.", tid="TERM"),
            _make_obs("deal-alice", "v1", "Term B.", tid="GOVERNING_LAW"),
        ],
    )
    data = build_after_action_data(out_dir)
    assert data["semantic_coverage"]["classification_pct"] == 100.0


def test_classification_coverage_none_classified(tmp_path: Path) -> None:
    """0% when all observations are unclassified."""
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    _write_trail(out_dir, "deal-alice", ordered_versions=["v1"], provenance="our_paper")
    _write_scope(out_dir, [{"document_id": "deal-alice", "in_scope": True}])
    _write_observations(
        out_dir,
        [_make_obs("deal-alice", "v1", "Unclassified.", tid=None)],
    )
    data = build_after_action_data(out_dir)
    assert data["semantic_coverage"]["classification_pct"] == 0.0


def test_classification_coverage_no_observations(tmp_path: Path) -> None:
    """No observations → 0% (no division by zero)."""
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    _write_trail(out_dir, "deal-alice", ordered_versions=["v1"], provenance="our_paper")
    _write_scope(out_dir, [{"document_id": "deal-alice", "in_scope": True}])
    # No observations.jsonl
    data = build_after_action_data(out_dir)
    assert data["semantic_coverage"]["classification_pct"] == 0.0
    assert data["semantic_coverage"]["total_observations"] == 0


# ---------------------------------------------------------------------------
# Needs attention and Honesty sections
# ---------------------------------------------------------------------------


def test_needs_review_obs_appears_in_needs_attention(tmp_path: Path) -> None:
    """Acceptance: needs_review items are enumerated in Needs-attention section."""
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    _write_trail(out_dir, "deal-alice", ordered_versions=["v1"], provenance="our_paper")
    _write_scope(out_dir, [{"document_id": "deal-alice", "in_scope": True}])
    _write_observations(
        out_dir,
        [
            _make_obs("deal-alice", "v1", "Normal clause.", deviation="none"),
            _make_obs("deal-alice", "v1", "Needs review clause.", deviation="needs_review"),
        ],
    )
    report = build_after_action_report(out_dir)
    assert "needs_review" in report


def test_needs_attention_data_item_numbers(tmp_path: Path) -> None:
    """Needs-attention items have sequential item_number."""
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    _write_trail(out_dir, "deal-alice", ordered_versions=["v1"], provenance="our_paper")
    _write_scope(out_dir, [{"document_id": "deal-alice", "in_scope": True}])
    _write_observations(
        out_dir,
        [
            _make_obs("deal-alice", "v1", "A.", deviation="needs_review"),
            _make_obs("deal-alice", "v1", "B.", deviation="needs_review"),
        ],
    )
    data = build_after_action_data(out_dir)
    item_numbers = [item["item_number"] for item in data["needs_attention"]]
    assert item_numbers == [1, 2]


def test_honesty_section_lists_blank_fields(tmp_path: Path) -> None:
    """Acceptance: blank/defaulted fields are enumerated in the Honesty section."""
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    _write_trail(out_dir, "deal-alice", ordered_versions=["v1"], provenance="our_paper")
    _write_scope(out_dir, [{"document_id": "deal-alice", "in_scope": True}])
    _write_observations(out_dir, [_make_obs("deal-alice", "v1", "Clause.")])
    # Playbook with a clause that has no our_standard
    playbook = _make_minimal_playbook()
    playbook["clauses"][0]["our_standard"] = None
    _write_playbook(out_dir, playbook)
    report = build_after_action_report(out_dir)
    assert "our_standard" in report


def test_honesty_flags_zero_clause_playbook(tmp_path: Path) -> None:
    """Issue #208: a schema-valid but ZERO-clause playbook must be flagged as
    a blank/defaulted field — previously every honesty check was per-clause,
    so the emptiest possible playbook produced "No blank or defaulted fields
    detected"."""
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    _write_trail(out_dir, "deal-alice", ordered_versions=["v1"], provenance="our_paper")
    _write_scope(out_dir, [{"document_id": "deal-alice", "in_scope": True}])
    # Every observation unsigned → the position compiler withholds all of
    # them → zero clauses; the report must name that likely cause.
    _write_observations(out_dir, [_make_obs("deal-alice", "v1", "Clause.", outcome="unsigned")])
    playbook = _make_minimal_playbook()
    playbook["clauses"] = []
    _write_playbook(out_dir, playbook)

    report = build_after_action_report(out_dir)

    assert "ZERO clause positions" in report
    assert "unsigned" in report
    assert "No blank or defaulted fields detected" not in report


def test_honesty_section_notes_present(tmp_path: Path) -> None:
    """Honesty notes are always present."""
    out_dir = _make_out_dir(tmp_path)
    report = build_after_action_report(out_dir)
    assert "GC-authored Posture" in report
    assert "Floor" in report


def test_honesty_flags_under_grounded_standard_position(tmp_path: Path) -> None:
    """Issue #107: "standard"/"acceptable_variants_exist" positions built on
    too few our-paper citations are the more dangerous under-grounding case
    (they read as settled guidance) — must be flagged, not just
    negotiable/hold_firm."""
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    _write_trail(out_dir, "deal-alice", ordered_versions=["v1"], provenance="our_paper")
    _write_scope(out_dir, [{"document_id": "deal-alice", "in_scope": True}])
    _write_observations(out_dir, [_make_obs("deal-alice", "v1", "Clause.")])
    playbook = _make_minimal_playbook()
    # clause.term is position="standard" with n_our_paper=5 by default (safe);
    # override to a single-citation "standard" — exactly the under-grounded
    # case the AAR check must now catch.
    playbook["clauses"][0]["rollup"]["position"] = "standard"
    playbook["clauses"][0]["rollup"]["confidence"]["n_our_paper"] = 1
    _write_playbook(out_dir, playbook)
    data = build_after_action_data(out_dir)
    human_required = data["honesty"]["human_input_required"]
    flagged = [item for item in human_required if item["clause_id"] == "clause.term"]
    assert len(flagged) == 1
    assert flagged[0]["position"] == "standard"


def test_honesty_reversal_count_in_data(tmp_path: Path) -> None:
    """Reversal observations are counted correctly."""
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    _write_trail(out_dir, "deal-alice", ordered_versions=["v1"], provenance="our_paper")
    _write_scope(out_dir, [{"document_id": "deal-alice", "in_scope": True}])
    _write_observations(
        out_dir,
        [
            _make_obs("deal-alice", "v1", "Signed clause.", outcome="signed"),
            _make_obs("deal-alice", "v1", "Reversed clause.", outcome="proposed_then_reversed"),
        ],
    )
    data = build_after_action_data(out_dir)
    assert data["honesty"]["reversal_observation_count"] == 1


# ---------------------------------------------------------------------------
# Determinism — no wall-clock variance
# ---------------------------------------------------------------------------


def test_report_is_deterministic(tmp_path: Path) -> None:
    """Acceptance: same out_dir produces identical report on two calls."""
    out_dir = _make_out_dir(tmp_path)
    report1 = build_after_action_report(out_dir)
    report2 = build_after_action_report(out_dir)
    assert report1 == report2


def test_data_timestamp_from_playbook(tmp_path: Path) -> None:
    """generated_at is derived from compiler.generated_at, not wall clock."""
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    _write_trail(out_dir, "deal-alice", ordered_versions=["v1"], provenance="our_paper")
    _write_scope(out_dir, [{"document_id": "deal-alice", "in_scope": True}])
    _write_observations(out_dir, [])
    fixed_ts = "2026-01-15T09:00:00Z"
    _write_playbook(out_dir, _make_minimal_playbook(generated_at=fixed_ts))
    data = build_after_action_data(out_dir)
    assert data["generated_at"] == fixed_ts


# ---------------------------------------------------------------------------
# write_after_action_report — --out writes both .md and .json
# ---------------------------------------------------------------------------


def test_write_after_action_report_creates_md(tmp_path: Path) -> None:
    """write_after_action_report creates the .md file."""
    out_dir = _make_out_dir(tmp_path)
    dest = tmp_path / "report.md"
    write_after_action_report(out_dir, dest)
    assert dest.exists()
    content = dest.read_text(encoding="utf-8")
    assert "# Playbook After-Action Report" in content


def test_write_after_action_report_creates_json_twin(tmp_path: Path) -> None:
    """Acceptance: --out also writes report.json alongside."""
    out_dir = _make_out_dir(tmp_path)
    dest = tmp_path / "report.md"
    write_after_action_report(out_dir, dest)
    json_path = tmp_path / "report.json"
    assert json_path.exists()
    data = json.loads(json_path.read_text(encoding="utf-8"))
    # Must have all six sections
    assert "corpus_coverage" in data
    assert "backbone_health" in data
    assert "judgment_economics" in data
    assert "semantic_coverage" in data
    assert "needs_attention" in data
    assert "honesty" in data


def test_write_after_action_report_atomic(tmp_path: Path) -> None:
    """No .tmp file left behind after a successful write."""
    out_dir = _make_out_dir(tmp_path)
    dest = tmp_path / "report.md"
    write_after_action_report(out_dir, dest)
    assert not dest.with_suffix(".tmp").exists()
    assert not (tmp_path / "report.json").with_suffix(".tmp").exists()


# ---------------------------------------------------------------------------
# build_after_action_data — section structure tests
# ---------------------------------------------------------------------------


def test_data_has_all_six_sections(tmp_path: Path) -> None:
    """build_after_action_data returns all six sections."""
    out_dir = _make_out_dir(tmp_path)
    data = build_after_action_data(out_dir)
    assert "corpus_coverage" in data
    assert "backbone_health" in data
    assert "judgment_economics" in data
    assert "semantic_coverage" in data
    assert "needs_attention" in data
    assert "honesty" in data


def test_data_corpus_coverage_counts(tmp_path: Path) -> None:
    """corpus_coverage counts match fixture (2 in-scope, 0 out-of-scope)."""
    out_dir = _make_out_dir(tmp_path)
    data = build_after_action_data(out_dir)
    cc = data["corpus_coverage"]
    assert cc["in_scope_count"] == 2
    assert cc["out_of_scope_count"] == 0
    assert cc["total_documents"] == 2


def test_data_backbone_health_counts(tmp_path: Path) -> None:
    """backbone_health counts match fixture (2 ordered, 2 signed)."""
    out_dir = _make_out_dir(tmp_path)
    data = build_after_action_data(out_dir)
    bh = data["backbone_health"]
    assert bh["total_trails"] == 2
    assert bh["ordered_count"] == 2
    assert bh["signed_count"] == 2


def test_data_backbone_health_reversal_count_nonzero(tmp_path: Path) -> None:
    """Issue #106: trail["reversals"] populated → non-zero backbone reversal count.

    Previously pipeline.py never wrote a "reversals" key into the trail dict
    at all, so aar._build_backbone_health's `trail.get("reversals", [])`
    always fell back to empty — reversal_count was permanently 0 even when
    detect_reversals found genuine reversals.
    """
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    _write_trail(
        out_dir,
        "deal-alice",
        ordered_versions=["v1", "v2"],
        signed_version="v2",
        reversals=[
            {
                "taxonomy_id": "ind",
                "clause_path": "1",
                "version_inserted": "v1",
                "version_removed": "v2",
                "proposed_text": "rejected ask",
            }
        ],
    )
    _write_scope(out_dir, [{"document_id": "deal-alice", "in_scope": True}])
    _write_observations(
        out_dir,
        [_make_obs("deal-alice", "v2", "Some clause.", outcome="proposed_then_reversed")],
    )
    data = build_after_action_data(out_dir)
    bh = data["backbone_health"]
    assert bh["reversal_count"] == 1
    assert bh["trails"][0]["reversals"]


def test_data_rollup_position_histogram(tmp_path: Path) -> None:
    """Rollup position histogram is built from the playbook clauses."""
    out_dir = _make_out_dir(tmp_path)
    data = build_after_action_data(out_dir)
    hist = data["semantic_coverage"]["rollup_position_histogram"]
    # Fixture playbook has 1 standard, 1 negotiable
    assert hist.get("standard", 0) == 1
    assert hist.get("negotiable", 0) == 1


def test_data_deviation_distribution(tmp_path: Path) -> None:
    """deviation_distribution is built correctly from observations."""
    out_dir = _make_out_dir(tmp_path)
    data = build_after_action_data(out_dir)
    dev_dist = data["semantic_coverage"]["deviation_distribution"]
    # All 3 fixture obs have deviation="none"
    assert dev_dist.get("none", 0) == 3


def test_data_provenance_distribution(tmp_path: Path) -> None:
    """provenance_distribution is built from observations."""
    out_dir = _make_out_dir(tmp_path)
    data = build_after_action_data(out_dir)
    prov_dist = data["semantic_coverage"]["provenance_distribution"]
    assert prov_dist.get("our_paper", 0) == 3


def test_missing_out_dir_raises(tmp_path: Path) -> None:
    """FileNotFoundError raised for non-existent out_dir."""
    with pytest.raises(FileNotFoundError):
        build_after_action_data(tmp_path / "nonexistent")


def test_report_missing_out_dir_raises(tmp_path: Path) -> None:
    """build_after_action_report raises FileNotFoundError for missing dir."""
    with pytest.raises(FileNotFoundError):
        build_after_action_report(tmp_path / "nonexistent")


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_no_playbook_still_renders(tmp_path: Path) -> None:
    """Report renders even without playbook.opf.json."""
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    _write_trail(out_dir, "deal-alice", ordered_versions=["v1"], provenance="our_paper")
    _write_scope(out_dir, [{"document_id": "deal-alice", "in_scope": True}])
    _write_observations(out_dir, [_make_obs("deal-alice", "v1", "A clause.")])
    report = build_after_action_report(out_dir)
    assert "## Corpus Coverage" in report
    assert "## Semantic Coverage" in report


def test_no_observations_still_renders(tmp_path: Path) -> None:
    """Report renders even without observations.jsonl."""
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    _write_trail(out_dir, "deal-alice", ordered_versions=["v1"], provenance="our_paper")
    _write_scope(out_dir, [{"document_id": "deal-alice", "in_scope": True}])
    report = build_after_action_report(out_dir)
    assert "## Corpus Coverage" in report


def test_judgment_economics_no_judge_dir(tmp_path: Path) -> None:
    """Judgment economics section shows stub-mode message when no judge/ dir."""
    out_dir = _make_out_dir(tmp_path)
    report = build_after_action_report(out_dir)
    # No judge/ dir in fixture → stub-judge mode message
    assert "stub-judge mode" in report or "No `judge/` directory" in report


def test_judgment_economics_with_judge_dir(tmp_path: Path) -> None:
    """Judgment economics reads verdicts and pending counts from judge/ dir."""
    out_dir = _make_out_dir(tmp_path)
    judge_dir = out_dir / "judge"
    judge_dir.mkdir()
    # Write 2 verdicts
    verdicts_path = judge_dir / "verdicts.jsonl"
    verdicts_path.write_text(
        json.dumps({"key": "abc", "verdict": {}})
        + "\n"
        + json.dumps({"key": "def", "verdict": {}})
        + "\n",
        encoding="utf-8",
    )
    # Write 1 pending
    pending_path = judge_dir / "pending.jsonl"
    pending_path.write_text(
        json.dumps({"key": "xyz", "kind": "classify", "payload": {}}) + "\n",
        encoding="utf-8",
    )
    data = build_after_action_data(out_dir)
    je = data["judgment_economics"]
    assert je["verdicts_in_store"] == 2
    assert je["pending_count"] == 1
    assert je["pending_by_kind"].get("classify", 0) == 1


# ---------------------------------------------------------------------------
# CLI: playbook report
# ---------------------------------------------------------------------------


def test_report_cmd_stdout(tmp_path: Path) -> None:
    """report with no --out prints to stdout and exits 0."""
    out_dir = _make_out_dir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(cli, ["report", str(out_dir)])
    assert result.exit_code == 0, result.output
    assert "# Playbook After-Action Report" in result.output


def test_report_cmd_writes_md_and_json(tmp_path: Path) -> None:
    """Acceptance: report --out report.md writes both .md and .json."""
    out_dir = _make_out_dir(tmp_path)
    report_path = tmp_path / "report.md"
    runner = CliRunner()
    result = runner.invoke(cli, ["report", str(out_dir), "--out", str(report_path)])
    assert result.exit_code == 0, result.output
    assert report_path.exists()
    json_path = tmp_path / "report.json"
    assert json_path.exists()
    content = report_path.read_text(encoding="utf-8")
    assert "# Playbook After-Action Report" in content


def test_report_cmd_missing_dir_exits_nonzero(tmp_path: Path) -> None:
    """report <nonexistent> exits non-zero with ERROR message."""
    runner = CliRunner()
    result = runner.invoke(cli, ["report", str(tmp_path / "no-such-dir")])
    assert result.exit_code != 0
    assert "ERROR" in result.output
