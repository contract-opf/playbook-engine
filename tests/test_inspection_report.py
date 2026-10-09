"""Tests for inspection_report.py.

SECURITY NOTE: All fixtures are programmatically constructed with synthetic
text.  No real agreement files are committed or referenced.  Fictional party
and author names only.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from playbook_engine.cli import cli
from playbook_engine.inspection_report import (
    build_inspection_report,
    render_review_flags,
    write_inspection_report,
)

# ---------------------------------------------------------------------------
# Fixture helpers
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


def _make_obs(
    doc_id: str,
    version: str,
    text: str,
    tid: str | None = "TERM",
    deviation: str = "none",
    direction: str = "neutral",
    magnitude: str = "none",
    outcome: str = "signed",
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
        "provenance": "our_paper",
        "outcome": outcome,
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
    return out_dir


# ---------------------------------------------------------------------------
# build_inspection_report: content tests
# ---------------------------------------------------------------------------


def test_report_contains_header(tmp_path: Path) -> None:
    out_dir = _make_out_dir(tmp_path)
    report = build_inspection_report(out_dir)
    assert "# Playbook Inspection Report" in report


def test_report_lists_document_counts(tmp_path: Path) -> None:
    out_dir = _make_out_dir(tmp_path)
    report = build_inspection_report(out_dir)
    assert "2 in scope / 2 total" in report


def test_report_hints_banner_does_not_prescribe_no_cache(tmp_path: Path) -> None:
    """Issue #197: hints.yaml is hashed into the per-document cache key (see
    artifact_store.make_doc_key), so a plain `playbook mine` re-run already
    picks up a hints.yaml edit for the affected document — the banner must
    not send the reader to `--no-cache` (which forces a full corpus
    re-extraction) for this case.
    """
    out_dir = _make_out_dir(tmp_path)
    report = build_inspection_report(out_dir)
    assert "re-run `playbook mine`" in report
    assert "--no-cache" not in report


def test_report_observation_count(tmp_path: Path) -> None:
    out_dir = _make_out_dir(tmp_path)
    report = build_inspection_report(out_dir)
    assert "3 total" in report  # 2 deal-alice + 1 deal-bob


def test_report_contains_both_doc_sections(tmp_path: Path) -> None:
    out_dir = _make_out_dir(tmp_path)
    report = build_inspection_report(out_dir)
    assert "## deal-alice" in report
    assert "## deal-bob" in report


def test_report_shows_provenance(tmp_path: Path) -> None:
    out_dir = _make_out_dir(tmp_path)
    report = build_inspection_report(out_dir)
    assert "our_paper" in report
    assert "counterparty_paper" in report


def test_report_shows_version_order(tmp_path: Path) -> None:
    out_dir = _make_out_dir(tmp_path)
    report = build_inspection_report(out_dir)
    assert "v1 → v2" in report


def test_report_shows_signed_copy(tmp_path: Path) -> None:
    out_dir = _make_out_dir(tmp_path)
    report = build_inspection_report(out_dir)
    assert "v2" in report  # signed copy for deal-alice


def test_report_shows_scope_rationale(tmp_path: Path) -> None:
    out_dir = _make_out_dir(tmp_path)
    report = build_inspection_report(out_dir)
    assert "Accepted without LLM judgment" in report


def test_report_shows_taxonomy_ids(tmp_path: Path) -> None:
    out_dir = _make_out_dir(tmp_path)
    report = build_inspection_report(out_dir)
    assert "TERM" in report
    assert "GOVERNING_LAW" in report


def test_report_shows_unclassified_section(tmp_path: Path) -> None:
    out_dir = _make_out_dir(tmp_path)
    report = build_inspection_report(out_dir)
    assert "Unclassified" in report


def test_report_truncates_long_text(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    _write_trail(
        out_dir,
        "deal-alice",
        ordered_versions=["v1"],
        signed_version="v1",
        provenance="our_paper",
        basis="single",
    )
    _write_scope(out_dir, [{"document_id": "deal-alice", "in_scope": True}])
    long_text = "X" * 200
    _write_observations(out_dir, [_make_obs("deal-alice", "v1", long_text)])
    report = build_inspection_report(out_dir)
    # Table cells should not contain the full 200-char string
    assert "X" * 200 not in report
    assert "…" in report


def test_report_escapes_pipe_in_text(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    _write_trail(
        out_dir, "deal-alice", ordered_versions=["v1"], signed_version="v1", provenance="our_paper"
    )
    _write_scope(out_dir, [{"document_id": "deal-alice", "in_scope": True}])
    _write_observations(out_dir, [_make_obs("deal-alice", "v1", "clause | with | pipes")])
    report = build_inspection_report(out_dir)
    assert "clause \\| with \\| pipes" in report


# ---------------------------------------------------------------------------
# build_inspection_report: edge cases
# ---------------------------------------------------------------------------


def test_missing_out_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        build_inspection_report(tmp_path / "nonexistent")


def test_no_scope_json_still_renders(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    _write_trail(
        out_dir, "deal-alice", ordered_versions=["v1"], signed_version="v1", provenance="our_paper"
    )
    # No scope.json — should still work
    report = build_inspection_report(out_dir)
    assert "## deal-alice" in report


def test_no_observations_jsonl_still_renders(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    _write_trail(
        out_dir, "deal-alice", ordered_versions=["v1"], signed_version="v1", provenance="our_paper"
    )
    _write_scope(out_dir, [{"document_id": "deal-alice", "in_scope": True}])
    # No observations.jsonl
    report = build_inspection_report(out_dir)
    assert "No observations" in report


def test_empty_trail_dir_produces_empty_body(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "trail").mkdir()
    report = build_inspection_report(out_dir)
    assert "# Playbook Inspection Report" in report
    assert "## " not in report  # no document sections


def test_out_of_scope_doc_shows_marker(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    _write_trail(
        out_dir,
        "deal-alice",
        ordered_versions=["v1"],
        signed_version="v1",
        provenance="counterparty_paper",
    )
    _write_scope(out_dir, [{"document_id": "deal-alice", "in_scope": False}])
    report = build_inspection_report(out_dir)
    assert "out of scope" in report


def test_out_of_scope_doc_appears_in_report_even_without_trail(tmp_path: Path) -> None:
    """B1 fix: out-of-scope docs have no trail/ file but must still appear."""
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    # deal-alice is in-scope and has a trail
    _write_trail(
        out_dir, "deal-alice", ordered_versions=["v1"], signed_version="v1", provenance="our_paper"
    )
    # deal-eve is out-of-scope — pipeline never writes trail/deal-eve.json
    _write_scope(
        out_dir,
        [
            {
                "document_id": "deal-alice",
                "in_scope": True,
                "scope_rationale": "Looks good.",
                "scope_confidence": 0.9,
            },
            {
                "document_id": "deal-eve",
                "in_scope": False,
                "scope_rationale": "Not an affiliation agreement.",
                "scope_confidence": 0.8,
            },
        ],
    )
    report = build_inspection_report(out_dir)
    assert "## deal-alice" in report
    assert "## deal-eve" in report
    assert "Not an affiliation agreement." in report


def test_document_total_uses_scope_count(tmp_path: Path) -> None:
    """A2 fix: total includes out-of-scope docs, not just trail docs."""
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    _write_trail(
        out_dir, "deal-alice", ordered_versions=["v1"], signed_version="v1", provenance="our_paper"
    )
    _write_scope(
        out_dir,
        [
            {"document_id": "deal-alice", "in_scope": True},
            {"document_id": "deal-eve", "in_scope": False},
        ],
    )
    report = build_inspection_report(out_dir)
    assert "1 in scope / 2 total" in report


def test_observation_table_includes_version_column(tmp_path: Path) -> None:
    """A4 fix: observation table has a Version column for traceability."""
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    _write_trail(
        out_dir,
        "deal-alice",
        ordered_versions=["v1", "v2"],
        signed_version="v2",
        provenance="our_paper",
    )
    _write_scope(out_dir, [{"document_id": "deal-alice", "in_scope": True}])
    _write_observations(
        out_dir,
        [
            _make_obs("deal-alice", "v1", "Draft clause text."),
            _make_obs("deal-alice", "v2", "Final clause text."),
        ],
    )
    report = build_inspection_report(out_dir)
    assert "| Version |" in report
    assert "v1" in report
    assert "v2" in report


def test_inspect_cmd_missing_dir_exercises_error_handler(tmp_path: Path) -> None:
    """A3 fix: with exists=True removed, the FileNotFoundError path is real."""
    runner = CliRunner()
    result = runner.invoke(cli, ["inspect", str(tmp_path / "no-such-dir")])
    assert result.exit_code != 0
    assert "ERROR" in result.output or result.exit_code == 1


# ---------------------------------------------------------------------------
# write_inspection_report
# ---------------------------------------------------------------------------


def test_write_inspection_report_creates_file(tmp_path: Path) -> None:
    out_dir = _make_out_dir(tmp_path)
    report_path = tmp_path / "report.md"
    write_inspection_report(out_dir, report_path)
    assert report_path.exists()
    content = report_path.read_text(encoding="utf-8")
    assert "# Playbook Inspection Report" in content


def test_write_inspection_report_atomic(tmp_path: Path) -> None:
    """No .tmp file left behind after a successful write."""
    out_dir = _make_out_dir(tmp_path)
    report_path = tmp_path / "report.md"
    write_inspection_report(out_dir, report_path)
    tmp_file = report_path.with_suffix(".tmp")
    assert not tmp_file.exists()


# ---------------------------------------------------------------------------
# inspect CLI command
# ---------------------------------------------------------------------------


def test_inspect_cmd_stdout(tmp_path: Path) -> None:
    """inspect with no --out prints to stdout, exits 0."""
    out_dir = _make_out_dir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(cli, ["inspect", str(out_dir)])
    assert result.exit_code == 0
    assert "# Playbook Inspection Report" in result.output


def test_inspect_cmd_writes_file(tmp_path: Path) -> None:
    """inspect --out writes the report to a file."""
    out_dir = _make_out_dir(tmp_path)
    report_path = tmp_path / "report.md"
    runner = CliRunner()
    result = runner.invoke(cli, ["inspect", str(out_dir), "--out", str(report_path)])
    assert result.exit_code == 0
    assert report_path.exists()
    assert "# Playbook Inspection Report" in report_path.read_text(encoding="utf-8")


def test_inspect_cmd_missing_out_dir_exits_nonzero(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["inspect", str(tmp_path / "no-such-dir")])
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# Retired surfaces (issue #239): the inspection report is the skill's
# checkpoint only — no coherence-flag section, no floor-candidates section.
# ---------------------------------------------------------------------------


def test_report_has_no_coherence_or_floor_candidate_sections(tmp_path: Path) -> None:
    out_dir = _make_out_dir(tmp_path)
    (out_dir / "floor.candidates.json").write_text(
        json.dumps({"candidates": [{"id": "cand-001", "statement": "Never X."}]}),
        encoding="utf-8",
    )
    report = build_inspection_report(out_dir)
    assert "## Coherence Flags" not in report
    assert "Floor candidates" not in report
    assert "cand-001" not in report


def test_report_observation_table_shows_the_standard_fact_not_a_risk_verdict(
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    _write_trail(out_dir, "deal-a", ordered_versions=["v1"], signed_version="v1")
    _write_scope(out_dir, [{"document_id": "deal-a", "in_scope": True}])
    standard = _make_obs("deal-a", "v1", "Our standard clause.")
    standard["standard"] = True
    other = _make_obs("deal-a", "v1", "Their wording of the clause.")
    other["standard"] = False
    _write_observations(out_dir, [standard, other])
    report = build_inspection_report(out_dir)
    assert "| Standard | Outcome |" in report
    assert "| yes |" in report and "| no |" in report
    assert "Risk" not in report and "Deviation" not in report


# ---------------------------------------------------------------------------
# render_review_flags — issue #59
# ---------------------------------------------------------------------------


def _make_review_flag(
    kind: str = "ambiguous_version_chain",
    severity: str = "warn",
    document_id: str | None = "doc-1",
    suggested_action: str = "review it",
) -> dict:
    return {
        "document_id": document_id,
        "stage": "trail",
        "kind": kind,
        "severity": severity,
        "detail": "some detail",
        "suggested_action": suggested_action,
    }


def test_render_review_flags_empty_returns_empty_string() -> None:
    """No flags → empty string."""
    assert render_review_flags([]) == ""


def test_render_review_flags_shows_header() -> None:
    """Non-empty flags → 'Needs attention' header."""
    result = render_review_flags([_make_review_flag()])
    assert "## Needs attention" in result


def test_render_review_flags_shows_kind() -> None:
    result = render_review_flags([_make_review_flag(kind="scope_judge_failed")])
    assert "scope_judge_failed" in result


def test_render_review_flags_shows_severity() -> None:
    result = render_review_flags([_make_review_flag(severity="warn")])
    assert "warn" in result


def test_render_review_flags_block_severity_bold() -> None:
    """Block severity is rendered in bold."""
    result = render_review_flags([_make_review_flag(severity="block")])
    assert "**block**" in result


def test_render_review_flags_none_document_rendered_as_corpus() -> None:
    """document_id=None is rendered as corpus marker."""
    result = render_review_flags([_make_review_flag(document_id=None)])
    assert "*(corpus)*" in result


def test_render_review_flags_shows_suggested_action() -> None:
    result = render_review_flags([_make_review_flag(suggested_action="Do this now")])
    assert "Do this now" in result


# ---------------------------------------------------------------------------
# build_inspection_report: Needs attention section — issue #59
# ---------------------------------------------------------------------------


def _write_review_json(out_dir: Path, flags: list[dict]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "review.json").write_text(json.dumps({"flags": flags}), encoding="utf-8")


def _write_manifest(out_dir: Path, docs: list[dict]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "corpus_manifest.json").write_text(json.dumps(docs), encoding="utf-8")


def test_report_needs_attention_section_present_when_review_json_exists(
    tmp_path: Path,
) -> None:
    """review.json with flags → '## Needs attention' section in report."""
    out_dir = _make_out_dir(tmp_path)
    _write_review_json(out_dir, [_make_review_flag()])
    report = build_inspection_report(out_dir)
    assert "## Needs attention" in report


def test_report_needs_attention_shows_flag_kind(tmp_path: Path) -> None:
    out_dir = _make_out_dir(tmp_path)
    _write_review_json(out_dir, [_make_review_flag(kind="weak_signed_anchor")])
    report = build_inspection_report(out_dir)
    assert "weak_signed_anchor" in report


def test_report_needs_attention_shows_flag_severity(tmp_path: Path) -> None:
    out_dir = _make_out_dir(tmp_path)
    _write_review_json(out_dir, [_make_review_flag(severity="block")])
    report = build_inspection_report(out_dir)
    assert "block" in report


def test_report_no_needs_attention_without_review_json(tmp_path: Path) -> None:
    """Absent review.json → no 'Needs attention' section (back-compat)."""
    out_dir = _make_out_dir(tmp_path)
    report = build_inspection_report(out_dir)
    assert "## Needs attention" not in report


def test_report_no_needs_attention_for_empty_flags(tmp_path: Path) -> None:
    """review.json with empty flags → no 'Needs attention' section."""
    out_dir = _make_out_dir(tmp_path)
    _write_review_json(out_dir, [])
    report = build_inspection_report(out_dir)
    assert "## Needs attention" not in report


def test_report_dedupes_version_ingest_failure_present_in_both_sources(
    tmp_path: Path,
) -> None:
    """A version_ingest failure recorded in BOTH corpus_manifest.json (read
    directly by ``_version_ingest_review_flags``) and a review.json sidecar's
    flag must render as a single Needs-Attention row, not two (issue #89
    fix-round-1: the two call sites' ``suggested_action`` strings must match
    verbatim so the dedupe key
    collapses the cross-source duplicate).
    """
    out_dir = _make_out_dir(tmp_path)
    _write_manifest(
        out_dir,
        [
            {
                "document_id": "deal-alice",
                "version_ingest": [
                    {
                        "version": "v1",
                        "status": "failed",
                        "error": "empty clause tree",
                        "extractor": "rtf",
                    }
                ],
            }
        ],
    )
    # Mirrors exactly what a review.json sidecar would carry for the
    # identical underlying failure.
    _write_review_json(
        out_dir,
        [
            {
                "document_id": "deal-alice",
                "stage": "ingest",
                "kind": "version_ingest_failed",
                "severity": "warn",
                "detail": ("Version 'v1' failed to ingest and was never mined: empty clause tree"),
                "suggested_action": (
                    "Version 'v1' failed to ingest and was never mined: empty clause tree. "
                    "Inspect the source file and re-run 'playbook mine' with --no-cache."
                ),
            }
        ],
    )
    report = build_inspection_report(out_dir)
    assert report.count("version_ingest_failed") == 1, (
        "a failure present in both corpus_manifest.json and review.json must render as "
        "exactly one Needs-Attention row, not two"
    )


# ---------------------------------------------------------------------------
# version_ingest fallback advisory flag — issue #81
# ---------------------------------------------------------------------------


def test_report_shows_version_ingest_fallback_flag(tmp_path: Path) -> None:
    """A version mined via the legacy extractor after a real degradation
    (reason="backend-error"/"env-missing") gets an advisory info flag in the
    Needs-Attention section — distinct from an actual ingest failure, since
    the version WAS mined, just possibly lower-fidelity."""
    out_dir = _make_out_dir(tmp_path)
    _write_manifest(
        out_dir,
        [
            {
                "document_id": "deal-alice",
                "version_ingest": [
                    {
                        "version": "v1",
                        "status": "ok",
                        "error": None,
                        "extractor": "legacy",
                        "reason": "backend-error",
                    }
                ],
            }
        ],
    )
    report = build_inspection_report(out_dir)
    assert "version_ingest_fallback" in report
    assert "version_ingest_failed" not in report


def test_report_flags_ocr_recovered_version_without_claiming_no_ocr(tmp_path: Path) -> None:
    """Issue #231: an OCR-recovered scan (reason="ocr-recovered", written by
    extract_blocks when only the ocrmypdf pass found text) is still an
    advisory fallback — but its message must not say "no OCR", and must not
    read like the born-digital backend-error fallback."""
    out_dir = _make_out_dir(tmp_path)
    _write_manifest(
        out_dir,
        [
            {
                "document_id": "deal-alice",
                "version_ingest": [
                    {
                        "version": "v1",
                        "status": "ok",
                        "error": None,
                        "extractor": "legacy",
                        "reason": "ocr-recovered",
                    }
                ],
            }
        ],
    )
    report = build_inspection_report(out_dir)
    assert "version_ingest_fallback" in report
    assert "recovered by OCR" in report
    assert "no OCR" not in report
    assert "fell back to legacy" not in report


def test_report_no_fallback_flag_for_declared_legacy(tmp_path: Path) -> None:
    """reason="declared" (a deliberate config choice) must not produce an
    advisory fallback flag."""
    out_dir = _make_out_dir(tmp_path)
    _write_manifest(
        out_dir,
        [
            {
                "document_id": "deal-alice",
                "version_ingest": [
                    {
                        "version": "v1",
                        "status": "ok",
                        "error": None,
                        "extractor": "legacy",
                        "reason": "declared",
                    }
                ],
            }
        ],
    )
    report = build_inspection_report(out_dir)
    assert "version_ingest_fallback" not in report


def test_report_dedupes_version_ingest_fallback_present_in_both_sources(
    tmp_path: Path,
) -> None:
    """Mirrors test_report_dedupes_version_ingest_failure_present_in_both_sources
    for the NEW fallback kind (issue #81): a review.json sidecar's flag and
    inspection_report._version_ingest_review_flags must build byte-identical
    suggested_action text for the SAME underlying fallback, or a version
    present in both corpus_manifest.json and review.json renders as two
    Needs-Attention rows instead of one.
    """
    out_dir = _make_out_dir(tmp_path)
    _write_manifest(
        out_dir,
        [
            {
                "document_id": "deal-alice",
                "version_ingest": [
                    {
                        "version": "v1",
                        "status": "ok",
                        "error": None,
                        "extractor": "legacy",
                        "reason": "backend-error",
                    }
                ],
            }
        ],
    )
    # Mirrors exactly what a review.json sidecar would carry for the
    # identical underlying fallback.
    _write_review_json(
        out_dir,
        [
            {
                "document_id": "deal-alice",
                "stage": "ingest",
                "kind": "version_ingest_fallback",
                "severity": "info",
                "detail": (
                    "Version 'v1' was mined via the legacy extractor: docling failed on "
                    "this file and the engine fell back to legacy."
                ),
                "suggested_action": (
                    "Version 'v1' was mined via the legacy extractor: docling failed on "
                    "this file and the engine fell back to legacy — no OCR, no "
                    "docling-derived heading structure for this version. Review the "
                    "extracted text, or install docling and re-run 'playbook mine' with "
                    "--no-cache to recover the higher-fidelity docling extraction."
                ),
            }
        ],
    )
    report = build_inspection_report(out_dir)
    assert report.count("version_ingest_fallback") == 1, (
        "a fallback present in both corpus_manifest.json and review.json must render as "
        "exactly one Needs-Attention row, not two"
    )


# ---------------------------------------------------------------------------
# Trail confidence fields — issue #59
# ---------------------------------------------------------------------------


def test_report_shows_provenance_confidence(tmp_path: Path) -> None:
    """provenance_confidence is surfaced in the inspection report."""
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    _write_trail(
        out_dir,
        "deal-alice",
        ordered_versions=["v1"],
        provenance="our_paper",
        provenance_confidence=0.82,
        provenance_is_ambiguous=False,
    )
    _write_scope(out_dir, [{"document_id": "deal-alice", "in_scope": True}])
    report = build_inspection_report(out_dir)
    assert "0.82" in report
    assert "Provenance confidence" in report


def test_report_shows_provenance_is_ambiguous(tmp_path: Path) -> None:
    """provenance_is_ambiguous=True is flagged in the report."""
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    _write_trail(
        out_dir,
        "deal-alice",
        ordered_versions=["v1"],
        provenance="unknown",
        provenance_confidence=0.55,
        provenance_is_ambiguous=True,
    )
    _write_scope(out_dir, [{"document_id": "deal-alice", "in_scope": True}])
    report = build_inspection_report(out_dir)
    assert "ambiguous" in report


def test_report_shows_signed_copy_confidence(tmp_path: Path) -> None:
    """signed_copy_confidence is surfaced in the inspection report."""
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    _write_trail(
        out_dir,
        "deal-alice",
        ordered_versions=["v1", "v2"],
        signed_version="v2",
        provenance="our_paper",
        signed_copy_confidence=0.91,
    )
    _write_scope(out_dir, [{"document_id": "deal-alice", "in_scope": True}])
    report = build_inspection_report(out_dir)
    assert "0.91" in report
    assert "Signed copy confidence" in report


def test_report_shows_signed_copy_none_when_undetected(tmp_path: Path) -> None:
    """No signed copy detected → the row is still rendered, as '(none)'.

    Previously the 'Signed copy' row was omitted entirely whenever
    signed_version was falsy, silently leaving reviewers unable to
    distinguish "no signed copy" from "field not rendered".
    """
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    _write_trail(
        out_dir,
        "deal-alice",
        ordered_versions=["v1"],
        signed_version=None,
        provenance="our_paper",
    )
    _write_scope(out_dir, [{"document_id": "deal-alice", "in_scope": True}])
    report = build_inspection_report(out_dir)
    assert "Signed copy" in report
    assert "(none)" in report
    assert "| Signed copy | (none) |" in report


def test_report_shows_chain_shape(tmp_path: Path) -> None:
    """shape is surfaced in the inspection report."""
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    _write_trail(
        out_dir,
        "deal-alice",
        ordered_versions=["v1"],
        provenance="our_paper",
        shape="linear",
    )
    _write_scope(out_dir, [{"document_id": "deal-alice", "in_scope": True}])
    report = build_inspection_report(out_dir)
    assert "Chain shape" in report
    assert "linear" in report


def test_report_confidence_fields_absent_no_crash(tmp_path: Path) -> None:
    """Trail with no confidence fields renders without crashing."""
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    _write_trail(
        out_dir,
        "deal-alice",
        ordered_versions=["v1"],
        provenance="our_paper",
    )
    _write_scope(out_dir, [{"document_id": "deal-alice", "in_scope": True}])
    # Must not raise
    report = build_inspection_report(out_dir)
    assert "## deal-alice" in report
