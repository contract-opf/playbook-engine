"""Tests for the checkpoint-review module (issue #57).

SECURITY NOTE: All fixtures are programmatically constructed with synthetic
data.  No real agreements are referenced.  Fictional document identifiers only.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from playbook_engine.review import ReviewFlag, review_out_dir, write_review

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_scope(tmp_path: pathlib.Path, documents: list[dict]) -> None:
    scope = {
        "agreement_type_id": "test-agreement",
        "stats": {"total": len(documents), "in_scope": 0, "out_of_scope": 0, "judge_error": 0},
        "documents": documents,
    }
    (tmp_path / "scope.json").write_text(json.dumps(scope), encoding="utf-8")


def _write_trail(tmp_path: pathlib.Path, doc_id: str, trail: dict) -> None:
    trail_dir = tmp_path / "trail"
    trail_dir.mkdir(exist_ok=True)
    (trail_dir / f"{doc_id}.json").write_text(json.dumps(trail), encoding="utf-8")


def _write_obs(tmp_path: pathlib.Path, observations: list[dict]) -> None:
    lines = "\n".join(json.dumps(o) for o in observations)
    (tmp_path / "observations.jsonl").write_text(lines, encoding="utf-8")


def _flags_of_kind(flags: list[ReviewFlag], kind: str) -> list[ReviewFlag]:
    return [f for f in flags if f.kind == kind]


# ---------------------------------------------------------------------------
# Clean artifacts → empty list
# ---------------------------------------------------------------------------


def test_empty_dir_returns_no_flags(tmp_path: pathlib.Path) -> None:
    """Completely empty output directory produces zero flags."""
    assert review_out_dir(tmp_path) == []


def test_clean_artifacts_return_no_flags(tmp_path: pathlib.Path) -> None:
    """Well-formed artifacts with no issues produce zero flags."""
    _write_scope(
        tmp_path,
        [
            {
                "document_id": "doc-clean",
                "in_scope": True,
                "scope_rationale": "ok",
                "scope_confidence": 0.95,
                "basis": "deterministic",
            }
        ],
    )
    _write_trail(
        tmp_path,
        "doc-clean",
        {
            "document_id": "doc-clean",
            "basis": "deterministic",
            "shape": "linear",
            "signed_copy_confidence": 0.90,
            "provenance_is_ambiguous": False,
            "provenance_confidence": 0.85,
        },
    )
    _write_obs(
        tmp_path,
        [
            {
                "observation_id": "obs-1",
                "citation": {"document_id": "doc-clean", "version": 1, "clause_path": "1"},
                "deviation": "none",
                "basis": "deterministic",
            }
        ],
    )
    flags = review_out_dir(tmp_path)
    assert flags == []


# ---------------------------------------------------------------------------
# Scope rule: judge_error → block
# ---------------------------------------------------------------------------


def test_scope_judge_error_emits_block(tmp_path: pathlib.Path) -> None:
    _write_scope(
        tmp_path,
        [
            {
                "document_id": "doc-scope-err",
                "in_scope": True,
                "scope_rationale": "judge failed",
                "scope_confidence": 0.0,
                "basis": "judge_error",
            }
        ],
    )
    flags = review_out_dir(tmp_path)
    matched = _flags_of_kind(flags, "scope_judge_failed")
    assert len(matched) == 1
    f = matched[0]
    assert f.document_id == "doc-scope-err"
    assert f.severity == "block"
    assert f.stage == "scope"


def test_scope_normal_basis_no_flag(tmp_path: pathlib.Path) -> None:
    _write_scope(
        tmp_path,
        [
            {
                "document_id": "doc-ok",
                "in_scope": True,
                "scope_rationale": "in scope",
                "scope_confidence": 0.92,
                "basis": "deterministic",
            }
        ],
    )
    flags = review_out_dir(tmp_path)
    assert _flags_of_kind(flags, "scope_judge_failed") == []


# ---------------------------------------------------------------------------
# Trail rules
# ---------------------------------------------------------------------------


def _base_trail(doc_id: str = "doc-trail") -> dict:
    """Return a trail dict that triggers no flags."""
    return {
        "document_id": doc_id,
        "basis": "deterministic",
        "shape": "linear",
        "signed_copy_confidence": 0.95,
        "provenance_is_ambiguous": False,
        "provenance_confidence": 0.85,
    }


@pytest.mark.parametrize("basis", ["greedy", "llm"])
def test_trail_ambiguous_version_chain(tmp_path: pathlib.Path, basis: str) -> None:
    trail = {**_base_trail("doc-t1"), "basis": basis}
    _write_trail(tmp_path, "doc-t1", trail)
    flags = review_out_dir(tmp_path)
    matched = _flags_of_kind(flags, "ambiguous_version_chain")
    assert len(matched) == 1
    assert matched[0].document_id == "doc-t1"
    assert matched[0].severity == "warn"
    assert matched[0].stage == "trail"


def test_trail_deterministic_basis_no_ambiguity_flag(tmp_path: pathlib.Path) -> None:
    _write_trail(tmp_path, "doc-det", _base_trail("doc-det"))
    flags = review_out_dir(tmp_path)
    assert _flags_of_kind(flags, "ambiguous_version_chain") == []


@pytest.mark.parametrize("shape", ["fork", "gap"])
def test_trail_fork_or_gap_shape(tmp_path: pathlib.Path, shape: str) -> None:
    trail = {**_base_trail("doc-fork"), "shape": shape}
    _write_trail(tmp_path, "doc-fork", trail)
    flags = review_out_dir(tmp_path)
    matched = _flags_of_kind(flags, "fork_or_missing_draft")
    assert len(matched) == 1
    assert matched[0].document_id == "doc-fork"
    assert matched[0].severity == "warn"


def test_trail_linear_shape_no_fork_flag(tmp_path: pathlib.Path) -> None:
    _write_trail(tmp_path, "doc-lin", _base_trail("doc-lin"))
    flags = review_out_dir(tmp_path)
    assert _flags_of_kind(flags, "fork_or_missing_draft") == []


def test_trail_weak_signed_anchor_below_threshold(tmp_path: pathlib.Path) -> None:
    trail = {**_base_trail(), "signed_copy_confidence": 0.50}
    _write_trail(tmp_path, "doc-signed", trail)
    flags = review_out_dir(tmp_path)
    matched = _flags_of_kind(flags, "weak_signed_anchor")
    assert len(matched) == 1
    assert matched[0].severity == "warn"


def test_trail_signed_confidence_at_threshold_no_flag(tmp_path: pathlib.Path) -> None:
    trail = {**_base_trail(), "signed_copy_confidence": 0.70}
    _write_trail(tmp_path, "doc-sat", trail)
    flags = review_out_dir(tmp_path)
    assert _flags_of_kind(flags, "weak_signed_anchor") == []


def test_trail_signed_confidence_none_no_flag(tmp_path: pathlib.Path) -> None:
    """Absent signed_copy_confidence (e.g. single-version doc) must not flag."""
    trail = {**_base_trail(), "signed_copy_confidence": None}
    _write_trail(tmp_path, "doc-none-sig", trail)
    flags = review_out_dir(tmp_path)
    assert _flags_of_kind(flags, "weak_signed_anchor") == []


def test_trail_provenance_is_ambiguous_true(tmp_path: pathlib.Path) -> None:
    trail = {**_base_trail(), "provenance_is_ambiguous": True}
    _write_trail(tmp_path, "doc-prov1", trail)
    flags = review_out_dir(tmp_path)
    matched = _flags_of_kind(flags, "unreliable_provenance")
    assert len(matched) == 1
    assert matched[0].severity == "warn"


def test_trail_provenance_confidence_below_threshold(tmp_path: pathlib.Path) -> None:
    trail = {**_base_trail(), "provenance_confidence": 0.60}
    _write_trail(tmp_path, "doc-prov2", trail)
    flags = review_out_dir(tmp_path)
    matched = _flags_of_kind(flags, "unreliable_provenance")
    assert len(matched) == 1


def test_trail_provenance_confidence_at_threshold_no_flag(tmp_path: pathlib.Path) -> None:
    trail = {**_base_trail(), "provenance_confidence": 0.70}
    _write_trail(tmp_path, "doc-provok", trail)
    flags = review_out_dir(tmp_path)
    assert _flags_of_kind(flags, "unreliable_provenance") == []


# ---------------------------------------------------------------------------
# Observation rules
# ---------------------------------------------------------------------------


def _base_obs(obs_id: str = "obs-1", doc_id: str = "doc-obs") -> dict:
    return {
        "observation_id": obs_id,
        "citation": {"document_id": doc_id, "version": 1, "clause_path": "1"},
        "deviation": "none",
        "basis": "deterministic",
    }


def test_observation_basis_judge_error(tmp_path: pathlib.Path) -> None:
    obs = {**_base_obs(), "basis": "judge_error"}
    _write_obs(tmp_path, [obs])
    flags = review_out_dir(tmp_path)
    matched = _flags_of_kind(flags, "deviation_needs_review")
    assert len(matched) == 1
    assert matched[0].severity == "warn"
    assert matched[0].stage == "observation"
    assert matched[0].document_id == "doc-obs"


def test_observation_deviation_needs_review(tmp_path: pathlib.Path) -> None:
    obs = {**_base_obs(), "deviation": "needs_review"}
    _write_obs(tmp_path, [obs])
    flags = review_out_dir(tmp_path)
    matched = _flags_of_kind(flags, "deviation_needs_review")
    assert len(matched) == 1
    assert matched[0].severity == "warn"


def test_observation_both_judge_error_and_needs_review(tmp_path: pathlib.Path) -> None:
    """A single observation with both triggers should produce exactly one flag."""
    obs = {**_base_obs(), "basis": "judge_error", "deviation": "needs_review"}
    _write_obs(tmp_path, [obs])
    flags = review_out_dir(tmp_path)
    matched = _flags_of_kind(flags, "deviation_needs_review")
    assert len(matched) == 1


def test_observation_clean_no_flag(tmp_path: pathlib.Path) -> None:
    _write_obs(tmp_path, [_base_obs()])
    flags = review_out_dir(tmp_path)
    assert _flags_of_kind(flags, "deviation_needs_review") == []


def test_multiple_observations_multiple_flags(tmp_path: pathlib.Path) -> None:
    observations = [
        {**_base_obs("obs-a", "doc-A"), "basis": "judge_error"},
        {**_base_obs("obs-b", "doc-B"), "deviation": "needs_review"},
        _base_obs("obs-c", "doc-C"),  # clean
    ]
    _write_obs(tmp_path, observations)
    flags = review_out_dir(tmp_path)
    matched = _flags_of_kind(flags, "deviation_needs_review")
    assert len(matched) == 2
    doc_ids = {f.document_id for f in matched}
    assert doc_ids == {"doc-A", "doc-B"}


# ---------------------------------------------------------------------------
# ReviewFlag dataclass
# ---------------------------------------------------------------------------


def test_reviewflag_invalid_severity_raises() -> None:
    with pytest.raises(ValueError, match="severity"):
        ReviewFlag(
            document_id="x",
            stage="scope",
            kind="bad",
            severity="critical",  # invalid
            detail="oops",
            suggested_action="fix it",
        )


def test_reviewflag_to_dict() -> None:
    f = ReviewFlag(
        document_id="doc-1",
        stage="trail",
        kind="ambiguous_version_chain",
        severity="warn",
        detail="something",
        suggested_action="review it",
    )
    d = f.to_dict()
    assert d["document_id"] == "doc-1"
    assert d["stage"] == "trail"
    assert d["kind"] == "ambiguous_version_chain"
    assert d["severity"] == "warn"
    assert d["detail"] == "something"
    assert d["suggested_action"] == "review it"


# ---------------------------------------------------------------------------
# write_review — issue #59
# ---------------------------------------------------------------------------


def _write_coherence_flags(tmp_path: pathlib.Path, entries: list[dict]) -> None:
    (tmp_path / "coherence_flags.json").write_text(json.dumps(entries), encoding="utf-8")


def test_write_review_returns_path(tmp_path: pathlib.Path) -> None:
    """write_review returns the path to review.json."""
    path = write_review(tmp_path)
    assert path == tmp_path / "review.json"
    assert path.exists()


def test_write_review_empty_dir_produces_empty_flags(tmp_path: pathlib.Path) -> None:
    """Empty output directory → review.json with flags=[]."""
    path = write_review(tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data == {"flags": []}


def test_write_review_round_trips(tmp_path: pathlib.Path) -> None:
    """review.json round-trips: all flag keys are present and correct."""
    _write_scope(
        tmp_path,
        [
            {
                "document_id": "doc-rt",
                "in_scope": True,
                "scope_rationale": "judge failed",
                "scope_confidence": 0.0,
                "basis": "judge_error",
            }
        ],
    )
    path = write_review(tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert "flags" in data
    flags = data["flags"]
    assert len(flags) == 1
    f = flags[0]
    assert f["document_id"] == "doc-rt"
    assert f["stage"] == "scope"
    assert f["kind"] == "scope_judge_failed"
    assert f["severity"] == "block"
    assert "detail" in f
    assert "suggested_action" in f


def test_write_review_folds_in_coherence_flags(tmp_path: pathlib.Path) -> None:
    """write_review incorporates coherence_flags.json entries."""
    _write_coherence_flags(
        tmp_path,
        [
            {
                "clause_id": "clause.indemnification",
                "reason": "low n_our_paper (n=1)",
                "severity": "warn",
            }
        ],
    )
    path = write_review(tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    flags = data["flags"]
    coherence = [f for f in flags if f["stage"] == "coherence"]
    assert len(coherence) == 1
    cf = coherence[0]
    assert cf["kind"] == "low_coherence"
    assert cf["severity"] == "warn"
    assert cf["document_id"] is None
    assert "clause.indemnification" in cf["detail"]


def test_write_review_no_coherence_flags_file(tmp_path: pathlib.Path) -> None:
    """Absent coherence_flags.json produces no coherence flags."""
    path = write_review(tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    coherence = [f for f in data["flags"] if f["stage"] == "coherence"]
    assert coherence == []


def test_write_review_coherence_flags_invalid_severity_coerced(
    tmp_path: pathlib.Path,
) -> None:
    """An invalid coherence severity is coerced to 'warn' without crashing."""
    _write_coherence_flags(
        tmp_path,
        [{"clause_id": "clause.x", "reason": "bad sev", "severity": "critical"}],
    )
    path = write_review(tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    coherence = [f for f in data["flags"] if f["stage"] == "coherence"]
    assert len(coherence) == 1
    assert coherence[0]["severity"] == "warn"


def test_write_review_multiple_sources(tmp_path: pathlib.Path) -> None:
    """write_review combines scope, trail, observation, and coherence flags."""
    # scope flag
    _write_scope(
        tmp_path,
        [
            {
                "document_id": "doc-multi",
                "in_scope": True,
                "basis": "judge_error",
            }
        ],
    )
    # trail flag (ambiguous)
    _write_trail(
        tmp_path,
        "doc-multi",
        {
            "document_id": "doc-multi",
            "basis": "greedy",
            "shape": "linear",
        },
    )
    # coherence flag
    _write_coherence_flags(
        tmp_path,
        [{"clause_id": "clause.term", "reason": "low count", "severity": "warn"}],
    )
    path = write_review(tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    flags = data["flags"]
    stages = {f["stage"] for f in flags}
    assert "scope" in stages
    assert "trail" in stages
    assert "coherence" in stages


def test_write_review_missing_out_dir_raises(tmp_path: pathlib.Path) -> None:
    """write_review raises FileNotFoundError when out_dir does not exist."""
    with pytest.raises(FileNotFoundError):
        write_review(tmp_path / "no-such-dir")


def test_write_review_atomic(tmp_path: pathlib.Path) -> None:
    """No .tmp file left behind after a successful write."""
    write_review(tmp_path)
    tmp_file = (tmp_path / "review.json").with_suffix(".tmp")
    assert not tmp_file.exists()
