"""Tests for Floor-candidate proposal (issue #166).

Acceptance criteria verified here (mirrors the issue's Required verification):

  - Every ``outcome: proposed_then_reversed`` observation in Evidence is a
    candidate red line (OPF §3.7 rule 4), grouped by taxonomy_id, citing the
    contributing reversal observation(s).
  - The Posture interview's Q4 ("sacred_clauses") answer seeds candidates too
    (OPF §7).
  - Proposal is NEVER auto-promoted: ``playbook floor propose`` never touches
    the OPF ``floor.invariants`` (spec rule 4).
  - No reversals + no Q4 answer -> ``{"candidates": []}``, exit 0.

SECURITY NOTE: All fixtures are synthetic, minimal dicts — no real legal text,
no real parties.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from click.testing import CliRunner

from playbook_engine.cli import cli
from playbook_engine.floor_candidates import (
    derive_interview_q4_candidates,
    derive_reversal_candidates,
    propose_floor_candidates,
    write_floor_candidates,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _reversal_observation(
    *,
    observation_id: str = "doc-a/2/8.1",
    taxonomy_id: str | None = "uncapped_liability",
    document_id: str = "doc-a",
    version: int = 2,
    clause_path: str = "8.1",
    full_text: str = "The Vendor's liability shall be uncapped for any breach.",
) -> dict[str, Any]:
    return {
        "observation_id": observation_id,
        "taxonomy_id": taxonomy_id,
        "text_summary": full_text[:200],
        "full_text": full_text,
        "citation": {
            "document_id": document_id,
            "version": version,
            "clause_path": clause_path,
            "char_span": None,
            "version_id": None,
        },
        "deviation": "substantive",
        "risk_delta": {"direction": "neutral", "magnitude": "none"},
        "provenance": "counterparty_paper",
        "outcome": "proposed_then_reversed",
        "confidence": None,
        "basis": "deterministic",
    }


def _signed_observation(observation_id: str = "doc-a/2/1.1") -> dict[str, Any]:
    obs = _reversal_observation(observation_id=observation_id, clause_path="1.1")
    obs["outcome"] = "signed"
    return obs


def _minimal_v02_doc(**overrides: Any) -> dict[str, Any]:
    doc: dict[str, Any] = {
        "opf_version": "0.2",
        "agreement_type": {"id": "test-agreement", "name": "Test Agreement"},
        "baseline": {"has_canonical_template": False},
        "taxonomy": {"source": "custom", "entries": []},
        "evidence": {"clauses": [], "clause_library": []},
        "posture": {},
        "floor": {},
        "corpus": {"documents": [], "stats": {}},
        "compiler": {
            "name": "playbook-engine",
            "version": "0.1.0",
            "run_id": "run-abc",
            "generated_at": "2026-01-01T00:00:00Z",
        },
        "identity": {
            "content_hash": "sha256:" + "0" * 64,
            "section_digests": {
                "evidence": "sha256:" + "1" * 64,
                "posture": "sha256:" + "2" * 64,
                "floor": "sha256:" + "3" * 64,
                "curation": "sha256:" + "4" * 64,
            },
        },
    }
    doc.update(overrides)
    return doc


# ---------------------------------------------------------------------------
# derive_reversal_candidates
# ---------------------------------------------------------------------------


def test_reversal_yields_candidate() -> None:
    observations = [_reversal_observation()]

    candidates = derive_reversal_candidates(observations)

    assert len(candidates) == 1
    cand = candidates[0]
    assert cand.source == "reversal"
    assert "uncapped liability" in cand.statement.lower()
    assert cand.statement.startswith("Never accept")
    assert len(cand.citations) >= 1
    cite = cand.citations[0]
    assert cite.document_id == "doc-a"
    assert cite.version == 2
    assert cite.clause_path == "8.1"


def test_reversal_ignores_non_reversed_observations() -> None:
    observations = [_reversal_observation(), _signed_observation()]

    candidates = derive_reversal_candidates(observations)

    assert len(candidates) == 1  # only the reversed one becomes a candidate


def test_reversal_groups_by_taxonomy_id_across_documents() -> None:
    observations = [
        _reversal_observation(observation_id="doc-a/2/8.1", document_id="doc-a"),
        _reversal_observation(
            observation_id="doc-b/3/9.1", document_id="doc-b", version=3, clause_path="9.1"
        ),
    ]

    candidates = derive_reversal_candidates(observations)

    assert len(candidates) == 1  # same taxonomy_id -> one candidate
    assert "2 deal" in candidates[0].rationale
    assert len(candidates[0].citations) == 2


def test_reversal_unclassified_observations_do_not_collapse() -> None:
    observations = [
        _reversal_observation(
            observation_id="doc-a/2/8.1", taxonomy_id=None, full_text="Unusual clause A."
        ),
        _reversal_observation(
            observation_id="doc-a/2/9.1",
            taxonomy_id=None,
            clause_path="9.1",
            full_text="Unusual clause B.",
        ),
    ]

    candidates = derive_reversal_candidates(observations)

    assert len(candidates) == 2  # distinct unclassified reversals stay distinct


# ---------------------------------------------------------------------------
# derive_interview_q4_candidates
# ---------------------------------------------------------------------------


def test_interview_q4_yields_candidates() -> None:
    answers = {"sacred_clauses": "uncapped liability; IP assignment"}

    candidates = derive_interview_q4_candidates(answers)

    assert len(candidates) == 2
    assert all(c.source == "interview_q4" for c in candidates)
    assert all(c.citations == [] for c in candidates)
    statements = {c.statement for c in candidates}
    assert any("uncapped liability" in s for s in statements)
    assert any("IP assignment" in s for s in statements)


def test_interview_q4_missing_answer_yields_no_candidates() -> None:
    assert derive_interview_q4_candidates({}) == []
    assert derive_interview_q4_candidates(None) == []
    assert derive_interview_q4_candidates({"sacred_clauses": "   "}) == []
    assert derive_interview_q4_candidates({"rounds": "2 rounds"}) == []


# ---------------------------------------------------------------------------
# propose_floor_candidates — combined, pure
# ---------------------------------------------------------------------------


def test_propose_floor_candidates_combines_and_ids_sequentially() -> None:
    observations = [_reversal_observation()]
    answers = {"sacred_clauses": "uncapped liability; IP assignment"}

    result = propose_floor_candidates(observations, answers)

    ids = [c["id"] for c in result["candidates"]]
    assert ids == ["cand-001", "cand-002", "cand-003"]
    sources = [c["source"] for c in result["candidates"]]
    assert sources == ["reversal", "interview_q4", "interview_q4"]


def test_empty_corpus_empty_candidates() -> None:
    result = propose_floor_candidates([], None)
    assert result == {"candidates": []}


# ---------------------------------------------------------------------------
# write_floor_candidates — I/O
# ---------------------------------------------------------------------------


def test_write_floor_candidates_reads_observations_and_posture(tmp_path: Path) -> None:
    obs_path = tmp_path / "observations.jsonl"
    obs_path.write_text(
        json.dumps(_reversal_observation()) + "\n",
        encoding="utf-8",
    )
    doc = _minimal_v02_doc(
        posture={
            "generation": {
                "interview": [
                    {"q": "sacred_clauses", "question": "...", "answer": "IP assignment"},
                ]
            }
        }
    )
    (tmp_path / "playbook.opf.json").write_text(json.dumps(doc), encoding="utf-8")

    out_path = write_floor_candidates(tmp_path)

    assert out_path == tmp_path / "floor.candidates.json"
    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert len(written["candidates"]) == 2
    sources = {c["source"] for c in written["candidates"]}
    assert sources == {"reversal", "interview_q4"}


def test_write_floor_candidates_no_playbook_no_observations(tmp_path: Path) -> None:
    out_path = write_floor_candidates(tmp_path)
    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert written == {"candidates": []}


# ---------------------------------------------------------------------------
# CLI — playbook floor propose
# ---------------------------------------------------------------------------


def _invoke(*args: str) -> tuple[int, str]:
    runner = CliRunner()
    result = runner.invoke(cli, list(args))
    return result.exit_code, result.output


def test_no_auto_promotion(tmp_path: Path) -> None:
    obs_path = tmp_path / "observations.jsonl"
    obs_path.write_text(json.dumps(_reversal_observation()) + "\n", encoding="utf-8")
    doc = _minimal_v02_doc(floor={"invariants": []})
    opf_path = tmp_path / "playbook.opf.json"
    original_bytes = json.dumps(doc).encode("utf-8")
    opf_path.write_bytes(original_bytes)

    exit_code, output = _invoke("floor", "propose", str(tmp_path))

    assert exit_code == 0, output
    assert (tmp_path / "floor.candidates.json").exists()
    # playbook.opf.json is byte-identical — 'floor propose' never writes to it.
    assert opf_path.read_bytes() == original_bytes
    written = json.loads(opf_path.read_bytes())
    assert written["floor"]["invariants"] == []


def test_cli_floor_propose_empty_corpus(tmp_path: Path) -> None:
    exit_code, output = _invoke("floor", "propose", str(tmp_path))

    assert exit_code == 0, output
    assert "0 candidates" in output
    written = json.loads((tmp_path / "floor.candidates.json").read_text(encoding="utf-8"))
    assert written == {"candidates": []}


def test_cli_floor_propose_missing_out_dir_fails() -> None:
    exit_code, output = _invoke("floor", "propose", "/nonexistent/out/dir")
    assert exit_code != 0
