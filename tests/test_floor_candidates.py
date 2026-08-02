"""Tests for Floor-candidate proposal (issue #166).

Acceptance criteria verified here (mirrors the issue's Required verification):

  - Every ``outcome: proposed_then_reversed`` observation in Evidence is a
    candidate hard line (OPF §3.7 rule 4), grouped by taxonomy_id, citing the
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

import pytest
from click.testing import CliRunner

from playbook_engine.cli import cli
from playbook_engine.floor_candidates import (
    FloorCandidateError,
    derive_interview_q4_candidates,
    derive_reversal_candidates,
    promote_interview_q4_invariants,
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
# promote_interview_q4_invariants — direct Floor promotion (issue #89)
# ---------------------------------------------------------------------------


def test_promote_q4_writes_new_invariants_with_attribution() -> None:
    answers = {"sacred_clauses": "uncapped liability; IP assignment"}

    result = promote_interview_q4_invariants(answers, posture_version=1, existing_invariants=[])

    assert len(result) == 2
    for inv in result:
        assert inv["id"]
        assert inv["statement"].startswith("Do not concede on ")
        assert "posture interview v1" in inv["rationale"]
        assert "sacred_clauses" in inv["rationale"]
    statements = {inv["statement"] for inv in result}
    assert any("uncapped liability" in s for s in statements)
    assert any("IP assignment" in s for s in statements)
    # Every id is a stable slug of its statement's named item, not a
    # sequential cand-NNN — unlike the candidate ids assigned by
    # propose_floor_candidates, these ARE the OPF floor.invariants[].id.
    ids = {inv["id"] for inv in result}
    assert "uncapped-liability" in ids
    assert "ip-assignment" in ids


def test_promote_q4_no_answer_returns_existing_invariants_unchanged() -> None:
    existing = [{"id": "hand-authored", "statement": "Never do X.", "rationale": "Because."}]

    assert (
        promote_interview_q4_invariants(None, posture_version=1, existing_invariants=existing)
        == existing
    )
    assert (
        promote_interview_q4_invariants({}, posture_version=1, existing_invariants=existing)
        == existing
    )
    assert (
        promote_interview_q4_invariants(
            {"sacred_clauses": "   "}, posture_version=1, existing_invariants=existing
        )
        == existing
    )
    assert (
        promote_interview_q4_invariants(
            {"rounds": "2 rounds"}, posture_version=1, existing_invariants=existing
        )
        == existing
    )


def test_promote_q4_preserves_hand_authored_invariants() -> None:
    hand_authored = {
        "id": "no-uncapped-liability",
        "statement": "Never accept uncapped liability.",
        "rationale": "Categorically unacceptable regardless of deal value.",
    }
    answers = {"sacred_clauses": "IP assignment"}

    result = promote_interview_q4_invariants(
        answers, posture_version=1, existing_invariants=[hand_authored]
    )

    assert hand_authored in result  # byte-identical, untouched
    assert len(result) == 2


def test_promote_q4_rerun_same_answer_is_true_noop() -> None:
    answers = {"sacred_clauses": "uncapped liability; IP assignment"}

    first = promote_interview_q4_invariants(answers, posture_version=1, existing_invariants=[])
    # Simulate a second interview run (posture.version bumps to 2) with the
    # exact same answer.
    second = promote_interview_q4_invariants(answers, posture_version=2, existing_invariants=first)

    assert second == first  # no duplicates, no rationale churn — a true no-op
    assert len(second) == 2
    ids = [inv["id"] for inv in second]
    assert len(ids) == len(set(ids))  # OPF-SPEC.md §3.13: no duplicate sibling ids


def test_promote_q4_rerun_with_changed_wording_updates_in_place() -> None:
    """Issue #89 review finding 4: this test must actually drive the
    ``existing_index is not None`` branch with a DIFFERING ``statement`` for
    the SAME slug. Pre-fix, this test ran the identical answer twice (a
    no-op) and then a genuinely different item (an append) — the
    update-in-place branch (the one branch that can rewrite an existing
    entry, and so the one finding 2 guards) was never exercised at all."""
    first = promote_interview_q4_invariants(
        {"sacred_clauses": "uncapped liability"}, posture_version=1, existing_invariants=[]
    )
    assert len(first) == 1
    assert first[0]["id"] == "uncapped-liability"
    assert first[0]["statement"] == "Do not concede on uncapped liability."

    # Re-run with different casing for the SAME item: same slug
    # ("uncapped-liability"), a genuinely different statement string. The
    # existing entry carries THIS function's own attribution marker (it was
    # itself written by the promotion above), so this is the legitimate
    # update-in-place case — not the foreign-collision case finding 2
    # guards against (see
    # test_promote_q4_refuses_to_overwrite_colliding_hand_authored_id).
    second = promote_interview_q4_invariants(
        {"sacred_clauses": "Uncapped Liability"}, posture_version=2, existing_invariants=first
    )

    assert len(second) == 1  # updated in place, not appended alongside
    assert second[0]["id"] == "uncapped-liability"
    assert second[0]["statement"] == "Do not concede on Uncapped Liability."
    assert second[0]["statement"] != first[0]["statement"]
    assert "posture interview v2" in second[0]["rationale"]


def test_promote_q4_rerun_dropped_item_is_not_deleted() -> None:
    # First run names two items; second run's answer only re-names one.
    # The dropped item's invariant must survive (upsert, never a delete).
    first = promote_interview_q4_invariants(
        {"sacred_clauses": "uncapped liability; IP assignment"},
        posture_version=1,
        existing_invariants=[],
    )
    second = promote_interview_q4_invariants(
        {"sacred_clauses": "uncapped liability"}, posture_version=2, existing_invariants=first
    )

    assert len(second) == 2
    ids = {inv["id"] for inv in second}
    assert "uncapped-liability" in ids
    assert "ip-assignment" in ids


def test_promote_q4_tolerates_bare_string_invariants() -> None:
    # A hand-edited playbook MAY carry a bare-string floor invariant
    # (document_renderer.py/prompt_renderer.py both tolerate this shape) —
    # the merge must pass it through untouched, not crash on .get().
    existing: list[Any] = ["No indemnity cap below $1M"]
    answers = {"sacred_clauses": "IP assignment"}

    result = promote_interview_q4_invariants(
        answers, posture_version=1, existing_invariants=existing
    )

    assert "No indemnity cap below $1M" in result
    assert len(result) == 2


def test_promote_q4_semicolon_separated_items_get_distinct_ids() -> None:
    answers = {"sacred_clauses": "Liability caps and student-data protection"}

    result = promote_interview_q4_invariants(answers, posture_version=1, existing_invariants=[])

    assert len(result) == 1
    assert result[0]["id"] == "liability-caps-and-student-data-protection"
    assert result[0]["statement"] == "Do not concede on Liability caps and student-data protection."


def test_promote_q4_statement_does_not_invert_sacred_clause_into_prohibition() -> None:
    """Issue #89 review finding 3 regression: Q4 asks which clause types are
    non-negotiable -- i.e. things the legal owner insists on KEEPING. The
    promoted ACTIVE invariant must not read as "Never accept <the thing we
    want>." -- that would instruct the Floor judge to force
    negotiation-unacceptable on any clause that CONTAINS student-data
    protection, the opposite of the legal owner's intent."""
    answers = {"sacred_clauses": "Liability caps and student-data protection"}

    result = promote_interview_q4_invariants(answers, posture_version=1, existing_invariants=[])

    assert len(result) == 1
    statement = result[0]["statement"]
    assert not statement.lower().startswith("never accept")
    assert "accept" not in statement.lower()
    assert "Liability caps and student-data protection" in statement


def test_promote_q4_refuses_to_overwrite_colliding_hand_authored_id() -> None:
    """Issue #89 review finding 2 regression: an existing invariant whose id
    happens to equal a freshly Q4-named item's slug, but which this
    function did NOT itself promote (no matching attribution marker in its
    rationale), must never be silently overwritten -- even though the ids
    collide byte-for-byte. A hand-authored, signed-off statement (plus any
    x_* extension field) must survive untouched; the promotion fails
    loudly instead of silently destroying it."""
    hand_authored = {
        "id": "ip-assignment",
        "statement": "Never accept present-tense assignment of pre-existing IP.",
        "rationale": "Signed off by the GC 2026-03-01 after board review.",
        "x_signed_by": "gc@example.com",
    }
    answers = {"sacred_clauses": "IP assignment"}  # slugifies to "ip-assignment"

    with pytest.raises(FloorCandidateError, match="ip-assignment"):
        promote_interview_q4_invariants(
            answers, posture_version=1, existing_invariants=[hand_authored]
        )

    # The exception is raised before any mutation -- the caller's own dict
    # is completely untouched (never mutated in place, never replaced).
    assert hand_authored == {
        "id": "ip-assignment",
        "statement": "Never accept present-tense assignment of pre-existing IP.",
        "rationale": "Signed off by the GC 2026-03-01 after board review.",
        "x_signed_by": "gc@example.com",
    }


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
