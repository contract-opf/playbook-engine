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

Also covers issue #90's review-checklist candidate promotion (the OTHER
route into ``floor.invariants``, gated on an explicit human accept decision
— never auto-promotion):

  - :func:`candidate_invariant_id` / :func:`promote_floor_candidate` — a
    single accepted candidate is promoted idempotently, never overwriting a
    foreign (non-self-authored) colliding id.
  - :func:`resolve_floor_candidate_decisions` — pure resolution of a
    ``feedback.json`` ``"floor"`` block: accept/reject/malformed/unknown.
  - :func:`apply_floor_review` — the I/O wrapper that reads and rewrites
    ``floor.candidates.json``.

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
    apply_floor_review,
    candidate_invariant_id,
    candidate_q4_invariant_id,
    derive_interview_q4_candidates,
    derive_reversal_candidates,
    promote_floor_candidate,
    promote_interview_q4_invariants,
    propose_floor_candidates,
    read_floor_candidates,
    resolve_floor_candidate_decisions,
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


def test_write_floor_candidates_preserves_decision_across_repropose(tmp_path: Path) -> None:
    """Issue #90 review finding 4 regression: floor.candidates.json is the
    ONLY record of a rejection (an accepted candidate is separately
    protected by its presence in floor.invariants) -- re-deriving candidates
    with UNCHANGED inputs must not silently reset a previously-recorded
    decision back to undecided."""
    obs_path = tmp_path / "observations.jsonl"
    obs_path.write_text(json.dumps(_reversal_observation()) + "\n", encoding="utf-8")
    doc = _minimal_v02_doc()
    (tmp_path / "playbook.opf.json").write_text(json.dumps(doc), encoding="utf-8")

    write_floor_candidates(tmp_path)
    first = json.loads((tmp_path / "floor.candidates.json").read_text(encoding="utf-8"))
    assert len(first["candidates"]) == 1
    cand_id = first["candidates"][0]["id"]
    statement = first["candidates"][0]["statement"]

    apply_floor_review(tmp_path, {cand_id: {"decision": "reject"}})
    rejected = json.loads((tmp_path / "floor.candidates.json").read_text(encoding="utf-8"))
    assert rejected["candidates"][0]["decision"] == "rejected"

    write_floor_candidates(tmp_path)
    second = json.loads((tmp_path / "floor.candidates.json").read_text(encoding="utf-8"))

    assert len(second["candidates"]) == 1
    assert second["candidates"][0]["statement"] == statement
    assert second["candidates"][0]["decision"] == "rejected"


def test_write_floor_candidates_drops_decision_when_statement_no_longer_recurs(
    tmp_path: Path,
) -> None:
    """A rejected candidate whose underlying evidence disappears on
    re-derivation has nothing to carry its decision to -- it simply drops,
    same as the candidate itself; this is not a resurrection bug."""
    obs_path = tmp_path / "observations.jsonl"
    obs_path.write_text(json.dumps(_reversal_observation()) + "\n", encoding="utf-8")
    doc = _minimal_v02_doc()
    (tmp_path / "playbook.opf.json").write_text(json.dumps(doc), encoding="utf-8")

    write_floor_candidates(tmp_path)
    first = json.loads((tmp_path / "floor.candidates.json").read_text(encoding="utf-8"))
    cand_id = first["candidates"][0]["id"]
    apply_floor_review(tmp_path, {cand_id: {"decision": "reject"}})

    # Evidence disappears entirely.
    obs_path.write_text("", encoding="utf-8")
    write_floor_candidates(tmp_path)
    second = json.loads((tmp_path / "floor.candidates.json").read_text(encoding="utf-8"))
    assert second["candidates"] == []


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


def test_cli_floor_propose_rejected_candidate_stays_rejected_on_second_run(
    tmp_path: Path,
) -> None:
    """Issue #90 review finding 4 regression, via the actual CLI command: a
    rejected candidate must not be resurrected as a fresh, undecided
    proposal by a second `playbook floor propose` run."""
    obs_path = tmp_path / "observations.jsonl"
    obs_path.write_text(json.dumps(_reversal_observation()) + "\n", encoding="utf-8")
    doc = _minimal_v02_doc()
    (tmp_path / "playbook.opf.json").write_text(json.dumps(doc), encoding="utf-8")

    exit_code, output = _invoke("floor", "propose", str(tmp_path))
    assert exit_code == 0, output
    first = json.loads((tmp_path / "floor.candidates.json").read_text(encoding="utf-8"))
    cand_id = first["candidates"][0]["id"]

    apply_floor_review(tmp_path, {cand_id: {"decision": "reject"}})

    exit_code, output = _invoke("floor", "propose", str(tmp_path))
    assert exit_code == 0, output
    second = json.loads((tmp_path / "floor.candidates.json").read_text(encoding="utf-8"))
    assert len(second["candidates"]) == 1
    assert second["candidates"][0]["decision"] == "rejected"


# ---------------------------------------------------------------------------
# candidate_invariant_id / promote_floor_candidate — issue #90
# ---------------------------------------------------------------------------


def _floor_candidate(
    *,
    id: str = "cand-001",  # noqa: A002
    statement: str = "Never accept uncapped liability.",
    rationale: str = "Proposed then reversed before signing in 2 deals.",
    source: str = "reversal",
    citations: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if citations is None:
        citations = [{"document_id": "doc-a", "version": 2, "clause_path": "8.1"}]
    return {
        "id": id,
        "statement": statement,
        "rationale": rationale,
        "source": source,
        "citations": citations,
    }


def test_candidate_invariant_id_is_a_slug_of_the_statement() -> None:
    candidate = _floor_candidate(statement="Never accept uncapped liability.")
    assert candidate_invariant_id(candidate) == "never-accept-uncapped-liability"


def test_candidate_invariant_id_stable_across_calls() -> None:
    candidate = _floor_candidate()
    assert candidate_invariant_id(candidate) == candidate_invariant_id(candidate)


def test_promote_floor_candidate_appends_new_invariant() -> None:
    candidate = _floor_candidate()

    result = promote_floor_candidate(candidate, existing_invariants=[])

    assert len(result) == 1
    inv = result[0]
    assert inv["id"] == "never-accept-uncapped-liability"
    assert inv["statement"] == "Never accept uncapped liability."
    assert "Proposed then reversed before signing in 2 deals." in inv["rationale"]
    assert "doc-a v2 §8.1" in inv["rationale"]
    assert "Accepted via review feedback" in inv["rationale"]
    assert "cand-001" in inv["rationale"]


def test_promote_floor_candidate_interview_q4_source_has_no_evidence_line() -> None:
    """An interview_q4-sourced candidate carries no citations (see
    derive_interview_q4_candidates) — the promoted rationale must not
    fabricate an 'Evidence:' clause for it."""
    candidate = _floor_candidate(
        id="cand-002",
        statement="Never accept IP assignment.",
        rationale='Named as non-negotiable in the Posture interview (Q4 "sacred_clauses").',
        source="interview_q4",
        citations=[],
    )

    result = promote_floor_candidate(candidate, existing_invariants=[])

    assert "Evidence:" not in result[0]["rationale"]
    assert "Accepted via review feedback" in result[0]["rationale"]


def test_promote_floor_candidate_rerun_same_statement_is_true_noop() -> None:
    candidate = _floor_candidate()

    first = promote_floor_candidate(candidate, existing_invariants=[])
    second = promote_floor_candidate(candidate, existing_invariants=first)

    assert second == first
    assert len(second) == 1
    ids = [inv["id"] for inv in second]
    assert len(ids) == len(set(ids))  # OPF-SPEC.md §3.13: no duplicate sibling ids


def test_promote_floor_candidate_rerun_with_changed_statement_updates_in_place() -> None:
    candidate = _floor_candidate(statement="Never accept uncapped liability.")
    first = promote_floor_candidate(candidate, existing_invariants=[])
    assert len(first) == 1

    changed = dict(candidate)
    changed["statement"] = "Never accept uncapped liability of any kind."
    second = promote_floor_candidate(changed, existing_invariants=first)

    # Different statement -> different slug -> a SECOND entry, not an
    # in-place update: candidate_invariant_id is keyed on the statement
    # itself, so an edited statement is, by construction, a different id.
    # The in-place-update branch is exercised by a candidate whose
    # statement is unchanged on rerun (the true-no-op test above) plus the
    # multi-accept-in-one-call case below.
    assert len(second) == 2


def test_promote_floor_candidate_two_accepts_in_sequence_both_present() -> None:
    liability = _floor_candidate(id="cand-001", statement="Never accept uncapped liability.")
    ip = _floor_candidate(
        id="cand-002",
        statement="Never accept IP assignment.",
        citations=[{"document_id": "doc-b", "version": 1, "clause_path": "3"}],
    )

    merged = promote_floor_candidate(liability, existing_invariants=[])
    merged = promote_floor_candidate(ip, existing_invariants=merged)

    assert len(merged) == 2
    ids = {inv["id"] for inv in merged}
    assert ids == {"never-accept-uncapped-liability", "never-accept-ip-assignment"}


def test_promote_floor_candidate_refuses_to_overwrite_colliding_hand_authored_id() -> None:
    """Issue #89 review finding 2's foreign-collision guard, applied to the
    candidate-acceptance path: an existing invariant whose id happens to
    equal this candidate's derived slug, but which was NOT written by an
    earlier acceptance of this SAME candidate, must never be silently
    overwritten."""
    hand_authored = {
        "id": "never-accept-uncapped-liability",
        "statement": "Never accept uncapped liability under any circumstances.",
        "rationale": "Signed off by the GC 2026-03-01 after board review.",
    }
    candidate = _floor_candidate(statement="Never accept uncapped liability.")

    with pytest.raises(FloorCandidateError, match="never-accept-uncapped-liability"):
        promote_floor_candidate(candidate, existing_invariants=[hand_authored])

    # Never mutated in place.
    assert hand_authored == {
        "id": "never-accept-uncapped-liability",
        "statement": "Never accept uncapped liability under any circumstances.",
        "rationale": "Signed off by the GC 2026-03-01 after board review.",
    }


def test_promote_floor_candidate_result_is_schema_shaped() -> None:
    """Only id/statement/rationale — additionalProperties: false in
    spec/playbook.schema-0.3.json's floor.invariants[] item."""
    candidate = _floor_candidate()
    result = promote_floor_candidate(candidate, existing_invariants=[])
    assert set(result[0]) == {"id", "statement", "rationale"}


# ---------------------------------------------------------------------------
# candidate_q4_invariant_id — the Q4 path's OTHER id derivation for the
# same candidate (issue #90 review finding 1)
# ---------------------------------------------------------------------------


def test_candidate_q4_invariant_id_derives_slug_of_underlying_item() -> None:
    candidate = _floor_candidate(
        statement="Never accept liability caps.", source="interview_q4", citations=[]
    )
    assert candidate_q4_invariant_id(candidate) == "liability-caps"


def test_candidate_q4_invariant_id_matches_promote_interview_q4_invariants_id() -> None:
    """The whole point: this id must equal the id
    promote_interview_q4_invariants (issue #89) would derive for the SAME
    item, even though candidate_invariant_id (slugging this candidate's own
    statement) never would."""
    item = "Liability caps and student-data protection"
    candidate = _floor_candidate(
        statement=f"Never accept {item}.", source="interview_q4", citations=[]
    )
    promoted = promote_interview_q4_invariants(
        {"sacred_clauses": item}, posture_version=1, existing_invariants=[]
    )

    assert candidate_q4_invariant_id(candidate) == promoted[0]["id"]
    assert candidate_invariant_id(candidate) != promoted[0]["id"]  # the OTHER id disagrees


def test_candidate_q4_invariant_id_none_for_reversal_source() -> None:
    candidate = _floor_candidate(statement="Never accept uncapped liability.", source="reversal")
    assert candidate_q4_invariant_id(candidate) is None


def test_candidate_q4_invariant_id_none_when_statement_does_not_match_q4_shape() -> None:
    candidate = _floor_candidate(
        statement="This does not follow the draft shape", source="interview_q4", citations=[]
    )
    assert candidate_q4_invariant_id(candidate) is None


# ---------------------------------------------------------------------------
# promote_floor_candidate refuses a Q4 item already signed under its OTHER
# id (issue #90 review finding 1)
# ---------------------------------------------------------------------------


def test_promote_floor_candidate_refuses_when_q4_item_already_signed_under_other_id() -> None:
    """An interview_q4-sourced candidate's underlying item may already be
    present in floor.invariants under promote_interview_q4_invariants's OWN,
    differently-derived id (_slugify_statement_item, not
    candidate_invariant_id) and differently-worded statement ("Do not
    concede on X." vs this candidate's draft "Never accept X."). Accepting
    the candidate must never append a second, opposite-polarity invariant
    for the same item -- it must be refused, the same as any other foreign
    id collision."""
    item = "liability caps"
    existing = promote_interview_q4_invariants(
        {"sacred_clauses": item}, posture_version=1, existing_invariants=[]
    )
    candidate = _floor_candidate(
        id="cand-001", statement=f"Never accept {item}.", source="interview_q4", citations=[]
    )

    with pytest.raises(FloorCandidateError, match="liability-caps"):
        promote_floor_candidate(candidate, existing_invariants=existing)

    # Never mutated: still exactly the one Q4-promoted entry, byte-identical.
    assert existing == promote_interview_q4_invariants(
        {"sacred_clauses": item}, posture_version=1, existing_invariants=[]
    )
    assert len(existing) == 1


def test_promote_floor_candidate_reversal_source_unaffected_by_q4_guard() -> None:
    """The new Q4 cross-id guard must never fire for a source: reversal
    candidate -- candidate_q4_invariant_id is None for those, so an
    unrelated Q4-promoted invariant sharing no id with it must not block a
    perfectly normal reversal-candidate acceptance."""
    q4_invariants = promote_interview_q4_invariants(
        {"sacred_clauses": "liability caps"}, posture_version=1, existing_invariants=[]
    )
    candidate = _floor_candidate(statement="Never accept uncapped liability.", source="reversal")

    result = promote_floor_candidate(candidate, existing_invariants=q4_invariants)

    assert len(result) == 2
    ids = {inv["id"] for inv in result}
    assert "liability-caps" in ids
    assert "never-accept-uncapped-liability" in ids


# ---------------------------------------------------------------------------
# resolve_floor_candidate_decisions — pure feedback.json "floor" resolution
# (issue #90)
# ---------------------------------------------------------------------------


def test_resolve_floor_decisions_accept_promotes_and_marks_candidate() -> None:
    candidate = _floor_candidate()
    result = resolve_floor_candidate_decisions(
        {"cand-001": {"decision": "accept"}}, [candidate], existing_invariants=[]
    )

    assert result.promoted == ["cand-001"]
    assert result.rejected == []
    assert result.skipped == {}
    assert result.invariants_changed is True
    assert result.invariants[0]["id"] == "never-accept-uncapped-liability"
    assert result.candidates_changed is True
    assert result.candidates[0]["decision"] == "accepted"


def test_resolve_floor_decisions_reject_marks_candidate_never_touches_invariants() -> None:
    candidate = _floor_candidate()
    result = resolve_floor_candidate_decisions(
        {"cand-001": {"decision": "reject"}}, [candidate], existing_invariants=[]
    )

    assert result.rejected == ["cand-001"]
    assert result.promoted == []
    assert result.invariants == []
    assert result.invariants_changed is False
    assert result.candidates_changed is True
    assert result.candidates[0]["decision"] == "rejected"


def test_resolve_floor_decisions_malformed_top_level_block_skipped() -> None:
    result = resolve_floor_candidate_decisions(
        ["not", "an", "object"], [_floor_candidate()], existing_invariants=[]
    )

    assert "floor" in result.skipped
    assert result.promoted == []
    assert result.rejected == []
    assert result.invariants_changed is False
    assert result.candidates_changed is False


def test_resolve_floor_decisions_malformed_entry_skipped() -> None:
    result = resolve_floor_candidate_decisions(
        {"cand-001": "accept"},  # not an object
        [_floor_candidate()],
        existing_invariants=[],
    )

    assert "floor:cand-001" in result.skipped
    assert result.promoted == []


def test_resolve_floor_decisions_unknown_decision_value_skipped() -> None:
    result = resolve_floor_candidate_decisions(
        {"cand-001": {"decision": "maybe"}}, [_floor_candidate()], existing_invariants=[]
    )

    assert "floor:cand-001" in result.skipped
    assert result.promoted == []
    assert result.rejected == []


def test_resolve_floor_decisions_missing_decision_key_skipped() -> None:
    result = resolve_floor_candidate_decisions(
        {"cand-001": {"comment": "no decision given"}}, [_floor_candidate()], existing_invariants=[]
    )

    assert "floor:cand-001" in result.skipped
    # Issue #90 review finding 3 regression: a comment-only entry (no
    # "decision" key at all -- exactly the shape the page's Export JS
    # produces when a reviewer types a note and leaves the radio on
    # Undecided -- must still be captured, independent of the
    # missing-decision skip message above.
    assert result.comments == [
        ("cand-001", "Never accept uncapped liability.", "no decision given")
    ]
    assert result.promoted == []
    assert result.rejected == []
    assert result.candidates_changed is False


def test_resolve_floor_decisions_unknown_candidate_id_skipped() -> None:
    result = resolve_floor_candidate_decisions(
        {"cand-999": {"decision": "accept"}}, [_floor_candidate()], existing_invariants=[]
    )

    assert "floor:cand-999" in result.skipped
    assert result.promoted == []


def test_resolve_floor_decisions_unsupported_key_inside_entry_skipped() -> None:
    result = resolve_floor_candidate_decisions(
        {"cand-001": {"decision": "accept", "bogus_key": "x"}},
        [_floor_candidate()],
        existing_invariants=[],
    )

    assert "floor:cand-001" in result.skipped
    assert any("bogus_key" in msg for msg in result.skipped["floor:cand-001"])
    # The recognised "decision" key is still honored alongside the report.
    assert result.promoted == ["cand-001"]


def test_resolve_floor_decisions_comment_captured_for_notes() -> None:
    result = resolve_floor_candidate_decisions(
        {"cand-001": {"decision": "reject", "comment": "too broad as worded"}},
        [_floor_candidate()],
        existing_invariants=[],
    )

    assert result.comments == [
        ("cand-001", "Never accept uncapped liability.", "too broad as worded")
    ]


def test_resolve_floor_decisions_foreign_collision_skipped_not_raised() -> None:
    """A collision with a hand-authored (non-self) invariant is reported via
    skipped, not raised — one bad floor entry must not abort resolution of
    the rest of a feedback.json (issue #138 discipline)."""
    hand_authored = {
        "id": "never-accept-uncapped-liability",
        "statement": "Never accept uncapped liability, ever.",
        "rationale": "Signed off by the GC.",
    }
    result = resolve_floor_candidate_decisions(
        {"cand-001": {"decision": "accept"}},
        [_floor_candidate()],
        existing_invariants=[hand_authored],
    )

    assert "floor:cand-001" in result.skipped
    assert result.promoted == []
    assert result.invariants == [hand_authored]
    assert result.invariants_changed is False
    # The candidate's own decision is never marked "accepted" when the
    # promotion itself failed — a reviewer sees an accurate, unresolved state.
    assert result.candidates_changed is False


def test_resolve_floor_decisions_reapply_same_decision_is_idempotent() -> None:
    candidate = _floor_candidate()
    first = resolve_floor_candidate_decisions(
        {"cand-001": {"decision": "accept"}}, [candidate], existing_invariants=[]
    )

    # Second call starts from the first call's own outputs — the same shape
    # apply_floor_review's caller would thread through on a second apply.
    second = resolve_floor_candidate_decisions(
        {"cand-001": {"decision": "accept"}},
        first.candidates,
        existing_invariants=first.invariants,
    )

    assert second.promoted == ["cand-001"]  # still reported, even though...
    assert second.invariants_changed is False  # ...nothing actually changed
    assert second.candidates_changed is False  # decision was already "accepted"
    assert second.invariants == first.invariants


def test_resolve_floor_decisions_accept_one_reject_another() -> None:
    liability = _floor_candidate(id="cand-001", statement="Never accept uncapped liability.")
    ip = _floor_candidate(id="cand-002", statement="Never accept IP assignment.", citations=[])

    result = resolve_floor_candidate_decisions(
        {"cand-001": {"decision": "accept"}, "cand-002": {"decision": "reject"}},
        [liability, ip],
        existing_invariants=[],
    )

    assert result.promoted == ["cand-001"]
    assert result.rejected == ["cand-002"]
    assert len(result.invariants) == 1
    by_id = {c["id"]: c for c in result.candidates}
    assert by_id["cand-001"]["decision"] == "accepted"
    assert by_id["cand-002"]["decision"] == "rejected"


# ---------------------------------------------------------------------------
# apply_floor_review — I/O wrapper (issue #90)
# ---------------------------------------------------------------------------


def test_apply_floor_review_reads_and_rewrites_candidates_json(tmp_path: Path) -> None:
    candidates_path = tmp_path / "floor.candidates.json"
    candidates_path.write_text(json.dumps({"candidates": [_floor_candidate()]}), encoding="utf-8")

    result = apply_floor_review(
        tmp_path, {"cand-001": {"decision": "accept"}}, existing_invariants=[]
    )

    assert result.promoted == ["cand-001"]
    on_disk = json.loads(candidates_path.read_text(encoding="utf-8"))
    assert on_disk["candidates"][0]["decision"] == "accepted"


def test_apply_floor_review_no_op_when_nothing_changed(tmp_path: Path) -> None:
    """Second identical apply doesn't even rewrite the file (byte-identical
    mtime-preserving no-op), proving the idempotency the ticket requires."""
    candidates_path = tmp_path / "floor.candidates.json"
    candidates_path.write_text(json.dumps({"candidates": [_floor_candidate()]}), encoding="utf-8")

    first = apply_floor_review(
        tmp_path, {"cand-001": {"decision": "reject"}}, existing_invariants=[]
    )
    assert first.candidates_changed is True
    bytes_after_first = candidates_path.read_bytes()

    second = apply_floor_review(
        tmp_path, {"cand-001": {"decision": "reject"}}, existing_invariants=[]
    )
    assert second.candidates_changed is False
    assert candidates_path.read_bytes() == bytes_after_first


def test_apply_floor_review_never_writes_playbook_opf_json(tmp_path: Path) -> None:
    """apply_floor_review only ever touches floor.candidates.json — writing
    playbook.opf.json (with the identity refresh curation pins also need)
    is exclusively viewer.apply_feedback's job."""
    candidates_path = tmp_path / "floor.candidates.json"
    candidates_path.write_text(json.dumps({"candidates": [_floor_candidate()]}), encoding="utf-8")

    apply_floor_review(tmp_path, {"cand-001": {"decision": "accept"}}, existing_invariants=[])

    assert not (tmp_path / "playbook.opf.json").exists()


def test_apply_floor_review_missing_candidates_file_reports_unknown(tmp_path: Path) -> None:
    result = apply_floor_review(
        tmp_path, {"cand-001": {"decision": "accept"}}, existing_invariants=[]
    )

    assert "floor:cand-001" in result.skipped
    assert result.candidates_changed is False


def test_read_floor_candidates_missing_file_returns_empty(tmp_path: Path) -> None:
    assert read_floor_candidates(tmp_path) == []


def test_read_floor_candidates_malformed_json_returns_empty(tmp_path: Path) -> None:
    (tmp_path / "floor.candidates.json").write_text("not json", encoding="utf-8")
    assert read_floor_candidates(tmp_path) == []


def test_read_floor_candidates_non_utf8_file_returns_empty(tmp_path: Path) -> None:
    """Issue #90 review finding 6 regression: a non-UTF-8 sidecar must
    degrade to 'no candidates' the same as malformed JSON, not raise
    UnicodeDecodeError past this function and abort the whole page render."""
    (tmp_path / "floor.candidates.json").write_bytes(b"\xff\xfe\x00\x01garbage")
    assert read_floor_candidates(tmp_path) == []


def test_read_floor_candidates_returns_candidates_list(tmp_path: Path) -> None:
    candidate = _floor_candidate()
    (tmp_path / "floor.candidates.json").write_text(
        json.dumps({"candidates": [candidate]}), encoding="utf-8"
    )
    assert read_floor_candidates(tmp_path) == [candidate]
