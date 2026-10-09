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

The review-checklist accept/reject path (``view apply``'s ``"floor"`` block,
issue #90) was retired with the review HTML (issue #239): a candidate is
accepted by recording its hard line with ``playbook floor sign``.

SECURITY NOTE: All fixtures are synthetic, minimal dicts — no real legal text,
no real parties.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from click.testing import CliRunner

from playbook_engine.canonicalize import compute_section_digests, content_hash
from playbook_engine.cli import cli
from playbook_engine.floor_candidates import (
    _Q5_REJECTION_COMMENT,
    FloorCandidateError,
    count_below_min_deals_reversals,
    count_structural_reversals_omitted,
    derive_interview_q4_candidates,
    derive_reversal_candidates,
    is_q4_item_sentence_shaped,
    promote_interview_q4_invariants,
    propose_floor_candidates,
    q4_q5_contradictions,
    q4_sentence_shaped_items,
    sign_floor_invariant,
    sign_invariant_id,
    write_floor_candidates,
)
from playbook_engine.validator import validate_document

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


def _minimal_doc(**overrides: Any) -> dict[str, Any]:
    doc: dict[str, Any] = {
        "opf_version": "0.5",
        "agreement_type": {"id": "test-agreement", "name": "Test Agreement"},
        "baseline": {"has_canonical_template": False},
        "taxonomy": {"source": "custom", "entries": []},
        "evidence": {"clauses": [], "precedent": []},
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
    assert cand.statement.startswith("Do not concede on")
    assert len(cand.citations) >= 1
    cite = cand.citations[0]
    assert cite.document_id == "doc-a"
    assert cite.version == 2
    assert cite.clause_path == "8.1"


def test_reversal_candidate_carries_taxonomy_id() -> None:
    """Issue #102: the group key (the raw taxonomy_id, not the humanized
    prose it gets flattened into for the statement) survives onto the
    candidate itself, so a later suppression check can match on it."""
    observations = [_reversal_observation(taxonomy_id="uncapped_liability")]

    candidates = derive_reversal_candidates(observations)

    assert candidates[0].taxonomy_id == "uncapped_liability"


def test_interview_q4_candidate_has_no_taxonomy_id() -> None:
    """A Q4-named item carries no clause taxonomy — taxonomy_id stays None."""
    candidates = derive_interview_q4_candidates({"sacred_clauses": "liability caps"})
    assert candidates[0].taxonomy_id is None


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


def test_reversal_unclassified_observations_are_excluded() -> None:
    """An UNCLASSIFIED reversal is not a proposable hard line.

    Superseded `test_reversal_unclassified_observations_do_not_collapse`,
    which asserted each unclassified reversal became its own candidate. That
    was correct about not COLLAPSING them, but wrong about surfacing them at
    all: with no `taxonomy_id` the statement is built by quoting the raw
    clause text, so the reviewer is asked to sign hard lines reading `Never
    accept "shall".`, `Never accept "3 3".`, `Never accept "4 1".` —
    segmentation debris, not legal positions. Measured on a real corpus:
    67 of 530 reversals were unclassified and produced 67 of the 90 reversal
    candidates, so three-quarters of the checklist was noise, against
    OPF-SPEC.md §3.7.1's "keep the Floor minimal".

    They are counted and reported (see
    `test_write_floor_candidates_reports_unclassified_omitted_count`), never
    silently dropped.
    """
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

    assert candidates == []


def test_reversal_classified_still_yields_candidate_alongside_unclassified() -> None:
    """Excluding the unclassified ones must not drop the real one beside them."""
    observations = [
        _reversal_observation(observation_id="doc-a/2/8.1"),
        _reversal_observation(
            observation_id="doc-a/2/9.1",
            taxonomy_id=None,
            clause_path="9.1",
            full_text="3 3",
        ),
    ]

    candidates = derive_reversal_candidates(observations)

    assert len(candidates) == 1
    assert '"' not in candidates[0].statement  # never a quoted raw fragment


def test_reversal_statement_is_not_inverted_into_a_prohibition() -> None:
    """`Never accept governing law.` tells a Floor judge to reject any clause
    CONTAINING governing law — backwards, and the exact inversion issue #89
    already fixed on the promoted path (`_q4_promoted_statement`). A reversal
    candidate means "we asked for this and backed down", so the hard line is
    "don't back down", not "reject the clause".
    """
    cand = derive_reversal_candidates([_reversal_observation()])[0]

    assert not cand.statement.startswith("Never accept")
    assert cand.statement.startswith("Do not concede on")


# ---------------------------------------------------------------------------
# Reversal noise filters (issue #106): structural taxonomy + --min-deals
# ---------------------------------------------------------------------------


def test_reversal_structural_taxonomy_excluded() -> None:
    """A reversal classified under a taxonomy entry curated `structural:
    true` (issue #106) is excluded exactly like an unclassified one — it is
    administrative/boilerplate framing, not a proposable hard line."""
    observations = [_reversal_observation(taxonomy_id="parties_and_recitals")]

    candidates = derive_reversal_candidates(
        observations, structural_ids=frozenset({"parties_and_recitals"})
    )

    assert candidates == []


def test_reversal_structural_exclusion_does_not_drop_non_structural_siblings() -> None:
    """Excluding a structural taxonomy_id must not drop an unrelated,
    non-structural candidate derived alongside it."""
    observations = [
        _reversal_observation(observation_id="doc-a/2/8.1", taxonomy_id="uncapped_liability"),
        _reversal_observation(
            observation_id="doc-a/2/9.1",
            taxonomy_id="parties_and_recitals",
            clause_path="9.1",
        ),
    ]

    candidates = derive_reversal_candidates(
        observations, structural_ids=frozenset({"parties_and_recitals"})
    )

    assert len(candidates) == 1
    assert candidates[0].taxonomy_id == "uncapped_liability"


def test_count_structural_reversals_omitted() -> None:
    observations = [
        _reversal_observation(observation_id="doc-a/2/8.1", taxonomy_id="parties_and_recitals"),
        _reversal_observation(
            observation_id="doc-a/2/9.1", taxonomy_id="parties_and_recitals", clause_path="9.1"
        ),
        _reversal_observation(observation_id="doc-a/2/10.1", taxonomy_id="uncapped_liability"),
    ]

    assert (
        count_structural_reversals_omitted(observations, frozenset({"parties_and_recitals"})) == 2
    )


def test_reversal_below_min_deals_dropped() -> None:
    """A reversal cited by only 1 distinct document is a plausible fluke,
    not a corroborated pattern -- dropped when min_deals=2."""
    observations = [_reversal_observation()]  # single document ("doc-a")

    candidates = derive_reversal_candidates(observations, min_deals=2)

    assert candidates == []


def test_reversal_min_deals_default_is_unfiltered() -> None:
    """derive_reversal_candidates' own default (min_deals=1) must not
    filter -- every pre-#106 call site (and write_floor_candidates' own
    default) is unaffected; only the CLI raises the threshold to 2."""
    observations = [_reversal_observation()]

    candidates = derive_reversal_candidates(observations)

    assert len(candidates) == 1


def test_reversal_min_deals_boundary_exactly_kept() -> None:
    """Exactly `min_deals` distinct documents citing a group is KEPT, not
    dropped -- the threshold is a floor, not a strict minimum (mutation
    gate: an off-by-one here, `<=` instead of `<`, must fail this test)."""
    observations = [
        _reversal_observation(observation_id="doc-a/2/8.1", document_id="doc-a"),
        _reversal_observation(
            observation_id="doc-b/3/9.1", document_id="doc-b", version=3, clause_path="9.1"
        ),
    ]

    candidates = derive_reversal_candidates(observations, min_deals=2)

    assert len(candidates) == 1


def test_count_below_min_deals_reversals() -> None:
    observations = [
        _reversal_observation(),  # 1 doc, below threshold of 2
        _reversal_observation(
            observation_id="doc-x/2/1.1",
            taxonomy_id="cap_on_liability",
            document_id="doc-x",
            clause_path="1.1",
        ),
        _reversal_observation(
            observation_id="doc-y/2/1.1",
            taxonomy_id="cap_on_liability",
            document_id="doc-y",
            clause_path="1.1",
        ),  # 2 docs, meets threshold
    ]

    assert count_below_min_deals_reversals(observations, min_deals=2) == 1


def test_propose_floor_candidates_threads_structural_and_min_deals() -> None:
    observations = [_reversal_observation(taxonomy_id="parties_and_recitals")]

    result = propose_floor_candidates(
        observations, structural_ids=frozenset({"parties_and_recitals"})
    )
    assert result["candidates"] == []

    result2 = propose_floor_candidates([_reversal_observation()], min_deals=2)
    assert result2["candidates"] == []


def test_write_floor_candidates_structural_omitted_sibling(tmp_path: Path) -> None:
    obs_path = tmp_path / "observations.jsonl"
    obs_path.write_text(
        json.dumps(_reversal_observation(taxonomy_id="parties_and_recitals")) + "\n",
        encoding="utf-8",
    )

    out_path = write_floor_candidates(tmp_path, structural_ids=frozenset({"parties_and_recitals"}))
    written = json.loads(out_path.read_text(encoding="utf-8"))

    assert written["candidates"] == []
    assert written["structural_reversals_omitted"] == 1
    assert written["below_min_deals_omitted"] == 0


def test_write_floor_candidates_below_min_deals_sibling(tmp_path: Path) -> None:
    obs_path = tmp_path / "observations.jsonl"
    obs_path.write_text(json.dumps(_reversal_observation()) + "\n", encoding="utf-8")

    out_path = write_floor_candidates(tmp_path, min_deals=2)
    written = json.loads(out_path.read_text(encoding="utf-8"))

    assert written["candidates"] == []
    assert written["below_min_deals_omitted"] == 1
    assert written["structural_reversals_omitted"] == 0


def test_cli_floor_propose_min_deals_default_drops_single_document(tmp_path: Path) -> None:
    """The CLI's own default (--min-deals 2) drops a single-document
    reversal, even though the underlying write_floor_candidates() function's
    own default (1) would not."""
    obs_path = tmp_path / "observations.jsonl"
    obs_path.write_text(json.dumps(_reversal_observation()) + "\n", encoding="utf-8")

    exit_code, output = _invoke("floor", "propose", str(tmp_path))

    assert exit_code == 0, output
    written = json.loads((tmp_path / "floor.candidates.json").read_text(encoding="utf-8"))
    assert written["candidates"] == []
    assert written["below_min_deals_omitted"] == 1


def test_cli_floor_propose_min_deals_flag_keeps_single_document(tmp_path: Path) -> None:
    obs_path = tmp_path / "observations.jsonl"
    obs_path.write_text(json.dumps(_reversal_observation()) + "\n", encoding="utf-8")

    exit_code, output = _invoke("floor", "propose", str(tmp_path), "--min-deals", "1")

    assert exit_code == 0, output
    written = json.loads((tmp_path / "floor.candidates.json").read_text(encoding="utf-8"))
    assert len(written["candidates"]) == 1
    assert written["below_min_deals_omitted"] == 0


def test_cli_floor_propose_min_deals_boundary_exactly_two_kept(tmp_path: Path) -> None:
    obs_path = tmp_path / "observations.jsonl"
    obs_path.write_text(
        "\n".join(
            json.dumps(o)
            for o in (
                _reversal_observation(observation_id="doc-a/2/8.1", document_id="doc-a"),
                _reversal_observation(
                    observation_id="doc-b/3/9.1",
                    document_id="doc-b",
                    version=3,
                    clause_path="9.1",
                ),
            )
        )
        + "\n",
        encoding="utf-8",
    )

    exit_code, output = _invoke("floor", "propose", str(tmp_path))

    assert exit_code == 0, output
    written = json.loads((tmp_path / "floor.candidates.json").read_text(encoding="utf-8"))
    assert len(written["candidates"]) == 1
    assert written["below_min_deals_omitted"] == 0


def test_cli_floor_propose_config_excludes_structural_candidate(tmp_path: Path) -> None:
    """--config supplies the taxonomy that curates `structural: true` —
    without it, structural exclusion is skipped entirely (empty set)."""
    obs_path = tmp_path / "observations.jsonl"
    obs_path.write_text(
        "\n".join(
            json.dumps(o)
            for o in (
                _reversal_observation(observation_id="doc-a/2/8.1", document_id="doc-a"),
                _reversal_observation(
                    observation_id="doc-b/3/8.1",
                    document_id="doc-b",
                    version=3,
                    taxonomy_id="parties_and_recitals",
                ),
                _reversal_observation(
                    observation_id="doc-c/2/8.1",
                    document_id="doc-c",
                    taxonomy_id="parties_and_recitals",
                ),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    taxonomy_path = tmp_path / "taxonomy.yaml"
    taxonomy_path.write_text(
        yaml.dump(
            {
                "source": "test",
                "entries": [
                    {
                        "id": "uncapped_liability",
                        "label": "Uncapped Liability",
                        "status": "active",
                        "cuad_origin": None,
                    },
                    {
                        "id": "parties_and_recitals",
                        "label": "Parties & Recitals",
                        "status": "active",
                        "cuad_origin": None,
                        "structural": True,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.dump(
            {
                "agreement_type": {"id": "test-agreement", "name": "Test Agreement"},
                "baseline": {"template": None},
                "taxonomy": str(taxonomy_path),
            }
        ),
        encoding="utf-8",
    )

    exit_code, output = _invoke("floor", "propose", str(tmp_path), "--config", str(config_path))

    assert exit_code == 0, output
    written = json.loads((tmp_path / "floor.candidates.json").read_text(encoding="utf-8"))
    # doc-a's uncapped_liability is single-document and dropped by the CLI's
    # default --min-deals 2; doc-b/doc-c's parties_and_recitals is structural
    # and excluded by --config. Nothing should survive.
    assert written["candidates"] == []
    assert all(c.get("taxonomy_id") != "parties_and_recitals" for c in written["candidates"])
    assert written["structural_reversals_omitted"] == 2


def test_q4_candidate_statement_is_not_inverted_into_a_prohibition() -> None:
    """Same inversion, same fix, on the Q4 candidate draft."""
    cands = derive_interview_q4_candidates({"sacred_clauses": "uncapped liability"})

    assert not cands[0].statement.startswith("Never accept")
    assert cands[0].statement.startswith("Do not concede on")


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
    # (document_renderer.py tolerates this shape) —
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
# is_q4_item_sentence_shaped / q4_sentence_shaped_items / sentence-shaped
# skip in promote_interview_q4_invariants (issue #104)
# ---------------------------------------------------------------------------

_SENTENCE_SHAPED_ITEM = (
    "limitation of liability, if present, must not be unilateral in the counterparty's favor"
)


def test_is_q4_item_sentence_shaped_flags_conditional_prose() -> None:
    assert is_q4_item_sentence_shaped(_SENTENCE_SHAPED_ITEM) is True


def test_is_q4_item_sentence_shaped_leaves_bare_names_alone() -> None:
    assert is_q4_item_sentence_shaped("Indemnification") is False
    assert is_q4_item_sentence_shaped("uncapped liability") is False
    assert is_q4_item_sentence_shaped("IP assignment") is False


def test_is_q4_item_sentence_shaped_word_count_boundary_is_exclusive() -> None:
    """Issue #104 reviewer gate: a long-but-legitimate clause-TYPE name —
    exactly 7 words, no conditional marker — must stay name-shaped. The
    heuristic's word-count threshold is ``> 7``, not ``>= 7``."""
    seven_word_name = "Limitation of liability and consequential damages waiver"
    assert len(seven_word_name.split()) == 7  # guards the boundary this test probes
    assert is_q4_item_sentence_shaped(seven_word_name) is False

    eight_word_name = seven_word_name + " clause"
    assert len(eight_word_name.split()) == 8
    assert is_q4_item_sentence_shaped(eight_word_name) is True


def test_is_q4_item_sentence_shaped_catches_each_marker_word() -> None:
    assert is_q4_item_sentence_shaped("Governing law unless otherwise agreed") is True
    assert is_q4_item_sentence_shaped("Arbitration shall be binding") is True
    assert is_q4_item_sentence_shaped("Confidentiality provided the deal closes") is True


def test_q4_sentence_shaped_items_returns_only_flagged_items_in_order() -> None:
    answer = f"Indemnification; {_SENTENCE_SHAPED_ITEM}; IP assignment"
    assert q4_sentence_shaped_items({"sacred_clauses": answer}) == [_SENTENCE_SHAPED_ITEM]


def test_q4_sentence_shaped_items_empty_when_nothing_flagged() -> None:
    assert q4_sentence_shaped_items({"sacred_clauses": "uncapped liability; IP assignment"}) == []
    assert q4_sentence_shaped_items(None) == []
    assert q4_sentence_shaped_items({}) == []
    assert q4_sentence_shaped_items({"sacred_clauses": "   "}) == []


def test_promote_q4_skips_sentence_shaped_item_entirely() -> None:
    """Fail-first (issue #104): a sentence-shaped item must never be
    templated or promoted into floor.invariants."""
    result = promote_interview_q4_invariants(
        {"sacred_clauses": _SENTENCE_SHAPED_ITEM}, posture_version=1, existing_invariants=[]
    )

    assert result == []  # neither templated nor promoted -- true skip, not an append


def test_promote_q4_mixed_answer_promotes_name_shaped_skips_sentence_shaped() -> None:
    """Fail-first (issue #104): a mixed answer promotes the name-shaped
    item as before and skips the sentence-shaped one, without raising."""
    answer = f"Indemnification; {_SENTENCE_SHAPED_ITEM}"

    result = promote_interview_q4_invariants(
        {"sacred_clauses": answer}, posture_version=1, existing_invariants=[]
    )

    assert len(result) == 1
    assert result[0]["id"] == "indemnification"
    assert result[0]["statement"] == "Do not concede on Indemnification."


def test_promote_q4_pure_name_answer_is_byte_identical_to_pre_fix_behavior() -> None:
    """Regression (issue #104): an answer made entirely of name-shaped
    items is unaffected by the sentence-shaped skip."""
    answers = {"sacred_clauses": "uncapped liability; IP assignment"}

    result = promote_interview_q4_invariants(answers, posture_version=1, existing_invariants=[])

    assert len(result) == 2
    ids = {inv["id"] for inv in result}
    assert ids == {"uncapped-liability", "ip-assignment"}


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
# taxonomy_id — additive key on FloorCandidate / floor.candidates.json
# (issue #102)
# ---------------------------------------------------------------------------


def test_propose_floor_candidates_reversal_carries_taxonomy_id() -> None:
    """Renumbering (propose_floor_candidates rebuilds each FloorCandidate to
    assign a stable cand-NNN id) must not drop taxonomy_id along the way."""
    observations = [_reversal_observation(taxonomy_id="uncapped_liability")]

    result = propose_floor_candidates(observations, None)

    assert result["candidates"][0]["taxonomy_id"] == "uncapped_liability"


def test_propose_floor_candidates_interview_q4_has_no_taxonomy_id_key() -> None:
    """Additive key omitted entirely (not written as null) when absent —
    byte-identical to pre-#102 output for every interview_q4 candidate."""
    result = propose_floor_candidates([], {"sacred_clauses": "IP assignment"})
    assert "taxonomy_id" not in result["candidates"][0]


def test_write_floor_candidates_round_trips_taxonomy_id(tmp_path: Path) -> None:
    """write_floor_candidates persists taxonomy_id into floor.candidates.json."""
    obs_path = tmp_path / "observations.jsonl"
    obs_path.write_text(
        json.dumps(_reversal_observation(taxonomy_id="limitation_of_liability")) + "\n",
        encoding="utf-8",
    )

    write_floor_candidates(tmp_path)
    candidates = json.loads((tmp_path / "floor.candidates.json").read_text(encoding="utf-8"))[
        "candidates"
    ]

    assert len(candidates) == 1
    assert candidates[0]["taxonomy_id"] == "limitation_of_liability"


# ---------------------------------------------------------------------------
# Interview-Q5 ("flexible_clauses") auto-rejection (issue #105)
# ---------------------------------------------------------------------------


def test_q5_matching_taxonomy_auto_rejects_with_attribution() -> None:
    """A REVERSAL candidate whose taxonomy_id normalizes equal to a Q5 item
    is auto-rejected with the attributing comment."""
    observations = [_reversal_observation(taxonomy_id="renewal_notice")]
    answers = {"flexible_clauses": "renewal notice"}

    result = propose_floor_candidates(observations, answers)

    candidate = result["candidates"][0]
    assert candidate["decision"] == "rejected"
    assert candidate["comment"] == (
        "Posture interview Q5 (flexible_clauses): named as a willing concession."
    )


def test_q5_non_matching_items_are_inert() -> None:
    """A Q5 item that names a DIFFERENT clause type leaves the candidate
    pending -- undecided, no comment."""
    observations = [_reversal_observation(taxonomy_id="uncapped_liability")]
    answers = {"flexible_clauses": "renewal notice; IP assignment"}

    result = propose_floor_candidates(observations, answers)

    candidate = result["candidates"][0]
    assert "decision" not in candidate
    assert "comment" not in candidate


def test_q5_multiple_items_semicolon_split() -> None:
    """Q5 splits on ';' exactly like Q4 (_q4_items) -- each named item
    independently matches its own candidate."""
    observations = [
        _reversal_observation(taxonomy_id="renewal_notice", observation_id="doc-a/2/1"),
        _reversal_observation(taxonomy_id="ip_assignment", observation_id="doc-a/2/2"),
        _reversal_observation(taxonomy_id="uncapped_liability", observation_id="doc-a/2/3"),
    ]
    answers = {"flexible_clauses": "renewal notice; ip assignment"}

    result = propose_floor_candidates(observations, answers)
    by_taxonomy = {c["taxonomy_id"]: c for c in result["candidates"]}

    assert by_taxonomy["renewal_notice"]["decision"] == "rejected"
    assert by_taxonomy["ip_assignment"]["decision"] == "rejected"
    assert "decision" not in by_taxonomy["uncapped_liability"]


def test_q4_promotion_path_unaffected_by_q5_rejection() -> None:
    """Q5 rejection only ever touches source: reversal candidates -- an
    interview_q4 candidate (no taxonomy_id at all) is never marked rejected
    by this mechanism, even if its item text happens to also appear in Q5."""
    answers = {
        "sacred_clauses": "IP assignment",
        "flexible_clauses": "renewal notice",
    }

    result = propose_floor_candidates([], answers)

    q4_candidate = result["candidates"][0]
    assert q4_candidate["source"] == "interview_q4"
    assert "decision" not in q4_candidate


def test_q5_match_requires_exact_normalized_equality_not_substring() -> None:
    """Mutation-guard: a Q5 item that is merely a SUBSTRING/superstring of a
    taxonomy label must never match -- only exact normalized-slug equality.
    "renewal" must not match the (broader) "renewal_notice_period" taxonomy,
    and "auto renewal option" must not match the (narrower) "renewal"
    taxonomy."""
    observations = [_reversal_observation(taxonomy_id="renewal_notice_period")]
    answers = {"flexible_clauses": "renewal"}

    result = propose_floor_candidates(observations, answers)
    assert "decision" not in result["candidates"][0]

    observations2 = [_reversal_observation(taxonomy_id="renewal")]
    answers2 = {"flexible_clauses": "auto renewal option"}

    result2 = propose_floor_candidates(observations2, answers2)
    assert "decision" not in result2["candidates"][0]


def test_q5_rejection_never_touches_floor_invariants() -> None:
    """Spec note (OPF-SPEC.md §3.7 rule 4): the Q5 auto-rejection is a
    rejection of a PROPOSAL, never a Floor promotion -- propose_floor_candidates
    never writes floor.invariants at all, and this candidate carries no
    invariant-shaped output."""
    observations = [_reversal_observation(taxonomy_id="renewal_notice")]
    answers = {"flexible_clauses": "renewal notice"}

    result = propose_floor_candidates(observations, answers)

    assert "floor" not in result
    assert "invariants" not in result


def test_q5_rejection_is_rederived_every_run_and_stays_on_the_reversal_candidate(
    tmp_path: Path,
) -> None:
    """The Q5 auto-rejection is derived from the interview answer on every
    run — nothing is carried over from a prior floor.candidates.json. A
    contradictory interview (the same clause type named in BOTH Q4 and Q5)
    makes the reversal candidate and the interview_q4 candidate derive the
    IDENTICAL statement; only the reversal candidate is ever auto-rejected."""
    (tmp_path / "observations.jsonl").write_text(
        json.dumps(_reversal_observation(taxonomy_id="renewal_notice")) + "\n",
        encoding="utf-8",
    )
    doc = {
        "posture": {
            "generation": {
                "interview": [
                    {"q": "sacred_clauses", "answer": "renewal notice"},
                    {"q": "flexible_clauses", "answer": "renewal notice"},
                ]
            }
        }
    }
    opf_path = tmp_path / "playbook.opf.json"
    opf_path.write_text(json.dumps(doc), encoding="utf-8")
    candidates_path = tmp_path / "floor.candidates.json"

    for _ in range(2):
        write_floor_candidates(tmp_path)
        by_source = {
            c["source"]: c
            for c in json.loads(candidates_path.read_text(encoding="utf-8"))["candidates"]
        }
        assert by_source["reversal"]["statement"] == by_source["interview_q4"]["statement"]
        assert by_source["reversal"]["decision"] == "rejected"
        assert by_source["reversal"]["comment"] == _Q5_REJECTION_COMMENT
        assert "decision" not in by_source["interview_q4"]
        assert "comment" not in by_source["interview_q4"]

    # Change the Q5 answer: the rejection is gone with it — it was never
    # stored anywhere but derived.
    doc["posture"]["generation"]["interview"][1]["answer"] = "IP assignment"
    opf_path.write_text(json.dumps(doc), encoding="utf-8")
    write_floor_candidates(tmp_path)
    after = json.loads(candidates_path.read_text(encoding="utf-8"))["candidates"]
    reversal = next(c for c in after if c["source"] == "reversal")
    assert "decision" not in reversal and "comment" not in reversal


# ---------------------------------------------------------------------------
# Q4/Q5 contradictory-interview warning (issue #105 reviewer gate)
# ---------------------------------------------------------------------------


def test_q4_q5_contradictions_empty_when_no_overlap() -> None:
    answers = {"sacred_clauses": "liability cap", "flexible_clauses": "renewal notice"}
    assert q4_q5_contradictions(answers) == []


def test_q4_q5_contradictions_empty_when_either_unanswered() -> None:
    assert q4_q5_contradictions(None) == []
    assert q4_q5_contradictions({"sacred_clauses": "liability cap"}) == []
    assert q4_q5_contradictions({"flexible_clauses": "renewal notice"}) == []


def test_q4_q5_contradictions_fires_on_exact_normalized_overlap() -> None:
    answers = {"sacred_clauses": "renewal notice", "flexible_clauses": "renewal notice"}
    warnings = q4_q5_contradictions(answers)
    assert len(warnings) == 1
    assert "renewal notice" in warnings[0]
    assert "sacred_clauses" in warnings[0]
    assert "flexible_clauses" in warnings[0]


def test_q4_q5_contradictions_does_not_fire_on_substring_overlap() -> None:
    """Mutation-guard: same discipline as the candidate-rejection match --
    exact normalized equality, never substring."""
    answers = {"sacred_clauses": "renewal", "flexible_clauses": "auto renewal option"}
    assert q4_q5_contradictions(answers) == []


def test_propose_floor_candidates_surfaces_q4_q5_contradiction_warning() -> None:
    answers = {"sacred_clauses": "renewal notice", "flexible_clauses": "renewal notice"}
    result = propose_floor_candidates([], answers)
    assert result["warnings"]
    assert "renewal notice" in result["warnings"][0]


def test_propose_floor_candidates_omits_warnings_key_when_none() -> None:
    """Additive key, omitted (never an empty list) when there's nothing to
    warn about -- the locked empty-input shape stays exact."""
    result = propose_floor_candidates([], None)
    assert result == {"candidates": []}
    assert "warnings" not in result


# ---------------------------------------------------------------------------
# write_floor_candidates — I/O
# ---------------------------------------------------------------------------


def test_write_floor_candidates_reads_observations_and_posture(tmp_path: Path) -> None:
    obs_path = tmp_path / "observations.jsonl"
    obs_path.write_text(
        json.dumps(_reversal_observation()) + "\n",
        encoding="utf-8",
    )
    doc = _minimal_doc(
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


def test_write_floor_candidates_reports_unclassified_omitted_count(tmp_path: Path) -> None:
    """Excluded is not the same as hidden.

    `derive_reversal_candidates` drops unclassified reversals so the review
    checklist is not three-quarters segmentation debris — but a reviewer
    reading a short checklist must be able to tell that reversals were set
    aside, and how many, rather than reading it as "this is everything the
    corpus proposed".
    """
    out = tmp_path / "out"
    out.mkdir()
    (out / "observations.jsonl").write_text(
        "\n".join(
            json.dumps(o)
            for o in (
                _reversal_observation(observation_id="doc-a/2/8.1"),
                _reversal_observation(
                    observation_id="doc-a/2/9.1",
                    taxonomy_id=None,
                    clause_path="9.1",
                    full_text="3 3",
                ),
                _reversal_observation(
                    observation_id="doc-a/2/9.2",
                    taxonomy_id=None,
                    clause_path="9.2",
                    full_text="shall",
                ),
            )
        )
        + "\n",
        encoding="utf-8",
    )

    written = json.loads(write_floor_candidates(out).read_text(encoding="utf-8"))

    assert len(written["candidates"]) == 1  # only the classified one
    assert written["unclassified_reversals_omitted"] == 2


def test_write_floor_candidates_no_playbook_no_observations(tmp_path: Path) -> None:
    out_path = write_floor_candidates(tmp_path)
    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert written == {
        "candidates": [],
        "unclassified_reversals_omitted": 0,
        "structural_reversals_omitted": 0,
        "below_min_deals_omitted": 0,
    }


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
    doc = _minimal_doc(floor={"invariants": []})
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
    assert written == {
        "candidates": [],
        "unclassified_reversals_omitted": 0,
        "structural_reversals_omitted": 0,
        "below_min_deals_omitted": 0,
    }


def test_cli_floor_propose_missing_out_dir_fails() -> None:
    exit_code, output = _invoke("floor", "propose", "/nonexistent/out/dir")
    assert exit_code != 0


def test_cli_floor_propose_prints_q4_q5_contradiction_warning(tmp_path: Path) -> None:
    """issue #105 review round 2 finding 4: the Reviewer gate's chosen
    behavior for a contradictory interview (same clause type named in both
    Q4 "sacred_clauses" and Q5 "flexible_clauses") is a CLI WARN line --
    this pins that the CLI actually prints it, and that the seam it depends
    on (write_floor_candidates persisting the additive "warnings" key into
    floor.candidates.json, for propose_floor_candidates to have produced it
    from) round-trips through the real file, not just the pure function."""
    doc = _minimal_doc(
        posture={
            "generation": {
                "interview": [
                    {"q": "sacred_clauses", "question": "...", "answer": "renewal notice"},
                    {"q": "flexible_clauses", "question": "...", "answer": "renewal notice"},
                ]
            }
        }
    )
    (tmp_path / "playbook.opf.json").write_text(json.dumps(doc), encoding="utf-8")

    exit_code, output = _invoke("floor", "propose", str(tmp_path))

    assert exit_code == 0, output
    assert "WARN" in output
    assert "renewal notice" in output
    assert "sacred_clauses" in output
    assert "flexible_clauses" in output

    written = json.loads((tmp_path / "floor.candidates.json").read_text(encoding="utf-8"))
    assert written["warnings"]
    assert "renewal notice" in written["warnings"][0]


# ---------------------------------------------------------------------------
# sign_floor_invariant / sign_invariant_id — issue #103
# ---------------------------------------------------------------------------

_CONDITIONAL_STATEMENT = (
    "Limitation of liability, if present, must not be unilateral in the counterparty's favor."
)

# issue #127: `sign_floor_invariant`/`playbook floor sign` now REQUIRE a human
# attribution — a synthetic, obviously-fictional name (never a real person),
# reused across every test below that isn't itself testing the requirement.
_TEST_SIGNED_BY = "Test Legal Owner"


def test_sign_floor_invariant_records_statement_verbatim() -> None:
    """A conditional statement with commas — exactly the shape Q4's
    semicolon-split templating would garble — must survive byte-for-byte."""
    result = sign_floor_invariant(
        _CONDITIONAL_STATEMENT, signed_by=_TEST_SIGNED_BY, existing_invariants=[]
    )

    assert len(result) == 1
    assert result[0]["statement"] == _CONDITIONAL_STATEMENT


def test_sign_floor_invariant_default_id_is_slug_of_statement() -> None:
    result = sign_floor_invariant(
        "Never accept uncapped liability.", signed_by=_TEST_SIGNED_BY, existing_invariants=[]
    )
    assert result[0]["id"] == sign_invariant_id("Never accept uncapped liability.")
    assert result[0]["id"] == "never-accept-uncapped-liability"


def test_sign_floor_invariant_id_override() -> None:
    result = sign_floor_invariant(
        _CONDITIONAL_STATEMENT,
        invariant_id="liability-not-unilateral",
        signed_by=_TEST_SIGNED_BY,
        existing_invariants=[],
    )
    assert result[0]["id"] == "liability-not-unilateral"


def test_sign_floor_invariant_default_rationale() -> None:
    result = sign_floor_invariant(
        "Never accept uncapped liability.", signed_by=_TEST_SIGNED_BY, existing_invariants=[]
    )
    assert result[0]["rationale"] == "Hand-authored via `playbook floor sign`."


def test_sign_floor_invariant_custom_rationale() -> None:
    result = sign_floor_invariant(
        "Never accept uncapped liability.",
        rationale="Signed off by the GC 2026-08-20.",
        signed_by=_TEST_SIGNED_BY,
        existing_invariants=[],
    )
    assert result[0]["rationale"] == "Signed off by the GC 2026-08-20."


def test_sign_floor_invariant_taxonomy_id_stored_as_x_prefixed() -> None:
    """Not a bare `taxonomy_id` — spec/playbook.schema-0.5.json's frozen
    floor.invariants[] item is `additionalProperties: false`; only the
    `^x_` escape hatch is schema-safe without a spec version bump."""
    result = sign_floor_invariant(
        "Never accept uncapped liability.",
        taxonomy_id="uncapped_liability",
        signed_by=_TEST_SIGNED_BY,
        existing_invariants=[],
    )
    assert result[0]["x_taxonomy_id"] == "uncapped_liability"
    assert "taxonomy_id" not in result[0]


def test_sign_floor_invariant_no_taxonomy_id_key_when_clause_omitted() -> None:
    result = sign_floor_invariant(
        "Never accept uncapped liability.", signed_by=_TEST_SIGNED_BY, existing_invariants=[]
    )
    assert "x_taxonomy_id" not in result[0]


def test_sign_floor_invariant_stamps_signed_by_and_signed_at() -> None:
    """The structural attribution fields issue #127 adds: `x_signed_by` is
    recorded verbatim, `x_signed_at` defaults to a non-blank timestamp when
    the caller doesn't supply one."""
    result = sign_floor_invariant(
        "Never accept uncapped liability.", signed_by=_TEST_SIGNED_BY, existing_invariants=[]
    )
    assert result[0]["x_signed_by"] == _TEST_SIGNED_BY
    assert isinstance(result[0]["x_signed_at"], str) and result[0]["x_signed_at"]


def test_sign_floor_invariant_signed_at_override_recorded_verbatim() -> None:
    result = sign_floor_invariant(
        "Never accept uncapped liability.",
        signed_by=_TEST_SIGNED_BY,
        signed_at="2026-01-01T00:00:00+00:00",
        existing_invariants=[],
    )
    assert result[0]["x_signed_at"] == "2026-01-01T00:00:00+00:00"


def test_sign_floor_invariant_rejects_blank_signed_by() -> None:
    """issue #127: --signed-by is not optional — this is the structural
    guarantee the ticket exists to add. Blank/whitespace-only and omitted
    (default None) must both be refused."""
    with pytest.raises(FloorCandidateError):
        sign_floor_invariant("Never accept uncapped liability.", existing_invariants=[])
    with pytest.raises(FloorCandidateError):
        sign_floor_invariant(
            "Never accept uncapped liability.", signed_by="   ", existing_invariants=[]
        )


def test_sign_floor_invariant_rejects_rationale_that_names_the_signer() -> None:
    """issue #209 (2026-08-24 skill QA audit, finding #92): rationale ships
    verbatim into every consumer's model-facing review prompt while
    x_signed_by/x_signed_at never do — a rationale that repeats the signer's
    name duplicates structural attribution into the one field this engine
    cannot keep confidential. Reproduces the real-world failure mode: a
    hand-authored rationale reading 'Hand-authored and signed by the legal
    owner (<name>), <date>.' instead of a legal-justification-only sentence."""
    with pytest.raises(FloorCandidateError, match=_TEST_SIGNED_BY):
        sign_floor_invariant(
            "Never accept uncapped liability.",
            signed_by=_TEST_SIGNED_BY,
            rationale=f"Hand-authored and signed by the legal owner ({_TEST_SIGNED_BY}), "
            "2026-08-21.",
            existing_invariants=[],
        )


def test_sign_floor_invariant_rejects_rationale_naming_signer_case_insensitively() -> None:
    with pytest.raises(FloorCandidateError):
        sign_floor_invariant(
            "Never accept uncapped liability.",
            signed_by=_TEST_SIGNED_BY,
            rationale=f"Signed by {_TEST_SIGNED_BY.upper()} on 2026-08-21.",
            existing_invariants=[],
        )


def test_sign_floor_invariant_rerun_with_bad_rationale_stays_idempotent_noop() -> None:
    """The name-in-rationale guard must never break the documented
    idempotent-no-op contract: an EXACT rerun (same id, same statement) of a
    call that would now be rejected if it were new must still short-circuit
    to the existing entry, never re-validate and raise."""
    already_bad = [
        {
            "id": "never-accept-uncapped-liability",
            "statement": "Never accept uncapped liability.",
            "rationale": f"Hand-authored and signed by the legal owner ({_TEST_SIGNED_BY}), "
            "2026-08-21.",
            "x_signed_by": _TEST_SIGNED_BY,
            "x_signed_at": "2026-08-21T00:00:00+00:00",
        }
    ]
    result = sign_floor_invariant(
        "Never accept uncapped liability.",
        signed_by=_TEST_SIGNED_BY,
        rationale=f"Hand-authored and signed by the legal owner ({_TEST_SIGNED_BY}), 2026-08-21.",
        existing_invariants=already_bad,
    )
    assert result == already_bad
    assert result[0] is already_bad[0]


def test_sign_floor_invariant_appends_after_existing_entries() -> None:
    existing = [{"id": "existing-one", "statement": "Existing.", "rationale": "r"}]
    result = sign_floor_invariant(
        "Never accept uncapped liability.",
        signed_by=_TEST_SIGNED_BY,
        existing_invariants=existing,
    )
    assert result[0] == existing[0]
    assert len(result) == 2


def test_sign_floor_invariant_rejects_blank_statement() -> None:
    with pytest.raises(FloorCandidateError):
        sign_floor_invariant("   ", signed_by=_TEST_SIGNED_BY, existing_invariants=[])


def test_sign_floor_invariant_same_id_same_statement_is_idempotent_noop() -> None:
    first = sign_floor_invariant(
        _CONDITIONAL_STATEMENT, signed_by=_TEST_SIGNED_BY, existing_invariants=[]
    )
    second = sign_floor_invariant(
        _CONDITIONAL_STATEMENT, signed_by=_TEST_SIGNED_BY, existing_invariants=first
    )
    assert second == first
    assert second[0] is first[0]  # not even a fresh dict — true no-op


def test_sign_floor_invariant_noop_ignores_a_new_rationale_or_clause() -> None:
    """Literal reading of the ticket's collision rule: 'same id + same
    statement = idempotent no-op', full stop — not 'no-op unless the
    caller also asked to change something else'. There is no in-place
    update path here (see the function's docstring); a caller wanting to
    change rationale/taxonomy_id on an already-signed statement removes or
    edits the invariant directly."""
    first = sign_floor_invariant(
        _CONDITIONAL_STATEMENT,
        rationale="Original rationale.",
        signed_by=_TEST_SIGNED_BY,
        existing_invariants=[],
    )
    second = sign_floor_invariant(
        _CONDITIONAL_STATEMENT,
        rationale="A different rationale.",
        taxonomy_id="limitation_of_liability",
        signed_by="A Different Signer",
        existing_invariants=first,
    )
    assert second == first
    assert second[0]["rationale"] == "Original rationale."
    assert second[0]["x_signed_by"] == _TEST_SIGNED_BY
    assert "x_taxonomy_id" not in second[0]


def test_sign_floor_invariant_same_id_different_statement_refuses() -> None:
    first = sign_floor_invariant(
        _CONDITIONAL_STATEMENT, signed_by=_TEST_SIGNED_BY, existing_invariants=[]
    )
    inv_id = first[0]["id"]

    with pytest.raises(FloorCandidateError, match=inv_id):
        sign_floor_invariant(
            "A completely different statement.",
            invariant_id=inv_id,
            signed_by=_TEST_SIGNED_BY,
            existing_invariants=first,
        )

    # Never mutated in place.
    assert first[0]["statement"] == _CONDITIONAL_STATEMENT


def test_sign_floor_invariant_never_overwrites_a_foreign_entry() -> None:
    """Colliding with a hand-authored entry from an entirely different
    producer (e.g. Q4 promotion) must refuse exactly the same way — this
    function draws no distinction between 'foreign' and 'self-authored'
    collisions (unlike promote_interview_q4_invariants'
    attribution-marker guard): ANY id collision
    with a different statement is refused."""
    hand_authored = {
        "id": "no-uncapped-liability",
        "statement": "Never accept uncapped liability under any circumstances.",
        "rationale": "Board-approved 2026-01-01.",
    }
    with pytest.raises(FloorCandidateError):
        sign_floor_invariant(
            "Never accept uncapped liability.",
            invariant_id="no-uncapped-liability",
            signed_by=_TEST_SIGNED_BY,
            existing_invariants=[hand_authored],
        )
    assert hand_authored == {
        "id": "no-uncapped-liability",
        "statement": "Never accept uncapped liability under any circumstances.",
        "rationale": "Board-approved 2026-01-01.",
    }


def test_sign_invariant_id_blank_override_falls_back_to_slug() -> None:
    assert sign_invariant_id("Never accept uncapped liability.", "   ") == (
        "never-accept-uncapped-liability"
    )


def test_sign_invariant_id_all_punctuation_statement_falls_back() -> None:
    assert sign_invariant_id("...", None) == "floor-invariant"


# ---------------------------------------------------------------------------
# CLI — playbook floor sign — issue #103
# ---------------------------------------------------------------------------


def _write_signable_doc(tmp_path: Path, **doc_overrides: Any) -> Path:
    doc = _minimal_doc(floor={"invariants": []}, **doc_overrides)
    opf_path = tmp_path / "playbook.opf.json"
    opf_path.write_text(json.dumps(doc), encoding="utf-8")
    return opf_path


def _write_taxonomy_config(tmp_path: Path, entry_ids: list[str]) -> Path:
    """Minimal engine config + taxonomy YAML for --clause validation tests."""
    taxonomy_path = tmp_path / "taxonomy.yaml"
    taxonomy_path.write_text(
        yaml.dump(
            {
                "source": "custom",
                "entries": [
                    {
                        "id": entry_id,
                        "label": entry_id.replace("_", " ").title(),
                        "status": "active",
                        "cuad_origin": None,
                    }
                    for entry_id in entry_ids
                ],
            }
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "playbook.config.yaml"
    config_path.write_text(
        yaml.dump(
            {
                "agreement_type": {"id": "test-agreement", "name": "Test Agreement"},
                "baseline": {"template": None},
                "taxonomy": str(taxonomy_path),
            }
        ),
        encoding="utf-8",
    )
    return config_path


def test_cli_floor_sign_writes_verbatim_statement(tmp_path: Path) -> None:
    opf_path = _write_signable_doc(tmp_path)

    exit_code, output = _invoke(
        "floor",
        "sign",
        str(tmp_path),
        "--statement",
        _CONDITIONAL_STATEMENT,
        "--signed-by",
        _TEST_SIGNED_BY,
    )

    assert exit_code == 0, output
    written = json.loads(opf_path.read_text(encoding="utf-8"))
    assert len(written["floor"]["invariants"]) == 1
    invariant = written["floor"]["invariants"][0]
    signed_at = invariant.pop("x_signed_at")
    assert isinstance(signed_at, str) and signed_at
    assert invariant == {
        "id": sign_invariant_id(_CONDITIONAL_STATEMENT),
        "statement": _CONDITIONAL_STATEMENT,
        "rationale": "Hand-authored via `playbook floor sign`.",
        "x_signed_by": _TEST_SIGNED_BY,
    }


def test_cli_floor_sign_requires_signed_by(tmp_path: Path) -> None:
    """issue #127: the CLI, not just the underlying function, refuses to
    sign without a human attribution — this is the primary gap the ticket
    closes (an agent could previously run this command with no identifying
    field at all)."""
    _write_signable_doc(tmp_path)

    exit_code, output = _invoke(
        "floor", "sign", str(tmp_path), "--statement", _CONDITIONAL_STATEMENT
    )

    assert exit_code != 0
    assert "signed-by" in output.lower()


def test_cli_floor_sign_refreshes_content_hash(tmp_path: Path) -> None:
    """Issue #103's Reviewer gate: 'a forgotten hash recompute must be
    caught by a test, not luck.' Direct equality against a freshly computed
    content_hash of the expected final document — the same proof pattern
    test_posture.py uses for apply_posture_interview."""
    doc_overrides: dict[str, Any] = {"floor": {"invariants": []}}
    doc = _minimal_doc(**doc_overrides)
    opf_path = tmp_path / "playbook.opf.json"
    opf_path.write_text(json.dumps(doc), encoding="utf-8")

    exit_code, output = _invoke(
        "floor",
        "sign",
        str(tmp_path),
        "--statement",
        _CONDITIONAL_STATEMENT,
        "--signed-by",
        _TEST_SIGNED_BY,
    )
    assert exit_code == 0, output

    written = json.loads(opf_path.read_text(encoding="utf-8"))
    expected_doc = dict(written)
    assert written["identity"]["content_hash"] == content_hash(expected_doc)
    assert written["identity"]["content_hash"] != doc["identity"]["content_hash"]
    assert written["identity"]["section_digests"] == compute_section_digests(written)
    assert (
        written["identity"]["section_digests"]["floor"]
        != doc["identity"]["section_digests"]["floor"]
    )


def test_cli_floor_sign_warns_on_posture_floor_conflict(tmp_path: Path) -> None:
    """Issue #103's Reviewer gate: 'floor sign' runs check_posture_floor_conflict
    (see the rationale comment at cli.py) and its SHOULD-warn output must
    actually be exercised by a test, not left dead code behind an always-empty
    fixture posture."""
    _write_signable_doc(
        tmp_path,
        posture={"system_prompt": "The liability cap is flexible to close a deal."},
    )

    exit_code, output = _invoke(
        "floor",
        "sign",
        str(tmp_path),
        "--statement",
        _CONDITIONAL_STATEMENT,
        "--signed-by",
        _TEST_SIGNED_BY,
    )

    assert exit_code == 0, output
    assert "WARN" in output, output


def test_cli_floor_sign_output_passes_playbook_validate(tmp_path: Path) -> None:
    opf_path = _write_signable_doc(tmp_path)
    config_path = _write_taxonomy_config(tmp_path, ["limitation_of_liability"])

    exit_code, output = _invoke(
        "floor",
        "sign",
        str(tmp_path),
        "--statement",
        _CONDITIONAL_STATEMENT,
        "--signed-by",
        _TEST_SIGNED_BY,
        "--clause",
        "limitation_of_liability",
        "--config",
        str(config_path),
    )
    assert exit_code == 0, output

    written = json.loads(opf_path.read_text(encoding="utf-8"))
    result = validate_document(written)
    blocking = [str(e) for e in result.errors if e.blocking]
    assert result.ok, blocking
    # A signed invariant carries structural attribution — no SHOULD-warn
    # about it either (issue #127).
    assert not any("structural attribution" in str(e) for e in result.errors)


def test_cli_floor_sign_idempotent_rerun_does_not_rewrite_the_file(tmp_path: Path) -> None:
    opf_path = _write_signable_doc(tmp_path)

    exit_code, _ = _invoke(
        "floor",
        "sign",
        str(tmp_path),
        "--statement",
        _CONDITIONAL_STATEMENT,
        "--signed-by",
        _TEST_SIGNED_BY,
    )
    assert exit_code == 0
    first_bytes = opf_path.read_bytes()

    exit_code2, output2 = _invoke(
        "floor",
        "sign",
        str(tmp_path),
        "--statement",
        _CONDITIONAL_STATEMENT,
        "--signed-by",
        _TEST_SIGNED_BY,
    )
    assert exit_code2 == 0, output2
    assert "no-op" in output2
    assert opf_path.read_bytes() == first_bytes


def test_cli_floor_sign_collision_refuses_and_exits_nonzero(tmp_path: Path) -> None:
    opf_path = _write_signable_doc(tmp_path)
    exit_code, _ = _invoke(
        "floor",
        "sign",
        str(tmp_path),
        "--statement",
        _CONDITIONAL_STATEMENT,
        "--signed-by",
        _TEST_SIGNED_BY,
    )
    assert exit_code == 0
    inv_id = json.loads(opf_path.read_text(encoding="utf-8"))["floor"]["invariants"][0]["id"]
    original_bytes = opf_path.read_bytes()

    exit_code2, output2 = _invoke(
        "floor",
        "sign",
        str(tmp_path),
        "--statement",
        "A totally different statement.",
        "--id",
        inv_id,
        "--signed-by",
        _TEST_SIGNED_BY,
    )

    assert exit_code2 != 0
    assert "ERROR" in output2
    # Never overwritten.
    assert opf_path.read_bytes() == original_bytes


def test_cli_floor_sign_unknown_clause_lists_valid_ids_and_exits_nonzero(tmp_path: Path) -> None:
    _write_signable_doc(tmp_path)
    config_path = _write_taxonomy_config(tmp_path, ["limitation_of_liability", "indemnification"])

    exit_code, output = _invoke(
        "floor",
        "sign",
        str(tmp_path),
        "--statement",
        _CONDITIONAL_STATEMENT,
        "--signed-by",
        _TEST_SIGNED_BY,
        "--clause",
        "not_a_real_clause",
        "--config",
        str(config_path),
    )

    assert exit_code != 0
    assert "not_a_real_clause" in output
    assert "limitation_of_liability" in output
    assert "indemnification" in output


def test_cli_floor_sign_clause_without_config_exits_nonzero(tmp_path: Path) -> None:
    _write_signable_doc(tmp_path)

    exit_code, output = _invoke(
        "floor",
        "sign",
        str(tmp_path),
        "--statement",
        _CONDITIONAL_STATEMENT,
        "--signed-by",
        _TEST_SIGNED_BY,
        "--clause",
        "limitation_of_liability",
    )

    assert exit_code != 0
    assert "--config" in output


def test_cli_floor_sign_missing_out_dir_fails() -> None:
    exit_code, output = _invoke(
        "floor",
        "sign",
        "/nonexistent/out/dir",
        "--statement",
        _CONDITIONAL_STATEMENT,
        "--signed-by",
        _TEST_SIGNED_BY,
    )
    assert exit_code != 0


def test_cli_floor_sign_missing_playbook_fails(tmp_path: Path) -> None:
    exit_code, output = _invoke(
        "floor",
        "sign",
        str(tmp_path),
        "--statement",
        _CONDITIONAL_STATEMENT,
        "--signed-by",
        _TEST_SIGNED_BY,
    )
    assert exit_code != 0
    assert "playbook.opf.json" in output


def test_cli_floor_propose_help_points_at_floor_sign_not_a_retired_surface() -> None:
    """Issue #103/#239: the docstring must name ``floor sign`` as the way to
    accept a candidate — not the (never real) curation CLI, nor the retired
    review page and ``view apply``."""
    exit_code, output = _invoke("floor", "propose", "--help")
    assert exit_code == 0
    assert "playbook floor sign" in " ".join(output.split())
    for retired in ("curation CLI", "review.html", "view apply", "export feedback"):
        assert retired not in output
