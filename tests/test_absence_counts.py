"""Verified absence counts (issue #251, OPF 0.6): "signed N deals without clause X".

Every document under test is built by the production path: real ``Observation``
rows through ``compile_clause_positions`` and ``assemble_playbook`` (so
``evidence.absence`` is written by ``absence.build_absence`` and the digest by
``build_digest_v5``), then validated with the production validator. Nothing is a
hand-written ``absence`` record except where a test deliberately corrupts one to
see the validator refuse it.
"""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Any

from playbook_engine import absence
from playbook_engine.agent_judge import (
    PendingQueue,
    StoreBackedAbsenceCheckJudge,
    VerdictStore,
    infer_verdict_kind,
    validate_verdict,
)
from playbook_engine.clause_classifier import _content_tokens
from playbook_engine.clause_position_compiler import compile_clause_positions
from playbook_engine.clause_tree import ClauseNode
from playbook_engine.digest import build_digest_uncapped, build_digest_v5, digest_token_estimate
from playbook_engine.observation_builder import Observation, ObservationCitation
from playbook_engine.playbook_assembler import assemble_playbook
from playbook_engine.precedent import build_precedent_evidence
from playbook_engine.resegmentation import non_clause_reason
from playbook_engine.rubric import JUDGE_KINDS, RubricStamp, rubric_version
from playbook_engine.scorecard import build_scorecard
from playbook_engine.validator import validate_document

# ---------------------------------------------------------------------------
# Fixtures: texts whose scores against the standard were measured, not guessed
# ---------------------------------------------------------------------------

ASSIGN = "assignment"
TERM = "term"

ASSIGN_STD = (
    "Neither party may assign this Agreement or any rights or obligations under it "
    "without the prior written consent of the other party, and any attempted assignment "
    "without such consent is void."
)
TERM_STD = (
    "This Agreement begins on the effective date and continues for two years unless "
    "terminated earlier by either party on thirty days written notice."
)
#: Reworded assignment: scores 0.222 against ASSIGN_STD, above the guard (0.10)
#: and below the content-similarity assign threshold (0.25).
ASSIGN_NEAR = (
    "Neither side may assign or transfer its rights under this contract unless the "
    "other party agrees in writing."
)
#: Scores 0.105 against ASSIGN_STD: just above the guard.
ASSIGN_EDGE = (
    "Transfer of this contract, or of any right under it, by either side requires the "
    "written approval of the other party beforehand."
)
#: A cut-off phrase the re-segmentation pass marks as no clause ("fragment");
#: it scores 0.333 against ASSIGN_STD, well above the guard.
ASSIGN_FRAGMENT = "without the prior written consent of the other party,"
UNRELATED = (
    "The Institution shall maintain commercial general liability insurance in the "
    "amount of two million dollars per occurrence."
)
SIGNATURE = "Signed: Name: Title: Date: Senior Vice President"

_TAXONOMY = {
    "source": "custom",
    "entries": [
        {
            "id": tid,
            "label": label,
            "status": "custom",
            "cuad_origin": None,
            "description": label,
        }
        for tid, label in ((ASSIGN, "Assignment"), (TERM, "Term"))
    ],
}


def _obs(
    doc: str,
    tid: str | None,
    text: str,
    *,
    path: str,
    outcome: str = "signed",
    non_clause: str | None = None,
) -> Observation:
    return Observation(
        observation_id=f"{doc}/3/{path}",
        taxonomy_id=tid,
        text_summary=text[:100],
        full_text=text,
        citation=ObservationCitation(
            document_id=doc, version=3, clause_path=path, char_span=(0, len(text))
        ),
        deviation="none",
        risk_delta={"direction": "neutral", "magnitude": "none"},
        provenance="our_paper",
        outcome=outcome,
        non_clause=non_clause,
    )


def _non_clause(text: str) -> str | None:
    """The ``x_non_clause`` label mining gives an unclassified node of *text*:
    taken from the producer, never typed in, so a fixture cannot drift from it."""
    return non_clause_reason(
        ClauseNode(clause_path="0", heading=None, text=text, char_span=(0, len(text)))
    )


def _template() -> list[Observation]:
    return [
        _obs("template", ASSIGN, ASSIGN_STD, path="9"),
        _obs("template", TERM, TERM_STD, path="2"),
    ]


def _doc(doc_id: str, *, signed: bool = True) -> dict[str, Any]:
    return {
        "document_id": doc_id,
        "provenance": "our_paper",
        "in_scope": True,
        "versions": 3,
        "signed_version": 3 if signed else None,
        "signed_on": None,
        "signed_on_basis": None,
        "version_order_basis": "edit_distance_chain",
    }


#: The deals of the main fixture and what each is for.
def _observations() -> list[Observation]:
    big = " ".join([UNRELATED, UNRELATED.replace("two million", "five million"), ASSIGN_NEAR])
    return [
        # deal-a signed both clauses: counted by neither absence count.
        _obs("deal-a", ASSIGN, ASSIGN_STD, path="9"),
        _obs("deal-a", TERM, TERM_STD, path="2"),
        # deal-b: only a term; its one unclassified passage is unrelated.
        _obs("deal-b", TERM, TERM_STD, path="2"),
        _obs("deal-b", None, UNRELATED, path="5"),
        # deal-c: a missed classification -- an unclassified reworded assignment.
        _obs("deal-c", TERM, TERM_STD, path="2"),
        _obs("deal-c", None, ASSIGN_NEAR, path="7"),
        # deal-d: unsigned; never counted.
        _obs("deal-d", TERM, TERM_STD, path="2", outcome="unsigned"),
        # deal-e: signed, but its executed copy produced no rows at all.
        # deal-f: an assignment phrase sits in a node that is no clause.
        _obs("deal-f", TERM, TERM_STD, path="2"),
        _obs("deal-f", None, ASSIGN_FRAGMENT, path="8", non_clause=_non_clause(ASSIGN_FRAGMENT)),
        # deal-g: a node the classifier filed under assignment, too short to be precedent.
        _obs("deal-g", TERM, TERM_STD, path="2"),
        _obs("deal-g", ASSIGN, "Assignment.", path="4"),
        # deal-h: the reworded assignment is one sentence of a big mixed node.
        _obs("deal-h", TERM, TERM_STD, path="2"),
        _obs("deal-h", None, big.replace(". ", ".\n"), path="6"),
    ]


_DOCS = [_doc(f"deal-{c}") for c in "abcefgh"] + [_doc("deal-d", signed=False)]


def _evidence(observations: list[Observation] | None = None) -> dict[str, Any]:
    obs = _observations() if observations is None else observations
    positions, _, _ = compile_clause_positions(obs, _template())
    return build_precedent_evidence(
        agreement_type_id="fixture",
        clause_positions=positions,
        observations=obs,
        corpus_documents=_DOCS,
        party=None,
    )


def _records(evidence: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(r["document_id"], r["taxonomy_id"]): r for r in evidence["absence"]["records"]}


def _playbook(observations: list[Observation] | None = None) -> dict[str, Any]:
    obs = _observations() if observations is None else observations
    positions, _, _ = compile_clause_positions(obs, _template())
    return assemble_playbook(
        agreement_type={"id": "fixture", "name": "Fixture Agreement"},
        baseline={"has_canonical_template": True},
        taxonomy=_TAXONOMY,
        clause_positions=positions,
        corpus_documents=copy.deepcopy(_DOCS),
        generated_at="2026-01-01T00:00:00Z",
        observations=obs,
    )


# ---------------------------------------------------------------------------
# The three statuses
# ---------------------------------------------------------------------------


def test_a_verified_absence_has_no_signed_text_near_the_clause() -> None:
    record = _records(_evidence())[("deal-b", ASSIGN)]
    assert record["status"] == absence.STATUS_VERIFIED
    assert record["basis"] == absence.BASIS_SCORED
    assert record["n_checked"] == 2  # its term (filed elsewhere) and one unrelated passage
    assert record["candidates"] == []
    assert record["id"] == absence.absence_id("fixture", "deal-b", ASSIGN)


def test_a_near_miss_becomes_possibly_present_with_its_candidate() -> None:
    record = _records(_evidence())[("deal-c", ASSIGN)]
    assert record["status"] == absence.STATUS_POSSIBLE
    assert record["basis"] == absence.BASIS_SCORED
    (candidate,) = record["candidates"]
    assert candidate["ref"]["document_id"] == "deal-c"
    assert candidate["ref"]["clause_path"] == "7"
    # Above the guard, but too weak for the classifier to assign it (0.25).
    assert absence.ABSENCE_GUARD <= candidate["score"] < 0.25


def test_the_guard_is_well_below_the_content_similarity_assign_threshold() -> None:
    from playbook_engine.clause_classifier import CONTENT_ASSIGN_THRESHOLD

    assert 0 < absence.ABSENCE_GUARD <= CONTENT_ASSIGN_THRESHOLD / 2


def test_the_score_that_decides_is_the_score_the_record_shows() -> None:
    """A text at 0.105 is a candidate; one at 0.0 is not. The record carries the
    rounded number the guard was compared with."""
    obs = [o for o in _observations() if o.citation.document_id != "deal-c"] + [
        _obs("deal-c", TERM, TERM_STD, path="2"),
        _obs("deal-c", None, ASSIGN_EDGE, path="7"),
    ]
    (candidate,) = _records(_evidence(obs))[("deal-c", ASSIGN)]["candidates"]
    assert candidate["score"] == 0.105
    assert candidate["score"] >= absence.ABSENCE_GUARD


def test_a_passage_that_is_no_clause_does_not_hold_a_deal_back() -> None:
    # The fixture is a state mining writes: the producer marks the phrase no
    # clause, and as a clause it would score above the guard.
    assert _non_clause(ASSIGN_FRAGMENT) is not None
    reference = _content_tokens(ASSIGN_STD)
    score = max(absence._jaccard(w, reference) for w in absence._windows(ASSIGN_FRAGMENT))
    assert score >= absence.ABSENCE_GUARD
    record = _records(_evidence())[("deal-f", ASSIGN)]
    assert record["status"] == absence.STATUS_VERIFIED
    assert record["n_checked"] == 1  # only its term: the fragment is not examined


def test_a_clause_stated_inside_a_block_filed_under_another_type_is_found() -> None:
    """A missed classification is not only an unclassified node: the assignment
    sentence sits inside a block the classifier filed under the term clause."""
    mixed = TERM_STD + "\n" + ASSIGN_NEAR
    obs = [o for o in _observations() if o.citation.document_id != "deal-b"] + [
        _obs("deal-b", TERM, mixed, path="2"),
    ]
    record = _records(_evidence(obs))[("deal-b", ASSIGN)]
    assert record["status"] == absence.STATUS_POSSIBLE
    (candidate,) = record["candidates"]
    assert candidate["filed_under"] == TERM
    assert candidate["ref"]["clause_path"] == "2"
    assert candidate["score"] >= 0.2
    # and a node that is merely filed elsewhere, about something else, holds nothing back
    other = _records(_evidence())[("deal-b", ASSIGN)]
    assert other["status"] == absence.STATUS_VERIFIED


def test_a_candidate_says_what_the_classifier_filed_it_under() -> None:
    records = _records(_evidence())
    assert records[("deal-c", ASSIGN)]["candidates"][0]["filed_under"] is None
    assert records[("deal-g", ASSIGN)]["candidates"][0]["filed_under"] == ASSIGN


def test_a_sentence_of_a_big_mixed_node_is_found() -> None:
    """The clause is one sentence of a node three times its size: the node as a
    whole scores under the guard, its sentence does not."""
    record = _records(_evidence())[("deal-h", ASSIGN)]
    assert record["status"] == absence.STATUS_POSSIBLE
    assert record["candidates"][0]["score"] >= 0.2


def test_a_node_filed_under_the_clause_but_too_short_for_precedent_is_possibly_present() -> None:
    record = _records(_evidence())[("deal-g", ASSIGN)]
    assert record["status"] == absence.STATUS_POSSIBLE
    assert record["candidates"][0]["score"] == 1.0
    assert record["candidates"][0]["ref"]["clause_path"] == "4"


def test_a_signed_copy_with_no_readable_text_is_unverifiable_not_absent() -> None:
    for tid in (ASSIGN, TERM):
        record = _records(_evidence())[("deal-e", tid)]
        assert record["status"] == absence.STATUS_POSSIBLE
        assert record["basis"] == absence.BASIS_UNVERIFIABLE
        assert record["candidates"] == []


def test_a_signed_copy_with_big_unreadable_passages_is_unverifiable() -> None:
    """A page that extracted as columns of digits may hold the clause: no absence
    can be verified from that copy, however clean the rest of it reads."""
    junk = " ".join(str(1000 + i) for i in range(300))  # ~1,500 characters of digits
    # Mining marks a node of digits no clause (numbering_only): it is left out of
    # the nodes examined, yet still counts as unreadable text.
    assert _non_clause(junk) is not None
    obs = _observations() + [_obs("deal-b", None, junk, path="30", non_clause=_non_clause(junk))]
    records = _records(_evidence(obs))
    assert records[("deal-b", ASSIGN)]["status"] == absence.STATUS_POSSIBLE
    assert records[("deal-b", ASSIGN)]["basis"] == absence.BASIS_UNVERIFIABLE
    # other deals are unaffected, and a short numeric table is no junk
    assert records[("deal-f", ASSIGN)]["status"] == absence.STATUS_VERIFIED
    table = " ".join(str(1000 + i) for i in range(20))
    assert _non_clause(table) is not None
    small = _observations() + [
        _obs("deal-b", None, table, path="30", non_clause=_non_clause(table))
    ]
    assert _records(_evidence(small))[("deal-b", ASSIGN)]["status"] == absence.STATUS_VERIFIED


def test_only_signed_deals_without_a_signed_text_have_a_record() -> None:
    records = _records(_evidence())
    deals = {doc for doc, _tid in records}
    assert "deal-d" not in deals  # no executed copy
    assert ("deal-a", ASSIGN) not in records and ("deal-a", TERM) not in records
    assert ("deal-b", TERM) not in records  # it signed a term
    # every other signed deal x clause pair has exactly one record
    assert len(records) == 7 * 2 - 2 - 5  # 14 pairs - deal-a's two - five deals that signed a term


def test_a_clause_struck_before_signing_still_counts_as_signed_without_it() -> None:
    struck = _observations() + [
        _obs("deal-b", ASSIGN, ASSIGN_STD, path="9", outcome="opening"),
    ]
    evidence = _evidence(struck)
    precedent = {(p["document_id"], p["taxonomy_id"]): p for p in evidence["precedent"]}
    assert precedent[("deal-b", ASSIGN)]["signed_text"] is None  # it opened with it, struck it
    assert _records(evidence)[("deal-b", ASSIGN)]["status"] == absence.STATUS_VERIFIED


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_the_records_are_a_pure_function_of_the_store() -> None:
    first = _evidence()
    assert _evidence() == first
    shuffled = _observations()
    random.Random(7).shuffle(shuffled)
    assert _evidence(shuffled)["absence"] == first["absence"]


def test_records_are_sorted_and_candidates_best_first_and_capped() -> None:
    records = _evidence()["absence"]["records"]
    keys = [(r["document_id"], r["taxonomy_id"]) for r in records]
    assert keys == sorted(keys)
    many = _observations() + [
        _obs("deal-b", None, ASSIGN_NEAR.replace("Neither", f"N{i}"), path=f"{20 + i}")
        for i in range(5)
    ]
    candidates = _records(_evidence(many))[("deal-b", ASSIGN)]["candidates"]
    assert len(candidates) == absence.MAX_CANDIDATES
    scores = [c["score"] for c in candidates]
    assert scores == sorted(scores, reverse=True)


def test_paper_side_never_changes_the_records() -> None:
    flipped = [
        Observation(**{**o.__dict__, "provenance": "counterparty_paper"}) for o in _observations()
    ]
    assert _evidence(flipped)["absence"] == _evidence()["absence"]


# ---------------------------------------------------------------------------
# The playbook: schema, validator, digest
# ---------------------------------------------------------------------------


def test_the_assembled_playbook_validates_and_carries_the_guard() -> None:
    playbook = _playbook()
    result = validate_document(playbook)
    assert result.ok, [str(e) for e in result.errors if e.blocking]
    assert playbook["evidence"]["absence"]["guard"] == absence.ABSENCE_GUARD


def test_the_digest_counts_verified_and_possible_per_clause() -> None:
    digest = build_digest_v5(_playbook())
    by_tid = {c["taxonomy_id"]: c for c in digest["clauses"]}
    # assignment: b, f verified; c, e (unverifiable), g, h possibly present
    assert by_tid[ASSIGN]["n_signed_without_verified"] == 2
    assert by_tid[ASSIGN]["n_signed_without_possible"] == 4
    assert [d["document_id"] for d in by_tid[ASSIGN]["verified_absent_deals"]] == [
        "deal-b",
        "deal-f",
    ]
    # term: only deal-e (unreadable) signed without it
    assert by_tid[TERM]["n_signed_without_verified"] == 0
    assert by_tid[TERM]["n_signed_without_possible"] == 1
    assert by_tid[TERM]["verified_absent_deals"] == []
    entry = by_tid[ASSIGN]["verified_absent_deals"][0]
    assert entry["absence_id"] == absence.absence_id("fixture", "deal-b", ASSIGN)
    assert set(entry) == {"document_id", "signed_on", "absence_id"}


def test_verified_and_possible_plus_deals_with_a_text_are_every_signed_deal() -> None:
    playbook = _playbook()
    digest = build_digest_v5(playbook)
    n_signed = digest["corpus"]["n_signed"]
    for clause in digest["clauses"]:
        signed_with = len(
            {
                p["document_id"]
                for p in playbook["evidence"]["precedent"]
                if p["taxonomy_id"] == clause["taxonomy_id"] and p["signed"] and p["signed_text"]
            }
        )
        assert (
            clause["n_signed_without_verified"] + clause["n_signed_without_possible"] + signed_with
            == n_signed
        )


def test_the_verified_deals_list_is_capped_like_the_other_lists_but_the_count_is_not() -> None:
    obs = _observations() + [_obs(f"deal-x{i}", TERM, TERM_STD, path="2") for i in range(8)]
    docs_extra = [_doc(f"deal-x{i}") for i in range(8)]
    positions, _, _ = compile_clause_positions(obs, _template())
    playbook = assemble_playbook(
        agreement_type={"id": "fixture", "name": "Fixture Agreement"},
        baseline={"has_canonical_template": True},
        taxonomy=_TAXONOMY,
        clause_positions=positions,
        corpus_documents=[*copy.deepcopy(_DOCS), *docs_extra],
        generated_at="2026-01-01T00:00:00Z",
        observations=obs,
    )
    entry = {c["taxonomy_id"]: c for c in playbook["digest"]["clauses"]}[ASSIGN]
    assert entry["n_signed_without_verified"] == 10
    assert len(entry["verified_absent_deals"]) == 5
    uncapped = {c["taxonomy_id"]: c for c in build_digest_uncapped(playbook)["clauses"]}
    assert len(uncapped[ASSIGN]["verified_absent_deals"]) == 10
    assert digest_token_estimate(playbook["digest"]) > 0


def _errors(doc: dict[str, Any]) -> str:
    return "\n".join(str(e) for e in validate_document(doc).errors if e.blocking)


def test_the_validator_requires_the_absence_section() -> None:
    playbook = _playbook()
    del playbook["evidence"]["absence"]
    assert "'absence' is a required property" in _errors(playbook)


def test_the_validator_refuses_a_missing_record() -> None:
    playbook = _playbook()
    playbook["evidence"]["absence"]["records"].pop(0)
    assert "has no record for 1 signed deal / clause pair" in _errors(playbook)


def test_the_validator_refuses_a_record_for_a_deal_that_signed_the_clause() -> None:
    playbook = _playbook()
    record = copy.deepcopy(playbook["evidence"]["absence"]["records"][0])
    record.update(
        document_id="deal-a",
        taxonomy_id=ASSIGN,
        id=absence.absence_id("fixture", "deal-a", ASSIGN),
    )
    playbook["evidence"]["absence"]["records"].append(record)
    assert "it did not sign without it" in _errors(playbook)


def test_the_validator_refuses_a_record_for_an_unsigned_deal() -> None:
    playbook = _playbook()
    record = copy.deepcopy(playbook["evidence"]["absence"]["records"][0])
    record.update(
        document_id="deal-d", id=absence.absence_id("fixture", "deal-d", record["taxonomy_id"])
    )
    playbook["evidence"]["absence"]["records"].append(record)
    assert "not an in-scope deal with a detected executed copy" in _errors(playbook)


def test_the_validator_refuses_a_status_the_evidence_does_not_support() -> None:
    playbook = _playbook()
    records = playbook["evidence"]["absence"]["records"]
    verified = next(r for r in records if r["status"] == absence.STATUS_VERIFIED)
    verified["status"] = absence.STATUS_POSSIBLE
    assert "implies 'absent_verified'" in _errors(playbook)
    verified["status"] = absence.STATUS_VERIFIED
    possible = next(r for r in records if r["candidates"])
    possible["status"] = absence.STATUS_VERIFIED
    assert "implies 'possibly_present'" in _errors(playbook)


def test_the_validator_refuses_a_wrong_id_a_duplicate_and_a_weak_or_misordered_candidate() -> None:
    playbook = _playbook()
    records = playbook["evidence"]["absence"]["records"]
    records[0]["id"] = "abs." + "0" * 16
    assert "does not match the id recomputed" in _errors(playbook)
    playbook = _playbook()
    records = playbook["evidence"]["absence"]["records"]
    records.append(copy.deepcopy(records[0]))
    assert "duplicate absence id" in _errors(playbook)
    playbook = _playbook()
    possible = next(r for r in playbook["evidence"]["absence"]["records"] if r["candidates"])
    possible["candidates"][0]["score"] = 0.01
    assert "below the section's guard" in _errors(playbook)
    playbook = _playbook()
    possible = next(
        r for r in playbook["evidence"]["absence"]["records"] if len(r["candidates"]) == 0
    )
    possible["candidates"] = [
        {"ref": {"document_id": "deal-a", "version": 3, "clause_path": "9"}, "score": 0.5}
    ]
    assert "not the record's deal" in _errors(playbook)


def test_the_validator_refuses_a_digest_that_disagrees_with_the_absence_records() -> None:
    playbook = _playbook()
    playbook["digest"]["clauses"][0]["n_signed_without_verified"] += 1
    assert "digest does not equal build_digest_v5" in _errors(playbook)


# ---------------------------------------------------------------------------
# The spot-check: absence_check
# ---------------------------------------------------------------------------


def _subjects(limit: int = absence.CHECK_SAMPLE_MAX) -> list[absence.AbsenceSubject]:
    evidence = _evidence()
    return absence.check_subjects(
        clauses=evidence["clauses"],
        records=evidence["absence"]["records"],
        observations=_observations(),
        limit=limit,
    )


def test_the_sample_is_every_verified_absence_when_there_are_few() -> None:
    verified = [
        r for r in _evidence()["absence"]["records"] if r["status"] == absence.STATUS_VERIFIED
    ]
    subjects = _subjects()
    assert len(subjects) == len(verified)
    assert {s.absence_id for s in subjects} == {r["id"] for r in verified}


def test_a_larger_set_is_sampled_by_lowest_hash_deterministically() -> None:
    subjects = _subjects(limit=2)
    assert len(subjects) == 2
    assert [s.key for s in subjects] == sorted(s.key for s in _subjects())[:2]
    assert [s.key for s in _subjects(limit=2)] == [s.key for s in subjects]


def test_the_question_carries_the_clause_and_the_whole_signed_text_and_no_deal() -> None:
    subject = next(
        s for s in _subjects() if s.absence_id == absence.absence_id("fixture", "deal-b", ASSIGN)
    )
    payload = subject.payload
    assert payload["stage"] == absence.CHECK_KIND
    assert payload["clause_type"] == {"id": ASSIGN, "label": "Assignment"}
    assert payload["our_standard"] == ASSIGN_STD
    assert [n["text"] for n in payload["signed_text"]] == [TERM_STD, UNRELATED]
    assert [n["clause_type"] for n in payload["signed_text"]] == ["Term", None]
    blob = json.dumps(payload)
    assert "deal-b" not in blob and "our_paper" not in blob and "paper" not in payload


def test_the_verdict_kind_is_registered_and_validated() -> None:
    assert absence.CHECK_KIND in JUDGE_KINDS
    assert infer_verdict_kind({"absent": True}) == absence.CHECK_KIND
    validate_verdict(absence.CHECK_KIND, {"absent": False, "reason": "States it in section 9."})
    for bad in ({"absent": "yes"}, {}, {"absent": True, "reason": 3}):
        try:
            validate_verdict(absence.CHECK_KIND, bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad!r}")


def test_the_store_backed_judge_queues_a_miss_and_replays_a_verdict(tmp_path: Path) -> None:
    store = VerdictStore(tmp_path / "verdicts.jsonl")
    pending = PendingQueue(tmp_path / "pending.jsonl")
    judge = StoreBackedAbsenceCheckJudge(store=store, pending=pending)
    subjects = _subjects()
    assert [judge.judge(s) for s in subjects] == [None] * len(subjects)
    queued = [json.loads(x) for x in (tmp_path / "pending.jsonl").read_text().splitlines()]
    assert {q["key"] for q in queued} == {s.key for s in subjects}
    assert {q["kind"] for q in queued} == {absence.CHECK_KIND}
    assert len(subjects) == 2
    store.put_by_key(subjects[0].key, {"absent": True, "reason": "Nothing like it."})
    store.put_by_key(subjects[1].key, {"absent": "maybe"})  # malformed: re-queued, unchecked
    replay = StoreBackedAbsenceCheckJudge(store=store, pending=None)
    assert [replay.judge(s) for s in subjects] == [True, None]
    store.put_by_key(subjects[1].key, {"absent": False, "reason": "Section 7 states it."})
    assert [replay.judge(s) for s in subjects] == [True, False]


def test_the_scorecard_counts_the_records_and_the_check(tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    playbook = _playbook()
    (out / "playbook.opf.json").write_text(json.dumps(playbook), encoding="utf-8")
    (out / "observations.jsonl").write_text(
        "".join(json.dumps(o.to_dict()) + "\n" for o in _observations()), encoding="utf-8"
    )
    subjects = absence.check_subjects(
        clauses=playbook["evidence"]["clauses"],
        records=playbook["evidence"]["absence"]["records"],
        observations=_observations(),
    )
    store = VerdictStore(out / "judge" / "verdicts.jsonl")
    current = RubricStamp(kind=absence.CHECK_KIND, version=rubric_version(absence.CHECK_KIND))
    store.put_by_key(subjects[0].key, {"absent": True}, rubric=current)
    store.put_by_key(subjects[1].key, {"absent": False}, rubric=current)
    section = build_scorecard(out)["absence"]
    assert section["pairs"] == len(playbook["evidence"]["absence"]["records"])
    assert section["verified"] + section["possible"] == section["pairs"]
    assert section["unverifiable"] == 2
    assert section["digest_without_verified"] == section["verified"]
    assert section["digest_without_possible"] == section["possible"]
    assert section["check_sample"] == len(subjects)
    # a verdict of the current rubric counts; one saying "present" is a wrong absence
    assert section["checked"] == 2 and section["wrong"] == 1
    assert all(not isinstance(v, str) or v in {"other"} for v in section.values())
