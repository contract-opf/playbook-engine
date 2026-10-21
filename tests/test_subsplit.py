"""Agent-judged sentence-level sub-split of multi-topic nodes (issue #257).

Unit tests drive :mod:`playbook_engine.subsplit` on hand-built clause trees
with profiles learned by ``build_profiles`` from synthetic exemplars (the same
producer ``mine_corpus`` uses, at the sub-split's own dominance). Verdicts are
made the way production makes them: a draft through
``subsplit.validate_draft`` (what ``judge-apply --verdicts`` stores), a blind
check through ``subsplit_check.apply_check_records`` (what ``judge-apply
--check`` stores), replayed by the production ``StoreBackedSubsplitJudge``.
The integration tests at the bottom drive ``mine_corpus`` over RTF fixtures
with an injected segmenter, then the real CLI (``judge-apply --verdicts``,
``judge --check subsplit``, ``judge-apply --check``) and ``build_scorecard``.

SECURITY NOTE: synthetic, fictional text only, written for these tests.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from click.testing import CliRunner

from playbook_engine import subsplit as ss
from playbook_engine.agent_judge import (
    PendingQueue,
    StoreBackedSubsplitJudge,
    VerdictStore,
    infer_verdict_kind,
    validate_verdict,
)
from playbook_engine.clause_classifier import ClassifiedClause, ClauseClassification
from playbook_engine.clause_tree import ClauseNode, ClauseTree
from playbook_engine.cli import cli
from playbook_engine.config import load_config
from playbook_engine.equivalence_check import CHECKER_EFFORT, CHECKER_MODEL
from playbook_engine.observation_builder import read_observations_jsonl
from playbook_engine.pipeline import mine_corpus
from playbook_engine.playbook_assembler import _CORPUS_DOCUMENT_INTERNAL_KEYS
from playbook_engine.resegmentation import build_profiles, resegment
from playbook_engine.rubric import JUDGE_KINDS, RubricStamp, rubric_version
from playbook_engine.scorecard import build_scorecard
from playbook_engine.segmentation_grounding import Block, GroundingError, SegNode
from playbook_engine.subsplit_check import apply_check_records, build_check_queues
from playbook_engine.taxonomy import load_taxonomy
from tests.test_pipeline_llm_seg import _TAXONOMY_PATH, _write_rtf

# ---------------------------------------------------------------------------
# Synthetic corpus vocabulary
# ---------------------------------------------------------------------------

_TERM = [
    "This pact begins on the start date and stays in force for a first span of three school years.",
    "The pact begins on the start date and stays in force for a first span of two school years.",
    "This pact begins on its start date and stays in force for a first span of four school years.",
    "The pact begins on the start date and stays in force for a first span of five school years.",
]
_RENEWAL = [
    "When the first span ends the pact rolls over for further yearly spans unless a side gives "
    "word of non-renewal sixty days before the span ends.",
    "At the end of each span the pact rolls over for a further yearly span unless either side "
    "gives word of non-renewal ninety days before it ends.",
    "The pact rolls over for further yearly spans unless a side gives written word of "
    "non-renewal at least sixty days before the current span ends.",
    "Unless a side gives word of non-renewal thirty days before a span ends, the pact rolls "
    "over for a further yearly span.",
]
_ENTIRE = [
    "This writing is the whole bargain of the sides on its subject and replaces all earlier "
    "talks, letters and understandings.",
    "This writing sets out the whole bargain on its subject and replaces all earlier talks, "
    "letters and understandings of the sides.",
    "The whole bargain on this subject is this writing, which replaces all earlier talks, "
    "letters and understandings between the sides.",
    "This writing holds the whole bargain of the sides on its subject; it replaces all earlier "
    "understandings, talks and letters.",
]
_SEVER = [
    "If a court holds any term void, that term is read down to the extent the law allows and "
    "the remaining terms stay valid.",
    "A term a court holds void is read down to the extent the law allows, and the remaining "
    "terms stay valid and binding.",
    "Should a court hold a term void, it is read down to the extent the law allows; the "
    "remaining terms stay valid.",
    "Where a court holds a term void the term is read down to the extent the law allows and "
    "all remaining terms stay valid.",
]
_LAW = [
    "Disputes between the sides are decided under the statutes of the province where the "
    "hospital sits, and only the tribunals of that province may hear any such dispute.",
    "The statutes of the hospital's province decide every dispute between the hospital and "
    "the college, and only tribunals of that province may hear a dispute.",
    "Every dispute is decided under the statutes of the province where the hospital sits, "
    "and only the tribunals there have power to hear such disputes between the sides.",
    "Statutes of the province in which the hospital sits decide each dispute between the "
    "sides, and only the tribunals of that province may hear it.",
]

#: A term clause that also states the renewal, in one block (the mixed deal's).
#: Its renewal sentence carries too few renewal words for the word-count split
#: pass (#255/#256) to move it, which is the gap this issue closes.
_TERM_AND_RENEWAL = (
    _TERM[3] + " After that span, and subject to the board's budget review and the dean's "
    "consent, the pact rolls over for another year unless a side objects in good time."
)
#: An entire-agreement clause followed by severability, in one block.
_ENTIRE_AND_SEVER = (
    _ENTIRE[3] + " If a court or an arbitrator ever holds a term of this writing void or "
    "unworkable in practice, the sides will meet in good faith and that term is read down as "
    "the law allows."
)
#: A two-sentence term clause that touches renewal words but states one thing.
_TERM_ONLY = (
    _TERM[0] + " The span does not roll over by itself, and no side may stretch it past the "
    "last school day of its final year."
)

_LABELS = {
    "term": "Term",
    "renewal_term": "Renewal / Auto-Renewal",
    "entire_agreement": "Entire Agreement",
    "severability": "Severability",
    "governing_law": "Governing Law",
}


def _exemplars(deals: int = 3) -> list[tuple[str, str, str]]:
    out: list[tuple[str, str, str]] = []
    for i in range(deals):
        for tid, texts in (
            ("term", _TERM),
            ("renewal_term", _RENEWAL),
            ("entire_agreement", _ENTIRE),
            ("severability", _SEVER),
            ("governing_law", _LAW),
        ):
            out.append((f"deal-{i}", tid, texts[i]))
    return out


def _profiles() -> Any:
    return build_profiles(_exemplars(), dominance=ss.CANDIDATE_DOMINANCE)


def _taxonomy() -> Any:
    return load_taxonomy(_TAXONOMY_PATH)


def _rubric() -> str:
    return rubric_version(ss.SUBSPLIT_KIND, taxonomy=_taxonomy())


def _labels() -> dict[str, str]:
    return {e.id: e.label for e in _taxonomy().classifier_entries()}


def _node(path: str, text: str, start: int = 0) -> ClauseNode:
    return ClauseNode(
        clause_path=path, heading=None, text=text, char_span=(start, start + len(text))
    )


def _cls(tid: str | None) -> ClauseClassification:
    if tid is None:
        return ClauseClassification(taxonomy_id=None, confidence=0.0, basis="unclassified")
    return ClauseClassification(taxonomy_id=tid, confidence=0.9, basis="llm_segmenter")


def _version(nodes: list[ClauseNode], types: list[str | None]) -> tuple[ClauseTree, list[Any]]:
    tree = ClauseTree(document_id="d", version="v1", source_file="d.rtf", nodes=nodes)
    return tree, [ClassifiedClause(n, _cls(t)) for n, t in zip(nodes, types, strict=True)]


def _subject(text: str) -> ss.SubsplitSubject:
    return ss.subject_for(text, ss.allowed_types(_labels()), _rubric())


def _split_draft(*spans: tuple[str, str, str | None]) -> dict[str, Any]:
    return {
        "split": True,
        "spans": [{"start": s, "end": e, "taxonomy_id": t} for s, e, t in spans],
        "reason": "Two provisions of different types.",
        "basis": "agent",
    }


_NO_SPLIT = {"split": False, "reason": "One provision.", "basis": "agent"}


def _draft(store: VerdictStore, subject: ss.SubsplitSubject, draft: dict[str, Any]) -> None:
    """Store *draft* the way ``judge-apply --verdicts`` does."""
    stamp = RubricStamp(kind=ss.SUBSPLIT_KIND, version=subject.rubric_version or "")
    store.put_by_key(subject.key, ss.validate_draft(draft, subject.payload), rubric=stamp)


def _check(
    store: VerdictStore, subject: ss.SubsplitSubject, answer: dict[str, Any]
) -> tuple[int, int, int]:
    """Record a pinned checker's *answer* the way ``judge-apply --check`` does."""
    record = {
        "key": subject.key,
        "split": answer["split"],
        "spans": answer.get("spans", []),
        "reason": "Checked.",
        "model": CHECKER_MODEL,
        "effort": CHECKER_EFFORT,
    }
    result = apply_check_records(store, [(1, record)], {subject.key: subject})
    for key, verdict in result.updates:
        prior = store.get_record_by_key(key)
        store.put_by_key(key, verdict, rubric=prior.rubric if prior else None)
    return result.agreed, result.disagreed, result.adjudicated


def _judge(tmp_path: Path) -> StoreBackedSubsplitJudge:
    return StoreBackedSubsplitJudge(
        store=VerdictStore(tmp_path / "judge" / "verdicts.jsonl"),
        pending=PendingQueue(tmp_path / "judge" / "pending.jsonl"),
    )


def _apply(
    nodes: list[ClauseNode], types: list[str | None], judge: Any = None
) -> ss.SubsplitOutcome:
    tree, classified = _version(nodes, types)
    return ss.apply_subsplits(
        tree, classified, _profiles(), labels=_labels(), judge=judge, rubric_version=_rubric()
    )


# ---------------------------------------------------------------------------
# Candidates and the question
# ---------------------------------------------------------------------------


def test_the_strict_split_pass_leaves_the_co_located_pairs_mixed() -> None:
    """The gap this issue closes: #255/#256's word-count split does not cut them."""
    strict = build_profiles(_exemplars())
    nodes = [_node("1", _TERM_AND_RENEWAL), _node("2", _ENTIRE_AND_SEVER, 400)]
    tree, classified = _version(nodes, ["term", "entire_agreement"])
    res = resegment(tree, classified, strict)
    assert res.splits == 0
    assert [n.text for n in res.tree.nodes] == [_TERM_AND_RENEWAL, _ENTIRE_AND_SEVER]


def test_mixed_nodes_are_candidates_and_single_topic_ones_are_not() -> None:
    profiles = _profiles()
    assert ss.candidate_types(_node("1", _TERM_AND_RENEWAL), "term", profiles) == ["renewal_term"]
    assert ss.candidate_types(_node("1", _ENTIRE_AND_SEVER), "entire_agreement", profiles) == [
        "severability"
    ]
    # One sentence, one topic: nothing to ask.
    assert ss.candidate_types(_node("1", _LAW[3]), "governing_law", profiles) == []
    # A node the split or join pass made is left alone.
    made = _node("1", _TERM_AND_RENEWAL)
    made.x_resegmented = {"split_from": "1", "basis": "block_profile"}
    assert ss.candidate_types(made, "term", profiles) == []


def test_the_question_is_keyed_on_text_and_type_list_and_numbers_the_units() -> None:
    subject = _subject(_TERM_AND_RENEWAL)
    payload = subject.payload
    assert payload["stage"] == ss.SUBSPLIT_KIND
    assert [u["id"] for u in payload["units"]] == ["u0", "u1"]
    assert payload["units"][1]["text"].startswith("After that span")
    assert {t["id"] for t in payload["taxonomy"]} == set(_labels())
    # Same text and types, same key; the rubric is not part of it.
    again = ss.subject_for(_TERM_AND_RENEWAL, ss.allowed_types(_labels()), "other-rubric")
    assert again.key == subject.key
    assert _subject(_ENTIRE_AND_SEVER).key != subject.key


def test_a_leading_heading_line_is_never_a_unit_of_its_own() -> None:
    text = "5. Term\n" + _TERM_AND_RENEWAL
    units = ss.subsplit_units(text)
    assert len(units) == 2
    assert text[units[0][0] : units[0][1]].startswith("5. Term\nThe pact begins")


# ---------------------------------------------------------------------------
# Grounding
# ---------------------------------------------------------------------------


def test_grounding_rejects_spans_that_do_not_cover_the_text_exactly() -> None:
    payload = _subject(_TERM_AND_RENEWAL).payload
    # Covers u0 only: the text has two units.
    with pytest.raises(GroundingError, match="cover the text exactly"):
        ss.ground(_split_draft(("u0", "u0", "term"), ("u1", "u1", "renewal_term"))
                  | {"spans": [{"start": "u0", "end": "u0", "taxonomy_id": "term"},
                               {"start": "u1", "end": "u2", "taxonomy_id": "renewal_term"}]},
                  payload)  # fmt: skip
    with pytest.raises(GroundingError, match="contiguous"):
        ss.ground(_split_draft(("u0", "u0", "term"), ("u2", "u2", "renewal_term")), payload)
    with pytest.raises(GroundingError, match="empty"):
        ss.ground(_split_draft(("u0", "u1", "term"), ("u2", "u1", "renewal_term")), payload)
    with pytest.raises(GroundingError, match="not in the allowed list"):
        ss.ground(_split_draft(("u0", "u0", "term"), ("u1", "u1", "made_up_type")), payload)
    with pytest.raises(GroundingError, match="same taxonomy_id"):
        ss.ground(_split_draft(("u0", "u0", "term"), ("u1", "u1", "term")), payload)
    with pytest.raises(GroundingError, match="at least two"):
        ss.ground({"split": True, "spans": [{"start": "u0", "end": "u1", "taxonomy_id": "term"}]},
                  payload)  # fmt: skip
    # Units that are not the text's own sentences never ground.
    tampered = {**payload, "units": [{"id": "u0", "text": _TERM_AND_RENEWAL}]}
    with pytest.raises(GroundingError):
        ss.ground(_split_draft(("u0", "u0", "term"), ("u1", "u1", "renewal_term")), tampered)
    # The well-formed partition grounds.
    assert ss.ground(_split_draft(("u0", "u0", "term"), ("u1", "u1", "renewal_term")), payload) == [
        (0, 0, "term"),
        (1, 1, "renewal_term"),
    ]
    assert ss.ground(_NO_SPLIT, payload) == []


def test_a_draft_that_does_not_ground_is_refused_at_apply_time() -> None:
    payload = _subject(_TERM_AND_RENEWAL).payload
    with pytest.raises(ValueError, match="grounding"):
        ss.validate_draft(_split_draft(("u0", "u0", "term"), ("u1", "u3", "renewal_term")), payload)
    with pytest.raises(ValueError, match="unexpected field"):
        ss.validate_draft({**_NO_SPLIT, "check": {"agreed": True}})
    with pytest.raises(ValueError, match="basis"):
        ss.validate_draft({**_NO_SPLIT, "basis": "guess"})
    assert ss.validate_draft(_NO_SPLIT) == {
        "split": False,
        "reason": "One provision.",
        "basis": "agent",
    }


def test_subsplit_is_a_registered_verdict_and_rubric_kind() -> None:
    assert ss.SUBSPLIT_KIND in JUDGE_KINDS
    assert infer_verdict_kind(_NO_SPLIT) == ss.SUBSPLIT_KIND
    validate_verdict(ss.SUBSPLIT_KIND, _split_draft(("u0", "u0", "term"), ("u1", "u1", None)))
    with pytest.raises(ValueError):
        validate_verdict(ss.SUBSPLIT_KIND, {"split": "yes", "reason": "x", "basis": "agent"})


# ---------------------------------------------------------------------------
# Apply
# ---------------------------------------------------------------------------


def test_a_term_and_renewal_block_is_split_into_two_typed_spans(tmp_path: Path) -> None:
    judge = _judge(tmp_path)
    nodes = [_node("1", _TERM_AND_RENEWAL, 10), _node("2", _LAW[3], 400)]
    subject = _subject(_TERM_AND_RENEWAL)
    answer = _split_draft(("u0", "u0", "term"), ("u1", "u1", "renewal_term"))
    _draft(judge.store, subject, answer)
    assert _check(judge.store, subject, answer) == (1, 0, 0)

    out = _apply(nodes, ["term", "governing_law"], judge)
    assert (out.candidates, out.split, out.no_split, out.spans_added) == (1, 1, 0, 1)
    first = out.tree.resolve_path("1")
    second = out.tree.resolve_path("1.s1")
    assert first is not None and second is not None
    assert first.text == _TERM[3]
    assert second.text.startswith("After that span") and second.text.endswith("in good time.")
    # Char-span citations into the original block.
    assert first.char_span == (10, 10 + len(_TERM[3]))
    start = _TERM_AND_RENEWAL.index("After that span")
    assert second.char_span == (10 + start, 10 + len(_TERM_AND_RENEWAL))
    for node in (first, second):
        meta = node.x_resegmented
        assert meta is not None
        assert meta["subsplit_from"] == "1" and meta["basis"] == ss.BASIS_AGENT
        assert meta["origin_span"] == [10, 10 + len(_TERM_AND_RENEWAL)]
        assert meta["key"] == subject.key
    by_path = {cc.node.clause_path: cc.classification for cc in out.classified}
    assert by_path["1"].taxonomy_id == "term" and by_path["1"].basis == "llm_segmenter"
    assert by_path["1.s1"].taxonomy_id == "renewal_term"
    assert by_path["1.s1"].basis == ss.CLASSIFICATION_BASIS
    assert [s.key for s in out.subjects] == [subject.key]


def test_an_entire_agreement_and_severability_block_is_split(tmp_path: Path) -> None:
    judge = _judge(tmp_path)
    subject = _subject(_ENTIRE_AND_SEVER)
    answer = _split_draft(("u0", "u0", "entire_agreement"), ("u1", "u1", "severability"))
    _draft(judge.store, subject, answer)
    _check(judge.store, subject, answer)

    out = _apply([_node("7", _ENTIRE_AND_SEVER)], ["entire_agreement"], judge)
    assert (out.split, out.spans_added) == (1, 1)
    by_path = {cc.node.clause_path: cc for cc in out.classified}
    assert by_path["7"].node.text == _ENTIRE[3]
    assert by_path["7.s1"].classification.taxonomy_id == "severability"
    assert by_path["7.s1"].node.text.startswith("If a court or an arbitrator")


def test_a_single_topic_block_answered_no_split_is_left_whole(tmp_path: Path) -> None:
    judge = _judge(tmp_path)
    text = _TERM_ONLY
    assert ss.candidate_types(_node("1", text), "term", _profiles())
    subject = _subject(text)
    _draft(judge.store, subject, _NO_SPLIT)
    _check(judge.store, subject, _NO_SPLIT)

    out = _apply([_node("1", text)], ["term"], judge)
    assert (out.candidates, out.split, out.no_split, out.spans_added) == (1, 0, 1, 0)
    (node,) = out.tree.all_nodes()
    assert node.text == text and node.x_resegmented is None


def test_only_a_settled_verdict_is_applied(tmp_path: Path) -> None:
    judge = _judge(tmp_path)
    subject = _subject(_TERM_AND_RENEWAL)
    answer = _split_draft(("u0", "u0", "term"), ("u1", "u1", "renewal_term"))
    nodes = [_node("1", _TERM_AND_RENEWAL)]

    # No verdict: queued, nothing changes.
    out = _apply(nodes, ["term"], judge)
    assert out.split == 0 and out.candidates == 1
    queued = [
        json.loads(x) for x in (tmp_path / "judge" / "pending.jsonl").read_text().splitlines()
    ]
    assert [(q["kind"], q["key"], q["rubric_version"]) for q in queued] == [
        (ss.SUBSPLIT_KIND, subject.key, _rubric())
    ]

    # A draft nobody checked: not applied, and not queued again.
    _draft(judge.store, subject, answer)
    fresh = StoreBackedSubsplitJudge(
        store=judge.store, pending=PendingQueue(tmp_path / "judge" / "pending2.jsonl")
    )
    assert _apply(nodes, ["term"], fresh).split == 0
    assert not (tmp_path / "judge" / "pending2.jsonl").exists()

    # A disputed draft: still not applied.
    other = _split_draft(("u0", "u0", "term"), ("u1", "u1", "termination_convenience"))
    assert _check(judge.store, subject, other) == (0, 1, 0)
    assert _apply(nodes, ["term"], fresh).split == 0

    # Adjudicated: applied with the adjudicator's spans.
    assert _check(judge.store, subject, answer) == (0, 0, 1)
    out = _apply(nodes, ["term"], fresh)
    assert out.split == 1
    by_path = {cc.node.clause_path: cc.classification.taxonomy_id for cc in out.classified}
    assert by_path == {"1": "term", "1.s1": "renewal_term"}


def test_an_owner_verdict_is_applied_unchecked_and_a_null_span_stays_unclassified(
    tmp_path: Path,
) -> None:
    judge = _judge(tmp_path)
    subject = _subject(_TERM_AND_RENEWAL)
    _draft(judge.store, subject, {**_split_draft(("u0", "u0", "term"), ("u1", "u1", None)),
                                  "basis": "owner"})  # fmt: skip
    out = _apply([_node("1", _TERM_AND_RENEWAL)], ["term"], judge)
    by_path = {cc.node.clause_path: cc.classification for cc in out.classified}
    assert by_path["1.s1"].taxonomy_id is None and by_path["1.s1"].basis == "unclassified"


def test_a_stored_verdict_under_a_moved_rubric_is_requeued(tmp_path: Path) -> None:
    judge = _judge(tmp_path)
    subject = _subject(_TERM_AND_RENEWAL)
    stale = ss.validate_draft({**_NO_SPLIT, "basis": "owner"}, subject.payload)
    judge.store.put_by_key(
        subject.key, stale, rubric=RubricStamp(kind=ss.SUBSPLIT_KIND, version="v0+000000000000")
    )
    out = _apply([_node("1", _TERM_AND_RENEWAL)], ["term"], judge)
    assert out.no_split == 0
    queued = (tmp_path / "judge" / "pending.jsonl").read_text().splitlines()
    assert len(queued) == 1


def test_without_a_judge_candidates_are_counted_and_nothing_changes() -> None:
    nodes = [_node("1", _TERM_AND_RENEWAL)]
    out = _apply(nodes, ["term"])
    assert (out.candidates, out.split, out.subjects) == (1, 0, [])
    assert [n.text for n in out.tree.all_nodes()] == [_TERM_AND_RENEWAL]


def test_apply_is_pure_and_two_runs_are_byte_identical(tmp_path: Path) -> None:
    judge = _judge(tmp_path)
    for text, spans in (
        (_TERM_AND_RENEWAL, (("u0", "u0", "term"), ("u1", "u1", "renewal_term"))),
        (_ENTIRE_AND_SEVER, (("u0", "u0", "entire_agreement"), ("u1", "u1", "severability"))),
    ):
        subject = _subject(text)
        _draft(judge.store, subject, _split_draft(*spans))
        _check(judge.store, subject, _split_draft(*spans))
    nodes = [_node("1", _TERM_AND_RENEWAL), _node("2", _ENTIRE_AND_SEVER, 300)]
    tree, classified = _version(nodes, ["term", "entire_agreement"])
    before = tree.to_json()

    def run() -> str:
        out = ss.apply_subsplits(
            tree, classified, _profiles(), labels=_labels(), judge=judge, rubric_version=_rubric()
        )
        return json.dumps(
            {
                "tree": out.tree.to_dict(),
                "classified": [cc.to_dict() for cc in out.classified],
                "subjects": [s.to_dict() for s in out.subjects],
            },
            sort_keys=True,
        )

    first, second = run(), run()
    assert first == second
    assert tree.to_json() == before  # the input is never mutated


# ---------------------------------------------------------------------------
# Check protocol
# ---------------------------------------------------------------------------


def test_check_queues_hold_the_question_without_the_draft(tmp_path: Path) -> None:
    store = VerdictStore(tmp_path / "verdicts.jsonl")
    a, b, c = (
        _subject(_TERM_AND_RENEWAL),
        _subject(_ENTIRE_AND_SEVER),
        _subject(_LAW[3] + " " + _LAW[2]),
    )
    _draft(store, a, _split_draft(("u0", "u0", "term"), ("u1", "u1", "renewal_term")))
    _draft(store, b, _split_draft(("u0", "u0", "entire_agreement"), ("u1", "u1", "severability")))
    _check(store, b, _NO_SPLIT)  # disputed
    queues = build_check_queues(store, [a, b, c])
    assert [i["key"] for i in queues.check] == [a.key]
    assert queues.check[0]["payload"] == a.payload  # no draft in the blind item
    assert [i["key"] for i in queues.adjudication] == [b.key]
    adj = queues.adjudication[0]["payload"]
    assert adj["draft"]["split"] is True and adj["check"]["split"] is False
    assert queues.undrafted == 1


def test_a_check_answer_is_grounded_and_must_come_from_the_pinned_checker(tmp_path: Path) -> None:
    store = VerdictStore(tmp_path / "verdicts.jsonl")
    subject = _subject(_TERM_AND_RENEWAL)
    _draft(store, subject, _NO_SPLIT)
    bad_spans = {
        "key": subject.key,
        "split": True,
        "spans": [{"start": "u0", "end": "u0", "taxonomy_id": "term"}],
        "reason": "x",
        "model": CHECKER_MODEL,
        "effort": CHECKER_EFFORT,
    }
    with pytest.raises(ValueError, match="line 3: grounding"):
        apply_check_records(store, [(3, bad_spans)], {subject.key: subject})
    other_model = {**bad_spans, "split": False, "spans": [], "model": "some-other-model"}
    with pytest.raises(ValueError, match="checker model"):
        apply_check_records(store, [(1, other_model)], {subject.key: subject})


# ---------------------------------------------------------------------------
# Integration: mine_corpus, the CLI and the scorecard
# ---------------------------------------------------------------------------

_HEADINGS = {
    "1. Term": "term",
    "2. Entire Agreement": "entire_agreement",
    "3. Governing Law": "governing_law",
    "4. Renewal": "renewal_term",
    "5. Severability": "severability",
}


def _deal_body(i: int) -> str:
    if i == 3:  # the mixed deal: renewal inside the term, severability inside the merger
        return (
            rf"1. Term\par {_TERM_AND_RENEWAL}\par "
            rf"2. Entire Agreement\par {_ENTIRE_AND_SEVER}\par "
            rf"3. Governing Law\par {_LAW[i]}\par "
        )
    return (
        rf"1. Term\par {_TERM[i]}\par "
        rf"2. Entire Agreement\par {_ENTIRE[i]}\par "
        rf"3. Governing Law\par {_LAW[i]}\par "
        rf"4. Renewal\par {_RENEWAL[i]}\par "
        rf"5. Severability\par {_SEVER[i]}\par "
    )


def _segment_fn(canonical_text: str, blocks: list[Block]) -> list[SegNode]:
    del canonical_text
    nodes = []
    for n, i in enumerate(range(0, len(blocks), 2), start=1):
        head, body = blocks[i], blocks[i + 1]
        nodes.append(
            SegNode(
                node_id=f"n{n}",
                parent_id=None,
                order=n,
                heading=head.text,
                taxonomy_id=_HEADINGS[head.text],
                start_block_id=head.block_id,
                end_block_id=body.block_id,
                start_quote=head.text[:6],
                end_quote=body.text[-10:],
            )
        )
    return nodes


def _corpus(tmp_path: Path) -> tuple[Path, Path]:
    corpus = tmp_path / "corpus"
    for i in range(4):
        deal = corpus / f"deal-00{i + 1}"
        deal.mkdir(parents=True, exist_ok=True)
        _write_rtf(deal / "v1.rtf", _deal_body(i))
    cfg = {
        "agreement_type": {"id": "educational-affiliation", "name": "Educational Affiliation"},
        "baseline": {"template": None},
        "taxonomy": str(_TAXONOMY_PATH),
        "provenance": {"our_party_aliases": ["Alpha Corp"]},
    }
    config_path = tmp_path / "playbook.config.yaml"
    config_path.write_text(yaml.dump(cfg), encoding="utf-8")
    return corpus, config_path


def _mine(corpus: Path, config_path: Path, out: Path) -> list[dict[str, Any]]:
    judge = StoreBackedSubsplitJudge(
        store=VerdictStore(out / "judge" / "verdicts.jsonl"),
        pending=PendingQueue(out / "judge" / "pending.jsonl"),
    )
    (out / "judge" / "pending.jsonl").unlink(missing_ok=True)
    mine_corpus(
        corpus_dir=corpus,
        config=load_config(config_path),
        taxonomy=_taxonomy(),
        out_dir=out,
        use_llm_segmentation=True,
        llm_segment_fn=_segment_fn,
        subsplit_judge=judge,
    )
    return read_observations_jsonl(out / "observations.jsonl")


def _cli(*args: str) -> Any:
    result = CliRunner().invoke(cli, list(args))
    assert result.exit_code == 0, result.output
    return result


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return path


def _drain(out: Path, tmp_path: Path) -> None:
    """Draft and blind-check every queued subsplit item through the real CLI."""
    pending = [
        json.loads(x)
        for x in (out / "judge" / "pending.jsonl").read_text().splitlines()
        if json.loads(x)["kind"] == ss.SUBSPLIT_KIND
    ]
    answers: dict[str, dict[str, Any]] = {}
    for item in pending:
        units = item["payload"]["units"]
        text = item["payload"]["text"]
        # The agent segmenter's node text opens with its heading line, which
        # the first unit carries (subsplit_units folds it in).
        if text.endswith(_TERM_AND_RENEWAL):
            answers[item["key"]] = _split_draft(("u0", "u0", "term"), ("u1", "u1", "renewal_term"))
        elif text.endswith(_ENTIRE_AND_SEVER):
            answers[item["key"]] = _split_draft(
                ("u0", "u0", "entire_agreement"), ("u1", f"u{len(units) - 1}", "severability")
            )
        else:
            answers[item["key"]] = dict(_NO_SPLIT)
    drafts = _write_jsonl(
        tmp_path / "drafts.jsonl", [{"key": k, "verdict": v} for k, v in answers.items()]
    )
    _cli("judge-apply", str(out), "--verdicts", str(drafts))
    config_path = tmp_path / "playbook.config.yaml"
    _cli("judge", str(out), "--config", str(config_path), "--check", "subsplit")
    blind = [
        json.loads(x)
        for x in (out / "judge" / "subsplit-check-pending.jsonl").read_text().splitlines()
    ]
    assert {item["key"] for item in blind} == set(answers)
    assert all("draft" not in item["payload"] for item in blind)
    checks = _write_jsonl(
        tmp_path / "checks.jsonl",
        [
            {
                "key": item["key"],
                "split": answers[item["key"]]["split"],
                "spans": answers[item["key"]].get("spans", []),
                "reason": "Independent answer.",
                "model": CHECKER_MODEL,
                "effort": CHECKER_EFFORT,
            }
            for item in blind
        ],
    )
    _cli("judge-apply", str(out), "--check", str(checks))


def test_mine_queues_drafts_checks_and_applies_the_subsplit(tmp_path: Path) -> None:
    corpus, config_path = _corpus(tmp_path)
    out = tmp_path / "out"
    rows = _mine(corpus, config_path, out)

    # First round: the mixed deal's two nodes are asked about, nothing changes yet.
    queued = [json.loads(x) for x in (out / "judge" / "pending.jsonl").read_text().splitlines()]
    sub_items = [q for q in queued if q["kind"] == ss.SUBSPLIT_KIND]
    texts = {q["payload"]["text"] for q in sub_items}
    assert "1. Term\n" + _TERM_AND_RENEWAL in texts
    assert "2. Entire Agreement\n" + _ENTIRE_AND_SEVER in texts
    subjects = ss.load_subjects(out / "judge")
    assert {s.key for s in subjects} == {q["key"] for q in sub_items}
    assert not [r for r in rows if "subsplit_from" in (r.get("x_resegmented") or {})]

    _drain(out, tmp_path)
    rows = _mine(corpus, config_path, out)
    mixed = {
        r["citation"]["clause_path"]: r for r in rows if r["citation"]["document_id"] == "deal-004"
    }
    assert mixed["1"]["taxonomy_id"] == "term"
    assert mixed["1"]["full_text"] == "1. Term\n" + _TERM[3]
    assert mixed["1.s1"]["taxonomy_id"] == "renewal_term"
    assert mixed["1.s1"]["x_classification_basis"] == ss.CLASSIFICATION_BASIS
    assert mixed["1.s1"]["x_resegmented"]["subsplit_from"] == "1"
    assert mixed["2"]["full_text"] == "2. Entire Agreement\n" + _ENTIRE[3]
    assert mixed["2.s1"]["taxonomy_id"] == "severability"
    # The clean deals are untouched.
    assert not [
        r for r in rows if r["citation"]["document_id"] != "deal-004" and "x_resegmented" in r
    ]
    # The persisted tree cites inside the original block.
    tree = ClauseTree.load(out / "normalized" / "deal-004" / "v1.clauses.json")
    cut = tree.resolve_path("1.s1")
    assert cut is not None and cut.x_resegmented is not None
    origin = cut.x_resegmented["origin_span"]
    assert origin[0] <= cut.char_span[0] and cut.char_span[1] <= origin[1]

    manifest = json.loads((out / "corpus_manifest.json").read_text())
    counts = {d["document_id"]: d["subsplit"] for d in manifest}
    assert counts["deal-004"] == {"candidates": 2, "split": 2, "no_split": 0, "spans_added": 2}
    assert "subsplit" in _CORPUS_DOCUMENT_INTERNAL_KEYS

    card = build_scorecard(out)["subsplit"]
    assert card["split"] == 2 and card["spans_added"] == 2
    assert card["observations_subsplit"] >= 2
    assert card["questions"] == len(subjects)
    assert card["answered_split"] == 2
    assert card["answered_split"] + card["answered_no_split"] == card["questions"]
    assert card["undrafted"] == 0
    assert card["check"]["agreed"] == card["questions"] and card["agreement_rate"] == 1.0


def test_two_mines_over_the_same_store_are_byte_identical(tmp_path: Path) -> None:
    corpus, config_path = _corpus(tmp_path)
    first = tmp_path / "a" / "out"
    _mine(corpus, config_path, first)
    _drain(first, tmp_path)
    _mine(corpus, config_path, first)

    second = tmp_path / "b" / "out"
    (second / "judge").mkdir(parents=True)
    (second / "judge" / "verdicts.jsonl").write_bytes(
        (first / "judge" / "verdicts.jsonl").read_bytes()
    )
    _mine(corpus, config_path, second)
    for name in ("observations.jsonl", "corpus_manifest.json", "judge/subsplit-subjects.jsonl"):
        assert (first / name).read_bytes() == (second / name).read_bytes(), name
    trees_a = sorted((first / "normalized").rglob("*.clauses.json"))
    trees_b = sorted((second / "normalized").rglob("*.clauses.json"))
    assert [t.name for t in trees_a] == [t.name for t in trees_b]
    for a, b in zip(trees_a, trees_b, strict=True):
        assert a.read_bytes() == b.read_bytes()
    # A warm re-mine (the stage cache) reproduces it too, and queues nothing.
    before = (first / "observations.jsonl").read_bytes()
    _mine(corpus, config_path, first)
    assert (first / "observations.jsonl").read_bytes() == before
    pending = first / "judge" / "pending.jsonl"
    assert not pending.exists() or not [
        x for x in pending.read_text().splitlines() if json.loads(x)["kind"] == ss.SUBSPLIT_KIND
    ]
