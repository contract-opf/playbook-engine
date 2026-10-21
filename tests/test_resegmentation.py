"""Re-segmentation: split mixed clauses, rejoin truncated ones, mark non-clauses (issue #255).

Unit tests drive :mod:`playbook_engine.resegmentation` on hand-built clause
trees with profiles learned by ``build_profiles`` from synthetic exemplars (the
same producer ``mine_corpus`` uses); the integration tests at the bottom drive
the real producers end to end: ``mine_corpus`` over RTF fixtures with an
injected segmenter, the production ``StoreBackedResegmentCheckJudge`` and its
``PendingQueue``, and ``build_scorecard``.

SECURITY NOTE: synthetic, fictional text only, written for these tests.
"""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest
import yaml
from docx import Document
from docx.oxml.ns import qn
from lxml import etree

from playbook_engine import resegmentation as rs
from playbook_engine.agent_judge import (
    PendingQueue,
    StoreBackedResegmentCheckJudge,
    VerdictStore,
    infer_verdict_kind,
    validate_verdict,
)
from playbook_engine.agent_judge import _payload_key as judge_payload_key
from playbook_engine.clause_classifier import (
    ClassifiedClause,
    ClauseClassification,
    reclassify_unclassified,
)
from playbook_engine.clause_tree import ClauseNode, ClauseTree
from playbook_engine.config import load_config
from playbook_engine.docx_ingester import TrackedChange, TrackedChanges
from playbook_engine.observation_builder import read_observations_jsonl
from playbook_engine.pipeline import mine_corpus, queue_resegment_checks
from playbook_engine.playbook_assembler import _CORPUS_DOCUMENT_INTERNAL_KEYS
from playbook_engine.rubric import JUDGE_KINDS, RubricStamp, rubric_version
from playbook_engine.scorecard import ENUM_LABELS, build_scorecard
from playbook_engine.segmentation_grounding import Block, SegNode
from playbook_engine.taxonomy import load_taxonomy
from tests.test_pipeline_llm_seg import _TAXONOMY_PATH, _write_rtf

# ---------------------------------------------------------------------------
# Synthetic corpus vocabulary
# ---------------------------------------------------------------------------

_INSURANCE = [
    "Both sides must keep a liability policy in force, with an insurer rated by the trade "
    "bureau, for amounts no lower than those in Schedule B per incident, and must hand over "
    "proof of the policy naming the other side as co-insured.",
    "The hospital must keep a liability policy and a malpractice policy in force with a rated "
    "insurer for amounts per incident set in Schedule B, and must hand the college proof of "
    "each policy naming it as co-insured.",
    "The college must keep a liability policy and a student accident policy in force with a "
    "rated insurer for amounts per incident set in Schedule B, and must hand the hospital "
    "proof of each policy naming it as co-insured.",
    "Both sides must keep a liability policy and a malpractice policy in force with a rated "
    "insurer, in amounts per incident under Schedule B, and must hand over proof of each "
    "policy naming the other side as co-insured.",
]
_PARTIES = [
    "The hospital and the college sign here because they want to run a joint teaching "
    "programme in which the college sends its learners to the hospital's wards for "
    "supervised practice.",
    "Whereas the college runs a teaching programme and the hospital has wards open for "
    "supervised practice, the college and the hospital want to run that programme jointly "
    "for the college's learners.",
    "The college and the hospital make this pact to run a joint teaching programme of "
    "supervised ward practice for learners enrolled with the college, as the opening "
    "statements describe.",
    "Between the hospital and the college: the college runs a teaching programme, the "
    "hospital offers wards for supervised practice, and both want to run it jointly for the "
    "college's learners.",
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

#: An insurance passage inside a preamble: it states coverage in this corpus's
#: own words, with nothing about parties or recitals.
_EMBEDDED_INSURANCE = (
    "Both sides must keep a liability policy in force with a rated insurer for the amounts "
    "per incident that Schedule B lists. Both sides must hand over proof of the policy, "
    "naming the other as co-insured, when asked."
)

#: Just the first sentence of the passage above.
_FOREIGN_ONE = (
    "Both sides must keep a liability policy in force with a rated insurer for the amounts "
    "per incident that Schedule B lists."
)

_PARTIES_ID = "parties_and_recitals"


def _profiles() -> rs.TypeProfiles:
    exemplars: list[tuple[str, str, str]] = []
    for i in range(4):
        exemplars.append((f"deal-{i}", "insurance", _INSURANCE[i]))
        exemplars.append((f"deal-{i}", _PARTIES_ID, _PARTIES[i]))
        exemplars.append((f"deal-{i}", "governing_law", _LAW[i]))
    return rs.build_profiles(exemplars)


#: Secrecy exemplars, for the redline join's type evidence.
_SECRECY = [
    "Each side keeps the trade records it gets from the other side private, and shows "
    "them only to advisers bound by a like duty of secrecy.",
    "Trade records one side gets from the other stay private; the side that gets them "
    "shows them only to advisers who owe a like duty of secrecy.",
    "Neither side lets out the trade records of the other; both keep such records private "
    "and show them only to advisers under a like duty of secrecy.",
    "Every trade record passed between the sides stays private, and a side shows such "
    "records only to its advisers who take on a like duty of secrecy.",
]


def _join_profiles() -> rs.TypeProfiles:
    """:func:`_profiles` plus a ``confidentiality`` type learned from :data:`_SECRECY`."""
    exemplars: list[tuple[str, str, str]] = []
    for i in range(4):
        exemplars.append((f"deal-{i}", "insurance", _INSURANCE[i]))
        exemplars.append((f"deal-{i}", _PARTIES_ID, _PARTIES[i]))
        exemplars.append((f"deal-{i}", "governing_law", _LAW[i]))
        exemplars.append((f"deal-{i}", "confidentiality", _SECRECY[i]))
    return rs.build_profiles(exemplars)


def _cls(tid: str | None, basis: str | None = None) -> ClauseClassification:
    if tid is None:
        return ClauseClassification(taxonomy_id=None, confidence=0.0, basis="unclassified")
    return ClauseClassification(taxonomy_id=tid, confidence=0.45, basis=basis or "llm_segmenter")


def _node(path: str, text: str, start: int, *, heading: str | None = None) -> ClauseNode:
    return ClauseNode(
        clause_path=path, heading=heading, text=text, char_span=(start, start + len(text))
    )


def _tree(nodes: list[ClauseNode]) -> ClauseTree:
    return ClauseTree(document_id="d", version="v1", source_file="d.rtf", nodes=nodes)


def _canonical(nodes: list[ClauseNode]) -> str:
    """The canonical text the nodes' spans index into (blocks joined by a newline)."""
    out = ""
    for n in nodes:
        out = out + "\n" * (n.char_span[0] - len(out)) + n.text
    return out


def _run(
    nodes: list[ClauseNode],
    types: list[str | None],
    profiles: rs.TypeProfiles | None = None,
    tracked: TrackedChanges | None = None,
) -> rs.ResegmentResult:
    tree = _tree(nodes)
    classified = [ClassifiedClause(n, _cls(t)) for n, t in zip(nodes, types, strict=True)]
    return rs.resegment(
        tree,
        classified,
        profiles if profiles is not None else _profiles(),
        labels={"insurance": "Insurance"},
        tracked=tracked,
    )


def _two_adjacent(first: str, second: str) -> list[ClauseNode]:
    """Two sibling nodes with the canonical one-newline gap between their spans."""
    a = _node("1", first, 0)
    return [a, _node("2", second, a.char_span[1] + 1)]


# ---------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------


def test_units_split_paragraphs_then_sentences_and_list_items() -> None:
    text = (
        "7. Heading\n"
        "First sentence here. Second sentence follows (see Section 4.2 for more). "
        "Inc. and Co. stay whole.\n"
        "\n"
        "Duties: (a) first duty; (b) second duty; (c) third duty."
    )
    pieces = [text[s:e] for s, e in rs.split_units(text)]
    assert pieces == [
        "7. Heading",
        "First sentence here.",
        "Second sentence follows (see Section 4.2 for more).",
        "Inc. and Co. stay whole.",
        "Duties:",
        "(a) first duty;",
        "(b) second duty;",
        "(c) third duty.",
    ]


def test_units_never_cut_inside_a_number_or_abbreviation() -> None:
    text = "See 4.2.1 and e.g. the schedule. U.S. law applies to Section 7. Insurance follows."
    pieces = [text[s:e] for s, e in rs.split_units(text)]
    assert pieces == [
        "See 4.2.1 and e.g. the schedule.",
        "U.S. law applies to Section 7. Insurance follows.",
    ]


# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------


def test_profile_needs_corpus_evidence_for_a_target() -> None:
    profiles = _profiles()
    assert profiles.targets == frozenset({"insurance", _PARTIES_ID, "governing_law"})

    # Two deals are not enough to make a type a target, however many words.
    two_deals = rs.build_profiles(
        [(f"deal-{i}", "insurance", _INSURANCE[i]) for i in range(2)]
        + [(f"deal-{i}", "insurance", _INSURANCE[i + 2]) for i in range(2)]
    )
    assert "insurance" not in two_deals.targets

    # Our standard alone never creates a target.
    template_only = rs.build_profiles([], {"insurance": " ".join(_INSURANCE)})
    assert template_only.targets == frozenset()
    assert template_only.tokens["insurance"]


def test_profile_ignores_words_spread_over_types_and_is_order_independent() -> None:
    # Every exemplar of every type uses "agreement notice written" (spread
    # over types), so those are no mark of any one of them.
    exemplars = [
        (f"deal-{i}", tid, f"{text} The accord needs a signed letter sent to the opposite side.")
        for i in range(4)
        for tid, text in (
            ("insurance", _INSURANCE[i]),
            ("governing_law", _LAW[i]),
            (_PARTIES_ID, _PARTIES[i]),
        )
    ]
    forward = rs.build_profiles(exemplars)
    backward = rs.build_profiles(list(reversed(exemplars)))
    assert forward == backward
    assert forward.fingerprint() == backward.fingerprint()
    for words in forward.tokens.values():
        assert "written" not in words
        assert "notice" not in words


# ---------------------------------------------------------------------------
# Split pass
# ---------------------------------------------------------------------------


def test_preamble_with_embedded_insurance_paragraph_is_split() -> None:
    text = f"1. Parties\n{_PARTIES[0]}\n{_EMBEDDED_INSURANCE}"
    node = _node("1", text, 40, heading="Parties")
    full = _canonical([node])
    result = _run([node], [_PARTIES_ID])

    assert result.splits == 1 and result.joins == 0
    kept, cut = list(result.tree.all_nodes())
    assert kept.clause_path == "1" and cut.clause_path == "1.r1"
    assert cut.text == _EMBEDDED_INSURANCE
    assert "insurance" not in kept.text.lower()
    # Citations resolve inside the original span, to the very bytes.
    assert full[cut.char_span[0] : cut.char_span[1]] == cut.text
    assert full[kept.char_span[0] : kept.char_span[1]] == kept.text
    assert node.char_span[0] <= cut.char_span[0] and cut.char_span[1] <= node.char_span[1]
    assert kept.char_span[1] <= cut.char_span[0]
    # Provenance on the new node, classification on the matching entry.
    meta = cut.x_resegmented
    assert meta is not None
    assert meta["split_from"] == "1"
    assert meta["origin_span"] == list(node.char_span)
    assert meta["to"] == {"id": "insurance", "label": "Insurance"}
    assert meta["basis"] == rs.BASIS_BLOCK_PROFILE
    assert meta["scores"]["score"] >= rs.SPLIT_SCORE_THRESHOLD
    by_path = {cc.node.clause_path: cc.classification for cc in result.classified}
    assert by_path["1"].taxonomy_id == _PARTIES_ID and by_path["1"].basis == "llm_segmenter"
    assert by_path["1.r1"].taxonomy_id == "insurance"
    assert by_path["1.r1"].basis == "resegmented"
    assert by_path["1.r1"].confidence <= rs.CONFIDENCE_CAP < 0.70
    # The tree stays valid, in document order.
    result.tree.validate(full)
    assert [cc.node.clause_path for cc in result.classified] == ["1", "1.r1"]


def test_split_works_inside_a_single_block_at_sentence_boundaries() -> None:
    # One paragraph, no newline: the cut falls between sentences.
    text = f"{_PARTIES[1]} {_EMBEDDED_INSURANCE}"
    node = _node("1", text, 0)
    result = _run([node], [_PARTIES_ID])
    assert result.splits == 1
    kept, cut = list(result.tree.all_nodes())
    assert kept.text == _PARTIES[1]
    assert cut.text == _EMBEDDED_INSURANCE
    assert cut.char_span == (len(_PARTIES[1]) + 1, len(text))


def test_a_clause_after_the_foreign_passage_keeps_its_own_type() -> None:
    tail = _PARTIES[2]
    text = f"{_PARTIES[0]}\n{_EMBEDDED_INSURANCE}\n{tail}"
    node = _node("3", text, 10)
    result = _run([node], [_PARTIES_ID])
    paths = [cc.node.clause_path for cc in result.classified]
    assert paths == ["3", "3.r1", "3.r2"]
    types = [cc.classification.taxonomy_id for cc in result.classified]
    assert types == [_PARTIES_ID, "insurance", _PARTIES_ID]
    assert result.splits == 1  # only the foreign passage counts as a split
    full = _canonical([node])
    result.tree.validate(full)
    assert [full[n.char_span[0] : n.char_span[1]] for n in result.tree.all_nodes()] == [
        n.text for n in result.tree.all_nodes()
    ]
    # Its own type is restored by the very classification the node had.
    assert result.classified[2].classification == result.classified[0].classification


def test_a_weak_match_is_not_split() -> None:
    weak = (
        "The college may buy insurance for its own students if it chooses to do so at its own "
        "cost and in its own discretion."
    )
    node = _node("1", f"{_PARTIES[0]}\n{weak}", 0)
    result = _run([node], [_PARTIES_ID])
    assert result.splits == 0
    assert [n.clause_path for n in result.tree.all_nodes()] == ["1"]
    assert result.tree.to_dict() == _tree([node]).to_dict()


def test_a_unit_that_fits_its_own_type_is_not_moved() -> None:
    # An insurance clause (typed insurance) that also states a limit sentence:
    # nothing in it is foreign.
    node = _node("2", f"{_INSURANCE[0]} {_INSURANCE[1]}", 0)
    result = _run([node], ["insurance"])
    assert result.splits == 0


def test_the_first_unit_of_a_node_is_never_moved() -> None:
    # A node that opens with the foreign passage keeps it: its first unit
    # defines what the segmenter meant the node to be.
    node = _node("1", f"{_FOREIGN_ONE}\n{_PARTIES[0]}", 0)
    result = _run([node], [_PARTIES_ID])
    assert result.splits == 0


def test_heading_line_stays_with_its_node() -> None:
    body = f"1. Parties\n{_PARTIES[0]}\n{_EMBEDDED_INSURANCE}"
    node = _node("1", body, 0, heading="Parties")
    node.heading_span = (0, 10)
    result = _run([node], [_PARTIES_ID])
    assert result.splits == 1
    kept = next(iter(result.tree.all_nodes()))
    assert kept.text == f"1. Parties\n{_PARTIES[0]}"
    assert kept.heading_span == (0, 10) and kept.heading == "Parties"
    # A heading line plus the unit after it anchor the node: a node whose only
    # body is a foreign passage keeps it (the segmenter's call stands).
    bare = _node("1", f"1. Parties\n{_FOREIGN_ONE}", 0, heading="Parties")
    assert _run([bare], [_PARTIES_ID]).splits == 0


def test_unclassified_and_unprofiled_nodes_are_left_alone() -> None:
    text = f"{_PARTIES[0]}\n{_EMBEDDED_INSURANCE}"
    assert _run([_node("1", text, 0)], [None]).splits == 0
    assert _run([_node("1", text, 0)], [_PARTIES_ID], rs.EMPTY_PROFILES).splits == 0


def test_inputs_are_not_mutated_and_two_runs_are_byte_identical() -> None:
    text = f"{_PARTIES[0]}\n{_EMBEDDED_INSURANCE}"
    nodes_a = [
        _node("1", text, 0),
        *_two_adjacent("The student shall wear", "a badge on site.")[:0],
    ]
    snapshot = _tree(nodes_a).to_json()
    first = _run(nodes_a, [_PARTIES_ID])
    assert _tree(nodes_a).to_json() == snapshot  # input tree untouched
    second = _run(nodes_a, [_PARTIES_ID])
    assert first.tree.to_json() == second.tree.to_json()
    assert [c.classification for c in first.classified] == [
        c.classification for c in second.classified
    ]
    # And independent of how the profiles were assembled.
    reversed_profiles = rs.build_profiles(
        list(
            reversed(
                [
                    (f"deal-{i}", tid, text)
                    for i in range(4)
                    for tid, text in (
                        ("insurance", _INSURANCE[i]),
                        (_PARTIES_ID, _PARTIES[i]),
                        ("governing_law", _LAW[i]),
                    )
                ]
            )
        )
    )
    third = _run(nodes_a, [_PARTIES_ID], reversed_profiles)
    assert first.tree.to_json() == third.tree.to_json()


# ---------------------------------------------------------------------------
# Join pass
# ---------------------------------------------------------------------------


def test_a_mid_sentence_break_is_joined() -> None:
    first = "The student shall wear the identification badge issued by the college and shall"
    second = "present it to the site supervisor on arrival each day."
    nodes = _two_adjacent(first, second)
    result = _run(nodes, [_PARTIES_ID, _PARTIES_ID])
    assert result.joins == 1
    (joined,) = list(result.tree.all_nodes())
    assert joined.clause_path == "1"
    assert joined.text == f"{first}\n{second}"
    assert joined.char_span == (0, nodes[1].char_span[1])
    full = _canonical(nodes)
    assert full[joined.char_span[0] : joined.char_span[1]] == joined.text
    meta = joined.x_resegmented
    assert meta is not None
    assert meta["joined_with"] == ["2"]
    assert meta["basis"] == [rs.BASIS_LOWERCASE_START]
    assert meta["origin_spans"] == [list(nodes[0].char_span), list(nodes[1].char_span)]
    assert meta["at"] == [len(first) + 1]
    assert len(result.classified) == 1
    assert result.classified[0].classification.taxonomy_id == _PARTIES_ID


def test_two_complete_adjacent_clauses_are_not_joined() -> None:
    nodes = _two_adjacent(
        "The student shall wear the identification badge issued by the college.",
        "the college may revise its badge design on thirty days notice.",  # lowercase, but complete first
    )
    assert _run(nodes, [_PARTIES_ID, _PARTIES_ID]).joins == 0
    nodes = _two_adjacent(
        "Either side may end this arrangement through a signed letter handed to its counterpart",
        "Every signatory keeps the others informed of any change of address.",  # capitalised start
    )
    assert _run(nodes, [_PARTIES_ID, _PARTIES_ID]).joins == 0
    nodes = _two_adjacent(
        "The first sentence is whole and stands alone as a complete statement.",
        "The second sentence is likewise whole and stands alone as well.",
    )
    assert _run(nodes, [_PARTIES_ID, _PARTIES_ID]).joins == 0


def test_a_list_cut_between_items_is_joined_by_numbering() -> None:
    first = "The coordinator shall report (a) any absence of the student; (b) any incident on site;"
    second = "(c) any complaint received from a client of the site."
    result = _run(_two_adjacent(first, second), [_PARTIES_ID, _PARTIES_ID])
    assert result.joins == 1
    meta = next(iter(result.tree.all_nodes())).x_resegmented
    assert meta is not None and meta["basis"] == [rs.BASIS_NUMBERING]
    # A marker that does not follow the last one is not a continuation.
    result = _run(_two_adjacent(first, "(f) any complaint received from a client."), ["a", "a"])
    assert result.joins == 0


def test_join_requires_a_shared_type() -> None:
    first = "The student shall wear the identification badge issued by the college and shall"
    second = "present it to the site supervisor on arrival each day."
    assert _run(_two_adjacent(first, second), [_PARTIES_ID, "insurance"]).joins == 0
    # The next node unclassified: it takes the first's type.
    result = _run(_two_adjacent(first, second), [_PARTIES_ID, None])
    assert result.joins == 1
    assert result.classified[0].classification.taxonomy_id == _PARTIES_ID
    # The first unclassified and the next classified: no.
    assert _run(_two_adjacent(first, second), [None, _PARTIES_ID]).joins == 0
    # Both unclassified (a fragment pair) rejoin.
    assert _run(_two_adjacent(first, second), [None, None]).joins == 1


def test_join_needs_adjacent_spans_and_no_heading_of_its_own() -> None:
    first = "The student shall wear the identification badge issued by the college and shall"
    second = "present it to the site supervisor on arrival each day."
    far = [_node("1", first, 0), _node("2", second, len(first) + 5)]
    assert _run(far, [_PARTIES_ID] * 2).joins == 0
    # The segmenter's label (an agent segmenter gives every node one, and it is
    # not in the text) is its name for the node, not a clause start.
    labelled = _two_adjacent(first, second)
    for n in labelled:
        n.heading = "Student Duties"
    result = _run(labelled, [_PARTIES_ID] * 2)
    assert result.joins == 1
    joined = next(iter(result.tree.all_nodes()))
    assert joined.x_resegmented is not None
    assert joined.x_resegmented["basis"] == [rs.BASIS_LOWERCASE_START]
    # A heading the extractor found is a clause start.
    found = _two_adjacent(first, second)
    found[1].heading_span = (found[1].char_span[0], found[1].char_span[0] + 7)
    assert _run(found, [_PARTIES_ID] * 2).joins == 0
    # So is a title line opening the next node, here after a list marker that
    # would otherwise continue the numbering.
    items = "The coordinator shall report (a) any absence of the student; (b) any incident on site;"
    titled = _two_adjacent(items, "(c) Site Visit Records\nThe coordinator logs every site visit.")
    for n in titled:
        n.heading = "Reports"
    assert _run(titled, [_PARTIES_ID] * 2).joins == 0
    untitled = _two_adjacent(items, "(c) any complaint received from a client of the site.")
    for n in untitled:
        n.heading = "Reports"
    assert _run(untitled, [_PARTIES_ID] * 2).joins == 1


def _insertion(start: int, end: int, author: str = "A. Author") -> TrackedChange:
    return TrackedChange("insertion", author, None, "x", "1", (start, end))


def _tracked(*changes: TrackedChange) -> TrackedChanges:
    return TrackedChanges(document_id="d", version="v1", changes=list(changes))


def _paragraph_local(nodes: list[ClauseNode], author: str = "A. Author") -> TrackedChanges:
    """One insertion per node, exactly the node's span: what the extractor records
    for a ``w:ins`` in each of several paragraphs (one span per paragraph)."""
    return _tracked(*(_insertion(*n.char_span, author) for n in nodes))


_PIECE_ONE = "Each side shall keep the other side's trade records private for the whole term."
_PIECE_TWO = "A side may show those records to its own advisers who accept the same duty."


def _labelled_pair() -> list[ClauseNode]:
    """Two pieces of one passage, each carrying the segmenter's label (an agent
    segmenter gives every node one, and the label is not in the text)."""
    nodes = _two_adjacent(_PIECE_ONE, _PIECE_TWO)
    for n in nodes:
        n.heading = "Confidentiality"
    return nodes


def test_paragraph_local_insertions_meeting_at_the_cut_are_stitched() -> None:
    # Both pieces end in sentence punctuation and carry a segmenter label: only
    # the tracked insertion that runs across the cut says they are one passage.
    nodes = _labelled_pair()
    types = ["confidentiality", "confidentiality"]
    result = _run(nodes, types, tracked=_paragraph_local(nodes))
    assert result.joins == 1
    (joined,) = list(result.tree.all_nodes())
    assert joined.text == f"{_PIECE_ONE}\n{_PIECE_TWO}"
    assert joined.x_resegmented is not None
    assert joined.x_resegmented["basis"] == [rs.BASIS_REDLINE]
    assert joined.x_resegmented["scores"] == {}
    assert result.classified[0].classification.taxonomy_id == "confidentiality"
    # An unclassified next piece joins when its own words are the node's type's
    # (more of them than of any other type's).
    profiles = _join_profiles()
    result = _run(
        _labelled_pair(), ["confidentiality", None], profiles, tracked=_paragraph_local(nodes)
    )
    assert result.joins == 1
    assert result.classified[0].classification.taxonomy_id == "confidentiality"
    # Without those words (no profile for the type) the insertion alone is no evidence.
    assert (
        _run(_labelled_pair(), ["confidentiality", None], tracked=_paragraph_local(nodes)).joins
        == 0
    )
    # Pieces neither of which has a type give no type to test: not stitched.
    assert (
        _run(_labelled_pair(), [None, None], profiles, tracked=_paragraph_local(nodes)).joins == 0
    )


def test_redline_stitching_needs_one_author_and_insertions_that_meet_at_the_cut() -> None:
    types = ["confidentiality", "confidentiality"]
    nodes = _labelled_pair()
    # No tracked change: two complete, labelled nodes stay two.
    assert _run(nodes, types).joins == 0
    # Two authors: two insertions, not one passage.
    first, second = nodes
    other = _tracked(_insertion(*first.char_span), _insertion(*second.char_span, "B. Other"))
    assert _run(nodes, types, tracked=other).joins == 0
    # An author the document does not name never links insertions.
    unnamed = _tracked(_insertion(*first.char_span, ""), _insertion(*second.char_span, ""))
    assert _run(nodes, types, tracked=unnamed).joins == 0
    # An insertion that stops short of the cut, or starts after the edge, does not meet.
    short = _tracked(
        _insertion(first.char_span[0], first.char_span[1] - 10), _insertion(*second.char_span)
    )
    assert _run(nodes, types, tracked=short).joins == 0
    late = _tracked(
        _insertion(*first.char_span), _insertion(second.char_span[0] + 10, second.char_span[1])
    )
    assert _run(nodes, types, tracked=late).joins == 0
    # Deletions and span-less insertions are no evidence.
    gone = _tracked(
        TrackedChange("deletion", "A. Author", None, "x", "1", None),
        TrackedChange("insertion", "A. Author", None, "x", "1", None),
    )
    assert _run(nodes, types, tracked=gone).joins == 0
    # Pieces of different types are different clauses whatever the redline says.
    assert _run(nodes, ["confidentiality", "insurance"], tracked=_paragraph_local(nodes)).joins == 0


def test_a_single_insertion_straddling_the_cut_is_stitched() -> None:
    nodes = _two_adjacent(_PIECE_ONE, _PIECE_TWO)
    cut = nodes[0].char_span[1]
    straddle = _tracked(_insertion(cut - 20, cut + 20))
    assert _run(nodes, ["confidentiality"] * 2, tracked=straddle).joins == 1


#: Text the not-a-clause filter marks, a clause of another split target, a
#: clause with no profiled type's words, and a lead-in: what an author's
#: insertion can run into after the clause it started in.
_SIGNATURES = "By: ______________\nName: ______________\nTitle: ______________"
_OTHER_LAW = (
    "A dispute between the sides is decided under the statutes of the province where the "
    "hospital sits, and its tribunals alone may hear it."
)
_OTHER_NOTICE = "Each notice under this arrangement goes by post to the address on the cover page."
_LEAD_IN = "The receiving side also takes on the terms below:"


def _agent_nodes(texts: list[str], labels: list[str]) -> list[ClauseNode]:
    """Sibling nodes of one block each, with the canonical one-newline gap between
    their spans and the segmenter's label on every node (never in the text): the
    agent producer's shape."""
    nodes = []
    start = 0
    for i, (text, label) in enumerate(zip(texts, labels, strict=True), start=1):
        nodes.append(_node(str(i), text, start, heading=label))
        start += len(text) + 1
    return nodes


def test_an_insertion_running_into_a_signature_block_is_not_stitched() -> None:
    # One author's insertion runs from the clause on into the signature lines;
    # the extractor records one span per paragraph, meeting at every cut.
    nodes = _agent_nodes(
        [_PIECE_ONE, _PIECE_TWO, _SIGNATURES], ["Confidentiality", "Confidentiality", "Execution"]
    )
    assert rs.non_clause_reason(nodes[2]) == "signature_block"
    profiles = _join_profiles()
    # Whether the segmenter left the signature lines untyped or typed them with
    # the clause, the clause's pieces are stitched and the signature lines are not.
    for types in (["confidentiality", "confidentiality", None], ["confidentiality"] * 3):
        result = _run(nodes, types, profiles, tracked=_paragraph_local(nodes))
        assert result.joins == 1
        out = list(result.tree.all_nodes())
        assert [n.text for n in out] == [f"{_PIECE_ONE}\n{_PIECE_TWO}", _SIGNATURES]
        assert out[1].x_resegmented is None
    # Left untyped, the signature lines are then marked, outside precedent.
    result = _run(
        nodes, ["confidentiality", "confidentiality", None], profiles, _paragraph_local(nodes)
    )
    marked = rs.mark_non_clauses(result.classified)
    assert [cc.node.x_non_clause for cc in marked] == [None, "signature_block"]
    # A clause straight into the signature lines: nothing is stitched.
    pair = _agent_nodes([_PIECE_ONE, _SIGNATURES], ["Confidentiality", "Execution"])
    assert _run(pair, ["confidentiality", None], profiles, _paragraph_local(pair)).joins == 0


def test_an_insertion_running_into_a_clause_of_another_type_is_not_stitched() -> None:
    profiles = _join_profiles()
    # The split pass's own scorer reads the next piece as another split target.
    assert rs._score_unit(rs._unit_tokens(_OTHER_LAW), "confidentiality", profiles) is not None
    law = _agent_nodes([_PIECE_ONE, _OTHER_LAW], ["Confidentiality", "Disputes"])
    assert _run(law, ["confidentiality", None], profiles, _paragraph_local(law)).joins == 0
    # A clause with none of the type's words: the insertion alone is no evidence
    # that it continues the clause (a counterparty inserts whole clauses in one go).
    assert rs._score_unit(rs._unit_tokens(_OTHER_NOTICE), "confidentiality", profiles) is None
    notice = _agent_nodes([_PIECE_ONE, _OTHER_NOTICE], ["Confidentiality", "Notices"])
    assert _run(notice, ["confidentiality", None], profiles, _paragraph_local(notice)).joins == 0
    # Neither piece typed: no type to test, so nothing is stitched.
    assert _run(notice, [None, None], profiles, _paragraph_local(notice)).joins == 0
    # In a longer run the stitching stops at the first piece of another clause.
    run = _agent_nodes(
        [_PIECE_ONE, _PIECE_TWO, _OTHER_NOTICE, _OTHER_LAW],
        ["Confidentiality", "Confidentiality", "Notices", "Disputes"],
    )
    result = _run(run, ["confidentiality", None, None, None], profiles, _paragraph_local(run))
    assert result.joins == 1
    assert [n.text for n in result.tree.all_nodes()] == [
        f"{_PIECE_ONE}\n{_PIECE_TWO}",
        _OTHER_NOTICE,
        _OTHER_LAW,
    ]


def test_a_lead_in_inside_an_insertion_joins_only_with_what_it_introduces() -> None:
    profiles = _join_profiles()
    nodes = _agent_nodes([_PIECE_ONE, _LEAD_IN, _PIECE_TWO], ["Confidentiality"] * 3)
    assert rs.non_clause_reason(nodes[1]) == "lead_in"
    result = _run(
        nodes, ["confidentiality", None, "confidentiality"], profiles, _paragraph_local(nodes)
    )
    assert result.joins == 2
    (joined,) = list(result.tree.all_nodes())
    assert joined.text == "\n".join([_PIECE_ONE, _LEAD_IN, _PIECE_TWO])
    # Followed by a clause of another type, the lead-in is not taken in.
    other = _agent_nodes([_PIECE_ONE, _LEAD_IN, _OTHER_NOTICE], ["Confidentiality"] * 3)
    assert (
        _run(other, ["confidentiality", None, None], profiles, _paragraph_local(other)).joins == 0
    )
    # Nor when the insertion ends with it.
    last = _agent_nodes([_PIECE_ONE, _LEAD_IN], ["Confidentiality"] * 2)
    assert _run(last, ["confidentiality", None], profiles, _paragraph_local(last)).joins == 0


def test_a_textual_join_never_takes_in_text_that_is_not_a_clause_or_is_another_type() -> None:
    first = "The student shall wear the identification badge issued by the college and shall"
    for junk, reason in (
        ("page 3 of 12", "header_footer"),
        ("l I i | ~ ` ' , . ; : l i", "ocr_junk"),
    ):
        nodes = _two_adjacent(first, junk)
        assert rs.non_clause_reason(nodes[1]) == reason
        assert _run(nodes, [_PARTIES_ID, None]).joins == 0
    # Nor extends such text: an email header cut off mid-line is no clause start.
    mail = "From: dean@example.test\nTo: office@example.test\nSubject: placement of"
    nodes = _two_adjacent(mail, "the students for the spring term.")
    assert rs.non_clause_reason(nodes[0]) == "email"
    assert _run(nodes, [None, None]).joins == 0
    # An unclassified continuation the scorer reads as another type is that clause.
    nodes = _two_adjacent(
        "The college and the hospital run the programme together, and both sides",
        "must keep a liability policy in force with a rated insurer for the amounts per "
        "incident in Schedule B.",
    )
    assert _run(nodes, [_PARTIES_ID, None]).joins == 0


def test_a_piece_that_opens_with_its_own_heading_is_not_stitched() -> None:
    nodes = _two_adjacent(
        _PIECE_ONE, "Notices. Each notice under this arrangement goes in writing."
    )
    nodes[1].heading = "Notices"  # the label is also the text's opening words
    types = ["confidentiality", "confidentiality"]
    assert _run(nodes, types, tracked=_paragraph_local(nodes)).joins == 0
    titled = _two_adjacent(_PIECE_ONE, "Entire Agreement\nThis writing is the whole deal.")
    titled[1].heading = "Miscellaneous"
    assert _run(titled, types, tracked=_paragraph_local(titled)).joins == 0
    # A heading the extractor found is a clause start too.
    found = _two_adjacent(_PIECE_ONE, _PIECE_TWO)
    found[1].heading_span = (found[1].char_span[0], found[1].char_span[0] + 5)
    assert _run(found, types, tracked=_paragraph_local(found)).joins == 0


def test_a_redline_run_longer_than_the_textual_chain_cap_is_one_passage() -> None:
    pieces = [
        f"Clause piece number {i} keeps the supplier records private for the term."
        for i in range(10)
    ]
    nodes = []
    start = 0
    for i, piece in enumerate(pieces, start=1):
        node = _node(str(i), piece, start)
        node.heading = "Confidentiality"
        nodes.append(node)
        start += len(piece) + 1
    assert len(nodes) > rs.JOIN_MAX_CHAIN
    types = ["confidentiality", *([None] * 9)]
    result = _run(nodes, types, _join_profiles(), tracked=_paragraph_local(nodes))
    assert result.joins == len(nodes) - 1
    (joined,) = list(result.tree.all_nodes())
    assert joined.text == "\n".join(pieces)
    meta = joined.x_resegmented
    assert meta is not None and meta["basis"] == [rs.BASIS_REDLINE] * (len(nodes) - 1)
    # The size cap still holds.
    big = [
        _node("1", "word " * 900 + "end.", 0),
        _node("2", "word " * 900 + "end.", 0),
    ]
    big[1] = _node("2", big[1].text, big[0].char_span[1] + 1)
    assert len(big[0].text) * 2 > rs.JOIN_MAX_CHARS
    assert _run(big, ["confidentiality"] * 2, tracked=_paragraph_local(big)).joins == 0


def test_a_chain_of_fragments_rejoins_but_is_bounded() -> None:
    parts = ["The tenant shall keep the premises clean and shall"] + [
        f"keep part {i} in good order and shall" for i in range(6)
    ]
    nodes = []
    start = 0
    for i, p in enumerate(parts, start=1):
        nodes.append(_node(str(i), p, start))
        start += len(p) + 1
    result = _run(nodes, [None] * len(nodes))
    # At most JOIN_MAX_CHAIN nodes merge into one; the rest follow on.
    first = next(iter(result.tree.all_nodes()))
    assert first.text.count("\n") == rs.JOIN_MAX_CHAIN - 1
    assert result.joins >= 2


def test_a_split_off_node_is_not_merged_onward() -> None:
    text = f"{_PARTIES[0]}\n{_EMBEDDED_INSURANCE}"
    node = _node("1", text, 0)
    follower = _node("2", "and the rest of the sentence follows here.", node.char_span[1] + 1)
    result = _run([node, follower], [_PARTIES_ID, _PARTIES_ID])
    assert result.splits == 1
    assert result.joins == 0


# ---------------------------------------------------------------------------
# Not-a-clause filter
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("3.2", "numbering_only"),
        ("(b)", "numbering_only"),
        ("iv.", "numbering_only"),
        ("By: ______________\nName: ______________\nTitle: ______________", "signature_block"),
        ("Envelope ID: 7F3A-0001 certificate of completion", "signature_block"),
        ("[Intentionally Omitted]", "heading_or_placeholder"),
        ("ARTICLE FOUR", "heading_or_placeholder"),
        ("Page 3 of 12", "header_footer"),
        ("Confidential and Proprietary", "header_footer"),
        ("From: someone@example.test\nTo: another@example.test\nSubject: Re: draft", "email"),
        ("The signatories accept the duties set out next:", "lead_in"),
        ("and the other terms agreed", "fragment"),
        ("l I i | ~ ` ' , . ; : l i", "ocr_junk"),
        (
            "Student\nSignature \\\\\\\\\\\\\nName \\\\\\\\\\\\\\nAddress \\\\\\\\\\\\\\\\",
            "signature_block",
        ),
        ("Title: Associate Dean Date: 12 May 2030", "signature_block"),
        ("4.7. Retention of Records. [Intentionally Omitted]", "heading_or_placeholder"),
        ("[End of Agreement]", "heading_or_placeholder"),
        ("Affiliation Agreement for Student Placements", "heading_or_placeholder"),
        ("2. Term and Termination", "heading_or_placeholder"),
        ("AGREEMENT TO KEEP RECORDS.", "heading_or_placeholder"),
        ("The signatories accept what follows:", "lead_in"),
        # A weak signing phrase with nothing operative around it is a signature block.
        ("Document electronically signed on 3 June 2030 by the dean", "signature_block"),
        ("Signature page follows", "signature_block"),
    ],
)
def test_non_clause_reasons(text: str, reason: str) -> None:
    node = _node("1", text, 0)
    assert rs.non_clause_reason(node) == reason
    assert reason in rs.NON_CLAUSE_REASONS


@pytest.mark.parametrize(
    "text",
    [
        _PARTIES[0],
        "Either side may bring this arrangement to a close for any reason by sending the other "
        "a signed letter sixty days ahead.",
        "No student shall be treated as an employee of the hospital for any purpose.",
        "The coordinator will keep a record of each session attended by the student.",
        # Clauses about signing: operative sentences, not signature blocks.
        "This arrangement may be executed in several counterparts, and a counterpart sent by "
        "electronic signature is as good as one signed in ink.",
        "The signatories agree that this arrangement may be electronically signed and that a "
        "copy carrying a digital signature is valid for every purpose.",
        "Signature pages sent by email shall be binding on the side that sent them.",
    ],
)
def test_real_clauses_are_not_marked(text: str) -> None:
    assert rs.non_clause_reason(_node("1", text, 0)) is None


def test_only_unclassified_nodes_are_marked_and_never_removed() -> None:
    nodes = [_node("1", "3.2", 0), _node("2", "4.", 4), _node("3", _PARTIES[0], 8)]
    classified = [
        ClassifiedClause(nodes[0], _cls(None)),
        ClassifiedClause(nodes[1], _cls("term")),  # classified: left alone
        ClassifiedClause(nodes[2], _cls(None)),
    ]
    out = rs.mark_non_clauses(classified)
    assert [cc.node.x_non_clause for cc in out] == ["numbering_only", None, None]
    assert [cc.node.clause_path for cc in out] == ["1", "2", "3"]
    # A pending or failed judge verdict is not an answer: not marked.
    pending = ClauseClassification(taxonomy_id=None, confidence=0.0, basis="needs_review")
    waiting = ClassifiedClause(_node("1", "3.2", 0), pending)
    assert rs.mark_non_clauses([waiting])[0].node.x_non_clause is None


def test_a_non_clause_is_never_asked_of_the_judge() -> None:
    from playbook_engine.taxonomy import load_taxonomy as _load

    junk = _node("1", "Page 3 of 12", 0)
    junk.x_non_clause = "header_footer"
    real = _node("2", "The coordinator will keep a record of each session attended.", 20)
    asked: list[str] = []

    class _Judge:
        def classify_batch(self, nodes, taxonomy, hints=None):  # type: ignore[no-untyped-def]
            asked.extend(n.clause_path for n in nodes)
            return [_cls(None) for _ in nodes]

    reclassify_unclassified(
        [ClassifiedClause(junk, _cls(None)), ClassifiedClause(real, _cls(None))],
        _load(_TAXONOMY_PATH),
        _Judge(),  # type: ignore[arg-type]
    )
    assert asked == ["2"]


def test_the_deterministic_classifier_does_not_ask_the_judge_about_a_non_clause() -> None:
    from playbook_engine.clause_classifier import classify_tree

    junk = _node("1", "Page 3 of 12", 0)
    real = _node("2", "The coordinator will keep a record of each session attended.", 20)
    asked: list[str] = []

    class _Judge:
        def classify_batch(self, nodes, taxonomy, hints=None):  # type: ignore[no-untyped-def]
            asked.extend(n.clause_path for n in nodes)
            return [ClauseClassification(None, 0.0, "needs_review") for _ in nodes]

    taxonomy = load_taxonomy(_TAXONOMY_PATH)
    classify_tree(_tree([junk, real]), taxonomy, _Judge())  # type: ignore[arg-type]
    assert asked == ["1", "2"]  # without the predicate the page marker is asked
    asked.clear()
    out = classify_tree(
        _tree([junk, real]),
        taxonomy,
        _Judge(),  # type: ignore[arg-type]
        skip_judge=lambda node: rs.non_clause_reason(node) is not None,
    )
    assert asked == ["2"]
    # Left unclassified (not pending), so the filter marks it and a rejoin can still use it.
    assert out[0].classification.basis == "unclassified"
    assert rs.mark_non_clauses(out)[0].node.x_non_clause == "header_footer"


def test_mine_on_the_deterministic_path_queues_no_judge_item_for_a_non_clause(
    tmp_path: Path,
) -> None:
    from playbook_engine.agent_judge import StoreBackedClassificationJudge

    corpus, config_path = _corpus(tmp_path)
    for i in range(4):  # a running page marker ahead of the first heading
        _write_rtf(
            corpus / f"deal-00{i + 1}" / "v1.rtf",
            r"Page 1 of 3\par " + _deal_body(i, mixed=False),
        )
    out = tmp_path / "out"
    judge = StoreBackedClassificationJudge(
        store=VerdictStore(out / "judge" / "verdicts.jsonl"),
        pending=PendingQueue(out / "judge" / "pending.jsonl"),
    )
    mine_corpus(
        corpus_dir=corpus,
        config=load_config(config_path),
        taxonomy=load_taxonomy(_TAXONOMY_PATH),
        out_dir=out,
        use_llm_segmentation=False,
        classification_judge=judge,
    )
    pending = out / "judge" / "pending.jsonl"
    assert not pending.exists() or "Page 1 of 3" not in pending.read_text()
    rows = read_observations_jsonl(out / "observations.jsonl")
    marked = [r for r in rows if r.get("x_non_clause") == "header_footer"]
    assert len(marked) == 4 and all(r["taxonomy_id"] is None for r in marked)


def test_clause_tree_round_trips_the_resegmentation_fields() -> None:
    node = _node("1.r1", "text", 3)
    node.x_resegmented = {"split_from": "1", "basis": "block_profile", "scores": {}}
    node.x_non_clause = "fragment"
    again = ClauseNode.from_dict(json.loads(json.dumps(node.to_dict())))
    assert again.x_resegmented == node.x_resegmented and again.x_non_clause == "fragment"
    assert "x_resegmented" not in _node("1", "t", 0).to_dict()
    bad = {**node.to_dict(), "x_resegmented": "no"}
    with pytest.raises(ValueError):
        ClauseNode.from_dict(bad)


def test_a_lead_in_with_a_modal_verb_is_a_clause_not_a_lead_in() -> None:
    # "may be ended as follows:" says what may happen: left for the classifier.
    text = "Notwithstanding the term above, this arrangement may be ended as follows:"
    assert rs.non_clause_reason(_node("1", text, 0)) is None


def test_looks_like_heading() -> None:
    for heading in (
        "Term and Termination",
        "2. TERM AND TERMINATION",
        "[End of Agreement]",
        "AGREEMENT TO KEEP RECORDS.",
    ):
        assert rs.looks_like_heading(heading), heading
    for sentence in (
        "The student shall keep a record of each session.",
        "Records: kept by the college",
        "Each Party Shall Keep, Always",
        "this is lower case at the start",
    ):
        assert not rs.looks_like_heading(sentence), sentence


# ---------------------------------------------------------------------------
# What a split must not do
# ---------------------------------------------------------------------------


def test_a_node_whose_only_body_would_move_is_left_to_the_segmenter() -> None:
    # Heading plus a body that reads as another type: the segmenter typed the
    # node knowing the heading and where it sits; this is not a mixture.
    node = _node("1", f"1. Parties\n{_FOREIGN_ONE}\nSee the schedule.", 0)
    assert _run([node], [_PARTIES_ID]).splits == 0


def test_later_list_items_and_back_references_stay_with_the_unit_before() -> None:
    insurance_item = (
        "(a) Both sides must keep a liability policy in force with a rated insurer for the "
        "amounts per incident that Schedule B lists;"
    )
    later_item = (
        "(b) both sides must hand over proof of each policy, naming the other as co-insured, "
        "whenever the rated insurer renews it;"
    )
    back_reference = (
        "Such proof of the policy must show the liability amounts per incident that the rated "
        "insurer has agreed to."
    )
    # A later list item never moves on its own; the first item may.
    lone = _node("1", f"{_PARTIES[0]}\n{later_item}", 0)
    assert _run([lone], [_PARTIES_ID]).splits == 0
    first = _node("1", f"{_PARTIES[0]}\n{insurance_item}\n{later_item}", 0)
    result = _run([first], [_PARTIES_ID])
    assert result.splits == 1
    moved = [n for n in result.tree.all_nodes() if n.clause_path == "1.r1"]
    assert moved and later_item in moved[0].text  # the list moved whole
    # A sentence that refers back follows the unit before it.
    lone_back = _node("1", f"{_PARTIES[0]}\n{back_reference}", 0)
    assert _run([lone_back], [_PARTIES_ID]).splits == 0


# ---------------------------------------------------------------------------
# End to end: mine_corpus over RTF with an injected segmenter
# ---------------------------------------------------------------------------

_DEAL_HEADINGS = {
    "1. Parties": _PARTIES_ID,
    "2. Insurance": "insurance",
    "3. Governing Law": "governing_law",
    # The segmenter found no type for these two.
    "4. Schedule": None,
    "5. Notes": None,
}

_NOTES = "The coordinator keeps a dated log of every visit and hands a copy to the dean each term."


def _deal_body(i: int, *, mixed: bool, extra: bool = False) -> str:
    parties = _PARTIES[i] + (" " + _EMBEDDED_INSURANCE if mixed else "")
    body = (
        rf"1. Parties\par {parties}\par "
        rf"2. Insurance\par {_INSURANCE[i]}\par "
        rf"3. Governing Law\par {_LAW[i]}\par "
    )
    if extra:
        body += rf"4. Schedule\par Page 2 of 3\par 5. Notes\par {_NOTES}\par "
    return body


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
                taxonomy_id=_DEAL_HEADINGS[head.text],
                start_block_id=head.block_id,
                end_block_id=body.block_id,
                start_quote=head.text[:10],
                end_quote=body.text[-10:],
            )
        )
    return nodes


def _corpus(tmp_path: Path, *, extra: bool = False) -> tuple[Path, Path]:
    corpus = tmp_path / "corpus"
    for i in range(4):
        deal = corpus / f"deal-00{i + 1}"
        deal.mkdir(parents=True, exist_ok=True)
        _write_rtf(deal / "v1.rtf", _deal_body(i, mixed=(i == 3), extra=(extra and i == 0)))
    cfg = {
        "agreement_type": {"id": "educational-affiliation", "name": "Educational Affiliation"},
        "baseline": {"template": None},
        "taxonomy": str(_TAXONOMY_PATH),
        "provenance": {"our_party_aliases": ["Alpha Corp"]},
    }
    config_path = tmp_path / "playbook.config.yaml"
    config_path.write_text(yaml.dump(cfg), encoding="utf-8")
    return corpus, config_path


def _mine(
    tmp_path: Path, out: Path, *, extra: bool = False, judge: object | None = None
) -> list[dict]:  # type: ignore[type-arg]
    corpus, config_path = _corpus(tmp_path, extra=extra)
    mine_corpus(
        corpus_dir=corpus,
        config=load_config(config_path),
        taxonomy=load_taxonomy(_TAXONOMY_PATH),
        out_dir=out,
        use_llm_segmentation=True,
        llm_segment_fn=_segment_fn,
        classification_judge=judge,  # type: ignore[arg-type]
    )
    return read_observations_jsonl(out / "observations.jsonl")


def test_mine_splits_the_mixed_deal_and_records_it(tmp_path: Path) -> None:
    out = tmp_path / "out"
    rows = _mine(tmp_path, out)

    mixed = [r for r in rows if r["citation"]["document_id"] == "deal-004"]
    by_tid = {r["taxonomy_id"]: r for r in mixed}
    # The insurance passage now has its own observation, typed by the pass.
    assert by_tid["insurance"]["x_classification_basis"] in ("resegmented", "llm_segmenter")
    marked = [r for r in mixed if "x_resegmented" in r]
    split_rows = [r for r in marked if "to" in r["x_resegmented"]]
    assert len(split_rows) == 1
    # The node the passage was cut out of records that it was cut, scores empty.
    (shrunk,) = [r for r in marked if "to" not in r["x_resegmented"]]
    assert shrunk["taxonomy_id"] == _PARTIES_ID and shrunk["x_resegmented"]["scores"] == {}
    (split_row,) = split_rows
    assert split_row["taxonomy_id"] == "insurance"
    assert split_row["x_classification_basis"] == "resegmented"
    assert split_row["x_resegmented"]["split_from"] == "1"
    assert split_row["citation"]["clause_path"] == "1.r1"
    # The preamble observation no longer carries the passage.
    assert "co-insured" not in by_tid[_PARTIES_ID]["full_text"]
    assert "co-insured" in split_row["full_text"]
    # The untouched deals have no split rows.
    assert not [
        r for r in rows if r["citation"]["document_id"] != "deal-004" and "x_resegmented" in r
    ]

    manifest = json.loads((out / "corpus_manifest.json").read_text())
    counts = {d["document_id"]: d["resegmentation"] for d in manifest}
    assert counts["deal-004"] == {"splits": 1, "joins": 0}
    assert counts["deal-001"] == {"splits": 0, "joins": 0}

    # The persisted tree carries the new node and its provenance, citing
    # a span inside the original block.
    tree = ClauseTree.load(out / "normalized" / "deal-004" / "v1.clauses.json")
    cut = tree.resolve_path("1.r1")
    assert cut is not None and cut.x_resegmented is not None
    origin = cut.x_resegmented["origin_span"]
    assert origin[0] <= cut.char_span[0] and cut.char_span[1] <= origin[1]
    # The persisted tree (new node, provenance and all) is a valid clause-tree artifact.
    schema = json.loads(
        (Path(__file__).parent.parent / "spec" / "clause-tree.schema.json").read_text()
    )
    jsonschema.validate(
        instance=json.loads((out / "normalized" / "deal-004" / "v1.clauses.json").read_text()),
        schema=schema,
    )
    # It stays out of the published playbook's corpus documents.
    assert "resegmentation" in _CORPUS_DOCUMENT_INTERNAL_KEYS


def test_mine_marks_a_non_clause_and_never_asks_the_judge_about_it(tmp_path: Path) -> None:
    from playbook_engine.agent_judge import StoreBackedClassificationJudge

    out = tmp_path / "out"
    judge = StoreBackedClassificationJudge(
        store=VerdictStore(out / "judge" / "verdicts.jsonl"),
        pending=PendingQueue(out / "judge" / "pending.jsonl"),
    )
    rows = _mine(tmp_path, out, extra=True, judge=judge)

    queued = [
        json.loads(line) for line in (out / "judge" / "pending.jsonl").read_text().splitlines()
    ]
    # The page marker is no clause: only the operative note reaches the judge.
    assert [q["payload"]["heading"] for q in queued] == ["5. Notes"]

    by_path = {
        r["citation"]["clause_path"]: r for r in rows if r["citation"]["document_id"] == "deal-001"
    }
    assert by_path["4"]["x_non_clause"] == "heading_or_placeholder"
    assert by_path["4"]["taxonomy_id"] is None
    assert "x_non_clause" not in by_path["5"] and by_path["5"]["taxonomy_id"] is None
    # The mark is on the persisted tree too, and the node is still there to cite.
    tree = ClauseTree.load(out / "normalized" / "deal-001" / "v1.clauses.json")
    node = tree.resolve_path("4")
    assert node is not None and node.x_non_clause == "heading_or_placeholder"
    # It never leaks into a deal that has no such node, and the scorecard
    # leaves it out of the classified share.
    assert not [
        r for r in rows if r["citation"]["document_id"] != "deal-001" and "x_non_clause" in r
    ]
    block = build_scorecard(out)["classification"]
    assert block["non_clause"] == 1 and block["non_clause_by_reason"] == {
        "heading_or_placeholder": 1
    }
    assert block["clauses"] == block["observations"] - 1


def test_mine_is_deterministic_and_cached_runs_agree(tmp_path: Path) -> None:
    first_dir, second_dir = tmp_path / "a" / "out", tmp_path / "b" / "out"
    _mine(tmp_path / "a", first_dir)
    _mine(tmp_path / "b", second_dir)
    for name in ("observations.jsonl", "corpus_manifest.json"):
        assert (first_dir / name).read_bytes() == (second_dir / name).read_bytes(), name
    trees_a = sorted((first_dir / "normalized").rglob("*.clauses.json"))
    trees_b = sorted((second_dir / "normalized").rglob("*.clauses.json"))
    assert [t.name for t in trees_a] == [t.name for t in trees_b]
    for a, b in zip(trees_a, trees_b, strict=True):
        assert a.read_bytes() == b.read_bytes()
    # A warm re-mine of the same out-dir (the stage cache) reproduces it too.
    before = (first_dir / "observations.jsonl").read_bytes()
    _mine(tmp_path / "a", first_dir)
    assert (first_dir / "observations.jsonl").read_bytes() == before


def test_a_corpus_too_small_to_learn_a_type_splits_nothing(tmp_path: Path) -> None:
    corpus, config_path = _corpus(tmp_path)
    for i in range(1, 3):  # keep only two deals, one of them the mixed one
        for f in (corpus / f"deal-00{i}").iterdir():
            f.unlink()
        (corpus / f"deal-00{i}").rmdir()
    out = tmp_path / "out"
    mine_corpus(
        corpus_dir=corpus,
        config=load_config(config_path),
        taxonomy=load_taxonomy(_TAXONOMY_PATH),
        out_dir=out,
        use_llm_segmentation=True,
        llm_segment_fn=_segment_fn,
    )
    rows = read_observations_jsonl(out / "observations.jsonl")
    assert not [r for r in rows if "x_resegmented" in r]


# ---------------------------------------------------------------------------
# End to end: a counterparty insertion typed over several paragraphs
# ---------------------------------------------------------------------------
#
# Producer shapes these tests copy (and the unit fixtures above cannot): the
# extractor records ONE insertion span per paragraph-local w:ins, so a
# multi-paragraph insertion is several spans that meet at the paragraph edges;
# an agent segmenter cuts only at block boundaries and gives every node a label
# heading that is not in the node's text; and normalized/ text is pseudonymized
# after the node offsets are recorded.

_ENTITY = "Northgate Hospital Trust"
_INSERTED = [
    f"{_ENTITY} shall keep the supplier ledgers of the other side private while the "
    "supervisor is on site.",
    "The college also takes on the terms below:",
    "The college may show those ledgers to its own advisers who accept the same duty.",
    "Neither side shall copy a ledger except where a regulator asks for the copy.",
    "These duties end five years after the arrangement ends.",
]
# The same author's insertion runs on past the clause: into a clause of another
# type the segmenter found no type for, then into the signature lines.
_INSERTED_AFTER = [
    "Each notice under this arrangement goes by post to the address on the cover page.",
    "By: ______________ Name: ______________ Title: ______________",
]
# The segmenter's label and type per block: the preamble, the five pieces of
# the clause (it typed each piece and left the lead-in untyped, the shape the
# acceptance corpus's split insertion has), the notice clause and the signature
# lines (untyped), the law.
_REDLINE_LABELS = [
    "Programme",
    *["Confidentiality"] * 5,
    "Notices",
    "Execution",
    "Governing Law",
]
_REDLINE_TYPES = [
    None,
    "confidentiality",
    None,
    *["confidentiality"] * 3,
    None,
    None,
    "governing_law",
]


def _write_redlined_docx(path: Path, paragraphs: list[tuple[str, str | None]]) -> None:
    """A .docx of ``(text, author)`` paragraphs; a named author's text is a tracked insertion."""
    doc = Document()
    for n, (text, author) in enumerate(paragraphs):
        para = doc.add_paragraph()
        if author is None:
            para.add_run(text)
            continue
        ins = etree.SubElement(para._p, qn("w:ins"))
        ins.set(qn("w:id"), str(n + 1))
        ins.set(qn("w:author"), author)
        ins.set(qn("w:date"), "2030-03-15T10:00:00Z")
        run = etree.SubElement(ins, qn("w:r"))
        etree.SubElement(run, qn("w:t")).text = text
    doc.save(str(path))


def _redline_segment_fn(canonical_text: str, blocks: list[Block]) -> list[SegNode]:
    """One node per block, each with the segmenter's label as its heading (never in the text)."""
    del canonical_text
    assert len(blocks) == len(_REDLINE_LABELS)  # one block per paragraph
    return [
        SegNode(
            node_id=f"n{n}",
            parent_id=None,
            order=n,
            heading=_REDLINE_LABELS[n - 1],
            taxonomy_id=_REDLINE_TYPES[n - 1],
            start_block_id=block.block_id,
            end_block_id=block.block_id,
            start_quote=block.text[:10],
            end_quote=block.text[-10:],
        )
        for n, block in enumerate(blocks, start=1)
    ]


def _mine_redlined(tmp_path: Path, *, known_entities: list[str]) -> Path:
    corpus = tmp_path / "corpus"
    deal = corpus / "deal-001"
    deal.mkdir(parents=True)
    _write_redlined_docx(
        deal / "v1.docx",
        [
            ("Both sides agree to run the shared programme described here.", None),
            *[(text, "Counsel") for text in [*_INSERTED, *_INSERTED_AFTER]],
            ("This arrangement follows the law of the province where the hospital sits.", None),
        ],
    )
    cfg = {
        "agreement_type": {"id": "educational-affiliation", "name": "Educational Affiliation"},
        "baseline": {"template": None},
        "taxonomy": str(_TAXONOMY_PATH),
        "provenance": {"our_party_aliases": ["Alpha Corp"], "known_entities": known_entities},
    }
    config_path = tmp_path / "playbook.config.yaml"
    config_path.write_text(yaml.dump(cfg), encoding="utf-8")
    out = tmp_path / "out"
    mine_corpus(
        corpus_dir=corpus,
        config=load_config(config_path),
        taxonomy=load_taxonomy(_TAXONOMY_PATH),
        out_dir=out,
        use_llm_segmentation=True,
        llm_segment_fn=_redline_segment_fn,
        entity_registry_path=tmp_path / "registry.json",
    )
    return out


def test_mine_stitches_a_multi_paragraph_insertion_the_segmenter_cut_at_every_block(
    tmp_path: Path,
) -> None:
    out = _mine_redlined(tmp_path, known_entities=[])
    tree = ClauseTree.load(out / "normalized" / "deal-001" / "v1.clauses.json")
    nodes = list(tree.all_nodes())
    # Programme preamble, the stitched clause, the notice clause, the signature
    # lines, governing law: the five pieces of the clause are one node whose
    # label is the segmenter's; the insertion's run into the next clause and
    # the signature lines is not stitched.
    assert [n.heading for n in nodes] == [
        "Programme",
        "Confidentiality",
        "Notices",
        "Execution",
        "Governing Law",
    ]
    joined = nodes[1]
    assert joined.heading_span is None  # the label is not in the text
    assert joined.text == "\n".join(_INSERTED)
    meta = joined.x_resegmented
    assert meta is not None
    assert meta["basis"] == [rs.BASIS_REDLINE] * 4 and meta["scores"] == {}
    assert len(meta["joined_with"]) == 4
    assert [n.text for n in nodes[2:4]] == _INSERTED_AFTER
    assert nodes[2].x_resegmented is None and nodes[3].x_resegmented is None
    assert nodes[3].x_non_clause == "signature_block"
    # And it reaches the observations as one typed clause, with the record.
    rows = read_observations_jsonl(out / "observations.jsonl")
    (row,) = [r for r in rows if r["taxonomy_id"] == "confidentiality"]
    assert row["x_resegmented"]["basis"] == [rs.BASIS_REDLINE] * 4
    manifest = json.loads((out / "corpus_manifest.json").read_text())
    assert manifest[0]["resegmentation"] == {"splits": 0, "joins": 4}
    # The sample put to the check shows the four passages, each whole.
    (subject,) = rs.check_subjects(out)
    assert subject.payload["check"] == "join"
    assert subject.payload["passages"] == _INSERTED


def test_join_passages_are_cut_in_the_right_places_when_names_are_pseudonymized(
    tmp_path: Path,
) -> None:
    out = _mine_redlined(tmp_path, known_entities=[_ENTITY])
    tree = ClauseTree.load(out / "normalized" / "deal-001" / "v1.clauses.json")
    joined = next(n for n in tree.all_nodes() if n.x_resegmented)
    assert _ENTITY not in joined.text  # the name was aliased, so the text changed length
    (subject,) = rs.check_subjects(out)
    passages = subject.payload["passages"]
    assert len(passages) == 5
    # Each passage is a whole piece, the name aliased: no cut falls inside a word.
    assert passages[1:] == _INSERTED[1:]
    assert passages[0].endswith(_INSERTED[0].split(_ENTITY, 1)[1])
    assert _ENTITY not in passages[0]
    assert "\n".join(passages) == joined.text
    # The observation's own record moved with its text.
    rows = read_observations_jsonl(out / "observations.jsonl")
    (row,) = [r for r in rows if r["taxonomy_id"] == "confidentiality"]
    cuts = [0, *row["x_resegmented"]["at"], len(row["full_text"])]
    pieces = [row["full_text"][a:b].strip() for a, b in zip(cuts, cuts[1:], strict=False)]
    assert pieces == passages


# ---------------------------------------------------------------------------
# The resegment_check kind
# ---------------------------------------------------------------------------


def test_check_sample_is_the_recorded_splits_and_joins_and_deterministic(tmp_path: Path) -> None:
    out = tmp_path / "out"
    _mine(tmp_path, out)
    subjects = rs.check_subjects(out)
    assert [s.payload["check"] for s in subjects] == ["split"]
    (subject,) = subjects
    assert subject.payload["clause_type"] == {"id": "insurance", "label": "Insurance"}
    assert subject.payload["text"].startswith("Both sides must keep")
    assert subject.payload["context_before"].endswith("for the college's learners.")
    assert subject.key == judge_payload_key(subject.payload)
    assert rs.check_subjects(out) == subjects  # same store, same sample


def test_check_sample_is_capped_at_sixty_and_keeps_the_lowest_hashes(tmp_path: Path) -> None:
    out = tmp_path / "out"
    nodes = []
    start = 0
    for i in range(75):
        first = f"The tenant number {i} shall keep the premises clean and shall"
        nodes.append(_node(f"{2 * i + 1}", first, start))
        start += len(first) + 1
        second = f"keep item {i} in good order at all times."
        nodes.append(_node(f"{2 * i + 2}", second, start))
        start += len(second) + 1
    # Joins at the same level only chain 4 deep; make the pairs separate trees.
    for i in range(0, len(nodes), 2):
        tree = _tree(nodes[i : i + 2])
        classified = [ClassifiedClause(n, _cls(None)) for n in tree.nodes]
        res = rs.resegment(tree, classified)
        assert res.joins == 1
        target = out / "normalized" / "doc" / f"v{i}.clauses.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        res.tree.write(target)
    subjects = rs.check_subjects(out)
    assert len(subjects) == rs.CHECK_SAMPLE_MAX == 60
    assert [s.key for s in subjects] == sorted(s.key for s in subjects)
    everything = rs.check_subjects(out, limit=1000)
    assert len(everything) == 75
    assert [s.key for s in subjects] == [s.key for s in everything][:60]
    assert all(s.payload["check"] == "join" and len(s.payload["passages"]) == 2 for s in subjects)


def test_check_kind_is_a_registered_verdict_and_rubric_kind() -> None:
    assert rs.CHECK_KIND in JUDGE_KINDS
    assert rubric_version(rs.CHECK_KIND).startswith("v1+")
    validate_verdict(rs.CHECK_KIND, {"resegment_ok": True})
    validate_verdict(rs.CHECK_KIND, {"resegment_ok": False, "reason": "a different clause"})
    assert infer_verdict_kind({"resegment_ok": True}) == rs.CHECK_KIND
    for bad in ({"resegment_ok": "yes"}, {}, {"resegment_ok": True, "reason": 3}):
        with pytest.raises(ValueError):
            validate_verdict(rs.CHECK_KIND, bad)
    # Producers cannot smuggle an engine-internal basis.
    with pytest.raises(ValueError):
        validate_verdict(rs.CHECK_KIND, {"resegment_ok": True, "basis": "needs_review"})
    assert rs.CHECK_KIND in ENUM_LABELS


def test_judge_queues_replays_and_requeues_on_a_moved_rubric(tmp_path: Path) -> None:
    out = tmp_path / "out"
    _mine(tmp_path, out)
    store = VerdictStore(out / "judge" / "verdicts.jsonl")
    pending = PendingQueue(out / "judge" / "pending.jsonl")
    judge = StoreBackedResegmentCheckJudge(store=store, pending=pending)

    subjects, unchecked = queue_resegment_checks(out, judge)
    assert (subjects, unchecked) == (1, 1)
    queued = [
        json.loads(line) for line in (out / "judge" / "pending.jsonl").read_text().splitlines()
    ]
    assert [q["kind"] for q in queued] == [rs.CHECK_KIND]
    assert queued[0]["rubric_version"] == rubric_version(rs.CHECK_KIND)

    # A queued item is idempotent (deduplicated by key).
    queue_resegment_checks(out, judge)
    assert len((out / "judge" / "pending.jsonl").read_text().splitlines()) == 1

    # Bank the answer exactly as judge-apply does: stamped with the queued rubric.
    store.put_by_key(
        queued[0]["key"],
        {"resegment_ok": True},
        rubric=RubricStamp(kind=rs.CHECK_KIND, version=queued[0]["rubric_version"]),
    )
    assert queue_resegment_checks(out, judge) == (1, 0)

    # A verdict banked under another rubric is not replayed: re-queued.
    store.put_by_key(
        queued[0]["key"],
        {"resegment_ok": True},
        rubric=RubricStamp(kind=rs.CHECK_KIND, version="v0.stale"),
    )
    assert queue_resegment_checks(out, judge) == (1, 1)

    # A malformed stored verdict reads as unchecked rather than crashing.
    store.put_by_key(
        queued[0]["key"],
        {"resegment_ok": "maybe"},
        rubric=RubricStamp(kind=rs.CHECK_KIND, version=queued[0]["rubric_version"]),
    )
    assert queue_resegment_checks(out, judge) == (1, 1)


def test_scorecard_counts_the_pass_and_its_check(tmp_path: Path) -> None:
    out = tmp_path / "out"
    _mine(tmp_path, out)
    card = build_scorecard(out)
    assert card["scorecard_version"] == 10
    section = card["resegmentation"]
    assert section == {
        "splits": 1,
        "joins": 0,
        "observations_split": 1,
        "observations_joined": 0,
        "check_sample": 1,
        "checked": 0,
        "wrong": 0,
    }
    (subject,) = rs.check_subjects(out)
    store = VerdictStore(out / "judge" / "verdicts.jsonl")
    store.put_by_key(
        subject.key,
        {"resegment_ok": False},
        rubric=RubricStamp(kind=rs.CHECK_KIND, version=rubric_version(rs.CHECK_KIND)),
    )
    section = build_scorecard(out)["resegmentation"]
    assert (section["checked"], section["wrong"]) == (1, 1)
    # Counts only: every key is a field name and every value a count.
    assert all(isinstance(v, int) for v in section.values())


def test_scorecard_leaves_non_clauses_out_of_the_classified_share(tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    rows = [
        {"observation_id": "d/v1/1", "taxonomy_id": "term", "outcome": "signed",
         "x_classification_basis": "llm_segmenter", "citation": {"document_id": "d"}},
        {"observation_id": "d/v1/2", "taxonomy_id": None, "outcome": "signed",
         "x_classification_basis": "unclassified", "x_non_clause": "numbering_only",
         "citation": {"document_id": "d"}},
        {"observation_id": "d/v1/3", "taxonomy_id": None, "outcome": "signed",
         "x_classification_basis": "unclassified", "citation": {"document_id": "d"}},
    ]  # fmt: skip
    (out / "observations.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
    )
    block = build_scorecard(out)["classification"]
    assert block["observations"] == 3
    assert block["non_clause"] == 1
    assert block["non_clause_by_reason"] == {"numbering_only": 1}
    assert block["clauses"] == 2
    assert block["classified"] == 1
    assert block["classified_share"] == 0.5
    assert block["by_basis"] == {"llm_segmenter": 1, "unclassified": 1}
