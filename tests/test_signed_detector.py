"""Tests for the signed-copy detector.

SECURITY NOTE: Fixtures are either programmatically constructed ClauseTree
objects or RTF documents written as Python string literals at test runtime
(see the absorbed-trailer section, which must go through a real ingester to
reproduce its bug).  No real agreement files are committed or referenced.
Party names use fictional identifiers only ("Alice Corp", "Beta Ltd",
"AlphaCorp Holdings", "Beta Industries", "Party A", "Party B", "Alice",
"Bob", "Dana Reyes", "Morgan Ellery").
"""

from __future__ import annotations

import json
from pathlib import Path

from playbook_engine.clause_tree import ClauseNode, ClauseTree
from playbook_engine.rtf_ingester import ingest_rtf
from playbook_engine.segmentation_grounding import Block, SegNode, ground_segmentation
from playbook_engine.signed_detector import (
    _SIG_HEADING,
    _SIG_TRAILER,
    AMBIGUITY_THRESHOLD,
    SignedJudge,
    SignedStatus,
    _count_by_lines,
    _node_subtree_text,
    _signature_nodes,
    detect_signed,
    strip_signature_block,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tree(*nodes: ClauseNode) -> ClauseTree:
    return ClauseTree(document_id="test", version="v1", source_file="test.docx", nodes=list(nodes))


def _node(
    path: str,
    heading: str | None = None,
    text: str = "",
) -> ClauseNode:
    return ClauseNode(
        clause_path=path,
        heading=heading,
        text=text,
        char_span=(0, max(1, len(heading or ""))),
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _docusign_tree() -> ClauseTree:
    """Tree that contains a DocuSign envelope id — strongest signal."""
    return _tree(
        _node("1", "Definitions", "Defined terms appear herein."),
        _node(
            "2",
            "Signatures",
            "DocuSign Envelope ID: 12A34B56-78CD-90EF-ABCD-123456789ABC\n"
            "By: Alice Smith\n"
            "By: Bob Jones",
        ),
    )


def _dual_filled_tree() -> ClauseTree:
    """Two filled 'By:' lines — dual-party execution."""
    return _tree(
        _node("1", "Representations", "Alice Corp represents the following."),
        _node(
            "9",
            "Signatures",
            "By: Alice Smith\nTitle: CEO\nBy: Bob Jones\nTitle: VP",
        ),
    )


def _single_filled_tree() -> ClauseTree:
    """One filled 'By:' line — single party signed."""
    return _tree(
        _node("1", "Obligations", "Party A shall deliver."),
        _node("8", "Signature", "By: Alice Smith\nTitle: Director"),
    )


def _slash_s_tree() -> ClauseTree:
    """Electronic /s/ format signature."""
    return _tree(
        _node("1", "Terms", "The parties agree."),
        _node(
            "7",
            "Execution",
            "/s/ Alice Smith\nDate: 2025-01-15\n/s/ Bob Jones\nDate: 2025-01-15",
        ),
    )


def _single_slash_s_tree() -> ClauseTree:
    """Single /s/ — electronic_signature basis."""
    return _tree(
        _node("1", "Terms", "Body text."),
        _node("5", "Signatures", "/s/ Alice Smith\nDate: 2025-03-01"),
    )


def _blank_blocks_tree() -> ClauseTree:
    """Signature section exists but all 'By:' lines are blank."""
    return _tree(
        _node("1", "Obligations", "Party A shall deliver."),
        _node(
            "9",
            "Signatures",
            "By: _____________________________\nTitle: _______________\n"
            "By: _____________________________\nTitle: _______________",
        ),
    )


def _no_sig_tree() -> ClauseTree:
    """No signature section at all."""
    return _tree(
        _node("1", "Definitions", "Terms defined herein."),
        _node("2", "Obligations", "Party A shall deliver."),
    )


def _witness_whereof_tree() -> ClauseTree:
    """'In Witness Whereof' heading (common signed-copy pattern)."""
    return _tree(
        _node("1", "General", "Body text."),
        _node(
            "10",
            "In Witness Whereof",
            "By: Alice Smith\nTitle: CEO\nBy: Bob Jones\nTitle: President",
        ),
    )


def _empty_sig_section_tree() -> ClauseTree:
    """Signature heading with no body text — empty section."""
    return _tree(
        _node("1", "Definitions", "Body text."),
        _node("9", "Signatures", ""),
    )


def _trailer_only_zero_evidence_tree() -> ClauseTree:
    """Body text mentions execution boilerplate but carries zero signature
    evidence, and no node heading matches _SIG_HEADING.

    This is the issue #117 case: the ONLY reason any node qualifies as a
    signature node is a body-text _SIG_TRAILER hit ("in witness whereof")
    inside an ordinary "Miscellaneous" clause — no filled/blank By: line, no
    /s/ marker, no real heading.  Must land on the confident
    unsigned_trailer_reference basis, not the ambiguous
    empty_signature_section one.
    """
    return _tree(
        _node("1", "Definitions", "Body text."),
        _node(
            "12",
            "Miscellaneous",
            "This Agreement may be executed in counterparts, each of which "
            "IN WITNESS WHEREOF shall constitute an original.",
        ),
    )


def _mixed_heading_and_trailer_zero_evidence_tree() -> ClauseTree:
    """One node matches via a real heading, another only via body-text
    _SIG_TRAILER — both with zero By:/`/s/` evidence.

    Heading provenance must dominate: the document stays in the ambiguous
    empty_signature_section bucket at 0.60 rather than the confident
    unsigned_trailer_reference bucket, because a real heading elsewhere is
    stronger evidence a genuine signature section exists.
    """
    return _tree(
        _node("1", "Definitions", "Body text."),
        _node("9", "Signatures", ""),
        _node(
            "12",
            "Miscellaneous",
            "This Agreement may be executed in counterparts, each of which "
            "IN WITNESS WHEREOF shall constitute an original.",
        ),
    )


def _mixed_filled_blank_tree() -> ClauseTree:
    """One party signed, one blank — should count as single_signature."""
    return _tree(
        _node(
            "8",
            "Signatures",
            "By: Alice Smith\nTitle: CEO\nBy: _____________________\nTitle: VP",
        ),
    )


def _table_layout_dual_signatures_tree() -> ClauseTree:
    """A signed execution page laid out as a 2-column DOCX table.

    Mirrors docx_ingester._flatten_table's output: every cell in the table
    (both rows, both columns) is joined with " | " into ONE line, so both
    "By:" occurrences land mid-line rather than at line start (issue #94).
    """
    return _tree(
        _node(
            "9",
            "Signatures",
            "ALICE CORP | BETA LTD | By: Alice Smith | By: Bob Jones | "
            "Title: CEO | Title: President",
        ),
    )


# ---------------------------------------------------------------------------
# SignedStatus dataclass
# ---------------------------------------------------------------------------


def test_signed_status_fields() -> None:
    s = SignedStatus(signed=True, basis="dual_signatures", confidence=0.90)
    assert s.signed is True
    assert s.basis == "dual_signatures"
    assert s.confidence == 0.90


def test_signed_status_requires_confidence() -> None:
    import pytest

    with pytest.raises(TypeError):
        SignedStatus(signed=False, basis="no_signature_section")  # type: ignore[call-arg]


def test_signed_status_frozen() -> None:
    s = SignedStatus(signed=True, basis="docusign_cert", confidence=0.95)
    try:
        s.signed = False  # type: ignore[misc]
        raise AssertionError("should have raised")
    except (AttributeError, TypeError):
        pass


def test_signed_status_invalid_basis() -> None:
    import pytest

    with pytest.raises(ValueError, match="Unknown basis"):
        SignedStatus(signed=True, basis="made_up_basis", confidence=0.9)


def test_signed_status_confidence_out_of_range() -> None:
    import pytest

    with pytest.raises(ValueError, match="confidence"):
        SignedStatus(signed=True, basis="docusign_cert", confidence=1.5)


# ---------------------------------------------------------------------------
# _count_by_lines
# ---------------------------------------------------------------------------


def test_count_by_lines_filled() -> None:
    text = "By: Alice Smith\nTitle: CEO"
    filled, blank = _count_by_lines(text)
    assert filled == 1
    assert blank == 0


def test_count_by_lines_blank_underscores() -> None:
    text = "By: _____________________________"
    filled, blank = _count_by_lines(text)
    assert filled == 0
    assert blank == 1


def test_count_by_lines_blank_empty() -> None:
    text = "By:    "
    filled, blank = _count_by_lines(text)
    assert filled == 0
    assert blank == 1


def test_count_by_lines_dual_filled() -> None:
    text = "By: Alice Smith\nBy: Bob Jones"
    filled, blank = _count_by_lines(text)
    assert filled == 2
    assert blank == 0


def test_count_by_lines_mixed() -> None:
    text = "By: Alice Smith\nBy: _____________________"
    filled, blank = _count_by_lines(text)
    assert filled == 1
    assert blank == 1


def test_count_by_lines_no_by_lines() -> None:
    text = "No signature block here."
    filled, blank = _count_by_lines(text)
    assert filled == 0
    assert blank == 0


def test_count_by_lines_table_layout_mid_line() -> None:
    """Two 'By:' cells flattened into one pipe-joined table line (issue #94)."""
    text = "ALICE CORP | BETA LTD | By: Alice Smith | By: Bob Jones"
    filled, blank = _count_by_lines(text)
    assert filled == 2
    assert blank == 0


def test_count_by_lines_table_layout_blank_mid_line() -> None:
    """Two blank 'By:' cells mid-line must still count as blank, not filled."""
    text = "By: _______________ | By: _______________"
    filled, blank = _count_by_lines(text)
    assert filled == 0
    assert blank == 2


# ---------------------------------------------------------------------------
# _signature_nodes
# ---------------------------------------------------------------------------


def test_signature_nodes_finds_signatures_heading() -> None:
    tree = _dual_filled_tree()
    nodes = _signature_nodes(tree)
    assert any(node.clause_path == "9" for node, _provenance in nodes)


def test_signature_nodes_finds_execution_heading() -> None:
    tree = _slash_s_tree()
    nodes = _signature_nodes(tree)
    assert len(nodes) >= 1


def test_signature_nodes_finds_in_witness_whereof() -> None:
    tree = _witness_whereof_tree()
    nodes = _signature_nodes(tree)
    assert len(nodes) >= 1


def test_signature_nodes_empty_on_no_sig_tree() -> None:
    tree = _no_sig_tree()
    nodes = _signature_nodes(tree)
    assert nodes == []


def test_signature_nodes_tags_heading_provenance() -> None:
    """A node whose heading matches _SIG_HEADING is tagged 'heading' (issue #117)."""
    tree = _dual_filled_tree()
    nodes = _signature_nodes(tree)
    assert any(node.clause_path == "9" and provenance == "heading" for node, provenance in nodes)


def test_signature_nodes_tags_trailer_provenance() -> None:
    """A node with no matching heading, matched only via body-text _SIG_TRAILER,
    is tagged 'trailer' (issue #117)."""
    tree = _trailer_only_zero_evidence_tree()
    nodes = _signature_nodes(tree)
    assert nodes == [(tree.nodes[1], "trailer")]


# ---------------------------------------------------------------------------
# detect_signed: positive cases
# ---------------------------------------------------------------------------


def test_detect_signed_docusign_cert() -> None:
    result = detect_signed(_docusign_tree())
    assert result.signed is True
    assert result.basis == "docusign_cert"
    assert result.confidence >= 0.90


def test_detect_signed_dual_signatures() -> None:
    result = detect_signed(_dual_filled_tree())
    assert result.signed is True
    assert result.basis == "dual_signatures"
    assert result.confidence >= 0.85


def test_table_layout_dual_signatures() -> None:
    """A signed execution page laid out as a 2-column table must still yield
    basis=dual_signatures (issue #94: table flattening put both 'By:' cells
    mid-line, defeating the line-start-anchored regex)."""
    result = detect_signed(_table_layout_dual_signatures_tree())
    assert result.signed is True
    assert result.basis == "dual_signatures"
    assert result.confidence >= 0.85


def test_detect_signed_single_signature() -> None:
    result = detect_signed(_single_filled_tree())
    assert result.signed is True
    assert result.basis == "single_signature"
    assert result.confidence >= 0.70


def test_detect_signed_slash_s_dual() -> None:
    result = detect_signed(_slash_s_tree())
    assert result.signed is True
    assert result.basis == "dual_signatures"


def test_detect_signed_slash_s_single() -> None:
    result = detect_signed(_single_slash_s_tree())
    assert result.signed is True
    assert result.basis == "electronic_signature"


def test_detect_signed_witness_whereof() -> None:
    result = detect_signed(_witness_whereof_tree())
    assert result.signed is True


# ---------------------------------------------------------------------------
# detect_signed: negative cases
# ---------------------------------------------------------------------------


def test_detect_not_signed_blank_blocks() -> None:
    result = detect_signed(_blank_blocks_tree())
    assert result.signed is False
    assert result.basis == "blank_signature_blocks"
    assert result.confidence >= 0.70


def test_detect_not_signed_no_section() -> None:
    result = detect_signed(_no_sig_tree())
    assert result.signed is False
    assert result.basis == "no_signature_section"
    assert result.confidence >= 0.70


def test_detect_not_signed_empty_sig_section() -> None:
    """Heading-matched empty section: unchanged 0.60, still escalates (issue #117)."""
    result = detect_signed(_empty_sig_section_tree())
    assert result.signed is False
    assert result.basis == "empty_signature_section"
    assert result.confidence == 0.60
    assert result.confidence < AMBIGUITY_THRESHOLD, "must still land below threshold to escalate"


# ---------------------------------------------------------------------------
# detect_signed: provenance-split confidence (issue #117)
#
# `d9ffde7` widened _signature_nodes to also match _SIG_TRAILER in body text
# (the absorbed-trailer fix).  That widening pulled 70/207 real-corpus
# documents — trailer boilerplate mentioned in passing, with zero filled or
# blank By: evidence — down to the ambiguous 0.60 empty_signature_section
# confidence, sending them to LLM arbitration where they previously did not
# go at all.  These tests cover the provenance split that fixes it: a
# trailer-only match with zero evidence gets a confident, deterministic
# "not signed" instead.
# ---------------------------------------------------------------------------


def test_detect_not_signed_trailer_only_zero_evidence_is_confident() -> None:
    """Trailer-only match, zero By:/`/s/` evidence → confident not-signed,
    NOT the ambiguous empty_signature_section basis."""
    result = detect_signed(_trailer_only_zero_evidence_tree())
    assert result.signed is False
    assert result.basis == "unsigned_trailer_reference"
    assert result.confidence >= AMBIGUITY_THRESHOLD, "must not need escalation"


def test_signed_judge_not_called_for_trailer_only_zero_evidence() -> None:
    """The judge must NOT be invoked for the confident trailer-only case."""
    tree = _trailer_only_zero_evidence_tree()
    verdict = SignedStatus(signed=True, basis="llm", confidence=0.5)
    judge = _RecordingJudge(verdict)

    result = detect_signed(tree, signed_judge=judge)

    assert judge.calls == [], "judge must not be called once confidence clears AMBIGUITY_THRESHOLD"
    assert result.basis == "unsigned_trailer_reference"


def test_detect_not_signed_mixed_heading_and_trailer_stays_heading_provenance() -> None:
    """A document with both a heading match and a trailer-only match keeps
    heading provenance — a real heading elsewhere dominates, so the document
    stays ambiguous at 0.60 rather than jumping to the confident trailer-only
    basis."""
    result = detect_signed(_mixed_heading_and_trailer_zero_evidence_tree())
    assert result.signed is False
    assert result.basis == "empty_signature_section"
    assert result.confidence == 0.60


# ---------------------------------------------------------------------------
# detect_signed: edge cases
# ---------------------------------------------------------------------------


def test_detect_signed_mixed_filled_blank_counts_single() -> None:
    """One filled + one blank → single_signature (not dual)."""
    result = detect_signed(_mixed_filled_blank_tree())
    assert result.signed is True
    assert result.basis == "single_signature"


def test_detect_signed_empty_tree() -> None:
    tree = ClauseTree(document_id="d", version="v1", source_file="f")
    result = detect_signed(tree)
    assert result.signed is False
    assert result.basis == "no_signature_section"


def test_detect_signed_confidence_in_range() -> None:
    for tree in [
        _docusign_tree(),
        _dual_filled_tree(),
        _single_filled_tree(),
        _blank_blocks_tree(),
        _no_sig_tree(),
    ]:
        r = detect_signed(tree)
        assert 0.0 <= r.confidence <= 1.0, f"confidence {r.confidence} out of range for {r}"


def test_ambiguity_threshold_constant() -> None:
    assert 0.0 < AMBIGUITY_THRESHOLD < 1.0


def test_high_confidence_above_ambiguity() -> None:
    """DocuSign cert and dual-party must be above the ambiguity threshold."""
    assert detect_signed(_docusign_tree()).confidence > AMBIGUITY_THRESHOLD
    assert detect_signed(_dual_filled_tree()).confidence > AMBIGUITY_THRESHOLD


def test_blank_blocks_above_ambiguity() -> None:
    """Definitive blank-block detection should also be above ambiguity threshold."""
    assert detect_signed(_blank_blocks_tree()).confidence >= AMBIGUITY_THRESHOLD


# ---------------------------------------------------------------------------
# detect_signed: case-insensitive heading matching
# ---------------------------------------------------------------------------


def test_heading_case_insensitive_signatures() -> None:
    tree = _tree(_node("9", "SIGNATURES", "By: Alice Smith"))
    result = detect_signed(tree)
    assert result.signed is True


def test_heading_case_insensitive_execution() -> None:
    tree = _tree(_node("9", "EXECUTION", "By: Alice Smith\nBy: Bob Jones"))
    result = detect_signed(tree)
    assert result.signed is True


# ---------------------------------------------------------------------------
# Regression: B1 — signature content in segmenter-promoted children
# ---------------------------------------------------------------------------


def test_b1_signature_in_child_nodes_detected() -> None:
    """Signatures promoted to child nodes by the segmenter must be found.

    Before B1 fix: _node_subtree_text did not recurse → parent text was empty
    → detect_signed returned blank_signature_blocks/signed=False even for an
    executed agreement whose By: lines had been promoted to children.
    """
    sig_node = ClauseNode(
        clause_path="9",
        heading="Signatures",
        text="",
        char_span=(0, 10),
        children=[
            ClauseNode(
                clause_path="9.a",
                heading=None,
                text="By: Alice Smith\nTitle: CEO",
                char_span=(11, 36),
            ),
            ClauseNode(
                clause_path="9.b",
                heading=None,
                text="By: Bob Jones\nTitle: President",
                char_span=(37, 67),
            ),
        ],
    )
    tree = _tree(_node("1", "Obligations", "Alice Corp shall deliver."), sig_node)
    result = detect_signed(tree)
    assert result.signed is True
    assert result.basis == "dual_signatures"


def test_b1_node_subtree_text_recurses() -> None:
    """_node_subtree_text must include descendant text."""
    parent = ClauseNode(
        clause_path="9",
        heading="Signatures",
        text="",
        char_span=(0, 10),
        children=[
            ClauseNode(
                clause_path="9.a",
                heading=None,
                text="By: Alice Smith",
                char_span=(11, 26),
            ),
        ],
    )
    text = _node_subtree_text(parent)
    assert "By: Alice Smith" in text


# ---------------------------------------------------------------------------
# Regression: B2 — unsigned template mentioning DocuSign must not fire cert
# ---------------------------------------------------------------------------


def test_b2_unsigned_template_with_docusign_mention() -> None:
    """A bare DocuSign mention (no UUID) must not be classified as cert-signed.

    Before B2 fix: _DOCUSIGN_CERT matched 'DocuSign Envelope ID' anywhere,
    including instructional template text, returning signed=True at 0.95.
    """
    tree = _tree(
        _node(
            "1",
            "Instructions",
            "Send via DocuSign. A DocuSign Envelope ID will be assigned automatically.",
        ),
        _node(
            "9",
            "Signatures",
            "By: _____________________________\nBy: _____________________________",
        ),
    )
    result = detect_signed(tree)
    assert result.signed is False


def test_b2_real_docusign_uuid_still_fires() -> None:
    """A real UUID-format DocuSign Envelope ID must still trigger cert detection."""
    tree = _tree(
        _node(
            "9",
            "Signatures",
            "DocuSign Envelope ID: 12a34b56-78cd-90ef-abcd-123456789abc\n"
            "By: Alice Smith\nBy: Bob Jones",
        ),
    )
    result = detect_signed(tree)
    assert result.signed is True
    assert result.basis == "docusign_cert"


# ---------------------------------------------------------------------------
# Regression: B3 — "Execution of Services" must not be a signature section
# ---------------------------------------------------------------------------


def test_b3_execution_of_services_not_sig_section() -> None:
    """'Execution of Services' heading must NOT match the signature section pattern.

    Before B3 fix: _SIG_HEADING matched 'execution' anywhere in the heading,
    treating business-clause headings as signature sections and driving spurious
    blank_signature_blocks determinations.
    """
    tree = _tree(
        _node("1", "Execution of Services", "Alice Corp shall execute the services."),
        _node("2", "Obligations", "Party A shall deliver."),
    )
    sig_nodes = _signature_nodes(tree)
    assert sig_nodes == [], f"Expected no signature nodes, got {[n.clause_path for n in sig_nodes]}"


def test_b3_execution_alone_is_sig_section() -> None:
    """'EXECUTION' as a standalone heading must still match."""
    tree = _tree(_node("9", "EXECUTION", "By: Alice Smith"))
    assert len(_signature_nodes(tree)) == 1


# ---------------------------------------------------------------------------
# SignedJudge protocol seam (P2.4)
# ---------------------------------------------------------------------------


class _RecordingJudge:
    """Test double: records calls and returns a configurable SignedStatus."""

    def __init__(self, verdict: SignedStatus) -> None:
        self.calls: list[str] = []
        self._verdict = verdict

    def judge(self, signature_subtree: str) -> SignedStatus:
        self.calls.append(signature_subtree)
        return self._verdict


def test_signed_judge_protocol_importable() -> None:
    """SignedJudge must be importable and satisfy the Protocol at runtime."""
    # runtime_checkable lets us verify structural conformance without an LLM.
    verdict = SignedStatus(signed=True, basis="llm", confidence=0.85)
    judge = _RecordingJudge(verdict)
    assert isinstance(judge, SignedJudge)


def test_signed_judge_called_on_low_confidence_empty_section() -> None:
    """Judge is called with a non-empty signature_subtree when confidence=0.60.

    The empty_signature_section case (signed_detector.py:194) returns
    confidence=0.60 — the archetypal trigger for LLM arbitration.  The judge
    must receive the sig_text (signature section subtree), which is non-empty
    because the heading itself contributes text via _node_subtree_text.
    """
    tree = _empty_sig_section_tree()  # Signatures node with empty body → confidence=0.60
    verdict = SignedStatus(signed=True, basis="llm", confidence=0.85)
    judge = _RecordingJudge(verdict)

    result = detect_signed(tree, signed_judge=judge)

    assert len(judge.calls) == 1, "judge must be called exactly once"
    assert judge.calls[0] != "", "judge must receive a non-empty signature_subtree"
    assert result is verdict, "judge verdict must replace the low-confidence result"


def test_signed_judge_not_called_on_high_confidence() -> None:
    """Judge must NOT be called when confidence >= AMBIGUITY_THRESHOLD.

    Dual signatures return confidence=0.90 — well above 0.70.
    """
    tree = _dual_filled_tree()
    verdict = SignedStatus(signed=False, basis="llm", confidence=0.10)
    judge = _RecordingJudge(verdict)

    result = detect_signed(tree, signed_judge=judge)

    assert judge.calls == [], "judge must not be called for high-confidence result"
    assert result.basis == "dual_signatures", "original deterministic result must be returned"


def test_signed_judge_verdict_replaces_low_confidence_result() -> None:
    """Judge's SignedStatus fully replaces the low-confidence deterministic result."""
    tree = _empty_sig_section_tree()
    # Diverge from the deterministic result in every field so substitution is unambiguous.
    verdict = SignedStatus(signed=True, basis="llm", confidence=0.88)
    judge = _RecordingJudge(verdict)

    result = detect_signed(tree, signed_judge=judge)

    assert result.signed is True
    assert result.basis == "llm"
    assert result.confidence == 0.88


# ---------------------------------------------------------------------------
# Regression: execution trailer absorbed into the preceding numbered clause
#
# These fixtures deliberately go through the real RTF ingester rather than
# building a ClauseTree by hand.  The bug lives in the *interaction* between
# the ingester (which starts a clause only on a NUMBERED paragraph, so an
# unnumbered "IN WITNESS WHEREOF …" trailer is appended to the body of the
# last numbered clause) and the detector (which used to look for signature
# sections in headings only).  A synthesised tree would dodge the ingester and
# test nothing.
#
# Before the fix all four fixtures below returned
# SignedStatus(signed=False, basis="no_signature_section", confidence=0.85) —
# a confident wrong answer, which then withholds every observation for the
# document downstream (the issue #200 failure mode).
# ---------------------------------------------------------------------------

_WITNESS_LINE = (
    r"IN WITNESS WHEREOF, the parties have executed this Agreement as of the "
    r"date first written above.\par "
)


def _trailer_rtf(tmp_path: Path, trailer: str, name: str = "executed.rtf") -> Path:
    """Write an RTF of numbered clauses followed by an UNNUMBERED *trailer*."""
    body = (
        r"1. Parties and Recitals\par "
        r"This Agreement is entered into by the parties named below.\par "
        r"2. Purpose\par "
        r"The parties wish to exchange confidential information.\par "
        r"3. Counterparts\par "
        r"This Agreement may be executed in counterparts.\par " + trailer
    )
    content = (
        r"{\rtf1\ansi\deff0"
        r"{\fonttbl{\f0\froman\fcharset0 Times New Roman;}}"
        r"\f0\fs24 " + body + r"}"
    )
    dest = tmp_path / name
    dest.write_text(content, encoding="utf-8")
    return dest


def _assert_trailer_was_absorbed(tree: ClauseTree) -> None:
    """Guard the fixtures' premise: the trailer must NOT have its own heading.

    If the ingester ever learns to split unnumbered execution trailers into
    their own node, these fixtures stop exercising the absorbed-trailer path
    and start passing for the ordinary heading reason instead.  Fail loudly
    so the fixture gets re-pointed rather than silently going vacuous.
    """
    headings = [n.heading for n in tree.all_nodes() if n.heading]
    assert not any(_SIG_HEADING.search(h) for h in headings), (
        f"fixture premise broken: a node heading now matches _SIG_HEADING ({headings!r}); "
        "the trailer is no longer absorbed, so this test no longer covers the bug"
    )


def test_absorbed_slash_s_trailer_is_detected_as_signed(tmp_path: Path) -> None:
    """Numbered clauses + unnumbered /s/ trailer must read as an executed copy."""
    path = _trailer_rtf(
        tmp_path,
        _WITNESS_LINE + r"AlphaCorp Holdings, Inc.\par "
        r"By: /s/ Dana Reyes\par "
        r"Name: Dana Reyes\par "
        r"Title: Vice President, Legal\par "
        r"Beta Industries, LLC\par "
        r"By: /s/ Morgan Ellery\par "
        r"Name: Morgan Ellery\par "
        r"Title: General Counsel\par ",
    )
    tree = ingest_rtf(path, "doc", "v1").tree
    _assert_trailer_was_absorbed(tree)

    result = detect_signed(tree)

    assert result.signed is True, f"executed copy read as unsigned: {result}"
    assert result.basis == "dual_signatures"
    assert result.confidence >= AMBIGUITY_THRESHOLD


def test_absorbed_wet_signature_trailer_is_detected_as_signed(tmp_path: Path) -> None:
    """The same trailer with typed names and no /s/ markers must also be caught.

    A full-text /s/ fallback alone would miss this: the only signal is the
    filled "By:" blocks inside the absorbed trailer, which are reachable only
    once the trailer's body promotes its clause to a signature node.
    """
    path = _trailer_rtf(
        tmp_path,
        _WITNESS_LINE + r"AlphaCorp Holdings, Inc.\par "
        r"By: Dana Reyes\par "
        r"Title: Vice President, Legal\par "
        r"Beta Industries, LLC\par "
        r"By: Morgan Ellery\par "
        r"Title: General Counsel\par ",
    )
    tree = ingest_rtf(path, "doc", "v1").tree
    _assert_trailer_was_absorbed(tree)

    result = detect_signed(tree)

    assert result.signed is True, f"executed copy read as unsigned: {result}"
    assert result.basis == "dual_signatures"


def test_absorbed_slash_s_trailer_without_witness_phrase(tmp_path: Path) -> None:
    """A /s/ block with no execution boilerplate falls back to document-wide /s/.

    Nothing here promotes a node to a signature section, so this exercises the
    unlocalized fallback — signed, but at the degraded 0.85 rather than the
    0.90 a localized dual-signature section earns.
    """
    path = _trailer_rtf(
        tmp_path,
        r"AlphaCorp Holdings, Inc.\par "
        r"By: /s/ Dana Reyes\par "
        r"Beta Industries, LLC\par "
        r"By: /s/ Morgan Ellery\par ",
    )
    tree = ingest_rtf(path, "doc", "v1").tree
    _assert_trailer_was_absorbed(tree)
    assert _signature_nodes(tree) == [], "fixture must have no signature section at all"

    result = detect_signed(tree)

    assert result.signed is True, f"executed copy read as unsigned: {result}"
    assert result.basis == "dual_signatures"
    assert result.confidence == 0.85, "unlocalized markers must not claim localized confidence"


def test_absorbed_unsigned_template_trailer_stays_unsigned(tmp_path: Path) -> None:
    """An UNSIGNED template with the same shape must still be unsigned.

    The verdict was already correct before the fix, but for the wrong reason
    ("no_signature_section" — the blocks were never seen at all).  Now the
    blocks are found and correctly judged blank.
    """
    path = _trailer_rtf(
        tmp_path,
        _WITNESS_LINE + r"AlphaCorp Holdings, Inc.\par "
        r"By: _____________________\par "
        r"Title: _______________\par "
        r"Beta Industries, LLC\par "
        r"By: _____________________\par "
        r"Title: _______________\par ",
    )
    tree = ingest_rtf(path, "doc", "v1").tree
    _assert_trailer_was_absorbed(tree)

    result = detect_signed(tree)

    assert result.signed is False
    assert result.basis == "blank_signature_blocks"


# ---------------------------------------------------------------------------
# Guards on the widened matching
# ---------------------------------------------------------------------------


def test_sig_trailer_ignores_ordinary_signature_prose() -> None:
    """_SIG_TRAILER must not fire on the word "signature" in ordinary body text.

    This is why the body-text match uses the strict _SIG_TRAILER subset rather
    than _SIG_HEADING: counterparts and notices clauses talk about signatures
    constantly, and matching them would turn business clauses into signature
    sections.
    """
    for prose in [
        "Signature pages may be delivered by facsimile or electronic transmission.",
        "Each notice must bear the authorized signature of an officer.",
        "Alice Corp shall execute the services described in Schedule A.",
        "This Agreement may be executed in counterparts.",
    ]:
        assert not _SIG_TRAILER.search(prose), f"_SIG_TRAILER must not match: {prose!r}"


def test_sig_trailer_body_match_does_not_break_execution_of_services() -> None:
    """The B3 guard must survive the body-text widening.

    "Execution of Services" is matched by neither the heading rule (anchored)
    nor the body rule (_SIG_TRAILER omits "execution" entirely).
    """
    tree = _tree(
        _node("1", "Execution of Services", "Alice Corp shall execute the services."),
        _node("2", "Obligations", "Party A shall deliver."),
    )
    assert _signature_nodes(tree) == []


def test_single_unlocalized_slash_s_is_ambiguous_and_escalates() -> None:
    """One /s/ with no signature section is below threshold and reaches the judge.

    A lone unlocalized marker — a stray "/s/" in an exhibit form, say — is
    genuinely ambiguous, so it must escalate rather than assert.
    """
    tree = _tree(_node("1", "Terms", "The parties agree.\n/s/ Alice Smith"))
    assert _signature_nodes(tree) == []

    bare = detect_signed(tree)
    assert bare.basis == "electronic_signature"
    assert bare.confidence < AMBIGUITY_THRESHOLD, "a lone unlocalized marker must escalate"

    verdict = SignedStatus(signed=False, basis="llm", confidence=0.9)
    judge = _RecordingJudge(verdict)
    assert detect_signed(tree, signed_judge=judge) is verdict
    assert len(judge.calls) == 1


def test_localized_slash_s_keeps_higher_confidence_than_unlocalized() -> None:
    """The section-localized path must stay strictly more confident."""
    localized = detect_signed(_slash_s_tree())
    tree = _tree(_node("1", "Terms", "/s/ Alice Smith\n/s/ Bob Jones"))
    assert _signature_nodes(tree) == []
    unlocalized = detect_signed(tree)

    assert localized.basis == unlocalized.basis == "dual_signatures"
    assert unlocalized.confidence < localized.confidence


# ---------------------------------------------------------------------------
# strip_signature_block (issue #217)
# ---------------------------------------------------------------------------
#
# Driven through the real RTF ingester (the producer that absorbs an
# unnumbered execution trailer into the last clause), so the trees below have
# exactly the shape the pipeline hands strip_signature_block.

_COUNTERPARTS_BODY = "This Agreement may be executed in counterparts."

_SIGNED_TRAILER = (
    _WITNESS_LINE + r"Acme Widgets, Inc.\par "
    r"By: /s/ Sam Signer\par "
    r"Name: Sam Signer\par "
    r"Title: Director\par "
    r"Example Supplies LLC\par "
    r"By: /s/ Robin Roe\par "
    r"Name: Robin Roe\par "
    r"Title: Manager\par "
)


def _rtf_normalized_text(path: Path) -> str:
    from striprtf.striprtf import rtf_to_text

    from playbook_engine.rtf_ingester import _split_lines

    raw = path.read_text(encoding="utf-8", errors="replace")
    return "\n".join(_split_lines(rtf_to_text(raw, encoding="utf-8", errors="replace")))


def test_strip_cuts_witness_trailer_from_last_clause(tmp_path: Path) -> None:
    path = _trailer_rtf(tmp_path, _SIGNED_TRAILER)
    tree = ingest_rtf(path, "doc", "v1").tree
    _assert_trailer_was_absorbed(tree)
    normalized = _rtf_normalized_text(path)

    stripped, block = strip_signature_block(tree)

    assert block is not None
    assert block.basis == "trailer"
    assert block.clause_path == "3"
    last = stripped.resolve_path("3")
    assert last is not None
    assert last.text == _COUNTERPARTS_BODY
    for residue in ("IN WITNESS WHEREOF", "By:", "Sam Signer", "Robin Roe", "Example Supplies"):
        assert residue not in json.dumps(stripped.to_dict())
    # The clause's span now ends with its kept text; the block's span resolves
    # to exactly the removed trailer, through the end of the document.
    assert ClauseTree.resolve_span(normalized, last.char_span) == (
        "3. Counterparts\n" + _COUNTERPARTS_BODY
    )
    assert block.char_span is not None
    removed = ClauseTree.resolve_span(normalized, block.char_span)
    assert removed.startswith("IN WITNESS WHEREOF")
    assert removed.endswith("Title: Manager")
    assert block.char_span[1] == len(normalized)
    stripped.validate(full_text=normalized)
    # Earlier clauses are untouched.
    assert stripped.resolve_path("1") == tree.resolve_path("1")


def test_strip_never_mutates_input_and_detect_signed_still_reads_it(tmp_path: Path) -> None:
    """The unstripped tree keeps the block — detect_signed's evidence."""
    tree = ingest_rtf(_trailer_rtf(tmp_path, _SIGNED_TRAILER), "doc", "v1").tree
    before = tree.to_dict()
    stripped, _block = strip_signature_block(tree)
    assert tree.to_dict() == before
    assert detect_signed(tree).signed is True
    assert "By:" not in (stripped.resolve_path("3") or _node("x")).text


def test_strip_field_group_without_trailer_takes_party_captions(tmp_path: Path) -> None:
    """No IN WITNESS WHEREOF: a By:/Name: field group in the last clause, plus
    the party-caption lines directly above it, is the block."""
    trailer = (
        r"Acme Widgets, Inc.\par "
        r"By: Sam Signer\par "
        r"Name: Sam Signer\par "
        r"[Counterparty]\par "
        r"By: ______________\par "
        r"Name: ______________\par "
    )
    path = _trailer_rtf(tmp_path, trailer)
    tree = ingest_rtf(path, "doc", "v1").tree
    stripped, block = strip_signature_block(tree)
    assert block is not None
    assert block.basis == "signature_fields"
    last = stripped.resolve_path("3")
    assert last is not None
    assert last.text == _COUNTERPARTS_BODY
    assert block.char_span is not None
    removed = ClauseTree.resolve_span(_rtf_normalized_text(path), block.char_span)
    assert removed.startswith("Acme Widgets, Inc.")


def test_strip_trailer_run_onto_last_sentence_line(tmp_path: Path) -> None:
    """A text layer can run the trailer onto the clause's last line: the cut
    happens right after the sentence end, keeping the clause sentence."""
    body = (
        r"1. Counterparts\par "
        r"This Agreement may be executed in counterparts. IN WITNESS WHEREOF, the "
        r"parties have executed this Agreement.\par "
        r"By: /s/ Sam Signer\par "
        r"Name: Sam Signer\par "
    )
    path = tmp_path / "runon.rtf"
    path.write_text(
        r"{\rtf1\ansi\deff0{\fonttbl{\f0\froman\fcharset0 Times New Roman;}}\f0\fs24 " + body + "}",
        encoding="utf-8",
    )
    stripped, block = strip_signature_block(ingest_rtf(path, "doc", "v1").tree)
    assert block is not None and block.basis == "trailer"
    node = stripped.resolve_path("1")
    assert node is not None
    assert node.text == "This Agreement may be executed in counterparts."


def test_strip_leaves_mid_sentence_mention_alone() -> None:
    """A clause that merely MENTIONS execution boilerplate mid-sentence, with
    no field group, is not a signature block."""
    text = "The parties agree that the phrase in witness whereof has no special effect here."
    tree = _tree(
        ClauseNode(clause_path="1", heading="Misc", text=text, char_span=(0, 5 + len(text)))
    )
    stripped, block = strip_signature_block(tree)
    assert block is None
    assert stripped is tree


def test_strip_single_by_line_is_not_a_block() -> None:
    """One stray "By:" line in prose is not a signature block."""
    text = "Notices may be sent as follows.\nBy: registered mail to the address above."
    tree = _tree(ClauseNode(clause_path="1", heading="Notices", text=text, char_span=(0, 90)))
    assert strip_signature_block(tree) == (tree, None)


def test_strip_keeps_exhibit_after_signature_page(tmp_path: Path) -> None:
    """Only the block-holding clause is cut; an exhibit after it is kept.
    Driven through the DOCX ingester (heading-styled signature page)."""
    from docx import Document

    from playbook_engine.docx_ingester import ingest_docx

    doc = Document()
    doc.add_heading("Law", level=1)
    doc.add_paragraph("Governed by the law of the forum.")
    doc.add_heading("Signatures", level=1)
    doc.add_paragraph("IN WITNESS WHEREOF, the parties sign.")
    doc.add_paragraph("By: /s/ Sam Signer")
    doc.add_paragraph("Name: Sam Signer")
    doc.add_heading("Exhibit A", level=1)
    doc.add_paragraph("Exhibit A lists the Confidential Information categories.")
    path = tmp_path / "exhibit.docx"
    doc.save(str(path))
    result = ingest_docx(path, "doc", "v1")
    normalized = "\n".join(u.text for u in result.units)

    stripped, block = strip_signature_block(result.tree)

    assert block is not None and block.basis == "trailer"
    sig = next(n for n in stripped.all_nodes() if n.heading == "Signatures")
    assert block.clause_path == sig.clause_path
    assert sig.text == ""
    # Nothing kept: the span shrinks back to the heading line.
    assert sig.char_span == sig.heading_span
    assert block.char_span is not None
    assert ClauseTree.resolve_span(normalized, block.char_span) == (
        "IN WITNESS WHEREOF, the parties sign.\nBy: /s/ Sam Signer\nName: Sam Signer"
    )
    exhibit = next(n for n in stripped.all_nodes() if n.heading == "Exhibit A")
    assert exhibit.text == "Exhibit A lists the Confidential Information categories."
    stripped.validate(full_text=normalized)


def test_strip_trailer_heading_node_empties_its_body(tmp_path: Path) -> None:
    """The RTF ingester reads a short ALL-CAPS "IN WITNESS WHEREOF" line as a
    heading, putting the block in that node's body — all of it is cut."""
    path = _trailer_rtf(
        tmp_path,
        r"IN WITNESS WHEREOF\par "
        r"Acme Widgets, Inc.\par "
        r"By: /s/ Sam Signer\par "
        r"Name: Sam Signer\par ",
    )
    tree = ingest_rtf(path, "doc", "v1").tree
    trailer_node = next(n for n in tree.all_nodes() if n.heading == "IN WITNESS WHEREOF")
    assert "Sam Signer" in trailer_node.text  # fixture premise

    stripped, block = strip_signature_block(tree)

    assert block is not None and block.basis == "trailer"
    assert block.clause_path == trailer_node.clause_path
    # Looked up by heading, not clause_path: the RTF ingester numbers an
    # unnumbered ALL-CAPS heading from its own counter, so its path can
    # collide with an earlier numbered clause's.
    cut = next(n for n in stripped.all_nodes() if n.heading == "IN WITNESS WHEREOF")
    assert cut.text == ""
    assert cut.char_span == cut.heading_span
    assert block.char_span is not None
    removed = ClauseTree.resolve_span(_rtf_normalized_text(path), block.char_span)
    assert removed == "Acme Widgets, Inc.\nBy: /s/ Sam Signer\nName: Sam Signer"
    last = stripped.resolve_path("3")
    assert last is not None and last.text == _COUNTERPARTS_BODY


def test_strip_guard_span_not_covering_own_text_cuts_text_reports_no_span() -> None:
    """Defensive guard only (strip_signature_block's ``text_start`` check):
    a node whose char_span cannot end at its own text (here a heading-only
    span with no heading_span) cannot relate its text to offsets, so the text
    is still cut and the block's span is reported None.

    No production caller passes such a node: strip_signature_block only sees
    freshly ingested or grounded trees (whole-clause spans since issue #217),
    and stored normalized/ trees never reach it. This hand-built fixture
    pins the guard's behaviour, not a pipeline input shape.
    """
    text = "Counterparts are fine.\nIN WITNESS WHEREOF, signed.\nBy: /s/ Sam Signer"
    node = ClauseNode("3", "Counterparts", text, (100, 115))
    stripped, block = strip_signature_block(_tree(node))
    assert block is not None and block.char_span is None
    kept = stripped.resolve_path("3")
    assert kept is not None
    assert kept.text == "Counterparts are fine."
    assert kept.char_span == (100, 115)


# ---------------------------------------------------------------------------
# strip_signature_block on heading-less nodes real producers emit (issue #217)
# ---------------------------------------------------------------------------
#
# The LLM/agent segmentation path (segmentation.llm / segmentation.agent —
# the canary corpus and real corpora) builds its trees with
# segmentation_grounding.ground_segmentation: no heading_span, and a
# char_span that is exactly the span of the node's full text (heading line
# included). The ingesters' synthetic pre-heading "0" node has the same
# shape. These trees come from those real producers, not from hand-built
# nodes, so a regression that skips such nodes turns red here.

_GROUNDED_PARAS = (
    "1. Purpose",
    "The parties wish to exchange confidential information.",
    "2. Counterparts",
    _COUNTERPARTS_BODY,
    "IN WITNESS WHEREOF, the parties have executed this Agreement.",
    "Acme Widgets, Inc.",
    "By: /s/ Sam Signer",
    "Name: Sam Signer",
    "Example Supplies LLC",
    "By: /s/ Robin Roe",
    "Name: Robin Roe",
)
_GROUNDED_BLOCK_FIRST = 4  # index of the IN WITNESS WHEREOF paragraph / block
_SIG_RESIDUE = ("IN WITNESS WHEREOF", "By:", "Sam Signer", "Robin Roe", "Example Supplies")


def _grounding_fixture(tmp_path: Path) -> tuple[str, list[Block]]:
    """Canonical text + block stream from the real legacy DOCX extractor —
    the same (canonical_text, blocks) pair the agent path grounds against."""
    from docx import Document

    from playbook_engine.extraction import extract_blocks

    doc = Document()
    for para in _GROUNDED_PARAS:
        doc.add_paragraph(para)
    path = tmp_path / "grounded.docx"
    doc.save(str(path))
    canonical_text, blocks, _label = extract_blocks(path, extractor="legacy")
    assert [b.text for b in blocks] == list(_GROUNDED_PARAS)  # fixture premise
    return canonical_text, blocks


def _ground(
    canonical_text: str, blocks: list[Block], ranges: list[tuple[str, int, int]]
) -> ClauseTree:
    """Ground one top-level SegNode per ``(heading, first_block, last_block)``."""
    seg_nodes = [
        SegNode(
            node_id=f"c{i}",
            parent_id=None,
            order=i,
            heading=heading,
            taxonomy_id=None,
            start_block_id=blocks[first].block_id,
            end_block_id=blocks[last].block_id,
            start_quote=blocks[first].text[:40],
            end_quote=blocks[last].text[-40:],
        )
        for i, (heading, first, last) in enumerate(ranges, start=1)
    ]
    return ground_segmentation(
        document_id="doc",
        version="v1",
        source_file="grounded.docx",
        canonical_text=canonical_text,
        blocks=blocks,
        seg_nodes=seg_nodes,
    ).tree


def test_strip_grounded_tree_block_inside_last_node(tmp_path: Path) -> None:
    """Agent path, execution block segmented INTO the last clause: the text is
    cut, char_span shrinks to the kept text, and the block's span is exactly
    the removed block."""
    canonical, blocks = _grounding_fixture(tmp_path)
    last_block = len(blocks) - 1
    tree = _ground(canonical, blocks, [("1. Purpose", 0, 1), ("2. Counterparts", 2, last_block)])
    grounded = tree.resolve_path("2")
    assert grounded is not None
    # Producer-shape premise: no heading_span; char_span is the full text's span.
    assert grounded.heading_span is None
    assert grounded.char_span == (blocks[2].char_span[0], len(canonical))
    assert ClauseTree.resolve_span(canonical, grounded.char_span) == grounded.text
    assert "By: /s/ Robin Roe" in grounded.text

    stripped, block = strip_signature_block(tree)

    assert block is not None
    assert block.basis == "trailer"
    assert block.clause_path == "2"
    cut = stripped.resolve_path("2")
    assert cut is not None
    # Grounded text starts at the heading block, so what is kept is the
    # heading line plus the clause's own sentence.
    assert cut.text == "2. Counterparts\n" + _COUNTERPARTS_BODY
    assert cut.char_span == (grounded.char_span[0], blocks[3].char_span[1])
    assert ClauseTree.resolve_span(canonical, cut.char_span) == cut.text
    assert block.char_span == (blocks[_GROUNDED_BLOCK_FIRST].char_span[0], len(canonical))
    assert ClauseTree.resolve_span(canonical, block.char_span) == "\n".join(
        _GROUNDED_PARAS[_GROUNDED_BLOCK_FIRST:]
    )
    stripped.validate(full_text=canonical)
    assert stripped.resolve_path("1") == tree.resolve_path("1")
    for residue in _SIG_RESIDUE:
        assert residue not in json.dumps(stripped.to_dict())


def test_strip_grounded_tree_block_as_own_node(tmp_path: Path) -> None:
    """Agent path, execution block segmented as its OWN node (the canary's
    shape): its text becomes empty and its char_span zero-length at the
    block start; the block's span is the node's whole former span."""
    canonical, blocks = _grounding_fixture(tmp_path)
    last_block = len(blocks) - 1
    tree = _ground(
        canonical,
        blocks,
        [
            ("1. Purpose", 0, 1),
            ("2. Counterparts", 2, 3),
            ("Execution", _GROUNDED_BLOCK_FIRST, last_block),
        ],
    )
    execution = tree.resolve_path("3")
    assert execution is not None
    assert execution.heading_span is None  # producer-shape premise
    block_start = blocks[_GROUNDED_BLOCK_FIRST].char_span[0]
    assert execution.char_span == (block_start, len(canonical))

    stripped, block = strip_signature_block(tree)

    assert block is not None
    assert block.basis == "trailer"
    assert block.clause_path == "3"
    cut = stripped.resolve_path("3")
    assert cut is not None
    assert cut.text == ""
    assert cut.char_span == (block_start, block_start)
    assert block.char_span == execution.char_span
    assert ClauseTree.resolve_span(canonical, block.char_span) == "\n".join(
        _GROUNDED_PARAS[_GROUNDED_BLOCK_FIRST:]
    )
    stripped.validate(full_text=canonical)
    assert stripped.resolve_path("1") == tree.resolve_path("1")
    assert stripped.resolve_path("2") == tree.resolve_path("2")
    for residue in _SIG_RESIDUE:
        assert residue not in json.dumps(stripped.to_dict())


def test_strip_synthetic_zero_node_of_heading_less_document(tmp_path: Path) -> None:
    """A heading-less document (a letter agreement) is one synthetic "0" node
    from the real RTF ingester — no heading_span, char_span = its text's span.
    The block at its end is cut and located exactly."""
    sentence = "This letter agreement confirms that each party will keep the other party's secrets."
    path = tmp_path / "letter.rtf"
    path.write_text(
        r"{\rtf1\ansi\deff0{\fonttbl{\f0\froman\fcharset0 Times New Roman;}}\f0\fs24 "
        + sentence
        + r"\par "
        + _SIGNED_TRAILER
        + "}",
        encoding="utf-8",
    )
    tree = ingest_rtf(path, "doc", "v1").tree
    normalized = _rtf_normalized_text(path)
    (zero,) = tree.nodes
    # Producer-shape premise: the synthetic pre-heading node holds everything.
    assert (zero.clause_path, zero.heading, zero.heading_span) == ("0", None, None)
    assert zero.char_span == (0, len(normalized))
    assert "By: /s/ Robin Roe" in zero.text

    stripped, block = strip_signature_block(tree)

    assert block is not None
    assert block.basis == "trailer"
    assert block.clause_path == "0"
    (cut,) = stripped.nodes
    assert cut.text == sentence
    assert cut.char_span == (0, len(sentence))
    assert block.char_span is not None
    removed = ClauseTree.resolve_span(normalized, block.char_span)
    assert removed.startswith("IN WITNESS WHEREOF")
    assert removed.endswith("Title: Manager")
    assert block.char_span[1] == len(normalized)
    stripped.validate(full_text=normalized)
    for residue in _SIG_RESIDUE:
        assert residue not in json.dumps(stripped.to_dict())
