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

from pathlib import Path

from playbook_engine.clause_tree import ClauseNode, ClauseTree
from playbook_engine.rtf_ingester import ingest_rtf
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
