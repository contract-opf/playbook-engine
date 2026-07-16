"""Tests for the signed-copy detector.

SECURITY NOTE: All fixtures use programmatically constructed ClauseTree
objects with synthetic text.  No real agreement files are referenced.
Party names use fictional identifiers only ("Alice Corp", "Beta Ltd",
"Party A", "Party B", "Alice", "Bob").
"""

from __future__ import annotations

from playbook_engine.clause_tree import ClauseNode, ClauseTree
from playbook_engine.signed_detector import (
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
    assert any(n.clause_path == "9" for n in nodes)


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
    result = detect_signed(_empty_sig_section_tree())
    assert result.signed is False
    assert result.basis == "empty_signature_section"


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
