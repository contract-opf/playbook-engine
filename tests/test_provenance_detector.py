"""Tests for the provenance detector.

SECURITY NOTE: All fixtures use programmatically constructed ClauseTree
objects with synthetic text.  No real agreement files are referenced.
Party names use fictional identifiers only ("ACME", "Alpha Corp",
"Beta Ltd", "Party A").
"""

from __future__ import annotations

from playbook_engine.clause_tree import ClauseNode, ClauseTree
from playbook_engine.config import ProvenanceConfig
from playbook_engine.provenance_detector import (
    AMBIGUITY_THRESHOLD,
    PROVENANCE_JUDGE_BASES,
    ProvenanceJudge,
    ProvenanceResult,
    _alias_position_in_recital,
    _alias_re,
    _any_alias_in_text,
    _extract_letterhead,
    _extract_preamble,
    _fingerprint,
    _similarity,
    detect_provenance,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_OUR_ALIASES = ["ACME", "Acme Human Capital", "Acme Works"]


def _config(aliases: list[str] | None = None) -> ProvenanceConfig:
    return ProvenanceConfig(our_party_aliases=_OUR_ALIASES if aliases is None else aliases)


def _node(path: str, heading: str | None = None, text: str = "") -> ClauseNode:
    return ClauseNode(
        clause_path=path,
        heading=heading,
        text=text,
        char_span=(0, max(1, len(heading or ""))),
    )


def _tree(*nodes: ClauseNode, doc_id: str = "doc") -> ClauseTree:
    return ClauseTree(document_id=doc_id, version="v1", source_file="doc.docx", nodes=list(nodes))


# ---------------------------------------------------------------------------
# Our-paper fixtures
# ---------------------------------------------------------------------------


def _our_paper_tree() -> ClauseTree:
    """Agreement where ACME is the first-named party."""
    return _tree(
        _node(
            "0",
            None,
            "This Agreement is entered into by and between ACME, Inc. "
            '("Company") and Alpha Corp ("Client").',
        ),
        _node("1", "Definitions", "Defined terms appear herein."),
        _node("2", "Services", "Company shall provide services as described."),
        _node("3", "Payment", "Client shall pay within thirty (30) days."),
    )


def _our_paper_from_template() -> ClauseTree:
    """Very similar to the template — high template similarity."""
    return _tree(
        _node("0", None, "This Agreement is by and between ACME Works and Beta Ltd."),
        _node("1", "Definitions", "Terms defined herein."),
        _node("2", "Services", "ACME Works shall deliver the engagement."),
        _node("3", "Payment", "Payment shall be thirty days after invoice."),
        _node("4", "Term", "The agreement lasts one year."),
        _node("5", "Termination", "Either party may terminate on thirty days notice."),
    )


# ---------------------------------------------------------------------------
# Counterparty-paper fixtures
# ---------------------------------------------------------------------------


def _counterparty_paper_tree() -> ClauseTree:
    """Agreement where Alpha Corp (counterparty) is the first-named party."""
    return _tree(
        _node(
            "0",
            None,
            "This Agreement is entered into by and between Alpha Corp "
            '("Company") and ACME, Inc. ("Service Provider").',
        ),
        _node("1", "Definitions", "Defined terms appear herein."),
        _node("2", "Services", "Service Provider shall perform services."),
        _node("3", "Payment", "Company shall pay within forty-five (45) days."),
    )


def _counterparty_no_alias_tree() -> ClauseTree:
    """Agreement with no mention of any of our aliases."""
    return _tree(
        _node(
            "0",
            None,
            "This Agreement is between Alpha Corp and Beta Ltd.",
        ),
        _node("1", "Services", "Alpha Corp shall provide services to Beta Ltd."),
        _node("2", "Payment", "Beta Ltd shall pay within sixty days."),
    )


def _counterparty_template_tree() -> ClauseTree:
    """Very different from our template — should be counterparty paper."""
    return _tree(
        _node("0", None, "This Master Services Agreement is between Alpha Corp and its client."),
        _node("1", "Background", "Alpha Corp is a leading provider of professional services."),
        _node("2", "Scope of Work", "Alpha Corp shall provide all deliverables as specified."),
        _node("3", "Fees", "Client shall pay all fees within sixty days of invoice."),
        _node("4", "Intellectual Property", "All IP shall remain with Alpha Corp."),
        _node("5", "Indemnification", "Client indemnifies Alpha Corp against all claims."),
        _node("6", "Governing Law", "This agreement is governed by the laws of New York."),
    )


# ---------------------------------------------------------------------------
# Template (our canonical form)
# ---------------------------------------------------------------------------


def _our_template_tree() -> ClauseTree:
    """Our canonical template document."""
    return _tree(
        _node("0", None, "This Agreement is by and between ACME Works and the Client."),
        _node("1", "Definitions", "Terms defined herein."),
        _node("2", "Services", "ACME Works shall deliver the engagement."),
        _node("3", "Payment", "Payment shall be thirty days after invoice."),
        _node("4", "Term", "The agreement lasts one year."),
        _node("5", "Termination", "Either party may terminate on thirty days notice."),
    )


# ---------------------------------------------------------------------------
# ProvenanceResult dataclass
# ---------------------------------------------------------------------------


def test_provenance_result_our_paper() -> None:
    r = ProvenanceResult(provenance="our_paper", confidence=0.85, basis="alias_first_party")
    assert r.provenance == "our_paper"
    assert r.confidence == 0.85
    assert r.basis == "alias_first_party"


def test_provenance_result_counterparty() -> None:
    r = ProvenanceResult(
        provenance="counterparty_paper", confidence=0.75, basis="alias_second_party"
    )
    assert r.provenance == "counterparty_paper"


def test_provenance_result_frozen() -> None:
    r = ProvenanceResult(provenance="our_paper", confidence=0.85, basis="alias_first_party")
    try:
        r.provenance = "counterparty_paper"  # type: ignore[misc]
        raise AssertionError("should have raised")
    except (AttributeError, TypeError):
        pass


def test_provenance_result_invalid_provenance() -> None:
    import pytest

    with pytest.raises(ValueError, match="Unknown provenance"):
        ProvenanceResult(provenance="unknown", confidence=0.80, basis="alias_first_party")


def test_provenance_result_invalid_basis() -> None:
    import pytest

    with pytest.raises(ValueError, match="Unknown basis"):
        ProvenanceResult(provenance="our_paper", confidence=0.80, basis="bad_basis")


def test_provenance_result_confidence_out_of_range() -> None:
    import pytest

    with pytest.raises(ValueError, match="confidence"):
        ProvenanceResult(provenance="our_paper", confidence=1.5, basis="alias_first_party")


# ---------------------------------------------------------------------------
# _fingerprint and _similarity (unit)
# ---------------------------------------------------------------------------


def test_fingerprint_returns_strings() -> None:
    tree = _our_paper_tree()
    fp = _fingerprint(tree)
    assert isinstance(fp, list)
    assert all(isinstance(s, str) for s in fp)


def test_fingerprint_strips_whitespace() -> None:
    tree = _tree(_node("1", heading="  Definitions  ", text="  Body.  "))
    fp = _fingerprint(tree)
    assert all(s == s.strip() for s in fp)


def test_similarity_identical() -> None:
    fp = _fingerprint(_our_template_tree())
    assert _similarity(fp, fp) == 1.0


def test_similarity_different() -> None:
    fp1 = _fingerprint(_our_template_tree())
    fp2 = _fingerprint(_counterparty_template_tree())
    assert _similarity(fp1, fp2) < 0.5


def test_similarity_empty_both() -> None:
    assert _similarity([], []) == 1.0


def test_similarity_one_empty() -> None:
    assert _similarity(["a", "b"], []) == 0.0


# ---------------------------------------------------------------------------
# _alias_position_in_recital (unit)
# ---------------------------------------------------------------------------


def test_alias_position_first() -> None:
    text = 'This Agreement is by and between ACME, Inc. ("Company") and Alpha Corp ("Client").'
    result = _alias_position_in_recital(text, _OUR_ALIASES)
    assert result == "first"


def test_alias_position_second() -> None:
    text = 'This Agreement is between Alpha Corp ("Company") and ACME ("Service Provider").'
    result = _alias_position_in_recital(text, _OUR_ALIASES)
    assert result == "second"


def test_alias_position_unknown_no_between() -> None:
    text = "Alpha Corp and ACME agree to the following terms."
    result = _alias_position_in_recital(text, _OUR_ALIASES)
    assert result == "unknown"


def test_alias_position_alias_not_in_recital() -> None:
    text = "This Agreement is between Alpha Corp and Beta Ltd."
    result = _alias_position_in_recital(text, _OUR_ALIASES)
    assert result == "unknown"


def test_alias_position_case_insensitive() -> None:
    text = "between acme human capital and other party."
    result = _alias_position_in_recital(text, _OUR_ALIASES)
    assert result == "first"


# ---------------------------------------------------------------------------
# _any_alias_in_text (unit)
# ---------------------------------------------------------------------------


def test_any_alias_in_text_found() -> None:
    assert _any_alias_in_text("ACME shall provide services.", _OUR_ALIASES) is True


def test_any_alias_in_text_not_found() -> None:
    assert _any_alias_in_text("Alpha Corp and Beta Ltd agree.", _OUR_ALIASES) is False


def test_any_alias_case_insensitive() -> None:
    assert _any_alias_in_text("acme human capital is the provider.", _OUR_ALIASES) is True


# ---------------------------------------------------------------------------
# detect_provenance: our-paper cases
# ---------------------------------------------------------------------------


def test_detect_our_paper_alias_first() -> None:
    result = detect_provenance(_our_paper_tree(), _config())
    assert result.provenance == "our_paper"
    assert result.confidence >= 0.80
    assert result.basis == "alias_first_party"


def test_detect_our_paper_template_similarity() -> None:
    template = _our_template_tree()
    similar = _our_paper_from_template()
    result = detect_provenance(similar, _config(), template_tree=template)
    assert result.provenance == "our_paper"
    assert result.basis == "template_similarity"
    assert result.confidence >= 0.70


def test_detect_our_paper_alias_present_fallback() -> None:
    """Alias present but no 'between...and' pattern → alias_present basis."""
    tree = _tree(
        _node("0", None, "ACME provides services under this agreement."),
        _node("1", "Terms", "The client pays monthly."),
    )
    result = detect_provenance(tree, _config())
    assert result.provenance == "our_paper"
    assert result.basis == "alias_present"


# ---------------------------------------------------------------------------
# detect_provenance: counterparty-paper cases
# ---------------------------------------------------------------------------


def test_detect_counterparty_paper_alias_second() -> None:
    result = detect_provenance(_counterparty_paper_tree(), _config())
    assert result.provenance == "counterparty_paper"
    assert result.confidence >= 0.70
    assert result.basis == "alias_second_party"


def test_detect_counterparty_paper_no_alias() -> None:
    result = detect_provenance(_counterparty_no_alias_tree(), _config())
    assert result.provenance == "counterparty_paper"
    assert result.basis == "alias_absent"


def test_detect_counterparty_paper_template_dissimilar() -> None:
    template = _our_template_tree()
    result = detect_provenance(_counterparty_template_tree(), _config(), template_tree=template)
    assert result.provenance == "counterparty_paper"
    assert result.basis == "template_similarity"


def test_detect_template_dissimilar_is_ambiguous_for_escalation() -> None:
    """Low template similarity is unreliable (extraction noise can zero it out even for
    our-paper docs), so a dissimilar verdict must be AMBIGUOUS — it escalates to the
    ProvenanceJudge instead of acting on false high confidence. Regression: the real
    corpus produced confident-but-wrong counterparty @ 0.92 from ~0.04 similarity."""
    template = _our_template_tree()
    result = detect_provenance(_counterparty_template_tree(), _config(), template_tree=template)
    assert result.basis == "template_similarity"
    assert result.is_ambiguous is True
    assert result.confidence < AMBIGUITY_THRESHOLD


# ---------------------------------------------------------------------------
# detect_provenance: no aliases configured
# ---------------------------------------------------------------------------


def test_detect_no_aliases_configured() -> None:
    # B1 fix: default is counterparty_paper (safe per OPF §2.2)
    result = detect_provenance(_our_paper_tree(), _config(aliases=[]))
    assert result.provenance == "counterparty_paper"
    assert result.confidence == 0.50
    assert result.basis == "no_aliases_configured"


# ---------------------------------------------------------------------------
# detect_provenance: edge cases
# ---------------------------------------------------------------------------


def test_detect_confidence_in_range() -> None:
    for tree in [_our_paper_tree(), _counterparty_paper_tree(), _counterparty_no_alias_tree()]:
        r = detect_provenance(tree, _config())
        assert 0.0 <= r.confidence <= 1.0


def test_detect_empty_tree_no_aliases() -> None:
    tree = ClauseTree(document_id="d", version="v1", source_file="f")
    result = detect_provenance(tree, _config())
    assert result.provenance == "counterparty_paper"
    assert result.basis == "alias_absent"


def test_ambiguity_threshold_constant() -> None:
    assert 0.0 < AMBIGUITY_THRESHOLD < 1.0


def test_high_confidence_our_paper_above_ambiguity() -> None:
    result = detect_provenance(_our_paper_tree(), _config())
    assert result.confidence >= AMBIGUITY_THRESHOLD


def test_high_confidence_counterparty_above_ambiguity() -> None:
    result = detect_provenance(_counterparty_paper_tree(), _config())
    assert result.confidence >= AMBIGUITY_THRESHOLD


# ---------------------------------------------------------------------------
# is_ambiguous property (B2 fix)
# ---------------------------------------------------------------------------


def test_is_ambiguous_true_when_below_threshold() -> None:
    """B2 fix: is_ambiguous must reflect AMBIGUITY_THRESHOLD."""
    r = ProvenanceResult(provenance="our_paper", confidence=0.60, basis="alias_present")
    assert r.is_ambiguous is True


def test_is_ambiguous_false_when_above_threshold() -> None:
    r = ProvenanceResult(provenance="our_paper", confidence=0.85, basis="alias_first_party")
    assert r.is_ambiguous is False


def test_no_aliases_result_is_ambiguous() -> None:
    """B1/B2: no-aliases-configured result must be ambiguous (confidence=0.50)."""
    result = detect_provenance(_our_paper_tree(), _config(aliases=[]))
    assert result.is_ambiguous is True


def test_alias_present_result_is_ambiguous() -> None:
    """alias_present (0.65) is below AMBIGUITY_THRESHOLD — must be flagged."""
    tree = _tree(_node("0", None, "ACME provides services under this agreement."))
    result = detect_provenance(tree, _config())
    assert result.basis == "alias_present"
    assert result.is_ambiguous is True


def test_alias_absent_result_is_ambiguous() -> None:
    """alias_absent (0.65) is below threshold."""
    result = detect_provenance(_counterparty_no_alias_tree(), _config())
    assert result.basis == "alias_absent"
    assert result.is_ambiguous is True


def test_high_confidence_not_ambiguous() -> None:
    """alias_first_party (0.85) and alias_second_party (0.75) must not be ambiguous."""
    assert not detect_provenance(_our_paper_tree(), _config()).is_ambiguous
    assert not detect_provenance(_counterparty_paper_tree(), _config()).is_ambiguous


# ---------------------------------------------------------------------------
# Regression: NB1 — alias must not match as substring of longer entity name
# ---------------------------------------------------------------------------


def test_nb1_alias_re_word_boundary() -> None:
    """'Acme' alias must NOT match inside 'Acmeseal Technologies'."""
    pattern = _alias_re("Acme")
    assert pattern.search("Acmeseal Technologies Inc.") is None
    assert pattern.search("ACME, Inc.") is not None


def test_nb1_alias_present_no_false_positive() -> None:
    """A tree that mentions 'Acmeseal' (not our alias) must not trigger alias_present."""
    tree = _tree(
        _node("0", None, "This agreement is between Acmeseal Technologies and Beta Ltd."),
        _node("1", "Services", "Acmeseal Technologies provides the services."),
    )
    result = detect_provenance(tree, _config())
    # Our alias "ACME" must not match "Acmeseal" — result should be counterparty
    assert result.provenance == "counterparty_paper"
    assert result.basis in ("alias_absent", "alias_second_party", "no_aliases_configured")


# ---------------------------------------------------------------------------
# ProvenanceJudge seam (P2.2)
# ---------------------------------------------------------------------------


class _StubJudge:
    """Scripted stub ProvenanceJudge for testing the seam plumbing."""

    def __init__(self, result: ProvenanceResult) -> None:
        self._result = result
        self.calls: list[tuple[str, str, str]] = []  # (preamble, letterhead, agreement_type)

    def judge(self, preamble: str, letterhead: str, agreement_type: str) -> ProvenanceResult:
        self.calls.append((preamble, letterhead, agreement_type))
        return self._result


def _stub_result(provenance: str = "our_paper") -> ProvenanceResult:
    return ProvenanceResult(provenance=provenance, confidence=0.90, basis="llm")


def test_judge_protocol_importable() -> None:
    """ProvenanceJudge must be importable and usable as a type."""
    assert ProvenanceJudge is not None


def test_judge_called_on_ambiguous_result() -> None:
    """When confidence < AMBIGUITY_THRESHOLD, the judge must be called and its result used."""
    # alias_present returns confidence=0.65 which is ambiguous.
    tree = _tree(_node("0", None, "ACME provides services under this agreement."))
    judge = _StubJudge(_stub_result("our_paper"))

    result = detect_provenance(tree, _config(), provenance_judge=judge)

    assert len(judge.calls) == 1, "judge must be called exactly once"
    assert result.basis == "llm"
    assert result.confidence == 0.90
    assert result.provenance == "our_paper"


def test_judge_not_called_on_high_confidence_non_judge_basis() -> None:
    """When confidence >= AMBIGUITY_THRESHOLD and basis not in PROVENANCE_JUDGE_BASES,
    the judge must NOT be called."""
    # alias_first_party → confidence=0.85, basis="alias_first_party" (not in PROVENANCE_JUDGE_BASES).
    tree = _our_paper_tree()
    judge = _StubJudge(_stub_result())

    result = detect_provenance(tree, _config(), provenance_judge=judge)

    assert len(judge.calls) == 0, "judge must not be called for high-confidence alias_first_party"
    assert result.basis == "alias_first_party"


def test_judge_called_on_name_order_basis() -> None:
    """alias_second_party is in PROVENANCE_JUDGE_BASES → judge called even though not ambiguous."""
    # alias_second_party → confidence=0.75, basis="alias_second_party", is_ambiguous=False.
    tree = _counterparty_paper_tree()
    judge_result = _stub_result("counterparty_paper")
    judge = _StubJudge(judge_result)

    result = detect_provenance(tree, _config(), provenance_judge=judge)

    assert len(judge.calls) == 1, "judge must be called for alias_second_party basis"
    assert result.basis == "llm"


def test_judge_called_on_no_aliases_configured() -> None:
    """no_aliases_configured is in PROVENANCE_JUDGE_BASES → judge called."""
    tree = _our_paper_tree()
    judge = _StubJudge(_stub_result("our_paper"))

    result = detect_provenance(tree, _config(aliases=[]), provenance_judge=judge)

    assert len(judge.calls) == 1
    assert result.basis == "llm"


def test_judge_result_replaces_heuristic() -> None:
    """The judge's ProvenanceResult must replace the original in the pipeline output."""
    tree = _tree(_node("0", None, "ACME provides services under this agreement."))
    # Deterministic result would be our_paper with confidence=0.65 and basis=alias_present.
    # Judge overrides to counterparty_paper.
    judge = _StubJudge(
        ProvenanceResult(provenance="counterparty_paper", confidence=0.88, basis="llm")
    )

    result = detect_provenance(tree, _config(), provenance_judge=judge)

    assert result.provenance == "counterparty_paper"
    assert result.confidence == 0.88
    assert result.basis == "llm"


def test_judge_payload_is_preamble_and_letterhead_only() -> None:
    """Payload sent to judge must be preamble+letterhead slice, not the full document."""
    # Build a tree with many nodes so the full doc is definitely longer than 5 lines.
    long_tree = _tree(
        _node("0", None, "By and between Alpha Corp and Beta Ltd."),
        _node("1", "Background", "Alpha Corp is a provider of services."),
        _node("2", "Scope", "Alpha Corp shall provide all deliverables as specified."),
        _node("3", "Fees", "Client shall pay all fees within sixty days."),
        _node("4", "IP", "All intellectual property shall remain with Alpha Corp."),
        _node("5", "Indemnification", "Client indemnifies Alpha Corp against all claims."),
        _node("6", "Governing Law", "This agreement is governed by the laws of New York."),
        _node("7", "Miscellaneous", "This agreement constitutes the entire understanding."),
    )
    judge = _StubJudge(_stub_result())
    detect_provenance(long_tree, _config(), provenance_judge=judge, agreement_type="MSA")

    assert len(judge.calls) == 1
    preamble, letterhead, agreement_type = judge.calls[0]

    # Preamble must not contain text from nodes beyond _PREAMBLE_MAX_LINES lines.
    # The tree has 8 text nodes, each 1 line → 8 lines total; preamble must cap at 5.
    preamble_line_count = len(preamble.splitlines())
    assert preamble_line_count <= 5, (
        f"preamble must contain at most 5 lines, got {preamble_line_count}"
    )
    # Full doc has 8 lines — preamble (5 lines) must be shorter.
    full_lines = sum(
        len([ln for ln in (n.text or "").splitlines() if ln.strip()]) for n in long_tree.all_nodes()
    )
    assert preamble_line_count < full_lines, "preamble must be a subset of the full document"
    assert agreement_type == "MSA"


def test_agreement_type_forwarded_to_judge() -> None:
    """The agreement_type label must be forwarded to the judge unmodified."""
    tree = _tree(_node("0", None, "ACME provides services."))  # alias_present → ambiguous
    judge = _StubJudge(_stub_result())

    detect_provenance(tree, _config(), provenance_judge=judge, agreement_type="Services Agreement")

    assert judge.calls[0][2] == "Services Agreement"


def test_no_judge_no_change_in_behavior() -> None:
    """Without a judge, detect_provenance returns the same results as before."""
    for tree in [_our_paper_tree(), _counterparty_paper_tree(), _counterparty_no_alias_tree()]:
        without_judge = detect_provenance(tree, _config())
        with_none_judge = detect_provenance(tree, _config(), provenance_judge=None)
        assert without_judge == with_none_judge


def test_provenance_judge_bases_constant() -> None:
    """PROVENANCE_JUDGE_BASES must contain the name-order and unknown basis values."""
    assert "alias_second_party" in PROVENANCE_JUDGE_BASES
    assert "no_aliases_configured" in PROVENANCE_JUDGE_BASES


# ---------------------------------------------------------------------------
# _extract_preamble and _extract_letterhead helpers
# ---------------------------------------------------------------------------


def test_extract_preamble_returns_first_lines() -> None:
    """_extract_preamble must return a short slice of the document body."""
    tree = _our_paper_tree()
    preamble = _extract_preamble(tree)
    assert isinstance(preamble, str)
    assert len(preamble) > 0
    # _our_paper_tree has 4 nodes with text (4 lines).  _PREAMBLE_MAX_LINES is 5,
    # so all lines fit — the result must still be a non-empty, finite string.
    line_count = len(preamble.splitlines())
    assert line_count <= 5


def test_extract_letterhead_returns_first_heading() -> None:
    """_extract_letterhead must return the first heading found, stripped."""
    tree = _tree(
        _node("0", None, "Preamble text."),
        _node("1", "  Master Services Agreement  ", "Body text."),
    )
    letterhead = _extract_letterhead(tree)
    assert letterhead == "Master Services Agreement"


def test_extract_letterhead_empty_when_no_headings() -> None:
    """_extract_letterhead must return '' when no heading nodes exist."""
    tree = _tree(_node("0", None, "Only body text."))
    assert _extract_letterhead(tree) == ""


def test_extract_preamble_empty_tree() -> None:
    """_extract_preamble on an empty tree must return ''."""
    tree = ClauseTree(document_id="d", version="v1", source_file="f")
    assert _extract_preamble(tree) == ""
