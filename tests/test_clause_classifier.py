"""Tests for the taxonomy classifier (L3, issue #15).

SECURITY NOTE: All fixtures use programmatically constructed ClauseTree
objects with synthetic text.  No real agreement files are referenced.
Party names use fictional identifiers only ("Alice Corp", "Beta Ltd").
"""

from __future__ import annotations

import pytest

from playbook_engine.clause_classifier import (
    AMBIGUITY_THRESHOLD,
    AUTO_CLASSIFY_THRESHOLD,
    ClassificationHint,
    ClassificationJudge,
    ClauseClassification,
    classify_tree,
)
from playbook_engine.clause_tree import ClauseNode, ClauseTree
from playbook_engine.taxonomy import Taxonomy, TaxonomyEntry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _entry(
    entry_id: str,
    label: str,
    status: str = "active",
) -> TaxonomyEntry:
    return TaxonomyEntry(id=entry_id, label=label, status=status, cuad_origin=None, description="")


def _taxonomy(*entries: TaxonomyEntry) -> Taxonomy:
    return Taxonomy(source="test", entries=list(entries))


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
# MockClassificationJudge — deterministic, heading-based
# ---------------------------------------------------------------------------


class MockClassificationJudge:
    """Deterministic judge that maps headings to taxonomy entries by case-insensitive
    substring match.  Unknown headings and text-only nodes get 'unclassified'."""

    def classify_batch(
        self,
        nodes: list[ClauseNode],
        taxonomy: Taxonomy,
        hints: list[ClassificationHint | None] | None = None,
    ) -> list[ClauseClassification]:
        eligible = [e for e in taxonomy.entries if e.is_classifier_eligible]
        results = []
        for node in nodes:
            text = ((node.heading or "") + " " + (node.text or "")).lower()
            best_id = None
            for entry in eligible:
                if entry.id.lower() in text or entry.label.lower() in text:
                    best_id = entry.id
                    break
            if best_id:
                results.append(
                    ClauseClassification(
                        taxonomy_id=best_id,
                        confidence=0.75,
                        basis="judge",
                    )
                )
            else:
                results.append(
                    ClauseClassification(
                        taxonomy_id=None,
                        confidence=0.0,
                        basis="unclassified",
                    )
                )
        return results


class RaisingJudge:
    """Judge that always raises — simulates LLM failure."""

    def classify_batch(
        self,
        nodes: list[ClauseNode],
        taxonomy: Taxonomy,
        hints: list[ClassificationHint | None] | None = None,
    ) -> list[ClauseClassification]:
        raise RuntimeError("LLM service unavailable")


class BadBasisJudge:
    """Judge that returns wrong basis values — for programming-error testing."""

    def classify_batch(
        self,
        nodes: list[ClauseNode],
        taxonomy: Taxonomy,
        hints: list[ClassificationHint | None] | None = None,
    ) -> list[ClauseClassification]:
        return [
            ClauseClassification(
                taxonomy_id=None,
                confidence=0.0,
                basis="exact_match",  # must NOT come from a judge
            )
            for _ in nodes
        ]


class WrongLengthJudge:
    """Judge that returns wrong number of classifications."""

    def classify_batch(
        self,
        nodes: list[ClauseNode],
        taxonomy: Taxonomy,
        hints: list[ClassificationHint | None] | None = None,
    ) -> list[ClauseClassification]:
        return []  # always returns empty


class BadTaxonomyIdJudge:
    """Judge that returns a taxonomy_id not in the taxonomy."""

    def classify_batch(
        self,
        nodes: list[ClauseNode],
        taxonomy: Taxonomy,
        hints: list[ClassificationHint | None] | None = None,
    ) -> list[ClauseClassification]:
        return [
            ClauseClassification(taxonomy_id="nonexistent_id", confidence=0.9, basis="judge")
            for _ in nodes
        ]


# ---------------------------------------------------------------------------
# Standard taxonomy fixture
# ---------------------------------------------------------------------------

_STD_TAXONOMY = _taxonomy(
    _entry("indemnification", "Indemnification"),
    _entry("governing_law", "Governing Law"),
    _entry("term", "Term"),
    _entry("termination", "Termination"),
    _entry("insurance", "Insurance"),
    _entry("confidentiality", "Confidentiality"),
    _entry("notices", "Notices"),
    _entry("custom_clause", "Student Rotation Protocols", "custom"),
    _entry("inactive_entry", "Inactive Entry", "inactive"),
)

# ---------------------------------------------------------------------------
# In-band taxonomy + heading for judge-path tests
#
# With the gate [0.70, 0.85), tests that need the judge path cannot use
# headings with zero similarity to the taxonomy (those are auto-unclassified).
# We use a controlled Jaccard scenario:
#
#   entry_label tokens = {alpha, beta, gamma, delta, epsilon, zeta, eta, iota}  (8)
#   heading tokens     = {alpha, beta, gamma, delta, epsilon, zeta, eta, theta} (8)
#   intersection = 7, union = 9 → Jaccard = 7/9 ≈ 0.778  ∈ [0.70, 0.85)
# ---------------------------------------------------------------------------

_INBAND_ENTRY_LABEL = "alpha beta gamma delta epsilon zeta eta iota"
_INBAND_HEADING = "alpha beta gamma delta epsilon zeta eta theta"
_INBAND_TAXONOMY = _taxonomy(_entry("inband_entry", _INBAND_ENTRY_LABEL))


# ---------------------------------------------------------------------------
# ClauseClassification dataclass
# ---------------------------------------------------------------------------


def test_clause_classification_fields() -> None:
    c = ClauseClassification(taxonomy_id="indemnification", confidence=0.95, basis="exact_match")
    assert c.taxonomy_id == "indemnification"
    assert c.confidence == 0.95
    assert c.basis == "exact_match"


def test_clause_classification_frozen() -> None:
    c = ClauseClassification(taxonomy_id="term", confidence=1.0, basis="exact_match")
    with pytest.raises((AttributeError, TypeError)):
        c.taxonomy_id = "something_else"  # type: ignore[misc]


def test_clause_classification_invalid_basis() -> None:
    with pytest.raises(ValueError, match="Unknown basis"):
        ClauseClassification(taxonomy_id="ind", confidence=0.9, basis="bad_basis")


def test_clause_classification_confidence_out_of_range() -> None:
    with pytest.raises(ValueError, match="confidence"):
        ClauseClassification(taxonomy_id="ind", confidence=1.5, basis="exact_match")


def test_clause_classification_unclassified_must_have_none_id() -> None:
    with pytest.raises(ValueError, match="taxonomy_id must be None"):
        ClauseClassification(taxonomy_id="ind", confidence=0.0, basis="unclassified")


def test_clause_classification_judge_error_must_have_none_id() -> None:
    with pytest.raises(ValueError, match="taxonomy_id must be None"):
        ClauseClassification(taxonomy_id="ind", confidence=0.0, basis="judge_error")


def test_clause_classification_to_dict() -> None:
    c = ClauseClassification(taxonomy_id="governing_law", confidence=0.90, basis="judge")
    d = c.to_dict()
    assert d["taxonomy_id"] == "governing_law"
    assert "confidence" in d
    assert d["basis"] == "judge"


def test_clause_classification_is_ambiguous_below_threshold() -> None:
    c = ClauseClassification(taxonomy_id="ind", confidence=0.60, basis="judge")
    assert c.is_ambiguous is True


def test_clause_classification_is_ambiguous_none_id() -> None:
    c = ClauseClassification(taxonomy_id=None, confidence=0.0, basis="unclassified")
    assert c.is_ambiguous is True


def test_clause_classification_not_ambiguous_above_threshold() -> None:
    c = ClauseClassification(taxonomy_id="ind", confidence=0.90, basis="exact_match")
    assert c.is_ambiguous is False


# ---------------------------------------------------------------------------
# ClassificationJudge protocol
# ---------------------------------------------------------------------------


def test_mock_judge_is_classification_judge_protocol() -> None:
    assert isinstance(MockClassificationJudge(), ClassificationJudge)


# ---------------------------------------------------------------------------
# classify_tree: fast path — exact heading match
# ---------------------------------------------------------------------------


def test_exact_heading_match_indemnification() -> None:
    tree = _tree(_node("1", "Indemnification", "Each party indemnifies."))
    results = classify_tree(tree, _STD_TAXONOMY, MockClassificationJudge())
    assert len(results) == 1
    assert results[0].classification.taxonomy_id == "indemnification"
    assert results[0].classification.basis == "exact_match"
    assert results[0].classification.confidence == 1.0


def test_exact_heading_match_case_insensitive() -> None:
    tree = _tree(_node("1", "GOVERNING LAW", "California law."))
    results = classify_tree(tree, _STD_TAXONOMY, MockClassificationJudge())
    assert results[0].classification.taxonomy_id == "governing_law"
    assert results[0].classification.basis == "exact_match"


def test_exact_heading_match_strips_punctuation() -> None:
    """'Indemnification.' (trailing period) must still match."""
    tree = _tree(_node("1", "Indemnification.", "Each party indemnifies."))
    results = classify_tree(tree, _STD_TAXONOMY, MockClassificationJudge())
    assert results[0].classification.taxonomy_id == "indemnification"


def test_exact_heading_match_custom_entry() -> None:
    tree = _tree(_node("1", "Student Rotation Protocols", "Protocol text."))
    results = classify_tree(tree, _STD_TAXONOMY, MockClassificationJudge())
    assert results[0].classification.taxonomy_id == "custom_clause"
    assert results[0].classification.basis == "exact_match"


# ---------------------------------------------------------------------------
# classify_tree: fast path — heading similarity
# ---------------------------------------------------------------------------


def test_heading_similarity_limitation_on_liability() -> None:
    """'Limitation on Liability' should match 'Limitation of Liability' via token overlap.

    After stop-word removal ('of', 'on' are stops), both become {'limitation','liability'}.
    """
    taxonomy = _taxonomy(
        _entry("limitation_of_liability", "Limitation of Liability"),
    )
    tree = _tree(_node("1", "Limitation on Liability", "Cap on damages."))
    results = classify_tree(tree, taxonomy, MockClassificationJudge())
    result = results[0]
    assert result.classification.taxonomy_id == "limitation_of_liability"
    assert result.classification.basis in ("exact_match", "heading_similarity")


def test_heading_similarity_confidence_in_range() -> None:
    taxonomy = _taxonomy(_entry("limitation_of_liability", "Limitation of Liability"))
    tree = _tree(_node("1", "Limitation on Liability", "Cap."))
    results = classify_tree(tree, taxonomy, MockClassificationJudge())
    assert 0.0 <= results[0].classification.confidence <= 1.0


# ---------------------------------------------------------------------------
# classify_tree: fast path — unclassified (empty node)
# ---------------------------------------------------------------------------


def test_empty_node_is_unclassified() -> None:
    tree = _tree(_node("1", None, ""))
    results = classify_tree(tree, _STD_TAXONOMY, MockClassificationJudge())
    assert results[0].classification.basis == "unclassified"
    assert results[0].classification.taxonomy_id is None


def test_empty_heading_with_text_goes_to_judge() -> None:
    """A node with no heading but with text must be delegated to the judge.

    The mock judge matches by substring; using 'indemnification' in the text
    ensures the judge returns basis='judge' (not 'unclassified'), confirming
    the fast path correctly delegated rather than handled this node itself.
    """
    tree = _tree(_node("1", None, "This indemnification provision covers all losses."))
    results = classify_tree(tree, _STD_TAXONOMY, MockClassificationJudge())
    assert results[0].classification.basis == "judge"
    assert results[0].classification.taxonomy_id == "indemnification"


# ---------------------------------------------------------------------------
# classify_tree: judge path
# ---------------------------------------------------------------------------


def test_ambiguous_heading_delegates_to_judge() -> None:
    """A heading with low similarity to any taxonomy entry goes to the judge."""
    tree = _tree(_node("1", "Miscellaneous Provisions", "General text here."))
    results = classify_tree(tree, _STD_TAXONOMY, MockClassificationJudge())
    assert results[0].classification.basis in ("judge", "unclassified")


def test_judge_receives_correct_number_of_nodes() -> None:
    """Judge is only called for nodes the fast path couldn't classify.

    Uses _INBAND_TAXONOMY so the ambiguous node has best_sim in [0.70, 0.85)
    and is forwarded to the judge rather than auto-unclassified.
    """
    call_log: list[int] = []

    class CountingJudge:
        def classify_batch(
            self,
            nodes: list[ClauseNode],
            taxonomy: Taxonomy,
            hints: list[ClassificationHint | None] | None = None,
        ) -> list[ClauseClassification]:
            call_log.append(len(nodes))
            return [
                ClauseClassification(taxonomy_id=None, confidence=0.0, basis="unclassified")
                for _ in nodes
            ]

    # Build a taxonomy that has an exact-match entry AND an in-band entry so
    # we can mix fast-path and judge-path nodes in one tree.
    mixed_taxonomy = _taxonomy(
        _entry("indemnification", "Indemnification"),  # exact-match node
        _entry("governing_law", "Governing Law"),  # exact-match node
        _entry("inband_entry", _INBAND_ENTRY_LABEL),  # in-band node
    )
    tree = _tree(
        _node("1", "Indemnification", "Indemnify."),  # fast path (exact)
        _node("2", "Governing Law", "California."),  # fast path (exact)
        _node("3", _INBAND_HEADING, "General."),  # judge path (in-band)
    )
    classify_tree(tree, mixed_taxonomy, CountingJudge())
    assert len(call_log) == 1
    assert call_log[0] == 1  # only the in-band heading went to the judge


# ---------------------------------------------------------------------------
# classify_tree: judge error path
# ---------------------------------------------------------------------------


def test_judge_raises_returns_judge_error() -> None:
    """A node in the ambiguity band whose judge raises must get basis='judge_error'."""
    tree = _tree(_node("1", _INBAND_HEADING, "Some text."))
    results = classify_tree(tree, _INBAND_TAXONOMY, RaisingJudge())
    assert results[0].classification.basis == "judge_error"
    assert results[0].classification.taxonomy_id is None
    assert results[0].classification.confidence == 0.0


def test_judge_raises_does_not_drop_node() -> None:
    """A raising judge must still produce a result for every node."""
    inband_tax = _taxonomy(
        _entry("indemnification", "Indemnification"),
        _entry("inband_entry", _INBAND_ENTRY_LABEL),
    )
    tree = _tree(
        _node("1", "Indemnification", "Indemnify."),  # fast path
        _node("2", _INBAND_HEADING, "Some text."),  # judge path (in-band)
    )
    results = classify_tree(tree, inband_tax, RaisingJudge())
    assert len(results) == 2
    assert results[1].classification.basis == "judge_error"


# ---------------------------------------------------------------------------
# classify_tree: judge contract enforcement
# ---------------------------------------------------------------------------


def test_judge_bad_basis_raises() -> None:
    """Judge returning non-judge basis raises ValueError (programming error)."""
    tree = _tree(_node("1", _INBAND_HEADING, "Text."))
    with pytest.raises(ValueError, match="unexpected basis"):
        classify_tree(tree, _INBAND_TAXONOMY, BadBasisJudge())


def test_judge_wrong_length_raises() -> None:
    """Judge returning wrong-length batch raises ValueError."""
    tree = _tree(_node("1", _INBAND_HEADING, "Text."))
    with pytest.raises(ValueError, match="classify_batch"):
        classify_tree(tree, _INBAND_TAXONOMY, WrongLengthJudge())


def test_judge_bad_taxonomy_id_raises() -> None:
    """Judge returning a taxonomy_id not in the taxonomy raises ValueError."""
    tree = _tree(_node("1", _INBAND_HEADING, "Text."))
    with pytest.raises(ValueError, match="nonexistent_id"):
        classify_tree(tree, _INBAND_TAXONOMY, BadTaxonomyIdJudge())


# ---------------------------------------------------------------------------
# Inactive entries must not be assigned
# ---------------------------------------------------------------------------


def test_inactive_entry_not_assigned_exact() -> None:
    """Exact match on an inactive entry should NOT be assigned — delegate to judge."""
    tree = _tree(_node("1", "Inactive Entry", "Some text."))
    results = classify_tree(tree, _STD_TAXONOMY, MockClassificationJudge())
    # The inactive entry label matches exactly, but it's ineligible.
    # The fast path skips it; the mock judge also can't find it (it's ineligible).
    assert results[0].classification.taxonomy_id != "inactive_entry"


# ---------------------------------------------------------------------------
# classify_tree: structural invariants
# ---------------------------------------------------------------------------


def test_returns_one_per_node() -> None:
    tree = _tree(
        _node("1", "Term", "One year."),
        _node("2", "Indemnification", "Hold harmless."),
        _node("3", "Governing Law", "California."),
    )
    results = classify_tree(tree, _STD_TAXONOMY, MockClassificationJudge())
    assert len(results) == 3


def test_result_node_references_original() -> None:
    node = _node("7", "Term", "One year.")
    tree = _tree(node)
    results = classify_tree(tree, _STD_TAXONOMY, MockClassificationJudge())
    assert results[0].node is node


def test_empty_tree_returns_empty() -> None:
    tree = ClauseTree(document_id="d", version="v1", source_file="f")
    results = classify_tree(tree, _STD_TAXONOMY, MockClassificationJudge())
    assert results == []


def test_empty_taxonomy_no_assignments() -> None:
    tree = _tree(_node("1", "Indemnification", "Text."))
    results = classify_tree(tree, Taxonomy(source="empty", entries=[]), MockClassificationJudge())
    assert results[0].classification.taxonomy_id is None


# ---------------------------------------------------------------------------
# ClassifiedClause.to_dict
# ---------------------------------------------------------------------------


def test_classified_clause_to_dict_keys() -> None:
    tree = _tree(_node("3", "Indemnification", "Hold harmless."))
    result = classify_tree(tree, _STD_TAXONOMY, MockClassificationJudge())[0]
    d = result.to_dict()
    assert "clause_path" in d
    assert "taxonomy_id" in d
    assert "confidence" in d
    assert "basis" in d


def test_classified_clause_to_dict_values() -> None:
    tree = _tree(_node("3", "Indemnification", "Hold harmless."))
    result = classify_tree(tree, _STD_TAXONOMY, MockClassificationJudge())[0]
    d = result.to_dict()
    assert d["clause_path"] == "3"
    assert d["taxonomy_id"] == "indemnification"


# ---------------------------------------------------------------------------
# Acceptance test
# ---------------------------------------------------------------------------


def test_acceptance_classifies_standard_affiliation_clauses() -> None:
    """Core acceptance: standard affiliation-agreement clauses get correct tags."""
    tree = _tree(
        _node("0", None, "This affiliation agreement is between Alice Corp and Beta Hospital."),
        _node("1", "Term", "This agreement lasts one year."),
        _node("2", "Indemnification", "Each party shall indemnify the other."),
        _node("3", "Insurance", "Each party maintains liability insurance."),
        _node("4", "Governing Law", "This agreement is governed by California law."),
        _node("5", "Termination", "Either party may terminate on thirty days notice."),
        _node("6", "Notices", "Notices shall be sent by certified mail."),
    )
    results = classify_tree(tree, _STD_TAXONOMY, MockClassificationJudge())
    assert len(results) == 7

    by_path = {r.node.clause_path: r for r in results}
    assert by_path["1"].classification.taxonomy_id == "term"
    assert by_path["2"].classification.taxonomy_id == "indemnification"
    assert by_path["3"].classification.taxonomy_id == "insurance"
    assert by_path["4"].classification.taxonomy_id == "governing_law"
    assert by_path["5"].classification.taxonomy_id == "termination"
    assert by_path["6"].classification.taxonomy_id == "notices"
    # Preamble (no heading) — mock judge or unclassified
    assert by_path["0"].classification.basis in ("judge", "unclassified", "judge_error")


def test_acceptance_high_confidence_standard_clauses() -> None:
    """Standard clauses classified by exact match must have confidence=1.0."""
    tree = _tree(
        _node("1", "Indemnification", "Hold harmless."),
        _node("2", "Governing Law", "California."),
    )
    results = classify_tree(tree, _STD_TAXONOMY, MockClassificationJudge())
    for r in results:
        assert r.classification.confidence >= AMBIGUITY_THRESHOLD


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


def test_ambiguity_threshold_in_range() -> None:
    assert 0.0 < AMBIGUITY_THRESHOLD < 1.0


def test_auto_classify_threshold_above_ambiguity() -> None:
    assert AUTO_CLASSIFY_THRESHOLD > AMBIGUITY_THRESHOLD


# ---------------------------------------------------------------------------
# Issue #50 acceptance criteria: gate band [0.70, 0.85) + hint passing
# ---------------------------------------------------------------------------


class _SpyJudge:
    """Judge that records calls and the hints it received."""

    def __init__(self, result_basis: str = "judge") -> None:
        self.called = False
        self.received_hints: list[ClassificationHint | None] | None = None
        self._result_basis = result_basis

    def classify_batch(
        self,
        nodes: list[ClauseNode],
        taxonomy: Taxonomy,
        hints: list[ClassificationHint | None] | None = None,
    ) -> list[ClauseClassification]:
        self.called = True
        self.received_hints = list(hints) if hints is not None else None
        return [
            ClauseClassification(taxonomy_id=None, confidence=0.0, basis="unclassified")
            for _ in nodes
        ]


def test_below_ambiguity_threshold_not_sent_to_judge() -> None:
    """Node with best_sim = 0.60 (< 0.70): judge NOT called; result is unclassified.

    Jaccard construction:
      heading tokens  = {kappa, lambda, mu, nu, xi}         (5 tokens)
      entry tokens    = {kappa, lambda, mu}                  (3 tokens)
      intersection    = {kappa, lambda, mu}  (3)
      union           = {kappa, lambda, mu, nu, xi}          (5)
      Jaccard         = 3/5 = 0.60  (< AMBIGUITY_THRESHOLD)
    """
    taxonomy = _taxonomy(_entry("entry_kappa", "kappa lambda mu"))
    spy = _SpyJudge()
    tree = _tree(_node("1", "kappa lambda mu nu xi", "Some text."))
    results = classify_tree(tree, taxonomy, spy)
    assert spy.called is False, "Judge must NOT be called when best_sim < AMBIGUITY_THRESHOLD"
    assert len(results) == 1
    assert results[0].classification.taxonomy_id is None
    assert results[0].classification.basis == "unclassified"


def test_in_band_node_sent_to_judge_with_hint() -> None:
    """Node with best_sim ≈ 0.778 (in [0.70, 0.85)): judge called; hint carries best_id and best_sim.

    Jaccard construction:
      heading tokens  = {alpha, beta, gamma, delta, epsilon, zeta, eta, theta}  (8 tokens)
      entry tokens    = {alpha, beta, gamma, delta, epsilon, zeta, eta, iota}   (8 tokens)
      intersection    = 7 tokens
      union           = 9 tokens
      Jaccard         = 7/9 ≈ 0.778  (in [AMBIGUITY_THRESHOLD, AUTO_CLASSIFY_THRESHOLD))
    """
    taxonomy = _taxonomy(_entry("entry_alpha", "alpha beta gamma delta epsilon zeta eta iota"))
    spy = _SpyJudge()
    tree = _tree(_node("1", "alpha beta gamma delta epsilon zeta eta theta", "Some text."))
    classify_tree(tree, taxonomy, spy)
    assert spy.called is True, "Judge MUST be called when best_sim is in the ambiguity band"
    assert spy.received_hints is not None, "Hints must be passed to judge for in-band nodes"
    assert len(spy.received_hints) == 1
    hint = spy.received_hints[0]
    assert hint is not None
    assert hint.best_id == "entry_alpha"
    assert abs(hint.best_sim - (7 / 9)) < 1e-9, f"expected 7/9 ≈ 0.778, got {hint.best_sim}"


def test_above_auto_classify_threshold_not_sent_to_judge() -> None:
    """Node with best_sim = 0.90 (>= 0.85): judge NOT called; result is auto-classified.

    'Limitation on Liability' vs 'Limitation of Liability': after removing stop
    words ('on', 'of'), both yield {'limitation', 'liability'} → Jaccard = 1.0 >= 0.85.
    """
    taxonomy = _taxonomy(_entry("limitation_of_liability", "Limitation of Liability"))
    spy = _SpyJudge()
    tree = _tree(_node("1", "Limitation on Liability", "Cap on damages."))
    results = classify_tree(tree, taxonomy, spy)
    assert spy.called is False, "Judge must NOT be called when best_sim >= AUTO_CLASSIFY_THRESHOLD"
    assert results[0].classification.taxonomy_id == "limitation_of_liability"
    assert results[0].classification.basis in ("exact_match", "heading_similarity")
    assert results[0].classification.confidence >= AUTO_CLASSIFY_THRESHOLD
