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
    CONTENT_ASSIGN_THRESHOLD,
    CONTENT_CONFIDENCE_CAP,
    CONTENT_MARGIN_RATIO,
    INHERITED_CONFIDENCE_CAP,
    ClassificationHint,
    ClassificationJudge,
    ClassifiedClause,
    ClauseClassification,
    _build_label_index,
    _build_label_tokens,
    _content_tokens,
    _fast_classify,
    assign_by_content,
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


# ---------------------------------------------------------------------------
# Issue #66: precompute taxonomy-label token sets in the classify_tree fast path
#
# _fast_classify's Jaccard loop used to call _tokens(entry.label) once per
# (node, entry) pair even though entry labels never change within a run.
# classify_tree now precomputes label_tokens once and threads it through.
# ---------------------------------------------------------------------------


def test_fast_classify_direct_call_without_label_tokens_still_works() -> None:
    """Direct callers that invoke _fast_classify without the new label_tokens
    kwarg (e.g. pre-#66 test code) must still get correct results via the
    lazy-compute fallback."""
    taxonomy = _taxonomy(_entry("limitation_of_liability", "Limitation of Liability"))
    eligible = [e for e in taxonomy.entries if e.is_classifier_eligible]
    label_index = _build_label_index(eligible)
    node = _node("1", "Limitation on Liability", "Cap on damages.")

    cls, hint = _fast_classify(node, eligible, label_index)

    assert cls is not None
    assert cls.taxonomy_id == "limitation_of_liability"
    assert cls.basis == "heading_similarity"
    assert hint is None


def test_fast_classify_with_precomputed_label_tokens_matches_lazy_path() -> None:
    """Passing precomputed label_tokens must produce byte-identical results to
    the lazy fallback — the precompute is purely a performance change."""
    taxonomy = _taxonomy(_entry("limitation_of_liability", "Limitation of Liability"))
    eligible = [e for e in taxonomy.entries if e.is_classifier_eligible]
    label_index = _build_label_index(eligible)
    label_tokens = _build_label_tokens(eligible)
    node = _node("1", "Limitation on Liability", "Cap on damages.")

    cls_lazy, hint_lazy = _fast_classify(node, eligible, label_index)
    cls_precomputed, hint_precomputed = _fast_classify(
        node, eligible, label_index, label_tokens=label_tokens
    )

    assert cls_precomputed == cls_lazy
    assert hint_precomputed == hint_lazy


def test_label_tokens_precomputed_once_per_classify_tree_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression for issue #66: entry.label must be tokenized once per
    classify_tree call, not once per (node, entry) pair.

    Before the fix, _fast_classify called _tokens(entry.label) inside its
    per-node Jaccard loop, so the number of calls scaled with
    nodes x entries. After the fix, classify_tree precomputes label_tokens
    once via _build_label_tokens and threads it through every _fast_classify
    call, so entry-label tokenization no longer scales with node count.
    """
    import playbook_engine.clause_classifier as cc

    real_tokens = cc._tokens
    call_count = 0

    def counting_tokens(text: str) -> frozenset[str]:
        nonlocal call_count
        call_count += 1
        return real_tokens(text)

    monkeypatch.setattr(cc, "_tokens", counting_tokens)

    # Disjoint vocabularies (entryword* vs headword*, distinct index per
    # entry/node) guarantee zero token overlap for every (node, entry) pair,
    # so every node reaches the Jaccard loop and lands on "unclassified"
    # (best_sim = 0.0 < AMBIGUITY_THRESHOLD) without needing the judge.
    n_entries = 5
    n_nodes = 4
    taxonomy = _taxonomy(
        *[_entry(f"tax_{i}", f"entryword{i}a entryword{i}b") for i in range(n_entries)]
    )
    tree = _tree(*[_node(str(i), f"headword{i}a headword{i}b", "text") for i in range(n_nodes)])

    results = classify_tree(tree, taxonomy, MockClassificationJudge())

    # Sanity: confirm every node actually reached (and exited) the Jaccard
    # loop rather than short-circuiting before it.
    assert len(results) == n_nodes
    for r in results:
        assert r.classification.basis == "unclassified"
        assert r.classification.taxonomy_id is None

    # Entry labels tokenized once total (by _build_label_tokens, called once
    # from classify_tree), plus once per node heading (h_tokens) — NOT once
    # per (node, entry) pair. Pre-fix this would be n_nodes + n_nodes *
    # n_entries = 4 + 20 = 24; post-fix it is n_entries + n_nodes = 9.
    assert call_count == n_entries + n_nodes, (
        f"expected {n_entries + n_nodes} _tokens calls (label tokenized once "
        f"each + heading tokenized once per node), got {call_count} — "
        "entry.label is likely being re-tokenized per node again"
    )


# ---------------------------------------------------------------------------
# Heading-less child inheritance (issue #222)
# ---------------------------------------------------------------------------

_IND_BODY = (
    "Alice Corp shall indemnify Beta Ltd against:\n"
    "(a) third party claims arising from breach; and\n"
    "(b) losses caused by gross negligence, including:\n"
    "(i) bodily injury; and\n"
    "(ii) property damage."
)


def _segmented(heading: str | None, body: str = _IND_BODY) -> ClauseTree:
    """A tree whose heading-less children come from the real segmenter (the
    production producer of ``heading=None`` (a)/(b) and (i)/(ii) nodes)."""
    from playbook_engine.segmenter import segment

    return segment(
        _tree(ClauseNode(clause_path="7", heading=heading, text=body, char_span=(0, len(body))))
    )


def _default_judge() -> ClassificationJudge:
    """The pipeline's default (no-judge) classification judge."""
    from playbook_engine.pipeline import _NullClassificationJudge

    return _NullClassificationJudge()


def test_heading_less_child_inherits_parent_classification() -> None:
    tree = _segmented("Indemnification")
    children = [n for n in tree.all_nodes() if n.clause_path != "7"]
    assert children and all(n.heading is None for n in children)

    tax = _taxonomy(_entry("indemnification", "Indemnification"))
    result = {
        cc.node.clause_path: cc.classification for cc in classify_tree(tree, tax, _default_judge())
    }

    assert result["7"].taxonomy_id == "indemnification"
    assert result["7"].basis == "exact_match"
    for path in ("7.a", "7.b", "7.b.i", "7.b.ii"):
        cls = result[path]
        assert cls.taxonomy_id == "indemnification", path
        assert cls.basis == "inherited", path
        assert cls.confidence == pytest.approx(min(1.0, INHERITED_CONFIDENCE_CAP)), path


def test_inherited_confidence_is_capped_by_a_weaker_parent() -> None:
    """min(parent_conf, 0.6): a parent classified below the cap passes its own
    (lower) confidence down."""

    class _LowConfidenceJudge:
        def classify_batch(self, nodes, taxonomy, hints=None):  # type: ignore[no-untyped-def]
            return [
                ClauseClassification(taxonomy_id="indemnification", confidence=0.4, basis="judge")
                if node.heading
                else ClauseClassification(taxonomy_id=None, confidence=0.0, basis="unclassified")
                for node in nodes
            ]

    # Heading Jaccard 0.75 against the label: in the judge band.
    tree = _segmented("Mutual Indemnification Obligations Generally")
    tax = _taxonomy(_entry("indemnification", "Mutual Indemnification Obligations"))
    result = {
        cc.node.clause_path: cc.classification
        for cc in classify_tree(tree, tax, _LowConfidenceJudge())
    }
    assert result["7"].confidence == pytest.approx(0.4)
    assert result["7.a"].basis == "inherited"
    assert result["7.a"].confidence == pytest.approx(0.4)


def test_child_of_unclassified_parent_stays_unclassified() -> None:
    tree = _segmented("Miscellaneous Matters Of Note")
    tax = _taxonomy(_entry("indemnification", "Indemnification"))
    for cc in classify_tree(tree, tax, _default_judge()):
        assert cc.classification.taxonomy_id is None
        assert cc.classification.basis != "inherited"


def test_judge_specific_fit_for_child_is_kept() -> None:
    """A judge that classifies the child itself wins over inheritance; a
    judge_error is never masked by it."""

    class _ChildJudge:
        def classify_batch(self, nodes, taxonomy, hints=None):  # type: ignore[no-untyped-def]
            out = []
            for node in nodes:
                if "bodily injury" in node.text:
                    out.append(
                        ClauseClassification(taxonomy_id="insurance", confidence=0.9, basis="judge")
                    )
                elif "property damage" in node.text:
                    out.append(
                        ClauseClassification(taxonomy_id=None, confidence=0.0, basis="judge_error")
                    )
                else:
                    out.append(
                        ClauseClassification(taxonomy_id=None, confidence=0.0, basis="unclassified")
                    )
            return out

    tree = _segmented("Indemnification")
    tax = _taxonomy(_entry("indemnification", "Indemnification"), _entry("insurance", "Insurance"))
    result = {
        cc.node.clause_path: cc.classification for cc in classify_tree(tree, tax, _ChildJudge())
    }
    assert result["7.b.i"].taxonomy_id == "insurance"
    assert result["7.b.i"].basis == "judge"
    assert result["7.b.ii"].basis == "judge_error"
    assert result["7.b.ii"].taxonomy_id is None
    assert result["7.a"].basis == "inherited"


def test_top_level_heading_less_node_does_not_inherit() -> None:
    """Only a child inherits: a top-level text-only node has no parent."""
    tree = _tree(_node("1", None, "Some text-only paragraph."))
    tax = _taxonomy(_entry("indemnification", "Indemnification"))
    (cc,) = classify_tree(tree, tax, _default_judge())
    assert cc.classification.taxonomy_id is None
    assert cc.classification.basis == "unclassified"


def test_inherited_basis_is_valid() -> None:
    cls = ClauseClassification(taxonomy_id="indemnification", confidence=0.6, basis="inherited")
    assert cls.is_ambiguous  # 0.6 < AMBIGUITY_THRESHOLD


def test_headed_child_with_no_taxonomy_match_does_not_inherit() -> None:
    """A subsection with its own heading (e.g. an ingested "4.1 ..." child) that
    matches nothing in the taxonomy stays unclassified: only a heading-less
    child inherits its parent's taxonomy_id."""
    body = "Each party shall give written notice of any claim within thirty days."
    child = ClauseNode(
        clause_path="7.1",
        heading="Notice Procedure",
        text=body,
        char_span=(20, 20 + len(body)),
    )
    parent = ClauseNode(
        clause_path="7",
        heading="Indemnification",
        text="",
        char_span=(0, 20 + len(body)),
        children=[child],
    )
    tax = _taxonomy(_entry("indemnification", "Indemnification"))
    result = {
        cc.node.clause_path: cc.classification
        for cc in classify_tree(_tree(parent), tax, _default_judge())
    }

    assert result["7"].taxonomy_id == "indemnification"
    assert result["7"].basis == "exact_match"
    assert result["7.1"].taxonomy_id is None
    assert result["7.1"].basis == "unclassified"


# ---------------------------------------------------------------------------
# Content-similarity fallback (issue #235)
# ---------------------------------------------------------------------------
#
# Exemplars and node texts below are built from invented single-word tokens
# (alpha, beta, ...), so each pair's token Jaccard is known exactly:
#   |A ∩ B| / |A ∪ B| over the non-stop-word tokens.

_EXEMPLARS = {
    "survival": "alpha beta gamma delta",
    "venue": "kappa lambda mu nu",
}


def _content_tax() -> Taxonomy:
    return _taxonomy(
        _entry("survival", "Survival Period"),
        _entry("venue", "Dispute Resolution Venue"),
    )


def _classify_unheaded(
    text: str,
    exemplars: dict[str, str] | None = _EXEMPLARS,
    taxonomy: Taxonomy | None = None,
    heading: str = "Zebra Provisions",
) -> ClauseClassification:
    """Classify one top-level node whose heading matches no taxonomy label."""
    tree = _tree(_node("1", heading, text))
    result = classify_tree(
        tree,
        taxonomy or _content_tax(),
        _default_judge(),
        content_exemplars=exemplars,
    )
    return result[0].classification


def test_content_similarity_constants_are_conservative() -> None:
    assert CONTENT_ASSIGN_THRESHOLD >= 0.25
    assert CONTENT_MARGIN_RATIO >= 2.0
    assert CONTENT_CONFIDENCE_CAP < AMBIGUITY_THRESHOLD


def test_content_tokens_drop_boilerplate_and_keep_numerals() -> None:
    tokens = _content_tokens("Within 30 days, the Party shall notify the Other Party.")
    assert tokens == frozenset({"within", "30", "days", "notify"})


def test_content_similarity_assigns_best_exemplar() -> None:
    # Jaccard vs survival: 3/5 = 0.6; vs venue: 0.
    cls = _classify_unheaded("alpha beta gamma epsilon")
    assert cls.taxonomy_id == "survival"
    assert cls.basis == "content_similarity"
    # The 0.6 score is pulled down to the cap.
    assert cls.confidence == CONTENT_CONFIDENCE_CAP


def test_content_similarity_margin_rule_rejects_a_near_tie() -> None:
    # Both exemplars score 0.6 (3 shared of 5): the best is no better than the
    # runner-up, so nothing is assigned. A guess between two types plants a
    # false precedent.
    exemplars = {"survival": "alpha beta gamma delta", "venue": "alpha beta gamma epsilon"}
    cls = _classify_unheaded("alpha beta gamma zeta", exemplars)
    assert cls.taxonomy_id is None
    assert cls.basis == "unclassified"


def test_content_similarity_margin_rule_rejects_a_lead_under_the_ratio() -> None:
    # survival 4/5 = 0.8 vs venue 3/6 = 0.5 (alpha, beta, gamma shared of six
    # distinct tokens): best clears the threshold but
    # leads the runner-up by only 1.6x (< CONTENT_MARGIN_RATIO).
    exemplars = {"survival": "alpha beta gamma delta", "venue": "alpha beta gamma zeta"}
    cls = _classify_unheaded("alpha beta gamma delta epsilon", exemplars)
    assert cls.taxonomy_id is None
    assert cls.basis == "unclassified"
    # ...and the same best score IS assigned once the runner-up falls far enough.
    far = {"survival": "alpha beta gamma delta", "venue": "kappa lambda mu nu"}
    assert _classify_unheaded("alpha beta gamma delta epsilon", far).taxonomy_id == "survival"


def test_content_similarity_threshold_rejects_a_weak_best() -> None:
    # Jaccard vs survival: 1/10 = 0.1; the runner-up is 0, so the margin rule
    # passes trivially and only the threshold stands in the way.
    cls = _classify_unheaded("alpha one two three four five six")
    assert cls.taxonomy_id is None
    assert cls.basis == "unclassified"


def test_content_similarity_confidence_stays_below_ambiguity_threshold() -> None:
    # A verbatim copy scores 1.0, which the cap pulls below AMBIGUITY_THRESHOLD
    # so the assignment never reads as a verified judge verdict.
    cls = _classify_unheaded("alpha beta gamma delta")
    assert cls.basis == "content_similarity"
    assert cls.confidence == CONTENT_CONFIDENCE_CAP
    assert cls.confidence < AMBIGUITY_THRESHOLD
    assert cls.is_ambiguous

    # Below the cap the confidence is the score itself, and every score the
    # threshold admits stays under AMBIGUITY_THRESHOLD.
    low = _classify_unheaded("alpha beta one two three four", {"survival": "alpha beta gamma"})
    assert low.basis == "content_similarity"  # 2/7 = 0.2857 >= threshold
    assert CONTENT_ASSIGN_THRESHOLD <= low.confidence < AMBIGUITY_THRESHOLD
    assert low.confidence == pytest.approx(2 / 7)


def test_content_similarity_is_a_noop_without_exemplars() -> None:
    for exemplars in (None, {}):
        cls = _classify_unheaded("alpha beta gamma delta", exemplars)
        assert cls.taxonomy_id is None
        assert cls.basis == "unclassified"


def test_content_similarity_never_overrides_a_heading_or_judge_classification() -> None:
    # The heading says venue; the text resembles the survival exemplar. The
    # heading path wins: content similarity only fills what is left empty.
    cls = _classify_unheaded("alpha beta gamma delta", heading="Dispute Resolution Venue")
    assert cls.taxonomy_id == "venue"
    assert cls.basis == "exact_match"


def test_inheritance_wins_over_content_similarity_for_a_heading_less_child() -> None:
    child = ClauseNode(
        clause_path="7.a",
        heading=None,
        text="alpha beta gamma delta",  # resembles the survival exemplar exactly
        char_span=(20, 40),
    )
    parent = ClauseNode(
        clause_path="7",
        heading="Dispute Resolution Venue",
        text="",
        char_span=(0, 40),
        children=[child],
    )
    result = {
        cc.node.clause_path: cc.classification
        for cc in classify_tree(
            _tree(parent), _content_tax(), _default_judge(), content_exemplars=_EXEMPLARS
        )
    }
    assert result["7"].taxonomy_id == "venue"
    assert result["7.a"].taxonomy_id == "venue"
    assert result["7.a"].basis == "inherited"


def test_content_similarity_fills_a_heading_less_child_inheritance_left_empty() -> None:
    """A child of an UNclassified parent inherits nothing, so content similarity
    may still place it."""
    child = ClauseNode(
        clause_path="7.a", heading=None, text="alpha beta gamma delta", char_span=(20, 40)
    )
    parent = ClauseNode(
        clause_path="7", heading="Zebra Provisions", text="", char_span=(0, 40), children=[child]
    )
    result = {
        cc.node.clause_path: cc.classification
        for cc in classify_tree(
            _tree(parent), _content_tax(), _default_judge(), content_exemplars=_EXEMPLARS
        )
    }
    assert result["7"].basis == "unclassified"
    assert result["7.a"].taxonomy_id == "survival"
    assert result["7.a"].basis == "content_similarity"


def test_content_similarity_skips_nodes_without_body_text() -> None:
    cls = _classify_unheaded("", heading="alpha beta gamma delta")
    assert cls.taxonomy_id is None
    assert cls.basis == "unclassified"


def test_content_similarity_ignores_ineligible_taxonomy_entries() -> None:
    """An exemplar for an inactive entry is never a candidate (OPF §5)."""
    taxonomy = _taxonomy(
        _entry("survival", "Survival Period", status="inactive"),
        _entry("venue", "Dispute Resolution Venue"),
    )
    cls = _classify_unheaded("alpha beta gamma delta", taxonomy=taxonomy)
    assert cls.taxonomy_id is None
    assert cls.basis == "unclassified"


def test_content_similarity_only_touches_unclassified_nodes() -> None:
    """judge_error / needs_review / judge / llm_segmenter results pass through
    assign_by_content unchanged, even over a text that matches an exemplar."""
    text = "alpha beta gamma delta"
    keep = [
        ClauseClassification(taxonomy_id=None, confidence=0.0, basis="judge_error"),
        ClauseClassification(taxonomy_id=None, confidence=0.0, basis="needs_review"),
        ClauseClassification(taxonomy_id="venue", confidence=0.9, basis="judge"),
        ClauseClassification(taxonomy_id="venue", confidence=0.6, basis="llm_segmenter"),
    ]
    classified = [
        ClassifiedClause(node=_node(str(i), None, text), classification=cls)
        for i, cls in enumerate(keep, start=1)
    ]
    out = assign_by_content(classified, _EXEMPLARS, eligible_ids={"survival", "venue"})
    assert [cc.classification for cc in out] == keep


def test_content_similarity_basis_is_valid() -> None:
    cls = ClauseClassification(taxonomy_id="survival", confidence=0.4, basis="content_similarity")
    assert cls.is_ambiguous
    assert cls.to_dict()["basis"] == "content_similarity"
