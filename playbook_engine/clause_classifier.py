"""Taxonomy classifier — L3 pipeline stage.

Tags each ``ClauseNode`` in a ``ClauseTree`` with an *active* or *custom*
taxonomy entry (OPF §5: "a compiler MUST only classify clauses into active
or custom entries").

Fast path (deterministic):
  If the clause heading matches an eligible taxonomy entry exactly (case-
  insensitive) or with token-Jaccard similarity ≥ ``AUTO_CLASSIFY_THRESHOLD``,
  the assignment is made without an LLM call.  Most standard-heading clauses
  (e.g. "Indemnification", "Governing Law") qualify.

Slow path (LLM — injected ``ClassificationJudge``):
  Clauses whose headings fall in the ambiguity band
  ``[AMBIGUITY_THRESHOLD, AUTO_CLASSIFY_THRESHOLD)`` = ``[0.70, 0.85)`` are
  batched and passed to the injected judge along with a ``ClassificationHint``
  carrying the fast-path's best match.  Text-only (heading-less) nodes are
  also sent to the judge (without a hint).  In tests a deterministic
  ``MockClassificationJudge`` is substituted.

If the judge raises (LLM timeout, parse error, refusal), affected clauses are
returned with ``basis="judge_error"`` and ``taxonomy_id=None`` — never silently
dropped.

``ClauseClassification.basis`` values:
  ``"exact_match"``        — heading matched a taxonomy label exactly.
  ``"heading_similarity"`` — Jaccard ≥ ``AUTO_CLASSIFY_THRESHOLD``.
  ``"judge"``              — delegated to the injected judge.
  ``"judge_error"``        — judge raised; node recorded as unclassified.
  ``"unclassified"``       — no heading/text, or Jaccard < ``AMBIGUITY_THRESHOLD``
                             (below gate); cannot classify without the judge.
  ``"llm_segmenter"``      — assigned by the LLM segmenter's single combined
                             segment+classify pass (see
                             ``pipeline._classified_from_taxonomy_by_path``),
                             not by a dedicated, separately-verified
                             ``ClassificationJudge`` call. Deliberately
                             distinct from ``"judge"`` and always paired with
                             a below-``AMBIGUITY_THRESHOLD`` confidence so
                             these assignments are never mistaken for a
                             verified judge verdict downstream (issue #86).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from playbook_engine.clause_tree import ClauseNode, ClauseTree
from playbook_engine.taxonomy import Taxonomy, TaxonomyEntry

# ---------------------------------------------------------------------------
# Hint datatype
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClassificationHint:
    """Structured hint passed to the judge for in-band ambiguous nodes.

    Attributes:
        best_id:  The nearest taxonomy entry id found by the fast path, or
                  ``None`` if no eligible entry was found.
        best_sim: The token-Jaccard similarity score for *best_id* (0.0–1.0).
    """

    best_id: str | None
    best_sim: float


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AMBIGUITY_THRESHOLD: float = 0.70
"""Confidence below which a classification is considered uncertain."""

AUTO_CLASSIFY_THRESHOLD: float = 0.85
"""Minimum token-Jaccard similarity between a clause heading and a taxonomy
entry label to assign automatically, without an LLM call."""

_BASIS_VALUES = frozenset(
    {
        "exact_match",
        "heading_similarity",
        "judge",
        "judge_error",
        "needs_review",
        "unclassified",
        "llm_segmenter",
    }
)

_STOP_WORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "in",
        "is",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "with",
    }
)

# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClauseClassification:
    """The taxonomy classification for one clause node.

    Attributes:
        taxonomy_id:  Entry id from the taxonomy, or ``None`` if unclassifiable.
        confidence:   Float in [0, 1].  Values below ``AMBIGUITY_THRESHOLD``
                      flag uncertain assignments for human review.
        basis:        How the classification was reached.
    """

    taxonomy_id: str | None
    confidence: float
    basis: str

    def __post_init__(self) -> None:
        if self.basis not in _BASIS_VALUES:
            raise ValueError(
                f"Unknown basis: {self.basis!r}. Must be one of {sorted(_BASIS_VALUES)}"
            )
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")
        if (
            self.basis in ("unclassified", "judge_error", "needs_review")
            and self.taxonomy_id is not None
        ):
            raise ValueError(
                f"taxonomy_id must be None when basis={self.basis!r}; got {self.taxonomy_id!r}"
            )

    @property
    def is_ambiguous(self) -> bool:
        """True when confidence < AMBIGUITY_THRESHOLD or taxonomy_id is None."""
        return self.taxonomy_id is None or self.confidence < AMBIGUITY_THRESHOLD

    def to_dict(self) -> dict[str, Any]:
        return {
            "taxonomy_id": self.taxonomy_id,
            "confidence": round(self.confidence, 6),
            "basis": self.basis,
        }


@dataclass(frozen=True)
class ClassifiedClause:
    """One ``ClauseNode`` paired with its taxonomy classification."""

    node: ClauseNode
    classification: ClauseClassification

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "clause_path": self.node.clause_path,
            "heading": self.node.heading,
        }
        d.update(self.classification.to_dict())
        return d


# ---------------------------------------------------------------------------
# Judge protocol (LLM integration point)
# ---------------------------------------------------------------------------


@runtime_checkable
class ClassificationJudge(Protocol):
    """Protocol for LLM-based clause classification.

    Implementations receive a batch of nodes that the deterministic fast path
    could not confidently classify, plus the eligible taxonomy entries, and
    return one ``ClauseClassification`` per node **in the same order**.

    Contract:
    - Return exactly ``len(nodes)`` classifications.
    - Each classification must have ``basis`` in
      ``{"judge", "judge_error", "unclassified"}``; any other value raises
      ``ValueError`` inside ``classify_tree()``.
    - Do NOT return ``basis`` values reserved for the fast path
      (``"exact_match"``, ``"heading_similarity"``).
    """

    def classify_batch(
        self,
        nodes: list[ClauseNode],
        taxonomy: Taxonomy,
        hints: list[ClassificationHint | None] | None = None,
    ) -> list[ClauseClassification]:
        """Classify each node in *nodes* against *taxonomy*.

        Args:
            nodes:    Clause nodes requiring LLM judgment.
            taxonomy: The full taxonomy (judge may use any entry for context,
                      but MUST only return active/custom ``taxonomy_id`` values).
            hints:    Optional fast-path hints, one per node (same order).
                      Each hint carries ``best_id`` and ``best_sim`` from the
                      nearest fast-path match so the judge can verify rather
                      than re-derive.  Individual elements may be ``None`` for
                      nodes where Jaccard similarity is not applicable
                      (e.g. text-only nodes).  The outer list may also be
                      ``None`` when no in-band nodes are present.

        Returns:
            One ``ClauseClassification(basis="judge")`` per node, same order.
        """
        ...  # pragma: no cover


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def classify_tree(
    tree: ClauseTree,
    taxonomy: Taxonomy,
    judge: ClassificationJudge,
    *,
    ambiguity_threshold: float = AMBIGUITY_THRESHOLD,
    auto_classify_threshold: float = AUTO_CLASSIFY_THRESHOLD,
) -> list[ClassifiedClause]:
    """Classify every node in *tree* against *taxonomy*.

    Nodes classified by the fast path (exact match, Jaccard ≥
    ``auto_classify_threshold``, or Jaccard < ``ambiguity_threshold``) never
    touch the LLM.  Only nodes whose Jaccard similarity falls in the
    ``[ambiguity_threshold, auto_classify_threshold)`` band, plus heading-less
    text-only nodes, are batched and sent to the judge.  In-band nodes receive
    a ``ClassificationHint`` so the judge can verify rather than re-derive.

    If the judge raises, all nodes in that batch receive
    ``basis="judge_error"`` with ``taxonomy_id=None`` and
    ``confidence=0.0`` — they are never silently discarded.

    Args:
        tree:                    Segmented clause tree to classify.
        taxonomy:                Curated taxonomy; only active and custom
                                 entries are eligible.
        judge:                   Injected ``ClassificationJudge`` for
                                 ambiguous clauses.
        ambiguity_threshold:     Producer-configurable (issue #168,
                                 config.classification.ambiguity_threshold) —
                                 below this Jaccard similarity a clause is
                                 auto-unclassified rather than escalated to
                                 the judge. Defaults to ``AMBIGUITY_THRESHOLD``.
        auto_classify_threshold: Producer-configurable (issue #168,
                                 config.classification.auto_classify_threshold)
                                 — at or above this Jaccard similarity a
                                 clause is auto-classified without the judge.
                                 Defaults to ``AUTO_CLASSIFY_THRESHOLD``.

    Returns:
        One ``ClassifiedClause`` per node in ``tree.all_nodes()`` order.

    Raises:
        ValueError: if the judge returns a wrong-length batch, or if any
                    returned classification has ``basis != "judge"``.
    """
    eligible = _eligible_entries(taxonomy)
    eligible_by_id = {e.id: e for e in eligible}
    label_index = _build_label_index(eligible)

    nodes = list(tree.all_nodes())
    results: list[ClassifiedClause | None] = [None] * len(nodes)

    # Indices of nodes that need judge evaluation, and the corresponding hints.
    judge_indices: list[int] = []
    judge_hints: list[ClassificationHint | None] = []

    for i, node in enumerate(nodes):
        cls, hint = _fast_classify(
            node,
            eligible,
            label_index,
            ambiguity_threshold=ambiguity_threshold,
            auto_classify_threshold=auto_classify_threshold,
        )
        if cls is not None:
            results[i] = ClassifiedClause(node=node, classification=cls)
        else:
            judge_indices.append(i)
            judge_hints.append(hint)

    if judge_indices:
        batch_nodes = [nodes[i] for i in judge_indices]
        # Pass hints only when at least one is non-None; otherwise pass None
        # for backward compatibility with judge implementations that pre-date
        # this parameter.
        hints_arg: list[ClassificationHint | None] | None = (
            judge_hints if any(h is not None for h in judge_hints) else None
        )
        try:
            judge_results = judge.classify_batch(batch_nodes, taxonomy, hints=hints_arg)
        except Exception as exc:  # noqa: BLE001
            judge_results = [
                ClauseClassification(
                    taxonomy_id=None,
                    confidence=0.0,
                    basis="judge_error",
                )
            ] * len(batch_nodes)
            _ = exc  # consumed; rationale is encoded in the basis field

        if len(judge_results) != len(batch_nodes):
            raise ValueError(
                f"ClassificationJudge.classify_batch() returned "
                f"{len(judge_results)} results for {len(batch_nodes)} nodes."
            )

        for idx, classification in zip(judge_indices, judge_results, strict=True):
            if classification.basis not in ("judge", "judge_error", "needs_review", "unclassified"):
                raise ValueError(
                    f"ClassificationJudge returned unexpected basis={classification.basis!r} "
                    f"for node {nodes[idx].clause_path!r}; "
                    "must be 'judge', 'judge_error', 'needs_review', or 'unclassified'."
                )
            # Validate taxonomy_id against eligible entries.
            if (
                classification.taxonomy_id is not None
                and classification.taxonomy_id not in eligible_by_id
            ):
                raise ValueError(
                    f"Judge returned taxonomy_id={classification.taxonomy_id!r} which "
                    "is not an active/custom entry in the supplied taxonomy."
                )
            results[idx] = ClassifiedClause(node=nodes[idx], classification=classification)

    return [r for r in results if r is not None]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _eligible_entries(taxonomy: Taxonomy) -> list[TaxonomyEntry]:
    """Return only active and custom entries (OPF §5)."""
    return [e for e in taxonomy.entries if e.is_classifier_eligible]


def _build_label_index(
    entries: list[TaxonomyEntry],
) -> dict[str, str]:
    """Build {normalized_label: entry_id} for fast exact-match lookup."""
    return {_normalize(e.label): e.id for e in entries}


def _normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    s = text.lower()
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _tokens(text: str) -> frozenset[str]:
    """Return meaningful tokens (stop words excluded)."""
    return frozenset(w for w in _normalize(text).split() if w not in _STOP_WORDS)


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _fast_classify(
    node: ClauseNode,
    eligible: list[TaxonomyEntry],
    label_index: dict[str, str],
    *,
    ambiguity_threshold: float = AMBIGUITY_THRESHOLD,
    auto_classify_threshold: float = AUTO_CLASSIFY_THRESHOLD,
) -> tuple[ClauseClassification | None, ClassificationHint | None]:
    """Attempt deterministic classification.

    Returns a 2-tuple ``(classification, hint)``:

    - ``(ClauseClassification, None)`` — classified by fast path; no judge needed.
    - ``(None, ClassificationHint)``  — in ambiguity band; queue for judge with hint.
    - ``(None, None)``                — no heading tokens or text-only; queue for judge
                                        without a hint (Jaccard not applicable).
    """
    heading = (node.heading or "").strip()
    text = (node.text or "").strip()

    if not heading and not text:
        return (
            ClauseClassification(
                taxonomy_id=None,
                confidence=0.0,
                basis="unclassified",
            ),
            None,
        )

    if not heading:
        return (None, None)  # text-only node → needs judge, no Jaccard hint

    norm = _normalize(heading)

    # Exact heading match (case-insensitive).
    if norm in label_index:
        return (
            ClauseClassification(
                taxonomy_id=label_index[norm],
                confidence=1.0,
                basis="exact_match",
            ),
            None,
        )

    # Jaccard similarity match.
    h_tokens = _tokens(heading)
    if not h_tokens:
        return (None, None)

    best_id: str | None = None
    best_sim: float = 0.0
    for entry in eligible:
        sim = _jaccard(h_tokens, _tokens(entry.label))
        if sim > best_sim:
            best_sim = sim
            best_id = entry.id

    if best_id is not None and best_sim >= auto_classify_threshold:
        return (
            ClauseClassification(
                taxonomy_id=best_id,
                confidence=best_sim,
                basis="heading_similarity",
            ),
            None,
        )

    # Below auto_classify_threshold.
    if best_sim < ambiguity_threshold:
        # Confidence too low even for judge escalation — auto-unclassified.
        return (
            ClauseClassification(
                taxonomy_id=None,
                confidence=best_sim,
                basis="unclassified",
            ),
            None,
        )

    # In the [AMBIGUITY_THRESHOLD, AUTO_CLASSIFY_THRESHOLD) band → judge with hint.
    return (None, ClassificationHint(best_id=best_id, best_sim=best_sim))
