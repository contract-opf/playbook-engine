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

Inheritance (issue #222): a heading-less child node (e.g. a segmenter-promoted
``(a)``/``(b)`` sub-item) that ends up with no taxonomy fit
(``basis="unclassified"``) inherits its classified parent's ``taxonomy_id`` with
confidence ``min(parent_confidence, INHERITED_CONFIDENCE_CAP)`` and
``basis="inherited"`` — it is structurally part of its parent clause, and
without this it was absent from the playbook in default mode, where no judge
classifies heading-less nodes. A judge's specific taxonomy fit for the child,
a ``judge_error`` and a ``needs_review`` are kept as they are.

Content similarity (issue #235): a node every other path left
``basis="unclassified"`` and that has body text is compared with our standard's
own clause text, per taxonomy_id (``content_exemplars``: the template's
classified nodes joined). Counterparty forms use their own headings
("Exceptions", "Protection", "Required Disclosure"), so heading matching alone
leaves their clauses out of precedent; the content test recovers them without
a judge and without a key. It is deterministic and deliberately conservative:
a node is assigned only when its best score reaches
``CONTENT_ASSIGN_THRESHOLD`` AND is at least ``CONTENT_MARGIN_RATIO`` times the
runner-up's, at a confidence capped at ``CONTENT_CONFIDENCE_CAP`` (below
``AMBIGUITY_THRESHOLD``, so it never reads as a verified judge verdict). It
runs last, after the heading paths, the judge and parent inheritance, so it
only fills what those left empty; it never runs on the template itself (the
exemplars come from the template's classification) and is a no-op in emergent
mode (no exemplars). The measurements behind the constants are on the
constants' docstrings.

``ClauseClassification.basis`` values:
  ``"exact_match"``        — heading matched a taxonomy label exactly.
  ``"heading_similarity"`` — Jaccard ≥ ``AUTO_CLASSIFY_THRESHOLD``.
  ``"judge"``              — delegated to the injected judge.
  ``"judge_error"``        — judge raised; node recorded as unclassified.
  ``"unclassified"``       — no heading/text, or Jaccard < ``AMBIGUITY_THRESHOLD``
                             (below gate); cannot classify without the judge.
  ``"inherited"``          — a heading-less child given its classified
                             parent's taxonomy_id (issue #222), at a confidence
                             capped at ``INHERITED_CONFIDENCE_CAP``.
  ``"llm_segmenter"``      — assigned by the LLM segmenter's single combined
                             segment+classify pass (see
                             ``pipeline._classified_from_taxonomy_by_path``),
                             not by a dedicated, separately-verified
                             ``ClassificationJudge`` call. Deliberately
                             distinct from ``"judge"`` and always paired with
                             a below-``AMBIGUITY_THRESHOLD`` confidence so
                             these assignments are never mistaken for a
                             verified judge verdict downstream (issue #86).
  ``"content_similarity"`` — a node left unclassified by every other path,
                             assigned by comparing its text with our
                             standard's own clause text per taxonomy_id
                             (issue #235), at a confidence capped at
                             ``CONTENT_CONFIDENCE_CAP``.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Mapping
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

INHERITED_CONFIDENCE_CAP: float = 0.6
"""Ceiling on the confidence of a ``basis="inherited"`` classification (issue
#222): a heading-less child takes ``min(parent_confidence, 0.6)`` — below
``AMBIGUITY_THRESHOLD``, so an inherited assignment always reads as uncertain."""

CONTENT_ASSIGN_THRESHOLD: float = 0.25
"""Minimum content similarity (issue #235) between an unclassified node's text
and our standard's clause text for one taxonomy_id before the node is assigned
that type. Measured on the NDA example's six deals (stopword-filtered token
Jaccard, ``_CONTENT_STOP_WORDS``): the seven correct third-party matches score
0.28-0.53 in their terminal versions; the clauses that must stay unclassified
score at most 0.12 (our-paper personal-data clause), 0.11 (``Confidential
Information`` definition) and 0.06 (the no-obligation clause). 0.25 sits in the
gap, and ``CONTENT_MARGIN_RATIO`` covers the near-ties above it."""

CONTENT_MARGIN_RATIO: float = 2.0
"""The best content score must also be at least this many times the
runner-up's: an assignment between two nearly-equal types would be a guess.
Measured: the seven correct matches lead their runner-up by 3.0x-13.1x; the
near-ties that clear the threshold (a trade-secret survival sentence, 0.25 vs
0.15; a permitted-disclosure sentence, 0.30 vs 0.17) lead by 1.6x-1.8x and are
rejected. Never lower the ratio below 1.5 to recover a clause: a wrong
assignment plants a false precedent."""

CONTENT_CONFIDENCE_CAP: float = 0.5
"""Ceiling on the confidence of a ``basis="content_similarity"`` classification
(issue #235): ``min(best_score, 0.5)`` — below ``AMBIGUITY_THRESHOLD``, so a
content assignment never reads as a verified judge verdict (same rule as
``"llm_segmenter"``, issue #86, and ``INHERITED_CONFIDENCE_CAP``)."""

_BASIS_VALUES = frozenset(
    {
        "exact_match",
        "heading_similarity",
        "judge",
        "judge_error",
        "needs_review",
        "unclassified",
        "llm_segmenter",
        "inherited",
        "content_similarity",
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

#: Stop words for the content-similarity test (issue #235): ``_STOP_WORDS``
#: plus contract boilerplate present in nearly every clause of every agreement
#: type (modals, "party", "agreement", pronouns, relatives). Deliberately
#: contains no agreement-type vocabulary ("confidential", "student", ...):
#: those words are what separates one clause type from another, and the
#: engine is agreement-type-general. Numerals are kept (``_normalize`` keeps
#: digits), so "five (5) years" and "thirty (30) days" count as content.
_CONTENT_STOP_WORDS: frozenset[str] = _STOP_WORDS | frozenset(
    {
        "any",
        "agreement",
        "been",
        "each",
        "either",
        "has",
        "have",
        "if",
        "it",
        "may",
        "not",
        "other",
        "parties",
        "party",
        "shall",
        "such",
        "than",
        "their",
        "then",
        "these",
        "those",
        "which",
        "who",
        "whom",
        "whose",
        "will",
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
    content_exemplars: Mapping[str, str] | None = None,
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
        content_exemplars:       Optional ``{taxonomy_id: our standard's clause
                                 text}`` (issue #235; every template node of
                                 the type, joined). When given, a node still
                                 ``basis="unclassified"`` after the heading
                                 paths, the judge and parent inheritance is
                                 compared with each eligible exemplar and
                                 assigned under ``basis="content_similarity"``
                                 (see :func:`assign_by_content`). ``None`` or
                                 empty (emergent mode, and the template
                                 itself) makes this a no-op.

    Returns:
        One ``ClassifiedClause`` per node in ``tree.all_nodes()`` order. A
        heading-less child left with no taxonomy fit (``basis="unclassified"``)
        inherits its classified parent's ``taxonomy_id`` (``basis="inherited"``,
        confidence ``min(parent, INHERITED_CONFIDENCE_CAP)``; issue #222).

    Raises:
        ValueError: if the judge returns a wrong-length batch, or if any
                    returned classification has ``basis != "judge"``.
    """
    eligible = _eligible_entries(taxonomy)
    eligible_by_id = {e.id: e for e in eligible}
    label_index = _build_label_index(eligible)
    label_tokens = _build_label_tokens(eligible)

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
            label_tokens=label_tokens,
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

    final = [r for r in results if r is not None]
    inherited = _inherit_from_parents(tree, final)
    return assign_by_content(inherited, content_exemplars, eligible_ids=eligible_by_id)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parent_indices(tree: ClauseTree) -> list[int | None]:
    """Each node's parent index in ``tree.all_nodes()`` order (``None`` at the top)."""
    parents: list[int | None] = []

    def _walk(children: list[ClauseNode], parent: int | None) -> None:
        for child in children:
            idx = len(parents)
            parents.append(parent)
            _walk(child.children, idx)

    _walk(tree.nodes, None)
    return parents


def _inherit_from_parents(
    tree: ClauseTree, classified: list[ClassifiedClause]
) -> list[ClassifiedClause]:
    """Give each heading-less, unclassified child its classified parent's
    taxonomy_id (issue #222; see the module docstring).

    *classified* is in ``tree.all_nodes()`` (pre-)order, so a parent is final
    before its children are visited — a grandchild inherits through an
    inherited child, still capped at ``INHERITED_CONFIDENCE_CAP``.
    """
    parents = _parent_indices(tree)
    if len(parents) != len(classified):  # pragma: no cover - defensive
        return classified
    out = list(classified)
    for i, cc in enumerate(out):
        parent = parents[i]
        if parent is None:
            continue
        node = cc.node
        if (node.heading or "").strip() or not (node.text or "").strip():
            continue
        if cc.classification.taxonomy_id is not None or cc.classification.basis != "unclassified":
            continue
        parent_cls = out[parent].classification
        if parent_cls.taxonomy_id is None:
            continue
        out[i] = ClassifiedClause(
            node=node,
            classification=ClauseClassification(
                taxonomy_id=parent_cls.taxonomy_id,
                confidence=min(parent_cls.confidence, INHERITED_CONFIDENCE_CAP),
                basis="inherited",
            ),
        )
    return out


def _content_tokens(text: str) -> frozenset[str]:
    """Content words of *text* for the content-similarity test (issue #235):
    lowercase, punctuation stripped, ``_CONTENT_STOP_WORDS`` dropped, numerals
    kept."""
    return frozenset(w for w in _normalize(text).split() if w not in _CONTENT_STOP_WORDS)


def assign_by_content(
    classified: list[ClassifiedClause],
    content_exemplars: Mapping[str, str] | None,
    *,
    eligible_ids: Collection[str] | None = None,
) -> list[ClassifiedClause]:
    """Assign still-unclassified nodes by content similarity to our standard
    (issue #235; see the module docstring).

    Every node whose basis is ``"unclassified"`` and that has body text is
    scored (token Jaccard over ``_content_tokens``) against each exemplar in
    *content_exemplars* — ``{taxonomy_id: our standard's clause text}``, every
    template node of the type joined, restricted to *eligible_ids* when given
    (active/custom taxonomy entries only, OPF §5). It becomes
    ``basis="content_similarity"`` only when the best score is at least
    ``CONTENT_ASSIGN_THRESHOLD`` AND at least ``CONTENT_MARGIN_RATIO`` times
    the runner-up's, with confidence ``min(best, CONTENT_CONFIDENCE_CAP)``.
    Everything else — including a node a judge, a heading or parent
    inheritance classified, a ``judge_error`` and a ``needs_review`` — is
    returned unchanged, in the same order.

    Deterministic and keyless: a pure function of the node text and the
    exemplars. A no-op when *content_exemplars* is ``None`` or empty (emergent
    mode). Callers MUST NOT pass exemplars when classifying the template
    itself — they come from the template's own classification.

    Used by :func:`classify_tree` and, for the LLM/agent segmentation path
    (which bypasses ``classify_tree``), by the pipeline directly, so the
    fallback applies wherever a node's final basis is ``"unclassified"``.
    """
    if not content_exemplars:
        return classified
    exemplar_tokens = [
        (tid, _content_tokens(text))
        for tid, text in sorted(content_exemplars.items())
        if (eligible_ids is None or tid in eligible_ids) and (text or "").strip()
    ]
    exemplar_tokens = [(tid, toks) for tid, toks in exemplar_tokens if toks]
    if not exemplar_tokens:
        return classified
    out: list[ClassifiedClause] = []
    for cc in classified:
        cls = cc.classification
        text = (cc.node.text or "").strip()
        if cls.basis != "unclassified" or cls.taxonomy_id is not None or not text:
            out.append(cc)
            continue
        node_tokens = _content_tokens(text)
        if not node_tokens:
            out.append(cc)
            continue
        scored = sorted(
            ((_jaccard(node_tokens, toks), tid) for tid, toks in exemplar_tokens),
            key=lambda pair: (-pair[0], pair[1]),
        )
        best, best_tid = scored[0]
        runner_up = scored[1][0] if len(scored) > 1 else 0.0
        if best < CONTENT_ASSIGN_THRESHOLD or best < CONTENT_MARGIN_RATIO * runner_up:
            out.append(cc)
            continue
        out.append(
            ClassifiedClause(
                node=cc.node,
                classification=ClauseClassification(
                    taxonomy_id=best_tid,
                    confidence=min(best, CONTENT_CONFIDENCE_CAP),
                    basis="content_similarity",
                ),
            )
        )
    return out


def _eligible_entries(taxonomy: Taxonomy) -> list[TaxonomyEntry]:
    """Return only active and custom entries (OPF §5)."""
    return [e for e in taxonomy.entries if e.is_classifier_eligible]


def _build_label_index(
    entries: list[TaxonomyEntry],
) -> dict[str, str]:
    """Build {normalized_label: entry_id} for fast exact-match lookup."""
    return {_normalize(e.label): e.id for e in entries}


def _build_label_tokens(
    entries: list[TaxonomyEntry],
) -> list[tuple[TaxonomyEntry, frozenset[str]]]:
    """Precompute ``(entry, _tokens(entry.label))`` pairs once per classify_tree
    call (issue #66).

    ``entry.label`` is constant for the entire run, so re-tokenizing it (two
    regex passes + split + frozenset) inside the per-node Jaccard loop in
    ``_fast_classify`` is pure waste: 400 headed nodes x 41 entries x 100
    versions measured at ~14s of avoidable work per mine.
    """
    return [(e, _tokens(e.label)) for e in entries]


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
    label_tokens: list[tuple[TaxonomyEntry, frozenset[str]]] | None = None,
) -> tuple[ClauseClassification | None, ClassificationHint | None]:
    """Attempt deterministic classification.

    Returns a 2-tuple ``(classification, hint)``:

    - ``(ClauseClassification, None)`` — classified by fast path; no judge needed.
    - ``(None, ClassificationHint)``  — in ambiguity band; queue for judge with hint.
    - ``(None, None)``                — no heading tokens or text-only; queue for judge
                                        without a hint (Jaccard not applicable).

    Args:
        label_tokens: Precomputed ``(entry, _tokens(entry.label))`` pairs for
                      *eligible*, built once per ``classify_tree`` call so the
                      Jaccard loop below never re-tokenizes a taxonomy label
                      (issue #66). Computed lazily from *eligible* when not
                      supplied, so direct callers (e.g. tests) keep working
                      unchanged.
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

    if label_tokens is None:
        label_tokens = _build_label_tokens(eligible)

    best_id: str | None = None
    best_sim: float = 0.0
    for entry, entry_tokens in label_tokens:
        sim = _jaccard(h_tokens, entry_tokens)
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
