"""Segmentation QA — deterministic gates + verify/repair loop (fail-loud).

LLM-first segmentation (see :mod:`playbook_engine.llm_segmenter` and
:mod:`playbook_engine.segmentation_grounding`) is only safe because nothing
downstream ever trusts the model's output directly: every candidate
segmentation must pass a battery of deterministic gates before it is accepted,
and if it never does, the document is flagged for human review rather than
silently degraded to a worse (or wrong) tree.

Gates (run in this order by :func:`run_gates`):

1. **grounding**       — :func:`~playbook_engine.segmentation_grounding.ground_segmentation`
                          resolves the LLM's block-anchored ``SegNode``s against
                          the canonical block stream. A
                          :class:`~playbook_engine.segmentation_grounding.GroundingError`
                          (unknown block id, inverted span, boundary-quote
                          mismatch, cyclic/orphaned parent, ...) fails this gate.
2. **coverage**        — the union of every node's own character span (leaf
                          and internal alike — grounding truncates an internal
                          node's own span/text to its heading/preamble,
                          excluding its children's spans, so every node's span
                          is disjoint and must be counted; see
                          :mod:`playbook_engine.segmentation_grounding`'s
                          module docstring) covers all of ``canonical_text``
                          except whitespace: no gaps other than whitespace,
                          and no overlaps between spans.
3. **reconstruction**  — concatenating every node's own text in span order
                          reproduces the non-whitespace content of
                          ``canonical_text`` exactly.
4. **tree**            — :meth:`~playbook_engine.clause_tree.ClauseTree.validate`
                          passes (no duplicate paths, children nest inside
                          their parent's span, siblings are ordered, spans are
                          in-bounds).
5. **taxonomy**        — every non-``null`` ``taxonomy_id`` the model assigned
                          is one of the caller's allowed ``taxonomy_ids``.

The schema gate (well-formed JSON matching ``SegNode``) is enforced upstream
by structured output (see ``llm_segmenter._parse_seg_nodes`` /
``CLAUSE_TREE_SCHEMA``) and is not repeated here. The consistency/verifier-LLM
gate mentioned in the design is exactly the *outer* verify→repair loop below —
each repair attempt is itself a fresh pass through every gate above.

:func:`segment_verify_repair` is the fail-loud driver: it calls the injected
``segment_fn`` (the real LLM segmenter in production; a fake/canned callable
in tests — this module never imports :mod:`playbook_engine.llm_segmenter` or
makes a network call), runs the gates, and on failure re-invokes ``segment_fn``
for a bounded number of repair attempts. ``segment_fn`` is called as
``segment_fn(canonical_text, blocks)`` per the ticket's base call contract,
with one addition: a *repair-aware* ``segment_fn`` may declare a third
positional parameter — the previous attempt's :class:`SegmentationQAError`
(``None`` on the first call) — and :func:`segment_verify_repair` detects this
via :func:`inspect.signature` (checked once, before the loop) and passes it
on every subsequent call. This is what makes a repair an actual repair
rather than "same model, same input, hope sampling differs": the injected
callable can fold the previous gate failure's detail (e.g. "gap [3812,
4102) before node 7 contains: ...") into its next prompt. A ``segment_fn``
that only accepts the base two-argument shape keeps working unchanged — the
third argument is never forced on it. A production caller binds
``segment_fn`` to :func:`~playbook_engine.llm_segmenter.segment_document`
with ``taxonomy_ids`` and ``client``/``model`` already fixed (e.g. via a
closure) and its own third parameter accepting the last error, which it
threads into ``segment_document``'s ``repair_feedback`` argument. If every
attempt (the first call plus up to ``max_repairs`` retries) still fails a
gate, :func:`segment_verify_repair` raises :class:`SegmentationQAError` —
there is no deterministic-segmenter fallback, by design.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable

from playbook_engine.clause_tree import ClauseTree, ClauseTreeError
from playbook_engine.segmentation_grounding import (
    Block,
    GroundingError,
    GroundingResult,
    SegNode,
    ground_segmentation,
)

# ---------------------------------------------------------------------------
# Error
# ---------------------------------------------------------------------------


class SegmentationQAError(Exception):
    """Raised when a document cannot pass the deterministic QA gates.

    Raised by :func:`run_gates` naming the specific failing gate, and by
    :func:`segment_verify_repair` when every attempt (initial + repairs) still
    fails. By design there is no fallback: a QA failure means the document is
    flagged for human review, never silently degraded to a worse tree.
    """


# ---------------------------------------------------------------------------
# Gate 2 — coverage
# ---------------------------------------------------------------------------


def _check_coverage(canonical_text: str, tree: ClauseTree) -> None:
    """Gate: every node's own span covers all of ``canonical_text`` except
    whitespace, with no overlaps.

    Every node — leaf or internal — owns a distinct slice of the source:
    :func:`~playbook_engine.segmentation_grounding.ground_segmentation`
    truncates a node with children so its own span/text covers only its own
    heading/preamble, and its children cover the body content in separate,
    later spans (see that module's docstring, and ``ClauseTree``'s ``text``
    field: "body text of this node, not including children"). So coverage
    must walk *every* node's span, not just leaves — an internal node's own
    heading text would otherwise never be counted as covered.

    Raises:
        SegmentationQAError: on a coverage gap (non-whitespace text not
                              claimed by any node's span) or an overlap
                              between two node spans.
    """
    nodes = sorted(tree.all_nodes(), key=lambda n: n.char_span)

    cursor = 0
    for node in nodes:
        start, end = node.char_span
        if start < cursor:
            raise SegmentationQAError(
                f"coverage gate: node {node.clause_path!r} span [{start}, {end}) "
                f"overlaps the preceding node (cursor at {cursor})"
            )
        gap = canonical_text[cursor:start]
        if gap.strip():
            raise SegmentationQAError(
                f"coverage gate: gap [{cursor}, {start}) before node "
                f"{node.clause_path!r} contains non-whitespace text: {gap!r}"
            )
        cursor = max(cursor, end)

    trailing = canonical_text[cursor:]
    if trailing.strip():
        raise SegmentationQAError(
            f"coverage gate: trailing gap [{cursor}, {len(canonical_text)}) after the "
            f"last node contains non-whitespace text: {trailing!r}"
        )


# ---------------------------------------------------------------------------
# Gate 3 — reconstruction
# ---------------------------------------------------------------------------


def _strip_all_ws(text: str) -> str:
    """Remove every whitespace character — stricter than ``_norm_ws``.

    Used only for the reconstruction gate: leaves are joined with no
    separator (a leaf's ``.text`` never includes the inter-block whitespace
    that sits *between* two sibling leaves — that gap is exactly what the
    coverage gate already tolerates), so comparing against ``canonical_text``
    must ignore *all* whitespace rather than collapse it to single spaces —
    collapsing would still require a separator space at every leaf boundary
    that the naive join never inserts, producing spurious failures on the
    overwhelmingly common case of leaves separated by real whitespace.
    """
    return "".join(text.split())


def _check_reconstruction(canonical_text: str, tree: ClauseTree) -> None:
    """Gate: concatenated node texts (in span order) reproduce
    ``canonical_text``'s non-whitespace content exactly.

    Walks every node (not just leaves) for the same reason as the coverage
    gate: each node's own ``.text`` (its heading/preamble, excluding
    children) is real source content that must round-trip too.

    Raises:
        SegmentationQAError: when the node texts, concatenated in span order
                              and stripped of all whitespace, do not equal
                              ``canonical_text`` stripped of all whitespace.
    """
    nodes = sorted(tree.all_nodes(), key=lambda n: n.char_span)
    reconstructed = _strip_all_ws("".join(node.text for node in nodes))
    expected = _strip_all_ws(canonical_text)
    if reconstructed != expected:
        raise SegmentationQAError(
            "reconstruction gate: concatenated node text does not reproduce "
            f"canonical_text (reconstructed {len(reconstructed)} chars, "
            f"expected {len(expected)} chars)"
        )


# ---------------------------------------------------------------------------
# Gate 4 — tree
# ---------------------------------------------------------------------------


def _check_tree(canonical_text: str, tree: ClauseTree) -> None:
    """Gate: ``tree.validate()`` — structural invariants hold.

    Raises:
        SegmentationQAError: wrapping any ``ClauseTreeError`` from ``validate``
                              (duplicate path, out-of-bounds span, sibling
                              order violation, child-outside-parent, path
                              prefix mismatch).
    """
    try:
        tree.validate(full_text=canonical_text)
    except ClauseTreeError as exc:
        raise SegmentationQAError(f"tree gate: {exc}") from exc


# ---------------------------------------------------------------------------
# Gate 5 — taxonomy
# ---------------------------------------------------------------------------


def _check_taxonomy(taxonomy_by_path: dict[str, str | None], taxonomy_ids: list[str]) -> None:
    """Gate: every non-``null`` assigned ``taxonomy_id`` is an allowed id.

    Raises:
        SegmentationQAError: naming the clause path and the out-of-enum id.
    """
    allowed = set(taxonomy_ids)
    for clause_path, taxonomy_id in taxonomy_by_path.items():
        if taxonomy_id is not None and taxonomy_id not in allowed:
            raise SegmentationQAError(
                f"taxonomy gate: clause {clause_path!r} has taxonomy_id "
                f"{taxonomy_id!r} which is not in the allowed taxonomy_ids"
            )


# ---------------------------------------------------------------------------
# run_gates — gates 1-5 in order
# ---------------------------------------------------------------------------


def run_gates(
    canonical_text: str,
    blocks: list[Block],
    seg_nodes: list[SegNode],
    *,
    taxonomy_ids: list[str],
    document_id: str = "doc",
    version: str = "v1",
    source_file: str = "",
) -> GroundingResult:
    """Run every deterministic QA gate against a candidate segmentation.

    Args:
        canonical_text: The document's full canonical text.
        blocks:         Block stream in reading order (see
                        :mod:`playbook_engine.segmentation_grounding`).
        seg_nodes:      The candidate segmentation to gate (typically the raw
                        output of an LLM segmenter call).
        taxonomy_ids:   Allowed taxonomy ids for the taxonomy gate.
        document_id:    Passed through to the resulting tree's metadata.
                        Gate outcomes never depend on this value.
        version:        Passed through to the resulting tree's metadata.
                        Gate outcomes never depend on this value.
        source_file:    Passed through to the resulting tree's metadata.
                        Gate outcomes never depend on this value.

    Returns:
        The grounded :class:`~playbook_engine.segmentation_grounding.GroundingResult`
        once every gate has passed.

    Raises:
        SegmentationQAError: naming the first failing gate and the reason.
                              Fail loud — no fallback.
    """
    try:
        result = ground_segmentation(
            document_id=document_id,
            version=version,
            source_file=source_file,
            canonical_text=canonical_text,
            blocks=blocks,
            seg_nodes=seg_nodes,
        )
    except GroundingError as exc:
        raise SegmentationQAError(f"grounding gate: {exc}") from exc

    _check_coverage(canonical_text, result.tree)
    _check_reconstruction(canonical_text, result.tree)
    _check_tree(canonical_text, result.tree)
    _check_taxonomy(result.taxonomy_by_path, taxonomy_ids)

    return result


# ---------------------------------------------------------------------------
# segment_fn arity detection — is this a repair-aware callable?
# ---------------------------------------------------------------------------


def _accepts_last_error(segment_fn: Callable[..., list[SegNode]]) -> bool:
    """Whether *segment_fn* opts into receiving the previous attempt's
    :class:`SegmentationQAError` as a third positional argument.

    A repair-aware ``segment_fn`` declares a third parameter — positional
    (with or without a default) or ``**kwargs`` — that
    :func:`segment_verify_repair` fills with the previous attempt's failure
    (``None`` on the first call). A ``segment_fn`` matching only the base
    ``(canonical_text, blocks)`` shape is left untouched: it is never called
    with a third argument, so existing two-argument fakes/closures keep
    working unchanged.

    Checked once per :func:`segment_verify_repair` call (arity cannot change
    between attempts), not on every attempt.
    """
    try:
        sig = inspect.signature(segment_fn)
    except (TypeError, ValueError):
        # Builtins / C callables without an inspectable signature: assume
        # the conservative base two-argument shape.
        return False

    params = list(sig.parameters.values())
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params):
        return True

    callable_positionally = [
        p
        for p in params
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    if len(callable_positionally) >= 3:
        return True

    return any(p.kind is inspect.Parameter.VAR_POSITIONAL for p in params)


# ---------------------------------------------------------------------------
# segment_verify_repair — verify/repair loop (fail-loud, no fallback)
# ---------------------------------------------------------------------------


def segment_verify_repair(
    canonical_text: str,
    blocks: list[Block],
    *,
    taxonomy_ids: list[str],
    segment_fn: Callable[..., list[SegNode]],
    max_repairs: int = 2,
) -> GroundingResult:
    """Segment, verify against the QA gates, and repair on failure.

    Calls ``segment_fn(canonical_text, blocks)`` to obtain a candidate
    segmentation and runs :func:`run_gates` against it. If the gates fail,
    ``segment_fn`` is re-invoked up to ``max_repairs`` additional times. When
    ``segment_fn`` is repair-aware (see :func:`_accepts_last_error` — it
    declares a third parameter), every call after the first also receives
    the previous attempt's :class:`SegmentationQAError` as that third
    positional argument, so the "repair" is an actual repair — the injected
    callable can fold the specific gate failure into its next prompt —
    rather than re-invoking with byte-identical arguments and hoping
    sampling differs. A ``segment_fn`` matching only the base two-argument
    shape is called exactly as before. If every attempt still fails, the QA
    failure is raised rather than swallowed: by design there is no
    deterministic-segmenter fallback.

    Args:
        canonical_text: The document's full canonical text.
        blocks:         Block stream in reading order.
        taxonomy_ids:   Allowed taxonomy ids for the taxonomy gate. Not
                        passed to ``segment_fn`` — a production caller binds
                        ``segment_fn`` to
                        ``playbook_engine.llm_segmenter.segment_document``
                        with ``taxonomy_ids`` (and ``client``/``model``)
                        already fixed, e.g. via ``functools.partial`` or a
                        closure, so the call shape below is stable
                        regardless of what the underlying segmenter itself
                        needs.
        segment_fn:     Injected segmenter callable — the real LLM segmenter
                        in production (pre-bound as described above), a
                        fake/canned callable in tests. Called as
                        ``segment_fn(canonical_text, blocks)``, or
                        ``segment_fn(canonical_text, blocks, last_error)``
                        when repair-aware (see above), on every attempt.
        max_repairs:    Maximum number of re-segmentation attempts after the
                        first call. Total attempts = ``1 + max_repairs``.

    Returns:
        The grounded :class:`~playbook_engine.segmentation_grounding.GroundingResult`
        from the first attempt (initial or repair) that passes every gate.

    Raises:
        SegmentationQAError: if every attempt (initial + all repairs) fails
                              a gate. Fail loud — no fallback.
    """
    last_error: SegmentationQAError | None = None
    repair_aware = _accepts_last_error(segment_fn)

    for _attempt in range(max_repairs + 1):
        seg_nodes = (
            segment_fn(canonical_text, blocks, last_error)
            if repair_aware
            else segment_fn(canonical_text, blocks)
        )
        try:
            return run_gates(
                canonical_text,
                blocks,
                seg_nodes,
                taxonomy_ids=taxonomy_ids,
            )
        except SegmentationQAError as exc:
            last_error = exc

    raise SegmentationQAError(
        f"segment_verify_repair: exhausted {max_repairs} repair attempt(s) "
        f"({max_repairs + 1} total attempts); last failure: {last_error}"
    ) from last_error
