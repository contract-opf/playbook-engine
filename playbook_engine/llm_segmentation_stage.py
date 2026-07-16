"""LLM-segmentation stage — thin orchestration for the L1 seam (issue #74).

Wires the extractor (:mod:`playbook_engine.extraction`), the LLM segmenter
(:mod:`playbook_engine.llm_segmenter`), and the QA verify/repair loop
(:mod:`playbook_engine.segmentation_qa`) into one call that turns a source
document straight into a grounded :class:`~playbook_engine.clause_tree.ClauseTree`
plus its per-clause taxonomy assignments — the LLM-segmentation alternative to
``segment(ingest(...).tree)``.

This module owns no policy of its own: it composes three already-tested
seams in the documented order (extract → segment/verify/repair) and adds
nothing but the default production binding of ``segment_fn``. Every gate
(grounding, coverage, reconstruction, tree, taxonomy) and the repair loop
itself live in :mod:`playbook_engine.segmentation_qa`; a
:class:`~playbook_engine.segmentation_qa.SegmentationQAError` from that module
propagates unchanged — there is no fallback to the deterministic
:mod:`playbook_engine.segmenter` here, by design (see the ticket's Out of
scope: this path must never silently degrade to the deterministic segmenter).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from playbook_engine.extraction import ExtractionCache, extract_blocks
from playbook_engine.llm_segmenter import DEFAULT_MODEL
from playbook_engine.llm_segmenter_batch import SegmentationVerdictCache
from playbook_engine.segmentation_grounding import Block, GroundingResult, SegNode
from playbook_engine.segmentation_qa import (
    SegmentationQAError,
    run_gates,
    segment_verify_repair,
)

#: Call shape every ``segment_fn`` must satisfy — identical to
#: ``segmentation_qa.segment_verify_repair``'s own contract, so a caller-supplied
#: fake and the production default are interchangeable.
SegmentFn = Callable[[str, list[Block]], list[SegNode]]


def _default_segment_fn(taxonomy_ids: list[str]) -> SegmentFn:
    """Bind :func:`~playbook_engine.llm_segmenter.segment_document` to *taxonomy_ids*.

    The real ``anthropic`` client is never constructed here: importing
    ``llm_segmenter`` at call time (not module import time) and passing
    ``client=None`` defers that to ``segment_document`` itself, which only
    builds a live client lazily, on first use. No test exercises this
    function — tests always inject their own ``segment_fn``.

    The returned closure is repair-aware (see
    :func:`~playbook_engine.segmentation_qa._accepts_last_error`): it
    declares a third ``last_error`` parameter that
    ``segment_verify_repair`` fills with the previous attempt's
    :class:`~playbook_engine.segmentation_qa.SegmentationQAError` on every
    repair, and threads it into ``segment_document``'s ``repair_feedback``
    so a retry's prompt actually reflects what failed rather than
    re-sending byte-identical input.
    """

    def _segment(
        canonical_text: str,
        blocks: list[Block],
        last_error: SegmentationQAError | None = None,
    ) -> list[SegNode]:
        from playbook_engine.llm_segmenter import segment_document

        return segment_document(
            canonical_text,
            blocks,
            taxonomy_ids,
            repair_feedback=str(last_error) if last_error is not None else None,
        )

    return _segment


def segment_to_tree(
    path: Path,
    *,
    taxonomy_ids: list[str],
    segment_fn: SegmentFn | None = None,
    max_repairs: int = 2,
    cache: SegmentationVerdictCache | None = None,
    model: str = DEFAULT_MODEL,
    extraction_cache: ExtractionCache | None = None,
) -> GroundingResult:
    """Extract + LLM-segment + verify/repair *path* into a grounded tree.

    ``extract_blocks(path)`` → ``segment_verify_repair(canonical_text, blocks,
    taxonomy_ids=taxonomy_ids, segment_fn=segment_fn)``. This is the
    LLM-segmentation alternative to ``segment(ingest(...).tree)``: same
    ``ClauseTree`` output contract, but classification happens in the same
    LLM pass as segmentation (see ``GroundingResult.taxonomy_by_path``), so
    callers on this path skip ``classify_tree`` entirely.

    The returned tree's ``document_id``/``version``/``source_file`` are
    whatever :func:`~playbook_engine.segmentation_qa.run_gates` defaults to
    (``"doc"``/``"v1"``/``""``) — ``segment_verify_repair`` does not accept
    those identifiers. Callers that need the real per-document/per-version
    identity (e.g. the pipeline, which already knows ``doc_id``/``vid`` from
    the corpus layout) must set ``tree.document_id`` / ``tree.version`` /
    ``tree.source_file`` on the returned result themselves — ``ClauseTree``
    is a plain mutable dataclass, so this is a direct assignment, not a
    rebuild.

    Args:
        path:         Source document (``.docx``, ``.pdf``, or ``.rtf``).
        taxonomy_ids: Allowed taxonomy ids the LLM may assign (see
                      :mod:`playbook_engine.llm_segmenter`); also the
                      taxonomy gate's allow-list.
        segment_fn:   Injectable segmenter callable matching
                      ``Callable[[str, list[Block]], list[SegNode]]``. When
                      ``None`` (the default), binds
                      :func:`~playbook_engine.llm_segmenter.segment_document`
                      to *taxonomy_ids* with a lazily-constructed client.
                      Tests always inject a fake so no network call is made.
        max_repairs:  Passed through to
                      :func:`~playbook_engine.segmentation_qa.segment_verify_repair`.
        cache:        Optional :class:`~playbook_engine.llm_segmenter_batch.SegmentationVerdictCache`.
                      When given, ``canonical_text`` is checked first — a hit
                      skips ``segment_fn`` entirely and re-grounds the cached
                      ``SegNode`` list via :func:`~playbook_engine.segmentation_qa.run_gates`
                      (no LLM call at all), mirroring
                      :func:`~playbook_engine.llm_segmenter_batch.segment_documents_batch`'s
                      own cache usage. On a miss, the verify/repair loop runs
                      as normal and the *winning* attempt's raw ``SegNode``
                      output is written to the cache only once the whole loop
                      succeeds — never after an individual attempt — so a
                      QA-gate failure can never poison the cache with a
                      result that would otherwise make every subsequent
                      repair attempt replay the same failing output (the
                      batch path never hits this: it has no repair loop at
                      all). Defaults to ``None`` (no caching — every call
                      re-invokes ``segment_fn``).
        model:        The *actual* model id ``segment_fn`` calls through to
                      (see :data:`~playbook_engine.llm_segmenter.DEFAULT_MODEL`
                      and ``config.segmentation.model`` — issue #131). Used
                      only as a cache-key component: it must match whatever
                      model ``segment_fn`` was bound to, or a config's model
                      change would silently replay another model's cached
                      segmentation instead of busting the cache. Defaults to
                      ``DEFAULT_MODEL`` to match ``segment_fn``'s own default
                      when neither is overridden.
        extraction_cache: Optional :class:`~playbook_engine.extraction.ExtractionCache`.
                      When given, passed straight through to
                      :func:`~playbook_engine.extraction.extract_blocks` — a
                      hit against *path*'s current content skips extraction
                      entirely (no docling/pdfplumber/python-docx/pandoc
                      call). Independent of ``cache`` above, which only
                      covers the LLM segmentation call itself: extraction,
                      not segmentation, is the dominant cost on a real
                      corpus with scanned PDFs (issue #132). Defaults to
                      ``None`` (no caching — every call re-extracts).

    Returns:
        The grounded :class:`~playbook_engine.segmentation_grounding.GroundingResult`
        (``tree`` + ``taxonomy_by_path``) once every QA gate has passed.

    Raises:
        ExtractionError:      ``path`` cannot be extracted (see
                               :mod:`playbook_engine.extraction`).
        SegmentationQAError:  every attempt (initial + repairs) still fails a
                               gate. Fail loud — no deterministic-segmenter
                               fallback.
    """
    # extractor (docling vs legacy) is not needed here — mine_corpus records
    # it per version via extraction.detect_extractor before this ever runs
    # (see pipeline._compute_doc_result), so this path only needs the text.
    canonical_text, blocks, _extractor = extract_blocks(path, cache=extraction_cache)

    if cache is not None:
        cached_nodes = cache.get(canonical_text, model=model)
        if cached_nodes is not None:
            return run_gates(canonical_text, blocks, cached_nodes, taxonomy_ids=taxonomy_ids)

    fn: SegmentFn = segment_fn if segment_fn is not None else _default_segment_fn(taxonomy_ids)

    last_seg_nodes: list[SegNode] = []

    def _recording_fn(_canonical_text: str, _blocks: list[Block]) -> list[SegNode]:
        nonlocal last_seg_nodes
        last_seg_nodes = fn(_canonical_text, _blocks)
        return last_seg_nodes

    result = segment_verify_repair(
        canonical_text,
        blocks,
        taxonomy_ids=taxonomy_ids,
        segment_fn=_recording_fn if cache is not None else fn,
        max_repairs=max_repairs,
    )

    if cache is not None:
        cache.put(canonical_text, last_seg_nodes, model=model)

    return result


__all__: list[str] = ["SegmentFn", "segment_to_tree"]
