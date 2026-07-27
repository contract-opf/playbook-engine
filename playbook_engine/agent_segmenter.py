"""Agent-as-segmenter — store-backed L1 segmentation (issue #191).

Mirrors the store-backed judges (:mod:`playbook_engine.agent_judge`) for the
segmentation stage, so a **key-free** session lets the AGENT segment documents
into fine-grained clauses instead of falling back to the coarse deterministic
segmenter (:mod:`playbook_engine.segmenter`).

Flow (mirrors ``judge`` / ``judge-apply``):

1. ``playbook segment`` extracts each version and, for every document whose
   segmentation is not yet cached, appends its ``Block`` stream to
   ``<out>/segment/pending.jsonl``.
2. The agent reads pending, partitions each document's blocks into contiguous
   clause ranges (``SegNode`` list), and writes ``<out>/segment/verdicts.jsonl``.
3. ``playbook segment-apply`` loads those ``SegNode`` lists into the
   :class:`~playbook_engine.llm_segmenter_batch.SegmentationVerdictCache`.
4. ``playbook mine`` (config ``segmentation.agent: true``) replays the cached
   segmentation via ``segment_to_tree``'s cache-hit path — **no API call**.

On a cache miss during ``mine``, :class:`StoreBackedSegmentFn` queues the
document and raises :class:`AgentSegmentationPending` (a ``SegmentationQAError``)
so ``mine`` quarantines it for this round — exactly how
``StoreBackedScopeJudge`` signals "no verdict yet" via ``ScopeNeedsReviewError``.

The agent segments **at block boundaries** — each ``SegNode`` spans a contiguous
range of whole blocks (``start_block_id``..``end_block_id``), so grounding /
coverage / reconstruction gates (:mod:`playbook_engine.segmentation_grounding`)
pass by construction without character-exact quote surgery.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from playbook_engine.agent_judge import PendingQueue, _payload_key
from playbook_engine.segmentation_grounding import Block, SegNode
from playbook_engine.segmentation_qa import SegmentationQAError

#: Cache-key ``model`` component used for every agent-produced segmentation.
#: ``segment``, ``segment-apply``, and ``mine`` must all use this same sentinel
#: so a put by ``segment-apply`` is a hit for ``mine`` (the SegmentationVerdictCache
#: key hashes canonical_text + model + prompt_version + schema_hash + effort).
AGENT_SEGMENTER_MODEL = "store-backed-agent"


class AgentSegmentationPending(SegmentationQAError):
    """Raised by :class:`StoreBackedSegmentFn` on a cache miss.

    A subclass of ``SegmentationQAError`` so ``mine_corpus``'s existing
    per-document ``except (SegmentationQAError, ...)`` handler quarantines the
    document (queued for the agent) instead of aborting the whole run — the
    same "retain + flag, never silently drop" contract the scope gate uses.
    """


def block_to_dict(block: Block) -> dict[str, Any]:
    """Serialise a ``Block`` for the segment pending queue."""
    return {
        "block_id": block.block_id,
        "page": block.page,
        "char_span": list(block.char_span),
        "text": block.text,
    }


def segment_payload_key(canonical_text: str) -> str:
    """Content-hash key for a document's segmentation (dedup + verdict join).

    Hashes only ``stage`` + ``canonical_text`` so the same document content
    dedups across versions/agreements — mirrors the deviation judge's
    content-only key.
    """
    return _payload_key({"stage": "segment", "canonical_text": canonical_text})


@dataclass
class StoreBackedSegmentFn:
    """A ``SegmentFn`` that queues un-segmented documents for the agent.

    Slotted in as ``mine_corpus(llm_segment_fn=…)`` alongside a
    ``SegmentationVerdictCache`` (which owns the hit-path replay). This callable
    is therefore only ever invoked on a **cache miss**: it records the document
    to the pending queue (deduplicated by content hash) and raises
    :class:`AgentSegmentationPending` so the document quarantines this round.

    Accepts the optional third ``last_error`` argument
    (``segment_verify_repair`` is repair-aware) but ignores it — there is no
    LLM retry here; the agent is the segmenter.

    ``taxonomy_ids`` MUST be set to the corpus's real classifier ids (issue
    #40's fix). ``segment_apply_cmd`` gates every agent-produced verdict with
    ``run_gates(..., taxonomy_ids=payload["taxonomy_ids"])`` — the same
    allow-list the ``playbook segment`` pre-pass writes into its own queued
    payload. Leaving this at its empty-list default (as this class did before
    issue #40) makes the taxonomy gate reject every verdict that assigns a
    real (non-null) ``taxonomy_id``, since none of them are ever "in" an empty
    allow-list — even though the same document queued via ``segment`` passes.

    ``document_id``/``version`` are included for parity with the payload
    shape ``segment_cmd`` writes (and so the gate's error message names a
    document instead of ``"?"``), but — unlike ``taxonomy_ids`` — they are
    **not** real per-document identity: a single ``StoreBackedSegmentFn``
    instance is shared across every document ``mine_corpus`` segments in a
    run (bound once in ``cli.py``'s ``_llm_segmentation_kwargs``), and the
    ``SegmentFn`` contract this callable implements has no per-call document
    identity to thread through. They default to ``run_gates``'s own
    ``"doc"``/``"v1"`` defaults.
    """

    pending: PendingQueue
    taxonomy_ids: list[str] = field(default_factory=list)
    document_id: str = "doc"
    version: str = "v1"
    _seen: set[str] = field(default_factory=set, init=False, repr=False)

    def __call__(
        self,
        canonical_text: str,
        blocks: list[Block],
        last_error: SegmentationQAError | None = None,
    ) -> list[SegNode]:
        key = segment_payload_key(canonical_text)
        if key not in self._seen:
            self._seen.add(key)
            self.pending.add(
                key,
                "segment",
                {
                    "document_id": self.document_id,
                    "version": self.version,
                    "taxonomy_ids": self.taxonomy_ids,
                    "canonical_text": canonical_text,
                    "blocks": [block_to_dict(b) for b in blocks],
                },
            )
        raise AgentSegmentationPending(
            f"document queued for agent segmentation ({len(blocks)} block(s)); "
            "run `playbook segment-apply` after producing SegNodes"
        )
