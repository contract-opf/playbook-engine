"""LLM segmentation — Message Batches, verdict cache, cross-version consistency (issue #75).

Three production-readiness pieces for the one-time corpus pass, layered on top
of :mod:`playbook_engine.llm_segmenter`'s per-document call shape:

- :func:`segment_documents_batch` — segments many documents in one **Anthropic
  Message Batches** call (``client.messages.batches.create`` → poll
  ``retrieve`` → iterate ``results``), keyed by ``custom_id`` (batch results
  are unordered — callers must never assume position). An optional
  :class:`SegmentationVerdictCache` short-circuits documents whose content has
  already been segmented, sending only cache misses to the batch.

- :class:`SegmentationVerdictCache` — a content-hash cache mirroring
  :class:`~playbook_engine.agent_judge.VerdictStore` (same JSONL-on-disk,
  load-on-init, corrupt-line-skip contract); reuses ``VerdictStore`` directly
  rather than reimplementing it. Cache key: ``sha256(canonical_text + model +
  prompt_version + schema_hash + effort)`` — changing the model, prompt
  version, schema shape, or thinking effort invalidates the cache.

- :func:`normalize_trail` — an agreement-level pass over every verified
  version of one agreement's clause trees. Given each version's clause
  headings + taxonomy labels, one Opus call normalizes inconsistent
  ``taxonomy_id`` assignments across the trail (e.g. the same clause labeled
  ``indemnification`` in v1 but ``limitation_of_liability`` in v3 purely from
  independent per-version LLM calls) so that downstream diffing sees a
  consistent taxonomy per clause lineage.

This module makes no live API calls in tests: every function accepts an
injectable ``client`` exposing the same ``.messages``/``.messages.batches``
surface as :mod:`playbook_engine.llm_segmenter`; a real ``anthropic.Anthropic()``
client is only constructed lazily when ``client is None``.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from playbook_engine.agent_judge import VerdictStore
from playbook_engine.clause_tree import ClauseTree
from playbook_engine.llm_segmenter import (
    DEFAULT_MODEL,
    SegmentationLLMError,
    _clause_tree_schema,
    _parse_seg_nodes,
    _serialize_blocks,
    _system_prompt_with_taxonomy,
)
from playbook_engine.segmentation_grounding import Block, SegNode

# ---------------------------------------------------------------------------
# Cache identity constants
# ---------------------------------------------------------------------------

#: Bumped whenever the segmenter's system prompt or request shape changes in
#: a way that should invalidate previously cached segmentations, independent
#: of the schema's own shape (captured separately by ``SCHEMA_HASH``).
PROMPT_VERSION = "v1"

#: SHA-256 of the schema's static shape (``_clause_tree_schema([])`` — the
#: per-node field set and structure, independent of the specific taxonomy id
#: enum, which varies per corpus and would otherwise make the cache key
#: unstable across documents that legitimately share a model/prompt/effort).
#: Changing the node schema (new/removed/renamed field) changes this hash and
#: invalidates the cache.
SCHEMA_HASH = hashlib.sha256(
    json.dumps(_clause_tree_schema([]), sort_keys=True).encode()
).hexdigest()

#: Default ``output_config.effort`` — mirrors ``DEFAULT_MODEL``'s role as
#: this module's single source of truth, and is likewise folded into the
#: pipeline's stage-cache fingerprint.
DEFAULT_EFFORT = "high"


# ---------------------------------------------------------------------------
# SegmentationVerdictCache — content-hash cache, mirrors agent_judge.VerdictStore
# ---------------------------------------------------------------------------


class SegmentationVerdictCache:
    """Judge-once, deterministic-replay cache for LLM segmentation output.

    Wraps a :class:`~playbook_engine.agent_judge.VerdictStore` rather than
    reimplementing content-hash JSONL storage — see that class for the
    on-disk format and corrupt-line-skip contract.

    Cache key inputs: ``canonical_text``, ``model``, ``prompt_version``,
    ``schema_hash``, ``effort``. All five are part of the stored payload (not
    hashed separately), so a change to any one of them is a cache miss.
    """

    def __init__(self, cache_path: Path) -> None:
        self._store = VerdictStore(cache_path)

    def get(
        self,
        canonical_text: str,
        *,
        model: str,
        prompt_version: str = PROMPT_VERSION,
        schema_hash: str = SCHEMA_HASH,
        effort: str = DEFAULT_EFFORT,
    ) -> list[SegNode] | None:
        """Return the cached ``SegNode`` list, or ``None`` on a cache miss."""
        payload = _cache_payload(
            canonical_text,
            model=model,
            prompt_version=prompt_version,
            schema_hash=schema_hash,
            effort=effort,
        )
        cached = self._store.get(payload)
        if cached is None:
            return None
        return [_seg_node_from_dict(raw) for raw in cached["nodes"]]

    def put(
        self,
        canonical_text: str,
        seg_nodes: list[SegNode],
        *,
        model: str,
        prompt_version: str = PROMPT_VERSION,
        schema_hash: str = SCHEMA_HASH,
        effort: str = DEFAULT_EFFORT,
    ) -> None:
        """Store *seg_nodes* for the given cache-key inputs."""
        payload = _cache_payload(
            canonical_text,
            model=model,
            prompt_version=prompt_version,
            schema_hash=schema_hash,
            effort=effort,
        )
        self._store.put(payload, {"nodes": [_seg_node_to_dict(n) for n in seg_nodes]})


def _cache_payload(
    canonical_text: str,
    *,
    model: str,
    prompt_version: str,
    schema_hash: str,
    effort: str,
) -> dict[str, str]:
    """Build the payload dict whose content hash is the cache key.

    ``VerdictStore`` hashes the full JSON-serialised payload — including all
    five components here reproduces the ticket's specified key exactly:
    ``sha256(canonical_text + model + prompt_version + schema_hash + effort)``.
    """
    return {
        "canonical_text": canonical_text,
        "model": model,
        "prompt_version": prompt_version,
        "schema_hash": schema_hash,
        "effort": effort,
    }


def _seg_node_to_dict(node: SegNode) -> dict[str, Any]:
    return {
        "node_id": node.node_id,
        "parent_id": node.parent_id,
        "order": node.order,
        "heading": node.heading,
        "taxonomy_id": node.taxonomy_id,
        "start_block_id": node.start_block_id,
        "end_block_id": node.end_block_id,
        "start_quote": node.start_quote,
        "end_quote": node.end_quote,
    }


def _seg_node_from_dict(raw: dict[str, Any]) -> SegNode:
    return SegNode(
        node_id=raw["node_id"],
        parent_id=raw["parent_id"],
        order=raw["order"],
        heading=raw["heading"],
        taxonomy_id=raw["taxonomy_id"],
        start_block_id=raw["start_block_id"],
        end_block_id=raw["end_block_id"],
        start_quote=raw.get("start_quote", ""),
        end_quote=raw.get("end_quote", ""),
    )


# ---------------------------------------------------------------------------
# segment_documents_batch — Anthropic Message Batches
# ---------------------------------------------------------------------------


class SegmentationBatchItem:
    """One document submitted to :func:`segment_documents_batch`.

    Attributes:
        custom_id:      Caller-chosen unique id for this document within the
                         batch (e.g. ``f"{doc_id}/{version}"``). Results are
                         returned unordered by the Batches API and are keyed
                         back to the request by this id — never by position.
        canonical_text: The document's full canonical text (also the cache
                         key input; unused directly in the request body, same
                         as ``segment_document``'s ``canonical_text`` param).
        blocks:         The document's block stream in reading order.
    """

    __slots__ = ("custom_id", "canonical_text", "blocks")

    def __init__(self, custom_id: str, canonical_text: str, blocks: list[Block]) -> None:
        self.custom_id = custom_id
        self.canonical_text = canonical_text
        self.blocks = blocks


#: Terminal batch processing_status value — see the Message Batches API.
_BATCH_ENDED_STATUS = "ended"

#: Seconds to sleep between polls of ``.batches.retrieve``. Overridable via
#: ``segment_documents_batch(..., poll_interval_s=0)`` in tests, so the fake
#: client's canned "already ended" response never actually sleeps.
_DEFAULT_POLL_INTERVAL_S = 5.0

#: Cap on poll attempts before giving up (issue #98): the Message Batches API
#: guarantees a batch reaches a terminal state within 24h, so at the default
#: 5s interval this is a ~24h ceiling. Without a cap, a batch stuck in a
#: non-terminal state (API incident, etc.) polls ``.retrieve`` forever with
#: no way to distinguish "still processing" from "never coming back". A real
#: completion (usually minutes) never comes close to this cap.
_DEFAULT_MAX_POLLS = 17280


class BatchPollCapExceededError(Exception):
    """Raised when a batch is still non-terminal after ``max_polls`` polls.

    The batch itself is not cancelled — it may still complete server-side.
    Re-running with the same ``batch_id`` (surfaced in the message) lets an
    operator re-attach and inspect it via the Anthropic API/console rather
    than resubmitting the same documents as a new (billed) batch.
    """


def _build_batch_request(
    item: SegmentationBatchItem,
    *,
    taxonomy_ids: list[str],
    model: str,
    max_tokens: int,
) -> dict[str, Any]:
    """Build one ``Request(custom_id=..., params=...)`` dict for the batch.

    Mirrors ``segment_document``'s single-call request body exactly (same
    schema, system prompt, block serialization) — the only difference between
    the single-call and batched paths is that this body travels inside a
    ``messages.batches.create(requests=[...])`` envelope instead of a direct
    ``messages.create(**kwargs)`` call.
    """
    schema = _clause_tree_schema(taxonomy_ids)
    system_prompt = _system_prompt_with_taxonomy(taxonomy_ids)
    user_content = _serialize_blocks(item.blocks)
    return {
        "custom_id": item.custom_id,
        "params": {
            "model": model,
            "max_tokens": max_tokens,
            "thinking": {"type": "adaptive"},
            "output_config": {
                "effort": "high",
                "format": {"type": "json_schema", "schema": schema},
            },
            "system": [
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            "messages": [{"role": "user", "content": user_content}],
        },
    }


def _extract_seg_nodes_from_result(custom_id: str, result: Any) -> list[SegNode]:
    """Parse one ``MessageBatchIndividualResponse.result`` into ``SegNode``s.

    Raises:
        SegmentationLLMError: the result is not a succeeded result, or its
                               message content doesn't parse per
                               ``llm_segmenter._parse_seg_nodes``.
    """
    result_type = getattr(result, "type", None)
    if result_type != "succeeded":
        raise SegmentationLLMError(
            f"batch result for custom_id {custom_id!r} did not succeed (type={result_type!r})"
        )

    message = getattr(result, "message", None)
    content = getattr(message, "content", None) if message is not None else None
    if not content:
        raise SegmentationLLMError(
            f"batch result for custom_id {custom_id!r} has no content blocks"
        )

    # Same adaptive-thinking skip as segment_document: the answer is the
    # first (and only) text block, not necessarily content[0].
    text_block = next((b for b in content if getattr(b, "type", None) == "text"), None)
    if text_block is None:
        raise SegmentationLLMError(
            f"batch result for custom_id {custom_id!r} has no text content block"
        )

    return _parse_seg_nodes(text_block.text)


def segment_documents_batch(
    items: list[SegmentationBatchItem],
    *,
    taxonomy_ids: list[str],
    client: Any = None,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 32000,
    cache: SegmentationVerdictCache | None = None,
    prompt_version: str = PROMPT_VERSION,
    effort: str = DEFAULT_EFFORT,
    poll_interval_s: float = _DEFAULT_POLL_INTERVAL_S,
    max_polls: int = _DEFAULT_MAX_POLLS,
    progress: Callable[[str], None] = lambda _: None,
) -> dict[str, list[SegNode]]:
    """Segment many documents in one Anthropic Message Batches call.

    Batch is 50% the cost of individual calls and supports the same
    structured-output + prompt-caching shape as :func:`segment_document`.

    When *cache* is given, each item's ``canonical_text`` is checked first;
    only cache misses are sent to the batch, and every freshly-segmented
    result is written back to the cache before returning. Passing no *cache*
    (the default) always calls the batch for every item — useful for callers
    that manage caching themselves, or tests exercising the batch mechanics
    in isolation.

    Args:
        items:           Documents to segment, one per ``custom_id``.
        taxonomy_ids:    Allowed taxonomy ids the model may assign (see
                         :mod:`playbook_engine.llm_segmenter`).
        client:          Injectable Anthropic client exposing
                         ``.messages.batches.create/.retrieve/.results``.
                         When ``None``, a real ``anthropic.Anthropic()``
                         client is constructed lazily.
        model:           Model id. Defaults to Claude Opus 4.8.
        max_tokens:      Passed through to each request's params.
        cache:           Optional :class:`SegmentationVerdictCache`.
        prompt_version:  Cache-key component; see ``SegmentationVerdictCache``.
        effort:          Cache-key component; the ``output_config.effort``
                         value used for every request in the batch.
        poll_interval_s: Seconds to sleep between ``.retrieve`` polls while
                         the batch is still processing. Tests pass ``0`` with
                         a fake client whose canned response is already
                         ``"ended"`` on the first poll, so no real sleep
                         occurs in the suite.
        max_polls:       Give up after this many ``.retrieve`` polls (see
                         :class:`BatchPollCapExceededError`). Defaults to a
                         ~24h ceiling at the default poll interval.
        progress:        Callable receiving a status line
                         (``processing_status`` + ``request_counts``) after
                         every poll, so a long batch is never silent.
                         Defaults to a no-op.

    Returns:
        ``{custom_id: [SegNode, ...]}`` — one entry per item in *items*,
        regardless of the order the Batches API returned results in.

    Raises:
        SegmentationLLMError: any batch result fails to parse (see
                               :func:`~playbook_engine.llm_segmenter.segment_document`
                               for the same per-document error contract).
        BatchPollCapExceededError: the batch is still non-terminal after
                               *max_polls* polls.
    """
    if client is None:
        import anthropic

        client = anthropic.Anthropic()

    resolved: dict[str, list[SegNode]] = {}
    to_submit: list[SegmentationBatchItem] = []

    for item in items:
        if cache is not None:
            hit = cache.get(
                item.canonical_text,
                model=model,
                prompt_version=prompt_version,
                effort=effort,
            )
            if hit is not None:
                resolved[item.custom_id] = hit
                continue
        to_submit.append(item)

    if not to_submit:
        return resolved

    requests = [
        _build_batch_request(item, taxonomy_ids=taxonomy_ids, model=model, max_tokens=max_tokens)
        for item in to_submit
    ]
    batch = client.messages.batches.create(requests=requests)
    batch_id = batch.id

    status = getattr(batch, "processing_status", None)
    polls = 0
    while status != _BATCH_ENDED_STATUS:
        polls += 1
        if polls > max_polls:
            raise BatchPollCapExceededError(
                f"batch {batch_id} still {status!r} after {polls - 1} polls "
                f"(cap {max_polls}); it has not been cancelled — re-attach to "
                f"batch_id={batch_id!r} via the Anthropic API/console once it "
                "completes rather than resubmitting these documents"
            )
        if poll_interval_s > 0:
            time.sleep(poll_interval_s)
        batch = client.messages.batches.retrieve(batch_id)
        status = getattr(batch, "processing_status", None)
        counts = getattr(batch, "request_counts", None)
        progress(f"  batch {batch_id}: {status} (request_counts={counts})")

    by_custom_id = {item.custom_id: item for item in to_submit}
    for entry in client.messages.batches.results(batch_id):
        custom_id = entry.custom_id
        seg_nodes = _extract_seg_nodes_from_result(custom_id, entry.result)
        resolved[custom_id] = seg_nodes
        if cache is not None and custom_id in by_custom_id:
            cache.put(
                by_custom_id[custom_id].canonical_text,
                seg_nodes,
                model=model,
                prompt_version=prompt_version,
                effort=effort,
            )

    return resolved


# ---------------------------------------------------------------------------
# normalize_trail — agreement-level cross-version consistency pass
# ---------------------------------------------------------------------------


class NormalizeTrailError(Exception):
    """Raised when the model's cross-version normalization response can't be
    parsed into a ``{version_id: {clause_path: taxonomy_id}}`` mapping.

    By design there is no fallback — a malformed response must not silently
    keep (or silently discard) the pre-normalization per-version labels.
    """


_NORMALIZE_TRAIL_SYSTEM_PROMPT = """\
You are reviewing one agreement's version trail. Each version was segmented
and classified independently, so the same clause lineage may have drifted to
different taxonomy labels across versions purely from independent per-version
calls (e.g. "indemnification" in v1 but "limitation_of_liability" in v3 for
what is clearly the same clause). You will be given, for each version, its
top-level and nested clause headings together with their current taxonomy_id
assignments.

Your job: propose a normalized taxonomy_id for every (version, clause_path)
pair so that the same clause lineage carries a consistent label across every
version in the trail. Only change a label when you are confident it is the
same clause type as elsewhere in the trail; when in doubt, keep the existing
label unchanged. Also flag any clause whose *boundaries* look inconsistent
across versions (e.g. one version merges two clauses that are split in
another) in `boundary_flags`, but do not attempt to re-segment — boundary
issues are for human review, not automatic repair here.
"""


def _normalize_trail_schema(taxonomy_ids: list[str]) -> dict[str, Any]:
    """Structured-output schema for the cross-version normalization response."""
    return {
        "type": "object",
        "properties": {
            "versions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "version_id": {"type": "string"},
                        "clauses": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "clause_path": {"type": "string"},
                                    "taxonomy_id": {
                                        "type": ["string", "null"],
                                        "enum": [*taxonomy_ids, None],
                                    },
                                },
                                "required": ["clause_path", "taxonomy_id"],
                                "additionalProperties": False,
                            },
                        },
                    },
                    "required": ["version_id", "clauses"],
                    "additionalProperties": False,
                },
            },
            "boundary_flags": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "version_id": {"type": "string"},
                        "clause_path": {"type": "string"},
                        "note": {"type": "string"},
                    },
                    "required": ["version_id", "clause_path", "note"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["versions", "boundary_flags"],
        "additionalProperties": False,
    }


def _serialize_trail_for_prompt(
    version_trees: dict[str, ClauseTree],
    taxonomy_by_version: dict[str, dict[str, str | None]],
) -> str:
    """Serialize each version's clause headings + current labels for the prompt."""
    lines: list[str] = []
    for vid, tree in version_trees.items():
        lines.append(f"[version {vid}]")
        labels = taxonomy_by_version.get(vid, {})
        for node in tree.all_nodes():
            path = node.clause_path or "?"
            tid = labels.get(path)
            lines.append(f"  {path}: heading={node.heading!r} taxonomy_id={tid!r}")
    return "\n".join(lines)


class NormalizeTrailResult:
    """Output of :func:`normalize_trail`.

    Attributes:
        taxonomy_by_version: Normalized ``{version_id: {clause_path:
                              taxonomy_id}}`` — the same shape as the input
                              *taxonomy_by_version*, with labels unified
                              across the trail.
        boundary_flags:      Human-review flags for clauses whose boundaries
                              look inconsistent across versions. Each is a
                              dict with ``version_id``/``clause_path``/``note``.
    """

    __slots__ = ("taxonomy_by_version", "boundary_flags")

    def __init__(
        self,
        taxonomy_by_version: dict[str, dict[str, str | None]],
        boundary_flags: list[dict[str, str]],
    ) -> None:
        self.taxonomy_by_version = taxonomy_by_version
        self.boundary_flags = boundary_flags


def _parse_normalize_trail_response(raw_json: str) -> NormalizeTrailResult:
    """Parse the model's JSON text into a ``NormalizeTrailResult``.

    Raises:
        NormalizeTrailError: on invalid JSON or a shape that doesn't match
                              the expected schema.
    """
    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        raise NormalizeTrailError(f"model response is not valid JSON: {exc}") from exc

    if not isinstance(data, dict) or "versions" not in data:
        raise NormalizeTrailError("model response missing top-level 'versions' key")

    versions_raw = data["versions"]
    if not isinstance(versions_raw, list):
        raise NormalizeTrailError("model response 'versions' must be a list")

    taxonomy_by_version: dict[str, dict[str, str | None]] = {}
    for i, v in enumerate(versions_raw):
        if not isinstance(v, dict) or "version_id" not in v or "clauses" not in v:
            raise NormalizeTrailError(f"versions[{i}] missing 'version_id' or 'clauses'")
        vid = v["version_id"]
        clauses_raw = v["clauses"]
        if not isinstance(clauses_raw, list):
            raise NormalizeTrailError(f"versions[{i}] 'clauses' must be a list")
        labels: dict[str, str | None] = {}
        for j, c in enumerate(clauses_raw):
            if not isinstance(c, dict) or "clause_path" not in c or "taxonomy_id" not in c:
                raise NormalizeTrailError(
                    f"versions[{i}].clauses[{j}] missing 'clause_path' or 'taxonomy_id'"
                )
            labels[c["clause_path"]] = c["taxonomy_id"]
        taxonomy_by_version[vid] = labels

    boundary_flags_raw = data.get("boundary_flags", [])
    if not isinstance(boundary_flags_raw, list):
        raise NormalizeTrailError("model response 'boundary_flags' must be a list")
    boundary_flags: list[dict[str, str]] = []
    for k, f in enumerate(boundary_flags_raw):
        if not isinstance(f, dict) or not all(
            key in f for key in ("version_id", "clause_path", "note")
        ):
            raise NormalizeTrailError(
                f"boundary_flags[{k}] missing 'version_id', 'clause_path', or 'note'"
            )
        boundary_flags.append(
            {"version_id": f["version_id"], "clause_path": f["clause_path"], "note": f["note"]}
        )

    return NormalizeTrailResult(
        taxonomy_by_version=taxonomy_by_version, boundary_flags=boundary_flags
    )


def normalize_trail(
    version_trees: dict[str, ClauseTree],
    taxonomy_by_version: dict[str, dict[str, str | None]],
    *,
    taxonomy_ids: list[str],
    client: Any = None,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 8000,
) -> NormalizeTrailResult:
    """Normalize taxonomy labels across every verified version of one agreement.

    One Opus call given the per-version clause headings + current labels;
    returns a normalized ``{version_id: {clause_path: taxonomy_id}}`` mapping
    plus any boundary-inconsistency flags for human review. Runs after
    per-version segmentation for an agreement — see
    :func:`~playbook_engine.pipeline.mine_corpus`.

    Args:
        version_trees:       Every verified version's ``ClauseTree`` for one
                              agreement, keyed by version id.
        taxonomy_by_version: Current per-version ``{clause_path: taxonomy_id}``
                              labels (e.g. the LLM segmenter's
                              ``taxonomy_by_path`` per version) to normalize.
        taxonomy_ids:        Allowed taxonomy ids the model may assign.
        client:              Injectable Anthropic client exposing
                              ``.messages.create(**kwargs)``. When ``None``,
                              a real ``anthropic.Anthropic()`` client is
                              constructed lazily.
        model:               Model id. Defaults to Claude Opus 4.8.
        max_tokens:          Passed through to ``messages.create``.

    Returns:
        :class:`NormalizeTrailResult` (normalized labels + boundary flags).

    Raises:
        NormalizeTrailError: the response isn't parseable JSON matching the
                              expected schema.
    """
    if client is None:
        import anthropic

        client = anthropic.Anthropic()

    schema = _normalize_trail_schema(taxonomy_ids)
    user_content = _serialize_trail_for_prompt(version_trees, taxonomy_by_version)

    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        thinking={"type": "adaptive"},
        output_config={
            "effort": "high",
            "format": {"type": "json_schema", "schema": schema},
        },
        system=[
            {
                "type": "text",
                "text": _NORMALIZE_TRAIL_SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": user_content}],
    )

    content = getattr(response, "content", None)
    if not content:
        raise NormalizeTrailError("model response has no content blocks")

    text_block = next(
        (b for b in content if getattr(b, "type", None) == "text"),
        None,
    )
    if text_block is None:
        raise NormalizeTrailError(
            "model response has no text content block "
            f"(block types: {[getattr(b, 'type', None) for b in content]!r})"
        )

    return _parse_normalize_trail_response(text_block.text)


#: Call shape every ``normalize_trail``-compatible callable must satisfy — lets
#: :mod:`playbook_engine.pipeline` inject a fake in tests, same DI pattern as
#: ``llm_segmentation_stage.SegmentFn``.
NormalizeTrailFn = Callable[
    [dict[str, ClauseTree], dict[str, "dict[str, str | None]"]], NormalizeTrailResult
]


__all__ = [
    "PROMPT_VERSION",
    "SCHEMA_HASH",
    "SegmentationVerdictCache",
    "SegmentationBatchItem",
    "segment_documents_batch",
    "NormalizeTrailError",
    "NormalizeTrailResult",
    "NormalizeTrailFn",
    "normalize_trail",
]
