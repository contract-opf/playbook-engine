"""LLM segmenter core — LLM-first segmentation (issue #72).

Calls **Claude Opus 4.8** with structured output to turn a document's
canonical text + :class:`~playbook_engine.segmentation_grounding.Block` stream
into a clause tree, expressed as a flat list of
:class:`~playbook_engine.segmentation_grounding.SegNode`. Segmentation and
taxonomy classification happen in one pass: each returned node carries both
its block-range boundaries and its ``taxonomy_id`` (or ``null`` for non-clause
noise such as signature/DocuSign stamps).

The model is never trusted for character offsets or verbatim clause text —
only block references (``start_block_id``/``end_block_id``) and short
boundary quotes used downstream to verify those references. See
:mod:`playbook_engine.segmentation_grounding` for the deterministic grounding
step that turns this module's output into a
:class:`~playbook_engine.clause_tree.ClauseTree`.

This module makes no live API calls in tests: callers inject a fake
``client`` object exposing ``.messages.create(**kwargs)``; ``segment_document``
only constructs a real ``anthropic.Anthropic()`` client when ``client is None``.
"""

from __future__ import annotations

import json
from typing import Any

from playbook_engine.segmentation_grounding import Block, SegNode

# ---------------------------------------------------------------------------
# Error
# ---------------------------------------------------------------------------


class SegmentationLLMError(Exception):
    """Raised when the model's response cannot be parsed into ``SegNode``s.

    Covers a response body that isn't valid JSON and JSON that doesn't match
    the ``CLAUSE_TREE_SCHEMA`` shape (missing/extra keys, wrong types). By
    design there is no fallback — the caller surfaces this for review rather
    than grounding a malformed segmentation.
    """


#: Default segmentation model id — the single source of truth for every
#: ``model=`` default across the LLM-segmentation surface
#: (:func:`segment_document` here; :func:`~playbook_engine.llm_segmenter_batch.segment_documents_batch`
#: and :func:`~playbook_engine.llm_segmenter_batch.normalize_trail`, which
#: import this constant rather than redeclaring it — see issue #131, which
#: replaced two independently-hardcoded ``"claude-opus-4-8"`` literals with
#: this one constant). Also consumed by :mod:`playbook_engine.pipeline` for
#: its L1-L4 stage-cache config fingerprint: bumping this constant must
#: invalidate every cached doc, the same way bumping ``PROMPT_VERSION``/
#: ``SCHEMA_HASH`` (in ``llm_segmenter_batch``) does. Overridable per corpus
#: via ``config.segmentation.model`` (see :mod:`playbook_engine.config`) —
#: this constant is only the *default* when a config omits that field.
DEFAULT_MODEL = "claude-opus-4-8"


# ---------------------------------------------------------------------------
# Structured-output schema (mirrors SegNode — see segmentation_grounding.py)
# ---------------------------------------------------------------------------


def _node_schema(taxonomy_ids: list[str]) -> dict[str, Any]:
    """Build the per-node schema, inlining the allowed ``taxonomy_id`` enum."""
    return {
        "type": "object",
        "properties": {
            "node_id": {
                "type": "string",
                "description": "Unique id for this node, e.g. 'n1', 'n2a'.",
            },
            "parent_id": {
                "type": ["string", "null"],
                "description": ("node_id of the parent clause, or null for a root-level clause."),
            },
            "order": {
                "type": "integer",
                "description": "1-based position among this node's siblings.",
            },
            "heading": {
                "type": ["string", "null"],
                "description": "The clause's heading text as it appears in the source, or null.",
            },
            "taxonomy_id": {
                "type": ["string", "null"],
                "enum": [*taxonomy_ids, None],
                "description": (
                    "One of the allowed taxonomy ids, or null for non-clause noise "
                    "(e.g. signature/DocuSign stamps) that isn't a substantive clause."
                ),
            },
            "start_block_id": {
                "type": "string",
                "description": "block_id of the first block in this clause's range.",
            },
            "end_block_id": {
                "type": "string",
                "description": "block_id of the last block in this clause's range.",
            },
            "start_quote": {
                "type": "string",
                "description": "Short verbatim quote from the start of the clause's text.",
            },
            "end_quote": {
                "type": "string",
                "description": "Short verbatim quote from the end of the clause's text.",
            },
        },
        "required": [
            "node_id",
            "parent_id",
            "order",
            "heading",
            "taxonomy_id",
            "start_block_id",
            "end_block_id",
            "start_quote",
            "end_quote",
        ],
        "additionalProperties": False,
    }


def _clause_tree_schema(taxonomy_ids: list[str]) -> dict[str, Any]:
    """Build the full structured-output schema for a given taxonomy id list."""
    return {
        "type": "object",
        "properties": {
            "nodes": {
                "type": "array",
                "items": _node_schema(taxonomy_ids),
            },
        },
        "required": ["nodes"],
        "additionalProperties": False,
    }


#: Schema shape matching :class:`~playbook_engine.segmentation_grounding.SegNode`
#: for a document with no taxonomy ids (i.e. only ``null`` is allowed). Callers
#: that need the real allowed-id enum use ``_clause_tree_schema(taxonomy_ids)``
#: internally via :func:`segment_document`; this module-level constant exists
#: so the schema's static shape (object → ``nodes: [...]``, per-node fields,
#: ``additionalProperties: false``) can be inspected/tested independent of any
#: particular taxonomy.
CLAUSE_TREE_SCHEMA: dict[str, Any] = _clause_tree_schema([])


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------


SEGMENTER_SYSTEM_PROMPT = """\
You are a legal-document segmenter. You will be given a numbered stream of
text blocks extracted from a contract, in reading order. Your job is to
split that block stream into clauses and return each clause as a node
referencing the blocks it spans.

Rules:
- Reference each clause only by `start_block_id` and `end_block_id` — the
  block ids at the start and end of its range. NEVER invent character
  offsets or positions; you only ever see and reference block ids.
- `start_quote` and `end_quote` must be short, exact, VERBATIM excerpts from
  the source blocks — copy them character-for-character. Do NOT paraphrase,
  summarize, or correct spelling/punctuation anywhere in your output.
- Nest sub-clauses under their parent using `parent_id`. Top-level clauses
  have `parent_id: null`. Use `order` for this node's 1-based position among
  its siblings (siblings sharing a `parent_id`).
- Assign `taxonomy_id` from the allowed list below. If a block is not a
  substantive clause — boilerplate signature blocks, DocuSign/e-signature
  stamps, page headers/footers, or other non-clause noise — still emit a
  node for it (so every block is accounted for), but set `taxonomy_id` to
  null.
- Every block in the input must be covered by exactly one node's block
  range. Do not skip blocks and do not let ranges overlap.
"""


def _system_prompt_with_taxonomy(taxonomy_ids: list[str]) -> str:
    """Append the allowed taxonomy id list to the base system prompt."""
    ids_text = ", ".join(taxonomy_ids) if taxonomy_ids else "(none — use null for every node)"
    return f"{SEGMENTER_SYSTEM_PROMPT}\nAllowed taxonomy ids: {ids_text}"


# ---------------------------------------------------------------------------
# Block-stream serialization
# ---------------------------------------------------------------------------


def _serialize_blocks(blocks: list[Block]) -> str:
    """Serialize the block stream as ``"[b0] <text>\\n[b1] <text>…"``."""
    return "\n".join(f"[{b.block_id}] {b.text}" for b in blocks)


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def _parse_seg_nodes(raw_json: str) -> list[SegNode]:
    """Parse the model's JSON text into ``SegNode`` objects.

    Raises:
        SegmentationLLMError: on invalid JSON or a shape that doesn't match
                               the expected ``{"nodes": [...]}`` structure.
    """
    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        raise SegmentationLLMError(f"model response is not valid JSON: {exc}") from exc

    if not isinstance(data, dict) or "nodes" not in data:
        raise SegmentationLLMError("model response missing top-level 'nodes' key")

    nodes_raw = data["nodes"]
    if not isinstance(nodes_raw, list):
        raise SegmentationLLMError("model response 'nodes' must be a list")

    required_keys = (
        "node_id",
        "parent_id",
        "order",
        "heading",
        "taxonomy_id",
        "start_block_id",
        "end_block_id",
    )

    seg_nodes: list[SegNode] = []
    for i, raw in enumerate(nodes_raw):
        if not isinstance(raw, dict):
            raise SegmentationLLMError(f"nodes[{i}] is not an object")
        missing = [k for k in required_keys if k not in raw]
        if missing:
            raise SegmentationLLMError(f"nodes[{i}] missing required key(s): {missing}")
        try:
            seg_nodes.append(
                SegNode(
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
            )
        except (TypeError, ValueError) as exc:
            raise SegmentationLLMError(f"nodes[{i}] does not match SegNode shape: {exc}") from exc

    return seg_nodes


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def segment_document(
    canonical_text: str,
    blocks: list[Block],
    taxonomy_ids: list[str],
    *,
    client: Any = None,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 32000,
    repair_feedback: str | None = None,
) -> list[SegNode]:
    """Segment + classify a document in one LLM pass.

    Args:
        canonical_text: The document's full canonical text (unused directly
                         in the request — the block stream is what's sent —
                         but accepted per the extractor/segmenter contract and
                         reserved for future prompt-shaping use).
        blocks:          The document's block stream in reading order (see
                         :mod:`playbook_engine.segmentation_grounding`).
        taxonomy_ids:    Allowed taxonomy ids the model may assign (plus
                         implicit ``null`` for non-clause noise).
        client:          Injectable Anthropic client exposing
                         ``.messages.create(**kwargs)``. When ``None`` (the
                         default), a real ``anthropic.Anthropic()`` client is
                         constructed lazily — tests always inject a fake
                         client so no network call or API key is needed.
        model:           Model id. Defaults to Claude Opus 4.8.
        max_tokens:       Passed through to ``messages.create``.
        repair_feedback: Optional failure detail from a previous QA-gate
                         failure (typically ``str(SegmentationQAError)``,
                         e.g. from :func:`~playbook_engine.segmentation_qa.segment_verify_repair`
                         re-invoking this function on a repair attempt).
                         When given, prepended to the user message so a
                         retry's prompt names the specific problem to fix
                         instead of re-sending byte-identical input.
                         Defaults to ``None`` (first attempt / no feedback
                         to thread) — request shape is unchanged from
                         before this parameter existed.

    Returns:
        The model's segmentation as a list of ``SegNode`` (block-anchored;
        not yet grounded to the source — see ``ground_segmentation``).

    Raises:
        SegmentationLLMError: the response isn't parseable JSON matching the
                               expected schema.
    """
    del canonical_text  # not sent directly; block stream carries the text

    if client is None:
        import anthropic

        client = anthropic.Anthropic()

    schema = _clause_tree_schema(taxonomy_ids)
    system_prompt = _system_prompt_with_taxonomy(taxonomy_ids)
    user_content = _serialize_blocks(blocks)
    if repair_feedback is not None:
        # Folded into the (uncached) user message, not the cached system
        # prompt, so repeat calls with no feedback keep hitting the system
        # prompt's ephemeral cache — only repair attempts pay for a miss.
        user_content = (
            "Your previous segmentation of this document failed automated QA "
            f"with this error — fix this specific issue:\n{repair_feedback}\n\n"
            f"{user_content}"
        )

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
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": user_content}],
    )

    content = getattr(response, "content", None)
    if not content:
        raise SegmentationLLMError("model response has no content blocks")

    # With adaptive thinking enabled, the response content may begin with one
    # or more thinking blocks before the answer. The structured-output JSON is
    # carried in the first (and only) text block — find it, skipping any
    # leading thinking/other blocks. Do NOT assume content[0] is the text.
    text_block = next(
        (b for b in content if getattr(b, "type", None) == "text"),
        None,
    )
    if text_block is None:
        raise SegmentationLLMError(
            "model response has no text content block "
            f"(block types: {[getattr(b, 'type', None) for b in content]!r})"
        )

    return _parse_seg_nodes(text_block.text)
