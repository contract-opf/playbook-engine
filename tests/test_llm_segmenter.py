"""Tests for llm_segmenter — LLM segmenter core (issue #72).

All tests use a **fake** Anthropic client; no live API calls, no network,
no API key needed.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from playbook_engine.llm_segmenter import (
    DEFAULT_MODEL,
    SEGMENTER_SYSTEM_PROMPT,
    SegmentationLLMError,
    segment_document,
)
from playbook_engine.segmentation_grounding import Block, SegNode

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _blocks() -> list[Block]:
    return [
        Block(block_id="b0", page=0, char_span=(0, 10), text="Indemnification heading"),
        Block(block_id="b1", page=0, char_span=(11, 20), text="sub clause text"),
        Block(block_id="b2", page=0, char_span=(21, 30), text="/s/ Jane Doe DocuSign"),
    ]


class _FakeMessages:
    """Records the kwargs it was called with and returns a canned response."""

    def __init__(self, response_text: str) -> None:
        self.response_text = response_text
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=self.response_text)])


class _FakeClient:
    def __init__(self, response_text: str) -> None:
        self.messages = _FakeMessages(response_text)


_VALID_RESPONSE = json.dumps(
    {
        "nodes": [
            {
                "node_id": "n1",
                "parent_id": None,
                "order": 1,
                "heading": "Indemnification",
                "taxonomy_id": "indemnification",
                "start_block_id": "b0",
                "end_block_id": "b0",
                "start_quote": "Indemnification",
                "end_quote": "heading",
            },
            {
                "node_id": "n1a",
                "parent_id": "n1",
                "order": 1,
                "heading": None,
                "taxonomy_id": "indemnification",
                "start_block_id": "b1",
                "end_block_id": "b1",
                "start_quote": "sub",
                "end_quote": "text",
            },
            {
                "node_id": "n2",
                "parent_id": None,
                "order": 2,
                "heading": None,
                "taxonomy_id": None,
                "start_block_id": "b2",
                "end_block_id": "b2",
                "start_quote": "/s/",
                "end_quote": "DocuSign",
            },
        ]
    }
)


# ---------------------------------------------------------------------------
# Happy path — acceptance criterion 1
# ---------------------------------------------------------------------------


def test_returns_correct_seg_nodes_from_canned_response() -> None:
    client = _FakeClient(_VALID_RESPONSE)
    result = segment_document(
        "canonical text",
        _blocks(),
        ["indemnification", "governing_law"],
        client=client,
    )

    assert result == [
        SegNode(
            node_id="n1",
            parent_id=None,
            order=1,
            heading="Indemnification",
            taxonomy_id="indemnification",
            start_block_id="b0",
            end_block_id="b0",
            start_quote="Indemnification",
            end_quote="heading",
        ),
        SegNode(
            node_id="n1a",
            parent_id="n1",
            order=1,
            heading=None,
            taxonomy_id="indemnification",
            start_block_id="b1",
            end_block_id="b1",
            start_quote="sub",
            end_quote="text",
        ),
        SegNode(
            node_id="n2",
            parent_id=None,
            order=2,
            heading=None,
            taxonomy_id=None,
            start_block_id="b2",
            end_block_id="b2",
            start_quote="/s/",
            end_quote="DocuSign",
        ),
    ]


def test_root_child_and_null_taxonomy_node_types_are_correct() -> None:
    client = _FakeClient(_VALID_RESPONSE)
    result = segment_document("text", _blocks(), ["indemnification"], client=client)

    root, child, noise = result
    assert root.parent_id is None
    assert root.taxonomy_id == "indemnification"
    assert child.parent_id == "n1"
    assert child.order == 1
    assert noise.parent_id is None
    assert noise.taxonomy_id is None
    assert isinstance(root, SegNode)


# ---------------------------------------------------------------------------
# Request-shape assertions — acceptance criterion 2
# ---------------------------------------------------------------------------


def test_request_uses_expected_model_and_output_config() -> None:
    """No ``model=`` override -> the request uses the shared ``DEFAULT_MODEL``
    constant, not a private hardcoded literal (issue #131: this constant is
    also what ``config.SegmentationConfig.model`` defaults to, and what
    ``llm_segmenter_batch`` imports rather than redeclaring)."""
    client = _FakeClient(_VALID_RESPONSE)
    segment_document("text", _blocks(), ["indemnification"], client=client)

    assert len(client.messages.calls) == 1
    kwargs = client.messages.calls[0]

    assert kwargs["model"] == DEFAULT_MODEL
    assert kwargs["output_config"]["format"]["type"] == "json_schema"
    assert kwargs["output_config"]["effort"] == "high"
    assert kwargs["thinking"]["type"] == "adaptive"


def test_request_omits_temperature_and_budget_tokens() -> None:
    client = _FakeClient(_VALID_RESPONSE)
    segment_document("text", _blocks(), ["indemnification"], client=client)

    kwargs = client.messages.calls[0]
    assert "temperature" not in kwargs
    assert "top_p" not in kwargs
    assert "budget_tokens" not in kwargs
    assert "budget_tokens" not in kwargs.get("thinking", {})


def test_request_max_tokens_and_model_are_overridable() -> None:
    client = _FakeClient(_VALID_RESPONSE)
    segment_document(
        "text",
        _blocks(),
        ["indemnification"],
        client=client,
        model="claude-opus-4-7",
        max_tokens=4096,
    )

    kwargs = client.messages.calls[0]
    assert kwargs["model"] == "claude-opus-4-7"
    assert kwargs["max_tokens"] == 4096


def test_output_schema_embeds_allowed_taxonomy_ids() -> None:
    client = _FakeClient(_VALID_RESPONSE)
    segment_document("text", _blocks(), ["indemnification", "governing_law"], client=client)

    kwargs = client.messages.calls[0]
    schema = kwargs["output_config"]["format"]["schema"]
    node_schema = schema["properties"]["nodes"]["items"]
    taxonomy_enum = node_schema["properties"]["taxonomy_id"]["enum"]
    assert "indemnification" in taxonomy_enum
    assert "governing_law" in taxonomy_enum
    assert None in taxonomy_enum
    assert schema["additionalProperties"] is False
    assert set(node_schema["required"]) == {
        "node_id",
        "parent_id",
        "order",
        "heading",
        "taxonomy_id",
        "start_block_id",
        "end_block_id",
        "start_quote",
        "end_quote",
    }
    assert node_schema["additionalProperties"] is False


def test_system_prompt_includes_taxonomy_ids() -> None:
    client = _FakeClient(_VALID_RESPONSE)
    segment_document("text", _blocks(), ["indemnification"], client=client)

    kwargs = client.messages.calls[0]
    system_text = kwargs["system"][0]["text"]
    assert SEGMENTER_SYSTEM_PROMPT.strip() in system_text
    assert "indemnification" in system_text
    assert kwargs["system"][0]["cache_control"] == {"type": "ephemeral"}


def test_block_stream_is_serialized_with_block_ids() -> None:
    client = _FakeClient(_VALID_RESPONSE)
    segment_document("text", _blocks(), ["indemnification"], client=client)

    kwargs = client.messages.calls[0]
    user_content = kwargs["messages"][0]["content"]
    assert "[b0] Indemnification heading" in user_content
    assert "[b1] sub clause text" in user_content
    assert "[b2] /s/ Jane Doe DocuSign" in user_content


def test_repair_feedback_is_prepended_to_user_message_not_system_prompt() -> None:
    """``repair_feedback`` (e.g. ``str(SegmentationQAError)`` from a repair
    attempt) lands in the uncached user message, ahead of the block stream —
    the cached system prompt is untouched so repeat non-repair calls keep
    hitting its ephemeral cache."""
    client = _FakeClient(_VALID_RESPONSE)
    segment_document(
        "text",
        _blocks(),
        ["indemnification"],
        client=client,
        repair_feedback="coverage gate: gap [3812, 4102) before node '1' contains: 'oops'",
    )

    kwargs = client.messages.calls[0]
    user_content = kwargs["messages"][0]["content"]
    assert "coverage gate: gap [3812, 4102)" in user_content
    assert user_content.index("coverage gate") < user_content.index("[b0]")
    system_text = kwargs["system"][0]["text"]
    assert "coverage gate" not in system_text


def test_no_repair_feedback_leaves_user_message_unchanged() -> None:
    """Default ``repair_feedback=None`` — request shape is byte-identical to
    before the parameter existed."""
    client = _FakeClient(_VALID_RESPONSE)
    segment_document("text", _blocks(), ["indemnification"], client=client)

    kwargs = client.messages.calls[0]
    user_content = kwargs["messages"][0]["content"]
    assert (
        user_content
        == "[b0] Indemnification heading\n[b1] sub clause text\n[b2] /s/ Jane Doe DocuSign"
    )


# ---------------------------------------------------------------------------
# Malformed response — acceptance criterion 3
# ---------------------------------------------------------------------------


def test_invalid_json_raises_segmentation_llm_error() -> None:
    client = _FakeClient("not json at all {{{")
    with pytest.raises(SegmentationLLMError, match="not valid JSON"):
        segment_document("text", _blocks(), ["indemnification"], client=client)


def test_missing_nodes_key_raises() -> None:
    client = _FakeClient(json.dumps({"wrong_key": []}))
    with pytest.raises(SegmentationLLMError, match="missing top-level 'nodes' key"):
        segment_document("text", _blocks(), ["indemnification"], client=client)


def test_nodes_not_a_list_raises() -> None:
    client = _FakeClient(json.dumps({"nodes": "not a list"}))
    with pytest.raises(SegmentationLLMError, match="must be a list"):
        segment_document("text", _blocks(), ["indemnification"], client=client)


def test_node_missing_required_key_raises() -> None:
    bad_node = {
        "node_id": "n1",
        "parent_id": None,
        "order": 1,
        "heading": None,
        "taxonomy_id": None,
        "start_block_id": "b0",
        # end_block_id missing
    }
    client = _FakeClient(json.dumps({"nodes": [bad_node]}))
    with pytest.raises(SegmentationLLMError, match="missing required key"):
        segment_document("text", _blocks(), ["indemnification"], client=client)


def test_node_not_an_object_raises() -> None:
    client = _FakeClient(json.dumps({"nodes": ["not an object"]}))
    with pytest.raises(SegmentationLLMError, match="is not an object"):
        segment_document("text", _blocks(), ["indemnification"], client=client)


def test_empty_content_raises() -> None:
    client = _FakeClient.__new__(_FakeClient)
    client.messages = _FakeMessages(_VALID_RESPONSE)
    client.messages.create = lambda **kwargs: SimpleNamespace(content=[])  # type: ignore[method-assign]
    with pytest.raises(SegmentationLLMError, match="no content blocks"):
        segment_document("text", _blocks(), ["indemnification"], client=client)


def test_response_with_no_text_block_raises() -> None:
    """A response containing only non-text blocks (e.g. thinking, no answer)
    is malformed and must raise — there is no JSON to parse."""
    client = _FakeClient.__new__(_FakeClient)
    client.messages = _FakeMessages(_VALID_RESPONSE)
    client.messages.create = lambda **kwargs: SimpleNamespace(  # type: ignore[method-assign]
        content=[SimpleNamespace(type="thinking", thinking="...")]
    )
    with pytest.raises(SegmentationLLMError, match="no text content block"):
        segment_document("text", _blocks(), ["indemnification"], client=client)


def test_thinking_block_before_text_is_skipped_and_parsed() -> None:
    """Adaptive thinking prepends thinking block(s) before the answer; the
    text block after them is the structured output and must be parsed, not
    rejected. Regression guard for the content[0]-is-text assumption."""
    client = _FakeClient.__new__(_FakeClient)
    client.messages = _FakeMessages(_VALID_RESPONSE)
    client.messages.create = lambda **kwargs: SimpleNamespace(  # type: ignore[method-assign]
        content=[
            SimpleNamespace(type="thinking", thinking="deliberating..."),
            SimpleNamespace(type="text", text=_VALID_RESPONSE),
        ]
    )
    result = segment_document("text", _blocks(), ["indemnification"], client=client)
    assert [n.node_id for n in result] == ["n1", "n1a", "n2"]


# ---------------------------------------------------------------------------
# No live API calls — lazy client construction only on client=None
# ---------------------------------------------------------------------------


def test_empty_seg_nodes_list_is_valid() -> None:
    client = _FakeClient(json.dumps({"nodes": []}))
    result = segment_document("text", _blocks(), ["indemnification"], client=client)
    assert result == []
