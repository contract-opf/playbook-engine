"""Tests for llm_segmenter_batch — batch + cache + cross-version consistency (issue #75).

Acceptance criteria verified here:

  AC-1: With a mocked batches client, segment_documents_batch returns the
        correct {custom_id: [SegNode...]} mapping, keyed by custom_id — a
        test with shuffled result order still maps correctly.
  AC-2: The cache round-trips: a second call with the same (canonical_text,
        model, prompt_version, schema_hash, effort) returns the stored
        SegNodes without calling the client; a changed model/prompt_version
        misses and re-calls.
  AC-3: normalize_trail with a mocked client applies label normalization
        across versions — an inconsistent label gets unified per the mocked
        response.

All tests use a **fake** Anthropic client; no live API calls, no network,
no API key needed.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from playbook_engine.clause_tree import ClauseNode, ClauseTree
from playbook_engine.llm_segmenter import SegmentationLLMError
from playbook_engine.llm_segmenter_batch import (
    PROMPT_VERSION,
    BatchPollCapExceededError,
    NormalizeTrailError,
    SegmentationBatchItem,
    SegmentationVerdictCache,
    normalize_trail,
    segment_documents_batch,
)
from playbook_engine.segmentation_grounding import Block, SegNode

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _blocks(prefix: str = "b") -> list[Block]:
    return [
        Block(block_id=f"{prefix}0", page=0, char_span=(0, 10), text="Indemnification heading"),
        Block(block_id=f"{prefix}1", page=0, char_span=(11, 20), text="sub clause text"),
    ]


def _node(node_id: str = "n1", taxonomy_id: str | None = "indemnification") -> dict[str, Any]:
    return {
        "node_id": node_id,
        "parent_id": None,
        "order": 1,
        "heading": "Indemnification",
        "taxonomy_id": taxonomy_id,
        "start_block_id": "b0",
        "end_block_id": "b1",
        "start_quote": "Indemnification",
        "end_quote": "text",
    }


def _seg_response_text(node_id: str = "n1", taxonomy_id: str | None = "indemnification") -> str:
    return json.dumps({"nodes": [_node(node_id, taxonomy_id)]})


def _expected_node(node_id: str = "n1", taxonomy_id: str | None = "indemnification") -> SegNode:
    return SegNode(
        node_id=node_id,
        parent_id=None,
        order=1,
        heading="Indemnification",
        taxonomy_id=taxonomy_id,
        start_block_id="b0",
        end_block_id="b1",
        start_quote="Indemnification",
        end_quote="text",
    )


# ---------------------------------------------------------------------------
# Fake batches client
# ---------------------------------------------------------------------------


class _FakeBatchesResource:
    """Records .create()/.retrieve() calls; returns canned results keyed by custom_id.

    ``results_by_custom_id``: custom_id -> "succeeded" response text (JSON string
    matching the segmenter's structured-output shape), or a
    ``(result_type, detail)`` tuple to synthesize a non-succeeded result.

    ``result_order``: optional explicit ordering of custom_ids for ``.results()``
    — used to prove shuffled result order still maps correctly by custom_id.
    """

    def __init__(
        self,
        results_by_custom_id: dict[str, str | tuple[str, str]],
        *,
        result_order: list[str] | None = None,
        statuses: list[str] | None = None,
    ) -> None:
        self.results_by_custom_id = results_by_custom_id
        self.result_order = result_order or list(results_by_custom_id.keys())
        # Sequence of processing_status values returned by successive
        # .retrieve() calls; defaults to ended immediately (no polling delay).
        self._statuses = list(statuses) if statuses is not None else ["ended"]
        self.create_calls: list[dict[str, Any]] = []
        self.retrieve_calls: list[str] = []
        self.results_calls: list[str] = []

    def create(self, **kwargs: Any) -> Any:
        self.create_calls.append(kwargs)
        initial_status = self._statuses[0] if self._statuses else "ended"
        return SimpleNamespace(id="batch_123", processing_status=initial_status)

    def retrieve(self, batch_id: str) -> Any:
        self.retrieve_calls.append(batch_id)
        idx = len(self.retrieve_calls)
        status = self._statuses[idx] if idx < len(self._statuses) else "ended"
        return SimpleNamespace(id=batch_id, processing_status=status)

    def results(self, batch_id: str) -> list[Any]:
        self.results_calls.append(batch_id)
        out = []
        for custom_id in self.result_order:
            spec = self.results_by_custom_id[custom_id]
            if isinstance(spec, tuple):
                result_type, _detail = spec
                result = SimpleNamespace(type=result_type)
            else:
                result = SimpleNamespace(
                    type="succeeded",
                    message=SimpleNamespace(content=[SimpleNamespace(type="text", text=spec)]),
                )
            out.append(SimpleNamespace(custom_id=custom_id, result=result))
        return out


class _FakeMessagesWithBatches:
    def __init__(self, batches: _FakeBatchesResource, create_response_text: str = "") -> None:
        self.batches = batches
        self._create_response_text = create_response_text
        self.create_calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.create_calls.append(kwargs)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=self._create_response_text)]
        )


class _FakeClient:
    def __init__(self, batches: _FakeBatchesResource, create_response_text: str = "") -> None:
        self.messages = _FakeMessagesWithBatches(batches, create_response_text)


# ---------------------------------------------------------------------------
# AC-1: segment_documents_batch — correct custom_id mapping
# ---------------------------------------------------------------------------


class TestSegmentDocumentsBatch:
    def test_returns_seg_nodes_keyed_by_custom_id(self) -> None:
        batches = _FakeBatchesResource(
            {
                "doc-a": _seg_response_text("n1", "indemnification"),
                "doc-b": _seg_response_text("n2", "governing_law"),
            }
        )
        client = _FakeClient(batches)
        items = [
            SegmentationBatchItem("doc-a", "text a", _blocks("a")),
            SegmentationBatchItem("doc-b", "text b", _blocks("b")),
        ]

        result = segment_documents_batch(
            items,
            taxonomy_ids=["indemnification", "governing_law"],
            client=client,
            poll_interval_s=0,
        )

        assert set(result.keys()) == {"doc-a", "doc-b"}
        assert result["doc-a"] == [_expected_node("n1", "indemnification")]
        assert result["doc-b"] == [_expected_node("n2", "governing_law")]

    def test_shuffled_result_order_still_maps_correctly(self) -> None:
        """Batch results are unordered — must key by custom_id, never position."""
        batches = _FakeBatchesResource(
            {
                "doc-a": _seg_response_text("n1", "indemnification"),
                "doc-b": _seg_response_text("n2", "governing_law"),
                "doc-c": _seg_response_text("n3", None),
            },
            # Deliberately NOT in submission order.
            result_order=["doc-c", "doc-a", "doc-b"],
        )
        client = _FakeClient(batches)
        items = [
            SegmentationBatchItem("doc-a", "text a", _blocks("a")),
            SegmentationBatchItem("doc-b", "text b", _blocks("b")),
            SegmentationBatchItem("doc-c", "text c", _blocks("c")),
        ]

        result = segment_documents_batch(
            items,
            taxonomy_ids=["indemnification", "governing_law"],
            client=client,
            poll_interval_s=0,
        )

        assert result["doc-a"] == [_expected_node("n1", "indemnification")]
        assert result["doc-b"] == [_expected_node("n2", "governing_law")]
        assert result["doc-c"] == [_expected_node("n3", None)]

    def test_polls_retrieve_until_ended(self) -> None:
        """A batch not immediately ended is polled via .retrieve() until it is."""
        batches = _FakeBatchesResource(
            {"doc-a": _seg_response_text()},
            statuses=["in_progress", "in_progress", "ended"],
        )
        client = _FakeClient(batches)
        items = [SegmentationBatchItem("doc-a", "text a", _blocks("a"))]

        result = segment_documents_batch(
            items, taxonomy_ids=["indemnification"], client=client, poll_interval_s=0
        )

        assert result["doc-a"] == [_expected_node()]
        assert len(batches.retrieve_calls) == 2  # polled twice before "ended"

    def test_progress_callback_reports_status_on_every_poll(self) -> None:
        """Long batches must not be silent — issue #98."""
        batches = _FakeBatchesResource(
            {"doc-a": _seg_response_text()},
            statuses=["in_progress", "in_progress", "ended"],
        )
        client = _FakeClient(batches)
        items = [SegmentationBatchItem("doc-a", "text a", _blocks("a"))]
        lines: list[str] = []

        segment_documents_batch(
            items,
            taxonomy_ids=["indemnification"],
            client=client,
            poll_interval_s=0,
            progress=lines.append,
        )

        # One progress line per poll (2 polls before "ended"), each naming the
        # batch and its current status so a stuck run is distinguishable from
        # a slow one.
        assert len(lines) == 2
        assert all("batch_123" in line for line in lines)
        assert "in_progress" in lines[0]
        assert "ended" in lines[1]

    def test_poll_cap_exceeded_raises_without_looping_forever(self) -> None:
        """A batch that never reaches 'ended' must fail loudly, not poll forever."""
        batches = _FakeBatchesResource(
            {"doc-a": _seg_response_text()},
            statuses=["in_progress"] * 10,  # never ends within the test's cap
        )
        client = _FakeClient(batches)
        items = [SegmentationBatchItem("doc-a", "text a", _blocks("a"))]

        with pytest.raises(BatchPollCapExceededError, match="batch_123"):
            segment_documents_batch(
                items,
                taxonomy_ids=["indemnification"],
                client=client,
                poll_interval_s=0,
                max_polls=3,
            )

        # Gave up at the cap rather than looping until the fake's canned
        # statuses ran out and defaulted to "ended".
        assert len(batches.retrieve_calls) == 3

    def test_request_shape_has_custom_id_and_params(self) -> None:
        batches = _FakeBatchesResource({"doc-a": _seg_response_text()})
        client = _FakeClient(batches)
        items = [SegmentationBatchItem("doc-a", "text a", _blocks("a"))]

        segment_documents_batch(
            items, taxonomy_ids=["indemnification"], client=client, poll_interval_s=0
        )

        assert len(batches.create_calls) == 1
        requests = batches.create_calls[0]["requests"]
        assert len(requests) == 1
        assert requests[0]["custom_id"] == "doc-a"
        params = requests[0]["params"]
        assert params["model"] == "claude-opus-4-8"
        assert params["output_config"]["format"]["type"] == "json_schema"
        assert "[a0] Indemnification heading" in params["messages"][0]["content"]

    def test_non_succeeded_result_raises(self) -> None:
        batches = _FakeBatchesResource({"doc-a": ("errored", "rate limited")})
        client = _FakeClient(batches)
        items = [SegmentationBatchItem("doc-a", "text a", _blocks("a"))]

        with pytest.raises(SegmentationLLMError, match="did not succeed"):
            segment_documents_batch(
                items, taxonomy_ids=["indemnification"], client=client, poll_interval_s=0
            )

    def test_empty_items_returns_empty_mapping_without_calling_client(self) -> None:
        batches = _FakeBatchesResource({})
        client = _FakeClient(batches)

        result = segment_documents_batch(
            [], taxonomy_ids=["indemnification"], client=client, poll_interval_s=0
        )

        assert result == {}
        assert batches.create_calls == []


# ---------------------------------------------------------------------------
# AC-2: SegmentationVerdictCache — round-trip + model/prompt_version miss
# ---------------------------------------------------------------------------


class TestSegmentationVerdictCache:
    def test_miss_returns_none(self, tmp_path: Path) -> None:
        cache = SegmentationVerdictCache(tmp_path / "seg_cache.jsonl")
        assert cache.get("some text", model="claude-opus-4-8") is None

    def test_put_then_get_round_trips(self, tmp_path: Path) -> None:
        cache = SegmentationVerdictCache(tmp_path / "seg_cache.jsonl")
        nodes = [_expected_node("n1", "indemnification")]
        cache.put("some text", nodes, model="claude-opus-4-8")

        assert cache.get("some text", model="claude-opus-4-8") == nodes

    def test_persists_across_instances(self, tmp_path: Path) -> None:
        path = tmp_path / "seg_cache.jsonl"
        nodes = [_expected_node()]
        cache1 = SegmentationVerdictCache(path)
        cache1.put("doc text", nodes, model="claude-opus-4-8")

        cache2 = SegmentationVerdictCache(path)
        assert cache2.get("doc text", model="claude-opus-4-8") == nodes

    def test_changed_model_misses(self, tmp_path: Path) -> None:
        cache = SegmentationVerdictCache(tmp_path / "seg_cache.jsonl")
        nodes = [_expected_node()]
        cache.put("doc text", nodes, model="claude-opus-4-8")

        assert cache.get("doc text", model="claude-opus-4-7") is None

    def test_changed_prompt_version_misses(self, tmp_path: Path) -> None:
        cache = SegmentationVerdictCache(tmp_path / "seg_cache.jsonl")
        nodes = [_expected_node()]
        cache.put("doc text", nodes, model="claude-opus-4-8", prompt_version="v1")

        assert cache.get("doc text", model="claude-opus-4-8", prompt_version="v2") is None

    def test_changed_schema_hash_misses(self, tmp_path: Path) -> None:
        cache = SegmentationVerdictCache(tmp_path / "seg_cache.jsonl")
        nodes = [_expected_node()]
        cache.put("doc text", nodes, model="claude-opus-4-8", schema_hash="hash-a")

        assert cache.get("doc text", model="claude-opus-4-8", schema_hash="hash-b") is None

    def test_changed_effort_misses(self, tmp_path: Path) -> None:
        cache = SegmentationVerdictCache(tmp_path / "seg_cache.jsonl")
        nodes = [_expected_node()]
        cache.put("doc text", nodes, model="claude-opus-4-8", effort="high")

        assert cache.get("doc text", model="claude-opus-4-8", effort="low") is None

    def test_default_prompt_version_and_schema_hash_are_module_constants(
        self, tmp_path: Path
    ) -> None:
        """Calling get/put without prompt_version/schema_hash uses the module's
        own PROMPT_VERSION/SCHEMA_HASH constants — a round trip with no
        explicit override still hits."""
        cache = SegmentationVerdictCache(tmp_path / "seg_cache.jsonl")
        nodes = [_expected_node()]
        cache.put("doc text", nodes, model="claude-opus-4-8")
        assert cache.get("doc text", model="claude-opus-4-8", prompt_version=PROMPT_VERSION) == (
            nodes
        )


class TestSegmentDocumentsBatchWithCache:
    def test_second_call_hits_cache_without_calling_client(self, tmp_path: Path) -> None:
        """A second segment_documents_batch call with the same cache and the
        same canonical_text must not call the client at all (full cache hit)."""
        cache = SegmentationVerdictCache(tmp_path / "seg_cache.jsonl")
        batches = _FakeBatchesResource({"doc-a": _seg_response_text()})
        client = _FakeClient(batches)
        items = [SegmentationBatchItem("doc-a", "shared text", _blocks("a"))]

        first = segment_documents_batch(
            items,
            taxonomy_ids=["indemnification"],
            client=client,
            cache=cache,
            poll_interval_s=0,
        )
        assert first["doc-a"] == [_expected_node()]
        assert len(batches.create_calls) == 1

        second = segment_documents_batch(
            items,
            taxonomy_ids=["indemnification"],
            client=client,
            cache=cache,
            poll_interval_s=0,
        )
        assert second["doc-a"] == [_expected_node()]
        # No new batch submitted — the cache satisfied the whole request.
        assert len(batches.create_calls) == 1

    def test_partial_cache_hit_only_submits_misses(self, tmp_path: Path) -> None:
        """One cached item + one new item: only the new item goes to the batch."""
        cache = SegmentationVerdictCache(tmp_path / "seg_cache.jsonl")
        cache.put("cached text", [_expected_node("n-cached")], model="claude-opus-4-8")
        batches = _FakeBatchesResource({"doc-new": _seg_response_text("n-new")})
        client = _FakeClient(batches)
        items = [
            SegmentationBatchItem("doc-cached", "cached text", _blocks("a")),
            SegmentationBatchItem("doc-new", "new text", _blocks("b")),
        ]

        result = segment_documents_batch(
            items,
            taxonomy_ids=["indemnification"],
            client=client,
            cache=cache,
            poll_interval_s=0,
        )

        assert result["doc-cached"] == [_expected_node("n-cached")]
        assert result["doc-new"] == [_expected_node("n-new")]
        # Only the cache-miss item was submitted to the batch.
        submitted_ids = {r["custom_id"] for r in batches.create_calls[0]["requests"]}
        assert submitted_ids == {"doc-new"}

    def test_changed_model_forces_recall_via_cache_miss(self, tmp_path: Path) -> None:
        cache = SegmentationVerdictCache(tmp_path / "seg_cache.jsonl")
        batches = _FakeBatchesResource({"doc-a": _seg_response_text()})
        client = _FakeClient(batches)
        items = [SegmentationBatchItem("doc-a", "shared text", _blocks("a"))]

        segment_documents_batch(
            items,
            taxonomy_ids=["indemnification"],
            client=client,
            cache=cache,
            model="claude-opus-4-8",
            poll_interval_s=0,
        )
        assert len(batches.create_calls) == 1

        segment_documents_batch(
            items,
            taxonomy_ids=["indemnification"],
            client=client,
            cache=cache,
            model="claude-opus-4-7",
            poll_interval_s=0,
        )
        # Different model -> cache miss -> a second batch call was made.
        assert len(batches.create_calls) == 2

    def test_fresh_results_are_written_back_to_cache(self, tmp_path: Path) -> None:
        cache_path = tmp_path / "seg_cache.jsonl"
        cache = SegmentationVerdictCache(cache_path)
        batches = _FakeBatchesResource({"doc-a": _seg_response_text()})
        client = _FakeClient(batches)
        items = [SegmentationBatchItem("doc-a", "shared text", _blocks("a"))]

        segment_documents_batch(
            items,
            taxonomy_ids=["indemnification"],
            client=client,
            cache=cache,
            poll_interval_s=0,
        )

        # A fresh cache instance pointed at the same file sees the write.
        cache2 = SegmentationVerdictCache(cache_path)
        assert cache2.get("shared text", model="claude-opus-4-8") == [_expected_node()]


# ---------------------------------------------------------------------------
# AC-3: normalize_trail — label unification across versions
# ---------------------------------------------------------------------------


def _tree(version: str, headings: list[tuple[str, str]]) -> ClauseTree:
    """Build a minimal single-level ClauseTree; headings is [(path, heading), ...]."""
    nodes = [
        ClauseNode(clause_path=path, heading=heading, text=heading, char_span=(0, len(heading)))
        for path, heading in headings
    ]
    return ClauseTree(
        document_id="deal-001", version=version, source_file=f"{version}.docx", nodes=nodes
    )


class TestNormalizeTrail:
    def test_unifies_inconsistent_label_per_mocked_response(self) -> None:
        """v1 labels clause '1' as indemnification; v3 (independently
        segmented) labeled the same clause limitation_of_liability. The
        mocked response unifies both to indemnification."""
        version_trees = {
            "v1": _tree("v1", [("1", "Indemnification")]),
            "v3": _tree("v3", [("1", "Indemnification (Revised)")]),
        }
        taxonomy_by_version = {
            "v1": {"1": "indemnification"},
            "v3": {"1": "limitation_of_liability"},  # drifted label — same clause
        }
        mocked_response = json.dumps(
            {
                "versions": [
                    {
                        "version_id": "v1",
                        "clauses": [{"clause_path": "1", "taxonomy_id": "indemnification"}],
                    },
                    {
                        "version_id": "v3",
                        "clauses": [{"clause_path": "1", "taxonomy_id": "indemnification"}],
                    },
                ],
                "boundary_flags": [],
            }
        )
        client = _FakeClient(_FakeBatchesResource({}), create_response_text=mocked_response)

        result = normalize_trail(
            version_trees,
            taxonomy_by_version,
            taxonomy_ids=["indemnification", "limitation_of_liability"],
            client=client,
        )

        assert result.taxonomy_by_version["v1"]["1"] == "indemnification"
        assert result.taxonomy_by_version["v3"]["1"] == "indemnification"  # unified
        assert result.boundary_flags == []

    def test_boundary_flags_are_surfaced(self) -> None:
        version_trees = {"v1": _tree("v1", [("1", "Indemnification")])}
        taxonomy_by_version = {"v1": {"1": "indemnification"}}
        mocked_response = json.dumps(
            {
                "versions": [
                    {
                        "version_id": "v1",
                        "clauses": [{"clause_path": "1", "taxonomy_id": "indemnification"}],
                    }
                ],
                "boundary_flags": [
                    {
                        "version_id": "v1",
                        "clause_path": "1",
                        "note": "merges two clauses split in v2",
                    }
                ],
            }
        )
        client = _FakeClient(_FakeBatchesResource({}), create_response_text=mocked_response)

        result = normalize_trail(
            version_trees,
            taxonomy_by_version,
            taxonomy_ids=["indemnification"],
            client=client,
        )

        assert len(result.boundary_flags) == 1
        assert result.boundary_flags[0]["note"] == "merges two clauses split in v2"

    def test_request_includes_headings_and_current_labels(self) -> None:
        version_trees = {"v1": _tree("v1", [("1", "Indemnification")])}
        taxonomy_by_version = {"v1": {"1": "indemnification"}}
        mocked_response = json.dumps(
            {
                "versions": [
                    {
                        "version_id": "v1",
                        "clauses": [{"clause_path": "1", "taxonomy_id": "indemnification"}],
                    }
                ],
                "boundary_flags": [],
            }
        )
        client = _FakeClient(_FakeBatchesResource({}), create_response_text=mocked_response)

        normalize_trail(
            version_trees,
            taxonomy_by_version,
            taxonomy_ids=["indemnification"],
            client=client,
        )

        kwargs = client.messages.create_calls[0]
        user_content = kwargs["messages"][0]["content"]
        assert "v1" in user_content
        assert "Indemnification" in user_content
        assert "indemnification" in user_content
        assert kwargs["output_config"]["format"]["type"] == "json_schema"

    def test_invalid_json_raises_normalize_trail_error(self) -> None:
        client = _FakeClient(_FakeBatchesResource({}), create_response_text="not json {{{")

        with pytest.raises(NormalizeTrailError, match="not valid JSON"):
            normalize_trail(
                {"v1": _tree("v1", [("1", "X")])},
                {"v1": {"1": "indemnification"}},
                taxonomy_ids=["indemnification"],
                client=client,
            )

    def test_missing_versions_key_raises(self) -> None:
        client = _FakeClient(
            _FakeBatchesResource({}), create_response_text=json.dumps({"wrong_key": []})
        )

        with pytest.raises(NormalizeTrailError, match="missing top-level 'versions' key"):
            normalize_trail(
                {"v1": _tree("v1", [("1", "X")])},
                {"v1": {"1": "indemnification"}},
                taxonomy_ids=["indemnification"],
                client=client,
            )

    def test_empty_content_raises(self) -> None:
        client = _FakeClient.__new__(_FakeClient)
        client.messages = _FakeMessagesWithBatches(_FakeBatchesResource({}))
        client.messages.create = lambda **kwargs: SimpleNamespace(content=[])  # type: ignore[method-assign]

        with pytest.raises(NormalizeTrailError, match="no content blocks"):
            normalize_trail(
                {"v1": _tree("v1", [("1", "X")])},
                {"v1": {"1": "indemnification"}},
                taxonomy_ids=["indemnification"],
                client=client,
            )
