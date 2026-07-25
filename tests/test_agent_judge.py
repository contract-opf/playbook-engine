"""Tests for the agent-as-judge bridge core — issue #64.

Acceptance criteria verified here:

  AC-1: A store hit returns the stored verdict as the correct dataclass.
  AC-2: A miss records exactly one pending entry and returns the needs-review sentinel.
  AC-3: Duplicate payloads within a single batch produce exactly one pending-queue entry.
  AC-4: VerdictStore round-trips verdicts across separate instances pointed at the same file.
  AC-5: The pending payload contains the full clause text (assert len > 500 for a long fixture).
  AC-6: The three judges are drop-in for mine_corpus — signatures match the protocols exactly.

SECURITY NOTE: All fixtures use programmatically constructed synthetic content.
No real agreement files or real party names are used.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from playbook_engine.agent_judge import (
    PendingQueue,
    ScopeNeedsReviewError,
    StoreBackedClassificationJudge,
    StoreBackedDeviationJudge,
    StoreBackedProvenanceJudge,
    StoreBackedScopeJudge,
    VerdictStore,
    _payload_key,
)
from playbook_engine.clause_classifier import ClassificationJudge
from playbook_engine.clause_tree import ClauseNode, ClauseTree
from playbook_engine.config import AgreementType
from playbook_engine.deviation_classifier import DeviationJudge
from playbook_engine.provenance_detector import ProvenanceJudge
from playbook_engine.scope_gate import ScopeJudge

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_node(
    heading: str,
    text: str,
    clause_path: str = "1",
) -> ClauseNode:
    """Build a minimal ClauseNode with required fields."""
    return ClauseNode(
        heading=heading,
        text=text,
        clause_path=clause_path,
        char_span=(0, len(text)),
    )


@dataclass
class _FakeTaxonomy:
    entries: list[Any]


@dataclass
class _FakeTaxEntry:
    id: str


def _taxonomy(*ids: str) -> _FakeTaxonomy:
    return _FakeTaxonomy(entries=[_FakeTaxEntry(id=i) for i in ids])


def _make_store_and_pending(tmp_path: Path) -> tuple[VerdictStore, PendingQueue]:
    store = VerdictStore(tmp_path / "judge" / "verdicts.jsonl")
    pending = PendingQueue(tmp_path / "judge" / "pending.jsonl")
    return store, pending


def _make_tree(document_id: str, headings: list[str]) -> ClauseTree:
    nodes = [
        ClauseNode(
            clause_path=str(i + 1),
            heading=h,
            text=f"Text for {h}.",
            char_span=(0, 10),
        )
        for i, h in enumerate(headings)
    ]
    return ClauseTree(document_id=document_id, version="v1", source_file="doc.docx", nodes=nodes)


# ---------------------------------------------------------------------------
# _payload_key
# ---------------------------------------------------------------------------


class TestPayloadKey:
    def test_stable_across_calls(self) -> None:
        payload = {"stage": "classify", "text": "Some clause text.", "heading": "Test"}
        assert _payload_key(payload) == _payload_key(payload)

    def test_differs_for_distinct_payloads(self) -> None:
        p1 = {"stage": "classify", "text": "Clause A."}
        p2 = {"stage": "classify", "text": "Clause B."}
        assert _payload_key(p1) != _payload_key(p2)

    def test_full_text_is_not_truncated(self) -> None:
        """Key must differ when text differs only past char 500 (no truncation)."""
        shared_prefix = "X" * 600
        p1 = {"text": shared_prefix + " SUFFIX-A"}
        p2 = {"text": shared_prefix + " SUFFIX-B"}
        assert _payload_key(p1) != _payload_key(p2)


# ---------------------------------------------------------------------------
# VerdictStore
# ---------------------------------------------------------------------------


class TestVerdictStore:
    def test_get_miss_returns_none(self, tmp_path: Path) -> None:
        store = VerdictStore(tmp_path / "v.jsonl")
        assert store.get({"stage": "classify", "text": "hello"}) is None

    def test_put_then_get_returns_verdict(self, tmp_path: Path) -> None:
        store = VerdictStore(tmp_path / "v.jsonl")
        payload = {"stage": "classify", "text": "clause text"}
        verdict = {"taxonomy_id": "tax-001", "confidence": 0.9, "basis": "judge"}
        store.put(payload, verdict)
        assert store.get(payload) == verdict

    def test_persists_across_instances(self, tmp_path: Path) -> None:
        """AC-4: VerdictStore round-trips verdicts across separate instances."""
        path = tmp_path / "v.jsonl"
        payload = {"stage": "classify", "text": "indemnification clause"}
        verdict = {"taxonomy_id": "tax-indem", "confidence": 0.95, "basis": "judge"}

        store1 = VerdictStore(path)
        store1.put(payload, verdict)

        # New instance reads from disk.
        store2 = VerdictStore(path)
        assert store2.get(payload) == verdict

    def test_file_created_in_subdirectory(self, tmp_path: Path) -> None:
        """VerdictStore creates parent directories on first write."""
        path = tmp_path / "judge" / "verdicts.jsonl"
        assert not path.parent.exists()
        store = VerdictStore(path)
        store.put({"k": 1}, {"v": 1})
        assert path.exists()

    def test_corrupt_line_skipped_silently(self, tmp_path: Path) -> None:
        """Corrupt lines in the JSONL file must not crash startup."""
        path = tmp_path / "v.jsonl"
        path.write_text('{"key":"k1","verdict":{"r":1}}\nNOT-JSON\n', encoding="utf-8")
        store = VerdictStore(path)
        assert store._store.get("k1") == {"r": 1}

    def test_overwrite_updates_in_memory(self, tmp_path: Path) -> None:
        """Subsequent put for the same payload updates the in-memory view."""
        store = VerdictStore(tmp_path / "v.jsonl")
        payload = {"k": "same"}
        store.put(payload, {"v": 1})
        store.put(payload, {"v": 2})
        assert store.get(payload) == {"v": 2}


# ---------------------------------------------------------------------------
# PendingQueue
# ---------------------------------------------------------------------------


class TestPendingQueue:
    def test_add_writes_record(self, tmp_path: Path) -> None:
        path = tmp_path / "pending.jsonl"
        q = PendingQueue(path)
        q.add("key1", "classify", {"text": "clause"})
        lines = [json.loads(line) for line in path.read_text().splitlines()]
        assert len(lines) == 1
        assert lines[0]["key"] == "key1"
        assert lines[0]["kind"] == "classify"
        assert lines[0]["payload"] == {"text": "clause"}

    def test_dedup_same_key_within_instance(self, tmp_path: Path) -> None:
        """Duplicate key on the same queue instance → only one file entry."""
        path = tmp_path / "pending.jsonl"
        q = PendingQueue(path)
        q.add("key1", "classify", {"text": "clause"})
        q.add("key1", "classify", {"text": "clause"})
        lines = path.read_text().splitlines()
        assert len(lines) == 1

    def test_add_returns_true_on_new_key(self, tmp_path: Path) -> None:
        q = PendingQueue(tmp_path / "pending.jsonl")
        assert q.add("key1", "classify", {"text": "a"}) is True

    def test_add_returns_false_on_duplicate_key(self, tmp_path: Path) -> None:
        q = PendingQueue(tmp_path / "pending.jsonl")
        q.add("key1", "classify", {"text": "a"})
        assert q.add("key1", "classify", {"text": "a"}) is False

    def test_different_keys_both_written(self, tmp_path: Path) -> None:
        path = tmp_path / "pending.jsonl"
        q = PendingQueue(path)
        q.add("key1", "classify", {"text": "a"})
        q.add("key2", "deviation", {"hunk": "b"})
        lines = path.read_text().splitlines()
        assert len(lines) == 2

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        path = tmp_path / "judge" / "pending.jsonl"
        q = PendingQueue(path)
        q.add("key1", "provenance", {"preamble": "x"})
        assert path.exists()


# ---------------------------------------------------------------------------
# StoreBackedClassificationJudge
# ---------------------------------------------------------------------------


class TestStoreBackedClassificationJudge:
    """AC-1, AC-2, AC-3, AC-5, AC-6 for classification."""

    def test_implements_protocol(self, tmp_path: Path) -> None:
        """AC-6: StoreBackedClassificationJudge is a valid ClassificationJudge."""
        store, pending = _make_store_and_pending(tmp_path)
        judge = StoreBackedClassificationJudge(store=store, pending=pending)
        assert isinstance(judge, ClassificationJudge)

    def test_miss_returns_needs_review_sentinel(self, tmp_path: Path) -> None:
        """AC-2: store miss → needs_review sentinel returned."""
        store, pending = _make_store_and_pending(tmp_path)
        judge = StoreBackedClassificationJudge(store=store, pending=pending)
        node = _make_node("Indemnification", "The party shall indemnify...", "1")
        tax = _taxonomy("tax-001", "tax-002")

        results = judge.classify_batch([node], tax)

        assert len(results) == 1
        assert results[0].basis == "needs_review"
        assert results[0].taxonomy_id is None
        assert results[0].confidence == 0.0

    def test_miss_records_pending_entry(self, tmp_path: Path) -> None:
        """AC-2: store miss → exactly one pending entry written."""
        store, pending = _make_store_and_pending(tmp_path)
        judge = StoreBackedClassificationJudge(store=store, pending=pending)
        node = _make_node("Payment Terms", "Monthly payment of fees.", "1")
        tax = _taxonomy("tax-001")

        judge.classify_batch([node], tax)

        path = tmp_path / "judge" / "pending.jsonl"
        lines = path.read_text().splitlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["kind"] == "classify"
        assert "text" in record["payload"]

    def test_hit_returns_stored_verdict_as_clause_classification(self, tmp_path: Path) -> None:
        """AC-1: store hit → ClauseClassification with stored values, basis='judge'."""
        store, pending = _make_store_and_pending(tmp_path)
        judge = StoreBackedClassificationJudge(store=store, pending=pending)
        node = _make_node("Governing Law", "This agreement is governed by...", "1")
        tax = _taxonomy("tax-001")

        # Pre-populate the store with a verdict for this node's payload.
        payload = {
            "stage": "classify",
            "text": node.text,
            "heading": node.heading,
            "taxonomy_ids": ["tax-001"],
        }
        store.put(payload, {"taxonomy_id": "tax-001", "confidence": 0.92, "basis": "judge"})

        results = judge.classify_batch([node], tax)

        assert len(results) == 1
        assert results[0].taxonomy_id == "tax-001"
        assert results[0].confidence == pytest.approx(0.92)
        assert results[0].basis == "judge"

    def test_hit_does_not_write_to_pending_queue(self, tmp_path: Path) -> None:
        """AC-1: store hit → no pending entry written."""
        store, pending = _make_store_and_pending(tmp_path)
        judge = StoreBackedClassificationJudge(store=store, pending=pending)
        node = _make_node("Confidentiality", "Information is confidential.", "1")
        tax = _taxonomy("tax-001")

        payload = {
            "stage": "classify",
            "text": node.text,
            "heading": node.heading,
            "taxonomy_ids": ["tax-001"],
        }
        store.put(payload, {"taxonomy_id": "tax-001", "confidence": 0.88, "basis": "judge"})

        judge.classify_batch([node], tax)

        queue_path = tmp_path / "judge" / "pending.jsonl"
        assert not queue_path.exists()

    def test_duplicate_payloads_in_batch_produce_one_pending_entry(self, tmp_path: Path) -> None:
        """AC-3: duplicate payloads in one batch → exactly one pending-queue entry."""
        store, pending = _make_store_and_pending(tmp_path)
        judge = StoreBackedClassificationJudge(store=store, pending=pending)
        # Two nodes with identical text and heading → same payload key.
        node_a = _make_node("Warranty", "As-is warranty disclaimer text.", "1")
        node_b = _make_node("Warranty", "As-is warranty disclaimer text.", "2")
        tax = _taxonomy("tax-001")

        results = judge.classify_batch([node_a, node_b], tax)

        assert len(results) == 2
        for r in results:
            assert r.basis == "needs_review"

        path = tmp_path / "judge" / "pending.jsonl"
        lines = path.read_text().splitlines()
        assert len(lines) == 1, (
            "Two identical payloads in one batch must produce exactly one pending entry"
        )

    def test_pending_payload_contains_full_clause_text(self, tmp_path: Path) -> None:
        """AC-5: pending payload must carry full clause text (> 500 chars)."""
        store, pending = _make_store_and_pending(tmp_path)
        judge = StoreBackedClassificationJudge(store=store, pending=pending)
        long_text = "A" * 100 + " indemnification obligation text " + "B" * 400
        node = _make_node("Indemnification", long_text, "1")
        tax = _taxonomy("tax-001")

        judge.classify_batch([node], tax)

        path = tmp_path / "judge" / "pending.jsonl"
        record = json.loads(path.read_text().splitlines()[0])
        stored_text = record["payload"]["text"]
        assert len(stored_text) > 500, (
            f"Full clause text must be stored untruncated; got len={len(stored_text)}"
        )
        assert stored_text == long_text

    def test_result_count_matches_input_count(self, tmp_path: Path) -> None:
        """Result list length must equal input node count (protocol contract)."""
        store, pending = _make_store_and_pending(tmp_path)
        judge = StoreBackedClassificationJudge(store=store, pending=pending)
        nodes = [_make_node(f"Clause{i}", f"Text {i}.", str(i)) for i in range(5)]
        tax = _taxonomy("tax-001", "tax-002")
        results = judge.classify_batch(nodes, tax)
        assert len(results) == 5

    def test_mixed_hit_and_miss_in_one_batch(self, tmp_path: Path) -> None:
        """Mix of store hit and miss in one batch returns correct results per node."""
        store, pending = _make_store_and_pending(tmp_path)
        judge = StoreBackedClassificationJudge(store=store, pending=pending)
        node_hit = _make_node("Governing Law", "Governed by NY law.", "1")
        node_miss = _make_node("Arbitration", "Disputes resolved by AAA.", "2")
        tax = _taxonomy("tax-001")

        # Pre-populate store for node_hit only.
        payload_hit = {
            "stage": "classify",
            "text": node_hit.text,
            "heading": node_hit.heading,
            "taxonomy_ids": ["tax-001"],
        }
        store.put(payload_hit, {"taxonomy_id": "tax-001", "confidence": 0.85, "basis": "judge"})

        results = judge.classify_batch([node_hit, node_miss], tax)

        assert results[0].basis == "judge"
        assert results[0].taxonomy_id == "tax-001"
        assert results[1].basis == "needs_review"

    def test_malformed_verdict_is_isolated_and_logged(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A malformed stored verdict (e.g. an invalid ``basis`` — the actual
        real-world cause, issue #182/#191 overnight run: a doc bug told the
        verdict-supplier to use ``basis="llm"``, which ``ClauseClassification``
        rejects) must not raise out of ``classify_batch``, must be isolated to
        its own node (re-queued as needs_review), and must be logged at
        WARNING so a bad basis is never silently swallowed again.
        """
        store, pending = _make_store_and_pending(tmp_path)
        judge = StoreBackedClassificationJudge(store=store, pending=pending)
        node = _make_node("Indemnification", "The party shall indemnify...", "1")
        tax = _taxonomy("tax-001")

        payload = {
            "stage": "classify",
            "text": node.text,
            "heading": node.heading,
            "taxonomy_ids": ["tax-001"],
        }
        # Invalid: "llm" is not in ClauseClassification._BASIS_VALUES.
        store.put(payload, {"taxonomy_id": "tax-001", "confidence": 0.9, "basis": "llm"})

        with caplog.at_level(logging.WARNING, logger="playbook_engine.agent_judge"):
            results = judge.classify_batch([node], tax)

        assert len(results) == 1
        assert results[0].basis == "needs_review"
        assert any(
            "malformed stored verdict" in r.message.lower() and r.levelno == logging.WARNING
            for r in caplog.records
        ), [r.message for r in caplog.records]

        path = tmp_path / "judge" / "pending.jsonl"
        lines = path.read_text().splitlines()
        assert len(lines) == 1
        assert json.loads(lines[0])["kind"] == "classify"


# ---------------------------------------------------------------------------
# StoreBackedDeviationJudge
# ---------------------------------------------------------------------------


class TestStoreBackedDeviationJudge:
    """AC-1, AC-2, AC-3, AC-5, AC-6 for deviation."""

    def test_implements_protocol(self, tmp_path: Path) -> None:
        """AC-6: StoreBackedDeviationJudge is a valid DeviationJudge."""
        store, pending = _make_store_and_pending(tmp_path)
        judge = StoreBackedDeviationJudge(store=store, pending=pending)
        assert isinstance(judge, DeviationJudge)

    def test_miss_returns_needs_review_sentinel(self, tmp_path: Path) -> None:
        """AC-2: store miss → needs_review sentinel returned."""
        store, pending = _make_store_and_pending(tmp_path)
        judge = StoreBackedDeviationJudge(store=store, pending=pending)
        items = [{"hunk": "[BEFORE]\nold text\n[AFTER]\nnew text"}]

        results = judge.assess_batch(items, our_standard="Standard text.")

        assert len(results) == 1
        assert results[0].basis == "needs_review"
        assert results[0].deviation == "needs_review"
        assert results[0].confidence is None

    def test_miss_records_pending_entry(self, tmp_path: Path) -> None:
        """AC-2: store miss → exactly one pending entry written."""
        store, pending = _make_store_and_pending(tmp_path)
        judge = StoreBackedDeviationJudge(store=store, pending=pending)
        items = [{"hunk": "[BEFORE]\nold\n[AFTER]\nnew"}]

        judge.assess_batch(items, our_standard="Standard.")

        path = tmp_path / "judge" / "pending.jsonl"
        lines = path.read_text().splitlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["kind"] == "deviation"
        assert "hunk" in record["payload"]
        assert "our_standard" in record["payload"]

    def test_hit_returns_stored_verdict_as_deviation_result(self, tmp_path: Path) -> None:
        """AC-1: store hit → DeviationResult with stored values, basis='judge'."""
        store, pending = _make_store_and_pending(tmp_path)
        judge = StoreBackedDeviationJudge(store=store, pending=pending)
        hunk = "[BEFORE]\nProvide services\n[AFTER]\nProvide extended services"
        our_standard = "Provide services as agreed."
        items = [{"hunk": hunk}]

        # Build the payload the judge will use to look up the store — the
        # hash key is content-only (stage/hunk/our_standard); traceability
        # context like clause_path is deliberately excluded (issue #109).
        payload = {
            "stage": "deviation",
            "hunk": hunk,
            "our_standard": our_standard,
        }
        store.put(
            payload,
            {
                "deviation": "substantive",
                "risk_delta": {"direction": "worse", "magnitude": "minor"},
                "basis": "judge",
                "rationale": "Services extended beyond agreed scope.",
                "confidence": 0.85,
            },
        )

        results = judge.assess_batch(items, our_standard=our_standard)

        assert len(results) == 1
        assert results[0].deviation == "substantive"
        assert results[0].risk_delta.direction == "worse"
        assert results[0].risk_delta.magnitude == "minor"
        assert results[0].basis == "judge"
        assert results[0].confidence == pytest.approx(0.85)

    def test_malformed_verdict_is_isolated_not_batch_poisoning(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A single malformed stored verdict must not poison the whole batch (issue #182).

        A RiskDelta invariant violation (direction='neutral' requires
        magnitude='none') previously raised out of assess_batch; the caller's
        blanket except then quarantined EVERY clause in the taxonomy group as
        basis='judge_error'. The bad verdict must be isolated to its own item
        (re-queued as needs_review) while the rest of the batch replays — and
        the isolation must be logged at WARNING (previously silent, which is
        exactly what let an invalid "basis": "llm" in every deviation verdict
        go unnoticed for a full overnight run before this fix).
        """
        store, pending = _make_store_and_pending(tmp_path)
        judge = StoreBackedDeviationJudge(store=store, pending=pending)
        our_standard = "Standard clause."

        good_hunk = "[BEFORE]\na\n[AFTER]\nb"
        bad_hunk = "[BEFORE]\nc\n[AFTER]\nd"
        store.put(
            {"stage": "deviation", "hunk": good_hunk, "our_standard": our_standard},
            {
                "deviation": "substantive",
                "risk_delta": {"direction": "worse", "magnitude": "minor"},
                "basis": "judge",
            },
        )
        # Invalid: neutral direction must have magnitude 'none'.
        store.put(
            {"stage": "deviation", "hunk": bad_hunk, "our_standard": our_standard},
            {
                "deviation": "substantive",
                "risk_delta": {"direction": "neutral", "magnitude": "minor"},
                "basis": "judge",
            },
        )

        with caplog.at_level(logging.WARNING, logger="playbook_engine.agent_judge"):
            results = judge.assess_batch(
                [{"hunk": good_hunk}, {"hunk": bad_hunk}], our_standard=our_standard
            )

        assert len(results) == 2
        # Good item replays normally.
        assert results[0].deviation == "substantive"
        assert results[0].basis == "judge"
        # Bad item is isolated: re-queued as needs_review, not judge_error over the batch.
        assert results[1].deviation == "needs_review"
        assert results[1].basis == "needs_review"
        assert any(
            "malformed stored verdict" in r.message.lower() and r.levelno == logging.WARNING
            for r in caplog.records
        ), [r.message for r in caplog.records]

    def test_hit_does_not_write_to_pending_queue(self, tmp_path: Path) -> None:
        """AC-1: store hit → no pending entry written."""
        store, pending = _make_store_and_pending(tmp_path)
        judge = StoreBackedDeviationJudge(store=store, pending=pending)
        hunk = "[BEFORE]\nold\n[AFTER]\nnew"
        our_standard = "Standard."

        payload = {
            "stage": "deviation",
            "hunk": hunk,
            "our_standard": our_standard,
        }
        store.put(
            payload,
            {
                "deviation": "none",
                "risk_delta": {"direction": "neutral", "magnitude": "none"},
                "basis": "judge",
                "rationale": "",
                "confidence": 0.9,
            },
        )

        judge.assess_batch([{"hunk": hunk}], our_standard=our_standard)

        queue_path = tmp_path / "judge" / "pending.jsonl"
        assert not queue_path.exists()

    def test_duplicate_payloads_in_batch_produce_one_pending_entry(self, tmp_path: Path) -> None:
        """AC-3: duplicate items in one assess_batch call → exactly one pending entry."""
        store, pending = _make_store_and_pending(tmp_path)
        judge = StoreBackedDeviationJudge(store=store, pending=pending)
        hunk = "[BEFORE]\nProvide service\n[AFTER]\nProvide extended service"
        items = [{"hunk": hunk}, {"hunk": hunk}]

        results = judge.assess_batch(items, our_standard="Standard.")

        assert len(results) == 2
        for r in results:
            assert r.basis == "needs_review"

        path = tmp_path / "judge" / "pending.jsonl"
        lines = path.read_text().splitlines()
        assert len(lines) == 1, (
            "Two identical items in one batch must produce exactly one pending entry"
        )

    def test_pending_payload_contains_full_our_standard(self, tmp_path: Path) -> None:
        """AC-5 (deviation): pending payload must carry full our_standard text."""
        store, pending = _make_store_and_pending(tmp_path)
        judge = StoreBackedDeviationJudge(store=store, pending=pending)
        long_standard = "Standard clause text. " * 30  # > 500 chars
        items = [{"hunk": "[BEFORE]\nold\n[AFTER]\nnew"}]

        judge.assess_batch(items, our_standard=long_standard)

        path = tmp_path / "judge" / "pending.jsonl"
        record = json.loads(path.read_text().splitlines()[0])
        stored_standard = record["payload"]["our_standard"]
        assert len(stored_standard) > 500, (
            f"Full our_standard must be stored untruncated; got len={len(stored_standard)}"
        )
        assert stored_standard == long_standard

    def test_result_count_matches_input_count(self, tmp_path: Path) -> None:
        """Result list length must equal input item count (protocol contract)."""
        store, pending = _make_store_and_pending(tmp_path)
        judge = StoreBackedDeviationJudge(store=store, pending=pending)
        items = [{"hunk": f"[BEFORE]\n{i}\n[AFTER]\n{i}_new"} for i in range(4)]
        results = judge.assess_batch(items, our_standard="Standard.")
        assert len(results) == 4

    def test_pending_payload_records_context(self, tmp_path: Path) -> None:
        """Issue #109: pending payload records taxonomy_id/clause_path/document_id
        context when the item carries it."""
        store, pending = _make_store_and_pending(tmp_path)
        judge = StoreBackedDeviationJudge(store=store, pending=pending)
        items = [
            {
                "hunk": "[BEFORE]\nold\n[AFTER]\nnew",
                "taxonomy_id": "indemnification",
                "clause_path": "3.2",
                "document_id": "doc-a",
            }
        ]

        judge.assess_batch(items, our_standard="Standard.")

        path = tmp_path / "judge" / "pending.jsonl"
        record = json.loads(path.read_text().splitlines()[0])
        payload = record["payload"]
        assert payload["taxonomy_id"] == "indemnification"
        assert payload["clause_path"] == "3.2"
        assert payload["document_id"] == "doc-a"

    def test_content_hash_ignores_context_preserving_cross_doc_dedup(self, tmp_path: Path) -> None:
        """Issue #109: two items with identical hunk+our_standard but different
        clause_path/taxonomy_id/document_id must share one pending entry AND
        one cache key — the content-hash key must not include traceability
        context, or the exact same clause appearing in two different
        documents would never dedup."""
        store, pending = _make_store_and_pending(tmp_path)
        judge = StoreBackedDeviationJudge(store=store, pending=pending)
        hunk = "[BEFORE]\nProvide services\n[AFTER]\nProvide extended services"
        items = [
            {
                "hunk": hunk,
                "taxonomy_id": "services",
                "clause_path": "1.1",
                "document_id": "doc-a",
            },
            {
                "hunk": hunk,
                "taxonomy_id": "services",
                "clause_path": "9.9",
                "document_id": "doc-b",
            },
        ]

        results = judge.assess_batch(items, our_standard="Standard.")

        assert len(results) == 2
        path = tmp_path / "judge" / "pending.jsonl"
        lines = path.read_text().splitlines()
        assert len(lines) == 1, (
            "Identical hunk/our_standard from two different documents must "
            "produce exactly one pending entry — context must not affect the hash"
        )

        # Now confirm a store hit for doc-a's payload also serves doc-b's
        # identical hunk/standard — i.e. the store key really is content-only.
        store2 = VerdictStore(tmp_path / "judge2" / "verdicts.jsonl")
        pending2 = PendingQueue(tmp_path / "judge2" / "pending.jsonl")
        store2.put(
            {"stage": "deviation", "hunk": hunk, "our_standard": "Standard."},
            {
                "deviation": "substantive",
                "risk_delta": {"direction": "worse", "magnitude": "minor"},
                "basis": "judge",
                "rationale": "",
                "confidence": 0.8,
            },
        )
        judge2 = StoreBackedDeviationJudge(store=store2, pending=pending2)
        results2 = judge2.assess_batch(items, our_standard="Standard.")
        assert results2[0].basis == "judge"
        assert results2[1].basis == "judge"


# ---------------------------------------------------------------------------
# StoreBackedProvenanceJudge
# ---------------------------------------------------------------------------


class TestStoreBackedProvenanceJudge:
    """AC-1, AC-2, AC-6 for provenance."""

    def test_implements_protocol(self, tmp_path: Path) -> None:
        """AC-6: StoreBackedProvenanceJudge is a valid ProvenanceJudge."""
        store, pending = _make_store_and_pending(tmp_path)
        judge = StoreBackedProvenanceJudge(store=store, pending=pending)
        assert isinstance(judge, ProvenanceJudge)

    def test_miss_returns_needs_review_sentinel(self, tmp_path: Path) -> None:
        """AC-2: store miss → low-confidence needs_review sentinel returned."""
        store, pending = _make_store_and_pending(tmp_path)
        judge = StoreBackedProvenanceJudge(store=store, pending=pending)

        result = judge.judge(
            preamble="This agreement is between Alpha Corp and Beta Inc.",
            letterhead="Master Services Agreement",
            agreement_type="Master Services Agreement",
        )

        assert result.basis == "needs_review"
        assert result.confidence == 0.0

    def test_miss_records_pending_entry(self, tmp_path: Path) -> None:
        """AC-2: store miss → exactly one pending entry with full provenance payload."""
        store, pending = _make_store_and_pending(tmp_path)
        judge = StoreBackedProvenanceJudge(store=store, pending=pending)

        judge.judge(
            preamble="Between Alpha and Beta.",
            letterhead="Services Agreement",
            agreement_type="Services Agreement",
        )

        path = tmp_path / "judge" / "pending.jsonl"
        lines = path.read_text().splitlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["kind"] == "provenance"
        assert "preamble" in record["payload"]
        assert "letterhead" in record["payload"]
        assert "agreement_type" in record["payload"]

    def test_hit_returns_stored_verdict_as_provenance_result(self, tmp_path: Path) -> None:
        """AC-1: store hit → ProvenanceResult with stored values."""
        store, pending = _make_store_and_pending(tmp_path)
        judge = StoreBackedProvenanceJudge(store=store, pending=pending)
        preamble = "Between Alpha Corp (Provider) and Beta Inc (Client)."
        letterhead = "Master Services Agreement"
        agreement_type = "MSA"

        payload = {
            "stage": "provenance",
            "preamble": preamble,
            "letterhead": letterhead,
            "agreement_type": agreement_type,
        }
        store.put(payload, {"provenance": "our_paper", "confidence": 0.90, "basis": "llm"})

        result = judge.judge(preamble, letterhead, agreement_type)

        assert result.provenance == "our_paper"
        assert result.confidence == pytest.approx(0.90)
        assert result.basis == "llm"

    def test_hit_does_not_write_to_pending_queue(self, tmp_path: Path) -> None:
        """AC-1: store hit → no pending entry written."""
        store, pending = _make_store_and_pending(tmp_path)
        judge = StoreBackedProvenanceJudge(store=store, pending=pending)
        preamble = "Parties: Alpha Corp and Beta Inc."
        letterhead = "NDA"
        agreement_type = "NDA"

        payload = {
            "stage": "provenance",
            "preamble": preamble,
            "letterhead": letterhead,
            "agreement_type": agreement_type,
        }
        store.put(payload, {"provenance": "counterparty_paper", "confidence": 0.80, "basis": "llm"})

        judge.judge(preamble, letterhead, agreement_type)

        queue_path = tmp_path / "judge" / "pending.jsonl"
        assert not queue_path.exists()

    def test_repeated_miss_produces_one_pending_entry(self, tmp_path: Path) -> None:
        """AC-3 (provenance): calling judge() twice with same args → one pending entry."""
        store, pending = _make_store_and_pending(tmp_path)
        judge = StoreBackedProvenanceJudge(store=store, pending=pending)

        judge.judge(preamble="Same.", letterhead="Same MSA", agreement_type="MSA")
        judge.judge(preamble="Same.", letterhead="Same MSA", agreement_type="MSA")

        path = tmp_path / "judge" / "pending.jsonl"
        lines = path.read_text().splitlines()
        assert len(lines) == 1, (
            "Repeated call with identical args must produce exactly one pending entry"
        )

    def test_miss_sentinel_is_low_confidence(self, tmp_path: Path) -> None:
        """Provenance miss sentinel must have confidence=0.0 so deterministic default is not trusted."""
        store, pending = _make_store_and_pending(tmp_path)
        judge = StoreBackedProvenanceJudge(store=store, pending=pending)

        result = judge.judge(
            preamble="Some preamble text.",
            letterhead="Agreement Title",
            agreement_type="Services",
        )

        assert result.confidence == 0.0, (
            "Provenance miss must return confidence=0.0 to prevent deterministic default being trusted"
        )

    def test_malformed_verdict_is_isolated_and_logged(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A malformed stored verdict (e.g. an unknown ``provenance`` value)
        must not raise out of ``judge()``; it must fall back to the
        needs_review sentinel, re-queue the payload, and log a WARNING
        (issue #182 — same pattern as the other three store-backed judges).
        """
        store, pending = _make_store_and_pending(tmp_path)
        judge = StoreBackedProvenanceJudge(store=store, pending=pending)
        preamble = "Between Alpha Corp and Beta Inc."
        letterhead = "MSA"
        agreement_type = "MSA"

        payload = {
            "stage": "provenance",
            "preamble": preamble,
            "letterhead": letterhead,
            "agreement_type": agreement_type,
        }
        # Invalid: not in ProvenanceResult._PROVENANCE_VALUES.
        store.put(payload, {"provenance": "our_standard", "confidence": 0.8, "basis": "llm"})

        with caplog.at_level(logging.WARNING, logger="playbook_engine.agent_judge"):
            result = judge.judge(preamble, letterhead, agreement_type)

        assert result.basis == "needs_review"
        assert any(
            "malformed stored verdict" in r.message.lower() and r.levelno == logging.WARNING
            for r in caplog.records
        ), [r.message for r in caplog.records]

        path = tmp_path / "judge" / "pending.jsonl"
        lines = path.read_text().splitlines()
        assert len(lines) == 1
        assert json.loads(lines[0])["kind"] == "provenance"


# ---------------------------------------------------------------------------
# VerdictStore persistence — round-trip across instances (AC-4)
# ---------------------------------------------------------------------------


class TestVerdictStoreRoundTrip:
    """AC-4: full round-trip for all three judge kinds."""

    def test_classify_verdict_round_trips(self, tmp_path: Path) -> None:
        path = tmp_path / "verdicts.jsonl"
        payload = {
            "stage": "classify",
            "text": "some clause",
            "heading": "Indemnification",
            "taxonomy_ids": ["t1"],
        }
        verdict = {"taxonomy_id": "t1", "confidence": 0.91, "basis": "judge"}

        store1 = VerdictStore(path)
        store1.put(payload, verdict)

        store2 = VerdictStore(path)
        assert store2.get(payload) == verdict

    def test_deviation_verdict_round_trips(self, tmp_path: Path) -> None:
        path = tmp_path / "verdicts.jsonl"
        payload = {
            "stage": "deviation",
            "hunk": "[BEFORE]\nold\n[AFTER]\nnew",
            "our_standard": "Standard.",
            "clause_path": "3.1",
        }
        verdict = {
            "deviation": "substantive",
            "risk_delta": {"direction": "worse", "magnitude": "material"},
            "basis": "judge",
            "rationale": "Material adverse change.",
            "confidence": 0.88,
        }

        store1 = VerdictStore(path)
        store1.put(payload, verdict)

        store2 = VerdictStore(path)
        assert store2.get(payload) == verdict

    def test_provenance_verdict_round_trips(self, tmp_path: Path) -> None:
        path = tmp_path / "verdicts.jsonl"
        payload = {
            "stage": "provenance",
            "preamble": "Between us and them.",
            "letterhead": "MSA",
            "agreement_type": "MSA",
        }
        verdict = {"provenance": "our_paper", "confidence": 0.92, "basis": "llm"}

        store1 = VerdictStore(path)
        store1.put(payload, verdict)

        store2 = VerdictStore(path)
        assert store2.get(payload) == verdict


# ---------------------------------------------------------------------------
# StoreBackedScopeJudge — issue #87
# ---------------------------------------------------------------------------


class TestStoreBackedScopeJudge:
    """A store hit replays the verdict; a miss queues for review and never
    auto-accepts (contrast with the ``_AllInScopeJudge`` stub's blind
    ``in_scope=True`` at confidence 0.5)."""

    def test_implements_protocol(self, tmp_path: Path) -> None:
        store, pending = _make_store_and_pending(tmp_path)
        judge = StoreBackedScopeJudge(store=store, pending=pending)
        assert isinstance(judge, ScopeJudge)

    def test_hit_returns_out_of_scope_decision(self, tmp_path: Path) -> None:
        """A stored out-of-scope verdict yields an out-of-scope ScopeDecision."""
        store, pending = _make_store_and_pending(tmp_path)
        judge = StoreBackedScopeJudge(store=store, pending=pending)
        tree = _make_tree("doc-dpa", ["Data Processing", "Sub-processors"])
        agreement_type = AgreementType(id="eiaa", name="Educational Affiliation Agreement")

        payload = {
            "stage": "scope",
            "agreement_type_id": "eiaa",
            "document_id": "doc-dpa",
            "clause_heads": ["Data Processing", "Sub-processors"],
        }
        store.put(
            payload,
            {
                "in_scope": False,
                "scope_rationale": "This is a Data Processing Agreement, not an affiliation agreement.",
                "scope_confidence": 0.93,
            },
        )

        decision = judge.judge(tree, agreement_type)

        assert decision.in_scope is False
        assert decision.basis == "judge"
        assert decision.scope_confidence == pytest.approx(0.93)
        assert "Data Processing Agreement" in decision.scope_rationale

    def test_hit_returns_in_scope_decision(self, tmp_path: Path) -> None:
        store, pending = _make_store_and_pending(tmp_path)
        judge = StoreBackedScopeJudge(store=store, pending=pending)
        tree = _make_tree("doc-eiaa", ["Placement Terms", "Supervision"])
        agreement_type = AgreementType(id="eiaa", name="Educational Affiliation Agreement")

        payload = {
            "stage": "scope",
            "agreement_type_id": "eiaa",
            "document_id": "doc-eiaa",
            "clause_heads": ["Placement Terms", "Supervision"],
        }
        store.put(
            payload,
            {
                "in_scope": True,
                "scope_rationale": "Matches the affiliation-agreement clause profile.",
                "scope_confidence": 0.97,
            },
        )

        decision = judge.judge(tree, agreement_type)

        assert decision.in_scope is True
        assert decision.basis == "judge"
        assert decision.scope_confidence == pytest.approx(0.97)

    def test_hit_does_not_write_to_pending_queue(self, tmp_path: Path) -> None:
        store, pending = _make_store_and_pending(tmp_path)
        judge = StoreBackedScopeJudge(store=store, pending=pending)
        tree = _make_tree("doc-1", ["Term"])
        agreement_type = AgreementType(id="eiaa", name="Educational Affiliation Agreement")

        payload = {
            "stage": "scope",
            "agreement_type_id": "eiaa",
            "document_id": "doc-1",
            "clause_heads": ["Term"],
        }
        store.put(
            payload,
            {"in_scope": True, "scope_rationale": "In scope.", "scope_confidence": 0.9},
        )

        judge.judge(tree, agreement_type)

        queue_path = tmp_path / "judge" / "pending.jsonl"
        assert not queue_path.exists()

    def test_miss_routes_to_pending_queue_rather_than_auto_accepting(self, tmp_path: Path) -> None:
        """An unstored doc queues for review and raises, rather than silently
        returning the stub default's in_scope=True at confidence 0.5."""
        store, pending = _make_store_and_pending(tmp_path)
        judge = StoreBackedScopeJudge(store=store, pending=pending)
        tree = _make_tree("doc-unstored", ["Indemnification", "Governing Law"])
        agreement_type = AgreementType(id="eiaa", name="Educational Affiliation Agreement")

        with pytest.raises(ScopeNeedsReviewError):
            judge.judge(tree, agreement_type)

        path = tmp_path / "judge" / "pending.jsonl"
        lines = path.read_text().splitlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["kind"] == "scope"
        assert record["payload"]["document_id"] == "doc-unstored"
        assert record["payload"]["clause_heads"] == ["Indemnification", "Governing Law"]

    def test_miss_produces_one_pending_entry_on_repeat(self, tmp_path: Path) -> None:
        """Repeated misses on the same document produce exactly one pending entry."""
        store, pending = _make_store_and_pending(tmp_path)
        judge = StoreBackedScopeJudge(store=store, pending=pending)
        tree = _make_tree("doc-repeat", ["Confidentiality"])
        agreement_type = AgreementType(id="eiaa", name="Educational Affiliation Agreement")

        for _ in range(2):
            with pytest.raises(ScopeNeedsReviewError):
                judge.judge(tree, agreement_type)

        path = tmp_path / "judge" / "pending.jsonl"
        lines = path.read_text().splitlines()
        assert len(lines) == 1

    def test_scope_gate_converts_miss_into_retained_judge_error(self, tmp_path: Path) -> None:
        """End-to-end: scope_gate() catches the raise and retains-for-review
        rather than dropping or auto-accepting the document."""
        from playbook_engine.scope_gate import scope_gate

        store, pending = _make_store_and_pending(tmp_path)
        judge = StoreBackedScopeJudge(store=store, pending=pending)
        tree = _make_tree("doc-e2e", ["Non-Compete", "Liquidated Damages"])
        agreement_type = AgreementType(id="eiaa", name="Educational Affiliation Agreement")

        decision = scope_gate(tree, agreement_type, judge)

        assert decision.basis == "judge_error"
        assert decision.in_scope is True  # retained pending review, not silently dropped
        assert decision.scope_confidence == 0.0

        path = tmp_path / "judge" / "pending.jsonl"
        assert len(path.read_text().splitlines()) == 1

    def test_malformed_verdict_is_isolated_and_logged(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A malformed stored verdict (missing ``in_scope``) must not raise an
        unhandled ``KeyError`` out of ``judge()``; it must be re-queued and
        raise ``ScopeNeedsReviewError`` (the same contract a store miss uses),
        logged at WARNING (issue #182 — same pattern as the other three
        store-backed judges).
        """
        store, pending = _make_store_and_pending(tmp_path)
        judge = StoreBackedScopeJudge(store=store, pending=pending)
        tree = _make_tree("doc-malformed", ["Term"])
        agreement_type = AgreementType(id="eiaa", name="Educational Affiliation Agreement")

        payload = {
            "stage": "scope",
            "agreement_type_id": "eiaa",
            "document_id": "doc-malformed",
            "clause_heads": ["Term"],
        }
        # Invalid: missing required "in_scope" key.
        store.put(payload, {"scope_rationale": "Incomplete verdict.", "scope_confidence": 0.9})

        with (
            caplog.at_level(logging.WARNING, logger="playbook_engine.agent_judge"),
            pytest.raises(ScopeNeedsReviewError),
        ):
            judge.judge(tree, agreement_type)

        assert any(
            "malformed stored verdict" in r.message.lower() and r.levelno == logging.WARNING
            for r in caplog.records
        ), [r.message for r in caplog.records]

        path = tmp_path / "judge" / "pending.jsonl"
        lines = path.read_text().splitlines()
        assert len(lines) == 1
        assert json.loads(lines[0])["kind"] == "scope"
