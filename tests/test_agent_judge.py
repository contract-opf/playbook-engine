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
    StoreBackedProvenanceJudge,
    StoreBackedScopeJudge,
    VerdictStore,
    _payload_key,
    validate_verdict,
)
from playbook_engine.clause_classifier import ClassificationJudge
from playbook_engine.clause_tree import ClauseNode, ClauseTree
from playbook_engine.config import AgreementType
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


@dataclass
class _FakeTaxEntryWithStatus:
    id: str
    status: str


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
        assert store.get_by_key("k1") == {"r": 1}

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
        q.add("key2", "provenance", {"preamble": "b"})
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

    def test_pending_payload_excludes_inactive_taxonomy_ids(self, tmp_path: Path) -> None:
        """Issue #151: REFERENCE.md calls ``taxonomy_ids`` the "flat list of
        allowed ids" and tells the judge to never invent one outside it. An
        ``inactive`` entry is not allowed (OPF §5: a compiler may only
        classify into active/custom entries) so it must never appear in the
        payload — a judge that picked one from the list used to bank a
        verdict that crashed ``classify_tree`` on replay.
        """
        store, pending = _make_store_and_pending(tmp_path)
        judge = StoreBackedClassificationJudge(store=store, pending=pending)
        node = _make_node("Assignment", "Neither party may assign this agreement.", "1")
        tax = _FakeTaxonomy(
            entries=[
                _FakeTaxEntryWithStatus(id="tax-001", status="active"),
                _FakeTaxEntryWithStatus(id="tax-002", status="custom"),
                _FakeTaxEntryWithStatus(id="tax-003", status="inactive"),
            ]
        )

        judge.classify_batch([node], tax)

        path = tmp_path / "judge" / "pending.jsonl"
        record = json.loads(path.read_text().splitlines()[0])
        assert record["payload"]["taxonomy_ids"] == ["tax-001", "tax-002"]

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

    def test_stored_unknown_verdict_is_malformed_and_requeued(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Issue #225: "unknown" is the engine's own no-verdict sentinel, not a
        judge answer. ProvenanceResult accepts it, so a hand-written store row
        carrying it must hit the same two-sides check validate_verdict applies:
        treated as malformed, re-queued and logged — never replayed as an
        ``llm`` verdict."""
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
        store.put(payload, {"provenance": "unknown", "confidence": 0.9, "basis": "llm"})

        with caplog.at_level(logging.WARNING, logger="playbook_engine.agent_judge"):
            result = judge.judge(preamble, letterhead, agreement_type)

        assert result.basis == "needs_review"
        assert result.confidence == 0.0
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


class TestValidateVerdictBasisWhitelist:
    """Issue #247: validate_verdict's contract is "any verdict accepted here is
    guaranteed to replay" — but it only rejected _UNRESOLVED_VERDICT_BASES and
    otherwise accepted any valid basis. A plausible-but-nonstandard value
    passed here but crashed clause_classifier.classify_tree on the next
    replay, outside the quarantine catch set — aborting the whole corpus run
    after the verdict was already committed to the append-only verdicts.jsonl.
    """

    def test_classify_rejects_exact_match_basis(self) -> None:
        """ "exact_match" is a valid _BASIS_VALUES member but classify_tree raises
        on replay for anything outside 'judge' / 'unclassified'."""
        with pytest.raises(ValueError, match="basis"):
            validate_verdict(
                "classify",
                {"taxonomy_id": "x", "confidence": 0.9, "basis": "exact_match"},
            )

    def test_classify_rejects_heading_similarity_basis(self) -> None:
        with pytest.raises(ValueError, match="basis"):
            validate_verdict(
                "classify",
                {"taxonomy_id": "x", "confidence": 0.9, "basis": "heading_similarity"},
            )

    def test_classify_rejects_llm_segmenter_basis(self) -> None:
        """Verifier correction: "llm_segmenter" also passes the old whitelist-less
        check and crashes classify_tree on replay."""
        with pytest.raises(ValueError, match="basis"):
            validate_verdict(
                "classify",
                {"taxonomy_id": "x", "confidence": 0.9, "basis": "llm_segmenter"},
            )

    def test_classify_accepts_judge_basis(self) -> None:
        validate_verdict(
            "classify",
            {"taxonomy_id": "indemnification", "confidence": 0.9, "basis": "judge"},
        )

    def test_classify_accepts_unclassified_basis(self) -> None:
        """ "unclassified" is engine-accepted on replay for classify."""
        validate_verdict(
            "classify",
            {"taxonomy_id": None, "confidence": 0.0, "basis": "unclassified"},
        )

    def test_provenance_rejects_template_similarity_basis(self) -> None:
        """Issue #162: "template_similarity" is a valid ProvenanceResult
        _BASIS_VALUES member (the engine's own deterministic detector uses it)
        but a producer-supplied verdict only ever represents LLM judgment —
        REFERENCE.md documents "llm" as the one correct provenance basis, so
        anything else must be rejected here rather than silently accepted."""
        with pytest.raises(ValueError, match="basis"):
            validate_verdict(
                "provenance",
                {"provenance": "our_paper", "confidence": 0.9, "basis": "template_similarity"},
            )

    def test_provenance_rejects_hint_basis(self) -> None:
        with pytest.raises(ValueError, match="basis"):
            validate_verdict(
                "provenance",
                {"provenance": "counterparty_paper", "confidence": 0.8, "basis": "hint"},
            )

    def test_provenance_accepts_llm_basis(self) -> None:
        """The one basis a producer-supplied provenance verdict may carry."""
        validate_verdict(
            "provenance",
            {"provenance": "our_paper", "confidence": 0.82, "basis": "llm"},
        )


class TestValidateVerdictConfidenceType:
    """Issue #161: a stringified confidence (a common LLM-producer mistake)
    must raise an actionable ``ValueError`` naming the field, not a bare
    ``TypeError`` from the dataclass's ``0.0 <= confidence <= 1.0`` comparison.
    """

    def test_classify_string_confidence_raises_value_error_not_type_error(self) -> None:
        with pytest.raises(ValueError, match="confidence"):
            validate_verdict(
                "classify",
                {"taxonomy_id": "indemnification", "confidence": "0.8", "basis": "judge"},
            )

    def test_scope_string_confidence_raises_value_error_not_type_error(self) -> None:
        with pytest.raises(ValueError, match="scope_confidence"):
            validate_verdict(
                "scope",
                {
                    "in_scope": True,
                    "scope_rationale": "synthetic rationale",
                    "scope_confidence": "high",
                },
            )

    def test_provenance_string_confidence_raises_value_error_not_type_error(self) -> None:
        with pytest.raises(ValueError, match="confidence"):
            validate_verdict(
                "provenance",
                {"provenance": "counterparty_paper", "confidence": "0.9", "basis": "llm"},
            )

    def test_provenance_unknown_stored_verdict_is_rejected(self) -> None:
        """Issue #225: "unknown" is the engine's own no-verdict sentinel, not
        a judge answer — a stored verdict must name one of the two sides."""
        with pytest.raises(ValueError, match="Unknown provenance"):
            validate_verdict(
                "provenance",
                {"provenance": "unknown", "basis": "llm", "confidence": 0.9},
            )


# ---------------------------------------------------------------------------
# The kind seam (issue #239): a new judge kind registers its verdict shape and
# its rubric; judge-apply, the rubric stamp and the queue then work per kind.
# ---------------------------------------------------------------------------


class TestJudgeKindSeam:
    """A kind that is not built in plugs into validation, inference, the
    rubric framework, the queue and the store without touching any of them."""

    @pytest.fixture
    def widget_kind(self, monkeypatch: pytest.MonkeyPatch) -> str:
        from playbook_engine import agent_judge, rubric

        monkeypatch.setattr(agent_judge, "_VERDICT_KINDS", dict(agent_judge._VERDICT_KINDS))
        monkeypatch.setattr(rubric, "RUBRIC_PROMPT_VERSIONS", dict(rubric.RUBRIC_PROMPT_VERSIONS))
        monkeypatch.setattr(rubric, "_DERIVED_SURFACES", dict(rubric._DERIVED_SURFACES))
        monkeypatch.setattr(rubric, "JUDGE_KINDS", rubric.JUDGE_KINDS)

        def validate(verdict: dict[str, Any]) -> None:
            if verdict.get("equivalent") not in (True, False):
                raise ValueError("'equivalent' must be a JSON boolean")

        agent_judge.register_verdict_kind("widget", validate, lambda v: "equivalent" in v)
        rubric.register_judge_kind(
            "widget", "v1", lambda *, taxonomy, agreement_type: {"answers": [True, False]}
        )
        return "widget"

    def test_builtin_kinds_are_exactly_the_non_deviation_ones(self) -> None:
        from playbook_engine import rubric

        assert set(rubric.JUDGE_KINDS) == {"classify", "provenance", "scope"}

    def test_a_deviation_verdict_is_no_longer_a_kind(self) -> None:
        from playbook_engine.agent_judge import infer_verdict_kind

        verdict = {"deviation": "substantive", "risk_delta": {"direction": "worse"}}
        assert infer_verdict_kind(verdict) is None
        with pytest.raises(ValueError, match="unknown pending-item kind"):
            validate_verdict("deviation", verdict)

    def test_registered_kind_is_validated_at_apply_time(self, widget_kind: str) -> None:
        validate_verdict(widget_kind, {"equivalent": True})
        with pytest.raises(ValueError, match="JSON boolean"):
            validate_verdict(widget_kind, {"equivalent": "yes"})

    def test_registered_kind_is_inferred_from_its_shape(self, widget_kind: str) -> None:
        from playbook_engine.agent_judge import infer_verdict_kind

        assert infer_verdict_kind({"equivalent": False}) == widget_kind
        # The built-in shapes still infer to themselves.
        assert infer_verdict_kind({"in_scope": True}) == "scope"
        assert infer_verdict_kind({"taxonomy_id": None}) == "classify"

    def test_registered_kind_gets_a_rubric_version(self, widget_kind: str) -> None:
        from playbook_engine import rubric

        version = rubric.rubric_version(widget_kind)
        assert version.startswith("v1+")
        assert widget_kind in rubric.current_versions()

    def test_registered_kind_round_trips_through_queue_and_store(
        self, widget_kind: str, tmp_path: Path
    ) -> None:
        from playbook_engine import rubric

        store, pending = _make_store_and_pending(tmp_path)
        payload = {"stage": widget_kind, "left": "a", "right": "b"}
        key = _payload_key(payload)
        version = rubric.rubric_version(widget_kind)
        assert pending.add(key, widget_kind, payload, version) is True

        queued = json.loads((tmp_path / "judge" / "pending.jsonl").read_text().splitlines()[0])
        assert queued["kind"] == widget_kind and queued["rubric_version"] == version

        store.put_by_key(
            key, {"equivalent": True}, rubric=rubric.RubricStamp(kind=widget_kind, version=version)
        )
        record = VerdictStore(tmp_path / "judge" / "verdicts.jsonl").get_record(payload)
        assert record is not None
        assert record.verdict == {"equivalent": True}
        assert record.rubric_kind == widget_kind and record.rubric_version == version
