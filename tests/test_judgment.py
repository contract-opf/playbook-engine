"""Tests for the batched, content-addressed judgment runtime — issue #62.

Acceptance criteria verified here:

  AC-1: Second compile run over the same corpus → second run makes ZERO judge
        calls (all verdict-cache hits).
  AC-2: Two documents containing an identical ambiguous clause → the judge is
        invoked ONCE, not twice (cross-document dedup).
  AC-3: A batched stub receives MORE than one payload in a single judge() call
        when multiple ambiguous clauses exist (assert batch_size > 1).
  AC-4: Cache key includes model/prompt identity — changing the model identifier
        INVALIDATES cached verdicts (cache miss).
  AC-5: The judge_error path is exercised and handled centrally in this layer.

SECURITY NOTE: All fixtures use programmatically constructed synthetic content.
No real agreement files or real party names are used.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from playbook_engine.clause_classifier import ClassificationHint, ClauseClassification
from playbook_engine.clause_tree import ClauseNode, ClauseTree
from playbook_engine.config import AgreementType
from playbook_engine.deviation_classifier import DeviationResult, RiskDelta
from playbook_engine.judgment import (
    BatchedClassificationJudge,
    BatchedDeviationJudge,
    BatchedScopeJudge,
    JudgmentCache,
    _payload_key,
)
from playbook_engine.scope_gate import ScopeDecision

# ---------------------------------------------------------------------------
# Helpers — synthetic judges and trees
# ---------------------------------------------------------------------------


def _make_agreement_type(type_id: str = "test-agreement") -> AgreementType:
    return AgreementType(id=type_id, name="Test Agreement", description=None)


def _make_clause_tree(headings: list[str]) -> ClauseTree:
    """Build a minimal ClauseTree from a list of headings."""
    nodes = [
        ClauseNode(
            heading=h,
            text=f"Text for {h}.",
            clause_path=str(i + 1),
            char_span=(i * 20, (i + 1) * 20),
        )
        for i, h in enumerate(headings)
    ]
    return ClauseTree(document_id="test-doc", version="v1", source_file="test.rtf", nodes=nodes)


def _make_clause_node(heading: str, text: str, clause_path: str = "1") -> ClauseNode:
    """Build a minimal ClauseNode with required fields."""
    return ClauseNode(heading=heading, text=text, clause_path=clause_path, char_span=(0, len(text)))


@dataclass
class _CountingScopeJudge:
    """Scope judge that counts calls and records which trees it received."""

    call_count: int = 0
    result: ScopeDecision = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.result is None:
            self.result = ScopeDecision(
                in_scope=True,
                scope_rationale="Synthetic in-scope verdict.",
                scope_confidence=0.9,
                basis="judge",
            )

    def judge(self, tree: ClauseTree, agreement_type: AgreementType) -> ScopeDecision:
        self.call_count += 1
        return self.result


@dataclass
class _CountingClassificationJudge:
    """Classification judge that counts batch calls and records batch sizes."""

    call_count: int = 0
    batch_sizes: list[int] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.batch_sizes is None:
            self.batch_sizes = []

    def classify_batch(
        self,
        nodes: list[ClauseNode],
        taxonomy: Any,
        hints: Any = None,
    ) -> list[ClauseClassification]:
        self.call_count += 1
        self.batch_sizes.append(len(nodes))
        return [
            ClauseClassification(taxonomy_id="tax-001", confidence=0.8, basis="judge")
            for _ in nodes
        ]


@dataclass
class _CountingDeviationJudge:
    """Deviation judge that counts batch calls and records batch sizes."""

    call_count: int = 0
    batch_sizes: list[int] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.batch_sizes is None:
            self.batch_sizes = []

    def assess_batch(
        self,
        items: list[dict[str, str]],
        our_standard: str,
    ) -> list[DeviationResult]:
        self.call_count += 1
        self.batch_sizes.append(len(items))
        return [
            DeviationResult(
                deviation="substantive",
                risk_delta=RiskDelta(direction="worse", magnitude="minor"),
                basis="judge",
            )
            for _ in items
        ]


@dataclass
class _RaisingClassificationJudge:
    """Classification judge that always raises (for judge_error testing)."""

    def classify_batch(
        self,
        nodes: list[ClauseNode],
        taxonomy: Any,
        hints: Any = None,
    ) -> list[ClauseClassification]:
        raise RuntimeError("Simulated LLM timeout")


@dataclass
class _HintEchoingClassificationJudge:
    """Classification judge that echoes each hint's ``best_id`` as ``taxonomy_id``.

    Enforces the ``ClassificationJudge`` contract with ``zip(..., strict=True)``:
    ``hints`` must be positional, one per node, same order and same length as
    ``nodes`` — exactly like a real hint-sensitive delegate would require.
    """

    def classify_batch(
        self,
        nodes: list[ClauseNode],
        taxonomy: Any,
        hints: list[ClassificationHint | None] | None = None,
    ) -> list[ClauseClassification]:
        if hints is None:
            hints = [None] * len(nodes)
        return [
            ClauseClassification(
                taxonomy_id=hint.best_id if hint is not None else None,
                confidence=hint.best_sim if hint is not None else 0.0,
                basis="judge" if hint is not None else "unclassified",
            )
            for _node, hint in zip(nodes, hints, strict=True)
        ]


@dataclass
class _RaisingScopeJudge:
    """Scope judge that always raises."""

    def judge(self, tree: ClauseTree, agreement_type: AgreementType) -> ScopeDecision:
        raise RuntimeError("Simulated LLM refusal")


@dataclass
class _RaisingDeviationJudge:
    """Deviation judge that always raises."""

    def assess_batch(self, items: list[dict[str, str]], our_standard: str) -> list[DeviationResult]:
        raise RuntimeError("Simulated rate-limit")


@dataclass
class _ScriptedScopeJudge:
    """Scope judge returning one scripted result per call, in order.

    Simulates a store-backed judge whose backing verdict store gains a real
    verdict between two rounds sharing the same persisted ``JudgmentCache``
    file (issue #182 "B4") — round 1 = store miss/error, round 2 (identical
    payload) = store hit.
    """

    results: list[ScopeDecision]
    call_count: int = 0

    def judge(self, tree: ClauseTree, agreement_type: AgreementType) -> ScopeDecision:
        result = self.results[self.call_count]
        self.call_count += 1
        return result


@dataclass
class _ScriptedClassificationJudge:
    """Classification judge returning one scripted batch result per call.

    See :class:`_ScriptedScopeJudge` — same round-1-miss/round-2-hit shape,
    for classification.
    """

    results: list[ClauseClassification]
    call_count: int = 0

    def classify_batch(
        self,
        nodes: list[ClauseNode],
        taxonomy: Any,
        hints: Any = None,
    ) -> list[ClauseClassification]:
        result = self.results[self.call_count]
        self.call_count += 1
        return [result for _ in nodes]


@dataclass
class _ScriptedDeviationJudge:
    """Deviation judge returning one scripted batch result per call.

    See :class:`_ScriptedScopeJudge` — same round-1-miss/round-2-hit shape,
    for deviation.
    """

    results: list[DeviationResult]
    call_count: int = 0

    def assess_batch(self, items: list[dict[str, str]], our_standard: str) -> list[DeviationResult]:
        result = self.results[self.call_count]
        self.call_count += 1
        return [result for _ in items]


@dataclass
class _FakeTaxonomy:
    entries: list[Any]


@dataclass
class _FakeTaxEntry:
    id: str


# ---------------------------------------------------------------------------
# JudgmentCache unit tests
# ---------------------------------------------------------------------------


class TestJudgmentCache:
    """Basic cache store/retrieve mechanics."""

    def test_get_miss_returns_none(self, tmp_path: Path) -> None:
        cache = JudgmentCache(tmp_path / "verdicts.jsonl", model_id="stub-v1")
        result = cache.get({"stage": "scope", "some": "payload"})
        assert result is None

    def test_put_then_get_returns_verdict(self, tmp_path: Path) -> None:
        cache = JudgmentCache(tmp_path / "verdicts.jsonl", model_id="stub-v1")
        payload = {"stage": "scope", "x": 1}
        verdict = {
            "in_scope": True,
            "scope_rationale": "ok",
            "scope_confidence": 0.9,
            "basis": "judge",
        }
        cache.put(payload, verdict)
        assert cache.get(payload) == verdict

    def test_persists_across_instances(self, tmp_path: Path) -> None:
        """Cache must survive a process restart (new instance reads from disk)."""
        path = tmp_path / "verdicts.jsonl"
        payload = {"stage": "test", "v": 42}
        verdict = {"result": "cached"}

        cache1 = JudgmentCache(path, model_id="model-a")
        cache1.put(payload, verdict)

        cache2 = JudgmentCache(path, model_id="model-a")
        assert cache2.get(payload) == verdict

    def test_model_id_change_invalidates_verdicts(self, tmp_path: Path) -> None:
        """AC-4: different model_id → different cache key → miss."""
        path = tmp_path / "verdicts.jsonl"
        payload = {"stage": "scope", "clauses": ["indemnification"]}

        cache_a = JudgmentCache(path, model_id="model-a")
        cache_a.put(payload, {"verdict": "a"})

        cache_b = JudgmentCache(path, model_id="model-b")
        # model-b never stored this payload → miss.
        assert cache_b.get(payload) is None

    def test_hit_count_increments_on_hit(self, tmp_path: Path) -> None:
        cache = JudgmentCache(tmp_path / "v.jsonl", model_id="x")
        payload = {"k": "v"}
        cache.put(payload, {"r": 1})
        assert cache.hit_count == 0
        cache.get(payload)
        assert cache.hit_count == 1

    def test_miss_count_increments_on_put(self, tmp_path: Path) -> None:
        cache = JudgmentCache(tmp_path / "v.jsonl", model_id="x")
        assert cache.miss_count == 0
        cache.put({"k": 1}, {"r": 1})
        assert cache.miss_count == 1
        cache.put({"k": 2}, {"r": 2})
        assert cache.miss_count == 2

    def test_payload_key_stable(self) -> None:
        """Same payload + model_id → same key."""
        p = {"a": 1, "b": [1, 2]}
        k1 = _payload_key(p, "m")
        k2 = _payload_key(p, "m")
        assert k1 == k2

    def test_payload_key_differs_on_model_id_change(self) -> None:
        """AC-4 key-level: different model_id → different hash."""
        p = {"a": 1}
        assert _payload_key(p, "model-a") != _payload_key(p, "model-b")

    def test_corrupt_line_skipped_silently(self, tmp_path: Path) -> None:
        """Corrupt lines in the JSONL file must not crash startup."""
        path = tmp_path / "verdicts.jsonl"
        path.write_text('{"key":"k1","verdict":{"r":1}}\nNOT-JSON\n', encoding="utf-8")
        cache = JudgmentCache(path, model_id="x")
        # k1 was valid and should load; corrupt line is silently skipped.
        assert cache._store.get("k1") == {"r": 1}


# ---------------------------------------------------------------------------
# BatchedScopeJudge tests
# ---------------------------------------------------------------------------


class TestBatchedScopeJudge:
    """AC-1 (scope), AC-4, AC-5 for scope judging."""

    def test_first_call_delegates_to_underlying_judge(self, tmp_path: Path) -> None:
        cache = JudgmentCache(tmp_path / "v.jsonl", model_id="stub-v1")
        delegate = _CountingScopeJudge()
        judge = BatchedScopeJudge(delegate=delegate, cache=cache)
        tree = _make_clause_tree(["Indemnification", "Governing Law"])

        result = judge.judge(tree, _make_agreement_type())
        assert delegate.call_count == 1
        assert result.in_scope is True

    def test_second_call_same_tree_is_cache_hit(self, tmp_path: Path) -> None:
        """AC-1 (scope): second call with identical tree → zero new judge calls."""
        cache = JudgmentCache(tmp_path / "v.jsonl", model_id="stub-v1")
        delegate = _CountingScopeJudge()
        judge = BatchedScopeJudge(delegate=delegate, cache=cache)
        tree = _make_clause_tree(["Indemnification", "Governing Law"])

        judge.judge(tree, _make_agreement_type())
        judge.judge(tree, _make_agreement_type())

        assert delegate.call_count == 1, (
            "Identical tree on second call must be a cache hit (zero new judge calls)"
        )

    def test_cross_run_cache_hit(self, tmp_path: Path) -> None:
        """AC-1: second run (new judge instance, same cache file) → zero calls."""
        cache_path = tmp_path / "v.jsonl"
        tree = _make_clause_tree(["Indemnification", "Governing Law"])

        # First run.
        delegate1 = _CountingScopeJudge()
        judge1 = BatchedScopeJudge(
            delegate=delegate1, cache=JudgmentCache(cache_path, model_id="m1")
        )
        judge1.judge(tree, _make_agreement_type())
        assert delegate1.call_count == 1

        # Second run — new judge instance, same cache file.
        delegate2 = _CountingScopeJudge()
        judge2 = BatchedScopeJudge(
            delegate=delegate2, cache=JudgmentCache(cache_path, model_id="m1")
        )
        judge2.judge(tree, _make_agreement_type())
        assert delegate2.call_count == 0, (
            "New JudgmentCache instance loading persisted file must produce a cache hit"
        )

    def test_judge_error_returns_retain_sentinel(self, tmp_path: Path) -> None:
        """AC-5 (scope): judge error → retain with judge_error basis; no propagation."""
        cache = JudgmentCache(tmp_path / "v.jsonl", model_id="stub-v1")
        judge = BatchedScopeJudge(delegate=_RaisingScopeJudge(), cache=cache)
        result = judge.judge(_make_clause_tree(["Term"]), _make_agreement_type())

        assert result.basis == "judge_error"
        assert result.in_scope is True, "Document must be retained (never silently dropped)"
        assert result.scope_confidence == 0.0

    def test_judge_error_is_not_cached_and_rechecks_delegate(self, tmp_path: Path) -> None:
        """A judge_error verdict must NOT be cached (issue #182 "B4"): once
        cached, a later round for the exact same document would keep
        replaying the stale error forever instead of ever re-consulting the
        delegate — exactly the ``rm -rf out/.cache`` symptom from the
        overnight corpus run. Round 1 errors; round 2 (same tree, same
        cache) must reach the delegate again and get its resolved verdict.
        """
        cache = JudgmentCache(tmp_path / "v.jsonl", model_id="stub-v1")
        errored = ScopeDecision(
            in_scope=True,
            scope_rationale="Errored on round 1.",
            scope_confidence=0.0,
            basis="judge_error",
        )
        resolved = ScopeDecision(
            in_scope=True,
            scope_rationale="Resolved on round 2.",
            scope_confidence=0.95,
            basis="judge",
        )
        delegate = _ScriptedScopeJudge(results=[errored, resolved])
        judge = BatchedScopeJudge(delegate=delegate, cache=cache)
        tree = _make_clause_tree(["Indemnification"])
        agreement_type = _make_agreement_type()

        round1 = judge.judge(tree, agreement_type)
        assert round1.basis == "judge_error"

        round2 = judge.judge(tree, agreement_type)
        assert round2.basis == "judge"
        assert round2.scope_confidence == 0.95
        assert delegate.call_count == 2, "judge_error must never be served from cache"

    def test_model_id_change_invalidates_scope_cache(self, tmp_path: Path) -> None:
        """AC-4 (scope): changing model_id → cache miss → fresh judge call."""
        cache_path = tmp_path / "v.jsonl"
        tree = _make_clause_tree(["Indemnification"])

        delegate1 = _CountingScopeJudge()
        j1 = BatchedScopeJudge(delegate=delegate1, cache=JudgmentCache(cache_path, model_id="m1"))
        j1.judge(tree, _make_agreement_type())

        delegate2 = _CountingScopeJudge()
        j2 = BatchedScopeJudge(delegate=delegate2, cache=JudgmentCache(cache_path, model_id="m2"))
        j2.judge(tree, _make_agreement_type())

        assert delegate2.call_count == 1, (
            "Changed model_id must invalidate cached verdict → judge called again"
        )


# ---------------------------------------------------------------------------
# BatchedClassificationJudge tests
# ---------------------------------------------------------------------------


class TestBatchedClassificationJudge:
    """AC-2, AC-3, AC-4, AC-5 for classification judging."""

    def _taxonomy(self, *ids: str) -> _FakeTaxonomy:
        return _FakeTaxonomy(entries=[_FakeTaxEntry(id=i) for i in ids])

    def test_first_call_delegates_all_nodes(self, tmp_path: Path) -> None:
        cache = JudgmentCache(tmp_path / "v.jsonl", model_id="stub-v1")
        delegate = _CountingClassificationJudge()
        judge = BatchedClassificationJudge(delegate=delegate, cache=cache)
        nodes = [_make_clause_node("Indemnification", "Text.", clause_path="1")]
        taxonomy = self._taxonomy("tax-001", "tax-002")

        results = judge.classify_batch(nodes, taxonomy)
        assert len(results) == 1
        assert delegate.call_count == 1

    def test_cross_doc_dedup_second_call_is_cache_hit(self, tmp_path: Path) -> None:
        """AC-2: identical clause appearing in two documents → judge called ONCE."""
        cache = JudgmentCache(tmp_path / "v.jsonl", model_id="stub-v1")
        delegate = _CountingClassificationJudge()
        judge = BatchedClassificationJudge(delegate=delegate, cache=cache)
        taxonomy = self._taxonomy("tax-001")

        # Same heading/text → same payload hash.
        node1 = _make_clause_node("Confidentiality", "Keep it secret.", clause_path="1")
        node2 = _make_clause_node("Confidentiality", "Keep it secret.", clause_path="1")

        judge.classify_batch([node1], taxonomy)
        judge.classify_batch([node2], taxonomy)

        assert delegate.call_count == 1, (
            "Identical clause in two calls must produce exactly one judge call (cross-doc dedup)"
        )

    def test_intra_batch_dedup_single_dispatch(self, tmp_path: Path) -> None:
        """AC-2 (intra-batch): two identical clauses in ONE classify_batch call → one delegate call.

        This exercises the corpus-wide single-dispatch case described in issue #62:
        payloads are gathered corpus-wide and dispatched in a single batch.  Duplicate
        payloads within that batch must produce exactly one payload to the delegate.
        """
        cache = JudgmentCache(tmp_path / "v.jsonl", model_id="stub-v1")
        delegate = _CountingClassificationJudge()
        judge = BatchedClassificationJudge(delegate=delegate, cache=cache)
        taxonomy = self._taxonomy("tax-001")

        # Two nodes with identical heading and text — same payload key.
        node1 = _make_clause_node("Confidentiality", "Keep it secret.", clause_path="1")
        node2 = _make_clause_node("Confidentiality", "Keep it secret.", clause_path="2")

        results = judge.classify_batch([node1, node2], taxonomy)

        assert len(results) == 2, "Result list must match input length"
        assert delegate.call_count == 1, (
            "Two identical clauses in one batch must produce exactly one delegate call"
        )
        assert delegate.batch_sizes[0] == 1, (
            "Delegate must receive only one (deduplicated) payload, not two"
        )

    def test_batch_size_greater_than_one(self, tmp_path: Path) -> None:
        """AC-3: multiple uncached nodes in one call → delegate receives batch_size > 1."""
        cache = JudgmentCache(tmp_path / "v.jsonl", model_id="stub-v1")
        delegate = _CountingClassificationJudge()
        judge = BatchedClassificationJudge(delegate=delegate, cache=cache)
        taxonomy = self._taxonomy("tax-001")

        nodes = [
            _make_clause_node("Term", "One year.", clause_path="1"),
            _make_clause_node("Payment", "Monthly.", clause_path="2"),
            _make_clause_node("Warranty", "As-is.", clause_path="3"),
        ]
        judge.classify_batch(nodes, taxonomy)

        assert delegate.call_count == 1
        assert delegate.batch_sizes[0] > 1, (
            "Delegate must receive all uncached nodes in one batch call"
        )
        assert delegate.batch_sizes[0] == 3

    def test_mixed_cache_hit_and_miss(self, tmp_path: Path) -> None:
        """Nodes already cached are not re-sent; only new nodes go to the delegate."""
        cache = JudgmentCache(tmp_path / "v.jsonl", model_id="stub-v1")
        delegate = _CountingClassificationJudge()
        judge = BatchedClassificationJudge(delegate=delegate, cache=cache)
        taxonomy = self._taxonomy("tax-001")

        node_a = _make_clause_node("Term", "One year.", clause_path="1")
        node_b = _make_clause_node("Payment", "Monthly fees.", clause_path="2")

        # Cache node_a only.
        judge.classify_batch([node_a], taxonomy)
        assert delegate.call_count == 1
        assert delegate.batch_sizes[-1] == 1

        # Second call: node_a hits, node_b misses.
        judge.classify_batch([node_a, node_b], taxonomy)
        assert delegate.call_count == 2
        assert delegate.batch_sizes[-1] == 1, "Only the uncached node_b should be in the batch"

    def test_second_call_same_nodes_zero_judge_calls(self, tmp_path: Path) -> None:
        """AC-1 (classification): repeat call with same nodes → zero new judge calls."""
        cache = JudgmentCache(tmp_path / "v.jsonl", model_id="stub-v1")
        delegate = _CountingClassificationJudge()
        judge = BatchedClassificationJudge(delegate=delegate, cache=cache)
        taxonomy = self._taxonomy("tax-001")

        nodes = [_make_clause_node("Indemnification", "Big liability.", clause_path="1")]
        judge.classify_batch(nodes, taxonomy)
        judge.classify_batch(nodes, taxonomy)

        assert delegate.call_count == 1, "Second call with same nodes must be a cache hit"

    def test_judge_error_returns_error_sentinel_per_node(self, tmp_path: Path) -> None:
        """AC-5: raising judge → all nodes get basis='judge_error'; no propagation."""
        cache = JudgmentCache(tmp_path / "v.jsonl", model_id="stub-v1")
        judge = BatchedClassificationJudge(delegate=_RaisingClassificationJudge(), cache=cache)
        taxonomy = self._taxonomy("tax-001")

        nodes = [
            _make_clause_node("Term", "One year.", clause_path="1"),
            _make_clause_node("Payment", "Monthly.", clause_path="2"),
        ]
        results = judge.classify_batch(nodes, taxonomy)

        assert len(results) == 2
        for r in results:
            assert r.basis == "judge_error"
            assert r.taxonomy_id is None
            assert r.confidence == 0.0

    def test_needs_review_is_not_cached_and_rechecks_delegate(self, tmp_path: Path) -> None:
        """A needs_review verdict (store-backed judge miss) must NOT be
        cached (issue #182 "B4"): the real-world manifestation was a
        drain-loop round where ``playbook judge-apply`` had already added the
        real verdict to the store, but the SAME persisted
        ``out/.cache/verdicts.jsonl`` kept serving the stale needs_review
        sentinel from an earlier round — forcing a manual
        ``rm -rf out/.cache`` to make any progress. Round 1 misses; round 2
        (identical clause payload, same cache) must reach the delegate again
        and get its now-resolved verdict.
        """
        cache = JudgmentCache(tmp_path / "v.jsonl", model_id="stub-v1")
        node = _make_clause_node("Indemnification", "The party shall indemnify...", "1")
        taxonomy = self._taxonomy("tax-001")

        delegate = _ScriptedClassificationJudge(
            results=[
                ClauseClassification(taxonomy_id=None, confidence=0.0, basis="needs_review"),
                ClauseClassification(taxonomy_id="tax-001", confidence=0.9, basis="judge"),
            ]
        )
        judge = BatchedClassificationJudge(delegate=delegate, cache=cache)

        round1 = judge.classify_batch([node], taxonomy)
        assert round1[0].basis == "needs_review"

        round2 = judge.classify_batch([node], taxonomy)
        assert round2[0].basis == "judge"
        assert round2[0].taxonomy_id == "tax-001"
        assert delegate.call_count == 2, "needs_review must never be served from cache"

    def test_needs_review_is_not_cached_across_separate_cache_instances(
        self, tmp_path: Path
    ) -> None:
        """Same as above, but across two separate ``JudgmentCache`` instances
        pointed at the same persisted file — the shape a real drain loop
        actually takes: each ``playbook judge``/``mine`` invocation is a new
        process that reconstructs the cache from ``out/.cache/verdicts.jsonl``.
        """
        cache_path = tmp_path / "v.jsonl"
        node = _make_clause_node("Confidentiality", "Keep it secret.", "1")
        taxonomy = self._taxonomy("tax-001")

        delegate1 = _ScriptedClassificationJudge(
            results=[ClauseClassification(taxonomy_id=None, confidence=0.0, basis="needs_review")]
        )
        judge1 = BatchedClassificationJudge(
            delegate=delegate1, cache=JudgmentCache(cache_path, model_id="stub-v1")
        )
        round1 = judge1.classify_batch([node], taxonomy)
        assert round1[0].basis == "needs_review"

        delegate2 = _ScriptedClassificationJudge(
            results=[ClauseClassification(taxonomy_id="tax-001", confidence=0.9, basis="judge")]
        )
        judge2 = BatchedClassificationJudge(
            delegate=delegate2, cache=JudgmentCache(cache_path, model_id="stub-v1")
        )
        round2 = judge2.classify_batch([node], taxonomy)

        assert round2[0].basis == "judge"
        assert round2[0].taxonomy_id == "tax-001"
        assert delegate2.call_count == 1, (
            "A fresh process loading the persisted cache file must still re-check the "
            "delegate for a previously needs_review payload, not replay the stale miss"
        )

    def test_dedup_realigns_hints_to_unique_nodes(self, tmp_path: Path) -> None:
        """Issue #110: hints must stay positionally aligned with the deduped
        delegate batch, not the original full-length node list.

        Two duplicate nodes (same payload) come first, followed by a distinct
        node, so ``len(hints) != len(unique_nodes)``. The fake delegate
        enforces one-hint-per-node with ``zip(..., strict=True)`` — if the
        wrapper forwarded the original, un-deduped hints list, this would
        raise ``ValueError`` (length mismatch) instead of misclassifying
        silently.
        """
        cache = JudgmentCache(tmp_path / "v.jsonl", model_id="stub-v1")
        judge = BatchedClassificationJudge(delegate=_HintEchoingClassificationJudge(), cache=cache)
        taxonomy = self._taxonomy("tax-001", "tax-002", "tax-003")

        dup1 = _make_clause_node("Confidentiality", "Keep it secret.", clause_path="1")
        dup2 = _make_clause_node("Confidentiality", "Keep it secret.", clause_path="2")
        distinct = _make_clause_node("Term", "One year.", clause_path="3")

        hints = [
            ClassificationHint(best_id="tax-001", best_sim=0.91),  # dup1's own hint (canonical)
            ClassificationHint(best_id="tax-999", best_sim=0.10),  # dup2's hint — must be dropped
            ClassificationHint(best_id="tax-003", best_sim=0.77),  # distinct's hint
        ]

        results = judge.classify_batch([dup1, dup2, distinct], taxonomy, hints)

        assert len(results) == 3
        # dup1 and dup2 share one delegate call keyed off dup1 (canonical, first-seen) —
        # both must reflect dup1's hint, never dup2's.
        assert results[0].taxonomy_id == "tax-001"
        assert results[1].taxonomy_id == "tax-001"
        # distinct's own hint must be preserved, correctly paired via its own index.
        assert results[2].taxonomy_id == "tax-003"
        assert results[2].confidence == 0.77

    def test_model_id_change_invalidates_classification_cache(self, tmp_path: Path) -> None:
        """AC-4 (classification): changing model_id → new judge call."""
        cache_path = tmp_path / "v.jsonl"
        taxonomy = self._taxonomy("tax-001")
        node = _make_clause_node("Confidentiality", "Keep it secret.", clause_path="1")

        delegate1 = _CountingClassificationJudge()
        j1 = BatchedClassificationJudge(
            delegate=delegate1, cache=JudgmentCache(cache_path, model_id="m1")
        )
        j1.classify_batch([node], taxonomy)

        delegate2 = _CountingClassificationJudge()
        j2 = BatchedClassificationJudge(
            delegate=delegate2, cache=JudgmentCache(cache_path, model_id="m2")
        )
        j2.classify_batch([node], taxonomy)

        assert delegate2.call_count == 1, "model_id change must invalidate cached verdicts"

    def test_returns_correct_count(self, tmp_path: Path) -> None:
        """Result list length matches input node count."""
        cache = JudgmentCache(tmp_path / "v.jsonl", model_id="stub-v1")
        delegate = _CountingClassificationJudge()
        judge = BatchedClassificationJudge(delegate=delegate, cache=cache)
        taxonomy = self._taxonomy("tax-001")

        nodes = [
            _make_clause_node(f"Clause{i}", f"Text {i}.", clause_path=str(i + 1)) for i in range(5)
        ]
        results = judge.classify_batch(nodes, taxonomy)
        assert len(results) == 5


# ---------------------------------------------------------------------------
# BatchedDeviationJudge tests
# ---------------------------------------------------------------------------


class TestBatchedDeviationJudge:
    """AC-1, AC-2, AC-3, AC-4, AC-5 for deviation judging."""

    def test_first_call_delegates(self, tmp_path: Path) -> None:
        cache = JudgmentCache(tmp_path / "v.jsonl", model_id="stub-v1")
        delegate = _CountingDeviationJudge()
        judge = BatchedDeviationJudge(delegate=delegate, cache=cache)
        items = [{"hunk": "[BEFORE]\nold text\n[AFTER]\nnew text"}]

        results = judge.assess_batch(items, our_standard="Standard text.")
        assert len(results) == 1
        assert delegate.call_count == 1

    def test_second_call_same_items_is_cache_hit(self, tmp_path: Path) -> None:
        """AC-1 (deviation): repeat call → zero new judge calls."""
        cache = JudgmentCache(tmp_path / "v.jsonl", model_id="stub-v1")
        delegate = _CountingDeviationJudge()
        judge = BatchedDeviationJudge(delegate=delegate, cache=cache)
        items = [{"hunk": "[BEFORE]\nold\n[AFTER]\nnew"}]

        judge.assess_batch(items, our_standard="Standard.")
        judge.assess_batch(items, our_standard="Standard.")

        assert delegate.call_count == 1, "Identical items on second call must be a cache hit"

    def test_cross_doc_dedup(self, tmp_path: Path) -> None:
        """AC-2 (deviation): same hunk in two docs → judge called once."""
        cache = JudgmentCache(tmp_path / "v.jsonl", model_id="stub-v1")
        delegate = _CountingDeviationJudge()
        judge = BatchedDeviationJudge(delegate=delegate, cache=cache)
        items = [{"hunk": "[BEFORE]\nProvide service\n[AFTER]\nProvide extended service"}]
        standard = "Provide service as agreed."

        judge.assess_batch(items, our_standard=standard)
        judge.assess_batch(items, our_standard=standard)

        assert delegate.call_count == 1, "Same hunk seen in two documents → one judge call"

    def test_intra_batch_dedup_single_dispatch(self, tmp_path: Path) -> None:
        """AC-2 (deviation intra-batch): two identical items in ONE assess_batch → one delegate call.

        This exercises the corpus-wide single-dispatch case described in issue #62:
        duplicate payloads within a single batch must produce exactly one payload
        to the delegate, not two.
        """
        cache = JudgmentCache(tmp_path / "v.jsonl", model_id="stub-v1")
        delegate = _CountingDeviationJudge()
        judge = BatchedDeviationJudge(delegate=delegate, cache=cache)
        identical_item = {"hunk": "[BEFORE]\nProvide service\n[AFTER]\nExtended service"}
        standard = "Provide service as agreed."

        results = judge.assess_batch([identical_item, identical_item], our_standard=standard)

        assert len(results) == 2, "Result list must match input length"
        assert delegate.call_count == 1, (
            "Two identical items in one batch must produce exactly one delegate call"
        )
        assert delegate.batch_sizes[0] == 1, (
            "Delegate must receive only one (deduplicated) payload, not two"
        )

    def test_batch_size_greater_than_one(self, tmp_path: Path) -> None:
        """AC-3 (deviation): multiple uncached items → delegate gets batch_size > 1."""
        cache = JudgmentCache(tmp_path / "v.jsonl", model_id="stub-v1")
        delegate = _CountingDeviationJudge()
        judge = BatchedDeviationJudge(delegate=delegate, cache=cache)
        items = [{"hunk": f"[BEFORE]\nold {i}\n[AFTER]\nnew {i}"} for i in range(4)]

        judge.assess_batch(items, our_standard="Standard.")

        assert delegate.batch_sizes[0] > 1
        assert delegate.batch_sizes[0] == 4

    def test_judge_error_returns_needs_review(self, tmp_path: Path) -> None:
        """AC-5 (deviation): raising judge → needs_review / judge_error basis."""
        cache = JudgmentCache(tmp_path / "v.jsonl", model_id="stub-v1")
        judge = BatchedDeviationJudge(delegate=_RaisingDeviationJudge(), cache=cache)
        items = [{"hunk": "[BEFORE]\na\n[AFTER]\nb"}]

        results = judge.assess_batch(items, our_standard="Standard.")
        assert len(results) == 1
        assert results[0].deviation == "needs_review"
        assert results[0].basis == "judge_error"

    def test_needs_review_is_not_cached_and_rechecks_delegate(self, tmp_path: Path) -> None:
        """A needs_review verdict must NOT be cached (issue #182 "B4") —
        same convergence requirement as classification/scope: round 1 misses,
        round 2 (identical hunk/standard, same cache) must reach the
        delegate again rather than replaying the stale miss forever.
        """
        cache = JudgmentCache(tmp_path / "v.jsonl", model_id="stub-v1")
        items = [{"hunk": "[BEFORE]\nold text\n[AFTER]\nnew text"}]
        standard = "Standard text."

        delegate = _ScriptedDeviationJudge(
            results=[
                DeviationResult(
                    deviation="needs_review",
                    risk_delta=RiskDelta("neutral", "none"),
                    basis="needs_review",
                ),
                DeviationResult(
                    deviation="substantive",
                    risk_delta=RiskDelta("worse", "minor"),
                    basis="judge",
                ),
            ]
        )
        judge = BatchedDeviationJudge(delegate=delegate, cache=cache)

        round1 = judge.assess_batch(items, our_standard=standard)
        assert round1[0].basis == "needs_review"

        round2 = judge.assess_batch(items, our_standard=standard)
        assert round2[0].basis == "judge"
        assert round2[0].deviation == "substantive"
        assert delegate.call_count == 2, "needs_review must never be served from cache"

    def test_model_id_change_invalidates_deviation_cache(self, tmp_path: Path) -> None:
        """AC-4 (deviation): changing model_id → cache miss → fresh judge call."""
        cache_path = tmp_path / "v.jsonl"
        items = [{"hunk": "[BEFORE]\nold\n[AFTER]\nnew"}]
        standard = "Standard."

        delegate1 = _CountingDeviationJudge()
        j1 = BatchedDeviationJudge(
            delegate=delegate1, cache=JudgmentCache(cache_path, model_id="m1")
        )
        j1.assess_batch(items, our_standard=standard)

        delegate2 = _CountingDeviationJudge()
        j2 = BatchedDeviationJudge(
            delegate=delegate2, cache=JudgmentCache(cache_path, model_id="m2")
        )
        j2.assess_batch(items, our_standard=standard)

        assert delegate2.call_count == 1, "model_id change must invalidate cached verdicts"

    def test_verdict_round_trips_correctly(self, tmp_path: Path) -> None:
        """Cached DeviationResult must survive serialisation round-trip intact."""
        cache = JudgmentCache(tmp_path / "v.jsonl", model_id="stub-v1")
        delegate = _CountingDeviationJudge()
        judge = BatchedDeviationJudge(delegate=delegate, cache=cache)
        items = [{"hunk": "[BEFORE]\nold text\n[AFTER]\nnew text"}]

        result_first = judge.assess_batch(items, our_standard="Standard text.")

        # Second call hits cache; reconstruct from serialised form.
        result_second = judge.assess_batch(items, our_standard="Standard text.")

        assert result_first[0].deviation == result_second[0].deviation
        assert result_first[0].risk_delta.direction == result_second[0].risk_delta.direction
        assert result_first[0].risk_delta.magnitude == result_second[0].risk_delta.magnitude
        assert result_first[0].basis == result_second[0].basis


# ---------------------------------------------------------------------------
# Finding 4 — Lossy cache key boundary characterisation tests
# ---------------------------------------------------------------------------


class TestLossyCacheKeyBoundaries:
    """Characterisation tests that pin the known lossy-key trade-offs in judgment.py.

    Two cache key constructions are intentionally lossy:

      Classification key — clause text is truncated to text[:500].
        Two clauses identical in their first 500 chars but differing afterwards
        will hash to the same key and share a verdict.  This is documented as an
        accepted trade-off (see judgment.py).  These tests assert the current
        behaviour so any silent change is caught.

      Scope key — only heading text (up to 20 headings) is included; body text
        is excluded.  Two documents with identical heading sets but different body
        text will share a scope verdict.  This is documented as an accepted
        trade-off (see judgment.py).  These tests assert the current behaviour.
    """

    def _taxonomy(self, *ids: str) -> _FakeTaxonomy:
        return _FakeTaxonomy(entries=[_FakeTaxEntry(id=i) for i in ids])

    # ------------------------------------------------------------------
    # Classification key — text[:500] truncation
    # ------------------------------------------------------------------

    def test_classification_clauses_identical_in_first_500_chars_share_cache_key(
        self, tmp_path: Path
    ) -> None:
        """Characterisation: two clauses identical in first 500 chars but differing after
        produce the same classification cache key (known trade-off — text[:500] truncation).

        The delegate is called ONCE, not twice, because both payloads hash identically.
        If this assertion starts failing, the truncation limit has been changed — update
        the comment in judgment.py and bump model_id to bust the cache.
        """
        cache = JudgmentCache(tmp_path / "v.jsonl", model_id="stub-v1")
        delegate = _CountingClassificationJudge()
        judge = BatchedClassificationJudge(delegate=delegate, cache=cache)
        taxonomy = self._taxonomy("tax-001")

        shared_prefix = "A" * 500
        # node_a and node_b are identical in chars 0–499; differ only afterwards.
        node_a = _make_clause_node("Confidentiality", shared_prefix + " UNIQUE-A", clause_path="1")
        node_b = _make_clause_node("Confidentiality", shared_prefix + " UNIQUE-B", clause_path="2")

        # Both nodes in a single batch: only one unique payload (truncated key is the same).
        results = judge.classify_batch([node_a, node_b], taxonomy)

        assert len(results) == 2, "Result list must match input length"
        # Accepted trade-off: two nodes with the same first 500 chars collide to one
        # delegate call.  If this changes, the trade-off decision must be revisited.
        assert delegate.call_count == 1, (
            "text[:500] truncation means clauses differing only past char 500 share a key "
            "(accepted trade-off — see judgment.py comment)"
        )
        assert delegate.batch_sizes[0] == 1, (
            "Only one deduplicated payload should reach the delegate"
        )

    def test_classification_clauses_differing_within_first_500_chars_get_distinct_keys(
        self, tmp_path: Path
    ) -> None:
        """Sanity check: clauses that differ within first 500 chars get distinct keys."""
        cache = JudgmentCache(tmp_path / "v.jsonl", model_id="stub-v1")
        delegate = _CountingClassificationJudge()
        judge = BatchedClassificationJudge(delegate=delegate, cache=cache)
        taxonomy = self._taxonomy("tax-001")

        node_a = _make_clause_node(
            "Confidentiality", "Text variant A — early difference.", clause_path="1"
        )
        node_b = _make_clause_node(
            "Confidentiality", "Text variant B — early difference.", clause_path="2"
        )

        results = judge.classify_batch([node_a, node_b], taxonomy)

        assert len(results) == 2
        # Both nodes differ within first 500 chars → two distinct payloads → two calls.
        assert delegate.batch_sizes[0] == 2, (
            "Clauses differing within first 500 chars must produce distinct cache keys"
        )

    # ------------------------------------------------------------------
    # Scope key — headings-only (no body text)
    # ------------------------------------------------------------------

    def test_scope_documents_with_same_headings_but_different_body_share_verdict(
        self, tmp_path: Path
    ) -> None:
        """Characterisation: two documents with identical heading sets but different body text
        produce the same scope cache key and share a verdict (known trade-off — headings-only
        scope key).

        The delegate is called ONCE across both judge() calls.
        If this assertion starts failing, the scope key construction has been changed —
        update the comment in judgment.py and bump model_id to bust the cache.
        """
        cache = JudgmentCache(tmp_path / "v.jsonl", model_id="stub-v1")
        delegate = _CountingScopeJudge()
        judge = BatchedScopeJudge(delegate=delegate, cache=cache)
        agreement_type = _make_agreement_type()

        headings = ["Indemnification", "Governing Law", "Term"]

        # tree_a and tree_b have identical heading sets; body text is different.
        tree_a = ClauseTree(
            document_id="doc-a",
            version="v1",
            source_file="a.rtf",
            nodes=[
                ClauseNode(
                    heading=h,
                    text=f"Body text for {h} in document A — unique content AAA.",
                    clause_path=str(i + 1),
                    char_span=(i * 50, (i + 1) * 50),
                )
                for i, h in enumerate(headings)
            ],
        )
        tree_b = ClauseTree(
            document_id="doc-b",
            version="v1",
            source_file="b.rtf",
            nodes=[
                ClauseNode(
                    heading=h,
                    text=f"Body text for {h} in document B — completely different BBB.",
                    clause_path=str(i + 1),
                    char_span=(i * 50, (i + 1) * 50),
                )
                for i, h in enumerate(headings)
            ],
        )

        judge.judge(tree_a, agreement_type)
        judge.judge(tree_b, agreement_type)

        # Accepted trade-off: same headings → same scope key → one delegate call.
        assert delegate.call_count == 1, (
            "headings-only scope key means two docs with identical headings share a verdict "
            "(accepted trade-off — see judgment.py comment)"
        )

    def test_scope_documents_with_different_headings_get_distinct_verdicts(
        self, tmp_path: Path
    ) -> None:
        """Sanity check: documents with different heading sets get distinct scope verdicts."""
        cache = JudgmentCache(tmp_path / "v.jsonl", model_id="stub-v1")
        delegate = _CountingScopeJudge()
        judge = BatchedScopeJudge(delegate=delegate, cache=cache)
        agreement_type = _make_agreement_type()

        tree_a = _make_clause_tree(["Indemnification", "Governing Law"])
        tree_b = _make_clause_tree(["Payment Terms", "Warranty"])

        judge.judge(tree_a, agreement_type)
        judge.judge(tree_b, agreement_type)

        assert delegate.call_count == 2, (
            "Documents with distinct heading sets must produce distinct scope cache keys"
        )
