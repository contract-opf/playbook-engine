"""Batched, content-addressed judgment runtime — issue #62.

Wraps the existing judge protocols (ScopeJudge, ClassificationJudge,
DeviationJudge) with:

  1. **Payload collection** — callers push payloads into a ``JudgmentRuntime``
     instance before dispatch.  The runtime batches all payloads in a single
     ``judge()`` call instead of calling the judge once per document or clause.

  2. **Content-addressed verdict cache** — each verdict is keyed by a SHA-256
     of the payload content *plus* a model/prompt identity string.  Identical
     clause text is judged once, ever, even when it appears in multiple
     documents.  The cache is persisted in ``out/.cache/verdicts.jsonl``.

  3. **Centralised ``judge_error`` handling** — any exception raised by a
     delegate judge is caught here and converted to the appropriate sentinel
     value (``ScopeDecision(basis="judge_error")``, ``"judge_error"`` basis on
     classification, ``"needs_review"`` on deviation).  Each judge no longer
     needs to implement its own error contract.

Interface:

    from playbook_engine.judgment import JudgmentCache, BatchedScopeJudge

    cache = JudgmentCache(cache_dir / "verdicts.jsonl", model_id="stub-v1")
    scope_judge = BatchedScopeJudge(delegate=my_real_judge, cache=cache)
    # ... collect payloads ...
    decision = scope_judge.judge(tree, agreement_type)

The ``judge()`` / ``assess_batch()`` / ``classify_batch()`` signatures on the
wrappers are intentionally identical to the underlying Protocol definitions —
existing pipeline code works without changes.

Security: no raw agreement text is stored in the cache.  Payloads are
serialised as compact JSON before hashing; the cache persists only the
serialised payload hash and the resulting verdict dict, not the clause text.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from playbook_engine.clause_classifier import ClassificationJudge, ClauseClassification
from playbook_engine.clause_tree import ClauseNode, ClauseTree
from playbook_engine.config import AgreementType
from playbook_engine.deviation_classifier import DeviationJudge, DeviationResult, RiskDelta
from playbook_engine.scope_gate import ScopeDecision, ScopeJudge

# ---------------------------------------------------------------------------
# Cache key construction
# ---------------------------------------------------------------------------

_NEUTRAL_ZERO = RiskDelta(direction="neutral", magnitude="none")

# Verdict bases that mean "not actually resolved yet" — a store-backed judge
# miss (issue #64/#182: ``needs_review``) or a delegate exception
# (``judge_error``). Caching these would let this content-addressed cache
# permanently shadow the real answer: once a payload's ``needs_review``/
# ``judge_error`` verdict is cached here, a LATER call for the exact same
# payload — even after e.g. a ``VerdictStore`` used as the delegate has since
# been populated with the real verdict via ``playbook judge-apply`` — hits
# this cache first and never reaches the delegate again to notice. The CLI
# avoids this by forcing ``no_cache=True`` whenever store-backed judges are
# wired (``cli._verdict_store_kwargs``), but that's a convention any other
# caller of ``mine_corpus(..., no_cache=False)`` (the library default) with a
# store-backed delegate could forget. Never caching an unresolved basis in
# the first place makes that class of caller-error impossible.
_UNRESOLVED_BASES = frozenset({"needs_review", "judge_error"})


def _payload_key(payload: Any, model_id: str) -> str:
    """SHA-256 of the JSON-serialised payload plus the model/prompt identity.

    Including *model_id* in the key means that changing the (stubbed) model
    identifier invalidates cached verdicts — satisfying AC-4.
    """
    raw = json.dumps({"payload": payload, "model_id": model_id}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Persistent verdict cache
# ---------------------------------------------------------------------------


class JudgmentCache:
    """Content-addressed, persistent verdict cache.

    Verdicts are keyed by ``hash(payload + model_id)`` and stored in a
    ``.jsonl`` file.  The cache survives across runs.

    All cached values are plain JSON-serialisable dicts (no domain objects).
    """

    def __init__(self, cache_path: Path, *, model_id: str = "default") -> None:
        """Initialise the cache.

        Args:
            cache_path:  Path to the ``.jsonl`` file (created on first write).
            model_id:    Model / prompt identity string.  Changing this value
                         invalidates all previously cached verdicts.
        """
        self._cache_path = cache_path
        self._model_id = model_id
        self._store: dict[str, Any] = {}  # key -> verdict dict
        self._hits = 0
        self._misses = 0
        self._load()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def hit_count(self) -> int:
        """Cache hits since instantiation."""
        return self._hits

    @property
    def miss_count(self) -> int:
        """Cache misses (new judge calls) since instantiation."""
        return self._misses

    @property
    def model_id(self) -> str:
        """Model / prompt identity string embedded in each cache key."""
        return self._model_id

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, payload: Any) -> Any | None:
        """Return the cached verdict dict for *payload*, or ``None`` on miss."""
        key = _payload_key(payload, self._model_id)
        hit = self._store.get(key)
        if hit is not None:
            self._hits += 1
        return hit

    def put(self, payload: Any, verdict: Any) -> None:
        """Store *verdict* for *payload*.  Both must be JSON-serialisable."""
        key = _payload_key(payload, self._model_id)
        self._store[key] = verdict
        self._misses += 1
        self._append(key, verdict)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Read the cache file into memory (best-effort; corrupt lines skipped)."""
        if not self._cache_path.exists():
            return
        try:
            for line in self._cache_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    self._store[entry["key"]] = entry["verdict"]
                except Exception:  # noqa: BLE001
                    pass  # corrupt line — skip; do not crash startup
        except Exception:  # noqa: BLE001
            pass  # unreadable file — start with empty cache

    def _append(self, key: str, verdict: Any) -> None:
        """Append a single entry to the JSONL file (atomic-ish)."""
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        entry = json.dumps({"key": key, "verdict": verdict}, ensure_ascii=False) + "\n"
        # Append mode: safe for sequential use; not concurrent-safe (acceptable per issue scope).
        with self._cache_path.open("a", encoding="utf-8") as fh:
            fh.write(entry)


# ---------------------------------------------------------------------------
# Centralised judge_error sentinel builders
# ---------------------------------------------------------------------------


def _scope_error_decision() -> ScopeDecision:
    return ScopeDecision(
        in_scope=True,
        scope_rationale="Judge error — document retained pending human review.",
        scope_confidence=0.0,
        basis="judge_error",
    )


def _classification_error() -> ClauseClassification:
    return ClauseClassification(taxonomy_id=None, confidence=0.0, basis="judge_error")


def _deviation_error() -> DeviationResult:
    return DeviationResult(deviation="needs_review", risk_delta=_NEUTRAL_ZERO, basis="judge_error")


# ---------------------------------------------------------------------------
# Scope judge wrapper
# ---------------------------------------------------------------------------


@dataclass
class BatchedScopeJudge:
    """Caching wrapper around a ``ScopeJudge`` delegate.

    Implements the ``ScopeJudge`` protocol interface so it can be used as a
    drop-in replacement anywhere a ``ScopeJudge`` is expected.

    ``judge()`` is called once per document (scope decisions are document-level,
    not clause-level), so the "batching" here is specifically the caching layer:
    identical trees produce a cache hit.
    """

    delegate: ScopeJudge
    cache: JudgmentCache

    def judge(self, tree: ClauseTree, agreement_type: AgreementType) -> ScopeDecision:
        """Return a scope decision for *tree*, using the cache when possible.

        Converts any delegate exception into ``basis="judge_error"`` (retain
        the document; flag for review) rather than propagating.
        """
        # Build a compact, stable payload representation for the cache key.
        #
        # TRADE-OFF — headings-only scope key: the key uses up to 20 sorted
        # headings but does NOT include clause body text.  Two documents with
        # identical heading sets but different body text will therefore share a
        # scope verdict.  This is an accepted trade-off: scope decisions depend
        # primarily on the *type* of clauses present (signalled by headings), not
        # their precise wording.  If a future scope judge needs full body text,
        # this payload must be extended (and the cache busted via a model_id bump).
        payload = {
            "stage": "scope",
            "agreement_type_id": agreement_type.id,
            "clause_heads": sorted(n.heading or "" for n in tree.all_nodes() if n.heading)[
                :20
            ],  # top-20 headings for compactness
        }

        cached = self.cache.get(payload)
        if cached is not None:
            return ScopeDecision(**cached)

        try:
            result = self.delegate.judge(tree, agreement_type)
        except Exception:  # noqa: BLE001
            result = _scope_error_decision()

        # Never cache an unresolved verdict (issue #182 "B4") — see
        # _UNRESOLVED_BASES. A delegate exception here always means
        # "judge_error"; a real store-backed delegate's own "not yet judged"
        # signal is instead a raised ScopeNeedsReviewError (caught by
        # scope_gate(), not by this wrapper), so "judge_error" is the only
        # unresolved basis reachable at this call site.
        if result.basis not in _UNRESOLVED_BASES:
            self.cache.put(
                payload,
                {
                    "in_scope": result.in_scope,
                    "scope_rationale": result.scope_rationale,
                    "scope_confidence": result.scope_confidence,
                    "basis": result.basis,
                },
            )
        return result


# ---------------------------------------------------------------------------
# Classification judge wrapper
# ---------------------------------------------------------------------------


@dataclass
class BatchedClassificationJudge:
    """Caching, batched wrapper around a ``ClassificationJudge`` delegate.

    Implements the ``ClassificationJudge`` protocol.

    ``classify_batch()`` deduplicates payloads before calling the delegate
    (cross-document dedup) and caches each result by content hash.
    """

    delegate: ClassificationJudge
    cache: JudgmentCache

    def classify_batch(
        self,
        nodes: list[ClauseNode],
        taxonomy: Any,
        hints: Any = None,
    ) -> list[ClauseClassification]:
        """Classify *nodes*, using the verdict cache to avoid redundant calls.

        Steps:
        1. Build a per-node payload (text + taxonomy labels).
        2. Check the cache; collect uncached nodes.
        3. Call the delegate once with all uncached nodes (true batching).
        4. Store new verdicts in the cache.
        5. Reconstruct results in original order.
        """
        # Build payloads (stable, compact JSON-serialisable dicts).
        tax_labels = sorted(e.id for e in taxonomy.entries) if hasattr(taxonomy, "entries") else []
        payloads: list[dict[str, Any]] = []
        for node in nodes:
            # TRADE-OFF — 500-char text truncation in classification key: two
            # clauses that are identical in their first 500 characters but differ
            # afterwards will hash to the same key and share a cached verdict.
            # This is an accepted trade-off for compactness and performance; real-
            # world clauses rarely diverge only beyond char 500, and the model_id
            # component ensures the collision boundary is explicit and testable.
            # If a clause is known to require full-text discrimination, the
            # truncation limit should be raised here (and model_id bumped to bust
            # the existing cache).
            payloads.append(
                {
                    "stage": "classify",
                    "text": (node.text or "")[:500],
                    "heading": node.heading or "",
                    "taxonomy_ids": tax_labels,
                }
            )

        # Separate cache hits from misses; dedup identical payloads within the batch so
        # that two nodes with the same content produce exactly one delegate call even when
        # they arrive in the same corpus-wide dispatch.
        results: list[ClauseClassification | None] = [None] * len(nodes)

        # unique_key → (canonical_node, canonical_payload) for the delegate batch.
        unique_key_to_canonical: dict[str, tuple[ClauseNode, dict[str, Any]]] = {}
        # unique_key → list of result indices that share this payload.
        key_to_result_indices: dict[str, list[int]] = {}

        for i, (node, payload) in enumerate(zip(nodes, payloads, strict=True)):
            cached = self.cache.get(payload)
            if cached is not None:
                results[i] = ClauseClassification(**cached)
            else:
                key = _payload_key(payload, self.cache.model_id)
                if key not in unique_key_to_canonical:
                    unique_key_to_canonical[key] = (node, payload)
                    key_to_result_indices[key] = []
                key_to_result_indices[key].append(i)

        # Dispatch a single batch for all unique uncached payloads.
        unique_keys = list(unique_key_to_canonical)
        unique_nodes = [unique_key_to_canonical[k][0] for k in unique_keys]
        unique_payloads = [unique_key_to_canonical[k][1] for k in unique_keys]
        # Hints are positional per the ClassificationJudge contract (one per
        # node, same order). Rebuild the hints list to parallel unique_nodes,
        # keeping the hint of each key's canonical (first-seen) node —
        # otherwise dedup would silently misalign hints with the wrong clause.
        unique_hints = (
            [hints[key_to_result_indices[k][0]] for k in unique_keys] if hints is not None else None
        )

        if unique_nodes:
            try:
                new_verdicts = self.delegate.classify_batch(unique_nodes, taxonomy, unique_hints)
            except Exception:  # noqa: BLE001
                new_verdicts = [_classification_error() for _ in unique_nodes]

            for key, verdict, node_payload in zip(
                unique_keys, new_verdicts, unique_payloads, strict=True
            ):
                # Never cache an unresolved verdict (issue #182 "B4") — see
                # _UNRESOLVED_BASES. A store-backed delegate's "no verdict
                # yet" (basis="needs_review") must keep re-checking the store
                # on every call, not get permanently pinned to this miss the
                # first time it's seen.
                if verdict.basis not in _UNRESOLVED_BASES:
                    self.cache.put(
                        node_payload,
                        {
                            "taxonomy_id": verdict.taxonomy_id,
                            "confidence": verdict.confidence,
                            "basis": verdict.basis,
                        },
                    )
                for ri in key_to_result_indices[key]:
                    results[ri] = verdict

        return [r for r in results if r is not None]


# ---------------------------------------------------------------------------
# Deviation judge wrapper
# ---------------------------------------------------------------------------


@dataclass
class BatchedDeviationJudge:
    """Caching, batched wrapper around a ``DeviationJudge`` delegate.

    Implements the ``DeviationJudge`` protocol.

    ``assess_batch()`` deduplicates payloads before calling the delegate and
    caches each result.
    """

    delegate: DeviationJudge
    cache: JudgmentCache

    def assess_batch(
        self,
        items: list[dict[str, str]],
        our_standard: str,
    ) -> list[DeviationResult]:
        """Assess deviation for *items*, using the verdict cache.

        Steps:
        1. Build per-item payloads (hunk + our_standard + stage tag).
        2. Cache hits are returned directly.
        3. Uncached items are dispatched as a single batch to the delegate.
        4. New verdicts are stored in the cache.
        5. Results are returned in original order.
        """
        payloads: list[dict[str, Any]] = []
        for item in items:
            payloads.append(
                {
                    "stage": "deviation",
                    "hunk": item.get("hunk", ""),
                    "our_standard": our_standard[:500],
                }
            )

        results: list[DeviationResult | None] = [None] * len(items)

        def _from_cache(cached: dict[str, Any]) -> DeviationResult:
            risk_raw = cached["risk_delta"]
            return DeviationResult(
                deviation=cached["deviation"],
                risk_delta=RiskDelta(
                    direction=risk_raw["direction"],
                    magnitude=risk_raw["magnitude"],
                ),
                basis=cached["basis"],
                rationale=cached.get("rationale", ""),
                confidence=cached.get("confidence"),
            )

        # Dedup identical payloads within the batch so that two items with the same
        # hunk + our_standard produce exactly one delegate call even in a corpus-wide dispatch.
        unique_key_to_canonical: dict[str, tuple[dict[str, str], dict[str, Any]]] = {}
        key_to_result_indices: dict[str, list[int]] = {}

        for i, (item, payload) in enumerate(zip(items, payloads, strict=True)):
            cached = self.cache.get(payload)
            if cached is not None:
                results[i] = _from_cache(cached)
            else:
                key = _payload_key(payload, self.cache.model_id)
                if key not in unique_key_to_canonical:
                    unique_key_to_canonical[key] = (item, payload)
                    key_to_result_indices[key] = []
                key_to_result_indices[key].append(i)

        unique_keys = list(unique_key_to_canonical)
        unique_items = [unique_key_to_canonical[k][0] for k in unique_keys]
        unique_payloads = [unique_key_to_canonical[k][1] for k in unique_keys]

        if unique_items:
            try:
                new_verdicts = self.delegate.assess_batch(unique_items, our_standard)
            except Exception:  # noqa: BLE001
                new_verdicts = [_deviation_error() for _ in unique_items]

            for key, verdict, item_payload in zip(
                unique_keys, new_verdicts, unique_payloads, strict=True
            ):
                # Never cache an unresolved verdict (issue #182 "B4") — see
                # _UNRESOLVED_BASES.
                if verdict.basis not in _UNRESOLVED_BASES:
                    self.cache.put(
                        item_payload,
                        {
                            "deviation": verdict.deviation,
                            "risk_delta": {
                                "direction": verdict.risk_delta.direction,
                                "magnitude": verdict.risk_delta.magnitude,
                            },
                            "basis": verdict.basis,
                            "rationale": verdict.rationale,
                            "confidence": verdict.confidence,
                        },
                    )
                for ri in key_to_result_indices[key]:
                    results[ri] = verdict

        return [r for r in results if r is not None]
