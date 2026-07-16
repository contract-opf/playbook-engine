"""Agent-as-judge bridge core — issue #64.

Provides a store-backed "agent-as-judge" layer that lets an external caller
supply **real verdicts** into the L1–L4 pipeline through the existing judge
dependency-injection seam.

On a payload it has seen before (store hit), replays the stored verdict as the
correct dataclass.  On a new payload (store miss), records the **full**
untruncated payload to a pending queue and returns the engine's needs-review
sentinel.

Components:

- ``VerdictStore`` — persistent JSONL keyed by a stable SHA-256 content hash
  of the full payload.  Mirrors the style of ``JudgmentCache`` in
  ``judgment.py`` but in its own namespace and without text truncation.
  Default file: ``<out>/judge/verdicts.jsonl``.

- ``PendingQueue`` — appends unique full payloads (deduplicated by key) to
  ``<out>/judge/pending.jsonl``.  Each record carries the payload key, the
  judge kind, and the full payload dict.

- ``StoreBackedClassificationJudge`` — implements ``ClassificationJudge``.
- ``StoreBackedDeviationJudge``      — implements ``DeviationJudge``.
- ``StoreBackedProvenanceJudge``     — implements ``ProvenanceJudge``.
- ``StoreBackedScopeJudge``          — implements ``ScopeJudge``.

These are drop-in replacements for the judge parameters of
``mine_corpus(scope_judge=…, classification_judge=…, deviation_judge=…,
provenance_judge=…)``.

Note on ``StoreBackedScopeJudge`` (issue #87): unlike the other three,
``ScopeJudge.judge()`` may only return ``ScopeDecision(basis="judge")`` —
``scope_gate()`` raises ``ValueError`` on any other basis — so a store miss
cannot be expressed as a ``basis="needs_review"`` return value the way the
other judges do it. Instead it raises ``ScopeNeedsReviewError`` after queuing
the payload; ``scope_gate()`` catches that and converts it into a retained,
zero-confidence ``basis="judge_error"`` decision, never the stub default's
blind ``in_scope=True`` at confidence 0.5.

Security: full clause text IS stored in the pending queue (by design — the
external caller needs it to render the verdict).  The store itself stores the
verdict dict plus key; it does NOT re-store the payload.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from playbook_engine.clause_classifier import ClauseClassification
from playbook_engine.clause_tree import ClauseNode, ClauseTree
from playbook_engine.config import AgreementType
from playbook_engine.deviation_classifier import DeviationResult, RiskDelta
from playbook_engine.provenance_detector import ProvenanceResult
from playbook_engine.scope_gate import ScopeDecision

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level sentinels
# ---------------------------------------------------------------------------

_NEUTRAL_ZERO = RiskDelta(direction="neutral", magnitude="none")


# ---------------------------------------------------------------------------
# Key construction — full payload, no truncation
# ---------------------------------------------------------------------------


def _payload_key(payload: Any) -> str:
    """SHA-256 of the JSON-serialised payload.

    Unlike ``judgment._payload_key``, this function does NOT include a
    ``model_id`` component — the store-backed judges are keyed purely by
    content, and the external verdict-supplier is responsible for versioning.
    Also unlike the judgment cache, text is NOT truncated: the full payload
    is hashed to prevent false collisions across clauses that share a long
    prefix but differ later.
    """
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()


# ---------------------------------------------------------------------------
# VerdictStore
# ---------------------------------------------------------------------------


class VerdictStore:
    """Persistent JSONL store: content hash → verdict dict.

    Each record: ``{"key": "<sha256>", "verdict": {…}}``.

    ``get(payload) -> dict | None``  — return stored verdict dict or None.
    ``put(payload, verdict)``        — append new verdict; update in-memory.

    Load-on-init: reads the JSONL file into memory on construction.
    Corrupt lines are silently skipped (same contract as ``JudgmentCache``).
    """

    def __init__(self, store_path: Path) -> None:
        self._store_path = store_path
        self._store: dict[str, Any] = {}  # key -> verdict dict
        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, payload: Any) -> dict[str, Any] | None:
        """Return the stored verdict dict for *payload*, or ``None`` on miss."""
        key = _payload_key(payload)
        return self._store.get(key)

    def put(self, payload: Any, verdict: dict[str, Any]) -> None:
        """Store *verdict* for *payload* (JSON-serialisable dicts required)."""
        key = _payload_key(payload)
        self._store[key] = verdict
        self._append(key, verdict)

    def put_by_key(self, key: str, verdict: dict[str, Any]) -> None:
        """Store *verdict* directly by its pre-computed *key*.

        Used by ``playbook judge-apply`` to import verdicts whose keys were
        computed by the producer (e.g. from a ``pending.jsonl`` export) without
        re-hashing the original payload.

        Args:
            key:     SHA-256 hex string (as produced by ``_payload_key``).
            verdict: JSON-serialisable verdict dict.
        """
        self._store[key] = verdict
        self._append(key, verdict)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Read the store file into memory (best-effort; corrupt lines skipped)."""
        if not self._store_path.exists():
            return
        try:
            for line in self._store_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    self._store[entry["key"]] = entry["verdict"]
                except Exception:  # noqa: BLE001
                    pass  # corrupt line — skip; do not crash startup
        except Exception:  # noqa: BLE001
            pass  # unreadable file — start with empty store

    def _append(self, key: str, verdict: dict[str, Any]) -> None:
        """Append a single entry to the JSONL file."""
        self._store_path.parent.mkdir(parents=True, exist_ok=True)
        entry = json.dumps({"key": key, "verdict": verdict}, ensure_ascii=False) + "\n"
        with self._store_path.open("a", encoding="utf-8") as fh:
            fh.write(entry)


# ---------------------------------------------------------------------------
# PendingQueue
# ---------------------------------------------------------------------------


class PendingQueue:
    """Append-only queue of pending payloads awaiting external verdict.

    Each record: ``{"key": "<sha256>", "kind": "classify"|"deviation"|"provenance",
    "payload": {…}}``.

    Deduplication: payloads with the same key are recorded at most once,
    even across multiple ``add()`` calls on the same instance.  This is the
    within-instance dedup (within-batch + cross-batch for the same object);
    persistence does not deduplicate across runs (the external caller is
    responsible for that).
    """

    def __init__(self, queue_path: Path) -> None:
        self._queue_path = queue_path
        self._seen_keys: set[str] = set()

    def add(self, key: str, kind: str, payload: Any) -> bool:
        """Append *payload* to the queue if *key* has not been seen before.

        Args:
            key:     Content hash from ``_payload_key(payload)``.
            kind:    One of ``"classify"``, ``"deviation"``, ``"provenance"``.
            payload: The full, untruncated judge payload dict.

        Returns:
            ``True`` if a new entry was written; ``False`` if *key* was already
            seen (deduplicated).
        """
        if key in self._seen_keys:
            return False
        self._seen_keys.add(key)
        self._append(key, kind, payload)
        return True

    def _append(self, key: str, kind: str, payload: Any) -> None:
        """Write a single record to the JSONL file."""
        self._queue_path.parent.mkdir(parents=True, exist_ok=True)
        record = (
            json.dumps({"key": key, "kind": kind, "payload": payload}, ensure_ascii=False) + "\n"
        )
        with self._queue_path.open("a", encoding="utf-8") as fh:
            fh.write(record)


# ---------------------------------------------------------------------------
# Needs-review sentinels
# ---------------------------------------------------------------------------


def _classification_needs_review() -> ClauseClassification:
    """Sentinel returned when no stored verdict is available for a classify payload."""
    return ClauseClassification(taxonomy_id=None, confidence=0.0, basis="needs_review")


def _deviation_needs_review() -> DeviationResult:
    """Sentinel returned when no stored verdict is available for a deviation payload."""
    return DeviationResult(
        deviation="needs_review",
        risk_delta=_NEUTRAL_ZERO,
        basis="needs_review",
        rationale="No stored verdict — clause queued for human review.",
        confidence=None,
    )


def _provenance_needs_review() -> ProvenanceResult:
    """Sentinel returned when no stored verdict is available for a provenance payload.

    Returns a low-confidence result so the deterministic detector default is
    not silently trusted.
    """
    return ProvenanceResult(
        provenance="counterparty_paper",
        confidence=0.0,
        basis="needs_review",
    )


# ---------------------------------------------------------------------------
# StoreBackedClassificationJudge
# ---------------------------------------------------------------------------


@dataclass
class StoreBackedClassificationJudge:
    """``ClassificationJudge`` that replays stored verdicts or queues new payloads.

    Implements ``ClassificationJudge.classify_batch`` and is a drop-in
    replacement for the ``classification_judge`` parameter of ``mine_corpus``.

    On a store hit: returns ``ClauseClassification(basis="judge")`` reconstructed
    from the stored verdict dict.

    On a store miss: appends the full clause payload (text + heading +
    taxonomy ids) to the pending queue and returns
    ``ClauseClassification(basis="needs_review")``.

    Duplicate payloads within a single ``classify_batch`` call produce exactly
    one pending-queue entry (deduplicated by key).
    """

    store: VerdictStore
    pending: PendingQueue
    _seen_keys: set[str] = field(default_factory=set, init=False, repr=False)

    def classify_batch(
        self,
        nodes: list[ClauseNode],
        taxonomy: Any,
        hints: Any = None,
    ) -> list[ClauseClassification]:
        """Classify *nodes* from the store or queue them for external review.

        Args:
            nodes:    Clause nodes to classify.
            taxonomy: The full taxonomy (used to extract taxonomy ids for payload).
            hints:    Ignored (pass-through for protocol compatibility).

        Returns:
            One ``ClauseClassification`` per node in the same order.
        """
        tax_labels = sorted(e.id for e in taxonomy.entries) if hasattr(taxonomy, "entries") else []

        results: list[ClauseClassification] = []
        for node in nodes:
            # Full text — NOT truncated (contrast with judgment.py text[:500]).
            payload = {
                "stage": "classify",
                "text": node.text or "",
                "heading": node.heading or "",
                "taxonomy_ids": tax_labels,
            }
            key = _payload_key(payload)

            cached = self.store.get(payload)
            if cached is not None:
                try:
                    results.append(
                        ClauseClassification(
                            taxonomy_id=cached.get("taxonomy_id"),
                            confidence=cached.get("confidence", 0.0),
                            basis=cached.get("basis", "judge"),
                        )
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    # Isolate one malformed stored verdict (issue #182): must
                    # not raise out of classify_batch and get the whole
                    # taxonomy batch quarantined as basis='judge_error' by the
                    # caller's blanket except (see StoreBackedDeviationJudge
                    # for the same pattern).
                    _log.warning(
                        "StoreBackedClassificationJudge: malformed stored "
                        "verdict for key %s (%s); re-queuing for review",
                        key,
                        exc,
                    )
                    self.pending.add(key, "classify", payload)
                    results.append(_classification_needs_review())
            else:
                # Queue for external verdict (deduplicated by key).
                self.pending.add(key, "classify", payload)
                results.append(_classification_needs_review())

        return results


# ---------------------------------------------------------------------------
# StoreBackedDeviationJudge
# ---------------------------------------------------------------------------


@dataclass
class StoreBackedDeviationJudge:
    """``DeviationJudge`` that replays stored verdicts or queues new payloads.

    Implements ``DeviationJudge.assess_batch`` and is a drop-in replacement for
    the ``deviation_judge`` parameter of ``mine_corpus``.

    On a store hit: returns ``DeviationResult(basis="judge")`` reconstructed
    from the stored verdict dict.

    On a store miss: appends the full deviation payload (hunk + our_standard
    plus traceability context — taxonomy_id, clause_path, document_id when
    present) to the pending queue and returns
    ``DeviationResult(basis="needs_review")``.

    Duplicate payloads within a single ``assess_batch`` call produce exactly one
    pending-queue entry.

    Content-hash key vs. stored context (issue #109): the cache/dedup key is
    derived from ``stage`` + ``hunk`` + ``our_standard`` only. Traceability
    context (``taxonomy_id``, ``clause_path``, ``document_id``) is recorded
    in the pending payload for human/tooling review but deliberately excluded
    from the hash — those fields are per-clause/per-document metadata, not
    judgment content, and folding them in would defeat cross-document dedup
    of an identical hunk/standard pair (e.g. the same boilerplate clause
    appearing verbatim in two different agreements would otherwise queue as
    two distinct pending items and never share a verdict).
    """

    store: VerdictStore
    pending: PendingQueue

    def assess_batch(
        self,
        items: list[dict[str, str]],
        our_standard: str,
    ) -> list[DeviationResult]:
        """Assess deviation for *items* from the store or queue for external review.

        Args:
            items:        Hunk payload dicts (each must have at minimum a
                         ``"hunk"`` key; ``"taxonomy_id"``, ``"clause_path"``,
                         and ``"document_id"`` are optional traceability
                         context — see class docstring).
            our_standard: Canonical text from the playbook standard for this clause type.

        Returns:
            One ``DeviationResult`` per item in the same order.
        """
        results: list[DeviationResult] = []
        for item in items:
            # Full hunk + full our_standard — NOT truncated (contrast with
            # judgment.py). This is the content-hash payload only: it must NOT
            # include taxonomy_id/clause_path/document_id (see class docstring).
            hash_payload = {
                "stage": "deviation",
                "hunk": item.get("hunk", ""),
                "our_standard": our_standard,
            }
            key = _payload_key(hash_payload)

            cached = self.store.get(hash_payload)
            if cached is not None:
                # Reconstruct per-item defensively (issue #182): a single
                # malformed stored verdict (e.g. a RiskDelta invariant
                # violation like direction='neutral'+magnitude='minor', or a
                # missing key) must not raise out of assess_batch and get the
                # WHOLE taxonomy batch quarantined as basis='judge_error' by
                # the caller's blanket except. Isolate the bad verdict: treat
                # it as a miss (re-queue for review) so only that clause is
                # affected, and the rest of the batch replays normally.
                try:
                    risk_raw = cached["risk_delta"]
                    results.append(
                        DeviationResult(
                            deviation=cached["deviation"],
                            risk_delta=RiskDelta(
                                direction=risk_raw["direction"],
                                magnitude=risk_raw["magnitude"],
                            ),
                            basis=cached.get("basis", "judge"),
                            rationale=cached.get("rationale", ""),
                            confidence=cached.get("confidence"),
                        )
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    _log.warning(
                        "StoreBackedDeviationJudge: malformed stored verdict "
                        "for key %s (%s); re-queuing for review",
                        key,
                        exc,
                    )
                    self.pending.add(key, "deviation", {**hash_payload})
                    results.append(_deviation_needs_review())
            else:
                # Full payload recorded to the pending queue carries the
                # traceability context alongside the hashed content — the key
                # above stays content-only so cross-document dedup holds.
                full_payload = {
                    **hash_payload,
                    "taxonomy_id": item.get("taxonomy_id", ""),
                    "clause_path": item.get("clause_path", ""),
                    "document_id": item.get("document_id", ""),
                }
                self.pending.add(key, "deviation", full_payload)
                results.append(_deviation_needs_review())

        return results


# ---------------------------------------------------------------------------
# StoreBackedProvenanceJudge
# ---------------------------------------------------------------------------


@dataclass
class StoreBackedProvenanceJudge:
    """``ProvenanceJudge`` that replays stored verdicts or queues new payloads.

    Implements ``ProvenanceJudge.judge`` and is a drop-in replacement for the
    ``provenance_judge`` parameter of ``mine_corpus``.

    On a store hit: returns ``ProvenanceResult(basis="llm")`` reconstructed from
    the stored verdict dict.  (The store may hold any ``_BASIS_VALUES``-valid
    basis; "llm" is the canonical basis for a judge-supplied result per the
    ``ProvenanceJudge`` protocol contract.)

    On a store miss: appends the full provenance payload (preamble + letterhead +
    agreement_type + candidate aliases) to the pending queue and returns a
    low-confidence ``ProvenanceResult(basis="needs_review")`` so the
    deterministic detector default is not silently trusted.
    """

    store: VerdictStore
    pending: PendingQueue

    def judge(
        self,
        preamble: str,
        letterhead: str,
        agreement_type: str,
    ) -> ProvenanceResult:
        """Return a provenance determination from the store or queue for review.

        Args:
            preamble:       First few lines of document body (recital block).
            letterhead:     Document title / heading block.
            agreement_type: Human-readable agreement type label.

        Returns:
            ``ProvenanceResult`` with ``basis="llm"`` on a store hit, or
            ``ProvenanceResult(basis="needs_review")`` on a miss.
        """
        payload = {
            "stage": "provenance",
            "preamble": preamble,
            "letterhead": letterhead,
            "agreement_type": agreement_type,
        }
        key = _payload_key(payload)

        cached = self.store.get(payload)
        if cached is not None:
            try:
                return ProvenanceResult(
                    provenance=cached["provenance"],
                    confidence=cached.get("confidence", 0.0),
                    basis=cached.get("basis", "llm"),
                )
            except (KeyError, TypeError, ValueError) as exc:
                # Isolate one malformed stored verdict (issue #182) — same
                # pattern as StoreBackedDeviationJudge/ClassificationJudge.
                _log.warning(
                    "StoreBackedProvenanceJudge: malformed stored verdict "
                    "for key %s (%s); re-queuing for review",
                    key,
                    exc,
                )
                self.pending.add(key, "provenance", payload)
                return _provenance_needs_review()
        self.pending.add(key, "provenance", payload)
        return _provenance_needs_review()


# ---------------------------------------------------------------------------
# StoreBackedScopeJudge
# ---------------------------------------------------------------------------


class ScopeNeedsReviewError(Exception):
    """Raised by ``StoreBackedScopeJudge.judge()`` on a store miss.

    ``ScopeJudge.judge()`` is contractually restricted to returning
    ``ScopeDecision(basis="judge")`` — ``scope_gate()`` raises ``ValueError``
    on any other basis returned from a successful call — so "no verdict yet"
    cannot be expressed as a sentinel return value the way the classify/
    deviation/provenance store-backed judges use ``basis="needs_review"``.

    Raising instead lets ``scope_gate()``'s existing exception handling do
    the right thing: it converts this into ``ScopeDecision(basis=
    "judge_error", in_scope=True, scope_confidence=0.0)`` — the document is
    retained and flagged for review, never auto-accepted at the stub
    default's confidence 0.5.
    """


@dataclass
class StoreBackedScopeJudge:
    """``ScopeJudge`` that replays stored verdicts or queues new payloads.

    Implements ``ScopeJudge.judge`` and is a drop-in replacement for the
    ``scope_judge`` parameter of ``mine_corpus``. Closes the issue #87 hole
    where every CLI path fell back to ``_AllInScopeJudge`` (every document
    auto-accepted as in-scope at confidence 0.5, regardless of content).

    On a store hit: returns ``ScopeDecision(basis="judge")`` reconstructed
    from the stored verdict dict — including out-of-scope verdicts, which
    the stub could never produce.

    On a store miss: appends the full scope payload (agreement type id and
    every clause heading in the document — not capped, unlike the headings-
    only cache key in ``judgment.BatchedScopeJudge``) to the pending queue
    and raises ``ScopeNeedsReviewError``. See that error's docstring for why
    raising (rather than returning a sentinel) is required here.

    Duplicate payloads across calls produce exactly one pending-queue entry.
    """

    store: VerdictStore
    pending: PendingQueue

    def judge(
        self,
        tree: ClauseTree,
        agreement_type: AgreementType,
    ) -> ScopeDecision:
        """Return a scope decision from the store, or queue it for review.

        Args:
            tree:            Segmented clause tree of the document to evaluate.
            agreement_type:  Target agreement type from the engine config.

        Returns:
            ``ScopeDecision`` with ``basis="judge"`` on a store hit.

        Raises:
            ScopeNeedsReviewError: on a store miss, after the payload has
                been queued to the pending queue.
        """
        payload = {
            "stage": "scope",
            "agreement_type_id": agreement_type.id,
            "document_id": tree.document_id,
            "clause_heads": [node.heading or "" for node in tree.all_nodes()],
        }
        key = _payload_key(payload)

        cached = self.store.get(payload)
        if cached is not None:
            try:
                return ScopeDecision(
                    in_scope=cached["in_scope"],
                    scope_rationale=cached.get("scope_rationale")
                    or "Replayed from stored verdict.",
                    scope_confidence=cached.get("scope_confidence", 0.0),
                    basis="judge",
                )
            except (KeyError, TypeError, ValueError) as exc:
                # Isolate one malformed stored verdict (issue #182) — same
                # pattern as the other three store-backed judges. Scope has
                # no needs_review sentinel to return (see class docstring),
                # so re-queue and raise exactly as the miss path below does.
                _log.warning(
                    "StoreBackedScopeJudge: malformed stored verdict for "
                    "document %s, key %s (%s); re-queuing for review",
                    tree.document_id,
                    key,
                    exc,
                )
                self.pending.add(key, "scope", payload)
                raise ScopeNeedsReviewError(
                    f"Malformed stored scope verdict for document {tree.document_id!r} — "
                    "re-queued for external review."
                ) from exc

        self.pending.add(key, "scope", payload)
        raise ScopeNeedsReviewError(
            f"No stored scope verdict for document {tree.document_id!r} — "
            "queued for external review."
        )
