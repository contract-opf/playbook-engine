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
  Default file: ``<out>/judge/verdicts.jsonl``.  Each record also carries the
  **rubric stamp** the verdict was produced under (optional on read — absent
  on records banked before rubric versioning existed).

- ``PendingQueue`` — appends unique full payloads (deduplicated by key) to
  ``<out>/judge/pending.jsonl``.  Each record carries the payload key, the
  judge kind, the full payload dict, and the rubric version in force when
  the item was queued.

- ``StoreBackedClassificationJudge`` — implements ``ClassificationJudge``.
- ``StoreBackedDeviationJudge``      — implements ``DeviationJudge``.
- ``StoreBackedProvenanceJudge``     — implements ``ProvenanceJudge``.
- ``StoreBackedScopeJudge``          — implements ``ScopeJudge``.

These are drop-in replacements for the judge parameters of
``mine_corpus(scope_judge=…, classification_judge=…, deviation_judge=…,
provenance_judge=…)``.

Rubric versioning (see :mod:`playbook_engine.rubric`): because the store key
is purely content-derived, a change to the *judging criteria* — the taxonomy,
the deviation vocabulary, the prose rubric in the ``playbook-from-corpus``
skill — would otherwise replay every banked verdict forever, since the clause
text never moved.  Each store hit is therefore checked against the rubric
currently in force: ``current`` replays, ``stale`` re-queues, and ``legacy``
(unstamped) replays but is counted and reported.  ``RubricPolicy`` carries
both the policy knobs and the run tally the CLI reports from.

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
from playbook_engine.rubric import (
    RubricPolicy,
    RubricStamp,
    classifier_eligible_ids,
    rubric_version,
)
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
    ``model_id`` component, and it deliberately does NOT include the rubric
    version either: the key stays purely content-derived so that cross-
    document dedup of identical clause text keeps working and so that a
    rubric bump does not silently orphan the entire banked verdict store
    (a key that never matches again is indistinguishable from an empty
    store). Rubric identity is carried *beside* the verdict instead — see
    :class:`~playbook_engine.rubric.RubricStamp` and
    :class:`~playbook_engine.rubric.RubricPolicy` — so a bump is a counted,
    reported, reversible event rather than a vanishing act.

    Also unlike the judgment cache, text is NOT truncated: the full payload
    is hashed to prevent false collisions across clauses that share a long
    prefix but differ later.
    """
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()


# ---------------------------------------------------------------------------
# VerdictStore
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StoredVerdict:
    """One record read back out of a :class:`VerdictStore`.

    ``rubric`` is ``None`` for every record written before rubric versioning
    existed (the "legacy" state) — see :mod:`playbook_engine.rubric`.
    """

    verdict: dict[str, Any]
    rubric: RubricStamp | None = None

    @property
    def rubric_version(self) -> str | None:
        return self.rubric.version if self.rubric else None

    @property
    def rubric_kind(self) -> str | None:
        return self.rubric.kind if self.rubric else None


class VerdictStore:
    """Persistent JSONL store: content hash → verdict dict (+ rubric stamp).

    Each record: ``{"key": "<sha256>", "verdict": {…}, "rubric": {"kind": …,
    "version": …}}``. The ``rubric`` member is optional on read — records
    written before rubric versioning carry no stamp and load as
    ``StoredVerdict(rubric=None)``, so an existing store keeps working
    untouched and no banked judgment is discarded on upgrade.

    ``get(payload) -> dict | None``       — stored verdict dict, or None.
    ``get_record(payload)``               — verdict + rubric stamp, or None.
    ``put(payload, verdict, rubric=…)``   — append; update in-memory.

    Load-on-init: reads the JSONL file into memory on construction.
    Corrupt lines are silently skipped (same contract as ``JudgmentCache``).
    Later lines for the same key win, which is what makes ``restamp`` an
    append rather than a rewrite: the original record stays on disk as an
    audit trail of what the verdict was banked under.
    """

    def __init__(self, store_path: Path) -> None:
        self._store_path = store_path
        self._store: dict[str, StoredVerdict] = {}  # key -> record
        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, payload: Any) -> dict[str, Any] | None:
        """Return the stored verdict dict for *payload*, or ``None`` on miss."""
        record = self._store.get(_payload_key(payload))
        return record.verdict if record is not None else None

    def get_record(self, payload: Any) -> StoredVerdict | None:
        """Return the full stored record (verdict + rubric stamp), or ``None``."""
        return self._store.get(_payload_key(payload))

    def put(
        self, payload: Any, verdict: dict[str, Any], *, rubric: RubricStamp | None = None
    ) -> None:
        """Store *verdict* for *payload* (JSON-serialisable dicts required)."""
        self.put_by_key(_payload_key(payload), verdict, rubric=rubric)

    def get_by_key(self, key: str) -> dict[str, Any] | None:
        """Return the stored verdict dict for a pre-computed *key*, or ``None``.

        Counterpart of ``put_by_key`` — used by ``playbook judge`` to detect
        pending items that are re-queues of stored-but-malformed verdicts.
        """
        record = self._store.get(key)
        return record.verdict if record is not None else None

    def get_record_by_key(self, key: str) -> StoredVerdict | None:
        """Return the full stored record for a pre-computed *key*, or ``None``."""
        return self._store.get(key)

    def put_by_key(
        self, key: str, verdict: dict[str, Any], *, rubric: RubricStamp | None = None
    ) -> None:
        """Store *verdict* directly by its pre-computed *key*.

        Used by ``playbook judge-apply`` to import verdicts whose keys were
        computed by the producer (e.g. from a ``pending.jsonl`` export) without
        re-hashing the original payload.

        Args:
            key:     SHA-256 hex string (as produced by ``_payload_key``).
            verdict: JSON-serialisable verdict dict.
            rubric:  Rubric the verdict was produced under. ``None`` records
                     the verdict unstamped (legacy) — reported on every
                     subsequent judge run until migrated.
        """
        self._store[key] = StoredVerdict(verdict=verdict, rubric=rubric)
        self._append(key, verdict, rubric)

    def restamp(self, key: str, rubric: RubricStamp) -> bool:
        """Re-stamp an existing verdict with *rubric*, keeping the verdict.

        The migration primitive behind ``playbook judge-migrate``: it banks
        an operator's explicit decision that a stored judgment still stands
        under the named rubric. Implemented as an append (last line wins on
        load), so the prior stamp — or its absence — remains on disk.

        Returns:
            ``True`` if the key existed and was re-stamped, ``False`` otherwise.
        """
        record = self._store.get(key)
        if record is None:
            return False
        self.put_by_key(key, record.verdict, rubric=rubric)
        return True

    def records(self) -> list[tuple[str, StoredVerdict]]:
        """Return ``(key, record)`` for every stored verdict, key-sorted."""
        return sorted(self._store.items())

    def __len__(self) -> int:
        return len(self._store)

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
                    self._store[entry["key"]] = StoredVerdict(
                        verdict=entry["verdict"],
                        rubric=RubricStamp.from_dict(entry.get("rubric")),
                    )
                except Exception:  # noqa: BLE001
                    pass  # corrupt line — skip; do not crash startup
        except Exception:  # noqa: BLE001
            pass  # unreadable file — start with empty store

    def _append(self, key: str, verdict: dict[str, Any], rubric: RubricStamp | None) -> None:
        """Append a single entry to the JSONL file."""
        self._store_path.parent.mkdir(parents=True, exist_ok=True)
        record: dict[str, Any] = {"key": key, "verdict": verdict}
        if rubric is not None:
            record["rubric"] = rubric.to_dict()
        entry = json.dumps(record, ensure_ascii=False) + "\n"
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

    def add(self, key: str, kind: str, payload: Any, rubric_version: str | None = None) -> bool:
        """Append *payload* to the queue if *key* has not been seen before.

        Args:
            key:            Content hash from ``_payload_key(payload)``.
            kind:           One of ``"classify"``, ``"deviation"``,
                            ``"provenance"``, ``"scope"``, ``"segment"``.
            payload:        The full, untruncated judge payload dict.
            rubric_version: Rubric in force when this item was queued. Written
                            to the record so ``playbook judge-apply`` can stamp
                            the returned verdict with the rubric the question
                            was actually asked under — without needing to
                            re-derive it (and without needing the config at
                            apply time).

        Returns:
            ``True`` if a new entry was written; ``False`` if *key* was already
            seen (deduplicated).
        """
        if key in self._seen_keys:
            return False
        self._seen_keys.add(key)
        self._append(key, kind, payload, rubric_version)
        return True

    def _append(self, key: str, kind: str, payload: Any, rubric_version: str | None = None) -> None:
        """Write a single record to the JSONL file."""
        self._queue_path.parent.mkdir(parents=True, exist_ok=True)
        entry: dict[str, Any] = {"key": key, "kind": kind, "payload": payload}
        if rubric_version is not None:
            entry["rubric_version"] = rubric_version
        record = json.dumps(entry, ensure_ascii=False) + "\n"
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
# Apply-time verdict validation (used by `playbook judge-apply`)
# ---------------------------------------------------------------------------

#: Basis values that mean "no real judgment happened". A verdict *file*
#: (producer-supplied) must never carry one: on replay it either loops the
#: item forever or, worse, replays as a permanently-unjudged result while the
#: pending queue looks drained. The store-backed judges set these themselves.
_UNRESOLVED_VERDICT_BASES = frozenset({"needs_review", "judge_error", "stub"})

#: The only basis values a *producer-supplied* deviation verdict may carry.
#: Mirrors the whitelist ``deviation_classifier.assess_deviations`` enforces on
#: replay (``"judge"``, ``"judge_error"``, ``"needs_review"``), minus the two
#: engine-internal bases already rejected above — so a plausible-but-nonstandard
#: value like "reworded_equivalent" or "deterministic" (both valid
#: ``_BASIS_VALUES``) is caught here instead of aborting the next mine run.
_DEVIATION_REPLAYABLE_BASES = frozenset({"judge"})

#: The only basis values a *producer-supplied* classify verdict may carry.
#: Mirrors the whitelist ``clause_classifier.classify_tree`` enforces on
#: replay (``"judge"``, ``"judge_error"``, ``"needs_review"``, ``"unclassified"``),
#: minus the two engine-internal bases already rejected above.
_CLASSIFY_REPLAYABLE_BASES = frozenset({"judge", "unclassified"})


def validate_verdict(kind: str, verdict: dict[str, Any]) -> None:
    """Validate a producer-supplied *verdict* for *kind* at apply time.

    Reconstructs the exact dataclass the store-backed judge would build on
    replay, so any verdict accepted here is guaranteed to replay instead of
    silently re-queueing (the issue #182 malformed-verdict loop). Raises
    ``ValueError`` with an actionable message on the first problem.

    Args:
        kind:    Pending-item kind: ``classify`` / ``deviation`` /
                 ``provenance`` / ``scope``.
        verdict: The verdict dict from the producer's JSONL line.
    """
    basis = verdict.get("basis")
    if basis in _UNRESOLVED_VERDICT_BASES:
        raise ValueError(
            f"basis {basis!r} is engine-internal (means 'not judged'); a "
            "supplied verdict must carry a real basis — use 'judge' "
            "('llm' for provenance)"
        )
    if kind == "classify":
        classify_basis = verdict.get("basis", "judge")
        if classify_basis not in _CLASSIFY_REPLAYABLE_BASES:
            raise ValueError(
                f"basis {classify_basis!r} is not replayable for a classify verdict; "
                f"a supplied verdict must carry basis in "
                f"{sorted(_CLASSIFY_REPLAYABLE_BASES)!r} (clause_classifier.classify_tree "
                "rejects anything else on replay)"
            )
        ClauseClassification(
            taxonomy_id=verdict.get("taxonomy_id"),
            confidence=verdict.get("confidence", 0.0),
            basis=classify_basis,
        )
    elif kind == "deviation":
        if verdict.get("deviation") == "needs_review":
            raise ValueError(
                "deviation 'needs_review' is engine-internal; judge the hunk "
                "as none / reworded_equivalent / substantive (flag doubts via "
                "'needs_review': true, keeping a real deviation value)"
            )
        if "risk_delta" not in verdict or not isinstance(verdict["risk_delta"], dict):
            raise ValueError(
                "missing 'risk_delta' object — always required; use "
                '{"direction": "neutral", "magnitude": "none"} for '
                "none/reworded_equivalent deviations"
            )
        deviation_basis = verdict.get("basis", "judge")
        if deviation_basis not in _DEVIATION_REPLAYABLE_BASES:
            raise ValueError(
                f"basis {deviation_basis!r} is not replayable for a deviation verdict; "
                f"a supplied verdict must carry basis in "
                f"{sorted(_DEVIATION_REPLAYABLE_BASES)!r} "
                "(deviation_classifier.assess_deviations rejects anything else on replay)"
            )
        risk_raw = verdict["risk_delta"]
        DeviationResult(
            deviation=verdict.get("deviation", ""),
            risk_delta=RiskDelta(
                direction=risk_raw.get("direction", ""),
                magnitude=risk_raw.get("magnitude", ""),
            ),
            basis=deviation_basis,
            rationale=verdict.get("rationale", ""),
            confidence=verdict.get("confidence"),
        )
    elif kind == "provenance":
        if "provenance" not in verdict:
            raise ValueError("missing 'provenance' field")
        ProvenanceResult(
            provenance=verdict["provenance"],
            confidence=verdict.get("confidence", 0.0),
            basis=verdict.get("basis", "llm"),
        )
    elif kind == "scope":
        if not isinstance(verdict.get("in_scope"), bool):
            raise ValueError("'in_scope' must be a JSON boolean")
        ScopeDecision(
            in_scope=verdict["in_scope"],
            scope_rationale=verdict.get("scope_rationale") or "Replayed from stored verdict.",
            scope_confidence=verdict.get("scope_confidence", 0.0),
            basis="judge",
        )
    else:
        raise ValueError(f"unknown pending-item kind {kind!r}")


def infer_verdict_kind(verdict: dict[str, Any]) -> str | None:
    """Best-effort kind inference for a verdict whose key is not in pending.

    Field names are mutually exclusive across the four verdict shapes, so
    this is unambiguous when it returns at all; ``None`` means undecidable.
    """
    if "deviation" in verdict or "risk_delta" in verdict:
        return "deviation"
    if "provenance" in verdict:
        return "provenance"
    if "in_scope" in verdict:
        return "scope"
    if "taxonomy_id" in verdict:
        return "classify"
    return None


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
    #: Staleness policy + shared run tally (see :mod:`playbook_engine.rubric`).
    #: The default instance replays legacy verdicts and re-queues stale ones.
    rubric: RubricPolicy = field(default_factory=RubricPolicy)
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
        # Active/custom entries only (issue #151) — matches cli.py's
        # segment_cmd (taxonomy.classifier_entries()) and OPF §5 ("a
        # compiler MUST only classify clauses into active or custom
        # entries"). An inactive entry must never appear in the "allowed
        # ids" a judge is shown, or a verdict naming it passes
        # validate_verdict at apply time and then crashes classify_tree on
        # replay.
        tax_labels = classifier_eligible_ids(taxonomy)
        # Computed once per batch from the taxonomy actually in force, so an
        # edit to spec/taxonomy/*.yaml (a re-worded label or description that
        # leaves the id set — and therefore the payload key — untouched)
        # invalidates the classify verdicts it should.
        current_rubric = rubric_version("classify", taxonomy=taxonomy)

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

            record = self.store.get_record(payload)
            if (
                record is not None
                and not self.rubric.evaluate(
                    "classify", record.rubric_version, current_rubric
                ).replay
            ):
                # Stored under a rubric that has since moved (or unstamped
                # under --strict-rubric): the banked answer is an answer to a
                # different question. Re-queue instead of replaying silently.
                self.pending.add(key, "classify", payload, current_rubric)
                results.append(_classification_needs_review())
                continue
            cached = record.verdict if record is not None else None
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
                    self.pending.add(key, "classify", payload, current_rubric)
                    results.append(_classification_needs_review())
            else:
                # Queue for external verdict (deduplicated by key).
                self.pending.add(key, "classify", payload, current_rubric)
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
    plus traceability context — taxonomy_id, clause_path, document_id,
    version_from, version_to when present) to the pending queue and returns
    ``DeviationResult(basis="needs_review")``.

    Duplicate payloads within a single ``assess_batch`` call produce exactly one
    pending-queue entry.

    Content-hash key vs. stored context (issue #109): the cache/dedup key is
    derived from ``stage`` + ``hunk`` + ``our_standard`` only. Traceability
    context (``taxonomy_id``, ``clause_path``, ``document_id``, and — since
    issue #166 — ``version_from``/``version_to``) is recorded in the pending
    payload for human/tooling review but deliberately excluded from the
    hash — those fields are per-clause/per-document metadata, not judgment
    content, and folding them in would defeat cross-document dedup of an
    identical hunk/standard pair (e.g. the same boilerplate clause appearing
    verbatim in two different agreements would otherwise queue as two
    distinct pending items and never share a verdict).

    ``version_from``/``version_to`` (issue #166) are the normalized-tree
    version ids (``ClauseDiff.clause_version_before``/``clause_version_after``)
    a relocation-triage reviewer needs to open the right
    ``$OUT/normalized/<document_id>/*.clauses.json`` files when checking
    whether a clause that disappeared in one hunk reappears, unchanged, in an
    adjacent version — see REFERENCE.md's relocation-triage bullet.
    """

    store: VerdictStore
    pending: PendingQueue
    #: See ``StoreBackedClassificationJudge.rubric``.
    rubric: RubricPolicy = field(default_factory=RubricPolicy)

    def assess_batch(
        self,
        items: list[dict[str, str]],
        our_standard: str,
    ) -> list[DeviationResult]:
        """Assess deviation for *items* from the store or queue for external review.

        Args:
            items:        Hunk payload dicts (each must have at minimum a
                         ``"hunk"`` key; ``"taxonomy_id"``, ``"clause_path"``,
                         ``"document_id"``, ``"version_from"``, and
                         ``"version_to"`` are optional traceability
                         context — see class docstring).
            our_standard: Canonical text from the playbook standard for this clause type.

        Returns:
            One ``DeviationResult`` per item in the same order.
        """
        current_rubric = rubric_version("deviation")
        results: list[DeviationResult] = []
        for item in items:
            # Full hunk + full our_standard — NOT truncated (contrast with
            # judgment.py). This is the content-hash payload only: it must NOT
            # include taxonomy_id/clause_path/document_id/version_from/
            # version_to (see class docstring).
            hash_payload = {
                "stage": "deviation",
                "hunk": item.get("hunk", ""),
                "our_standard": our_standard,
            }
            key = _payload_key(hash_payload)

            record = self.store.get_record(hash_payload)
            if (
                record is not None
                and not self.rubric.evaluate(
                    "deviation", record.rubric_version, current_rubric
                ).replay
            ):
                # Rubric moved under this verdict — re-queue with the same
                # traceability context the miss path records.
                self.pending.add(
                    key,
                    "deviation",
                    {
                        **hash_payload,
                        "taxonomy_id": item.get("taxonomy_id", ""),
                        "clause_path": item.get("clause_path", ""),
                        "document_id": item.get("document_id", ""),
                        "version_from": item.get("version_from", ""),
                        "version_to": item.get("version_to", ""),
                    },
                    current_rubric,
                )
                results.append(_deviation_needs_review())
                continue
            cached = record.verdict if record is not None else None
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
                    # Re-queue with the same traceability context the miss
                    # path records, so a re-queued item is reviewable too.
                    self.pending.add(
                        key,
                        "deviation",
                        {
                            **hash_payload,
                            "taxonomy_id": item.get("taxonomy_id", ""),
                            "clause_path": item.get("clause_path", ""),
                            "document_id": item.get("document_id", ""),
                            "version_from": item.get("version_from", ""),
                            "version_to": item.get("version_to", ""),
                        },
                        current_rubric,
                    )
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
                    "version_from": item.get("version_from", ""),
                    "version_to": item.get("version_to", ""),
                }
                self.pending.add(key, "deviation", full_payload, current_rubric)
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
    #: See ``StoreBackedClassificationJudge.rubric``. Provenance has no
    #: derived input at this seam (``our_party_aliases`` never reaches the
    #: ``ProvenanceJudge`` protocol), so only the manual half moves.
    rubric: RubricPolicy = field(default_factory=RubricPolicy)

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

        current_rubric = rubric_version("provenance")

        record = self.store.get_record(payload)
        if (
            record is not None
            and not self.rubric.evaluate("provenance", record.rubric_version, current_rubric).replay
        ):
            self.pending.add(key, "provenance", payload, current_rubric)
            return _provenance_needs_review()
        cached = record.verdict if record is not None else None
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
                self.pending.add(key, "provenance", payload, current_rubric)
                return _provenance_needs_review()
        self.pending.add(key, "provenance", payload, current_rubric)
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
    #: See ``StoreBackedClassificationJudge.rubric``. The scope rubric's
    #: derived half is the agreement-type definition being gated on, so
    #: editing its description/aliases in the config re-queues scope verdicts.
    rubric: RubricPolicy = field(default_factory=RubricPolicy)

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

        current_rubric = rubric_version("scope", agreement_type=agreement_type)

        record = self.store.get_record(payload)
        if (
            record is not None
            and not self.rubric.evaluate("scope", record.rubric_version, current_rubric).replay
        ):
            self.pending.add(key, "scope", payload, current_rubric)
            raise ScopeNeedsReviewError(
                f"Stored scope verdict for document {tree.document_id!r} was made "
                "under an older rubric — re-queued for re-judgement."
            )
        cached = record.verdict if record is not None else None
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
                self.pending.add(key, "scope", payload, current_rubric)
                raise ScopeNeedsReviewError(
                    f"Malformed stored scope verdict for document {tree.document_id!r} — "
                    "re-queued for external review."
                ) from exc

        self.pending.add(key, "scope", payload, current_rubric)
        raise ScopeNeedsReviewError(
            f"No stored scope verdict for document {tree.document_id!r} — "
            "queued for external review."
        )
