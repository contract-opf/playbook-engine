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
- ``StoreBackedProvenanceJudge``     — implements ``ProvenanceJudge``.
- ``StoreBackedScopeJudge``          — implements ``ScopeJudge``.

These are drop-in replacements for the judge parameters of
``mine_corpus(scope_judge=…, classification_judge=…, provenance_judge=…)``.
There is no deviation judge: every deviation is the deterministic standard
check, so no deviation item is ever queued.

Adding a judge kind (the seam the ``equivalence`` kind plugs into): write the
store-backed judge beside the three above, register its verdict shape with
:func:`register_verdict_kind` (apply-time validation and kind inference for
``playbook judge-apply``) and its rubric with
:func:`playbook_engine.rubric.register_judge_kind`. Everything generic —
``VerdictStore``, ``PendingQueue``, the rubric stamp, ``judge`` /
``judge-apply`` and the plan output — already works per kind.

Rubric versioning (see :mod:`playbook_engine.rubric`): because the store key
is purely content-derived, a change to the *judging criteria* — the taxonomy,
the prose rubric in the ``playbook-from-corpus`` skill — would otherwise
replay every banked verdict forever, since the clause text never moved.  Each store hit is therefore checked against the rubric
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
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from playbook_engine.clause_classifier import ClauseClassification
from playbook_engine.clause_tree import ClauseNode, ClauseTree
from playbook_engine.config import AgreementType
from playbook_engine.provenance_detector import (
    _PROVENANCE_VALUES,
    PROVENANCE_UNKNOWN,
    ProvenanceResult,
)
from playbook_engine.rubric import (
    RubricPolicy,
    RubricStamp,
    classifier_eligible_ids,
    rubric_version,
)
from playbook_engine.scope_gate import ScopeDecision

_log = logging.getLogger(__name__)

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
    Later lines for the same key win, so a re-applied verdict is an append
    rather than a rewrite: the original record stays on disk as an audit
    trail of what the verdict was banked under.
    """

    def __init__(self, store_path: Path) -> None:
        self._store_path = store_path
        self._store: dict[str, StoredVerdict] = {}  # key -> record
        # Open lookup captures (issue #219) — see capture_lookups().
        self._captures: list[set[str]] = []
        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, payload: Any) -> dict[str, Any] | None:
        """Return the stored verdict dict for *payload*, or ``None`` on miss."""
        record = self._lookup(_payload_key(payload))
        return record.verdict if record is not None else None

    def get_record(self, payload: Any) -> StoredVerdict | None:
        """Return the full stored record (verdict + rubric stamp), or ``None``."""
        return self._lookup(_payload_key(payload))

    @contextmanager
    def capture_lookups(self) -> Iterator[set[str]]:
        """Collect every key looked up through ``get``/``get_record`` meanwhile.

        Issue #219: the L1-L4 stage cache stays on under store-backed judges
        by recording, with each cached document result, the verdict keys that
        result was built from — then refusing to replay it when any of them
        has since changed (see :meth:`fingerprint`). Nestable; a lookup is
        recorded into every open capture.
        """
        seen: set[str] = set()
        self._captures.append(seen)
        try:
            yield seen
        finally:
            self._captures.remove(seen)

    def fingerprint(self, key: str) -> str | None:
        """Opaque digest of what *key* currently holds — ``None`` when absent.

        Covers the verdict AND its rubric stamp: a ``judge-apply`` that
        overwrites a verdict moves it, so a cached result built on the old
        record is not replayed.
        """
        record = self._store.get(key)
        if record is None:
            return None
        raw = json.dumps(
            {
                "verdict": record.verdict,
                "rubric": record.rubric.to_dict() if record.rubric is not None else None,
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(raw.encode()).hexdigest()

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
                     subsequent judge run.
        """
        self._store[key] = StoredVerdict(verdict=verdict, rubric=rubric)
        self._append(key, verdict, rubric)

    def records(self) -> list[tuple[str, StoredVerdict]]:
        """Return ``(key, record)`` for every stored verdict, key-sorted."""
        return sorted(self._store.items())

    def __len__(self) -> int:
        return len(self._store)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _lookup(self, key: str) -> StoredVerdict | None:
        for seen in self._captures:
            seen.add(key)
        return self._store.get(key)

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

    Each record: ``{"key": "<sha256>", "kind": "classify"|"provenance"|"scope"|
    "segment", "payload": {…}}`` (plus the registered kinds).

    Deduplication: payloads with the same key are recorded at most once,
    even across multiple ``add()`` calls on the same instance.  This is the
    within-instance dedup (within-batch + cross-batch for the same object);
    persistence does not deduplicate across runs (the external caller is
    responsible for that).
    """

    def __init__(self, queue_path: Path) -> None:
        self._queue_path = queue_path
        self._seen_keys: set[str] = set()
        # Open add captures (issue #219) — see capture_adds().
        self._captures: list[set[str]] = []

    @contextmanager
    def capture_adds(self) -> Iterator[set[str]]:
        """Collect the key of every ``add`` call meanwhile — deduplicated or not.

        Issue #219: a document result that queued anything rests on an
        unresolved verdict and must never be replayed from the stage cache —
        replaying it would also drop its items from this round's queue. The
        within-instance dedup in :meth:`add` is exactly why the attempt, not
        the write, is what gets recorded: a second document asking the same
        question is just as unresolved as the first.
        """
        seen: set[str] = set()
        self._captures.append(seen)
        try:
            yield seen
        finally:
            self._captures.remove(seen)

    def add(self, key: str, kind: str, payload: Any, rubric_version: str | None = None) -> bool:
        """Append *payload* to the queue if *key* has not been seen before.

        Args:
            key:            Content hash from ``_payload_key(payload)``.
            kind:           One of ``"classify"``, ``"provenance"``,
                            ``"scope"``, ``"segment"`` (or a kind
                            registered through :func:`register_verdict_kind`).
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
        for seen in self._captures:
            seen.add(key)
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


def _provenance_needs_review() -> ProvenanceResult:
    """Sentinel returned when no stored verdict is available for a provenance payload.

    No side was determined, so the result says so: ``"unknown"`` at 0.0
    (always ambiguous), recorded as paper ``"unknown"`` — never coerced to
    ``counterparty_paper`` (issue #225).
    """
    return ProvenanceResult(
        provenance=PROVENANCE_UNKNOWN,
        confidence=0.0,
        basis="needs_review",
    )


def _require_provenance_side(provenance: Any) -> None:
    """Raise ``ValueError`` unless a stored provenance verdict names a side.

    A stored verdict names a side (the judge's two-valued answer vocabulary,
    rubric._PROVENANCE_ANSWERS). ``"unknown"`` is reserved for the engine's
    own no-verdict sentinel (issue #225), so it is rejected here — at apply
    time and on store replay alike — before ``ProvenanceResult``, which
    accepts it.
    """
    if provenance not in _PROVENANCE_VALUES:
        raise ValueError(
            f"Unknown provenance: {provenance!r}. Must be one of {sorted(_PROVENANCE_VALUES)}"
        )


# ---------------------------------------------------------------------------
# Apply-time verdict validation (used by `playbook judge-apply`)
# ---------------------------------------------------------------------------

#: Basis values that mean "no real judgment happened". A verdict *file*
#: (producer-supplied) must never carry one: on replay it either loops the
#: item forever or, worse, replays as a permanently-unjudged result while the
#: pending queue looks drained. The store-backed judges set these themselves.
_UNRESOLVED_VERDICT_BASES = frozenset({"needs_review", "judge_error", "stub"})

#: The only basis values a *producer-supplied* classify verdict may carry.
#: Mirrors the whitelist ``clause_classifier.classify_tree`` enforces on
#: replay (``"judge"``, ``"judge_error"``, ``"needs_review"``, ``"unclassified"``),
#: minus the two engine-internal bases already rejected above.
_CLASSIFY_REPLAYABLE_BASES = frozenset({"judge", "unclassified"})

#: The only basis value a *producer-supplied* provenance verdict may carry.
#: ``provenance_detector.ProvenanceResult`` accepts a wide ``_BASIS_VALUES``
#: set (``template_similarity``, ``alias_first_party``, ``hint``, ...) because
#: the engine's own deterministic detectors construct it with those bases —
#: but a producer verdict file only ever represents LLM judgment, so only
#: ``"llm"`` is replayable here (REFERENCE.md: "this is the one kind where
#: 'llm' is correct").
_PROVENANCE_REPLAYABLE_BASES = frozenset({"llm"})


def _validate_confidence_field(
    verdict: dict[str, Any], field: str, *, allow_none: bool = False
) -> None:
    """Raise ``ValueError`` if ``verdict[field]`` is not a valid confidence.

    Runs *before* the corresponding dataclass is constructed so a
    producer-supplied stringified number (e.g. ``"0.8"``) never reaches the
    dataclass's ``0.0 <= confidence <= 1.0`` comparison and raises a bare,
    line-number-free ``TypeError`` instead of an actionable ``ValueError``
    (issue #161). Absent fields are left to the caller's ``dict.get(...,
    default)`` fallback and are not validated here.

    Args:
        verdict:    The verdict dict from the producer's JSONL line.
        field:      Field name to check (e.g. ``"confidence"``,
                    ``"scope_confidence"``).
        allow_none: Whether an explicit ``null`` is acceptable.
    """
    if field not in verdict:
        return
    value = verdict[field]
    if value is None:
        if allow_none:
            return
        raise ValueError(f"'{field}' must not be null")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"'{field}' must be a number in [0, 1], got {value!r}")
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"'{field}' must be in [0, 1], got {value!r}")


def _validate_classify(verdict: dict[str, Any]) -> None:
    classify_basis = verdict.get("basis", "judge")
    if classify_basis not in _CLASSIFY_REPLAYABLE_BASES:
        raise ValueError(
            f"basis {classify_basis!r} is not replayable for a classify verdict; "
            f"a supplied verdict must carry basis in "
            f"{sorted(_CLASSIFY_REPLAYABLE_BASES)!r} (clause_classifier.classify_tree "
            "rejects anything else on replay)"
        )
    _validate_confidence_field(verdict, "confidence")
    ClauseClassification(
        taxonomy_id=verdict.get("taxonomy_id"),
        confidence=verdict.get("confidence", 0.0),
        basis=classify_basis,
    )


def _validate_provenance(verdict: dict[str, Any]) -> None:
    if "provenance" not in verdict:
        raise ValueError("missing 'provenance' field")
    provenance_basis = verdict.get("basis", "llm")
    if provenance_basis not in _PROVENANCE_REPLAYABLE_BASES:
        raise ValueError(
            f"basis {provenance_basis!r} is not replayable for a provenance verdict; "
            f"a supplied verdict must carry basis in "
            f"{sorted(_PROVENANCE_REPLAYABLE_BASES)!r} (a producer verdict file only "
            "ever represents LLM judgment; deterministic bases are set by the engine "
            "itself, not by you)"
        )
    _require_provenance_side(verdict["provenance"])
    _validate_confidence_field(verdict, "confidence")
    ProvenanceResult(
        provenance=verdict["provenance"],
        confidence=verdict.get("confidence", 0.0),
        basis=provenance_basis,
    )


def _validate_scope(verdict: dict[str, Any]) -> None:
    if not isinstance(verdict.get("in_scope"), bool):
        raise ValueError("'in_scope' must be a JSON boolean")
    _validate_confidence_field(verdict, "scope_confidence")
    ScopeDecision(
        in_scope=verdict["in_scope"],
        scope_rationale=verdict.get("scope_rationale") or "Replayed from stored verdict.",
        scope_confidence=verdict.get("scope_confidence", 0.0),
        basis="judge",
    )


#: ``kind -> (validate, matches)`` for every pending-item kind ``judge-apply``
#: accepts. ``validate(verdict)`` raises ``ValueError`` on a verdict the
#: kind's store-backed judge could not replay; ``matches(verdict)`` says
#: whether a verdict's field shape belongs to the kind (used when the key is
#: not in ``pending.jsonl``). Extend with :func:`register_verdict_kind`.
_VERDICT_KINDS: dict[
    str, tuple[Callable[[dict[str, Any]], None], Callable[[dict[str, Any]], bool]]
] = {
    "provenance": (_validate_provenance, lambda v: "provenance" in v),
    "scope": (_validate_scope, lambda v: "in_scope" in v),
    "classify": (_validate_classify, lambda v: "taxonomy_id" in v),
}


def register_verdict_kind(
    kind: str,
    validate: Callable[[dict[str, Any]], None],
    matches: Callable[[dict[str, Any]], bool],
) -> None:
    """Teach ``judge-apply`` a new pending-item *kind*.

    *validate* must reconstruct what the kind's store-backed judge builds on
    replay and raise ``ValueError`` with an actionable message otherwise, so a
    verdict accepted here always replays. *matches* is the kind's field-shape
    test for :func:`infer_verdict_kind`; field names must be mutually
    exclusive with every other registered kind. Register the kind's rubric
    with :func:`playbook_engine.rubric.register_judge_kind` too.
    """
    _VERDICT_KINDS[kind] = (validate, matches)


def validate_verdict(kind: str, verdict: dict[str, Any]) -> None:
    """Validate a producer-supplied *verdict* for *kind* at apply time.

    Reconstructs the exact dataclass the store-backed judge would build on
    replay, so any verdict accepted here is guaranteed to replay instead of
    silently re-queueing (the issue #182 malformed-verdict loop). Confidence
    fields are type/range-checked up front (issue #161) so a stringified or
    out-of-range confidence raises an actionable ``ValueError`` here instead
    of a bare ``TypeError`` from dataclass construction. Raises
    ``ValueError`` with an actionable message on the first problem.

    Args:
        kind:    Pending-item kind: ``classify`` / ``provenance`` / ``scope``
                 (or a kind added with :func:`register_verdict_kind`).
        verdict: The verdict dict from the producer's JSONL line.
    """
    basis = verdict.get("basis")
    if basis in _UNRESOLVED_VERDICT_BASES:
        raise ValueError(
            f"basis {basis!r} is engine-internal (means 'not judged'); a "
            "supplied verdict must carry a real basis — use 'judge' "
            "('llm' for provenance)"
        )
    registered = _VERDICT_KINDS.get(kind)
    if registered is None:
        raise ValueError(f"unknown pending-item kind {kind!r}")
    registered[0](verdict)


def infer_verdict_kind(verdict: dict[str, Any]) -> str | None:
    """Best-effort kind inference for a verdict whose key is not in pending.

    Field names are mutually exclusive across the verdict shapes, so this is
    unambiguous when it returns at all; ``None`` means undecidable.
    """
    for kind, (_validate, matches) in _VERDICT_KINDS.items():
        if matches(verdict):
            return kind
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
                    # caller's blanket except.
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
    agreement_type + candidate aliases) to the pending queue and returns
    ``ProvenanceResult(provenance="unknown", confidence=0.0,
    basis="needs_review")`` — no side is claimed until a verdict exists, and
    the deal's paper is recorded ``"unknown"`` (issue #225).
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
                # Same two-sides check as validate_verdict: a stored
                # "unknown" (e.g. a hand-written store row) is malformed and
                # re-queued, never replayed as a verdict (issue #225).
                _require_provenance_side(cached["provenance"])
                return ProvenanceResult(
                    provenance=cached["provenance"],
                    confidence=cached.get("confidence", 0.0),
                    basis=cached.get("basis", "llm"),
                )
            except (KeyError, TypeError, ValueError) as exc:
                # Isolate one malformed stored verdict (issue #182) — same
                # pattern as StoreBackedClassificationJudge.
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
    cannot be expressed as a sentinel return value the way the classify and
    provenance store-backed judges use ``basis="needs_review"``.

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
                # pattern as the other store-backed judges. Scope has
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
