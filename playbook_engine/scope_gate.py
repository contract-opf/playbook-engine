"""Scope gate — L1b pipeline stage.

Decides whether each document in the corpus is *in scope* for the target
agreement type.  Filename is NOT dispositive; the decision is based on the
document's purpose and clause profile.

Out-of-scope documents are NEVER silently dropped.  Every ``ScopeDecision``
carries a ``scope_rationale`` and ``scope_confidence``, and all decisions are
accumulated in a ``ScopeLog`` for persistent storage in ``scope.json`` and
in the OPF ``corpus[]`` field (§3.6).

Architecture:
  ``scope_gate()`` applies deterministic pre-checks first (empty document →
  out of scope; trivially short document → low-confidence check) and then
  delegates substantive judgment to a ``ScopeJudge`` instance.

  The ``ScopeJudge`` protocol is the LLM integration point: at runtime, a
  real judge calls an LLM with the document's clause profile and the target
  agreement type; in tests, a ``MockScopeJudge`` is injected.

  Using a protocol rather than a flag keeps the deterministic/LLM boundary
  explicit and testable.

  Use ``scope_gate_and_record()`` (not bare ``scope_gate()``) in pipeline
  loops to guarantee §3.6: every document — including out-of-scope and
  judge-error ones — is recorded in the log before the loop continues.

``ScopeDecision.basis`` values:
  ``"deterministic_empty"``    — no extractable text; skipped without LLM.
  ``"deterministic_trivial"``  — document too short to be a real agreement.
  ``"judge"``                  — decision delegated to a real (LLM-backed) judge.
  ``"stub"``                   — decision delegated to a stub judge (no LLM
                                 configured, e.g. ``_AllInScopeJudge``, the CLI
                                 default). Distinct from ``"judge"`` so downstream
                                 artifacts (scope.json, the assembled playbook) can
                                 tell a rubber-stamped scope decision from one a
                                 real judge actually evaluated — a stub verdict is
                                 not evidence that the document is genuinely in
                                 scope, only that it was never rejected.
  ``"judge_error"``            — judge raised; document is RETAINED (``in_scope=True``)
                                 with zero confidence so it is never silently dropped.
                                 The document is flagged for review rather than deleted.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from playbook_engine.clause_tree import ClauseTree
from playbook_engine.config import AgreementType

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MIN_CLAUSE_COUNT: int = 2
"""Documents with fewer nodes than this are treated as trivially short."""

# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

_BASIS_VALUES = frozenset(
    {"deterministic_empty", "deterministic_trivial", "judge", "stub", "judge_error"}
)


@dataclass(frozen=True)
class ScopeDecision:
    """The scope decision for one document.

    Attributes:
        in_scope:          True if the document is in scope for the playbook.
        scope_rationale:   Human-readable explanation (always present).
        scope_confidence:  Float in [0, 1].
        basis:             How the decision was reached (``_BASIS_VALUES``).
    """

    in_scope: bool
    scope_rationale: str
    scope_confidence: float
    basis: str

    def __post_init__(self) -> None:
        if self.basis not in _BASIS_VALUES:
            raise ValueError(
                f"Unknown basis: {self.basis!r}. Must be one of {sorted(_BASIS_VALUES)}"
            )
        if not 0.0 <= self.scope_confidence <= 1.0:
            raise ValueError(f"scope_confidence must be in [0, 1], got {self.scope_confidence}")
        if not self.scope_rationale.strip():
            raise ValueError("scope_rationale must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "in_scope": self.in_scope,
            "scope_rationale": self.scope_rationale,
            "scope_confidence": round(self.scope_confidence, 6),
            "basis": self.basis,
        }


# ---------------------------------------------------------------------------
# Judge protocol (LLM integration point)
# ---------------------------------------------------------------------------


@runtime_checkable
class ScopeJudge(Protocol):
    """Protocol for scope judgment.

    Implementations may call an LLM, apply rule-based heuristics, or both.
    The ``scope_gate`` function calls this only after passing deterministic
    pre-checks — callers never need to handle empty-document edge cases.

    Contract: real (LLM-backed) implementations MUST return a
    ``ScopeDecision`` with ``basis="judge"``.  A stub implementation that
    performs no genuine judgment (e.g. ``_AllInScopeJudge``, the CLI default
    when no LLM is configured) MUST instead return ``basis="stub"`` — never
    ``"judge"``, which would masquerade a fabricated default as a real
    verdict.  Returning any other basis value raises ``ValueError`` inside
    ``scope_gate()``.
    """

    def judge(
        self,
        tree: ClauseTree,
        agreement_type: AgreementType,
    ) -> ScopeDecision:
        """Return a scope decision for *tree* against *agreement_type*.

        Args:
            tree:            Segmented clause tree of the document.
            agreement_type:  Target agreement type from the engine config.

        Returns:
            ``ScopeDecision`` with ``basis="judge"`` (real judge) or
            ``basis="stub"`` (stub judge — no LLM configured).
        """
        ...  # pragma: no cover


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def scope_gate(
    tree: ClauseTree,
    agreement_type: AgreementType,
    judge: ScopeJudge,
) -> ScopeDecision:
    """Apply the L1b scope gate to *tree*.

    Deterministic pre-checks run before the judge to avoid unnecessary LLM
    calls on trivially invalid documents.

    If the judge raises (e.g. LLM timeout, refusal, parse error), the document
    is RETAINED (``in_scope=True``) with ``basis="judge_error"`` and
    ``scope_confidence=0.0`` so that §3.6 is preserved — the document is never
    silently dropped.  The retained document is flagged for review rather than
    fabricating an out-of-scope verdict.

    Args:
        tree:            Segmented clause tree of the document to evaluate.
        agreement_type:  Target agreement type (from engine config).
        judge:           ``ScopeJudge`` instance for substantive evaluation.

    Returns:
        ``ScopeDecision``.  Out-of-scope results carry a ``scope_rationale``
        explaining the exclusion and are never silently discarded.

    Raises:
        ValueError: if the judge returns a ``ScopeDecision`` whose ``basis``
                    is one of the deterministic pre-check values
                    (``"deterministic_empty"`` / ``"deterministic_trivial"``),
                    which are reserved for ``scope_gate`` itself and indicate a
                    programming error in the judge implementation.  ``"judge"``,
                    ``"stub"`` and ``"judge_error"`` are all legitimate judge
                    returns — the last is produced by caching wrappers such as
                    ``BatchedScopeJudge`` when a delegate raises.
    """
    all_nodes = list(tree.all_nodes())

    # Pre-check 1: empty document.
    if not all_nodes:
        return ScopeDecision(
            in_scope=False,
            scope_rationale="Document produced no extractable text or clauses.",
            scope_confidence=0.95,
            basis="deterministic_empty",
        )

    # Pre-check 2: trivially short — fewer than MIN_CLAUSE_COUNT nodes.
    # Such a document cannot meaningfully represent any agreement type.
    if len(all_nodes) < MIN_CLAUSE_COUNT:
        return ScopeDecision(
            in_scope=False,
            scope_rationale=(
                f"Document has only {len(all_nodes)} clause node(s) — "
                f"too short to evaluate as a {agreement_type.name}."
            ),
            scope_confidence=0.80,
            basis="deterministic_trivial",
        )

    # Delegate to the judge for substantive evaluation.
    # Wrap in a broad except to prevent any LLM/network/parse failure from
    # silently dropping the document (§3.6).
    try:
        decision = judge.judge(tree, agreement_type)
    except Exception as exc:  # noqa: BLE001
        # Retain the document rather than silently deleting it.  A transient LLM
        # failure is NOT a verdict — setting in_scope=True here keeps the document
        # in the pipeline so it can be routed for human review (§P1.5 / §2 error
        # model).  Callers must inspect basis="judge_error" to route it.
        return ScopeDecision(
            in_scope=True,
            scope_rationale=(
                f"Scope judge raised an unexpected error: {type(exc).__name__}: {exc}. "
                "Document retained and flagged for review."
            ),
            scope_confidence=0.0,
            basis="judge_error",
        )

    # Programming-error guard: judges must not return the deterministic
    # pre-check basis values — those are reserved for scope_gate itself.
    #   - "judge"  — real (LLM-backed) judge
    #   - "stub"   — stub judge (e.g. _AllInScopeJudge)
    #   - "judge_error" — legitimate sentinel: produced by scope_gate's own
    #                     except block above OR returned as a normal value by a
    #                     caching wrapper (e.g. BatchedScopeJudge) that catches
    #                     its delegate's exception internally. Must NOT be
    #                     rejected here, or a wrapped raising delegate would
    #                     crash the compile instead of being retained for review.
    if decision.basis in ("deterministic_empty", "deterministic_trivial"):
        raise ValueError(
            f"ScopeJudge.judge() returned reserved pre-check basis={decision.basis!r}; "
            "implementations must return basis='judge', 'stub', or 'judge_error'."
        )

    return decision


def scope_gate_and_record(
    tree: ClauseTree,
    document_id: str,
    agreement_type: AgreementType,
    judge: ScopeJudge,
    log: ScopeLog,
) -> ScopeDecision:
    """Run the scope gate and immediately record the result in *log*.

    This is the preferred entry point for pipeline loops.  Calling bare
    ``scope_gate()`` and recording separately risks forgetting to log
    out-of-scope decisions, which would violate OPF §3.6.

    Args:
        tree:            Segmented clause tree of the document to evaluate.
        document_id:     Stable identifier for the document (for the log).
        agreement_type:  Target agreement type (from engine config).
        judge:           ``ScopeJudge`` instance for substantive evaluation.
        log:             ``ScopeLog`` to record the decision into.

    Returns:
        The ``ScopeDecision`` that was recorded.
    """
    decision = scope_gate(tree, agreement_type, judge)
    log.record(document_id, decision)
    return decision


# ---------------------------------------------------------------------------
# Scope log
# ---------------------------------------------------------------------------


@dataclass
class ScopeLogEntry:
    """One entry in the scope log, corresponding to one document."""

    document_id: str
    decision: ScopeDecision

    def to_dict(self) -> dict[str, Any]:
        d = self.decision.to_dict()
        d["document_id"] = self.document_id
        return d


@dataclass
class ScopeLog:
    """Accumulates scope decisions and serialises them to ``scope.json``.

    Usage::

        log = ScopeLog(agreement_type_id="educational-affiliation")
        scope_gate_and_record(tree, "doc-001", agreement_type, judge, log)
        log.write(out_dir / "scope.json")
    """

    agreement_type_id: str
    entries: list[ScopeLogEntry] = field(default_factory=list)

    def record(self, document_id: str, decision: ScopeDecision) -> None:
        """Add a scope decision for *document_id*."""
        self.entries.append(ScopeLogEntry(document_id=document_id, decision=decision))

    @property
    def in_scope_ids(self) -> list[str]:
        """Return IDs of in-scope documents (excludes judge_error retained docs)."""
        return [
            e.document_id
            for e in self.entries
            if e.decision.in_scope and e.decision.basis != "judge_error"
        ]

    @property
    def out_of_scope_ids(self) -> list[str]:
        """Return IDs of out-of-scope documents."""
        return [e.document_id for e in self.entries if not e.decision.in_scope]

    @property
    def judge_error_ids(self) -> list[str]:
        """Return IDs of documents retained after a scope-judge failure.

        These documents are in the pipeline (``in_scope=True``) but require
        human review — their scope verdict could not be computed.
        """
        return [e.document_id for e in self.entries if e.decision.basis == "judge_error"]

    def to_dict(self) -> dict[str, Any]:
        docs: list[dict[str, Any]] = []
        in_scope_count = 0
        out_of_scope_count = 0
        judge_error_count = 0
        for e in self.entries:
            docs.append(e.to_dict())
            if e.decision.basis == "judge_error":
                judge_error_count += 1
            elif e.decision.in_scope:
                in_scope_count += 1
            else:
                out_of_scope_count += 1
        return {
            "agreement_type_id": self.agreement_type_id,
            "stats": {
                "total": len(self.entries),
                "in_scope": in_scope_count,
                "out_of_scope": out_of_scope_count,
                "judge_error": judge_error_count,
            },
            "documents": docs,
        }

    def write(self, path: Path) -> None:
        """Write the log to *path* as JSON (atomic rename to avoid partial writes)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        os.replace(tmp, path)
