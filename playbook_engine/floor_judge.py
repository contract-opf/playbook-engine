"""Floor judge — optional NL invariants + LLM-judge detector (issue #145).

Re-envisions the OPF Floor (spec/playbook.schema-0.2.json ``floor.invariants``)
around *judgment*, not regex. Per the issue #145 Q4 direction (2026-07-09,
settled):

  - Floor entries are OPTIONAL natural-language invariants (e.g. "never accept
    uncapped liability"), never a required field a corpus-mined playbook must
    populate — ``floor.invariants`` MAY be empty and the playbook still
    compiles cleanly (see :func:`floor_coverage_gate`, zero-invariants case).
  - An LLM judge is the ONLY detector. There is no lexical/regex matcher —
    a judge decides whether an invariant holds against the in-scope clauses
    it is evaluated over.
  - The ONE deterministic part is a COVERAGE gate: every *present* invariant
    must be evaluated and its verdict + rationale logged, or the run fails
    loud. An invariant that was silently never checked is exactly the failure
    mode the Floor exists to prevent.

Architecture mirrors the pipeline's other LLM-integration seams
(:mod:`playbook_engine.scope_gate`, :mod:`playbook_engine.deviation_classifier`):
a frozen result dataclass, a ``Protocol`` judge (real judges call an LLM; tests
inject a fake), and a gate function that enforces the deterministic contract
around the judge's output. ``FloorCoverageError`` propagates uncaught on
failure — same fail-loud contract as ``SegmentationQAError`` and
``ScopeJudge``'s reserved-basis ``ValueError``: there is no silent fallback.

Scope note: this module is the *mechanism* only, verified offline with a fake
judge (per issue #145's required verification). Per OPF-SPEC.md
§3.7, the Floor is enforced at *review time*, in both paper contexts (our
paper: over the standard-form diff; their paper: over the extracted,
taxonomy-tagged clauses of the document under negotiation) — not at corpus-
compile time, when there is no single "document under review" to evaluate
against, only aggregated corpus evidence. Wiring a real call site into a live
review flow, and authoring real invariant content, are explicitly out of
scope here (see issue #151, needs-human) — this module is built to the same
Protocol+gate dependency-injection seam as every other judge in this codebase
specifically so that a future review pipeline (in this repo or a consumer)
can adopt it with no further plumbing.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VERDICT_VALUES = frozenset({"clear", "violation", "needs_review"})
_BASIS_VALUES = frozenset({"judge", "stub", "judge_error"})


# ---------------------------------------------------------------------------
# FloorInvariant — schema-0.2 floor.invariants[] item
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FloorInvariant:
    """One optional natural-language Floor invariant.

    Mirrors ``spec/playbook.schema-0.2.json``'s ``floor.invariants[]`` item
    shape exactly: ``id``/``statement`` required, ``rationale`` optional.

    Attributes:
        id:        Stable identifier (referenced by ``FloorVerdict.invariant_id``).
        statement: The natural-language invariant itself, e.g. "Never accept
                   uncapped liability." Judged, never lexically matched.
        rationale: Optional human-readable justification for why this is a
                   Floor rule (not Posture) — e.g. the admission test result.
    """

    id: str
    statement: str
    rationale: str = ""

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("FloorInvariant.id must not be empty")
        if not self.statement.strip():
            raise ValueError("FloorInvariant.statement must not be empty")

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"id": self.id, "statement": self.statement}
        if self.rationale:
            d["rationale"] = self.rationale
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> FloorInvariant:
        return cls(id=d["id"], statement=d["statement"], rationale=d.get("rationale", ""))


# ---------------------------------------------------------------------------
# FloorVerdict — the judge's answer for one invariant
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FloorVerdict:
    """The judge's verdict for one invariant, evaluated against in-scope clauses.

    Attributes:
        invariant_id: The :class:`FloorInvariant.id` this verdict answers for.
                      :func:`floor_coverage_gate` matches verdicts back to
                      invariants by this field — a judge that returns a
                      verdict under the wrong id is indistinguishable from one
                      that dropped the invariant, and the coverage gate fails
                      loud either way.
        verdict:      ``"clear"`` (invariant held; no violation found),
                      ``"violation"`` (the judge found the invariant broken by
                      an in-scope clause), or ``"needs_review"`` (the judge
                      could not reach a confident verdict — this still counts
                      as *covered* for gate purposes, since a verdict WAS
                      logged, but must be surfaced to a human rather than
                      silently treated as clear).
        rationale:    Human-readable explanation — always required, mirroring
                      every other judge result in this codebase
                      (``ScopeDecision.scope_rationale``,
                      ``DeviationResult.rationale``, ...).
        citation:     Evidence pointer for the clause the verdict is grounded
                      in (e.g. ``{"clause_id": ..., "document_id": ...}``).
                      Required when ``verdict == "violation"`` — a violation
                      must never be surfaced without pointing at the
                      offending clause; optional otherwise (a "clear" verdict
                      may have nothing specific to cite).
        basis:        ``"judge"`` (real LLM-backed judge), ``"stub"`` (no LLM
                      configured — a rubber-stamped verdict, not evidence of
                      genuine judgment), or ``"judge_error"`` (the judge
                      raised and a wrapper caught it while still logging a
                      verdict) — mirrors ``ScopeDecision``/``DeviationResult``'s
                      ``basis`` convention.
    """

    invariant_id: str
    verdict: str
    rationale: str
    citation: dict[str, Any] | None = None
    basis: str = "judge"

    def __post_init__(self) -> None:
        if self.verdict not in _VERDICT_VALUES:
            raise ValueError(
                f"FloorVerdict.verdict must be one of {sorted(_VERDICT_VALUES)!r}; "
                f"got {self.verdict!r}"
            )
        if self.basis not in _BASIS_VALUES:
            raise ValueError(
                f"FloorVerdict.basis must be one of {sorted(_BASIS_VALUES)!r}; got {self.basis!r}"
            )
        if not self.rationale.strip():
            raise ValueError("FloorVerdict.rationale must not be empty")
        if self.verdict == "violation" and self.citation is None:
            raise ValueError(
                "FloorVerdict.citation is required when verdict='violation' — a "
                "violation must never be surfaced without pointing at the "
                "offending clause."
            )

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "invariant_id": self.invariant_id,
            "verdict": self.verdict,
            "rationale": self.rationale,
            "basis": self.basis,
        }
        if self.citation is not None:
            d["citation"] = self.citation
        return d


# ---------------------------------------------------------------------------
# FloorJudge protocol (LLM integration point)
# ---------------------------------------------------------------------------


@runtime_checkable
class FloorJudge(Protocol):
    """Protocol for Floor invariant judgment — the LLM integration point.

    Per issue #145's Q4 direction, the ONLY detector is a judge (never a
    regex/lexical matcher). Implementations MAY call an LLM; tests inject a
    fake that returns pre-canned verdicts.

    Contract: :meth:`evaluate_batch` returns at most one :class:`FloorVerdict`
    per invariant (never more than one per id), but MAY return *fewer* than
    ``len(invariants)`` — e.g. because the judge failed on one invariant, or a
    bug dropped it silently. That gap is exactly what :func:`floor_coverage_gate`
    exists to catch: it is not the judge's job to guarantee its own coverage,
    only to attempt every invariant it was given — the gate is the
    deterministic backstop that turns a silent gap into a fail-loud error.
    """

    def evaluate_batch(
        self,
        invariants: Sequence[FloorInvariant],
        clauses: Sequence[dict[str, Any]],
    ) -> list[FloorVerdict]:
        """Evaluate every invariant in *invariants* against *clauses*.

        Args:
            invariants: The Floor's present invariants (schema-0.2
                        ``floor.invariants[]``).
            clauses:    The in-scope clauses of the document under review —
                        minimal dicts (e.g. ``{"clause_id": ..., "text": ...,
                        "heading": ...}``). This protocol intentionally does
                        not depend on ``ClauseTree``/``ClauseNode`` so a judge
                        can be built and tested independent of the
                        corpus-mining pipeline's data model.

        Returns:
            One :class:`FloorVerdict` per invariant it was able to evaluate,
            keyed by :attr:`FloorVerdict.invariant_id`. May be a strict subset
            of *invariants* — see the class docstring.
        """
        ...  # pragma: no cover


# ---------------------------------------------------------------------------
# Coverage gate (fail-loud)
# ---------------------------------------------------------------------------


class FloorCoverageError(Exception):
    """Raised when one or more present Floor invariants were never evaluated.

    Mirrors ``SegmentationQAError``/``ScopeJudge``'s fail-loud contract: there
    is no fallback and no silent default verdict. An unevaluated Floor
    invariant is exactly the failure mode the Floor exists to prevent (a red
    line silently not checked), so :func:`floor_coverage_gate` raises rather
    than returning a partial report.
    """


@dataclass(frozen=True)
class FloorCoverageReport:
    """Coverage-gate result: every present invariant was evaluated.

    Attributes:
        invariants_total: Count of invariants the gate was asked to cover
                          (``0`` for the "no Floor" case).
        verdicts:         One :class:`FloorVerdict` per invariant, in the same
                          order as the invariants passed to the gate.
        violations:       The subset of ``verdicts`` whose ``verdict ==
                          "violation"`` — the caller's escalation/block signal.
    """

    invariants_total: int
    verdicts: tuple[FloorVerdict, ...]
    violations: tuple[FloorVerdict, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "invariants_total": self.invariants_total,
            "verdicts": [v.to_dict() for v in self.verdicts],
            "violations": [v.to_dict() for v in self.violations],
        }


def floor_coverage_gate(
    invariants: Sequence[FloorInvariant],
    judge: FloorJudge,
    clauses: Sequence[dict[str, Any]],
) -> FloorCoverageReport:
    """Evaluate every present Floor invariant via *judge* and enforce full coverage.

    Zero invariants (the corpus-mined default — Floor invariants are OPTIONAL
    per #145's Q4 direction) trivially passes: the judge is never called and an
    empty report is returned. This is what lets a corpus-mined playbook with no
    invariants compile cleanly.

    Otherwise, calls ``judge.evaluate_batch(invariants, clauses)`` once and
    cross-checks the returned verdicts against the input invariants by id. Any
    invariant with no matching verdict — whether because the judge silently
    dropped it, returned a verdict under the wrong id, or raised — is a
    coverage failure and raises :class:`FloorCoverageError` naming every
    missing invariant. There is no partial-success return value: a caller
    either gets full coverage or an exception.

    Args:
        invariants: The Floor's present invariants. Empty means "no Floor" —
                    see above.
        judge:      :class:`FloorJudge` instance.
        clauses:    In-scope clauses of the document under review, passed
                    through to the judge unmodified.

    Returns:
        :class:`FloorCoverageReport` — includes every verdict (in invariant
        order) and the subset that are violations.

    Raises:
        FloorCoverageError: if the judge raises, or if any present invariant
            has no corresponding verdict in the judge's return value.
    """
    if not invariants:
        return FloorCoverageReport(invariants_total=0, verdicts=(), violations=())

    try:
        verdicts = judge.evaluate_batch(invariants, clauses)
    except Exception as exc:  # noqa: BLE001
        raise FloorCoverageError(
            f"FloorJudge.evaluate_batch raised {type(exc).__name__}: {exc}. "
            f"{len(invariants)} invariant(s) went unevaluated."
        ) from exc

    # Last-writer-wins on a duplicate id: a judge returning two verdicts for
    # the same invariant is a contract bug, but it does not itself leave any
    # invariant *uncovered* — the coverage gate's sole job — so this is
    # generous rather than raising.
    verdicts_by_id: dict[str, FloorVerdict] = {v.invariant_id: v for v in verdicts}

    missing = [inv.id for inv in invariants if inv.id not in verdicts_by_id]
    if missing:
        raise FloorCoverageError(
            f"{len(missing)} Floor invariant(s) went unevaluated: {missing!r}. "
            "Every present invariant must be evaluated and its verdict/rationale "
            "logged before the Floor coverage gate can pass (issue #145)."
        )

    ordered_verdicts = tuple(verdicts_by_id[inv.id] for inv in invariants)
    violations = tuple(v for v in ordered_verdicts if v.verdict == "violation")
    return FloorCoverageReport(
        invariants_total=len(invariants),
        verdicts=ordered_verdicts,
        violations=violations,
    )
