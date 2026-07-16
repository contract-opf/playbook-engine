"""Tests for playbook_engine.floor_judge (issue #145).

Verified entirely offline with a fake ``FloorJudge`` — no LLM, no network.
Covers the three required scenarios: (a) zero invariants compiles cleanly,
(b) an unevaluated invariant fails the coverage gate loud, and (c) a
"violation" verdict is surfaced with its citation/rationale.
"""

from __future__ import annotations

import pytest

from playbook_engine.floor_judge import (
    FloorCoverageError,
    FloorInvariant,
    FloorVerdict,
    floor_coverage_gate,
)

# ---------------------------------------------------------------------------
# Fake judge
# ---------------------------------------------------------------------------


class _FakeFloorJudge:
    """Fake ``FloorJudge``: returns pre-canned verdicts keyed by invariant id.

    Invariants absent from ``verdicts_by_id`` are silently NOT returned by
    ``evaluate_batch`` — simulating a judge that dropped/never evaluated them,
    which is exactly the gap ``floor_coverage_gate`` must catch.
    """

    def __init__(self, verdicts_by_id: dict[str, FloorVerdict]) -> None:
        self._verdicts_by_id = verdicts_by_id

    def evaluate_batch(self, invariants, clauses):  # noqa: ANN001
        return [
            self._verdicts_by_id[inv.id] for inv in invariants if inv.id in self._verdicts_by_id
        ]


class _RaisingFloorJudge:
    """Fake judge that always raises — simulates an LLM timeout/error."""

    def evaluate_batch(self, invariants, clauses):  # noqa: ANN001
        raise RuntimeError("LLM timeout")


def _clear(invariant_id: str) -> FloorVerdict:
    return FloorVerdict(
        invariant_id=invariant_id, verdict="clear", rationale="No violation found in-scope."
    )


def _violation(invariant_id: str) -> FloorVerdict:
    return FloorVerdict(
        invariant_id=invariant_id,
        verdict="violation",
        rationale="Clause 3.2 imposes uncapped liability, contradicting the invariant.",
        citation={"clause_id": "3.2", "document_id": "doc-1"},
    )


# ---------------------------------------------------------------------------
# (a) zero invariants -> playbook still compiles
# ---------------------------------------------------------------------------


def test_zero_invariants_compiles_cleanly() -> None:
    """Zero invariants trivially passes: the judge is never even called."""

    class _NeverCallJudge:
        def evaluate_batch(self, invariants, clauses):  # noqa: ANN001
            raise AssertionError("judge must not be called with zero invariants")

    report = floor_coverage_gate(invariants=[], judge=_NeverCallJudge(), clauses=[])

    assert report.invariants_total == 0
    assert report.verdicts == ()
    assert report.violations == ()


# ---------------------------------------------------------------------------
# (b) a present invariant left unevaluated -> coverage gate fails loud
# ---------------------------------------------------------------------------


def test_unevaluated_invariant_fails_coverage_gate_loud() -> None:
    invariants = [
        FloorInvariant(id="no-uncapped-liability", statement="Never accept uncapped liability."),
        FloorInvariant(
            id="no-unilateral-termination",
            statement="Never accept termination for convenience solely by the counterparty.",
        ),
    ]
    # Fake judge only evaluates the first invariant -> second goes uncovered.
    judge = _FakeFloorJudge({"no-uncapped-liability": _clear("no-uncapped-liability")})

    with pytest.raises(FloorCoverageError) as exc_info:
        floor_coverage_gate(invariants=invariants, judge=judge, clauses=[])

    assert "no-unilateral-termination" in str(exc_info.value)


def test_all_invariants_unevaluated_fails_coverage_gate_loud() -> None:
    invariants = [FloorInvariant(id="x", statement="Never do X.")]
    judge = _FakeFloorJudge({})  # returns nothing at all

    with pytest.raises(FloorCoverageError) as exc_info:
        floor_coverage_gate(invariants=invariants, judge=judge, clauses=[])

    assert "x" in str(exc_info.value)


def test_judge_exception_fails_coverage_gate_loud() -> None:
    """A judge that raises is also a coverage failure, not a silent skip."""
    invariants = [FloorInvariant(id="x", statement="Never do X.")]

    with pytest.raises(FloorCoverageError) as exc_info:
        floor_coverage_gate(invariants=invariants, judge=_RaisingFloorJudge(), clauses=[])

    assert "LLM timeout" in str(exc_info.value)


# ---------------------------------------------------------------------------
# (c) a fake "violation" verdict is surfaced with its citation/rationale
# ---------------------------------------------------------------------------


def test_violation_verdict_surfaced_with_citation_and_rationale() -> None:
    invariants = [
        FloorInvariant(id="no-uncapped-liability", statement="Never accept uncapped liability.")
    ]
    judge = _FakeFloorJudge({"no-uncapped-liability": _violation("no-uncapped-liability")})

    report = floor_coverage_gate(
        invariants=invariants,
        judge=judge,
        clauses=[{"clause_id": "3.2", "text": "Liability shall be uncapped."}],
    )

    assert report.invariants_total == 1
    assert len(report.verdicts) == 1
    assert len(report.violations) == 1

    violation = report.violations[0]
    assert violation.invariant_id == "no-uncapped-liability"
    assert violation.verdict == "violation"
    assert violation.citation == {"clause_id": "3.2", "document_id": "doc-1"}
    assert "uncapped liability" in violation.rationale


def test_mixed_clear_and_violation_all_covered() -> None:
    """Multiple invariants, all evaluated: clear + violation both surfaced,
    and coverage still passes (a violation is not itself a coverage failure)."""
    invariants = [
        FloorInvariant(id="a", statement="Never accept X."),
        FloorInvariant(id="b", statement="Never accept Y."),
    ]
    judge = _FakeFloorJudge({"a": _clear("a"), "b": _violation("b")})

    report = floor_coverage_gate(invariants=invariants, judge=judge, clauses=[])

    assert report.invariants_total == 2
    assert len(report.verdicts) == 2
    assert len(report.violations) == 1
    assert report.violations[0].invariant_id == "b"


def test_needs_review_verdict_counts_as_covered() -> None:
    """A 'needs_review' verdict IS a logged verdict — it satisfies coverage,
    even though it is not a confident clear/violation determination."""
    invariants = [FloorInvariant(id="a", statement="Never accept X.")]
    verdict = FloorVerdict(
        invariant_id="a", verdict="needs_review", rationale="Ambiguous clause; escalate to human."
    )
    judge = _FakeFloorJudge({"a": verdict})

    report = floor_coverage_gate(invariants=invariants, judge=judge, clauses=[])

    assert report.invariants_total == 1
    assert report.verdicts == (verdict,)
    assert report.violations == ()


# ---------------------------------------------------------------------------
# FloorInvariant / FloorVerdict validation
# ---------------------------------------------------------------------------


def test_floor_invariant_requires_non_empty_id() -> None:
    with pytest.raises(ValueError):
        FloorInvariant(id="  ", statement="Never do X.")


def test_floor_invariant_requires_non_empty_statement() -> None:
    with pytest.raises(ValueError):
        FloorInvariant(id="x", statement="")


def test_floor_invariant_round_trips_to_dict() -> None:
    inv = FloorInvariant(id="x", statement="Never do X.", rationale="Because Y.")
    d = inv.to_dict()
    assert d == {"id": "x", "statement": "Never do X.", "rationale": "Because Y."}
    assert FloorInvariant.from_dict(d) == inv


def test_floor_invariant_to_dict_omits_empty_rationale() -> None:
    inv = FloorInvariant(id="x", statement="Never do X.")
    assert inv.to_dict() == {"id": "x", "statement": "Never do X."}


def test_floor_verdict_rejects_unknown_verdict_value() -> None:
    with pytest.raises(ValueError):
        FloorVerdict(invariant_id="x", verdict="maybe", rationale="...")


def test_floor_verdict_rejects_unknown_basis() -> None:
    with pytest.raises(ValueError):
        FloorVerdict(invariant_id="x", verdict="clear", rationale="...", basis="magic")


def test_floor_verdict_violation_requires_citation() -> None:
    with pytest.raises(ValueError):
        FloorVerdict(invariant_id="x", verdict="violation", rationale="Found a violation.")


def test_floor_verdict_requires_non_empty_rationale() -> None:
    with pytest.raises(ValueError):
        FloorVerdict(invariant_id="x", verdict="clear", rationale="  ")


def test_floor_coverage_report_to_dict() -> None:
    invariants = [FloorInvariant(id="a", statement="Never accept X.")]
    judge = _FakeFloorJudge({"a": _violation("a")})
    report = floor_coverage_gate(invariants=invariants, judge=judge, clauses=[])

    d = report.to_dict()
    assert d["invariants_total"] == 1
    assert len(d["verdicts"]) == 1
    assert len(d["violations"]) == 1
    assert d["violations"][0]["invariant_id"] == "a"
