"""Tests for playbook_engine.eval_harness (issue #158).

Verified entirely offline with FAKE ``FloorJudge`` / ``RedactionJudge``
implementations — no LLM, no network. Covers the required scenario: the
harness scores a set of fixtures against a fake judge deterministically —
precision/recall computed correctly, and a planted miss lowers the score —
for both the Floor-detection and redaction-residue domains, plus the
JSON fixture loaders.

SECURITY NOTE: All fixtures use synthetic text and fictional party names
only. No real agreement text or real document paths are used.
"""

from __future__ import annotations

import json

import pytest

from playbook_engine.eval_harness import (
    EvalHarnessError,
    FloorFixture,
    RedactionFixture,
    load_floor_fixtures,
    load_redaction_fixtures,
    run_floor_eval,
    run_redaction_eval,
    score_binary,
)
from playbook_engine.export_profile import RedactionFinding
from playbook_engine.floor_judge import FloorCoverageError, FloorInvariant, FloorVerdict

# ---------------------------------------------------------------------------
# score_binary — domain-agnostic scoring
# ---------------------------------------------------------------------------


def test_score_binary_perfect_predictions() -> None:
    score = score_binary(
        "floor",
        ids=["a", "b", "c"],
        expected=[True, False, True],
        predicted=[True, False, True],
    )
    assert score.total == 3
    assert score.true_positives == 2
    assert score.true_negatives == 1
    assert score.false_positives == 0
    assert score.false_negatives == 0
    assert score.precision == 1.0
    assert score.recall == 1.0
    assert score.misses == ()


def test_score_binary_planted_miss_lowers_recall_and_precision() -> None:
    """A single false negative (planted miss) must show up in the score."""
    perfect = score_binary(
        "floor",
        ids=["a", "b", "c", "d"],
        expected=[True, True, False, False],
        predicted=[True, True, False, False],
    )
    assert perfect.recall == 1.0
    assert perfect.precision == 1.0

    # Plant a miss: fixture "b" (expected violation) is now predicted clear.
    with_miss = score_binary(
        "floor",
        ids=["a", "b", "c", "d"],
        expected=[True, True, False, False],
        predicted=[True, False, False, False],
    )
    assert with_miss.true_positives == 1
    assert with_miss.false_negatives == 1
    assert with_miss.recall == pytest.approx(0.5)
    assert with_miss.precision == 1.0  # no false positives introduced
    assert with_miss.misses == ("b",)
    assert with_miss.recall < perfect.recall


def test_score_binary_false_positive_lowers_precision() -> None:
    score = score_binary(
        "redaction",
        ids=["a", "b"],
        expected=[False, False],
        predicted=[True, False],
    )
    assert score.false_positives == 1
    assert score.precision == 0.0
    assert score.recall == 1.0  # vacuous: no positives were expected
    assert score.misses == ("a",)


def test_score_binary_vacuous_precision_when_no_positives_predicted() -> None:
    score = score_binary("floor", ids=["a"], expected=[False], predicted=[False])
    assert score.precision == 1.0
    assert score.recall == 1.0


def test_score_binary_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError):
        score_binary("floor", ids=["a", "b"], expected=[True], predicted=[True, False])


def test_score_binary_to_dict_roundtrip() -> None:
    score = score_binary("floor", ids=["a"], expected=[True], predicted=[True])
    d = score.to_dict()
    assert d["domain"] == "floor"
    assert d["true_positives"] == 1
    assert d["misses"] == []


# ---------------------------------------------------------------------------
# Fake judges
# ---------------------------------------------------------------------------


class _FakeFloorJudge:
    """Fake FloorJudge: returns a pre-canned verdict for every invariant it knows."""

    def __init__(self, verdicts_by_id: dict[str, FloorVerdict]) -> None:
        self._verdicts_by_id = verdicts_by_id

    def evaluate_batch(self, invariants, clauses):  # noqa: ANN001
        return [
            self._verdicts_by_id[inv.id] for inv in invariants if inv.id in self._verdicts_by_id
        ]


class _FakeRedactionJudge:
    """Fake RedactionJudge: returns a pre-canned finding for every path it knows."""

    def __init__(self, findings_by_path: dict[str, RedactionFinding]) -> None:
        self._findings_by_path = findings_by_path

    def evaluate_batch(self, samples):  # noqa: ANN001
        return [self._findings_by_path[s.path] for s in samples if s.path in self._findings_by_path]


class _RaisingRedactionJudge:
    def evaluate_batch(self, samples):  # noqa: ANN001
        raise RuntimeError("LLM timeout")


def _clear(invariant_id: str) -> FloorVerdict:
    return FloorVerdict(invariant_id=invariant_id, verdict="clear", rationale="No violation found.")


def _violation(invariant_id: str) -> FloorVerdict:
    return FloorVerdict(
        invariant_id=invariant_id,
        verdict="violation",
        rationale="Clause imposes uncapped liability.",
        citation={"clause_id": "3.2", "document_id": "doc-1"},
    )


def _clean(path: str) -> RedactionFinding:
    return RedactionFinding(path=path, has_residue=False, rationale="No identifying detail.")


def _residue(path: str) -> RedactionFinding:
    return RedactionFinding(
        path=path,
        has_residue=True,
        rationale="Still identifies the counterparty by description.",
        rewritten_text="[REDACTED]",
    )


# ---------------------------------------------------------------------------
# run_floor_eval
# ---------------------------------------------------------------------------


def _floor_fixture(fid: str, expect_violation: bool) -> FloorFixture:
    return FloorFixture(
        id=fid,
        invariant=FloorInvariant(id=fid, statement="Never accept uncapped liability."),
        clauses=({"clause_id": "3.2", "text": "Liability shall be uncapped."},),
        expect_violation=expect_violation,
    )


def test_run_floor_eval_perfect_judge_scores_perfectly() -> None:
    fixtures = [_floor_fixture("cap-1", True), _floor_fixture("cap-2", False)]
    judge = _FakeFloorJudge({"cap-1": _violation("cap-1"), "cap-2": _clear("cap-2")})

    score = run_floor_eval(fixtures, judge)

    assert score.domain == "floor"
    assert score.total == 2
    assert score.true_positives == 1
    assert score.true_negatives == 1
    assert score.precision == 1.0
    assert score.recall == 1.0
    assert score.misses == ()


def test_run_floor_eval_planted_miss_lowers_score() -> None:
    """Ground truth expects a violation on cap-1; fake judge wrongly says clear."""
    fixtures = [_floor_fixture("cap-1", True), _floor_fixture("cap-2", False)]
    judge = _FakeFloorJudge({"cap-1": _clear("cap-1"), "cap-2": _clear("cap-2")})

    score = run_floor_eval(fixtures, judge)

    assert score.true_positives == 0
    assert score.false_negatives == 1
    assert score.recall == 0.0
    assert score.misses == ("cap-1",)


def test_run_floor_eval_needs_review_counts_as_negative() -> None:
    fixtures = [_floor_fixture("cap-1", True)]
    verdict = FloorVerdict(invariant_id="cap-1", verdict="needs_review", rationale="Ambiguous.")
    judge = _FakeFloorJudge({"cap-1": verdict})

    score = run_floor_eval(fixtures, judge)

    assert score.true_positives == 0
    assert score.false_negatives == 1


def test_run_floor_eval_propagates_coverage_error() -> None:
    """A judge that drops a fixture's invariant fails loud, not a silent miss."""
    fixtures = [_floor_fixture("cap-1", True)]
    judge = _FakeFloorJudge({})  # never evaluates anything

    with pytest.raises(FloorCoverageError):
        run_floor_eval(fixtures, judge)


# ---------------------------------------------------------------------------
# run_redaction_eval
# ---------------------------------------------------------------------------


def test_run_redaction_eval_perfect_judge_scores_perfectly() -> None:
    fixtures = [
        RedactionFixture(
            id="r1", text="the large southeastern teaching hospital", expect_residue=True
        ),
        RedactionFixture(id="r2", text="Party B agrees to the terms.", expect_residue=False),
    ]
    judge = _FakeRedactionJudge({"r1": _residue("r1"), "r2": _clean("r2")})

    score = run_redaction_eval(fixtures, judge)

    assert score.domain == "redaction"
    assert score.total == 2
    assert score.true_positives == 1
    assert score.true_negatives == 1
    assert score.precision == 1.0
    assert score.recall == 1.0
    assert score.misses == ()


def test_run_redaction_eval_planted_miss_lowers_score() -> None:
    """Ground truth expects residue on r1; fake judge wrongly says clean."""
    fixtures = [
        RedactionFixture(
            id="r1", text="the large southeastern teaching hospital", expect_residue=True
        ),
        RedactionFixture(id="r2", text="Party B agrees to the terms.", expect_residue=False),
    ]
    judge = _FakeRedactionJudge({"r1": _clean("r1"), "r2": _clean("r2")})

    score = run_redaction_eval(fixtures, judge)

    assert score.true_positives == 0
    assert score.false_negatives == 1
    assert score.recall == 0.0
    assert score.misses == ("r1",)


def test_run_redaction_eval_missing_finding_raises() -> None:
    fixtures = [RedactionFixture(id="r1", text="text", expect_residue=False)]
    judge = _FakeRedactionJudge({})  # returns nothing

    with pytest.raises(EvalHarnessError, match="r1"):
        run_redaction_eval(fixtures, judge)


def test_run_redaction_eval_judge_exception_raises() -> None:
    fixtures = [RedactionFixture(id="r1", text="text", expect_residue=False)]

    with pytest.raises(EvalHarnessError, match="LLM timeout"):
        run_redaction_eval(fixtures, _RaisingRedactionJudge())


# ---------------------------------------------------------------------------
# Fixture loaders
# ---------------------------------------------------------------------------


def test_load_floor_fixtures_roundtrip(tmp_path) -> None:
    path = tmp_path / "floor_fixtures.json"
    path.write_text(
        json.dumps(
            [
                {
                    "id": "cap-1",
                    "invariant": {
                        "id": "no-uncapped-liability",
                        "statement": "Never accept uncapped liability.",
                        "rationale": "Admission-test failure.",
                    },
                    "clauses": [{"clause_id": "3.2", "text": "Liability shall be uncapped."}],
                    "expect_violation": True,
                }
            ]
        )
    )

    fixtures = load_floor_fixtures(path)

    assert len(fixtures) == 1
    assert fixtures[0].id == "cap-1"
    assert fixtures[0].invariant.id == "no-uncapped-liability"
    assert fixtures[0].invariant.rationale == "Admission-test failure."
    assert fixtures[0].clauses == ({"clause_id": "3.2", "text": "Liability shall be uncapped."},)
    assert fixtures[0].expect_violation is True


def test_load_floor_fixtures_rejects_duplicate_ids(tmp_path) -> None:
    path = tmp_path / "floor_fixtures.json"
    path.write_text(
        json.dumps(
            [
                {
                    "id": "dup",
                    "invariant": {"id": "x", "statement": "Never do X."},
                    "clauses": [],
                    "expect_violation": True,
                },
                {
                    "id": "dup",
                    "invariant": {"id": "y", "statement": "Never do Y."},
                    "clauses": [],
                    "expect_violation": False,
                },
            ]
        )
    )

    with pytest.raises(EvalHarnessError, match="dup"):
        load_floor_fixtures(path)


def test_load_redaction_fixtures_roundtrip(tmp_path) -> None:
    path = tmp_path / "redaction_fixtures.json"
    path.write_text(
        json.dumps(
            [
                {
                    "id": "r1",
                    "text": "the large southeastern teaching hospital",
                    "expect_residue": True,
                },
                {"id": "r2", "text": "Party B agrees.", "expect_residue": False},
            ]
        )
    )

    fixtures = load_redaction_fixtures(path)

    assert len(fixtures) == 2
    assert fixtures[0] == RedactionFixture(
        id="r1", text="the large southeastern teaching hospital", expect_residue=True
    )
    assert fixtures[1].expect_residue is False


def test_load_redaction_fixtures_rejects_duplicate_ids(tmp_path) -> None:
    path = tmp_path / "redaction_fixtures.json"
    path.write_text(
        json.dumps(
            [
                {"id": "dup", "text": "a", "expect_residue": True},
                {"id": "dup", "text": "b", "expect_residue": False},
            ]
        )
    )

    with pytest.raises(EvalHarnessError, match="dup"):
        load_redaction_fixtures(path)


# ---------------------------------------------------------------------------
# End-to-end: load fixtures from disk, run against a fake judge, score
# ---------------------------------------------------------------------------


def test_end_to_end_floor_fixtures_from_disk(tmp_path) -> None:
    path = tmp_path / "floor_fixtures.json"
    path.write_text(
        json.dumps(
            [
                {
                    "id": "cap-1",
                    "invariant": {"id": "cap-1", "statement": "Never accept uncapped liability."},
                    "clauses": [{"clause_id": "3.2", "text": "Liability shall be uncapped."}],
                    "expect_violation": True,
                },
                {
                    "id": "cap-2",
                    "invariant": {"id": "cap-2", "statement": "Never accept uncapped liability."},
                    "clauses": [{"clause_id": "4.1", "text": "Liability is capped at fees paid."}],
                    "expect_violation": False,
                },
            ]
        )
    )
    fixtures = load_floor_fixtures(path)
    judge = _FakeFloorJudge({"cap-1": _violation("cap-1"), "cap-2": _clear("cap-2")})

    score = run_floor_eval(fixtures, judge)

    assert score.precision == 1.0
    assert score.recall == 1.0
