"""Judgment eval harness — offline scaffolding for #151/#152 (issue #158).

Builds the STRUCTURE of a live-eval harness for the two judgment passes that
are otherwise verified only with fake judges: the Floor detector
(:mod:`playbook_engine.floor_judge`, issue #145/#151) and the redaction
semantic-residue pass (:mod:`playbook_engine.export_profile`, issue
#146/#152). Per the issue #158 direction (2026-07-10, settled):

  - This module loads LABELED fixtures — a clause set + invariant paired
    with the expected Floor verdict, or free text paired with the expected
    residue flag.
  - It runs a PLUGGABLE judge over those fixtures — a fake/canned judge in
    tests (offline, deterministic), a real LLM-backed judge later (out of
    scope here).
  - It scores precision/recall against the expected labels.

The live run against a real model, and authoring the real invariant/leak
corpora, stay a human/live task (#151/#152 remain open for that). This
module is the mechanism only, verified offline — same scope boundary as
``floor_judge`` and ``export_profile`` before it.

Positive class convention: for both domains, "positive" means the judge
FOUND the thing it exists to catch — ``verdict == "violation"`` for Floor,
``has_residue is True`` for redaction. Scoring is otherwise domain-agnostic
(:func:`score_binary`), so both harnesses share one confusion-matrix/
precision/recall implementation.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from playbook_engine.export_profile import RedactionJudge, TextSample
from playbook_engine.floor_judge import FloorInvariant, FloorJudge, floor_coverage_gate

__all__ = [
    "EvalHarnessError",
    "EvalScore",
    "FloorFixture",
    "RedactionFixture",
    "load_floor_fixtures",
    "load_redaction_fixtures",
    "run_floor_eval",
    "run_redaction_eval",
    "score_binary",
]


class EvalHarnessError(Exception):
    """Raised on a harness CONTRACT break: a duplicate fixture id, or a judge
    that left a fixture unevaluated. Never raised for a "wrong verdict" —
    that is exactly what the precision/recall score exists to surface, not
    an error condition.
    """


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FloorFixture:
    """One labeled Floor-detection case: an invariant judged over a clause set.

    Attributes:
        id: Stable fixture identifier — surfaced in :attr:`EvalScore.misses`
            when the judge's verdict disagrees with ``expect_violation``.
        invariant: The Floor invariant under test (same shape
                   :func:`playbook_engine.floor_judge.floor_coverage_gate`
                   takes).
        clauses: In-scope clauses to judge the invariant against — minimal
                 dicts, same shape as ``FloorJudge.evaluate_batch``'s
                 ``clauses`` argument.
        expect_violation: ``True`` if a correctly-functioning judge should
                 return ``verdict="violation"`` for this invariant/clause
                 pairing; ``False`` if it should return ``"clear"`` (or
                 ``"needs_review"`` — see :func:`run_floor_eval`, which
                 treats anything other than ``"violation"`` as the negative
                 class for scoring purposes).
    """

    id: str
    invariant: FloorInvariant
    clauses: tuple[dict[str, Any], ...]
    expect_violation: bool

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("FloorFixture.id must not be empty")


@dataclass(frozen=True)
class RedactionFixture:
    """One labeled redaction-residue case: free text judged for semantic residue.

    Attributes:
        id: Stable fixture identifier. Also used as the
            :attr:`playbook_engine.export_profile.TextSample.path` handed to
            the judge, so it must be unique within a fixture set (see
            :func:`load_redaction_fixtures`).
        text: The free text to judge (mirrors ``TextSample.text``).
        expect_residue: ``True`` if a correctly-functioning judge should flag
                 this text as still identifying the counterparty
                 (``has_residue=True``); ``False`` if the text should be
                 judged clean.
    """

    id: str
    text: str
    expect_residue: bool

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("RedactionFixture.id must not be empty")


def _load_json_array(path: str | Path) -> list[dict[str, Any]]:
    data = json.loads(Path(path).read_text())
    if not isinstance(data, list):
        raise EvalHarnessError(f"fixture file {path!s} must contain a JSON array of fixtures")
    return data


def load_floor_fixtures(path: str | Path) -> list[FloorFixture]:
    """Load labeled Floor-detection fixtures from a JSON file.

    Expected shape — a JSON array of objects::

        [
          {
            "id": "cap-1",
            "invariant": {"id": "no-uncapped-liability",
                          "statement": "Never accept uncapped liability."},
            "clauses": [{"clause_id": "3.2", "text": "Liability is uncapped..."}],
            "expect_violation": true
          },
          ...
        ]

    Raises:
        EvalHarnessError: if two fixtures share an ``id``.
    """
    fixtures: list[FloorFixture] = []
    seen_ids: set[str] = set()
    for item in _load_json_array(path):
        fid = item["id"]
        if fid in seen_ids:
            raise EvalHarnessError(f"duplicate Floor fixture id: {fid!r} in {path!s}")
        seen_ids.add(fid)
        fixtures.append(
            FloorFixture(
                id=fid,
                invariant=FloorInvariant.from_dict(item["invariant"]),
                clauses=tuple(item.get("clauses", [])),
                expect_violation=bool(item["expect_violation"]),
            )
        )
    return fixtures


def load_redaction_fixtures(path: str | Path) -> list[RedactionFixture]:
    """Load labeled redaction-residue fixtures from a JSON file.

    Expected shape — a JSON array of objects::

        [
          {"id": "r1", "text": "the large southeastern teaching hospital",
           "expect_residue": true},
          {"id": "r2", "text": "Party B agrees that...", "expect_residue": false}
        ]

    Raises:
        EvalHarnessError: if two fixtures share an ``id``.
    """
    fixtures: list[RedactionFixture] = []
    seen_ids: set[str] = set()
    for item in _load_json_array(path):
        fid = item["id"]
        if fid in seen_ids:
            raise EvalHarnessError(f"duplicate redaction fixture id: {fid!r} in {path!s}")
        seen_ids.add(fid)
        fixtures.append(
            RedactionFixture(
                id=fid,
                text=item["text"],
                expect_residue=bool(item["expect_residue"]),
            )
        )
    return fixtures


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvalScore:
    """Precision/recall of a judge's predictions against labeled fixtures.

    Attributes:
        domain: ``"floor"`` or ``"redaction"`` — which harness produced this
                score.
        total: Number of fixtures scored.
        true_positives / false_positives / false_negatives / true_negatives:
                Confusion-matrix counts (``expected`` vs. ``predicted``
                positive class — see the module docstring).
        precision: ``tp / (tp + fp)``, or ``1.0`` (vacuously correct) when
                the judge predicted no positives at all.
        recall: ``tp / (tp + fn)``, or ``1.0`` (vacuously correct) when no
                fixture was labeled positive.
        misses: Ids of fixtures where ``predicted != expected`` (both false
                positives and false negatives), in fixture order — the
                harness's primary debugging output.
    """

    domain: str
    total: int
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int
    precision: float
    recall: float
    misses: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "total": self.total,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "true_negatives": self.true_negatives,
            "precision": self.precision,
            "recall": self.recall,
            "misses": list(self.misses),
        }


def score_binary(
    domain: str,
    ids: Sequence[str],
    expected: Sequence[bool],
    predicted: Sequence[bool],
) -> EvalScore:
    """Score *predicted* against *expected* labels and compute precision/recall.

    Domain-agnostic — both :func:`run_floor_eval` and :func:`run_redaction_eval`
    reduce their judge's output to parallel ``expected``/``predicted`` boolean
    sequences (aligned by ``ids``) and delegate here.

    Args:
        domain: Label carried through onto :attr:`EvalScore.domain`.
        ids: Fixture ids, same length and order as *expected*/*predicted*.
        expected: The labeled ground truth, one bool per fixture.
        predicted: The judge's verdict reduced to a bool, one per fixture.

    Raises:
        ValueError: if the three sequences are not the same length.
    """
    if not (len(ids) == len(expected) == len(predicted)):
        raise ValueError(
            f"score_binary: ids/expected/predicted must be the same length; got "
            f"{len(ids)}, {len(expected)}, {len(predicted)}"
        )

    tp = fp = fn = tn = 0
    misses: list[str] = []
    for fid, exp, pred in zip(ids, expected, predicted, strict=True):
        if exp and pred:
            tp += 1
        elif exp and not pred:
            fn += 1
            misses.append(fid)
        elif not exp and pred:
            fp += 1
            misses.append(fid)
        else:
            tn += 1

    precision = 1.0 if (tp + fp) == 0 else tp / (tp + fp)
    recall = 1.0 if (tp + fn) == 0 else tp / (tp + fn)

    return EvalScore(
        domain=domain,
        total=len(ids),
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        true_negatives=tn,
        precision=precision,
        recall=recall,
        misses=tuple(misses),
    )


# ---------------------------------------------------------------------------
# Runners — pluggable judge over labeled fixtures
# ---------------------------------------------------------------------------


def run_floor_eval(fixtures: Sequence[FloorFixture], judge: FloorJudge) -> EvalScore:
    """Run *judge* over each :class:`FloorFixture` and score precision/recall.

    Each fixture is evaluated in its own :func:`floor_coverage_gate` call
    (a single invariant against that fixture's clause set) so one fixture's
    clauses never leak into another's judgment. A ``"needs_review"`` verdict
    counts as the negative (non-violation) class for scoring — it is not a
    confident detection, so it cannot count as a true positive.

    Args:
        fixtures: Labeled Floor-detection cases.
        judge: The judge under evaluation (fake in tests; real LLM later).

    Returns:
        :class:`EvalScore` with ``domain="floor"``.

    Raises:
        FloorCoverageError: propagated uncaught if *judge* breaks its
            coverage contract (raises, or drops the fixture's invariant) —
            same fail-loud rule as everywhere else this judge protocol is
            used; a fixture that was never evaluated must not be silently
            scored as correct or incorrect.
    """
    ids: list[str] = []
    expected: list[bool] = []
    predicted: list[bool] = []
    for fx in fixtures:
        report = floor_coverage_gate(
            invariants=[fx.invariant], judge=judge, clauses=list(fx.clauses)
        )
        verdict = report.verdicts[0]
        ids.append(fx.id)
        expected.append(fx.expect_violation)
        predicted.append(verdict.verdict == "violation")
    return score_binary("floor", ids, expected, predicted)


def run_redaction_eval(fixtures: Sequence[RedactionFixture], judge: RedactionJudge) -> EvalScore:
    """Run *judge* over every :class:`RedactionFixture` and score precision/recall.

    All fixtures are judged in a single ``evaluate_batch`` call (each
    fixture's ``id`` doubles as its ``TextSample.path``), then every
    fixture's ``id`` is cross-checked against the returned findings —
    mirroring :mod:`playbook_engine.export_profile`'s coverage contract.

    Args:
        fixtures: Labeled redaction-residue cases.
        judge: The judge under evaluation (fake in tests; real LLM later).

    Returns:
        :class:`EvalScore` with ``domain="redaction"``.

    Raises:
        EvalHarnessError: if *judge* raises, or leaves any fixture
            unevaluated (returns a finding for a strict subset of the
            samples it was handed).
    """
    samples = [TextSample(path=fx.id, text=fx.text) for fx in fixtures]
    try:
        findings = judge.evaluate_batch(samples)
    except Exception as exc:  # noqa: BLE001
        raise EvalHarnessError(
            f"RedactionJudge raised {type(exc).__name__}: {exc}. "
            f"{len(fixtures)} fixture(s) went unevaluated."
        ) from exc

    findings_by_path = {f.path: f for f in findings}
    missing = [fx.id for fx in fixtures if fx.id not in findings_by_path]
    if missing:
        raise EvalHarnessError(
            f"RedactionJudge left {len(missing)} fixture(s) unevaluated: {missing!r}."
        )

    ids = [fx.id for fx in fixtures]
    expected = [fx.expect_residue for fx in fixtures]
    predicted = [findings_by_path[fx.id].has_residue for fx in fixtures]
    return score_binary("redaction", ids, expected, predicted)
