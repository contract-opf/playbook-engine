"""Blind check and adjudication of ``subsplit`` verdicts (issue #257).

A sub-split puts a clause boundary into a deal's text, and a false boundary is
worse than a mixed node, so a drafted verdict is applied only once it has been
checked, the way a ``prior_form_origin`` verdict counts
(:mod:`playbook_engine.prior_form_origin_check`):

1. **Draft.** A coder, or an agent acting as the judge, drafts ``{split,
   spans, reason, basis}`` into the verdict store (``judge-apply
   --verdicts``).
2. **Blind check.** ``playbook judge --check subsplit <out>`` writes
   ``judge/subsplit-check-pending.jsonl``: one item per drafted, unchecked
   verdict with exactly the question the drafter saw (from
   ``judge/subsplit-subjects.jsonl``, written by the last mine) and NOT the
   draft. A separate agent pinned to
   :data:`~playbook_engine.equivalence_check.CHECKER_MODEL` at
   :data:`~playbook_engine.equivalence_check.CHECKER_EFFORT` answers ``{split,
   spans, reason}``; ``playbook judge-apply --check`` records it with the
   checker's model id and effort and rejects any other pair unless
   ``--allow-checker-model`` is passed.
3. **Agreement** is the same answer: the same split decision and, for a
   split, the same span boundaries and the same ids
   (:func:`~playbook_engine.subsplit.agreement_signature`). It keeps the
   verdict with ``check: {agreed: true, by, adjudicated: false}``.
   **Disagreement** records ``agreed: false`` and puts the item on
   ``judge/subsplit-adjudication-pending.jsonl``; a fresh pinned agent sees
   both answers and its answer replaces the verdict with ``adjudicated: true``.
4. **Applying.** Only an owner's, an agreed or an adjudicated verdict is
   applied by the re-segmentation stage
   (:func:`~playbook_engine.subsplit.is_settled`); an unchecked or disputed
   one changes nothing.

Every answer is grounded against the question exactly as a draft is
(:func:`~playbook_engine.subsplit.ground`). The store keeps the whole trail
(``draft``, ``checker``, ``adjudicator``). There is no API path for this kind:
answer the queues with a pinned agent.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from playbook_engine.agent_judge import VerdictStore
from playbook_engine.equivalence_check import CHECKER_EFFORT, CHECKER_MODEL
from playbook_engine.segmentation_grounding import GroundingError
from playbook_engine.subsplit import (
    SubsplitSubject,
    agreement_signature,
    ground,
    validate_replayable,
)

__all__ = [
    "ADJUDICATION_KIND",
    "ADJUDICATION_PENDING",
    "CHECK_KIND",
    "CHECK_PENDING",
    "ApplyResult",
    "CheckQueues",
    "apply_check_records",
    "build_check_queues",
    "check_state",
    "load_check_records",
    "write_check_queues",
]

#: Queue file names under ``<out>/judge/``.
CHECK_PENDING = "subsplit-check-pending.jsonl"
ADJUDICATION_PENDING = "subsplit-adjudication-pending.jsonl"

#: Pending-item kinds of the two queues.
CHECK_KIND = "subsplit_check"
ADJUDICATION_KIND = "subsplit_adjudication"

_RECORD_KEYS = frozenset({"key", "split", "spans", "reason", "model", "effort"})


def check_state(verdict: Mapping[str, Any]) -> str:
    """``owner`` | ``unchecked`` | ``disputed`` | ``agreed`` | ``adjudicated``."""
    if verdict.get("basis") == "owner":
        return "owner"
    check = verdict.get("check")
    if not isinstance(check, Mapping):
        return "unchecked"
    if check.get("adjudicated") is True:
        return "adjudicated"
    return "agreed" if check.get("agreed") is True else "disputed"


def _stored(store: VerdictStore, key: str) -> dict[str, Any] | None:
    verdict = store.get_by_key(key)
    if not isinstance(verdict, dict):
        return None
    try:
        validate_replayable(verdict)
    except ValueError:
        return None
    return verdict


def _judgment(verdict: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"split": verdict.get("split")}
    if verdict.get("split") is True:
        out["spans"] = verdict.get("spans")
    out["reason"] = verdict.get("reason")
    return out


@dataclass
class CheckQueues:
    """The two queues one ``judge --check subsplit`` round emits."""

    check: list[dict[str, Any]] = field(default_factory=list)
    adjudication: list[dict[str, Any]] = field(default_factory=list)
    undrafted: int = 0
    settled: int = 0


def build_check_queues(store: VerdictStore, subjects: Iterable[SubsplitSubject]) -> CheckQueues:
    """Queue every drafted-but-unchecked verdict and every unadjudicated dispute."""
    queues = CheckQueues()
    for subject in subjects:
        verdict = _stored(store, subject.key)
        if verdict is None:
            queues.undrafted += 1
            continue
        state = check_state(verdict)
        if state == "unchecked":
            queues.check.append(
                {"key": subject.key, "kind": CHECK_KIND, "payload": dict(subject.payload)}
            )
        elif state == "disputed":
            draft = verdict.get("draft") or verdict
            checker = verdict.get("checker") or {}
            queues.adjudication.append(
                {
                    "key": subject.key,
                    "kind": ADJUDICATION_KIND,
                    "payload": {
                        **subject.payload,
                        "draft": _judgment(draft),
                        "check": _judgment(checker),
                    },
                }
            )
        else:
            queues.settled += 1
    return queues


def write_check_queues(judge_dir: Path, queues: CheckQueues) -> tuple[Path, Path]:
    """Write both queue files (always, so absence never means "not run")."""
    judge_dir.mkdir(parents=True, exist_ok=True)
    paths = (judge_dir / CHECK_PENDING, judge_dir / ADJUDICATION_PENDING)
    for path, items in zip(paths, (queues.check, queues.adjudication), strict=True):
        body = "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in items)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(body, encoding="utf-8")
        tmp.replace(path)
    return paths


def load_check_records(path: Path) -> list[tuple[int, dict[str, Any]]]:
    """Parse a ``--check`` file into ``(line_number, record)`` pairs.

    Raises ``ValueError`` (with the line number) on malformed JSON or a record
    that is not ``{key, split, spans, reason, model, effort}`` (``spans`` is
    ``[]`` for a no-split answer)."""
    records: list[tuple[int, dict[str, Any]]] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"line {lineno}: invalid JSON: {exc}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"line {lineno}: expected a JSON object")
        if set(record) != _RECORD_KEYS:
            raise ValueError(
                f"line {lineno}: a check record is {{key, split, spans, reason, model, effort}}"
                f"; missing {sorted(_RECORD_KEYS - set(record))}"
                f"; unexpected {sorted(set(record) - _RECORD_KEYS)}"
            )
        records.append((lineno, record))
    return records


def _validate_record(
    record: Mapping[str, Any], payload: Mapping[str, Any], *, allow_checker_model: bool
) -> dict[str, Any]:
    """The record's judgment, grounded against the question *payload*."""
    if not isinstance(record["key"], str) or not record["key"]:
        raise ValueError("'key' must be a non-empty string")
    if not isinstance(record["reason"], str) or not record["reason"].strip():
        raise ValueError("'reason' must be a non-empty string (one sentence)")
    model, effort = record["model"], record["effort"]
    if not isinstance(model, str) or not model or not isinstance(effort, str) or not effort:
        raise ValueError("'model' and 'effort' must be non-empty strings (the checker's own)")
    if not allow_checker_model:
        if model != CHECKER_MODEL:
            raise ValueError(
                f"checker model {model!r} is not {CHECKER_MODEL!r}; a check must be answered by "
                f"{CHECKER_MODEL} (pass --allow-checker-model to override explicitly)"
            )
        if effort != CHECKER_EFFORT:
            raise ValueError(
                f"checker effort {effort!r} is not {CHECKER_EFFORT!r}; a check must run at "
                f"{CHECKER_EFFORT} effort (pass --allow-checker-model to override explicitly)"
            )
    judgment: dict[str, Any] = {"split": record["split"], "spans": record["spans"]}
    try:
        ground(judgment, payload)
        signature = agreement_signature(judgment)
    except (GroundingError, ValueError) as exc:
        raise ValueError(f"grounding: {exc}") from exc
    if signature[0] is False:
        return {"split": False}
    return {
        "split": True,
        "spans": [{"start": s, "end": e, "taxonomy_id": t} for s, e, t in signature[1:]],
    }


@dataclass
class ApplyResult:
    """Counts of what a ``judge-apply --check`` file did to subsplit verdicts."""

    agreed: int = 0
    disagreed: int = 0
    adjudicated: int = 0
    skipped_owner: int = 0
    #: ``(key, new verdict)`` pairs to write, in order; at most one per key.
    updates: list[tuple[str, dict[str, Any]]] = field(default_factory=list)


def apply_check_records(
    store: VerdictStore,
    records: Iterable[tuple[int, dict[str, Any]]],
    subjects: Mapping[str, SubsplitSubject],
    *,
    allow_checker_model: bool = False,
) -> ApplyResult:
    """Validate every record, then compute the verdict updates (store untouched).

    *subjects* maps each key to its question (``subsplit.load_subjects``); an
    answer is grounded against it. The stage is inferred from the verdict
    stored BEFORE this file: no check yet is a blind-check answer, a recorded
    disagreement an adjudication; a key twice in one file is rejected. Raises
    ``ValueError`` naming the line of the first problem."""
    result = ApplyResult()
    first_line: dict[str, int] = {}
    for lineno, record in records:
        try:
            key = record["key"]
            subject = subjects.get(key) if isinstance(key, str) else None
            if subject is None:
                raise ValueError(
                    "no subsplit question under this key in judge/subsplit-subjects.jsonl — "
                    "re-run `playbook judge` (or mine) so the question is recorded"
                )
            judgment = _validate_record(
                record, subject.payload, allow_checker_model=allow_checker_model
            )
            if key in first_line:
                raise ValueError(
                    f"key {key!r} already answered on line {first_line[key]} of this file — "
                    "one answer per key per file; record the adjudication from the "
                    "adjudication queue in a later file"
                )
            first_line[key] = lineno
            verdict = _stored(store, key)
            if verdict is None:
                raise ValueError(
                    "no drafted subsplit verdict under this key — draft it with "
                    "`judge-apply --verdicts` first"
                )
            state = check_state(verdict)
            if state == "owner":
                result.skipped_owner += 1
                continue
            if state in ("agreed", "adjudicated"):
                raise ValueError(
                    "this verdict is already settled (agreed or adjudicated); re-draft it to "
                    "check it again"
                )
            trail = {
                **judgment,
                "reason": record["reason"].strip(),
                "by": record["model"],
                "effort": record["effort"],
            }
            draft = verdict.get("draft") or {**_judgment(verdict), "basis": verdict["basis"]}
            if state == "unchecked":
                agreed = agreement_signature(judgment) == agreement_signature(verdict)
                result.agreed += agreed
                result.disagreed += not agreed
                updated = {
                    **_judgment(verdict),
                    "basis": verdict["basis"],
                    "check": {"agreed": agreed, "by": record["model"], "adjudicated": False},
                    "draft": draft,
                    "checker": trail,
                }
            else:  # disputed: this answer is the adjudication
                result.adjudicated += 1
                updated = {
                    **judgment,
                    "reason": record["reason"].strip(),
                    "basis": verdict["basis"],
                    "check": {"agreed": False, "by": record["model"], "adjudicated": True},
                    "draft": draft,
                    "checker": verdict.get("checker"),
                    "adjudicator": trail,
                }
            result.updates.append((key, updated))
        except ValueError as exc:
            raise ValueError(f"line {lineno}: {exc}") from exc
    return result
