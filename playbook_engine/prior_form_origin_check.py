"""Blind check and adjudication of ``prior_form_origin`` verdicts (issue #250).

Whether a recurring first draft on our paper is OUR former language is a
judgment with a cost when it is wrong (a counterparty's wording filed as ours
hides their changed openings), so a drafted verdict counts only once it has
been checked, the way an equivalence verdict is
(:mod:`playbook_engine.equivalence_check`):

1. **Draft.** A coder, or an agent acting as the judge, drafts
   ``{ours, reason, basis}`` into the verdict store.
2. **Blind check.** ``playbook judge --check prior_form_origin <out>`` writes
   ``judge/origin-check-pending.jsonl``: one item per drafted, unchecked verdict
   with exactly the payload the drafter saw and NOT the draft. A separate
   agent pinned to :data:`~playbook_engine.equivalence_check.CHECKER_MODEL` at
   :data:`~playbook_engine.equivalence_check.CHECKER_EFFORT` answers
   ``{ours, reason}``; ``playbook judge-apply --check`` records it with the
   checker's model id and effort and rejects any other pair unless
   ``--allow-checker-model`` is passed.
3. **Agreement** is the same ``ours``. It keeps the verdict with ``check:
   {agreed: true, by, adjudicated: false}``. **Disagreement** records
   ``agreed: false`` and puts the item on ``judge/origin-adjudication-pending.jsonl``;
   a fresh pinned agent sees both answers and its answer replaces the verdict
   with ``adjudicated: true``.
4. **Counting.** Only an owner's, an agreed or an adjudicated verdict makes
   its cluster a prior standard (:func:`~playbook_engine.prior_reconstruction.outcome_of`);
   an unchecked or disputed one is pending and changes nothing.

The store keeps the whole trail (``draft``, ``checker``, ``adjudicator``).
There is no API path for this kind: answer the queues with a pinned agent.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from playbook_engine.agent_judge import VerdictStore
from playbook_engine.equivalence import SETTLED_OWNER, SETTLED_UNCHECKED, check_state
from playbook_engine.equivalence_check import CHECKER_EFFORT, CHECKER_MODEL
from playbook_engine.prior_reconstruction import OriginSubject, verdict_of

__all__ = [
    "ADJUDICATION_KIND",
    "ADJUDICATION_PENDING",
    "CHECK_KIND",
    "CHECK_PENDING",
    "ApplyResult",
    "CheckQueues",
    "apply_check_records",
    "build_check_queues",
    "load_check_records",
    "write_check_queues",
]

#: Queue file names under ``<out>/judge/`` (apart from the equivalence queues).
CHECK_PENDING = "origin-check-pending.jsonl"
ADJUDICATION_PENDING = "origin-adjudication-pending.jsonl"

#: Pending-item kinds of the two queues.
CHECK_KIND = "prior_form_origin_check"
ADJUDICATION_KIND = "prior_form_origin_adjudication"

_RECORD_KEYS = frozenset({"key", "ours", "reason", "model", "effort"})


def _state(verdict: Mapping[str, Any]) -> str:
    """``owner`` | ``unchecked`` | ``disputed`` | ``settled`` for a stored verdict."""
    state = check_state(verdict)
    if state == SETTLED_OWNER:
        return "owner"
    if state in (SETTLED_UNCHECKED, "disputed"):
        return state
    return "settled"


def _judgment(verdict: Mapping[str, Any]) -> dict[str, Any]:
    return {"ours": verdict.get("ours"), "reason": verdict.get("reason")}


@dataclass
class CheckQueues:
    """The two queues one ``judge --check prior_form_origin`` round emits."""

    check: list[dict[str, Any]] = field(default_factory=list)
    adjudication: list[dict[str, Any]] = field(default_factory=list)
    undrafted: int = 0
    settled: int = 0


def build_check_queues(store: VerdictStore, subjects: Iterable[OriginSubject]) -> CheckQueues:
    """Queue every drafted-but-unchecked verdict and every unadjudicated dispute."""
    queues = CheckQueues()
    for subject in subjects:
        verdict = verdict_of(store.get_by_key(subject.key))
        if verdict is None:
            queues.undrafted += 1
            continue
        state = _state(verdict)
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
    that is not ``{key, ours, reason, model, effort}``."""
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
                f"line {lineno}: a check record is {{key, ours, reason, model, effort}}"
                f"; missing {sorted(_RECORD_KEYS - set(record))}"
                f"; unexpected {sorted(set(record) - _RECORD_KEYS)}"
            )
        records.append((lineno, record))
    return records


def _validate_record(record: Mapping[str, Any], *, allow_checker_model: bool) -> None:
    if not isinstance(record["key"], str) or not record["key"]:
        raise ValueError("'key' must be a non-empty string")
    if not isinstance(record["ours"], bool):
        raise ValueError("'ours' must be a JSON boolean")
    if not isinstance(record["reason"], str) or not record["reason"].strip():
        raise ValueError("'reason' must be a non-empty string (one sentence)")
    model, effort = record["model"], record["effort"]
    if not isinstance(model, str) or not model or not isinstance(effort, str) or not effort:
        raise ValueError("'model' and 'effort' must be non-empty strings (the checker's own)")
    if allow_checker_model:
        return
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


@dataclass
class ApplyResult:
    """Counts of what a ``judge-apply --check`` file did to origin verdicts."""

    agreed: int = 0
    disagreed: int = 0
    adjudicated: int = 0
    skipped_owner: int = 0
    #: ``(key, new verdict)`` pairs to write, in order; at most one per key.
    updates: list[tuple[str, dict[str, Any]]] = field(default_factory=list)


def apply_check_records(
    store: VerdictStore,
    records: Iterable[tuple[int, dict[str, Any]]],
    *,
    allow_checker_model: bool = False,
) -> ApplyResult:
    """Validate every record, then compute the verdict updates (store untouched).

    The stage is inferred from the verdict stored BEFORE this file: no check
    yet is a blind-check answer, a recorded disagreement an adjudication; a key
    twice in one file is rejected so one file cannot move a verdict through
    both. Raises ``ValueError`` naming the line of the first problem."""
    result = ApplyResult()
    first_line: dict[str, int] = {}
    for lineno, record in records:
        try:
            _validate_record(record, allow_checker_model=allow_checker_model)
            key = record["key"]
            if key in first_line:
                raise ValueError(
                    f"key {key!r} already answered on line {first_line[key]} of this file — "
                    "one answer per key per file; record the adjudication from the "
                    "adjudication queue in a later file"
                )
            first_line[key] = lineno
            verdict = verdict_of(store.get_by_key(key))
            if verdict is None:
                raise ValueError(
                    "no drafted prior_form_origin verdict under this key — draft it with "
                    "`judge-apply --verdicts` first"
                )
            state = _state(verdict)
            if state == "owner":
                result.skipped_owner += 1
                continue
            if state == "settled":
                raise ValueError(
                    "this verdict is already settled (agreed or adjudicated); re-draft it to "
                    "check it again"
                )
            trail = {
                "ours": record["ours"],
                "reason": record["reason"].strip(),
                "by": record["model"],
                "effort": record["effort"],
            }
            draft = verdict.get("draft") or {**_judgment(verdict), "basis": verdict["basis"]}
            if state == "unchecked":
                agreed = record["ours"] == verdict["ours"]
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
                    "ours": record["ours"],
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
