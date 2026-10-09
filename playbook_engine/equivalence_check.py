"""Blind check and adjudication of equivalence verdicts (issue #240).

Every drafted ``vs_standard`` verdict is checked by an independent model before
anyone relies on it, with no owner gate in front of the consumer:

1. **Draft.** A coder, or the store-backed judge, drafts a verdict
   (``basis`` ``"agent"`` / ``"judge"``).
2. **Blind check.** ``playbook judge --check equivalence <out>`` writes
   ``judge/check-pending.jsonl``: one item per drafted, unchecked verdict
   carrying exactly the payload the drafter saw — WITHOUT the drafted label or
   reason. A SEPARATE agent pinned to :data:`CHECKER_MODEL` at reasoning effort
   :data:`CHECKER_EFFORT` answers it (never the drafter), or the keyed API path
   (:func:`check_via_api`) does.
3. **Record.** ``playbook judge-apply --check <file>`` records each answer with
   the checker's model id and effort and REJECTS a record whose model id or
   effort is not the pinned pair, unless ``--allow-checker-model`` is passed:
   enforcement is the recorded id, not the launch configuration.
4. **Agreement** keeps the verdict, with ``check: {agreed: true, by, adjudicated:
   false}``. **Disagreement** records ``check: {agreed: false, by,
   adjudicated: false}`` (the draft still flows through) and puts the item on
   ``judge/adjudication-pending.jsonl``; a fresh pinned agent sees both labels
   and reasons, the text and our standard, and its answer replaces the verdict
   with ``check: {agreed: false, by, adjudicated: true}``.
5. **Flow-through.** An unchecked draft is written to the playbook anyway
   (``check: null``); nothing blocks. A verdict whose ``basis`` is ``"owner"``
   is never checked and wins over any agent or check answer.

The store keeps the whole trail (``draft``, ``checker``, ``adjudicator``)
beside the verdict, and the verdict store is append-only, so the original draft
stays on disk after a check or adjudication.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from playbook_engine.agent_judge import VerdictStore
from playbook_engine.equivalence import (
    BASIS_OWNER,
    LABELS,
    EquivalenceSubject,
)

__all__ = [
    "ADJUDICATION_KIND",
    "ADJUDICATION_PENDING",
    "CHECKER_EFFORT",
    "CHECK_KIND",
    "CHECKER_MODEL",
    "CHECK_PENDING",
    "ApplyResult",
    "CheckQueues",
    "apply_check_records",
    "build_check_queues",
    "check_via_api",
    "load_check_records",
    "write_check_queues",
]

_log = logging.getLogger(__name__)

#: The pinned checker/adjudicator: Claude Opus 5.5 at ``xhigh`` reasoning effort.
CHECKER_MODEL = "claude-opus-5-5"
CHECKER_EFFORT = "xhigh"

#: Queue file names under ``<out>/judge/``.
CHECK_PENDING = "check-pending.jsonl"
ADJUDICATION_PENDING = "adjudication-pending.jsonl"

#: Pending-item kinds of the two queues.
CHECK_KIND = "equivalence_check"
ADJUDICATION_KIND = "equivalence_adjudication"


# ---------------------------------------------------------------------------
# Verdict state
# ---------------------------------------------------------------------------


def _is_equivalence_verdict(verdict: Mapping[str, Any] | None) -> bool:
    return (
        isinstance(verdict, Mapping)
        and verdict.get("label") in LABELS
        and isinstance(verdict.get("reason"), str)
    )


def _state(verdict: Mapping[str, Any]) -> str:
    """``owner`` | ``unchecked`` | ``disputed`` | ``settled`` for a stored verdict."""
    if verdict.get("basis") == BASIS_OWNER:
        return "owner"
    check = verdict.get("check")
    if not isinstance(check, Mapping):
        return "unchecked"
    if check.get("agreed") is False and check.get("adjudicated") is not True:
        return "disputed"
    return "settled"


# ---------------------------------------------------------------------------
# Queues
# ---------------------------------------------------------------------------


@dataclass
class CheckQueues:
    """The two queues one ``judge --check`` round emits."""

    #: Blind check items: the drafter's payload, no label or reason.
    check: list[dict[str, Any]] = field(default_factory=list)
    #: Adjudication items: the payload plus both labels and reasons.
    adjudication: list[dict[str, Any]] = field(default_factory=list)
    #: Subjects with no stored verdict yet (nothing to check).
    undrafted: int = 0
    #: Subjects already settled (agreed, adjudicated, or owner-decided).
    settled: int = 0


def build_check_queues(store: VerdictStore, subjects: Iterable[EquivalenceSubject]) -> CheckQueues:
    """Queue every drafted-but-unchecked verdict and every unadjudicated dispute."""
    queues = CheckQueues()
    for subject in subjects:
        verdict = store.get_by_key(subject.key)
        if not _is_equivalence_verdict(verdict):
            queues.undrafted += 1
            continue
        assert verdict is not None
        state = _state(verdict)
        if state == "unchecked":
            queues.check.append(
                {"key": subject.key, "kind": CHECK_KIND, "payload": dict(subject.payload)}
            )
        elif state == "disputed":
            draft = verdict.get("draft") or {"label": verdict["label"], "reason": verdict["reason"]}
            checker = verdict.get("checker") or {}
            queues.adjudication.append(
                {
                    "key": subject.key,
                    "kind": ADJUDICATION_KIND,
                    "payload": {
                        **subject.payload,
                        "draft": {"label": draft.get("label"), "reason": draft.get("reason")},
                        "check": {"label": checker.get("label"), "reason": checker.get("reason")},
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


# ---------------------------------------------------------------------------
# Applying check / adjudication answers
# ---------------------------------------------------------------------------

#: A check record, one JSON object per line of the ``--check`` file.
_RECORD_KEYS = frozenset({"key", "label", "reason", "model", "effort"})


def load_check_records(path: Path) -> list[tuple[int, dict[str, Any]]]:
    """Parse a ``--check`` file into ``(line_number, record)`` pairs.

    Raises ``ValueError`` (with the line number) on malformed JSON or a record
    that is not ``{key, label, reason, model, effort}``.
    """
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
        missing = sorted(_RECORD_KEYS - set(record))
        extra = sorted(set(record) - _RECORD_KEYS)
        if missing or extra:
            raise ValueError(
                f"line {lineno}: a check record is {{key, label, reason, model, effort}}"
                + (f"; missing {missing}" if missing else "")
                + (f"; unexpected {extra}" if extra else "")
            )
        records.append((lineno, record))
    return records


def _validate_record(record: Mapping[str, Any], *, allow_checker_model: bool) -> None:
    if not isinstance(record["key"], str) or not record["key"]:
        raise ValueError("'key' must be a non-empty string")
    if record["label"] not in LABELS:
        raise ValueError(f"'label' must be one of {list(LABELS)}, got {record['label']!r}")
    if not isinstance(record["reason"], str) or not record["reason"].strip():
        raise ValueError("'reason' must be a non-empty string")
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
    """Counts of what a ``judge-apply --check`` file did."""

    agreed: int = 0
    disagreed: int = 0
    adjudicated: int = 0
    skipped_owner: int = 0
    #: ``(key, new verdict)`` pairs to write, in order; at most one per key.
    updates: list[tuple[str, dict[str, Any]]] = field(default_factory=list)


def _transition(
    verdict: dict[str, Any], record: Mapping[str, Any], result: ApplyResult
) -> dict[str, Any]:
    answer = {
        "label": record["label"],
        "reason": record["reason"],
        "by": record["model"],
        "effort": record["effort"],
    }
    draft = verdict.get("draft") or {
        "label": verdict["label"],
        "reason": verdict["reason"],
        "basis": verdict["basis"],
    }
    state = _state(verdict)
    if state == "unchecked":
        agreed = record["label"] == verdict["label"]
        result.agreed += agreed
        result.disagreed += not agreed
        return {
            "label": verdict["label"],
            "reason": verdict["reason"],
            "basis": verdict["basis"],
            "check": {"agreed": agreed, "by": record["model"], "adjudicated": False},
            "draft": draft,
            "checker": answer,
        }
    # state == "disputed": the answer is the adjudication.
    result.adjudicated += 1
    return {
        "label": record["label"],
        "reason": record["reason"],
        "basis": verdict["basis"],
        "check": {"agreed": False, "by": record["model"], "adjudicated": True},
        "draft": draft,
        "checker": verdict.get("checker"),
        "adjudicator": answer,
    }


def apply_check_records(
    store: VerdictStore,
    records: Iterable[tuple[int, dict[str, Any]]],
    *,
    allow_checker_model: bool = False,
) -> ApplyResult:
    """Validate every record, then compute the verdict updates (store untouched).

    The answer stage is inferred from the verdict stored BEFORE this file: no
    check yet means a blind-check answer, a recorded disagreement means an
    adjudication. Every record is judged against that pre-file state, never
    against an earlier record of the same file, and a key that appears more
    than once in one file is rejected — so one file can never move a verdict
    through both the blind check and the adjudication (the adjudicator must
    answer the adjudication item, which carries both labels and reasons).
    Raises ``ValueError`` naming the line of the first problem, so nothing is
    loaded if any record fails. The caller writes ``result.updates`` with
    ``store.put_by_key`` (keeping each key's rubric stamp).
    """
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
            stored = store.get_by_key(key)
            if not _is_equivalence_verdict(stored):
                raise ValueError(
                    "no drafted equivalence verdict under this key — draft it with "
                    "`judge-apply --verdicts` first"
                )
            verdict = dict(stored)  # type: ignore[arg-type]
            state = _state(verdict)
            if state == "owner":
                result.skipped_owner += 1
                continue
            if state == "settled":
                raise ValueError(
                    "this verdict is already settled (agreed or adjudicated); re-draft it to "
                    "check it again"
                )
            result.updates.append((key, _transition(verdict, record, result)))
        except ValueError as exc:
            raise ValueError(f"line {lineno}: {exc}") from exc
    return result


# ---------------------------------------------------------------------------
# Optional keyed API path
# ---------------------------------------------------------------------------

_CHECK_SYSTEM_PROMPT = """\
You are checking one clause from a contract negotiation precedent record. You are
given our standard clause for a clause type, a candidate clause text, and the
party whose perspective counts (`perspective_party`). Compare the candidate's
legal effect for that party with our standard's, and answer with exactly one
label and one sentence naming the operative difference:

- equivalent: the same legal effect for the party as our standard;
- more_protective: better for the party than our standard;
- less_protective: worse for the party than our standard;
- different_concept: it does something our standard does not, or omits what our
  standard does, so the two cannot be compared as more or less protective.

Judge the language only. You are told nothing about who proposed it, whose
paper it was on, or how a negotiation ended, and none of that may matter.
"""

_ADJUDICATE_SYSTEM_PROMPT = (
    _CHECK_SYSTEM_PROMPT
    + """
Two earlier reviewers disagreed about this candidate. Their labels and reasons are
included as `draft` and `check`. Decide afresh from the texts; do not simply side
with either of them.
"""
)

_ANSWER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "enum": list(LABELS)},
        "reason": {"type": "string"},
    },
    "required": ["label", "reason"],
    "additionalProperties": False,
}


def _build_request(item: Mapping[str, Any], *, max_tokens: int) -> dict[str, Any]:
    """One Message Batches request for a check or adjudication item.

    Deliberate omissions, verified against the API reference for this model:
    no ``thinking`` parameter (thinking is always on and ``disabled`` is a
    400) and no forced ``tool_choice`` (a 400) — the answer comes back through
    structured outputs. ``effort`` is set explicitly because the model's
    default is ``medium``.
    """
    adjudicating = item.get("kind") == ADJUDICATION_KIND
    return {
        "custom_id": item["key"],
        "params": {
            "model": CHECKER_MODEL,
            "max_tokens": max_tokens,
            "output_config": {
                "effort": CHECKER_EFFORT,
                "format": {"type": "json_schema", "schema": _ANSWER_SCHEMA},
            },
            "system": _ADJUDICATE_SYSTEM_PROMPT if adjudicating else _CHECK_SYSTEM_PROMPT,
            "messages": [
                {
                    "role": "user",
                    "content": json.dumps(item["payload"], ensure_ascii=False, indent=2),
                }
            ],
        },
    }


def check_via_api(
    items: list[dict[str, Any]],
    *,
    client: Any,
    max_tokens: int = 16000,
    poll_interval_s: float = 30.0,
    max_polls: int = 2880,
    progress: Callable[[str], None] = lambda _: None,
) -> tuple[list[dict[str, Any]], int]:
    """Answer check or adjudication *items* through the Message Batches API.

    Returns ``(records, unchecked)``: one ``{key, label, reason, model,
    effort}`` record per answered item — ``model`` is the model id the API
    reports the response actually ran on, never the requested one — and the
    count of items left unchecked (a refusal, an errored or unparsable
    result). There is NO server-side refusal fallback: a fallback would answer
    on a different model and defeat the point of the check, so a
    ``stop_reason`` of ``"refusal"`` leaves the item unchecked and logged.
    """
    if not items:
        return [], 0
    batch = client.messages.batches.create(
        requests=[_build_request(item, max_tokens=max_tokens) for item in items]
    )
    batch_id = batch.id
    status = getattr(batch, "processing_status", None)
    polls = 0
    while status != "ended":
        polls += 1
        if polls > max_polls:
            raise RuntimeError(
                f"batch {batch_id} still {status!r} after {polls - 1} polls; it was not "
                "cancelled — collect its results from the Anthropic console rather than "
                "resubmitting these items"
            )
        if poll_interval_s > 0:
            time.sleep(poll_interval_s)
        batch = client.messages.batches.retrieve(batch_id)
        status = getattr(batch, "processing_status", None)
        progress(f"  batch {batch_id}: {status}")

    records: list[dict[str, Any]] = []
    unchecked = 0
    answered: set[str] = set()
    for entry in client.messages.batches.results(batch_id):
        key = entry.custom_id
        answered.add(key)
        result = entry.result
        message = getattr(result, "message", None)
        if getattr(result, "type", None) != "succeeded" or message is None:
            _log.warning("equivalence check %s: batch result %r; left unchecked", key, result.type)
            unchecked += 1
            continue
        if getattr(message, "stop_reason", None) == "refusal":
            _log.warning("equivalence check %s: the model refused; left unchecked", key)
            unchecked += 1
            continue
        text_block = next(
            (b for b in (message.content or []) if getattr(b, "type", None) == "text"), None
        )
        try:
            answer = json.loads(text_block.text)  # type: ignore[union-attr]
            label, reason = answer["label"], answer["reason"]
            if label not in LABELS or not isinstance(reason, str) or not reason.strip():
                raise ValueError("answer is not a valid label and reason")
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            _log.warning("equivalence check %s: unparsable answer (%s); left unchecked", key, exc)
            unchecked += 1
            continue
        records.append(
            {
                "key": key,
                "label": label,
                "reason": reason,
                "model": str(getattr(message, "model", "")),
                "effort": CHECKER_EFFORT,
            }
        )
    unchecked += sum(1 for item in items if item["key"] not in answered)
    return records, unchecked
