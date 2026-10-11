"""Blind check and adjudication of equivalence verdicts (issue #240).

Every drafted ``vs_standard`` verdict is checked by an independent model before
anyone relies on it, with no owner gate in front of the consumer:

1. **Draft.** A coder, or the store-backed judge, drafts a verdict
   (``basis`` ``"agent"`` / ``"judge"``).
2. **Blind check.** ``playbook judge --check equivalence <out>`` writes
   ``judge/check-pending.jsonl``: one item per drafted, unchecked verdict
   carrying exactly the payload the drafter saw — WITHOUT the drafted
   judgment. A SEPARATE agent pinned to :data:`CHECKER_MODEL` at reasoning
   effort :data:`CHECKER_EFFORT` answers it with the same structured judgment
   (parts, additions, reason; never the drafter), or the keyed API path
   (:func:`check_via_api`) does.
3. **Record.** ``playbook judge-apply --check <file>`` records each answer with
   the checker's model id and effort and REJECTS a record whose model id or
   effort is not the pinned pair, unless ``--allow-checker-model`` is passed:
   enforcement is the recorded id, not the launch configuration.
4. **Agreement** (issue #243) is the same derived label AND the same set of
   ``(part_id, status, direction)`` over the parts that are not equivalent
   or not a neutral omission (:func:`~playbook_engine.equivalence.agreement_signature`);
   both halves are recorded (``label_agreed``, ``parts_agreed``) so the
   scorecard reports them separately. Agreement keeps the verdict, with
   ``check: {agreed: true, label_agreed: true, parts_agreed: true, by,
   adjudicated: false}``. **Disagreement** records ``agreed: false`` (the
   draft still flows through) and puts the item on
   ``judge/adjudication-pending.jsonl``; a fresh pinned agent sees both
   judgments, the text and our standard's parts, and its answer replaces the
   verdict with ``adjudicated: true``.
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
    EFFECT_NEUTRAL,
    EFFECTS,
    LABEL_EQUIVALENT,
    LABELS,
    RELATIONS,
    STATUS_PRESENT,
    STATUSES,
    EquivalenceSubject,
    agreement_signature,
    canonical_judgment,
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
        and isinstance(verdict.get("parts"), list)
        and isinstance(verdict.get("reason"), str)
    )


def _judgment(verdict: Mapping[str, Any]) -> dict[str, Any]:
    """The structured judgment of a stored verdict, draft or answer."""
    return {k: verdict.get(k) for k in ("label", "reason", "parts", "additions")}


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


# ---------------------------------------------------------------------------
# Applying check / adjudication answers
# ---------------------------------------------------------------------------

#: A check record, one JSON object per line of the ``--check`` file. ``label``
#: is optional: it is derived from the parts, and a supplied one must match.
_RECORD_KEYS = frozenset({"key", "parts", "additions", "reason", "model", "effort"})
_OPTIONAL_RECORD_KEYS = frozenset({"label"})


def load_check_records(path: Path) -> list[tuple[int, dict[str, Any]]]:
    """Parse a ``--check`` file into ``(line_number, record)`` pairs.

    Raises ``ValueError`` (with the line number) on malformed JSON or a record
    that is not ``{key, parts, additions, reason, model, effort[, label]}``.
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
        extra = sorted(set(record) - _RECORD_KEYS - _OPTIONAL_RECORD_KEYS)
        if missing or extra:
            raise ValueError(
                f"line {lineno}: a check record is "
                "{key, parts, additions, reason, model, effort}"
                + (f"; missing {missing}" if missing else "")
                + (f"; unexpected {extra}" if extra else "")
            )
        records.append((lineno, record))
    return records


def _validate_record(record: Mapping[str, Any], *, allow_checker_model: bool) -> None:
    if not isinstance(record["key"], str) or not record["key"]:
        raise ValueError("'key' must be a non-empty string")
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
    #: Of the blind checks: same derived label / same non-equivalent parts.
    label_agreed: int = 0
    parts_agreed: int = 0
    adjudicated: int = 0
    skipped_owner: int = 0
    #: ``(key, new verdict)`` pairs to write, in order; at most one per key.
    updates: list[tuple[str, dict[str, Any]]] = field(default_factory=list)


def _transition(
    verdict: dict[str, Any], answer: dict[str, Any], record: Mapping[str, Any], result: ApplyResult
) -> dict[str, Any]:
    """The verdict after *answer* (a canonical judgment) is recorded on it."""
    trail = {**answer, "by": record["model"], "effort": record["effort"]}
    draft = verdict.get("draft") or {**_judgment(verdict), "basis": verdict["basis"]}
    state = _state(verdict)
    if state == "unchecked":
        label_agreed = answer["label"] == verdict["label"]
        parts_agreed = agreement_signature(answer) == agreement_signature(verdict)
        agreed = label_agreed and parts_agreed
        result.agreed += agreed
        result.disagreed += not agreed
        result.label_agreed += label_agreed
        result.parts_agreed += parts_agreed
        return {
            **_judgment(verdict),
            "basis": verdict["basis"],
            "check": {
                "agreed": agreed,
                "label_agreed": label_agreed,
                "parts_agreed": parts_agreed,
                "by": record["model"],
                "adjudicated": False,
            },
            "draft": draft,
            "checker": trail,
        }
    # state == "disputed": the answer is the adjudication. The check's own
    # agreement halves stay as recorded (they describe the blind check).
    result.adjudicated += 1
    prior = verdict.get("check") or {}
    return {
        **answer,
        "basis": verdict["basis"],
        "check": {
            "agreed": False,
            "label_agreed": bool(prior.get("label_agreed")),
            "parts_agreed": bool(prior.get("parts_agreed")),
            "by": record["model"],
            "adjudicated": True,
        },
        "draft": draft,
        "checker": verdict.get("checker"),
        "adjudicator": trail,
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
            part_ids = [str(p.get("part_id")) for p in verdict.get("parts") or []]
            answer = canonical_judgment(
                {k: record.get(k) for k in ("parts", "additions", "reason", "label")}, part_ids
            )
            result.updates.append((key, _transition(verdict, answer, record, result)))
        except ValueError as exc:
            raise ValueError(f"line {lineno}: {exc}") from exc
    return result


# ---------------------------------------------------------------------------
# Optional keyed API path
# ---------------------------------------------------------------------------

_CHECK_SYSTEM_PROMPT = """\
You are checking one clause from a contract negotiation precedent record. You are
given the numbered parts of our standard clause for a clause type
(`our_standard_parts`, each with a `part_id`), a candidate clause text, and the
party whose perspective counts (`perspective_party`). Compare the candidate with
our standard PART BY PART, for that party:

- for EVERY part, `status` is `present` when the candidate keeps that part's
  substance (perhaps reworded or changed) and `omitted` when it drops it;
- `direction` of a present part: `equivalent` (same legal effect),
  `more_protective` or `less_protective` (better or worse for the party than
  that part of our standard);
- `direction` of an omitted part: the effect of the omission for the party,
  `more_protective`, `less_protective` or `neutral` (dropping an obligation that
  falls on the party can be more protective; dropping a recital is neutral);
- `reason`: one sentence for a part that is not equivalent or an omission that
  is not neutral, otherwise an empty string;
- `additions`: each thing the candidate adds that no part covers, with a
  one-sentence `summary` and its `effect` for the party (empty when none);
- top-level `reason`: one sentence naming the single most important difference
  (or saying the legal effect is the same).

Judge the language only. You are told nothing about who proposed it, whose
paper it was on, or how a negotiation ended, and none of that may matter.
"""

_ADJUDICATE_SYSTEM_PROMPT = (
    _CHECK_SYSTEM_PROMPT
    + """
Two earlier reviewers disagreed about this candidate. Their judgments are included
as `draft` and `check`. Decide afresh from the texts; do not simply side with
either of them.
"""
)

#: Structured-output schema of an answer. One flat ``direction`` per part
#: (structured outputs cannot express "relation when present, effect when
#: omitted"); :func:`_judgment_from_answer` maps it onto the judgment shape.
_ANSWER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "parts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "part_id": {"type": "string"},
                    "status": {"type": "string", "enum": list(STATUSES)},
                    "direction": {
                        "type": "string",
                        "enum": sorted(set(RELATIONS) | set(EFFECTS)),
                    },
                    "reason": {"type": "string"},
                },
                "required": ["part_id", "status", "direction", "reason"],
                "additionalProperties": False,
            },
        },
        "additions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string"},
                    "effect": {"type": "string", "enum": list(EFFECTS)},
                },
                "required": ["summary", "effect"],
                "additionalProperties": False,
            },
        },
        "reason": {"type": "string"},
    },
    "required": ["parts", "additions", "reason"],
    "additionalProperties": False,
}


def _judgment_from_answer(answer: Mapping[str, Any]) -> dict[str, Any]:
    """An API answer (one ``direction`` per part) as a judgment record body.

    A present part's direction is its ``relation`` (``neutral`` reads as
    ``equivalent``); an omitted part's is its ``effect`` (``equivalent`` reads
    as ``neutral``). A reason on an equivalent part or a neutral omission is
    dropped. :func:`canonical_judgment` then validates the result.
    """
    parts: list[dict[str, Any]] = []
    for part in answer.get("parts") or []:
        status = part.get("status")
        direction = part.get("direction")
        out: dict[str, Any] = {"part_id": part.get("part_id"), "status": status}
        if status == STATUS_PRESENT:
            out["relation"] = LABEL_EQUIVALENT if direction == EFFECT_NEUTRAL else direction
            notable = out["relation"] != LABEL_EQUIVALENT
        else:
            out["effect"] = EFFECT_NEUTRAL if direction == LABEL_EQUIVALENT else direction
            notable = out["effect"] != EFFECT_NEUTRAL
        if notable:
            out["reason"] = part.get("reason")
        parts.append(out)
    return {
        "parts": parts,
        "additions": list(answer.get("additions") or []),
        "reason": answer.get("reason"),
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

    Returns ``(records, unchecked)``: one ``{key, parts, additions, reason,
    model, effort}`` record per answered item — ``model`` is the model id the API
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
    by_key = {item["key"]: item.get("payload") or {} for item in items}
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
            body = _judgment_from_answer(json.loads(text_block.text))  # type: ignore[union-attr]
            part_ids = [
                str(p.get("part_id"))
                for p in (by_key.get(key) or {}).get("our_standard_parts") or []
            ]
            canonical_judgment(body, part_ids or None)
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            _log.warning("equivalence check %s: unparsable answer (%s); left unchecked", key, exc)
            unchecked += 1
            continue
        records.append(
            {
                "key": key,
                **body,
                "model": str(getattr(message, "model", "")),
                "effort": CHECKER_EFFORT,
            }
        )
    unchecked += sum(1 for item in items if item["key"] not in answered)
    return records, unchecked
