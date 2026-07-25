"""Chat-to-curate: minimal deterministic CLI grammar for embedded curation — issue #159.

Answers the #104 "visualize + chat to fine-tune" interaction goal with the
AFK-able slice: a small, offline-testable text command grammar (NOT
LLM-driven natural-language parsing — see issue #159's Out of scope note)
that lets an attorney type short instructions and have them applied through
the same embedded-curation layer the HTML viewer's Export-feedback path uses
(``playbook_engine.curation`` / issue #147):

    pin governing_law to usually_conceded
    pin clause governing_law to stance usually_conceded: keep as filed
    note indemnification: check this again once the MSA renews

Two instruction kinds:

- ``pin <clause> to <stance>[: <comment>]`` — embeds a
  ``playbook_engine.curation.CurationPin`` for the clause, recording the
  attorney's asserted ``position`` (the ``<stance>`` token) AND the clause's
  *current* ``historical_stance``/``rollup.position`` as ``baseline_stance``
  — what the attorney is overriding FROM. The leading filler words
  ``clause``/``stance`` are optional sugar; ``pin <clause> to <stance>`` and
  ``pin clause <clause> to stance <stance>`` parse identically.
- ``note <clause>: <text>`` — a free-text note, appended to
  ``out_dir/viewer_notes.md`` (same sink ``apply_feedback`` uses).

A clause reference resolves against, case-insensitively: the clause's
viewer item number (``C1``, ``C2``, ...; see ``viewer._build_index``), its
stable ``id``, its ``taxonomy_id``, or its ``title``.

Every call to ``apply_curate_commands()`` also refreshes conflict status on
EVERY already-embedded pin (not just ones touched by this batch) by running
``curation.merge_curation()`` against the clause stances currently in the
OPF — mirroring what a pipeline recompile does. This is what lets "a pin
conflicting with new evidence" surface even when the evidence moved via some
other path (a recompile, a hand edit) between two ``curate`` invocations:
the next ``curate`` call — pin, note, or otherwise — reports it.

Unresolvable clause references and unparseable lines are reported back as
unapplied (not silently dropped and not fatal — issue #138's "don't claim
false success" principle), so a batch with one typo doesn't lose every other
instruction in it.
"""

from __future__ import annotations

import datetime
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from playbook_engine.canonicalize import compute_section_digests, content_hash
from playbook_engine.curation import CurationPin, merge_curation
from playbook_engine.opf_accessors import clause_stance, playbook_clauses
from playbook_engine.playbook_assembler import write_playbook
from playbook_engine.viewer import _build_index

__all__ = [
    "CurateError",
    "CurateOutcome",
    "CurateResult",
    "apply_curate_commands",
]

_PIN_RE = re.compile(
    r"^pin\s+(?:clause\s+)?(?P<clause>.+?)\s+to\s+(?:stance\s+)?(?P<stance>\S+)"
    r"\s*(?::\s*(?P<comment>.+))?$",
    re.IGNORECASE,
)
_NOTE_RE = re.compile(
    r"^note\s+(?:clause\s+)?(?P<clause>.+?)\s*:\s*(?P<text>.+)$",
    re.IGNORECASE,
)


class CurateError(ValueError):
    """A curate instruction line could not be parsed."""


@dataclass(frozen=True)
class _ParsedCommand:
    action: str  # "pin" | "note"
    clause_ref: str
    value: str  # asserted stance (pin) or note text (note)
    comment: str | None = None  # pin only


@dataclass
class CurateOutcome:
    """The result of applying (or failing to apply) one curate instruction line.

    Attributes:
        command:   The raw instruction line, verbatim (trimmed).
        action:    ``"pin"``, ``"note"``, ``"conflict"`` (a pre-existing pin
                   whose evidence has moved, surfaced during this run's
                   refresh — see module docstring), or ``"error"`` (line
                   could not be parsed / clause reference did not resolve).
        clause_id: The resolved clause's stable ``id``, or the raw reference
                   text when resolution failed.
        applied:   Whether this instruction was honored.
        conflict:  Whether this outcome reports a pin/evidence conflict.
        detail:    Human-readable explanation, echoed by the CLI.
    """

    command: str
    action: str
    clause_id: str
    applied: bool
    conflict: bool = False
    detail: str = ""


@dataclass
class CurateResult:
    """Summary of an ``apply_curate_commands()`` run.

    Attributes:
        outcomes:      One ``CurateOutcome`` per instruction line, in order,
                        PLUS one trailing ``"conflict"`` outcome per pin
                        whose evidence has moved since it was made (see
                        module docstring).
        pins_written:  Number of ``pin`` instructions applied this run.
        notes_written: Number of ``note`` instructions applied this run.
    """

    outcomes: list[CurateOutcome] = field(default_factory=list)
    pins_written: int = 0
    notes_written: int = 0

    @property
    def conflicts(self) -> list[CurateOutcome]:
        return [o for o in self.outcomes if o.conflict]


def _parse_line(line: str) -> _ParsedCommand:
    pin_match = _PIN_RE.match(line)
    if pin_match:
        comment = pin_match.group("comment")
        return _ParsedCommand(
            action="pin",
            clause_ref=pin_match.group("clause").strip(),
            value=pin_match.group("stance").strip(),
            comment=comment.strip() if comment else None,
        )

    note_match = _NOTE_RE.match(line)
    if note_match:
        return _ParsedCommand(
            action="note",
            clause_ref=note_match.group("clause").strip(),
            value=note_match.group("text").strip(),
        )

    raise CurateError(
        f"unrecognized instruction: {line!r} "
        "(expected 'pin <clause> to <stance>[: <comment>]' or 'note <clause>: <text>')"
    )


def _clause_lookup(doc: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Map every case-insensitive clause reference (item#/id/taxonomy_id/title) to its payload.

    First match wins on collision (deterministic index order from
    ``_build_index`` — clauses sorted by ``taxonomy_id`` then ``id``); an MVP
    limitation, not a silent-ambiguity hazard, since the grammar has no way
    to disambiguate further.
    """
    lookup: dict[str, dict[str, Any]] = {}
    for item_num, kind, payload in _build_index(doc):
        if kind != "clause":
            continue
        for key in (
            item_num,
            payload.get("_clause_id"),
            payload.get("taxonomy_id"),
            payload.get("title"),
        ):
            if not key:
                continue
            lk = str(key).strip().lower()
            lookup.setdefault(lk, payload)
    return lookup


def apply_curate_commands(
    out_dir: Path,
    commands: list[str],
    *,
    pinned_by: str | None = None,
    now: str | None = None,
) -> CurateResult:
    """Parse and apply a sequence of curate instructions against OUT_DIR's playbook.

    Args:
        out_dir:   Directory containing ``playbook.opf.json`` (from
                   ``playbook compile``/``project``).
        commands:  Instruction lines (see module docstring for grammar).
                   Blank lines and lines starting with ``#`` are skipped.
        pinned_by: Optional attribution stamped on any pin created
                   (``CurationPin.pinned_by``).
        now:       ISO-8601 timestamp to stamp on pins/conflicts created this
                   run. Defaults to the current UTC time.

    Returns:
        A ``CurateResult`` — one outcome per instruction line plus one per
        newly-detected pin/evidence conflict (module docstring).

    Raises:
        FileNotFoundError: If ``out_dir/playbook.opf.json`` does not exist.
    """
    opf_path = out_dir / "playbook.opf.json"
    if not opf_path.exists():
        raise FileNotFoundError(f"playbook.opf.json not found in {out_dir}")

    doc: dict[str, Any] = json.loads(opf_path.read_text(encoding="utf-8"))
    clauses = playbook_clauses(doc)
    clause_stances = {c["id"]: clause_stance(c) for c in clauses if c.get("id")}
    clause_lookup = _clause_lookup(doc)

    timestamp = now or datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")

    result = CurateResult()
    notes: list[str] = []
    existing_pins_raw = list((doc.get("curation") or {}).get("pins") or [])
    pins_by_clause_id: dict[str, dict[str, Any]] = {p["clause_id"]: p for p in existing_pins_raw}

    for raw_line in commands:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        try:
            parsed = _parse_line(line)
        except CurateError as exc:
            result.outcomes.append(
                CurateOutcome(
                    command=line, action="error", clause_id="", applied=False, detail=str(exc)
                )
            )
            continue

        clause = clause_lookup.get(parsed.clause_ref.lower())
        if clause is None:
            result.outcomes.append(
                CurateOutcome(
                    command=line,
                    action="error",
                    clause_id=parsed.clause_ref,
                    applied=False,
                    detail=f"no clause matches {parsed.clause_ref!r}",
                )
            )
            continue

        clause_id = str(clause.get("_clause_id") or clause.get("id"))

        if parsed.action == "pin":
            pin = CurationPin(
                clause_id=clause_id,
                item_id=str(clause.get("_clause_num", "")),
                position=parsed.value,
                baseline_stance=clause_stances.get(clause_id, "unknown"),
                pinned_at=timestamp,
                pinned_by=pinned_by,
                comment=parsed.comment,
            )
            pins_by_clause_id[clause_id] = pin.to_dict()
            result.pins_written += 1
            result.outcomes.append(
                CurateOutcome(
                    command=line,
                    action="pin",
                    clause_id=clause_id,
                    applied=True,
                    detail=f"pinned to {parsed.value!r}",
                )
            )
        else:
            title = clause.get("title", "")
            notes.append(f"**{clause.get('_clause_num', '')}** ({title}): {parsed.value}")
            result.notes_written += 1
            result.outcomes.append(
                CurateOutcome(
                    command=line,
                    action="note",
                    clause_id=clause_id,
                    applied=True,
                    detail="note recorded",
                )
            )

    # Refresh conflict status for EVERY pin (existing + just-added) against the
    # stances snapshot taken above — same deterministic check a pipeline
    # recompile runs (curation.merge_curation), just invoked here so drift is
    # surfaced the moment someone next runs `curate`, not only on recompile.
    merged = merge_curation(
        {"pins": list(pins_by_clause_id.values())}, clause_stances, checked_at=timestamp
    )
    final_pins = merged.get("pins", [])

    for pin_dict in final_pins:
        conflict = pin_dict.get("conflict")
        if conflict:
            result.outcomes.append(
                CurateOutcome(
                    command=f"<existing pin: {pin_dict['clause_id']}>",
                    action="conflict",
                    clause_id=pin_dict["clause_id"],
                    applied=True,
                    conflict=True,
                    detail=conflict.get(
                        "reason", "evidence changed since this position was pinned"
                    ),
                )
            )

    if final_pins != existing_pins_raw:
        doc["curation"] = {"pins": final_pins}
        if "identity" in doc:
            doc["identity"]["content_hash"] = content_hash(doc)
            doc["identity"]["section_digests"] = compute_section_digests(doc)
        write_playbook(doc, opf_path)

    if notes:
        notes_path = out_dir / "viewer_notes.md"
        existing_notes = notes_path.read_text(encoding="utf-8") if notes_path.exists() else ""
        new_notes = "\n".join(f"- {n}" for n in notes)
        combined = f"{existing_notes}\n{new_notes}\n" if existing_notes else f"{new_notes}\n"
        notes_path.write_text(combined, encoding="utf-8")

    return result
