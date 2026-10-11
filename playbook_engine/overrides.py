"""Owner overrides — the optional ``overrides.json`` of a derivation out-dir (issue #241).

The engine relies on model judgment throughout; a person MAY confirm or
correct a short list of those judgments (and edit the authored Posture and
Floor text) in the Review tab of ``index.html``. The page saves the edits to
``<out>/overrides.json``; this module validates that file and folds it into
the playbook. Nothing waits on it: a run without ``overrides.json`` is
complete.

File format (``overrides_version`` 1, the one and only version)::

    {
      "overrides_version": 1,
      "based_on": "sha256:...",                 # optional: identity.content_hash the page showed
      "overrides": [
        {"target": "vs_standard", "id": "<64-hex cache key>",
         "value": {"label": "less_protective", "reason": "<optional sentence>"},
         "basis": "owner", "note": "<optional>"},
        {"target": "vs_standard", "id": "<64-hex cache key>",
         "value": {"parts": [{"part_id": "<id>", "status": "present",
                              "relation": "less_protective", "reason": "<optional>"}],
                   "additions": [...], "reason": "<optional sentence>"},
         "basis": "owner"},
        {"target": "posture", "id": "system_prompt",
         "value": "<text>", "basis": "owner"},
        {"target": "floor", "id": "<invariant id>", "field": "statement" | "rationale",
         "value": "<text>", "basis": "owner"}
      ]
    }

Keys are stable ids: a ``vs_standard`` entry is addressed by the verdict cache
key of OPF-SPEC §3.5.6 (:func:`playbook_engine.equivalence.equivalence_key`) —
ONE verdict per distinct text, shared by every precedent record, in every role,
that carries the text — and a Posture or Floor entry by the field it replaces.
Every entry carries ``basis: "owner"``. A ``vs_standard`` value either sets the
derived label's INPUTS — ``parts`` (merged by ``part_id`` over the current
judgment; every part of our standard must end up judged) and optionally
``additions`` — so the label is re-derived (issue #243), with the owner's
``reason``; without one, the judgment's reason is kept only while the part map
and addition tokens it was written for are unchanged, and a fallback replaces
it otherwise (it explained the old map, and maybe the old label) — or sets the
``label`` itself: the label the parts already imply confirms the judgment, any
other label is recorded as ``owner_label`` with its reason, over the model's parts.
Either way the judgment is re-stamped ``basis: "owner"`` with no check (the
owner's word is final), and the override writes the
same owner verdict into the run's verdict store (when the out-dir has one, so a
re-projection replays it), and — like every override — is folded in by
``playbook project`` and, on demand, ``playbook apply-overrides``. The digest,
dossiers, manifest, ``precedent.jsonl`` hash and ``identity.content_hash`` are
RECOMPUTED by the engine, never hand-edited.

Rejection is all-or-nothing: a malformed file or an entry naming an id the
playbook does not carry raises :class:`OverridesError` listing every problem,
and nothing is written.
"""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from playbook_engine.agent_judge import StoredVerdict, VerdictStore
from playbook_engine.canonicalize import compute_section_digests, content_hash
from playbook_engine.digest import build_digest_v4
from playbook_engine.dossiers import refresh_derived_sections
from playbook_engine.equivalence import (
    BASIS_OWNER,
    EQUIVALENCE_KIND,
    LABELS,
    addition_map,
    canonical_judgment,
    is_eligible,
    iter_slots,
    part_map,
    slot_key,
    standard_parts,
)
from playbook_engine.opf_accessors import (
    SIDECARS_KEY,
    perspective_party,
    precedent_sidecar_manifest,
)
from playbook_engine.playbook_assembler import write_playbook, write_precedent_sidecar
from playbook_engine.rubric import RubricStamp, rubric_version
from playbook_engine.validator import ValidationResult, validate_document

__all__ = [
    "OVERRIDES_FILENAME",
    "OVERRIDES_VERSION",
    "ApplyResult",
    "Override",
    "OverridesError",
    "apply_overrides",
    "apply_overrides_to_dir",
    "floor_edit_refusal",
    "fold_overrides",
    "load_overrides",
    "parse_overrides",
    "refresh_after_edit",
]

OVERRIDES_FILENAME = "overrides.json"
OVERRIDES_VERSION = 1

TARGET_VS_STANDARD = "vs_standard"
TARGET_POSTURE = "posture"
TARGET_FLOOR = "floor"
TARGETS = (TARGET_VS_STANDARD, TARGET_POSTURE, TARGET_FLOOR)

#: The one Posture field an override can replace.
POSTURE_FIELD = "system_prompt"
#: The Floor invariant fields an override can replace.
FLOOR_FIELDS = ("statement", "rationale")

_TOP_KEYS = frozenset({"overrides_version", "based_on", "overrides"})
_ENTRY_KEYS = frozenset({"target", "id", "field", "value", "basis", "note"})
_KEY_RE = re.compile(r"^[0-9a-f]{64}$")

#: Marker on an owner-edited Posture or Floor invariant (``x_`` keys are open).
OWNER_MARKER = "x_basis"

_REASON_FALLBACK = "Label set by the owner."


def floor_edit_refusal(invariant: Mapping[str, Any], field: str) -> str | None:
    """Why *field* of this Floor invariant cannot be overridden, or ``None`` if it can.

    The Floor is owned by other commands, and an override must not break their
    invariants (the attribution a reader relies on):

    - an invariant carrying ``x_signed_by`` was signed as exactly the text it
      holds (``playbook floor sign``, which itself refuses to overwrite one);
      editing either field would leave a sign-off on text nobody signed;
    - the ``rationale`` of an invariant the Posture interview's Q4 promoted IS
      the engine's attribution marker, not authored text; replacing it would
      orphan the invariant and make the next interview refuse to update it;
    - the ``statement`` of an invariant bound to a predicate (``x_condition``
      other than ``"judged"``) is what that predicate checks.

    Used by :func:`apply_overrides` (reject) and by the page (no edit box).
    """
    from playbook_engine.floor_candidates import floor_invariant_attribution  # noqa: PLC0415

    attribution = floor_invariant_attribution(dict(invariant))
    if attribution == "signed":
        return (
            "it carries a sign-off (x_signed_by) on its current text; change it with "
            "`playbook floor sign`, not here"
        )
    if field == "rationale" and attribution == "posture_interview":
        return (
            "its rationale is the Posture interview's attribution marker, not authored text; "
            "edit the statement or re-answer the interview"
        )
    if field == "statement" and invariant.get("x_condition", "judged") != "judged":
        return (
            "its statement is what its x_condition predicate checks; "
            "change it with `playbook floor sign`"
        )
    return None


class OverridesError(Exception):
    """``overrides.json`` was rejected; ``problems`` lists every reason, one line each."""

    def __init__(self, problems: list[str]) -> None:
        self.problems = list(problems)
        super().__init__("; ".join(self.problems))


@dataclass(frozen=True)
class Override:
    """One validated entry of ``overrides.json``."""

    target: str
    id: str
    value: Any
    field: str | None = None
    note: str | None = None
    index: int = 0

    @property
    def identity(self) -> tuple[str, str, str | None]:
        return (self.target, self.id, self.field)

    def describe(self) -> str:
        where = f"{self.target} {self.id}" + (f".{self.field}" if self.field else "")
        return f"overrides[{self.index}] ({where})"


@dataclass
class ApplyResult:
    """What an application did (counts only)."""

    #: Number of entries that changed the playbook, by target.
    changed: dict[str, int] = field(default_factory=lambda: dict.fromkeys(TARGETS, 0))
    #: Number of entries that were already in effect (idempotent re-application).
    unchanged: int = 0
    #: ``vs_standard`` cache key -> the owner verdict to write into the verdict store.
    verdicts: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: Entries left out (lenient application only), one reason line each.
    skipped: list[str] = field(default_factory=list)

    @property
    def n_changed(self) -> int:
        return sum(self.changed.values())


# ---------------------------------------------------------------------------
# Loading and shape validation
# ---------------------------------------------------------------------------


def _nonempty_str(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def parse_overrides(raw: Any) -> list[Override]:
    """Validate a parsed ``overrides.json`` object and return its entries.

    Raises:
        OverridesError: every shape problem found, one line each.
    """
    problems: list[str] = []
    if not isinstance(raw, dict):
        raise OverridesError(["the file must contain a JSON object"])
    extra_top = sorted(set(raw) - _TOP_KEYS)
    if extra_top:
        problems.append(f"unknown top-level key(s) {extra_top}")
    version = raw.get("overrides_version")
    if version != OVERRIDES_VERSION or isinstance(version, bool):
        problems.append(f"overrides_version must be {OVERRIDES_VERSION}, got {version!r}")
    based_on = raw.get("based_on")
    if based_on is not None and not _nonempty_str(based_on):
        problems.append("based_on must be a non-empty string when present")
    entries = raw.get("overrides")
    if not isinstance(entries, list):
        problems.append("overrides must be a list")
        raise OverridesError(problems)

    parsed: list[Override] = []
    seen: dict[tuple[str, str, str | None], int] = {}
    for i, entry in enumerate(entries):
        where = f"overrides[{i}]"
        if not isinstance(entry, dict):
            problems.append(f"{where}: must be an object")
            continue
        extra = sorted(set(entry) - _ENTRY_KEYS)
        if extra:
            problems.append(f"{where}: unknown key(s) {extra}")
        target = entry.get("target")
        if target not in TARGETS:
            problems.append(f"{where}: target must be one of {list(TARGETS)}, got {target!r}")
            continue
        ident = entry.get("id")
        if not isinstance(ident, str) or not ident.strip():
            problems.append(f"{where}: id must be a non-empty string")
            continue
        if entry.get("basis") != BASIS_OWNER:
            problems.append(f'{where}: basis must be "{BASIS_OWNER}", got {entry.get("basis")!r}')
        note = entry.get("note")
        if note is not None and not isinstance(note, str):
            problems.append(f"{where}: note must be a string when present")
            note = None
        fld = entry.get("field")
        value = entry.get("value")
        ok = True
        if target == TARGET_VS_STANDARD:
            if not _KEY_RE.match(ident):
                problems.append(f"{where}: a vs_standard id is the 64-hex verdict cache key")
                ok = False
            if fld is not None:
                problems.append(f"{where}: a vs_standard entry takes no field")
                ok = False
            why = _vs_value_problem(value)
            if why is not None:
                problems.append(f"{where}: {why}")
                ok = False
        elif target == TARGET_POSTURE:
            if ident != POSTURE_FIELD:
                problems.append(f'{where}: the only posture field is "{POSTURE_FIELD}"')
                ok = False
            if fld is not None:
                problems.append(f"{where}: a posture entry takes no field")
                ok = False
            if not _nonempty_str(value):
                problems.append(f"{where}: value must be non-empty text")
                ok = False
        else:  # floor
            if fld not in FLOOR_FIELDS:
                problems.append(f"{where}: floor field must be one of {list(FLOOR_FIELDS)}")
                ok = False
            if not _nonempty_str(value):
                problems.append(f"{where}: value must be non-empty text")
                ok = False
        if not ok:
            continue
        item = Override(target=target, id=ident, value=value, field=fld, note=note or None, index=i)
        if item.identity in seen:
            problems.append(
                f"{where}: duplicates overrides[{seen[item.identity]}] "
                f"({item.target} {item.id}{'.' + fld if fld else ''})"
            )
            continue
        seen[item.identity] = i
        parsed.append(item)
    if problems:
        raise OverridesError(problems)
    return parsed


def load_overrides(path: Path) -> list[Override]:
    """Read and validate *path* (``overrides.json``).

    Raises:
        OverridesError: unreadable, not JSON, or any shape problem.
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise OverridesError([f"could not read {path}: {exc}"]) from exc
    except ValueError as exc:
        raise OverridesError([f"{path} is not valid JSON: {exc}"]) from exc
    try:
        return parse_overrides(raw)
    except OverridesError as exc:
        raise OverridesError([f"{path.name}: {p}" for p in exc.problems]) from exc


# ---------------------------------------------------------------------------
# Applying to a playbook
# ---------------------------------------------------------------------------


_VS_LABEL_KEYS = frozenset({"label", "reason"})
_VS_PARTS_KEYS = frozenset({"parts", "additions", "reason"})
_PARTS_FALLBACK = "Set by the owner."


def _vs_value_problem(value: Any) -> str | None:
    """Why *value* is not a ``vs_standard`` override value, or ``None``.

    ``{label[, reason]}`` or ``{parts[, additions][, reason]}`` (issue #243).
    A part names its ``part_id`` and ``status`` and a ``relation`` (present)
    or ``effect`` (omitted); its ``reason`` is optional here (the owner's
    word needs none).
    """
    if not isinstance(value, dict):
        return "vs_standard value must be an object"
    if "label" in value:
        if set(value) - _VS_LABEL_KEYS:
            return "vs_standard value is {label[, reason]} or {parts[, additions][, reason]}"
        if value.get("label") not in LABELS:
            return f"label must be one of {list(LABELS)}, got {value.get('label')!r}"
    elif "parts" in value:
        if set(value) - _VS_PARTS_KEYS:
            return "vs_standard value is {label[, reason]} or {parts[, additions][, reason]}"
        parts = value.get("parts")
        if not isinstance(parts, list) or not parts:
            return "parts must be a non-empty list of {part_id, status, relation|effect}"
        for i, part in enumerate(parts):
            try:
                _owner_part(part, i)
            except ValueError as exc:
                return str(exc)
        additions = value.get("additions")
        if additions is not None:
            try:
                canonical_judgment(
                    {"parts": [_PLACEHOLDER_PART], "additions": additions, "reason": "x"}
                )
            except ValueError as exc:
                return str(exc)
    else:
        return "vs_standard value is {label[, reason]} or {parts[, additions][, reason]}"
    if "reason" in value and not _nonempty_str(value["reason"]):
        return "reason must be a non-empty sentence when present"
    return None


_PLACEHOLDER_PART = {"part_id": "p", "status": "present", "relation": "equivalent"}


def _owner_part(part: Any, i: int) -> dict[str, Any]:
    """One owner-set part, in judgment shape (a missing reason filled in)."""
    if not isinstance(part, dict):
        raise ValueError(f"parts[{i}] must be an object")
    filled = dict(part)
    direction = filled.get("relation", filled.get("effect"))
    if direction in ("more_protective", "less_protective") and not _nonempty_str(
        filled.get("reason")
    ):
        filled["reason"] = _PARTS_FALLBACK
    if direction in ("equivalent", "neutral"):
        filled.pop("reason", None)
    canonical_judgment({"parts": [filled], "additions": [], "reason": "x"})
    return filled


def _same_map(judgment: Mapping[str, Any], current: Mapping[str, Any] | None) -> bool:
    """Whether *judgment* has the part map and addition tokens of *current*
    (the judgment a ``reason`` was written for), so that reason still fits."""
    if not isinstance(current, Mapping):
        return False
    return (part_map(judgment), addition_map(judgment)) == (
        part_map(current),
        addition_map(current),
    )


def _vs_standard_object(
    entry: Override, current: Mapping[str, Any] | None, part_ids: list[str]
) -> dict[str, Any]:
    """The owner's ``vs_standard`` for *entry* over the *current* judgment.

    Raises ``ValueError`` when the entry cannot apply: a label for a text
    nobody has judged (there are no parts to keep), or parts that leave a
    part of our standard unjudged or name one it does not have.
    """
    value = entry.value
    current_parts = list(current.get("parts") or []) if isinstance(current, Mapping) else []
    current_adds = list(current.get("additions") or []) if isinstance(current, Mapping) else []
    current_reason = current.get("reason") if isinstance(current, Mapping) else None
    if "parts" in value:
        merged = {p.get("part_id"): dict(p) for p in current_parts}
        for i, part in enumerate(value["parts"]):
            owner = _owner_part(part, i)
            merged[owner["part_id"]] = owner
        judgment = canonical_judgment(
            {
                "parts": list(merged.values()),
                "additions": value.get("additions", current_adds),
                "reason": value.get("reason") or _PARTS_FALLBACK,
            },
            part_ids,
        )
        if not value.get("reason") and _same_map(judgment, current) and current_reason:
            # The current reason was written for this very part map and these
            # additions, so it still describes them; a part map the owner
            # changed (and so, possibly, the derived label) gets the owner's
            # reason or the fallback — never the reason for the old map.
            judgment["reason"] = current_reason
        return {**judgment, "basis": BASIS_OWNER, "check": None}
    if not current_parts:
        raise ValueError(
            "this text has no judgment to set the label of — give its per-part relations "
            "(value.parts) instead, or judge it first"
        )
    judgment = canonical_judgment(
        {"parts": current_parts, "additions": current_adds, "reason": current_reason}, part_ids
    )
    label = value["label"]
    out: dict[str, Any] = {**judgment, "basis": BASIS_OWNER, "check": None}
    if label == judgment["label"]:
        if value.get("reason"):
            out["reason"] = value["reason"]
        return out
    prior = current.get("owner_label") if isinstance(current, Mapping) else None
    kept = (
        prior.get("reason") if isinstance(prior, Mapping) and prior.get("label") == label else None
    )
    out["owner_label"] = {"label": label, "reason": value.get("reason") or kept or _REASON_FALLBACK}
    out["label"] = label
    return out


def _stored_verdict(vs: Mapping[str, Any]) -> dict[str, Any]:
    """The verdict-store record of an owner ``vs_standard``."""
    out = {k: vs[k] for k in ("parts", "additions", "label", "reason")}
    out["basis"] = BASIS_OWNER
    if "owner_label" in vs:
        out["owner_label"] = dict(vs["owner_label"])
    return out


def apply_overrides(
    playbook: dict[str, Any], entries: list[Override], *, strict: bool = True
) -> ApplyResult:
    """Apply *entries* to *playbook* in place; derived sections are NOT yet refreshed.

    Strict (the default, ``playbook apply-overrides``): an entry that cannot be
    applied raises. Lenient (``strict=False``, ``playbook project``): such an
    entry is left out and named in :attr:`ApplyResult.skipped`, and the rest
    are applied; a stale entry after a re-derivation is the normal case and an
    optional review file never gates a run.

    Raises:
        OverridesError: (strict only) an entry names an id the playbook does
            not carry, or a field it may not change. The playbook may be partly
            edited when this raises, so callers apply to a copy
            (:func:`fold_overrides` does).
    """
    result = ApplyResult()
    problems: list[str] = []

    raw_evidence = playbook.get("evidence")
    evidence: dict[str, Any] = raw_evidence if isinstance(raw_evidence, dict) else {}
    agreement = playbook.get("agreement_type")
    type_id = agreement.get("id") if isinstance(agreement, dict) else None
    party = perspective_party(playbook)
    slots_by_key: dict[str, list[Any]] = {}
    if any(e.target == TARGET_VS_STANDARD for e in entries) and isinstance(type_id, str):
        for slot in iter_slots(evidence):
            if not is_eligible(slot, party):
                continue
            key = slot_key(slot, type_id, party)
            if key is not None:
                slots_by_key.setdefault(key, []).append(slot)

    for entry in entries:
        if entry.target == TARGET_VS_STANDARD:
            slots = slots_by_key.get(entry.id)
            if not slots:
                problems.append(
                    f"{entry.describe()}: no judged text in this playbook has that verdict key "
                    "(the corpus changed since the page was made? remove the entry from "
                    f"{OVERRIDES_FILENAME})"
                )
                continue
            before = slots[0].entry.get("vs_standard")
            part_ids = [str(p.get("part_id")) for p in standard_parts(slots[0].clause)]
            try:
                new = _vs_standard_object(entry, before, part_ids)
            except ValueError as exc:
                problems.append(f"{entry.describe()}: {exc}")
                continue
            moved = False
            for slot in slots:
                if slot.entry.get("vs_standard") != new:
                    slot.entry["vs_standard"] = json.loads(json.dumps(new))
                    moved = True
            result.verdicts[entry.id] = _stored_verdict(new)
            if moved:
                result.changed[TARGET_VS_STANDARD] += 1
            else:
                result.unchanged += 1
        elif entry.target == TARGET_POSTURE:
            posture = playbook.get("posture")
            posture = dict(posture) if isinstance(posture, dict) else {}
            if posture.get(POSTURE_FIELD) == entry.value:
                result.unchanged += 1
                continue
            prior = posture.get("version")
            posture[POSTURE_FIELD] = entry.value
            posture["version"] = (prior if isinstance(prior, int) and prior >= 1 else 0) + 1
            posture[OWNER_MARKER] = BASIS_OWNER
            playbook["posture"] = posture
            result.changed[TARGET_POSTURE] += 1
        else:
            floor = playbook.get("floor")
            invariants = floor.get("invariants") if isinstance(floor, dict) else None
            found = next(
                (
                    inv
                    for inv in invariants or []
                    if isinstance(inv, dict) and inv.get("id") == entry.id
                ),
                None,
            )
            if found is None:
                problems.append(
                    f"{entry.describe()}: the Floor has no invariant with that id "
                    f"(remove the entry from {OVERRIDES_FILENAME})"
                )
                continue
            refusal = floor_edit_refusal(found, entry.field or "")
            if refusal is not None:
                problems.append(
                    f"{entry.describe()}: cannot override the {entry.field} of "
                    f"{entry.id!r}: {refusal} (remove the entry from {OVERRIDES_FILENAME})"
                )
                continue
            if found.get(entry.field or "") == entry.value:
                result.unchanged += 1
                continue
            found[entry.field or ""] = entry.value
            found[OWNER_MARKER] = BASIS_OWNER
            result.changed[TARGET_FLOOR] += 1
    if problems:
        if strict:
            raise OverridesError(problems)
        result.skipped.extend(problems)
    return result


def refresh_after_edit(playbook: dict[str, Any]) -> None:
    """Recompute everything that is a function of the edited content, in place.

    The ``precedent.jsonl`` content address, the digest, the manifest, dossiers
    and provenance index, and ``identity.content_hash`` / ``section_digests`` —
    in the order the assembler builds them.
    """
    if SIDECARS_KEY in playbook:
        playbook[SIDECARS_KEY] = precedent_sidecar_manifest(playbook)
    if "digest" in playbook:
        playbook["digest"] = build_digest_v4(playbook)
    refresh_derived_sections(playbook)
    if isinstance(playbook.get("identity"), dict):
        playbook["identity"]["content_hash"] = content_hash(playbook)
        playbook["identity"]["section_digests"] = compute_section_digests(playbook)


class _OverlayStore:
    """A verdict store seen through pending owner verdicts (nothing is written yet)."""

    def __init__(self, base: VerdictStore | None, pending: Mapping[str, dict[str, Any]]) -> None:
        self._base = base
        self._pending = {k: StoredVerdict(verdict=dict(v)) for k, v in pending.items()}

    def get_record_by_key(self, key: str) -> StoredVerdict | None:
        if key in self._pending:
            return self._pending[key]
        return self._base.get_record_by_key(key) if self._base is not None else None


def _validation_problems(result: ValidationResult) -> list[str]:
    return [f"the edited playbook no longer validates: {e}" for e in result.errors if e.blocking]


def _store_path(out_dir: Path) -> Path:
    return out_dir / "judge" / "verdicts.jsonl"


def fold_overrides(
    playbook: dict[str, Any],
    entries: list[Override],
    out_dir: Path,
    *,
    strict: bool = True,
) -> tuple[dict[str, Any], ApplyResult]:
    """The playbook with *entries* applied, re-derived and validated; owner verdicts stored.

    Pure with respect to *playbook* (a deep copy is edited) and to the disk
    until everything has validated; only then are the owner verdicts appended
    to the out-dir's verdict store (when it has one). The caller writes the
    playbook (and its sidecar).

    Lenient (``strict=False``): entries that cannot be applied are reported in
    ``result.skipped``; if the rest leave the playbook invalid, none is applied
    (the unedited *playbook* is returned) and that is reported too.

    Raises:
        OverridesError: (strict only) an unknown id, or an edit that leaves the
            playbook invalid.
    """
    edited = copy.deepcopy(playbook)
    result = apply_overrides(edited, entries, strict=strict)
    if result.n_changed:
        refresh_after_edit(edited)
    store_path = _store_path(out_dir)
    base = VerdictStore(store_path) if store_path.is_file() else None
    overlay = _OverlayStore(base, result.verdicts) if base is not None else None
    problems = _validation_problems(validate_document(edited, verdict_store=overlay))
    if problems:
        if strict:
            raise OverridesError(problems)
        return playbook, ApplyResult(skipped=[*result.skipped, *problems])
    if base is not None:
        stamp = RubricStamp(kind=EQUIVALENCE_KIND, version=rubric_version(EQUIVALENCE_KIND))
        for key, verdict in sorted(result.verdicts.items()):
            current = base.get_record_by_key(key)
            if current is not None and current.verdict == verdict:
                continue
            base.put_by_key(key, verdict, rubric=stamp)
    return edited, result


def apply_overrides_to_dir(out_dir: Path, overrides_path: Path | None = None) -> ApplyResult:
    """``playbook apply-overrides``: fold ``<out>/overrides.json`` into ``<out>/playbook.opf.json``.

    Rewrites the playbook (and ``precedent.jsonl``) only when an entry changed
    something; re-applying an applied file is a no-op.

    Raises:
        FileNotFoundError: no ``playbook.opf.json`` / ``overrides.json``.
        OverridesError: see :func:`fold_overrides`; nothing is written.
    """
    opf_path = out_dir / "playbook.opf.json"
    path = overrides_path or out_dir / OVERRIDES_FILENAME
    if not opf_path.is_file():
        raise FileNotFoundError(f"{opf_path} not found — nothing to apply the overrides to")
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found — there are no overrides to apply")
    entries = load_overrides(path)
    try:
        playbook = json.loads(opf_path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise OverridesError([f"{opf_path} is not valid JSON: {exc}"]) from exc
    edited, result = fold_overrides(playbook, entries, out_dir)
    if result.n_changed:
        write_playbook(edited, opf_path)
        write_precedent_sidecar(edited, opf_path)
    return result
