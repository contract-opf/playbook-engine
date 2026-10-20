"""The ``vs_standard`` equivalence judgment — OPF 0.6 (issues #240, #243).

The question a legal reviewer asks of every precedent is how the signed
language differs from our standard, and whether that is more or less risky for
us. The exact-match ``standard`` fact (issue #220) answers only "did they sign
our words unchanged"; on counterparty paper it is almost always false even
where the concept is the same. So each DISTINCT non-standard text in the
precedent record (a deal's signed text, a non-standard opening, a refused ask)
is judged ONCE against our standard and the answer is carried on the text as
``vs_standard``.

Part by part (issue #243)
-------------------------

Our standard for a clause type is an ordered list of **parts**: the template
passages of the type in template order, form front matter excluded
(``our_standard.parts[]``, :mod:`playbook_engine.template_standards`). Each
part has a stable ``part_id`` — a short content hash of its normalized text
(:func:`part_id_of`), so it survives a template edit elsewhere. The judge does
not pick one label for the whole text; it says, for EVERY part, whether the
candidate keeps it (``present``, with a ``relation``) or drops it
(``omitted``, with the ``effect`` of the omission), and lists what the
candidate ADDS that no part covers::

    {"parts": [{"part_id", "status": "present" | "omitted",
                "relation": "equivalent" | "more_protective" | "less_protective",  # present
                "effect": "more_protective" | "less_protective" | "neutral",       # omitted
                "reason": "<one sentence; only when not equivalent / neutral>"}],
     "additions": [{"summary": "<one sentence>",
                    "effect": "more_protective" | "less_protective" | "neutral"}],
     "label": <DERIVED by derive_label, never judged>,
     "reason": "<one sentence: the single most important difference>",
     "basis": "judge" | "agent" | "owner",
     "check": {agreed, label_agreed, parts_agreed, by, adjudicated} | null,
     "owner_label": {"label", "reason"}       # optional, basis "owner" only
    }

Directions are always relative to ``perspective.party``: omitting an
obligation that falls on us can be more protective, and the judge says which.
The ``label`` (:func:`derive_label`, a pure function) is ``different_concept``
when no part is present, ``equivalent`` when every present part is equivalent
and every omission and addition is neutral, ``more_protective`` /
``less_protective`` when every non-neutral effect points one way, and
``mixed`` when they point both ways. Facts are recorded once (the per-part
relations); the label is their meaning. An owner may override that meaning
(``owner_label``), and then ``label`` is the owner's.

This narrows owner decision (c) of issue #227 ("no judged verdicts on the
consumer path") for this one judgment only. It is an index the consuming
model can check against the cited text, never an instruction; ``null`` means
"not yet judged" and is never a guess.

This module is the pure, store-free half: part ids, the derived label and the
part map, the cache key, the judge payload, which texts are eligible, and
writing judgments onto an ``evidence`` dict. The store-backed judge (queue on
miss, replay on hit) is
:class:`playbook_engine.agent_judge.StoreBackedEquivalenceJudge`; the blind
check and adjudication round is :mod:`playbook_engine.equivalence_check`.

Cache key
---------

``sha256`` over the JSON array ``[agreement_type.id, taxonomy_id,
perspective.party, grouping_key(text), [grouping_key(part) for each part of
our standard, in order]]`` where ``grouping_key`` is the OPF-SPEC §3.5.4
grouping key (:func:`~playbook_engine.precedent.normalize_variant_text`).
Never the paper side, never a document id, never the opening draft, never the
role: one judgment per distinct text against one ordered list of parts,
reused by every deal and every role (signed, opening, refused) that carries
it. The judge payload is likewise free of anything that identifies a deal, a
counterparty or a paper side, so the answer cannot depend on them.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

__all__ = [
    "ADDITION_TOKENS",
    "BASES",
    "BASIS_OWNER",
    "EFFECTS",
    "EQUIVALENCE_KIND",
    "LABELS",
    "PART_MAP_LEGEND",
    "PART_TOKENS",
    "RELATIONS",
    "ROLES",
    "ROLE_OPENING",
    "ROLE_REFUSED",
    "ROLE_SIGNED",
    "STANDARD_PART_SEPARATOR",
    "STATUSES",
    "LABEL_BY_OWNER",
    "LABEL_BY_PARTS",
    "EquivalenceSubject",
    "Slot",
    "addition_map",
    "agreement_signature",
    "canonical_judgment",
    "check_owner_label",
    "collect_subjects",
    "derive_label",
    "equivalence_key",
    "is_eligible",
    "iter_slots",
    "label_by",
    "label_evidence",
    "label_reason",
    "part_id_of",
    "part_ids_of",
    "part_map",
    "slot_key",
    "standard_part_texts",
    "standard_parts",
    "stamp_standard_parts",
    "summarize",
    "validate_draft_verdict",
    "vs_standard_of",
]

#: The pending-item / rubric kind of the equivalence judge.
EQUIVALENCE_KIND = "equivalence"

LABEL_EQUIVALENT = "equivalent"
LABEL_MORE = "more_protective"
LABEL_LESS = "less_protective"
LABEL_MIXED = "mixed"
LABEL_DIFFERENT = "different_concept"

#: ``vs_standard.label`` vocabulary, relative to ``perspective.party``. DERIVED
#: from the per-part relations (:func:`derive_label`), never judged directly.
LABELS = (LABEL_EQUIVALENT, LABEL_MORE, LABEL_LESS, LABEL_MIXED, LABEL_DIFFERENT)

STATUS_PRESENT = "present"
STATUS_OMITTED = "omitted"
#: ``vs_standard.parts[].status``: does the candidate keep this part of our standard?
STATUSES = (STATUS_PRESENT, STATUS_OMITTED)

EFFECT_NEUTRAL = "neutral"
#: ``relation`` of a present part: how the candidate's version compares with it.
RELATIONS = (LABEL_EQUIVALENT, LABEL_MORE, LABEL_LESS)
#: ``effect`` of an omitted part, and of an addition, for ``perspective.party``.
EFFECTS = (LABEL_MORE, LABEL_LESS, EFFECT_NEUTRAL)

#: One token per part in the compact part map (digest, dossiers, Review tab).
PART_TOKENS: dict[tuple[str, str], str] = {
    (STATUS_PRESENT, LABEL_EQUIVALENT): "=",
    (STATUS_PRESENT, LABEL_MORE): "+",
    (STATUS_PRESENT, LABEL_LESS): "-",
    (STATUS_OMITTED, LABEL_MORE): "x+",
    (STATUS_OMITTED, LABEL_LESS): "x-",
    (STATUS_OMITTED, EFFECT_NEUTRAL): "x0",
}

#: One token per addition (something the candidate adds that no part covers).
ADDITION_TOKENS: dict[str, str] = {LABEL_MORE: "+", LABEL_LESS: "-", EFFECT_NEUTRAL: "0"}

#: What each token means — carried in the digest so the map reads on its own.
PART_MAP_LEGEND: dict[str, dict[str, str]] = {
    "part_map": {
        "=": "our part is present with the same legal effect",
        "+": "our part is present but more protective for perspective.party",
        "-": "our part is present but less protective for perspective.party",
        "x+": "our part is omitted, which is more protective for perspective.party",
        "x-": "our part is omitted, which is less protective for perspective.party",
        "x0": "our part is omitted with no effect on perspective.party",
    },
    "additions": {
        "+": "an addition no part covers, more protective for perspective.party",
        "-": "an addition no part covers, less protective for perspective.party",
        "0": "an addition no part covers, neutral for perspective.party",
    },
    "label_by": {
        "parts": "the default, so a digest entry omits label_by: the label is derived from "
        "the part map and additions; reason is the judgment's most important difference",
        "owner": "the owner set the label over the judged part map and additions, which may "
        "imply another label; reason is the owner's and wins",
    },
}

#: ``label_by`` of a judged text: the label is derived from its parts, or the
#: owner set it over them (``owner_label``).
LABEL_BY_PARTS = "parts"
LABEL_BY_OWNER = "owner"

#: Joins the parts of our standard into ``our_standard.text`` (the same
#: separator ``template_standards`` joins template nodes with).
STANDARD_PART_SEPARATOR = "\n\n"

#: Hex characters of a ``part_id`` (a second part with the same normalized
#: text in one standard gets ``-2``, a third ``-3``, ...).
PART_ID_LEN = 8

BASIS_JUDGE = "judge"
BASIS_AGENT = "agent"
BASIS_OWNER = "owner"
#: ``vs_standard.basis`` vocabulary. ``owner`` (an after-the-fact correction)
#: wins over any agent or check answer.
BASES = (BASIS_JUDGE, BASIS_AGENT, BASIS_OWNER)

ROLE_SIGNED = "signed"
ROLE_OPENING = "opening"
ROLE_REFUSED = "refused"
ROLES = (ROLE_SIGNED, ROLE_OPENING, ROLE_REFUSED)

#: Verdict keys a producer-supplied draft may carry. ``label`` is optional
#: (it is derived; a supplied one must match). ``check`` and the audit trail
#: are written only by ``judge-apply --check``.
_DRAFT_KEYS = frozenset({"parts", "additions", "reason", "basis", "label"})
_PART_KEYS = frozenset({"part_id", "status", "relation", "effect", "reason"})
_ADDITION_KEYS = frozenset({"summary", "effect"})


# ---------------------------------------------------------------------------
# Standard parts
# ---------------------------------------------------------------------------


def _grouping_key(text: str, party: str | None) -> str:
    from playbook_engine.precedent import normalize_variant_text  # noqa: PLC0415

    return normalize_variant_text(text, party=party)


def part_id_of(text: str, party: str | None) -> str:
    """The content hash of one part of our standard: the first
    :data:`PART_ID_LEN` hex characters of ``sha256`` over its §3.5.4 grouping
    key. Two parts with the same normalized text hash alike (see
    :func:`part_ids_of`)."""
    return hashlib.sha256(_grouping_key(text, party).encode("utf-8")).hexdigest()[:PART_ID_LEN]


def part_ids_of(texts: Sequence[str], party: str | None) -> list[str]:
    """The ``part_id`` of each part of one standard, in order.

    The n-th part (n >= 2) whose content hash an earlier part of the same
    standard already has gets ``<hash>-<n>``, so every id is unique within the
    standard and still survives an edit elsewhere in the template.
    """
    seen: dict[str, int] = {}
    ids: list[str] = []
    for text in texts:
        base = part_id_of(text, party)
        seen[base] = seen.get(base, 0) + 1
        ids.append(base if seen[base] == 1 else f"{base}-{seen[base]}")
    return ids


def stamp_standard_parts(our_standard: dict[str, Any] | None, party: str | None) -> None:
    """(Re)compute ``part_id`` and ``order`` of every part of *our_standard*, in place.

    ``our_standard.text`` is the parts joined in order
    (:data:`STANDARD_PART_SEPARATOR`). Called by ``precedent.restamp_evidence``
    so the ids are always those of the shipped (invisible-character-stripped)
    text.
    """
    if not isinstance(our_standard, dict):
        return
    parts = our_standard.get("parts")
    if not isinstance(parts, list):
        return
    texts = [str(p.get("text", "")) for p in parts if isinstance(p, dict)]
    for i, (part, pid) in enumerate(zip(parts, part_ids_of(texts, party), strict=True)):
        part["part_id"] = pid
        part["order"] = i + 1


def standard_parts(clause: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """The ``our_standard.parts`` of an evidence clause (``[]`` when none)."""
    std = clause.get("our_standard") if isinstance(clause, Mapping) else None
    parts = std.get("parts") if isinstance(std, Mapping) else None
    return [p for p in parts if isinstance(p, dict)] if isinstance(parts, list) else []


def standard_part_texts(clause: Mapping[str, Any] | None) -> list[str]:
    """The texts of the parts of an evidence clause's standard, in order."""
    return [str(p.get("text", "")) for p in standard_parts(clause)]


# ---------------------------------------------------------------------------
# The derived label and the part map
# ---------------------------------------------------------------------------


def _direction(part: Mapping[str, Any]) -> str | None:
    return part.get("relation") if part.get("status") == STATUS_PRESENT else part.get("effect")


def derive_label(parts: Sequence[Mapping[str, Any]], additions: Sequence[Mapping[str, Any]]) -> str:
    """The ``vs_standard.label`` the per-part relations imply (a pure function).

    - ``different_concept``: no part of our standard is present;
    - ``equivalent``: at least one part is present, every present part is
      equivalent, and every omission and every addition is neutral;
    - ``more_protective`` / ``less_protective``: every non-equivalent,
      non-neutral effect points that way;
    - ``mixed``: they point both ways.
    """
    if not any(p.get("status") == STATUS_PRESENT for p in parts):
        return LABEL_DIFFERENT
    directions = {_direction(p) for p in parts} | {a.get("effect") for a in additions}
    directions &= {LABEL_MORE, LABEL_LESS}
    if not directions:
        return LABEL_EQUIVALENT
    if len(directions) == 2:
        return LABEL_MIXED
    return LABEL_MORE if LABEL_MORE in directions else LABEL_LESS


def part_map(vs: Mapping[str, Any]) -> list[str]:
    """One token per part of *vs* (in standard order), see :data:`PART_TOKENS`."""
    return [
        PART_TOKENS.get((str(p.get("status")), str(_direction(p))), "?")
        for p in vs.get("parts") or []
        if isinstance(p, Mapping)
    ]


def addition_map(vs: Mapping[str, Any]) -> list[str]:
    """One token per addition of *vs*, see :data:`ADDITION_TOKENS`."""
    return [
        ADDITION_TOKENS.get(str(a.get("effect")), "?")
        for a in vs.get("additions") or []
        if isinstance(a, Mapping)
    ]


def _owner_label(vs: Mapping[str, Any]) -> Mapping[str, Any] | None:
    owner = vs.get("owner_label")
    return owner if isinstance(owner, Mapping) else None


def label_by(vs: Mapping[str, Any]) -> str:
    """Who set the label of judgment *vs*: :data:`LABEL_BY_OWNER` when the
    owner overrode the derived label (``owner_label``), else
    :data:`LABEL_BY_PARTS` (the label is derived from the parts)."""
    return LABEL_BY_OWNER if _owner_label(vs) is not None else LABEL_BY_PARTS


def label_reason(vs: Mapping[str, Any]) -> Any:
    """The reason a model-facing projection carries beside the label of *vs*.

    The owner's ``owner_label.reason`` when the owner set the label (the
    judgment's own reason explains the derived label the owner overrode, so
    it would contradict the label shown), else the judgment's ``reason``.
    """
    owner = _owner_label(vs)
    if owner is not None and isinstance(owner.get("reason"), str):
        return owner["reason"]
    return vs.get("reason")


def agreement_signature(judgment: Mapping[str, Any]) -> frozenset[tuple[str, str, str]]:
    """The set of ``(part_id, status, direction)`` of every part of *judgment*
    whose relation is not ``equivalent`` or whose effect is not ``neutral``.

    Two judgments agree on the parts when their signatures are equal: a part
    both call equivalent or a neutral omission is no disagreement, but the
    same direction reached by keeping a part in one and dropping it in the
    other is (the part map shows them differently).
    """
    return frozenset(
        (str(p.get("part_id")), str(p.get("status")), str(_direction(p)))
        for p in judgment.get("parts") or []
        if isinstance(p, Mapping) and _direction(p) in (LABEL_MORE, LABEL_LESS)
    )


# ---------------------------------------------------------------------------
# Judgment shapes
# ---------------------------------------------------------------------------


def _nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _canonical_part(part: Any, i: int) -> dict[str, Any]:
    where = f"parts[{i}]"
    if not isinstance(part, Mapping):
        raise ValueError(f"{where} must be an object")
    extra = sorted(set(part) - _PART_KEYS)
    if extra:
        raise ValueError(f"{where}: unexpected field(s) {extra}")
    pid = part.get("part_id")
    if not _nonempty(pid):
        raise ValueError(f"{where}: 'part_id' must be the id of a part of our standard")
    status = part.get("status")
    if status not in STATUSES:
        raise ValueError(f"{where}: 'status' must be one of {list(STATUSES)}, got {status!r}")
    out: dict[str, Any] = {"part_id": pid, "status": status}
    if status == STATUS_PRESENT:
        if "effect" in part:
            raise ValueError(f"{where}: a present part takes a 'relation', not an 'effect'")
        relation = part.get("relation")
        if relation not in RELATIONS:
            raise ValueError(
                f"{where}: a present part needs a 'relation', one of {list(RELATIONS)}, "
                f"got {relation!r}"
            )
        out["relation"] = relation
        notable = relation != LABEL_EQUIVALENT
    else:
        if "relation" in part:
            raise ValueError(f"{where}: an omitted part takes an 'effect', not a 'relation'")
        effect = part.get("effect")
        if effect not in EFFECTS:
            raise ValueError(
                f"{where}: an omitted part needs an 'effect', one of {list(EFFECTS)}, "
                f"got {effect!r}"
            )
        out["effect"] = effect
        notable = effect != EFFECT_NEUTRAL
    reason = part.get("reason")
    if notable:
        if not _nonempty(reason):
            raise ValueError(
                f"{where}: a part that is not equivalent (or an omission that is not neutral) "
                "needs a one-sentence 'reason'"
            )
        out["reason"] = reason
    elif reason is not None:
        raise ValueError(
            f"{where}: 'reason' is written only for a part that is not equivalent or an "
            "omission that is not neutral"
        )
    return out


def _canonical_addition(addition: Any, i: int) -> dict[str, Any]:
    where = f"additions[{i}]"
    if not isinstance(addition, Mapping):
        raise ValueError(f"{where} must be an object")
    extra = sorted(set(addition) - _ADDITION_KEYS)
    if extra:
        raise ValueError(f"{where}: unexpected field(s) {extra}")
    if not _nonempty(addition.get("summary")):
        raise ValueError(f"{where}: 'summary' must be one sentence naming the addition")
    effect = addition.get("effect")
    if effect not in EFFECTS:
        raise ValueError(f"{where}: 'effect' must be one of {list(EFFECTS)}, got {effect!r}")
    return {"summary": addition["summary"], "effect": effect}


def canonical_judgment(
    judgment: Mapping[str, Any], part_ids: Sequence[str] | None = None
) -> dict[str, Any]:
    """Validate a structured judgment and return ``{parts, additions, label, reason}``.

    ``parts`` must name every part of our standard exactly once (*part_ids*,
    in standard order, when known: the result is ordered the same way);
    ``additions`` is a list (possibly empty); ``reason`` one sentence. The
    ``label`` is derived (:func:`derive_label`); a supplied one must equal it.
    Raises ``ValueError`` naming the first problem.
    """
    raw_parts = judgment.get("parts")
    if not isinstance(raw_parts, list) or not raw_parts:
        raise ValueError("'parts' must be a non-empty list, one entry per part of our standard")
    parts = [_canonical_part(p, i) for i, p in enumerate(raw_parts)]
    ids = [p["part_id"] for p in parts]
    dupes = sorted({pid for pid in ids if ids.count(pid) > 1})
    if dupes:
        raise ValueError(f"part(s) {dupes} judged more than once")
    if part_ids is not None:
        unknown = sorted(set(ids) - set(part_ids))
        if unknown:
            raise ValueError(f"unknown part_id(s) {unknown}: not parts of our standard")
        missing = [pid for pid in part_ids if pid not in ids]
        if missing:
            raise ValueError(f"part(s) {missing} of our standard not judged")
        by_id = {p["part_id"]: p for p in parts}
        parts = [by_id[pid] for pid in part_ids]
    raw_additions = judgment.get("additions")
    if not isinstance(raw_additions, list):
        raise ValueError("'additions' must be a list (empty when the candidate adds nothing)")
    additions = [_canonical_addition(a, i) for i, a in enumerate(raw_additions)]
    reason = judgment.get("reason")
    if not _nonempty(reason):
        raise ValueError("'reason' must be one sentence naming the most important difference")
    label = derive_label(parts, additions)
    supplied = judgment.get("label")
    if supplied is not None and supplied != label:
        raise ValueError(
            f"'label' {supplied!r} is not what the parts imply ({label!r}); the label is "
            "derived from the parts and additions — omit it or correct the parts"
        )
    return {"parts": parts, "additions": additions, "label": label, "reason": reason}


def check_owner_label(value: Any) -> dict[str, str]:
    """Validate an ``owner_label`` (``{label, reason}``) and return it."""
    if not isinstance(value, Mapping) or set(value) != {"label", "reason"}:
        raise ValueError("'owner_label' must be {label, reason}")
    if value["label"] not in LABELS:
        raise ValueError(f"owner_label.label must be one of {list(LABELS)}")
    if not _nonempty(value["reason"]):
        raise ValueError("owner_label.reason must be a non-empty sentence")
    return {"label": value["label"], "reason": value["reason"]}


def validate_draft_verdict(
    verdict: dict[str, Any], part_ids: Sequence[str] | None = None
) -> dict[str, Any]:
    """Validate a producer-supplied equivalence verdict (``judge-apply``).

    ``{parts, additions, reason, basis[, label]}`` and nothing else (see the
    module docstring): ``basis`` is required (``"agent"`` for a coder acting
    as the judge, ``"judge"`` for the store-backed judge, ``"owner"`` for an
    after-the-fact correction), the label is derived, and ``check`` is
    written only by ``judge-apply --check``. Returns the verdict as it is
    stored (the derived ``label`` included).
    """
    extra = sorted(set(verdict) - _DRAFT_KEYS)
    if extra:
        raise ValueError(
            f"unexpected field(s) {extra} on an equivalence verdict; a verdict carries only "
            "parts, additions, reason and basis (the label is derived; checks are recorded "
            "with `judge-apply --check`)"
        )
    basis = verdict.get("basis")
    if basis not in BASES:
        raise ValueError(f"'basis' must be one of {list(BASES)}, got {basis!r}")
    return {**canonical_judgment(verdict, part_ids), "basis": basis}


def _check_of(check: Any) -> dict[str, Any] | None:
    if check is None:
        return None
    if (
        not isinstance(check, Mapping)
        or not all(
            isinstance(check.get(k), bool)
            for k in ("agreed", "label_agreed", "parts_agreed", "adjudicated")
        )
        or not _nonempty(check.get("by"))
    ):
        raise ValueError(
            "'check' must be null or {agreed, label_agreed, parts_agreed, by, adjudicated}"
        )
    return {
        "agreed": check["agreed"],
        "label_agreed": check["label_agreed"],
        "parts_agreed": check["parts_agreed"],
        "by": check["by"],
        "adjudicated": check["adjudicated"],
    }


def vs_standard_of(
    verdict: Mapping[str, Any], part_ids: Sequence[str] | None = None
) -> dict[str, Any]:
    """The ``vs_standard`` object a stored *verdict* projects to.

    *part_ids* are the ids of our standard's parts, in order, when known: a
    stored judgment about other parts is malformed. Raises ``ValueError`` on
    a malformed record (it is then treated as unjudged, never replayed).
    """
    judgment = canonical_judgment(
        {k: verdict.get(k) for k in ("parts", "additions", "reason")}, part_ids
    )
    basis = verdict.get("basis")
    if basis not in BASES:
        raise ValueError(f"'basis' must be one of {list(BASES)}, got {basis!r}")
    out: dict[str, Any] = {
        "label": judgment["label"],
        "reason": judgment["reason"],
        "parts": judgment["parts"],
        "additions": judgment["additions"],
        "basis": basis,
        "check": _check_of(verdict.get("check")),
    }
    if verdict.get("owner_label") is not None:
        if basis != BASIS_OWNER:
            raise ValueError("'owner_label' is only set by the owner (basis 'owner')")
        owner = check_owner_label(verdict["owner_label"])
        out["owner_label"] = owner
        out["label"] = owner["label"]
    stored_label = verdict.get("label")
    if stored_label is not None and stored_label != out["label"]:
        raise ValueError(
            f"stored label {stored_label!r} is not what the parts imply ({out['label']!r})"
        )
    return out


# ---------------------------------------------------------------------------
# Key and payload
# ---------------------------------------------------------------------------


def equivalence_key(
    agreement_type_id: str,
    taxonomy_id: str,
    party: str | None,
    text: str,
    standard_parts: Sequence[str],
) -> str:
    """The verdict-store key of *text* judged against the parts of our standard.

    See the module docstring: the agreement type, the clause type, the party,
    the text and the ORDERED list of our standard's parts — no paper side,
    no document id, no role. ``party`` is ``perspective.party`` (``None``
    when the playbook has no perspective); two texts with the same §3.5.4
    grouping key share a key.
    """
    if isinstance(standard_parts, str):
        raise TypeError("standard_parts is the list of part texts, not one string")
    raw = json.dumps(
        [
            agreement_type_id,
            taxonomy_id,
            party or "",
            _grouping_key(text, party),
            [_grouping_key(part, party) for part in standard_parts],
        ],
        ensure_ascii=False,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class EquivalenceSubject:
    """One distinct text awaiting (or carrying) an equivalence verdict.

    ``payload`` is exactly what a drafter or a blind checker sees: the
    taxonomy id and title, the perspective party, the numbered parts of our
    standard, the candidate text and the roles it plays. No deal,
    counterparty, paper side or outcome.
    """

    key: str
    payload: dict[str, Any]
    roles: tuple[str, ...]

    @property
    def part_ids(self) -> list[str]:
        """The ids of our standard's parts, in order."""
        return [str(p["part_id"]) for p in self.payload.get("our_standard_parts") or []]


# ---------------------------------------------------------------------------
# Slots — the places in evidence.precedent a judgment can sit
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Slot:
    """One text entry of ``evidence.precedent`` that may carry ``vs_standard``."""

    role: str
    clause: dict[str, Any] | None
    record: dict[str, Any]
    #: The ``{text, ref}`` (or refused-ask) dict the judgment is written onto.
    entry: dict[str, Any]


def iter_slots(evidence: Mapping[str, Any]) -> Iterator[Slot]:
    """Every text entry of *evidence* that has a ``vs_standard`` field.

    ``signed_text`` and a non-null ``opening_text`` of each precedent record,
    and each ``refused_asks[]`` entry, in record order.
    """
    clauses = {
        c.get("taxonomy_id"): c for c in evidence.get("clauses") or [] if isinstance(c, dict)
    }
    for record in evidence.get("precedent") or []:
        if not isinstance(record, dict):
            continue
        clause = clauses.get(record.get("taxonomy_id"))
        signed = record.get("signed_text")
        if isinstance(signed, dict):
            yield Slot(ROLE_SIGNED, clause, record, signed)
        opening = record.get("opening_text")
        if isinstance(opening, dict):
            yield Slot(ROLE_OPENING, clause, record, opening)
        for ask in record.get("refused_asks") or []:
            if isinstance(ask, dict):
                yield Slot(ROLE_REFUSED, clause, record, ask)


def _standard_text(clause: dict[str, Any] | None) -> str | None:
    std = clause.get("our_standard") if isinstance(clause, dict) else None
    text = std.get("text") if isinstance(std, dict) else None
    return text if isinstance(text, str) and text.strip() else None


def slot_key(slot: Slot, agreement_type_id: str, party: str | None) -> str | None:
    """The verdict key of *slot*, or ``None`` when it has no standard or text."""
    standard = _standard_text(slot.clause)
    parts = standard_part_texts(slot.clause)
    text = slot.entry.get("text")
    if standard is None or not parts or not isinstance(text, str) or not text.strip():
        return None
    taxonomy_id = slot.record.get("taxonomy_id")
    if not isinstance(taxonomy_id, str):
        return None
    return equivalence_key(agreement_type_id, taxonomy_id, party, text, parts)


def is_eligible(slot: Slot, party: str | None) -> bool:
    """Whether *slot*'s text is ever judged against our standard.

    Never judged (and never queued): a text with no ``our_standard`` to be
    compared with (emergent mode); a signed text whose ``standard`` fact is
    true or whose grouping key equals our standard's (an exact match is
    equivalent by definition and stays ``null``); an opening that is not
    ``non_standard``; an empty text.
    """
    standard = _standard_text(slot.clause)
    text = slot.entry.get("text")
    if standard is None or not isinstance(text, str):
        return False
    text_key = _grouping_key(text, party)
    if not text_key or text_key == _grouping_key(standard, party):
        return False
    if slot.role == ROLE_SIGNED:
        return slot.record.get("standard") is not True
    if slot.role == ROLE_OPENING:
        return slot.record.get("opened_with") == "non_standard"
    return True


def collect_subjects(
    evidence: Mapping[str, Any], agreement_type_id: str, party: str | None
) -> list[EquivalenceSubject]:
    """The distinct texts of *evidence* to judge, in first-seen order.

    Same text in two deals, or in two roles, is one subject: the §3.5.4
    grouping key decides. ``roles`` lists every role the text plays. The
    payload numbers the parts of our standard (``n``, from 1) and gives each
    its ``part_id``, which the answer must use.
    """
    by_key: dict[str, dict[str, Any]] = {}
    for slot in iter_slots(evidence):
        if not is_eligible(slot, party):
            continue
        key = slot_key(slot, agreement_type_id, party)
        if key is None:
            continue
        found = by_key.get(key)
        if found is None:
            clause = slot.clause or {}
            by_key[key] = {
                "payload": {
                    "stage": EQUIVALENCE_KIND,
                    "agreement_type_id": agreement_type_id,
                    "taxonomy_id": slot.record.get("taxonomy_id"),
                    "taxonomy_title": clause.get("title") or "",
                    "perspective_party": party,
                    "our_standard_parts": [
                        {"n": i + 1, "part_id": p.get("part_id"), "text": p.get("text")}
                        for i, p in enumerate(standard_parts(clause))
                    ],
                    "candidate": slot.entry["text"],
                },
                "roles": {slot.role},
            }
        else:
            found["roles"].add(slot.role)
    subjects: list[EquivalenceSubject] = []
    for key, found in by_key.items():
        roles = tuple(r for r in ROLES if r in found["roles"])
        payload = dict(found["payload"])
        payload["roles"] = list(roles)
        subjects.append(EquivalenceSubject(key=key, payload=payload, roles=roles))
    return subjects


# ---------------------------------------------------------------------------
# Writing judgments onto evidence
# ---------------------------------------------------------------------------


def label_evidence(
    evidence: dict[str, Any],
    agreement_type_id: str,
    party: str | None,
    vs_by_key: Mapping[str, dict[str, Any] | None],
) -> None:
    """Set ``vs_standard`` on every text entry of *evidence*, in place.

    An eligible text takes the judgment *vs_by_key* holds for its key
    (``None`` when absent: not yet judged, never guessed); every other text —
    an exact match with our standard, a clause with no ``our_standard`` — is
    ``None``.
    """
    for slot in iter_slots(evidence):
        vs: dict[str, Any] | None = None
        if is_eligible(slot, party):
            key = slot_key(slot, agreement_type_id, party)
            vs = vs_by_key.get(key) if key is not None else None
        slot.entry["vs_standard"] = json.loads(json.dumps(vs)) if vs is not None else None


# ---------------------------------------------------------------------------
# Counts (scorecard, project report)
# ---------------------------------------------------------------------------


def summarize(
    evidence: Mapping[str, Any], agreement_type_id: str, party: str | None
) -> dict[str, Any]:
    """Counts-only summary of the judgments in *evidence*.

    Per role: ``eligible`` and ``labelled`` DISTINCT texts (a text that plays
    two roles counts once in each), ``unjudged``, and the verdict-state counts
    ``drafted`` (every judgment), ``checked`` (a check is recorded),
    ``agreed``, ``adjudicated``, ``unchecked`` and — for a verdict that
    disagreed with its check and still awaits adjudication — ``disputed``.
    ``totals`` counts distinct texts across roles; ``by_label`` is the label
    histogram of those; ``agreement_rate`` is agreed / checked, ``None`` when
    nothing is checked, and ``label_agreement_rate`` / ``parts_agreement_rate``
    split it: the checker reached the same label, and the same non-equivalent
    parts (:func:`agreement_signature`). ``part_tokens`` and
    ``addition_tokens`` count the part-map tokens of the labelled texts, and
    ``standard_parts`` the parts of each clause's standard.
    """
    per_role: dict[str, dict[str, set[str]]] = {
        r: {
            "eligible": set(),
            "labelled": set(),
            "checked": set(),
            "agreed": set(),
            "adjudicated": set(),
            "disputed": set(),
        }
        for r in ROLES
    }
    judgments: dict[str, Mapping[str, Any]] = {}
    for slot in iter_slots(evidence):
        if not is_eligible(slot, party):
            continue
        key = slot_key(slot, agreement_type_id, party)
        if key is None:
            continue
        bucket = per_role[slot.role]
        bucket["eligible"].add(key)
        vs = slot.entry.get("vs_standard")
        if not isinstance(vs, dict):
            continue
        bucket["labelled"].add(key)
        judgments[key] = vs
        check = vs.get("check")
        if isinstance(check, dict):
            bucket["checked"].add(key)
            if check.get("adjudicated") is True:
                bucket["adjudicated"].add(key)
            elif check.get("agreed") is True:
                bucket["agreed"].add(key)
            else:
                bucket["disputed"].add(key)

    def _row(sets: dict[str, set[str]]) -> dict[str, int]:
        eligible, labelled, checked = (
            len(sets["eligible"]),
            len(sets["labelled"]),
            len(sets["checked"]),
        )
        return {
            "eligible": eligible,
            "unjudged": eligible - labelled,
            "drafted": labelled,
            "checked": checked,
            "agreed": len(sets["agreed"]),
            "adjudicated": len(sets["adjudicated"]),
            "disputed": len(sets["disputed"]),
            "unchecked": labelled - checked,
        }

    union: dict[str, set[str]] = {
        name: set().union(*(per_role[r][name] for r in ROLES)) for name in per_role[ROLES[0]]
    }
    totals = _row(union)
    by_label = {
        label: sum(1 for key in union["labelled"] if judgments[key].get("label") == label)
        for label in LABELS
    }
    checks = [
        judgments[key]["check"]
        for key in union["checked"]
        if isinstance(judgments[key].get("check"), dict)
    ]
    label_agreed = sum(1 for c in checks if c.get("label_agreed") is True)
    parts_agreed = sum(1 for c in checks if c.get("parts_agreed") is True)
    part_tokens = dict.fromkeys(PART_TOKENS.values(), 0)
    addition_tokens = dict.fromkeys(ADDITION_TOKENS.values(), 0)
    for key in union["labelled"]:
        for token in part_map(judgments[key]):
            if token in part_tokens:
                part_tokens[token] += 1
        for token in addition_map(judgments[key]):
            if token in addition_tokens:
                addition_tokens[token] += 1
    n_parts = [
        len(standard_parts(c))
        for c in evidence.get("clauses") or []
        if isinstance(c, dict) and standard_parts(c)
    ]
    checked_total = totals["checked"]
    return {
        "by_role": {r: _row(per_role[r]) for r in ROLES},
        "totals": totals,
        "by_label": by_label,
        "agreement_rate": (round(totals["agreed"] / checked_total, 4) if checked_total else None),
        "label_agreed": label_agreed,
        "parts_agreed": parts_agreed,
        "label_agreement_rate": round(label_agreed / checked_total, 4) if checked_total else None,
        "parts_agreement_rate": round(parts_agreed / checked_total, 4) if checked_total else None,
        "part_tokens": part_tokens,
        "addition_tokens": addition_tokens,
        "standard_parts": {
            "clauses": len(n_parts),
            "parts": sum(n_parts),
            "max": max(n_parts) if n_parts else 0,
            "mean": round(sum(n_parts) / len(n_parts), 2) if n_parts else None,
        },
    }
