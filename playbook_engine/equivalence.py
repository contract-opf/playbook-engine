"""The ``vs_standard`` equivalence label — OPF 0.5 (issue #240).

The question a legal reviewer asks of every precedent is whether the signed
language is functionally equivalent to our standard, or more or less risky for
us. The exact-match ``standard`` fact (issue #220) answers only "did they sign
our words unchanged"; on counterparty paper it is almost always false even
where the concept is the same. So each DISTINCT non-standard text in the
precedent record (a deal's signed text, a non-standard opening, a refused ask)
is judged ONCE against our standard and the answer is carried on the text as
``vs_standard``::

    {"label": "equivalent" | "more_protective" | "less_protective" | "different_concept",
     "reason": "<one sentence, the operative difference>",
     "basis": "judge" | "agent" | "owner",
     "check": {"agreed": bool, "by": "<model id>", "adjudicated": bool} | null}

This narrows owner decision (c) of issue #227 ("no judged verdicts on the
consumer path") for this one label only. The label is an index the consuming
model can check against the cited text, never an instruction; ``null`` means
"not yet judged" and is never a guess.

This module is the pure, store-free half: the cache key, the judge payload,
which texts are eligible, and writing labels onto an ``evidence`` dict. The
store-backed judge (queue on miss, replay on hit) is
:class:`playbook_engine.agent_judge.StoreBackedEquivalenceJudge`; the blind
check and adjudication round is :mod:`playbook_engine.equivalence_check`.

Cache key
---------

``sha256`` over the JSON array ``[agreement_type.id, taxonomy_id,
perspective.party, grouping_key(text), grouping_key(our standard text)]``
where ``grouping_key`` is the OPF-SPEC §3.5.4 grouping key
(:func:`~playbook_engine.precedent.normalize_variant_text`). Never the paper
side, never a document id, never the opening draft, never the role: one
verdict per distinct text, reused by every deal and every role (signed,
opening, refused) that carries it. The judge payload is likewise free of
anything that identifies a deal, a counterparty or a paper side, so the answer
cannot depend on them.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any

__all__ = [
    "BASES",
    "BASIS_OWNER",
    "EQUIVALENCE_KIND",
    "LABELS",
    "ROLES",
    "ROLE_OPENING",
    "ROLE_REFUSED",
    "ROLE_SIGNED",
    "EquivalenceSubject",
    "Slot",
    "collect_subjects",
    "equivalence_key",
    "is_eligible",
    "iter_slots",
    "label_evidence",
    "slot_key",
    "summarize",
    "validate_draft_verdict",
    "vs_standard_of",
]

#: The pending-item / rubric kind of the equivalence judge.
EQUIVALENCE_KIND = "equivalence"

#: ``vs_standard.label`` vocabulary, relative to ``perspective.party``.
LABELS = ("equivalent", "more_protective", "less_protective", "different_concept")

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

#: Verdict keys a producer-supplied draft may carry. ``check`` and the audit
#: trail are written only by ``judge-apply --check``.
_DRAFT_KEYS = frozenset({"label", "reason", "basis"})


# ---------------------------------------------------------------------------
# Key and payload
# ---------------------------------------------------------------------------


def _grouping_key(text: str, party: str | None) -> str:
    from playbook_engine.precedent import normalize_variant_text  # noqa: PLC0415

    return normalize_variant_text(text, party=party)


def equivalence_key(
    agreement_type_id: str,
    taxonomy_id: str,
    party: str | None,
    text: str,
    standard_text: str,
) -> str:
    """The verdict-store key of *text* judged against *standard_text*.

    See the module docstring: five fields, no paper side, no document id, no
    role. ``party`` is ``perspective.party`` (``None`` when the playbook has
    no perspective); two texts with the same §3.5.4 grouping key share a key.
    """
    raw = json.dumps(
        [
            agreement_type_id,
            taxonomy_id,
            party or "",
            _grouping_key(text, party),
            _grouping_key(standard_text, party),
        ],
        ensure_ascii=False,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class EquivalenceSubject:
    """One distinct text awaiting (or carrying) an equivalence verdict.

    ``payload`` is exactly what a drafter or a blind checker sees: the
    taxonomy id and title, the perspective party, our standard, the candidate
    text and the roles it plays. No deal, counterparty, paper side or
    outcome.
    """

    key: str
    payload: dict[str, Any]
    roles: tuple[str, ...]


# ---------------------------------------------------------------------------
# Slots — the places in evidence.precedent a label can sit
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Slot:
    """One text entry of ``evidence.precedent`` that may carry ``vs_standard``."""

    role: str
    clause: dict[str, Any] | None
    record: dict[str, Any]
    #: The ``{text, ref}`` (or refused-ask) dict the label is written onto.
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
    text = slot.entry.get("text")
    if standard is None or not isinstance(text, str) or not text.strip():
        return None
    taxonomy_id = slot.record.get("taxonomy_id")
    if not isinstance(taxonomy_id, str):
        return None
    return equivalence_key(agreement_type_id, taxonomy_id, party, text, standard)


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
    grouping key decides. ``roles`` lists every role the text plays.
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
                    "our_standard": _standard_text(slot.clause),
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
# Verdict shapes
# ---------------------------------------------------------------------------


def _require_label_reason(verdict: Mapping[str, Any]) -> None:
    label = verdict.get("label")
    if label not in LABELS:
        raise ValueError(f"'label' must be one of {list(LABELS)}, got {label!r}")
    reason = verdict.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("'reason' must be a non-empty string (one sentence naming the difference)")


def validate_draft_verdict(verdict: dict[str, Any]) -> None:
    """Validate a producer-supplied equivalence verdict (``judge-apply``).

    ``{"label", "reason", "basis"}`` and nothing else: ``basis`` is required
    (``"agent"`` for a coder acting as the judge, ``"judge"`` for the
    store-backed judge, ``"owner"`` for an after-the-fact correction), and
    ``check`` is written only by ``judge-apply --check``.
    """
    extra = sorted(set(verdict) - _DRAFT_KEYS)
    if extra:
        raise ValueError(
            f"unexpected field(s) {extra} on an equivalence verdict; a verdict carries only "
            "label, reason and basis (checks are recorded with `judge-apply --check`)"
        )
    _require_label_reason(verdict)
    basis = verdict.get("basis")
    if basis not in BASES:
        raise ValueError(f"'basis' must be one of {list(BASES)}, got {basis!r}")


def vs_standard_of(verdict: Mapping[str, Any]) -> dict[str, Any]:
    """The ``vs_standard`` object a stored *verdict* projects to.

    Raises ``ValueError`` on a malformed record (it is then treated as
    unjudged, never replayed).
    """
    _require_label_reason(verdict)
    basis = verdict.get("basis")
    if basis not in BASES:
        raise ValueError(f"'basis' must be one of {list(BASES)}, got {basis!r}")
    check = verdict.get("check")
    if check is not None:
        if (
            not isinstance(check, Mapping)
            or not isinstance(check.get("agreed"), bool)
            or not isinstance(check.get("adjudicated"), bool)
            or not isinstance(check.get("by"), str)
            or not check["by"]
        ):
            raise ValueError("'check' must be null or {agreed: bool, by: str, adjudicated: bool}")
        check = {
            "agreed": check["agreed"],
            "by": check["by"],
            "adjudicated": check["adjudicated"],
        }
    return {
        "label": verdict["label"],
        "reason": verdict["reason"],
        "basis": basis,
        "check": check,
    }


# ---------------------------------------------------------------------------
# Writing labels onto evidence
# ---------------------------------------------------------------------------


def label_evidence(
    evidence: dict[str, Any],
    agreement_type_id: str,
    party: str | None,
    vs_by_key: Mapping[str, dict[str, Any] | None],
) -> None:
    """Set ``vs_standard`` on every text entry of *evidence*, in place.

    An eligible text takes the label *vs_by_key* holds for its key (``None``
    when absent: not yet judged, never guessed); every other text — an exact
    match with our standard, a clause with no ``our_standard`` — is ``None``.
    """
    for slot in iter_slots(evidence):
        vs: dict[str, Any] | None = None
        if is_eligible(slot, party):
            key = slot_key(slot, agreement_type_id, party)
            vs = vs_by_key.get(key) if key is not None else None
        slot.entry["vs_standard"] = dict(vs) if vs is not None else None


# ---------------------------------------------------------------------------
# Counts (scorecard, project report)
# ---------------------------------------------------------------------------


def summarize(
    evidence: Mapping[str, Any], agreement_type_id: str, party: str | None
) -> dict[str, Any]:
    """Counts-only summary of the labels in *evidence*.

    Per role: ``eligible`` and ``labelled`` DISTINCT texts (a text that plays
    two roles counts once in each), ``unjudged``, and the verdict-state counts
    ``drafted`` (every label), ``checked`` (a check is recorded),
    ``agreed``, ``adjudicated``, ``unchecked`` and — for a verdict that
    disagreed with its check and still awaits adjudication — ``disputed``.
    ``totals`` counts distinct texts across roles; ``by_label`` is the label
    histogram of those; ``agreement_rate`` is agreed / checked, ``None`` when
    nothing is checked.
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
    labels: dict[str, str] = {}
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
        labels[key] = str(vs.get("label"))
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
        label: sum(1 for key in union["labelled"] if labels.get(key) == label) for label in LABELS
    }
    checked_total = totals["checked"]
    agreed_total = totals["agreed"]
    return {
        "by_role": {r: _row(per_role[r]) for r in ROLES},
        "totals": totals,
        "by_label": by_label,
        "agreement_rate": (round(agreed_total / checked_total, 4) if checked_total else None),
    }
