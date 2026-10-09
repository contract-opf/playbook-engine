"""The Review tab's list — the SHORTEST useful set of model judgments (issue #241).

Every judgment on the consumer path is a ``vs_standard`` label of OPF-SPEC
§3.5.6, carried once per distinct text. A person MAY confirm or change those
labels, but never has to, so the list is cut to what the consuming model is
actually shown: the texts that appear in the playbook's ``digest`` (after the
digest's per-list cap and its collapse of the ``equivalent`` variants), one row
per distinct text however many deals or roles (signed, opening, refused) carry
it. A judged text the digest does not show changes nothing the model sees, so it
is not a row; ``n_not_shown`` counts them.

Order: ``less_protective``, ``different_concept``, ``more_protective``,
``equivalent``; inside a label, the text with the most deals first. Each row
carries the verdict cache key the page addresses an override by
(:mod:`playbook_engine.overrides`) and a locator the page uses to show the full
text on demand (the row itself holds only the sentence-boundary summary).

Pure and read-only: the rows are a function of the playbook.
"""

from __future__ import annotations

from typing import Any

from playbook_engine.digest import (
    build_digest_v4,
    clause_precedent_groups_with_members,
    playbook_precedent_records,
)
from playbook_engine.equivalence import (
    LABELS,
    ROLE_OPENING,
    ROLE_REFUSED,
    ROLE_SIGNED,
    ROLES,
    equivalence_key,
    is_eligible,
    iter_slots,
    slot_key,
)
from playbook_engine.observation_builder import summarize_clause_text
from playbook_engine.opf_accessors import perspective_party, playbook_clauses

__all__ = ["REVIEW_LABEL_ORDER", "ReviewRows", "build_review_rows"]

#: Display order of the labels in the Review tab (worse or unrankable first).
REVIEW_LABEL_ORDER = ("less_protective", "different_concept", "more_protective", "equivalent")

assert set(REVIEW_LABEL_ORDER) == set(LABELS)

_GROUP_ROLE = {
    "signed_variants": ROLE_SIGNED,
    "refused_asks": ROLE_REFUSED,
    "changed_openings": ROLE_OPENING,
}


class ReviewRows:
    """The rows and the counts around them."""

    def __init__(self, rows: list[dict[str, Any]], n_not_shown: int, n_unjudged: int) -> None:
        self.rows = rows
        #: Judged distinct texts the digest does not show (no row).
        self.n_not_shown = n_not_shown
        #: Digest-visible distinct texts with no label yet (nothing to confirm).
        self.n_unjudged = n_unjudged

    def by_label(self) -> dict[str, int]:
        return {label: sum(1 for r in self.rows if r["label"] == label) for label in LABELS}


def _cite(ref: Any) -> str:
    if not isinstance(ref, dict):
        return ""
    return (
        f"{ref.get('document_id', '')} · v{ref.get('version', '')} · §{ref.get('clause_path', '')}"
    )


def _check_state(vs: dict[str, Any]) -> str:
    if vs.get("basis") == "owner":
        return "owner"
    check = vs.get("check")
    if not isinstance(check, dict):
        return "unchecked"
    if check.get("adjudicated") is True:
        return "adjudicated"
    return "agreed" if check.get("agreed") is True else "disputed"


def _digest_entries(digest: dict[str, Any], tid: Any) -> dict[str, list[dict[str, Any]]]:
    for clause in digest.get("clauses") or []:
        if isinstance(clause, dict) and clause.get("taxonomy_id") == tid:
            return {name: list(clause.get(name) or []) for name in _GROUP_ROLE}
    return {name: [] for name in _GROUP_ROLE}


def build_review_rows(doc: dict[str, Any]) -> ReviewRows:
    """The Review rows of *doc* (``[]`` when it has no evidence or no ``our_standard``)."""
    evidence = doc.get("evidence")
    agreement = doc.get("agreement_type")
    type_id = agreement.get("id") if isinstance(agreement, dict) else None
    if not isinstance(evidence, dict) or not isinstance(type_id, str):
        return ReviewRows([], 0, 0)
    party = perspective_party(doc)
    raw_digest = doc.get("digest")
    digest: dict[str, Any] = raw_digest if isinstance(raw_digest, dict) else build_digest_v4(doc)
    precedent = playbook_precedent_records(doc)

    by_key: dict[str, dict[str, Any]] = {}
    shown: set[str] = set()
    all_judged: set[str] = set()
    unjudged: set[str] = set()

    for clause in playbook_clauses(doc):
        tid = clause.get("taxonomy_id")
        standard = clause.get("our_standard")
        standard_text = standard.get("text") if isinstance(standard, dict) else None
        if not isinstance(standard_text, str) or not standard_text.strip():
            continue
        entries = _digest_entries(digest, tid)
        equivalent_collapsed = any(
            e.get("label") == "equivalent" and "n_texts" in e for e in entries["signed_variants"]
        )
        groups = clause_precedent_groups_with_members(tid, precedent, party=party)
        for name, role in _GROUP_ROLE.items():
            for entry, members in groups[name]:
                if role == ROLE_REFUSED:
                    ask, record = members[0]
                    text = ask.get("text")
                    locator = {
                        "record": record.get("id"),
                        "role": role,
                        "i": next(
                            (i for i, a in enumerate(record.get("refused_asks") or []) if a is ask),
                            0,
                        ),
                    }
                    deals = {r.get("document_id") for _, r in members}
                else:
                    record = members[0]
                    holder = record.get("signed_text" if role == ROLE_SIGNED else "opening_text")
                    text = holder.get("text") if isinstance(holder, dict) else None
                    locator = {"record": record.get("id"), "role": role}
                    deals = {r.get("document_id") for r in members}
                if not isinstance(text, str) or not text.strip():
                    continue
                key = equivalence_key(type_id, str(tid), party, text, standard_text)
                label = entry.get("label")
                if name == "signed_variants" and label == "equivalent":
                    visible = equivalent_collapsed
                else:
                    visible = entry in entries[name]
                if label is None:
                    if visible and key not in all_judged:
                        unjudged.add(key)
                    continue
                unjudged.discard(key)
                all_judged.add(key)
                if visible:
                    shown.add(key)
                vs = _first_vs(members, role)
                row = by_key.get(key)
                if row is None:
                    by_key[key] = {
                        "key": key,
                        "taxonomy_id": tid,
                        "title": clause.get("title") or tid,
                        "label": label,
                        "reason": (vs or {}).get("reason") or "",
                        "basis": (vs or {}).get("basis") or "",
                        "check": _check_state(vs or {}),
                        "roles": [role],
                        "text": summarize_clause_text(text),
                        "standard": summarize_clause_text(standard_text),
                        "cite": _cite(entry.get("ref")),
                        "loc": locator,
                        "_deals": set(deals),
                    }
                else:
                    if role not in row["roles"]:
                        row["roles"].append(role)
                    row["_deals"] |= deals

    rows: list[dict[str, Any]] = []
    for key in shown:
        row = by_key[key]
        row["roles"] = [r for r in ROLES if r in row["roles"]]
        row["n_deals"] = len({d for d in row.pop("_deals") if d is not None})
        rows.append(row)
    order = {label: i for i, label in enumerate(REVIEW_LABEL_ORDER)}
    rows.sort(key=lambda r: (order[r["label"]], -r["n_deals"], str(r["title"]), r["key"]))
    # Every judged distinct text of the playbook, whether or not the digest groups it (the
    # last draft of an unsigned deal is judged but never grouped): the shown ones are rows,
    # the rest are only counted.
    judged = {
        key
        for slot in iter_slots(evidence)
        if isinstance(slot.entry.get("vs_standard"), dict) and is_eligible(slot, party)
        for key in (slot_key(slot, type_id, party),)
        if key is not None
    }
    return ReviewRows(rows, n_not_shown=len(judged - shown), n_unjudged=len(unjudged))


def _first_vs(members: list[Any], role: str) -> dict[str, Any] | None:
    """The ``vs_standard`` object of the group's first member, in its role."""
    first = members[0]
    if role == ROLE_REFUSED:
        holder: Any = first[0]
    else:
        holder = first.get("signed_text" if role == ROLE_SIGNED else "opening_text")
    vs = holder.get("vs_standard") if isinstance(holder, dict) else None
    return vs if isinstance(vs, dict) else None
