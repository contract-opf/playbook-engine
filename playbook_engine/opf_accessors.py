"""OPF accessors — the one place that knows where an OPF document keeps things.

Every consumer that needs a playbook's clauses or precedent records MUST go
through these accessors rather than reaching into ``doc["evidence"]``
directly, so a future OPF version only needs to change one file.

The engine reads and writes exactly one format, OPF 0.4 (issue #238; the
0.1-0.3 shapes and their accessors were retired — git history has them):
``evidence.clauses`` holds one entry per clause type (with counts) and
``evidence.precedent`` one verdict-free record per (deal, clause type)
(issue #223). :func:`playbook_clauses` and :func:`playbook_precedent` /
:func:`clause_precedent` read them.

Issue #224 adds the query surface over the record: :func:`find_precedent`
(one clause's records, or their refused asks, ranked by distinct-deal count
and recency), :func:`precedent_by_id` (one record by its stable
``prec.<sha>`` id), and :func:`precedent_jsonl` — the ``precedent.jsonl``
sidecar ``playbook project`` writes next to ``playbook.opf.json`` (one
record per line, sorted by id), whose sha256 the document carries under
``x_sidecars`` (:func:`precedent_sidecar_manifest`, checked by
:func:`verify_precedent_sidecar`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from playbook_engine.canonicalize import canonicalize, file_sha256, sha256_hex

__all__ = [
    "is_precedent_shape",
    "playbook_precedent",
    "clause_precedent",
    "PRECEDENT_ORDER_KEYS",
    "PRECEDENT_SIDECAR",
    "SIDECARS_KEY",
    "find_precedent",
    "precedent_by_id",
    "precedent_jsonl",
    "precedent_sidecar_manifest",
    "verify_precedent_sidecar",
    "perspective_party",
    "playbook_clauses",
]


def is_precedent_shape(doc: dict[str, Any]) -> bool:
    """True when *doc* carries an ``evidence.precedent`` list (issue #223).

    A structural guard for callers that need the record before reading it
    (``playbook precedent``); a document without one is not a valid OPF 0.4
    document.
    """
    evidence = doc.get("evidence")
    return isinstance(evidence, dict) and isinstance(evidence.get("precedent"), list)


def playbook_precedent(doc: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the ``evidence.precedent`` records (``[]`` when absent).

    One record per (deal, clause type): the deal's signed text, whether it
    is our standard language, whether the clause moved, and the asks refused
    before signing — facts only, never a judged verdict.

    Args:
        doc: A parsed ``playbook.opf.json`` dict.

    Returns:
        The precedent list (non-dict entries dropped), or ``[]``.
    """
    if not is_precedent_shape(doc):
        return []
    return [p for p in doc["evidence"]["precedent"] if isinstance(p, dict)]


def clause_precedent(doc: dict[str, Any], clause: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the precedent records for one clause (matched on taxonomy_id).

    Args:
        doc:    A parsed ``playbook.opf.json`` dict.
        clause: One clause dict from :func:`playbook_clauses`.

    Returns:
        That clause's precedent records in document order, or ``[]``.
    """
    tid = clause.get("taxonomy_id")
    return [p for p in playbook_precedent(doc) if p.get("taxonomy_id") == tid]


# ---------------------------------------------------------------------------
# Query surface (issue #224)
# ---------------------------------------------------------------------------

#: Ranking keys :func:`find_precedent` accepts in ``order``.
PRECEDENT_ORDER_KEYS: frozenset[str] = frozenset({"n_deals", "last_signed"})

#: File name of the precedent sidecar written next to ``playbook.opf.json``.
PRECEDENT_SIDECAR = "precedent.jsonl"

#: Root-level vendor key (OPF-SPEC §10.1) recording the sidecars' content
#: addresses. ``compiler`` is closed to extensions, so the manifest lives at
#: the document root, where it participates in ``identity.content_hash``.
SIDECARS_KEY = "x_sidecars"


def _signed_at_sort(value: Any) -> tuple[bool, tuple[int, ...]]:
    """Sort key putting the latest ``signed_at`` first and unknown last."""
    from playbook_engine.digest import _signed_at_key  # noqa: PLC0415

    key = _signed_at_key(value)
    return (key is None, tuple(-x for x in (key or (0, 0, 0))))


def _latest_signed(values: list[Any]) -> Any:
    from playbook_engine.digest import _latest  # noqa: PLC0415

    return _latest(values)


def _group_key(text: Any, party: str | None) -> str:
    """The OPF-SPEC §3.5.4 grouping key of *text* (``""`` when absent)."""
    if not isinstance(text, str):
        return ""
    from playbook_engine.precedent import normalize_variant_text  # noqa: PLC0415

    return normalize_variant_text(text, party=party)


def _rank(
    items: list[tuple[str, dict[str, Any], Any, bool]],
    order: tuple[str, ...],
    tiebreak: Any,
) -> list[tuple[dict[str, Any], int]]:
    """Rank ``(group_key, item, signed_at, counts)`` tuples by their group's stats.

    A group is every counting item (``counts`` true) sharing a grouping key;
    its ``n_deals`` is the number of distinct deals (``document_id``) in it
    and its ``last_signed`` the latest ``signed_at`` among them. Counting
    items are ordered by *order* (each key descending, unknown dates last),
    then by grouping key, so a group stays contiguous, then by *tiebreak*.
    Non-counting items never enter a group's stats and rank after every
    counting item, by grouping key then *tiebreak*; their reported count
    is ``0``.
    """
    deals: dict[str, set[Any]] = {}
    dates: dict[str, list[Any]] = {}
    for key, item, signed_at, counts in items:
        if not counts:
            continue
        deals.setdefault(key, set()).add(item.get("document_id"))
        dates.setdefault(key, []).append(signed_at)
    n_deals = {k: len(v) for k, v in deals.items()}
    last = {k: _latest_signed(v) for k, v in dates.items()}

    def sort_key(entry: tuple[str, dict[str, Any], Any, bool]) -> tuple[Any, ...]:
        key, item, _, counts = entry
        if not counts:
            return (True, key, tiebreak(item))
        parts: list[Any] = []
        for name in order:
            if name == "n_deals":
                parts.append(-n_deals[key])
            else:
                parts.append(_signed_at_sort(last[key]))
        return (False, *parts, key, tiebreak(item))

    return [
        (item, n_deals[key] if counts else 0)
        for key, item, _, counts in sorted(items, key=sort_key)
    ]


def _clause_taxonomy_id(doc: dict[str, Any], clause: str) -> str | None:
    """Resolve *clause* (a taxonomy id or an ``evidence.clauses[].id``)."""
    for c in playbook_clauses(doc):
        if c.get("taxonomy_id") == clause:
            return clause
    for c in playbook_clauses(doc):
        if c.get("id") == clause and isinstance(c.get("taxonomy_id"), str):
            return str(c["taxonomy_id"])
    return None


def find_precedent(
    doc: dict[str, Any],
    taxonomy_id: str,
    *,
    refused: bool = False,
    limit: int | None = None,
    order: tuple[str, ...] = ("n_deals", "last_signed"),
) -> list[dict[str, Any]]:
    """Return one clause's precedent records — or their refused asks — ranked.

    Records of signed deals (``signed: true``) are grouped by their signed
    text's OPF-SPEC §3.5.4 grouping key (``precedent.normalize_variant_text``
    with the document's ``perspective.party``; a clause struck before
    signing groups under no text), and ranked by *order*: ``"n_deals"`` —
    distinct signed deals in the record's group, most first — and
    ``"last_signed"`` — the group's latest ``signed_at``, newest first,
    unknown last. Ties fall to the grouping key, then the record's own
    ``signed_at`` (newest first), ``document_id`` and ``id``, so the result
    is deterministic. An unsigned deal's ``signed_text`` is its last draft,
    not signed precedent, so its records never count toward a group (as in
    ``digest.clause_precedent_groups``) and come after every signed record,
    ordered by grouping key, then ``document_id`` and ``id``. Nothing is
    judged and paper side never enters the ranking.

    With ``refused=True`` the result is the clause's refused asks instead:
    each ask's own fields (``text``, ``round``, ``ref``) plus the
    ``precedent_id``, ``document_id`` and ``taxonomy_id`` of the record that
    carries it, grouped by the ask text the same way (``last_signed`` is the
    latest ``signed_at`` of the deals that refused it); ties fall to the
    earliest round, then ``document_id``.

    Precedent records are returned as the document's own dicts — the same
    objects :func:`playbook_precedent` yields — so a caller serializing one
    gets exactly what the playbook carries.

    Args:
        doc:         A parsed ``playbook.opf.json`` dict.
        taxonomy_id: The clause's ``taxonomy_id`` (an ``evidence.clauses[].id``
                     such as ``clause.governing_law`` is accepted too).
        refused:     Return the refused asks instead of the records.
        limit:       Keep at most this many results (``None`` = all).
        order:       Ranking keys, applied in sequence, from
                     :data:`PRECEDENT_ORDER_KEYS`.

    Returns:
        The ranked records or asks; ``[]`` for an unknown clause.

    Raises:
        ValueError: an unknown *order* key, or a negative *limit*.
    """
    unknown = [k for k in order if k not in PRECEDENT_ORDER_KEYS]
    if unknown:
        raise ValueError(
            f"unknown order key(s) {unknown}; expected any of {sorted(PRECEDENT_ORDER_KEYS)}"
        )
    if limit is not None and limit < 0:
        raise ValueError(f"limit must be >= 0, got {limit}")
    tid = _clause_taxonomy_id(doc, taxonomy_id)
    if tid is None:
        return []
    party = perspective_party(doc)
    records = [p for p in playbook_precedent(doc) if p.get("taxonomy_id") == tid]

    if not refused:
        items = []
        for record in records:
            signed_text = record.get("signed_text")
            text = signed_text.get("text") if isinstance(signed_text, dict) else None
            # Only a signed deal is precedent for its text: an unsigned
            # record's signed_text is the last draft (OPF-SPEC §3.5.4), so it
            # never counts toward a group and ranks after every signed one.
            items.append(
                (
                    _group_key(text, party),
                    record,
                    record.get("signed_at"),
                    record.get("signed") is True,
                )
            )
        ranked = _rank(
            items,
            order,
            lambda r: (
                _signed_at_sort(r.get("signed_at")),
                str(r.get("document_id")),
                str(r.get("id")),
            ),
        )
        out = [record for record, _ in ranked]
    else:
        asks = []
        for record in records:
            for ask in record.get("refused_asks") or []:
                if not isinstance(ask, dict):
                    continue
                entry = {
                    "precedent_id": record.get("id"),
                    "document_id": record.get("document_id"),
                    "taxonomy_id": record.get("taxonomy_id"),
                    **ask,
                }
                asks.append(
                    (_group_key(ask.get("text"), party), entry, record.get("signed_at"), True)
                )
        ranked = _rank(
            asks,
            order,
            lambda a: (
                a.get("round") if isinstance(a.get("round"), int) else 0,
                str(a.get("document_id")),
                str(a.get("precedent_id")),
            ),
        )
        out = [ask for ask, _ in ranked]
    return out if limit is None else out[:limit]


def precedent_by_id(doc: dict[str, Any], precedent_id: str) -> dict[str, Any] | None:
    """Return the ``evidence.precedent`` record whose ``id`` is *precedent_id*.

    Ids are stable (``"prec."`` + a content hash of agreement type, deal,
    clause type and signed text — OPF-SPEC §3.5.4), so an id read from the
    digest's ``precedent_ids`` or from ``precedent.jsonl`` addresses the
    same record in the playbook.

    Returns:
        The record (the document's own dict), or ``None`` when absent.
    """
    return next((p for p in playbook_precedent(doc) if p.get("id") == precedent_id), None)


def precedent_jsonl(doc: dict[str, Any]) -> str:
    """Return the ``precedent.jsonl`` sidecar text for *doc*.

    One ``evidence.precedent`` record per line, each the canonical JSON
    serialization (sorted keys, no insignificant whitespace — the same form
    ``identity.content_hash`` hashes) of the record exactly as the playbook
    carries it, lines sorted by ``id`` and each ending in a newline. A pure
    function of ``evidence.precedent``: ``""`` when there are no records.
    """
    lines = sorted((str(p.get("id")), canonicalize(p)) for p in playbook_precedent(doc))
    return "".join(line + "\n" for _, line in lines)


def precedent_sidecar_manifest(doc: dict[str, Any]) -> dict[str, Any]:
    """Return the ``x_sidecars`` entry describing *doc*'s ``precedent.jsonl``.

    ``{"precedent.jsonl": {"sha256": "sha256:<hex>", "records": N}}`` —
    the sha256 of the sidecar's UTF-8 bytes (:func:`precedent_jsonl`) and
    its line count, so a consumer can check a ``precedent.jsonl`` belongs
    to this playbook (:func:`verify_precedent_sidecar`).
    """
    return {
        PRECEDENT_SIDECAR: {
            "sha256": sha256_hex(precedent_jsonl(doc)),
            "records": len(playbook_precedent(doc)),
        }
    }


def verify_precedent_sidecar(doc: dict[str, Any], path: Path) -> bool:
    """True when the file at *path* is the ``precedent.jsonl`` *doc* records.

    Compares the file's sha256 with ``x_sidecars["precedent.jsonl"].sha256``;
    ``False`` when the document records no such sidecar or the file is
    missing.
    """
    sidecars = doc.get(SIDECARS_KEY)
    entry = sidecars.get(PRECEDENT_SIDECAR) if isinstance(sidecars, dict) else None
    expected = entry.get("sha256") if isinstance(entry, dict) else None
    if not isinstance(expected, str) or not Path(path).is_file():
        return False
    return file_sha256(path) == expected


def perspective_party(doc: dict[str, Any]) -> str | None:
    """Return ``perspective.party`` — the side the playbook reviews for — or ``None``.

    The one party name an OPF document itself carries. The grouping key
    (``precedent.normalize_variant_text``) neutralizes it, so every reader
    of a document — counts, digest, renderers, validator — must take it from
    here to group texts the same way.

    Args:
        doc: A parsed ``playbook.opf.json`` dict.

    Returns:
        The party name, or ``None`` when there is no perspective, no
        ``party``, or the party is not a non-blank string.
    """
    perspective = doc.get("perspective")
    party = perspective.get("party") if isinstance(perspective, dict) else None
    return party if isinstance(party, str) and party.strip() else None


def playbook_clauses(doc: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the playbook's ``evidence.clauses`` list (``[]`` when absent).

    Args:
        doc: A parsed ``playbook.opf.json`` dict.

    Returns:
        The clause list, or ``[]`` if the document carries none.
    """
    evidence = doc.get("evidence")
    clauses = evidence.get("clauses") if isinstance(evidence, dict) else None
    return clauses if isinstance(clauses, list) else []
