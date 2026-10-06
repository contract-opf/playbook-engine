"""Shape-agnostic OPF accessors — issue #154.

Issue #140 migrated ``playbook_assembler`` to emit OPF **v0.2** documents:
clauses moved from a top-level ``clauses`` array to ``evidence.clauses``, and
each clause's prescriptive ``rollup.position`` was replaced by a descriptive
``summary.historical_stance`` (``rollup.confidence`` -> ``summary.confidence``
likewise). Consumers that read the v0.1 shape directly (``aar.py``,
``viewer.py``) silently degraded to empty output against a real compiled
v0.2 playbook — no exception, just zero clauses.

This module is the single place that knows both shapes. Every consumer that
needs a playbook's clauses or a clause's historical stance/confidence MUST
go through these accessors rather than reaching into ``doc["clauses"]`` or
``clause["rollup"]`` directly, so a future OPF version only needs to change
one file.

v0.1 fixtures (hand-authored in existing test suites) continue to work
unchanged — every accessor here falls back to the v0.1 shape when the v0.2
key is absent.

OPF 0.4 (issue #223) replaces the 0.2/0.3 evidence shape with the
verdict-free per-deal precedent record: ``evidence.clauses`` keeps one
entry per clause type (with counts, but no ``summary`` or
``observed_positions``) and ``evidence.precedent`` holds one record per
(deal, clause type). :func:`playbook_clauses` reads both shapes;
:func:`playbook_precedent` / :func:`clause_precedent` read the 0.4 records
and return ``[]`` for any older document. On a 0.4 clause the 0.2/0.3
accessors (:func:`clause_stance`, :func:`clause_confidence`,
:func:`clause_trail`) degrade to their documented "absent" values —
``"unknown"``, ``{}``, ``[]`` — rather than inventing a stance.

Issue #224 adds the query surface over the 0.4 record:
:func:`find_precedent` (one clause's records, or their refused asks, ranked
by distinct-deal count and recency), :func:`precedent_by_id` (one record by
its stable ``prec.<sha>`` id), and :func:`precedent_jsonl` — the
``precedent.jsonl`` sidecar ``playbook project`` writes next to
``playbook.opf.json`` (one record per line, sorted by id), whose sha256 the
document carries under ``x_sidecars`` (:func:`precedent_sidecar_manifest`,
checked by :func:`verify_precedent_sidecar`).
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
    "playbook_clause_library",
    "clause_stance",
    "clause_confidence",
    "clause_is_thin",
    "observation_dynamics",
    "clause_trail",
]


def is_precedent_shape(doc: dict[str, Any]) -> bool:
    """True when *doc* carries the OPF 0.4 precedent record (issue #223).

    Keyed on ``evidence.precedent`` being present as a list, not on the
    ``opf_version`` string alone, so a renderer handed a hand-built 0.4-shaped
    fixture reads it the same way as a compiled one.
    """
    evidence = doc.get("evidence")
    return isinstance(evidence, dict) and isinstance(evidence.get("precedent"), list)


def playbook_precedent(doc: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the OPF 0.4 ``evidence.precedent`` records (``[]`` otherwise).

    One record per (deal, clause type): the deal's signed text, whether it
    is our standard language, whether the clause moved, and the asks refused
    before signing — facts only, never a judged verdict. Pre-0.4 documents
    have no precedent records and yield ``[]``.

    Args:
        doc: A parsed ``playbook.opf.json`` dict (any OPF version).

    Returns:
        The precedent list (non-dict entries dropped), or ``[]``.
    """
    if not is_precedent_shape(doc):
        return []
    return [p for p in doc["evidence"]["precedent"] if isinstance(p, dict)]


def clause_precedent(doc: dict[str, Any], clause: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the precedent records for one clause (matched on taxonomy_id).

    Args:
        doc:    A parsed ``playbook.opf.json`` dict (any OPF version).
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
        doc:         A parsed ``playbook.opf.json`` dict (any OPF version;
                     pre-0.4 documents yield ``[]``).
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
    function of ``evidence.precedent``: ``""`` for a pre-0.4 document.
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

    The one party name an OPF document itself carries. OPF 0.4's grouping
    key (``precedent.normalize_variant_text``) neutralizes it, so every
    reader of a 0.4 document — counts, digest, renderers, validator — must
    take it from here to group texts the same way.

    Args:
        doc: A parsed ``playbook.opf.json`` dict (any OPF version).

    Returns:
        The party name, or ``None`` when there is no perspective, no
        ``party``, or the party is not a non-blank string.
    """
    perspective = doc.get("perspective")
    party = perspective.get("party") if isinstance(perspective, dict) else None
    return party if isinstance(party, str) and party.strip() else None


def playbook_clauses(doc: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the playbook's clause list, regardless of OPF version.

    OPF v0.2 nests clauses under ``evidence.clauses``; OPF v0.1 (and hand
    -authored test fixtures) keep them at the top-level ``clauses`` key.
    v0.2 takes precedence when ``evidence`` is present as a dict — a v0.2
    document never carries a top-level ``clauses`` key (see
    ``playbook_assembler.assemble_playbook``), so there is no ambiguity in
    practice.

    Args:
        doc: A parsed ``playbook.opf.json`` dict (either OPF version).

    Returns:
        The clause list, or ``[]`` if neither shape is present.
    """
    evidence = doc.get("evidence")
    if isinstance(evidence, dict) and "clauses" in evidence:
        clauses = evidence.get("clauses")
        return clauses if isinstance(clauses, list) else []

    clauses = doc.get("clauses")
    return clauses if isinstance(clauses, list) else []


def playbook_clause_library(doc: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the playbook's clause-concept library, regardless of OPF version.

    Mirrors :func:`playbook_clauses`: OPF v0.2 nests the library under
    ``evidence.clause_library``; OPF v0.1 (and hand-authored test fixtures)
    keep it at the top-level ``clause_library`` key. v0.2 takes precedence
    when ``evidence`` is present as a dict (issue #188 — the publish
    transform and export-profile residue sampling both need version-agnostic
    access to ``ClauseConcept.description``/``notes``).

    Args:
        doc: A parsed ``playbook.opf.json`` dict (either OPF version).

    Returns:
        The clause-concept list, or ``[]`` if neither shape is present.
    """
    evidence = doc.get("evidence")
    if isinstance(evidence, dict) and "clause_library" in evidence:
        library = evidence.get("clause_library")
        return library if isinstance(library, list) else []

    library = doc.get("clause_library")
    return library if isinstance(library, list) else []


def clause_stance(clause: dict[str, Any]) -> str:
    """Return one clause's historical stance / rollup position, version-agnostic.

    OPF v0.2 carries this as ``summary.historical_stance`` (descriptive: "what
    has the corpus shown"); OPF v0.1 carried it as ``rollup.position``
    (prescriptive). v0.2 takes precedence when ``summary`` is present as a
    dict.

    Args:
        clause: One clause dict from ``playbook_clauses()``.

    Returns:
        The stance/position string, or ``"unknown"`` if neither shape is
        present.
    """
    summary = clause.get("summary")
    if isinstance(summary, dict) and "historical_stance" in summary:
        return str(summary.get("historical_stance") or "unknown")

    rollup = clause.get("rollup")
    if isinstance(rollup, dict):
        return str(rollup.get("position") or "unknown")

    return "unknown"


def clause_confidence(clause: dict[str, Any]) -> dict[str, Any]:
    """Return one clause's confidence block, version-agnostic.

    OPF v0.2 carries this as ``summary.confidence``; OPF v0.1 carried it as
    ``rollup.confidence``. Both shapes carry the same inner keys (``score``,
    ``n_our_paper``, ``n_counterparty_paper``, ``evidence_sufficient``,
    ...) — only the wrapper key changed.

    Args:
        clause: One clause dict from ``playbook_clauses()``.

    Returns:
        The confidence dict, or ``{}`` if neither shape is present.
    """
    summary = clause.get("summary")
    if isinstance(summary, dict) and "confidence" in summary:
        confidence = summary.get("confidence")
        return confidence if isinstance(confidence, dict) else {}

    rollup = clause.get("rollup")
    if isinstance(rollup, dict):
        confidence = rollup.get("confidence")
        return confidence if isinstance(confidence, dict) else {}

    return {}


def clause_is_thin(clause: dict[str, Any]) -> bool:
    """Whether one clause's evidence is "thin" — the shared trigger issue
    #91 (review-HTML attention sort) and issue #92 (prompt-renderer heading
    marker) both key off.

    ``True`` when ``confidence.evidence_sufficient`` is explicitly
    ``False``, OR every ``observed_positions`` entry on record has
    ``precedent_count == 1`` (nothing behind this clause has ever recurred
    in the corpus). A clause with NO observed positions at all is not, by
    the second branch alone, "thin" — that case is caught by the first
    branch in any playbook the compiler itself produced, since the
    compiler sets ``evidence_sufficient`` False whenever ``n_our_paper``
    falls short of its configured minimum (``clause_position_compiler.py``).

    Mirrors ``prompt_renderer._thin_marker``'s trigger condition exactly;
    kept here as the version-agnostic, publicly reusable primitive so a
    second consumer (issue #91) never has to re-derive or drift from the
    same definition.

    Args:
        clause: One clause dict from ``playbook_clauses()``.

    Returns:
        ``True`` if the clause's evidence is thin by either trigger.
    """
    confidence = clause_confidence(clause)
    positions = [p for p in (clause.get("observed_positions") or []) if isinstance(p, dict)]
    evidence_insufficient = confidence.get("evidence_sufficient") is False
    single_precedent_only = bool(positions) and all(
        p.get("precedent_count") == 1 for p in positions
    )
    return evidence_insufficient or single_precedent_only


def observation_dynamics(obs: dict[str, Any]) -> dict[str, Any]:
    """Return one observation's negotiation-dynamics fields (issue #177).

    OPF v0.2 §3.5.3 fields are optional-when-underivable, so a v0.2 document
    without dynamics (or any v0.1 observation) simply yields ``{}`` — a key
    appears in the result only when the observation actually carries it.

    Args:
        obs: One entry of a clause's ``observed_positions`` (either OPF
             version).

    Returns:
        Dict with any of ``proposed_by`` / ``observed_at`` /
        ``counterparty_ref`` that are present; ``{}`` otherwise.
    """
    dynamics: dict[str, Any] = {}
    for key in ("proposed_by", "observed_at", "counterparty_ref"):
        value = obs.get(key)
        if value is not None:
            dynamics[key] = value
    return dynamics


def clause_trail(clause: dict[str, Any]) -> list[dict[str, Any]]:
    """Return one clause's ``negotiation_trail`` (issue #177), or ``[]``.

    v0.2 documents compiled before §3.5.3 (and every v0.1 document) carry no
    trail; they read cleanly as an empty list.

    Args:
        clause: One clause dict (either OPF version).

    Returns:
        The trail entry list, or ``[]`` when absent/malformed.
    """
    trail = clause.get("negotiation_trail")
    return trail if isinstance(trail, list) else []
