"""OPF 0.5 evidence — the verdict-free per-deal precedent record (issues #223, #233).

The evidence shapes of OPF 0.1-0.3 described categories the engine could no
longer honestly fill once judged deviation verdicts left the consumer path
(issue #220, and the judge itself was retired in issue #239) and the deal
became the unit of precedent (issue #216); they were retired (issue #238 —
git history has them). OPF 0.5's evidence is two
lists:

``evidence.clauses[]``
    One entry per clause type: ``{id, taxonomy_id, title, our_standard,
    n_deals, n_signed_standard, n_variants, n_refused}``. Every count is
    derived from ``evidence.precedent`` (the validator recomputes them).

``evidence.precedent[]``
    One record per (deal, clause type): what that deal signed for the clause
    (``signed_text``), whether that text is OUR standard language
    (``standard`` — the deterministic exact check of issue #220, never a
    judged verdict), what the clause opened with (``opened_with`` and, for
    any distinct opening, ``opening_text`` — issue #233), whether the clause
    moved during the negotiation, and the counterparty asks refused before
    signing (``refused_asks``). Paper
    side is carried as honest metadata (``paper``) and never partitions,
    gates or weights anything (owner decision 2026-09-13 (b)).

Nothing here is judged.

The builders are pure functions of the L4 store (observations, round moves,
the corpus manifest) plus the compiled clause positions (which carry each
clause's id, title and ``our_standard``) — two compiles of the same store
produce byte-identical evidence.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from typing import Any

from playbook_engine.canonicalize import canonicalize
from playbook_engine.clause_position_compiler import (
    ClausePosition,
    OPFCitation,
    _is_degenerate_observation_text,
)
from playbook_engine.observation_builder import (
    OPENED_NON_STANDARD,
    OPENED_STANDARD,
    OUTCOME_OPENING,
    Observation,
    RoundMove,
)

__all__ = [
    "PAPER_OURS",
    "PAPER_THEIRS",
    "PAPER_UNKNOWN",
    "PRECEDENT_ID_PREFIX",
    "build_precedent_evidence",
    "clause_counts",
    "normalize_variant_text",
    "paper_of_corpus_document",
    "precedent_id",
    "restamp_evidence",
]

#: Every precedent id starts with this prefix (OPF-SPEC §3.5, OPF 0.5).
PRECEDENT_ID_PREFIX = "prec."

#: Hex characters of the sha256 kept in a precedent id. 64 bits is far beyond
#: any corpus's (deal, clause) count, and every id is unique by construction
#: anyway: (document_id, taxonomy_id) is unique per record and is part of the
#: hashed payload. The validator rejects a duplicate id regardless.
_PRECEDENT_ID_HEX = 16

PAPER_OURS = "ours"
PAPER_THEIRS = "theirs"
PAPER_UNKNOWN = "unknown"

#: ``paper_basis`` values. Paper side is metadata only — never a gate.
_PAPER_BASIS_DETECTED = "provenance_detection"
_PAPER_BASIS_AMBIGUOUS = "ambiguous_detection"
_PAPER_BASIS_NOT_RECORDED = "not_recorded"

_TERMINAL_OUTCOMES = frozenset({"signed", "unsigned"})
_REFUSED_OUTCOME = "proposed_then_reversed"


def precedent_id(
    agreement_type_id: str, document_id: str, taxonomy_id: str, signed_text: str | None
) -> str:
    """Return the stable id of one precedent record.

    ``"prec." + sha256(canonical([agreement_type_id, document_id,
    taxonomy_id, signed_text or ""]))[:16]`` — the canonical JSON array
    (``canonicalize``) is the hashed payload, so the field boundaries are
    unambiguous. A precedent with no signed text (the clause was struck
    before signing) hashes the empty string in that slot. Stable across
    recompiles of the same store; changes when the deal's signed text for
    the clause changes.
    """
    payload = canonicalize([agreement_type_id, document_id, taxonomy_id, signed_text or ""])
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"{PRECEDENT_ID_PREFIX}{digest[:_PRECEDENT_ID_HEX]}"


#: An entity-registry alias (``entity_registry.EntityRegistry.alias_for``):
#: ``Counterparty-<n>``, matched case-insensitively on word boundaries. The
#: born-safe pseudonymization pass writes one into clause text in place of
#: every known counterparty name, a different ``<n>`` per counterparty.
_COUNTERPARTY_ALIAS_RE = re.compile(r"(?<!\w)counterparty-[0-9]+(?!\w)", re.IGNORECASE)

#: The neutral token every counterparty alias is rewritten to before
#: grouping. Distinct from ``normalize_for_standard``'s ``"party"`` token
#: (which ``perspective.party`` becomes), so a clause whose obligation runs
#: the other way between the parties never groups with the original.
_COUNTERPARTY_TOKEN = "counterparty"


def normalize_variant_text(text: str, *, party: str | None) -> str:
    """Grouping key for signed variants, refused asks and openings (OPF 0.5 / digest 3).

    Normative as OPF-SPEC §3.5.4 "Grouping key". The standard check's
    exact-after-normalization rule
    (``deviation_classifier.normalize_for_standard``, issue #220), with the
    party names taken from the document itself, because the names the
    standard check neutralizes (the producer's configured
    ``provenance.our_party_aliases`` and ``known_entities``) are never in
    the document, so a validator or a port could not recompute them:

    1. every entity-registry alias (``Counterparty-<n>``, case-insensitive,
       on word boundaries) becomes the token ``counterparty`` — each deal's
       counterparty carries its own alias, so without this two deals that
       signed the same words split into per-deal singletons;
    2. ``normalize_for_standard(text, [party])``: whitespace collapsed,
       *party* (``perspective.party``, see
       :func:`~playbook_engine.opf_accessors.perspective_party`; nothing when
       ``None``) rewritten to the token ``party`` case-insensitively on
       word boundaries, then case and punctuation dropped.

    Word order and every content token — negators, modals, numerals — survive,
    and our party and the counterparty stay distinct tokens, so two texts
    group together only when they are the same words in the same order with
    the same party in each place. There is deliberately no similarity
    tolerance (it absorbed negation flips and changed amounts).
    """
    # Deferred: deviation_classifier's own import graph reaches this
    # package's compiler modules (see normalize_for_standard's note).
    from playbook_engine.deviation_classifier import normalize_for_standard  # noqa: PLC0415

    neutral = _COUNTERPARTY_ALIAS_RE.sub(_COUNTERPARTY_TOKEN, text)
    return normalize_for_standard(neutral, (party,) if party else ())


def paper_of_corpus_document(corpus_doc: dict[str, Any] | None) -> str:
    """The three-valued paper side a corpus document's provenance records.

    ``corpus.documents[].provenance`` is a frozen two-valued field, so an
    ambiguous detection (``provenance_is_ambiguous: true``) or a missing
    side is ``"unknown"`` — never the side the two-valued field had to carry
    (issue #225). Shared with the validator's paper cross-check.
    """
    if corpus_doc is None or corpus_doc.get("provenance_is_ambiguous") is True:
        return PAPER_UNKNOWN
    provenance = corpus_doc.get("provenance")
    if provenance == "our_paper":
        return PAPER_OURS
    if provenance == "counterparty_paper":
        return PAPER_THEIRS
    return PAPER_UNKNOWN


def _paper(
    corpus_doc: dict[str, Any] | None, rows: Iterable[Observation] = ()
) -> tuple[str, str, float | None]:
    """Three-valued paper side for one deal: ``(paper, paper_basis, confidence)``.

    An ambiguous provenance detection is ``"unknown"`` — never coerced to a
    side. Metadata only: nothing in OPF 0.5 partitions, gates or weights by
    paper side.

    ``paper_basis`` is the detection signal the deal's observations carry
    (``Observation.paper_basis`` — e.g. ``"template_similarity"``,
    ``"needs_review"``; issue #225); a store mined before that was recorded
    falls back to ``"provenance_detection"`` / ``"ambiguous_detection"`` /
    ``"not_recorded"``.
    """
    paper = paper_of_corpus_document(corpus_doc)
    detected_basis = next((o.paper_basis for o in rows if o.paper_basis), None)
    if corpus_doc is None:
        return paper, detected_basis or _PAPER_BASIS_NOT_RECORDED, None
    confidence = corpus_doc.get("provenance_confidence")
    conf = float(confidence) if isinstance(confidence, (int, float)) else None
    if detected_basis is not None:
        return paper, detected_basis, conf
    if corpus_doc.get("provenance_is_ambiguous") is True:
        return paper, _PAPER_BASIS_AMBIGUOUS, conf
    if paper == PAPER_UNKNOWN:
        return paper, _PAPER_BASIS_NOT_RECORDED, conf
    return paper, _PAPER_BASIS_DETECTED, conf


def _ref(obs: Observation) -> dict[str, Any]:
    """An OPF citation for *obs* (``version_id`` is engine-internal)."""
    return OPFCitation(
        document_id=obs.citation.document_id,
        version=obs.citation.version,
        clause_path=obs.citation.clause_path,
        char_span=obs.citation.char_span,
    ).to_dict()


def _ask_round(ref: dict[str, Any]) -> int:
    """The negotiation round an ask was made in: its cited draft's ordinal - 1.

    Round ``r`` is the version transition ``r -> r+1`` (the
    ``negotiation_trail`` numbering, issue #177), so text first read in
    version ``k`` arrived in round ``k - 1``; text already in the opening
    draft (version 1) is round 0. A citation with no integer ordinal is
    round 0 (the opening draft is the only draft every deal has).
    """
    version = ref.get("version")
    if isinstance(version, int) and version >= 1:
        return version - 1
    return 0


def _text_entry(obs: Observation) -> dict[str, Any]:
    return {"text": obs.full_text, "ref": _ref(obs)}


def _is_signed_deal(
    corpus_doc: dict[str, Any] | None, terminal: Observation | None, others: Iterable[Observation]
) -> bool:
    """Whether the deal has a detected executed copy.

    ``corpus.documents[].signed_version`` is authoritative when recorded;
    otherwise the store's own outcomes decide (a ``"signed"`` terminal row or
    an ``"opening"`` row both exist only in a deal with a detected executed
    copy).
    """
    if corpus_doc is not None and "signed_version" in corpus_doc:
        return corpus_doc.get("signed_version") is not None
    if terminal is not None:
        return terminal.outcome == "signed"
    return any(o.outcome == OUTCOME_OPENING for o in others)


def build_precedent_evidence(
    *,
    agreement_type_id: str,
    clause_positions: list[ClausePosition],
    observations: list[Observation],
    corpus_documents: list[dict[str, Any]],
    round_moves: list[RoundMove] | None = None,
    party: str | None,
) -> dict[str, Any]:
    """Build the OPF 0.5 ``evidence`` section: ``{clauses, precedent}``.

    One precedent per (document_id, taxonomy_id) carrying any classified
    observation of a compiled clause type:

    - ``signed_text``: the deal's terminal text for the clause (the executed
      copy when ``signed`` is true, the last draft when it is false), or
      ``null`` when the clause was struck before signing.
    - ``opened_with`` (issue #233): what the clause opened with —
      ``"standard"``, ``"non_standard"``, ``"absent"`` (added during the
      negotiation) or ``null`` (not determined: no detected executed copy, or
      a store that predates the field — never ``"absent"``). Read off the
      terminal row, else the opening row, else a refused-ask row, else
      ``null``.
    - ``opening_text``: the first-draft text of the clause (every first-draft
      node bound into the clause type, joined in first-draft order and citing
      the first), recorded whatever its origin when ``opened_with`` is
      ``"standard"`` or ``"non_standard"`` AND either ``signed_text`` is
      ``null`` or the §3.5.4 grouping key of the opening differs from that of
      ``signed_text`` (so a whitespace/case/punctuation-only edit is not an
      opening); ``null`` otherwise. ``null`` means "not recorded / not
      distinct", never "opened with the signed text" when ``opened_with`` is
      ``null``.
    - ``standard``: the terminal row's deterministic standard fact
      (exact match after ``normalize_for_standard``, issue #220); false when
      there is no terminal text.
    - ``rounds``: the number of distinct negotiation rounds in which the
      clause changed (``round_moves``); ``moved``: it changed, was struck,
      or carried a refused ask — always true when ``opening_text`` is
      recorded.
    - ``refused_asks``: every ``proposed_then_reversed`` row — text proposed
      and then struck before signing — with its round and citation; always
      empty when ``signed`` is false (issue #221).

    ``signed_at`` is never set: the reference compiler extracts no signing
    date and never fabricates one (OPF-SPEC §3.5.4), so the digest's
    ``first_signed`` / ``last_signed`` and every
    ``signed_variants[].last_signed`` are always ``null`` in reference
    output. The optional field exists for a third-party producer that
    records signing dates; the 0.5 conformance vectors that set it model one.

    Sub-sentence fragments are excluded exactly as the clause-type compiler
    excludes them (``MIN_OBSERVATION_TEXT_LEN``); unclassified observations
    and clause types with no compiled position never reach precedent.

    *party* is the document's ``perspective.party`` (``None`` when it has
    none) — the clause counts group texts by :func:`normalize_variant_text`,
    which neutralizes it.
    """
    docs_by_id = {
        str(d.get("document_id")): d for d in corpus_documents if d.get("document_id") is not None
    }
    tids = {cp.taxonomy_id for cp in clause_positions}

    groups: dict[tuple[str, str], list[Observation]] = {}
    for obs in observations:
        tid = obs.taxonomy_id
        if tid is None or tid not in tids:
            continue
        if obs.citation.document_id == "template":
            continue
        if _is_degenerate_observation_text(obs.full_text):
            continue
        groups.setdefault((obs.citation.document_id, tid), []).append(obs)

    rounds_by_key: dict[tuple[str, str], set[int]] = {}
    for move in round_moves or []:
        if move.taxonomy_id is None:
            continue
        rounds_by_key.setdefault((move.document_id, move.taxonomy_id), set()).add(move.round)

    precedent: list[dict[str, Any]] = []
    for document_id, tid in sorted(groups):
        rows = groups[(document_id, tid)]
        terminal = next((o for o in rows if o.outcome in _TERMINAL_OUTCOMES), None)
        # At most one opening row per (deal, clause): the first-draft text of
        # the whole clause type, already joined by observation_builder.
        opening = next((o for o in rows if o.outcome == OUTCOME_OPENING), None)
        corpus_doc = docs_by_id.get(document_id)
        signed = _is_signed_deal(corpus_doc, terminal, rows)
        # Issue #221: a deal with no detected executed copy records no
        # refused asks — with no signed terminal, "struck before signing" is
        # not established. observation_builder no longer writes such rows;
        # this also keeps an observations.jsonl mined before the fix from
        # reintroducing them at project time.
        refused = [o for o in rows if o.outcome == _REFUSED_OUTCOME] if signed else []
        if terminal is None and opening is None and not refused:
            continue
        paper, paper_basis, paper_confidence = _paper(corpus_doc, rows)
        signed_text = _text_entry(terminal) if terminal is not None else None
        # Issue #233: only a deal with a detected executed copy has an
        # anchored opening; a store that predates the field (no row carries
        # ``opened_with``) leaves it null, never "absent".
        opened_with = (
            next(
                (
                    o.opened_with
                    for o in (terminal, opening, *refused)
                    if o is not None and o.opened_with is not None
                ),
                None,
            )
            if signed
            else None
        )
        opening_text = (
            _text_entry(opening)
            if (
                opening is not None
                and opened_with in (OPENED_STANDARD, OPENED_NON_STANDARD)
                and (
                    signed_text is None
                    or normalize_variant_text(opening.full_text, party=party)
                    != normalize_variant_text(signed_text["text"], party=party)
                )
            )
            else None
        )
        refused_asks = sorted(
            (
                {"text": o.full_text, "round": _ask_round(ref), "ref": ref}
                for o in refused
                for ref in (_ref(o),)
            ),
            key=lambda a: (a["round"], a["text"], canonicalize(a["ref"])),
        )
        n_rounds = len(rounds_by_key.get((document_id, tid), set()))
        counterparty_ref = next(
            (o.counterparty_ref for o in rows if o.counterparty_ref is not None), None
        )

        record: dict[str, Any] = {
            "id": "",  # stamped by restamp_evidence below
            "taxonomy_id": tid,
            "document_id": document_id,
        }
        if counterparty_ref is not None:
            record["counterparty_ref"] = dict(counterparty_ref)
        record.update(
            {
                "paper": paper,
                "paper_basis": paper_basis,
                "paper_confidence": paper_confidence,
                "signed": signed,
                "rounds": n_rounds,
                "signed_text": signed_text,
                "opened_with": opened_with,
                "opening_text": opening_text,
                "standard": bool(terminal is not None and terminal.standard),
                "moved": bool(n_rounds or opening_text is not None or refused),
                "refused_asks": refused_asks,
            }
        )
        precedent.append(record)

    clauses: list[dict[str, Any]] = [
        {
            "id": cp.id,
            "taxonomy_id": cp.taxonomy_id,
            "title": cp.title,
            "our_standard": cp.our_standard.to_dict() if cp.our_standard else None,
        }
        for cp in clause_positions
    ]
    evidence = {"clauses": clauses, "precedent": precedent}
    restamp_evidence(evidence, agreement_type_id, party=party)
    return evidence


def restamp_evidence(
    evidence: dict[str, Any], agreement_type_id: str, *, party: str | None
) -> None:
    """(Re)compute every derived value of an OPF 0.5 ``evidence`` in place.

    Each precedent's ``id`` (:func:`precedent_id`) and each clause's
    ``n_*`` counts (:func:`clause_counts`, grouping by
    :func:`normalize_variant_text` with *party* — the document's
    ``perspective.party``) are functions of the texts they describe. The
    assembler strips zero-width/bidi-control characters from the whole
    document after building it, so it calls this again afterwards — the ids
    and counts a validator recomputes over the shipped text must be the ones
    the document carries.
    """
    precedent = evidence["precedent"]
    for record in precedent:
        signed_text = record.get("signed_text")
        new_id = precedent_id(
            agreement_type_id,
            record["document_id"],
            record["taxonomy_id"],
            signed_text["text"] if isinstance(signed_text, dict) else None,
        )
        record["id"] = new_id
    for clause in evidence["clauses"]:
        clause.update(clause_counts(clause["taxonomy_id"], precedent, party=party))


def clause_counts(
    taxonomy_id: str, precedent: list[dict[str, Any]], *, party: str | None
) -> dict[str, int]:
    """The four ``evidence.clauses[]`` counts one clause's precedent implies.

    - ``n_deals``: distinct deals with a precedent record for the clause.
    - ``n_signed_standard``: distinct signed deals whose signed text is our
      standard language.
    - ``n_variants``: distinct non-standard texts signed (signed deals only),
      grouped by :func:`normalize_variant_text` with *party* (the document's
      ``perspective.party``, :func:`~playbook_engine.opf_accessors.perspective_party`).
    - ``n_refused``: distinct refused-ask texts, grouped the same way.

    Shared by the assembler and the validator's count cross-check, and equal
    to the digest's ``n_variants_total``/``n_refused_total``.
    """
    records = [p for p in precedent if isinstance(p, dict) and p.get("taxonomy_id") == taxonomy_id]
    deals = {p.get("document_id") for p in records}
    standard_deals = {
        p.get("document_id")
        for p in records
        if p.get("signed") is True and p.get("standard") is True and p.get("signed_text")
    }
    variants = {
        key
        for p in records
        if p.get("signed") is True and p.get("standard") is not True
        for key in (_entry_key(p.get("signed_text"), party),)
        if key
    }
    refused = {
        key
        for p in records
        for ask in (p.get("refused_asks") or [])
        for key in (_entry_key(ask, party),)
        if key
    }
    return {
        "n_deals": len(deals),
        "n_signed_standard": len(standard_deals),
        "n_variants": len(variants),
        "n_refused": len(refused),
    }


def _entry_key(entry: Any, party: str | None) -> str:
    if not isinstance(entry, dict):
        return ""
    text = entry.get("text")
    return normalize_variant_text(text, party=party) if isinstance(text, str) else ""
