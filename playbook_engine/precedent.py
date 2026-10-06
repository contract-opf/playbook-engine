"""OPF 0.4 evidence — the verdict-free per-deal precedent record (issue #223).

OPF 0.3's evidence shape (``observed_positions``, ``clause_library``,
``summary.{historical_stance, acceptable_if, fallbacks, rejected,
confidence}``, ``negotiation_trail``) describes categories the engine can no
longer honestly fill once judged deviation verdicts left the consumer path
(issue #220) and the deal became the unit of precedent (issue #216). OPF 0.4
replaces it with two lists:

``evidence.clauses[]``
    One entry per clause type: ``{id, taxonomy_id, title, our_standard,
    n_deals, n_signed_standard, n_variants, n_refused}``. Every count is
    derived from ``evidence.precedent`` (the validator recomputes them).

``evidence.precedent[]``
    One record per (deal, clause type): what that deal signed for the clause
    (``signed_text``), whether that text is OUR standard language
    (``standard`` — the deterministic exact check of issue #220, never a
    judged verdict), whether the clause moved during the negotiation, and
    the counterparty asks refused before signing (``refused_asks``). Paper
    side is carried as honest metadata (``paper``) and never partitions,
    gates or weights anything (owner decision 2026-09-13 (b)).

Nothing here is judged. Judged deviation verdicts, when an opt-in
``--with-deviation-judge`` run produced them, travel under the vendor
namespace ``x_judgments`` keyed by precedent id (see
:func:`build_x_judgments`) and never enter ``evidence.precedent``.

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
    OUTCOME_CONCEDED_BEFORE_SIGNING,
    Observation,
    RoundMove,
)
from playbook_engine.opf_accessors import perspective_party

__all__ = [
    "PAPER_OURS",
    "PAPER_THEIRS",
    "PAPER_UNKNOWN",
    "PRECEDENT_ID_PREFIX",
    "build_precedent_evidence",
    "build_x_judgments",
    "clause_counts",
    "normalize_variant_text",
    "paper_of_corpus_document",
    "precedent_id",
    "refresh_derived",
    "restamp_evidence",
]

#: Every precedent id starts with this prefix (OPF-SPEC §3.5, OPF 0.4).
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
_JUDGED_BASIS = "judge"
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
    """Grouping key for signed variants and refused asks (OPF 0.4 / digest 3).

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
    side. Metadata only: nothing in OPF 0.4 partitions, gates or weights by
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


def _clause_order_key(obs: Observation) -> tuple[int, int]:
    """Order rows of one (deal, clause) by where their text sits in its draft.

    A cited draft ordinal first (a non-integer version sorts as the opening
    draft), then the cited span's start; a row with no span sorts last.
    """
    version = obs.citation.version
    span = obs.citation.char_span
    return (
        version if isinstance(version, int) else 0,
        span[0] if span else 2**63 - 1,
    )


def _joined_text_entry(rows: list[Observation]) -> dict[str, Any] | None:
    """Every row's text joined in clause order with ``"\\n"``, citing the first.

    observation_builder writes one ``conceded_before_signing`` row per struck
    node of a clause, so a deal that struck several nodes of a multi-node
    standard has several rows; joining them exactly as observation_builder
    joins a terminal group's nodes keeps every struck text (no row is dropped).
    ``None`` when there are no rows.
    """
    if not rows:
        return None
    ordered = sorted(rows, key=_clause_order_key)
    return {
        "text": "\n".join(o.full_text for o in ordered if o.full_text),
        "ref": _ref(ordered[0]),
    }


def _is_signed_deal(
    corpus_doc: dict[str, Any] | None, terminal: Observation | None, others: Iterable[Observation]
) -> bool:
    """Whether the deal has a detected executed copy.

    ``corpus.documents[].signed_version`` is authoritative when recorded;
    otherwise the store's own outcomes decide (a ``"signed"`` terminal row or
    a ``conceded_before_signing`` row both exist only in a deal with a
    detected executed copy).
    """
    if corpus_doc is not None and "signed_version" in corpus_doc:
        return corpus_doc.get("signed_version") is not None
    if terminal is not None:
        return terminal.outcome == "signed"
    return any(o.outcome == OUTCOME_CONCEDED_BEFORE_SIGNING for o in others)


def build_precedent_evidence(
    *,
    agreement_type_id: str,
    clause_positions: list[ClausePosition],
    observations: list[Observation],
    corpus_documents: list[dict[str, Any]],
    round_moves: list[RoundMove] | None = None,
    party: str | None,
) -> dict[str, Any]:
    """Build the OPF 0.4 ``evidence`` section: ``{clauses, precedent}``.

    One precedent per (document_id, taxonomy_id) carrying any classified
    observation of a compiled clause type:

    - ``signed_text``: the deal's terminal text for the clause (the executed
      copy when ``signed`` is true, the last draft when it is false), or
      ``null`` when the clause was struck before signing.
    - ``opening_text``: the text the deal opened with, when the store records
      it as distinct evidence — today our standard language struck before
      signing (``conceded_before_signing``): every such row of the (deal,
      clause), joined in clause order with ``"\\n"`` and citing the first
      (one row is written per struck node); ``null`` otherwise. ``null``
      means "not recorded", never "opened with the signed text".
    - ``standard``: the terminal row's deterministic standard fact
      (exact match after ``normalize_for_standard``, issue #220); false when
      there is no terminal text.
    - ``rounds``: the number of distinct negotiation rounds in which the
      clause changed (``round_moves``); ``moved``: it changed, was struck,
      or carried a refused ask.
    - ``refused_asks``: every ``proposed_then_reversed`` row — text proposed
      and then struck before signing — with its round and citation; always
      empty when ``signed`` is false (issue #221).

    ``signed_at`` is never set: the reference compiler extracts no signing
    date and never fabricates one (OPF-SPEC §3.5.4), so the digest's
    ``first_signed`` / ``last_signed`` and every
    ``signed_variants[].last_signed`` are always ``null`` in reference
    output. The optional field exists for a third-party producer that
    records signing dates; the 0.4 conformance vectors that set it model one.

    Sub-sentence fragments are excluded exactly as the 0.3 compiler excludes
    them (``MIN_OBSERVATION_TEXT_LEN``); unclassified observations and clause
    types with no compiled position never reach precedent.

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
        # Every conceded row of the (deal, clause), never only the first: one
        # row is written per struck node, so a deal that struck two nodes of
        # a multi-node standard has two rows and both texts belong here.
        conceded = [o for o in rows if o.outcome == OUTCOME_CONCEDED_BEFORE_SIGNING]
        corpus_doc = docs_by_id.get(document_id)
        signed = _is_signed_deal(corpus_doc, terminal, rows)
        # Issue #221: a deal with no detected executed copy records no
        # refused asks — with no signed terminal, "struck before signing" is
        # not established. observation_builder no longer writes such rows;
        # this also keeps an observations.jsonl mined before the fix from
        # reintroducing them at project time.
        refused = [o for o in rows if o.outcome == _REFUSED_OUTCOME] if signed else []
        if terminal is None and not conceded and not refused:
            continue
        paper, paper_basis, paper_confidence = _paper(corpus_doc, rows)
        signed_text = _text_entry(terminal) if terminal is not None else None
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
                "opening_text": _joined_text_entry(conceded),
                "standard": bool(terminal is not None and terminal.standard),
                "moved": bool(n_rounds or conceded or refused),
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
) -> dict[str, str]:
    """(Re)compute every derived value of an OPF 0.4 ``evidence`` in place.

    Each precedent's ``id`` (:func:`precedent_id`) and each clause's
    ``n_*`` counts (:func:`clause_counts`, grouping by
    :func:`normalize_variant_text` with *party* — the document's
    ``perspective.party``) are functions of the texts they describe. The
    assembler strips zero-width/bidi-control characters from the whole
    document after building it, so it calls this again afterwards — the ids
    and counts a validator recomputes over the shipped text must be the ones
    the document carries.

    Returns the ``{old_id: new_id}`` map of every id that changed, so a
    caller can re-key anything that references precedent ids
    (``x_judgments``).
    """
    precedent = evidence["precedent"]
    renamed: dict[str, str] = {}
    for record in precedent:
        signed_text = record.get("signed_text")
        new_id = precedent_id(
            agreement_type_id,
            record["document_id"],
            record["taxonomy_id"],
            signed_text["text"] if isinstance(signed_text, dict) else None,
        )
        old_id = record.get("id")
        if isinstance(old_id, str) and old_id and old_id != new_id:
            renamed[old_id] = new_id
        record["id"] = new_id
    for clause in evidence["clauses"]:
        clause.update(clause_counts(clause["taxonomy_id"], precedent, party=party))
    return renamed


def refresh_derived(doc: dict[str, Any]) -> None:
    """Re-derive everything an OPF 0.4 document computes from its own text.

    For a transform that rewrites evidence text after assembly (``playbook
    publish``'s scrub/redaction, ``export_profile``'s residue rewrites): the
    precedent ids and clause counts (:func:`restamp_evidence`), every
    ``x_judgments[].precedent_id`` that referenced a renamed id, and the
    digest (rebuilt with ``digest.build_digest`` so it carries the
    transformed text — never a stale copy of the pre-transform text — and
    the transformed ``perspective``; the counts and the digest group texts
    with the transformed ``perspective.party``). ``identity`` is left to the
    caller, which re-stamps it last. A no-op on any pre-0.4 document.
    """
    evidence = doc.get("evidence")
    agreement_type = doc.get("agreement_type")
    if (
        doc.get("opf_version") != "0.4"
        or not isinstance(evidence, dict)
        or not isinstance(evidence.get("precedent"), list)
        or not isinstance(evidence.get("clauses"), list)
        or not isinstance(agreement_type, dict)
    ):
        return
    renamed = restamp_evidence(
        evidence, str(agreement_type.get("id")), party=perspective_party(doc)
    )
    judgments = doc.get("x_judgments")
    if renamed and isinstance(judgments, list):
        for judgment in judgments:
            if isinstance(judgment, dict) and judgment.get("precedent_id") in renamed:
                judgment["precedent_id"] = renamed[judgment["precedent_id"]]
    if "digest" in doc:
        from playbook_engine.digest import build_digest  # noqa: PLC0415

        doc["digest"] = build_digest(doc)


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


def build_x_judgments(
    observations: list[Observation],
    precedent: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Judged deviation verdicts for an opt-in judged run, keyed by precedent.

    Only rows a real judge assessed (``basis == "judge"``) — an opt-in
    ``--with-deviation-judge`` run (issue #220); a consumer-path store yields
    ``[]``. Each entry is ``{precedent_id, deviation, risk_delta, basis}`` for the
    precedent's terminal row. These are vendor-namespace data
    (``x_judgments``) — never part of ``evidence.precedent`` and never read
    by the digest.
    """
    by_key = {(p["document_id"], p["taxonomy_id"]): p["id"] for p in precedent}
    out: list[dict[str, Any]] = []
    for obs in observations:
        if obs.outcome not in _TERMINAL_OUTCOMES or obs.taxonomy_id is None:
            continue
        if obs.basis != _JUDGED_BASIS:
            # Only a real judge verdict is a judgment; deterministic, stub
            # and judge-fallback rows assert nothing judged.
            continue
        pid = by_key.get((obs.citation.document_id, obs.taxonomy_id))
        if pid is None:
            continue
        out.append(
            {
                "precedent_id": pid,
                "deviation": obs.deviation,
                "risk_delta": dict(obs.risk_delta),
                "basis": obs.basis,
            }
        )
    out.sort(key=lambda j: j["precedent_id"])
    return out
