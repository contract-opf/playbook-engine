"""Model-facing digest of an OPF playbook — the top-level `digest` section.

**digest_version 3** (OPF 0.4, issue #223; the only digest the engine
builds — digest 2 was retired with OPF 0.3, issue #238): the verdict-free
projection of the per-deal precedent record. Per clause: our standard, how
many deals signed it, and the non-standard signed variants and refused asks,
grouped by OPF-SPEC §3.5.4's exact grouping key
(``precedent.normalize_variant_text``) with distinct-deal counts, citations
and precedent ids; plus ``perspective``, ``agreement_type`` and corpus
counts. No stance, band or risk field.

The full OPF document carries every precedent's full text and measures in
the millions of characters on a real corpus — far beyond what a consuming
review application can put in a model's context. The digest is the compact
projection designed for exactly that use: every variant and ask carries a
``ref`` citation and its ``precedent_ids``, which resolve into the full
playbook for on-demand drill-down (the digest itself never contains
``full_text``).

Emitted by ``assemble_playbook`` as the top-level ``digest`` section, and
extractable standalone via ``playbook digest``. The digest is a pure
function of the document, so it participates in ``identity.content_hash``
like any other content section.

Size discipline: the budget is ~40K tokens (chars/4 rule of thumb — this
codebase has no tokenizer dependency) and is ENFORCED by construction, not
aspirational: both lists start capped at ``EXEMPLAR_TOP_N`` and the cap
tightens stepwise until the digest fits; ``n_variants_total`` /
``n_refused_total`` always report the uncapped totals.
"""

from __future__ import annotations

import re
from typing import Any

from playbook_engine.canonicalize import canonicalize
from playbook_engine.opf_accessors import perspective_party, playbook_clauses

#: Schema version of the digest section itself — bump on any shape change so
#: consumers can dispatch (the digest is consumed outside this repo).
DIGEST_VERSION = "3"

#: Per-list cap (signed_variants, refused_asks): the default/loosest cap,
#: tightened stepwise down to ``_MIN_TOP_N`` until the digest fits the
#: token budget. There is no material-risk exception — risk is a judged
#: verdict and never reaches the consumer path.
EXEMPLAR_TOP_N = 5

#: build_digest never tightens the per-list cap below this.
_MIN_TOP_N = 1

#: The hard size budget build_digest enforces (chars/4 rule of thumb).
DIGEST_TOKEN_BUDGET = 40_000


# ---------------------------------------------------------------------------
# digest_version 3 — projection of the OPF 0.4 precedent record (issue #223)
# ---------------------------------------------------------------------------

_QUARTER_RE = re.compile(r"^(\d{4})-Q([1-4])$")
_DATE_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")


def _signed_at_key(value: Any) -> tuple[int, int, int] | None:
    """Sortable key for a ``signed_at`` (``YYYY-MM-DD`` or ``YYYY-Qn``).

    A quarter sorts at its first day, so a published (quarter-coarsened)
    document orders the same way its exact-date source did at quarter
    granularity. Anything else is ``None`` (unknown — sorts last).
    """
    if not isinstance(value, str):
        return None
    m = _DATE_RE.match(value)
    if m:
        return (int(m.group(1)), int(m.group(2)), int(m.group(3)))
    m = _QUARTER_RE.match(value)
    if m:
        return (int(m.group(1)), (int(m.group(2)) - 1) * 3 + 1, 1)
    return None


def _latest(values: list[Any]) -> Any:
    """The latest ``signed_at`` among *values* (``None`` when none is known)."""
    keyed = [(k, v) for v in values for k in (_signed_at_key(v),) if k is not None]
    return max(keyed)[1] if keyed else None


def _earliest(values: list[Any]) -> Any:
    keyed = [(k, v) for v in values for k in (_signed_at_key(v),) if k is not None]
    return min(keyed)[1] if keyed else None


def _latest_first_key(record: dict[str, Any]) -> tuple[bool, tuple[int, ...], str]:
    """Sort key: latest ``signed_at`` first (unknown last), then document_id."""
    k = _signed_at_key(record.get("signed_at"))
    return (k is None, tuple(-x for x in (k or (0, 0, 0))), str(record.get("document_id")))


def playbook_precedent_records(playbook: dict[str, Any]) -> list[dict[str, Any]]:
    """``evidence.precedent`` of the document (``[]`` when absent)."""
    evidence = playbook.get("evidence")
    records = evidence.get("precedent") if isinstance(evidence, dict) else None
    return [p for p in records if isinstance(p, dict)] if isinstance(records, list) else []


def clause_precedent_groups(
    taxonomy_id: Any, precedent: list[dict[str, Any]], *, party: str | None
) -> dict[str, list[dict[str, Any]]]:
    """Every signed variant and refused ask of one clause, grouped and ranked.

    Uncapped — the digest caps these lists, and the HTML/prompt renderers
    read them whole. Grouping is :func:`precedent.normalize_variant_text`
    with *party*, the document's ``perspective.party``
    (``opf_accessors.perspective_party``) — exact after normalization, no
    similarity tolerance.

    - ``signed_variants``: non-standard text signed in a signed deal. Each
      group: ``{text, n_deals, last_signed, ref, precedent_ids}`` — ``text``
      is the sentence-boundary summary (≤ 300 chars,
      ``summarize_clause_text``) of the group's representative (latest
      ``signed_at``, then lowest ``document_id``), ``ref`` its citation.
      Ordered ``n_deals`` desc, ``last_signed`` desc (unknown last), then
      normalized text.
    - ``refused_asks``: text proposed and struck before signing, across all
      deals. Each group: ``{text, n_deals, ref, precedent_ids}``, the
      representative being the earliest-round ask (then lowest
      ``document_id``). Ordered ``n_deals`` desc, then normalized text.

    ``n_deals`` counts distinct deals (``document_id``) — never rows.
    """
    from playbook_engine.observation_builder import summarize_clause_text  # noqa: PLC0415
    from playbook_engine.precedent import normalize_variant_text  # noqa: PLC0415

    records = [p for p in precedent if p.get("taxonomy_id") == taxonomy_id]

    variants: dict[str, dict[str, Any]] = {}
    for p in records:
        signed_text = p.get("signed_text")
        if p.get("signed") is not True or p.get("standard") is True:
            continue
        if not isinstance(signed_text, dict) or not isinstance(signed_text.get("text"), str):
            continue
        key = normalize_variant_text(signed_text["text"], party=party)
        if not key:
            continue
        g = variants.setdefault(key, {"members": []})
        g["members"].append(p)

    variant_out: list[tuple[Any, dict[str, Any]]] = []
    for key, g in variants.items():
        members = g["members"]
        rep = sorted(members, key=_latest_first_key)[0]
        last_signed = _latest([m.get("signed_at") for m in members])
        n = len({m.get("document_id") for m in members})
        entry = {
            "text": summarize_clause_text(rep["signed_text"]["text"]),
            "n_deals": n,
            "last_signed": last_signed,
            "ref": rep["signed_text"].get("ref"),
            "precedent_ids": sorted({str(m.get("id")) for m in members}),
        }
        last_key = _signed_at_key(last_signed)
        sort_key = (
            -n,
            last_key is None,
            tuple(-x for x in (last_key or (0, 0, 0))),
            key,
        )
        variant_out.append((sort_key, entry))
    variant_out.sort(key=lambda t: t[0])

    refused: dict[str, dict[str, Any]] = {}
    for p in records:
        for ask in p.get("refused_asks") or []:
            if not isinstance(ask, dict) or not isinstance(ask.get("text"), str):
                continue
            key = normalize_variant_text(ask["text"], party=party)
            if not key:
                continue
            g = refused.setdefault(key, {"asks": [], "deals": set(), "ids": set()})
            g["asks"].append((ask, p))
            g["deals"].add(p.get("document_id"))
            g["ids"].add(str(p.get("id")))

    refused_out: list[tuple[Any, dict[str, Any]]] = []
    for key, g in refused.items():
        rep_ask, _rep_p = sorted(
            g["asks"],
            key=lambda ap: (
                ap[0].get("round") if isinstance(ap[0].get("round"), int) else 0,
                str(ap[1].get("document_id")),
            ),
        )[0]
        n = len(g["deals"])
        entry = {
            "text": summarize_clause_text(rep_ask["text"]),
            "n_deals": n,
            "ref": rep_ask.get("ref"),
            "precedent_ids": sorted(g["ids"]),
        }
        refused_out.append(((-n, key), entry))
    refused_out.sort(key=lambda t: t[0])

    return {
        "signed_variants": [e for _, e in variant_out],
        "refused_asks": [e for _, e in refused_out],
    }


def _corpus_summary(playbook: dict[str, Any], precedent: list[dict[str, Any]]) -> dict[str, Any]:
    """``digest.corpus``: in-scope deal counts and the signed-date range."""
    corpus = playbook.get("corpus")
    documents = corpus.get("documents") if isinstance(corpus, dict) else None
    docs = [d for d in documents or [] if isinstance(d, dict) and d.get("in_scope", True)]
    signed_ids = {p.get("document_id") for p in precedent if p.get("signed") is True}
    n_signed = 0
    for d in docs:
        if "signed_version" in d:
            n_signed += d.get("signed_version") is not None
        else:
            n_signed += d.get("document_id") in signed_ids
    signed_ats = [p.get("signed_at") for p in precedent if p.get("signed") is True]
    return {
        "n_deals": len(docs),
        "n_signed": n_signed,
        "first_signed": _earliest(signed_ats),
        "last_signed": _latest(signed_ats),
    }


def _build_digest_at(playbook: dict[str, Any], top_n: int | None) -> dict[str, Any]:
    """Digest 3 with a fixed per-list cap of *top_n* (``None`` = uncapped)."""
    precedent = playbook_precedent_records(playbook)
    party = perspective_party(playbook)
    clauses: list[dict[str, Any]] = []
    for clause in playbook_clauses(playbook):
        tid = clause.get("taxonomy_id")
        groups = clause_precedent_groups(tid, precedent, party=party)
        variants = groups["signed_variants"]
        refused = groups["refused_asks"]
        our_standard = clause.get("our_standard")
        clauses.append(
            {
                "id": clause.get("id"),
                "taxonomy_id": tid,
                "title": clause.get("title"),
                "our_standard": our_standard if isinstance(our_standard, dict) else None,
                "n_deals": clause.get("n_deals"),
                "n_signed_standard": clause.get("n_signed_standard"),
                "signed_variants": variants if top_n is None else variants[:top_n],
                "refused_asks": refused if top_n is None else refused[:top_n],
                "n_variants_total": len(variants),
                "n_refused_total": len(refused),
            }
        )
    agreement_type = playbook.get("agreement_type")
    perspective = playbook.get("perspective")
    return {
        "digest_version": DIGEST_VERSION,
        "perspective": perspective if isinstance(perspective, dict) else None,
        "agreement_type": (
            {k: agreement_type[k] for k in ("id", "name") if k in agreement_type}
            if isinstance(agreement_type, dict)
            else None
        ),
        "corpus": _corpus_summary(playbook, precedent),
        "clauses": clauses,
    }


def build_digest(
    playbook: dict[str, Any], *, token_budget: int | None = DIGEST_TOKEN_BUDGET
) -> dict[str, Any]:
    """Build the digest_version 3 digest of an OPF 0.4 document (issue #223).

    Verdict-free: per clause, our standard, how many deals signed it, the
    non-standard signed variants and the refused asks, each grouped by
    normalized text with its distinct-deal count, latest signing date,
    citation and the precedent ids behind it. No stance, band, risk or
    deviation field; no ``full_text`` (variant and ask ``text`` is the
    ≤ 300-char sentence-boundary summary). ``perspective`` is always present
    (``null`` when the document has none) so a consumer never has to read
    it from elsewhere.

    The token budget is enforced by construction: both lists start capped
    at ``EXEMPLAR_TOP_N`` and the cap tightens stepwise to ``_MIN_TOP_N``
    until the digest fits. ``n_variants_total``/``n_refused_total`` always
    report the uncapped totals, so a consumer knows what the cap left out.
    Pass ``token_budget=None`` for the loosest cap unconditionally.
    """
    digest = _build_digest_at(playbook, EXEMPLAR_TOP_N)
    if token_budget is None:
        return digest
    for top_n in range(EXEMPLAR_TOP_N - 1, _MIN_TOP_N - 1, -1):
        if digest_token_estimate(digest) <= token_budget:
            break
        digest = _build_digest_at(playbook, top_n)
    return digest


def digest_token_estimate(digest: dict[str, Any]) -> int:
    """Rough token estimate — canonical chars / 4, the repo-wide rule of thumb."""
    return len(canonicalize(digest)) // 4
