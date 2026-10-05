"""Model-facing digest of an OPF playbook — the top-level `digest` section.

Two shapes, dispatched on ``opf_version`` by :func:`build_digest`:

- **digest_version 3** (OPF 0.4, issue #223, :func:`build_digest_v3`) — the
  verdict-free projection of the per-deal precedent record: per clause, our
  standard, how many deals signed it, and the non-standard signed variants
  and refused asks, grouped by OPF-SPEC §3.5.4's exact grouping key
  (``precedent.normalize_variant_text``) with distinct-deal counts,
  citations and precedent ids; plus ``perspective``,
  ``agreement_type`` and corpus counts. No stance, band or risk field.
- **digest_version 2** (OPF 0.3 and older, frozen) — described below.

The full OPF document carries every observation's ``full_text`` and measures
in the millions of characters on a real corpus — far beyond what a consuming
review application can put in a model's context. The digest is the compact
projection designed for exactly that use: per clause, the stance, the
preferred/concession/unacceptable variation summaries, and a deduplicated,
frequency-annotated sample of exemplar forms, each carrying an
``example_ref`` citation that resolves into the full playbook for on-demand
drill-down (a consumer fetches ``full_text`` from the full OPF when needed —
the digest itself never contains ``full_text``).

Emitted by ``assemble_playbook`` as the top-level ``digest`` section of an
OPF 0.3 document, and extractable standalone via ``playbook digest``. The
digest is a pure function of the evidence section, so it participates in
``identity.content_hash`` like any other content section.

Size discipline: the budget is ~40K tokens (chars/4 rule of thumb — this
codebase has no tokenizer dependency) and is ENFORCED by construction, not
aspirational: every list — preferred variations, concessions, unacceptable
variations, exemplar forms — is deduplicated by normalized text and capped
at the top-N by evidentiary weight (``n`` = distinct deals) plus
every material-risk group; if the digest still exceeds the budget,
``build_digest`` tightens the cap stepwise (5 → 4 → 3) until it fits.
Surviving entries are never truncated or paraphrased — a preferred
variation's ``if``/``to`` language ships verbatim; only the compiler-
generated ``rationale`` narration is left to the full OPF (reachable via
``observation_ref``).
"""

from __future__ import annotations

import re
from typing import Any

from playbook_engine.canonicalize import canonicalize
from playbook_engine.opf_accessors import clause_stance, perspective_party, playbook_clauses

#: Schema version of the digest section itself — bump on any shape change so
#: consumers can dispatch (the digest is consumed outside this repo).
#: v2: preferred_variations deduped/ranked/capped like the other lists; digest
#: entries carry {if, to, observation_ref, n, band} (rationale stays in the
#: full OPF). ``n`` is the number of distinct deals (``document_id``) behind an
#: entry — changed in place 2026-09-25 (issue #216, owner-authorized exception
#: recorded in spec/CHANGELOG.md); it previously summed ``precedent_count``.
DIGEST_VERSION = "3"

#: The digest version an OPF 0.3 (or older) document's digest carries. A 0.3
#: document keeps digest_version 2 forever — 0.3 is frozen (spec/CHANGELOG.md);
#: ``build_digest`` dispatches on ``opf_version`` (issue #223).
DIGEST_VERSION_V2 = "2"

#: OPF versions whose evidence is the verdict-free per-deal precedent record
#: (issue #223) — their digest is digest_version 3.
_PRECEDENT_OPF_VERSIONS = frozenset({"0.4"})

#: Digest 3 per-list cap (signed_variants, refused_asks): the loosest cap,
#: tightened stepwise down to ``_MIN_TOP_N_V3`` until the digest fits the
#: token budget. Unlike digest 2 there is no material-risk exception — risk
#: is a judged verdict and never reaches the consumer path.
_MIN_TOP_N_V3 = 1

#: List selection (all four lists): keep the top N deduplicated entries by
#: observation count, plus every entry carrying material risk regardless of
#: rank. This is the default/loosest cap; build_digest tightens it to fit
#: the token budget.
EXEMPLAR_TOP_N = 5

#: The hard size budget build_digest enforces (chars/4 rule of thumb).
DIGEST_TOKEN_BUDGET = 40_000

#: build_digest never tightens the per-list cap below this.
_MIN_TOP_N = 3

#: Frequency bands for exemplar forms — coarse language a reviewing model can
#: use directly ("often signed as…") without re-deriving statistics.
_BAND_OFTEN_MIN = 10
_BAND_SOMETIMES_MIN = 2


def _normalize_text(text: str) -> str:
    """Normalization used to dedupe exemplar forms (case/punct/ws-insensitive)."""
    s = text.lower()
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _band(n: int) -> str:
    if n >= _BAND_OFTEN_MIN:
        return "often"
    if n >= _BAND_SOMETIMES_MIN:
        return "sometimes"
    return "rare"


def _is_material(obs: dict[str, Any]) -> bool:
    risk = obs.get("risk_delta") or {}
    return isinstance(risk, dict) and risk.get("magnitude") == "material"


def _deal_key(ref: Any) -> tuple[str, str]:
    """The deal a digest row counts toward: its citation's ``document_id``.

    Issue #216: the deal is the unit of precedent, so every digest ``n`` is
    the number of DISTINCT deals in a group — never a sum of
    ``precedent_count``, which the compiler stamps on every row of a text
    (summing it reported a text signed in k deals as n = k*k). Every OPF
    observation carries ``example_ref.document_id`` (required by every
    playbook schema), so a row without one is not a conforming observation
    and raises rather than being counted as a guessed deal.
    """
    if isinstance(ref, dict) and ref.get("document_id") is not None:
        return ("doc", str(ref["document_id"]))
    raise ValueError(
        "digest: observation has no example_ref.document_id — every OPF "
        "observation must cite the deal it was observed in"
    )


def _dedupe_rank(
    observations: list[dict[str, Any]], *, include_deviation: bool, top_n: int = EXEMPLAR_TOP_N
) -> list[dict[str, Any]]:
    """Dedupe observations by normalized text and rank by frequency.

    The digest's one size discipline, applied uniformly to exemplar forms,
    concessions, and unacceptable variations: group by the normalized
    ``full_text`` (falling back to ``text_summary``); ``n`` is the number of
    distinct deals (``example_ref.document_id``, see ``_deal_key``) among the
    group's members — never a sum of ``precedent_count`` (issue #216); keep
    the top ``EXEMPLAR_TOP_N`` groups by ``n`` plus every group containing
    material risk, in rank order. Output entries carry ``text_summary``
    ONLY — never ``full_text``; ``example_ref`` is the drill-down path.
    """
    groups: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for obs in observations:
        key = _normalize_text(str(obs.get("full_text") or obs.get("text_summary") or ""))
        if not key:
            continue
        if key not in groups:
            groups[key] = {"n": 0, "deals": set(), "rep": obs, "material": False}
            order.append(key)
        g = groups[key]
        g["deals"].add(_deal_key(obs.get("example_ref")))
        g["n"] = len(g["deals"])
        if _is_material(obs):
            g["material"] = True
            # A material observation is the most informative representative.
            g["rep"] = obs

    first_seen = {k: i for i, k in enumerate(order)}
    ranked = sorted(order, key=lambda k: (-groups[k]["n"], first_seen[k]))
    keep = set(ranked[:top_n]) | {k for k in ranked if groups[k]["material"]}

    forms: list[dict[str, Any]] = []
    for key in ranked:
        if key not in keep:
            continue
        g = groups[key]
        rep = g["rep"]
        form: dict[str, Any] = {
            "text_summary": rep.get("text_summary", ""),
            "n": g["n"],
            "band": _band(g["n"]),
        }
        if include_deviation and rep.get("deviation") is not None:
            form["deviation"] = rep["deviation"]
        if rep.get("risk_delta") is not None:
            form["risk_delta"] = rep["risk_delta"]
        if rep.get("example_ref") is not None:
            form["example_ref"] = rep["example_ref"]
        forms.append(form)
    return forms


def _exemplar_forms(
    observed_positions: list[dict[str, Any]], top_n: int = EXEMPLAR_TOP_N
) -> list[dict[str, Any]]:
    return _dedupe_rank(observed_positions, include_deviation=True, top_n=top_n)


def _preferred_variations(clause: dict[str, Any], top_n: int) -> list[Any]:
    """Project acceptable_if entries to the digest: dedupe, rank, cap.

    Same discipline as the other three lists. Grouping key: the normalized
    ``if``+``to`` text (or the whole entry for legacy bare strings). Rank
    weight ``n``: the number of distinct deals (issue #216) behind the group
    — each entry's own ``observation_ref`` deal plus every deal among the
    clause's ``observed_positions`` whose normalized text equals the entry's
    ``to`` language and whose ``outcome`` is ``"signed"`` (the compiler
    lists each accepted text once, so its deals are read off the signed
    positions carrying it; a deal that refused the text is not acceptance
    precedent); never a sum of
    ``precedent_count``. The underlying observation (for the material-risk
    check) is resolved by matching ``observation_ref`` against
    ``observed_positions``. Surviving dict entries ship
    ``if``/``to`` VERBATIM plus ``observation_ref``, ``n``, and ``band`` —
    the compiler-generated ``rationale`` narration stays in the full OPF.
    Legacy bare-string entries pass through as strings.
    """
    entries = (clause.get("summary") or {}).get("acceptable_if") or []
    if not entries:
        return []

    obs_by_ref: dict[tuple[Any, Any, Any], dict[str, Any]] = {}
    # Normalized observed text -> the distinct deals carrying it (issue #216).
    deals_by_text: dict[str, set[tuple[str, Any]]] = {}
    for pos in clause.get("observed_positions") or []:
        pos_ref = pos.get("example_ref") or {}
        obs_by_ref[
            (pos_ref.get("document_id"), pos_ref.get("version"), pos_ref.get("clause_path"))
        ] = pos
        # Only SIGNED positions are acceptance precedent (the compiler builds
        # acceptable_if from signed rows only): a deal where the same text
        # was proposed_then_reversed refused it, and never counts here.
        if pos.get("outcome") != "signed":
            continue
        text_key = _normalize_text(str(pos.get("full_text") or pos.get("text_summary") or ""))
        if text_key:
            deals_by_text.setdefault(text_key, set()).add(_deal_key(pos_ref))

    groups: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for i, entry in enumerate(entries):
        obs: dict[str, Any] = {}
        deals: set[tuple[str, Any]] = set()
        if isinstance(entry, str):
            key = _normalize_text(entry)
            deals |= deals_by_text.get(key, set())
        else:
            key = _normalize_text(f"{entry.get('if', '')} {entry.get('to', '')}")
            ref = entry.get("observation_ref") or {}
            obs = obs_by_ref.get(
                (ref.get("document_id"), ref.get("version"), ref.get("clause_path")), {}
            )
            deals |= deals_by_text.get(_normalize_text(str(entry.get("to") or "")), set())
            if isinstance(ref, dict) and ref.get("document_id") is not None:
                deals.add(_deal_key(ref))
        if not key:
            continue
        if not deals:
            # Nothing resolvable (a legacy bare-string entry, which the
            # schema allows, with no signed position carrying its text): the
            # entry itself is one deal's evidence.
            deals.add(("entry", i))
        n = len(deals)
        if key not in groups:
            groups[key] = {"n": 0, "deals": set(), "rep": entry, "rep_n": -1, "material": False}
            order.append(key)
        g = groups[key]
        g["deals"] |= deals
        g["n"] = len(g["deals"])
        if _is_material(obs):
            g["material"] = True
        if n > g["rep_n"]:
            g["rep"], g["rep_n"] = entry, n

    first_seen = {k: i for i, k in enumerate(order)}
    ranked = sorted(order, key=lambda k: (-groups[k]["n"], first_seen[k]))
    keep = set(ranked[:top_n]) | {k for k in ranked if groups[k]["material"]}

    out: list[Any] = []
    for key in ranked:
        if key not in keep:
            continue
        g = groups[key]
        rep = g["rep"]
        if isinstance(rep, str):
            out.append(rep)
            continue
        projected: dict[str, Any] = {"if": rep.get("if"), "to": rep.get("to")}
        if rep.get("observation_ref") is not None:
            projected["observation_ref"] = rep["observation_ref"]
        projected["n"] = g["n"]
        projected["band"] = _band(g["n"])
        out.append(projected)
    return out


def _build_digest_at(playbook: dict[str, Any], top_n: int) -> dict[str, Any]:
    """Build the digest with a fixed per-list cap of *top_n*."""
    digest_clauses: list[dict[str, Any]] = []
    for clause in playbook_clauses(playbook):
        summary = clause.get("summary") or {}
        our_standard = clause.get("our_standard")
        entry: dict[str, Any] = {
            "id": clause.get("id"),
            "taxonomy_id": clause.get("taxonomy_id"),
            "title": clause.get("title"),
            "historical_stance": clause_stance(clause),
            "stance_detail": summary.get("stance_detail"),
            "our_standard": our_standard if isinstance(our_standard, dict) else None,
            "preferred_variations": _preferred_variations(clause, top_n),
            "concessions": _dedupe_rank(
                summary.get("fallbacks") or [], include_deviation=False, top_n=top_n
            ),
            "unacceptable": _dedupe_rank(
                summary.get("rejected") or [], include_deviation=False, top_n=top_n
            ),
            "exemplar_forms": _exemplar_forms(clause.get("observed_positions") or [], top_n),
        }
        digest_clauses.append(entry)

    return {
        "digest_version": DIGEST_VERSION_V2,
        "clause_count": len(digest_clauses),
        "clauses": digest_clauses,
    }


def build_digest(
    playbook: dict[str, Any], *, token_budget: int | None = DIGEST_TOKEN_BUDGET
) -> dict[str, Any]:
    """Build the digest section from an assembled playbook's evidence.

    Dispatches on ``opf_version``: an OPF 0.4 document (the verdict-free
    per-deal precedent record, issue #223) gets a digest_version 3 digest
    (:func:`build_digest_v3`); every older document (0.1/0.2/0.3) gets the
    frozen digest_version 2 shape, so the CLI can still derive a digest from
    a pre-0.4 artifact and a 0.3 document's digest never changes.

    Enforces *token_budget* by construction: starts at the default per-list
    cap (``EXEMPLAR_TOP_N``) and tightens it stepwise until the digest fits.
    Pass ``token_budget=None`` for the loosest cap unconditionally. For
    digest 2 the cap stops at ``_MIN_TOP_N`` and material-risk entries are
    never dropped, so an extreme corpus can still exceed the budget — the
    CLI warns in that case.
    """
    if playbook.get("opf_version") in _PRECEDENT_OPF_VERSIONS:
        return build_digest_v3(playbook, token_budget=token_budget)
    digest = _build_digest_at(playbook, EXEMPLAR_TOP_N)
    if token_budget is None:
        return digest
    for top_n in range(EXEMPLAR_TOP_N - 1, _MIN_TOP_N - 1, -1):
        if digest_token_estimate(digest) <= token_budget:
            break
        digest = _build_digest_at(playbook, top_n)
    return digest


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
    """``evidence.precedent`` of an OPF 0.4 document (``[]`` when absent)."""
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


def _build_digest_v3_at(playbook: dict[str, Any], top_n: int | None) -> dict[str, Any]:
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


def build_digest_v3(
    playbook: dict[str, Any], *, token_budget: int | None = DIGEST_TOKEN_BUDGET
) -> dict[str, Any]:
    """Build a digest_version 3 digest from an OPF 0.4 document (issue #223).

    Verdict-free: per clause, our standard, how many deals signed it, the
    non-standard signed variants and the refused asks, each grouped by
    normalized text with its distinct-deal count, latest signing date,
    citation and the precedent ids behind it. No stance, band, risk or
    deviation field; no ``full_text`` (variant and ask ``text`` is the
    ≤ 300-char sentence-boundary summary). ``perspective`` is always present
    (``null`` when the document has none) so a consumer never has to read
    it from elsewhere.

    The token budget is enforced by construction: both lists start capped
    at ``EXEMPLAR_TOP_N`` and the cap tightens stepwise to ``_MIN_TOP_N_V3``
    until the digest fits. ``n_variants_total``/``n_refused_total`` always
    report the uncapped totals, so a consumer knows what the cap left out.
    """
    digest = _build_digest_v3_at(playbook, EXEMPLAR_TOP_N)
    if token_budget is None:
        return digest
    for top_n in range(EXEMPLAR_TOP_N - 1, _MIN_TOP_N_V3 - 1, -1):
        if digest_token_estimate(digest) <= token_budget:
            break
        digest = _build_digest_v3_at(playbook, top_n)
    return digest


def digest_token_estimate(digest: dict[str, Any]) -> int:
    """Rough token estimate — canonical chars / 4, the repo-wide rule of thumb."""
    return len(canonicalize(digest)) // 4
