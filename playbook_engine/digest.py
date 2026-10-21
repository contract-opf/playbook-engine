"""Model-facing digest of an OPF playbook — the top-level `digest` section.

**digest_version 5** (OPF 0.6, issues #223, #234, #240; the only digest the
engine builds — digest 2 was retired with OPF 0.3, issue #238, and digest 3
was replaced in place by this one): the verdict-free projection of the
per-deal precedent record. Per clause: our standard, how many deals signed
it, how many opened with it and kept it, a positions count (signed our
standard, or each ``vs_standard`` label, or unjudged), the non-standard signed
variants (each saying how many deals conceded it from our standard and how
many signed it exactly as the counterparty proposed), the refused asks and
the non-standard openings that were not signed as proposed, grouped by
OPF-SPEC §3.5.4's exact grouping key (``precedent.normalize_variant_text``)
with distinct-deal counts, citations and precedent ids; plus ``perspective``,
``agreement_type``, corpus counts and the clause types the playbook has no
evidence for at all. No stance, band or risk field. The one judged value is
the ``vs_standard`` judgment (issues #240, #243), carried as an index the
consumer checks against the cited text, never an instruction: each judged
text's label plus its compact part map — one token per part of our standard
(present and equivalent, more or less protective, or omitted with its
effect), one token per addition, ``label_by: "owner"`` when the owner set the
label over the parts (absent, the label is derived from them) and the
one-sentence reason (the owner's when the owner set the label). Our standard is carried as its ordered parts, and
``part_map_legend`` says what each token means. Per clause, ``standard_history``
(issue #249) lists each of OUR former forms with its in-use dates, the deals
that opened on it (kept, conceded, moved to today's) and how its clause
compares with today's standard part by part, plus the owner's optional
``why_changed`` note — the only reason a change is ever given.

The full OPF document carries every precedent's full text and measures in
the millions of characters on a real corpus — far beyond what a consuming
review application can put in a model's context. The digest is the compact
projection designed for exactly that use: every variant, ask and changed
opening carries a ``ref`` citation and its ``precedent_ids``, which resolve
into the full playbook for on-demand drill-down (the digest itself never
contains ``full_text``).

Emitted by ``assemble_playbook`` as the top-level ``digest`` section — the
one place the digest lives (there is no standalone sidecar). The digest is a
pure function of the document, so it participates in ``identity.content_hash``
like any other content section.

Size discipline: the budget is ~40K tokens (chars/4 rule of thumb — this
codebase has no tokenizer dependency) and is ENFORCED by construction, not
aspirational: the three lists start capped at ``EXEMPLAR_TOP_N`` and the cap
tightens stepwise until the digest fits; ``n_variants_total`` /
``n_refused_total`` / ``n_changed_openings_total`` always report the uncapped
totals.
"""

from __future__ import annotations

from typing import Any

from playbook_engine.canonicalize import canonicalize
from playbook_engine.deal_dates import signed_on_key
from playbook_engine.equivalence import (
    LABEL_BY_OWNER,
    PART_MAP_LEGEND,
    SETTLED,
    addition_map,
    label_by,
    label_reason,
    part_map,
    settled_counts,
    settled_of,
    standard_parts,
)
from playbook_engine.equivalence import LABELS as EQUIVALENCE_LABELS
from playbook_engine.opf_accessors import perspective_party, playbook_clauses
from playbook_engine.precedent import clause_counts, normalize_variant_text, prior_form_counts

#: Schema version of the digest section itself — bump on any shape change so
#: consumers can dispatch (the digest is consumed outside this repo).
DIGEST_VERSION_V5 = "5"

#: Per-list cap (signed_variants, refused_asks, changed_openings, applied
#: after the equivalent variants collapse): the default/loosest cap,
#: tightened stepwise down to ``_MIN_TOP_N`` until the digest fits the
#: token budget. There is no material-risk exception — risk is a judged
#: verdict and never reaches the consumer path.
EXEMPLAR_TOP_N = 5

#: build_digest_v5 never tightens the per-list cap below this.
_MIN_TOP_N = 1

#: The hard size budget build_digest_v5 enforces (chars/4 rule of thumb).
DIGEST_TOKEN_BUDGET = 40_000


# ---------------------------------------------------------------------------
# digest_version 5 — projection of the OPF 0.6 precedent record (issues #223, #234)
# ---------------------------------------------------------------------------


def _signed_on_key(value: Any) -> tuple[int, int, int] | None:
    """Sortable key for a ``signed_on`` (``YYYY-MM-DD`` or ``YYYY-MM``); ``None`` if unknown.

    A month sorts before every day in it. Anything else (a malformed value, a
    date outside 1980-2099, ``None``) is ``None`` — unknown, sorts last.
    """
    return signed_on_key(value)


def _latest(values: list[Any]) -> str | None:
    """The latest signing month among *values* (``None`` when none is known)."""
    keyed = [k for v in values for k in (_signed_on_key(v),) if k is not None]
    return None if not keyed else f"{max(keyed)[0]:04d}-{max(keyed)[1]:02d}"


def _earliest(values: list[Any]) -> str | None:
    """The earliest signing month among *values* (``None`` when none is known)."""
    keyed = [k for v in values for k in (_signed_on_key(v),) if k is not None]
    return None if not keyed else f"{min(keyed)[0]:04d}-{min(keyed)[1]:02d}"


def _latest_first_key(record: dict[str, Any]) -> tuple[bool, tuple[int, ...], str, str]:
    """Sort key: latest ``signed_on`` first (unknown last), then document_id, then id.

    The id is the final tie-break, so the order is total even between two
    records of one deal (a deal has one record per clause type, so only a
    hand-built document has them).
    """
    k = _signed_on_key(record.get("signed_on"))
    return (
        k is None,
        tuple(-x for x in (k or (0, 0, 0))),
        str(record.get("document_id")),
        str(record.get("id")),
    )


def playbook_precedent_records(playbook: dict[str, Any]) -> list[dict[str, Any]]:
    """``evidence.precedent`` of the document (``[]`` when absent)."""
    evidence = playbook.get("evidence")
    records = evidence.get("precedent") if isinstance(evidence, dict) else None
    return [p for p in records if isinstance(p, dict)] if isinstance(records, list) else []


def _label_of(entry: Any) -> str | None:
    """The ``vs_standard.label`` of a ``{text, ref, vs_standard?}`` entry, or ``None``.

    ``None`` means "not judged" — an absent key, a ``null`` and a malformed
    value all read the same (OPF-SPEC §3.5.6).
    """
    vs = entry.get("vs_standard") if isinstance(entry, dict) else None
    label = vs.get("label") if isinstance(vs, dict) else None
    return label if label in VS_LABELS else None


def _judged(entry: Any) -> dict[str, Any] | None:
    """The ``vs_standard`` of *entry* when it is judged, else ``None``."""
    vs = entry.get("vs_standard") if isinstance(entry, dict) else None
    return vs if isinstance(vs, dict) and vs.get("label") in VS_LABELS else None


def part_map_fields(entry: Any) -> dict[str, Any]:
    """The compact part map of a judged text (issue #243): ``{part_map,
    additions, reason, settled[, label_by]}`` — one token per part of our
    standard, one token per addition
    (:data:`~playbook_engine.equivalence.PART_TOKENS`,
    :data:`~playbook_engine.equivalence.ADDITION_TOKENS`), the one-sentence
    reason — the owner's when the owner set the label, else the most important
    difference — and how settled the judgment is (issue #248:
    ``agreed``, ``adjudicated``, ``unchecked`` or ``owner``, derived from its
    basis and blind check); all ``None`` when the text is not judged. ``label_by: "owner"``
    is present only when the owner overrode the derived label
    (``owner_label``); absent, the label is derived from the parts (``parts``,
    the default the legend states), which saves a key on every other entry of
    a budgeted digest. The per-part reasons stay in the evidence and the
    dossiers."""
    vs = _judged(entry)
    if vs is None:
        return {"part_map": None, "additions": None, "reason": None, "settled": None}
    out: dict[str, Any] = {
        "part_map": part_map(vs),
        "additions": addition_map(vs),
        "reason": label_reason(vs),
        "settled": settled_of(vs),
    }
    if label_by(vs) == LABEL_BY_OWNER:
        out["label_by"] = LABEL_BY_OWNER
    return out


def clause_precedent_groups(
    taxonomy_id: Any, precedent: list[dict[str, Any]], *, party: str | None
) -> dict[str, list[dict[str, Any]]]:
    """Every signed variant, refused ask and changed opening of one clause, grouped and ranked.

    Uncapped and NOT collapsed — the digest collapses the ``equivalent``
    variants and caps these lists (:func:`_arrange_variants`), and the HTML
    renderer reads them whole. Grouping is
    :func:`precedent.normalize_variant_text` with *party*, the document's
    ``perspective.party`` (``opf_accessors.perspective_party``) — exact after
    normalization, no similarity tolerance.

    - ``signed_variants``: non-standard text signed in a signed deal. Each
      group: ``{text, n_deals, last_signed, n_from_standard, n_unchanged,
      label, ref, precedent_ids}`` — ``text`` is the sentence-boundary summary
      (≤ 300 chars, ``summarize_clause_text``) of the group's representative
      (latest ``signed_on``, then lowest ``document_id``), ``ref`` its
      citation and ``label`` the representative's ``vs_standard.label``
      (``None`` = not judged). ``n_from_standard`` counts the group's deals
      whose clause opened with our standard (a concession on record);
      ``n_unchanged`` those whose clause opened non-standard and was signed
      exactly as it opened (``opening_text`` null). Ordered ``n_deals`` desc,
      ``last_signed`` desc (unknown last), then normalized text.
    - ``refused_asks``: text proposed and struck before signing, across all
      deals. Each group: ``{text, n_deals, label, ref, precedent_ids}``, the
      representative being the earliest-round ask (then lowest
      ``document_id``). Ordered ``n_deals`` desc, then normalized text.
    - ``changed_openings``: non-standard opening language that was not signed
      as proposed (``precedent.changed_opening_members``). Each group:
      ``{text, n_deals, n_to_standard, n_struck, label, ref, precedent_ids}``
      — ``text`` the summary of the representative's (lowest ``document_id``)
      ``opening_text``, ``n_to_standard`` the deals whose ``standard`` is
      true, ``n_struck`` those whose ``signed_text`` is null. Ordered
      ``n_deals`` desc, then normalized text.

    ``n_deals`` counts distinct deals (``document_id``) — never rows.
    """
    grouped = clause_precedent_groups_with_members(taxonomy_id, precedent, party=party)
    return {name: [entry for entry, _ in rows] for name, rows in grouped.items()}


def clause_precedent_groups_with_members(
    taxonomy_id: Any, precedent: list[dict[str, Any]], *, party: str | None
) -> dict[str, list[tuple[dict[str, Any], list[Any]]]]:
    """:func:`clause_precedent_groups` with every group's members.

    Same groups in the same order; each row is ``(entry, members)``:
    ``members`` the group's precedent records for ``signed_variants`` and
    ``changed_openings``, and its ``(ask, record)`` pairs (each ask in the
    group with the record carrying it) for ``refused_asks``. The critic
    dossiers (:mod:`playbook_engine.dossiers`) choose their excerpt records
    from these.
    """
    from playbook_engine.observation_builder import summarize_clause_text  # noqa: PLC0415
    from playbook_engine.precedent import (  # noqa: PLC0415
        changed_opening_members,
        normalize_variant_text,
    )

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

    variant_out: list[tuple[Any, dict[str, Any], list[dict[str, Any]]]] = []
    for key, g in variants.items():
        members = g["members"]
        rep = sorted(members, key=_latest_first_key)[0]
        signed_ons = [m.get("signed_on") for m in members]
        last_signed = _latest(signed_ons)
        n = len({m.get("document_id") for m in members})
        entry = {
            "text": summarize_clause_text(rep["signed_text"]["text"]),
            "n_deals": n,
            "first_signed": _earliest(signed_ons),
            "last_signed": last_signed,
            "n_from_standard": len(
                {m.get("document_id") for m in members if m.get("opened_with") == "standard"}
            ),
            "n_unchanged": len(
                {
                    m.get("document_id")
                    for m in members
                    if m.get("opened_with") == "non_standard" and m.get("opening_text") is None
                }
            ),
            "label": _label_of(rep["signed_text"]),
            **part_map_fields(rep["signed_text"]),
            "ref": rep["signed_text"].get("ref"),
            "precedent_ids": sorted({str(m.get("id")) for m in members}),
        }
        last_key = _signed_on_key(last_signed)
        sort_key = (
            -n,
            last_key is None,
            tuple(-x for x in (last_key or (0, 0, 0))),
            key,
        )
        variant_out.append((sort_key, entry, members))
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

    refused_out: list[tuple[Any, dict[str, Any], list[Any]]] = []
    for key, g in refused.items():
        rep_ask, _rep_p = sorted(
            g["asks"],
            key=lambda ap: (
                ap[0].get("round") if isinstance(ap[0].get("round"), int) else 0,
                str(ap[1].get("document_id")),
                str(ap[1].get("id")),
            ),
        )[0]
        n = len(g["deals"])
        ask_dates = [p.get("signed_on") for _ask, p in g["asks"]]
        entry = {
            "text": summarize_clause_text(rep_ask["text"]),
            "n_deals": n,
            "first_signed": _earliest(ask_dates),
            "last_signed": _latest(ask_dates),
            "label": _label_of(rep_ask),
            **part_map_fields(rep_ask),
            "ref": rep_ask.get("ref"),
            "precedent_ids": sorted(g["ids"]),
        }
        refused_out.append(((-n, key), entry, g["asks"]))
    refused_out.sort(key=lambda t: t[0])

    changed_out: list[tuple[Any, dict[str, Any], list[dict[str, Any]]]] = []
    for key, members in changed_opening_members(records, party=party).items():
        rep = sorted(members, key=lambda m: (str(m.get("document_id")), str(m.get("id"))))[0]
        n = len({m.get("document_id") for m in members})
        entry = {
            "text": summarize_clause_text(rep["opening_text"]["text"]),
            "n_deals": n,
            "n_to_standard": len(
                {m.get("document_id") for m in members if m.get("standard") is True}
            ),
            "n_struck": len(
                {m.get("document_id") for m in members if m.get("signed_text") is None}
            ),
            "label": _label_of(rep["opening_text"]),
            **part_map_fields(rep["opening_text"]),
            "ref": rep["opening_text"].get("ref"),
            "precedent_ids": sorted({str(m.get("id")) for m in members}),
        }
        changed_out.append(((-n, key), entry, members))
    changed_out.sort(key=lambda t: t[0])

    return {
        "signed_variants": [(e, members) for _, e, members in variant_out],
        "refused_asks": [(e, pairs) for _, e, pairs in refused_out],
        "changed_openings": [(e, members) for _, e, members in changed_out],
    }


#: ``vs_standard.label`` vocabulary (OPF-SPEC §3.5.6): the single source is
#: ``equivalence.LABELS``, so a label added there is never read here as unjudged.
VS_LABELS = EQUIVALENCE_LABELS

#: How many exemplar texts the collapsed "equivalent to our standard" entry carries.
_EQUIVALENT_EXEMPLARS = 2

#: Display tier of a signed-variant entry (lower first): the variants worse
#: for us in some part lead (``less_protective`` and ``mixed``, issue #243),
#: then the unrankable ones, then those nobody has judged, then the more
#: protective ones; the collapsed equivalent entry closes the list.
_VARIANT_TIERS = {
    "less_protective": 0,
    "mixed": 0,
    "different_concept": 1,
    None: 2,
    "more_protective": 3,
    "equivalent": 4,
}


def arrange_variant_slots(variants: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """The digest's signed-variant order as *slots* of underlying group entries.

    *variants* are :func:`clause_precedent_groups`' ``signed_variants`` (one
    per grouping key). Every group labelled ``equivalent`` shares ONE slot
    (in group order), closing the list; every other group is a slot of its
    own, ordered ``less_protective``/``mixed`` first, then
    ``different_concept``, then unjudged, then ``more_protective`` (each tier
    in group order).
    :func:`_arrange_variants` renders the slots; the critic dossiers read
    them to pick excerpts in exactly the digest's order.
    """
    equivalent = [v for v in variants if v.get("label") == "equivalent"]
    rest = [v for v in variants if v.get("label") != "equivalent"]
    ordered = sorted(
        enumerate(rest), key=lambda iv: (_VARIANT_TIERS.get(iv[1].get("label"), 1), iv[0])
    )
    slots = [[v] for _, v in ordered]
    if equivalent:
        slots.append(equivalent)
    return slots


def _arrange_variants(variants: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Order the signed variants by label tier and collapse the equivalent ones.

    The slots of :func:`arrange_variant_slots`: the equivalent slot becomes
    ONE entry, ``{label: "equivalent", n_deals, n_texts, n_from_standard,
    n_unchanged, settled_counts, last_signed, exemplars, precedent_ids}``:
    ``n_deals`` the distinct deals across the groups, ``n_texts`` how many
    distinct texts collapsed into it, ``settled_counts`` how many of those
    texts are in each ``settled`` state (issue #248, they sum to ``n_texts``), ``exemplars`` the first :data:`_EQUIVALENT_EXEMPLARS`
    groups' ``{text, ref}`` in group order. The digest cap applies to the
    result.
    """
    out: list[dict[str, Any]] = []
    for slot in arrange_variant_slots(variants):
        if slot[0].get("label") != "equivalent":
            out.append(slot[0])
            continue
        equivalent = slot
        ids = sorted({pid for v in equivalent for pid in v["precedent_ids"]})
        out.append(
            {
                "label": "equivalent",
                "n_deals": sum(v["n_deals"] for v in equivalent),
                "n_texts": len(equivalent),
                "n_from_standard": sum(v["n_from_standard"] for v in equivalent),
                "n_unchanged": sum(v["n_unchanged"] for v in equivalent),
                "settled_counts": settled_counts(v.get("settled") for v in equivalent),
                "first_signed": _earliest([v.get("first_signed") for v in equivalent]),
                "last_signed": _latest([v.get("last_signed") for v in equivalent]),
                "exemplars": [
                    {"text": v["text"], "ref": v["ref"]} for v in equivalent[:_EQUIVALENT_EXEMPLARS]
                ],
                "precedent_ids": ids,
            }
        )
    return out


def _positions(clause_records: list[dict[str, Any]]) -> dict[str, int]:
    """Distinct signed deals with a signed text, by position.

    ``standard`` (the signed text is our standard language), each
    ``vs_standard`` label of a non-standard signed text, and ``unjudged``
    (a non-standard signed text with no label). Their sum is the number of
    signed deals that signed anything for the clause; ``standard`` equals
    ``n_signed_standard``.
    """
    deals: dict[str, set[Any]] = {"standard": set(), "unjudged": set()}
    for label in VS_LABELS:
        deals[label] = set()
    for p in clause_records:
        signed_text = p.get("signed_text")
        if p.get("signed") is not True or not signed_text or not isinstance(signed_text, dict):
            continue
        bucket = "standard" if p.get("standard") is True else _label_of(signed_text) or "unjudged"
        deals[bucket].add(p.get("document_id"))
    return {
        "standard": len(deals["standard"]),
        **{label: len(deals[label]) for label in VS_LABELS},
        "unjudged": len(deals["unjudged"]),
    }


def uncovered_clause_types(playbook: dict[str, Any]) -> list[dict[str, str]]:
    """``[{taxonomy_id, label}]`` for every classifier-eligible taxonomy entry with no evidence.

    Eligible means ``status`` is ``active`` or ``custom`` (the same rule as
    ``TaxonomyEntry.is_classifier_eligible``); "no evidence" means no
    ``evidence.clauses[]`` entry for the ``taxonomy_id``. Sorted by
    ``taxonomy_id``, never capped. Its meaning is "a recognised clause type
    with no precedent in this corpus", nothing more.
    """
    covered = {c.get("taxonomy_id") for c in playbook_clauses(playbook)}
    taxonomy = playbook.get("taxonomy")
    entries = taxonomy.get("entries") if isinstance(taxonomy, dict) else None
    out: dict[str, str] = {}
    for entry in entries if isinstance(entries, list) else []:
        if not isinstance(entry, dict) or entry.get("status") not in ("active", "custom"):
            continue
        tid = entry.get("id")
        if not isinstance(tid, str) or tid in covered:
            continue
        label = entry.get("label")
        out[tid] = label if isinstance(label, str) else tid
    return [{"taxonomy_id": tid, "label": out[tid]} for tid in sorted(out)]


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
    signed_ons = [p.get("signed_on") for p in precedent if p.get("signed") is True]
    return {
        "n_deals": len(docs),
        "n_signed": n_signed,
        "first_signed": _earliest(signed_ons),
        "last_signed": _latest(signed_ons),
    }


def digest_standard(clause: dict[str, Any]) -> dict[str, Any] | None:
    """Our standard as the digest carries it (issue #243): ``{parts: [{part_id,
    text}], source_ref}`` — the parts the part maps index, in order, without
    repeating their joined ``text`` (``null`` when the clause has none)."""
    std = clause.get("our_standard")
    if not isinstance(std, dict):
        return None
    return {
        "parts": [
            {"part_id": p.get("part_id"), "text": p.get("text")} for p in standard_parts(clause)
        ],
        "source_ref": std.get("source_ref"),
    }


#: ``in_use_basis`` of a standard-history form: where its in-use dates came from.
IN_USE_CONFIG = "config"
IN_USE_DEAL_DATES = "deal_dates"
IN_USE_MIXED = "mixed"


def _in_use_dates(
    prior_template: dict[str, Any] | None, opened: list[dict[str, Any]]
) -> dict[str, Any]:
    """The in-use dates of one former form, and where they came from.

    The dates belong to the FORM, not to a clause: *opened* is every signed
    deal's precedent record that opened on the form in ANY clause type, so the
    form carries the same dates in every clause's ``standard_history``.

    Each end is the config's own value (``baseline.prior_templates[]``, at the
    precision the owner gave) when it was given, else inferred from the deals
    that opened on the form: the earliest signing month for ``in_use_from``,
    the latest for ``in_use_until`` (a month, ``YYYY-MM``). ``in_use_basis`` is
    ``config`` when every date given is the config's, ``deal_dates`` when every
    one is inferred, ``mixed`` when both kinds appear, and ``null`` when
    there is no date at all — never guessed.
    """
    given = prior_template or {}
    signed_ons = [p.get("signed_on") for p in opened]
    from_cfg, until_cfg = given.get("in_use_from"), given.get("in_use_until")
    first = from_cfg if from_cfg is not None else _earliest(signed_ons)
    last = until_cfg if until_cfg is not None else _latest(signed_ons)
    kinds = {
        kind
        for value, cfg in ((first, from_cfg), (last, until_cfg))
        for kind in ((IN_USE_CONFIG if cfg is not None else IN_USE_DEAL_DATES),)
        if value is not None
    }
    basis = None if not kinds else next(iter(kinds)) if len(kinds) == 1 else IN_USE_MIXED
    return {"in_use_from": first, "in_use_until": last, "in_use_basis": basis}


def standard_history(
    clause: dict[str, Any],
    records: list[dict[str, Any]],
    prior_templates: dict[str, dict[str, Any]],
    *,
    party: str | None,
    all_records: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """The clause's ``standard_history`` (OPF 0.6 / digest 5, issue #249).

    ``{forms, why_changed}``: ``forms`` has one entry per former form of ours
    with text for the clause, in config order; ``why_changed`` is the owner's
    note (``{text, basis: "owner"}``) or ``null`` — the only reason a change is
    ever given. Each form carries:

    - ``form`` (its label), ``in_use_from``/``in_use_until``/``in_use_basis``
      (:func:`_in_use_dates`), inferred once per form from *all_records* (every
      clause type's records; default *records*) so a form's dates are the same
      in every clause;
    - ``n_opened``, ``n_kept``, ``n_conceded``, ``n_moved_to_current``: distinct
      signed deals that opened on the form, kept what we sent, conceded it
      (``n_opened - n_kept``), and ended on today's standard
      (:func:`~playbook_engine.precedent.prior_form_counts`);
    - ``unchanged``: the form's clause reads, under the §3.5.4 grouping key,
      exactly as today's standard — nothing to compare, so no part map;
    - ``label`` / ``part_map`` / ``additions`` / ``reason`` / ``settled``
      (``label_by`` when the owner set the label): the form's clause judged
      against today's standard, part by part, like any other text
      (:func:`part_map_fields`); null when not judged;
    - ``parts``: the form's own parts, in order, as ``{part_id, text}`` — a
      part whose text is one of today's standard (same ``part_id``) carries
      ``text: null`` to keep the digest compact;
    - ``ref``: where the clause sits in the form.
    """
    current_parts = {p.get("part_id") for p in standard_parts(clause)}
    standard = clause.get("our_standard")
    standard_key = (
        normalize_variant_text(standard["text"], party=party)
        if isinstance(standard, dict) and isinstance(standard.get("text"), str)
        else None
    )
    forms: list[dict[str, Any]] = []
    for entry in clause.get("prior_standards") or []:
        if not isinstance(entry, dict):
            continue
        label = entry.get("form_label")
        opened = [
            p
            for p in (records if all_records is None else all_records)
            if p.get("signed") is True
            and p.get("opened_with") == "prior_standard"
            and p.get("opened_with_form") == label
        ]
        counts = prior_form_counts(records, label)
        text = entry.get("text")
        forms.append(
            {
                "form": label,
                **_in_use_dates(prior_templates.get(str(label)), opened),
                "n_opened": counts["n_opened"],
                "n_kept": counts["n_kept"],
                "n_conceded": counts["n_opened"] - counts["n_kept"],
                "n_moved_to_current": counts["n_moved_to_current"],
                "unchanged": bool(
                    standard_key
                    and isinstance(text, str)
                    and normalize_variant_text(text, party=party) == standard_key
                ),
                "label": _label_of(entry),
                **part_map_fields(entry),
                "parts": [
                    {
                        "part_id": part.get("part_id"),
                        "text": None if part.get("part_id") in current_parts else part.get("text"),
                    }
                    for part in entry.get("parts") or []
                    if isinstance(part, dict)
                ],
                "ref": entry.get("source_ref"),
            }
        )
    note = clause.get("why_changed")
    return {"forms": forms, "why_changed": note if isinstance(note, dict) else None}


def _build_digest_at(playbook: dict[str, Any], top_n: int | None) -> dict[str, Any]:
    """Digest 4 with a fixed per-list cap of *top_n* (``None`` = uncapped)."""
    precedent = playbook_precedent_records(playbook)
    party = perspective_party(playbook)
    baseline = playbook.get("baseline")
    prior_templates = {
        str(t.get("label")): t
        for t in (baseline.get("prior_templates") if isinstance(baseline, dict) else None) or []
        if isinstance(t, dict)
    }
    clauses: list[dict[str, Any]] = []
    for clause in playbook_clauses(playbook):
        tid = clause.get("taxonomy_id")
        groups = clause_precedent_groups(tid, precedent, party=party)
        variants = groups["signed_variants"]
        refused = groups["refused_asks"]
        changed = groups["changed_openings"]
        listed = _arrange_variants(variants)
        counts = clause_counts(str(tid), precedent, party=party)
        clauses.append(
            {
                "id": clause.get("id"),
                "taxonomy_id": tid,
                "title": clause.get("title"),
                "our_standard": digest_standard(clause),
                "n_deals": clause.get("n_deals"),
                "first_signed": _earliest(
                    [p.get("signed_on") for p in precedent if p.get("taxonomy_id") == tid]
                ),
                "last_signed": _latest(
                    [p.get("signed_on") for p in precedent if p.get("taxonomy_id") == tid]
                ),
                "n_signed_standard": clause.get("n_signed_standard"),
                "n_opened_standard": counts["n_opened_standard"],
                "n_kept_standard": counts["n_kept_standard"],
                "standard_history": standard_history(
                    clause,
                    [p for p in precedent if p.get("taxonomy_id") == tid],
                    prior_templates,
                    party=party,
                    all_records=precedent,
                ),
                "positions": _positions([p for p in precedent if p.get("taxonomy_id") == tid]),
                "signed_variants": listed if top_n is None else listed[:top_n],
                "refused_asks": refused if top_n is None else refused[:top_n],
                "changed_openings": changed if top_n is None else changed[:top_n],
                "n_variants_total": len(variants),
                "n_refused_total": len(refused),
                "n_changed_openings_total": len(changed),
            }
        )
    agreement_type = playbook.get("agreement_type")
    perspective = playbook.get("perspective")
    return {
        "digest_version": DIGEST_VERSION_V5,
        "perspective": perspective if isinstance(perspective, dict) else None,
        "agreement_type": (
            {k: agreement_type[k] for k in ("id", "name") if k in agreement_type}
            if isinstance(agreement_type, dict)
            else None
        ),
        "corpus": _corpus_summary(playbook, precedent),
        "part_map_legend": PART_MAP_LEGEND,
        "clauses": clauses,
        "uncovered_clause_types": uncovered_clause_types(playbook),
    }


def build_digest_v5(
    playbook: dict[str, Any], *, token_budget: int | None = DIGEST_TOKEN_BUDGET
) -> dict[str, Any]:
    """Build the digest_version 5 digest of an OPF 0.6 document (issues #223, #234, #240).

    Per clause, our standard, how many deals signed it, how many opened with
    it (``n_opened_standard``) and kept it (``n_kept_standard``), the
    ``positions`` of the signed texts, the non-standard signed variants (each
    with ``n_from_standard`` and ``n_unchanged``; those labelled
    ``equivalent`` collapsed into one entry), the refused asks and the
    ``changed_openings``, each grouped by normalized text with its
    distinct-deal count, citation and the precedent ids behind it; at the top
    level, ``uncovered_clause_types``. No stance, band, risk or deviation
    field; no ``full_text`` (variant, ask and opening ``text`` is the
    ≤ 300-char sentence-boundary summary). ``perspective`` is always present
    (``null`` when the document has none) so a consumer never has to read it
    from elsewhere.

    The token budget is enforced by construction: the three lists start
    capped at ``EXEMPLAR_TOP_N`` (after the equivalent variants collapse) and
    the cap tightens stepwise to ``_MIN_TOP_N`` until the digest fits.
    ``n_variants_total``/``n_refused_total``/``n_changed_openings_total``
    always report the uncapped totals, so a consumer knows what the cap left
    out. Pass ``token_budget=None`` for the loosest cap unconditionally.
    """
    digest = _build_digest_at(playbook, EXEMPLAR_TOP_N)
    if token_budget is None:
        return digest
    for top_n in range(EXEMPLAR_TOP_N - 1, _MIN_TOP_N - 1, -1):
        if digest_token_estimate(digest) <= token_budget:
            break
        digest = _build_digest_at(playbook, top_n)
    return digest


def build_digest_uncapped(playbook: dict[str, Any]) -> dict[str, Any]:
    """The digest with no cap on any list — what the capped digest is cut from.

    Not an OPF section (a document never carries it); counts-only diagnostics
    (``playbook scorecard``) read it to total entries the cap may have dropped.
    """
    return _build_digest_at(playbook, None)


def digest_settled_counts(digest: dict[str, Any]) -> dict[str, dict[str, int]]:
    """Counts-only: judged texts of *digest* by ``settled`` outcome (issue #248).

    ``{signed_variants, refused_asks, changed_openings, total}``, each
    ``{agreed, adjudicated, unchecked, owner}`` over the digest's entries: an
    entry counts once under its own ``settled``, a collapsed equivalent entry
    once per text under its ``settled_counts``, and an unjudged entry
    (``settled`` null) not at all. Pass the uncapped digest
    (:func:`build_digest_uncapped`) to count what the cap may have dropped
    too. Over a whole playbook, each list's counts equal the scorecard's
    ``equivalence`` counts of the same role (``settled``) for the texts the
    list holds.
    """
    out: dict[str, dict[str, int]] = {
        name: dict.fromkeys(SETTLED, 0)
        for name in ("signed_variants", "refused_asks", "changed_openings")
    }
    for clause in digest.get("clauses") or []:
        if not isinstance(clause, dict):
            continue
        for name, counts in out.items():
            for entry in clause.get(name) or []:
                if not isinstance(entry, dict):
                    continue
                collapsed = entry.get("settled_counts")
                if isinstance(collapsed, dict):
                    for state in SETTLED:
                        counts[state] += int(collapsed.get(state) or 0)
                elif entry.get("settled") in counts:
                    counts[entry["settled"]] += 1
    out["total"] = {state: sum(out[name][state] for name in tuple(out)) for state in SETTLED}
    return out


def digest_token_estimate(digest: dict[str, Any]) -> int:
    """Rough token estimate — canonical chars / 4, the repo-wide rule of thumb."""
    return len(canonicalize(digest)) // 4
