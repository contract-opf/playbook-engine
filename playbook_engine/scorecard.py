"""Counts-only scorecard of a derivation out-dir — ``playbook scorecard`` (issue #237).

A maintainer runs a private evaluation corpus through the engine and needs to
say, in public, what a change did to it: how many documents were mined, how
much of the corpus classified, how many precedent records, openings, signed
variants and refused asks came out, how big the digest is, and how much work
is still queued for the store-backed agent. The scorecard answers that with
**integers, ratios and closed-vocabulary labels only**, so it can be pasted
into an issue or PR without leaking the corpus.

The leak guard is structural, not a promise:

- Nothing in the scorecard is keyed by a document, party, clause type or
  file. Per-deal and per-clause figures are aggregated (mean / median / min /
  max, or an anonymous sorted list of ratios) before they are written.
- Every label that comes from the out-dir — classification bases, dropped-
  observation reasons, pending-queue kinds, paper sides, ``opened_with``
  values, OPF/digest versions — is checked against a closed vocabulary
  (:data:`ENUM_LABELS`) and written as ``"other"`` when it is not in it. A
  corrupted or hostile out-dir cannot smuggle a string through a key.
- Every number read from the out-dir and passed through or summed is
  accepted only as a count — a non-bool ``int`` — and is ``null`` (or left
  out of the sum) otherwise, so a string, list or dict planted where a
  count belongs cannot pass through either.
- ``tests/test_cli_scorecard.py`` asserts that every key in the output is a
  field name or enum label and every string value is an enum label.

Inputs (all optional — a field whose source is absent is ``null``, never an
error): ``playbook.opf.json``, ``observations.jsonl``,
``template_observations.jsonl``, ``corpus_manifest.json``,
``quarantine.json``, ``judge/{pending,verdicts}.jsonl`` and
``segment/{pending,cache}.jsonl``.

Paper side appears only under ``classification.by_paper`` and
``template_drift`` — diagnostics of the corpus, never inputs to the artifact.
A field whose section the playbook does not carry (the critic ``dossiers``,
issue #228) is ``null``.
"""

from __future__ import annotations

import json
import statistics
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from playbook_engine.agent_judge import VerdictStore
from playbook_engine.agent_segmenter import AGENT_SEGMENTER_MODEL
from playbook_engine.canonicalize import canonicalize
from playbook_engine.deal_dates import BASES as SIGNED_ON_BASES
from playbook_engine.deal_dates import DEAL_DATE_KIND, coverage
from playbook_engine.dossiers import DOSSIER_MIN_BUDGET, dossier_budget, dossier_standard_text
from playbook_engine.equivalence import (
    ADDITION_TOKENS,
    PART_TOKENS,
    ROLE_OPENING,
    ROLE_SIGNED,
    SETTLED,
    is_eligible,
    iter_slots,
    settled_of,
    slot_key,
)
from playbook_engine.equivalence import LABELS as EQUIVALENCE_LABELS
from playbook_engine.equivalence import ROLES as EQUIVALENCE_ROLES
from playbook_engine.equivalence import summarize as summarize_equivalence
from playbook_engine.equivalence_check import CHECKER_MODEL
from playbook_engine.llm_segmenter_batch import SegmentationVerdictCache
from playbook_engine.observation_builder import (
    DROPPED_ORIGIN_UNDETERMINED,
    DROPPED_PRIOR_STANDARD_REMOVED,
    DROPPED_REFUSED_UNSIGNED,
    DROPPED_STANDARD_REMOVED_UNSIGNED,
    DROPPED_SURVIVES_IN_TERMINAL,
    OUTCOME_OPENING,
)
from playbook_engine.opf_accessors import perspective_party, playbook_clauses
from playbook_engine.precedent import normalize_variant_text
from playbook_engine.resegmentation import (
    CHECK_KIND as RESEGMENT_CHECK_KIND,
)
from playbook_engine.resegmentation import (
    NON_CLAUSE_REASONS,
    check_subjects,
)
from playbook_engine.rubric import rubric_version

__all__ = [
    "SCORECARD_FILENAME",
    "SCORECARD_VERSION",
    "ENUM_LABELS",
    "build_scorecard",
    "compare_scorecards",
    "flatten_scorecard",
    "pending_queue_counts",
    "render_table",
    "write_scorecard",
]

#: Shape version of ``scorecard.json`` — bump on any field change so two
#: scorecards of different shapes are never compared silently.
#:
#: v2 (issue #237): ``precedent.opened_with`` (every record by
#: ``opened_with``) became ``precedent.openings_by_opened_with`` (openings
#: only — records with a non-null ``opening_text``) plus
#: ``precedent.records_by_opened_with`` (every record).
#:
#: v3 (issue #233): ``opening_drift`` (per clause with ``our_standard``, the
#: share of our-paper signed deals opening with our standard) and
#: ``template_drift.over_one_third_below_half``; ``opening`` rows are no
#: longer counted as classification observations.
#:
#: v4 (issue #240): the ``equivalence`` section (``vs_standard`` verdicts
#: drafted / checked / agreed / adjudicated / unchecked by role, the agreement
#: rate and the checker model ids seen in the store).
#:
#: v5 (issue #234): the ``digest`` section gains ``n_opened_standard``,
#: ``n_kept_standard``, ``signed_variants_n_from_standard``,
#: ``signed_variants_n_unchanged``, ``changed_openings`` and
#: ``uncovered_clause_types`` (digest 4); a list counts as capped against the
#: changed openings too.
#:
#: v6 (issue #228): the ``dossiers`` section gains ``cut_texts``,
#: ``excerpts_dropped``, ``over_budget_single_excerpt`` and ``max_budget``.
#:
#: v7 (issue #243): ``equivalence`` judges part by part — ``by_label`` gains
#: ``mixed``; the section gains ``label_agreed`` / ``parts_agreed`` and their
#: rates (agreement on the label and on the parts, reported separately),
#: ``part_tokens`` / ``addition_tokens`` (part-map token counts) and
#: ``standard_parts`` (parts per standard).
#:
#: v8 (issue #247): the ``signed_on`` section (in-scope signed deals dated, by
#: basis, and how many have no date).
#:
#: v9 (issue #248): ``equivalence`` rows gain ``settled`` (the judged texts by
#: outcome: agreed, adjudicated, unchecked, owner) and the ``digest`` section
#: gains ``settled`` (the same outcomes counted over the uncapped digest, per
#: list and in total), ``settled_not_listed`` (per list, the judged texts of
#: that role the digest leaves out for a named reason, counted directly) and
#: ``settled_unreconciled`` (what is left after that; must be 0).
#:
#: v10 (issue #255): ``classification`` gains ``clauses`` (observations that
#: are clauses), ``non_clause`` and ``non_clause_by_reason`` (the ones marked
#: ``x_non_clause``: a bare section number, a signature block, ...), and
#: ``classified_share`` / ``by_basis`` now range over the clauses only; the
#: ``resegmented`` classification basis appears; and a ``resegmentation``
#: section counts the splits and joins, the observations they produced and the
#: agent check of a sample of them.
#:
#: v11 (issue #249): ``precedent`` and the ``digest`` section count the clauses
#: that opened on one of our former forms (``opened_with`` gains
#: ``prior_standard``); ``prior_standards`` totals the former forms, their
#: clause entries, the opened / kept / moved-to-current deals and the entries
#: by ``settled`` outcome; ``equivalence.by_role`` gains ``prior_standard``;
#: ``dropped_observations`` gains ``removed_prior_standard``.
SCORECARD_VERSION = 11

#: Default file name, written into the out-dir.
SCORECARD_FILENAME = "scorecard.json"

#: Any label outside its closed vocabulary is written as this.
OTHER = "other"

#: Classification bases (``clause_classifier._BASIS_VALUES``) plus the three
#: the scorecard itself introduces: ``aligned`` (the observation's
#: taxonomy_id came from its aligned row, ``pipeline``), ``unrecorded``
#: (an observation from a store that predates ``x_classification_basis``) and
#: ``non_clause`` (the run-manifest coverage line's label for an unclassified
#: observation whose node is no clause, issue #255).
CLASSIFICATION_BASES = frozenset(
    {
        "exact_match",
        "heading_similarity",
        "judge",
        "judge_error",
        "needs_review",
        "unclassified",
        "llm_segmenter",
        "inherited",
        "content_similarity",
        "resegmented",
        "aligned",
        "unrecorded",
        "non_clause",
    }
)

DROPPED_REASONS = frozenset(
    {
        DROPPED_SURVIVES_IN_TERMINAL,
        DROPPED_ORIGIN_UNDETERMINED,
        DROPPED_STANDARD_REMOVED_UNSIGNED,
        DROPPED_REFUSED_UNSIGNED,
        DROPPED_PRIOR_STANDARD_REMOVED,
    }
)

#: ``PendingQueue`` record kinds (agent_judge / agent_segmenter). A kind added
#: through ``agent_judge.register_verdict_kind`` must be added here too, or its
#: pending items are counted under ``other``.
PENDING_KINDS = frozenset(
    {"classify", "provenance", "scope", "segment", "equivalence", DEAL_DATE_KIND}
    | {RESEGMENT_CHECK_KIND}
)

#: Checker model ids the scorecard names (the pinned one); any other recorded
#: id is written as ``other`` — a model id from the store is still data from
#: the out-dir.
CHECKER_MODELS = frozenset({CHECKER_MODEL})

#: ``vs_standard`` labels, the three roles a judged text can play, and the
#: part-map tokens (issue #243).
EQUIVALENCE_VOCABULARY = (
    frozenset(EQUIVALENCE_LABELS)
    | frozenset(EQUIVALENCE_ROLES)
    | frozenset(PART_TOKENS.values())
    | frozenset(ADDITION_TOKENS.values())
)

#: Observation ``provenance`` values — the paper-side diagnostic split.
PAPER_SIDES = frozenset({"our_paper", "counterparty_paper", "unknown"})

#: ``opened_with`` (OPF 0.6, issue #233); a null value is ``undetermined``.
OPENED_WITH = frozenset({"standard", "prior_standard", "non_standard", "absent", "undetermined"})

#: The engine's one format; any other version a playbook claims is labelled
#: ``other``.
OPF_VERSIONS = frozenset({"0.6"})
DIGEST_VERSIONS = frozenset({"5"})

#: Every string the scorecard may carry as a value or a data-derived key.
ENUM_LABELS = frozenset(
    {OTHER}
    | CLASSIFICATION_BASES
    | DROPPED_REASONS
    | PENDING_KINDS
    | PAPER_SIDES
    | OPENED_WITH
    | frozenset(SIGNED_ON_BASES)
    | OPF_VERSIONS
    | DIGEST_VERSIONS
    | CHECKER_MODELS
    | EQUIVALENCE_VOCABULARY
    | frozenset(NON_CLAUSE_REASONS)
)


# ---------------------------------------------------------------------------
# Readers — every one returns None when its source is absent or unreadable.
# ---------------------------------------------------------------------------


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _read_jsonl(path: Path) -> list[dict[str, Any]] | None:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    records: list[dict[str, Any]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if isinstance(record, dict):
            records.append(record)
    return records


def _label(value: Any, vocabulary: frozenset[str]) -> str:
    """*value* if it is a label of *vocabulary*, else ``"other"``."""
    return value if isinstance(value, str) and value in vocabulary else OTHER


def _count(value: Any) -> int | None:
    """*value* if it is a count (a non-bool ``int``), else ``None``.

    The numeric half of the leak guard: every number the scorecard copies or
    sums from the out-dir goes through here, so a string, list or dict
    planted where a count belongs is ``null``, never written out.
    """
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _sum_counts(values: Iterable[Any]) -> int:
    """Sum of the counts among *values*; anything else is skipped."""
    return sum(n for n in map(_count, values) if n is not None)


def _ratio(num: int | float, den: int | float) -> float | None:
    return round(num / den, 4) if den else None


def _stats(values: list[int] | list[float]) -> dict[str, Any]:
    """Anonymous distribution summary — never which item had which value."""
    if not values:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "n": len(values),
        "mean": round(statistics.fmean(values), 4),
        "median": round(float(statistics.median(values)), 4),
        "min": round(float(min(values)), 4),
        "max": round(float(max(values)), 4),
    }


def _sorted_counts(counter: Counter[str]) -> dict[str, int]:
    return {k: counter[k] for k in sorted(counter)}


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


def _corpus_section(
    manifest: Any, quarantine: Any, playbook: dict[str, Any] | None
) -> dict[str, Any]:
    docs = [d for d in manifest if isinstance(d, dict)] if isinstance(manifest, list) else None
    stats: dict[str, Any] = {}
    if playbook is not None:
        corpus = playbook.get("corpus")
        raw = corpus.get("stats") if isinstance(corpus, dict) else None
        stats = raw if isinstance(raw, dict) else {}
    if docs is not None:
        documents: int | None = len(docs)
        versions: int | None = _sum_counts(d.get("versions") for d in docs)
        signed: int | None = sum(1 for d in docs if d.get("signed_version") is not None)
        in_scope: int | None = sum(1 for d in docs if d.get("in_scope") is True)
    else:
        # No manifest: fall back to the playbook's own corpus.stats. These
        # are copied straight through, so each must be a count or is null.
        documents = _count(stats.get("documents_total"))
        versions = _count(stats.get("versions_total"))
        in_scope = _count(stats.get("documents_in_scope"))
        signed = None
        if playbook is not None:
            corpus = playbook.get("corpus")
            pdocs = corpus.get("documents") if isinstance(corpus, dict) else None
            if isinstance(pdocs, list):
                signed = sum(
                    1 for d in pdocs if isinstance(d, dict) and d.get("signed_version") is not None
                )
    return {
        "documents": documents,
        "versions": versions,
        "signed": signed,
        "in_scope": in_scope,
        "quarantined": len(quarantine) if isinstance(quarantine, list) else None,
    }


def _template_section(
    template_obs: list[dict[str, Any]] | None, playbook: dict[str, Any] | None
) -> dict[str, Any]:
    standards = (
        sum(1 for o in template_obs if o.get("taxonomy_id")) if template_obs is not None else None
    )
    has_template: bool | None = None
    clauses: int | None = None
    with_standard: int | None = None
    if playbook is not None:
        baseline = playbook.get("baseline")
        if isinstance(baseline, dict) and isinstance(baseline.get("has_canonical_template"), bool):
            has_template = baseline["has_canonical_template"]
        clause_list = playbook_clauses(playbook)
        clauses = len(clause_list)
        with_standard = sum(
            1
            for c in clause_list
            if isinstance(c, dict) and isinstance(c.get("our_standard"), dict)
        )
    return {
        "has_canonical_template": has_template,
        "standards_classified": standards,
        "clauses": clauses,
        "clauses_with_our_standard": with_standard,
        "our_standard_coverage": (
            _ratio(with_standard, clauses)
            if clauses is not None and with_standard is not None
            else None
        ),
    }


def _classification_block(observations: list[dict[str, Any]]) -> dict[str, Any]:
    """Classified share, distinct types per deal, and (when recorded) bases.

    ``opening`` rows (issue #233) restate a first-draft clause that is already
    counted through its terminal observation, so they are not classification
    observations: leaving them in would inflate the counts and break the
    comparison against a baseline taken before openings were recorded.
    """
    observations = [o for o in observations if o.get("outcome") != OUTCOME_OPENING]
    # Issue #255: an observation whose node is not a clause (a bare section
    # number, a signature block, ...) is no classification observation either.
    non_clause = [o for o in observations if o.get("x_non_clause") and not o.get("taxonomy_id")]
    non_clause_ids = {id(o) for o in non_clause}
    clauses = [o for o in observations if id(o) not in non_clause_ids]
    classified = [o for o in clauses if o.get("taxonomy_id")]
    types_by_deal: dict[str, set[str]] = {}
    for o in observations:
        citation = o.get("citation")
        doc_id = citation.get("document_id") if isinstance(citation, dict) else None
        if not isinstance(doc_id, str):
            continue
        types = types_by_deal.setdefault(doc_id, set())
        tid = o.get("taxonomy_id")
        if isinstance(tid, str) and tid:
            types.add(tid)
    by_basis: dict[str, int] | None = None
    if any("x_classification_basis" in o for o in observations):
        counter: Counter[str] = Counter(
            _label(o.get("x_classification_basis", "unrecorded"), CLASSIFICATION_BASES)
            for o in clauses
        )
        by_basis = _sorted_counts(counter)
    reasons = frozenset(NON_CLAUSE_REASONS)
    return {
        "observations": len(observations),
        "clauses": len(clauses),
        "non_clause": len(non_clause),
        "non_clause_by_reason": _sorted_counts(
            Counter(_label(o.get("x_non_clause"), reasons) for o in non_clause)
        ),
        "classified": len(classified),
        "classified_share": _ratio(len(classified), len(clauses)),
        "by_basis": by_basis,
        "deals": len(types_by_deal),
        "distinct_types_per_deal": _stats([len(t) for t in types_by_deal.values()]),
    }


def _classification_section(observations: list[dict[str, Any]] | None) -> dict[str, Any] | None:
    if observations is None:
        return None
    section = _classification_block(observations)
    # Paper-side parity diagnostic only: one deal's observations share one
    # provenance, so a split by observation provenance is a split by deal.
    by_paper: dict[str, list[dict[str, Any]]] = {}
    for o in observations:
        by_paper.setdefault(_label(o.get("provenance"), PAPER_SIDES), []).append(o)
    section["by_paper"] = {side: _classification_block(by_paper[side]) for side in sorted(by_paper)}
    return section


def _resegmentation_section(
    manifest: Any, observations: list[dict[str, Any]] | None, out_dir: Path
) -> dict[str, Any] | None:
    """Counts of what the re-segmentation pass did and how a sample of it checked out (issue #255).

    ``splits`` / ``joins`` are the nodes the pass split out and merged over all
    versions of all deals (from the corpus manifest); ``observations_split`` /
    ``observations_joined`` the precedent-store observations built from such a
    node. ``check_sample`` is the size of the ``resegment_check`` sample (the
    recorded splits and joins, capped), ``checked`` how many of those have a
    stored verdict under the current rubric and ``wrong`` how many of those
    say the pass was wrong. Counts only.
    """
    if manifest is None and observations is None:
        return None
    docs = [d for d in manifest if isinstance(d, dict)] if isinstance(manifest, list) else []
    recorded = [d["resegmentation"] for d in docs if isinstance(d.get("resegmentation"), dict)]
    split_obs = joined_obs = 0
    for o in observations or []:
        meta = o.get("x_resegmented")
        if not isinstance(meta, dict):
            continue
        if isinstance(meta.get("to"), dict):
            split_obs += 1  # a node re-homed to another type, not the node it left
        elif "joined_with" in meta:
            joined_obs += 1
    subjects = check_subjects(out_dir)
    store = VerdictStore(out_dir / "judge" / "verdicts.jsonl")
    current = rubric_version(RESEGMENT_CHECK_KIND)
    checked = wrong = 0
    for subject in subjects:
        stored = store.get_record_by_key(subject.key)
        if stored is None or stored.rubric_version != current:
            continue
        ok = stored.verdict.get("resegment_ok")
        if not isinstance(ok, bool):
            continue
        checked += 1
        wrong += 0 if ok else 1
    return {
        "splits": _sum_counts(r.get("splits") for r in recorded) if recorded else None,
        "joins": _sum_counts(r.get("joins") for r in recorded) if recorded else None,
        "observations_split": split_obs if observations is not None else None,
        "observations_joined": joined_obs if observations is not None else None,
        "check_sample": len(subjects),
        "checked": checked,
        "wrong": wrong,
    }


def _precedent_section(playbook: dict[str, Any] | None) -> dict[str, Any] | None:
    if playbook is None:
        return None
    evidence = playbook.get("evidence")
    raw = evidence.get("precedent") if isinstance(evidence, dict) else None
    if not isinstance(raw, list):
        return None
    records = [p for p in raw if isinstance(p, dict)]
    signed = [p for p in records if p.get("signed") is True]
    variants = [
        p
        for p in signed
        if p.get("standard") is not True and isinstance(p.get("signed_text"), dict)
    ]
    refused = sum(len(p.get("refused_asks") or []) for p in records)
    # An opening is a record with a non-null opening_text (a distinct first
    # draft); the ticket's "openings by opened_with" splits exactly these.
    openings = [p for p in records if p.get("opening_text") is not None]
    openings_by: dict[str, int] | None = None
    records_by: dict[str, int] | None = None
    signed_undetermined: int | None = None
    if any("opened_with" in p for p in records):
        openings_by = _sorted_counts(Counter(_opened_with(p) for p in openings))
        records_by = _sorted_counts(Counter(_opened_with(p) for p in records))
        signed_undetermined = sum(1 for p in signed if p.get("opened_with") is None)
    return {
        "records": len(records),
        "signed": len(signed),
        "signed_standard": sum(1 for p in signed if p.get("standard") is True),
        "signed_variants": len(variants),
        "refused_asks": refused,
        "openings": len(openings),
        "openings_by_opened_with": openings_by,
        "records_by_opened_with": records_by,
        "signed_opened_with_undetermined": signed_undetermined,
    }


def _prior_standards_section(playbook: dict[str, Any] | None) -> dict[str, Any] | None:
    """Counts of our former forms and the deals that opened on them (issue #249).

    ``forms`` are the supplied forms (``baseline.prior_templates``);
    ``clauses_with_history`` the clauses with at least one form's text and
    ``entries`` those (clause, form) texts, of which ``unchanged`` read exactly
    as today's standard, ``judged`` carry a ``vs_standard`` and ``by_settled``
    the judged ones by outcome. ``opened`` / ``kept`` / ``moved_to_current``
    total the clauses' ``n_opened_prior`` / ``n_kept_prior`` /
    ``n_moved_to_current`` (distinct deals per clause); ``why_changed`` the
    clauses carrying an owner note. Counts only.
    """
    if playbook is None:
        return None
    evidence = playbook.get("evidence")
    raw = evidence.get("clauses") if isinstance(evidence, dict) else None
    if not isinstance(raw, list):
        return None
    clauses = [c for c in raw if isinstance(c, dict)]
    baseline = playbook.get("baseline")
    templates = baseline.get("prior_templates") if isinstance(baseline, dict) else None
    party = perspective_party(playbook)
    entries = judged = unchanged = with_history = notes = 0
    by_settled = dict.fromkeys(SETTLED, 0)
    for clause in clauses:
        priors = [p for p in clause.get("prior_standards") or [] if isinstance(p, dict)]
        with_history += bool(priors)
        notes += isinstance(clause.get("why_changed"), dict)
        standard = clause.get("our_standard")
        standard_text = standard.get("text") if isinstance(standard, dict) else None
        for prior in priors:
            entries += 1
            text = prior.get("text")
            if (
                isinstance(text, str)
                and isinstance(standard_text, str)
                and normalize_variant_text(text, party=party)
                == normalize_variant_text(standard_text, party=party)
            ):
                unchanged += 1
            vs = prior.get("vs_standard")
            if isinstance(vs, dict):
                judged += 1
                by_settled[settled_of(vs)] += 1
    return {
        "forms": len(templates) if isinstance(templates, list) else 0,
        "clauses_with_history": with_history,
        "entries": entries,
        "unchanged": unchanged,
        "judged": judged,
        "by_settled": by_settled,
        "opened": sum(_count(c.get("n_opened_prior")) or 0 for c in clauses),
        "kept": sum(_count(c.get("n_kept_prior")) or 0 for c in clauses),
        "moved_to_current": sum(_count(c.get("n_moved_to_current")) or 0 for c in clauses),
        "why_changed": notes,
    }


def _signed_on_section(playbook: dict[str, Any] | None) -> dict[str, Any] | None:
    """Counts of signed deals by how their signing date was established (issue #247).

    Read from the playbook's own ``corpus.documents[]``. ``null_dates`` are
    signed deals with no date; ``unsigned`` in-scope deals have none by
    definition. Counts only: no date value leaves the playbook.
    """
    if playbook is None:
        return None
    corpus = playbook.get("corpus")
    docs = corpus.get("documents") if isinstance(corpus, dict) else None
    if not isinstance(docs, list):
        return None
    cov = coverage([d for d in docs if isinstance(d, dict)])
    return {
        "in_scope": cov.in_scope,
        "signed": cov.signed,
        "dated": cov.dated,
        "null_dates": cov.null_dates,
        "by_basis": {_label(b, frozenset(SIGNED_ON_BASES)): n for b, n in cov.by_basis.items()},
    }


def _opened_with(record: dict[str, Any]) -> str:
    """A precedent record's ``opened_with`` label; ``null`` is ``undetermined``."""
    value = record.get("opened_with")
    return "undetermined" if value is None else _label(value, OPENED_WITH)


def _template_drift_section(
    playbook: dict[str, Any] | None, *, field: str = "standard", value: Any = True
) -> dict[str, Any] | None:
    """Per clause with ``our_standard``: share of our-paper signed deals whose
    record has ``field == value`` (default: signed our standard, ``standard``).

    The first sign of whether older forms of our paper need their own
    handling (epic #236). Anonymous: a sorted list of ratios plus summary
    statistics, never which clause type had which share.
    """
    if playbook is None:
        return None
    evidence = playbook.get("evidence")
    raw = evidence.get("precedent") if isinstance(evidence, dict) else None
    if not isinstance(raw, list):
        return None
    standard_tids = {
        c.get("taxonomy_id")
        for c in playbook_clauses(playbook)
        if isinstance(c, dict) and isinstance(c.get("our_standard"), dict)
    }
    deals: dict[Any, set[Any]] = {}
    standard_deals: dict[Any, set[Any]] = {}
    for p in raw:
        if not isinstance(p, dict) or p.get("signed") is not True or p.get("paper") != "ours":
            continue
        tid = p.get("taxonomy_id")
        if tid not in standard_tids:
            continue
        deals.setdefault(tid, set()).add(p.get("document_id"))
        if p.get(field) == value:
            standard_deals.setdefault(tid, set()).add(p.get("document_id"))
    shares = sorted(
        round(len(standard_deals.get(tid, set())) / len(ds), 4) for tid, ds in deals.items()
    )
    summary = _stats(shares)
    return {
        "clauses": len(shares),
        "median": summary["median"],
        "min": summary["min"],
        "max": summary["max"],
        "below_half": sum(1 for s in shares if s < 0.5),
        "over_one_third_below_half": sum(1 for s in shares if s < 0.5) * 3 > len(shares),
        "shares": shares,
    }


def _digest_section(playbook: dict[str, Any] | None) -> dict[str, Any] | None:
    if playbook is None:
        return None
    digest = playbook.get("digest")
    if not isinstance(digest, dict):
        return None
    from playbook_engine.digest import (  # noqa: PLC0415
        build_digest_uncapped,
        digest_settled_counts,
        digest_token_estimate,
    )

    clauses = [c for c in digest.get("clauses") or [] if isinstance(c, dict)]
    variant_groups = 0
    refused_groups = 0
    capped = 0
    opened_standard = 0
    kept_standard = 0
    changed_openings = 0
    for c in clauses:
        n_variants = _count(c.get("n_variants_total"))
        n_refused = _count(c.get("n_refused_total"))
        n_changed = _count(c.get("n_changed_openings_total"))
        if n_variants is not None:
            variant_groups += n_variants
        if n_refused is not None:
            refused_groups += n_refused
        if n_changed is not None:
            changed_openings += n_changed
        opened_standard += _count(c.get("n_opened_standard")) or 0
        kept_standard += _count(c.get("n_kept_standard")) or 0
        # A list is capped when it shows fewer entries than the uncapped
        # digest holds; the collapsed equivalent variants are one entry that
        # stands for ``n_texts`` distinct texts, so count texts, not entries.
        shown_variants = sum(
            (_count(v.get("n_texts")) or 1) if isinstance(v, dict) else 0
            for v in c.get("signed_variants") or []
        )
        shown_refused = len(c.get("refused_asks") or [])
        shown_changed = len(c.get("changed_openings") or [])
        if (
            (n_variants is not None and shown_variants < n_variants)
            or (n_refused is not None and shown_refused < n_refused)
            or (n_changed is not None and shown_changed < n_changed)
        ):
            capped += 1
    # Signed-variant opening facts, totalled over the UNCAPPED digest so the
    # cap cannot hide a deal. Every deal sits in exactly one variant group.
    from_standard: int | None = 0
    unchanged: int | None = 0
    settled: dict[str, dict[str, int]] | None
    try:
        uncapped = build_digest_uncapped(playbook)
        settled = digest_settled_counts(uncapped)
        for c in uncapped["clauses"]:
            for v in c["signed_variants"]:
                from_standard = (from_standard or 0) + (_count(v.get("n_from_standard")) or 0)
                unchanged = (unchanged or 0) + (_count(v.get("n_unchanged")) or 0)
    except (TypeError, ValueError, KeyError, AttributeError):
        # A playbook too malformed to rebuild its digest: report nothing
        # rather than a half-total (the validator names the real problem).
        from_standard = unchanged = None
        settled = None
    uncovered = digest.get("uncovered_clause_types")
    history = [
        c["standard_history"]
        for c in clauses
        if isinstance(c.get("standard_history"), dict)
        and isinstance(c["standard_history"].get("forms"), list)
    ]
    return {
        "digest_version": _label(digest.get("digest_version"), DIGEST_VERSIONS),
        "token_estimate": digest_token_estimate(digest),
        "clauses": len(clauses),
        "capped_clauses": capped,
        "signed_variant_groups": variant_groups,
        "refused_ask_groups": refused_groups,
        "n_opened_standard": opened_standard,
        "n_kept_standard": kept_standard,
        "signed_variants_n_from_standard": from_standard,
        "signed_variants_n_unchanged": unchanged,
        "changed_openings": changed_openings,
        "settled": settled,
        "standard_history_forms": sum(len(h["forms"]) for h in history),
        "uncovered_clause_types": len(uncovered) if isinstance(uncovered, list) else None,
    }


def _judge_unresolved(out_dir: Path) -> Counter[str]:
    """Judge items still waiting on a verdict, by kind.

    The store-backed judges re-queue an item under the SAME key when the
    banked verdict's rubric is stale, legacy under ``--strict-rubric``, or
    malformed — so a key's mere presence in ``judge/verdicts.jsonl`` does not
    answer it. An item counts as answered only when the store's last record
    for its key carries the rubric the item was queued under, which is the
    stamp ``playbook judge-apply`` writes when it banks the answer.
    """
    pending = _read_jsonl(out_dir / "judge" / "pending.jsonl")
    if not pending:
        return Counter()
    store = VerdictStore(out_dir / "judge" / "verdicts.jsonl")
    counter: Counter[str] = Counter()
    seen: set[Any] = set()
    for rec in pending:
        key = rec.get("key")
        if key in seen:
            continue
        seen.add(key)
        stored = store.get_record_by_key(key) if isinstance(key, str) else None
        if stored is not None and stored.rubric_version == rec.get("rubric_version"):
            continue
        counter[_label(rec.get("kind"), PENDING_KINDS)] += 1
    return counter


def _segment_unresolved(out_dir: Path) -> int:
    """Segment items whose document the segmentation cache cannot replay yet.

    The queue key (:func:`segment_payload_key`) and the cache key (canonical
    text + model + prompt version + schema hash + effort) are different
    hashes, so the item is resolved with the same lookup ``playbook segment``
    itself uses rather than by key.
    """
    pending = _read_jsonl(out_dir / "segment" / "pending.jsonl")
    if not pending:
        return 0
    cache = SegmentationVerdictCache(out_dir / "segment" / "cache.jsonl")
    unresolved = 0
    seen: set[Any] = set()
    for rec in pending:
        key = rec.get("key")
        if key in seen:
            continue
        seen.add(key)
        payload = rec.get("payload")
        text = payload.get("canonical_text") if isinstance(payload, dict) else None
        if isinstance(text, str) and cache.get(text, model=AGENT_SEGMENTER_MODEL) is not None:
            continue
        unresolved += 1
    return unresolved


def pending_queue_counts(out_dir: Path) -> tuple[int, int]:
    """``(judge_pending, segment_pending)`` — the agent work still owed.

    Shared by the scorecard's ``queues`` section and any driver script that
    must stop before ``mine``/``project`` while agent work is outstanding.
    """
    return sum(_judge_unresolved(out_dir).values()), _segment_unresolved(out_dir)


def _queues_section(out_dir: Path) -> dict[str, Any]:
    """Pending items the stores have not answered yet."""
    judge = _judge_unresolved(out_dir)
    return {
        "judge_pending": sum(judge.values()),
        "judge_pending_by_kind": _sorted_counts(judge),
        "segment_pending": _segment_unresolved(out_dir),
    }


def _dropped_section(playbook: dict[str, Any] | None, manifest: Any) -> dict[str, Any] | None:
    by_reason: Counter[str] = Counter()
    found = False
    if playbook is not None:
        corpus = playbook.get("corpus")
        stats = corpus.get("stats") if isinstance(corpus, dict) else None
        dropped = stats.get("dropped_observations") if isinstance(stats, dict) else None
        if isinstance(dropped, dict) and isinstance(dropped.get("by_reason"), dict):
            found = True
            for reason, n in dropped["by_reason"].items():
                if _count(n) is not None:
                    by_reason[_label(reason, DROPPED_REASONS)] += n
    if not found and isinstance(manifest, list):
        for d in manifest:
            dropped = d.get("dropped_observations") if isinstance(d, dict) else None
            if isinstance(dropped, dict):
                found = True
                for reason, n in dropped.items():
                    if _count(n) is not None:
                        by_reason[_label(reason, DROPPED_REASONS)] += n
    if not found:
        return None
    return {"count": sum(by_reason.values()), "by_reason": _sorted_counts(by_reason)}


def _equivalence_section(playbook: dict[str, Any] | None, out_dir: Path) -> dict[str, Any] | None:
    """The ``vs_standard`` verdicts of the playbook, by role (issue #240).

    Distinct texts, counts only: how many are eligible, drafted, checked,
    agreed, adjudicated, still disputed and unchecked, per role and in total,
    plus the same texts by the ``settled`` outcome the digest carries (issue #248);
    the label histogram; the agreement rate (agreed over checked); and the
    checker model id(s) recorded in the verdict store.
    """
    if playbook is None:
        return None
    evidence = playbook.get("evidence")
    agreement_type = playbook.get("agreement_type")
    agreement_type_id = agreement_type.get("id") if isinstance(agreement_type, dict) else None
    if not isinstance(evidence, dict) or not isinstance(agreement_type_id, str):
        return None
    summary = summarize_equivalence(evidence, agreement_type_id, perspective_party(playbook))
    models: Counter[str] = Counter()
    for _key, record in VerdictStore(out_dir / "judge" / "verdicts.jsonl").records():
        check = record.verdict.get("check")
        if isinstance(check, dict) and isinstance(check.get("by"), str):
            models[_label(check["by"], CHECKER_MODELS)] += 1
    summary["checker_models"] = _sorted_counts(models)
    return summary


#: The digest list that holds the texts of each equivalence role.
_DIGEST_LIST_OF_ROLE = {
    "signed": "signed_variants",
    "refused": "refused_asks",
    "opening": "changed_openings",
}


def _unlisted_settled(
    playbook: dict[str, Any], agreement_type_id: str, party: str | None
) -> dict[str, dict[str, int]]:
    """Judged texts the digest leaves out for the two named reasons, by list and outcome.

    Counted straight from the precedent records by the producers' own rules,
    never as a difference of counts. A text is counted only if no listed slot
    of the same role carries it, since the scorecard counts distinct texts.

    - ``signed_variants``: a signed-role text on a record that is not a
      signed deal (the last draft of an unsigned deal).
    - ``changed_openings``: an opening text that repeats one of its own
      record's refused asks (``precedent.changed_opening_members`` drops it;
      the digest lists the text as a refused ask).
    """
    evidence = playbook.get("evidence")
    out: dict[str, dict[str, int]] = {
        name: dict.fromkeys(SETTLED, 0) for name in _DIGEST_LIST_OF_ROLE.values()
    }
    if not isinstance(evidence, dict):
        return out
    excluded: dict[str, dict[str, set[str]]] = {
        role: {state: set() for state in SETTLED} for role in _DIGEST_LIST_OF_ROLE
    }
    listed: dict[str, set[str]] = {role: set() for role in _DIGEST_LIST_OF_ROLE}
    for slot in iter_slots(evidence):
        if slot.role not in _DIGEST_LIST_OF_ROLE:
            # A former form's clause (issue #249) sits in the digest's
            # ``standard_history``, not in one of these three lists.
            continue
        if not is_eligible(slot, party):
            continue
        key = slot_key(slot, agreement_type_id, party)
        vs = slot.entry.get("vs_standard")
        if key is None or not isinstance(vs, dict):
            continue
        if slot.role == ROLE_SIGNED:
            is_excluded = slot.record.get("signed") is not True
        elif slot.role == ROLE_OPENING:
            asks = slot.record.get("refused_asks")
            own_asks = {
                normalize_variant_text(a["text"], party=party)
                for a in (asks if isinstance(asks, list) else [])
                if isinstance(a, dict) and isinstance(a.get("text"), str)
            }
            text = slot.entry.get("text")
            is_excluded = isinstance(text, str) and (
                normalize_variant_text(text, party=party) in own_asks
            )
        else:
            is_excluded = False
        if is_excluded:
            excluded[slot.role][settled_of(vs)].add(key)
        else:
            listed[slot.role].add(key)
    for role, name in _DIGEST_LIST_OF_ROLE.items():
        for state in SETTLED:
            out[name][state] = len(excluded[role][state] - listed[role])
    return out


def _add_settled_not_listed(
    playbook: dict[str, Any] | None,
    digest: dict[str, Any] | None,
    equivalence: dict[str, Any] | None,
) -> None:
    """Reconcile ``digest.settled`` with ``equivalence.by_role[*].settled`` (issue #248).

    The digest lists a judged text only as a signed variant of a signed deal,
    a refused ask, or a changed opening, so a role's judged texts are the
    digest list's plus the ones it leaves out for a named reason:
    ``settled_not_listed`` counts those directly (:func:`_unlisted_settled`):
    the last draft of an unsigned deal, and an opening that repeats one of its
    own deal's refused asks (the digest lists it as a refused ask).
    ``settled_unreconciled`` is what is left, per list and outcome: the
    equivalence count minus the digest count minus ``settled_not_listed``. It
    must be 0 everywhere; a nonzero or negative value is a grouping or
    projection bug, not a design choice.
    """
    if playbook is None or digest is None or equivalence is None:
        return
    settled = digest.get("settled")
    by_role = equivalence.get("by_role")
    agreement_type = playbook.get("agreement_type")
    agreement_type_id = agreement_type.get("id") if isinstance(agreement_type, dict) else None
    if (
        not isinstance(settled, dict)
        or not isinstance(by_role, dict)
        or not isinstance(agreement_type_id, str)
    ):
        return
    not_listed = _unlisted_settled(playbook, agreement_type_id, perspective_party(playbook))
    unreconciled: dict[str, dict[str, int]] = {}
    for role, name in _DIGEST_LIST_OF_ROLE.items():
        row = by_role.get(role)
        row_settled = row.get("settled") if isinstance(row, dict) else None
        listed = settled.get(name)
        if not isinstance(row_settled, dict) or not isinstance(listed, dict):
            return
        unreconciled[name] = {
            state: (_count(row_settled.get(state)) or 0)
            - (_count(listed.get(state)) or 0)
            - not_listed[name][state]
            for state in listed
        }
    digest["settled_not_listed"] = not_listed
    digest["settled_unreconciled"] = unreconciled


def _record_texts(playbook: dict[str, Any]) -> set[str]:
    """Every opening, signed and refused-ask text the precedent records hold."""
    evidence = playbook.get("evidence")
    records = evidence.get("precedent") if isinstance(evidence, dict) else None
    texts: set[str] = set()
    for record in records if isinstance(records, list) else []:
        if not isinstance(record, dict):
            continue
        refs: list[Any] = [record.get("opening_text"), record.get("signed_text")]
        asks = record.get("refused_asks")
        refs.extend(asks if isinstance(asks, list) else [])
        for ref in refs:
            text = ref.get("text") if isinstance(ref, dict) else None
            if isinstance(text, str):
                texts.add(text)
    return texts


def _dossiers_section(playbook: dict[str, Any] | None) -> dict[str, Any] | None:
    """Critic dossier counts and sizes (OPF 0.6, issue #228) — ``null`` when not carried.

    ``cut_texts`` counts excerpt texts that are NOT a precedent record's own
    text whole (it must be 0: no text is ever cut part-way);
    ``excerpts_dropped`` the selected excerpts dropped whole to fit a budget;
    ``over_budget_single_excerpt`` every dossier over its budget, each holding
    its single kept excerpt (the only dossier that may exceed its budget: a
    dossier drops whole Floor rules and whole excerpts, keeping the first, to
    fit, and the validator refuses any other over its budget);
    ``max_budget`` the largest per-clause budget.
    """
    dossiers = playbook.get("dossiers") if playbook is not None else None
    if playbook is None or not isinstance(dossiers, dict):
        return None
    known = _record_texts(playbook)
    sizes: list[int] = []
    budgets: list[int] = []
    cut = dropped = over = 0
    for dossier in dossiers.values():
        sizes.append(len(canonicalize(dossier)) // 4)
        if not isinstance(dossier, dict):
            budgets.append(DOSSIER_MIN_BUDGET)
            continue
        budgets.append(dossier_budget(dossier_standard_text(dossier)))
        dropped += _count(dossier.get("n_omitted")) or 0
        excerpts = dossier.get("excerpts")
        if sizes[-1] > budgets[-1] and isinstance(excerpts, list) and len(excerpts) == 1:
            over += 1
        for excerpt in excerpts if isinstance(excerpts, list) else []:
            if not isinstance(excerpt, dict):
                continue
            for key in ("opening", "signed"):
                text = excerpt.get(key)
                if text is not None and not (isinstance(text, str) and text in known):
                    cut += 1
    summary = _stats(sizes)
    return {
        "count": len(sizes),
        "max_tokens": summary["max"],
        "median_tokens": summary["median"],
        "max_budget": max(budgets) if budgets else None,
        "cut_texts": cut,
        "excerpts_dropped": dropped,
        "over_budget_single_excerpt": over,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_scorecard(out_dir: Path) -> dict[str, Any]:
    """Build the counts-only scorecard of a ``mine``/``project`` out-dir."""
    loaded = _read_json(out_dir / "playbook.opf.json")
    playbook = loaded if isinstance(loaded, dict) else None
    manifest = _read_json(out_dir / "corpus_manifest.json")
    observations = _read_jsonl(out_dir / "observations.jsonl")
    template_obs = _read_jsonl(out_dir / "template_observations.jsonl")
    quarantine = _read_json(out_dir / "quarantine.json")
    digest_section = _digest_section(playbook)
    equivalence_section = _equivalence_section(playbook, out_dir)
    _add_settled_not_listed(playbook, digest_section, equivalence_section)
    return {
        "scorecard_version": SCORECARD_VERSION,
        "opf_version": (
            _label(playbook.get("opf_version"), OPF_VERSIONS) if playbook is not None else None
        ),
        "corpus": _corpus_section(manifest, quarantine, playbook),
        "template": _template_section(template_obs, playbook),
        "classification": _classification_section(observations),
        "resegmentation": _resegmentation_section(manifest, observations, out_dir),
        "precedent": _precedent_section(playbook),
        "prior_standards": _prior_standards_section(playbook),
        "signed_on": _signed_on_section(playbook),
        "template_drift": _template_drift_section(playbook),
        # Issue #233: the drift of what each clause OPENED with — per clause
        # with our_standard, our-paper signed deals opening with our standard
        # over our-paper signed deals that carry the clause. The trigger for
        # the deferred decision on older Exos forms is
        # ``over_one_third_below_half``.
        "opening_drift": _template_drift_section(playbook, field="opened_with", value="standard"),
        "dropped_observations": _dropped_section(playbook, manifest),
        "digest": digest_section,
        "equivalence": equivalence_section,
        "queues": _queues_section(out_dir),
        "dossiers": _dossiers_section(playbook),
    }


def write_scorecard(scorecard: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(scorecard, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def flatten_scorecard(scorecard: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Dotted path -> scalar leaf. A list is reported by its length (``path[]``)."""
    flat: dict[str, Any] = {}
    for key in sorted(scorecard):
        value = scorecard[key]
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            nested = flatten_scorecard(value, path)
            flat.update(nested if nested else {path: None})
        elif isinstance(value, list):
            flat[f"{path}[]"] = len(value)
        else:
            flat[path] = value
    return flat


def compare_scorecards(
    current: dict[str, Any], baseline: dict[str, Any]
) -> list[tuple[str, Any, Any, Any]]:
    """``(field, baseline, current, delta)`` for every field of either card.

    ``delta`` is ``current - baseline`` when both are numbers (bools excluded),
    else ``None``.
    """
    cur = flatten_scorecard(current)
    base = flatten_scorecard(baseline)
    rows: list[tuple[str, Any, Any, Any]] = []
    for path in sorted(set(cur) | set(base)):
        b: Any = base.get(path)
        c: Any = cur.get(path)
        delta: Any = None
        if _is_number(b) and _is_number(c):
            delta = round(c - b, 4) if isinstance(c, float) or isinstance(b, float) else c - b
        rows.append((path, b, c, delta))
    return rows


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _cell(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _format_rows(header: Iterable[str], rows: list[list[str]]) -> str:
    table = [list(header), *rows]
    widths = [max(len(r[i]) for r in table) for i in range(len(table[0]))]
    lines = ["  ".join(cell.ljust(widths[i]) for i, cell in enumerate(r)).rstrip() for r in table]
    lines.insert(1, "  ".join("-" * w for w in widths))
    return "\n".join(lines)


def render_table(scorecard: dict[str, Any], baseline: dict[str, Any] | None = None) -> str:
    """Plain-text table of the scorecard, or of its comparison with *baseline*."""
    if baseline is None:
        rows = [[path, _cell(v)] for path, v in flatten_scorecard(scorecard).items()]
        return _format_rows(("field", "value"), rows)
    compared = compare_scorecards(scorecard, baseline)
    rows = [
        [path, _cell(b), _cell(c), "" if d is None else (f"+{d}" if d > 0 else str(d))]
        for path, b, c, d in compared
    ]
    return _format_rows(("field", "baseline", "current", "delta"), rows)
