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
Fields the current OPF does not carry yet (the critic ``dossiers``, issue
#228) are ``null`` until a playbook carries them.
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
from playbook_engine.equivalence import LABELS as EQUIVALENCE_LABELS
from playbook_engine.equivalence import ROLES as EQUIVALENCE_ROLES
from playbook_engine.equivalence import summarize as summarize_equivalence
from playbook_engine.equivalence_check import CHECKER_MODEL
from playbook_engine.llm_segmenter_batch import SegmentationVerdictCache
from playbook_engine.observation_builder import (
    DROPPED_ORIGIN_UNDETERMINED,
    DROPPED_REFUSED_UNSIGNED,
    DROPPED_STANDARD_REMOVED_UNSIGNED,
    DROPPED_SURVIVES_IN_TERMINAL,
    OUTCOME_OPENING,
)
from playbook_engine.opf_accessors import perspective_party, playbook_clauses

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
SCORECARD_VERSION = 4

#: Default file name, written into the out-dir.
SCORECARD_FILENAME = "scorecard.json"

#: Any label outside its closed vocabulary is written as this.
OTHER = "other"

#: Classification bases (``clause_classifier._BASIS_VALUES``) plus the two
#: the scorecard itself introduces: ``aligned`` (the observation's
#: taxonomy_id came from its aligned row, ``pipeline``) and ``unrecorded``
#: (an observation from a store that predates ``x_classification_basis``).
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
        "aligned",
        "unrecorded",
    }
)

DROPPED_REASONS = frozenset(
    {
        DROPPED_SURVIVES_IN_TERMINAL,
        DROPPED_ORIGIN_UNDETERMINED,
        DROPPED_STANDARD_REMOVED_UNSIGNED,
        DROPPED_REFUSED_UNSIGNED,
    }
)

#: ``PendingQueue`` record kinds (agent_judge / agent_segmenter). A kind added
#: through ``agent_judge.register_verdict_kind`` must be added here too, or its
#: pending items are counted under ``other``.
PENDING_KINDS = frozenset({"classify", "provenance", "scope", "segment", "equivalence"})

#: Checker model ids the scorecard names (the pinned one); any other recorded
#: id is written as ``other`` — a model id from the store is still data from
#: the out-dir.
CHECKER_MODELS = frozenset({CHECKER_MODEL})

#: ``vs_standard`` labels and the three roles a judged text can play.
EQUIVALENCE_VOCABULARY = frozenset(EQUIVALENCE_LABELS) | frozenset(EQUIVALENCE_ROLES)

#: Observation ``provenance`` values — the paper-side diagnostic split.
PAPER_SIDES = frozenset({"our_paper", "counterparty_paper", "unknown"})

#: ``opened_with`` (OPF 0.5, issue #233); a null value is ``undetermined``.
OPENED_WITH = frozenset({"standard", "non_standard", "absent", "undetermined"})

#: The engine's one format; any other version a playbook claims is labelled
#: ``other``.
OPF_VERSIONS = frozenset({"0.5"})
DIGEST_VERSIONS = frozenset({"3", "4"})

#: Every string the scorecard may carry as a value or a data-derived key.
ENUM_LABELS = frozenset(
    {OTHER}
    | CLASSIFICATION_BASES
    | DROPPED_REASONS
    | PENDING_KINDS
    | PAPER_SIDES
    | OPENED_WITH
    | OPF_VERSIONS
    | DIGEST_VERSIONS
    | CHECKER_MODELS
    | EQUIVALENCE_VOCABULARY
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
    classified = [o for o in observations if o.get("taxonomy_id")]
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
            for o in observations
        )
        by_basis = _sorted_counts(counter)
    return {
        "observations": len(observations),
        "classified": len(classified),
        "classified_share": _ratio(len(classified), len(observations)),
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
    from playbook_engine.digest import digest_token_estimate  # noqa: PLC0415

    clauses = [c for c in digest.get("clauses") or [] if isinstance(c, dict)]
    variant_groups = 0
    refused_groups = 0
    capped = 0
    for c in clauses:
        n_variants = _count(c.get("n_variants_total"))
        n_refused = _count(c.get("n_refused_total"))
        if n_variants is not None:
            variant_groups += n_variants
        if n_refused is not None:
            refused_groups += n_refused
        shown_variants = len(c.get("signed_variants") or [])
        shown_refused = len(c.get("refused_asks") or [])
        if (n_variants is not None and shown_variants < n_variants) or (
            n_refused is not None and shown_refused < n_refused
        ):
            capped += 1
    return {
        "digest_version": _label(digest.get("digest_version"), DIGEST_VERSIONS),
        "token_estimate": digest_token_estimate(digest),
        "clauses": len(clauses),
        "capped_clauses": capped,
        "signed_variant_groups": variant_groups,
        "refused_ask_groups": refused_groups,
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
    agreed, adjudicated, still disputed and unchecked, per role and in total;
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


def _dossiers_section(playbook: dict[str, Any] | None) -> dict[str, Any] | None:
    """Critic dossier sizes (OPF 0.5, issue #228) — ``null`` until carried."""
    dossiers = playbook.get("dossiers") if playbook is not None else None
    if not isinstance(dossiers, dict):
        return None
    sizes = [len(canonicalize(d)) // 4 for d in dossiers.values()]
    summary = _stats(sizes)
    return {
        "count": len(sizes),
        "max_tokens": summary["max"],
        "median_tokens": summary["median"],
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
    return {
        "scorecard_version": SCORECARD_VERSION,
        "opf_version": (
            _label(playbook.get("opf_version"), OPF_VERSIONS) if playbook is not None else None
        ),
        "corpus": _corpus_section(manifest, quarantine, playbook),
        "template": _template_section(template_obs, playbook),
        "classification": _classification_section(observations),
        "precedent": _precedent_section(playbook),
        "template_drift": _template_drift_section(playbook),
        # Issue #233: the drift of what each clause OPENED with — per clause
        # with our_standard, our-paper signed deals opening with our standard
        # over our-paper signed deals that carry the clause. The trigger for
        # the deferred decision on older Exos forms is
        # ``over_one_third_below_half``.
        "opening_drift": _template_drift_section(playbook, field="opened_with", value="standard"),
        "dropped_observations": _dropped_section(playbook, manifest),
        "digest": _digest_section(playbook),
        "equivalence": _equivalence_section(playbook, out_dir),
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
