"""Pipeline orchestration — corpus → playbook (L1 → L5).

Wires all pipeline stages into a single ``compile_corpus()`` call.
LLM-facing stages accept injected judge objects; stub implementations
are provided as CLI defaults when no real LLM is configured.

Security: no agreement content is stored in this module.
All corpus content is read from caller-supplied paths at runtime.
"""

from __future__ import annotations

import dataclasses
import datetime
import json
import os
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from playbook_engine.artifact_store import (
    ArtifactStore,
    _sha256_file,
    make_config_fingerprint,
    make_doc_key,
)
from playbook_engine.canonicalize import file_sha256
from playbook_engine.clause_aligner import AlignmentJudge, align_versions
from playbook_engine.clause_classifier import (
    AMBIGUITY_THRESHOLD,
    AUTO_CLASSIFY_THRESHOLD,
    ClassificationJudge,
    ClassifiedClause,
    ClauseClassification,
    classify_tree,
)
from playbook_engine.clause_differ import ClauseDiff, diff_aligned
from playbook_engine.clause_library_compiler import compile_clause_library
from playbook_engine.clause_position_compiler import (
    CoherenceJudge,
    compile_clause_positions,
)
from playbook_engine.clause_tree import ClauseTree
from playbook_engine.config import EngineConfig
from playbook_engine.deviation_classifier import (
    DeviationJudge,
    DeviationResult,
    RiskDelta,
    assess_deviations,
)
from playbook_engine.docx_ingester import TrackedChanges, ingest_docx
from playbook_engine.entity_registry import (
    DEFAULT_REGISTRY_PATH,
    EntityRegistry,
    entity_slug,
    pseudonymize_document_id,
    pseudonymize_text,
    write_holdout_map,
)
from playbook_engine.extraction import ExtractionCache, detect_extractor, extract_blocks
from playbook_engine.judgment import (
    BatchedClassificationJudge,
    BatchedDeviationJudge,
    BatchedScopeJudge,
    JudgmentCache,
)
from playbook_engine.llm_segmentation_stage import SegmentFn, segment_to_tree
from playbook_engine.llm_segmenter import DEFAULT_MODEL
from playbook_engine.llm_segmenter_batch import (
    DEFAULT_EFFORT,
    PROMPT_VERSION,
    SCHEMA_HASH,
    NormalizeTrailError,
    NormalizeTrailFn,
    NormalizeTrailResult,
    SegmentationBatchItem,
    SegmentationVerdictCache,
    normalize_trail,
    segment_documents_batch,
)
from playbook_engine.observation_builder import (
    Observation,
    ObservationCitation,
    RoundMove,
    build_observations,
    build_round_moves,
    read_observations_jsonl,
    read_round_moves_jsonl,
    round_move_from_dict,
    truncate_move_summaries,
    write_observations_jsonl,
    write_round_moves_jsonl,
)
from playbook_engine.pdf_ingester import ingest_pdf
from playbook_engine.playbook_assembler import assemble_playbook, write_playbook
from playbook_engine.provenance_detector import ProvenanceJudge, ProvenanceResult, detect_provenance
from playbook_engine.reversal_detector import detect_reversals
from playbook_engine.rtf_ingester import ingest_rtf
from playbook_engine.scope_gate import (
    ScopeDecision,
    ScopeJudge,
    ScopeLog,
    scope_gate,
)
from playbook_engine.segmentation_grounding import Block, SegNode
from playbook_engine.segmentation_qa import SegmentationQAError, run_gates
from playbook_engine.segmenter import segment
from playbook_engine.signed_detector import SignedJudge, SignedStatus, detect_signed
from playbook_engine.taxonomy import Taxonomy
from playbook_engine.tracked_changes_overlay import HunkEnrichment, enrich_clause_diff
from playbook_engine.version_orderer import (
    Hints,
    HintsError,
    TrailJudge,
    VersionInput,
    order_versions,
)

_TEXT_SUMMARY_MAX = 200
_SUPPORTED_EXTENSIONS = frozenset({".docx", ".pdf", ".rtf"})

# Media types for version_files content addresses (issue #185, OPF §4).
_MEDIA_TYPES: dict[str, str] = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".doc": "application/msword",
    ".rtf": "application/rtf",
    ".txt": "text/plain",
}

# Bump on any change to the "unchanged clause vs. template" deviation
# comparison logic (deviation_classifier.assess_deviations' unchanged fast
# path; _observations_from_single_version) — folded into mine_corpus's
# config_fp so a warm L1-L4 stage cache from before the change is never
# replayed verbatim (issue #103).
#
# v2 (issue #105): template_std_by_tid now resolves from t_obs.full_text
# instead of t_obs.text_summary, and cached observations gained a full_text
# field — a warm cache from before this fix has neither, so
# _restore_observations' full_text fallback would silently replay the old
# 200-char-truncated our_standard/full_text forever without this bump.
#
# v3 (issue #106): build_observations now also emits a direct Observation per
# whole-clause ReversalRecord (previously dropped silently — see
# observation_builder.build_observations), and the trail dict gained a
# "reversals" key — a warm cache from before this fix has neither, so a
# document with a genuine whole-clause reversal would keep replaying the old
# (incomplete) observations.jsonl/trail entry forever without this bump.
#
# v4 (issues #177/#185): the per-doc result gained "round_moves", cached
# observations gained the dynamics keys (proposed_by/observed_at/
# counterparty_ref), and corpus_doc gained "version_files" — a warm cache
# from before those features would replay results with none of them
# (mine_corpus reads round_moves via .get), so an upgraded engine would
# silently produce playbooks with no negotiation_trail, no dynamics, no
# content addresses and no corpus.snapshot, forever, on any warm store.
_DEVIATION_VS_TEMPLATE_VERSION = 4

# Legacy binary Word format — not ingestible directly, but common in
# negotiation history from the 2000s-2010s. Flagged distinctly (not lumped
# into the generic "unsupported files" case) since silently dropping an
# early .doc draft can misrepresent a late redline as the negotiation's
# start, skewing provenance and deviation direction (issue #100).
_LEGACY_EXTENSIONS = frozenset({".doc"})
_LEGACY_FORMAT_INSTRUCTION = "soffice --convert-to docx"

# ---------------------------------------------------------------------------
# Stub judges — CLI defaults when no real LLM is configured
# ---------------------------------------------------------------------------


class _AllInScopeJudge:
    """Accepts every document as in-scope (no LLM required).

    This is a stub used when no real ``ScopeJudge`` is injected. It must NOT
    claim ``basis="judge"`` — that masquerades a fabricated default as a real
    scope verdict, indistinguishable downstream (scope.json, the assembled
    playbook) from a document an LLM actually evaluated for relevance. It
    emits an honest ``basis="stub"`` instead.
    """

    def judge(self, tree: ClauseTree, agreement_type: Any) -> Any:
        return ScopeDecision(
            in_scope=True,
            scope_rationale="Accepted without LLM judgment (stub mode).",
            scope_confidence=0.5,
            basis="stub",
        )


class _NullClassificationJudge:
    """Marks all ambiguous nodes as unclassified; Jaccard fast-path handles the rest."""

    def classify_batch(
        self,
        nodes: list[Any],
        taxonomy: Any,
        hints: Any = None,
    ) -> list[ClauseClassification]:
        return [
            ClauseClassification(taxonomy_id=None, confidence=0.0, basis="unclassified")
            for _ in nodes
        ]


_NEUTRAL_RISK = RiskDelta(direction="neutral", magnitude="none")


class _NullDeviationJudge:
    """Records every changed clause as unjudged (no LLM available to assess it).

    This is a stub used when no real ``DeviationJudge`` is injected. It must NOT
    claim ``basis="judge"`` — that masquerades a fabricated default as a real
    judge verdict, and (worse) the position compiler treats a neutral-risk
    judged deviation as an "acceptable variant", so every stub-mode clause would
    manufacture a confident ``acceptable_variants_exist`` position from a clause
    nothing actually judged. It emits an honest ``basis="needs_review"`` instead,
    which the position compiler excludes from acceptable-variant / position
    logic (see ``_UNJUDGED_BASES``). ``deviation`` stays ``"substantive"`` — the
    clause genuinely changed (unchanged/near-identical clauses never reach the
    judge) and the OPF observed-position schema only permits
    none/reworded_equivalent/substantive — but the neutral ``risk_delta`` is a
    placeholder, not a real risk assessment, which ``needs_review`` signals.
    """

    def assess_batch(self, items: list[dict[str, str]], our_standard: str) -> list[DeviationResult]:
        return [
            DeviationResult(
                deviation="substantive",
                risk_delta=_NEUTRAL_RISK,
                basis="needs_review",
                rationale="No deviation judge configured; clause flagged for review (stub mode).",
            )
            for _ in items
        ]


def _judge_identity(judge: Any) -> str:
    """Return a stable identity string for a judge instance.

    Combines the judge's concrete class name with an optional ``model_id``
    attribute the judge may expose (real LLM-backed judges should set this to
    their model/prompt version so upgrading the underlying model busts the
    cache too; judges that don't declare one fall back to ``"unversioned"``).

    The class name alone already distinguishes every judge implementation in
    this codebase today (the stub judges above, ``StoreBackedScopeJudge`` and
    friends in ``agent_judge.py``, and any test fake) — this is the load-
    bearing half of the fix for issue #102: a verdict cached while
    ``_AllInScopeJudge`` was injected must never be replayed once a
    differently-classed (e.g. real LLM-backed) judge takes its place, because
    the previous hardcoded ``model_id="stub-v1"`` could not tell them apart.
    """
    model_id = getattr(judge, "model_id", None)
    return f"{type(judge).__name__}:{model_id or 'unversioned'}"


# ---------------------------------------------------------------------------
# Error type
# ---------------------------------------------------------------------------


@dataclass
class PipelineError(Exception):
    """Raised when the pipeline cannot complete."""

    message: str

    def __str__(self) -> str:
        return self.message


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _normalize_version(version: str) -> int | str | None:
    """Normalize a filename-stem version to OPF corpus schema (integer | null).

    "v2" → 2, "v1" → 1, "2" → 2. Non-parseable → None (the schema allows null).
    """
    if not version:
        return None
    stripped = version.lstrip("vV")
    try:
        return int(stripped)
    except ValueError:
        return None


def _ingest_file(path: Path, document_id: str, version: str) -> ClauseTree:
    """Ingest one agreement file, dispatching by extension → ClauseTree."""
    return _ingest_file_tracked(path, document_id, version)[0]


def _ingest_file_tracked(
    path: Path, document_id: str, version: str
) -> tuple[ClauseTree, TrackedChanges | None]:
    """Ingest one agreement file, dispatching by extension → ``(tree, tracked)``.

    Same dispatch as :func:`_ingest_file`, but also surfaces the DOCX
    tracked-changes side-channel (issue #88) so callers can attribute
    redline authorship downstream — see
    :mod:`playbook_engine.tracked_changes_overlay`. ``tracked`` is always
    ``None`` for RTF/PDF (no tracked-changes concept) and for a DOCX file
    with no ``w:ins``/``w:del`` elements (``TrackedChanges.changes`` empty).
    """
    ext = path.suffix.lower()
    if ext == ".docx":
        result = ingest_docx(path, document_id, version)
        return result.tree, result.tracked
    if ext == ".rtf":
        return ingest_rtf(path, document_id, version).tree, None
    if ext == ".pdf":
        return ingest_pdf(path, document_id, version).tree, None
    raise ValueError(f"Unsupported file extension {ext!r}; expected .docx, .pdf, or .rtf")


def _discover_versions(doc_dir: Path) -> list[Path]:
    """Return agreement files in a document directory, sorted by name."""
    return sorted(
        p for p in doc_dir.iterdir() if p.is_file() and p.suffix.lower() in _SUPPORTED_EXTENSIONS
    )


def _discover_legacy_doc_files(doc_dir: Path) -> list[Path]:
    """Return legacy .doc files in a document directory, sorted by name.

    These are excluded from :func:`_discover_versions` (the engine cannot read
    them) but are common in real negotiation history, so callers surface them
    distinctly rather than silently losing early drafts — see
    ``_LEGACY_EXTENSIONS``.
    """
    return sorted(
        p for p in doc_dir.iterdir() if p.is_file() and p.suffix.lower() in _LEGACY_EXTENSIONS
    )


def _llm_segment_file(
    path: Path,
    document_id: str,
    version: str,
    taxonomy_ids: list[str],
    segment_fn: SegmentFn | None,
    segmentation_cache: SegmentationVerdictCache | None = None,
    model: str = DEFAULT_MODEL,
    extraction_cache: ExtractionCache | None = None,
) -> tuple[ClauseTree, dict[str, str | None]]:
    """LLM-segment one agreement file → ``(tree, taxonomy_by_path)``.

    The LLM-segmentation alternative to ``segment(_ingest_file(...))``: same
    ``(document_id, version)`` call shape as ``_ingest_file``, same
    ``ClauseTree`` return contract, but classification happens in the same
    LLM pass (``taxonomy_by_path`` carries it) so callers on this path skip
    ``classify_tree`` entirely.

    ``segment_to_tree`` itself has no notion of the corpus's real
    ``document_id``/``version``/source filename (``segment_verify_repair``
    doesn't accept them) — set them here so the normalized tree and every
    downstream citation (trail, observations) carry the real identity, not
    ``run_gates``'s ``"doc"``/``"v1"``/``""`` defaults.

    ``segmentation_cache``, when given, is forwarded to ``segment_to_tree``
    so a repeat run over unchanged source content skips the LLM call
    entirely (issue #91) — the same content-hash cache
    :func:`~playbook_engine.llm_segmenter_batch.segment_documents_batch`
    already honors on the batch path.

    ``model`` must be the *actual* model id ``segment_fn`` calls through to
    (``config.segmentation.model`` — see issue #131) and is forwarded to
    ``segment_to_tree`` unchanged, which uses it as part of the cache key.
    Passing a stale/default model id here while ``segment_fn`` was bound to a
    different one would let a config's model change silently replay another
    model's cached segmentation instead of busting the cache.

    ``extraction_cache``, when given, is forwarded to ``segment_to_tree`` so a
    repeat run over unchanged source content skips extraction (docling/
    pdfplumber/python-docx/pandoc) entirely, independent of
    ``segmentation_cache`` (issue #132).
    """
    result = segment_to_tree(
        path,
        taxonomy_ids=taxonomy_ids,
        segment_fn=segment_fn,
        cache=segmentation_cache,
        model=model,
        extraction_cache=extraction_cache,
    )
    result.tree.document_id = document_id
    result.tree.version = version
    result.tree.source_file = path.name
    return result.tree, result.taxonomy_by_path


def _batch_custom_id(doc_id: str, version: str) -> str:
    """Build the ``custom_id`` used to key one version into the batch (issue #76)."""
    return f"{doc_id}/{version}"


@dataclass
class _BatchExtraction:
    """One version's extracted content, held between the pre-pass and grounding.

    Populated by :func:`_collect_batch_items` for every version file that
    extracts cleanly; a version whose extraction fails is simply absent here
    (same "skip this one version, warn, keep going" tolerance as the
    synchronous ``_ingest_file``/``_llm_segment_file`` loop in
    ``_compute_doc_result`` — extraction failure is not a QA-gate failure and
    must not abort the whole corpus batch).
    """

    canonical_text: str
    blocks: list[Block]
    source_file: str


def _collect_batch_items(
    doc_versions: dict[str, dict[str, Path]],
    progress: Callable[[str], None],
    extraction_cache: ExtractionCache | None = None,
) -> tuple[list[SegmentationBatchItem], dict[str, dict[str, _BatchExtraction]]]:
    """Extract every version file up front and build the corpus-wide batch request.

    Args:
        doc_versions: ``{doc_id: {version_id: path}}`` — every document's
                      version files to extract, as discovered by the caller.
        progress:     Progress callback (mirrors ``_compute_doc_result``'s
                      per-file warning convention).
        extraction_cache: Optional :class:`~playbook_engine.extraction.ExtractionCache`,
                      forwarded to ``extract_blocks`` for every version — a
                      hit skips extraction entirely for that version
                      (issue #132).

    Returns:
        ``(items, extractions)`` — *items* is the flat list of
        :class:`~playbook_engine.llm_segmenter_batch.SegmentationBatchItem`
        to submit in one :func:`~playbook_engine.llm_segmenter_batch.segment_documents_batch`
        call, keyed by :func:`_batch_custom_id`. *extractions* mirrors
        *doc_versions*' nesting (``{doc_id: {version_id: _BatchExtraction}}``)
        so the caller can look up the ``canonical_text``/``blocks`` a given
        batch result belongs to once grounding runs. A version whose
        extraction fails is present in neither return value.
    """
    items: list[SegmentationBatchItem] = []
    extractions: dict[str, dict[str, _BatchExtraction]] = {}

    for doc_id, versions in doc_versions.items():
        for vid, path in versions.items():
            try:
                canonical_text, blocks, _extractor = extract_blocks(path, cache=extraction_cache)
            except Exception as exc:  # noqa: BLE001 — same tolerance as the sync path
                progress(f"    WARNING: {path.name}: {exc}")
                continue
            extractions.setdefault(doc_id, {})[vid] = _BatchExtraction(
                canonical_text=canonical_text, blocks=blocks, source_file=path.name
            )
            items.append(
                SegmentationBatchItem(_batch_custom_id(doc_id, vid), canonical_text, blocks)
            )

    return items, extractions


def _ground_batch_result(
    doc_id: str,
    version: str,
    extraction: _BatchExtraction,
    seg_nodes: list[SegNode],
    taxonomy_ids: list[str],
) -> tuple[ClauseTree, dict[str, str | None]]:
    """Run the deterministic QA gates against one batched version's ``SegNode``s.

    The batch path has no per-document ``segment_fn`` to retry with — unlike
    :func:`~playbook_engine.llm_segmentation_stage.segment_to_tree`'s
    verify/repair loop, a gate failure here is not resubmitted to the model.
    This is intentional (see issue #76's "keep fail-loud QA... same contract
    as the sync path" — fail-loud is the contract being mirrored, not the
    repair mechanics, which would mean either a synchronous per-doc fallback
    call or a second batch round-trip, both out of scope here): a
    :class:`~playbook_engine.segmentation_qa.SegmentationQAError` propagates
    uncaught, flagging the document for human review exactly as the
    synchronous path's exhausted-repairs failure does.

    Returns:
        ``(tree, taxonomy_by_path)`` — same contract as ``_llm_segment_file``.

    Raises:
        SegmentationQAError: the batched segmentation fails any gate.
    """
    result = run_gates(
        extraction.canonical_text,
        extraction.blocks,
        seg_nodes,
        taxonomy_ids=taxonomy_ids,
        document_id=doc_id,
        version=version,
        source_file=extraction.source_file,
    )
    return result.tree, result.taxonomy_by_path


def _default_normalize_trail_fn(taxonomy_ids: list[str]) -> NormalizeTrailFn:
    """Bind :func:`~playbook_engine.llm_segmenter_batch.normalize_trail` to *taxonomy_ids*.

    Same lazy-construction pattern as ``_default_segment_fn`` in
    ``llm_segmentation_stage.py``: the real ``anthropic`` client is never
    constructed here, only deferred to ``normalize_trail`` itself via
    ``client=None``. No test exercises this function directly — tests always
    inject their own ``normalize_trail_fn``.
    """

    def _normalize(
        version_trees: dict[str, ClauseTree],
        taxonomy_by_version: dict[str, dict[str, str | None]],
    ) -> NormalizeTrailResult:
        return normalize_trail(version_trees, taxonomy_by_version, taxonomy_ids=taxonomy_ids)

    return _normalize


_LLM_SEGMENTER_CONFIDENCE: float = 0.45
"""Calibrated confidence assigned to every ``_classified_from_taxonomy_by_path``
taxonomy assignment (issue #86).

The LLM segmenter makes its taxonomy_id call in the same untrusted-input pass
as segmentation itself, with no dedicated ``ClassificationJudge`` verifying it
and no per-clause confidence signal in its structured output — a single Opus
pass over counterparty text is not grounds for the certainty a real judge
verdict would carry (a document instructing the model to mislabel a clause
would pass every structural QA gate untouched). This constant is deliberately
below both review thresholds this codebase checks against a classification's
confidence: ``clause_classifier.AMBIGUITY_THRESHOLD`` (0.70 — trips
``ClauseClassification.is_ambiguous``) and the hardcoded 0.5 cutoff in
``aar._build_needs_attention`` (trips the after-action report's "needs
attention" low-confidence flag). Every LLM-segmented, taxonomy-assigned
clause therefore always surfaces for human review — there is no real signal
yet to distinguish a confident LLM call from a shaky one; see this constant's
docstring for the two follow-up options (per-clause LLM confidence,
cross-version ``normalize_trail`` disagreement) that could replace the flat
default with a calibrated one.
"""


def _classified_from_taxonomy_by_path(
    tree: ClauseTree, taxonomy_by_path: dict[str, str | None]
) -> list[ClassifiedClause]:
    """Build ``ClassifiedClause``s directly from an LLM ``taxonomy_by_path`` map.

    Bypasses ``classify_tree`` for LLM-segmented documents: segmentation and
    classification already happened in one LLM pass (see
    :mod:`playbook_engine.llm_segmentation_stage`), so there is no second,
    separate classify judge call on this path. A clause_path missing from
    *taxonomy_by_path* (should not happen — grounding populates one entry per
    node) is treated as unclassified rather than raising, matching
    ``classify_tree``'s own "never silently drop a node" contract.

    An assigned taxonomy_id gets ``basis="llm_segmenter"`` at
    ``_LLM_SEGMENTER_CONFIDENCE`` — NOT ``basis="judge"``/``confidence=1.0``.
    Asserting certainty here would let a single unverified LLM pass over
    untrusted counterparty text masquerade as a verified judge verdict, and
    downstream confidence-based review gating (``classification_confidences``
    feeding ``build_observations`` below, and ``aar._build_needs_attention``)
    would then never flag a misclassified LLM-segmented clause (issue #86).
    ``basis="unclassified"`` (confidence 0.0) is unchanged for ``tid is None``
    — that is the LLM's explicit null for non-clause noise, not a low-
    confidence taxonomy assignment.
    """
    result: list[ClassifiedClause] = []
    for node in tree.all_nodes():
        tid = taxonomy_by_path.get(node.clause_path or "?")
        classification = (
            ClauseClassification(
                taxonomy_id=tid, confidence=_LLM_SEGMENTER_CONFIDENCE, basis="llm_segmenter"
            )
            if tid is not None
            else ClauseClassification(taxonomy_id=None, confidence=0.0, basis="unclassified")
        )
        result.append(ClassifiedClause(node=node, classification=classification))
    return result


def _build_template_observations(
    template_tree: ClauseTree,
    taxonomy: Taxonomy,
    classification_judge: ClassificationJudge,
    *,
    ambiguity_threshold: float = AMBIGUITY_THRESHOLD,
    auto_classify_threshold: float = AUTO_CLASSIFY_THRESHOLD,
) -> list[Observation]:
    """Classify the template tree and emit per-clause Observation objects."""
    classified = classify_tree(
        template_tree,
        taxonomy,
        classification_judge,
        ambiguity_threshold=ambiguity_threshold,
        auto_classify_threshold=auto_classify_threshold,
    )
    obs: list[Observation] = []
    for cc in classified:
        if cc.classification.taxonomy_id is None:
            continue
        # Skip classified-but-empty template clauses (e.g. a heading-only node).
        # Emitting one would build an ``OurStandard`` with empty ``text`` and
        # fail projection ("our_standard.text is empty", validator.py) — a
        # clause with no real template text simply contributes no standard, so
        # the deal clause falls back to emergent/negotiable (issue #182).
        if not (cc.node.text or "").strip():
            continue
        clause_path = cc.node.clause_path or "?"
        obs.append(
            Observation(
                observation_id=f"template/template/{clause_path}",
                taxonomy_id=cc.classification.taxonomy_id,
                text_summary=(cc.node.text or "")[:_TEXT_SUMMARY_MAX],
                full_text=cc.node.text or "",
                citation=ObservationCitation(
                    document_id="template",
                    version="template",
                    clause_path=clause_path,
                    char_span=cc.node.char_span,
                    version_id="template",
                ),
                deviation="none",
                risk_delta={"direction": "neutral", "magnitude": "none"},
                provenance="our_paper",
                outcome="signed",
                confidence=cc.classification.confidence,
                basis=None,  # template observations bypass the deviation classifier
            )
        )
    return obs


def _single_version_clause_diffs(
    classified: list[ClassifiedClause], version_id: str
) -> list[ClauseDiff]:
    """Build one ``kind="unchanged"`` ``ClauseDiff`` per classified clause.

    A single-version document has no negotiation trail to diff against, but
    ``assess_deviations``'s "unchanged" fast path already knows how to compare
    an unchanged clause to the canonical template for its taxonomy_id (see
    ``deviation_classifier.py`` — issue #103): ``text_before == text_after``
    is exactly the "nothing changed within this document" signal that fast
    path expects, so representing each clause this way lets
    ``_assess_deviations_with_standards`` run the identical template-diff
    logic the multi-version path uses, instead of a separate code path.

    Args:
        classified: Classified clauses of the document's single version.
        version_id: The actual version id (normalized-tree file stem) this
            document's only version was ingested as — threaded onto each
            ``ClauseDiff`` as ``clause_version_before``/``clause_version_after``
            so the resulting citation resolves to a real file (issue #108),
            not just the caller's display ordinal.
    """
    diffs: list[ClauseDiff] = []
    for cc in classified:
        clause_path = cc.node.clause_path or "?"
        text = cc.node.text or ""
        diffs.append(
            ClauseDiff(
                taxonomy_id=cc.classification.taxonomy_id,
                clause_path_before=clause_path,
                clause_path_after=clause_path,
                kind="unchanged",
                hunks=(),
                text_before=text,
                text_after=text,
                clause_version_before=version_id,
                clause_version_after=version_id,
                char_span_before=cc.node.char_span,
                char_span_after=cc.node.char_span,
            )
        )
    return diffs


def _observations_from_single_version(
    doc_id: str,
    version: int | str,
    version_id: str,
    provenance: str,
    classified: list[ClassifiedClause],
    has_signed_copy: bool,
    template_std_by_tid: dict[str, str],
    deviation_judge: DeviationJudge,
    our_party_aliases: list[str] | None = None,
) -> list[Observation]:
    """Create observations from a single-version document, diffed against the template.

    Previously every clause of a single-version document was hardcoded
    ``deviation="none"``/``basis="deterministic"`` — a document with no
    negotiation trail was never actually checked against the canonical
    template at all (issue #103). This now builds a synthetic "unchanged"
    ``ClauseDiff`` per clause (see ``_single_version_clause_diffs``) and runs
    it through the same ``_assess_deviations_with_standards`` /
    ``assess_deviations`` template-comparison path the multi-version
    "unchanged across the negotiation trail" case uses: a clause whose
    taxonomy_id has a template clause that it actually differs from is routed
    to *deviation_judge*, not silently recorded as matching. A clause with no
    corresponding template text (``template_std_by_tid`` has no entry for its
    taxonomy_id, or no template is configured at all — an empty string either
    way) still gets ``deviation="none"``/``basis="deterministic"``: there is
    nothing to compare it against, the same honest "no assessment possible"
    contract as before, not a fabricated match.

    ``has_signed_copy`` mirrors ``build_observations``'s same-named parameter:
    it reflects whether ``detect_signed``/``order_versions`` actually
    identified this version as an executed copy, NOT whether it happens to be
    the only version present. When False, ``outcome`` is ``"unsigned"``
    rather than a fabricated ``"signed"`` — a single-version document with no
    detected signature block is an unexecuted draft, not an accepted position
    (issue #83).

    ``version_id`` is the actual normalized-tree file stem for this document's
    only version, threaded onto every resulting citation alongside ``version``
    (the display ordinal) so it resolves to a real file (issue #108).
    """
    diffs = _single_version_clause_diffs(classified, version_id)
    deviation_results = _assess_deviations_with_standards(
        diffs, template_std_by_tid, deviation_judge, document_id=doc_id
    )

    cls_conf_by_path: dict[str, float] = {
        (cc.node.clause_path or "?"): cc.classification.confidence for cc in classified
    }
    classification_confidences = [
        cls_conf_by_path.get(cd.clause_path_after or cd.clause_path_before or "?")
        for cd, _ in deviation_results
    ]

    return build_observations(
        doc_id,
        version,
        provenance,
        deviation_results,
        reversals=[],  # a single-version document has no negotiation trail to reverse
        classification_confidences=classification_confidences,
        has_signed_copy=has_signed_copy,
        our_party_aliases=our_party_aliases,
    )


def _restore_observations(raw_list: list[dict[str, Any]]) -> list[Observation]:
    """Reconstruct Observation objects from read_observations_jsonl() dicts."""
    result: list[Observation] = []
    for raw in raw_list:
        cit = raw["citation"]
        cs_raw = cit.get("char_span")
        attr_raw = raw.get("attribution")
        attribution = (
            HunkEnrichment(
                author=attr_raw["author"],
                date=attr_raw["date"],
                tracked_type=attr_raw["tracked_type"],
            )
            if attr_raw
            else None
        )
        result.append(
            Observation(
                observation_id=raw["observation_id"],
                taxonomy_id=raw["taxonomy_id"],
                text_summary=raw["text_summary"],
                full_text=raw.get("full_text", raw["text_summary"]),
                citation=ObservationCitation(
                    document_id=cit["document_id"],
                    version=cit["version"],
                    clause_path=cit["clause_path"],
                    char_span=tuple(cs_raw) if cs_raw else None,
                    version_id=cit.get("version_id"),
                ),
                deviation=raw["deviation"],
                risk_delta=raw["risk_delta"],
                provenance=raw["provenance"],
                outcome=raw["outcome"],
                confidence=raw.get("confidence"),
                basis=raw.get("basis"),
                attribution=attribution,
                proposed_by=raw.get("proposed_by"),
                observed_at=raw.get("observed_at"),
                counterparty_ref=raw.get("counterparty_ref"),
            )
        )
    return result


def _assess_deviations_with_standards(
    net_diffs: list[Any],
    template_std_by_tid: dict[str, str],
    deviation_judge: DeviationJudge,
    document_id: str | None = None,
) -> list[Any]:
    """Call assess_deviations per taxonomy_id so each group gets the correct our_standard.

    Preserves the original diff order in the returned list.

    ``document_id`` (issue #109) is passed straight through to
    ``assess_deviations`` so every judge batch item carries the owning
    document's id for traceability — see that function's docstring.
    """
    from itertools import groupby

    # Group by taxonomy_id while tracking original indices to preserve order.
    indexed = list(enumerate(net_diffs))
    result: list[Any] = [None] * len(net_diffs)

    def _tid_key(item: tuple[int, Any]) -> str:
        tid = item[1].taxonomy_id
        return "" if tid is None else tid  # None sorts with empty-string group

    def _tid(item: tuple[int, Any]) -> str | None:
        return item[1].taxonomy_id  # type: ignore[no-any-return]

    for tid, group_iter in groupby(sorted(indexed, key=_tid_key), key=_tid):
        group_items = list(group_iter)
        indices = [i for i, _ in group_items]
        diffs = [d for _, d in group_items]
        our_std = template_std_by_tid.get(tid or "", "")
        assessed = assess_deviations(diffs, our_std, deviation_judge, document_id=document_id)
        for orig_idx, pair in zip(indices, assessed, strict=True):
            result[orig_idx] = pair

    return [r for r in result if r is not None]


def _attribution_for_diff(
    clause_diff: ClauseDiff,
    tracked_changes: TrackedChanges | None,
) -> HunkEnrichment | None:
    """Best-effort tracked-changes attribution for one net ``ClauseDiff`` (issue #88).

    ``tracked_changes`` is the SIGNED/last version's own DOCX side-channel
    (``tracked_by_vid[signed_vid]`` in ``_compute_doc_result``) — real-world
    redlining tracks each author's edits against the file they received, so
    the executed/final DOCX's ``w:ins``/``w:del`` is the closest available
    signal for "who proposed this" even when ``net_diffs`` (first → signed)
    spans more than one negotiation round. This is an approximation for
    documents with more than two versions, not a full per-round attribution
    history — that would require enriching each *consecutive* diff instead
    of the net diff, which observation_builder does not model today.

    Returns ``None`` when there is no side-channel at all (PDF/RTF, clean
    DOCX, or an LLM-segmented version — see ``_compute_doc_result``'s
    ``tracked_by_vid`` docstring), when the diff has no hunks (added/removed/
    unchanged clauses), or when no hunk matched a tracked change closely
    enough (see ``tracked_changes_overlay._MATCH_THRESHOLD``).
    """
    if tracked_changes is None or not clause_diff.hunks:
        return None
    enriched = enrich_clause_diff(clause_diff, tracked_changes)
    return next((eh.enrichment for eh in enriched if eh.enrichment is not None), None)


def _atomic_json_write(data: Any, path: Path) -> None:
    """Atomically write *data* as JSON to *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def _alias_version_field(value: Any, known_entities: list[str], registry: EntityRegistry) -> Any:
    """Alias a citation/manifest ``version``/``version_id`` field (issue #182).

    These are staged filename stems that embed the counterparty name (e.g.
    "01__… Oglethorpe University 6.14.23"), so run them through the whole-word
    text pseudonymizer. A plain ordinal (``int``) carries no name and is
    returned unchanged.
    """
    if isinstance(value, str):
        return pseudonymize_text(value, known_entities, registry)
    return value


def _pseudonymize_observation_id(
    observation_id: str, known_entities: list[str], registry: EntityRegistry
) -> str:
    """Alias the document-id segment of an ``observation_id`` (issue #182).

    ``observation_id`` is ``<document_id>/<version>/<clause_path>`` (the
    clause_path may itself contain ``.`` and a ``#<count>`` suffix). Only the
    leading document-id segment can carry a raw counterparty name, so split on
    the first ``/``, pseudonymize that segment with the same token-match rule
    used for ``citation.document_id``, and rejoin — leaving version/clause
    structure intact.
    """
    doc_part, sep, rest = observation_id.partition("/")
    aliased = pseudonymize_document_id(doc_part, known_entities, registry)
    return aliased + sep + rest


def _pseudonymize_observations(
    observations: list[Observation], known_entities: list[str], registry: EntityRegistry
) -> list[Observation]:
    """Return *observations* with clause text and every document id aliased.

    Rewrites ``text_summary``, ``full_text``, ``citation.document_id``, and the
    document-id segment of ``observation_id`` for every known entity name
    (issues #153, #182) — ``Observation``/``ObservationCitation`` are frozen
    dataclasses, so a fresh copy is built per row via ``dataclasses.replace``
    rather than mutated in place. Pseudonymizing ``observation_id`` here (not
    just the citation) keeps the id free of raw counterparty names and keeps it
    consistent with the aliased ``citation.document_id``.
    """
    out: list[Observation] = []
    for obs in observations:
        new_citation = ObservationCitation(
            document_id=pseudonymize_document_id(
                obs.citation.document_id, known_entities, registry
            ),
            version=_alias_version_field(obs.citation.version, known_entities, registry),
            clause_path=obs.citation.clause_path,
            char_span=obs.citation.char_span,
            version_id=_alias_version_field(obs.citation.version_id, known_entities, registry),
        )
        out.append(
            dataclasses.replace(
                obs,
                observation_id=_pseudonymize_observation_id(
                    obs.observation_id, known_entities, registry
                ),
                text_summary=pseudonymize_text(obs.text_summary, known_entities, registry),
                full_text=pseudonymize_text(obs.full_text, known_entities, registry),
                citation=new_citation,
            )
        )
    return out


def _pseudonymize_trail(
    trail: dict[str, Any], known_entities: list[str], registry: EntityRegistry
) -> dict[str, Any]:
    """Return a copy of *trail* with its ``document_id`` aliased (issue #182).

    Mirrors ``_pseudonymize_observations`` for the trail store so trail
    ``document_id`` matches the aliased ``citation.document_id`` that
    ``inspect`` joins on, and so the trail carries no raw counterparty name.
    """
    if not trail.get("document_id"):
        return dict(trail)
    new = dict(trail)
    new["document_id"] = pseudonymize_document_id(trail["document_id"], known_entities, registry)
    return new


def _attach_counterparty_refs(
    observations: list[Observation], known_entities: list[str], registry: EntityRegistry
) -> list[Observation]:
    """Set ``counterparty_ref`` on observations whose deal has exactly one
    known-entity match (issue #177, OPF §3.5.3).

    Must run on RAW (pre-pseudonymization) observations — the match is
    against real entity names in the document id / clause text, which the
    pseudonymization pass is about to erase. The attached value carries only
    the born-safe registry alias, never the raw name. A deal matching zero
    or multiple known entities gets no ref — ambiguity is omitted, not
    guessed.
    """
    texts_by_doc: dict[str, list[str]] = {}
    for obs in observations:
        texts_by_doc.setdefault(obs.citation.document_id, []).append(obs.full_text)

    # Compile once per entity, not per (doc, entity). \b cannot terminate a
    # name ending in a non-word char ("Acme Corp." — no word char ever
    # follows the "."), so use edge lookarounds instead: they assert
    # no-word-char-adjacent, which holds at punctuation and string edges.
    patterns = [
        (name, re.compile(r"(?<!\w)" + re.escape(name) + r"(?!\w)", re.IGNORECASE))
        for name in known_entities
        if name
    ]

    ref_by_doc: dict[str, dict[str, str]] = {}
    for doc_id, texts in texts_by_doc.items():
        matched: list[str] = []
        for name, pattern in patterns:
            slug_hit = _entity_slug_in_document_id(doc_id, name)
            if slug_hit or any(pattern.search(t) for t in texts):
                matched.append(name)
                if len(matched) > 1:
                    break  # ambiguous — no ref will be attached
        if len(matched) == 1:
            ref_by_doc[doc_id] = {"alias": registry.alias_for(matched[0])}

    return [
        dataclasses.replace(obs, counterparty_ref=ref_by_doc[obs.citation.document_id])
        if obs.citation.document_id in ref_by_doc and obs.counterparty_ref is None
        else obs
        for obs in observations
    ]


def _entity_slug_in_document_id(document_id: str, entity_name: str) -> bool:
    """Whether *entity_name*'s slug-token sequence appears in *document_id*
    (same normalized-token match ``pseudonymize_document_id`` performs)."""
    doc_tokens = entity_slug(document_id).split("-")
    name_tokens = entity_slug(entity_name).split("-")
    n = len(name_tokens)
    if n == 0 or not name_tokens[0]:
        return False
    return any(doc_tokens[i : i + n] == name_tokens for i in range(len(doc_tokens) - n + 1))


def _pseudonymize_round_moves(
    moves: list[RoundMove], known_entities: list[str], registry: EntityRegistry
) -> list[RoundMove]:
    """Alias raw entity names out of round moves (issue #177) — same born-safe
    pass ``_pseudonymize_observations`` applies, covering ``document_id``,
    the citation, and ``change_summary`` (which quotes clause text)."""
    out: list[RoundMove] = []
    for move in moves:
        new_citation = ObservationCitation(
            document_id=pseudonymize_document_id(
                move.citation.document_id, known_entities, registry
            ),
            version=_alias_version_field(move.citation.version, known_entities, registry),
            clause_path=move.citation.clause_path,
            char_span=move.citation.char_span,
            version_id=_alias_version_field(move.citation.version_id, known_entities, registry),
        )
        out.append(
            dataclasses.replace(
                move,
                document_id=pseudonymize_document_id(move.document_id, known_entities, registry),
                change_summary=pseudonymize_text(move.change_summary, known_entities, registry),
                citation=new_citation,
            )
        )
    return out


def _pseudonymize_corpus_documents(
    corpus_documents: list[dict[str, Any]], known_entities: list[str], registry: EntityRegistry
) -> list[dict[str, Any]]:
    """Return *corpus_documents* with each entry's ``document_id`` aliased (issue #153).

    ``corpus_documents`` (``corpus_manifest.json``) is embedded verbatim into
    ``playbook.opf.json``'s ``documents`` field (see ``playbook_assembler``),
    so its ``document_id`` must carry the same alias as the matching
    observations' ``citation.document_id`` for the compiled OPF to be
    consistent, not just the observation store.
    """
    out = []
    for doc in corpus_documents:
        new_doc = dict(doc)
        if "document_id" in new_doc:
            new_doc["document_id"] = pseudonymize_document_id(
                new_doc["document_id"], known_entities, registry
            )
        # version_ingest / signed_version embed the staged filename stem, which
        # carries the counterparty name (issue #182) — alias those too so the
        # manifest embedded in playbook.opf.json holds no raw names.
        if isinstance(new_doc.get("version_ingest"), list):
            new_doc["version_ingest"] = [
                {**vi, "version": _alias_version_field(vi.get("version"), known_entities, registry)}
                if isinstance(vi, dict)
                else vi
                for vi in new_doc["version_ingest"]
            ]
        if isinstance(new_doc.get("signed_version"), str):
            new_doc["signed_version"] = _alias_version_field(
                new_doc["signed_version"], known_entities, registry
            )
        out.append(new_doc)
    return out


def _our_aliases_match_any_tree(trees: Iterable[ClauseTree], aliases: list[str]) -> bool:
    """True when any configured our_party alias appears anywhere in any tree.

    Scans every node's heading AND body text of every version tree, so
    recitals/preambles and signature blocks — the places party names actually
    live — count. Whole-word, case-insensitive, same match shape as the
    corpus-level warning that consumes this (issue #201). Vacuously False for
    an empty alias list (the caller's warning is gated on aliases being
    configured, so the value is never read in that case).
    """
    patterns = [
        re.compile(r"(?<!\w)" + re.escape(a) + r"(?!\w)", re.IGNORECASE) for a in aliases if a
    ]
    if not patterns:
        return False
    for tree in trees:
        for node in tree.all_nodes():
            for surface in (node.heading, node.text):
                if surface and any(p.search(surface) for p in patterns):
                    return True
    return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _compute_doc_result(
    doc_id: str,
    doc_dir: Path,
    version_files: list[Path],
    out_dir: Path,
    config: EngineConfig,
    taxonomy: Taxonomy,
    template_tree: ClauseTree | None,
    template_std_by_tid: dict[str, str],
    _scope_judge: ScopeJudge,
    _cls_judge: ClassificationJudge,
    _dev_judge: DeviationJudge,
    alignment_judge: AlignmentJudge | None,
    trail_judge: TrailJudge | None,
    progress: Callable[[str], None],
    signed_judge: SignedJudge | None = None,
    provenance_judge: ProvenanceJudge | None = None,
    use_llm_segmentation: bool = False,
    llm_segment_fn: SegmentFn | None = None,
    normalize_trail_across_versions: bool = False,
    normalize_trail_fn: NormalizeTrailFn | None = None,
    batch_seg_nodes: dict[str, list[SegNode]] | None = None,
    batch_extractions: dict[str, _BatchExtraction] | None = None,
    segmentation_cache: SegmentationVerdictCache | None = None,
    extraction_cache: ExtractionCache | None = None,
) -> dict[str, Any] | None:
    """Compute L1–L4 for a single document; return a cacheable result dict or None on skip.

    Returns a dict with keys:
      - ``corpus_doc``:    corpus_documents entry (JSON-serialisable). Includes
                           ``versions`` / ``versions_mined`` (versions that
                           actually ingested, NOT files found — see
                           ``versions_found``) and ``version_ingest`` (a
                           per-version ``{version, status, error, extractor}``
                           record for every version file found, "ok" or
                           "failed" — issue #89. ``extractor`` is the file
                           suffix on the deterministic path, or
                           ``"docling"``/``"legacy"`` on the LLM-segmentation
                           path (issue #129 — see ``extraction.detect_extractor``).
      - ``observations``:  list of serialised Observation dicts.
      - ``trail``:         trail dict, or None for out-of-scope documents.
      - ``scope_decision``: scope decision fields (for replaying into ScopeLog).

    Returns ``None`` if the document has no processable versions.

    Note: does NOT mutate any ``ScopeLog``; the caller is responsible for
    replaying ``scope_decision`` into the active log (both on cache hit and miss).

    When ``use_llm_segmentation`` is True, L1 segments each version via
    :func:`~playbook_engine.llm_segmentation_stage.segment_to_tree` instead of
    ``segment(_ingest_file(...))`` — the LLM classifies each clause in the
    same pass, so L3 skips ``classify_tree`` for this document's versions and
    uses the LLM's per-clause taxonomy assignments directly (see
    ``_classified_from_taxonomy_by_path``). This path never falls back to the
    deterministic segmenter: a ``SegmentationQAError`` propagates uncaught,
    same as any other per-version ingest exception below — the document is
    flagged for review, not silently degraded.

    When ``normalize_trail_across_versions`` is also True (only meaningful
    together with ``use_llm_segmentation=True`` — each version's taxonomy_id
    otherwise already comes from a single shared judge, not independent LLM
    calls per version), :func:`~playbook_engine.llm_segmenter_batch.normalize_trail`
    runs once per agreement after every version has been segmented, replacing
    ``llm_taxonomy_by_path`` with its normalized labels before L3 classification
    reads them. A ``NormalizeTrailError`` propagates uncaught — same fail-loud
    contract as a segmentation QA failure, no silent fallback to the
    un-normalized per-version labels.

    ``batch_seg_nodes``/``batch_extractions`` (only meaningful together with
    ``use_llm_segmentation=True``) carry this document's already-segmented
    ``SegNode`` output from a prior corpus-wide
    :func:`~playbook_engine.llm_segmenter_batch.segment_documents_batch` call
    (see ``mine_corpus``'s ``use_batch_segmentation``) — when either is given
    for a version, L1 grounds those nodes via :func:`_ground_batch_result`
    instead of calling ``_llm_segment_file`` (no per-document LLM call here at
    all). A version absent from ``batch_seg_nodes`` falls back to the normal
    per-document path for that version only (e.g. a version whose pre-pass
    extraction failed never entered the batch — see ``_collect_batch_items``).
    A ``SegmentationQAError`` from grounding a batched result propagates
    uncaught exactly like the synchronous LLM path — no repair loop, no
    deterministic-segmenter fallback (see ``_ground_batch_result``).

    ``segmentation_cache`` (only meaningful together with
    ``use_llm_segmentation=True``) is forwarded to ``_llm_segment_file`` for
    any version NOT already resolved via ``batch_seg_nodes`` — i.e. it also
    covers the per-document synchronous LLM path, not just the batch
    pre-pass (issue #91). A version's content-hash cache hit skips its LLM
    call entirely, same judge-once contract as the batch path.

    ``extraction_cache`` (only meaningful together with
    ``use_llm_segmentation=True``, and only for versions NOT already resolved
    via ``batch_seg_nodes`` — those go through ``batch_extractions`` instead,
    populated by ``_collect_batch_items``, which takes its own
    ``extraction_cache``) is forwarded to ``_llm_segment_file`` so a repeat
    run over unchanged source content skips extraction (docling/pdfplumber/
    python-docx/pandoc) entirely — independent of ``segmentation_cache``,
    which only covers the LLM segmentation call (issue #132).

    A version whose ingest yields an EMPTY ``ClauseTree`` from a non-empty
    source file (e.g. a scanned/image PDF on the deterministic path, where no
    OCR is wired) is treated as an ingest failure — recorded via the same
    per-version warning as any other extraction exception, never added to
    ``version_trees``. This prevents an unreadable version from silently
    becoming ``first_tree`` and being misclassified by the scope gate as
    ``deterministic_empty`` (issue #82).
    """
    # L1: Ingest + segment each version
    version_trees: dict[str, ClauseTree] = {}
    llm_taxonomy_by_path: dict[str, dict[str, str | None]] = {}
    # Populated only on the deterministic DOCX path — LLM-segmented versions
    # have no tracked-changes capture yet (extraction.py has no w:ins/w:del
    # handling), so those versions are simply absent here and
    # tracked_by_vid.get(vid) below returns None (issue #88).
    tracked_by_vid: dict[str, TrackedChanges | None] = {}
    # Per-version ingest status (issue #89): every version FILE FOUND gets an
    # entry here, "ok" or "failed" — this is what lets corpus_manifest.json
    # distinguish "versions found" from "versions actually mined" instead of
    # letting a failed version disappear after nothing but a scrolled-past
    # progress-line WARNING (see corpus_doc["version_ingest"] below).
    version_ingest: dict[str, dict[str, Any]] = {}
    # Per-version content addresses (issue #185, OPF §4): sha256 of the
    # SOURCE file bytes, computed here where the file is being read anyway,
    # so a consumer holding the corpus can verify it has the cited document.
    sha256_by_vid: dict[str, str] = {}
    media_type_by_vid: dict[str, str] = {}
    taxonomy_ids = [e.id for e in taxonomy.classifier_entries()]
    for vf in version_files:
        vid = vf.stem
        sha256_by_vid[vid] = file_sha256(vf)
        media_type_by_vid[vid] = _MEDIA_TYPES.get(vf.suffix.lower(), "application/octet-stream")
        # Per-version extractor recorded in version_ingest/corpus_manifest.json
        # (issue #129): the deterministic path always uses the file suffix
        # (unchanged). The LLM-segmentation path (sync, batch pre-pass, or a
        # segmentation-cache hit — all funnel through extraction.extract_blocks)
        # previously collapsed this to a flat "llm", which hid whether docling
        # or a legacy pdfplumber/python-docx/pandoc adapter actually ran behind
        # a suppressed logging.info line (extraction.py:135-139). Computed via
        # detect_extractor (a pure PATH check) up front so it is known even if
        # extraction/ingest subsequently fails for this version.
        extractor = detect_extractor(vf) if use_llm_segmentation else vf.suffix.lower().lstrip(".")
        try:
            if use_llm_segmentation and batch_seg_nodes is not None and vid in batch_seg_nodes:
                extraction = (batch_extractions or {})[vid]
                tree, tax_by_path = _ground_batch_result(
                    doc_id, vid, extraction, batch_seg_nodes[vid], taxonomy_ids
                )
                llm_taxonomy_by_path[vid] = tax_by_path
            elif use_llm_segmentation:
                tree, tax_by_path = _llm_segment_file(
                    vf,
                    doc_id,
                    vid,
                    taxonomy_ids,
                    llm_segment_fn,
                    segmentation_cache,
                    model=config.segmentation.model,
                    extraction_cache=extraction_cache,
                )
                llm_taxonomy_by_path[vid] = tax_by_path
            else:
                raw_tree, tracked = _ingest_file_tracked(vf, doc_id, vid)
                tree = segment(raw_tree)
                tracked_by_vid[vid] = tracked

            if not list(tree.all_nodes()) and vf.stat().st_size > 0:
                # An empty ClauseTree from a non-empty source file is an
                # ingest FAILURE, not a success — e.g. a scanned/image PDF on
                # the deterministic path (pdf_ingester.NullOCRAdapter; no OCR
                # is wired) silently yields zero clauses with no error (issue
                # #82). Treating that as success would let this version enter
                # version_trees and become `first_tree` below, which the
                # scope gate misreads as `deterministic_empty` ("this
                # agreement is out of scope") — one unreadable scan would
                # knock the whole negotiation trail out of the corpus. Raise
                # here so it is recorded per-version exactly like any other
                # extraction failure (the `except Exception` clause below)
                # and never reaches the scope gate as the representative
                # version.
                raise ValueError(
                    "ingest produced an empty clause tree from a non-empty "
                    "source file — treating as an extraction failure"
                )

            tree.write(out_dir / "normalized" / doc_id / f"{vid}.clauses.json")
            version_trees[vid] = tree
            version_ingest[vid] = {"status": "ok", "error": None, "extractor": extractor}
        except SegmentationQAError:
            # Fail loud, by design: a QA-gate failure on the LLM path must
            # never be swallowed into a per-file warning + skipped version —
            # that would silently drop a version rather than flag the
            # document for review (see llm_segmentation_stage.segment_to_tree
            # and segmentation_qa.segment_verify_repair). It propagates out of
            # this per-document function so the corpus loop can quarantine THIS
            # document (recorded in quarantine.json) without aborting the whole
            # run — see mine_corpus's ``quarantined`` handling. Every other
            # exception below (extraction/ingest failure, malformed source)
            # keeps the pre-existing "skip this one version file" behavior.
            raise
        except Exception as exc:  # noqa: BLE001
            progress(f"    WARNING: {vf.name}: {exc}")
            version_ingest[vid] = {"status": "failed", "error": str(exc), "extractor": extractor}

    if not version_trees:
        progress(f"  {doc_id}: all ingests failed — skipping")
        return None

    # Alias sanity signal (issue #201): scan EVERY version's full tree text —
    # headings and bodies, so recitals/preambles and signature blocks count —
    # for any configured our_party alias. Computed here (not corpus-level)
    # because this is the only scope where all versions' text is in memory;
    # the corpus-level check used to scan only mined observation texts (head-
    # version clause bodies), which misses the exact places party names live
    # and false-alarmed on corpora whose aliases appear only in a recital or
    # a non-head version. Cached with the doc result; safe because the stage-
    # cache config fingerprint already includes provenance_aliases, so an
    # alias change re-runs this.
    our_alias_matched = _our_aliases_match_any_tree(
        version_trees.values(), config.provenance.our_party_aliases
    )

    # L1c: Cross-version taxonomy normalization (opt-in, LLM-segmented only).
    # Runs after every version is segmented and before L3 classification reads
    # llm_taxonomy_by_path — see the docstring above for the fail-loud contract.
    if normalize_trail_across_versions and use_llm_segmentation and len(version_trees) > 1:
        _normalize_fn: NormalizeTrailFn = normalize_trail_fn or _default_normalize_trail_fn(
            taxonomy_ids
        )
        normalized = _normalize_fn(version_trees, llm_taxonomy_by_path)
        llm_taxonomy_by_path = normalized.taxonomy_by_version

    # L1b: Scope gate (first ingested version) — result stored in cache, NOT in scope_log.
    first_tree = next(iter(version_trees.values()))
    decision = scope_gate(first_tree, config.agreement_type, _scope_judge)

    scope_decision_dict: dict[str, Any] = {
        "in_scope": decision.in_scope,
        "scope_rationale": decision.scope_rationale,
        "scope_confidence": decision.scope_confidence,
        "basis": decision.basis,
    }

    # version_ingest (issue #89): one entry per version FILE FOUND, in discovery
    # order, so a failed extraction/segmentation is a durable manifest record —
    # not just a progress-line WARNING that a cache hit wouldn't even re-print.
    version_ingest_list = [
        {
            "version": vf.stem,
            **version_ingest.get(
                vf.stem, {"status": "unknown", "error": "not attempted", "extractor": None}
            ),
        }
        for vf in version_files
    ]

    corpus_doc: dict[str, Any] = {
        "document_id": doc_id,
        "provenance": "counterparty_paper",  # refined below for in-scope docs
        "in_scope": decision.in_scope,
        # "versions" is versions MINED (not files found) — see versions_found
        # below. A version whose ingest failed must never be counted as if it
        # had been read (that was the bug: corpus_doc["versions"] used to be
        # len(version_files), overstating coverage for a document with any
        # failed version).
        "versions": len(version_trees),
        "versions_mined": len(version_trees),
        "versions_found": len(version_files),
        "version_ingest": version_ingest_list,
    }

    if not decision.in_scope:
        corpus_doc["scope_rationale"] = decision.scope_rationale
        # Content addresses (issue #185) for out-of-scope documents too:
        # snapshot.manifest_hash names the WHOLE corpus state the playbook
        # was compiled from, and out-of-scope docs are part of that state
        # (they are retained in corpus.documents by §3.8). No negotiation
        # ordering exists for them, so ordinals follow discovery order —
        # citations never target out-of-scope docs, so the ordinal is
        # identity bookkeeping only.
        corpus_doc["version_files"] = [
            {
                "version": i + 1,
                "sha256": sha256_by_vid[vf.stem],
                "media_type": media_type_by_vid[vf.stem],
            }
            for i, vf in enumerate(version_files)
            if vf.stem in sha256_by_vid
        ]
        progress(f"    out-of-scope: {decision.scope_rationale[:70]}")
        return {
            "corpus_doc": corpus_doc,
            "observations": [],
            "trail": None,
            "scope_decision": scope_decision_dict,
            "our_alias_matched": our_alias_matched,
        }

    # L2: Signed detection, version ordering, provenance
    signed_status_by_vid: dict[str, Any] = {}
    version_inputs = []
    for vid, tree in version_trees.items():
        ss = detect_signed(tree, signed_judge=signed_judge)
        signed_status_by_vid[vid] = ss
        version_inputs.append(VersionInput(version_id=vid, tree=tree, signed=ss))
    # hints.yaml is optional (Hints.load returns empty Hints for a missing
    # file) but a malformed one raises HintsError, which propagates uncaught
    # out of this function exactly like SegmentationQAError above — the
    # corpus loop (mine_corpus) quarantines just this document rather than
    # silently discarding the lawyer's correction or aborting the whole run.
    hints_path = doc_dir / "hints.yaml"
    hints = Hints.load(hints_path) if hints_path.exists() else None

    if hints is not None:
        known_vids = {vi.version_id for vi in version_inputs}
        if hints.signed_version is not None and hints.signed_version not in known_vids:
            progress(
                f"    WARNING: {doc_id}: hints.yaml signed_version "
                f"{hints.signed_version!r} matches no discovered version "
                f"(known: {sorted(known_vids)}) — hint ignored"
            )
        if hints.order:
            unmatched = [vid for vid in hints.order if vid not in known_vids]
            if unmatched:
                progress(
                    f"    WARNING: {doc_id}: hints.yaml order entries "
                    f"{unmatched!r} match no discovered version "
                    f"(known: {sorted(known_vids)}) — those entries are ignored"
                )

    # Apply signed_version hint: override the SignedStatus for the hinted version
    # so order_versions anchors it as the signed copy, regardless of the heuristic.
    if hints is not None and hints.signed_version is not None:
        hint_svid = hints.signed_version
        for vi in version_inputs:
            if vi.version_id == hint_svid:
                # Replace with a definitive signed=True status; hint wins.
                vi.signed = SignedStatus(signed=True, basis="hint", confidence=1.0)
                signed_status_by_vid[hint_svid] = vi.signed
            elif vi.signed.signed:
                # Demote any other version that the heuristic picked as signed.
                vi.signed = SignedStatus(signed=False, basis="hint", confidence=1.0)
                signed_status_by_vid[vi.version_id] = vi.signed

    version_order = order_versions(version_inputs, hints, trail_judge=trail_judge)

    earliest_vid = version_order.ordered_ids[0] if version_order.ordered_ids else None
    prov_tree = version_trees[earliest_vid] if earliest_vid else first_tree
    prov_result = detect_provenance(
        prov_tree,
        config.provenance,
        template_tree=template_tree,
        provenance_judge=provenance_judge,
        agreement_type=config.agreement_type.name,
    )

    # Apply provenance hint: override the detected provenance unconditionally.
    if hints is not None and hints.provenance is not None:
        prov_result = ProvenanceResult(
            provenance=hints.provenance,
            confidence=1.0,
            basis="hint",
        )

    provenance = "counterparty_paper" if prov_result.is_ambiguous else prov_result.provenance

    # has_signed_copy: whether order_versions actually anchored a signed
    # version, not whether one was assumed for chain-ordering purposes below.
    # signed_copy_confidence must NEVER be computed from a fallback version —
    # reporting a signed=False determination's confidence (e.g. 0.85 for
    # basis="no_signature_section") as if it were confidence in a signed copy
    # is exactly the fabrication issue #83 closes. When no version was
    # detected as signed, confidence is None, full stop.
    has_signed_copy = version_order.signed_id is not None
    signed_copy_confidence: float | None = None
    if version_order.signed_id is not None:
        signed_copy_status = signed_status_by_vid.get(version_order.signed_id)
        signed_copy_confidence = (
            signed_copy_status.confidence if signed_copy_status is not None else None
        )

    trail: dict[str, Any] = {
        "document_id": doc_id,
        "provenance": provenance,
        "provenance_confidence": prov_result.confidence,
        "provenance_is_ambiguous": prov_result.is_ambiguous,
        "signed_copy_confidence": signed_copy_confidence,
        # Populated below (multi-version documents only) from detect_reversals();
        # a single-version document has no negotiation trail to reverse, so it
        # keeps this default empty list (issue #106 — previously this key was
        # never written at all, so aar.py's backbone reversal count was
        # permanently 0 regardless of what detect_reversals actually found).
        "reversals": [],
        **version_order.to_dict(),
    }

    # L3: Classify each version. LLM-segmented versions already carry their
    # taxonomy_id from the L1 LLM pass — bypass classify_tree entirely for
    # those (no separate classify judge for LLM-segmented docs).
    ordered_ids = list(version_order.ordered_ids) or list(version_trees.keys())
    classified_by_version: dict[str, list[ClassifiedClause]] = {}
    for vid in ordered_ids:
        if vid in llm_taxonomy_by_path:
            classified_by_version[vid] = _classified_from_taxonomy_by_path(
                version_trees[vid], llm_taxonomy_by_path[vid]
            )
        else:
            classified_by_version[vid] = classify_tree(
                version_trees[vid],
                taxonomy,
                _cls_judge,
                ambiguity_threshold=config.classification.ambiguity_threshold,
                auto_classify_threshold=config.classification.auto_classify_threshold,
            )

    # L4: Diff + reversals + deviations → observations
    #
    # signed_vid is a positional anchor only (which tree to diff the chain
    # against / which ordinal to cite as "the last version"), NOT a claim that
    # this version was executed — that claim is has_signed_copy, computed
    # above from version_order.signed_id alone and threaded into
    # _observations_from_single_version/build_observations below so outcome
    # is never fabricated as "signed" when no signed copy was detected.
    signed_vid = version_order.signed_id or ordered_ids[-1]
    signed_ordinal = (
        ordered_ids.index(signed_vid) + 1 if signed_vid in ordered_ids else len(ordered_ids)
    )

    round_moves: list[RoundMove] = []
    if len(ordered_ids) < 2:
        doc_obs = _observations_from_single_version(
            doc_id,
            signed_ordinal,
            signed_vid,
            provenance,
            classified_by_version[signed_vid],
            has_signed_copy=has_signed_copy,
            template_std_by_tid=template_std_by_tid,
            deviation_judge=_dev_judge,
            our_party_aliases=config.provenance.our_party_aliases,
        )
    else:
        classified_versions = [(vid, classified_by_version[vid]) for vid in ordered_ids]
        alignments = align_versions(classified_versions, alignment_judge=alignment_judge)
        doc_diff = diff_aligned(alignments, ordered_ids)
        reversals = detect_reversals(doc_diff)
        # Negotiation dynamics (issue #177): surface the per-round diffs as
        # RoundMove records instead of discarding them — these become each
        # ClausePosition's negotiation_trail at L5.
        round_moves = build_round_moves(
            doc_id,
            doc_diff,
            tracked_by_vid=tracked_by_vid,
            our_party_aliases=config.provenance.our_party_aliases,
        )
        # Issue #106: record detected reversals on the trail itself — this is
        # what aar.py's backbone health section counts (previously read
        # trail["reversals"] via a default-empty .get() that this key never
        # populated, so the AAR always reported zero reversals even when
        # detect_reversals found some).
        trail["reversals"] = [r.to_dict() for r in reversals]

        net_diffs = list(doc_diff.net.diffs)
        deviation_results = _assess_deviations_with_standards(
            net_diffs, template_std_by_tid, _dev_judge, document_id=doc_id
        )

        signed_classified = classified_by_version[signed_vid]
        cls_conf_by_path: dict[str, float] = {
            (cc.node.clause_path or "?"): cc.classification.confidence for cc in signed_classified
        }
        classification_confidences = [
            cls_conf_by_path.get(cd.clause_path_after or cd.clause_path_before or "?")
            for cd, _ in deviation_results
        ]

        # Tracked-changes attribution (issue #88): best-effort, from the
        # signed/last version's own DOCX side-channel — see
        # _attribution_for_diff for why that version is the right source
        # even though net_diffs can span more than one negotiation round.
        signed_tracked = tracked_by_vid.get(signed_vid)
        attributions = [_attribution_for_diff(cd, signed_tracked) for cd, _ in deviation_results]

        doc_obs = build_observations(
            doc_id,
            signed_ordinal,
            provenance,
            deviation_results,
            reversals,
            classification_confidences,
            has_signed_copy=has_signed_copy,
            attributions=attributions,
            our_party_aliases=config.provenance.our_party_aliases,
        )

    corpus_doc["provenance"] = provenance
    corpus_doc["provenance_confidence"] = prov_result.confidence
    corpus_doc["provenance_is_ambiguous"] = prov_result.is_ambiguous
    # null when no signed copy was detected (issue #202): signed_ordinal is a
    # positional fallback (last version) for diffing, and publishing it as
    # signed_version made the projected document claim an execution the trail
    # (signed_version: null), report ("0/N signed copies"), and every
    # observation (outcome="unsigned") all deny. Schema allows null.
    corpus_doc["signed_version"] = signed_ordinal if has_signed_copy else None
    corpus_doc["version_order_basis"] = version_order.basis
    # version_files (issue #185): one entry per MINED version, keyed by the
    # same negotiation ordinal citations use (ordered_ids position, 1-based).
    # Failed-ingest versions have no ordinal and are visible in
    # version_ingest instead.
    corpus_doc["version_files"] = [
        {
            "version": i + 1,
            "sha256": sha256_by_vid[vid],
            "media_type": media_type_by_vid[vid],
        }
        for i, vid in enumerate(ordered_ids)
        if vid in sha256_by_vid
    ]

    # Serialise observations for caching. Observation.to_dict IS the cache
    # shape — a second hand-maintained field list here is how a new field
    # silently vanishes from cached runs only (review finding, 2026-07-13).
    obs_dicts: list[dict[str, Any]] = [obs.to_dict() for obs in doc_obs]

    return {
        "corpus_doc": corpus_doc,
        "observations": obs_dicts,
        "trail": trail,
        # Serialized RoundMoves (issue #177); .get()-read by mine_corpus so
        # cached results from before this feature simply contribute none.
        "round_moves": [rm.to_dict() for rm in round_moves],
        "scope_decision": scope_decision_dict,
        # Per-doc alias sanity signal (issue #201); .get()-read by mine_corpus
        # so cached results from before this field fall back to the coarser
        # observation-text scan there.
        "our_alias_matched": our_alias_matched,
    }


def mine_corpus(
    corpus_dir: Path,
    config: EngineConfig,
    taxonomy: Taxonomy,
    out_dir: Path,
    *,
    scope_judge: ScopeJudge | None = None,
    classification_judge: ClassificationJudge | None = None,
    deviation_judge: DeviationJudge | None = None,
    alignment_judge: AlignmentJudge | None = None,
    trail_judge: TrailJudge | None = None,
    signed_judge: SignedJudge | None = None,
    provenance_judge: ProvenanceJudge | None = None,
    no_cache: bool = False,
    use_llm_segmentation: bool = False,
    llm_segment_fn: SegmentFn | None = None,
    normalize_trail_across_versions: bool = False,
    normalize_trail_fn: NormalizeTrailFn | None = None,
    use_batch_segmentation: bool = False,
    segmentation_cache: SegmentationVerdictCache | None = None,
    segment_documents_batch_fn: Callable[..., dict[str, list[SegNode]]] | None = None,
    extraction_cache: ExtractionCache | None = None,
    entity_registry_path: Path | None = None,
    progress: Callable[[str], None] = lambda _: None,
) -> None:
    """Run L1–L4 (ingest → scope → classify → diff/deviation) and write the observation store.

    Writes to ``{out_dir}/``:

    - ``observations.jsonl``   — per-clause observations (the store contract).
    - ``corpus_manifest.json`` — per-document metadata.
    - ``scope.json``           — scope-gate decisions.
    - ``trail/{doc_id}.json``  — version-order and provenance signals per document.
    - ``normalized/``          — segmented clause trees per version.
    - ``.cache/``              — content-addressed stage cache (key → artifact).

    Does **not** write ``playbook.opf.json``.  Run :func:`project_playbook` afterwards
    (or use :func:`compile_corpus` for the combined end-to-end flow).

    Args:
        corpus_dir:           Root corpus directory (one subdirectory per agreement).
        config:               Engine configuration (agreement type, baseline, taxonomy).
        taxonomy:             Loaded taxonomy object.
        out_dir:              Output directory for intermediates.
        scope_judge:          L1b judge; defaults to stub (all in-scope).
        classification_judge: L3 judge; defaults to stub (Jaccard + all-unclassified).
                              Ignored for documents segmented via
                              ``use_llm_segmentation`` (see below).
        deviation_judge:      L4 judge; defaults to stub (substantive + neutral risk).
        alignment_judge:      L3 alignment judge; defaults to None (deterministic only).
        trail_judge:          Version-ordering judge; defaults to None (deterministic only).
        signed_judge:         L2 signed-copy judge; defaults to None (deterministic only).
        provenance_judge:     L2 provenance judge; defaults to None (deterministic only).
        no_cache:             If True, skip the cache and force a full recompute.
        use_llm_segmentation: If True, L1 segments every document version via
                              :func:`~playbook_engine.llm_segmentation_stage.segment_to_tree`
                              instead of the deterministic
                              ``segment(ingest(...).tree)`` path. The LLM
                              classifies each clause in the same pass, so
                              ``classification_judge``/``classify_tree`` are
                              bypassed for these documents — their
                              ``taxonomy_id`` comes directly from the LLM's
                              grounded output. Defaults to False (the
                              deterministic path remains the default).
                              Never falls back to the deterministic segmenter
                              on QA failure — a
                              :class:`~playbook_engine.segmentation_qa.SegmentationQAError`
                              propagates uncaught for that document version.
        llm_segment_fn:       Injectable segmenter callable for the LLM path
                              (``Callable[[str, list[Block]], list[SegNode]]``).
                              Only used when ``use_llm_segmentation=True``.
                              Defaults to None, meaning
                              :func:`~playbook_engine.llm_segmentation_stage.segment_to_tree`
                              binds :func:`~playbook_engine.llm_segmenter.segment_document`
                              to a lazily-constructed client. Tests inject a
                              fake so no live API call is made.
        normalize_trail_across_versions: See :func:`_compute_doc_result`; forwarded
                              unchanged. Only meaningful with
                              ``use_llm_segmentation=True``.
        normalize_trail_fn:  See :func:`_compute_doc_result`; forwarded unchanged.
        use_batch_segmentation: If True (only meaningful together with
                              ``use_llm_segmentation=True`` — this flag only
                              changes *how* the LLM segmentation calls
                              happen, not whether they happen at all), every
                              document version's blocks are extracted in one
                              pre-pass and segmented via a single
                              corpus-wide :func:`~playbook_engine.llm_segmenter_batch.segment_documents_batch`
                              call (Anthropic Message Batches — 50% the cost
                              of the per-document synchronous calls
                              ``use_llm_segmentation`` alone makes), instead
                              of one ``segment_document`` call per version
                              inside the per-document loop. Each version's
                              batched ``SegNode`` output still passes through
                              the same deterministic QA gates as the
                              synchronous path (see
                              :func:`_ground_batch_result`), but with **no
                              repair loop**: a gate failure raises
                              :class:`~playbook_engine.segmentation_qa.SegmentationQAError`
                              immediately rather than re-prompting, since
                              there is no per-document ``segment_fn`` to
                              retry with in batch mode. A version whose
                              pre-pass extraction fails is simply absent from
                              the batch and falls back to the normal
                              per-document LLM path for that version only
                              (see ``_collect_batch_items``). Defaults to
                              False (the existing per-document synchronous
                              LLM path remains the default even when
                              ``use_llm_segmentation=True``).
        segmentation_cache:   Optional :class:`~playbook_engine.llm_segmenter_batch.SegmentationVerdictCache`.
                              Only used when ``use_llm_segmentation=True``.
                              Passed through to ``segment_documents_batch``
                              when ``use_batch_segmentation=True`` (repeat
                              runs over unchanged document content skip the
                              batch entirely for those versions), AND to the
                              per-document synchronous LLM path
                              (``_llm_segment_file``/``segment_to_tree``) for
                              any version not resolved via the batch —
                              including every version when
                              ``use_batch_segmentation=False`` — so that path
                              is judge-once/deterministic-replay too (issue
                              #91: this used to be silently batch-only).
                              Defaults to None (no cache — every run
                              re-invokes the LLM for every version).
        segment_documents_batch_fn: Injectable callable matching
                              :func:`~playbook_engine.llm_segmenter_batch.segment_documents_batch`'s
                              signature. Only used when
                              ``use_batch_segmentation=True``. Defaults to
                              None, meaning the real
                              :func:`~playbook_engine.llm_segmenter_batch.segment_documents_batch`
                              is called with a lazily-constructed client
                              (``client=None``). Tests inject a fake (or bind
                              the real function to a fake Anthropic client)
                              so no live API call is made.
        extraction_cache:     Optional :class:`~playbook_engine.extraction.ExtractionCache`.
                              Only used when ``use_llm_segmentation=True``.
                              Forwarded to the batch pre-pass
                              (``_collect_batch_items``) and to the
                              per-document synchronous LLM path
                              (``_llm_segment_file``/``segment_to_tree``) for
                              any version not resolved via the batch — a hit
                              against a version's current file content skips
                              extraction (docling/pdfplumber/python-docx/
                              pandoc) entirely. Independent of
                              ``segmentation_cache`` (which only covers the
                              LLM segmentation call) and independent of
                              ``no_cache`` (which controls the separate L1-L4
                              ``ArtifactStore``/``JudgmentCache`` stage
                              cache) — deliberately so: ``playbook judge``
                              forces ``no_cache=True`` to avoid replaying
                              stale ``needs_review`` sentinels from the
                              store-backed judges, but that must not also
                              force every judge round to re-extract/re-OCR
                              every version of every agreement from scratch
                              (issue #132). Defaults to None (no caching —
                              every run re-extracts).
        entity_registry_path: Path to the persisted entity->alias registry
                              used to pseudonymize ``config.provenance.known_entities``
                              (issue #153). Defaults to
                              :data:`~playbook_engine.entity_registry.DEFAULT_REGISTRY_PATH`
                              (a corpus-wide cache dir) so the same entity
                              gets the same alias across runs/out_dirs by
                              default. Ignored entirely when
                              ``config.provenance.known_entities`` is empty —
                              no registry file is read or written and no
                              held-out map is created.
        progress:             Callable receiving progress message strings.

    Raises:
        PipelineError:  On an unrecoverable pipeline error.
    """
    _scope_judge: ScopeJudge = scope_judge or _AllInScopeJudge()
    _cls_judge: ClassificationJudge = classification_judge or _NullClassificationJudge()
    _dev_judge: DeviationJudge = deviation_judge or _NullDeviationJudge()

    # Judge identity — combines each delegate's class name (+ optional model_id
    # attribute) into a single fingerprint fragment (issue #102). Computed from
    # the raw delegates BEFORE they're wrapped in Batched*Judge below, since
    # every wrapped judge would otherwise report the same wrapper class name.
    # Used both as the verdict cache's model_id (so a verdict cached under one
    # judge is never replayed for a differently-identified judge) and folded
    # into config_fp below (so the L1-L4 stage cache can't replay a whole
    # cached document result computed under the old judge set either — a
    # verdict-cache fix alone doesn't help if the stage cache never even
    # reaches the judges).
    judge_identity = json.dumps(
        {
            "scope": _judge_identity(_scope_judge),
            "classification": _judge_identity(_cls_judge),
            "deviation": _judge_identity(_dev_judge),
        },
        sort_keys=True,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    obs_path = out_dir / "observations.jsonl"
    manifest_path = out_dir / "corpus_manifest.json"

    # Content-addressed stage cache — disabled when no_cache=True.
    store: ArtifactStore | None = None if no_cache else ArtifactStore(out_dir / ".cache")

    # Judgment verdict cache — wraps the inline judges with batching + content-addressed
    # caching so identical clause payloads are judged once per corpus (issue #62).
    # The cache persists across runs in out/.cache/verdicts.jsonl.
    #
    # When no_cache=True the verdict cache is skipped entirely (same flag that
    # disables the #61 stage cache).  This guarantees that "force a full recompute"
    # means a full recompute — no stale verdict hits from a previous run.
    if not no_cache:
        verdict_cache = JudgmentCache(
            out_dir / ".cache" / "verdicts.jsonl",
            model_id=judge_identity,
        )
        _scope_judge = BatchedScopeJudge(delegate=_scope_judge, cache=verdict_cache)
        _cls_judge = BatchedClassificationJudge(delegate=_cls_judge, cache=verdict_cache)
        _dev_judge = BatchedDeviationJudge(delegate=_dev_judge, cache=verdict_cache)

    # Config fingerprint: encodes the fields that affect L1-L4 outputs.
    # Hash the template file's *content* (not its path) so that changing the
    # file's text under the same path correctly busts per-doc cache entries.
    template_content_hash: str | None = None
    if config.baseline.template_path and config.baseline.template_path.exists():
        template_content_hash = _sha256_file(config.baseline.template_path)
    config_fp = make_config_fingerprint(
        {
            "agreement_type_id": config.agreement_type.id,
            "provenance_aliases": sorted(config.provenance.our_party_aliases),
            "template_content_hash": template_content_hash,
            # Switching segmentation paths changes L1 output for identical
            # source files — must bust the cache, not replay a stale tree
            # segmented (and classified) the other way.
            "use_llm_segmentation": use_llm_segmentation,
            # The batch path has no repair loop (see _ground_batch_result), so
            # the same source content can in principle segment differently
            # under batch vs. synchronous LLM calls — never replay one path's
            # stage-cached tree as if it were the other's.
            "use_batch_segmentation": use_batch_segmentation,
            # Toggling cross-version taxonomy normalization changes the L1
            # output for every version of every multi-version agreement — a
            # prior run's un-normalized cached trees must not be replayed
            # silently once this is switched on (issue #90).
            "normalize_trail_across_versions": normalize_trail_across_versions,
            # The segmenter's model id, prompt version, output schema shape,
            # and effort each change what L1 produces for identical source
            # content. These are read from the same module-level constants
            # segment_documents_batch/normalize_trail default to — bumping
            # any one of them (a code change, not a config change) must bust
            # every per-doc cache entry rather than replay a tree produced by
            # the old model/prompt/schema/effort (issue #90).
            "segmentation_model": config.segmentation.model,
            "segmentation_prompt_version": PROMPT_VERSION,
            "segmentation_schema_hash": SCHEMA_HASH,
            "segmentation_effort": DEFAULT_EFFORT,
            # Producer-configurable classification bands (issue #168) change
            # which clauses classify_tree auto-classifies, escalates to the
            # judge, or auto-unclassifies for identical source content — a
            # prior run's classifications under the old thresholds must not
            # be replayed silently once these are changed.
            "classification_ambiguity_threshold": config.classification.ambiguity_threshold,
            "classification_auto_classify_threshold": (
                config.classification.auto_classify_threshold
            ),
            # L1-L4 output depends on which judges produced it, not just which
            # config values were passed — swapping the injected scope/
            # classification/deviation judge (e.g. stub -> real, or one real
            # judge for another) must bust every per-doc stage-cache entry.
            # Without this, the #61 ArtifactStore would replay a whole cached
            # document result computed under the old judge set, and the L1-L4
            # loop would never even reach the judges (let alone the verdict
            # cache) to notice the identity changed (issue #102).
            "judge_identity": judge_identity,
            # Deviation assessment now diffs "unchanged" clauses (including
            # every clause of a single-version document) against the
            # canonical template rather than hardcoding deviation="none"
            # (issue #103) — a code change, not a config change, but one that
            # changes L1-L4 output for identical source content + judges. Bump
            # this constant on any future change to that comparison logic so
            # a warm cache from before the fix is never replayed verbatim.
            "deviation_vs_template_version": _DEVIATION_VS_TEMPLATE_VERSION,
        }
    )

    # -----------------------------------------------------------------------
    # Ingest template — always deterministic (not cached; cheap)
    # -----------------------------------------------------------------------
    template_tree: ClauseTree | None = None
    if config.baseline.template_path:
        try:
            raw_tree = _ingest_file(config.baseline.template_path, "template", "template")
            template_tree = segment(raw_tree)
            progress(f"  template: {config.baseline.template_path.name}")
        except Exception as exc:  # noqa: BLE001
            progress(f"  WARNING: could not ingest template: {exc}")

    t_observations = (
        _build_template_observations(
            template_tree,
            taxonomy,
            _cls_judge,
            ambiguity_threshold=config.classification.ambiguity_threshold,
            auto_classify_threshold=config.classification.auto_classify_threshold,
        )
        if template_tree
        else []
    )

    # -----------------------------------------------------------------------
    # L1 → L4: per-document, optionally cached
    # -----------------------------------------------------------------------
    # our_standard fed to every deviation judge (including the agent-as-judge
    # path — agent_judge.StoreBackedDeviationJudge.assess_batch, whose
    # docstring already claims "NOT truncated") must be the full clause text,
    # not the 200-char text_summary — a 200-char fragment of a real
    # indemnification/insurance clause is not a usable standard to diff
    # against (issue #105).
    template_std_by_tid: dict[str, str] = {}
    for t_obs in t_observations:
        if t_obs.taxonomy_id is not None and t_obs.taxonomy_id not in template_std_by_tid:
            template_std_by_tid[t_obs.taxonomy_id] = t_obs.full_text

    all_observations: list[Observation] = []
    all_round_moves: list[RoundMove] = []
    corpus_documents: list[dict[str, Any]] = []
    # Per-doc our_party-alias match signals (issue #201): True/False from
    # _compute_doc_result's full-tree scan, None for cached results that
    # predate the field. Consumed by the alias sanity check below.
    alias_match_flags: list[bool | None] = []
    # Trails are collected here and written AFTER the born-safe pseudonymization
    # pass (issue #182) so the trail's document_id + filename carry the alias,
    # not the raw counterparty name — keeping them consistent with the
    # pseudonymized observation ids that `inspect` joins on.
    all_trails: list[tuple[str, dict[str, Any]]] = []

    scope_log = ScopeLog(agreement_type_id=config.agreement_type.id)
    doc_dirs = sorted(d for d in corpus_dir.iterdir() if d.is_dir())

    # -------------------------------------------------------------------
    # Batch-segmentation pre-pass (opt-in, issue #76): extract every
    # document version up front and segment the whole corpus in one
    # Message Batches call, before the per-document loop below. Only
    # meaningful together with use_llm_segmentation=True — see mine_corpus's
    # docstring. Per-document results are looked up by _compute_doc_result
    # via batch_seg_nodes/batch_extractions (keyed by doc_id, then version
    # id) so the existing per-document L1-L4 stage cache and control flow
    # are otherwise unchanged.
    # -------------------------------------------------------------------
    batch_seg_nodes_by_doc: dict[str, dict[str, list[SegNode]]] = {}
    batch_extractions_by_doc: dict[str, dict[str, _BatchExtraction]] = {}
    if use_batch_segmentation and use_llm_segmentation:
        doc_versions: dict[str, dict[str, Path]] = {}
        skipped_cache_hit_docs = 0
        for doc_dir in doc_dirs:
            version_files = _discover_versions(doc_dir)
            if not version_files:
                continue
            # Issue #92: a document whose L1-L4 stage cache is already warm
            # will be replayed verbatim by store.get_or_compute in the
            # per-document loop below — it never looks at
            # batch_seg_nodes_by_doc/batch_extractions_by_doc for a cache
            # hit. Extracting and submitting such a document to the (paid)
            # batch here is pure waste: at 40x3 scale, a re-run where only
            # one document changed would otherwise re-extract and
            # re-segment all 120 versions. The cache key inputs (file
            # hashes, config fingerprint, hints) are all available before
            # extraction, so compute it and skip the pre-pass for cache
            # hits entirely.
            if store is not None:
                hints_path = doc_dir / "hints.yaml"
                doc_cache_key = make_doc_key(
                    doc_dir.name, version_files, config_fp, "l1-l4", hints_path
                )
                if store.contains(doc_cache_key):
                    skipped_cache_hit_docs += 1
                    continue
            doc_versions[doc_dir.name] = {vf.stem: vf for vf in version_files}

        if skipped_cache_hit_docs:
            progress(
                f"  batch pre-pass: skipping {skipped_cache_hit_docs} document(s) "
                "already satisfied by the L1-L4 stage cache"
            )

        taxonomy_ids = [e.id for e in taxonomy.classifier_entries()]
        items, batch_extractions_by_doc = _collect_batch_items(
            doc_versions, progress, extraction_cache=extraction_cache
        )
        progress(f"  batch segmentation: {len(items)} version(s) to segment")

        _batch_fn = segment_documents_batch_fn or segment_documents_batch
        results_by_custom_id = _batch_fn(
            items,
            taxonomy_ids=taxonomy_ids,
            cache=segmentation_cache,
            progress=progress,
        )
        for custom_id, seg_nodes in results_by_custom_id.items():
            doc_id, _, vid = custom_id.partition("/")
            batch_seg_nodes_by_doc.setdefault(doc_id, {})[vid] = seg_nodes

    def _run_doc(
        doc_id: str,
        doc_dir: Path,
        version_files: list[Path],
        doc_batch_seg_nodes: dict[str, list[SegNode]] | None,
        doc_batch_extractions: dict[str, _BatchExtraction] | None,
    ) -> Any:
        """Compute one document's L1–L4 result, via the stage cache when present."""
        if store is not None:
            hints_path = doc_dir / "hints.yaml"
            cache_key = make_doc_key(doc_id, version_files, config_fp, "l1-l4", hints_path)

            def _compute(
                _doc_id: str = doc_id,
                _doc_dir: Path = doc_dir,
                _vfs: list[Path] = version_files,
                _batch_seg_nodes: dict[str, list[SegNode]] | None = doc_batch_seg_nodes,
                _batch_extractions: dict[str, _BatchExtraction] | None = doc_batch_extractions,
            ) -> Any:
                return _compute_doc_result(
                    _doc_id,
                    _doc_dir,
                    _vfs,
                    out_dir,
                    config,
                    taxonomy,
                    template_tree,
                    template_std_by_tid,
                    _scope_judge,
                    _cls_judge,
                    _dev_judge,
                    alignment_judge,
                    trail_judge,
                    progress,
                    signed_judge=signed_judge,
                    provenance_judge=provenance_judge,
                    use_llm_segmentation=use_llm_segmentation,
                    llm_segment_fn=llm_segment_fn,
                    normalize_trail_across_versions=normalize_trail_across_versions,
                    normalize_trail_fn=normalize_trail_fn,
                    batch_seg_nodes=_batch_seg_nodes,
                    batch_extractions=_batch_extractions,
                    segmentation_cache=segmentation_cache,
                    extraction_cache=extraction_cache,
                )

            return store.get_or_compute(cache_key, _compute)

        return _compute_doc_result(
            doc_id,
            doc_dir,
            version_files,
            out_dir,
            config,
            taxonomy,
            template_tree,
            template_std_by_tid,
            _scope_judge,
            _cls_judge,
            _dev_judge,
            alignment_judge,
            trail_judge,
            progress,
            signed_judge=signed_judge,
            provenance_judge=provenance_judge,
            use_llm_segmentation=use_llm_segmentation,
            llm_segment_fn=llm_segment_fn,
            normalize_trail_across_versions=normalize_trail_across_versions,
            normalize_trail_fn=normalize_trail_fn,
            batch_seg_nodes=doc_batch_seg_nodes,
            batch_extractions=doc_batch_extractions,
            segmentation_cache=segmentation_cache,
            extraction_cache=extraction_cache,
        )

    # Documents whose LLM segmentation/normalization failed a fail-loud QA gate,
    # or whose hints.yaml is malformed (HintsError — see version_orderer.Hints.load).
    # These are quarantined (recorded in quarantine.json + a loud summary) so a
    # single bad document flags itself for review WITHOUT aborting the whole
    # corpus run and discarding every other document's artifacts — the run still
    # writes observations.jsonl for the documents that passed. A quarantined
    # document is never silently dropped and never degraded to the deterministic
    # segmenter.
    quarantined: list[dict[str, str]] = []

    for doc_dir in doc_dirs:
        doc_id = doc_dir.name
        version_files = _discover_versions(doc_dir)

        legacy_doc_files = _discover_legacy_doc_files(doc_dir)
        if legacy_doc_files:
            names = ", ".join(f.name for f in legacy_doc_files)
            progress(
                f"  {doc_id}: legacy .doc file(s) ignored ({names}) — the engine "
                f"cannot read them; convert with `{_LEGACY_FORMAT_INSTRUCTION} "
                f"{legacy_doc_files[0].name}` and re-run, or this document's "
                "negotiation history may start later than it actually did"
            )

        if not version_files:
            progress(f"  {doc_id}: no supported files (.docx/.pdf/.rtf) — skipping")
            continue

        progress(f"  {doc_id}: {len(version_files)} version file(s)")

        doc_batch_seg_nodes = batch_seg_nodes_by_doc.get(doc_id)
        doc_batch_extractions = batch_extractions_by_doc.get(doc_id)

        try:
            result = _run_doc(
                doc_id, doc_dir, version_files, doc_batch_seg_nodes, doc_batch_extractions
            )
        except (SegmentationQAError, NormalizeTrailError, HintsError) as exc:
            reason = f"{type(exc).__name__}: {exc}"
            progress(f"    QUARANTINED {doc_id}: {reason}")
            quarantined.append({"document_id": doc_id, "reason": reason})
            continue

        if result is None:
            continue

        # Replay the scope decision into scope_log (required whether result came from
        # cache or was freshly computed — OPF §3.6 mandates every document is logged).
        sd = result.get("scope_decision")
        if sd is not None:
            scope_log.record(
                doc_id,
                ScopeDecision(
                    in_scope=sd["in_scope"],
                    scope_rationale=sd["scope_rationale"],
                    scope_confidence=sd["scope_confidence"],
                    basis=sd["basis"],
                ),
            )

        # Collect the trail; it is materialised on disk after the born-safe
        # pseudonymization pass below (issue #182) so its document_id + filename
        # carry the alias rather than the raw counterparty name.
        if result["trail"] is not None:
            all_trails.append((doc_id, result["trail"]))

        # Reconstruct Observation objects from cached dicts.
        doc_obs = _restore_observations(result["observations"])
        all_observations.extend(doc_obs)
        # Round moves (issue #177) — .get(): cached results predating the
        # feature carry no key and simply contribute no trail entries.
        all_round_moves.extend(round_move_from_dict(raw) for raw in result.get("round_moves", []))
        corpus_documents.append(result["corpus_doc"])
        # Alias sanity flags (issue #201) — .get(): cached results predating
        # the field contribute None (unknown) and the corpus-level check
        # below falls back to its observation-text scan for those.
        alias_match_flags.append(result.get("our_alias_matched"))
        progress(f"    {len(doc_obs)} observation(s)")

    # Fail loud but isolated: surface every quarantined document prominently and
    # persist the reasons so a reviewer can act on them.
    if quarantined:
        _atomic_json_write(quarantined, out_dir / "quarantine.json")
        ids = ", ".join(q["document_id"] for q in quarantined)
        progress(
            f"  WARNING: {len(quarantined)} document(s) quarantined for review "
            f"(see quarantine.json): {ids}"
        )

    # Alias sanity check (issue #182): if provenance.our_party_aliases are
    # configured but NONE appear anywhere in the corpus, "us" is almost
    # certainly misconfigured — the classic trap is configuring the brand
    # name while the recitals use the full legal-entity name. Provenance keys
    # on these aliases, so a zero-match set silently mis-attributes every
    # document. The primary signal is the per-document full-tree scan from
    # _compute_doc_result (issue #201 — every version, headings + bodies, so
    # recitals and signature blocks count); the observation-text scan below
    # only backstops cached doc results that predate that field, since those
    # contribute None flags. Checked on RAW text, before pseudonymization.
    our_aliases = [a for a in config.provenance.our_party_aliases if a]
    if our_aliases:
        alias_patterns = [
            re.compile(r"(?<!\w)" + re.escape(a) + r"(?!\w)", re.IGNORECASE) for a in our_aliases
        ]
        matched_any = any(flag is True for flag in alias_match_flags) or any(
            pat.search(obs.full_text) for obs in all_observations for pat in alias_patterns
        )
        if not matched_any:
            progress(
                f"  WARNING: none of the {len(our_aliases)} configured "
                "provenance.our_party_aliases appear anywhere in the corpus (all "
                "versions scanned, headings and body text) — 'us' may be "
                "misconfigured (check the party names your agreements' recitals "
                "actually use). Provenance may be mis-detected for the whole corpus."
            )

    # Born-safe pseudonymization (issue #153): known entity names configured
    # in config.provenance.known_entities are deterministically replaced with
    # stable aliases before observations/corpus_documents ever reach the
    # on-disk store, so the persisted artifact (observations.jsonl,
    # corpus_manifest.json -> playbook.opf.json) never carries a raw
    # counterparty name. The registry's alias->entity reverse map is the
    # sensitive artifact from here on; it is written to a restricted-
    # permission sidecar OUTSIDE the OPF, never embedded in it. A no-op
    # (no registry file touched, no sidecar written) when known_entities
    # is empty — the overwhelmingly common case today.
    known_entities = config.provenance.known_entities
    entity_registry: EntityRegistry | None = (
        EntityRegistry.load(entity_registry_path or DEFAULT_REGISTRY_PATH)
        if known_entities
        else None
    )
    if known_entities and entity_registry is not None:
        # counterparty_ref (issue #177) needs the RAW names to match a deal
        # to its known entity, so it runs BEFORE the erasing pass below and
        # attaches only the born-safe alias.
        all_observations = _attach_counterparty_refs(
            all_observations, known_entities, entity_registry
        )
        all_observations = _pseudonymize_observations(
            all_observations, known_entities, entity_registry
        )
        t_observations = _pseudonymize_observations(t_observations, known_entities, entity_registry)
        all_round_moves = _pseudonymize_round_moves(
            all_round_moves, known_entities, entity_registry
        )
        corpus_documents = _pseudonymize_corpus_documents(
            corpus_documents, known_entities, entity_registry
        )
        # Scope log keys on document_id too (issue #182): alias it so scope.json
        # joins the pseudonymized trail/observation ids in `inspect` instead of
        # producing phantom raw-named entries with "No observations".
        for scope_entry in scope_log.entries:
            scope_entry.document_id = pseudonymize_document_id(
                scope_entry.document_id, known_entities, entity_registry
            )
        write_holdout_map(out_dir / "alias_map.json", entity_registry)

    # Materialise trails now (issue #182): after the pseudonymization pass so the
    # trail's document_id AND filename carry the alias, never the raw
    # counterparty name — keeping them consistent with the aliased
    # citation.document_id that `inspect` joins on.
    trail_dir = out_dir / "trail"
    trail_dir.mkdir(parents=True, exist_ok=True)
    for raw_doc_id, trail in all_trails:
        if entity_registry is not None:
            trail = _pseudonymize_trail(trail, known_entities, entity_registry)
        out_doc_id = trail.get("document_id") or raw_doc_id
        _atomic_json_write(trail, trail_dir / f"{out_doc_id}.json")

    # Write intermediates
    scope_log.write(out_dir / "scope.json")
    write_observations_jsonl(all_observations, obs_path)
    # Round moves (issue #177) — written post-pseudonymization like
    # observations.jsonl; project_playbook reads it back for the
    # negotiation_trail (absent file → no trail, e.g. a pre-#177 store).
    # Truncation runs strictly AFTER the aliasing above: slicing raw text
    # first can cut an entity name mid-word, and a cut name survives the
    # whole-word pseudonymization match (born-safe leak — review finding).
    write_round_moves_jsonl(truncate_move_summaries(all_round_moves), out_dir / "round_moves.jsonl")
    # Persist template observations so project_playbook can read them without re-ingesting.
    template_obs_path = out_dir / "template_observations.jsonl"
    write_observations_jsonl(t_observations, template_obs_path)
    _atomic_json_write(corpus_documents, manifest_path)
    if store is not None:
        progress(
            f"L1-L4 complete: {len(all_observations)} observations, {len(corpus_documents)} docs "
            f"(cache hits={store.hit_count}, misses={store.miss_count})"
        )
    else:
        progress(
            f"L1-L4 complete: {len(all_observations)} observations, {len(corpus_documents)} docs"
        )


def project_playbook(
    out_dir: Path,
    config: EngineConfig,
    taxonomy: Taxonomy,
    *,
    coherence_judge: CoherenceJudge | None = None,
    progress: Callable[[str], None] = lambda _: None,
) -> dict[str, Any]:
    """Run L5 only — read the observation store and write ``playbook.opf.json``.

    Reads from ``{out_dir}/``:

    - ``observations.jsonl``          — written by :func:`mine_corpus`.
    - ``corpus_manifest.json``        — written by :func:`mine_corpus`.
    - ``template_observations.jsonl`` — written by :func:`mine_corpus` (may be absent or empty).
    - ``scope.json``                  — written by :func:`mine_corpus` (may be absent; only
                                        feeds the assembler's stub-basis watermark, issue #101).
    - ``playbook.opf.json``           — a PRIOR compile's output, if present, read only for
                                        its ``curation`` section (attorney-pinned positions,
                                        issue #147), so pins survive this recompile and any
                                        conflict with fresh evidence is (re-)flagged. Absent
                                        on a first compile.

    When a ``coherence_judge`` is supplied this function performs LLM calls
    for flagged clauses.  Without one all L5 logic is deterministic given the
    store.

    Args:
        out_dir:         Output directory that already contains the observation store.
        config:          Engine configuration (agreement type, baseline, taxonomy).
        taxonomy:        Loaded taxonomy object.
        coherence_judge: L5 coherence judge; defaults to None (coherence check skipped).
                         When set, flags are written to ``{out_dir}/coherence_flags.json``.
        progress:        Callable receiving progress message strings.

    Returns:
        Validated playbook dict (also written to ``{out_dir}/playbook.opf.json``).

    Raises:
        PipelineError:  If the observation store is missing or empty.
        AssemblyError:  If the assembled playbook fails schema validation.
    """
    obs_path = out_dir / "observations.jsonl"
    manifest_path = out_dir / "corpus_manifest.json"

    if not obs_path.exists() or not manifest_path.exists():
        missing = obs_path if not obs_path.exists() else manifest_path
        raise PipelineError(
            f"Observation store not found: {missing}. "
            "Run 'playbook mine' first to populate the store."
        )

    corpus_documents = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw_observations = read_observations_jsonl(obs_path)

    # Read scope.json (written by mine_corpus) purely to feed the assembler's
    # stub-basis watermark (issue #101) — the scope decision's basis (e.g.
    # "stub" for the no-LLM default _AllInScopeJudge) never reaches an
    # Observation, so it must be threaded in separately. Absence is tolerated
    # (e.g. a hand-built store in a test) and simply contributes no signal.
    scope_path = out_dir / "scope.json"
    scope_bases: list[str] = []
    if scope_path.exists():
        scope_data = json.loads(scope_path.read_text(encoding="utf-8"))
        scope_bases = [d["basis"] for d in scope_data.get("documents", []) if "basis" in d]

    if not raw_observations and not corpus_documents:
        raise PipelineError(
            f"Observation store is empty: {obs_path} contains no observations. "
            "Run 'playbook mine' on a non-empty corpus first."
        )

    all_observations = _restore_observations(raw_observations)
    progress(f"  loaded {len(all_observations)} observations, {len(corpus_documents)} docs")

    # Read persisted template observations — no ingest or judge calls.
    template_obs_path = out_dir / "template_observations.jsonl"
    raw_t_observations = read_observations_jsonl(template_obs_path)
    t_observations = _restore_observations(raw_t_observations)
    if t_observations:
        progress(f"  loaded {len(t_observations)} template observation(s) from store")

    # -----------------------------------------------------------------------
    # L5: Compile playbook (deterministic given the store)
    # -----------------------------------------------------------------------
    progress("L5: compiling clause positions + library…")

    # Round moves (issue #177) — absent for pre-#177 stores or
    # single-version corpora; read_round_moves_jsonl returns [] then and no
    # negotiation_trail is emitted.
    round_moves = read_round_moves_jsonl(out_dir / "round_moves.jsonl")
    if round_moves:
        progress(f"  loaded {len(round_moves)} round move(s) from store")

    taxonomy_titles = {e.id: e.label for e in taxonomy.entries}
    clause_positions, coherence_flags, unclassified_coverage = compile_clause_positions(
        all_observations,
        t_observations,
        taxonomy_titles=taxonomy_titles,
        coherence_judge=coherence_judge,
        min_evidence_n=config.provenance.min_evidence_n,
        round_moves=round_moves,
    )
    clause_library, _library_unclassified_coverage = compile_clause_library(all_observations)

    # Persist coherence flags (empty list when no judge configured).
    coherence_flags_path = out_dir / "coherence_flags.json"
    _atomic_json_write([f.to_dict() for f in coherence_flags], coherence_flags_path)

    # Assemble baseline dict
    baseline_dict: dict[str, Any] = {
        "has_canonical_template": config.baseline.has_canonical_template,
    }
    if config.baseline.template_path:
        baseline_dict["template_ref"] = {
            "document_id": "template",
            "title": "Canonical Template",
            "source": str(config.baseline.template_path),
        }
        # Content address for the template (issue #185, §4) — same
        # verification path as corpus version_files. Omitted (never
        # fabricated) when the file is not readable at projection time.
        template_path = Path(config.baseline.template_path)
        if template_path.is_file():
            baseline_dict["template_ref"]["sha256"] = file_sha256(template_path)

    # Assemble agreement_type dict
    agreement_type_dict: dict[str, Any] = {
        "id": config.agreement_type.id,
        "name": config.agreement_type.name,
    }
    if config.agreement_type.description:
        agreement_type_dict["description"] = config.agreement_type.description
    if config.agreement_type.aliases:
        agreement_type_dict["aliases"] = list(config.agreement_type.aliases)

    # Assemble taxonomy dict
    taxonomy_dict: dict[str, Any] = {
        "source": taxonomy.source,
        "entries": [
            {
                "id": e.id,
                "label": e.label,
                "status": e.status,
                "cuad_origin": e.cuad_origin,
                "description": e.description,
            }
            for e in taxonomy.entries
        ],
    }

    # Assemble perspective dict (issue #165): only emitted into the assembled
    # document when BOTH party and counterparty_type are known — the OPF
    # schema requires the whole `perspective` object or nothing, and neither
    # field may be fabricated (see assemble_playbook's `perspective`
    # docstring). A party-only default (from provenance.our_party_aliases)
    # lives on config.perspective for other consumers, but is not sufficient
    # on its own to answer "what kind of counterparty is across the table".
    perspective_dict: dict[str, str] | None = None
    if config.perspective.party is not None and config.perspective.counterparty_type is not None:
        perspective_dict = {
            "party": config.perspective.party,
            "counterparty_type": config.perspective.counterparty_type,
        }

    generated_at = datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")

    # Issue #147: read the PRIOR compile's curation overlay (attorney-pinned
    # positions), if a playbook already exists in out_dir, so the merge layer
    # inside assemble_playbook can preserve pins across this recompile and
    # flag/clear conflicts against the freshly recomputed historical_stance.
    # Absent on a first compile — no prior pins to carry forward.
    out_file = out_dir / "playbook.opf.json"
    existing_curation: dict[str, Any] | None = None
    if out_file.exists():
        try:
            prior_playbook = json.loads(out_file.read_text(encoding="utf-8"))
            existing_curation = prior_playbook.get("curation")
        except (json.JSONDecodeError, OSError):
            existing_curation = None

    playbook = assemble_playbook(
        agreement_type=agreement_type_dict,
        baseline=baseline_dict,
        taxonomy=taxonomy_dict,
        clause_positions=clause_positions,
        clause_library=clause_library,
        corpus_documents=corpus_documents,
        generated_at=generated_at,
        observations=all_observations,
        scope_bases=scope_bases,
        unclassified_coverage=unclassified_coverage,
        perspective=perspective_dict,
        min_evidence_n=config.provenance.min_evidence_n,
        existing_curation=existing_curation,
    )

    write_playbook(playbook, out_file)
    progress(f"Playbook written: {out_file}")

    return playbook


_STOP_AFTER_CHOICES: frozenset[str] = frozenset({"intermediates"})


def compile_corpus(
    corpus_dir: Path,
    config: EngineConfig,
    taxonomy: Taxonomy,
    out_dir: Path,
    *,
    scope_judge: ScopeJudge | None = None,
    classification_judge: ClassificationJudge | None = None,
    deviation_judge: DeviationJudge | None = None,
    alignment_judge: AlignmentJudge | None = None,
    trail_judge: TrailJudge | None = None,
    signed_judge: SignedJudge | None = None,
    provenance_judge: ProvenanceJudge | None = None,
    coherence_judge: CoherenceJudge | None = None,
    no_cache: bool = False,
    # Backward-compatibility alias: ``resume=False`` maps to ``no_cache=True``.
    resume: bool = True,
    use_llm_segmentation: bool = False,
    llm_segment_fn: SegmentFn | None = None,
    normalize_trail_across_versions: bool = False,
    normalize_trail_fn: NormalizeTrailFn | None = None,
    use_batch_segmentation: bool = False,
    segmentation_cache: SegmentationVerdictCache | None = None,
    segment_documents_batch_fn: Callable[..., dict[str, list[SegNode]]] | None = None,
    extraction_cache: ExtractionCache | None = None,
    entity_registry_path: Path | None = None,
    stop_after: str | None = None,
    progress: Callable[[str], None] = lambda _: None,
) -> dict[str, Any]:
    """Compile a corpus directory into a validated OPF playbook.

    Convenience wrapper that runs :func:`mine_corpus` (L1–L4) then
    :func:`project_playbook` (L5).  The content-addressed stage cache is used
    by default; pass *no_cache=True* to force a full recompute.

    Args:
        corpus_dir:           Root corpus directory (one subdirectory per agreement).
        config:               Engine configuration (agreement type, baseline, taxonomy).
        taxonomy:             Loaded taxonomy object.
        out_dir:              Output directory for intermediates + playbook.opf.json.
        scope_judge:          L1b judge; defaults to stub (all in-scope).
        classification_judge: L3 judge; defaults to stub (Jaccard + all-unclassified).
        deviation_judge:      L4 judge; defaults to stub (substantive + neutral risk).
        alignment_judge:      L3 alignment judge; defaults to None (deterministic only).
        trail_judge:          Version-ordering judge; defaults to None (deterministic only).
        signed_judge:         L2 signed-copy judge; defaults to None (deterministic only).
        provenance_judge:     L2 provenance judge; defaults to None (deterministic only).
        coherence_judge:      L5 coherence judge; defaults to None (coherence check skipped).
                              When set, flags are written to ``{out_dir}/coherence_flags.json``.
        no_cache:             If True, skip the cache and force a full recompute.
        resume:               Deprecated — use *no_cache* instead.  ``resume=False``
                              is equivalent to ``no_cache=True``.
        use_llm_segmentation: Passed through to :func:`mine_corpus`; see there
                              for the full contract. Defaults to False.
        llm_segment_fn:       Passed through to :func:`mine_corpus`; only used
                              when ``use_llm_segmentation=True``.
        normalize_trail_across_versions,
        normalize_trail_fn,
        use_batch_segmentation,
        segmentation_cache,
        segment_documents_batch_fn,
        extraction_cache:
                              Passed through to :func:`mine_corpus` so a compile
                              run segments identically to a ``mine``/``judge``
                              run; see there for the full contract. All default
                              off (deterministic path).
        entity_registry_path: Passed through to :func:`mine_corpus`; see there
                              for the full contract (issue #153).
        stop_after:           If ``"intermediates"``, stop after writing
                              ``scope.json``, ``observations.jsonl``,
                              ``corpus_manifest.json``, and ``trail/<doc>.json``
                              and return a status dict instead of the playbook.
                              ``playbook.opf.json`` is NOT written.
                              Supported values: ``"intermediates"``.
        progress:             Callable receiving progress message strings.

    Returns:
        Validated playbook dict (also written to ``{out_dir}/playbook.opf.json``),
        or a status dict ``{"stopped_after": "intermediates", "out_dir": str,
        "documents": int}`` when *stop_after* is set.

    Raises:
        ValueError:     If *stop_after* is not a recognised checkpoint name.
        PipelineError:  On an unrecoverable pipeline error.
        AssemblyError:  If the assembled playbook fails schema validation.
    """
    if stop_after is not None and stop_after not in _STOP_AFTER_CHOICES:
        raise ValueError(
            f"Unsupported stop_after value {stop_after!r}. "
            f"Supported values: {sorted(_STOP_AFTER_CHOICES)}"
        )

    out_dir.mkdir(parents=True, exist_ok=True)

    # Honour the deprecated ``resume`` param: resume=False → no_cache=True.
    effective_no_cache = no_cache or (not resume)

    mine_corpus(
        corpus_dir=corpus_dir,
        config=config,
        taxonomy=taxonomy,
        out_dir=out_dir,
        scope_judge=scope_judge,
        classification_judge=classification_judge,
        deviation_judge=deviation_judge,
        alignment_judge=alignment_judge,
        trail_judge=trail_judge,
        signed_judge=signed_judge,
        provenance_judge=provenance_judge,
        no_cache=effective_no_cache,
        use_llm_segmentation=use_llm_segmentation,
        llm_segment_fn=llm_segment_fn,
        normalize_trail_across_versions=normalize_trail_across_versions,
        normalize_trail_fn=normalize_trail_fn,
        use_batch_segmentation=use_batch_segmentation,
        segmentation_cache=segmentation_cache,
        segment_documents_batch_fn=segment_documents_batch_fn,
        extraction_cache=extraction_cache,
        entity_registry_path=entity_registry_path,
        progress=progress,
    )

    if stop_after == "intermediates":
        # Count documents recorded in the manifest written by mine_corpus.
        manifest_path = out_dir / "corpus_manifest.json"
        doc_count = 0
        if manifest_path.exists():
            doc_count = len(json.loads(manifest_path.read_text(encoding="utf-8")))
        progress("Stopped after intermediates (no playbook compiled).")
        return {
            "stopped_after": "intermediates",
            "out_dir": str(out_dir),
            "documents": doc_count,
        }

    return project_playbook(
        out_dir=out_dir,
        config=config,
        taxonomy=taxonomy,
        coherence_judge=coherence_judge,
        progress=progress,
    )
