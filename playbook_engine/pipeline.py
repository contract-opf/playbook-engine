"""Pipeline orchestration — corpus → playbook (L1 → L5).

Wires all pipeline stages into a single ``compile_corpus()`` call.
LLM-facing stages accept injected judge objects; stub implementations
are provided as CLI defaults when no real LLM is configured.

Security: no agreement content is stored in this module.
All corpus content is read from caller-supplied paths at runtime.
"""

from __future__ import annotations

import contextlib
import dataclasses
import datetime
import json
import os
import re
import shutil
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from playbook_engine.agent_judge import PendingQueue, StoreBackedEquivalenceJudge, VerdictStore
from playbook_engine.artifact_store import (
    ArtifactStore,
    _sha256_file,
    make_config_fingerprint,
    make_doc_key,
    make_stage_key,
    write_text_if_changed,
)
from playbook_engine.canonicalize import file_sha256
from playbook_engine.clause_aligner import AlignmentJudge, align_versions
from playbook_engine.clause_classifier import (
    AMBIGUITY_THRESHOLD,
    AUTO_CLASSIFY_THRESHOLD,
    ClassificationJudge,
    ClassifiedClause,
    ClauseClassification,
    assign_by_content,
    classify_tree,
)
from playbook_engine.clause_differ import ClauseDiff, diff_aligned
from playbook_engine.clause_position_compiler import compile_clause_positions
from playbook_engine.clause_tree import ClauseTree
from playbook_engine.config import EngineConfig
from playbook_engine.deviation_classifier import assess_deviations_deterministic
from playbook_engine.docx_ingester import TextUnit, TrackedChanges, ingest_docx
from playbook_engine.entity_registry import (
    DEFAULT_REGISTRY_PATH,
    EntityRegistry,
    entity_slug,
    known_entities_with_no_match,
    pseudonymize_document_id,
    pseudonymize_text,
    write_holdout_map,
)
from playbook_engine.equivalence import EquivalenceSubject, collect_subjects
from playbook_engine.extraction import (
    FAILURE_TIMEOUT,
    ExtractionCache,
    ExtractionError,
    ExtractorLabel,
    bridge_tracked_change_spans,
    detect_extractor,
    document_timestamp,
    extract_blocks,
    extract_docx_units_and_tracked_changes,
    ocrmypdf_available,
)
from playbook_engine.judgment import (
    BatchedClassificationJudge,
    BatchedScopeJudge,
    JudgmentCache,
)
from playbook_engine.llm_segmentation_stage import (
    SegmentFn,
    extractor_label_of,
    segment_to_tree,
)
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
from playbook_engine.natural_sort import natural_sort_key
from playbook_engine.observation_builder import (
    Observation,
    ObservationCitation,
    RoundMove,
    build_observations,
    build_round_moves,
    observations_jsonl_text,
    read_observations_jsonl,
    read_round_moves_jsonl,
    round_move_from_dict,
    round_moves_jsonl_text,
    summarize_clause_text,
    truncate_move_summaries,
    truncate_search_snippets,
)
from playbook_engine.overrides import (
    OVERRIDES_FILENAME,
    OverridesError,
    fold_overrides,
    load_overrides,
)
from playbook_engine.pdf_ingester import ingest_pdf
from playbook_engine.playbook_assembler import (
    _strip_invisible,
    assemble_playbook,
    write_playbook,
    write_precedent_sidecar,
)
from playbook_engine.precedent import build_precedent_evidence
from playbook_engine.provenance_detector import (
    PROVENANCE_UNKNOWN,
    ProvenanceJudge,
    ProvenanceResult,
    detect_provenance,
    two_valued_side,
)
from playbook_engine.reversal_detector import detect_reversals
from playbook_engine.rtf_ingester import ingest_rtf
from playbook_engine.rubric import RubricPolicy, current_versions
from playbook_engine.scope_gate import (
    ScopeDecision,
    ScopeJudge,
    ScopeLog,
    scope_gate,
)
from playbook_engine.segmentation_grounding import Block, SegNode
from playbook_engine.segmentation_qa import SegmentationQAError, run_gates
from playbook_engine.segmenter import segment
from playbook_engine.signed_detector import (
    SignedJudge,
    SignedStatus,
    detect_signed,
    strip_signature_block,
)
from playbook_engine.taxonomy import Taxonomy
from playbook_engine.template_standards import (
    form_front_matter,
    front_matter_indices,
    template_standards,
)
from playbook_engine.tracked_changes_overlay import (
    HunkEnrichment,
    enrich_clause_diff,
    round_level_fallback_attribution,
)
from playbook_engine.version_orderer import (
    Hints,
    HintsError,
    TrailJudge,
    VersionInput,
    order_versions,
)

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
#
# v5: clause alignment gained a global move-matching phase
# (clause_aligner._match_moves) — a clause that relocates between drafts now
# aligns to itself by content instead of degenerating into a delete+add pair,
# which changes L3/L4 output (alignments, diffs, observations) for identical
# source content. A warm cache from before this change would keep replaying
# relocation artifacts forever without this bump.
#
# v6: on the agent/LLM segmentation path the baseline template is now
# segmented+classified through the same store-backed path as the corpus
# documents (previously: deterministic segment + heading-similarity
# classify_tree, which classified nothing on a real template and silently
# degraded template mode to emergent). template_std_by_tid now populates, so
# per-doc deviation assessments computed against the old empty standards must
# not be replayed.
#
# v7 (issue #167): assess_deviations gained a third deterministic fast path
# — an added/removed clause whose normalized text also occurs in the
# counterpart version's clause tree is now classified basis="alignment"
# instead of being sent to the judge (see deviation_classifier module
# docstring and pipeline._assess_deviations_with_standards's
# counterpart_clause_texts). A warm cache from before this change has
# per-doc deviation results computed without this fast path — it would keep
# replaying the old judge-routed (and judge-cost-heavy) verdicts forever
# without this bump, even though the deterministic result for identical
# source content is now available for free.
#
# v8 (issue #216): build_observations now emits exactly one terminal
# observation per (document, taxonomy_id) from the terminal version's own
# tree, never labels removed-before-signing text "signed" (it is classified
# by origin against template_std_by_tid), and corpus_doc
# gained "dropped_observations" — a warm cache would otherwise keep
# replaying the old one-row-per-node observations (inflated precedent) forever.
#
# v9 (issue #220): with no deviation judge configured (the new default) every
# deviation is the deterministic standard check instead of a needs_review
# stub, and observations gained a "standard" field — a warm cache would
# otherwise replay stub-mode verdicts with no standard fact forever.
#
# v10 (issue #229): the origin test for text removed before signing
# (observation_builder._is_standard_language) is now the exact
# is_standard_text check (plus a word-boundary fragment test on
# normalize_for_standard output, party names neutralized), not the token
# Jaccard >= 0.92 — a first draft that flips a negation or deletes a carve-out
# from our clause is their refused ask, no longer our concession. A warm
# cache would otherwise replay the old conceded_before_signing rows forever.
#
# v11 (issue #225): an ambiguous provenance detection is recorded "unknown"
# instead of being coerced to counterparty_paper — the trail's and every
# observation's provenance become "unknown" for such a deal, observations
# gained "paper_basis"/"paper_confidence" (the detection signal the OPF 0.5
# precedent record's paper side rests on), the alias_first_party confidence
# dropped to 0.70, template similarity now compares node fingerprints, and
# corpus_doc provenance is written through two_valued_side. A warm cache
# would otherwise replay the old relabelled side (and no paper_basis) forever.
#
# v12 (issue #221): placeholder signature lines ("By: [Name]", "By: Name:",
# "By: Authorized Signatory") no longer count as signed, so a deal's signed
# anchor (trail signed_version, corpus_doc signed_version, every observation's
# outcome) can change; each version's own document timestamp now seeds
# order_versions (the trail gained "version_timestamps" and an unsigned deal's
# chain direction can change); and a deal with no detected signed copy gets
# no reversals and no proposed_then_reversed observations (counted under
# dropped_observations "refused_ask_no_signed_copy"). A warm cache would
# otherwise replay the old signed anchors, orderings and refused asks forever.
#
# v13 (issue #222): the bucket aligner binds two clauses only at a token
# Jaccard >= ALIGNMENT_AMBIGUITY_THRESHOLD (or through the narrow
# localized-edit rescue) and similarity-matches equal-count buckets
# instead of zipping them by position; a reversal now requires fewer than
# half of the proposed content tokens to survive (trail reversals gained
# "retained" / "alignment_confidence"); heading-less children inherit their
# parent's classification (basis "inherited"); and observations gained
# "x_alignment_confidence". A warm cache would otherwise replay the old
# position-zipped alignments, subset-rule reversals and unclassified
# children forever.
#
# v14 (issue #232): a move row the global move phase chained from a later
# draft onwards is now extended backwards, by the bucket path's bind rule, to
# an earlier draft's unmatched copy of the clause, so a clause edited in an
# early round and then carried into the signed copy is one modified row
# instead of a removed + added pair. Alignments, diffs, reversals and
# observations change for identical source content; a warm cache would
# otherwise replay the old fabricated concessions and refused asks forever.
#
# v15 (issue #237): observations gained "x_classification_basis" — how the
# cited node's taxonomy_id was reached (or "aligned"), which `playbook
# scorecard` counts. A warm cache would otherwise replay observations with
# no classification basis forever, and the scorecard would report them all
# as unrecorded.
#
# v16 (issue #237): an unclassified observation's basis is "unclassified"
# even when its cited node is classified ("aligned" now only means
# "classified via the aligned row"). A warm v15 cache would otherwise replay
# those observations as "aligned".
#
# v17 (issue #239): the judged deviation path is gone. Observation output is
# unchanged on the deterministic path, but the judge identity folded into the
# stage-cache key lost its "deviation" component, and observations from a
# judged run (the removed opt-in layer) must never be replayed as current.
#
# v18 (issue #235): a node every other path left unclassified is now assigned
# by content similarity to our standard's clause text (basis
# "content_similarity", clause_classifier.assign_by_content), so counterparty-
# form clauses with their own headings gain a taxonomy_id. L3 (and so L4's
# alignments, observations and precedent) changes for identical inputs; a warm
# cache would otherwise replay those clauses as unclassified forever.
#
# v19 (issue #233): L4 records what every clause opened with — each terminal
# observation gains "opened_with" and a signed deal gains one "opening"
# observation per clause type whose first-draft text differs from what was
# signed (observation_builder.OUTCOME_OPENING). L4 output changes for
# identical inputs; a warm cache would otherwise replay observations with no
# opening evidence forever.
_DEVIATION_VS_TEMPLATE_VERSION = 19

# Bump whenever the SHAPE of what _compute_doc_result records into
# version_ingest changes in a way that must invalidate a warm L1-L4 stage
# cache (out/.cache) — same convention as _DEVIATION_VS_TEMPLATE_VERSION
# above, folded into config_fp alongside it.
#
# v1 (issue #81): version_ingest entries gained a "reason" field
# (env-missing | backend-error | declared | None) alongside "extractor", and
# a live per-file docling->legacy fallback on the LLM-segmentation path is
# now labeled "legacy" from the label extract_blocks actually returned
# (previously mislabeled "docling" — the up-front detect_extractor(vf)
# PATH-check guess — until a cache-hit replay happened to correct it; see
# extraction.ExtractorLabel). A warm cache entry from before this fix has
# neither the corrected label nor any "reason" key at all, so
# config.extraction.max_fallback would silently find zero fallbacks against
# a replayed pre-#81 result even on a corpus that DID fall back.
#
# v2 (issue #118): tracked_by_vid's char_spans on the LLM-segmentation path
# are now bridged into the tree's own coordinate space (or cleared to None
# when the bridge can't be confirmed) instead of passed through raw — see
# _bridge_tracked_changes_if_needed. Round-move/clause attribution
# (observations' "attribution" field, threaded through version_ingest's
# sibling cached output — the whole per-doc L1-L4 result, not just
# version_ingest itself) changes for identical source content and config: a
# warm out/.cache entry from before this fix would keep replaying
# corpus-wide all-"unknown" attribution as if it were still current.
#
# v3 (issue #217): "ok" version_ingest entries gained
# "signature_block_span" — and the per-doc result's trees/observations now
# have the signature block cut out of the last clause's text. A warm entry
# from before this carries neither.
#
# v4 (issue #218): a FAILED version_ingest row now records the extraction
# failure's closed-enum reason ("timeout" | "no-text" — ExtractionError.reason)
# instead of always None, and a docling-environment PDF with no text gets an
# ocrmypdf retry before it can fail at all. A warm out/.cache entry from
# before this would replay reason=None for a timed-out version (so a consumer
# could not tell "retry me" from "genuinely unreadable") and would never give
# a scanned PDF its second OCR pass.
#
# v5 (issue #231): text recovered by the ocrmypdf second OCR path is now
# recorded with reason "ocr-recovered" instead of "backend-error", and a
# deal holding a version whose text a fallback recovered after a docling
# timeout is no longer stage-cached. A warm entry from before this would
# replay "backend-error" for an OCR-recovered scan and would keep replaying a
# timeout-pinned legacy extraction, never retrying docling.
_VERSION_INGEST_REASON_VERSION = 5

# Bump whenever the SHAPE of what _compute_doc_result records into
# version_trees changes in a way that must invalidate a warm L1-L4 stage
# cache (out/.cache) — same convention as _VERSION_INGEST_REASON_VERSION
# above.
#
# v1 (issue #139): the per-doc result gained "version_trees" (each mined
# version's ClauseTree serialised via to_dict()), so normalized/ can be
# stale-cleared and rewritten under the aliased document_id AFTER the
# born-safe pseudonymization pass — mirroring trail/'s treatment — instead
# of being written raw, mid-loop, under doc_id (see mine_corpus's
# "Materialise normalized/ clause trees now" comment). A warm cache entry
# from before this fix has no "version_trees" key at all
# (``result.get("version_trees", {})`` degrades to empty), so replaying it
# would silently skip writing that document's trees on this run — bump so
# every existing entry recomputes once and the trees are captured.
#
# v2 (issue #217): every ingester node's char_span now covers the whole
# clause (heading start → end of own body text) with the heading line in a
# new heading_span, and the signature block is cut out of the last clause's
# text — a warm entry from before this would replay heading-only spans and
# signature-block text into normalized/ and every citation.
_NORMALIZED_TREES_CACHE_VERSION = 2

# Layered stage cache (issue #219). L1 (ingest + segment) is cached PER
# VERSION FILE under stage "l1", keyed by the file's name + content and only
# the configuration that changes L1 output; L2-L4 is cached per document
# under stage "l2-l4", keyed by the L1 records it consumed plus everything
# else (template, taxonomy, judges, thresholds). A taxonomy or template edit
# therefore replays every L1 tree and recomputes only classification onward.
# Bump _L1_RECORD_VERSION whenever _l1_version_record's output changes for
# identical inputs (L2-L4 changes keep using _DEVIATION_VS_TEMPLATE_VERSION
# and the other per-result versions above).
_L1_STAGE = "l1"
_L2_L4_STAGE = "l2-l4"
_L1_RECORD_VERSION = 1
# Where a cached record keeps the store entries it was built from, as
# ``{key: fingerprint}`` (issue #219) — an L1 record the segmentation-store
# entries it was grounded from, an L2-L4 result the judge verdicts it replayed.
# A replay re-checks every one against the store and recomputes on any change.
_L1_DEPS_KEY = "segmentation_deps"
_VERDICT_DEPS_KEY = "verdict_deps"
# Rubric tally an L2-L4 result made when computed, replayed into the run's
# RubricPolicy on a cache hit so the stale/legacy report stays complete.
_RUBRIC_COUNTS_KEY = "rubric_counts"
# Set (to True) on an L1 record whose text a fallback recovered after a
# docling timeout (issue #231): such a record is never stage-cached, and its
# deal is never L2-L4-cached either.
_L1_TIMED_OUT_KEY = "timed_out"

# version_ingest[].reason values that represent a real DEGRADATION — the
# legacy adapter ran because docling was unavailable or crashed on this
# file, not because it was deliberately declared (issue #81). This is
# exactly what config.extraction.max_fallback counts and what the CLI/review
# advisory flags surface; "declared" is a producer's deliberate choice and
# never counts as a fallback.
_FALLBACK_REASONS = frozenset({"env-missing", "backend-error", "ocr-recovered"})

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


def _distinct(items: Iterable[Any]) -> list[Any]:
    """*items* with repeats (by identity) dropped, first-seen order kept."""
    out: list[Any] = []
    for item in items:
        if not any(item is seen for seen in out):
            out.append(item)
    return out


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


def _llm_tracked_changes(
    vf: Path, progress: Callable[[str], None]
) -> tuple[TrackedChanges | None, list[TextUnit]]:
    """Best-effort DOCX tracked-changes side-channel for an LLM-segmented version (issue #85).

    The deterministic branch of ``_compute_doc_result`` gets its
    ``tracked_by_vid`` entry "for free" from ``_ingest_file_tracked``, since
    that same call also builds the version's ``ClauseTree``. The
    LLM-segmentation branches build their tree via
    :mod:`playbook_engine.llm_segmentation_stage`/``_ground_batch_result``
    instead, which never touches ``docx_ingester`` at all — so this calls
    :func:`~playbook_engine.extraction.extract_docx_units_and_tracked_changes`
    directly on *vf* to fill that gap, independent of whichever adapter
    (``extract_blocks``'s docling or legacy path) produced this version's
    canonical text/blocks.

    Unlike the deterministic branch — where a failure here means the whole
    version has no usable tree either, so letting it propagate is correct —
    an LLM-segmented version's tree has ALREADY been produced successfully
    by the time this runs. Tracked-changes attribution is a bonus signal
    (see ``tracked_changes_overlay.py``'s module docstring: "It degrades
    silently for PDFs, clean DOCX files, and any version pair where no
    tracked-changes data was captured"), so a DOCX that ``python-docx``
    cannot open here (but docling could, upstream) must not retroactively
    fail an otherwise-successful version — any exception is caught, logged
    via *progress*, and degrades to ``(None, [])``, exactly like a PDF/RTF
    version.

    Returns ``(tracked, units)`` (issue #118) — *units* is docx_ingester's
    own ordered text-unit stream for *vf*, the coordinate space *tracked*'s
    char_spans are offset into. Callers whose ``ClauseTree`` came from a
    DIFFERENT extractor (the docling path) need both to bridge — see
    :func:`_bridge_tracked_changes_if_needed`. Always ``[]`` alongside a
    ``None`` *tracked* (non-DOCX, or the exception path above).
    """
    try:
        result = extract_docx_units_and_tracked_changes(vf)
        if result is None:
            return None, []
        units, tracked = result
        return tracked, units
    except Exception as exc:  # noqa: BLE001 — bonus signal, must never fail the version
        progress(f"    WARNING: {vf.name}: tracked-changes side-channel unavailable ({exc})")
        return None, []


def _extract_blocks_for_bridge(
    vf: Path,
    extraction_cache: ExtractionCache | None,
    refresh_extraction: bool,
    extractor: str,
) -> list[Block] | None:
    """Best-effort re-fetch of *vf*'s extracted blocks, for tracked-changes
    span bridging on the SYNCHRONOUS LLM-segmentation path (issue #118).

    ``_llm_segment_file`` (via ``segment_to_tree``) already called
    ``extract_blocks`` for this exact ``(path, extraction_cache, extractor)``
    combination moments ago and, as a side effect, wrote its blocks into
    *extraction_cache* under that key. Correctness — not just speed —
    requires this call to land on that SAME cache entry: the bridge must
    translate spans into the coordinate space the tree was just built from,
    and a second real extraction (even a byte-identical one) is not
    guaranteed to reproduce that space. The production call site
    (``_compute_doc_result``, pipeline.py:~1818) therefore always passes
    ``refresh_extraction=False`` here regardless of the run's own
    ``--no-cache``/``refresh_extraction`` setting, making this a cache hit
    in the common case — not a second real extraction (docling/pdfplumber/
    python-docx/pandoc never actually re-runs). That guarantee holds only
    for that call site's argument, though: a caller that passes
    ``refresh_extraction=True`` here, or a ``None`` *extraction_cache*
    (nothing to hit), does force a real second extraction — this function
    does not enforce ``refresh=False`` itself. Only called at all when a
    bridge is actually needed (see ``_bridge_tracked_changes_if_needed``'s
    callers): most DOCX versions carry no tracked changes, so the common
    case never reaches this function.

    Returns ``None`` (never raises) on failure — tracked-changes attribution
    is a bonus signal and must not retroactively fail an already-successful
    version, same tolerance as ``_llm_tracked_changes``. ``None`` here
    degrades ``_bridge_tracked_changes_if_needed`` to stripping the
    now-untrustworthy raw spans rather than translating them.
    """
    try:
        _canonical_text, blocks, _label = extract_blocks(
            vf, cache=extraction_cache, refresh=refresh_extraction, extractor=extractor
        )
        return blocks
    except Exception:  # noqa: BLE001 — bonus signal, must never fail the version
        return None


def _bridge_tracked_changes_if_needed(
    tracked: TrackedChanges | None,
    units: list[TextUnit],
    blocks: list[Block] | None,
    extractor_label: ExtractorLabel | None,
) -> TrackedChanges | None:
    """Translate *tracked*'s char_spans into *blocks*'s coordinate space
    when *extractor_label* means they are NOT already in it (issue #118).

    - ``extractor_label`` is ``None`` or ``"legacy"``: no bridge needed —
      the legacy DOCX adapter reuses ``docx_ingester``'s own paragraph-join
      text verbatim (see ``extraction.py``'s module docstring, "DOCX
      (fallback)"), so *tracked*'s char_spans are ALREADY in the tree's
      coordinate space. Returned unchanged (``None`` covers both "no
      side-channel" and the deterministic path, which never calls this at
      all — see its callers).
    - Otherwise (docling, or any future non-legacy DOCX extractor): a bridge
      IS needed. When *blocks* is available, translate via
      :func:`~playbook_engine.extraction.bridge_tracked_change_spans`. When
      *blocks* is ``None`` (the best-effort re-extraction in
      ``_extract_blocks_for_bridge`` failed) or *units* is empty, there is
      no way to confirm the coordinate space — per issue #118's safety
      gate, an untranslated raw span must never be trusted here (numeric
      coincidence risk on repeated boilerplate), so every change's
      char_span is cleared to ``None`` rather than left as-is. This is a
      deliberate behavior change from pre-#118: previously the raw
      (possibly wrong-coordinate-space) span reached
      ``tracked_changes_overlay.enrich_clause_diff`` unconditionally.
      Clause-path matching and the round-level fallback tier
      (``tracked_changes_overlay.round_level_fallback_attribution``) still
      work with a cleared span; a spurious span-overlap match does not.
    """
    if tracked is None or not tracked.changes:
        return tracked
    if extractor_label is None or extractor_label == "legacy":
        return tracked
    if blocks is None or not units:
        return TrackedChanges(
            document_id=tracked.document_id,
            version=tracked.version,
            changes=[dataclasses.replace(c, char_span=None) for c in tracked.changes],
        )
    return bridge_tracked_change_spans(tracked, units, blocks)


def _discover_versions(doc_dir: Path) -> list[Path]:
    """Return agreement files in a document directory, in natural sort order."""
    return sorted(
        (p for p in doc_dir.iterdir() if p.is_file() and p.suffix.lower() in _SUPPORTED_EXTENSIONS),
        key=lambda p: (natural_sort_key(p.stem), p.name),
    )


def _discover_legacy_doc_files(doc_dir: Path) -> list[Path]:
    """Return legacy .doc files in a document directory, in natural sort order.

    These are excluded from :func:`_discover_versions` (the engine cannot read
    them) but are common in real negotiation history, so callers surface them
    distinctly rather than silently losing early drafts — see
    ``_LEGACY_EXTENSIONS``.
    """
    return sorted(
        (p for p in doc_dir.iterdir() if p.is_file() and p.suffix.lower() in _LEGACY_EXTENSIONS),
        key=lambda p: (natural_sort_key(p.stem), p.name),
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
    refresh_extraction: bool = False,
    extractor: str = "auto",
) -> tuple[ClauseTree, dict[str, str | None], ExtractorLabel]:
    """LLM-segment one agreement file → ``(tree, taxonomy_by_path, extractor_label)``.

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

    ``refresh_extraction``, when True, is forwarded to ``segment_to_tree`` so
    this call bypasses ``extraction_cache``'s reads (always re-extracts) while
    still refreshing the cache — see ``mine_corpus``'s parameter of the same
    name (issue #78).

    ``extractor`` is forwarded to ``segment_to_tree`` unchanged — the
    declared extractor environment (``config.extraction.extractor``, issue
    #80). Defaults to ``"auto"`` (today's behavior).

    The returned ``extractor_label`` is the real :class:`~playbook_engine.extraction.ExtractorLabel`
    ``segment_to_tree``/``extract_blocks`` resolved for *path* — the actual
    post-fallback label, not a PATH-check guess (issue #81). Callers that
    record ``version_ingest``/``corpus_manifest.json`` entries (see
    ``_compute_doc_result``) use this directly instead of the old up-front
    ``detect_extractor(vf)`` guess, which could not see a live per-file
    docling->legacy fallback.
    """
    result, extractor_label = segment_to_tree(
        path,
        taxonomy_ids=taxonomy_ids,
        segment_fn=segment_fn,
        cache=segmentation_cache,
        model=model,
        extraction_cache=extraction_cache,
        refresh_extraction=refresh_extraction,
        extractor=extractor,
    )
    result.tree.document_id = document_id
    result.tree.version = version
    result.tree.source_file = path.name
    return result.tree, result.taxonomy_by_path, extractor_label


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

    ``extractor_label`` (issue #81) is the real
    :class:`~playbook_engine.extraction.ExtractorLabel` :func:`extract_blocks`
    resolved for this version — ``_compute_doc_result`` reads it straight off
    this object for a batch-resolved version's ``version_ingest`` entry,
    instead of the old up-front ``detect_extractor(vf)`` PATH-check guess.
    """

    canonical_text: str
    blocks: list[Block]
    source_file: str
    extractor_label: ExtractorLabel


def _collect_batch_items(
    doc_versions: dict[str, dict[str, Path]],
    progress: Callable[[str], None],
    extraction_cache: ExtractionCache | None = None,
    refresh_extraction: bool = False,
    extractor: str = "auto",
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
        refresh_extraction: Forwarded to ``extract_blocks`` as its ``refresh``
                      argument for every version — bypasses cache reads while
                      still refreshing the cache (issue #78). Defaults to False.
        extractor:    Forwarded to ``extract_blocks`` unchanged for every
                      version — the declared extractor environment
                      (``config.extraction.extractor``, issue #80). Defaults
                      to ``"auto"`` (today's behavior). Note: the caller
                      (``mine_corpus``) is expected to have already validated
                      a declared ``"docling"`` is available up front (see
                      ``cli._llm_segmentation_kwargs``), so a per-version
                      ``ExtractionError`` here is a genuine per-file failure,
                      not a corpus-wide misconfiguration — hence it keeps the
                      existing "warn and skip this one version" tolerance
                      below rather than aborting the whole batch pre-pass.

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
                canonical_text, blocks, extractor_label = extract_blocks(
                    path, cache=extraction_cache, refresh=refresh_extraction, extractor=extractor
                )
            except Exception as exc:  # noqa: BLE001 — same tolerance as the sync path
                progress(f"    WARNING: {path.name}: {exc}")
                continue
            extractions.setdefault(doc_id, {})[vid] = _BatchExtraction(
                canonical_text=canonical_text,
                blocks=blocks,
                source_file=path.name,
                extractor_label=extractor_label,
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
below the review threshold this codebase checks against a classification's
confidence: ``clause_classifier.AMBIGUITY_THRESHOLD`` (0.70 — trips
``ClauseClassification.is_ambiguous``). Because this sentinel is stamped on
every LLM-segmented, taxonomy-assigned clause, a reader should treat the
whole cohort as one spot-check, not as hundreds of by-design flags. There is no real
signal yet to distinguish a confident LLM call from a shaky one; see this
constant's docstring for the two follow-up options (per-clause LLM
confidence, cross-version ``normalize_trail`` disagreement) that could
replace the flat default with a calibrated one.
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
    feeding ``build_observations`` below) would then never flag a misclassified LLM-segmented clause (issue #86).
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
    return _template_observations_from_classified(classified)


def _content_exemplars(template_std_by_tid: dict[str, str]) -> dict[str, str]:
    """``{taxonomy_id: our standard's clause text}`` for the content-similarity
    fallback (issue #235): the clause type's complete standard, every template
    node carrying the taxonomy_id joined in document order (issue #242) — the
    text ``our_standard`` carries, so form front matter is no exemplar. Not the
    origin reference (*template_std_nodes_by_tid*), which keeps the front
    matter. Empty in emergent mode (no template standards)."""
    return dict(template_std_by_tid)


def _template_observations_from_classified(
    classified: list[ClassifiedClause],
) -> list[Observation]:
    """Emit per-clause template Observations from already-classified clauses.

    Split out of ``_build_template_observations`` so the agent/LLM
    segmentation path (which classifies in the same pass — no
    ``classify_tree``) can reuse the identical observation shape.

    *classified* is the whole template in document order. Form front matter
    (issue #242) is decided here, over EVERY node, before the unclassified
    ones are dropped: an unclassified operative sentence ends the front
    matter, so a fill-in table after it (a fee schedule) stays a clause. The
    verdict is recorded on the observation (``form_front_matter``) and
    persisted, so ``project`` reads the same answer ``mine`` computed.
    """
    front = front_matter_indices([cc.node.text or "" for cc in classified])
    obs: list[Observation] = []
    for i, cc in enumerate(classified):
        if cc.classification.taxonomy_id is None:
            continue
        # Skip classified-but-empty template clauses (e.g. a heading-only node).
        # Emitting one would build an ``OurStandard`` with empty ``text`` and
        # fail projection ("our_standard.text is empty", validator.py) — a
        # clause with no real template text simply contributes no standard, so
        # the deal clause falls back to emergent/negotiable (issue #182).
        if not (cc.node.text or "").strip():
            continue
        # A form's fill-in cover table stays a template observation (issue
        # #242), flagged form_front_matter: it is no standard
        # (template_standards excludes it from our_standard), but it is still
        # the origin reference for our own text struck before signing (issue
        # #216).
        clause_path = cc.node.clause_path or "?"
        obs.append(
            Observation(
                observation_id=f"template/template/{clause_path}",
                taxonomy_id=cc.classification.taxonomy_id,
                text_summary=summarize_clause_text(cc.node.text or ""),
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
                form_front_matter=i in front,
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
    our_party_aliases: list[str] | None = None,
    our_authors: list[str] | None = None,
    template_std_nodes_by_tid: dict[str, list[str]] | None = None,
    party_names: Sequence[str] = (),
) -> list[Observation]:
    """Create observations from a single-version document, diffed against the template.

    Every clause gets the deterministic standard check (issue #220) — see
    ``_assess_deviations_with_standards``.

    A single-version document has no negotiation trail, but it is still
    checked against the canonical template (issue #103): this builds a
    synthetic "unchanged" ``ClauseDiff`` per clause (see
    ``_single_version_clause_diffs``) and runs it through the same
    ``_assess_deviations_with_standards`` path the multi-version "unchanged
    across the negotiation trail" case uses. A clause with no corresponding
    template text (``template_std_by_tid`` has no entry for its taxonomy_id,
    or no template is configured at all — an empty string either way) is not
    standard: there is nothing to be standard against.

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
        diffs,
        template_std_by_tid,
        template_std_nodes_by_tid=template_std_nodes_by_tid,
        party_names=party_names,
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
        our_authors=our_authors,
        # Issue #216: one observation per taxonomy_id from this (only)
        # version's own tree — the deal is the unit of precedent.
        terminal_clauses=classified,
        terminal_version_id=version_id,
        # Issue #220: the reference every observation's `standard` fact is
        # computed against; it also decides deviation.
        standard_text_by_tid=(
            template_std_nodes_by_tid
            if template_std_nodes_by_tid is not None
            else template_std_by_tid
        ),
        party_names=party_names,
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
                search_snippet=raw.get("search_snippet") or "",
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
                standard=raw.get("standard"),
                paper_basis=raw.get("paper_basis"),
                paper_confidence=raw.get("paper_confidence"),
                alignment_confidence=raw.get("x_alignment_confidence"),
                classification_basis=raw.get("x_classification_basis"),
                opened_with=raw.get("opened_with"),
                form_front_matter=bool(raw.get("x_form_front_matter", False)),
            )
        )
    return result


def _observation_classification_basis(
    obs: Observation, classification_by_node: dict[tuple[str, str], ClauseClassification]
) -> str | None:
    """How *obs*'s taxonomy_id was reached (issue #237), or ``None`` if unknown.

    The ``ClauseClassification.basis`` of the node the citation points at
    when that node's own taxonomy_id is the observation's; ``"aligned"`` when
    the observation's taxonomy_id came from its aligned row instead (the
    row's latest member, issues #222/#232); ``"unclassified"`` whenever the
    observation itself carries no taxonomy_id, so ``"aligned"`` only ever
    means "classified via the aligned row"; ``None`` when the cited node is
    not in the classified trees.
    """
    if obs.taxonomy_id is None:
        return "unclassified"
    cit = obs.citation
    if cit.version_id is None:
        return None
    classification = classification_by_node.get((cit.version_id, cit.clause_path or "?"))
    if classification is None:
        return None
    if classification.taxonomy_id != obs.taxonomy_id:
        return "aligned"
    return classification.basis


def _standard_party_names(config: EngineConfig) -> list[str]:
    """Party names neutralized by the deterministic standard check (issue #220).

    Our own aliases plus every configured ``provenance.known_entities`` name
    — the same real names the born-safe entity registry aliases after L4 —
    so neither side's name decides whether a clause is our standard.
    """
    return [*config.provenance.our_party_aliases, *config.provenance.known_entities]


def _assess_deviations_with_standards(
    net_diffs: list[Any],
    template_std_by_tid: dict[str, str],
    *,
    template_std_nodes_by_tid: dict[str, list[str]] | None = None,
    party_names: Sequence[str] = (),
) -> list[Any]:
    """Run the deterministic standard check per taxonomy_id (issue #220).

    Preserves the original diff order in the returned list.

    No judge is consulted and nothing is ever queued. Each row gets the
    deterministic standard check
    (``deviation_classifier.assess_deviations_deterministic``) — "none" for
    our standard text, "substantive" otherwise, ``basis="deterministic"`` —
    against every template node for its taxonomy_id
    (*template_std_nodes_by_tid*, falling back to the joined
    *template_std_by_tid*), with *party_names* neutralized.
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
        nodes: str | list[str] = (
            template_std_nodes_by_tid.get(tid or "", []) if template_std_nodes_by_tid else []
        ) or template_std_by_tid.get(tid or "", "")
        assessed = assess_deviations_deterministic(diffs, nodes, party_names)
        for orig_idx, pair in zip(indices, assessed, strict=True):
            result[orig_idx] = pair

    return [r for r in result if r is not None]


def _attribution_for_diff(
    clause_diff: ClauseDiff,
    tracked_changes: TrackedChanges | None,
    *,
    single_round: bool,
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

    ``single_round`` — ``True`` only when the document has exactly two
    versions (``len(doc_diff.consecutive) == 1``), i.e. the net diff (first
    → signed) IS the one and only negotiation round, so ``tracked_changes``
    (the signed version's side channel) genuinely IS "that round"'s side
    channel. It gates the round-level fallback below (issue #118 fix round
    2, finding 1): for a document with more than two versions, the net diff
    can bundle hunks that actually originated in an EARLIER round, authored
    by someone who never appears in the signed version's own tracked-changes
    session at all (their edits were already accepted into plain text by
    the time the signed version was drafted). Firing the fallback there
    would attribute that earlier round's change to the signed round's sole
    author — confidently wrong, not an honest "unknown". Direct per-hunk
    matching via ``enrich_clause_diff`` is unaffected by this gate: it only
    ever matches a hunk to a specific ``TrackedChange`` record actually
    co-located with that clause, so a real match still returns regardless
    of how many rounds the net diff spans.

    Returns ``None`` when there is no side-channel at all (PDF/RTF, a clean
    DOCX with no ``w:ins``/``w:del`` elements, or a DOCX ``python-docx``
    could not open — see ``_compute_doc_result``'s ``tracked_by_vid``
    comment; this applies equally regardless of segmentation mode since
    issue #85), when the diff has no hunks (added/removed/unchanged
    clauses), or when no hunk matched a tracked change closely enough (see
    ``tracked_changes_overlay._MATCH_THRESHOLD``) AND the round-level
    fallback tier below also declines to fire (either because it isn't
    ``single_round`` or because ``tracked_changes`` carries more than one
    distinct author).

    Round-level fallback (issue #118): when ``enrich_clause_diff`` finds no
    per-hunk match at all for this diff AND ``single_round`` is ``True``,
    ``tracked_changes_overlay.round_level_fallback_attribution`` gets one
    last try — it fires only when ``tracked_changes`` (this version's
    ENTIRE side channel, not just candidates near this clause) carries
    exactly one distinct author, in which case every real content change in
    the round is attributable to them without per-hunk matching. This is
    deliberately layered ON TOP of ``enrich_clause_diff`` rather than folded
    into it: ``enrich_clause_diff`` has its own direct unit-test suite built
    on single-tracked-change fixtures asserting "no match" for dissimilar/
    wrong-clause text, and those fixtures are single-author by construction
    — folding the fallback in there would flip every one of those
    assertions. Keeping it here means ``enrich_clause_diff`` stays a pure
    per-hunk matcher and this (real observation/attribution) call site is
    the only place the coarser round-level guess can surface.
    """
    if tracked_changes is None or not clause_diff.hunks:
        return None
    enriched = enrich_clause_diff(clause_diff, tracked_changes)
    matched = next((eh.enrichment for eh in enriched if eh.enrichment is not None), None)
    if matched is not None:
        return matched
    if not single_round:
        return None
    return round_level_fallback_attribution(clause_diff.hunks[0], tracked_changes)


def _classification_confidence_for_diff(
    clause_diff: ClauseDiff,
    conf_by_version_path: dict[str, dict[str, float]],
) -> float | None:
    """Classification confidence for one net ``ClauseDiff`` (issue #65).

    ``conf_by_version_path`` is ``{version_id: {clause_path: confidence}}``,
    built from EVERY version's classified clauses (``classified_by_version``),
    not just the signed version's. A removed clause's ``clause_path`` is a
    BEFORE-side path read from an earlier draft (the net diff's base
    version), not from the signed version — looking it up in a map built
    only from the signed version's classified clauses either misses
    (``None``) or, worse, collides with an unrelated signed clause that
    happens to occupy the same path number after renumbering, silently
    attaching that clause's confidence to the removed clause's observation.
    That collision can suppress (or spuriously add) a low-confidence review
    flag for a genuinely uncertain removed-clause observation.

    Selects the side (after, else before) the same way
    ``observation_builder.build_observations`` already selects
    ``citation.version_id``/``char_span`` — a removed clause's citation
    comes from the before side, so its confidence must be read from that
    same side's version, never the signed/last version this diff batch is
    filed under.
    """
    vid: str | None
    path: str | None
    if clause_diff.clause_path_after is not None:
        vid, path = clause_diff.clause_version_after, clause_diff.clause_path_after
    else:
        vid, path = clause_diff.clause_version_before, clause_diff.clause_path_before
    if vid is None or path is None:
        return None
    return conf_by_version_path.get(vid, {}).get(path)


def _json_text(data: Any) -> str:
    """The exact text :func:`_atomic_json_write` writes for *data*."""
    return json.dumps(data, indent=2, ensure_ascii=False) + "\n"


def _atomic_json_write(data: Any, path: Path) -> None:
    """Atomically write *data* as JSON to *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(_json_text(data), encoding="utf-8")
    os.replace(tmp, path)


def _build_version_alias_map(version_ingest: Any) -> dict[str, str]:
    """Exact raw-stem -> ordinal-label map for one document's versions (issue #143).

    ``version_ingest`` is one entry per version FILE FOUND, in discovery
    order (see ``_build_version_ingest_list``'s docstring), so entry *i*'s
    stem is given the stable label ``f"v{i + 1}"``, which is how the playbook's
    ``version_ingest[].version`` names it.

    Unlike the whole-word ``known_entities`` substring match
    ``_alias_version_field`` otherwise falls back to, this doesn't depend on
    the counterparty's name being enumerated (correctly, completely, and in
    a form that appears as whitespace-separated words in the filename) in
    ``config.provenance.known_entities`` — it maps every stem this document
    actually has, so an unlisted, abbreviated, or concatenated name embedded
    in a filename is no longer a coverage gap (issue #143's evidence: a
    version stem with no internal whitespace at all, so no known-entity
    substring match could ever have caught it).
    """
    out: dict[str, str] = {}
    if not isinstance(version_ingest, list):
        return out
    for i, vi in enumerate(version_ingest):
        stem = vi.get("version") if isinstance(vi, dict) else None
        if isinstance(stem, str):
            out[stem] = f"v{i + 1}"
    return out


def _alias_version_field(
    value: Any,
    known_entities: list[str],
    registry: EntityRegistry,
    version_alias_map: dict[str, str] | None = None,
) -> Any:
    """Alias a citation/manifest ``version``/``version_id`` field (issue #182).

    These are staged filename stems that embed the counterparty name (e.g.
    "01__… Oglethorpe University 6.14.23"), so run them through the whole-word
    text pseudonymizer. A plain ordinal (``int``) carries no name and is
    returned unchanged.

    *version_alias_map* (issue #143), when given, is an EXACT raw-stem ->
    ordinal-label map (see :func:`_build_version_alias_map`) for the SAME
    document this value belongs to. It is tried FIRST and, on a hit, wins
    outright — an exact lookup against the document's own known stems cannot
    miss the way whole-word substring matching against ``known_entities``
    can. Only when *value* isn't one of that document's known stems (or no
    map was given) does this fall back to the substring pseudonymizer below,
    as defense in depth for any residual name fragment reaching this
    function some other way (e.g. a caller with no manifest-derived map to
    give, like ``citation.version``/``citation.version_id``).
    """
    if isinstance(value, str):
        if version_alias_map is not None and value in version_alias_map:
            return version_alias_map[value]
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
    observations: list[Observation],
    known_entities: list[str],
    registry: EntityRegistry,
    version_alias_by_doc: dict[str, dict[str, str]] | None = None,
) -> list[Observation]:
    """Return *observations* with clause text and every document id aliased.

    Rewrites ``text_summary``, ``full_text``, ``search_snippet``,
    ``citation.document_id``, and the document-id segment of
    ``observation_id`` for every known entity name (issues #153, #182, #95)
    — ``Observation``/``ObservationCitation`` are frozen dataclasses, so a
    fresh copy is built per row via ``dataclasses.replace`` rather than
    mutated in place. Pseudonymizing ``observation_id`` here (not just the
    citation) keeps the id free of raw counterparty names and keeps it
    consistent with the aliased ``citation.document_id``.

    ``search_snippet`` is still UNTRUNCATED at this point (see
    ``Observation.search_snippet``'s docstring) — it is pseudonymized here
    alongside ``text_summary``/``full_text`` while whole, and only capped to
    its final short-phrase length afterward, by
    ``truncate_search_snippets``. Never reorder those two steps: truncating
    first can bisect a known-entity name mid-word and defeat this function's
    whole-word alias match, leaking the fragment.

    *version_alias_by_doc* (issue #143): the ``{raw document_id: version_alias_map}``
    dict returned by :func:`_pseudonymize_corpus_documents`, keyed by the
    SAME raw ``citation.document_id`` this observation carries (looked up
    BEFORE that field is aliased below). Passing the matching per-document
    map into :func:`_alias_version_field` for ``citation.version``/
    ``citation.version_id`` gives them the same exact-stem-match coverage
    ``_pseudonymize_trail``/``_pseudonymize_corpus_documents`` already have,
    instead of relying solely on the whole-word ``known_entities`` substring
    fallback, which misses a squashed (no-whitespace) counterparty name.
    """
    out: list[Observation] = []
    for obs in observations:
        version_alias_map = (
            version_alias_by_doc.get(obs.citation.document_id)
            if version_alias_by_doc is not None
            else None
        )
        new_citation = ObservationCitation(
            document_id=pseudonymize_document_id(
                obs.citation.document_id, known_entities, registry
            ),
            version=_alias_version_field(
                obs.citation.version, known_entities, registry, version_alias_map
            ),
            clause_path=obs.citation.clause_path,
            char_span=obs.citation.char_span,
            version_id=_alias_version_field(
                obs.citation.version_id, known_entities, registry, version_alias_map
            ),
        )
        out.append(
            dataclasses.replace(
                obs,
                observation_id=_pseudonymize_observation_id(
                    obs.observation_id, known_entities, registry
                ),
                text_summary=pseudonymize_text(obs.text_summary, known_entities, registry),
                full_text=pseudonymize_text(obs.full_text, known_entities, registry),
                search_snippet=pseudonymize_text(obs.search_snippet, known_entities, registry),
                citation=new_citation,
            )
        )
    return out


def _pseudonymize_trail(
    trail: dict[str, Any],
    known_entities: list[str],
    registry: EntityRegistry,
    version_alias_map: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Return a copy of *trail* with every raw counterparty name aliased (issue #182, #34).

    Mirrors ``_pseudonymize_observations``/``_pseudonymize_corpus_documents``
    for the trail store: ``document_id`` matches the aliased
    ``citation.document_id`` that ``inspect`` joins on, and ``ordered_versions``,
    ``signed_version``, and ``pairwise_distances[].from``/``to`` (spread in
    from ``version_order.to_dict()``, all staged filename stems) are aliased
    the same way ``version_ingest``/``signed_version`` are for the manifest.
    ``reversals`` entries (``ReversalRecord.to_dict()``) carry the same raw
    version stems in ``version_inserted``/``version_removed`` plus raw clause
    text in ``proposed_text`` — both are aliased too, as are the keys of
    ``version_timestamps`` (issue #221, raw version stems), so no raw
    counterparty name survives anywhere in the trail body.

    *version_alias_map* (issue #143): the SAME document's exact raw-stem ->
    ordinal-label map (see :func:`_build_version_alias_map`), built by the
    caller from this document's ``corpus_manifest.json`` ``version_ingest``
    entry so the trail's version labels ("v1"/"v2"/…) match the manifest's —
    a reader cross-referencing the two artifacts sees the same labels for
    the same physical file. See ``_alias_version_field`` for the exact-map-
    first, substring-fallback precedence.
    """
    if not trail.get("document_id"):
        return dict(trail)
    new = dict(trail)
    new["document_id"] = pseudonymize_document_id(trail["document_id"], known_entities, registry)
    if isinstance(new.get("ordered_versions"), list):
        new["ordered_versions"] = [
            _alias_version_field(v, known_entities, registry, version_alias_map)
            for v in new["ordered_versions"]
        ]
    if isinstance(new.get("signed_version"), str):
        new["signed_version"] = _alias_version_field(
            new["signed_version"], known_entities, registry, version_alias_map
        )
    if isinstance(new.get("pairwise_distances"), list):
        new["pairwise_distances"] = [
            {
                **pd,
                "from": _alias_version_field(
                    pd.get("from"), known_entities, registry, version_alias_map
                ),
                "to": _alias_version_field(
                    pd.get("to"), known_entities, registry, version_alias_map
                ),
            }
            if isinstance(pd, dict)
            else pd
            for pd in new["pairwise_distances"]
        ]
    if isinstance(new.get("reversals"), list):
        new["reversals"] = [
            {
                **r,
                "version_inserted": _alias_version_field(
                    r.get("version_inserted"), known_entities, registry, version_alias_map
                ),
                "version_removed": _alias_version_field(
                    r.get("version_removed"), known_entities, registry, version_alias_map
                ),
                "proposed_text": pseudonymize_text(r["proposed_text"], known_entities, registry)
                if isinstance(r.get("proposed_text"), str)
                else r.get("proposed_text"),
            }
            if isinstance(r, dict)
            else r
            for r in new["reversals"]
        ]
    if isinstance(new.get("version_timestamps"), dict):
        # Issue #221: keyed by the same raw version stems.
        new["version_timestamps"] = {
            _alias_version_field(v, known_entities, registry, version_alias_map): ts
            for v, ts in new["version_timestamps"].items()
        }
    return new


def _pseudonymize_clause_node(
    node: dict[str, Any], known_entities: list[str], registry: EntityRegistry
) -> dict[str, Any]:
    """Return a copy of one ``ClauseNode.to_dict()`` with ``heading``/``text`` aliased.

    Recurses into ``children`` — see ``_pseudonymize_clause_tree`` below.
    """
    new = dict(node)
    if isinstance(new.get("heading"), str):
        new["heading"] = pseudonymize_text(new["heading"], known_entities, registry)
    if isinstance(new.get("text"), str):
        new["text"] = pseudonymize_text(new["text"], known_entities, registry)
    new["children"] = [
        _pseudonymize_clause_node(c, known_entities, registry) for c in new.get("children", [])
    ]
    return new


def _pseudonymize_clause_tree(
    tree_dict: dict[str, Any],
    known_entities: list[str],
    registry: EntityRegistry,
    version_alias_map: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Return a copy of one ``ClauseTree.to_dict()`` with every raw counterparty name aliased.

    Mirrors ``_pseudonymize_trail`` (issue #139): ``document_id`` is aliased
    the same way trail/corpus_manifest entries are — so the aliased id this
    returns matches the ``out_dir / "normalized" / <document_id>`` lookup
    a reader joins on the aliased document_id from the compiled
    playbook — and every node's ``heading``/``text`` (headings AND clause
    bodies) is aliased too, so no raw counterparty name survives in
    normalized/ the way it doesn't in observations.jsonl or trail/.

    ``version`` and ``source_file`` (issue #139 review round 3) are BOTH raw
    staged-filename material too — ``version`` is the exact stem
    (``ClauseTree.version = vid``, see ``_compute_doc_result``'s per-version
    loop) and ``source_file`` is that stem's full filename with extension
    (``path.name``) — and the codebase's own ``_alias_version_field``
    docstring documents that a staged version filename routinely embeds the
    raw counterparty name (e.g. "01__… Oglethorpe University 6.14.23"). Both
    are aliased the same way ``citation.version``/``citation.version_id``
    already are: an exact match against *version_alias_map* (issue #143's
    ``{raw stem: ordinal label}`` map for this document, built by
    ``_build_version_alias_map``) wins outright; ``source_file`` never hits
    that map exactly (it carries an extension the map's stems don't), so it
    falls through to the same whole-word ``known_entities`` substring
    pseudonymization ``_alias_version_field`` already provides as defense in
    depth for exactly this case. The caller then uses this aliased
    ``version`` (not the raw ``vid`` dict key) to build the output filename,
    so the on-disk name never carries the raw stem either.
    """
    if not tree_dict.get("document_id"):
        return dict(tree_dict)
    new = dict(tree_dict)
    new["document_id"] = pseudonymize_document_id(
        tree_dict["document_id"], known_entities, registry
    )
    new["version"] = _alias_version_field(
        tree_dict.get("version"), known_entities, registry, version_alias_map
    )
    new["source_file"] = _alias_version_field(
        tree_dict.get("source_file"), known_entities, registry, version_alias_map
    )
    new["nodes"] = [
        _pseudonymize_clause_node(n, known_entities, registry) for n in tree_dict.get("nodes", [])
    ]
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
    moves: list[RoundMove],
    known_entities: list[str],
    registry: EntityRegistry,
    version_alias_by_doc: dict[str, dict[str, str]] | None = None,
) -> list[RoundMove]:
    """Alias raw entity names out of round moves (issue #177) — same born-safe
    pass ``_pseudonymize_observations`` applies, covering ``document_id``,
    the citation, and ``change_summary`` (which quotes clause text).

    *version_alias_by_doc* (issue #143): same ``{raw document_id: version_alias_map}``
    dict ``_pseudonymize_observations`` accepts — looked up against the
    citation's raw ``document_id`` before it is aliased, and threaded into
    ``citation.version``/``citation.version_id`` for the same exact-stem
    coverage described there.
    """
    out: list[RoundMove] = []
    for move in moves:
        version_alias_map = (
            version_alias_by_doc.get(move.citation.document_id)
            if version_alias_by_doc is not None
            else None
        )
        new_citation = ObservationCitation(
            document_id=pseudonymize_document_id(
                move.citation.document_id, known_entities, registry
            ),
            version=_alias_version_field(
                move.citation.version, known_entities, registry, version_alias_map
            ),
            clause_path=move.citation.clause_path,
            char_span=move.citation.char_span,
            version_id=_alias_version_field(
                move.citation.version_id, known_entities, registry, version_alias_map
            ),
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
) -> tuple[list[dict[str, Any]], dict[str, dict[str, str]]]:
    """Return *corpus_documents* with each entry's ``document_id`` aliased (issue #153).

    ``corpus_documents`` (``corpus_manifest.json``) feeds directly into
    ``playbook.opf.json``'s ``documents`` field (see ``playbook_assembler`` —
    every key survives except a small schema-only strip-list, e.g.
    ``version_ingest[].reason``, issue #81), so its ``document_id`` must
    carry the same alias as the matching observations'
    ``citation.document_id`` for the compiled OPF to be consistent, not just
    the observation store.

    Also returns a ``{raw document_id: version_alias_map}`` dict (issue
    #143) — one exact raw-stem -> ordinal-label map per document (see
    :func:`_build_version_alias_map`), keyed by the document's RAW,
    pre-alias id (matching ``all_trails``' key) so the caller can pass the
    SAME per-document map into :func:`_pseudonymize_trail`, keeping the
    trail's version labels consistent with the manifest's.
    """
    out = []
    version_alias_by_doc: dict[str, dict[str, str]] = {}
    for doc in corpus_documents:
        raw_doc_id = doc.get("document_id")
        version_alias_map = _build_version_alias_map(doc.get("version_ingest"))
        if isinstance(raw_doc_id, str):
            version_alias_by_doc[raw_doc_id] = version_alias_map

        new_doc = dict(doc)
        if "document_id" in new_doc:
            new_doc["document_id"] = pseudonymize_document_id(
                new_doc["document_id"], known_entities, registry
            )
        # version_ingest / signed_version embed the staged filename stem, which
        # carries the counterparty name (issue #182) — alias those too so the
        # manifest embedded in playbook.opf.json holds no raw names. Issue
        # #143: pass the exact per-document version_alias_map so this no
        # longer depends on the name appearing, whole-word, in
        # known_entities — see _alias_version_field.
        if isinstance(new_doc.get("version_ingest"), list):
            new_doc["version_ingest"] = [
                {
                    **vi,
                    "version": _alias_version_field(
                        vi.get("version"), known_entities, registry, version_alias_map
                    ),
                }
                if isinstance(vi, dict)
                else vi
                for vi in new_doc["version_ingest"]
            ]
        if isinstance(new_doc.get("signed_version"), str):
            new_doc["signed_version"] = _alias_version_field(
                new_doc["signed_version"], known_entities, registry, version_alias_map
            )
        out.append(new_doc)
    return out, version_alias_by_doc


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


def _build_version_ingest_list(
    version_files: list[Path], version_ingest: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    """One entry per version FILE FOUND, in discovery order (issue #89).

    A version absent from *version_ingest* — its try/except in
    ``_compute_doc_result``'s per-version loop never ran, including a version
    that loop had not reached yet when a ``SegmentationQAError`` aborted it
    (issue #83) — gets the "not attempted" default fill. Shared by
    ``_compute_doc_result``'s normal return path and
    ``_build_quarantine_corpus_doc`` below so both build this list the same
    way: a failed/unattempted extraction or segmentation is a durable
    manifest record, not just a progress-line WARNING a cache hit wouldn't
    even re-print.
    """
    return [
        {
            "version": vf.stem,
            **version_ingest.get(
                vf.stem,
                {
                    "status": "unknown",
                    "error": "not attempted",
                    "extractor": None,
                    "reason": None,
                },
            ),
        }
        for vf in version_files
    ]


def _build_quarantine_corpus_doc(
    doc_id: str,
    version_files: list[Path],
    version_ingest: dict[str, dict[str, Any]],
    sha256_by_vid: dict[str, str],
    media_type_by_vid: dict[str, str],
) -> dict[str, Any]:
    """Build a PARTIAL ``corpus_documents`` entry for a document quarantined
    mid-L1 by a ``SegmentationQAError`` (issue #83).

    Attached to the exception (see ``SegmentationQAError.partial_corpus_doc``)
    by ``_compute_doc_result``'s per-version ``except`` block and read back by
    ``mine_corpus``'s quarantine handler, so a document whose L1 loop never
    reaches the scope gate/normal ``corpus_doc`` construction below still
    gets *some* durable record in ``corpus_manifest.json`` — before this fix,
    only ``quarantine.json`` recorded the failure, at document granularity,
    with no per-version status and no extractor.

    This is explicitly a SNAPSHOT of how far L1 got, not a fabricated
    complete document record — every field below is chosen so nothing
    downstream can mistake it for a normal, fully-scope-gated document:

    - ``in_scope=False`` with a fixed, generic ``scope_rationale`` explaining
      the quarantine (never a real scope decision — the scope gate requires a
      successfully-ingested first version and never runs here). Both the
      schema's own ``in_scope==false`` conditional and
      ``validator._check_out_of_scope_rationale`` (OPF §3.6) already REQUIRE
      exactly this pairing for any non-in-scope document, so this satisfies
      that existing fail-closed gate rather than needing to loosen it.
      Deliberately does NOT interpolate the QA error text: ``scope_rationale``
      passes straight through
      ``playbook_assembler._sanitize_corpus_documents_for_schema`` into the
      PUBLISHED playbook untouched (that function only rewrites
      ``version_ingest``) — keeping it a fixed string is one less place that
      would need auditing for leaked content.
    - ``x_quarantined=True`` — the sanctioned ``x_`` extension point
      (spec/playbook.schema-0.5.json's ``corpus.documents.items``; NOT
      stripped by ``playbook_assembler._sanitize_corpus_documents_for_schema``,
      which only rewrites ``version_ingest`` entries) gives a downstream
      consumer an explicit, unambiguous way to recognize "quarantined
      partial record" without parsing ``scope_rationale`` text — on top of
      ``quarantine.json`` (unchanged shape by this fix, beyond aliasing its
      ``document_id`` — see ``mine_corpus``) remaining the canonical list of
      quarantined document_ids.
    - ``versions``/``versions_mined`` count only the versions THIS loop
      already recorded an "ok" ``version_ingest`` row for before the
      failure — same "versions successfully ingested, not files found"
      meaning ``_compute_doc_result``'s normal corpus_doc uses, and the same
      "versions_mined > 0 with zero observations" shape an out-of-scope
      document already has today (a scope-gated-out document contributes no
      observations either) — so nothing downstream needs new handling for
      that combination.
    - ``version_ingest`` (via ``_build_version_ingest_list``) carries the
      failing version's true ``status="failed"``/QA ``error``/``extractor``/
      ``reason`` (issue #81 shape) plus an "ok" row for every version
      already mined and a "not attempted" row for every version this loop
      never reached — this is the field the ticket's acceptance criteria is
      actually about. ``error`` is a message built only from gate
      names/offsets/lengths (see ``SegmentationQAError``'s docstring) —
      never a raw slice of source content — so it is safe to persist here
      unbounded, for operator diagnosis.
    - ``version_files`` (content addresses) for every version file THIS loop
      has reached so far — ``sha256_by_vid``/``media_type_by_vid`` are
      populated up front for the current file before extraction is
      attempted each iteration, so a version not yet reached simply has no
      entry yet (guarded below, same pattern the out-of-scope path uses).
    - Zero observations: this document contributes none (L2-L4 never ran),
      so no clause position/floor/posture content can ever be attributed to
      a quarantined document regardless of what its corpus_doc entry says —
      ``mine_corpus`` never extends ``all_observations`` for it.
    """
    versions_mined = sum(1 for v in version_ingest.values() if v.get("status") == "ok")
    return {
        "document_id": doc_id,
        "provenance": "counterparty_paper",
        "in_scope": False,
        "scope_rationale": (
            "QA-quarantined during extraction/segmentation — scope was never "
            "determined. See corpus_manifest.json's version_ingest for the "
            "QA error."
        ),
        "versions": versions_mined,
        "versions_mined": versions_mined,
        "versions_found": len(version_files),
        "version_ingest": _build_version_ingest_list(version_files, version_ingest),
        "version_files": [
            {
                "version": i + 1,
                "sha256": sha256_by_vid[vf.stem],
                "media_type": media_type_by_vid[vf.stem],
            }
            for i, vf in enumerate(version_files)
            if vf.stem in sha256_by_vid
        ],
        "x_quarantined": True,
    }


# Signature of the L1 cache hook _collect_l1 takes (issue #219): given a
# version file and the closure that computes its L1 record, return the record
# — from mine_corpus's per-version stage cache, or by calling the closure.
L1Fetch = Callable[[Path, Callable[[], dict[str, Any]]], dict[str, Any]]


@dataclass
class _L1State:
    """Everything L1 (ingest + segment, per version) hands to L2-L4 (issue #219).

    Built by :func:`_collect_l1` from one JSON record per version — fresh or
    replayed from the per-version stage cache, the SAME record either way, so
    a warm run and a cold run feed L2-L4 identical inputs.
    ``l1_fingerprint`` digests every version's source sha256 plus its record
    (or its failure entry) and keys the per-document L2-L4 layer: an L1
    recompute that reproduces the same records from the same bytes still
    replays L2-L4, but a bytes-only change misses it (the cached result embeds
    each version's sha256).
    """

    version_trees: dict[str, ClauseTree] = dataclasses.field(default_factory=dict)
    version_tree_dicts: dict[str, dict[str, Any]] = dataclasses.field(default_factory=dict)
    unstripped_trees: dict[str, ClauseTree] = dataclasses.field(default_factory=dict)
    llm_taxonomy_by_path: dict[str, dict[str, str | None]] = dataclasses.field(default_factory=dict)
    tracked_by_vid: dict[str, TrackedChanges | None] = dataclasses.field(default_factory=dict)
    timestamp_by_vid: dict[str, str | None] = dataclasses.field(default_factory=dict)
    version_ingest: dict[str, dict[str, Any]] = dataclasses.field(default_factory=dict)
    sha256_by_vid: dict[str, str] = dataclasses.field(default_factory=dict)
    media_type_by_vid: dict[str, str] = dataclasses.field(default_factory=dict)
    l1_fingerprint: str = ""


def _l1_version_record(
    vf: Path,
    doc_id: str,
    vid: str,
    *,
    config: EngineConfig,
    taxonomy_ids: list[str],
    progress: Callable[[str], None],
    label_out: list[ExtractorLabel],
    use_llm_segmentation: bool = False,
    llm_segment_fn: SegmentFn | None = None,
    batch_seg_nodes: dict[str, list[SegNode]] | None = None,
    batch_extractions: dict[str, _BatchExtraction] | None = None,
    segmentation_cache: SegmentationVerdictCache | None = None,
    extraction_cache: ExtractionCache | None = None,
    refresh_extraction: bool = False,
) -> dict[str, Any]:
    """L1 for ONE version file → a JSON-serialisable record (issue #219).

    The record is what the per-version stage cache stores: the UNSTRIPPED
    clause tree (signature-block stripping is cheap and re-run by
    :func:`_collect_l1` on every replay), the LLM path's per-clause taxonomy
    assignments, the tracked-changes side channel, the version's authorship
    timestamp and the extractor that produced the text. It depends only on
    the source file and the segmentation/extraction configuration — never on
    the taxonomy's wording, the template or the judges — which is what lets a
    change to any of those replay L1 and recompute only L2-L4.

    Raises on any per-version failure (an empty tree from a non-empty file
    included); nothing is cached for a failed version. *label_out* receives
    the resolved extractor label as soon as one exists, so the caller's
    failure record still names the extractor.
    """
    tax_by_path: dict[str, str | None] | None = None
    if use_llm_segmentation and batch_seg_nodes is not None and vid in batch_seg_nodes:
        extraction = (batch_extractions or {})[vid]
        tree, tax_by_path = _ground_batch_result(
            doc_id, vid, extraction, batch_seg_nodes[vid], taxonomy_ids
        )
        extractor_label = extraction.extractor_label
        label_out.append(extractor_label)
        tracked, units = _llm_tracked_changes(vf, progress)
        # extraction.blocks (issue #118) is the SAME batch-pre-pass
        # extraction that produced this version's tree — no second
        # extract_blocks call needed on this branch, unlike the
        # synchronous one below.
        tracked = _bridge_tracked_changes_if_needed(
            tracked, units, extraction.blocks, extractor_label
        )
    elif use_llm_segmentation:
        tree, tax_by_path, extractor_label = _llm_segment_file(
            vf,
            doc_id,
            vid,
            taxonomy_ids,
            llm_segment_fn,
            segmentation_cache,
            model=config.segmentation.model,
            extraction_cache=extraction_cache,
            refresh_extraction=refresh_extraction,
            extractor=config.extraction.extractor,
        )
        label_out.append(extractor_label)
        tracked, units = _llm_tracked_changes(vf, progress)
        # Bridge (issue #118) only when actually needed: a non-legacy
        # extractor AND at least one change carries a char_span worth
        # translating. Most DOCX versions carry no tracked changes at
        # all, so this re-extraction is the exception, not the rule
        # (see _extract_blocks_for_bridge's cache-hit note).
        bridge_blocks: list[Block] | None = None
        if (
            tracked is not None
            and any(c.char_span is not None for c in tracked.changes)
            and extractor_label is not None
            and extractor_label != "legacy"
        ):
            bridge_blocks = _extract_blocks_for_bridge(
                vf, extraction_cache, False, config.extraction.extractor
            )
        tracked = _bridge_tracked_changes_if_needed(tracked, units, bridge_blocks, extractor_label)
    else:
        raw_tree, tracked = _ingest_file_tracked(vf, doc_id, vid)
        tree = segment(raw_tree)

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
        # extraction failure (_collect_l1's `except Exception` clause)
        # and never reaches the scope gate as the representative
        # version.
        raise ValueError(
            "ingest produced an empty clause tree from a non-empty "
            "source file — treating as an extraction failure"
        )

    label = label_out[-1] if label_out else None
    record: dict[str, Any] = {
        "tree": tree.to_dict(),
        "tax_by_path": tax_by_path,
        "tracked": tracked.to_dict() if tracked is not None else None,
        "timestamp": document_timestamp(vf, tracked),
        # The extractor version_ingest records: the file suffix on the
        # deterministic path, the REAL resolved label on the LLM path
        # (issue #81/#129). Only the closed-enum ``reason`` travels — never
        # ExtractorLabel.detail, which embeds the absolute source path.
        "extractor": label.extractor if label is not None else vf.suffix.lower().lstrip("."),
        "reason": label.reason if label is not None else None,
    }
    if label is not None and label.timed_out:
        # Issue #231: this version's text came from a fallback after a
        # docling timeout — a property of THIS run, not of the source bytes.
        # The flag keeps the record out of the per-version stage cache (see
        # mine_corpus's _make_l1_fetch) and the deal out of the L2-L4 cache
        # (_collect_l1 reports the version as timed out), so the next run
        # retries docling instead of replaying the legacy extraction. Set
        # only when true, so an ordinary record's shape is unchanged.
        record[_L1_TIMED_OUT_KEY] = True
    return record


def _collect_l1(
    doc_id: str,
    version_files: list[Path],
    *,
    config: EngineConfig,
    taxonomy_ids: list[str],
    progress: Callable[[str], None],
    timed_out_versions: list[str] | None = None,
    fetch_l1: L1Fetch | None = None,
    **l1_kwargs: Any,
) -> _L1State:
    """Run (or replay) L1 for every version of one document (issue #219).

    *fetch_l1*, when given, is the per-version stage-cache hook (see
    :data:`L1Fetch`); without it every version is computed. Per-version
    failure handling is unchanged from before the L1/L2-L4 split: a
    ``SegmentationQAError`` propagates (the document is quarantined) carrying
    a partial corpus_doc snapshot; any other exception records a failed
    ``version_ingest`` entry and the version is skipped.
    """
    state = _L1State()
    fingerprint_parts: list[dict[str, Any]] = []
    for vf in version_files:
        vid = vf.stem
        state.sha256_by_vid[vid] = file_sha256(vf)
        state.media_type_by_vid[vid] = _MEDIA_TYPES.get(
            vf.suffix.lower(), "application/octet-stream"
        )
        # Per-version extractor recorded in version_ingest/corpus_manifest.json
        # (issue #129). The deterministic path always uses the file suffix
        # (unchanged). The LLM-segmentation path (sync, batch pre-pass, or a
        # segmentation-cache hit — all funnel through extraction.extract_blocks)
        # gets the REAL label extract_blocks/_llm_segment_file/
        # _ground_batch_result resolved (issue #81) — the record carries it,
        # and label_out has it for a failure that happens after it resolved.
        # `extractor` starts as the deterministic-path default so the
        # except-Exception branch below still has a sane fallback value for a
        # version whose LLM-path resolution never got far enough to produce a
        # real label (e.g. extract_blocks itself raised).
        extractor = vf.suffix.lower().lstrip(".")
        label_out: list[ExtractorLabel] = []

        def _compute(
            _vf: Path = vf, _vid: str = vid, _label_out: list[ExtractorLabel] = label_out
        ) -> dict[str, Any]:
            return _l1_version_record(
                _vf,
                doc_id,
                _vid,
                config=config,
                taxonomy_ids=taxonomy_ids,
                progress=progress,
                label_out=_label_out,
                **l1_kwargs,
            )

        try:
            record = fetch_l1(vf, _compute) if fetch_l1 is not None else _compute()
            tree = ClauseTree.from_dict(record["tree"])
            if record.get("tax_by_path") is not None:
                state.llm_taxonomy_by_path[vid] = record["tax_by_path"]
            state.tracked_by_vid[vid] = (
                TrackedChanges.from_dict(record["tracked"])
                if record.get("tracked") is not None
                else None
            )

            # issue #217: cut the signature block (IN WITNESS WHEREOF,
            # By:/Name:/Title:, signatory names) out of the last clause's
            # text — it is not clause language, and signatories' names are a
            # pseudonymization residue path. The unstripped tree is kept for
            # the L2 detectors that need the block (see unstripped_trees).
            state.unstripped_trees[vid] = tree
            tree, signature_block = strip_signature_block(tree)
            state.timestamp_by_vid[vid] = record.get("timestamp")

            # issue #139: do NOT write to normalized/ here — that used to
            # write raw, pre-pseudonymization content under the RAW doc_id,
            # mid-loop, so a run with known_entities configured left the raw
            # counterparty name in both the directory name and every node's
            # text forever (never stale-cleared, never rewritten on a
            # stage-cache hit — see the born-safe pseudonymization pass
            # below and mine_corpus's "Materialise normalized/ clause trees
            # now" comment, which mirrors trail/'s treatment). Only the
            # serialised dict is captured here, into the cacheable result.
            state.version_tree_dicts[vid] = tree.to_dict()
            state.version_trees[vid] = tree
            state.version_ingest[vid] = {
                "status": "ok",
                "error": None,
                "extractor": record["extractor"],
                # None on the deterministic path (never a fallback) and
                # whenever docling ran clean with no degradation — see
                # ExtractorLabel.reason (issue #81).
                "reason": record.get("reason"),
                # issue #217: where the stripped signature block sat in this
                # version's normalized text ([start, end) — same coordinates
                # as ClauseNode.char_span), or None when no block was found
                # (or its offsets could not be related to the tree). Engine-
                # internal: corpus_manifest.json carries it; the frozen
                # OPF 0.5 schema's version_ingest (additionalProperties:
                # false) does not, so playbook_assembler's
                # _VERSION_INGEST_SCHEMA_KEYS strips it from the published
                # playbook.
                "signature_block_span": (
                    list(signature_block.char_span)
                    if signature_block is not None and signature_block.char_span is not None
                    else None
                ),
            }
            fingerprint_parts.append(
                {
                    "name": vf.name,
                    # The source bytes' content address: L2-L4 copies it into
                    # corpus_doc["version_files"][].sha256 (OPF §4), so a
                    # bytes-only change that leaves the L1 record identical
                    # must still miss L2-L4.
                    "sha256": state.sha256_by_vid[vid],
                    "record": {k: v for k, v in record.items() if k != _L1_DEPS_KEY},
                }
            )
            if timed_out_versions is not None and record.get(_L1_TIMED_OUT_KEY):
                # Issue #231: this version's text came from a fallback after
                # a docling timeout. ExtractionCache already refused to store
                # it and the L1 layer refuses too; storing the deal result
                # would replay the legacy extraction one layer up and the
                # next run would never retry docling — pinning the trail as
                # mixed-extractor.
                timed_out_versions.append(vid)
        except SegmentationQAError as exc:
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
            #
            # issue #83: record THIS version's failure into version_ingest —
            # same shape as the "ok" row above and the generic-failure row
            # below — so the extractor that produced the failing stream isn't
            # erased. That alone is not enough for it to reach
            # corpus_manifest.json: this function's own locals (including
            # version_ingest) are discarded the instant this exception
            # unwinds past its caller, so a PARTIAL corpus_doc snapshot
            # (however far L1 got — this vid's failure, every earlier vid's
            # "ok" row, every later vid's not-yet-attempted default) is built
            # now and attached to the exception for mine_corpus's quarantine
            # handler to append to corpus_documents (see
            # SegmentationQAError.partial_corpus_doc and
            # _build_quarantine_corpus_doc). The document is still
            # quarantined — this restores audit visibility, it does not
            # change the fail-loud contract above. ``str(exc)`` is safe to
            # persist unbounded for THIS class's own coverage/reconstruction/
            # tree/taxonomy gates (proven at their own source to carry only
            # gate names, node/clause identifiers, and OFFSETS/LENGTHS — see
            # that class's docstring) — but NOT yet for the grounding gate:
            # this ``except`` also catches a ``GroundingError`` wrapped
            # verbatim (segmentation_qa.py's ``run_gates``/
            # ``segment_verify_repair``, e.g. ``f"grounding gate: {exc}"``),
            # whose own 8 raise sites (segmentation_grounding.py:158,160,
            # 162,169,179,181,191,203) interpolate the model's own
            # unconstrained ``node_id``/``start_block_id``/``end_block_id``/
            # ``parent_id`` (no enum/pattern in llm_segmenter.py's schema,
            # unlike ``taxonomy_id``). That gap is real and open — reported,
            # not fixed, in issue #98's rescope comment, which records the
            # verified line list for its own follow-up ticket (not yet
            # filed); this diff does not close it.
            extractor_label = label_out[-1] if label_out else None
            if extractor_label is not None:
                extractor = extractor_label.extractor
            state.version_ingest[vid] = {
                "status": "failed",
                "error": str(exc),
                "extractor": extractor,
                "reason": extractor_label.reason if extractor_label is not None else None,
            }
            exc.partial_corpus_doc = _build_quarantine_corpus_doc(
                doc_id,
                version_files,
                state.version_ingest,
                state.sha256_by_vid,
                state.media_type_by_vid,
            )
            raise
        except Exception as exc:  # noqa: BLE001
            progress(f"    WARNING: {vf.name}: {exc}")
            extractor_label = label_out[-1] if label_out else None
            if extractor_label is not None:
                extractor = extractor_label.extractor
            state.version_ingest[vid] = {
                "status": "failed",
                # NEVER str(exc) here (issue #98) — this is a catch-all for
                # every OTHER exception extraction/ingest can raise (the
                # SegmentationQAError branch above is handled separately —
                # see its own comment above for exactly what is, and is NOT,
                # proven safe to persist there; the grounding gate it can
                # wrap is a known-open gap, not a proven-safe sibling):
                # ExtractionError/DocxIngesterError/PdfIngesterError/
                # RtfIngesterError messages routinely embed the absolute source
                # path (extraction.py's own "file not found: {path}"/
                # "extraction yielded no text: {path}"/docling messages
                # interpolate it directly; python-docx's PackageNotFoundError
                # does too, confirmed empirically), and SegmentationLLMError can
                # embed a JSON-parse snippet of the model's own response. None
                # of that is provably structural, so only the exception TYPE is
                # safe to persist. This field is not a diagnostic sink:
                # version_ingest[].error is schema-sanctioned straight into the
                # PUBLISHED playbook.opf.json (_VERSION_INGEST_SCHEMA_KEYS,
                # playbook_assembler.py), and is echoed verbatim by
                # inspection_report.py's _version_ingest_review_flags — one
                # unsafe write here leaks through both persisted artifacts at
                # once.
                "error": type(exc).__name__,
                "extractor": extractor,
                "reason": (
                    extractor_label.reason
                    if extractor_label is not None
                    # issue #218: a closed enum ("timeout" | "no-text" |
                    # None), never text from the message — safe to persist
                    # for the same reason "error" above is only a type name.
                    else (exc.reason if isinstance(exc, ExtractionError) else None)
                ),
            }
            fingerprint_parts.append(
                {
                    "name": vf.name,
                    "sha256": state.sha256_by_vid[vid],
                    "failed": state.version_ingest[vid],
                }
            )
            # Issue #231: extraction recovered this version after a timeout,
            # then segmentation raised. label_out is empty in that case
            # (_llm_segment_file raised before returning the label), so read
            # the label segment_to_tree carried out on the exception — the
            # recovery is run-only and must keep the deal out of the stage
            # cache exactly like a plain timeout.
            recovered_label = (
                extractor_label if extractor_label is not None else extractor_label_of(exc)
            )
            if timed_out_versions is not None and (
                state.version_ingest[vid]["reason"] == FAILURE_TIMEOUT
                or (recovered_label is not None and recovered_label.timed_out)
            ):
                timed_out_versions.append(vid)
    state.l1_fingerprint = make_config_fingerprint(fingerprint_parts)
    return state


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
    refresh_extraction: bool = False,
    template_std_nodes_by_tid: dict[str, list[str]] | None = None,
    timed_out_versions: list[str] | None = None,
) -> dict[str, Any] | None:
    """Compute L1–L4 for a single document; return a cacheable result dict or None on skip.

    *template_std_nodes_by_tid* (issue #216) is every template node's text
    per taxonomy_id, in document order — the origin reference for a clause
    removed before signing. When ``None`` (legacy callers), the joined
    *template_std_by_tid* standard stands in.

    Deviations come from the deterministic standard check, never a judge.

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
      - ``version_trees``: ``{version_id: ClauseTree.to_dict()}`` for every
                           mined version (issue #139) — raw, pre-
                           pseudonymization, exactly like ``observations``
                           and ``trail`` above. ``mine_corpus`` collects this
                           across every document and materialises
                           ``normalized/`` from it AFTER the born-safe
                           pseudonymization pass (stale-cleared and written
                           under the aliased document_id, mirroring trail/'s
                           treatment) instead of this function writing raw
                           doc_id-named trees to disk directly — so a stage-
                           cache hit still yields a tree write on this run.
      - ``scope_decision``: scope decision fields (for replaying into ScopeLog).

    Returns ``None`` if the document has no processable versions.

    *timed_out_versions* (issue #218), when given, receives the version id of
    every version whose extraction failed with reason ``"timeout"`` — and
    (issue #231) of every version whose text a fallback recovered after a
    timeout (:attr:`~playbook_engine.extraction.ExtractorLabel.timed_out`).
    The caller uses it to keep this result — including the all-failed ``None`` —
    OUT of the L1-L4 stage cache: a timeout is a property of the run, not of
    the source bytes the cache key hashes, so caching it would stop every
    later run from retrying the version.

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

    ``refresh_extraction`` is forwarded to ``_llm_segment_file`` alongside
    ``extraction_cache`` — see ``mine_corpus``'s parameter of the same name
    (issue #78).

    A version whose ingest yields an EMPTY ``ClauseTree`` from a non-empty
    source file (e.g. a scanned/image PDF on the deterministic path, where no
    OCR is wired) is treated as an ingest failure — recorded via the same
    per-version warning as any other extraction exception, never added to
    ``version_trees``. This prevents an unreadable version from silently
    becoming ``first_tree`` and being misclassified by the scope gate as
    ``deterministic_empty`` (issue #82).
    """
    taxonomy_ids = [e.id for e in taxonomy.classifier_entries()]
    l1 = _collect_l1(
        doc_id,
        version_files,
        config=config,
        taxonomy_ids=taxonomy_ids,
        progress=progress,
        timed_out_versions=timed_out_versions,
        use_llm_segmentation=use_llm_segmentation,
        llm_segment_fn=llm_segment_fn,
        batch_seg_nodes=batch_seg_nodes,
        batch_extractions=batch_extractions,
        segmentation_cache=segmentation_cache,
        extraction_cache=extraction_cache,
        refresh_extraction=refresh_extraction,
    )
    return _compute_doc_from_l1(
        doc_id,
        doc_dir,
        version_files,
        l1,
        config,
        taxonomy,
        template_tree,
        template_std_by_tid,
        _scope_judge,
        _cls_judge,
        alignment_judge,
        trail_judge,
        progress,
        signed_judge=signed_judge,
        provenance_judge=provenance_judge,
        use_llm_segmentation=use_llm_segmentation,
        normalize_trail_across_versions=normalize_trail_across_versions,
        normalize_trail_fn=normalize_trail_fn,
        template_std_nodes_by_tid=template_std_nodes_by_tid,
    )


def _compute_doc_from_l1(
    doc_id: str,
    doc_dir: Path,
    version_files: list[Path],
    l1: _L1State,
    config: EngineConfig,
    taxonomy: Taxonomy,
    template_tree: ClauseTree | None,
    template_std_by_tid: dict[str, str],
    _scope_judge: ScopeJudge,
    _cls_judge: ClassificationJudge,
    alignment_judge: AlignmentJudge | None,
    trail_judge: TrailJudge | None,
    progress: Callable[[str], None],
    signed_judge: SignedJudge | None = None,
    provenance_judge: ProvenanceJudge | None = None,
    use_llm_segmentation: bool = False,
    normalize_trail_across_versions: bool = False,
    normalize_trail_fn: NormalizeTrailFn | None = None,
    template_std_nodes_by_tid: dict[str, list[str]] | None = None,
) -> dict[str, Any] | None:
    """L2-L4 for one document from its L1 state (issue #219).

    The second half of :func:`_compute_doc_result` (see there for the
    returned dict): cross-version normalization, scope gate, signed/version
    order/provenance (L2), classification (L3), diff/reversals/deviations
    (L4). ``mine_corpus`` caches its result per document under a key built
    from ``l1.l1_fingerprint``.
    """
    version_trees = l1.version_trees
    version_tree_dicts = l1.version_tree_dicts
    unstripped_trees = l1.unstripped_trees
    llm_taxonomy_by_path = l1.llm_taxonomy_by_path
    tracked_by_vid = l1.tracked_by_vid
    timestamp_by_vid = l1.timestamp_by_vid
    version_ingest = l1.version_ingest
    sha256_by_vid = l1.sha256_by_vid
    media_type_by_vid = l1.media_type_by_vid
    taxonomy_ids = [e.id for e in taxonomy.classifier_entries()]

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
        unstripped_trees.values(), config.provenance.our_party_aliases
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
    first_vid = next(iter(version_trees))
    first_tree = version_trees[first_vid]
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
    version_ingest_list = _build_version_ingest_list(version_files, version_ingest)

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
            # Out-of-scope documents still get their trees materialised
            # (issue #139) — the pre-fix code wrote them unconditionally,
            # before this scope check even ran.
            "version_trees": version_tree_dicts,
            "scope_decision": scope_decision_dict,
            "our_alias_matched": our_alias_matched,
        }

    # L2: Signed detection, version ordering, provenance
    signed_status_by_vid: dict[str, Any] = {}
    version_inputs = []
    for vid, tree in version_trees.items():
        # The unstripped tree (issue #217): the signature block this reads
        # was cut from version_trees' clause text.
        ss = detect_signed(unstripped_trees[vid], signed_judge=signed_judge)
        signed_status_by_vid[vid] = ss
        version_inputs.append(
            VersionInput(version_id=vid, tree=tree, signed=ss, timestamp=timestamp_by_vid.get(vid))
        )
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
    # Unstripped (issue #217), like the template_tree mine_corpus passes in:
    # the alias-presence signal reads party names wherever they appear,
    # signature blocks included, and the template-similarity signal must
    # compare like with like.
    prov_tree = unstripped_trees[earliest_vid] if earliest_vid else unstripped_trees[first_vid]
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

    # Issue #225: an ambiguous detection is "unknown" — never coerced to a
    # side. (The coercion this replaces flipped e.g. an alias_present
    # our_paper lean to counterparty_paper.) Paper side is deal metadata
    # only; it gates nothing in OPF 0.5.
    provenance = PROVENANCE_UNKNOWN if prov_result.is_ambiguous else prov_result.provenance

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
        # never written at all, so any reversal count read from the trail was
        # permanently 0 regardless of what detect_reversals actually found).
        "reversals": [],
        **version_order.to_dict(),
        # Issue #221: each mined version's own authorship timestamp (null when
        # it carries none) — the evidence order_versions tie-broke with,
        # alongside any hints.yaml timestamps (which override per version).
        "version_timestamps": {vid: timestamp_by_vid.get(vid) for vid in version_trees},
    }

    # L3: Classify each version. LLM-segmented versions already carry their
    # taxonomy_id from the L1 LLM pass — bypass classify_tree entirely for
    # those (no separate classify judge for LLM-segmented docs).
    #
    # Issue #235: a node still unclassified after that (heading paths, judge,
    # parent inheritance, or the LLM's explicit null) is compared with our
    # standard's own clause text and assigned only on a conservative
    # threshold + margin (basis "content_similarity"). Emergent mode has no
    # template, so no exemplars: a no-op there. The template itself never
    # takes this path (see _build_template_observations).
    content_exemplars = _content_exemplars(template_std_by_tid)
    ordered_ids = list(version_order.ordered_ids) or list(version_trees.keys())
    classified_by_version: dict[str, list[ClassifiedClause]] = {}
    for vid in ordered_ids:
        if vid in llm_taxonomy_by_path:
            classified_by_version[vid] = assign_by_content(
                _classified_from_taxonomy_by_path(version_trees[vid], llm_taxonomy_by_path[vid]),
                content_exemplars,
                eligible_ids=set(taxonomy_ids),
            )
        else:
            classified_by_version[vid] = classify_tree(
                version_trees[vid],
                taxonomy,
                _cls_judge,
                ambiguity_threshold=config.classification.ambiguity_threshold,
                auto_classify_threshold=config.classification.auto_classify_threshold,
                content_exemplars=content_exemplars,
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
    # Issue #216: per-reason count of net-diff rows that produced no
    # observation (no signed slot, and either its text survives in the signed
    # version or, removed before signing, its origin cannot be determined) —
    # recorded on corpus_doc below and summed into corpus.stats at L5.
    dropped_observations: dict[str, int] = {}
    if len(ordered_ids) < 2:
        doc_obs = _observations_from_single_version(
            doc_id,
            signed_ordinal,
            signed_vid,
            provenance,
            classified_by_version[signed_vid],
            has_signed_copy=has_signed_copy,
            template_std_by_tid=template_std_by_tid,
            our_party_aliases=config.provenance.our_party_aliases,
            our_authors=config.provenance.our_authors,
            template_std_nodes_by_tid=template_std_nodes_by_tid,
            party_names=_standard_party_names(config),
        )
    else:
        classified_versions = [(vid, classified_by_version[vid]) for vid in ordered_ids]
        alignments = align_versions(classified_versions, alignment_judge=alignment_judge)
        doc_diff = diff_aligned(alignments, ordered_ids)
        # Issue #221: reversals need a signed terminal — with no detected
        # signed copy, ordered_ids[-1] is only the last draft (for an
        # unsigned deal often a tie-broken chain direction), so text absent
        # from it was never "refused before signing". The same gate
        # build_observations receives below.
        reversals = detect_reversals(
            doc_diff, has_signed_copy=has_signed_copy and ordered_ids[-1] == signed_vid
        )
        # Negotiation dynamics (issue #177): surface the per-round diffs as
        # RoundMove records instead of discarding them — L5 derives each
        # precedent's rounds/moved from them.
        round_moves = build_round_moves(
            doc_id,
            doc_diff,
            tracked_by_vid=tracked_by_vid,
            our_party_aliases=config.provenance.our_party_aliases,
            our_authors=config.provenance.our_authors,
        )
        # Issue #106: record detected reversals on the trail itself, so a
        # reader of the trail sees them (previously a default-empty .get()
        # on a key that was never populated reported zero reversals even
        # when detect_reversals found some).
        trail["reversals"] = [r.to_dict() for r in reversals]

        net_diffs = list(doc_diff.net.diffs)
        deviation_results = _assess_deviations_with_standards(
            net_diffs,
            template_std_by_tid,
            template_std_nodes_by_tid=template_std_nodes_by_tid,
            party_names=_standard_party_names(config),
        )

        # Per-version confidence map (issue #65) — see
        # _classification_confidence_for_diff for why a removed clause's
        # confidence must come from ITS version, not the signed version's.
        cls_conf_by_version_path: dict[str, dict[str, float]] = {
            vid: {(cc.node.clause_path or "?"): cc.classification.confidence for cc in classified}
            for vid, classified in classified_by_version.items()
        }
        classification_confidences = [
            _classification_confidence_for_diff(cd, cls_conf_by_version_path)
            for cd, _ in deviation_results
        ]

        # Tracked-changes attribution (issue #88): best-effort, from the
        # signed/last version's own DOCX side-channel — see
        # _attribution_for_diff for why that version is the right source
        # even though net_diffs can span more than one negotiation round,
        # and why single_round (True only for a 2-version document, where
        # the net diff IS the one round) gates its round-level fallback
        # tier (issue #118 fix round 2, finding 1).
        signed_tracked = tracked_by_vid.get(signed_vid)
        single_round = len(doc_diff.consecutive) == 1
        attributions = [
            _attribution_for_diff(cd, signed_tracked, single_round=single_round)
            for cd, _ in deviation_results
        ]

        # Issue #216: the net diff's after side is ordered_ids[-1]; its
        # observations are only "signed" when that terminal IS the detected
        # signed copy (order_versions anchors the chain there). Draft text
        # must never be reported as signed.
        terminal_vid = ordered_ids[-1]
        doc_obs = build_observations(
            doc_id,
            signed_ordinal,
            provenance,
            deviation_results,
            reversals,
            classification_confidences,
            has_signed_copy=has_signed_copy and terminal_vid == signed_vid,
            attributions=attributions,
            our_party_aliases=config.provenance.our_party_aliases,
            our_authors=config.provenance.our_authors,
            # Same vid → negotiation-ordinal map build_round_moves derives
            # from version_order: a reversal/removed-clause citation must
            # carry ITS draft's ordinal, not signed_ordinal, or it resolves
            # (via version_files) to a file the cited clause is not in.
            ordinal_by_vid={vid: i + 1 for i, vid in enumerate(ordered_ids)},
            # Issue #216: exactly one terminal observation per taxonomy_id,
            # built from the terminal version's own classified tree (its
            # text, document order, first-node citation).
            terminal_clauses=classified_by_version[terminal_vid],
            terminal_version_id=terminal_vid,
            dropped=dropped_observations,
            # Issue #216: the origin reference for a clause removed before
            # signing — our standard language struck is our concession (only
            # when the deal has a detected executed copy), non-standard
            # language struck is their refused ask.
            standard_text_by_tid=(
                template_std_nodes_by_tid
                if template_std_nodes_by_tid is not None
                else template_std_by_tid
            ),
            party_names=_standard_party_names(config),
        )

    # Issue #225: paper_basis / paper_confidence travel on every observation
    # of the deal (template observations carry none), so the L5 precedent
    # record states which detection signal its paper side rests on.
    doc_obs = [
        dataclasses.replace(
            obs, paper_basis=prov_result.basis, paper_confidence=prov_result.confidence
        )
        for obs in doc_obs
    ]

    # Issue #237: record how each observation's cited node was classified,
    # for `playbook scorecard` (classification by basis). Looked up by the
    # citation's (version_id, clause_path) in the same classified trees the
    # observations were built from; an observation whose taxonomy_id differs
    # from its node's own classification took it from its aligned row
    # ("aligned").
    classification_by_node: dict[tuple[str, str], ClauseClassification] = {
        (vid, cc.node.clause_path or "?"): cc.classification
        for vid, classified in classified_by_version.items()
        for cc in classified
    }
    doc_obs = [
        dataclasses.replace(
            obs,
            classification_basis=_observation_classification_basis(obs, classification_by_node),
        )
        for obs in doc_obs
    ]

    # corpus.documents[].provenance is a frozen two-valued field and keeps
    # today's value (issue #225): an undetermined ("unknown") side is written
    # the §2.3 way via two_valued_side, and provenance_is_ambiguous: true
    # records that the side was not determined.
    corpus_doc["provenance"] = two_valued_side(provenance)
    # Issue #216: summed into corpus.stats.dropped_observations by
    # assemble_playbook (and stripped from the embedded corpus document).
    corpus_doc["dropped_observations"] = dict(sorted(dropped_observations.items()))
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
        # Serialised trees (issue #139); .get()-read by mine_corpus so
        # cached results from before this feature simply contribute none
        # (see _NORMALIZED_TREES_CACHE_VERSION, which busts those entries).
        "version_trees": version_tree_dicts,
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
    refresh_extraction: bool = False,
    entity_registry_path: Path | None = None,
    progress: Callable[[str], None] = lambda _: None,
    cache_dir: Path | None = None,
    force_rewrite: bool = False,
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
        alignment_judge:      L3 alignment judge; defaults to None (deterministic only).
        trail_judge:          Version-ordering judge; defaults to None (deterministic only).
        signed_judge:         L2 signed-copy judge; defaults to None (deterministic only).
        provenance_judge:     L2 provenance judge; defaults to None (deterministic only).
        no_cache:             If True, skip the cache and force a full recompute.
                              Store-backed judges (``agent_judge.StoreBacked*``)
                              no longer need it (issue #219): each cached
                              L2-L4 result records the verdict keys it replayed
                              and is recomputed when any of them changed, and a
                              result that queued anything is never cached.
        cache_dir:            Where the stage cache lives; defaults to
                              ``out_dir/.cache``. ``judge --plan-only`` mines
                              into a temp out-dir but reads the real out-dir's
                              cache through this (issue #219).
        force_rewrite:        Rewrite every intermediate even when its content
                              is unchanged. By default (issue #219) a file whose
                              bytes would not change is left untouched.
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
                              cache) — deliberately so: an operator
                              ``--no-cache`` (and, before issue #219, the
                              ``no_cache=True`` that ``playbook judge``'s
                              store-backed judges used to force) must not
                              also force every round to re-extract/re-OCR
                              every version of every agreement from scratch
                              (issue #132). Defaults to None (no caching —
                              every run re-extracts). See
                              ``refresh_extraction`` below for the
                              operator-facing lever that DOES force
                              re-extraction on demand.
        refresh_extraction:   If True, bypass ``extraction_cache``'s reads for
                              this run — every version is re-extracted from
                              source (docling/pdfplumber/python-docx/pandoc)
                              — while writes still happen, leaving a fresh,
                              correct cache behind for subsequent runs
                              (issue #78). Deliberately a SEPARATE signal
                              from ``no_cache`` above: the judge path used
                              to force ``no_cache=True`` (until issue #219
                              left the stage cache on under store-backed
                              judges) without wanting to force re-extraction
                              (that would re-burn docling OCR timeouts every
                              round — issue #132), so this is threaded
                              independently. ``cli.py``'s ``mine`` command
                              sources it from the operator's own
                              ``--no-cache`` flag; ``playbook judge`` never
                              sets it. Ignored when ``extraction_cache`` is
                              None. Defaults to False.
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
    # There is no deviation judge: every deviation is the deterministic
    # standard check (see _assess_deviations_with_standards).

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
        },
        sort_keys=True,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    obs_path = out_dir / "observations.jsonl"
    manifest_path = out_dir / "corpus_manifest.json"

    # Content-addressed stage cache — disabled when no_cache=True.
    stage_cache_dir = cache_dir if cache_dir is not None else out_dir / ".cache"
    store: ArtifactStore | None = None if no_cache else ArtifactStore(stage_cache_dir)

    # Store-backed judges (issue #219). Their VerdictStore is the verdict
    # cache: it is the authoritative record of every verdict, rubric stamp and
    # all. The stage cache stays ON under them — each cached L2-L4 result
    # records the verdict keys it replayed (VerdictStore.capture_lookups) and
    # is refused on replay when any of those verdicts changed, and a result
    # whose compute queued anything (PendingQueue.capture_adds — a miss, a
    # stale rubric, a malformed stored verdict) is never cached at all, so a
    # replayed document never hides a pending item from the round's queue.
    raw_judges: list[Any] = [_scope_judge, _cls_judge, provenance_judge]
    verdict_stores: list[VerdictStore] = _distinct(
        getattr(j, "store", None)
        for j in raw_judges
        if isinstance(getattr(j, "store", None), VerdictStore)
    )
    pending_queues: list[PendingQueue] = _distinct(
        getattr(j, "pending", None)
        for j in raw_judges
        if isinstance(getattr(j, "pending", None), PendingQueue)
    )
    rubric_policies: list[RubricPolicy] = _distinct(
        getattr(j, "rubric", None)
        for j in raw_judges
        if isinstance(getattr(j, "rubric", None), RubricPolicy)
    )
    segmentation_store = (
        segmentation_cache.verdict_store if segmentation_cache is not None else None
    )

    def _store_backed(judge: Any) -> bool:
        return isinstance(getattr(judge, "store", None), VerdictStore)

    # Judgment verdict cache — wraps the inline judges with batching + content-addressed
    # caching so identical clause payloads are judged once per corpus (issue #62).
    # The cache persists across runs in out/.cache/verdicts.jsonl.
    #
    # When no_cache=True the verdict cache is skipped entirely (same flag that
    # disables the #61 stage cache).  This guarantees that "force a full recompute"
    # means a full recompute — no stale verdict hits from a previous run.
    #
    # Never around a store-backed judge (issue #219): this cache would answer
    # in its place, so a verdict later overwritten (judge-apply) in the
    # VerdictStore would keep replaying the old answer,
    # a stale-rubric verdict would never be re-queued, and the lookup the
    # stage cache's dependency check rests on would never happen.
    if not no_cache:
        verdict_cache = JudgmentCache(
            stage_cache_dir / "verdicts.jsonl",
            model_id=judge_identity,
        )
        if not _store_backed(_scope_judge):
            _scope_judge = BatchedScopeJudge(delegate=_scope_judge, cache=verdict_cache)
        if not _store_backed(_cls_judge):
            _cls_judge = BatchedClassificationJudge(delegate=_cls_judge, cache=verdict_cache)

    # Config fingerprint prep: hash the template file's *content* (not its
    # path) so that changing the file's text under the same path correctly
    # busts per-doc cache entries. The fingerprint itself is assembled AFTER
    # template ingestion below (see the comment there) so it also captures
    # the template's classification OUTCOME, not just the file on disk.
    template_content_hash: str | None = None
    if config.baseline.template_path and config.baseline.template_path.exists():
        template_content_hash = _sha256_file(config.baseline.template_path)

    # -----------------------------------------------------------------------
    # Ingest template
    # -----------------------------------------------------------------------
    # On the agent/LLM segmentation path the template MUST be segmented and
    # classified the same way the corpus documents are (store-backed, one
    # taxonomy_id per clause in the same pass). The deterministic
    # segment+classify_tree fallback relies on heading similarity, which on a
    # real template routinely classifies nothing — leaving template_std_by_tid
    # empty and silently degrading a template-mode run to emergent mode (no
    # per-clause our_standard, no claimable stances).
    template_tree: ClauseTree | None = None
    template_classified: list[ClassifiedClause] | None = None
    if config.baseline.template_path:
        try:
            if use_llm_segmentation:
                # The template is not a corpus document — it never enters
                # corpus_documents/version_ingest and is not counted against
                # config.extraction.max_fallback (issue #81's (document_id,
                # version, reason) tuple spec is per corpus document), so its
                # own extractor label is discarded here.
                template_tree, t_tax_by_path, _template_extractor_label = _llm_segment_file(
                    config.baseline.template_path,
                    "template",
                    "template",
                    [e.id for e in taxonomy.classifier_entries()],
                    llm_segment_fn,
                    segmentation_cache,
                    model=config.segmentation.model,
                    extraction_cache=extraction_cache,
                    refresh_extraction=refresh_extraction,
                    extractor=config.extraction.extractor,
                )
                template_classified = _classified_from_taxonomy_by_path(
                    strip_signature_block(template_tree)[0], t_tax_by_path
                )
            else:
                raw_tree = _ingest_file(config.baseline.template_path, "template", "template")
                template_tree = segment(raw_tree)
            progress(f"  template: {config.baseline.template_path.name}")
        except SegmentationQAError as exc:
            # Agent path, template content not yet in the segment store: the
            # segmentation queue now carries it. Warn LOUDLY — continuing
            # without a template silently changes the run's mode.
            template_tree = None
            template_classified = None
            progress(
                f"  WARNING: template segmentation pending ({exc}) — run "
                "'playbook segment'/'segment-apply' and re-mine, or this run "
                "proceeds WITHOUT a template (emergent mode)"
            )
        except Exception as exc:  # noqa: BLE001
            progress(f"  WARNING: could not ingest template: {exc}")

    # Our standards come from the template's clause language only — its
    # signature block is cut out exactly as for corpus versions (issue #217).
    # template_tree itself stays unstripped: _compute_doc_result hands it to
    # detect_provenance, whose template-similarity signal compares it with
    # each document's unstripped tree.
    if template_classified is not None:
        t_observations = _template_observations_from_classified(template_classified)
    elif template_tree:
        t_observations = _build_template_observations(
            strip_signature_block(template_tree)[0],
            taxonomy,
            _cls_judge,
            ambiguity_threshold=config.classification.ambiguity_threshold,
            auto_classify_threshold=config.classification.auto_classify_threshold,
        )
    else:
        t_observations = []
    if config.baseline.template_path and template_tree is not None:
        n_std = sum(1 for o in t_observations if o.taxonomy_id)
        # Issue #242: front matter is classified but is no standard — say how
        # many nodes left our_standard, so the exclusion is never silent.
        n_front = len(form_front_matter(t_observations))
        progress(
            f"  template standards: {n_std} clause(s) classified; "
            f"{n_front} form front-matter node(s) excluded from our_standard"
        )

    # -----------------------------------------------------------------------
    # L1 → L4: per-document, optionally cached
    # -----------------------------------------------------------------------
    # our_standard fed to the standard check must be the full clause text,
    # not the ≤ 300-char text_summary — a ≤ 300-char fragment of a real
    # indemnification/insurance clause is not a usable standard to compare
    # against (issue #105).
    # Issue #242: a clause type's standard is EVERY template node carrying its
    # taxonomy_id, in document order, joined (a lead-in node plus its
    # operative limbs) — the same text ``our_standard`` carries — and a form's
    # fill-in cover table is no standard (playbook_engine.template_standards).
    t_standards = template_standards(t_observations)
    template_std_by_tid: dict[str, str] = {tid: std.text for tid, std in t_standards.items()}
    # Issue #216: the ORIGIN reference for a clause removed before signing,
    # the deterministic standard check and every observation's `standard` fact
    # is EVERY non-empty template node carrying its taxonomy_id, in document
    # order, node by node — form front matter INCLUDED (issue #242: it is
    # left out of our_standard only). Otherwise our own text from a later
    # node of a multi-node standard clause, or from our cover table, struck
    # before signing, would be misread as a counterparty ask we refused.
    template_std_nodes_by_tid: dict[str, list[str]] = {}
    for t_obs in t_observations:
        if t_obs.taxonomy_id is not None:
            template_std_nodes_by_tid.setdefault(t_obs.taxonomy_id, []).append(t_obs.full_text)

    # Config fingerprint: encodes the fields that affect L1-L4 outputs.
    # Assembled HERE — after template ingestion above, not from the template
    # file's content hash alone — so that the template's classification
    # OUTCOME (not just the bytes on disk) busts the per-doc stage cache
    # (issue #243). On the agent/LLM segmentation path, when the template's
    # segmentation verdict isn't yet in the store, the except block above
    # catches SegmentationQAError, leaves template_tree=None and
    # template_std_by_tid={}, and the run proceeds in emergent mode; any doc
    # whose own segmentation verdict is already store-resident then computes
    # against EMPTY standards. Folding template_tree_present and a fingerprint
    # of the actual template_std_by_tid contents in here means that once the
    # advertised remediation ("run 'playbook segment'/'segment-apply' and
    # re-mine") makes the template classify, config_fp changes even though
    # template_content_hash is identical — so store.get_or_compute recomputes
    # every per-doc result against the real standards instead of replaying
    # the template-less cached ones verbatim, forever.
    # L1 fingerprint (issue #219): ONLY what changes a version's L1 record
    # (_l1_version_record) for identical source bytes — the extraction
    # environment and the segmentation path/model/prompt. The template, the
    # taxonomy wording, the judges and the thresholds are deliberately absent:
    # they act at L2-L4, so changing one replays every L1 tree.
    l1_config_fp = make_config_fingerprint(
        {
            "l1_record_version": _L1_RECORD_VERSION,
            # Which extractor produced the source text changes L1 ingest
            # output for byte-identical source files — legacy
            # (pdfplumber/python-docx/pandoc) has no OCR and can garble
            # columns/scanned text, so docling and legacy can disagree on the
            # SAME bytes (see extraction.detect_extractor). Installing or
            # removing docling between runs must bust every per-doc
            # stage-cache entry rather than replay an L1-L4 result derived
            # from the OTHER environment's extraction as if it were current
            # (issue #79). "extractor_env" itself is still exactly that
            # original PATH-only check (constant across every file in the
            # run, hence corpus_dir rather than a per-file path — passed for
            # clarity, not because its content matters): it is blind to a
            # config-DECLARED extractor (config.extraction.extractor, issue
            # #80), since detect_extractor never looks at config. Without
            # more, flipping the config between "auto"/"docling"/"legacy" on
            # the SAME host leaves "extractor_env" unchanged whenever the
            # PATH check's own answer doesn't move, so a declared "legacy"
            # run would silently replay docling-derived L1-L4 results (or
            # vice versa) instead of busting the cache. "declared_extractor"
            # below closes that gap by folding the raw declared value itself
            # into the fingerprint, independently of what the PATH check
            # reports. Adding either field intentionally invalidates every
            # existing L1-L4 stage-cache entry once, the same one-time cost
            # as any other fingerprint-field addition.
            "extractor_env": detect_extractor(corpus_dir),
            "declared_extractor": config.extraction.extractor,
            # Issue #218: whether the ocrmypdf second OCR path is on PATH
            # changes L1 output for a byte-identical scanned PDF under a
            # docling environment (recovered text vs a failed version), so
            # installing or removing it must bust every per-doc entry the
            # same way "extractor_env" does for docling itself.
            "ocrmypdf_available": ocrmypdf_available(),
            "use_llm_segmentation": use_llm_segmentation,
            # The batch path has no repair loop (see _ground_batch_result), so
            # the same source content can in principle segment differently
            # under batch vs. synchronous LLM calls — never replay one path's
            # stage-cached tree as if it were the other's.
            "use_batch_segmentation": use_batch_segmentation,
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
            # The LLM path classifies in the same pass as it segments, against
            # the classifier-eligible taxonomy ids — so on that path (only)
            # the id set is an L1 input too.
            "llm_taxonomy_ids": (
                sorted(e.id for e in taxonomy.classifier_entries())
                if use_llm_segmentation
                else None
            ),
            # version_ingest's "reason" travels in the L1 record.
            "version_ingest_reason_version": _VERSION_INGEST_REASON_VERSION,
        }
    )

    l2_config_fp = make_config_fingerprint(
        {
            "agreement_type_id": config.agreement_type.id,
            # The scope gate judges against the whole agreement-type
            # definition, not only its id (issue #219).
            "agreement_type": {
                "name": config.agreement_type.name,
                "description": config.agreement_type.description,
                "aliases": sorted(config.agreement_type.aliases),
            },
            "provenance_aliases": sorted(config.provenance.our_party_aliases),
            # Issue #220: known_entities are neutralized by the deterministic
            # standard check (_standard_party_names), so they change L4's
            # standard/deviation output for identical source content.
            "standard_party_names": sorted(config.provenance.known_entities),
            # Issue #119: our_authors feeds party_side_for_author exactly like
            # our_party_aliases does (proposed_by/moved_by) — a config change
            # here must bust the per-doc cache the same way an alias change
            # does, or a stale "unknown" persists after the corpus's true
            # author list is filled in.
            "provenance_authors": sorted(config.provenance.our_authors),
            "template_content_hash": template_content_hash,
            "template_tree_present": template_tree is not None,
            "template_standards": make_config_fingerprint(sorted(template_std_by_tid.items())),
            # Issue #216: the origin reference build_observations classifies
            # removed-before-signing text against — every template node per
            # taxonomy_id, so a change to any later node busts the cache too.
            "template_origin_standards": make_config_fingerprint(
                sorted(template_std_nodes_by_tid.items())
            ),
            # Switching segmentation paths also switches L3 (the LLM path's
            # own per-clause taxonomy vs classify_tree) — an L2-L4 input too,
            # not only an L1 one.
            "use_llm_segmentation": use_llm_segmentation,
            # Toggling cross-version taxonomy normalization changes the L1
            # output for every version of every multi-version agreement — a
            # prior run's un-normalized cached trees must not be replayed
            # silently once this is switched on (issue #90).
            "normalize_trail_across_versions": normalize_trail_across_versions,
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
            # Taxonomy content feeds classification; editing the taxonomy file
            # must bust the L2-L4 cache rather than replay results classified
            # against the old entries (the judges' own verdict keys already
            # include taxonomy_ids — this closes the same hole for the stage
            # cache). This entry list alone missed ``status`` (issue #219):
            # retiring an entry (status: inactive) removes it from what a
            # clause may be classified into while its id/label/description
            # stay put. "rubric_versions" below closes that: its classify
            # version is built from rubric.taxonomy_digest — the digest of
            # exactly the classifier-eligible (active/custom) surface — so a
            # status flip moves it.
            "taxonomy_entries": sorted((e.id, e.label, e.description) for e in taxonomy.entries),
            # The rubric in force for every judge kind: the taxonomy digest
            # above (classify), the agreement-type definition (scope) and the
            # answer enums and prompt versions (all kinds). Under store-backed
            # judges a moved rubric re-queues a stored verdict, so a result
            # replayed across a rubric change would be wrong; the staleness
            # policy (--accept-stale / --strict-rubric) decides the same thing
            # and is not in the verdict store either.
            "rubric_versions": current_versions(
                taxonomy=taxonomy, agreement_type=config.agreement_type
            ),
            "rubric_policies": [
                [policy.strict_legacy, policy.accept_stale] for policy in rubric_policies
            ],
            # Deviation assessment now diffs "unchanged" clauses (including
            # every clause of a single-version document) against the
            # canonical template rather than hardcoding deviation="none"
            # (issue #103) — a code change, not a config change, but one that
            # changes L1-L4 output for identical source content + judges. Bump
            # this constant on any future change to that comparison logic so
            # a warm cache from before the fix is never replayed verbatim.
            "deviation_vs_template_version": _DEVIATION_VS_TEMPLATE_VERSION,
            # version_ingest's "reason" field and the corrected live-fallback
            # label (issue #81) — see _VERSION_INGEST_REASON_VERSION above.
            # Without this, a warm per-doc stage-cache entry from before the
            # fix would keep replaying corpus_doc dicts whose version_ingest
            # entries carry no "reason" key (and, for a live fallback, the
            # WRONG "docling" label), making max_fallback/the review flags/
            # the CLI reason breakdown all silently blind.
            "version_ingest_reason_version": _VERSION_INGEST_REASON_VERSION,
            # The per-doc result gained "version_trees" (issue #139) — see
            # _NORMALIZED_TREES_CACHE_VERSION above. Without this, a warm
            # per-doc stage-cache entry from before the fix has no
            # "version_trees" key, so mine_corpus's normalized/ rewrite pass
            # would silently skip that document's trees on this run.
            "normalized_trees_cache_version": _NORMALIZED_TREES_CACHE_VERSION,
        }
    )

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
    # Serialised clause trees, same deferred-materialisation pattern as
    # all_trails above (issue #139) — see "Materialise normalized/ clause
    # trees now" below. Unlike all_trails, every document contributes here
    # regardless of scope (normalized/ has always covered out-of-scope docs
    # too — see _compute_doc_result's out-of-scope return).
    all_version_trees: list[tuple[str, dict[str, dict[str, Any]]]] = []

    scope_log = ScopeLog(agreement_type_id=config.agreement_type.id)
    # Match corpus_linter.lint_corpus and cli.segment_cmd: dot-directories
    # (.cache, .git, .DS_Store, an `out` dir accidentally nested inside the
    # corpus) are never agreement folders. Without this filter mine_corpus
    # disagreed with the linter's document count and emitted a spurious
    # "no supported files — skipping" line for every dot-directory (issue #54,
    # merged finding).
    doc_dirs = sorted(d for d in corpus_dir.iterdir() if d.is_dir() and not d.name.startswith("."))

    # Fail loud on a corpus with zero minable documents (issue #54): without
    # this guard mine_corpus wrote an empty observation store and exited 0
    # with a green "OK", and the failure only surfaced later at project/
    # compile time with a message that misleadingly says to (re-)run mine
    # even when compile just did. This must trigger only when NO doc_dir
    # yields any supported version file — a doc_dir that yields versions but
    # is later quarantined (failed extraction, QA gate) is a different,
    # already-fail-loud failure mode and must not be conflated with this one.
    if not any(_discover_versions(d) for d in doc_dirs):
        loose_files = sorted(
            p
            for p in corpus_dir.iterdir()
            if p.is_file() and p.suffix.lower() in _SUPPORTED_EXTENSIONS
        )
        loose_hint = (
            f" — {len(loose_files)} supported file(s) sit directly in the corpus root; "
            "run 'playbook stage' to lay them out, or move each agreement into its own folder"
            if loose_files
            else ""
        )
        raise PipelineError(
            f"no agreement documents found under {corpus_dir}: the engine expects one "
            f"subfolder per agreement containing .docx/.pdf/.rtf files{loose_hint}; "
            "run 'playbook lint-corpus' for details"
        )

    # -------------------------------------------------------------------
    # Layered stage-cache helpers (issue #219)
    # -------------------------------------------------------------------
    def _l1_key(doc_id: str, vf: Path) -> str:
        return make_doc_key(doc_id, [vf], l1_config_fp, _L1_STAGE)

    def _l1_record_current(record: Any) -> bool:
        """A cached L1 record replays only while every segmentation-store
        entry it was grounded from still holds what it held then."""
        deps = record.get(_L1_DEPS_KEY) if isinstance(record, dict) else None
        if deps is None:
            return False
        if not deps:
            return True
        if segmentation_store is None:
            return False
        return all(segmentation_store.fingerprint(k) == fp for k, fp in deps.items())

    def _verdicts_current(result: Any) -> bool:
        """A cached L2-L4 result replays only while every stored verdict it was
        built from is unchanged — overwritten (judge-apply) or removed forces a
        recompute."""
        if result is None:
            return True  # no version ingested: no judge was ever asked
        deps = result.get(_VERDICT_DEPS_KEY) if isinstance(result, dict) else None
        if deps is None or len(deps) != len(verdict_stores):
            return False
        return all(
            st.fingerprint(k) == fp
            for st, st_deps in zip(verdict_stores, deps, strict=True)
            for k, fp in st_deps.items()
        )

    def _make_l1_fetch(doc_id: str) -> L1Fetch:
        assert store is not None
        stage_store = store

        def _fetch(vf: Path, compute: Callable[[], dict[str, Any]]) -> dict[str, Any]:
            unresolved = [False]
            timed_out = [False]

            def _compute() -> dict[str, Any]:
                if segmentation_store is None:
                    record = compute()
                    timed_out[0] = bool(record.get(_L1_TIMED_OUT_KEY))
                    return {**record, _L1_DEPS_KEY: {}}
                with segmentation_store.capture_lookups() as seen:
                    record = compute()
                timed_out[0] = bool(record.get(_L1_TIMED_OUT_KEY))
                deps = {k: segmentation_store.fingerprint(k) for k in sorted(seen)}
                unresolved[0] = any(fp is None for fp in deps.values())
                return {**record, _L1_DEPS_KEY: deps}

            def _cacheable() -> bool:
                # Never cache an unresolved segmentation lookup, nor (issue
                # #231) text a fallback recovered after a docling timeout.
                return not unresolved[0] and not timed_out[0]

            record: dict[str, Any] = stage_store.get_or_compute(
                _l1_key(doc_id, vf),
                _compute,
                cacheable=_cacheable,
                is_valid=_l1_record_current,
                stage=_L1_STAGE,
            )
            return record

        return _fetch

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
            # Issue #92: a version whose L1 stage cache is already warm will
            # be replayed by store.get_or_compute in the per-document loop
            # below — it never looks at batch_seg_nodes_by_doc/
            # batch_extractions_by_doc for a cache hit. Extracting and
            # submitting such a version to the (paid) batch here is pure
            # waste: at 40x3 scale, a re-run where only one document changed
            # would otherwise re-extract and re-segment all 120 versions. The
            # L1 key inputs (file name + content, L1 fingerprint) are all
            # available before extraction, so check each version and leave
            # the cache hits out (issue #219: per version, not per document —
            # L1 is cached per version now).
            uncached = {
                vf.stem: vf
                for vf in version_files
                if store is None
                or not store.contains(_l1_key(doc_dir.name, vf), is_valid=_l1_record_current)
            }
            if not uncached:
                skipped_cache_hit_docs += 1
                continue
            doc_versions[doc_dir.name] = uncached

        if skipped_cache_hit_docs:
            progress(
                f"  batch pre-pass: skipping {skipped_cache_hit_docs} document(s) "
                "already satisfied by the L1 stage cache"
            )

        taxonomy_ids = [e.id for e in taxonomy.classifier_entries()]
        items, batch_extractions_by_doc = _collect_batch_items(
            doc_versions,
            progress,
            extraction_cache=extraction_cache,
            refresh_extraction=refresh_extraction,
            extractor=config.extraction.extractor,
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
        timed_out_versions: list[str],
    ) -> Any:
        """Compute one document's L1–L4 result, via the layered stage cache when present.

        L1 runs (or replays) per version through the "l1" layer, then L2-L4
        runs (or replays) per document through the "l2-l4" layer, keyed by
        the L1 records it consumes (issue #219). *timed_out_versions*
        collects every version whose extraction timed out (issue #218); a
        result with any is returned but never stored.
        """
        l1 = _collect_l1(
            doc_id,
            version_files,
            config=config,
            taxonomy_ids=[e.id for e in taxonomy.classifier_entries()],
            progress=progress,
            timed_out_versions=timed_out_versions,
            fetch_l1=_make_l1_fetch(doc_id) if store is not None else None,
            use_llm_segmentation=use_llm_segmentation,
            llm_segment_fn=llm_segment_fn,
            batch_seg_nodes=doc_batch_seg_nodes,
            batch_extractions=doc_batch_extractions,
            segmentation_cache=segmentation_cache,
            extraction_cache=extraction_cache,
            refresh_extraction=refresh_extraction,
        )

        def _compute_l2_l4(_l1: _L1State = l1) -> Any:
            return _compute_doc_from_l1(
                doc_id,
                doc_dir,
                version_files,
                _l1,
                config,
                taxonomy,
                template_tree,
                template_std_by_tid,
                _scope_judge,
                _cls_judge,
                alignment_judge,
                trail_judge,
                progress,
                signed_judge=signed_judge,
                provenance_judge=provenance_judge,
                use_llm_segmentation=use_llm_segmentation,
                normalize_trail_across_versions=normalize_trail_across_versions,
                normalize_trail_fn=normalize_trail_fn,
                template_std_nodes_by_tid=template_std_nodes_by_tid,
            )

        if store is None:
            return _compute_l2_l4()

        computed = [False]
        unresolved = [False]

        def _compute_recorded() -> Any:
            """L2-L4, recording the stored verdicts it read (issue #219)."""
            computed[0] = True
            before = [dict(policy.counts) for policy in rubric_policies]
            with contextlib.ExitStack() as stack:
                seen = [stack.enter_context(st.capture_lookups()) for st in verdict_stores]
                queued = [stack.enter_context(q.capture_adds()) for q in pending_queues]
                result = _compute_l2_l4()
            if result is None:
                return None
            deps = [
                {k: st.fingerprint(k) for k in sorted(keys)}
                for st, keys in zip(verdict_stores, seen, strict=True)
            ]
            # Anything queued — a missing verdict, a stale rubric, a malformed
            # stored row — means this result carries needs_review sentinels:
            # never cache it, so the next run recomputes it and re-queues the
            # item into that round's (freshly reset) pending queue.
            unresolved[0] = any(queued) or any(
                fp is None for st_deps in deps for fp in st_deps.values()
            )
            result[_VERDICT_DEPS_KEY] = deps
            result[_RUBRIC_COUNTS_KEY] = [
                sorted(
                    [kind, state, n - b.get((kind, state), 0)]
                    for (kind, state), n in policy.counts.items()
                    if n - b.get((kind, state), 0)
                )
                for policy, b in zip(rubric_policies, before, strict=True)
            ]
            return result

        # Never cache a deal with a timed-out version (issue #218) —
        # neither a partial result nor the all-failed None, nor (issue #231)
        # a result holding a version a fallback recovered after a timeout. The
        # ExtractionCache already refuses to negative-cache the timeout;
        # storing the per-deal result here would replay it one layer up
        # and the next run would never retry the version.
        def _cacheable(_timed_out: list[str] = timed_out_versions) -> bool:
            return not _timed_out and not unresolved[0]

        cache_key = make_stage_key(
            doc_id, _L2_L4_STAGE, l1.l1_fingerprint, l2_config_fp, doc_dir / "hints.yaml"
        )
        result = store.get_or_compute(
            cache_key,
            _compute_recorded,
            cacheable=_cacheable,
            is_valid=_verdicts_current,
            stage=_L2_L4_STAGE,
        )
        if not computed[0] and isinstance(result, dict):
            # A replayed result never consulted the judges, so feed its
            # recorded rubric tally back in — the stale/legacy report after
            # the run must not depend on which documents were cache hits.
            for policy, counts in zip(
                rubric_policies, result.get(_RUBRIC_COUNTS_KEY) or [], strict=False
            ):
                for kind, state, n in counts:
                    policy.counts[(kind, state)] = policy.counts.get((kind, state), 0) + n
        return result

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

        doc_timed_out: list[str] = []
        try:
            result = _run_doc(
                doc_id,
                doc_dir,
                version_files,
                doc_batch_seg_nodes,
                doc_batch_extractions,
                doc_timed_out,
            )
        except (SegmentationQAError, NormalizeTrailError, HintsError) as exc:
            reason = f"{type(exc).__name__}: {exc}"
            progress(f"    QUARANTINED {doc_id}: {reason}")
            quarantined.append({"document_id": doc_id, "reason": reason})
            # issue #83: a SegmentationQAError raised from inside
            # _compute_doc_result's per-version loop carries a partial
            # corpus_doc snapshot (see _build_quarantine_corpus_doc) —
            # append it so this document's extractor label and QA status
            # still reach corpus_manifest.json instead of the document
            # vanishing from it entirely. quarantine.json (this document_id
            # was just appended to the in-memory `quarantined` list above;
            # the file itself is written later, see below) stays the
            # canonical list of quarantined document_ids either way — this
            # only adds a second, richer record.
            # NormalizeTrailError/HintsError never carry this attribute (it
            # defaults to None on the base class) — a cross-version
            # normalization failure or a malformed hints.yaml has no single
            # failing version to attribute an extractor to, and extending
            # partial-record support to those is out of scope for this fix,
            # so today's behavior (no corpus_documents entry) is unchanged
            # for those two.
            if isinstance(exc, SegmentationQAError) and exc.partial_corpus_doc is not None:
                corpus_documents.append(exc.partial_corpus_doc)
            continue

        if result is None:
            # Every version of this document failed extraction/ingest —
            # _compute_doc_result returned None. Without a durable record the
            # document would vanish from every artifact (no corpus_manifest.json
            # entry, no scope.json entry, no quarantine.json entry) and
            # `validate` would pass on a silently thinner playbook. Quarantine
            # it, same shape as the fail-loud QA-gate entries above.
            reason = "all versions failed extraction/ingest"
            if doc_timed_out:
                # Issue #218: keep the timeout visible — it is the one
                # failure the next run retries (nothing was cached for it).
                # A fixed string: never a path or name (quarantine.json is
                # persisted).
                reason += " (timeout: not cached, the next run will retry)"
            progress(f"    QUARANTINED {doc_id}: {reason}")
            quarantined.append({"document_id": doc_id, "reason": reason})
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

        # Collect this document's serialised clause trees the same way
        # (issue #139) — .get(): a cached result from before this fix has no
        # "version_trees" key and contributes none (see
        # _NORMALIZED_TREES_CACHE_VERSION, which busts those entries so this
        # is a one-time gap, not a permanent one). Unconditional (unlike the
        # trail above): normalized/ has always covered out-of-scope
        # documents too.
        all_version_trees.append((doc_id, result.get("version_trees", {})))

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

    # Fail loud but isolated: surface every quarantined document prominently.
    # This log line names the RAW document_id (console/error output, same
    # convention the fallback-budget message below uses — not subject to the
    # born-safe pseudonymization contract persisted artifacts uphold).
    # quarantine.json itself is written further below, AFTER the born-safe
    # pseudonymization pass (issue #83) — see the comment there for why.
    if quarantined:
        ids = ", ".join(q["document_id"] for q in quarantined)
        progress(
            f"  WARNING: {len(quarantined)} document(s) quarantined for review "
            f"(see quarantine.json): {ids}"
        )

    # Extraction fallback budget check happens AFTER every artifact below is
    # written (see the end of this function) — a run that exceeds the
    # budget still leaves a complete, correct corpus_manifest.json/
    # observations.jsonl behind for the operator to inspect, it just also
    # raises. Computed from corpus_documents captured here, BEFORE the
    # born-safe pseudonymization pass below reassigns that name, so the
    # tally reflects every version_ingest entry regardless of pseudonymization
    # (reason is a closed enum, never a raw name — see _FALLBACK_REASONS).
    fallbacks: list[tuple[str, str, str]] = [
        (doc.get("document_id", "?"), ver.get("version", "?"), ver["reason"])
        for doc in corpus_documents
        for ver in (doc.get("version_ingest", []) or [])
        if isinstance(ver, dict) and ver.get("reason") in _FALLBACK_REASONS
    ]

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

    # Known-entity residue warning (issue #136): if a configured
    # provenance.known_entities name matches none of the raw corpus text,
    # pseudonymize_text (below) has nothing to substitute for it — either the
    # name is a config typo, or it appears in a form
    # entity_registry._fuzzy_name_pattern's contiguous word-sequence matcher
    # can't catch (e.g. a stopword-stripped registry entry like "University
    # <City>" that skips over the "of" an actual "University of <City>"
    # recital uses — the exact shape that let real counterparty names survive
    # into the "shareable" playbook.opf.html, skill-QA finding #57). Checked
    # on RAW observation text, before the pseudonymization pass below
    # rewrites it. Mirrors the our_party_aliases zero-match warning above
    # (issue #182) for the counterparty side.
    known_entities_configured = [e for e in config.provenance.known_entities if e]
    if known_entities_configured:
        unmatched_entities = known_entities_with_no_match(
            known_entities_configured, [obs.full_text for obs in all_observations]
        )
        if unmatched_entities:
            names = ", ".join(repr(n) for n in unmatched_entities)
            progress(
                f"  WARNING: {len(unmatched_entities)} of {len(known_entities_configured)} "
                f"configured provenance.known_entities name(s) match no observation text "
                f"and will NOT be pseudonymized ({names}) — verify each is spelled exactly "
                "as it appears in the corpus (a stopword-stripped or otherwise inexact "
                "spelling silently fails to redact); if the name genuinely never occurs, "
                "remove it. An unmatched known-entity name is a pseudonymization gap, not "
                "a benign no-op."
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
    # Per-document {raw stem: ordinal label} maps (issue #143), populated
    # below when pseudonymization runs — read back by the trail-writing loop
    # further down so trail.json's version labels match corpus_manifest.json's
    # for the same document. Empty (never consulted) when known_entities
    # isn't configured, same as every other pseudonymization artifact here.
    version_alias_by_doc: dict[str, dict[str, str]] = {}
    if known_entities and entity_registry is not None:
        # counterparty_ref (issue #177) needs the RAW names to match a deal
        # to its known entity, so it runs BEFORE the erasing pass below and
        # attaches only the born-safe alias.
        all_observations = _attach_counterparty_refs(
            all_observations, known_entities, entity_registry
        )
        # #143: computed BEFORE the observations/round_moves passes below so
        # their citation.version/citation.version_id get the same exact
        # raw-stem -> ordinal-label coverage corpus_documents/trail already
        # have, instead of relying solely on the whole-word known_entities
        # substring fallback (which misses a squashed, no-whitespace name).
        # Looked up by RAW document_id, matching this dict's keys — see
        # _pseudonymize_corpus_documents's docstring.
        corpus_documents, version_alias_by_doc = _pseudonymize_corpus_documents(
            corpus_documents, known_entities, entity_registry
        )
        all_observations = _pseudonymize_observations(
            all_observations, known_entities, entity_registry, version_alias_by_doc
        )
        t_observations = _pseudonymize_observations(
            t_observations, known_entities, entity_registry, version_alias_by_doc
        )
        all_round_moves = _pseudonymize_round_moves(
            all_round_moves, known_entities, entity_registry, version_alias_by_doc
        )
        # Scope log keys on document_id too (issue #182): alias it so scope.json
        # joins the pseudonymized trail/observation ids in `inspect` instead of
        # producing phantom raw-named entries with "No observations".
        for scope_entry in scope_log.entries:
            scope_entry.document_id = pseudonymize_document_id(
                scope_entry.document_id, known_entities, entity_registry
            )
        # quarantine.json's document_id (issue #83): aliased with the same
        # registry so it matches corpus_manifest.json's document_id exactly.
        # A QA-quarantined document can now ALSO carry a partial
        # corpus_documents entry (see _build_quarantine_corpus_doc above),
        # and a consumer that cross-references the two files by document_id
        # (to avoid double-counting a quarantined document as if it were also
        # successfully mined) would otherwise never find a match — quarantine.json used to be written
        # BEFORE this pass ran, so it kept the raw, un-aliased id even when
        # corpus_manifest.json's matching entry was aliased. Aliasing here
        # also removes the raw counterparty name from quarantine.json's
        # document_id field as a side effect.
        #
        # quarantine.json's reason (issue #96, defense in depth): HintsError
        # and SegmentationQAError are both already built to never embed raw
        # source content at the source (see version_orderer.HintsError's
        # docstring and segmentation_qa._check_coverage), so for those two
        # this pass is a second, redundant layer, not the primary control.
        # It IS load-bearing for any OTHER exception type reaching this
        # quarantine handler whose message happens to embed a known entity
        # name as an ordinary, whole-word-matchable token — e.g. a future
        # NormalizeTrailError raised from a caller-supplied
        # normalize_trail_fn (see
        # test_quarantine_reason_defense_in_depth_pseudonymization_fires).
        # Whole-word match only (entity_registry._fuzzy_name_pattern) — a
        # name glued to a trailing word character defeats it, which is
        # exactly why the source-level "never embed it" fix on
        # HintsError/SegmentationQAError is not optional and this pass alone
        # would not have been enough for either of them.
        #
        # This reason text is read by whoever triages quarantine.json (the
        # skill's checkpoint) — quarantine.json is reason's only consumer.
        # It never reaches corpus_manifest.json or playbook.opf.json.
        quarantined = [
            {
                **q,
                "document_id": pseudonymize_document_id(
                    q["document_id"], known_entities, entity_registry
                ),
                "reason": pseudonymize_text(q["reason"], known_entities, entity_registry),
            }
            for q in quarantined
        ]
        write_holdout_map(out_dir / "alias_map.json", entity_registry)

    # Materialise trails now (issue #182): after the pseudonymization pass so the
    # trail's document_id AND filename carry the alias, never the raw
    # counterparty name — keeping them consistent with the aliased
    # citation.document_id that `inspect` joins on.
    trail_dir = out_dir / "trail"
    trail_dir.mkdir(parents=True, exist_ok=True)
    # Clear stale entries first (issue #51): the loop below rewrites every
    # current trail, so this is a full-rewrite of the directory — mirroring
    # observations.jsonl/corpus_manifest.json semantics. Without this, a
    # document removed/renamed in the corpus leaves a phantom trail/<doc>.json,
    # and — more seriously — a raw-named trail/<RawCounterpartyName>.json from
    # a run BEFORE known_entities was configured survives beside the
    # born-safe aliased file written by a later run, leaking the raw name in
    # both filename and content.
    #
    # Issue #219: only an unchanged trail is left untouched — a warm run no
    # longer rewrites (and re-timestamps) every file; --force-rewrite rewrites
    # them all. Every target path is planned first, the stale-clear runs
    # BEFORE the writes (as it always did) and removes every entry whose exact
    # name this run will not write, then the writes run. Clearing first
    # matters on a case-insensitive filesystem (macOS APFS): after a
    # case-only rename (Delta-Ventures -> delta-ventures) the old-case entry
    # IS the new file, so a clear that ran after the writes and compared
    # names would unlink what was just written. Cleared first, the old-case
    # name is removed and the write recreates the file under the new name.
    planned_trails: dict[Path, str] = {}
    for raw_doc_id, trail in all_trails:
        if entity_registry is not None:
            trail = _pseudonymize_trail(
                trail,
                known_entities,
                entity_registry,
                version_alias_by_doc.get(raw_doc_id),
            )
        out_doc_id = trail.get("document_id") or raw_doc_id
        planned_trails[trail_dir / f"{out_doc_id}.json"] = _json_text(trail)
    for stale_trail in trail_dir.glob("*.json"):
        if stale_trail not in planned_trails:
            stale_trail.unlink()
    for trail_path, trail_text in planned_trails.items():
        write_text_if_changed(trail_path, trail_text, force=force_rewrite)

    # Materialise normalized/ clause trees now (issue #139): after the
    # pseudonymization pass, mirroring trail/'s treatment immediately above —
    # stale-clear the whole directory, then rewrite every version's tree
    # under the ALIASED document_id (never the raw doc_id/counterparty
    # name), from all_version_trees (every document's contribution, cached
    # or freshly computed this run — see _NORMALIZED_TREES_CACHE_VERSION).
    # Previously this was written raw, mid-loop, under doc_id — never
    # stale-cleared and never rewritten on a stage-cache hit (audit finding
    # #48 / issue #139). Like trail/ (issue #219), every target is planned
    # first, then every entry this run will not write (any file at any
    # depth, plus any directory left empty) is cleared BEFORE the writes,
    # and only changed trees are rewritten.
    normalized_dir = out_dir / "normalized"
    if force_rewrite and normalized_dir.exists():
        shutil.rmtree(normalized_dir)
    planned_trees: dict[Path, str] = {}
    for raw_doc_id, doc_version_trees in all_version_trees:
        for vid, tree_dict in doc_version_trees.items():
            if entity_registry is not None:
                tree_dict = _pseudonymize_clause_tree(
                    tree_dict,
                    known_entities,
                    entity_registry,
                    version_alias_by_doc.get(raw_doc_id),
                )
            out_doc_id = tree_dict.get("document_id") or raw_doc_id
            # Use the (possibly aliased) "version" field for the filename,
            # never the raw `vid` dict key directly (issue #139 review round
            # 3): ClauseTree.version is set to vid at ingest time
            # (_compute_doc_result), so before pseudonymization the two are
            # identical and this is a no-op; after it, tree_dict["version"]
            # is the ordinal-labeled alias while `vid` is still the raw,
            # counterparty-name-bearing staged filename stem — building the
            # path from `vid` would leak that name into the output filename
            # even though the directory (out_doc_id) was correctly aliased.
            out_version = tree_dict.get("version") or vid
            tree_path = normalized_dir / out_doc_id / f"{out_version}.clauses.json"
            planned_trees[tree_path] = ClauseTree.from_dict(tree_dict).to_json()
    if normalized_dir.exists():
        # Reverse-sorted, so a directory comes after everything inside it.
        for entry in sorted(normalized_dir.rglob("*"), reverse=True):
            if entry.is_dir():
                if not any(entry.iterdir()):
                    entry.rmdir()
            elif entry not in planned_trees:
                entry.unlink()
    for tree_path, tree_text in planned_trees.items():
        write_text_if_changed(tree_path, tree_text, force=force_rewrite)

    # Write intermediates
    #
    # quarantine.json (issue #83): always rewritten (even when empty), same
    # as every other run-level artifact below — otherwise a stale
    # quarantine.json from a prior run keeps flagging documents a subsequent
    # clean run resolved. Written here (after the pseudonymization pass
    # above, alongside the other post-pseudonymization artifacts) rather
    # than immediately after the per-document loop, so its document_id is
    # aliased exactly like corpus_manifest.json's — see the aliasing above.
    #
    # Issue #219: every artifact below is written only when its content
    # changed (or under --force-rewrite) — see write_text_if_changed.
    write_text_if_changed(out_dir / "quarantine.json", _json_text(quarantined), force=force_rewrite)
    #
    # search_snippet (issue #95) is truncated to its final short-phrase length
    # HERE — unconditionally, regardless of whether known_entities pseudonymization
    # ran above — never earlier: truncating before pseudonymization can bisect
    # a known-entity name mid-word and defeat the whole-word alias match in
    # _pseudonymize_observations, leaking the fragment (same class of bug the
    # round_moves truncation below already guards against for change_summary).
    write_text_if_changed(out_dir / "scope.json", scope_log.to_json_text(), force=force_rewrite)
    write_text_if_changed(
        obs_path,
        observations_jsonl_text(truncate_search_snippets(all_observations)),
        force=force_rewrite,
    )
    # Round moves (issue #177) — written post-pseudonymization like
    # observations.jsonl; project_playbook reads it back for each
    # precedent's rounds/moved (absent file → rounds 0, e.g. a pre-#177 store).
    # Truncation runs strictly AFTER the aliasing above: slicing raw text
    # first can cut an entity name mid-word, and a cut name survives the
    # whole-word pseudonymization match (born-safe leak — review finding).
    write_text_if_changed(
        out_dir / "round_moves.jsonl",
        round_moves_jsonl_text(truncate_move_summaries(all_round_moves)),
        force=force_rewrite,
    )
    # Persist template observations so project_playbook can read them without re-ingesting.
    template_obs_path = out_dir / "template_observations.jsonl"
    write_text_if_changed(
        template_obs_path,
        observations_jsonl_text(truncate_search_snippets(t_observations)),
        force=force_rewrite,
    )
    write_text_if_changed(manifest_path, _json_text(corpus_documents), force=force_rewrite)
    if store is not None:
        # "cache hits=" counts DOCUMENTS (the per-document L2-L4 layer, the
        # line's long-standing meaning); the layer line below breaks the
        # stage cache down (issue #219).
        l1_hits, l1_misses = store.stage_stats(_L1_STAGE)
        doc_hits, doc_misses = store.stage_stats(_L2_L4_STAGE)
        progress(
            f"L1-L4 complete: {len(all_observations)} observations, {len(corpus_documents)} docs "
            f"(cache hits={doc_hits}, misses={doc_misses})"
        )
        progress(
            f"  stage cache: L1 (per version) hits {l1_hits}, misses {l1_misses}; "
            f"L2-L4 (per document) hits {doc_hits}, misses {doc_misses}"
        )
    else:
        progress(
            f"L1-L4 complete: {len(all_observations)} observations, {len(corpus_documents)} docs"
        )

    # Extraction fallback budget (issue #81): count every version_ingest
    # entry across the whole run whose "reason" reflects a DEGRADATION — the
    # file was extracted via the legacy adapter because docling was
    # unavailable ("env-missing"), crashed on this specific file
    # ("backend-error"), or yielded no text that only ocrmypdf recovered
    # ("ocr-recovered", issue #231) — see extraction.ExtractorLabel and _FALLBACK_REASONS
    # above (fallbacks itself was computed earlier, before pseudonymization
    # reassigned corpus_documents — see the comment there). A config-DECLARED
    # "legacy" run ("declared") is a deliberate choice, not a degradation,
    # and never counts here. config.extraction.max_fallback is the number of
    # such degradations a run tolerates before failing outright; None (the
    # default) is unbounded — today's behavior, unchanged. Checked LAST, once
    # every artifact above has already been written correctly — an operator
    # who exceeds the budget still gets a complete, inspectable
    # corpus_manifest.json/observations.jsonl, not a half-finished run; the
    # raise below is a fail-loud POLICY gate on top of that ground truth, not
    # a precondition for producing it. Uses the RAW (pre-pseudonymization)
    # document_id/version, same as every progress() line above that already
    # names raw document_ids — this is local console/error output, not a
    # persisted artifact, so it is not subject to the born-safe
    # pseudonymization contract the artifacts above uphold.
    max_fallback = config.extraction.max_fallback
    if max_fallback is not None and len(fallbacks) > max_fallback:
        offending = ", ".join(
            f"{doc_id}/{version} ({reason})" for doc_id, version, reason in fallbacks
        )
        raise PipelineError(
            f"extraction.max_fallback ({max_fallback}) exceeded: {len(fallbacks)} "
            f"version(s) fell back to the legacy extractor this run — {offending}. "
            "Install/repair docling, raise extraction.max_fallback, or set "
            "extraction.extractor to 'legacy' if this is expected."
        )


@dataclass(frozen=True)
class _ProjectionStore:
    """What L5 reads from a ``mine`` out-dir (see :func:`_read_projection_store`)."""

    corpus_documents: list[dict[str, Any]]
    observations: list[Observation]
    template_observations: list[Observation]
    round_moves: list[RoundMove]
    scope_bases: list[str]


def _read_projection_store(
    out_dir: Path, *, progress: Callable[[str], None] = lambda _: None
) -> _ProjectionStore:
    """Read the observation store L5 compiles from — shared by ``project`` and
    the equivalence queue (issue #240), so both see exactly the same texts.

    Raises:
        PipelineError: the store is missing or empty.
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

    # search_snippet (issue #95) is capped here, not just at mine-write time
    # (pipeline.py's mine_corpus, ~lines 2779/2789): a store mined before
    # search_snippet existed carries no key for it, so _restore_observations
    # defaults it to the UNTRUNCATED full_text (Observation.__post_init__).
    # Re-running truncate_search_snippets on read makes project_playbook
    # correct against such a legacy store too, whether it runs right after
    # mine_corpus in the same process or as the separate `playbook project`
    # command against an older store. Safe to apply post-restore with no
    # re-pseudonymization: whatever full_text/search_snippet already
    # persisted to observations.jsonl went through _pseudonymize_observations
    # before it was written, so it is already born-safe; this is a pure
    # length cap, and _shape_search_snippet is idempotent on values already
    # capped by a fresh store's own mine-time truncation.
    all_observations = truncate_search_snippets(_restore_observations(raw_observations))
    progress(f"  loaded {len(all_observations)} observations, {len(corpus_documents)} docs")

    # Read persisted template observations — no ingest or judge calls.
    template_obs_path = out_dir / "template_observations.jsonl"
    raw_t_observations = read_observations_jsonl(template_obs_path)
    t_observations = truncate_search_snippets(_restore_observations(raw_t_observations))
    if t_observations:
        progress(f"  loaded {len(t_observations)} template observation(s) from store")

    # Round moves (issue #177) — absent for pre-#177 stores or
    # single-version corpora; read_round_moves_jsonl returns [] then and
    # every precedent records rounds = 0.
    round_moves = read_round_moves_jsonl(out_dir / "round_moves.jsonl")
    if round_moves:
        progress(f"  loaded {len(round_moves)} round move(s) from store")

    return _ProjectionStore(
        corpus_documents=corpus_documents,
        observations=all_observations,
        template_observations=t_observations,
        round_moves=round_moves,
        scope_bases=scope_bases,
    )


def _perspective_dict(config: EngineConfig) -> dict[str, str] | None:
    """The playbook's ``perspective`` object (issue #165), or ``None``.

    Emitted only when BOTH party and counterparty_type are known — the OPF
    schema requires the whole object or nothing, and neither field may be
    fabricated (see ``assemble_playbook``'s ``perspective``). A party-only
    default (from ``provenance.our_party_aliases``) lives on
    ``config.perspective`` for other consumers, but is not sufficient on its
    own to answer "what kind of counterparty is across the table".
    """
    if config.perspective.party is not None and config.perspective.counterparty_type is not None:
        return {
            "party": config.perspective.party,
            "counterparty_type": config.perspective.counterparty_type,
        }
    return None


def equivalence_subjects(
    out_dir: Path, config: EngineConfig, taxonomy: Taxonomy
) -> list[EquivalenceSubject]:
    """The distinct texts of *out_dir*'s precedent record to judge against our standard.

    Issue #240. Rebuilds the very evidence ``project_playbook`` would write —
    same store, same clause positions, same invisible-character strip, same
    ``perspective.party`` — and returns :func:`~playbook_engine.equivalence.collect_subjects`
    over it, so the keys a judge round queues are the keys ``project`` later
    looks up. Empty when no canonical template is configured (emergent mode:
    no clause has an ``our_standard``) or when *out_dir* holds no
    observation store yet.
    """
    if not config.baseline.has_canonical_template:
        return []
    try:
        store = _read_projection_store(out_dir)
    except PipelineError:
        return []
    clause_positions, _flags, _coverage = compile_clause_positions(
        store.observations,
        store.template_observations,
        taxonomy_titles={e.id: e.label for e in taxonomy.entries},
    )
    perspective = _perspective_dict(config)
    party = perspective["party"] if perspective else None
    evidence = _strip_invisible(
        build_precedent_evidence(
            agreement_type_id=config.agreement_type.id,
            clause_positions=clause_positions,
            observations=store.observations,
            corpus_documents=store.corpus_documents,
            round_moves=store.round_moves,
            party=party,
        )
    )
    return collect_subjects(evidence, config.agreement_type.id, party)


def queue_equivalence(
    out_dir: Path,
    config: EngineConfig,
    taxonomy: Taxonomy,
    judge: StoreBackedEquivalenceJudge,
) -> tuple[int, int]:
    """Ask *judge* about every equivalence subject of *out_dir*; return ``(subjects, unjudged)``.

    A subject with no replayable stored verdict is queued on the judge's
    pending queue (deduplicated by key).
    """
    subjects = equivalence_subjects(out_dir, config, taxonomy)
    unjudged = sum(1 for subject in subjects if judge.judge(subject) is None)
    return len(subjects), unjudged


def project_playbook(
    out_dir: Path,
    config: EngineConfig,
    taxonomy: Taxonomy,
    *,
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
                                        its ``posture`` and ``floor`` sections, which a
                                        recompile carries forward verbatim (issue #123).
                                        Absent on a first compile.
    - ``overrides.json``              — the owner's optional edits (issue #241), folded
                                        into the projection when present: label
                                        confirmations and changes, Posture and Floor text.
                                        Absent unless the Review tab of ``index.html`` saved it.

    All L5 logic is deterministic given the store — zero LLM calls.

    Args:
        out_dir:         Output directory that already contains the observation store.
        config:          Engine configuration (agreement type, baseline, taxonomy).
        taxonomy:        Loaded taxonomy object.
        progress:        Callable receiving progress message strings.

    Writes ``{out_dir}/playbook.opf.json`` (OPF 0.5, the one format the
    engine emits — issue #238), ``{out_dir}/coherence_flags.json`` (the
    fragment-quarantine warnings) and ``{out_dir}/precedent.jsonl`` — one
    ``evidence.precedent`` record per line, sorted by id, whose sha256 the
    playbook records under ``x_sidecars`` (issue #224; see
    ``write_precedent_sidecar``).

    Returns:
        Validated playbook dict (also written to ``{out_dir}/playbook.opf.json``).

    Raises:
        PipelineError:  If the observation store is missing or empty.
        AssemblyError:  If the assembled playbook fails schema validation.
    """
    store = _read_projection_store(out_dir, progress=progress)
    # -----------------------------------------------------------------------
    # L5: Compile playbook (deterministic given the store)
    # -----------------------------------------------------------------------
    progress("L5: compiling clause types + precedent…")
    corpus_documents = store.corpus_documents
    all_observations = store.observations
    t_observations = store.template_observations
    round_moves = store.round_moves
    scope_bases = store.scope_bases

    taxonomy_titles = {e.id: e.label for e in taxonomy.entries}
    clause_positions, coherence_flags, unclassified_coverage = compile_clause_positions(
        all_observations,
        t_observations,
        taxonomy_titles=taxonomy_titles,
    )

    # Persist coherence flags (fragment quarantine; empty list when none).
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
    perspective_dict = _perspective_dict(config)

    generated_at = datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")

    # Issue #240: each eligible text's ``vs_standard`` is read from the run's
    # verdict store (``<out>/judge/verdicts.jsonl``) when there is one — a
    # read-only judge (no queue), so ``project`` never writes a pending item;
    # a text with no replayable verdict stays null and is reported, not
    # guessed. Only a template-mode run has any ``our_standard`` to judge
    # against; the store file is read only when it exists, so a run that never
    # judged anything validates without a store.
    equivalence_judge: StoreBackedEquivalenceJudge | None = None
    verdicts_path = out_dir / "judge" / "verdicts.jsonl"
    if config.baseline.has_canonical_template and verdicts_path.is_file():
        equivalence_judge = StoreBackedEquivalenceJudge(store=VerdictStore(verdicts_path))
        progress(f"  equivalence verdicts: {verdicts_path}")

    # Issue #123: read the PRIOR compile's `posture` and `floor`, if a
    # playbook already exists in out_dir, and carry them forward VERBATIM.
    # Both are human-authored (`playbook posture interview` / `playbook
    # floor sign`) and live ONLY inside playbook.opf.json — a recompile that doesn't preserve them destroys a
    # GC's signed Posture/Floor with no other recovery path, contradicting
    # SKILL.md Route C's "the Posture and Floor you already signed should
    # survive" promise. `.get(...)` defaults to None (not {}) so an absent
    # key on the prior document is indistinguishable from "nothing to carry
    # forward" for assemble_playbook's existing_posture/existing_floor.
    out_file = out_dir / "playbook.opf.json"
    existing_posture: dict[str, Any] | None = None
    existing_floor: dict[str, Any] | None = None
    if out_file.exists():
        try:
            prior_playbook = json.loads(out_file.read_text(encoding="utf-8"))
            existing_posture = prior_playbook.get("posture")
            existing_floor = prior_playbook.get("floor")
        except (json.JSONDecodeError, OSError):
            existing_posture = None
            existing_floor = None

    playbook = assemble_playbook(
        agreement_type=agreement_type_dict,
        baseline=baseline_dict,
        taxonomy=taxonomy_dict,
        clause_positions=clause_positions,
        corpus_documents=corpus_documents,
        generated_at=generated_at,
        observations=all_observations,
        scope_bases=scope_bases,
        unclassified_coverage=unclassified_coverage,
        perspective=perspective_dict,
        existing_posture=existing_posture,
        existing_floor=existing_floor,
        round_moves=round_moves,
        equivalence_judge=equivalence_judge,
    )

    # Issue #241: the owner's optional corrections (`overrides.json`, saved by
    # the Review tab of index.html) are folded into the projection, and the
    # digest, dossiers and identity recomputed by the engine. Absent file, no
    # step. The file is optional and never gates a run: after a re-derivation
    # from a changed corpus a stale entry is the normal case, so an entry that
    # cannot be applied (or a file that cannot be read) is skipped, counted and
    # reported with its reason, and the projection continues. `playbook
    # apply-overrides` is the strict command.
    overrides_path = out_dir / OVERRIDES_FILENAME
    if overrides_path.is_file():
        skipped: list[str] = []
        try:
            entries = load_overrides(overrides_path)
        except OverridesError as exc:
            entries = []
            skipped = list(exc.problems)
            progress(f"  overrides: {OVERRIDES_FILENAME} could not be used, none applied")
        if entries:
            playbook, applied = fold_overrides(playbook, entries, out_dir, strict=False)
            skipped = applied.skipped
            progress(
                f"  overrides: {len(entries)} entr{'y' if len(entries) == 1 else 'ies'} "
                f"({applied.n_changed} changed the playbook, {applied.unchanged} already in "
                f"effect, {len(skipped)} skipped)"
            )
        for reason in skipped:
            progress(f"  overrides skipped: {reason}")

    write_playbook(playbook, out_file)
    progress(f"Playbook written: {out_file}")
    # One evidence.precedent record per line, sorted by id (issue #224); its
    # sha256 is recorded in the playbook's x_sidecars.
    sidecar = write_precedent_sidecar(playbook, out_file)
    if sidecar is not None:
        progress(f"Precedent sidecar written: {sidecar}")

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
    alignment_judge: AlignmentJudge | None = None,
    trail_judge: TrailJudge | None = None,
    signed_judge: SignedJudge | None = None,
    provenance_judge: ProvenanceJudge | None = None,
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
    refresh_extraction: bool = False,
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
        alignment_judge:      L3 alignment judge; defaults to None (deterministic only).
        trail_judge:          Version-ordering judge; defaults to None (deterministic only).
        signed_judge:         L2 signed-copy judge; defaults to None (deterministic only).
        provenance_judge:     L2 provenance judge; defaults to None (deterministic only).
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
        refresh_extraction:   Passed through to :func:`mine_corpus` unchanged
                              (NOT folded into *effective_no_cache* below —
                              it is a deliberately separate signal; see
                              ``mine_corpus``'s parameter of the same name,
                              issue #78). Defaults to False.
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
        refresh_extraction=refresh_extraction,
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
        progress=progress,
    )
