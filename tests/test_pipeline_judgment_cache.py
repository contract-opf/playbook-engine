"""Pipeline-level judgment-cache tests — issue #62.

AC-1 (pipeline): two compile runs over the same corpus → the second run makes
ZERO judge calls (all verdicts served from the persisted verdict cache).

The test installs counting stub judges *before* mine_corpus wraps them in
BatchedScopeJudge / BatchedClassificationJudge / BatchedDeviationJudge so we
can observe the raw delegate call counts.

Finding 1 isolation strategy
-----------------------------
The *original* tautological test ran both runs with the default no_cache=False,
which meant the #61 stage cache (ArtifactStore) short-circuited _compute_doc_result
on the second run — so judges were never reached regardless of whether the #62
verdict cache existed.

The rewrite defeats the #61 stage cache between runs while keeping the #62 verdict
cache live:
  - Run 1 executes normally (no_cache=False) → populates both the stage cache and
    the verdict cache.
  - Between runs: delete the ArtifactStore index.json (and its artifact files) so
    the stage cache is empty.  The verdicts.jsonl file is left intact.
  - Run 2 executes with no_cache=False → stage cache is empty so _compute_doc_result
    runs for every document.  The ONLY thing that can produce zero judge calls is the
    persistent verdict cache reading verdicts.jsonl.

The test will FAIL if the verdict cache is bypassed, because _compute_doc_result
will run (stage cache is empty) and the raw delegate judges will be called.

SECURITY NOTE: All fixtures use programmatically constructed RTF text with
synthetic, fictional content.  No real agreement files or party names are used.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import yaml

from playbook_engine.artifact_store import make_config_fingerprint
from playbook_engine.clause_classifier import ClauseClassification
from playbook_engine.config import load_config
from playbook_engine.deviation_classifier import DeviationResult, RiskDelta
from playbook_engine.observation_builder import read_observations_jsonl
from playbook_engine.pipeline import mine_corpus
from playbook_engine.scope_gate import ScopeDecision
from playbook_engine.segmentation_grounding import Block, SegNode
from playbook_engine.taxonomy import load_taxonomy

# ---------------------------------------------------------------------------
# RTF fixture helpers (identical pattern to test_pipeline_project.py)
# ---------------------------------------------------------------------------

_RTF_PROLOGUE = (
    r"{\rtf1\ansi\deff0"
    r"{\fonttbl{\f0\froman\fcharset0 Times New Roman;}}"
    r"\f0\fs24 "
)
_RTF_EPILOGUE = r"}"

_TAXONOMY_PATH = Path(__file__).parent.parent / "spec" / "taxonomy" / "affiliation-agreement.yaml"

_CORPUS_BODY = (
    r"1. Indemnification\par "
    r"Gamma LLC shall indemnify Delta University against third-party claims "
    r"arising from the placement programme.\par "
    r"2. Governing Law\par "
    r"This agreement is governed by the laws of the State of Delaware.\par "
    r"3. Term\par "
    r"This agreement commences on the date of execution and continues for two years.\par "
)


def _rtf(body: str) -> str:
    return _RTF_PROLOGUE + body + _RTF_EPILOGUE


def _write_rtf(path: Path, body: str) -> None:
    path.write_text(_rtf(body), encoding="utf-8")


# ---------------------------------------------------------------------------
# Counting stub judges — count raw delegate calls, not cache-layer calls
# ---------------------------------------------------------------------------

_NEUTRAL_RISK = RiskDelta(direction="neutral", magnitude="none")


@dataclass
class _CountingScopeJudge:
    call_count: int = 0

    def judge(self, tree: Any, agreement_type: Any) -> ScopeDecision:
        self.call_count += 1
        return ScopeDecision(
            in_scope=True,
            scope_rationale="Synthetic in-scope (pipeline test).",
            scope_confidence=0.9,
            basis="judge",
        )


@dataclass
class _CountingOutOfScopeJudge:
    """A differently-classed scope judge that returns the opposite verdict.

    Used to prove that a verdict cached under ``_CountingScopeJudge`` is never
    replayed once a differently-identified judge is injected (issue #102) —
    the class name alone must be enough to bust both the #61 stage cache
    (via config_fp's judge_identity field) and the #62 verdict cache.
    """

    call_count: int = 0

    def judge(self, tree: Any, agreement_type: Any) -> ScopeDecision:
        self.call_count += 1
        return ScopeDecision(
            in_scope=False,
            scope_rationale="Real judge verdict — must not be masked by a stale stub cache.",
            scope_confidence=0.85,
            basis="judge",
        )


@dataclass
class _CountingClassificationJudge:
    call_count: int = 0
    total_nodes: int = 0

    def classify_batch(
        self,
        nodes: list[Any],
        taxonomy: Any,
        hints: Any = None,
    ) -> list[ClauseClassification]:
        self.call_count += 1
        self.total_nodes += len(nodes)
        return [
            ClauseClassification(taxonomy_id=None, confidence=0.0, basis="unclassified")
            for _ in nodes
        ]


@dataclass
class _CountingDeviationJudge:
    call_count: int = 0
    total_items: int = 0

    def assess_batch(
        self,
        items: list[dict[str, str]],
        our_standard: str,
    ) -> list[DeviationResult]:
        self.call_count += 1
        self.total_items += len(items)
        return [
            DeviationResult(deviation="substantive", risk_delta=_NEUTRAL_RISK, basis="judge")
            for _ in items
        ]


# ---------------------------------------------------------------------------
# Corpus + config factory
# ---------------------------------------------------------------------------


def _make_corpus(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Build a synthetic two-document corpus; return (corpus_dir, config_path, out_dir)."""
    corpus_dir = tmp_path / "corpus"
    for doc_name, _body in [("deal-alpha", _CORPUS_BODY), ("deal-beta", _CORPUS_BODY)]:
        doc_dir = corpus_dir / doc_name
        doc_dir.mkdir(parents=True)
        # Two versions so L4 diff + deviation judge is exercised.
        _write_rtf(doc_dir / "v1.rtf", _CORPUS_BODY)
        _write_rtf(
            doc_dir / "v2.rtf",
            _CORPUS_BODY.replace("two years", "three years"),
        )

    cfg = {
        "agreement_type": {
            "id": "educational-affiliation",
            "name": "Educational Affiliation Agreement",
        },
        "baseline": {},
        "taxonomy": str(_TAXONOMY_PATH),
        "provenance": {"our_party_aliases": ["Gamma LLC"]},
    }
    config_path = tmp_path / "playbook.config.yaml"
    config_path.write_text(yaml.dump(cfg), encoding="utf-8")

    out_dir = tmp_path / "out"
    return corpus_dir, config_path, out_dir


def _bust_stage_cache(out_dir: Path) -> None:
    """Remove the #61 ArtifactStore entries (index + artifact files) from out_dir/.cache/,
    leaving verdicts.jsonl intact so the #62 verdict cache survives.

    This forces _compute_doc_result to run on the next mine_corpus call while
    keeping verdict-cache hits available — isolating the #62 cache as the sole
    source of zero judge calls on the second run.
    """
    cache_dir = out_dir / ".cache"
    if not cache_dir.exists():
        return
    # Remove the ArtifactStore index so all stage-cache lookups miss.
    index_path = cache_dir / "index.json"
    if index_path.exists():
        index_path.unlink()
    # Remove per-artifact subdirectories (two-char hex prefix dirs).
    for child in cache_dir.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
    # verdicts.jsonl is a flat file in cache_dir — it is NOT removed here.


# ---------------------------------------------------------------------------
# AC-1 pipeline test (Finding-1 rewrite)
# ---------------------------------------------------------------------------


def test_second_compile_run_makes_zero_judge_calls(tmp_path: Path) -> None:
    """AC-1: second mine_corpus run over the same corpus makes zero judge calls.

    Isolation strategy (Finding 1):
      The #61 stage cache (ArtifactStore) is busted between runs by deleting
      the index.json and artifact files from out/.cache/ while leaving
      verdicts.jsonl intact.  This forces _compute_doc_result to execute on the
      second run — so if the verdict cache were absent or bypassed, the delegate
      judges would be called.  The ONLY reason the delegate judges see zero calls
      on the second run is that the #62 verdict cache serves every verdict from
      verdicts.jsonl.

    This test FAILS if the verdict cache is bypassed (e.g. if JudgmentCache is
    not constructed, or verdicts.jsonl is cleared).
    """
    corpus_dir, config_path, out_dir = _make_corpus(tmp_path)
    taxonomy = load_taxonomy(_TAXONOMY_PATH)
    cfg = load_config(config_path)

    # -----------------------------------------------------------------------
    # First run — populates both the stage cache and the verdict cache.
    # -----------------------------------------------------------------------
    scope_judge_1 = _CountingScopeJudge()
    cls_judge_1 = _CountingClassificationJudge()
    dev_judge_1 = _CountingDeviationJudge()

    mine_corpus(
        corpus_dir=corpus_dir,
        config=cfg,
        taxonomy=taxonomy,
        out_dir=out_dir,
        scope_judge=scope_judge_1,
        classification_judge=cls_judge_1,
        deviation_judge=dev_judge_1,
        no_cache=False,
    )

    # Sanity-check: first run must have made at least some judge calls.
    assert scope_judge_1.call_count > 0, "First run must call the scope judge at least once"

    # Verify verdicts.jsonl was written (the #62 cache populated).
    verdicts_path = out_dir / ".cache" / "verdicts.jsonl"
    assert verdicts_path.exists(), "First run must populate verdicts.jsonl"

    # -----------------------------------------------------------------------
    # Bust the #61 stage cache while keeping verdicts.jsonl.
    # After this, _compute_doc_result WILL run on the second call — the only
    # thing that can suppress judge calls is the #62 verdict cache.
    # -----------------------------------------------------------------------
    _bust_stage_cache(out_dir)

    # Confirm the stage cache is cleared.
    assert not (out_dir / ".cache" / "index.json").exists(), (
        "Stage cache index must be removed before second run"
    )
    # Confirm verdicts.jsonl is still present.
    assert verdicts_path.exists(), "verdicts.jsonl must survive stage-cache bust"

    # -----------------------------------------------------------------------
    # Second run — stage cache is empty so _compute_doc_result runs for each
    # document, but verdicts.jsonl has all verdicts from run 1.
    # -----------------------------------------------------------------------
    scope_judge_2 = _CountingScopeJudge()
    cls_judge_2 = _CountingClassificationJudge()
    dev_judge_2 = _CountingDeviationJudge()

    mine_corpus(
        corpus_dir=corpus_dir,
        config=cfg,
        taxonomy=taxonomy,
        out_dir=out_dir,
        scope_judge=scope_judge_2,
        classification_judge=cls_judge_2,
        deviation_judge=dev_judge_2,
        no_cache=False,
    )

    assert scope_judge_2.call_count == 0, (
        f"Second run must make zero scope judge calls (verdict cache hit); "
        f"got {scope_judge_2.call_count}"
    )
    assert cls_judge_2.call_count == 0, (
        f"Second run must make zero classification judge calls (verdict cache hit); "
        f"got {cls_judge_2.call_count}"
    )
    assert dev_judge_2.call_count == 0, (
        f"Second run must make zero deviation judge calls (verdict cache hit); "
        f"got {dev_judge_2.call_count}"
    )


# ---------------------------------------------------------------------------
# Finding 3 — no_cache=True must bypass the verdict cache
# ---------------------------------------------------------------------------


def test_no_cache_bypasses_verdict_cache(tmp_path: Path) -> None:
    """Finding 3: no_cache=True must re-invoke judges even when verdicts.jsonl exists.

    Steps:
      1. Run mine_corpus with no_cache=False → populates verdicts.jsonl.
      2. Run mine_corpus again with no_cache=True → judges must be called again;
         the verdict cache must NOT serve stale verdicts.
    """
    corpus_dir, config_path, out_dir = _make_corpus(tmp_path)
    taxonomy = load_taxonomy(_TAXONOMY_PATH)
    cfg = load_config(config_path)

    # Run 1: populate the verdict cache normally.
    scope_judge_1 = _CountingScopeJudge()
    mine_corpus(
        corpus_dir=corpus_dir,
        config=cfg,
        taxonomy=taxonomy,
        out_dir=out_dir,
        scope_judge=scope_judge_1,
        no_cache=False,
    )
    assert scope_judge_1.call_count > 0, "First run must call the scope judge"
    verdicts_path = out_dir / ".cache" / "verdicts.jsonl"
    assert verdicts_path.exists(), "verdicts.jsonl must exist after first run"

    # Run 2 with no_cache=True: even though verdicts.jsonl exists, judges must be re-called.
    scope_judge_2 = _CountingScopeJudge()
    cls_judge_2 = _CountingClassificationJudge()
    dev_judge_2 = _CountingDeviationJudge()

    mine_corpus(
        corpus_dir=corpus_dir,
        config=cfg,
        taxonomy=taxonomy,
        out_dir=out_dir,
        scope_judge=scope_judge_2,
        classification_judge=cls_judge_2,
        deviation_judge=dev_judge_2,
        no_cache=True,
    )

    assert scope_judge_2.call_count > 0, (
        "no_cache=True must bypass the verdict cache and re-invoke the scope judge; "
        f"got {scope_judge_2.call_count} calls (expected > 0)"
    )


# ---------------------------------------------------------------------------
# Issue #102 — judge identity must be part of both caches' keys
# ---------------------------------------------------------------------------


def test_verdict_not_replayed_after_judge_class_change(tmp_path: Path) -> None:
    """A verdict cached under one judge class must MISS — both stage cache and
    verdict cache — once a differently-classed judge is injected, with NO
    manual stage-cache busting between runs.

    Before issue #102: JudgmentCache was constructed with a hardcoded
    model_id="stub-v1" and the #61 stage-cache config fingerprint didn't
    encode judge identity at all. So swapping ``_CountingScopeJudge`` (always
    in-scope) for ``_CountingOutOfScopeJudge`` (always out-of-scope) between
    runs, with the same corpus/config/out_dir, would silently replay run 1's
    cached in-scope result via the #61 ArtifactStore stage cache — the run-2
    judge would never even be called, and scope.json would still show
    in_scope=True.

    This test asserts the run-2 judge WAS called and that scope.json reflects
    its (opposite) verdict — proving neither cache masked the judge change.
    """
    corpus_dir, config_path, out_dir = _make_corpus(tmp_path)
    taxonomy = load_taxonomy(_TAXONOMY_PATH)
    cfg = load_config(config_path)

    # Run 1 — populates both caches under _CountingScopeJudge (always in-scope).
    scope_judge_1 = _CountingScopeJudge()
    mine_corpus(
        corpus_dir=corpus_dir,
        config=cfg,
        taxonomy=taxonomy,
        out_dir=out_dir,
        scope_judge=scope_judge_1,
        no_cache=False,
    )
    assert scope_judge_1.call_count > 0, "First run must call the scope judge at least once"

    scope_log_1 = json.loads((out_dir / "scope.json").read_text(encoding="utf-8"))
    assert all(doc["in_scope"] is True for doc in scope_log_1["documents"]), (
        "First run's documents must be in-scope per _CountingScopeJudge"
    )

    # Run 2 — SAME out_dir, NO stage-cache busting, differently-classed judge.
    scope_judge_2 = _CountingOutOfScopeJudge()
    mine_corpus(
        corpus_dir=corpus_dir,
        config=cfg,
        taxonomy=taxonomy,
        out_dir=out_dir,
        scope_judge=scope_judge_2,
        no_cache=False,
    )

    assert scope_judge_2.call_count > 0, (
        "Second run must call the new judge — a stale stage-cache or verdict-"
        f"cache hit would skip it entirely; got {scope_judge_2.call_count} calls"
    )

    scope_log_2 = json.loads((out_dir / "scope.json").read_text(encoding="utf-8"))
    assert all(doc["in_scope"] is False for doc in scope_log_2["documents"]), (
        "Second run must reflect _CountingOutOfScopeJudge's real verdict, not a "
        f"stale in-scope replay from run 1: {scope_log_2['documents']}"
    )
    assert all(doc["basis"] == "judge" for doc in scope_log_2["documents"])
    assert all(
        doc["scope_rationale"] == "Real judge verdict — must not be masked by a stale stub cache."
        for doc in scope_log_2["documents"]
    )


def test_config_fingerprint_differs_across_judge_identity(tmp_path: Path) -> None:
    """The #61 stage-cache config fingerprint must differ when judge identity
    differs, holding every other field constant (issue #102).

    Mirrors the ``judge_identity`` field ``mine_corpus`` folds into its
    ``make_config_fingerprint`` call — two otherwise-identical config dicts
    that differ only in ``judge_identity`` must fingerprint differently, or a
    judge swap would never bust the #61 stage cache at all.
    """
    base_fields = {
        "agreement_type_id": "educational-affiliation",
        "provenance_aliases": ["Gamma LLC"],
        "template_content_hash": None,
        "use_llm_segmentation": False,
        "use_batch_segmentation": False,
        "normalize_trail_across_versions": False,
        "segmentation_model": "n/a",
        "segmentation_prompt_version": "n/a",
        "segmentation_schema_hash": "n/a",
        "segmentation_effort": "n/a",
    }

    fp_stub = make_config_fingerprint(
        {
            **base_fields,
            "judge_identity": '{"classification": "_NullClassificationJudge:unversioned"}',
        }
    )
    fp_real = make_config_fingerprint(
        {
            **base_fields,
            "judge_identity": '{"classification": "_CountingClassificationJudge:unversioned"}',
        }
    )

    assert fp_stub != fp_real, (
        "make_config_fingerprint must produce different fingerprints for "
        "different judge_identity values with every other field held constant"
    )


def test_config_fingerprint_differs_across_extractor_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The #61 stage-cache config fingerprint must carry the extractor
    environment (issue #79): it must differ across a legacy/docling switch
    and stay stable across repeat runs in the SAME environment.

    Before issue #79, ``config_fp`` had no ``extractor_env`` component, so
    installing (or removing) docling between two ``mine_corpus`` runs over an
    UNCHANGED corpus left every #61 per-doc stage-cache key unchanged too —
    every L1-L4 result derived from the OLD environment's extraction would be
    replayed verbatim under the NEW one, even though legacy (no OCR) and
    docling can disagree on the same source bytes.

    Drives three real ``mine_corpus`` runs over the same corpus/out_dir with
    ``shutil.which`` monkeypatched (the same PATH check
    ``extraction.detect_extractor`` makes) and inspects the #61
    ``ArtifactStore`` index directly — entries are never evicted (see
    ``ArtifactStore``'s docstring), so a same-environment repeat run must add
    ZERO new keys and an environment switch must add exactly one new key per
    document. Uses only the default stub judges — the deterministic RTF
    ingest path this corpus exercises never itself calls docling, so any
    behavioural difference observed here comes solely from ``config_fp``.
    """
    corpus_dir, config_path, out_dir = _make_corpus(tmp_path)
    taxonomy = load_taxonomy(_TAXONOMY_PATH)
    cfg = load_config(config_path)
    index_path = out_dir / ".cache" / "index.json"

    # Run 1 — simulate a legacy-only host (docling absent from PATH).
    monkeypatch.setattr(shutil, "which", lambda _cmd: None)
    mine_corpus(
        corpus_dir=corpus_dir, config=cfg, taxonomy=taxonomy, out_dir=out_dir, no_cache=False
    )
    keys_legacy_1 = set(json.loads(index_path.read_text(encoding="utf-8")).keys())
    assert len(keys_legacy_1) == 2, (
        f"expected one stage-cache key per document (deal-alpha, deal-beta), got {keys_legacy_1}"
    )

    # Run 2 — SAME (legacy) environment, no manual cache-busting: an
    # unchanged environment must produce a STABLE fingerprint, i.e. a pure
    # cache replay that adds no new keys.
    mine_corpus(
        corpus_dir=corpus_dir, config=cfg, taxonomy=taxonomy, out_dir=out_dir, no_cache=False
    )
    keys_legacy_2 = set(json.loads(index_path.read_text(encoding="utf-8")).keys())
    assert keys_legacy_2 == keys_legacy_1, (
        "a repeat run in the SAME extractor environment must not add new stage-cache "
        f"keys — the fingerprint must be stable within one environment; got {keys_legacy_2}"
    )

    # Run 3 — SAME corpus/out_dir, but docling now "installed": must bust the
    # #61 stage cache for every document (a NEW key per doc) rather than
    # replaying the legacy-derived entries under an environment that can
    # produce materially different L1 text (docling has OCR; legacy does not).
    monkeypatch.setattr(
        shutil, "which", lambda cmd: "/usr/bin/docling" if cmd == "docling" else None
    )
    mine_corpus(
        corpus_dir=corpus_dir, config=cfg, taxonomy=taxonomy, out_dir=out_dir, no_cache=False
    )
    keys_docling = set(json.loads(index_path.read_text(encoding="utf-8")).keys())
    new_keys = keys_docling - keys_legacy_1
    assert len(new_keys) == 2, (
        "switching extractor environments (legacy -> docling) with no other config "
        "change must produce a NEW stage-cache key per document, not replay the "
        f"legacy-derived entries verbatim; index now has {keys_docling}"
    )


# ---------------------------------------------------------------------------
# Issue #243 — config_fp must fold in the template's classification OUTCOME,
# not just the template file's content hash, so a template-pending -> template-
# available transition busts the #61 stage cache.
# ---------------------------------------------------------------------------

_HEADING_TAXONOMY = {
    "1. Indemnification": "indemnification",
    "2. Governing Law": "governing_law",
}

# Unique to the template's Indemnification clause — used by
# _template_pending_segment_fn below to detect "this is the template" from
# content alone (SegmentFn only receives (canonical_text, blocks), never a
# document id).
_TEMPLATE_MARKER = "hold harmless"

_TEMPLATE_BODY = (
    r"1. Indemnification\par "
    r"Gamma LLC shall indemnify and hold harmless Delta University from any and all "
    r"claims, liabilities, and damages whatsoever, without limitation, arising under "
    r"or in connection with this agreement in perpetuity.\par "
    r"2. Governing Law\par "
    r"This agreement is governed by the laws of the State of Delaware.\par "
)

# Indemnification clause text deliberately shares little vocabulary with
# _TEMPLATE_BODY's (Jaccard similarity well under the 0.92 reworded-equivalent
# threshold) so it routes to the deviation judge once a template standard
# exists. Governing Law is identical to the template's — a same-text control
# clause that must stay deterministic in both runs.
_DOC_BODY = (
    r"1. Indemnification\par "
    r"Gamma LLC shall provide reasonable indemnification to Delta University limited "
    r"solely to direct damages arising from Gamma LLC's own negligence.\par "
    r"2. Governing Law\par "
    r"This agreement is governed by the laws of the State of Delaware.\par "
)


def _pairing_segment_fn(canonical_text: str, blocks: list[Block]) -> list[SegNode]:
    """Pair consecutive (heading, body) blocks into classified clause nodes.

    Matches the RTF fixtures' block-per-paragraph shape from extract_blocks:
    each clause is exactly two blocks (a numbered heading, then its body).
    Segments *and* classifies correctly for both the template and the
    document — the "template segmentation succeeds" half of issue #243's
    repro.
    """
    del canonical_text
    nodes: list[SegNode] = []
    order = 1
    i = 0
    while i < len(blocks):
        heading_block = blocks[i]
        tid = _HEADING_TAXONOMY.get(heading_block.text)
        body_block = blocks[i + 1] if i + 1 < len(blocks) else heading_block
        nodes.append(
            SegNode(
                node_id=f"n{order}",
                parent_id=None,
                order=order,
                heading=heading_block.text,
                taxonomy_id=tid,
                start_block_id=heading_block.block_id,
                end_block_id=body_block.block_id,
                start_quote=heading_block.text[:10],
                end_quote=(body_block.text[-10:] if body_block is not heading_block else ""),
            )
        )
        i += 2 if body_block is not heading_block else 1
        order += 1
    return nodes


def _template_pending_segment_fn(canonical_text: str, blocks: list[Block]) -> list[SegNode]:
    """Segment the corpus document fine; fail the template's coverage gate.

    Detects "this is the template" via ``_TEMPLATE_MARKER`` (unique to the
    template's Indemnification clause) since ``SegmentFn`` only receives
    ``(canonical_text, blocks)`` — never a document id. Mirrors
    test_pipeline_llm_seg.py's ``_gate_failing_segment_fn``: covering only
    the first block leaves the rest of ``canonical_text`` uncovered, so
    ``segment_verify_repair`` exhausts every repair attempt and
    ``SegmentationQAError`` propagates — exactly the "template's segmentation
    verdict is not yet in the store" scenario from the issue.
    """
    if _TEMPLATE_MARKER in canonical_text:
        first = blocks[0]
        return [
            SegNode(
                node_id="n1",
                parent_id=None,
                order=1,
                heading=first.text,
                taxonomy_id=None,
                start_block_id=first.block_id,
                end_block_id=first.block_id,
            )
        ]
    return _pairing_segment_fn(canonical_text, blocks)


def _make_template_corpus(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Single-doc, single-version corpus + a template file; return (corpus_dir, config_path, out_dir)."""
    corpus_dir = tmp_path / "corpus"
    doc_dir = corpus_dir / "deal-alpha"
    doc_dir.mkdir(parents=True)
    _write_rtf(doc_dir / "v1.rtf", _DOC_BODY)

    template_path = tmp_path / "template.rtf"
    _write_rtf(template_path, _TEMPLATE_BODY)

    cfg = {
        "agreement_type": {
            "id": "educational-affiliation",
            "name": "Educational Affiliation Agreement",
        },
        "baseline": {"template": "template.rtf"},
        "taxonomy": str(_TAXONOMY_PATH),
        "provenance": {"our_party_aliases": ["Gamma LLC"]},
    }
    config_path = tmp_path / "playbook.config.yaml"
    config_path.write_text(yaml.dump(cfg), encoding="utf-8")

    out_dir = tmp_path / "out"
    return corpus_dir, config_path, out_dir


def test_stage_cache_busts_when_template_becomes_available(tmp_path: Path) -> None:
    """issue #243: the #61 L1-L4 stage cache must not replay a template-less
    result once the template's segmentation verdict becomes available.

    Run 1: ``use_llm_segmentation=True`` with a segment_fn that raises
    ``SegmentationQAError`` for the template only (the corpus document
    segments fine) — ``mine_corpus`` catches it, warns, and proceeds in
    emergent mode (``template_std_by_tid == {}``). The document's
    Indemnification clause has no template standard to compare against, so
    it is recorded ``deviation="none"``/``basis="deterministic"`` and the
    per-doc result is cached under ``config_fp``.

    Run 2: SAME out_dir, NO manual stage-cache busting, a working segment_fn
    that classifies the template too. Before issue #243's fix, ``config_fp``
    only hashed the template file's *content* — unchanged between runs — so
    the #61 stage cache would replay run 1's cached result verbatim: the
    recording deviation judge would never be called and the clause would
    stay ``basis="deterministic"`` forever, despite a real template standard
    now existing to diff it against. This test asserts the judge WAS called
    and the clause's basis reflects the real comparison, not a stale replay.
    """
    corpus_dir, config_path, out_dir = _make_template_corpus(tmp_path)
    taxonomy = load_taxonomy(_TAXONOMY_PATH)
    cfg = load_config(config_path)

    # -----------------------------------------------------------------------
    # Run 1 — template segmentation pending (SegmentationQAError caught),
    # document segments fine.
    # -----------------------------------------------------------------------
    mine_corpus(
        corpus_dir=corpus_dir,
        config=cfg,
        taxonomy=taxonomy,
        out_dir=out_dir,
        use_llm_segmentation=True,
        llm_segment_fn=_template_pending_segment_fn,
        deviation_judge=_CountingDeviationJudge(),
    )

    obs_path = out_dir / "observations.jsonl"
    raw_obs_1 = read_observations_jsonl(obs_path)
    indem_obs_1 = [o for o in raw_obs_1 if o["taxonomy_id"] == "indemnification"]
    assert indem_obs_1, "Run 1 must produce an Indemnification observation"
    assert all(o["basis"] == "deterministic" for o in indem_obs_1), (
        "Run 1 (template pending) must record the Indemnification clause "
        f"deterministically — no template standard exists yet: {indem_obs_1}"
    )

    # -----------------------------------------------------------------------
    # Run 2 — SAME out_dir, no manual cache busting. Template now classifies.
    # -----------------------------------------------------------------------
    dev_judge_2 = _CountingDeviationJudge()
    mine_corpus(
        corpus_dir=corpus_dir,
        config=cfg,
        taxonomy=taxonomy,
        out_dir=out_dir,
        use_llm_segmentation=True,
        llm_segment_fn=_pairing_segment_fn,
        deviation_judge=dev_judge_2,
    )

    assert dev_judge_2.call_count > 0, (
        "Run 2 must call the deviation judge for the Indemnification clause "
        "now that the template is available — a stale #61 stage-cache hit "
        f"would skip it entirely; got {dev_judge_2.call_count} calls"
    )

    raw_obs_2 = read_observations_jsonl(obs_path)
    indem_obs_2 = [o for o in raw_obs_2 if o["taxonomy_id"] == "indemnification"]
    assert indem_obs_2, "Run 2 must produce an Indemnification observation"
    assert all(o["basis"] == "judge" for o in indem_obs_2), (
        "Run 2 must route the Indemnification clause (differs from the now-"
        "available template) through the judge, not replay run 1's "
        f"template-less deterministic result verbatim: {indem_obs_2}"
    )
