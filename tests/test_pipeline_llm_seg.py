"""Pipeline integration tests for LLM segmentation wiring (issues #74, #76).

Verifies the three acceptance criteria for the synchronous LLM-segmentation
path (#74):

1. End-to-end on a synthetic corpus with an injected fake ``segment_fn``:
   ``mine_corpus(..., use_llm_segmentation=True)`` produces
   ``observations.jsonl`` whose clauses reflect the LLM tree and carry the
   LLM ``taxonomy_id`` (classified, not all-None).
2. The produced ``ClauseTree`` satisfies the existing downstream contract:
   diff/deviation run without error, and ``playbook validate`` on the
   resulting playbook passes (``ValidationResult.ok`` is True — the
   equivalent of ``playbook validate`` exiting 0).
3. With a fake ``segment_fn`` returning gate-failing output, the run raises
   ``SegmentationQAError`` (fail loud) rather than silently degrading.

Plus the batch-segmentation pre-pass wiring (#76, near the bottom of this
file): ``mine_corpus(..., use_batch_segmentation=True)`` extracts every
version up front and segments the whole corpus via one mocked Message
Batches client, feeding the batched ``SegNode`` output into the same
grounding + QA + observation flow — same classified-observations and
fail-loud-QA contracts as the synchronous path, exercised through the real
``segment_documents_batch`` wiring (not a fake segment function).

SECURITY NOTE: All fixtures use programmatically constructed RTF text with
synthetic, fictional content.  No real agreement files are referenced, and
the fake ``segment_fn``/batches client never calls a live LLM — no network,
no API key. Fictional party names only (e.g. "Alpha Corp", "Beta University").
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

from playbook_engine.aar import build_after_action_data
from playbook_engine.clause_classifier import AMBIGUITY_THRESHOLD
from playbook_engine.clause_tree import ClauseNode, ClauseTree
from playbook_engine.config import load_config
from playbook_engine.extraction import extract_blocks
from playbook_engine.llm_segmenter_batch import (
    NormalizeTrailResult,
    SegmentationVerdictCache,
    segment_documents_batch,
)
from playbook_engine.observation_builder import read_observations_jsonl
from playbook_engine.pipeline import (
    _classified_from_taxonomy_by_path,
    mine_corpus,
    project_playbook,
)
from playbook_engine.segmentation_grounding import Block, SegNode
from playbook_engine.taxonomy import load_taxonomy
from playbook_engine.validator import validate_document

# ---------------------------------------------------------------------------
# RTF fixture helpers (same convention as test_pipeline_project.py /
# test_pipeline_provenance.py — extract_blocks supports .rtf via pandoc, so
# these fixtures exercise the real extractor, not a synthetic Block list).
# ---------------------------------------------------------------------------

_RTF_PROLOGUE = (
    r"{\rtf1\ansi\deff0"
    r"{\fonttbl{\f0\froman\fcharset0 Times New Roman;}}"
    r"\f0\fs24 "
)
_RTF_EPILOGUE = r"}"


def _rtf(body: str) -> str:
    return _RTF_PROLOGUE + body + _RTF_EPILOGUE


def _write_rtf(path: Path, body: str) -> None:
    path.write_text(_rtf(body), encoding="utf-8")


_TAXONOMY_PATH = Path(__file__).parent.parent / "spec" / "taxonomy" / "affiliation-agreement.yaml"

# Two clauses, each "N. Heading\par body text\par" — extract_blocks (RTF via
# pandoc) turns each \par-delimited paragraph into its own Block, in order:
# b0=heading1, b1=body1, b2=heading2, b3=body2. See _fake_segment_fn below,
# which relies on exactly this block-per-paragraph shape.
_V1_BODY = (
    r"1. Indemnification\par "
    r"Alpha Corp shall indemnify Beta University against third-party claims "
    r"arising from the placement programme.\par "
    r"2. Governing Law\par "
    r"This agreement is governed by the laws of the State of California.\par "
)

# v2: signed copy — same clauses, slightly different Indemnification body
# (so the diff/deviation stage has a genuine changed clause to assess), plus
# a signatures block so detect_signed() anchors it as signed.
_V2_BODY = (
    r"1. Indemnification\par "
    r"Alpha Corp shall indemnify and hold harmless Beta University against "
    r"third-party claims arising from the placement programme.\par "
    r"2. Governing Law\par "
    r"This agreement is governed by the laws of the State of California.\par "
    r"3. Signatures\par "
    r"By: Jane Doe, Alpha Corp\par "
)

# Heading text -> taxonomy_id, used by the fake segment_fn to classify each
# clause in the same pass as segmentation (mirrors what the real LLM
# segmenter does per llm_segmenter.SEGMENTER_SYSTEM_PROMPT).
_HEADING_TAXONOMY = {
    "1. Indemnification": "indemnification",
    "2. Governing Law": "governing_law",
}


# ---------------------------------------------------------------------------
# Fake segment_fn — deterministic, block-pair based (no LLM, no network)
# ---------------------------------------------------------------------------


def _fake_segment_fn(canonical_text: str, blocks: list[Block]) -> list[SegNode]:
    """Pair consecutive (heading, body) blocks into clause nodes.

    Matches the RTF fixtures' block-per-paragraph shape from extract_blocks:
    each clause is exactly two blocks (a numbered heading, then its body).
    A block whose text isn't in ``_HEADING_TAXONOMY`` (e.g. "3. Signatures")
    is still covered by a node (every block must be accounted for), but with
    ``taxonomy_id=None`` — the LLM's own convention for non-clause noise.
    """
    del canonical_text
    nodes: list[SegNode] = []
    order = 1
    i = 0
    while i < len(blocks):
        heading_block = blocks[i]
        tid = _HEADING_TAXONOMY.get(heading_block.text)
        if tid is None:
            # Non-clause block (e.g. a signatures heading + its one body
            # block) — still emit a node covering both so coverage holds.
            end_block = blocks[i + 1] if i + 1 < len(blocks) else heading_block
            nodes.append(
                SegNode(
                    node_id=f"n{order}",
                    parent_id=None,
                    order=order,
                    heading=heading_block.text,
                    taxonomy_id=None,
                    start_block_id=heading_block.block_id,
                    end_block_id=end_block.block_id,
                )
            )
            i += 2 if end_block is not heading_block else 1
        else:
            body_block = blocks[i + 1]
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
                    end_quote=body_block.text[-10:],
                )
            )
            i += 2
        order += 1
    return nodes


def _gate_failing_segment_fn(canonical_text: str, blocks: list[Block]) -> list[SegNode]:
    """Return a segmentation that fails the coverage gate on every attempt.

    Only covers the first block, leaving the rest of ``canonical_text``
    uncovered — a deterministic, always-failing candidate so
    ``segment_verify_repair`` exhausts every repair attempt and
    ``SegmentationQAError`` propagates (acceptance criterion 3).
    """
    del canonical_text
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


# ---------------------------------------------------------------------------
# Corpus + config factory
# ---------------------------------------------------------------------------


def _make_corpus(tmp_path: Path, *, two_versions: bool) -> tuple[Path, Path, Path]:
    """Build a synthetic corpus + config; return (corpus_dir, config_path, out_dir)."""
    corpus_dir = tmp_path / "corpus"
    deal_dir = corpus_dir / "deal-001"
    deal_dir.mkdir(parents=True)
    _write_rtf(deal_dir / "v1.rtf", _V1_BODY)
    if two_versions:
        _write_rtf(deal_dir / "v2.rtf", _V2_BODY)

    cfg = {
        "agreement_type": {
            "id": "educational-affiliation",
            "name": "Educational Affiliation Agreement",
        },
        "baseline": {"template": None},
        "taxonomy": str(_TAXONOMY_PATH),
        "provenance": {"our_party_aliases": ["Alpha Corp"]},
    }
    config_path = tmp_path / "playbook.config.yaml"
    config_path.write_text(yaml.dump(cfg), encoding="utf-8")

    out_dir = tmp_path / "out"
    return corpus_dir, config_path, out_dir


# ---------------------------------------------------------------------------
# AC-1: end-to-end — observations carry the LLM taxonomy_id (classified)
# ---------------------------------------------------------------------------


def test_llm_segmentation_produces_classified_observations(tmp_path: Path) -> None:
    """mine_corpus(use_llm_segmentation=True) with a fake segment_fn produces
    observations whose taxonomy_id reflects the LLM tree — not all None.
    """
    corpus_dir, config_path, out_dir = _make_corpus(tmp_path, two_versions=False)
    taxonomy = load_taxonomy(_TAXONOMY_PATH)
    cfg = load_config(config_path)

    mine_corpus(
        corpus_dir=corpus_dir,
        config=cfg,
        taxonomy=taxonomy,
        out_dir=out_dir,
        use_llm_segmentation=True,
        llm_segment_fn=_fake_segment_fn,
    )

    obs_path = out_dir / "observations.jsonl"
    assert obs_path.exists()
    raw_obs = read_observations_jsonl(obs_path)
    assert raw_obs, "mine_corpus must write at least one observation"

    taxonomy_ids = [o["taxonomy_id"] for o in raw_obs]
    assert not all(tid is None for tid in taxonomy_ids), (
        "LLM-segmented observations must carry the LLM's taxonomy_id (classified), not be all-None"
    )
    assert set(taxonomy_ids) == {"indemnification", "governing_law"}

    # basis="judge" — carried straight from the LLM pass, no classify_tree call.
    assert all(o["basis"] in ("judge", "deterministic") for o in raw_obs)


def test_llm_segmented_normalized_tree_is_written(tmp_path: Path) -> None:
    """The per-version ClauseTree written to normalized/ reflects the LLM tree
    (two top-level clauses) and carries the real document_id/version identity
    — not run_gates' "doc"/"v1" placeholder defaults.
    """
    corpus_dir, config_path, out_dir = _make_corpus(tmp_path, two_versions=False)
    taxonomy = load_taxonomy(_TAXONOMY_PATH)
    cfg = load_config(config_path)

    mine_corpus(
        corpus_dir=corpus_dir,
        config=cfg,
        taxonomy=taxonomy,
        out_dir=out_dir,
        use_llm_segmentation=True,
        llm_segment_fn=_fake_segment_fn,
    )

    tree_path = out_dir / "normalized" / "deal-001" / "v1.clauses.json"
    assert tree_path.exists()
    tree = ClauseTree.load(tree_path)
    assert tree.document_id == "deal-001"
    assert tree.version == "v1"
    assert tree.source_file == "v1.rtf"
    assert [n.heading for n in tree.nodes] == ["1. Indemnification", "2. Governing Law"]


# ---------------------------------------------------------------------------
# Issue #86: LLM-segmenter taxonomy assignments must not be asserted as
# confidence=1.0/basis="judge" — a single unverified LLM pass over untrusted
# counterparty text is not a real judge verdict, and downstream
# confidence-based review gating must actually be able to flag it.
# ---------------------------------------------------------------------------


def test_classified_from_taxonomy_by_path_is_not_asserted_confidence_1_judge() -> None:
    """An LLM-assigned taxonomy_id must not come back as confidence=1.0/basis="judge".

    That combination masquerades a single unverified LLM pass as a real,
    separately-verified judge verdict, and is indistinguishable downstream
    from an actual ClassificationJudge call. Assert the replacement: a
    dedicated basis distinct from "judge", and a confidence low enough that
    ``ClauseClassification.is_ambiguous`` (below ``AMBIGUITY_THRESHOLD``) is
    True — i.e. this assignment is flagged as needing review, not treated as
    certain.
    """
    tree = ClauseTree(
        document_id="deal-001",
        version="v1",
        source_file="v1.rtf",
        nodes=[
            ClauseNode(
                clause_path="1",
                heading="Indemnification",
                text="Alpha Corp shall indemnify Beta University.",
                char_span=(0, 10),
            )
        ],
    )
    classified = _classified_from_taxonomy_by_path(tree, {"1": "indemnification"})

    assert len(classified) == 1
    cc = classified[0].classification
    assert cc.taxonomy_id == "indemnification"
    assert not (cc.confidence == 1.0 and cc.basis == "judge"), (
        "LLM-segmented taxonomy assignment must not be asserted as a real "
        "judge verdict (confidence=1.0, basis='judge')"
    )
    assert cc.basis != "judge"
    assert cc.confidence < AMBIGUITY_THRESHOLD
    assert cc.is_ambiguous, "a below-threshold LLM-segmenter confidence must read as ambiguous"


def test_llm_segmentation_low_confidence_surfaces_in_after_action_review(tmp_path: Path) -> None:
    """A below-threshold LLM-segmenter classification confidence must actually
    reach the after-action report's needs-attention section — the concrete
    downstream confidence-based review gate in this repo (aar._build_needs_attention,
    fed by classification_confidences -> build_observations -> Observation.confidence).

    Before the fix, confidence=1.0 meant these clauses could never trip this
    gate; the LLM misclassifying an indemnification clause would present as a
    certainty in the observation store.
    """
    corpus_dir, config_path, out_dir = _make_corpus(tmp_path, two_versions=False)
    taxonomy = load_taxonomy(_TAXONOMY_PATH)
    cfg = load_config(config_path)

    mine_corpus(
        corpus_dir=corpus_dir,
        config=cfg,
        taxonomy=taxonomy,
        out_dir=out_dir,
        use_llm_segmentation=True,
        llm_segment_fn=_fake_segment_fn,
    )
    project_playbook(out_dir=out_dir, config=cfg, taxonomy=taxonomy)

    raw_obs = read_observations_jsonl(out_dir / "observations.jsonl")
    classified_obs = [o for o in raw_obs if o["taxonomy_id"] is not None]
    assert classified_obs, "must have at least one LLM-classified observation to check"
    assert all(o["confidence"] < 0.5 for o in classified_obs), (
        "LLM-segmented classifications must carry a confidence below the "
        "after-action report's low-confidence threshold"
    )

    data = build_after_action_data(out_dir)
    needs_attention = data["needs_attention"]
    assert any("low confidence" in "".join(item.get("reasons", [])) for item in needs_attention), (
        f"expected a low-confidence needs_attention item; got {needs_attention}"
    )


# ---------------------------------------------------------------------------
# AC-2: downstream contract — diff/deviation + playbook validate
# ---------------------------------------------------------------------------


def test_llm_segmentation_two_versions_diff_and_playbook_validate(tmp_path: Path) -> None:
    """Two LLM-segmented versions run through diff/deviation without error,
    and the resulting playbook passes playbook-schema validation.
    """
    corpus_dir, config_path, out_dir = _make_corpus(tmp_path, two_versions=True)
    taxonomy = load_taxonomy(_TAXONOMY_PATH)
    cfg = load_config(config_path)

    mine_corpus(
        corpus_dir=corpus_dir,
        config=cfg,
        taxonomy=taxonomy,
        out_dir=out_dir,
        use_llm_segmentation=True,
        llm_segment_fn=_fake_segment_fn,
    )

    raw_obs = read_observations_jsonl(out_dir / "observations.jsonl")
    assert raw_obs, "two-version diff must still produce observations"
    # The signed version (v2) has 2 real clauses + the "Signatures" noise
    # node (taxonomy_id=None) — deviation assessment must not choke on the
    # unclassified node, and classified clauses must still show up.
    assert {"indemnification", "governing_law"} <= {o["taxonomy_id"] for o in raw_obs}

    playbook = project_playbook(out_dir=out_dir, config=cfg, taxonomy=taxonomy)

    result = validate_document(playbook)
    assert result.ok, f"playbook validate must pass: {[str(e) for e in result.errors]}"


# ---------------------------------------------------------------------------
# AC-3: QA-gate failure fails loud but ISOLATED — the failing document is
# quarantined (recorded in quarantine.json), not silently degraded to the
# deterministic segmenter and not aborting the whole corpus run.
# ---------------------------------------------------------------------------


def test_gate_failing_segment_fn_quarantines_document(tmp_path: Path) -> None:
    """A segment_fn whose output never passes the QA gates must quarantine that
    document — recorded in quarantine.json with a SegmentationQAError reason —
    rather than silently degrading the tree, silently dropping the version, or
    aborting the whole run. Here the only document fails, so the run completes
    with zero observations but a populated quarantine.json.
    """
    corpus_dir, config_path, out_dir = _make_corpus(tmp_path, two_versions=False)
    taxonomy = load_taxonomy(_TAXONOMY_PATH)
    cfg = load_config(config_path)

    # Does NOT raise — the run completes despite the QA failure.
    mine_corpus(
        corpus_dir=corpus_dir,
        config=cfg,
        taxonomy=taxonomy,
        out_dir=out_dir,
        use_llm_segmentation=True,
        llm_segment_fn=_gate_failing_segment_fn,
    )

    assert not read_observations_jsonl(out_dir / "observations.jsonl"), (
        "the quarantined document must contribute no observations"
    )
    quarantine = json.loads((out_dir / "quarantine.json").read_text(encoding="utf-8"))
    assert [q["document_id"] for q in quarantine] == ["deal-001"]
    assert "SegmentationQAError" in quarantine[0]["reason"]


# ---------------------------------------------------------------------------
# Segmentation verdict cache on the synchronous LLM path (issue #91)
# ---------------------------------------------------------------------------


def test_segmentation_cache_hits_on_sync_path(tmp_path: Path) -> None:
    """A SegmentationVerdictCache passed as segmentation_cache must be honored
    on the synchronous per-document LLM path too (not just the batch
    pre-pass): a second mine_corpus run (with the L1-L4 stage cache disabled
    so _compute_doc_result actually re-executes) hits the segmentation cache
    and never re-invokes the injected segment_fn.
    """
    corpus_dir, config_path, out_dir = _make_corpus(tmp_path, two_versions=False)
    taxonomy = load_taxonomy(_TAXONOMY_PATH)
    cfg = load_config(config_path)

    seg_cache = SegmentationVerdictCache(tmp_path / "seg_cache.jsonl")
    call_count = 0

    def _counting_segment_fn(canonical_text: str, blocks: list[Block]) -> list[SegNode]:
        nonlocal call_count
        call_count += 1
        return _fake_segment_fn(canonical_text, blocks)

    common_kwargs: dict[str, Any] = {
        "corpus_dir": corpus_dir,
        "config": cfg,
        "taxonomy": taxonomy,
        "use_llm_segmentation": True,
        "llm_segment_fn": _counting_segment_fn,
        "segmentation_cache": seg_cache,
        "no_cache": True,  # disable the unrelated L1-L4 stage cache
    }

    mine_corpus(out_dir=out_dir, **common_kwargs)
    assert call_count == 1

    out_dir_2 = tmp_path / "out2"
    mine_corpus(out_dir=out_dir_2, **common_kwargs)
    # No second LLM call for the second run — the content-hash cache
    # satisfied the version without touching segment_fn at all.
    assert call_count == 1

    raw_obs = read_observations_jsonl(out_dir_2 / "observations.jsonl")
    assert {o["taxonomy_id"] for o in raw_obs} == {"indemnification", "governing_law"}


# ---------------------------------------------------------------------------
# Deterministic path unaffected — regression guard
# ---------------------------------------------------------------------------


def test_default_use_llm_segmentation_false_unaffected(tmp_path: Path) -> None:
    """use_llm_segmentation defaults to False: the deterministic segmenter
    path (segment(ingest(...).tree)) must remain the default, unchanged.
    """
    corpus_dir, config_path, out_dir = _make_corpus(tmp_path, two_versions=False)
    taxonomy = load_taxonomy(_TAXONOMY_PATH)
    cfg = load_config(config_path)

    # No use_llm_segmentation kwarg at all — must not require llm_segment_fn
    # and must not touch the LLM path (no fake injected, so any accidental
    # LLM-path call would try to construct a real anthropic client and fail).
    mine_corpus(
        corpus_dir=corpus_dir,
        config=cfg,
        taxonomy=taxonomy,
        out_dir=out_dir,
    )

    raw_obs = read_observations_jsonl(out_dir / "observations.jsonl")
    assert raw_obs
    # Deterministic path: the fast-path heading matcher in clause_classifier
    # (exact/Jaccard match against the taxonomy label) still classifies
    # obvious headings like "Indemnification" without any judge — the stub
    # _NullClassificationJudge only affects nodes that reach the judge. The
    # regression guard that matters here is basis: LLM-segmented
    # observations get basis="judge" straight from the LLM pass (see the
    # AC-1 test above); the deterministic path must never produce that
    # basis, since no judge (real or stub) here ever returns basis="judge"
    # (_NullClassificationJudge always returns basis="unclassified").
    bases = {o["basis"] for o in raw_obs}
    assert "judge" not in bases, (
        f"deterministic path must not produce basis='judge' observations; got {bases}"
    )


# ---------------------------------------------------------------------------
# Cross-version normalization wiring (issue #75) — mine_corpus must forward the
# opt-in down to _compute_doc_result, and the normalized labels must be used.
# ---------------------------------------------------------------------------


def test_mine_corpus_forwards_normalize_trail_opt_in(tmp_path: Path) -> None:
    """mine_corpus(normalize_trail_across_versions=True, normalize_trail_fn=...)
    must actually invoke the injected normalize_fn over the version trail and
    feed its normalized taxonomy back into classification.

    Regression guard: the opt-in params were once accepted by mine_corpus but
    never forwarded to _compute_doc_result, so the pass silently never ran.
    """
    corpus_dir, config_path, out_dir = _make_corpus(tmp_path, two_versions=True)
    taxonomy = load_taxonomy(_TAXONOMY_PATH)
    cfg = load_config(config_path)

    received: list[dict[str, dict[str, str | None]]] = []

    def _fake_normalize_fn(
        version_trees: dict[str, ClauseTree],
        taxonomy_by_version: dict[str, dict[str, str | None]],
    ) -> NormalizeTrailResult:
        # Record the call, then normalize by dropping every "indemnification"
        # label to None — an observable change that proves the returned result
        # is actually used, not merely that the fn was called.
        received.append(taxonomy_by_version)
        remapped = {
            vid: {path: (None if tid == "indemnification" else tid) for path, tid in labels.items()}
            for vid, labels in taxonomy_by_version.items()
        }
        return NormalizeTrailResult(taxonomy_by_version=remapped, boundary_flags=[])

    mine_corpus(
        corpus_dir=corpus_dir,
        config=cfg,
        taxonomy=taxonomy,
        out_dir=out_dir,
        use_llm_segmentation=True,
        llm_segment_fn=_fake_segment_fn,
        normalize_trail_across_versions=True,
        normalize_trail_fn=_fake_normalize_fn,
    )

    # Called exactly once, over a trail of both versions.
    assert len(received) == 1, "normalize_trail_fn must be invoked once per agreement"
    assert len(received[0]) == 2, "the normalize pass must see every version in the trail"

    # The normalized result was used: the dropped "indemnification" label no
    # longer appears in any observation, while the untouched label survives.
    non_null = {o["taxonomy_id"] for o in read_observations_jsonl(out_dir / "observations.jsonl")}
    non_null.discard(None)
    assert "indemnification" not in non_null, "normalized labels must replace the per-version ones"
    assert "governing_law" in non_null, "untouched labels must survive normalization"


def test_normalize_trail_not_run_for_single_version(tmp_path: Path) -> None:
    """The cross-version pass is a no-op for a single-version agreement
    (len(version_trees) == 1) even when opted in — nothing to normalize across.
    """
    corpus_dir, config_path, out_dir = _make_corpus(tmp_path, two_versions=False)
    taxonomy = load_taxonomy(_TAXONOMY_PATH)
    cfg = load_config(config_path)

    calls: list[dict[str, dict[str, str | None]]] = []

    def _spy_normalize_fn(
        version_trees: dict[str, ClauseTree],
        taxonomy_by_version: dict[str, dict[str, str | None]],
    ) -> NormalizeTrailResult:
        calls.append(taxonomy_by_version)
        return NormalizeTrailResult(taxonomy_by_version=taxonomy_by_version, boundary_flags=[])

    mine_corpus(
        corpus_dir=corpus_dir,
        config=cfg,
        taxonomy=taxonomy,
        out_dir=out_dir,
        use_llm_segmentation=True,
        llm_segment_fn=_fake_segment_fn,
        normalize_trail_across_versions=True,
        normalize_trail_fn=_spy_normalize_fn,
    )

    assert calls == [], "single-version agreement must not trigger the cross-version pass"


# ---------------------------------------------------------------------------
# Batch segmentation pre-pass wiring (issue #76) — mocked Message Batches
# client exercised through the real segment_documents_batch/mine_corpus
# wiring, not a fake segment_fn.
# ---------------------------------------------------------------------------


def _seg_nodes_to_response_text(seg_nodes: list[SegNode]) -> str:
    """Serialize SegNodes into the structured-output JSON text a batch result carries.

    Mirrors ``llm_segmenter._parse_seg_nodes``'s expected ``{"nodes": [...]}``
    shape exactly (see that function) — the fake batches client below returns
    this text as a "succeeded" result's message content.
    """
    return json.dumps(
        {
            "nodes": [
                {
                    "node_id": n.node_id,
                    "parent_id": n.parent_id,
                    "order": n.order,
                    "heading": n.heading,
                    "taxonomy_id": n.taxonomy_id,
                    "start_block_id": n.start_block_id,
                    "end_block_id": n.end_block_id,
                    "start_quote": n.start_quote,
                    "end_quote": n.end_quote,
                }
                for n in seg_nodes
            ]
        }
    )


def _gate_failing_response_for(blocks: list[Block]) -> list[SegNode]:
    """Same always-failing shape as ``_gate_failing_segment_fn`` above, reused
    by the fake batches client's canned "gate_failing" mode."""
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


class _FakeBatchesResource:
    """Minimal fake of the Anthropic Message Batches surface.

    Computes each custom_id's canned response text lazily from
    *blocks_by_custom_id* at ``.results()`` time via *segment_fn* — the same
    live-recompute-from-blocks approach ``_fake_segment_fn`` itself uses,
    rather than hardcoding block ids into a canned JSON string that could
    silently drift from the fixture text.
    """

    def __init__(
        self,
        blocks_by_custom_id: dict[str, list[Block]],
        segment_fn: Any,
        *,
        gate_failing: bool = False,
    ) -> None:
        self._blocks_by_custom_id = blocks_by_custom_id
        self._segment_fn = segment_fn
        self._gate_failing = gate_failing
        self.create_calls: list[dict[str, Any]] = []
        self.retrieve_calls: list[str] = []
        self.results_calls: list[str] = []
        self._requests_by_batch_id: dict[str, list[dict[str, Any]]] = {}

    def create(self, **kwargs: Any) -> Any:
        self.create_calls.append(kwargs)
        batch_id = f"batch_{len(self.create_calls)}"
        self._requests_by_batch_id[batch_id] = kwargs["requests"]
        return SimpleNamespace(id=batch_id, processing_status="ended")

    def retrieve(self, batch_id: str) -> Any:
        self.retrieve_calls.append(batch_id)
        return SimpleNamespace(id=batch_id, processing_status="ended")

    def results(self, batch_id: str) -> list[Any]:
        self.results_calls.append(batch_id)
        out = []
        for req in self._requests_by_batch_id[batch_id]:
            custom_id = req["custom_id"]
            blocks = self._blocks_by_custom_id[custom_id]
            seg_nodes = (
                _gate_failing_response_for(blocks)
                if self._gate_failing
                else self._segment_fn("", blocks)
            )
            text = _seg_nodes_to_response_text(seg_nodes)
            result = SimpleNamespace(
                type="succeeded",
                message=SimpleNamespace(content=[SimpleNamespace(type="text", text=text)]),
            )
            out.append(SimpleNamespace(custom_id=custom_id, result=result))
        return out


class _FakeBatchClient:
    """Fake Anthropic client exposing only the batches surface segment_documents_batch uses."""

    def __init__(
        self, blocks_by_custom_id: dict[str, list[Block]], *, gate_failing: bool = False
    ) -> None:
        self.messages = SimpleNamespace(
            batches=_FakeBatchesResource(
                blocks_by_custom_id, _fake_segment_fn, gate_failing=gate_failing
            )
        )


def _make_batch_client(corpus_dir: Path, *, gate_failing: bool = False) -> _FakeBatchClient:
    """Build a fake batches client pre-registered with every version's blocks.

    Extracts every ``.rtf`` file under *corpus_dir* the same way
    ``_collect_batch_items`` does and keys each version's block stream by its
    ``{doc_id}/{version}`` custom_id, so the fake's canned response for a
    given custom_id always matches the fixture it corresponds to.
    """
    blocks_by_custom_id: dict[str, list[Block]] = {}
    for doc_dir in sorted(d for d in corpus_dir.iterdir() if d.is_dir()):
        for vf in sorted(doc_dir.glob("*.rtf")):
            _canonical_text, blocks, _extractor = extract_blocks(vf)
            blocks_by_custom_id[f"{doc_dir.name}/{vf.stem}"] = blocks
    return _FakeBatchClient(blocks_by_custom_id, gate_failing=gate_failing)


def test_batch_segmentation_produces_classified_observations(tmp_path: Path) -> None:
    """mine_corpus(use_batch_segmentation=True) with a mocked batches client
    produces observations whose taxonomy_id reflects the batched SegNode
    output — same classified-observations contract as the synchronous path,
    but through the real segment_documents_batch wiring.
    """
    corpus_dir, config_path, out_dir = _make_corpus(tmp_path, two_versions=False)
    taxonomy = load_taxonomy(_TAXONOMY_PATH)
    cfg = load_config(config_path)

    client = _make_batch_client(corpus_dir)

    def _batch_fn(items: Any, *, taxonomy_ids: Any, cache: Any = None, **_kwargs: Any) -> Any:
        return segment_documents_batch(
            items, taxonomy_ids=taxonomy_ids, client=client, cache=cache, poll_interval_s=0
        )

    mine_corpus(
        corpus_dir=corpus_dir,
        config=cfg,
        taxonomy=taxonomy,
        out_dir=out_dir,
        use_llm_segmentation=True,
        use_batch_segmentation=True,
        segment_documents_batch_fn=_batch_fn,
    )

    raw_obs = read_observations_jsonl(out_dir / "observations.jsonl")
    assert raw_obs, "mine_corpus must write at least one observation"

    taxonomy_ids = [o["taxonomy_id"] for o in raw_obs]
    assert not all(tid is None for tid in taxonomy_ids), (
        "batch-segmented observations must carry the batched taxonomy_id, not be all-None"
    )
    assert set(taxonomy_ids) == {"indemnification", "governing_law"}

    # Exactly one corpus-wide batch call was made (not one per document).
    assert len(client.messages.batches.create_calls) == 1
    submitted_ids = {r["custom_id"] for r in client.messages.batches.create_calls[0]["requests"]}
    assert submitted_ids == {"deal-001/v1"}


def test_batch_segmentation_two_versions_diff_and_playbook_validate(tmp_path: Path) -> None:
    """Two batch-segmented versions run through diff/deviation without error,
    and the resulting playbook passes playbook-schema validation — the
    downstream contract must hold identically to the synchronous LLM path.
    """
    corpus_dir, config_path, out_dir = _make_corpus(tmp_path, two_versions=True)
    taxonomy = load_taxonomy(_TAXONOMY_PATH)
    cfg = load_config(config_path)

    client = _make_batch_client(corpus_dir)

    def _batch_fn(items: Any, *, taxonomy_ids: Any, cache: Any = None, **_kwargs: Any) -> Any:
        return segment_documents_batch(
            items, taxonomy_ids=taxonomy_ids, client=client, cache=cache, poll_interval_s=0
        )

    mine_corpus(
        corpus_dir=corpus_dir,
        config=cfg,
        taxonomy=taxonomy,
        out_dir=out_dir,
        use_llm_segmentation=True,
        use_batch_segmentation=True,
        segment_documents_batch_fn=_batch_fn,
    )

    raw_obs = read_observations_jsonl(out_dir / "observations.jsonl")
    assert raw_obs, "two-version diff must still produce observations"
    assert {"indemnification", "governing_law"} <= {o["taxonomy_id"] for o in raw_obs}

    # Both versions' custom_ids went into the single corpus-wide batch call.
    submitted_ids = {r["custom_id"] for r in client.messages.batches.create_calls[0]["requests"]}
    assert submitted_ids == {"deal-001/v1", "deal-001/v2"}

    playbook = project_playbook(out_dir=out_dir, config=cfg, taxonomy=taxonomy)

    result = validate_document(playbook)
    assert result.ok, f"playbook validate must pass: {[str(e) for e in result.errors]}"


def test_batch_segmentation_gate_failure_quarantines_document(tmp_path: Path) -> None:
    """A batched SegNode result that fails the QA gates at grounding time must
    quarantine that document (fail loud, recorded in quarantine.json) — no
    repair loop, no silent fallback to a worse tree, no aborting the whole run.
    Same isolated-quarantine contract as the synchronous path.
    """
    corpus_dir, config_path, out_dir = _make_corpus(tmp_path, two_versions=False)
    taxonomy = load_taxonomy(_TAXONOMY_PATH)
    cfg = load_config(config_path)

    client = _make_batch_client(corpus_dir, gate_failing=True)

    def _batch_fn(items: Any, *, taxonomy_ids: Any, cache: Any = None, **_kwargs: Any) -> Any:
        return segment_documents_batch(
            items, taxonomy_ids=taxonomy_ids, client=client, cache=cache, poll_interval_s=0
        )

    mine_corpus(
        corpus_dir=corpus_dir,
        config=cfg,
        taxonomy=taxonomy,
        out_dir=out_dir,
        use_llm_segmentation=True,
        use_batch_segmentation=True,
        segment_documents_batch_fn=_batch_fn,
    )

    assert not read_observations_jsonl(out_dir / "observations.jsonl"), (
        "the quarantined document must contribute no observations"
    )
    quarantine = json.loads((out_dir / "quarantine.json").read_text(encoding="utf-8"))
    assert [q["document_id"] for q in quarantine] == ["deal-001"]
    assert "SegmentationQAError" in quarantine[0]["reason"]


def test_batch_segmentation_cache_avoids_second_batch_call(tmp_path: Path) -> None:
    """A SegmentationVerdictCache passed as segmentation_cache must be honored:
    a second mine_corpus run (with the L1-L4 stage cache disabled so
    _compute_doc_result actually re-executes) hits the segmentation cache and
    makes no second batch .create() call.
    """
    corpus_dir, config_path, out_dir = _make_corpus(tmp_path, two_versions=False)
    taxonomy = load_taxonomy(_TAXONOMY_PATH)
    cfg = load_config(config_path)

    client = _make_batch_client(corpus_dir)
    seg_cache = SegmentationVerdictCache(tmp_path / "seg_cache.jsonl")

    def _batch_fn(items: Any, *, taxonomy_ids: Any, cache: Any = None, **_kwargs: Any) -> Any:
        return segment_documents_batch(
            items, taxonomy_ids=taxonomy_ids, client=client, cache=cache, poll_interval_s=0
        )

    common_kwargs: dict[str, Any] = {
        "corpus_dir": corpus_dir,
        "config": cfg,
        "taxonomy": taxonomy,
        "use_llm_segmentation": True,
        "use_batch_segmentation": True,
        "segment_documents_batch_fn": _batch_fn,
        "segmentation_cache": seg_cache,
        "no_cache": True,  # disable the unrelated L1-L4 stage cache
    }

    mine_corpus(out_dir=out_dir, **common_kwargs)
    assert len(client.messages.batches.create_calls) == 1

    out_dir_2 = tmp_path / "out2"
    mine_corpus(out_dir=out_dir_2, **common_kwargs)
    # No new batch submitted for the second run — the content-hash cache
    # satisfied every item without touching the client at all.
    assert len(client.messages.batches.create_calls) == 1

    raw_obs = read_observations_jsonl(out_dir_2 / "observations.jsonl")
    assert {o["taxonomy_id"] for o in raw_obs} == {"indemnification", "governing_law"}


def test_batch_segmentation_default_false_unaffected(tmp_path: Path) -> None:
    """use_batch_segmentation defaults to False: the synchronous per-document
    LLM path (use_llm_segmentation alone) must remain unaffected — no
    unrequested batch call, no requirement to pass segment_documents_batch_fn.
    """
    corpus_dir, config_path, out_dir = _make_corpus(tmp_path, two_versions=False)
    taxonomy = load_taxonomy(_TAXONOMY_PATH)
    cfg = load_config(config_path)

    # No use_batch_segmentation kwarg, no segment_documents_batch_fn — any
    # accidental batch-path call would try to construct a real anthropic
    # client and fail.
    mine_corpus(
        corpus_dir=corpus_dir,
        config=cfg,
        taxonomy=taxonomy,
        out_dir=out_dir,
        use_llm_segmentation=True,
        llm_segment_fn=_fake_segment_fn,
    )

    raw_obs = read_observations_jsonl(out_dir / "observations.jsonl")
    assert raw_obs
    assert {o["taxonomy_id"] for o in raw_obs} == {"indemnification", "governing_law"}


# ---------------------------------------------------------------------------
# Issue #92: batch pre-pass must exclude L1-L4 stage-cache hits
# ---------------------------------------------------------------------------


def test_batch_prepass_skips_stage_cache_hits(tmp_path: Path) -> None:
    """A document whose L1-L4 stage cache is already warm must be excluded
    from the batch-segmentation pre-pass entirely on a re-run.

    Without the fix, every document's version files are extracted and
    submitted to the (paid) Message Batch on every run — including documents
    the per-document loop below is about to replay verbatim from
    ``ArtifactStore``, whose batched results are then simply thrown away.

    Runs ``mine_corpus`` three times over the same ``out_dir`` (so the
    on-disk stage cache carries over between calls) with a fake
    ``segment_documents_batch_fn`` that records the custom_ids it was asked
    to submit on each call:

    1. Both documents are new -> both submitted.
    2. Nothing changed -> both are cache hits -> zero items submitted.
    3. Only deal-002 changes -> deal-001 (still cached) must be excluded;
       only deal-002's version is submitted.
    """
    corpus_dir = tmp_path / "corpus"
    deal1_dir = corpus_dir / "deal-001"
    deal2_dir = corpus_dir / "deal-002"
    deal1_dir.mkdir(parents=True)
    deal2_dir.mkdir(parents=True)
    _write_rtf(deal1_dir / "v1.rtf", _V1_BODY)
    _write_rtf(deal2_dir / "v1.rtf", _V1_BODY)

    cfg = {
        "agreement_type": {
            "id": "educational-affiliation",
            "name": "Educational Affiliation Agreement",
        },
        "baseline": {"template": None},
        "taxonomy": str(_TAXONOMY_PATH),
        "provenance": {"our_party_aliases": ["Alpha Corp"]},
    }
    config_path = tmp_path / "playbook.config.yaml"
    config_path.write_text(yaml.dump(cfg), encoding="utf-8")
    out_dir = tmp_path / "out"

    taxonomy = load_taxonomy(_TAXONOMY_PATH)
    config = load_config(config_path)

    submitted_by_call: list[set[str]] = []

    def _make_tracking_batch_fn() -> Any:
        # A fresh fake client per call (mirrors _make_batch_client's registry
        # picking up whatever the corpus currently contains on disk).
        client = _make_batch_client(corpus_dir)

        def _batch_fn(items: Any, *, taxonomy_ids: Any, cache: Any = None, **_kwargs: Any) -> Any:
            submitted_by_call.append({item.custom_id for item in items})
            return segment_documents_batch(
                items, taxonomy_ids=taxonomy_ids, client=client, cache=cache, poll_interval_s=0
            )

        return _batch_fn

    run_kwargs: dict[str, Any] = {
        "corpus_dir": corpus_dir,
        "config": config,
        "taxonomy": taxonomy,
        "out_dir": out_dir,
        "use_llm_segmentation": True,
        "use_batch_segmentation": True,
    }

    # 1) First run: both documents are new — both go into the pre-pass.
    mine_corpus(segment_documents_batch_fn=_make_tracking_batch_fn(), **run_kwargs)
    assert submitted_by_call[-1] == {"deal-001/v1", "deal-002/v1"}

    # 2) Second run, same out_dir (warm stage cache): nothing changed on
    # disk — both documents are cache hits, so the pre-pass must submit
    # zero items.
    mine_corpus(segment_documents_batch_fn=_make_tracking_batch_fn(), **run_kwargs)
    assert submitted_by_call[-1] == set()

    # 3) Third run: only deal-002 changes. deal-001 is still a cache hit and
    # must stay excluded from the pre-pass — only deal-002's version goes in.
    _write_rtf(deal2_dir / "v1.rtf", _V2_BODY)
    mine_corpus(segment_documents_batch_fn=_make_tracking_batch_fn(), **run_kwargs)
    assert submitted_by_call[-1] == {"deal-002/v1"}

    # Sanity: the corpus as a whole still produced observations for both
    # documents across the store (final manifest reflects both).
    raw_obs = read_observations_jsonl(out_dir / "observations.jsonl")
    assert {o["citation"]["document_id"] for o in raw_obs} >= {"deal-001", "deal-002"}
