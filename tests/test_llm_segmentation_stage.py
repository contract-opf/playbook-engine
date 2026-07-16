"""Tests for llm_segmentation_stage — thin L1 orchestration seam (issue #74).

All tests inject a fake ``segment_fn``; none construct a real Anthropic
client or make a network call. Fixtures use a synthetic RTF file (extracted
via the real ``extract_blocks``/pandoc path) so grounding operates on a real
Block stream, not a hand-rolled one.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from playbook_engine.llm_segmentation_stage import segment_to_tree
from playbook_engine.segmentation_grounding import Block, GroundingResult, SegNode
from playbook_engine.segmentation_qa import SegmentationQAError

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

_RTF_PROLOGUE = (
    r"{\rtf1\ansi\deff0"
    r"{\fonttbl{\f0\froman\fcharset0 Times New Roman;}}"
    r"\f0\fs24 "
)
_RTF_EPILOGUE = r"}"

_BODY = (
    r"1. Indemnification\par "
    r"Alpha Corp shall indemnify Beta University against third-party claims.\par "
)


def _write_rtf(path: Path, body: str) -> None:
    path.write_text(_RTF_PROLOGUE + body + _RTF_EPILOGUE, encoding="utf-8")


def _two_block_segment_fn(canonical_text: str, blocks: list[Block]) -> list[SegNode]:
    """One clause spanning both blocks (heading + body)."""
    del canonical_text
    return [
        SegNode(
            node_id="n1",
            parent_id=None,
            order=1,
            heading=blocks[0].text,
            taxonomy_id="indemnification",
            start_block_id=blocks[0].block_id,
            end_block_id=blocks[-1].block_id,
        )
    ]


def _gate_failing_segment_fn(canonical_text: str, blocks: list[Block]) -> list[SegNode]:
    """Covers only the first block — fails the coverage gate."""
    del canonical_text
    return [
        SegNode(
            node_id="n1",
            parent_id=None,
            order=1,
            heading=None,
            taxonomy_id=None,
            start_block_id=blocks[0].block_id,
            end_block_id=blocks[0].block_id,
        )
    ]


# ---------------------------------------------------------------------------
# segment_to_tree — happy path with an injected segment_fn
# ---------------------------------------------------------------------------


def test_segment_to_tree_extracts_and_grounds_with_injected_segment_fn(tmp_path: Path) -> None:
    path = tmp_path / "v1.rtf"
    _write_rtf(path, _BODY)

    result = segment_to_tree(
        path,
        taxonomy_ids=["indemnification"],
        segment_fn=_two_block_segment_fn,
    )

    assert isinstance(result, GroundingResult)
    assert len(result.tree.nodes) == 1
    assert result.tree.nodes[0].heading == "1. Indemnification"
    assert result.taxonomy_by_path == {"1": "indemnification"}


def test_segment_to_tree_returned_tree_uses_run_gates_defaults(tmp_path: Path) -> None:
    """segment_to_tree does not know the caller's real document_id/version —
    the returned tree carries run_gates' placeholder identity ("doc"/"v1"/"");
    callers that need the real identity (e.g. the pipeline) must set it
    themselves on the returned GroundingResult.tree.
    """
    path = tmp_path / "v1.rtf"
    _write_rtf(path, _BODY)

    result = segment_to_tree(
        path,
        taxonomy_ids=["indemnification"],
        segment_fn=_two_block_segment_fn,
    )

    assert result.tree.document_id == "doc"
    assert result.tree.version == "v1"
    assert result.tree.source_file == ""


# ---------------------------------------------------------------------------
# QA-gate failure propagates — no deterministic-segmenter fallback
# ---------------------------------------------------------------------------


def test_segment_to_tree_propagates_segmentation_qa_error(tmp_path: Path) -> None:
    path = tmp_path / "v1.rtf"
    _write_rtf(path, _BODY)

    with pytest.raises(SegmentationQAError):
        segment_to_tree(
            path,
            taxonomy_ids=["indemnification"],
            segment_fn=_gate_failing_segment_fn,
        )


# ---------------------------------------------------------------------------
# Extraction failure propagates unchanged
# ---------------------------------------------------------------------------


def test_segment_to_tree_missing_file_raises_extraction_error(tmp_path: Path) -> None:
    from playbook_engine.extraction import ExtractionError

    missing = tmp_path / "does-not-exist.rtf"
    with pytest.raises(ExtractionError):
        segment_to_tree(
            missing,
            taxonomy_ids=["indemnification"],
            segment_fn=_two_block_segment_fn,
        )


# ---------------------------------------------------------------------------
# Default segment_fn binding — delegates to llm_segmenter.segment_document
# with taxonomy_ids fixed, client left as None (lazy real-client construction
# happens inside segment_document itself, not here).
# ---------------------------------------------------------------------------


def test_default_segment_fn_binds_taxonomy_ids_and_delegates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When no segment_fn is injected, segment_to_tree's default binding
    calls llm_segmenter.segment_document with (canonical_text, blocks,
    taxonomy_ids) and client=None — never constructing a real client itself.
    """
    path = tmp_path / "v1.rtf"
    _write_rtf(path, _BODY)

    calls: list[tuple[str, list[Block], list[str]]] = []

    def _fake_segment_document(
        canonical_text: str, blocks: list[Block], taxonomy_ids: list[str], **kwargs: object
    ) -> list[SegNode]:
        calls.append((canonical_text, blocks, taxonomy_ids))
        assert kwargs.get("client") is None or "client" not in kwargs
        return _two_block_segment_fn(canonical_text, blocks)

    monkeypatch.setattr("playbook_engine.llm_segmenter.segment_document", _fake_segment_document)

    result = segment_to_tree(path, taxonomy_ids=["indemnification"])  # no segment_fn injected

    assert len(calls) == 1
    called_text, called_blocks, called_taxonomy_ids = calls[0]
    assert called_taxonomy_ids == ["indemnification"]
    assert len(called_blocks) == 2
    assert result.taxonomy_by_path == {"1": "indemnification"}
