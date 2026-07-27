"""Tests for the agent-as-segmenter store-backed loop (issue #191).

SECURITY NOTE: all fixtures are synthetic; no real corpus content is used.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from playbook_engine.agent_judge import PendingQueue
from playbook_engine.agent_segmenter import (
    AGENT_SEGMENTER_MODEL,
    AgentSegmentationPending,
    StoreBackedSegmentFn,
    block_to_dict,
    segment_payload_key,
)
from playbook_engine.cli import _llm_segmentation_kwargs, cli
from playbook_engine.config import load_config
from playbook_engine.llm_segmenter_batch import SegmentationVerdictCache
from playbook_engine.segmentation_grounding import Block
from playbook_engine.taxonomy import load_taxonomy

# ---------------------------------------------------------------------------
# Config: `segmentation.agent` implies llm+cache and forces the sentinel model
# ---------------------------------------------------------------------------


def _write_config(tmp_path: Path, *, agent: bool) -> Path:
    tax = tmp_path / "tax.yaml"
    tax.write_text(
        "source: x\nentries:\n  - id: term\n    label: Term\n    status: active\n",
        encoding="utf-8",
    )
    cfg = tmp_path / "playbook.config.yaml"
    cfg.write_text(
        yaml.dump(
            {
                "agreement_type": {"id": "a", "name": "A"},
                "taxonomy": "tax.yaml",
                "baseline": {"template": None},
                "segmentation": {"agent": agent},
            }
        ),
        encoding="utf-8",
    )
    return cfg


def test_agent_config_forces_sentinel_and_implies_llm_cache(tmp_path: Path) -> None:
    cfg = load_config(_write_config(tmp_path, agent=True))
    assert cfg.segmentation.agent is True
    assert cfg.segmentation.llm is True  # agent implies llm-first
    assert cfg.segmentation.cache is True  # agent implies cache
    assert cfg.segmentation.model == AGENT_SEGMENTER_MODEL  # sentinel forced


def test_non_agent_config_unchanged(tmp_path: Path) -> None:
    cfg = load_config(_write_config(tmp_path, agent=False))
    assert cfg.segmentation.agent is False
    assert cfg.segmentation.llm is False
    assert cfg.segmentation.model != AGENT_SEGMENTER_MODEL


# ---------------------------------------------------------------------------
# StoreBackedSegmentFn — queues on call, raises so mine quarantines
# ---------------------------------------------------------------------------


def _blocks() -> list[Block]:
    return [
        Block(block_id="b0", page=1, char_span=(0, 5), text="Hello"),
        Block(block_id="b1", page=1, char_span=(5, 11), text=" world"),
    ]


def test_store_backed_segment_fn_queues_and_raises(tmp_path: Path) -> None:
    pending_path = tmp_path / "segment" / "pending.jsonl"
    fn = StoreBackedSegmentFn(pending=PendingQueue(pending_path))

    with pytest.raises(AgentSegmentationPending):
        fn("Hello world", _blocks())

    lines = pending_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["kind"] == "segment"
    assert rec["payload"]["canonical_text"] == "Hello world"
    assert [b["block_id"] for b in rec["payload"]["blocks"]] == ["b0", "b1"]


def test_store_backed_segment_fn_dedups_within_instance(tmp_path: Path) -> None:
    pending_path = tmp_path / "segment" / "pending.jsonl"
    fn = StoreBackedSegmentFn(pending=PendingQueue(pending_path))

    for _ in range(3):  # e.g. segment_verify_repair retries
        with pytest.raises(AgentSegmentationPending):
            fn("Hello world", _blocks())

    lines = [line for line in pending_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 1  # deduped by content hash


def test_store_backed_segment_fn_includes_taxonomy_ids_in_payload(tmp_path: Path) -> None:
    """Regression for issue #40: the queued payload must carry the corpus's
    real taxonomy_ids (matching the shape `playbook segment` writes), or
    segment_apply_cmd's taxonomy gate rejects every verdict that assigns a
    real taxonomy_id."""
    pending_path = tmp_path / "segment" / "pending.jsonl"
    fn = StoreBackedSegmentFn(
        pending=PendingQueue(pending_path),
        taxonomy_ids=["term"],
        document_id="fixture-doc",
        version="v1",
    )

    with pytest.raises(AgentSegmentationPending):
        fn("Hello world", _blocks())

    lines = pending_path.read_text(encoding="utf-8").splitlines()
    rec = json.loads(lines[0])
    assert rec["payload"]["taxonomy_ids"] == ["term"]
    assert rec["payload"]["document_id"] == "fixture-doc"
    assert rec["payload"]["version"] == "v1"


def test_llm_segmentation_kwargs_binds_real_taxonomy_ids_for_agent_mode(
    tmp_path: Path,
) -> None:
    """Regression for issue #40: this exercises the actual production wiring
    in ``cli._llm_segmentation_kwargs`` (not a hand-built ``StoreBackedSegmentFn``)
    to prove the ``mine``-first flow binds the corpus's real taxonomy_ids —
    not the empty default — into the queued payload's allow-list. Must fail
    if cli.py's ``taxonomy_ids=[e.id for e in taxonomy.classifier_entries()]``
    is ever dropped back to ``StoreBackedSegmentFn(pending=...)`` alone."""
    config_path = _write_config(tmp_path, agent=True)
    cfg = load_config(config_path)
    taxonomy = load_taxonomy(tmp_path / "tax.yaml")

    kwargs = _llm_segmentation_kwargs(cfg, taxonomy, tmp_path / "out", lambda _msg: None)

    expected = [e.id for e in taxonomy.classifier_entries()]
    assert expected  # sanity: the fixture taxonomy must actually have entries
    assert kwargs["llm_segment_fn"].taxonomy_ids == expected


def test_mine_queued_verdict_with_valid_taxonomy_id_passes_segment_apply(
    tmp_path: Path,
) -> None:
    """Regression for issue #40: a document queued via StoreBackedSegmentFn
    (the `mine`-first flow) with a real taxonomy_ids allow-list must let
    segment-apply accept a verdict that assigns a valid (non-null)
    taxonomy_id — this used to exit 1 with "taxonomy gate" because the
    mine-queued payload carried no taxonomy_ids at all (empty allow-list)."""
    out_dir = tmp_path / "out"
    seg_dir = out_dir / "segment"
    canonical = "Hello world"

    # Queue exactly as `mine` does on a cache miss: via StoreBackedSegmentFn,
    # bound to the corpus's real taxonomy_ids (as cli.py's
    # _llm_segmentation_kwargs now does).
    fn = StoreBackedSegmentFn(
        pending=PendingQueue(seg_dir / "pending.jsonl"),
        taxonomy_ids=["term"],
        document_id="fixture-doc",
        version="v1",
    )
    with pytest.raises(AgentSegmentationPending):
        fn(canonical, _blocks())

    verdicts = tmp_path / "seg-verdicts.jsonl"
    verdicts.write_text(
        json.dumps(
            {
                "canonical_text": canonical,
                "nodes": [
                    {
                        "node_id": "n1",
                        "parent_id": None,
                        "order": 1,
                        "heading": "One",
                        "taxonomy_id": "term",  # a real, valid taxonomy_id
                        "start_block_id": "b0",
                        "end_block_id": "b1",
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(cli, ["segment-apply", str(out_dir), "--verdicts", str(verdicts)])
    assert result.exit_code == 0, result.output

    cache = SegmentationVerdictCache(seg_dir / "cache.jsonl")
    nodes = cache.get(canonical, model=AGENT_SEGMENTER_MODEL)
    assert nodes is not None
    assert nodes[0].taxonomy_id == "term"


def test_block_to_dict_shape() -> None:
    d = block_to_dict(Block(block_id="b7", page=2, char_span=(3, 9), text="clause"))
    assert d == {"block_id": "b7", "page": 2, "char_span": [3, 9], "text": "clause"}


def test_segment_payload_key_is_content_stable() -> None:
    assert segment_payload_key("same text") == segment_payload_key("same text")
    assert segment_payload_key("a") != segment_payload_key("b")


# ---------------------------------------------------------------------------
# segment-apply → SegmentationVerdictCache round-trip (the mine replay key)
# ---------------------------------------------------------------------------


def test_segment_apply_populates_cache_for_mine(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    (out_dir / "segment").mkdir(parents=True)
    canonical = "Clause one.\nClause two."
    verdicts = tmp_path / "seg-verdicts.jsonl"
    verdicts.write_text(
        json.dumps(
            {
                "canonical_text": canonical,
                "nodes": [
                    {
                        "node_id": "n1",
                        "parent_id": None,
                        "order": 1,
                        "heading": "One",
                        "taxonomy_id": "term",
                        "start_block_id": "b0",
                        "end_block_id": "b0",
                        "start_quote": "",
                        "end_quote": "",
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(cli, ["segment-apply", str(out_dir), "--verdicts", str(verdicts)])
    assert result.exit_code == 0, result.output

    # mine reads the cache with the same sentinel model — the entry must hit.
    cache = SegmentationVerdictCache(out_dir / "segment" / "cache.jsonl")
    nodes = cache.get(canonical, model=AGENT_SEGMENTER_MODEL)
    assert nodes is not None
    assert len(nodes) == 1
    assert nodes[0].taxonomy_id == "term"
    assert nodes[0].start_block_id == "b0"


def test_segment_apply_rejects_malformed_line(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"nodes": []}\n', encoding="utf-8")  # missing canonical_text
    result = CliRunner().invoke(cli, ["segment-apply", str(out_dir), "--verdicts", str(bad)])
    assert result.exit_code == 1
    assert "canonical_text" in result.output


# ---------------------------------------------------------------------------
# segment-apply QA gating (pre-derivation QA): a bad partition must be
# rejected at apply time, never cached (a cached-bad partition wedges the
# document — `segment` reports it cached while every mine round quarantines it)
# ---------------------------------------------------------------------------


def _write_seg_pending(out_dir: Path, canonical: str, n_blocks: int) -> None:
    (out_dir / "segment").mkdir(parents=True, exist_ok=True)
    # Blocks partition the canonical text in order (newline-joined lines).
    lines = canonical.split("\n")
    assert len(lines) == n_blocks
    blocks = []
    pos = 0
    for i, line in enumerate(lines):
        start = canonical.index(line, pos)
        end = start + len(line)
        blocks.append({"block_id": f"b{i}", "page": 0, "char_span": [start, end], "text": line})
        pos = end
    item = {
        "key": segment_payload_key(canonical),
        "kind": "segment",
        "payload": {
            "document_id": "fixture-doc",
            "version": "v1",
            "taxonomy_ids": ["term"],
            "canonical_text": canonical,
            "blocks": blocks,
        },
    }
    (out_dir / "segment" / "pending.jsonl").write_text(json.dumps(item) + "\n", encoding="utf-8")


def test_segment_apply_rejects_gate_failing_partition(tmp_path: Path) -> None:
    """A partition that skips a block fails the coverage gate at apply time."""
    out_dir = tmp_path / "out"
    canonical = "Clause one.\nClause two."
    _write_seg_pending(out_dir, canonical, 2)
    verdicts = tmp_path / "seg-verdicts.jsonl"
    verdicts.write_text(
        json.dumps(
            {
                "canonical_text": canonical,
                "nodes": [
                    {
                        "node_id": "n1",
                        "parent_id": None,
                        "order": 1,
                        "heading": "One",
                        "taxonomy_id": "term",
                        "start_block_id": "b0",
                        "end_block_id": "b0",  # b1 never covered
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    result = CliRunner().invoke(cli, ["segment-apply", str(out_dir), "--verdicts", str(verdicts)])
    assert result.exit_code != 0
    assert "QA gate" in result.output or "fails QA" in result.output
    # Nothing cached: the document must stay pending, not wedge.
    cache = SegmentationVerdictCache(out_dir / "segment" / "cache.jsonl")
    assert cache.get(canonical, model=AGENT_SEGMENTER_MODEL) is None


def test_segment_apply_gates_pass_for_full_partition(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    canonical = "Clause one.\nClause two."
    _write_seg_pending(out_dir, canonical, 2)
    verdicts = tmp_path / "seg-verdicts.jsonl"
    verdicts.write_text(
        json.dumps(
            {
                "canonical_text": canonical,
                "nodes": [
                    {
                        "node_id": "n1",
                        "parent_id": None,
                        "order": 1,
                        "heading": "One",
                        "taxonomy_id": "term",
                        "start_block_id": "b0",
                        "end_block_id": "b1",
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    result = CliRunner().invoke(cli, ["segment-apply", str(out_dir), "--verdicts", str(verdicts)])
    assert result.exit_code == 0, result.output
    cache = SegmentationVerdictCache(out_dir / "segment" / "cache.jsonl")
    assert cache.get(canonical, model=AGENT_SEGMENTER_MODEL) is not None
