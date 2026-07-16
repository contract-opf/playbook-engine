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
from playbook_engine.cli import cli
from playbook_engine.config import load_config
from playbook_engine.llm_segmenter_batch import SegmentationVerdictCache
from playbook_engine.segmentation_grounding import Block

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
