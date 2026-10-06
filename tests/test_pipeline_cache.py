"""Layered stage cache under store-backed judges (issue #219).

Before #219 a verdict store at ``out/judge/verdicts.jsonl`` forced
``no_cache=True``: every ``judge`` round and every post-judge ``mine``
re-ingested, re-ordered, re-aligned and re-diffed every document. The cache
key was one monolithic per-document "l1-l4" blob that hashed absolute file
paths and folded every global setting, and a warm run rewrote every
intermediate. These tests pin the replacement:

  - the cache stays on under store-backed judges and a warm run hits every
    document — without hiding a pending item or a changed verdict;
  - L1 (per version) and L2-L4 (per document) are cached apart, so a
    taxonomy edit replays L1 and recomputes only classification onward, and
    an L1-only change that reproduces the same trees replays L2-L4;
  - moving the corpus directory invalidates nothing;
  - an unchanged intermediate is not rewritten (``--force-rewrite`` does).

All corpora are the committed synthetic examples (examples/nda — fictional
parties) or programmatic RTF built here. No network, no API key: the NDA runs
use config.smoke.yaml (deterministic segmentation), exactly like
tests/test_nda_smoke.py.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml
from click.testing import CliRunner
from click.testing import Result as CliResult

from playbook_engine import pipeline as pipeline_module
from playbook_engine.cli import cli
from playbook_engine.config import load_config
from playbook_engine.llm_segmenter_batch import SegmentationVerdictCache
from playbook_engine.observation_builder import read_observations_jsonl
from playbook_engine.pipeline import mine_corpus
from playbook_engine.segmentation_grounding import Block, SegNode
from playbook_engine.taxonomy import load_taxonomy

_REPO_ROOT = Path(__file__).resolve().parent.parent
_NDA_DIR = _REPO_ROOT / "examples" / "nda"
_NDA_CORPUS = _NDA_DIR / "corpus"
_NDA_SMOKE_CONFIG = _NDA_DIR / "config.smoke.yaml"
_NDA_VERDICTS = _NDA_DIR / "canned-verdicts.jsonl"
_NDA_TAXONOMY = _REPO_ROOT / "spec" / "taxonomy" / "nda.yaml"
_AFFILIATION_TAXONOMY = _REPO_ROOT / "spec" / "taxonomy" / "affiliation-agreement.yaml"

# 6 deals, 17 version files (examples/nda/corpus).
_NDA_DOCS = 6
_NDA_VERSIONS = 17

_STAGE_LINE = re.compile(
    r"stage cache: L1 \(per version\) hits (\d+), misses (\d+); "
    r"L2-L4 \(per document\) hits (\d+), misses (\d+)"
)


def _invoke(args: list[str]) -> CliResult:
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == 0, f"{args[0]} failed:\n{result.output}\n{result.exception!r}"
    return result


def _stage_counts(output: str) -> tuple[int, int, int, int]:
    """``(l1_hits, l1_misses, doc_hits, doc_misses)`` from a run's progress output."""
    match = _STAGE_LINE.search(output)
    assert match is not None, f"no stage-cache line in output:\n{output}"
    l1_hits, l1_misses, doc_hits, doc_misses = (int(g) for g in match.groups())
    return l1_hits, l1_misses, doc_hits, doc_misses


def _mine_cli(out_dir: Path, *extra: str, corpus: Path = _NDA_CORPUS) -> str:
    return _invoke(
        ["mine", str(corpus), "--config", str(_NDA_SMOKE_CONFIG), "--out", str(out_dir), *extra]
    ).output


def _judge_apply(out_dir: Path, verdicts: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    _invoke(["judge-apply", str(out_dir), "--verdicts", str(verdicts)])


def _canned_lines() -> list[str]:
    return [line for line in _NDA_VERDICTS.read_text(encoding="utf-8").splitlines() if line]


def _rubric_note(output: str) -> str | None:
    """The legacy-rubric NOTE line (a tally of stored verdicts replayed), if any."""
    for line in output.splitlines():
        if "carry no rubric version" in line:
            return line
    return None


# ---------------------------------------------------------------------------
# Store-backed judges keep the cache on
# ---------------------------------------------------------------------------


def test_warm_run_with_populated_verdict_store_hits_every_document(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ticket's headline: with every verdict banked, a second ``mine`` over
    the same out-dir replays every document (and every version's L1) instead
    of re-mining the corpus — and produces byte-identical artifacts, with the
    same rubric tally (a replayed document re-reports the stored verdicts it
    was built from, or the legacy NOTE would undercount on warm runs)."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    out_dir = tmp_path / "out"
    _judge_apply(out_dir, _NDA_VERDICTS)

    cold = _mine_cli(out_dir)
    assert "store-backed judges active" in cold
    assert _stage_counts(cold) == (0, _NDA_VERSIONS, 0, _NDA_DOCS)
    cold_obs = (out_dir / "observations.jsonl").read_text(encoding="utf-8")
    cold_trails = {p.name: p.read_text() for p in (out_dir / "trail").glob("*.json")}

    warm = _mine_cli(out_dir)
    assert _stage_counts(warm) == (_NDA_VERSIONS, 0, _NDA_DOCS, 0)
    assert f"cache hits={_NDA_DOCS}, misses=0" in warm
    assert (out_dir / "observations.jsonl").read_text(encoding="utf-8") == cold_obs
    assert {p.name: p.read_text() for p in (out_dir / "trail").glob("*.json")} == cold_trails

    assert _rubric_note(cold) is not None, "canned verdicts are unstamped (legacy)"
    assert _rubric_note(warm) == _rubric_note(cold)


def test_documents_with_pending_verdicts_are_never_replayed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A document whose compute queued anything carries needs_review sentinels
    and must recompute every round — or ``judge`` (which resets pending.jsonl
    each round) would replay it and silently drop its items from the queue,
    and the drain loop would "converge" with questions never asked."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    out_dir = tmp_path / "out"
    lines = _canned_lines()
    # Scope (6) + the first two classify verdicts: the remaining classify and
    # both provenance verdicts stay pending.
    partial = tmp_path / "partial.jsonl"
    partial.write_text("\n".join(lines[:8]) + "\n", encoding="utf-8")
    rest = tmp_path / "rest.jsonl"
    rest.write_text("\n".join(lines[8:]) + "\n", encoding="utf-8")
    _judge_apply(out_dir, partial)

    def _judge_round() -> tuple[str, str]:
        output = _invoke(
            ["judge", str(_NDA_CORPUS), "--config", str(_NDA_SMOKE_CONFIG), "--out", str(out_dir)]
        ).output
        return output, (out_dir / "judge" / "pending.jsonl").read_text(encoding="utf-8")

    round_1, pending_1 = _judge_round()
    round_2, pending_2 = _judge_round()
    assert pending_1.strip(), "the partial store must leave items pending"
    assert pending_2 == pending_1, "a warm round must queue exactly what the cold one did"

    _, _, doc_hits_2, doc_misses_2 = _stage_counts(round_2)
    assert doc_misses_2 > 0, "documents with pending verdicts must recompute"
    assert doc_hits_2 + doc_misses_2 == _NDA_DOCS
    # Every version's L1 replays regardless — no verdict feeds L1.
    assert _stage_counts(round_2)[:2] == (_NDA_VERSIONS, 0)

    # Banking the rest resolves everything: the next round recomputes the
    # previously-pending documents and queues nothing; the round after that
    # replays every document.
    _judge_apply(out_dir, rest)
    round_3, pending_3 = _judge_round_or_empty(out_dir)
    assert _stage_counts(round_3)[3] == doc_misses_2
    assert pending_3.strip() == ""
    round_4, _ = _judge_round_or_empty(out_dir)
    assert _stage_counts(round_4)[2:] == (_NDA_DOCS, 0)

    # And the replayed store equals a cold mine over the full store.
    fresh = tmp_path / "fresh"
    _judge_apply(fresh, _NDA_VERDICTS)
    _mine_cli(fresh, "--no-cache")
    assert (out_dir / "observations.jsonl").read_text() == (
        fresh / "observations.jsonl"
    ).read_text()


def _judge_round_or_empty(out_dir: Path) -> tuple[str, str]:
    output = _invoke(
        ["judge", str(_NDA_CORPUS), "--config", str(_NDA_SMOKE_CONFIG), "--out", str(out_dir)]
    ).output
    pending = out_dir / "judge" / "pending.jsonl"
    return output, pending.read_text(encoding="utf-8") if pending.exists() else ""


def test_changed_stored_verdict_recomputes_only_the_documents_that_read_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A verdict overwritten by ``judge-apply`` must not be shadowed by the
    stage cache: the document(s) whose cached result replayed it recompute,
    every other document still replays. A scope verdict is per document (its
    payload carries the document_id), so exactly one deal depends on it."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    out_dir = tmp_path / "out"
    _judge_apply(out_dir, _NDA_VERDICTS)
    _mine_cli(out_dir)

    changed = json.loads(_canned_lines()[0])
    assert "in_scope" in changed["verdict"]
    changed["verdict"]["scope_confidence"] = 0.61
    changed_path = tmp_path / "changed.jsonl"
    changed_path.write_text(json.dumps(changed) + "\n", encoding="utf-8")
    _judge_apply(out_dir, changed_path)

    warm = _mine_cli(out_dir)
    assert _stage_counts(warm) == (_NDA_VERSIONS, 0, _NDA_DOCS - 1, 1)
    scope = json.loads((out_dir / "scope.json").read_text())
    assert 0.61 in {d["scope_confidence"] for d in scope["documents"]}

    # Same artifacts as a cold mine over the same final store.
    fresh = tmp_path / "fresh"
    fresh_verdicts = tmp_path / "fresh.jsonl"
    fresh_verdicts.write_text(
        _NDA_VERDICTS.read_text(encoding="utf-8") + changed_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    _judge_apply(fresh, fresh_verdicts)
    _mine_cli(fresh, "--no-cache")
    for name in ("observations.jsonl", "scope.json", "corpus_manifest.json"):
        assert (out_dir / name).read_text() == (fresh / name).read_text(), name


def test_plan_only_reads_through_the_real_stage_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``judge --plan-only`` used to run a full L1-L4 pass into a temp dir.
    It now reads the real out-dir's stage cache: after a fully-judged mine it
    replays every document and still reports the exact (empty) plan."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    out_dir = tmp_path / "out"
    _judge_apply(out_dir, _NDA_VERDICTS)
    _mine_cli(out_dir)

    plan = _invoke(
        [
            "judge",
            str(_NDA_CORPUS),
            "--config",
            str(_NDA_SMOKE_CONFIG),
            "--out",
            str(out_dir),
            "--plan-only",
        ]
    ).output
    assert _stage_counts(plan) == (_NDA_VERSIONS, 0, _NDA_DOCS, 0)
    assert "0 pending items" in plan


# ---------------------------------------------------------------------------
# Split layers: a setting change invalidates only the layer it affects
# ---------------------------------------------------------------------------


def _mine_api(
    out_dir: Path,
    *,
    taxonomy_path: Path = _NDA_TAXONOMY,
    corpus: Path = _NDA_CORPUS,
    no_cache: bool = False,
    **kwargs: Any,
) -> list[str]:
    lines: list[str] = []
    mine_corpus(
        corpus_dir=corpus,
        config=load_config(_NDA_SMOKE_CONFIG),
        taxonomy=load_taxonomy(taxonomy_path),
        out_dir=out_dir,
        no_cache=no_cache,
        progress=lines.append,
        **kwargs,
    )
    return lines


def _flip_status(tmp_path: Path, entry_id: str, status: str) -> Path:
    raw = yaml.safe_load(_NDA_TAXONOMY.read_text(encoding="utf-8"))
    flipped = False
    for entry in raw["entries"]:
        if entry["id"] == entry_id:
            assert entry["status"] != status
            entry["status"] = status
            flipped = True
    assert flipped
    path = tmp_path / "nda-flipped.yaml"
    path.write_text(yaml.dump(raw), encoding="utf-8")
    return path


def test_taxonomy_status_flip_invalidates_classification_only(tmp_path: Path) -> None:
    """Retiring an entry (status: inactive) changes what a clause may be
    classified into without touching its id/label/description — the old key
    ignored status entirely and replayed the stale classification. It must
    recompute L2-L4 for every document while replaying every L1 tree (the
    deterministic segmenter never reads the taxonomy).

    The retired entry is one the TEMPLATE never classifies into (a
    counterparty-introduced limitation of liability), so the template-
    standards part of the key cannot be what notices the change."""
    out_dir = tmp_path / "out"
    _mine_api(out_dir)
    before = read_observations_jsonl(out_dir / "observations.jsonl")
    template = read_observations_jsonl(out_dir / "template_observations.jsonl")
    assert any(o["taxonomy_id"] == "limitation_of_liability" for o in before)
    assert not any(o["taxonomy_id"] == "limitation_of_liability" for o in template)

    flipped = _flip_status(tmp_path, "limitation_of_liability", "inactive")
    lines = _mine_api(out_dir, taxonomy_path=flipped)
    assert _stage_counts("\n".join(lines)) == (_NDA_VERSIONS, 0, 0, _NDA_DOCS)
    after = read_observations_jsonl(out_dir / "observations.jsonl")
    assert not any(o["taxonomy_id"] == "limitation_of_liability" for o in after)

    # Warm-with-flip equals cold-with-flip.
    cold = tmp_path / "cold"
    _mine_api(cold, taxonomy_path=flipped, no_cache=True)
    assert (out_dir / "observations.jsonl").read_text() == (cold / "observations.jsonl").read_text()


def test_l1_only_change_replays_l2_l4_when_trees_are_identical(tmp_path: Path) -> None:
    """A segmenter prompt bump is an L1 input: every version's L1 recomputes.
    On the deterministic path the trees come out identical, so L2-L4 — keyed
    by the L1 records, not by the setting — replays every document."""
    out_dir = tmp_path / "out"
    _mine_api(out_dir)
    obs = (out_dir / "observations.jsonl").read_text()

    original = pipeline_module.PROMPT_VERSION
    try:
        pipeline_module.PROMPT_VERSION = "v-bumped-219"  # type: ignore[misc]
        lines = _mine_api(out_dir)
    finally:
        pipeline_module.PROMPT_VERSION = original  # type: ignore[misc]
    assert _stage_counts("\n".join(lines)) == (0, _NDA_VERSIONS, _NDA_DOCS, 0)
    assert (out_dir / "observations.jsonl").read_text() == obs


def test_moving_the_corpus_directory_does_not_invalidate(tmp_path: Path) -> None:
    """Keys hash each file's name, not its absolute path: the same corpus at a
    different location (another checkout, Docker vs host) is all hits."""
    first = tmp_path / "a" / "corpus"
    shutil.copytree(_NDA_CORPUS, first)
    out_dir = tmp_path / "out"
    _mine_api(out_dir, corpus=first)
    obs = (out_dir / "observations.jsonl").read_text()

    moved = tmp_path / "b" / "elsewhere" / "corpus"
    moved.parent.mkdir(parents=True)
    shutil.move(str(first), str(moved))
    lines = _mine_api(out_dir, corpus=moved)
    assert _stage_counts("\n".join(lines)) == (_NDA_VERSIONS, 0, _NDA_DOCS, 0)
    assert (out_dir / "observations.jsonl").read_text() == obs


# ---------------------------------------------------------------------------
# L1 replays only while the segmentation store still agrees
# ---------------------------------------------------------------------------

_RTF_PROLOGUE = r"{\rtf1\ansi\deff0{\fonttbl{\f0\froman\fcharset0 Times New Roman;}}\f0\fs24 "
_SEG_BODY = (
    r"1. Indemnification\par "
    r"Gamma Ltd shall indemnify Delta Inc against third-party claims "
    r"arising from the placement programme.\par "
    r"2. Governing Law\par "
    r"This agreement is governed by the laws of the State of California.\par "
)


def test_resegmented_version_is_not_shadowed_by_the_l1_cache(tmp_path: Path) -> None:
    """``segment-apply`` can replace a version's banked segmentation for the
    same canonical text (SegmentationVerdictCache.put — the call segment_apply
    makes). The L1 record grounded from the old entry must not replay: L1
    recomputes for that version, and the new classification reaches the
    observations."""
    corpus = tmp_path / "corpus"
    (corpus / "deal-a").mkdir(parents=True)
    (corpus / "deal-a" / "v1.rtf").write_text(_RTF_PROLOGUE + _SEG_BODY + "}", encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.dump(
            {
                "agreement_type": {
                    "id": "educational-affiliation",
                    "name": "Educational Affiliation Agreement",
                },
                "baseline": {},
                "taxonomy": str(_AFFILIATION_TAXONOMY),
                "provenance": {"our_party_aliases": ["Gamma Ltd"]},
            }
        ),
        encoding="utf-8",
    )
    cfg = load_config(config_path)
    taxonomy = load_taxonomy(_AFFILIATION_TAXONOMY)
    seg_cache = SegmentationVerdictCache(tmp_path / "seg.jsonl")
    seen: dict[str, list[Block]] = {}

    def _segment(canonical_text: str, blocks: list[Block]) -> list[SegNode]:
        seen[canonical_text] = blocks
        return _pair_nodes(blocks, {"1. Indemnification": "indemnification"})

    def _run() -> tuple[list[str], list[dict[str, Any]]]:
        lines: list[str] = []
        mine_corpus(
            corpus_dir=corpus,
            config=cfg,
            taxonomy=taxonomy,
            out_dir=tmp_path / "out",
            use_llm_segmentation=True,
            llm_segment_fn=_segment,
            segmentation_cache=seg_cache,
            progress=lines.append,
        )
        return lines, read_observations_jsonl(tmp_path / "out" / "observations.jsonl")

    _run()
    lines, obs = _run()
    assert _stage_counts("\n".join(lines)) == (1, 0, 1, 0)
    assert "governing_law" not in {o["taxonomy_id"] for o in obs}

    # Re-segment the same text: now clause 2 is classified too.
    ((canonical_text, blocks),) = seen.items()
    seg_cache.put(
        canonical_text,
        _pair_nodes(
            blocks,
            {"1. Indemnification": "indemnification", "2. Governing Law": "governing_law"},
        ),
        model=cfg.segmentation.model,
    )
    lines, obs = _run()
    l1_hits, l1_misses, _, _ = _stage_counts("\n".join(lines))
    assert (l1_hits, l1_misses) == (0, 1)
    assert "governing_law" in {o["taxonomy_id"] for o in obs}


def _pair_nodes(blocks: list[Block], taxonomy_by_heading: dict[str, str]) -> list[SegNode]:
    """One clause node per (heading, body) block pair — the RTF fixture's shape."""
    nodes: list[SegNode] = []
    for order, i in enumerate(range(0, len(blocks), 2), start=1):
        heading, body = blocks[i], blocks[min(i + 1, len(blocks) - 1)]
        nodes.append(
            SegNode(
                node_id=f"n{order}",
                parent_id=None,
                order=order,
                heading=heading.text,
                taxonomy_id=taxonomy_by_heading.get(heading.text),
                start_block_id=heading.block_id,
                end_block_id=body.block_id,
            )
        )
    return nodes


# ---------------------------------------------------------------------------
# Intermediates are written only when their content changes
# ---------------------------------------------------------------------------

_OLD_NS = 1_000_000_000_000_000_000  # 2001-09-09, well before any real write


def _age(paths: list[Path]) -> None:
    for p in paths:
        os.utime(p, ns=(_OLD_NS, _OLD_NS))


def test_unchanged_intermediates_are_not_rewritten(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    _mine_cli(out_dir)
    tracked = [
        out_dir / "observations.jsonl",
        out_dir / "round_moves.jsonl",
        out_dir / "corpus_manifest.json",
        out_dir / "scope.json",
        out_dir / "quarantine.json",
        out_dir / "template_observations.jsonl",
        *sorted((out_dir / "trail").glob("*.json")),
        *sorted((out_dir / "normalized").rglob("*.clauses.json")),
    ]
    assert len(tracked) > 6 + _NDA_DOCS
    _age(tracked)

    _mine_cli(out_dir)
    rewritten = [p for p in tracked if p.stat().st_mtime_ns != _OLD_NS]
    assert rewritten == [], f"unchanged intermediates were rewritten: {rewritten}"

    _mine_cli(out_dir, "--force-rewrite")
    assert all(p.stat().st_mtime_ns != _OLD_NS for p in tracked)


def test_stale_trail_and_tree_files_are_still_cleared(tmp_path: Path) -> None:
    """Writing only what changed must not keep files this run did not
    produce: a removed deal's trail and trees still disappear (issue #51/#139
    contract), and its emptied normalized/ directory with them."""
    corpus = tmp_path / "corpus"
    shutil.copytree(_NDA_CORPUS, corpus)
    out_dir = tmp_path / "out"
    _mine_cli(out_dir, corpus=corpus)
    gone = sorted(p.name for p in corpus.iterdir() if p.is_dir())[0]
    assert (out_dir / "trail" / f"{gone}.json").exists()
    assert (out_dir / "normalized" / gone).is_dir()

    shutil.rmtree(corpus / gone)
    _mine_cli(out_dir, corpus=corpus)
    assert not (out_dir / "trail" / f"{gone}.json").exists()
    assert not (out_dir / "normalized" / gone).exists()
    assert len(list((out_dir / "trail").glob("*.json"))) == _NDA_DOCS - 1


def test_case_only_deal_rename_keeps_the_rewritten_trail_and_trees(tmp_path: Path) -> None:
    """A case-only rename of a deal folder (Delta-Ventures -> delta-ventures)
    must leave exactly the new-case trail and normalized/ tree files. On a
    case-insensitive filesystem (macOS APFS, the owner's platform) the old-case
    entry IS the new file: a stale-clear that ran after the writes and compared
    names would unlink what was just written, leaving trail/ and normalized/
    empty. The stale-clear runs before the writes, so the old-case names go
    and the writes recreate them under the new case — on any filesystem."""
    corpus = tmp_path / "corpus"
    shutil.copytree(_NDA_CORPUS, corpus)
    (corpus / "delta-ventures").rename(corpus / "Delta-Ventures")
    out_dir = tmp_path / "out"
    _mine_cli(out_dir, corpus=corpus)
    assert "Delta-Ventures.json" in os.listdir(out_dir / "trail")
    assert "Delta-Ventures" in os.listdir(out_dir / "normalized")
    old_trees = sorted(os.listdir(out_dir / "normalized" / "Delta-Ventures"))
    assert old_trees

    # Via an intermediate name, so the rename is case-only on any filesystem.
    (corpus / "Delta-Ventures").rename(corpus / "renaming")
    (corpus / "renaming").rename(corpus / "delta-ventures")
    _mine_cli(out_dir, corpus=corpus)

    trails = os.listdir(out_dir / "trail")
    assert "delta-ventures.json" in trails
    assert "Delta-Ventures.json" not in trails
    assert len([t for t in trails if t.endswith(".json")]) == _NDA_DOCS
    normalized = os.listdir(out_dir / "normalized")
    assert "delta-ventures" in normalized
    assert "Delta-Ventures" not in normalized
    assert sorted(os.listdir(out_dir / "normalized" / "delta-ventures")) == old_trees

    # And the result equals a cold run over the renamed corpus.
    cold = tmp_path / "cold"
    _mine_cli(cold, "--no-cache", corpus=corpus)
    assert sorted(os.listdir(cold / "trail")) == sorted(trails)
    assert (out_dir / "trail" / "delta-ventures.json").read_text() == (
        cold / "trail" / "delta-ventures.json"
    ).read_text()
    for name in old_trees:
        assert (out_dir / "normalized" / "delta-ventures" / name).read_text() == (
            cold / "normalized" / "delta-ventures" / name
        ).read_text()


# ---------------------------------------------------------------------------
# The L2-L4 key covers every value its result copies from outside L1
# ---------------------------------------------------------------------------


def test_bytes_only_change_refreshes_the_manifest_sha256(tmp_path: Path) -> None:
    """corpus_doc["version_files"][].sha256 (OPF §4 content address) is copied
    into the cached L2-L4 result from the source file, not from the L1 record.
    Trailing whitespace changes the bytes but not the extracted text, so the
    version's L1 record is identical — the L2-L4 key must still miss for that
    document, or the warm manifest keeps the old hash and no longer equals a
    cold run's."""
    corpus = tmp_path / "corpus"
    shutil.copytree(_NDA_CORPUS, corpus)
    out_dir = tmp_path / "out"
    _mine_api(out_dir, corpus=corpus)
    before = json.loads((out_dir / "corpus_manifest.json").read_text())

    changed = corpus / "beta-industries" / "v1.rtf"
    changed.write_bytes(changed.read_bytes() + b"\n\n")
    lines = _mine_api(out_dir, corpus=corpus)
    # L1 recomputes only the changed version; only its document misses L2-L4.
    assert _stage_counts("\n".join(lines)) == (_NDA_VERSIONS - 1, 1, _NDA_DOCS - 1, 1)

    cold = tmp_path / "cold"
    _mine_api(cold, corpus=corpus, no_cache=True)
    warm_manifest = (out_dir / "corpus_manifest.json").read_text()
    assert warm_manifest == (cold / "corpus_manifest.json").read_text()
    assert json.loads(warm_manifest) != before  # the new sha256 replaced the old one
