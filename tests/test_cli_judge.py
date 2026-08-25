"""Tests for ``playbook judge`` and ``playbook judge-apply`` CLI commands — issue #66.

Verifies the full end-to-end fixture path:
  judge (records pending) → judge-apply (loads canned verdicts) → mine → project
  → validate (schema-valid, taxonomy_id populated for judged clauses).

Acceptance criteria covered:

  AC-1: judge --plan reports >0 pending items and exits 0 without writing
        observations.jsonl.
  AC-2: Full path (judge → judge-apply → mine → project) yields an
        observations.jsonl with taxonomy_id populated for the judged clauses
        (non-blank), and playbook validate exits 0.
  AC-3: Re-running judge after judge-apply does not re-queue already-judged
        items (idempotent — 0 pending after verdicts applied).

SECURITY NOTE: All corpus content is from the pre-committed synthetic
fixture under examples/judge-fixture/. No real legal text, no real parties.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml
from click.testing import CliRunner

from playbook_engine import extraction
from playbook_engine.cli import cli
from playbook_engine.observation_builder import read_observations_jsonl
from playbook_engine.validator import load_opf_file, validate_document

# ---------------------------------------------------------------------------
# Fixture paths
# ---------------------------------------------------------------------------

_FIXTURE_DIR = Path(__file__).parent.parent / "examples" / "judge-fixture"
_CORPUS_DIR = _FIXTURE_DIR / "corpus"
_CONFIG_PATH = _FIXTURE_DIR / "config.yaml"
_CANNED_VERDICTS = _FIXTURE_DIR / "canned-verdicts.jsonl"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _invoke(*args: str) -> tuple[int, str]:
    """Invoke the playbook CLI with *args*; return (exit_code, output)."""
    runner = CliRunner()
    result = runner.invoke(cli, list(args))
    return result.exit_code, result.output


# ---------------------------------------------------------------------------
# AC-1: judge --plan reports pending items without writing observations
# ---------------------------------------------------------------------------


def test_judge_plan_exits_zero(tmp_path: Path) -> None:
    """judge --plan exits 0."""
    code, _ = _invoke(
        "judge",
        str(_CORPUS_DIR),
        "--config",
        str(_CONFIG_PATH),
        "--out",
        str(tmp_path),
        "--plan",
    )
    assert code == 0


def test_judge_plan_reports_pending_items(tmp_path: Path) -> None:
    """AC-1: judge --plan reports >0 pending items."""
    _, output = _invoke(
        "judge",
        str(_CORPUS_DIR),
        "--config",
        str(_CONFIG_PATH),
        "--out",
        str(tmp_path),
        "--plan",
    )
    assert "Pending items:" in output
    # Extract the count — must be > 0.
    for line in output.splitlines():
        if line.strip().startswith("Pending items:"):
            count_str = line.strip().split(":")[1].strip().split()[0]
            count = int(count_str)
            assert count > 0, f"Expected >0 pending items; got {count}"
            break
    else:
        pytest.fail(f"'Pending items:' not found in output:\n{output}")


def test_judge_plan_does_not_write_observations(tmp_path: Path) -> None:
    """AC-1: judge --plan must NOT write observations.jsonl."""
    _invoke(
        "judge",
        str(_CORPUS_DIR),
        "--config",
        str(_CONFIG_PATH),
        "--out",
        str(tmp_path),
        "--plan",
    )
    assert not (tmp_path / "observations.jsonl").exists(), (
        "judge --plan must NOT write observations.jsonl"
    )


def test_judge_plan_pending_counts_by_kind(tmp_path: Path) -> None:
    """judge --plan breaks down pending counts by kind."""
    _, output = _invoke(
        "judge",
        str(_CORPUS_DIR),
        "--config",
        str(_CONFIG_PATH),
        "--out",
        str(tmp_path),
        "--plan",
    )
    # At least one kind must be reported.
    assert any(kind in output for kind in ("classify", "deviation", "provenance")), (
        f"No kind breakdown found in output:\n{output}"
    )


# ---------------------------------------------------------------------------
# Issue #76: --plan-only is the preferred alias for judge's dry-run flag
# (the bare --plan name collides with `stage --plan`, which EXECUTES a plan
# instead of previewing — same name, opposite risk profile on a sibling
# subcommand). Both spellings must keep working and behave identically.
# ---------------------------------------------------------------------------


def test_judge_plan_only_alias_behaves_identically_to_plan(tmp_path: Path) -> None:
    """judge --plan-only must report the same pending count as judge --plan."""
    code_plan, output_plan = _invoke(
        "judge",
        str(_CORPUS_DIR),
        "--config",
        str(_CONFIG_PATH),
        "--out",
        str(tmp_path / "via-plan"),
        "--plan",
    )
    code_plan_only, output_plan_only = _invoke(
        "judge",
        str(_CORPUS_DIR),
        "--config",
        str(_CONFIG_PATH),
        "--out",
        str(tmp_path / "via-plan-only"),
        "--plan-only",
    )
    assert code_plan == 0
    assert code_plan_only == 0
    assert not (tmp_path / "via-plan-only" / "observations.jsonl").exists()

    def _pending_count(output: str) -> int:
        for line in output.splitlines():
            if line.strip().startswith("Pending items:"):
                return int(line.strip().split(":")[1].strip().split()[0])
        pytest.fail(f"'Pending items:' not found in output:\n{output}")

    assert _pending_count(output_plan) == _pending_count(output_plan_only)


def test_judge_help_shows_plan_only_and_plan() -> None:
    """issue #76: --help must document both the preferred `--plan-only`
    spelling and the `--plan` alias."""
    code, output = _invoke("judge", "--help")
    assert code == 0
    assert "--plan-only, --plan" in output


# ---------------------------------------------------------------------------
# Issue #134: --plan token estimate must scale with real payload size, and
# must report a segmentation-cost line (previously the estimate was a flat
# `total_items * 200` guess that ignored payload size entirely and never
# mentioned LLM-segmentation spend at all).
# ---------------------------------------------------------------------------


def _make_payload_size_corpus(tmp_path: Path, name: str, filler_repeat: int) -> tuple[Path, Path]:
    """Single-document, single-version, deterministic-segmentation corpus.

    Two clauses (indemnification, governing law) — the same structure as
    ``examples/judge-fixture``'s corpus, so it produces the same *kinds* of
    pending items — but the indemnification clause's body is padded with
    ``filler_repeat`` copies of a filler sentence. Varying only this padding
    keeps the pending-item COUNT identical between two calls of this helper
    while making the payload (and thus the honest token estimate) much
    larger for a bigger ``filler_repeat``.
    """
    corpus_dir = tmp_path / name / "corpus"
    deal_dir = corpus_dir / "deal-one"
    deal_dir.mkdir(parents=True)

    filler = "This clause contains additional negotiated boilerplate text. " * filler_repeat
    body = (
        r"1. Indemnification\par "
        + f"FixtureCorp shall indemnify the Institution against third-party claims. {filler}"
        + r"\par "
        r"2. Governing Law\par "
        r"This agreement is governed by the laws of the State of Delaware.\par "
    )
    _write_rtf(deal_dir / "v1.rtf", body)

    taxonomy_path = (
        Path(__file__).parent.parent / "spec" / "taxonomy" / "affiliation-agreement.yaml"
    )
    cfg = {
        "agreement_type": {"id": "fixture-affiliation", "name": "Fixture Affiliation Agreement"},
        "baseline": {"template": None},
        "taxonomy": str(taxonomy_path),
        "provenance": {"our_party_aliases": ["FixtureCorp"]},
    }
    config_path = tmp_path / name / "playbook.config.yaml"
    config_path.write_text(yaml.dump(cfg), encoding="utf-8")
    return corpus_dir, config_path


def _parse_plan_output(output: str) -> tuple[int, int]:
    """Return ``(pending_count, token_estimate)`` parsed from ``judge --plan``
    output's ``"Pending items: N (token estimate: ~X)"`` line."""
    for line in output.splitlines():
        stripped = line.strip()
        if stripped.startswith("Pending items:"):
            after_label = stripped.split(":", 1)[1]
            count = int(after_label.strip().split()[0])
            token_str = stripped.rsplit("~", 1)[1].rstrip(")").replace(",", "")
            return count, int(token_str)
    pytest.fail(f"'Pending items:' not found in output:\n{output}")


def test_plan_estimate_scales_with_payload(tmp_path: Path) -> None:
    """Two pending sets of equal item COUNT but very different payload sizes
    must produce different token estimates (issue #134) — the old flat
    ``total_items * 200`` estimate produced the identical number for both,
    which is the bug this ticket closes. Also asserts the new segmentation
    cost line is present, even on the deterministic (non-LLM) segmentation
    path where it correctly reports zero spend rather than omitting
    segmentation from the plan output entirely."""
    small_corpus, small_config = _make_payload_size_corpus(tmp_path, "small", filler_repeat=1)
    big_corpus, big_config = _make_payload_size_corpus(tmp_path, "big", filler_repeat=200)

    code_small, output_small = _invoke(
        "judge",
        str(small_corpus),
        "--config",
        str(small_config),
        "--out",
        str(tmp_path / "small" / "out"),
        "--plan",
    )
    code_big, output_big = _invoke(
        "judge",
        str(big_corpus),
        "--config",
        str(big_config),
        "--out",
        str(tmp_path / "big" / "out"),
        "--plan",
    )

    assert code_small == 0, f"small-payload plan failed:\n{output_small}"
    assert code_big == 0, f"big-payload plan failed:\n{output_big}"

    count_small, tokens_small = _parse_plan_output(output_small)
    count_big, tokens_big = _parse_plan_output(output_big)

    assert count_small == count_big, (
        f"expected equal pending-item counts (same clause structure); "
        f"got small={count_small} big={count_big}"
    )
    assert tokens_big > tokens_small, (
        "expected the larger-payload corpus to produce a strictly larger "
        f"token estimate; got small={tokens_small} big={tokens_big} "
        f"(small output:\n{output_small}\nbig output:\n{output_big})"
    )

    assert "Segmentation:" in output_small, f"no segmentation-cost line in:\n{output_small}"
    assert "Segmentation:" in output_big, f"no segmentation-cost line in:\n{output_big}"


# ---------------------------------------------------------------------------
# AC-2: Full end-to-end path: judge → judge-apply → mine → project → validate
# ---------------------------------------------------------------------------


@pytest.fixture()
def e2e_out(tmp_path: Path) -> Path:
    """Shared output directory for the end-to-end flow."""
    return tmp_path / "out"


def test_e2e_judge_writes_observations(e2e_out: Path) -> None:
    """judge (first run) writes observations.jsonl."""
    code, output = _invoke(
        "judge",
        str(_CORPUS_DIR),
        "--config",
        str(_CONFIG_PATH),
        "--out",
        str(e2e_out),
    )
    assert code == 0, f"judge failed:\n{output}"
    assert (e2e_out / "observations.jsonl").exists(), "observations.jsonl not written by judge"


def test_e2e_judge_writes_pending_queue(e2e_out: Path) -> None:
    """judge (first run) writes pending.jsonl."""
    _invoke(
        "judge",
        str(_CORPUS_DIR),
        "--config",
        str(_CONFIG_PATH),
        "--out",
        str(e2e_out),
    )
    pending_path = e2e_out / "judge" / "pending.jsonl"
    assert pending_path.exists(), "pending.jsonl not written by judge"
    lines = [
        json.loads(line)
        for line in pending_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(lines) > 0, "pending.jsonl must not be empty after first judge run"


def test_e2e_judge_apply_exits_zero(e2e_out: Path) -> None:
    """judge-apply exits 0 when loading the canned verdicts file."""
    # First run judge to create the judge/ directory.
    _invoke(
        "judge",
        str(_CORPUS_DIR),
        "--config",
        str(_CONFIG_PATH),
        "--out",
        str(e2e_out),
    )
    code, output = _invoke(
        "judge-apply",
        str(e2e_out),
        "--verdicts",
        str(_CANNED_VERDICTS),
    )
    assert code == 0, f"judge-apply failed:\n{output}"


def test_e2e_full_path_taxonomy_ids_populated(e2e_out: Path) -> None:
    """AC-2: Full path → observations.jsonl has taxonomy_id populated for judged clauses."""
    # Step 1: judge (first run, records pending).
    code, output = _invoke(
        "judge",
        str(_CORPUS_DIR),
        "--config",
        str(_CONFIG_PATH),
        "--out",
        str(e2e_out),
    )
    assert code == 0, f"judge failed:\n{output}"

    # Step 2: judge-apply (load canned verdicts).
    code, output = _invoke(
        "judge-apply",
        str(e2e_out),
        "--verdicts",
        str(_CANNED_VERDICTS),
    )
    assert code == 0, f"judge-apply failed:\n{output}"

    # Step 3: mine (re-mines with store-backed judges, now populated).
    code, output = _invoke(
        "mine",
        str(_CORPUS_DIR),
        "--config",
        str(_CONFIG_PATH),
        "--out",
        str(e2e_out),
    )
    assert code == 0, f"mine failed:\n{output}"

    # Step 4: project.
    code, output = _invoke(
        "project",
        str(e2e_out),
        "--config",
        str(_CONFIG_PATH),
    )
    assert code == 0, f"project failed:\n{output}"

    # Check: taxonomy_id populated for clauses that went through the judge.
    obs_list = read_observations_jsonl(e2e_out / "observations.jsonl")
    assert len(obs_list) > 0, "observations.jsonl is empty"

    # All observations must have a non-None taxonomy_id (judge verdicts provided all).
    none_tax = [o for o in obs_list if o["taxonomy_id"] is None]
    assert none_tax == [], (
        f"Expected all taxonomy_ids to be populated after judge-apply; "
        f"still None for: {[o['observation_id'] for o in none_tax]}"
    )

    # At least one observation must have basis='judge' (store-backed verdict replayed).
    judge_obs = [o for o in obs_list if o["basis"] == "judge"]
    assert len(judge_obs) > 0, (
        "Expected at least one observation with basis='judge' after applying verdicts"
    )


def test_e2e_validate_exits_zero(e2e_out: Path) -> None:
    """AC-2: playbook validate exits 0 on the compiled playbook."""
    # Run the full pipeline first.
    _invoke("judge", str(_CORPUS_DIR), "--config", str(_CONFIG_PATH), "--out", str(e2e_out))
    _invoke("judge-apply", str(e2e_out), "--verdicts", str(_CANNED_VERDICTS))
    _invoke("mine", str(_CORPUS_DIR), "--config", str(_CONFIG_PATH), "--out", str(e2e_out))
    _invoke("project", str(e2e_out), "--config", str(_CONFIG_PATH))

    playbook_path = e2e_out / "playbook.opf.json"
    assert playbook_path.exists(), "playbook.opf.json not written"

    doc = load_opf_file(playbook_path)
    result = validate_document(doc)
    blocking = [str(e) for e in result.errors if e.blocking]
    assert blocking == [], f"Schema validation errors after full e2e path: {blocking}"


# ---------------------------------------------------------------------------
# AC-3: Idempotent — re-running judge after judge-apply queues 0 new items
# ---------------------------------------------------------------------------


def test_judge_idempotent_after_apply(tmp_path: Path) -> None:
    """AC-3: Re-running judge after judge-apply reports 0 pending items."""
    out_dir = tmp_path / "out"

    # Step 1: first judge run.
    _invoke("judge", str(_CORPUS_DIR), "--config", str(_CONFIG_PATH), "--out", str(out_dir))

    # Step 2: load canned verdicts.
    _invoke("judge-apply", str(out_dir), "--verdicts", str(_CANNED_VERDICTS))

    # Step 3: re-run judge --plan; must show 0 pending.
    code, output = _invoke(
        "judge",
        str(_CORPUS_DIR),
        "--config",
        str(_CONFIG_PATH),
        "--out",
        str(out_dir),
        "--plan",
    )
    assert code == 0, f"judge --plan (after apply) failed:\n{output}"
    assert "0 pending" in output, (
        f"Expected '0 pending' in output after all verdicts applied; got:\n{output}"
    )


def test_judge_writes_empty_pending_file_when_round_completes_clean(tmp_path: Path) -> None:
    """Issue #170: a completed round with 0 new pending items must leave an
    EMPTY pending.jsonl on disk, not an absent one.

    judge_cmd unlinks pending.jsonl at the top of every round and
    PendingQueue only creates the file lazily via ``add()`` — so before the
    fix, a round that queued 0 items (e.g. this second round, after all
    verdicts were applied) left the file absent, which is indistinguishable
    from a round that crashed mid-``mine`` before ever touching the queue.
    REFERENCE.md's done-criteria treats "file exists and is empty" as done
    and absence as NOT done, so the file must actually exist here.
    """
    out_dir = tmp_path / "out"

    # Round 1: queues pending items.
    code, output = _invoke("judge", str(_CORPUS_DIR), "--config", str(_CONFIG_PATH), "--out", str(out_dir))
    assert code == 0, f"judge (round 1) failed:\n{output}"

    # Apply canned verdicts so round 2 has nothing left to queue.
    code, output = _invoke(
        "judge-apply", str(out_dir), "--verdicts", str(_CANNED_VERDICTS)
    )
    assert code == 0, f"judge-apply failed:\n{output}"

    # Round 2: a clean, fully-judged re-run — 0 new pending items.
    code, output = _invoke("judge", str(_CORPUS_DIR), "--config", str(_CONFIG_PATH), "--out", str(out_dir))
    assert code == 0, f"judge (round 2) failed:\n{output}"
    assert "(0 pending items)" in output, f"Expected 0 pending items; got:\n{output}"

    pending_path = out_dir / "judge" / "pending.jsonl"
    assert pending_path.exists(), (
        "pending.jsonl must exist (empty) after a round that completed with "
        "0 pending items — its absence must be reserved for an interrupted "
        "round (issue #170)"
    )
    assert pending_path.read_text(encoding="utf-8") == "", (
        f"Expected pending.jsonl to be empty; got: {pending_path.read_text(encoding='utf-8')!r}"
    )


# ---------------------------------------------------------------------------
# Edge cases and error paths
# ---------------------------------------------------------------------------


def test_judge_apply_missing_verdicts_flag(tmp_path: Path) -> None:
    """judge-apply without --verdicts exits non-zero."""
    code, _ = _invoke("judge-apply", str(tmp_path))
    assert code != 0


# ---------------------------------------------------------------------------
# Issue #55: judge-apply against a typo'd/nonexistent OUT_DIR must fail fast
# instead of silently creating a fresh, unreachable verdict store there.
# ---------------------------------------------------------------------------


def test_judge_apply_nonexistent_out_dir_exits_nonzero(tmp_path: Path) -> None:
    """judge-apply against a nonexistent OUT_DIR exits 1 and creates nothing."""
    missing_dir = tmp_path / "totally-nonexistent-dir"
    code, output = _invoke("judge-apply", str(missing_dir), "--verdicts", str(_CANNED_VERDICTS))
    assert code == 1, f"expected exit 1 for missing OUT_DIR; got {code}:\n{output}"
    assert "ERROR" in output
    assert "does not exist" in output
    assert not missing_dir.exists(), (
        "judge-apply must not create OUT_DIR when it is missing (issue #55)"
    )


def test_judge_apply_preseeded_out_dir_still_succeeds(tmp_path: Path) -> None:
    """A fresh OUT_DIR with no judge/ subdir yet (pre-seeding workflow) still works.

    The guard must reject a missing OUT_DIR itself, not the judge/ subdir —
    otherwise it would block copying verdicts.jsonl into a fresh out dir
    before any `playbook judge` round has run (see _verdict_store_kwargs).
    """
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    assert not (out_dir / "judge").exists()
    code, output = _invoke("judge-apply", str(out_dir), "--verdicts", str(_CANNED_VERDICTS))
    assert code == 0, f"judge-apply on a pre-seeded fresh OUT_DIR failed:\n{output}"
    assert (out_dir / "judge" / "verdicts.jsonl").exists()


def test_judge_apply_invalid_json_exits_nonzero(tmp_path: Path) -> None:
    """judge-apply with an invalid JSONL file exits non-zero."""
    bad_file = tmp_path / "bad.jsonl"
    bad_file.write_text("NOT JSON\n", encoding="utf-8")
    code, output = _invoke("judge-apply", str(tmp_path), "--verdicts", str(bad_file))
    assert code != 0


def test_judge_apply_missing_key_field_exits_nonzero(tmp_path: Path) -> None:
    """judge-apply rejects a record missing the 'key' field."""
    bad_file = tmp_path / "bad.jsonl"
    bad_file.write_text(json.dumps({"verdict": {"taxonomy_id": "x"}}) + "\n", encoding="utf-8")
    code, output = _invoke("judge-apply", str(tmp_path), "--verdicts", str(bad_file))
    assert code != 0


def test_judge_apply_missing_verdict_field_exits_nonzero(tmp_path: Path) -> None:
    """judge-apply rejects a record missing the 'verdict' field."""
    bad_file = tmp_path / "bad.jsonl"
    bad_file.write_text(json.dumps({"key": "abc123"}) + "\n", encoding="utf-8")
    code, output = _invoke("judge-apply", str(tmp_path), "--verdicts", str(bad_file))
    assert code != 0


def test_judge_subset_limits_pending(tmp_path: Path) -> None:
    """--subset N limits the pending queue to N items."""
    out_dir = tmp_path / "out"
    code, output = _invoke(
        "judge",
        str(_CORPUS_DIR),
        "--config",
        str(_CONFIG_PATH),
        "--out",
        str(out_dir),
        "--subset",
        "2",
    )
    assert code == 0, f"judge --subset failed:\n{output}"
    pending_path = out_dir / "judge" / "pending.jsonl"
    assert pending_path.exists(), "pending.jsonl not written"
    lines = [line for line in pending_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) <= 2, f"Expected at most 2 pending items with --subset 2; got {len(lines)}"


# ---------------------------------------------------------------------------
# Issue #295: --plan must honor --subset instead of silently ignoring it.
# ---------------------------------------------------------------------------


def test_judge_plan_with_subset_reflects_subset_count(tmp_path: Path) -> None:
    """judge --plan --subset 2 reports the capped count, not the full corpus.

    Before the fix, the --plan branch returned before --subset was ever
    read, so this combination silently reported full-corpus pending counts.
    """
    code, output = _invoke(
        "judge",
        str(_CORPUS_DIR),
        "--config",
        str(_CONFIG_PATH),
        "--out",
        str(tmp_path),
        "--plan",
        "--subset",
        "2",
    )
    assert code == 0, f"judge --plan --subset 2 failed:\n{output}"
    assert "Pending items: 2" in output, (
        f"Expected 'Pending items: 2' with --subset 2; got:\n{output}"
    )


def test_judge_plan_with_invalid_subset_exits_nonzero(tmp_path: Path) -> None:
    """judge --plan --subset 0 must be rejected, same as normal mode.

    Before the fix, the subset<=0 validation lived only in normal mode
    (after the --plan branch's early return), so this combination passed
    silently despite being invalid.
    """
    code, output = _invoke(
        "judge",
        str(_CORPUS_DIR),
        "--config",
        str(_CONFIG_PATH),
        "--out",
        str(tmp_path),
        "--plan",
        "--subset",
        "0",
    )
    assert code != 0, f"judge --plan --subset 0 should fail; got exit 0:\n{output}"
    assert "--subset must be a positive integer" in output


def test_judge_missing_config_exits_nonzero(tmp_path: Path) -> None:
    """judge with a missing config file exits non-zero."""
    code, _ = _invoke(
        "judge",
        str(_CORPUS_DIR),
        "--config",
        str(tmp_path / "nonexistent.yaml"),
    )
    assert code != 0


# ---------------------------------------------------------------------------
# Issue #132: extraction reuse across judge rounds (LLM-segmentation path)
#
# ``judge`` (and ``judge --plan``) force no_cache=True on the L1-L4
# ArtifactStore/JudgmentCache to avoid replaying stale needs_review
# sentinels from the store-backed judges — but that must not also force a
# full re-extraction (docling/pandoc/pdfplumber/python-docx) of every
# version on every round. This is proven at the CLI level, over the real
# LLM-segmentation path, by counting calls into the underlying RTF
# extraction helper (mirrors tests/test_cli.py's
# mine/judge segmentation-wiring fixtures) — no live network call is
# made (anthropic.Anthropic is monkeypatched).
# ---------------------------------------------------------------------------

_RTF_PROLOGUE = (
    r"{\rtf1\ansi\deff0"
    r"{\fonttbl{\f0\froman\fcharset0 Times New Roman;}}"
    r"\f0\fs24 "
)
_RTF_EPILOGUE = r"}"


def _write_rtf(path: Path, body: str) -> None:
    path.write_text(_RTF_PROLOGUE + body + _RTF_EPILOGUE, encoding="utf-8")


# Two clauses, single document version — a single-clause document is
# rejected by the scope gate as "too short to evaluate" (same convention as
# tests/test_cli.py's _LLM_DEAL_V1).
_LLM_DEAL_V1 = (
    r"1. Indemnification\par "
    r"Alpha Corp shall indemnify Beta University against third-party claims "
    r"arising from the placement programme.\par "
    r"2. Governing Law\par "
    r"This agreement is governed by the laws of the State of California.\par "
)

_LLM_SEGMENT_RESPONSE = json.dumps(
    {
        "nodes": [
            {
                "node_id": "n1",
                "parent_id": None,
                "order": 1,
                "heading": "1. Indemnification",
                "taxonomy_id": "indemnification",
                "start_block_id": "b0",
                "end_block_id": "b1",
                "start_quote": "1. Indemn",
                "end_quote": "programme.",
            },
            {
                "node_id": "n2",
                "parent_id": None,
                "order": 2,
                "heading": "2. Governing Law",
                "taxonomy_id": "governing_law",
                "start_block_id": "b2",
                "end_block_id": "b3",
                "start_quote": "2. Govern",
                "end_quote": "California.",
            },
        ]
    }
)


def _make_llm_judge_corpus(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Single-document, single-version corpus + config with
    ``segmentation: {llm: true, cache: true}`` — the ``cache: true`` flag
    wires both the segmentation verdict cache and (issue #132) the new
    ``ExtractionCache`` through ``cli._llm_segmentation_kwargs``."""
    corpus_dir = tmp_path / "corpus"
    deal_dir = corpus_dir / "deal-001"
    deal_dir.mkdir(parents=True)
    _write_rtf(deal_dir / "v1.rtf", _LLM_DEAL_V1)

    taxonomy_path = (
        Path(__file__).parent.parent / "spec" / "taxonomy" / "affiliation-agreement.yaml"
    )
    cfg = {
        "agreement_type": {
            "id": "educational-affiliation",
            "name": "Educational Affiliation Agreement",
        },
        "baseline": {"template": None},
        "taxonomy": str(taxonomy_path),
        "provenance": {"our_party_aliases": ["Alpha Corp"]},
        "segmentation": {"llm": True, "cache": True},
    }
    config_path = tmp_path / "playbook.config.yaml"
    config_path.write_text(yaml.dump(cfg), encoding="utf-8")

    out_dir = tmp_path / "out"
    return corpus_dir, config_path, out_dir


class _RecordingMessages:
    """Fake ``client.messages`` — records calls, returns a canned response."""

    def __init__(self, response_text: str) -> None:
        self.response_text = response_text
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=self.response_text)])


class _RecordingAnthropicClient:
    """Fake ``anthropic.Anthropic()`` instance — no live network call."""

    instances: list[_RecordingAnthropicClient] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.messages = _RecordingMessages(_LLM_SEGMENT_RESPONSE)
        _RecordingAnthropicClient.instances.append(self)


@pytest.fixture
def _fake_anthropic(monkeypatch: pytest.MonkeyPatch) -> type[_RecordingAnthropicClient]:
    """Monkeypatch ``anthropic.Anthropic`` and set a dummy API key so the LLM
    segmentation path never makes a live call (mirrors tests/test_cli.py)."""
    import anthropic

    _RecordingAnthropicClient.instances = []
    monkeypatch.setattr(anthropic, "Anthropic", _RecordingAnthropicClient)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-dummy-key")
    return _RecordingAnthropicClient


@pytest.fixture
def _count_rtf_extractions(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """Wrap ``extraction._extract_rtf_lines`` (the real pandoc-backed helper)
    with a counter, so tests can assert extraction was NOT re-invoked on a
    cache hit."""
    calls: list[Path] = []
    real_extract_rtf_lines = extraction._extract_rtf_lines

    def _counting(path: Path) -> list[tuple[str, int]]:
        calls.append(path)
        return real_extract_rtf_lines(path)

    monkeypatch.setattr(extraction, "_extract_rtf_lines", _counting)
    return calls


def test_judge_second_round_reuses_cached_extraction(
    tmp_path: Path,
    _fake_anthropic: type[_RecordingAnthropicClient],
    _count_rtf_extractions: list[Path],
) -> None:
    """A second ``judge`` round over an unchanged corpus must NOT re-extract
    (issue #132) — before the fix, ``judge`` forced no_cache=True on the
    L1-L4 stage cache (to avoid replaying stale needs_review sentinels),
    which incidentally forced a full re-extraction of every version on every
    round too."""
    corpus_dir, config_path, out_dir = _make_llm_judge_corpus(tmp_path)
    runner = CliRunner()

    result_1 = runner.invoke(
        cli,
        ["judge", str(corpus_dir), "--config", str(config_path), "--out", str(out_dir)],
    )
    assert result_1.exit_code == 0, f"first judge round failed:\n{result_1.output}"
    assert len(_count_rtf_extractions) == 1, "first judge round must extract the one version"

    result_2 = runner.invoke(
        cli,
        ["judge", str(corpus_dir), "--config", str(config_path), "--out", str(out_dir)],
    )
    assert result_2.exit_code == 0, f"second judge round failed:\n{result_2.output}"
    assert len(_count_rtf_extractions) == 1, (
        "second judge round must reuse the cached extracted blocks, not re-extract"
    )


def test_judge_plan_reuses_extraction_from_prior_judge_round(
    tmp_path: Path,
    _fake_anthropic: type[_RecordingAnthropicClient],
    _count_rtf_extractions: list[Path],
) -> None:
    """``judge --plan`` reads the cache the real ``out_dir`` already has warm
    (issue #132) — before the fix, --plan mined into a fresh
    TemporaryDirectory with no extraction cache of its own, so it always
    re-extracted from scratch even immediately after a full judge round."""
    corpus_dir, config_path, out_dir = _make_llm_judge_corpus(tmp_path)
    runner = CliRunner()

    result = runner.invoke(
        cli,
        ["judge", str(corpus_dir), "--config", str(config_path), "--out", str(out_dir)],
    )
    assert result.exit_code == 0, f"judge failed:\n{result.output}"
    assert len(_count_rtf_extractions) == 1

    result_plan = runner.invoke(
        cli,
        [
            "judge",
            str(corpus_dir),
            "--config",
            str(config_path),
            "--out",
            str(out_dir),
            "--plan",
        ],
    )
    assert result_plan.exit_code == 0, f"judge --plan failed:\n{result_plan.output}"
    assert len(_count_rtf_extractions) == 1, (
        "judge --plan must reuse the extraction cache from the prior judge round, "
        "not re-mine the corpus from scratch"
    )


# ---------------------------------------------------------------------------
# Issue #78: ``mine --no-cache`` must force re-extraction at the CLI seam
#
# ``mine_cmd`` originally had no --no-cache option at all, so an operator
# following the engine's own version_ingest_failed remediation ("Inspect the
# source file and re-run 'playbook mine' with --no-cache" — inspection_report.py)
# hit "Error: No such option '--no-cache'." on exactly
# the mine->project workflow the remediation text points at.
#
# The tests above (and the lower-level tests in test_pipeline_llm_seg.py /
# test_extraction.py) all enter at the mine_corpus/extract_blocks API layer,
# so none of them exercise cli.py's ``"refresh_extraction": no_cache`` kwarg
# itself — the actual operator-facing deliverable of issue #78. These two
# tests invoke ``mine`` through CliRunner (the real entry point an operator
# uses) and count extractor calls across the full CLI wiring.
# ---------------------------------------------------------------------------


def test_mine_no_cache_forces_reextraction(
    tmp_path: Path,
    _fake_anthropic: type[_RecordingAnthropicClient],
    _count_rtf_extractions: list[Path],
) -> None:
    """``mine --no-cache`` must re-extract despite a warm
    ``extraction_cache.jsonl``, and must leave the cache refreshed so a
    subsequent plain run stays warm (issue #78 round 2).

    Before the fix, ``mine`` had no ``--no-cache`` option at all — this test
    exercises the real CLI entry point (``CliRunner``), not the
    ``mine_corpus``/``extract_blocks`` API layer directly, so it actually
    covers the operator-facing deliverable.
    """
    corpus_dir, config_path, out_dir = _make_llm_judge_corpus(tmp_path)
    runner = CliRunner()

    def _mine(*extra_args: str) -> None:
        result = runner.invoke(
            cli,
            [
                "mine",
                str(corpus_dir),
                "--config",
                str(config_path),
                "--out",
                str(out_dir),
                *extra_args,
            ],
        )
        assert result.exit_code == 0, f"mine {' '.join(extra_args)} failed:\n{result.output}"

    _mine()
    assert len(_count_rtf_extractions) == 1, "first mine must extract the one version"

    _mine()
    assert len(_count_rtf_extractions) == 1, (
        "plain repeat mine must reuse the warm extraction cache, not re-extract"
    )

    _mine("--no-cache")
    assert len(_count_rtf_extractions) == 2, (
        "mine --no-cache must force re-extraction despite a warm extraction_cache.jsonl (issue #78)"
    )

    _mine()
    assert len(_count_rtf_extractions) == 2, (
        "mine after --no-cache must reuse the freshly-refreshed extraction "
        "cache — proving --no-cache left the cache warm, not just bypassed"
    )


def test_mine_verdict_store_forced_no_cache_does_not_reextract(
    tmp_path: Path,
    _fake_anthropic: type[_RecordingAnthropicClient],
    _count_rtf_extractions: list[Path],
) -> None:
    """A verdict-store-forced ``no_cache=True`` (``_verdict_store_kwargs``,
    triggered whenever ``out_dir/judge/verdicts.jsonl`` exists) must NOT
    force re-extraction on the ``mine`` path (issue #78): ``mine_kwargs``
    merges as ``{"no_cache": no_cache, "refresh_extraction": no_cache,
    **seg_kwargs, **verdict_kwargs}``, so ``verdict_kwargs`` can override
    ``no_cache`` to True without the operator ever passing --no-cache, but
    must not also override ``refresh_extraction``.
    """
    corpus_dir, config_path, out_dir = _make_llm_judge_corpus(tmp_path)
    runner = CliRunner()

    def _mine() -> str:
        result = runner.invoke(
            cli,
            ["mine", str(corpus_dir), "--config", str(config_path), "--out", str(out_dir)],
        )
        assert result.exit_code == 0, f"mine failed:\n{result.output}"
        return result.output

    _mine()
    assert len(_count_rtf_extractions) == 1, "first mine must extract the one version"

    verdicts_path = out_dir / "judge" / "verdicts.jsonl"
    verdicts_path.parent.mkdir(parents=True, exist_ok=True)
    verdicts_path.touch()

    output = _mine()
    assert "store-backed judges active" in output, (
        "mine must have wired store-backed judges (forcing no_cache=True) "
        "for this assertion to actually exercise the forced-no_cache path"
    )
    assert len(_count_rtf_extractions) == 1, (
        "a verdict-store-forced no_cache=True must not force re-extraction — "
        "extraction_cache must stay warm across post-judge mine rounds"
    )


# ---------------------------------------------------------------------------
# Apply-time semantic verdict validation (pre-derivation QA)
# ---------------------------------------------------------------------------


def test_judge_apply_rejects_invalid_magnitude(tmp_path: Path) -> None:
    """The documented-then-corrected 'moderate' magnitude fails at apply, not replay."""
    bad_file = tmp_path / "verdicts.jsonl"
    bad_file.write_text(
        json.dumps(
            {
                "key": "a" * 64,
                "verdict": {
                    "deviation": "substantive",
                    "risk_delta": {"direction": "worse", "magnitude": "moderate"},
                    "basis": "judge",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    code, output = _invoke("judge-apply", str(tmp_path), "--verdicts", str(bad_file))
    assert code != 0
    assert "magnitude" in output


def test_judge_apply_rejects_missing_risk_delta(tmp_path: Path) -> None:
    bad_file = tmp_path / "verdicts.jsonl"
    bad_file.write_text(
        json.dumps({"key": "a" * 64, "verdict": {"deviation": "none", "basis": "judge"}}) + "\n",
        encoding="utf-8",
    )
    code, output = _invoke("judge-apply", str(tmp_path), "--verdicts", str(bad_file))
    assert code != 0
    assert "risk_delta" in output


def test_judge_apply_rejects_unresolved_basis(tmp_path: Path) -> None:
    """basis='needs_review' means 'not judged' — a supplied verdict may not carry it."""
    bad_file = tmp_path / "verdicts.jsonl"
    bad_file.write_text(
        json.dumps(
            {
                "key": "a" * 64,
                "verdict": {
                    "provenance": "our_paper",
                    "confidence": 0.9,
                    "basis": "needs_review",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    code, output = _invoke("judge-apply", str(tmp_path), "--verdicts", str(bad_file))
    assert code != 0
    assert "basis" in output


def test_judge_apply_rejects_neutral_with_nonzero_magnitude(tmp_path: Path) -> None:
    bad_file = tmp_path / "verdicts.jsonl"
    bad_file.write_text(
        json.dumps(
            {
                "key": "a" * 64,
                "verdict": {
                    "deviation": "substantive",
                    "risk_delta": {"direction": "neutral", "magnitude": "minor"},
                    "basis": "judge",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    code, output = _invoke("judge-apply", str(tmp_path), "--verdicts", str(bad_file))
    assert code != 0


def test_judge_apply_rejects_string_confidence_with_actionable_error(tmp_path: Path) -> None:
    """Issue #161: a stringified confidence (a common LLM-producer mistake)
    must fail with a line-numbered, field-named error, not a bare traceback."""
    bad_file = tmp_path / "verdicts.jsonl"
    bad_file.write_text(
        json.dumps(
            {
                "key": "a" * 64,
                "verdict": {
                    "taxonomy_id": "indemnification",
                    "confidence": "0.8",
                    "basis": "judge",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    code, output = _invoke("judge-apply", str(tmp_path), "--verdicts", str(bad_file))
    assert code != 0
    assert "line 1" in output
    assert "confidence" in output
    assert "Traceback" not in output


def test_judge_apply_rejects_deviation_string_confidence(tmp_path: Path) -> None:
    """Issue #161: deviation verdicts previously skipped confidence validation
    entirely and banked a malformed value silently."""
    bad_file = tmp_path / "verdicts.jsonl"
    bad_file.write_text(
        json.dumps(
            {
                "key": "a" * 64,
                "verdict": {
                    "deviation": "substantive",
                    "risk_delta": {"direction": "worse", "magnitude": "minor"},
                    "basis": "judge",
                    "confidence": "0.7",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    code, output = _invoke("judge-apply", str(tmp_path), "--verdicts", str(bad_file))
    assert code != 0
    assert "confidence" in output


def test_judge_apply_accepts_valid_verdicts_of_each_kind(tmp_path: Path) -> None:
    good_file = tmp_path / "verdicts.jsonl"
    lines = [
        {
            "key": "a" * 64,
            "verdict": {"taxonomy_id": "indemnification", "confidence": 0.9, "basis": "judge"},
        },
        {
            "key": "b" * 64,
            "verdict": {
                "deviation": "substantive",
                "risk_delta": {"direction": "worse", "magnitude": "material"},
                "confidence": 0.8,
                "basis": "judge",
                "rationale": "Cap removed.",
            },
        },
        {
            "key": "c" * 64,
            "verdict": {"provenance": "our_paper", "confidence": 0.85, "basis": "llm"},
        },
        {"key": "d" * 64, "verdict": {"in_scope": True, "scope_confidence": 0.95}},
    ]
    good_file.write_text("\n".join(json.dumps(rec) for rec in lines) + "\n", encoding="utf-8")
    code, output = _invoke("judge-apply", str(tmp_path), "--verdicts", str(good_file))
    assert code == 0, output
    assert "loaded 4 verdict(s)" in output


def test_judge_apply_loads_nothing_when_a_later_line_fails(tmp_path: Path) -> None:
    """All-or-nothing load (issue #160): a valid line preceding an invalid one
    must NOT be banked in the store — validation happens for every line
    before anything is written, so one bad line downstream voids the whole
    batch, not just itself.
    """
    mixed_file = tmp_path / "verdicts.jsonl"
    lines = [
        # Valid classification verdict — would load fine on its own.
        {
            "key": "a" * 64,
            "verdict": {"taxonomy_id": "indemnification", "confidence": 0.9, "basis": "judge"},
        },
        # Invalid: missing 'verdict' field entirely.
        {"key": "b" * 64},
    ]
    mixed_file.write_text("\n".join(json.dumps(rec) for rec in lines) + "\n", encoding="utf-8")
    code, output = _invoke("judge-apply", str(tmp_path), "--verdicts", str(mixed_file))
    assert code != 0, output

    store_path = tmp_path / "judge" / "verdicts.jsonl"
    if store_path.exists():
        stored_keys = {json.loads(line)["key"] for line in store_path.read_text().splitlines()}
        assert "a" * 64 not in stored_keys, (
            "the earlier valid line was banked despite a later line failing — "
            "judge-apply must validate all lines before loading any"
        )
    # else: store file was never created at all, which also satisfies "nothing loaded".
