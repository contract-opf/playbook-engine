"""Tests for the CLI entry point, including the ``compile`` command.

SECURITY NOTE: All corpus fixtures are programmatically constructed with
synthetic text.  No real agreement files are committed or referenced.
Fictional party/document names only (e.g. "Alpha Corp", "Beta University").
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml
from click.testing import CliRunner

from playbook_engine.cli import cli
from playbook_engine.validator import validate_document

# ---------------------------------------------------------------------------
# Existing smoke tests
# ---------------------------------------------------------------------------


def test_help() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "playbook-engine" in result.output


def test_version() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert "0.2.0" in result.output


def test_validate_stub_exits_nonzero() -> None:
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        f.write(b"{}")
        path = f.name
    try:
        runner = CliRunner()
        result = runner.invoke(cli, ["validate", path])
        assert result.exit_code != 0
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# Compile command fixture helpers
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


# Synthetic agreement text (fictional parties only — Alpha Corp, Beta University)
_DEAL_001_V1 = (
    r"1. Indemnification\par "
    r"Alpha Corp shall indemnify Beta University against third-party claims "
    r"arising from the placement programme.\par "
    r"2. Governing Law\par "
    r"This agreement is governed by the laws of the State of California.\par "
    r"3. Term\par "
    r"This agreement commences on the date of execution and continues for one year.\par "
)

_DEAL_001_V2 = (
    r"1. Indemnification\par "
    r"The parties shall mutually indemnify each other against third-party claims "
    r"arising from the placement programme.\par "
    r"2. Governing Law\par "
    r"This agreement is governed by the laws of the State of New York.\par "
    r"3. Term\par "
    r"This agreement commences on the date of execution and continues for one year.\par "
)

_DEAL_002_V1 = (
    r"1. Indemnification\par "
    r"Beta University shall not be liable for any claims related to student placements.\par "
    r"2. Governing Law\par "
    r"This agreement is governed by the laws of the State of Texas.\par "
    r"3. Term\par "
    r"Initial term of two years with automatic renewal unless terminated with 60 days notice.\par "
)


def _make_corpus(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Build a synthetic fixture corpus and config; return (corpus_dir, config_path, out_dir)."""
    # Corpus layout:
    #   corpus/
    #     deal-001/  (two versions)
    #     deal-002/  (one version)
    corpus_dir = tmp_path / "corpus"
    deal_001 = corpus_dir / "deal-001"
    deal_001.mkdir(parents=True)
    _write_rtf(deal_001 / "v1.rtf", _DEAL_001_V1)
    _write_rtf(deal_001 / "v2.rtf", _DEAL_001_V2)

    deal_002 = corpus_dir / "deal-002"
    deal_002.mkdir()
    _write_rtf(deal_002 / "v1.rtf", _DEAL_002_V1)

    # Config YAML (no template — emergent playbook)
    # Taxonomy points to the existing affiliation taxonomy in the repo.
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
    }
    config_path = tmp_path / "playbook.config.yaml"
    config_path.write_text(yaml.dump(cfg), encoding="utf-8")

    out_dir = tmp_path / "out"
    return corpus_dir, config_path, out_dir


# ---------------------------------------------------------------------------
# Compile command: acceptance tests
# ---------------------------------------------------------------------------


def test_compile_produces_schema_valid_playbook(tmp_path: Path) -> None:
    """Acceptance: compile produces a schema-valid OPF playbook.opf.json."""
    corpus_dir, config_path, out_dir = _make_corpus(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "compile",
            str(corpus_dir),
            "--config",
            str(config_path),
            "--out",
            str(out_dir),
        ],
    )
    assert result.exit_code == 0, f"compile failed:\n{result.output}"

    playbook_path = out_dir / "playbook.opf.json"
    assert playbook_path.exists(), "playbook.opf.json not written"

    playbook = json.loads(playbook_path.read_text())
    validation = validate_document(playbook)
    blocking = [str(e) for e in validation.errors if e.blocking]
    assert blocking == [], f"Schema validation errors: {blocking}"


def test_compile_writes_all_intermediates(tmp_path: Path) -> None:
    """All expected intermediate files are written."""
    corpus_dir, config_path, out_dir = _make_corpus(tmp_path)
    runner = CliRunner()
    runner.invoke(
        cli,
        [
            "compile",
            str(corpus_dir),
            "--config",
            str(config_path),
            "--out",
            str(out_dir),
        ],
    )

    assert (out_dir / "observations.jsonl").exists()
    assert (out_dir / "scope.json").exists()
    assert (out_dir / "corpus_manifest.json").exists()
    assert (out_dir / "trail" / "deal-001.json").exists()
    assert (out_dir / "trail" / "deal-002.json").exists()
    assert (out_dir / "normalized" / "deal-001" / "v1.clauses.json").exists()
    assert (out_dir / "normalized" / "deal-001" / "v2.clauses.json").exists()
    assert (out_dir / "normalized" / "deal-002" / "v1.clauses.json").exists()


def test_compile_playbook_opf_version(tmp_path: Path) -> None:
    """Compiled playbook has opf_version='0.2'."""
    corpus_dir, config_path, out_dir = _make_corpus(tmp_path)
    runner = CliRunner()
    runner.invoke(
        cli,
        [
            "compile",
            str(corpus_dir),
            "--config",
            str(config_path),
            "--out",
            str(out_dir),
        ],
    )
    pb = json.loads((out_dir / "playbook.opf.json").read_text())
    assert pb["opf_version"] == "0.2"


def test_compile_corpus_stats_correct(tmp_path: Path) -> None:
    """Corpus stats reflect the fixture corpus (2 documents, 3 version files)."""
    corpus_dir, config_path, out_dir = _make_corpus(tmp_path)
    runner = CliRunner()
    runner.invoke(
        cli,
        [
            "compile",
            str(corpus_dir),
            "--config",
            str(config_path),
            "--out",
            str(out_dir),
        ],
    )
    pb = json.loads((out_dir / "playbook.opf.json").read_text())
    stats = pb["corpus"]["stats"]
    assert stats["documents_total"] == 2
    assert stats["documents_in_scope"] == 2
    assert stats["versions_total"] == 3  # deal-001 has 2, deal-002 has 1


def test_compile_exit_code_zero_on_success(tmp_path: Path) -> None:
    """CLI exits 0 on successful compile."""
    corpus_dir, config_path, out_dir = _make_corpus(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "compile",
            str(corpus_dir),
            "--config",
            str(config_path),
            "--out",
            str(out_dir),
        ],
    )
    assert result.exit_code == 0


def test_compile_cache_hit_produces_identical_content(tmp_path: Path) -> None:
    """Second run with the content-addressed cache produces byte-identical observations."""
    corpus_dir, config_path, out_dir = _make_corpus(tmp_path)
    runner = CliRunner()
    # First run — primes the cache.
    runner.invoke(
        cli,
        [
            "compile",
            str(corpus_dir),
            "--config",
            str(config_path),
            "--out",
            str(out_dir),
        ],
    )
    first_content = (out_dir / "observations.jsonl").read_text(encoding="utf-8")

    # Second run — should use the cache and produce identical observations.
    runner.invoke(
        cli,
        [
            "compile",
            str(corpus_dir),
            "--config",
            str(config_path),
            "--out",
            str(out_dir),
        ],
    )
    second_content = (out_dir / "observations.jsonl").read_text(encoding="utf-8")
    assert second_content == first_content, (
        "observations.jsonl content must be identical on a cache-hit second run"
    )


def test_compile_no_cache_flag_reruns_pipeline(tmp_path: Path) -> None:
    """--no-cache disables the stage cache and forces a full recompute."""
    corpus_dir, config_path, out_dir = _make_corpus(tmp_path)
    runner = CliRunner()
    runner.invoke(
        cli,
        [
            "compile",
            str(corpus_dir),
            "--config",
            str(config_path),
            "--out",
            str(out_dir),
        ],
    )
    first_content = (out_dir / "observations.jsonl").read_text(encoding="utf-8")

    result = runner.invoke(
        cli,
        [
            "compile",
            str(corpus_dir),
            "--config",
            str(config_path),
            "--out",
            str(out_dir),
            "--no-cache",
        ],
    )
    assert result.exit_code == 0, f"compile --no-cache exited non-zero: {result.output}"
    second_content = (out_dir / "observations.jsonl").read_text(encoding="utf-8")
    # Content must be identical (same corpus, same config → deterministic output).
    assert second_content == first_content


def test_compile_missing_config_exits_nonzero(tmp_path: Path) -> None:
    """Missing config file causes CLI to exit non-zero."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "compile",
            str(corpus_dir),
            "--config",
            str(tmp_path / "nonexistent.yaml"),
        ],
    )
    assert result.exit_code != 0


def test_compile_scope_json_contains_both_docs(tmp_path: Path) -> None:
    """scope.json records a decision for every corpus document."""
    corpus_dir, config_path, out_dir = _make_corpus(tmp_path)
    runner = CliRunner()
    runner.invoke(
        cli,
        [
            "compile",
            str(corpus_dir),
            "--config",
            str(config_path),
            "--out",
            str(out_dir),
        ],
    )
    scope = json.loads((out_dir / "scope.json").read_text())
    doc_ids = {d["document_id"] for d in scope["documents"]}
    assert "deal-001" in doc_ids
    assert "deal-002" in doc_ids


def test_compile_wires_scope_judge_when_verdict_store_exists(tmp_path: Path) -> None:
    """Issue #102: `compile` must wire store-backed judges — like `mine` does
    — when OUT_DIR already has a verdict store, never silently recompute with
    the stub judges and overwrite already-judged observations.jsonl.

    With an empty store, every document is a miss: the store-backed scope
    judge queues it and raises, which ``scope_gate()`` converts into a
    retained ``basis="judge_error"`` decision at confidence 0.0. This must
    NOT be the stub's ``basis="judge"``/confidence 0.5/"stub mode" rationale
    — before this fix, `compile` never even checked for a verdict store, so
    it always ran the stub judges regardless of OUT_DIR's contents.
    """
    corpus_dir, config_path, out_dir = _make_corpus(tmp_path)
    judge_dir = out_dir / "judge"
    judge_dir.mkdir(parents=True)
    (judge_dir / "verdicts.jsonl").write_text("", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["compile", str(corpus_dir), "--config", str(config_path), "--out", str(out_dir)],
    )
    assert result.exit_code == 0, f"compile failed:\n{result.output}"
    assert "store-backed judges active" in result.output

    scope_log = json.loads((out_dir / "scope.json").read_text(encoding="utf-8"))
    documents = scope_log["documents"]
    assert documents, "no documents recorded in scope.json"
    for doc in documents:
        assert doc["basis"] != "judge" or doc["scope_confidence"] != 0.5, (
            f"document {doc['document_id']} was auto-accepted by the stub scope judge: {doc}"
        )
        assert "stub mode" not in doc["scope_rationale"]

    pending_path = judge_dir / "pending.jsonl"
    assert pending_path.exists(), "unstored documents must be queued for scope review"


# ---------------------------------------------------------------------------
# mine command: acceptance tests
# ---------------------------------------------------------------------------


def test_mine_exits_zero(tmp_path: Path) -> None:
    """mine command exits 0 on success."""
    corpus_dir, config_path, out_dir = _make_corpus(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "mine",
            str(corpus_dir),
            "--config",
            str(config_path),
            "--out",
            str(out_dir),
        ],
    )
    assert result.exit_code == 0, f"mine failed:\n{result.output}"


def test_mine_writes_observation_store(tmp_path: Path) -> None:
    """mine writes observations.jsonl and corpus_manifest.json."""
    corpus_dir, config_path, out_dir = _make_corpus(tmp_path)
    runner = CliRunner()
    runner.invoke(
        cli,
        [
            "mine",
            str(corpus_dir),
            "--config",
            str(config_path),
            "--out",
            str(out_dir),
        ],
    )
    assert (out_dir / "observations.jsonl").exists(), "observations.jsonl not written"
    assert (out_dir / "corpus_manifest.json").exists(), "corpus_manifest.json not written"
    assert (out_dir / "scope.json").exists(), "scope.json not written"


def test_mine_does_not_write_playbook(tmp_path: Path) -> None:
    """mine must NOT write playbook.opf.json."""
    corpus_dir, config_path, out_dir = _make_corpus(tmp_path)
    runner = CliRunner()
    runner.invoke(
        cli,
        [
            "mine",
            str(corpus_dir),
            "--config",
            str(config_path),
            "--out",
            str(out_dir),
        ],
    )
    assert not (out_dir / "playbook.opf.json").exists(), "mine must not produce playbook.opf.json"


def test_mine_writes_all_intermediates(tmp_path: Path) -> None:
    """mine writes trail/ and normalized/ intermediates."""
    corpus_dir, config_path, out_dir = _make_corpus(tmp_path)
    runner = CliRunner()
    runner.invoke(
        cli,
        [
            "mine",
            str(corpus_dir),
            "--config",
            str(config_path),
            "--out",
            str(out_dir),
        ],
    )
    assert (out_dir / "trail" / "deal-001.json").exists()
    assert (out_dir / "trail" / "deal-002.json").exists()
    assert (out_dir / "normalized" / "deal-001" / "v1.clauses.json").exists()
    assert (out_dir / "normalized" / "deal-001" / "v2.clauses.json").exists()
    assert (out_dir / "normalized" / "deal-002" / "v1.clauses.json").exists()


# ---------------------------------------------------------------------------
# mine command: store-backed scope judge wiring — issue #87
# ---------------------------------------------------------------------------


def test_mine_wires_scope_judge_when_verdict_store_exists(tmp_path: Path) -> None:
    """When a verdict store exists, mine must wire a real scope judge — not the
    ``_AllInScopeJudge`` stub that auto-accepts everything at confidence 0.5.

    With an empty store, every document is a miss: the store-backed scope
    judge queues it and raises, which ``scope_gate()`` converts into a
    retained ``basis="judge_error"`` decision at confidence 0.0. This must
    NOT be the stub's ``basis="judge"``/confidence 0.5/"stub mode" rationale.
    """
    corpus_dir, config_path, out_dir = _make_corpus(tmp_path)
    judge_dir = out_dir / "judge"
    judge_dir.mkdir(parents=True)
    (judge_dir / "verdicts.jsonl").write_text("", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["mine", str(corpus_dir), "--config", str(config_path), "--out", str(out_dir)],
    )
    assert result.exit_code == 0, f"mine failed:\n{result.output}"
    assert "store-backed judges active" in result.output

    scope_log = json.loads((out_dir / "scope.json").read_text(encoding="utf-8"))
    documents = scope_log["documents"]
    assert documents, "no documents recorded in scope.json"
    for doc in documents:
        assert doc["basis"] != "judge" or doc["scope_confidence"] != 0.5, (
            f"document {doc['document_id']} was auto-accepted by the stub scope judge: {doc}"
        )
        assert "stub mode" not in doc["scope_rationale"]

    pending_path = judge_dir / "pending.jsonl"
    assert pending_path.exists(), "unstored documents must be queued for scope review"
    pending_kinds = {
        json.loads(line)["kind"] for line in pending_path.read_text().splitlines() if line.strip()
    }
    assert "scope" in pending_kinds, "scope payloads must be queued when no stored verdict exists"


def test_mine_replays_stored_out_of_scope_verdict(tmp_path: Path) -> None:
    """A stored out-of-scope scope verdict must flow through mine as out-of-scope
    — proving the wiring replays real judgments, not just avoids the stub."""
    corpus_dir, config_path, out_dir = _make_corpus(tmp_path)
    judge_dir = out_dir / "judge"
    judge_dir.mkdir(parents=True)
    verdicts_path = judge_dir / "verdicts.jsonl"
    verdicts_path.write_text("", encoding="utf-8")

    runner = CliRunner()

    # First run: empty store — every document is queued to pending as a scope miss.
    first = runner.invoke(
        cli,
        ["mine", str(corpus_dir), "--config", str(config_path), "--out", str(out_dir)],
    )
    assert first.exit_code == 0, f"first mine failed:\n{first.output}"

    pending_path = judge_dir / "pending.jsonl"
    scope_records = [
        json.loads(line)
        for line in pending_path.read_text().splitlines()
        if line.strip() and json.loads(line)["kind"] == "scope"
    ]
    assert scope_records, "expected at least one queued scope payload"

    # Supply a real out-of-scope verdict for every queued scope payload.
    verdict_lines = [
        json.dumps(
            {
                "key": rec["key"],
                "verdict": {
                    "in_scope": False,
                    "scope_rationale": "Off-topic document per stored verdict.",
                    "scope_confidence": 0.91,
                },
            }
        )
        for rec in scope_records
    ]
    verdicts_path.write_text("\n".join(verdict_lines) + "\n", encoding="utf-8")

    # Second run: store now has verdicts for every previously-queued document.
    second = runner.invoke(
        cli,
        ["mine", str(corpus_dir), "--config", str(config_path), "--out", str(out_dir)],
    )
    assert second.exit_code == 0, f"second mine failed:\n{second.output}"

    scope_log = json.loads((out_dir / "scope.json").read_text(encoding="utf-8"))
    queued_doc_ids = set()
    for rec in scope_records:
        # Reconstruct which document each queued payload belonged to.
        queued_doc_ids.add(rec["payload"]["document_id"])

    replayed = [doc for doc in scope_log["documents"] if doc["document_id"] in queued_doc_ids]
    assert replayed, "no replayed documents found in scope.json"
    for doc in replayed:
        assert doc["basis"] == "judge"
        assert doc["in_scope"] is False
        assert doc["scope_rationale"] == "Off-topic document per stored verdict."


# ---------------------------------------------------------------------------
# project command: acceptance tests
# ---------------------------------------------------------------------------


def test_project_after_mine_produces_schema_valid_playbook(tmp_path: Path) -> None:
    """project reads the observation store written by mine and produces a schema-valid playbook."""
    corpus_dir, config_path, out_dir = _make_corpus(tmp_path)
    runner = CliRunner()

    # Step 1: mine
    mine_result = runner.invoke(
        cli,
        [
            "mine",
            str(corpus_dir),
            "--config",
            str(config_path),
            "--out",
            str(out_dir),
        ],
    )
    assert mine_result.exit_code == 0, f"mine failed:\n{mine_result.output}"

    # Step 2: project
    project_result = runner.invoke(
        cli,
        [
            "project",
            str(out_dir),
            "--config",
            str(config_path),
        ],
    )
    assert project_result.exit_code == 0, f"project failed:\n{project_result.output}"

    playbook_path = out_dir / "playbook.opf.json"
    assert playbook_path.exists(), "playbook.opf.json not written by project"

    playbook = json.loads(playbook_path.read_text())
    validation = validate_document(playbook)
    blocking = [str(e) for e in validation.errors if e.blocking]
    assert blocking == [], f"Schema validation errors: {blocking}"


def test_project_exits_zero(tmp_path: Path) -> None:
    """project exits 0 when the store exists."""
    corpus_dir, config_path, out_dir = _make_corpus(tmp_path)
    runner = CliRunner()
    runner.invoke(
        cli,
        ["mine", str(corpus_dir), "--config", str(config_path), "--out", str(out_dir)],
    )
    result = runner.invoke(
        cli,
        ["project", str(out_dir), "--config", str(config_path)],
    )
    assert result.exit_code == 0, f"project failed:\n{result.output}"


def test_project_only_changes_playbook_not_observations(tmp_path: Path) -> None:
    """Re-running project does not rewrite observations.jsonl (no re-mining)."""
    corpus_dir, config_path, out_dir = _make_corpus(tmp_path)
    runner = CliRunner()

    runner.invoke(
        cli,
        ["mine", str(corpus_dir), "--config", str(config_path), "--out", str(out_dir)],
    )
    obs_mtime_before = (out_dir / "observations.jsonl").stat().st_mtime

    runner.invoke(
        cli,
        ["project", str(out_dir), "--config", str(config_path)],
    )
    obs_mtime_after = (out_dir / "observations.jsonl").stat().st_mtime

    assert obs_mtime_after == obs_mtime_before, "project must not rewrite observations.jsonl"


def test_project_missing_store_exits_nonzero(tmp_path: Path) -> None:
    """project against a missing store exits non-zero with a clear error."""
    empty_dir = tmp_path / "empty_out"
    empty_dir.mkdir()

    _, config_path, _ = _make_corpus(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["project", str(empty_dir), "--config", str(config_path)],
    )
    assert result.exit_code != 0
    # Error message should reference the missing store
    assert "observations.jsonl" in result.output or "not found" in result.output.lower()


def test_project_output_matches_compile(tmp_path: Path) -> None:
    """mine then project produces an identical corpus stats section as compile."""
    corpus_dir, config_path, out_dir_compile = _make_corpus(tmp_path)
    out_dir_split = tmp_path / "out_split"
    runner = CliRunner()

    # compile path
    runner.invoke(
        cli,
        ["compile", str(corpus_dir), "--config", str(config_path), "--out", str(out_dir_compile)],
    )

    # mine + project path
    runner.invoke(
        cli,
        ["mine", str(corpus_dir), "--config", str(config_path), "--out", str(out_dir_split)],
    )
    runner.invoke(
        cli,
        ["project", str(out_dir_split), "--config", str(config_path)],
    )

    pb_compile = json.loads((out_dir_compile / "playbook.opf.json").read_text())
    pb_split = json.loads((out_dir_split / "playbook.opf.json").read_text())

    assert pb_compile["corpus"]["stats"] == pb_split["corpus"]["stats"]


# ---------------------------------------------------------------------------
# compile --stop-after intermediates (issue #55)
# ---------------------------------------------------------------------------


def test_compile_stop_after_intermediates_writes_intermediates(tmp_path: Path) -> None:
    """AC: --stop-after intermediates writes scope.json, observations.jsonl,
    corpus_manifest.json and trail/ but NOT playbook.opf.json."""
    corpus_dir, config_path, out_dir = _make_corpus(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "compile",
            str(corpus_dir),
            "--config",
            str(config_path),
            "--out",
            str(out_dir),
            "--stop-after",
            "intermediates",
        ],
    )
    assert result.exit_code == 0, f"compile --stop-after intermediates failed:\n{result.output}"

    # Intermediates must be present.
    assert (out_dir / "scope.json").exists(), "scope.json not written"
    assert (out_dir / "observations.jsonl").exists(), "observations.jsonl not written"
    assert (out_dir / "corpus_manifest.json").exists(), "corpus_manifest.json not written"
    assert (out_dir / "trail" / "deal-001.json").exists(), "trail/deal-001.json not written"
    assert (out_dir / "trail" / "deal-002.json").exists(), "trail/deal-002.json not written"

    # Playbook must NOT be written.
    assert not (out_dir / "playbook.opf.json").exists(), (
        "playbook.opf.json must NOT be written with --stop-after intermediates"
    )


def test_compile_stop_after_intermediates_prints_checkpoint(tmp_path: Path) -> None:
    """AC: --stop-after intermediates prints the checkpoint reached, not the playbook path."""
    corpus_dir, config_path, out_dir = _make_corpus(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "compile",
            str(corpus_dir),
            "--config",
            str(config_path),
            "--out",
            str(out_dir),
            "--stop-after",
            "intermediates",
        ],
    )
    assert result.exit_code == 0, f"compile --stop-after intermediates failed:\n{result.output}"
    assert "stopped after intermediates" in result.output, (
        f"Expected 'stopped after intermediates' in output, got:\n{result.output}"
    )
    assert "playbook.opf.json" not in result.output, (
        "Output must not mention playbook.opf.json when stopped early"
    )


def test_compile_full_run_unaffected_by_stop_after_none(tmp_path: Path) -> None:
    """A full run (no --stop-after) is unchanged: writes playbook.opf.json and prints OK."""
    corpus_dir, config_path, out_dir = _make_corpus(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "compile",
            str(corpus_dir),
            "--config",
            str(config_path),
            "--out",
            str(out_dir),
        ],
    )
    assert result.exit_code == 0, f"full compile failed:\n{result.output}"
    assert (out_dir / "playbook.opf.json").exists(), "playbook.opf.json not written"
    assert "playbook.opf.json" in result.output, (
        "Full run output must contain playbook.opf.json path"
    )


# ---------------------------------------------------------------------------
# mine command: LLM-first segmentation wiring (issue #80)
#
# ``segmentation.llm`` in the config must cause ``mine_cmd`` to wire
# ``use_llm_segmentation``/``use_batch_segmentation``/``segmentation_cache``
# through to ``mine_corpus``.  Proven at the CLI level by monkeypatching
# ``anthropic.Anthropic`` (the same client the real, non-injected
# ``segment_document``/``segment_documents_batch`` construct lazily when no
# client is passed) and asserting it is actually invoked instead of the
# deterministic segmenter. No live API calls, no network, no API key.
# ---------------------------------------------------------------------------

# Two clauses, single document version — mirrors the block-per-paragraph RTF
# fixture convention used in tests/test_pipeline_llm_seg.py (a single-clause
# document is rejected by the scope gate as "too short to evaluate").
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


def _make_llm_corpus(tmp_path: Path, *, segmentation: dict[str, Any]) -> tuple[Path, Path, Path]:
    """Single-document, single-version corpus + config with a segmentation block."""
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
        "segmentation": segmentation,
    }
    config_path = tmp_path / "playbook.config.yaml"
    config_path.write_text(yaml.dump(cfg), encoding="utf-8")

    out_dir = tmp_path / "out"
    return corpus_dir, config_path, out_dir


# Same clause structure AND start/end quotes as _LLM_DEAL_V1 (Indemnification,
# then Governing Law) so the single canned _LLM_SEGMENT_RESPONSE (fixed block
# ids b0-b3, quotes anchored to "1. Indemn"..."programme." and
# "2. Govern"..."California.") grounds against this version too. Only the
# document is duplicated to reach a second version — content is intentionally
# identical since the QA grounding gate checks exact quote text.
_LLM_DEAL_V2 = _LLM_DEAL_V1

#: Canned normalize_trail response: both versions map to the same taxonomy
#: labels the segmenter already assigned (clause_path "1"/"2" — 1-based
#: dotted numbering per playbook_engine.clause_tree), so normalization is a
#: no-op beyond proving the call happened.
_NORMALIZE_TRAIL_RESPONSE = json.dumps(
    {
        "versions": [
            {
                "version_id": "v1",
                "clauses": [
                    {"clause_path": "1", "taxonomy_id": "indemnification"},
                    {"clause_path": "2", "taxonomy_id": "governing_law"},
                ],
            },
            {
                "version_id": "v2",
                "clauses": [
                    {"clause_path": "1", "taxonomy_id": "indemnification"},
                    {"clause_path": "2", "taxonomy_id": "governing_law"},
                ],
            },
        ],
        "boundary_flags": [],
    }
)


def _make_llm_corpus_multi_version(
    tmp_path: Path, *, segmentation: dict[str, Any]
) -> tuple[Path, Path, Path]:
    """Single-document, two-version corpus + config with a segmentation block.

    Two versions of the same agreement are required to reach the
    ``normalize_trail_fn`` call: mine_corpus only invokes it when
    ``len(version_trees) > 1`` (see pipeline._compute_doc_result).
    """
    corpus_dir = tmp_path / "corpus"
    deal_dir = corpus_dir / "deal-001"
    deal_dir.mkdir(parents=True)
    _write_rtf(deal_dir / "v1.rtf", _LLM_DEAL_V1)
    _write_rtf(deal_dir / "v2.rtf", _LLM_DEAL_V2)

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
        "segmentation": segmentation,
    }
    config_path = tmp_path / "playbook.config.yaml"
    config_path.write_text(yaml.dump(cfg), encoding="utf-8")

    out_dir = tmp_path / "out"
    return corpus_dir, config_path, out_dir


class _RecordingMessages:
    """Fake ``client.messages`` — records calls, returns a canned response.

    ``normalize_trail_response_text``, when given, is returned instead of
    ``response_text`` for calls whose structured-output schema is the
    normalize_trail shape (top-level ``versions`` key) rather than the
    segmentation shape (top-level ``nodes`` key) — the two calls share this
    same fake ``messages.create`` but expect differently-shaped JSON back.
    """

    def __init__(
        self, response_text: str, normalize_trail_response_text: str | None = None
    ) -> None:
        self.response_text = response_text
        self.normalize_trail_response_text = normalize_trail_response_text
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        text = self.response_text
        if self.normalize_trail_response_text is not None:
            schema = kwargs.get("output_config", {}).get("format", {}).get("schema", {})
            if "versions" in schema.get("properties", {}):
                text = self.normalize_trail_response_text
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)])


class _RecordingBatches:
    """Fake ``client.messages.batches`` — ends immediately (no polling/sleep),
    returns the same canned segmentation response for every submitted item."""

    def __init__(self, response_text: str) -> None:
        self.response_text = response_text
        self.create_calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.create_calls.append(kwargs)
        return SimpleNamespace(id="batch_cli_test", processing_status="ended")

    def retrieve(self, batch_id: str) -> Any:  # pragma: no cover - not hit; ends immediately
        return SimpleNamespace(id=batch_id, processing_status="ended")

    def results(self, batch_id: str) -> list[Any]:
        del batch_id
        out = []
        for call in self.create_calls:
            for req in call["requests"]:
                out.append(
                    SimpleNamespace(
                        custom_id=req["custom_id"],
                        result=SimpleNamespace(
                            type="succeeded",
                            message=SimpleNamespace(
                                content=[SimpleNamespace(type="text", text=self.response_text)]
                            ),
                        ),
                    )
                )
        return out


class _RecordingAnthropicClient:
    """Fake ``anthropic.Anthropic()`` instance — tracks whether it was constructed."""

    instances: list[_RecordingAnthropicClient] = []
    #: Set by tests that also exercise normalize_trail (a second, differently-
    #: shaped structured-output call through the same fake client) before
    #: invoking the CLI; None preserves the plain segmentation-only behavior.
    normalize_trail_response_text: str | None = None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.messages = _RecordingMessages(
            _LLM_SEGMENT_RESPONSE,
            normalize_trail_response_text=_RecordingAnthropicClient.normalize_trail_response_text,
        )
        self.messages.batches = _RecordingBatches(_LLM_SEGMENT_RESPONSE)  # type: ignore[attr-defined]
        _RecordingAnthropicClient.instances.append(self)


@pytest.fixture
def _fake_anthropic(monkeypatch: pytest.MonkeyPatch) -> type[_RecordingAnthropicClient]:
    """Monkeypatch ``anthropic.Anthropic`` so the real (non-injected) client
    construction inside ``segment_document``/``segment_documents_batch``
    never makes a live call.

    Also sets a dummy ``ANTHROPIC_API_KEY`` (issue #131's credential preflight
    now runs before ``_llm_segmentation_kwargs`` builds these kwargs at all —
    every test that exercises the LLM path needs this set or ``mine``/
    ``compile``/``judge`` would exit 1 before ever reaching the mocked
    client)."""
    import anthropic

    _RecordingAnthropicClient.instances = []
    _RecordingAnthropicClient.normalize_trail_response_text = None
    monkeypatch.setattr(anthropic, "Anthropic", _RecordingAnthropicClient)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-dummy-key")
    return _RecordingAnthropicClient


# ---------------------------------------------------------------------------
# Credential preflight (issue #131): segmentation.llm=true with no
# ANTHROPIC_API_KEY must fail fast, before extraction, with a plain-language
# message -- not an unhandled traceback surfacing only after docling has
# already ground through the whole corpus.
# ---------------------------------------------------------------------------


def test_mine_preflight_missing_api_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``segmentation.llm: true`` + no ``ANTHROPIC_API_KEY`` -> ``mine`` exits 1 with
    a friendly message, and never even reaches extraction (no anthropic client
    construction, no corpus_manifest.json/observations.jsonl written)."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    corpus_dir, config_path, out_dir = _make_llm_corpus(tmp_path, segmentation={"llm": True})
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["mine", str(corpus_dir), "--config", str(config_path), "--out", str(out_dir)],
    )
    assert result.exit_code == 1
    assert "Traceback" not in result.output, (
        f"missing API key must produce a friendly message, not a traceback:\n{result.output}"
    )
    assert "ANTHROPIC_API_KEY" in result.output
    assert not (out_dir / "observations.jsonl").exists(), (
        "the preflight must fail before mine_corpus ever runs, so no store is written"
    )


def test_compile_preflight_missing_api_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Same preflight, wired the same way, for ``compile``."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    corpus_dir, config_path, out_dir = _make_llm_corpus(tmp_path, segmentation={"llm": True})
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["compile", str(corpus_dir), "--config", str(config_path), "--out", str(out_dir)],
    )
    assert result.exit_code == 1
    assert "Traceback" not in result.output
    assert "ANTHROPIC_API_KEY" in result.output


def test_judge_preflight_missing_api_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Same preflight, wired the same way, for ``judge``."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    corpus_dir, config_path, out_dir = _make_llm_corpus(tmp_path, segmentation={"llm": True})
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["judge", str(corpus_dir), "--config", str(config_path), "--out", str(out_dir)],
    )
    assert result.exit_code == 1
    assert "Traceback" not in result.output
    assert "ANTHROPIC_API_KEY" in result.output


def test_mine_without_segmentation_llm_never_requires_api_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: the preflight is config-gated on ``segmentation.llm`` — a
    corpus with no ``segmentation:`` block (the deterministic default) must
    keep working with no ANTHROPIC_API_KEY set at all."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    corpus_dir, config_path, out_dir = _make_corpus(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["mine", str(corpus_dir), "--config", str(config_path), "--out", str(out_dir)],
    )
    assert result.exit_code == 0, f"mine failed:\n{result.output}"


def test_mine_without_segmentation_block_never_touches_anthropic(
    tmp_path: Path, _fake_anthropic: type[_RecordingAnthropicClient]
) -> None:
    """Regression: no ``segmentation:`` block -> deterministic path only, no client built."""
    corpus_dir, config_path, out_dir = _make_corpus(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["mine", str(corpus_dir), "--config", str(config_path), "--out", str(out_dir)],
    )
    assert result.exit_code == 0, f"mine failed:\n{result.output}"
    assert _fake_anthropic.instances == [], (
        "anthropic.Anthropic must not be constructed when segmentation.llm is absent"
    )


def test_mine_segmentation_llm_true_invokes_anthropic_client(
    tmp_path: Path, _fake_anthropic: type[_RecordingAnthropicClient]
) -> None:
    """``segmentation: {llm: true}`` wires the LLM path through mine_corpus, which
    constructs and calls the (mocked) anthropic.Anthropic client rather than
    running the deterministic segmenter."""
    corpus_dir, config_path, out_dir = _make_llm_corpus(tmp_path, segmentation={"llm": True})
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["mine", str(corpus_dir), "--config", str(config_path), "--out", str(out_dir)],
    )
    assert result.exit_code == 0, f"mine failed:\n{result.output}"
    assert "segmentation: llm" in result.output

    assert len(_fake_anthropic.instances) == 1, (
        "anthropic.Anthropic must be constructed exactly once for the LLM segmentation path"
    )
    client = _fake_anthropic.instances[0]
    assert len(client.messages.calls) == 1, "the mocked client must have been invoked"

    obs_path = out_dir / "observations.jsonl"
    assert obs_path.exists()
    raw_obs = [json.loads(line) for line in obs_path.read_text().splitlines() if line.strip()]
    assert raw_obs, "mine must write at least one observation"
    assert any(o["taxonomy_id"] == "indemnification" for o in raw_obs), (
        "observation must carry the taxonomy_id from the mocked LLM response"
    )


def test_mine_segmentation_model_override_reaches_anthropic_call(
    tmp_path: Path, _fake_anthropic: type[_RecordingAnthropicClient]
) -> None:
    """``segmentation.model`` is config data, not a hardcoded literal (issue #131):
    a non-default model id set in the config must be the actual ``model=``
    the mocked Anthropic request carries."""
    corpus_dir, config_path, out_dir = _make_llm_corpus(
        tmp_path, segmentation={"llm": True, "model": "claude-opus-4-7-custom"}
    )
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["mine", str(corpus_dir), "--config", str(config_path), "--out", str(out_dir)],
    )
    assert result.exit_code == 0, f"mine failed:\n{result.output}"

    client = _fake_anthropic.instances[0]
    assert client.messages.calls[0]["model"] == "claude-opus-4-7-custom"


def test_mine_records_extractor_per_version(
    tmp_path: Path, _fake_anthropic: type[_RecordingAnthropicClient]
) -> None:
    """corpus_manifest.json records which extractor (docling vs legacy) ran
    per version on the LLM-segmentation path, and ``mine`` echoes a summary —
    mirroring the ``segmentation: ...`` echo (issue #129: this was previously
    only visible via a suppressed ``logging.info`` line in extraction.py)."""
    corpus_dir, config_path, out_dir = _make_llm_corpus(tmp_path, segmentation={"llm": True})
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["mine", str(corpus_dir), "--config", str(config_path), "--out", str(out_dir)],
    )
    assert result.exit_code == 0, f"mine failed:\n{result.output}"

    # No docling binary in this test environment -> legacy adapter ran.
    assert "extraction: legacy=1" in result.output

    manifest = json.loads((out_dir / "corpus_manifest.json").read_text(encoding="utf-8"))
    doc_entry = next(d for d in manifest if d["document_id"] == "deal-001")
    ingest_by_version = {v["version"]: v for v in doc_entry["version_ingest"]}
    assert ingest_by_version["v1"]["extractor"] == "legacy"


def test_mine_segmentation_llm_batch_cache_wires_all_three(
    tmp_path: Path, _fake_anthropic: type[_RecordingAnthropicClient]
) -> None:
    """``segmentation: {llm: true, batch: true, cache: true}`` activates all three
    and writes the on-disk segmentation cache."""
    corpus_dir, config_path, out_dir = _make_llm_corpus(
        tmp_path, segmentation={"llm": True, "batch": True, "cache": True}
    )
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["mine", str(corpus_dir), "--config", str(config_path), "--out", str(out_dir)],
    )
    assert result.exit_code == 0, f"mine failed:\n{result.output}"
    assert "segmentation: llm+batch+cache" in result.output

    assert len(_fake_anthropic.instances) >= 1, (
        "anthropic.Anthropic must be constructed for the batch segmentation path"
    )
    assert (out_dir / "segmentation_cache.jsonl").exists(), (
        "segmentation_cache=True must write out/segmentation_cache.jsonl"
    )


def test_mine_segmentation_llm_normalize_trail_wires_through(
    tmp_path: Path, _fake_anthropic: type[_RecordingAnthropicClient]
) -> None:
    """``segmentation: {llm: true, normalize_trail: true}`` wires
    ``normalize_trail_fn`` through mine_corpus the same way the llm/batch/cache
    flags do: a two-version corpus is required because mine_corpus only calls
    normalize_trail_fn when more than one version was segmented (see
    pipeline._compute_doc_result's ``len(version_trees) > 1`` guard) — a
    single-version corpus would leave the wiring untested even though
    ``mine_kwargs["normalize_trail_fn"]`` was set.
    """
    _fake_anthropic.normalize_trail_response_text = _NORMALIZE_TRAIL_RESPONSE
    corpus_dir, config_path, out_dir = _make_llm_corpus_multi_version(
        tmp_path, segmentation={"llm": True, "normalize_trail": True}
    )
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["mine", str(corpus_dir), "--config", str(config_path), "--out", str(out_dir)],
    )
    assert result.exit_code == 0, f"mine failed:\n{result.output}"
    assert "segmentation: llm+normalize_trail" in result.output

    assert len(_fake_anthropic.instances) >= 1, (
        "anthropic.Anthropic must be constructed for the normalize_trail path"
    )
    all_calls = [call for inst in _fake_anthropic.instances for call in inst.messages.calls]
    normalize_calls = [
        call
        for call in all_calls
        if "versions"
        in call.get("output_config", {}).get("format", {}).get("schema", {}).get("properties", {})
    ]
    assert normalize_calls, (
        "normalize_trail must have called the mocked client with the "
        "normalize_trail structured-output schema (top-level 'versions' key)"
    )

    obs_path = out_dir / "observations.jsonl"
    assert obs_path.exists()
    raw_obs = [json.loads(line) for line in obs_path.read_text().splitlines() if line.strip()]
    assert raw_obs, "mine must write at least one observation"
    assert any(o["taxonomy_id"] == "indemnification" for o in raw_obs), (
        "observation must carry the taxonomy_id from the normalized labels"
    )


def test_mine_segmentation_llm_false_is_unaffected(tmp_path: Path) -> None:
    """An explicit ``segmentation: {llm: false}`` behaves exactly like no block at all."""
    corpus_dir, config_path, out_dir = _make_corpus(tmp_path)
    with config_path.open() as fh:
        cfg = yaml.safe_load(fh)
    cfg["segmentation"] = {"llm": False}
    config_path.write_text(yaml.dump(cfg), encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["mine", str(corpus_dir), "--config", str(config_path), "--out", str(out_dir)],
    )
    assert result.exit_code == 0, f"mine failed:\n{result.output}"
    assert "segmentation:" not in result.output


# ---------------------------------------------------------------------------
# ``compile`` and ``judge`` must segment the SAME way ``mine`` does when
# ``segmentation.llm`` is on. If they fell back to the deterministic segmenter,
# a compiled playbook (or the judge drain loop's verdict keys) would never line
# up with the LLM-segmented observation store and the loop could not converge.
# Proven the same way as the mine tests: the (mocked) anthropic client is
# constructed only on the LLM path.
# ---------------------------------------------------------------------------


def test_compile_segmentation_llm_true_invokes_anthropic_client(
    tmp_path: Path, _fake_anthropic: type[_RecordingAnthropicClient]
) -> None:
    """``segmentation: {llm: true}`` must drive ``compile`` through the LLM
    segmenter, not the deterministic path (convergence fix)."""
    corpus_dir, config_path, out_dir = _make_llm_corpus(tmp_path, segmentation={"llm": True})
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["compile", str(corpus_dir), "--config", str(config_path), "--out", str(out_dir)],
    )
    assert result.exit_code == 0, f"compile failed:\n{result.output}"
    assert "segmentation: llm" in result.output
    assert len(_fake_anthropic.instances) >= 1, (
        "compile must construct anthropic.Anthropic on the segmentation.llm path"
    )


def test_compile_without_segmentation_block_never_touches_anthropic(
    tmp_path: Path, _fake_anthropic: type[_RecordingAnthropicClient]
) -> None:
    """Regression: no ``segmentation:`` block -> compile stays deterministic."""
    corpus_dir, config_path, out_dir = _make_corpus(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["compile", str(corpus_dir), "--config", str(config_path), "--out", str(out_dir)],
    )
    assert result.exit_code == 0, f"compile failed:\n{result.output}"
    assert _fake_anthropic.instances == [], (
        "anthropic.Anthropic must not be constructed when segmentation.llm is absent"
    )


def test_judge_segmentation_llm_true_invokes_anthropic_client(
    tmp_path: Path, _fake_anthropic: type[_RecordingAnthropicClient]
) -> None:
    """``segmentation: {llm: true}`` must drive ``judge``'s mine pass through the
    LLM segmenter so its verdict keys match an LLM-segmented store (the
    convergence blocker this fixes)."""
    corpus_dir, config_path, out_dir = _make_llm_corpus(tmp_path, segmentation={"llm": True})
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["judge", str(corpus_dir), "--config", str(config_path), "--out", str(out_dir)],
    )
    assert result.exit_code == 0, f"judge failed:\n{result.output}"
    assert "segmentation: llm" in result.output
    assert len(_fake_anthropic.instances) >= 1, (
        "judge must construct anthropic.Anthropic on the segmentation.llm path"
    )
