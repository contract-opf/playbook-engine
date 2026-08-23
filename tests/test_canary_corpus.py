"""Canary corpus + CI gate — the incident this test exists to turn red.

2026-08-22, production corpus re-derivation: ``docling`` had silently
disappeared from the host venv, so extraction fell back to the ``legacy``
adapter. The extraction cache keys on ``extractor_env`` (issue #77), so every
version missed it and was re-extracted by a *different* adapter; the
segmentation cache keys on the resulting **canonical text**, so every
document then missed that too and quarantined. 43 of 44 documents came back
``AgentSegmentationPending`` and observations fell from ~2,400 to 66.

The engine reported a *segmentation* problem. The fault was two layers below,
in the extractor environment. Nothing in CI caught it, because nothing in CI
ever (a) stated which extractor a run expects, or (b) replayed a warm out-dir
and checked that the caches actually carried it.

This module is that check, over the tiny synthetic corpus at
``examples/canary/`` (four DOCX documents across two negotiations, two of them
tracked-changes redlines). It asserts, in order of how far below the symptom
each layer sits:

1. **Extractor identity** (:func:`test_canary_extractor_env_matches_expectation`)
   — the extractor environment the run resolves to is the one
   ``examples/canary/expected.json`` says it should be, checked BEFORE any
   pipeline work. This is the assertion that would have named the real fault.
2. **Warm replay** (:func:`test_canary_warm_replay_does_no_extraction`) — a
   second run against the same out-dir performs **zero** extraction and
   quarantines **zero** documents, with the L1-L4 stage cache deleted first
   so the extraction and segmentation caches are the only things that can
   carry it. Those are exactly the two caches that missed in the incident.
3. **Derivation output** (:func:`test_canary_cold_run_matches_expected`) —
   observation counts, round-move counts and attribution, and compiled
   playbook clause count all equal committed values.

Plus three mutation tests proving each of those bites rather than passing
vacuously, and :func:`test_canary_reproduces_the_incident`, which replays the
incident itself against a warm out-dir and asserts both that it quarantines
everything AND that the extractor check catches it first.

Hermetic and keyless, like ``make smoke-nda``:
  - No network, no ``ANTHROPIC_API_KEY`` (``segmentation.agent: true`` is the
    store-backed, key-free segmentation loop; the partition itself is replayed
    from the committed ``examples/canary/segment-verdicts.jsonl``).
  - No ``docling`` and no ``pandoc``: the corpus is DOCX only and the config
    pins ``extraction.extractor: legacy``, so extraction is pure python-docx.
  - All output under pytest's ``tmp_path``.

Marked ``@pytest.mark.smoke``; run via ``make smoke-canary`` or
``pytest tests/test_canary_corpus.py -q -m smoke``.

Regenerating the fixtures (all three, in order) — see examples/canary/README.md::

    python examples/canary/build_corpus.py
    python examples/canary/build_verdicts.py
    python examples/canary/build_expected.py
"""

from __future__ import annotations

import collections
import hashlib
import json
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
import yaml
from click.testing import CliRunner
from click.testing import Result as CliResult

from playbook_engine import extraction as extraction_mod
from playbook_engine import pipeline as pipeline_mod
from playbook_engine.cli import cli
from playbook_engine.config import load_config
from playbook_engine.extraction import _resolve_extractor_env

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CANARY_DIR = _REPO_ROOT / "examples" / "canary"
_CORPUS_DIR = _CANARY_DIR / "corpus"
_CONFIG_PATH = _CANARY_DIR / "config.yaml"
_VERDICTS_PATH = _CANARY_DIR / "segment-verdicts.jsonl"
_EXPECTED_PATH = _CANARY_DIR / "expected.json"


def load_expected() -> dict[str, Any]:
    """The committed canary expectations (``examples/canary/expected.json``)."""
    return json.loads(_EXPECTED_PATH.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Extraction spy — "did this run do any extraction work at all?"
# ---------------------------------------------------------------------------


@dataclass
class ExtractionSpy:
    """Counts real extractor invocations inside :func:`extract_blocks`.

    Wraps the two functions that do the actual per-file adapter work —
    ``_extract_legacy_lines`` (python-docx / pdfplumber / pandoc) and
    ``_extract_docling_lines`` (the docling subprocess). Anything the
    extraction cache serves never reaches either, so ``count == 0`` is a
    direct, unambiguous statement that a run performed zero extraction.

    Deliberately NOT counted: ``extraction.extract_tracked_changes``, the
    DOCX tracked-changes side channel. It is a cheap second python-docx parse
    that is not cached by design and runs on every pass; folding it in would
    make "zero extraction" unachievable and the assertion meaningless.
    """

    count: int = 0
    paths: list[str] = field(default_factory=list)


@contextmanager
def extraction_spy() -> Iterator[ExtractionSpy]:
    """Count extractor invocations for the duration of the block.

    Restores exactly the two attributes it replaced rather than going through
    ``monkeypatch``: a spy is scoped to one run inside a test that has usually
    already made other patches (a deleted ``ANTHROPIC_API_KEY``, a swapped
    ``detect_extractor``), and ``monkeypatch.undo()`` would tear all of those
    down too on the way out.
    """
    spy = ExtractionSpy()
    real_legacy = extraction_mod._extract_legacy_lines
    real_docling = extraction_mod._extract_docling_lines

    def _legacy(path: Path, suffix: str) -> Any:
        spy.count += 1
        spy.paths.append(path.name)
        return real_legacy(path, suffix)

    def _docling(path: Path) -> Any:
        spy.count += 1
        spy.paths.append(path.name)
        return real_docling(path)

    extraction_mod._extract_legacy_lines = _legacy  # type: ignore[assignment]
    extraction_mod._extract_docling_lines = _docling  # type: ignore[assignment]
    try:
        yield spy
    finally:
        extraction_mod._extract_legacy_lines = real_legacy  # type: ignore[assignment]
        extraction_mod._extract_docling_lines = real_docling  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Running the canary
# ---------------------------------------------------------------------------


def _invoke(args: list[str]) -> CliResult:
    return CliRunner().invoke(cli, args)


def _check(result: CliResult, label: str) -> CliResult:
    assert result.exit_code == 0, f"{label} failed (exit {result.exit_code}):\n{result.output}"
    return result


def run_cold(out_dir: Path, *, config_path: Path = _CONFIG_PATH) -> dict[str, CliResult]:
    """First, cold pass: segment -> segment-apply -> mine -> project -> validate.

    ``segment`` is what actually extracts (it walks every version and fills
    ``extraction_cache.jsonl``); ``segment-apply`` loads the committed
    partition into ``segment/cache.jsonl``; ``mine`` then replays both caches.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "segment": _check(
            _invoke(
                ["segment", str(_CORPUS_DIR), "--config", str(config_path), "--out", str(out_dir)]
            ),
            "segment",
        ),
        "segment-apply": _check(
            _invoke(["segment-apply", str(out_dir), "--verdicts", str(_VERDICTS_PATH)]),
            "segment-apply",
        ),
        "mine": _check(
            _invoke(
                ["mine", str(_CORPUS_DIR), "--config", str(config_path), "--out", str(out_dir)]
            ),
            "mine",
        ),
        "project": _check(
            _invoke(["project", str(out_dir), "--config", str(config_path)]), "project"
        ),
    }
    results["validate"] = _check(
        _invoke(["validate", str(out_dir / "playbook.opf.json")]), "validate"
    )
    return results


def run_warm(out_dir: Path, *, config_path: Path = _CONFIG_PATH) -> dict[str, CliResult]:
    """Replay against an already-populated *out_dir*.

    The L1-L4 stage cache (``out_dir/.cache`` — ``ArtifactStore`` plus
    ``JudgmentCache``) is deleted first. Without that, a second ``mine`` would
    short-circuit on stage-cache hits before extraction was ever reached, and
    "zero extraction" would be true for a reason that has nothing to do with
    the caches this test is about. Deleting it leaves ``extraction_cache.jsonl``
    and ``segment/cache.jsonl`` — the two caches that missed in the incident —
    as the only things that can carry the replay.

    No ``segment-apply``: the segmentation cache must already be warm.
    """
    shutil.rmtree(out_dir / ".cache", ignore_errors=True)
    return {
        "segment": _check(
            _invoke(
                ["segment", str(_CORPUS_DIR), "--config", str(config_path), "--out", str(out_dir)]
            ),
            "warm segment",
        ),
        "mine": _check(
            _invoke(
                ["mine", str(_CORPUS_DIR), "--config", str(config_path), "--out", str(out_dir)]
            ),
            "warm mine",
        ),
        "project": _check(
            _invoke(["project", str(out_dir), "--config", str(config_path)]), "warm project"
        ),
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def measure(out_dir: Path) -> dict[str, Any]:
    """Reduce a completed run to the shape of ``expected.json``.

    Kept as one dict compared with a single ``==`` so a drift anywhere in the
    derivation shows up in one pytest diff rather than as whichever assertion
    happened to be written first.
    """
    manifest = json.loads((out_dir / "corpus_manifest.json").read_text(encoding="utf-8"))
    observations = _read_jsonl(out_dir / "observations.jsonl")
    moves = _read_jsonl(out_dir / "round_moves.jsonl")
    quarantine = json.loads((out_dir / "quarantine.json").read_text(encoding="utf-8"))
    playbook = json.loads((out_dir / "playbook.opf.json").read_text(encoding="utf-8"))

    ingest_rows = [row for doc in manifest for row in doc.get("version_ingest", [])]
    return {
        "extractor_env": sorted({str(row["extractor"]) for row in ingest_rows}),
        "extractor_reason": sorted({str(row["reason"]) for row in ingest_rows}),
        "documents": len(manifest),
        "versions": len(ingest_rows),
        "versions_failed": sum(1 for row in ingest_rows if row["status"] != "ok"),
        "quarantined": len(quarantine),
        "observations": len(observations),
        "observations_by_document": dict(
            sorted(collections.Counter(o["citation"]["document_id"] for o in observations).items())
        ),
        "round_moves": len(moves),
        "round_moves_by_moved_by": dict(
            sorted(collections.Counter(m["moved_by"] for m in moves).items())
        ),
        "playbook_clauses": len(playbook["evidence"]["clauses"]),
        "corpus_sha256": corpus_sha256(),
    }


def corpus_sha256() -> dict[str, str]:
    """sha256 of every committed corpus file, by corpus-relative path.

    The extraction cache keys on exactly this (``file_sha256`` in
    ``extraction._extraction_cache_payload``), so pinning it here makes an
    accidentally-regenerated corpus fail as a corpus change rather than as a
    mysterious count drift three assertions later.
    """
    return {
        str(path.relative_to(_CORPUS_DIR)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(_CORPUS_DIR.rglob("*.docx"))
    }


# ---------------------------------------------------------------------------
# 1. Extractor identity — the layer the incident actually broke at
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_canary_extractor_env_matches_expectation() -> None:
    """The resolved extractor environment is the one the canary expects.

    Pure resolution, no file I/O beyond the config: this is the pre-flight
    check the incident needed. ``extraction.extractor: legacy`` in the canary
    config makes the answer independent of whether docling happens to be
    installed on this host — so a mismatch here means the *engine's*
    resolution changed, not that someone's laptop differs.
    """
    expected = load_expected()
    cfg = load_config(_CONFIG_PATH)

    resolved = {
        str(path.relative_to(_CORPUS_DIR)): _resolve_extractor_env(cfg.extraction.extractor, path)
        for path in sorted(_CORPUS_DIR.rglob("*.docx"))
    }
    assert resolved, "canary corpus is empty — did examples/canary/corpus/ get lost?"
    assert set(resolved.values()) == set(expected["extractor_env"]), (
        f"resolved extractor environment {sorted(set(resolved.values()))} != expected "
        f"{expected['extractor_env']}. This is an EXTRACTION-layer fault, not a "
        "segmentation one: a run under a different extractor produces different "
        "canonical text, misses the extraction AND segmentation caches, and "
        "quarantines everything downstream (see this module's docstring)."
    )


@pytest.mark.smoke
def test_declared_docling_fails_loudly_when_docling_is_absent(tmp_path: Path) -> None:
    """Mutation gate for the check above: a config that REQUIRES docling on a
    host without it must fail immediately, by name, before any file is read.

    This is the outright-catch half of "the resolved extractor environment is
    what the run expects" (issue #80). Skipped where docling really is
    installed — there the declaration is satisfiable and there is nothing to
    catch.
    """
    if shutil.which("docling") is not None:
        pytest.skip("docling is installed on this host — nothing to catch")

    cfg_raw = yaml.safe_load(_CONFIG_PATH.read_text(encoding="utf-8"))
    cfg_raw["extraction"]["extractor"] = "docling"
    cfg_raw["taxonomy"] = str(load_config(_CONFIG_PATH).taxonomy_path)
    bad_config = tmp_path / "config.docling.yaml"
    bad_config.write_text(yaml.dump(cfg_raw), encoding="utf-8")

    result = _invoke(
        [
            "segment",
            str(_CORPUS_DIR),
            "--config",
            str(bad_config),
            "--out",
            str(tmp_path / "out"),
        ]
    )
    assert result.exit_code != 0, f"declared docling silently downgraded:\n{result.output}"
    assert "docling" in result.output.lower()
    assert "not found on path" in result.output.lower()


# ---------------------------------------------------------------------------
# 2 + 3. Cold run, committed counts, and the warm replay
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_canary_cold_run_matches_expected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A cold run quarantines nothing and reproduces the committed counts.

    Also asserts the extraction spy fires (``count > 0``) — without that, the
    ``count == 0`` in the warm test could pass because the spy was never wired
    to anything.
    """
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    out_dir = tmp_path / "out"

    with extraction_spy() as spy:
        run_cold(out_dir)

    expected = load_expected()
    assert spy.count == expected["versions"], (
        f"cold run extracted {spy.count} file(s), expected one per version "
        f"({expected['versions']}) — extracted: {spy.paths}"
    )
    assert measure(out_dir) == expected


@pytest.mark.smoke
def test_canary_warm_replay_does_no_extraction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Warm replay: zero extraction, zero quarantine, identical output.

    The incident's signature was a warm out-dir that stopped being warm. Here
    the L1-L4 stage cache is deleted before the replay, so the only things
    that can carry it are ``extraction_cache.jsonl`` and
    ``segment/cache.jsonl`` — and the assertion is that they do, completely.
    """
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    out_dir = tmp_path / "out"
    run_cold(out_dir)
    expected = load_expected()

    with extraction_spy() as spy:
        results = run_warm(out_dir)

    assert spy.count == 0, (
        f"warm replay re-extracted {spy.count} file(s) ({spy.paths}) — the "
        "extraction cache did not carry the replay. Check extractor_env in the "
        "cache key (extraction._extraction_cache_payload, issue #77)."
    )
    assert "Segmentation pending: 0" in results["segment"].output, (
        "warm `segment` still had documents to queue — the segmentation cache "
        f"missed:\n{results['segment'].output}"
    )
    assert json.loads((out_dir / "quarantine.json").read_text(encoding="utf-8")) == []
    assert measure(out_dir) == expected


@pytest.mark.smoke
def test_warm_replay_check_bites_when_the_extraction_cache_is_lost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mutation gate for the warm-replay assertion.

    Delete ``extraction_cache.jsonl`` and the same replay must re-extract every
    version — proving ``spy.count == 0`` above is a real observation about the
    cache and not an artifact of a replay that never reaches extraction at all.
    """
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    out_dir = tmp_path / "out"
    run_cold(out_dir)

    (out_dir / "extraction_cache.jsonl").unlink()
    with extraction_spy() as spy:
        run_warm(out_dir)

    assert spy.count == load_expected()["versions"], (
        f"expected a cold re-extraction of every version, got {spy.count} "
        "— the warm-replay assertion above would pass vacuously"
    )


# ---------------------------------------------------------------------------
# The incident itself
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_canary_reproduces_the_incident(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Replay 2026-08-22 against a warm out-dir, then catch it at the right layer.

    Simulated exactly as it happened, minus the need for a real docling
    install: the out-dir is warmed under ``legacy``, then the *environment*
    changes underneath it (``detect_extractor`` starts answering ``"docling"``
    and a stand-in docling adapter returns a different block stream, as a
    genuinely different extractor would). The config is switched to ``auto``,
    which is what a real host runs and what lets the environment change be
    felt at all.

    The cascade then reproduces itself:
      extractor env changes -> extraction cache key changes -> miss ->
      re-extraction under a different adapter -> different canonical text ->
      segmentation cache miss -> StoreBackedSegmentFn queues and raises
      AgentSegmentationPending -> every document quarantines.

    The test asserts BOTH halves of the point: that ``mine`` reports the
    symptom two layers above the fault (mass quarantine, collapsed
    observations, and not one word about the extractor), and that the
    canary's own pre-flight extractor check names the actual fault first.
    """
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    out_dir = tmp_path / "out"
    run_cold(out_dir)
    assert json.loads((out_dir / "quarantine.json").read_text(encoding="utf-8")) == []
    baseline_observations = len(_read_jsonl(out_dir / "observations.jsonl"))
    assert baseline_observations == load_expected()["observations"]

    # `auto` is what a real host runs: it asks the environment, every time.
    cfg_raw = yaml.safe_load(_CONFIG_PATH.read_text(encoding="utf-8"))
    cfg_raw["extraction"]["extractor"] = "auto"
    cfg_raw["taxonomy"] = str(load_config(_CONFIG_PATH).taxonomy_path)
    auto_config = tmp_path / "config.auto.yaml"
    auto_config.write_text(yaml.dump(cfg_raw), encoding="utf-8")

    # --- the environment changes underneath the warm out-dir ---------------
    real_legacy = extraction_mod._extract_legacy_lines

    def _fake_docling_lines(path: Path) -> list[tuple[str, int]]:
        # A different adapter, so: a different block stream, and therefore a
        # different canonical text for the very same bytes.
        return [("## " + text, page) for text, page in real_legacy(path, path.suffix.lower())]

    monkeypatch.setattr(extraction_mod, "detect_extractor", lambda _path: "docling")
    monkeypatch.setattr(pipeline_mod, "detect_extractor", lambda _path: "docling")
    monkeypatch.setattr(extraction_mod, "_extract_docling_lines", _fake_docling_lines)

    # --- what the operator saw: a segmentation problem ---------------------
    shutil.rmtree(out_dir / ".cache", ignore_errors=True)
    mine_result = _invoke(
        ["mine", str(_CORPUS_DIR), "--config", str(auto_config), "--out", str(out_dir)]
    )
    quarantine = json.loads((out_dir / "quarantine.json").read_text(encoding="utf-8"))
    assert len(quarantine) == load_expected()["documents"], (
        f"expected every document to quarantine, got {quarantine}:\n{mine_result.output}"
    )
    assert all("segment" in str(entry.get("reason", "")).lower() for entry in quarantine), (
        "the reported reason should be a SEGMENTATION one — that is the whole "
        f"problem: {quarantine}"
    )
    assert len(_read_jsonl(out_dir / "observations.jsonl")) < baseline_observations

    # --- what the canary sees: an extraction problem, before any of that ---
    cfg = load_config(auto_config)
    resolved = {_resolve_extractor_env(cfg.extraction.extractor, p) for p in _CORPUS_DIR.rglob("*")}
    assert resolved == {"docling"}
    assert resolved != set(load_expected()["extractor_env"]), (
        "the pre-flight extractor check must disagree with the committed "
        "expectation here — otherwise it would not have caught the incident"
    )
