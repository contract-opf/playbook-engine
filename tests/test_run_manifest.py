"""Tests for the run-level provenance manifest and its preflight (issue #121).

The incident this closes: docling vanished from the host venv, ``extraction.
extractor: auto`` silently resolved to ``legacy``, every version missed the
``extractor_env``-keyed extraction cache, every version then missed the
canonical-text-keyed segmentation cache, and 43 of 44 documents landed in
``AgentSegmentationPending`` quarantine with observations down from ~2,400 to
66. The visible error was two layers below the real fault.

What these tests pin down, in the order that matters:

  1. SILENCE on the normal path — no prior manifest, or a matching one, must
     produce ZERO output. The project owner's constraint is that no warning
     should be needed; a preflight that chatters trains people to skim.
  2. The docling-disappeared case is caught, explained in plain English, and
     STOPS the run before any work is done or discarded.
  3. The copy-pasteable block leaks NOTHING: no paths, no party names, no
     document ids, no clause text. It is meant for a public issue tracker.
  4. Round-trip and tolerance: a manifest survives write/read, and a missing,
     corrupt, or older-schema manifest degrades to "first run" rather than
     blocking anybody.

SECURITY NOTE: the only corpus content touched here is the pre-committed
synthetic NDA example (fictional parties, fictional clause text).
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from playbook_engine import run_manifest as rm
from playbook_engine.cli import cli
from playbook_engine.config import load_config

_EXAMPLE_CONFIG = Path(__file__).parent.parent / "examples" / "nda" / "config.yaml"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def config():
    """The synthetic NDA example's real config (agent segmentation, auto extractor)."""
    return load_config(_EXAMPLE_CONFIG)


def _flat(text: str) -> str:
    """Collapse whitespace — the report is hard-wrapped to 72 columns, so an
    assertion on a phrase must not depend on where the wrap happened to land."""
    return " ".join(text.split())


def _docling_env(config, tmp_path: Path) -> rm.RunEnvironment:
    """A captured environment as if docling WERE installed."""
    env = rm.capture_environment(config, tmp_path)
    return replace(env, resolved_extractor="docling", docling_available=True)


def _legacy_env(config, tmp_path: Path) -> rm.RunEnvironment:
    """The same environment after docling vanished from the venv."""
    env = rm.capture_environment(config, tmp_path)
    return replace(env, resolved_extractor="legacy", docling_available=False)


#: The incident's real shape: 44 documents, 161 versions between them.
_DOCUMENTS = 44
_VERSIONS = 161


def _seed_out_dir(out_dir: Path, environment: rm.RunEnvironment) -> None:
    """Write a plausible warm out-dir: manifest + caches + a corpus manifest.

    Sized to the Aug 22 incident (44 documents / 161 versions) because the
    counts the plain-English consequence lines quote back to the user are read
    from these real files, not stubbed — a test that fakes the counts would
    not catch the report quoting the wrong one.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    entry = '{"key": "x", "verdict": {}}\n'
    (out_dir / "extraction_cache.jsonl").write_text(entry * _VERSIONS, encoding="utf-8")
    (out_dir / "segment").mkdir(exist_ok=True)
    (out_dir / "segment" / "cache.jsonl").write_text(entry * _VERSIONS, encoding="utf-8")

    # Spread _VERSIONS across _DOCUMENTS so version_ingest sums to the real
    # figure (3 or 4 versions per document, as a live negotiation corpus has).
    per_doc = [_VERSIONS // _DOCUMENTS] * _DOCUMENTS
    for i in range(_VERSIONS % _DOCUMENTS):
        per_doc[i] += 1
    docs = [
        {
            "document_id": f"doc-{i:03d}",
            "version_ingest": [
                {"version": v, "extractor": "docling", "reason": None} for v in range(n)
            ],
        }
        for i, n in enumerate(per_doc)
    ]
    (out_dir / "corpus_manifest.json").write_text(json.dumps(docs), encoding="utf-8")
    rm.write_run_manifest(out_dir, environment, command="mine")


# ---------------------------------------------------------------------------
# 1. Silence on the normal path
# ---------------------------------------------------------------------------


def test_preflight_is_silent_on_a_fresh_out_dir(config, tmp_path, capsys):
    """No prior manifest = first run = correct by construction. Say nothing."""
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    env = rm.preflight(out_dir, config, tmp_path, command="mine", echo=print)
    assert env.engine_version
    assert capsys.readouterr().out == ""


def test_preflight_is_silent_when_the_environment_matches(config, tmp_path, capsys):
    """The overwhelmingly common case: same machine, same engine, same config."""
    out_dir = tmp_path / "out"
    _seed_out_dir(out_dir, rm.capture_environment(config, tmp_path))
    rm.preflight(out_dir, config, tmp_path, command="mine", echo=print)
    assert capsys.readouterr().out == ""


def test_preflight_is_silent_when_the_manifest_is_corrupt(config, tmp_path, capsys):
    """A mangled manifest must degrade to 'first run', never block a good run."""
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / rm.RUN_MANIFEST_FILENAME).write_text("{not json", encoding="utf-8")
    rm.preflight(out_dir, config, tmp_path, command="mine", echo=print)
    assert capsys.readouterr().out == ""


def test_preflight_is_silent_when_the_manifest_schema_is_unknown(config, tmp_path, capsys):
    """A manifest from a FUTURE engine must not crash or nag an older one."""
    out_dir = tmp_path / "out"
    _seed_out_dir(out_dir, rm.capture_environment(config, tmp_path))
    path = out_dir / rm.RUN_MANIFEST_FILENAME
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["schema_version"] = "99"
    path.write_text(json.dumps(raw), encoding="utf-8")
    assert rm.read_run_manifest(out_dir) is None
    rm.preflight(out_dir, config, tmp_path, command="mine", echo=print)
    assert capsys.readouterr().out == ""


def test_older_manifest_missing_fields_produces_no_findings(config, tmp_path):
    """An upgrade that ADDS a manifest field must not fire on the first run after."""
    out_dir = tmp_path / "out"
    current = rm.capture_environment(config, tmp_path)
    _seed_out_dir(out_dir, current)
    path = out_dir / rm.RUN_MANIFEST_FILENAME
    raw = json.loads(path.read_text(encoding="utf-8"))
    for dropped in ("segmentation_effort", "artifact_cache_format", "git_sha"):
        raw["environment"].pop(dropped, None)
    path.write_text(json.dumps(raw), encoding="utf-8")

    previous = rm.read_run_manifest(out_dir)
    assert previous is not None
    assert rm.compare(previous.environment, current, rm.collect_counts(out_dir)) == []


# ---------------------------------------------------------------------------
# 2. The docling-disappeared case
# ---------------------------------------------------------------------------


def test_docling_disappeared_is_blocking(config, tmp_path):
    counts = {"versions": 161, "segmentation_cache_entries": 161, "documents": 44}
    findings = rm.compare(_docling_env(config, tmp_path), _legacy_env(config, tmp_path), counts)
    assert [f.code for f in findings] == ["extractor-environment-changed"]
    assert findings[0].blocking is True


def test_docling_disappeared_report_is_plain_english(config, tmp_path):
    """The headline must be a sentence a lawyer can act on, not a hash diff."""
    out_dir = tmp_path / "out"
    _seed_out_dir(out_dir, _docling_env(config, tmp_path))
    previous = rm.read_run_manifest(out_dir)
    assert previous is not None
    current = _legacy_env(config, tmp_path)
    counts = rm.collect_counts(out_dir)
    findings = rm.compare(previous.environment, current, counts)
    report = rm.render_report(findings, previous, current, counts, command="mine", stopping=True)

    # Names the problem in the user's terms.
    flat = _flat(report)
    assert "docling isn't available here" in flat
    # States the practical consequence, with the real count off disk.
    assert f"All {_VERSIONS} document version(s) in this folder would be read again" in flat
    assert f"The {_VERSIONS} saved clause grouping(s)" in flat
    assert "segmentation-pending" in flat
    # Gives the concrete fix.
    assert "pip install docling" in flat
    assert "--accept-environment-change" in flat
    # Reassures that nothing was destroyed.
    assert "no work has been thrown away" in flat
    # No stack-trace / diff-of-hashes vocabulary in the prose section.
    prose = report.split(rm._PASTE_OPEN)[0]
    for jargon in ("Traceback", "extractor_env", "sha256", "AgentSegmentationPending"):
        assert jargon not in prose


def test_preflight_raises_and_carries_the_report(config, tmp_path, monkeypatch):
    out_dir = tmp_path / "out"
    _seed_out_dir(out_dir, _docling_env(config, tmp_path))
    monkeypatch.setattr(rm.shutil, "which", lambda name: None)

    with pytest.raises(rm.EnvironmentMismatch) as excinfo:
        rm.preflight(out_dir, config, tmp_path, command="mine")
    assert "docling isn't available here" in _flat(excinfo.value.report)
    assert [f.code for f in excinfo.value.findings] == ["extractor-environment-changed"]


def test_accept_environment_change_downgrades_to_a_printed_report(config, tmp_path, monkeypatch):
    """The escape hatch explains the rework, then proceeds — it never goes quiet."""
    out_dir = tmp_path / "out"
    _seed_out_dir(out_dir, _docling_env(config, tmp_path))
    monkeypatch.setattr(rm.shutil, "which", lambda name: None)

    printed: list[str] = []
    rm.preflight(out_dir, config, tmp_path, command="mine", echo=printed.append, accept_change=True)
    assert len(printed) == 1
    assert "docling isn't available here" in _flat(printed[0])
    assert "--accept-environment-change was given" in _flat(printed[0])


def test_advisory_only_difference_gets_one_short_line_and_proceeds(config, tmp_path):
    """A deliberate config edit must not be answered with a bug-report block.

    Escalating a normal user action to the full stop-the-run treatment is how
    the real warning ends up being skimmed past.
    """
    out_dir = tmp_path / "out"
    _seed_out_dir(out_dir, replace(rm.capture_environment(config, tmp_path), config_hash="stale"))

    printed: list[str] = []
    rm.preflight(out_dir, config, tmp_path, command="mine", echo=printed.append)

    assert len(printed) == 1
    assert printed[0].startswith("  note: ")
    assert "has been edited" in printed[0]
    # Short form: no rules, no bug-report block, no remediation essay.
    assert rm._PASTE_OPEN not in printed[0]
    assert "─" not in printed[0]
    assert len(printed[0].splitlines()) <= 2


# ---------------------------------------------------------------------------
# 3. The copy-pasteable block must be safe for a public issue
# ---------------------------------------------------------------------------


def test_paste_block_leaks_no_customer_data(tmp_path):
    """Hard invariant: no paths, no party names, no document ids, no counterparties.

    Built deliberately from a config whose every free-text field is a
    recognisable secret string, then asserted absent from the block. If you
    add a field to the report, add its secret here.
    """
    secret_config = tmp_path / "secret.config.yaml"
    taxonomy_path = Path(__file__).parent.parent / "spec" / "taxonomy" / "nda.yaml"
    secret_config.write_text(
        yaml.dump(
            {
                "agreement_type": {
                    "id": "nda",
                    "name": "SECRETAGREEMENTNAME",
                    "description": "SECRETDESCRIPTION",
                    "aliases": ["SECRETALIAS"],
                },
                "baseline": {},
                "taxonomy": str(taxonomy_path),
                "provenance": {
                    "our_party_aliases": ["SECRETPARTY"],
                    "our_authors": ["SECRETAUTHOR"],
                    "known_entities": ["SECRETCOUNTERPARTY"],
                },
                "perspective": {
                    "party": "SECRETPARTY",
                    "counterparty_type": "SECRETCOUNTERPARTYTYPE",
                },
            }
        ),
        encoding="utf-8",
    )
    config = load_config(secret_config)

    out_dir = tmp_path / "SECRETFOLDERNAME" / "out"
    previous_env = replace(
        rm.capture_environment(config, tmp_path),
        resolved_extractor="docling",
        docling_available=True,
    )
    _seed_out_dir(out_dir, previous_env)
    previous = rm.read_run_manifest(out_dir)
    assert previous is not None
    current = replace(previous_env, resolved_extractor="legacy", docling_available=False)
    counts = rm.collect_counts(out_dir)
    block = rm.render_paste_block(
        rm.compare(previous.environment, current, counts), previous, current, counts
    )

    for secret in (
        "SECRETAGREEMENTNAME",
        "SECRETDESCRIPTION",
        "SECRETALIAS",
        "SECRETPARTY",
        "SECRETAUTHOR",
        "SECRETCOUNTERPARTY",
        "SECRETCOUNTERPARTYTYPE",
        "SECRETFOLDERNAME",
        "doc-000",
        str(out_dir),
        str(tmp_path),
    ):
        assert secret not in block, f"{secret!r} leaked into the paste block"

    # ...and it still carries the facts a maintainer needs.
    assert "extractor-environment-changed" in block
    assert "docling -> legacy" in block
    assert "count versions" in block


def test_manifest_file_itself_leaks_no_paths(config, tmp_path):
    """The on-disk manifest is also gitignore-adjacent evidence — keep it clean."""
    out_dir = tmp_path / "SECRETFOLDERNAME" / "out"
    _seed_out_dir(out_dir, rm.capture_environment(config, tmp_path))
    text = (out_dir / rm.RUN_MANIFEST_FILENAME).read_text(encoding="utf-8")
    assert "SECRETFOLDERNAME" not in text
    assert str(tmp_path) not in text
    assert "AlphaCorp" not in text  # the example config's party alias


# ---------------------------------------------------------------------------
# 4. Round-trip, counts, and the other findings
# ---------------------------------------------------------------------------


def test_manifest_round_trips(config, tmp_path):
    out_dir = tmp_path / "out"
    env = rm.capture_environment(config, tmp_path)
    _seed_out_dir(out_dir, env)
    loaded = rm.read_run_manifest(out_dir)
    assert loaded is not None
    assert loaded.environment == env
    assert loaded.written_by == "mine"
    assert loaded.counts["documents"] == _DOCUMENTS
    assert loaded.counts["versions"] == _VERSIONS
    assert loaded.counts["extraction_cache_entries"] == _VERSIONS


def test_counts_degrade_to_zero_rather_than_raising(tmp_path):
    """collect_counts runs while REPORTING a problem; it must never become one."""
    counts = rm.collect_counts(tmp_path / "does-not-exist")
    assert counts["documents"] == 0
    assert counts["extraction_cache_entries"] == 0


def test_extraction_cache_format_bump_is_blocking(config, tmp_path):
    """Caches built before an Aug 3-style 1->2->3 bump were silently unreachable."""
    previous = replace(rm.capture_environment(config, tmp_path), extraction_cache_format="2")
    current = rm.capture_environment(config, tmp_path)
    findings = rm.compare(previous, current, {"versions": 161})
    assert "extraction-cache-format-changed" in [f.code for f in findings]
    assert all(f.blocking for f in findings if f.code == "extraction-cache-format-changed")


def test_engine_downgrade_is_blocking_but_upgrade_is_not(config, tmp_path):
    """The stale-Docker-image case: an older engine would look successful."""
    current = rm.capture_environment(config, tmp_path)

    downgrade = rm.compare(replace(current, engine_version="9.9.9"), current, {})
    assert [f.code for f in downgrade] == ["engine-version-downgrade"]
    assert downgrade[0].blocking is True
    assert "playbook --version" in " ".join(downgrade[0].fix)
    assert "docker-build" in " ".join(downgrade[0].fix)

    upgrade = rm.compare(replace(current, engine_version="0.0.1"), current, {})
    assert [f.code for f in upgrade] == ["engine-version-changed"]
    assert upgrade[0].blocking is False


def test_segmentation_identity_change_is_blocking(config, tmp_path):
    current = rm.capture_environment(config, tmp_path)
    previous = replace(current, segmentation_mode="llm")
    findings = rm.compare(previous, current, {"segmentation_cache_entries": 161})
    assert [f.code for f in findings] == ["segmentation-identity-changed"]
    assert findings[0].blocking is True
    assert "llm → agent" in findings[0].headline


def test_config_edit_is_advisory_not_blocking(config, tmp_path):
    current = rm.capture_environment(config, tmp_path)
    findings = rm.compare(replace(current, config_hash="deadbeef" * 8), current, {"documents": 44})
    assert [f.code for f in findings] == ["config-changed"]
    assert findings[0].blocking is False


def test_platform_and_python_changes_are_not_compared(config, tmp_path):
    """Host vs. container is a healthy workflow — comparing these would nag on it."""
    current = rm.capture_environment(config, tmp_path)
    previous = replace(current, platform="Linux-6.1-x86_64", python_version="3.11.9")
    assert rm.compare(previous, current, {}) == []


def test_config_hash_moves_with_the_taxonomy_file(config, tmp_path):
    """Editing the taxonomy changes engine output, so it must move the hash."""
    before = rm.compute_config_hash(config)
    copied = tmp_path / "taxonomy.yaml"
    copied.write_text(
        config.taxonomy_path.read_text(encoding="utf-8") + "\n# edited\n", encoding="utf-8"
    )
    after = rm.compute_config_hash(replace(config, taxonomy_path=copied))
    assert before != after


def test_findings_are_ordered_blocking_first(config, tmp_path):
    current = rm.capture_environment(config, tmp_path)
    previous = replace(
        current,
        config_hash="deadbeef" * 8,  # advisory
        resolved_extractor="docling",  # blocking
        docling_available=True,
    )
    current = replace(current, resolved_extractor="legacy", docling_available=False)
    codes = [f.code for f in rm.compare(previous, current, {"versions": 3})]
    assert codes[0] == "extractor-environment-changed"
    assert "config-changed" in codes


# ---------------------------------------------------------------------------
# 5. CLI wiring
# ---------------------------------------------------------------------------


def test_mine_and_judge_expose_the_flag_the_report_names():
    """The report tells people to pass --accept-environment-change; it must exist.

    A drift between the remediation text and the real option would be worse
    than printing nothing at all.
    """
    runner = CliRunner()
    for command in ("mine", "judge", "segment"):
        result = runner.invoke(cli, [command, "--help"])
        assert result.exit_code == 0
        assert "--accept-environment-change" in result.output


def test_mine_stops_before_doing_work_when_docling_vanished(config, tmp_path, monkeypatch):
    """End-to-end: the CLI exits 1 with the plain-English report and mines nothing."""
    corpus_dir = Path(__file__).parent.parent / "examples" / "nda" / "corpus"
    out_dir = tmp_path / "out"
    _seed_out_dir(out_dir, _docling_env(config, corpus_dir))
    # Blow away the observation store so "did it do work?" is unambiguous.
    (out_dir / "observations.jsonl").unlink(missing_ok=True)

    import playbook_engine.run_manifest as target

    monkeypatch.setattr(target.shutil, "which", lambda name: None)

    runner = CliRunner()
    result = runner.invoke(
        cli, ["mine", str(corpus_dir), "--config", str(_EXAMPLE_CONFIG), "--out", str(out_dir)]
    )
    assert result.exit_code == 1
    assert "docling isn't available here" in _flat(result.output)
    assert not (out_dir / "observations.jsonl").exists()


def test_segment_stops_before_doing_work_when_docling_vanished(config, tmp_path, monkeypatch):
    """End-to-end regression for issue #173.

    Before the fix, ``segment`` had no provenance preflight at all: it would
    silently re-extract every version under the degraded (legacy) environment
    and bank agent-segmentation work against canonical_text hashes that a
    later ``mine`` would never recognize, only refusing at that later ``mine``.
    Now it must refuse here, before any of that work starts, exactly the way
    ``mine`` does.
    """
    corpus_dir = Path(__file__).parent.parent / "examples" / "nda" / "corpus"
    out_dir = tmp_path / "out"
    _seed_out_dir(out_dir, _docling_env(config, corpus_dir))
    # Blow away the caches `segment` would otherwise (re)write, so "did it do
    # any extraction work?" is unambiguous.
    (out_dir / "extraction_cache.jsonl").unlink(missing_ok=True)
    (out_dir / "segment" / "cache.jsonl").unlink(missing_ok=True)
    pending_path = out_dir / "segment" / "pending.jsonl"

    import playbook_engine.run_manifest as target

    monkeypatch.setattr(target.shutil, "which", lambda name: None)

    runner = CliRunner()
    result = runner.invoke(
        cli, ["segment", str(corpus_dir), "--config", str(_EXAMPLE_CONFIG), "--out", str(out_dir)]
    )
    assert result.exit_code == 1
    assert "docling isn't available here" in _flat(result.output)
    assert not pending_path.exists()


def test_segment_stamps_the_run_manifest_on_success(config, tmp_path):
    """A fresh, successful `segment` run leaves a run_manifest.json behind.

    Mirrors `mine`'s stamp at the end of a successful run (issue #121) — this
    is the half of issue #173 that lets the *next* `segment`/`mine`/`judge`
    against this out_dir detect a subsequently-drifted environment at all.
    """
    corpus_dir = Path(__file__).parent.parent / "examples" / "nda" / "corpus"
    out_dir = tmp_path / "out"

    runner = CliRunner()
    result = runner.invoke(
        cli, ["segment", str(corpus_dir), "--config", str(_EXAMPLE_CONFIG), "--out", str(out_dir)]
    )
    assert result.exit_code == 0, result.output
    manifest = rm.read_run_manifest(out_dir)
    assert manifest is not None
    assert manifest.written_by == "segment"
