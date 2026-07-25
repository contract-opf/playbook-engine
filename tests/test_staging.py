"""Tests for staging.py — generalized corpus staging (playbook stage).

SECURITY NOTE: All fixtures are programmatically constructed with synthetic
text.  No real agreement files are committed or referenced.
Fictional party and company names only.

The three persistent example fixture trees under examples/staging-fixtures/
are also exercised here to guarantee they stay in sync with the code.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from playbook_engine.cli import cli
from playbook_engine.staging import DEFAULT_STAGING_ROOT, detect_layout, scaffold_config, stage

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FIXTURES = Path(__file__).parent.parent / "examples" / "staging-fixtures"


def _write_rtf(path: Path, text: str = "stub") -> None:
    path.write_text(f"{{\\rtf1 {text}}}\n", encoding="utf-8")


def _read_hints(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


# ---------------------------------------------------------------------------
# detect_layout
# ---------------------------------------------------------------------------


class TestDetectLayout:
    def test_flat_layout(self, tmp_path: Path) -> None:
        (tmp_path / "deal-a").mkdir()
        _write_rtf(tmp_path / "deal-a" / "v1.rtf")
        assert detect_layout(tmp_path) == "flat"

    def test_clm_nested_layout(self, tmp_path: Path) -> None:
        (tmp_path / "deal-a" / "Versions").mkdir(parents=True)
        _write_rtf(tmp_path / "deal-a" / "Versions" / "v1.rtf")
        assert detect_layout(tmp_path) == "clm_nested"

    def test_manifest_layout(self, tmp_path: Path) -> None:
        (tmp_path / "manifest.jsonl").write_text("", encoding="utf-8")
        assert detect_layout(tmp_path) == "manifest"

    def test_manifest_takes_priority_over_nested(self, tmp_path: Path) -> None:
        """manifest.jsonl present → manifest, even if Versions/ subdir exists."""
        (tmp_path / "deal-a" / "Versions").mkdir(parents=True)
        (tmp_path / "manifest.jsonl").write_text("", encoding="utf-8")
        assert detect_layout(tmp_path) == "manifest"

    def test_empty_dir_is_unknown(self, tmp_path: Path) -> None:
        """No subfolders at all → no evidence of any known layout (issue #186:
        previously fell through to "flat" by default, the bug this issue
        fixes — "flat" now requires at least one agreement subfolder with
        supported files)."""
        assert detect_layout(tmp_path) == "unknown"

    # ---- example fixtures ----

    def test_fixture_flat_corpus(self) -> None:
        assert detect_layout(FIXTURES / "flat-corpus") == "flat"

    def test_fixture_clm_nested_corpus(self) -> None:
        assert detect_layout(FIXTURES / "clm-nested-corpus") == "clm_nested"

    def test_fixture_manifest_corpus(self) -> None:
        assert detect_layout(FIXTURES / "manifest-corpus") == "manifest"


# ---------------------------------------------------------------------------
# stage — flat layout
# ---------------------------------------------------------------------------


class TestStageFlat:
    def test_creates_output_dir(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        (src / "deal-a").mkdir(parents=True)
        _write_rtf(src / "deal-a" / "v1.rtf")
        stage(src, tmp_path / "out")
        assert (tmp_path / "out").is_dir()

    def test_symlinks_created(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        (src / "deal-a").mkdir(parents=True)
        _write_rtf(src / "deal-a" / "v1.rtf")
        stage(src, tmp_path / "out")
        links = list((tmp_path / "out" / "deal-a").iterdir())
        symlinks = [p for p in links if p.is_symlink()]
        assert len(symlinks) == 1

    def test_order_in_hints_matches_sorted_files(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        (src / "deal-a").mkdir(parents=True)
        _write_rtf(src / "deal-a" / "v1.rtf")
        _write_rtf(src / "deal-a" / "v2.rtf")
        stage(src, tmp_path / "out")
        hints = _read_hints(tmp_path / "out" / "deal-a" / "hints.yaml")
        assert hints["order"] == ["01__v1", "02__v2"]

    def test_signed_detected_from_stem(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        (src / "deal-a").mkdir(parents=True)
        _write_rtf(src / "deal-a" / "v1.rtf")
        _write_rtf(src / "deal-a" / "v2_signed_final.rtf")
        stage(src, tmp_path / "out")
        hints = _read_hints(tmp_path / "out" / "deal-a" / "hints.yaml")
        assert hints.get("signed_version") == "02__v2_signed_final"

    def test_executed_detected_from_stem(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        (src / "deal-a").mkdir(parents=True)
        _write_rtf(src / "deal-a" / "v1.rtf")
        _write_rtf(src / "deal-a" / "v2_executed.rtf")
        stage(src, tmp_path / "out")
        hints = _read_hints(tmp_path / "out" / "deal-a" / "hints.yaml")
        assert hints.get("signed_version") == "02__v2_executed"

    def test_no_signed_if_no_cue(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        (src / "deal-a").mkdir(parents=True)
        _write_rtf(src / "deal-a" / "v1.rtf")
        _write_rtf(src / "deal-a" / "v2.rtf")
        stage(src, tmp_path / "out")
        hints = _read_hints(tmp_path / "out" / "deal-a" / "hints.yaml")
        assert hints.get("signed_version") is None

    def test_unsigned_stem_not_marked_signed(self, tmp_path: Path) -> None:
        """Issue #96: a filename that merely *contains* "signed" as a substring
        of "unsigned", or that describes a not-yet-signed draft, must not
        hijack the signed anchor — CORPUS-LAYOUT.md promises filenames are
        never trusted.
        """
        src = tmp_path / "src"
        (src / "deal-a").mkdir(parents=True)
        _write_rtf(src / "deal-a" / "v1.rtf")
        _write_rtf(src / "deal-a" / "unsigned-execution-copy.rtf")
        _write_rtf(src / "deal-a" / "draft-to-be-signed.rtf")
        stage(src, tmp_path / "out")
        hints = _read_hints(tmp_path / "out" / "deal-a" / "hints.yaml")
        assert hints.get("signed_version") is None

    def test_true_signed_stem_still_detected_alongside_negations(self, tmp_path: Path) -> None:
        """A real signed copy is still anchored even when negation-cue files
        (drafts, unsigned copies) are also present in the same folder.
        """
        src = tmp_path / "src"
        (src / "deal-a").mkdir(parents=True)
        _write_rtf(src / "deal-a" / "unsigned-execution-copy.rtf")
        _write_rtf(src / "deal-a" / "v2_signed_final.rtf")
        stage(src, tmp_path / "out")
        hints = _read_hints(tmp_path / "out" / "deal-a" / "hints.yaml")
        assert hints.get("signed_version") == "02__v2_signed_final"

    def test_result_counts(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        (src / "deal-a").mkdir(parents=True)
        (src / "deal-b").mkdir(parents=True)
        _write_rtf(src / "deal-a" / "v1.rtf")
        _write_rtf(src / "deal-b" / "v1.rtf")
        _write_rtf(src / "deal-b" / "v2.rtf")
        result = stage(src, tmp_path / "out")
        assert result.layout == "flat"
        assert result.agreement_count == 2
        assert result.staged_count == 3

    def test_empty_deal_dir_skipped(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        (src / "deal-a").mkdir(parents=True)
        (src / "deal-b").mkdir(parents=True)
        _write_rtf(src / "deal-a" / "v1.rtf")
        # deal-b has no supported files
        result = stage(src, tmp_path / "out")
        assert result.agreement_count == 1
        assert not (tmp_path / "out" / "deal-b").exists()

    def test_out_dir_wiped_on_rerun(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        (src / "deal-a").mkdir(parents=True)
        _write_rtf(src / "deal-a" / "v1.rtf")
        out = tmp_path / "out"
        stage(src, out)
        # drop a stray file, rerun → it disappears
        (out / "stray.txt").write_text("stray")
        stage(src, out)
        assert not (out / "stray.txt").exists()

    # ---- example fixture ----

    def test_fixture_flat_corpus_stages_correctly(self, tmp_path: Path) -> None:
        result = stage(FIXTURES / "flat-corpus", tmp_path / "out")
        assert result.layout == "flat"
        assert result.agreement_count == 2
        assert result.staged_count == 4
        hints_alpha = _read_hints(tmp_path / "out" / "deal-alpha" / "hints.yaml")
        assert "01__01_draft" in hints_alpha["order"]
        assert hints_alpha.get("signed_version") is not None


# ---------------------------------------------------------------------------
# stage — CLM-nested layout
# ---------------------------------------------------------------------------


class TestStageCLMNested:
    def _make_clm(self, tmp_path: Path) -> Path:
        src = tmp_path / "src"
        (src / "deal-a" / "Versions").mkdir(parents=True)
        (src / "deal-b" / "Versions").mkdir(parents=True)
        _write_rtf(src / "deal-a" / "Versions" / "v1.rtf")
        _write_rtf(src / "deal-a" / "Versions" / "v2.rtf")
        _write_rtf(src / "deal-b" / "Versions" / "v1.rtf")
        return src

    def test_versions_subfolder_staged(self, tmp_path: Path) -> None:
        src = self._make_clm(tmp_path)
        result = stage(src, tmp_path / "out")
        assert result.layout == "clm_nested"
        assert result.agreement_count == 2

    def test_executed_pdf_appended_and_marked_signed(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        (src / "deal-a" / "Versions").mkdir(parents=True)
        _write_rtf(src / "deal-a" / "Versions" / "v1.rtf")
        (src / "deal-a" / "EXECUTED_deal-a.pdf").write_bytes(b"%PDF-1.4 stub")
        stage(src, tmp_path / "out")
        hints = _read_hints(tmp_path / "out" / "deal-a" / "hints.yaml")
        assert hints.get("signed_version") is not None
        assert "EXECUTED_deal-a" in hints["signed_version"]

    def test_hints_order_includes_all_versions(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        (src / "deal-a" / "Versions").mkdir(parents=True)
        _write_rtf(src / "deal-a" / "Versions" / "v1.rtf")
        _write_rtf(src / "deal-a" / "Versions" / "v2.rtf")
        (src / "deal-a" / "EXECUTED_deal-a.pdf").write_bytes(b"%PDF-1.4 stub")
        stage(src, tmp_path / "out")
        hints = _read_hints(tmp_path / "out" / "deal-a" / "hints.yaml")
        assert len(hints["order"]) == 3

    # ---- example fixture ----

    def test_fixture_clm_nested_corpus(self, tmp_path: Path) -> None:
        result = stage(FIXTURES / "clm-nested-corpus", tmp_path / "out")
        assert result.layout == "clm_nested"
        assert result.agreement_count == 2
        # Each deal has 2 Versions/ files + 1 EXECUTED top-level PDF → 3 each
        assert result.staged_count == 6
        hints_gamma = _read_hints(tmp_path / "out" / "deal-gamma" / "hints.yaml")
        assert hints_gamma.get("signed_version") is not None
        assert "EXECUTED_deal-gamma" in hints_gamma["signed_version"]


# ---------------------------------------------------------------------------
# stage — manifest layout
# ---------------------------------------------------------------------------


class TestStageManifest:
    def _make_manifest_src(self, tmp_path: Path) -> Path:
        src = tmp_path / "src"
        docs = src / "docs"
        (docs / "deal-a").mkdir(parents=True)
        (docs / "deal-b").mkdir(parents=True)
        _write_rtf(docs / "deal-a" / "v1.rtf")
        _write_rtf(docs / "deal-a" / "v2.rtf")
        _write_rtf(docs / "deal-b" / "v1.rtf")
        lines = [
            json.dumps(
                {
                    "folder": "deal-a",
                    "filename_on_disk": "docs/deal-a/v1.rtf",
                    "versionNumber": 1,
                    "original_filename": "draft.rtf",
                    "status": "DRAFT",
                }
            ),
            json.dumps(
                {
                    "folder": "deal-a",
                    "filename_on_disk": "docs/deal-a/v2.rtf",
                    "versionNumber": 2,
                    "original_filename": "final.rtf",
                    "status": "EXECUTED",
                }
            ),
            json.dumps(
                {
                    "folder": "deal-b",
                    "filename_on_disk": "docs/deal-b/v1.rtf",
                    "versionNumber": 1,
                    "original_filename": "draft.rtf",
                    "status": "DRAFT",
                }
            ),
        ]
        (src / "manifest.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return src

    def test_layout_detected_as_manifest(self, tmp_path: Path) -> None:
        src = self._make_manifest_src(tmp_path)
        assert detect_layout(src) == "manifest"

    def test_stage_manifest_counts(self, tmp_path: Path) -> None:
        src = self._make_manifest_src(tmp_path)
        result = stage(src, tmp_path / "out")
        assert result.layout == "manifest"
        assert result.agreement_count == 2
        assert result.staged_count == 3

    def test_signed_version_from_executed_status(self, tmp_path: Path) -> None:
        src = self._make_manifest_src(tmp_path)
        stage(src, tmp_path / "out")
        hints = _read_hints(tmp_path / "out" / "deal-a" / "hints.yaml")
        assert hints.get("signed_version") == "02__final"

    def test_order_respects_version_number(self, tmp_path: Path) -> None:
        src = self._make_manifest_src(tmp_path)
        stage(src, tmp_path / "out")
        hints = _read_hints(tmp_path / "out" / "deal-a" / "hints.yaml")
        assert hints["order"] == ["01__draft", "02__final"]

    def test_filename_relative_to_docs_subdir(self, tmp_path: Path) -> None:
        """Real CLM manifests give filename_on_disk relative to docs/ (no docs/ prefix).

        Regression for a real-world corpus (stage found 0 of 161 files): the agreement
        folders live under docs/ but the manifest paths omit that prefix, so
        resolution must also try the docs/ base.
        """
        src = tmp_path / "src"
        vdir = src / "docs" / "deal-z" / "Versions"
        vdir.mkdir(parents=True)
        _write_rtf(vdir / "01_IN_REVIEW.rtf")
        _write_rtf(vdir / "02_EXECUTED.rtf")
        lines = [
            json.dumps(
                {
                    "folder": "deal-z",
                    "filename_on_disk": "deal-z/Versions/01_IN_REVIEW.rtf",
                    "versionNumber": 1,
                    "original_filename": "draft.rtf",
                    "status": "IN_REVIEW",
                }
            ),
            json.dumps(
                {
                    "folder": "deal-z",
                    "filename_on_disk": "deal-z/Versions/02_EXECUTED.rtf",
                    "versionNumber": 2,
                    "original_filename": "final.rtf",
                    "status": "EXECUTED",
                }
            ),
        ]
        (src / "manifest.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")

        result = stage(src, tmp_path / "out")

        assert result.layout == "manifest"
        assert result.staged_count == 2
        assert result.agreement_count == 1
        assert not result.missing
        hints = _read_hints(tmp_path / "out" / "deal-z" / "hints.yaml")
        assert hints.get("signed_version") == "02__final"

    def test_missing_file_reported(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        (src / "docs" / "deal-a").mkdir(parents=True)
        _write_rtf(src / "docs" / "deal-a" / "v1.rtf")
        line = json.dumps(
            {
                "folder": "deal-a",
                "filename_on_disk": "docs/deal-a/nonexistent.rtf",
                "versionNumber": 99,
                "original_filename": "nonexistent.rtf",
                "status": "DRAFT",
            }
        )
        (src / "manifest.jsonl").write_text(line + "\n", encoding="utf-8")
        result = stage(src, tmp_path / "out")
        assert len(result.missing) == 1

    def test_custom_manifest_and_docs_paths(self, tmp_path: Path) -> None:
        """manifest_path and docs_path override defaults."""
        docs = tmp_path / "custom-docs"
        (docs / "deal-c").mkdir(parents=True)
        _write_rtf(docs / "deal-c" / "v1.rtf")
        manifest = tmp_path / "custom-manifest.jsonl"
        manifest.write_text(
            json.dumps(
                {
                    "folder": "deal-c",
                    "filename_on_disk": "deal-c/v1.rtf",
                    "versionNumber": 1,
                    "original_filename": "draft.rtf",
                    "status": "DRAFT",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        # src_dir has no manifest.jsonl — supply overrides explicitly
        src = tmp_path / "src"
        src.mkdir()
        (src / "manifest.jsonl").write_text("", encoding="utf-8")  # empty → still manifest layout
        result = stage(src, tmp_path / "out", manifest_path=manifest, docs_path=docs)
        assert result.staged_count == 1

    # ---- example fixture ----

    def test_fixture_manifest_corpus(self, tmp_path: Path) -> None:
        src = FIXTURES / "manifest-corpus"
        result = stage(
            src,
            tmp_path / "out",
            docs_path=src,
        )
        assert result.layout == "manifest"
        assert result.agreement_count == 2
        assert result.staged_count == 5
        hints_eps = _read_hints(tmp_path / "out" / "deal-epsilon" / "hints.yaml")
        assert hints_eps.get("signed_version") == "03__executed_final"
        hints_zeta = _read_hints(tmp_path / "out" / "deal-zeta" / "hints.yaml")
        assert hints_zeta.get("signed_version") == "02__signed"


# ---------------------------------------------------------------------------
# stage — copy_files=True (issue #130)
# ---------------------------------------------------------------------------


class TestStageCopyFiles:
    """``copy_files=True`` writes real file copies instead of absolute
    symlinks, so the staged tree survives crossing a filesystem boundary
    (e.g. staged on the host, then bind-mounted read-only into a container).
    """

    def test_default_still_symlinks(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        (src / "deal-a").mkdir(parents=True)
        _write_rtf(src / "deal-a" / "v1.rtf")
        stage(src, tmp_path / "out")
        staged_file = next((tmp_path / "out" / "deal-a").glob("01__*"))
        assert staged_file.is_symlink()

    def test_copy_flag_creates_real_files_not_symlinks(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        (src / "deal-a").mkdir(parents=True)
        _write_rtf(src / "deal-a" / "v1.rtf", text="original content")
        stage(src, tmp_path / "out", copy_files=True)
        staged_file = next((tmp_path / "out" / "deal-a").glob("01__*"))
        assert not staged_file.is_symlink()
        assert "original content" in staged_file.read_text(encoding="utf-8")

    def test_copy_flag_survives_source_removal(self, tmp_path: Path) -> None:
        """The whole point: once copied, the staged file no longer depends on
        the source path existing (unlike a symlink, whose target would dangle).
        """
        src = tmp_path / "src"
        (src / "deal-a").mkdir(parents=True)
        _write_rtf(src / "deal-a" / "v1.rtf", text="stand-alone")
        stage(src, tmp_path / "out", copy_files=True)
        shutil.rmtree(src)
        staged_file = next((tmp_path / "out" / "deal-a").glob("01__*"))
        assert "stand-alone" in staged_file.read_text(encoding="utf-8")

    def test_copy_flag_clm_nested_layout(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        (src / "deal-a" / "Versions").mkdir(parents=True)
        _write_rtf(src / "deal-a" / "Versions" / "v1.rtf")
        (src / "deal-a" / "EXECUTED_deal-a.pdf").write_bytes(b"%PDF-1.4 stub")
        stage(src, tmp_path / "out", copy_files=True)
        for staged_file in (tmp_path / "out" / "deal-a").glob("*"):
            if staged_file.name != "hints.yaml":
                assert not staged_file.is_symlink()

    def test_copy_flag_manifest_layout(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        docs = src / "docs" / "deal-a"
        docs.mkdir(parents=True)
        _write_rtf(docs / "v1.rtf")
        (src / "manifest.jsonl").write_text(
            json.dumps(
                {
                    "folder": "deal-a",
                    "filename_on_disk": "docs/deal-a/v1.rtf",
                    "versionNumber": 1,
                    "original_filename": "draft.rtf",
                    "status": "DRAFT",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        stage(src, tmp_path / "out", copy_files=True)
        staged_file = next((tmp_path / "out" / "deal-a").glob("01__*"))
        assert not staged_file.is_symlink()


# ---------------------------------------------------------------------------
# scaffold_config
# ---------------------------------------------------------------------------


class TestScaffoldConfig:
    def test_writes_playbook_config_yaml(self, tmp_path: Path) -> None:
        """Scaffolded config must use the canonical filename (issue #130):
        every other CLI command / doc references ``playbook.config.yaml``,
        not ``config.yaml``.
        """
        src = tmp_path / "my-agreement"
        src.mkdir()
        out = tmp_path / "out"
        scaffold_config(src, out)
        assert (out / "playbook.config.yaml").is_file()
        assert not (out / "config.yaml").exists()

    def test_id_is_valid_slug(self, tmp_path: Path) -> None:
        src = tmp_path / "My Agreement Type"
        src.mkdir()
        out = tmp_path / "out"
        skeleton = scaffold_config(src, out)
        slug = skeleton["agreement_type"]["id"]
        import re  # noqa: PLC0415

        assert re.match(r"^[a-z0-9-]+$", slug), f"id not a valid slug: {slug!r}"

    def test_template_detected_when_present(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        src.mkdir()
        _write_rtf(src / "template_v1.rtf")
        skeleton = scaffold_config(src, tmp_path / "out")
        assert skeleton["baseline"]["template"] == "template_v1.rtf"

    def test_template_null_when_absent(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        src.mkdir()
        skeleton = scaffold_config(src, tmp_path / "out")
        assert skeleton["baseline"]["template"] is None

    def test_taxonomy_placeholder_present(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        src.mkdir()
        skeleton = scaffold_config(src, tmp_path / "out")
        assert skeleton["taxonomy"] == "FILL_IN_TAXONOMY_PATH"

    def test_our_party_aliases_empty_list(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        src.mkdir()
        skeleton = scaffold_config(src, tmp_path / "out")
        assert skeleton["provenance"]["our_party_aliases"] == []

    def test_roundtrip_yaml_is_parseable(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        src.mkdir()
        out = tmp_path / "out"
        scaffold_config(src, out)
        raw = yaml.safe_load((out / "playbook.config.yaml").read_text(encoding="utf-8"))
        assert isinstance(raw, dict)
        assert "agreement_type" in raw
        assert "baseline" in raw
        assert "taxonomy" in raw
        assert "provenance" in raw


# ---------------------------------------------------------------------------
# Integration: stage output passes lint-corpus
# ---------------------------------------------------------------------------


class TestStagedOutputLintClean:
    """staged flat fixture → playbook lint-corpus reports no errors."""

    def test_flat_fixture_lint_clean(self, tmp_path: Path) -> None:
        result = stage(FIXTURES / "flat-corpus", tmp_path / "out")
        assert result.agreement_count > 0
        runner = CliRunner()
        invocation = runner.invoke(cli, ["lint-corpus", str(tmp_path / "out")])
        assert invocation.exit_code == 0, invocation.output

    def test_clm_nested_fixture_lint_clean(self, tmp_path: Path) -> None:
        result = stage(FIXTURES / "clm-nested-corpus", tmp_path / "out")
        assert result.agreement_count > 0
        runner = CliRunner()
        invocation = runner.invoke(cli, ["lint-corpus", str(tmp_path / "out")])
        assert invocation.exit_code == 0, invocation.output

    def test_manifest_fixture_lint_clean(self, tmp_path: Path) -> None:
        src = FIXTURES / "manifest-corpus"
        result = stage(src, tmp_path / "out", docs_path=src)
        assert result.agreement_count > 0
        runner = CliRunner()
        invocation = runner.invoke(cli, ["lint-corpus", str(tmp_path / "out")])
        assert invocation.exit_code == 0, invocation.output


# ---------------------------------------------------------------------------
# CLI: playbook stage
# ---------------------------------------------------------------------------


class TestStageCLI:
    def test_stage_cmd_flat_exits_zero(self, tmp_path: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["stage", str(FIXTURES / "flat-corpus"), "--out", str(tmp_path / "out")],
        )
        assert result.exit_code == 0, result.output
        assert "OK" in result.output

    def test_stage_cmd_creates_config_skeleton(self, tmp_path: Path) -> None:
        runner = CliRunner()
        out = tmp_path / "out"
        runner.invoke(cli, ["stage", str(FIXTURES / "flat-corpus"), "--out", str(out)])
        assert (out / "playbook.config.yaml").is_file()

    def test_stage_cmd_reports_layout(self, tmp_path: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["stage", str(FIXTURES / "clm-nested-corpus"), "--out", str(tmp_path / "out")],
        )
        assert "clm_nested" in result.output

    def test_stage_cmd_nonexistent_src_fails(self, tmp_path: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["stage", str(tmp_path / "no-such-dir")])
        assert result.exit_code != 0

    def test_stage_cmd_copy_flag_creates_real_files(self, tmp_path: Path) -> None:
        runner = CliRunner()
        out = tmp_path / "out"
        result = runner.invoke(
            cli,
            ["stage", str(FIXTURES / "flat-corpus"), "--out", str(out), "--copy"],
        )
        assert result.exit_code == 0, result.output
        staged_file = next((out / "deal-alpha").glob("01__*"))
        assert not staged_file.is_symlink()


# ---------------------------------------------------------------------------
# Default staging output path (issue #135)
# ---------------------------------------------------------------------------
#
# The default used to be /tmp/pbe-staging — world-readable, so any local user
# could enumerate staged corpus entries (even as symlinks, filenames alone can
# leak information). It must default to a user-owned path instead.


class TestDefaultStagingRoot:
    def test_default_staging_root_not_under_tmp(self) -> None:
        assert "/tmp" not in str(DEFAULT_STAGING_ROOT), (
            f"DEFAULT_STAGING_ROOT must not be under world-readable /tmp: {DEFAULT_STAGING_ROOT}"
        )

    def test_default_staging_root_under_user_home(self) -> None:
        assert str(DEFAULT_STAGING_ROOT).startswith(str(Path.home())), (
            "DEFAULT_STAGING_ROOT must be a user-owned path under the home directory, "
            f"got: {DEFAULT_STAGING_ROOT}"
        )

    def test_stage_cmd_without_out_uses_default_staging_root(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``playbook stage`` with no ``--out`` stages under DEFAULT_STAGING_ROOT.

        Patches the constant to a tmp_path so the test never touches the real
        user home directory.
        """
        import playbook_engine.staging as staging_module

        fake_root = tmp_path / "fake-cache" / "playbook-engine" / "staging"
        monkeypatch.setattr(staging_module, "DEFAULT_STAGING_ROOT", fake_root)

        runner = CliRunner()
        result = runner.invoke(cli, ["stage", str(FIXTURES / "flat-corpus")])
        assert result.exit_code == 0, result.output

        staged = fake_root / "flat-corpus"
        assert staged.is_dir()
        assert (staged / "playbook.config.yaml").is_file()
