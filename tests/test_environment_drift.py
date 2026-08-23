"""Prevention tests for the three ways the environment silently drifted.

Each class here corresponds to one observed incident, and asserts the property
that makes it structurally hard to repeat — not merely that a warning is
printed somewhere.

1. A staged corpus of absolute symlinks read as EMPTY under
   ``docker run -v "$CORPUS:/work/corpus:ro"``, and the linter blamed the file
   formats. Prevention: staging writes real copies by default, and broken
   symlinks are named as such when they do occur.
2. ``docling`` vanished from a host venv between runs; the next derivation
   silently used the legacy extractor and quarantined 43 of 44 documents.
   Prevention: the preflight is a precondition of ``mine``/``segment``, not a
   separate command an ad-hoc invocation can skip, and the external
   environment is declared in one checkable place.
3. A local Docker image three days stale held engine 0.2.0 against a 1.0.1
   source tree. Prevention: the image is stamped and ``make docker-run``
   compares the stamp to the source tree before starting a container.

No real corpus content is used anywhere — synthetic fixtures only.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest
from click.testing import CliRunner

from playbook_engine import __version__
from playbook_engine.cli import cli
from playbook_engine.corpus_linter import lint_corpus, scan_symlinks
from playbook_engine.environment import (
    EXTERNAL_TOOLS,
    ImageStamp,
    probe_environment,
    read_image_stamp,
)
from playbook_engine.staging import stage

REPO_ROOT = Path(__file__).resolve().parent.parent


def _write_docx_stub(path: Path, text: str = "synthetic") -> None:
    """Write a placeholder file with a supported extension.

    Contents are irrelevant to every assertion in this module — nothing here
    extracts a document; the checks under test are all about which files the
    walker can *see*.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _minimal_config(path: Path) -> Path:
    path.write_text(
        "agreement_type:\n"
        "  id: synthetic\n"
        "  name: Synthetic\n"
        "taxonomy: builtin:cuad-base\n"
        "provenance:\n"
        "  our_party_aliases: [Acme]\n",
        encoding="utf-8",
    )
    return path


# ---------------------------------------------------------------------------
# 1. Symlink staging under a read-only container mount
# ---------------------------------------------------------------------------


class TestStagingDefaultsToRealFiles:
    def test_stage_default_produces_no_symlinks(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        _write_docx_stub(src / "deal-a" / "v1.docx")
        _write_docx_stub(src / "deal-a" / "v2.docx")

        stage(src, tmp_path / "staged")

        staged = [p for p in (tmp_path / "staged" / "deal-a").iterdir() if p.name != "hints.yaml"]
        assert staged, "nothing was staged"
        assert not any(p.is_symlink() for p in staged)

    def test_default_staged_corpus_is_readable_with_source_gone(self, tmp_path: Path) -> None:
        """Stand-in for the read-only container mount: the source path is not
        reachable, and the corpus must still be fully visible."""
        src = tmp_path / "src"
        _write_docx_stub(src / "deal-a" / "v1.docx")
        staged = tmp_path / "staged"
        stage(src, staged)

        # Simulate "the host path does not exist from in here".
        src.rename(tmp_path / "src-moved")

        report = lint_corpus(staged)
        assert not report.has_errors, [i.message for i in report.errors()]

    def test_cli_symlink_flag_is_the_opt_out(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        _write_docx_stub(src / "deal-a" / "v1.docx")
        runner = CliRunner()

        result = runner.invoke(cli, ["stage", str(src), "--out", str(tmp_path / "s1")])
        assert result.exit_code == 0, result.output
        assert "(copied)" in result.output

        result = runner.invoke(cli, ["stage", str(src), "--out", str(tmp_path / "s2"), "--symlink"])
        assert result.exit_code == 0, result.output
        assert "(symlinked)" in result.output
        assert (tmp_path / "s2" / "deal-a" / "01__v1.docx").is_symlink()

    def test_symlink_staging_says_plainly_what_it_costs(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        _write_docx_stub(src / "deal-a" / "v1.docx")
        result = CliRunner().invoke(
            cli, ["stage", str(src), "--out", str(tmp_path / "s"), "--symlink"]
        )
        assert "only works on this machine" in result.output
        assert "container" in result.output


class TestDanglingSymlinksAreNamed:
    """The corpus must never again present as "no supported files found" when
    the real problem is that every file is a broken symlink."""

    @staticmethod
    def _corpus_of_dangling_links(tmp_path: Path) -> Path:
        corpus = tmp_path / "corpus"
        (corpus / "deal-a").mkdir(parents=True)
        for name in ("01__v1.docx", "02__v2.docx"):
            (corpus / "deal-a" / name).symlink_to(tmp_path / "gone" / name)
        return corpus

    def test_scan_classifies_dangling(self, tmp_path: Path) -> None:
        corpus = self._corpus_of_dangling_links(tmp_path)
        scan = scan_symlinks(corpus, [corpus / "deal-a"])
        assert scan.dangling_count == 2
        assert scan.escaping_count == 0

    def test_scan_classifies_escaping_but_resolvable(self, tmp_path: Path) -> None:
        outside = tmp_path / "outside"
        _write_docx_stub(outside / "v1.docx")
        corpus = tmp_path / "corpus"
        (corpus / "deal-a").mkdir(parents=True)
        (corpus / "deal-a" / "01__v1.docx").symlink_to(outside / "v1.docx")

        scan = scan_symlinks(corpus, [corpus / "deal-a"])
        assert scan.escaping_count == 1
        assert scan.dangling_count == 0

    def test_lint_blames_the_symlinks_not_the_file_formats(self, tmp_path: Path) -> None:
        corpus = self._corpus_of_dangling_links(tmp_path)
        report = lint_corpus(corpus)

        codes = {i.code for i in report.errors()}
        assert "CORPUS_DANGLING_SYMLINKS" in codes
        # The mis-diagnoses this replaces.
        assert "NO_SUPPORTED_FILES" not in codes
        assert "DOC_NO_SUPPORTED_FILES" not in codes

    def test_message_names_the_container_cause_and_the_fix(self, tmp_path: Path) -> None:
        corpus = self._corpus_of_dangling_links(tmp_path)
        report = lint_corpus(corpus)
        msg = next(i.message for i in report.errors() if i.code == "CORPUS_DANGLING_SYMLINKS")
        assert "docker run" in msg
        assert "playbook stage" in msg
        assert "silently" in msg

    def test_escaping_links_warn_but_do_not_block_a_host_run(self, tmp_path: Path) -> None:
        outside = tmp_path / "outside"
        _write_docx_stub(outside / "v1.docx")
        _write_docx_stub(outside / "v2.docx")
        corpus = tmp_path / "corpus"
        (corpus / "deal-a").mkdir(parents=True)
        for name in ("v1.docx", "v2.docx"):
            (corpus / "deal-a" / f"01__{name}").symlink_to(outside / name)

        report = lint_corpus(corpus)
        assert not report.has_errors, [i.message for i in report.errors()]
        assert "CORPUS_SYMLINKS_ESCAPE_ROOT" in {i.code for i in report.warnings()}

    def test_a_truly_empty_folder_still_says_so(self, tmp_path: Path) -> None:
        """The suppression must be scoped to the symlink case — an ordinary
        empty document folder keeps its original, correct error."""
        corpus = tmp_path / "corpus"
        (corpus / "deal-a").mkdir(parents=True)
        (corpus / "deal-a" / "notes.txt").write_text("nothing usable", encoding="utf-8")

        report = lint_corpus(corpus)
        assert "DOC_NO_SUPPORTED_FILES" in {i.code for i in report.errors()}


# ---------------------------------------------------------------------------
# 2. Preflight is a precondition, and the environment is declared
# ---------------------------------------------------------------------------


class TestPreflightIsAPrecondition:
    """``lint-corpus`` existed but was optional, so an ad-hoc ``mine`` skipped
    it entirely. It now gates the two commands that read the corpus."""

    @pytest.mark.parametrize("command", ["mine", "segment"])
    def test_command_refuses_a_corpus_of_dangling_symlinks(
        self, tmp_path: Path, command: str
    ) -> None:
        corpus = tmp_path / "corpus"
        (corpus / "deal-a").mkdir(parents=True)
        (corpus / "deal-a" / "01__v1.docx").symlink_to(tmp_path / "gone.docx")
        config = _minimal_config(tmp_path / "playbook.config.yaml")

        result = CliRunner().invoke(
            cli,
            [command, str(corpus), "--config", str(config), "--out", str(tmp_path / "out")],
        )
        assert result.exit_code == 1, result.output
        assert "CORPUS_DANGLING" not in result.output  # codes are internal
        assert "symlink" in result.output.lower()
        # It stopped BEFORE doing any work.
        assert not (tmp_path / "out" / "observations.jsonl").exists()

    @pytest.mark.parametrize("command", ["mine", "segment"])
    def test_skip_preflight_is_available_and_says_so(self, tmp_path: Path, command: str) -> None:
        corpus = tmp_path / "corpus"
        (corpus / "deal-a").mkdir(parents=True)
        (corpus / "deal-a" / "01__v1.docx").symlink_to(tmp_path / "gone.docx")
        config = _minimal_config(tmp_path / "playbook.config.yaml")

        result = CliRunner().invoke(
            cli,
            [
                command,
                str(corpus),
                "--config",
                str(config),
                "--out",
                str(tmp_path / "out"),
                "--skip-preflight",
            ],
        )
        assert "preflight: skipped" in result.output
        # It got past the gate — whatever it fails on afterwards is not the gate.
        assert "Preflight found" not in result.output

    def test_declared_docling_without_docling_blocks_mine(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The docling-vanished incident, prevented: with the extractor
        declared, a host that has lost the binary cannot start a run."""
        monkeypatch.setattr("shutil.which", lambda name: None)
        corpus = tmp_path / "corpus"
        _write_docx_stub(corpus / "deal-a" / "01__v1.docx")
        _write_docx_stub(corpus / "deal-a" / "02__v2.docx")
        config = tmp_path / "playbook.config.yaml"
        config.write_text(
            "agreement_type:\n"
            "  id: synthetic\n"
            "  name: Synthetic\n"
            "taxonomy: builtin:cuad-base\n"
            "provenance:\n"
            "  our_party_aliases: [Acme]\n"
            "segmentation:\n"
            "  agent: true\n"
            "extraction:\n"
            "  extractor: docling\n",
            encoding="utf-8",
        )

        result = CliRunner().invoke(
            cli, ["mine", str(corpus), "--config", str(config), "--out", str(tmp_path / "out")]
        )
        assert result.exit_code == 1, result.output
        assert "docling" in result.output
        assert not (tmp_path / "out" / "observations.jsonl").exists()

    def test_scaffolded_config_explains_the_extractor_choice(self, tmp_path: Path) -> None:
        """A declared extractor is what converts the silent downgrade into a
        refusal — so the scaffold has to put that choice in front of people."""
        from playbook_engine.staging import scaffold_config

        src = tmp_path / "src"
        _write_docx_stub(src / "deal-a" / "v1.docx")
        out = tmp_path / "out"
        out.mkdir()
        scaffold_config(src, out)

        text = (out / "playbook.config.yaml").read_text(encoding="utf-8")
        assert "extractor: docling" in text
        assert "silently" in text


class TestEnvironmentIsDeclared:
    def test_every_tool_the_engine_shells_out_to_is_declared(self) -> None:
        """``docling`` and ``pandoc`` are resolved by ``shutil.which`` inside
        extraction.py. If a new one appears there, it belongs in the table."""
        declared = {t.name for t in EXTERNAL_TOOLS}
        source = (REPO_ROOT / "playbook_engine" / "extraction.py").read_text(encoding="utf-8")
        looked_up = set(re.findall(r'shutil\.which\(\s*["\'](\w[\w.-]*)["\']', source))
        assert looked_up <= declared, f"undeclared external tool(s): {looked_up - declared}"

    def test_every_declared_tool_says_what_its_absence_costs(self) -> None:
        for tool in EXTERNAL_TOOLS:
            assert tool.purpose and tool.consequence and tool.install, tool.name

    def test_probe_never_raises_and_reports_the_running_engine(self) -> None:
        env = probe_environment(stamp_path=Path("/nonexistent/stamp.json"))
        assert env.engine_version == __version__
        assert env.image is None
        assert env.in_project_image is False
        assert {s.tool.name for s in env.tools} == {t.name for t in EXTERNAL_TOOLS}

    def test_api_key_presence_is_reported_without_the_value(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret-value")
        env = probe_environment(stamp_path=Path("/nonexistent/stamp.json"))
        assert env.anthropic_key_set is True
        assert "sk-ant-secret-value" not in repr(env)

    def test_doctor_runs_and_names_the_engine(self) -> None:
        result = CliRunner().invoke(cli, ["doctor"])
        assert f"engine   : {__version__}" in result.output

    def test_doctor_strict_fails_when_a_tool_is_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("shutil.which", lambda name: None)
        result = CliRunner().invoke(cli, ["doctor", "--strict"])
        assert result.exit_code == 1
        assert "docling" in result.output

    def test_doctor_lenient_passes_on_a_bare_host(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A host with no optional tools is usable, so plain ``doctor`` must not
        cry wolf — it explains the cost and exits 0."""
        monkeypatch.setattr("shutil.which", lambda name: None)
        result = CliRunner().invoke(cli, ["doctor"])
        assert result.exit_code == 0, result.output
        assert "optional tool(s) missing" in result.output


# ---------------------------------------------------------------------------
# 3. Stale Docker image
# ---------------------------------------------------------------------------


class TestImageStamp:
    def test_stamp_round_trips(self, tmp_path: Path) -> None:
        stamp = tmp_path / "stamp.json"
        stamp.write_text('{"engine_version": "9.9.9", "git_sha": "deadbee"}', encoding="utf-8")
        assert read_image_stamp(stamp) == ImageStamp(engine_version="9.9.9", git_sha="deadbee")

    def test_missing_or_malformed_stamp_is_not_an_error(self, tmp_path: Path) -> None:
        assert read_image_stamp(tmp_path / "absent.json") is None
        bad = tmp_path / "bad.json"
        bad.write_text("not json at all", encoding="utf-8")
        assert read_image_stamp(bad) is None

    def test_stamp_disagreeing_with_the_running_engine_is_detected(self, tmp_path: Path) -> None:
        stamp = tmp_path / "stamp.json"
        stamp.write_text('{"engine_version": "0.2.0", "git_sha": "abc1234"}', encoding="utf-8")
        env = probe_environment(stamp_path=stamp)
        assert env.in_project_image is True
        assert env.image_matches_engine() is False


class TestDockerfileStamping:
    """The stamp has to be produced by the build, and has to be verified
    against what pip actually installed — a build arg on its own can lie."""

    def test_dockerfile_declares_build_args_and_labels(self) -> None:
        text = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
        assert "ARG ENGINE_VERSION" in text
        assert "ARG ENGINE_GIT_SHA" in text
        assert "org.opencontainers.image.version" in text
        assert "org.opencontainers.image.revision" in text
        assert "/etc/playbook-engine-image.json" in text

    def test_dockerfile_refuses_a_build_arg_that_does_not_match_the_install(self) -> None:
        text = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
        assert "build refused" in text
        assert "playbook_engine.__version__" in text


class TestDockerRunIsGated:
    def test_makefile_reads_the_version_from_the_source_tree(self) -> None:
        makefile = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
        assert "ENGINE_VERSION :=" in makefile
        assert "playbook_engine/__init__.py" in makefile

    def test_docker_run_depends_on_docker_check(self) -> None:
        makefile = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
        assert re.search(r"^docker-run:\s+docker-check\s*$", makefile, re.M)

    def test_docker_build_passes_the_stamp_in(self) -> None:
        makefile = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
        assert "--build-arg ENGINE_VERSION=" in makefile
        assert "--build-arg ENGINE_GIT_SHA=" in makefile

    def test_makefile_version_matches_the_package(self) -> None:
        """The gate is only as good as the number it compares against.

        Runs the Makefile's own ``$(shell ...)`` command, lifted verbatim out of
        the Makefile, so a change to either side breaks this.
        """
        makefile = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
        match = re.search(r"^ENGINE_VERSION := \$\(shell (.+)\)$", makefile, re.M)
        assert match, "ENGINE_VERSION is no longer computed with $(shell ...)"
        out = subprocess.run(
            match.group(1), cwd=REPO_ROOT, capture_output=True, text=True, shell=True, check=True
        )
        assert out.stdout.strip() == __version__

    @pytest.mark.skipif(
        os.environ.get("PLAYBOOK_SKIP_MAKE_TESTS") == "1", reason="make tests disabled"
    )
    def test_docker_check_refuses_an_unlabelled_image(self) -> None:
        """No image, or an image with no version label, must fail closed —
        unknown provenance is not the same as verified-good. (Without Docker
        installed the target fails at the inspect step, which is the same
        refusal.)"""
        out = subprocess.run(
            ["make", "docker-check", "DOCKER_IMAGE=playbook-engine-does-not-exist"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        assert out.returncode != 0
        assert "make docker-build" in out.stdout + out.stderr

    @pytest.mark.skipif(
        os.environ.get("PLAYBOOK_SKIP_MAKE_TESTS") == "1", reason="make tests disabled"
    )
    def test_allow_stale_image_escape_hatch(self) -> None:
        out = subprocess.run(
            ["make", "docker-check", "ALLOW_STALE_IMAGE=1"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        assert out.returncode == 0
        assert "skipped" in out.stdout
