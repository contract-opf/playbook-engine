"""Regression tests for the wheel's ``spec/`` packaging (issue #182 / B1).

``config._BUILTIN_TAXONOMY_DIR`` and ``validator._SCHEMA_PATH_V1/V2`` are all
``Path(__file__).parent.parent / "spec"`` — correct only if ``spec/`` (the
JSON schemas and builtin taxonomies) actually ships alongside the installed
package. Every host dev install in this suite is editable (``pip install -e
.``), so ``spec/`` is just the repo's own ``spec/`` directory and this class
of bug is invisible to every other test here. It only bites a NON-editable
install (the Docker image's ``pip install /app``), where hatchling's default
wheel build silently omits ``spec/`` — ``builtin:`` taxonomies and
``validate``/``project`` crashed with ``FileNotFoundError`` in the container
even though the files sat right there at ``/app/spec``.

These tests build a real wheel with hatchling's own PEP 517 hook, pip-install
it non-editably into an isolated directory, and exercise the installed copy
in a subprocess. A fresh interpreter is required for the exercise: the
current test process already has the dev checkout's editable
``playbook_engine`` cached in ``sys.modules`` (re-importing wouldn't reload
from the target dir), and separately, the editable install registers an
import finder that shadows ``sys.path`` ordering — ``PYTHONPATH`` alone does
not win against it. Prepending the target dir via ``sys.path.insert(0, ...)``
as the very first statement of a fresh interpreter does.

``test_sdist_contains_spec_dir`` (issue #114) closes the other half of the
same gap: the PyPI publish workflow ships both an sdist and a wheel, and
``force-include`` is a wheel-only hatchling setting — nothing upstream of
this test ever asserted the sdist tarball itself carries ``spec/`` too, only
the wheel.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).parent.parent


@pytest.fixture(scope="module")
def packaged_install(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build the wheel and ``pip install --no-deps --target`` it into an
    isolated directory; return that directory.

    Built once per test module (the build + install is the expensive part)
    and shared read-only by every test below.
    """
    workdir = tmp_path_factory.mktemp("packaging")
    wheel_dir = workdir / "wheel"
    wheel_dir.mkdir()

    build_script = f"import hatchling.build as hb; print(hb.build_wheel({str(wheel_dir)!r}))"
    build = subprocess.run(
        [sys.executable, "-c", build_script],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert build.returncode == 0, build.stderr
    wheel_name = build.stdout.strip().splitlines()[-1]
    wheel_path = wheel_dir / wheel_name

    target = workdir / "target"
    install = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "--no-deps",
            "--target",
            str(target),
            str(wheel_path),
        ],
        capture_output=True,
        text=True,
    )
    assert install.returncode == 0, install.stderr
    return target


def _run_in_target(target: Path, script: str) -> dict[str, object]:
    """Run *script* in a fresh interpreter resolving ``playbook_engine``
    against *target* only, and return its one JSON line of stdout.
    """
    preamble = f"import sys; sys.path.insert(0, {str(target)!r})\n"
    result = subprocess.run(
        [sys.executable, "-c", preamble + script],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])  # type: ignore[no-any-return]


def test_wheel_contains_spec_dir(packaged_install: Path) -> None:
    """``spec/`` (schemas + builtin taxonomies) actually landed in the
    install, not just the source tree used to build it.
    """
    assert (packaged_install / "spec" / "playbook.schema-0.2.json").is_file()
    assert (packaged_install / "spec" / "playbook.schema.json").is_file()
    taxonomy_names = sorted(p.name for p in (packaged_install / "spec" / "taxonomy").glob("*.yaml"))
    assert "affiliation-agreement.yaml" in taxonomy_names


def test_sdist_contains_spec_dir(tmp_path_factory: pytest.TempPathFactory) -> None:
    """The sdist tarball — the other artifact the PyPI publish workflow
    ships (issue #114) — must also carry ``spec/``.

    ``force-include`` (pyproject.toml) only governs the wheel target;
    hatchling's sdist build includes the source tree by its own VCS/include
    rules, but nothing before this test ever actually opened the built
    tarball to confirm ``spec/`` survives into it too.
    """
    workdir = tmp_path_factory.mktemp("sdist")

    build_script = f"import hatchling.build as hb; print(hb.build_sdist({str(workdir)!r}))"
    build = subprocess.run(
        [sys.executable, "-c", build_script],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert build.returncode == 0, build.stderr
    sdist_name = build.stdout.strip().splitlines()[-1]
    sdist_path = workdir / sdist_name

    with tarfile.open(sdist_path, "r:gz") as tf:
        members = tf.getnames()

    # Members are prefixed with the sdist's top-level dir (e.g.
    # "playbook_engine-1.0.0/spec/..."). Anchor to that exact root so a
    # stray nested "spec/" elsewhere in the tree (e.g. a worktree copy)
    # cannot satisfy the assertion in place of the real top-level spec/.
    assert sdist_name.endswith(".tar.gz")
    root = sdist_name[: -len(".tar.gz")]
    assert f"{root}/spec/playbook.schema-0.2.json" in members, members
    assert f"{root}/spec/playbook.schema-0.3.json" in members, members
    assert f"{root}/spec/playbook.schema.json" in members, members
    assert f"{root}/spec/taxonomy/affiliation-agreement.yaml" in members, members


def test_sdist_excludes_claude_scratch_but_keeps_skills(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """The ``[tool.hatch.build.targets.sdist] exclude`` table (pyproject.toml,
    issue #114) must actually keep untracked ``.claude/`` scratch content —
    local worktree copies, ``settings.local.json`` — out of the sdist, while
    still shipping the git-tracked ``.claude/skills/`` content.

    On a clean CI checkout, neither ``.claude/worktrees/`` nor
    ``.claude/settings.local.json`` exists, so a bare "no worktrees member"
    assertion would pass vacuously even if the exclude table were deleted
    outright. This test creates real probe paths first — removing exactly
    what it created afterward, whether or not they already existed — so the
    assertion is non-vacuous on every checkout, local or CI.
    """
    claude_dir = _REPO_ROOT / ".claude"
    worktrees_dir = claude_dir / "worktrees"
    probe_worktree_file = worktrees_dir / "_sdist_exclude_probe.txt"
    settings_file = claude_dir / "settings.local.json"

    created_worktrees_dir = not worktrees_dir.exists()
    worktrees_dir.mkdir(parents=True, exist_ok=True)
    probe_worktree_file.write_text("probe", encoding="utf-8")

    created_settings_file = not settings_file.exists()
    if created_settings_file:
        settings_file.write_text("{}", encoding="utf-8")

    try:
        workdir = tmp_path_factory.mktemp("sdist-exclude")
        build_script = f"import hatchling.build as hb; print(hb.build_sdist({str(workdir)!r}))"
        build = subprocess.run(
            [sys.executable, "-c", build_script],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
        )
        assert build.returncode == 0, build.stderr
        sdist_name = build.stdout.strip().splitlines()[-1]
        sdist_path = workdir / sdist_name

        with tarfile.open(sdist_path, "r:gz") as tf:
            members = tf.getnames()

        assert sdist_name.endswith(".tar.gz")
        root = sdist_name[: -len(".tar.gz")]
        claude_prefix = f"{root}/.claude/"
        skills_prefix = f"{root}/.claude/skills/"

        leaked = [
            m for m in members if m.startswith(claude_prefix) and not m.startswith(skills_prefix)
        ]
        assert leaked == [], (
            "sdist exclude table (pyproject.toml) failed to keep untracked "
            f".claude/ scratch content out of the sdist: {leaked}"
        )

        # The fix must not regress into excluding ALL of .claude/ — the
        # git-tracked skill content still has to ship.
        assert f"{root}/.claude/skills/playbook-from-corpus/SKILL.md" in members, members
        assert f"{root}/.claude/skills/playbook-from-corpus/REFERENCE.md" in members, members
        assert f"{root}/.claude/skills/playbook-from-corpus/estimate_runtime.py" in members, members
    finally:
        probe_worktree_file.unlink(missing_ok=True)
        if created_worktrees_dir:
            worktrees_dir.rmdir()
        if created_settings_file:
            settings_file.unlink(missing_ok=True)


def test_builtin_taxonomy_resolves_from_packaged_install(
    packaged_install: Path, tmp_path: Path
) -> None:
    """``builtin:`` taxonomy resolution (config.py) must work against the
    package's own bundled spec/ when installed non-editably — this is
    exactly what broke ``mine``/``project`` in Docker.
    """
    config_yaml = tmp_path / "engine.yaml"
    config_yaml.write_text(
        "agreement_type:\n"
        "  id: test-type\n"
        '  name: "Test Agreement"\n'
        "baseline:\n"
        "  template: null\n"
        "taxonomy: builtin:affiliation-agreement.yaml\n",
        encoding="utf-8",
    )
    script = f"""
import json
from pathlib import Path
from playbook_engine.config import load_config
cfg = load_config(Path({str(config_yaml)!r}))
print(json.dumps({{
    "taxonomy_path": str(cfg.taxonomy_path),
}}))
"""
    out = _run_in_target(packaged_install, script)
    resolved = Path(out["taxonomy_path"])  # type: ignore[arg-type]
    assert resolved.is_relative_to(packaged_install), out
    assert resolved.is_file()


def test_schema_loads_from_packaged_install(packaged_install: Path) -> None:
    """The OPF schema files (validator.py) must load from the package's own
    spec/ when installed non-editably — ``validate`` crashed with
    ``FileNotFoundError`` in Docker before B1.
    """
    script = """
import json
from playbook_engine.validator import validate_document
# Intentionally minimal/invalid doc: only schema *loading* is under test
# here (the FileNotFoundError B1 caused), not full document validity.
result = validate_document({"opf_version": "0.2"})
print(json.dumps({"messages": [e.message for e in result.errors]}))
"""
    out = _run_in_target(packaged_install, script)
    assert isinstance(out["messages"], list)
