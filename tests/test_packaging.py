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
"""

from __future__ import annotations

import json
import subprocess
import sys
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
