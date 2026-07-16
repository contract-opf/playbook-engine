"""Test for issue #176: ``playbook --version`` must report both the engine
version and the OPF version(s) it validates, so bug reports are unambiguous
about which OPF schema a given engine build understands.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from click.testing import CliRunner

from playbook_engine.cli import cli

_PYPROJECT = Path(__file__).parent.parent / "pyproject.toml"


def test_version_shows_engine_and_opf() -> None:
    pyproject_version = tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))["project"]["version"]

    runner = CliRunner()
    result = runner.invoke(cli, ["--version"])

    assert result.exit_code == 0
    assert pyproject_version in result.output
    assert "OPF" in result.output
