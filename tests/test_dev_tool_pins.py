"""Issue #263: ruff and mypy are pinned to exact versions in the dev extras.

An unpinned newer ruff turned ``main`` red on a rule the local ruff passed;
CI installs ``.[dev]``, so the pin lives there and local, hook and CI agree.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

PYPROJECT = Path(__file__).parent.parent / "pyproject.toml"


def _dev_requirement(name: str) -> str:
    extras = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"][
        "optional-dependencies"
    ]
    matches = [r for r in extras["dev"] if re.match(rf"{re.escape(name)}\b", r)]
    assert len(matches) == 1, f"expected exactly one {name} requirement in the dev extras"
    return matches[0]


@pytest.mark.parametrize("tool", ["ruff", "mypy"])
def test_lint_tools_are_pinned_exactly(tool: str) -> None:
    assert re.fullmatch(rf"{tool}==\d+(\.\d+)+", _dev_requirement(tool))
