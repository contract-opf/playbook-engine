#!/usr/bin/env python3
"""Regenerate examples/canary/expected.json — the canary's committed values.

Runs the exact canary sequence ``tests/test_canary_corpus.py`` runs (it
imports that module's own helpers, so the two can never drift) into a
throwaway directory, reduces the result with the same ``measure()``, and
writes it out.

This is a **golden file**: regenerating it is a deliberate act whose whole
point is that the change shows up in ``git diff``. If a diff appears that you
did not intend, that is the canary doing its job — find out what moved before
committing the new numbers.

Run from the repo root, after build_corpus.py and build_verdicts.py::

    python examples/canary/build_expected.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from tests.test_canary_corpus import (  # noqa: E402
    _EXPECTED_PATH,
    measure,
    run_cold,
)


def main() -> None:
    os.environ.pop("ANTHROPIC_API_KEY", None)
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp) / "out"
        run_cold(out_dir)
        expected = measure(out_dir)

    _EXPECTED_PATH.write_text(
        json.dumps(expected, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {_EXPECTED_PATH}")
    print(json.dumps(expected, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
