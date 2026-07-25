"""Executable-documentation test for the Quickstart — issue #173.

``examples/README.md`` documents a copy-pasteable clone -> compiled-playbook
walkthrough over the committed ``examples/judge-fixture/`` corpus, using the
fixture's pre-computed ("canned") judge verdicts so the whole path runs
without an ``ANTHROPIC_API_KEY``. This test makes that walkthrough
rot-proof: it parses the actual ``sh`` command blocks out of the README
(between the ``<!-- quickstart:start -->`` / ``<!-- quickstart:end -->``
markers — the one-time venv/install step and the Docker variant live
outside that region deliberately, since neither should be auto-executed
here) and replays them for real, in a temp directory, against the repo's
own installed CLI.

Two things are asserted:

1. ``test_quickstart_commands_run`` — every documented command exits 0 with
   ``ANTHROPIC_API_KEY`` absent from the subprocess environment, and the
   playbook that comes out the other end passes ``validate_document``.
2. ``test_quickstart_expected_output_markers`` — the "expected output" text
   blocks shown in the README actually match what the commands print, so a
   change to CLI flags/output text breaks this test until the README is
   updated to match, rather than silently going stale.
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest

from playbook_engine.validator import load_opf_file, validate_document

_REPO_ROOT = Path(__file__).parent.parent
_README = _REPO_ROOT / "examples" / "README.md"
_START_MARKER = "<!-- quickstart:start -->"
_END_MARKER = "<!-- quickstart:end -->"

# The literal output-directory string used throughout examples/README.md's
# quickstart commands. Substituted for a pytest tmp_path before execution so
# the test never writes into the real repo tree (and never collides with a
# concurrent run) while every other argument stays exactly as documented.
_OUT_PLACEHOLDER = "out/quickstart-demo"

# Output substrings that must appear BOTH in examples/README.md's "expected
# output" blocks AND in what the corresponding command actually printed.
# Keeps the documented output honest without pinning the whole file's exact
# text (timestamps/paths vary; these substrings do not).
_EXPECTED_MARKERS = [
    "no errors, 2 warning(s)",  # lint-corpus
    "loaded 11 verdict(s)",  # judge-apply
    "L1-L4 complete: 7 observations, 2 docs",  # mine
    "Playbook written:",  # project
]


def _extract_quickstart_commands(readme_text: str) -> list[str]:
    """Return the shell commands inside the quickstart marker region.

    Only fenced ```sh blocks are collected (the "expected output" blocks are
    fenced as ```text and are never treated as commands to run). Blank lines
    and ``#`` comment lines within a block are dropped.
    """
    assert _START_MARKER in readme_text, f"{_START_MARKER!r} not found in {_README}"
    assert _END_MARKER in readme_text, f"{_END_MARKER!r} not found in {_README}"
    start = readme_text.index(_START_MARKER)
    end = readme_text.index(_END_MARKER)
    region = readme_text[start:end]

    commands: list[str] = []
    for block in re.findall(r"```sh\n(.*?)```", region, flags=re.DOTALL):
        for line in block.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            commands.append(line)
    return commands


@dataclass
class _StepResult:
    command: str
    returncode: int
    stdout: str
    stderr: str


@dataclass
class _QuickstartRun:
    steps: list[_StepResult]
    out_dir: Path


@pytest.fixture(scope="module")
def _readme_text() -> str:
    assert _README.exists(), f"{_README} not found — issue #173's Quickstart doc is missing"
    return _README.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def quickstart_run(tmp_path_factory: pytest.TempPathFactory, _readme_text: str) -> _QuickstartRun:
    """Execute every documented quickstart command once; share results across tests."""
    tmp_dir = tmp_path_factory.mktemp("quickstart")
    out_dir = tmp_dir / "quickstart-demo"

    commands = _extract_quickstart_commands(_readme_text)
    assert commands, "no ```sh command blocks found in the quickstart marker region"

    # No ANTHROPIC_API_KEY in the subprocess environment — the whole point of
    # the canned-verdicts fixture is that the first-run path never needs one.
    env = dict(os.environ)
    env.pop("ANTHROPIC_API_KEY", None)

    steps: list[_StepResult] = []
    for raw_cmd in commands:
        cmd = raw_cmd.replace(_OUT_PLACEHOLDER, str(out_dir))
        proc = subprocess.run(  # noqa: S603
            shlex.split(cmd),
            cwd=_REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        steps.append(
            _StepResult(
                command=cmd,
                returncode=proc.returncode,
                stdout=proc.stdout,
                stderr=proc.stderr,
            )
        )
        assert proc.returncode == 0, (
            f"quickstart command failed (exit {proc.returncode}): {cmd}\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )

    return _QuickstartRun(steps=steps, out_dir=out_dir)


def test_quickstart_commands_run(quickstart_run: _QuickstartRun) -> None:
    """AC-1: every documented command runs clean and the resulting playbook validates."""
    assert len(quickstart_run.steps) >= 6, (
        f"expected at least 6 quickstart commands, got {len(quickstart_run.steps)}"
    )

    playbook_path = quickstart_run.out_dir / "playbook.opf.json"
    assert playbook_path.exists(), f"quickstart did not produce {playbook_path}"

    doc = load_opf_file(playbook_path)
    result = validate_document(doc)
    blocking = [str(e) for e in result.errors if e.blocking]
    assert result.ok, f"quickstart-produced playbook fails validation: {blocking}"

    # The canned verdicts cover every clause in the fixture — the whole point
    # of pre-loading the verdict store before the first `mine` — so no
    # observation should be left dangling as needs_review.
    obs_path = quickstart_run.out_dir / "observations.jsonl"
    assert obs_path.exists()
    import json  # noqa: PLC0415

    obs_lines = [
        json.loads(line)
        for line in obs_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert obs_lines, "observations.jsonl is empty"
    still_needs_review = [o for o in obs_lines if o.get("basis") == "needs_review"]
    assert still_needs_review == [], (
        f"quickstart should leave 0 needs_review observations (canned verdicts cover "
        f"the whole fixture); found {len(still_needs_review)}"
    )

    # Schema-valid is NOT enough: the walkthrough's payoff (viewer HTML,
    # render-prompt) is only real if the playbook actually compiled clause
    # positions. This exact gap shipped once — the fixture had no detectable
    # signed copy, every observation landed outcome="unsigned", the position
    # compiler withheld all of them, and the "validating" playbook rendered a
    # blank viewer page (issue #200). The fixture's per-deal hints.yaml
    # (signed_version) is what keeps this populated.
    clauses = doc.get("evidence", {}).get("clauses", [])
    assert clauses, (
        "quickstart-produced playbook has zero evidence.clauses — schema-valid "
        "but semantically empty (viewer renders a blank page; see issue #200)"
    )


def test_quickstart_expected_output_markers(
    _readme_text: str, quickstart_run: _QuickstartRun
) -> None:
    """AC-2: the README's documented expected output matches what the commands
    actually printed, so drift in CLI output text breaks this test rather than
    silently rotting the docs."""
    combined_stdout = "\n".join(step.stdout for step in quickstart_run.steps)

    for marker in _EXPECTED_MARKERS:
        assert marker in _readme_text, (
            f"examples/README.md is missing documented marker: {marker!r}"
        )
        assert marker in combined_stdout, (
            f"actual quickstart run output no longer contains: {marker!r} — "
            f"examples/README.md's expected output has drifted from reality"
        )

    # The README must show the run's document/clause counts and confirm
    # the playbook validates, not just link to the validate command.
    assert "2 docs" in _readme_text
    assert re.search(r"\bvalid\b", _readme_text, flags=re.IGNORECASE)
