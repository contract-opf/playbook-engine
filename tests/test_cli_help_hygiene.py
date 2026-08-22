"""CLI --help hygiene: no internal issue-number citations or internal
symbol/class names in anything a user sees via ``playbook <cmd> --help``.

1.0 ships to PyPI, which makes ``--help`` output and the adopter docs the
first thing a stranger reads. Internal issue numbers and dev jargon there
are a launch-quality bug, not a nit.

Convention (see the originating ticket): issue-number citations and
internal symbol/class names are fine in code comments and docstrings that
are NOT surfaced as ``--help`` text. They must never appear in text a user
sees by running ``playbook <cmd> --help`` — whether that text comes from an
explicit ``help=`` kwarg or (click's default behavior when no ``help=`` is
given) from the command's docstring. This test walks the RENDERED output,
not source locations, because a docstring-as-help command can leak a
citation that a naive ``grep`` over ``help=`` kwargs alone would miss.
"""

from __future__ import annotations

import re

import click
from click.testing import CliRunner

from playbook_engine.cli import cli

# A bare "#123" citation, or the "internal#123" spelling used for issues
# that predate the 2026-07-25 public-repo cutover.
_ISSUE_NUMBER_RE = re.compile(r"internal#\d+|(?<!\w)#\d+")

# Internal symbol/class names that mean nothing to a non-developer end user.
_FORBIDDEN_LITERALS = ("SegNode", "JudgmentCache", "mine_corpus", "no_cache")


def _violations(text: str) -> list[str]:
    """Return every forbidden pattern/literal found in `text` (possibly empty)."""
    hits = list(_ISSUE_NUMBER_RE.findall(text))
    hits.extend(literal for literal in _FORBIDDEN_LITERALS if literal in text)
    return hits


def _walk_commands(cmd: click.Command, path: list[str]):
    """Yield ``(path, command)`` for `cmd` and, recursively, every subcommand.

    Recurses into any ``click.Group`` (the current spelling of what click
    8's deprecated ``MultiCommand`` covered) by querying the command tree
    itself rather than a hand-enumerated list, so a newly added subcommand
    or group gets swept automatically.
    """
    yield path, cmd
    if isinstance(cmd, click.Group):
        ctx = click.Context(cmd)
        for name in cmd.list_commands(ctx):
            sub = cmd.get_command(ctx, name)
            if sub is not None:
                yield from _walk_commands(sub, [*path, name])


def _all_help_texts() -> dict[str, str]:
    """Render every command/subcommand's ``--help`` text, keyed by its path."""
    runner = CliRunner()
    texts: dict[str, str] = {}
    for path, _cmd in _walk_commands(cli, []):
        result = runner.invoke(cli, [*path, "--help"])
        assert result.exit_code == 0, (
            f"`playbook {' '.join(path)} --help` exited {result.exit_code}: {result.output}"
        )
        texts[" ".join(path) or "(root)"] = result.output
    return texts


def test_help_tree_has_no_forbidden_patterns() -> None:
    """Walk the full click command tree; no rendered --help text may contain
    an internal issue-number citation or an internal symbol/class name.
    """
    texts = _all_help_texts()
    failures = {name: v for name, text in texts.items() if (v := _violations(text))}
    assert not failures, (
        "internal issue numbers / symbol names leaked into --help output:\n"
        + "\n".join(f"  {name}: {sorted(set(v))}" for name, v in sorted(failures.items()))
    )


def test_help_tree_walk_reaches_every_known_command() -> None:
    """Sanity-check the walk isn't accidentally shallow — e.g. a subcommand
    whose ``--help`` errors out would otherwise be silently absent from the
    hygiene sweep above instead of failing loud.
    """
    texts = _all_help_texts()
    expected = {
        "(root)",
        "validate",
        "render-prompt",
        "resolve-citation",
        "publish",
        "taxonomy",
        "taxonomy merge",
        "mine",
        "project",
        "lint-corpus",
        "inspect",
        "stage",
        "judge",
        "judge-apply",
        "segment",
        "segment-apply",
        "induce-taxonomy",
        "report",
        "digest",
        "view",
        "view render",
        "view bundle",
        "view apply",
        "posture",
        "posture questions",
        "posture interview",
        "floor",
        "floor propose",
        "floor sign",
        "curate",
    }
    assert expected <= set(texts)


def test_violation_detector_has_teeth() -> None:
    """Prove `_violations` actually catches every forbidden pattern before
    trusting it to guard the real --help surface — the same class of leak
    (e.g. the ``SegNode`` in a ``help=`` kwarg) that motivated this file.
    """
    assert _violations("wires up a SegNode internally") == ["SegNode"]
    assert _violations("replays from the JudgmentCache") == ["JudgmentCache"]
    assert _violations("calls mine_corpus directly") == ["mine_corpus"]
    assert _violations("pass no_cache=True to force it") == ["no_cache"]
    assert _violations("fixed in issue #191") == ["#191"]
    assert _violations("see internal#209 for history") == ["internal#209"]
    assert _violations("a heading like '# Section' with no digits") == []
    assert _violations("plain behavioral language only") == []
