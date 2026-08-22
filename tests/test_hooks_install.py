"""Regression tests for ``make hooks`` (issue #116, fix round 1).

``make hooks`` must install the tracked lint/format pre-push gate
(``scripts/pre-push-lint.sh``) at git's REAL hooks directory — the one
``git rev-parse --git-path hooks`` resolves, not the literal path
``.git/hooks`` — because that literal path is wrong in two configurations
that are otherwise invisible until someone actually tries to push from
them:

* a linked worktree, where ``.git`` is a FILE (not a directory), so
  ``mkdir -p .git/hooks`` fails outright; and
* a clone with ``core.hooksPath`` set, where git never even looks at
  ``.git/hooks`` — a hook copied there is installed-looking but inert
  (a silent false success).

These tests build a real throwaway git repo (a copy of the tracked
``Makefile`` and ``scripts/pre-push-lint.sh``, nothing else) so the actual
Makefile recipe under test is exercised end to end, not a reimplementation
of it.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).parent.parent


def _git(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    assert result.returncode == 0, f"git {args} failed: {result.stderr}"
    return result


def _hooks_dir(cwd: Path) -> Path:
    """Resolve the REAL hooks directory git will use from *cwd* — the same
    resolution ``make hooks`` and ``ignore/git-hooks/install.sh`` use.
    """
    out = _git(["rev-parse", "--git-path", "hooks"], cwd).stdout.strip()
    path = Path(out)
    return path if path.is_absolute() else cwd / path


def _make_hooks(cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["make", "hooks"], cwd=cwd, capture_output=True, text=True)


def _resolve_ruff() -> Path:
    """Locate a real ruff binary to seed into each throwaway repo's own
    ``.venv/bin/ruff`` (issue #116 fix round 2, finding 4): so that
    ``scripts/pre-push-lint.sh``'s FIRST resolution branch
    (``$repo_root/.venv/bin/ruff``) succeeds inside the throwaway repo on
    its own — the "installed hook blocks a violation" tests must not pass
    only because the *calling* process's PATH happens to already carry a
    venv's ruff.
    """
    venv_ruff = _REPO_ROOT / ".venv" / "bin" / "ruff"
    if venv_ruff.is_file():
        return venv_ruff
    found = shutil.which("ruff")
    if found is not None:
        return Path(found)
    pytest.fail(
        f"no ruff binary found (checked {venv_ruff} and PATH) — cannot "
        "seed a throwaway repo that can resolve ruff on its own"
    )


def _seed_repo(root: Path) -> None:
    """Create a throwaway git repo at *root* containing just the tracked
    Makefile and scripts/pre-push-lint.sh — the two files ``make hooks``
    actually needs — with an initial commit.
    """
    root.mkdir(parents=True)
    _git(["init", "-q"], root)
    _git(["config", "user.email", "test@example.com"], root)
    _git(["config", "user.name", "Test"], root)
    (root / "scripts").mkdir()
    shutil.copy(
        _REPO_ROOT / "scripts" / "pre-push-lint.sh",
        root / "scripts" / "pre-push-lint.sh",
    )
    (root / "scripts" / "pre-push-lint.sh").chmod(0o755)
    shutil.copy(_REPO_ROOT / "Makefile", root / "Makefile")
    (root / "README.md").write_text("throwaway test repo\n", encoding="utf-8")
    _git(["add", "-A"], root)
    _git(["commit", "-q", "-m", "init"], root)
    # Seed a resolvable ruff (see _resolve_ruff) independent of the calling
    # process's PATH — left untracked, like a real .venv.
    venv_bin = root / ".venv" / "bin"
    venv_bin.mkdir(parents=True)
    os.symlink(_resolve_ruff(), venv_bin / "ruff")


@pytest.fixture
def hooks_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    _seed_repo(repo)
    return repo


def test_make_hooks_installs_at_resolved_hooks_path(hooks_repo: Path) -> None:
    """(i) the hook lands exactly where ``git rev-parse --git-path hooks``
    says it should, not at the literal string ``.git/hooks``.
    """
    result = _make_hooks(hooks_repo)
    assert result.returncode == 0, result.stderr

    dest = _hooks_dir(hooks_repo) / "pre-push"
    assert dest.is_file(), f"hook not found at resolved path {dest}"
    assert os.access(dest, os.X_OK), f"hook at {dest} is not executable"
    assert dest.read_bytes() == (hooks_repo / "scripts" / "pre-push-lint.sh").read_bytes()


def test_installed_hook_blocks_a_seeded_format_violation(hooks_repo: Path) -> None:
    """(ii) the hook that ``make hooks`` installs actually fires: a seeded
    ruff-format violation makes it exit non-zero — and specifically because
    ruff caught ``bad.py``'s formatting, not because of some unrelated
    failure mode (e.g. pre-push-lint.sh's separate "ruff not found"
    fail-closed branch, which also prints "BLOCKED" — issue #116 fix round
    2, finding 4). ``_seed_repo`` guarantees this throwaway repo can
    resolve a real ruff on its own, so this can't pass for the wrong
    reason either.
    """
    result = _make_hooks(hooks_repo)
    assert result.returncode == 0, result.stderr

    (hooks_repo / "bad.py").write_text("x=1\n", encoding="utf-8")

    hook = _hooks_dir(hooks_repo) / "pre-push"
    run = subprocess.run([str(hook)], cwd=hooks_repo, capture_output=True, text=True)
    assert run.returncode != 0, (
        f"installed hook did not block a seeded format violation; "
        f"stdout={run.stdout!r} stderr={run.stderr!r}"
    )
    assert "BLOCKED" in run.stderr
    assert "[fmt]" in run.stderr, run.stderr
    assert "bad.py" in run.stderr, run.stderr


def test_installed_hook_passes_clean_tree(hooks_repo: Path) -> None:
    """Sanity counterpart to the above: a clean tree is NOT blocked, so the
    non-zero exit above is attributable to the seeded violation.
    """
    result = _make_hooks(hooks_repo)
    assert result.returncode == 0, result.stderr

    hook = _hooks_dir(hooks_repo) / "pre-push"
    run = subprocess.run([str(hook)], cwd=hooks_repo, capture_output=True, text=True)
    assert run.returncode == 0, f"clean tree unexpectedly blocked: {run.stderr}"


def test_make_hooks_refuses_without_clobbering_a_richer_hook(hooks_repo: Path) -> None:
    """(iii) when a richer hook — one that genuinely DELEGATES to
    scripts/pre-push-lint.sh (here, alongside its own secrets/confidential
    scanning) — is already installed, ``make hooks`` leaves it untouched —
    and this refusal is a successful no-op (exit 0), not a build failure, so
    it composes in a chain like ``make install hooks``. A hook that only
    *scans for* secrets/confidential terms WITHOUT a genuine delegation line
    has zero lint/format coverage of its own and is refused instead, not
    accepted as equivalent — see
    ``test_make_hooks_refuses_a_secrets_only_hook_without_lint_delegation``
    below.
    """
    dest = _hooks_dir(hooks_repo) / "pre-push"
    dest.parent.mkdir(parents=True, exist_ok=True)
    richer = (
        "#!/usr/bin/env bash\n"
        "# richer local hook: also scans SECRET_PATTERNS before delegating\n"
        f'exec "{hooks_repo}/scripts/pre-push-lint.sh"\n'
    )
    dest.write_text(richer, encoding="utf-8")
    dest.chmod(0o755)

    result = _make_hooks(hooks_repo)
    assert result.returncode == 0, (
        f"refusal-to-clobber must be a successful no-op, not a make error: {result.stderr}"
    )
    assert dest.read_text(encoding="utf-8") == richer, "richer hook was overwritten"


def test_make_hooks_does_not_treat_a_comment_only_mention_as_delegating(
    hooks_repo: Path,
) -> None:
    """Regression for issue #116 fix round 3, finding 1: the "already
    delegates" guard used to be ``grep -qE 'pre-push-lint\\.sh' "$dest"``
    over the WHOLE file, so any *mention* of the filename counted as
    delegation — including one that appears only inside a comment. A hook
    whose entire body is ``#!/usr/bin/env bash`` + a
    ``# TODO(someday): delegate to scripts/pre-push-lint.sh`` comment +
    ``exit 0`` has zero lint/format coverage of its own, but the old guard
    made ``make hooks`` print "already delegates to
    scripts/pre-push-lint.sh", exit 0, and install nothing — leaving the
    clone with no lint/format gate at all while reporting success. ``make
    hooks`` must instead recognize this as an ordinary non-delegating hook:
    back it up and install the real tracked gate, so the clone ends up
    genuinely protected.
    """
    dest = _hooks_dir(hooks_repo) / "pre-push"
    dest.parent.mkdir(parents=True, exist_ok=True)
    comment_only = (
        "#!/usr/bin/env bash\n# TODO(someday): delegate to scripts/pre-push-lint.sh\nexit 0\n"
    )
    dest.write_text(comment_only, encoding="utf-8")
    dest.chmod(0o755)

    result = _make_hooks(hooks_repo)
    assert result.returncode == 0, result.stderr
    assert "already delegates" not in result.stdout + result.stderr, (
        "a comment-only mention of pre-push-lint.sh must not be accepted as "
        f"genuine delegation: stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert dest.read_text(encoding="utf-8") != comment_only, (
        "the non-delegating stub must be replaced with the real gate, not "
        "left in place as a false no-op success"
    )

    (hooks_repo / "bad.py").write_text("x=1\n", encoding="utf-8")
    run = subprocess.run([str(dest)], cwd=hooks_repo, capture_output=True, text=True)
    assert run.returncode != 0, (
        "the clone must end up genuinely gated once make hooks replaces the "
        f"non-delegating stub; stdout={run.stdout!r} stderr={run.stderr!r}"
    )
    assert "BLOCKED" in run.stderr


def test_make_hooks_does_not_treat_an_echo_only_mention_as_delegating(
    hooks_repo: Path,
) -> None:
    """Regression for issue #116 fix round 4, finding 1: the round-3 guard
    (``grep -qE '^[[:space:]]*[^#]*pre-push-lint\\.sh'``) excludes
    comment-only mentions but still accepts ANY other non-comment mention —
    including one that only ever gets printed, never executed. A hook whose
    fallback branch just echoes a message *about* the script (verbatim from
    the real maintainer hook, ``ignore/git-hooks/pre-push:124``) has zero
    lint/format coverage of its own, but the round-3 guard still made
    ``make hooks`` print "already delegates to scripts/pre-push-lint.sh",
    exit 0, and install nothing — leaving the clone with no lint/format gate
    at all while reporting success. ``make hooks`` must instead recognize an
    echo/printf line as text *about* the script, not a call to it: back the
    stub up and install the real tracked gate, so the clone ends up
    genuinely protected. This must fail against the pre-fix Makefile:69
    regex and pass only once that guard requires a real invocation.
    """
    dest = _hooks_dir(hooks_repo) / "pre-push"
    dest.parent.mkdir(parents=True, exist_ok=True)
    echo_only = (
        "#!/usr/bin/env bash\n"
        'echo "pre-push: scripts/pre-push-lint.sh not found — skipping '
        'lint/format gate; CI remains the backstop." >&2\n'
    )
    dest.write_text(echo_only, encoding="utf-8")
    dest.chmod(0o755)

    result = _make_hooks(hooks_repo)
    assert result.returncode == 0, result.stderr
    assert "already delegates" not in result.stdout + result.stderr, (
        "a printed message that only mentions pre-push-lint.sh must not be "
        f"accepted as genuine delegation: stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert dest.read_text(encoding="utf-8") != echo_only, (
        "the non-delegating stub must be replaced with the real gate, not "
        "left in place as a false no-op success"
    )

    (hooks_repo / "bad.py").write_text("x=1\n", encoding="utf-8")
    run = subprocess.run([str(dest)], cwd=hooks_repo, capture_output=True, text=True)
    assert run.returncode != 0, (
        "the clone must end up genuinely gated once make hooks replaces the "
        f"non-delegating stub; stdout={run.stdout!r} stderr={run.stderr!r}"
    )
    assert "BLOCKED" in run.stderr


def test_make_hooks_refuses_a_secrets_only_hook_without_lint_delegation(
    hooks_repo: Path,
) -> None:
    """(v) regression for the false-equivalence bug (issue #116 fix round
    2, findings 1 & 2): a hook that scans for secrets/confidential terms
    but does NOT delegate to scripts/pre-push-lint.sh has zero lint/format
    coverage of its own — it is not "already at least as strong" as the
    gate this target installs. Treating it as equivalent (the pre-fix
    behavior) let ``make hooks`` report success while silently leaving the
    repo with no lint/format gate at all. ``make hooks`` must instead
    refuse loudly (non-zero exit) and must not touch the existing hook.
    """
    dest = _hooks_dir(hooks_repo) / "pre-push"
    dest.parent.mkdir(parents=True, exist_ok=True)
    secrets_only = (
        "#!/usr/bin/env bash\n"
        "# scans for CONFIDENTIAL_PATTERNS and SECRET_PATTERNS\n"
        "echo secrets-only-hook\n"
    )
    dest.write_text(secrets_only, encoding="utf-8")
    dest.chmod(0o755)

    result = _make_hooks(hooks_repo)
    assert result.returncode != 0, (
        "make hooks must not silently succeed when the existing hook scans "
        "for secrets but has no lint/format delegation "
        f"(stdout={result.stdout!r} stderr={result.stderr!r})"
    )
    assert "already at least as strong" not in result.stderr, (
        "must not claim equivalence with a hook that has zero lint/format "
        f"coverage of its own: {result.stderr!r}"
    )
    assert dest.read_text(encoding="utf-8") == secrets_only, (
        "the non-delegating hook must be left exactly as-is on refusal, not silently replaced"
    )

    # Follow the refusal's own instructions: add a line that genuinely
    # invokes scripts/pre-push-lint.sh (not just a comment mentioning the
    # filename). make hooks must then accept it as equivalent (exit 0,
    # no-op) — and, critically, the EFFECTIVE installed hook must actually
    # block a seeded format violation once it genuinely delegates, proving
    # the composition rule holds in practice and not just at the sentinel.
    delegating = secrets_only + f'exec "{hooks_repo}/scripts/pre-push-lint.sh"\n'
    dest.write_text(delegating, encoding="utf-8")
    dest.chmod(0o755)

    result2 = _make_hooks(hooks_repo)
    assert result2.returncode == 0, result2.stderr
    assert dest.read_text(encoding="utf-8") == delegating, (
        "a hook that genuinely delegates must be left in place untouched"
    )

    (hooks_repo / "bad.py").write_text("x=1\n", encoding="utf-8")
    run = subprocess.run([str(dest)], cwd=hooks_repo, capture_output=True, text=True)
    assert run.returncode != 0, (
        "the effective pre-push hook claims to delegate to "
        "pre-push-lint.sh but did not block a seeded format violation; "
        f"stdout={run.stdout!r} stderr={run.stderr!r}"
    )
    assert "BLOCKED" in run.stderr
    assert "bad.py" in run.stderr


def test_make_hooks_installs_correctly_from_a_linked_worktree(tmp_path: Path) -> None:
    """(iv) a linked worktree has a FILE at .git, not a directory — the
    literal path ``.git/hooks`` cannot be created there. ``make hooks`` run
    from inside the worktree must still succeed and install into the real
    (shared) hooks directory.
    """
    repo = tmp_path / "repo"
    _seed_repo(repo)

    assert (repo / ".git").is_dir()

    worktree = tmp_path / "wt"
    _git(["worktree", "add", "-q", str(worktree), "-b", "wtbranch"], repo)
    assert (worktree / ".git").is_file(), "expected a linked worktree (.git is a file)"

    result = _make_hooks(worktree)
    assert result.returncode == 0, (
        f"make hooks failed in a linked worktree (mkdir .git/hooks would "
        f"die with 'Not a directory' if hooks_dir weren't resolved via "
        f"git rev-parse --git-path hooks): {result.stderr}"
    )

    dest = _hooks_dir(worktree) / "pre-push"
    assert dest.is_file()
    assert os.access(dest, os.X_OK)
    # Worktrees share one hooks directory with the main checkout.
    assert dest == _hooks_dir(repo) / "pre-push"


def test_make_hooks_respects_core_hooks_path(tmp_path: Path) -> None:
    """(iv) with ``core.hooksPath`` set, git never reads ``.git/hooks`` at
    all. A hook copied to the literal ``.git/hooks/pre-push`` would be a
    silent false success — installed-looking, never invoked. ``make hooks``
    must install at the configured path instead.
    """
    repo = tmp_path / "repo"
    _seed_repo(repo)

    custom_hooks = tmp_path / "custom-hooks"
    custom_hooks.mkdir()
    _git(["config", "core.hooksPath", str(custom_hooks)], repo)

    result = _make_hooks(repo)
    assert result.returncode == 0, result.stderr

    dest = custom_hooks / "pre-push"
    assert dest.is_file(), (
        "hook was not installed at the configured core.hooksPath; "
        "make hooks must resolve the path git actually uses"
    )
    assert os.access(dest, os.X_OK)
    # And the literal .git/hooks/pre-push must NOT have been (mis)used.
    assert not (repo / ".git" / "hooks" / "pre-push").exists()
