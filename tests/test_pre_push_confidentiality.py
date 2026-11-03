"""Tests for the tracked, name-free pre-push confidentiality scanner
(``scripts/pre-push-confidentiality.py``, QA-2026-11-03 #3).

All names here are invented. The scanner is exercised end to end against real
throwaway repos: a work clone pushing to a bare remote, with the hook's stdin
(``<lref> <lsha> <rref> <rsha>`` lines) built the way git builds it.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).parent.parent
_SCRIPT = _ROOT / "scripts" / "pre-push-confidentiality.py"
_ZERO = "0" * 40
_NAME = "Zorblax Polytechnic Institute"


def _load():
    spec = importlib.util.spec_from_file_location("pre_push_confidentiality", _SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


scanner = _load()


def _git(cwd: Path, *args: str) -> str:
    env = {**os.environ, "GIT_AUTHOR_NAME": "T", "GIT_AUTHOR_EMAIL": "t@example.com"}
    env |= {"GIT_COMMITTER_NAME": "T", "GIT_COMMITTER_EMAIL": "t@example.com"}
    proc = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, env=env)
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip()


def _commit(repo: Path, files: dict[str, str], msg: str = "work") -> str:
    for name, content in files.items():
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", msg)
    return _git(repo, "rev-parse", "HEAD")


@pytest.fixture
def clone(tmp_path: Path) -> Path:
    remote = tmp_path / "remote.git"
    _git(tmp_path, "init", "-q", "--bare", str(remote))
    work = tmp_path / "work"
    _git(tmp_path, "init", "-q", "-b", "main", str(work))
    _git(work, "remote", "add", "origin", str(remote))
    _commit(work, {"README.md": "hello\n"}, "initial")
    _git(work, "push", "-q", "origin", "main", "--no-verify")
    return work


@pytest.fixture
def names_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    d = tmp_path / "names"
    d.mkdir()
    (d / "terms.txt").write_text(f"# private\n{_NAME}\n", encoding="utf-8")
    monkeypatch.setenv("PLAYBOOK_CONFIDENTIAL_NAMES_DIR", str(d))
    return d


def _run_hook(repo: Path, lines: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(_SCRIPT), "origin", "file:///x"],
        cwd=repo,
        input=lines,
        capture_output=True,
        text=True,
    )


def _push_line(repo: Path, rsha: str | None = None) -> str:
    head = _git(repo, "rev-parse", "HEAD")
    remote_sha = rsha if rsha is not None else _git(repo, "rev-parse", "origin/main")
    return f"refs/heads/main {head} refs/heads/main {remote_sha}\n"


@pytest.mark.parametrize(
    "variant",
    [
        "Zorblax Polytechnic Institute",
        "zorblax-polytechnic-institute",
        "ZORBLAX_POLYTECHNIC_INSTITUTE",
        "Zorblax Polytechnic  Institute",
        "Zorblax​ Polytechnic Institute",
        "ZorblaxPolytechnicInstitute",
    ],
)
def test_blocks_content_in_every_spelling(clone: Path, names_dir: Path, variant: str) -> None:
    _commit(clone, {"src/a.py": f"x = '{variant}'\n"})
    res = _run_hook(clone, _push_line(clone))
    assert res.returncode == 1, res.stderr
    assert "src/a.py" in res.stderr
    assert "zorblax" not in res.stderr.lower()  # the report never echoes the term


def test_blocks_a_name_only_in_an_intermediate_commit(clone: Path, names_dir: Path) -> None:
    _commit(clone, {"a.py": f"# {_NAME}\n"}, "wip")
    _commit(clone, {"a.py": "# clean\n"}, "fix")
    res = _run_hook(clone, _push_line(clone))
    assert res.returncode == 1, "tip tree is clean but history carries the name"


def test_blocks_a_file_name_slug(clone: Path, names_dir: Path) -> None:
    _commit(clone, {"docs/zorblax_polytechnic_institute_notes.md": "nothing here\n"})
    res = _run_hook(clone, _push_line(clone))
    assert res.returncode == 1
    assert "file name" in res.stderr


def test_blocks_a_commit_message(clone: Path, names_dir: Path) -> None:
    _commit(clone, {"a.py": "x = 1\n"}, f"fix: handle {_NAME} edge case")
    res = _run_hook(clone, _push_line(clone))
    assert res.returncode == 1
    assert "commit message" in res.stderr


def test_new_branch_scans_commits_not_on_the_remote(clone: Path, names_dir: Path) -> None:
    _git(clone, "checkout", "-q", "-b", "feature")
    _commit(clone, {"a.py": f"# {_NAME}\n"})
    head = _git(clone, "rev-parse", "HEAD")
    res = _run_hook(clone, f"refs/heads/feature {head} refs/heads/feature {_ZERO}\n")
    assert res.returncode == 1


def test_clean_push_and_delete_only_push_pass(clone: Path, names_dir: Path) -> None:
    _commit(clone, {"a.py": "x = 1\n"})
    assert _run_hook(clone, _push_line(clone)).returncode == 0
    delete = f"(delete) {_ZERO} refs/heads/old {'a' * 40}\n"
    assert _run_hook(clone, delete).returncode == 0


def test_a_name_removed_before_the_push_range_is_not_rescanned(
    clone: Path, names_dir: Path
) -> None:
    """Only commits being pushed are scanned: an already-published commit is out of range."""
    _commit(clone, {"a.py": f"# {_NAME}\n"})
    _git(clone, "push", "-q", "origin", "main", "--no-verify")
    _commit(clone, {"b.py": "x = 1\n"})
    assert _run_hook(clone, _push_line(clone)).returncode == 0


def test_fails_closed_without_a_term_list(
    clone: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setenv("PLAYBOOK_CONFIDENTIAL_NAMES_DIR", str(empty))
    _commit(clone, {"a.py": "x = 1\n"})
    res = _run_hook(clone, _push_line(clone))
    assert res.returncode == 1
    assert "fail closed" in res.stderr
    # A missing directory is the same as an empty one.
    monkeypatch.setenv("PLAYBOOK_CONFIDENTIAL_NAMES_DIR", str(tmp_path / "nope"))
    assert _run_hook(clone, _push_line(clone)).returncode == 1


def test_generate_names_reads_the_entity_registry(
    clone: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = tmp_path / "entity_registry.json"
    registry.write_text(
        json.dumps({"aliases": {"x": "Counterparty-1"}, "canonical": {"x": _NAME}}),
        encoding="utf-8",
    )
    out = tmp_path / "gen"
    monkeypatch.setenv("PLAYBOOK_CONFIDENTIAL_NAMES_DIR", str(out))
    proc = subprocess.run(
        [sys.executable, str(_SCRIPT), "--generate-names", "--registry", str(registry)],
        cwd=clone,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert (out / "entity-registry.txt").stat().st_mode & 0o077 == 0
    _commit(clone, {"a.py": f"# {_NAME.lower()}\n"})
    assert _run_hook(clone, _push_line(clone)).returncode == 1


def test_short_terms_need_word_boundaries() -> None:
    terms = scanner.Terms(["Acme"])
    assert terms.find("the Acme deal")
    assert not terms.find("acmetools and pacmen")


def test_scanner_source_names_nothing_real() -> None:
    """The tracked file must stay name-free: no registry-looking literals."""
    src = _SCRIPT.read_text(encoding="utf-8")
    assert "university" not in src.lower()
    assert "contract-opf" not in src
