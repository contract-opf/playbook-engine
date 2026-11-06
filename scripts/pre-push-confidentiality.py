#!/usr/bin/env python3
"""Tracked, name-free pre-push confidentiality scanner (QA-2026-11-03 #3).

This file deliberately contains NO names. The terms it blocks live in private
files that are never tracked: every ``*.txt`` in ``<git-common-dir>/confidentiality/``
(shared by all worktrees of a clone, never part of a push), or in the directory
named by ``$PLAYBOOK_CONFIDENTIAL_NAMES_DIR``. One term per line; ``#`` starts a
comment. Generate the entity-registry part with::

    python3 scripts/pre-push-confidentiality.py --generate-names [--registry PATH]

(``make hooks`` does this when a registry exists). Add hand-picked terms
(codenames, internal domains, people) as extra ``*.txt`` files in the same
directory.

It FAILS CLOSED: if there are refs to push and no term list can be found (or it
is empty), the push is blocked. Bypass a verified false positive with
``git push --no-verify``.

What it scans, for EVERY commit being pushed (``<remote sha>..<local sha>``;
for a new branch, every commit not already on a remote-tracking ref): the
commit message and author/committer identities, every path touched, and every
line added. Text is normalised before comparing -- Unicode NFKC, zero-width
characters dropped, case folded, every run of non-alphanumerics collapsed to a
single space -- so ``Acme Widgets LLC``, ``acme-widgets-llc``,
``ACME_WIDGETS_LLC`` and ``Acme<NBSP>Widgets  LLC`` are all the same term. Long
terms are also compared with all separators removed (``AcmeWidgetsLLC``).

Usage as a git hook: ``pre-push-confidentiality.py <remote-name> <remote-url>``
with the usual ``<lref> <lsha> <rref> <rsha>`` lines on stdin.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import unicodedata
from pathlib import Path

ZERO = "0" * 40
_SQUASH_MIN = 10  # terms at least this long (separators removed) also match squashed
_MIN_TERM = 3  # normalised terms shorter than this are ignored (noise)
_NON_ALNUM = re.compile(r"[^0-9a-z]+")


def normalize(text: str) -> str:
    """Case/separator/Unicode-insensitive form: lowercase alphanumerics, single spaces."""
    text = unicodedata.normalize("NFKC", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Cf")
    return _NON_ALNUM.sub(" ", text.casefold()).strip()


class Terms:
    """The private term list, pre-normalised for fast matching."""

    def __init__(self, raw: list[str]) -> None:
        seen: dict[str, str] = {}
        for term in raw:
            norm = normalize(term)
            if len(norm) >= _MIN_TERM:
                seen.setdefault(norm, term)
        self.spaced = {n: f" {n} " for n in seen}
        self.squashed = {
            n.replace(" ", ""): n for n in seen if len(n.replace(" ", "")) >= _SQUASH_MIN
        }

    def __len__(self) -> int:
        return len(self.spaced)

    def find(self, text: str) -> list[str]:
        """Normalised terms found in *text* (never the text itself)."""
        norm = normalize(text)
        padded = f" {norm} "
        hits = [n for n, p in self.spaced.items() if p in padded]
        if self.squashed:
            flat = norm.replace(" ", "")
            hits += [n for sq, n in self.squashed.items() if sq in flat and n not in hits]
        return hits


def parse_names_file(text: str) -> list[str]:
    out = []
    for line in text.splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.append(line)
    return out


def _git(args: list[str], cwd: str | None = None) -> str:
    proc = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout


def names_dir(cwd: str | None = None) -> Path:
    override = os.environ.get("PLAYBOOK_CONFIDENTIAL_NAMES_DIR")
    if override:
        return Path(override)
    common = _git(["rev-parse", "--path-format=absolute", "--git-common-dir"], cwd).strip()
    return Path(common) / "confidentiality"


def load_terms(directory: Path) -> Terms:
    raw: list[str] = []
    if directory.is_dir():
        for f in sorted(directory.glob("*.txt")):
            raw += parse_names_file(f.read_text(encoding="utf-8", errors="replace"))
    return Terms(raw)


def commits_to_scan(lsha: str, rsha: str, remote_name: str, cwd: str | None) -> list[str]:
    if rsha != ZERO:
        known = subprocess.run(
            ["git", "cat-file", "-e", f"{rsha}^{{commit}}"], cwd=cwd, capture_output=True
        )
        if known.returncode == 0:
            return _git(["rev-list", "--reverse", f"{rsha}..{lsha}"], cwd).split()
    # New branch (or remote tip unknown locally): everything not on any remote ref.
    args = ["rev-list", "--reverse", lsha, "--not", f"--remotes={remote_name}"]
    return _git(args, cwd).split()


def scan_commit(sha: str, terms: Terms, cwd: str | None = None) -> list[str]:
    """Return name-free descriptions of where commit *sha* hits a term."""
    problems: list[str] = []
    meta = _git(["show", "-s", "--format=%an%n%ae%n%cn%n%ce%n%B", sha], cwd)
    if terms.find(meta):
        problems.append(f"{sha[:10]}: commit message or author/committer identity")
    parents = _git(["rev-list", "--parents", "-n", "1", sha], cwd).split()[1:]
    if len(parents) > 1:
        # Only what the merge itself introduced: a conflict resolution can add
        # lines neither (separately scanned) parent has.
        for line in _git(["diff-tree", "--cc", "--no-commit-id", "-U0", sha], cwd).splitlines():
            if line[:2].strip(" ") in ("+", "++") and terms.find(line[2:]):
                problems.append(f"{sha[:10]}: content added by the merge")
                break
        return problems
    base = ["diff-tree", "-r", "--root", "--no-commit-id", "--no-renames"]
    for path in filter(None, _git([*base, "--name-only", "-z", sha], cwd).split("\0")):
        if terms.find(path):
            problems.append(f"{sha[:10]}: file name {path!r}")
    current = "?"
    flagged: set[str] = set()
    for line in _git([*base, "-p", "-U0", "-a", sha], cwd).splitlines():
        if line.startswith("+++ "):
            current = line[6:] if line.startswith("+++ b/") else line[4:]
        elif line.startswith("+") and current not in flagged and terms.find(line[1:]):
            flagged.add(current)
            problems.append(f"{sha[:10]}: added content in {current!r}")
    return problems


def scan_push(
    stdin_lines: list[str], remote_name: str, terms: Terms, cwd: str | None = None
) -> list[str]:
    problems: list[str] = []
    for line in stdin_lines:
        parts = line.split()
        if len(parts) < 4:
            continue
        lsha, rsha = parts[1], parts[3]
        if lsha == ZERO:  # deletion
            continue
        for sha in commits_to_scan(lsha, rsha, remote_name, cwd):
            problems += scan_commit(sha, terms, cwd)
    return list(dict.fromkeys(problems))


def generate_names(registry: Path, out_dir: Path) -> int:
    """Write ``entity-registry.txt`` (real names from the private registry) into *out_dir*."""
    data = json.loads(registry.read_text(encoding="utf-8"))
    names = sorted({v for v in data.get("canonical", {}).values() if v and v.strip()})
    out_dir.mkdir(parents=True, exist_ok=True)
    fd = os.open(out_dir / "entity-registry.txt", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write("# generated from the entity registry; private, never commit\n")
        f.write("\n".join(names) + "\n")
    return len(names)


def main(argv: list[str]) -> int:
    if argv and argv[0] == "--generate-names":
        registry = Path.home() / ".cache" / "playbook-engine" / "entity_registry.json"
        if len(argv) >= 3 and argv[1] == "--registry":
            registry = Path(argv[2])
        if not registry.is_file():
            print(f"no entity registry at {registry}; nothing generated", file=sys.stderr)
            return 1
        out = names_dir()
        print(f"wrote {generate_names(registry, out)} names to {out}/entity-registry.txt")
        return 0

    remote_name = argv[0] if argv else ""
    lines = [ln for ln in sys.stdin.read().splitlines() if ln.strip()]
    if not any(len(ln.split()) >= 4 and ln.split()[1] != ZERO for ln in lines):
        return 0  # nothing but deletions (or nothing) to push
    try:
        terms = load_terms(names_dir())
    except Exception as exc:  # fail closed
        print(f"pre-push BLOCKED: cannot load the confidential term list ({exc}).", file=sys.stderr)
        return 1
    if not len(terms):
        print(
            "pre-push BLOCKED: no confidential term list found (fail closed).\n"
            "  Generate one:  python3 scripts/pre-push-confidentiality.py --generate-names\n"
            "  (or put one term per line in any *.txt under <git-common-dir>/confidentiality/).\n"
            "  Verified false positive / no private terms by design:  git push --no-verify",
            file=sys.stderr,
        )
        return 1
    try:
        problems = scan_push(lines, remote_name, terms)
    except Exception as exc:  # fail closed
        print(f"pre-push BLOCKED: scan failed ({exc}).", file=sys.stderr)
        return 1
    if problems:
        print("\npre-push BLOCKED: a confidential term would be published:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        print(
            "Remove it from the commit(s) (rewrite local history if it is in an earlier "
            "commit of this push). False positive:  git push --no-verify",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
