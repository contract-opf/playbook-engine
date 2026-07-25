"""Generalized corpus staging — ``playbook stage``.

Detects the directory layout of a source corpus, flattens each negotiation
trail into the flat ``<agreement>/<NN>__<name>`` layout the engine walker
expects, and scaffolds per-agreement ``hints.yaml`` plus a top-level
``playbook.config.yaml`` skeleton.

Generalizes an earlier corpus-specific staging script so the skill can
be pointed at an arbitrary corpus directory.

Layout detection:
  ``flat``        — ``<agreement>/<version-files>`` with no ``Versions/`` sub-folder.
  ``clm_nested``  — per-agreement ``Versions/`` sub-folder (CLM export style);
                    optional top-level ``EXECUTED_*.pdf`` for the signed copy.
  ``manifest``    — a ``manifest.jsonl`` is present at the corpus root; fields:
                    ``folder``, ``filename_on_disk``, ``versionNumber``,
                    ``original_filename``, ``status`` (``EXECUTED`` = signed copy).

Security: no real agreement content is stored here.
All corpus content is accessed read-only from caller-supplied paths at runtime.
Staging writes only to ``out_dir`` (default ``DEFAULT_STAGING_ROOT/<name>``, a
user-owned cache directory rather than the world-readable ``/tmp`` — see
issue #135; real corpus content, even as symlinks, should not land somewhere
other local users can enumerate).

By default, version files are placed as absolute symlinks back to the
source corpus (cheap, no duplication). Symlink targets are host paths,
though — if the staged tree is going to cross a filesystem boundary (e.g.
staged on the host, then bind-mounted read-only into a container that
cannot see the host path), pass ``copy_files=True`` (CLI: ``stage --copy``)
to write real file copies instead, so the staged tree is self-contained.
See issue #130.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, TypedDict

_SUPPORTED = frozenset({".rtf", ".docx", ".pdf"})

# Default staging root — user-owned cache dir, not world-readable /tmp
# (issue #135: /tmp/pbe-staging let any local user enumerate staged corpus
# entries). Overridable per-invocation via ``playbook stage --out``.
DEFAULT_STAGING_ROOT = Path.home() / ".cache" / "playbook-engine" / "staging"

# Filename-cue signed detection is advisory scaffolding only (CORPUS-LAYOUT.md
# promises the engine never trusts filenames for anything load-bearing;
# ``detect_signed`` — content-based — is the real source of truth). Tokenize
# on non-alphanumeric separators so "v2_signed_final" still matches on the
# whole word "signed", but a *negation* token anywhere in the stem vetoes the
# match outright — this is what keeps "unsigned-execution-copy" (contains the
# substring "signed" inside "unsigned") and "draft-to-be-signed" (a draft that
# is not yet signed) from hijacking the signed anchor. See issue #96.
_SIGNED_TOKENS = frozenset({"signed", "executed"})
_NEGATION_TOKENS = frozenset(
    {"unsigned", "unexecuted", "draft", "pending", "redline", "unsign", "unexecute"}
)
_NEGATION_PHRASES = (
    ("to", "be", "signed"),
    ("to", "be", "executed"),
)


def _looks_signed(stem: str) -> bool:
    """Return True iff *stem* carries an unambiguous "this is the signed copy" cue.

    Advisory only — a filename match here just seeds staging's ``hints.yaml``;
    it is never a substitute for content-based ``detect_signed``.
    """
    tokens = [t for t in re.split(r"[^a-z0-9]+", stem.lower()) if t]
    token_set = set(tokens)
    if token_set & _NEGATION_TOKENS:
        return False
    for phrase in _NEGATION_PHRASES:
        n = len(phrase)
        if any(tuple(tokens[i : i + n]) == phrase for i in range(len(tokens) - n + 1)):
            return False
    return bool(token_set & _SIGNED_TOKENS)


class _ManifestRecord(TypedDict, total=False):
    """One line from a manifest.jsonl file."""

    folder: str
    filename_on_disk: str
    versionNumber: int
    original_filename: str
    status: str


LayoutKind = Literal["flat", "clm_nested", "manifest", "unknown"]


class UnknownLayoutError(ValueError):
    """Raised by :func:`stage` when *src_dir*'s layout cannot be determined.

    Unlike the three known layouts, an ``"unknown"`` corpus (see
    :func:`detect_layout`) cannot be staged directly — silently guessing
    would risk mis-grouping or mis-ordering a negotiation trail with no
    signal the caller could catch (issue #186: today's fallback-to-``flat``
    is exactly that silent-misdetection bug). Callers must go through the
    propose-then-execute path instead: ``playbook stage --plan-only`` writes
    a ``staging_plan.json`` for review (:func:`playbook_engine.intake_plan.
    build_staging_plan`), then ``playbook stage --plan staging_plan.json``
    executes it (:func:`playbook_engine.intake_plan.execute_staging_plan`).
    """


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class StagingResult:
    """Summary of a completed staging run.

    Attributes:
        out_dir:        Absolute path to the staging output directory.
        layout:         Detected source layout kind.
        staged_count:   Total number of version-file symlinks created.
        agreement_count: Number of agreement folders written.
        missing:        Source files present in the manifest but not on disk.
    """

    out_dir: Path
    layout: LayoutKind
    staged_count: int
    agreement_count: int
    missing: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Layout detection
# ---------------------------------------------------------------------------


def detect_layout(src_dir: Path) -> LayoutKind:
    """Detect the directory layout of *src_dir*.

    Rules (applied in order):
    1. ``"manifest"``   — a ``manifest.jsonl`` file exists at *src_dir*.
    2. ``"clm_nested"`` — at least one direct child directory contains a
                          ``Versions/`` sub-folder.
    3. ``"flat"``       — at least one direct child directory itself directly
                          contains supported version files (``<agreement>/
                          <version-files>``).
    4. ``"unknown"``    — none of the above match confidently: e.g. loose
                          files sitting directly at *src_dir* root (no
                          per-agreement subfolders at all), or an ad-hoc tree
                          (email-export folders, mixed naming) that doesn't
                          encode the negotiation structure in its paths.
                          Previously this fell through to ``"flat"`` by
                          default, which silently mis-staged corpora with no
                          real layout (issue #186) — callers must route
                          ``"unknown"`` through the propose-then-execute path
                          (see :class:`UnknownLayoutError`) instead.

    Args:
        src_dir: Root corpus directory to inspect.

    Returns:
        One of ``"manifest"``, ``"clm_nested"``, ``"flat"``, or ``"unknown"``.
    """
    if (src_dir / "manifest.jsonl").exists():
        return "manifest"

    subdirs = [c for c in src_dir.iterdir() if c.is_dir() and not c.name.startswith(".")]

    for child in subdirs:
        if (child / "Versions").is_dir():
            return "clm_nested"

    for child in subdirs:
        if any(p.is_file() and p.suffix.lower() in _SUPPORTED for p in child.iterdir()):
            return "flat"

    return "unknown"


# ---------------------------------------------------------------------------
# Minimal YAML string quoting helper
# ---------------------------------------------------------------------------


def _q(s: str) -> str:
    """Minimally quote a string for YAML (double-quoted form)."""
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _place(src: Path, dest: Path, *, copy_files: bool) -> None:
    """Place *src* at *dest* — a real copy if *copy_files*, else an absolute symlink.

    Symlinks are the default (cheap, no duplication) but carry host paths as
    their targets; ``copy_files=True`` makes the staged output self-contained
    so it survives crossing a filesystem/mount boundary (issue #130).
    """
    if copy_files:
        shutil.copy2(src, dest)
    else:
        os.symlink(src.resolve(), dest)


# ---------------------------------------------------------------------------
# Stage
# ---------------------------------------------------------------------------


def stage(
    src_dir: Path,
    out_dir: Path,
    *,
    manifest_path: Path | None = None,
    docs_path: Path | None = None,
    copy_files: bool = False,
) -> StagingResult:
    """Flatten *src_dir* into *out_dir* using the detected layout.

    For every agreement found in *src_dir* the output layout is::

        out_dir/
          <agreement>/
            01__<original_name>.docx   (symlink → src, or a real copy)
            02__<original_name>.docx   (symlink → src, or a real copy)
            ...
            hints.yaml                 (order + signed_version)

    Args:
        src_dir:       Source corpus root.
        out_dir:       Destination directory.  Recreated on each call.
        manifest_path: Override manifest file path (``manifest`` layout only).
                       Defaults to ``src_dir / "manifest.jsonl"``.
        docs_path:     Override base for resolving manifest ``filename_on_disk``
                       paths (``manifest`` layout only).  Defaults to ``src_dir``.
                       Each file is resolved against ``docs_path``, the corpus
                       root, and ``src_dir/docs`` (first existing match wins), so
                       both root-relative and docs/-relative manifests work.
        copy_files:    Write real file copies instead of absolute symlinks.
                       Use this when the staged output will cross a filesystem
                       boundary (e.g. staged on the host, then bind-mounted
                       read-only into a container) — symlink targets are host
                       paths and dangle in that scenario (issue #130).

    Returns:
        :class:`StagingResult` with counts and missing-file details.

    Raises:
        UnknownLayoutError: *src_dir*'s layout could not be determined (see
            :func:`detect_layout`). Does not touch *out_dir* in this case.
    """
    layout = detect_layout(src_dir)

    if layout == "unknown":
        raise UnknownLayoutError(
            f"{src_dir}: could not determine a known layout (no manifest.jsonl, "
            "no Versions/ nesting, no <agreement>/<version-files> subfolders). "
            "Run `playbook stage --plan-only` to write a staging_plan.json "
            "proposal for review, then `playbook stage --plan staging_plan.json` "
            "to execute it."
        )

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    if layout == "manifest":
        return _stage_manifest(
            src_dir,
            out_dir,
            layout=layout,
            manifest_path=manifest_path or src_dir / "manifest.jsonl",
            docs_path=docs_path or src_dir,
            copy_files=copy_files,
        )
    if layout == "clm_nested":
        return _stage_clm_nested(src_dir, out_dir, layout=layout, copy_files=copy_files)
    return _stage_flat(src_dir, out_dir, layout=layout, copy_files=copy_files)


def _write_hints(dest: Path, order: list[str], signed: str | None) -> None:
    # Hints are matched against the engine's version_id, which is the file STEM
    # (no extension).  The staged symlinks carry extensions, so strip them here —
    # otherwise the order/signed_version hints never match a version and
    # signed-copy detection silently fails across the whole corpus.
    order_ids = [Path(n).stem for n in order]
    lines = "order:\n" + "".join(f"  - {_q(n)}\n" for n in order_ids)
    if signed:
        lines += f"signed_version: {_q(Path(signed).stem)}\n"
    (dest / "hints.yaml").write_text(lines, encoding="utf-8")


def _stage_flat(
    src_dir: Path, out_dir: Path, *, layout: LayoutKind, copy_files: bool = False
) -> StagingResult:
    """Stage a flat layout: ``<agreement>/<version-files>``."""
    staged = 0
    agreement_count = 0

    for doc_dir in sorted(
        d for d in src_dir.iterdir() if d.is_dir() and not d.name.startswith(".")
    ):
        version_files = sorted(
            p for p in doc_dir.iterdir() if p.is_file() and p.suffix.lower() in _SUPPORTED
        )
        if not version_files:
            continue

        dest = out_dir / doc_dir.name
        dest.mkdir()
        order: list[str] = []
        signed: str | None = None

        for i, src in enumerate(version_files, start=1):
            name = f"{i:02d}__{src.name}"
            _place(src, dest / name, copy_files=copy_files)
            order.append(name)
            # Flat layout: whole-word "signed"/"executed" cue, minus negations
            # (see ``_looks_signed`` — advisory only, never a hard trust of
            # the filename).
            if _looks_signed(src.stem):
                signed = name
            staged += 1

        _write_hints(dest, order, signed)
        agreement_count += 1

    return StagingResult(
        out_dir=out_dir,
        layout=layout,
        staged_count=staged,
        agreement_count=agreement_count,
    )


def _stage_clm_nested(
    src_dir: Path, out_dir: Path, *, layout: LayoutKind, copy_files: bool = False
) -> StagingResult:
    """Stage a CLM-nested layout: ``<agreement>/Versions/<files>`` + optional top-level ``EXECUTED_*.pdf``."""
    staged = 0
    agreement_count = 0

    for doc_dir in sorted(
        d for d in src_dir.iterdir() if d.is_dir() and not d.name.startswith(".")
    ):
        versions_dir = doc_dir / "Versions"
        if not versions_dir.is_dir():
            # Fall back: treat as flat agreement folder.
            version_files = sorted(
                p for p in doc_dir.iterdir() if p.is_file() and p.suffix.lower() in _SUPPORTED
            )
        else:
            version_files = sorted(
                p for p in versions_dir.iterdir() if p.is_file() and p.suffix.lower() in _SUPPORTED
            )

        if not version_files:
            continue

        dest = out_dir / doc_dir.name
        dest.mkdir()
        order: list[str] = []
        signed: str | None = None

        for i, src in enumerate(version_files, start=1):
            name = f"{i:02d}__{src.name}"
            _place(src, dest / name, copy_files=copy_files)
            order.append(name)
            staged += 1

        # Check for top-level EXECUTED_*.pdf (signed copy outside Versions/)
        executed_files = sorted(
            p
            for p in doc_dir.iterdir()
            if p.is_file() and p.name.startswith("EXECUTED_") and p.suffix.lower() == ".pdf"
        )
        for exec_src in executed_files:
            n = len(order) + 1
            name = f"{n:02d}__{exec_src.name}"
            _place(exec_src, dest / name, copy_files=copy_files)
            order.append(name)
            signed = name  # last EXECUTED wins
            staged += 1

        # Also detect signed from filename cues in Versions/ (advisory only —
        # see ``_looks_signed``).
        if signed is None:
            for entry_name in order:
                if _looks_signed(Path(entry_name).stem):
                    signed = entry_name

        _write_hints(dest, order, signed)
        agreement_count += 1

    return StagingResult(
        out_dir=out_dir,
        layout=layout,
        staged_count=staged,
        agreement_count=agreement_count,
    )


def _stage_manifest(
    src_dir: Path,
    out_dir: Path,
    *,
    layout: LayoutKind,
    manifest_path: Path,
    docs_path: Path,
    copy_files: bool = False,
) -> StagingResult:
    """Stage a manifest-driven layout (JSONL manifest).

    Each manifest line has: ``folder``, ``filename_on_disk``, ``versionNumber``,
    ``original_filename``, ``status`` (contains ``"EXECUTED"`` for signed copy).
    """
    by_folder: dict[str, list[_ManifestRecord]] = defaultdict(list)
    with manifest_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rec = json.loads(line)
                by_folder[rec["folder"]].append(rec)

    staged = 0
    missing: list[str] = []
    agreement_count = 0

    for folder, recs in sorted(by_folder.items()):
        recs.sort(key=lambda r: r["versionNumber"])
        dest = out_dir / folder
        dest.mkdir()
        order: list[str] = []
        signed: str | None = None

        for r in recs:
            filename = r["filename_on_disk"]
            nn = f"{r['versionNumber']:02d}"
            # ``filename_on_disk`` may be relative to the configured docs path,
            # the corpus root, or a ``docs/`` subdir.  Real CLM exports nest
            # agreements under ``docs/`` and omit that prefix from the manifest;
            # other manifests embed it.  Try each base, then fall back to an
            # ordinal-prefix glob within the candidate ``Versions/`` folders.
            bases = (docs_path, src_dir, src_dir / "docs")
            src: Path | None = None
            for base in bases:
                cand = base / filename
                if cand.exists():
                    src = cand
                    break
            if src is None:
                for base in bases:
                    vdir = (base / filename).parent
                    globbed = sorted(vdir.glob(f"{nn}_*")) if vdir.is_dir() else []
                    if len(globbed) == 1:
                        src = globbed[0]
                        break
            if src is None:
                missing.append(f"{docs_path / filename}  (glob {nn}_* -> 0)")
                continue

            name = f"{r['versionNumber']:02d}__{r['original_filename']}"
            _place(src, dest / name, copy_files=copy_files)
            order.append(name)
            if "EXECUTED" in (r.get("status") or ""):
                signed = name  # last EXECUTED wins (highest versionNumber)
            staged += 1

        if not order:
            dest.rmdir()
            continue

        _write_hints(dest, order, signed)
        agreement_count += 1

    return StagingResult(
        out_dir=out_dir,
        layout=layout,
        staged_count=staged,
        agreement_count=agreement_count,
        missing=missing,
    )


# ---------------------------------------------------------------------------
# Config scaffold
# ---------------------------------------------------------------------------


_CONFIG_FILENAME = "playbook.config.yaml"


def scaffold_config(src_dir: Path, out_dir: Path) -> dict[str, Any]:
    """Emit a ``playbook.config.yaml`` skeleton with required fields.

    Writes ``out_dir/playbook.config.yaml`` (the canonical config filename
    used throughout the CLI/docs — see issue #130) and returns the skeleton
    as a dict.

    Fields inferred:
    - ``agreement_type.id``  — slug derived from *src_dir* name
      (lower-cased, non-alphanumeric chars replaced with ``-``).
    - ``agreement_type.name`` — title-cased version of the dir name.
    - ``baseline.template``   — relative path to first ``*template*`` file
      found at *src_dir* root, or ``null``.
    - ``taxonomy``            — placeholder string ``"FILL_IN_TAXONOMY_PATH"``.
      Replace with a corpus-specific relative path, or ``builtin:<name>``
      to reference one of the engine's bundled taxonomies (resolved
      independent of this file's location — see ``config.py``).
    - ``provenance.our_party_aliases`` — empty list (for human to fill).
    - ``perspective`` — commented-out ``party``/``counterparty_type``
      placeholder block (issue #165). Left commented (not emitted as active
      config) because both fields are optional and neither has a safe
      default to fabricate: ``party`` normally defaults from
      ``provenance.our_party_aliases[0]`` once that's filled in, but
      ``counterparty_type`` has no derivable default at all.

    Args:
        src_dir: Source corpus root (used for name inference and template scan).
        out_dir: Destination directory where ``playbook.config.yaml`` is written.

    Returns:
        The skeleton dict (same content as the written YAML).
    """
    import re  # noqa: PLC0415

    dir_name = src_dir.name
    # slug: lowercase, replace non-alphanumeric with dash, collapse runs
    slug = re.sub(r"[^a-z0-9]+", "-", dir_name.lower()).strip("-")
    if not slug:
        slug = "agreement"

    # Title: replace separators with spaces
    title = re.sub(r"[-_]+", " ", dir_name).title()

    # Template: first *template* file at src_dir root (any supported ext)
    template_rel: str | None = None
    for candidate in sorted(src_dir.iterdir()):
        if (
            candidate.is_file()
            and "template" in candidate.name.lower()
            and candidate.suffix.lower() in _SUPPORTED
        ):
            template_rel = str(candidate.name)
            break

    skeleton: dict[str, Any] = {
        "agreement_type": {
            "id": slug,
            "name": title,
        },
        "baseline": {
            "template": template_rel,
        },
        "taxonomy": "FILL_IN_TAXONOMY_PATH",
        "provenance": {
            "our_party_aliases": [],
            # Known counterparty entity names to pseudonymize at ingest
            # (issue #153) — fill in from the corpus manifest/folder names,
            # same workflow as our_party_aliases above.
            "known_entities": [],
        },
    }

    import yaml  # noqa: PLC0415

    yaml_text = yaml.dump(skeleton, allow_unicode=True, sort_keys=False)
    # Inline guidance for the provenance lists must be REAL yaml comments —
    # a `#` comment on the Python dict above is dropped by yaml.dump, so a
    # first-timer sees only bare empty lists with no hint that leaving them
    # empty silently disables provenance detection + born-safe pseudonymization
    # (issue #182). Inject the guidance into the dumped text instead.
    yaml_text = yaml_text.replace(
        "  our_party_aliases: []",
        "  our_party_aliases: []  # REQUIRED for provenance: every form of OUR\n"
        "  #   name in the recitals — legal entities + defined terms\n"
        "  #   (e.g. 'Acme Corp, LLC', 'Acme', 'Facility'). `mine` warns if none match.",
    )
    yaml_text = yaml_text.replace(
        "  known_entities: []",
        "  known_entities: []  # counterparty names to pseudonymize at ingest\n"
        "  #   (born-safe, issue #153). Full legal name PLUS its abbreviation/acronym\n"
        "  #   (e.g. 'The City University of New York' AND 'CUNY') — empty = NO\n"
        "  #   pseudonymization. Derive these from the corpus, not guesswork.",
    )
    # Issue #165: perspective — commented and blank, unlike the other
    # sections above, because it's the one block where filling in a
    # placeholder value would be worse than leaving it unset (neither field
    # may be fabricated; see EngineConfig.perspective / assemble_playbook).
    yaml_text += (
        "\n"
        '# Optional (issue #165): perspective — whose "us" this playbook is\n'
        "# reviewed as. Uncomment and fill in both fields to have the playbook\n"
        "# carry a top-level `perspective` block; `party` alone also defaults\n"
        "# from provenance.our_party_aliases[0] once that's filled in above.\n"
        "# perspective:\n"
        '#   party: ""              # our legal entity/party name\n'
        '#   counterparty_type: ""  # what the other side typically is,\n'
        '#                          # e.g. "Educational Institution"\n'
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / _CONFIG_FILENAME).write_text(yaml_text, encoding="utf-8")

    return skeleton
