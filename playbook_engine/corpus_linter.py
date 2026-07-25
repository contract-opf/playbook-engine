"""Corpus-layout linter — pre-flight check before a compile run.

Reports what is missing or misconfigured in a corpus directory so a
non-engineer can fix it before running ``playbook compile``.

The linter produces a ``LintReport`` with a list of ``LintItem`` entries,
each classified as ``"ok"``, ``"warning"``, or ``"error"``.  Errors block
compilation; warnings are advisory.

Usage::

    from playbook_engine.corpus_linter import lint_corpus
    report = lint_corpus(corpus_dir, config_path=cfg)
    for item in report.items:
        print(item.level, item.message)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from playbook_engine.pipeline import (
    _LEGACY_EXTENSIONS,
    _LEGACY_FORMAT_INSTRUCTION,
    _SUPPORTED_EXTENSIONS,
    _discover_versions,
)

_MIN_VERSIONS_FOR_COMPARISON = 2
_IGNORED_STEMS = frozenset({"hints"})  # hints.yaml is intentional, not a stray file


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LintItem:
    """One finding from the linter."""

    level: str  # "ok" | "warning" | "error"
    code: str  # machine-readable tag (e.g. "EMPTY_CORPUS", "MISSING_CONFIG")
    message: str
    path: Path | None = None  # the relevant path, if applicable


@dataclass
class LintReport:
    """Aggregated lint findings for a corpus directory."""

    corpus_dir: Path
    items: list[LintItem] = field(default_factory=list)

    def add(self, level: str, code: str, message: str, path: Path | None = None) -> None:
        self.items.append(LintItem(level=level, code=code, message=message, path=path))

    @property
    def has_errors(self) -> bool:
        return any(i.level == "error" for i in self.items)

    @property
    def has_warnings(self) -> bool:
        return any(i.level == "warning" for i in self.items)

    @property
    def ok(self) -> bool:
        return not self.has_errors

    def errors(self) -> list[LintItem]:
        return [i for i in self.items if i.level == "error"]

    def warnings(self) -> list[LintItem]:
        return [i for i in self.items if i.level == "warning"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def lint_corpus(
    corpus_dir: Path,
    config_path: Path | None = None,
) -> LintReport:
    """Validate a corpus directory layout and optional config.

    Args:
        corpus_dir:   Root corpus directory to validate.
        config_path:  Optional path to the engine config YAML.

    Returns:
        ``LintReport`` with ok/warning/error items.
    """
    report = LintReport(corpus_dir=corpus_dir)

    # -----------------------------------------------------------------------
    # Corpus directory itself
    # -----------------------------------------------------------------------
    if not corpus_dir.exists():
        report.add(
            "error", "CORPUS_NOT_FOUND", f"Corpus directory not found: {corpus_dir}", corpus_dir
        )
        return report

    if not corpus_dir.is_dir():
        report.add(
            "error", "NOT_A_DIRECTORY", f"Corpus path is not a directory: {corpus_dir}", corpus_dir
        )
        return report

    report.add("ok", "CORPUS_EXISTS", f"Corpus directory exists: {corpus_dir}", corpus_dir)

    # -----------------------------------------------------------------------
    # Document subdirectories
    # -----------------------------------------------------------------------
    doc_dirs = sorted(d for d in corpus_dir.iterdir() if d.is_dir() and not d.name.startswith("."))

    if not doc_dirs:
        report.add(
            "error",
            "EMPTY_CORPUS",
            "No document subdirectories found. "
            "Create one folder per agreement and put all versions inside it.",
            corpus_dir,
        )
    else:
        report.add("ok", "HAS_DOCUMENTS", f"{len(doc_dirs)} document subdirectory(s) found")

    total_supported = 0
    for doc_dir in doc_dirs:
        _lint_doc_dir(doc_dir, report)
        total_supported += len(_discover_versions(doc_dir))

    if doc_dirs and total_supported == 0:
        report.add(
            "error",
            "NO_SUPPORTED_FILES",
            "No .docx, .pdf, or .rtf files found in any document directory. "
            "The engine supports these formats only.",
            corpus_dir,
        )

    # -----------------------------------------------------------------------
    # Config (optional)
    # -----------------------------------------------------------------------
    if config_path is not None:
        _lint_config(config_path, report)

    return report


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _lint_doc_dir(doc_dir: Path, report: LintReport) -> None:
    """Check one document subdirectory."""
    version_files = _discover_versions(doc_dir)
    all_files = [f for f in doc_dir.iterdir() if f.is_file()]
    legacy_doc_files = [f for f in all_files if f.suffix.lower() in _LEGACY_EXTENSIONS]
    unsupported = [
        f
        for f in all_files
        if f.suffix.lower() not in _SUPPORTED_EXTENSIONS
        and f.suffix.lower() not in _LEGACY_EXTENSIONS
        and f.stem.lower() not in _IGNORED_STEMS
    ]

    # Legacy .doc files are called out distinctly from the generic
    # DOC_UNSUPPORTED_FILES warning below (issue #100): real negotiation
    # history from the 2000s-2010s is full of binary .doc files, and lumping
    # them into "unsupported files present" with no conversion instruction
    # means an early .doc draft is silently dropped and a later redline gets
    # mistaken for the negotiation's start. Reported before the
    # no-version-files early-return so it still fires when .doc is the ONLY
    # file present.
    if legacy_doc_files:
        names = ", ".join(f.name for f in sorted(legacy_doc_files, key=lambda f: f.name)[:5])
        report.add(
            "warning",
            "DOC_LEGACY_FORMAT",
            f"'{doc_dir.name}': legacy .doc file(s) present ({names}). "
            "The engine does not read .doc directly and these are silently "
            "excluded from the negotiation trail — if any are early drafts, "
            f"the trail will start later than it actually did. Convert with "
            f"`{_LEGACY_FORMAT_INSTRUCTION} <file>` and re-run.",
            doc_dir,
        )

    if not version_files:
        report.add(
            "error",
            "DOC_NO_SUPPORTED_FILES",
            f"'{doc_dir.name}': no .docx, .pdf, or .rtf files found. "
            "Add at least one version file, or remove this folder.",
            doc_dir,
        )
        return

    report.add("ok", "DOC_HAS_FILES", f"'{doc_dir.name}': {len(version_files)} version file(s)")

    _lint_duplicate_stems(doc_dir, version_files, report)

    if len(version_files) < _MIN_VERSIONS_FOR_COMPARISON:
        report.add(
            "warning",
            "DOC_SINGLE_VERSION",
            f"'{doc_dir.name}': only 1 version file. "
            "Add more versions for negotiation history (draft + signed is the minimum).",
            doc_dir,
        )

    if unsupported:
        names = ", ".join(f.name for f in unsupported[:5])
        report.add(
            "warning",
            "DOC_UNSUPPORTED_FILES",
            f"'{doc_dir.name}': unsupported files present ({names}). "
            "The engine ignores them; remove to keep the folder clean.",
            doc_dir,
        )

    # Hint about a hints.yaml file being present (informational)
    hints_file = doc_dir / "hints.yaml"
    if hints_file.exists():
        report.add(
            "ok", "DOC_HAS_HINTS", f"'{doc_dir.name}': hints.yaml present (used for ordering hints)"
        )


def _lint_duplicate_stems(doc_dir: Path, version_files: list[Path], report: LintReport) -> None:
    """Flag version files that share a filename stem across extensions.

    The pipeline keys each version by ``vf.stem`` (``pipeline.py``'s
    ``_compute_doc_result``, ``_ingest_file_tracked``/``_llm_segment_file``
    callers), so e.g. ``signed.pdf`` and ``signed.docx`` in the same folder
    collide: the second silently overwrites the first in ``version_trees``
    (and its ``_batch_custom_id``), one version of the negotiation record
    disappears, and ``corpus_doc["versions"]`` still reports both as mined
    (issue #95). Flag it here as a blocking error so users fix the layout
    before running ``compile``, rather than lose a version silently.

    Comparison is case-insensitive since most of the target filesystems
    (macOS, Windows) are case-insensitive-preserving in practice, and a
    ``Signed.pdf``/``signed.docx`` pair collides identically.
    """
    by_stem: dict[str, list[Path]] = {}
    for vf in version_files:
        by_stem.setdefault(vf.stem.lower(), []).append(vf)

    for stem, files in sorted(by_stem.items()):
        if len(files) < 2:
            continue
        names = ", ".join(f.name for f in sorted(files, key=lambda f: f.name))
        report.add(
            "error",
            "DOC_DUPLICATE_VERSION_STEM",
            f"'{doc_dir.name}': multiple files share the stem '{stem}' ({names}). "
            "The engine keys each version by filename stem, so one of these will "
            "silently overwrite the other during compile. Rename the files to "
            "unique stems (e.g. 'signed-pdf.pdf', 'signed-docx.docx').",
            doc_dir,
        )


def _lint_config(config_path: Path, report: LintReport) -> None:
    """Check the engine config YAML."""
    if not config_path.exists():
        report.add(
            "error", "CONFIG_NOT_FOUND", f"Config file not found: {config_path}", config_path
        )
        return

    try:
        raw: Any = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        report.add(
            "error", "CONFIG_INVALID_YAML", f"Config file is not valid YAML: {exc}", config_path
        )
        return

    if not isinstance(raw, dict):
        report.add(
            "error",
            "CONFIG_NOT_A_MAPPING",
            "Config file must be a YAML mapping at the top level.",
            config_path,
        )
        return

    report.add("ok", "CONFIG_VALID_YAML", f"Config file is valid YAML: {config_path}")

    # agreement_type
    at = raw.get("agreement_type", {})
    if not isinstance(at, dict) or not at.get("id") or not at.get("name"):
        report.add(
            "error",
            "CONFIG_MISSING_AGREEMENT_TYPE",
            "agreement_type.id and agreement_type.name are required.",
            config_path,
        )
    else:
        report.add("ok", "CONFIG_AGREEMENT_TYPE", f"Agreement type: {at['name']} (id={at['id']})")

    # taxonomy
    tax_val = raw.get("taxonomy")
    if not tax_val:
        report.add(
            "error", "CONFIG_MISSING_TAXONOMY", "taxonomy path is required in config.", config_path
        )
    else:
        # Resolve via the config loader's own resolver so the ``builtin:``
        # scheme (and relative paths) are honoured identically — a literal
        # join here rejected ``builtin:...`` as a bogus path (issue #182).
        from playbook_engine.config import ConfigError, resolve_taxonomy_path

        try:
            tax_path = resolve_taxonomy_path(str(tax_val), config_path.parent)
        except ConfigError as exc:
            report.add("error", "CONFIG_TAXONOMY_NOT_FOUND", str(exc), config_path)
        else:
            report.add(
                "ok", "CONFIG_TAXONOMY_EXISTS", f"Taxonomy file exists: {tax_path.name}", tax_path
            )

    # baseline template (optional)
    bl = raw.get("baseline", {})
    template_val = bl.get("template") if isinstance(bl, dict) else None
    if template_val:
        tpl_path = (config_path.parent / template_val).resolve()
        if not tpl_path.is_file():
            report.add(
                "error",
                "CONFIG_TEMPLATE_NOT_FOUND",
                f"baseline.template not found: {tpl_path}",
                tpl_path,
            )
        else:
            report.add(
                "ok", "CONFIG_TEMPLATE_EXISTS", f"Template file exists: {tpl_path.name}", tpl_path
            )
    else:
        report.add(
            "warning",
            "CONFIG_NO_TEMPLATE",
            "No baseline template configured. "
            "An emergent playbook will be built from deal observations only; "
            "positions will not have an our_standard reference. "
            "Add baseline.template if you have a canonical template.",
        )

    # segmentation.llm credentials (issue #131): lint-corpus is the documented
    # preflight tool, so it must catch a missing ANTHROPIC_API_KEY here rather
    # than let a user discover it only when ``mine``/``compile``/``judge``
    # itself refuses to run (or, before that fix, after docling had already
    # ground through extraction).
    seg = raw.get("segmentation", {})
    if isinstance(seg, dict) and seg.get("llm") and not os.environ.get("ANTHROPIC_API_KEY"):
        report.add(
            "error",
            "CONFIG_SEGMENTATION_LLM_NO_CREDENTIALS",
            "segmentation.llm is enabled but ANTHROPIC_API_KEY is not set. "
            "Set the ANTHROPIC_API_KEY environment variable before running "
            "mine/compile/judge (see README.md), or run the "
            "playbook-from-corpus skill in Claude Code, which performs the "
            "judgment stages on your Claude plan without an API key. LLM "
            "segmentation currently requires an API key — see "
            "docs/PLAN-FIRST.md.",
            config_path,
        )
