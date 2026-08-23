"""Corpus-layout linter — pre-flight check before a compile run.

Reports what is missing or misconfigured in a corpus directory so a
non-engineer can fix it before running ``playbook mine``.

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
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from playbook_engine.config import (
    AGREEMENT_TYPE_KEYS,
    BASELINE_KEYS,
    CLASSIFICATION_KEYS,
    EXTRACTION_KEYS,
    PERSPECTIVE_KEYS,
    PROVENANCE_KEYS,
    SEGMENTATION_KEYS,
    TOP_LEVEL_KEYS,
    unknown_key_message,
)
from playbook_engine.pipeline import (
    _LEGACY_EXTENSIONS,
    _LEGACY_FORMAT_INSTRUCTION,
    _SUPPORTED_EXTENSIONS,
    _discover_versions,
)

_MIN_VERSIONS_FOR_COMPARISON = 2
_IGNORED_STEMS = frozenset({"hints"})  # hints.yaml is intentional, not a stray file

# Valid values for extraction.extractor (issue #82) — duplicated as a literal
# rather than imported from extraction._VALID_EXTRACTORS/config._VALID_EXTRACTORS,
# mirroring config.py's own documented precedent (its comment directly above
# its copy of this same set) of each module owning its validation vocabulary
# instead of reaching into another module's private constant.
_VALID_EXTRACTORS = frozenset({"docling", "legacy", "auto"})


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
# Symlink escape / dangle scan
# ---------------------------------------------------------------------------

# How many offending paths to name inline before collapsing into "+N more" —
# a symlink-staged 161-version corpus would otherwise dump 161 lines.
_SYMLINK_NAMES_CAP = 5

# Extensions worth reporting on when they arrive as a broken symlink. Anything
# the engine could conceivably have read: the supported set plus the legacy
# .doc set, so a corpus of dangling .doc links is still explained rather than
# silently ignored.
_LINKABLE_EXTENSIONS = _SUPPORTED_EXTENSIONS | _LEGACY_EXTENSIONS


@dataclass(frozen=True)
class SymlinkScan:
    """Symlinked version files found in a corpus, split by how they behave.

    ``dangling`` links resolve to nothing on this machine — ``Path.is_file()``
    is ``False`` for them, so :func:`_discover_versions` drops them without a
    word and the corpus presents as "no supported files found" (issues #121,
    #130). That is the failure this scan exists to name.

    ``escaping`` links do resolve here, but to a target outside the corpus
    root. They work on this host and break the moment the corpus tree is read
    from somewhere that cannot see the target path — most commonly
    ``docker run -v "$CORPUS:/work/corpus:ro"``, where only the corpus dir is
    mounted. Advisory, not blocking: on a plain host run they are fine.
    """

    dangling: dict[Path, list[Path]] = field(default_factory=dict)
    escaping: dict[Path, list[Path]] = field(default_factory=dict)

    @property
    def dangling_count(self) -> int:
        return sum(len(v) for v in self.dangling.values())

    @property
    def escaping_count(self) -> int:
        return sum(len(v) for v in self.escaping.values())


def scan_symlinks(corpus_dir: Path, doc_dirs: list[Path]) -> SymlinkScan:
    """Classify every symlinked version file under *doc_dirs*.

    Args:
        corpus_dir: Corpus root; a link resolving outside it is "escaping".
        doc_dirs:   Per-agreement directories to scan (non-recursive — the
                    engine walker is non-recursive too).

    Returns:
        A :class:`SymlinkScan`.
    """
    corpus_root = corpus_dir.resolve()
    scan = SymlinkScan()
    for doc_dir in doc_dirs:
        dangling: list[Path] = []
        escaping: list[Path] = []
        try:
            entries = sorted(doc_dir.iterdir())
        except OSError:
            continue
        for entry in entries:
            if not entry.is_symlink():
                continue
            if entry.suffix.lower() not in _LINKABLE_EXTENSIONS:
                continue
            try:
                target = entry.resolve(strict=True)
            except (OSError, RuntimeError):
                # Missing target, or a symlink loop — either way the engine
                # cannot read it.
                dangling.append(entry)
                continue
            if corpus_root not in target.parents:
                escaping.append(entry)
        if dangling:
            scan.dangling[doc_dir] = dangling
        if escaping:
            scan.escaping[doc_dir] = escaping
    return scan


def _sample(paths: list[Path], corpus_dir: Path) -> str:
    """Render up to :data:`_SYMLINK_NAMES_CAP` paths relative to *corpus_dir*."""
    shown = paths[:_SYMLINK_NAMES_CAP]
    rendered = ", ".join(str(_relative(p, corpus_dir)) for p in shown)
    extra = len(paths) - len(shown)
    return f"{rendered} (+{extra} more)" if extra else rendered


def _relative(path: Path, base: Path) -> Path:
    try:
        return path.relative_to(base)
    except ValueError:
        return path


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

    # Broken/escaping symlinks are scanned BEFORE the per-doc-dir walk so the
    # walk can report the real cause. A dangling symlink is invisible to
    # ``Path.is_file()``, so without this the whole corpus presents as
    # "no supported files found" — the exact mis-diagnosis that made a
    # symlink-staged corpus look empty under `docker run -v ...:ro` (#121/#130).
    symlinks = scan_symlinks(corpus_dir, doc_dirs)
    _report_symlinks(symlinks, corpus_dir, report)

    total_supported = 0
    # Lower-cased suffixes across every discovered version file in the
    # corpus (e.g. {".docx", ".pdf"}) — collected here, alongside the
    # existing per-doc-dir walk, so the extraction-environment checks below
    # (issue #82) know whether a missing docling/pandoc binary matters for
    # THIS corpus without re-walking the tree.
    corpus_suffixes: set[str] = set()
    for doc_dir in doc_dirs:
        _lint_doc_dir(doc_dir, report, dangling=symlinks.dangling.get(doc_dir, []))
        versions = _discover_versions(doc_dir)
        total_supported += len(versions)
        corpus_suffixes.update(vf.suffix.lower() for vf in versions)

    # Only claim "no supported files" when that is actually the problem. When
    # the corpus is full of broken symlinks, _report_symlinks above has already
    # said so precisely, and repeating the format complaint here would send the
    # reader off to check file extensions that are perfectly fine.
    if doc_dirs and total_supported == 0 and not symlinks.dangling:
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
        _lint_config(config_path, report, corpus_suffixes)

    return report


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _report_symlinks(scan: SymlinkScan, corpus_dir: Path, report: LintReport) -> None:
    """Turn a :class:`SymlinkScan` into lint findings.

    Dangling links are a blocking ERROR: they are silently dropped by the
    walker, so the run would otherwise "succeed" over a corpus with most of
    its documents missing. Escaping-but-resolvable links are a WARNING: they
    work here and break elsewhere, which is worth saying once but is not a
    reason to refuse a host-local run.
    """
    if scan.dangling:
        n = scan.dangling_count
        report.add(
            "error",
            "CORPUS_DANGLING_SYMLINKS",
            f"{n} file(s) in this corpus are symlinks whose target does not exist "
            f"here, across {len(scan.dangling)} document folder(s): "
            f"{_sample([q for v in scan.dangling.values() for q in v], corpus_dir)}. "
            "The engine cannot read them and would silently skip every one, so this "
            "corpus would compile as if those documents were never there. This is "
            "what a symlink-staged corpus looks like from inside a container: "
            '`docker run -v "$CORPUS:/work/corpus:ro"` mounts only the corpus '
            "directory, so absolute symlinks pointing outside it resolve to nothing. "
            "Fix: re-run `playbook stage <src> --out <corpus>` (real copies are the "
            "default), or point the corpus at a directory whose symlink targets are "
            "visible from where the engine runs.",
            corpus_dir,
        )
    if scan.escaping:
        n = scan.escaping_count
        report.add(
            "warning",
            "CORPUS_SYMLINKS_ESCAPE_ROOT",
            f"{n} file(s) in this corpus are symlinks pointing outside the corpus "
            f"directory: {_sample([q for v in scan.escaping.values() for q in v], corpus_dir)}. "
            "They resolve on this machine, so a host run is fine — but only the "
            "corpus directory is mounted into the container, so these will dangle "
            "(and be silently skipped) under `make docker-run`. Re-stage without "
            "`--symlink` to get real copies if you plan to run in Docker.",
            corpus_dir,
        )


def _lint_doc_dir(doc_dir: Path, report: LintReport, dangling: list[Path] | None = None) -> None:
    """Check one document subdirectory.

    ``dangling``: broken symlinks already found in *doc_dir* by
    :func:`scan_symlinks`. They are the reason the directory can look empty,
    so when there are any, the DOC_NO_SUPPORTED_FILES error is suppressed —
    the corpus-level CORPUS_DANGLING_SYMLINKS error names the real cause and
    the real fix.
    """
    dangling = dangling or []
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
        if not dangling:
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
    before running ``mine``, rather than lose a version silently.

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


def _lint_config(config_path: Path, report: LintReport, corpus_suffixes: set[str]) -> None:
    """Check the engine config YAML.

    ``corpus_suffixes``: lower-cased extensions across every version file in
    the corpus (``lint_corpus``'s per-doc-dir walk, done once and passed in
    rather than re-walked here) — used by the extraction-environment checks
    below (issue #82) to scope the docling/pandoc PDF/RTF checks to corpora
    that actually contain those formats.
    """
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

    # Unknown/typo'd config keys (issue #53): load_config rejects these
    # outright (fail-closed), but lint-corpus is the documented preflight
    # tool, so it must catch them here too rather than let a typo like
    # top-level `provenence:` (silently disabling pseudonymization) or
    # `segmentation: {llm-first: true}` (silently disabling LLM segmentation)
    # slip through as "OK — no errors". Mirrors load_config's checks using
    # the SAME known-key sets (playbook_engine.config) so the two never
    # drift apart — same pattern as resolve_taxonomy_path() reuse above.
    top_level_unknown = unknown_key_message(raw, TOP_LEVEL_KEYS)
    if top_level_unknown:
        report.add("error", "CONFIG_UNKNOWN_KEY", top_level_unknown, config_path)
    for section_key, known_keys in (
        ("agreement_type", AGREEMENT_TYPE_KEYS),
        ("baseline", BASELINE_KEYS),
        ("provenance", PROVENANCE_KEYS),
        ("perspective", PERSPECTIVE_KEYS),
        ("segmentation", SEGMENTATION_KEYS),
        ("classification", CLASSIFICATION_KEYS),
        ("extraction", EXTRACTION_KEYS),
    ):
        section_raw = raw.get(section_key)
        if not isinstance(section_raw, dict):
            continue  # missing/malformed section is reported elsewhere
        section_unknown = unknown_key_message(section_raw, known_keys, section_key)
        if section_unknown:
            report.add("error", "CONFIG_UNKNOWN_KEY", section_unknown, config_path)

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
    if template_val and not isinstance(template_val, str):
        report.add(
            "error",
            "CONFIG_TEMPLATE_INVALID",
            f"baseline.template must be a path string or null, got {type(template_val).__name__}",
            config_path,
        )
    elif template_val:
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

    # provenance section (issue #56): the scaffold (staging.py) marks
    # our_party_aliases "REQUIRED for provenance" and claims "mine warns if
    # none match", but mine's alias sanity check (pipeline.py) only fires
    # when aliases are configured-but-unmatched, and provenance_detector.py
    # silently defaults every document to
    # counterparty_paper/basis=no_aliases_configured when the list is empty
    # -- with no warning anywhere else in the pipeline. lint-corpus is the
    # documented preflight tool, so it must catch both: (1) a malformed
    # provenance section (mirroring load_config's own checks in config.py,
    # so lint-corpus never says "OK" about a config that ``mine`` will
    # refuse to load), reported as an ERROR since it is fatal; and (2) a
    # present-but-empty our_party_aliases list, reported as a WARNING since
    # the pipeline runs safely (just with degraded provenance) without it.
    prov = raw.get("provenance", {})
    if not isinstance(prov, dict):
        # A bare `provenance:` key (present but null, or any other
        # non-mapping scalar) parses fine as YAML but load_config rejects it
        # outright ("config.provenance must be a mapping"). Report it as an
        # ERROR here -- not the milder CONFIG_NO_OUR_PARTY_ALIASES warning
        # below -- so this doesn't silently pass as "OK — no errors" for a
        # config mine will hard-fail on.
        report.add(
            "error",
            "CONFIG_PROVENANCE_INVALID",
            f"provenance must be a mapping, got {type(prov).__name__}.",
            config_path,
        )
    else:
        aliases_raw = prov.get("our_party_aliases", [])
        if not isinstance(aliases_raw, list):
            # Same reasoning as above: load_config raises
            # "provenance.our_party_aliases must be a list" for this shape
            # (e.g. a bare `our_party_aliases:` key, or a string instead of
            # a list) -- fatal, so it's an ERROR, not a warning.
            report.add(
                "error",
                "CONFIG_ALIASES_INVALID",
                f"provenance.our_party_aliases must be a list, got {type(aliases_raw).__name__}.",
                config_path,
            )
        elif not [a for a in aliases_raw if a]:
            report.add(
                "warning",
                "CONFIG_NO_OUR_PARTY_ALIASES",
                "provenance.our_party_aliases is empty — every document will "
                "default to counterparty_paper (provenance cannot be "
                "determined); list every form of your own party's name from "
                "the recitals.",
                config_path,
            )

    # segmentation.llm credentials (issue #131): lint-corpus is the documented
    # preflight tool, so it must catch a missing ANTHROPIC_API_KEY here rather
    # than let a user discover it only when ``mine``/``judge``
    # itself refuses to run (or, before that fix, after docling had already
    # ground through extraction).
    #
    # Agent-as-segmenter (issue #191) is a key-free path: cli._llm_segmentation_kwargs
    # checks segmentation.agent BEFORE this same ANTHROPIC_API_KEY gate and returns
    # early (store-backed, no live call) when it's set -- that gate is only for the
    # live-LLM path. A config that sets both llm and agent (equivalent to the engine;
    # see config.py's SegmentationConfig construction) must mirror that precedence
    # here, or this preflight check contradicts the very command it precedes (issue
    # #286 / public #68).
    seg = raw.get("segmentation", {})
    if (
        isinstance(seg, dict)
        and seg.get("llm")
        and not seg.get("agent")
        and not os.environ.get("ANTHROPIC_API_KEY")
    ):
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

    # extraction environment (issue #82): lint-corpus is the documented
    # preflight tool (see the ANTHROPIC_API_KEY gate directly above, which
    # these are modeled on), so it must also catch a broken docling/pandoc
    # story before mine/compile burns LLM budget mining a corpus that just
    # went through a silently degraded backend -- the Jul 14 host run
    # (161/161 legacy, no OCR, no heading structure) is exactly the run a
    # 200ms preflight check here would have stopped.
    #
    # Gating: extraction.extract_blocks (docling/pdfplumber/pandoc) is only
    # ever reached from the LLM-segmentation path
    # (pipeline._compute_doc_result's use_llm_segmentation branch ->
    # _llm_segment_file). The deterministic ingest path
    # (_ingest_file_tracked -> ingest_docx/ingest_rtf/ingest_pdf, backed by
    # python-docx/striprtf/pdfplumber directly) never calls it -- see
    # ExtractionConfig's own docstring in config.py ("a declared extractor
    # has nothing to govern" there) -- so these checks must be gated the
    # same way the real runtime preflight is (cli._llm_segmentation_kwargs,
    # and segment_cmd's inline copy of it) or they misfire on every
    # deterministic-path corpus. Concretely: this repo's own
    # examples/judge-fixture/ corpus ships three .rtf files with no
    # segmentation: section at all -- an ungated CONFIG_EXTRACTION_RTF_NO_PANDOC
    # would fire on any docling+pandoc-less host and break the documented
    # quickstart's "no errors, 2 warning(s)" marker
    # (tests/test_quickstart.py), the same failure mode
    # tests/test_cli.py::test_mine_docling_declared_without_segmentation_llm_never_checked
    # already guards against for the runtime preflight. ``agent`` is OR'd
    # with ``llm`` (not AND-NOT'd like the credentials gate above) because
    # this mirrors load_config's own resolution
    # (``llm=bool(seg_raw.get("llm")) or agent_seg``, config.py) -- the
    # agent path still calls extract_blocks, it just segments key-free once
    # extraction is done.
    #
    # Config reading: reads the raw ``extraction`` mapping the same way
    # every other section in this function does (raw YAML, not a resolved
    # EngineConfig) so one broken section doesn't collapse this function's
    # independent, per-field reporting down to load_config's first
    # ConfigError.
    #
    # Shape validation: a value load_config would reject -- ``extraction:``
    # present but not a mapping, or ``extraction.extractor`` present but
    # outside the docling/legacy/auto vocabulary load_config uses
    # (config.py's EXTRACTION_KEYS/_VALID_EXTRACTORS -- this module's own
    # _VALID_EXTRACTORS copy above) -- is reported as an ERROR here,
    # mirroring the CONFIG_PROVENANCE_INVALID / CONFIG_ALIASES_INVALID
    # pattern above: this is precisely the raw-vs-resolved gap that bit #68,
    # and lint-corpus must never say "OK" about a config ``mine``
    # will refuse to load (the same invariant the provenance section above
    # states explicitly). The CONFIG_UNKNOWN_KEY check above already flags
    # an unknown key inside extraction: (e.g. a typo'd ``extractr``); this
    # catches the section being the wrong shape entirely, or ``extractor``
    # holding a value outside the allowed vocabulary (e.g. a typo'd
    # ``doclng``).
    #
    # This shape check runs unconditionally -- deliberately NOT nested under
    # ``extraction_relevant`` below -- because load_config validates
    # extraction: regardless of segmentation.llm/agent. Only the
    # docling/pandoc environment checks further down are gated on
    # extraction_relevant: those ask "is the environment broken for the
    # extraction that would actually run", which is meaningless when
    # extract_blocks is never reached (see the Gating note above).
    extr_shape = raw.get("extraction", {})
    if not isinstance(extr_shape, dict):
        report.add(
            "error",
            "CONFIG_EXTRACTION_INVALID",
            f"extraction must be a mapping, got {type(extr_shape).__name__}.",
            config_path,
        )
    else:
        extractor_declared = extr_shape.get("extractor", "auto")
        if not isinstance(extractor_declared, str) or extractor_declared not in _VALID_EXTRACTORS:
            report.add(
                "error",
                "CONFIG_EXTRACTION_INVALID_EXTRACTOR",
                "extraction.extractor must be one of 'docling', 'legacy', or "
                f"'auto' (got {extractor_declared!r}).",
                config_path,
            )

    extraction_relevant = isinstance(seg, dict) and (bool(seg.get("llm")) or bool(seg.get("agent")))

    if extraction_relevant:
        # A value load_config would reject is already reported as an error
        # above; it falls back to "auto" here so these environment checks
        # still run against a sane default instead of skipping outright.
        extr = raw.get("extraction", {})
        extractor_raw = extr.get("extractor", "auto") if isinstance(extr, dict) else "auto"
        if extractor_raw not in _VALID_EXTRACTORS:
            extractor_raw = "auto"

        docling_present = shutil.which("docling") is not None

        if extractor_raw == "docling" and not docling_present:
            report.add(
                "error",
                "CONFIG_EXTRACTION_DOCLING_MISSING",
                "extraction.extractor is set to 'docling' but the docling "
                "binary was not found on PATH. Install docling, run this "
                "corpus inside the project's container (see Dockerfile), or "
                "set extraction.extractor to 'legacy' or 'auto' (or omit "
                "the extraction: section) to use the legacy adapters "
                "instead.",
                config_path,
            )

        if extractor_raw == "auto" and not docling_present and ".pdf" in corpus_suffixes:
            report.add(
                "warning",
                "CONFIG_EXTRACTION_AUTO_PDF_NO_DOCLING",
                "docling was not found on PATH and this corpus contains "
                ".pdf file(s). extraction.extractor defaults to 'auto', "
                "which falls back to the legacy pdfplumber adapter (no "
                "OCR) for PDFs -- scanned/image-only PDFs will yield "
                "garbage or no text. Install docling for better PDF "
                "extraction, or set extraction.extractor: legacy to accept "
                "this deliberately and silence the warning.",
                config_path,
            )

        # A declared "legacy" always uses the pandoc-backed legacy RTF
        # adapter, even when docling IS on PATH --
        # extraction._resolve_extractor_env resolves a declared "legacy" to
        # "legacy" verbatim regardless of docling's availability (that's the
        # whole point: a deterministic, container-free run). "auto" only
        # reaches that same pandoc-backed adapter when docling is absent. A
        # declared "docling" never reaches it: either docling is present and
        # handles the RTF file directly, or docling is absent and the
        # CONFIG_EXTRACTION_DOCLING_MISSING error above already fires and
        # blocks the run before any file is touched -- a second, redundant
        # RTF-specific error here would misname which fix actually unblocks
        # the corpus.
        resolves_rtf_via_legacy = extractor_raw == "legacy" or (
            extractor_raw == "auto" and not docling_present
        )
        if ".rtf" in corpus_suffixes and resolves_rtf_via_legacy and shutil.which("pandoc") is None:
            report.add(
                "error",
                "CONFIG_EXTRACTION_RTF_NO_PANDOC",
                "This corpus contains .rtf file(s) and the extraction "
                "environment resolves to the legacy adapters "
                "(extraction.extractor is 'legacy', or 'auto' with docling "
                "not on PATH), but pandoc was not found on PATH. The "
                "legacy RTF adapter requires pandoc and raises "
                "ExtractionError at mine/compile time without it. Install "
                "pandoc (e.g. `brew install pandoc` or `apt-get install "
                "pandoc`), or install docling.",
                config_path,
            )
