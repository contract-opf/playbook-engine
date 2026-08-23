"""Run-level provenance manifest: what environment built this out-dir (issue #121).

Every artifact the engine writes already records *something* about how it was
produced — ``corpus_manifest.json`` records a per-version ``extractor`` and
``reason``, ``report.json`` records ``compiler_version``/``generated_at``,
``playbook.opf.json`` records ``identity.content_hash`` — but **nothing reads
any of it back on a later run**. The result is that an out-dir can be
re-processed under a materially different environment with no signal at all,
which is exactly how the Aug 22 incident happened:

    docling vanished from the host venv → ``extraction.extractor: auto``
    silently resolved to ``legacy`` → every version missed the
    ``extractor_env``-keyed extraction cache (#77) → the legacy adapter's
    canonical text differs from docling's → every version then missed the
    canonical-text-keyed segmentation cache → 43 of 44 documents quarantined
    as ``AgentSegmentationPending`` and observations fell ~2,400 → 66.

The reported error was two layers below the real fault. Nothing in the run
output named the extractor environment.

This module closes that loop with one small file, ``run_manifest.json``,
written into the out-dir by a successful ``mine``/``judge`` and **read back
by the next** ``mine``/``judge`` before any work starts.

Design notes
------------

*The happy path is silent.* The project owner's direction is that the
first-run experience must simply be correct, and that a warning is the
exception, not the product. So :func:`preflight` prints nothing at all when
the environment matches (or when there is no prior manifest — a fresh out-dir
is correct by construction). Counts are only computed, and files only read,
once a mismatch is already established.

*The mismatch path is written for a lawyer.* When something does differ,
:func:`render_report` emits plain English: what changed, what it means for
this run in terms of work redone, and the concrete fix — followed by one
copy-pasteable block of environment facts safe to paste into a public issue.

*Why a manifest rather than reading the extraction cache.* Issue #121
suggests inferring the previous environment from ``extraction_cache.jsonl``.
That does not actually work: the cache's *key* is a SHA-256 of the payload
(so ``extractor_env`` is not recoverable from it), and the stored *value*'s
``extractor`` field is the adapter that ran, not the environment the entry
was filed under — a docling-environment run with a per-file ``backend-error``
fallback stores ``"legacy"`` there (see
:meth:`playbook_engine.extraction.ExtractionCache.put`). A dedicated manifest
records the environment unambiguously, and covers the cache format versions,
engine build, and config identity that the extraction cache says nothing
about at all.

Confidentiality
---------------

The manifest and its copy-pasteable report contain **no customer data**:
no filesystem paths (an out-dir/corpus path embeds counterparty names —
see ``pipeline._alias_version_field``), no document ids, no party names, no
clause text. Only versions, booleans, closed-enum strings, opaque hashes and
integer counts. This is a hard invariant, enforced by
``tests/test_run_manifest.py``; do not add a field that can carry a name.
"""

from __future__ import annotations

import datetime
import json
import os
import platform
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from playbook_engine import __version__
from playbook_engine.artifact_store import _CACHE_FORMAT_VERSION as _ARTIFACT_CACHE_FORMAT_VERSION
from playbook_engine.artifact_store import make_config_fingerprint
from playbook_engine.extraction import _EXTRACTION_CACHE_FORMAT_VERSION, detect_extractor
from playbook_engine.llm_segmenter_batch import DEFAULT_EFFORT, PROMPT_VERSION, SCHEMA_HASH

if TYPE_CHECKING:  # pragma: no cover - typing only
    from playbook_engine.config import EngineConfig

#: Filename written into the out-dir.
RUN_MANIFEST_FILENAME = "run_manifest.json"

#: Schema version of the manifest file itself. Bump only on a shape change
#: that older readers cannot tolerate; :func:`read_run_manifest` treats an
#: unknown value as "no usable manifest" rather than raising, so an out-dir
#: written by a newer engine never crashes an older one.
RUN_MANIFEST_SCHEMA_VERSION = "1"

#: Escape-hatch env var for builds with no ``.git`` (the Docker image, a
#: wheel install). Set at build time — e.g.
#: ``ARG GIT_SHA`` / ``ENV PLAYBOOK_ENGINE_GIT_SHA=$GIT_SHA`` — so a stale
#: container still reports *which* build it is. Without it, ``git_sha`` is
#: simply ``None`` and the engine version alone carries the identity.
GIT_SHA_ENV_VAR = "PLAYBOOK_ENGINE_GIT_SHA"


# ---------------------------------------------------------------------------
# Environment capture
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunEnvironment:
    """The facts about *this* process that determine what an out-dir contains.

    Every field is either a version string, a closed-enum string, a boolean,
    or an opaque hash — never a path, name, or any document-derived text (see
    the module docstring's confidentiality note).
    """

    engine_version: str
    git_sha: str | None
    git_dirty: bool | None
    declared_extractor: str  # config.extraction.extractor: docling|legacy|auto
    resolved_extractor: str  # what "auto" actually resolves to right now
    # Nullable because a manifest written by an older engine may predate the
    # field: "unknown" and "false" must stay distinguishable, or a missing
    # value would read as "docling was absent" and fire a bogus finding.
    docling_available: bool | None
    pandoc_available: bool | None
    extraction_cache_format: str
    artifact_cache_format: str
    segmentation_mode: str  # deterministic|llm|llm-batch|agent
    segmentation_model: str
    segmentation_prompt_version: str
    segmentation_schema_hash: str
    segmentation_effort: str
    config_hash: str
    python_version: str
    platform: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> RunEnvironment:
        """Build from a stored dict, tolerating missing and wrong-typed keys.

        A manifest written by an older engine may lack a field this one knows
        about, and a hand-edited one may hold the wrong type. Both degrade to
        the "unknown" value (``""`` for strings, ``None`` for booleans and
        optional strings), which :func:`compare` skips outright — so an engine
        upgrade never fires a spurious "changed" finding for a field nobody
        touched, and a mangled manifest never crashes a run.

        Written out field-by-field rather than by ``**raw`` splat so the
        stored file can never inject an unexpected key or type into the
        dataclass.
        """

        def _s(key: str) -> str:
            value = raw.get(key)
            return value if isinstance(value, str) else ""

        def _os(key: str) -> str | None:
            value = raw.get(key)
            return value if isinstance(value, str) and value else None

        def _ob(key: str) -> bool | None:
            value = raw.get(key)
            return value if isinstance(value, bool) else None

        return cls(
            engine_version=_s("engine_version"),
            git_sha=_os("git_sha"),
            git_dirty=_ob("git_dirty"),
            declared_extractor=_s("declared_extractor"),
            resolved_extractor=_s("resolved_extractor"),
            docling_available=_ob("docling_available"),
            pandoc_available=_ob("pandoc_available"),
            extraction_cache_format=_s("extraction_cache_format"),
            artifact_cache_format=_s("artifact_cache_format"),
            segmentation_mode=_s("segmentation_mode"),
            segmentation_model=_s("segmentation_model"),
            segmentation_prompt_version=_s("segmentation_prompt_version"),
            segmentation_schema_hash=_s("segmentation_schema_hash"),
            segmentation_effort=_s("segmentation_effort"),
            config_hash=_s("config_hash"),
            python_version=_s("python_version"),
            platform=_s("platform"),
        )


@dataclass(frozen=True)
class RunManifest:
    """A :class:`RunEnvironment` plus when/how it was recorded."""

    schema_version: str
    written_at: str  # ISO-8601 UTC, seconds precision
    written_by: str  # "mine" | "judge"
    environment: RunEnvironment
    counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "written_at": self.written_at,
            "written_by": self.written_by,
            "environment": self.environment.to_dict(),
            "counts": dict(self.counts),
        }


def _git_sha_and_dirty() -> tuple[str | None, bool | None]:
    """Best-effort ``(short_sha, dirty)`` for the checkout this package lives in.

    Returns ``(None, None)`` for a non-git install (wheel, site-packages) or
    when ``git`` is not on PATH — the manifest simply carries no build
    identity beyond ``engine_version`` in that case, which is correct rather
    than fabricated. :data:`GIT_SHA_ENV_VAR` overrides the lookup entirely so
    a container image can bake in its own build sha (``dirty`` is then
    reported as ``False``: a baked image is by definition a fixed build).

    Never raises, and never blocks: a 5s timeout guards against a pathological
    repo, and the process is fully detached from the terminal (no stderr
    noise on a partial checkout).
    """
    baked = os.environ.get(GIT_SHA_ENV_VAR)
    if baked:
        return baked.strip()[:12], False

    repo_root = Path(__file__).resolve().parent.parent
    if not (repo_root / ".git").exists():
        return None, None

    def _git(*args: str) -> str | None:
        try:
            proc = subprocess.run(  # noqa: S603
                ["git", *args],
                cwd=repo_root,
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if proc.returncode != 0:
            return None
        return proc.stdout.strip()

    sha = _git("rev-parse", "--short=12", "HEAD")
    if sha is None:
        return None, None
    status = _git("status", "--porcelain")
    return sha, (bool(status) if status is not None else None)


def _segmentation_mode(config: EngineConfig) -> str:
    """Collapse the segmentation config booleans into one closed-enum label.

    Mirrors the branch order in ``cli._llm_segmentation_kwargs``: ``agent``
    wins over ``batch`` wins over ``llm``. The label exists so a change of
    segmentation *path* — which changes L1 output for identical source text,
    hence the ``use_llm_segmentation``/``use_batch_segmentation`` fields in
    ``pipeline``'s stage-cache fingerprint — is legible to a human in one
    word instead of three booleans.
    """
    if config.segmentation.agent:
        return "agent"
    if not config.segmentation.llm:
        return "deterministic"
    return "llm-batch" if config.segmentation.batch else "llm"


def compute_config_hash(config: EngineConfig) -> str:
    """Opaque, stable hash of every config value that affects engine output.

    Includes the *content* hash of the taxonomy file and of the canonical
    template (not their paths): editing either changes what the engine
    produces from byte-identical source documents, so both must move the
    hash. Paths themselves are deliberately excluded — they embed
    counterparty names, and moving an out-dir is not a semantic change.

    The result is a SHA-256 hex digest, so the manifest (and the
    copy-pasteable report) can carry config identity without carrying any
    config *content*: party aliases, known entities and author names all feed
    the hash but none of them are recoverable from it.
    """
    payload: dict[str, Any] = {
        "agreement_type": {
            "id": config.agreement_type.id,
            "name": config.agreement_type.name,
            "description": config.agreement_type.description,
            "aliases": sorted(config.agreement_type.aliases),
        },
        "baseline": {
            "has_canonical_template": config.baseline.has_canonical_template,
            "template_sha256": _sha256_file_or_none(config.baseline.template_path),
        },
        "perspective": {
            "party": config.perspective.party,
            "counterparty_type": config.perspective.counterparty_type,
        },
        "provenance": {
            "our_party_aliases": sorted(config.provenance.our_party_aliases),
            "our_authors": sorted(config.provenance.our_authors),
            "known_entities": sorted(config.provenance.known_entities),
            "min_evidence_n": config.provenance.min_evidence_n,
        },
        "segmentation": {
            "llm": config.segmentation.llm,
            "batch": config.segmentation.batch,
            "cache": config.segmentation.cache,
            "normalize_trail": config.segmentation.normalize_trail,
            "agent": config.segmentation.agent,
            "model": config.segmentation.model,
        },
        "classification": {
            "ambiguity_threshold": config.classification.ambiguity_threshold,
            "auto_classify_threshold": config.classification.auto_classify_threshold,
        },
        "extraction": {
            "extractor": config.extraction.extractor,
            "max_fallback": config.extraction.max_fallback,
        },
        "scan_role_words_extra": sorted(config.scan_role_words_extra),
        "scan_stopwords_extra": sorted(config.scan_stopwords_extra),
        "taxonomy_sha256": _sha256_file_or_none(config.taxonomy_path),
    }
    return make_config_fingerprint(payload)


def _sha256_file_or_none(path: Path | None) -> str | None:
    """SHA-256 of *path*'s bytes, or ``None`` when absent/unreadable."""
    if path is None:
        return None
    try:
        return _sha256_bytes(path.read_bytes())
    except OSError:
        return None


def _sha256_bytes(data: bytes) -> str:
    import hashlib  # noqa: PLC0415

    return hashlib.sha256(data).hexdigest()


def capture_environment(config: EngineConfig, corpus_dir: Path) -> RunEnvironment:
    """Snapshot the environment this process would run *config* under.

    ``resolved_extractor`` deliberately uses the same
    :func:`~playbook_engine.extraction.detect_extractor` PATH check the
    pipeline's own stage-cache fingerprint uses (``pipeline`` line ~2689),
    combined with the declared override the way
    :func:`~playbook_engine.extraction._resolve_extractor_env` does — so this
    field is exactly the ``extractor_env`` that will key every extraction
    cache lookup in the run about to start.

    *corpus_dir* is passed only to mirror that same call shape; the PATH
    check ignores its content (it is constant across every file in a run).
    It is never recorded.
    """
    declared = config.extraction.extractor
    detected = detect_extractor(corpus_dir)
    resolved = detected if declared == "auto" else declared
    sha, dirty = _git_sha_and_dirty()
    return RunEnvironment(
        engine_version=__version__,
        git_sha=sha,
        git_dirty=dirty,
        declared_extractor=declared,
        resolved_extractor=resolved,
        docling_available=shutil.which("docling") is not None,
        pandoc_available=shutil.which("pandoc") is not None,
        extraction_cache_format=_EXTRACTION_CACHE_FORMAT_VERSION,
        artifact_cache_format=_ARTIFACT_CACHE_FORMAT_VERSION,
        segmentation_mode=_segmentation_mode(config),
        segmentation_model=config.segmentation.model,
        segmentation_prompt_version=PROMPT_VERSION,
        segmentation_schema_hash=SCHEMA_HASH,
        segmentation_effort=DEFAULT_EFFORT,
        config_hash=compute_config_hash(config),
        python_version=platform.python_version(),
        platform=platform.platform(terse=True),
    )


# ---------------------------------------------------------------------------
# Counts — only ever computed when something already went wrong
# ---------------------------------------------------------------------------


def _count_lines(path: Path) -> int:
    """Number of newline-terminated records in *path* (0 when absent).

    Counts bytes rather than parsing JSON: ``extraction_cache.jsonl`` for a
    real corpus is tens of megabytes and this runs on the failure path, where
    a fast approximate count of "how much work is at stake" is worth far more
    than an exact parse.
    """
    if not path.exists():
        return 0
    total = 0
    try:
        with path.open("rb") as fh:
            while chunk := fh.read(1 << 20):
                total += chunk.count(b"\n")
    except OSError:
        return 0
    return total


def collect_counts(out_dir: Path) -> dict[str, int]:
    """How much work this out-dir currently represents.

    Feeds the plain-English consequence line ("161 documents would be re-read
    from scratch"). Every value degrades to ``0`` rather than raising — this
    runs while reporting a problem and must never become the problem.
    """
    counts: dict[str, int] = {
        "documents": 0,
        "versions": 0,
        "extraction_cache_entries": _count_lines(out_dir / "extraction_cache.jsonl"),
        "segmentation_cache_entries": _count_lines(out_dir / "segment" / "cache.jsonl"),
        "observations": _count_lines(out_dir / "observations.jsonl"),
        "judge_verdicts": _count_lines(out_dir / "judge" / "verdicts.jsonl"),
    }
    manifest_path = out_dir / "corpus_manifest.json"
    if manifest_path.exists():
        try:
            docs = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            docs = []
        if isinstance(docs, list):
            counts["documents"] = len(docs)
            counts["versions"] = sum(
                len(d.get("version_ingest") or []) for d in docs if isinstance(d, dict)
            )
    return counts


# ---------------------------------------------------------------------------
# Read / write
# ---------------------------------------------------------------------------


def write_run_manifest(
    out_dir: Path,
    environment: RunEnvironment,
    *,
    command: str,
    counts: dict[str, int] | None = None,
) -> Path:
    """Atomically write ``run_manifest.json`` into *out_dir*; return its path.

    Called at the END of a successful ``mine``/``judge``, never at the start:
    a manifest is a claim about what produced the artifacts sitting next to
    it, and a run that crashed halfway produced nothing worth claiming. (A
    crashed run therefore leaves the *previous* manifest in place, which is
    the correct description of the out-dir's contents.)
    """
    manifest = RunManifest(
        schema_version=RUN_MANIFEST_SCHEMA_VERSION,
        written_at=datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        written_by=command,
        environment=environment,
        counts=counts if counts is not None else collect_counts(out_dir),
    )
    path = out_dir / RUN_MANIFEST_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(manifest.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)
    return path


def read_run_manifest(out_dir: Path) -> RunManifest | None:
    """Load *out_dir*'s manifest, or ``None`` when there isn't a usable one.

    Returns ``None`` — never raises — for a missing file, unreadable file,
    malformed JSON, or a ``schema_version`` this engine does not know. A
    corrupt manifest must degrade to "first run" (silent, correct) rather
    than blocking a run that would otherwise succeed.
    """
    path = out_dir / RUN_MANIFEST_FILENAME
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(raw, dict):
        return None
    if raw.get("schema_version") != RUN_MANIFEST_SCHEMA_VERSION:
        return None
    env_raw = raw.get("environment")
    if not isinstance(env_raw, dict):
        return None
    counts_raw = raw.get("counts")
    counts = (
        {str(k): int(v) for k, v in counts_raw.items() if isinstance(v, int)}
        if isinstance(counts_raw, dict)
        else {}
    )
    try:
        environment = RunEnvironment.from_dict(env_raw)
    except TypeError:
        return None
    return RunManifest(
        schema_version=RUN_MANIFEST_SCHEMA_VERSION,
        written_at=str(raw.get("written_at", "")),
        written_by=str(raw.get("written_by", "")),
        environment=environment,
        counts=counts,
    )


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Finding:
    """One plain-English difference between the stored and current environment.

    Attributes:
        code:        Stable machine id, e.g. ``"extractor-environment-changed"``.
                     Appears in the copy-pasteable report so a maintainer can
                     grep for it; never shown as the headline.
        blocking:    ``True`` when proceeding would silently discard or
                     invalidate work already on disk. Blocking findings stop
                     the run; non-blocking ones print and continue.
        headline:    One sentence, no jargon, naming what is different.
        consequence: Lines describing what this run would actually do about
                     it — the "161 documents would be re-read" part.
        fix:         Lines describing how to make it right.
    """

    code: str
    blocking: bool
    headline: str
    consequence: list[str]
    fix: list[str]


def _known(*values: Any) -> bool:
    """True when every value is a real recorded value, not an unknown.

    An older manifest that predates a field stores nothing for it; comparing
    ``"" != "3"`` would fire a "changed" finding on the very first run after
    an upgrade, for a field nobody actually changed. Skipping unknowns keeps
    the upgrade path silent, which is the whole point.
    """
    return all(v is not None and v != "" for v in values)


def _version_tuple(text: str) -> tuple[int, ...] | None:
    """Parse a dotted numeric version, or ``None`` if it isn't one."""
    parts = text.split(".")
    try:
        return tuple(int(p) for p in parts)
    except ValueError:
        return None


def compare(  # noqa: C901
    previous: RunEnvironment,
    current: RunEnvironment,
    counts: dict[str, int],
) -> list[Finding]:
    """Diff two environments into plain-English :class:`Finding` objects.

    Ordered most-severe first. Returns ``[]`` when nothing meaningful moved —
    the overwhelmingly common case, and the one that must stay silent.

    Deliberately NOT compared: ``python_version`` and ``platform``. Both are
    recorded (they matter in a bug report) but both legitimately differ
    between a host run and a container run of the same corpus, so comparing
    them would produce a warning on a completely healthy workflow — exactly
    the "warning nobody reads" failure mode this design exists to avoid.
    """
    findings: list[Finding] = []
    versions = counts.get("versions") or counts.get("extraction_cache_entries") or 0
    seg_entries = counts.get("segmentation_cache_entries", 0)
    docs = counts.get("documents", 0)

    # --- 1. The incident itself: the extractor environment moved -----------
    if _known(previous.resolved_extractor, current.resolved_extractor) and (
        previous.resolved_extractor != current.resolved_extractor
    ):
        lost_docling = (
            previous.resolved_extractor == "docling" and current.resolved_extractor == "legacy"
        )
        if lost_docling and not current.docling_available:
            headline = (
                "This output folder was built with docling, but docling isn't available here."
            )
        else:
            headline = (
                f"This output folder was built with the "
                f"{_reader_name(previous.resolved_extractor)}, but this run would use "
                f"the {_reader_name(current.resolved_extractor)}."
            )
        consequence = [
            f"All {versions or 'the'} document version(s) in this folder would be "
            "read again from scratch. The two readers produce different text for "
            "the same file, so none of the saved extractions can be reused.",
        ]
        if seg_entries:
            consequence.append(
                f"The {seg_entries} saved clause grouping(s) are matched to the old "
                "text, so they would not match either. Every document would be sent "
                "back for segmentation, and documents waiting on that get set aside "
                "(you would see them reported as segmentation-pending)."
            )
        if counts.get("judge_verdicts"):
            consequence.append(
                f"The {counts['judge_verdicts']} decision(s) already recorded in the "
                "judge store are keyed to the old clause grouping, so some of that "
                "review work would need to be redone."
            )
        if lost_docling:
            fix = [
                "Put docling back on this machine (`pip install docling`), or run "
                "this corpus inside the project's container (see Dockerfile), then "
                "run this command again. Nothing else needs to change.",
                "If you genuinely want to re-read everything with the built-in "
                "reader instead, put `extractor: legacy` under `extraction:` in "
                "your config and re-run with --accept-environment-change.",
            ]
        else:
            fix = [
                "Run in the same environment this folder was built in "
                f"({previous.resolved_extractor}), or start a fresh output folder "
                "for this environment.",
                "If the change is deliberate, re-run with "
                "--accept-environment-change to accept the rework.",
            ]
        findings.append(
            Finding(
                code="extractor-environment-changed",
                blocking=True,
                headline=headline,
                consequence=consequence,
                fix=fix,
            )
        )

    # --- 2. The silent-unreachable-cache case (format bumps 1→2→3) ---------
    if _known(previous.extraction_cache_format, current.extraction_cache_format) and (
        previous.extraction_cache_format != current.extraction_cache_format
    ):
        findings.append(
            Finding(
                code="extraction-cache-format-changed",
                blocking=True,
                headline=(
                    "This version of the engine reads documents differently than the "
                    "version that built this output folder, so none of the saved "
                    "extractions can be reused."
                ),
                consequence=[
                    f"All {versions or 'the'} document version(s) would be read again "
                    "from scratch (this is a one-time cost — after this run the saved "
                    "extractions are current again).",
                    "For scanned or image-heavy documents this can take a long time.",
                ],
                fix=[
                    "This is expected after an engine upgrade and the result will be "
                    "correct. Re-run with --accept-environment-change to go ahead.",
                    "If you did not mean to change engine version, check which engine "
                    "you are running (`playbook --version`) — a container image can be "
                    "older than your checkout.",
                ],
            )
        )

    # --- 3. Running an OLDER engine than built this folder -----------------
    if _known(previous.engine_version, current.engine_version) and (
        previous.engine_version != current.engine_version
    ):
        prev_v = _version_tuple(previous.engine_version)
        cur_v = _version_tuple(current.engine_version)
        downgrade = prev_v is not None and cur_v is not None and cur_v < prev_v
        if downgrade:
            findings.append(
                Finding(
                    code="engine-version-downgrade",
                    blocking=True,
                    headline=(
                        f"This output folder was built by engine "
                        f"{previous.engine_version}, but you are running the older "
                        f"{current.engine_version}."
                    ),
                    consequence=[
                        "Fixes that shaped this folder's results are not in the engine "
                        "you are running now. The run would very likely finish and look "
                        "successful while quietly undoing them.",
                    ],
                    fix=[
                        "Check what you are actually running: `playbook --version`. If "
                        "you are using the container, rebuild the image (`make "
                        "docker-build`) — a local image can be days behind your "
                        "checkout.",
                        "If you really do mean to run the older engine, re-run with "
                        "--accept-environment-change.",
                    ],
                )
            )
        else:
            findings.append(
                Finding(
                    code="engine-version-changed",
                    blocking=False,
                    headline=(
                        f"Engine upgraded since this folder was built: "
                        f"{previous.engine_version} → {current.engine_version}."
                    ),
                    consequence=[
                        "Results may differ from the last run for reasons that are "
                        "improvements, not errors.",
                    ],
                    fix=["Nothing to do. Noted here so a change in results isn't a surprise."],
                )
            )

    # --- 4. Segmentation identity moved: clause groupings won't match ------
    seg_prev = (
        previous.segmentation_mode,
        previous.segmentation_model,
        previous.segmentation_prompt_version,
        previous.segmentation_schema_hash,
        previous.segmentation_effort,
    )
    seg_cur = (
        current.segmentation_mode,
        current.segmentation_model,
        current.segmentation_prompt_version,
        current.segmentation_schema_hash,
        current.segmentation_effort,
    )
    if _known(*seg_prev, *seg_cur) and seg_prev != seg_cur:
        changed_mode = previous.segmentation_mode != current.segmentation_mode
        headline = (
            (
                f"The way documents get split into clauses changed "
                f"({previous.segmentation_mode} → {current.segmentation_mode})."
            )
            if changed_mode
            else "The clause-splitting model or prompt changed since this folder was built."
        )
        findings.append(
            Finding(
                code="segmentation-identity-changed",
                blocking=True,
                headline=headline,
                consequence=[
                    f"The {seg_entries or 'saved'} stored clause grouping(s) were "
                    "produced the old way, so they will not be reused. Every document "
                    "would be split again.",
                    "Clause-level decisions already recorded may not line up with the "
                    "new groupings.",
                ],
                fix=[
                    "Put the previous setting back in your config if this wasn't intended.",
                    "If it was intended, re-run with --accept-environment-change (and "
                    "expect to review the re-split clauses).",
                ],
            )
        )

    # --- 5. Config edited (advisory) ---------------------------------------
    if _known(previous.config_hash, current.config_hash) and (
        previous.config_hash != current.config_hash
    ):
        findings.append(
            Finding(
                code="config-changed",
                blocking=False,
                headline="Your config or taxonomy has been edited since this folder was built.",
                consequence=[
                    f"Results for the {docs or 'existing'} document(s) already in this "
                    "folder will be recomputed under the new settings.",
                ],
                fix=["Nothing to do if the edit was intentional."],
            )
        )

    # --- 6. Different engine build, same version (advisory) ----------------
    if (
        _known(previous.git_sha, current.git_sha)
        and previous.git_sha != current.git_sha
        and previous.engine_version == current.engine_version
    ):
        findings.append(
            Finding(
                code="engine-build-changed",
                blocking=False,
                headline=(
                    "Same engine version, different build — code changed without a version bump."
                ),
                consequence=["Results may differ slightly from the last run."],
                fix=["Nothing to do. Noted so a change in results isn't a surprise."],
            )
        )

    findings.sort(key=lambda f: not f.blocking)
    return findings


def _reader_name(extractor: str) -> str:
    """Human name for an extractor id, for use mid-sentence."""
    return {"docling": "docling reader", "legacy": "built-in reader"}.get(
        extractor, f"{extractor} reader"
    )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

#: Prose wraps to 72 columns: narrow enough to stay readable in a default
#: terminal, an email, and a pasted GitHub issue without re-flowing.
_WIDTH = 72
_RULE = "─" * _WIDTH
#: Label column in the paste block. Wide enough for the longest label
#: ("count segmentation_cache_entries"), so no row ever runs its value into
#: its label — a maintainer skims this block, and a single ragged line is
#: enough to make it look like output nobody proof-read.
_LABEL_WIDTH = 34
_PASTE_OPEN = "--------8<-------- playbook-engine environment report --------8<--------"
_PASTE_CLOSE = "--------8<--------------------- end ----------------------------8<--------"


def _fmt(value: Any) -> str:
    if value is None or value == "":
        return "(not recorded)"
    if value is True:
        return "yes"
    if value is False:
        return "no"
    return str(value)


def _row(label: str, value: Any) -> str:
    return f"  {label:<{_LABEL_WIDTH}}{value}"


def _pair(label: str, before: Any, after: Any) -> str:
    b, a = _fmt(before), _fmt(after)
    return _row(label, f"{b} -> {a}" if b != a else b)


def _para(text: str) -> list[str]:
    """Wrap a standalone paragraph to :data:`_WIDTH` at the report's indent."""
    import textwrap  # noqa: PLC0415

    return textwrap.wrap(text, width=_WIDTH, initial_indent="  ", subsequent_indent="  ")


def render_paste_block(
    findings: list[Finding],
    previous: RunManifest,
    current: RunEnvironment,
    counts: dict[str, int],
) -> str:
    """Render the copy-pasteable block for a maintainer bug report.

    Every line is a version, a boolean, a closed-enum string, an opaque hash,
    or an integer count. No paths, no document ids, no party names, no clause
    text — see the module docstring. This block is meant to be pasted into a
    PUBLIC issue tracker, so that invariant is not negotiable; if you add a
    line here, add a case to ``tests/test_run_manifest.py``'s leak test.

    Values are shown as ``before -> after`` when they differ and as a single
    value when they don't, so the thing that actually moved is visible at a
    glance rather than buried in two columns of identical text.
    """
    p = previous.environment
    lines = [
        _PASTE_OPEN,
        _row("findings", ", ".join(f.code for f in findings)),
        _pair("engine version", p.engine_version, current.engine_version),
        _pair("engine build (git)", p.git_sha, current.git_sha),
        _pair("engine build dirty", p.git_dirty, current.git_dirty),
        _pair("extractor declared", p.declared_extractor, current.declared_extractor),
        _pair("extractor resolved", p.resolved_extractor, current.resolved_extractor),
        _pair("docling on PATH", p.docling_available, current.docling_available),
        _pair("pandoc on PATH", p.pandoc_available, current.pandoc_available),
        _pair(
            "extraction cache format", p.extraction_cache_format, current.extraction_cache_format
        ),
        _pair("stage cache format", p.artifact_cache_format, current.artifact_cache_format),
        _pair("segmentation mode", p.segmentation_mode, current.segmentation_mode),
        _pair("segmentation model", p.segmentation_model, current.segmentation_model),
        _pair(
            "segmentation prompt",
            p.segmentation_prompt_version,
            current.segmentation_prompt_version,
        ),
        _pair(
            "segmentation schema",
            _short(p.segmentation_schema_hash),
            _short(current.segmentation_schema_hash),
        ),
        _pair("segmentation effort", p.segmentation_effort, current.segmentation_effort),
        _pair("config hash", _short(p.config_hash), _short(current.config_hash)),
        _pair("python", p.python_version, current.python_version),
        _pair("platform", p.platform, current.platform),
        _row(
            "manifest written",
            f"{_fmt(previous.written_at)} by '{_fmt(previous.written_by)}'",
        ),
    ]
    lines.extend(_row(f"count {key}", value) for key, value in sorted(counts.items()))
    lines.append(_PASTE_CLOSE)
    return "\n".join(lines)


def _short(value: str | None) -> str | None:
    """First 12 chars of a hex digest — enough to compare, short enough to read."""
    return value[:12] if value else value


def render_report(
    findings: list[Finding],
    previous: RunManifest,
    current: RunEnvironment,
    counts: dict[str, int],
    *,
    command: str,
    stopping: bool,
) -> str:
    """Render the full plain-English report for a BLOCKING mismatch.

    Shape, top to bottom: what changed → what it means for this run → how to
    fix it → whether we stopped → the copy-pasteable block. A lawyer reading
    only the first three lines should still know what to do; a maintainer
    receiving only the last block should still be able to diagnose it.

    Reserved for the blocking case on purpose. An advisory-only difference (a
    config edit, an engine upgrade) is a normal thing a person did on purpose,
    and answering it with forty lines and a bug-report block would teach them
    that this output is noise — which is precisely how the real warning gets
    skimmed past when it finally appears. Advisories get
    :func:`render_notes` instead.
    """
    title = (
        "Stopping: this output folder was built in a different environment"
        if stopping
        else "Heads up: something changed since this output folder was built"
    )
    out: list[str] = ["", _RULE, f"  {title}", _RULE, ""]

    for finding in findings:
        out.extend(_para(finding.headline))
        out.append("")
        out.append("  What that means for this run:")
        out.extend(_bullets(finding.consequence))
        out.append("")
        out.append("  How to fix it:")
        out.extend(_bullets(finding.fix))
        out.append("")

    if stopping:
        out.extend(
            _para(
                "Nothing has been changed and no work has been thrown away — "
                f"`playbook {command}` stopped before it started."
            )
        )
    elif any(f.blocking for f in findings):
        out.extend(
            _para(
                "Continuing anyway because --accept-environment-change was given. "
                f"`playbook {command}` will redo the work described above."
            )
        )
    else:
        out.extend(_para(f"Continuing with `playbook {command}`."))
    out.append("")
    out.extend(
        _para(
            "If you want help, copy everything between the lines below and send it "
            "to the maintainers. It contains no contract text, no party names, and "
            "no file paths — it is safe to paste into a public issue:"
        )
    )
    out.append("")
    out.append(render_paste_block(findings, previous, current, counts))
    out.append("")
    return "\n".join(out)


def render_notes(findings: list[Finding]) -> str:
    """Render advisory-only differences as one short ``note:`` line each.

    No rules, no bullets, no paste block — a config edit or an engine upgrade
    is something the user did deliberately, and all they need is one sentence
    confirming the engine noticed. Keeping this small is what preserves the
    signal value of :func:`render_report`'s full stop-the-run form.
    """
    lines: list[str] = []
    for finding in findings:
        lines.extend(_para(f"note: {finding.headline}"))
    return "\n".join(lines)


def _bullets(items: list[str]) -> list[str]:
    """Wrap *items* as ``  - `` bullets at :data:`_WIDTH` with hanging indent."""
    import textwrap  # noqa: PLC0415

    lines: list[str] = []
    for item in items:
        wrapped = textwrap.wrap(
            item, width=_WIDTH, initial_indent="    - ", subsequent_indent="      "
        )
        lines.extend(wrapped or ["    - "])
    return lines


# ---------------------------------------------------------------------------
# Preflight — the one function the CLI calls
# ---------------------------------------------------------------------------


class EnvironmentMismatch(Exception):
    """Raised by :func:`preflight` when a blocking difference is found.

    Carries the already-rendered, already-plain-English :attr:`report`; the
    CLI prints it verbatim and exits 1. Nothing about it needs formatting at
    the call site — the message a lawyer reads is composed here, once.
    """

    def __init__(self, report: str, findings: list[Finding]) -> None:
        super().__init__("run environment differs from the one that built this output folder")
        self.report = report
        self.findings = findings


def preflight(
    out_dir: Path,
    config: EngineConfig,
    corpus_dir: Path,
    *,
    command: str,
    echo: Any = None,
    accept_change: bool = False,
) -> RunEnvironment:
    """Compare this environment against *out_dir*'s stored manifest.

    Returns the captured :class:`RunEnvironment` so the caller can hand it
    straight to :func:`write_run_manifest` when the run succeeds, without
    capturing it twice (and without the two snapshots ever disagreeing).

    Silent — no output whatsoever — in all of the normal cases: no prior
    manifest (a fresh out-dir is correct by construction), an unreadable
    manifest, or an environment that matches. This is the design constraint,
    not an optimisation: the engine is meant to work without ever explaining
    itself, and any output here would train people to skim past the one time
    it matters.

    Args:
        command: ``"mine"`` or ``"judge"`` — named in the report so the
                 remediation text refers to the command the user actually ran.
        echo:    ``click.echo``-like sink. Receives the one-line ``note:``
                 form for advisory differences, or the full report when a
                 blocking difference is accepted via *accept_change*. Blocking
                 differences that are NOT accepted never go here — they raise.
        accept_change: The operator passed ``--accept-environment-change``.
                 Blocking findings are still explained in full (so the rework
                 is on the record) but printed instead of raised, and the run
                 proceeds.

    Raises:
        EnvironmentMismatch: A blocking difference was found and
                             *accept_change* is False.
    """
    current = capture_environment(config, corpus_dir)
    previous = read_run_manifest(out_dir)
    if previous is None:
        return current

    counts = collect_counts(out_dir)
    findings = compare(previous.environment, current, counts)
    if not findings:
        return current

    if any(f.blocking for f in findings):
        report = render_report(
            findings,
            previous,
            current,
            counts,
            command=command,
            stopping=not accept_change,
        )
        if not accept_change:
            raise EnvironmentMismatch(report, findings)
        if echo is not None:
            echo(report)
        return current

    # Advisory only: one short line each, no bug-report block. Deliberate
    # user actions (a config edit, an engine upgrade) get acknowledged, not
    # escalated — see render_report's docstring.
    if echo is not None:
        echo(render_notes(findings))
    return current
