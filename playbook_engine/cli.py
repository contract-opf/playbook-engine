"""Top-level CLI entry point."""

from __future__ import annotations

import os
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import click
import yaml

from playbook_engine import __version__
from playbook_engine.canonicalize import compute_section_digests, content_hash
from playbook_engine.config import ConfigError, load_config
from playbook_engine.corpus_linter import lint_corpus
from playbook_engine.dossiers import refresh_derived_sections, validate_condition
from playbook_engine.floor_candidates import (
    FloorCandidateError,
    sign_floor_invariant,
    sign_invariant_id,
    write_floor_candidates,
)
from playbook_engine.inspection_report import build_inspection_report, write_inspection_report
from playbook_engine.pipeline import PipelineError, mine_corpus, project_playbook
from playbook_engine.playbook_assembler import (
    AssemblyError,
    write_playbook,
)
from playbook_engine.posture import (
    INTERVIEW_QUESTIONS,
    PostureError,
    apply_posture_interview,
    check_posture_floor_conflict,
)
from playbook_engine.run_manifest import (
    EnvironmentMismatch,
    RunEnvironment,
    classification_coverage,
    preflight,
    render_coverage_line,
    write_run_manifest,
)
from playbook_engine.segmentation_qa import SegmentationQAError
from playbook_engine.taxonomy import Taxonomy, TaxonomyError, load_taxonomy, merge_taxonomy
from playbook_engine.validator import SUPPORTED_OPF_VERSIONS, load_opf_file, validate_document

# e.g. "0.5" — engine version and OPF version drift independently, so
# `--version` reports both
# to keep bug reports unambiguous about which OPF schema a given engine
# build validates against (issue #176).
_OPF_VERSIONS_STR = ", ".join(sorted(SUPPORTED_OPF_VERSIONS))


def _refuse_unsupported_opf_version(doc: Any, source: Path) -> None:
    """Exit 1 when *doc* does not claim a supported opf_version (issue #238).

    The engine reads exactly one format. A command that reads a playbook
    (view bundle) must refuse a retired 0.1-0.4
    document loudly rather than render it as an empty or stale artifact —
    the same silent empty render opf_accessors exists to prevent (#154).
    The message mirrors the validator's "unsupported opf_version" error.
    """
    version = doc.get("opf_version") if isinstance(doc, dict) else None
    # isinstance first: a hand-edited list/dict value is unhashable.
    if isinstance(version, str) and version in SUPPORTED_OPF_VERSIONS:
        return
    click.secho(
        f"ERROR: {source}: unsupported opf_version {version!r} (supported: "
        f"{_OPF_VERSIONS_STR}) — the engine reads and writes only OPF 0.5; the "
        "0.1-0.4 formats were retired (spec/CHANGELOG.md)",
        fg="red",
        err=True,
    )
    raise SystemExit(1)


def _llm_segmentation_kwargs(
    cfg: Any,
    taxonomy: Taxonomy,
    out_dir: Path,
    echo: Callable[[str], None],
    *,
    stats: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Build the LLM-segmentation kwargs shared by ``mine`` and ``judge``.

    LLM-first segmentation is config-gated (``segmentation.llm``) so existing
    configs/fixtures with no ``segmentation:`` section are byte-for-byte
    unchanged — the deterministic segmenter remains the default.  Every command
    that segments a corpus MUST build these kwargs the same way: if ``mine``
    segments via the LLM but ``judge`` falls back to the deterministic
    segmenter, their clause keys diverge and the judge drain loop can never
    converge against the LLM-segmented observation store.

    Before anything else, this checks that Anthropic credentials are actually
    present (issue #131): ``segment_document``/``segment_documents_batch``
    only construct ``anthropic.Anthropic()`` lazily and that constructor
    doesn't validate the key either, so a missing ``ANTHROPIC_API_KEY``
    previously surfaced only once the first live call ran — after docling had
    already ground through extraction for the whole corpus, and as a raw
    traceback (only ``PipelineError`` was caught at the CLI boundary). Raising
    here, before any of that, turns it into an immediate, plain-language
    ``ConfigError`` that every caller already knows how to render and exit 1
    on.

    Args:
        stats: Optional mutable counter dict, updated in place with
               ``"segmentation_calls"`` (versions actually sent to the LLM —
               i.e. cache misses; a cache hit never invokes the wrapped
               closures below) and ``"segmentation_chars"`` (total character
               count of the block streams sent for those calls). Passed by
               ``judge --plan`` (issue #134) so the plan output can report a
               real segmentation-cost line instead of omitting the largest
               spend in a live run entirely. ``None`` (the default, and what
               ``mine`` always passes) disables collection.

    Returns an empty dict when ``segmentation.llm`` is off (deterministic path)
    — ``stats`` is left untouched (stays at caller-supplied zero) in that case.

    Raises:
        ConfigError: ``segmentation.llm`` is on but no Anthropic credentials
                     resolve from the environment.
    """
    kwargs: dict[str, Any] = {}
    if not cfg.segmentation.llm:
        return kwargs

    # Fail loud, ONCE, before any per-version work starts: a declared
    # extraction.extractor: docling on a docling-less host must never
    # silently downgrade every version to the legacy adapters (no OCR, no
    # heading detection) — see extraction.py's module docstring and issue
    # #80. Checked here rather than inside the per-version extract_blocks
    # loop so a whole-corpus misconfiguration produces ONE clear,
    # actionable error instead of N per-version "extraction failed"
    # warnings a human could mistake for N unrelated bad files — same
    # "raise before any per-file work starts" shape as the
    # ANTHROPIC_API_KEY check below (issue #131). extract_blocks() also
    # re-checks this per call (defense in depth for direct callers that
    # bypass this function, e.g. the standalone `segment` command, which
    # runs the same check inline — see segment_cmd).
    if cfg.extraction.extractor == "docling" and shutil.which("docling") is None:
        raise ConfigError(
            "extraction.extractor is set to 'docling' in the config, but the "
            "docling binary was not found on PATH. Install docling, run this "
            "corpus inside the project's container (see Dockerfile), or set "
            "extraction.extractor to 'legacy' or 'auto' (or omit the "
            "extraction: section) to use the legacy adapters instead."
        )

    # Agent-as-segmenter (issue #191): key-free store-backed segmentation. The
    # agent produces SegNodes via `segment`/`segment-apply`; `mine` replays them
    # from the cache. On a miss, StoreBackedSegmentFn queues the doc and raises
    # so mine quarantines it — no API key, no live call. Must precede the
    # ANTHROPIC_API_KEY check below (that gate is for the live-LLM path only).
    if cfg.segmentation.agent:
        from playbook_engine.agent_judge import PendingQueue  # noqa: PLC0415
        from playbook_engine.agent_segmenter import StoreBackedSegmentFn  # noqa: PLC0415
        from playbook_engine.extraction import ExtractionCache  # noqa: PLC0415
        from playbook_engine.llm_segmenter_batch import SegmentationVerdictCache  # noqa: PLC0415

        seg_dir = out_dir / "segment"
        kwargs["use_llm_segmentation"] = True
        pending_path = seg_dir / "pending.jsonl"
        # Fresh queue each round (mirrors the standalone `segment` command,
        # issue #182) — PendingQueue only dedups within its own instance
        # (see its docstring), so leaving a prior round's file in place
        # would make every subsequent mine/judge invocation append on top of
        # it, letting already-resolved entries linger and duplicate forever
        # instead of a fresh queue reflecting only this round's actual cache
        # misses (issue #156).
        pending_path.unlink(missing_ok=True)
        # taxonomy_ids must match the allow-list `playbook segment` writes into
        # its own queued payload (segment_cmd), or segment_apply_cmd's taxonomy
        # gate rejects every mine-queued verdict that assigns a real
        # taxonomy_id (issue #40).
        kwargs["llm_segment_fn"] = StoreBackedSegmentFn(
            pending=PendingQueue(pending_path),
            taxonomy_ids=[e.id for e in taxonomy.classifier_entries()],
        )
        kwargs["segmentation_cache"] = SegmentationVerdictCache(seg_dir / "cache.jsonl")
        kwargs["extraction_cache"] = ExtractionCache(out_dir / "extraction_cache.jsonl")
        echo("  segmentation: agent (store-backed, key-free)")
        return kwargs

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise ConfigError(
            "segmentation.llm is enabled in the config, but no Anthropic API "
            "credentials were found. Set the ANTHROPIC_API_KEY environment "
            "variable before running this command (see README.md), or run "
            "the playbook-from-corpus skill in Claude Code, which performs "
            "the judgment stages on your Claude plan without an API key. "
            "LLM segmentation currently requires an API key — see "
            "docs/PLAN-FIRST.md."
        )

    from playbook_engine.llm_segmenter import segment_document  # noqa: PLC0415

    # segment_document's real signature is
    # (canonical_text, blocks, taxonomy_ids, *, client=None, ...) — one
    # positional arg more than the SegmentFn contract mine_corpus calls
    # through (Callable[[str, list[Block]], list[SegNode]]), so it must be
    # bound to this corpus's taxonomy_ids first (same pattern as
    # llm_segmentation_stage._default_segment_fn). client stays
    # unbound/None so segment_document lazily constructs the real
    # anthropic.Anthropic() client itself. ``model`` is bound to
    # ``cfg.segmentation.model`` (issue #131) so the model actually used is
    # config data, not the function's own hardcoded default.
    #
    # This closure is also repair-aware: it declares a third ``last_error``
    # parameter (see segmentation_qa._accepts_last_error), which
    # segment_verify_repair fills with the previous attempt's
    # SegmentationQAError on every repair, and threads through to
    # segment_document's ``repair_feedback`` so a retry's prompt reflects
    # what actually failed instead of re-sending byte-identical input.
    taxonomy_ids = [e.id for e in taxonomy.classifier_entries()]
    model = cfg.segmentation.model

    def _llm_segment_fn(
        canonical_text: str,
        blocks: Any,
        last_error: SegmentationQAError | None = None,
        *,
        _taxonomy_ids: list[str] = taxonomy_ids,
        _model: str = model,
    ) -> Any:
        if stats is not None:
            # Only reached on a cache miss (or repair re-attempt) — see
            # llm_segmentation_stage.segment_to_tree, which checks
            # segmentation_cache BEFORE ever calling this closure. A repair
            # attempt calls this again for the same version, which is
            # correct here: a repair is a second real API call with a real
            # token cost, not a duplicate to be filtered out.
            stats["segmentation_calls"] = stats.get("segmentation_calls", 0) + 1
            stats["segmentation_chars"] = stats.get("segmentation_chars", 0) + sum(
                len(b.text) for b in blocks
            )
        return segment_document(
            canonical_text,
            blocks,
            _taxonomy_ids,
            repair_feedback=str(last_error) if last_error is not None else None,
            model=_model,
        )

    kwargs["use_llm_segmentation"] = True
    kwargs["llm_segment_fn"] = _llm_segment_fn
    mode_bits = ["llm"]

    if cfg.segmentation.batch:
        from playbook_engine.llm_segmenter_batch import (  # noqa: PLC0415
            segment_documents_batch,
        )

        # segment_documents_batch is called positionally-then-keyword by
        # pipeline._collect_batch_items's caller as
        # ``_batch_fn(items, taxonomy_ids=..., cache=..., progress=...)`` — bind
        # ``model`` here the same way the sync closure above does, rather than
        # handing over the bare function (which would silently keep using
        # segment_documents_batch's own DEFAULT_MODEL regardless of
        # cfg.segmentation.model).
        def _segment_documents_batch_fn(
            items: Any,
            *,
            taxonomy_ids: list[str],
            cache: Any = None,
            progress: Callable[[str], None] = lambda _: None,
            _model: str = model,
        ) -> Any:
            if stats is not None:
                # segment_documents_batch does its own cache filtering
                # internally (only ``to_submit`` items are billed) — mirror
                # that check here so the count matches what will actually be
                # sent, without needing segment_documents_batch itself to
                # accept a stats param. A cache.get() call is a cheap local
                # JSONL-store lookup, so checking it twice (here and again
                # inside segment_documents_batch) has no meaningful cost.
                for item in items:
                    hit = (
                        cache.get(item.canonical_text, model=_model) if cache is not None else None
                    )
                    if hit is None:
                        stats["segmentation_calls"] = stats.get("segmentation_calls", 0) + 1
                        stats["segmentation_chars"] = stats.get("segmentation_chars", 0) + sum(
                            len(b.text) for b in item.blocks
                        )
            return segment_documents_batch(
                items,
                taxonomy_ids=taxonomy_ids,
                model=_model,
                cache=cache,
                progress=progress,
            )

        kwargs["use_batch_segmentation"] = True
        kwargs["segment_documents_batch_fn"] = _segment_documents_batch_fn
        mode_bits.append("batch")

    if cfg.segmentation.cache:
        from playbook_engine.extraction import ExtractionCache  # noqa: PLC0415
        from playbook_engine.llm_segmenter_batch import (  # noqa: PLC0415
            SegmentationVerdictCache,
        )

        kwargs["segmentation_cache"] = SegmentationVerdictCache(
            out_dir / "segmentation_cache.jsonl"
        )
        # Extraction (docling/pdfplumber/python-docx/pandoc) is the dominant
        # cost on a real corpus with scanned PDFs — far more than the LLM
        # segmentation call above, which segmentation_cache already covers.
        # Rooted at the real out_dir (not a temp dir — see judge_cmd's
        # --plan mode below), so it stays warm across every judge/mine/
        # compile round, independent of the no_cache stage-cache flag
        # (issue #132; store-backed judges no longer force it — issue #219).
        kwargs["extraction_cache"] = ExtractionCache(out_dir / "extraction_cache.jsonl")
        mode_bits.append("cache")

    if cfg.segmentation.normalize_trail:
        from playbook_engine.llm_segmenter_batch import normalize_trail  # noqa: PLC0415

        # Same arity mismatch as segment_document above: normalize_trail
        # requires taxonomy_ids as a keyword-only arg beyond the
        # NormalizeTrailFn contract, so bind it here too (mirrors
        # llm_segmenter_batch._default_normalize_trail_fn), plus ``model`` for
        # the same config-not-code reason as the sync/batch closures above.
        def _normalize_trail_fn(
            version_trees: Any,
            taxonomy_by_version: Any,
            *,
            _taxonomy_ids: list[str] = taxonomy_ids,
            _model: str = model,
        ) -> Any:
            return normalize_trail(
                version_trees, taxonomy_by_version, taxonomy_ids=_taxonomy_ids, model=_model
            )

        kwargs["normalize_trail_across_versions"] = True
        kwargs["normalize_trail_fn"] = _normalize_trail_fn
        mode_bits.append("normalize_trail")

    echo(f"  segmentation: {'+'.join(mode_bits)}")
    return kwargs


def _echo_segmentation_cost_line(stats: dict[str, int], echo: Callable[[str], None]) -> None:
    """Print the segmentation-cost line for ``judge --plan`` (issue #134).

    ``stats`` is the counter dict threaded through ``_llm_segmentation_kwargs``
    (``stats=``) — it stays at zero for a config with ``segmentation.llm``
    off (deterministic path, no LLM spend to report) or once every version's
    canonical text already hits ``segmentation_cache``. Printed unconditionally
    (even at zero) so the plan's go/no-go gate always names segmentation
    explicitly instead of a human having to know to ask about it.
    """
    calls = stats.get("segmentation_calls", 0)
    token_estimate = stats.get("segmentation_chars", 0) // 4
    echo(f"Segmentation: {calls} version(s) not yet cached (token estimate: ~{token_estimate:,})")


def _echo_rubric_report(policy: Any, echo: Callable[[str], None]) -> None:
    """Report rubric staleness for a completed store-backed judge run.

    Makes the previously-invisible failure mode visible: a store hit whose
    banked rubric no longer matches the one in force. ``stale`` items have
    already been re-queued by the judges (unless ``--accept-stale``);
    ``legacy`` items replayed but carry no stamp at all, so nobody can say
    whether they are still valid.
    """
    from playbook_engine.rubric import STATE_LEGACY, STATE_STALE  # noqa: PLC0415

    stale = policy.total(STATE_STALE)
    legacy = policy.total(STATE_LEGACY)
    if stale:
        tail = (
            "replayed anyway (--accept-stale) — this run only"
            if policy.accept_stale
            else "re-queued for re-judgement"
        )
        click.secho(
            f"WARNING: {stale} stored verdict(s) were made under an older rubric "
            f"({policy.format_breakdown(STATE_STALE)}) — {tail}.",
            fg="yellow",
            err=True,
        )
    if legacy:
        verb = "re-queued (--strict-rubric)" if policy.strict_legacy else "replayed"
        click.secho(
            f"NOTE: {legacy} stored verdict(s) carry no rubric version "
            f"({policy.format_breakdown(STATE_LEGACY)}) — their validity under the "
            f"current rubric is unknown; {verb}.",
            fg="yellow",
            err=True,
        )


def _verdict_store_kwargs(out_dir: Path, echo: Callable[[str], None]) -> dict[str, Any]:
    """Wire store-backed judges when a verdict store exists at ``out_dir/judge/verdicts.jsonl``.

    Used by ``mine`` (issue #102) — before this, ``mine`` never checked for
    a verdict store at all, so it always ran the stub judges even over an
    ``out_dir`` where a ``playbook judge`` / ``judge-apply`` round had
    already populated real verdicts, silently overwriting the judged
    ``observations.jsonl`` with stub-mode sentinels.

    Every deviation is the deterministic standard check, so no judge is
    wired for it: only scope, classification and provenance are judged while
    mining (the equivalence label is read at ``project`` and queued by
    ``playbook judge`` after mining, issue #240).

    The L1-L4 stage cache stays ON (issue #219 — this used to force
    ``no_cache=True``, re-mining every document on every round). The
    ``VerdictStore`` is still the authoritative source for judge verdicts:
    ``mine_corpus`` never puts its own verdict cache in front of a
    store-backed judge, records with each cached document result the verdict
    keys it replayed (recomputing it when any of them changes), and never
    caches a result that queued anything — so no stale ``needs_review``
    sentinel can be replayed across rounds.

    Returns an empty dict when no verdict store exists (the stub judges
    remain the default, same as before).
    """
    verdicts_path = out_dir / "judge" / "verdicts.jsonl"
    if not verdicts_path.exists():
        return {}

    from playbook_engine.agent_judge import (  # noqa: PLC0415
        PendingQueue,
        StoreBackedClassificationJudge,
        StoreBackedProvenanceJudge,
        StoreBackedScopeJudge,
        VerdictStore,
    )
    from playbook_engine.rubric import RubricPolicy  # noqa: PLC0415

    store = VerdictStore(verdicts_path)
    pending = PendingQueue(out_dir / "judge" / "pending.jsonl")
    # One policy instance shared by every judge so the caller can report a
    # single coherent rubric tally afterwards (``_echo_rubric_report``).
    policy = RubricPolicy()
    echo(f"  judge store: {verdicts_path} (store-backed judges active)")
    return {
        "scope_judge": StoreBackedScopeJudge(store=store, pending=pending, rubric=policy),
        "classification_judge": StoreBackedClassificationJudge(
            store=store, pending=pending, rubric=policy
        ),
        "provenance_judge": StoreBackedProvenanceJudge(store=store, pending=pending, rubric=policy),
        "_rubric_policy": policy,
    }


#: version_ingest[].reason values that represent a real extraction
#: DEGRADATION (issue #81) — mirrors pipeline._FALLBACK_REASONS. Duplicated
#: (not imported) since this function only ever reads the already-written
#: JSON manifest, never pipeline internals — same "own its vocabulary as
#: plain literals" convention config.py's _VALID_EXTRACTORS docstring
#: documents for the analogous cross-module case.
_FALLBACK_REASONS = ("env-missing", "backend-error", "ocr-recovered")

#: Cap on how many fallback document/version names _echo_extractor_summary
#: prints inline before collapsing the rest into a "+N more" tail — a
#: corpus-wide docling outage could otherwise dump hundreds of lines.
_FALLBACK_NAMES_CAP = 10


def _echo_extractor_summary(out_dir: Path, echo: Callable[[str], None]) -> None:
    """Echo how many mined versions used each extractor, and name any fallbacks.

    Reads ``out_dir/corpus_manifest.json`` (already written by
    ``mine_corpus``/``compile_corpus`` by the time this runs) and tallies
    each version's ``version_ingest[].extractor`` value — every value seen,
    not just ``"docling"``/``"legacy"``: the deterministic path records the
    raw file suffix (``"docx"``, ``"pdf"``, ``"rtf"``) there, so a
    docling-less deterministic-only run now prints a line too instead of
    being silently blind (issue #81; previously this function filtered to
    just ``"docling"``/``"legacy"`` and returned early with nothing to show).
    Mirrors the ``segmentation: ...`` echo above so the docling-vs-legacy
    choice is a first-class part of ``mine`` output rather than
    only a ``logging.info`` line suppressed by default Python logging config
    (issue #129) — a host run without docling silently extracting scanned
    PDFs with no OCR was otherwise invisible to the operator.

    When any version's ``version_ingest[].reason`` is a real degradation
    (``"env-missing"``/``"backend-error"``/``"ocr-recovered"`` — never
    ``"declared"``, a
    deliberate config choice, not a degradation), a second block breaks the
    fallback count down by reason and names the affected document/version
    pairs, up to :data:`_FALLBACK_NAMES_CAP` with a "+N more" tail — the
    same information ``config.extraction.max_fallback`` enforces against,
    surfaced even when the run stayed under budget (or is unbounded).

    Silent no-op when the manifest is missing or empty.
    """
    import json  # noqa: PLC0415

    manifest_path = out_dir / "corpus_manifest.json"
    if not manifest_path.exists():
        return

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    counts: dict[str, int] = {}
    reason_counts: dict[str, int] = {}
    fallback_names: list[str] = []
    for doc in manifest:
        doc_id = doc.get("document_id", "?")
        for v in doc.get("version_ingest", []) or []:
            if not isinstance(v, dict):
                continue
            ext = v.get("extractor")
            if ext:
                counts[ext] = counts.get(ext, 0) + 1
            reason = v.get("reason")
            if reason in _FALLBACK_REASONS:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
                fallback_names.append(f"{doc_id}/{v.get('version', '?')}")

    if not counts:
        return

    summary = ", ".join(f"{ext}={n}" for ext, n in sorted(counts.items()))
    echo(f"  extraction: {summary}")

    if not fallback_names:
        return

    reason_summary = ", ".join(f"{r}={n}" for r, n in sorted(reason_counts.items()))
    echo(f"  extraction fallback: {len(fallback_names)} version(s) ({reason_summary})")
    shown = fallback_names[:_FALLBACK_NAMES_CAP]
    more = len(fallback_names) - len(shown)
    tail = f", +{more} more" if more > 0 else ""
    echo(f"    {', '.join(shown)}{tail}")


# ---------------------------------------------------------------------------
# Run provenance manifest (issue #121)
# ---------------------------------------------------------------------------

#: Shared ``--accept-environment-change`` flag for ``mine``/``judge``/``segment``.
#: Defined once as a reusable decorator so the three commands can never drift
#: apart on the flag name — the preflight report NAMES this flag as the way
#: forward, so a mismatch between the text and one command's actual option
#: would be worse than no message at all.
_accept_environment_change_option = click.option(
    "--accept-environment-change",
    "accept_environment_change",
    is_flag=True,
    default=False,
    help=(
        "Proceed even though this output folder was built in a different "
        "environment (different document reader, engine version, or clause "
        "splitter). The differences are still explained first; this only "
        "says you accept the rework they cause."
    ),
)


def _preflight_environment(
    out_dir: Path,
    cfg: Any,
    corpus_dir: Path,
    *,
    command: str,
    accept_change: bool,
) -> RunEnvironment:
    """Run the provenance preflight and return the captured environment.

    Silent on the normal path (see :func:`run_manifest.preflight`). On a
    blocking difference, prints the already-composed plain-English report to
    stderr and exits 1 — no extra ``ERROR:`` prefix, no exception text: the
    report IS the message, and wrapping it in CLI chrome would bury the one
    sentence a non-engineer needs to read.

    Returned so the caller can pass the SAME snapshot to
    :func:`_record_run_manifest` when the run succeeds — capturing twice
    would let a mid-run PATH change silently produce a manifest that doesn't
    describe the run that just happened.
    """
    try:
        return preflight(
            out_dir,
            cfg,
            corpus_dir,
            command=command,
            echo=click.echo,
            accept_change=accept_change,
        )
    except EnvironmentMismatch as exc:
        click.echo(exc.report, err=True)
        raise SystemExit(1) from exc


def _record_run_manifest(
    out_dir: Path,
    environment: RunEnvironment,
    command: str,
    classification_coverage: dict[str, int] | None = None,
) -> None:
    """Stamp *out_dir* with the environment that just produced it.

    Best-effort by design: a read-only or full disk must not turn an
    otherwise-successful mine/judge/segment into a failure over a bookkeeping
    file.
    The cost of a missing manifest is one silent run next time — the cost of
    failing here is throwing away a completed corpus run.
    """
    try:
        write_run_manifest(
            out_dir, environment, command=command, classification_coverage=classification_coverage
        )
    except OSError as exc:  # pragma: no cover - disk-full / read-only out-dir
        click.secho(f"note: could not write run_manifest.json ({exc})", fg="yellow", err=True)


@click.group()
@click.version_option(
    __version__,
    prog_name="playbook-engine",
    message=f"%(prog)s %(version)s (OPF {_OPF_VERSIONS_STR})",
)
def cli() -> None:
    """playbook-engine: compile a corpus of agreements into an OPF playbook."""


@cli.command()
@click.argument("file", type=click.Path(exists=True, path_type=Path))
def validate(file: Path) -> None:
    """Validate an OPF document against the schema and normative rules."""
    try:
        doc = load_opf_file(file)
    except Exception as exc:  # noqa: BLE001
        click.secho(f"ERROR: could not parse {file}: {exc}", fg="red", err=True)
        raise SystemExit(1) from exc

    # Issue #240: next to a derivation's playbook sits its verdict store; with
    # it, every vs_standard label must trace to a stored verdict.
    from playbook_engine.agent_judge import VerdictStore  # noqa: PLC0415

    verdicts_path = file.resolve().parent / "judge" / "verdicts.jsonl"
    result = validate_document(
        doc, verdict_store=VerdictStore(verdicts_path) if verdicts_path.is_file() else None
    )

    for err in result.errors:
        color = "red" if err.blocking else "yellow"
        click.secho(str(err), fg=color, err=err.blocking)

    if result.ok:
        click.secho(f"OK  {file}", fg="green")
    else:
        n_blocking = sum(1 for e in result.errors if e.blocking)
        click.secho(f"FAIL {file}: {n_blocking} error(s)", fg="red", err=True)
        raise SystemExit(1)


@cli.command(name="resolve-citation")
@click.argument("playbook_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--clause",
    "clause_id",
    default=None,
    help="evidence.clauses[].id, e.g. clause.indemnification (with --obs).",
)
@click.option(
    "--obs",
    "obs_index",
    type=int,
    default=None,
    help=("Index into that clause's evidence.precedent records."),
)
@click.option(
    "--precedent-id",
    "precedent_id",
    default=None,
    help=("An evidence.precedent[].id (prec.<sha>) to resolve, instead of --clause/--obs."),
)
@click.option(
    "--corpus-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Directory holding the corpus source files.",
)
def resolve_citation_cmd(
    playbook_file: Path,
    clause_id: str | None,
    obs_index: int | None,
    precedent_id: str | None,
    corpus_dir: Path,
) -> None:
    """Resolve one observation's citation to a hash-verified source file (OPF §4).

    Address the citation either by --clause and --obs, or (OPF 0.5) by
    --precedent-id. Looks up the cited (document_id, version) in
    corpus.documents[].version_files, finds the file under CORPUS-DIR whose
    sha256 matches, and prints the path plus clause_path/char_span. Exits 1 on
    hash mismatch or a missing content address — the reference implementation
    consumers copy.
    """
    from playbook_engine.citation_resolver import (
        CitationResolutionError,
        resolve_citation,
        resolve_precedent_citation,
    )

    by_index = clause_id is not None or obs_index is not None
    if precedent_id is not None and by_index:
        raise click.UsageError("pass either --precedent-id or --clause/--obs, not both")
    if precedent_id is None and (clause_id is None or obs_index is None):
        raise click.UsageError("pass --clause and --obs together, or --precedent-id")

    try:
        doc = load_opf_file(playbook_file)
    except Exception as exc:  # noqa: BLE001
        click.secho(f"ERROR: could not parse {playbook_file}: {exc}", fg="red", err=True)
        raise SystemExit(1) from exc

    try:
        if precedent_id is not None:
            resolved = resolve_precedent_citation(doc, precedent_id, corpus_dir)
        else:
            assert clause_id is not None and obs_index is not None
            resolved = resolve_citation(doc, clause_id, obs_index, corpus_dir)
    except CitationResolutionError as exc:
        click.secho(f"ERROR: {exc}", fg="red", err=True)
        raise SystemExit(1) from exc

    click.secho(resolved.describe(), fg="green")
    click.echo(f"file: {resolved.file_path.resolve()}")
    if resolved.clause_path:
        click.echo(f"clause_path: {resolved.clause_path}")
    if resolved.char_span:
        click.echo(f"char_span: [{resolved.char_span[0]}, {resolved.char_span[1]}]")


def _precedent_table(rows: list[dict[str, Any]], *, refused: bool) -> str:
    """Render precedent records (or refused asks) as a plain-text table."""

    def snippet(text: Any) -> str:
        flat = " ".join(str(text).split()) if isinstance(text, str) else "-"
        return flat if len(flat) <= 60 else flat[:57] + "..."

    header: tuple[str, ...]
    body: list[tuple[str, ...]]
    if refused:
        header = ("precedent_id", "document_id", "round", "text")
        body = [
            (
                str(r.get("precedent_id")),
                str(r.get("document_id")),
                str(r.get("round")),
                snippet(r.get("text")),
            )
            for r in rows
        ]
    else:
        header = (
            "id",
            "document_id",
            "signed",
            "signed_at",
            "standard",
            "moved",
            "refused",
            "text",
        )

        def signed_text(r: dict[str, Any]) -> Any:
            st = r.get("signed_text")
            return st.get("text") if isinstance(st, dict) else None

        body = [
            (
                str(r.get("id")),
                str(r.get("document_id")),
                "yes" if r.get("signed") else "no",
                str(r.get("signed_at") or "-"),
                "yes" if r.get("standard") else "no",
                "yes" if r.get("moved") else "no",
                str(len(r.get("refused_asks") or [])),
                snippet(signed_text(r)) if signed_text(r) is not None else "(struck)",
            )
            for r in rows
        ]
    widths = [max(len(h), *(len(row[i]) for row in body)) for i, h in enumerate(header)]
    lines = ["  ".join(h.ljust(w) for h, w in zip(header, widths, strict=True)).rstrip()]
    lines += [
        "  ".join(c.ljust(w) for c, w in zip(row, widths, strict=True)).rstrip() for row in body
    ]
    return "\n".join(lines)


@cli.command(name="precedent")
@click.argument("playbook_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--clause",
    "clause",
    default=None,
    help="Clause taxonomy_id (e.g. governing_law) or evidence.clauses[].id.",
)
@click.option(
    "--id",
    "precedent_id",
    default=None,
    help="One precedent record by its stable id (prec.<sha>).",
)
@click.option(
    "--refused",
    is_flag=True,
    default=False,
    help="Return the refused asks instead of the precedent records.",
)
@click.option(
    "--limit",
    type=click.IntRange(min=0),
    default=None,
    help="Keep at most N results.",
)
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["jsonl", "table"]),
    default="table",
    show_default=True,
    help="jsonl: one canonical JSON object per line; table: a plain-text table.",
)
def precedent_cmd(
    playbook_file: Path,
    clause: str | None,
    precedent_id: str | None,
    refused: bool,
    limit: int | None,
    fmt: str,
) -> None:
    """Query an OPF 0.5 playbook's precedent records.

    --clause returns one clause's records, ranked by how many distinct deals
    signed the same text, then by most recent signing, with unsigned deals'
    records (their text is the last draft) after every signed one; with
    --refused, the asks refused before signing instead. --id returns one record (with
    --refused, its refused asks). With neither, every record sorted by id —
    as jsonl, byte-identical to the precedent.jsonl sidecar `playbook project`
    writes. Records are printed exactly as the playbook carries them.
    """
    from playbook_engine.canonicalize import canonicalize
    from playbook_engine.opf_accessors import (
        find_precedent,
        is_precedent_shape,
        playbook_clauses,
        playbook_precedent,
        precedent_by_id,
    )

    if clause is not None and precedent_id is not None:
        raise click.UsageError("pass either --clause or --id, not both")

    try:
        doc = load_opf_file(playbook_file)
    except Exception as exc:  # noqa: BLE001
        click.secho(f"ERROR: could not parse {playbook_file}: {exc}", fg="red", err=True)
        raise SystemExit(1) from exc

    # One format (issues #238, #233): a retired OPF 0.4 playbook has the same
    # evidence.precedent shape but not the opening evidence, so it is refused
    # by version rather than read as a 0.5 document with `opened_with` missing.
    _refuse_unsupported_opf_version(doc, playbook_file)
    if not is_precedent_shape(doc):
        click.secho(
            f"ERROR: {playbook_file.name} carries no evidence.precedent (OPF "
            f"{doc.get('opf_version')!s}); precedent queries need an OPF 0.5 playbook",
            fg="red",
            err=True,
        )
        raise SystemExit(1)

    rows: list[dict[str, Any]]
    if precedent_id is not None:
        record = precedent_by_id(doc, precedent_id)
        if record is None:
            click.secho(f"ERROR: no precedent record with id {precedent_id!r}", fg="red", err=True)
            raise SystemExit(1)
        if refused:
            rows = [
                {
                    "precedent_id": record.get("id"),
                    "document_id": record.get("document_id"),
                    "taxonomy_id": record.get("taxonomy_id"),
                    **ask,
                }
                for ask in record.get("refused_asks") or []
                if isinstance(ask, dict)
            ]
        else:
            rows = [record]
    elif clause is not None:
        known = {c.get("taxonomy_id") for c in playbook_clauses(doc)} | {
            c.get("id") for c in playbook_clauses(doc)
        }
        if clause not in known:
            names = ", ".join(sorted(str(c.get("taxonomy_id")) for c in playbook_clauses(doc)))
            click.secho(f"ERROR: no clause {clause!r} (known: {names})", fg="red", err=True)
            raise SystemExit(1)
        rows = find_precedent(doc, clause, refused=refused)
    else:
        if refused:
            raise click.UsageError("--refused needs --clause or --id")
        rows = sorted(playbook_precedent(doc), key=lambda p: str(p.get("id")))
    if limit is not None:
        rows = rows[:limit]

    if fmt == "jsonl":
        for row in rows:
            click.echo(canonicalize(row))
    elif rows:
        click.echo(_precedent_table(rows, refused=refused))
    else:
        click.echo("(no results)", err=True)


@cli.group(name="taxonomy")
def taxonomy_group() -> None:
    """Manage clause taxonomies."""


@taxonomy_group.command(name="merge")
@click.argument("taxonomy_file", type=click.Path(exists=True, path_type=Path))
@click.argument("upstream_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--out",
    "out_path",
    type=click.Path(path_type=Path),
    default=None,
    help="Write merged taxonomy here (default: overwrite taxonomy_file).",
)
@click.option(
    "--dry-run", is_flag=True, default=False, help="Print the merged taxonomy; do not write."
)
@click.option(
    "--new-source",
    "new_source",
    type=str,
    default=None,
    help=(
        "Update the taxonomy's source field to this value when new entries are added "
        "(e.g. 'CUAD-v2'). Has no effect if no new entries are found."
    ),
)
def taxonomy_merge(
    taxonomy_file: Path,
    upstream_file: Path,
    out_path: Path | None,
    dry_run: bool,
    new_source: str | None,
) -> None:
    """Merge a newer upstream taxonomy into an existing curated taxonomy.

    TAXONOMY_FILE is the curated taxonomy to update.
    UPSTREAM_FILE is the newer upstream release (entries section only, or a full taxonomy YAML).
    Known ids keep their existing status; new ids enter as inactive.
    """
    try:
        existing = load_taxonomy(taxonomy_file)
    except TaxonomyError as exc:
        click.secho(f"ERROR loading taxonomy: {exc}", fg="red", err=True)
        raise SystemExit(1) from exc

    try:
        upstream_raw = yaml.safe_load(upstream_file.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        click.secho(f"ERROR loading upstream file: {exc}", fg="red", err=True)
        raise SystemExit(1) from exc

    if isinstance(upstream_raw, dict):
        upstream_entries = upstream_raw.get("entries", [])
    elif isinstance(upstream_raw, list):
        upstream_entries = upstream_raw
    else:
        click.secho("ERROR: upstream file must be a YAML mapping or list", fg="red", err=True)
        raise SystemExit(1)

    merged = merge_taxonomy(existing, upstream_entries, new_source=new_source)

    n_added = len(merged.entries) - len(existing.entries)
    click.echo(
        f"Merged: {len(existing.entries)} existing + {n_added} new entries "
        f"→ {len(merged.entries)} total"
    )
    if new_source and n_added == 0:
        click.secho(
            "Note: --new-source ignored (no new entries were added; source unchanged).",
            fg="yellow",
            err=True,
        )

    if dry_run:
        click.echo("--- dry run: showing merged entries ---")
        for entry in merged.entries:
            marker = " [NEW]" if entry.id not in {e.id for e in existing.entries} else ""
            click.echo(f"  {entry.status:8s} {entry.id}{marker}")
        return

    dest = out_path or taxonomy_file
    _write_taxonomy(merged, dest)
    click.secho(f"Written: {dest}", fg="green")


def _write_taxonomy(taxonomy: Taxonomy, dest: Path) -> None:
    """Write taxonomy back to YAML, preserving comments from original where possible."""
    # Round-trip via structured data (comments are lost but structure is correct).
    data = {
        "source": taxonomy.source,
        "entries": [
            {
                "id": e.id,
                "label": e.label,
                "status": e.status,
                "cuad_origin": e.cuad_origin,
                "description": e.description,
                "structural": e.structural,
            }
            for e in taxonomy.entries
        ],
    }
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    tmp.write_text(yaml.dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    tmp.replace(dest)


# ---------------------------------------------------------------------------
# Corpus preflight (shared by `mine` and `segment`)
# ---------------------------------------------------------------------------

_SKIP_PREFLIGHT_HELP = (
    "Skip the corpus/config preflight (the same checks `playbook lint-corpus` "
    "runs) and go straight to work. Only useful when you have read the errors "
    "and decided they do not apply."
)


def _run_corpus_preflight(
    corpus_dir: Path,
    config_path: Path,
    *,
    skip: bool,
    command: str,
) -> None:
    """Run `lint-corpus`'s checks as a PRECONDITION, not a suggestion.

    ``lint-corpus`` has always been the documented preflight, but nothing made
    it mandatory — an ad-hoc ``playbook mine ...`` skipped it entirely, which is
    how a corpus of dangling symlinks (issue #130) and a host that had quietly
    lost its ``docling`` binary (issue #121) both got all the way to a finished,
    wrong-looking-like-right derivation. Running it here means the expensive,
    hard-to-audit stages cannot start in an environment the linter can already
    tell is broken.

    Only errors and warnings are echoed — the per-document "OK" lines belong to
    ``lint-corpus`` itself, and would bury this command's own output.

    Args:
        corpus_dir:  Corpus root about to be processed.
        config_path: Engine config, validated alongside it.
        skip:        Honour ``--skip-preflight`` and do nothing.
        command:     Name of the calling command, for the message.

    Raises:
        SystemExit: preflight found blocking errors.
    """
    if skip:
        click.secho(
            f"preflight: skipped (--skip-preflight). {command} will run against whatever is there.",
            fg="yellow",
            err=True,
        )
        return

    report = lint_corpus(corpus_dir, config_path=config_path)

    for item in report.warnings():
        click.secho(f"  WARN {item.message}", fg="yellow", err=True)

    if not report.has_errors:
        n_warn = len(report.warnings())
        suffix = f" ({n_warn} warning(s))" if n_warn else ""
        click.echo(f"preflight: OK{suffix}")
        return

    for item in report.errors():
        click.secho(f"  ERR  {item.message}", fg="red", err=True)
    click.secho(
        f"\nPreflight found {len(report.errors())} problem(s) with this corpus or "
        f"config, so `{command}` stopped before doing any work — running anyway "
        "would produce a playbook that looks fine and is not. Fix the errors "
        "above and re-run, or pass --skip-preflight if you are certain they do "
        "not apply.",
        fg="red",
        err=True,
    )
    raise SystemExit(1)


@cli.command(name="mine")
@click.argument("corpus_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option(
    "--config", "config_path", type=click.Path(exists=True, path_type=Path), required=True
)
@click.option(
    "--out",
    "out_path",
    type=click.Path(path_type=Path),
    default=None,
    help="Output directory for the observation store (default: <corpus_dir>/../out).",
)
@click.option(
    "--no-cache",
    "no_cache",
    is_flag=True,
    default=False,
    help=(
        "Disable the stage cache and force a full recompute — including "
        "re-extraction (docling/pdfplumber/python-docx/pandoc), even if "
        "extraction_cache.jsonl is warm."
    ),
)
@click.option(
    "--force-rewrite",
    "force_rewrite",
    is_flag=True,
    default=False,
    help=(
        "Rewrite every intermediate (observations.jsonl, trail/, normalized/, …) "
        "even when its content is unchanged. By default a file whose content "
        "would not change is left untouched."
    ),
)
@click.option(
    "--skip-preflight",
    "skip_preflight",
    is_flag=True,
    default=False,
    help=_SKIP_PREFLIGHT_HELP,
)
@click.option(
    "--entity-registry",
    "entity_registry_path",
    type=click.Path(path_type=Path),
    default=None,
    help=(
        "Path to the born-safe entity registry (alias->real-name map). Defaults "
        "to ~/.cache/playbook-engine/entity_registry.json — a machine-global, "
        "persistent file. Point it into your gitignored output dir (e.g. "
        "<out>/entity_registry.json) to keep all sensitive real-name data in one "
        "place. Only relevant when provenance.known_entities is set."
    ),
)
@_accept_environment_change_option
def mine_cmd(
    corpus_dir: Path,
    config_path: Path,
    out_path: Path | None,
    no_cache: bool,
    force_rewrite: bool,
    skip_preflight: bool,
    entity_registry_path: Path | None,
    accept_environment_change: bool,
) -> None:
    """Mine CORPUS_DIR and write the observation store (L1–L4).

    Runs ingest, scope gate, classification, alignment, and the
    standard check for every agreement in CORPUS_DIR and writes:

    \b
      observations.jsonl    — per-clause observation store
      corpus_manifest.json  — per-document metadata
      scope.json            — scope-gate decisions
      trail/<doc_id>.json   — version-order and provenance signals
      normalized/           — segmented clause trees
      run_manifest.json     — the environment this run used (checked by the next run)

    Pass --no-cache to disable the content-addressed stage cache and force a
    full recompute even if intermediates already exist — this also forces
    re-extraction (docling/pdfplumber/python-docx/pandoc) even if
    extraction_cache.jsonl already has a warm entry for a version's current
    content, so a suspect extraction can be recomputed rather than silently
    replayed. Reads are bypassed; extraction_cache.jsonl is still
    refreshed with the new result, so a subsequent run without --no-cache
    stays warm.

    The stage cache is layered: L1 (ingest + segment) per
    version file, L2-L4 per document keyed by its L1 output — a template,
    taxonomy or threshold change replays every L1 tree — and it stays on when
    a verdict store is present, recomputing exactly the documents whose
    stored verdicts changed or are still pending. An intermediate whose
    content would not change is not rewritten; ``--force-rewrite`` rewrites
    them all.

    Checks the stored run manifest first (so a changed environment is named as
    the root cause), then runs the ``lint-corpus`` checks and refuses to start
    if any of them fail — a broken corpus layout or extraction environment produces a
    plausible-looking but wrong observation store, which is much more expensive
    to notice later than up front. ``--skip-preflight`` opts out.

    Does NOT write playbook.opf.json.  Run ``playbook project`` afterwards
    to compile the playbook from the store.
    """
    try:
        cfg = load_config(config_path)
    except ConfigError as exc:
        click.secho(f"Config error: {exc}", fg="red", err=True)
        raise SystemExit(1) from exc

    try:
        taxonomy = load_taxonomy(cfg.taxonomy_path)
    except TaxonomyError as exc:
        click.secho(f"Taxonomy error: {exc}", fg="red", err=True)
        raise SystemExit(1) from exc

    out_dir = (out_path or corpus_dir.parent / "out").resolve()

    click.echo(f"corpus : {corpus_dir}")
    click.echo(f"config : {config_path}")
    click.echo(f"out    : {out_dir}")

    # Before ANY work starts (issue #121): does this out-dir's stored
    # run_manifest.json describe the environment we are about to run in?
    # Silent when it does (or on a fresh out-dir) — the whole point is that
    # the normal path never says anything. When it doesn't, this prints one
    # plain-English explanation and exits, rather than letting a lost docling
    # install quietly re-extract the whole corpus under the legacy adapters
    # and surface hours later as AgentSegmentationPending quarantine.
    #
    # Ordered BEFORE _run_corpus_preflight deliberately: when the environment
    # changed, that IS the root cause, and this explanation subsumes the
    # downstream symptoms the corpus checks would otherwise report first (a
    # lost docling makes the corpus checks complain that .rtf sources need
    # docling — true, but two steps removed from what actually changed).
    environment = _preflight_environment(
        out_dir,
        cfg,
        corpus_dir,
        command="mine",
        accept_change=accept_environment_change,
    )

    _run_corpus_preflight(corpus_dir, config_path, skip=skip_preflight, command="playbook mine")

    # Segment the same way ``judge`` does.
    try:
        seg_kwargs = _llm_segmentation_kwargs(cfg, taxonomy, out_dir, click.echo)
    except ConfigError as exc:
        click.secho(f"Config error: {exc}", fg="red", err=True)
        raise SystemExit(1) from exc

    # If a verdict store exists (populated by ``playbook judge-apply``), wire in
    # the store-backed judges so the mining step replays stored verdicts rather
    # than generating new needs_review sentinels (issue #102). The stage cache
    # stays on under them (issue #219) — see ``_verdict_store_kwargs``.
    #
    # ``refresh_extraction`` is sourced directly from the raw ``no_cache``
    # flag (the operator's literal --no-cache) (issue #78): only an operator
    # who declares the extraction suspect re-extracts/re-OCRs the corpus.
    verdict_kwargs = _verdict_store_kwargs(out_dir, click.echo)
    # Not a mine_corpus parameter — the shared RubricPolicy the wired judges
    # tally into, read back for reporting after the run.
    rubric_policy = verdict_kwargs.pop("_rubric_policy", None)
    mine_kwargs: dict[str, Any] = {
        "no_cache": no_cache,
        "refresh_extraction": no_cache,
        "force_rewrite": force_rewrite,
        **seg_kwargs,
        **verdict_kwargs,
    }

    try:
        mine_corpus(
            corpus_dir=corpus_dir.resolve(),
            config=cfg,
            taxonomy=taxonomy,
            out_dir=out_dir,
            progress=click.echo,
            entity_registry_path=(entity_registry_path.resolve() if entity_registry_path else None),
            **mine_kwargs,
        )
    except PipelineError as exc:
        click.secho(f"ERROR: {exc}", fg="red", err=True)
        raise SystemExit(1) from exc

    _echo_extractor_summary(out_dir, click.echo)
    # Classification coverage by basis (issue #235): counts only, also kept in
    # run_manifest.json. Shows how much of the corpus reached a taxonomy_id by
    # heading, judge, inheritance or content similarity, and how much did not.
    coverage = classification_coverage(out_dir)
    coverage_line = render_coverage_line(coverage)
    if coverage_line:
        click.echo(coverage_line)
    if rubric_policy is not None:
        _echo_rubric_report(rubric_policy, click.echo)
    # Stamp the out-dir with what just built it, so the NEXT run has
    # something to check itself against (issue #121). Written only after
    # mine_corpus returned successfully — a manifest is a claim about the
    # artifacts sitting next to it.
    _record_run_manifest(out_dir, environment, "mine", classification_coverage=coverage)
    click.secho(f"OK  {out_dir / 'observations.jsonl'}", fg="green")


def _echo_equivalence_report(playbook: dict[str, Any], echo: Callable[[str], None]) -> None:
    """Report the ``vs_standard`` coverage of a projected playbook (issue #240).

    Counts only. An unjudged text stays ``null`` in the playbook, so the number
    left to judge is reported rather than hidden. Silent when nothing is
    eligible (emergent mode, or every non-standard text already exact).
    """
    from playbook_engine.equivalence import summarize  # noqa: PLC0415
    from playbook_engine.opf_accessors import perspective_party  # noqa: PLC0415

    agreement_type = playbook.get("agreement_type")
    agreement_type_id = agreement_type.get("id") if isinstance(agreement_type, dict) else None
    evidence = playbook.get("evidence")
    if not isinstance(agreement_type_id, str) or not isinstance(evidence, dict):
        return
    totals = summarize(evidence, agreement_type_id, perspective_party(playbook))["totals"]
    if not totals["eligible"]:
        return
    echo(
        f"equivalence: {totals['eligible']} distinct text(s), {totals['drafted']} labelled "
        f"({totals['unchecked']} unchecked, {totals['disputed']} disputed), "
        f"{totals['unjudged']} unjudged"
    )
    if totals["unjudged"]:
        echo("  run `playbook judge` to queue the unjudged texts, then `judge-apply` and project")


@cli.command(name="project")
@click.argument("out_dir", type=click.Path(file_okay=False, path_type=Path))
@click.option(
    "--config", "config_path", type=click.Path(exists=True, path_type=Path), required=True
)
def project_cmd(out_dir: Path, config_path: Path) -> None:
    """Project the observation store in OUT_DIR into a playbook (L5 only).

    Reads ``observations.jsonl`` and ``corpus_manifest.json`` from OUT_DIR
    (written by ``playbook mine``) and compiles them into a schema-valid
    OPF 0.5 ``playbook.opf.json`` (the one format the engine emits) plus its
    ``precedent.jsonl`` sidecar, using purely deterministic logic — zero
    ingest work, zero LLM calls.

    Re-running ``project`` after changing the projection logic changes the
    playbook without re-mining the corpus.

    OUT_DIR must already contain the observation store produced by
    ``playbook mine``.
    """
    try:
        cfg = load_config(config_path)
    except ConfigError as exc:
        click.secho(f"Config error: {exc}", fg="red", err=True)
        raise SystemExit(1) from exc

    try:
        taxonomy = load_taxonomy(cfg.taxonomy_path)
    except TaxonomyError as exc:
        click.secho(f"Taxonomy error: {exc}", fg="red", err=True)
        raise SystemExit(1) from exc

    out_dir_resolved = out_dir.resolve()

    click.echo(f"store  : {out_dir_resolved}")
    click.echo(f"config : {config_path}")

    try:
        playbook = project_playbook(
            out_dir=out_dir_resolved,
            config=cfg,
            taxonomy=taxonomy,
            progress=click.echo,
        )
    except (PipelineError, AssemblyError) as exc:
        click.secho(f"ERROR: {exc}", fg="red", err=True)
        raise SystemExit(1) from exc

    _echo_equivalence_report(playbook, click.echo)
    click.secho(f"OK  {out_dir_resolved / 'playbook.opf.json'}", fg="green")


@cli.command(name="doctor")
@click.option(
    "--strict",
    "strict",
    is_flag=True,
    default=False,
    help=(
        "Exit non-zero if anything the engine can shell out to is missing, not "
        "just the things that are definitely wrong. Use this in setup scripts "
        "and CI, where a partial environment should stop the line."
    ),
)
def doctor_cmd(strict: bool) -> None:
    """Report what this machine provides, in plain English.

    Corpus-free and config-free, so it can be run before there is anything else
    to run. Names the engine version, the container image stamp if there is one,
    and every external tool the pipeline can reach — with what its absence would
    silently cost you.

    Exit code is 0 unless something is definitely wrong: running inside the
    project's own Docker image with one of the tools that image installs
    missing, or an image whose stamp disagrees with the engine actually
    imported. ``--strict`` additionally fails on any missing tool.
    """
    from playbook_engine.environment import probe_environment  # noqa: PLC0415

    env = probe_environment()

    click.echo(f"engine   : {env.engine_version}")
    click.echo(f"python   : {env.python_version}")
    click.echo(f"platform : {env.platform_name}")
    if env.image is not None:
        click.echo(f"runtime  : project Docker image (built from commit {env.image.git_sha})")
    else:
        click.echo("runtime  : host install (not the project Docker image)")
    click.echo(
        "api key  : "
        + ("ANTHROPIC_API_KEY is set" if env.anthropic_key_set else "ANTHROPIC_API_KEY not set")
    )
    click.echo("")

    for status in env.tools:
        if status.present:
            click.secho(f"  OK   {status.tool.name}  ({status.path})", fg="green")
        else:
            click.secho(f"  --   {status.tool.name}  not installed", fg="yellow")
            click.echo(f"         used for: {status.tool.purpose}")
            click.echo(f"         without it: {status.tool.consequence}")
            click.echo(f"         to install: {status.tool.install}")

    problems: list[str] = []

    # An engine/stamp disagreement means the code running is not the code the
    # image was built and verified around — most often an editable checkout
    # bind-mounted over the installed package. Results from such a container
    # cannot be attributed to a version at all.
    if env.image_matches_engine() is False and env.image is not None:
        problems.append(
            f"This container is stamped as engine {env.image.engine_version}, but the "
            f"engine that actually loaded is {env.engine_version}. Something is "
            "overriding the installed package (a bind-mounted checkout, usually), so "
            "results from this container cannot be tied to a released version."
        )

    # Inside the project image these tools are installed by the Dockerfile. If
    # one is gone, the image is broken or has been modified — not a user
    # forgetting an optional dependency.
    if env.in_project_image:
        for status in env.missing(expected_in_image_only=True):
            problems.append(
                f"{status.tool.name} is missing from inside the project image, which "
                "installs it. This image is broken or has been modified — rebuild it "
                "with `make docker-build`."
            )

    if strict:
        for status in env.missing():
            problems.append(f"{status.tool.name} is not installed ({status.tool.install}).")

    click.echo("")
    if problems:
        for problem in problems:
            click.secho(f"  ERR  {problem}", fg="red", err=True)
        raise SystemExit(1)

    n_missing = len(env.missing())
    if n_missing:
        click.secho(
            f"OK — usable, with {n_missing} optional tool(s) missing (see above for "
            "what each one costs you).",
            fg="green",
        )
    else:
        click.secho("OK — everything the engine can use is installed.", fg="green")


@cli.command(name="lint-corpus")
@click.argument("corpus_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path),
    default=None,
    help="Engine config YAML to validate alongside the corpus.",
)
def lint_corpus_cmd(corpus_dir: Path, config_path: Path | None) -> None:
    """Check CORPUS_DIR layout before a compile run.

    Reports errors (blocking) and warnings (advisory) so you can fix the
    layout before running ``playbook mine``.  Exits 0 when no errors are
    found, non-zero otherwise.
    """
    report = lint_corpus(corpus_dir, config_path=config_path)

    for item in report.items:
        if item.level == "ok":
            click.secho(f"  OK   {item.message}", fg="green")
        elif item.level == "warning":
            click.secho(f"  WARN {item.message}", fg="yellow", err=True)
        else:
            click.secho(f"  ERR  {item.message}", fg="red", err=True)

    if report.has_errors:
        n_err = len(report.errors())
        n_warn = len(report.warnings())
        click.secho(
            f"\n{n_err} error(s), {n_warn} warning(s) — fix errors before running compile.",
            fg="red",
            err=True,
        )
        raise SystemExit(1)
    n_warn = len(report.warnings())
    msg = "no errors"
    if n_warn:
        msg += f", {n_warn} warning(s)"
    click.secho(f"\nOK — {msg}", fg="green")


@cli.command(name="inspect")
@click.argument("out_dir", type=click.Path(file_okay=False, path_type=Path))
@click.option(
    "--out",
    "report_path",
    type=click.Path(path_type=Path),
    default=None,
    help="Write the report to this file (default: print to stdout).",
)
def inspect_cmd(out_dir: Path, report_path: Path | None) -> None:
    """Render trail/ and observations.jsonl as a human-readable Markdown report.

    OUT_DIR is the output directory produced by ``playbook mine``.

    Lets a lawyer verify the engine's structural inferences — version ordering,
    signed-copy identification, provenance, and per-clause outcomes — before
    trusting the compiled playbook.  If an inference is wrong, add a
    ``hints.yaml`` to the document folder and re-run ``playbook mine`` (no
    flag needed — the hints file is hashed into that document's cache key).
    """
    try:
        if report_path:
            write_inspection_report(out_dir.resolve(), report_path.resolve())
            click.secho(f"OK  {report_path}", fg="green")
        else:
            click.echo(build_inspection_report(out_dir.resolve()))
    except FileNotFoundError as exc:
        click.secho(f"ERROR: {exc}", fg="red", err=True)
        raise SystemExit(1) from exc


# Printed when `stage --symlink` is used. Symlink staging is a deliberate
# opt-out from the (copying) default, so this is the "if there is an issue,
# say so in plain English" half — not a warning anyone hits by accident.
_SYMLINK_STAGING_NOTICE = (
    "NOTE  staged with --symlink: these are absolute symlinks into the source "
    "corpus, so this staged tree only works on this machine. It will read as an "
    "EMPTY corpus inside a container (`make docker-run` bind-mounts it read-only "
    "and every link dangles). Re-run `playbook stage` without --symlink to get "
    "real copies."
)


@cli.command(name="stage")
@click.argument("src_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option(
    "--out",
    "out_dir",
    type=click.Path(path_type=Path),
    default=None,
    help=(
        "Staging output directory (default: ~/.cache/playbook-engine/staging/<src_dir_name>). "
        "If it already exists, it is replaced — refused if it overlaps SRC_DIR, or if it's a "
        "non-empty directory that isn't itself a previous staging output."
    ),
)
@click.option(
    "--copy/--symlink",
    "copy_files",
    default=True,
    help=(
        "--copy (default) writes real file copies, so the staged corpus is "
        "self-contained and still readable after it crosses a filesystem "
        "boundary — e.g. bind-mounted read-only into the container by "
        "`make docker-run`. --symlink writes absolute symlinks instead: cheaper "
        "and duplication-free, but valid only on this host — inside a container "
        "every link dangles and the corpus reads as empty."
    ),
)
@click.option(
    "--plan-only",
    "plan_only",
    is_flag=True,
    default=False,
    help=(
        "Don't stage — write a staging_plan.json proposal (deals/order/signed, "
        "assembled from file contents and metadata) to the output directory for "
        "review. Required first step for a corpus whose layout is 'unknown'; "
        "works for any layout."
    ),
)
@click.option(
    "--from-plan",
    "--plan",
    "plan_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help=(
        "Execute a staging_plan.json previously written by --plan-only "
        "(optionally hand/skill-edited) instead of detecting the layout. "
        "Preferred spelling is --from-plan; --plan is accepted as a "
        "deprecated alias for one release (on `judge`, --plan means "
        "dry-run/preview instead, so the same flag name carried opposite "
        "risk profiles on sibling subcommands)."
    ),
)
def stage_cmd(
    src_dir: Path,
    out_dir: Path | None,
    copy_files: bool,
    plan_only: bool,
    plan_path: Path | None,
) -> None:
    """Stage SRC_DIR into the flat layout the engine walker expects.

    Detects the directory layout (flat, CLM-nested, or manifest-driven),
    flattens each negotiation trail into ``out/<agreement>/<NN>__<name>``
    as real file copies (or absolute symlinks with ``--symlink``), writes per-agreement
    ``hints.yaml`` (order + signed_version), and emits a
    ``playbook.config.yaml`` skeleton.

    When the layout can't be determined (``unknown`` — loose files, ad-hoc
    trees, no per-agreement subfolders) staging refuses to guess; run with
    ``--plan-only`` first to assemble a ``staging_plan.json`` proposal from
    file contents/metadata, review/edit it, then re-run with
    ``--from-plan staging_plan.json`` (``--plan`` also accepted, deprecated)
    to execute it.

    Writes only to the output directory (default:
    ~/.cache/playbook-engine/staging/<name>, a user-owned cache dir rather
    than world-readable /tmp). Never modifies SRC_DIR.
    """
    import json  # noqa: PLC0415

    from playbook_engine.intake_plan import build_staging_plan, execute_staging_plan
    from playbook_engine.staging import (  # noqa: PLC0415
        DEFAULT_STAGING_ROOT,
        ensure_staging_dest,
        scaffold_config,
        stage,
    )

    resolved = src_dir.resolve()
    dest = (out_dir or DEFAULT_STAGING_ROOT / resolved.name).resolve()

    click.echo(f"src    : {resolved}")
    click.echo(f"out    : {dest}")

    if plan_path is not None:
        try:
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            click.secho(f"ERROR: {plan_path} is not valid JSON: {exc}", fg="red", err=True)
            raise SystemExit(1) from exc
        try:
            result = execute_staging_plan(plan, resolved, dest, copy_files=copy_files)
        except ValueError as exc:
            click.secho(f"ERROR: {exc}", fg="red", err=True)
            raise SystemExit(1) from exc
        # execute_staging_plan's _recreate_out_dir rmtrees dest before staging,
        # which would silently destroy the hand-edited plan the user just
        # reviewed if it lived at dest/staging_plan.json (issue #36). Re-persist
        # the executed plan now that dest exists again, so re-running the exact
        # `stage --plan` command this tool prints stays possible and the
        # reviewed/edited plan survives as the staging record.
        executed_plan_file = dest / "staging_plan.json"
        executed_plan_file.write_text(json.dumps(plan, indent=2), encoding="utf-8")
        click.echo(f"layout : {result.layout} (from plan {plan_path})")
        click.echo(
            f"staged : {result.staged_count} version(s) across {result.agreement_count} agreement(s)"
            + (" (copied)" if copy_files else " (symlinked)")
        )
        if not copy_files:
            click.secho(_SYMLINK_STAGING_NOTICE, fg="yellow", err=True)
        click.echo(f"plan   : {executed_plan_file} (preserved as the staging record)")
        scaffold_config(resolved, dest)
        click.echo(f"config : {dest / 'playbook.config.yaml'} (skeleton — fill in taxonomy path)")
        click.secho(f"OK  {dest}", fg="green")
        return

    if plan_only:
        try:
            ensure_staging_dest(resolved, dest)
        except ValueError as exc:
            click.secho(f"ERROR: {exc}", fg="red", err=True)
            raise SystemExit(1) from exc
        plan = build_staging_plan(resolved, progress=click.echo)
        plan_file = dest / "staging_plan.json"
        plan_file.write_text(json.dumps(plan, indent=2), encoding="utf-8")
        click.echo(f"plan   : {plan_file}")
        click.echo(
            f"        {len(plan['deals'])} candidate deal(s), "
            f"{len(plan['unassigned'])} unassigned file(s)"
        )
        click.secho(
            f"OK  wrote {plan_file} — review/edit, then run "
            f"`playbook stage --from-plan {plan_file}`",
            fg="green",
        )
        return

    try:
        result = stage(resolved, dest, copy_files=copy_files)
    except ValueError as exc:
        click.secho(f"ERROR: {exc}", fg="red", err=True)
        raise SystemExit(1) from exc

    click.echo(f"layout : {result.layout}")
    click.echo(
        f"staged : {result.staged_count} version(s) across {result.agreement_count} agreement(s)"
        + (" (copied)" if copy_files else " (symlinked)")
    )
    if not copy_files:
        click.secho(_SYMLINK_STAGING_NOTICE, fg="yellow", err=True)

    scaffold_config(resolved, dest)
    click.echo(f"config : {dest / 'playbook.config.yaml'} (skeleton — fill in taxonomy path)")

    if result.missing:
        click.secho(
            f"WARN {len(result.missing)} manifest file(s) missing on disk:", fg="yellow", err=True
        )
        for m in result.missing[:10]:
            click.secho(f"       {m}", fg="yellow", err=True)

    click.secho(f"OK  {dest}", fg="green")


def _queue_equivalence_round(mined_dir: Path, cfg: Any, taxonomy: Any, judge: Any) -> None:
    """Queue the equivalence questions of the store just mined into *mined_dir* (issue #240).

    A no-op without a canonical template (no clause has an ``our_standard``
    to be compared with). Runs after mining because the questions come from the
    precedent record, not from a seam inside ``mine_corpus``.
    """
    from playbook_engine.pipeline import queue_equivalence  # noqa: PLC0415

    if not cfg.baseline.has_canonical_template:
        return
    subjects, unjudged = queue_equivalence(mined_dir, cfg, taxonomy, judge)
    click.echo(
        f"  equivalence: {subjects} distinct text(s) against our standard, {unjudged} unjudged"
    )


def _judge_check_equivalence(out_dir: Path, cfg: Any, taxonomy: Any, *, use_api: bool) -> None:
    """``playbook judge --check equivalence OUT``: emit (and with --api, answer) the check queues."""
    from playbook_engine.agent_judge import VerdictStore  # noqa: PLC0415
    from playbook_engine.equivalence_check import (  # noqa: PLC0415
        apply_check_records,
        build_check_queues,
        check_via_api,
        write_check_queues,
    )
    from playbook_engine.pipeline import equivalence_subjects  # noqa: PLC0415

    verdicts_path = out_dir / "judge" / "verdicts.jsonl"
    if not out_dir.is_dir() or not (out_dir / "observations.jsonl").is_file():
        click.secho(
            f"ERROR: {out_dir} holds no observation store — run `playbook mine` first",
            fg="red",
            err=True,
        )
        raise SystemExit(1)
    if not cfg.baseline.has_canonical_template:
        click.secho(
            "OK  no canonical template configured: nothing is judged against a standard, "
            "so there is nothing to check",
            fg="green",
        )
        return
    subjects = equivalence_subjects(out_dir, cfg, taxonomy)
    store = VerdictStore(verdicts_path)
    queues = build_check_queues(store, subjects)

    if use_api:
        import anthropic  # noqa: PLC0415

        if not os.environ.get("ANTHROPIC_API_KEY"):
            click.secho(
                "ERROR: --api needs ANTHROPIC_API_KEY (Anthropic credentials were not found); "
                "answer the queues with a claude-opus-5-5 / xhigh agent instead and record "
                "them with `playbook judge-apply --check`",
                fg="red",
                err=True,
            )
            raise SystemExit(1)
        client = anthropic.Anthropic()
        unchecked = 0
        for stage, items in (("check", queues.check), ("adjudication", None)):
            if items is None:
                # The adjudication queue is rebuilt after the checks are applied.
                queues = build_check_queues(store, subjects)
                items = queues.adjudication
            records, missed = check_via_api(items, client=client, progress=click.echo)
            unchecked += missed
            try:
                result = apply_check_records(
                    store, [(i, r) for i, r in enumerate(records, start=1)]
                )
            except ValueError as exc:
                click.secho(f"ERROR: {stage} answers rejected: {exc}", fg="red", err=True)
                raise SystemExit(1) from exc
            for key, verdict in result.updates:
                prior = store.get_record_by_key(key)
                store.put_by_key(key, verdict, rubric=prior.rubric if prior else None)
            click.echo(
                f"  {stage}: {len(records)} answered, {missed} left unchecked "
                f"(agreed {result.agreed}, disagreed {result.disagreed}, "
                f"adjudicated {result.adjudicated})"
            )
        queues = build_check_queues(store, subjects)
        if unchecked:
            click.secho(f"WARN: {unchecked} item(s) left unchecked", fg="yellow", err=True)

    check_path, adjudication_path = write_check_queues(out_dir / "judge", queues)
    click.secho(f"OK  {check_path}", fg="green")
    click.echo(
        f"equivalence check: {len(queues.check)} to check, {len(queues.adjudication)} to "
        f"adjudicate, {queues.settled} settled, {queues.undrafted} not drafted yet"
    )
    if queues.adjudication:
        click.echo(f"  adjudication queue: {adjudication_path}")


@cli.command(name="judge")
@click.argument("corpus_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option(
    "--config", "config_path", type=click.Path(exists=True, path_type=Path), required=True
)
@click.option(
    "--out",
    "out_path",
    type=click.Path(path_type=Path),
    default=None,
    help="Output directory (default: <corpus_dir>/../out).",
)
@click.option(
    "--plan-only",
    "--plan",
    "plan_only",
    is_flag=True,
    default=False,
    help=(
        "Print deduped pending counts by kind and a rough token estimate, then exit "
        "without writing observations.jsonl. Preferred spelling is --plan-only "
        "(matches `stage --plan-only`'s preview meaning); --plan is accepted as "
        "an alias."
    ),
)
@click.option(
    "--subset",
    "subset",
    type=int,
    default=None,
    help="Record at most N pending items (trial mode).",
)
@click.option(
    "--accept-stale",
    "accept_stale",
    is_flag=True,
    default=False,
    help=(
        "Replay stored verdicts whose rubric version no longer matches the current "
        "one, instead of re-queueing them. Counts are still reported. Use when you "
        "have decided a rubric change does not affect the banked judgments."
    ),
)
@click.option(
    "--strict-rubric",
    "strict_rubric",
    is_flag=True,
    default=False,
    help=(
        "Treat stored verdicts that carry NO rubric version (banked before rubric "
        "versioning existed) as stale and re-queue them, instead of replaying them "
        "with a warning."
    ),
)
@click.option(
    "--entity-registry",
    "entity_registry_path",
    type=click.Path(path_type=Path),
    default=None,
    help=(
        "Path to the born-safe entity registry (alias->real-name map). Defaults "
        "to ~/.cache/playbook-engine/entity_registry.json — a machine-global, "
        "persistent file. Point it into your gitignored output dir (e.g. "
        "<out>/entity_registry.json, matching 'playbook mine --entity-registry') "
        "to keep all sensitive real-name data in one place across mine and judge "
        "rounds against the same out-dir. Only relevant when "
        "provenance.known_entities is set."
    ),
)
@click.option(
    "--check",
    "check_kind",
    type=click.Choice(["equivalence"]),
    default=None,
    help=(
        "Instead of mining, write the blind-check queue for drafted verdicts of this kind "
        "(judge/check-pending.jsonl: the drafter's payload, no label or reason) and the "
        "adjudication queue for disputed ones (judge/adjudication-pending.jsonl). With "
        "--check the first argument is the derivation OUT dir. Answer the queues with a "
        "separate claude-opus-5-5 / xhigh agent and record the answers with "
        "`playbook judge-apply --check`, or pass --api."
    ),
)
@click.option(
    "--api",
    "use_api",
    is_flag=True,
    default=False,
    help=(
        "With --check: answer the check and adjudication queues through the Anthropic "
        "Message Batches API (claude-opus-5-5, effort xhigh; needs ANTHROPIC_API_KEY) and "
        "record the answers. A refusal leaves the item unchecked; there is no fallback model."
    ),
)
@click.option(
    "--skip-preflight",
    "skip_preflight",
    is_flag=True,
    default=False,
    help=_SKIP_PREFLIGHT_HELP,
)
@_accept_environment_change_option
def judge_cmd(
    corpus_dir: Path,
    config_path: Path,
    out_path: Path | None,
    plan_only: bool,
    subset: int | None,
    accept_stale: bool,
    strict_rubric: bool,
    entity_registry_path: Path | None,
    check_kind: str | None,
    use_api: bool,
    skip_preflight: bool,
    accept_environment_change: bool,
) -> None:
    """Mine the corpus with store-backed judges and emit the pending review queue.

    Scope, classification and provenance items are queued while mining, and
    once a canonical template is configured an ``equivalence``
    item for every distinct non-standard text of the precedent record — a
    deal's signed text, a non-standard opening, a refused ask — that has no
    verdict against our standard yet. Every deviation is the deterministic
    standard check, so nothing is judged for it.

    ``--check equivalence OUT`` skips mining and queues the independent blind
    check (and adjudication) of the verdicts already drafted into OUT; see the
    option's help.

    Reads the verdict store at <out>/judge/verdicts.jsonl and replays any
    previously supplied verdicts.  For every new clause payload not in the store,
    appends a full record to <out>/judge/pending.jsonl.

    The stage cache stays on: a document is replayed from it only
    when every stored verdict it was built from is unchanged, and a document
    with anything pending is never cached — so each round re-mines exactly
    the documents whose verdicts moved or are still outstanding, and the
    pending queue is complete. ``--plan-only`` reads through the same cache.

    Use ``playbook judge-apply`` to load verdicts into the store, then re-run
    ``playbook judge`` to confirm no new items are pending.  Finally run
    ``playbook mine`` + ``playbook project`` for the final playbook.

    Runs the same ``lint-corpus`` preflight ``mine``/``segment`` run and refuses
    to start if it fails — the drain loop calls ``judge`` over hours or days, so
    a corpus that breaks between rounds (staged tree moved, symlinks now
    dangling, config template path broken) must not sail into a judge round.
    ``--skip-preflight`` opts out.
    """
    from playbook_engine.agent_judge import (  # noqa: PLC0415
        PendingQueue,
        StoreBackedClassificationJudge,
        StoreBackedEquivalenceJudge,
        StoreBackedProvenanceJudge,
        StoreBackedScopeJudge,
        VerdictStore,
    )
    from playbook_engine.rubric import RubricPolicy, current_versions  # noqa: PLC0415

    try:
        cfg = load_config(config_path)
    except ConfigError as exc:
        click.secho(f"Config error: {exc}", fg="red", err=True)
        raise SystemExit(1) from exc

    try:
        taxonomy = load_taxonomy(cfg.taxonomy_path)
    except TaxonomyError as exc:
        click.secho(f"Taxonomy error: {exc}", fg="red", err=True)
        raise SystemExit(1) from exc

    if use_api and check_kind is None:
        click.secho("ERROR: --api only applies with --check", fg="red", err=True)
        raise SystemExit(1)
    if check_kind is not None:
        _judge_check_equivalence((out_path or corpus_dir).resolve(), cfg, taxonomy, use_api=use_api)
        return

    out_dir = (out_path or corpus_dir.parent / "out").resolve()
    judge_dir = out_dir / "judge"
    verdicts_path = judge_dir / "verdicts.jsonl"
    pending_path = judge_dir / "pending.jsonl"

    # One policy shared by all four judges: the staleness knobs AND the tally
    # both --plan-only and normal mode report from.
    rubric_policy = RubricPolicy(strict_legacy=strict_rubric, accept_stale=accept_stale)
    versions = current_versions(taxonomy=taxonomy, agreement_type=cfg.agreement_type)

    click.echo(f"corpus  : {corpus_dir}")
    click.echo(f"config  : {config_path}")
    click.echo(f"out     : {out_dir}")
    click.echo(f"verdicts: {verdicts_path}")
    click.echo("rubric  : " + "  ".join(f"{k}={v}" for k, v in sorted(versions.items())))

    # Provenance preflight (issue #121) — same check ``mine`` runs, for the
    # same reason, and deliberately BEFORE the --plan branch: --plan writes
    # its observations into a TemporaryDirectory but reads through the real
    # out_dir's extraction/segmentation caches, so a changed environment
    # makes a plan estimate wrong (every version reported as uncached) even
    # though nothing durable is written.
    environment = _preflight_environment(
        out_dir,
        cfg,
        corpus_dir,
        command="judge",
        accept_change=accept_environment_change,
    )

    # Corpus preflight (issue #172) — the same `lint-corpus` gate ``mine`` and
    # ``segment`` run, and for the same reason: without it, a corpus that goes
    # bad between drain-loop rounds (staged tree moved, symlinks now dangling,
    # config template path broken) sails straight into `mine_corpus` below and
    # either produces a confusing raw pipeline error or — for per-file
    # extraction failures, which only warn-and-quarantine — a "finished,
    # wrong-looking-like-right" thin derivation. Ordered AFTER the environment
    # preflight and BEFORE the --plan branch, matching `mine`: --plan still
    # reads the corpus (see the --plan comment below), so it needs this gate too.
    _run_corpus_preflight(corpus_dir, config_path, skip=skip_preflight, command="playbook judge")

    # Segment exactly the way ``mine`` does, or the store-backed judges here
    # generate verdict keys that never match the LLM-segmented observation store
    # and the drain loop cannot converge.  Keyed off the real out_dir so the
    # segmentation AND extraction caches (issue #132) are shared with plan
    # mode and later ``mine`` runs — even though --plan mode below
    # still writes observations.jsonl/corpus_manifest.json/etc. into an
    # ephemeral TemporaryDirectory (it must never touch the real out_dir's
    # observation store — see its own docstring), the *caches* it reads
    # through are rooted here, so a --plan run reuses whatever a prior
    # judge/mine/compile round already extracted/segmented for this out_dir
    # instead of re-mining every version's content from scratch.
    # Collected by the LLM-segmentation closures (issue #134) so --plan can
    # report the segmentation spend alongside the judge-item estimate — the
    # single largest cost in a live run, and previously absent from the plan
    # gate entirely. Harmless (a few dict-counter updates) when judge is run
    # without --plan; only the --plan branch below reads it.
    seg_stats: dict[str, int] = {}

    try:
        seg_kwargs = _llm_segmentation_kwargs(cfg, taxonomy, out_dir, click.echo, stats=seg_stats)
    except ConfigError as exc:
        click.secho(f"Config error: {exc}", fg="red", err=True)
        raise SystemExit(1) from exc

    store = VerdictStore(verdicts_path)

    # Validate --subset once, up front, so it applies uniformly to both the
    # --plan branch and the normal-mode branch below (issue #295: this check
    # used to live only in normal mode, so `--plan --subset 0` passed
    # silently instead of being rejected like a normal-mode run would be).
    if subset is not None and subset <= 0:
        click.secho("ERROR: --subset must be a positive integer", fg="red", err=True)
        raise SystemExit(1)

    # --plan mode: report pending counts without writing observations.
    # We still run mine_corpus (with a temp pending queue) to compute the plan.
    if plan_only:
        import tempfile  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as _tmp:
            plan_pending = PendingQueue(Path(_tmp) / "pending.jsonl")
            scope_judge = StoreBackedScopeJudge(
                store=store, pending=plan_pending, rubric=rubric_policy
            )
            cls_judge = StoreBackedClassificationJudge(
                store=store, pending=plan_pending, rubric=rubric_policy
            )
            prov_judge = StoreBackedProvenanceJudge(
                store=store, pending=plan_pending, rubric=rubric_policy
            )

            try:
                mine_corpus(
                    corpus_dir=corpus_dir.resolve(),
                    config=cfg,
                    taxonomy=taxonomy,
                    out_dir=Path(_tmp) / "mine_out",
                    scope_judge=scope_judge,
                    classification_judge=cls_judge,
                    provenance_judge=prov_judge,
                    # Issue #219: the plan reads through the REAL out-dir's
                    # stage cache rather than re-mining every document into
                    # the temp dir. A replayed document is one whose every
                    # stored verdict is unchanged and which queued nothing,
                    # so it adds nothing to the plan; every other document
                    # recomputes and queues exactly what the round would.
                    cache_dir=out_dir / ".cache",
                    # extraction_cache must stay warm across judge rounds or
                    # every round re-burns docling OCR from scratch (issue
                    # #78; the regression issue #132 originally fixed).
                    refresh_extraction=False,
                    entity_registry_path=(
                        entity_registry_path.resolve() if entity_registry_path else None
                    ),
                    progress=click.echo,
                    **seg_kwargs,
                )
                _queue_equivalence_round(
                    Path(_tmp) / "mine_out",
                    cfg,
                    taxonomy,
                    StoreBackedEquivalenceJudge(
                        store=store, pending=plan_pending, rubric=rubric_policy
                    ),
                )
            except PipelineError as exc:
                click.secho(f"ERROR: {exc}", fg="red", err=True)
                raise SystemExit(1) from exc

            plan_pending_path = Path(_tmp) / "pending.jsonl"
            if not plan_pending_path.exists():
                click.secho("OK  0 pending items (all verdicts already in store)", fg="green")
                _echo_segmentation_cost_line(seg_stats, click.echo)
                _echo_rubric_report(rubric_policy, click.echo)
                return

            import json  # noqa: PLC0415

            pending_records = [
                json.loads(line)
                for line in plan_pending_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

            # Apply --subset (issue #295): --plan previously ignored --subset
            # entirely and reported full-corpus counts/token estimate even
            # though the intended run would only process the first N items.
            # Cap the plan-mode pending_records here, before counts/token
            # estimate are computed, so the plan reflects the trial run the
            # caller is about to execute.
            if subset is not None and len(pending_records) > subset:
                click.secho(
                    f"--subset {subset}: plan reflects first {subset} of "
                    f"{len(pending_records)} pending items",
                    fg="yellow",
                )
                pending_records = pending_records[:subset]

            counts: dict[str, int] = {}
            for rec in pending_records:
                counts[rec["kind"]] = counts.get(rec["kind"], 0) + 1

            total = sum(counts.values())
            # Token estimate from the real payload sizes (issue #134) — a
            # flat per-item average previously ignored that a provenance
            # payload (preamble + letterhead) can differ from a short
            # classify payload by an order of magnitude. ``//4`` is the same chars-per-token
            # rule of thumb used elsewhere for rough English-text estimates
            # (there is no tokenizer dependency in this codebase).
            total_chars = sum(
                len(json.dumps(rec["payload"], sort_keys=True, ensure_ascii=False))
                for rec in pending_records
            )
            token_estimate = total_chars // 4

            click.echo(f"Pending items: {total} (token estimate: ~{token_estimate:,})")
            for kind, count in sorted(counts.items()):
                click.echo(f"  {kind}: {count}")
            _echo_segmentation_cost_line(seg_stats, click.echo)
            _echo_rubric_report(rubric_policy, click.echo)
        return

    # Normal mode: write observations and update pending queue.
    # Reset the pending queue so each round is rewritten from scratch (the
    # contract SKILL.md documents). PendingQueue appends and never truncates,
    # so without this a re-run after judge-apply keeps the prior round's stale
    # entries and the drain loop can never reach empty (issue #182).
    pending_path.unlink(missing_ok=True)

    pending_queue = PendingQueue(pending_path)
    scope_judge = StoreBackedScopeJudge(store=store, pending=pending_queue, rubric=rubric_policy)
    cls_judge = StoreBackedClassificationJudge(
        store=store, pending=pending_queue, rubric=rubric_policy
    )
    prov_judge = StoreBackedProvenanceJudge(
        store=store, pending=pending_queue, rubric=rubric_policy
    )

    try:
        mine_corpus(
            corpus_dir=corpus_dir.resolve(),
            config=cfg,
            taxonomy=taxonomy,
            out_dir=out_dir,
            scope_judge=scope_judge,
            classification_judge=cls_judge,
            provenance_judge=prov_judge,
            # The stage cache stays on (issue #219) — see the --plan-only
            # branch above. extraction_cache must stay warm across judge
            # rounds (issue #78).
            refresh_extraction=False,
            entity_registry_path=(entity_registry_path.resolve() if entity_registry_path else None),
            progress=click.echo,
            **seg_kwargs,
        )
        _queue_equivalence_round(
            out_dir,
            cfg,
            taxonomy,
            StoreBackedEquivalenceJudge(store=store, pending=pending_queue, rubric=rubric_policy),
        )
    except PipelineError as exc:
        click.secho(f"ERROR: {exc}", fg="red", err=True)
        raise SystemExit(1) from exc

    # Stamp the out-dir with the environment that just wrote into it (issue
    # #121). Normal mode only — the --plan branch above returns before this,
    # since a plan run writes its observations to a temp dir and has no claim
    # to make about what produced THIS out-dir's artifacts.
    _record_run_manifest(out_dir, environment, "judge")

    # Report pending counts.
    import json  # noqa: PLC0415

    if pending_path.exists():
        pending_lines = [
            line for line in pending_path.read_text(encoding="utf-8").splitlines() if line.strip()
        ]
        pending_records = [json.loads(line) for line in pending_lines]
        counts = {}
        for rec in pending_records:
            counts[rec["kind"]] = counts.get(rec["kind"], 0) + 1
        total_pending = len(pending_records)

        # Apply --subset: truncate the pending queue to N items.
        if subset is not None and total_pending > subset:
            click.secho(
                f"--subset {subset}: keeping first {subset} of {total_pending} pending items",
                fg="yellow",
            )
            truncated_lines = pending_lines[:subset]
            pending_path.write_text("\n".join(truncated_lines) + "\n", encoding="utf-8")
            pending_records = pending_records[:subset]
            counts = {}
            for rec in pending_records:
                counts[rec["kind"]] = counts.get(rec["kind"], 0) + 1
            total_pending = subset

        click.secho(f"OK  {out_dir / 'observations.jsonl'}", fg="green")
        click.echo(f"Pending items: {total_pending}")
        for kind, count in sorted(counts.items()):
            click.echo(f"  {kind}: {count}")

        # A pending item whose key is ALREADY in the verdict store is a
        # re-queue of a stored verdict that failed replay reconstruction
        # (malformed enum/basis/missing field — issue #182). Surface it:
        # without this the drain loop shrinks-then-stalls with no visible
        # reason (the only trace is a suppressed logging.warning).
        # A rubric-driven re-queue (stored verdict made under an older rubric)
        # is ALSO a key that is already in the store, but it is expected and
        # already reported by _echo_rubric_report — excluded here so the
        # malformed-verdict warning keeps meaning what it says.
        requeued = [
            rec
            for rec in pending_records
            if (stored := store.get_record_by_key(rec["key"])) is not None
            and rubric_policy.would_replay(stored.rubric_version, rec.get("rubric_version"))
        ]
        if requeued:
            requeue_counts: dict[str, int] = {}
            for rec in requeued:
                requeue_counts[rec["kind"]] = requeue_counts.get(rec["kind"], 0) + 1
            breakdown = ", ".join(f"{k}: {c}" for k, c in sorted(requeue_counts.items()))
            click.secho(
                f"WARNING: {len(requeued)} pending item(s) are re-queues of stored "
                f"verdicts that failed replay validation ({breakdown}) — re-emitting "
                "the same verdict will loop forever; fix the verdict (see "
                "REFERENCE.md enum values) and judge-apply again",
                fg="yellow",
                err=True,
            )
        _echo_rubric_report(rubric_policy, click.echo)
    else:
        click.secho(f"OK  {out_dir / 'observations.jsonl'} (0 pending items)", fg="green")
        _echo_rubric_report(rubric_policy, click.echo)

    # This round ran mine_corpus to completion (the except PipelineError
    # branch above exits before here). pending_path is unlinked at the top
    # of this function and PendingQueue only creates the file lazily via
    # add(), so a round that queues 0 new items leaves no file on disk —
    # indistinguishable from a round killed mid-mine, which also leaves the
    # file absent. Write an explicit empty file so ABSENCE unambiguously
    # means "interrupted", never "finished with nothing pending" (issue
    # #170). REFERENCE.md's done-criteria requires the file to exist and be
    # empty; it no longer accepts absence as done.
    if not pending_path.exists():
        pending_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_pending = pending_path.with_name(pending_path.name + ".tmp")
        tmp_pending.write_text("", encoding="utf-8")
        os.replace(tmp_pending, pending_path)


def _judge_apply_check(out_dir: Path, check_path: Path, allow_checker_model: bool) -> None:
    """``judge-apply --check``: record equivalence check / adjudication answers."""
    from playbook_engine.agent_judge import VerdictStore  # noqa: PLC0415
    from playbook_engine.equivalence_check import (  # noqa: PLC0415
        apply_check_records,
        load_check_records,
    )

    out_dir_resolved = out_dir.resolve()
    if not out_dir_resolved.is_dir():
        click.secho(f"ERROR: {out_dir_resolved} does not exist", fg="red", err=True)
        raise SystemExit(1)
    store = VerdictStore(out_dir_resolved / "judge" / "verdicts.jsonl")
    try:
        result = apply_check_records(
            store, load_check_records(check_path), allow_checker_model=allow_checker_model
        )
    except ValueError as exc:
        click.secho(f"ERROR: {exc}", fg="red", err=True)
        raise SystemExit(1) from exc
    for key, verdict in result.updates:
        prior = store.get_record_by_key(key)
        # Same rubric stamp as the draft: the check answers the question as asked.
        store.put_by_key(key, verdict, rubric=prior.rubric if prior else None)
    click.secho(
        f"OK  recorded {len(result.updates)} check answer(s) into "
        f"{out_dir_resolved / 'judge' / 'verdicts.jsonl'}",
        fg="green",
    )
    click.echo(
        f"  agreed {result.agreed}, disagreed {result.disagreed} (awaiting adjudication), "
        f"adjudicated {result.adjudicated}, owner-decided (skipped) {result.skipped_owner}"
    )
    if result.disagreed:
        click.echo("  re-run `playbook judge --check equivalence` to queue the disagreements")


@cli.command(name="judge-apply")
@click.argument("out_dir", type=click.Path(file_okay=False, path_type=Path))
@click.option(
    "--verdicts",
    "verdicts_path",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="JSONL file of verdicts to load (each line: {'key': '<sha256>', 'verdict': {...}}).",
)
@click.option(
    "--check",
    "check_path",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help=(
        "JSONL file of equivalence check or adjudication answers instead of verdicts (each "
        "line: {'key', 'label', 'reason', 'model', 'effort'}). A record whose model is not "
        "claude-opus-5-5 or whose effort is not xhigh is rejected unless "
        "--allow-checker-model is passed."
    ),
)
@click.option(
    "--allow-checker-model",
    "allow_checker_model",
    is_flag=True,
    default=False,
    help="With --check: accept a checker model or effort other than claude-opus-5-5 / xhigh.",
)
def judge_apply_cmd(
    out_dir: Path,
    verdicts_path: Path | None,
    check_path: Path | None,
    allow_checker_model: bool,
) -> None:
    """Load verdicts from a JSONL file into the verdict store.

    Reads OUT_DIR/judge/verdicts.jsonl (created by ``playbook judge``) and
    merges in the verdicts from VERDICTS_PATH.  Each line in VERDICTS_PATH must
    be a JSON object with a ``key`` (SHA-256 string) and a ``verdict`` dict.

    Malformed lines are rejected with a non-zero exit code and the line number
    reported.  All lines are validated first; nothing is loaded if any line
    fails.

    After applying verdicts, re-run ``playbook judge`` to confirm the pending
    queue is empty, then run ``playbook mine`` + ``playbook project`` to compile
    the final playbook with the judged taxonomy_ids populated.

    ``--check FILE`` loads the answers to the equivalence blind-check and
    adjudication queues (``playbook judge --check equivalence``) instead; see
    the option.
    """
    import json  # noqa: PLC0415

    if (verdicts_path is None) == (check_path is None):
        click.secho("ERROR: pass exactly one of --verdicts or --check", fg="red", err=True)
        raise SystemExit(1)
    if allow_checker_model and check_path is None:
        click.secho("ERROR: --allow-checker-model only applies with --check", fg="red", err=True)
        raise SystemExit(1)
    if check_path is not None:
        _judge_apply_check(out_dir, check_path, allow_checker_model)
        return
    assert verdicts_path is not None

    from playbook_engine.agent_judge import (  # noqa: PLC0415
        VerdictStore,
        infer_verdict_kind,
        validate_verdict,
    )
    from playbook_engine.rubric import RubricStamp  # noqa: PLC0415

    out_dir_resolved = out_dir.resolve()
    if not out_dir_resolved.is_dir():
        click.secho(
            f"ERROR: {out_dir_resolved} does not exist — is the path right? "
            "(the judge/ subdirectory is created automatically on first write, "
            "so this only rejects a missing OUT_DIR itself, not a fresh "
            "pre-seeded one)",
            fg="red",
            err=True,
        )
        raise SystemExit(1)
    verdicts_store_path = out_dir_resolved / "judge" / "verdicts.jsonl"

    # Kind lookup for semantic validation: pending.jsonl (when present) maps
    # each key to its item kind; verdicts for keys not currently pending fall
    # back to field-shape inference (see infer_verdict_kind).
    #
    # The same pass reads each item's ``rubric_version`` — the rubric that was
    # in force when the question was posed. Stamping the incoming verdict with
    # THAT (rather than recomputing "now") is what makes the stamp mean "this
    # is an answer to the question as asked", and it keeps judge-apply free of
    # any --config dependency: the version travels with the queue.
    pending_kinds: dict[str, str] = {}
    pending_rubrics: dict[str, str] = {}
    pending_path = out_dir_resolved / "judge" / "pending.jsonl"
    if pending_path.is_file():
        for pline in pending_path.read_text(encoding="utf-8").splitlines():
            pline = pline.strip()
            if not pline:
                continue
            try:
                pitem = json.loads(pline)
            except json.JSONDecodeError:
                continue
            if isinstance(pitem, dict) and "key" in pitem and "kind" in pitem:
                pending_kinds[pitem["key"]] = pitem["kind"]
                rubric_v = pitem.get("rubric_version")
                if isinstance(rubric_v, str) and rubric_v:
                    pending_rubrics[pitem["key"]] = rubric_v

    # Validate all lines first before touching the store.
    raw_lines = verdicts_path.read_text(encoding="utf-8").splitlines()
    valid_records: list[tuple[str, dict[str, Any], str]] = []
    unknown_keys = 0
    for lineno, line in enumerate(raw_lines, start=1):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            click.secho(f"ERROR: line {lineno}: invalid JSON: {exc}", fg="red", err=True)
            raise SystemExit(1) from exc

        if not isinstance(record, dict):
            click.secho(
                f"ERROR: line {lineno}: expected a JSON object, got {type(record).__name__}",
                fg="red",
                err=True,
            )
            raise SystemExit(1)
        if "key" not in record:
            click.secho(f"ERROR: line {lineno}: missing 'key' field", fg="red", err=True)
            raise SystemExit(1)
        if "verdict" not in record:
            click.secho(f"ERROR: line {lineno}: missing 'verdict' field", fg="red", err=True)
            raise SystemExit(1)
        if not isinstance(record["key"], str):
            click.secho(f"ERROR: line {lineno}: 'key' must be a string", fg="red", err=True)
            raise SystemExit(1)
        if not isinstance(record["verdict"], dict):
            click.secho(
                f"ERROR: line {lineno}: 'verdict' must be a JSON object", fg="red", err=True
            )
            raise SystemExit(1)

        # Semantic validation (issue #182 family): a verdict that would fail
        # dataclass reconstruction on replay must be rejected HERE, with a
        # line number, instead of silently re-queueing forever at mine time.
        kind = pending_kinds.get(record["key"]) or infer_verdict_kind(record["verdict"])
        if kind is None:
            click.secho(
                f"ERROR: line {lineno}: cannot determine verdict kind (key not in "
                "pending.jsonl and no recognizable verdict fields)",
                fg="red",
                err=True,
            )
            raise SystemExit(1)
        try:
            validate_verdict(kind, record["verdict"])
        except (ValueError, TypeError) as exc:
            # TypeError is a defensive backstop: validate_verdict type-checks
            # confidence fields up front (issue #161) so a malformed verdict
            # should already surface as ValueError here, but any dataclass
            # construction this function reaches must never leak a bare,
            # line-number-free traceback to the producer.
            click.secho(f"ERROR: line {lineno} ({kind}): {exc}", fg="red", err=True)
            raise SystemExit(1) from exc
        if record["key"] not in pending_kinds and pending_kinds:
            unknown_keys += 1
        valid_records.append((record["key"], record["verdict"], kind))

    if unknown_keys:
        click.secho(
            f"WARN: {unknown_keys} verdict key(s) not in the current pending queue — "
            "they load into the store but will not drain any pending item "
            "(typo'd key, or a queue from a different round?)",
            fg="yellow",
            err=True,
        )

    if not valid_records:
        click.secho("WARN: no valid verdict records found in file", fg="yellow", err=True)
        return

    # Load all validated records into the store.
    store = VerdictStore(verdicts_store_path)
    loaded = 0
    unstamped = 0
    for key, verdict, kind in valid_records:
        # Use put_by_key to load verdicts by their pre-computed key directly,
        # bypassing the payload hashing step (the key was computed by the producer).
        version = pending_rubrics.get(key)
        stamp = RubricStamp(kind=kind, version=version) if version else None
        if stamp is None:
            # No queue entry to source the rubric from (key not in the current
            # pending.jsonl, or a queue written before rubric versioning).
            # Deliberately NOT recomputed from the current rubric: that would
            # assert something about this verdict that nothing here knows.
            unstamped += 1
        store.put_by_key(key, verdict, rubric=stamp)
        loaded += 1

    click.secho(f"OK  loaded {loaded} verdict(s) into {verdicts_store_path}", fg="green")
    if unstamped:
        click.secho(
            f"WARN: {unstamped} verdict(s) loaded without a rubric stamp — their "
            "pending entry carried no rubric_version (queue predates rubric "
            "versioning, or the key is not in the current queue). They will be "
            "reported as unversioned on the next `playbook judge` run.",
            fg="yellow",
            err=True,
        )


@cli.command(name="segment")
@click.argument("corpus_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option(
    "--config", "config_path", type=click.Path(exists=True, path_type=Path), required=True
)
@click.option(
    "--out",
    "out_path",
    type=click.Path(path_type=Path),
    default=None,
    help="Output directory (default: <corpus_dir>/../out).",
)
@click.option(
    "--skip-preflight",
    "skip_preflight",
    is_flag=True,
    default=False,
    help=_SKIP_PREFLIGHT_HELP,
)
@_accept_environment_change_option
def segment_cmd(
    corpus_dir: Path,
    config_path: Path,
    out_path: Path | None,
    skip_preflight: bool,
    accept_environment_change: bool,
) -> None:
    """Emit the agent segmentation queue for CORPUS_DIR.

    Key-free store-backed segmentation. Extracts each version and, for every
    document whose segmentation is not yet cached, appends its ``Block`` stream
    to ``<out>/segment/pending.jsonl``. Read that queue, partition each
    document's blocks into contiguous clause ranges — one node per clause —
    write them to a verdicts JSONL, then run
    ``playbook segment-apply`` and ``playbook mine``. No API key is used.

    The queue is rewritten from scratch each run, so re-running after
    ``segment-apply`` reports only what still needs segmenting (empty = done).
    Requires ``segmentation.agent: true`` in the config.

    Checks the stored run manifest first (so a changed environment is named as
    the root cause), then runs the same ``lint-corpus`` preflight ``mine``
    does, for the same reason: this is the stage that actually reads every
    version file and pays for extraction, so a corpus the walker cannot see
    (dangling symlinks) or an extraction environment that has silently
    degraded must stop the run here, not hours later when the banked
    segmentation work turns out to be keyed to canonical_text extracted under
    the wrong environment. ``--skip-preflight`` opts out.
    """
    from playbook_engine.agent_judge import PendingQueue  # noqa: PLC0415
    from playbook_engine.agent_segmenter import (  # noqa: PLC0415
        AGENT_SEGMENTER_MODEL,
        block_to_dict,
        segment_payload_key,
    )
    from playbook_engine.extraction import ExtractionCache, extract_blocks  # noqa: PLC0415
    from playbook_engine.llm_segmenter_batch import SegmentationVerdictCache  # noqa: PLC0415
    from playbook_engine.pipeline import _discover_versions  # noqa: PLC0415

    try:
        cfg = load_config(config_path)
    except ConfigError as exc:
        click.secho(f"Config error: {exc}", fg="red", err=True)
        raise SystemExit(1) from exc

    out_dir = (out_path or corpus_dir.parent / "out").resolve()

    # Provenance preflight — same check `mine`/`judge` run, for the same
    # reason, and deliberately BEFORE `_run_corpus_preflight` (mirrors
    # `mine`'s ordering comment): `segment` is the first stage on the agent
    # path that reads every version file and pays for extraction, so a
    # docling that vanished from the venv must be named as the root cause
    # here — not surfaced two steps removed by the corpus linter, and not
    # left to be discovered only at the subsequent `mine`, after an entire
    # agent segmentation pass has already been banked against canonical_text
    # hashes that changed out from under it.
    environment = _preflight_environment(
        out_dir,
        cfg,
        corpus_dir,
        command="segment",
        accept_change=accept_environment_change,
    )

    _run_corpus_preflight(corpus_dir, config_path, skip=skip_preflight, command="playbook segment")

    if not cfg.segmentation.agent:
        click.secho(
            "ERROR: `segment` requires `segmentation.agent: true` in the config.",
            fg="red",
            err=True,
        )
        raise SystemExit(1)
    # Fail loud, ONCE, before any per-version work starts — mirrors
    # cli._llm_segmentation_kwargs's identical check (issue #80). This
    # command has no shared kwargs-building helper of its own (it builds
    # its extraction_cache/etc. inline below), so the check is repeated
    # here rather than factored out, to avoid a drive-by refactor of its
    # existing structure.
    if cfg.extraction.extractor == "docling" and shutil.which("docling") is None:
        click.secho(
            "ERROR: extraction.extractor is set to 'docling' in the config, "
            "but the docling binary was not found on PATH. Install docling, "
            "run this corpus inside the project's container (see "
            "Dockerfile), or set extraction.extractor to 'legacy' or 'auto' "
            "(or omit the extraction: section) to use the legacy adapters "
            "instead.",
            fg="red",
            err=True,
        )
        raise SystemExit(1)
    try:
        taxonomy = load_taxonomy(cfg.taxonomy_path)
    except TaxonomyError as exc:
        click.secho(f"Taxonomy error: {exc}", fg="red", err=True)
        raise SystemExit(1) from exc

    seg_dir = out_dir / "segment"
    seg_dir.mkdir(parents=True, exist_ok=True)
    pending_path = seg_dir / "pending.jsonl"
    pending_path.unlink(missing_ok=True)  # fresh queue each round (mirrors judge, issue #182)

    cache = SegmentationVerdictCache(seg_dir / "cache.jsonl")
    extraction_cache = ExtractionCache(out_dir / "extraction_cache.jsonl")
    pending = PendingQueue(pending_path)
    taxonomy_ids = [e.id for e in taxonomy.classifier_entries()]

    click.echo(f"corpus : {corpus_dir}")
    click.echo(f"out    : {out_dir}")

    n_docs = n_versions = n_queued = n_cached = 0
    doc_dirs = sorted(
        d for d in corpus_dir.resolve().iterdir() if d.is_dir() and not d.name.startswith(".")
    )
    for doc_dir in doc_dirs:
        versions = _discover_versions(doc_dir)
        if not versions:
            continue
        n_docs += 1
        for path in versions:
            n_versions += 1
            try:
                canonical_text, blocks, _extractor = extract_blocks(
                    path, cache=extraction_cache, extractor=cfg.extraction.extractor
                )
            except Exception as exc:  # noqa: BLE001
                click.secho(
                    f"  WARNING: {doc_dir.name}/{path.name}: extraction failed ({exc}) — skipped",
                    fg="yellow",
                    err=True,
                )
                continue
            if cache.get(canonical_text, model=AGENT_SEGMENTER_MODEL) is not None:
                n_cached += 1
                continue
            if pending.add(
                segment_payload_key(canonical_text),
                "segment",
                {
                    "document_id": doc_dir.name,
                    "version": path.stem,
                    "taxonomy_ids": taxonomy_ids,
                    "canonical_text": canonical_text,
                    "blocks": [block_to_dict(b) for b in blocks],
                },
            ):
                n_queued += 1

    click.echo(
        f"Segmentation pending: {n_queued} "
        f"(cached: {n_cached}, versions: {n_versions}, docs: {n_docs})"
    )
    # Scoped extraction-cache invalidation (extraction._EXTRACTION_CACHE_FORMAT_LADDER):
    # an older-format entry is migrated in place where the format change
    # provably did not affect it, and only re-extracted where it did. Reported
    # because a migration is exactly the extraction an operator did NOT pay for
    # this run — silence would make that saving invisible.
    if extraction_cache.migrated_count or extraction_cache.invalidated_count:
        click.echo(
            f"Extraction cache: {extraction_cache.migrated_count} entr"
            f"{'y' if extraction_cache.migrated_count == 1 else 'ies'} migrated to the "
            f"current format (no re-extraction), {extraction_cache.invalidated_count} "
            "re-extracted"
        )
    if n_queued == 0:
        click.secho("OK  all documents segmented (cache full) — run `playbook mine`", fg="green")
    else:
        click.secho(
            f"OK  {pending_path} — segment each item, then `playbook segment-apply`", fg="green"
        )
    # Stamp the out-dir with what just extracted it (issue #173), so a
    # subsequent `segment`/`mine`/`judge` against this out_dir has something
    # to check itself against — mirrors `mine`'s identical stamp at the end
    # of its own run. Written only after the extraction pass above finished,
    # not on the early-exit paths above it.
    _record_run_manifest(out_dir, environment, "segment")


@cli.command(name="segment-apply")
@click.argument("out_dir", type=click.Path(file_okay=False, path_type=Path))
@click.option(
    "--verdicts",
    "verdicts_path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help=(
        'JSONL of segmentations (each line: {"canonical_text": "...", '
        '"nodes": [<segmentation node>, ...]}).'
    ),
)
def segment_apply_cmd(out_dir: Path, verdicts_path: Path) -> None:
    """Load agent-produced segmentation nodes into the segmentation cache.

    Each line of VERDICTS_PATH is a JSON object with ``canonical_text`` (echoed
    from the pending item) and ``nodes`` (a list of segmentation node dicts —
    ``node_id``, ``parent_id``, ``order``, ``heading``, ``taxonomy_id``,
    ``start_block_id``, ``end_block_id``, and optional ``start_quote`` /
    ``end_quote``). Nodes should partition the document's blocks into contiguous
    clause ranges. All lines are validated before the cache is touched.

    After applying, re-run ``playbook segment`` to confirm the queue is empty,
    then ``playbook mine`` (which replays the cached segmentation — no API call).
    """
    import json  # noqa: PLC0415

    from playbook_engine.agent_segmenter import (  # noqa: PLC0415
        AGENT_SEGMENTER_MODEL,
        segment_payload_key,
    )
    from playbook_engine.llm_segmenter_batch import (  # noqa: PLC0415
        SegmentationVerdictCache,
        _seg_node_from_dict,
    )
    from playbook_engine.segmentation_grounding import Block  # noqa: PLC0415
    from playbook_engine.segmentation_qa import SegmentationQAError, run_gates  # noqa: PLC0415

    out_resolved = out_dir.resolve()
    if not out_resolved.is_dir():
        click.secho(
            f"ERROR: {out_resolved} does not exist — is the path right? "
            "(the segment/ subdirectory is created automatically on first "
            "write, so this only rejects a missing OUT_DIR itself, not a "
            "fresh pre-seeded one)",
            fg="red",
            err=True,
        )
        raise SystemExit(1)
    cache_path = out_resolved / "segment" / "cache.jsonl"

    # Pending payloads carry the block stream + allowed taxonomy ids for each
    # queued document, keyed by canonical_text content hash — exactly what the
    # QA gates need. Gate BEFORE caching (a bad partition that reaches the
    # cache wedges the document: `segment` reports it cached while every
    # `mine`/`judge` round quarantines it, and nothing ever re-queues it).
    pending_payloads: dict[str, dict[str, Any]] = {}
    seg_pending_path = out_resolved / "segment" / "pending.jsonl"
    if seg_pending_path.is_file():
        for pline in seg_pending_path.read_text(encoding="utf-8").splitlines():
            pline = pline.strip()
            if not pline:
                continue
            try:
                pitem = json.loads(pline)
            except json.JSONDecodeError:
                continue
            if (
                isinstance(pitem, dict)
                and "key" in pitem
                and isinstance(pitem.get("payload"), dict)
            ):
                pending_payloads[pitem["key"]] = pitem["payload"]

    raw_lines = verdicts_path.read_text(encoding="utf-8").splitlines()
    records: list[tuple[str, list[Any]]] = []
    ungated = 0
    for lineno, line in enumerate(raw_lines, start=1):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as exc:
            click.secho(f"ERROR: line {lineno}: invalid JSON: {exc}", fg="red", err=True)
            raise SystemExit(1) from exc
        if not isinstance(rec, dict) or "canonical_text" not in rec or "nodes" not in rec:
            click.secho(
                f"ERROR: line {lineno}: expected {{'canonical_text', 'nodes'}}", fg="red", err=True
            )
            raise SystemExit(1)
        try:
            nodes = [_seg_node_from_dict(n) for n in rec["nodes"]]
        except (KeyError, TypeError) as exc:
            click.secho(f"ERROR: line {lineno}: malformed SegNode: {exc}", fg="red", err=True)
            raise SystemExit(1) from exc

        payload = pending_payloads.get(segment_payload_key(rec["canonical_text"]))
        if payload is None:
            ungated += 1
        else:
            blocks = [
                Block(
                    block_id=b["block_id"],
                    page=b["page"],
                    char_span=(b["char_span"][0], b["char_span"][1]),
                    text=b["text"],
                )
                for b in payload.get("blocks", [])
            ]
            try:
                run_gates(
                    rec["canonical_text"],
                    blocks,
                    nodes,
                    taxonomy_ids=list(payload.get("taxonomy_ids", [])),
                    document_id=str(payload.get("document_id") or "doc"),
                )
            except SegmentationQAError as exc:
                click.secho(
                    f"ERROR: line {lineno} "
                    f"({payload.get('document_id', '?')}): segmentation fails QA gates: {exc}",
                    fg="red",
                    err=True,
                )
                raise SystemExit(1) from exc
        records.append((rec["canonical_text"], nodes))

    if ungated:
        click.secho(
            f"WARN: {ungated} segmentation(s) had no matching item in "
            f"{seg_pending_path} — cached without QA gating (re-apply of an "
            "already-cached document, or a stale queue)",
            fg="yellow",
            err=True,
        )

    cache = SegmentationVerdictCache(cache_path)
    for canonical_text, nodes in records:
        cache.put(canonical_text, nodes, model=AGENT_SEGMENTER_MODEL)

    click.secho(f"OK  loaded {len(records)} segmentation(s) into {cache_path}", fg="green")


@cli.command(name="induce-taxonomy")
@click.argument("corpus_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option(
    "--out",
    "out_path",
    type=click.Path(path_type=Path),
    default=None,
    help="Write candidate taxonomy YAML here (default: stdout).",
)
@click.option(
    "--representation-threshold",
    "representation_threshold",
    type=float,
    default=None,
    help=(
        "Minimum fraction of documents a cluster must appear in to receive "
        "active/custom status (default: 0.20).  Only meaningful for corpora "
        "of five or more documents."
    ),
)
def induce_taxonomy_cmd(
    corpus_dir: Path,
    out_path: Path | None,
    representation_threshold: float | None,
) -> None:
    """Induce a candidate taxonomy from a corpus of agreement documents.

    Ingests all agreements in CORPUS_DIR (one sub-directory per agreement,
    using the highest-versioned file per agreement), clusters their clause headings,
    and emits a taxonomy YAML in OPF spec/taxonomy/ format ready for attorney
    review.  The output is loadable by ``playbook mine --config``.

    Clause headings are mapped to CUAD v1 categories automatically (built-in).
    Unmapped headings that appear in enough documents receive status: custom.

    Common workflow::

        playbook induce-taxonomy corpus/ --out candidate-taxonomy.yaml

    Edit the output to promote/demote entries, then pass it as the taxonomy
    in your config YAML.
    """
    from playbook_engine.clause_tree import ClauseTree
    from playbook_engine.induction_version_selector import (
        VersionCandidate,
        select_representative_version,
    )
    from playbook_engine.pipeline import _discover_versions, _ingest_file
    from playbook_engine.taxonomy_inductor import (
        REPRESENTATION_THRESHOLD,
        emit_taxonomy_yaml,
        induce_taxonomy,
    )

    # Discover and ingest all documents in the corpus
    threshold = (
        representation_threshold
        if representation_threshold is not None
        else REPRESENTATION_THRESHOLD
    )

    trees: list[ClauseTree] = []
    corpus_resolved = corpus_dir.resolve()
    doc_dirs = sorted(
        d for d in corpus_resolved.iterdir() if d.is_dir() and not d.name.startswith(".")
    )
    for doc_dir in doc_dirs:
        # Same discovery pipeline.py's ingest/compile path uses (issue #58) —
        # a taxonomy induced from a different file set than the one later
        # mined/compiled silently drifts. Order isn't load-bearing here:
        # select_representative_version accepts candidates in any order.
        version_files = _discover_versions(doc_dir)
        if not version_files:
            continue
        doc_id = doc_dir.name

        # Ingest EVERY version file for this agreement (not just the
        # filename-highest one) so the representative version can be
        # selected on content — signed-copy detection and the
        # edit-distance chain — rather than filename sort (issue #169).
        candidates: list[VersionCandidate] = []
        for version_path in version_files:
            version = version_path.stem
            try:
                tree = _ingest_file(version_path, doc_id, version)
            except Exception as exc:  # noqa: BLE001
                click.secho(
                    f"  WARN could not ingest {version_path.name}: {exc}", fg="yellow", err=True
                )
                continue
            candidates.append(VersionCandidate(path=version_path, tree=tree))

        if not candidates:
            continue

        selected = select_representative_version(candidates)
        if len(candidates) > 1:
            click.echo(
                f"  {doc_id}: representative version = {selected.path.name} "
                f"(basis={selected.basis})",
                err=True,
            )
        trees.append(selected.tree)

    if not trees:
        click.secho("ERROR: no documents could be ingested from the corpus.", fg="red", err=True)
        raise SystemExit(1)

    click.echo(f"Ingested {len(trees)} document(s).", err=True)

    kwargs = {"representation_threshold": threshold}
    result = induce_taxonomy(trees, **kwargs)

    click.echo(
        f"Induced {len(result.induced_entries)} candidate entries "
        f"({sum(1 for ie in result.induced_entries if ie.entry.status == 'active')} active, "
        f"{sum(1 for ie in result.induced_entries if ie.entry.status == 'custom')} custom, "
        f"{sum(1 for ie in result.induced_entries if ie.entry.status == 'inactive')} inactive).",
        err=True,
    )

    if out_path:
        emit_taxonomy_yaml(result, out_path.resolve())
        click.secho(f"OK  {out_path}", fg="green")
    else:
        import yaml as _yaml  # noqa: PLC0415

        entries_data = []
        for ie in result.induced_entries:
            e = ie.entry
            entries_data.append(
                {
                    "id": e.id,
                    "label": e.label,
                    "status": e.status,
                    "cuad_origin": e.cuad_origin,
                    "description": e.description,
                    **({"examples": [ex.to_dict() for ex in ie.examples]} if ie.examples else {}),
                }
            )
        click.echo(
            _yaml.dump(
                {"source": "induced", "entries": entries_data},
                allow_unicode=True,
                default_flow_style=False,
                sort_keys=False,
            )
        )


@cli.command(name="scorecard")
@click.argument("out_dir", type=click.Path(file_okay=False, path_type=Path))
@click.option(
    "--compare",
    "compare_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="A previous scorecard.json to compare against (prints baseline, current and delta).",
)
@click.option(
    "--out",
    "scorecard_path",
    type=click.Path(path_type=Path),
    default=None,
    help="Write the scorecard JSON to this file (default: <out_dir>/scorecard.json).",
)
def scorecard_cmd(out_dir: Path, compare_path: Path | None, scorecard_path: Path | None) -> None:
    """Write a counts-only scorecard of a derivation OUT_DIR and print it.

    Integers, ratios and closed-vocabulary labels only — no clause text,
    party name, file name, document id or clause type — so a maintainer can
    post the result of a private evaluation corpus in public. Covers corpus
    size, template-standard coverage, classification by basis (with a
    paper-side parity diagnostic), precedent, openings, dropped
    observations, digest size and pending agent queues. A field the
    artifact does not carry yet is null. With --compare, prints the delta
    against an earlier scorecard.json.
    """
    import json as _json  # noqa: PLC0415

    from playbook_engine.scorecard import (  # noqa: PLC0415
        SCORECARD_FILENAME,
        build_scorecard,
        render_table,
        write_scorecard,
    )

    resolved = out_dir.resolve()
    if not resolved.is_dir():
        click.secho(f"ERROR: {resolved} is not a directory", fg="red", err=True)
        raise SystemExit(1)
    baseline = None
    if compare_path is not None:
        try:
            baseline = _json.loads(compare_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            click.secho(f"ERROR: could not read {compare_path}: {exc}", fg="red", err=True)
            raise SystemExit(1) from exc
        if not isinstance(baseline, dict):
            click.secho(f"ERROR: {compare_path} is not a scorecard object", fg="red", err=True)
            raise SystemExit(1)

    card = build_scorecard(resolved)
    dest = scorecard_path.resolve() if scorecard_path else resolved / SCORECARD_FILENAME
    write_scorecard(card, dest)
    click.echo(render_table(card, baseline))
    click.secho(f"OK  {dest}", fg="green")


@cli.group(name="view")
def view_group() -> None:
    """Render the human-readable OPF bundle."""


@view_group.command(name="bundle")
@click.argument("out_dir", type=click.Path(file_okay=False, path_type=Path))
@click.option(
    "--out",
    "out_file",
    type=click.Path(path_type=Path),
    default=None,
    help=(
        "Write the HTML to this file (default: <out_dir>/playbook.opf.html). "
        "Also prints the path on success."
    ),
)
def view_bundle_cmd(out_dir: Path, out_file: Path | None) -> None:
    """Render the single-file OPF bundle: OUT_DIR/playbook.opf.html.

    The full human document plus a digest summary, with the CANONICAL OPF
    JSON and the digest embedded verbatim in <script type="application/json">
    blocks (ids: opf-canonical, opf-digest). The bare playbook.opf.json
    remains the canonical artifact; the bundle contains it, never replaces
    it — a consumer extracts the JSON block and verifies
    identity.content_hash. The bundle stays alias-only. This is NOT a
    guarantee of pseudonymization — known_entities matching is best-effort,
    so run the mandatory residue check (see the playbook-from-corpus skill)
    before treating the bundle as shareable.
    """
    import json as _json  # noqa: PLC0415

    from playbook_engine.document_renderer import render_bundle_html  # noqa: PLC0415

    resolved = out_dir.resolve()
    dest = out_file.resolve() if out_file else resolved / "playbook.opf.html"

    opf_path = resolved / "playbook.opf.json"
    if opf_path.exists():
        try:
            doc = _json.loads(opf_path.read_text(encoding="utf-8"))
        except ValueError as exc:  # includes json.JSONDecodeError
            click.secho(f"ERROR: could not parse {opf_path}: {exc}", fg="red", err=True)
            raise SystemExit(1) from exc
        _refuse_unsupported_opf_version(doc, opf_path)

    try:
        render_bundle_html(resolved, out_file=dest)
    except FileNotFoundError as exc:
        click.secho(f"ERROR: {exc}", fg="red", err=True)
        raise SystemExit(1) from exc

    click.secho(f"OK  {dest}", fg="green")


@cli.group(name="posture")
def posture_group() -> None:
    """Author the OPF Posture via a short GC interview."""


@posture_group.command(name="questions")
def posture_questions_cmd() -> None:
    """List the canonical interview question ids (for --answers-file JSON keys)."""
    for iq in INTERVIEW_QUESTIONS:
        click.echo(f"{iq.q}: {iq.question}")


@posture_group.command(name="interview")
@click.argument("out_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option(
    "--answers-file",
    "answers_file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help=(
        "JSON file of {question_id: answer} pairs (see 'playbook posture questions' "
        "for the canonical ids). When omitted, prompts interactively on the terminal."
    ),
)
@click.option(
    "--base-version",
    "base_version",
    type=click.IntRange(min=1),
    default=None,
    help=(
        "The last known posture.version from a prior playbook this run's "
        "OUT_DIR does NOT itself carry (e.g. a re-derivation into a wiped "
        "or freshly re-derived out-dir). The new posture.version is one more "
        "than max(OUT_DIR's own carried-forward version, this value), so the "
        "governed counter continues forward instead of silently restarting at "
        "1. Omit when recompiling in place — OUT_DIR's own prior posture is "
        "sufficient there."
    ),
)
def posture_interview_cmd(
    out_dir: Path, answers_file: Path | None, base_version: int | None
) -> None:
    """Run the Posture interview and write a versioned Posture into OUT_DIR/playbook.opf.json.

    Asks the canonical 3-6 question set (OPF-SPEC.md §7), assembles
    the answers deterministically into ``posture.system_prompt``, and writes
    the result into OUT_DIR/playbook.opf.json as a governed, versioned block:
    a re-run against an existing Posture whose answers actually changed
    bumps ``posture.version`` by 1. A byte-identical re-run (same answers as
    the existing Posture) is a no-op — ``posture.version`` is left
    untouched rather than bumped for a revision that never happened.

    A re-derivation into a wiped or freshly re-derived OUT_DIR has no prior
    Posture of its own to bump from — pass ``--base-version`` with the last
    known version from the playbook this run supersedes so the counter
    continues forward instead of restarting at 1.

    Warns (non-blocking) if the assembled Posture softens language around a
    concept a Floor invariant protects — a possible Posture-vs-Floor conflict
    for a human to review.

    OUT_DIR must already contain a playbook.opf.json (from 'playbook mine'
    followed by 'playbook project').
    """
    import datetime  # noqa: PLC0415
    import json  # noqa: PLC0415

    out_dir_resolved = out_dir.resolve()

    if answers_file is not None:
        try:
            raw = json.loads(answers_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            click.secho(f"ERROR: invalid JSON in {answers_file}: {exc}", fg="red", err=True)
            raise SystemExit(1) from exc
        if not isinstance(raw, dict):
            click.secho(
                f"ERROR: {answers_file} must contain a JSON object of {{question_id: answer}}",
                fg="red",
                err=True,
            )
            raise SystemExit(1)
        answers = {str(k): str(v) for k, v in raw.items()}
    else:
        click.echo("Posture interview — press Enter to skip a question.\n")
        answers = {}
        for iq in INTERVIEW_QUESTIONS:
            reply = click.prompt(iq.question, default="", show_default=False)
            if reply.strip():
                answers[iq.q] = reply.strip()

    generated_at = datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")

    try:
        result = apply_posture_interview(
            out_dir_resolved,
            answers,
            generated_at=generated_at,
            base_version=base_version,
        )
    except (FileNotFoundError, PostureError) as exc:
        click.secho(f"ERROR: {exc}", fg="red", err=True)
        raise SystemExit(1) from exc

    if result.changed:
        click.secho(f"OK  posture.version={result.version} written to {result.path}", fg="green")
    else:
        click.secho(
            f"OK  posture.version={result.version} unchanged (no-op — answers matched "
            f"the existing Posture) — {result.path}",
            fg="green",
        )
    for warning in result.warnings:
        click.secho(f"WARN  {warning}", fg="yellow")


@cli.group(name="floor")
def floor_group() -> None:
    """Propose Floor candidates for legal review."""


@floor_group.command(name="propose")
@click.argument("out_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help=(
        "Engine config YAML — supplies the taxonomy, to exclude reversals "
        "classified under a taxonomy entry curated 'structural: true'. "
        "Optional; omit to skip structural exclusion."
    ),
)
@click.option(
    "--min-deals",
    "min_deals",
    type=int,
    default=2,
    help=(
        "Minimum number of distinct documents that must cite a reversal "
        "before it becomes a candidate; a single-document "
        "reversal is a plausible fluke, not a corroborated pattern."
    ),
)
def floor_propose_cmd(out_dir: Path, config_path: Path | None, min_deals: int) -> None:
    """Derive Floor candidates from reversals + the Posture interview's Q4 answer.

    Reads OUT_DIR/observations.jsonl (every ``outcome: proposed_then_reversed``
    observation is a candidate hard line — OPF-SPEC.md §3.7 rule 4)
    and, if a Posture interview has been run, OUT_DIR/playbook.opf.json's
    ``posture.generation.interview`` Q4 ("sacred_clauses") answer. Writes
    OUT_DIR/floor.candidates.json and prints a summary table.

    This is a REVIEW ARTIFACT for the legal owner — it never writes to the
    OPF ``floor`` section, and never auto-promotes a candidate into
    ``floor.invariants``. Accepting a candidate is a human act: the legal
    owner reads the table and records each hard line they accept with
    ``playbook floor sign``, using the candidate's statement or wording of
    their own. Do not hand-edit ``floor.invariants``.
    """
    import json  # noqa: PLC0415

    structural_ids: frozenset[str] = frozenset()
    if config_path is not None:
        try:
            cfg = load_config(config_path)
        except ConfigError as exc:
            click.secho(f"Config error: {exc}", fg="red", err=True)
            raise SystemExit(1) from exc
        try:
            taxonomy = load_taxonomy(cfg.taxonomy_path)
        except TaxonomyError as exc:
            click.secho(f"Taxonomy error: {exc}", fg="red", err=True)
            raise SystemExit(1) from exc
        structural_ids = frozenset(e.id for e in taxonomy.entries if e.structural)

    out_dir_resolved = out_dir.resolve()
    result_path = write_floor_candidates(
        out_dir_resolved, structural_ids=structural_ids, min_deals=min_deals
    )
    written = json.loads(result_path.read_text(encoding="utf-8"))
    candidates = written["candidates"]
    # issue #105: a Q4/Q5 contradictory-interview warning (loud, non-blocking
    # — see floor_candidates.q4_q5_contradictions) — printed regardless of
    # whether there are any candidates at all.
    for warning in written.get("warnings", []):
        click.secho(f"WARN  {warning}", fg="yellow")

    if not candidates:
        click.secho(f"OK  {result_path} (0 candidates)", fg="green")
        return

    click.secho(f"OK  {result_path} ({len(candidates)} candidate(s))", fg="green")
    click.echo("")
    click.echo(f"{'id':<10} {'source':<14} {'citations':<10} statement")
    click.echo("-" * 70)
    for c in candidates:
        click.echo(f"{c['id']:<10} {c['source']:<14} {len(c['citations']):<10} {c['statement']}")


@floor_group.command(name="sign")
@click.argument("out_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option(
    "--statement",
    required=True,
    help="The invariant, verbatim — recorded exactly as given, never templated.",
)
@click.option(
    "--signed-by",
    "signed_by",
    required=True,
    help=(
        "Name of the human legal owner signing this hard line. Required — "
        "recorded as a structural attribution field, not just free-form "
        "rationale text, so a later review can tell who actually signed it."
    ),
)
@click.option(
    "--id",
    "invariant_id",
    default=None,
    help="floor.invariants[].id to use (default: a kebab-case slug of --statement).",
)
@click.option(
    "--clause",
    "taxonomy_id",
    default=None,
    help=(
        "Clause-taxonomy id this invariant is about — validated against "
        "--config's taxonomy; on a mismatch, prints the valid ids."
    ),
)
@click.option(
    "--rationale",
    default=None,
    help=(
        "Legal justification only — never the signer's name or a sign-off "
        "date; --signed-by already records that structurally, and rationale "
        "ships verbatim into every consumer's model-facing review prompt. "
        "(default: 'Hand-authored via `playbook floor sign`.')."
    ),
)
@click.option(
    "--requires-presence/--no-requires-presence",
    "required_presence",
    default=None,
    help=(
        "Whether the clause's absence or deletion is a hard rejection. Recorded "
        "as x_required_presence for the hard-rule manifest; unstated means the "
        "manifest does not demand presence."
    ),
)
@click.option(
    "--condition",
    "condition",
    default=None,
    help=(
        "A machine-evaluable predicate for the hard-rule manifest, as JSON: "
        '{"type": "required_phrases", "phrases": [...]}, '
        '{"type": "numeric_bound", "pattern": "<one capture group>", "min"/"max": n} or '
        '{"type": "cross_reference", "clause_id": "..."}; or the word judged. '
        "Unstated means judged."
    ),
)
@click.option(
    "--proof",
    "permissible_proof",
    multiple=True,
    help="What a reviewed document may show to satisfy the rule (repeatable).",
)
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Engine config YAML — required only when --clause is given, to resolve the taxonomy.",
)
def floor_sign_cmd(
    out_dir: Path,
    statement: str,
    signed_by: str,
    invariant_id: str | None,
    taxonomy_id: str | None,
    rationale: str | None,
    required_presence: bool | None,
    condition: str | None,
    permissible_proof: tuple[str, ...],
    config_path: Path | None,
) -> None:
    """Record a verbatim, hand-authored Floor invariant.

    Unlike the Posture interview's Q4 templating ("Do not concede on
    {item}.") or an accepted 'floor propose' candidate's compiler-drafted
    wording, STATEMENT is written into OUT_DIR/playbook.opf.json's
    ``floor.invariants`` exactly as given — the path for a conditional hard
    line ("limitation of liability, if present, must not be unilateral in
    the counterparty's favor") that either of those templates would
    otherwise garble.

    --signed-by is required: the human legal owner's name, recorded as a
    structural attribution field alongside the statement, not folded into
    free-form --rationale text where nothing would distinguish a genuine
    sign-off from an agent-typed one.

    Idempotent: signing the same statement under the same id twice is a
    no-op (the ORIGINAL --signed-by / --rationale / --clause are kept —
    a rerun's values, if different, are ignored, not merged in). Signing an
    id that already carries a DIFFERENT statement is refused — this command
    never overwrites an existing invariant; edit or remove the conflicting
    one first, or choose a different --id.

    OUT_DIR must already contain a playbook.opf.json (from 'playbook mine'
    followed by 'playbook project').
    """
    out_dir_resolved = out_dir.resolve()
    opf_path = out_dir_resolved / "playbook.opf.json"
    if not opf_path.exists():
        click.secho(
            f"ERROR: {opf_path} not found — run 'playbook mine' and 'playbook project' first.",
            fg="red",
            err=True,
        )
        raise SystemExit(1)

    if taxonomy_id is not None:
        if config_path is None:
            click.secho(
                "ERROR: --clause requires --config, to resolve the taxonomy it is "
                "validated against.",
                fg="red",
                err=True,
            )
            raise SystemExit(1)
        try:
            cfg = load_config(config_path)
        except ConfigError as exc:
            click.secho(f"Config error: {exc}", fg="red", err=True)
            raise SystemExit(1) from exc
        try:
            taxonomy = load_taxonomy(cfg.taxonomy_path)
        except TaxonomyError as exc:
            click.secho(f"Taxonomy error: {exc}", fg="red", err=True)
            raise SystemExit(1) from exc
        valid_ids = sorted(e.id for e in taxonomy.classifier_entries())
        if taxonomy_id not in valid_ids:
            click.secho(
                f"ERROR: unknown --clause {taxonomy_id!r}. Valid taxonomy ids:",
                fg="red",
                err=True,
            )
            for vid in valid_ids:
                click.echo(f"  {vid}", err=True)
            raise SystemExit(1)

    import datetime  # noqa: PLC0415
    import json  # noqa: PLC0415

    parsed_condition: Any = None
    if condition is not None:
        try:
            parsed_condition = condition if condition.strip() == "judged" else json.loads(condition)
        except json.JSONDecodeError as exc:
            click.secho(f"ERROR: --condition is not valid JSON: {exc}", fg="red", err=True)
            raise SystemExit(1) from exc
        why = validate_condition(parsed_condition)
        if why:
            click.secho(f"ERROR: --condition is not valid: {why}", fg="red", err=True)
            raise SystemExit(1)

    doc = load_opf_file(opf_path)
    existing_invariants = (doc.get("floor") or {}).get("invariants") or []
    signed_at = datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")
    try:
        invariants = sign_floor_invariant(
            statement,
            invariant_id=invariant_id,
            taxonomy_id=taxonomy_id,
            rationale=rationale,
            signed_by=signed_by,
            signed_at=signed_at,
            existing_invariants=existing_invariants,
            required_presence=required_presence,
            condition=parsed_condition,
            permissible_proof=list(permissible_proof),
        )
    except FloorCandidateError as exc:
        click.secho(f"ERROR: {exc}", fg="red", err=True)
        raise SystemExit(1) from exc

    inv_id = sign_invariant_id(statement, invariant_id)
    changed = invariants != existing_invariants
    if changed:
        floor_section = dict(doc.get("floor") or {})
        floor_section["invariants"] = invariants
        doc["floor"] = floor_section
        # The manifest and the dossiers read the Floor (issue #228).
        refresh_derived_sections(doc)
        # MUST recompute — floor is part of identity.content_hash (see
        # posture.apply_posture_interview, the reference pattern this
        # mirrors); a stale hash silently misdescribes the document to any
        # consumer that trusts it.
        if "identity" in doc:
            doc["identity"]["content_hash"] = content_hash(doc)
            doc["identity"]["section_digests"] = compute_section_digests(doc)
        write_playbook(doc, opf_path)
        click.secho(f"OK  floor invariant {inv_id!r} signed to {opf_path}", fg="green")
    else:
        click.secho(
            f"OK  floor invariant {inv_id!r} already signed (no-op) — {opf_path}", fg="green"
        )

    # Decision (issue #103 Reviewer gate): YES, duplicate validator.py:526's
    # non-blocking SHOULD-warn here too, even though `playbook validate` will
    # catch it later. A hand-signed invariant is exactly the moment a GC is
    # staring at both the new statement and the existing Posture — surfacing
    # a conflict immediately, in the same command, is cheaper to act on than
    # waiting for a separate validate pass to report it after the fact.
    posture_prompt = ((doc.get("posture") or {}).get("system_prompt")) or ""
    for warning in check_posture_floor_conflict(posture_prompt, invariants):
        click.secho(f"WARN  {warning}", fg="yellow")
