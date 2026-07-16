"""Top-level CLI entry point."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import click
import yaml

from playbook_engine import __version__
from playbook_engine.aar import build_after_action_report, write_after_action_report
from playbook_engine.config import ConfigError, load_config
from playbook_engine.corpus_linter import lint_corpus
from playbook_engine.floor_candidates import write_floor_candidates
from playbook_engine.inspection_report import build_inspection_report, write_inspection_report
from playbook_engine.natural_sort import natural_sort_key as _natural_sort_key
from playbook_engine.pipeline import PipelineError, compile_corpus, mine_corpus, project_playbook
from playbook_engine.playbook_assembler import AssemblyError
from playbook_engine.posture import INTERVIEW_QUESTIONS, PostureError, apply_posture_interview
from playbook_engine.segmentation_qa import SegmentationQAError
from playbook_engine.taxonomy import Taxonomy, TaxonomyError, load_taxonomy, merge_taxonomy
from playbook_engine.validator import SUPPORTED_OPF_VERSIONS, load_opf_file, validate_document

# e.g. "0.1, 0.2" — engine version and OPF version drift independently
# (engine 0.1.0 shipped OPF 0.2's predecessor), so `--version` reports both
# to keep bug reports unambiguous about which OPF schema a given engine
# build validates against (issue #176).
_OPF_VERSIONS_STR = ", ".join(sorted(SUPPORTED_OPF_VERSIONS))


def _llm_segmentation_kwargs(
    cfg: Any,
    taxonomy: Taxonomy,
    out_dir: Path,
    echo: Callable[[str], None],
    *,
    stats: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Build the LLM-segmentation kwargs shared by ``mine``, ``compile``, and ``judge``.

    LLM-first segmentation is config-gated (``segmentation.llm``) so existing
    configs/fixtures with no ``segmentation:`` section are byte-for-byte
    unchanged — the deterministic segmenter remains the default.  Every command
    that segments a corpus MUST build these kwargs the same way: if ``mine``
    segments via the LLM but ``compile``/``judge`` fall back to the deterministic
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
               ``mine``/``compile`` always pass) disables collection.

    Returns an empty dict when ``segmentation.llm`` is off (deterministic path)
    — ``stats`` is left untouched (stays at caller-supplied zero) in that case.

    Raises:
        ConfigError: ``segmentation.llm`` is on but no Anthropic credentials
                     resolve from the environment.
    """
    kwargs: dict[str, Any] = {}
    if not cfg.segmentation.llm:
        return kwargs

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
        kwargs["llm_segment_fn"] = StoreBackedSegmentFn(
            pending=PendingQueue(seg_dir / "pending.jsonl")
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
        # compile round, independent of the no_cache value store-backed
        # judges force for the verdict-cache layers (issue #132).
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


def _verdict_store_kwargs(out_dir: Path, echo: Callable[[str], None]) -> dict[str, Any]:
    """Wire store-backed judges when a verdict store exists at ``out_dir/judge/verdicts.jsonl``.

    Shared by ``mine`` and ``compile`` (issue #102) — before this, ``compile``
    never checked for a verdict store at all, so it always ran the stub
    judges even over an ``out_dir`` where a ``playbook judge`` /
    ``judge-apply`` round had already populated real verdicts, silently
    overwriting the judged ``observations.jsonl`` with stub-mode sentinels.

    The verdict-cache layer is bypassed (``no_cache=True``) to prevent stale
    sentinels cached under the stub judges from persisting across rounds —
    the ``VerdictStore`` is the authoritative source for judge verdicts.

    Returns an empty dict when no verdict store exists (the stub judges
    remain the default, same as before).
    """
    verdicts_path = out_dir / "judge" / "verdicts.jsonl"
    if not verdicts_path.exists():
        return {}

    from playbook_engine.agent_judge import (  # noqa: PLC0415
        PendingQueue,
        StoreBackedClassificationJudge,
        StoreBackedDeviationJudge,
        StoreBackedProvenanceJudge,
        StoreBackedScopeJudge,
        VerdictStore,
    )

    store = VerdictStore(verdicts_path)
    pending = PendingQueue(out_dir / "judge" / "pending.jsonl")
    echo(f"  judge store: {verdicts_path} (store-backed judges active)")
    return {
        "scope_judge": StoreBackedScopeJudge(store=store, pending=pending),
        "classification_judge": StoreBackedClassificationJudge(store=store, pending=pending),
        "deviation_judge": StoreBackedDeviationJudge(store=store, pending=pending),
        "provenance_judge": StoreBackedProvenanceJudge(store=store, pending=pending),
        "no_cache": True,
    }


def _echo_extractor_summary(out_dir: Path, echo: Callable[[str], None]) -> None:
    """Echo how many mined versions used ``docling`` vs. the legacy adapters.

    Reads ``out_dir/corpus_manifest.json`` (already written by
    ``mine_corpus``/``compile_corpus`` by the time this runs) and tallies
    each version's ``version_ingest[].extractor`` value. Mirrors the
    ``segmentation: ...`` echo above so the docling-vs-legacy choice is a
    first-class part of ``mine``/``compile`` output rather than only a
    ``logging.info`` line suppressed by default Python logging config
    (issue #129) — a host run without docling silently extracting scanned
    PDFs with no OCR was otherwise invisible to the operator. Silent no-op
    when the manifest is missing or no version recorded a ``"docling"``/
    ``"legacy"`` extractor (e.g. a purely deterministic-segmentation run,
    where ``extractor`` is the file suffix instead — nothing to summarize).
    """
    import json  # noqa: PLC0415

    manifest_path = out_dir / "corpus_manifest.json"
    if not manifest_path.exists():
        return

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    counts: dict[str, int] = {}
    for doc in manifest:
        for v in doc.get("version_ingest", []):
            ext = v.get("extractor")
            if ext in ("docling", "legacy"):
                counts[ext] = counts.get(ext, 0) + 1

    if not counts:
        return

    summary = ", ".join(f"{ext}={n}" for ext, n in sorted(counts.items()))
    echo(f"  extraction: {summary}")


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

    result = validate_document(doc)

    for err in result.errors:
        color = "red" if err.blocking else "yellow"
        click.secho(str(err), fg=color, err=not err.blocking)

    if result.ok:
        click.secho(f"OK  {file}", fg="green")
    else:
        n_blocking = sum(1 for e in result.errors if e.blocking)
        click.secho(f"FAIL {file}: {n_blocking} error(s)", fg="red", err=True)
        raise SystemExit(1)


@cli.command(name="render-prompt")
@click.argument("playbook_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--out",
    "out_file",
    type=click.Path(path_type=Path),
    default=None,
    help="Write the rendered prompt to this file (default: stdout).",
)
def render_prompt_cmd(playbook_file: Path, out_file: Path | None) -> None:
    """Compose Evidence+Posture+Floor into a review-ready system prompt (issue #179).

    The reference prompt-pack consumer: pure Markdown a user pastes into any chat
    LLM alongside a contract to review. No API calls, no redline generation — the
    determinism boundary (§5) rendered as instructions.
    """
    from playbook_engine.prompt_renderer import render_prompt

    try:
        doc = load_opf_file(playbook_file)
    except Exception as exc:  # noqa: BLE001
        click.secho(f"ERROR: could not parse {playbook_file}: {exc}", fg="red", err=True)
        raise SystemExit(1) from exc

    rendered = render_prompt(doc)
    if out_file is not None:
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(rendered, encoding="utf-8")
        click.secho(f"wrote {out_file}", fg="green")
    else:
        click.echo(rendered)


@cli.command(name="resolve-citation")
@click.argument("playbook_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--clause",
    "clause_id",
    required=True,
    help="evidence.clauses[].id, e.g. clause.indemnification",
)
@click.option(
    "--obs",
    "obs_index",
    type=int,
    required=True,
    help="Index into that clause's observed_positions.",
)
@click.option(
    "--corpus-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Directory holding the corpus source files.",
)
def resolve_citation_cmd(
    playbook_file: Path, clause_id: str, obs_index: int, corpus_dir: Path
) -> None:
    """Resolve one observation's citation to a hash-verified source file (OPF §4).

    Looks up the cited (document_id, version) in corpus.documents[].version_files,
    finds the file under CORPUS-DIR whose sha256 matches, and prints the path plus
    clause_path/char_span. Exits 1 on hash mismatch or a missing content address —
    the reference implementation consumers copy (issue #185).
    """
    from playbook_engine.citation_resolver import CitationResolutionError, resolve_citation

    try:
        doc = load_opf_file(playbook_file)
    except Exception as exc:  # noqa: BLE001
        click.secho(f"ERROR: could not parse {playbook_file}: {exc}", fg="red", err=True)
        raise SystemExit(1) from exc

    try:
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


@cli.command(name="publish")
@click.argument("playbook_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--out",
    "out_file",
    type=click.Path(path_type=Path),
    required=True,
    help="Write the party-anonymous public playbook to this path.",
)
@click.option(
    "--party-label",
    default="the company",
    show_default=True,
    help="Replaces perspective.party.",
)
@click.option(
    "--counterparty-label",
    default="the counterparty",
    show_default=True,
    help="Replaces every GENERIC free-text '(the) counterparty' mention. "
    "Per-deal numbered aliases (e.g. Counterparty-7) are never touched.",
)
@click.option(
    "--keep-dates",
    is_flag=True,
    default=False,
    help="Skip coarsening observed_at to YYYY-Qn.",
)
@click.option(
    "--accept-residue-risk",
    is_flag=True,
    default=False,
    help="Publish even if the independent verify pass flags residual semantic residue.",
)
def publish_cmd(
    playbook_file: Path,
    out_file: Path,
    party_label: str,
    counterparty_label: str,
    keep_dates: bool,
    accept_residue_risk: bool,
) -> None:
    """Produce a party-anonymous public playbook (issue #188).

    Runs the six-step publication transform: party/counterparty role-label
    normalization, DMS-path stripping, date coarsening, a deterministic
    no-known-entity backstop (scans the entity registry's real names against
    every string in the output — a hit fails loud, unconditionally), the
    full-surface semantic-residue judgment + independent verify pass, and an
    identity recompute (the public doc is a different artifact; its
    identity.supersedes names the private doc's content_hash).

    No LLM is wired here — this defaults to stub judges (basis="stub"),
    same as every other zero-configuration path in this engine. Wiring a
    real judge is a separate concern (see playbook_engine/export_profile.py).
    """
    import datetime  # noqa: PLC0415
    from collections.abc import Sequence  # noqa: PLC0415

    from playbook_engine.entity_registry import DEFAULT_REGISTRY_PATH, EntityRegistry
    from playbook_engine.export_profile import RedactionFinding, TextSample, VerifyFinding
    from playbook_engine.playbook_assembler import write_playbook
    from playbook_engine.publisher import PublishError, publish_playbook

    try:
        doc = load_opf_file(playbook_file)
    except Exception as exc:  # noqa: BLE001
        click.secho(f"ERROR: could not parse {playbook_file}: {exc}", fg="red", err=True)
        raise SystemExit(1) from exc

    # Real names for the deterministic backstop (step 4): the entity
    # registry's alias -> canonical-name map IS the held-out real-name list
    # (write_holdout_map persists this same data to a sidecar) — an absent
    # registry (no corpus ever mined on this machine) yields an empty list,
    # making the backstop a no-op rather than a crash.
    registry = EntityRegistry.load(DEFAULT_REGISTRY_PATH)
    known_entity_names = list(registry.alias_map().values())

    class _StubRedactionJudge:
        """No LLM configured — an honest basis='stub' no-residue verdict."""

        def evaluate_batch(self, samples: Sequence[TextSample]) -> list[RedactionFinding]:
            return [
                RedactionFinding(
                    path=s.path,
                    has_residue=False,
                    rationale="No LLM configured (stub mode).",
                    basis="stub",
                )
                for s in samples
            ]

    class _StubVerifyJudge:
        """No LLM configured — an honest basis='stub' no-leak verdict."""

        def evaluate_batch(self, samples: Sequence[TextSample]) -> list[VerifyFinding]:
            return [
                VerifyFinding(
                    path=s.path,
                    leaked=False,
                    rationale="No LLM configured (stub mode).",
                    basis="stub",
                )
                for s in samples
            ]

    try:
        report = publish_playbook(
            doc,
            redaction_judge=_StubRedactionJudge(),
            verify_judge=_StubVerifyJudge(),
            known_entity_names=known_entity_names,
            published_at=datetime.datetime.now(datetime.UTC).isoformat(),
            party_label=party_label,
            counterparty_label=counterparty_label,
            keep_dates=keep_dates,
            accept_residue_risk=accept_residue_risk,
        )
    except PublishError as exc:
        click.secho(f"ERROR: {exc}", fg="red", err=True)
        raise SystemExit(1) from exc

    write_playbook(report.doc, out_file)
    click.secho(f"wrote {out_file}", fg="green")
    if report.leaked:
        click.secho(
            f"WARNING: {len(report.leaked)} residue finding(s) published anyway "
            "(--accept-residue-risk):",
            fg="yellow",
        )
        for finding in report.leaked:
            click.echo(f"  {finding.path}: {finding.rationale}")

    # Proper-noun residue report (issue #211): the reviewer's checkable list of
    # every name-shaped string surviving in the output. Advisory — written
    # beside the playbook for human/GC classification before publication.
    import json  # noqa: PLC0415

    residue_path = out_file.parent / "residue_report.json"
    residue_path.write_text(
        json.dumps(
            {"proper_noun_findings": [f.to_dict() for f in report.proper_noun_findings]},
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    n = len(report.proper_noun_findings)
    if n:
        click.secho(
            f"residue report: {n} proper-noun-like string(s) remain — review "
            f"{residue_path} before publishing (confirm none is a counterparty).",
            fg="yellow",
        )
        for pn in report.proper_noun_findings[:10]:
            click.echo(f"  {pn.text}  (×{pn.count})")
        if n > 10:
            click.echo(f"  … and {n - 10} more in {residue_path.name}")
    else:
        click.secho(
            f"residue report: no proper-noun-like strings remain (see {residue_path}).",
            fg="green",
        )


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
            }
            for e in taxonomy.entries
        ],
    }
    dest.write_text(yaml.dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")


@cli.command(name="compile")
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
    "--no-cache",
    "no_cache",
    is_flag=True,
    default=False,
    help="Disable the stage cache and force a full recompute.",
)
@click.option(
    "--stop-after",
    "stop_after",
    type=click.Choice(["intermediates"]),
    default=None,
    help=(
        "Stop the pipeline after the named checkpoint and skip later stages. "
        "'intermediates' stops after L1–L4 (scope.json, observations.jsonl, "
        "corpus_manifest.json, trail/) — playbook.opf.json is NOT written."
    ),
)
@click.option(
    "--entity-registry",
    "entity_registry_path",
    type=click.Path(path_type=Path),
    default=None,
    help=(
        "Path to the born-safe entity registry (alias->real-name map). Defaults "
        "to ~/.cache/playbook-engine/entity_registry.json. Point it into your "
        "gitignored output dir to keep all sensitive real-name data in one place. "
        "Only relevant when provenance.known_entities is set."
    ),
)
def compile_playbook(
    corpus_dir: Path,
    config_path: Path,
    out_path: Path | None,
    no_cache: bool,
    stop_after: str | None,
    entity_registry_path: Path | None,
) -> None:
    """Compile CORPUS_DIR into a playbook.opf.json.

    Runs the full L1→L5 pipeline: ingest, scope gate, structure, classify,
    mine deltas, and compile.  LLM stages use conservative stub judges when
    no real LLM is configured — unless OUT_DIR already has a verdict store at
    judge/verdicts.jsonl (from a prior ``playbook judge`` + ``judge-apply``
    round against this out dir), in which case the same store-backed judges
    ``mine`` uses are wired in instead, so a real judgment round is never
    silently overwritten by a stub recompute.

    Pass --no-cache to disable the content-addressed stage cache and force a
    full recompute even if intermediates already exist.

    Pass --stop-after intermediates to stop after the L1–L4 intermediates are
    written, without proceeding to L5 playbook compilation.
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

    # Segment the same way ``mine``/``judge`` do — otherwise a config with
    # ``segmentation.llm`` on would compile a deterministically-segmented
    # playbook that doesn't line up with a mined observation store.
    try:
        seg_kwargs = _llm_segmentation_kwargs(cfg, taxonomy, out_dir, click.echo)
    except ConfigError as exc:
        click.secho(f"Config error: {exc}", fg="red", err=True)
        raise SystemExit(1) from exc

    # If a verdict store exists at out_dir/judge/verdicts.jsonl (populated by
    # ``playbook judge`` + ``judge-apply``), wire in the same store-backed
    # judges ``mine`` uses — otherwise ``compile`` would run this out_dir's
    # documents back through the stub judges and silently overwrite the
    # already-judged observations.jsonl with fabricated stub verdicts
    # (issue #102). ``_verdict_store_kwargs`` forces no_cache=True when a
    # store is wired, which deliberately overrides the --no-cache flag's
    # default (``no_cache``) below.
    verdict_kwargs = _verdict_store_kwargs(out_dir, click.echo)
    compile_kwargs: dict[str, Any] = {"no_cache": no_cache, **seg_kwargs, **verdict_kwargs}

    try:
        result = compile_corpus(
            corpus_dir=corpus_dir.resolve(),
            config=cfg,
            taxonomy=taxonomy,
            out_dir=out_dir,
            stop_after=stop_after,
            progress=click.echo,
            entity_registry_path=(entity_registry_path.resolve() if entity_registry_path else None),
            **compile_kwargs,
        )
    except (PipelineError, AssemblyError) as exc:
        click.secho(f"ERROR: {exc}", fg="red", err=True)
        raise SystemExit(1) from exc

    _echo_extractor_summary(out_dir, click.echo)

    if stop_after is not None:
        click.secho(
            f"OK  stopped after {result['stopped_after']} (no playbook compiled)",
            fg="green",
        )
    else:
        click.secho(f"OK  {out_dir / 'playbook.opf.json'}", fg="green")


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
def mine_cmd(
    corpus_dir: Path,
    config_path: Path,
    out_path: Path | None,
    entity_registry_path: Path | None,
) -> None:
    """Mine CORPUS_DIR and write the observation store (L1–L4).

    Runs ingest, scope gate, classification, alignment, and deviation
    assessment for every agreement in CORPUS_DIR and writes:

    \b
      observations.jsonl    — per-clause observation store
      corpus_manifest.json  — per-document metadata
      scope.json            — scope-gate decisions
      trail/<doc_id>.json   — version-order and provenance signals
      normalized/           — segmented clause trees

    Does NOT write playbook.opf.json.  Run ``playbook project`` afterwards
    to compile the playbook from the store, or use ``playbook compile`` for
    the combined end-to-end flow.
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

    # If a verdict store exists (populated by ``playbook judge-apply``), wire in
    # the store-backed judges so the mining step replays stored verdicts rather
    # than generating new needs_review sentinels.
    mine_kwargs: dict[str, Any] = _verdict_store_kwargs(out_dir, click.echo)
    try:
        mine_kwargs.update(_llm_segmentation_kwargs(cfg, taxonomy, out_dir, click.echo))
    except ConfigError as exc:
        click.secho(f"Config error: {exc}", fg="red", err=True)
        raise SystemExit(1) from exc

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
    click.secho(f"OK  {out_dir / 'observations.jsonl'}", fg="green")


@cli.command(name="project")
@click.argument("out_dir", type=click.Path(file_okay=False, path_type=Path))
@click.option(
    "--config", "config_path", type=click.Path(exists=True, path_type=Path), required=True
)
def project_cmd(out_dir: Path, config_path: Path) -> None:
    """Project the observation store in OUT_DIR into a playbook (L5 only).

    Reads ``observations.jsonl`` and ``corpus_manifest.json`` from OUT_DIR
    (written by ``playbook mine``) and compiles them into a schema-valid
    ``playbook.opf.json`` using purely deterministic rollup logic — zero
    ingest work, zero LLM calls.

    Re-running ``project`` after tuning rollup / position logic changes the
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
        project_playbook(
            out_dir=out_dir_resolved,
            config=cfg,
            taxonomy=taxonomy,
            progress=click.echo,
        )
    except (PipelineError, AssemblyError) as exc:
        click.secho(f"ERROR: {exc}", fg="red", err=True)
        raise SystemExit(1) from exc

    click.secho(f"OK  {out_dir_resolved / 'playbook.opf.json'}", fg="green")


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
    layout before running ``playbook compile``.  Exits 0 when no errors are
    found, non-zero otherwise.
    """
    report = lint_corpus(corpus_dir, config_path=config_path)

    for item in report.items:
        if item.level == "ok":
            click.secho(f"  OK   {item.message}", fg="green")
        elif item.level == "warning":
            click.secho(f"  WARN {item.message}", fg="yellow")
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

    OUT_DIR is the output directory produced by ``playbook compile``.

    Lets a lawyer verify the engine's structural inferences — version ordering,
    signed-copy identification, provenance, and per-clause deviations — before
    trusting the compiled playbook.  If an inference is wrong, add a
    ``hints.yaml`` to the document folder and re-run with ``--no-cache``.
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


@cli.command(name="stage")
@click.argument("src_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option(
    "--out",
    "out_dir",
    type=click.Path(path_type=Path),
    default=None,
    help=("Staging output directory (default: ~/.cache/playbook-engine/staging/<src_dir_name>)."),
)
@click.option(
    "--copy",
    "copy_files",
    is_flag=True,
    default=False,
    help=(
        "Write real file copies instead of absolute symlinks. Use this when the "
        "staged output will cross a filesystem boundary (e.g. staged on the host, "
        "then bind-mounted read-only into a container) — symlink targets are host "
        "paths and dangle in that scenario."
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
        "review. Required first step for a corpus whose layout is 'unknown' "
        "(issue #186); works for any layout."
    ),
)
@click.option(
    "--plan",
    "plan_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help=(
        "Execute a staging_plan.json previously written by --plan-only "
        "(optionally hand/skill-edited) instead of detecting the layout."
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
    using symlinks (or real copies with ``--copy``), writes per-agreement
    ``hints.yaml`` (order + signed_version), and emits a
    ``playbook.config.yaml`` skeleton.

    When the layout can't be determined (``unknown`` — loose files, ad-hoc
    trees, no per-agreement subfolders) staging refuses to guess; run with
    ``--plan-only`` first to assemble a ``staging_plan.json`` proposal from
    file contents/metadata (issue #186), review/edit it, then re-run with
    ``--plan staging_plan.json`` to execute it.

    Writes only to the output directory (default:
    ~/.cache/playbook-engine/staging/<name>, a user-owned cache dir rather
    than world-readable /tmp — see issue #135). Never modifies SRC_DIR.
    """
    import json  # noqa: PLC0415

    from playbook_engine.intake_plan import build_staging_plan, execute_staging_plan
    from playbook_engine.staging import (  # noqa: PLC0415
        DEFAULT_STAGING_ROOT,
        UnknownLayoutError,
        scaffold_config,
        stage,
    )

    resolved = src_dir.resolve()
    dest = (out_dir or DEFAULT_STAGING_ROOT / resolved.name).resolve()

    click.echo(f"src    : {resolved}")
    click.echo(f"out    : {dest}")

    if plan_path is not None:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        result = execute_staging_plan(plan, resolved, dest, copy_files=copy_files)
        click.echo(f"layout : {result.layout} (from plan {plan_path})")
        click.echo(
            f"staged : {result.staged_count} version(s) across {result.agreement_count} agreement(s)"
            + (" (copied)" if copy_files else " (symlinked)")
        )
        scaffold_config(resolved, dest)
        click.echo(f"config : {dest / 'playbook.config.yaml'} (skeleton — fill in taxonomy path)")
        click.secho(f"OK  {dest}", fg="green")
        return

    if plan_only:
        plan = build_staging_plan(resolved)
        dest.mkdir(parents=True, exist_ok=True)
        plan_file = dest / "staging_plan.json"
        plan_file.write_text(json.dumps(plan, indent=2), encoding="utf-8")
        click.echo(f"plan   : {plan_file}")
        click.echo(
            f"        {len(plan['deals'])} candidate deal(s), "
            f"{len(plan['unassigned'])} unassigned file(s)"
        )
        click.secho(
            f"OK  wrote {plan_file} — review/edit, then run `playbook stage --plan {plan_file}`",
            fg="green",
        )
        return

    try:
        result = stage(resolved, dest, copy_files=copy_files)
    except UnknownLayoutError as exc:
        click.secho(f"ERROR: {exc}", fg="red", err=True)
        raise SystemExit(1) from exc

    click.echo(f"layout : {result.layout}")
    click.echo(
        f"staged : {result.staged_count} version(s) across {result.agreement_count} agreement(s)"
        + (" (copied)" if copy_files else " (symlinked)")
    )

    scaffold_config(resolved, dest)
    click.echo(f"config : {dest / 'playbook.config.yaml'} (skeleton — fill in taxonomy path)")

    if result.missing:
        click.secho(
            f"WARN {len(result.missing)} manifest file(s) missing on disk:", fg="yellow", err=True
        )
        for m in result.missing[:10]:
            click.secho(f"       {m}", fg="yellow", err=True)

    click.secho(f"OK  {dest}", fg="green")


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
    "--plan",
    "plan_only",
    is_flag=True,
    default=False,
    help=(
        "Print deduped pending counts by kind and a rough token estimate, then exit "
        "without writing observations.jsonl."
    ),
)
@click.option(
    "--subset",
    "subset",
    type=int,
    default=None,
    help="Record at most N pending items (trial mode).",
)
def judge_cmd(
    corpus_dir: Path,
    config_path: Path,
    out_path: Path | None,
    plan_only: bool,
    subset: int | None,
) -> None:
    """Run mine_corpus with store-backed judges and emit the pending review queue.

    Reads the verdict store at <out>/judge/verdicts.jsonl and replays any
    previously supplied verdicts.  For every new clause payload not in the store,
    appends a full record to <out>/judge/pending.jsonl.

    The verdict-cache layer (out/.cache/verdicts.jsonl) is intentionally bypassed
    when using store-backed judges (no_cache=True).  This prevents stale
    needs_review sentinels from being replayed across rounds — the store-backed
    judges own the verdict life-cycle, not the JudgmentCache.

    Use ``playbook judge-apply`` to load verdicts into the store, then re-run
    ``playbook judge`` to confirm no new items are pending.  Finally run
    ``playbook mine`` + ``playbook project`` for the final playbook.
    """
    from playbook_engine.agent_judge import (  # noqa: PLC0415
        PendingQueue,
        StoreBackedClassificationJudge,
        StoreBackedDeviationJudge,
        StoreBackedProvenanceJudge,
        StoreBackedScopeJudge,
        VerdictStore,
    )

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
    judge_dir = out_dir / "judge"
    verdicts_path = judge_dir / "verdicts.jsonl"
    pending_path = judge_dir / "pending.jsonl"

    click.echo(f"corpus  : {corpus_dir}")
    click.echo(f"config  : {config_path}")
    click.echo(f"out     : {out_dir}")
    click.echo(f"verdicts: {verdicts_path}")

    # Segment exactly the way ``mine`` does, or the store-backed judges here
    # generate verdict keys that never match the LLM-segmented observation store
    # and the drain loop cannot converge.  Keyed off the real out_dir so the
    # segmentation AND extraction caches (issue #132) are shared with plan
    # mode and later ``mine``/``compile`` runs — even though --plan mode below
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

    # --plan mode: report pending counts without writing observations.
    # We still run mine_corpus (with a temp pending queue) to compute the plan.
    if plan_only:
        import tempfile  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as _tmp:
            plan_pending = PendingQueue(Path(_tmp) / "pending.jsonl")
            scope_judge = StoreBackedScopeJudge(store=store, pending=plan_pending)
            cls_judge = StoreBackedClassificationJudge(store=store, pending=plan_pending)
            dev_judge = StoreBackedDeviationJudge(store=store, pending=plan_pending)
            prov_judge = StoreBackedProvenanceJudge(store=store, pending=plan_pending)

            try:
                mine_corpus(
                    corpus_dir=corpus_dir.resolve(),
                    config=cfg,
                    taxonomy=taxonomy,
                    out_dir=Path(_tmp) / "mine_out",
                    scope_judge=scope_judge,
                    classification_judge=cls_judge,
                    deviation_judge=dev_judge,
                    provenance_judge=prov_judge,
                    no_cache=True,
                    progress=click.echo,
                    **seg_kwargs,
                )
            except PipelineError as exc:
                click.secho(f"ERROR: {exc}", fg="red", err=True)
                raise SystemExit(1) from exc

            plan_pending_path = Path(_tmp) / "pending.jsonl"
            if not plan_pending_path.exists():
                click.secho("OK  0 pending items (all verdicts already in store)", fg="green")
                _echo_segmentation_cost_line(seg_stats, click.echo)
                return

            import json  # noqa: PLC0415

            pending_records = [
                json.loads(line)
                for line in plan_pending_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            counts: dict[str, int] = {}
            for rec in pending_records:
                counts[rec["kind"]] = counts.get(rec["kind"], 0) + 1

            total = sum(counts.values())
            # Token estimate from the real payload sizes (issue #134) — a
            # flat per-item average previously ignored that a provenance
            # payload (preamble + letterhead) and a deviation payload (a
            # full diff hunk) can differ from a short classify payload by
            # an order of magnitude. ``//4`` is the same chars-per-token
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
        return

    # Normal mode: write observations and update pending queue.
    if subset is not None and subset <= 0:
        click.secho("ERROR: --subset must be a positive integer", fg="red", err=True)
        raise SystemExit(1)

    # Reset the pending queue so each round is rewritten from scratch (the
    # contract SKILL.md documents). PendingQueue appends and never truncates,
    # so without this a re-run after judge-apply keeps the prior round's stale
    # entries and the drain loop can never reach empty (issue #182).
    pending_path.unlink(missing_ok=True)

    pending_queue = PendingQueue(pending_path)
    scope_judge = StoreBackedScopeJudge(store=store, pending=pending_queue)
    cls_judge = StoreBackedClassificationJudge(store=store, pending=pending_queue)
    dev_judge = StoreBackedDeviationJudge(store=store, pending=pending_queue)
    prov_judge = StoreBackedProvenanceJudge(store=store, pending=pending_queue)

    try:
        mine_corpus(
            corpus_dir=corpus_dir.resolve(),
            config=cfg,
            taxonomy=taxonomy,
            out_dir=out_dir,
            scope_judge=scope_judge,
            classification_judge=cls_judge,
            deviation_judge=dev_judge,
            provenance_judge=prov_judge,
            no_cache=True,
            progress=click.echo,
            **seg_kwargs,
        )
    except PipelineError as exc:
        click.secho(f"ERROR: {exc}", fg="red", err=True)
        raise SystemExit(1) from exc

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
            total_pending = subset

        click.secho(f"OK  {out_dir / 'observations.jsonl'}", fg="green")
        click.echo(f"Pending items: {total_pending}")
        for kind, count in sorted(counts.items()):
            click.echo(f"  {kind}: {min(count, subset) if subset else count}")
    else:
        click.secho(f"OK  {out_dir / 'observations.jsonl'} (0 pending items)", fg="green")


@cli.command(name="judge-apply")
@click.argument("out_dir", type=click.Path(file_okay=False, path_type=Path))
@click.option(
    "--verdicts",
    "verdicts_path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="JSONL file of verdicts to load (each line: {'key': '<sha256>', 'verdict': {...}}).",
)
def judge_apply_cmd(out_dir: Path, verdicts_path: Path) -> None:
    """Load verdicts from a JSONL file into the verdict store.

    Reads OUT_DIR/judge/verdicts.jsonl (created by ``playbook judge``) and
    merges in the verdicts from VERDICTS_PATH.  Each line in VERDICTS_PATH must
    be a JSON object with a ``key`` (SHA-256 string) and a ``verdict`` dict.

    Malformed lines are rejected with a non-zero exit code and the line number
    reported.  Valid lines are appended to the store even if earlier lines fail
    (partial loads are not performed — all lines are validated first).

    After applying verdicts, re-run ``playbook judge`` to confirm the pending
    queue is empty, then run ``playbook mine`` + ``playbook project`` to compile
    the final playbook with the judged taxonomy_ids populated.
    """
    import json  # noqa: PLC0415

    from playbook_engine.agent_judge import VerdictStore  # noqa: PLC0415

    out_dir_resolved = out_dir.resolve()
    verdicts_store_path = out_dir_resolved / "judge" / "verdicts.jsonl"

    # Validate all lines first before touching the store.
    raw_lines = verdicts_path.read_text(encoding="utf-8").splitlines()
    valid_records: list[tuple[str, dict[str, Any]]] = []
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
        valid_records.append((record["key"], record["verdict"]))

    if not valid_records:
        click.secho("WARN: no valid verdict records found in file", fg="yellow", err=True)
        return

    # Load all validated records into the store.
    store = VerdictStore(verdicts_store_path)
    loaded = 0
    for key, verdict in valid_records:
        # Use put_by_key to load verdicts by their pre-computed key directly,
        # bypassing the payload hashing step (the key was computed by the producer).
        store.put_by_key(key, verdict)
        loaded += 1

    click.secho(f"OK  loaded {loaded} verdict(s) into {verdicts_store_path}", fg="green")


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
def segment_cmd(corpus_dir: Path, config_path: Path, out_path: Path | None) -> None:
    """Emit the agent segmentation queue for CORPUS_DIR (issue #191).

    Key-free store-backed segmentation. Extracts each version and, for every
    document whose segmentation is not yet cached, appends its ``Block`` stream
    to ``<out>/segment/pending.jsonl``. Read that queue, partition each
    document's blocks into contiguous clause ranges (``SegNode`` list) — one
    per clause — write them to a verdicts JSONL, then run
    ``playbook segment-apply`` and ``playbook mine``. No API key is used.

    The queue is rewritten from scratch each run, so re-running after
    ``segment-apply`` reports only what still needs segmenting (empty = done).
    Requires ``segmentation.agent: true`` in the config.
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
    if not cfg.segmentation.agent:
        click.secho(
            "ERROR: `segment` requires `segmentation.agent: true` in the config.",
            fg="red",
            err=True,
        )
        raise SystemExit(1)
    try:
        taxonomy = load_taxonomy(cfg.taxonomy_path)
    except TaxonomyError as exc:
        click.secho(f"Taxonomy error: {exc}", fg="red", err=True)
        raise SystemExit(1) from exc

    out_dir = (out_path or corpus_dir.parent / "out").resolve()
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
                canonical_text, blocks, _extractor = extract_blocks(path, cache=extraction_cache)
            except Exception as exc:  # noqa: BLE001
                click.secho(
                    f"  WARNING: {doc_dir.name}/{path.name}: extraction failed ({exc}) — skipped",
                    fg="yellow",
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
    if n_queued == 0:
        click.secho("OK  all documents segmented (cache full) — run `playbook mine`", fg="green")
    else:
        click.secho(
            f"OK  {pending_path} — segment each item, then `playbook segment-apply`", fg="green"
        )


@cli.command(name="segment-apply")
@click.argument("out_dir", type=click.Path(file_okay=False, path_type=Path))
@click.option(
    "--verdicts",
    "verdicts_path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help='JSONL of segmentations (each line: {"canonical_text": "...", "nodes": [SegNode, ...]}).',
)
def segment_apply_cmd(out_dir: Path, verdicts_path: Path) -> None:
    """Load agent-produced ``SegNode`` lists into the segmentation cache (issue #191).

    Each line of VERDICTS_PATH is a JSON object with ``canonical_text`` (echoed
    from the pending item) and ``nodes`` (a list of ``SegNode`` dicts —
    ``node_id``, ``parent_id``, ``order``, ``heading``, ``taxonomy_id``,
    ``start_block_id``, ``end_block_id``, and optional ``start_quote`` /
    ``end_quote``). Nodes should partition the document's blocks into contiguous
    clause ranges. All lines are validated before the cache is touched.

    After applying, re-run ``playbook segment`` to confirm the queue is empty,
    then ``playbook mine`` (which replays the cached segmentation — no API call).
    """
    import json  # noqa: PLC0415

    from playbook_engine.agent_segmenter import AGENT_SEGMENTER_MODEL  # noqa: PLC0415
    from playbook_engine.llm_segmenter_batch import (  # noqa: PLC0415
        SegmentationVerdictCache,
        _seg_node_from_dict,
    )

    cache_path = out_dir.resolve() / "segment" / "cache.jsonl"
    raw_lines = verdicts_path.read_text(encoding="utf-8").splitlines()
    records: list[tuple[str, list[Any]]] = []
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
        records.append((rec["canonical_text"], nodes))

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
    review.  The output is loadable by ``playbook compile --config``.

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

    _SUPPORTED = frozenset({".docx", ".pdf", ".rtf"})
    trees: list[ClauseTree] = []
    corpus_resolved = corpus_dir.resolve()
    doc_dirs = sorted(
        d for d in corpus_resolved.iterdir() if d.is_dir() and not d.name.startswith(".")
    )
    for doc_dir in doc_dirs:
        version_files = sorted(
            (
                p
                for p in doc_dir.iterdir()
                if p.is_file() and p.suffix.lower() in _SUPPORTED and p.stem.lower() != "hints"
            ),
            key=lambda p: _natural_sort_key(p.stem),
        )
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
                ext = version_path.suffix.lower()
                if ext == ".docx":
                    from playbook_engine.docx_ingester import ingest_docx

                    tree = ingest_docx(version_path, doc_id, version).tree
                elif ext == ".rtf":
                    from playbook_engine.rtf_ingester import ingest_rtf

                    tree = ingest_rtf(version_path, doc_id, version).tree
                elif ext == ".pdf":
                    from playbook_engine.pdf_ingester import ingest_pdf

                    tree = ingest_pdf(version_path, doc_id, version).tree
                else:  # pragma: no cover - _SUPPORTED filters to these three
                    continue
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


@cli.command(name="report")
@click.argument("out_dir", type=click.Path(file_okay=False, path_type=Path))
@click.option(
    "--out",
    "report_path",
    type=click.Path(path_type=Path),
    default=None,
    help=(
        "Write the Markdown report to this file (default: print to stdout). "
        "When supplied, also writes a JSON twin alongside at the same stem "
        "(e.g. --out report.md writes both report.md and report.json)."
    ),
)
def report_cmd(out_dir: Path, report_path: Path | None) -> None:
    """Render an after-action report from a compiled OUT_DIR.

    Reads ``scope.json``, ``trail/*.json``, ``observations.jsonl``,
    ``playbook.opf.json``, and ``judge/`` from OUT_DIR and renders a
    structured Markdown report with six sections: Corpus Coverage, Backbone
    Health, Judgment Economics, Semantic Coverage, Needs Attention, and
    Honesty.

    When ``--out report.md`` is supplied, a JSON twin (``report.json``) is
    written alongside so downstream tooling can consume the structured data.
    """
    try:
        if report_path:
            write_after_action_report(out_dir.resolve(), report_path.resolve())
            click.secho(f"OK  {report_path}", fg="green")
            json_twin = report_path.with_suffix(".json")
            click.secho(f"OK  {json_twin}", fg="green")
        else:
            click.echo(build_after_action_report(out_dir.resolve()))
    except FileNotFoundError as exc:
        click.secho(f"ERROR: {exc}", fg="red", err=True)
        raise SystemExit(1) from exc


@cli.group(name="view")
def view_group() -> None:
    """Render a review HTML surface or apply reviewer feedback."""


@view_group.command(name="render")
@click.argument("out_dir", type=click.Path(file_okay=False, path_type=Path))
@click.option(
    "--out",
    "out_file",
    type=click.Path(path_type=Path),
    default=None,
    help=(
        "Write the HTML to this file (default: <out_dir>/playbook.review.html). "
        "Also prints the path on success."
    ),
)
@click.option(
    "--alias-map",
    "alias_map_file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help=(
        "Path to the held-out alias->entity map (e.g. <out_dir>/alias_map.json, "
        "written by 'playbook mine' — issue #153). When given, resolves aliases "
        "to real names in the rendered HTML for authorized reviewers (issue #146). "
        "Possessing read access to this restricted file IS the authorization gate; "
        "omit this flag to render a still-safe, alias-only view."
    ),
)
def view_render_cmd(out_dir: Path, out_file: Path | None, alias_map_file: Path | None) -> None:
    """Render OUT_DIR/playbook.opf.json as a self-contained review HTML.

    The HTML file embeds the full playbook JSON, requires no network access,
    and contains per-clause comment boxes plus an Export feedback button.

    OUT_DIR is the output directory produced by ``playbook compile``.
    """
    from playbook_engine.viewer import load_alias_map, render_review_html  # noqa: PLC0415

    resolved = out_dir.resolve()
    dest = out_file.resolve() if out_file else resolved / "playbook.review.html"

    alias_map = load_alias_map(alias_map_file.resolve()) if alias_map_file else None

    try:
        render_review_html(resolved, out_file=dest, alias_map=alias_map)
    except FileNotFoundError as exc:
        click.secho(f"ERROR: {exc}", fg="red", err=True)
        raise SystemExit(1) from exc

    click.secho(f"OK  {dest}", fg="green")
    if alias_map:
        click.secho(f"  aliases resolved to real names using {alias_map_file}", fg="cyan")


@view_group.command(name="apply")
@click.argument("out_dir", type=click.Path(file_okay=False, path_type=Path))
@click.argument("feedback_file", type=click.Path(exists=True, path_type=Path))
def view_apply_cmd(out_dir: Path, feedback_file: Path) -> None:
    """Apply FEEDBACK_FILE corrections to OUT_DIR.

    Translates provenance/signed/order corrections into per-document
    hints.yaml entries, classification corrections into VerdictStore entries,
    free-text notes/comments into viewer_notes.md, and ``override``
    (attorney-pinned position) corrections into a ``curation`` pin embedded
    directly in ``playbook.opf.json`` (issue #147) — it survives a later
    recompile and is flagged if fresh evidence contradicts it. Any key it
    cannot honor is reported as not applied rather than counted toward a
    false "OK" (issue #138).

    OUT_DIR is the output directory produced by ``playbook compile``.
    FEEDBACK_FILE is the feedback.json produced by the HTML viewer.
    """
    from playbook_engine.viewer import apply_feedback  # noqa: PLC0415

    resolved = out_dir.resolve()

    try:
        result = apply_feedback(resolved, feedback_file.resolve())
    except (FileNotFoundError, ValueError) as exc:
        click.secho(f"ERROR: {exc}", fg="red", err=True)
        raise SystemExit(1) from exc

    if result.hints_written:
        for doc_id in result.hints_written:
            click.secho(f"  hints.yaml updated for {doc_id}", fg="cyan")
    if result.verdicts_written:
        click.secho(
            f"  {result.verdicts_written} verdict(s) written to judge/verdicts.jsonl", fg="cyan"
        )
    if result.notes_written:
        click.secho(f"  notes appended to {resolved / 'viewer_notes.md'}", fg="cyan")
    if result.pins_written:
        for item_num in result.pins_written:
            click.secho(f"  {item_num}: position pinned in playbook.opf.json", fg="cyan")
    for item_num, messages in result.skipped.items():
        for message in messages:
            click.secho(f"  {item_num}: not applied — {message}", fg="yellow")

    applied = bool(
        result.hints_written
        or result.verdicts_written
        or result.notes_written
        or result.pins_written
    )
    if applied:
        click.secho("OK  feedback applied", fg="green")
    else:
        click.secho("NOTE  no feedback applied — all keys unsupported or unresolved", fg="yellow")


@cli.group(name="posture")
def posture_group() -> None:
    """Author the OPF Posture via a short GC interview (issue #156)."""


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
def posture_interview_cmd(out_dir: Path, answers_file: Path | None) -> None:
    """Run the Posture interview and write a versioned Posture into OUT_DIR/playbook.opf.json.

    Asks the canonical 3-6 question set (OPF-SPEC.md §7), assembles
    the answers deterministically into ``posture.system_prompt``, and writes
    the result into OUT_DIR/playbook.opf.json as a governed, versioned block:
    each re-run against an existing Posture bumps ``posture.version`` by 1.

    Warns (non-blocking) if the assembled Posture softens language around a
    concept a Floor invariant protects — a possible Posture-vs-Floor conflict
    for a human to review.

    OUT_DIR must already contain a playbook.opf.json (from 'playbook
    compile'/'project').
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
        result = apply_posture_interview(out_dir_resolved, answers, generated_at=generated_at)
    except (FileNotFoundError, PostureError) as exc:
        click.secho(f"ERROR: {exc}", fg="red", err=True)
        raise SystemExit(1) from exc

    click.secho(f"OK  posture.version={result.version} written to {result.path}", fg="green")
    for warning in result.warnings:
        click.secho(f"WARN  {warning}", fg="yellow")


@cli.group(name="floor")
def floor_group() -> None:
    """Propose Floor candidates for legal review (issue #166)."""


@floor_group.command(name="propose")
@click.argument("out_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
def floor_propose_cmd(out_dir: Path) -> None:
    """Derive Floor candidates from reversals + the Posture interview's Q4 answer.

    Reads OUT_DIR/observations.jsonl (every ``outcome: proposed_then_reversed``
    observation is a candidate red line — OPF-SPEC.md §3.7 rule 4)
    and, if a Posture interview has been run, OUT_DIR/playbook.opf.json's
    ``posture.generation.interview`` Q4 ("sacred_clauses") answer. Writes
    OUT_DIR/floor.candidates.json and prints a summary table.

    This is a REVIEW ARTIFACT for the legal owner — it never writes to the
    OPF ``floor`` section, and never auto-promotes a candidate into
    ``floor.invariants``. Accepting a candidate is a human act: edit
    ``floor.invariants`` directly, or via the curation CLI.
    """
    import json  # noqa: PLC0415

    out_dir_resolved = out_dir.resolve()
    result_path = write_floor_candidates(out_dir_resolved)
    candidates = json.loads(result_path.read_text(encoding="utf-8"))["candidates"]

    if not candidates:
        click.secho(f"OK  {result_path} (0 candidates)", fg="green")
        return

    click.secho(f"OK  {result_path} ({len(candidates)} candidate(s))", fg="green")
    click.echo("")
    click.echo(f"{'id':<10} {'source':<14} {'citations':<10} statement")
    click.echo("-" * 70)
    for c in candidates:
        click.echo(f"{c['id']:<10} {c['source']:<14} {len(c['citations']):<10} {c['statement']}")


@cli.command(name="curate")
@click.argument("out_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option(
    "--command",
    "commands_inline",
    multiple=True,
    help=(
        "A single curate instruction, e.g. 'pin governing_law to usually_conceded' "
        "or 'note governing_law: check next cycle'. Repeatable."
    ),
)
@click.option(
    "--file",
    "commands_file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Path to a file of curate instructions, one per line ('#' starts a comment line).",
)
@click.option(
    "--by",
    "pinned_by",
    default=None,
    help="Attribution stamped on any pin created this run (curation.pins[].pinned_by).",
)
def curate_cmd(
    out_dir: Path,
    commands_inline: tuple[str, ...],
    commands_file: Path | None,
    pinned_by: str | None,
) -> None:
    """Apply chat-style curate instructions to OUT_DIR/playbook.opf.json (issue #159).

    A minimal deterministic command grammar (not LLM-driven parsing) for the
    #104 "chat to fine-tune" interaction goal:

    \b
        pin governing_law to usually_conceded[: optional comment]
        note governing_law: free-text note

    Pins are embedded directly in ``curation.pins`` (issue #147 — survive a
    later recompile, and are flagged if fresh evidence contradicts the
    pinned position); notes are appended to ``viewer_notes.md``. Every run
    also refreshes conflict status on every already-embedded pin, so
    evidence that moved via any other path (a recompile, a hand edit) is
    reported the next time ``curate`` runs.

    Give instructions via repeated ``--command``, a ``--file`` of one
    instruction per line, or both.

    OUT_DIR must already contain a playbook.opf.json (from 'playbook
    compile'/'project').
    """
    from playbook_engine.chat_curate import apply_curate_commands  # noqa: PLC0415

    commands: list[str] = list(commands_inline)
    if commands_file is not None:
        commands.extend(commands_file.read_text(encoding="utf-8").splitlines())

    if not commands:
        click.secho("ERROR: no curate instructions given (--command / --file)", fg="red", err=True)
        raise SystemExit(1)

    try:
        result = apply_curate_commands(out_dir.resolve(), commands, pinned_by=pinned_by)
    except FileNotFoundError as exc:
        click.secho(f"ERROR: {exc}", fg="red", err=True)
        raise SystemExit(1) from exc

    for outcome in result.outcomes:
        if outcome.action == "conflict":
            click.secho(f"CONFLICT  {outcome.clause_id}: {outcome.detail}", fg="red")
        elif outcome.applied:
            click.secho(f"OK  {outcome.action} {outcome.clause_id}: {outcome.detail}", fg="green")
        else:
            click.secho(f"SKIP  {outcome.command!r} — {outcome.detail}", fg="yellow")

    summary = f"applied {result.pins_written} pin(s), {result.notes_written} note(s)"
    if result.conflicts:
        summary += f", {len(result.conflicts)} conflict(s) flagged"
    click.secho(summary, fg="cyan")
