"""Checkpoint-review orchestration — the deterministic engine behind the
orchestrator skill (issue #60).

This module implements the non-LLM machinery for the checkpoint-review loop:

1. Run ``compile_corpus(..., stop_after="intermediates")`` to produce L1–L4
   artifacts without a playbook.
2. Run :func:`playbook_engine.review.write_review` to emit ``review.json``
   with structured :class:`~playbook_engine.review.ReviewFlag` objects.
3. Triage each flag via :func:`triage_flags` → PASS / INTERVENE / ESCALATE.
4. For INTERVENE decisions, write a ``hints.yaml`` to the relevant document
   subdirectory and/or re-run the compile with ``no_cache=True``.
5. For ESCALATE decisions, record the flag in the review report for a human.
6. Run the full compile to produce ``playbook.opf.json`` — UNLESS an
   unresolved ``block``-severity escalation remains, in which case
   compilation is skipped unless the caller passes ``force=True`` (issue #112).

Real-LLM validation is covered by the golden eval (issue #29), not here.
This module contains zero LLM calls and is fully deterministic.

See ``docs/ORCHESTRATION.md`` for the full intervention vocabulary and the
checklist→action mapping.  For the manual version of this loop see
``docs/QUICK-COMPILE.md:108-134``.

Security: No agreement content is stored in this module.
All corpus content is read from caller-supplied paths at runtime.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml

from playbook_engine.review import ReviewFlag, write_review

# ---------------------------------------------------------------------------
# Decision vocabulary
# ---------------------------------------------------------------------------


class Decision(StrEnum):
    """Orchestrator decision for a single :class:`~playbook_engine.review.ReviewFlag`.

    Attributes:
        PASS:      Flag is informational or already resolved — no intervention.
        INTERVENE: Engine can attempt an automated correction:
                   write a ``hints.yaml`` or re-run with ``no_cache=True``.
        ESCALATE:  Flag requires human review; record and continue.
    """

    PASS = "pass"
    INTERVENE = "intervene"
    ESCALATE = "escalate"


class InterventionType(StrEnum):
    """How the orchestrator intervenes for an INTERVENE decision.

    Attributes:
        WRITE_HINTS: Write or update the document's ``hints.yaml`` and re-run
                     the compile with ``no_cache=True`` to pick up the hints.
        RERUN:       Re-run with ``no_cache=True`` without hints (e.g. after a
                     transient judge error).
    """

    WRITE_HINTS = "write_hints"
    RERUN = "rerun"


@dataclass
class TriageResult:
    """Triage outcome for a single :class:`~playbook_engine.review.ReviewFlag`.

    Attributes:
        flag:              The original flag being triaged.
        decision:          PASS, INTERVENE, or ESCALATE.
        intervention_type: How to intervene (WRITE_HINTS / RERUN), or None for
                           PASS and ESCALATE decisions.
        hint_overrides:    Key→value pairs to merge into hints.yaml, keyed by
                           the YAML field name (e.g. ``{"provenance": "our_paper"}``).
                           Only populated for WRITE_HINTS interventions.
    """

    flag: ReviewFlag
    decision: Decision
    intervention_type: InterventionType | None = None
    hint_overrides: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Triage logic — flag.kind → Decision
# ---------------------------------------------------------------------------

#: Flag kinds that PASS without intervention (purely informational).
_PASS_KINDS: frozenset[str] = frozenset()

#: Flag kinds for which the orchestrator tries a WRITE_HINTS intervention.
_HINTS_KINDS: frozenset[str] = frozenset(
    {
        "weak_signed_anchor",
        "unreliable_provenance",
    }
)

#: Flag kinds for which the orchestrator tries a RERUN intervention.
_RERUN_KINDS: frozenset[str] = frozenset(
    {
        "scope_judge_failed",
        "deviation_needs_review",
    }
)

#: Flag kinds always escalated (require human review; no automated intervention).
_ESCALATE_KINDS: frozenset[str] = frozenset(
    {
        "fork_or_missing_draft",
        "ambiguous_version_chain",
        "low_coherence",
    }
)

#: Fallback hint overrides by kind when the flag does not supply specific values.
#: For kinds that need user-supplied data (e.g. which version is signed), we
#: escalate rather than write an incomplete hints.yaml.
_HINT_OVERRIDES_BY_KIND: dict[str, dict[str, Any]] = {
    # unreliable_provenance → default to counterparty_paper (conservative).
    "unreliable_provenance": {"provenance": "counterparty_paper"},
}


def triage_flags(
    flags: list[ReviewFlag], *, new_verdicts_available: bool = False
) -> list[TriageResult]:
    """Map each :class:`~playbook_engine.review.ReviewFlag` to a :class:`TriageResult`.

    Decision rules (in precedence order):

    1. **ESCALATE** — ``block`` severity always escalates regardless of kind.
    2. **ESCALATE** — kinds in ``_ESCALATE_KINDS`` (require human review).
    3. **INTERVENE / WRITE_HINTS** — ``weak_signed_anchor`` and
       ``unreliable_provenance`` (hints can be auto-applied for provenance;
       signed-anchor escalates because we cannot reliably pick the right ID).
    4. **INTERVENE / RERUN** — ``scope_judge_failed`` and
       ``deviation_needs_review``, but *only* when ``new_verdicts_available``
       is True.  Otherwise **ESCALATE**.
    5. **PASS** — everything else.

    Note: ``weak_signed_anchor`` is escalated (not WRITE_HINTS) because the
    orchestrator cannot determine which version is the signed copy without
    corpus content access.  The human must supply ``signed_version`` in
    hints.yaml manually.

    Rule 4 rationale (issue #112): ``scope_judge_failed`` and
    ``deviation_needs_review`` are produced either by stub judges (fully
    deterministic — a bare re-run reproduces the identical flag) or by
    store-backed judges (identical until a *new* verdict has actually been
    applied to the verdict store, or the judge itself was swapped for a
    working one).  A bare ``no_cache=True`` re-run changes neither of those
    things, so blindly RERUN-ing accomplishes nothing but burning a compile
    pass and re-logging the same flag.  Callers must therefore supply
    ``new_verdicts_available=True`` only when they have concrete evidence the
    underlying cause changed (e.g. a ``playbook judge-apply`` round landed
    new verdicts since the flags were generated, or the judge instance was
    replaced). The safe default is ``False`` — escalate to a human instead of
    silently re-running for no effect.

    Args:
        flags: List of flags from :func:`~playbook_engine.review.review_out_dir`
               or :func:`~playbook_engine.review.write_review`.
        new_verdicts_available: True when the caller has evidence that new
               verdicts have been applied (or the judge was swapped) since
               these flags were produced — the only condition under which a
               RERUN intervention can plausibly change the outcome.

    Returns:
        One :class:`TriageResult` per flag, in the same order.
    """
    results: list[TriageResult] = []
    for flag in flags:
        results.append(_triage_one(flag, new_verdicts_available=new_verdicts_available))
    return results


def _triage_one(flag: ReviewFlag, *, new_verdicts_available: bool = False) -> TriageResult:
    # Rule 1: block severity → always ESCALATE.
    if flag.severity == "block":
        return TriageResult(flag=flag, decision=Decision.ESCALATE)

    # Rule 2: kinds that always escalate.
    if flag.kind in _ESCALATE_KINDS:
        return TriageResult(flag=flag, decision=Decision.ESCALATE)

    # Rule 3a: weak_signed_anchor — escalate (cannot auto-pick the signed version).
    if flag.kind == "weak_signed_anchor":
        return TriageResult(flag=flag, decision=Decision.ESCALATE)

    # Rule 3b: unreliable_provenance → WRITE_HINTS with conservative default.
    if flag.kind in _HINTS_KINDS:
        overrides = _HINT_OVERRIDES_BY_KIND.get(flag.kind, {})
        return TriageResult(
            flag=flag,
            decision=Decision.INTERVENE,
            intervention_type=InterventionType.WRITE_HINTS,
            hint_overrides=overrides,
        )

    # Rule 4: RERUN intervention — only when there's evidence a re-run could
    # actually change the outcome (new verdicts applied / judge swapped).
    # Without that evidence, a re-run against the same stub or unchanged
    # verdict store reproduces the identical flag, so escalate instead.
    if flag.kind in _RERUN_KINDS:
        if new_verdicts_available:
            return TriageResult(
                flag=flag,
                decision=Decision.INTERVENE,
                intervention_type=InterventionType.RERUN,
            )
        return TriageResult(flag=flag, decision=Decision.ESCALATE)

    # Rule 5: PASS.
    return TriageResult(flag=flag, decision=Decision.PASS)


# ---------------------------------------------------------------------------
# Intervention helpers
# ---------------------------------------------------------------------------


def apply_hints(corpus_dir: Path, document_id: str, hint_overrides: dict[str, Any]) -> Path:
    """Merge *hint_overrides* into the document's ``hints.yaml``.

    Reads the existing hints file if present (so existing keys are preserved),
    merges the overrides (overrides win on conflict), and writes atomically.

    Args:
        corpus_dir:     Root corpus directory containing document subdirectories.
        document_id:    Name of the document subdirectory.
        hint_overrides: Keys/values to write or update in hints.yaml.

    Returns:
        Path to the written hints.yaml.

    Raises:
        FileNotFoundError: If *corpus_dir* does not exist.
        ValueError:         If *document_id* is empty or *hint_overrides* is empty.
    """
    if not corpus_dir.exists():
        raise FileNotFoundError(f"Corpus directory not found: {corpus_dir}")
    if not document_id:
        raise ValueError("document_id must not be empty")
    if not hint_overrides:
        raise ValueError("hint_overrides must not be empty")

    doc_dir = corpus_dir / document_id
    doc_dir.mkdir(parents=True, exist_ok=True)

    hints_path = doc_dir / "hints.yaml"

    # Load existing hints if present so we don't clobber unrelated keys.
    existing: dict[str, Any] = {}
    if hints_path.exists():
        try:
            existing = yaml.safe_load(hints_path.read_text(encoding="utf-8")) or {}
        except Exception:  # noqa: BLE001
            existing = {}
    if not isinstance(existing, dict):
        existing = {}

    merged = {**existing, **hint_overrides}

    tmp = hints_path.with_suffix(".yaml.tmp")
    tmp.write_text(yaml.dump(merged, allow_unicode=True, sort_keys=False), encoding="utf-8")
    tmp.replace(hints_path)
    return hints_path


# ---------------------------------------------------------------------------
# Orchestration result
# ---------------------------------------------------------------------------


@dataclass
class OrchestrationResult:
    """Summary of a :func:`run_checkpoint_review` execution.

    Attributes:
        out_dir:           Output directory used.
        review_path:       Path to the written ``review.json``.
        triage:            Per-flag triage decisions.
        escalations:       Flags that require human review.
        interventions:     Flags that triggered automated intervention.
        hints_written:     Paths to hints.yaml files written by interventions.
        playbook:          The final playbook dict, or None when skipped due to
                           unresolved blocking escalations.
        rerun_triggered:   True when a RERUN intervention re-ran the compile.
        blocked_by_escalation: True when a ``block``-severity escalation
                           suppressed the full compile (``playbook`` is None
                           because ``force`` was not set — see issue #112).
    """

    out_dir: Path
    review_path: Path
    triage: list[TriageResult]
    escalations: list[TriageResult]
    interventions: list[TriageResult]
    hints_written: list[Path]
    playbook: dict[str, Any] | None
    rerun_triggered: bool
    blocked_by_escalation: bool = False


# ---------------------------------------------------------------------------
# Main orchestration entry point
# ---------------------------------------------------------------------------


def run_checkpoint_review(
    corpus_dir: Path,
    config: Any,  # EngineConfig — imported at call time to avoid circular import
    taxonomy: Any,  # Taxonomy — imported at call time to avoid circular import
    out_dir: Path,
    *,
    scope_judge: Any = None,
    classification_judge: Any = None,
    deviation_judge: Any = None,
    alignment_judge: Any = None,
    trail_judge: Any = None,
    signed_judge: Any = None,
    provenance_judge: Any = None,
    coherence_judge: Any = None,
    no_cache: bool = False,
    new_verdicts_available: bool = False,
    force: bool = False,
    progress: Any = None,
) -> OrchestrationResult:
    """Run the full checkpoint-review loop.

    Steps (mirrors the doc ``docs/ORCHESTRATION.md``):

    1. ``compile_corpus(..., stop_after="intermediates")`` — L1–L4 only.
    2. ``write_review(out_dir)`` — emit ``review.json``.
    3. ``triage_flags(flags)`` — classify each flag as PASS/INTERVENE/ESCALATE.
    4. For each INTERVENE flag: apply hints or mark for re-run.
    5. If any WRITE_HINTS or RERUN interventions fired, re-run
       ``compile_corpus(..., no_cache=True, stop_after="intermediates")``.
    6. Re-run ``write_review`` on the updated artifacts.
    7. Run the full ``compile_corpus`` (L1–L5) — produces ``playbook.opf.json``,
       UNLESS an unresolved ``block``-severity escalation remains and ``force``
       is False (see below).

    Note (issue #112): A ``block``-severity escalation (e.g. a scope judge
    that raised an error) means the review found something the engine could
    not verify.  Compiling a playbook anyway and treating it as done is how a
    corpus with a genuinely unverified document silently reaches downstream
    consumers.  So: if any escalation carries ``severity == "block"``, Step 7
    is skipped and ``playbook`` is None (``blocked_by_escalation=True``)
    unless the caller explicitly passes ``force=True`` to compile anyway.
    Non-blocking escalations (``warn``) never suppress compilation — a human
    can review those and decide whether to trust the playbook.

    Args:
        corpus_dir:           Root corpus directory.
        config:               Engine configuration (EngineConfig).
        taxonomy:             Loaded taxonomy object.
        out_dir:              Output directory for intermediates + playbook.
        scope_judge:          Optional scope judge (stub used if None).
        classification_judge: Optional classification judge (stub if None).
        deviation_judge:      Optional deviation judge (stub if None).
        alignment_judge:      Optional alignment judge (None = deterministic).
        trail_judge:          Optional trail/version-ordering judge (None = det.).
        signed_judge:         Optional signed-copy judge (None = deterministic).
        provenance_judge:     Optional provenance judge (None = deterministic).
        coherence_judge:      Optional coherence judge (None = skipped).
        no_cache:             Force full recompute on the initial mine pass.
        new_verdicts_available: Pass True when new verdicts have been applied
                              (e.g. via ``playbook judge-apply``) or a judge was
                              swapped since these flags were last produced —
                              the only condition under which a RERUN
                              intervention for ``scope_judge_failed`` /
                              ``deviation_needs_review`` can change the
                              outcome. Default False routes those kinds to
                              ESCALATE instead (see :func:`triage_flags`).
        force:                Compile the full playbook even when an
                              unresolved ``block``-severity escalation
                              remains. Default False (safe).
        progress:             Callable receiving progress message strings.

    Returns:
        :class:`OrchestrationResult` summarising flags, decisions, and paths.
    """
    from playbook_engine.pipeline import compile_corpus  # avoid circular import

    _progress = progress or (lambda _: None)

    # ------------------------------------------------------------------
    # Step 1: L1–L4 only
    # ------------------------------------------------------------------
    _progress("Orchestrator: step 1 — compiling intermediates (L1–L4)...")
    compile_corpus(
        corpus_dir=corpus_dir,
        config=config,
        taxonomy=taxonomy,
        out_dir=out_dir,
        scope_judge=scope_judge,
        classification_judge=classification_judge,
        deviation_judge=deviation_judge,
        alignment_judge=alignment_judge,
        trail_judge=trail_judge,
        signed_judge=signed_judge,
        provenance_judge=provenance_judge,
        coherence_judge=coherence_judge,
        no_cache=no_cache,
        stop_after="intermediates",
        progress=_progress,
    )

    # ------------------------------------------------------------------
    # Step 2: Review
    # ------------------------------------------------------------------
    _progress("Orchestrator: step 2 — reviewing artifacts...")
    review_path = write_review(out_dir)
    review_data = json.loads(review_path.read_text(encoding="utf-8"))
    from playbook_engine.review import ReviewFlag as _RF

    flags: list[ReviewFlag] = [
        _RF(
            document_id=f["document_id"],
            stage=f["stage"],
            kind=f["kind"],
            severity=f["severity"],
            detail=f["detail"],
            suggested_action=f["suggested_action"],
        )
        for f in review_data.get("flags", [])
    ]

    # ------------------------------------------------------------------
    # Step 3: Triage
    # ------------------------------------------------------------------
    _progress(f"Orchestrator: step 3 — triaging {len(flags)} flag(s)...")
    triage_results = triage_flags(flags, new_verdicts_available=new_verdicts_available)
    escalations = [t for t in triage_results if t.decision == Decision.ESCALATE]
    interventions = [t for t in triage_results if t.decision == Decision.INTERVENE]

    # ------------------------------------------------------------------
    # Step 4: Interventions
    # ------------------------------------------------------------------
    hints_written: list[Path] = []
    rerun_needed = False

    for tri in interventions:
        flag = tri.flag
        if tri.intervention_type == InterventionType.WRITE_HINTS:
            doc_id = flag.document_id
            if doc_id and tri.hint_overrides:
                _progress(
                    f"Orchestrator: step 4 — writing hints for {doc_id!r} "
                    f"({flag.kind}): {tri.hint_overrides}"
                )
                path = apply_hints(corpus_dir, doc_id, tri.hint_overrides)
                hints_written.append(path)
                rerun_needed = True
        elif tri.intervention_type == InterventionType.RERUN:
            _progress(f"Orchestrator: step 4 — scheduling re-run for {flag.kind!r} flag")
            rerun_needed = True

    # ------------------------------------------------------------------
    # Step 5: Re-run L1–L4 if interventions fired
    # ------------------------------------------------------------------
    rerun_triggered = False
    if rerun_needed:
        _progress("Orchestrator: step 5 — re-running L1–L4 with no_cache=True...")
        compile_corpus(
            corpus_dir=corpus_dir,
            config=config,
            taxonomy=taxonomy,
            out_dir=out_dir,
            scope_judge=scope_judge,
            classification_judge=classification_judge,
            deviation_judge=deviation_judge,
            alignment_judge=alignment_judge,
            trail_judge=trail_judge,
            signed_judge=signed_judge,
            provenance_judge=provenance_judge,
            coherence_judge=coherence_judge,
            no_cache=True,
            stop_after="intermediates",
            progress=_progress,
        )
        rerun_triggered = True

        # Step 6: Re-review after intervention.
        _progress("Orchestrator: step 6 — re-reviewing updated artifacts...")
        review_path = write_review(out_dir)

    # ------------------------------------------------------------------
    # Step 7: Full compile (L1–L5) — unless a block-severity escalation is
    # unresolved and the caller has not forced compilation (issue #112).
    # ------------------------------------------------------------------
    blocking_escalations = [t for t in escalations if t.flag.severity == "block"]
    if blocking_escalations and not force:
        _progress(
            f"Orchestrator: step 7 — skipped: {len(blocking_escalations)} "
            "unresolved block-severity escalation(s); pass force=True to "
            "compile anyway."
        )
        return OrchestrationResult(
            out_dir=out_dir,
            review_path=review_path,
            triage=triage_results,
            escalations=escalations,
            interventions=interventions,
            hints_written=hints_written,
            playbook=None,
            rerun_triggered=rerun_triggered,
            blocked_by_escalation=True,
        )

    _progress("Orchestrator: step 7 — running full compile (L1–L5)...")
    playbook = compile_corpus(
        corpus_dir=corpus_dir,
        config=config,
        taxonomy=taxonomy,
        out_dir=out_dir,
        scope_judge=scope_judge,
        classification_judge=classification_judge,
        deviation_judge=deviation_judge,
        alignment_judge=alignment_judge,
        trail_judge=trail_judge,
        signed_judge=signed_judge,
        provenance_judge=provenance_judge,
        coherence_judge=coherence_judge,
        no_cache=False,
        stop_after=None,
        progress=_progress,
    )

    return OrchestrationResult(
        out_dir=out_dir,
        review_path=review_path,
        triage=triage_results,
        escalations=escalations,
        interventions=interventions,
        hints_written=hints_written,
        playbook=playbook,
        rerun_triggered=rerun_triggered,
        blocked_by_escalation=False,
    )
