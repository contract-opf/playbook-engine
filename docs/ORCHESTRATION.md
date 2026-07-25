# playbook-engine orchestration

> **This document describes the checkpoint-review loop, not the primary
> workflow.** For the real derivation path (LLM-backed judgment producing a
> playbook with actual semantics), see
> [`.claude/skills/playbook-from-corpus/SKILL.md`](../.claude/skills/playbook-from-corpus/SKILL.md).

Autonomously reviews a compiled corpus's intermediate artifacts and intervenes,
delegating to the LLM judges only on flagged items. Token-efficient by design — it
reads structured artifacts (`review.json` / `trail/` / `scope.json` / `observations.jsonl`),
never the raw corpus.

---

## What this skill does

The checkpoint-review loop runs **before** the final
playbook is compiled, automatically triaging review flags and intervening where possible.

1. **Stops after intermediates** — runs `playbook compile --stop-after intermediates`
   to produce L1–L4 artifacts without a playbook.
2. **Reviews artifacts** — runs the review module to get structured
   `ReviewFlag` objects and writes `review.json`.
3. **Triages each flag** — applies the intervention vocabulary to decide
   **PASS | INTERVENE | ESCALATE**.
4. **Intervenes** — either writes a `hints.yaml` to the relevant document
   subfolder and/or re-runs the compile with `--no-cache` to pick up the correction.
5. **Escalates** — records flags that require human review in the review report.
6. **Compiles the playbook** — runs the full `playbook compile` to produce `playbook.opf.json`.

> **LLM judge calls are out of scope here.**  The orchestrator gates on artifact-level
> flags; the judges gate on per-item confidence bands.  Real-LLM behaviour is validated
> by the golden eval suite, not this skill.

---

## Before you start

Your corpus must already pass `playbook lint-corpus`. See
[`QUICK-COMPILE.md`](QUICK-COMPILE.md) for corpus layout and config requirements.

---

## Checkpoint-review loop

The orchestrator runs the following steps in sequence:

```
Step 1  compile_corpus(stop_after="intermediates")
           └─ Writes: scope.json, trail/, observations.jsonl, corpus_manifest.json
Step 2  write_review(out_dir)
           └─ Writes: review.json with structured ReviewFlag list
Step 3  triage_flags(flags)
           └─ Returns: list[TriageResult] with PASS | INTERVENE | ESCALATE per flag
Step 4  For each INTERVENE:
           ├─ WRITE_HINTS → write/merge hints.yaml + mark re-run needed
           └─ RERUN      → mark re-run needed (transient error retry)
Step 5  If re-run needed:
           └─ compile_corpus(no_cache=True, stop_after="intermediates")
Step 6  If re-run happened:
           └─ write_review(out_dir)   # re-review updated artifacts
Step 7  compile_corpus()              # full L1–L5; produces playbook.opf.json
           └─ SKIPPED if an unresolved block-severity escalation remains and
              force=False (see below)
```

The full compile (Step 7) runs unless an unresolved `block`-severity escalation remains
— in that case it is skipped (`playbook=None`, `blocked_by_escalation=True`) unless the
caller passes `force=True`. Non-blocking (`warn`) escalations are recorded but never
block compilation — a human can review those and decide whether to trust the playbook
or re-run with manual corrections.

---

## Intervention vocabulary

### Decision: PASS

Flag is informational or already resolved. No action taken.

| Trigger | Explanation |
|---------|------------|
| Any flag kind not listed below | Treated as informational; logged but ignored. |

### Decision: INTERVENE

The orchestrator attempts an automated correction.

| Flag kind | Intervention type | Action |
|-----------|-------------------|--------|
| `unreliable_provenance` | `WRITE_HINTS` | Writes `provenance: counterparty_paper` to the document's `hints.yaml` (conservative default). The operator should verify this is correct. |
| `scope_judge_failed` (warn) | `RERUN`, only if `new_verdicts_available=True` | Re-runs the compile with `--no-cache`. Otherwise ESCALATE — see below. |
| `deviation_needs_review` | `RERUN`, only if `new_verdicts_available=True` | Re-runs the compile with `--no-cache`. Otherwise ESCALATE — see below. |

> **`WRITE_HINTS`** merges overrides into the existing `hints.yaml` (existing keys are
> preserved). Re-runs the compile with `no_cache=True` to pick up the new hints.
>
> **`RERUN`** re-runs without modifying hints. It is gated on
> `new_verdicts_available`, because `scope_judge_failed`/`deviation_needs_review`
> are produced either by
> stub judges (fully deterministic — a bare re-run reproduces the identical flag) or by
> store-backed judges (identical output until a *new* verdict has actually been applied,
> which a bare re-run does not do). So RERUN only fires when the caller has concrete
> evidence the underlying cause changed — e.g. a `playbook judge-apply` round landed new
> verdicts, or the judge was swapped for a working one. The default (`False`) escalates
> instead of silently re-running for no effect.

### Decision: ESCALATE

Flag requires human review. The orchestrator records it in the review report.
Non-blocking (`warn`) escalations do **not** block compilation. `block`-severity
escalations **do** — Step 7 is skipped (`playbook=None`) unless `force=True`.

| Flag kind | Reason for escalation |
|-----------|----------------------|
| `weak_signed_anchor` | The orchestrator cannot determine which version is signed without corpus content access. Add `signed_version: <filename_stem>` to `hints.yaml` manually. |
| `ambiguous_version_chain` | The version ordering used a non-deterministic basis (`greedy` or `llm`). Review manually or supply an explicit `order:` list in `hints.yaml`. |
| `fork_or_missing_draft` | A fork or missing intermediate draft was detected. Locate the missing version and re-compile. |
| `low_coherence` | A clause position was flagged as low-confidence by the CoherenceJudge. Review clause position reliability before publishing. |
| `scope_judge_failed` / `deviation_needs_review` (no `new_verdicts_available`) | RERUN would reproduce the identical flag; escalate instead. |
| Any flag with `severity: block` | Block-severity flags always escalate regardless of kind, **and suppress the full compile unless `force=True`**. |

---

## Checklist → action mapping

| Severity | Kind | Decision | Action |
|----------|------|----------|--------|
| block | `scope_judge_failed` | ESCALATE | (block severity wins) — **also suppresses Step 7 unless `force=True`** |
| warn | `scope_judge_failed` | ESCALATE, unless `new_verdicts_available=True` | Default: no evidence a re-run helps. With evidence: INTERVENE / RERUN L1–L4 |
| warn | `deviation_needs_review` | ESCALATE, unless `new_verdicts_available=True` | Default: no evidence a re-run helps. With evidence: INTERVENE / RERUN L1–L4 |
| warn | `unreliable_provenance` | INTERVENE / WRITE_HINTS | Write `provenance:` hint, re-run |
| warn | `weak_signed_anchor` | ESCALATE | Human must supply `signed_version:` |
| warn | `ambiguous_version_chain` | ESCALATE | Human must verify or supply `order:` |
| warn | `fork_or_missing_draft` | ESCALATE | Human must locate missing draft |
| warn | `low_coherence` | ESCALATE | Human must review clause position |

---

## Hints format reference

`hints.yaml` lives alongside the agreement files in its corpus subfolder.
For the full format, see the “Step 4 — Review the intermediates” section of [docs/QUICK-COMPILE.md](QUICK-COMPILE.md).

```yaml
# corpus/university-of-example/hints.yaml
signed_version: v3-fully-executed.pdf   # stem only (no extension also accepted)
provenance: our_paper                   # our_paper | counterparty_paper
order:
  - v1-draft-we-sent.docx
  - v2-their-redline.docx
  - v3-fully-executed.pdf
```

Re-run after editing:

```bash
playbook compile ./corpus --config ./playbook.config.yaml --out ./out --no-cache
```

---

## Python API

The orchestrator is exposed as a Python function for programmatic use and testing:

```python
from playbook_engine.review_orchestration import run_checkpoint_review

result = run_checkpoint_review(
    corpus_dir=corpus_dir,
    config=cfg,
    taxonomy=taxonomy,
    out_dir=out_dir,
    # new_verdicts_available=True,  # only if a judge-apply round landed new verdicts
    # force=True,                  # only to compile past an unresolved block escalation
    progress=print,
)

# result.escalations  — TriageResult list requiring human review
# result.hints_written — hints.yaml paths written by the orchestrator
# result.playbook     — the final playbook dict, or None if a block-severity
#                        escalation suppressed compilation (force=False)
# result.blocked_by_escalation — True when playbook is None for that reason
```

### TriageResult fields

| Field | Type | Description |
|-------|------|-------------|
| `flag` | `ReviewFlag` | The original flag |
| `decision` | `Decision` | PASS / INTERVENE / ESCALATE |
| `intervention_type` | `InterventionType \| None` | WRITE_HINTS / RERUN, or None |
| `hint_overrides` | `dict` | Key/value pairs merged into hints.yaml (WRITE_HINTS only) |

---

## Limitations

- **No LLM calls.** The orchestrator uses only stub judges and deterministic logic.
  Real-LLM judge implementations are BYO (bring-your-own).
- **No corpus content access.** The orchestrator reads only structured artifacts
  (`review.json`, `trail/`, `scope.json`, `observations.jsonl`). It cannot read
  the actual agreement text to make content-based decisions.
- **Conservative provenance default.** When writing a provenance hint, the
  orchestrator defaults to `counterparty_paper` (the safe/conservative choice).
  The operator should verify this is correct before publishing the playbook.
- **Single-pass intervention.** The orchestrator performs one intervention round
  and one re-compile. It does not loop until all flags are resolved.
