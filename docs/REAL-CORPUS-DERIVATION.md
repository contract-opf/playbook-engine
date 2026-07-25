# Real-corpus derivation — plan/execute discipline

**Status:** durable process doc. No client names, paths, or corpus content
belong in this file; keep it that way.

This document is for whoever (human or agent) runs the `playbook-from-corpus`
skill (`.claude/skills/playbook-from-corpus/SKILL.md`) against a **real** corpus —
i.e. actual negotiated agreements, not the synthetic fixtures under
`examples/`. It captures the two-phase discipline and guardrails that a real
run needs beyond what the ordered pipeline in the skill already documents.

---

## Why a separate phase discipline

Fixture runs are cheap and reversible: bad output just means re-running.
Real-corpus runs are not:

- LLM judgment calls have real token cost, scaling with corpus size (often
  thousands of clause nodes).
- The corpus itself may be confidential client material — it must never enter
  `git` (see the skill's Guardrails section).
- A bad provenance or classification judgment on a real agreement has legal
  consequences if trusted uncritically downstream.

So a real run is planned before it is executed, with an explicit go/no-go
checkpoint in between.

---

## Phase A — plan (investigate, then write the plan down)

Before spending any judgment budget:

1. **Confirm the environment.** Corpus location, config file, taxonomy, and
   baseline template all exist and are current — repo state and corpus
   contents both drift.
2. **Stage and lint.** Run `playbook stage` (if the corpus has a nested
   layout) and `playbook lint-corpus` until it exits 0. Do not proceed past
   lint errors.
3. **Decide the judge mechanism.** Confirm how classification, deviation, and
   provenance judgments will be supplied for this run (agent-as-judge via the
   packaged skill, or API-key-backed judges — see PLAN-FIRST.md) — do not
   assume; check `playbook_engine/` for the current judge seams before
   starting.
4. **Write a stage-by-stage plan** with explicit checkpoints: stage → lint →
   mine (deterministic backbone) → judge (classification / deviation /
   provenance) → project → validate → report/inspect. Identify where you will
   pause to sanity-check the backbone (trail order, signed-copy detection,
   provenance plausibility) *before* spending judgment effort on it — a bad
   backbone makes every downstream judgment worthless.
5. **Note the OPF version target.** Confirm whether the run targets the
   current OPF schema version and whether any section (e.g. Posture/Floor)
   requires a human interview rather than corpus-derived content — flag those
   as human-input-dependent rather than inventing them.

## Phase B — execute

Carry out the plan from Phase A. Checkpoint after `mine` and report backbone
health before spending judgment effort. Use `playbook judge --plan` to
estimate token cost and get human go/no-go before a full-corpus judgment pass;
consider a `--subset` trial first. Produce the final playbook, run `playbook
validate`, and render the inspection/report outputs for human review.

---

## Guardrails

- **Do not fabricate legal content.** Low-confidence judgments are flagged
  (`needs_review: true`), never guessed — especially provenance and entity
  aliases.
- **Real corpus content stays out of git.** Stage to a user-owned cache
  directory (`playbook stage` defaults to `~/.cache/playbook-engine/staging`),
  never to a world-readable shared path. Keep any local scratch files (staging
  configs, one-off adapters) out of the tracked repo.
- **Token cost is real.** Dedupe clauses by content hash before judging; judge
  changed hunks, not whole documents; cache verdicts. Estimate judgment volume
  and surface it before committing to a full-corpus pass.
- **Report honestly.** If a stage is stubbed or a judge seam isn't wired yet,
  say so in the report rather than presenting partial output as complete.
