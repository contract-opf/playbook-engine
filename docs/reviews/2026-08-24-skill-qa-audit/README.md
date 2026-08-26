# Skill QA Audit — 2026-08-24

Comprehensive three-layer QA audit of the `playbook-from-corpus` skill:
internal consistency, doc-vs-implementation, and doc-vs-reality against a
real corpus run and the Contract Toaster consumer. Ten parallel review
workstreams plus an adversarial verification pass; 123 candidate findings,
122 confirmed, 1 refuted.

## Contents (public-safe, scrubbed)

- [`findings.md`](findings.md) — all 122 confirmed findings, grouped by
  area, severity-ordered, each linking its GitHub issue.
- [`report.html`](report.html) — the full report: verdict, findings,
  inconsistency register (65 rows), producer/consumer seam analysis,
  prioritised roadmap, coverage map, strengths.

Every finding was filed as a self-contained GitHub issue for the autonomous
fix loop: label **`skill-qa-audit`** — 90 issues on this repo (#123–#212),
5 on `contract-opf/contract-toaster` (#13–#17). Run progress: the "AFK run
log" issue (#213).

## What is deliberately NOT here

The raw findings appendix (per-finding adversarial-verifier notes) and the
machine-readable `audit.json` quote real counterparty names from the private
corpus, so they are local-only, in the repo checkout's gitignored
`reviews/2026-08-24-skill-qa-audit/` directory alongside the report/issue
generator scripts. This split is intentional: nothing under `docs/` may
carry corpus content. The two copies here were mechanically screened against
a counterparty-identifier blocklist before being written.

Evidence line numbers reference `main` at 500e0aa (2026-08-24); the fix
wave that followed (label `skill-qa-audit`, 2026-08-25/26) moves them.
