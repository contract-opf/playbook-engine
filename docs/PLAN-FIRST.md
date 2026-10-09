# Plan-first: what needs an API key, and what runs on your Claude plan

**Anthropic-native, with honest docs about what needs a key.** The
`playbook-from-corpus` skill (an operator's Claude Code plan) is the
first-class way to derive a playbook — not `ANTHROPIC_API_KEY` billing.
API keys are for headless/batch operation: LLM-first segmentation
(`llm_segmenter.py` / `llm_segmenter_batch.py`) calls the Anthropic API
directly and always requires one. Everything else in the pipeline —
including every judgment stage — is either fully deterministic or is
performed by the agent running the skill, reasoning on your existing
Claude plan, with no API key involved.

This table is the stage-by-stage source of truth. If a stage isn't listed
here, assume it's deterministic and needs no LLM at all.

| Stage | Needs `ANTHROPIC_API_KEY`? | Runs on your Claude plan via the skill? | Notes |
|---|---|---|---|
| `playbook stage` | No | N/A (deterministic) | Flattens a nested export layout (e.g. CLM `Versions/` folders) and writes `hints.yaml`. No LLM involved. |
| `playbook lint-corpus` | Only to *check for* the key | N/A (deterministic) | The documented preflight tool: if `segmentation.llm` is on in config and `ANTHROPIC_API_KEY` is unset, this fails loud here — before `mine`/`judge` would, and before extraction has ground through the corpus. |
| `playbook mine` — default (deterministic segmentation) | No | N/A (deterministic) | The default L1–L4 skeleton (ingest, scope gate, classification, alignment). No LLM calls, no token spend. |
| `playbook mine` — `segmentation.llm: true` | **Yes** — real token spend | No | Config-gated opt-in. Every document version is sent to the LLM segmenter (direct Anthropic API call, optionally batched via Message Batches with `batch: true`). There is no deterministic fallback once enabled — an LLM error fails the run rather than silently degrading. This is the one stage that cannot run on a Claude plan alone; it needs `ANTHROPIC_API_KEY` in the environment. |
| `playbook judge` / `playbook judge-apply` | No | **Yes** | The judgment core. The skill's agent reads `out/judge/pending.jsonl`, makes each scope/classification/provenance call by reasoning directly (on the operator's own Claude Code plan), and writes a verdicts JSONL that `judge-apply` records. No API key, no direct Anthropic API call — the model doing the judging *is* the Claude Code session running the skill. **Deviation is not judged:** every clause's deviation is the deterministic standard check (does its text match our template clause?), so nothing is ever queued for it. |
| `playbook project` | No | N/A (deterministic) | Compiles L5 (playbook assembly) deterministically from the observation store built by `mine`/`judge-apply`. |
| `playbook validate` | No | N/A (deterministic) | JSON Schema validation of the output `playbook.opf.json` against `spec/playbook.schema-0.5.json`; any other `opf_version` is rejected as unsupported. |
| `playbook inspect` | No | N/A (deterministic) | Renders the mined trail/observations as a Markdown checkpoint report (version order, signed copy, provenance, flags). |
| `playbook view bundle` | No | N/A (deterministic) | A deterministic HTML build of `index.html`: the one human-readable artifact (five tabs, an optional editor that saves `overrides.json`). |
| `playbook floor propose` / `playbook floor sign` | No | N/A (deterministic) | `propose` derives Floor candidates from `outcome: proposed_then_reversed` observations plus the Posture interview's sacred-clauses answer; `sign` records a human-authored hard line verbatim. Pure derivation and I/O, no LLM. |
| Docker image publish (`.github/workflows/docker-publish.yml`) | No | N/A | A maintainer/CI-only release step, unrelated to running a derivation over your corpus. |

## The short version

- **One stage needs an API key: LLM-first segmentation** (`segmentation.llm: true` in `playbook.config.yaml`), because it makes direct Anthropic API calls (optionally batched) outside of any Claude Code session.
- **Every judgment stage — the expensive, high-value LLM work — runs on your existing Claude plan** through the `playbook-from-corpus` skill (`.claude/skills/playbook-from-corpus/`), with the agent acting as the judge. No API key, no per-token billing.
- **Deviation is not a judgment stage.** The playbook supplies precedent — whether each deal signed our standard text, and what was refused — and the consumer (the review model reading the playbook) does the judging.
- **Everything else is deterministic:** staging, linting, projection, validation, inspection, the bundle, and Floor proposals never touch an LLM at all.

If you hit the `ANTHROPIC_API_KEY` preflight error from `lint-corpus` or from
`mine`/`judge`, it means `segmentation.llm` is on in your config —
either set `ANTHROPIC_API_KEY` to run it live, or drop back to the
deterministic segmenter (remove/disable `segmentation.llm`) and run the rest
of the pipeline, including all judgment, through the skill on your Claude
plan.
