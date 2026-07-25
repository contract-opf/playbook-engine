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
| `playbook lint-corpus` | Only to *check for* the key | N/A (deterministic) | The documented preflight tool: if `segmentation.llm` is on in config and `ANTHROPIC_API_KEY` is unset, this fails loud here — before `mine`/`compile`/`judge` would, and before extraction has ground through the corpus. |
| `playbook mine` — default (deterministic segmentation) | No | N/A (deterministic) | The default L1–L4 skeleton (ingest, scope gate, classification, alignment). No LLM calls, no token spend. |
| `playbook mine` — `segmentation.llm: true` | **Yes** — real token spend | No | Config-gated opt-in. Every document version is sent to the LLM segmenter (direct Anthropic API call, optionally batched via Message Batches with `batch: true`). There is no deterministic fallback once enabled — an LLM error fails the run rather than silently degrading. This is the one stage that cannot run on a Claude plan alone; it needs `ANTHROPIC_API_KEY` in the environment. |
| `playbook judge` / `playbook judge-apply` | No | **Yes** | The judgment core. The skill's agent reads `out/judge/pending.jsonl`, makes each classification/deviation/provenance call by reasoning directly (on the operator's own Claude Code plan), and writes a verdicts JSONL that `judge-apply` records. No API key, no direct Anthropic API call — the model doing the judging *is* the Claude Code session running the skill. |
| `playbook project` | No | N/A (deterministic) | Compiles L5 (playbook assembly) deterministically from the observation store built by `mine`/`judge-apply`. |
| `playbook validate` | No | N/A (deterministic) | JSON Schema validation of the output `playbook.opf.json` against `spec/playbook.schema-0.2.json`. |
| `playbook report` | No | N/A (deterministic) | Renders the after-action report (coverage, backbone health, judgment economics) from stored data. |
| `playbook view render` / `playbook view apply` | No | Partially | `render` is a deterministic HTML build. `apply` ingests a reviewer's exported `feedback.json`; re-judging any new pending items that surfaces is, again, a skill/agent judgment step, not an API call. |
| `playbook curate` | No | Yes (informally) | A deterministic command grammar (`pin ... to ...`, `note ...`) — not LLM-driven parsing inside the engine. An operator, or the skill's agent translating a natural-language request into this grammar, can issue commands; the engine itself never calls an LLM here. |
| `playbook floor propose` | No | N/A (deterministic) | Derives Floor candidates from `outcome: proposed_then_reversed` observations plus the Posture interview's sacred-clauses answer. Pure derivation, no LLM. |
| `playbook render-prompt` | No | N/A (deterministic) | Composes Evidence+Posture+Floor into a review-ready Markdown prompt. No API calls — it's meant to be pasted into *any* chat LLM (Anthropic or otherwise) alongside a contract to review; that downstream review conversation is out of scope for this repo. |
| Docker image publish (`.github/workflows/docker-publish.yml`) | No | N/A | A maintainer/CI-only release step, unrelated to running a derivation over your corpus. |

## The short version

- **One stage needs an API key: LLM-first segmentation** (`segmentation.llm: true` in `playbook.config.yaml`), because it makes direct Anthropic API calls (optionally batched) outside of any Claude Code session.
- **Every judgment stage — the expensive, high-value LLM work — runs on your existing Claude plan** through the `playbook-from-corpus` skill (`.claude/skills/playbook-from-corpus/`), with the agent acting as the judge. No API key, no per-token billing.
- **Everything else is deterministic:** staging, linting, projection, validation, reporting, viewing, curating, and Floor proposals never touch an LLM at all.

If you hit the `ANTHROPIC_API_KEY` preflight error from `lint-corpus` or from
`mine`/`compile`/`judge`, it means `segmentation.llm` is on in your config —
either set `ANTHROPIC_API_KEY` to run it live, or drop back to the
deterministic segmenter (remove/disable `segmentation.llm`) and run the rest
of the pipeline, including all judgment, through the skill on your Claude
plan.
