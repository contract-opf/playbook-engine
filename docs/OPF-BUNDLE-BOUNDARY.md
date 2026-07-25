# OPF ↔ Bundle Ownership Boundary

**Status:** Current. Supersedes the earlier converter framing.
(#NNN references cite the project's original development tracker.)

## The canonical-format decision

**OPF v0.2 is the canonical playbook format.** There is no second, parallel
"review-engine playbook format" that OPF must be converted into. What a review
engine installs and runs is **an OPF document plus a
thin bundle wrapper** — not a distinct schema requiring a field-by-field
converter.

#115 proposed building a converter (OPF → review-engine-playbook skeleton) as a
bridge between "two formats." That framing is superseded: once OPF v0.2
carries everything a playbook needs to *know* — Evidence with its
negotiation dynamics, Posture prose, the Floor, `agreement_type` matching,
`perspective`, `de_minimis` (§3.5–§3.7 of `OPF-SPEC.md`) — there is
nothing left for a converter to invent. The remaining gap between an OPF
document and a runnable review is not a format-mapping problem; it is the
bundle wrapper described below, which is a packaging concern, not a
knowledge-representation one.

## What OPF owns

OPF is the single source of negotiation **knowledge and intent** for an
agreement type. Concretely, OPF owns:

- **Knowledge** — the compiled, cited record of what the corpus shows:
  `evidence.clauses`, `evidence.clause_library`, `historical_stance`,
  `acceptable_if`, `fallbacks`, `rejected` (§3.5).
- **Intent** — the negotiation posture: `posture.system_prompt`, generated
  from the compiler interview and grounded in Evidence (§3.6).
- **Red lines** — the deterministic floor: `floor.invariants`, the
  things that must never slip regardless of model judgment (§3.7).
- **Perspective** — whose side the playbook is reviewed from: the
  top-level `perspective` object (`party`, `counterparty_type`).
- **De minimis** — the `de_minimis` list of change categories accepted even
  when technically novel. This is negotiation knowledge, not a runtime
  policy, so it lives in OPF.

There is deliberately no structured accept/reject condition list alongside
the Posture prose: structured decision conditions are exactly the
prescriptive style v0.2 retired. Prose Posture, the Floor's invariants, and
cited Evidence are the interface a consumer reasons from.

If a question is "what do we know, or what do we want, about this
agreement type" — the answer lives in OPF, full stop.

## What the bundle owns

The consuming app's release bundle owns everything about **how a
specific deployment runs** a playbook, none of which is agreement-type
knowledge:

- **Model policy** — which model(s) serve review, prompting/runtime
  configuration, retrieval and threshold settings. OPF §1 non-goals: "OPF
  does not encode the review engine's thresholds, retrieval, or model
  policy."
- **Output / leakage contract** — the shape of what the review engine is
  allowed to say back to a user, redaction rules, and any
  audience-specific leakage constraints on top of what Posture already
  targets (`OPF-SPEC.md` §3.6 covers *who* the output is for;
  the bundle covers *what is and isn't allowed to leave the box*).
- **Release** — versioning, sign-off, activation, rollback, and
  quarantine of what's actually running in production. Per
  `OPF-SPEC.md` §8: "Rollback and quarantine operate on the
  **bundle** (all sections + bound standard form + model policy), not on
  a single section in isolation."

The bundle wraps an OPF document; it does not re-derive or restate the
knowledge OPF already carries. A bundle that duplicates Evidence/Posture/
Floor content instead of referencing the OPF document it wraps has
drifted from this boundary.

## Summary table

| Concern | Owner | Where it lives |
|---|---|---|
| Corpus-derived evidence | OPF | `evidence` (§3.5) |
| Negotiation intent | OPF | `posture.system_prompt` (§3.6) |
| Red lines / hard rejections | OPF | `floor` (§3.7) |
| Whose side we review from | OPF | `perspective` |
| Accepted-even-if-novel changes | OPF | `de_minimis` |
| Model / retrieval / threshold policy | Bundle | consumer's release governance |
| Output shape / leakage contract | Bundle | consumer's release governance |
| Version, sign-off, activation, rollback | Bundle | consumer's release governance |

## Cross-references

- `docs/OPF-SPEC.md` §1 (non-goals), §5 (determinism boundary),
  §6 (producer/author/consumer responsibilities), §8 (governance &
  lineage) — the normative spec text this document summarizes.
- #115 — the converter-framing issue this document supersedes.
- #139 — added the v0.2 schema fields (`perspective`, `de_minimis`) this
  boundary assumes.
