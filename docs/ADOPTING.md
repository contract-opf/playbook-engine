# Adopting OPF: from a folder of deals to a playbook your team trusts

This is the adopter's path. Each stage is useful on its own; stop wherever
you like. Time-to-value is deliberately front-loaded — the first two stages
take minutes.

## Stage 0 — See it work (1 minute, no API key)

Run the [quickstart](../examples/README.md) on the committed synthetic
corpus. You end with a validating `playbook.opf.json` and — the part worth
pausing on — `playbook render-prompt`, which turns it into a system prompt
you can paste into any chat LLM next to a contract. That loop (corpus →
playbook → prompt → review) is the whole product in miniature.

## Stage 1 — Point it at your own agreements

**Gather one agreement type.** One playbook covers one type (MSAs, NDAs,
affiliation agreements, ...). You need, per deal, whatever versions you
have: ideally the signed copy plus the drafts exchanged along the way.
DOCX with tracked changes is the richest signal (it carries who proposed
each change); PDF and RTF work; scans work on the Docker runtime (OCR).

**Lay out the corpus.** The expected shape is one directory per deal
([docs/CORPUS-LAYOUT.md](CORPUS-LAYOUT.md)). Exported from a CLM or DMS in
some other shape? `playbook stage` proposes a `staging_plan.json` from the
files' contents and metadata — you review the plan, then execute it.
Nothing moves until you approve.

**Write the config.** Four decisions, one YAML:

```yaml
agreement_type: { id: msa, name: "Master Services Agreement" }
baseline: { template: template/our-msa-template.docx }   # or null
taxonomy: builtin:cuad-base.yaml       # or your own; see below
provenance:
  our_party_aliases: ["YourCo", "YourCo Holdings, LLC"]  # every name variant you sign under
  known_entities: ["Counterparty A Inc.", ...]           # enables born-safe pseudonymization
perspective: { party: "YourCo", counterparty_type: "Vendor" }
```

Two of these matter more than they look:

- `our_party_aliases` is how the engine tells your edits from theirs —
  it drives provenance detection and who-proposed-what attribution.
- `known_entities` activates the **born-safe rule**: counterparty names
  are replaced with stable aliases *at ingest*, so no stored artifact ever
  carries a raw name. The reverse map is written to a restricted sidecar
  that never enters the playbook. If you skip this, real names flow into
  the artifacts — configure it before the first real run.

**Taxonomy:** start with `builtin:cuad-base.yaml` (the genuine CUAD v1
41 categories) plus `builtin:general-commercial.yaml`, prune what doesn't
apply (`status: inactive` — never delete), or induce one from your own
corpus with `playbook induce-taxonomy` and curate the result.

**Compile:**

```sh
playbook lint-corpus ./corpus --config playbook.config.yaml   # catches layout problems first
playbook mine ./corpus --config playbook.config.yaml --out ./out
playbook project ./out --config playbook.config.yaml
playbook validate ./out/playbook.opf.json
```

The LLM-judgment stages run three ways — pick one
([docs/PLAN-FIRST.md](PLAN-FIRST.md)):

1. **Claude plan, no API key** (first-class path): the packaged
   `playbook-from-corpus` Claude Code skill drives the pipeline and acts
   as the judge interactively. One command from the repo root — and if
   you use this path, you can skip the config-writing above; the skill
   interviews you for it:

   ```sh
   claude "$(cat docs/prompts/create-playbook.md)"
   ```

   (Or just open the repo in Claude Code and say *"derive a playbook from
   my corpus"* — the skill's trigger phrases do the rest.)
2. **API key**: fully headless; batch-friendly.
3. **Stub judges** (default with no key): everything deterministic still
   runs; judgment-dependent fields are conservatively capped and the
   playbook is watermarked `stub_basis_present` so downstream consumers
   know not to trust it for real review.

## Stage 2 — Read what it found

```sh
playbook inspect ./out          # inspection report: coverage, confidence, flags
playbook view render ./out      # human-readable playbook walkthrough
```

Judge the output the way you'd judge an associate's memo: every position
cites `document / version / clause`. Follow a few citations —
`playbook resolve-citation` verifies you're holding the exact cited bytes
(sha256 content addresses). The negotiation trails are usually where teams
have the "oh, it really *does* know" moment: round-by-round, who asked for
what, and where it landed.

## Stage 3 — Make it yours (curation + posture)

- **Pin what the corpus gets wrong.** An attorney pin
  (`curation.pins[]`) overrides a compiled stance and *survives
  recompiles*; if fresh evidence later contradicts a pin, the engine flags
  the conflict deterministically instead of silently dropping either.
- **Run the posture interview.** A short structured interview (rounds,
  risk appetite, what's sacred, audience) becomes `posture.system_prompt` —
  your negotiation intent, grounded in the compiled evidence.
- **Author the Floor.** The compiler *proposes* candidates from every
  ask your history reversed; you decide which become invariants. Keep it
  minimal — the admission test in the spec (§3.7.1) exists so the Floor
  never regrows into a rigid per-clause script.

## Stage 4 — Use it

- **Anywhere, today:** `playbook render-prompt` → paste into any chat LLM
  with the contract under review. The prompt encodes the determinism
  boundary in plain language: red lines are non-negotiable, posture shapes
  judgment, evidence is cited history.
- **In tooling:** consume `playbook.opf.json` directly — it's stable,
  schema-validated JSON with content-addressed citations. The
  [bundle boundary](OPF-BUNDLE-BOUNDARY.md) doc says exactly what a
  review engine owns vs what the playbook owns.
- **Recompile on every new deal.** The playbook is not a trained model;
  "retraining" is re-running the compiler. Caches make incremental runs
  cheap, pins survive, and `corpus.snapshot.manifest_hash` records exactly
  which corpus state produced which playbook.

## Stage 5 — Share it (optional)

`playbook publish` produces a **party-anonymous** export: role-label
party ("the company" / "the counterparty"), quarter-coarsened dates,
numbered deal pseudonyms, and an LLM residue pass over every free-text
surface with an independent verify pass behind it. That's what makes it
plausible to publish a real playbook as an educational artifact — or just
to share one across teams — without shipping your counterparties' names.

## Why bet on the format?

Because it's the part designed to outlive everything else. OPF is
CC-BY-4.0 spec text plus a JSON Schema; documents carry their own
integrity (content hash, section digests) and their own provenance
(citations that resolve by sha256 against your corpus). Vendor needs go in
the reserved `x_*` namespace instead of forks. If this engine disappears
tomorrow, your playbook is still a self-describing, self-verifying record
of your negotiation knowledge — which is precisely the property you'd want
before investing your history in it.

Questions the spec doesn't answer? Open a
[spec question](../.github/ISSUE_TEMPLATE/spec-question.yml). Found a
problem? [CONTRIBUTING.md](../CONTRIBUTING.md) has the dev loop
(`make install`, `make all`).
