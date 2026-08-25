# playbook-from-corpus — Judge reference

Agent reference for acting as the LLM judge in the `playbook-from-corpus`
derivation loop. Contains judge prompts, guardrails, and machine-checkable
done-criteria.

---

## Judge prompts

Each item in `out/judge/pending.jsonl` has a `kind` field that determines
the judgment task.

### Classification (`kind: classify`)

**Input fields:** `text` (full clause text), `heading`, `taxonomy_ids`
(flat list of allowed ids — labels/descriptions live in the taxonomy YAML;
read it once and keep it at hand).

**Task:** Assign the best-fit `taxonomy_id` from `taxonomy_ids`. Return
`null` if no entry fits with reasonable confidence.

**Prompt (adapt as needed):**

> You are classifying a clause from a legal agreement. The clause text is:
>
> ---
> {text}
> ---
>
> Allowed taxonomy ids (see the taxonomy YAML for labels/descriptions):
> {taxonomy_ids}
>
> Assign the single best-fit taxonomy ID from the list above. If no entry
> fits with reasonable confidence, return null.
>
> Respond with JSON: `{"taxonomy_id": "<id or null>", "confidence": 0.0–1.0, "rationale": "..."}`

**Rules:**
- Confidence < 0.70: set `needs_review: true` in the verdict. **This is
  behavioral, not advisory:** 0.70 is `clause_classifier.AMBIGUITY_THRESHOLD`
  — a stored classify confidence below it trips
  `ClauseClassification.is_ambiguous` in the engine. A second, independent
  cutoff also applies downstream regardless of what you set here: the AAR's
  "needs attention" report separately flags any observation whose stored
  `confidence` is < 0.5 (`aar.py`'s hardcoded low-confidence check). Both
  read the same stored `confidence` field, so there is no reason to shade a
  genuine judgment below its true value — score honestly and let the
  thresholds do their job. (This `kind: classify` prompt is the drain-loop
  judge path. Step 2a's agent-segmentation classification is a separate
  path — no dedicated judge verdict — and always stamps a flat
  `confidence: 0.45` regardless of your actual certainty; the AAR aggregates
  that whole flat-confidence cohort into one spot-check line rather than
  flagging each individually, so it does not flood the "needs attention"
  table the way a genuine per-clause low score would — see issue #181.)
- Never invent a `taxonomy_id` not in the provided list.
- Prefer specificity: if multiple entries match, pick the most specific.

---

### Deviation assessment (`kind: deviation`)

**Input fields:** `hunk` (a `[BEFORE]`/`[AFTER]` version-to-version diff of
the clause), `our_standard` (our template language for this clause type —
empty string when no baseline template is configured), plus traceability
context: `taxonomy_id`, `clause_path`, `document_id`.

**Task:** Classify the deviation from the template hunk and assess risk.

**Prompt (adapt as needed):**

> You are assessing a clause deviation against our standard template language.
>
> Our standard (empty in emergent mode — then judge the [BEFORE]/[AFTER]
> change itself: did the negotiated edit move risk for us?):
> ---
> {our_standard}
> ---
>
> Negotiated change (version-to-version hunk):
> ---
> {hunk}
> ---
>
> Classify the deviation as one of:
> - `none` — substantively identical to the template; only formatting/style differences
> - `reworded_equivalent` — different wording, same legal effect
> - `substantive` — materially different legal obligation or right
>
> For `substantive`, also assess:
> - `risk_delta.direction`: `better` / `neutral` / `worse` (relative to our template)
> - `risk_delta.magnitude`: `none` / `minor` / `material` — these three
>   EXACTLY (engine enum `deviation_classifier._MAGNITUDE_VALUES`);
>   `moderate`/`major` are invalid and now rejected at `judge-apply`
>
> Respond with JSON: `{"deviation": "<none|reworded_equivalent|substantive>", "risk_delta": {"direction": "...", "magnitude": "..."}, "confidence": 0.0–1.0, "rationale": "..."}`

**Rules:**
- **Template mode vs emergent mode — what to compare:** when `our_standard`
  is NON-EMPTY, judge the AFTER text (or the single block) against OUR
  STANDARD — the question is "how far does the negotiated outcome sit from
  our template language for this clause type?", not "did this edit move
  risk?". Notes: `our_standard` may be the whole template document for
  preamble-ish items — compare against the portion relevant to this clause's
  subject (use `taxonomy_id` and the hunk's own heading as the guide); if the
  template genuinely has no language on the hunk's subject, judge the
  negotiated change itself (emergent style) and say so in the rationale.
  When `our_standard` is EMPTY, judge the [BEFORE]/[AFTER] change itself.
  A clause APPEARING only (no BEFORE): with a template, judge that single
  text against the standard; without one, judge whether accepting it as
  written is risk-relevant vs standard boilerplate.
- **Relocation triage FIRST (do this before judging items one by one):**
  **pair-scanning the pending set alone will NOT find most relocations.** A
  relocation has two sides: the clause disappearing (a hunk) and the same
  clause reappearing unchanged elsewhere. Only the disappearing side
  necessarily produces a pending item — an unchanged reappearance is an
  `unchanged`-kind diff, which is resolved deterministically and never
  reaches the judge, so it is simply absent from `pending.jsonl`. Scanning
  hunk pairs within the pending set catches only the minority of relocations
  where *both* sides happen to also be judge-routed hunks (near-identical
  rewrites, or a version where the clause also moved to a differently
  classified taxonomy slot). To find the rest: (1) group the pending
  deviation items by `document_id`; (2) for each item whose hunk shows a
  clause disappearing or appearing, use its `version_from`/`version_to`
  fields to open the adjacent per-version clause trees at
  `$OUT/normalized/<document_id>/NN__*.clauses.json` (each node carries
  `clause_path`/`heading`/`text`) and search the other version's node texts
  for the clause body, after normalizing whitespace/quotes/numbering;
  presence in both versions ⇒ alignment artifact. Auto-verdict BOTH sides of
  each confirmed relocation as `none` / neutral / none with rationale
  "alignment artifact — clause relocated, text unchanged" (confidence 0.8+,
  no needs_review). Only judge the remaining hunks individually. On real
  corpora one-third to as much as ~90% of deviation items can be relocation
  echoes (a real 44-agreement run saw 143/163, ~88%); judging them blind
  wastes effort and floods the report with needs_review flags.
  As of issue #167, `mine`'s deviation-classifier stage runs exactly this
  containment check itself before an added/removed hunk ever reaches
  `pending.jsonl` — `basis: "alignment"` in `observations.jsonl` marks a
  clause the engine already auto-verdicted this way. This manual pass is
  now only needed for what the engine's check misses: fuzzy relocations
  (reworded during the move, not verbatim) and moved-then-merged text the
  normalizer's plain substring test doesn't catch.
- Confidence < 0.65: set `needs_review: true`. **This is advisory only, not
  behavioral:** unlike the classify and provenance thresholds above, no
  engine constant enforces a deviation confidence cutoff — deviation
  confidence is never persisted to `observations.jsonl` and no downstream
  reader (AAR, compiler, viewer) gates on it, so 0.65 is a calibration
  target for your own judgment, not a value the pipeline will act on. Flag
  genuine uncertainty anyway — a human reading the report benefits even
  though the engine does not.
- ALWAYS include `risk_delta` — for `none` / `reworded_equivalent` use
  `{"direction": "neutral", "magnitude": "none"}` (omitting it fails
  apply-time validation; the neutral direction requires magnitude `none`).

---

### Provenance (`kind: provenance`)

**Input fields:** `preamble` (opening recital block), `letterhead`
(title/heading block), `agreement_type`. The payload does NOT carry the
known-alias list — read `provenance.our_party_aliases` from
`playbook.config.yaml` yourself before judging provenance items.

**Task:** Determine whether this document originated from our paper or the
counterparty's paper.

**Prompt (adapt as needed):**

> You are determining which party drafted the original version of this
> agreement by reading its recital/opening section.
>
> Our known names/aliases: {our_party_aliases from playbook.config.yaml}
>
> Agreement opening (recital):
> ---
> {preamble}
> ---
>
> Determine provenance:
> - `our_paper` — we drafted the original; our standard language is the base
> - `counterparty_paper` — the counterparty drafted the original; their language is the base
>
> Signals to look for:
> - Which party's name appears in the "agreement template" or "standard form" reference?
> - Which party's address block appears first (in US contracts, the drafting party often appears first)?
> - Does the recital say "our form", "standard agreement", "template provided by"?
> - Indemnification structure: our paper typically protects us first.
>
> If the recital does not provide enough signal, return `needs_review: true`
> rather than guessing.
>
> Respond with JSON: `{"provenance": "<our_paper|counterparty_paper>", "confidence": 0.0–1.0, "rationale": "...", "needs_review": false}`

**Rules:**
- Unknown entity name (a party name in the recital not in
  `provenance.our_party_aliases`): record the alias in `rationale`, set
  `needs_review: true`. Do not silently assume it is us.
- Confidence < 0.7: set `needs_review: true`. **Calibration matters:** the
  engine SILENTLY flips any stored provenance verdict with confidence below
  0.70 to `counterparty_paper` at mine time (`pipeline.py` ambiguity rule)
  and never re-queues it — a correct `our_paper` verdict at 0.55 is
  discarded corpus-wide. When the recital evidence is real, say so with
  confidence ≥ 0.70; reserve sub-0.70 for genuine uncertainty.
- Conservative default when genuinely uncertain: `counterparty_paper` (the
  safe choice — it attributes less favorable positions to the counterparty,
  not to us).

---

## Verdict format

Each line written to the verdicts JSONL file (for `playbook judge-apply`):

```json
{"key": "<sha256-hex>", "verdict": { ... }}
```

The `key` is the SHA-256 from the `pending.jsonl` item. The `verdict` schema
depends on kind.

**`basis` is NOT one shared enum — each judge type has its own, and they do
NOT all include `"llm"`.** Using the wrong value is rejected at
`playbook judge-apply`, with the offending line number, before it ever
reaches the store: `validate_verdict` reconstructs the exact dataclass the
store-backed judge would build on replay (`deviation` accepts only
`{"judge"}`, `classify` accepts `{"judge", "unclassified"}`, `provenance`
builds a `ProvenanceResult` that rejects `"judge"` — see `agent_judge.py`'s
`_DEVIATION_REPLAYABLE_BASES` / `_CLASSIFY_REPLAYABLE_BASES`) and raises on
the first mismatch. The silent-requeue-then-late-`project`/`validate`-failure
chain this section used to describe is exactly what that apply-time
validation (issue #182) exists to prevent; it can still happen for a verdict
written straight into the store by hand (bypassing `judge-apply`) or banked
before the #182 hardening — and even then `playbook judge` surfaces the
re-queue loudly, as a `WARNING`, rather than silently. Use exactly the value
shown for each kind below:
- Classification (`ClauseClassification.basis`, `clause_classifier.py`) — use
  `"judge"` for an agent-produced verdict that found a taxonomy fit, or
  `"unclassified"` (with `taxonomy_id: null`) for a producer-supplied
  no-fit verdict — both are replayable (`_CLASSIFY_REPLAYABLE_BASES`).
  `exact_match` / `heading_similarity` / `judge_error` / `needs_review` /
  `llm_segmenter` are set by the engine itself, not by you, and are rejected.
- Deviation (`DeviationResult.basis`, `deviation_classifier.py`) — use
  `"judge"` for an agent-produced verdict (also accepts `deterministic` /
  `reworded_equivalent` / `judge_error` / `needs_review`, set by the engine).
- Provenance (`ProvenanceResult.basis`, `provenance_detector.py`) — use
  `"llm"` — this is the one kind where `"llm"` is correct; `validate_verdict`
  rejects any other basis from a producer-supplied verdict (deterministic
  bases like `template_similarity` / `alias_first_party` / `hint` are set by
  the engine itself, not by you).
- Scope — no `basis` field; it is forced to `"judge"` on replay.

**Classification verdict:**
```json
{
  "taxonomy_id": "indemnification",
  "confidence": 0.88,
  "basis": "judge",
  "rationale": "Clause defines mutual indemnification obligations.",
  "needs_review": false
}
```

**Deviation verdict:**
```json
{
  "deviation": "substantive",
  "risk_delta": {"direction": "worse", "magnitude": "minor"},
  "confidence": 0.75,
  "basis": "judge",
  "rationale": "Counterparty caps indemnification at contract value; template is uncapped.",
  "needs_review": false
}
```

**Provenance verdict:**
```json
{
  "provenance": "counterparty_paper",
  "confidence": 0.82,
  "basis": "llm",
  "rationale": "Recital references 'University Standard Agreement Form'.",
  "needs_review": false
}
```

**Scope verdict** (`kind: scope`) — one per document; decide whether the
document is an instance of this agreement type. `in_scope` is required;
`scope_rationale` and `scope_confidence` are optional (`basis` is forced to
`"judge"` on replay). Payload gives `agreement_type_id`, `document_id`, and the
document's `clause_heads`.
```json
{
  "in_scope": true,
  "scope_confidence": 0.95,
  "scope_rationale": "Educational-institution affiliation agreement for student internships."
}
```
An out-of-scope document (`in_scope: false`) is retained but excluded from the
playbook. Deviation `risk_delta` must obey its invariant: `direction: "neutral"`
requires `magnitude: "none"` — a mismatch is rejected (and, per issue #182, is
isolated to that one clause rather than quarantining the whole batch).

### SegNode (agent segmentation — `segment` / `segment-apply`, issue #191)

Each `segment/pending.jsonl` item is one *version* of one document (a
5-version document queues 5 items; `document_id`/`version` identify which),
and gives that version's `canonical_text`, its `blocks`
(`{block_id, page, char_span, text}`), and the allowed `taxonomy_ids`.
Partition the blocks into contiguous clause ranges — one `SegNode` per clause —
and write one verdict line per pending item to the verdicts JSONL (identical
version texts across documents dedup by content hash, so you only need one
line per distinct `canonical_text`):

```json
{
  "canonical_text": "<echoed verbatim from the pending item>",
  "nodes": [
    {"node_id": "n1", "parent_id": null, "order": 1, "heading": "Recitals",
     "taxonomy_id": "parties_and_recitals",
     "start_block_id": "b0", "end_block_id": "b3",
     "start_quote": "", "end_quote": ""},
    {"node_id": "n2", "parent_id": null, "order": 2, "heading": "Indemnification",
     "taxonomy_id": "indemnification",
     "start_block_id": "b4", "end_block_id": "b9", "start_quote": "", "end_quote": ""}
  ]
}
```

**Rules:**
- Cover **every** block exactly once — clause ranges are contiguous and
  partition `b0`..`b<last>` (the coverage/reconstruction gates enforce this).
- `parent_id: null` for top-level clauses; set it to a parent `node_id` to nest
  (dotted clause paths are derived from the tree). `order` is 1-based within a
  parent.
- `taxonomy_id` must be from the item's `taxonomy_ids`, or `null` if no entry
  fits — this doubles as first-pass classification, so the judge loop then has
  no `classify` items.
- `start_quote`/`end_quote` may be `""` for block-aligned clauses; the range is
  reconstructed from the block span. (Splitting *within* a block would require
  exact boundary quotes — the live-LLM segmenter's job, not this path.)

---

## Rubric versions

Verdicts are cached by a hash of the clause **content**, so nothing about a
changed *rubric* moves the key. Without a version, edits to the criteria
below would replay every previously banked verdict forever, silently.

Each stored verdict therefore carries a stamp — `{"rubric": {"kind": ...,
"version": "<manual>+<derived>"}}` — recording the rubric it was produced
under. `playbook judge` compares it against the rubric in force:

| state | meaning | behaviour |
|-------|---------|-----------|
| current | stamp matches | replays |
| stale | stamp differs | **re-queued for re-judgement** (`--accept-stale` to replay) |
| legacy | no stamp (banked before versioning) | replays, reported every run until migrated (`--strict-rubric` to re-queue instead) |

**The manual half** is `RUBRIC_PROMPT_VERSIONS` in
`playbook_engine/rubric.py`, one entry per kind. Bump the entry for a kind
when the corresponding **Judge prompts** section above changes in a way that
could change a reasonable judge's answer: a new or removed answer category, a
reversed default, a changed definition of "material" or "substantive". Do
**not** bump for typo fixes, reworded examples, or added guardrails that only
restate existing rules — that would discard thousands of sound verdicts.

**The derived half** needs no discipline; it moves on its own when the
machine-readable rubric moves: the classifier-eligible taxonomy entries
(id + label + description) for `classify`, the agreement-type definition for
`scope`, the answer vocabularies for `deviation` and `provenance`. Editing
`spec/taxonomy/*.yaml` re-queues classify verdicts and nothing else.

Migrating an existing store:

```bash
playbook judge-migrate ./out --config ./corpus/playbook.config.yaml --dry-run
playbook judge-migrate ./out --config ./corpus/playbook.config.yaml
```

The first reports how many verdicts are current / legacy / stale. The second
adopts the **legacy** ones — stamping them with the current rubric while
leaving the judgments untouched, so banked human work is preserved rather
than re-queued. Known-stale verdicts are deliberately left alone unless you
add `--accept-stale` (optionally scoped with `--kind`), which is an explicit
assertion that the rubric change does not affect those answers. Re-stamping
appends to `verdicts.jsonl`; the prior record stays as an audit trail.

**This is not unconditionally safe.** Adopting a legacy verdict stamps it
with the rubric *currently* in force for its kind — it does not ask whether
that verdict was actually produced under an equivalent question. Run
`--dry-run` first, and only adopt once you have satisfied yourself the
rubric has not moved under those verdicts since they were banked.

**Concrete rule for `deviation`:** issues #117/#118/#119 changed extraction
and tracked-change attribution, so the `[BEFORE]`/`[AFTER]` hunks the
deviation judge reasons over are no longer assembled the way they were for
verdict stores produced before that landed (July-2026 runs). Those legacy
deviation verdicts are answers to a materially different question than
today's rubric asks (see `RUBRIC_PROMPT_VERSIONS` in
`playbook_engine/rubric.py`). For a store whose deviation verdicts predate
#117, do **not** run unscoped `judge-migrate` — it will blank-stamp them as
current. Scope adoption to the other kinds instead and let `deviation`
re-queue for real re-judgement:

```bash
playbook judge-migrate ./out --config ./corpus/playbook.config.yaml \
  --kind classify --kind provenance --kind scope
```

---

## Guardrails

1. **No fabrication.** Every verdict must be grounded in the actual clause
   text or recital. Do not invent legal interpretations.

2. **Flag, don't guess.** When confidence is below threshold, set
   `needs_review: true` and include the uncertainty in `rationale`. **This
   flag is not read back anywhere today** — `ClauseClassification`,
   `DeviationResult`, and `ProvenanceResult` have no `needs_review` field,
   replay reconstruction (`agent_judge.py`) discards the stored boolean when
   it rebuilds a verdict from the store, and the after-action report's Needs
   Attention section never joins back to `verdicts.jsonl` to read it. Setting
   it is still worth doing — it is a durable, human-auditable record in
   `verdicts.jsonl` for a future manual pass over the store — but do not tell
   a reviewer, or assume yourself, that flagging a doubtful verdict here
   causes it to surface in the report. The only signal Needs Attention
   currently derives from a judged clause is the compiled observation's
   **classification** confidence dropping below 0.5 (`aar.py`); deviation and
   provenance confidence are never copied onto an observation at all, so no
   threshold on those ever reaches the report. If a low-confidence deviation
   or provenance call genuinely needs a human's eyes before this round ships,
   say so directly — in the round summary you give the human running this
   skill — rather than relying on `needs_review: true` to do that for you.

3. **Unknown aliases.** If a party name in the corpus is not in
   `provenance.known_entities`, record it explicitly in `rationale` and set
   `needs_review: true`. Supply a curated alias list to the human before the
   next round.

4. **`needs_review` is an internal flag only.** It must be resolved (by human
   review or re-judgment) before `project`. The OPF observation schema enum is
   `{none, reworded_equivalent, substantive}` — `needs_review` is not a valid
   `deviation` value.

5. **Deduplication.** `playbook judge` deduplicates by content hash. Judge
   each unique clause payload once; verdicts propagate automatically to all
   documents sharing that clause.

6. **Corpus confidentiality.** Real agreement text is private. Do not log,
   echo, or store clause text outside the local `out/` directory.

7. **Posture / Floor fields.** `posture.system_prompt` (Posture section) and the
   walk-away floor (`floor.invariants`, Floor section) require the GC interview
   (OPF-SPEC.md §7) and cannot be derived from the corpus — never invent them.
   (`historical_stance` is different: it lives in the Evidence section, is
   purely descriptive — OPF-SPEC.md §2.2 — and is exactly what compiles
   straight from the corpus; do not treat it as interview-gated.) With the human
   present, run SKILL.md Step 7a (interview) and Step 7b (floor propose → sign);
   without them, list both as pending human input in the report and say that an
   evidence-only playbook is a complete document (Rung 0), not an unfinished one.
   Q4 `sacred_clauses` is written straight into signed `floor.invariants` (the
   human authored it) **only when the item is a bare clause-type name**;
   compiler-derived candidates require an explicit accept round-tripped
   through `playbook view apply`. A sentence-shaped or conditional item
   either of those templates would garble ("X, if present, must not be Y";
   more than 7 words; or containing "if"/"unless"/"must"/"shall"/"provided",
   issue #104) is skipped from promotion — the interview prints a WARN
   instead of writing it — and goes through
   `playbook floor sign --statement "..." --signed-by "<name>"`
   instead, verbatim — never by hand-editing `floor.invariants`. `--signed-by`
   is required: it names the human legal owner signing off, recorded as a
   structural `x_signed_by` field the command refuses to omit — get an
   explicit in-chat confirmation of the exact statement before running it.

---

## Done-criteria (machine-checkable)

The derivation is **done** when all four conditions hold:

1. **`out/judge/pending.jsonl` exists and is empty.**

   Absence does **not** count as done — `playbook judge` unlinks
   `pending.jsonl` at the start of every round and only writes it lazily as
   items are queued, so a round killed mid-`mine` (e.g. an overnight run that
   died) also leaves the file absent. A completed round with 0 new pending
   items always writes an explicit empty file, so `-f` (must exist) is what
   distinguishes "finished, nothing pending" from "interrupted before
   finishing" (issue #170).

   ```bash
   # Confirm: file exists AND is empty
   [ -f ./out/judge/pending.jsonl ] && [ ! -s ./out/judge/pending.jsonl ]
   echo "Exit: $?"   # must be 0
   ```

2. **`playbook validate` exits 0.**

   ```bash
   playbook validate ./out/playbook.opf.json
   echo "Exit: $?"   # must be 0
   ```

3. **Report, viewer, and the two packaged artifacts are generated**
   (`report.md`, `report.json`, `playbook.review.html`, `playbook.opf.html`,
   `playbook.digest.json` exist in `out/`).

   SKILL.md Step 9 names `playbook.opf.html` as the **packaged
   internal/stakeholder playbook** and `playbook.digest.json` as the
   model-facing digest — both are produced before Step 10's report runs, on
   every route (A, B, and C all reach Steps 9 and 10). A run that stops after
   `report`/`inspect` without `view bundle`/`digest` is not done: the GC has
   the internal annotation surface (`playbook.review.html`) but not the
   artifact meant to leave the room. `playbook.opf.html` is **not** a
   guarantee of pseudonymization on its own — SKILL.md Step 9's mandatory
   residue check must be run before treating it as shareable, and a
   genuinely external release must go through Step 11 (`publish`)'s hard,
   list-independent backstop; `playbook.opf.html` is not the "hand it to
   anyone" artifact it used to be described as.

   ```bash
   test -f ./out/report.md && test -f ./out/report.json \
     && test -f ./out/playbook.review.html && test -f ./out/playbook.opf.html \
     && test -f ./out/playbook.digest.json
   echo "Exit: $?"   # must be 0
   ```

4. **`out/quarantine.json` is empty, or every entry has been explicitly
   triaged.**

   A document quarantined during `mine` (SegmentationQAError, HintsError,
   all-versions-failed-ingest) contributes no `pending.jsonl` items, so
   criteria 1–3 above can all hold while a whole agreement's negotiation
   history is silently absent from the evidence — the only trace is a line
   in the report's Needs Attention section (see `aar.py`'s
   `_load_quarantine`), which is prose, not a gate. `quarantine.json` is
   rewritten (even when empty) every `mine` run, so it always reflects the
   current state, not a stale prior round.

   ```bash
   PATH="$PWD/.venv/bin:$PATH" .venv/bin/python -c \
     'import json,sys; q=json.load(open("./out/quarantine.json")); sys.exit(1 if q else 0)'
   echo "Exit: $?"   # 0 = no quarantined documents
   ```

   A non-zero exit does **not** by itself mean the derivation is unfinished
   — it means every entry must be triaged before you call it done: re-segment
   the document (fix `hints.yaml`, adjust extraction) and re-run `mine`, or
   explicitly accept the exclusion by naming the `document_id` and reason in
   your round summary to the human running this skill (the report's Needs
   Attention section is entirely tool-generated by `playbook report` and
   cannot itself record a human decision). Never treat an untriaged
   `quarantine.json` entry as background noise the loop can converge past.

Items that remain for human review may be **listed in the report** under
"Needs Attention" — they do not block the done-criteria above, but the
signals that actually land there must not be silently suppressed. Per
Guardrail 2 above, Needs Attention today only derives from: quarantined
documents (`_load_quarantine`, see criterion 4 above), a compiled
observation's **classification** confidence below 0.5 (`aar.py`), genuinely
unresolved/unjudged pending items, failed version ingests, and a
`corpus_manifest.json`-vs-`playbook.opf.json` corpus-count disagreement — the
signature of an `out_dir` that mixes artifacts from two different runs, e.g.
a non-atomic backup copy (`aar.py`'s manifest-vs-playbook check). Setting
`needs_review: true` on a deviation or provenance verdict is **not** among
these — it is not currently copied onto an observation, so it does not by
itself reach the report; if such a call needs a human's eyes before this
round ships, say so directly in the round summary instead.

---

## Common failure modes

| Symptom | Cause | Fix |
|---------|-------|-----|
| `validate` exits non-zero after `project` | Residual `needs_review` or malformed `deviation` value in store | Drain the judge loop; fix malformed verdicts |
| `pending.jsonl` does not shrink between rounds | Verdicts not applied or wrong `key` | Check `judge-apply` output; confirm keys match |
| All provenance = `counterparty_paper` | Recitals not loaded or entity alias list empty | Supply `provenance.our_party_aliases`; re-run provenance round |
| All clauses unclassified (`taxonomy_id: null`) | Taxonomy mismatch with document content | Check taxonomy covers the agreement type; refine taxonomy entries |
| Trail ordering wrong | No `order:` hint; version-orderer used greedy fallback | Add explicit `order:` list to `hints.yaml`; re-run `mine` |
