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
  `ClauseClassification.is_ambiguous` in the engine. There is no reason to
  shade a genuine judgment below its true value — score honestly and let the
  threshold do its job. (This `kind: classify` prompt is the drain-loop
  judge path. Step 2a's agent-segmentation classification is a separate
  path — no dedicated judge verdict — and always stamps a flat
  `confidence: 0.45` regardless of your actual certainty, marking that whole
  cohort as one sample to spot-check — see issue #181.)
- Never invent a `taxonomy_id` not in the provided list.
- Prefer specificity: if multiple entries match, pick the most specific.

---

### Deviation — not a judge kind

There is no deviation item to judge and no deviation verdict to write: every
clause's deviation is the deterministic standard check (its text matches our
template clause → `none`, otherwise `substantive`, `basis: "deterministic"`).
The consumer (the review model) does the judging; the playbook supplies
precedent. A `judge-apply` line shaped like a deviation verdict is rejected
("cannot determine verdict kind").

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
  engine treats any stored provenance verdict with confidence below 0.70 as
  ambiguous at mine time (`pipeline.py` ambiguity rule) and records the
  deal's paper side as `"unknown"` — the trail and every observation say
  `"unknown"`, the OPF 0.5 precedent records say `paper: "unknown"`, and
  `corpus.documents[]` flags it `provenance_is_ambiguous: true` (issue #225).
  The side you named is not used and the verdict is not re-queued, so a
  correct `our_paper` verdict at 0.55 leaves that deal's paper side unknown.
  When the recital evidence is real, say so with confidence ≥ 0.70; reserve
  sub-0.70 for genuine uncertainty.
- Conservative default when genuinely uncertain: `counterparty_paper` (the
  safe choice — it attributes less favorable positions to the counterparty,
  not to us).

### Equivalence (`kind: equivalence`)

Queued by `playbook judge` AFTER mining, once a canonical template is
configured: one item per DISTINCT non-standard text of the precedent record (a
deal's signed text, a non-standard opening, a refused ask) that has no verdict
against our standard yet. The same words in ten deals, on either paper, in any
role, are one item and one shared verdict (the key is over the agreement type,
clause type, `perspective.party`, the text and our standard, never a deal or a
paper side). A text that exactly matches our standard (`standard: true`) is
never queued, and neither is any text in emergent mode (no template, no
`our_standard`).

**Input fields:** `taxonomy_id`, `taxonomy_title`, `perspective_party`,
`our_standard` (our standard text for the clause type), `candidate` (the text
to judge), `roles` (which of `signed` / `opening` / `refused` it plays).
`agreement_type_id` and `stage` are context. There is no deal name,
counterparty, paper side or outcome in the payload; none of those may enter
your answer.

**Task:** Compare the candidate's legal effect for `perspective_party` with
our standard's, and answer with one label and one sentence.

**Prompt (adapt as needed):**

> You are comparing a clause with our standard clause for the same clause
> type. Perspective: {perspective_party}.
>
> Our standard:
> ---
> {our_standard}
> ---
>
> Candidate:
> ---
> {candidate}
> ---
>
> Which describes the candidate relative to our standard, for
> {perspective_party}?
> - `equivalent` -- the same legal effect (wording, order and defined-term
>   names may differ)
> - `more_protective` -- better for {perspective_party} than our standard
> - `less_protective` -- worse for {perspective_party} than our standard
> - `different_concept` -- it does something our standard does not (or omits
>   what it does), so the two cannot be ranked
>
> Respond with JSON: `{"label": "<one of the four>", "reason": "<one
> sentence naming the operative difference>", "basis": "agent"}`

**Rules:**
- `reason` is ONE sentence naming the operative difference, never a restatement
  of the label. No deal, party or counterparty names.
- Judge the language, not the negotiation. Do not use how the deal ended, who
  proposed the text, or whose paper it was on.
- `basis` is required: `"agent"` when you (the coder or agent) judge,
  `"judge"` for the store-backed judge, `"owner"` only for an after-the-fact
  correction by the owner (it wins over any agent or check answer and is never
  checked).
- Do not write `check`: checks are recorded only by `judge-apply --check`.
- When the candidate does something our standard does not (or the two cannot
  be ranked), `different_concept` is the honest label: it makes no
  protectiveness claim. Do not invent a ranking. A text you leave out of the
  verdict file stays queued (and `null` in the playbook), so the drain loop
  cannot finish until it is answered.

**The blind check.** Every drafted verdict is then checked by a SEPARATE
agent pinned to `claude-opus-5-5` at reasoning effort `xhigh`, which never
sees the draft:

```bash
playbook judge --check equivalence $OUT --config <config>
# -> $OUT/judge/check-pending.jsonl: the drafter's payload WITHOUT label/reason
# -> $OUT/judge/adjudication-pending.jsonl: disputed items (both answers)
```

Answer each queue line with `{"key", "label", "reason", "model", "effort"}`,
writing the model id you actually ran on and the effort, then:

```bash
playbook judge-apply $OUT --check $OUT/my-checks-<date>.jsonl
```

`judge-apply --check` rejects a record whose `model` is not `claude-opus-5-5`
or whose `effort` is not `xhigh` unless `--allow-checker-model` is passed. A
match keeps the verdict (`check: {agreed: true, ...}`); a mismatch keeps the
draft flagged `agreed: false` and queues it for adjudication, which a FRESH
pinned agent answers (its answer replaces the verdict, `adjudicated: true`).
Every line of a `--check` file is judged against the store as it stood before
that file, and a key repeated in one file is rejected: record the adjudication
answers in a later file, after re-running `judge --check equivalence`.
The drafter never answers its own check. `playbook judge --check equivalence
--api` answers both queues through the Message Batches API when
`ANTHROPIC_API_KEY` is set (a refusal leaves the item unchecked; there is no
fallback model). An unchecked draft still reaches the playbook (`check:
null`): nothing blocks.

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
store-backed judge would build on replay (`classify` accepts
`{"judge", "unclassified"}`, `provenance` builds a `ProvenanceResult` that
rejects `"judge"` — see `agent_judge.py`'s `_CLASSIFY_REPLAYABLE_BASES`) and raises on
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
  `llm_segmenter` / `inherited` are set by the engine itself, not by you, and
  are rejected. A no-fit (`"unclassified"`) answer for a heading-less
  sub-item (an `(a)`/`(b)` or `(i)`/`(ii)` child) whose parent clause is
  classified is resolved to the parent's `taxonomy_id` with
  `basis: "inherited"` (confidence capped at 0.6, issue #222); a specific
  taxonomy fit you give the child is kept.
- Provenance (`ProvenanceResult.basis`, `provenance_detector.py`) — use
  `"llm"` — this is the one kind where `"llm"` is correct; `validate_verdict`
  rejects any other basis from a producer-supplied verdict (deterministic
  bases like `template_similarity` / `alias_first_party` / `hint` are set by
  the engine itself, not by you).
- Scope — no `basis` field; it is forced to `"judge"` on replay.
- Equivalence — `basis` is REQUIRED and is one of `"agent"` (you),
  `"judge"` (the store-backed judge) or `"owner"` (an owner correction);
  anything else is rejected.

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
playbook.

**Equivalence verdict** (`kind: equivalence`) — one per distinct text; exactly
`label`, `reason`, `basis` and nothing else:
```json
{
  "label": "less_protective",
  "reason": "Drops the recipient's burden of establishing that an exclusion applies.",
  "basis": "agent"
}
```
`label` is `equivalent`, `more_protective`, `less_protective` or
`different_concept`, relative to `perspective_party`.

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
| legacy | no stamp (banked before versioning) | replays, reported every run (`--strict-rubric` to re-queue instead) |

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
`scope`, the answer vocabulary for `provenance`, the label vocabulary for
`equivalence`. Editing
`spec/taxonomy/*.yaml` re-queues classify verdicts and nothing else.

A store banked before versioning (legacy verdicts) keeps replaying and is
reported on every run; if you have reason to distrust it, `--strict-rubric`
re-queues it for re-judgement. There is no command that re-stamps a store
in place: a verdict is re-asked, or kept as it is.

---

## Guardrails

1. **No fabrication.** Every verdict must be grounded in the actual clause
   text or recital. Do not invent legal interpretations.

2. **Flag, don't guess.** When confidence is below threshold, set
   `needs_review: true` and include the uncertainty in `rationale`. **This
   flag is not read back anywhere today** — `ClauseClassification` and
   `ProvenanceResult` have no `needs_review` field, and replay reconstruction
   (`agent_judge.py`) discards the stored boolean when it rebuilds a verdict
   from the store. Setting it is still worth doing — it is a durable,
   human-auditable record in `verdicts.jsonl` for a future manual pass over
   the store — but do not tell a reviewer, or assume yourself, that flagging
   a doubtful verdict here causes it to surface anywhere. If a low-confidence
   call genuinely needs a human's eyes before this round ships, say so
   directly — in the round summary you give the human running this skill —
   rather than relying on `needs_review: true` to do that for you.

3. **Unknown aliases.** If a party name in the corpus is not in
   `provenance.known_entities`, record it explicitly in `rationale` and set
   `needs_review: true`. Supply a curated alias list to the human before the
   next round.

4. **`needs_review` is an internal flag only.** It must be resolved (by human
   review or re-judgment) before `project`. It is not a valid value of any
   OPF field.

5. **Deduplication.** `playbook judge` deduplicates by content hash. Judge
   each unique clause payload once; verdicts propagate automatically to all
   documents sharing that clause.

6. **Corpus confidentiality.** Real agreement text is private. Do not log,
   echo, or store clause text outside the local `out/` directory.

7. **Posture / Floor fields.** `posture.system_prompt` (Posture section) and the
   walk-away floor (`floor.invariants`, Floor section) require the GC interview
   (OPF-SPEC.md §7) and cannot be derived from the corpus — never invent them.
   (The Evidence section is different: it is purely descriptive — per-deal
   precedent facts and counts, OPF-SPEC.md §3.5.4 — and compiles straight
   from the corpus; do not treat it as interview-gated.) With the human
   present, run SKILL.md Step 7a (interview) and Step 7b (floor propose → sign);
   without them, list both as pending human input in your round summary and say that an
   evidence-only playbook is a complete document (Rung 0), not an unfinished one.
   Q4 `sacred_clauses` is written straight into signed `floor.invariants` (the
   human authored it) **only when the item is a bare clause-type name**;
   compiler-derived candidates become hard lines only when the human signs
   them with `playbook floor sign`. A sentence-shaped or conditional item
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

3. **The packaged artifact is generated** (`index.html` exists in `out/`).

   SKILL.md Step 9 names `index.html` as the **one human-readable artifact**
   (the playbook and an optional editor, five tabs); it is produced on every
   route (A, B and C all reach Step 9). A run that stops after `validate`
   without `view bundle` is not done: the GC has the canonical JSON but not the
   page, and the closing (open the page, print both paths, print the toaster
   install steps, one line saying review is optional) has not happened. The
   digest is not a separate file: it is the `digest` section of
   `playbook.opf.json`. `index.html` is **not** a guarantee of pseudonymization
   on its own — SKILL.md Step 9's mandatory residue check must be run before
   treating it as shareable, and nothing in the engine anonymizes a playbook
   for public release. If `overrides.json` exists, run
   `apply-overrides` (always, whatever the file times; it is a no-op once the
   edits are in effect), re-validate, rebuild the page.

   ```bash
   test -f ./out/index.html
   echo "Exit: $?"   # must be 0
   ```

4. **`out/quarantine.json` is empty, or every entry has been explicitly
   triaged.**

   A document quarantined during `mine` (SegmentationQAError, HintsError,
   all-versions-failed-ingest) contributes no `pending.jsonl` items, so
   criteria 1–3 above can all hold while a whole agreement's negotiation
   history is silently absent from the evidence — the only trace is
   `quarantine.json` itself, which is a file, not a gate. `quarantine.json` is
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
   your round summary to the human running this skill (`quarantine.json` is
   tool-generated and cannot itself record a human decision). Never treat an untriaged
   `quarantine.json` entry as background noise the loop can converge past.

Items that remain for human review (unknown aliases, a doubtful
low-confidence call, a thin `floor.candidates.json`) do not block the
done-criteria above, but they must not be silently suppressed: name them in the
round summary. Setting `needs_review: true` on a verdict is **not** a way to
report them — it is not copied onto anything a reader sees.

---

## Common failure modes

| Symptom | Cause | Fix |
|---------|-------|-----|
| `validate` exits non-zero after `project` | Residual `needs_review` or a malformed verdict in the store | Drain the judge loop; fix malformed verdicts |
| `pending.jsonl` does not shrink between rounds | Verdicts not applied or wrong `key` | Check `judge-apply` output; confirm keys match |
| All provenance = `counterparty_paper` | Recitals not loaded or entity alias list empty | Supply `provenance.our_party_aliases`; re-run provenance round |
| All clauses unclassified (`taxonomy_id: null`) | Taxonomy mismatch with document content | Check taxonomy covers the agreement type; refine taxonomy entries |
| Trail ordering wrong | No `order:` hint; version-orderer used greedy fallback | Add explicit `order:` list to `hints.yaml`; re-run `mine` |
