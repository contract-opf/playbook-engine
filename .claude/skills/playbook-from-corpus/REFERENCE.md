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

**Task:** Assign the best-fit `taxonomy_id` from `taxonomy_entries`. Return
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
- Confidence < 0.6: set `needs_review: true` in the verdict.
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
- **Relocation triage FIRST (do this before judging items one by one):** group
  the pending deviation items by `document_id` and scan each document-pair's
  full hunk set for relocations — the same (or near-identical) clause text
  disappearing in one hunk and reappearing, possibly renumbered, in another
  hunk of the same document. Auto-verdict BOTH sides of each relocation pair
  as `none` / neutral / none with rationale "alignment artifact — clause
  relocated, text unchanged" (confidence 0.8+, no needs_review). Only judge
  the remaining hunks individually. On real corpora a third or more of
  deviation items are relocation echoes; judging them blind wastes effort and
  floods the report with needs_review flags.
- If no template hunk is provided (`template_hunk` is null), deviation
  assessment is relative to the modal observed position, not a template.
- Confidence < 0.65: set `needs_review: true`.
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
- Unknown entity name (a party name in the recital not in `known_aliases`):
  record the alias in `rationale`, set `needs_review: true`. Do not silently
  assume it is us.
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
NOT all include `"llm"`.** Using the wrong value doesn't raise where you'd
notice it: the verdict hits the store, fails reconstruction on the bad
`basis`, and is silently re-queued as `needs_review` — which then fails
schema validation much later, at `project`/`validate`. Use exactly the value
shown for each kind below:
- Classification (`ClauseClassification.basis`, `clause_classifier.py`) — use
  `"judge"` for an agent-produced verdict (also accepts `exact_match` /
  `heading_similarity` / `judge_error` / `needs_review` / `unclassified` /
  `llm_segmenter`, but those are set by the engine itself, not by you).
- Deviation (`DeviationResult.basis`, `deviation_classifier.py`) — use
  `"judge"` for an agent-produced verdict (also accepts `deterministic` /
  `reworded_equivalent` / `judge_error` / `needs_review`, set by the engine).
- Provenance (`ProvenanceResult.basis`, `provenance_detector.py`) — use
  `"llm"` — this is the one kind where `"llm"` is correct.
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

Each `segment/pending.jsonl` item gives a document's `canonical_text`, its
`blocks` (`{block_id, page, char_span, text}`), and the allowed `taxonomy_ids`.
Partition the blocks into contiguous clause ranges — one `SegNode` per clause —
and write one verdict line per document to the verdicts JSONL:

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

## Guardrails

1. **No fabrication.** Every verdict must be grounded in the actual clause
   text or recital. Do not invent legal interpretations.

2. **Flag, don't guess.** When confidence is below threshold, set
   `needs_review: true` and include the uncertainty in `rationale`. The
   after-action report will list these for human follow-up.

3. **Unknown aliases.** If a party name in the corpus is not in `known_aliases`,
   record it explicitly in `rationale` and set `needs_review: true`. Supply a
   curated alias list to the human before the next round.

4. **`needs_review` is an internal flag only.** It must be resolved (by human
   review or re-judgment) before `project`. The OPF observation schema enum is
   `{none, reworded_equivalent, substantive}` — `needs_review` is not a valid
   `deviation` value.

5. **Deduplication.** `playbook judge` deduplicates by content hash. Judge
   each unique clause payload once; verdicts propagate automatically to all
   documents sharing that clause.

6. **Corpus confidentiality.** Real agreement text is private. Do not log,
   echo, or store clause text outside the local `out/` directory.

7. **Posture / Floor fields.** The `historical_stance` (Posture section) and the
   walk-away floor (Floor section) require the GC interview (OPF-SPEC.md §7) and
   cannot be derived from the corpus — never invent them. With the human
   present, run SKILL.md Step 7a (interview) and Step 7b (floor propose → sign);
   without them, list both as pending human input in the report and say that an
   evidence-only playbook is a complete document (Rung 0), not an unfinished one.
   Q4 `sacred_clauses` is written straight into signed `floor.invariants` (the
   human authored it); compiler-derived candidates require an explicit accept
   round-tripped through `playbook view apply`.

---

## Done-criteria (machine-checkable)

The derivation is **done** when all three conditions hold:

1. **`out/judge/pending.jsonl` is empty** (or absent).

   ```bash
   # Confirm: exit 0 with empty output, or file absent
   [ ! -f ./out/judge/pending.jsonl ] || [ ! -s ./out/judge/pending.jsonl ]
   ```

2. **`playbook validate` exits 0.**

   ```bash
   playbook validate ./out/playbook.opf.json
   echo "Exit: $?"   # must be 0
   ```

3. **Report and viewer are generated** (`report.md`, `report.json`,
   `playbook.review.html` exist in `out/`).

   ```bash
   test -f ./out/report.md && test -f ./out/report.json && test -f ./out/playbook.review.html
   echo "Exit: $?"   # must be 0
   ```

Items that remain for human review are **listed in the report** under
"Needs Attention" — they do not block the done-criteria above, but must not
be silently suppressed.

---

## Common failure modes

| Symptom | Cause | Fix |
|---------|-------|-----|
| `validate` exits non-zero after `project` | Residual `needs_review` or malformed `deviation` value in store | Drain the judge loop; fix malformed verdicts |
| `pending.jsonl` grows every round | Verdicts not applied or wrong `key` | Check `judge-apply` output; confirm keys match |
| All provenance = `counterparty_paper` | Recitals not loaded or entity alias list empty | Supply `known_aliases`; re-run provenance round |
| All clauses unclassified (`taxonomy_id: null`) | Taxonomy mismatch with document content | Check taxonomy covers the agreement type; refine taxonomy entries |
| Trail ordering wrong | No `order:` hint; version-orderer used greedy fallback | Add explicit `order:` list to `hints.yaml`; re-run `mine` |
