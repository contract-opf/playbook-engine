# playbook-from-corpus — Judge reference

Agent reference for acting as the LLM judge in the `playbook-from-corpus`
derivation loop. Contains judge prompts, guardrails, and machine-checkable
done-criteria.

---

## Judge prompts

Each item in `out/judge/pending.jsonl` has a `kind` field that determines
the judgment task.

### Classification (`kind: classification`)

**Input fields:** `clause_text`, `taxonomy_entries` (list of `{id, label}`),
`context` (surrounding clause text, if available).

**Task:** Assign the best-fit `taxonomy_id` from `taxonomy_entries`. Return
`null` if no entry fits with reasonable confidence.

**Prompt (adapt as needed):**

> You are classifying a clause from a legal agreement. The clause text is:
>
> ---
> {clause_text}
> ---
>
> Taxonomy entries:
> {taxonomy_entries}
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

**Input fields:** `clause_text`, `template_hunk` (our standard for this
clause type), `taxonomy_id`.

**Task:** Classify the deviation from the template hunk and assess risk.

**Prompt (adapt as needed):**

> You are assessing a clause deviation against our standard template language.
>
> Our standard (template hunk):
> ---
> {template_hunk}
> ---
>
> Counterparty/negotiated version:
> ---
> {clause_text}
> ---
>
> Classify the deviation as one of:
> - `none` — substantively identical to the template; only formatting/style differences
> - `reworded_equivalent` — different wording, same legal effect
> - `substantive` — materially different legal obligation or right
>
> For `substantive`, also assess:
> - `risk_delta.direction`: `better` / `neutral` / `worse` (relative to our template)
> - `risk_delta.magnitude`: `none` / `minor` / `moderate` / `major`
>
> Respond with JSON: `{"deviation": "<none|reworded_equivalent|substantive>", "risk_delta": {"direction": "...", "magnitude": "..."}, "confidence": 0.0–1.0, "rationale": "..."}`

**Rules:**
- If no template hunk is provided (`template_hunk` is null), deviation
  assessment is relative to the modal observed position, not a template.
- Confidence < 0.65: set `needs_review: true`.
- Do not assess risk magnitude for `none` or `reworded_equivalent`.

---

### Provenance (`kind: provenance`)

**Input fields:** `document_id`, `recital_text` (opening recital/header of
the document), `known_aliases` (list of known names for "us").

**Task:** Determine whether this document originated from our paper or the
counterparty's paper.

**Prompt (adapt as needed):**

> You are determining which party drafted the original version of this
> agreement by reading its recital/opening section.
>
> Our known names/aliases: {known_aliases}
>
> Agreement opening (recital):
> ---
> {recital_text}
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
- Confidence < 0.7: set `needs_review: true`.
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

7. **OPF v0.2 fields.** The `historical_stance` (Posture section) and walk-away
   floor (Floor section) require a GC interview and cannot be derived from the
   corpus. List them as pending human input in the report — never invent them.

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
