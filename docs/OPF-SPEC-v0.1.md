# Open Playbook Format (OPF) — Specification

> **SUPERSEDED by [OPF v0.2](OPF-SPEC.md)** — retained for history; do not implement against this version.

**Version:** 0.1 (draft)
**Status:** Under development. Breaking changes expected until 1.0.
**Serialization:** JSON (canonical). YAML permitted for authoring; tools MUST accept both and treat them as equivalent.

---

## 1. Purpose and scope

OPF is an open format for representing a **negotiation playbook** for a single agreement type, compiled from a corpus of real negotiated agreements. A playbook tells a reviewer — human or LLM — how to evaluate a clause in a new agreement: what the preferred position is, what variants have been accepted, what has been conceded, and what has been rejected, each grounded in cited precedent.

OPF describes **what the playbook knows**, not how to apply it. Applying a playbook to a new document (triage, scoring, redline generation) is the job of a separate *review engine*. OPF is the interface between the two.

### Non-goals
- OPF is not a contract format and does not represent agreements themselves.
- OPF does not encode the review engine's behavior or thresholds.
- OPF is not a trained model; it carries no weights.

## 2. Core concepts

| Term | Meaning |
|---|---|
| **Agreement type** | One playbook covers exactly one type (e.g. "Educational Affiliation Agreement"). |
| **Baseline / our paper** | The party's own canonical template, when one exists. Edits against it are *deviations* — the strongest signal. |
| **Counterparty paper** (a.k.a. third-party paper) | A document drafted on the other side's template. Clauses surviving here were *tolerated*, not necessarily *endorsed*. |
| **Clause position** | A template-anchored entry: our standard for one clause type + every observed variant + derived guidance. |
| **Clause concept** | A provenance-indexed clause entry used to match clauses in *counterparty paper*, where there is no baseline to diff against. |
| **Observation** | A single occurrence of a clause variant in the corpus, with its outcome and provenance. |
| **Deviation** | How much an observed clause differs from our standard: `none` / `reworded_equivalent` / `substantive`. |
| **Risk delta** | The risk change relative to our standard: a `direction` (`better`/`neutral`/`worse`) and a `magnitude` (`none`/`minor`/`material`). |
| **Outcome** | What happened to an observed variant: `signed` (in the executed copy) or `proposed_then_reversed` (appeared in a draft then removed before signing — an explicit rejection). |
| **Provenance** | `our_paper` or `counterparty_paper`. |

### 2.1 The risk-delta model (why there is no "concession" field)

OPF deliberately does **not** carry flat `acceptable_variant` / `concession` labels. Those conflate two things you cannot reliably infer separately from text, and they hide the reasoning. Instead an observation carries orthogonal attributes — `deviation`, `risk_delta`, `provenance`, `outcome` — and the human-facing distinction is **derived**:

- *Acceptable variant* = signed observation with `risk_delta.direction = neutral`.
- *Concession* = signed observation with `risk_delta.direction = worse` (magnitude tells you how reluctant).
- *Rejected ask* = observation with `outcome = proposed_then_reversed`.

This keeps the format auditable (you can always see *why* something is a concession) and general across agreement types.

### 2.2 The provenance rule (normative)

> Only **our-paper** drafting (the canonical template and our-paper deals) MAY define an *opening position* (`our_standard`, and any `rollup.position` stronger than `negotiable`).
> **Counterparty-paper** observations MAY only inform tolerance bounds (`rollup.fallbacks`) and populate the `clause_library`. A counterparty-paper-only clause MUST NOT set `our_standard`.

Rationale: a clause surviving in the counterparty's template means we failed to strike it, not that we'd propose it. Survivorship is not endorsement.

## 3. Document structure

A playbook is a single JSON object:

```jsonc
{
  "opf_version": "0.1",
  "agreement_type": { "id": "...", "name": "...", "description": "..." },
  "baseline": { ... },          // §3.2
  "taxonomy": { ... },          // §3.3
  "clauses": [ ClausePosition ],      // §3.4 — template-anchored
  "clause_library": [ ClauseConcept ],// §3.5 — concept-indexed
  "corpus": { ... },            // §3.6 — provenance/audit
  "compiler": { ... }           // §3.7 — generation metadata
}
```

### 3.1 Top-level fields
- `opf_version` (string, required) — the OPF version this document conforms to.
- `agreement_type` (object, required) — `id` (slug), `name`, `description`.

### 3.2 `baseline`
```jsonc
{
  "has_canonical_template": true,
  "template_ref": { "document_id": "...", "title": "...", "source": "path or URI" },
  "notes": "free text"
}
```
If `has_canonical_template` is `false`, the playbook is *emergent* (no baseline). The provenance rule still applies; with no template, `our_standard` is set only from our-paper deals, and many clauses will be `negotiable` by default.

### 3.3 `taxonomy`
A curated clause taxonomy. Entries may be inactive so curation choices survive upstream upgrades (§5).
```jsonc
{
  "source": "CUAD-v1",                 // upstream taxonomy + version, or "custom"
  "entries": [
    {
      "id": "indemnification",
      "label": "Indemnification",
      "status": "active",              // active | inactive | custom
      "cuad_origin": "Indemnification",// null for custom entries
      "description": "Who bears third-party claim risk."
    }
  ]
}
```

### 3.4 `ClausePosition` (template-anchored)
```jsonc
{
  "id": "clause.indemnification",
  "taxonomy_id": "indemnification",
  "title": "Indemnification",
  "our_standard": {
    "text": "…canonical language…",
    "source_ref": { "document_id": "template", "clause_path": "8", "char_span": [start, end] }
  },
  "observed_positions": [
    {
      "text_summary": "Mutual indemnification, negligence-based.",
      "example_ref": { "document_id": "…", "version": 4, "clause_path": "8.1", "char_span": [s, e] },
      "deviation": "substantive",
      "risk_delta": { "direction": "worse", "magnitude": "minor" },
      "provenance": "our_paper",
      "outcome": "signed",
      "precedent_count": 3
    }
  ],
  "rollup": {
    "position": "negotiable",          // standard | acceptable_variants_exist | negotiable | hold_firm
    "acceptable_if": [ "mutual", "negligence-limited" ],
    "fallbacks": [ /* worse-risk signed observations, ordered least-to-most costly */ ],
    "rejected": [ /* observations with outcome = proposed_then_reversed */ ],
    "confidence": { "score": 0.0, "basis": "precedent_count + provenance_mix", "n_our_paper": 3, "n_counterparty_paper": 1 }
  }
}
```
All `*_ref` objects are **citations** (§4) and are REQUIRED wherever text is asserted.

### 3.5 `ClauseConcept` (concept library, for counterparty paper)
Used when reviewing a document drafted on the counterparty's template, where clauses must be matched by concept and risk rather than diffed against our template.
```jsonc
{
  "concept_id": "concept.governing_law",
  "taxonomy_id": "governing_law",
  "description": "Which state's law governs and where disputes are heard.",
  "risk_profile": "Venue/forum exposure; cost of litigating away from home.",
  "accepted_forms": [
    {
      "text_summary": "Counterparty's home state law, no forum clause.",
      "example_ref": { "document_id": "…", "version": 5, "clause_path": "…", "char_span": [s, e] },
      "provenance": "counterparty_paper",
      "risk_delta_vs_our_standard": { "direction": "worse", "magnitude": "minor" }
    }
  ],
  "notes": "Tolerated in N of M counterparty-paper deals."
}
```

### 3.6 `corpus`
Audit trail of what compiled this playbook.
```jsonc
{
  "documents": [
    { "document_id": "…", "title": "…", "provenance": "our_paper",
      "in_scope": true, "scope_rationale": "…", "scope_confidence": 0.93,
      "versions": 5, "signed_version": 5, "version_order_basis": "edit_distance_chain+signed_anchor" }
  ],
  "stats": { "documents_total": 12, "documents_in_scope": 10, "versions_total": 37 }
}
```
Out-of-scope documents MUST be retained here with `in_scope: false` and a `scope_rationale` — never silently dropped (§ ARCHITECTURE).

### 3.7 `compiler`
```jsonc
{ "name": "playbook-engine", "version": "x.y.z", "run_id": "…", "generated_at": "ISO-8601 (supplied by caller)" }
```

## 4. Citations

Every asserted clause text MUST be traceable. A citation is:
```jsonc
{ "document_id": "string", "version": 4, "clause_path": "8.1", "char_span": [start, end] }
```
- `document_id`, `version`, and `clause_path` are REQUIRED — a citation carrying only `document_id` is not traceable in practice and is non-conformant (see `spec/playbook.schema.json`'s `citation` definition). `char_span` is optional (a clause may be cited precisely enough by `clause_path` alone).
- `clause_path` is the dotted clause numbering in the normalized document (§ ARCHITECTURE), not the raw PDF page.
- `char_span`, when present, indexes into the document's full normalized text — the same document-relative coordinate system as `ClauseNode.char_span` in the clause-tree artifact (`spec/clause-tree.schema.json`), not an offset relative to the clause's own text.
- `version` is the inferred ordinal (1-based); `"template"` is a reserved `document_id` for the baseline. Every citation's `(document_id, version)` MUST resolve against `corpus.documents`: `document_id` (unless `"template"`) MUST be present in `corpus.documents`, and `version` (when numeric) MUST NOT exceed that document's `versions` count. A citation that fails to resolve is a dangling citation and non-conformant.

## 5. Taxonomy curation & upgrades

Taxonomies are derived from upstream sources (e.g. CUAD) but curated per type. Pruning is done by setting `status: "inactive"`, never by deletion. Custom additions use `status: "custom"` with `cuad_origin: null`. When an upstream taxonomy is upgraded, new categories are merged in as `inactive` by default; existing curation is preserved. A compiler MUST only classify clauses into `active` or `custom` entries.

## 6. Confidence

Confidence is advisory, not statistical authority. It is a function of precedent count and provenance mix, with our-paper observations weighted above counterparty-paper. With small corpora most entries will be low-confidence; consumers MUST surface confidence and MUST NOT treat a single precedent as a rule. The exact scoring function is left to the compiler and recorded in `compiler`.

## 7. Conformance

A **conformant playbook** validates against [`spec/playbook.schema.json`](../spec/playbook.schema.json) and obeys the normative rules in §2.2 and §3.6.
A **conformant producer** emits conformant playbooks and records out-of-scope documents.
A **conformant consumer** honors provenance (§2.2) and surfaces confidence (§6) and citations (§4).

## 8. Versioning

OPF uses semantic versioning. `opf_version` is required in every document. Pre-1.0 minor versions may break compatibility.

## Appendix A — Changelog
- **0.1** — Initial draft. Risk-delta model, provenance rule, dual structure (clause positions + clause library), citation requirement, taxonomy curation model.

## Appendix B — Ratification notes (v0.1)

Walk-through of real clause situations against the model (using synthetic fixtures):

**Concession** — modeled as `outcome: signed` + `risk_delta.direction: worse`. The signed example_ref is the citation; `rollup.fallbacks` accumulates these ordered least-to-most-costly. No separate "concession" field needed.

**Acceptable variant** — `outcome: signed` + `risk_delta.direction: neutral`. Accumulated in `rollup.acceptable_if` (natural-language conditions) and the observation's `example_ref` is the citation.

**Rejected ask** — `outcome: proposed_then_reversed`. Accumulated in `rollup.rejected`. The example_ref cites the draft version in which it appeared.

**Counterparty-paper clause_library entry** — appears in `clause_library[].accepted_forms` with `provenance: counterparty_paper`. No `our_standard` is set; the provenance rule is not violated. `risk_delta_vs_our_standard` captures the risk relative to what we *would* want.

**Emergent playbook (no template)** — `baseline.has_canonical_template: false`, `template_ref: null`. All `our_standard` fields are `null`; positions are derived from signed-deal observations only. Rollup confidence is typically lower.

**`acceptable_if` and counterparty-paper tolerance** — `rollup.acceptable_if` may carry text derived from counterparty-paper observations (e.g. "asymmetric notice accepted where all performance is on-site"). §2.2 explicitly names `rollup.fallbacks` and `clause_library` as the sanctioned counterparty-paper channels; `acceptable_if` is not enumerated but serves the same tolerance-bounds function. This is permitted: the provenance rule bars counterparty paper from *setting an opening position*, not from informing tolerance text alongside our-paper grounding.

**Open spec gap — conditional acceptability:** `rollup.acceptable_if` is free text, so threshold-conditional tolerances ("asymmetric notice accepted only for multi-year contracts >$500K") cannot be evaluated deterministically by a consumer. OPF v0.1 has no structured conditional field. This is a known expressiveness gap; a structured `acceptable_if_conditions` (with operator/threshold fields) is a candidate for a future version. Captured for follow-up.
