# Architecture — the corpus → playbook compiler

The engine turns a directory of agreements into an [OPF](OPF-SPEC.md) playbook. It is a pipeline of layers. The governing rule: **deterministic where possible, LLM only for semantic judgment.** Every box that can be done with parsing/diffing is done that way, so runs are reproducible and cheap; the LLM is invoked only for calls that genuinely require reading comprehension, and only ever on *changed* or *unclassified* spans — never on whole documents repeatedly.

```
            ┌──────────────────────────────────────────────────────────────┐
  INPUT     │  corpus/  (one folder per document; versions inside)          │
            │  + config (agreement type, baseline template, taxonomy)       │
            └──────────────────────────────────────────────────────────────┘
                                     │
  L1  INGEST & NORMALIZE   docx / pdf / rtf  →  clause tree  (deterministic)
        - extract text + structure; preserve tracked changes where present
        - OCR scanned/signed PDFs
        - emit a normalized clause tree per version: {path, heading, text, span, children}
                                     │
  L1b SCOPE GATE            in-scope? (LLM judgment, logged with rationale)
        - decide per document whether it is fundamentally THIS agreement type
        - filename is NOT dispositive; purpose + clause profile decides
        - out-of-scope docs are RETAINED in corpus[] with in_scope:false + rationale
                                     │
  L2  STRUCTURE THE TRAIL   (deterministic + light LLM arbitration)
        - signed detection: signature blocks, e-sign certs, digital-sig objects
        - version ordering: edit-distance chain anchored at the signed terminal;
          seeded by timestamps/filename dates WHEN trustworthy; NOT dependent on
          status labels (general-purpose: corpora may lack them)
        - provenance detection: our_paper vs counterparty_paper
                                     │
  L3  SEGMENT & CLASSIFY    clause segmentation + taxonomy tagging (LLM, clause-scoped)
        - tag each clause into an ACTIVE taxonomy entry
        - align "the same clause" across versions of one document
                                     │
  L4  MINE DELTAS           (deterministic diff + deterministic standard facts)
        - consecutive diffs (vᵢ→vᵢ₊₁) = negotiation moves
        - net diff (template/first → signed) = durable outcome
        - REVERSAL detection: inserted-then-removed-before-signing = proposed_then_reversed
        - DEFAULT (the consumer path): no deviation judge. Each observation carries
          two deterministic facts — `standard` (its normalized text equals, or
          token-Jaccard near-equals at the documented threshold, the template
          clause for its taxonomy_id; party names neutralized) and `outcome` —
          and `deviation` is derived from `standard` ("none" / "substantive",
          basis "deterministic", neutral placeholder risk_delta). Nothing is
          queued; the consumer (a review model) does the judging
        - OPT-IN (`--with-deviation-judge`): an advisory judged layer for
          posture/floor work — the judge assigns deviation + risk_delta vs
          our_standard on the changed hunks
        - ONE terminal observation per (deal, taxonomy_id): the signed version's nodes
          for that clause, joined in document order, cited to the first node
        - text removed before signing is never "signed": a clause with no signed
          slot whose own text occurs verbatim (fill-in blanks aside) in one
          clause-sized, contiguous stretch of the signed version (e.g.
          relocated) is dropped and counted (corpus.stats.dropped_observations);
          otherwise it was removed (narrowed or replaced text included, even
          when its words recur), and its ORIGIN (never the deal's paper side,
          tested against every template node for the taxonomy_id) decides:
            · our standard language struck → our CONCESSION (a conceded deal in
              stance_detail and the position; never a rejected/refused ask) —
              only in a deal with a detected executed copy; in an unsigned
              deal it is dropped and counted, never a concession
            · non-standard (their) language struck → proposed_then_reversed
              (in an unsigned deal too: like any reversal, within-trail history)
            · no standard to compare against → dropped and counted
                                     │
  L5  COMPILE PLAYBOOK      aggregate observations → OPF (deterministic assembly)
        - build ClausePosition[] (template-anchored) honoring the provenance rule
        - build ClauseConcept[] (concept library) for counterparty-paper matching
        - compute rollups (acceptable_if / fallbacks / rejected), confidence, citations
        - consumer path (no judged deviations): no stance, tolerance or fallback is
          read from risk direction — historical_stance "no_signal", acceptable_if /
          fallbacks empty, stance_detail {held: n_signed_standard, of: n_deals,
          basis: "all"}; refused asks (rejected) are kept
        - every evidence count (precedent_count, n_our_paper / n_counterparty_paper,
          stance_detail held/of, digest n) counts DISTINCT DEALS
                                     │
            ┌──────────────────────────────────────────────────────────────┐
  OUTPUT    │  playbook.opf.json (validates: playbook.schema-0.3.json)     │
            └──────────────────────────────────────────────────────────────┘
```

## Why each non-obvious choice

**Version ordering without status labels.** Status fields (`IN_REVIEW`/`EXECUTED`) are convenient but absent in messy corpora, so they cannot be the backbone. Instead: detect the signed copy (terminal anchor), then order the remaining versions as the most-parsimonious edit path ending at that terminal (minimize total edit distance step to step). Trustworthy timestamps/filename dates *seed* the ordering; the LLM only arbitrates ambiguous ties. This is content-derived and therefore portable.

**Diff is the backbone; tracked changes are a bonus.** Word tracked changes exist in only a minority of files and never in PDFs, so a tracked-changes-first design cannot generalize. Deterministic text diff between ordered versions is the primary signal. Where tracked changes *are* present they enrich an observation with author + accept/reject intent that text diff cannot recover.

**Reversal detection recovers "rejected" without labels.** A span inserted in one version and removed before the signed terminal is an explicit rejection — derivable purely from ordered diffs. This is the cleanest "unacceptable" signal in any corpus.

**Each negotiation is one precedent.** Version count reflects how hard a deal was, not how important it is. The *signed outcome* counts once per deal; intermediate reversals are supplementary and must not double-count an outcome. Concretely: L4 emits exactly one signed (or, with no detected executed copy, unsigned) observation per deal per taxonomy_id, built from the signed version's own text — a clause the segmenter split across several nodes is still one precedent, and a draft's text that was replaced before signing is never reported as signed. L5 counts precedent per normalized text, `n_our_paper`/`n_counterparty_paper`, and `stance_detail` held/of as the number of distinct deals (`document_id`), never as observation rows. The digest's `n` is likewise the number of distinct deals in each group, never a sum of `precedent_count`, so a deal carrying two texts that the digest's looser normalization merges counts once.

## Intermediate artifacts (not part of OPF)

The engine writes inspectable intermediates so runs are debuggable and re-runnable:
- `normalized/<doc>/<version>.clauses.json` — the clause tree per version.
- `trail/<doc>.json` — inferred order, signed version, provenance, per-round diffs.
- `observations.jsonl` — one row per clause observation feeding L5: one terminal row per (deal, taxonomy_id), one per unclassified terminal node, one `proposed_then_reversed` row per reversal, and, for a clause removed before signing, one row classified by the origin of its text: `conceded_before_signing` for our standard language (only in a deal with a detected executed copy) or `proposed_then_reversed` for non-standard language. Removed text that survives in the signed version, whose origin cannot be determined, or that is our standard in a deal with no detected executed copy produces no row and is counted in `corpus.stats.dropped_observations`.
- `scope.json` — the scope-gate decisions and rationales.

These let a human (or a workflow) verify L2/L4 before trusting the compiled playbook.

## Configuration (per agreement type)

A type is defined by config, not code:
```yaml
agreement_type: { id: educational-affiliation, name: "Educational Affiliation Agreement" }
baseline:
  template: ./template/internship-template.rtf   # or null for emergent playbooks
taxonomy: ./spec/taxonomy/affiliation-agreement.yaml
provenance:
  our_party_aliases: ["FixtureCorp", "FixtureCorp Holdings", "FixtureCorp Works"]
```
The compiler is agreement-type-agnostic; all type knowledge lives here plus the taxonomy.

### Taxonomy: supplied or induced

The clause taxonomy is **data, not code** — nothing in the engine is specific to a single agreement type. You can either:
- **Supply** a taxonomy (e.g. the affiliation taxonomy under `spec/taxonomy/`, optionally CUAD-merged), or
- **Induce** one from the corpus when you have no taxonomy for a new agreement type. A pre-pass clusters clauses across the corpus, proposes categories (mapping to CUAD where possible, else `custom`), defaults each to `active`/`inactive` by representation, and emits a candidate taxonomy YAML with example citations for attorney review.

Induction runs after segmentation and before classification: ingest → segment → *(induce taxonomy if none)* → human review → classify. This is what makes the engine reusable across agreement types without code changes.

## Packaging for non-engineers

The engine ships as a **skill** with bundled scripts. A non-technical user with access to an LLM runs the skill; it inspects their directory, tells them how to lay it out (see [CORPUS-LAYOUT.md](CORPUS-LAYOUT.md)), runs the deterministic stages, drives the LLM for the judgment stages, and emits a validated playbook. The deterministic scripts are the heavy lifting; the skill instructions orchestrate them.
