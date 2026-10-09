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
        - signed detection: signature blocks, e-sign certs, digital-sig objects;
          template placeholders ("By: [Name]", "By: Name:", "By: Authorized
          Signatory") never count as signed
        - version ordering: edit-distance chain anchored at the signed terminal;
          ties seeded by each version's own document timestamp (DOCX core.xml
          modified/created + latest tracked-change w:date, PDF /ModDate), which
          hints.yaml timestamps override per version; NOT dependent on
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
          (only in a deal with a detected signed copy — with none, the last
          draft is not a signed terminal and no refused ask is recorded)
        - NO deviation judge. Each observation carries two deterministic
          facts — `standard` (its normalized text EXACTLY equals the template
          clause for its taxonomy_id; party names neutralized) and `outcome` —
          and `deviation` is derived from `standard` ("none" / "substantive",
          basis "deterministic", constant neutral placeholder risk_delta).
          Nothing is queued; the consumer (a review model) does the judging
        - ONE terminal observation per (deal, taxonomy_id): the signed version's nodes
          for that clause, joined in document order, cited to the first node
        - text removed before signing is never "signed": a clause with no signed
          slot whose own text occurs verbatim (fill-in blanks aside) in one
          clause-sized, contiguous stretch of the signed version (e.g.
          relocated) is dropped and counted (corpus.stats.dropped_observations);
          otherwise it was removed (narrowed or replaced text included, even
          when its words recur), and its ORIGIN (never the deal's paper side,
          tested against every template node for the taxonomy_id) decides:
            · our standard language struck → our CONCESSION (never a
              rejected/refused ask) —
              only in a deal with a detected executed copy; in an unsigned
              deal it is dropped and counted, never a concession
            · non-standard (their) language struck → proposed_then_reversed —
              only in a deal with a detected executed copy; in an unsigned
              deal it is dropped and counted (refused_ask_no_signed_copy), so
              an unsigned deal's precedent records carry no refused_asks
            · no standard to compare against → dropped and counted
        - what each clause OPENED with (issue #233; only in a deal with a
          detected executed copy): the first draft's text of a clause type
          (every first-draft node bound into its aligned rows) is a fact,
          whatever its origin. Each terminal row carries opened_with
          (standard | non_standard | absent), and one `opening` row per clause
          type whose first-draft text differs from what was signed carries
          that text. Text relocated, not struck, yields no opening.
                                     │
  L5  COMPILE PLAYBOOK      aggregate observations → OPF 0.5 (deterministic assembly)
        - decide the clause types and each one's our_standard (template only)
        - evidence = {clauses, precedent} (issue #223) — one verdict-free
          precedent per (deal, clause) (signed_text, standard, rounds/moved,
          opened_with, opening_text, refused_asks, paper as metadata only);
          clause n_* counts
          and the digest_version 3 digest are derived from it (precedent.py)
        - every evidence count counts DISTINCT DEALS; nothing is read from a
          judged verdict or risk direction
        - one format only: OPF 0.1–0.4 and digest 2 were retired (issues #238, #233)
                                     │
            ┌──────────────────────────────────────────────────────────────┐
  OUTPUT    │  playbook.opf.json (validates: playbook.schema-0.5.json)     │
            └──────────────────────────────────────────────────────────────┘
```

## Why each non-obvious choice

**Version ordering without status labels.** Status fields (`IN_REVIEW`/`EXECUTED`) are convenient but absent in messy corpora, so they cannot be the backbone. Instead: detect the signed copy (terminal anchor), then order the remaining versions as the most-parsimonious edit path ending at that terminal (minimize total edit distance step to step). Trustworthy timestamps/filename dates *seed* the ordering; the LLM only arbitrates ambiguous ties. This is content-derived and therefore portable.

**Diff is the backbone; tracked changes are a bonus.** Word tracked changes exist in only a minority of files and never in PDFs, so a tracked-changes-first design cannot generalize. Deterministic text diff between ordered versions is the primary signal. Where tracked changes *are* present they enrich an observation with author + accept/reject intent that text diff cannot recover.

**Reversal detection recovers "rejected" without labels.** A span inserted in one version and removed before the signed terminal is an explicit rejection — derivable purely from ordered diffs. This is the cleanest "unacceptable" signal in any corpus.

**Each negotiation is one precedent.** Version count reflects how hard a deal was, not how important it is. The *signed outcome* counts once per deal; intermediate reversals are supplementary and must not double-count an outcome. Concretely: L4 emits exactly one signed (or, with no detected executed copy, unsigned) observation per deal per taxonomy_id, built from the signed version's own text — a clause the segmenter split across several nodes is still one precedent, and a draft's text that was replaced before signing is never reported as signed. L5 writes one precedent record per (deal, clause), and every clause count (`n_deals`, `n_signed_standard`, `n_variants`, `n_refused`) and every digest `n_deals` is a number of distinct deals (`document_id`), never of observation rows.

## Intermediate artifacts (not part of OPF)

The engine writes inspectable intermediates so runs are debuggable and re-runnable:
- `normalized/<doc>/<version>.clauses.json` — the clause tree per version.
- `trail/<doc>.json` — inferred order, signed version, provenance, per-round diffs.
- `observations.jsonl` — one row per clause observation feeding L5: one terminal row per (deal, taxonomy_id), one per unclassified terminal node, one `proposed_then_reversed` row per reversal, one `opening` row per clause type whose first-draft text differs from what was signed (a signed deal only; OPF 0.5), and, for a clause removed before signing, one row classified by the origin of its text: `conceded_before_signing` for our standard language (only in a deal with a detected executed copy) or `proposed_then_reversed` for non-standard language. Removed text that survives in the signed version, or that is our standard in a deal with no detected executed copy, produces no row and is counted in `corpus.stats.dropped_observations`. Removed text whose origin cannot be determined (for example a clause type with no template clause) produces no `conceded_before_signing` or `proposed_then_reversed` row and is still counted there, but in a deal with a detected executed copy it does yield the clause type's `opening` row, so the first draft's text of that clause reaches precedent as `opening_text` with `signed_text` null.
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
