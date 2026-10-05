# Spec changelog

One entry per spec-affecting change (schema files, `digest` shape/semantics,
canonicalization, normative validator rules). Downstream consumers vendor
these schemas hermetically — this file is how they diff intent without
reverse-engineering `git log`.

## Versioning policy (normative, effective 2026-07-16)

- **A published `opf_version` is immutable once a consumer exists.** Shape or
  *semantic* changes to what a document's fields mean get a NEW `opf_version`
  (0.3 → 0.4), never an in-place edit of a released schema. "The schema still
  validates old data" is NOT sufficient — two artifacts claiming the same
  version must mean the same thing.
- **`digest_version` governs the digest section independently** and follows
  the same rule: any change to the digest's shape OR selection semantics
  (what the lists include/exclude, how entries are ranked or capped) bumps it.
- CI enforces the paper trail: `tests/test_spec_consistency.py` pins each
  schema file's sha256 to the **Current pins** table below — any schema edit
  fails CI until this changelog gains an entry and updated pins. That
  friction is deliberate.

## Current pins

| File | sha256 |
|---|---|
| `playbook.schema.json` | `c1a25b477eeb71c9a6daa2d2d390793301df5e4e6539503e622b72d2f8276962` |
| `playbook.schema-0.2.json` | `eae5f882f9289f2144cc784109d3dd04de7673d6e563d195fd693fd38ae1138d` |
| `playbook.schema-0.3.json` | `d2d81ca1c4f7547b508b2a22310906ce9a3bf43a2436e8730f0b1e4c9b0a0e15` |
| `playbook.schema-0.4.json` | `14359a861088a100cbf5e9345b9bd92c185f73d1016939030edb867876f81e7f` |
| `spec/conformance/` (manifest.json + vectors/*.json, concatenated) | `9a9a9ee2cceff98226e7b11db83b847396e1a9e33b79ada1c78b92b554eeab2b` |
| `spec/conformance/0.4/` (manifest.json + vectors/*.json, concatenated) | `1abb133bf3b3b9c9583fb33501b0994941de4cf5f264843e2334db13922c54af` |

Current `DIGEST_VERSION`: **3** (`playbook_engine/digest.py`). An OPF 0.3 (or
older) document's digest is digest_version **2** (`DIGEST_VERSION_V2`),
frozen with 0.3.

## History

### 2026-10-05 — OPF 0.4 + digest_version 3: the verdict-free per-deal precedent record (issue #223)

New schema file `playbook.schema-0.4.json` (`opf_version` "0.4"); every
earlier schema file is byte-identical (pins above unchanged) and 0.1/0.2/0.3
documents keep validating against them. `playbook.schema.json` is the frozen
0.1 schema, not a pointer, so it is untouched: the validator's
`_SCHEMA_PATH_BY_VERSION` maps "0.4" to the new file. The reference compiler
emits 0.4 by default (`playbook project --opf-version 0.3` keeps the 0.3
assembler for one release). OPF-SPEC §3.5.4, §3.12.1, §10, §10.2 and §11
document it.

- **Evidence shape.** `evidence` is `{clauses, precedent}`.
  `clauses[]` = `{id, taxonomy_id, title, our_standard, n_deals,
  n_signed_standard, n_variants, n_refused}`; `precedent[]` = one record per
  (deal, clause type): `{id, taxonomy_id, document_id, counterparty_ref?,
  paper ("ours"|"theirs"|"unknown"), paper_basis, paper_confidence, signed,
  signed_at?, rounds, signed_text ({text, ref} | null), opening_text
  ({text, ref} | null), standard, moved, refused_asks[{text, round, ref}]}`.
  No `observed_positions`, `clause_library`, `summary` or
  `negotiation_trail`; no stance, band, risk or deviation verdict.
  `signed_text` is null when the clause was struck before signing (the
  ticket's sketch did not cover that case; the record still needs it, e.g.
  our standard struck — carried as `opening_text`). Precedent and clause
  records are closed (no `x_*`): judged verdicts from an opt-in judged run
  go only under the root `x_judgments` extension, keyed by precedent id.
- **Precedent id.** `"prec." + sha256(canonical([agreement_type.id,
  document_id, taxonomy_id, signed_text.text or ""]))[:16]`.
- **digest_version 3.** `{digest_version, perspective (copied; null when
  absent), agreement_type {id, name}, corpus {n_deals, n_signed,
  first_signed, last_signed}, clauses[{id, taxonomy_id, title, our_standard,
  n_deals, n_signed_standard, signed_variants[{text, n_deals, last_signed,
  ref, precedent_ids}], refused_asks[{text, n_deals, ref, precedent_ids}],
  n_variants_total, n_refused_total}]}`. Grouping (the clause counts and
  the digest) is an exact key over the text and the document alone
  (`precedent.normalize_variant_text`, OPF-SPEC §3.5.4 "Grouping key"):
  every `Counterparty-<n>` registry alias becomes the token `counterparty`,
  then `normalize_for_standard` runs with the document's `perspective.party`
  as its only party name (rewritten to `party`). It is NOT the standard
  check's party-name substitution: that list (`our_party_aliases` +
  `known_entities`) is producer configuration the document does not carry,
  so a validator or port could not recompute a key built from it; and
  without the alias rewrite, deals that signed the same words with
  different counterparties split into per-deal singletons. Our party and
  the counterparty are distinct tokens, so a party-swapped text never
  merges. Variant/ask `text` is the ≤ 300-char sentence-boundary summary;
  lists are capped 5 → 1 to fit the budget, totals always uncapped. No
  `clause_count` key (the clause list's length).
- **New normative validator rules (0.4 only)**, each blocking: precedent ids
  unique and equal to their recomputation; at most one precedent per
  (document_id, taxonomy_id); every precedent's taxonomy_id names a clause
  and its document_id a corpus document, with `signed` agreeing with
  `signed_version`; `standard: true` requires `signed_text`; the clause
  `n_*` counts equal what `precedent` implies; `signed_at` is a real date or
  quarter; a present digest equals `build_digest(document)`. The 0.2/0.3
  §2.2 provenance and evidence-depth rules do not apply to 0.4 (no stance;
  paper side gates nothing).
- **Conformance.** A new, separately stamped vector set
  `spec/conformance/0.4/` (manifest + 4 vectors, pinned above) generated by
  `scripts/generate_conformance_vectors.py --opf-version 0.4`. The 0.3 set is
  byte-identical (its pin is unchanged), and the script's default mode still
  reproduces it exactly — it now stamps that set with `DIGEST_VERSION_V2`
  because `DIGEST_VERSION` is 3.
- **Transforms re-derive.** `playbook publish` and `export_profile` re-derive
  a 0.4 document's precedent ids, counts and digest after rewriting text
  (`precedent.refresh_derived`), so the digest never carries a stale copy of
  pre-transform text or the real `perspective.party`; `publish` coarsens
  `precedent[].signed_at` to `YYYY-Qn` like `observed_at`.

### 2026-09-25 — digest v2 `n` counts distinct deals (issue #216) — OWNER-AUTHORIZED IN-PLACE EXCEPTION

**Deliberate, owner-authorized exception to OPF-SPEC §11 (published-version
immutability) and to this file's versioning policy.** The semantics of the
digest's `n` change in place under `digest_version` **2** — no
`digest_version` bump, no `opf_version` bump, and no schema file changes
(every `playbook.schema*.json` is byte-identical; its pin above is
unchanged). The owner authorized this on 2026-09-25 for issue #216 only.

- **Semantic change:** in every digest list (`exemplar_forms`,
  `concessions`, `unacceptable`, `preferred_variations`), `n` is now the
  number of **distinct deals** — distinct `example_ref.document_id` values
  (for `preferred_variations`: the entry's own `observation_ref` deal plus
  every `observed_positions` deal that SIGNED its `to` text — a deal where
  that text was `proposed_then_reversed` refused it and is not acceptance
  precedent) — in the group.
  It previously summed each member's `precedent_count`. The compiler
  stamps each row of a text with that text's deal count, so the old rule
  reported a text signed in k deals as n = k×k (the NDA example's
  assignment clause showed `n: 16`, band "often", on a 4-deal group). A
  row whose citation carries no `document_id` is non-conforming (every
  playbook schema requires `example_ref.document_id`), so it is never
  counted as a guessed deal: `build_digest` raises `ValueError` on it.
  `band` thresholds are unchanged and now apply to deal counts. The
  `n` description strings in `playbook.schema-0.3.json` (which still say
  `n` counts observations sharing the same normalized text,
  precedent_count-weighted) are superseded by this entry; they stay
  unchanged only because that schema file is frozen.
- **Why in place:** the owner decided on 2026-09-13 that the deal is the
  unit of precedent (epic #227). The old `n` over-counted deals, so digests
  already bound under `digest_version` 2 mis-stated precedent strength.
  Correcting `n` under the same version fixes those digests on the next
  compile instead of carrying the over-count forward.
- **Conformance vectors regenerated** (`scripts/generate_conformance_vectors.py`):
  `expected.digest` changes in 003–008 and 011–012, where each clause's
  single observed row with `precedent_count: 3` is now `n: 1`, band
  "rare". Their `input`, `canonical`, `content_hash` and `section_digests`
  are unchanged. Vector 013's input was rebuilt so every observed row is
  its own deal's single signed row for the clause, on a distinct
  `document_id` (the shape the compiler now emits), with
  `confidence.n_our_paper` set to the distinct our-paper deal count (29);
  it still pins the `n=10` "often" and `n=9`/`n=2` "sometimes"
  boundaries. One exception is deliberate: its collision group adds a
  second signed row, with another spelling the digest merges, to a deal
  that already carries one. The compiler never emits that shape; it
  models a hand-curated or legacy-store `observed_positions` input, which
  the digest must still count once per deal. 001, 002, 009 and 010 carry no clauses
  and are byte-identical. The generator now pins the vector set's
  `engine_version` stamp at `1.0.0` so a re-run reproduces the committed
  files exactly. The pin above is updated.

### 2026-09-25 — clause-tree `ClauseNode` gains optional `heading_span`; `char_span` covers the whole clause (issue #217)

`spec/clause-tree.schema.json`'s `$defs.ClauseNode` gained an optional
`heading_span` property (a 2-integer array, or `null`) recording the
heading line alone, and its `char_span` description now states what the
engine emits: the whole clause, from the start of its heading line through
the end of its own body text (children excluded) — the span OPF citations'
`char_span` resolve to (OPF-SPEC §4, whose wording already said so; the
ingesters previously emitted heading-only spans). `heading_span` is absent
when a node has no separate heading line (the synthetic pre-heading `"0"`
node, sub-clauses promoted from body text, LLM/agent-grounded nodes);
when present, `heading_span[0] == char_span[0]` and
`heading_span[1] <= char_span[1]` (`ClauseTree.validate` invariant 6).
Additive/optional — older serialized clause-tree files with no
`heading_span` key still validate and load unchanged. Like the `page`
entry below, this is the intermediate clause-tree artifact, not
`playbook.schema-0.3.json`: no `opf_version`/`digest_version` bump, and the
file stays outside the **Current pins** table.

### 2026-08-22 — Conformance vectors for canonicalization + digest (issue #115)

Added `spec/conformance/` (`manifest.json` + `vectors/*.json`): the
normative, plain-JSON, standalone-consumable conformance vector set for
canonicalization/content-hashing (`canonicalize.py`) and `digest` section
construction (`digest.py`), stamped `opf_version` 0.3 / `digest_version` 2 /
reference `engine_version` 1.0.0. `docs/OPF-SPEC.md` gained §10.2
documenting them as the normative definition. No schema, `opf_version`, or
`digest_version` change — this entry exists because canonicalization is
explicitly within this changelog's stated scope.
`tests/test_conformance_vectors.py` checks this engine against the same
frozen vectors; see `spec/conformance/README.md` for what edge case each
vector isolates. This is the mechanism issue #113's normative-rule-change
policy anticipated ("an independent [implementation] checked against the
conformance vectors") — the drift class it defends against is
Contract Toaster's hand-maintained source-level ports of these two files,
pinned to an engine commit by docstring rather than to anything
mechanically checkable until now.

**Fix round 1 amendment (same day):** added vector 013 (digest dedupe/rank/
top-N-plus-material capping and the "often"/"sometimes"/"rare" frequency-
band boundaries — `_dedupe_rank`/`_preferred_variations` in `digest.py`),
which vectors 001-012 left completely unexercised despite the normative
digest-construction conformance claim. Also added this table's
`spec/conformance/` pin, mechanically enforcing the "never edited in place"
rule `spec/conformance/README.md` already stated but nothing previously
checked (`tests/test_spec_consistency.py::test_spec_changelog_pins_conformance_vectors`).

### 2026-08-22 — OPF 1.0: stability policy and normative-rule-change policy (issue #113)

OPF-SPEC §11 gained two new normative statements, effective at 1.0: the
**stability policy** (1.x changes are additive-only — new OPTIONAL fields
and new `x_*` extensions permitted; no new REQUIRED field, no removing or
retyping an existing field, no changing a normative MUST without a 2.0
release) and the **normative-rule-change policy** (any new or changed
MUST — whether or not it touches the schema — gets its own entry under a
`### Normative rule changes` heading in `CHANGELOG.md`, in the release it
ships under). §3.13's id-uniqueness rule, recorded in `CHANGELOG.md` for
the first time by this same release, is the motivating case: it shipped as
a blocking validator rule with no schema change and no version bump, the
exact drift class both new policies exist to make greppable. No schema,
`opf_version`, or `digest_version` change — the document shape
(`opf_version` "0.3") is unchanged; this entry exists because both new
rules are normative validator rules, within this changelog's stated scope,
even though their durable record lives in `CHANGELOG.md`.

### 2026-08-03 — clause-tree `ClauseNode` gains optional `page` (issue #86)

`spec/clause-tree.schema.json`'s `$defs.ClauseNode` gained an optional
`page` property (`integer >= 1`, or `null`) recording the 1-based source
page the clause begins on, mirroring `playbook_engine/clause_tree.py`'s
new `ClauseNode.page`. Additive/optional — older serialized clause-tree
files with no `page` key still validate and load unchanged. This is the
intermediate clause-tree artifact ("intermediate artifact, not OPF" per
the schema's own `description`), not `playbook.schema-0.3.json` — no
`opf_version`/`digest_version` bump, and the file is intentionally outside
the **Current pins** table's CI-enforced hash check above (that table, and
`tests/test_spec_consistency.py::test_spec_changelog_pins_every_schema`,
cover only `playbook.schema*.json`).

### 2026-07-29 — sibling-id uniqueness is now a blocking normative rule (issue #70)

OPF-SPEC §3.13 (new): `evidence.clauses[].id`, `evidence.clause_library[].concept_id`,
`floor.invariants[].id`, and `corpus.documents[].document_id` MUST be unique
among their siblings, in every OPF version (0.1's top-level `clauses`/
`clause_library` shape included). Not schema-expressible, so enforced only by
`validator.validate_document` (`_check_duplicate_ids`, fail-closed) — a
duplicate sibling id made `export_profile`'s id-keyed sample/rewrite paths
collapse two entries onto one, silently shipping the first duplicate's
flagged text unmodified. No schema or `opf_version`/`digest_version` bump —
purely a new normative validator rule, same category as the digest
consistency check (§3.12).

### 2026-07-16 — OPF 0.3 FROZEN (at digest_version 2, engine PR #230)

`opf_version` 0.3 and `digest_version` 2 are frozen as of this entry. Any
further spec-affecting change goes to 0.4 (or digest_version 3). Consumers
should vendor `playbook.schema-0.3.json` at or after engine commit
`204057e` (merge of PR #230) — earlier same-day 0.3 states are superseded
(see below) and reject current artifacts.

### 2026-07-16 — digest budget enforcement (PR #230; digest_version 1 → 2)

- `preferred_variations` deduplicated/ranked/capped like the other lists;
  digest entries are now `{if, to, observation_ref, n, band}` projections
  (new `$defs.digestPreferredVariation`; the compiler-generated `rationale`
  stays in the full OPF). **Breaking for consumers of the day-old
  digest_version 1 shape**: entries validated as `acceptableIfEntry`
  (which requires `rationale`) no longer appear in digests.
- `build_digest` enforces the ~40K-token budget by construction (per-list
  cap tightens 5 → 4 → 3).
- `digest_version` bumped `"1"` → `"2"`.

### 2026-07-16 — digest list semantics change (PR #229) — RETROSPECTIVE NOTE

`concessions`/`unacceptable` moved from a 1:1 projection of
`summary.fallbacks`/`rejected` to deduplicated, precedent-weighted,
top-5-plus-material capped lists; `digestObservationSummary` gained
`n`/`band`. **This was a semantic change shipped without a
`digest_version` bump — a violation of the policy above, which this
changelog exists to prevent recurring.** It was corrected hours later by
PR #230's bump to digest_version 2 (which covers both changes); no
consumer had bound an artifact in the gap. Flagged by a downstream
review-engine team — the pin-and-verify discipline that caught it is the
intended consumer posture.

### 2026-07-16 — OPF 0.3 introduced (PR #227)

- New optional top-level `digest` section (`playbook.schema-0.3.json`,
  OPF-SPEC §3.12); validator accepts 0.1/0.2/0.3; digest covered by
  `identity.content_hash`; normative rules: digest clause ids must match
  evidence, no `full_text` anywhere in the digest.
- Single-file bundle artifact `playbook.opf.html` (`view bundle`) embedding
  the canonical JSON + digest in `<script type="application/json">` blocks
  (ids `opf-canonical`/`opf-digest`, `</` escaped as `<\/`).
- `playbook.schema-0.2.json` unchanged; 0.2 documents remain valid.

### Earlier

- **0.2** — three-section model (Evidence/Posture/Floor); see OPF-SPEC
  Appendix B.
- **0.1** — initial draft; retained for history (`playbook.schema.json`,
  `docs/OPF-SPEC-v0.1.md`).
