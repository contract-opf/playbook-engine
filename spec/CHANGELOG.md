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

Current `DIGEST_VERSION`: **2** (`playbook_engine/digest.py`).

## History

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
