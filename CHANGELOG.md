# Changelog

All notable changes to the OPF standard and the playbook-engine are
documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project uses semantic versioning (`opf_version` for the format).
As of 1.0, the format's stability policy (spec §11) applies: 1.x changes
are additive-only, and any new or changed normative MUST — even one that
touches no schema field — gets its own entry under a `### Normative rule
changes` heading in the release it ships under.

## [1.0.1] - 2026-08-22

- **Round-move attribution now recovers real author attribution on the
  default extraction path** (issue #118, building on #112): tracked-change
  positions are bridged between docling's and the legacy DOCX adapter's
  text-coordinate spaces via order-preserving text-unit alignment, instead
  of relying on character offsets that only matched when both sides
  happened to already share a coordinate system. Measured against the
  production corpus: attribution precision improved from 61.6% to 91.3%
  and wrong-clause attributions dropped ~80%, where previously `moved_by`
  read `"unknown"` corpus-wide for any docling-extracted redlined document
  — i.e. nearly all of them, since docling is the default extractor.
- **`party_side_for_author` no longer defaults an unmatched author to
  `"counterparty"`** (issue #119): previously, once any
  `provenance.our_party_aliases` were configured, an author matching none
  of them was silently assumed to be the counterparty rather than reported
  as `"unknown"` — guessing a side the engine had no positive evidence for.
  A new `provenance.our_authors` config field (personal names, initials,
  and/or emails — distinct from the entity-name `our_party_aliases`) lets
  a corpus positively identify its own people; an author matching neither
  list now correctly reads `"unknown"`. **This changes output for any
  corpus with `our_party_aliases` already configured** — re-derive to pick
  up corrected attribution.
- **Signed-copy detection**: a trailer-matched signature section with no
  filled `By:`/`/s/` evidence and no heading corroboration now resolves to
  a confident, deterministic `unsigned_trailer_reference` basis instead of
  the ambiguous `empty_signature_section` bucket (issue #117), removing an
  LLM-arbitration escalation that `d9ffde7`'s absorbed-trailer fix had
  introduced for roughly a third of one measured corpus. Heading-matched
  empty sections are unchanged and still escalate to arbitration.
- `README.md` now links the real, corpus-derived, party-anonymous
  published playbook example (issue #99).

## [1.0.0] - 2026-08-22

Engine 1.0.0 / OPF 1.0 — first non-beta release. The format is no longer
"breaking changes possible until 1.0"; see the spec's §11 stability policy.

### Normative rule changes

- **§11 Stability policy** (docs/OPF-SPEC.md §11, normative, effective at
  1.0): within the 1.x series a release MAY add a new OPTIONAL field or a
  new `x_*` vendor extension; it MUST NOT add a new REQUIRED field, remove
  or retype an existing field, or change what an existing field means or
  how its contents are selected — any of those requires a 2.0 release.
- **§11 Normative-rule-change policy** (docs/OPF-SPEC.md §11, normative,
  effective at 1.0): any new or changed MUST — whether or not it touches
  the schema — MUST get an entry under this heading, in the release it
  ships under. This entry is the first one recorded under the policy it
  announces.
- **§3.13 Identifier uniqueness** (docs/OPF-SPEC.md §3.13, normative,
  effective 2026-07-29, pre-existing): `evidence.clauses[].id`,
  `evidence.clause_library[].concept_id`, `floor.invariants[].id`, and
  `corpus.documents[].document_id` MUST be unique among siblings, in every
  OPF version. This rule shipped (issue #70) as a blocking validator rule
  with no schema change and no version bump — the motivating case for the
  policy above — and had no `CHANGELOG.md` record until this entry.

### Removed

- Removed `playbook_engine/eval_harness.py`, `playbook_engine/review.py`,
  `playbook_engine/review_orchestration.py`, and `docs/ORCHESTRATION.md`
  (issue #110) — 1,368 lines of CLI-unreachable code: `eval_harness.py` had
  zero package importers (its own docstring scoped the live-eval run out of
  its purpose), and `review_orchestration.py` / `review.py` were reachable
  only from their own dedicated tests, never from `playbook_engine/cli.py`.
  `review.py`'s `write_review()`/`review.json` artifact was also read by two
  cross-cutting privacy tests (`tests/test_born_safe_holistic.py`,
  `tests/test_pipeline_llm_seg.py`) as a redundant secondary check —
  `review.json` is derived entirely from `scope.json` / `trail/*.json` /
  `observations.jsonl` / `corpus_manifest.json` / `coherence_flags.json`,
  each of which those tests already assert directly (`coherence_flags.json`
  is likewise swept directly by `tests/test_born_safe_holistic.py`), so the
  coverage is unchanged with those checks removed. `inspection_report.py`'s independent, CLI-reachable
  `_version_ingest_review_flags` / `_load_review_flags` (an optional,
  back-compat `review.json` sidecar reader) are untouched. The removed code
  is recoverable at commit `7737a1e` (the last commit where these files were
  present) for anyone reviving the eval-harness or checkpoint-orchestration
  work tracked by issues #151/#152 (pre-migration tracker numbers; not
  issues in this repo).
- `posture.rubric` removed — prose Posture + Floor + Evidence are the
  interface.

### Breaking

- **Breaking**: Removed the `playbook compile` and `playbook view document`
  CLI commands (issue #109) — both were redundant spellings of existing
  pipelines (`compile`'s options were exactly `mine`'s plus `--stop-after`,
  and `view document` was a self-declared deprecated alias). Use
  `playbook mine` followed by `playbook project` in place of `compile`, and
  `playbook view bundle` in place of `view document`.
- **Breaking**: `playbook_engine.publisher`'s party-scan vocabulary is now
  agreement-type-neutral by default (issue #107) — the education-specific
  role words (`educational`, `academic`, `affiliated`, `affiliate`) and
  stopwords (`school`, `student`, `students`, `university`, `college`) that
  used to be baked into `publish_playbook`'s defaults are no longer assumed.
  This changes two independent things for an education-flavored corpus:
  - Step-5.5's institution gate (not suppressed by `--accept-residue-risk`;
    individual matches can be exempted only via `--config`'s
    `scan_role_words_extra`) now HARD-BLOCKS publish, raising
    `PublishError`, on phrases it used to treat
    as a benign role/qualifier and let through, e.g. "the affiliated
    university" or "each educational university representative" (a bare
    "University" alone was never matched by this gate, before or after).
  - The advisory proper-noun sweep (pre-migration tracker #211; not an
    issue in this repo) (`residue_report.json`, does not block) surfaces
    more generic institution nouns like "school" or "student" that it used
    to treat as stopwords and silently drop.
  - `playbook publish` now has a `--config <path>` option: pass the
    corpus's own engine config (one carrying `scan_role_words_extra` /
    `scan_stopwords_extra` — see `config.py`'s module docstring, and the
    shipped `examples/affiliation-config/playbook.config.yaml`) to merge
    those words back into both the gate and the sweep and restore prior
    scan behavior. Without `--config`, publish uses the neutral defaults
    only, regardless of what the corpus was mined with.

### Added

- Negotiation dynamics in Evidence (§3.5.3): `proposed_by`, `observed_at`,
  `counterparty_ref`, `summary.stance_detail`, per-clause
  `negotiation_trail`.
- Resolvable citations (§4.1): `corpus.documents[].version_files` content
  addresses, `corpus.snapshot.manifest_hash`, `playbook resolve-citation`.
- Reserved `x_*` vendor-extension namespace (§10.1).
- Reference consumer: `playbook render-prompt` composes
  Evidence+Posture+Floor into a review-ready system prompt.
- Conformance vectors for canonicalization + digest (§10.2): `spec/conformance/`
  (`manifest.json` + plain-JSON `vectors/*.json`), the standalone,
  non-Python-dependency normative reference a downstream port of
  `canonicalize.py`/`digest.py` must reproduce to be conformant, checked
  against by `tests/test_conformance_vectors.py` (issue #115).
- `provenance.our_authors` config list (issue #119): the people-namespace
  counterpart to `provenance.our_party_aliases` — personal names,
  initials, and/or email addresses, matched against DOCX tracked-change
  (`w:ins`/`w:del`) author metadata separately from the entity/org names
  `our_party_aliases` holds. Optional; defaults to `[]`.

### Fixed

- **Correctness**: `observation_builder.party_side_for_author` no longer
  guesses `"counterparty"` for a tracked-change author that fails to match
  any configured `our_party_aliases` (issue #119). Previously, once ANY
  `our_party_aliases` were configured, an author matching none of them
  fell through to `"counterparty"` — the exact guess the function's own
  "never guess" docstring said never happens, and (verified against a real
  production corpus) a systematic one-directional bias, since a DOCX
  tracked-change `author` is a person's name/initials, a namespace
  `our_party_aliases` (entity/org names) was never going to match by
  containment. An author matching neither `our_party_aliases` nor the new
  `our_authors` is now `"unknown"`, symmetric with the already-correct
  no-aliases-configured case. **Behavior change**: for any corpus with
  `our_party_aliases` configured today, previously-`"counterparty"`
  `proposed_by`/`moved_by` values for authors not in `our_authors` become
  `"unknown"` on the next mine.
- **Correctness**: round-move/clause tracked-changes attribution
  (`moved_by`/`proposed_by`) now works on the default DOCX extraction path
  (issue #118). `TrackedChange.char_span` is always an offset into
  `docx_ingester`'s own paragraph-join text, but under
  `extraction.extractor: auto` (the default), the diffed `ClauseTree` for a
  DOCX usually comes from docling instead — a different coordinate space —
  so span-overlap candidate selection (`tracked_changes_overlay.
  enrich_clause_diff`, issue #112) was comparing numerically coincidental
  offsets on that path, reliable only on the legacy-adapter path. A new
  coordinate-space bridge (`extraction.bridge_tracked_change_spans`)
  aligns `docx_ingester`'s own text-unit stream against docling's block
  stream (order-preserving, so a repeated boilerplate clause aligns to its
  own position rather than the first occurrence) and translates each
  tracked change's span into the tree's actual coordinate space before
  matching runs; a version whose bridge can't be confirmed has its spans
  cleared rather than left as an untrustworthy raw value. A round-level
  fallback tier (`tracked_changes_overlay.round_level_fallback_attribution`)
  attributes an otherwise-unmatched clause change to a version's tracked-
  changes author when that version's side channel carries exactly one
  distinct author string — refusing outright whenever two or more distinct
  authors are present, never guessing between them. Also fixes
  `tracked_changes_overlay._jaccard` returning a false "perfect" `1.0` for
  two empty (all-stopword) token sets instead of `0.0`. **Behavior
  change**: a mine over a DOCX corpus using the default docling extractor
  now recovers real `moved_by`/`proposed_by` attribution where it
  previously read `"unknown"` corpus-wide; ships together with issue #119
  above so the recovered attribution resolves to `"us"`/`"counterparty"`/
  `"unknown"` correctly rather than converting honest unknowns into
  confidently wrong `"counterparty"` guesses.

## [0.2.0]

OPF v0.2 — the three-section model. Summary of the spec's Appendix B:

- Three-section document: **Evidence / Posture / Floor**, with the
  determinism boundary (§5) — Evidence advisory, Posture soft, Floor hard.
- `historical_stance` (descriptive) replaces `rollup.position`
  (prescriptive).
- `composes` — pinned external clause-intelligence modules, recorded for
  lineage (§3.4).
- Producer/author/consumer responsibilities (§6); Posture interview (§7);
  lineage boundary with the consumer (§8).
- `identity` — canonical serialization, `content_hash`, per-section
  digests, producer-assigned `id`/`version`/`supersedes` (§3.10).
- `curation` — embedded attorney-pinned positions surviving recompile with
  deterministic conflict-flagging (§3.11).

## [0.1.0]

- Initial draft: risk-delta model, provenance rule, dual structure (clause
  positions + clause library), citation requirement, taxonomy curation
  model.
