# Changelog

All notable changes to the OPF standard and the playbook-engine are
documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project uses semantic versioning (`opf_version` for the format).
As of 1.0, the format's stability policy (spec §11) applies: 1.x changes
are additive-only, and any new or changed normative MUST — even one that
touches no schema field — gets its own entry under a `### Normative rule
changes` heading in the release it ships under.

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
