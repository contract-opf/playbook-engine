# Changelog

All notable changes to the OPF standard and the playbook-engine are
documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project uses semantic versioning (`opf_version` for the format;
pre-1.0 minor versions may break compatibility).

## [Unreleased]

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
  - The advisory #211 proper-noun sweep (`residue_report.json`, does not
    block) surfaces more generic institution nouns like "school" or
    "student" that it used to treat as stopwords and silently drop.
  - `playbook publish` now has a `--config <path>` option: pass the
    corpus's own engine config (one carrying `scan_role_words_extra` /
    `scan_stopwords_extra` — see `config.py`'s module docstring, and the
    shipped `examples/affiliation-config/playbook.config.yaml`) to merge
    those words back into both the gate and the sweep and restore prior
    scan behavior. Without `--config`, publish uses the neutral defaults
    only, regardless of what the corpus was mined with.
- Negotiation dynamics in Evidence (§3.5.3): `proposed_by`, `observed_at`,
  `counterparty_ref`, `summary.stance_detail`, per-clause
  `negotiation_trail`.
- Resolvable citations (§4.1): `corpus.documents[].version_files` content
  addresses, `corpus.snapshot.manifest_hash`, `playbook resolve-citation`.
- Reserved `x_*` vendor-extension namespace (§10.1).
- `posture.rubric` removed — prose Posture + Floor + Evidence are the
  interface.
- Reference consumer: `playbook render-prompt` composes
  Evidence+Posture+Floor into a review-ready system prompt.

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
