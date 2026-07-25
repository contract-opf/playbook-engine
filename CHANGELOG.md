# Changelog

All notable changes to the OPF standard and the playbook-engine are
documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project uses semantic versioning (`opf_version` for the format;
pre-1.0 minor versions may break compatibility).

## [Unreleased]

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
