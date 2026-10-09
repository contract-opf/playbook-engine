# OPF 0.5 conformance vectors — canonicalization + digest_version 4 + manifest and dossiers

The separately stamped vector set for `opf_version` "0.5" / `digest_version`
"4" (issues #223, #233, #234, #240). It replaces the 0.4 set, which the 0.5
conversion retired (the same four documents with the opening evidence of
OPF-SPEC §3.5.5 added, plus vector 005), and the digest 3 expectations, which
digest 4 (OPF-SPEC §3.12.2) replaced in place: vectors 001-005 now expect the
digest 4 shape, and vectors 006 and 007 are new. The contract (see `../README.md`): every
`vectors/*.json` is a plain-JSON `{input, expected}` pair, and an independent implementation that reproduces each vector's
`expected.canonical` / `content_hash` / `section_digests` / `digest` (and, since
issue #228, `manifest` / `dossiers` / `provenance_index`) from its
`input` is conformant for this format version. `manifest.json` stamps the
format version and indexes the vectors.

| Vector | Isolates |
|---|---|
| `001-minimal-no-perspective` | The digest 4 skeleton: `perspective` present and `null`, `agreement_type` `{id, name}`, zero corpus counts, null signing dates, an empty `uncovered_clause_types`. |
| `002-variants-refused-and-exclusions` | Grouping by exact normalization (case/punctuation merge, a negator does not), the `last_signed` ordering with a `YYYY-Qn` quarter, representatives, refused asks grouped across deals, and every exclusion (standard text, unsigned deals, struck clauses). |
| `003-cap-totals-and-summary` | The top-5 cap with uncapped `n_variants_total`, the full ordering key, and the ≤ 300-char sentence-boundary summary. |
| `004-party-alias-grouping` | The grouping key's party neutralization (OPF-SPEC §3.5.4): two deals whose texts differ only by the counterparty's `Counterparty-<n>` alias group to `n_deals` 2 (signed variants and refused asks alike); the parties' places swapped stays a separate variant (`counterparty` and `party` are distinct tokens); and the order of two one-deal variants pins the `perspective.party` rewrite. |
| `005-opening-evidence` | The 0.5 precedent record's opening evidence (OPF-SPEC §3.5.5): `opened_with` and `opening_text` for an edited standard, a non-standard opening changed to our standard, a non-standard opening signed unchanged, a clause added in round 2 (`absent`), an unsigned deal (both null), a case-only edit (no distinct opening under the grouping key) and a struck non-standard opening (`signed_text` null, no refused ask). Digest 4 projects it: two deals opened with our standard and one kept it, one signed variant (three deals) with `n_from_standard` 1 and `n_unchanged` 1, and two changed openings (one ended at our standard, one struck). |
| `006-opening-rules` | The digest 4 opening rules on one clause of ten deals: `n_opened_standard` / `n_kept_standard`, `n_from_standard` and `n_unchanged` on each signed variant, and `changed_openings` — a non-standard opening that ended at our standard grouped with its respelling in another deal (`n_deals` 2, `n_to_standard` 1), a struck one with no refused ask (`n_struck` 1), one excluded because the same words are its own deal's refused ask, plus `absent` and an unsigned deal that count nowhere. |
| `007-equivalence-label-and-coverage` | The `vs_standard` label in the digest (OPF-SPEC §3.5.6, §3.12.2): equivalent variants collapse into one entry (`n_deals`, `n_texts`, two exemplars), the others keep the tier order (less protective and different concept, unjudged, more protective), `positions` counts signed deals by label, the refused ask and the changed opening carry their labels, and `uncovered_clause_types` lists the active and custom taxonomy entries with no evidence (an inactive one never appears). |
| `008-hard-rules-and-dossiers` | The hard-rule manifest, the critic dossiers and the provenance index (OPF-SPEC §3.12.3, issue #228): seven Floor invariants become seven rules in Floor order with their defaults (`required_presence` false, `judged`, no proof) and the three predicate specs; the venue dossier takes a concession on record first although another variant has more deals, then the first changed opening, from five candidate groups, and lists both Floor rules; a rule that demands presence or carries a predicate names its clause (the cross-reference rule names `clause.survival`); no text is ever cut part-way: the term dossier (no standard, so a 1,000-token budget) keeps its long first excerpt whole although it alone exceeds the budget, drops the second whole, naming it (`n_omitted`, `omitted_precedent_ids`), then drops its Floor rule whole (`n_floor_rules` 1, none listed), the one case where a dossier may exceed its budget; the survival dossier has no excerpt and three Floor rules over its budget, so the last listed is dropped whole and it fits (`n_floor_rules` 3, two listed); the assignment dossier pins how a group's record is chosen: the concession's own record (not the deal that signed the same variant as proposed, although that deal is the lower document and precedent id), then a `refused_ask` excerpt (the ask as `opening`, what was signed instead as `signed`, outcome `ask_refused`) from the latest-signed deals' lower `precedent_id`, not the lowest document; every deal but one records source-file hashes, so the provenance index carries the corpus snapshot hash and each excerpt deal's `version_files` (empty for the one without); the provenance index names each selected record and its source deal in selection order, marked `omitted` when the dossier dropped it: the term dossier's dropped excerpt resolves to deal-e, which no kept excerpt cites and which `documents` lists for that row. Every vector's `expected` now carries `manifest`, `dossiers` and `provenance_index` as well (empty rules, one dossier per clause for 001-007). |

**`signed_at` is producer-supplied, not reference-compiler output.** The
reference compiler never emits `signed_at` (it extracts no signing date and
never fabricates one — OPF-SPEC §3.5.4), so in reference output every
`first_signed` / `last_signed` is `null`. Vector 002's hand-set `signed_at`
values (including the `YYYY-Qn` representative) model a third-party producer
that does record signing dates; they pin how a conformant digest orders and
picks representatives when the field is present.

Every input is self-consistent (it passes `playbook validate`): precedent ids
and clause counts are stamped with the reference functions. All content is
synthetic. Regenerate only with
`scripts/generate_conformance_vectors.py`, and only for a
new format-version stamp — never in place.
