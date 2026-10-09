# OPF 0.5 conformance vectors — canonicalization + digest_version 3

The separately stamped vector set for `opf_version` "0.5" / `digest_version`
"3" (issues #223, #233). It replaces the 0.4 set, which the 0.5 conversion
retired: the same four documents with the opening evidence of OPF-SPEC
§3.5.5 added, plus vector 005. The contract (see `../README.md`): every
`vectors/*.json` is a plain-JSON `{input, expected}` pair, and an independent implementation that reproduces each vector's
`expected.canonical` / `content_hash` / `section_digests` / `digest` from its
`input` is conformant for this format version. `manifest.json` stamps the
format version and indexes the vectors.

| Vector | Isolates |
|---|---|
| `001-minimal-no-perspective` | The digest 3 skeleton: `perspective` present and `null`, `agreement_type` `{id, name}`, zero corpus counts, null signing dates. |
| `002-variants-refused-and-exclusions` | Grouping by exact normalization (case/punctuation merge, a negator does not), the `last_signed` ordering with a `YYYY-Qn` quarter, representatives, refused asks grouped across deals, and every exclusion (standard text, unsigned deals, struck clauses). |
| `003-cap-totals-and-summary` | The top-5 cap with uncapped `n_variants_total`, the full ordering key, and the ≤ 300-char sentence-boundary summary. |
| `004-party-alias-grouping` | The grouping key's party neutralization (OPF-SPEC §3.5.4): two deals whose texts differ only by the counterparty's `Counterparty-<n>` alias group to `n_deals` 2 (signed variants and refused asks alike); the parties' places swapped stays a separate variant (`counterparty` and `party` are distinct tokens); and the order of two one-deal variants pins the `perspective.party` rewrite. |
| `005-opening-evidence` | The 0.5 precedent record's opening evidence (OPF-SPEC §3.5.5): `opened_with` and `opening_text` for an edited standard, a non-standard opening changed to our standard, a non-standard opening signed unchanged, a clause added in round 2 (`absent`), an unsigned deal (both null), a case-only edit (no distinct opening under the grouping key) and a struck non-standard opening (`signed_text` null, no refused ask). Opening evidence is not projected into digest 3, so `expected.digest` is unchanged by it; the vector pins that the document canonicalizes, hashes and validates. |

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
