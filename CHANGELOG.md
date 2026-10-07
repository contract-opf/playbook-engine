# Changelog

All notable changes to the OPF standard and the playbook-engine are
documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project uses semantic versioning (`opf_version` for the format).
As of 1.0, the format's stability policy (spec §11) applies: 1.x changes
are additive-only, and any new or changed normative MUST — even one that
touches no schema field — gets its own entry under a `### Normative rule
changes` heading in the release it ships under.

## [Unreleased]

- **`playbook scorecard`: a counts-only scorecard of a derivation out-dir
  (issue #237).** Writes `<out-dir>/scorecard.json` and prints it as a
  table: documents, versions, signed and in-scope deals; template standards
  classified and `our_standard` coverage; classification by basis, the
  classified share of observations and distinct classified clause types
  per deal, each also split by paper side as a parity diagnostic only;
  precedent records, signed variants, refused asks and openings;
  `corpus.stats.dropped_observations` by reason; the template-drift
  distribution (per clause with `our_standard`, the share of our-paper
  signed deals that signed it); digest token estimate and capped clauses;
  and unanswered judge and segmentation queue items. The split of openings
  (precedent records with a non-null `opening_text`) by `opened_with`, the
  split of all precedent records by `opened_with` (#233) and dossier sizes
  (#228) are `null` until a playbook carries them.
  `--compare <scorecard.json>` prints the delta against an earlier card.
  The output is integers, ratios and closed-vocabulary labels only: every
  label read from the out-dir outside its vocabulary is written as
  `"other"`, every number read from it must be an integer count or is
  `null`, and nothing is keyed by a deal, file or clause type. To feed
  the basis counts, observations gain the vendor key
  `x_classification_basis`: how the cited node's taxonomy_id was reached,
  or `"aligned"` when it came from the aligned row. The L1-L4 stage cache
  version (`_DEVIATION_VS_TEMPLATE_VERSION`) moves to 16.

- **An earlier draft's copy of a clause joins the move row it belongs to
  (issue #232).** The aligner's global move phase chains a clause across
  drafts only at a near-exact or high-Jaccard match, and takes every clause
  it chains out of the taxonomy-bucket path. A clause edited in round one and
  then carried unchanged into the signed copy (v1 "five (5) years" → v2
  "three (3) years" == v3) was therefore chained from v2 onwards only. v1's
  copy had nothing left to bind to, and the deal showed a same-round
  `removed` + `added` pair. Downstream, that pair turned the opening text
  into a fabricated refused ask, or into a fabricated concession when it
  was our standard. After the move phase, a backward pass now offers each
  earlier draft's unmatched clauses to the move rows that start at the next
  draft, within one taxonomy bucket and under the bucket path's own bind
  rule (#222: Jaccard >= 0.70, or the localized-edit rescue, ranked Jaccard
  first). The bucket's free clauses in the next draft compete in the same
  ranking, so a row never takes a clause from a better partner. The pass
  runs newest draft pair first, so a row can be extended one draft at a
  time across several early edits. A clause a row already holds is never
  re-bound, and the row keeps its taxonomy_id (its latest member's). An
  extended row is `match_basis` "content_jaccard", and its
  `alignment_confidence` is its worst link's Jaccard. In the NDA example,
  six clauses become one row each. Four precedents change: governing law in
  `beta-industries` and two `gamma-holdings` clauses lose a fabricated
  concession, and `epsilon-systems`' compelled-disclosure clause loses a
  fabricated refused ask. Each now has `rounds` 1 and `opening_text` null,
  because the store records an opening text only for a clause struck before
  signing. `corpus.stats.dropped_observations` falls from 7 to 5. The L1-L4
  stage cache version (`_DEVIATION_VS_TEMPLATE_VERSION`) moves to 14.

- **A docling timeout recovered by a fallback is no longer cached; OCR-
  recovered text has its own reason (issue #231).** When docling times out
  on a version and the legacy fallback (pdfplumber for PDF, pandoc for RTF,
  python-docx for DOCX) or the `ocrmypdf` pass then recovers its text, that
  text is used for the run but stored in neither the extraction cache nor
  the per-deal L1-L4 stage cache, so the next `mine` retries docling on the
  version instead of leaving one legacy-extracted version among docling
  siblings for good (a mixed-extractor trail, the source of #122's alignment
  artifacts). A fallback after a docling crash, and a docling result the
  normalized-DOCX retry produced after a timeout, are cached as before.
  `version_ingest[].reason` and the published `x_ingest_reason` gain the
  value `ocr-recovered` for text recovered by the `ocrmypdf` pass;
  `backend-error` now means only a fallback that read the file's own text
  layer. `ocr-recovered` counts against `extraction.max_fallback` and in the
  `mine` fallback tally, and the inspection report flags it. The extraction
  cache format moves to 5: only stored `backend-error` fallbacks are
  re-extracted once (each may have been a timeout recovery or an OCR
  recovery), and `_VERSION_INGEST_REASON_VERSION` moves to 5.
  Also from #218, not noted at the time: on the deterministic segmenter, a
  scanned PDF (no text layer) used as `baseline.template` now raises
  `NoOCRRuntimeError` during template ingest. `mine` does not abort: it
  prints `WARNING: could not ingest template: ...` with the OCR remedy and
  runs WITHOUT a template (emergent mode, no "template standards" line).
  Before #218 such a template ingested as an empty tree and the run
  reported 0 template standards without saying why.

- **Signed anchors and version order from the documents themselves (issue
  #221).** Placeholder signature lines never count as signed: a `By:` value
  that is a bracketed placeholder (`[Name]`, `<Signature>`), `Name:`/
  `Title:`/`Date:` label residue, or has no run of two or more letters is
  blank, and `/s/ [Name]` is not an electronic signature. A section whose
  only filled values are generic captions (`By: Authorized Signatory`) is no
  longer `dual_signatures` at 0.90: it reads as not signed at 0.60 and
  escalates to a wired `signed_judge`. Each version's own timestamp (DOCX
  core-properties `modified`/`created` and the latest tracked-change
  `w:date`; PDF `/ModDate`) now seeds `order_versions`, so an unsigned
  deal's chain direction is no longer a lexicographic tie-break; `hints.yaml`
  timestamps still override per version, and the trail records
  `version_timestamps`. A deal with no detected signed copy gets no
  reversal detection and no `proposed_then_reversed` observations (counted
  under `dropped_observations` reason `refused_ask_no_signed_copy`), so its
  OPF 0.4 precedent records (`signed: false`) carry no `refused_asks`; the
  per-clause count of unsigned deals is what `evidence.precedent[].signed`
  already implies. No schema change. The L1-L4 stage cache moves to
  `_DEVIATION_VS_TEMPLATE_VERSION` 12.

- **Extraction reliability on scanned and slow documents (issue #218).** On
  the `extract_blocks` path (docling environment), a PDF that docling and
  pdfplumber return no text for gets a second OCR pass through
  `ocrmypdf --skip-text` (tesseract, already in the Docker image) before it
  is recorded as failed. A failure now carries a closed-enum reason
  (`timeout` or `no-text`) in `version_ingest[].reason`. A timeout is never
  cached, neither in the extraction cache nor in the per-deal L1-L4 stage
  cache (`out/.cache`), so the next `mine` retries it; a docling-environment
  no-text failure recorded while `ocrmypdf` was absent stops counting once
  it is installed, and `ocrmypdf` availability is part of the stage-cache
  fingerprint. The deterministic segmenter, which has no OCR in any runtime,
  now raises `NoOCRRuntimeError` on a PDF with no text layer instead of
  returning an empty tree, and the message names `segmentation.agent: true`
  / `segmentation.llm: true` on the Docker runtime. The published
  `corpus.documents[]` gains two `x_` extensions (no schema file changes):
  `x_mixed_extractors` (the deal's ingested versions came through more than
  one extractor) and `x_ingest_reason` (per-version reasons, index-aligned
  with `version_ingest`). `playbook doctor` lists `ocrmypdf`. The extraction
  cache format moves to 4 (only docling-environment PDF failures are
  invalidated) and `_VERSION_INGEST_REASON_VERSION` to 4.

- **The origin test for text removed before signing is the exact standard
  check (issue #229).** Whether a first-draft clause struck before signing
  was our standard language (our concession, `conceded_before_signing`) or
  theirs (their refused ask, `proposed_then_reversed`) is now decided by
  `deviation_classifier.is_standard_text`, the same exact match after
  `normalize_for_standard` (known party names neutralized) that #220 put on
  the consumer path, instead of the order-blind token Jaccard >= 0.92 it
  retired. A draft that flips a negation ("Neither party may assign" ->
  "Either party may assign"), deletes a mid-clause carve-out or negates a
  protection is their ask, no longer recorded as our concession. A fragment
  of a multi-node standard still counts as ours: its normalized text must
  occur on word boundaries inside the normalized whole standard (both sides
  run through `normalize_for_standard` with the same party names), so
  "either" no longer matches inside "neither". As before, a fragment that is
  a leading or trailing piece of our clause is still read as ours. The L1-L4
  stage cache identity (`_DEVIATION_VS_TEMPLATE_VERSION`) is bumped to 10.

- **OPF 0.4 + digest_version 3: the verdict-free per-deal precedent record
  (issue #223).** `playbook project` now emits `opf_version` "0.4"
  (`spec/playbook.schema-0.4.json`, a new file; 0.1/0.2/0.3 schemas are
  untouched and keep validating). `evidence` becomes `{clauses, precedent}`:
  one `precedent` record per (deal, clause) carrying the deal's signed text,
  whether it is our standard language (the deterministic exact check),
  `rounds`/`moved`, the text it opened with when our standard was struck,
  the asks refused before signing, and paper side as three-valued metadata
  that gates nothing; each clause carries `n_deals`, `n_signed_standard`,
  `n_variants` and `n_refused`, all distinct deals or distinct texts. No
  stance, band, risk or deviation verdict remains in evidence or digest;
  verdicts from an opt-in judged run go under the root `x_judgments`
  extension. The digest (`digest_version` "3") carries `perspective`,
  `agreement_type`, corpus counts, and per clause the grouped signed
  variants and refused asks with deal counts, citations and precedent ids.
  `playbook project --opf-version 0.3` keeps the 0.3 shape for one release.
  `validate`, `digest`, `view`, `report`, `render-prompt`,
  `resolve-citation`, `publish` and `export_profile` all read 0.4 (publish
  and export re-derive the digest after rewriting text, and publish
  coarsens `signed_at`). New conformance vectors under
  `spec/conformance/0.4/`; the 0.3 set is unchanged.
  `examples/nda/playbook.opf.json` is regenerated as 0.4.

- **Judged deviation/risk verdicts leave the consumer path (issue #220).**
  The consumer (a capable review model) does the judging; the playbook
  supplies precedent (owner decision 2026-09-13). With no deviation judge
  configured, now the default, L4 runs a deterministic standard check
  instead: each observation carries a `standard` fact (its text, with
  whitespace, case, punctuation and known party names normalized away,
  exactly equals the template clause for its
  taxonomy_id; a deal's clause split across nodes is compared as one merged
  text). There is deliberately no similarity tolerance: an order-blind
  token score absorbed reversed obligations ("Neither party may assign" to
  "Either party may assign"), deleted carve-outs and negation flips as our
  standard and `deviation` is derived from it: `none`
  when standard, `substantive` otherwise, `basis: "deterministic"`, with the
  neutral/none `risk_delta` placeholder OPF 0.3 still requires. Nothing is
  ever `needs_review` and nothing is queued, so identical signed text gets
  the identical answer whatever opening draft it was reached from. `project`
  reads no stance out of it: `historical_stance` is `no_signal`,
  `acceptable_if` and `fallbacks` are empty, and `stance_detail` is
  `{held: deals that signed our standard, of: every deal with the clause,
  basis: "all"}` (refused asks in `rejected` are kept). Judged deviation
  verdicts are an opt-in advisory layer for posture/floor work:
  `playbook judge --with-deviation-judge` / `playbook mine
  --with-deviation-judge`; by default the plan and the drain loop report
  `deviation: 0 pending`, and `judge-apply` notes that stored deviation
  verdicts only replay under the flag. Scope, classification and provenance
  judges are unchanged. The deviation-cache version is bumped to 9.
  `examples/nda/canned-verdicts.jsonl` drops its deviation verdicts and
  `examples/nda/playbook.opf.json` was regenerated. No schema file changed.

- **The deal is the unit of precedent (issue #216).** L4 now emits exactly
  one signed (or unsigned) observation per (deal, taxonomy_id), built from
  the signed version's own tree: its nodes for that clause joined in
  document order and cited to the first node. A clause split across several
  nodes is one precedent, not several. Text removed before signing is never
  `outcome: "signed"`. A clause with no signed slot whose own normalized
  text still occurs verbatim (fill-in blanks aside) in one clause-sized,
  contiguous stretch of signed nodes, for example a relocation the aligner
  left unpaired, is dropped and counted under the new
  `corpus.stats.dropped_observations` (reason `survives_in_terminal`).
  Otherwise it was removed, and the ORIGIN of its text decides what that
  means, never the deal's paper side (owner decision 2026-09-13): text
  narrowed or replaced before signing is removed even when its words recur
  in the signed copy; our standard (template) language struck — tested
  against every template node carrying the clause's taxonomy_id, not only
  the first — is our concession — the new
  engine-internal outcome `conceded_before_signing`, which never reaches
  `observed_positions`, `summary.rejected`, the digest's `unacceptable`
  list, render_prompt's refused asks or Floor candidates, and counts its
  deal as conceded in `stance_detail` and makes the position `negotiable`
  (only in a deal with a detected executed copy: in an unsigned deal it is
  dropped and counted, reason `removed_standard_no_signed_copy`, so a deal
  never shown to be executed is never a concession, issue #83);
  non-standard language struck is their refused ask,
  `proposed_then_reversed`, cited to the draft it came from (in an unsigned
  deal too, like any reversal); and a removal
  with no standard to compare against is dropped and counted (reason
  `removed_origin_undetermined`). Every detected reversal is its own
  `proposed_then_reversed` observation carrying its proposed text; a
  first-draft path number never claims it. `precedent_count`,
  `confidence.n_our_paper`/`n_counterparty_paper`, `stance_detail`
  held/of, and the digest's `n` all count distinct deals; the digest's
  `preferred_variations` count only deals that signed the text. Digest v2 `n`
  changed in place under an owner-authorized exception; the conformance
  vectors were regenerated and the change is recorded in
  `spec/CHANGELOG.md`. The deviation-cache version is bumped to 8, so a warm
  run rebuilds its observations. `examples/nda/playbook.opf.json` was
  regenerated. No schema file changed.

- **Citations span the whole clause; signature blocks leave clause text;
  summaries end on a sentence boundary** (issue #217). Every ingester
  (RTF/DOCX/PDF) now makes a clause node's `char_span` cover the whole
  clause — heading line through the end of its own body text, children
  excluded — instead of the heading line alone, so an OPF citation's
  `char_span` resolves to the clause language as OPF-SPEC §4 describes; the
  heading line moves to a new optional `ClauseNode.heading_span` (additive
  clause-tree schema field, see `spec/CHANGELOG.md`). A deterministic
  detector (`signed_detector.strip_signature_block`) cuts the execution
  trailer — "IN WITNESS WHEREOF", party captions, `By:`/`Name:`/`Title:`
  lines, signatory names — out of the clause it was absorbed into, after
  signed-copy detection, the our-party alias scan and provenance detection
  have read the unstripped tree; where it sat is recorded as
  `signature_block_span` on `corpus_manifest.json`'s `version_ingest` rows
  (engine-internal — the frozen OPF 0.3 `version_ingest` schema does not
  carry it, so the published playbook never does). `text_summary` — and with
  it `acceptable_if.if` and the digest's verbatim variant text — is now the
  first ≤ 300 characters ending on a sentence boundary (word boundary when
  the window holds no usable sentence end) instead of a mid-word 200-char
  cut. The per-doc stage cache is invalidated once
  (`_VERSION_INGEST_REASON_VERSION` 3, `_NORMALIZED_TREES_CACHE_VERSION` 2).
  `examples/nda/playbook.opf.json` regenerated: its counterparts clause no
  longer carries signature-block text (10 → 6 observations — the extra four
  were signature-block edits between drafts), and six canned verdicts that
  judged only signature-block differences were replaced by one for the
  remaining genuine counterparts rewording. No `opf_version` or
  `digest_version` change.

- **Canary corpus + CI gate** (`examples/canary/`,
  `tests/test_canary_corpus.py`, `make smoke-canary`): a four-document
  synthetic DOCX corpus — two negotiations, both with a version pair, two of
  them tracked-changes redlines — plus a hermetic, keyless CI job that
  asserts (a) the extractor environment a run resolves to matches a committed
  expectation, (b) a warm-cache replay performs **zero** re-extraction and
  quarantines **zero** documents with the L1–L4 stage cache deleted, so the
  extraction and segmentation caches alone must carry it, and (c)
  observation, round-move, and playbook-clause counts match
  `examples/canary/expected.json`. Written after the 2026-08-22 production
  re-derivation in which `docling` silently vanished from the host venv:
  extraction fell back to `legacy`, the extraction cache key changed
  (`extractor_env`, issue #77), the canonical text changed with it, the
  segmentation cache missed, and 43 of 44 documents quarantined as
  `AgentSegmentationPending` with observations falling from ~2,400 to 66 —
  reported by the engine as a segmentation problem two layers above the
  actual fault, and caught by nothing in CI.
  `tests/test_canary_corpus.py::test_canary_reproduces_the_incident` replays
  that exact cascade and asserts the new extractor check names the real
  layer first. Runs in ~4s and installs no system packages (DOCX only, no
  `pandoc`, no `docling`, no `ANTHROPIC_API_KEY`). No engine behavior change.
- **Extraction-cache format changes are now scoped instead of discarding the
  whole cache.** A change to `extract_blocks`'s output used to be expressed by
  bumping a `format_version` inside the cache KEY, which invalidated every
  entry for every file — a full corpus re-extraction (1h45m–5h17m on a
  44-document / 161-version corpus, against ~0 for a warm cache). Two such bumps landed within one month (three hours apart on the same day,
  so a corpus older than either paid one full re-extraction, not two), for
  changes that each affected only a subset of entries. A format bump is now a rung on a ladder
  (`playbook_engine/cache_format.py`) that declares what it did to stored
  entries: a `migrate` callback that rewrites them into the new shape, an
  `affects` predicate that names only the entries the change actually broke,
  or an explicit `discard_all=True` for a change that genuinely invalidates
  everything. `ExtractionCache` probes older-version keys on a miss and walks
  any hit up that ladder, so an entry survives unless a rung says otherwise.
  The two historical bumps are retro-fitted as the first two rungs: issue #81
  (structured `ExtractorLabel`) is a migration, and issue #84 (normalize-and-
  retry for redline DOCX) is a predicate matching only DOCX entries that
  recorded a docling→legacy fallback. Migration is conservative by
  construction — a predicate that cannot decide, or a label that cannot be
  reconstructed exactly, invalidates the entry rather than guessing it
  current. No cache file format change: stale entries are re-filed under the
  current key on first lookup, and `ExtractionCache(..., migrate=False)`
  restores the old cold-read behavior.
- **Judge verdicts are now versioned by rubric, so a criteria change
  invalidates the verdicts it should — and only those.** The verdict store is
  keyed by clause *content*, so previously a change to the judging criteria
  (the taxonomy under `spec/taxonomy/`, the answer vocabularies, the judge
  prompts in the `playbook-from-corpus` skill) replayed every banked verdict
  unchanged and unreported: a re-derivation seeded ~1,444 stored verdicts and
  re-queued only 246, with no way to tell whether the ~1,200 replays still
  held. Every verdict now carries the rubric it was produced under
  (`"<manual>+<derived>"`; see `playbook_engine/rubric.py`) — a
  hand-maintained half for the prose rubric, and a derived half that tracks
  the taxonomy for `classify` and the agreement-type definition for `scope`
  automatically. A hit stamped with a rubric that has since moved is
  re-queued instead of replayed, and `playbook judge` / `judge --plan-only`
  report the counts.
- **New `playbook judge-migrate`** — the upgrade path for an existing store.
  Reports how many stored verdicts are current / legacy (unstamped) / stale,
  and adopts the legacy ones by stamping them with the current rubric,
  preserving the banked human judgment rather than discarding it. Known-stale
  verdicts are re-stamped only under an explicit `--accept-stale` (scopeable
  with `--kind`). Re-stamping appends, leaving the prior record as an audit
  trail. Existing stores keep loading unchanged until migrated.
- **New `playbook judge --accept-stale` / `--strict-rubric`** — replay
  known-stale verdicts as-is for one run, or conversely treat unstamped
  pre-versioning verdicts as stale and re-queue them.
- `judge/pending.jsonl` records now carry the `rubric_version` in force when
  the item was queued, so `judge-apply` stamps each incoming verdict with the
  rubric the question was actually asked under (no `--config` needed).
Environment-drift prevention. Three real incidents on the same machine — a
vanished `docling`, a three-day-stale Docker image, and a symlink-staged corpus
that read as empty inside the container — all had the same shape: the
environment changed, nothing announced it, and the run completed looking like a
success. These changes make the wrong environment hard to be in, rather than
adding warnings after the fact.

- **`playbook stage` writes real file copies by default** (was: absolute
  symlinks). The documented Docker-first workflow bind-mounts the corpus
  read-only, where host symlinks dangle — and a dangling symlink is invisible
  to `Path.is_file()`, so a fully staged 161-version corpus reported "no
  .docx/.pdf/.rtf files found in any document directory" for every folder.
  `--copy` was documented as the fix but was not the default, so the documented
  primary path and actual practice had diverged. **`--symlink` is the opt-out**
  and prints plainly what it costs; `--copy` is still accepted and is now
  a no-op. Library callers: `staging.stage()` and
  `intake_plan.execute_staging_plan()` now default `copy_files=True`.
- **Broken symlinks in a corpus are named as such.** `lint-corpus` gains
  `CORPUS_DANGLING_SYMLINKS` (error) and `CORPUS_SYMLINKS_ESCAPE_ROOT`
  (warning), and suppresses the misleading `NO_SUPPORTED_FILES` /
  `DOC_NO_SUPPORTED_FILES` errors when broken symlinks are the actual cause.
  The message names the read-only container mount and the fix.
- **The preflight is now a precondition, not a suggestion.** `playbook mine`,
  `playbook segment`, and `playbook judge` run the `lint-corpus` checks before
  doing any work and refuse to start on errors, so an ad-hoc invocation can no
  longer skip them (#172). `--skip-preflight` opts out.
- **`playbook doctor`** — a new corpus-free, config-free command reporting the
  engine version, the container image stamp, and every external tool the
  pipeline can shell out to, with what each absent one silently costs. Backed
  by a new `playbook_engine.environment` module that *declares* that set in one
  place instead of leaving it implicit in `shutil.which` calls. `--strict`
  exits non-zero on any missing tool, for setup scripts and CI.
- **The Docker image is stamped and checked.** The build records the engine
  version and git commit as OCI labels and at
  `/etc/playbook-engine-image.json`, and refuses to build if the declared
  version disagrees with what pip installed. `make docker-run` now depends on a
  new `make docker-check`, which refuses to start a container whose engine
  version differs from the source checkout (an unlabelled image fails the same
  way) and warns on a commit mismatch. `ALLOW_STALE_IMAGE=1` opts out.
- **The scaffolded `playbook.config.yaml`** now carries a commented
  `extraction:` block explaining that the default `auto` degrades silently when
  `docling` is absent, and that declaring `extractor: docling` turns that into a
  refusal to start.
- **A run now records the environment that produced its output directory,
  and the next run checks itself against it** (issue #121). `mine`/`judge`/
  `segment` write `<out>/run_manifest.json` — engine version and git build, the
  resolved extractor environment, the extraction/stage cache format
  versions, the segmentation model/prompt/schema identity, and an opaque
  config+taxonomy hash — and preflight the next run against it before any
  work starts. Previously nothing read any provenance back on a later run:
  when `docling` vanished from a host venv, `extraction.extractor: auto`
  silently resolved to `legacy`, every version missed the
  `extractor_env`-keyed extraction cache (#77) and then the
  canonical-text-keyed segmentation cache, and 43 of 44 documents landed
  in `AgentSegmentationPending` quarantine with observations down from
  ~2,400 to 66 — reported two layers below the real fault.
  A matching environment prints **nothing**. A mismatch that would silently
  redo or invalidate work (extractor environment changed, extraction cache
  format bumped, engine downgraded, clause splitter changed) stops the run
  before it starts and explains in plain English what would be redone, how
  to fix it, and emits a copy-pasteable block of environment facts — no
  paths, party names, or contract text — safe to paste into a public issue.
  Pass `--accept-environment-change` to proceed anyway. Advisory differences
  (a config edit, an engine upgrade) print one short `note:` line and
  continue. Existing output directories have no manifest and are treated as
  a first run: silent, then stamped.

### Normative rule changes

For `opf_version` "0.4" a conformant validator MUST reject: a precedent id
that is duplicated or differs from its recomputation; more than one
precedent per (document_id, taxonomy_id); a precedent whose taxonomy_id
names no clause or whose document_id is not in `corpus.documents`; a
precedent whose `signed` disagrees with its document's `signed_version`;
`standard: true` with no `signed_text`; clause `n_*` counts that differ
from what `precedent` implies; a `signed_at` that is not a date or
quarter; and a digest that differs from `build_digest(document)`
(OPF-SPEC §3.5.4, §3.12.1).

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
