---
name: playbook-from-corpus
description: >-
  Use for any work on an OPF negotiation playbook — deriving one from a corpus of
  agreement files, OR authoring the Posture and Floor on a playbook that already
  exists. Use when the user has a folder of contracts and wants a
  playbook.opf.json with real semantics, acting as the LLM judge. Use when the
  user says "derive a playbook", "run the corpus pipeline", "mine and judge the
  corpus", or "produce the OPF playbook from agreements". ALSO use when the user
  says "run the posture interview", "author the posture", "add hard lines",
  "sign the floor", "my playbook has no posture", "update my playbook", or wants
  to climb the control ladder on an evidence-only playbook. Asks first whether
  this is a full derivation or an update to an existing out-dir, then walks only
  the steps that route needs — stage, lint-corpus, mine, checkpoint/inspect,
  judge (plan/subset/drain loop), judge-apply, project, posture interview, floor
  propose/sign, validate, report, view.
---

# playbook-from-corpus

Orchestrates the full negotiation-playbook derivation from a raw corpus of
agreement files to a validated `playbook.opf.json`, with the agent acting as
the LLM judge throughout the judgment stages. The engine commands (built in
prior slices) do the work; this skill coordinates the ordered pipeline and
decision points.

---

## Before you start

You need:

1. A corpus directory following the layout in
   [`docs/QUICK-COMPILE.md`](../../../docs/QUICK-COMPILE.md) (one subfolder
   per agreement, all negotiation versions inside).
2. A `playbook.config.yaml` referencing your taxonomy YAML and, if available,
   a baseline template.
3. A designation of which party is "us" (needed for provenance judgment) and
   the agreement display name (used in the report header). Collect these with
   one interactive prompt before beginning.

If the corpus has a nested layout (e.g. CLM export with `Versions/` subfolders),
run `playbook stage` first — it flattens the tree and writes `hints.yaml` files
derived from the export manifest.

---

## Step 0 — Pick the route (ask this FIRST, before anything else)

Most of this pipeline is expensive and unattended; the part that needs the
human is six questions. So establish which of these the user is actually
doing before running a single command:

> "Before I start — which of these are you doing?
>  **A. Full derivation.** You have a folder of agreements and no playbook yet.
>     I'll run the whole pipeline: stage → judge → project → interview → render.
>     Long (mostly unattended), and I'll need ~10 minutes of your attention near
>     the end.
>  **B. Author Posture / Floor on a playbook you already have.** There's a
>     validated `playbook.opf.json` with Evidence in it, and `posture` / `floor`
>     are empty. This is six questions and a sign-off — about ten minutes, no
>     re-derivation, no corpus needed.
>  **C. Re-derive Evidence from a changed corpus, keeping what you authored.**
>     The agreements changed; the Posture and Floor you already signed should
>     survive."

**Route B is the common case and the cheap one.** A playbook whose `posture` and
`floor` are `{}` is not broken — evidence-only is a complete, legitimate OPF
document (Rung 0, `docs/ADOPTING.md`). But it is also the state every freshly
derived playbook lands in, because Posture and Floor are *forward-looking
intent* that no corpus contains. If the user says "my playbook is missing
posture", they want Route B, not a re-derivation.

Confirm the route by measuring, never by asking twice — read the playbook and
say what you found:

```bash
python3 -c "
import json,sys; d=json.load(open(sys.argv[1]))
ev=(d.get('evidence') or {}).get('clauses') or []
print(f\"OPF {d.get('opf_version')} | evidence: {len(ev)} clauses | \"
      f\"posture: {len(d.get('posture') or {})} keys | floor: {len(d.get('floor') or {})} keys\")
" out/<name>/playbook.opf.json
```

| Route | Steps to run |
|---|---|
| **A** — full derivation | 1 → 11 (everything below, in order) |
| **B** — author Posture/Floor | **7a → 7b → 8 → 9 → 10** only |
| **C** — re-derive, keep authored | 1 → 7, then **re-run 7a/7b only if the interview answers changed**, then 8 → 10. Curation pins and signed Floor invariants survive a recompile by design; say so, and flag any conflict the recompile raises rather than resolving it silently. |

Route B needs no corpus and no config — every command in Steps 7a/7b reads and
writes the single out-dir. If the user only has an out-dir, that is sufficient;
do not ask them to produce a corpus they do not need.

One mechanical caveat: `make docker-run` **always** mounts `CORPUS` (defaulting
to `./corpus`, see the Makefile), so a Route B user with no corpus directory
will hit a mount error. On Route B either use the venv form directly —
`playbook posture interview $OUT ...` — or pass the out-dir as a harmless
placeholder: `make docker-run CORPUS=$OUT OUT=$OUT ARGS="..."`.

## Interactive setup (Route A and C only — skip for Route B)

Before starting, ask the human only what you cannot derive yourself:

> "Two quick questions before I begin:
> 1. What display name should appear in the playbook? (e.g. 'Educational Affiliation Agreement')
> 2. Do you have a baseline template file I should use as the canonical standard? (optional — I'll derive an emergent playbook if not)"

Do **not** ask "which party is us" or "who are your counterparties" — derive
those yourself in the next step. Record the two answers; they drive the report
header and deviation mode.

## Derive party names automatically (do this before `mine`)

Provenance judgment and born-safe pseudonymization both hinge on party names,
and asking the human for them is unreliable — a first-timer names their brand
("Acme") while the actual recitals name a legal entity ("Acme Holdings II,
LLC" — often a historical or subsidiary entity that predates the brand) and
per-deal defined terms ("Facility", "AGENCY"). So **read the corpus and
populate the config yourself:**

1. After staging, read each deal's opening recital/preamble (the signed copy is
   ideal). Extract, for each: the party that is **us** (our legal entity name(s)
   + the defined term used) and the **counterparty** (its full legal name).
2. Write both into `$CORPUS/playbook.config.yaml` **without asking** when you are
   reasonably confident:
   - `provenance.our_party_aliases`: every distinct form of *our* name you see
     across deals (legal entities + defined terms — e.g. `Acme Holdings II,
     LLC`, `Acme Holdings, Inc.`, `Acme`, `Facility`).
   - `provenance.known_entities`: each counterparty's full legal name **plus its
     obvious abbreviation/acronym** — pseudonymization matches whole tokens
     exactly, so add each counterparty's short form (e.g. `BSU` for "Beta
     State University") alongside the full name, or the short form leaks into
     ids/filenames unpseudonymized.
3. Only ask the human about names you genuinely cannot resolve from the text
   (an unrecognizable counterparty, an ambiguous "us"). Show the final lists for
   a single confirm, then proceed.

`mine` warns if none of `our_party_aliases` match any document text — that means
"us" is still wrong; fix it before trusting provenance. These lists drive
provenance judgment (step 7), born-safe pseudonymization, and the report.

---

## Running commands: Docker vs. venv

Per the repo's decided runtime direction (see README.md § Installation),
**Docker is the engine runtime; this skill is the orchestration layer** —
every `playbook ...` invocation below is shown wrapped in the documented
`docker run` / `make docker-run` invocation from the README/Makefile. If
you're on the venv install path instead, drop the wrapper and run the bare
`playbook ...` command that appears inside each `ARGS="..."` value directly —
both runtimes are fully supported; venv is equally valid for born-digital
corpora (README § Installation has the tradeoff table).

Set these once, pointing at your corpus and output directories on the host:

```bash
export CORPUS=/path/to/corpus   # host corpus dir (or Step 1's staged output)
export OUT=/path/to/out         # host output dir — engine writes here, and
                                 # you write verdicts.jsonl / feedback.json here too
```

Build the image once (`make docker-build`), then every step below is:

```bash
make docker-run CORPUS=$CORPUS OUT=$OUT ARGS="<subcommand> ..."
```

which expands to the full form documented in the README:

```bash
docker run --rm -i \
  -v "$CORPUS":/work/corpus:ro \
  -v "$OUT":/work/out \
  -e ANTHROPIC_API_KEY \
  playbook-engine <subcommand> ...
```

(`-i` only, no `-t`: an agent-driven shell has no TTY, and `-t` aborts with
"the input device is not a TTY". `make docker-run` already uses the
non-TTY-safe form.)

Two things this changes about the steps below:

1. **Paths inside `ARGS=` are container paths** (`/work/corpus`, `/work/out`),
   not host paths — the `-v` mounts above do the translation. Config files
   (`playbook.config.yaml`) must live *inside* the corpus directory (so
   they're visible under the read-only mount) — put yours at
   `$CORPUS/playbook.config.yaml` and reference it as
   `/work/corpus/playbook.config.yaml`.
2. **Any file the agent writes for the container to read on the next call**
   (a verdicts JSONL for `judge-apply`, an exported `feedback.json` for
   `view apply`) must be written under `$OUT` — the read-write mount. A file
   written anywhere else is invisible inside the container. `hints.yaml`
   edits are the one exception: those are written directly onto the host
   corpus tree with the agent's own file tools between runs, never through
   the container, so the read-only mount doesn't block them.
3. **Template mode — verify it actually activated.** When
   `baseline.template` is set, `playbook mine` prints
   `template standards: N clause(s) classified`. **If N is 0, template mode
   silently degraded to emergent** (no per-clause `our_standard`, no
   claimable stances) — do not proceed; investigate the template's
   segmentation/classification. On the agent/LLM segmentation path the
   template is segmented through the same store-backed path as the corpus
   documents (a blank form that also appears as a corpus version is a cache
   hit). Also expect: flipping a corpus from emergent to template mode
   re-keys every deviation verdict (payloads embed `our_standard`), so a
   full re-judging pass is normal, not a bug.
4. **Stage with `--copy`, and keep config + template inside the corpus.**
   Under the read-only mount, symlinks written by a host-side `stage` dangle
   inside the container, so stage with `--copy` (real files). The baseline
   template and `playbook.config.yaml` must live **inside** the corpus dir
   too — a relative `../../..` template path escapes `/work/corpus` and won't
   resolve. Put the template beside the config under `$CORPUS`.

**No API key? The deterministic path renders a full playbook.** Without
`ANTHROPIC_API_KEY` the engine uses deterministic segmentation and you act as
the judge — the pipeline still runs end-to-end to a validated
`playbook.opf.json` and rendered HTML. Clause boundaries are coarser and
clauses with no template match degrade to emergent/negotiable, but the run is
correct and honestly labelled. LLM segmentation (`segmentation.llm: true` +
a key) is the quality upgrade, not a requirement.

**Born-safe sidecar — keep it in one gitignored place.** Pass
`--entity-registry /work/out/entity_registry.json` to `mine` so the
sensitive alias→real-name registry co-locates with `alias_map.json` under
`$OUT` (both gitignored) instead of the machine-global
`~/.cache/playbook-engine/entity_registry.json` default. Under Docker the
in-container `~/.cache` copy is ephemeral anyway, so `$OUT/alias_map.json` +
`$OUT/entity_registry.json` are the only durable sensitive artifacts — no
post-run "purge the global registry?" step needed.

---

## Pre-flight: estimate and confirm (do this BEFORE any expensive step)

Extraction/OCR is by far the most expensive step — a scanned PDF can OCR
5–10× slower than a born-digital one (docling RapidOCR on CPU, up to a 600s
per-file timeout). **Before running `stage`/`segment`/`mine`, give the human a
concrete ETA and wait for a go-ahead** so a multi-hour run is never a surprise.

Run the bundled estimator against the (staged) corpus — it only uses
pdfplumber, so it runs on the host venv in seconds and needs no Docker:

```bash
python .claude/skills/playbook-from-corpus/estimate_runtime.py <corpus_dir>
```

It classifies every version (born-digital PDF / scanned PDF / DOCX), prints a
wall-clock **extraction ETA range**, a rough extracted-token size, the `$0`
API-cost note (key-free), and the approximate judgment load. Show that summary
to the human verbatim, then ask: proceed as-is, exclude the scanned agreements
to finish faster, or OCR the scans separately? Only start extraction once they
confirm. (Time constants are calibrated from a real 44-agreement / 161-version
run; they are estimates — present the range, not a promise.)

---

## Ordered pipeline

### Step 1 — Stage (if needed)

If the corpus is in a nested export layout:

```bash
make docker-run CORPUS=./raw-corpus OUT=~/.cache/playbook-engine/staging \
  ARGS="stage /work/corpus --out /work/out --copy"
```

`--copy` writes real files instead of absolute symlinks — required whenever the
staged output is bind-mounted read-only into a container (host symlinks dangle
there). Confirm the output: each agreement has its own subfolder, a `hints.yaml`
with `order` and `signed_version`, and no raw corpus files are modified. The
staged output directory becomes `$CORPUS` for every step from here on — copy the
baseline template and `playbook.config.yaml` into it too (see Docker note above).

### Step 1a — Assemble the corpus story (layout `unknown`)

Run this when `playbook stage` reports layout `unknown`, or the operator
points at a messy directory with no per-agreement structure — loose files in
one folder, an email-export tree, ad-hoc naming. The deterministic core
can propose a plan from file contents and embedded metadata but
cannot resolve everything on its own; this stage is where the skill, acting
as the LLM, arbitrates the low-confidence parts.

1. Propose a plan instead of staging directly:

   ```bash
   make docker-run CORPUS=./raw-corpus OUT=~/.cache/playbook-engine/staging \
     ARGS="stage /work/corpus --out /work/out --plan-only"
   ```

   Read the resulting `staging_plan.json`: `deals` (each with `deal_id`,
   `counterparty_guess`, `confidence`, `files`), `unassigned`, `warnings`.

2. For each low-confidence cluster (`confidence` well below 1.0) or each
   `unassigned` file, read *that document's own text/metadata* and decide:
   same deal as an existing cluster? a new deal on its own? out of scope
   (leave it for the scope gate — do NOT delete)? Record a one-line
   rationale for every decision by appending to that file's `evidence` list
   in the plan — the same auditability discipline the scope gate already
   uses for its rationale log.

   Example — a plan before arbitration (one confident deal, one ambiguous
   singleton, one unassigned file):

   ```json
   {
     "layout": "unknown",
     "deals": [
       {
         "deal_id": "acme-corp",
         "counterparty_guess": "Acme Corp",
         "confidence": 0.92,
         "files": [
           {"path": "doc_003.docx", "proposed_version": 1, "signed": false,
            "evidence": ["fs_mtime:2026-01-10", "signed:none", "party_candidate:Acme Corp"]},
           {"path": "doc_007.pdf", "proposed_version": 2, "signed": true,
            "evidence": ["fs_mtime:2026-02-02", "signed:filename_signed_cue", "party_candidate:Acme Corp"]}
         ]
       },
       {
         "deal_id": "deal-2",
         "counterparty_guess": null,
         "confidence": 0.41,
         "files": [
           {"path": "Fwd-FWD-LEGALADMIN-redline.docx", "proposed_version": 1, "signed": false,
            "evidence": ["fs_mtime:2026-01-22", "signed:none"]}
         ]
       }
     ],
     "unassigned": [
       {"path": "MBM-revised-5-15-2023.docx", "reason": "no cluster above threshold (nearest content distance 0.94 to any other file)"}
     ],
     "warnings": []
   }
   ```

   After arbitration — the skill read `MBM-revised-5-15-2023.docx` and
   `Fwd-FWD-LEGALADMIN-redline.docx`, found the same counterparty named in
   both bodies (never trusting the editor-initials filename), and moved the
   unassigned file into `deal-2` with a recorded rationale:

   ```json
   {
     "layout": "unknown",
     "deals": [
       {
         "deal_id": "acme-corp",
         "counterparty_guess": "Acme Corp",
         "confidence": 0.92,
         "files": [
           {"path": "doc_003.docx", "proposed_version": 1, "signed": false,
            "evidence": ["fs_mtime:2026-01-10", "signed:none", "party_candidate:Acme Corp"]},
           {"path": "doc_007.pdf", "proposed_version": 2, "signed": true,
            "evidence": ["fs_mtime:2026-02-02", "signed:filename_signed_cue", "party_candidate:Acme Corp"]}
         ]
       },
       {
         "deal_id": "deal-2",
         "counterparty_guess": null,
         "confidence": 0.41,
         "files": [
           {"path": "Fwd-FWD-LEGALADMIN-redline.docx", "proposed_version": 1, "signed": false,
            "evidence": ["fs_mtime:2026-01-22", "signed:none",
                         "skill_rationale: body text names the same counterparty as MBM-revised-5-15-2023; same deal despite unrelated filenames"]},
           {"path": "MBM-revised-5-15-2023.docx", "proposed_version": 2, "signed": false,
            "evidence": ["fs_mtime:2026-02-14", "signed:none",
                         "skill_rationale: moved from unassigned — body text matches deal-2's counterparty despite the editor-initials filename"]}
         ]
       }
     ],
     "unassigned": [],
     "warnings": []
   }
   ```

3. Ask the operator only the questions the content genuinely can't answer,
   batched into one message — e.g. "These 3 files reference no counterparty
   I can identify — do you recognize them?" Anything you *could* resolve
   from content or metadata proceeds without a question.

4. Show the operator the final assembled story — one short table per deal
   (deal, version order, which file is signed) — for a single confirm, then
   execute the plan:

   ```bash
   make docker-run CORPUS=./raw-corpus OUT=~/.cache/playbook-engine/staging \
     ARGS="stage /work/corpus --out /work/out --from-plan /work/out/staging_plan.json"
   ```

Hard rules for this stage:
- **Never drop a file silently.** A file still unassigned after arbitration
  is not deleted — list it in the run report and route it to a quarantine
  folder so a human can look at it later.
- **Never trust filenames over content.** An editor-initials filename
  ("...-Redline-MBM-...") or an email-forward folder name is never evidence
  of deal membership or counterparty on its own — only document text or
  embedded metadata is.
- **Never fabricate an ordering.** If two versions in a cluster genuinely
  tie (no content or timestamp signal resolves them), leave them flagged —
  that's what the existing L2 arbitration is for. Guessing here would
  silently corrupt the negotiation trail.

### Step 2 — Lint until clean

```bash
make docker-run CORPUS=./corpus OUT=./out \
  ARGS="lint-corpus /work/corpus --config /work/corpus/playbook.config.yaml"
```

Fix every **ERR** before continuing. **WARN** items are advisory. Re-run until
exit 0. Common fixes:

| Error | Fix |
|-------|-----|
| `CORPUS_NOT_FOUND` | Check the path |
| `DOC_NO_SUPPORTED_FILES` | Add `.docx`/`.pdf`/`.rtf` to the subfolder |
| `CONFIG_NOT_FOUND` | Create `playbook.config.yaml` from the example |
| `CONFIG_TEMPLATE_NOT_FOUND` | Fix the `baseline.template` path or set to `null` |

### Step 2a — Agent segmentation (key-free, optional)

Without an `ANTHROPIC_API_KEY`, `mine` falls back to the **deterministic**
segmenter, whose clause boundaries are hit-or-miss on real agreements (fine on
clean born-digital DOCX, coarse on irregular/table-heavy layouts). With
`segmentation.agent: true` in the config, **the agent segments instead** — the
same store-backed loop the judges use, no API key:

**Resume rule (check FIRST):** if `$OUT/segment-verdicts.jsonl` already
exists from a prior session, run `segment-apply` on it immediately — a prior
run may have banked hours of segmentation work and died one command short
(the verdicts are validated and QA-gated at apply, so a stale/partial file is
rejected loudly, never silently cached). Only re-segment items still pending
after that.

```bash
# 1. Emit the segmentation queue (one item per un-cached document version):
make docker-run CORPUS=./corpus OUT=./out \
  ARGS="segment /work/corpus --config /work/corpus/playbook.config.yaml --out /work/out"
# 2. Agent: read $OUT/segment/pending.jsonl. For each item, partition its
#    `blocks` into contiguous clause ranges — one SegNode per clause, with a
#    heading and best-fit taxonomy_id (classification happens here, in the same
#    pass). Write $OUT/segment-verdicts.jsonl (see REFERENCE.md § SegNode).
make docker-run CORPUS=./corpus OUT=./out \
  ARGS="segment-apply /work/out --verdicts /work/out/segment-verdicts.jsonl"
# 3. Re-run `segment` to confirm 0 pending, then `mine` replays it — no API call.
```

**Context discipline:** the pending queue embeds each document's full text
twice and can run to several MB — NEVER bulk-read it into your context.
Stream it item-by-item with a script (`jq -c`, or Python `json.loads` per
line), and generate the verdicts file programmatically: echo `canonical_text`
byte-for-byte from the pending item (the cache joins on its hash — one
character of drift is a permanent miss) and emit only your `nodes` as new
content. Judge every document's structure yourself — scripting the ECHO is
required; scripting the SEGMENTATION JUDGMENT is fabrication.

**Benefits:** semantic (not heuristic) clause grouping, and first-pass
classification for free (the judge loop then has **no `classify` items** left —
only deviation/provenance/scope).

**Honest ceiling:** the agent segments at **block boundaries** — it groups the
extractor's blocks but cannot sub-split one. So granularity is capped by
extraction (a document that extracts to a few giant blocks stays coarse);
character-exact within-block splitting remains the live-LLM segmenter's edge.
Skip this step and let `mine` use the deterministic segmenter when that's good
enough for the corpus.

### Step 3 — Mine the backbone

```bash
make docker-run CORPUS=./corpus OUT=./out \
  ARGS="mine /work/corpus --config /work/corpus/playbook.config.yaml --out /work/out \
        --entity-registry /work/out/entity_registry.json"
```

`--entity-registry` pins the sensitive alias→real-name registry into the
gitignored `$OUT` (see the born-safe note under "Running commands"); omit it
only if you deliberately want the machine-global `~/.cache` default. Runs L1–L4
(ingest, scope gate, classification skeleton, alignment, deviation placeholders).
Writes `scope.json`, `trail/`, `observations.jsonl`, `corpus_manifest.json`,
`normalized/`. If `provenance.our_party_aliases` matched no document text, `mine`
prints a WARNING — treat it as a signal that "us" is misconfigured (see "Derive
party names").

By default this uses the **deterministic** segmenter/classifier skeleton — no
LLM calls, no token spend. For a real corpus run, opt in to **LLM-first
segmentation** by adding a `segmentation:` block to `playbook.config.yaml`:

```yaml
segmentation:
  llm: true    # required to enable the LLM path at all
  batch: true  # submit the whole corpus as one Message Batch (~50% cheaper)
  cache: true  # cache segmentation verdicts in out/segmentation_cache.jsonl
```

- `llm: true` alone switches every document version to the LLM segmenter
  (`playbook_engine/llm_segmenter.py`), which also classifies each clause in
  the same pass — the deterministic classifier is bypassed for these
  documents. **There is no deterministic fallback**: if an LLM call errors,
  `playbook mine` fails loud rather than silently degrading.
- Add `batch: true` to use `llm_segmenter_batch.py`'s corpus-wide Message
  Batches call instead of one synchronous call per document version.
- Add `cache: true` so repeat runs over unchanged document content skip
  re-segmenting those versions.
- Add `normalize_trail: true` to additionally run LLM-based cross-version
  clause-label normalization.

This path requires a real `ANTHROPIC_API_KEY` in the environment (real token
spend) — it is not the stub/mocked path used in CI. Docling extraction
activates automatically inside the container when available (no flag needed;
`playbook_engine/extraction.py` prefers docling via `shutil.which`).
`playbook mine` prints which segmentation mode is active (e.g.
`segmentation: llm+batch+cache`) so you can confirm the intended path ran.

### Step 4 — Checkpoint: inspect before judging

**Do not spend judgment effort until the backbone is healthy.**

```bash
make docker-run CORPUS=./corpus OUT=./out ARGS="inspect /work/out"
```

Sanity-check before proceeding:

- Are trails ordered correctly (earliest draft → latest → signed)?
- Is the signed copy identified for each agreement?
- Are reversals detected where expected?
- Does provenance signal look plausible (not all documents the same)?

If trails or provenance signals are wrong, add `hints.yaml` files to the
relevant corpus subfolders (see the "Step 4 — Review the intermediates" section of `docs/QUICK-COMPILE.md`), then re-run
`playbook mine`.

### Step 5 — Estimate and trial

**Pre-flight — reuse a warm extraction cache; never re-OCR what's done.**
Extraction/OCR is the multi-hour step, and it is cached at
`$OUT/extraction_cache.jsonl` keyed by **file content hash plus the current
extractor environment** (docling on PATH or not) — so any run pointed at an
`$OUT` a prior or parallel run already used skips extraction for every
unchanged version extracted under the SAME environment, and goes straight to
segmentation/judging. **Always point `$OUT` at the same output directory
across runs** (that is the whole mechanism — a fresh `$OUT` throws the cache
away and re-OCRs from scratch). Installing or removing `docling` between runs
against the same `$OUT` deliberately does NOT replay the cache for affected
versions — each is re-extracted once under the new environment rather than
staying frozen at a possibly-worse legacy-only extraction (issue #77).

**You do not have to police this by hand.** A successful `mine`/`judge`
stamps `$OUT/run_manifest.json` with the environment that produced the
output directory (engine version and build, resolved extractor, cache format
versions, config hash), and the *next* `mine`/`judge` against that `$OUT`
checks itself against it before doing any work. When everything matches it
says nothing at all. When it doesn't — docling gone from the venv, a stale
container image running an older engine, a changed clause splitter — it
stops before touching anything and prints a plain-English explanation of
what would be redone, how to fix it, and a copy-pasteable block of
environment facts (no corpus content) to send to the maintainers. Re-run
with `--accept-environment-change` if the change was deliberate and the
rework is acceptable (issue #121).

Run the pre-flight estimator (host venv, seconds, no docling needed). It
checks the cache for **both** the `docling` and `legacy` key variants of
each file, regardless of whether docling is installed on the host running
the script — so it correctly *detects* a warm cache hit even when the corpus
was actually extracted inside the docker container (which has docling; the
host venv typically does not). Pass the **same `$OUT`** so it reports
what's already cached:

```bash
python .claude/skills/playbook-from-corpus/estimate_runtime.py ./corpus ./out
```

Only a hit under the **target environment** — the one the upcoming
`judge`/`mine` run will actually extract under — counts as 0 wall-clock.
This defaults to `docling`, since the documented pipeline always extracts
inside the container (`make docker-run ... mine/judge`, Dockerfile has
docling). A file cached under `legacy` only is reported separately (`N
version(s) cached under legacy only — will be re-extracted under docling`)
rather than folded into the "already extracted" count — it will genuinely
re-run. If you are intentionally extracting on the host instead of the
container (no docling there), set
`PLAYBOOK_ESTIMATE_TARGET_ENV=legacy` before running the estimator so
host-cached entries are credited correctly instead of being reported as a
needless re-OCR.

If it reports `~0m — corpus already extracted (cache hit)`, the expensive step
is done: proceed directly to `judge`/`mine` (they replay the cache
automatically — no flag needed). If it reports uncached versions (including
any reported as cached under the non-target environment only), that count
is the real remaining OCR cost; surface the ETA to the human before
committing.

**Plan:** count pending items and estimate token cost before committing to a
full-corpus pass.

```bash
make docker-run CORPUS=./corpus OUT=./out \
  ARGS="judge /work/corpus --config /work/corpus/playbook.config.yaml --out /work/out --plan-only"
```

Review the deduped counts by kind (classification, deviation, provenance),
the judgment token estimate (scaled from the real pending payload sizes, not
a flat guess), and the separate `Segmentation: N version(s) not yet cached`
line — LLM segmentation is a full block-stream call per un-cached document
version and is typically the largest spend in the run, so it must be part of
the go/no-go decision, not just the judgment total. Surface all of this to
the human before proceeding.

**Subset trial:** judge a small sample to validate judgment quality.

```bash
make docker-run CORPUS=./corpus OUT=./out \
  ARGS="judge /work/corpus --config /work/corpus/playbook.config.yaml --out /work/out --subset 20"
```

Read `out/judge/pending.jsonl`. Apply verdicts for the sample items, confirm
the format is correct, then proceed to the full drain.

### Step 6 — Unattended judge-drain loop

**Loop invariant:** repeat until `out/judge/pending.jsonl` is empty.

```bash
# Round N:
make docker-run CORPUS=./corpus OUT=./out \
  ARGS="judge /work/corpus --config /work/corpus/playbook.config.yaml --out /work/out"
# → writes $OUT/judge/pending.jsonl (freshly rewritten each round)

# Agent: read $OUT/judge/pending.jsonl; judge each item; write
# $OUT/my-verdicts.jsonl (must live under $OUT so the container can see it)
make docker-run CORPUS=./corpus OUT=./out \
  ARGS="judge-apply /work/out --verdicts /work/out/my-verdicts.jsonl"

# Loop: re-run judge to confirm queue is drained or get next batch
make docker-run CORPUS=./corpus OUT=./out \
  ARGS="judge /work/corpus --config /work/corpus/playbook.config.yaml --out /work/out"
# → if pending.jsonl is empty, loop exits
```

**Judging each item** (see `REFERENCE.md` for prompts):

- **Classification:** assign the best-fit `taxonomy_id` from the taxonomy, or
  `null` if the clause does not fit any entry. Low-confidence items: flag in the
  report, do not guess.
- **Deviation:** classify as `none` / `reworded_equivalent` / `substantive`
  against the baseline template hunk. Assess `risk_delta` direction and
  magnitude.
- **Provenance:** read the document's recital/header text to determine
  `our_paper` vs `counterparty_paper`. Unknown entity aliases: record for
  human review; do not silently guess.

**Low-confidence verdicts:** mark `needs_review: true` in the verdict. These
are listed in the after-action report (step 9) for human follow-up.

`needs_review` is **not** a valid OPF observation enum value — do not let
un-applied `needs_review` verdicts reach `project`. The drain loop must finish.

**Fresh queue each round:** `pending.jsonl` is rewritten from scratch each
`playbook judge` run. Do not accumulate across rounds.

**Termination guard:** the pending count must strictly shrink every round.
If it fails to shrink for 2 consecutive rounds, STOP and diagnose — do not
keep re-emitting the same verdicts. `playbook judge` now prints a WARNING
when pending items are re-queues of stored verdicts that failed replay
validation (bad enum value, wrong `basis`, missing `risk_delta` — see
REFERENCE.md for the exact enums); `judge-apply` also rejects those verdicts
up front with a line number. A queue that will not drain is a malformed-
verdict bug, never something to wait out.

### Step 7 — Project the playbook

Once `pending.jsonl` is empty:

```bash
make docker-run CORPUS=./corpus OUT=./out \
  ARGS="mine /work/corpus --config /work/corpus/playbook.config.yaml --out /work/out"
make docker-run CORPUS=./corpus OUT=./out \
  ARGS="project /work/out --config /work/corpus/playbook.config.yaml"
```

`mine` replays the verdict store to populate full semantics. `project` compiles
L5 (playbook assembly) deterministically from the observation store.

### Step 7a — Posture interview (Rung 1) — THE ONLY STEP THAT NEEDS THE HUMAN

Everything before this was derived from the corpus. Posture cannot be: it is
*forward-looking intent*, and no pile of past agreements contains it. This is
why a freshly projected playbook always has `posture: {}` — not a failure, just
the boundary of what derivation can know (OPF-SPEC.md §7).

**Never ask an open-ended question.** Every question is presented as
multiple choice with options computed from the corpus you just mined, and the
free-text escape is always available (`AskUserQuestion` renders an "Other"
choice automatically — never remove it, and never tell the user to type instead
of choosing). A blank "what is your risk appetite?" prompt makes the user do the
analysis you already did for them.

**Compute the options before asking.** Read `$OUT/observations.jsonl` and
`$OUT/round_moves.jsonl` and turn them into concrete choices:

```bash
# reversal candidates (seeds Q1/sacred), outcome + provenance mix (Q2/Q3),
# and rounds-to-settle per document (Q4)
python3 - <<'PY'
import json, collections, statistics
D = "OUT_DIR_HERE"
obs = [json.loads(l) for l in open(f"{D}/observations.jsonl") if l.strip()]
pb  = json.load(open(f"{D}/playbook.opf.json"))
titles = {c.get("taxonomy_id") or c.get("id"): c.get("title")
          for c in (pb["evidence"].get("clauses") or [])}
rev = collections.Counter(o.get("taxonomy_id") for o in obs
                          if o.get("outcome") == "proposed_then_reversed")
print("most REVERSED:", [(titles.get(k, k), v) for k, v in rev.most_common(8)])

# Rank the SECOND axis too — see the warning below. A clause that is usually
# ABSENT barely registers as reversed, because it was never there to fight over.
def doc_of(o):
    c = o.get("citation"); return c.get("document_id") if isinstance(c, dict) else None
docs = {doc_of(o) for o in obs} - {None}
seen = collections.defaultdict(set)
for o in obs:
    if o.get("taxonomy_id") and doc_of(o): seen[o["taxonomy_id"]].add(doc_of(o))
absent = sorted(titles, key=lambda t: len(seen.get(t, ())))
print("most ABSENT:", [(titles[t], f"{len(seen.get(t,()))}/{len(docs)}") for t in absent[:8]])
print("provenance:", dict(collections.Counter(o.get("provenance") for o in obs)))
rm = [json.loads(l) for l in open(f"{D}/round_moves.jsonl") if l.strip()]
per = collections.defaultdict(int)
for m in rm:
    if m.get("round"): per[m["document_id"]] = max(per[m["document_id"]], m["round"])
v = sorted(per.values())
print("rounds/doc: median", statistics.median(v), "mean", round(statistics.mean(v),1),
      "p90", v[int(len(v)*0.9)-1], "n", len(v))
PY
```

**Ask in order of importance, not in spec order.** OPF-SPEC.md §7 numbers the
questions 1–6, but only one of them creates hard-binding content. Lead with it:

| Order | id | Why it ranks here |
|---|---|---|
| 1 | `sacred_clauses` | **The only answer that becomes hard-binding.** Writes straight into signed `floor.invariants`; forces the outcome on every future review. |
| 2 | `risk_appetite` | Governs the default disposition on every non-material change — the most-exercised sentence in the Posture. |
| 3 | `leverage` | Frames whether the model argues from strength or accommodation. |
| 4 | `rounds` | Sets when to escalate rather than keep trading. |
| 5 | `flexible_clauses` | The mirror of Q1, but soft — shapes concession order, binds nothing. |
| 6 | `audience` | Style of the rationale, not the substance of the judgment. |

Batch them as **two `AskUserQuestion` rounds** (the tool takes at most 4
questions and 2–4 options each): rounds 1–4 first, then 5–6. Do not send six
separate prompts.

**Write options at medium depth.** Each option label is a short phrase; each
`description` states what choosing it *commits the playbook to*, and cites the
corpus number behind it. Not `"Collaborative"` — rather
`"Collaborative — we usually want the placement"` with a description naming that
94% of observations are on our own paper. The user should be able to choose
without reading anything else, and should learn something from the options
themselves.

**Order options within a question by what the corpus supports**, most-supported
first, so the likely answer is the first thing read. Never pad to four options
when two are honest.

**Rank the sacred-clauses options on BOTH axes — reversal count AND absence
rate.** This is the one mistake in this step that is easy to make and hard to
notice. Ranking candidates by "most reversed" surfaces the clauses that get
*fought over*, and is structurally blind to the clauses that are silently
*missing* — a clause that is not in the document cannot be reversed, so it
scores zero on the axis you are sorting by, no matter how badly it is needed.
Measured on a real corpus: Limitation of Liability appeared in **9 of 43
documents (20%)** — the single most-absent clause — and drew only 4 mentions
across 2,662 observations, so a reversal-ranked option set omitted it entirely
while the user's actual top hard line was *"limitation of liability must always
be present."*

So offer both kinds and say which kind each is:

- **"don't let them weaken X"** — from the reversal ranking; X is present and
  contested.
- **"X must be present at all"** — from the absence ranking; the Floor's job
  here is to fix what the history got wrong, not to encode it. Say the absence
  rate out loud in the option description ("present in 9 of 43 — your history
  is the argument FOR this hard line, not against it").

An absence-shaped invariant is often the most valuable thing in the whole
Floor, precisely because the corpus cannot propose it: the corpus is the record
of it not happening.

**Write the answers to a file as you collect them**, then apply them. The
interview is the expensive thing to lose — a crash mid-way should never cost
the user their answers:

```bash
# Question ids, if you need to confirm them:
playbook posture questions

# Write $OUT/posture-answers.json as you go, then:
playbook posture interview $OUT --answers-file $OUT/posture-answers.json
# Docker form: make docker-run OUT=./out ARGS="posture interview /work/out --answers-file /work/out/posture-answers.json"
```

Three rules that are easy to get wrong:

- **Three answers is enough.** Each answered question becomes one sentence of
  `posture.system_prompt`. Skipping questions is legitimate; inventing answers
  is not.
- **Q4 is different in kind, and signs itself.** Naming what is non-negotiable
  is a human authoring a hard line in natural language, so that answer is
  written **directly into signed `floor.invariants`**, attributed to the
  interview. This is *not* the auto-promotion OPF-SPEC.md §3.7 rule 4 forbids —
  that rule bars promoting a **compiler-derived** candidate without an explicit
  accept. A human-authored statement's sign-off is the act of writing it. Do not
  ask the user to "confirm" Q4 a second time in Step 7b; it will render there as
  an inert already-signed row.
- **Re-running is safe.** The same statement always resolves to the same id and
  updates in place — never a duplicate (a repeated id is a blocking validator
  error, §3.13). Offer to re-run whenever the user's answers change.

**If the human is not available**, stop here. Do not fabricate a Posture. Leave
`posture: {}`, note it as pending human input in the Step 9 report, and say
plainly that the playbook is a complete evidence-only document (Rung 0) that can
ship as-is — `render-prompt` will mark it **ADVISORY ONLY — NOTHING BELOW IS
BINDING** rather than shipping unmarked guidance.

### Step 7b — Floor: propose, then let the human sign (Rung 2)

Every ask the user's own history reversed is a *candidate* hard line — a pattern
the compiler may surface but must never assert on its own.

```bash
playbook floor propose $OUT --config $CORPUS/playbook.config.yaml
# writes $OUT/floor.candidates.json + a summary table.
# --config supplies the taxonomy, so candidates classified under a
# 'structural: true' entry (e.g. Parties & Recitals) are excluded — omit it
# and structural exclusion silently turns off (issue #106).
# --min-deals (default 2) additionally requires a reversal be corroborated
# across at least that many distinct documents before it becomes a candidate.
```

`floor propose` is a **review artifact only**. It never writes to the OPF `floor`
section and never promotes a candidate. Accepting is a human act, and it
round-trips through the review HTML — do **not** hand-edit `floor.invariants`:

```bash
playbook view render $OUT        # candidates appear as a "Proposed hard lines"
                                 # checklist above the clause sections of
                                 # $OUT/playbook.review.html
# Human: accept / reject / undecided on each, then click "Export feedback"
# Save the export to $OUT/feedback.json, then:
playbook view apply $OUT
```

`accept` signs the candidate into `floor.invariants` with attribution; `reject`
records that a human looked and declined, so it is not re-proposed. **Undecided
is the default and nothing expires unreviewed** — an undecided candidate simply
reappears next time. Tell the user that: it means they can stop halfway without
losing anything or leaving a landmine.

**A conditional hard line needs `floor sign`, not a candidate.** Both paths
above only ever produce "Do not concede on {clause type}." — fine for an
unconditional rule, but a real hard line is often conditional ("limitation of
liability, *if present*, must not be unilateral in the counterparty's favor"),
and neither template can express that; forcing it through either one just
garbles the sentence. When the human dictates a hard line like that, write it
down **verbatim**:

```bash
playbook floor sign $OUT --statement "Limitation of liability, if present, must not be unilateral in the counterparty's favor."
# Optional: --id a-slug, --rationale "attribution", and --clause TAXONOMY_ID
# (validated against --config's taxonomy — pass --config when you use --clause)
```

Re-running with the identical `--statement` under the same id is a no-op;
re-running with a different `--statement` under an id that already carries one
is refused, never silently overwritten. This is the ONLY sanctioned way to put
a hand-authored statement into `floor.invariants` outside of Q4/the
accept-checklist above — never hand-edit `floor.invariants` directly, even for
a one-word fix.

Keep the Floor small. §3.7.1's admission test is the standard — a Floor
invariant is something that forces the outcome on every review, fail-closed, so
a bloated Floor turns into noise the consumer must still evaluate on every run.

### Step 8 — Validate (must exit 0)

```bash
make docker-run CORPUS=./corpus OUT=./out ARGS="validate /work/out/playbook.opf.json"
```

A non-zero exit here means the pipeline is not done. Common causes:

- Residual `needs_review` verdicts that were not applied — drain the loop.
- Schema violations in a verdict — fix the malformed verdict and re-apply.

Do not paper over validation failures. Fix the root cause.

### Step 9 — Report and inspect

```bash
make docker-run CORPUS=./corpus OUT=./out ARGS="report /work/out --out /work/out/report.md"
make docker-run CORPUS=./corpus OUT=./out ARGS="inspect /work/out --out /work/out/inspection.md"
```

Review both outputs. The report surfaces:

- Corpus coverage (how many agreements contributed)
- Backbone health (trail quality, provenance distribution)
- Judgment economics (items judged, low-confidence count)
- Semantic coverage (classified vs unclassified clauses)
- Needs-attention items (unknown aliases, low-confidence provenance, and
  Posture/Floor fields that require the GC interview — listed, never invented).
  If `posture`/`floor` are still empty here, say so explicitly and offer Step 7a
  rather than filing it as a defect: evidence-only is Rung 0, a legitimate
  endpoint, not an unfinished state.
- Honesty section (what remains stubbed or unresolved)

### Step 10 — View (render + bundle) and digest

```bash
make docker-run CORPUS=./corpus OUT=./out ARGS="view render /work/out"
make docker-run CORPUS=./corpus OUT=./out ARGS="view bundle /work/out"
make docker-run CORPUS=./corpus OUT=./out ARGS="digest /work/out"
```

Writes exactly **two** HTML artifacts, both self-contained (open from the
host — the container has no browser), plus the digest sidecar:

- `$OUT/playbook.review.html` — the **internal annotation surface**: numbered
  items, comment boxes, Export-feedback button. Share with reviewers doing the
  correction pass. This is the only view that accepts `--alias-map`.
- `$OUT/playbook.opf.html` — **THE shareable/uploadable playbook** (OPF 0.3),
  and the only artifact to hand a stakeholder or a consuming review
  application. One file containing: the full readable document (stance chips,
  preferred variations, acceptable concessions, unacceptable asks, exemplar
  forms; empty Posture/Floor labelled pending), a digest summary, and the
  CANONICAL OPF JSON plus digest embedded verbatim in
  `<script type="application/json">` blocks (ids `opf-canonical`/`opf-digest`).
  Takes no `--alias-map` by design — it embeds the canonical, pseudonymized
  JSON, so resolving real names would leak them and break hash verification.
- `$OUT/playbook.digest.json` — the **model-facing digest** standalone
  (`playbook digest`): per-clause stances, preferred variations verbatim,
  concession/unacceptable summaries, frequency-banded deduplicated exemplar
  forms with `example_ref` drill-down. Target ~40K tokens; the command warns
  if it exceeds that.

The bare `$OUT/playbook.opf.json` remains the **canonical source of truth on
disk** — the bundle contains it, never replaces it. A consumer extracts the
`opf-canonical` block, JSON-parses it, and verifies `identity.content_hash`
against `playbook_engine.canonicalize.content_hash`.

`view bundle` writes `playbook.opf.html`; `playbook.document.html` is no
longer emitted as a separate artifact. `view bundle` takes no `--alias-map`
— only `view render --alias-map /work/out/alias_map.json` resolves real
names, for an internal-eyes-only copy; without it every rendering stays
alias-only.

---

## Step 11 — Publish party-anonymous (only when RELEASING publicly)

Skip this unless the playbook is going to be shared outside the org. It
produces a public artifact with the party's own name role-labelled, dates
coarsened, source paths stripped, and a **residue report** for sign-off.

```bash
make docker-run CORPUS=./corpus OUT=./out \
  ARGS="publish /work/out/playbook.opf.json --out /work/out/playbook.public.opf.json"
```

Two safety layers run automatically:

1. **Hard backstop** — if any *known* entity name (from the run's entity
   registry) survives, publish fails loud and writes nothing. Non-negotiable.
2. **`residue_report.json`** (written beside the output) — the
   list-independent sweep: every proper-noun-like string still present in the
   published text, needing no name list. This is the reviewer's checkable
   artifact.

**Then YOU (the agent) classify the residue report before any human sign-off.**
Read `$OUT/residue_report.json` and bucket every entry:

- **OUR-PARTY** — the publishing org's own names/aliases (expected).
- **PLACE** — governing-law states/cities (e.g. "State of New York") — benign.
- **GENERIC** — capitalized boilerplate that isn't a name ("Workers'
  Compensation", "Effective Date") — benign.
- **UNKNOWN** — anything that could be a counterparty (an institution or
  company name you cannot account for).

Hand the reviewer/GC a short grouped summary — **not** the raw JSON — with the
UNKNOWN bucket first. Then:

- **Any UNKNOWN** → do **not** publish. It means a counterparty name escaped
  pseudonymization (usually an incomplete `known_entities` list). Fix the
  entity extraction (re-read recitals + signature/notice blocks for that
  deal), re-mine, and re-run publish until UNKNOWN is empty.
- **UNKNOWN empty (only OUR-PARTY / PLACE / GENERIC remain)** → the artifact
  is name-clean; the residue report is what the GC signs off against.

The confidence comes from the sweep being **exhaustive over name-shaped
strings**, so "no counterparty names" is a checkable claim, not a promise that
a hand-built list was complete.

---

## Feedback re-entry

After a reviewer has annotated the HTML surface and exported `feedback.json`:

```bash
# feedback.json must be placed at $OUT/feedback.json (the writable mount) —
# see "Running commands" above.
make docker-run CORPUS=./corpus OUT=./out ARGS="view apply /work/out /work/out/feedback.json"
```

This writes corrected `hints.yaml` files and VerdictStore entries. Then
re-judge and re-project:

```bash
make docker-run CORPUS=./corpus OUT=./out \
  ARGS="judge /work/corpus --config /work/corpus/playbook.config.yaml --out /work/out"
# Agent: judge any new pending items, write $OUT/correction-verdicts.jsonl
make docker-run CORPUS=./corpus OUT=./out \
  ARGS="judge-apply /work/out --verdicts /work/out/correction-verdicts.jsonl"
make docker-run CORPUS=./corpus OUT=./out \
  ARGS="mine /work/corpus --config /work/corpus/playbook.config.yaml --out /work/out"
make docker-run CORPUS=./corpus OUT=./out \
  ARGS="project /work/out --config /work/corpus/playbook.config.yaml"
make docker-run CORPUS=./corpus OUT=./out ARGS="validate /work/out/playbook.opf.json"
make docker-run CORPUS=./corpus OUT=./out ARGS="report /work/out --out /work/out/report.md"
```

---

## Guardrails

- **Do not fabricate legal content.** Every verdict must be traceable to the
  actual clause text. Low-confidence judgments are flagged, not invented.
- **Unknown entity aliases** (party names not on the known-alias list) must be
  listed for human review — do not silently assume they are "us".
- **Real corpus stays out of git.** Stage under the user-owned cache dir
  (`playbook stage` defaults to `~/.cache/playbook-engine/staging`, not the
  world-readable `/tmp`) and keep local scratch files out of the repo (add
  any local scratch dir to `.gitignore`). Never commit corpus files or
  derivation outputs.
- **Token efficiency.** Deduplicate clauses by content hash before judging;
  `playbook judge` does this automatically. Judge changed hunks, not full
  documents. Use `--plan-only` to estimate before committing to a full-corpus pass.
- **Posture and Floor are never derived, and never invented.** They are
  forward-looking intent; no corpus contains them. When the human is available,
  run Step 7a/7b and let them author it. When the human is not available, leave
  `posture: {}` / `floor: {}` and list them as pending human input in the
  report — an evidence-only playbook is a complete OPF document, not a broken
  one. The only content that may enter `floor.invariants` without an explicit
  per-candidate accept is the interview's own Q4 answer and a `floor sign`
  statement, because a human wrote them (OPF-SPEC.md §3.7 rule 4). A
  compiler-derived candidate NEVER auto-promotes.

---

## Reference

- `REFERENCE.md` — judge prompts, guardrails, and machine-checkable
  done-criteria.
- [`docs/QUICK-COMPILE.md`](../../../docs/QUICK-COMPILE.md) — no-LLM smoke
  run: non-engineer corpus layout guide and hints format, stub judges only
  (structurally valid, semantically blank output).
- `docs/REAL-CORPUS-DERIVATION.md` — two-phase plan/execute discipline and
  guardrails for running a real (non-fixture) derivation end to end.
