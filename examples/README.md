# Examples

| Path | What's there |
|---|---|
| [`judge-fixture/`](judge-fixture/) | Synthetic corpus + pre-computed ("canned") judge verdicts used by the Quickstart below and by `tests/test_cli_judge.py` |
| [`staging-fixtures/`](staging-fixtures/) | Corpus-layout variants (flat, CLM-nested, manifest) used by `playbook stage` tests |
| [`fixtures/`](fixtures/) | Deliberately invalid OPF documents (a missing and an unknown `opf_version`) used by the validator's test suite |
| [`affiliation-config/`](affiliation-config/) | A worked `playbook.config.yaml` for the Educational Affiliation Agreement taxonomy |
| [`canary/`](canary/) | A tiny synthetic 4-document DOCX corpus (two negotiations, two tracked-changes redlines) that exists to fail loudly when the extraction layer moves: extractor identity, warm-cache replay with zero re-extraction and zero quarantine, and committed derivation counts (`tests/test_canary_corpus.py` / `make smoke-canary`) — see [`canary/README.md`](canary/README.md) |
| [`nda/`](nda/) | A synthetic second agreement type (Mutual NDA): the OPF 0.5 reference playbook at `nda/playbook.opf.json` (`opf_version` "0.5": per-deal precedent record, `digest_version` "4"; populated Posture/Floor, reproducible from committed inputs — `tests/test_nda_derive_reproducible.py`), plus a bare structural smoke path (`tests/test_nda_smoke.py` / `make smoke-nda`) — see [`nda/README.md`](nda/README.md) |

The engine emits and validates exactly one format, OPF 0.5. The worked
examples in the retired OPF 0.1 and 0.2 formats were removed with those
formats (issue #238); git history has them. `nda/playbook.opf.json` is the
reference playbook.

## Quickstart: judge-fixture → playbook

This walks you from a fresh clone to a validating `playbook.opf.json` with
populated HARD LINES and POSTURE — the product at full strength — using the
committed synthetic fixture at [`judge-fixture/`](judge-fixture/) — **no
`ANTHROPIC_API_KEY` required**. Total wall time: well under a minute on the
fixture. Run every command below from the repo root.

The fixture ships `judge-fixture/canned-verdicts.jsonl` — pre-computed judge
verdicts, keyed by clause content hash, standing in for the LLM/attorney
review round a real corpus needs. Loading it into the verdict store *before*
the first `mine` means that first pass comes out fully judged (every clause
classified, every provenance call made) instead of queuing everything as
`needs_review`. Deviation needs no verdict — it is the deterministic standard
check.

Each deal directory also carries a `hints.yaml` naming its executed (signed)
copy — the minimal synthetic RTFs have no signature blocks for the engine's
signed-copy detection to find, and without a signed version every observation
is withheld from the compiled playbook. On a real corpus the executed copy is
detected from signature blocks, `/s/` markers, or DocuSign certificates;
`hints.yaml` is the documented override for when that detection needs
correcting (see [docs/CORPUS-LAYOUT.md](../docs/CORPUS-LAYOUT.md)).

### 0. Install (one-time)

```sh
python3 -m venv .venv
source .venv/bin/activate
make install
```

See the main [README's Installation section](../README.md#installation) for
the Docker alternative (recommended for a real corpus).

<!-- quickstart:start -->
### 1. Lint the corpus

Checks the corpus layout and config before spending any compile time on them.

```sh
playbook lint-corpus examples/judge-fixture/corpus --config examples/judge-fixture/config.yaml
```

Expected output (tail):

```text
OK — no errors, 2 warning(s)
```

(The 2 warnings are expected and harmless for this fixture: no baseline
template configured, and `deal-beta` has only one version on file.)

### 2. Load the canned verdicts

```sh
mkdir -p out/quickstart-demo
playbook judge-apply out/quickstart-demo --verdicts examples/judge-fixture/canned-verdicts.jsonl
```

Expected output:

```text
OK  loaded 7 verdict(s) into out/quickstart-demo/judge/verdicts.jsonl
```

### 3. Mine the corpus

Runs ingest, scope-gate, classification, alignment, and the standard check
for every agreement — replaying the verdicts just loaded instead of calling
an LLM.

```sh
playbook mine examples/judge-fixture/corpus --config examples/judge-fixture/config.yaml --out out/quickstart-demo
```

Expected output (tail):

```text
  judge store: out/quickstart-demo/judge/verdicts.jsonl (store-backed judges active)
  deal-alpha: 2 version file(s)
    7 observation(s)
  deal-beta: 1 version file(s)
    3 observation(s)
L1-L4 complete: 10 observations, 2 docs
```

### 4. Project the playbook

Compiles the observation store into `playbook.opf.json` — purely
deterministic, zero LLM calls.

```sh
playbook project out/quickstart-demo --config examples/judge-fixture/config.yaml
```

Expected output (tail):

```text
Playbook written: out/quickstart-demo/playbook.opf.json
OK  out/quickstart-demo/playbook.opf.json
```

### 5. Run the Posture interview

Authors the OPF Posture from a short GC interview — six canonical
questions, answered here from a committed fixture file so the walkthrough
stays non-interactive (`--answers-file` skips the terminal prompt entirely).
This is the first rung of the [control
ladder](../docs/ADOPTING.md#how-much-control-do-you-want): a name-shaped
`sacred_clauses` answer (like the fixture's, below) is promoted directly into
`floor.invariants` — a human (here, the fixture file) authoring hard lines
outright, no compiled candidate to review first. A sentence-shaped item
(one reading as a full clause rather than a bare clause-type name) is
skipped instead and needs a follow-up `playbook floor sign` — see
ADOPTING.md.

```sh
playbook posture interview out/quickstart-demo --answers-file examples/judge-fixture/posture-answers.json
```

Expected output:

```text
OK  posture.version=1 written to out/quickstart-demo/playbook.opf.json
```

### 6. Validate

```sh
playbook validate out/quickstart-demo/playbook.opf.json
```

Expected output — the playbook is schema-valid:

```text
OK  out/quickstart-demo/playbook.opf.json
```

### 7. View it

Renders the one human-readable page: a self-contained, no-network `index.html`
with five tabs, plus the canonical OPF JSON and digest as machine-readable
blocks:

```sh
playbook view bundle out/quickstart-demo
```

Expected output:

```text
OK  out/quickstart-demo/index.html
```

Open `out/quickstart-demo/index.html` in a browser. Its tabs:

- **Start here**: the playbook's identity (`content_hash`, OPF version,
  perspective), the file paths, and the steps to install it in the toaster.
- **Playbook**: the posture, floor and digest per clause.
- **Evidence**: each clause's precedent (how it opened, what was signed, what
  was refused, deal counts).
- **Review (optional)**: the model's judgments you may confirm or change.
- **Posture & Floor**: the authored text, with edit fields.

Review and edits are optional; they save to `overrides.json` (the page's
Connect folder button in Chrome or Edge, otherwise Download edits), which
`playbook apply-overrides` folds in. Step 5's interview means the Floor and
Posture sections now carry real content instead of the empty-section markers.
The toaster reads `playbook.opf.json` directly; the page is for people.
<!-- quickstart:end -->

`out/quickstart-demo/` (and `out/` generally) is `.gitignore`d — safe to
delete and re-run at any time.

This walkthrough is executable documentation: `tests/test_quickstart.py`
parses the commands above out of this file and replays them in CI against
the committed fixture, asserting the final playbook validates. If the CLI's
flags or output text ever drift from what's written here, that test fails
until this file is updated to match.

## Docker variant

Same steps, run inside the reproducible Docker image instead of a local
venv (see the main README for why you'd pick Docker for a real corpus):

```sh
docker build -t playbook-engine .

docker run --rm -it \
  -v "$PWD/examples/judge-fixture":/work/corpus:ro \
  -v "$PWD/out/quickstart-demo":/work/out \
  playbook-engine lint-corpus /work/corpus/corpus --config /work/corpus/config.yaml

docker run --rm -it \
  -v "$PWD/examples/judge-fixture":/work/corpus:ro \
  -v "$PWD/out/quickstart-demo":/work/out \
  playbook-engine judge-apply /work/out --verdicts /work/corpus/canned-verdicts.jsonl

docker run --rm -it \
  -v "$PWD/examples/judge-fixture":/work/corpus:ro \
  -v "$PWD/out/quickstart-demo":/work/out \
  playbook-engine mine /work/corpus/corpus --config /work/corpus/config.yaml --out /work/out

docker run --rm -it \
  -v "$PWD/examples/judge-fixture":/work/corpus:ro \
  -v "$PWD/out/quickstart-demo":/work/out \
  playbook-engine project /work/out --config /work/corpus/config.yaml

docker run --rm -it \
  -v "$PWD/examples/judge-fixture":/work/corpus:ro \
  -v "$PWD/out/quickstart-demo":/work/out \
  playbook-engine posture interview /work/out --answers-file /work/corpus/posture-answers.json

docker run --rm -it \
  -v "$PWD/out/quickstart-demo":/work/out \
  playbook-engine validate /work/out/playbook.opf.json

docker run --rm -it \
  -v "$PWD/out/quickstart-demo":/work/out \
  playbook-engine view bundle /work/out
```

No `ANTHROPIC_API_KEY` forwarding needed for this fixture run either — omit
`-e ANTHROPIC_API_KEY` entirely (it's only required for LLM-assisted stages
against a real corpus).
