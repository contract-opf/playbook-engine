# Examples

| Path | What's there |
|---|---|
| [`judge-fixture/`](judge-fixture/) | Synthetic corpus + pre-computed ("canned") judge verdicts used by the Quickstart below and by `tests/test_cli_judge.py` |
| [`staging-fixtures/`](staging-fixtures/) | Corpus-layout variants (flat, CLM-nested, manifest) used by `playbook stage` tests |
| [`fixtures/`](fixtures/) | OPF documents used by the schema validator's test suite (valid + deliberately invalid) |
| [`affiliation-config/`](affiliation-config/) | A worked `playbook.config.yaml` for the Educational Affiliation Agreement taxonomy |
| `our-paper-baseline.v0.2.playbook.json` | A worked example playbook, current OPF v0.2 format |
| `our-paper-baseline.playbook.json`, `emergent-no-template.playbook.json` | Worked examples in the superseded OPF v0.1 format |

## Quickstart: judge-fixture → playbook

This walks you from a fresh clone to a validating `playbook.opf.json`, using the
committed synthetic fixture at [`judge-fixture/`](judge-fixture/) — **no
`ANTHROPIC_API_KEY` required**. Total wall time: well under a minute on the
fixture. Run every command below from the repo root.

The fixture ships `judge-fixture/canned-verdicts.jsonl` — pre-computed judge
verdicts, keyed by clause content hash, standing in for the LLM/attorney
review round a real corpus needs. Loading it into the verdict store *before*
the first `mine` means that first pass comes out fully judged (every clause
classified, every deviation and provenance call made) instead of queuing
everything as `needs_review`.

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
playbook judge-apply out/quickstart-demo --verdicts examples/judge-fixture/canned-verdicts.jsonl
```

Expected output:

```text
OK  loaded 11 verdict(s) into out/quickstart-demo/judge/verdicts.jsonl
```

### 3. Mine the corpus

Runs ingest, scope-gate, classification, alignment, and deviation assessment
for every agreement — replaying the verdicts just loaded instead of calling
an LLM.

```sh
playbook mine examples/judge-fixture/corpus --config examples/judge-fixture/config.yaml --out out/quickstart-demo
```

Expected output (tail):

```text
  judge store: out/quickstart-demo/judge/verdicts.jsonl (store-backed judges active)
  deal-alpha: 2 version file(s)
    4 observation(s)
  deal-beta: 1 version file(s)
    3 observation(s)
L1-L4 complete: 7 observations, 2 docs
```

### 4. Project the playbook

Compiles the observation store into `playbook.opf.json` — purely
deterministic rollup, zero LLM calls.

```sh
playbook project out/quickstart-demo --config examples/judge-fixture/config.yaml
```

Expected output (tail):

```text
Playbook written: out/quickstart-demo/playbook.opf.json
OK  out/quickstart-demo/playbook.opf.json
```

### 5. Validate

```sh
playbook validate out/quickstart-demo/playbook.opf.json
```

Expected output — the playbook is schema-valid:

```text
OK  out/quickstart-demo/playbook.opf.json
```

### 6. View it

Renders a self-contained, no-network review HTML with per-clause comment
boxes:

```sh
playbook view render out/quickstart-demo
```

Expected output:

```text
OK  out/quickstart-demo/playbook.review.html
```

Open `out/quickstart-demo/playbook.review.html` in a browser to see the
result.

### 7. Render a review prompt (optional)

Composes Evidence + Posture + Floor into a review-ready system prompt — pure
Markdown, pastable into any chat LLM alongside a contract to review:

```sh
playbook render-prompt out/quickstart-demo/playbook.opf.json --out out/quickstart-demo/review-prompt.md
```

Expected output:

```text
wrote out/quickstart-demo/review-prompt.md
```
<!-- quickstart:end -->

`out/quickstart-demo/` (and `out/` generally) is `.gitignore`d — safe to
delete and re-run at any time.

This walkthrough is executable documentation: `tests/test_quickstart.py`
parses the commands above out of this file and replays them in CI against
the committed fixture, asserting the final playbook validates. If the CLI's
flags or output text ever drift from what's written here, that test fails
until this file is updated to match.

## Docker variant

Same six steps, run inside the reproducible Docker image instead of a local
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
  -v "$PWD/out/quickstart-demo":/work/out \
  playbook-engine validate /work/out/playbook.opf.json

docker run --rm -it \
  -v "$PWD/out/quickstart-demo":/work/out \
  playbook-engine view render /work/out
```

No `ANTHROPIC_API_KEY` forwarding needed for this fixture run either — omit
`-e ANTHROPIC_API_KEY` entirely (it's only required for LLM-assisted stages
against a real corpus).
