# Canary corpus

A tiny synthetic corpus whose only job is to fail loudly, in seconds, when the
extraction layer moves underneath a derivation.

## The incident this exists for

2026-08-22, production corpus re-derivation. `docling` had silently
disappeared from the host venv, so extraction fell back to the `legacy`
adapter. The extraction cache keys on `extractor_env` (issue #77), so every
version missed it; re-extraction under a different adapter produced different
canonical text; the segmentation cache keys on that canonical text, so every
document missed that too and quarantined.

**43 of 44 documents came back `AgentSegmentationPending`. Observations fell
from ~2,400 to 66.** The engine reported a *segmentation* problem. The fault
was two layers below it, in the extractor environment. Nothing in CI caught
it, because nothing in CI ever stated which extractor a run expects, or
replayed a warm out-dir to check the caches actually carried it.

`tests/test_canary_corpus.py` is that check. `make smoke-canary` runs it.

## What's here

```
config.yaml              -- the canary's engine config (see "Pinned settings")
corpus/
  halcyon-freight/v1.docx   -- our paper, clean first draft
  halcyon-freight/v2.docx   -- TRACKED-CHANGES REDLINE back from their counsel,
                               executed (w:ins/w:del by a single author)
  meridian-assay/v1.docx    -- their paper, clean first draft
  meridian-assay/v2.docx    -- TRACKED-CHANGES REDLINE back from us, executed
segment-verdicts.jsonl   -- the committed agent segmentation, replayed by
                            `playbook segment-apply` (no API key, no LLM)
expected.json            -- committed canary values: extractor environment,
                            corpus hashes, observation / round-move /
                            playbook-clause counts
build_corpus.py          -- regenerates the DOCX files, byte-deterministically
build_verdicts.py        -- regenerates segment-verdicts.jsonl
build_expected.py        -- regenerates expected.json
```

Four documents, two negotiations, both with a version pair so version-pair
alignment and round-move derivation are exercised; two of the four are
tracked-changes redlines, keeping that path (issue #84 / `bdbdc5b`,
`playbook_engine/docx_normalizer.py`) under a committed gate.

### Wholly synthetic

Fictional parties (Vantage Orbital Systems, Halcyon Freight Limited, Meridian
Assay Group), fictional plain boilerplate, invented reviewer names. Nothing
here is copied, adapted, or derived from any real agreement or from any
gitignored `*-corpus/` directory. The corpus is generated from
`build_corpus.py`, which is committed alongside it and is the whole source of
its content — there is nothing in these files that is not in that script.

## Pinned settings, and why

Three settings in `config.yaml` are load-bearing:

- **`segmentation.agent: true`** — the key-free, store-backed segmentation
  loop. This is the *only* headless path that touches
  `playbook_engine.extraction` at all: the deterministic ingest path never
  calls `extract_blocks` and so never opens `extraction_cache.jsonl`. Since
  the incident started in the extraction cache, the canary has to sit on the
  extraction path. (This is exactly why `make smoke-nda` could not have
  caught it — it runs cold, on the deterministic path.)

- **`extraction.extractor: legacy`** — pins the extractor environment.
  `auto` would resolve to docling on a docling-equipped host and legacy
  elsewhere, so the canonical text, the segmentation cache keys, and every
  committed count would silently depend on what happens to be installed. That
  is the class of drift this canary detects; it must not also suffer from it.
  Pinning turns the resolved environment into a stated **expectation**
  (`expected.json`'s `extractor_env`) that the test asserts against.

- **`provenance.our_authors`** — the personal-name namespace for
  tracked-change attribution (issue #119). `A. Whitfield` is our reviewer, so
  round moves get a real two-sided split (`us` vs `unknown`) instead of a
  uniform one.

The corpus is DOCX only, so the canary needs neither `pandoc` nor `docling` —
it is pure `python-docx`, and its CI job installs no system packages at all.

## Running it

```sh
make smoke-canary
# or
pytest tests/test_canary_corpus.py -q -m smoke
```

Roughly four seconds, keyless, no network.

### The same thing by hand

```sh
OUT=/tmp/canary-out
mkdir -p "$OUT"
playbook segment       examples/canary/corpus --config examples/canary/config.yaml --out "$OUT"
playbook segment-apply "$OUT" --verdicts examples/canary/segment-verdicts.jsonl
playbook mine          examples/canary/corpus --config examples/canary/config.yaml --out "$OUT"
playbook project       "$OUT" --config examples/canary/config.yaml
playbook validate      "$OUT/playbook.opf.json"
```

Re-running `segment` against a warm `$OUT` must report
`Segmentation pending: 0` — that line is the warm-replay assertion in
human-readable form.

## Regenerating the fixtures

In this order, from the repo root:

```sh
python examples/canary/build_corpus.py    # DOCX bytes  -> corpus/
python examples/canary/build_verdicts.py  # partition   -> segment-verdicts.jsonl
python examples/canary/build_expected.py  # golden file -> expected.json
```

Each stage depends on the previous one: the segmentation cache keys on
canonical text (a function of the DOCX bytes *and* the pinned extractor), and
the expectations are measured from a full run of both.

`expected.json` is a **golden file**. Regenerating it is a deliberate act
whose point is that the change shows up in `git diff`. If numbers move that
you did not intend to move, that is the canary working — find out what
changed before committing them.
