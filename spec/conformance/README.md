# OPF conformance vectors — canonicalization + digest

These vectors are the **normative definition** of two things for the format
version stamped in `manifest.json`:

1. **Canonicalization and content addressing** (`playbook_engine/canonicalize.py`)
   — the whole-document canonical form, `identity.content_hash`, and the
   four `identity.section_digests` values (`evidence` / `posture` / `floor` /
   `curation`).
2. **The `digest` section** (`playbook_engine/digest.py`, OPF §3.12) — the
   compact model-facing projection of `evidence`.

An independent implementation (a different language, or a hand-maintained
port like the ones this suite exists to catch drift in — see issue #115)
that reproduces every `expected.*` value in `vectors/*.json` from that
vector's `input` **is conformant** with canonicalization and digest
construction for this format version. It needs no dependency on this repo
or on Python — every vector is plain JSON.

`tests/test_conformance_vectors.py` is the reference check: it recomputes
each vector from `input` using the engine's own `canonicalize.py`/`digest.py`
and asserts the result equals the vector's frozen `expected.*` values
byte-for-byte. The `expected.*` values are **frozen at generation time**
(via `scripts/generate_conformance_vectors.py`, a dev-only tool — see its
docstring) and never recomputed live by the test; that separation is what
makes the suite an actual drift detector instead of a tautology that always
passes.

## Format version binding

See `manifest.json`'s `format_version` object:

- `opf_version` — the document shape these vectors' `input`s use.
- `engine_version` — the `playbook_engine.__version__` the vectors were
  generated against.
- `digest_version` — the `digest.digest_version` the `expected.digest`
  values use (independent of `opf_version` — see OPF-SPEC.md §11).

Per the OPF-SPEC.md §11 immutability rule, once a consumer exists these
vectors are never edited in place for the same format-version stamp. A
format-version bump (a new `opf_version` or a new `DIGEST_VERSION`) gets a
**new, separately-stamped** vector set alongside the old one, not an
overwrite — exactly like a schema file itself.

## File layout

- `manifest.json` — the format-version stamp, the canonicalization/hash/
  digest algorithm in prose, and an index of every vector file.
- `vectors/NNN-<name>.json` — one vector per file:

  ```jsonc
  {
    "name": "…",
    "description": "…",             // what edge case this vector isolates
    "opf_version": "0.3",
    "engine_version": "1.0.0",
    "digest_version": "2",
    "input": { /* a canonicalization/digest fixture shaped like an OPF playbook document */ },
    "expected": {
      "canonical": "…",             // canonicalize_playbook(input)
      "content_hash": "sha256:…",   // content_hash(input)
      "section_digests": {          // compute_section_digests(input)
        "evidence": "sha256:…",
        "posture": "sha256:…",
        "floor": "sha256:…",
        "curation": "sha256:…"
      },
      "digest": { /* build_digest(input) */ }
    }
  }
  ```

Each vector's `input` is a **canonicalization/digest fixture**, not
necessarily a validator-clean OPF document: it exists to exercise
`canonicalize.py`/`digest.py`, not `playbook_engine.validator.validate_document`,
so most vectors are minimal or deliberately malformed in ways the validator
would reject (missing `corpus.documents` entries a citation would need to
resolve against, evidence with fewer supporting observations than §2.2's
depth rule requires, etc.) — canonicalization and digest construction don't
depend on validity. `010-floor-absent` is the sharpest case: it's
**deliberately schema-invalid** (`floor` is a required property; the whole
point of the 009/010 pair is to isolate present-but-empty vs. absent, which
requires actually omitting a required key). Do not route these inputs
through `validate_document` and treat a blocking error as a defect — it
isn't one.

## What each vector isolates

The vectors are deliberately paired so a mismatch tells you exactly which
rule broke, not just "something changed":

| Vectors | Edge case | Expected relationship |
|---|---|---|
| 001, 002 | Key ordering | `canonical`/`content_hash` **equal** — 002 is 001's content with every object's keys inserted in reverse order (top level and nested) |
| 003, 004 | Nested array order | `canonical`/`content_hash` **differ** — same two clause objects, `evidence.clauses` array reversed |
| 005 | Unicode literal emission | `canonical` contains the accented/CJK/emoji text **literally** (UTF-8), never `\uXXXX`-escaped |
| 006, 007 | Unicode normalization | `content_hash` **differs** — visually-identical "café" spelled NFC (precomposed) vs. NFD (decomposed); the engine does NOT normalize before hashing |
| 008 | Float/int formatting | `canonical` pins exact renderings: a whole-number float keeps its `.0`, plus float-precision, negative, zero, large-int, and exponential-notation cases |
| 009, 010 | Empty vs. absent field | `content_hash` **differs** (`floor: {}` vs. no `floor` key at all) but `section_digests.floor` is **equal** between the two |
| 011, 012 | Excluded run/curation metadata | `canonical`/`content_hash` **equal** despite unrecognizably different `identity`, `curation`, and `compiler.generated_at`/`run_id`; `section_digests.curation` still **differs** (curation is excluded from `content_hash` but keeps its own lineage digest, §3.11) |
| 013 | Digest dedupe/rank/cap + frequency bands | One clause's `observed_positions`/`acceptable_if`/`fallbacks`/`rejected` each carry more than `EXEMPLAR_TOP_N` (5) entries — `expected.digest` pins the deduped/ranked/capped output of `_dedupe_rank`/`_preferred_variations` (a normalize-collision pair merging into one entry, a `risk_delta`-material entry surviving the cap despite ranking outside the top-5, and a non-material entry actually dropped by the cap) plus the exact `n=10`→"often" and `n=9`/`n=2`→"sometimes" band-boundary values |

## Regenerating

Only when deliberately re-stamping for a new format version:

```bash
.venv/bin/python scripts/generate_conformance_vectors.py
```

Then add a `spec/CHANGELOG.md` entry — this directory is spec-affecting
content per that file's own scope note.
