# playbook-engine

**Your negotiation history is an asset. This compiles it.**

Every legal team already knows what it will accept — the knowledge just
lives in the one place nobody can query: years of redlines, tracked
changes, and signed PDFs. `playbook-engine` reads a folder of negotiated
agreements (the signed copy plus the drafts exchanged along the way) and
compiles a machine-readable **negotiation playbook**: for each clause,
your standard position, the variants you've accepted, what you've conceded
under pressure, what you've refused — and *how each negotiation actually
moved, round by round* — with **every statement cited to the exact
document, version, and span that proves it**.

The output is the **[Open Playbook Format (OPF)](docs/OPF-SPEC.md)** — an
open, versioned, JSON-Schema-validated standard. Any tool that reads OPF
can use your playbook; you are never locked into this engine, or any
engine.

```text
corpus of agreements  ──►  playbook-engine  ──►  playbook.opf.json
(DOCX / PDF / RTF,         (deterministic         every claim cited,
 signed + drafts)           + LLM judgment)        schema-validated
```

## Try it in one minute — no API key

The quickstart runs the full pipeline over a committed synthetic corpus
with canned judgments, from fresh clone to a validating playbook:

```sh
python3 -m venv .venv && source .venv/bin/activate && make install
playbook lint-corpus examples/judge-fixture/corpus --config examples/judge-fixture/config.yaml
playbook judge-apply out/quickstart-demo --verdicts examples/judge-fixture/canned-verdicts.jsonl
playbook mine examples/judge-fixture/corpus --config examples/judge-fixture/config.yaml --out out/quickstart-demo
playbook project out/quickstart-demo --config examples/judge-fixture/config.yaml
playbook validate out/quickstart-demo/playbook.opf.json
playbook render-prompt out/quickstart-demo/playbook.opf.json
```

That last command is the payoff: it composes the playbook into a
**review-ready system prompt** you can paste into any chat LLM next to a
contract you're reviewing. [examples/README.md](examples/README.md) walks
through every step's expected output, plus the Docker variant.

Ready for your own agreements? See **[docs/ADOPTING.md](docs/ADOPTING.md)**
— the path from a messy folder of deals to a curated, publishable playbook.

## What a playbook knows

OPF v0.2 is **one document with three sections**, each with a different
runtime binding — this is the design that makes it safe to point a
stochastic model at high-stakes legal work:

| Section | What it carries | Binding at review time |
|---|---|---|
| **Evidence** | What the corpus shows: accepted variants, fallbacks, refusals, per-round negotiation trails, held-rates — all cited | **Advisory** — the model reasons over it |
| **Posture** | Negotiation intent as prose, drafted from a short interview: risk appetite, what's sacred, what's flexible | **Soft** — shapes judgment, never a gate |
| **Floor** | The red lines, as judge-checkable invariants ("never accept uncapped liability") | **Hard** — a violation forces the outcome; the model cannot override it |

The knowledge is *descriptive, not prescriptive*: the format tells a
reviewer what your history shows and what you intend — it never freezes a
per-clause script. That's deliberate. Modern models reason well from
evidence; what they need is your evidence, your intent, and your
non-negotiables, cleanly separated.

Some things adopters tend to care about, built in from the start:

- **Citations resolve.** Every cited document version carries a sha256
  content address (`corpus.documents[].version_files`); `playbook
  resolve-citation` verifies a citation against your own corpus copy
  byte-for-byte. A playbook's `corpus.snapshot.manifest_hash` names the
  exact corpus state it was compiled from.
- **Negotiation dynamics, not just outcomes.** Who proposed each change,
  when, against which counterparty segment, and the round-by-round
  ask→landing trail — so a reviewer can tell deal-breakers from trading
  chips.
- **Confidentiality is architectural, not aspirational.** Known
  counterparty names are pseudonymized at ingest (*born-safe* — raw names
  never reach a stored artifact); `playbook publish` produces a
  party-anonymous export with an LLM residue pass over every free-text
  surface. See [SECURITY.md](SECURITY.md).
- **Extensible without forking.** The schema reserves an `x_*` vendor
  namespace at the sanctioned levels; extensions travel with the document
  and participate in its content hash.
- **Content-addressed identity.** Canonical serialization, whole-document
  `content_hash`, per-section digests — lineage is reconstructible, and
  attorney-pinned curation survives recompiles with deterministic
  conflict-flagging.

## How the compiler works

**Deterministic where possible, LLM only for judgment.** Extraction,
segmentation, version ordering, diffing, and assembly are reproducible
code paths; the LLM is reserved for semantic calls (what kind of clause is
this, did the risk shift, does this violate an invariant). Runs are cheap,
repeatable, and cache-aware — recompiling after adding one deal re-judges
only what changed.

```text
L1  ingest + segment      DOCX/PDF/RTF → normalized clause trees (+ tracked-changes side-channel)
L2  order versions        edit-distance chain anchored on the detected signed copy
L3  classify              clause → taxonomy entry (deterministic fast path, judge on the ambiguous band)
L4  diff + attribute      per-round diffs, reversals, who-proposed-what → cited observations
L5  compile + assemble    positions, fallbacks, trails, held-rates → validated playbook.opf.json
```

[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) covers each layer;
[docs/PLAN-FIRST.md](docs/PLAN-FIRST.md) explains which stages need an
`ANTHROPIC_API_KEY` and which run on a Claude plan via the packaged
[`playbook-from-corpus`](.claude/skills/playbook-from-corpus/) Claude Code
skill — for many teams, **no API billing is required at all**. The whole
skill path is one command from a clone:

```sh
claude "$(cat docs/prompts/create-playbook.md)"
```

That [committed prompt](docs/prompts/create-playbook.md) tells Claude Code
to treat you as a first-time user: it interviews you (where your files
are, which party is you, template or not), then drives the pipeline with
you as the judge.

Don't have a taxonomy? Ship with the type-neutral
[`builtin:cuad-base`](spec/taxonomy/cuad-base.yaml) (the genuine CUAD v1
41 categories, CC-BY-4.0-attributed), the general-commercial supplement,
or induce one from your own corpus with `playbook induce-taxonomy`.

## Installation

Two supported runtimes — pick Docker unless you're developing:

| Path | Extraction stack | When |
|---|---|---|
| **Docker** (recommended) | `docling` structure-preserving conversion + OCR for scans | Any real corpus |
| **Local venv** | Legacy per-format adapters (`pdfplumber`, `python-docx`, `pandoc`) — no OCR | Development, tests, born-digital corpora |

```sh
# Docker
docker build -t playbook-engine .
docker run --rm -it -v "$CORPUS":/work/corpus:ro -v "$OUT":/work/out \
  -e ANTHROPIC_API_KEY playbook-engine lint-corpus /work/corpus

# Local venv (Python 3.11+)
python3 -m venv .venv && source .venv/bin/activate && make install
playbook lint-corpus ./corpus --config ./playbook.config.yaml
```

`make docker-build` / `make docker-run` wrap the Docker path; the
extraction stack is a property of how you installed, not a flag.

## Repository layout

| Path | What's there |
|---|---|
| [`docs/OPF-SPEC.md`](docs/OPF-SPEC.md) | The Open Playbook Format standard, v0.2 (the keystone) |
| [`docs/ADOPTING.md`](docs/ADOPTING.md) | The adopter's path: quickstart → your corpus → curation → publishing |
| [`docs/prompts/create-playbook.md`](docs/prompts/create-playbook.md) | The launch prompt for the Claude Code skill path (`claude "$(cat …)"`) |
| [`docs/PLAN-FIRST.md`](docs/PLAN-FIRST.md) | Running on a Claude plan vs an API key, stage by stage |
| [`docs/CORPUS-LAYOUT.md`](docs/CORPUS-LAYOUT.md) | How to organize your input directory (and what to do if you can't) |
| [`docs/QUICK-COMPILE.md`](docs/QUICK-COMPILE.md) | The no-LLM stub smoke run (one-shot `playbook compile`) |
| [`docs/REAL-CORPUS-DERIVATION.md`](docs/REAL-CORPUS-DERIVATION.md) | Plan/execute discipline for a real (confidential) corpus run |
| [`docs/ORCHESTRATION.md`](docs/ORCHESTRATION.md) | The checkpoint-review loop for supervised derivations |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | The compiler pipeline, layer by layer |
| [`docs/OPF-BUNDLE-BOUNDARY.md`](docs/OPF-BUNDLE-BOUNDARY.md) | What OPF owns vs what a downstream review engine owns |
| [`docs/OPF-SPEC-v0.1.md`](docs/OPF-SPEC-v0.1.md) | The superseded v0.1 spec, retained for history |
| [`spec/`](spec/) | JSON Schemas (v0.2 + superseded v0.1) and shipped taxonomies |
| [`examples/`](examples/) | The flagship v0.2 example playbook, fixtures, and the quickstart corpus |

## Status

**OPF v0.2; pre-1.0.** The format may still change (breaking changes are
called out in [CHANGELOG.md](CHANGELOG.md) and the spec's Appendix B). The
engine's full pipeline is exercised end-to-end in CI — currently ~1,900
tests, all offline. Real-world derivation runs on a private educational-
affiliation corpus; a synthetic public showcase corpus is planned.

## Contributing

Start with [CONTRIBUTING.md](CONTRIBUTING.md) (dev setup is `make install`,
the bar is `make all`). Spec changes carry a higher bar than code changes —
the format is an interface others build on. Security and confidential-
material reporting: [SECURITY.md](SECURITY.md).

## License

Code: Apache-2.0. The OPF specification text: CC-BY-4.0, so the standard
can be freely adopted and adapted. Taxonomy data derives from
[CUAD](https://www.atticusprojectai.org/cuad) (The Atticus Project),
CC-BY-4.0 — see [NOTICE](NOTICE).
