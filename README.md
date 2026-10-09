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
other engine.

```text
corpus of agreements  ──►  playbook-engine  ──►  playbook.opf.json
(DOCX / PDF / RTF,         (deterministic         every claim cited,
 signed + drafts)           + LLM judgment)        schema-validated
```

## Try it in one minute — no API key

The quickstart runs the full pipeline over a committed synthetic corpus
with canned judgments, from fresh clone to a validating playbook:

```sh
python3 -m venv .venv
source .venv/bin/activate
make install
playbook lint-corpus examples/judge-fixture/corpus --config examples/judge-fixture/config.yaml
mkdir -p out/quickstart-demo
playbook judge-apply out/quickstart-demo --verdicts examples/judge-fixture/canned-verdicts.jsonl
playbook mine examples/judge-fixture/corpus --config examples/judge-fixture/config.yaml --out out/quickstart-demo
playbook project out/quickstart-demo --config examples/judge-fixture/config.yaml
playbook posture interview out/quickstart-demo --answers-file examples/judge-fixture/posture-answers.json
playbook validate out/quickstart-demo/playbook.opf.json
playbook view bundle out/quickstart-demo
```

The payoff is that last command: it writes `out/quickstart-demo/index.html`,
the one human-readable artifact — a single tabbed page (Start here, Playbook,
Evidence, an optional Review, Posture & Floor) carrying the hard lines and
posture the interview step above authored, the toaster install steps, and its
canonical OPF JSON and digest embedded verbatim. Open it in a browser; reviewing
or editing anything in it is optional, and edits save to `overrides.json`
(`playbook apply-overrides` folds them in). The consuming review application
reads `playbook.opf.json` itself.
[examples/README.md](examples/README.md) walks through every step's
expected output, plus the Docker variant.

A second worked agreement type — proving the engine isn't hardcoded to one
contract type — lives at [`examples/nda/`](examples/nda/): a synthetic
six-deal Mutual NDA corpus compiled into a full playbook with populated
Posture and Floor at [`examples/nda/playbook.opf.json`](examples/nda/playbook.opf.json),
reproducible with no `ANTHROPIC_API_KEY` from the committed corpus + canned
judge verdicts (see [examples/nda/README.md](examples/nda/README.md#the-worked-playbook)
for the exact commands). A faster, bare-bones structural check of the same
corpus is available via `make smoke-nda`.

Ready for your own agreements? See **[docs/ADOPTING.md](docs/ADOPTING.md)**
— the path from a messy folder of deals to a playbook the review model can
use. Not sure how much of that authoring to do? ADOPTING.md's [control
ladder](docs/ADOPTING.md#how-much-control-do-you-want) lays out the rungs
from zero-effort to signed hard lines, each with an honest cost.

Want to see this run at real scale, not a six-deal fixture? A genuine,
corpus-derived playbook — 44 real negotiated agreements, 24 clauses, 555
substantive deviations judged and cited to precedent — is published
party-anonymous (the publishing party named deliberately; every
counterparty pseudonymized as `Counterparty-N`; dates coarsened to
quarters) at [contract-opf/playbooks](https://github.com/contract-opf/playbooks),
rendered at <https://contract-opf.github.io/playbooks/>. It's evidence-only
today (no Posture or Floor yet) — a demonstration of derivation quality,
not the full control ladder above.

## What a playbook knows

OPF 1.0 (document shape `opf_version` "0.5", the one format the engine reads and writes) is **one document with three
sections**, each with a different runtime binding — this is the design
that makes it safe to point a stochastic model at high-stakes legal work:

| Section | What it carries | Binding at review time |
|---|---|---|
| **Evidence** | What the corpus shows, one verdict-free precedent record per (deal, clause): what each deal signed, whether that is our standard language, whether the clause moved, and the asks refused before signing — all cited | **Advisory** — the model reasons over it |
| **Posture** | Negotiation intent as prose, drafted from a short interview: risk appetite, what's sacred, what's flexible | **Soft** — shapes judgment, never a gate |
| **Floor** | The hard lines, as judge-checkable invariants ("never accept uncapped liability") | **Hard** — a violation forces the outcome; the model cannot override it |

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
- **Negotiation dynamics, not just outcomes.** For each deal and clause:
  how many rounds the clause changed in and whether it moved at all, the
  text it opened with, and the counterparty asks refused before signing —
  so a reviewer can tell deal-breakers from trading chips.
- **Confidentiality is architectural, not aspirational.** Known
  counterparty names are pseudonymized at ingest (*born-safe*), but
  `known_entities` matching is best-effort (whole-word, contiguous-sequence)
  — a misconfigured or incomplete list can still leave real names in a
  stored artifact, so this is not a guarantee of pseudonymization on its
  own: run the skill's mandatory residue check before sharing a compiled
  playbook, and treat anything leaving the org as needing its own review —
  the engine has no command that anonymizes a playbook for public release.
  A few raw-source
  working files under `$OUT` (the judge queues — content a human reviews
  and answers *before* the pseudonymization pass ever runs) are not
  pseudonymized and must be handled as sensitive — see
  [ADOPTING.md](docs/ADOPTING.md) and [SECURITY.md](SECURITY.md).
- **Extensible without forking.** The schema reserves an `x_*` vendor
  namespace at the sanctioned levels; extensions travel with the document
  and participate in its content hash.
- **Content-addressed identity.** Canonical serialization, whole-document
  `content_hash`, per-section digests — lineage is reconstructible, and the
  Posture and Floor a person authored survive recompiles verbatim.

## How the compiler works

**Deterministic where possible, LLM only for judgment.** Extraction,
segmentation, version ordering, diffing, and assembly are reproducible
code paths; the LLM is reserved for semantic calls (what kind of clause is
this, which side drafted it). Runs are cheap,
repeatable, and cache-aware — recompiling after adding one deal re-judges
only what changed.

```text
L1  ingest + segment      DOCX/PDF/RTF → normalized clause trees (+ tracked-changes side-channel)
L2  order versions        edit-distance chain anchored on the detected signed copy
L3  classify              clause → taxonomy entry (deterministic fast path, judge on the ambiguous band)
L4  diff + attribute      per-round diffs, reversals, who-proposed-what → cited observations
L5  compile + assemble    clause types + template our_standard, per-deal precedent,
                          distinct-deal counts → OPF 0.5 playbook.opf.json + digest 4
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
# Docker — build and run through make, so the image is stamped with the engine
# version it contains and checked against this checkout before each run
make docker-build
make docker-run CORPUS="$CORPUS" OUT="$OUT" ARGS="lint-corpus /work/corpus"

# Local venv (Python 3.11+; needs pandoc for the RTF path)
brew install pandoc   # macOS — Debian/Ubuntu: apt-get install -y pandoc
python3 -m venv .venv && source .venv/bin/activate && make install
playbook lint-corpus ./corpus --config ./corpus/playbook.config.yaml
```

`.venv/` is disposable and machine-local — it hard-codes absolute paths, so it
never survives a move between machines. Rebuild it any time with the two
commands above; `make all` verifies the result.

`make docker-build` / `make docker-run` wrap the Docker path; the
extraction stack is a property of how you installed, not a flag. They expand to
the plain forms below — but the `make` targets are what stamp the image and
compare it to your checkout, so a raw `docker build` / `docker run` gives up
the staleness check:

```sh
docker build -t playbook-engine .        # unstamped — make docker-run will reject it
docker run --rm -i -v "$CORPUS":/work/corpus:ro -v "$OUT":/work/out \
  -e ANTHROPIC_API_KEY playbook-engine lint-corpus /work/corpus
```

### Confirming the environment

`playbook doctor` reports what the machine actually provides — engine version,
whether you are inside the project's image and which commit it was built from,
and every external tool the pipeline can shell out to, with what each missing
one costs you. It needs no corpus and no config, so it can be the first thing
you run:

```sh
playbook doctor          # or: make docker-run ARGS="doctor"
playbook doctor --strict # non-zero if anything is missing — for setup scripts and CI
```

Two guards run without being asked:

- **`make docker-run` refuses a stale image.** It compares the image's stamped
  engine version against the source checkout and stops if they differ, because
  a stale image otherwise produces a derivation missing exactly the fixes the
  run was performed to apply — and it looks like a success. Rebuild with
  `make docker-build`, or set `ALLOW_STALE_IMAGE=1` to run the old image on
  purpose.
- **`playbook mine`, `playbook segment`, and `playbook judge` run the
  `lint-corpus` checks first** and refuse to start if any fail, so an ad-hoc
  invocation cannot skip the preflight. `--skip-preflight` opts out.

## Repository layout

| Path | What's there |
|---|---|
| [`docs/OPF-SPEC.md`](docs/OPF-SPEC.md) | The Open Playbook Format standard, v1.0 (the keystone) |
| [`docs/ADOPTING.md`](docs/ADOPTING.md) | The adopter's path: quickstart → your corpus → posture and floor → using the playbook |
| [`docs/prompts/create-playbook.md`](docs/prompts/create-playbook.md) | The launch prompt for the Claude Code skill path (`claude "$(cat …)"`) |
| [`docs/PLAN-FIRST.md`](docs/PLAN-FIRST.md) | Running on a Claude plan vs an API key, stage by stage |
| [`docs/CORPUS-LAYOUT.md`](docs/CORPUS-LAYOUT.md) | How to organize your input directory (and what to do if you can't) |
| [`docs/QUICK-COMPILE.md`](docs/QUICK-COMPILE.md) | The no-LLM stub smoke run (`playbook mine` → `playbook project`) |
| [`docs/REAL-CORPUS-DERIVATION.md`](docs/REAL-CORPUS-DERIVATION.md) | Plan/execute discipline for a real (confidential) corpus run |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | The compiler pipeline, layer by layer |
| [`docs/OPF-BUNDLE-BOUNDARY.md`](docs/OPF-BUNDLE-BOUNDARY.md) | What OPF owns vs what a downstream review engine owns |
| [`docs/OPF-SPEC-v0.1.md`](docs/OPF-SPEC-v0.1.md) | The superseded v0.1 spec, retained for history |
| [`spec/`](spec/) | The JSON Schema — `playbook.schema-0.5.json`, the one format the engine reads and writes — and shipped taxonomies |
| [`examples/`](examples/) | The reference playbook (a Mutual NDA) at [`examples/nda/`](examples/nda/), validator fixtures, and the quickstart corpus |

## Status

**Engine 1.0.1; OPF 1.0 (stable).** The engine reads and writes exactly one
format, `opf_version` 0.5 — the verdict-free per-deal precedent record,
with what each clause opened with (`opened_with`, `opening_text`; issue #233),
and a `digest_version` 4 digest (held and conceded counts, variant
provenance, openings not signed as proposed, uncovered clause types; issue
#234). The 0.1–0.4 formats, their
schemas and the digest 2 builder were retired (issues #238, #233): there is no
installed consumer to stay compatible with, so a 0.1–0.4 document is
rejected as an unsupported version; git history and
[contract-opf/opf](https://github.com/contract-opf/opf) keep them. A shape
or semantic change ships under a new `opf_version` rather than an in-place
edit (see the spec's §11).
The engine's full pipeline is exercised end-to-end in CI — currently ~2,600
tests, all offline. Real-world derivation runs on a private educational-
affiliation corpus; a synthetic public showcase corpus ships at
[`examples/nda/`](examples/nda/).

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
