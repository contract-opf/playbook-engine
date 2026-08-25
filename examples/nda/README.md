# NDA example

A wholly synthetic second agreement type (Mutual Non-Disclosure Agreement) --
fictional parties (AlphaCorp Holdings, Beta Industries, Delta Ventures,
Epsilon Systems, Gamma Holdings, Theta Logistics, Zeta Diagnostics),
fictional clause text, no real corpus and no confidential source material
(config.yaml's header declares this, and the repo's legal owner confirmed it
before these files were tracked -- see issue #111).

It exists to prove the engine is agreement-type-general on more than the one
type (Educational Affiliation) that has ever actually run -- and, since issue
#9, ships as a **fully worked, genuinely judged playbook** (populated
Posture and Floor, not just a compiling corpus):

```
config.yaml              -- the real config: template mode, agent segmentation
config.smoke.yaml        -- deterministic-segmentation override, see below
standard-form.rtf        -- AlphaCorp's canonical NDA template
corpus/                  -- six fictional negotiations (four on our paper, two
                             on counterparty paper)
canned-verdicts.jsonl    -- pre-computed judge verdicts (classification,
                             deviation, provenance, scope) for every item the
                             deterministic pipeline can't resolve on its own
posture-answers.json     -- the six-question GC-interview answers used to
                             author posture.system_prompt
playbook.opf.json        -- the derived, worked playbook: populated
                             evidence + posture + floor. THE reference
                             artifact -- see "The worked playbook" below.
```

## The worked playbook

`playbook.opf.json` is committed with genuine judged semantics: 26 clauses
across all six deals, a real `proposed_then_reversed` round-trip (the
`exclusions_from_confidential` and `survival_period` clauses in the
four-version `beta-industries` deal were narrowed/extended mid-negotiation
and then reversed back to our standard by signature -- the reversal
detector needs >=3 versions on a deal to observe this, which is why
`beta-industries` carries four), a populated `posture.system_prompt` from a
six-question GC interview, and three signed `floor.invariants`: two
auto-promoted from the interview's "sacred clauses" answer (exclusions from
Confidential Information; survival of confidentiality obligations) and one
hand-authored conditional hard line via `playbook floor sign` (limitation of
liability, if present, must not reach a confidentiality breach --
responding to the $50,000 liability cap that appears in three of the six
deals, introduced by the counterparty in two of them).

**It is reproducible from the committed inputs above, with no
`ANTHROPIC_API_KEY`:**

```sh
OUT=/tmp/nda-derive
mkdir -p "$OUT"
playbook lint-corpus examples/nda/corpus --config examples/nda/config.smoke.yaml
playbook judge-apply "$OUT" --verdicts examples/nda/canned-verdicts.jsonl
playbook mine        examples/nda/corpus --config examples/nda/config.smoke.yaml --out "$OUT"
playbook project      "$OUT" --config examples/nda/config.smoke.yaml
playbook posture interview "$OUT" --answers-file examples/nda/posture-answers.json
playbook floor sign   "$OUT" \
  --statement "Limitation of liability, if present, must not apply to a breach of the confidentiality obligations in this Agreement." \
  --id limitation-of-liability-confidentiality-carveout \
  --rationale "A \$50,000 liability cap appeared in 3 of 6 deals (epsilon-systems, theta-logistics, zeta-diagnostics) -- introduced by the counterparty in theta-logistics and zeta-diagnostics (both counterparty-paper) and carried into our own epsilon-systems draft; our taxonomy flags limitation-of-liability as normally absent from a mutual NDA because capping breach-of-confidence damages guts the agreement's only real remedy." \
  --clause limitation_of_liability \
  --signed-by "Legal Owner" \
  --config examples/nda/config.smoke.yaml
playbook validate     "$OUT/playbook.opf.json"
```

`tests/test_nda_derive_reproducible.py` replays this sequence in CI -- plus
one step this block omits, an intermediate `playbook judge` call that
asserts the canned verdicts drained the judge queue to `(0 pending items)`
before `project` runs -- and diffs the result against the committed
`playbook.opf.json` (modulo the wall-clock `generated_at` timestamps) -- if
the two ever drift, that test fails until `playbook.opf.json` is
regenerated from a fresh run of these commands.
`tests/test_examples_validate.py` additionally checks the
committed file validates, carries no real branding, and actually
demonstrates populated Posture/Floor rather than the empty-section state a
freshly-mined, not-yet-interviewed playbook would have.

## Two other ways to run the pipeline

The judged reproduction above is the reference path. Two variants exist
alongside it:

With `config.yaml` (agent segmentation, matches `docs/PLAN-FIRST.md`'s
judged path) via the packaged skill or `playbook segment` /
`playbook segment-apply` / `playbook mine` / `playbook judge` /
`playbook project` -- see the main [README](../../README.md) and
[docs/QUICK-COMPILE.md](../../docs/QUICK-COMPILE.md). (`playbook.opf.json`
above was produced with deterministic segmentation, not this path -- see
"Why `config.smoke.yaml` exists" below for why that's the right choice for a
committed, CI-reproducible fixture.)

For a quick, fully headless, no-LLM run with no verdict store at all (stub
judges, deterministic segmentation, structurally valid but semantically
blank -- see the banner at the top of `docs/QUICK-COMPILE.md`), use
`config.smoke.yaml` with no `judge-apply` step:

```sh
playbook lint-corpus examples/nda/corpus --config examples/nda/config.smoke.yaml
playbook mine        examples/nda/corpus --config examples/nda/config.smoke.yaml --out /tmp/nda-out
playbook project      /tmp/nda-out        --config examples/nda/config.smoke.yaml
playbook validate     /tmp/nda-out/playbook.opf.json
```

### Why `config.smoke.yaml` exists

`config.yaml` sets `segmentation.agent: true` -- a store-backed agent loop
(the same one `playbook segment`/`segment-apply` drive) that queues
unsegmented documents for a human/LLM pass and cannot complete headlessly.
`config.smoke.yaml` is otherwise identical but omits `segmentation:`
entirely, so segmentation falls back to its deterministic default (no LLM,
no `ANTHROPIC_API_KEY` read) -- which is also exactly why it's the config
used to derive the committed `playbook.opf.json` above: a CI-reproducible
fixture can't depend on a headless-incompatible agent loop, and template
mode's deterministic classifier already resolves the great majority of
clauses on its own (`canned-verdicts.jsonl` covers only what's left:
classification/deviation/provenance/scope items the deterministic pass
can't call on its own).

The bare stub-judge run above is what `tests/test_nda_smoke.py`
(`@pytest.mark.smoke`, also runnable via `make smoke-nda`) exercises:
`lint-corpus` -> `mine` -> `project` -> `validate`, hermetic, asserting a
non-zero `template standards: N clause(s) classified` count and an
evidence-only playbook (empty `posture`/`floor` -- the smoke run never
fabricates negotiation intent or hard lines, and never loads
`canned-verdicts.jsonl`). `tests/test_nda_derive_reproducible.py` is the
companion test for the genuinely-judged path: it loads
`canned-verdicts.jsonl`, runs the Posture interview and `floor sign`, and
diffs the result against the committed `playbook.opf.json`.
