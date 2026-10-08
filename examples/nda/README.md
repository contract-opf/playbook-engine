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
                             provenance, scope) for every item the
                             deterministic pipeline can't resolve on its own
                             (deviation is never judged by default -- it is
                             the deterministic standard check)
posture-answers.json     -- the six-question GC-interview answers used to
                             author posture.system_prompt
playbook.opf.json        -- the derived, worked playbook: populated
                             evidence + posture + floor. THE reference
                             artifact -- see "The worked playbook" below.
precedent.jsonl          -- the sidecar `playbook project` writes beside
                             playbook.opf.json: one evidence.precedent record
                             per line, sorted by id; its sha256 is recorded
                             under the playbook's root `x_sidecars`, and
                             test_committed_nda_sidecar_belongs_to_its_playbook
                             keeps the two in sync
```

## The worked playbook

`playbook.opf.json` is committed as an **OPF 0.4** document (`opf_version`
"0.4", issue #223): the verdict-free per-deal precedent record with a
`digest_version` "3" digest. 26 clauses across all six deals, one
`evidence.precedent` record per (deal, clause): what each deal signed, whether
that is our standard language (`standard` — an exact match after
normalization, never a judged verdict), whether the clause moved, and the
asks refused before signing. It demonstrates a real refused ask from a
proposed-then-reversed round-trip (a `residuals` clause inserted in v2 of the
three-version `zeta-diagnostics` deal and struck again before signing -- the
reversal detector needs >=3 versions on a deal to observe this); our own
standard clauses struck before signing (standard of care and venue in
`beta-industries`) as precedents whose `opening_text` is our standard and
whose `standard` is false -- our concession, never a refused ask; and no
stance, risk or deviation verdict anywhere (the consumer model does the
judging; each clause's `n_signed_standard` of `n_deals` is the fact it reads).
It also carries a populated `posture.system_prompt` from a six-question GC
interview, and three signed `floor.invariants`: two auto-promoted from the
interview's "sacred clauses" answer (exclusions from Confidential
Information; survival of confidentiality obligations) and one hand-authored
conditional hard line via `playbook floor sign` (limitation of liability, if
present, must not reach a confidentiality breach -- responding to the
$50,000 liability cap that appears in three of the six deals, introduced by
the counterparty in two of them). OPF 0.4 is the only format the engine
emits (issue #238).
`corpus.stats.dropped_observations` counts removed text that yields no
precedent.
A clause edited in an early round and then carried unchanged into the
signed copy is one `modified` clause (issue #232). The global move phase
chains the identical later copies into one row first, and that row is then
extended backwards, one draft at a time, under the bucket path's bind rule.
Before #232 the earlier copies were left with nothing to bind to, and the
deal showed a same-deal removed/added pair. Six clauses in this corpus are
now one row each. Five were edited in the first round: governing law in
`beta-industries` (Delaware to New York); confidentiality of discussions
and breach notice in `gamma-holdings`; the `epsilon-systems` first draft's
compelled-disclosure clause, which lacked the notice proviso the later
drafts add; and the `zeta-diagnostics` first draft's unclassified
exclusions paragraph (Jaccard 0.79 to the second draft's copy). The sixth
is `beta-industries`' survival period: three years, then five, then three
again. The removed/added pair had turned the opening text into a fabricated
concession (`beta-industries` governing law, `gamma-holdings`), a
fabricated refused ask (`epsilon-systems`), or a dropped observation
(`removed_origin_undetermined` in `zeta-diagnostics`, `survives_in_terminal`
in `beta-industries`). The four precedents that changed now have `rounds` 1
and `opening_text` null, because the store records an opening text only
for a clause struck before signing.
Below the 0.70 Jaccard threshold the aligner binds two drafts only as a
localized edit: Jaccard of at least 0.5 and exactly one contiguous edit span.
`beta-industries`' second draft struck the independent-development exclusion
and the burden sentence from the exclusions clause (Jaccard 0.68, three
separate edit spans). That draft is therefore its own row, and its whole text
is the refused ask the signed copy reversed. The old aligner reduced it to
the one-letter fragment "c". In `theta-logistics` and `zeta-diagnostics` the
compelled-disclosure clause gained an appended notice proviso (Jaccard 0.48).
It splits into removed + added, and the removed text is counted as
`survives_in_terminal`.

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
classification/provenance/scope items the deterministic pass can't call on
its own -- deviation needs no verdicts at all: it is the deterministic
standard check described above).

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
