# NDA example

A wholly synthetic second agreement type (Mutual Non-Disclosure Agreement) --
fictional parties (AlphaCorp Holdings, Beta Industries, Delta Ventures,
Epsilon Systems, Gamma Holdings, Theta Logistics, Zeta Diagnostics),
fictional clause text, no real corpus and no confidential source material
(config.yaml's header declares this, and the repo's legal owner confirmed it
before these files were tracked -- see issue #111).

It exists to prove the engine is agreement-type-general on more than the one
type (Educational Affiliation) that has ever actually run:

```
config.yaml           -- the real config: template mode, agent segmentation
config.smoke.yaml      -- deterministic-segmentation override, see below
standard-form.rtf      -- AlphaCorp's canonical NDA template
corpus/                -- six fictional negotiations (four on our paper, two
                           on counterparty paper)
```

## Running it

With `config.yaml` (agent segmentation, matches `docs/PLAN-FIRST.md`'s
judged path) via the packaged skill or `playbook segment` /
`playbook segment-apply` / `playbook mine` / `playbook judge` /
`playbook project` -- see the main [README](../../README.md) and
[docs/QUICK-COMPILE.md](../../docs/QUICK-COMPILE.md).

For a quick, fully headless, no-LLM run (stub judges, deterministic
segmentation, structurally valid but semantically blank -- see the banner at
the top of `docs/QUICK-COMPILE.md`), use `config.smoke.yaml` instead:

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
no `ANTHROPIC_API_KEY` read).

This is exactly what `tests/test_nda_smoke.py` (`@pytest.mark.smoke`, also
runnable via `make smoke-nda`) exercises: `lint-corpus` -> `mine` ->
`project` -> `validate`, hermetic, asserting a non-zero
`template standards: N clause(s) classified` count and an evidence-only
playbook (empty `posture`/`floor` -- the smoke run never fabricates
negotiation intent or hard lines).
