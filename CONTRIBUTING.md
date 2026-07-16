# Contributing

Thanks for your interest in the Open Playbook Format and its reference
compiler. By participating you agree to our
[Code of Conduct](CODE_OF_CONDUCT.md). Two things live in this repository, and they have different bars:

- **The OPF standard** (`docs/OPF-SPEC.md`, `spec/playbook.schema-0.2.json`)
  — the current v0.2 format. `docs/OPF-SPEC-v0.1.md` /
  `spec/playbook.schema.json` are superseded.
- **The engine** (`playbook_engine/`) — the compiler that produces OPF
  playbooks from a corpus of negotiated agreements.

## Dev setup

```sh
python3 -m venv .venv
source .venv/bin/activate
make install          # pip install -e ".[dev]"
```

## Run the checks

```sh
make all              # lint (ruff), format check, typecheck (mypy), tests
```

`make all` must be green before any PR. The test suite spawns the
`playbook` CLI, so run with the venv activated (or
`PATH="$PWD/.venv/bin:$PATH" make all`).

## Code style

ruff and mypy are the arbiters — if `make all` passes, style is settled; we
do not litigate formatting in review. Match the surrounding code's comment
density and naming. Tests accompany behavior changes: a fix without a
red-first regression test is incomplete.

## Spec changes vs code changes

Code changes need a green `make all` and a focused diff. **Spec/schema
changes carry a higher bar** — the format is an interface others build on:

- a rationale (what can't be expressed today, and why this shape),
- the schema update (`spec/playbook.schema-0.2.json`),
- an updated example (`examples/our-paper-baseline.v0.2.playbook.json`
  must keep validating — CI enforces this),
- a spec-text update (`docs/OPF-SPEC.md`) including Appendix B changelog,
- a note on versioning impact (pre-1.0, breaking changes are allowed but
  must be called out; see §11 Versioning & migration).

Vendor-specific needs belong in the reserved `x_*` extension namespace
(§10.1), not in new core fields.

## Ground rules

- **Never commit real agreements.** Corpora are private and gitignored.
  Use synthetic/redacted fixtures in `examples/`.
- **Deterministic before LLM.** If a stage can be done with parsing or
  diffing, it must be; LLM calls are reserved for semantic judgment.
- **Every clause assertion cites precedent.** No exceptions in produced
  playbooks.
- **Honor the provenance rule** (OPF §2.3): counterparty-paper
  observations never set an opening position.

## PR flow

1. Fork/branch, one coherent slice per PR.
2. Add or extend a fixture under `examples/` that exercises the change.
3. Run `make all`; fill in the PR template checklist.
4. Keep `docs/` in sync — undocumented behavior is a bug.
5. A maintainer reviews; spec changes may need a second maintainer.

## Maintainers

The conventions below apply to the maintainers' automated workflow, not to
external contributions.

- Issues intended for the autonomous build loop carry the `afk` label and
  are written self-contained: exhaustive Files-to-touch, named test
  commands, explicit `Blocked by`. Don't start an issue whose dependencies
  are open.
- The loop lands one ticket at a time (fresh-context coder → independent
  reviewer → land with SHA evidence). One loop per repo, never two writers
  in one tree.
- Validate any produced playbook against the schema for its declared
  `opf_version`.
