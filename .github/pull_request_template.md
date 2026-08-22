## What & why

<!-- One paragraph: the slice this PR delivers, and the issue it closes. -->

Closes #

## Checklist

- [ ] `make all` is green (run with the venv on PATH: `PATH="$PWD/.venv/bin:$PATH" make all`)
- [ ] Behavior changes come with tests (red-first where fixing a bug)
- [ ] Fixtures in `examples/` exercise the change; no real agreements or party names anywhere
- [ ] `docs/` updated if behavior or format changed

### Spec/schema changes only

- [ ] Rationale stated; `docs/OPF-SPEC.md` updated incl. Appendix B changelog
- [ ] `examples/our-paper-baseline.v0.2.playbook.json` still validates
- [ ] Versioning impact called out (1.x is additive-only; removing/retyping a
      field or changing a normative MUST requires 2.0 — see spec §11); a new
      or changed MUST gets its own `CHANGELOG.md` entry under
      `### Normative rule changes`
