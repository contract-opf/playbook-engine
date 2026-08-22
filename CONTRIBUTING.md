# Contributing

Thanks for your interest in the Open Playbook Format and its reference
compiler. By participating you agree to our
[Code of Conduct](CODE_OF_CONDUCT.md). Two things live in this repository, and they have different bars:

- **The OPF standard** (`docs/OPF-SPEC.md`, `spec/playbook.schema-0.3.json`)
  — the current format. `spec/playbook.schema-0.2.json` documents remain
  valid but are not current; `docs/OPF-SPEC-v0.1.md` /
  `spec/playbook.schema.json` are superseded.
- **The engine** (`playbook_engine/`) — the compiler that produces OPF
  playbooks from a corpus of negotiated agreements.

## Dev setup

Prerequisites: Python 3.11+ and `pandoc`. The RTF extraction path shells out
to pandoc, and the test fixtures are `.rtf`, so the suite fails without it.

```sh
brew install pandoc   # macOS — Debian/Ubuntu: apt-get install -y pandoc
python3 -m venv .venv
source .venv/bin/activate
make install          # pip install -e ".[dev]"
make hooks            # install the tracked lint/format pre-push gate (issue #116)
```

`.venv/` is disposable and machine-local (it hard-codes absolute paths, so it
never survives a move between machines). Delete it and re-run the three
commands above to rebuild from scratch; `make all` confirms the result.

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
- the schema update (`spec/playbook.schema-0.3.json`),
- an updated example (`examples/our-paper-baseline.v0.2.playbook.json`
  must keep validating — CI enforces this),
- a spec-text update (`docs/OPF-SPEC.md`) including Appendix B changelog,
- a note on versioning impact: as of OPF 1.0, 1.x changes are additive-only
  (new OPTIONAL fields, new `x_*` extensions); removing/retyping a field or
  changing a normative MUST requires 2.0 (see §11 Versioning & migration).
  A new or changed MUST — even one that touches no schema field — needs its
  own `CHANGELOG.md` entry under a `### Normative rule changes` heading.

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

## Cutting a release

Maintainer-only. `docker-publish.yml` and `pypi-publish.yml` do the actual
publishing; this is the sequence that fires them (issue #114).

1. **Bump the version.** `pyproject.toml` `[project].version`, following
   the §11 stability policy above (1.x is additive-only; see issue #113
   for the 1.0.0 precedent).
2. **Update `CHANGELOG.md`.** A new `## [X.Y.Z] - YYYY-MM-DD` entry. Any
   new or changed normative MUST — schema-touching or not — gets its own
   `### Normative rule changes` entry, per the changelog's own header
   policy.
3. **Tag and push.** `git tag vX.Y.Z && git push origin vX.Y.Z`. The tag
   push alone fires `.github/workflows/docker-publish.yml`, which builds
   and pushes `ghcr.io/contract-opf/playbook-engine:vX.Y.Z` and `:latest`.
4. **Publish a GitHub Release from that tag** (UI: Releases → Draft a new
   release → pick the tag; or
   `gh release create vX.Y.Z --generate-notes`). Publishing the release —
   not the tag push — is what fires `.github/workflows/pypi-publish.yml`:
   it builds the sdist + wheel, verifies both actually contain `spec/`
   (the wheel-only `force-include` setting doesn't cover the sdist —
   `tests/test_packaging.py::test_sdist_contains_spec_dir` is the
   in-repo regression test for that gap), and publishes to PyPI over
   Trusted Publishing (OIDC — no `PYPI_API_TOKEN` secret is configured
   or needed).

### One-time PyPI setup (before the first release)

`pypi-publish.yml` has no long-lived PyPI credential to configure; it
authenticates via OIDC Trusted Publishing instead. Before the *first* run,
a PyPI project owner must register the publisher — this is a PyPI-account
action, not something CI can do, and is the one release step that can't be
automated (flagged at landing, issue #114):

1. On PyPI, register (or claim) the `playbook-engine` project name —
   registering a Trusted Publisher can create the pending project even
   before any release is uploaded, so this may happen as part of step 2.
2. Add a Trusted Publisher: project → Publishing → Add a new publisher →
   GitHub → owner `contract-opf`, repository `playbook-engine`, workflow
   `pypi-publish.yml`, environment `pypi`.

Until this is done, steps 1–4 above still succeed (version bump, changelog,
tag, Docker image, GitHub Release) — only the `publish` job in
`pypi-publish.yml` fails, since PyPI has no publisher to trust yet.

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
- **CI gate on `main`** (issue #116 — `main` went red for 12 consecutive
  commits, one with zero CI coverage at all, before anyone noticed): a
  tracked, generic lint/format gate (`scripts/pre-push-lint.sh` — just
  `ruff check` / `ruff format --check`, nothing confidential) is
  installable on **any** clone — including linked worktrees and clones with
  `core.hooksPath` set — with `make hooks`, which resolves the real hooks
  directory via `git rev-parse --git-path hooks` and copies the gate there.
  This is local-only and bypassable (`--no-verify`,
  a clone where `make hooks` was never run, `git push --force` from CI
  credentials). The maintainer's own machine additionally layers a
  confidential/secrets scan on top via `ignore/git-hooks/install.sh`; that
  deny-list stays gitignored by design, so this extra layer is
  maintainer-local only and not reproducible from a fresh clone. By
  convention that hook is expected to delegate its own lint/format tier to
  `scripts/pre-push-lint.sh` so the two compose safely in either install
  order — but that convention lives outside this repository's tracked
  history, `make hooks` cannot verify it in advance, and it must not be
  assumed to hold on a machine you haven't checked; `make hooks` refuses
  loudly rather than reporting false success if it ever finds a
  secrets-scanning hook that lacks a genuine delegation line — a comment
  that merely mentions `pre-push-lint.sh` (e.g. a stray "TODO: delegate to
  ...") does not count as delegating (see the composition-rule comment
  above the `hooks` target in the Makefile). The durable
  server-side gate is still outstanding and requires repo-admin action:
  **enable branch protection on `main`** at
  Settings → Branches → Branch protection rules, requiring status checks to
  pass before merging, with contexts `lint-and-test (3.11)` and
  `lint-and-test (3.12)` (the two matrix jobs `.github/workflows/ci.yml`
  reports as of this writing — re-check the exact context names in a recent
  run before configuring, `gh api repos/contract-opf/playbook-engine/commits/<sha>/check-runs
  --jq '.check_runs[].name'`). Equivalent via API:
  `gh api -X PUT repos/contract-opf/playbook-engine/branches/main/protection
  -F 'required_status_checks[strict]=true'
  -f 'required_status_checks[contexts][]=lint-and-test (3.11)'
  -f 'required_status_checks[contexts][]=lint-and-test (3.12)'
  -F enforce_admins=true -F required_pull_request_reviews=null
  -F restrictions=null`. Note this repo has landed 90 commits on `main`
  against only 3 PRs in its entire history (issue #116: "~12 of the last 13
  commits landed direct-to-main") — requiring status checks on `main` itself
  (not just on PRs) still lets a passing check on an old commit block nothing
  useful if commits bypass review entirely; consider pairing with
  `required_pull_request_reviews` (forcing every change through a PR) as a
  separate, bigger process decision, not bundled into this one.

## Issue-number provenance

This repository went public on 2026-07-25 with a fresh single-commit history.
Issue references in code comments, commit messages, and migrated issue bodies
that predate the cutover (e.g. "issue #132", `internal#N`) refer to the
original private tracker, not this repository's issue numbers. Migrated issues
carry a provenance line naming their original number.
