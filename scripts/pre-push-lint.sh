#!/usr/bin/env bash
#
# Tracked pre-push lint/format gate (issue #116: main went red for 12
# consecutive commits — one with zero CI coverage at all — before anyone
# noticed, because nothing checked formatting before push).
#
# Deliberately generic: no confidentiality/secrets logic lives here, so this
# file is safe to track and publish. Install it as this clone's pre-push
# hook with `make hooks` — that works from a fresh clone, no `ignore/`
# required. The maintainer's fuller local hook
# (`ignore/git-hooks/pre-push`, gitignored — it additionally scans for
# secrets and confidential terms before pushing to a public remote) is BY
# CONVENTION expected to delegate its lint/format tier to this same script,
# so there is one source of truth instead of two copies drifting apart —
# but that convention lives in a gitignored, maintainer-local file this
# script cannot inspect, so it is not guaranteed and must not be assumed to
# hold on a machine you haven't checked (see the composition-rule comment
# above the `hooks` target in the Makefile).
#
# Deliberately mirrors `make lint fmt-check` only, NOT the full `make all`:
# mypy/pytest are slow enough on this repo to invite `--no-verify` on every
# push, which defeats the point (see reviewer gate on #116). The full suite
# still gates in CI.
#
# NOTE: checks the CURRENT WORKING TREE, not the exact tree of the SHA being
# pushed. For this repo's actual workflow — commit, then push immediately —
# HEAD's working tree and the pushed SHA are the same files, so this is
# exact in the common case. A push of a SHA that differs from the currently
# checked out tree (e.g. `git push origin other-local-branch:main`) will not
# be format-checked here; CI remains the backstop for that case.
#
set -eo pipefail

repo_root="$(git rev-parse --show-toplevel)"

if [ -x "$repo_root/.venv/bin/ruff" ]; then
  ruff_bin="$repo_root/.venv/bin/ruff"
elif command -v ruff >/dev/null 2>&1; then
  ruff_bin="ruff"
elif [ -d "$repo_root/.venv" ] || [ -f "$repo_root/pyproject.toml" ]; then
  # This looks like a developer checkout (.venv or pyproject.toml present)
  # whose ruff went missing — fail CLOSED. Fail-OPEN here would silently
  # degrade every push to no gate the moment a .venv gets removed/broken,
  # which is exactly the "gate didn't fire and nobody noticed" failure
  # mode issue #116 exists to eliminate.
  echo "" >&2
  echo "pre-push BLOCKED: ruff not found (checked $repo_root/.venv/bin/ruff and PATH)," >&2
  echo "but this looks like a developer checkout (.venv or pyproject.toml present)." >&2
  echo "Fix with:  make install" >&2
  echo "Verified false positive:  git push --no-verify" >&2
  exit 1
else
  # Unreachable for any checkout of THIS repo: pyproject.toml is tracked at
  # the root, so the fail-CLOSED branch above always fires here instead.
  # Retained for hypothetical non-checkout consumers of this script — e.g.
  # if it's ever copied standalone into a repo/dir with no pyproject.toml
  # and no .venv — so it degrades to skip-with-a-warning rather than
  # blocking a push it has no basis to gate.
  echo "pre-push-lint: ruff not found (checked $repo_root/.venv/bin/ruff and PATH)," >&2
  echo "  and this doesn't look like a developer checkout — skipping lint/format gate; CI remains the backstop." >&2
  exit 0
fi

set +e
lint_out="$("$ruff_bin" check "$repo_root" 2>&1)"
lint_status=$?
fmt_out="$("$ruff_bin" format --check "$repo_root" 2>&1)"
fmt_status=$?
set -e

if [ "$lint_status" -ne 0 ] || [ "$fmt_status" -ne 0 ]; then
  echo "" >&2
  echo "pre-push BLOCKED: ruff lint/format check failed (would fail CI)." >&2
  [ "$lint_status" -ne 0 ] && { echo "  [lint]" >&2; printf '%s\n' "$lint_out" | sed 's/^/    /' >&2; }
  [ "$fmt_status" -ne 0 ] && { echo "  [fmt]" >&2; printf '%s\n' "$fmt_out" | sed 's/^/    /' >&2; }
  echo "" >&2
  echo "Fix with:  make fmt   (and re-run) make lint" >&2
  echo "Verified false positive:  git push --no-verify" >&2
  exit 1
fi

exit 0
