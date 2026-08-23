.PHONY: install lint fmt typecheck test smoke-nda smoke-canary smoke all hooks docker-build docker-run

VENV := .venv
PY   := $(VENV)/bin/python
RUN  := $(VENV)/bin

DOCKER_IMAGE := playbook-engine
CORPUS       := $(CURDIR)/corpus
OUT          := $(CURDIR)/out

install:
	$(PY) -m pip install -e ".[dev]" -q

lint:
	$(RUN)/ruff check .

fmt:
	$(RUN)/ruff format .

fmt-check:
	$(RUN)/ruff format --check .

typecheck:
	$(RUN)/mypy playbook_engine

test:
	$(RUN)/pytest

# Hermetic, no-LLM second-agreement-type smoke run (issue #111): lint-corpus
# -> mine -> project -> validate over the synthetic examples/nda/ corpus.
# No network, no ANTHROPIC_API_KEY read. See examples/nda/README.md and
# tests/test_nda_smoke.py.
smoke-nda:
	$(RUN)/pytest tests/test_nda_smoke.py -q -m smoke

# Canary corpus gate: the 2026-08-22 incident (docling silently missing ->
# extraction fell back to legacy -> extraction cache key changed -> canonical
# text changed -> segmentation cache missed -> 43/44 documents quarantined as
# AgentSegmentationPending) turned red in seconds instead of not at all.
#
# SIBLING of smoke-nda, not a replacement. smoke-nda runs COLD on the
# deterministic ingest path, which never calls playbook_engine.extraction at
# all and therefore cannot exercise either the extractor environment or the
# extraction cache. This target covers exactly that gap: extractor identity,
# a warm-cache replay with zero re-extraction and zero quarantine, and
# committed derivation counts. Hermetic and keyless (DOCX only -- no pandoc,
# no docling, no ANTHROPIC_API_KEY). See examples/canary/README.md and
# tests/test_canary_corpus.py.
smoke-canary:
	$(RUN)/pytest tests/test_canary_corpus.py -q -m smoke

# Both hermetic, keyless smoke gates.
smoke: smoke-nda smoke-canary

all: lint fmt-check typecheck test

# Install the tracked lint/format pre-push gate (issue #116) into the
# REAL hooks directory — resolved via `git rev-parse --git-path hooks`, so
# this works from a fresh clone, a linked worktree (where .git is a file,
# not a directory), and a clone with core.hooksPath set (where .git/hooks
# is never consulted by git at all). No `ignore/` required.
# Composition rule (order-independent only when the maintainer-local hook
# genuinely delegates — see below): this target REFUSES to touch an
# already-installed pre-push hook that already delegates to
# scripts/pre-push-lint.sh — detected by finding a non-comment line that
# mentions that filename OUTSIDE of an echo/printf call, i.e. an actual
# invocation, not merely a string that gets printed. A comment mentioning
# the filename (e.g. a stray "TODO: delegate to...") or an echo/printf line
# that only prints a message containing the filename (e.g. a "not found —
# skipping" message) does NOT count: both are text *about* the script, not
# a call to it, so they fall through to the back-up-and-install path below
# — leaving it in place as a successful no-op, since it already
# has full lint/format coverage. A hook that merely scans for
# secrets/confidential terms (SECRET_PATTERNS / CONFIDENTIAL_PATTERNS) but
# does NOT delegate is a category error, not equivalence — a secrets
# scanner has zero lint/format coverage of its own — so this target FAILS
# LOUDLY (non-zero exit) instead of reporting success, with instructions to
# add the missing delegation line, rather than silently leaving the repo
# without a lint/format gate while claiming one is active. It only backs
# up and replaces a plain or unrelated existing hook.
# ignore/git-hooks/install.sh, in the other direction, always installs its
# own (richer) hook regardless of what's currently in place, and resolves
# the same real hooks directory. BY CONVENTION that hook is expected to
# delegate its lint/format tier to this same script — but that convention
# lives in a gitignored, maintainer-local file this target cannot inspect
# ahead of time, so it is not guaranteed and must not be assumed here; if
# it's ever missing, a subsequent `make hooks` refuses loudly (per above)
# instead of silently accepting the gap.
hooks:
	@hooks_dir="$$(git rev-parse --git-path hooks)"; \
	src="$$(git rev-parse --show-toplevel)/scripts/pre-push-lint.sh"; \
	dest="$$hooks_dir/pre-push"; \
	mkdir -p "$$hooks_dir"; \
	if [ -f "$$dest" ] && mentions="$$(grep -E '^[[:space:]]*[^#]*pre-push-lint\.sh' "$$dest" 2>/dev/null)" && [ -n "$$mentions" ] && printf '%s\n' "$$mentions" | grep -qvE '^[[:space:]]*(echo|printf)([[:space:]]|$$)'; then \
		echo "$$dest already delegates to scripts/pre-push-lint.sh." >&2; \
		echo "that hook is already at least as strong as this target installs, so it is being left in place (not an error)." >&2; \
		echo "to (re)install/refresh it instead, run: ignore/git-hooks/install.sh" >&2; \
		exit 0; \
	fi; \
	if [ -f "$$dest" ] && grep -qE 'CONFIDENTIAL_PATTERNS|SECRET_PATTERNS' "$$dest" 2>/dev/null; then \
		echo "$$dest scans for secrets/confidential terms but does NOT delegate to scripts/pre-push-lint.sh." >&2; \
		echo "a secrets scan has zero lint/format coverage of its own — treating it as equivalent would silently leave this clone WITHOUT the lint/format gate while reporting success, so refusing instead." >&2; \
		echo "fix: add a line invoking scripts/pre-push-lint.sh in $$dest (see the header comment in that script), then re-run: make hooks" >&2; \
		exit 1; \
	fi; \
	if [ -f "$$dest" ] && ! cmp -s "$$src" "$$dest"; then \
		cp "$$dest" "$$dest.bak.$$(date +%s 2>/dev/null || echo prev)"; \
		echo "existing pre-push backed up alongside $$dest"; \
	fi; \
	cp "$$src" "$$dest"; \
	chmod +x "$$dest"; \
	echo "installed lint/format pre-push gate -> $$dest"; \
	if [ ! -x "$(RUN)/ruff" ] && ! command -v ruff >/dev/null 2>&1; then \
		echo "WARNING: ruff not found ($(RUN)/ruff or PATH) — this looks like a developer checkout (pyproject.toml is tracked here), so scripts/pre-push-lint.sh fails CLOSED: the gate just installed will BLOCK EVERY PUSH until you run: make install" >&2; \
	fi; \
	echo "(maintainers: if the fuller confidential/secrets hook isn't installed yet, run ignore/git-hooks/install.sh — by convention it should delegate its lint/format tier to this same script; if it doesn't yet, add that delegation so installing in either order leaves the same protection active.)"

# Build the reproducible Python 3.13 runtime (docling + OCR + pandoc).
docker-build:
	docker build -t $(DOCKER_IMAGE) .

# Run `playbook ...` inside the image. Mounts CORPUS read-only and OUT
# read-write; forwards ANTHROPIC_API_KEY from the host environment.
# Override CORPUS/OUT/ARGS as needed, e.g. (config file placed alongside the
# corpus so it's visible under the read-only /work/corpus mount; `mine` has
# no `-o` short flag and requires `--config`):
#   make docker-run CORPUS=/path/to/corpus OUT=/path/to/out \
#     ARGS="mine /work/corpus --config /work/corpus/playbook.config.yaml --out /work/out"
# -i only (no -t): agent/CI shells have no TTY and `-t` aborts with
# "the input device is not a TTY"; interactive use works fine without it.
docker-run:
	docker run --rm -i \
		-v "$(CORPUS):/work/corpus:ro" \
		-v "$(OUT):/work/out" \
		-e ANTHROPIC_API_KEY \
		$(DOCKER_IMAGE) $(ARGS)
