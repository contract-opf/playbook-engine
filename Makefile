.PHONY: install lint fmt typecheck test all docker-build docker-run

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

all: lint fmt-check typecheck test

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
docker-run:
	docker run --rm -it \
		-v "$(CORPUS):/work/corpus:ro" \
		-v "$(OUT):/work/out" \
		-e ANTHROPIC_API_KEY \
		$(DOCKER_IMAGE) $(ARGS)
