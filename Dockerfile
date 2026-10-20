# Reproducible runtime for the playbook-engine pipeline.
#
# (No `# syntax=` directive: this Dockerfile uses only classic instructions,
# so the engine's built-in BuildKit frontend handles it — avoiding a network
# pull of the external dockerfile frontend that can time out on restricted
# networks.)
#
# Pinned to Python 3.13: torch/docling (and their OCR chain) have no
# Python 3.14 wheels yet, so the engine's *own* pyproject deliberately
# stays free of docling/torch (keeps `pip install playbook-engine` light
# and importable on 3.14 hosts) — this image is where docling actually
# runs, invoked as a CLI the same way the engine already shells out to
# `pandoc` for RTF extraction.
FROM python:3.13-slim

# System deps for document extraction + OCR. `libgl1`/`libglib2.0-0` are
# required by opencv (pulled in transitively by docling); the slim base
# omits them and docling crashes at runtime without them. Installed and
# cache-cleaned in one layer to keep the image lean.
RUN apt-get update && apt-get install -y --no-install-recommends \
    pandoc \
    tesseract-ocr \
    ocrmypdf \
    poppler-utils \
    ghostscript \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# The OCR engine docling is told to use (issue #252) must serve English. Fail
# the build here rather than let a scan fail, or fall back, at run time.
RUN tesseract --list-langs | grep -qx eng

# docling (large, changes rarely) is installed BEFORE the app is copied so
# editing engine code below does not invalidate this layer. Use the
# CPU-only torch build — the pipeline runs inference on CPU, and the default
# CUDA wheels would add ~6 GB of GPU libraries we never use.
#
# The whole docling/OCR stack is pinned in docker/constraints.txt (issue #244):
# an unpinned docling once changed its OCR engine selection under us and
# extraction silently fell back to the legacy reader. The same constraints
# apply to the docling install, so docling cannot move torch to another
# version. See that file for how to bump the pins deliberately.
COPY docker/constraints.txt /tmp/constraints.txt
RUN pip install --no-cache-dir -c /tmp/constraints.txt torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -c /tmp/constraints.txt docling

# All of docling's models (layout, table-structure, and the RapidOCR default
# OCR backend) are resolved through one setting: `settings.artifacts_path`,
# read from the `DOCLING_ARTIFACTS_PATH` env var (pydantic-settings,
# `env_prefix="DOCLING_"` - see `docling.datamodel.settings.AppSettings`).
# Every model downloader in `docling.utils.model_downloader.download_models`
# takes an explicit `local_dir` and calls `huggingface_hub.snapshot_download(...,
# local_dir=...)` directly, so pointing `HF_HOME` here would do nothing:
# `local_dir` downloads bypass the HF cache entirely. Setting
# `DOCLING_ARTIFACTS_PATH` is therefore the one override needed for both the
# HF-hosted models and RapidOCR, which docling resolves relative to the same
# `artifacts_path`.
ENV DOCLING_ARTIFACTS_PATH=/opt/docling-models

# Pre-fetch every model docling needs (as root, at build time) into that
# shared directory via docling's own `docling-tools models download` helper
# (installed by the `standard` extra pulled in by `docling` above). With no
# model list this downloads docling's default set — layout, tableformer,
# code_formula, picture_classifier, and RapidOCR — i.e. everything the
# default conversion pipeline (layout + table-structure + OCR) can reach, so
# no model is ever fetched at runtime by the non-root `playbook` user. Then
# open read/execute permissions (never write) for that user before the user
# switch below.
# HF_HUB_DISABLE_XET forces the classic HTTP download path: Hugging Face's
# xet CAS intermittently returns 401 to anonymous clients on CI runners
# (observed on the first public GitHub Actions build, 2026-07-14), and the
# classic path is not affected. Retry once more for ordinary transience.
RUN HF_HUB_DISABLE_XET=1 docling-tools models download --output-dir /opt/docling-models \
    || (sleep 15 && HF_HUB_DISABLE_XET=1 docling-tools models download --output-dir /opt/docling-models) \
    && chmod -R a+rX /opt/docling-models

# OCR engine and language (issue #252). The engine passes
# `--ocr-engine tesseract --ocr-lang iso:en` on every conversion
# (`playbook_engine/extraction.py::_run_docling`); this image carries the
# pinned docling 2.140.0 and the apt `tesseract-ocr` English data checked above.
# There is no env var or build-time lever for the OCR language: it has to be on
# the conversion call. History, so nobody re-learns it:
#   - docling's automatic engine is RapidOCR, whose default recognizer is
#     Chinese; it drops inter-word spaces on Latin-script text (docling
#     upstream issues #2887, #1635, #2927).
#   - docling 2.113.0 (the previous pin) ignored `--ocr-lang eng` and ran that
#     Chinese recognizer anyway: 130-150 s per page (a 10-page scan timed out
#     at 600 s), about 40% of words lost, 18-28 tokens joined per document.
#   - docling 2.140.0 honours `iso:en` for RapidOCR too, but needs 345-551 s for
#     the same 10-page scan; tesseract needs 107-124 s and recovers 0-10% more
#     words, so the engine names tesseract explicitly.

# Install the engine (changes often; kept in its own cheap layer so a code
# edit only re-runs this step, not the torch/docling download above).
COPY . /app
RUN pip install --no-cache-dir /app

# Stamp the image with the engine version and source commit it was built from,
# so a STALE image cannot silently produce a fix-free derivation.
#
# Not hypothetical: a local image three days behind the repo held engine 0.2.0
# while the source tree was at 1.0.1, and the documented Docker workflow ran
# happily inside it — producing a derivation missing exactly the fixes the run
# was performed to apply, and looking like a success. Nothing anywhere compared
# the two.
#
# ENGINE_VERSION is supplied by the build (`make docker-build` reads it from
# playbook_engine/__init__.py) rather than computed here, because LABEL cannot
# run code. A build arg can lie, so the RUN below refuses to build unless the
# arg matches the version pip actually installed — which is what makes the
# labels trustworthy enough for `make docker-run` to gate on via
# `docker image inspect`, without starting a container.
ARG ENGINE_VERSION=unknown
ARG ENGINE_GIT_SHA=unknown
RUN installed="$(python -c 'import playbook_engine; print(playbook_engine.__version__)')" \
    && if [ "$ENGINE_VERSION" != "unknown" ] && [ "$installed" != "$ENGINE_VERSION" ]; then \
         echo "build refused: --build-arg ENGINE_VERSION=$ENGINE_VERSION does not match the engine actually installed ($installed)" >&2; \
         exit 1; \
       fi \
    && printf '{"engine_version": "%s", "git_sha": "%s"}\n' "$installed" "$ENGINE_GIT_SHA" \
       > /etc/playbook-engine-image.json \
    && chmod a+r /etc/playbook-engine-image.json

LABEL org.opencontainers.image.title="playbook-engine" \
      org.opencontainers.image.version="$ENGINE_VERSION" \
      org.opencontainers.image.revision="$ENGINE_GIT_SHA" \
      org.opencontainers.image.source="https://github.com/contract-opf/playbook-engine"

# Run as non-root; /work is where corpus/out volumes get mounted, so the
# runtime user needs to own it.
RUN useradd --create-home --uid 1000 playbook && mkdir -p /work && chown playbook:playbook /work
WORKDIR /work
USER playbook

ENTRYPOINT ["playbook"]
