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

# docling (large, changes rarely) is installed BEFORE the app is copied so
# editing engine code below does not invalidate this layer. Pin the
# CPU-only torch build — the pipeline runs inference on CPU, and the default
# CUDA wheels would add ~6 GB of GPU libraries we never use.
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir docling

# All of docling's models (layout, table-structure, and the RapidOCR default
# OCR backend) are resolved through one setting: `settings.artifacts_path`,
# read from the `DOCLING_ARTIFACTS_PATH` env var (pydantic-settings,
# `env_prefix="DOCLING_"` — see `docling.datamodel.settings.AppSettings`).
# Every model downloader in `docling.utils.model_downloader.download_models`
# (layout, tableformer, code_formula, picture_classifier, RapidOCR) takes an
# explicit `local_dir` and calls `huggingface_hub.snapshot_download(...,
# local_dir=...)` directly — so pointing `HF_HOME` here would do nothing:
# `local_dir` downloads bypass the HF cache entirely. Setting
# `DOCLING_ARTIFACTS_PATH` is therefore the one override needed for both the
# HF-hosted models *and* RapidOCR, which docling resolves relative to the
# same `artifacts_path` (`RapidOcrModel` in
# `docling.models.stages.ocr.rapid_ocr_model`) — confirmed by reading the
# installed docling 2.110.0 source, not assumed.
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

# OCR quality investigation (word-joining on table-heavy scanned pages):
# read RapidOCR's resolution logic directly (`_resolve_rapidocr_language` in
# `docling.models.stages.ocr.rapid_ocr_model`). Root cause confirmed:
# `RapidOcrOptions.lang` defaults to `["chinese"]`, and running the Chinese
# recognition model on Latin-script text is exactly what drops inter-word
# spaces (docling upstream issues #2887, #1635, #2927 — docling's own code
# comments cite these). docling 2.110.0 already ships an `english`/`latin`
# RapidOCR model set that fixes this, but selecting it requires passing
# `--ocr-lang eng` (or setting `RapidOcrOptions.lang`) on the *conversion*
# call in `playbook_engine/extraction.py::_run_docling` — there is no
# env var / build-time-only lever for it (`AppSettings` in
# `docling.datamodel.settings` only exposes `artifacts_path`, not OCR
# language). Per this ticket's scope, `extraction.py` is not touched here;
# the fix is a one-line `--ocr-lang eng` (or `deu`/etc., per corpus
# language) addition to the `docling convert` invocation, tracked against
# the CLI-wiring ticket (#79 Notes / out-of-scope).

# Install the engine (changes often; kept in its own cheap layer so a code
# edit only re-runs this step, not the torch/docling download above).
COPY . /app
RUN pip install --no-cache-dir /app

# Run as non-root; /work is where corpus/out volumes get mounted, so the
# runtime user needs to own it.
RUN useradd --create-home --uid 1000 playbook && mkdir -p /work && chown playbook:playbook /work
WORKDIR /work
USER playbook

ENTRYPOINT ["playbook"]
