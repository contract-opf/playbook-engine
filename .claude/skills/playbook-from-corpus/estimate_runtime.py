#!/usr/bin/env python3
"""Pre-flight runtime estimate for a corpus, BEFORE the expensive extraction.

Counts agreements/versions, classifies each PDF as born-digital vs scanned
(scanned PDFs OCR ~5-10x slower under docling), and prints a wall-clock ETA
range plus rough corpus-size and judgment-load estimates. Uses only pdfplumber
(no docling/torch), so it runs on the host venv in seconds.

Usage: .venv/bin/python estimate_runtime.py <corpus_dir> [out_dir]

If an ``out_dir`` with a warm extraction cache exists (``<out>/extraction_cache.jsonl``,
written by a prior/parallel ``mine``/``judge``/``segment`` run over the same
files), already-extracted versions are detected and contribute ~0 to the ETA —
so a re-run over an already-OCR'd corpus reports minutes, not hours. Defaults
to ``<corpus>/../out`` (the CLI's default). The cache is keyed by file content
hash PLUS the extractor environment that wrote the entry (docling vs. legacy
— see issue #77). This script is documented to run on the HOST venv (no
docling installed), while ``make docker-run ... mine/judge`` extracts INSIDE
the container, which DOES have docling (see the Dockerfile) — so it never
assumes its own PATH matches the environment the cache was written under.
It probes BOTH the ``docling`` and ``legacy`` key variants for each file and
reports which environment actually produced each hit.

Only a hit under the TARGET environment — the one the upcoming run will
actually extract under, defaulting to ``docling`` since the documented
pipeline always extracts inside the container (issue #77 fix-round-2) —
counts as 0 wall-clock. A hit under the other environment only is reported
separately (it will genuinely miss and re-extract) rather than silently
folded into the "already extracted" figure. Operators who really run
extraction on the host (no container, no docling) can opt out explicitly
with ``PLAYBOOK_ESTIMATE_TARGET_ENV=legacy``.

Time constants are calibrated from a representative affiliation-agreement
corpus on a CPU laptop (docling cold-loads its model per document; scanned
signed copies can hit the 600s per-file timeout). They are ESTIMATES — report
a range, not a promise.
"""

from __future__ import annotations

import contextlib
import glob
import hashlib
import json
import os
import sys

# Must match ``extraction._EXTRACTION_CACHE_FORMAT_VERSION`` and the
# key recipe in ``agent_judge._payload_key`` / ``extraction._extraction_cache_payload``
# (content hash + extractor environment as of issue #77). Replicated here
# (stdlib only) so this pre-flight stays dependency-light — no engine import,
# no docling/torch. Degrades safely: if the recipe ever drifts, cache hits
# simply go undetected and the ETA is over-estimated (conservative), never
# wrong-low.
#
# Bumped to "2" alongside extraction._EXTRACTION_CACHE_FORMAT_VERSION (issue
# #81: extract_blocks's returned/cached third element gained a structured
# reason — see extraction.ExtractorLabel). Without this bump, this script's
# own cache-hit probe would keep matching pre-#81 keys that the real engine
# no longer does, over-reporting "already extracted" and under-estimating
# the ETA for a corpus whose warm cache the real run will actually miss.
#
# Bumped to "3" alongside extraction._EXTRACTION_CACHE_FORMAT_VERSION again
# (issue #84: a redline DOCX docling failure is now retried on a normalized
# copy before falling back to legacy — see docx_normalizer.py — so the
# correct output for the SAME file/environment changed). Same rationale as
# the #81 bump above: without moving in lockstep, this probe would keep
# matching pre-#84 keys the real engine no longer does.
_EXTRACTION_CACHE_FORMAT_VERSION = "3"

# Per-version wall-clock (seconds), docling on CPU. Born-digital = model
# cold-load + convert; scanned = the same plus RapidOCR over page images,
# which frequently approaches the 600s timeout.
_T_BORN_DIGITAL = 60  # ~1 min
_T_SCANNED = 330  # ~5.5 min (some finish faster, some hit 600s)
_T_DOCX = 45  # docx: docling or fast legacy fallback
# very rough: extracted tokens per MB of raw source, from observed runs.
_TOKENS_PER_MB = 13_800


def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# The only two values ``extraction.detect_extractor`` can ever return.
# Deliberately NOT auto-detected from THIS script's own ``shutil.which`` —
# this script runs on the host (no docling), while `make docker-run ...
# mine/judge` extracts inside the container (docling installed), so the
# two environments routinely differ. Probing both key variants below means
# a container-written entry is still found from a docling-less host.
_EXTRACTOR_ENVS = ("docling", "legacy")

# The extractor environment the UPCOMING run will actually extract under —
# i.e. the only one whose cache hits are genuine 0-wall-clock no-ops for
# THIS run (issue #77 fix-round-2). Defaults to "docling": the documented
# pipeline always extracts inside the docker container (`make docker-run
# ... mine/judge`), which has docling per the Dockerfile — a host venv
# without docling is not the documented extraction path. Operators who
# really run extraction on the host can opt out explicitly.
_TARGET_ENV_VAR = "PLAYBOOK_ESTIMATE_TARGET_ENV"
_DEFAULT_TARGET_ENV = "docling"


def _target_env() -> str:
    """Return the extractor environment the upcoming run will use.

    Reads ``_TARGET_ENV_VAR``, defaulting to ``_DEFAULT_TARGET_ENV``
    (``"docling"``) when unset. Exits with a clear error rather than
    silently misreporting the ETA if set to anything other than one of
    ``_EXTRACTOR_ENVS``.
    """
    env = os.environ.get(_TARGET_ENV_VAR, _DEFAULT_TARGET_ENV)
    if env not in _EXTRACTOR_ENVS:
        raise SystemExit(
            f"{_TARGET_ENV_VAR}={env!r} is invalid — must be one of {_EXTRACTOR_ENVS!r}"
        )
    return env


def _extraction_cache_key_from_digest(file_sha256: str, extractor_env: str) -> str:
    """Build a cache key from an ALREADY-COMPUTED content digest.

    Split out from :func:`_extraction_cache_key` so callers probing
    multiple ``extractor_env`` candidates for the same file (see
    :func:`_cached_envs`) can hash the file's bytes once and reuse the
    digest, instead of re-reading and re-hashing the file per candidate —
    this script is documented to run "in seconds" over a multi-GB corpus,
    which a per-candidate re-hash would silently undermine.
    """
    payload = {
        "file_sha256": file_sha256,
        "format_version": _EXTRACTION_CACHE_FORMAT_VERSION,
        "extractor_env": extractor_env,
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def _extraction_cache_key(path: str, extractor_env: str) -> str:
    return _extraction_cache_key_from_digest(_file_sha256(path), extractor_env)


def _cached_envs(path: str, cached_keys: set[str]) -> set[str]:
    """Return the SET of extractor environments whose cache key for *path*
    is present in *cached_keys* — empty when neither is.

    A file legitimately accumulates cache entries under BOTH ``docling`` and
    ``legacy`` over time (e.g. a legacy host run followed later by a docling
    container run over the same ``$OUT``), so every matching environment
    must be returned, not just the first one found in ``_EXTRACTOR_ENVS``
    iteration order — returning only the first match mis-reports a
    dual-cached file as missing whenever the target environment (see
    :func:`_target_env`) isn't the one checked first (issue #77
    fix-round-3). Callers must check membership (``target_env in
    cached_envs``), never equality against a single returned value.

    Hashes *path* once and reuses the digest for both candidate keys (issue
    #77 fix-round-2) — ``_extraction_cache_key`` would otherwise re-read and
    re-hash the file per candidate.

    Note: the caller decides whether a hit counts as 0 wall-clock for the
    upcoming run — that depends on whether :func:`_target_env` is a member
    of the returned set, not on whether the set is non-empty (see ``main``).
    """
    digest = _file_sha256(path)
    return {
        env
        for env in _EXTRACTOR_ENVS
        if _extraction_cache_key_from_digest(digest, env) in cached_keys
    }


def load_cached_keys(out_dir: str) -> set[str]:
    """Return the set of extraction-cache keys in ``<out_dir>/extraction_cache.jsonl``.

    Empty when the cache file is absent/unreadable — a cold cache, so nothing
    is treated as already-extracted.
    """
    keys: set[str] = set()
    cache_path = os.path.join(out_dir, "extraction_cache.jsonl")
    try:
        with open(cache_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                with contextlib.suppress(Exception):  # corrupt line — skip
                    keys.add(json.loads(line)["key"])
    except OSError:
        pass  # no cache yet
    return keys


_warned_missing_pdfplumber = False


def is_scanned_pdf(path: str) -> bool:
    """True when the PDF's first pages carry almost no extractable text."""
    global _warned_missing_pdfplumber
    try:
        import pdfplumber
    except ImportError:
        if not _warned_missing_pdfplumber:
            print(
                "WARNING: pdfplumber not installed in this Python — "
                "scanned-PDF detection disabled, every PDF is counted as "
                "born-digital. ETA may be several times too low. Run this "
                "script with the repo venv: .venv/bin/python "
                "estimate_runtime.py ...",
                file=sys.stderr,
            )
            _warned_missing_pdfplumber = True
        return False  # can't tell; treat as born-digital (optimistic)
    try:
        with pdfplumber.open(path) as doc:
            pages = doc.pages[:3]
            chars = sum(len(p.extract_text() or "") for p in pages)
            per_page = chars / max(1, len(pages))
            return per_page < 100
    except Exception:
        return True  # unreadable by pdfplumber -> likely image-only/scanned


def fmt(seconds: float) -> str:
    h, m = divmod(int(seconds) // 60, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m"


def main() -> None:
    corpus = sys.argv[1] if len(sys.argv) > 1 else "."
    # Default out_dir mirrors the CLI (<corpus>/../out); override with argv[2].
    default_out = os.path.join(os.path.dirname(os.path.abspath(corpus)), "out")
    out_dir = sys.argv[2] if len(sys.argv) > 2 else default_out
    cached_keys = load_cached_keys(out_dir)
    target_env = _target_env()
    other_env = next(e for e in _EXTRACTOR_ENVS if e != target_env)

    folders = sorted(d for d in glob.glob(os.path.join(corpus, "*")) if os.path.isdir(d))
    n_docx = n_pdf_born = n_pdf_scanned = n_rtf = 0
    n_cached = 0  # cached under the TARGET env — genuine 0 wall-clock for this run
    n_other_only = 0  # cached under the OTHER env ONLY (not target) — will still re-extract
    n_cached_by_env = {"docling": 0, "legacy": 0}  # informational: every hit, by env
    raw_bytes = 0
    est = 0.0  # only versions NOT cached under the target env contribute wall-clock
    for d in folders:
        for f in glob.glob(os.path.join(d, "*")):
            low = f.lower()
            if not low.endswith((".pdf", ".docx", ".rtf")):
                continue
            raw_bytes += os.path.getsize(f)
            if low.endswith(".docx"):
                n_docx += 1
                per_file = _T_DOCX
            elif low.endswith(".rtf"):
                n_rtf += 1
                per_file = _T_DOCX
            elif is_scanned_pdf(f):
                n_pdf_scanned += 1
                per_file = _T_SCANNED
            else:
                n_pdf_born += 1
                per_file = _T_BORN_DIGITAL
            # A warm extraction-cache hit only makes this version's
            # extraction a genuine no-op when the file is cached under the
            # TARGET environment — the one the upcoming run will actually
            # use (issue #77 fix-round-2). A file can legitimately be
            # cached under BOTH environments at once (e.g. a legacy host
            # run followed later by a docling container run over the same
            # $OUT), so this must check SET MEMBERSHIP, not equality
            # against a single "the" cached environment (issue #77
            # fix-round-3) — otherwise a dual-cached file is mis-reported
            # as missing whenever the target isn't whichever environment a
            # first-match probe happened to find first. A hit under the
            # OTHER environment ONLY (not target) is real information
            # (reported separately below) but is NOT 0 wall-clock: the
            # upcoming run will miss it and re-extract, exactly like an
            # uncached file, so it stays in `est` rather than being
            # credited to `n_cached`.
            cached_envs = _cached_envs(f, cached_keys) if cached_keys else set()
            for env in cached_envs:
                n_cached_by_env[env] += 1
            if target_env in cached_envs:
                n_cached += 1
            else:
                est += per_file
                if other_env in cached_envs:
                    n_other_only += 1

    versions = n_docx + n_pdf_born + n_pdf_scanned + n_rtf
    n_uncached = versions - n_cached
    lo, hi = est * 0.6, est * 1.8  # calibration + timeout variance
    tokens = int(raw_bytes / 1_000_000 * _TOKENS_PER_MB)

    print("=" * 60)
    print("PRE-FLIGHT ESTIMATE (before extraction)")
    print("=" * 60)
    print(f"Target extractor env      : {target_env}  (override: {_TARGET_ENV_VAR}={other_env})")
    print(f"Agreements (folders)      : {len(folders)}")
    print(f"Negotiation versions      : {versions}")
    print(f"  born-digital PDF        : {n_pdf_born}")
    print(f"  scanned PDF (needs OCR) : {n_pdf_scanned}   <-- the slow ones")
    print(f"  DOCX                    : {n_docx}")
    if n_rtf:
        print(f"  RTF                     : {n_rtf}")
    print(f"Raw source size           : {raw_bytes / 1e6:.1f} MB")
    print(f"Est. extracted corpus     : ~{tokens:,} tokens (very rough)")
    print("-" * 60)
    if n_cached or n_other_only:
        d_n, l_n = n_cached_by_env["docling"], n_cached_by_env["legacy"]
        print(
            f"Extraction cache          : {n_cached}/{versions} version(s) already "
            f"extracted under the target env ({target_env}) (docling: {d_n}, legacy: {l_n})"
        )
        if n_cached:
            print(f"                            in {out_dir} — skipped (0 wall-clock)")
        if n_other_only:
            print(
                f"                            {n_other_only} version(s) cached under "
                f"{other_env} only — will be re-extracted under {target_env}"
            )
    if n_uncached == 0 and versions:
        print("EXTRACTION/OCR ETA        : ~0m — corpus already extracted (cache hit)")
        print("                            proceed straight to segmentation/judging.")
    else:
        label = f"{n_uncached} uncached version(s)" if n_cached else "all versions"
        print(f"EXTRACTION/OCR ETA        : ~{fmt(lo)}–{fmt(hi)} wall-clock (CPU), {label}")
        print("                            (this is the expensive step; render is seconds)")
    print("LLM API cost              : $0  (key-free; agent is the judge)")
    print(f"Judgment load (rough)     : ~{versions} scope+provenance + a few hundred")
    print("                            deduped deviation items for the agent to judge")
    print("=" * 60)
    if n_uncached:
        print("Scanned PDFs dominate wall-clock. To finish faster you can OCR them")
        print("separately, exclude them, or accept the wait. Confirm before proceeding.")
        print("Point --out at a prior run's out dir to reuse its extraction cache.")


if __name__ == "__main__":
    main()
