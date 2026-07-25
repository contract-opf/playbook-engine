#!/usr/bin/env python3
"""Pre-flight runtime estimate for a corpus, BEFORE the expensive extraction.

Counts agreements/versions, classifies each PDF as born-digital vs scanned
(scanned PDFs OCR ~5-10x slower under docling), and prints a wall-clock ETA
range plus rough corpus-size and judgment-load estimates. Uses only pdfplumber
(no docling/torch), so it runs on the host venv in seconds.

Usage: python estimate_runtime.py <corpus_dir> [out_dir]

If an ``out_dir`` with a warm extraction cache exists (``<out>/extraction_cache.jsonl``,
written by a prior/parallel ``mine``/``judge``/``segment`` run over the same
files), already-extracted versions are detected and contribute ~0 to the ETA —
so a re-run over an already-OCR'd corpus reports minutes, not hours. Defaults
to ``<corpus>/../out`` (the CLI's default). The cache is keyed by file content
hash only, so any run pointed at the same out_dir reuses it.

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
# content-only key recipe in ``agent_judge._payload_key`` /
# ``extraction._extraction_cache_payload``. Replicated here (stdlib only) so
# this pre-flight stays dependency-light — no engine import, no docling/torch.
# Degrades safely: if the recipe ever drifts, cache hits simply go undetected
# and the ETA is over-estimated (conservative), never wrong-low.
_EXTRACTION_CACHE_FORMAT_VERSION = "1"

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


def _extraction_cache_key(path: str) -> str:
    payload = {
        "file_sha256": _file_sha256(path),
        "format_version": _EXTRACTION_CACHE_FORMAT_VERSION,
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()


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


def is_scanned_pdf(path: str) -> bool:
    """True when the PDF's first pages carry almost no extractable text."""
    try:
        import pdfplumber
    except ImportError:
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

    folders = sorted(d for d in glob.glob(os.path.join(corpus, "*")) if os.path.isdir(d))
    n_docx = n_pdf_born = n_pdf_scanned = n_rtf = 0
    n_cached = 0
    raw_bytes = 0
    est = 0.0  # only UNCACHED versions contribute wall-clock
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
            # A warm extraction-cache hit (prior/parallel run over the same
            # bytes) makes this version's extraction a no-op — 0 wall-clock.
            if cached_keys and _extraction_cache_key(f) in cached_keys:
                n_cached += 1
            else:
                est += per_file

    versions = n_docx + n_pdf_born + n_pdf_scanned + n_rtf
    n_uncached = versions - n_cached
    lo, hi = est * 0.6, est * 1.8  # calibration + timeout variance
    tokens = int(raw_bytes / 1_000_000 * _TOKENS_PER_MB)

    print("=" * 60)
    print("PRE-FLIGHT ESTIMATE (before extraction)")
    print("=" * 60)
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
    if n_cached:
        print(f"Extraction cache          : {n_cached}/{versions} version(s) already")
        print(f"                            extracted in {out_dir} — skipped (0 wall-clock)")
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
