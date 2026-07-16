"""The skill's pre-flight estimator must detect a warm extraction cache.

``.claude/skills/playbook-from-corpus/estimate_runtime.py`` is a standalone,
dependency-light script (stdlib only — no docling/torch import) that replicates
the engine's extraction-cache key recipe so it can report "already extracted,
0 wall-clock" without importing the engine. That replication is the fragile
part: if ``extraction``/``agent_judge`` ever change the key recipe or the
format version, the estimator would silently stop detecting cache hits and
scare the operator into a needless multi-hour re-OCR. This test pins the
coupling by proving the standalone key matches a real cache entry the ENGINE
wrote.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from playbook_engine.extraction import ExtractionCache, extract_blocks

_SCRIPT = (
    Path(__file__).parent.parent
    / ".claude"
    / "skills"
    / "playbook-from-corpus"
    / "estimate_runtime.py"
)


def _load_estimator():
    spec = importlib.util.spec_from_file_location("estimate_runtime", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_estimator_key_matches_engine_cache(tmp_path: Path) -> None:
    est = _load_estimator()

    src = tmp_path / "deal" / "v1.rtf"
    src.parent.mkdir(parents=True)
    src.write_text(r"{\rtf1 Fictional agreement text.}", encoding="utf-8")

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    cache = ExtractionCache(out_dir / "extraction_cache.jsonl")
    # The engine extracts and writes the cache entry under its OWN key.
    extract_blocks(src, cache=cache)

    cached_keys = est.load_cached_keys(str(out_dir))
    assert cached_keys, "estimator read no keys from a populated cache"
    assert est._extraction_cache_key(str(src)) in cached_keys, (
        "estimator's standalone key recipe no longer matches the engine's "
        "extraction-cache key — pre-flight would miss warm-cache hits and "
        "report a needless full re-OCR"
    )


def test_estimator_cold_cache_is_empty(tmp_path: Path) -> None:
    est = _load_estimator()
    # No cache file present → no keys, nothing treated as pre-extracted.
    assert est.load_cached_keys(str(tmp_path)) == set()
