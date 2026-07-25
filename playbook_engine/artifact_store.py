"""Content-addressed artifact cache for the playbook pipeline.

Provides per-document, per-stage caching keyed by a hash of:
  - the source file content(s)
  - any relevant configuration values
  - a stage code version sentinel (increment to bust all caches for a stage)

Artifacts are stored under ``{out_dir}/.cache/`` as JSON files, with a
small ``index.json`` mapping artifact keys to their relative paths.

Security: only serialised Python dicts / lists that have already passed
through the pipeline's own serialisation layer are written to disk.  No
raw agreement content is stored in this module.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

# Bump this whenever a stage's compute logic changes in a way that should
# invalidate all existing cached entries (e.g. bug-fix, schema change).
_CACHE_FORMAT_VERSION = "1"


def _sha256_file(path: Path) -> str:
    """Return the SHA-256 hex digest of *path*'s content."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_str(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def make_doc_key(
    doc_id: str,
    version_files: list[Path],
    config_fingerprint: str,
    stage: str,
    hints_path: Path | None = None,
) -> str:
    """Build a stable content-addressed key for one (doc_id, stage) pair.

    The key encodes:
    - the document id and stage name
    - SHA-256 of each version file's *content* (order-stable, lexicographic by path)
    - a config fingerprint (caller-supplied JSON-serialisable digest)
    - SHA-256 of ``hints.yaml`` content when present (drives version ordering)
    - the global cache-format version sentinel

    Using content hashes (not mtime) means ``cp -p`` / ``rsync -a`` produce
    cache hits, while a content change under a preserved mtime produces a miss.
    """
    h = hashlib.sha256()
    h.update(_CACHE_FORMAT_VERSION.encode())
    h.update(doc_id.encode())
    h.update(stage.encode())
    for vf in sorted(version_files):
        h.update(str(vf).encode())
        h.update(_sha256_file(vf).encode())
    h.update(config_fingerprint.encode())
    # hints.yaml drives version ordering → must be part of the key.
    if hints_path is not None and hints_path.exists():
        h.update(b"hints:")
        h.update(_sha256_file(hints_path).encode())
    else:
        h.update(b"hints:absent")
    return h.hexdigest()


def make_config_fingerprint(data: Any) -> str:
    """Stable SHA-256 fingerprint for any JSON-serialisable *data*."""
    return _sha256_str(json.dumps(data, sort_keys=True, ensure_ascii=False))


class ArtifactStore:
    """Content-addressed on-disk artifact cache.

    All cached values must be JSON-serialisable (dicts / lists / primitives).
    Entries are never automatically evicted; use ``no_cache=True`` or delete
    the ``.cache/`` directory to force a full recompute.

    Typical usage::

        store = ArtifactStore(out_dir / ".cache")
        key = make_doc_key(doc_id, version_files, cfg_fp, "l1-l4")
        result = store.get_or_compute(key, lambda: expensive_fn())
    """

    def __init__(self, cache_dir: Path) -> None:
        self._cache_dir = cache_dir
        self._index_path = cache_dir / "index.json"
        self._index: dict[str, str] = {}  # key -> relative path
        self._hits = 0
        self._misses = 0
        self._load_index()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    @property
    def hit_count(self) -> int:
        """Number of cache hits since this store was created."""
        return self._hits

    @property
    def miss_count(self) -> int:
        """Number of cache misses (recompute events) since this store was created."""
        return self._misses

    def get_or_compute(self, key: str, compute_fn: Any) -> Any:
        """Return the cached value for *key*, or call *compute_fn()* and cache it.

        *compute_fn* must return a JSON-serialisable value (dict or list).
        The returned object is always freshly deserialised from JSON, so callers
        should treat it as read-only.
        """
        if key in self._index:
            artifact_path = self._cache_dir / self._index[key]
            if artifact_path.exists():
                try:
                    value = json.loads(artifact_path.read_text(encoding="utf-8"))
                    self._hits += 1
                    return value
                except Exception:  # noqa: BLE001 — corrupt entry: recompute
                    pass

        # Cache miss — recompute and persist.
        value = compute_fn()
        self._persist(key, value)
        self._misses += 1
        return value

    def invalidate(self, key: str) -> None:
        """Remove *key* from the index (does not delete the artifact file)."""
        if key in self._index:
            del self._index[key]
            self._flush_index()

    def contains(self, key: str) -> bool:
        """Return True if *key* already has a cached artifact on disk.

        A pure peek — does not call ``compute_fn``, does not affect
        ``hit_count``/``miss_count``, and does not touch the index. Callers
        that want to skip expensive precomputation for documents that are
        already stage-cached (issue #92: excluding cache-hit documents from
        the batch-segmentation pre-pass) should use this instead of
        ``get_or_compute`` with a dummy ``compute_fn``.
        """
        if key not in self._index:
            return False
        return (self._cache_dir / self._index[key]).exists()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _persist(self, key: str, value: Any) -> None:
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        rel = f"{key[:2]}/{key}.json"
        artifact_path = self._cache_dir / rel
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = artifact_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(tmp, artifact_path)
        self._index[key] = rel
        self._flush_index()

    def _flush_index(self) -> None:
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        tmp = self._index_path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(self._index, sort_keys=True, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(tmp, self._index_path)

    def _load_index(self) -> None:
        if self._index_path.exists():
            try:
                self._index = json.loads(self._index_path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001 — corrupt index: start fresh
                self._index = {}
        else:
            self._index = {}
