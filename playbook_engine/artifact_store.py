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
from collections.abc import Callable
from pathlib import Path
from typing import Any

# Bump this whenever a stage's compute logic changes in a way that should
# invalidate all existing cached entries (e.g. bug-fix, schema change).
#
# "2" (issue #219): keys hash each version file's NAME, not its absolute path
# (a moved corpus or a Docker-vs-host run now shares one cache), and the
# per-document "l1-l4" blob is split into a per-version "l1" layer and a
# per-document "l2-l4" layer keyed by the L1 outputs.
_CACHE_FORMAT_VERSION = "2"


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
    - each version file's *name* and the SHA-256 of its *content*
      (order-stable, lexicographic by path)
    - a config fingerprint (caller-supplied JSON-serialisable digest)
    - SHA-256 of ``hints.yaml`` content when present (drives version ordering)
    - the global cache-format version sentinel

    Using content hashes (not mtime) means ``cp -p`` / ``rsync -a`` produce
    cache hits, while a content change under a preserved mtime produces a miss.
    Only the file's *name* is hashed, never its absolute path (issue #219):
    moving the corpus directory, or running the same corpus inside Docker and
    on the host, must not invalidate anything. The name stays in the key
    because it becomes the version id the stage output carries.
    """
    h = hashlib.sha256()
    h.update(_CACHE_FORMAT_VERSION.encode())
    h.update(doc_id.encode())
    h.update(stage.encode())
    for vf in sorted(version_files):
        h.update(vf.name.encode())
        h.update(_sha256_file(vf).encode())
    h.update(config_fingerprint.encode())
    # hints.yaml drives version ordering → must be part of the key.
    if hints_path is not None and hints_path.exists():
        h.update(b"hints:")
        h.update(_sha256_file(hints_path).encode())
    else:
        h.update(b"hints:absent")
    return h.hexdigest()


def make_stage_key(
    doc_id: str,
    stage: str,
    inputs_fingerprint: str,
    config_fingerprint: str,
    hints_path: Path | None = None,
) -> str:
    """Key for a stage whose inputs are an EARLIER stage's outputs (issue #219).

    ``make_doc_key`` hashes source files; a downstream layer (L2-L4) is keyed
    instead by a fingerprint of the upstream layer's results
    (*inputs_fingerprint*), so recomputing L1 to byte-identical output — a
    changed extractor environment that reads the same text, say — still
    replays L2-L4. ``hints.yaml`` is folded in exactly as ``make_doc_key``
    does: it drives version ordering, which is L2.
    """
    h = hashlib.sha256()
    h.update(_CACHE_FORMAT_VERSION.encode())
    h.update(doc_id.encode())
    h.update(stage.encode())
    h.update(b"inputs:")
    h.update(inputs_fingerprint.encode())
    h.update(config_fingerprint.encode())
    if hints_path is not None and hints_path.exists():
        h.update(b"hints:")
        h.update(_sha256_file(hints_path).encode())
    else:
        h.update(b"hints:absent")
    return h.hexdigest()


def make_config_fingerprint(data: Any) -> str:
    """Stable SHA-256 fingerprint for any JSON-serialisable *data*."""
    return _sha256_str(json.dumps(data, sort_keys=True, ensure_ascii=False))


def write_text_if_changed(path: Path, text: str, *, force: bool = False) -> bool:
    """Atomically write *text* to *path* unless the file already holds it.

    Issue #219: a warm run used to delete and rewrite every intermediate
    (``observations.jsonl``, ``trail/``, ``normalized/``, …) even when nothing
    changed, so every file's mtime moved and anything watching the out-dir
    saw a full rewrite. Comparing content first keeps an unchanged file
    untouched. *force* writes regardless (``mine --force-rewrite``).

    Returns ``True`` when the file was written.
    """
    if not force and path.is_file():
        try:
            if path.read_text(encoding="utf-8") == text:
                return False
        except (OSError, UnicodeDecodeError):
            pass  # unreadable: fall through and rewrite it
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
    return True


_MISSING = object()


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
        # Per-stage (hits, misses) — issue #219: one store serves several
        # layers (L1 per version, L2-L4 per document), each reported apart.
        self._stage_counts: dict[str, list[int]] = {}
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

    def stage_stats(self, stage: str) -> tuple[int, int]:
        """``(hits, misses)`` for lookups made with ``stage=`` *stage*."""
        hits, misses = self._stage_counts.get(stage, [0, 0])
        return hits, misses

    def get_or_compute(
        self,
        key: str,
        compute_fn: Any,
        *,
        cacheable: Callable[[], bool] | None = None,
        is_valid: Callable[[Any], bool] | None = None,
        stage: str | None = None,
    ) -> Any:
        """Return the cached value for *key*, or call *compute_fn()* and cache it.

        *compute_fn* must return a JSON-serialisable value (dict or list).
        The returned object is always freshly deserialised from JSON, so callers
        should treat it as read-only.

        *cacheable* (issue #218), when given, is asked AFTER *compute_fn* runs;
        ``False`` returns the freshly computed value WITHOUT persisting it, so
        the next run recomputes. For a result that reflects this run rather
        than the key's inputs (e.g. a version whose extraction timed out).

        *is_valid* (issue #219), when given, is asked about a cached value
        BEFORE it is replayed; ``False`` treats the entry as a miss — the value
        is recomputed and (subject to *cacheable*) overwritten. For a result
        whose validity rests on state outside the key, e.g. the stored judge
        verdicts a document's classifications were replayed from.

        *stage* only labels the lookup for :meth:`stage_stats`.
        """
        cached = self._read(key)
        if cached is not _MISSING and (is_valid is None or is_valid(cached)):
            self._hits += 1
            self._count(stage, hit=True)
            return cached

        # Cache miss — recompute and persist.
        value = compute_fn()
        self._misses += 1
        self._count(stage, hit=False)
        if cacheable is not None and not cacheable():
            return value
        self._persist(key, value)
        return value

    def invalidate(self, key: str) -> None:
        """Remove *key* from the index (does not delete the artifact file)."""
        if key in self._index:
            del self._index[key]
            self._flush_index()

    def contains(self, key: str, *, is_valid: Callable[[Any], bool] | None = None) -> bool:
        """Return True if *key* already has a cached artifact on disk.

        A pure peek — does not call ``compute_fn``, does not affect
        ``hit_count``/``miss_count``, and does not touch the index. Callers
        that want to skip expensive precomputation for documents that are
        already stage-cached (issue #92: excluding cache-hit documents from
        the batch-segmentation pre-pass) should use this instead of
        ``get_or_compute`` with a dummy ``compute_fn``.

        *is_valid* applies the same replay check :meth:`get_or_compute` would
        (issue #219), so a peek never promises a hit the lookup then refuses.
        """
        if is_valid is None:
            if key not in self._index:
                return False
            return (self._cache_dir / self._index[key]).exists()
        cached = self._read(key)
        return cached is not _MISSING and is_valid(cached)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _read(self, key: str) -> Any:
        """The cached value for *key*, or ``_MISSING`` (absent or corrupt)."""
        if key not in self._index:
            return _MISSING
        artifact_path = self._cache_dir / self._index[key]
        if not artifact_path.exists():
            return _MISSING
        try:
            return json.loads(artifact_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 — corrupt entry: recompute
            return _MISSING

    def _count(self, stage: str | None, *, hit: bool) -> None:
        if stage is None:
            return
        counts = self._stage_counts.setdefault(stage, [0, 0])
        counts[0 if hit else 1] += 1

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
