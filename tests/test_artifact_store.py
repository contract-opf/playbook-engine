"""Tests for ArtifactStore — issue #61 (content-addressed stage cache).

Acceptance criteria verified here:
  AC-cache-1: get_or_compute returns the cached value on second call (no recompute).
  AC-cache-2: A cp -p touch (content unchanged, mtime reset) → cache hit.
  AC-cache-3: Content changed under preserved mtime → cache miss.
  AC-cache-4: ArtifactStore persists an index.json mapping keys to artifact paths.
  AC-cache-5: make_doc_key is stable for the same inputs.
  AC-cache-6: make_doc_key differs when file content changes.
  AC-cache-7: make_doc_key differs when hints.yaml changes (regression: hints drives version order).
  AC-pipeline-1: Second compile run (no changes) → all cache hits, zero stage recomputation.
  AC-pipeline-2: Edit one document → only that document's observations recomputed.
  AC-pipeline-3: --no-cache forces a full recompute.
  AC-pipeline-4: Remove a document → untouched docs are cache hits; removed doc's observations gone.
  AC-pipeline-5: Edit hints.yaml → that document's stages recompute (cache miss).
  AC-pipeline-6: Mutate template file content → all per-doc caches bust (config_fp changes).
  AC-pipeline-7 (issue #90): Toggle normalize_trail_across_versions → all per-doc caches bust.
  AC-pipeline-8 (issue #90): Bump PROMPT_VERSION → all per-doc caches bust.
  AC-pipeline-9 (issue #90): Bump SCHEMA_HASH → all per-doc caches bust.

SECURITY NOTE: All fixtures use programmatically constructed RTF text with
synthetic, fictional content. No real agreement files are referenced.
Fictional party names only (e.g. "Gamma Ltd", "Delta Inc").
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import patch

import yaml

from playbook_engine import pipeline as pipeline_module
from playbook_engine.artifact_store import ArtifactStore, make_config_fingerprint, make_doc_key
from playbook_engine.config import load_config
from playbook_engine.pipeline import mine_corpus
from playbook_engine.taxonomy import load_taxonomy

# ---------------------------------------------------------------------------
# RTF fixture helpers
# ---------------------------------------------------------------------------

_RTF_PROLOGUE = (
    r"{\rtf1\ansi\deff0"
    r"{\fonttbl{\f0\froman\fcharset0 Times New Roman;}}"
    r"\f0\fs24 "
)
_RTF_EPILOGUE = r"}"


def _rtf(body: str) -> str:
    return _RTF_PROLOGUE + body + _RTF_EPILOGUE


def _write_rtf(path: Path, body: str) -> None:
    path.write_text(_rtf(body), encoding="utf-8")


_TAXONOMY_PATH = Path(__file__).parent.parent / "spec" / "taxonomy" / "affiliation-agreement.yaml"

_DOC_BODY_A = (
    r"1. Indemnification\par "
    r"Gamma Ltd shall indemnify Delta Inc against third-party claims "
    r"arising from the placement programme.\par "
    r"2. Governing Law\par "
    r"This agreement is governed by the laws of the State of California.\par "
    r"3. Term\par "
    r"This agreement commences on the date of execution and continues for one year.\par "
)

_DOC_BODY_A_MODIFIED = (
    r"1. Indemnification\par "
    r"Gamma Ltd shall indemnify Delta Inc against all third-party claims "
    r"arising from the placement programme including legal costs.\par "
    r"2. Governing Law\par "
    r"This agreement is governed by the laws of the State of Texas.\par "
    r"3. Term\par "
    r"This agreement commences on the date of execution and continues for two years.\par "
)

_DOC_BODY_B = (
    r"1. Confidentiality\par "
    r"Each party shall keep confidential all proprietary information of the other party.\par "
    r"2. Governing Law\par "
    r"This agreement is governed by the laws of the State of New York.\par "
    r"3. Term\par "
    r"Initial term of one year with automatic renewal on the same terms.\par "
)


# ---------------------------------------------------------------------------
# Corpus factory
# ---------------------------------------------------------------------------


def _make_two_doc_corpus(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Return (corpus_dir, config_path, out_dir) for a two-document corpus."""
    corpus_dir = tmp_path / "corpus"

    doc_a = corpus_dir / "deal-a"
    doc_a.mkdir(parents=True)
    _write_rtf(doc_a / "v1.rtf", _DOC_BODY_A)

    doc_b = corpus_dir / "deal-b"
    doc_b.mkdir(parents=True)
    _write_rtf(doc_b / "v1.rtf", _DOC_BODY_B)

    cfg = {
        "agreement_type": {
            "id": "educational-affiliation",
            "name": "Educational Affiliation Agreement",
        },
        "baseline": {},
        "taxonomy": str(_TAXONOMY_PATH),
        "provenance": {"our_party_aliases": ["Gamma Ltd"]},
    }
    config_path = tmp_path / "playbook.config.yaml"
    config_path.write_text(yaml.dump(cfg), encoding="utf-8")

    out_dir = tmp_path / "out"
    return corpus_dir, config_path, out_dir


# ===========================================================================
# Unit tests — ArtifactStore
# ===========================================================================


class TestArtifactStoreGetOrCompute:
    """AC-cache-1: cache hit on second call."""

    def test_first_call_invokes_compute(self, tmp_path: Path) -> None:
        store = ArtifactStore(tmp_path / ".cache")
        calls: list[int] = []

        def _fn() -> dict[str, int]:
            calls.append(1)
            return {"x": 42}

        result = store.get_or_compute("key-a", _fn)
        assert result == {"x": 42}
        assert len(calls) == 1
        assert store.miss_count == 1
        assert store.hit_count == 0

    def test_second_call_returns_cached_value(self, tmp_path: Path) -> None:
        store = ArtifactStore(tmp_path / ".cache")
        calls: list[int] = []

        def _fn() -> list[int]:
            calls.append(1)
            return [1, 2, 3]

        store.get_or_compute("key-b", _fn)
        result2 = store.get_or_compute("key-b", _fn)
        assert result2 == [1, 2, 3]
        assert len(calls) == 1  # compute_fn only called once
        assert store.hit_count == 1

    def test_new_store_instance_reads_existing_cache(self, tmp_path: Path) -> None:
        """Persisted index survives process restarts (new ArtifactStore instance)."""
        cache_dir = tmp_path / ".cache"
        store1 = ArtifactStore(cache_dir)
        store1.get_or_compute("persistent-key", lambda: {"persisted": True})

        # New instance — should read the index from disk.
        store2 = ArtifactStore(cache_dir)
        calls: list[int] = []
        result = store2.get_or_compute("persistent-key", lambda: calls.append(1) or {})
        assert result == {"persisted": True}
        assert len(calls) == 0
        assert store2.hit_count == 1

    def test_different_keys_are_independent(self, tmp_path: Path) -> None:
        store = ArtifactStore(tmp_path / ".cache")
        store.get_or_compute("key-x", lambda: {"a": 1})
        store.get_or_compute("key-y", lambda: {"b": 2})

        assert store.get_or_compute("key-x", dict) == {"a": 1}
        assert store.get_or_compute("key-y", dict) == {"b": 2}


class TestArtifactStoreIndex:
    """AC-cache-4: index.json persists key→path mappings."""

    def test_index_json_written(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / ".cache"
        store = ArtifactStore(cache_dir)
        store.get_or_compute("idx-key", lambda: {"v": 1})

        index_path = cache_dir / "index.json"
        assert index_path.exists()
        idx = json.loads(index_path.read_text())
        assert "idx-key" in idx
        assert idx["idx-key"].endswith(".json")

    def test_artifact_file_exists(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / ".cache"
        store = ArtifactStore(cache_dir)
        store.get_or_compute("art-key", lambda: [1, 2])

        idx = json.loads((cache_dir / "index.json").read_text())
        artifact_path = cache_dir / idx["art-key"]
        assert artifact_path.exists()
        assert json.loads(artifact_path.read_text()) == [1, 2]


class TestMakeDocKey:
    """AC-cache-5/6: key stability and content-sensitivity."""

    def test_stable_for_same_inputs(self, tmp_path: Path) -> None:
        vf = tmp_path / "v1.rtf"
        vf.write_text("content", encoding="utf-8")
        fp = make_config_fingerprint({"k": "v"})
        k1 = make_doc_key("doc-1", [vf], fp, "l1-l4")
        k2 = make_doc_key("doc-1", [vf], fp, "l1-l4")
        assert k1 == k2

    def test_differs_on_content_change(self, tmp_path: Path) -> None:
        """AC-cache-3: content change under same path → different key."""
        vf = tmp_path / "v1.rtf"
        vf.write_text("original", encoding="utf-8")
        fp = make_config_fingerprint({})
        k1 = make_doc_key("doc-1", [vf], fp, "l1-l4")

        vf.write_text("modified", encoding="utf-8")
        k2 = make_doc_key("doc-1", [vf], fp, "l1-l4")
        assert k1 != k2

    def test_same_content_different_mtime_same_key(self, tmp_path: Path) -> None:
        """AC-cache-2: cp -p scenario — content unchanged, mtime changes → same key."""
        vf = tmp_path / "v1.rtf"
        vf.write_text("same content", encoding="utf-8")
        fp = make_config_fingerprint({})
        k1 = make_doc_key("doc-1", [vf], fp, "l1-l4")

        # Simulate cp -p: re-write same content but change mtime via touch.
        time.sleep(0.01)
        vf.touch()
        k2 = make_doc_key("doc-1", [vf], fp, "l1-l4")
        assert k1 == k2, "Key must be content-based, not mtime-based"

    def test_differs_on_doc_id_change(self, tmp_path: Path) -> None:
        vf = tmp_path / "v1.rtf"
        vf.write_text("content", encoding="utf-8")
        fp = make_config_fingerprint({})
        k1 = make_doc_key("doc-1", [vf], fp, "l1-l4")
        k2 = make_doc_key("doc-2", [vf], fp, "l1-l4")
        assert k1 != k2

    def test_differs_on_stage_change(self, tmp_path: Path) -> None:
        vf = tmp_path / "v1.rtf"
        vf.write_text("content", encoding="utf-8")
        fp = make_config_fingerprint({})
        k1 = make_doc_key("doc-1", [vf], fp, "l1-l4")
        k2 = make_doc_key("doc-1", [vf], fp, "l5")
        assert k1 != k2

    def test_differs_on_config_change(self, tmp_path: Path) -> None:
        vf = tmp_path / "v1.rtf"
        vf.write_text("content", encoding="utf-8")
        fp1 = make_config_fingerprint({"agreement": "nda"})
        fp2 = make_config_fingerprint({"agreement": "msa"})
        k1 = make_doc_key("doc-1", [vf], fp1, "l1-l4")
        k2 = make_doc_key("doc-1", [vf], fp2, "l1-l4")
        assert k1 != k2

    def test_differs_on_hints_yaml_change(self, tmp_path: Path) -> None:
        """AC-cache-7: editing hints.yaml changes the key → stale results cannot be served."""
        vf = tmp_path / "v1.rtf"
        vf.write_text("content", encoding="utf-8")
        fp = make_config_fingerprint({})

        hints_path = tmp_path / "hints.yaml"
        hints_path.write_text("order: [v1]\n", encoding="utf-8")
        k1 = make_doc_key("doc-1", [vf], fp, "l1-l4", hints_path)

        hints_path.write_text("order: [v2, v1]\n", encoding="utf-8")
        k2 = make_doc_key("doc-1", [vf], fp, "l1-l4", hints_path)
        assert k1 != k2, "Editing hints.yaml must produce a different cache key"

    def test_absent_hints_yaml_differs_from_present(self, tmp_path: Path) -> None:
        """Adding a hints.yaml where none existed must produce a new cache key."""
        vf = tmp_path / "v1.rtf"
        vf.write_text("content", encoding="utf-8")
        fp = make_config_fingerprint({})
        hints_path = tmp_path / "hints.yaml"

        k_absent = make_doc_key("doc-1", [vf], fp, "l1-l4", hints_path)

        hints_path.write_text("order: [v1]\n", encoding="utf-8")
        k_present = make_doc_key("doc-1", [vf], fp, "l1-l4", hints_path)
        assert k_absent != k_present, "Creating hints.yaml must produce a different cache key"


# ===========================================================================
# Integration tests — pipeline incrementality
# ===========================================================================


class TestPipelineIncrementality:
    """AC-pipeline-1/2/3: incrementality acceptance criteria."""

    def _run_mine(
        self,
        corpus_dir: Path,
        config_path: Path,
        out_dir: Path,
        *,
        no_cache: bool = False,
        **mine_kwargs: object,
    ) -> None:
        cfg = load_config(config_path)
        taxonomy = load_taxonomy(_TAXONOMY_PATH)
        mine_corpus(
            corpus_dir=corpus_dir,
            config=cfg,
            taxonomy=taxonomy,
            out_dir=out_dir,
            no_cache=no_cache,
            **mine_kwargs,
        )

    def test_second_run_no_changes_all_cache_hits(self, tmp_path: Path) -> None:
        """AC-pipeline-1: second run with no changes → all cache hits, zero recompute."""
        corpus_dir, config_path, out_dir = _make_two_doc_corpus(tmp_path)

        # Track compute_fn invocations via a patch on the inner helper.
        compute_calls: list[str] = []
        original_get_or_compute = ArtifactStore.get_or_compute

        def _tracking_get_or_compute(self: ArtifactStore, key: str, compute_fn: object) -> object:
            def _counted(*args: object, **kwargs: object) -> object:
                compute_calls.append(key)
                return (compute_fn)()  # type: ignore[operator]

            return original_get_or_compute(self, key, _counted)

        # First run — primes the cache.
        with patch.object(ArtifactStore, "get_or_compute", _tracking_get_or_compute):
            self._run_mine(corpus_dir, config_path, out_dir)
        first_run_calls = list(compute_calls)
        assert len(first_run_calls) == 2, f"Expected 2 doc computes, got {first_run_calls}"

        compute_calls.clear()

        # Second run — should be all cache hits.
        with patch.object(ArtifactStore, "get_or_compute", _tracking_get_or_compute):
            self._run_mine(corpus_dir, config_path, out_dir)
        assert len(compute_calls) == 0, (
            f"Expected 0 recomputes on second run (all cache hits), got {compute_calls}"
        )

    def test_edit_one_doc_only_that_doc_recomputed(self, tmp_path: Path) -> None:
        """AC-pipeline-2: edit one doc → only that doc's stages recompute."""
        corpus_dir, config_path, out_dir = _make_two_doc_corpus(tmp_path)

        # Prime cache.
        self._run_mine(corpus_dir, config_path, out_dir)
        obs_first = (out_dir / "observations.jsonl").read_text(encoding="utf-8")

        # Modify deal-a's content.
        _write_rtf(corpus_dir / "deal-a" / "v1.rtf", _DOC_BODY_A_MODIFIED)

        compute_calls: list[str] = []
        original_get_or_compute = ArtifactStore.get_or_compute

        def _tracking_get_or_compute(self: ArtifactStore, key: str, compute_fn: object) -> object:
            def _counted(*args: object, **kwargs: object) -> object:
                compute_calls.append(key)
                return (compute_fn)()  # type: ignore[operator]

            return original_get_or_compute(self, key, _counted)

        with patch.object(ArtifactStore, "get_or_compute", _tracking_get_or_compute):
            self._run_mine(corpus_dir, config_path, out_dir)

        assert len(compute_calls) == 1, (
            f"Expected exactly 1 recompute (deal-a only), got {len(compute_calls)}: {compute_calls}"
        )
        # Observations should have changed (deal-a was modified).
        obs_second = (out_dir / "observations.jsonl").read_text(encoding="utf-8")
        assert obs_second != obs_first, "Observations must change when a document is edited"

    def test_no_cache_forces_full_recompute(self, tmp_path: Path) -> None:
        """AC-pipeline-3: --no-cache forces recompute of all documents."""
        corpus_dir, config_path, out_dir = _make_two_doc_corpus(tmp_path)

        # Prime cache.
        self._run_mine(corpus_dir, config_path, out_dir)

        # Second run with no_cache=True — ArtifactStore should not be used at all.
        compute_calls: list[str] = []
        original_get_or_compute = ArtifactStore.get_or_compute

        def _tracking_get_or_compute(self: ArtifactStore, key: str, compute_fn: object) -> object:
            compute_calls.append(key)
            return original_get_or_compute(self, key, compute_fn)

        with patch.object(ArtifactStore, "get_or_compute", _tracking_get_or_compute):
            self._run_mine(corpus_dir, config_path, out_dir, no_cache=True)

        # With no_cache=True, the store is None and get_or_compute is never called.
        assert len(compute_calls) == 0, (
            "With no_cache=True the ArtifactStore should not be instantiated"
        )

    def test_add_doc_only_new_doc_computed(self, tmp_path: Path) -> None:
        """Adding a document → only the new doc is computed; existing docs are cache hits."""
        corpus_dir, config_path, out_dir = _make_two_doc_corpus(tmp_path)

        # Prime cache with two docs.
        self._run_mine(corpus_dir, config_path, out_dir)

        # Add a third document.
        doc_c = corpus_dir / "deal-c"
        doc_c.mkdir()
        _write_rtf(
            doc_c / "v1.rtf",
            (
                r"1. Payment\par Epsilon Corp will pay Zeta Ltd monthly.\par "
                r"2. Term\par Three-year initial term.\par "
            ),
        )

        compute_calls: list[str] = []
        original_get_or_compute = ArtifactStore.get_or_compute

        def _tracking_get_or_compute(self: ArtifactStore, key: str, compute_fn: object) -> object:
            def _counted(*args: object, **kwargs: object) -> object:
                compute_calls.append(key)
                return (compute_fn)()  # type: ignore[operator]

            return original_get_or_compute(self, key, _counted)

        with patch.object(ArtifactStore, "get_or_compute", _tracking_get_or_compute):
            self._run_mine(corpus_dir, config_path, out_dir)

        assert len(compute_calls) == 1, (
            f"Expected exactly 1 recompute (new deal-c only), got {len(compute_calls)}"
        )

    def test_cp_p_preserved_mtime_is_cache_hit(self, tmp_path: Path) -> None:
        """AC-cache-2: content-preserving copy (same content, different mtime) → cache hit."""
        corpus_dir, config_path, out_dir = _make_two_doc_corpus(tmp_path)

        # Prime cache.
        self._run_mine(corpus_dir, config_path, out_dir)
        obs_first = (out_dir / "observations.jsonl").read_text(encoding="utf-8")

        # Simulate cp -p: copy deal-a/v1.rtf over itself (same content, update mtime).
        vf = corpus_dir / "deal-a" / "v1.rtf"
        content = vf.read_text(encoding="utf-8")
        time.sleep(0.02)  # ensure mtime differs
        vf.write_text(content, encoding="utf-8")  # same content
        # mtime is now newer but content is identical → must be a cache hit

        compute_calls: list[str] = []
        original_get_or_compute = ArtifactStore.get_or_compute

        def _tracking_get_or_compute(self: ArtifactStore, key: str, compute_fn: object) -> object:
            def _counted(*args: object, **kwargs: object) -> object:
                compute_calls.append(key)
                return (compute_fn)()  # type: ignore[operator]

            return original_get_or_compute(self, key, _counted)

        with patch.object(ArtifactStore, "get_or_compute", _tracking_get_or_compute):
            self._run_mine(corpus_dir, config_path, out_dir)

        assert len(compute_calls) == 0, (
            "Same content under updated mtime must be a cache hit (content hash, not mtime)"
        )
        obs_second = (out_dir / "observations.jsonl").read_text(encoding="utf-8")
        assert obs_second == obs_first, "Observations must be byte-for-byte identical on cache hit"

    def test_remove_doc_untouched_docs_are_cache_hits(self, tmp_path: Path) -> None:
        """AC-pipeline-4: remove a document → untouched docs are cache hits; removed doc gone."""
        corpus_dir, config_path, out_dir = _make_two_doc_corpus(tmp_path)

        # Prime cache with two docs (deal-a and deal-b).
        self._run_mine(corpus_dir, config_path, out_dir)
        obs_first = (out_dir / "observations.jsonl").read_text(encoding="utf-8")

        # Remove deal-b from the corpus.
        import shutil

        shutil.rmtree(corpus_dir / "deal-b")

        compute_calls: list[str] = []
        original_get_or_compute = ArtifactStore.get_or_compute

        def _tracking_get_or_compute(self: ArtifactStore, key: str, compute_fn: object) -> object:
            def _counted(*args: object, **kwargs: object) -> object:
                compute_calls.append(key)
                return (compute_fn)()  # type: ignore[operator]

            return original_get_or_compute(self, key, _counted)

        with patch.object(ArtifactStore, "get_or_compute", _tracking_get_or_compute):
            self._run_mine(corpus_dir, config_path, out_dir)

        # deal-a was not touched → must be a cache hit (0 recomputes).
        assert len(compute_calls) == 0, (
            f"Expected 0 recomputes after removing deal-b (deal-a untouched), got {compute_calls}"
        )

        # deal-b's observations must no longer appear in the output.
        obs_after = (out_dir / "observations.jsonl").read_text(encoding="utf-8")
        # The combined output changed (deal-b's lines are gone).
        assert obs_after != obs_first, (
            "Observations must change after removing a document from the corpus"
        )
        # Specifically, no observation should reference the removed document.
        obs_lines = [json.loads(line) for line in obs_after.splitlines() if line.strip()]
        assert all(obs["citation"]["document_id"] != "deal-b" for obs in obs_lines), (
            "Removed document's observations must not appear in the output"
        )

    def test_edit_hints_yaml_causes_cache_miss(self, tmp_path: Path) -> None:
        """AC-pipeline-5: editing hints.yaml triggers a recompute for that document."""
        corpus_dir, config_path, out_dir = _make_two_doc_corpus(tmp_path)

        # Add a second version to deal-a so hints.yaml has something to order.
        _write_rtf(corpus_dir / "deal-a" / "v2.rtf", _DOC_BODY_A_MODIFIED)

        # Prime cache (no hints.yaml present yet).
        self._run_mine(corpus_dir, config_path, out_dir)

        # Write a hints.yaml file for deal-a — this changes the cache key.
        hints_path = corpus_dir / "deal-a" / "hints.yaml"
        hints_path.write_text("order: [v1, v2]\n", encoding="utf-8")

        compute_calls: list[str] = []
        original_get_or_compute = ArtifactStore.get_or_compute

        def _tracking_get_or_compute(self: ArtifactStore, key: str, compute_fn: object) -> object:
            def _counted(*args: object, **kwargs: object) -> object:
                compute_calls.append(key)
                return (compute_fn)()  # type: ignore[operator]

            return original_get_or_compute(self, key, _counted)

        with patch.object(ArtifactStore, "get_or_compute", _tracking_get_or_compute):
            self._run_mine(corpus_dir, config_path, out_dir)

        # deal-a's key changed (hints.yaml added) → exactly 1 recompute.
        assert len(compute_calls) == 1, (
            f"Expected 1 recompute (deal-a hints.yaml added), got {len(compute_calls)}: {compute_calls}"
        )

    def test_mutate_template_content_busts_all_caches(self, tmp_path: Path) -> None:
        """AC-pipeline-6: mutating template file content → all per-doc caches bust."""
        corpus_dir = tmp_path / "corpus"
        doc_a = corpus_dir / "deal-a"
        doc_a.mkdir(parents=True)
        _write_rtf(doc_a / "v1.rtf", _DOC_BODY_A)
        doc_b = corpus_dir / "deal-b"
        doc_b.mkdir(parents=True)
        _write_rtf(doc_b / "v1.rtf", _DOC_BODY_B)

        # Create a template file (RTF) at the same path used throughout.
        template_path = tmp_path / "template.rtf"
        _write_rtf(template_path, _DOC_BODY_A)

        cfg = {
            "agreement_type": {
                "id": "educational-affiliation",
                "name": "Educational Affiliation Agreement",
            },
            "baseline": {"template": "template.rtf"},
            "taxonomy": str(_TAXONOMY_PATH),
            "provenance": {"our_party_aliases": ["Gamma Ltd"]},
        }
        config_path = tmp_path / "playbook.config.yaml"
        config_path.write_text(yaml.dump(cfg), encoding="utf-8")
        out_dir = tmp_path / "out"

        # Prime cache.
        self._run_mine(corpus_dir, config_path, out_dir)

        # Mutate template content (same path, new content).
        _write_rtf(template_path, _DOC_BODY_B)

        compute_calls: list[str] = []
        original_get_or_compute = ArtifactStore.get_or_compute

        def _tracking_get_or_compute(self: ArtifactStore, key: str, compute_fn: object) -> object:
            def _counted(*args: object, **kwargs: object) -> object:
                compute_calls.append(key)
                return (compute_fn)()  # type: ignore[operator]

            return original_get_or_compute(self, key, _counted)

        with patch.object(ArtifactStore, "get_or_compute", _tracking_get_or_compute):
            self._run_mine(corpus_dir, config_path, out_dir)

        # Both documents must recompute because the config fingerprint changed.
        assert len(compute_calls) == 2, (
            f"Expected 2 recomputes (template content changed → all caches bust), "
            f"got {len(compute_calls)}: {compute_calls}"
        )

    def test_toggle_normalize_trail_busts_all_caches(self, tmp_path: Path) -> None:
        """AC-pipeline-7 (issue #90): flipping ``normalize_trail_across_versions``
        alone, with source content and every other config value unchanged,
        must bust every per-doc cache entry — otherwise a prior run's
        un-normalized cached trees are replayed silently and the feature
        appears to do nothing.
        """
        corpus_dir, config_path, out_dir = _make_two_doc_corpus(tmp_path)

        # Prime cache with normalize_trail_across_versions=False (the default).
        self._run_mine(corpus_dir, config_path, out_dir, normalize_trail_across_versions=False)

        compute_calls: list[str] = []
        original_get_or_compute = ArtifactStore.get_or_compute

        def _tracking_get_or_compute(self: ArtifactStore, key: str, compute_fn: object) -> object:
            def _counted(*args: object, **kwargs: object) -> object:
                compute_calls.append(key)
                return (compute_fn)()  # type: ignore[operator]

            return original_get_or_compute(self, key, _counted)

        with patch.object(ArtifactStore, "get_or_compute", _tracking_get_or_compute):
            self._run_mine(corpus_dir, config_path, out_dir, normalize_trail_across_versions=True)

        assert len(compute_calls) == 2, (
            f"Expected 2 recomputes (normalize_trail_across_versions changed → "
            f"all caches bust), got {len(compute_calls)}: {compute_calls}"
        )

    def test_bumped_prompt_version_busts_all_caches(self, tmp_path: Path) -> None:
        """AC-pipeline-8 (issue #90): bumping ``PROMPT_VERSION`` (a code change
        simulating a segmenter prompt revision), with everything else
        unchanged, must bust every per-doc cache entry rather than replay a
        tree segmented under the old prompt.
        """
        corpus_dir, config_path, out_dir = _make_two_doc_corpus(tmp_path)

        self._run_mine(corpus_dir, config_path, out_dir)

        compute_calls: list[str] = []
        original_get_or_compute = ArtifactStore.get_or_compute

        def _tracking_get_or_compute(self: ArtifactStore, key: str, compute_fn: object) -> object:
            def _counted(*args: object, **kwargs: object) -> object:
                compute_calls.append(key)
                return (compute_fn)()  # type: ignore[operator]

            return original_get_or_compute(self, key, _counted)

        with (
            patch.object(pipeline_module, "PROMPT_VERSION", "v2-bumped"),
            patch.object(ArtifactStore, "get_or_compute", _tracking_get_or_compute),
        ):
            self._run_mine(corpus_dir, config_path, out_dir)

        assert len(compute_calls) == 2, (
            f"Expected 2 recomputes (PROMPT_VERSION bumped → all caches bust), "
            f"got {len(compute_calls)}: {compute_calls}"
        )

    def test_bumped_schema_hash_busts_all_caches(self, tmp_path: Path) -> None:
        """AC-pipeline-9 (issue #90): same as above for ``SCHEMA_HASH`` — a
        segmenter output-schema change must also bust the stage cache.
        """
        corpus_dir, config_path, out_dir = _make_two_doc_corpus(tmp_path)

        self._run_mine(corpus_dir, config_path, out_dir)

        compute_calls: list[str] = []
        original_get_or_compute = ArtifactStore.get_or_compute

        def _tracking_get_or_compute(self: ArtifactStore, key: str, compute_fn: object) -> object:
            def _counted(*args: object, **kwargs: object) -> object:
                compute_calls.append(key)
                return (compute_fn)()  # type: ignore[operator]

            return original_get_or_compute(self, key, _counted)

        with (
            patch.object(pipeline_module, "SCHEMA_HASH", "deadbeef-bumped"),
            patch.object(ArtifactStore, "get_or_compute", _tracking_get_or_compute),
        ):
            self._run_mine(corpus_dir, config_path, out_dir)

        assert len(compute_calls) == 2, (
            f"Expected 2 recomputes (SCHEMA_HASH bumped → all caches bust), "
            f"got {len(compute_calls)}: {compute_calls}"
        )
