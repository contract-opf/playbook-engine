"""Regression tests for normalized/ clause-tree materialisation (issue #139).

``normalized/<doc_id>/*.clauses.json`` used to be written raw, mid-loop,
under the RAW document_id — before the born-safe pseudonymization pass ran,
and only on a stage-cache MISS (a warm ``ArtifactStore`` hit skipped
``_compute_doc_result`` entirely, so nothing was (re)written at all). Unlike
``trail/`` (stale-cleared and alias-renamed every run — issue #182/#51),
``normalized/`` accumulated raw-named directories forever and went stale on
any cache-hit run.

The fix threads each version's serialised ``ClauseTree`` through the same
cacheable per-doc result dict ``observations``/``trail`` already travel
through, and defers the actual disk write to AFTER the corpus-wide
pseudonymization pass — mirroring trail/'s stale-clear-then-rewrite-under-
the-alias treatment exactly (see pipeline.mine_corpus's "Materialise
normalized/ clause trees now" comment).

This module proves the two concrete regressions the fix closes:

  1. ``test_normalized_trees_survive_a_warm_stage_cache_hit`` — a second
     ``mine_corpus`` run over an UNCHANGED corpus (a pure stage-cache hit —
     asserted via the store's own hit-count progress line, not assumed) still
     produces a correctly-aliased ``normalized/`` tree. Pre-fix, the second
     run's ``normalized/`` directory would be empty (or a repeat of whatever
     the stale-clear left behind), because ``_compute_doc_result`` — the only
     code path that ever wrote to ``normalized/`` — never runs on a cache hit.

  2. ``test_normalized_trees_stale_cleared_when_document_renamed`` — a
     document renamed/removed between two runs leaves no orphaned tree
     directory behind, the same guarantee issue #51 gives ``trail/``.

SECURITY NOTE: the entity name below is synthetic ("Example University") —
no real counterparty. No live LLM anywhere: the default stub judges
(``_AllInScopeJudge``/``_NullClassificationJudge``/``_NullDeviationJudge``)
handle every document, matching test_pipeline_judgment_cache.py's convention.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from playbook_engine.config import load_config
from playbook_engine.entity_registry import entity_slug
from playbook_engine.pipeline import mine_corpus
from playbook_engine.taxonomy import load_taxonomy

_TAXONOMY_PATH = Path(__file__).parent.parent / "spec" / "taxonomy" / "affiliation-agreement.yaml"

_RTF_PROLOGUE = (
    r"{\rtf1\ansi\deff0"
    r"{\fonttbl{\f0\froman\fcharset0 Times New Roman;}}"
    r"\f0\fs24 "
)
_RTF_EPILOGUE = r"}"

ENTITY_NAME = "Example University"
ENTITY_SLUG = entity_slug(ENTITY_NAME)  # "example-university"


def _write_rtf(path: Path, body: str) -> None:
    path.write_text(_RTF_PROLOGUE + body + _RTF_EPILOGUE, encoding="utf-8")


def _doc_body(marker: str) -> str:
    """Two clauses (scope_gate.MIN_CLAUSE_COUNT) — heading + body both name
    the entity, the everyday whole-word case entity_registry.pseudonymize_text
    is proven to catch."""
    return (
        rf"1. {ENTITY_NAME} Data Rights\par "
        rf"{ENTITY_NAME} shall retain exclusive ownership of all placement "
        rf"data submitted under this {marker} agreement.\par "
        r"2. Notices\par "
        r"All notices shall be delivered in writing to the addresses set "
        r"forth on the signature page.\par "
    )


def _write_config(tmp_path: Path) -> Path:
    cfg = {
        "agreement_type": {
            "id": "educational-affiliation",
            "name": "Educational Affiliation Agreement",
        },
        "baseline": {"template": None},
        "taxonomy": str(_TAXONOMY_PATH),
        "provenance": {"known_entities": [ENTITY_NAME]},
    }
    config_path = tmp_path / "playbook.config.yaml"
    config_path.write_text(yaml.dump(cfg), encoding="utf-8")
    return config_path


def _tree_files(out_dir: Path) -> list[Path]:
    normalized_dir = out_dir / "normalized"
    if not normalized_dir.exists():
        return []
    return sorted(normalized_dir.rglob("*.clauses.json"))


def _assert_no_raw_entity(paths: list[Path]) -> None:
    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert ENTITY_NAME not in text, (
            f"RAW ENTITY NAME LEAK in normalized/ tree content: {path}\n{text[:300]}"
        )
        assert not any(part.startswith(ENTITY_SLUG) for part in path.parts), (
            f"normalized/ path must be ALIASED, not the raw slug: {path}"
        )


# ---------------------------------------------------------------------------
# 1. Rewritten (not silently skipped) on a warm stage-cache hit.
# ---------------------------------------------------------------------------


def test_normalized_trees_survive_a_warm_stage_cache_hit(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "corpus"
    doc_dir = corpus_dir / f"{ENTITY_SLUG}-alpha"
    doc_dir.mkdir(parents=True)
    _write_rtf(doc_dir / "v1.rtf", _doc_body("alpha"))

    config_path = _write_config(tmp_path)
    taxonomy = load_taxonomy(_TAXONOMY_PATH)
    config = load_config(config_path)
    out_dir = tmp_path / "out"
    registry_path = tmp_path / "registry.json"

    progress_lines_1: list[str] = []
    mine_corpus(
        corpus_dir=corpus_dir,
        config=config,
        taxonomy=taxonomy,
        out_dir=out_dir,
        entity_registry_path=registry_path,
        no_cache=False,
        progress=progress_lines_1.append,
    )

    # Sanity: run 1 actually computed (a cache miss) and wrote a correctly
    # aliased tree — not a vacuous pass.
    assert any("misses=1" in line for line in progress_lines_1), (
        f"run 1 must be a stage-cache MISS; progress={progress_lines_1}"
    )
    trees_1 = _tree_files(out_dir)
    assert len(trees_1) == 1, f"expected exactly one normalized/ tree after run 1, got {trees_1}"
    assert not trees_1[0].parent.name.startswith(ENTITY_SLUG), (
        f"run 1's normalized/ directory must already be ALIASED: {trees_1[0]}"
    )
    _assert_no_raw_entity(trees_1)
    aliased_dir_name = trees_1[0].parent.name

    # ---- Run 2: identical corpus + config -> pure stage-cache HIT. --------
    progress_lines_2: list[str] = []
    mine_corpus(
        corpus_dir=corpus_dir,
        config=config,
        taxonomy=taxonomy,
        out_dir=out_dir,
        entity_registry_path=registry_path,
        no_cache=False,
        progress=progress_lines_2.append,
    )

    assert any("hits=1" in line and "misses=0" in line for line in progress_lines_2), (
        f"run 2 must be a pure stage-cache HIT (proves this test exercises the "
        f"cache-hit path, not a second cache miss); progress={progress_lines_2}"
    )

    # The load-bearing assertion (issue #139): pre-fix, _compute_doc_result —
    # the ONLY code path that ever wrote normalized/ — never runs on a cache
    # hit, so normalized/ would be empty after this second run (the run-1
    # tree having been stale-cleared, mirroring trail/'s issue #51
    # treatment, with nothing to replace it). The fix threads each version's
    # serialised tree through the cached per-doc result itself, so a cache
    # hit still yields a correctly aliased rewrite.
    trees_2 = _tree_files(out_dir)
    assert len(trees_2) == 1, (
        f"normalized/ must still contain exactly one tree after a stage-cache-hit "
        f"run, not be left empty; got {trees_2}"
    )
    assert trees_2[0].parent.name == aliased_dir_name, (
        f"the aliased directory name must be stable across runs; "
        f"run 1={aliased_dir_name!r} run 2={trees_2[0].parent.name!r}"
    )
    _assert_no_raw_entity(trees_2)


# ---------------------------------------------------------------------------
# 2. Stale-cleared when a document is renamed/removed between runs.
# ---------------------------------------------------------------------------


def test_normalized_trees_stale_cleared_when_document_renamed(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "corpus"
    old_doc_dir = corpus_dir / f"{ENTITY_SLUG}-old-name"
    old_doc_dir.mkdir(parents=True)
    _write_rtf(old_doc_dir / "v1.rtf", _doc_body("old"))

    config_path = _write_config(tmp_path)
    taxonomy = load_taxonomy(_TAXONOMY_PATH)
    config = load_config(config_path)
    out_dir = tmp_path / "out"
    registry_path = tmp_path / "registry.json"

    mine_corpus(
        corpus_dir=corpus_dir,
        config=config,
        taxonomy=taxonomy,
        out_dir=out_dir,
        entity_registry_path=registry_path,
        no_cache=False,
    )
    trees_1 = _tree_files(out_dir)
    assert len(trees_1) == 1
    old_aliased_dir = trees_1[0].parent

    # Rename the document in the corpus (same content, new folder name) —
    # a different content-addressed cache key, so this is a fresh compute,
    # not a cache hit; what this proves is the STALE-CLEAR half of the fix.
    new_doc_dir = corpus_dir / f"{ENTITY_SLUG}-new-name"
    old_doc_dir.rename(new_doc_dir)

    mine_corpus(
        corpus_dir=corpus_dir,
        config=config,
        taxonomy=taxonomy,
        out_dir=out_dir,
        entity_registry_path=registry_path,
        no_cache=False,
    )

    assert not old_aliased_dir.exists(), (
        f"normalized/ must stale-clear the renamed document's OLD aliased "
        f"directory, mirroring trail/'s issue #51 treatment: {old_aliased_dir} "
        "still exists after the rename"
    )
    trees_2 = _tree_files(out_dir)
    assert len(trees_2) == 1, f"expected exactly one tree after the rename, got {trees_2}"
    _assert_no_raw_entity(trees_2)


# ---------------------------------------------------------------------------
# 3. corpus_manifest.json's alias matches the manifest's own document_id
#    (sanity: normalized/'s directory name and the manifest's document_id
#    must agree — proves _pseudonymize_clause_tree's document_id aliasing
#    uses the SAME alias assignment as _pseudonymize_corpus_documents, not
#    an independently-computed one that happens to look similar).
# ---------------------------------------------------------------------------


def test_normalized_tree_filename_and_metadata_aliased_when_version_stem_embeds_entity(
    tmp_path: Path,
) -> None:
    """Regression for issue #139 review round 3.

    Every other fixture in this module names its version files generically
    ("v1.rtf"), so none of them exercise the realistic case this codebase
    documents and handles everywhere else (``_alias_version_field``'s
    docstring, ``test_entity_registry.py``'s fixtures): a STAGED version
    filename whose stem embeds the raw counterparty name, e.g.
    "01__Affiliation Agreement - Example University 6.14.23". Before this
    fix, ``ClauseTree.version``/``source_file`` traveled into
    ``normalized/`` unaliased, AND the on-disk filename was built from the
    same raw, never-aliased dict key — so the raw name survived verbatim in
    both the output FILENAME and the JSON file CONTENT even though the
    parent directory (document_id) was correctly aliased.
    """
    corpus_dir = tmp_path / "corpus"
    doc_dir = corpus_dir / f"{ENTITY_SLUG}-delta"
    doc_dir.mkdir(parents=True)
    raw_stem = f"01__Affiliation Agreement - {ENTITY_NAME} 6.14.23"
    _write_rtf(doc_dir / f"{raw_stem}.rtf", _doc_body("delta"))

    config_path = _write_config(tmp_path)
    taxonomy = load_taxonomy(_TAXONOMY_PATH)
    config = load_config(config_path)
    out_dir = tmp_path / "out"

    mine_corpus(
        corpus_dir=corpus_dir,
        config=config,
        taxonomy=taxonomy,
        out_dir=out_dir,
        entity_registry_path=tmp_path / "registry.json",
        no_cache=True,
    )

    trees = _tree_files(out_dir)
    assert len(trees) == 1, f"expected exactly one normalized/ tree, got {trees}"
    tree_path = trees[0]

    # The FILENAME (not just the parent directory) must not carry the raw
    # entity name — this is the finding-1 regression.
    assert ENTITY_NAME not in tree_path.name, (
        f"RAW ENTITY NAME LEAK in normalized/ tree FILENAME: {tree_path.name}"
    )
    assert tree_path.name == "v1.clauses.json", (
        f"single-version document's tree must be written under the aliased "
        f"ordinal label 'v1', not the raw staged-filename stem: {tree_path.name!r}"
    )

    # The JSON CONTENT's "version"/"source_file" fields must not carry the
    # raw entity name either — this is the finding-2 regression.
    tree_dict = json.loads(tree_path.read_text(encoding="utf-8"))
    assert ENTITY_NAME not in tree_dict["version"], (
        f"RAW ENTITY NAME LEAK in normalized/ tree content field 'version': "
        f"{tree_dict['version']!r}"
    )
    assert tree_dict["version"] == "v1", (
        f"'version' must be the aliased ordinal label, not the raw stem: {tree_dict['version']!r}"
    )
    assert ENTITY_NAME not in tree_dict["source_file"], (
        f"RAW ENTITY NAME LEAK in normalized/ tree content field 'source_file': "
        f"{tree_dict['source_file']!r}"
    )


def test_normalized_tree_dirname_matches_manifest_document_id(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "corpus"
    doc_dir = corpus_dir / f"{ENTITY_SLUG}-gamma"
    doc_dir.mkdir(parents=True)
    _write_rtf(doc_dir / "v1.rtf", _doc_body("gamma"))

    config_path = _write_config(tmp_path)
    taxonomy = load_taxonomy(_TAXONOMY_PATH)
    config = load_config(config_path)
    out_dir = tmp_path / "out"

    mine_corpus(
        corpus_dir=corpus_dir,
        config=config,
        taxonomy=taxonomy,
        out_dir=out_dir,
        entity_registry_path=tmp_path / "registry.json",
        no_cache=True,
    )

    manifest = json.loads((out_dir / "corpus_manifest.json").read_text(encoding="utf-8"))
    assert len(manifest) == 1
    manifest_doc_id = manifest[0]["document_id"]

    trees = _tree_files(out_dir)
    assert len(trees) == 1
    assert trees[0].parent.name == manifest_doc_id, (
        f"normalized/ directory name {trees[0].parent.name!r} must match "
        f"corpus_manifest.json's document_id {manifest_doc_id!r} — both must "
        "be the SAME alias assignment"
    )
