"""Tests for entity_registry — deterministic pseudonymization at ingest (issue #153).

SECURITY NOTE: All fixtures use programmatically constructed RTF text with
synthetic, fictional content. "State University" here is a stand-in fictional
counterparty name, not any real institution.
"""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import yaml

from playbook_engine.config import load_config
from playbook_engine.entity_registry import (
    EntityRegistry,
    pseudonymize_document_id,
    pseudonymize_text,
    write_holdout_map,
)
from playbook_engine.pipeline import mine_corpus, project_playbook
from playbook_engine.taxonomy import load_taxonomy

_TAXONOMY_PATH = Path(__file__).parent.parent / "spec" / "taxonomy" / "affiliation-agreement.yaml"

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


_KNOWN_ENTITY = "State University"

_BODY_DEAL_1 = (
    r"1. Indemnification\par "
    rf"Alpha Corp shall indemnify {_KNOWN_ENTITY} against third-party claims "
    r"arising from the placement programme.\par "
    r"2. Governing Law\par "
    r"This agreement is governed by the laws of the State of California.\par "
)

_BODY_DEAL_2 = (
    r"1. Indemnification\par "
    rf"{_KNOWN_ENTITY} shall provide reasonable cooperation to Alpha Corp "
    r"in connection with any third-party claim.\par "
    r"2. Governing Law\par "
    r"This agreement is governed by the laws of the State of Delaware.\par "
)


def _make_corpus(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Two-document corpus, both mentioning ``_KNOWN_ENTITY`` in clause text.

    Returns (corpus_dir, config_path, out_dir).
    """
    corpus_dir = tmp_path / "corpus"
    (corpus_dir / "deal-001").mkdir(parents=True)
    (corpus_dir / "deal-002").mkdir(parents=True)
    _write_rtf(corpus_dir / "deal-001" / "v1.rtf", _BODY_DEAL_1)
    _write_rtf(corpus_dir / "deal-002" / "v1.rtf", _BODY_DEAL_2)

    cfg = {
        "agreement_type": {
            "id": "educational-affiliation",
            "name": "Educational Affiliation Agreement",
        },
        "baseline": {},
        "taxonomy": str(_TAXONOMY_PATH),
        "provenance": {
            "our_party_aliases": ["Alpha Corp"],
            "known_entities": [_KNOWN_ENTITY],
        },
    }
    config_path = tmp_path / "playbook.config.yaml"
    config_path.write_text(yaml.dump(cfg), encoding="utf-8")
    out_dir = tmp_path / "out"
    return corpus_dir, config_path, out_dir


# ---------------------------------------------------------------------------
# EntityRegistry unit tests
# ---------------------------------------------------------------------------


def test_alias_for_assigns_stable_alias_and_persists(tmp_path: Path) -> None:
    """A freshly assigned alias is written through to disk immediately, so a
    second EntityRegistry.load() against the same path sees the same alias
    without any explicit save() call.
    """
    reg_path = tmp_path / "entity_registry.json"
    reg1 = EntityRegistry.load(reg_path)
    alias1 = reg1.alias_for(_KNOWN_ENTITY)

    assert reg_path.exists(), "alias_for must write through to disk on first sight of a new name"

    reg2 = EntityRegistry.load(reg_path)
    alias2 = reg2.alias_for(_KNOWN_ENTITY)
    assert alias1 == alias2, "the same entity must get the same alias across two registry loads"


def test_alias_for_is_case_and_whitespace_insensitive(tmp_path: Path) -> None:
    reg = EntityRegistry.load(tmp_path / "entity_registry.json")
    a1 = reg.alias_for("State University")
    a2 = reg.alias_for("state   university")
    a3 = reg.alias_for("STATE UNIVERSITY")
    assert a1 == a2 == a3


def test_alias_for_distinct_entities_get_distinct_aliases(tmp_path: Path) -> None:
    reg = EntityRegistry.load(tmp_path / "entity_registry.json")
    a1 = reg.alias_for("State University")
    a2 = reg.alias_for("Beta Hospital")
    assert a1 != a2


def test_alias_map_reverses_to_canonical_entity_name(tmp_path: Path) -> None:
    reg = EntityRegistry.load(tmp_path / "entity_registry.json")
    alias = reg.alias_for(_KNOWN_ENTITY)
    assert reg.alias_map() == {alias: _KNOWN_ENTITY}


# ---------------------------------------------------------------------------
# pseudonymize_text / pseudonymize_document_id unit tests
# ---------------------------------------------------------------------------


def test_pseudonymize_text_replaces_whole_word_occurrences(tmp_path: Path) -> None:
    reg = EntityRegistry.load(tmp_path / "entity_registry.json")
    text = f"{_KNOWN_ENTITY} shall indemnify Alpha Corp."
    out = pseudonymize_text(text, [_KNOWN_ENTITY], reg)
    assert _KNOWN_ENTITY not in out
    assert reg.alias_for(_KNOWN_ENTITY) in out


def test_pseudonymize_text_does_not_corrupt_substring_words(tmp_path: Path) -> None:
    """A known entity name must only replace WHOLE-WORD occurrences — a longer
    word merely containing the known name as a substring (e.g. "State
    Universityville") must be left alone.
    """
    reg = EntityRegistry.load(tmp_path / "entity_registry.json")
    text = "The State Universityville Annex is unaffected."
    out = pseudonymize_text(text, [_KNOWN_ENTITY], reg)
    assert out == text


def test_pseudonymize_document_id_replaces_matching_slug_span(tmp_path: Path) -> None:
    reg = EntityRegistry.load(tmp_path / "entity_registry.json")
    alias = reg.alias_for(_KNOWN_ENTITY)
    doc_id = "state-university-2023"
    out = pseudonymize_document_id(doc_id, [_KNOWN_ENTITY], reg)
    assert _KNOWN_ENTITY.lower().replace(" ", "-") not in out
    assert out.endswith("2023")
    assert alias.lower() in out


def test_pseudonymize_document_id_unchanged_when_no_known_entity_present(tmp_path: Path) -> None:
    reg = EntityRegistry.load(tmp_path / "entity_registry.json")
    out = pseudonymize_document_id("deal-001", [_KNOWN_ENTITY], reg)
    assert out == "deal-001"


def test_write_holdout_map_is_access_restricted_and_reverses_alias(tmp_path: Path) -> None:
    reg = EntityRegistry.load(tmp_path / "entity_registry.json")
    alias = reg.alias_for(_KNOWN_ENTITY)
    holdout_path = tmp_path / "alias_map.json"
    write_holdout_map(holdout_path, reg)

    written = json.loads(holdout_path.read_text(encoding="utf-8"))
    assert written == {alias: _KNOWN_ENTITY}

    mode = stat.S_IMODE(os.stat(holdout_path).st_mode)
    assert mode == 0o600, f"held-out map must be owner-only (0600); got {oct(mode)}"


# ---------------------------------------------------------------------------
# Integration: mine_corpus -> project_playbook (issue #153's Required verification)
# ---------------------------------------------------------------------------


def test_compiled_artifact_carries_alias_not_raw_entity_name(tmp_path: Path) -> None:
    """The compiled playbook (and the observation store feeding it) must
    contain the known entity's alias, never the raw name — across two
    different documents in the same corpus.
    """
    corpus_dir, config_path, out_dir = _make_corpus(tmp_path)
    registry_path = tmp_path / "entity_registry.json"

    taxonomy = load_taxonomy(_TAXONOMY_PATH)
    config = load_config(config_path)

    mine_corpus(
        corpus_dir=corpus_dir,
        config=config,
        taxonomy=taxonomy,
        out_dir=out_dir,
        entity_registry_path=registry_path,
    )

    obs_text = (out_dir / "observations.jsonl").read_text(encoding="utf-8")
    assert _KNOWN_ENTITY not in obs_text, "the raw entity name must never reach observations.jsonl"

    manifest_text = (out_dir / "corpus_manifest.json").read_text(encoding="utf-8")
    assert _KNOWN_ENTITY not in manifest_text

    playbook = project_playbook(out_dir=out_dir, config=config, taxonomy=taxonomy)
    playbook_text = json.dumps(playbook)
    assert _KNOWN_ENTITY not in playbook_text, (
        "the compiled playbook.opf.json must never carry the raw entity name"
    )

    # Same alias used for BOTH documents (cross-document stability).
    observations = [json.loads(line) for line in obs_text.splitlines() if line.strip()]
    doc_ids = {o["citation"]["document_id"] for o in observations}
    assert len(doc_ids) == 2, f"expected 2 distinct document ids, got {doc_ids}"

    reg = EntityRegistry.load(registry_path)
    alias = reg.alias_for(_KNOWN_ENTITY)
    assert alias in obs_text
    assert all(
        alias in o["full_text"]
        for o in observations
        if "indemnif" in o["full_text"].lower() or "cooperation" in o["full_text"].lower()
    ), "every clause mentioning the known entity must carry the SAME alias"

    # Held-out map: written as a sidecar, NOT part of the OPF, and reverses correctly.
    holdout_path = out_dir / "alias_map.json"
    assert holdout_path.exists()
    holdout = json.loads(holdout_path.read_text(encoding="utf-8"))
    assert holdout[alias] == _KNOWN_ENTITY
    assert "alias_map" not in playbook_text
    mode = stat.S_IMODE(os.stat(holdout_path).st_mode)
    assert mode == 0o600


def test_alias_stable_across_two_mine_corpus_runs(tmp_path: Path) -> None:
    """Re-running mine_corpus against a fresh out_dir but the SAME entity
    registry path must assign the identical alias to the same entity name.
    """
    corpus_dir, config_path, _ = _make_corpus(tmp_path)
    registry_path = tmp_path / "entity_registry.json"

    taxonomy = load_taxonomy(_TAXONOMY_PATH)
    config = load_config(config_path)

    out_dir_1 = tmp_path / "out1"
    mine_corpus(
        corpus_dir=corpus_dir,
        config=config,
        taxonomy=taxonomy,
        out_dir=out_dir_1,
        entity_registry_path=registry_path,
    )
    reg_after_run_1 = EntityRegistry.load(registry_path)
    alias_after_run_1 = reg_after_run_1.alias_for(_KNOWN_ENTITY)

    out_dir_2 = tmp_path / "out2"
    mine_corpus(
        corpus_dir=corpus_dir,
        config=config,
        taxonomy=taxonomy,
        out_dir=out_dir_2,
        entity_registry_path=registry_path,
    )
    reg_after_run_2 = EntityRegistry.load(registry_path)
    alias_after_run_2 = reg_after_run_2.alias_for(_KNOWN_ENTITY)

    assert alias_after_run_1 == alias_after_run_2, (
        "the same entity must get the same alias across two separate mine_corpus runs "
        "sharing the same entity_registry_path"
    )

    obs_2_text = (out_dir_2 / "observations.jsonl").read_text(encoding="utf-8")
    assert alias_after_run_1 in obs_2_text
    assert _KNOWN_ENTITY not in obs_2_text


def test_no_known_entities_configured_is_a_no_op(tmp_path: Path) -> None:
    """When provenance.known_entities is empty (today's default), no registry
    file is created and no held-out map is written — pure backward compat.
    """
    corpus_dir = tmp_path / "corpus"
    (corpus_dir / "deal-001").mkdir(parents=True)
    _write_rtf(corpus_dir / "deal-001" / "v1.rtf", _BODY_DEAL_1)

    cfg = {
        "agreement_type": {
            "id": "educational-affiliation",
            "name": "Educational Affiliation Agreement",
        },
        "baseline": {},
        "taxonomy": str(_TAXONOMY_PATH),
        "provenance": {"our_party_aliases": ["Alpha Corp"]},
    }
    config_path = tmp_path / "playbook.config.yaml"
    config_path.write_text(yaml.dump(cfg), encoding="utf-8")
    out_dir = tmp_path / "out"
    registry_path = tmp_path / "entity_registry.json"

    taxonomy = load_taxonomy(_TAXONOMY_PATH)
    config = load_config(config_path)
    assert config.provenance.known_entities == []

    mine_corpus(
        corpus_dir=corpus_dir,
        config=config,
        taxonomy=taxonomy,
        out_dir=out_dir,
        entity_registry_path=registry_path,
    )

    assert not registry_path.exists(), "no registry file should be touched with no known_entities"
    assert not (out_dir / "alias_map.json").exists(), "no held-out map should be written"
    obs_text = (out_dir / "observations.jsonl").read_text(encoding="utf-8")
    assert _KNOWN_ENTITY in obs_text, (
        "with no known_entities configured, clause text must pass through unchanged"
    )


# ---------------------------------------------------------------------------
# Born-safe id/version consistency (issue #182)
# ---------------------------------------------------------------------------


def test_alias_version_field_aliases_filename_stem(tmp_path: Path) -> None:
    """version/version_id filename stems embedding an entity name are aliased."""
    from playbook_engine.pipeline import _alias_version_field

    reg = EntityRegistry.load(tmp_path / "reg.json")  # empty registry
    known = ["Oglethorpe University"]
    out = _alias_version_field(
        "01__API-Internship Agreement - Oglethorpe University 6.14.23", known, reg
    )
    assert "Oglethorpe" not in out
    assert reg.alias_for("Oglethorpe University") in out
    # A plain ordinal carries no name and passes through untouched.
    assert _alias_version_field(3, known, reg) == 3


def test_pseudonymize_observation_id_aliases_doc_segment(tmp_path: Path) -> None:
    """observation_id's document segment is aliased; version/clause path preserved."""
    from playbook_engine.pipeline import _pseudonymize_observation_id

    reg = EntityRegistry.load(tmp_path / "reg.json")
    known = ["Oglethorpe University"]
    oid = "API-Internship-Agreement-Oglethorpe-University-6.14.23_ff1a5b80/3/0"
    out = _pseudonymize_observation_id(oid, known, reg)
    assert "oglethorpe" not in out.lower()
    assert out.endswith("/3/0")  # version/clause structure intact


def test_pseudonymize_trail_aliases_document_id(tmp_path: Path) -> None:
    """Trail document_id is aliased so it joins the pseudonymized observations."""
    from playbook_engine.pipeline import _pseudonymize_trail

    reg = EntityRegistry.load(tmp_path / "reg.json")
    known = ["Oglethorpe University"]
    trail = {"document_id": "API-Internship-Agreement-Oglethorpe-University-6.14.23_ff1a5b80"}
    out = _pseudonymize_trail(trail, known, reg)
    assert "oglethorpe" not in out["document_id"].lower()
    # Matches the aliased citation.document_id form pseudonymize_document_id emits.
    assert out["document_id"] == pseudonymize_document_id(trail["document_id"], known, reg)
