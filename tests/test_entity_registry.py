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


def test_pseudonymize_text_matches_name_ending_in_punctuation(tmp_path: Path) -> None:
    """A known entity name ending in punctuation (e.g. "Acme Corp.") must
    still be matched — a trailing ``\\b`` is unsatisfiable at a period-to-
    space edge, so pseudonymize_text must use non-word-char lookarounds
    instead (issue #245).
    """
    reg = EntityRegistry.load(tmp_path / "entity_registry.json")
    text = "Between Acme Corp. and the school"
    out = pseudonymize_text(text, ["Acme Corp."], reg)
    assert "Acme Corp." not in out
    assert reg.alias_for("Acme Corp.") in out

    # Substring protection at non-word edges must still hold: "CUNY" inside
    # "CUNYA" is not a known-name-then-non-word-boundary match.
    reg2 = EntityRegistry.load(tmp_path / "entity_registry2.json")
    out2 = pseudonymize_text("CUNYA is unaffected.", ["CUNY"], reg2)
    assert out2 == "CUNYA is unaffected."


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


def test_pseudonymize_trail_aliases_version_order_and_reversals(tmp_path: Path) -> None:
    """Issue #34: version_order fields AND reversals must be aliased too.

    _pseudonymize_trail previously aliased only ``document_id``, but the
    trail dict also spreads in ``version_order.to_dict()``
    (``ordered_versions``, ``signed_version``, ``pairwise_distances[].from``/
    ``to`` — all staged filename stems) and carries ``reversals`` entries
    (``version_inserted``/``version_removed`` — same raw stems — plus raw
    ``proposed_text`` clause text). This unit-level check exercises every one
    of those fields directly against ``_pseudonymize_trail``.
    """
    from playbook_engine.pipeline import _pseudonymize_trail

    reg = EntityRegistry.load(tmp_path / "reg.json")
    known = ["Oglethorpe University"]
    v1 = "01__Affiliation Agreement - Oglethorpe University 1.1.24"
    v2 = "02__Affiliation Agreement - Oglethorpe University 1.5.24"
    v3 = "03__Affiliation Agreement - Oglethorpe University 1.10.24"
    trail = {
        "document_id": "deal-001",
        "ordered_versions": [v1, v2, v3],
        "signed_version": v3,
        "basis": "hints",
        "total_distance": 1.5,
        "pairwise_distances": [
            {"from": v1, "to": v2, "distance": 0.5},
            {"from": v2, "to": v3, "distance": 1.0},
        ],
        "shape": "linear",
        "reversals": [
            {
                "taxonomy_id": "ind",
                "clause_path": "1",
                "version_inserted": v2,
                "version_removed": v3,
                "proposed_text": "Oglethorpe University shall retain audit rights.",
                "char_span": [0, 10],
            }
        ],
    }

    out = _pseudonymize_trail(trail, known, reg)
    serialized = json.dumps(out).lower()
    assert "oglethorpe" not in serialized, f"raw counterparty name leaked into trail: {out}"

    # Structure preserved: same number of entries, same non-name fields intact.
    assert len(out["ordered_versions"]) == 3
    assert out["signed_version"] == out["ordered_versions"][2]
    assert len(out["pairwise_distances"]) == 2
    assert out["pairwise_distances"][0]["distance"] == 0.5
    assert out["pairwise_distances"][1]["from"] == out["ordered_versions"][1]
    assert out["pairwise_distances"][1]["to"] == out["ordered_versions"][2]
    assert len(out["reversals"]) == 1
    rev = out["reversals"][0]
    assert rev["version_inserted"] == out["ordered_versions"][1]
    assert rev["version_removed"] == out["ordered_versions"][2]
    assert rev["taxonomy_id"] == "ind"  # non-name field untouched
    assert rev["char_span"] == [0, 10]


# ---------------------------------------------------------------------------
# Issue #34 (end-to-end): a mined corpus whose version filename stems AND
# whose reversed clause text carry a known_entities name must leave no trace
# of that name in any trail/*.json body.
# ---------------------------------------------------------------------------

_REV_BODY_V1 = (
    r"1. Indemnification\par "
    rf"Alpha Corp shall indemnify {_KNOWN_ENTITY} against third-party claims "
    r"arising from the placement programme.\par "
    r"2. Governing Law\par "
    r"This agreement is governed by the laws of the State of California.\par "
    r"3. Term\par "
    r"This agreement commences on the date of execution and continues for one year.\par "
)

# v2 bundles two changes: a short audit-rights sentence proposed into the
# Indemnification clause (later reversed) AND a substantially larger rewrite
# of Governing Law + Term (which survives to signing). The larger surviving
# rewrite is what the content-based orderer needs (v1<->v2 and v2<->v3 must
# each individually cost less than v1<->v3) to naturally infer the
# chronological chain v1 -> v2 -> v3 rather than short-circuiting through
# the smaller v1<->v3 edit distance — see issue #34 verification notes.
_REV_BODY_V2 = (
    r"1. Indemnification\par "
    rf"Alpha Corp shall indemnify {_KNOWN_ENTITY} against third-party claims "
    rf"arising from the placement programme. {_KNOWN_ENTITY} shall retain the "
    r"right to audit all placement records.\par "
    r"2. Governing Law\par "
    r"This agreement is governed by the laws of the State of Delaware, with "
    r"additional dispute resolution provisions including mandatory binding "
    r"arbitration administered by a neutral third-party arbitration body and "
    r"a three year audit window for financial records retention.\par "
    r"3. Term\par "
    r"This agreement commences on the date of execution and continues for "
    r"eighteen months, with a renewal option subject to mutual written "
    r"consent of both parties and thirty days advance written notice.\par "
)

# v3 (signed) keeps v2's Governing Law/Term rewrite but drops the
# audit-rights sentence — the planted-insert-then-revert pattern
# detect_reversals is built to catch.
_REV_BODY_V3 = (
    r"1. Indemnification\par "
    rf"Alpha Corp shall indemnify {_KNOWN_ENTITY} against third-party claims "
    r"arising from the placement programme.\par "
    r"2. Governing Law\par "
    r"This agreement is governed by the laws of the State of Delaware, with "
    r"additional dispute resolution provisions including mandatory binding "
    r"arbitration administered by a neutral third-party arbitration body and "
    r"a three year audit window for financial records retention.\par "
    r"3. Term\par "
    r"This agreement commences on the date of execution and continues for "
    r"eighteen months, with a renewal option subject to mutual written "
    r"consent of both parties and thirty days advance written notice.\par "
)


def test_pseudonymize_trail_end_to_end_no_raw_name_survives_mining(tmp_path: Path) -> None:
    """Mine a corpus with reversals whose version stems carry a known entity
    name; assert the raw name appears nowhere in any written trail/*.json.

    Regression guard for issue #34: pre-fix, ``ordered_versions``,
    ``signed_version``, ``pairwise_distances[].from``/``to``, and
    ``reversals[].version_inserted``/``version_removed``/``proposed_text``
    all carried the raw counterparty name straight through to the trail file
    even with ``known_entities`` configured.
    """
    corpus_dir = tmp_path / "corpus"
    deal_dir = corpus_dir / "deal-001"
    deal_dir.mkdir(parents=True)

    v1_name = f"01__Affiliation Agreement - {_KNOWN_ENTITY} 1.1.24.rtf"
    v2_name = f"02__Affiliation Agreement - {_KNOWN_ENTITY} 1.5.24.rtf"
    v3_name = f"03__Affiliation Agreement - {_KNOWN_ENTITY} 1.10.24.rtf"
    _write_rtf(deal_dir / v1_name, _REV_BODY_V1)
    _write_rtf(deal_dir / v2_name, _REV_BODY_V2)
    _write_rtf(deal_dir / v3_name, _REV_BODY_V3)

    # Pin signed_version explicitly (docs/CORPUS-LAYOUT.md's documented
    # hints.yaml form, entries WITH extensions) — a hard override, so the
    # signed anchor is deterministic. The chronological chain order itself
    # falls out of content-based inference alone (v2's Governing Law/Term
    # rewrite is deliberately the largest edit so v1<->v2 and v2<->v3 both
    # cost less than the v1<->v3 shortcut — see _REV_BODY_V2 comment).
    (deal_dir / "hints.yaml").write_text(
        yaml.dump({"signed_version": v3_name}),
        encoding="utf-8",
    )

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

    taxonomy = load_taxonomy(_TAXONOMY_PATH)
    config = load_config(config_path)

    mine_corpus(
        corpus_dir=corpus_dir,
        config=config,
        taxonomy=taxonomy,
        out_dir=out_dir,
    )

    trail_dir = out_dir / "trail"
    trail_files = list(trail_dir.glob("*.json"))
    assert trail_files, "mine_corpus must have written at least one trail file"

    found_reversal = False
    for trail_path in trail_files:
        raw_text = trail_path.read_text(encoding="utf-8")
        assert _KNOWN_ENTITY.lower() not in raw_text.lower(), (
            f"raw counterparty name leaked into {trail_path.name}: {raw_text}"
        )
        trail = json.loads(raw_text)
        if trail.get("reversals"):
            found_reversal = True
            for rev in trail["reversals"]:
                assert _KNOWN_ENTITY.lower() not in rev["version_inserted"].lower()
                assert _KNOWN_ENTITY.lower() not in rev["version_removed"].lower()
                assert _KNOWN_ENTITY.lower() not in rev["proposed_text"].lower()
        for v in trail.get("ordered_versions", []):
            assert _KNOWN_ENTITY.lower() not in v.lower()
        if trail.get("signed_version"):
            assert _KNOWN_ENTITY.lower() not in trail["signed_version"].lower()
        for pd in trail.get("pairwise_distances", []):
            assert _KNOWN_ENTITY.lower() not in pd["from"].lower()
            assert _KNOWN_ENTITY.lower() not in pd["to"].lower()

    assert found_reversal, (
        "fixture must actually produce a reversal (planted insert-then-revert) "
        "for this test to exercise the reversals-pseudonymization path"
    )
