"""Pipeline-level tests for the mine/project split (issue #63).

Verifies two acceptance criteria:
  AC-1: project_playbook performs NO _ingest_file and NO classify_tree/judge
        calls even when a template is configured.
  AC-2: Re-running project_playbook does not rewrite observations.jsonl
        (no re-mining side-effect).

SECURITY NOTE: All fixtures use programmatically constructed RTF text with
synthetic, fictional content.  No real agreement files are referenced.
Fictional party names only (e.g. "Alpha Corp", "Beta University").
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml
from docx import Document
from lxml import etree

from playbook_engine.canonicalize import compute_section_digests, content_hash
from playbook_engine.config import load_config
from playbook_engine.deviation_classifier import DeviationResult, RiskDelta
from playbook_engine.floor_candidates import sign_floor_invariant
from playbook_engine.observation_builder import Observation, ObservationCitation
from playbook_engine.pipeline import _NullDeviationJudge, mine_corpus, project_playbook
from playbook_engine.posture import apply_posture_interview
from playbook_engine.taxonomy import load_taxonomy

_W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def _w(tag: str) -> str:
    return f"{{{_W_NS}}}{tag}"


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

_CORPUS_BODY = (
    r"1. Indemnification\par "
    r"Alpha Corp shall indemnify Beta University against third-party claims "
    r"arising from the placement programme.\par "
    r"2. Governing Law\par "
    r"This agreement is governed by the laws of the State of California.\par "
    r"3. Term\par "
    r"This agreement commences on the date of execution and continues for one year.\par "
)

_TEMPLATE_BODY = (
    r"1. Indemnification\par "
    r"The service provider shall indemnify the institution against third-party claims.\par "
    r"2. Governing Law\par "
    r"This agreement is governed by the laws of the State of New York.\par "
    r"3. Term\par "
    r"Initial term of one year with automatic renewal.\par "
)


# ---------------------------------------------------------------------------
# Corpus + config factory
# ---------------------------------------------------------------------------


def _make_corpus_with_template(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    """Build a synthetic corpus + config WITH a template; return
    (corpus_dir, config_path, out_dir, template_path).
    """
    corpus_dir = tmp_path / "corpus"
    deal_dir = corpus_dir / "deal-001"
    deal_dir.mkdir(parents=True)
    _write_rtf(deal_dir / "v1.rtf", _CORPUS_BODY)

    template_dir = tmp_path / "template"
    template_dir.mkdir()
    template_path = template_dir / "template.rtf"
    _write_rtf(template_path, _TEMPLATE_BODY)

    cfg = {
        "agreement_type": {
            "id": "educational-affiliation",
            "name": "Educational Affiliation Agreement",
            "aliases": ["eiaa"],
        },
        "baseline": {"template": str(template_path)},
        "taxonomy": str(_TAXONOMY_PATH),
        "provenance": {"our_party_aliases": ["Alpha Corp"]},
    }
    config_path = tmp_path / "playbook.config.yaml"
    config_path.write_text(yaml.dump(cfg), encoding="utf-8")

    out_dir = tmp_path / "out"
    return corpus_dir, config_path, out_dir, template_path


class _FakeDeviationJudge:
    """Deterministic fake ``DeviationJudge`` — always returns a fixed
    substantive/worse/material verdict and records every batch it is asked
    to assess, so a test can assert it was actually invoked (as opposed to
    a clause being silently hardcoded deviation="none" without ever reaching
    the judge — issue #103)."""

    def __init__(self) -> None:
        self.received_items: list[dict[str, str]] = []

    def assess_batch(self, items: list[dict[str, str]], our_standard: str) -> list[DeviationResult]:
        self.received_items.extend(items)
        return [
            DeviationResult(
                deviation="substantive",
                risk_delta=RiskDelta(direction="worse", magnitude="material"),
                basis="judge",
                rationale="Fake judge: clause differs materially from the template.",
            )
            for _ in items
        ]


# ---------------------------------------------------------------------------
# AC-1: project_playbook makes no _ingest_file and no classify_tree calls
# ---------------------------------------------------------------------------


def test_project_playbook_no_ingest_no_judge_with_template(tmp_path: Path) -> None:
    """project_playbook must not call _ingest_file or classify_tree, even
    when a template is configured in the engine config.

    Regression guard for issue #63: previously project_playbook re-ingested
    the template via _ingest_file and re-classified it via classify_tree,
    violating the 'zero ingest / LLM work' criterion.
    """
    corpus_dir, config_path, out_dir, _template_path = _make_corpus_with_template(tmp_path)

    taxonomy = load_taxonomy(_TAXONOMY_PATH)
    cfg = load_config(config_path)

    # Step 1: mine — real run, writes template_observations.jsonl to the store.
    mine_corpus(
        corpus_dir=corpus_dir,
        config=cfg,
        taxonomy=taxonomy,
        out_dir=out_dir,
    )
    assert (out_dir / "template_observations.jsonl").exists(), (
        "mine_corpus must write template_observations.jsonl"
    )

    # Step 2: project — spy on _ingest_file and classify_tree; assert neither is called.
    ingest_spy = MagicMock(side_effect=AssertionError("_ingest_file must not be called by project"))
    classify_spy = MagicMock(
        side_effect=AssertionError("classify_tree must not be called by project")
    )

    with (
        patch("playbook_engine.pipeline._ingest_file", ingest_spy),
        patch("playbook_engine.pipeline.classify_tree", classify_spy),
    ):
        playbook = project_playbook(
            out_dir=out_dir,
            config=cfg,
            taxonomy=taxonomy,
        )

    # Spies must not have been called at all.
    ingest_spy.assert_not_called()
    classify_spy.assert_not_called()

    # And the playbook must still be produced.
    assert isinstance(playbook, dict)
    assert (out_dir / "playbook.opf.json").exists()


def test_compiled_agreement_type_carries_id_and_aliases(tmp_path: Path) -> None:
    """Issue #142: agreement_type is the shared cross-tool key. The compiler
    must pass ``config.agreement_type.aliases`` straight through into the
    assembled playbook alongside ``id``, so a consumer (whose own registry
    key is 'eiaa') can match on either without a hand-joined
    mapping table."""
    corpus_dir, config_path, out_dir, _template_path = _make_corpus_with_template(tmp_path)

    taxonomy = load_taxonomy(_TAXONOMY_PATH)
    cfg = load_config(config_path)
    assert cfg.agreement_type.aliases == ["eiaa"]

    mine_corpus(corpus_dir=corpus_dir, config=cfg, taxonomy=taxonomy, out_dir=out_dir)
    playbook = project_playbook(out_dir=out_dir, config=cfg, taxonomy=taxonomy)

    assert playbook["agreement_type"]["id"] == "educational-affiliation"
    assert playbook["agreement_type"]["aliases"] == ["eiaa"]

    on_disk = json.loads((out_dir / "playbook.opf.json").read_text(encoding="utf-8"))
    assert on_disk["agreement_type"]["aliases"] == ["eiaa"]


# ---------------------------------------------------------------------------
# AC-2: Re-running project does not rewrite observations.jsonl
# ---------------------------------------------------------------------------


def test_project_playbook_does_not_rewrite_observations(tmp_path: Path) -> None:
    """Re-running project_playbook must not touch observations.jsonl.

    Ensures that the project step is purely read-only with respect to the
    observation store written by mine_corpus.
    """
    corpus_dir, config_path, out_dir, _template_path = _make_corpus_with_template(tmp_path)

    taxonomy = load_taxonomy(_TAXONOMY_PATH)
    cfg = load_config(config_path)

    # Mine first.
    mine_corpus(
        corpus_dir=corpus_dir,
        config=cfg,
        taxonomy=taxonomy,
        out_dir=out_dir,
    )

    obs_path = out_dir / "observations.jsonl"
    mtime_after_mine = obs_path.stat().st_mtime

    # Project — must not overwrite observations.jsonl.
    project_playbook(
        out_dir=out_dir,
        config=cfg,
        taxonomy=taxonomy,
    )

    mtime_after_project = obs_path.stat().st_mtime
    assert mtime_after_project == mtime_after_mine, (
        "project_playbook must not rewrite observations.jsonl"
    )


# ---------------------------------------------------------------------------
# Issue #101: a default (fully zero-LLM) compile must watermark its output.
# ---------------------------------------------------------------------------

_CORPUS_BODY_V2 = (
    r"1. Indemnification\par "
    r"Alpha Corp shall indemnify Beta University against third-party claims "
    r"arising from the placement programme.\par "
    r"2. Governing Law\par "
    r"This agreement is governed by the laws of the State of Delaware.\par "
    r"3. Term\par "
    r"This agreement commences on the date of execution and continues for one year.\par "
)


def test_default_stub_judges_compile_watermarks_playbook(tmp_path: Path) -> None:
    """End-to-end regression guard for issue #101.

    A real ``playbook mine`` + ``playbook project`` run with NO judges
    configured (the CLI default) uses ``_AllInScopeJudge`` (scope) and
    ``_NullDeviationJudge`` (deviation)
    — both stubs, neither backed by an LLM. Neither stub ever emits an
    Observation with basis="stub" literally: ``_AllInScopeJudge``'s
    basis="stub" lands on the ``ScopeDecision`` (scope.json), not an
    Observation, and ``_NullDeviationJudge`` emits basis="needs_review" (a
    judge protocol IS wired for deviation, it's just the stub default).
    Before the fix, neither signal reached ``assemble_playbook``, so a
    default compile's ``compiler.stub_basis_present`` was always False — the
    exact liability scenario issue #101 exists to catch (a stub-derived
    playbook a consumer cannot tell apart from a real one). This test
    drives the actual ``mine_corpus`` -> ``project_playbook`` path with no
    judges passed at all and asserts the watermark now fires.
    """
    corpus_dir = tmp_path / "corpus"
    deal_dir = corpus_dir / "deal-001"
    deal_dir.mkdir(parents=True)
    # Two versions with a real clause-text change (governing law: California
    # -> Delaware) so the deviation classifier actually dispatches to
    # _NullDeviationJudge for a changed clause — a single-version document
    # never calls the deviation judge at all (see
    # _observations_from_single_version), so this must be multi-version.
    _write_rtf(deal_dir / "v1.rtf", _CORPUS_BODY)
    _write_rtf(deal_dir / "v2.rtf", _CORPUS_BODY_V2)

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

    taxonomy = load_taxonomy(_TAXONOMY_PATH)
    config = load_config(config_path)

    # No scope_judge / deviation_judge passed -> the pipeline's zero-LLM
    # defaults apply (_AllInScopeJudge, _NullDeviationJudge).
    mine_corpus(
        corpus_dir=corpus_dir,
        config=config,
        taxonomy=taxonomy,
        out_dir=out_dir,
    )

    # Sanity check the premise: the stub scope decision really did land on
    # scope.json (basis="stub"), not on any Observation.
    scope = json.loads((out_dir / "scope.json").read_text(encoding="utf-8"))
    assert scope["documents"][0]["basis"] == "stub"
    obs_lines = (out_dir / "observations.jsonl").read_text(encoding="utf-8").splitlines()
    observations = [json.loads(line) for line in obs_lines if line.strip()]
    assert all(obs.get("basis") != "stub" for obs in observations), (
        "no Observation should ever carry basis='stub' literally — the stub "
        "signal from _NullDeviationJudge is basis='needs_review'"
    )

    playbook = project_playbook(
        out_dir=out_dir,
        config=config,
        taxonomy=taxonomy,
    )

    assert playbook["compiler"]["stub_basis_present"] is True, (
        "a default (zero-LLM) compile must watermark its output so a "
        "consuming review application can refuse to run against it"
    )


# ---------------------------------------------------------------------------
# Issue #82: an empty ClauseTree from a non-empty file must be recorded as a
# per-version ingest failure, never fed to the scope gate as first_tree.
# ---------------------------------------------------------------------------


def test_empty_tree_ingest_is_recorded_failure_not_out_of_scope(tmp_path: Path) -> None:
    """A version whose ingest yields an EMPTY ClauseTree from a non-empty
    source file (e.g. a scanned/image PDF with no OCR wired on the
    deterministic path) must be treated as an ingest failure, not silently
    fed to the scope gate as ``first_tree``.

    Regression guard for issue #82: previously the empty tree entered
    ``version_trees`` as if ingestion had succeeded. If it was the
    alphabetically-first version, ``scope_gate`` saw zero clause nodes and
    classified the *entire agreement* out-of-scope with
    ``basis="deterministic_empty"`` — one unreadable version taking down an
    otherwise-valid negotiation trail.

    "v1.rtf" here is a well-formed but body-less RTF file (76 bytes of RTF
    control words, zero paragraph text) — deterministically produces an
    empty ClauseTree without any OCR/LLM mocking. "v2.rtf" carries real
    clause content and is alphabetically second, so before the fix it would
    never be reached: ``first_tree`` would already have been the empty v1.
    """
    corpus_dir = tmp_path / "corpus"
    deal_dir = corpus_dir / "deal-001"
    deal_dir.mkdir(parents=True)

    # v1: well-formed RTF, but no body text at all -> ingests to an empty tree.
    _write_rtf(deal_dir / "v1.rtf", "")
    assert (deal_dir / "v1.rtf").stat().st_size > 0, "fixture file must be non-empty"

    # v2: real clause content -> ingests to a valid, in-scope-able tree.
    _write_rtf(deal_dir / "v2.rtf", _CORPUS_BODY)

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

    taxonomy = load_taxonomy(_TAXONOMY_PATH)
    config = load_config(config_path)

    progress_lines: list[str] = []
    mine_corpus(
        corpus_dir=corpus_dir,
        config=config,
        taxonomy=taxonomy,
        out_dir=out_dir,
        progress=progress_lines.append,
    )

    # The empty version must have produced a per-version WARNING, not a silent skip.
    assert any("v1.rtf" in line and "WARNING" in line for line in progress_lines), (
        f"expected a per-version WARNING for the empty-tree version v1.rtf; got: {progress_lines}"
    )

    # normalized/deal-001/v1.clauses.json must NOT have been written (ingest failure,
    # not success) -- only v2 (the valid version) reaches the normalized-tree write.
    normalized_dir = out_dir / "normalized" / "deal-001"
    assert not (normalized_dir / "v1.clauses.json").exists(), (
        "an empty-tree ingest failure must not be written as a normalized tree"
    )
    assert (normalized_dir / "v2.clauses.json").exists(), (
        "the valid version v2 must still be ingested and normalized"
    )

    # The scope gate must have evaluated v2 (real content), not v1 (empty) --
    # so the document must NOT be marked out-of-scope as "deterministic_empty".
    scope = json.loads((out_dir / "scope.json").read_text(encoding="utf-8"))
    doc_entries = {d["document_id"]: d for d in scope["documents"]}
    assert "deal-001" in doc_entries, "deal-001 must have a scope decision recorded"
    decision = doc_entries["deal-001"]
    assert decision["basis"] != "deterministic_empty", (
        "the empty-tree version must not have reached the scope gate as first_tree: "
        f"got basis={decision['basis']!r}"
    )
    assert decision["in_scope"] is True, (
        f"deal-001 has a valid version (v2) with real clause content and should remain "
        f"in-scope; got decision={decision}"
    )


# ---------------------------------------------------------------------------
# Issue #89: a failed per-version ingest must be a durable manifest record,
# not just a scrolled-past progress-line WARNING, and must surface as a
# Needs-Attention item in the after-action report.
# ---------------------------------------------------------------------------


def test_failed_version_ingest_recorded_in_manifest_and_needs_attention(
    tmp_path: Path,
) -> None:
    """A document with one failing version file and one good version file must:

    - record ``versions_mined`` < ``versions_found`` in corpus_manifest.json
      (not the old ``versions == len(version_files)`` that counted files
      found rather than versions actually mined).
    - carry a per-version ``version_ingest`` entry for the failed version
      with ``status="failed"`` and a non-empty ``error`` string.
    - surface that failure as a Needs Attention item in the after-action
      report, not just a console WARNING that a cache hit wouldn't re-print.
    """
    corpus_dir = tmp_path / "corpus"
    deal_dir = corpus_dir / "deal-001"
    deal_dir.mkdir(parents=True)

    # v1: well-formed RTF, no body text -> ingest failure (empty clause tree
    # from a non-empty source file; same fixture shape as issue #82's test).
    _write_rtf(deal_dir / "v1.rtf", "")
    assert (deal_dir / "v1.rtf").stat().st_size > 0, "fixture file must be non-empty"

    # v2: real clause content -> ingests successfully.
    _write_rtf(deal_dir / "v2.rtf", _CORPUS_BODY)

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

    taxonomy = load_taxonomy(_TAXONOMY_PATH)
    config = load_config(config_path)

    mine_corpus(
        corpus_dir=corpus_dir,
        config=config,
        taxonomy=taxonomy,
        out_dir=out_dir,
        progress=lambda _: None,
    )

    manifest = json.loads((out_dir / "corpus_manifest.json").read_text(encoding="utf-8"))
    doc_entries = {d["document_id"]: d for d in manifest}
    assert "deal-001" in doc_entries
    entry = doc_entries["deal-001"]

    assert entry["versions_found"] == 2, (
        f"expected 2 version files found (v1.rtf, v2.rtf); got {entry['versions_found']}"
    )
    assert entry["versions_mined"] == 1, (
        f"expected only v2 to have been mined (v1 failed); got {entry['versions_mined']}"
    )
    assert entry["versions_mined"] < entry["versions_found"], (
        "versions_mined must be strictly less than versions_found when a version fails"
    )
    # Back-compat alias: "versions" now means versions MINED, not files found.
    assert entry["versions"] == entry["versions_mined"]

    ingest_by_version = {v["version"]: v for v in entry["version_ingest"]}
    assert set(ingest_by_version) == {"v1", "v2"}
    assert ingest_by_version["v1"]["status"] == "failed"
    assert ingest_by_version["v1"]["error"], "failed version must carry a non-empty error string"
    assert ingest_by_version["v1"]["extractor"] == "rtf"
    assert ingest_by_version["v2"]["status"] == "ok"

    # The failure must surface in the after-action report's Needs Attention
    # section, not just as a console WARNING.
    from playbook_engine.aar import build_after_action_data

    aar_data = build_after_action_data(out_dir)
    needs_attention = aar_data["needs_attention"]
    matches = [
        item
        for item in needs_attention
        if item["document_id"] == "deal-001" and item["version"] == "v1"
    ]
    assert matches, f"expected a Needs-Attention item for deal-001/v1; got {needs_attention}"
    assert any("version ingest failed" in r for r in matches[0]["reasons"])


# ---------------------------------------------------------------------------
# A document whose EVERY version fails ingest must be recorded durably in
# quarantine.json — previously the per-doc pass returned None and the
# document vanished from every artifact (no corpus_manifest.json entry, no
# scope.json entry, no quarantine.json entry), so `validate` passed on a
# silently thinner playbook.
# ---------------------------------------------------------------------------


def test_all_versions_failed_document_is_quarantined(tmp_path: Path) -> None:
    """A document with no successfully ingested version lands in
    quarantine.json (and hence the AAR's Needs Attention section), not in
    the void."""
    corpus_dir = tmp_path / "corpus"

    # deal-001: its ONLY version is a well-formed RTF with no body text ->
    # empty clause tree from a non-empty source file -> ingest failure for
    # every version (issue #82 fixture shape).
    deal1_dir = corpus_dir / "deal-001"
    deal1_dir.mkdir(parents=True)
    _write_rtf(deal1_dir / "v1.rtf", "")
    assert (deal1_dir / "v1.rtf").stat().st_size > 0, "fixture file must be non-empty"

    # deal-002: mines fine, so the run produces a normal manifest alongside
    # the quarantined document.
    deal2_dir = corpus_dir / "deal-002"
    deal2_dir.mkdir()
    _write_rtf(deal2_dir / "v1.rtf", _CORPUS_BODY)

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

    taxonomy = load_taxonomy(_TAXONOMY_PATH)
    config = load_config(config_path)

    mine_corpus(
        corpus_dir=corpus_dir,
        config=config,
        taxonomy=taxonomy,
        out_dir=out_dir,
        progress=lambda _: None,
    )

    quarantine_path = out_dir / "quarantine.json"
    assert quarantine_path.exists(), (
        "an all-versions-failed document must be recorded in quarantine.json"
    )
    quarantine = json.loads(quarantine_path.read_text(encoding="utf-8"))
    entries = {q["document_id"]: q for q in quarantine}
    assert "deal-001" in entries
    assert "all versions failed" in entries["deal-001"]["reason"]

    # The quarantined document contributes nothing to the manifest — the
    # quarantine record is its ONLY durable trace.
    manifest = json.loads((out_dir / "corpus_manifest.json").read_text(encoding="utf-8"))
    assert "deal-001" not in {d["document_id"] for d in manifest}
    assert "deal-002" in {d["document_id"] for d in manifest}

    # And the after-action report surfaces it as a Needs-Attention item.
    from playbook_engine.aar import build_after_action_data

    aar_data = build_after_action_data(out_dir)
    matches = [i for i in aar_data["needs_attention"] if i["document_id"] == "deal-001"]
    assert matches, (
        f"expected a Needs-Attention item for deal-001; got {aar_data['needs_attention']}"
    )
    assert any("quarantined" in r for r in matches[0]["reasons"])


# ---------------------------------------------------------------------------
# Issue #42: quarantine.json must be rewritten every run, not only when the
# current run has quarantined entries — otherwise a stale quarantine.json
# from a prior run keeps flagging documents that a subsequent clean run
# resolved.
# ---------------------------------------------------------------------------


def test_quarantine_json_cleared_on_clean_rerun(tmp_path: Path) -> None:
    """A document that quarantines on one run and mines cleanly on a
    subsequent run must not leave a stale quarantine.json entry behind."""
    corpus_dir = tmp_path / "corpus"

    # deal-001 starts out with no ingestable content -> every version fails
    # ingest -> quarantined (same fixture shape as
    # test_all_versions_failed_document_is_quarantined).
    deal1_dir = corpus_dir / "deal-001"
    deal1_dir.mkdir(parents=True)
    _write_rtf(deal1_dir / "v1.rtf", "")

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

    taxonomy = load_taxonomy(_TAXONOMY_PATH)
    config = load_config(config_path)

    mine_corpus(
        corpus_dir=corpus_dir,
        config=config,
        taxonomy=taxonomy,
        out_dir=out_dir,
        progress=lambda _: None,
    )

    quarantine_path = out_dir / "quarantine.json"
    assert quarantine_path.exists()
    first_run = json.loads(quarantine_path.read_text(encoding="utf-8"))
    assert {q["document_id"] for q in first_run} == {"deal-001"}

    # Fix the document (real content) and re-run into the SAME out_dir, the
    # documented cache/re-run design.
    _write_rtf(deal1_dir / "v1.rtf", _CORPUS_BODY)

    mine_corpus(
        corpus_dir=corpus_dir,
        config=config,
        taxonomy=taxonomy,
        out_dir=out_dir,
        progress=lambda _: None,
    )

    # quarantine.json must reflect the CURRENT run, not the stale prior one.
    assert quarantine_path.exists(), (
        "quarantine.json should still be present (as an empty list), not left stale"
    )
    second_run = json.loads(quarantine_path.read_text(encoding="utf-8"))
    assert second_run == [], (
        f"expected an empty quarantine.json after a clean re-run, got {second_run}"
    )

    # The AAR must not keep reporting deal-001 as needing attention.
    from playbook_engine.aar import build_after_action_data

    aar_data = build_after_action_data(out_dir)
    matches = [i for i in aar_data["needs_attention"] if i["document_id"] == "deal-001"]
    assert not matches, f"deal-001 should no longer be flagged, got {matches}"


# ---------------------------------------------------------------------------
# Issue #83: a trail with no detected signed copy must not fabricate
# signed_copy_confidence or outcome="signed".
# ---------------------------------------------------------------------------


def test_unsigned_trail_no_fabricated_confidence_or_outcome(tmp_path: Path) -> None:
    """A document where no version is detected as signed must record an
    honest 'no signed copy' trail state, not a fabricated one.

    Regression guard for issue #83: previously
    ``version_order.signed_id or ordered_ids[-1]`` silently treated the
    last-in-chain draft as the signed copy, reporting a signed=False
    determination's confidence (e.g. 0.85 for basis="no_signature_section")
    as if it were confidence in the signed copy, and every non-reversed
    observation was stamped ``outcome="signed"`` regardless of whether any
    execution evidence existed.

    Neither ``_CORPUS_BODY`` version below contains a signature section, so
    ``detect_signed`` deterministically returns ``signed=False`` for both
    (no judge/LLM required) and ``order_versions`` never anchors a signed_id.
    """
    corpus_dir = tmp_path / "corpus"
    deal_dir = corpus_dir / "deal-001"
    deal_dir.mkdir(parents=True)
    _write_rtf(deal_dir / "v1.rtf", _CORPUS_BODY)
    _write_rtf(deal_dir / "v2.rtf", _CORPUS_BODY.replace("one year", "two years"))

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

    taxonomy = load_taxonomy(_TAXONOMY_PATH)
    config = load_config(config_path)

    mine_corpus(
        corpus_dir=corpus_dir,
        config=config,
        taxonomy=taxonomy,
        out_dir=out_dir,
    )

    trail = json.loads((out_dir / "trail" / "deal-001.json").read_text(encoding="utf-8"))
    assert trail["signed_version"] is None, (
        f"no version should be identified as signed; got {trail['signed_version']!r}"
    )
    assert trail["signed_copy_confidence"] is None, (
        "signed_copy_confidence must be None when no signed copy was detected "
        f"(fabricated from a fallback version otherwise); got "
        f"{trail['signed_copy_confidence']!r}"
    )

    obs_lines = (out_dir / "observations.jsonl").read_text(encoding="utf-8").splitlines()
    observations = [json.loads(line) for line in obs_lines if line.strip()]
    deal_obs = [o for o in observations if o["citation"]["document_id"] == "deal-001"]
    assert deal_obs, "deal-001 must have observations"
    assert all(o["outcome"] != "signed" for o in deal_obs), (
        "no observation may carry outcome='signed' when no signed copy was detected: "
        f"{[o['outcome'] for o in deal_obs]}"
    )


# ---------------------------------------------------------------------------
# Issue #84: hints.yaml written per docs/CORPUS-LAYOUT.md's documented
# example (entries WITH file extensions) must still anchor the intended
# version, whose version_id is a bare file STEM.
# ---------------------------------------------------------------------------


def test_hints_signed_version_with_extension_anchors_stem(tmp_path: Path) -> None:
    """A hints.yaml naming ``fully-executed.pdf`` (extension included, as
    docs/CORPUS-LAYOUT.md documents) must anchor the version whose
    version_id (file stem) is ``fully-executed``.

    Regression guard for issue #84: version ids are file stems (``vf.stem``
    — pipeline.py's per-version loop), so a hint value carrying an
    extension previously never matched any real version_id and the
    signed_version override silently had no effect. Neither version's body
    here contains a signature section, so — mirroring
    ``test_unsigned_trail_no_fabricated_confidence_or_outcome`` above —
    ``detect_signed`` deterministically returns ``signed=False`` for both;
    the ONLY way ``trail["signed_version"]`` can end up set is the hint
    actually taking effect.
    """
    corpus_dir = tmp_path / "corpus"
    deal_dir = corpus_dir / "deal-001"
    deal_dir.mkdir(parents=True)
    _write_rtf(deal_dir / "draft-we-sent.rtf", _CORPUS_BODY)
    _write_rtf(deal_dir / "fully-executed.rtf", _CORPUS_BODY.replace("one year", "two years"))

    # As documented in CORPUS-LAYOUT.md's hints.yaml example, entries carry
    # extensions -- even though the file actually on disk here is .rtf.
    (deal_dir / "hints.yaml").write_text("signed_version: fully-executed.pdf\n", encoding="utf-8")

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

    taxonomy = load_taxonomy(_TAXONOMY_PATH)
    config = load_config(config_path)

    mine_corpus(
        corpus_dir=corpus_dir,
        config=config,
        taxonomy=taxonomy,
        out_dir=out_dir,
    )

    trail = json.loads((out_dir / "trail" / "deal-001.json").read_text(encoding="utf-8"))
    assert trail["signed_version"] == "fully-executed", (
        "hints.yaml's signed_version (given WITH an extension, as documented) "
        f"must anchor the version whose stem is 'fully-executed'; got {trail['signed_version']!r}"
    )
    assert trail["signed_copy_confidence"] == 1.0, (
        "a hint-driven signed_version override must report full confidence; "
        f"got {trail['signed_copy_confidence']!r}"
    )


# ---------------------------------------------------------------------------
# Issue #88: DOCX tracked-changes author/date attribution must reach the
# observation store, not be discarded as an unconsumed side-channel.
# ---------------------------------------------------------------------------


def _docx_no_tracked_changes(tmp_path: Path, filename: str) -> Path:
    """Baseline DOCX version, no tracked changes.

    Two headings — scope_gate.MIN_CLAUSE_COUNT requires at least 2 clause
    nodes or the document is rejected as "too short" before L4 ever runs.
    """
    doc = Document()
    doc.add_heading("Obligations", level=1)
    doc.add_paragraph("Party A shall provide services to client.")
    doc.add_heading("Governing Law", level=1)
    doc.add_paragraph("This agreement is governed by the laws of the State of California.")
    path = tmp_path / filename
    doc.save(str(path))
    return path


def _docx_with_tracked_insertion(tmp_path: Path, filename: str) -> Path:
    """Redlined DOCX version: 'promptly' inserted by Alice via w:ins.

    SECURITY NOTE: synthetic text and a fictional author name only, matching
    the tracked-changes fixture convention in tests/test_docx_ingester.py.
    """
    doc = Document()
    doc.add_heading("Obligations", level=1)
    p = doc.add_paragraph()
    p.add_run("Party A shall ")

    ins_elem = etree.SubElement(p._p, _w("ins"))
    ins_elem.set(_w("id"), "1")
    ins_elem.set(_w("author"), "Alice")
    ins_elem.set(_w("date"), "2024-03-15T10:00:00Z")
    r_ins = etree.SubElement(ins_elem, _w("r"))
    t_ins = etree.SubElement(r_ins, _w("t"))
    t_ins.text = "promptly "

    p.add_run("provide services to client.")
    doc.add_heading("Governing Law", level=1)
    doc.add_paragraph("This agreement is governed by the laws of the State of California.")

    path = tmp_path / filename
    doc.save(str(path))
    return path


def test_docx_redline_observation_carries_tracked_changes_attribution(tmp_path: Path) -> None:
    """A clause changed via a tracked ``w:ins`` must carry author/date attribution
    on its observation — regression guard for issue #88.

    Previously ``_ingest_file`` returned only ``DocxIngestResult.tree``,
    discarding ``.tracked`` entirely: no observation could ever carry
    tracked-changes attribution, and ``tracked_changes_overlay.py`` had no
    caller in the pipeline (dead code, per the audit finding). This wires
    ``DocxIngestResult.tracked`` through ``_compute_doc_result`` into
    ``build_observations`` via ``tracked_changes_overlay.enrich_clause_diff``.
    """
    corpus_dir = tmp_path / "corpus"
    deal_dir = corpus_dir / "deal-002"
    deal_dir.mkdir(parents=True)
    _docx_no_tracked_changes(deal_dir, "v1.docx")
    _docx_with_tracked_insertion(deal_dir, "v2.docx")

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

    taxonomy = load_taxonomy(_TAXONOMY_PATH)
    config = load_config(config_path)

    mine_corpus(
        corpus_dir=corpus_dir,
        config=config,
        taxonomy=taxonomy,
        out_dir=out_dir,
    )

    obs_lines = (out_dir / "observations.jsonl").read_text(encoding="utf-8").splitlines()
    observations = [json.loads(line) for line in obs_lines if line.strip()]
    deal_obs = [o for o in observations if o["citation"]["document_id"] == "deal-002"]
    assert deal_obs, "deal-002 must have observations"

    attributed = [o for o in deal_obs if o.get("attribution") is not None]
    assert attributed, (
        f"expected at least one observation with tracked-changes attribution; "
        f"got attributions={[o.get('attribution') for o in deal_obs]}"
    )
    assert attributed[0]["attribution"] == {
        "author": "Alice",
        "date": "2024-03-15T10:00:00Z",
        "tracked_type": "insertion",
    }


_ROUND2_OBLIGATIONS_BASE = (
    "Party shall provide comprehensive consulting support services under this agreement."
)
_ROUND2_GOVERNING_LAW_PREFIX = "Agreement governed by "
_ROUND2_GOVERNING_LAW_SUFFIX = "laws of State California with dispute resolved exclusively."


def _docx_round2_template(tmp_path: Path, filename: str) -> Path:
    """Template baseline for the 3-version multi-round fixture below.

    Longer clause text than ``_docx_no_tracked_changes`` (>= 8 non-stopword
    tokens per clause) so ``clause_aligner``'s global move-matching phase's
    Jaccard tier (``MOVE_JACCARD_MIN_TOKENS``/``MOVE_JACCARD_THRESHOLD``) can
    chain each clause across all three versions into ONE aligned row instead
    of splitting a lightly-edited clause into a separate added/removed pair
    — required for the net (v1->v3) diff to come out ``kind="modified"``
    with hunks at all.

    SECURITY NOTE: synthetic text only, no real agreement content.
    """
    doc = Document()
    doc.add_heading("Obligations", level=1)
    doc.add_paragraph(_ROUND2_OBLIGATIONS_BASE)
    doc.add_heading("Governing Law", level=1)
    doc.add_paragraph(_ROUND2_GOVERNING_LAW_PREFIX + _ROUND2_GOVERNING_LAW_SUFFIX)
    path = tmp_path / filename
    doc.save(str(path))
    return path


def _docx_round1_tracked_insertion(tmp_path: Path, filename: str) -> Path:
    """Round-1 redline: 'promptly' inserted by Bob via w:ins, tracked
    against THIS version. By the time the final version below is authored,
    this insertion has already been accepted (plain text, no tracked-change
    record survives) — mirrors a real multi-round negotiation where an
    earlier round's redline is folded into the base text before the next
    round starts. Governing Law is untouched (byte-identical to the
    template) — round 1 only redlines Obligations.

    SECURITY NOTE: synthetic text and fictional author names only.
    """
    doc = Document()
    doc.add_heading("Obligations", level=1)
    p = doc.add_paragraph()
    p.add_run("Party shall ")

    ins_elem = etree.SubElement(p._p, _w("ins"))
    ins_elem.set(_w("id"), "1")
    ins_elem.set(_w("author"), "Bob")
    ins_elem.set(_w("date"), "2024-02-01T09:00:00Z")
    r_ins = etree.SubElement(ins_elem, _w("r"))
    t_ins = etree.SubElement(r_ins, _w("t"))
    t_ins.text = "promptly "

    p.add_run(_ROUND2_OBLIGATIONS_BASE.removeprefix("Party shall "))
    doc.add_heading("Governing Law", level=1)
    doc.add_paragraph(_ROUND2_GOVERNING_LAW_PREFIX + _ROUND2_GOVERNING_LAW_SUFFIX)

    path = tmp_path / filename
    doc.save(str(path))
    return path


def _docx_round2_signed_with_tracked_insertion(tmp_path: Path, filename: str) -> Path:
    """Final/signed version: round 1's 'promptly' insertion is now plain
    text (accepted, untracked, byte-identical to round 1's Obligations); a
    NEW tracked insertion by Alice lands in a DIFFERENT clause (Governing
    Law). This version's own tracked-changes side channel therefore carries
    exactly ONE distinct author (Alice), even though the document's full
    history has two.

    SECURITY NOTE: synthetic text and fictional author names only.
    """
    doc = Document()
    doc.add_heading("Obligations", level=1)
    doc.add_paragraph(
        "Party shall promptly " + _ROUND2_OBLIGATIONS_BASE.removeprefix("Party shall ")
    )

    doc.add_heading("Governing Law", level=1)
    p = doc.add_paragraph()
    p.add_run(_ROUND2_GOVERNING_LAW_PREFIX)

    ins_elem = etree.SubElement(p._p, _w("ins"))
    ins_elem.set(_w("id"), "2")
    ins_elem.set(_w("author"), "Alice")
    ins_elem.set(_w("date"), "2024-03-15T10:00:00Z")
    r_ins = etree.SubElement(ins_elem, _w("r"))
    t_ins = etree.SubElement(r_ins, _w("t"))
    t_ins.text = "substantive "

    p.add_run(_ROUND2_GOVERNING_LAW_SUFFIX)

    path = tmp_path / filename
    doc.save(str(path))
    return path


def test_net_diff_attribution_does_not_leak_earlier_round_author_into_signed_round_fallback(
    tmp_path: Path,
) -> None:
    """Regression guard (issue #118 fix round 2, finding 1): the net diff's
    round-level fallback tier must not fire when net_diffs span more than
    one negotiation round — it consults only the SIGNED version's own
    tracked-changes side channel, which is not "that round" for a clause
    whose change actually originated in an earlier round.

    Three versions: v1 (template) -> v2 (round 1, Bob inserts "promptly" in
    Obligations, tracked) -> v3 (signed; Bob's insertion is now plain text,
    Alice makes a NEW tracked insertion in the unrelated Governing Law
    clause). v3's side channel has exactly one distinct author (Alice) —
    exactly the fallback tier's firing condition — but the Obligations
    clause's net change (the "promptly" insertion) has nothing to do with
    Alice's round. Before the fix, the fallback fired unconditionally for
    any unmatched hunk and wrongly attributed the Obligations change to
    Alice; after the fix it must stay unattributed ("unknown"), while the
    Governing Law clause (Alice's own real, per-hunk-matched change) must
    still resolve correctly to Alice — proving the fix only suppresses the
    coarse fallback, not the direct per-hunk match.
    """
    corpus_dir = tmp_path / "corpus"
    deal_dir = corpus_dir / "deal-003"
    deal_dir.mkdir(parents=True)
    _docx_round2_template(deal_dir, "v1.docx")
    _docx_round1_tracked_insertion(deal_dir, "v2.docx")
    _docx_round2_signed_with_tracked_insertion(deal_dir, "v3.docx")

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

    taxonomy = load_taxonomy(_TAXONOMY_PATH)
    config = load_config(config_path)

    mine_corpus(
        corpus_dir=corpus_dir,
        config=config,
        taxonomy=taxonomy,
        out_dir=out_dir,
    )

    obs_lines = (out_dir / "observations.jsonl").read_text(encoding="utf-8").splitlines()
    observations = [json.loads(line) for line in obs_lines if line.strip()]
    deal_obs = [o for o in observations if o["citation"]["document_id"] == "deal-003"]
    assert deal_obs, "deal-003 must have observations"

    obligations_obs = [o for o in deal_obs if "promptly" in o.get("full_text", "")]
    assert obligations_obs, (
        f"expected an observation carrying the 'promptly' insertion; got {deal_obs}"
    )
    for o in obligations_obs:
        assert o.get("attribution") is None, (
            "Obligations clause's change originated in round 1 (Bob), not "
            "the signed version's own side channel (Alice) — the "
            "round-level fallback must not leak the signed round's sole "
            f"author onto it; got attribution={o.get('attribution')!r}"
        )
        assert o.get("proposed_by") == "unknown", (
            f"expected proposed_by='unknown', got {o.get('proposed_by')!r}"
        )

    governing_law_obs = [o for o in deal_obs if "substantive laws" in o.get("full_text", "")]
    assert governing_law_obs, (
        f"expected an observation carrying Alice's Governing Law insertion; got {deal_obs}"
    )
    assert governing_law_obs[0]["attribution"] == {
        "author": "Alice",
        "date": "2024-03-15T10:00:00Z",
        "tracked_type": "insertion",
    }


# ---------------------------------------------------------------------------
# Issue #103: a single-version document's clauses must be diffed against the
# canonical template, not hardcoded deviation="none".
# ---------------------------------------------------------------------------


def test_single_version_clause_diffed_against_template(tmp_path: Path) -> None:
    """A single-version document with a signed clause that differs from the
    canonical template must route that clause through the deviation judge,
    not hardcode deviation="none"/basis="deterministic".

    Regression guard for issue #103: previously
    ``_observations_from_single_version`` recorded EVERY clause of a
    single-version document as deviation="none" unconditionally — a document
    with no negotiation trail was never actually checked against the
    template at all. ``_make_corpus_with_template`` gives a single-version
    document (``deal-001/v1.rtf``) whose clause texts (``_CORPUS_BODY``)
    differ materially from the configured template's (``_TEMPLATE_BODY`` —
    different party names, different governing-law state, different term
    language), so every classified clause should be routed to the injected
    fake judge.
    """
    corpus_dir, config_path, out_dir, _template_path = _make_corpus_with_template(tmp_path)

    taxonomy = load_taxonomy(_TAXONOMY_PATH)
    cfg = load_config(config_path)

    fake_judge = _FakeDeviationJudge()

    mine_corpus(
        corpus_dir=corpus_dir,
        config=cfg,
        taxonomy=taxonomy,
        out_dir=out_dir,
        deviation_judge=fake_judge,
    )

    assert fake_judge.received_items, (
        "the deviation judge must be invoked for a single-version document "
        "whose clauses differ from the canonical template — it was never "
        "reached at all before issue #103"
    )

    obs_lines = (out_dir / "observations.jsonl").read_text(encoding="utf-8").splitlines()
    observations = [json.loads(line) for line in obs_lines if line.strip()]
    deal_obs = [o for o in observations if o["citation"]["document_id"] == "deal-001"]
    assert deal_obs, "deal-001 must have observations"

    judged = [o for o in deal_obs if o["basis"] == "judge"]
    assert judged, (
        f"expected at least one observation routed through the injected "
        f"deviation judge (basis='judge'); got bases={[o['basis'] for o in deal_obs]}"
    )
    assert all(o["deviation"] == "substantive" for o in judged), (
        "the injected fake judge's verdict must reach the observation, not a "
        "hardcoded deviation='none'"
    )


# ---------------------------------------------------------------------------
# Issue #147: embedded attorney-pinned positions (curation overlay) survive a
# recompile and are conflict-flagged when fresh evidence contradicts them.
# ---------------------------------------------------------------------------


def _inject_pin(out_dir: Path, clause_id: str, position: str, baseline_stance: str) -> None:
    """Simulate an attorney pin having been embedded in a PRIOR compile.

    Writes ``curation.pins`` directly into ``playbook.opf.json`` — mirroring
    what ``viewer.apply_feedback``'s ``override`` handling does — so the next
    ``project_playbook`` call has a prior curation section to merge against.
    """
    opf_path = out_dir / "playbook.opf.json"
    doc = json.loads(opf_path.read_text(encoding="utf-8"))
    doc["curation"] = {
        "pins": [
            {
                "clause_id": clause_id,
                "item_id": "C1",
                "position": position,
                "baseline_stance": baseline_stance,
                "pinned_at": "2026-01-01T00:00:00Z",
            }
        ]
    }
    opf_path.write_text(json.dumps(doc), encoding="utf-8")


def _add_our_paper_observations(out_dir: Path, taxonomy_id: str, n: int) -> None:
    """Append *n* synthetic our-paper, deviation=none, signed observations for
    *taxonomy_id* to the observation store, and register their (synthetic)
    document ids in ``corpus_manifest.json`` so citation validation resolves.

    This is how the tests below simulate "new evidence was mined" between two
    ``project_playbook`` calls, without invoking the full ingest/judge
    pipeline (project_playbook's own contract is "given the store, recompile"
    — manipulating the store directly is a legitimate way to exercise that).
    """
    manifest_path = out_dir / "corpus_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    obs_path = out_dir / "observations.jsonl"

    new_obs: list[Observation] = []
    with obs_path.open("a", encoding="utf-8") as f:
        for i in range(1, n + 1):
            doc_id = f"synthetic-our-paper-{i}"
            manifest.append(
                {
                    "document_id": doc_id,
                    "provenance": "our_paper",
                    "in_scope": True,
                    "versions": 1,
                    "versions_mined": 1,
                    "versions_found": 1,
                    "version_ingest": [
                        {"version": "v1", "status": "ok", "error": None, "extractor": "rtf"}
                    ],
                }
            )
            obs = Observation(
                observation_id=f"{doc_id}/1/2",
                taxonomy_id=taxonomy_id,
                text_summary="Synthetic matching clause text.",
                citation=ObservationCitation(
                    document_id=doc_id, version=1, clause_path="2", char_span=(0, 10)
                ),
                deviation="none",
                risk_delta={"direction": "neutral", "magnitude": "none"},
                provenance="our_paper",
                outcome="signed",
                confidence=0.9,
                basis="judge",
            )
            new_obs.append(obs)
            f.write(json.dumps(obs.to_dict()) + "\n")

    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def test_curation_pin_survives_recompile_with_unchanged_evidence(tmp_path: Path) -> None:
    """A pinned position survives a recompile when evidence is unchanged:
    the pin is preserved verbatim and no conflict is raised.
    """
    corpus_dir, config_path, out_dir, _template_path = _make_corpus_with_template(tmp_path)
    taxonomy = load_taxonomy(_TAXONOMY_PATH)
    cfg = load_config(config_path)

    mine_corpus(corpus_dir=corpus_dir, config=cfg, taxonomy=taxonomy, out_dir=out_dir)
    playbook_v1 = project_playbook(
        out_dir=out_dir, config=cfg, taxonomy=taxonomy, opf_version="0.3"
    )

    clause = next(
        c for c in playbook_v1["evidence"]["clauses"] if c["taxonomy_id"] == "governing_law"
    )
    baseline_stance = clause["summary"]["historical_stance"]
    _inject_pin(
        out_dir,
        clause_id=clause["id"],
        position="consistently_held",
        baseline_stance=baseline_stance,
    )

    # Recompile with NO change to the observation store.
    playbook_v2 = project_playbook(
        out_dir=out_dir, config=cfg, taxonomy=taxonomy, opf_version="0.3"
    )

    pins = playbook_v2["curation"]["pins"]
    assert len(pins) == 1
    pin = pins[0]
    assert pin["clause_id"] == clause["id"]
    assert pin["position"] == "consistently_held", "the pinned position must be preserved verbatim"
    assert pin.get("conflict") is None, (
        "no conflict should be raised when evidence (and thus historical_stance) is unchanged"
    )

    clause_v2 = next(
        c for c in playbook_v2["evidence"]["clauses"] if c["taxonomy_id"] == "governing_law"
    )
    assert clause_v2["summary"]["historical_stance"] == baseline_stance, (
        "historical_stance itself must remain descriptive/unchanged — the pin does not overwrite it"
    )


def test_curation_pin_flags_conflict_on_contradicting_evidence(tmp_path: Path) -> None:
    """New evidence that changes a clause's historical_stance since a pin was
    made raises a conflict flag on that pin — the pin is NOT silently
    overridden or dropped.
    """
    corpus_dir, config_path, out_dir, _template_path = _make_corpus_with_template(tmp_path)
    taxonomy = load_taxonomy(_TAXONOMY_PATH)
    cfg = load_config(config_path)

    # Mined as an opt-in judged run (issue #230): the injected evidence below
    # is judged rows (basis="judge"), which only a judged-mode store carries,
    # and the mode recorded at mine time — not the rows — decides whether
    # project derives a stance at all (the consumer path never does).
    mine_corpus(
        corpus_dir=corpus_dir,
        config=cfg,
        taxonomy=taxonomy,
        out_dir=out_dir,
        deviation_judge=_NullDeviationJudge(),
    )
    playbook_v1 = project_playbook(
        out_dir=out_dir, config=cfg, taxonomy=taxonomy, opf_version="0.3"
    )

    clause = next(
        c for c in playbook_v1["evidence"]["clauses"] if c["taxonomy_id"] == "governing_law"
    )
    baseline_stance = clause["summary"]["historical_stance"]
    assert baseline_stance == "no_signal", (
        "premise: a single-document corpus has insufficient our-paper evidence "
        f"(see clause_position_compiler.MIN_EVIDENCE_N); got {baseline_stance!r}"
    )
    _inject_pin(
        out_dir,
        clause_id=clause["id"],
        position="consistently_held",
        baseline_stance=baseline_stance,
    )

    # Simulate new evidence: enough our-paper, deviation=none, signed
    # observations to push governing_law's historical_stance away from
    # "no_signal" (evidence_sufficient flips True; see
    # clause_position_compiler._historical_stance).
    _add_our_paper_observations(out_dir, taxonomy_id="governing_law", n=2)

    playbook_v2 = project_playbook(
        out_dir=out_dir, config=cfg, taxonomy=taxonomy, opf_version="0.3"
    )

    clause_v2 = next(
        c for c in playbook_v2["evidence"]["clauses"] if c["taxonomy_id"] == "governing_law"
    )
    recomputed_stance = clause_v2["summary"]["historical_stance"]
    assert recomputed_stance != baseline_stance, (
        "premise: the injected evidence must actually move historical_stance"
    )

    pin = playbook_v2["curation"]["pins"][0]
    assert pin["position"] == "consistently_held", (
        "the pin's asserted position must survive even when flagged — it is "
        "not silently overridden by the recomputed rollup"
    )
    conflict = pin.get("conflict")
    assert conflict is not None, "contradicting evidence must raise a conflict flag"
    assert conflict["recomputed_historical_stance"] == recomputed_stance
    assert baseline_stance in conflict["reason"]


def test_curation_pin_on_opf_04_survives_recompile_and_03_pin_is_flagged(
    tmp_path: Path,
) -> None:
    """Issue #223: curation is unchanged in OPF 0.4, which carries no stance.
    A pin made on a 0.4 document by the real producer (``playbook curate``)
    records ``baseline_stance`` "unknown" — the same accessor the recompile
    compares against — so it survives a 0.4 recompile with no conflict. A pin
    carried over from a 0.3 compile (a real stance as its baseline) is flagged
    once on the first 0.4 compile: its stance is no longer computed."""
    from playbook_engine.chat_curate import apply_curate_commands
    from playbook_engine.validator import validate_document

    corpus_dir, config_path, out_dir, _template_path = _make_corpus_with_template(tmp_path)
    taxonomy = load_taxonomy(_TAXONOMY_PATH)
    cfg = load_config(config_path)
    mine_corpus(corpus_dir=corpus_dir, config=cfg, taxonomy=taxonomy, out_dir=out_dir)

    project_playbook(out_dir=out_dir, config=cfg, taxonomy=taxonomy)
    apply_curate_commands(
        out_dir, ["pin governing_law to usually_held"], now="2026-01-01T00:00:00Z"
    )
    pinned = json.loads((out_dir / "playbook.opf.json").read_text(encoding="utf-8"))
    (pin,) = pinned["curation"]["pins"]
    assert pin["baseline_stance"] == "unknown"
    playbook_v2 = project_playbook(out_dir=out_dir, config=cfg, taxonomy=taxonomy)
    (pin_v2,) = playbook_v2["curation"]["pins"]
    assert pin_v2["position"] == "usually_held"
    assert pin_v2.get("conflict") is None

    # A 0.3 pin (real stance baseline) carried into a 0.4 compile.
    project_playbook(out_dir=out_dir, config=cfg, taxonomy=taxonomy, opf_version="0.3")
    (out_dir / "playbook.opf.json").write_text(
        json.dumps(
            {
                **json.loads((out_dir / "playbook.opf.json").read_text(encoding="utf-8")),
                "curation": {"pins": []},
            }
        ),
        encoding="utf-8",
    )
    apply_curate_commands(
        out_dir, ["pin governing_law to usually_held"], now="2026-01-01T00:00:00Z"
    )
    pinned_03 = json.loads((out_dir / "playbook.opf.json").read_text(encoding="utf-8"))
    assert pinned_03["curation"]["pins"][0]["baseline_stance"] == "no_signal"
    playbook_v3 = project_playbook(out_dir=out_dir, config=cfg, taxonomy=taxonomy)
    conflict = playbook_v3["curation"]["pins"][0]["conflict"]
    assert conflict["recomputed_historical_stance"] == "unknown"
    assert validate_document(playbook_v3).ok


def test_posture_and_floor_survive_recompile(tmp_path: Path) -> None:
    """Issue #123 regression: a re-run of `project_playbook` (Route C's
    "re-derive Evidence from a changed corpus") must carry forward the
    prior playbook's authored Posture and signed Floor VERBATIM, exactly
    like curation pins — not reset them to `{}`.

    Before the fix, `assemble_playbook` unconditionally wrote
    `playbook["posture"] = {}` / `playbook["floor"] = {}` on every compile,
    and `project_playbook` only ever read the prior playbook's `curation`
    key forward — so a GC's `playbook posture interview` + `playbook floor
    sign` work was silently destroyed by the next `project`, contradicting
    SKILL.md Route C's "the Posture and Floor you already signed should
    survive" promise.
    """
    corpus_dir, config_path, out_dir, _template_path = _make_corpus_with_template(tmp_path)
    taxonomy = load_taxonomy(_TAXONOMY_PATH)
    cfg = load_config(config_path)

    mine_corpus(corpus_dir=corpus_dir, config=cfg, taxonomy=taxonomy, out_dir=out_dir)
    project_playbook(out_dir=out_dir, config=cfg, taxonomy=taxonomy)

    # Simulate a GC authoring Posture (Route B step 7a) and signing an extra
    # hard line (`playbook floor sign`), exactly as they would between two
    # `project` runs.
    apply_posture_interview(
        out_dir,
        {
            "rounds": "Usually 2 rounds before escalating.",
            "leverage": "Collaborative; we often want the deal.",
            "risk_appetite": "Default to accept-to-close on non-material changes.",
            "sacred_clauses": "Liability cap and indemnification.",
            "flexible_clauses": "Term length and renewal mechanics.",
            "audience": "Terse rationale for a GC audience.",
        },
        generated_at="2026-01-01T00:00:00Z",
    )
    opf_path = out_dir / "playbook.opf.json"
    doc = json.loads(opf_path.read_text(encoding="utf-8"))
    signed_invariants = sign_floor_invariant(
        "Never accept unlimited liability.",
        rationale="Hand-signed for this test.",
        signed_by="Test Legal Owner",
        existing_invariants=doc["floor"]["invariants"],
    )
    doc["floor"]["invariants"] = signed_invariants
    opf_path.write_text(json.dumps(doc), encoding="utf-8")

    posture_before = json.loads(opf_path.read_text(encoding="utf-8"))["posture"]
    floor_before = json.loads(opf_path.read_text(encoding="utf-8"))["floor"]
    assert posture_before, "premise: the interview must have actually written a Posture"
    assert floor_before["invariants"], "premise: signing must have actually recorded an invariant"

    # Route C: re-derive Evidence from the (unchanged) corpus.
    playbook_v2 = project_playbook(out_dir=out_dir, config=cfg, taxonomy=taxonomy)

    assert playbook_v2["posture"] == posture_before, (
        "authored Posture must survive a recompile verbatim — Route C's promise"
    )
    assert playbook_v2["floor"] == floor_before, (
        "signed Floor invariants must survive a recompile verbatim — Route C's promise"
    )
    # And the carried-forward content is reflected in the recomputed identity
    # hashes (posture/floor ARE part of content_hash, unlike curation).
    assert playbook_v2["identity"]["content_hash"] == content_hash(playbook_v2)
    assert playbook_v2["identity"]["section_digests"] == compute_section_digests(playbook_v2)


def test_curation_pin_excluded_from_content_hash_but_digested_separately(tmp_path: Path) -> None:
    """Adding/updating a curation pin must not perturb identity.content_hash
    (curation is excluded — issue #147), but identity.section_digests.curation
    must change to track the pin content.
    """
    corpus_dir, config_path, out_dir, _template_path = _make_corpus_with_template(tmp_path)
    taxonomy = load_taxonomy(_TAXONOMY_PATH)
    cfg = load_config(config_path)

    mine_corpus(corpus_dir=corpus_dir, config=cfg, taxonomy=taxonomy, out_dir=out_dir)
    playbook_v1 = project_playbook(
        out_dir=out_dir, config=cfg, taxonomy=taxonomy, opf_version="0.3"
    )
    hash_v1 = content_hash(playbook_v1)
    assert hash_v1 == playbook_v1["identity"]["content_hash"]
    curation_digest_v1 = playbook_v1["identity"]["section_digests"]["curation"]

    clause = next(
        c for c in playbook_v1["evidence"]["clauses"] if c["taxonomy_id"] == "governing_law"
    )
    _inject_pin(
        out_dir,
        clause_id=clause["id"],
        position="consistently_held",
        baseline_stance=clause["summary"]["historical_stance"],
    )

    playbook_v2 = project_playbook(
        out_dir=out_dir, config=cfg, taxonomy=taxonomy, opf_version="0.3"
    )

    assert playbook_v2["identity"]["content_hash"] == hash_v1, (
        "a curation-only change (no evidence change) must not perturb content_hash"
    )
    assert playbook_v2["identity"]["section_digests"]["curation"] != curation_digest_v1, (
        "the curation section digest must change once a pin is present"
    )
    # Sanity: the digest really is computed the same way canonicalize.py defines it.
    assert playbook_v2["identity"]["section_digests"] == compute_section_digests(playbook_v2)


# ---------------------------------------------------------------------------
# Issue #65: a removed clause's classification confidence must be looked up
# in the version it was actually classified in, not the signed version's map.
# ---------------------------------------------------------------------------

_REMOVED_CLAUSE_V1_BODY = (
    r"1. Indemnification\par "
    r"Alpha Corp shall indemnify Beta University against third-party claims "
    r"arising from the placement programme.\par "
    r"2. Insurance\par "
    r"Alpha Corp shall maintain commercial general liability insurance.\par "
    r"3. Term\par "
    r"This agreement commences on the date of execution and continues for one year.\par "
)

# v2's clause 2 is a DIFFERENT clause (Confidentiality, not Insurance) that
# merely lands at the same path number once Insurance is gone -- exactly the
# renumbering-collision setup issue #65 describes. Its heading only
# partially overlaps the "Confidentiality" taxonomy label (Jaccard 0.5,
# below auto_classify_threshold), giving it a deterministic confidence
# (0.5) distinct from Insurance's exact-match 1.0, so a wrong lookup is
# unambiguously observable.
_REMOVED_CLAUSE_V2_BODY = (
    r"1. Indemnification\par "
    r"Alpha Corp shall indemnify Beta University against third-party claims "
    r"arising from the placement programme.\par "
    r"2. Confidentiality Requirements\par "
    r"Each party shall keep the other's confidential information secret.\par "
    r"3. Term\par "
    r"This agreement commences on the date of execution and continues for one year.\par "
)


# Our standard (template) Insurance clause text for the test below: either a
# DIFFERENT insurance clause (so v1's struck Insurance text is non-standard —
# their ask, refused) or v1's own text (our standard, struck — our
# concession). Issue #216: the origin of the struck text decides its outcome.
_TEMPLATE_INSURANCE_OTHER = (
    "Alpha Corp shall maintain professional indemnity cover of five million dollars."
)
_TEMPLATE_INSURANCE_SAME = "Alpha Corp shall maintain commercial general liability insurance."


@pytest.mark.parametrize(
    ("template_insurance", "expected_outcome"),
    [
        (_TEMPLATE_INSURANCE_OTHER, "proposed_then_reversed"),
        (_TEMPLATE_INSURANCE_SAME, "conceded_before_signing"),
    ],
)
def test_removed_clause_confidence_not_borrowed_from_signed_version(
    tmp_path: Path, template_insurance: str, expected_outcome: str
) -> None:
    """A removed clause's observation must carry ITS OWN draft-version
    classification confidence, never one borrowed from an unrelated clause
    that happens to occupy the same renumbered path in the signed version —
    regression guard for issue #65.

    v1's clause path "2" ("Insurance", heading exact-matches the taxonomy
    label -> confidence 1.0) is entirely absent from v2 (removed, not just
    modified). v2's clause path "2" ("Confidentiality Requirements", a
    partial heading match -> confidence 0.5) is a different, unrelated
    clause that merely lands at the same path number after renumbering.

    Before the fix, the confidence map (``cls_conf_by_path``) was built
    ONLY from the signed version's (v2) classified clauses, so looking up
    the removed clause's before-path "2" found v2's Confidentiality
    classification instead of v1's own Insurance classification — silently
    attaching confidence 0.5 to the Insurance observation instead of its
    real 1.0.

    Issue #216: the removed clause's outcome is decided by the ORIGIN of its
    text against the template's Insurance standard (parametrized: a
    different standard makes it their refused ask; the same text makes it
    our concession) — both through the real pipeline. The deal is SIGNED
    (hints.yaml names v2 as the executed copy): only a deal with a detected
    executed copy can concede (issue #83; see
    test_unsigned_deal_striking_our_standard_is_never_a_concession).
    """
    corpus_dir = tmp_path / "corpus"
    deal_dir = corpus_dir / "deal-001"
    deal_dir.mkdir(parents=True)
    _write_rtf(deal_dir / "v1.rtf", _REMOVED_CLAUSE_V1_BODY)
    _write_rtf(deal_dir / "v2.rtf", _REMOVED_CLAUSE_V2_BODY)
    (deal_dir / "hints.yaml").write_text("signed_version: v2.rtf\n", encoding="utf-8")
    template_path = tmp_path / "template.rtf"
    _write_rtf(template_path, rf"1. Insurance\par {template_insurance}\par ")

    cfg = {
        "agreement_type": {
            "id": "educational-affiliation",
            "name": "Educational Affiliation Agreement",
        },
        "baseline": {"template": str(template_path)},
        "taxonomy": str(_TAXONOMY_PATH),
        "provenance": {"our_party_aliases": ["Alpha Corp"]},
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

    obs_lines = (out_dir / "observations.jsonl").read_text(encoding="utf-8").splitlines()
    observations = [json.loads(line) for line in obs_lines if line.strip()]

    insurance_obs = [o for o in observations if o["taxonomy_id"] == "insurance"]
    assert len(insurance_obs) == 1, (
        f"expected exactly one 'insurance' observation (the removed v1 clause); "
        f"got {len(insurance_obs)}: {insurance_obs}"
    )
    removed = insurance_obs[0]
    manifest = json.loads((out_dir / "corpus_manifest.json").read_text(encoding="utf-8"))
    assert [d["signed_version"] for d in manifest] == [2], "premise: v2 is the executed copy"
    # Issue #216: text removed before signing is never "signed" — its own
    # text is absent from the signed v2, and its origin decides the outcome.
    assert removed["outcome"] == expected_outcome, removed
    assert removed["citation"]["version_id"] == "v1", (
        "premise: the removed Insurance clause's citation must resolve to v1 "
        f"(the version it actually came from); got {removed['citation']!r}"
    )
    assert removed["confidence"] == 1.0, (
        "the removed clause's confidence must be its OWN v1 classification "
        "(exact heading match -> 1.0), never a confidence borrowed from "
        f"whatever clause occupies path 2 in the signed version; got "
        f"{removed['confidence']!r}"
    )

    # The unrelated clause that legitimately occupies path "2" in the signed
    # version must keep ITS OWN confidence -- confirms the fix is a correct
    # per-version lookup, not just a hardcoded override in one direction.
    signed_path_2 = [
        o
        for o in observations
        if o["citation"]["version_id"] == "v2" and o["citation"]["clause_path"] == "2"
    ]
    assert len(signed_path_2) == 1, (
        f"expected exactly one observation citing v2 clause_path 2; got {signed_path_2}"
    )
    assert signed_path_2[0]["confidence"] == 0.5, (
        "the signed version's own clause at path 2 must keep its own "
        f"confidence; got {signed_path_2[0]['confidence']!r}"
    )


def test_removed_clause_with_no_template_is_dropped_as_origin_undetermined(
    tmp_path: Path,
) -> None:
    """Issue #216: with no template there is no standard to tell our struck
    language from their struck ask, so the removed Insurance clause is
    neither proposed_then_reversed nor a concession: no observation, and the
    drop is counted on the document's corpus_manifest.json row."""
    corpus_dir = tmp_path / "corpus"
    deal_dir = corpus_dir / "deal-001"
    deal_dir.mkdir(parents=True)
    _write_rtf(deal_dir / "v1.rtf", _REMOVED_CLAUSE_V1_BODY)
    _write_rtf(deal_dir / "v2.rtf", _REMOVED_CLAUSE_V2_BODY)
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
    mine_corpus(
        corpus_dir=corpus_dir,
        config=load_config(config_path),
        taxonomy=load_taxonomy(_TAXONOMY_PATH),
        out_dir=out_dir,
    )
    obs_lines = (out_dir / "observations.jsonl").read_text(encoding="utf-8").splitlines()
    observations = [json.loads(line) for line in obs_lines if line.strip()]
    assert [o for o in observations if o["taxonomy_id"] == "insurance"] == []
    manifest = json.loads((out_dir / "corpus_manifest.json").read_text(encoding="utf-8"))
    assert [d.get("dropped_observations") for d in manifest] == [{"removed_origin_undetermined": 1}]


def _mine_and_project_insurance(
    root: Path, *, deal_001: str | None, opf_version: str = "0.3"
) -> tuple[list[dict], list[dict], dict, dict]:
    """Mine + project a corpus whose deal-002 is signed and keeps our
    template's Insurance clause. *deal_001* adds a deal-001 that starts from
    our template (v1) and strikes its Insurance clause in v2: ``"unsigned"``
    (no hints.yaml, so no detected executed copy), ``"signed"`` (hints.yaml
    names v2), or ``None`` (no deal-001 at all). Returns (observations,
    corpus_manifest, the insurance evidence clause, corpus.stats)."""
    corpus_dir = root / "corpus"
    signed = corpus_dir / "deal-002"
    signed.mkdir(parents=True)
    _write_rtf(signed / "v1.rtf", _REMOVED_CLAUSE_V1_BODY)
    _write_rtf(signed / "v2.rtf", _REMOVED_CLAUSE_V1_BODY)
    (signed / "hints.yaml").write_text("signed_version: v2.rtf\n", encoding="utf-8")
    if deal_001 is not None:
        struck = corpus_dir / "deal-001"
        struck.mkdir(parents=True)
        _write_rtf(struck / "v1.rtf", _REMOVED_CLAUSE_V1_BODY)
        _write_rtf(struck / "v2.rtf", _REMOVED_CLAUSE_V2_BODY)
        if deal_001 == "signed":
            (struck / "hints.yaml").write_text("signed_version: v2.rtf\n", encoding="utf-8")
    template_path = root / "template.rtf"
    _write_rtf(template_path, _REMOVED_CLAUSE_V1_BODY)
    cfg = {
        "agreement_type": {
            "id": "educational-affiliation",
            "name": "Educational Affiliation Agreement",
        },
        "baseline": {"template": str(template_path)},
        "taxonomy": str(_TAXONOMY_PATH),
        "provenance": {"our_party_aliases": ["Alpha Corp"]},
    }
    config_path = root / "playbook.config.yaml"
    config_path.write_text(yaml.dump(cfg), encoding="utf-8")
    out_dir = root / "out"
    config = load_config(config_path)
    taxonomy = load_taxonomy(_TAXONOMY_PATH)
    mine_corpus(corpus_dir=corpus_dir, config=config, taxonomy=taxonomy, out_dir=out_dir)
    obs_lines = (out_dir / "observations.jsonl").read_text(encoding="utf-8").splitlines()
    observations = [json.loads(line) for line in obs_lines if line.strip()]
    manifest = json.loads((out_dir / "corpus_manifest.json").read_text(encoding="utf-8"))
    playbook = project_playbook(
        out_dir=out_dir, config=config, taxonomy=taxonomy, opf_version=opf_version
    )
    clause = next(c for c in playbook["evidence"]["clauses"] if c["taxonomy_id"] == "insurance")
    if opf_version == "0.4":
        # The 0.4 clause plus its precedent, for the issue #223 assertions.
        clause = {
            **clause,
            "_precedent": [
                p for p in playbook["evidence"]["precedent"] if p["taxonomy_id"] == "insurance"
            ],
        }
    return observations, manifest, clause, playbook["corpus"]["stats"]


def test_unsigned_deal_striking_our_standard_is_never_a_concession(tmp_path: Path) -> None:
    """Issue #216 + #83: a deal with no detected executed copy never counts
    as "we conceded". deal-001 strikes our standard Insurance clause but has
    no signed copy, so the strike produces no observation (counted under
    removed_standard_no_signed_copy) and the Insurance clause's stance_detail,
    position and observed_positions are exactly those of a corpus without
    deal-001. The signed control proves the guard is what keeps it out: the
    same strike in a signed deal-001 IS a conceded deal."""
    from playbook_engine.observation_builder import OUTCOME_CONCEDED_BEFORE_SIGNING

    observations, manifest, clause, stats = _mine_and_project_insurance(
        tmp_path / "unsigned", deal_001="unsigned"
    )
    assert [(d["document_id"], d["signed_version"]) for d in manifest] == [
        ("deal-001", None),
        ("deal-002", 2),
    ], "premise: deal-001 has no detected executed copy; deal-002 is signed"
    assert [o for o in observations if o["outcome"] == OUTCOME_CONCEDED_BEFORE_SIGNING] == []
    assert [o for o in observations if o["taxonomy_id"] == "insurance"] == [
        o
        for o in observations
        if o["citation"]["document_id"] == "deal-002" and o["taxonomy_id"] == "insurance"
    ]
    assert [d["dropped_observations"] for d in manifest] == [
        {"removed_standard_no_signed_copy": 1},
        {},
    ]
    assert stats["dropped_observations"]["by_reason"] == {"removed_standard_no_signed_copy": 1}
    assert (
        clause["summary"]["stance_detail"]["held"],
        clause["summary"]["stance_detail"]["of"],
    ) == (
        1,
        1,
    )
    assert [o["example_ref"]["document_id"] for o in clause["observed_positions"]] == ["deal-002"]

    _, _, without_deal_001, _ = _mine_and_project_insurance(tmp_path / "absent", deal_001=None)
    assert clause["summary"] == without_deal_001["summary"], (
        "an unsigned deal must not move stance_detail or the position"
    )
    assert clause["observed_positions"] == without_deal_001["observed_positions"]

    signed_obs, _, signed_clause, _ = _mine_and_project_insurance(
        tmp_path / "signed", deal_001="signed"
    )
    assert [
        o["citation"]["document_id"]
        for o in signed_obs
        if o["outcome"] == OUTCOME_CONCEDED_BEFORE_SIGNING
    ] == ["deal-001"], "control: the same strike in a signed deal is our concession"
    signed_detail = signed_clause["summary"]["stance_detail"]
    assert (signed_detail["held"], signed_detail["of"]) == (1, 2)


def test_unsigned_deal_striking_our_standard_opf_04_precedent(tmp_path: Path) -> None:
    """Issue #223: the same three corpora projected as OPF 0.4. An unsigned
    deal that struck our standard produces no precedent for the clause; the
    same strike in a signed deal is a precedent with no signed text, our
    standard as its opening text, ``standard: false``, ``moved: true`` — and
    never a refused ask. Counts are distinct deals."""
    _, _, unsigned, _ = _mine_and_project_insurance(
        tmp_path / "unsigned", deal_001="unsigned", opf_version="0.4"
    )
    assert [p["document_id"] for p in unsigned["_precedent"]] == ["deal-002"]
    (kept,) = unsigned["_precedent"]
    assert kept["signed"] is True and kept["standard"] is True
    assert (unsigned["n_deals"], unsigned["n_signed_standard"]) == (1, 1)

    _, _, signed, _ = _mine_and_project_insurance(
        tmp_path / "signed", deal_001="signed", opf_version="0.4"
    )
    by_deal = {p["document_id"]: p for p in signed["_precedent"]}
    assert set(by_deal) == {"deal-001", "deal-002"}
    struck = by_deal["deal-001"]
    assert struck["signed"] is True
    assert struck["signed_text"] is None
    assert struck["opening_text"] is not None
    assert struck["opening_text"]["ref"]["version"] == 1
    assert struck["standard"] is False and struck["moved"] is True
    assert struck["refused_asks"] == []
    assert (signed["n_deals"], signed["n_signed_standard"], signed["n_refused"]) == (2, 1, 0)


# ---------------------------------------------------------------------------
# Issue #95 fix round 1: project_playbook must cap x_search_snippet even for
# a legacy store that predates the search_snippet field (no key at all in
# observations.jsonl).
# ---------------------------------------------------------------------------

# Indemnification body is deliberately long (> _SEARCH_SNIPPET_MAX) so the
# premise -- that there is something to truncate -- is verifiable, not
# assumed. A filled "By:" signature line (signed_detector.py's
# single_signature basis) makes this single-version document detected as
# executed: without it outcome != "signed" and compile_clause_positions
# withholds the observation from observed_positions entirely, which would
# make the assertions below pass vacuously on an empty list.
_LEGACY_STORE_BODY = (
    r"1. Indemnification\par "
    r"Alpha Corp shall indemnify and hold harmless Beta University from and "
    r"against any and all third-party claims, losses, damages, and expenses "
    r"of every kind arising out of or relating to this agreement or the "
    r"placement programme described herein.\par "
    r"2. Governing Law\par "
    r"This agreement is governed by the laws of the State of California.\par "
    r"3. Signatures\par "
    r"By: Alice Smith\par "
)


def test_project_playbook_caps_search_snippet_for_legacy_store_missing_key(
    tmp_path: Path,
) -> None:
    """A store mined before search_snippet existed (no key in
    observations.jsonl at all) must still compile a capped x_search_snippet.

    Regression guard for issue #95 fix round 1: ``_restore_observations``
    defaults a missing ``search_snippet`` to the UNTRUNCATED ``full_text``
    (``Observation.__post_init__``). ``mine_corpus`` always caps before
    writing (its two ``truncate_search_snippets`` call sites), but
    ``project_playbook`` previously fed the restored observations straight
    into ``compile_clause_positions``/``compile_clause_library`` with no
    cap of its own. Running ``playbook project`` (a documented separate
    workflow step, ``cli.py``'s ``project`` command) against such a legacy
    store therefore emitted the ENTIRE clause text as ``x_search_snippet``
    in the shareable OPF tier instead of a short phrase.
    """
    corpus_dir = tmp_path / "corpus"
    deal_dir = corpus_dir / "deal-001"
    deal_dir.mkdir(parents=True)
    _write_rtf(deal_dir / "v1.rtf", _LEGACY_STORE_BODY)

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

    taxonomy = load_taxonomy(_TAXONOMY_PATH)
    config = load_config(config_path)

    mine_corpus(corpus_dir=corpus_dir, config=config, taxonomy=taxonomy, out_dir=out_dir)

    from playbook_engine.observation_builder import _SEARCH_SNIPPET_MAX

    obs_path = out_dir / "observations.jsonl"
    raw_observations = [
        json.loads(line)
        for line in obs_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert any(len(o["full_text"]) > _SEARCH_SNIPPET_MAX for o in raw_observations), (
        "premise: at least one observation's full_text must exceed the snippet "
        f"cap ({_SEARCH_SNIPPET_MAX}) or this test proves nothing"
    )

    # Simulate a pre-#95 store: strip the search_snippet key entirely, as a
    # store mined by older code would never have written it.
    for o in raw_observations:
        o.pop("search_snippet", None)
    obs_path.write_text("\n".join(json.dumps(o) for o in raw_observations) + "\n", encoding="utf-8")

    playbook = project_playbook(
        out_dir=out_dir, config=config, taxonomy=taxonomy, opf_version="0.3"
    )

    snippet_values = [
        op.get("x_search_snippet")
        for clause in playbook["evidence"]["clauses"]
        for op in clause.get("observed_positions", [])
        if op.get("x_search_snippet")
    ]
    assert snippet_values, "expected at least one compiled x_search_snippet"
    assert all(len(s) <= _SEARCH_SNIPPET_MAX for s in snippet_values), (
        f"every compiled x_search_snippet must be capped at {_SEARCH_SNIPPET_MAX} "
        f"chars, even when restored from a legacy store missing the "
        f"search_snippet key; got lengths={[len(s) for s in snippet_values]}"
    )


# ---------------------------------------------------------------------------
# Issue #216 fix round 2: the origin reference is EVERY template node
# carrying the taxonomy_id, never only the first.
# ---------------------------------------------------------------------------

_MULTI_NODE_INSURANCE_A = "Alpha Corp shall maintain commercial general liability insurance."
_MULTI_NODE_INSURANCE_B = (
    "Alpha Corp shall name Beta University as an additional insured on every policy."
)
_MULTI_NODE_TEMPLATE_BODY = (
    r"1. Indemnification\par "
    r"Alpha Corp shall indemnify Beta University against third-party claims "
    r"arising from the placement programme.\par "
    r"2. Insurance\par "
    rf"{_MULTI_NODE_INSURANCE_A}\par "
    r"3. Insurance\par "
    rf"{_MULTI_NODE_INSURANCE_B}\par "
    r"4. Term\par "
    r"This agreement commences on the date of execution and continues for one year.\par "
)
# The signed copy strikes the template's SECOND insurance node (§3).
_MULTI_NODE_SIGNED_BODY = (
    r"1. Indemnification\par "
    r"Alpha Corp shall indemnify Beta University against third-party claims "
    r"arising from the placement programme.\par "
    r"2. Insurance\par "
    rf"{_MULTI_NODE_INSURANCE_A}\par "
    r"3. Term\par "
    r"This agreement commences on the date of execution and continues for one year.\par "
)


def test_struck_later_node_of_multi_node_standard_is_our_concession(tmp_path: Path) -> None:
    """Our template splits its Insurance standard across two nodes sharing a
    taxonomy_id. Two deals start from our template and strike its SECOND
    insurance node before signing. That struck text is our own standard
    language, so each removal is conceded_before_signing — never a refused
    ask: it is absent from summary.rejected, the digest's unacceptable list
    and the Floor candidates, and neither deal counts as held."""
    from playbook_engine.floor_candidates import derive_reversal_candidates
    from playbook_engine.observation_builder import OUTCOME_CONCEDED_BEFORE_SIGNING

    corpus_dir = tmp_path / "corpus"
    for deal in ("deal-001", "deal-002"):
        deal_dir = corpus_dir / deal
        deal_dir.mkdir(parents=True)
        _write_rtf(deal_dir / "v1.rtf", _MULTI_NODE_TEMPLATE_BODY)
        _write_rtf(deal_dir / "v2.rtf", _MULTI_NODE_SIGNED_BODY)
        (deal_dir / "hints.yaml").write_text("signed_version: v2.rtf\n", encoding="utf-8")
    template_path = tmp_path / "template.rtf"
    _write_rtf(template_path, _MULTI_NODE_TEMPLATE_BODY)
    cfg = {
        "agreement_type": {
            "id": "educational-affiliation",
            "name": "Educational Affiliation Agreement",
        },
        "baseline": {"template": str(template_path)},
        "taxonomy": str(_TAXONOMY_PATH),
        "provenance": {"our_party_aliases": ["Alpha Corp"]},
    }
    config_path = tmp_path / "playbook.config.yaml"
    config_path.write_text(yaml.dump(cfg), encoding="utf-8")
    out_dir = tmp_path / "out"
    taxonomy = load_taxonomy(_TAXONOMY_PATH)
    config = load_config(config_path)

    mine_corpus(corpus_dir=corpus_dir, config=config, taxonomy=taxonomy, out_dir=out_dir)
    obs_lines = (out_dir / "observations.jsonl").read_text(encoding="utf-8").splitlines()
    observations = [json.loads(line) for line in obs_lines if line.strip()]

    template_obs = [
        json.loads(line)
        for line in (out_dir / "template_observations.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert [o["full_text"] for o in template_obs if o["taxonomy_id"] == "insurance"] == [
        _MULTI_NODE_INSURANCE_A,
        _MULTI_NODE_INSURANCE_B,
    ], "premise: the template's insurance standard spans two nodes"

    struck = [o for o in observations if o["full_text"] == _MULTI_NODE_INSURANCE_B]
    assert [(o["citation"]["document_id"], o["outcome"]) for o in struck] == [
        ("deal-001", OUTCOME_CONCEDED_BEFORE_SIGNING),
        ("deal-002", OUTCOME_CONCEDED_BEFORE_SIGNING),
    ]
    assert not [o for o in observations if o["outcome"] == "proposed_then_reversed"]
    assert derive_reversal_candidates(observations, min_deals=2) == []

    playbook = project_playbook(
        out_dir=out_dir, config=config, taxonomy=taxonomy, opf_version="0.3"
    )
    clause = next(c for c in playbook["evidence"]["clauses"] if c["taxonomy_id"] == "insurance")
    assert clause["summary"].get("rejected", []) == []
    assert _MULTI_NODE_INSURANCE_B not in [o.get("full_text") for o in clause["observed_positions"]]
    stance = clause["summary"]["stance_detail"]
    assert (stance["held"], stance["of"]) == (0, 2), stance
    assert clause["summary"]["historical_stance"] != "consistently_held"
    digest_clause = next(
        c for c in playbook["digest"]["clauses"] if c["taxonomy_id"] == "insurance"
    )
    assert digest_clause["unacceptable"] == []

    # OPF 0.4 (issue #223): each deal's struck later node is its opening
    # text, never a refused ask, and neither deal signed our standard.
    playbook_04 = project_playbook(out_dir=out_dir, config=config, taxonomy=taxonomy)
    records = [p for p in playbook_04["evidence"]["precedent"] if p["taxonomy_id"] == "insurance"]
    assert sorted(p["document_id"] for p in records) == ["deal-001", "deal-002"]
    for record in records:
        assert record["refused_asks"] == []
        assert record["opening_text"]["text"] == _MULTI_NODE_INSURANCE_B
        assert record["standard"] is False and record["moved"] is True
    clause_04 = next(
        c for c in playbook_04["evidence"]["clauses"] if c["taxonomy_id"] == "insurance"
    )
    assert (clause_04["n_deals"], clause_04["n_signed_standard"], clause_04["n_refused"]) == (
        2,
        0,
        0,
    )
    digest_04 = next(c for c in playbook_04["digest"]["clauses"] if c["taxonomy_id"] == "insurance")
    assert digest_04["refused_asks"] == []


# Issue #223 fix round 1: one conceded_before_signing row is written per
# struck node, so a deal that strikes TWO nodes of a multi-node standard has
# two rows for the (deal, clause) — the 0.4 precedent must carry both.
_MULTI_NODE_INSURANCE_C = "Alpha Corp shall deliver certificates of insurance on request."
_THREE_NODE_TEMPLATE_BODY = (
    r"1. Indemnification\par "
    r"Alpha Corp shall indemnify Beta University against third-party claims "
    r"arising from the placement programme.\par "
    r"2. Insurance\par "
    rf"{_MULTI_NODE_INSURANCE_A}\par "
    r"3. Insurance\par "
    rf"{_MULTI_NODE_INSURANCE_B}\par "
    r"4. Insurance\par "
    rf"{_MULTI_NODE_INSURANCE_C}\par "
    r"5. Term\par "
    r"This agreement commences on the date of execution and continues for one year.\par "
)


def test_two_struck_nodes_of_multi_node_standard_both_reach_opening_text(
    tmp_path: Path,
) -> None:
    """Our template's Insurance standard spans three nodes (A, B, C); two
    deals start from it and sign keeping only A. Each deal's store has a
    conceded_before_signing row for B and one for C, and its OPF 0.4
    precedent's opening_text carries BOTH struck texts (joined in clause
    order), never only the first row."""
    from playbook_engine.observation_builder import OUTCOME_CONCEDED_BEFORE_SIGNING

    corpus_dir = tmp_path / "corpus"
    for deal in ("deal-001", "deal-002"):
        deal_dir = corpus_dir / deal
        deal_dir.mkdir(parents=True)
        _write_rtf(deal_dir / "v1.rtf", _THREE_NODE_TEMPLATE_BODY)
        _write_rtf(deal_dir / "v2.rtf", _MULTI_NODE_SIGNED_BODY)
        (deal_dir / "hints.yaml").write_text("signed_version: v2.rtf\n", encoding="utf-8")
    template_path = tmp_path / "template.rtf"
    _write_rtf(template_path, _THREE_NODE_TEMPLATE_BODY)
    cfg = {
        "agreement_type": {
            "id": "educational-affiliation",
            "name": "Educational Affiliation Agreement",
        },
        "baseline": {"template": str(template_path)},
        "taxonomy": str(_TAXONOMY_PATH),
        "provenance": {"our_party_aliases": ["Alpha Corp"]},
    }
    config_path = tmp_path / "playbook.config.yaml"
    config_path.write_text(yaml.dump(cfg), encoding="utf-8")
    out_dir = tmp_path / "out"
    taxonomy = load_taxonomy(_TAXONOMY_PATH)
    config = load_config(config_path)

    mine_corpus(corpus_dir=corpus_dir, config=config, taxonomy=taxonomy, out_dir=out_dir)
    obs_lines = (out_dir / "observations.jsonl").read_text(encoding="utf-8").splitlines()
    observations = [json.loads(line) for line in obs_lines if line.strip()]
    for deal in ("deal-001", "deal-002"):
        conceded = sorted(
            o["full_text"]
            for o in observations
            if o["citation"]["document_id"] == deal
            and o["outcome"] == OUTCOME_CONCEDED_BEFORE_SIGNING
        )
        assert conceded == sorted([_MULTI_NODE_INSURANCE_B, _MULTI_NODE_INSURANCE_C]), (
            f"premise: {deal} has one conceded row per struck node; got {conceded!r}"
        )

    playbook = project_playbook(out_dir=out_dir, config=config, taxonomy=taxonomy)
    records = [p for p in playbook["evidence"]["precedent"] if p["taxonomy_id"] == "insurance"]
    assert sorted(p["document_id"] for p in records) == ["deal-001", "deal-002"]
    for record in records:
        opening = record["opening_text"]["text"]
        assert _MULTI_NODE_INSURANCE_B in opening, opening
        assert _MULTI_NODE_INSURANCE_C in opening, opening
        assert opening == f"{_MULTI_NODE_INSURANCE_B}\n{_MULTI_NODE_INSURANCE_C}"
        assert record["signed_text"]["text"] == _MULTI_NODE_INSURANCE_A
        assert record["refused_asks"] == []
        assert record["standard"] is False and record["moved"] is True


# ---------------------------------------------------------------------------
# Issue #218: on the deterministic (venv) path there is no OCR at all. A
# scanned PDF version must fail LOUD — a per-version WARNING naming the Docker
# runtime and a NoOCRRuntimeError row — instead of the silent empty tree the
# #82 guard above used to be the only backstop for.
# ---------------------------------------------------------------------------


def test_scanned_pdf_on_deterministic_path_fails_loud_naming_docker(tmp_path: Path) -> None:
    fpdf = pytest.importorskip("fpdf")

    corpus_dir = tmp_path / "corpus"
    deal_dir = corpus_dir / "deal-001"
    deal_dir.mkdir(parents=True)
    scan = fpdf.FPDF()
    scan.add_page()  # no text layer — what a scan looks like to pdfplumber
    scan.output(str(deal_dir / "v1.pdf"))
    _write_rtf(deal_dir / "v2.rtf", _CORPUS_BODY)

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

    progress_lines: list[str] = []
    mine_corpus(
        corpus_dir=corpus_dir,
        config=load_config(config_path),
        taxonomy=load_taxonomy(_TAXONOMY_PATH),
        out_dir=out_dir,
        progress=progress_lines.append,
    )

    warnings = [line for line in progress_lines if "v1.pdf" in line and "WARNING" in line]
    assert warnings, f"expected a per-version WARNING for v1.pdf; got {progress_lines}"
    assert "Docker runtime" in warnings[0]
    # Issue #218 fix round: the deterministic segmenter has no OCR in ANY
    # runtime (Docker included), so the message must name the segmentation
    # setting that reaches the OCR path, not just the runtime.
    assert "deterministic segmenter has no OCR in any runtime" in warnings[0]
    assert "segmentation.agent: true" in warnings[0]

    manifest = json.loads((out_dir / "corpus_manifest.json").read_text(encoding="utf-8"))
    rows = {r["version"]: r for r in manifest[0]["version_ingest"]}
    assert rows["v1"]["status"] == "failed"
    assert rows["v1"]["error"] == "NoOCRRuntimeError"
    assert rows["v2"]["status"] == "ok"
