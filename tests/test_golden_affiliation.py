"""Golden test: compile a synthetic affiliation corpus end-to-end.

This test locks in deterministic pipeline behavior for the educational
affiliation agreement type.  It uses a programmatically-constructed
synthetic corpus — the real corpus is gitignored and never committed.

SECURITY NOTE: All fixtures are programmatically generated with synthetic
text.  No real agreement files are committed or referenced.  Fictional
party/author names only (e.g. "Alpha Corp", "Beta University").

What is asserted:
  Structural (deterministic)
    - Schema-valid playbook emitted.
    - corpus.stats matches the fixture corpus layout.
    - trail/ files record correct version counts.
    - Observations produced (N > 0).
    - Pipeline is deterministic: two runs produce identical observations.jsonl.
    - Reversal detection: v1→v2 change reversed in v3 → outcome = "proposed_then_reversed".
    - Out-of-scope document excluded with rationale (using a keyword-based stub judge).

  LLM-stable properties (stub judges used, so exact text not asserted)
    - Every scope decision has in_scope, scope_rationale, scope_confidence.
    - Every observation has all required OPF citation fields.
    - Inspection report renders all documents including the excluded one.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from playbook_engine.config import load_config
from playbook_engine.inspection_report import build_inspection_report
from playbook_engine.pipeline import compile_corpus
from playbook_engine.scope_gate import ScopeDecision
from playbook_engine.taxonomy import load_taxonomy
from playbook_engine.validator import validate_document

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


# ---------------------------------------------------------------------------
# Synthetic clause text (fictional parties: Alpha Corp, Beta University, etc.)
# ---------------------------------------------------------------------------

# deal-alpha: 3-version agreement with a reversal.
# v1: one-way indemnification (Alpha Corp → Beta University)
# v2: counterparty redline to mutual indemnification
# v3: signed — reverts to one-way (reversal!)
_ALPHA_V1 = (
    r"1. Indemnification\par "
    r"Alpha Corp shall indemnify and hold harmless Beta University and its officers "
    r"from and against all third-party claims arising from placement activities.\par "
    r"2. Governing Law\par "
    r"This Agreement shall be governed by the laws of the State of California.\par "
    r"3. Term\par "
    r"This Agreement commences upon execution and continues for one academic year.\par "
    r"4. Termination\par "
    r"Either party may terminate this Agreement upon thirty days written notice.\par "
)

_ALPHA_V2 = (
    r"1. Indemnification\par "
    r"The parties shall mutually indemnify and hold harmless each other and their respective "
    r"officers from and against all third-party claims arising from placement activities.\par "
    r"2. Governing Law\par "
    r"This Agreement shall be governed by the laws of the State of California.\par "
    r"3. Term\par "
    r"This Agreement commences upon execution and continues for one academic year.\par "
    r"4. Termination\par "
    r"Either party may terminate this Agreement upon sixty days written notice.\par "
)

# v3 reverts Indemnification to v1 language (reversal of v2's mutual indemnification).
# A "Signatures" section with filled By: lines is added so detect_signed()
# identifies this as the executed copy and anchors it at the end of the chain.
_ALPHA_V3 = (
    r"1. Indemnification\par "
    r"Alpha Corp shall indemnify and hold harmless Beta University and its officers "
    r"from and against all third-party claims arising from placement activities.\par "
    r"2. Governing Law\par "
    r"This Agreement shall be governed by the laws of the State of California.\par "
    r"3. Term\par "
    r"This Agreement commences upon execution and continues for one academic year.\par "
    r"4. Termination\par "
    r"Either party may terminate this Agreement upon sixty days written notice.\par "
    r"5. Signatures\par "
    r"By: Alice Johnson, VP Operations, Alpha Corp\par "
    r"By: Robert Chen, Provost, Beta University\par "
)

# deal-beta: standard two-version affiliation agreement
_BETA_V1 = (
    r"1. Indemnification\par "
    r"Gamma College shall not be held liable for claims arising from student conduct "
    r"during clinical placements at Alpha Corp facilities.\par "
    r"2. Governing Law\par "
    r"This Agreement shall be governed by the laws of the State of Texas.\par "
    r"3. Term\par "
    r"Initial term of two years with automatic renewal unless either party provides "
    r"sixty days written notice of non-renewal.\par "
    r"4. Insurance\par "
    r"Students shall maintain professional liability insurance as required by Alpha Corp.\par "
)

_BETA_V2 = (
    r"1. Indemnification\par "
    r"Gamma College shall not be held liable for claims arising from student conduct "
    r"during clinical placements at Alpha Corp facilities.\par "
    r"2. Governing Law\par "
    r"This Agreement shall be governed by the laws of the State of New York.\par "
    r"3. Term\par "
    r"Initial term of two years with automatic renewal unless either party provides "
    r"ninety days written notice of non-renewal.\par "
    r"4. Insurance\par "
    r"Students shall maintain professional liability insurance as required by Alpha Corp. "
    r"Alpha Corp shall maintain general liability insurance of not less than one million dollars.\par "
)

# deal-data-sharing: deliberately off-type document (data-sharing agreement).
# A real LLM scope judge would exclude this; the test uses a keyword-based stub.
_DATA_SHARING_V1 = (
    r"1. Purpose\par "
    r"This Data Sharing Agreement governs the exchange of de-identified health data "
    r"between Alpha Corp and Delta Research Institute.\par "
    r"2. Data Types\par "
    r"Parties may share anonymized patient outcome data for research purposes only.\par "
    r"3. Term\par "
    r"This agreement is effective for a period of three years from execution.\par "
)


# ---------------------------------------------------------------------------
# Keyword-based scope judge (deterministic stub that excludes data-sharing docs)
# ---------------------------------------------------------------------------


class _KeywordScopeJudge:
    """Deterministic scope judge: excludes docs whose text contains 'data sharing agreement'."""

    def judge(self, tree: Any, agreement_type: str) -> ScopeDecision:
        all_text = " ".join(
            getattr(n, "text", "") for n in (tree.nodes if hasattr(tree, "nodes") else [])
        ).lower()
        if "data sharing agreement" in all_text or "data sharing" in all_text:
            return ScopeDecision(
                in_scope=False,
                scope_rationale="Document appears to be a data sharing agreement, not an affiliation agreement.",
                scope_confidence=0.9,
                basis="judge",
            )
        return ScopeDecision(
            in_scope=True,
            scope_rationale="Document appears to be an educational affiliation agreement.",
            scope_confidence=0.85,
            basis="judge",
        )


# ---------------------------------------------------------------------------
# Corpus + config fixture
# ---------------------------------------------------------------------------


def _make_golden_corpus(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Build the golden synthetic corpus; return (corpus_dir, config_path, out_dir)."""
    corpus_dir = tmp_path / "corpus"

    # deal-alpha: 3 versions with a reversal.
    # v3 carries a Signatures section (two filled By: lines) so detect_signed()
    # anchors it as the executed copy; the exhaustive chain then resolves to
    # v1→v2→v3 from content distances alone.
    # hints.yaml is present as an advisory (redundant for this fixture, but
    # exercises the hints-loading path as a belt-and-suspenders safeguard).
    alpha = corpus_dir / "deal-alpha"
    alpha.mkdir(parents=True)
    _write_rtf(alpha / "v1.rtf", _ALPHA_V1)
    _write_rtf(alpha / "v2.rtf", _ALPHA_V2)
    _write_rtf(alpha / "v3.rtf", _ALPHA_V3)
    (alpha / "hints.yaml").write_text("order:\n  - v1\n  - v2\n  - v3\n", encoding="utf-8")

    # deal-beta: 2 versions, normal progression
    beta = corpus_dir / "deal-beta"
    beta.mkdir()
    _write_rtf(beta / "v1.rtf", _BETA_V1)
    _write_rtf(beta / "v2.rtf", _BETA_V2)

    # deal-data-sharing: off-type document — will be excluded by _KeywordScopeJudge
    data_sharing = corpus_dir / "deal-data-sharing"
    data_sharing.mkdir()
    _write_rtf(data_sharing / "v1.rtf", _DATA_SHARING_V1)

    taxonomy_path = (
        Path(__file__).parent.parent / "spec" / "taxonomy" / "affiliation-agreement.yaml"
    )
    cfg = {
        "agreement_type": {
            "id": "educational-affiliation",
            "name": "Educational Affiliation Agreement",
        },
        "baseline": {"template": None},
        "taxonomy": str(taxonomy_path),
        "provenance": {"our_party_aliases": ["Alpha Corp"]},
    }
    config_path = tmp_path / "playbook.config.yaml"
    config_path.write_text(yaml.dump(cfg), encoding="utf-8")

    out_dir = tmp_path / "out"
    return corpus_dir, config_path, out_dir


# ---------------------------------------------------------------------------
# Golden assertions helpers
# ---------------------------------------------------------------------------


def _run_pipeline(corpus_dir: Path, config_path: Path, out_dir: Path, **kwargs: Any) -> dict:
    """Run compile_corpus and return the compiled playbook dict."""
    cfg = load_config(config_path)
    taxonomy = load_taxonomy(cfg.taxonomy_path)
    compile_corpus(
        corpus_dir=corpus_dir,
        config=cfg,
        taxonomy=taxonomy,
        out_dir=out_dir,
        scope_judge=_KeywordScopeJudge(),
        **kwargs,
    )
    return json.loads((out_dir / "playbook.opf.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Golden tests
# ---------------------------------------------------------------------------


def test_golden_schema_valid(tmp_path: Path) -> None:
    """The compiled playbook passes OPF schema validation."""
    corpus_dir, config_path, out_dir = _make_golden_corpus(tmp_path)
    playbook = _run_pipeline(corpus_dir, config_path, out_dir)
    result = validate_document(playbook)
    blocking = [str(e) for e in result.errors if e.blocking]
    assert blocking == [], f"Schema validation errors: {blocking}"


def test_golden_corpus_stats(tmp_path: Path) -> None:
    """Corpus stats: 3 documents total, 2 in scope (data-sharing excluded), 6 version files."""
    corpus_dir, config_path, out_dir = _make_golden_corpus(tmp_path)
    playbook = _run_pipeline(corpus_dir, config_path, out_dir)
    stats = playbook["corpus"]["stats"]
    assert stats["documents_total"] == 3
    assert stats["documents_in_scope"] == 2
    assert stats["versions_total"] == 6  # v1+v2+v3 (alpha) + v1+v2 (beta) + v1 (data-sharing)


def test_golden_trail_files_written(tmp_path: Path) -> None:
    """Trail files are written only for in-scope documents."""
    corpus_dir, config_path, out_dir = _make_golden_corpus(tmp_path)
    _run_pipeline(corpus_dir, config_path, out_dir)
    # Only in-scope docs get a trail
    trail_files = list((out_dir / "trail").glob("*.json"))
    trail_doc_ids = {f.stem for f in trail_files}
    assert "deal-alpha" in trail_doc_ids
    assert "deal-beta" in trail_doc_ids
    assert "deal-data-sharing" not in trail_doc_ids


def test_golden_observations_produced(tmp_path: Path) -> None:
    """Observations.jsonl contains at least one observation from each in-scope document."""
    corpus_dir, config_path, out_dir = _make_golden_corpus(tmp_path)
    _run_pipeline(corpus_dir, config_path, out_dir)
    obs_path = out_dir / "observations.jsonl"
    assert obs_path.exists()
    observations = [json.loads(line) for line in obs_path.read_text().splitlines() if line.strip()]
    assert len(observations) > 0
    doc_ids = {o["citation"]["document_id"] for o in observations}
    assert "deal-alpha" in doc_ids
    assert "deal-beta" in doc_ids
    assert "deal-data-sharing" not in doc_ids  # excluded


def test_golden_out_of_scope_doc_in_scope_json(tmp_path: Path) -> None:
    """The data-sharing agreement appears in scope.json with in_scope=False."""
    corpus_dir, config_path, out_dir = _make_golden_corpus(tmp_path)
    _run_pipeline(corpus_dir, config_path, out_dir)
    scope = json.loads((out_dir / "scope.json").read_text())
    by_id = {d["document_id"]: d for d in scope["documents"]}
    assert "deal-data-sharing" in by_id
    ds = by_id["deal-data-sharing"]
    assert ds["in_scope"] is False
    assert "data sharing" in ds["scope_rationale"].lower()
    assert ds["scope_confidence"] > 0


def test_golden_scope_decisions_have_required_fields(tmp_path: Path) -> None:
    """Every scope decision has in_scope, scope_rationale, and scope_confidence."""
    corpus_dir, config_path, out_dir = _make_golden_corpus(tmp_path)
    _run_pipeline(corpus_dir, config_path, out_dir)
    scope = json.loads((out_dir / "scope.json").read_text())
    for doc in scope["documents"]:
        assert "in_scope" in doc
        assert "scope_rationale" in doc
        assert isinstance(doc["scope_rationale"], str)
        assert len(doc["scope_rationale"]) > 0
        assert "scope_confidence" in doc
        assert 0.0 <= doc["scope_confidence"] <= 1.0


def test_golden_observations_have_required_fields(tmp_path: Path) -> None:
    """Every observation has all required OPF citation fields."""
    corpus_dir, config_path, out_dir = _make_golden_corpus(tmp_path)
    _run_pipeline(corpus_dir, config_path, out_dir)
    obs_path = out_dir / "observations.jsonl"
    for line in obs_path.read_text().splitlines():
        if not line.strip():
            continue
        obs = json.loads(line)
        assert "observation_id" in obs
        assert "text_summary" in obs
        assert "deviation" in obs
        assert "risk_delta" in obs
        assert "direction" in obs["risk_delta"]
        assert "magnitude" in obs["risk_delta"]
        citation = obs["citation"]
        assert "document_id" in citation
        assert "version" in citation
        assert "clause_path" in citation


def test_golden_reversal_detected(tmp_path: Path) -> None:
    """Reversal in deal-alpha: v2's mutual indemnification was reversed in v3.

    The v2 observation for the modified indemnification clause should have
    outcome='proposed_then_reversed'.  This is a purely deterministic assertion.
    """
    corpus_dir, config_path, out_dir = _make_golden_corpus(tmp_path)
    _run_pipeline(corpus_dir, config_path, out_dir)
    obs_path = out_dir / "observations.jsonl"
    observations = [json.loads(line) for line in obs_path.read_text().splitlines() if line.strip()]
    alpha_obs = [o for o in observations if o["citation"]["document_id"] == "deal-alpha"]
    reversed_obs = [o for o in alpha_obs if o.get("outcome") == "proposed_then_reversed"]
    assert len(reversed_obs) > 0, (
        "Expected at least one 'proposed_then_reversed' observation in deal-alpha. "
        "The v2 mutual-indemnification clause should have been detected as reversed "
        "when v3 reverted to v1's one-way language. "
        f"deal-alpha outcomes: {[o.get('outcome') for o in alpha_obs]}"
    )


def test_golden_pipeline_deterministic(tmp_path: Path) -> None:
    """Two runs on the same corpus produce identical observations.jsonl content."""
    corpus_dir, config_path, out_dir = _make_golden_corpus(tmp_path)
    # First run
    _run_pipeline(corpus_dir, config_path, out_dir)
    first_content = (out_dir / "observations.jsonl").read_text(encoding="utf-8")

    # Second run (force re-run to bypass resume cache)
    out_dir2 = tmp_path / "out2"
    _run_pipeline(corpus_dir, config_path, out_dir2)
    second_content = (out_dir2 / "observations.jsonl").read_text(encoding="utf-8")

    assert first_content == second_content, "Pipeline is not deterministic"


def test_golden_cache_hit_produces_identical_content(tmp_path: Path) -> None:
    """Second run with the content-addressed cache produces byte-identical observations."""
    corpus_dir, config_path, out_dir = _make_golden_corpus(tmp_path)
    _run_pipeline(corpus_dir, config_path, out_dir)
    first_content = (out_dir / "observations.jsonl").read_text(encoding="utf-8")

    # Second run — cache hits should produce the same content.
    _run_pipeline(corpus_dir, config_path, out_dir)
    second_content = (out_dir / "observations.jsonl").read_text(encoding="utf-8")
    assert second_content == first_content, (
        "observations.jsonl must be byte-identical on a cache-hit second run"
    )


def test_golden_inspection_report_renderable(tmp_path: Path) -> None:
    """The inspection report renders all 3 documents including the excluded one."""
    corpus_dir, config_path, out_dir = _make_golden_corpus(tmp_path)
    _run_pipeline(corpus_dir, config_path, out_dir)
    report = build_inspection_report(out_dir)
    assert "## deal-alpha" in report
    assert "## deal-beta" in report
    assert "## deal-data-sharing" in report
    assert "out of scope" in report
    assert "data sharing" in report.lower()


def test_golden_playbook_opf_version(tmp_path: Path) -> None:
    """Compiled playbook has opf_version='0.2'."""
    corpus_dir, config_path, out_dir = _make_golden_corpus(tmp_path)
    playbook = _run_pipeline(corpus_dir, config_path, out_dir)
    assert playbook["opf_version"] == "0.2"


def test_golden_playbook_agreement_type(tmp_path: Path) -> None:
    """Playbook agreement_type matches config."""
    corpus_dir, config_path, out_dir = _make_golden_corpus(tmp_path)
    playbook = _run_pipeline(corpus_dir, config_path, out_dir)
    assert playbook["agreement_type"]["id"] == "educational-affiliation"


def test_golden_key_clauses_present_with_citations(tmp_path: Path) -> None:
    """Key clause taxonomy_ids appear in the compiled playbook with observed positions.

    The classifier's Jaccard fast-path (not the null judge) classifies clauses
    deterministically against the affiliation taxonomy, so taxonomy_ids are present
    even in stub mode.  This asserts the third 'known truth' from the issue scope:
    key clauses present with citations.
    """
    corpus_dir, config_path, out_dir = _make_golden_corpus(tmp_path)
    playbook = _run_pipeline(corpus_dir, config_path, out_dir)
    clause_ids = {c["taxonomy_id"] for c in playbook.get("evidence", {}).get("clauses", [])}
    # At minimum, the indemnification and governing_law clauses should be classified.
    expected = {"indemnification", "governing_law"}
    missing = expected - clause_ids
    assert not missing, (
        f"Expected taxonomy_ids {expected} in compiled playbook clauses; "
        f"missing: {missing}. Present: {sorted(clause_ids)}"
    )


def test_golden_negotiation_dynamics(tmp_path: Path) -> None:
    """Negotiation dynamics (issue #177, OPF §3.5.3) on the golden corpus.

    deal-alpha is a 3-version negotiation, so its per-round diffs must
    surface as negotiation_trail entries (round_moves.jsonl at L4, grouped
    into clauses at L5), every clause summary carries a stance_detail whose
    held-rate is arithmetically sane, and changed clauses carry proposed_by
    ("unknown" here — the RTF corpus has no tracked-changes side-channel;
    dynamics are derived, never fabricated).
    """
    corpus_dir, config_path, out_dir = _make_golden_corpus(tmp_path)
    playbook = _run_pipeline(corpus_dir, config_path, out_dir)

    assert (out_dir / "round_moves.jsonl").exists(), "L4 must persist round moves"

    clauses = playbook["evidence"]["clauses"]
    assert clauses
    for clause in clauses:
        detail = clause["summary"].get("stance_detail")
        assert detail is not None, f"stance_detail missing on {clause['id']}"
        assert 0 <= detail["held"] <= detail["of"]
        assert detail["basis"] in {"our_paper", "all"}

    trails = [c for c in clauses if c.get("negotiation_trail")]
    assert trails, "multi-round deal-alpha must produce at least one negotiation_trail"
    for clause in trails:
        for entry in clause["negotiation_trail"]:
            assert entry["round"] >= 1
            assert entry["moved_by"] in {"us", "counterparty", "unknown"}
            assert entry["change_summary"]
            assert entry["ref"]["document_id"]

    # RTF corpus → no tracked changes → changed clauses are 'unknown', and
    # no observation may carry a fabricated observed_at.
    for clause in clauses:
        for obs in clause["observed_positions"]:
            assert obs.get("proposed_by") in {None, "us", "counterparty", "unknown"}
            assert obs.get("observed_at") is None or len(obs["observed_at"]) == 10
