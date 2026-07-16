"""End-to-end regression guard: real compiled playbook (OPF v0.2) → consumers.

Issue #140 changed ``playbook_assembler`` to emit OPF **v0.2** documents
(clauses under ``evidence``, per-clause ``summary.historical_stance`` instead
of ``rollup.position``). ``aar.py`` (``playbook report``) and ``viewer.py``
(``playbook view render``) previously read the v0.1 shape and would silently
degrade to empty output against a real compiled playbook — no existing test
caught it, because both consumers' own suites write hand-authored v0.1
fixtures to disk rather than compiling one.

This test closes that gap by driving the *real* ``mine_corpus`` →
``project_playbook`` path (which runs the actual assembler) to produce a
genuine ``playbook.opf.json``, then feeding that exact file into the AAR and
the review viewer. It asserts the compiled document really is v0.2 (so the
guard can never quietly revert to exercising a v0.1 doc) and that both
consumers surface its clauses rather than emitting empty output.

SECURITY NOTE: All fixtures are programmatically constructed RTF with
synthetic, fictional content only (e.g. "Alpha Corp", "Beta University"). No
real agreement files are referenced.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from playbook_engine.aar import build_after_action_data, build_after_action_report
from playbook_engine.config import load_config
from playbook_engine.opf_accessors import playbook_clauses
from playbook_engine.pipeline import mine_corpus, project_playbook
from playbook_engine.taxonomy import load_taxonomy
from playbook_engine.viewer import render_review_html

_TAXONOMY_PATH = Path(__file__).parent.parent / "spec" / "taxonomy" / "affiliation-agreement.yaml"

_RTF_PROLOGUE = (
    r"{\rtf1\ansi\deff0"
    r"{\fonttbl{\f0\froman\fcharset0 Times New Roman;}}"
    r"\f0\fs24 "
)
_RTF_EPILOGUE = r"}"

# Two versions with a real clause-text change (governing law: California ->
# Delaware) so the deal document has genuine mined observations to compile.
_CORPUS_BODY_V1 = (
    r"1. Indemnification\par "
    r"Alpha Corp shall indemnify Beta University against third-party claims "
    r"arising from the placement programme.\par "
    r"2. Governing Law\par "
    r"This agreement is governed by the laws of the State of California.\par "
    r"3. Term\par "
    r"This agreement commences on the date of execution and continues for one year.\par "
)
_CORPUS_BODY_V2 = _CORPUS_BODY_V1.replace("State of California", "State of Delaware")

_TEMPLATE_BODY = (
    r"1. Indemnification\par "
    r"The service provider shall indemnify the institution against third-party claims.\par "
    r"2. Governing Law\par "
    r"This agreement is governed by the laws of the State of New York.\par "
    r"3. Term\par "
    r"Initial term of one year with automatic renewal.\par "
)


def _write_rtf(path: Path, body: str) -> None:
    path.write_text(_RTF_PROLOGUE + body + _RTF_EPILOGUE, encoding="utf-8")


def _compile_real_playbook(tmp_path: Path) -> Path:
    """Run the real mine → project pipeline; return the ``out`` dir holding the
    compiled ``playbook.opf.json`` (a genuine OPF v0.2 document)."""
    corpus_dir = tmp_path / "corpus"
    deal_dir = corpus_dir / "deal-001"
    deal_dir.mkdir(parents=True)
    _write_rtf(deal_dir / "v1.rtf", _CORPUS_BODY_V1)
    _write_rtf(deal_dir / "v2.rtf", _CORPUS_BODY_V2)

    template_dir = tmp_path / "template"
    template_dir.mkdir()
    template_path = template_dir / "template.rtf"
    _write_rtf(template_path, _TEMPLATE_BODY)

    cfg_dict = {
        "agreement_type": {
            "id": "educational-affiliation",
            "name": "Educational Affiliation Agreement",
        },
        "baseline": {"template": str(template_path)},
        "taxonomy": str(_TAXONOMY_PATH),
        "provenance": {"our_party_aliases": ["Alpha Corp"]},
    }
    config_path = tmp_path / "playbook.config.yaml"
    config_path.write_text(yaml.dump(cfg_dict), encoding="utf-8")

    out_dir = tmp_path / "out"
    cfg = load_config(config_path)
    taxonomy = load_taxonomy(_TAXONOMY_PATH)

    # No judges passed — the CLI-default (fully deterministic) path.
    mine_corpus(corpus_dir=corpus_dir, config=cfg, taxonomy=taxonomy, out_dir=out_dir)
    project_playbook(out_dir=out_dir, config=cfg, taxonomy=taxonomy)
    return out_dir


def test_real_compile_emits_v02_with_evidence_clauses(tmp_path: Path) -> None:
    """Sanity anchor: the compiled document is OPF v0.2 with non-empty
    ``evidence.clauses`` — so the consumer assertions below are genuinely
    exercising the v0.2 shape (not a silently-reverted v0.1 one)."""
    out_dir = _compile_real_playbook(tmp_path)
    doc = json.loads((out_dir / "playbook.opf.json").read_text(encoding="utf-8"))

    assert doc["opf_version"] == "0.2"
    assert doc["evidence"]["clauses"], "compiled v0.2 playbook must have evidence.clauses"
    # The legacy top-level key must NOT exist — proves the regression is real:
    # a consumer reading doc["clauses"] would get nothing.
    assert "clauses" not in doc
    assert playbook_clauses(doc), "playbook_clauses must read the v0.2 evidence shape"


def test_report_reads_real_v02_playbook(tmp_path: Path) -> None:
    """``playbook report`` against a real compiled (v0.2) playbook surfaces its
    clauses — the stance histogram and clause count are populated, not empty."""
    out_dir = _compile_real_playbook(tmp_path)
    doc = json.loads((out_dir / "playbook.opf.json").read_text(encoding="utf-8"))
    n_clauses = len(doc["evidence"]["clauses"])

    data = build_after_action_data(out_dir)
    sem = data["semantic_coverage"]
    assert sem["total_clauses_in_playbook"] == n_clauses
    # Histogram is keyed by v0.2 historical_stance values and sums to the
    # clause count — before the fix it was empty (doc["clauses"] == []).
    hist = sem["rollup_position_histogram"]
    assert sum(hist.values()) == n_clauses
    assert "unknown" not in hist, (
        "every clause must yield a real stance, not the missing-key fallback"
    )

    # And the rendered Markdown actually shows the histogram section.
    report = build_after_action_report(out_dir)
    assert "Rollup-position histogram" in report


def test_view_render_reads_real_v02_playbook(tmp_path: Path) -> None:
    """``playbook view render`` against a real compiled (v0.2) playbook emits
    clause cards with their titles — not an empty review surface."""
    out_dir = _compile_real_playbook(tmp_path)
    doc = json.loads((out_dir / "playbook.opf.json").read_text(encoding="utf-8"))

    html = render_review_html(out_dir)
    # Every compiled clause title appears in the rendered review.
    titles = [c["title"] for c in doc["evidence"]["clauses"]]
    assert titles, "fixture must compile at least one clause"
    for title in titles:
        assert title in html, f"clause title {title!r} missing from rendered viewer HTML"
    # C1 numbering proves at least one clause card was rendered (empty index → no C1).
    assert "C1" in html
