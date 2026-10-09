"""End-to-end regression guard: real compiled playbook (OPF 0.4) → consumers.

Issue #140 showed that consumers reading a hand-authored fixture shape can
silently degrade to empty output against a real compiled playbook. This test
drives the *real* ``mine_corpus`` → ``project_playbook`` path (which runs
the actual assembler) to produce a genuine ``playbook.opf.json`` — OPF 0.4,
the one format the engine emits (issue #238) — then feeds that exact file
into the bundle (the one human-readable artifact), and asserts it surfaces
its clauses rather than emitting empty output.

SECURITY NOTE: All fixtures are programmatically constructed RTF with
synthetic, fictional content only (e.g. "Alpha Corp", "Beta University"). No
real agreement files are referenced.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from playbook_engine.config import load_config
from playbook_engine.document_renderer import render_bundle_html
from playbook_engine.opf_accessors import playbook_clauses
from playbook_engine.pipeline import mine_corpus, project_playbook
from playbook_engine.taxonomy import load_taxonomy

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


def _compile_real_playbook(tmp_path: Path, *, signed: bool = False) -> Path:
    """Run the real mine → project pipeline; return the ``out`` dir holding the
    compiled ``playbook.opf.json`` (a genuine OPF 0.4 document)."""
    corpus_dir = tmp_path / "corpus"
    deal_dir = corpus_dir / "deal-001"
    deal_dir.mkdir(parents=True)
    _write_rtf(deal_dir / "v1.rtf", _CORPUS_BODY_V1)
    _write_rtf(deal_dir / "v2.rtf", _CORPUS_BODY_V2)
    if signed:
        # The same hints.yaml route a user takes to name the executed copy.
        (deal_dir / "hints.yaml").write_text("signed_version: v2.rtf\n", encoding="utf-8")

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


def test_real_compile_emits_v04_with_evidence_clauses(tmp_path: Path) -> None:
    """Sanity anchor: the compiled document is OPF 0.4 with non-empty
    ``evidence.clauses`` and ``evidence.precedent``."""
    out_dir = _compile_real_playbook(tmp_path)
    doc = json.loads((out_dir / "playbook.opf.json").read_text(encoding="utf-8"))

    # Issue #223: the default compile is OPF 0.4 (evidence.clauses + precedent).
    assert doc["opf_version"] == "0.4"
    assert doc["evidence"]["precedent"], "compiled 0.4 playbook must carry precedent"
    assert doc["evidence"]["clauses"], "compiled playbook must have evidence.clauses"
    # The legacy top-level key must NOT exist — proves the regression is real:
    # a consumer reading doc["clauses"] would get nothing.
    assert "clauses" not in doc
    assert playbook_clauses(doc), "playbook_clauses must read the evidence shape"


def test_bundle_reads_real_v04_playbook(tmp_path: Path) -> None:
    """``playbook view bundle`` against a real compiled playbook emits every
    clause's title — not an empty document — and embeds the canonical OPF
    JSON and the digest the playbook carries."""
    out_dir = _compile_real_playbook(tmp_path)
    doc = json.loads((out_dir / "playbook.opf.json").read_text(encoding="utf-8"))

    html = render_bundle_html(out_dir)
    titles = [c["title"] for c in doc["evidence"]["clauses"]]
    assert titles, "fixture must compile at least one clause"
    for title in titles:
        assert title in html, f"clause title {title!r} missing from rendered bundle HTML"
    assert 'id="opf-canonical"' in html and 'id="opf-digest"' in html
    n_digest_clauses = len(doc["digest"]["clauses"])
    assert n_digest_clauses == len(doc["evidence"]["clauses"])


def test_bundle_shows_real_v04_precedent(tmp_path: Path) -> None:
    """Issue #223: the readable bundle shows a real 0.4 compile's signed
    variants (here the Delaware governing-law text the deal signed instead of
    our New York standard) — not an empty clause card."""
    out_dir = _compile_real_playbook(tmp_path, signed=True)
    doc = json.loads((out_dir / "playbook.opf.json").read_text(encoding="utf-8"))
    signed = [
        p
        for p in doc["evidence"]["precedent"]
        if p["taxonomy_id"] == "governing_law" and p["signed_text"]
    ]
    assert signed and signed[0]["signed"] is True
    assert "Delaware" in signed[0]["signed_text"]["text"], "premise: the deal signed Delaware"

    html = render_bundle_html(out_dir)
    assert "Delaware" in html
    assert "Signed variants" in html
