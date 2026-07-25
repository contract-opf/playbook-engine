"""Tests for the builtin CUAD-base taxonomy (issue #167).

`spec/taxonomy/cuad-base.yaml` is the GENUINE CUAD v1 41-category list —
nothing added: headings that merely resemble common contract sections
(Indemnification, Confidentiality, Force Majeure, ...) are not CUAD
categories and live in `general-commercial.yaml`, so CUAD provenance is
never claimed falsely. The taxonomy inductor loads its reference lists
from these YAMLs — a single source of truth guarded here.
"""

from __future__ import annotations

import re
from pathlib import Path

from playbook_engine.taxonomy import load_taxonomy

ROOT = Path(__file__).parent.parent
CUAD_BASE = ROOT / "spec" / "taxonomy" / "cuad-base.yaml"
GENERAL_COMMERCIAL = ROOT / "spec" / "taxonomy" / "general-commercial.yaml"

_SLUG_RE = re.compile(r"^[a-z0-9_]+$")


def test_cuad_base_loads_41_active() -> None:
    taxonomy = load_taxonomy(CUAD_BASE)
    assert len(taxonomy.entries) == 41, "CUAD v1 defines exactly 41 categories"
    assert all(e.status == "active" for e in taxonomy.entries)
    ids = [e.id for e in taxonomy.entries]
    assert len(set(ids)) == 41, "ids must be unique"
    assert all(_SLUG_RE.match(i) for i in ids), "ids must be slug-shaped"
    assert all(e.cuad_origin for e in taxonomy.entries), (
        "every cuad-base entry IS a CUAD category — cuad_origin always set"
    )


def test_cuad_base_carries_no_invented_categories() -> None:
    """The historical embedded list inflated CUAD with common headings that
    are not upstream categories — they must never reappear here."""
    taxonomy = load_taxonomy(CUAD_BASE)
    labels = {e.label.lower() for e in taxonomy.entries}
    for interloper in (
        "indemnification",
        "confidentiality",
        "force majeure",
        "severability",
        "entire agreement",
        "limitation of liability",
    ):
        assert interloper not in labels, f"{interloper!r} is not a CUAD v1 category"
    supplemental = load_taxonomy(GENERAL_COMMERCIAL)
    supplemental_labels = {e.label.lower() for e in supplemental.entries}
    assert "indemnification" in supplemental_labels
    assert all(e.cuad_origin is None for e in supplemental.entries), (
        "supplemental entries must never claim CUAD provenance"
    )


def test_inductor_matches_yaml() -> None:
    """Single source of truth: the inductor's reference lists ARE the YAMLs."""
    from playbook_engine.taxonomy_inductor import _CUAD_V1, _SUPPLEMENTAL

    cuad_yaml = [(e.id, e.label) for e in load_taxonomy(CUAD_BASE).entries]
    supplemental_yaml = [(e.id, e.label) for e in load_taxonomy(GENERAL_COMMERCIAL).entries]
    assert list(_CUAD_V1) == cuad_yaml
    assert list(_SUPPLEMENTAL) == supplemental_yaml


def test_classify_against_cuad_base(tmp_path: Path) -> None:
    """A fixture corpus classifies end-to-end with taxonomy: builtin:cuad-base
    (stub judges — fully offline)."""
    import json

    import yaml

    from playbook_engine.config import load_config
    from playbook_engine.pipeline import compile_corpus

    corpus = tmp_path / "corpus"
    deal = corpus / "deal-one"
    deal.mkdir(parents=True)
    rtf = (
        r"{\rtf1\ansi\deff0{\fonttbl{\f0\froman\fcharset0 Times New Roman;}}\f0\fs24 "
        r"1. Governing Law\par This Agreement is governed by Delaware law.\par "
        r"2. Insurance\par Each party maintains commercial general liability coverage.\par "
        r"3. Signatures\par By: A. Person, FixtureCorp\par By: B. Person, Counterparty\par }"
    )
    (deal / "v1.rtf").write_text(rtf, encoding="utf-8")

    cfg = {
        "agreement_type": {"id": "general-commercial", "name": "General Commercial Agreement"},
        "baseline": {"template": None},
        "taxonomy": "builtin:cuad-base.yaml",
        "provenance": {"our_party_aliases": ["FixtureCorp"]},
    }
    config_path = tmp_path / "playbook.config.yaml"
    config_path.write_text(yaml.dump(cfg), encoding="utf-8")

    config = load_config(config_path)
    taxonomy = load_taxonomy(config.taxonomy_path)
    out_dir = tmp_path / "out"
    compile_corpus(corpus_dir=corpus, config=config, taxonomy=taxonomy, out_dir=out_dir)

    playbook = json.loads((out_dir / "playbook.opf.json").read_text(encoding="utf-8"))
    classified = {c["taxonomy_id"] for c in playbook["evidence"]["clauses"]}
    assert "governing_law" in classified or "insurance" in classified, (
        f"expected CUAD-base classification, got {classified}"
    )


def test_attribution_header_present() -> None:
    for path in (CUAD_BASE, GENERAL_COMMERCIAL):
        text = path.read_text(encoding="utf-8")
        assert "CC-BY-4.0" in text, f"{path.name} missing license attribution"
        assert "Atticus" in text, f"{path.name} missing CUAD attribution"
