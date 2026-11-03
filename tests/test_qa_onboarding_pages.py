"""QA-2026-11-03 onboarding/page fixes (#12, #19, #21, #43).

SECURITY NOTE: synthetic text only; fictional parties.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from playbook_engine import extraction
from playbook_engine.cli import cli
from playbook_engine.config import load_config
from playbook_engine.corpus_linter import lint_corpus
from playbook_engine.document_renderer import INDEX_FILENAME
from playbook_engine.pipeline import mine_corpus, project_playbook
from playbook_engine.taxonomy import load_taxonomy
from playbook_engine.validator import validate_document
from tests.test_pipeline_llm_seg import _TAXONOMY_PATH, _blank_scan_pdf, _make_corpus

_NDA = Path(__file__).resolve().parent.parent / "examples" / "nda" / "playbook.opf.json"


def _page_text(out: Path) -> str:
    result = CliRunner().invoke(cli, ["view", "bundle", str(out)])
    assert result.exit_code == 0, result.output
    html = (out / INDEX_FILENAME).read_text(encoding="utf-8")
    # drop the machine-readable payload blocks: only the rendered page counts
    return "\n".join(
        part.split("</script>", 1)[-1] if i else part
        for i, part in enumerate(html.split('<script type="application/json"'))
    )


# ---------------------------------------------------------------------------
# #12: a deal whose every file fails extraction is counted, not dropped
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not extraction.shutil.which("pandoc"), reason="pandoc not installed")
def test_all_failed_deal_is_quarantined_in_the_corpus_and_counted_on_the_page(
    tmp_path: Path,
) -> None:
    corpus_dir, config_path, out_dir = _make_corpus(tmp_path, two_versions=True)
    scan_dir = corpus_dir / "deal-scan"
    scan_dir.mkdir()
    _blank_scan_pdf(scan_dir / "v1.pdf")
    _blank_scan_pdf(scan_dir / "v2.pdf")

    taxonomy = load_taxonomy(_TAXONOMY_PATH)
    cfg = load_config(config_path)
    mine_corpus(corpus_dir=corpus_dir, config=cfg, taxonomy=taxonomy, out_dir=out_dir)

    quarantine = json.loads((out_dir / "quarantine.json").read_text(encoding="utf-8"))
    assert [q["document_id"] for q in quarantine] == ["deal-scan"]
    manifest = {
        d["document_id"]: d
        for d in json.loads((out_dir / "corpus_manifest.json").read_text(encoding="utf-8"))
    }
    scan = manifest["deal-scan"]
    assert scan["x_quarantined"] is True and scan["in_scope"] is False
    assert [v["status"] for v in scan["version_ingest"]] == ["failed", "failed"]

    playbook = project_playbook(out_dir=out_dir, config=cfg, taxonomy=taxonomy)
    assert validate_document(playbook).ok
    assert playbook["corpus"]["stats"]["documents_total"] == 2
    text = _page_text(out_dir)
    assert "2 version file(s)" in text and "failed extraction" in text
    assert "1 quarantined document(s)" in text
    assert "0 version file(s)" not in text


def test_lint_corpus_warns_about_a_textless_pdf(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    deal = corpus / "deal-scan"
    deal.mkdir(parents=True)
    _blank_scan_pdf(deal / "v1.pdf")
    report = lint_corpus(corpus)
    hits = [i for i in report.items if i.code == "DOC_PDF_NO_TEXT"]
    assert hits and hits[0].level == "warning" and "v1.pdf" in hits[0].message


def test_lint_corpus_does_not_flag_an_unreadable_pdf_stub(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    (corpus / "deal").mkdir(parents=True)
    (corpus / "deal" / "v1.pdf").write_bytes(b"%PDF-1.4\n%stub")
    assert not [i for i in lint_corpus(corpus).items if i.code == "DOC_PDF_NO_TEXT"]


# ---------------------------------------------------------------------------
# #19 / #21 / #43: the pages
# ---------------------------------------------------------------------------


def _bundle(tmp_path: Path, mutate: Any = None) -> Path:
    doc = json.loads(_NDA.read_text(encoding="utf-8"))
    if mutate is not None:
        mutate(doc)
    out = tmp_path / "out"
    out.mkdir()
    (out / "playbook.opf.json").write_text(json.dumps(doc), encoding="utf-8")
    return out


def test_absence_evidence_is_rendered_per_clause(tmp_path: Path) -> None:
    doc = json.loads(_NDA.read_text(encoding="utf-8"))
    records = doc["evidence"]["absence"]["records"]
    assert any(r["status"] == "absent_verified" for r in records)
    assert any(r["status"] == "possibly_present" for r in records)
    text = _page_text(_bundle(tmp_path))
    assert "Signed without this clause: verified absent in" in text
    assert "possibly present in" in text and "absence is not claimed" in text


def test_deal_basis_is_rendered_and_ambiguity_is_flagged(tmp_path: Path) -> None:
    def make_ambiguous(doc: dict[str, Any]) -> None:
        doc["corpus"]["documents"][0]["provenance_is_ambiguous"] = True

    text = _page_text(_bundle(tmp_path, make_ambiguous))
    assert "<th>Basis</th>" in text
    assert "template similarity" in text
    assert "date from signature block" in text
    assert "AMBIGUOUS" in text
    assert "label judged by agent, unchecked" in text


def test_pseudonymization_sentence_only_when_entities_were_matched(tmp_path: Path) -> None:
    plain = _page_text(_bundle(tmp_path))
    assert "pseudonymized at ingestion" not in plain

    def alias_one(doc: dict[str, Any]) -> None:
        doc["evidence"]["precedent"][0]["counterparty_ref"] = {"alias": "Counterparty-1"}

    shutil.rmtree(tmp_path / "out")
    aliased = _page_text(_bundle(tmp_path, alias_one))
    assert "pseudonymized at ingestion" in aliased
