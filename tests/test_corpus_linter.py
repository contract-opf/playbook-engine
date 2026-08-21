"""Tests for corpus_linter.py.

SECURITY NOTE: All fixtures are programmatically constructed with synthetic
text.  No real agreement files are committed or referenced.  Fictional party
and author names only.
"""

from __future__ import annotations

import dataclasses
import shutil
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from playbook_engine.cli import cli
from playbook_engine.corpus_linter import LintItem, LintReport, lint_corpus

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


_UNSET = object()  # sentinel: distinguishes "key omitted" from "key: null" (provenance/extraction)


def _write_docx_stub(path: Path) -> None:
    """Write a minimal PK-magic-bytes stub so the file has the right extension."""
    # A real .docx is a ZIP; we just need a non-empty file recognised by extension.
    path.write_bytes(b"PK\x03\x04" + b"\x00" * 20)


def _write_pdf_stub(path: Path) -> None:
    """Write a minimal PDF-header stub so the file has the right extension."""
    path.write_bytes(b"%PDF-1.4\n%stub")


def _write_rtf_stub(path: Path) -> None:
    """Write a minimal RTF-header stub so the file has the right extension."""
    path.write_text(r"{\rtf1\ansi stub}", encoding="utf-8")


def _make_config(
    tmp_path: Path,
    *,
    taxonomy_path: Path | None = None,
    template_path: Path | None = None,
    agreement_id: str = "fictional-agreement",
    agreement_name: str = "Fictional Agreement",
    valid_yaml: bool = True,
    include_agreement_type: bool = True,
    include_taxonomy: bool = True,
    segmentation: dict | None = None,
    provenance: object = _UNSET,
    extraction: object = _UNSET,
) -> Path:
    """Build a config YAML at tmp_path/config.yaml and return the path."""
    config_path = tmp_path / "playbook.config.yaml"
    if not valid_yaml:
        config_path.write_text("not: valid: yaml: [[[", encoding="utf-8")
        return config_path

    data: dict = {}
    if include_agreement_type:
        data["agreement_type"] = {"id": agreement_id, "name": agreement_name}
    if include_taxonomy:
        if taxonomy_path:
            data["taxonomy"] = str(taxonomy_path)
        else:
            data["taxonomy"] = ""
    if template_path:
        data["baseline"] = {"template": str(template_path)}
    else:
        data["baseline"] = {"template": None}
    if segmentation is not None:
        data["segmentation"] = segmentation
    if provenance is not _UNSET:
        data["provenance"] = provenance
    if extraction is not _UNSET:
        data["extraction"] = extraction

    config_path.write_text(yaml.dump(data), encoding="utf-8")
    return config_path


def _make_taxonomy(tmp_path: Path) -> Path:
    """Write a minimal taxonomy YAML and return the path."""
    tax_path = tmp_path / "taxonomy.yaml"
    tax_path.write_text(
        "source: fictional\nentries:\n  - id: TERM\n    label: Term\n    status: active\n",
        encoding="utf-8",
    )
    return tax_path


# ---------------------------------------------------------------------------
# LintReport unit tests
# ---------------------------------------------------------------------------


def test_lint_report_has_errors_false_when_empty() -> None:
    r = LintReport(corpus_dir=Path("/tmp"))
    assert not r.has_errors


def test_lint_report_has_errors_true_on_error() -> None:
    r = LintReport(corpus_dir=Path("/tmp"))
    r.add("error", "TEST", "test error")
    assert r.has_errors


def test_lint_report_ok_property() -> None:
    r = LintReport(corpus_dir=Path("/tmp"))
    assert r.ok
    r.add("error", "X", "x")
    assert not r.ok


def test_lint_report_errors_and_warnings_filtered() -> None:
    r = LintReport(corpus_dir=Path("/tmp"))
    r.add("error", "E1", "err")
    r.add("warning", "W1", "warn")
    r.add("ok", "OK1", "ok")
    assert len(r.errors()) == 1
    assert len(r.warnings()) == 1


def test_lint_item_frozen() -> None:
    item = LintItem(level="ok", code="X", message="m")
    import pytest  # noqa: PLC0415

    with pytest.raises((AttributeError, TypeError, dataclasses.FrozenInstanceError)):  # type: ignore[attr-defined]
        item.level = "error"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# lint_corpus: corpus-level checks
# ---------------------------------------------------------------------------


def test_nonexistent_corpus_returns_error(tmp_path: Path) -> None:
    report = lint_corpus(tmp_path / "no-such-dir")
    assert report.has_errors
    codes = {i.code for i in report.errors()}
    assert "CORPUS_NOT_FOUND" in codes


def test_nonexistent_corpus_returns_early(tmp_path: Path) -> None:
    """Only CORPUS_NOT_FOUND is reported — no further checks run."""
    report = lint_corpus(tmp_path / "no-such-dir")
    assert len(report.items) == 1


def test_file_passed_as_corpus_returns_error(tmp_path: Path) -> None:
    f = tmp_path / "notadir.txt"
    f.write_text("hi")
    report = lint_corpus(f)
    assert any(i.code == "NOT_A_DIRECTORY" for i in report.errors())


def test_empty_corpus_dir_returns_error(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    report = lint_corpus(corpus)
    assert any(i.code == "EMPTY_CORPUS" for i in report.errors())


def test_corpus_with_no_supported_files_returns_error(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    (corpus / "deal-alice").mkdir(parents=True)
    (corpus / "deal-alice" / "notes.txt").write_text("stray")
    report = lint_corpus(corpus)
    codes = {i.code for i in report.errors()}
    assert "DOC_NO_SUPPORTED_FILES" in codes or "NO_SUPPORTED_FILES" in codes


def test_valid_corpus_no_errors(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    deal = corpus / "deal-alice"
    deal.mkdir(parents=True)
    _write_docx_stub(deal / "v1.docx")
    _write_docx_stub(deal / "v2.docx")
    report = lint_corpus(corpus)
    assert not report.has_errors


# ---------------------------------------------------------------------------
# lint_corpus: per-document checks
# ---------------------------------------------------------------------------


def test_single_version_doc_warns(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    deal = corpus / "deal-bob"
    deal.mkdir(parents=True)
    _write_docx_stub(deal / "v1.docx")
    report = lint_corpus(corpus)
    assert not report.has_errors
    assert any(i.code == "DOC_SINGLE_VERSION" for i in report.warnings())


def test_two_version_doc_no_single_version_warning(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    deal = corpus / "deal-alice"
    deal.mkdir(parents=True)
    _write_docx_stub(deal / "v1.docx")
    _write_docx_stub(deal / "v2.docx")
    report = lint_corpus(corpus)
    assert not any(i.code == "DOC_SINGLE_VERSION" for i in report.warnings())


def test_duplicate_version_stems_flagged(tmp_path: Path) -> None:
    """'signed.pdf' and 'signed.docx' share a stem — the pipeline keys versions
    by stem, so one would silently overwrite the other (issue #95)."""
    corpus = tmp_path / "corpus"
    deal = corpus / "deal-alice"
    deal.mkdir(parents=True)
    _write_docx_stub(deal / "signed.docx")
    (deal / "signed.pdf").write_bytes(b"%PDF-1.4\n%stub")
    report = lint_corpus(corpus)
    assert report.has_errors
    assert any(i.code == "DOC_DUPLICATE_VERSION_STEM" for i in report.errors())


def test_duplicate_version_stems_case_insensitive(tmp_path: Path) -> None:
    """'Signed.pdf' and 'signed.docx' collide identically on case-insensitive
    filesystems and in the pipeline's stem-based keying."""
    corpus = tmp_path / "corpus"
    deal = corpus / "deal-bob"
    deal.mkdir(parents=True)
    _write_docx_stub(deal / "signed.docx")
    (deal / "Signed.pdf").write_bytes(b"%PDF-1.4\n%stub")
    report = lint_corpus(corpus)
    assert any(i.code == "DOC_DUPLICATE_VERSION_STEM" for i in report.errors())


def test_distinct_version_stems_not_flagged(tmp_path: Path) -> None:
    """Distinct stems (v1, v2) never trigger the duplicate-stem error."""
    corpus = tmp_path / "corpus"
    deal = corpus / "deal-charlie"
    deal.mkdir(parents=True)
    _write_docx_stub(deal / "v1.docx")
    _write_docx_stub(deal / "v2.docx")
    report = lint_corpus(corpus)
    assert not any(i.code == "DOC_DUPLICATE_VERSION_STEM" for i in report.errors())


def test_unsupported_files_in_doc_dir_warns(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    deal = corpus / "deal-charlie"
    deal.mkdir(parents=True)
    _write_docx_stub(deal / "v1.docx")
    (deal / "spreadsheet.xlsx").write_bytes(b"stray")
    report = lint_corpus(corpus)
    assert any(i.code == "DOC_UNSUPPORTED_FILES" for i in report.warnings())


def test_legacy_doc_format_called_out(tmp_path: Path) -> None:
    """A .doc file gets a distinct legacy-format lint entry naming '.doc' and
    the conversion instruction — not the generic DOC_UNSUPPORTED_FILES text
    (issue #100)."""
    corpus = tmp_path / "corpus"
    deal = corpus / "deal-charlie"
    deal.mkdir(parents=True)
    _write_docx_stub(deal / "v1.docx")
    (deal / "early-draft.doc").write_bytes(b"\xd0\xcf\x11\xe0stub")
    report = lint_corpus(corpus)

    legacy_items = [i for i in report.items if i.code == "DOC_LEGACY_FORMAT"]
    assert len(legacy_items) == 1
    assert ".doc" in legacy_items[0].message
    assert "soffice --convert-to docx" in legacy_items[0].message
    assert "early-draft.doc" in legacy_items[0].message

    # The .doc file must not also be reported as a generic unsupported file.
    unsupported_items = [i for i in report.items if i.code == "DOC_UNSUPPORTED_FILES"]
    assert not unsupported_items


def test_legacy_doc_only_still_flags_no_supported_files(tmp_path: Path) -> None:
    """A doc dir with only a .doc file still errors on no supported files,
    but also names the .doc file distinctly."""
    corpus = tmp_path / "corpus"
    deal = corpus / "deal-dora"
    deal.mkdir(parents=True)
    (deal / "only-draft.doc").write_bytes(b"\xd0\xcf\x11\xe0stub")
    report = lint_corpus(corpus)

    assert any(i.code == "DOC_NO_SUPPORTED_FILES" for i in report.errors())
    assert any(i.code == "DOC_LEGACY_FORMAT" for i in report.warnings())


def test_hints_yaml_not_treated_as_unsupported(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    deal = corpus / "deal-alice"
    deal.mkdir(parents=True)
    _write_docx_stub(deal / "v1.docx")
    (deal / "hints.yaml").write_text("signed_version: v1.docx\n")
    report = lint_corpus(corpus)
    assert not any(i.code == "DOC_UNSUPPORTED_FILES" for i in report.warnings())


def test_hints_yaml_produces_ok_item(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    deal = corpus / "deal-alice"
    deal.mkdir(parents=True)
    _write_docx_stub(deal / "v1.docx")
    (deal / "hints.yaml").write_text("signed_version: v1.docx\n")
    report = lint_corpus(corpus)
    assert any(i.code == "DOC_HAS_HINTS" for i in report.items)


def test_multiple_docs_mixed_state(tmp_path: Path) -> None:
    """Two docs: one two-version (ok), one single-version (warn), one empty (error)."""
    corpus = tmp_path / "corpus"
    good = corpus / "deal-alice"
    good.mkdir(parents=True)
    _write_docx_stub(good / "v1.docx")
    _write_docx_stub(good / "v2.docx")

    single = corpus / "deal-bob"
    single.mkdir()
    _write_docx_stub(single / "v1.docx")

    empty = corpus / "deal-charlie"
    empty.mkdir()

    report = lint_corpus(corpus)
    assert report.has_errors
    assert any(i.code == "DOC_NO_SUPPORTED_FILES" for i in report.errors())
    assert any(i.code == "DOC_SINGLE_VERSION" for i in report.warnings())


# ---------------------------------------------------------------------------
# lint_corpus: config checks
# ---------------------------------------------------------------------------


def test_config_not_found_returns_error(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    (corpus / "deal-alice").mkdir(parents=True)
    _write_docx_stub(corpus / "deal-alice" / "v1.docx")
    report = lint_corpus(corpus, config_path=tmp_path / "no-config.yaml")
    assert any(i.code == "CONFIG_NOT_FOUND" for i in report.errors())


def test_config_invalid_yaml_returns_error(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    (corpus / "deal-alice").mkdir(parents=True)
    _write_docx_stub(corpus / "deal-alice" / "v1.docx")
    cfg = _make_config(tmp_path, valid_yaml=False)
    report = lint_corpus(corpus, config_path=cfg)
    assert any(i.code == "CONFIG_INVALID_YAML" for i in report.errors())


def test_config_missing_agreement_type_returns_error(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    (corpus / "deal-alice").mkdir(parents=True)
    _write_docx_stub(corpus / "deal-alice" / "v1.docx")
    tax = _make_taxonomy(tmp_path)
    cfg = _make_config(tmp_path, taxonomy_path=tax, include_agreement_type=False)
    report = lint_corpus(corpus, config_path=cfg)
    assert any(i.code == "CONFIG_MISSING_AGREEMENT_TYPE" for i in report.errors())


def test_config_missing_taxonomy_returns_error(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    (corpus / "deal-alice").mkdir(parents=True)
    _write_docx_stub(corpus / "deal-alice" / "v1.docx")
    cfg = _make_config(tmp_path, include_taxonomy=False)
    report = lint_corpus(corpus, config_path=cfg)
    assert any(i.code == "CONFIG_MISSING_TAXONOMY" for i in report.errors())


def test_config_taxonomy_not_found_returns_error(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    (corpus / "deal-alice").mkdir(parents=True)
    _write_docx_stub(corpus / "deal-alice" / "v1.docx")
    cfg = _make_config(tmp_path, taxonomy_path=tmp_path / "nonexistent-taxonomy.yaml")
    report = lint_corpus(corpus, config_path=cfg)
    assert any(i.code == "CONFIG_TAXONOMY_NOT_FOUND" for i in report.errors())


def test_config_builtin_taxonomy_scheme_resolves(tmp_path: Path) -> None:
    """lint-corpus must accept the ``builtin:`` taxonomy scheme (issue #182).

    The config loader and the shipped example config both use
    ``taxonomy: builtin:<name>``, but the linter joined it onto the config dir
    as a literal path, so a valid builtin taxonomy was reported not-found.
    """
    corpus = tmp_path / "corpus"
    (corpus / "deal-alice").mkdir(parents=True)
    _write_docx_stub(corpus / "deal-alice" / "v1.docx")
    cfg = tmp_path / "playbook.config.yaml"
    cfg.write_text(
        yaml.dump(
            {
                "agreement_type": {
                    "id": "educational-affiliation",
                    "name": "Educational Affiliation Agreement",
                },
                "taxonomy": "builtin:affiliation-agreement.yaml",
                "baseline": {"template": None},
            }
        ),
        encoding="utf-8",
    )
    report = lint_corpus(corpus, config_path=cfg)
    assert not any(i.code == "CONFIG_TAXONOMY_NOT_FOUND" for i in report.errors())
    assert any(i.code == "CONFIG_TAXONOMY_EXISTS" for i in report.items)


def test_config_builtin_taxonomy_missing_name_errors(tmp_path: Path) -> None:
    """A ``builtin:`` value naming a nonexistent taxonomy still errors (issue #182)."""
    corpus = tmp_path / "corpus"
    (corpus / "deal-alice").mkdir(parents=True)
    _write_docx_stub(corpus / "deal-alice" / "v1.docx")
    cfg = tmp_path / "playbook.config.yaml"
    cfg.write_text(
        yaml.dump(
            {
                "agreement_type": {"id": "x", "name": "X"},
                "taxonomy": "builtin:does-not-exist.yaml",
                "baseline": {"template": None},
            }
        ),
        encoding="utf-8",
    )
    report = lint_corpus(corpus, config_path=cfg)
    assert any(i.code == "CONFIG_TAXONOMY_NOT_FOUND" for i in report.errors())


def test_config_template_not_found_returns_error(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    (corpus / "deal-alice").mkdir(parents=True)
    _write_docx_stub(corpus / "deal-alice" / "v1.docx")
    tax = _make_taxonomy(tmp_path)
    cfg = _make_config(
        tmp_path,
        taxonomy_path=tax,
        template_path=tmp_path / "no-template.docx",
    )
    report = lint_corpus(corpus, config_path=cfg)
    assert any(i.code == "CONFIG_TEMPLATE_NOT_FOUND" for i in report.errors())


def test_config_template_non_string_returns_error(tmp_path: Path) -> None:
    """Issue #74: a YAML scalar (e.g. an int) for baseline.template must be
    reported as a diagnostic, not crash the linter with a raw TypeError."""
    corpus = tmp_path / "corpus"
    (corpus / "deal-alice").mkdir(parents=True)
    _write_docx_stub(corpus / "deal-alice" / "v1.docx")
    tax = _make_taxonomy(tmp_path)
    cfg_path = tmp_path / "playbook.config.yaml"
    cfg_path.write_text(
        yaml.dump(
            {
                "agreement_type": {"id": "fictional-agreement", "name": "Fictional Agreement"},
                "taxonomy": str(tax),
                "baseline": {"template": 2023},
            }
        ),
        encoding="utf-8",
    )
    report = lint_corpus(corpus, config_path=cfg_path)
    assert any(i.code == "CONFIG_TEMPLATE_INVALID" for i in report.errors())


def test_config_segmentation_llm_no_credentials_returns_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """lint-corpus is the documented preflight tool (issue #131) — it must catch
    a missing ANTHROPIC_API_KEY when segmentation.llm is on, not leave that to
    ``mine``/``judge`` at run time."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    corpus = tmp_path / "corpus"
    (corpus / "deal-alice").mkdir(parents=True)
    _write_docx_stub(corpus / "deal-alice" / "v1.docx")
    tax = _make_taxonomy(tmp_path)
    cfg = _make_config(tmp_path, taxonomy_path=tax, segmentation={"llm": True})
    report = lint_corpus(corpus, config_path=cfg)
    assert any(i.code == "CONFIG_SEGMENTATION_LLM_NO_CREDENTIALS" for i in report.errors())


def test_config_segmentation_llm_with_credentials_no_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same config, but with ANTHROPIC_API_KEY set -> no credential error."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-dummy-key")
    corpus = tmp_path / "corpus"
    (corpus / "deal-alice").mkdir(parents=True)
    _write_docx_stub(corpus / "deal-alice" / "v1.docx")
    tax = _make_taxonomy(tmp_path)
    cfg = _make_config(tmp_path, taxonomy_path=tax, segmentation={"llm": True})
    report = lint_corpus(corpus, config_path=cfg)
    assert not any(i.code == "CONFIG_SEGMENTATION_LLM_NO_CREDENTIALS" for i in report.errors())


def test_config_segmentation_llm_false_never_requires_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No ``segmentation.llm`` -> the credential check never fires."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    corpus = tmp_path / "corpus"
    (corpus / "deal-alice").mkdir(parents=True)
    _write_docx_stub(corpus / "deal-alice" / "v1.docx")
    tax = _make_taxonomy(tmp_path)
    cfg = _make_config(tmp_path, taxonomy_path=tax)
    report = lint_corpus(corpus, config_path=cfg)
    assert not any(i.code == "CONFIG_SEGMENTATION_LLM_NO_CREDENTIALS" for i in report.errors())


def test_config_segmentation_agent_true_no_credentials_no_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue #68: ``segmentation: {llm: true, agent: true}`` is the key-free,
    store-backed agent path (cli._llm_segmentation_kwargs checks ``agent``
    before its ANTHROPIC_API_KEY gate) -- lint-corpus must not contradict the
    command it preflights by demanding a key here too."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    corpus = tmp_path / "corpus"
    (corpus / "deal-alice").mkdir(parents=True)
    _write_docx_stub(corpus / "deal-alice" / "v1.docx")
    tax = _make_taxonomy(tmp_path)
    cfg = _make_config(tmp_path, taxonomy_path=tax, segmentation={"llm": True, "agent": True})
    report = lint_corpus(corpus, config_path=cfg)
    assert not any(i.code == "CONFIG_SEGMENTATION_LLM_NO_CREDENTIALS" for i in report.errors())


# ---------------------------------------------------------------------------
# lint_corpus: extraction environment checks (issue #82)
#
# extraction.extract_blocks (docling/pdfplumber/pandoc) is reachable ONLY
# from the LLM-segmentation path (segmentation.llm or segmentation.agent) --
# the deterministic ingest path (docx_ingester/rtf_ingester/pdf_ingester)
# never calls it (config.py's ExtractionConfig docstring: "a declared
# extractor has nothing to govern" there). Every fixture below therefore
# turns on ``segmentation: {llm: true}`` unless a test specifically
# exercises that gate itself. A dummy ANTHROPIC_API_KEY is set wherever a
# test asserts the ABSENCE of errors/warnings, so the unrelated
# CONFIG_SEGMENTATION_LLM_NO_CREDENTIALS check (issue #131) never
# contaminates that assertion.
# ---------------------------------------------------------------------------


def test_extraction_docling_declared_missing_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``extraction.extractor: docling`` + no docling on PATH must error --
    the exact host-run failure mode (Jul 14: 161/161 legacy) this ticket
    exists to catch at lint time instead of after a burned LLM budget."""
    monkeypatch.setattr(shutil, "which", lambda _cmd: None)
    corpus = tmp_path / "corpus"
    (corpus / "deal-alice").mkdir(parents=True)
    _write_docx_stub(corpus / "deal-alice" / "v1.docx")
    tax = _make_taxonomy(tmp_path)
    cfg = _make_config(
        tmp_path,
        taxonomy_path=tax,
        segmentation={"llm": True},
        extraction={"extractor": "docling"},
    )
    report = lint_corpus(corpus, config_path=cfg)
    assert any(i.code == "CONFIG_EXTRACTION_DOCLING_MISSING" for i in report.errors())


def test_extraction_docling_declared_present_no_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same config, but docling IS on PATH -> no error."""

    def fake_which(cmd: str) -> str | None:
        return "/usr/bin/docling" if cmd == "docling" else None

    monkeypatch.setattr(shutil, "which", fake_which)
    corpus = tmp_path / "corpus"
    (corpus / "deal-alice").mkdir(parents=True)
    _write_docx_stub(corpus / "deal-alice" / "v1.docx")
    tax = _make_taxonomy(tmp_path)
    cfg = _make_config(
        tmp_path,
        taxonomy_path=tax,
        segmentation={"llm": True},
        extraction={"extractor": "docling"},
    )
    report = lint_corpus(corpus, config_path=cfg)
    assert not any(i.code == "CONFIG_EXTRACTION_DOCLING_MISSING" for i in report.errors())


def test_extraction_checks_gated_on_llm_segmentation_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression guard: with no ``segmentation:`` section at all (the
    engine's own deterministic default), extract_blocks is never reached --
    docx_ingester/rtf_ingester/pdf_ingester never call docling/pandoc -- so a
    declared ``extraction.extractor: docling`` must not be flagged even on a
    docling-less host. This is exactly examples/judge-fixture/'s shape
    (.rtf files, no segmentation: section); an ungated check here would
    break tests/test_quickstart.py's "no errors, 2 warning(s)" marker on any
    docling+pandoc-less machine."""
    monkeypatch.setattr(shutil, "which", lambda _cmd: None)
    corpus = tmp_path / "corpus"
    (corpus / "deal-alice").mkdir(parents=True)
    _write_rtf_stub(corpus / "deal-alice" / "v1.rtf")
    tax = _make_taxonomy(tmp_path)
    cfg = _make_config(tmp_path, taxonomy_path=tax, extraction={"extractor": "docling"})
    report = lint_corpus(corpus, config_path=cfg)
    assert not any(i.code.startswith("CONFIG_EXTRACTION_") for i in report.items)


def test_extraction_checks_apply_under_agent_without_llm_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``segmentation: {agent: true}`` (``llm`` key omitted) still resolves
    to llm-first under the loader (``llm=bool(seg_raw.get("llm")) or
    agent_seg`` -- config.py), and the agent path still calls
    extract_blocks (it only skips the live API call) -- so the extraction
    checks must still apply."""
    monkeypatch.setattr(shutil, "which", lambda _cmd: None)
    corpus = tmp_path / "corpus"
    (corpus / "deal-alice").mkdir(parents=True)
    _write_docx_stub(corpus / "deal-alice" / "v1.docx")
    tax = _make_taxonomy(tmp_path)
    cfg = _make_config(
        tmp_path,
        taxonomy_path=tax,
        segmentation={"agent": True},
        extraction={"extractor": "docling"},
    )
    report = lint_corpus(corpus, config_path=cfg)
    assert any(i.code == "CONFIG_EXTRACTION_DOCLING_MISSING" for i in report.errors())


def test_extraction_auto_pdf_no_docling_warns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Default ``extractor: auto`` + no docling + a .pdf in the corpus must
    warn, not error -- legacy PDF via pdfplumber is degraded (no OCR) but
    functional, and erroring would break every docling-less dev loop
    (verifier correction on the ticket)."""
    monkeypatch.setattr(shutil, "which", lambda _cmd: None)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-dummy-key")
    corpus = tmp_path / "corpus"
    (corpus / "deal-alice").mkdir(parents=True)
    _write_pdf_stub(corpus / "deal-alice" / "v1.pdf")
    tax = _make_taxonomy(tmp_path)
    cfg = _make_config(tmp_path, taxonomy_path=tax, segmentation={"llm": True})
    report = lint_corpus(corpus, config_path=cfg)
    assert any(i.code == "CONFIG_EXTRACTION_AUTO_PDF_NO_DOCLING" for i in report.warnings())
    assert not report.has_errors


def test_extraction_auto_pdf_with_docling_no_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same corpus, but docling IS on PATH -> no warning (PDFs go through
    docling, not the legacy no-OCR pdfplumber adapter)."""

    def fake_which(cmd: str) -> str | None:
        return "/usr/bin/docling" if cmd == "docling" else None

    monkeypatch.setattr(shutil, "which", fake_which)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-dummy-key")
    corpus = tmp_path / "corpus"
    (corpus / "deal-alice").mkdir(parents=True)
    _write_pdf_stub(corpus / "deal-alice" / "v1.pdf")
    tax = _make_taxonomy(tmp_path)
    cfg = _make_config(tmp_path, taxonomy_path=tax, segmentation={"llm": True})
    report = lint_corpus(corpus, config_path=cfg)
    assert not any(i.code == "CONFIG_EXTRACTION_AUTO_PDF_NO_DOCLING" for i in report.warnings())


def test_extraction_legacy_declared_pdf_no_docling_no_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``extraction.extractor: legacy`` is a deliberate, informed choice to
    accept the legacy adapters even without docling -- must not warn."""
    monkeypatch.setattr(shutil, "which", lambda _cmd: None)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-dummy-key")
    corpus = tmp_path / "corpus"
    (corpus / "deal-alice").mkdir(parents=True)
    _write_pdf_stub(corpus / "deal-alice" / "v1.pdf")
    tax = _make_taxonomy(tmp_path)
    cfg = _make_config(
        tmp_path,
        taxonomy_path=tax,
        segmentation={"llm": True},
        extraction={"extractor": "legacy"},
    )
    report = lint_corpus(corpus, config_path=cfg)
    assert not any(i.code == "CONFIG_EXTRACTION_AUTO_PDF_NO_DOCLING" for i in report.warnings())


def test_extraction_auto_no_pdf_no_docling_no_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No .pdf anywhere in the corpus -> no warning, even docling-less."""
    monkeypatch.setattr(shutil, "which", lambda _cmd: None)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-dummy-key")
    corpus = tmp_path / "corpus"
    (corpus / "deal-alice").mkdir(parents=True)
    _write_docx_stub(corpus / "deal-alice" / "v1.docx")
    tax = _make_taxonomy(tmp_path)
    cfg = _make_config(tmp_path, taxonomy_path=tax, segmentation={"llm": True})
    report = lint_corpus(corpus, config_path=cfg)
    assert not any(i.code == "CONFIG_EXTRACTION_AUTO_PDF_NO_DOCLING" for i in report.warnings())


def test_extraction_rtf_no_pandoc_under_auto_no_docling_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """.rtf in the corpus, default ``auto`` extractor, neither pandoc nor
    docling on PATH must error -- the legacy RTF adapter raises
    ExtractionError at runtime today; this surfaces it at lint time
    instead."""
    monkeypatch.setattr(shutil, "which", lambda _cmd: None)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-dummy-key")
    corpus = tmp_path / "corpus"
    (corpus / "deal-alice").mkdir(parents=True)
    _write_rtf_stub(corpus / "deal-alice" / "v1.rtf")
    tax = _make_taxonomy(tmp_path)
    cfg = _make_config(tmp_path, taxonomy_path=tax, segmentation={"llm": True})
    report = lint_corpus(corpus, config_path=cfg)
    assert any(i.code == "CONFIG_EXTRACTION_RTF_NO_PANDOC" for i in report.errors())


def test_extraction_rtf_with_pandoc_no_docling_no_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """pandoc present (docling absent) -> the legacy RTF adapter works
    fine, no error."""

    def fake_which(cmd: str) -> str | None:
        return "/usr/local/bin/pandoc" if cmd == "pandoc" else None

    monkeypatch.setattr(shutil, "which", fake_which)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-dummy-key")
    corpus = tmp_path / "corpus"
    (corpus / "deal-alice").mkdir(parents=True)
    _write_rtf_stub(corpus / "deal-alice" / "v1.rtf")
    tax = _make_taxonomy(tmp_path)
    cfg = _make_config(tmp_path, taxonomy_path=tax, segmentation={"llm": True})
    report = lint_corpus(corpus, config_path=cfg)
    assert not any(i.code == "CONFIG_EXTRACTION_RTF_NO_PANDOC" for i in report.errors())


def test_extraction_rtf_with_docling_no_pandoc_no_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """docling present (pandoc absent), extractor auto -> docling handles
    RTF directly, no error."""

    def fake_which(cmd: str) -> str | None:
        return "/usr/bin/docling" if cmd == "docling" else None

    monkeypatch.setattr(shutil, "which", fake_which)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-dummy-key")
    corpus = tmp_path / "corpus"
    (corpus / "deal-alice").mkdir(parents=True)
    _write_rtf_stub(corpus / "deal-alice" / "v1.rtf")
    tax = _make_taxonomy(tmp_path)
    cfg = _make_config(tmp_path, taxonomy_path=tax, segmentation={"llm": True})
    report = lint_corpus(corpus, config_path=cfg)
    assert not any(i.code == "CONFIG_EXTRACTION_RTF_NO_PANDOC" for i in report.errors())


def test_extraction_rtf_declared_legacy_docling_present_no_pandoc_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``extraction.extractor: legacy`` always forces the legacy adapters --
    extraction._resolve_extractor_env resolves a declared "legacy" to
    "legacy" verbatim even when docling IS on PATH (that's the whole point:
    a deterministic, container-free run). So for .rtf the pandoc-backed
    legacy adapter always runs under this setting, and a missing pandoc must
    still error even though docling happens to be present -- this is the
    false negative a prior review round caught: docling's mere presence must
    NOT silence the check when the config forces legacy explicitly."""

    def fake_which(cmd: str) -> str | None:
        return "/usr/bin/docling" if cmd == "docling" else None

    monkeypatch.setattr(shutil, "which", fake_which)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-dummy-key")
    corpus = tmp_path / "corpus"
    (corpus / "deal-alice").mkdir(parents=True)
    _write_rtf_stub(corpus / "deal-alice" / "v1.rtf")
    tax = _make_taxonomy(tmp_path)
    cfg = _make_config(
        tmp_path,
        taxonomy_path=tax,
        segmentation={"llm": True},
        extraction={"extractor": "legacy"},
    )
    report = lint_corpus(corpus, config_path=cfg)
    assert any(i.code == "CONFIG_EXTRACTION_RTF_NO_PANDOC" for i in report.errors())


def test_extraction_rtf_declared_docling_missing_no_duplicate_rtf_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``extraction.extractor: docling`` + docling absent + .rtf present +
    pandoc also absent must produce ONLY CONFIG_EXTRACTION_DOCLING_MISSING
    -- not also CONFIG_EXTRACTION_RTF_NO_PANDOC, which would misname the fix (this
    corpus never reaches the pandoc-backed adapter in this configuration;
    installing docling is what unblocks it, not pandoc)."""
    monkeypatch.setattr(shutil, "which", lambda _cmd: None)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-dummy-key")
    corpus = tmp_path / "corpus"
    (corpus / "deal-alice").mkdir(parents=True)
    _write_rtf_stub(corpus / "deal-alice" / "v1.rtf")
    tax = _make_taxonomy(tmp_path)
    cfg = _make_config(
        tmp_path,
        taxonomy_path=tax,
        segmentation={"llm": True},
        extraction={"extractor": "docling"},
    )
    report = lint_corpus(corpus, config_path=cfg)
    assert any(i.code == "CONFIG_EXTRACTION_DOCLING_MISSING" for i in report.errors())
    assert not any(i.code == "CONFIG_EXTRACTION_RTF_NO_PANDOC" for i in report.errors())


def test_extraction_green_path_docling_present_silent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """docling present, corpus has both .pdf and .rtf -> no CONFIG_EXTRACTION_*
    items at all (silence on success, matching the ANTHROPIC_API_KEY gate's
    own style -- no OK item either)."""

    def fake_which(cmd: str) -> str | None:
        return "/usr/bin/docling" if cmd == "docling" else None

    monkeypatch.setattr(shutil, "which", fake_which)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-dummy-key")
    corpus = tmp_path / "corpus"
    (corpus / "deal-alice").mkdir(parents=True)
    _write_pdf_stub(corpus / "deal-alice" / "v1.pdf")
    (corpus / "deal-bob").mkdir(parents=True)
    _write_rtf_stub(corpus / "deal-bob" / "v1.rtf")
    tax = _make_taxonomy(tmp_path)
    cfg = _make_config(tmp_path, taxonomy_path=tax, segmentation={"llm": True})
    report = lint_corpus(corpus, config_path=cfg)
    assert not any(i.code.startswith("CONFIG_EXTRACTION_") for i in report.items)


def test_extraction_invalid_extractor_value_reports_error_and_falls_back_to_auto(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A garbage ``extraction.extractor`` value (something load_config would
    reject with a ConfigError -- "extraction.extractor must be one of
    'docling', 'legacy', or 'auto'") must be reported as a lint ERROR here
    too, mirroring CONFIG_PROVENANCE_INVALID -- so lint-corpus never says
    "OK" about a config mine/compile will refuse to load. It still falls
    back to "auto" for the docling/pandoc environment checks specifically
    (not for the shape report itself), so those checks run against a sane
    default instead of skipping outright."""
    monkeypatch.setattr(shutil, "which", lambda _cmd: None)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-dummy-key")
    corpus = tmp_path / "corpus"
    (corpus / "deal-alice").mkdir(parents=True)
    _write_docx_stub(corpus / "deal-alice" / "v1.docx")
    tax = _make_taxonomy(tmp_path)
    cfg = _make_config(
        tmp_path,
        taxonomy_path=tax,
        segmentation={"llm": True},
        extraction={"extractor": "bogus-value"},
    )
    report = lint_corpus(corpus, config_path=cfg)
    assert any(i.code == "CONFIG_EXTRACTION_INVALID_EXTRACTOR" for i in report.errors())
    # Treated as "auto" for the environment checks: no docling declared, so
    # no DOCLING_MISSING; no .pdf/.rtf in this corpus, so no other
    # environment item fires either.
    assert not any(i.code == "CONFIG_EXTRACTION_DOCLING_MISSING" for i in report.errors())
    assert not any(
        i.code in ("CONFIG_EXTRACTION_AUTO_PDF_NO_DOCLING", "CONFIG_EXTRACTION_RTF_NO_PANDOC")
        for i in report.items
    )


def test_extraction_non_mapping_section_reports_invalid_ungated(tmp_path: Path) -> None:
    """A bare ``extraction:`` key (YAML null, or any other non-mapping
    scalar) parses fine as YAML but load_config rejects it outright
    ("config.extraction must be a mapping") -- lint-corpus must report this
    as an ERROR, mirroring CONFIG_PROVENANCE_INVALID for the analogous
    ``provenance:`` shape. Deliberately no ``segmentation:`` section at all
    here (extraction_relevant is False, the deterministic-ingest default)
    to prove this shape check is NOT nested under extraction_relevant --
    load_config validates extraction: unconditionally, so lint-corpus must
    too, or a docling/pandoc-irrelevant corpus could still say "OK" about a
    config mine/compile hard-fails on."""
    corpus = tmp_path / "corpus"
    (corpus / "deal-alice").mkdir(parents=True)
    _write_docx_stub(corpus / "deal-alice" / "v1.docx")
    tax = _make_taxonomy(tmp_path)
    cfg = _make_config(tmp_path, taxonomy_path=tax, extraction=None)
    report = lint_corpus(corpus, config_path=cfg)
    assert any(i.code == "CONFIG_EXTRACTION_INVALID" for i in report.errors())


def test_extraction_section_typo_flagged_as_unknown_key(tmp_path: Path) -> None:
    """A misspelled ``extraction:`` sub-key must be flagged, mirroring the
    existing segmentation/provenance typo checks -- the extraction: section
    (issue #80) was previously absent from the per-section unknown-key
    loop, so a typo like ``extractr`` silently fell back to the "auto"
    default with no warning."""
    corpus = tmp_path / "corpus"
    (corpus / "deal-alice").mkdir(parents=True)
    _write_docx_stub(corpus / "deal-alice" / "v1.docx")
    tax = _make_taxonomy(tmp_path)
    cfg = _make_config(tmp_path, taxonomy_path=tax, extraction={"extractr": "docling"})
    report = lint_corpus(corpus, config_path=cfg)
    unknown_key_items = [i for i in report.errors() if i.code == "CONFIG_UNKNOWN_KEY"]
    assert unknown_key_items
    assert "extraction.extractr" in unknown_key_items[0].message
    assert "extraction.extractor" in unknown_key_items[0].message


def test_lint_corpus_cmd_docling_declared_missing_exits_nonzero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CLI-level acceptance criterion: ``playbook lint-corpus`` on a
    docling-less host with a declared-docling config exits 1 with the new
    error."""
    monkeypatch.setattr(shutil, "which", lambda _cmd: None)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-dummy-key")
    corpus = tmp_path / "corpus"
    (corpus / "deal-alice").mkdir(parents=True)
    _write_docx_stub(corpus / "deal-alice" / "v1.docx")
    tax = _make_taxonomy(tmp_path)
    cfg = _make_config(
        tmp_path,
        taxonomy_path=tax,
        segmentation={"llm": True},
        extraction={"extractor": "docling"},
    )
    runner = CliRunner()
    result = runner.invoke(cli, ["lint-corpus", str(corpus), "--config", str(cfg)])
    assert result.exit_code == 1
    assert "docling binary was not found on PATH" in result.output


def test_config_no_our_party_aliases_missing_provenance_warns(tmp_path: Path) -> None:
    """Issue #56: no ``provenance:`` section at all (the scaffold's own
    default before a first-timer fills anything in) must warn, not pass
    silently -- the scaffold marks our_party_aliases REQUIRED and claims
    mine warns if none match, but nothing does today."""
    corpus = tmp_path / "corpus"
    (corpus / "deal-alice").mkdir(parents=True)
    _write_docx_stub(corpus / "deal-alice" / "v1.docx")
    tax = _make_taxonomy(tmp_path)
    cfg = _make_config(tmp_path, taxonomy_path=tax)  # provenance omitted entirely
    report = lint_corpus(corpus, config_path=cfg)
    assert not report.has_errors
    assert any(i.code == "CONFIG_NO_OUR_PARTY_ALIASES" for i in report.warnings())


def test_config_no_our_party_aliases_empty_list_warns(tmp_path: Path) -> None:
    """Issue #56: the scaffolded ``our_party_aliases: []`` left unfilled by a
    first-timer must warn -- every document will silently default to
    counterparty_paper."""
    corpus = tmp_path / "corpus"
    (corpus / "deal-alice").mkdir(parents=True)
    _write_docx_stub(corpus / "deal-alice" / "v1.docx")
    tax = _make_taxonomy(tmp_path)
    cfg = _make_config(tmp_path, taxonomy_path=tax, provenance={"our_party_aliases": []})
    report = lint_corpus(corpus, config_path=cfg)
    assert not report.has_errors
    assert any(i.code == "CONFIG_NO_OUR_PARTY_ALIASES" for i in report.warnings())


def test_config_our_party_aliases_blank_only_still_warns(tmp_path: Path) -> None:
    """A list of only empty strings is functionally empty and must still warn."""
    corpus = tmp_path / "corpus"
    (corpus / "deal-alice").mkdir(parents=True)
    _write_docx_stub(corpus / "deal-alice" / "v1.docx")
    tax = _make_taxonomy(tmp_path)
    cfg = _make_config(tmp_path, taxonomy_path=tax, provenance={"our_party_aliases": ["", ""]})
    report = lint_corpus(corpus, config_path=cfg)
    assert any(i.code == "CONFIG_NO_OUR_PARTY_ALIASES" for i in report.warnings())


def test_config_our_party_aliases_present_no_warning(tmp_path: Path) -> None:
    """A filled-in our_party_aliases list must NOT trigger the warning."""
    corpus = tmp_path / "corpus"
    (corpus / "deal-alice").mkdir(parents=True)
    _write_docx_stub(corpus / "deal-alice" / "v1.docx")
    tax = _make_taxonomy(tmp_path)
    cfg = _make_config(
        tmp_path, taxonomy_path=tax, provenance={"our_party_aliases": ["FixtureCorp"]}
    )
    report = lint_corpus(corpus, config_path=cfg)
    assert not any(i.code == "CONFIG_NO_OUR_PARTY_ALIASES" for i in report.warnings())


def test_config_provenance_null_reports_invalid_not_alias_warning(tmp_path: Path) -> None:
    """Issue #56: a bare ``provenance:`` key (YAML null) is rejected outright
    by load_config ("config.provenance must be a mapping"). lint-corpus must
    report this as the CONFIG_PROVENANCE_INVALID ERROR -- not silently pass
    it through to the milder CONFIG_NO_OUR_PARTY_ALIASES warning path, which
    would print "OK" (well, "1 warning") about a config mine hard-fails on."""
    corpus = tmp_path / "corpus"
    (corpus / "deal-alice").mkdir(parents=True)
    _write_docx_stub(corpus / "deal-alice" / "v1.docx")
    tax = _make_taxonomy(tmp_path)
    cfg = _make_config(tmp_path, taxonomy_path=tax, provenance=None)
    report = lint_corpus(corpus, config_path=cfg)
    assert any(i.code == "CONFIG_PROVENANCE_INVALID" for i in report.errors())
    assert not any(i.code == "CONFIG_NO_OUR_PARTY_ALIASES" for i in report.warnings())


def test_config_our_party_aliases_null_reports_aliases_invalid(tmp_path: Path) -> None:
    """A bare ``our_party_aliases:`` key (YAML null) is rejected by
    load_config ("provenance.our_party_aliases must be a list") -- must be a
    lint ERROR, not the empty-list warning."""
    corpus = tmp_path / "corpus"
    (corpus / "deal-alice").mkdir(parents=True)
    _write_docx_stub(corpus / "deal-alice" / "v1.docx")
    tax = _make_taxonomy(tmp_path)
    cfg = _make_config(tmp_path, taxonomy_path=tax, provenance={"our_party_aliases": None})
    report = lint_corpus(corpus, config_path=cfg)
    assert any(i.code == "CONFIG_ALIASES_INVALID" for i in report.errors())
    assert not any(i.code == "CONFIG_NO_OUR_PARTY_ALIASES" for i in report.warnings())


def test_config_our_party_aliases_non_list_reports_aliases_invalid(tmp_path: Path) -> None:
    """A scalar string instead of a list must be flagged as CONFIG_ALIASES_INVALID."""
    corpus = tmp_path / "corpus"
    (corpus / "deal-alice").mkdir(parents=True)
    _write_docx_stub(corpus / "deal-alice" / "v1.docx")
    tax = _make_taxonomy(tmp_path)
    cfg = _make_config(tmp_path, taxonomy_path=tax, provenance={"our_party_aliases": "FixtureCorp"})
    report = lint_corpus(corpus, config_path=cfg)
    assert any(i.code == "CONFIG_ALIASES_INVALID" for i in report.errors())


def test_config_no_template_warns(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    (corpus / "deal-alice").mkdir(parents=True)
    _write_docx_stub(corpus / "deal-alice" / "v1.docx")
    tax = _make_taxonomy(tmp_path)
    cfg = _make_config(tmp_path, taxonomy_path=tax)  # template_path=None
    report = lint_corpus(corpus, config_path=cfg)
    assert not report.has_errors
    assert any(i.code == "CONFIG_NO_TEMPLATE" for i in report.warnings())


def test_config_valid_full_ok(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    (corpus / "deal-alice").mkdir(parents=True)
    _write_docx_stub(corpus / "deal-alice" / "v1.docx")
    _write_docx_stub(corpus / "deal-alice" / "v2.docx")
    tax = _make_taxonomy(tmp_path)
    template = tmp_path / "template.docx"
    _write_docx_stub(template)
    cfg = _make_config(
        tmp_path,
        taxonomy_path=tax,
        template_path=template,
        provenance={"our_party_aliases": ["FixtureCorp"]},
    )
    report = lint_corpus(corpus, config_path=cfg)
    assert not report.has_errors
    assert not report.has_warnings


def test_config_typoed_provenance_key_flagged(tmp_path: Path) -> None:
    """A misspelled top-level ``provenence:`` (issue #53) must be flagged as
    a lint error, not reported as "OK" while pseudonymization is silently
    disabled.
    """
    corpus = tmp_path / "corpus"
    (corpus / "deal-alice").mkdir(parents=True)
    _write_docx_stub(corpus / "deal-alice" / "v1.docx")
    tax = _make_taxonomy(tmp_path)
    cfg = _make_config(tmp_path, taxonomy_path=tax)
    data = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    data["provenence"] = {
        "our_party_aliases": ["FixtureCorp"],
        "known_entities": ["State University"],
    }
    cfg.write_text(yaml.dump(data), encoding="utf-8")
    report = lint_corpus(corpus, config_path=cfg)
    unknown_key_items = [i for i in report.errors() if i.code == "CONFIG_UNKNOWN_KEY"]
    assert unknown_key_items
    assert "provenence" in unknown_key_items[0].message
    assert "provenance" in unknown_key_items[0].message


def test_config_typoed_segmentation_key_flagged(tmp_path: Path) -> None:
    """A misspelled ``segmentation:`` sub-key must be flagged rather than
    silently leaving ``llm`` at its False default."""
    corpus = tmp_path / "corpus"
    (corpus / "deal-alice").mkdir(parents=True)
    _write_docx_stub(corpus / "deal-alice" / "v1.docx")
    tax = _make_taxonomy(tmp_path)
    cfg = _make_config(tmp_path, taxonomy_path=tax, segmentation={"lm": True})
    report = lint_corpus(corpus, config_path=cfg)
    unknown_key_items = [i for i in report.errors() if i.code == "CONFIG_UNKNOWN_KEY"]
    assert unknown_key_items
    assert "segmentation.lm" in unknown_key_items[0].message
    assert "segmentation.llm" in unknown_key_items[0].message


def test_no_config_path_skips_config_checks(tmp_path: Path) -> None:
    """When config_path is None, no CONFIG_* items are produced."""
    corpus = tmp_path / "corpus"
    (corpus / "deal-alice").mkdir(parents=True)
    _write_docx_stub(corpus / "deal-alice" / "v1.docx")
    report = lint_corpus(corpus, config_path=None)
    config_items = [i for i in report.items if i.code.startswith("CONFIG_")]
    assert config_items == []


def test_dot_directories_ignored(tmp_path: Path) -> None:
    """Hidden dot-prefixed dirs (.git, .DS_Store) are not treated as agreements."""
    corpus = tmp_path / "corpus"
    (corpus / ".git").mkdir(parents=True)
    (corpus / ".DS_Store").mkdir()
    report = lint_corpus(corpus)
    # Should read as an empty corpus, not as agreements with no supported files
    assert any(i.code == "EMPTY_CORPUS" for i in report.errors())
    assert not any(i.code == "DOC_NO_SUPPORTED_FILES" for i in report.errors())


# ---------------------------------------------------------------------------
# lint-corpus CLI command tests
# ---------------------------------------------------------------------------


def _make_minimal_corpus(tmp_path: Path) -> Path:
    corpus = tmp_path / "corpus"
    deal = corpus / "deal-alice"
    deal.mkdir(parents=True)
    _write_docx_stub(deal / "v1.docx")
    _write_docx_stub(deal / "v2.docx")
    return corpus


def test_lint_corpus_cmd_exits_zero_on_valid_corpus(tmp_path: Path) -> None:
    corpus = _make_minimal_corpus(tmp_path)
    runner = CliRunner()
    result = runner.invoke(cli, ["lint-corpus", str(corpus)])
    assert result.exit_code == 0


def test_lint_corpus_cmd_exits_zero_with_warnings(tmp_path: Path) -> None:
    """Warnings alone do not cause a non-zero exit."""
    corpus = tmp_path / "corpus"
    deal = corpus / "deal-alice"
    deal.mkdir(parents=True)
    _write_docx_stub(deal / "v1.docx")  # single version → warning
    runner = CliRunner()
    result = runner.invoke(cli, ["lint-corpus", str(corpus)])
    assert result.exit_code == 0
    assert "WARN" in result.output


def test_lint_corpus_cmd_warn_lines_go_to_stderr(tmp_path: Path) -> None:
    """WARN lines must land on stderr, not stdout (issue #60).

    Pre-fix, ``lint-corpus`` printed WARN (and OK) lines to stdout while ERR
    lines and the failure summary went to stderr — a scripted caller doing
    ``lint-corpus ... 2>err.log`` would miss every warning. WARN is a
    diagnostic and belongs on stderr; the closing ``OK`` summary line stays
    a result on stdout.
    """
    corpus = tmp_path / "corpus"
    deal = corpus / "deal-alice"
    deal.mkdir(parents=True)
    _write_docx_stub(deal / "v1.docx")  # single version → WARN
    runner = CliRunner()
    result = runner.invoke(cli, ["lint-corpus", str(corpus)])
    assert result.exit_code == 0
    assert "WARN" in result.stderr
    assert "WARN" not in result.stdout
    assert "OK — no errors" in result.stdout


def test_lint_corpus_cmd_exits_nonzero_on_empty_corpus(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    runner = CliRunner()
    result = runner.invoke(cli, ["lint-corpus", str(corpus)])
    assert result.exit_code != 0


def test_lint_corpus_cmd_config_optional(tmp_path: Path) -> None:
    """--config is optional for lint-corpus (unlike compile)."""
    corpus = _make_minimal_corpus(tmp_path)
    runner = CliRunner()
    result = runner.invoke(cli, ["lint-corpus", str(corpus)])
    assert result.exit_code == 0


def test_lint_corpus_cmd_config_errors_exit_nonzero(tmp_path: Path) -> None:
    corpus = _make_minimal_corpus(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "lint-corpus",
            str(corpus),
            "--config",
            str(tmp_path / "missing.yaml"),
        ],
    )
    assert result.exit_code != 0
    assert "ERR" in result.output
