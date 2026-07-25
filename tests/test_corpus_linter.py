"""Tests for corpus_linter.py.

SECURITY NOTE: All fixtures are programmatically constructed with synthetic
text.  No real agreement files are committed or referenced.  Fictional party
and author names only.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from playbook_engine.cli import cli
from playbook_engine.corpus_linter import LintItem, LintReport, lint_corpus

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _write_docx_stub(path: Path) -> None:
    """Write a minimal PK-magic-bytes stub so the file has the right extension."""
    # A real .docx is a ZIP; we just need a non-empty file recognised by extension.
    path.write_bytes(b"PK\x03\x04" + b"\x00" * 20)


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


def test_config_segmentation_llm_no_credentials_returns_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """lint-corpus is the documented preflight tool (issue #131) — it must catch
    a missing ANTHROPIC_API_KEY when segmentation.llm is on, not leave that to
    ``mine``/``compile``/``judge`` at run time."""
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
    cfg = _make_config(tmp_path, taxonomy_path=tax, template_path=template)
    report = lint_corpus(corpus, config_path=cfg)
    assert not report.has_errors
    assert not report.has_warnings


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
