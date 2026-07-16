"""Tests for issue #165: perspective (party/counterparty_type) config surface.

`perspective` is the field that most distinguishes one agreement type's
playbook from another's ("who is 'us', and what kind of counterparty is
across the table"). Before this issue it was accepted and emitted by
``playbook_assembler.assemble_playbook`` but had no config surface and
``pipeline.project_playbook`` never passed it through — an open-standard
OPF instance could only be produced by hand-editing JSON.

SECURITY NOTE: All fixtures use programmatically constructed RTF text with
synthetic, fictional content. No real agreement files are referenced.
Fictional party names only (e.g. "FixtureCorp", "Beta University").
"""

from __future__ import annotations

from pathlib import Path

import yaml

from playbook_engine.config import load_config
from playbook_engine.pipeline import mine_corpus, project_playbook
from playbook_engine.staging import scaffold_config
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


_CORPUS_BODY = (
    r"1. Indemnification\par "
    r"FixtureCorp shall indemnify Beta University against third-party claims "
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


def _make_corpus(tmp_path: Path, *, extra_config: dict | None = None) -> tuple[Path, Path, Path]:
    """Build a synthetic single-document corpus + a config carrying a
    canonical template; return (corpus_dir, config_path, out_dir).

    ``extra_config`` is merged into the base config dict, letting each test
    add its own ``perspective:`` block without duplicating this scaffolding.
    """
    corpus_dir = tmp_path / "corpus"
    deal_dir = corpus_dir / "deal-001"
    deal_dir.mkdir(parents=True)
    _write_rtf(deal_dir / "v1.rtf", _CORPUS_BODY)

    template_dir = tmp_path / "template"
    template_dir.mkdir()
    template_path = template_dir / "template.rtf"
    _write_rtf(template_path, _TEMPLATE_BODY)

    cfg: dict = {
        "agreement_type": {
            "id": "educational-affiliation",
            "name": "Educational Affiliation Agreement",
        },
        "baseline": {"template": str(template_path)},
        "taxonomy": str(_TAXONOMY_PATH),
        "provenance": {"our_party_aliases": ["FixtureCorp"]},
    }
    if extra_config:
        cfg.update(extra_config)

    config_path = tmp_path / "playbook.config.yaml"
    config_path.write_text(yaml.dump(cfg), encoding="utf-8")

    out_dir = tmp_path / "out"
    return corpus_dir, config_path, out_dir


# ---------------------------------------------------------------------------
# 1. Explicit perspective in config -> projected playbook carries it verbatim.
# ---------------------------------------------------------------------------


def test_perspective_from_config(tmp_path: Path) -> None:
    """A config with a full ``perspective:`` block must have both fields
    threaded verbatim through ``project_playbook`` into the assembled
    document -- previously ``project_playbook`` never passed ``perspective``
    to ``assemble_playbook`` at all, so a "who is us" OPF instance could only
    be produced by hand-editing JSON.
    """
    corpus_dir, config_path, out_dir = _make_corpus(
        tmp_path,
        extra_config={
            "perspective": {
                "party": "FixtureCorp",
                "counterparty_type": "public university",
            }
        },
    )

    taxonomy = load_taxonomy(_TAXONOMY_PATH)
    cfg = load_config(config_path)
    assert cfg.perspective.party == "FixtureCorp"
    assert cfg.perspective.counterparty_type == "public university"

    mine_corpus(corpus_dir=corpus_dir, config=cfg, taxonomy=taxonomy, out_dir=out_dir)
    playbook = project_playbook(out_dir=out_dir, config=cfg, taxonomy=taxonomy)

    assert playbook["perspective"] == {
        "party": "FixtureCorp",
        "counterparty_type": "public university",
    }


# ---------------------------------------------------------------------------
# 2. No perspective block, but our_party_aliases set -> party defaults;
#    counterparty_type is never fabricated.
# ---------------------------------------------------------------------------


def test_perspective_default_from_aliases(tmp_path: Path) -> None:
    """A config with no ``perspective:`` section but with
    ``provenance.our_party_aliases`` set must default ``perspective.party``
    from the first alias -- ``counterparty_type`` has no derivable default
    and must stay unset (never fabricated).

    This is a config-level default only: ``spec/playbook.schema-0.2.json``
    requires ``party`` AND ``counterparty_type`` together or neither, so a
    party-only value is NOT enough to populate the assembled playbook's
    top-level ``perspective`` key (see ``test_perspective_omitted_when_only_party_known``
    below for the assembled-document side of this same rule).
    """
    tax_dst = tmp_path / "taxonomy.yaml"
    tax_dst.write_text(_TAXONOMY_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    config_path = tmp_path / "playbook.config.yaml"
    config_path.write_text(
        """
agreement_type:
  id: educational-affiliation
  name: "Educational Affiliation Agreement"
baseline:
  template: null
taxonomy: taxonomy.yaml
provenance:
  our_party_aliases: ["FixtureCorp", "FixtureCorp Holdings, LLC"]
""",
        encoding="utf-8",
    )

    cfg = load_config(config_path)

    assert cfg.perspective.party == "FixtureCorp"
    assert cfg.perspective.counterparty_type is None


def test_perspective_omitted_when_only_party_known(tmp_path: Path) -> None:
    """When only ``party`` is known (defaulted from aliases, no explicit
    ``counterparty_type``), the assembled playbook must omit ``perspective``
    entirely rather than emit a schema-invalid partial object or fabricate a
    ``counterparty_type``.
    """
    corpus_dir, config_path, out_dir = _make_corpus(tmp_path)

    taxonomy = load_taxonomy(_TAXONOMY_PATH)
    cfg = load_config(config_path)
    assert cfg.perspective.party == "FixtureCorp"
    assert cfg.perspective.counterparty_type is None

    mine_corpus(corpus_dir=corpus_dir, config=cfg, taxonomy=taxonomy, out_dir=out_dir)
    playbook = project_playbook(out_dir=out_dir, config=cfg, taxonomy=taxonomy)

    assert "perspective" not in playbook


# ---------------------------------------------------------------------------
# 3. scaffold_config surfaces a perspective block for humans to fill in.
# ---------------------------------------------------------------------------


def test_scaffold_config_includes_perspective(tmp_path: Path) -> None:
    """``staging.scaffold_config()`` must surface a ``perspective:`` block
    in the scaffolded config file so a human authoring a new config knows
    the field exists -- commented and blank (not an active, fabricated
    value), since neither ``party`` nor ``counterparty_type`` has a safe
    placeholder to ship active by default.
    """
    src = tmp_path / "src"
    src.mkdir()
    out = tmp_path / "out"

    scaffold_config(src, out)

    written = (out / "playbook.config.yaml").read_text(encoding="utf-8")
    assert "perspective:" in written
    assert "party" in written
    assert "counterparty_type" in written

    # Every line of the appended perspective block must be commented out --
    # a "blank" block, not an active (and therefore fabricated) value. The
    # block starts at the first line mentioning "perspective" as its own
    # marker (the active `agreement_type`/`baseline`/etc. sections above it
    # never mention "perspective" at all).
    lines = written.splitlines()
    start = next(i for i, line in enumerate(lines) if "perspective" in line)
    perspective_block = lines[start:]
    assert perspective_block, "expected a non-empty perspective block"
    assert all(line.strip().startswith("#") for line in perspective_block), (
        f"expected every perspective-block line to be commented out; got {perspective_block}"
    )
    assert any("counterparty_type" in line for line in perspective_block)
    assert any("party" in line for line in perspective_block)

    # And the scaffolded file must still be valid, parseable YAML (the
    # comment block must not corrupt the rest of the document).
    parsed = yaml.safe_load(written)
    assert isinstance(parsed, dict)
    assert "perspective" not in parsed
