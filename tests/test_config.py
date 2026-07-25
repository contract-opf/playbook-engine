"""Tests for the per-agreement-type config loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from playbook_engine.clause_position_compiler import MIN_EVIDENCE_N
from playbook_engine.config import ConfigError, EngineConfig, load_config
from playbook_engine.llm_segmenter import DEFAULT_MODEL

AFFILIATION_CONFIG = (
    Path(__file__).parent.parent / "examples" / "affiliation-config" / "playbook.config.yaml"
)
TAXONOMY_PATH = Path(__file__).parent.parent / "spec" / "taxonomy" / "affiliation-agreement.yaml"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_config(tmp_path: Path, content: str) -> Path:
    cfg = tmp_path / "playbook.config.yaml"
    cfg.write_text(content, encoding="utf-8")
    return cfg


def _minimal_config(tmp_path: Path, *, with_template: bool = False) -> Path:
    tax_src = TAXONOMY_PATH
    tax_dst = tmp_path / "taxonomy.yaml"
    tax_dst.write_text(tax_src.read_text(), encoding="utf-8")

    tpl_line = "  template: null"
    if with_template:
        tpl = tmp_path / "template.docx"
        tpl.write_bytes(b"fake")
        tpl_line = "  template: template.docx"

    return _write_config(
        tmp_path,
        f"""
agreement_type:
  id: educational-affiliation
  name: "Educational Affiliation Agreement"
baseline:
{tpl_line}
taxonomy: taxonomy.yaml
provenance:
  our_party_aliases: ["FixtureCorp", "FixtureCorp Holdings, LLC"]
""",
    )


# ---------------------------------------------------------------------------
# Happy-path: bundled affiliation config
# ---------------------------------------------------------------------------


def test_load_affiliation_config() -> None:
    """The shipped affiliation config must load without errors."""
    cfg = load_config(AFFILIATION_CONFIG)
    assert isinstance(cfg, EngineConfig)
    assert cfg.agreement_type.id == "educational-affiliation"
    assert cfg.agreement_type.name == "Educational Affiliation Agreement"


def test_affiliation_config_taxonomy_path_exists() -> None:
    cfg = load_config(AFFILIATION_CONFIG)
    assert cfg.taxonomy_path.exists()
    assert cfg.taxonomy_path == TAXONOMY_PATH.resolve()


def test_affiliation_config_emergent_baseline() -> None:
    """The shipped affiliation config has no template (template: null)."""
    cfg = load_config(AFFILIATION_CONFIG)
    assert not cfg.baseline.has_canonical_template
    assert cfg.baseline.template_path is None


def test_affiliation_config_party_aliases() -> None:
    cfg = load_config(AFFILIATION_CONFIG)
    aliases = cfg.provenance.our_party_aliases
    assert "FixtureCorp" in aliases
    assert len(aliases) >= 1


def test_affiliation_config_agreement_type_aliases() -> None:
    """Issue #142: the shipped config declares 'eiaa' as an alias of the
    canonical 'educational-affiliation' id — this is the shared cross-tool
    key a consuming app (whose own registry key is 'eiaa') matches
    on instead of a hand-joined mapping."""
    cfg = load_config(AFFILIATION_CONFIG)
    assert cfg.agreement_type.aliases == ["eiaa"]


def test_config_path_is_absolute() -> None:
    cfg = load_config(AFFILIATION_CONFIG)
    assert cfg.config_path.is_absolute()
    assert cfg.taxonomy_path.is_absolute()


# ---------------------------------------------------------------------------
# Minimal configs
# ---------------------------------------------------------------------------


def test_minimal_config_loads(tmp_path: Path) -> None:
    path = _minimal_config(tmp_path)
    cfg = load_config(path)
    assert cfg.agreement_type.id == "educational-affiliation"
    assert cfg.provenance.our_party_aliases == ["FixtureCorp", "FixtureCorp Holdings, LLC"]


def test_config_with_template_path(tmp_path: Path) -> None:
    path = _minimal_config(tmp_path, with_template=True)
    cfg = load_config(path)
    assert cfg.baseline.has_canonical_template
    assert cfg.baseline.template_path is not None
    assert cfg.baseline.template_path.exists()


def test_no_party_aliases_defaults_to_empty(tmp_path: Path) -> None:
    tax = tmp_path / "taxonomy.yaml"
    tax.write_text(TAXONOMY_PATH.read_text(), encoding="utf-8")
    path = _write_config(
        tmp_path,
        """
agreement_type:
  id: test-type
  name: "Test Agreement"
baseline:
  template: null
taxonomy: taxonomy.yaml
""",
    )
    cfg = load_config(path)
    assert cfg.provenance.our_party_aliases == []


def test_no_agreement_type_aliases_defaults_to_empty(tmp_path: Path) -> None:
    path = _minimal_config(tmp_path)
    cfg = load_config(path)
    assert cfg.agreement_type.aliases == []


def test_agreement_type_aliases_not_list_raises(tmp_path: Path) -> None:
    tax = tmp_path / "taxonomy.yaml"
    tax.write_text(TAXONOMY_PATH.read_text(), encoding="utf-8")
    path = _write_config(
        tmp_path,
        """
agreement_type:
  id: test-type
  name: "Test Agreement"
  aliases: "eiaa"
baseline:
  template: null
taxonomy: taxonomy.yaml
""",
    )
    with pytest.raises(ConfigError, match="agreement_type.aliases must be a list"):
        load_config(path)


# ---------------------------------------------------------------------------
# segmentation (issue #80)
# ---------------------------------------------------------------------------


def test_no_segmentation_block_defaults_to_all_false(tmp_path: Path) -> None:
    """No ``segmentation:`` section (every existing fixture) -> all flags False."""
    path = _minimal_config(tmp_path)
    cfg = load_config(path)
    assert cfg.segmentation.llm is False
    assert cfg.segmentation.batch is False
    assert cfg.segmentation.cache is False
    assert cfg.segmentation.normalize_trail is False


def test_segmentation_block_parsed(tmp_path: Path) -> None:
    tax = tmp_path / "taxonomy.yaml"
    tax.write_text(TAXONOMY_PATH.read_text(), encoding="utf-8")
    path = _write_config(
        tmp_path,
        """
agreement_type:
  id: test-type
  name: "Test Agreement"
baseline:
  template: null
taxonomy: taxonomy.yaml
segmentation:
  llm: true
  batch: true
  cache: true
  normalize_trail: true
""",
    )
    cfg = load_config(path)
    assert cfg.segmentation.llm is True
    assert cfg.segmentation.batch is True
    assert cfg.segmentation.cache is True
    assert cfg.segmentation.normalize_trail is True


def test_segmentation_not_mapping_raises(tmp_path: Path) -> None:
    tax = tmp_path / "taxonomy.yaml"
    tax.write_text(TAXONOMY_PATH.read_text(), encoding="utf-8")
    path = _write_config(
        tmp_path,
        """
agreement_type:
  id: test-type
  name: "Test Agreement"
baseline:
  template: null
taxonomy: taxonomy.yaml
segmentation: not-a-map
""",
    )
    with pytest.raises(ConfigError, match="segmentation must be a mapping"):
        load_config(path)


# ---------------------------------------------------------------------------
# segmentation.model (issue #131)
# ---------------------------------------------------------------------------


def test_no_segmentation_block_model_defaults_to_shared_constant(tmp_path: Path) -> None:
    """No ``segmentation:`` section -> ``model`` defaults to the shared
    ``llm_segmenter.DEFAULT_MODEL`` constant, so existing configs/fixtures
    keep calling the exact model they always have."""
    path = _minimal_config(tmp_path)
    cfg = load_config(path)
    assert cfg.segmentation.model == DEFAULT_MODEL


def test_segmentation_block_without_model_defaults_to_shared_constant(tmp_path: Path) -> None:
    """A ``segmentation:`` block that omits ``model`` -> same default as no block at all."""
    tax = tmp_path / "taxonomy.yaml"
    tax.write_text(TAXONOMY_PATH.read_text(), encoding="utf-8")
    path = _write_config(
        tmp_path,
        """
agreement_type:
  id: test-type
  name: "Test Agreement"
baseline:
  template: null
taxonomy: taxonomy.yaml
segmentation:
  llm: true
""",
    )
    cfg = load_config(path)
    assert cfg.segmentation.model == DEFAULT_MODEL


def test_segmentation_model_override_parsed(tmp_path: Path) -> None:
    """``segmentation.model`` overrides the default — model is config data, not code
    (issue #131's core fix: no config/env override existed for the hardcoded model id)."""
    tax = tmp_path / "taxonomy.yaml"
    tax.write_text(TAXONOMY_PATH.read_text(), encoding="utf-8")
    path = _write_config(
        tmp_path,
        """
agreement_type:
  id: test-type
  name: "Test Agreement"
baseline:
  template: null
taxonomy: taxonomy.yaml
segmentation:
  llm: true
  model: "anthropic.claude-opus-4-8-bedrock"
""",
    )
    cfg = load_config(path)
    assert cfg.segmentation.model == "anthropic.claude-opus-4-8-bedrock"


def test_segmentation_model_empty_string_raises(tmp_path: Path) -> None:
    tax = tmp_path / "taxonomy.yaml"
    tax.write_text(TAXONOMY_PATH.read_text(), encoding="utf-8")
    path = _write_config(
        tmp_path,
        """
agreement_type:
  id: test-type
  name: "Test Agreement"
baseline:
  template: null
taxonomy: taxonomy.yaml
segmentation:
  model: ""
""",
    )
    with pytest.raises(ConfigError, match="segmentation.model must be a non-empty string"):
        load_config(path)


# ---------------------------------------------------------------------------
# provenance.min_evidence_n (issue #144)
# ---------------------------------------------------------------------------


def test_no_min_evidence_n_defaults_to_compiler_constant(tmp_path: Path) -> None:
    """No ``provenance.min_evidence_n`` key -> defaults to
    ``clause_position_compiler.MIN_EVIDENCE_N`` so an existing config keeps
    enforcing the same evidence-depth floor it always has."""
    path = _minimal_config(tmp_path)
    cfg = load_config(path)
    assert cfg.provenance.min_evidence_n == MIN_EVIDENCE_N


def test_min_evidence_n_override_parsed(tmp_path: Path) -> None:
    tax = tmp_path / "taxonomy.yaml"
    tax.write_text(TAXONOMY_PATH.read_text(), encoding="utf-8")
    path = _write_config(
        tmp_path,
        """
agreement_type:
  id: test-type
  name: "Test Agreement"
baseline:
  template: null
taxonomy: taxonomy.yaml
provenance:
  min_evidence_n: 3
""",
    )
    cfg = load_config(path)
    assert cfg.provenance.min_evidence_n == 3


def test_min_evidence_n_zero_raises(tmp_path: Path) -> None:
    tax = tmp_path / "taxonomy.yaml"
    tax.write_text(TAXONOMY_PATH.read_text(), encoding="utf-8")
    path = _write_config(
        tmp_path,
        """
agreement_type:
  id: test-type
  name: "Test Agreement"
baseline:
  template: null
taxonomy: taxonomy.yaml
provenance:
  min_evidence_n: 0
""",
    )
    with pytest.raises(ConfigError, match="min_evidence_n must be a positive integer"):
        load_config(path)


def test_min_evidence_n_not_an_integer_raises(tmp_path: Path) -> None:
    tax = tmp_path / "taxonomy.yaml"
    tax.write_text(TAXONOMY_PATH.read_text(), encoding="utf-8")
    path = _write_config(
        tmp_path,
        """
agreement_type:
  id: test-type
  name: "Test Agreement"
baseline:
  template: null
taxonomy: taxonomy.yaml
provenance:
  min_evidence_n: "two"
""",
    )
    with pytest.raises(ConfigError, match="min_evidence_n must be a positive integer"):
        load_config(path)


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


def test_missing_config_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nonexistent.yaml")


def test_invalid_yaml_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("{{not valid yaml: [", encoding="utf-8")
    with pytest.raises(ConfigError, match="not valid YAML"):
        load_config(bad)


def test_missing_agreement_type_raises(tmp_path: Path) -> None:
    tax = tmp_path / "taxonomy.yaml"
    tax.write_text(TAXONOMY_PATH.read_text(), encoding="utf-8")
    path = _write_config(tmp_path, "baseline:\n  template: null\ntaxonomy: taxonomy.yaml\n")
    with pytest.raises(ConfigError, match="agreement_type"):
        load_config(path)


def test_invalid_agreement_type_id_raises(tmp_path: Path) -> None:
    tax = tmp_path / "taxonomy.yaml"
    tax.write_text(TAXONOMY_PATH.read_text(), encoding="utf-8")
    path = _write_config(
        tmp_path,
        """
agreement_type:
  id: "Not A Valid Slug!"
  name: "Test"
baseline:
  template: null
taxonomy: taxonomy.yaml
""",
    )
    with pytest.raises(ConfigError, match=r"\^.*\[a-z0-9-\]"):
        load_config(path)


def test_missing_taxonomy_path_raises(tmp_path: Path) -> None:
    path = _write_config(
        tmp_path,
        """
agreement_type:
  id: test-type
  name: "Test"
baseline:
  template: null
taxonomy: does-not-exist.yaml
""",
    )
    with pytest.raises(ConfigError, match="taxonomy file not found"):
        load_config(path)


def test_taxonomy_path_is_directory_raises(tmp_path: Path) -> None:
    """NB-1 regression: a directory must not pass the taxonomy path check."""
    path = _write_config(
        tmp_path,
        f"""
agreement_type:
  id: test-type
  name: "Test"
baseline:
  template: null
taxonomy: {tmp_path}
""",
    )
    with pytest.raises(ConfigError, match="taxonomy file not found"):
        load_config(path)


def test_template_path_not_found_raises(tmp_path: Path) -> None:
    tax = tmp_path / "taxonomy.yaml"
    tax.write_text(TAXONOMY_PATH.read_text(), encoding="utf-8")
    path = _write_config(
        tmp_path,
        """
agreement_type:
  id: test-type
  name: "Test"
baseline:
  template: ghost-template.docx
taxonomy: taxonomy.yaml
""",
    )
    with pytest.raises(ConfigError, match="baseline.template not found"):
        load_config(path)


def test_agreement_type_not_mapping_raises(tmp_path: Path) -> None:
    """NB-3: agreement_type must be a mapping."""
    tax = tmp_path / "taxonomy.yaml"
    tax.write_text(TAXONOMY_PATH.read_text(), encoding="utf-8")
    path = _write_config(
        tmp_path,
        "agreement_type: just-a-string\nbaseline:\n  template: null\ntaxonomy: taxonomy.yaml\n",
    )
    with pytest.raises(ConfigError, match="agreement_type must be a mapping"):
        load_config(path)


def test_provenance_not_mapping_raises(tmp_path: Path) -> None:
    """NB-3: provenance must be a mapping when present."""
    tax = tmp_path / "taxonomy.yaml"
    tax.write_text(TAXONOMY_PATH.read_text(), encoding="utf-8")
    path = _write_config(
        tmp_path,
        "agreement_type:\n  id: test\n  name: T\nbaseline:\n  template: null\ntaxonomy: taxonomy.yaml\nprovenance: not-a-map\n",
    )
    with pytest.raises(ConfigError, match="provenance must be a mapping"):
        load_config(path)


def test_non_mapping_root_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("- just\n- a\n- list\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="YAML mapping"):
        load_config(bad)


# ---------------------------------------------------------------------------
# builtin: taxonomy scheme — relocatable configs (issue #130)
# ---------------------------------------------------------------------------


def _config_with_taxonomy(tmp_path: Path, taxonomy_value: str) -> Path:
    return _write_config(
        tmp_path,
        f"""
agreement_type:
  id: test-type
  name: "Test Agreement"
baseline:
  template: null
taxonomy: {taxonomy_value}
""",
    )


def test_builtin_taxonomy_resolves_regardless_of_config_location(tmp_path: Path) -> None:
    """A ``builtin:`` taxonomy resolves against the engine's own bundled
    spec/taxonomy/ dir, not the config file's directory — so it keeps
    working no matter where the config file itself lives.
    """
    nested = tmp_path / "somewhere" / "far" / "from" / "the" / "repo"
    nested.mkdir(parents=True)
    path = _config_with_taxonomy(nested, "builtin:affiliation-agreement.yaml")
    cfg = load_config(path)
    assert cfg.taxonomy_path == TAXONOMY_PATH.resolve()


def test_unknown_builtin_taxonomy_raises(tmp_path: Path) -> None:
    path = _config_with_taxonomy(tmp_path, "builtin:does-not-exist.yaml")
    with pytest.raises(ConfigError, match="builtin taxonomy"):
        load_config(path)


def test_builtin_scheme_with_empty_name_raises(tmp_path: Path) -> None:
    path = _config_with_taxonomy(tmp_path, '"builtin:"')
    with pytest.raises(ConfigError, match="'builtin:' scheme requires a name"):
        load_config(path)


def test_shipped_affiliation_config_survives_being_copied_elsewhere(tmp_path: Path) -> None:
    """Regression for the audit finding (issue #130): the shipped example
    config used to reference its taxonomy via a repo-relative
    ``../../spec/taxonomy/...`` path, which dangled the moment a user copied
    the config next to their own corpus. Copy the actual shipped config file
    far away from the repo and confirm it still loads.
    """
    dest_dir = tmp_path / "a-users-corpus-folder"
    dest_dir.mkdir()
    copied = dest_dir / "playbook.config.yaml"
    copied.write_text(AFFILIATION_CONFIG.read_text(encoding="utf-8"), encoding="utf-8")

    cfg = load_config(copied)

    assert cfg.taxonomy_path.exists()
    assert cfg.taxonomy_path == TAXONOMY_PATH.resolve()
