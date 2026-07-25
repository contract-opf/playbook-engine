"""Tests that CUAD-derived taxonomy data carries required CC-BY-4.0 attribution.

CUAD v1 (The Atticus Project) is distributed under CC-BY-4.0, which requires
attribution. These tests guard against that attribution silently disappearing:
a NOTICE file at the repo root, a reference to it from LICENSE, and a
CC-BY-4.0 header on every taxonomy YAML whose `source` starts with "CUAD".
"""

from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).parent.parent
NOTICE = REPO_ROOT / "NOTICE"
LICENSE = REPO_ROOT / "LICENSE"
TAXONOMY_DIR = REPO_ROOT / "spec" / "taxonomy"


def test_notice_exists_and_attributes_cuad() -> None:
    assert NOTICE.exists(), "NOTICE file must exist at repo root"
    text = NOTICE.read_text(encoding="utf-8")
    assert "CUAD" in text
    assert "The Atticus Project" in text
    assert "CC-BY-4.0" in text
    assert "http" in text


def test_license_references_notice() -> None:
    text = LICENSE.read_text(encoding="utf-8")
    assert "NOTICE" in text


def test_cuad_derived_taxonomies_carry_header() -> None:
    yaml_paths = sorted(TAXONOMY_DIR.glob("*.yaml"))
    assert yaml_paths, "expected at least one taxonomy YAML under spec/taxonomy/"

    checked_any = False
    for path in yaml_paths:
        text = path.read_text(encoding="utf-8")
        data = yaml.safe_load(text)
        source = (data or {}).get("source", "")
        if isinstance(source, str) and source.startswith("CUAD"):
            checked_any = True
            assert "CC-BY-4.0" in text, f"{path.name} has source={source!r} but no CC-BY-4.0 header"

    assert checked_any, "expected at least one CUAD-derived taxonomy YAML"
