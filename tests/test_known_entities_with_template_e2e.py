"""End-to-end: mine + project with ``known_entities`` AND a template (issue #264).

The 2.0.0 ordinal document alias also aliased the reserved ``template``
document id. Every unit and example test passed, because none ran mine +
project with ``provenance.known_entities`` set *and* a template baseline at
once; only the first real-corpus run found it. This test is that combination,
over the synthetic NDA example with its fictional parties as the known
entities.

The check is differential: the same corpus mined with and without the
entities must yield the same template-derived standards and the same number of
precedent records. Only names and ids may differ.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest
import yaml
from click.testing import CliRunner
from click.testing import Result as CliResult

from playbook_engine.cli import cli

_NDA_DIR = Path(__file__).resolve().parent.parent / "examples" / "nda"
_CORPUS_DIR = _NDA_DIR / "corpus"
_SMOKE_CONFIG = _NDA_DIR / "config.smoke.yaml"
_TAXONOMY = Path(__file__).resolve().parent.parent / "spec" / "taxonomy" / "nda.yaml"
_TEMPLATE_RE = re.compile(r"template standards: (\d+) clause")

# The example's invented counterparties (the folder names, title-cased).
_ENTITIES = [
    "Beta Industries",
    "Delta Ventures",
    "Epsilon Systems",
    "Gamma Holdings",
    "Theta Logistics",
    "Zeta Diagnostics",
]


def _invoke(args: list[str]) -> CliResult:
    return CliRunner().invoke(cli, args)


def _config(tmp_path: Path, name: str, known_entities: list[str]) -> Path:
    cfg = yaml.safe_load(_SMOKE_CONFIG.read_text(encoding="utf-8"))
    cfg["taxonomy"] = str(_TAXONOMY)
    cfg["baseline"]["template"] = str(_NDA_DIR / "standard-form.rtf")
    cfg["provenance"]["known_entities"] = known_entities
    path = tmp_path / f"{name}.yaml"
    path.write_text(yaml.dump(cfg), encoding="utf-8")
    return path


def _mine_and_project(tmp_path: Path, name: str, known_entities: list[str]) -> tuple[Path, int]:
    config = _config(tmp_path, name, known_entities)
    out_dir = tmp_path / name
    # Never the machine-global default registry: a test must not write there.
    registry = str(tmp_path / f"{name}-entity-registry.json")
    mined = _invoke(
        [
            "mine",
            str(_CORPUS_DIR),
            "--config",
            str(config),
            "--out",
            str(out_dir),
            "--entity-registry",
            registry,
        ]
    )
    assert mined.exit_code == 0, f"mine failed:\n{mined.output}"
    match = _TEMPLATE_RE.search(mined.output)
    assert match is not None, mined.output
    projected = _invoke(["project", str(out_dir), "--config", str(config)])
    assert projected.exit_code == 0, f"project failed:\n{projected.output}"
    return out_dir, int(match.group(1))


def _standards(playbook: dict[str, Any]) -> dict[str, str]:
    return {
        c["taxonomy_id"]: c["our_standard"]["text"]
        for c in playbook["evidence"]["clauses"]
        if isinstance(c.get("our_standard"), dict)
    }


@pytest.mark.smoke
def test_mine_and_project_with_known_entities_and_a_template(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    plain_dir, plain_n = _mine_and_project(tmp_path, "plain", [])
    safe_dir, safe_n = _mine_and_project(tmp_path, "safe", _ENTITIES)

    assert plain_n > 0
    assert safe_n == plain_n, "known_entities changed what the template classified"

    plain = json.loads((plain_dir / "playbook.opf.json").read_text(encoding="utf-8"))
    safe = json.loads((safe_dir / "playbook.opf.json").read_text(encoding="utf-8"))

    # The template is still the template: its standards survive, unaliased.
    assert _standards(safe), "no clause kept its template standard"
    assert _standards(safe) == _standards(plain)
    assert len(safe["evidence"]["precedent"]) == len(plain["evidence"]["precedent"])
    ids = {d["document_id"] for d in safe["corpus"]["documents"]}
    assert "template" not in ids, "the template is not a corpus document"
    assert not any(e.lower().replace(" ", "-") in i for i in ids for e in _ENTITIES), ids

    validated = _invoke(["validate", str(safe_dir / "playbook.opf.json")])
    assert validated.exit_code == 0, f"validate failed:\n{validated.output}"

    # Born safe: no raw entity name anywhere in the compiled playbook.
    text = json.dumps(safe).lower()
    for entity in _ENTITIES:
        assert entity.lower() not in text, entity
