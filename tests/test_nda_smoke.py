"""Hermetic no-LLM smoke test: NDA second-agreement-type (issue #111).

The engine claims to be agreement-type-general, but only one type
(Educational Affiliation) has ever actually run end-to-end. This proves the
claim on a second, wholly synthetic agreement type -- the committed NDA
example at examples/nda/ (see examples/nda/README.md and
docs/QUICK-COMPILE.md) -- through the real CLI: lint-corpus -> mine ->
project -> validate, entirely on the deterministic no-LLM pipeline path
(stub scope judge, Jaccard classification fast-path, heuristic deviation
stub -- same stub judges as tests/test_golden_affiliation.py and
tests/test_cli.py's mine/project acceptance tests).

Hermetic:
  - No network.
  - No ANTHROPIC_API_KEY read: config.smoke.yaml omits config.yaml's
    `segmentation.agent: true` (a store-backed agent loop that cannot run
    headlessly), leaving segmentation.llm off -- `_llm_segmentation_kwargs`
    returns {} and never touches the environment in that case.
  - All output is written under pytest's tmp_path; nothing touches the
    repo's gitignored out/ or any cache outside tmp_path.

Marked @pytest.mark.smoke; run directly with
``pytest tests/test_nda_smoke.py -q -m smoke`` or via ``make smoke-nda``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner
from click.testing import Result as CliResult

from playbook_engine.cli import cli

_REPO_ROOT = Path(__file__).resolve().parent.parent
_NDA_DIR = _REPO_ROOT / "examples" / "nda"
_CORPUS_DIR = _NDA_DIR / "corpus"
_SMOKE_CONFIG = _NDA_DIR / "config.smoke.yaml"
_TAXONOMY_PATH = _REPO_ROOT / "spec" / "taxonomy" / "nda.yaml"

_TEMPLATE_STANDARDS_RE = re.compile(r"template standards: (\d+) clause\(s\) classified")

_RTF_PROLOGUE = (
    r"{\rtf1\ansi\deff0"
    r"{\fonttbl{\f0\froman\fcharset0 Times New Roman;}}"
    r"\f0\fs24 "
)
_RTF_EPILOGUE = r"}"


def _rtf(body: str) -> str:
    return _RTF_PROLOGUE + body + _RTF_EPILOGUE


def _invoke(args: list[str]) -> CliResult:
    return CliRunner().invoke(cli, args)


# ---------------------------------------------------------------------------
# Main smoke run: lint-corpus -> mine -> project -> validate
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_nda_smoke_full_pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """lint-corpus -> mine -> project -> validate over the real NDA example.

    Asserts: every stage exits 0; template mode actually classified clauses
    (N > 0, not silently degraded to emergent); the compiled playbook.opf.json
    validates; the playbook is evidence-only (posture/floor empty -- the
    smoke run must not fabricate either).
    """
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    out_dir = tmp_path / "out"

    lint_result = _invoke(["lint-corpus", str(_CORPUS_DIR), "--config", str(_SMOKE_CONFIG)])
    assert lint_result.exit_code == 0, f"lint-corpus failed:\n{lint_result.output}"

    mine_result = _invoke(
        ["mine", str(_CORPUS_DIR), "--config", str(_SMOKE_CONFIG), "--out", str(out_dir)]
    )
    assert mine_result.exit_code == 0, f"mine failed:\n{mine_result.output}"

    # config.yaml's own comment warns that N=0 here means template mode
    # silently degraded to emergent -- this is the smoke test's core claim.
    match = _TEMPLATE_STANDARDS_RE.search(mine_result.output)
    assert match is not None, (
        "'template standards: N clause(s) classified' line missing from "
        f"mine output:\n{mine_result.output}"
    )
    assert int(match.group(1)) > 0, (
        "template mode silently degraded to emergent (0 clauses classified "
        f"against the NDA standard-form template):\n{mine_result.output}"
    )

    project_result = _invoke(["project", str(out_dir), "--config", str(_SMOKE_CONFIG)])
    assert project_result.exit_code == 0, f"project failed:\n{project_result.output}"

    playbook_path = out_dir / "playbook.opf.json"
    assert playbook_path.exists(), "project did not write playbook.opf.json"

    validate_result = _invoke(["validate", str(playbook_path)])
    assert validate_result.exit_code == 0, f"validate failed:\n{validate_result.output}"

    playbook = json.loads(playbook_path.read_text(encoding="utf-8"))
    assert playbook["agreement_type"]["id"] == "nda"
    assert playbook["evidence"]["clauses"], "no clauses observed at all"
    # Evidence-only: no posture interview or floor sign step is run here, so
    # the compiled playbook must not carry fabricated negotiation intent or
    # hard lines.
    assert playbook["posture"] == {}, "smoke run must not fabricate posture"
    assert playbook["floor"] == {}, "smoke run must not fabricate floor"


# ---------------------------------------------------------------------------
# Reviewer-gate mutation check: a bogus template must degrade to N=0,
# proving the N>0 assertion above actually bites.
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_nda_smoke_bogus_template_degrades_to_zero_clauses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A template with no classifiable clause content reports 0 clauses
    classified -- proving the N>0 assertion in the main smoke test is a
    real, biting check rather than one that would pass vacuously."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    bogus_template = tmp_path / "bogus-template.rtf"
    bogus_template.write_text(
        _rtf(r"Lorem ipsum dolor sit amet, consectetur adipiscing elit.\par "),
        encoding="utf-8",
    )

    bogus_cfg = yaml.safe_load(_SMOKE_CONFIG.read_text(encoding="utf-8"))
    bogus_cfg["baseline"]["template"] = str(bogus_template)
    bogus_cfg["taxonomy"] = str(_TAXONOMY_PATH)  # absolute: config now lives in tmp_path
    bogus_config_path = tmp_path / "config.bogus.yaml"
    bogus_config_path.write_text(yaml.dump(bogus_cfg), encoding="utf-8")

    out_dir = tmp_path / "out-bogus"
    result = _invoke(
        ["mine", str(_CORPUS_DIR), "--config", str(bogus_config_path), "--out", str(out_dir)]
    )
    assert result.exit_code == 0, f"mine failed:\n{result.output}"

    match = _TEMPLATE_STANDARDS_RE.search(result.output)
    assert match is not None, f"template-standards line missing:\n{result.output}"
    n = int(match.group(1))
    assert n == 0, (
        f"expected the bogus template to classify zero clauses (got {n}) -- "
        "if this fails, the fixture no longer proves the main test's N>0 "
        "assertion bites"
    )
