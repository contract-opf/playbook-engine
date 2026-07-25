"""CI guard for the shipped examples (issue #164).

The flagship examples are the reference artifacts adopters pattern-match
against, so every `examples/*.playbook.json` must pass the engine's own
validator under its declared `opf_version`, the v0.2 flagship must actually
demonstrate the headline sections (Posture, Floor, curation, dynamics), its
internal counts must agree with its lists, and no example may carry real
company branding.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from playbook_engine.validator import validate_document

ROOT = Path(__file__).parent.parent
EXAMPLE_PATHS = sorted((ROOT / "examples").glob("*.playbook.json"))
V02_FLAGSHIP = ROOT / "examples" / "our-paper-baseline.v0.2.playbook.json"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_examples_exist() -> None:
    assert EXAMPLE_PATHS, "no examples/*.playbook.json found"
    assert V02_FLAGSHIP in EXAMPLE_PATHS


@pytest.mark.parametrize("path", EXAMPLE_PATHS, ids=lambda p: p.name)
def test_all_examples_validate(path: Path) -> None:
    """Every shipped example validates under its declared opf_version —
    the guard that would have caught the flagship being non-conformant."""
    doc = _load(path)
    result = validate_document(doc)
    blocking = [str(e) for e in result.errors if e.blocking]
    assert result.ok, f"{path.name} fails its own engine's validation: {blocking}"


def test_v02_example_demonstrates_headline_sections() -> None:
    """The v0.2 flagship must demonstrate what defines v0.2 — not ship
    empty posture/floor."""
    doc = _load(V02_FLAGSHIP)

    invariants = doc["floor"].get("invariants", [])
    assert len(invariants) >= 2, "flagship must demonstrate >=2 floor.invariants"

    posture = doc["posture"]
    assert posture.get("system_prompt", "").strip(), "flagship posture must be populated"
    interview = posture.get("generation", {}).get("interview", [])
    assert len(interview) >= 3, "flagship must carry >=3 interview entries"

    pins = doc.get("curation", {}).get("pins", [])
    assert len(pins) >= 1, "flagship must demonstrate a curation pin"

    clauses = doc["evidence"]["clauses"]
    assert any(
        obs.get("full_text") for clause in clauses for obs in clause.get("observed_positions", [])
    ), "flagship must demonstrate full_text on at least one observation"
    assert any(clause.get("negotiation_trail") for clause in clauses), (
        "flagship must demonstrate a negotiation_trail (§3.5.3)"
    )
    assert doc.get("identity", {}).get("content_hash", "").startswith("sha256:")


def test_v02_example_confidence_counts_consistent() -> None:
    """confidence.n_our_paper / n_counterparty_paper must equal the actual
    provenance counts of observed_positions — the flagship previously
    claimed counts its own lists contradicted."""
    doc = _load(V02_FLAGSHIP)
    for clause in doc["evidence"]["clauses"]:
        confidence = clause["summary"]["confidence"]
        observed = clause.get("observed_positions", [])
        n_ours = sum(1 for o in observed if o.get("provenance") == "our_paper")
        n_theirs = sum(1 for o in observed if o.get("provenance") == "counterparty_paper")
        assert confidence.get("n_our_paper") == n_ours, clause["id"]
        assert confidence.get("n_counterparty_paper") == n_theirs, clause["id"]


@pytest.mark.parametrize("path", EXAMPLE_PATHS, ids=lambda p: p.name)
def test_examples_carry_no_real_branding(path: Path) -> None:
    """Examples must not read as a real company's positions (#164/#170)."""
    text = path.read_text(encoding="utf-8")
    assert not re.search(r"exos", text, flags=re.IGNORECASE), (
        f"{path.name} carries real branding — use the fictional FixtureCorp"
    )
