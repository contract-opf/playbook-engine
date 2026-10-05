"""Shape-agnostic OPF accessors (issue #154; OPF 0.4 precedent, issue #223).

The 0.4 cases read the real compiled NDA example
(``examples/nda/playbook.opf.json``, produced by ``playbook project``); the
0.1/0.2 cases read the committed validator fixtures. No hand-built shapes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from playbook_engine.opf_accessors import (
    clause_confidence,
    clause_precedent,
    clause_stance,
    clause_trail,
    is_precedent_shape,
    playbook_clause_library,
    playbook_clauses,
    playbook_precedent,
)

_ROOT = Path(__file__).parent.parent
_NDA = _ROOT / "examples" / "nda" / "playbook.opf.json"
_FIXTURES = _ROOT / "examples" / "fixtures"


def _load(path: Path) -> dict[str, Any]:
    doc: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return doc


def test_precedent_accessors_read_a_compiled_0_4_document() -> None:
    doc = _load(_NDA)
    assert doc["opf_version"] == "0.4"
    assert is_precedent_shape(doc)
    records = playbook_precedent(doc)
    assert records == doc["evidence"]["precedent"]
    for clause in playbook_clauses(doc):
        own = clause_precedent(doc, clause)
        assert all(p["taxonomy_id"] == clause["taxonomy_id"] for p in own)
        assert len({p["document_id"] for p in own}) == clause["n_deals"]
    assert sum(len(clause_precedent(doc, c)) for c in playbook_clauses(doc)) == len(records)


def test_0_3_accessors_degrade_to_absent_on_a_0_4_clause() -> None:
    """A 0.4 clause carries no stance, confidence, trail or clause library —
    the 0.2/0.3 accessors return their documented absent values rather than
    inventing one."""
    doc = _load(_NDA)
    clause = playbook_clauses(doc)[0]
    assert clause_stance(clause) == "unknown"
    assert clause_confidence(clause) == {}
    assert clause_trail(clause) == []
    assert playbook_clause_library(doc) == []


def test_precedent_accessors_are_empty_on_older_documents() -> None:
    for name in ("valid_v0_2_minimal.json", "minimal_valid.json"):
        doc = _load(_FIXTURES / name)
        assert not is_precedent_shape(doc)
        assert playbook_precedent(doc) == []
        for clause in playbook_clauses(doc):
            assert clause_precedent(doc, clause) == []


def test_precedent_accessor_drops_non_dict_entries() -> None:
    doc = _load(_NDA)
    doc["evidence"]["precedent"] = [None, *doc["evidence"]["precedent"][:1], "x"]
    assert playbook_precedent(doc) == doc["evidence"]["precedent"][1:2]
