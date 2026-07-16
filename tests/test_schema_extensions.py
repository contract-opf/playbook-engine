"""Tests for the x_* vendor-extension namespace (issue #180).

schema-0.2 closes every object with ``additionalProperties: false``; the
``x_*`` namespace is the sanctioned escape hatch so adopters can attach
vendor fields without forking the standard. Extensions are allowed at
exactly the eight levels listed in #180 — and nowhere hash-integrity or
mechanical resolvability depends on a closed shape (identity, citations,
agreement_type, taxonomy entries, compiler).

x_* fields ARE content: they participate in identity.content_hash and the
section digests, so two documents differing only in an x_* value have
different identities.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from playbook_engine.canonicalize import content_hash
from playbook_engine.validator import validate_document

FIXTURES = Path(__file__).parent.parent / "examples" / "fixtures"


def _minimal() -> dict[str, Any]:
    with (FIXTURES / "valid_v0_2_minimal.json").open() as f:
        return json.load(f)


def _assert_valid(doc: dict[str, Any], where: str) -> None:
    result = validate_document(doc)
    assert result.ok, f"x_* extension rejected at {where}: " + "; ".join(
        str(e) for e in result.errors
    )


def test_x_field_valid_at_each_level() -> None:
    """`x_vendor_note` must validate at each of the 8 sanctioned levels."""
    doc = _minimal()

    # 1. document root
    doc["x_vendor_note"] = "v"
    # 2. ClausePosition (evidence.clauses[] item)
    clause = doc["evidence"]["clauses"][0]
    clause["x_vendor_note"] = "v"
    # 3. Observation (observed_positions[] item)
    clause["observed_positions"][0]["x_vendor_note"] = "v"
    # 4. ClauseConcept (clause_library[] item)
    doc["evidence"]["clause_library"] = [
        {
            "concept_id": "concept.indemnification",
            "taxonomy_id": "indemnification",
            "description": "Who bears third-party claim risk.",
            "accepted_forms": [],
            "x_vendor_note": "v",
        }
    ]
    # 5. posture
    doc["posture"]["x_vendor_note"] = "v"
    # 6. floor and floor.invariants[] item
    doc["floor"]["x_vendor_note"] = "v"
    doc["floor"]["invariants"][0]["x_vendor_note"] = "v"
    # 7. curation pin (curation.pins[] item)
    doc["curation"] = {
        "pins": [
            {
                "clause_id": "clause.indemnification",
                "item_id": "C1",
                "position": "hold firm",
                "baseline_stance": "usually_held",
                "pinned_at": "2026-01-01T00:00:00Z",
                "x_vendor_note": "v",
            }
        ]
    }
    # 8. corpus.documents[] item
    doc["corpus"]["documents"][0]["x_vendor_note"] = "v"

    _assert_valid(doc, "the 8 sanctioned levels")


def test_x_field_rejected_in_identity() -> None:
    """identity is hash-integrity surface — extensions stay out."""
    doc = _minimal()
    doc["identity"] = {
        "content_hash": "sha256:" + "0" * 64,
        "section_digests": {
            "evidence": "sha256:" + "0" * 64,
            "posture": "sha256:" + "0" * 64,
            "floor": "sha256:" + "0" * 64,
        },
        "x_foo": "v",
    }
    result = validate_document(doc)
    assert not result.ok


def test_x_field_rejected_in_citation() -> None:
    """Citations must stay mechanically resolvable — extensions stay out."""
    doc = _minimal()
    observation = doc["evidence"]["clauses"][0]["observed_positions"][0]
    observation["example_ref"]["x_foo"] = "v"
    result = validate_document(doc)
    assert not result.ok


def test_unknown_nonprefixed_field_still_fails() -> None:
    """The escape hatch is x_* only; fail-loud stays for everything else."""
    doc = _minimal()
    doc["vendor_note"] = "v"
    result = validate_document(doc)
    assert not result.ok


def test_x_field_changes_content_hash() -> None:
    """x_* fields are content: they participate in content_hash."""
    doc = _minimal()
    extended = copy.deepcopy(doc)
    extended["x_vendor_note"] = "v"
    assert content_hash(doc) != content_hash(extended)
