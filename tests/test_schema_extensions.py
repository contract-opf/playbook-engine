"""Tests for the x_* vendor-extension namespace (issue #180).

The OPF 0.5 schema closes every object with ``additionalProperties: false``;
the ``x_*`` namespace is the sanctioned escape hatch so adopters can attach
vendor fields without forking the standard. Extensions are allowed at the
document root, ``posture``, ``floor`` and its invariants
and ``corpus.documents[]`` (and ``digest``, whose content the validator
additionally pins to its recomputation) — and nowhere hash integrity or
mechanical resolvability depends on a closed shape (identity, citations,
agreement_type, taxonomy entries, compiler), nor in the verdict-free
precedent record (``evidence.clauses[]`` / ``evidence.precedent[]``, issue
#223: judged extras go under the root ``x_judgments``).

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

NDA_PLAYBOOK = Path(__file__).parent.parent / "examples" / "nda" / "playbook.opf.json"


def _minimal() -> dict[str, Any]:
    """The compiled NDA example, identity dropped so an added field trips only
    the rule under test."""
    doc: dict[str, Any] = json.loads(NDA_PLAYBOOK.read_text(encoding="utf-8"))
    del doc["identity"]
    return doc


def _assert_valid(doc: dict[str, Any], where: str) -> None:
    result = validate_document(doc)
    assert result.ok, f"x_* extension rejected at {where}: " + "; ".join(
        str(e) for e in result.errors
    )


def test_x_field_valid_at_each_level() -> None:
    """`x_vendor_note` must validate at each sanctioned level."""
    doc = _minimal()

    doc["x_vendor_note"] = "v"
    doc["posture"]["x_vendor_note"] = "v"
    doc["floor"]["x_vendor_note"] = "v"
    doc["floor"]["invariants"][0]["x_vendor_note"] = "v"
    doc["corpus"]["documents"][0]["x_vendor_note"] = "v"

    _assert_valid(doc, "the sanctioned levels")


def test_x_field_rejected_in_the_precedent_record() -> None:
    """Clause and precedent records are closed (issue #223)."""
    for container in ("clauses", "precedent"):
        doc = _minimal()
        doc["evidence"][container][0]["x_vendor_note"] = "v"
        assert not validate_document(doc).ok, container


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
    record = next(p for p in doc["evidence"]["precedent"] if p["signed_text"])
    record["signed_text"]["ref"]["x_foo"] = "v"
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
