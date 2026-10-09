"""Conformance vector suite for canonicalize.py + digest.py — issue #115.

``spec/conformance/0.5/`` is the frozen, standalone-consumable (plain JSON,
no Python import required) normative definition of canonicalization,
content hashing, and digest construction for OPF 0.5 / digest_version 3 —
the one format the engine reads and writes (issue #238; the OPF 0.3 /
digest 2 set was retired with that format). This suite is the reference
check: for every vector, recompute ``canonicalize_playbook``/
``content_hash``/``compute_section_digests``/``build_digest`` from the
vector's ``input`` using THIS engine and assert the result equals the
vector's FROZEN ``expected.*`` values byte-for-byte.

The ``expected.*`` values are generated once by
``scripts/generate_conformance_vectors.py`` and committed — this test never
recomputes them at verification time from anywhere but the fixture file, so
a bug that changed canonicalize.py's/digest.py's output would actually be
caught here, not just asserted "self-consistent" against itself.
The ``test_mutated_*`` tests below prove that directly (in-memory
tampering, never touching the fixture file on disk).

See ``spec/conformance/0.5/README.md`` for the vector-by-vector rationale.

SECURITY NOTE: every vector's ``input`` is a synthetic, hand-built minimal
document — no real agreement content.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from playbook_engine.canonicalize import (
    canonicalize_playbook,
    compute_section_digests,
    content_hash,
)
from playbook_engine.digest import build_digest

ROOT = Path(__file__).parent.parent
CONFORMANCE_DIR = ROOT / "spec" / "conformance"
#: The OPF 0.5 / digest_version 3 set (issue #223).
CONFORMANCE_DIR_V05 = CONFORMANCE_DIR / "0.5"


def test_retired_0_3_vector_set_is_gone() -> None:
    """The OPF 0.3 / digest 2 set was retired with that format (issue #238)."""
    assert not (CONFORMANCE_DIR / "manifest.json").exists()
    assert not (CONFORMANCE_DIR / "vectors").exists()


# ---------------------------------------------------------------------------
# OPF 0.5 / digest_version 3 set (issue #223) — spec/conformance/0.5/
# ---------------------------------------------------------------------------


def _load_manifest_v05() -> dict[str, Any]:
    return json.loads((CONFORMANCE_DIR_V05 / "manifest.json").read_text(encoding="utf-8"))


def _vector_files_v05() -> list[str]:
    files = [entry["file"] for entry in _load_manifest_v05()["vectors"]]
    assert files, "0.5/manifest.json lists no vectors"
    return files


def _load_vector_v05(relative_file: str) -> dict[str, Any]:
    return json.loads((CONFORMANCE_DIR_V05 / relative_file).read_text(encoding="utf-8"))


def test_v05_manifest_lists_every_vector_file_on_disk() -> None:
    manifest_files = {entry["file"] for entry in _load_manifest_v05()["vectors"]}
    on_disk = {f"vectors/{p.name}" for p in (CONFORMANCE_DIR_V05 / "vectors").glob("*.json")}
    assert manifest_files == on_disk


def test_v05_manifest_is_stamped_0_5_digest_3() -> None:
    fv = _load_manifest_v05()["format_version"]
    assert fv["opf_version"] == "0.5"
    assert fv["digest_version"] == "3"
    assert fv["engine_version"]


@pytest.mark.parametrize("filename", _vector_files_v05())
def test_v05_vector_reproduces_exactly(filename: str) -> None:
    vector = _load_vector_v05(filename)
    doc = vector["input"]
    expected = vector["expected"]
    assert vector["opf_version"] == "0.5" and doc["opf_version"] == "0.5"
    assert canonicalize_playbook(doc) == expected["canonical"], filename
    assert content_hash(doc) == expected["content_hash"], filename
    assert compute_section_digests(doc) == expected["section_digests"], filename
    assert build_digest(doc) == expected["digest"], filename
    assert expected["digest"]["digest_version"] == "3"
    assert "full_text" not in json.dumps(expected["digest"])


@pytest.mark.parametrize("filename", _vector_files_v05())
def test_v05_vector_inputs_are_valid_0_5_documents(filename: str) -> None:
    """Every 0.5 input is a self-consistent document the validator accepts —
    precedent ids, clause counts, and (once embedded) the digest all agree."""
    from playbook_engine.validator import validate_document

    vector = _load_vector_v05(filename)
    doc = copy.deepcopy(vector["input"])
    doc["digest"] = vector["expected"]["digest"]
    result = validate_document(doc)
    assert result.ok, [str(e) for e in result.errors if e.blocking]


def test_v05_mutated_digest_is_detected() -> None:
    """A tampered expected digest (one deal count off) must not match."""
    vector = _load_vector_v05("vectors/002-variants-refused-and-exclusions.json")
    tampered = copy.deepcopy(vector["expected"]["digest"])
    tampered["clauses"][0]["signed_variants"][0]["n_deals"] += 1
    recomputed = build_digest(vector["input"])
    assert recomputed != tampered
    assert recomputed == vector["expected"]["digest"]


def test_v05_negator_is_never_merged_into_a_variant() -> None:
    """002 pins exact normalization: 'may not assign' is its own group, and the
    case/punctuation respelling of variant A is merged into A (3 deals)."""
    digest = _load_vector_v05("vectors/002-variants-refused-and-exclusions.json")["expected"][
        "digest"
    ]
    variants = digest["clauses"][0]["signed_variants"]
    assert [v["n_deals"] for v in variants] == [3, 1]
    assert " not " in variants[1]["text"]
    assert digest["clauses"][0]["refused_asks"][0]["n_deals"] == 3


def test_v05_counterparty_alias_never_splits_a_variant() -> None:
    """004 pins the grouping key's party neutralization (OPF-SPEC §3.5.4):
    texts differing only by the counterparty's Counterparty-<n> alias are one
    group with n_deals 2 (signed variants and refused asks alike), the
    parties' places swapped stays its own group, and the one-deal variants'
    order shows perspective.party was rewritten to 'party'."""
    vector = _load_vector_v05("vectors/004-party-alias-grouping.json")
    entry = vector["expected"]["digest"]["clauses"][0]
    assert [(v["n_deals"], v["ref"]["document_id"]) for v in entry["signed_variants"]] == [
        (2, "deal-a"),
        (1, "deal-d"),
        (1, "deal-c"),
    ]
    assert [a["n_deals"] for a in entry["refused_asks"]] == [2]
    clause = vector["input"]["evidence"]["clauses"][0]
    assert (clause["n_variants"], clause["n_refused"]) == (3, 1)


# ---------------------------------------------------------------------------
# Proof the suite has teeth (issue #115 reviewer gate): a tampered expected
# value must be DETECTED, not silently pass. These mutate only an in-memory
# copy — never the fixture file on disk.
# ---------------------------------------------------------------------------


def _flip_last_hex_char(sha: str) -> str:
    prefix, hexdigest = sha.split(":", 1)
    last = hexdigest[-1]
    flipped = "0" if last != "0" else "1"
    return f"{prefix}:{hexdigest[:-1]}{flipped}"


@pytest.mark.parametrize("filename", _vector_files_v05())
def test_mutated_content_hash_is_detected(filename: str) -> None:
    vector = _load_vector_v05(filename)
    tampered_expected = _flip_last_hex_char(vector["expected"]["content_hash"])
    recomputed = content_hash(vector["input"])
    assert recomputed != tampered_expected
    assert recomputed == vector["expected"]["content_hash"]


def test_mutated_canonical_bytes_are_detected() -> None:
    vector = _load_vector_v05("vectors/001-minimal-no-perspective.json")
    tampered_expected = vector["expected"]["canonical"].replace(
        '"opf_version":"0.5"', '"opf_version":"9.9"'
    )
    assert tampered_expected != vector["expected"]["canonical"]
    assert canonicalize_playbook(vector["input"]) != tampered_expected
    assert canonicalize_playbook(vector["input"]) == vector["expected"]["canonical"]
