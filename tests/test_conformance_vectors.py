"""Conformance vector suite for canonicalize.py + digest.py — issue #115.

``spec/conformance/`` is the frozen, standalone-consumable (plain JSON, no
Python import required) normative definition of canonicalization, content
hashing, and digest construction for the format version stamped in
``spec/conformance/manifest.json``. This suite is the reference check: for
every vector, recompute ``canonicalize_playbook``/``content_hash``/
``compute_section_digests``/``build_digest`` from the vector's ``input``
using THIS engine and assert the result equals the vector's FROZEN
``expected.*`` values byte-for-byte.

The ``expected.*`` values are generated once by
``scripts/generate_conformance_vectors.py`` and committed — this test never
recomputes them at verification time from anywhere but the fixture file, so
a bug that changed canonicalize.py's/digest.py's output would actually be
caught here, not just asserted "self-consistent" against itself.
``test_mutated_*_is_detected`` below proves that directly (in-memory
tampering, never touching the fixture file on disk).

See ``spec/conformance/README.md`` for the full vector-by-vector rationale.

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
VECTORS_DIR = CONFORMANCE_DIR / "vectors"


def _load_manifest() -> dict[str, Any]:
    return json.loads((CONFORMANCE_DIR / "manifest.json").read_text(encoding="utf-8"))


def _load_vector(relative_file: str) -> dict[str, Any]:
    return json.loads((CONFORMANCE_DIR / relative_file).read_text(encoding="utf-8"))


def _vector_files() -> list[str]:
    manifest = _load_manifest()
    files = [entry["file"] for entry in manifest["vectors"]]
    assert files, "manifest.json lists no vectors"
    return files


def _by_name(name: str) -> dict[str, Any]:
    for f in _vector_files():
        vector = _load_vector(f)
        if vector["name"] == name:
            return vector
    raise AssertionError(f"no conformance vector named {name!r}")


# ---------------------------------------------------------------------------
# Structural sanity — the manifest/vector files themselves are well-formed.
# ---------------------------------------------------------------------------


def test_manifest_lists_every_vector_file_on_disk() -> None:
    manifest_files = {entry["file"] for entry in _load_manifest()["vectors"]}
    on_disk = {f"vectors/{p.name}" for p in VECTORS_DIR.glob("*.json")}
    assert manifest_files == on_disk


def test_manifest_declares_format_version() -> None:
    fv = _load_manifest()["format_version"]
    assert fv["opf_version"]
    assert fv["engine_version"]
    assert fv["digest_version"]


# ---------------------------------------------------------------------------
# The reference check: every vector reproduces exactly from its `input`.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("filename", _vector_files())
def test_vector_reproduces_exactly(filename: str) -> None:
    vector = _load_vector(filename)
    doc = vector["input"]
    expected = vector["expected"]

    assert canonicalize_playbook(doc) == expected["canonical"], filename
    assert content_hash(doc) == expected["content_hash"], filename
    assert compute_section_digests(doc) == expected["section_digests"], filename
    assert build_digest(doc) == expected["digest"], filename


# ---------------------------------------------------------------------------
# Targeted pairwise checks — the reviewer-gate edge cases, asserted as
# RELATIONSHIPS between vectors, not just each vector's own self-check.
# ---------------------------------------------------------------------------


def test_key_ordering_pair_hashes_identically() -> None:
    """001 vs. 002: identical content, every object's keys inserted in
    reverse order (top level and nested) — canonicalization must be blind
    to input key order."""
    a = _by_name("minimal-ascii")
    b = _by_name("minimal-ascii-reordered-keys")
    assert list(a["input"].keys()) != list(b["input"].keys())
    assert a["expected"]["canonical"] == b["expected"]["canonical"]
    assert a["expected"]["content_hash"] == b["expected"]["content_hash"]


def test_array_order_pair_hashes_differently() -> None:
    """003 vs. 004: same two clause objects, evidence.clauses array
    reversed — array element order is semantic and must NOT be sorted."""
    a = _by_name("two-clauses-order-a")
    b = _by_name("two-clauses-order-b")
    assert a["expected"]["canonical"] != b["expected"]["canonical"]
    assert a["expected"]["content_hash"] != b["expected"]["content_hash"]


def test_unicode_is_emitted_literally_not_escaped() -> None:
    v = _by_name("unicode-literal-utf8")
    assert "\\u" not in v["expected"]["canonical"]
    assert "🔒" in v["expected"]["canonical"]
    assert "合同" in v["expected"]["canonical"]


def test_unicode_normalization_form_pair_hashes_differently() -> None:
    """006 vs. 007: visually-identical 'café' spelled NFC (precomposed) vs.
    NFD (decomposed) — the engine must NOT normalize Unicode before
    hashing, so these differ."""
    nfc = _by_name("unicode-nfc-form")
    nfd = _by_name("unicode-nfd-form")
    assert nfc["expected"]["canonical"] != nfd["expected"]["canonical"]
    assert nfc["expected"]["content_hash"] != nfd["expected"]["content_hash"]


def test_float_int_formatting_pins_exact_renderings() -> None:
    v = _by_name("float-int-formatting")
    canonical = v["expected"]["canonical"]
    # Whole-number float MUST keep its trailing .0 — a serializer that
    # collapses it to a bare integer (e.g. JavaScript's JSON.stringify)
    # would produce a different byte sequence and a different content_hash.
    assert '"whole_number_float":1.0' in canonical
    assert '"large_int":1000000000' in canonical
    assert '"zero_float":0.0' in canonical
    assert '"negative_float":-0.5' in canonical


def test_empty_vs_absent_floor_hashes_differently_but_section_digest_equal() -> None:
    """009 vs. 010: `floor: {}` present vs. the `floor` key entirely
    absent — content_hash sees the literal document shape (differs), but
    section_digests.floor defaults a missing section to `{}` (equal)."""
    present = _by_name("floor-present-empty")
    absent = _by_name("floor-absent")
    assert "floor" in present["input"]
    assert "floor" not in absent["input"]
    assert present["expected"]["content_hash"] != absent["expected"]["content_hash"]
    assert (
        present["expected"]["section_digests"]["floor"]
        == absent["expected"]["section_digests"]["floor"]
    )


def test_excluded_run_and_curation_metadata_do_not_perturb_content_hash() -> None:
    """011 vs. 012: unrecognizably different identity/curation/compiler run
    metadata must not change canonical bytes or content_hash — but
    section_digests.curation, which tracks curation lineage independently
    (OPF-SPEC.md §3.11), DOES still differ."""
    a = _by_name("excluded-metadata-variant-a")
    b = _by_name("excluded-metadata-variant-b")
    assert a["input"]["identity"] != b["input"]["identity"]
    assert a["input"]["curation"] != b["input"]["curation"]
    assert a["input"]["compiler"]["run_id"] != b["input"]["compiler"]["run_id"]

    assert a["expected"]["canonical"] == b["expected"]["canonical"]
    assert a["expected"]["content_hash"] == b["expected"]["content_hash"]

    digests_a = a["expected"]["section_digests"]
    digests_b = b["expected"]["section_digests"]
    assert digests_a["evidence"] == digests_b["evidence"]
    assert digests_a["posture"] == digests_b["posture"]
    assert digests_a["floor"] == digests_b["floor"]
    assert digests_a["curation"] != digests_b["curation"]


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


@pytest.mark.parametrize("filename", _vector_files())
def test_mutated_content_hash_is_detected(filename: str) -> None:
    vector = _load_vector(filename)
    doc = vector["input"]
    tampered_expected = _flip_last_hex_char(vector["expected"]["content_hash"])

    recomputed = content_hash(doc)

    # The whole point: comparing against a WRONG frozen value must fail.
    assert recomputed != tampered_expected, (
        f"{filename}: a tampered expected content_hash was not detected — "
        "the comparison in this suite would not catch real drift either"
    )
    # And the untampered value it was derived from still matches — proves
    # the mutation, not some unrelated bug, is what changed the outcome.
    assert recomputed == vector["expected"]["content_hash"]


def test_mutated_canonical_bytes_are_detected() -> None:
    vector = _by_name("minimal-ascii")
    doc = vector["input"]
    tampered_expected = vector["expected"]["canonical"].replace(
        '"opf_version":"0.3"', '"opf_version":"9.9"'
    )
    assert tampered_expected != vector["expected"]["canonical"]
    assert canonicalize_playbook(doc) != tampered_expected
    assert canonicalize_playbook(doc) == vector["expected"]["canonical"]


def test_mutated_digest_is_detected() -> None:
    vector = _by_name("two-clauses-order-a")
    doc = vector["input"]
    tampered_expected = copy.deepcopy(vector["expected"]["digest"])
    tampered_expected["clause_count"] = tampered_expected["clause_count"] + 1

    recomputed = build_digest(doc)
    assert recomputed != tampered_expected
    assert recomputed == vector["expected"]["digest"]
