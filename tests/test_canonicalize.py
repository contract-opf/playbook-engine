"""Tests for engine-native canonicalization + content hashing — issue #143.

Acceptance criteria verified here:

  - canonicalize() is stable across key-order and whitespace variation of a
    JSON-equivalent input.
  - content_hash() ignores the excluded fields: the `identity` block itself
    (self-referential) and `compiler.generated_at`/`compiler.run_id` (run
    metadata, not content).
  - content_hash() DOES change when actual content changes (evidence/posture/
    floor/corpus/taxonomy/etc.).
  - section_digest()/compute_section_digests() change only the digest of the
    section that actually changed — changing `evidence` must not perturb the
    `posture` or `floor` digest.
  - hash format matches `^sha256:[0-9a-f]{64}$` (the same pattern the schema
    already uses for `composes[].integrity`).

SECURITY NOTE: All fixtures are synthetic, minimal dicts — no real agreement
content.
"""

from __future__ import annotations

import copy
import re
from typing import Any

from playbook_engine.canonicalize import (
    canonicalize,
    canonicalize_playbook,
    compute_section_digests,
    content_hash,
    section_digest,
    sha256_hex,
)

_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


def _minimal_doc(**overrides: Any) -> dict[str, Any]:
    doc: dict[str, Any] = {
        "opf_version": "0.2",
        "agreement_type": {"id": "test-agreement", "name": "Test Agreement"},
        "baseline": {"has_canonical_template": False},
        "taxonomy": {"source": "custom", "entries": []},
        "evidence": {"clauses": [], "clause_library": []},
        "posture": {},
        "floor": {},
        "corpus": {"documents": [], "stats": {}},
        "compiler": {
            "name": "playbook-engine",
            "version": "0.1.0",
            "run_id": "run-abc",
            "generated_at": "2026-01-01T00:00:00Z",
        },
    }
    doc.update(overrides)
    return doc


# ---------------------------------------------------------------------------
# canonicalize() — stable serialization
# ---------------------------------------------------------------------------


def test_canonicalize_stable_across_key_order() -> None:
    a = {"b": 1, "a": 2}
    b = {"a": 2, "b": 1}
    assert canonicalize(a) == canonicalize(b)


def test_canonicalize_stable_across_nested_key_order() -> None:
    a = {"outer": {"z": 1, "y": {"n": 1, "m": 2}}}
    b = {"outer": {"y": {"m": 2, "n": 1}, "z": 1}}
    assert canonicalize(a) == canonicalize(b)


def test_canonicalize_no_insignificant_whitespace() -> None:
    out = canonicalize({"a": 1, "b": [1, 2]})
    assert " " not in out
    assert "\n" not in out


def test_canonicalize_does_not_reorder_arrays() -> None:
    """Array element order is semantic and must be preserved."""
    a = canonicalize({"items": ["z", "a", "m"]})
    b = canonicalize({"items": ["m", "a", "z"]})
    assert a != b


def test_sha256_hex_format() -> None:
    h = sha256_hex("hello")
    assert _HASH_RE.match(h)


def test_sha256_hex_deterministic() -> None:
    assert sha256_hex("same input") == sha256_hex("same input")


# ---------------------------------------------------------------------------
# content_hash() — excluded fields don't change the hash
# ---------------------------------------------------------------------------


def test_content_hash_format() -> None:
    assert _HASH_RE.match(content_hash(_minimal_doc()))


def test_content_hash_stable_across_key_order_and_whitespace() -> None:
    doc = _minimal_doc()
    # Round-tripping through a dict built in a different key order must not
    # change the canonical form or the hash.
    reordered = {k: doc[k] for k in reversed(list(doc.keys()))}
    assert content_hash(doc) == content_hash(reordered)


def test_content_hash_ignores_compiler_generated_at() -> None:
    doc_a = _minimal_doc()
    doc_b = copy.deepcopy(doc_a)
    doc_b["compiler"]["generated_at"] = "2099-12-31T23:59:59Z"
    assert content_hash(doc_a) == content_hash(doc_b)


def test_content_hash_ignores_compiler_run_id() -> None:
    doc_a = _minimal_doc()
    doc_b = copy.deepcopy(doc_a)
    doc_b["compiler"]["run_id"] = "a-completely-different-run-id"
    assert content_hash(doc_a) == content_hash(doc_b)


def test_content_hash_ignores_identity_block() -> None:
    """The identity block is self-referential (it will carry content_hash
    itself) and must not participate in the hash it defines."""
    doc_a = _minimal_doc()
    doc_b = copy.deepcopy(doc_a)
    doc_b["identity"] = {
        "id": "some-id",
        "version": "1.0.0",
        "content_hash": "sha256:" + "0" * 64,
        "section_digests": {
            "evidence": "sha256:" + "1" * 64,
            "posture": "sha256:" + "2" * 64,
            "floor": "sha256:" + "3" * 64,
        },
    }
    assert content_hash(doc_a) == content_hash(doc_b)


def test_content_hash_stable_when_identity_mutates() -> None:
    """Changing identity.content_hash/identity.id after the fact must not
    change the recomputed content_hash — it stays a pure function of content."""
    doc = _minimal_doc()
    doc["identity"] = {"id": "v1", "content_hash": "sha256:" + "a" * 64, "section_digests": {}}
    h1 = content_hash(doc)
    doc["identity"]["id"] = "v2"
    doc["identity"]["content_hash"] = "sha256:" + "b" * 64
    h2 = content_hash(doc)
    assert h1 == h2


# ---------------------------------------------------------------------------
# content_hash() — real content DOES change the hash
# ---------------------------------------------------------------------------


def test_content_hash_changes_when_evidence_changes() -> None:
    doc_a = _minimal_doc()
    doc_b = copy.deepcopy(doc_a)
    doc_b["evidence"]["clauses"].append({"id": "clause.new", "title": "New"})
    assert content_hash(doc_a) != content_hash(doc_b)


def test_content_hash_changes_when_posture_changes() -> None:
    doc_a = _minimal_doc()
    doc_b = copy.deepcopy(doc_a)
    doc_b["posture"] = {"system_prompt": "Be reasonable."}
    assert content_hash(doc_a) != content_hash(doc_b)


def test_content_hash_changes_when_floor_changes() -> None:
    doc_a = _minimal_doc()
    doc_b = copy.deepcopy(doc_a)
    doc_b["floor"] = {"invariants": [{"id": "no-x", "statement": "Never accept X."}]}
    assert content_hash(doc_a) != content_hash(doc_b)


def test_content_hash_changes_when_corpus_changes() -> None:
    doc_a = _minimal_doc()
    doc_b = copy.deepcopy(doc_a)
    doc_b["corpus"]["documents"].append(
        {"document_id": "d1", "provenance": "our_paper", "in_scope": True}
    )
    assert content_hash(doc_a) != content_hash(doc_b)


def test_content_hash_changes_when_compiler_name_changes() -> None:
    """Unlike generated_at/run_id, compiler.name/version ARE content-relevant
    provenance and must participate in the hash."""
    doc_a = _minimal_doc()
    doc_b = copy.deepcopy(doc_a)
    doc_b["compiler"]["name"] = "some-other-compiler"
    assert content_hash(doc_a) != content_hash(doc_b)


def test_canonicalize_playbook_excludes_identity_and_run_metadata() -> None:
    doc = _minimal_doc()
    doc["identity"] = {"id": "x", "content_hash": "sha256:" + "0" * 64, "section_digests": {}}
    canonical = canonicalize_playbook(doc)
    assert '"identity"' not in canonical
    assert '"generated_at"' not in canonical
    assert '"run_id"' not in canonical
    # compiler.name/version are content-relevant and must survive.
    assert '"playbook-engine"' in canonical


# ---------------------------------------------------------------------------
# section_digest() / compute_section_digests() — per-section isolation
# ---------------------------------------------------------------------------


def test_section_digest_format() -> None:
    assert _HASH_RE.match(section_digest({"clauses": []}))


def test_section_digest_deterministic() -> None:
    section = {"clauses": [{"id": "c1"}]}
    assert section_digest(section) == section_digest(copy.deepcopy(section))


def test_section_digest_stable_across_key_order() -> None:
    a = {"b": 1, "a": {"y": 1, "x": 2}}
    b = {"a": {"x": 2, "y": 1}, "b": 1}
    assert section_digest(a) == section_digest(b)


def test_compute_section_digests_returns_all_three() -> None:
    digests = compute_section_digests(_minimal_doc())
    # Issue #147: "curation" is a fourth digest, always computed (digests
    # `{}` when the document carries no curation key at all).
    assert set(digests.keys()) == {"evidence", "posture", "floor", "curation"}
    for h in digests.values():
        assert _HASH_RE.match(h)


def test_changing_evidence_only_changes_evidence_digest() -> None:
    doc_a = _minimal_doc()
    doc_b = copy.deepcopy(doc_a)
    doc_b["evidence"]["clauses"].append({"id": "clause.new", "title": "New"})

    digests_a = compute_section_digests(doc_a)
    digests_b = compute_section_digests(doc_b)

    assert digests_a["evidence"] != digests_b["evidence"]
    assert digests_a["posture"] == digests_b["posture"]
    assert digests_a["floor"] == digests_b["floor"]


def test_changing_posture_only_changes_posture_digest() -> None:
    doc_a = _minimal_doc()
    doc_b = copy.deepcopy(doc_a)
    doc_b["posture"] = {"system_prompt": "Default toward ACCEPT."}

    digests_a = compute_section_digests(doc_a)
    digests_b = compute_section_digests(doc_b)

    assert digests_a["posture"] != digests_b["posture"]
    assert digests_a["evidence"] == digests_b["evidence"]
    assert digests_a["floor"] == digests_b["floor"]


def test_changing_floor_only_changes_floor_digest() -> None:
    doc_a = _minimal_doc()
    doc_b = copy.deepcopy(doc_a)
    doc_b["floor"] = {"invariants": [{"id": "no-x", "statement": "Never accept X."}]}

    digests_a = compute_section_digests(doc_a)
    digests_b = compute_section_digests(doc_b)

    assert digests_a["floor"] != digests_b["floor"]
    assert digests_a["evidence"] == digests_b["evidence"]
    assert digests_a["posture"] == digests_b["posture"]


def test_missing_section_digests_as_empty_object() -> None:
    """A doc with no posture/floor content still gets a (deterministic) digest
    for the empty-but-present section — matches assemble_playbook()'s
    empty-but-present posture/floor convention."""
    doc = _minimal_doc()
    del doc["posture"]
    del doc["floor"]
    digests = compute_section_digests(doc)
    assert digests["posture"] == section_digest({})
    assert digests["floor"] == section_digest({})
