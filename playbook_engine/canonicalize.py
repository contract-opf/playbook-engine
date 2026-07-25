"""Canonical serialization + content hashing for OPF playbooks — issue #143.

Gives an OPF v0.2 document artifact identity: a deterministic canonical
serialization, a whole-document ``content_hash``, and per-section digests
(evidence/posture/floor) — so a consumer can record which exact playbook
governed which review, and lineage across compiles is reconstructible
(OPF-SPEC.md §8).

Canonical form (normative definition)
--------------------------------------
The canonical form of any JSON-serializable value is its ``json.dumps``
output with:

  - keys sorted recursively (``sort_keys=True``);
  - no insignificant whitespace (``separators=(",", ":")``);
  - UTF-8-safe non-ASCII emitted literally, not ``\\uXXXX``-escaped
    (``ensure_ascii=False``) — the hash is taken over the UTF-8 encoding of
    this string, so this only affects the human-readable form, not the hash.

Array element order is NOT touched — order is semantic (e.g.
``observed_positions`` order, taxonomy entry order) and reordering would
silently change meaning.

Whole-document ``content_hash`` excludes three things so it isn't
self-referential and isn't perturbed by non-content run/curation metadata:

  - the top-level ``identity`` object itself — it is where ``content_hash``
    (and the section digests) are written, so hashing it would make the hash
    depend on its own prior value;
  - ``compiler.generated_at`` / ``compiler.run_id`` — wall-clock timestamp and
    run identifier. Two compiles of byte-identical corpus content run a
    second apart (or with a different ``run_id``) must hash identically;
    only the compiler's *name*/*version* are content-relevant provenance.
  - the top-level ``curation`` object (issue #147) — attorney-pinned
    positions embedded in the OPF. A pin surviving a recompile unchanged, or
    a conflict flag being raised/cleared on a pin, is not itself a change to
    the corpus-derived content (evidence/posture/floor), so it must not
    perturb ``content_hash``. ``curation`` gets its own digest instead (see
    ``compute_section_digests`` below) so a consumer can still track its
    lineage independently.

A section digest (``evidence``/``posture``/``floor``/``curation``) is the
hash of that section's own canonical bytes in isolation — it does not
exclude anything, since a section has no self-referential or run-metadata
fields of its own. This lets a consumer pin/verify a single section without
needing the whole document (OPF-SPEC.md §7's ``grounded_in:
"evidence@<digest>"``).

Hash format: ``"sha256:" + hexdigest``, matching the pattern already used by
``composes[].integrity`` in ``spec/playbook.schema-0.2.json``
(``^sha256:[0-9a-f]{64}$``).
"""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any

# Top-level keys excluded from the whole-document canonical form. "identity"
# is self-referential: it carries content_hash/section_digests, the very
# values being computed. "curation" (issue #147) is the attorney-pin overlay
# — see module docstring for why it must not perturb content_hash.
_EXCLUDED_TOP_LEVEL_KEYS = frozenset({"identity", "curation"})

# `compiler` sub-keys excluded from the whole-document canonical form: run
# metadata, not playbook content. See module docstring.
_EXCLUDED_COMPILER_KEYS = frozenset({"generated_at", "run_id"})

_SECTION_NAMES = ("evidence", "posture", "floor", "curation")


def canonicalize(value: Any) -> str:
    """Return the canonical JSON string for *value*.

    Recursively sorted object keys, no insignificant whitespace, UTF-8-safe.
    Does not reorder arrays — element order is semantic.
    """
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_hex(canonical_str: str) -> str:
    """Return ``"sha256:" + hex digest`` over the UTF-8 bytes of *canonical_str*."""
    digest = hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def file_sha256(path: Any) -> str:
    """Return the ``"sha256:"``-prefixed content address of a file's bytes.

    The format-critical helper behind OPF §4 content addressing (issue
    #185): version_files, baseline.template_ref.sha256, and the citation
    resolver must agree byte-for-byte on this value, so they all call this
    one function. Streams in 64 KiB chunks — corpus files are real-world
    PDFs/DOCX and must not be slurped whole into memory.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return f"sha256:{h.hexdigest()}"


def canonicalize_playbook(playbook: dict[str, Any]) -> str:
    """Return the whole-document canonical form used for ``content_hash``.

    Strips the excluded top-level ``identity``/``curation`` keys and the
    excluded ``compiler`` sub-keys (``generated_at``, ``run_id``) from a deep
    copy of *playbook* before serializing — see module docstring for why.
    """
    doc = copy.deepcopy(playbook)
    for key in _EXCLUDED_TOP_LEVEL_KEYS:
        doc.pop(key, None)
    compiler = doc.get("compiler")
    if isinstance(compiler, dict):
        for key in _EXCLUDED_COMPILER_KEYS:
            compiler.pop(key, None)
    return canonicalize(doc)


def content_hash(playbook: dict[str, Any]) -> str:
    """Return the playbook's ``content_hash``.

    ``sha256:`` + the hex digest of ``canonicalize_playbook(playbook)``.
    Stable across key-order/whitespace variation of the input, across
    ``compiler.generated_at``/``run_id`` changes, and across any prior value
    of ``identity`` — changes when any actual content (evidence, posture,
    floor, corpus, taxonomy, etc.) changes.
    """
    return sha256_hex(canonicalize_playbook(playbook))


def section_digest(section: Any) -> str:
    """Return the digest of a single OPF section (evidence/posture/floor).

    Computed over the section's own canonical bytes in isolation — not the
    whole document — so a consumer can pin/verify one section's lineage
    independent of the rest of the playbook (OPF-SPEC.md §8).
    """
    return sha256_hex(canonicalize(section))


def compute_section_digests(playbook: dict[str, Any]) -> dict[str, str]:
    """Return ``{"evidence": ..., "posture": ..., "floor": ..., "curation": ...}``.

    Each value is ``section_digest(playbook.get(name, {}))`` for
    ``name in ("evidence", "posture", "floor", "curation")``. ``curation``
    digests ``{}`` (a stable, well-defined value) when the playbook carries
    no ``curation`` key at all — e.g. a corpus-only compile with no pins yet.
    """
    return {name: section_digest(playbook.get(name, {})) for name in _SECTION_NAMES}
