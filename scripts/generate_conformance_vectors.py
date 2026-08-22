#!/usr/bin/env python3
"""Regenerate the frozen conformance vectors under ``spec/conformance/`` — issue #115.

DEV TOOL, NOT PART OF THE TEST SUITE OR THE RUNTIME PACKAGE. Run this only
when deliberately re-stamping the conformance vectors for a new format
version (a new ``opf_version`` or a new ``DIGEST_VERSION``) — never as a
routine "regenerate the golden files" step. The whole point of
``spec/conformance/`` is that its ``expected.*`` values are FROZEN,
independently-computed-once numbers that ``tests/test_conformance_vectors.py``
checks the live engine against; overwriting them from a possibly-buggy
current engine on every run would turn the conformance suite into a tautology
that can never go red (exactly what the issue #115 reviewer gate calls
"self-consistency" and requires the suite NOT be).

Usage::

    .venv/bin/python scripts/generate_conformance_vectors.py

Writes ``spec/conformance/manifest.json`` and one file per vector under
``spec/conformance/vectors/``. Review the resulting diff like any other
spec change (it needs a ``spec/CHANGELOG.md`` entry) before committing.
"""

from __future__ import annotations

import copy
import json
import unicodedata
from pathlib import Path
from typing import Any

from playbook_engine import __version__ as ENGINE_VERSION
from playbook_engine.canonicalize import (
    canonicalize_playbook,
    compute_section_digests,
    content_hash,
)
from playbook_engine.digest import DIGEST_VERSION, build_digest

ROOT = Path(__file__).parent.parent
CONFORMANCE_DIR = ROOT / "spec" / "conformance"
VECTORS_DIR = CONFORMANCE_DIR / "vectors"

OPF_VERSION = "0.3"


def _base(*, clauses: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """A minimal OPF {OPF_VERSION} document — same skeleton as
    tests/test_canonicalize.py::_minimal_doc, kept in sync deliberately."""
    return {
        "opf_version": OPF_VERSION,
        "agreement_type": {"id": "conformance-fixture", "name": "Conformance Fixture Agreement"},
        "baseline": {"has_canonical_template": False},
        "taxonomy": {"source": "custom", "entries": []},
        "evidence": {"clauses": clauses or [], "clause_library": []},
        "posture": {},
        "floor": {},
        "corpus": {"documents": [], "stats": {}},
        "compiler": {
            "name": "playbook-engine",
            "version": ENGINE_VERSION,
            "run_id": "conformance-fixture-run",
            "generated_at": "2026-01-01T00:00:00Z",
        },
    }


def _clause(
    clause_id: str,
    title: str,
    *,
    document_id: str = "doc-1",
    char_span: list[int] | None = None,
    precedent_count: int = 3,
) -> dict[str, Any]:
    return {
        "id": clause_id,
        "taxonomy_id": clause_id.rsplit(".", 1)[-1],
        "title": title,
        "observed_positions": [
            {
                "text_summary": f"{title} — standard form.",
                "full_text": f"{title} — standard form, full text.",
                "example_ref": {
                    "document_id": document_id,
                    "version": 1,
                    "clause_path": "1",
                    "char_span": char_span or [0, 20],
                },
                "deviation": "none",
                "risk_delta": {"direction": "neutral", "magnitude": "none"},
                "provenance": "our_paper",
                "outcome": "signed",
                "precedent_count": precedent_count,
            }
        ],
        "summary": {
            "historical_stance": "usually_held",
            "acceptable_if": [],
            "fallbacks": [],
            "rejected": [],
            "confidence": {
                "score": 0.5,
                "basis": "precedent_count+provenance_mix",
                "n_our_paper": precedent_count,
                "n_counterparty_paper": 0,
            },
        },
    }


def _reverse_keys(d: dict[str, Any]) -> dict[str, Any]:
    return {k: d[k] for k in reversed(list(d.keys()))}


def _expected(doc: dict[str, Any]) -> dict[str, Any]:
    return {
        "canonical": canonicalize_playbook(doc),
        "content_hash": content_hash(doc),
        "section_digests": compute_section_digests(doc),
        "digest": build_digest(doc),
    }


def _vector(name: str, description: str, doc: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": name,
        "description": description,
        "opf_version": OPF_VERSION,
        "engine_version": ENGINE_VERSION,
        "digest_version": DIGEST_VERSION,
        "input": doc,
        "expected": _expected(doc),
    }


def build_vectors() -> list[tuple[str, dict[str, Any]]]:
    vectors: list[tuple[str, dict[str, Any]]] = []

    # 1/2 — key ordering: semantically identical documents, keys inserted in
    # a different order at the top level AND inside a nested object, must
    # produce byte-identical canonical form and content_hash.
    doc_001 = _base()
    vectors.append(
        (
            "001-minimal-ascii",
            _vector(
                "minimal-ascii",
                "Smallest well-formed document, ASCII only, no evidence clauses. "
                "Baseline for the key-ordering pair (002) and the empty/absent "
                "pair (009/010).",
                doc_001,
            ),
        )
    )

    doc_002 = _reverse_keys(doc_001)
    doc_002["compiler"] = _reverse_keys(doc_002["compiler"])
    doc_002["taxonomy"] = _reverse_keys(doc_002["taxonomy"])
    doc_002["agreement_type"] = _reverse_keys(doc_002["agreement_type"])
    # Python dicts compare equal regardless of insertion order, so the
    # meaningful check is the raw key ORDER differing, not dict equality.
    assert list(doc_002.keys()) != list(doc_001.keys())
    assert list(doc_002["compiler"].keys()) != list(doc_001["compiler"].keys())
    assert canonicalize_playbook(doc_002) == canonicalize_playbook(doc_001)  # same canonical form
    vectors.append(
        (
            "002-minimal-ascii-reordered-keys",
            _vector(
                "minimal-ascii-reordered-keys",
                "Same content as 001-minimal-ascii with every object's keys inserted "
                "in reverse order (top level, and the nested compiler/taxonomy/"
                "agreement_type objects). expected.canonical and "
                "expected.content_hash MUST equal 001's byte-for-byte — proves "
                "recursive key-sort independence.",
                doc_002,
            ),
        )
    )

    # 3/4 — nested arrays: element order within evidence.clauses (and the
    # nested char_span pairs inside each observation) is semantic and must
    # be preserved, never sorted.
    clause_alpha = _clause("clause.alpha", "Alpha Clause", char_span=[0, 21])
    clause_beta = _clause("clause.beta", "Beta Clause", document_id="doc-2", char_span=[5, 40])

    doc_003 = _base(clauses=[copy.deepcopy(clause_alpha), copy.deepcopy(clause_beta)])
    vectors.append(
        (
            "003-two-clauses-order-a",
            _vector(
                "two-clauses-order-a",
                "Two clauses [alpha, beta], each carrying a nested char_span array "
                "several levels deep (evidence.clauses[].observed_positions[]."
                "example_ref.char_span). Pairs with 004 (reversed order) to prove "
                "array element order is preserved, not sorted.",
                doc_003,
            ),
        )
    )

    doc_004 = _base(clauses=[copy.deepcopy(clause_beta), copy.deepcopy(clause_alpha)])
    assert canonicalize_playbook(doc_004) != canonicalize_playbook(doc_003)
    vectors.append(
        (
            "004-two-clauses-order-b",
            _vector(
                "two-clauses-order-b",
                "Same two clause objects as 003-two-clauses-order-a with the "
                "evidence.clauses array reversed to [beta, alpha]. "
                "expected.canonical and expected.content_hash MUST differ from "
                "003's.",
                doc_004,
            ),
        )
    )

    # 5 — unicode is emitted literally (UTF-8), never \uXXXX-escaped.
    unicode_title = "Café Non-Disclosure — “Confidentiality” 🔒 合同"
    clause_unicode = _clause("clause.unicode_literal", unicode_title)
    doc_005 = _base(clauses=[clause_unicode])
    canonical_005 = canonicalize_playbook(doc_005)
    assert unicode_title in canonical_005
    assert "\\u" not in canonical_005
    vectors.append(
        (
            "005-unicode-literal-utf8",
            _vector(
                "unicode-literal-utf8",
                "A clause title containing accented Latin, curly quotes, an "
                "em dash, an emoji, and CJK characters. expected.canonical MUST "
                "contain these code points literally (UTF-8) — a \\uXXXX-escaping "
                "implementation produces a DIFFERENT byte sequence and therefore "
                "a different content_hash than this vector's.",
                doc_005,
            ),
        )
    )

    # 6/7 — unicode normalization is NOT applied. Two visually-identical
    # spellings of "café" that differ in code points (NFC precomposed é vs.
    # NFD e + combining acute) must hash DIFFERENTLY — an implementation
    # that normalizes Unicode before hashing (e.g. a JS String.normalize()
    # step) will silently produce the wrong hash relative to this reference.
    nfc_title = f"NFC {unicodedata.normalize('NFC', 'café')}"
    nfd_title = f"NFC {unicodedata.normalize('NFD', 'café')}"  # label kept identical on purpose
    assert nfc_title != nfd_title
    assert nfc_title.encode("utf-8") != nfd_title.encode("utf-8")

    doc_006 = _base(clauses=[_clause("clause.unicode_nfc", nfc_title)])
    vectors.append(
        (
            "006-unicode-nfc-form",
            _vector(
                "unicode-nfc-form",
                "Clause title using the NFC (precomposed) spelling of an accented "
                "character. Pairs with 007 (NFD/decomposed spelling, visually "
                "identical) to prove the engine does NOT apply Unicode "
                "normalization before hashing.",
                doc_006,
            ),
        )
    )

    doc_007 = _base(clauses=[_clause("clause.unicode_nfc", nfd_title)])
    assert content_hash(doc_007) != content_hash(doc_006)
    vectors.append(
        (
            "007-unicode-nfd-form",
            _vector(
                "unicode-nfd-form",
                "Same clause id/structure as 006-unicode-nfc-form with the title's "
                "accented character spelled in NFD (decomposed) form instead — "
                "same rendered glyphs, different code points. "
                "expected.content_hash MUST differ from 006's.",
                doc_007,
            ),
        )
    )

    # 8 — float/int formatting: whole-number floats keep their trailing
    # ".0" (Python json.dumps(1.0) == "1.0", NOT "1" — a JSON serializer
    # that collapses whole-number floats to integers, as JavaScript's
    # JSON.stringify does, produces a different byte sequence here).
    doc_008 = _base(clauses=[_clause("clause.numeric", "Numeric Edge Cases")])
    doc_008["x_numeric_probe"] = {
        "whole_number_float": 1.0,
        "float_precision": 0.1 + 0.2,
        "negative_float": -0.5,
        "zero_float": 0.0,
        "large_int": 1_000_000_000,
        "small_exponent_float": 1e-10,
    }
    canonical_008 = canonicalize_playbook(doc_008)
    assert '"whole_number_float":1.0' in canonical_008
    assert '"large_int":1000000000' in canonical_008
    vectors.append(
        (
            "008-float-int-formatting",
            _vector(
                "float-int-formatting",
                "A synthetic x_numeric_probe object (schema-legal vendor extension "
                "at the document root, §10.1) isolating float/int formatting "
                "gotchas: a whole-number float that MUST keep its '.0', a "
                "floating-point-precision value (0.1 + 0.2), a negative float, "
                "0.0, a large integer, and a small-magnitude float that Python "
                "renders in exponential notation. expected.canonical is the "
                "byte-for-byte pin for each.",
                doc_008,
            ),
        )
    )

    # 9/10 — empty vs. absent: a present-but-empty `floor: {}` and an
    # entirely absent `floor` key must NOT hash the same (content_hash sees
    # the literal document shape) even though they resolve to the SAME
    # section_digest (compute_section_digests defaults a missing section to
    # `{}` — see canonicalize.py::compute_section_digests).
    doc_009 = _base()
    doc_009["floor"] = {}
    vectors.append(
        (
            "009-floor-present-empty",
            _vector(
                "floor-present-empty",
                "Document with a top-level floor key explicitly present and "
                "empty ({}). Pairs with 010 (floor key entirely absent) to prove "
                "content_hash distinguishes 'empty' from 'absent' even though "
                "section_digests.floor is identical between the two (both "
                "resolve to section_digest({})).",
                doc_009,
            ),
        )
    )

    doc_010 = _base()
    del doc_010["floor"]
    assert content_hash(doc_010) != content_hash(doc_009)
    assert compute_section_digests(doc_010)["floor"] == compute_section_digests(doc_009)["floor"]
    vectors.append(
        (
            "010-floor-absent",
            _vector(
                "floor-absent",
                "Same document as 009-floor-present-empty with the top-level "
                "floor key removed entirely (not merely emptied). "
                "expected.content_hash MUST differ from 009's; "
                "expected.section_digests.floor MUST equal 009's.",
                doc_010,
            ),
        )
    )

    # 11/12 — excluded run/curation metadata: identity, curation, and
    # compiler.generated_at/run_id must NOT perturb content_hash (or the
    # canonical bytes content_hash is taken over) even when wildly
    # different between two otherwise-identical documents. curation DOES
    # still get its own section_digest, which — unlike content_hash — DOES
    # change, since a consumer needs to be able to track curation lineage
    # independently (§3.11).
    shared_clause = _clause("clause.alpha", "Alpha Clause", char_span=[0, 21])

    doc_011 = _base(clauses=[copy.deepcopy(shared_clause)])
    doc_011["identity"] = {
        "id": "playbook-v1",
        "version": "1.0.0",
        "content_hash": "sha256:" + "0" * 64,
        "section_digests": {
            "evidence": "sha256:" + "1" * 64,
            "posture": "sha256:" + "2" * 64,
            "floor": "sha256:" + "3" * 64,
            "curation": "sha256:" + "4" * 64,
        },
    }
    doc_011["curation"] = {
        "pins": [
            {
                "clause_id": "clause.alpha",
                "item_id": "C1",
                "position": "consistently_held",
                "baseline_stance": "usually_held",
                "pinned_at": "2026-01-01T00:00:00Z",
            }
        ]
    }
    doc_011["compiler"]["generated_at"] = "2026-01-01T00:00:00Z"
    doc_011["compiler"]["run_id"] = "run-a"
    vectors.append(
        (
            "011-excluded-metadata-variant-a",
            _vector(
                "excluded-metadata-variant-a",
                "A document carrying identity, a curation pin, and "
                "compiler.generated_at/run_id. Pairs with 012 (same evidence/"
                "posture/floor, unrecognizably different identity/curation/"
                "run metadata) to prove content_hash and canonical bytes are "
                "identical across the pair, while section_digests.curation "
                "differs.",
                doc_011,
            ),
        )
    )

    doc_012 = _base(clauses=[copy.deepcopy(shared_clause)])
    doc_012["identity"] = {
        "id": "a-totally-different-id",
        "version": "9.9.9-does-not-exist",
        "content_hash": "sha256:" + "f" * 64,
        "section_digests": {
            "evidence": "sha256:" + "a" * 64,
            "posture": "sha256:" + "b" * 64,
            "floor": "sha256:" + "c" * 64,
            "curation": "sha256:" + "d" * 64,
        },
    }
    doc_012["curation"] = {
        "pins": [
            {
                "clause_id": "clause.alpha",
                "item_id": "C99",
                "position": "an entirely different asserted position",
                "baseline_stance": "mixed",
                "pinned_at": "2099-12-31T23:59:59Z",
                "comment": "unrelated to doc 011's pin in every way",
            }
        ]
    }
    doc_012["compiler"]["generated_at"] = "2099-12-31T23:59:59Z"
    doc_012["compiler"]["run_id"] = "a-totally-different-run-id-xyz"

    assert canonicalize_playbook(doc_012) == canonicalize_playbook(doc_011)
    assert content_hash(doc_012) == content_hash(doc_011)
    digests_011 = compute_section_digests(doc_011)
    digests_012 = compute_section_digests(doc_012)
    assert digests_011["evidence"] == digests_012["evidence"]
    assert digests_011["posture"] == digests_012["posture"]
    assert digests_011["floor"] == digests_012["floor"]
    assert digests_011["curation"] != digests_012["curation"]
    vectors.append(
        (
            "012-excluded-metadata-variant-b",
            _vector(
                "excluded-metadata-variant-b",
                "Same evidence/posture/floor as 011-excluded-metadata-variant-a "
                "with unrecognizably different identity, curation pin, and "
                "compiler.generated_at/run_id. expected.canonical and "
                "expected.content_hash MUST equal 011's byte-for-byte; "
                "expected.section_digests.curation MUST differ from 011's "
                "(curation is excluded from content_hash but still gets its own "
                "lineage digest, §3.11).",
                doc_012,
            ),
        )
    )

    # 13 — digest dedupe/rank/cap machinery + frequency-band boundaries
    # (issue #115 fix round 1, finding 1): vectors 001-012 give every clause
    # at most one observed_position and empty acceptable_if/fallbacks/
    # rejected, so digest.py's dedupe/rank/top-N-plus-material cap
    # (_dedupe_rank / _preferred_variations) and the "often"/"sometimes"/
    # "rare" band boundaries (_BAND_OFTEN_MIN=10, _BAND_SOMETIMES_MIN=2)
    # were never exercised despite the normative "conformant with ...
    # digest construction" claim (OPF-SPEC.md §10.2, README.md). This
    # vector pins all of it: one clause whose observed_positions,
    # acceptable_if, fallbacks, and rejected each carry more than
    # EXEMPLAR_TOP_N (5) entries; each list includes a pair that collides
    # only after _normalize_text (case/punctuation/whitespace) and a
    # risk_delta-material entry ranked outside the top-N that must survive
    # the cap anyway; observed_positions additionally pins n=10 ("often")
    # and n=9 / n=2 (both "sometimes" — the boundary just below "often" and
    # the minimum for "sometimes").

    def _digest_probe_observation(
        text: str,
        clause_path: str,
        *,
        n: int = 1,
        magnitude: str = "none",
        direction: str = "neutral",
    ) -> dict[str, Any]:
        return {
            "text_summary": text,
            "full_text": text,
            "example_ref": {
                "document_id": "doc-1",
                "version": 1,
                "clause_path": clause_path,
                "char_span": [0, len(text)],
            },
            "deviation": "none",
            "risk_delta": {"direction": direction, "magnitude": magnitude},
            "provenance": "our_paper",
            "outcome": "signed",
            "precedent_count": n,
        }

    def _digest_probe_dedupe_list(prefix: str) -> list[dict[str, Any]]:
        """8 observation-shaped entries / 7 dedupe groups, for `fallbacks`/
        `rejected`: five rare (n=1) fillers (one dropped by the cap — proves
        the cap actually removes entries), a material entry ranked outside
        the top-N that must survive anyway, and a pair colliding only after
        `_normalize_text`."""
        entries = [
            _digest_probe_observation(
                f"{prefix} filler variant {i}.", f"unresolvable-{prefix}-{i}", n=1
            )
            for i in range(5)
        ]
        entries.append(
            _digest_probe_observation(
                f"{prefix} rare but material variant.",
                f"unresolvable-{prefix}-material",
                n=1,
                magnitude="material",
                direction="worse",
            )
        )
        entries.append(
            _digest_probe_observation(
                f"{prefix} Collision Variant — Duplicate Spelling.",
                f"unresolvable-{prefix}-collision-a",
                n=1,
            )
        )
        entries.append(
            _digest_probe_observation(
                f"{prefix}   collision variant,, duplicate spelling",
                f"unresolvable-{prefix}-collision-b",
                n=1,
            )
        )
        return entries

    # observed_positions: 11 entries / 10 dedupe groups. Ranked by (-n,
    # first_seen): often(n=10) > sometimes-hi(n=9) > sometimes-lo(n=2,
    # fs earlier) > collision(n=2, fs later) > filler-5 [top-5 cutoff here]
    # > filler-6..9 (dropped) > material (n=1, ranked outside top-5 — kept
    # only via the material union).
    observed_often = _digest_probe_observation(
        "Standard delivery clause, often-signed form.", "pos-often", n=10
    )
    observed_sometimes_hi = _digest_probe_observation(
        "Standard fallback clause, just-below-often form.", "pos-sometimes-hi", n=9
    )
    observed_sometimes_lo = _digest_probe_observation(
        "Minimum sometimes-band clause form.", "pos-sometimes-lo", n=2
    )
    observed_filler = [
        _digest_probe_observation(f"Rare filler clause form {i}.", f"pos-filler-{i}", n=1)
        for i in range(5, 10)
    ]
    observed_material = _digest_probe_observation(
        "Rare but material risk clause form.",
        "pos-material",
        n=1,
        magnitude="material",
        direction="worse",
    )
    observed_collision_a = _digest_probe_observation(
        "Collision clause FORM — duplicate spelling.", "pos-collision-a", n=1
    )
    observed_collision_b = _digest_probe_observation(
        "collision   clause form,, duplicate spelling", "pos-collision-b", n=1
    )
    dedupe_cap_observed_positions = (
        [observed_often, observed_sometimes_hi, observed_sometimes_lo]
        + observed_filler
        + [observed_material, observed_collision_a, observed_collision_b]
    )
    assert len(dedupe_cap_observed_positions) == 11  # >= _BAND_OFTEN_MIN (10)

    def _digest_probe_acceptable_if(label: str, clause_path: str | None) -> dict[str, Any]:
        return {
            "if": f"If the counterparty proposes {label}.",
            "to": f"Accepted alternative for {label}.",
            "rationale": f"Conformance probe entry — {label}.",
            "observation_ref": {
                "document_id": "doc-1",
                "version": 1,
                "clause_path": clause_path or f"unresolvable-{label}",
            },
        }

    # acceptable_if resolves its `n`/materiality via observation_ref against
    # this SAME clause's observed_positions (playbook_engine/digest.py
    # ::_preferred_variations), so the material entry below points at
    # observed_material's exact (document_id, version, clause_path) triple
    # rather than carrying its own risk_delta.
    accept_filler = [_digest_probe_acceptable_if(f"filler variant {i}", None) for i in range(5)]
    accept_material = _digest_probe_acceptable_if("the material risk form", "pos-material")
    accept_collision_a = {
        "if": "If the Counterparty Deletes Data upon termination.",
        "to": "Deletion occurs within 30 days of termination.",
        "rationale": "Conformance probe entry — collision variant a.",
        "observation_ref": {
            "document_id": "doc-1",
            "version": 1,
            "clause_path": "unresolvable-collision",
        },
    }
    accept_collision_b = {
        "if": "if   the counterparty deletes data upon-termination",
        "to": "deletion occurs within 30 days of termination",
        "rationale": (
            "Conformance probe entry — collision variant b (differs from "
            "variant a only in case/punctuation/whitespace)."
        ),
        "observation_ref": {
            "document_id": "doc-1",
            "version": 1,
            "clause_path": "unresolvable-collision",
        },
    }
    dedupe_cap_acceptable_if = accept_filler + [
        accept_collision_a,
        accept_collision_b,
        accept_material,
    ]
    assert len(dedupe_cap_acceptable_if) == 8  # > EXEMPLAR_TOP_N (5)

    dedupe_cap_fallbacks = _digest_probe_dedupe_list("Fallback")
    dedupe_cap_rejected = _digest_probe_dedupe_list("Rejected")
    assert len(dedupe_cap_fallbacks) == 8  # > EXEMPLAR_TOP_N (5)
    assert len(dedupe_cap_rejected) == 8  # > EXEMPLAR_TOP_N (5)

    dedupe_cap_clause: dict[str, Any] = {
        "id": "clause.dedupe_rank_cap",
        "taxonomy_id": "dedupe_rank_cap",
        "title": "Digest Dedupe/Rank/Cap Machinery Probe",
        "observed_positions": dedupe_cap_observed_positions,
        "summary": {
            "historical_stance": "mixed",
            "acceptable_if": dedupe_cap_acceptable_if,
            "fallbacks": dedupe_cap_fallbacks,
            "rejected": dedupe_cap_rejected,
            "confidence": {
                "score": 0.5,
                "basis": "precedent_count+provenance_mix",
                "n_our_paper": 11,
                "n_counterparty_paper": 0,
            },
        },
    }

    doc_013 = _base(clauses=[dedupe_cap_clause])
    entry_013 = build_digest(doc_013)["clauses"][0]

    # --- self-verifying assertions: pin the exact cap/dedupe/band outcomes
    # the fix requires, so a future digest.py refactor that silently changes
    # this behavior fails HERE (at generation time), not just as an opaque
    # byte-diff in the frozen vector.
    exemplar_by_text = {f["text_summary"]: f for f in entry_013["exemplar_forms"]}
    assert len(entry_013["exemplar_forms"]) == 6  # top-5 + the 1 material; 4 dropped
    assert exemplar_by_text["Standard delivery clause, often-signed form."]["band"] == "often"
    assert (
        exemplar_by_text["Standard fallback clause, just-below-often form."]["band"] == "sometimes"
    )
    assert exemplar_by_text["Minimum sometimes-band clause form."]["band"] == "sometimes"
    assert exemplar_by_text["Rare but material risk clause form."]["band"] == "rare"
    assert (
        exemplar_by_text["Rare but material risk clause form."]["risk_delta"]["magnitude"]
        == "material"
    )
    collision_forms = [
        f for f in entry_013["exemplar_forms"] if "collision" in f["text_summary"].lower()
    ]
    assert len(collision_forms) == 1  # the pair merged into one group
    assert collision_forms[0]["n"] == 2
    assert collision_forms[0]["band"] == "sometimes"
    filler_forms = [
        f
        for f in entry_013["exemplar_forms"]
        if f["text_summary"].startswith("Rare filler clause form")
    ]
    assert len(filler_forms) == 1  # only 1 of 5 rare fillers survives the cap

    assert len(entry_013["preferred_variations"]) == 6  # top-5 + the 1 material; 1 dropped
    material_accept = next(
        a for a in entry_013["preferred_variations"] if "material risk form" in a["if"]
    )
    assert material_accept["n"] == 1
    collision_accept = [
        a
        for a in entry_013["preferred_variations"]
        if "counterparty deletes data" in a["if"].lower()
    ]
    assert len(collision_accept) == 1  # the pair merged into one group
    assert collision_accept[0]["n"] == 2

    assert len(entry_013["concessions"]) == 6  # top-5 + the 1 material; 1 dropped
    assert len(entry_013["unacceptable"]) == 6  # top-5 + the 1 material; 1 dropped
    assert next(c for c in entry_013["concessions"] if "material" in c["text_summary"])["n"] == 1
    assert next(u for u in entry_013["unacceptable"] if "material" in u["text_summary"])["n"] == 1

    vectors.append(
        (
            "013-digest-dedupe-rank-and-bands",
            _vector(
                "digest-dedupe-rank-and-bands",
                "One clause whose observed_positions (11 entries), "
                "acceptable_if (8), fallbacks (8), and rejected (8) each "
                "exceed EXEMPLAR_TOP_N (5), pinning digest.py's dedupe/rank/"
                "top-N-plus-material cap (_dedupe_rank / "
                "_preferred_variations) and the often/sometimes/rare band "
                "boundaries — a gap vectors 001-012 left unconstrained "
                "despite the normative digest-construction conformance "
                "claim. Each of the four lists includes one pair that "
                "collides only after _normalize_text (case/punctuation/"
                "whitespace) and one risk_delta-material entry ranked "
                "outside the top-5 by frequency that MUST survive the cap "
                "anyway; observed_positions additionally pins n=10 -> "
                "'often', and n=9 / n=2 -> 'sometimes' (the boundary just "
                "below 'often' and the minimum for 'sometimes').",
                doc_013,
            ),
        )
    )

    return vectors


def main() -> None:
    VECTORS_DIR.mkdir(parents=True, exist_ok=True)
    vectors = build_vectors()

    manifest_entries = []
    for filename, vector in vectors:
        path = VECTORS_DIR / f"{filename}.json"
        path.write_text(
            json.dumps(vector, indent=2, sort_keys=False, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        manifest_entries.append(
            {
                "name": vector["name"],
                "file": f"vectors/{filename}.json",
                "description": vector["description"],
            }
        )

    manifest = {
        "format_version": {
            "opf_version": OPF_VERSION,
            "engine_version": ENGINE_VERSION,
            "digest_version": DIGEST_VERSION,
        },
        "algorithm": (
            "canonical: json.dumps(value, sort_keys=True, separators=(',', ':'), "
            "ensure_ascii=False) restricted to the whole-document form used for "
            "content_hash — i.e. after removing the top-level 'identity' and "
            "'curation' keys and the 'compiler.generated_at'/'compiler.run_id' "
            "sub-keys from a copy of the input (see canonicalize.py). "
            "content_hash: 'sha256:' + hex(sha256(canonical.encode('utf-8'))). "
            "section_digests[name]: 'sha256:' + hex(sha256(canonical(input.get(name, "
            "{})).encode('utf-8'))) for name in evidence/posture/floor/curation — "
            "NOT excluding anything (a section has no self-referential fields). "
            "digest: playbook_engine.digest.build_digest(input) with the default "
            "token_budget."
        ),
        "vectors": manifest_entries,
    }
    (CONFORMANCE_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(vectors)} vectors + manifest.json to {CONFORMANCE_DIR}")


if __name__ == "__main__":
    main()
