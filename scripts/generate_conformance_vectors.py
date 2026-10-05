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

    .venv/bin/python scripts/generate_conformance_vectors.py                    # 0.3 set
    .venv/bin/python scripts/generate_conformance_vectors.py --opf-version 0.4  # 0.4 set

The default (0.3) mode writes ``spec/conformance/manifest.json`` and one file
per vector under ``spec/conformance/vectors/`` — the FROZEN OPF 0.3 /
digest_version 2 set; re-running it must reproduce the committed files
byte-for-byte. ``--opf-version 0.4`` (issue #223) writes the separately
stamped OPF 0.4 / digest_version 3 set under ``spec/conformance/0.4/``
(``manifest.json`` + ``vectors/``) and never touches the 0.3 set. Review the
resulting diff like any other spec change (it needs a ``spec/CHANGELOG.md``
entry) before committing.
"""

from __future__ import annotations

import argparse
import copy
import json
import unicodedata
from pathlib import Path
from typing import Any

from playbook_engine.canonicalize import (
    canonicalize_playbook,
    compute_section_digests,
    content_hash,
)
from playbook_engine.digest import DIGEST_VERSION as DIGEST_VERSION_V3
from playbook_engine.digest import DIGEST_VERSION_V2, build_digest
from playbook_engine.opf_accessors import perspective_party
from playbook_engine.precedent import clause_counts, precedent_id

#: The 0.3 set's digest stamp. ``digest.DIGEST_VERSION`` is the CURRENT digest
#: version (3 since issue #223); the frozen 0.3 set is digest_version 2.
DIGEST_VERSION = DIGEST_VERSION_V2

ROOT = Path(__file__).parent.parent
CONFORMANCE_DIR = ROOT / "spec" / "conformance"
VECTORS_DIR = CONFORMANCE_DIR / "vectors"

OPF_VERSION = "0.3"

#: The reference ``engine_version`` this vector set is stamped with (and the
#: fixture ``compiler.version`` every input carries). Pinned rather than read
#: from ``playbook_engine.__version__`` so re-running this script reproduces
#: the committed vectors byte-for-byte: the set was generated against 1.0.0
#: (issue #115), and its one in-place amendment — digest v2 ``n`` counting
#: distinct deals (issue #216, 2026-09-25, owner-authorized exception to
#: OPF-SPEC §11) — is recorded in spec/CHANGELOG.md, not in this stamp.
ENGINE_VERSION = "1.0.0"


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
        document_id: str = "doc-1",
    ) -> dict[str, Any]:
        return {
            "text_summary": text,
            "full_text": text,
            "example_ref": {
                "document_id": document_id,
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
        `_normalize_text`, cited to two distinct deals (n=2 — issue #216:
        `n` counts distinct `document_id`s)."""
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
                document_id="doc-2",
            )
        )
        return entries

    # observed_positions: 30 rows / 10 dedupe groups (issue #216): every
    # row of a text stamped with that text's distinct-deal precedent_count,
    # and `n` = the group's DISTINCT document_ids — never a sum of
    # precedent_count. Every row except the collision group's third is in
    # the shape the compiler emits: one signed row per (deal, clause), each
    # on its own document_id.
    # Ranked by (-n, first_seen): often(n=10) > sometimes-hi(n=9) >
    # sometimes-lo(n=2, fs earlier) > collision(n=2, fs later) > filler-5
    # [top-5 cutoff here] > filler-6..9 (dropped) > material (n=1, ranked
    # outside top-5 — kept only via the material union).
    def _digest_probe_deals(
        text: str, clause_path: str, k: int, deal_prefix: str
    ) -> list[dict[str, Any]]:
        return [
            _digest_probe_observation(text, clause_path, n=k, document_id=f"{deal_prefix}-{j}")
            for j in range(1, k + 1)
        ]

    observed_often = _digest_probe_deals(
        "Standard delivery clause, often-signed form.", "pos-often", 10, "doc-often"
    )
    observed_sometimes_hi = _digest_probe_deals(
        "Standard fallback clause, just-below-often form.", "pos-sometimes-hi", 9, "doc-hi"
    )
    observed_sometimes_lo = _digest_probe_deals(
        "Minimum sometimes-band clause form.", "pos-sometimes-lo", 2, "doc-lo"
    )
    observed_filler = [
        _digest_probe_observation(
            f"Rare filler clause form {i}.",
            f"pos-filler-{i}",
            n=1,
            document_id=f"doc-filler-{i}",
        )
        for i in range(5, 10)
    ]
    observed_material = _digest_probe_observation(
        "Rare but material risk clause form.",
        "pos-material",
        n=1,
        magnitude="material",
        direction="worse",
        document_id="doc-material",
    )
    # The collision pair spans two deals; the third row repeats spelling a in
    # the SAME deal as spelling b — one deal carrying two texts the digest
    # merges counts once, so the merged group's n is 2, not 3. That second
    # signed row for one (deal, clause) is NOT a shape the compiler emits
    # (it emits exactly one signed row per deal and taxonomy_id): it models
    # a hand-curated or legacy-store observed_positions input, which the
    # digest must still count by distinct deal.
    observed_collision_a = _digest_probe_observation(
        "Collision clause FORM — duplicate spelling.",
        "pos-collision-a",
        n=1,
        document_id="doc-collision-1",
    )
    observed_collision_b = _digest_probe_observation(
        "collision   clause form,, duplicate spelling",
        "pos-collision-b",
        n=1,
        document_id="doc-collision-2",
    )
    observed_collision_b_again = _digest_probe_observation(
        "Collision clause form -- duplicate spelling",
        "pos-collision-c",
        n=1,
        document_id="doc-collision-2",
    )
    dedupe_cap_observed_positions = (
        observed_often
        + observed_sometimes_hi
        + observed_sometimes_lo
        + observed_filler
        + [observed_material, observed_collision_a, observed_collision_b]
        + [observed_collision_b_again]
    )
    assert len(dedupe_cap_observed_positions) == 30
    # confidence.n_our_paper counts distinct our-paper DEALS (issue #216).
    dedupe_cap_n_our_paper = len(
        {
            row["example_ref"]["document_id"]
            for row in dedupe_cap_observed_positions
            if row["provenance"] == "our_paper"
        }
    )
    assert dedupe_cap_n_our_paper == 29

    def _digest_probe_acceptable_if(
        label: str, clause_path: str | None, document_id: str = "doc-1"
    ) -> dict[str, Any]:
        return {
            "if": f"If the counterparty proposes {label}.",
            "to": f"Accepted alternative for {label}.",
            "rationale": f"Conformance probe entry — {label}.",
            "observation_ref": {
                "document_id": document_id,
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
    accept_material = _digest_probe_acceptable_if(
        "the material risk form", "pos-material", document_id="doc-material"
    )
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
            "variant a only in case/punctuation/whitespace; cited to a second "
            "deal, so the merged entry's n is 2 distinct deals)."
        ),
        "observation_ref": {
            "document_id": "doc-2",
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
                "n_our_paper": dedupe_cap_n_our_paper,
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
    assert len(collision_forms) == 1  # the three rows merged into one group
    assert collision_forms[0]["n"] == 2  # 2 distinct deals, not 3 rows
    assert exemplar_by_text["Standard delivery clause, often-signed form."]["n"] == 10
    assert exemplar_by_text["Standard fallback clause, just-below-often form."]["n"] == 9
    assert exemplar_by_text["Minimum sometimes-band clause form."]["n"] == 2
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
    assert collision_accept[0]["n"] == 2  # doc-1 + doc-2

    assert len(entry_013["concessions"]) == 6  # top-5 + the 1 material; 1 dropped
    assert len(entry_013["unacceptable"]) == 6  # top-5 + the 1 material; 1 dropped
    assert next(c for c in entry_013["concessions"] if "material" in c["text_summary"])["n"] == 1
    assert next(u for u in entry_013["unacceptable"] if "material" in u["text_summary"])["n"] == 1
    for listed in (entry_013["concessions"], entry_013["unacceptable"]):
        assert next(e for e in listed if "collision" in e["text_summary"].lower())["n"] == 2

    vectors.append(
        (
            "013-digest-dedupe-rank-and-bands",
            _vector(
                "digest-dedupe-rank-and-bands",
                "One clause whose observed_positions (30 rows), "
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
                "below 'often' and the minimum for 'sometimes'). Every n is "
                "the number of DISTINCT example_ref.document_id values in its "
                "group, never a sum of precedent_count (issue #216, amended "
                "in place 2026-09-25): observed_positions carries one row per "
                "(deal, text) stamped with that text's deal count, and its "
                "collision group includes one deal carrying two of the merged "
                "spellings, which counts once (a hand-curated or legacy-store "
                "input shape the compiler itself never emits; the digest must "
                "still count it by distinct deal).",
                doc_013,
            ),
        )
    )

    return vectors


# ---------------------------------------------------------------------------
# OPF 0.4 / digest_version 3 set (issue #223) — spec/conformance/0.4/
# ---------------------------------------------------------------------------

OPF_VERSION_V04 = "0.4"
CONFORMANCE_DIR_V04 = CONFORMANCE_DIR / "0.4"
VECTORS_DIR_V04 = CONFORMANCE_DIR_V04 / "vectors"
_AGREEMENT_TYPE_ID = "conformance-fixture"


def _base_v04(
    *,
    clauses: list[dict[str, Any]] | None = None,
    precedent: list[dict[str, Any]] | None = None,
    documents: list[dict[str, Any]] | None = None,
    perspective: dict[str, str] | None = None,
) -> dict[str, Any]:
    """A minimal OPF 0.4 document. Clause counts and precedent ids are
    stamped by the reference functions (``precedent.clause_counts`` /
    ``precedent.precedent_id``) so every input is self-consistent — a vector
    a validator would reject is not a useful conformance input."""
    doc: dict[str, Any] = {
        "opf_version": OPF_VERSION_V04,
        "agreement_type": {"id": _AGREEMENT_TYPE_ID, "name": "Conformance Fixture Agreement"},
        "baseline": {"has_canonical_template": True},
        "taxonomy": {"source": "custom", "entries": []},
    }
    if perspective is not None:
        doc["perspective"] = perspective
    records = precedent or []
    for record in records:
        signed_text = record.get("signed_text")
        record["id"] = precedent_id(
            _AGREEMENT_TYPE_ID,
            record["document_id"],
            record["taxonomy_id"],
            signed_text["text"] if isinstance(signed_text, dict) else None,
        )
    party = perspective_party(doc)
    stamped = []
    for clause in clauses or []:
        stamped.append({**clause, **clause_counts(clause["taxonomy_id"], records, party=party)})
    doc["evidence"] = {"clauses": stamped, "precedent": records}
    doc["posture"] = {}
    doc["floor"] = {}
    doc["corpus"] = {"documents": documents or [], "stats": {}}
    doc["compiler"] = {
        "name": "playbook-engine",
        "version": ENGINE_VERSION,
        "run_id": "conformance-fixture-run",
        "generated_at": "2026-01-01T00:00:00Z",
    }
    return doc


def _ref(document_id: str, version: int, clause_path: str = "1") -> dict[str, Any]:
    return {
        "document_id": document_id,
        "version": version,
        "clause_path": clause_path,
        "char_span": [0, 40],
    }


def _deal(document_id: str, *, signed_version: int | None, provenance: str) -> dict[str, Any]:
    return {
        "document_id": document_id,
        "provenance": provenance,
        "in_scope": True,
        "versions": 3,
        "signed_version": signed_version,
    }


def _prec(
    document_id: str,
    taxonomy_id: str,
    *,
    signed_text: str | None,
    standard: bool = False,
    signed: bool = True,
    signed_at: str | None = None,
    opening_text: str | None = None,
    refused: list[tuple[str, int]] | None = None,
    paper: str = "ours",
    rounds: int = 0,
    counterparty_alias: str | None = None,
) -> dict[str, Any]:
    refused_asks = [
        {"text": text, "round": version - 1, "ref": _ref(document_id, version)}
        for text, version in (refused or [])
    ]
    record: dict[str, Any] = {
        "id": "",
        "taxonomy_id": taxonomy_id,
        "document_id": document_id,
    }
    if counterparty_alias is not None:
        record["counterparty_ref"] = {"alias": counterparty_alias}
    record.update(
        {
            "paper": paper,
            "paper_basis": (
                "provenance_detection" if paper != "unknown" else "ambiguous_detection"
            ),
            "paper_confidence": 0.9 if paper != "unknown" else None,
            "signed": signed,
        }
    )
    if signed_at is not None:
        record["signed_at"] = signed_at
    record.update(
        {
            "rounds": rounds,
            "signed_text": (
                {"text": signed_text, "ref": _ref(document_id, 3)}
                if signed_text is not None
                else None
            ),
            "opening_text": (
                {"text": opening_text, "ref": _ref(document_id, 1)}
                if opening_text is not None
                else None
            ),
            "standard": standard,
            "moved": bool(rounds or opening_text is not None or refused_asks),
            "refused_asks": refused_asks,
        }
    )
    return record


def _vector_v04(name: str, description: str, doc: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": name,
        "description": description,
        "opf_version": OPF_VERSION_V04,
        "engine_version": ENGINE_VERSION,
        "digest_version": DIGEST_VERSION_V3,
        "input": doc,
        "expected": _expected(doc),
    }


_STANDARD_TEXT = (
    "Neither party may assign this Agreement without the other party's prior written consent."
)


def build_vectors_v04() -> list[tuple[str, dict[str, Any]]]:
    """The OPF 0.4 / digest 3 vector set. Synthetic inputs only."""
    vectors: list[tuple[str, dict[str, Any]]] = []

    # 001 — smallest 0.4 document: no clauses, no perspective.
    doc_001 = _base_v04()
    digest_001 = build_digest(doc_001)
    assert digest_001["perspective"] is None
    assert digest_001["clauses"] == []
    vectors.append(
        (
            "001-minimal-no-perspective",
            _vector_v04(
                "minimal-no-perspective",
                "Smallest well-formed OPF 0.4 document: evidence {clauses: [], "
                "precedent: []}, no top-level perspective. expected.digest pins "
                "the digest_version 3 skeleton — perspective is PRESENT and null "
                "(never omitted), agreement_type is {id, name}, corpus counts are "
                "0 and first_signed/last_signed are null.",
                doc_001,
            ),
        )
    )

    # 002 — grouping, ordering and every exclusion rule on one clause.
    tid = "assignment"
    variant_a = "Either party may assign this Agreement to an affiliate without consent."
    documents_002 = [
        _deal("deal-a", signed_version=3, provenance="our_paper"),
        _deal("deal-b", signed_version=3, provenance="our_paper"),
        _deal("deal-c", signed_version=3, provenance="counterparty_paper"),
        _deal("deal-d", signed_version=3, provenance="counterparty_paper"),
        _deal("deal-e", signed_version=3, provenance="our_paper"),
        _deal("deal-f", signed_version=None, provenance="counterparty_paper"),
        _deal("deal-g", signed_version=3, provenance="our_paper"),
        _deal("deal-h", signed_version=3, provenance="counterparty_paper"),
    ]
    precedent_002 = [
        _prec("deal-a", tid, signed_text=_STANDARD_TEXT, standard=True, signed_at="2025-01-10"),
        _prec("deal-b", tid, signed_text=_STANDARD_TEXT, standard=True, signed_at="2025-02-10"),
        # Variant A, three deals; deal-d spells it with different case and
        # punctuation (same group), and carries the latest signed_at as a
        # publish-coarsened quarter.
        _prec("deal-c", tid, signed_text=variant_a, signed_at="2025-03-01", paper="theirs"),
        _prec(
            "deal-d",
            tid,
            signed_text="EITHER party may assign this Agreement to an affiliate, without consent",
            signed_at="2025-Q4",
            paper="theirs",
            refused=[("Either party may assign this Agreement freely.", 2)],
            rounds=1,
        ),
        _prec("deal-e", tid, signed_text=variant_a, rounds=1),
        # Variant B differs from A by a negator only: exact normalization keeps
        # it a separate group (no similarity tolerance).
        _prec(
            "deal-g",
            tid,
            signed_text="Either party may not assign this Agreement to an affiliate without consent.",
            signed_at="2024-06-30",
        ),
        # An unsigned deal's last-draft text is never a signed variant.
        _prec(
            "deal-f",
            tid,
            signed_text="Either party may assign this Agreement on notice.",
            signed=False,
            paper="unknown",
            refused=[("Either party may assign this Agreement freely.", 3)],
        ),
        # Our standard struck before signing: no signed text, an opening text,
        # and the same refused-ask text as deal-d in a different spelling.
        _prec(
            "deal-h",
            tid,
            signed_text=None,
            opening_text=_STANDARD_TEXT,
            refused=[("either party may assign this agreement freely", 1)],
            paper="theirs",
        ),
    ]
    doc_002 = _base_v04(
        clauses=[
            {
                "id": f"clause.{tid}",
                "taxonomy_id": tid,
                "title": "Assignment",
                "our_standard": {
                    "text": _STANDARD_TEXT,
                    "source_ref": {
                        "document_id": "template",
                        "version": "template",
                        "clause_path": "9",
                    },
                },
            }
        ],
        precedent=precedent_002,
        documents=documents_002,
        perspective={"party": "Fixture Co", "counterparty_type": "Fixture Counterparty"},
    )
    entry_002 = build_digest(doc_002)["clauses"][0]
    assert [v["n_deals"] for v in entry_002["signed_variants"]] == [3, 1]
    assert entry_002["signed_variants"][0]["last_signed"] == "2025-Q4"
    assert entry_002["refused_asks"][0]["n_deals"] == 3
    assert entry_002["n_signed_standard"] == 2 and entry_002["n_deals"] == 8
    vectors.append(
        (
            "002-variants-refused-and-exclusions",
            _vector_v04(
                "variants-refused-and-exclusions",
                "One clause, eight deals, pinning digest_version 3 grouping and "
                "every exclusion rule: two deals signed our standard "
                "(n_signed_standard 2, never a variant); variant A signed in "
                "three deals, one spelled with different case/punctuation (exact "
                "normalization merges it) — its representative is the deal with "
                "the latest signed_at, a YYYY-Qn quarter that sorts at the "
                "quarter's first day; variant B differs from A by the negator "
                "'not' only and stays a separate group; an unsigned deal's text "
                "is never a signed variant; a deal whose standard was struck "
                "(signed_text null, opening_text set) contributes no variant. "
                "The same refused ask in three deals (three spellings) is one "
                "group with n_deals 3 whose representative is the earliest-round "
                "ask. perspective is copied into the digest.",
                doc_002,
            ),
        )
    )

    # 003 — the cap and the uncapped totals; the 300-char summary.
    tid_003 = "limitation_of_liability"
    long_text = (
        "Each party's aggregate liability arising out of or relating to this "
        "Agreement shall not exceed fifty thousand dollars. This limitation does "
        "not apply to a breach of the confidentiality obligations in this "
        "Agreement, to a party's indemnification obligations, or to a party's "
        "gross negligence or wilful misconduct. Neither party is liable for any "
        "indirect, incidental or consequential damages."
    )
    assert len(long_text) > 300
    precedent_003 = [
        _prec(f"deal-{i}", tid_003, signed_text=f"Liability is capped at {i} times the fees paid.")
        for i in range(1, 7)
    ]
    precedent_003.append(_prec("deal-7", tid_003, signed_text=long_text))
    precedent_003.append(_prec("deal-8", tid_003, signed_text=long_text, signed_at="2025-05-05"))
    doc_003 = _base_v04(
        clauses=[
            {
                "id": f"clause.{tid_003}",
                "taxonomy_id": tid_003,
                "title": "Limitation of Liability",
                "our_standard": None,
            }
        ],
        precedent=precedent_003,
        documents=[
            _deal(f"deal-{i}", signed_version=3, provenance="our_paper") for i in range(1, 9)
        ],
    )
    entry_003 = build_digest(doc_003)["clauses"][0]
    assert entry_003["n_variants_total"] == 7
    assert len(entry_003["signed_variants"]) == 5
    assert len(entry_003["signed_variants"][0]["text"]) <= 300
    vectors.append(
        (
            "003-cap-totals-and-summary",
            _vector_v04(
                "cap-totals-and-summary",
                "One clause with seven distinct signed variants: the digest lists "
                "the top 5 (n_deals desc, last_signed desc with unknown last, then "
                "normalized text) and n_variants_total reports all 7. The "
                "two-deal variant's text exceeds 300 characters, pinning the "
                "sentence-boundary summary (a verbatim prefix ending on a "
                "sentence end, <= 300 chars). our_standard is null.",
                doc_003,
            ),
        )
    )

    # 004 — the grouping key's party neutralization. Each deal's counterparty
    # carries its own entity-registry alias (Counterparty-<n>) in the clause
    # text, so the key rewrites every alias to "counterparty" and the
    # document's perspective.party to "party" — two distinct tokens.
    tid_004 = "termination"
    party_004 = "Fixture Co"
    precedent_004 = [
        # Same words, different counterparty alias: one variant, n_deals 2.
        _prec(
            "deal-a",
            tid_004,
            signed_text=f"Counterparty-2 may terminate this Agreement on notice to {party_004}.",
            refused=[("Counterparty-2 may terminate this Agreement at any time.", 2)],
            counterparty_alias="Counterparty-2",
            rounds=1,
        ),
        _prec(
            "deal-b",
            tid_004,
            signed_text=f"Counterparty-5 may terminate this Agreement on notice to {party_004}.",
            counterparty_alias="Counterparty-5",
        ),
        # The same words with the parties' places swapped: our party and the
        # counterparty are distinct tokens, so it stays its own variant.
        _prec(
            "deal-c",
            tid_004,
            signed_text=f"{party_004} may terminate this Agreement on notice to Counterparty-3.",
            refused=[("Counterparty-3 may terminate this Agreement at any time.", 2)],
            counterparty_alias="Counterparty-3",
            rounds=1,
        ),
        # Sorts before deal-c's variant only once perspective.party is
        # rewritten ("notice ..." < "party ...", but "fixture co ..." < "notice ...").
        _prec(
            "deal-d",
            tid_004,
            signed_text=f"Notice of termination given to {party_004} is effective on receipt.",
        ),
    ]
    doc_004 = _base_v04(
        clauses=[
            {
                "id": f"clause.{tid_004}",
                "taxonomy_id": tid_004,
                "title": "Termination",
                "our_standard": {
                    "text": "Either party may terminate this Agreement on thirty days' notice.",
                    "source_ref": {
                        "document_id": "template",
                        "version": "template",
                        "clause_path": "11",
                    },
                },
            }
        ],
        precedent=precedent_004,
        documents=[_deal(f"deal-{c}", signed_version=3, provenance="our_paper") for c in "abcd"],
        perspective={"party": party_004, "counterparty_type": "Fixture Counterparty"},
    )
    entry_004 = build_digest(doc_004)["clauses"][0]
    assert [(v["n_deals"], v["ref"]["document_id"]) for v in entry_004["signed_variants"]] == [
        (2, "deal-a"),
        (1, "deal-d"),
        (1, "deal-c"),
    ]
    assert [a["n_deals"] for a in entry_004["refused_asks"]] == [2]
    assert doc_004["evidence"]["clauses"][0]["n_variants"] == 3
    assert doc_004["evidence"]["clauses"][0]["n_refused"] == 1
    vectors.append(
        (
            "004-party-alias-grouping",
            _vector_v04(
                "party-alias-grouping",
                "The grouping key's party neutralization (OPF-SPEC §3.5.4): "
                "every entity-registry alias (Counterparty-<n>, case-insensitive, "
                "on word boundaries) becomes the token 'counterparty' and the "
                "document's perspective.party becomes the token 'party' before "
                "case and punctuation are dropped. Two deals whose signed texts differ only by "
                "the counterparty alias (Counterparty-2 vs Counterparty-5) are "
                "one variant with n_deals 2, and two refused asks that differ "
                "only by alias are one group with n_deals 2. The same words "
                "with the parties' places swapped stay a separate variant (our "
                "party and the counterparty are distinct tokens). The order of "
                "the two one-deal variants pins the perspective.party rewrite: "
                "'notice of termination given to party ...' sorts before "
                "'party may terminate ...'.",
                doc_004,
            ),
        )
    )
    return vectors


def main_v04() -> None:
    VECTORS_DIR_V04.mkdir(parents=True, exist_ok=True)
    vectors = build_vectors_v04()
    manifest_entries = []
    for filename, vector in vectors:
        path = VECTORS_DIR_V04 / f"{filename}.json"
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
            "opf_version": OPF_VERSION_V04,
            "engine_version": ENGINE_VERSION,
            "digest_version": DIGEST_VERSION_V3,
        },
        "algorithm": (
            "canonical / content_hash / section_digests: exactly as the 0.3 set "
            "(../manifest.json). digest: playbook_engine.digest.build_digest(input) "
            "with the default token_budget, which for opf_version 0.4 is "
            "build_digest_v3 (OPF-SPEC.md §3.12.1): signed variants and refused asks "
            "grouped by the grouping key of OPF-SPEC.md §3.5.4 (precedent."
            "normalize_variant_text: every Counterparty-<n> alias -> 'counterparty', "
            "then deviation_classifier.normalize_for_standard with the document's "
            "perspective.party as the one party name -> 'party'), text = "
            "observation_builder.summarize_clause_text of the group representative."
        ),
        "vectors": manifest_entries,
    }
    (CONFORMANCE_DIR_V04 / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(vectors)} vectors + manifest.json to {CONFORMANCE_DIR_V04}")


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
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--opf-version", choices=("0.3", "0.4"), default="0.3")
    args = parser.parse_args()
    if args.opf_version == "0.4":
        main_v04()
    else:
        main()
