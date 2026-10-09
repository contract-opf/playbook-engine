#!/usr/bin/env python3
"""Regenerate the frozen conformance vectors under ``spec/conformance/0.5/`` — issue #115.

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

Writes the OPF 0.5 / digest_version 3 set (issues #223, #233) under
``spec/conformance/0.5/`` (``manifest.json`` + ``vectors/``); re-running it
must reproduce the committed files byte-for-byte. The OPF 0.3 / digest 2 set
and the 0.4 set (the same documents without opening evidence) were retired
with those formats (issues #238, #233) — git history has them. Review the resulting diff like any other spec change (it needs a
``spec/CHANGELOG.md`` entry) before committing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from playbook_engine.canonicalize import (
    canonicalize_playbook,
    compute_section_digests,
    content_hash,
)
from playbook_engine.digest import DIGEST_VERSION as DIGEST_VERSION_V3
from playbook_engine.digest import build_digest
from playbook_engine.opf_accessors import perspective_party
from playbook_engine.precedent import clause_counts, precedent_id

ROOT = Path(__file__).parent.parent
CONFORMANCE_DIR = ROOT / "spec" / "conformance"

#: The reference ``engine_version`` this vector set is stamped with (and the
#: fixture ``compiler.version`` every input carries). Pinned rather than read
#: from ``playbook_engine.__version__`` so re-running this script reproduces
#: the committed vectors byte-for-byte.
ENGINE_VERSION = "1.0.0"


def _expected(doc: dict[str, Any]) -> dict[str, Any]:
    return {
        "canonical": canonicalize_playbook(doc),
        "content_hash": content_hash(doc),
        "section_digests": compute_section_digests(doc),
        "digest": build_digest(doc),
    }


# ---------------------------------------------------------------------------
# OPF 0.5 / digest_version 3 set (issues #223, #233) — spec/conformance/0.5/
# ---------------------------------------------------------------------------

OPF_VERSION_V05 = "0.5"
CONFORMANCE_DIR_V05 = CONFORMANCE_DIR / "0.5"
VECTORS_DIR_V05 = CONFORMANCE_DIR_V05 / "vectors"
_AGREEMENT_TYPE_ID = "conformance-fixture"


def _base_v05(
    *,
    clauses: list[dict[str, Any]] | None = None,
    precedent: list[dict[str, Any]] | None = None,
    documents: list[dict[str, Any]] | None = None,
    perspective: dict[str, str] | None = None,
) -> dict[str, Any]:
    """A minimal OPF 0.5 document. Clause counts and precedent ids are
    stamped by the reference functions (``precedent.clause_counts`` /
    ``precedent.precedent_id``) so every input is self-consistent — a vector
    a validator would reject is not a useful conformance input."""
    doc: dict[str, Any] = {
        "opf_version": OPF_VERSION_V05,
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


#: Sentinel: ``_prec`` derives ``opened_with`` from the other arguments.
_DEFAULT: object = object()


def _prec(
    document_id: str,
    taxonomy_id: str,
    *,
    signed_text: str | None,
    standard: bool = False,
    signed: bool = True,
    signed_at: str | None = None,
    opening_text: str | None = None,
    opened_with: str | None | object = _DEFAULT,
    refused: list[tuple[str, int]] | None = None,
    paper: str = "ours",
    rounds: int = 0,
    counterparty_alias: str | None = None,
) -> dict[str, Any]:
    refused_asks = [
        {"text": text, "round": version - 1, "ref": _ref(document_id, version)}
        for text, version in (refused or [])
    ]
    if opened_with is _DEFAULT:
        # An unsigned deal's opening is not anchored (null); otherwise a
        # struck standard opened with the standard, and a clause with no
        # distinct opening opened with the text it signed.
        if not signed:
            opened_with = None
        elif opening_text is not None or standard:
            opened_with = "standard"
        else:
            opened_with = "non_standard"
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
            "opened_with": opened_with,
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


def _vector_v05(name: str, description: str, doc: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": name,
        "description": description,
        "opf_version": OPF_VERSION_V05,
        "engine_version": ENGINE_VERSION,
        "digest_version": DIGEST_VERSION_V3,
        "input": doc,
        "expected": _expected(doc),
    }


_STANDARD_TEXT = (
    "Neither party may assign this Agreement without the other party's prior written consent."
)


def build_vectors_v05() -> list[tuple[str, dict[str, Any]]]:
    """The OPF 0.5 / digest 3 vector set. Synthetic inputs only."""
    vectors: list[tuple[str, dict[str, Any]]] = []

    # 001 — smallest 0.5 document: no clauses, no perspective.
    doc_001 = _base_v05()
    digest_001 = build_digest(doc_001)
    assert digest_001["perspective"] is None
    assert digest_001["clauses"] == []
    vectors.append(
        (
            "001-minimal-no-perspective",
            _vector_v05(
                "minimal-no-perspective",
                "Smallest well-formed OPF 0.5 document: evidence {clauses: [], "
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
    doc_002 = _base_v05(
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
            _vector_v05(
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
    doc_003 = _base_v05(
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
            _vector_v05(
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
    doc_004 = _base_v05(
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
            _vector_v05(
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
    # 005 — opening evidence (OPF-SPEC §3.5.5, issue #233): opened_with and
    # opening_text for every origin, and each case where opening_text is null.
    tid_005 = "governing_law"
    std_005 = "This Agreement is governed by the laws of the State of Delaware."
    ny_005 = "This Agreement is governed by the laws of the State of New York."
    cap_005 = "Each party's liability is capped at fifty thousand dollars."
    precedent_005 = [
        # Opened with our standard, edited before signing: a distinct opening.
        _prec("deal-a", tid_005, signed_text=ny_005, opening_text=std_005, rounds=1),
        # Opened non-standard, changed to our standard: the opening is recorded
        # whatever its origin, and the signed text is standard.
        _prec(
            "deal-b",
            tid_005,
            signed_text=std_005,
            standard=True,
            opening_text=ny_005,
            opened_with="non_standard",
            rounds=1,
            paper="theirs",
        ),
        # Opened non-standard and signed unchanged: nothing distinct to record.
        _prec("deal-c", tid_005, signed_text=ny_005, opened_with="non_standard", paper="theirs"),
        # Added during the negotiation: the first draft had no such clause.
        _prec("deal-d", tid_005, signed_text=ny_005, opened_with="absent", rounds=1),
        # An unsigned deal's opening is not anchored.
        _prec("deal-e", tid_005, signed_text=ny_005, signed=False, paper="unknown"),
        # A case/whitespace/punctuation-only edit is not a distinct opening
        # under the section 3.5.4 grouping key.
        _prec(
            "deal-f",
            tid_005,
            signed_text=std_005.upper(),
            standard=True,
            rounds=1,
        ),
        # Opened non-standard and struck before signing: opening_text with no
        # signed text (and no claim that it was a refused ask).
        _prec(
            "deal-g",
            tid_005,
            signed_text=None,
            opening_text=cap_005,
            opened_with="non_standard",
        ),
    ]
    doc_005 = _base_v05(
        clauses=[
            {
                "id": f"clause.{tid_005}",
                "taxonomy_id": tid_005,
                "title": "Governing Law",
                "our_standard": {
                    "text": std_005,
                    "source_ref": {
                        "document_id": "template",
                        "version": "template",
                        "clause_path": "14",
                    },
                },
            }
        ],
        precedent=precedent_005,
        documents=[
            _deal("deal-a", signed_version=3, provenance="our_paper"),
            _deal("deal-b", signed_version=3, provenance="counterparty_paper"),
            _deal("deal-c", signed_version=3, provenance="counterparty_paper"),
            _deal("deal-d", signed_version=3, provenance="our_paper"),
            _deal("deal-e", signed_version=None, provenance="counterparty_paper"),
            _deal("deal-f", signed_version=3, provenance="our_paper"),
            _deal("deal-g", signed_version=3, provenance="our_paper"),
        ],
        perspective={"party": "Fixture Co", "counterparty_type": "Fixture Counterparty"},
    )
    clause_005 = doc_005["evidence"]["clauses"][0]
    assert (clause_005["n_deals"], clause_005["n_signed_standard"]) == (7, 2)
    assert clause_005["n_variants"] == 1  # deal-a/-c/-d share one signed text
    vectors.append(
        (
            "005-opening-evidence",
            _vector_v05(
                "opening-evidence",
                "Opening evidence (OPF-SPEC §3.5.5): opened_with and opening_text "
                "for seven deals of one clause. deal-a opened with our standard "
                "and signed an edit (opening_text = the standard); deal-b opened "
                "non-standard and signed our standard (opening_text recorded "
                "whatever its origin); deal-c opened non-standard and signed it "
                "unchanged (opened_with non_standard, opening_text null); deal-d "
                "added the clause in round 2 (opened_with absent, opening_text "
                "null); deal-e is unsigned (opened_with and opening_text null); "
                "deal-f's only edit is case (same grouping key, so opening_text "
                "null); deal-g opened non-standard and struck it before signing "
                "(opening_text set, signed_text null, no refused ask). The "
                "digest is unaffected: opening evidence is not projected into "
                "digest_version 3.",
                doc_005,
            ),
        )
    )
    return vectors


def main_v05() -> None:
    VECTORS_DIR_V05.mkdir(parents=True, exist_ok=True)
    vectors = build_vectors_v05()
    manifest_entries = []
    for filename, vector in vectors:
        path = VECTORS_DIR_V05 / f"{filename}.json"
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
            "opf_version": OPF_VERSION_V05,
            "engine_version": ENGINE_VERSION,
            "digest_version": DIGEST_VERSION_V3,
        },
        # The stamp text of the 0.4 manifest, carried over with the version
        # updated. "../manifest.json" was the retired 0.3 set's manifest;
        # spec/conformance/README.md carries the algorithm it described.
        "algorithm": (
            "canonical / content_hash / section_digests: exactly as the 0.3 set "
            "(../manifest.json). digest: playbook_engine.digest.build_digest(input) "
            "with the default token_budget, which for opf_version 0.5 is "
            "build_digest_v3 (OPF-SPEC.md §3.12.1): signed variants and refused asks "
            "grouped by the grouping key of OPF-SPEC.md §3.5.4 (precedent."
            "normalize_variant_text: every Counterparty-<n> alias -> 'counterparty', "
            "then deviation_classifier.normalize_for_standard with the document's "
            "perspective.party as the one party name -> 'party'), text = "
            "observation_builder.summarize_clause_text of the group representative."
        ),
        "vectors": manifest_entries,
    }
    (CONFORMANCE_DIR_V05 / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(vectors)} vectors + manifest.json to {CONFORMANCE_DIR_V05}")


if __name__ == "__main__":
    main_v05()
