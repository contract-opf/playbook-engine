#!/usr/bin/env python3
"""Regenerate the frozen conformance vectors under ``spec/conformance/0.5/`` — issue #115.

DEV TOOL, NOT PART OF THE TEST SUITE OR THE RUNTIME PACKAGE. Run this only
when deliberately re-stamping the conformance vectors for a new format
version (a new ``opf_version`` or a new ``DIGEST_VERSION_V4``) — never as a
routine "regenerate the golden files" step. The whole point of
``spec/conformance/`` is that its ``expected.*`` values are FROZEN,
independently-computed-once numbers that ``tests/test_conformance_vectors.py``
checks the live engine against; overwriting them from a possibly-buggy
current engine on every run would turn the conformance suite into a tautology
that can never go red (exactly what the issue #115 reviewer gate calls
"self-consistency" and requires the suite NOT be).

Usage::

    .venv/bin/python scripts/generate_conformance_vectors.py

Writes the OPF 0.5 / digest_version 4 set (issues #223, #233, #234, #240) under
``spec/conformance/0.5/`` (``manifest.json`` + ``vectors/``); re-running it
must reproduce the committed files byte-for-byte. The OPF 0.3 / digest 2 set
and the 0.4 set (the same documents without opening evidence) were retired
with those formats (issues #238, #233) — git history has them. Review the resulting diff like any other spec change (it needs a
``spec/CHANGELOG.md`` entry) before committing.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from playbook_engine.canonicalize import (
    canonicalize,
    canonicalize_playbook,
    compute_section_digests,
    content_hash,
    sha256_hex,
)
from playbook_engine.digest import DIGEST_VERSION_V4, build_digest_v4
from playbook_engine.dossiers import build_derived_sections
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
        "digest": build_digest_v4(doc),
        # Issue #228: the hard-rule manifest, critic dossiers and provenance
        # index are pure functions of the document too.
        **build_derived_sections(doc),
    }


# ---------------------------------------------------------------------------
# OPF 0.5 / digest_version 4 set (issues #223, #233, #234, #240) — spec/conformance/0.5/
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
    floor: dict[str, Any] | None = None,
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
    doc["floor"] = floor or {}
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


def _vs(label: str | None) -> dict[str, Any] | None:
    """A ``vs_standard`` object (OPF-SPEC §3.5.6) carrying *label*, or ``None``."""
    if label is None:
        return None
    return {"label": label, "reason": "Synthetic fixture reason.", "basis": "agent", "check": None}


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
    signed_vs: str | None = None,
    opening_vs: str | None = None,
    refused_vs: str | None = None,
) -> dict[str, Any]:
    refused_asks = [
        {
            "text": text,
            "round": version - 1,
            "ref": _ref(document_id, version),
            **({"vs_standard": _vs(refused_vs)} if refused_vs is not None else {}),
        }
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
                {
                    "text": signed_text,
                    "ref": _ref(document_id, 3),
                    **({"vs_standard": _vs(signed_vs)} if signed_vs is not None else {}),
                }
                if signed_text is not None
                else None
            ),
            "opened_with": opened_with,
            "opening_text": (
                {
                    "text": opening_text,
                    "ref": _ref(document_id, 1),
                    **({"vs_standard": _vs(opening_vs)} if opening_vs is not None else {}),
                }
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
        "digest_version": DIGEST_VERSION_V4,
        "input": doc,
        "expected": _expected(doc),
    }


_STANDARD_TEXT = (
    "Neither party may assign this Agreement without the other party's prior written consent."
)


def build_vectors_v05() -> list[tuple[str, dict[str, Any]]]:
    """The OPF 0.5 / digest 4 vector set. Synthetic inputs only."""
    vectors: list[tuple[str, dict[str, Any]]] = []

    # 001 — smallest 0.5 document: no clauses, no perspective.
    doc_001 = _base_v05()
    digest_001 = build_digest_v4(doc_001)
    assert digest_001["perspective"] is None
    assert digest_001["clauses"] == []
    vectors.append(
        (
            "001-minimal-no-perspective",
            _vector_v05(
                "minimal-no-perspective",
                "Smallest well-formed OPF 0.5 document: evidence {clauses: [], "
                "precedent: []}, no top-level perspective. expected.digest pins "
                "the digest_version 4 skeleton — perspective is PRESENT and null "
                "(never omitted), agreement_type is {id, name}, corpus counts are "
                "0 and first_signed/last_signed are null, uncovered_clause_types is an "
                "empty list.",
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
    entry_002 = build_digest_v4(doc_002)["clauses"][0]
    assert [v["n_deals"] for v in entry_002["signed_variants"]] == [3, 1]
    assert entry_002["signed_variants"][0]["last_signed"] == "2025-Q4"
    assert entry_002["refused_asks"][0]["n_deals"] == 3
    assert entry_002["n_signed_standard"] == 2 and entry_002["n_deals"] == 8
    vectors.append(
        (
            "002-variants-refused-and-exclusions",
            _vector_v05(
                "variants-refused-and-exclusions",
                "One clause, eight deals, pinning digest_version 4 grouping and "
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
    entry_003 = build_digest_v4(doc_003)["clauses"][0]
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
    entry_004 = build_digest_v4(doc_004)["clauses"][0]
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
                "(opening_text set, signed_text null, no refused ask). The digest "
                "projects it as n_opened_standard 2 (deal-a, deal-f) / "
                "n_kept_standard 1 (deal-f), one signed variant (deal-a, -c, "
                "-d) carrying n_from_standard 1 and n_unchanged 1 (deal-d's "
                "absent clause counts in neither), and two changed openings "
                "(deal-b's, which ended at our standard, and deal-g's struck "
                "one).",
                doc_005,
            ),
        )
    )
    # 006 — the opening rules of digest 4 (issue #234): every branch of
    # n_opened_standard / n_kept_standard, n_from_standard / n_unchanged and
    # changed_openings on one clause.
    tid_006 = "venue"
    std_006 = "The courts of the State of Delaware have exclusive jurisdiction."
    v1_006 = "The courts of the State of New York have exclusive jurisdiction."
    v2_006 = "The courts of the State of Texas have exclusive jurisdiction."
    v3_006 = "The parties will arbitrate every dispute in London."
    n1_006 = "Disputes will be heard by the courts of the State of Oregon."
    n2_006 = "Disputes will be heard by the courts of the State of Nevada."
    n3_006 = "Disputes will be heard by the courts of the State of Ohio."
    precedent_006 = [
        # Edited standard: opened with our standard, signed a variant.
        _prec("deal-a", tid_006, signed_text=v1_006, opening_text=std_006, rounds=1),
        # Struck standard: opened with our standard, nothing signed.
        _prec("deal-b", tid_006, signed_text=None, opening_text=std_006),
        # Non-standard changed to our standard.
        _prec(
            "deal-c",
            tid_006,
            signed_text=std_006,
            standard=True,
            opening_text=n1_006,
            opened_with="non_standard",
            rounds=1,
            paper="theirs",
        ),
        # Non-standard, signed unchanged.
        _prec("deal-d", tid_006, signed_text=v2_006, opened_with="non_standard", paper="theirs"),
        # Non-standard, struck, and the same words are one of this deal's own
        # refused asks: shown as a refused ask, excluded from changed_openings.
        _prec(
            "deal-e",
            tid_006,
            signed_text=None,
            opening_text=n2_006,
            opened_with="non_standard",
            refused=[(n2_006.upper(), 2)],
            paper="theirs",
        ),
        # Non-standard, struck, no refused ask: a changed opening (n_struck 1).
        _prec(
            "deal-f",
            tid_006,
            signed_text=None,
            opening_text=n3_006,
            opened_with="non_standard",
            paper="theirs",
        ),
        # Clause added in round 2: opened with neither.
        _prec("deal-g", tid_006, signed_text=v1_006, opened_with="absent", rounds=1),
        # Unsigned deal: ignored by every opening count.
        _prec("deal-h", tid_006, signed_text=v3_006, signed=False, paper="unknown"),
        # Opened with our standard and kept it.
        _prec("deal-i", tid_006, signed_text=std_006, standard=True),
        # A second deal opening with deal-c's words (respelled): one changed
        # opening with n_deals 2, n_to_standard 1.
        _prec(
            "deal-j",
            tid_006,
            signed_text=v3_006,
            opening_text=n1_006.upper(),
            opened_with="non_standard",
            rounds=1,
            paper="theirs",
        ),
    ]
    doc_006 = _base_v05(
        clauses=[
            {
                "id": f"clause.{tid_006}",
                "taxonomy_id": tid_006,
                "title": "Venue",
                "our_standard": {
                    "text": std_006,
                    "source_ref": {
                        "document_id": "template",
                        "version": "template",
                        "clause_path": "15",
                    },
                },
            }
        ],
        precedent=precedent_006,
        documents=[
            _deal("deal-a", signed_version=3, provenance="our_paper"),
            _deal("deal-b", signed_version=3, provenance="our_paper"),
            _deal("deal-c", signed_version=3, provenance="counterparty_paper"),
            _deal("deal-d", signed_version=3, provenance="counterparty_paper"),
            _deal("deal-e", signed_version=3, provenance="counterparty_paper"),
            _deal("deal-f", signed_version=3, provenance="counterparty_paper"),
            _deal("deal-g", signed_version=3, provenance="our_paper"),
            _deal("deal-h", signed_version=None, provenance="counterparty_paper"),
            _deal("deal-i", signed_version=3, provenance="our_paper"),
            _deal("deal-j", signed_version=3, provenance="counterparty_paper"),
        ],
        perspective={"party": "Fixture Co", "counterparty_type": "Fixture Counterparty"},
    )
    entry_006 = build_digest_v4(doc_006)["clauses"][0]
    assert (entry_006["n_opened_standard"], entry_006["n_kept_standard"]) == (3, 1)
    assert [e["n_deals"] for e in entry_006["changed_openings"]] == [2, 1]
    assert [e["n_struck"] for e in entry_006["changed_openings"]] == [0, 1]
    assert entry_006["n_changed_openings_total"] == 2
    assert len(entry_006["refused_asks"]) == 1
    vectors.append(
        (
            "006-opening-rules",
            _vector_v05(
                "opening-rules",
                "The digest 4 opening rules (OPF-SPEC §3.12.2) on one clause with "
                "ten deals. n_opened_standard counts signed deals whose opened_with "
                "is standard (deal-a's edited standard, deal-b's struck standard, "
                "deal-i's kept one) and n_kept_standard those that signed our "
                "standard (deal-i only). Variants carry n_from_standard (deal-a "
                "conceded the New York variant from our standard) and n_unchanged "
                "(deal-d signed its non-standard opening as proposed); deal-g's "
                "'absent' clause counts in neither. changed_openings: deal-c's "
                "non-standard opening that ended at our standard and deal-j's "
                "respelling of it are ONE entry (n_deals 2, n_to_standard 1); "
                "deal-f's struck opening with no refused ask is included "
                "(n_struck 1); deal-e's struck opening is also one of its own "
                "refused asks, so it is shown as a refused ask only; deal-h is "
                "unsigned and ignored.",
                doc_006,
            ),
        )
    )

    # 007 — the vs_standard label in the digest (issue #240) and
    # uncovered_clause_types (issue #234).
    tid_007 = "assignment"
    texts_007 = {
        "eq1": "Either party may assign this Agreement to an affiliate.",
        "eq2": "An affiliate assignment by either party is permitted.",
        "less": "Either party may assign this Agreement to anyone at any time.",
        "diff": "Assignment requires a payment of one million dollars.",
        "more": "Neither party may assign this Agreement even with consent.",
        "none": "Assignment is governed by the parties' later written agreement.",
    }
    precedent_007 = [
        _prec("deal-a", tid_007, signed_text=_STANDARD_TEXT, standard=True),
        _prec("deal-b", tid_007, signed_text=texts_007["eq1"], signed_vs="equivalent"),
        _prec("deal-c", tid_007, signed_text=texts_007["eq1"].upper(), signed_vs="equivalent"),
        _prec(
            "deal-d",
            tid_007,
            signed_text=texts_007["eq2"],
            signed_vs="equivalent",
            opening_text=_STANDARD_TEXT,
            rounds=1,
        ),
        _prec(
            "deal-e",
            tid_007,
            signed_text=texts_007["less"],
            signed_vs="less_protective",
            refused=[(texts_007["more"], 2)],
            refused_vs="more_protective",
            paper="theirs",
        ),
        _prec(
            "deal-f",
            tid_007,
            signed_text=texts_007["less"],
            signed_vs="less_protective",
            paper="theirs",
        ),
        _prec(
            "deal-g",
            tid_007,
            signed_text=texts_007["diff"],
            signed_vs="different_concept",
            opening_text=texts_007["more"],
            opening_vs="more_protective",
            opened_with="non_standard",
            rounds=1,
            paper="theirs",
        ),
        _prec("deal-h", tid_007, signed_text=texts_007["more"], signed_vs="more_protective"),
        _prec("deal-i", tid_007, signed_text=texts_007["none"]),
    ]
    doc_007 = _base_v05(
        clauses=[
            {
                "id": f"clause.{tid_007}",
                "taxonomy_id": tid_007,
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
        precedent=precedent_007,
        documents=[
            _deal(
                f"deal-{c}",
                signed_version=3,
                provenance="counterparty_paper" if c in "efg" else "our_paper",
            )
            for c in "abcdefghi"
        ],
        perspective={"party": "Fixture Co", "counterparty_type": "Fixture Counterparty"},
    )
    doc_007["taxonomy"] = {
        "source": "custom",
        "entries": [
            {"id": tid_007, "label": "Assignment", "status": "active"},
            {"id": "term", "label": "Term", "status": "active"},
            {"id": "audit_rights", "label": "Audit Rights", "status": "custom"},
            {"id": "retired_clause", "label": "Retired Clause", "status": "inactive"},
        ],
    }
    digest_007 = build_digest_v4(doc_007)
    entry_007 = digest_007["clauses"][0]
    assert [v["label"] for v in entry_007["signed_variants"]] == [
        "less_protective",
        "different_concept",
        None,
        "more_protective",
        "equivalent",
    ]
    assert entry_007["signed_variants"][-1]["n_deals"] == 3
    assert [u["taxonomy_id"] for u in digest_007["uncovered_clause_types"]] == [
        "audit_rights",
        "term",
    ]
    vectors.append(
        (
            "007-equivalence-label-and-coverage",
            _vector_v05(
                "equivalence-label-and-coverage",
                "The vs_standard label in the digest (OPF-SPEC §3.5.6, §3.12.2) and "
                "uncovered_clause_types. Signed variants labelled equivalent (three "
                "deals, two distinct texts; the respelled text merges under the "
                "grouping key) collapse into ONE entry with n_deals 3, n_texts 2, "
                "two exemplars and the concession count; the rest are listed "
                "individually in tier order: less_protective (two deals), "
                "different_concept, the unjudged one (a signed text with no "
                "label), more_protective, then the collapsed entry. "
                "positions counts signed deals by standard, each label and "
                "unjudged. The refused ask and the changed opening carry their "
                "labels. The taxonomy lists one covered entry, an active and a "
                "custom entry with no evidence (uncovered_clause_types, sorted by "
                "taxonomy_id) and an inactive entry (never listed).",
                doc_007,
            ),
        )
    )
    # 008 — the hard-rule manifest and the critic dossiers (issue #228).
    tid_a, tid_b, tid_c, tid_d = "venue", "term", "assignment", "survival"
    std_d = "Sections on confidentiality and payment survive the end of this Agreement."
    std_a = "The courts of the State of Delaware have exclusive jurisdiction."
    v1_a = "The courts of the State of New York have exclusive jurisdiction."
    v2_a = "The courts of the State of Texas have exclusive jurisdiction."
    n1_a = "Disputes will be heard by the courts of the State of Oregon."
    r1_a = "Each party waives its right to a trial by jury."
    r2_a = "Venue lies exclusively in the courts of Mars."
    long_b = " ".join(
        f"Section {i} of the term provisions binds each party for the full period."
        for i in range(80)
    )
    # Two long survival rules: with all three listed the survival dossier (no
    # excerpt) is over its budget, and dropping the last listed rule brings it
    # within it.
    long_d = [
        " ".join(
            f"Survival {name} {i}: this obligation outlasts the Agreement for its full period."
            for i in range(24)
        )
        for name in ("scope", "period")
    ]
    std_c = _STANDARD_TEXT
    v1_c = "Either party may assign this Agreement to an affiliate without consent."
    r1_c = "Either party may assign this Agreement freely without consent."
    # One signing date per deal, on every record of it (a producer that records them).
    signed_at_008 = {"deal-c": "2025-01-15", "deal-d": "2025-06-30", "deal-f": "2025-06-30"}
    precedent_008 = [
        _prec("deal-a", tid_a, signed_text=v1_a, opening_text=std_a, rounds=1),
        _prec("deal-b", tid_a, signed_text=v2_a, opened_with="non_standard", paper="theirs"),
        _prec("deal-c", tid_a, signed_text=v2_a, opened_with="non_standard", paper="theirs"),
        _prec(
            "deal-d",
            tid_a,
            signed_text=std_a,
            standard=True,
            opening_text=n1_a,
            opened_with="non_standard",
            rounds=1,
            paper="theirs",
        ),
        _prec("deal-e", tid_a, signed_text=std_a, standard=True, refused=[(r1_a, 2)], rounds=1),
        _prec("deal-f", tid_a, signed_text=std_a, standard=True, refused=[(r2_a, 2)], rounds=1),
        _prec("deal-a", tid_b, signed_text=long_b),
        # deal-e: no excerpt any dossier keeps cites it, so the provenance index
        # lists it only for the term dossier's dropped (omitted) excerpt.
        _prec("deal-e", tid_b, signed_text="The term of this Agreement is three years."),
        # Assignment: deal-a signed the affiliate variant as proposed and
        # deal-b conceded to it from our standard; three deals refused the
        # same ask and kept our standard.
        _prec("deal-a", tid_c, signed_text=v1_c, opened_with="non_standard"),
        _prec("deal-b", tid_c, signed_text=v1_c, opening_text=std_c, rounds=1, paper="theirs"),
        *(
            _prec(
                deal,
                tid_c,
                signed_text=std_c,
                standard=True,
                refused=[(r1_c, 2)],
                rounds=1,
                paper=paper,
            )
            for deal, paper in (("deal-c", "theirs"), ("deal-d", "theirs"), ("deal-f", "ours"))
        ),
    ]
    for record in precedent_008:
        if record["document_id"] in signed_at_008:
            record["signed_at"] = signed_at_008[record["document_id"]]
    floor_008 = {
        "invariants": [
            {
                "id": "venue-holds",
                "statement": "Venue must stay in Delaware or New York.",
                "rationale": "Our litigation counsel sits in those two states.",
                "x_signed_by": "Legal Owner",
                "x_signed_at": "2026-01-01T00:00:00+00:00",
                "x_taxonomy_id": tid_a,
                "x_required_presence": True,
                "x_condition": {
                    "type": "required_phrases",
                    "phrases": ["State of Delaware", "State of New York"],
                    "match": "any",
                },
                "x_permissible_proof": ["a venue waiver signed by the General Counsel"],
            },
            {
                "id": "venue-no-arbitration",
                "statement": "Do not agree to binding arbitration of venue disputes.",
                "x_signed_by": "Legal Owner",
                "x_signed_at": "2026-01-01T00:00:00+00:00",
                "x_taxonomy_id": tid_a,
            },
            {
                "id": "term-cap",
                "statement": "The term must not exceed five years.",
                "x_signed_by": "Legal Owner",
                "x_signed_at": "2026-01-01T00:00:00+00:00",
                "x_taxonomy_id": tid_b,
                "x_condition": {
                    "type": "numeric_bound",
                    "pattern": "([0-9]+) years",
                    "max": 5,
                    "unit": "years",
                },
            },
            {
                "id": "survival-reference",
                "statement": "Survival must refer back to the term clause.",
                "x_signed_by": "Legal Owner",
                "x_signed_at": "2026-01-01T00:00:00+00:00",
                "x_taxonomy_id": tid_d,
                "x_required_presence": True,
                "x_condition": {"type": "cross_reference", "clause_id": f"clause.{tid_b}"},
            },
            *(
                {
                    "id": f"survival-{name}",
                    "statement": statement,
                    "x_signed_by": "Legal Owner",
                    "x_signed_at": "2026-01-01T00:00:00+00:00",
                    "x_taxonomy_id": tid_d,
                }
                for name, statement in zip(("scope", "period"), long_d, strict=True)
            ),
            {
                "id": "no-ghost-clause",
                "statement": "Nothing here about a clause the corpus never saw.",
                "x_signed_by": "Legal Owner",
                "x_signed_at": "2026-01-01T00:00:00+00:00",
                "x_taxonomy_id": "ghost_clause",
            },
        ]
    }
    doc_008 = _base_v05(
        clauses=[
            {
                "id": f"clause.{tid_a}",
                "taxonomy_id": tid_a,
                "title": "Venue",
                "our_standard": {
                    "text": std_a,
                    "source_ref": {
                        "document_id": "template",
                        "version": "template",
                        "clause_path": "15",
                    },
                },
            },
            {"id": f"clause.{tid_b}", "taxonomy_id": tid_b, "title": "Term", "our_standard": None},
            {
                "id": f"clause.{tid_c}",
                "taxonomy_id": tid_c,
                "title": "Assignment",
                "our_standard": {
                    "text": std_c,
                    "source_ref": {
                        "document_id": "template",
                        "version": "template",
                        "clause_path": "12",
                    },
                },
            },
            {
                "id": f"clause.{tid_d}",
                "taxonomy_id": tid_d,
                "title": "Survival",
                "our_standard": {
                    "text": std_d,
                    "source_ref": {
                        "document_id": "template",
                        "version": "template",
                        "clause_path": "20",
                    },
                },
            },
        ],
        precedent=precedent_008,
        documents=[
            _deal("deal-a", signed_version=3, provenance="our_paper"),
            _deal("deal-b", signed_version=3, provenance="counterparty_paper"),
            _deal("deal-c", signed_version=3, provenance="counterparty_paper"),
            _deal("deal-d", signed_version=3, provenance="counterparty_paper"),
            _deal("deal-e", signed_version=3, provenance="our_paper"),
            _deal("deal-f", signed_version=3, provenance="our_paper"),
        ],
        perspective={"party": "Fixture Co", "counterparty_type": "Fixture Counterparty"},
        floor=floor_008,
    )
    # Source-file hashes for every deal but deal-f (which recorded none), and
    # the corpus snapshot hash over them (OPF-SPEC §3.8: the canonical JSON of
    # the sorted (document_id, version, sha256) triples).
    for corpus_doc in doc_008["corpus"]["documents"]:
        if corpus_doc["document_id"] != "deal-f":
            corpus_doc["version_files"] = [
                {
                    "version": v,
                    "sha256": sha256_hex(f"{corpus_doc['document_id']} version {v}"),
                }
                for v in (1, 2, 3)
            ]
    triples = sorted(
        (d["document_id"], vf["version"], vf["sha256"])
        for d in doc_008["corpus"]["documents"]
        for vf in d.get("version_files", [])
    )
    doc_008["corpus"]["snapshot"] = {
        "manifest_hash": sha256_hex(canonicalize([list(t) for t in triples]))
    }
    derived_008 = build_derived_sections(doc_008)
    assert [r["rule_id"] for r in derived_008["manifest"]["hard_rules"]] == [
        i["id"] for i in floor_008["invariants"]
    ]
    venue_008 = derived_008["dossiers"][f"clause.{tid_a}"]
    assert [e["kind"] for e in venue_008["excerpts"]] == ["signed_variant", "changed_opening"]
    assert venue_008["excerpts"][0]["opening"] == std_a  # the concession, not the Texas majority
    assert venue_008["n_floor_rules"] == 2 and venue_008["n_omitted"] == 0
    term_008 = derived_008["dossiers"][f"clause.{tid_b}"]
    # Term: the long signed text is the first excerpt, kept WHOLE although it alone
    # exceeds the 1,000-token budget (no our standard); the second is dropped whole
    # and named, then the term-cap rule is dropped whole too (still over budget).
    # Nothing is cut part-way.
    ids_b = {p["document_id"]: p["id"] for p in precedent_008 if p["taxonomy_id"] == tid_b}
    assert [e["signed"] for e in term_008["excerpts"]] == [long_b]
    assert term_008["omitted_precedent_ids"] == [ids_b["deal-e"]] and term_008["n_omitted"] == 1
    assert term_008["n_floor_rules"] == 1 and term_008["floor_rules"] == []
    assert len(canonicalize(term_008)) // 4 > 1_000
    # Survival: no excerpt; three Floor rules are over the budget, so the last
    # listed one is dropped whole and the dossier fits.
    survival_008 = derived_008["dossiers"][f"clause.{tid_d}"]
    assert survival_008["excerpts"] == [] and survival_008["n_floor_rules"] == 3
    assert [r["rule_id"] for r in survival_008["floor_rules"]] == [
        "survival-reference",
        "survival-scope",
    ]
    assert survival_008["floor_rules"][1]["statement"] == long_d[0]
    with_all = copy.deepcopy(survival_008)
    with_all["floor_rules"].append({"rule_id": "survival-period", "statement": long_d[1]})
    assert len(canonicalize(survival_008)) // 4 <= 1_000 < len(canonicalize(with_all)) // 4
    rules_008 = {r["rule_id"]: r for r in derived_008["manifest"]["hard_rules"]}
    assert rules_008["survival-reference"]["clause_id"] == f"clause.{tid_d}"
    assert rules_008["survival-reference"]["fallback_language"] == std_d
    # Assignment: the concession's own record although deal-a is the lower
    # document and precedent id, then the refused ask from the latest-signed
    # deals' lower precedent id (deal-f, although deal-d is the lower document).
    ids_c = {p["document_id"]: p["id"] for p in precedent_008 if p["taxonomy_id"] == tid_c}
    assert ids_c["deal-a"] < ids_c["deal-b"]
    assert ids_c["deal-f"] < ids_c["deal-d"] and min(ids_c.values()) == ids_c["deal-c"]
    assignment_008 = derived_008["dossiers"][f"clause.{tid_c}"]
    assert assignment_008["excerpts"] == [
        {
            "precedent_id": ids_c["deal-b"],
            "kind": "signed_variant",
            "opening": std_c,
            "signed": v1_c,
            "outcome": "signed",
        },
        {
            "precedent_id": ids_c["deal-f"],
            "kind": "refused_ask",
            "opening": r1_c,
            "signed": std_c,
            "outcome": "ask_refused",
        },
    ]
    index_008 = derived_008["provenance_index"]
    assert index_008["corpus_manifest_hash"] == doc_008["corpus"]["snapshot"]["manifest_hash"]
    files_008 = {d["document_id"]: d["version_files"] for d in index_008["documents"]}
    assert files_008["deal-f"] == [] and len(files_008["deal-b"]) == 3
    # The term dossier's dropped excerpt resolves through the index to deal-e,
    # which only that omitted row brings into documents.
    assert index_008["dossiers"][f"clause.{tid_b}"] == [
        {
            "precedent_id": term_008["excerpts"][0]["precedent_id"],
            "document_id": "deal-a",
            "kind": "signed_variant",
            "omitted": False,
        },
        {
            "precedent_id": ids_b["deal-e"],
            "document_id": "deal-e",
            "kind": "signed_variant",
            "omitted": True,
        },
    ]
    assert "deal-e" in files_008
    assert not any(
        row["document_id"] == "deal-e" and not row["omitted"]
        for rows in index_008["dossiers"].values()
        for row in rows
    )
    vectors.append(
        (
            "008-hard-rules-and-dossiers",
            _vector_v05(
                "hard-rules-and-dossiers",
                "The hard-rule manifest, the critic dossiers and the provenance index "
                "(OPF-SPEC §3.12.3). The Floor has seven invariants and the manifest seven "
                "rules in Floor order: venue-holds names clause.venue and carries a "
                "required_phrases condition, required_presence true and a permissible "
                "proof, with fallback_language the clause's standard; venue-no-arbitration "
                "states none of them (required_presence false, condition judged, no "
                "proof); term-cap names clause.term, which has no standard "
                "(fallback_language null), with a numeric_bound condition; "
                "survival-reference names clause.survival (a rule that demands presence or "
                "carries a predicate must name its clause) and carries a cross_reference "
                "condition with required_presence true; survival-scope and survival-period "
                "are two long judged rules on clause.survival; no-ghost-clause names a "
                "clause with no evidence (clause_id null). The venue dossier selects two "
                "excerpts from five candidate "
                "groups: first the concession on record (deal-a opened with our standard "
                "and signed the New York variant, although the Texas variant has more "
                "deals), then the first changed opening (deal-d's Oregon opening, which "
                "ended at our standard); its refused asks are not reached. It lists both "
                "Floor rules. No text is ever cut part-way. The term dossier "
                "(no standard, so a 1,000-token budget) selects two excerpts; its first, "
                "a long signed text, alone exceeds the budget and is kept whole; the "
                "second (deal-e's) is dropped whole and named (n_omitted 1, "
                "omitted_precedent_ids), then its Floor rule term-cap is dropped whole "
                "(n_floor_rules 1, none listed), since it is still over budget: a dossier "
                "holding its single kept excerpt is the only one that may exceed its budget. "
                "The survival dossier has a standard, no excerpt and three Floor rules that "
                "together exceed its budget, so the last listed (survival-period) is dropped "
                "whole and it fits (n_floor_rules 3, two listed). The assignment "
                "dossier pins how a group's record is chosen and a refused-ask excerpt: "
                "first the concession, deal-b's own opening-to-signed pair, although "
                "deal-a, which signed the same variant as proposed, is the lower document "
                "and precedent id (step 1 takes only members that opened with our "
                "standard); then, with no changed opening and the variant's group already "
                "chosen, the refused ask made in deal-c, deal-d and deal-f: deal-d and "
                "deal-f share the latest signed_at and deal-f has the lower precedent_id "
                "(deal-c has the lowest, but signed earlier; deal-d is the lower "
                "document), so the excerpt is deal-f's ask with what it signed instead "
                "(outcome ask_refused). Every deal but deal-f records source-file hashes "
                "and the corpus carries their snapshot hash: the provenance index repeats "
                "the hash and lists each excerpt deal's version_files (empty for deal-f). "
                "Its per-clause rows are in selection order, each marked omitted false or "
                "true: the term dossier's dropped excerpt is a row with omitted true that "
                "resolves to deal-e, which no kept excerpt cites, and deal-e is listed in "
                "documents for it.",
                doc_008,
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
            "digest_version": DIGEST_VERSION_V4,
        },
        # The stamp text of the 0.4 manifest, carried over with the version
        # updated. "../manifest.json" was the retired 0.3 set's manifest;
        # spec/conformance/README.md carries the algorithm it described.
        "algorithm": (
            "canonical / content_hash / section_digests: exactly as the 0.3 set "
            "(../manifest.json). digest: playbook_engine.digest.build_digest_v4(input) "
            "with the default token_budget, which for opf_version 0.5 is "
            "build_digest_v4 (OPF-SPEC.md §3.12.2): signed variants, refused asks and "
            "changed openings grouped by the grouping key of OPF-SPEC.md §3.5.4 (precedent."
            "normalize_variant_text: every Counterparty-<n> alias -> 'counterparty', "
            "then deviation_classifier.normalize_for_standard with the document's "
            "perspective.party as the one party name -> 'party'), text = "
            "observation_builder.summarize_clause_text of the group representative; the "
            "signed variants labelled equivalent (vs_standard) collapse into one entry "
            "and the cap applies after collapsing. manifest / dossiers / provenance_index: "
            "playbook_engine.dossiers.build_derived_sections(input) (OPF-SPEC.md §3.12.3)."
        ),
        "vectors": manifest_entries,
    }
    (CONFORMANCE_DIR_V05 / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(vectors)} vectors + manifest.json to {CONFORMANCE_DIR_V05}")


if __name__ == "__main__":
    main_v05()
