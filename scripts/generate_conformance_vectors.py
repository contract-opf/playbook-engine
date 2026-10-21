#!/usr/bin/env python3
"""Regenerate the frozen conformance vectors under ``spec/conformance/0.6/`` — issue #115.

DEV TOOL, NOT PART OF THE TEST SUITE OR THE RUNTIME PACKAGE. Run this only
when deliberately re-stamping the conformance vectors for a new format
version (a new ``opf_version`` or a new ``DIGEST_VERSION_V5``) — never as a
routine "regenerate the golden files" step. The whole point of
``spec/conformance/`` is that its ``expected.*`` values are FROZEN,
independently-computed-once numbers that ``tests/test_conformance_vectors.py``
checks the live engine against; overwriting them from a possibly-buggy
current engine on every run would turn the conformance suite into a tautology
that can never go red (exactly what the issue #115 reviewer gate calls
"self-consistency" and requires the suite NOT be).

Usage::

    .venv/bin/python scripts/generate_conformance_vectors.py

Writes the OPF 0.6 / digest_version 5 set (issues #223, #233, #234, #240) under
``spec/conformance/0.6/`` (``manifest.json`` + ``vectors/``); re-running it
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

from playbook_engine.absence import build_absence
from playbook_engine.canonicalize import (
    canonicalize,
    canonicalize_playbook,
    compute_section_digests,
    content_hash,
    sha256_hex,
)
from playbook_engine.digest import DIGEST_VERSION_V5, build_digest_v5
from playbook_engine.dossiers import build_derived_sections
from playbook_engine.equivalence import iter_slots, stamp_standard_parts, standard_parts
from playbook_engine.observation_builder import Observation, ObservationCitation
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
        "digest": build_digest_v5(doc),
        # Issue #228: the hard-rule manifest, critic dossiers and provenance
        # index are pure functions of the document too.
        **build_derived_sections(doc),
    }


# ---------------------------------------------------------------------------
# OPF 0.6 / digest_version 5 set (issues #223, #233, #234, #240) — spec/conformance/0.6/
# ---------------------------------------------------------------------------

OPF_VERSION_V06 = "0.6"
CONFORMANCE_DIR_V06 = CONFORMANCE_DIR / "0.6"
VECTORS_DIR_V06 = CONFORMANCE_DIR_V06 / "vectors"
_AGREEMENT_TYPE_ID = "conformance-fixture"


def _base_v06(
    *,
    clauses: list[dict[str, Any]] | None = None,
    precedent: list[dict[str, Any]] | None = None,
    documents: list[dict[str, Any]] | None = None,
    perspective: dict[str, str] | None = None,
    floor: dict[str, Any] | None = None,
    prior_templates: list[dict[str, Any]] | None = None,
    absence_nodes: list[Observation] | None = None,
) -> dict[str, Any]:
    """A minimal OPF 0.6 document. Clause counts and precedent ids are
    stamped by the reference functions (``precedent.clause_counts`` /
    ``precedent.precedent_id``) so every input is self-consistent — a vector
    a validator would reject is not a useful conformance input."""
    doc: dict[str, Any] = {
        "opf_version": OPF_VERSION_V06,
        "agreement_type": {"id": _AGREEMENT_TYPE_ID, "name": "Conformance Fixture Agreement"},
        "baseline": {"has_canonical_template": True},
        "taxonomy": {"source": "custom", "entries": []},
    }
    if prior_templates is not None:
        doc["baseline"]["prior_templates"] = prior_templates
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
        clause = _with_parts(clause)
        stamp_standard_parts(clause.get("our_standard"), party)
        for prior in clause.get("prior_standards", []):
            stamp_standard_parts(prior, party)
        stamped.append(
            {
                **{k: clause[k] for k in ("id", "taxonomy_id", "title", "our_standard")},
                "prior_standards": clause.get("prior_standards", []),
                "why_changed": clause.get("why_changed"),
                **clause_counts(clause["taxonomy_id"], records, party=party),
            }
        )
    doc["evidence"] = {"clauses": stamped, "precedent": records}
    _stamp_judgment_parts(doc["evidence"])
    doc["posture"] = {}
    doc["floor"] = floor or {}
    dates_by_doc: dict[str, set[str]] = {}
    for record in records:
        if record.get("signed_on") is not None:
            dates_by_doc.setdefault(record["document_id"], set()).add(record["signed_on"])
    corpus_documents = []
    for deal in documents or []:
        deal = dict(deal)
        dates = dates_by_doc.get(deal["document_id"], set())
        assert len(dates) <= 1, (deal["document_id"], dates)
        if "signed_on" not in deal:
            deal["signed_on"] = next(iter(dates)) if dates else None
            deal["signed_on_basis"] = "document_text" if dates else None
        assert not dates or dates == {deal["signed_on"]}, (deal, dates)
        corpus_documents.append(deal)
    doc["corpus"] = {"documents": corpus_documents, "stats": {}}
    doc["evidence"]["absence"] = _absence_section(doc, absence_nodes or [])
    doc["compiler"] = {
        "name": "playbook-engine",
        "version": ENGINE_VERSION,
        "run_id": "conformance-fixture-run",
        "generated_at": "2026-01-01T00:00:00Z",
    }
    return doc


def _node(document_id: str, clause_path: str, taxonomy_id: str | None, text: str) -> Observation:
    """A signed-text node of a fixture deal's executed copy (version 3): the row the
    reference compile would hold for it. *taxonomy_id* is the clause type the
    classifier filed it under, ``None`` when it left it unclassified."""
    return Observation(
        observation_id=f"{document_id}/3/{clause_path}",
        taxonomy_id=taxonomy_id,
        text_summary=text[:100],
        full_text=text,
        citation=ObservationCitation(
            document_id=document_id,
            version=3,
            clause_path=clause_path,
            char_span=(0, len(text)),
        ),
        deviation="none",
        risk_delta={"direction": "neutral", "magnitude": "none"},
        provenance="our_paper",
        outcome="signed",
    )


def _absence_section(doc: dict[str, Any], extra_nodes: list[Observation]) -> dict[str, Any]:
    """``evidence.absence`` of a fixture (OPF-SPEC §3.5.8), written by the reference
    ``absence.build_absence`` over the deals' signed-text nodes, never by hand: the
    validator cannot see ``n_checked`` or ``filed_under``, so a hand-written record
    could state what no compile can produce.

    The nodes are one per signed precedent record (filed under its clause type, at
    clause path ``p<n>``) plus *extra_nodes*: the passages of a deal's signed copy that
    are not precedent (unclassified ones, ones filed under another type, ones too short
    to be precedent). A deal with none has nothing to check."""
    evidence = doc["evidence"]
    nodes = [
        _node(p["document_id"], f"p{i}", p["taxonomy_id"], p["signed_text"]["text"])
        for i, p in enumerate(evidence["precedent"])
        if p["signed"] and p["signed_text"]
    ] + extra_nodes
    return build_absence(
        agreement_type_id=_AGREEMENT_TYPE_ID,
        clauses=evidence["clauses"],
        precedent=evidence["precedent"],
        observations=nodes,
        corpus_documents=doc["corpus"]["documents"],
    )


def _ref(document_id: str, version: int, clause_path: str = "1") -> dict[str, Any]:
    return {
        "document_id": document_id,
        "version": version,
        "clause_path": clause_path,
        "char_span": [0, 40],
    }


def _deal(
    document_id: str,
    *,
    signed_version: int | None,
    provenance: str,
    signed_on: str | None = None,
    basis: str | None = None,
) -> dict[str, Any]:
    """A corpus document. ``signed_on`` / ``basis`` are left out (and derived by
    ``_base_v06`` from the deal's precedent records, basis ``document_text``)
    unless given."""
    doc: dict[str, Any] = {
        "document_id": document_id,
        "provenance": provenance,
        "in_scope": True,
        "versions": 3,
        "signed_version": signed_version,
    }
    if signed_on is not None:
        doc["signed_on"] = signed_on
        doc["signed_on_basis"] = basis or "document_text"
    return doc


#: Sentinel: ``_prec`` derives ``opened_with`` from the other arguments.
_DEFAULT: object = object()


#: Stands for the one part of a one-part fixture standard until ``_base_v06``
#: stamps the real ``part_id``; ``_part(n)`` stands for the n-th part of a
#: standard of several (vector 009).
_PART = "__part__"


def _part(n: int) -> str:
    return f"{_PART}{n}"


#: A part-by-part judgment (OPF-SPEC §3.5.6, issue #243) that DERIVES each
#: label: ``(parts, additions)`` over a one-part standard.
_JUDGMENTS: dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]]]] = {
    "equivalent": ([{"part_id": _PART, "status": "present", "relation": "equivalent"}], []),
    "more_protective": (
        [
            {
                "part_id": _PART,
                "status": "present",
                "relation": "more_protective",
                "reason": "Synthetic fixture part reason.",
            }
        ],
        [],
    ),
    "less_protective": (
        [
            {
                "part_id": _PART,
                "status": "present",
                "relation": "less_protective",
                "reason": "Synthetic fixture part reason.",
            }
        ],
        [],
    ),
    "mixed": (
        [
            {
                "part_id": _PART,
                "status": "present",
                "relation": "less_protective",
                "reason": "Synthetic fixture part reason.",
            }
        ],
        [{"summary": "Synthetic fixture addition.", "effect": "more_protective"}],
    ),
    "different_concept": (
        [{"part_id": _PART, "status": "omitted", "effect": "neutral"}],
        [{"summary": "Synthetic fixture addition.", "effect": "neutral"}],
    ),
}


def _vs(label: str | dict[str, Any] | None) -> dict[str, Any] | None:
    """A ``vs_standard`` judgment (OPF-SPEC §3.5.6) whose parts derive *label*, or ``None``.

    A dict is a judgment written out in full (vector 009's several-part
    standard) and is copied as given."""
    if label is None:
        return None
    if isinstance(label, dict):
        return copy.deepcopy(label)
    parts, additions = _JUDGMENTS[label]
    return {
        "label": label,
        "reason": "Synthetic fixture reason.",
        "parts": [dict(p) for p in parts],
        "additions": [dict(a) for a in additions],
        "basis": "agent",
        "check": None,
    }


def _with_parts(clause: dict[str, Any]) -> dict[str, Any]:
    """*clause* with its ``our_standard`` cut into one part (ids stamped later)."""
    std = clause.get("our_standard")
    if not isinstance(std, dict) or "parts" in std:
        return clause
    return {
        **clause,
        "our_standard": {
            **std,
            "parts": [{"part_id": "", "order": 1, "text": std["text"]}],
        },
    }


def _stamp_judgment_parts(evidence: dict[str, Any]) -> None:
    """Point every fixture judgment at its clause's standard parts: ``_PART`` at
    the one part of a one-part standard, ``_part(n)`` at the n-th part."""
    ids = {c["taxonomy_id"]: [p["part_id"] for p in standard_parts(c)] for c in evidence["clauses"]}
    for slot in iter_slots(evidence):
        vs = slot.entry.get("vs_standard")
        if not isinstance(vs, dict):
            continue
        pids = ids[slot.record["taxonomy_id"]]
        for part in vs["parts"]:
            if part["part_id"] == _PART:
                (part["part_id"],) = pids
            else:
                part["part_id"] = pids[int(part["part_id"][len(_PART) :]) - 1]


def _prec(
    document_id: str,
    taxonomy_id: str,
    *,
    signed_text: str | None,
    standard: bool = False,
    signed: bool = True,
    signed_on: str | None = None,
    opening_text: str | None = None,
    opened_with: str | None | object = _DEFAULT,
    opened_with_form: str | None = None,
    refused: list[tuple[str, int]] | None = None,
    paper: str = "ours",
    rounds: int = 0,
    counterparty_alias: str | None = None,
    signed_vs: str | dict[str, Any] | None = None,
    opening_vs: str | dict[str, Any] | None = None,
    refused_vs: str | dict[str, Any] | None = None,
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
    record.update(
        {
            "signed_on": signed_on,
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
            **({"opened_with_form": opened_with_form} if opened_with_form is not None else {}),
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


def _vector_v06(name: str, description: str, doc: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": name,
        "description": description,
        "opf_version": OPF_VERSION_V06,
        "engine_version": ENGINE_VERSION,
        "digest_version": DIGEST_VERSION_V5,
        "input": doc,
        "expected": _expected(doc),
    }


_STANDARD_TEXT = (
    "Neither party may assign this Agreement without the other party's prior written consent."
)


def build_vectors_v06() -> list[tuple[str, dict[str, Any]]]:
    """The OPF 0.6 / digest 5 vector set. Synthetic inputs only."""
    vectors: list[tuple[str, dict[str, Any]]] = []

    # 001 — smallest 0.6 document: no clauses, no perspective.
    doc_001 = _base_v06()
    digest_001 = build_digest_v5(doc_001)
    assert digest_001["perspective"] is None
    assert digest_001["clauses"] == []
    vectors.append(
        (
            "001-minimal-no-perspective",
            _vector_v06(
                "minimal-no-perspective",
                "Smallest well-formed OPF 0.6 document: evidence {clauses: [], "
                "precedent: []}, no top-level perspective. expected.digest pins "
                "the digest_version 5 skeleton — perspective is PRESENT and null "
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
        _prec("deal-a", tid, signed_text=_STANDARD_TEXT, standard=True, signed_on="2025-01-10"),
        _prec("deal-b", tid, signed_text=_STANDARD_TEXT, standard=True, signed_on="2025-02-10"),
        # Variant A, three deals; deal-d spells it with different case and
        # punctuation (same group), and carries the latest signing date, known
        # to the month only.
        _prec("deal-c", tid, signed_text=variant_a, signed_on="2025-03-01", paper="theirs"),
        _prec(
            "deal-d",
            tid,
            signed_text="EITHER party may assign this Agreement to an affiliate, without consent",
            signed_on="2025-11",
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
            signed_on="2024-06-30",
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
    doc_002 = _base_v06(
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
    entry_002 = build_digest_v5(doc_002)["clauses"][0]
    assert [v["n_deals"] for v in entry_002["signed_variants"]] == [3, 1]
    assert entry_002["signed_variants"][0]["first_signed"] == "2025-03"
    assert entry_002["signed_variants"][0]["last_signed"] == "2025-11"
    assert entry_002["refused_asks"][0]["n_deals"] == 3
    assert entry_002["n_signed_standard"] == 2 and entry_002["n_deals"] == 8
    vectors.append(
        (
            "002-variants-refused-and-exclusions",
            _vector_v06(
                "variants-refused-and-exclusions",
                "One clause, eight deals, pinning digest_version 5 grouping and "
                "every exclusion rule: two deals signed our standard "
                "(n_signed_standard 2, never a variant); variant A signed in "
                "three deals, one spelled with different case/punctuation (exact "
                "normalization merges it) — its representative is the deal with "
                "the latest signed_on, a YYYY-MM month that sorts before every day "
                "in it (and whose digest first_signed / last_signed are 2025-03 "
                "and 2025-11); variant B differs from A by the negator "
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
    precedent_003.append(_prec("deal-8", tid_003, signed_text=long_text, signed_on="2025-05-05"))
    doc_003 = _base_v06(
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
    entry_003 = build_digest_v5(doc_003)["clauses"][0]
    assert entry_003["n_variants_total"] == 7
    assert len(entry_003["signed_variants"]) == 5
    assert len(entry_003["signed_variants"][0]["text"]) <= 300
    vectors.append(
        (
            "003-cap-totals-and-summary",
            _vector_v06(
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
    doc_004 = _base_v06(
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
    entry_004 = build_digest_v5(doc_004)["clauses"][0]
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
            _vector_v06(
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
    doc_005 = _base_v06(
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
            _vector_v06(
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
    # 006 — the opening rules of digest 5 (issue #234): every branch of
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
    doc_006 = _base_v06(
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
    entry_006 = build_digest_v5(doc_006)["clauses"][0]
    assert (entry_006["n_opened_standard"], entry_006["n_kept_standard"]) == (3, 1)
    assert [e["n_deals"] for e in entry_006["changed_openings"]] == [2, 1]
    assert [e["n_struck"] for e in entry_006["changed_openings"]] == [0, 1]
    assert entry_006["n_changed_openings_total"] == 2
    assert len(entry_006["refused_asks"]) == 1
    vectors.append(
        (
            "006-opening-rules",
            _vector_v06(
                "opening-rules",
                "The digest 5 opening rules (OPF-SPEC §3.12.2) on one clause with "
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
            opening_vs="mixed",
            opened_with="non_standard",
            rounds=1,
            paper="theirs",
        ),
        _prec("deal-h", tid_007, signed_text=texts_007["more"], signed_vs="more_protective"),
        _prec("deal-i", tid_007, signed_text=texts_007["none"]),
    ]
    doc_007 = _base_v06(
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
    digest_007 = build_digest_v5(doc_007)
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
            _vector_v06(
                "equivalence-label-and-coverage",
                "The vs_standard label in the digest (OPF-SPEC §3.5.6, §3.12.2) and "
                "uncovered_clause_types. Signed variants labelled equivalent (three "
                "deals, two distinct texts; the respelled text merges under the "
                "grouping key) collapse into ONE entry with n_deals 3, n_texts 2, "
                "two exemplars and the concession count; the rest are listed "
                "individually in tier order: less_protective (two deals), "
                "different_concept, the unjudged one (a signed text with no "
                "label), more_protective, then the collapsed entry. Each "
                "judged entry carries its part map (one token per part of our "
                "one-part standard), its addition tokens and its reason "
                "(issue #243); the unjudged one carries nulls. "
                "positions counts signed deals by standard, each label and "
                "unjudged. The refused ask and the changed opening carry their "
                "labels (the changed opening is mixed: its one part less "
                "protective, an addition more protective). The taxonomy lists "
                "one covered entry, an active and a "
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
    signed_on_008 = {"deal-c": "2025-01-15", "deal-d": "2025-06-30", "deal-f": "2025-06-30"}
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
        if record["document_id"] in signed_on_008:
            record["signed_on"] = signed_on_008[record["document_id"]]
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
    doc_008 = _base_v06(
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
            "vs_standard": None,
        },
        {
            "precedent_id": ids_c["deal-f"],
            "kind": "refused_ask",
            "opening": r1_c,
            "signed": std_c,
            "outcome": "ask_refused",
            "vs_standard": None,
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
            _vector_v06(
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
                "deal-f share the latest signed_on and deal-f has the lower precedent_id "
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
    # 009 — a standard of several parts, one of them repeated, and an owner
    # label over a judged part map (issue #243).
    tid_009 = "confidentiality"
    parts_009 = [
        "The Recipient shall hold the Discloser's Confidential Information in confidence.",
        "This Section survives the end of this Agreement.",
        "The Recipient shall notify the Discloser promptly of any unauthorized disclosure.",
        "This Section survives the end of this Agreement.",
    ]
    std_009 = "\n\n".join(parts_009)
    v1_009 = (
        "The Recipient shall hold the Discloser's Confidential Information in confidence "
        "and shall notify the Discloser within thirty days of any unauthorized disclosure, "
        "and shall return or destroy it on request. This Section survives the end of this "
        "Agreement."
    )
    v2_009 = (
        "The Recipient keeps the Discloser's Confidential Information in confidence and "
        "notifies the Discloser promptly of any unauthorized disclosure."
    )
    ask_009 = (
        "The Recipient shall hold the Discloser's Confidential Information in confidence "
        "and may disclose it to its advisers."
    )
    o1_009 = (
        "The Recipient shall use reasonable efforts to hold the Discloser's Confidential "
        "Information in confidence and shall notify the Discloser promptly of any "
        "unauthorized disclosure. This Section survives the end of this Agreement."
    )

    def _p(n: int, status: str, direction: str, reason: str | None = None) -> dict[str, Any]:
        key = "relation" if status == "present" else "effect"
        part: dict[str, Any] = {"part_id": _part(n), "status": status, key: direction}
        if reason is not None:
            part["reason"] = reason
        return part

    # Signed in deal-a after opening with our standard: mixed. Part 2 dropped
    # (neutral), part 3 less protective, part 4 (the repeat) kept, and an
    # addition more protective.
    vs_v1_009 = {
        "label": "mixed",
        "reason": "Synthetic fixture reason: a later notice deadline.",
        "parts": [
            _p(1, "present", "equivalent"),
            _p(2, "omitted", "neutral"),
            _p(3, "present", "less_protective", "Synthetic fixture: notice within thirty days."),
            _p(4, "present", "equivalent"),
        ],
        "additions": [
            {
                "summary": "Synthetic fixture: return or destroy on request.",
                "effect": "more_protective",
            }
        ],
        "basis": "agent",
        "check": None,
    }
    # Signed in deal-b: equivalent although it drops both survival sentences
    # (neutral omissions) and adds a neutral sentence.
    vs_v2_009 = {
        "label": "equivalent",
        "reason": "Synthetic fixture reason: same duties, no survival sentence.",
        "parts": [
            _p(1, "present", "equivalent"),
            _p(2, "omitted", "neutral"),
            _p(3, "present", "equivalent"),
            _p(4, "omitted", "neutral"),
        ],
        "additions": [],
        "basis": "agent",
        "check": None,
    }
    # Asked and refused in deal-c: less protective (part 3 dropped, an
    # adviser carve-out added).
    vs_ask_009 = {
        "label": "less_protective",
        "reason": "Synthetic fixture reason: no notice duty.",
        "parts": [
            _p(1, "present", "equivalent"),
            _p(2, "omitted", "neutral"),
            _p(3, "omitted", "less_protective", "Synthetic fixture: the notice duty is gone."),
            _p(4, "omitted", "neutral"),
        ],
        "additions": [
            {"summary": "Synthetic fixture: disclosure to advisers.", "effect": "less_protective"}
        ],
        "basis": "agent",
        "check": None,
    }
    # Opened in deal-d, which signed our standard: the parts derive mixed
    # (part 1 less protective, part 2 omitted more protective, part 4 the
    # repeat more protective) but the owner set the label to more_protective,
    # with a reason.
    vs_o1_009 = {
        "label": "more_protective",
        "reason": "Synthetic fixture reason: reasonable efforts only.",
        "parts": [
            _p(1, "present", "less_protective", "Synthetic fixture: reasonable efforts only."),
            _p(
                2,
                "omitted",
                "more_protective",
                "Synthetic fixture: the first survival sentence is gone.",
            ),
            _p(3, "present", "equivalent"),
            _p(4, "present", "more_protective", "Synthetic fixture: survival kept once, narrower."),
        ],
        "additions": [],
        "basis": "owner",
        "check": None,
        "owner_label": {
            "label": "more_protective",
            "reason": "Synthetic fixture owner reason: survival matters more here.",
        },
    }
    precedent_009 = [
        _prec(
            "deal-a",
            tid_009,
            signed_text=v1_009,
            signed_vs=vs_v1_009,
            opening_text=std_009,
            rounds=1,
        ),
        _prec("deal-b", tid_009, signed_text=v2_009, signed_vs=vs_v2_009, paper="theirs"),
        _prec(
            "deal-c",
            tid_009,
            signed_text=std_009,
            standard=True,
            refused=[(ask_009, 2)],
            refused_vs=vs_ask_009,
            rounds=1,
        ),
        _prec(
            "deal-d",
            tid_009,
            signed_text=std_009,
            standard=True,
            opening_text=o1_009,
            opening_vs=vs_o1_009,
            opened_with="non_standard",
            rounds=1,
            paper="theirs",
        ),
    ]
    doc_009 = _base_v06(
        clauses=[
            {
                "id": f"clause.{tid_009}",
                "taxonomy_id": tid_009,
                "title": "Confidentiality",
                "our_standard": {
                    "text": std_009,
                    "parts": [{"part_id": "", "order": 0, "text": t} for t in parts_009],
                    "source_ref": {
                        "document_id": "template",
                        "version": "template",
                        "clause_path": "4",
                    },
                },
            }
        ],
        precedent=precedent_009,
        documents=[
            _deal(
                f"deal-{c}",
                signed_version=3,
                provenance="counterparty_paper" if c in "bd" else "our_paper",
            )
            for c in "abcd"
        ],
        perspective={"party": "Fixture Co", "counterparty_type": "Fixture Counterparty"},
    )
    pids_009 = [p["part_id"] for p in doc_009["evidence"]["clauses"][0]["our_standard"]["parts"]]
    assert pids_009[3] == f"{pids_009[1]}-2", pids_009
    digest_009 = build_digest_v5(doc_009)
    entry_009 = digest_009["clauses"][0]
    assert [p["part_id"] for p in entry_009["our_standard"]["parts"]] == pids_009
    mixed_009 = entry_009["signed_variants"][0]
    assert mixed_009["label"] == "mixed" and "label_by" not in mixed_009
    assert mixed_009["part_map"] == ["=", "x0", "-", "="] and mixed_009["additions"] == ["+"]
    assert entry_009["signed_variants"][1]["n_texts"] == 1
    assert entry_009["refused_asks"][0]["part_map"] == ["=", "x0", "x-", "x0"]
    assert entry_009["refused_asks"][0]["additions"] == ["-"]
    opening_009 = entry_009["changed_openings"][0]
    assert opening_009["label"] == "more_protective" and opening_009["label_by"] == "owner"
    assert opening_009["part_map"] == ["-", "x+", "=", "+"]
    assert opening_009["reason"] == vs_o1_009["owner_label"]["reason"]
    dossier_009 = build_derived_sections(doc_009)["dossiers"][f"clause.{tid_009}"]
    excerpts_009 = dossier_009["excerpts"]
    assert [x["kind"] for x in excerpts_009] == ["signed_variant", "changed_opening"]
    assert [r["part_id"] for r in excerpts_009[0]["vs_standard"]["part_reasons"]] == [pids_009[2]]
    assert [r["part_id"] for r in excerpts_009[1]["vs_standard"]["part_reasons"]] == [
        pids_009[0],
        pids_009[1],
        pids_009[3],
    ]
    assert excerpts_009[1]["vs_standard"]["label_by"] == "owner"
    vectors.append(
        (
            "009-standard-parts-and-owner-label",
            _vector_v06(
                "standard-parts-and-owner-label",
                "A standard of four parts whose second and fourth parts are the same "
                "passage (OPF-SPEC §3.5, §3.5.6, §3.12.2, §3.12.3, issue #243). The "
                "fourth part's part_id is the second's content hash with -2 appended, "
                "and the digest's our_standard carries all four parts in order. Every "
                "part map has one token per part, in standard order. Deal-a opened with "
                "our standard and signed a mixed variant (=, x0, -, = plus a more "
                "protective addition): a concession, the first signed variant, and the "
                "dossier's first excerpt, whose part_reasons name only the third part. "
                "Deal-b signed a variant that drops both copies of the repeated passage "
                "as neutral omissions: still equivalent, so it collapses (n_texts 1). "
                "Deal-c's refused ask keeps the first part, drops the rest (the third "
                "less protective) and adds a less protective sentence: less_protective. "
                "Deal-d opened with language whose parts derive mixed (-, x+, =, +) but "
                "the owner set the label to more_protective with a reason (owner_label): "
                "its changed opening in the digest carries label more_protective, "
                "label_by owner and the owner's reason, never the judgment's own reason, "
                "with the part map as judged; every other judged entry carries no "
                "label_by (its label is derived from its parts, the default the legend "
                "states) and the judgment's reason. The dossier's second excerpt is that "
                "changed opening, with label_by owner, the owner's reason and part "
                "reasons for the first, second and fourth (repeated, -2) parts.",
                doc_009,
            ),
        )
    )

    # 010 - signing dates: signed_on / signed_on_basis on documents and
    # precedent, first_signed / last_signed on groups and clauses.
    vectors.append(build_vector_010())

    # 011 - how settled each judgment is (issue #248).
    vectors.append(build_vector_011())

    # 012 - our former forms: prior standards and the standard history (issue #249).
    vectors.append(build_vector_012())
    # 013 - a former form rebuilt from the corpus (issue #250).
    vectors.append(build_vector_013())
    vectors.append(build_vector_014())
    return vectors


def build_vector_010() -> tuple[str, dict[str, Any]]:
    tid = "assignment"
    variant_a = "Either party may assign this Agreement to an affiliate without consent."
    variant_b = "Either party may assign this Agreement on written notice to the other."
    ask = "Either party may assign this Agreement freely."
    standard_text = _STANDARD_TEXT
    # (deal, signed_on, basis): every basis, both precisions, an unsigned deal
    # and a signed deal whose date no source states (null, never guessed).
    dated = {
        "deal-a": ("2022-03-14", "document_text"),
        "deal-b": ("2023-07", "filename"),
        "deal-c": ("2023-09-30", "signature_block"),
        "deal-d": ("2024-01-05", "file_metadata"),
        "deal-e": ("2024-11-20", "document_text"),
    }
    precedent = [
        _prec(
            "deal-a", tid, signed_text=variant_a, signed_on="2022-03-14", opened_with="non_standard"
        ),
        _prec(
            "deal-b",
            tid,
            signed_text=variant_a,
            signed_on="2023-07",
            opening_text=standard_text,
            rounds=1,
            paper="theirs",
        ),
        _prec(
            "deal-c",
            tid,
            signed_text=standard_text,
            standard=True,
            signed_on="2023-09-30",
            refused=[(ask, 2)],
            rounds=1,
        ),
        _prec(
            "deal-d",
            tid,
            signed_text=variant_b,
            signed_on="2024-01-05",
            opened_with="non_standard",
            paper="theirs",
        ),
        _prec(
            "deal-e",
            tid,
            signed_text=standard_text,
            standard=True,
            signed_on="2024-11-20",
            refused=[(ask, 2)],
            rounds=1,
        ),
        # Signed, but no source states when: null, with a null basis. Its
        # refused ask is the undated one a producer can record (an unsigned
        # deal records none, issue #221).
        _prec(
            "deal-f",
            tid,
            signed_text=variant_b,
            opened_with="non_standard",
            paper="theirs",
            refused=[(ask, 2)],
            rounds=1,
        ),
        # No detected executed copy: no signing date (and no refused asks).
        _prec("deal-g", tid, signed_text=variant_a, signed=False, paper="unknown"),
    ]
    documents = [
        _deal(
            deal,
            signed_version=3,
            provenance="our_paper"
            if deal in ("deal-a", "deal-c", "deal-e")
            else "counterparty_paper",
            signed_on=on,
            basis=basis,
        )
        for deal, (on, basis) in dated.items()
    ] + [
        _deal("deal-f", signed_version=3, provenance="counterparty_paper"),
        _deal("deal-g", signed_version=None, provenance="counterparty_paper"),
    ]
    doc = _base_v06(
        clauses=[
            {
                "id": f"clause.{tid}",
                "taxonomy_id": tid,
                "title": "Assignment",
                "our_standard": {
                    "text": standard_text,
                    "source_ref": {
                        "document_id": "template",
                        "version": "template",
                        "clause_path": "9",
                    },
                },
            }
        ],
        precedent=precedent,
        documents=documents,
        perspective={"party": "Fixture Co", "counterparty_type": "Fixture Counterparty"},
    )
    digest = build_digest_v5(doc)
    clause = digest["clauses"][0]
    assert (clause["first_signed"], clause["last_signed"]) == ("2022-03", "2024-11")
    by_text = {v["n_deals"]: v for v in clause["signed_variants"]}
    assert (by_text[2]["first_signed"], by_text[2]["last_signed"]) == ("2022-03", "2023-07")
    refused = clause["refused_asks"][0]
    assert refused["n_deals"] == 3
    assert (refused["first_signed"], refused["last_signed"]) == ("2023-09", "2024-11")
    assert digest["corpus"] == {
        "n_deals": 7,
        "n_signed": 6,
        "first_signed": "2022-03",
        "last_signed": "2024-11",
    }
    return (
        "010-signing-dates",
        _vector_v06(
            "signing-dates",
            "When each deal was signed, as a fact (OPF-SPEC §3.8, §3.12.2, issue #247). "
            "corpus.documents[].signed_on is an ISO date or, when only the month is "
            "known, a YYYY-MM; signed_on_basis says where it was read (document_text, "
            "signature_block, filename, file_metadata) and is null exactly when signed_on "
            "is. Deal-f is signed but no source states a date (null, null); deal-g has no "
            "executed copy (signed_version null) and so no date. Every precedent record "
            "carries its document's signed_on. Digest 5 reports first_signed / last_signed "
            "as YYYY-MM: on the corpus (2022-03, 2024-11), on the clause (the same span "
            "over its precedent records), on each signed variant (variant A, two deals, "
            "2022-03 to 2023-07 - a month-only date sorts before every day in it), on the "
            "collapsed equivalent entry when there is one, and on each refused ask group "
            "(the ask refused in deals c and e is dated by those deals' records: 2023-09 "
            "to 2024-11, and deal-f's refused ask, from a signed deal with no date, adds "
            "none).",
            doc,
        ),
    )


def _settled_vs(label: str, *, basis: str = "agent", check: str | None = None) -> dict[str, Any]:
    """A judgment of *label* in one of the states a producer reaches (issue #248).

    ``check``: ``None`` (never checked), ``"agreed"`` (the blind check reached the
    same label and parts), ``"adjudicated"`` (it disagreed on the label and a
    fresh agent decided) or ``"disputed"`` (it disagreed and awaits
    adjudication). ``basis="owner"`` is what ``playbook project`` writes for an
    owner's correction: it carries no check.
    """
    checks: dict[str, dict[str, Any]] = {
        "agreed": {"agreed": True, "label_agreed": True, "parts_agreed": True},
        "adjudicated": {"agreed": False, "label_agreed": False, "parts_agreed": True},
        "disputed": {"agreed": False, "label_agreed": False, "parts_agreed": True},
    }
    vs = _vs(label)
    assert vs is not None
    vs["basis"] = basis
    vs["check"] = (
        None
        if check is None
        else {**checks[check], "by": "claude-opus-5-5", "adjudicated": check == "adjudicated"}
    )
    return vs


def build_vector_011() -> tuple[str, dict[str, Any]]:
    tid = "assignment"
    t = {
        "less": "Either party may assign this Agreement to anyone at any time.",
        "mixed": "Either party may assign this Agreement to an affiliate on notice.",
        "diff": "Assignment requires a payment of one million dollars.",
        "more": "Neither party may assign this Agreement even with consent.",
        "eq1": "Either party may assign this Agreement to an affiliate.",
        "eq2": "An affiliate assignment by either party is permitted.",
        "eq3": "Assignment to an affiliate by either party is allowed.",
        "eq4": "Either party may transfer this Agreement to an affiliate.",
        "ask1": "Either party may assign this Agreement freely.",
        "ask2": "Assignment is governed by the parties' later written agreement.",
        "open": "Neither party may assign this Agreement to anyone.",
    }
    precedent = [
        _prec("deal-a", tid, signed_text=_STANDARD_TEXT, standard=True),
        _prec(
            "deal-b",
            tid,
            signed_text=t["less"],
            signed_vs=_settled_vs("less_protective", check="agreed"),
        ),
        _prec(
            "deal-c",
            tid,
            signed_text=t["mixed"],
            signed_vs=_settled_vs("mixed", check="adjudicated"),
            opening_text=_STANDARD_TEXT,
            rounds=1,
        ),
        _prec(
            "deal-d",
            tid,
            signed_text=t["diff"],
            signed_vs=_settled_vs("different_concept", check="disputed"),
            paper="theirs",
        ),
        _prec(
            "deal-e",
            tid,
            signed_text=t["more"],
            signed_vs=_settled_vs("more_protective", basis="owner"),
            paper="theirs",
        ),
        _prec(
            "deal-f",
            tid,
            signed_text=t["eq1"],
            signed_vs=_settled_vs("equivalent", check="agreed"),
            refused=[(t["ask1"], 2)],
            refused_vs=_settled_vs("more_protective", check="adjudicated"),
            rounds=1,
        ),
        _prec(
            "deal-g",
            tid,
            signed_text=t["eq2"],
            signed_vs=_settled_vs("equivalent", check="agreed"),
            opening_text=t["open"],
            opening_vs=_settled_vs("mixed"),
            opened_with="non_standard",
            rounds=1,
            paper="theirs",
        ),
        _prec(
            "deal-h",
            tid,
            signed_text=t["eq3"],
            signed_vs=_settled_vs("equivalent", check="adjudicated"),
            refused=[(t["ask2"], 2)],
            rounds=1,
        ),
        _prec(
            "deal-i",
            tid,
            signed_text=t["eq4"],
            signed_vs=_settled_vs("equivalent", basis="owner"),
        ),
    ]
    doc = _base_v06(
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
        precedent=precedent,
        documents=[
            _deal(
                f"deal-{c}",
                signed_version=3,
                provenance="counterparty_paper" if c in "deg" else "our_paper",
            )
            for c in "abcdefghi"
        ],
        perspective={"party": "Fixture Co", "counterparty_type": "Fixture Counterparty"},
    )
    digest = build_digest_v5(doc)
    entry = digest["clauses"][0]
    # Independent expectations, written down from the fixture and not read from
    # the engine: the order is the label tiers (less/mixed in grouping-key order,
    # different_concept, more_protective, the collapsed equivalent entry last).
    settled = [v.get("settled", "collapsed") for v in entry["signed_variants"]]
    assert settled == ["adjudicated", "agreed", "unchecked", "owner", "collapsed"], settled
    assert entry["signed_variants"][-1]["settled_counts"] == {
        "agreed": 2,
        "adjudicated": 1,
        "unchecked": 0,
        "owner": 1,
    }
    assert [a["settled"] for a in entry["refused_asks"]] == [None, "adjudicated"]
    assert [o["settled"] for o in entry["changed_openings"]] == ["unchecked"]
    assert set(digest["part_map_legend"]["settled"]) == {
        "agreed",
        "adjudicated",
        "unchecked",
        "owner",
    }
    dossier = build_derived_sections(doc)["dossiers"][f"clause.{tid}"]
    assert [x["vs_standard"]["settled"] for x in dossier["excerpts"]] == [
        "adjudicated",
        "unchecked",
    ]
    return (
        "011-settled-outcomes",
        _vector_v06(
            "settled-outcomes",
            "How settled each judged text's vs_standard is (OPF-SPEC §3.5.6, §3.12.2, "
            "§3.12.3, issue #248): derived from the judgment's basis and blind check, "
            "never stored on the evidence. Four signed variants and the collapsed "
            "equivalent entry, in tier order: a mixed text the check disagreed on and a "
            "fresh agent decided (settled adjudicated), a less_protective text whose "
            "check agreed (agreed), a different_concept text whose check disagreed and "
            "still awaits adjudication (read as unchecked: nothing is settled yet), a "
            "more_protective text the owner set (owner; basis owner carries no check). "
            "The four equivalent texts collapse into one entry whose settled_counts is "
            "2 agreed, 1 adjudicated, 0 unchecked, 1 owner, summing to n_texts 4; it "
            "carries no settled of its own. Of the two refused asks one is adjudicated "
            "and one is not judged (settled null); the one changed opening was never "
            "checked (unchecked). Each judged digest entry carries settled beside its "
            "label and part map, part_map_legend.settled states the four values, and "
            "the dossier's two excerpts (the concession whose judgment was adjudicated, "
            "the changed opening that was unchecked) carry settled in their "
            "vs_standard.",
            doc,
        ),
    )


def build_vector_012() -> tuple[str, dict[str, Any]]:
    tid = "assignment"
    s1 = "Neither party may assign this Agreement without the other party's prior written consent."
    s2 = "Consent shall not be unreasonably withheld for an assignment to an affiliate."
    std_text = f"{s1}\n\n{s2}"
    p18b = "Consent may be withheld for any reason, including an assignment to an affiliate."
    text_18 = f"{s1}\n\n{p18b}"
    text_19 = "Either party may assign this Agreement to an affiliate without consent."
    variant = "Neither party may assign without consent, except to an affiliate on notice."
    vs_variant_other = "Assignment is prohibited."

    def _p(n: int, status: str, direction: str, reason: str | None = None) -> dict[str, Any]:
        key = "relation" if status == "present" else "effect"
        part: dict[str, Any] = {"part_id": _part(n), "status": status, key: direction}
        if reason is not None:
            part["reason"] = reason
        return part

    # The 2018 form against today's two parts: part 1 is the same passage,
    # part 2 lets consent be withheld for any reason.
    vs_18 = {
        "label": "less_protective",
        "reason": "Synthetic fixture reason: consent may be withheld for any reason.",
        "parts": [
            _p(1, "present", "equivalent"),
            _p(2, "present", "less_protective", "Synthetic fixture: no reasonableness limit."),
        ],
        "additions": [],
        "basis": "agent",
        "check": None,
    }
    # The 2019 form: neither part is there, and it adds a free affiliate assignment.
    vs_19 = {
        "label": "different_concept",
        "reason": "Synthetic fixture reason: a free affiliate assignment instead of consent.",
        "parts": [
            _p(1, "omitted", "less_protective", "Synthetic fixture: no consent requirement."),
            _p(2, "omitted", "neutral"),
        ],
        "additions": [
            {"summary": "Synthetic fixture: free affiliate assignment.", "effect": "neutral"}
        ],
        "basis": "agent",
        "check": None,
    }
    vs_signed_variant = {
        "label": "equivalent",
        "reason": "Synthetic fixture reason: same legal effect.",
        "parts": [_p(1, "present", "equivalent"), _p(2, "present", "equivalent")],
        "additions": [],
        "basis": "agent",
        "check": None,
    }
    form_18, form_19 = "2018 form", "2019 form"
    precedent = [
        # deal-a: opened on the 2018 form and signed it as it stood (kept).
        _prec(
            "deal-a",
            tid,
            signed_text=text_18,
            signed_on="2020-02",
            signed_vs=copy.deepcopy(vs_18),
            opened_with="prior_standard",
            opened_with_form=form_18,
        ),
        # deal-b: opened on the 2018 form and moved to today's standard.
        _prec(
            "deal-b",
            tid,
            signed_text=std_text,
            standard=True,
            signed_on="2020-09",
            opening_text=text_18,
            opened_with="prior_standard",
            opened_with_form=form_18,
            rounds=1,
        ),
        # deal-c: opened on the 2019 form and ended on a variant (conceded).
        _prec(
            "deal-c",
            tid,
            signed_text=variant,
            signed_on="2020-05",
            signed_vs=copy.deepcopy(vs_signed_variant),
            opening_text=text_19,
            opened_with="prior_standard",
            opened_with_form=form_19,
            rounds=1,
            paper="theirs",
        ),
        # deal-d: opened on the 2018 form and signed it as it stood (kept).
        _prec(
            "deal-d",
            tid,
            signed_text=text_18,
            signed_on="2021-03",
            signed_vs=copy.deepcopy(vs_18),
            opened_with="prior_standard",
            opened_with_form=form_18,
        ),
        # deal-e: opened with and kept today's standard.
        _prec(
            "deal-e",
            tid,
            signed_text=std_text,
            standard=True,
            signed_on="2023-01",
            opened_with="standard",
        ),
        # deal-f: the counterparty's own opening, not signed as proposed; the
        # text it signed is not judged. Its origin is theirs.
        _prec(
            "deal-f",
            tid,
            signed_text=vs_variant_other,
            signed_on="2022-06",
            opening_text="Either party may assign this Agreement freely.",
            opened_with="non_standard",
            rounds=1,
            paper="theirs",
        ),
    ]
    doc = _base_v06(
        clauses=[
            {
                "id": f"clause.{tid}",
                "taxonomy_id": tid,
                "title": "Assignment",
                "our_standard": {
                    "text": std_text,
                    "source_ref": {
                        "document_id": "template",
                        "version": "template",
                        "clause_path": "9",
                    },
                    "parts": [
                        {"part_id": "", "order": 1, "text": s1},
                        {"part_id": "", "order": 2, "text": s2},
                    ],
                },
                "prior_standards": [
                    {
                        "form_label": form_18,
                        "text": text_18,
                        "parts": [
                            {"part_id": "", "order": 1, "text": s1},
                            {"part_id": "", "order": 2, "text": p18b},
                        ],
                        "source_ref": {
                            "document_id": f"prior_template:{form_18}",
                            "version": "prior_template",
                            "clause_path": "9",
                        },
                        "vs_standard": copy.deepcopy(vs_18),
                    },
                    {
                        "form_label": form_19,
                        "text": text_19,
                        "parts": [{"part_id": "", "order": 1, "text": text_19}],
                        "source_ref": {
                            "document_id": f"prior_template:{form_19}",
                            "version": "prior_template",
                            "clause_path": "11",
                        },
                        "vs_standard": copy.deepcopy(vs_19),
                    },
                    {
                        "form_label": "2021 form",
                        "text": std_text,
                        "parts": [
                            {"part_id": "", "order": 1, "text": s1},
                            {"part_id": "", "order": 2, "text": s2},
                        ],
                        "source_ref": {
                            "document_id": "prior_template:2021 form",
                            "version": "prior_template",
                            "clause_path": "9",
                        },
                        "vs_standard": None,
                    },
                ],
                "why_changed": {
                    "text": "Synthetic fixture note: we stopped letting consent be withheld for any reason.",
                    "basis": "owner",
                },
            }
        ],
        precedent=precedent,
        documents=[
            _deal(
                f"deal-{c}",
                signed_version=3,
                provenance="counterparty_paper" if c in "cf" else "our_paper",
            )
            for c in "abcdef"
        ],
        perspective={"party": "Fixture Co", "counterparty_type": "Fixture Counterparty"},
        prior_templates=[
            {
                "label": form_18,
                "in_use_from": "2018-01",
                "in_use_until": "2019-12",
                "sha256": "sha256:" + "0" * 64,
            },
            {"label": form_19, "in_use_from": "2019-03", "sha256": "sha256:" + "1" * 64},
            {"label": "2021 form", "in_use_from": "2021", "sha256": "sha256:" + "2" * 64},
        ],
    )
    clause = doc["evidence"]["clauses"][0]
    # Independent expectations, written down from the fixture and not read from the engine.
    assert (clause["n_opened_prior"], clause["n_kept_prior"], clause["n_moved_to_current"]) == (
        4,
        2,
        1,
    )
    assert (clause["n_opened_standard"], clause["n_kept_standard"]) == (1, 1)
    assert clause["n_changed_openings"] == 1  # deal-f only: our own former forms are not asks
    digest = build_digest_v5(doc)
    history = digest["clauses"][0]["standard_history"]
    assert [f["form"] for f in history["forms"]] == [form_18, form_19, "2021 form"]
    f18, f19, f21 = history["forms"]
    assert (f18["n_opened"], f18["n_kept"], f18["n_conceded"], f18["n_moved_to_current"]) == (
        3,
        2,
        1,
        1,
    )
    assert (f19["n_opened"], f19["n_kept"], f19["n_conceded"], f19["n_moved_to_current"]) == (
        1,
        0,
        1,
        0,
    )
    assert (f21["n_opened"], f21["unchanged"], f21["part_map"]) == (0, True, None)
    assert (f18["in_use_from"], f18["in_use_until"], f18["in_use_basis"]) == (
        "2018-01",
        "2019-12",
        "config",
    )
    assert (f19["in_use_from"], f19["in_use_until"], f19["in_use_basis"]) == (
        "2019-03",
        "2020-05",
        "mixed",
    )
    assert (f21["in_use_from"], f21["in_use_until"], f21["in_use_basis"]) == (
        "2021",
        None,
        "config",
    )
    assert f18["part_map"] == ["=", "-"] and f19["part_map"] == ["x-", "x0"]
    assert f18["parts"][0]["text"] is None and f18["parts"][1]["text"] == p18b
    assert history["why_changed"]["basis"] == "owner"
    # Each form's clause is cited in that form's own bytes, never as "template".
    assert f18["ref"] == {
        "document_id": f"prior_template:{form_18}",
        "version": "prior_template",
        "clause_path": "9",
    }
    return (
        "012-prior-standards",
        _vector_v06(
            "prior-standards",
            "Our former forms and the standard history (OPF-SPEC §3.5.7, §3.12.2, issue "
            "#249). Three forms are supplied (baseline.prior_templates, with the content "
            "address and any dates the owner gave). Five of six deals' clauses opened on "
            "something of ours or a counterparty's: deal-a and deal-d opened on the 2018 "
            "form and signed it as it stood (opened_with prior_standard, opened_with_form "
            "2018 form, no distinct opening_text: kept), deal-b opened on it and moved to "
            "today's standard (opening_text recorded, standard true), deal-c opened on the "
            "2019 form and ended on a variant (conceded), deal-e opened with today's "
            "standard, and deal-f opened with the counterparty's own text (non_standard). "
            "The clause counts n_opened_prior 4, n_kept_prior 2 and n_moved_to_current 1 "
            "are derived from those records, and n_changed_openings is 1: only deal-f's "
            "opening is a counterparty ask, our own former wording never is. Each form's "
            "clause is judged once against today's two parts (role prior_standard, the "
            "same cache key as any text): the 2018 form's judgment is also the verdict of "
            "the two deals that signed its words. The 2021 form reads exactly as today's "
            "standard, so it is unchanged with no part map. The digest's standard_history "
            "lists the forms in config order with their in-use dates (config for the 2018 "
            "form, the config start and the latest signing month of its deals for the 2019 "
            "form: mixed, only the config start for the 2021 form), the opened, kept, "
            "conceded and moved-to-current deals, the part map against today's standard, "
            "and the form's own parts (null where the passage is today's); why_changed is "
            "the owner's note, the only reason ever given. Each form's clause is cited in "
            "that form's own bytes (document_id prior_template:<label>, version "
            "prior_template, resolved through the form's baseline.prior_templates sha256), "
            "never as template, which is today's form.",
            doc,
        ),
    )


def build_vector_013() -> tuple[str, dict[str, Any]]:
    tid = "assignment"
    std_text = (
        "Neither party may assign this Agreement without the other party's prior written consent."
    )
    old_text = "Either party may assign this Agreement to an affiliate without consent."
    label = "reconstructed 2020-02 to 2021-03"
    vs_old = {
        "label": "different_concept",
        "reason": "Synthetic fixture reason: a free affiliate assignment instead of consent.",
        "parts": [
            {
                "part_id": _part(1),
                "status": "omitted",
                "effect": "less_protective",
                "reason": "Synthetic fixture: no consent requirement.",
            }
        ],
        "additions": [
            {"summary": "Synthetic fixture: free affiliate assignment.", "effect": "neutral"}
        ],
        "basis": "agent",
        "check": None,
    }
    precedent = [
        # Three deals opened on the reconstructed form; deal-a kept it.
        _prec(
            "deal-a",
            tid,
            signed_text=old_text,
            signed_on="2020-02",
            signed_vs=copy.deepcopy(vs_old),
            opened_with="prior_standard",
            opened_with_form=label,
        ),
        _prec(
            "deal-b",
            tid,
            signed_text=std_text,
            standard=True,
            signed_on="2020-09",
            opening_text=old_text,
            opened_with="prior_standard",
            opened_with_form=label,
            rounds=1,
        ),
        _prec(
            "deal-c",
            tid,
            signed_text=old_text,
            signed_on="2021-03",
            signed_vs=copy.deepcopy(vs_old),
            opened_with="prior_standard",
            opened_with_form=label,
        ),
        _prec(
            "deal-d",
            tid,
            signed_text=std_text,
            standard=True,
            signed_on="2023-01",
            opened_with="standard",
        ),
    ]
    doc = _base_v06(
        clauses=[
            {
                "id": f"clause.{tid}",
                "taxonomy_id": tid,
                "title": "Assignment",
                "our_standard": {
                    "text": std_text,
                    "source_ref": {
                        "document_id": "template",
                        "version": "template",
                        "clause_path": "9",
                    },
                    "parts": [{"part_id": "", "order": 1, "text": std_text}],
                },
                "prior_standards": [
                    {
                        "form_label": label,
                        "text": old_text,
                        "parts": [{"part_id": "", "order": 1, "text": old_text}],
                        # A reconstructed form has no file: the dominant wording is
                        # cited where it sits in a corpus document.
                        "source_ref": {"document_id": "deal-a", "version": 3, "clause_path": "9"},
                        "vs_standard": copy.deepcopy(vs_old),
                    }
                ],
                "why_changed": None,
            }
        ],
        precedent=precedent,
        documents=[_deal(f"deal-{c}", signed_version=3, provenance="our_paper") for c in "abcd"],
        perspective={"party": "Fixture Co", "counterparty_type": "Fixture Counterparty"},
        prior_templates=[
            {
                "label": label,
                "basis": "reconstructed",
                "n_deals": 3,
                "first_signed": "2020-02",
                "last_signed": "2021-03",
            }
        ],
    )
    clause = doc["evidence"]["clauses"][0]
    # Independent expectations, written down from the fixture and not read from the engine.
    assert (clause["n_opened_prior"], clause["n_kept_prior"], clause["n_moved_to_current"]) == (
        3,
        2,
        1,
    )
    digest = build_digest_v5(doc)
    (form,) = digest["clauses"][0]["standard_history"]["forms"]
    assert form["form"] == label and form["basis"] == "reconstructed"
    assert (form["in_use_from"], form["in_use_until"], form["in_use_basis"]) == (
        "2020-02",
        "2021-03",
        "deal_dates",
    )
    assert (form["n_opened"], form["n_kept"], form["n_conceded"]) == (3, 2, 1)
    assert form["ref"] == {"document_id": "deal-a", "version": 3, "clause_path": "9"}
    return (
        "013-reconstructed-prior-form",
        _vector_v06(
            "reconstructed-prior-form",
            "A former form of ours rebuilt from the corpus when none was supplied "
            "(OPF-SPEC §3.5.7, issue #250). baseline.prior_templates holds one entry with "
            "basis reconstructed, n_deals 3 and the signing-month span of the deals that "
            "opened on it (first_signed 2020-02, last_signed 2021-03); it has no sha256 and "
            "no owner dates because it has no file. Three of four deals' clauses opened on "
            "it (opened_with prior_standard, opened_with_form the label): deal-a and deal-c "
            "signed it as it stood (kept), deal-b moved to today's standard; deal-d opened "
            "with today's standard. The prior standard's source_ref is the ordinary citation "
            "of the dominant wording in a corpus document (deal-a, version 3), never "
            "prior_template. The digest's standard_history form carries basis "
            "reconstructed and in-use dates inferred from the deals' signing months "
            "(in_use_basis deal_dates).",
            doc,
        ),
    )


def build_vector_014() -> tuple[str, dict[str, Any]]:
    """Verified absence counts (OPF-SPEC §3.5.8, issue #251)."""
    assign = "assignment"
    law = "governing_law"
    assign_text = (
        "Neither party may assign this Agreement without the other party's prior written consent."
    )
    law_text = "This Agreement is governed by the laws of the State of New York."
    law_other = "This Agreement is governed by the laws of the State of Delaware."

    def std(text: str, clause_path: str) -> dict[str, Any]:
        return {
            "text": text,
            "source_ref": {
                "document_id": "template",
                "version": "template",
                "clause_path": clause_path,
            },
        }

    deals = [
        ("deal-a", 3, "2025-01-10"),
        ("deal-b", 3, "2025-02-10"),
        ("deal-c", 3, "2025-03-10"),
        ("deal-d", 3, "2025-04"),
        ("deal-e", 3, None),
        ("deal-f", 3, "2025-06-20"),
        ("deal-g", None, None),
        ("deal-h", 3, "2025-08-01"),
    ]
    documents = [
        _deal(
            name,
            signed_version=version,
            provenance="our_paper",
            **({"signed_on": signed_on} if signed_on is not None else {}),
        )
        for name, version, signed_on in deals
    ]
    precedent = [
        # deal-a signed both clauses: it is counted by neither absence count.
        _prec("deal-a", assign, signed_text=assign_text, standard=True, signed_on="2025-01-10"),
        _prec("deal-a", law, signed_text=law_text, standard=True, signed_on="2025-01-10"),
        # deal-b signed only governing law.
        _prec("deal-b", law, signed_text=law_other, signed_on="2025-02-10"),
        # deal-f opened with our assignment standard and struck it before
        # signing: it signed without the clause, and the text it signed holds
        # nothing like it.
        _prec(
            "deal-f",
            assign,
            signed_text=None,
            opening_text=assign_text,
            signed_on="2025-06-20",
            opened_with="standard",
        ),
        _prec("deal-f", law, signed_text=law_text, standard=True, signed_on="2025-06-20"),
        # deal-g has no executed copy: it signed nothing, so it is in no count.
        _prec("deal-g", assign, signed_text=assign_text, standard=True, signed=False),
    ]

    # Passages of the executed copies that are not precedent. Their scores against
    # the two clauses were measured with the reference scorer, not guessed.
    unrelated = [
        "The Institution shall maintain commercial general liability insurance in the "
        "amount of two million dollars per occurrence.",
        "Each party shall keep the other's confidential information secret and use it "
        "only to perform this Agreement.",
        "Notices under this Agreement must be in writing and delivered by hand or by "
        "registered mail to the address of the recipient.",
        "This Agreement may be signed in counterparts, each of which is an original and "
        "all of which are one instrument.",
    ]

    def filler(document_id: str, count: int, first_path: int) -> list[Observation]:
        return [
            _node(document_id, str(first_path + i), None, unrelated[i % len(unrelated)])
            for i in range(count)
        ]

    absence_nodes = [
        # deal-c: seven unclassified passages, none close to either clause.
        *filler("deal-c", 7, 1),
        # deal-d: an unclassified passage restating assignment (0.308 against the
        # assignment clause: a missed classification, not an absence), and a passage the
        # classifier filed under a third clause type, subcontracting, that mentions
        # consent to delegating duties (0.133: a clause stated inside a block filed
        # elsewhere), plus three unrelated passages.
        _node(
            "deal-d",
            "7",
            None,
            "Neither side may assign or transfer its rights under this contract without "
            "prior approval from the other.",
        ),
        _node(
            "deal-d",
            "12",
            "subcontracting",
            "The Institution must obtain written consent before engaging a subcontractor "
            "to perform its duties.",
        ),
        *filler("deal-d", 3, 1),
        # deal-e has no nodes: its executed copy produced no readable text.
        # deal-h: a node the classifier assigned to governing law, too short to be
        # precedent (it scores 1.0 for governing law: it IS the classifier's finding of
        # the clause), plus two unrelated passages.
        _node("deal-h", "2", law, "Governing law: New York."),
        *filler("deal-h", 2, 3),
    ]
    doc = _base_v06(
        clauses=[
            {
                "id": f"clause.{assign}",
                "taxonomy_id": assign,
                "title": "Assignment",
                "our_standard": std(assign_text, "9"),
            },
            {
                "id": f"clause.{law}",
                "taxonomy_id": law,
                "title": "Governing Law",
                "our_standard": std(law_text, "10"),
            },
        ],
        precedent=precedent,
        documents=documents,
        perspective={"party": "Fixture Co", "counterparty_type": "Fixture Counterparty"},
        absence_nodes=absence_nodes,
    )
    digest = build_digest_v5(doc)
    by_tid = {c["taxonomy_id"]: c for c in digest["clauses"]}
    # Independent expectations, written down from the fixture and not read from the engine.
    # assignment: signed without it are b, c, d, e, f, h (deal-a signed it; deal-g is unsigned);
    # d and e are not verified.
    assert (
        by_tid[assign]["n_signed_without_verified"],
        by_tid[assign]["n_signed_without_possible"],
    ) == (4, 2)
    assert [d["document_id"] for d in by_tid[assign]["verified_absent_deals"]] == [
        "deal-h",
        "deal-f",
        "deal-c",
        "deal-b",
    ]
    # governing law: signed without it are c, d, e, h (a, b and f signed it).
    assert (
        by_tid[law]["n_signed_without_verified"],
        by_tid[law]["n_signed_without_possible"],
    ) == (2, 2)
    assert [d["document_id"] for d in by_tid[law]["verified_absent_deals"]] == ["deal-d", "deal-c"]
    return (
        "014-verified-absence",
        _vector_v06(
            "verified-absence",
            "Verified absence counts (OPF-SPEC §3.5.8, §3.12.2, issue #251): evidence.absence "
            "has one record for every signed deal and clause type with no signed text, split "
            "into absent_verified and possibly_present, and the digest counts them per clause "
            "(n_signed_without_verified, n_signed_without_possible) and lists the verified "
            "deals (verified_absent_deals, latest signed_on first, undated last). The records "
            "are the output of the reference scorer over the deals' signed-text nodes: every "
            "node of a signed copy not filed under the clause is counted in n_checked, and a "
            "node filed under the clause that is too short to be precedent scores 1.0. Two "
            "clauses (assignment, governing law), eight deals. deal-a signed both clauses and "
            "is counted by neither; deal-g has no executed copy and is in no count. "
            "Assignment: deal-b (one node examined, its governing law clause, nothing close), "
            "deal-c (seven unclassified passages, none close), deal-f (opened with our "
            "standard and struck it before signing: it still signed without the clause; one "
            "node examined, its governing law clause) and deal-h (three nodes) are verified "
            "absent (n_signed_without_verified 4); deal-d has an unclassified passage scoring "
            "0.308 against the clause and a passage the classifier filed under a third clause "
            "type (subcontracting) scoring 0.133 (a missed classification, so possibly "
            "present, candidates best first, each saying what the classifier filed it under, "
            "filed_under) and deal-e's executed copy yielded no nodes at all (basis "
            "unverifiable, possibly present, nothing checked): n_signed_without_possible 2. "
            "Governing law: deal-c and deal-d (whose five nodes score below the guard for "
            "this clause) are verified absent, ordered latest signed_on first (deal-d's "
            "month-only 2025-04 is after deal-c's 2025-03-10); deal-e is unverifiable and "
            "deal-h has a node the classifier assigned to the clause that was too short to "
            "be precedent (score 1.0, filed_under the clause itself), so possibly present. "
            "A node filed under the clause that is long enough to be precedent gives its deal "
            "a signed text and no absence record, so no record here cites one. The "
            "guard (0.1) is carried in evidence.absence.guard; a candidate scores at least "
            "that, and a scored record is absent_verified exactly when it has no candidate.",
            doc,
        ),
    )


def main_v06() -> None:
    VECTORS_DIR_V06.mkdir(parents=True, exist_ok=True)
    vectors = build_vectors_v06()
    manifest_entries = []
    for filename, vector in vectors:
        path = VECTORS_DIR_V06 / f"{filename}.json"
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
            "opf_version": OPF_VERSION_V06,
            "engine_version": ENGINE_VERSION,
            "digest_version": DIGEST_VERSION_V5,
        },
        # The stamp text of the 0.4 manifest, carried over with the version
        # updated. "../manifest.json" was the retired 0.3 set's manifest;
        # spec/conformance/README.md carries the algorithm it described.
        "algorithm": (
            "canonical / content_hash / section_digests: exactly as the 0.3 set "
            "(../manifest.json). digest: playbook_engine.digest.build_digest_v5(input) "
            "with the default token_budget, which for opf_version 0.6 is "
            "build_digest_v5 (OPF-SPEC.md §3.12.2): signed variants, refused asks and "
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
    (CONFORMANCE_DIR_V06 / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(vectors)} vectors + manifest.json to {CONFORMANCE_DIR_V06}")


if __name__ == "__main__":
    main_v06()
