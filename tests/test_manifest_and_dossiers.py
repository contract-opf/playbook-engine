"""Hard-rule manifest, critic dossiers and provenance index (OPF 0.5, issue #228).

Every document here is built by the real producer: Observation rows (the
shapes ``observation_builder`` emits) through ``compile_clause_positions`` and
``assemble_playbook``, with Floor invariants written by ``sign_floor_invariant``
(the function behind ``playbook floor sign``). A test that edits a built
document on purpose does so to prove the validator rejects it, and says so.

SECURITY NOTE: all fixtures are synthetic; no real agreements are referenced.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from playbook_engine.canonicalize import canonicalize, compute_section_digests, content_hash
from playbook_engine.cli import cli
from playbook_engine.dossiers import (
    DOSSIER_MIN_BUDGET,
    MAX_EXCERPTS,
    build_dossiers,
    build_manifest,
    build_provenance_index,
    dossier_budget,
    dossier_tokens,
    floor_rule_errors,
    refresh_derived_sections,
    validate_condition,
)
from playbook_engine.floor_candidates import FloorCandidateError, sign_floor_invariant
from playbook_engine.validator import validate_document
from tests.test_digest import _LABEL_TEXTS, _STD, _V_A, _V_B, _labelled_playbook, _v05_obs

ROOT = Path(__file__).parent.parent
NDA_PLAYBOOK = ROOT / "examples" / "nda" / "playbook.opf.json"

_TERM_STD = "The term of this Agreement is two years."


def _invariants(*specs: dict[str, Any]) -> dict[str, Any]:
    """A Floor written by the signing function, one invariant per spec."""
    invariants: list[dict[str, Any]] = []
    for spec in specs:
        invariants = sign_floor_invariant(
            spec.pop("statement"),
            signed_by="Legal Owner",
            signed_at="2026-01-01T00:00:00+00:00",
            existing_invariants=invariants,
            **spec,
        )
    return {"invariants": invariants}


def _playbook(
    rows: list,
    *,
    signed: dict[str, bool],
    floor: dict[str, Any] | None = None,
    with_term_standard: bool = False,
    std_text: str = _STD,
) -> dict[str, Any]:
    """An assembled playbook. ``governing_law`` has our standard (*std_text*); ``term``
    has none unless *with_term_standard*."""
    from playbook_engine.clause_position_compiler import compile_clause_positions
    from playbook_engine.observation_builder import Observation, ObservationCitation
    from playbook_engine.playbook_assembler import assemble_playbook

    def template(tid: str, text: str) -> Observation:
        return Observation(
            observation_id=f"template/{tid}",
            taxonomy_id=tid,
            text_summary=text,
            citation=ObservationCitation(
                document_id="template", version="template", clause_path="4", char_span=None
            ),
            deviation="none",
            risk_delta={"direction": "neutral", "magnitude": "none"},
            provenance="our_paper",
            outcome="signed",
        )

    templates = [template("governing_law", std_text)]
    if with_term_standard:
        templates.append(template("term", _TERM_STD))
    positions, _, _ = compile_clause_positions(rows, templates)
    documents = [
        {
            "document_id": doc_id,
            "provenance": "our_paper",
            "in_scope": True,
            "versions": 3,
            "signed_version": 3 if is_signed else None,
        }
        for doc_id, is_signed in sorted(signed.items())
    ]
    pb = assemble_playbook(
        agreement_type={"id": "nda", "name": "Mutual NDA"},
        baseline={"has_canonical_template": True},
        taxonomy={"source": "custom", "entries": []},
        clause_positions=positions,
        corpus_documents=documents,
        generated_at="2026-01-01T00:00:00Z",
        observations=rows,
        existing_floor=floor,
    )
    result = validate_document(pb)
    assert result.ok, [str(e) for e in result.errors if e.blocking]
    return pb


_O = _v05_obs


def _selection_rows() -> tuple[list, dict[str, bool]]:
    """One clause whose digest lists have MORE entries than the excerpt cap.

    - d1: opened with our standard, conceded to ``_V_A``.
    - d2, d3, d4: signed ``_V_B`` exactly as the counterparty opened it
      (the most-signed variant, so first in digest order).
    - d5: opened non-standard text N1, ended at our standard (changed opening).
    - d6: opened non-standard text N2 and it was struck (changed opening).
    - d7: opened with and kept our standard; refused ask R1 (struck again).
    - d8: the same with refused ask R2.
    Every shape is one ``observation_builder`` emits: a deal that opened with
    our standard and signed different text carries its OUTCOME_OPENING row.
    """
    n1 = "Disputes are governed by the laws of the State of Oregon."
    n2 = "Disputes are governed by the laws of the State of Nevada."
    r1 = "Each party waives any right to a jury trial."
    r2 = "Venue lies exclusively in the courts of Mars."
    rows = [
        _O("d1", _V_A, opened_with="standard"),
        _O("d1", _STD, standard=True, outcome="opening", version=1, opened_with="standard"),
        _O("d2", _V_B, opened_with="non_standard"),
        _O("d3", _V_B, opened_with="non_standard"),
        _O("d4", _V_B, opened_with="non_standard"),
        _O("d5", _STD, standard=True, opened_with="non_standard"),
        _O("d5", n1, outcome="opening", version=1, opened_with="non_standard"),
        _O("d6", n2, outcome="opening", version=1, opened_with="non_standard"),
        _O("d7", _STD, standard=True, opened_with="standard"),
        _O("d7", r1, outcome="proposed_then_reversed", version=2, opened_with="standard"),
        _O("d8", _STD, standard=True, opened_with="standard"),
        _O("d8", r2, outcome="proposed_then_reversed", version=2, opened_with="standard"),
    ]
    return rows, {f"d{i}": True for i in range(1, 9)}


def _record(pb: dict[str, Any], doc_id: str) -> dict[str, Any]:
    (rec,) = [p for p in pb["evidence"]["precedent"] if p["document_id"] == doc_id]
    return rec


# ---------------------------------------------------------------------------
# Presence, shape, determinism
# ---------------------------------------------------------------------------


def test_assembled_playbook_carries_all_three_sections_and_validates() -> None:
    rows, signed = _selection_rows()
    pb = _playbook(rows, signed=signed)
    assert set(pb["manifest"]) == {"hard_rules"}
    assert set(pb["dossiers"]) == {c["id"] for c in pb["evidence"]["clauses"]}
    assert pb["provenance_index"]["compiler"]["name"] == "playbook-engine"
    # Covered by identity: removing the sections changes the content hash.
    stripped = {
        k: v for k, v in pb.items() if k not in ("manifest", "dossiers", "provenance_index")
    }
    assert content_hash(stripped) != pb["identity"]["content_hash"]
    assert pb["identity"]["content_hash"] == content_hash(pb)


def test_rerunning_the_compiler_is_byte_identical() -> None:
    rows, signed = _selection_rows()
    floor = _invariants(
        {"statement": "Never accept uncapped liability.", "taxonomy_id": "governing_law"}
    )
    one = _playbook(rows, signed=signed, floor=floor)
    two = _playbook(list(rows), signed=signed, floor=copy.deepcopy(floor))
    for name in ("manifest", "dossiers", "provenance_index"):
        assert canonicalize(one[name]) == canonicalize(two[name]), name
    # Record order in the document does not matter either.
    shuffled = copy.deepcopy(one)
    shuffled["evidence"]["precedent"].reverse()
    assert build_dossiers(shuffled) == one["dossiers"]
    assert build_provenance_index(shuffled) == one["provenance_index"]


# ---------------------------------------------------------------------------
# Manifest: rules come only from the signed Floor
# ---------------------------------------------------------------------------


def test_manifest_has_one_rule_per_floor_invariant_with_defaults() -> None:
    rows, signed = _selection_rows()
    rows.append(
        _O(
            "d1",
            "The term of this Agreement is three years from the Effective Date.",
            tid="term",
            opened_with="non_standard",
        )
    )
    floor = _invariants(
        {
            "statement": "Governing law, if present, must stay Delaware or New York.",
            "invariant_id": "law-holds",
            "taxonomy_id": "governing_law",
            "required_presence": True,
            "condition": {
                "type": "required_phrases",
                "phrases": ["State of New York"],
                "match": "any",
            },
            "permissible_proof": ["a signed amendment from the GC", "  "],
        },
        {
            "statement": "Do not concede on the term.",
            "invariant_id": "term-holds",
            "taxonomy_id": "term",
        },
        {
            "statement": "Nothing about a clause we have no evidence for.",
            "invariant_id": "elsewhere",
            "taxonomy_id": "ghost_clause",
        },
        {"statement": "A rule that names no clause.", "invariant_id": "free"},
    )
    pb = _playbook(rows, signed=signed, floor=floor)
    rules = {r["rule_id"]: r for r in pb["manifest"]["hard_rules"]}
    assert [r["rule_id"] for r in pb["manifest"]["hard_rules"]] == [
        "law-holds",
        "term-holds",
        "elsewhere",
        "free",
    ]
    law = rules["law-holds"]
    assert law == {
        "rule_id": "law-holds",
        "clause_id": "clause.governing_law",
        "taxonomy_id": "governing_law",
        "statement": "Governing law, if present, must stay Delaware or New York.",
        "required_presence": True,
        "condition": {"type": "required_phrases", "phrases": ["State of New York"], "match": "any"},
        "permissible_proof": ["a signed amendment from the GC"],
        "fallback_language": _STD,
    }
    # No x_ keys: presence is not demanded, the rule is judged, no proof.
    term = rules["term-holds"]
    assert (term["required_presence"], term["condition"], term["permissible_proof"]) == (
        False,
        "judged",
        [],
    )
    # The clause has evidence but no standard text: nothing to insert.
    assert term["clause_id"] == "clause.term" and term["fallback_language"] is None
    # A clause the corpus has no evidence for, and a rule naming none.
    assert (rules["elsewhere"]["clause_id"], rules["elsewhere"]["taxonomy_id"]) == (
        None,
        "ghost_clause",
    )
    assert (rules["free"]["clause_id"], rules["free"]["taxonomy_id"]) == (None, None)
    assert rules["free"]["fallback_language"] is None
    # Every rule has a predicate spec or "judged".
    assert all(
        r["condition"] == "judged" or isinstance(r["condition"], dict) for r in rules.values()
    )


def test_manifest_never_reads_precedent_counts() -> None:
    """The same Floor over very different evidence gives the same rules; with
    no Floor there are none, whatever the corpus holds."""
    floor = _invariants({"statement": "Hold governing law.", "taxonomy_id": "governing_law"})
    rows, signed = _selection_rows()
    few = _playbook([_O("d1", _STD, standard=True)], signed={"d1": True}, floor=floor)
    many = _playbook(rows, signed=signed, floor=floor)
    assert few["manifest"] == many["manifest"]
    assert _playbook(rows, signed=signed)["manifest"] == {"hard_rules": []}


# ---------------------------------------------------------------------------
# Predicate specs
# ---------------------------------------------------------------------------

_GOOD = [
    "judged",
    {"type": "required_phrases", "phrases": ["a", "b"]},
    {"type": "required_phrases", "phrases": ["a"], "match": "all"},
    {"type": "numeric_bound", "pattern": r"\$([0-9,]+)", "max": 50000},
    {"type": "numeric_bound", "pattern": r"(\d+) days", "min": 30, "max": 90, "unit": "days"},
    {"type": "cross_reference", "clause_id": "clause.survival"},
]
_BAD = [
    None,
    "other",
    {},
    {"type": "mystery"},
    {"type": "required_phrases", "phrases": []},
    {"type": "required_phrases", "phrases": [" "]},
    {"type": "required_phrases", "phrases": ["a"], "match": "most"},
    {"type": "required_phrases", "phrases": ["a"], "extra": 1},
    {"type": "numeric_bound", "pattern": r"(\d+)"},
    {"type": "numeric_bound", "pattern": r"\d+", "max": 1},
    {"type": "numeric_bound", "pattern": r"(\d+)(\d+)", "max": 1},
    {"type": "numeric_bound", "pattern": "(", "max": 1},
    {"type": "numeric_bound", "pattern": r"(\d+)", "min": 5, "max": 1},
    {"type": "numeric_bound", "pattern": r"(\d+)", "max": True},
    {"type": "cross_reference", "clause_id": ""},
    {"type": "cross_reference"},
]


@pytest.mark.parametrize("spec", _GOOD)
def test_well_formed_conditions_are_accepted(spec: Any) -> None:
    assert validate_condition(spec) is None


@pytest.mark.parametrize("spec", _BAD, ids=lambda s: json.dumps(s))
def test_malformed_conditions_are_refused(spec: Any) -> None:
    assert validate_condition(spec)
    if spec is not None:  # None means "unstated" to the signing function
        with pytest.raises(FloorCandidateError):
            sign_floor_invariant("x", signed_by="Legal Owner", condition=spec)


def test_the_validator_refuses_a_malformed_manifest_key_on_the_floor() -> None:
    rows, signed = _selection_rows()
    pb = _playbook(rows, signed=signed, floor=_invariants({"statement": "Hold it."}))
    bad = copy.deepcopy(pb)
    del bad["identity"]
    bad["floor"]["invariants"][0]["x_condition"] = {"type": "required_phrases", "phrases": []}
    bad["floor"]["invariants"][0]["x_required_presence"] = "yes"
    messages = [str(e) for e in validate_document(bad).errors if e.blocking]
    assert any("x_condition" in m for m in messages)
    assert any("x_required_presence" in m for m in messages)


# ---------------------------------------------------------------------------
# A hard rule names its clause
# ---------------------------------------------------------------------------

_PHRASES = {"type": "required_phrases", "phrases": ["State of New York"]}


@pytest.mark.parametrize(
    "extra",
    [
        {"x_required_presence": True},
        {"x_condition": _PHRASES},
        {"x_condition": {"type": "cross_reference", "clause_id": "clause.term"}},
        {"x_required_presence": True, "x_condition": _PHRASES, "x_taxonomy_id": "  "},
    ],
    ids=["presence", "required_phrases", "cross_reference", "blank-clause"],
)
def test_presence_and_predicate_rules_must_name_a_clause(extra: dict[str, Any]) -> None:
    unanchored = {"id": "r", "statement": "Hold it.", **extra}
    assert any("x_taxonomy_id" in why for why in floor_rule_errors(unanchored))
    assert not floor_rule_errors({**unanchored, "x_taxonomy_id": "governing_law"})


@pytest.mark.parametrize(
    "extra",
    [
        {},
        {"x_required_presence": False},
        {"x_condition": "judged"},
        {"x_permissible_proof": ["a waiver"]},
    ],
)
def test_a_judged_rule_that_demands_nothing_needs_no_clause(extra: dict[str, Any]) -> None:
    assert floor_rule_errors({"id": "r", "statement": "Hold it.", **extra}) == []


def test_signing_refuses_a_presence_or_predicate_rule_with_no_clause() -> None:
    with pytest.raises(FloorCandidateError, match="x_taxonomy_id"):
        sign_floor_invariant("Must be present.", signed_by="Legal Owner", required_presence=True)
    with pytest.raises(FloorCandidateError, match="x_taxonomy_id"):
        sign_floor_invariant("Must say it.", signed_by="Legal Owner", condition=_PHRASES)
    # With a clause, or demanding nothing, it signs.
    ok = sign_floor_invariant(
        "Must be present.", signed_by="Legal Owner", required_presence=True, taxonomy_id="term"
    )
    assert ok[0]["x_taxonomy_id"] == "term"
    assert sign_floor_invariant("Judge it.", signed_by="Legal Owner", condition="judged")


def test_floor_sign_cli_refuses_requires_presence_with_no_clause(tmp_path: Path) -> None:
    rows, signed = _selection_rows()
    out = _write_out_dir(tmp_path, _playbook(rows, signed=signed))
    before = (out / "playbook.opf.json").read_bytes()
    for flags in (
        ["--requires-presence"],
        ["--requires-presence", "--condition", json.dumps(_PHRASES)],
        ["--condition", json.dumps({"type": "cross_reference", "clause_id": "clause.term"})],
    ):
        result = CliRunner().invoke(
            cli,
            [
                "floor", "sign", str(out),
                "--statement", "Some clause must be present.",
                "--signed-by", "Legal Owner",
                *flags,
            ],
        )  # fmt: skip
        assert result.exit_code != 0, result.output
        assert "x_taxonomy_id" in result.output
        assert (out / "playbook.opf.json").read_bytes() == before


def test_the_validator_refuses_an_unanchored_presence_rule_on_the_floor() -> None:
    rows, signed = _selection_rows()
    pb = _playbook(rows, signed=signed, floor=_invariants({"statement": "Hold it."}))
    bad = copy.deepcopy(pb)
    del bad["identity"]
    bad["floor"]["invariants"][0]["x_required_presence"] = True
    assert any("x_taxonomy_id" in str(e) for e in validate_document(bad).errors if e.blocking)


def test_the_schema_refuses_a_manifest_rule_with_no_clause_that_demands_presence() -> None:
    rows, signed = _selection_rows()
    pb = _playbook(rows, signed=signed, floor=_invariants({"statement": "Hold it."}))
    bad = copy.deepcopy(pb)
    del bad["identity"]
    bad["manifest"]["hard_rules"][0]["required_presence"] = True
    _assert_schema_refuses_unanchored_rule(bad)
    bad["manifest"]["hard_rules"][0]["required_presence"] = False
    bad["manifest"]["hard_rules"][0]["condition"] = _PHRASES
    _assert_schema_refuses_unanchored_rule(bad)


def _assert_schema_refuses_unanchored_rule(doc: dict[str, Any]) -> None:
    """The JSON schema itself (not only the recomputation check) refuses the rule."""
    errors = [
        e
        for e in validate_document(doc).errors
        if e.blocking and e.path == "manifest.hard_rules.0.taxonomy_id"
    ]
    assert errors, "no blocking error at manifest.hard_rules.0.taxonomy_id"
    assert any(str(e.message).startswith("Schema:") for e in errors), [str(e) for e in errors]


# ---------------------------------------------------------------------------
# Excerpt selection (deterministic, capped, verbatim)
# ---------------------------------------------------------------------------


def test_excerpts_are_a_concession_then_a_changed_opening_and_capped_at_two() -> None:
    rows, signed = _selection_rows()
    pb = _playbook(rows, signed=signed)
    digest = pb["digest"]["clauses"][0]
    # More entries than the cap in every list the order reads.
    assert len(digest["signed_variants"]) == 2
    assert len(digest["changed_openings"]) == 2 and len(digest["refused_asks"]) == 2
    # The most-signed variant leads the digest, but is not a concession.
    assert digest["signed_variants"][0]["n_from_standard"] == 0
    (dossier,) = pb["dossiers"].values()
    first, second = dossier["excerpts"]
    assert len(dossier["excerpts"]) == MAX_EXCERPTS
    # 1: the first signed-variants group with n_from_standard > 0: d1's concession.
    d1 = _record(pb, "d1")
    assert first == {
        "precedent_id": d1["id"],
        "kind": "signed_variant",
        "opening": _STD,
        "signed": _V_A,
        "outcome": "signed",
    }
    # 2: the first changed-openings group, in digest order.
    assert second["kind"] == "changed_opening"
    assert second["precedent_id"] in digest["changed_openings"][0]["precedent_ids"]
    assert second["opening"] == "Disputes are governed by the laws of the State of Oregon." or (
        second["opening"] == "Disputes are governed by the laws of the State of Nevada."
    )


def test_a_struck_opening_has_no_signed_text() -> None:
    rows, signed = _selection_rows()
    # Make the struck opening (d6) the first changed-openings group by removing d5.
    rows = [r for r in rows if r.citation.document_id != "d5"]
    pb = _playbook(rows, signed={k: v for k, v in signed.items() if k != "d5"})
    (dossier,) = pb["dossiers"].values()
    struck = dossier["excerpts"][1]
    assert struck["precedent_id"] == _record(pb, "d6")["id"]
    assert (struck["kind"], struck["signed"], struck["outcome"]) == (
        "changed_opening",
        None,
        "struck_before_signing",
    )
    assert struck["opening"] == "Disputes are governed by the laws of the State of Nevada."


def test_without_a_concession_or_opening_the_digest_order_fills_the_excerpts() -> None:
    """Rule 3: signed variants in digest order, then refused asks."""
    r1 = "Each party waives any right to a jury trial."
    rows = [
        _O("d1", _V_B, opened_with="non_standard"),
        _O("d2", _V_B, opened_with="non_standard"),
        _O("d3", _V_A, opened_with="non_standard"),
        _O("d4", _STD, standard=True, opened_with="standard"),
        _O("d4", r1, outcome="proposed_then_reversed", version=2, opened_with="standard"),
    ]
    pb = _playbook(rows, signed={f"d{i}": True for i in range(1, 5)})
    (dossier,) = pb["dossiers"].values()
    assert [e["kind"] for e in dossier["excerpts"]] == ["signed_variant", "signed_variant"]
    # Two-deal variant first (digest order). No signed_at is recorded, so the
    # lowest precedent_id picks its record: d2's, although d1 is the lower
    # document (the record the digest cites).
    d1, d2 = _record(pb, "d1"), _record(pb, "d2")
    assert d1["document_id"] < d2["document_id"] and d2["id"] < d1["id"]
    assert pb["digest"]["clauses"][0]["signed_variants"][0]["ref"]["document_id"] == "d1"
    assert dossier["excerpts"][0]["signed"] == _V_B
    assert dossier["excerpts"][0]["precedent_id"] == d2["id"]
    assert dossier["excerpts"][0]["opening"] is None  # signed text alone
    assert dossier["excerpts"][1]["signed"] == _V_A

    # Only a refused ask to show: the excerpt is the ask, then what was signed.
    only_ask = _playbook(rows[3:], signed={"d4": True})
    (dossier,) = only_ask["dossiers"].values()
    (ask,) = dossier["excerpts"]
    assert (ask["kind"], ask["opening"], ask["signed"], ask["outcome"]) == (
        "refused_ask",
        r1,
        _STD,
        "ask_refused",
    )


def test_the_latest_signed_at_picks_a_groups_record_before_the_precedent_id() -> None:
    """OPF-SPEC §3.12.3: latest ``signed_at`` first, then the lowest ``precedent_id``.

    The reference compiler records no ``signed_at``, so this test sets one on
    a built document on purpose (a producer that does record signing dates)
    and reads the dossiers' reference construction directly.
    """
    rows = [
        _O("d1", _V_B, opened_with="non_standard"),
        _O("d2", _V_B, opened_with="non_standard"),
        _O("d3", _V_B, opened_with="non_standard"),
    ]
    pb = _playbook(rows, signed={"d1": True, "d2": True, "d3": True})
    by_doc = {p["document_id"]: p for p in pb["evidence"]["precedent"]}
    lowest = min(by_doc.values(), key=lambda p: p["id"])
    assert lowest["document_id"] == "d2"
    (dossier,) = build_dossiers(pb).values()
    assert dossier["excerpts"][0]["precedent_id"] == lowest["id"]

    dated = copy.deepcopy(pb)
    for p in dated["evidence"]["precedent"]:
        p["signed_at"] = {"d1": "2025-03-01", "d2": "2025-01-01", "d3": "2025-03-01"}[
            p["document_id"]
        ]
    # d1 and d3 tie on the latest date; the lower precedent_id of the two wins.
    tied = sorted((by_doc["d1"]["id"], by_doc["d3"]["id"]))
    (dossier,) = build_dossiers(dated).values()
    assert dossier["excerpts"][0]["precedent_id"] == tied[0] != lowest["id"]


def test_a_concession_excerpt_is_the_concession_in_a_mixed_group() -> None:
    """Rule 1 takes its record among the members that opened with our standard.

    a0 signed ``_V_A`` exactly as the counterparty proposed it; d1 opened
    with our standard and conceded to the same ``_V_A`` (its OPENING row
    carries our standard). a0 is both the lower document and the lower
    precedent id, so any order that ignored the opening would show it.
    """
    rows = [
        _O("a0", _V_A, opened_with="non_standard"),
        _O("d1", _V_A, opened_with="standard"),
        _O("d1", _STD, standard=True, outcome="opening", version=1, opened_with="standard"),
    ]
    pb = _playbook(rows, signed={"a0": True, "d1": True})
    (group,) = pb["digest"]["clauses"][0]["signed_variants"]
    assert (group["n_deals"], group["n_from_standard"], group["n_unchanged"]) == (2, 1, 1)
    a0, d1 = _record(pb, "a0"), _record(pb, "d1")
    assert a0["document_id"] < d1["document_id"] and a0["id"] < d1["id"]
    (dossier,) = pb["dossiers"].values()
    # The opening -> signed pair of the concession; the group is not shown twice.
    assert dossier["excerpts"] == [
        {
            "precedent_id": d1["id"],
            "kind": "signed_variant",
            "opening": _STD,
            "signed": _V_A,
            "outcome": "signed",
        }
    ]


def test_selection_follows_the_collapsed_equivalent_entry(tmp_path: Path) -> None:
    """The equivalent variants stand for their groups in group order: the
    first with a concession on record is the first excerpt."""
    pb = _labelled_playbook(
        tmp_path,
        {
            "eq1": "equivalent",
            "eq2": "equivalent",
            "less": "less_protective",
            "diff": "different_concept",
            "more": "more_protective",
        },
    )
    digest = pb["digest"]["clauses"][0]
    collapsed = digest["signed_variants"][-1]
    assert collapsed["label"] == "equivalent" and collapsed["n_from_standard"] == 1
    (dossier,) = pb["dossiers"].values()
    first, second = dossier["excerpts"]
    assert first["signed"] == _LABEL_TEXTS["eq2"] and first["opening"] == _STD
    assert first["precedent_id"] in collapsed["precedent_ids"]
    assert second["kind"] == "changed_opening"
    assert second["opening"] == _LABEL_TEXTS["diff"]


def test_excerpt_text_is_verbatim_not_the_digest_summary() -> None:
    long_text = (
        _V_A + " " + " ".join(f"Clause sentence number {i} binds the parties." for i in range(30))
    )
    assert len(long_text) > 300
    rows = [_O("d1", long_text, opened_with="non_standard")]
    pb = _playbook(rows, signed={"d1": True})
    (dossier,) = pb["dossiers"].values()
    assert dossier["excerpts"][0]["signed"] == long_text
    assert pb["digest"]["clauses"][0]["signed_variants"][0]["text"] != long_text
    assert (dossier["n_omitted"], dossier["omitted_precedent_ids"]) == (0, [])


# ---------------------------------------------------------------------------
# Bounded size: whole excerpts are dropped, no text is ever cut
# ---------------------------------------------------------------------------


def _sentences(label: str, n: int) -> str:
    return " ".join(f"{label} sentence {i} of a long clause binds the parties." for i in range(n))


def _concession_and_changed_opening(text_a: str, text_b: str) -> tuple[list, dict[str, bool]]:
    """d1 conceded from our standard to *text_a*; d5 opened *text_b* and ended at our standard."""
    rows = [
        _O("d1", text_a, opened_with="standard"),
        _O("d1", _STD, standard=True, outcome="opening", version=1, opened_with="standard"),
        _O("d5", _STD, standard=True, opened_with="non_standard"),
        _O("d5", text_b, outcome="opening", version=1, opened_with="non_standard"),
    ]
    return rows, {"d1": True, "d5": True}


def test_a_second_excerpt_over_the_budget_is_dropped_whole_and_named() -> None:
    long_a = _sentences("Alpha", 40)
    long_b = _sentences("Bravo", 40)
    rows, signed = _concession_and_changed_opening(long_a, long_b)
    pb = _playbook(rows, signed=signed)
    (dossier,) = pb["dossiers"].values()
    # The two excerpts together are over the 1,000-token budget; the second
    # (the changed opening) is dropped WHOLE, the first kept whole.
    assert (len(long_a) + len(long_b)) // 4 > DOSSIER_MIN_BUDGET
    assert [e["kind"] for e in dossier["excerpts"]] == ["signed_variant"]
    assert dossier["excerpts"][0]["signed"] == long_a
    assert dossier["excerpts"][0]["opening"] == _STD
    assert dossier_tokens(dossier) <= DOSSIER_MIN_BUDGET
    dropped = _record(pb, "d5")["id"]
    assert dossier["n_omitted"] == 1 and dossier["omitted_precedent_ids"] == [dropped]
    # The dropped record still holds its text whole, reachable by its id.
    assert _record(pb, "d5")["opening_text"]["text"] == long_b
    assert validate_document(pb).ok


def test_a_floor_rule_is_kept_before_a_second_excerpt() -> None:
    """A rule that fits beside the first excerpt stays listed; the second
    excerpt, which does not fit beside them, is the part dropped (whole)."""
    long_a, long_b = _sentences("Alpha", 40), _sentences("Bravo", 40)
    rows, signed = _concession_and_changed_opening(long_a, long_b)
    statement = "Governing law must stay in New York. " + "It never moves. " * 20
    floor = _invariants(
        {"statement": statement, "invariant_id": "gl-ny", "taxonomy_id": "governing_law"}
    )
    pb = _playbook(rows, signed=signed, floor=floor)
    (dossier,) = pb["dossiers"].values()
    assert [(r["rule_id"], r["statement"]) for r in dossier["floor_rules"]] == [
        ("gl-ny", statement)
    ]
    assert [e["signed"] for e in dossier["excerpts"]] == [long_a]
    assert dossier["omitted_precedent_ids"] == [_record(pb, "d5")["id"]]
    assert dossier_tokens(dossier) <= DOSSIER_MIN_BUDGET
    # Without the rule the two excerpts would still not fit: dropping the rule
    # instead would not have kept the second excerpt.
    both = copy.deepcopy(dossier)
    both["floor_rules"] = []
    both["excerpts"].append({"precedent_id": "x", "signed": long_b})
    assert dossier_tokens(both) > DOSSIER_MIN_BUDGET
    assert validate_document(pb).ok


def test_a_single_excerpt_over_the_budget_is_kept_whole() -> None:
    huge = _sentences("Enormous", 2_000)
    rows = [
        _O("d1", huge, opened_with="non_standard"),
        _O(
            "d1",
            huge.replace("Enormous", "Vast"),
            outcome="opening",
            version=1,
            opened_with="non_standard",
        ),
    ]
    floor = _invariants(
        *[
            {
                "statement": f"Rule number {i}: " + "must hold. " * 300,
                "invariant_id": f"rule-{i}",
                "taxonomy_id": "governing_law",
                "rationale": "because " * 300,
            }
            for i in range(6)
        ]
    )
    pb = _playbook(rows, signed={"d1": True}, floor=floor)
    (dossier,) = pb["dossiers"].values()
    # The one excerpt is the record's own text, whole, although it alone is over budget.
    (excerpt,) = dossier["excerpts"]
    assert excerpt["signed"] == huge
    assert excerpt["opening"] == _record(pb, "d1")["opening_text"]["text"]
    assert dossier_tokens(dossier) > dossier_budget(dossier["our_standard"]["text"])
    assert (dossier["n_omitted"], dossier["omitted_precedent_ids"]) == (0, [])
    # Still over budget with its one excerpt, so every listed Floor rule is dropped
    # whole (n_floor_rules counts all six); the manifest states each verbatim.
    assert dossier["n_floor_rules"] == 6 and dossier["floor_rules"] == []
    assert [r["statement"] for r in pb["manifest"]["hard_rules"]] == [
        i["statement"] for i in floor["invariants"]
    ]
    assert validate_document(pb).ok


def test_the_budget_scales_with_our_standard() -> None:
    assert dossier_budget(None) == DOSSIER_MIN_BUDGET
    assert dossier_budget("x" * 400) == DOSSIER_MIN_BUDGET  # 3 * 100 tokens < 1,000
    assert dossier_budget("x" * 4_000) == 3_000  # 3 * 1,000 tokens
    # A clause with a long standard keeps both excerpts although together they
    # exceed 1,000 tokens; the same excerpts beside a short standard lose one.
    long_a, long_b = _sentences("Alpha", 40), _sentences("Bravo", 40)
    rows = [
        _O("d1", long_a, opened_with="non_standard"),
        _O("d2", long_b, opened_with="non_standard"),
    ]
    signed = {"d1": True, "d2": True}
    short = _playbook(rows, signed=signed)
    big_std = _STD + " " + _sentences("Standard", 60)
    assert dossier_budget(big_std) > DOSSIER_MIN_BUDGET
    scaled = _playbook(rows, signed=signed, std_text=big_std)
    (short_d,) = short["dossiers"].values()
    (scaled_d,) = scaled["dossiers"].values()
    assert len(short_d["excerpts"]) == 1 and short_d["n_omitted"] == 1
    assert len(scaled_d["excerpts"]) == 2 and scaled_d["n_omitted"] == 0
    assert dossier_tokens(scaled_d) > DOSSIER_MIN_BUDGET
    assert dossier_tokens(scaled_d) <= dossier_budget(scaled_d["our_standard"]["text"])
    assert scaled_d["our_standard"]["text"] == big_std
    assert validate_document(scaled).ok


@pytest.mark.parametrize("statement_chars", [1_244, 1_318, 4_000])
def test_long_statement_derived_rule_ids_are_listed_whole(
    tmp_path: Path, statement_chars: int
) -> None:
    """``floor sign`` without ``--id`` slugs the whole statement into the rule id.

    Nothing is cut: the dossier drops whole listed rules, last first, until the
    rest fit beside its first excerpt, and keeps its second excerpt, which fits
    beside them; every rule still listed keeps its id and statement whole (each
    still names its manifest rule), and the manifest states all three. Validate
    and a re-projection over the signed Floor pass.
    """
    from tests.test_floor_candidates import _write_taxonomy_config

    rows, signed = _selection_rows()
    out = _write_out_dir(tmp_path, _playbook(rows, signed=signed))
    config = _write_taxonomy_config(tmp_path, ["governing_law"])
    statements = []
    for n in range(3):
        head = f"Rule {n} governing law must stay in New York "
        statement = head + "and nowhere else " * ((statement_chars - len(head)) // 17)
        statements.append(statement)
        result = CliRunner().invoke(
            cli,
            [
                "floor", "sign", str(out),
                "--statement", statement,
                "--signed-by", "Legal Owner",
                "--clause", "governing_law",
                "--config", str(config),
            ],
        )  # fmt: skip
        assert result.exit_code == 0, result.output
    doc = json.loads((out / "playbook.opf.json").read_text(encoding="utf-8"))
    ids = [r["rule_id"] for r in doc["manifest"]["hard_rules"]]
    assert [r["statement"] for r in doc["manifest"]["hard_rules"]] == statements
    assert all(len(rule_id) > 1_000 for rule_id in ids)
    (dossier,) = doc["dossiers"].values()
    assert dossier["n_floor_rules"] == 3
    listed = dossier["floor_rules"]
    # Rules are dropped whole, last first: what remains listed is a whole prefix
    # of the Floor's rules (one rule of ~1,250 chars fits, none of 4,000), and
    # the dossier fits. The second excerpt fits beside them, so it is kept: an
    # excerpt is never dropped when it fits beside the rules left listed.
    assert len(listed) == (1 if statement_chars < 4_000 else 0)
    assert [(r["rule_id"], r["statement"]) for r in listed] == list(
        zip(ids, statements, strict=True)
    )[: len(listed)]
    assert len(dossier["excerpts"]) == 2 and dossier["n_omitted"] == 0
    assert dossier_tokens(dossier) <= dossier_budget(dossier["our_standard"]["text"])
    assert validate_document(doc).ok, [str(e) for e in validate_document(doc).errors]
    # Re-projecting over the signed Floor succeeds and reproduces the sections.
    again = _playbook(rows, signed=signed, floor=doc["floor"])
    for name in ("manifest", "dossiers", "provenance_index"):
        assert canonicalize(again[name]) == canonicalize(doc[name]), name


def test_long_floor_rules_are_dropped_whole_from_a_dossier_with_no_excerpt(
    tmp_path: Path,
) -> None:
    """A clause with only standard records (no excerpt) and three long Floor rules
    that together exceed its budget drops the last listed rule whole and fits:
    only a dossier holding its single kept excerpt may exceed its budget. The
    manifest still states all three rules, and the scorecard counts no dossier
    over budget."""
    from playbook_engine.scorecard import build_scorecard

    specs = [
        {
            "statement": f"Rule {n} governing law stays in New York. " + "Nowhere else. " * 100,
            "invariant_id": f"gl-{n}",
            "taxonomy_id": "governing_law",
        }
        for n in range(3)
    ]
    statements = [s["statement"] for s in specs]
    rows = [_O("d1", _STD, standard=True, opened_with="standard")]
    pb = _playbook(rows, signed={"d1": True}, floor=_invariants(*specs))
    (dossier,) = pb["dossiers"].values()
    assert dossier["excerpts"] == [] and dossier["n_omitted"] == 0
    assert dossier["n_floor_rules"] == 3
    assert [r["statement"] for r in dossier["floor_rules"]] == statements[:2]
    budget = dossier_budget(dossier["our_standard"]["text"])
    assert dossier_tokens(dossier) <= budget
    # With all three rules listed it would be over budget.
    all_listed = copy.deepcopy(dossier)
    all_listed["floor_rules"].append({"rule_id": "gl-2", "statement": statements[2]})
    assert dossier_tokens(all_listed) > budget
    assert [r["statement"] for r in pb["manifest"]["hard_rules"]] == statements
    assert validate_document(pb).ok
    card = build_scorecard(_write_out_dir(tmp_path, pb))["dossiers"]
    assert card["over_budget_single_excerpt"] == 0
    assert "over_budget_fixed_parts" not in card
    assert card["cut_texts"] == 0


def test_identifiers_are_never_cut_even_when_they_alone_exceed_the_budget() -> None:
    """A clause whose own taxonomy id (and so clause id) is over the budget still
    gets a dossier: identifiers whole, the first excerpt kept, the rest dropped,
    then its Floor rule dropped whole (still over budget with one excerpt)."""
    long_tid = "clause_type_" + "x" * 5_000
    rows = [
        _O("d1", _V_A, tid=long_tid, opened_with="non_standard"),
        _O("d2", _V_B, tid=long_tid, opened_with="non_standard"),
    ]
    floor = _invariants({"statement": "Hold it.", "invariant_id": "hold", "taxonomy_id": long_tid})
    pb = _playbook(rows, signed={"d1": True, "d2": True}, floor=floor)
    (clause_id,) = [c["id"] for c in pb["evidence"]["clauses"] if c["taxonomy_id"] == long_tid]
    dossier = pb["dossiers"][clause_id]
    assert dossier["taxonomy_id"] == long_tid and dossier["clause_id"] == clause_id
    assert dossier["n_floor_rules"] == 1 and dossier["floor_rules"] == []
    assert len(dossier["excerpts"]) == 1 and dossier["n_omitted"] == 1
    assert dossier_tokens(dossier) > DOSSIER_MIN_BUDGET
    assert validate_document(pb).ok
    rows = pb["provenance_index"]["dossiers"][clause_id]
    assert [(r["precedent_id"], r["omitted"]) for r in rows] == [
        (dossier["excerpts"][0]["precedent_id"], False),
        (dossier["omitted_precedent_ids"][0], True),
    ]


def _with_extra_deals(rows: list, signed: dict[str, bool], n: int, *, mix: bool) -> dict[str, Any]:
    """The selection fixture plus *n* synthetic deals.

    ``mix=False``: each is a counterparty variant signed as proposed (a new
    distinct text, no concession, no changed opening), so neither selection
    step can prefer it. ``mix=True``: concessions, signed-as-proposed variants
    and changed openings in turn, which DO displace the selection.
    """
    more = list(rows)
    more_signed = dict(signed)
    for i in range(n):
        doc = f"x{i:04d}"
        more_signed[doc] = True
        kind = i % 3 if mix else 1
        if kind == 0:  # a concession of a new variant
            text = f"Disputes are decided under the laws of Utopia section {i}."
            more.append(_O(doc, text, opened_with="standard"))
            more.append(
                _O(
                    doc,
                    _STD,
                    standard=True,
                    outcome="opening",
                    version=1,
                    opened_with="standard",
                )
            )
        elif kind == 1:  # a counterparty variant signed as proposed
            more.append(
                _O(doc, f"Venue lies in the courts of district {i}.", opened_with="non_standard")
            )
        else:  # a changed opening
            more.append(_O(doc, _STD, standard=True, opened_with="non_standard"))
            more.append(
                _O(
                    doc,
                    f"Law number {i} governs.",
                    outcome="opening",
                    version=1,
                    opened_with="non_standard",
                )
            )
    return _playbook(more, signed=more_signed)


def test_adding_precedents_that_leave_the_selection_unchanged_leaves_the_dossier_identical() -> (
    None
):
    """The ticket's invariant: adding N precedents never changes the dossier's size."""
    rows, signed = _selection_rows()
    base = _playbook(rows, signed=signed)
    (clause_id,) = base["dossiers"]
    base_d = base["dossiers"][clause_id]
    assert len(base_d["excerpts"]) == MAX_EXCERPTS
    for n in (5, 50, 500):
        pb = _with_extra_deals(rows, signed, n, mix=False)
        d = pb["dossiers"][clause_id]
        assert [e["precedent_id"] for e in d["excerpts"]] == [
            e["precedent_id"] for e in base_d["excerpts"]
        ]
        assert dossier_tokens(d) == dossier_tokens(base_d)
        assert canonicalize(d) == canonicalize(base_d)


def test_adding_precedents_that_displace_the_selection_keeps_the_budget_and_two_excerpts() -> None:
    rows, signed = _selection_rows()
    base = _playbook(rows, signed=signed)
    (clause_id,) = base["dossiers"]
    displaced = False
    for n in (12, 90, 300):
        pb = _with_extra_deals(rows, signed, n, mix=True)
        d = pb["dossiers"][clause_id]
        assert len(d["excerpts"]) == MAX_EXCERPTS
        assert dossier_tokens(d) <= DOSSIER_MIN_BUDGET
        displaced = displaced or [e["precedent_id"] for e in d["excerpts"]] != [
            e["precedent_id"] for e in base["dossiers"][clause_id]["excerpts"]
        ]
    assert displaced, "the mixed deals were meant to change which excerpts appear"


# ---------------------------------------------------------------------------
# Scorecard counts
# ---------------------------------------------------------------------------


def test_scorecard_counts_dropped_over_budget_and_cut_texts(tmp_path: Path) -> None:
    from playbook_engine.scorecard import build_scorecard

    # One clause that drops its second excerpt whole.
    long_a, long_b = _sentences("Alpha", 40), _sentences("Bravo", 40)
    rows, signed = _concession_and_changed_opening(long_a, long_b)
    dropping = _playbook(rows, signed=signed)
    out = _write_out_dir(tmp_path, dropping)
    card = build_scorecard(out)["dossiers"]
    assert card["count"] == 1 and card["max_budget"] == DOSSIER_MIN_BUDGET
    assert (
        card["cut_texts"],
        card["excerpts_dropped"],
        card["over_budget_single_excerpt"],
    ) == (0, 1, 0)
    assert "over_budget_fixed_parts" not in card
    # A single excerpt over the budget is counted, still with no cut text.
    huge = _sentences("Enormous", 2_000)
    single = _playbook([_O("d1", huge, opened_with="non_standard")], signed={"d1": True})
    card = build_scorecard(_write_out_dir(tmp_path / "single", single))["dossiers"]
    assert (
        card["cut_texts"],
        card["excerpts_dropped"],
        card["over_budget_single_excerpt"],
    ) == (0, 0, 1)
    # A tampered copy whose excerpt is a prefix cut of the record's text counts as cut.
    cut = copy.deepcopy(dropping)
    (dossier,) = cut["dossiers"].values()
    dossier["excerpts"][0]["signed"] = dossier["excerpts"][0]["signed"][:50] + "…"
    card = build_scorecard(_write_out_dir(tmp_path / "cut", cut))["dossiers"]
    assert card["cut_texts"] == 1


# ---------------------------------------------------------------------------
# Provenance index
# ---------------------------------------------------------------------------


def test_provenance_index_names_each_selected_excerpt_and_its_source_documents() -> None:
    rows, signed = _selection_rows()
    pb = _playbook(rows, signed=signed)
    index = pb["provenance_index"]
    (dossier,) = pb["dossiers"].values()
    (clause_id,) = pb["dossiers"]
    selected = index["dossiers"][clause_id]
    assert dossier["n_omitted"] == 0
    assert [s["precedent_id"] for s in selected] == [e["precedent_id"] for e in dossier["excerpts"]]
    assert [s["kind"] for s in selected] == [e["kind"] for e in dossier["excerpts"]]
    assert [s["omitted"] for s in selected] == [False] * len(selected)
    cited = {s["document_id"] for s in selected}
    assert {d["document_id"] for d in index["documents"]} == cited
    assert index["compiler"] == {
        "name": pb["compiler"]["name"],
        "version": pb["compiler"]["version"],
    }
    # No corpus files were recorded in this fixture, so no snapshot and no file hashes.
    assert index["corpus_manifest_hash"] is None
    assert all(d["version_files"] == [] for d in index["documents"])


VECTOR_008 = ROOT / "spec" / "conformance" / "0.5" / "vectors" / "008-hard-rules-and-dossiers.json"


def _assert_omitted_ids_resolve(
    precedent: list[dict[str, Any]],
    dossiers: dict[str, Any],
    index: dict[str, Any],
) -> int:
    """Every ``omitted_precedent_ids`` entry resolves to a document through *index*.

    For each dossier the index lists its kept excerpts (``omitted`` false, in
    dossier order) then its dropped ones (``omitted`` true); each dropped id's
    row names the deal of its precedent record, and that deal is in
    ``documents``. Returns how many omitted ids were resolved.
    """
    records = {p["id"]: p for p in precedent}
    documents = {d["document_id"] for d in index["documents"]}
    resolved = 0
    for clause_id, dossier in dossiers.items():
        rows = index["dossiers"][clause_id]
        assert [r["precedent_id"] for r in rows if not r["omitted"]] == [
            e["precedent_id"] for e in dossier["excerpts"]
        ], clause_id
        omitted = {r["precedent_id"]: r for r in rows if r["omitted"]}
        assert sorted(omitted) == dossier["omitted_precedent_ids"], clause_id
        for pid in dossier["omitted_precedent_ids"]:
            row = omitted[pid]
            assert row["document_id"] == records[pid]["document_id"], pid
            assert row["document_id"] in documents, pid
            resolved += 1
    assert documents == {r["document_id"] for rows in index["dossiers"].values() for r in rows}
    return resolved


def test_omitted_excerpts_resolve_through_the_provenance_index_on_a_synthetic_corpus() -> None:
    """The dropped excerpt's deal is cited by no kept excerpt, yet the index
    lists it (marked omitted) and adds its deal to ``documents``."""
    long_a, long_b = _sentences("Alpha", 40), _sentences("Bravo", 40)
    rows, signed = _concession_and_changed_opening(long_a, long_b)
    pb = _playbook(rows, signed=signed)
    (clause_id,) = pb["dossiers"]
    dropped = _record(pb, "d5")["id"]
    assert pb["dossiers"][clause_id]["omitted_precedent_ids"] == [dropped]
    resolved = _assert_omitted_ids_resolve(
        pb["evidence"]["precedent"], pb["dossiers"], pb["provenance_index"]
    )
    assert resolved == 1
    assert pb["provenance_index"]["dossiers"][clause_id][-1] == {
        "precedent_id": dropped,
        "document_id": "d5",
        "kind": "changed_opening",
        "omitted": True,
    }
    assert "d5" in {d["document_id"] for d in pb["provenance_index"]["documents"]}
    assert validate_document(pb).ok


def test_omitted_excerpts_resolve_through_the_provenance_index_on_vector_008() -> None:
    vector = json.loads(VECTOR_008.read_text(encoding="utf-8"))
    expected = vector["expected"]
    resolved = _assert_omitted_ids_resolve(
        vector["input"]["evidence"]["precedent"],
        expected["dossiers"],
        expected["provenance_index"],
    )
    assert resolved == sum(d["n_omitted"] for d in expected["dossiers"].values()) >= 1
    # The term dossier's dropped excerpt is the only row that cites deal-e.
    rows = [r for rs in expected["provenance_index"]["dossiers"].values() for r in rs]
    assert [r["omitted"] for r in rows if r["document_id"] == "deal-e"] == [True]


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


def _without_identity(pb: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(pb)
    del out["identity"]
    return out


@pytest.mark.parametrize("section", ["manifest", "dossiers", "provenance_index"])
def test_validator_rejects_an_edited_section(section: str) -> None:
    """A deliberately tampered copy (not a producible state) is refused."""
    rows, signed = _selection_rows()
    pb = _without_identity(
        _playbook(rows, signed=signed, floor=_invariants({"statement": "Hold it."}))
    )
    assert validate_document(pb).ok
    if section == "manifest":
        pb["manifest"]["hard_rules"][0]["required_presence"] = True
    elif section == "dossiers":
        next(iter(pb["dossiers"].values()))["n_floor_rules"] += 1
    else:
        pb["provenance_index"]["documents"] = []
    result = validate_document(pb)
    assert not result.ok
    assert any(
        e.path == section and "does not equal its recomputation" in e.message for e in result.errors
    )


def test_validator_rejects_a_dossier_over_its_budget_or_with_three_excerpts() -> None:
    """A deliberately tampered copy (not a producible state) is refused."""
    rows, signed = _selection_rows()
    pb = _without_identity(_playbook(rows, signed=signed))
    (clause_id,) = pb["dossiers"]
    pb["dossiers"][clause_id]["excerpts"].append(
        copy.deepcopy(pb["dossiers"][clause_id]["excerpts"][0])
    )
    pb["dossiers"][clause_id]["floor_rules"] = [{"rule_id": "x", "statement": "x" * 5_000}]
    messages = [str(e) for e in validate_document(pb).errors if e.blocking]
    assert any("token budget" in m for m in messages)
    assert any("at most 2" in m for m in messages)


def test_validator_refuses_a_dossier_with_no_excerpt_over_its_budget() -> None:
    """A deliberately tampered copy (not a producible state: the producer drops
    whole Floor rules to fit) is refused: only a dossier holding its single
    kept excerpt may exceed its budget."""
    rows = [_O("d1", _STD, standard=True, opened_with="standard")]
    pb = _without_identity(_playbook(rows, signed={"d1": True}))
    (clause_id,) = pb["dossiers"]
    dossier = pb["dossiers"][clause_id]
    assert dossier["excerpts"] == []
    dossier["floor_rules"] = [{"rule_id": "x", "statement": "x" * 5_000}]
    messages = [str(e) for e in validate_document(pb).errors if e.blocking]
    assert any("token budget" in m and "lists 0 excerpts" in m for m in messages)


def test_validator_refuses_a_cut_excerpt() -> None:
    """A deliberately tampered copy: a prefix cut of a record's text is not its recomputation."""
    rows, signed = _selection_rows()
    pb = _without_identity(_playbook(rows, signed=signed))
    (dossier,) = pb["dossiers"].values()
    dossier["excerpts"][0]["signed"] = dossier["excerpts"][0]["signed"][:20] + "…"
    result = validate_document(pb)
    assert not result.ok
    assert any(e.path == "dossiers" for e in result.errors)


def test_the_sections_are_optional() -> None:
    rows, signed = _selection_rows()
    pb = _without_identity(_playbook(rows, signed=signed))
    for name in ("manifest", "dossiers", "provenance_index"):
        del pb[name]
    assert validate_document(pb).ok


# ---------------------------------------------------------------------------
# Writers that change the Floor refresh the sections
# ---------------------------------------------------------------------------


def _write_out_dir(tmp_path: Path, pb: dict[str, Any]) -> Path:
    from playbook_engine.playbook_assembler import write_playbook

    out = tmp_path / "out"
    write_playbook(pb, out / "playbook.opf.json")
    return out


def test_floor_sign_refreshes_the_manifest_dossiers_and_identity(tmp_path: Path) -> None:
    from tests.test_floor_candidates import _write_taxonomy_config

    rows, signed = _selection_rows()
    pb = _playbook(rows, signed=signed)
    out = _write_out_dir(tmp_path, pb)
    config = _write_taxonomy_config(tmp_path, ["governing_law"])
    condition = json.dumps({"type": "required_phrases", "phrases": ["State of New York"]})
    result = CliRunner().invoke(
        cli,
        [
            "floor", "sign", str(out),
            "--statement", "Governing law must stay New York.",
            "--id", "law-ny",
            "--signed-by", "Legal Owner",
            "--clause", "governing_law", "--config", str(config),
            "--requires-presence",
            "--condition", condition,
            "--proof", "a waiver signed by the GC",
        ],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    doc = json.loads((out / "playbook.opf.json").read_text(encoding="utf-8"))
    (rule,) = doc["manifest"]["hard_rules"]
    assert rule["rule_id"] == "law-ny"
    assert rule["clause_id"] == "clause.governing_law"
    assert rule["required_presence"] is True
    assert rule["condition"] == json.loads(condition)
    assert rule["permissible_proof"] == ["a waiver signed by the GC"]
    assert doc["manifest"] == build_manifest(doc)
    assert doc["identity"]["content_hash"] == content_hash(doc)
    assert doc["identity"]["section_digests"] == compute_section_digests(doc)
    assert validate_document(doc).ok

    # A rerun of the same signing is a no-op (nothing is rewritten).
    before = (out / "playbook.opf.json").read_bytes()
    again = CliRunner().invoke(
        cli,
        [
            "floor",
            "sign",
            str(out),
            "--statement",
            "Governing law must stay New York.",
            "--id",
            "law-ny",
            "--signed-by",
            "Legal Owner",
            "--clause",
            "governing_law",
            "--config",
            str(config),
        ],
    )
    assert again.exit_code == 0, again.output
    assert (out / "playbook.opf.json").read_bytes() == before


def test_floor_sign_with_a_clause_feeds_the_clause_dossier_and_fallback(tmp_path: Path) -> None:
    from tests.test_floor_candidates import _write_taxonomy_config

    rows, signed = _selection_rows()
    out = _write_out_dir(tmp_path, _playbook(rows, signed=signed))
    config = _write_taxonomy_config(tmp_path, ["governing_law"])
    result = CliRunner().invoke(
        cli,
        [
            "floor",
            "sign",
            str(out),
            "--statement",
            "Governing law must stay New York.",
            "--signed-by",
            "Legal Owner",
            "--clause",
            "governing_law",
            "--config",
            str(config),
            "--rationale",
            "Venue risk.",
        ],
    )
    assert result.exit_code == 0, result.output
    doc = json.loads((out / "playbook.opf.json").read_text(encoding="utf-8"))
    (rule,) = doc["manifest"]["hard_rules"]
    assert (rule["clause_id"], rule["fallback_language"]) == ("clause.governing_law", _STD)
    (dossier,) = doc["dossiers"].values()
    assert dossier["n_floor_rules"] == 1
    assert dossier["floor_rules"][0]["statement"] == "Governing law must stay New York."
    assert dossier["floor_rules"][0]["rationale"] == "Venue risk."
    assert validate_document(doc).ok


@pytest.mark.parametrize(
    "bad", ["{not json", json.dumps({"type": "numeric_bound", "pattern": "x"})]
)
def test_floor_sign_refuses_a_bad_condition_and_writes_nothing(tmp_path: Path, bad: str) -> None:
    rows, signed = _selection_rows()
    out = _write_out_dir(tmp_path, _playbook(rows, signed=signed))
    before = (out / "playbook.opf.json").read_bytes()
    result = CliRunner().invoke(
        cli,
        [
            "floor",
            "sign",
            str(out),
            "--statement",
            "Hold it.",
            "--signed-by",
            "Legal Owner",
            "--condition",
            bad,
        ],
    )
    assert result.exit_code == 1
    assert "--condition" in result.output
    assert (out / "playbook.opf.json").read_bytes() == before


def test_posture_interview_q4_promotion_refreshes_the_sections(tmp_path: Path) -> None:
    from playbook_engine.posture import apply_posture_interview
    from tests.test_posture import _ANSWERS

    rows, signed = _selection_rows()
    out = _write_out_dir(tmp_path, _playbook(rows, signed=signed))
    apply_posture_interview(
        out,
        {**_ANSWERS, "sacred_clauses": "Governing law"},
        generated_at="2026-02-01T00:00:00+00:00",
    )
    doc = json.loads((out / "playbook.opf.json").read_text(encoding="utf-8"))
    assert [r["statement"] for r in doc["manifest"]["hard_rules"]] == [
        i["statement"] for i in doc["floor"]["invariants"]
    ]
    assert doc["manifest"]["hard_rules"], "the Q4 answer should have produced a rule"
    assert doc["identity"]["content_hash"] == content_hash(doc)
    assert validate_document(doc).ok


def test_refresh_leaves_a_document_without_the_sections_alone() -> None:
    doc: dict[str, Any] = {
        "floor": {"invariants": []},
        "evidence": {"clauses": [], "precedent": []},
    }
    assert refresh_derived_sections(doc) is False
    assert set(doc) == {"floor", "evidence"}


# ---------------------------------------------------------------------------
# The NDA worked example (3 signed invariants)
# ---------------------------------------------------------------------------


def test_nda_example_manifest_and_dossiers() -> None:
    doc = json.loads(NDA_PLAYBOOK.read_text(encoding="utf-8"))
    invariants = doc["floor"]["invariants"]
    assert len(invariants) == 3
    rules = doc["manifest"]["hard_rules"]
    assert [r["rule_id"] for r in rules] == [i["id"] for i in invariants]
    assert [r["statement"] for r in rules] == [i["statement"] for i in invariants]
    assert all(r["condition"] == "judged" or isinstance(r["condition"], dict) for r in rules)
    assert doc["manifest"] == build_manifest(doc)
    # One dossier per clause, every one within the ceiling and at most two excerpts.
    assert set(doc["dossiers"]) == {c["id"] for c in doc["evidence"]["clauses"]}
    assert len(doc["dossiers"]) == 26
    for clause_id, dossier in doc["dossiers"].items():
        assert (
            dossier_tokens(dossier) <= dossier_budget((dossier["our_standard"] or {}).get("text"))
            or len(dossier["excerpts"]) <= 1
        ), clause_id
        assert dossier["n_omitted"] == len(dossier["omitted_precedent_ids"])
        assert len(dossier["excerpts"]) <= MAX_EXCERPTS
        assert dossier["clause_id"] == clause_id
    # The signed liability carve-out reaches its clause's dossier.
    liability = doc["dossiers"]["clause.limitation_of_liability"]
    assert [r["rule_id"] for r in liability["floor_rules"]] == [
        "limitation-of-liability-confidentiality-carveout"
    ]
    assert validate_document(doc).ok
