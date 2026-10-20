"""How settled each ``vs_standard`` judgment is (issue #248, OPF 0.6 / digest 5).

``settled`` (``agreed`` | ``adjudicated`` | ``unchecked`` | ``owner``) is
derived from a judgment's ``basis`` and blind ``check`` and carried on the
digest's non-equivalent variants, refused asks and changed openings, as
``settled_counts`` on the collapsed equivalent entry, and on every dossier
excerpt. The inputs here are the conformance vector 011 document, whose
judgments are exactly the shapes the producers write: no check, an agreed
check, an adjudicated check, a disputed check (``agreed`` false, not yet
adjudicated; ``judge-apply --check``) and an owner correction
(``overrides.fold_overrides`` re-stamps ``basis: "owner"`` with ``check:
null``).
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from playbook_engine.digest import (
    build_digest_uncapped,
    build_digest_v5,
    digest_settled_counts,
    digest_token_estimate,
)
from playbook_engine.dossiers import build_derived_sections
from playbook_engine.equivalence import (
    PART_MAP_LEGEND,
    SETTLED,
    check_state,
    settled_counts,
    settled_of,
    summarize,
)
from playbook_engine.precedent import restamp_evidence
from playbook_engine.validator import validate_document

ROOT = Path(__file__).parent.parent
VECTOR_011 = ROOT / "spec" / "conformance" / "0.6" / "vectors" / "011-settled-outcomes.json"

_AGREED = {"agreed": True, "label_agreed": True, "parts_agreed": True, "by": "m"}
_ADJUDICATED = {"agreed": False, "label_agreed": False, "parts_agreed": True, "by": "m"}


def _check(base: dict[str, Any], *, adjudicated: bool) -> dict[str, Any]:
    return {**base, "adjudicated": adjudicated}


@pytest.mark.parametrize(
    ("vs", "settled", "state"),
    [
        ({"basis": "agent", "check": None}, "unchecked", "unchecked"),
        ({"basis": "judge"}, "unchecked", "unchecked"),
        ({"basis": "agent", "check": _check(_AGREED, adjudicated=False)}, "agreed", "agreed"),
        (
            {"basis": "agent", "check": _check(_ADJUDICATED, adjudicated=True)},
            "adjudicated",
            "adjudicated",
        ),
        # A disagreement awaiting adjudication: nothing is settled yet.
        (
            {"basis": "agent", "check": _check(_ADJUDICATED, adjudicated=False)},
            "unchecked",
            "disputed",
        ),
        ({"basis": "owner", "check": None}, "owner", "owner"),
        # An owner's correction wins over any check on file.
        ({"basis": "owner", "check": _check(_AGREED, adjudicated=False)}, "owner", "owner"),
        (
            {"basis": "owner", "check": _check(_ADJUDICATED, adjudicated=True)},
            "owner",
            "owner",
        ),
    ],
)
def test_settled_of_is_a_pure_function_of_basis_and_check(
    vs: dict[str, Any], settled: str, state: str
) -> None:
    assert settled_of(vs) == settled
    assert check_state(vs) == state


def test_settled_counts_lists_every_state_and_ignores_the_unjudged() -> None:
    assert settled_counts(["agreed", "agreed", None, "owner"]) == {
        "agreed": 2,
        "adjudicated": 0,
        "unchecked": 0,
        "owner": 1,
    }
    assert list(settled_counts([])) == list(SETTLED)


def _vector() -> dict[str, Any]:
    return json.loads(VECTOR_011.read_text(encoding="utf-8"))


def _valid_document(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    vector = _vector()
    doc = copy.deepcopy(vector["input"])
    doc["digest"] = copy.deepcopy(vector["expected"]["digest"])
    for name in ("manifest", "dossiers", "provenance_index"):
        doc[name] = copy.deepcopy(vector["expected"][name])
    if extra:
        doc.update(extra)
    return doc


def test_the_fixture_judgments_are_the_shapes_the_producers_write() -> None:
    """Every state the vector holds is one a producer reaches: no owner
    judgment carries a check (``overrides`` writes ``check: null``), and the
    disputed one is the ``agreed: false, adjudicated: false`` the check round
    leaves while it awaits adjudication."""
    doc = _vector()["input"]
    seen: dict[str, int] = {}
    for record in doc["evidence"]["precedent"]:
        entries = [record["signed_text"], record["opening_text"], *record["refused_asks"]]
        for entry in entries:
            vs = entry.get("vs_standard") if entry else None
            if not vs:
                continue
            state = check_state(vs)
            seen[state] = seen.get(state, 0) + 1
            if vs["basis"] == "owner":
                assert vs["check"] is None
    assert set(seen) == {"unchecked", "agreed", "adjudicated", "disputed", "owner"}
    assert validate_document(_valid_document()).ok


def test_digest_carries_settled_beside_each_judged_non_equivalent_entry() -> None:
    clause = build_digest_v5(_vector()["input"])["clauses"][0]
    variants = clause["signed_variants"]
    assert [(v["label"], v["settled"]) for v in variants[:-1]] == [
        ("mixed", "adjudicated"),
        ("less_protective", "agreed"),
        ("different_concept", "unchecked"),  # the disputed one reads unchecked
        ("more_protective", "owner"),
    ]
    assert [a["settled"] for a in clause["refused_asks"]] == [None, "adjudicated"]
    assert [o["settled"] for o in clause["changed_openings"]] == ["unchecked"]
    # An unjudged ask carries the nulls beside it, not a guessed outcome.
    unjudged = clause["refused_asks"][0]
    assert (unjudged["label"], unjudged["part_map"], unjudged["settled"]) == (None, None, None)


def test_the_collapsed_equivalent_entry_counts_each_outcome() -> None:
    clause = build_digest_v5(_vector()["input"])["clauses"][0]
    collapsed = clause["signed_variants"][-1]
    assert collapsed["label"] == "equivalent" and "settled" not in collapsed
    assert collapsed["settled_counts"] == {
        "agreed": 2,
        "adjudicated": 1,
        "unchecked": 0,
        "owner": 1,
    }
    assert sum(collapsed["settled_counts"].values()) == collapsed["n_texts"] == 4
    assert list(collapsed["settled_counts"]) == list(SETTLED)


def test_the_legend_states_the_four_values() -> None:
    digest = build_digest_v5(_vector()["input"])
    assert digest["part_map_legend"] == PART_MAP_LEGEND
    assert set(digest["part_map_legend"]["settled"]) == set(SETTLED)


def test_dossier_excerpts_carry_settled() -> None:
    excerpts = build_derived_sections(_vector()["input"])["dossiers"]["clause.assignment"][
        "excerpts"
    ]
    assert [(x["kind"], x["vs_standard"]["settled"]) for x in excerpts] == [
        ("signed_variant", "adjudicated"),
        ("changed_opening", "unchecked"),
    ]


def test_digest_outcome_counts_match_the_equivalence_totals() -> None:
    """The digest's outcome counts (every judged text of the uncapped digest,
    a collapsed entry once per text) equal the scorecard's ``equivalence``
    counts: per role the same texts, in total the sum of the roles."""
    doc = _vector()["input"]
    digest_counts = digest_settled_counts(build_digest_uncapped(doc))
    summary = summarize(doc["evidence"], doc["agreement_type"]["id"], "Fixture Co")
    roles = summary["by_role"]
    assert digest_counts["signed_variants"] == roles["signed"]["settled"]
    assert digest_counts["refused_asks"] == roles["refused"]["settled"]
    assert digest_counts["changed_openings"] == roles["opening"]["settled"]
    for state in SETTLED:
        assert digest_counts["total"][state] == sum(roles[r]["settled"][state] for r in roles)
    # Every drafted text has exactly one outcome.
    assert sum(summary["totals"]["settled"].values()) == summary["totals"]["drafted"]
    assert summary["totals"]["settled"] == {
        "agreed": 3,
        "adjudicated": 3,
        "unchecked": 2,
        "owner": 2,
    }
    assert summary["totals"]["unjudged"] == 1  # the second refused ask


def test_the_capped_digest_stays_within_budget() -> None:
    digest = build_digest_v5(_vector()["input"])
    assert digest_token_estimate(digest) <= 40_000


# -- the validator recomputes the digest and the dossiers ------------------


def _errors(doc: dict[str, Any]) -> str:
    result = validate_document(doc)
    return "\n".join(str(e) for e in result.errors if e.blocking)


def test_validator_rejects_a_wrong_settled_on_a_digest_variant() -> None:
    doc = _valid_document()
    doc["digest"]["clauses"][0]["signed_variants"][0]["settled"] = "agreed"
    assert "digest" in _errors(doc)


def test_validator_rejects_a_wrong_settled_counts_on_the_collapsed_entry() -> None:
    doc = _valid_document()
    counts = doc["digest"]["clauses"][0]["signed_variants"][-1]["settled_counts"]
    counts["agreed"], counts["unchecked"] = counts["agreed"] - 1, counts["unchecked"] + 1
    assert "digest" in _errors(doc)


def test_validator_rejects_a_wrong_settled_on_a_dossier_excerpt() -> None:
    doc = _valid_document()
    doc["dossiers"]["clause.assignment"]["excerpts"][0]["vs_standard"]["settled"] = "owner"
    assert "dossiers" in _errors(doc)


def test_validator_rejects_a_settled_the_schema_does_not_name() -> None:
    doc = _valid_document()
    doc["digest"]["clauses"][0]["refused_asks"][1]["settled"] = "mostly_agreed"
    assert _errors(doc)


def test_changing_a_check_changes_the_digest_settled() -> None:
    """The evidence is the only input: settle the disputed text by
    adjudication and its digest entry follows."""
    doc = _vector()["input"]
    disputed = next(
        e["vs_standard"]
        for p in doc["evidence"]["precedent"]
        for e in [p["signed_text"]]
        if e and e.get("vs_standard") and check_state(e["vs_standard"]) == "disputed"
    )
    disputed["check"]["adjudicated"] = True
    clause = build_digest_v5(doc)["clauses"][0]
    assert [v.get("settled") for v in clause["signed_variants"]][:3] == [
        "adjudicated",
        "agreed",
        "adjudicated",
    ]


def _entry(base: dict[str, Any], document_id: str, text: str, version: int) -> dict[str, Any]:
    """A text entry for *document_id*, judged like *base* (a fixture ``vs_standard``)."""
    return {
        "text": text,
        "ref": {
            "document_id": document_id,
            "version": version,
            "clause_path": "1",
            "char_span": [0, 40],
        },
        "vs_standard": copy.deepcopy(base["vs_standard"]),
    }


def _document_with_both_exclusions() -> dict[str, Any]:
    """The vector plus the records whose judged texts the digest leaves out.

    All are built the way ``precedent.build_precedent_evidence`` writes them,
    then restamped (ids, clause counts) and re-derived (digest, dossiers), so
    the document is one the engine can produce and the validator accepts:

    - an unsigned deal: a corpus document with no executed copy
      (``signed_version`` null), ``signed`` false, ``opened_with`` and
      ``signed_on`` null, no opening and no refused asks, holding only the
      last draft as ``signed_text``;
    - a signed deal whose opening repeats one of its own refused asks (the
      producer keeps ``opening_text`` for it; ``changed_opening_members``
      drops it because the digest lists the text as a refused ask).

    Each is seeded twice. Once with a text no other record holds (deal-z's
    last draft, deal-y's repeated ask, both adjudicated): the digest lists it
    nowhere, so it is not listed. Once with a text another record of the same
    role lists (deal-x's last draft is deal-b's signed variant, agreed;
    deal-w's repeated ask is deal-g's changed opening, unchecked, which deal-g
    opened with non-standard and did not refuse): the digest lists that text,
    and the scorecard counts distinct texts, so it is not counted as not
    listed. The second kind is the common one on a real corpus.
    """
    doc = _valid_document()
    precedent = doc["evidence"]["precedent"]
    judged = precedent[2]["signed_text"]  # an adjudicated judgment
    listed_variant = precedent[1]["signed_text"]  # deal-b's signed variant, agreed
    listed_opening = precedent[6]["opening_text"]  # deal-g's changed opening, unchecked
    signed_variant = precedent[8]["signed_text"]
    for document_id, versions, signed_version in (
        ("deal-z", 2, None),
        ("deal-y", 3, 3),
        ("deal-x", 2, None),
        ("deal-w", 3, 3),
    ):
        doc["corpus"]["documents"].append(
            {
                "document_id": document_id,
                "provenance": "our_paper",
                "in_scope": True,
                "versions": versions,
                "signed_version": signed_version,
                "signed_on": None,
                "signed_on_basis": None,
            }
        )
    base = {
        "taxonomy_id": "assignment",
        "paper": "ours",
        "paper_basis": "provenance_detection",
        "paper_confidence": 0.9,
        "standard": False,
    }

    def unsigned(document_id: str, judged_like: dict[str, Any], text: str) -> dict[str, Any]:
        return {
            "id": "",
            "document_id": document_id,
            **base,
            "signed": False,
            "signed_on": None,
            "rounds": 0,
            "signed_text": _entry(judged_like, document_id, text, 2),
            "opened_with": None,
            "opening_text": None,
            "moved": False,
            "refused_asks": [],
        }

    def repeats_its_refused_ask(
        document_id: str, judged_like: dict[str, Any], text: str
    ) -> dict[str, Any]:
        signed_text = copy.deepcopy(signed_variant)
        signed_text["ref"]["document_id"] = document_id
        return {
            "id": "",
            "document_id": document_id,
            **base,
            "signed": True,
            "signed_on": None,
            "rounds": 1,
            "signed_text": signed_text,
            "opened_with": "non_standard",
            "opening_text": _entry(judged_like, document_id, text, 1),
            "moved": True,
            "refused_asks": [{**_entry(judged_like, document_id, text, 2), "round": 1}],
        }

    precedent.extend(
        [
            unsigned("deal-z", judged, "Either party may assign this Agreement after ninety days."),
            repeats_its_refused_ask(
                "deal-y", judged, "Assignment needs the other party's notice and consent."
            ),
            unsigned("deal-x", listed_variant, listed_variant["text"]),
            repeats_its_refused_ask("deal-w", listed_opening, listed_opening["text"]),
        ]
    )
    restamp_evidence(
        doc["evidence"], doc["agreement_type"]["id"], party=doc["perspective"]["party"]
    )
    doc["digest"] = build_digest_v5(doc)
    doc.update(build_derived_sections(doc))
    return doc


def test_scorecard_reconciles_the_digest_outcomes_with_the_equivalence_outcomes(
    tmp_path: Path,
) -> None:
    """Equivalence = digest + the two named exclusions + nothing else.

    ``digest.settled_not_listed`` counts the exclusions directly (the last draft
    of an unsigned deal; an opening that repeats its own deal's refused ask)
    and ``digest.settled_unreconciled`` is what neither explains: it is zero in
    every cell.
    """
    from playbook_engine.scorecard import build_scorecard

    doc = _document_with_both_exclusions()
    assert validate_document(doc).ok, _errors(doc)
    (tmp_path / "playbook.opf.json").write_text(json.dumps(doc), encoding="utf-8")
    card = build_scorecard(tmp_path)
    settled = card["digest"]["settled"]
    not_listed = card["digest"]["settled_not_listed"]
    unreconciled = card["digest"]["settled_unreconciled"]
    by_role = card["equivalence"]["by_role"]
    for role, name in (
        ("signed", "signed_variants"),
        ("refused", "refused_asks"),
        ("opening", "changed_openings"),
    ):
        for state in SETTLED:
            assert unreconciled[name][state] == 0
            assert settled[name][state] + not_listed[name][state] == by_role[role]["settled"][state]
    # One last draft of an unsigned deal, one opening that repeats a refused ask,
    # both adjudicated (the judgment copied from the fixture); nothing else is left out.
    assert not_listed["signed_variants"]["adjudicated"] == 1
    assert not_listed["changed_openings"]["adjudicated"] == 1
    # The other two exclusions carry texts the digest lists elsewhere (deal-b's
    # signed variant, deal-g's changed opening), so neither is counted again.
    records = {p["document_id"]: p for p in doc["evidence"]["precedent"]}
    clause = doc["digest"]["clauses"][0]
    assert records["deal-x"]["signed_text"]["text"] in {
        v.get("text") for v in clause["signed_variants"]
    }
    assert records["deal-w"]["opening_text"]["text"] in {
        o["text"] for o in clause["changed_openings"]
    }
    assert not_listed["signed_variants"]["agreed"] == 0
    assert not_listed["changed_openings"]["unchecked"] == 0
    assert sum(sum(v.values()) for v in not_listed.values()) == 2


def test_the_scorecard_flags_a_difference_no_named_exclusion_explains(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A judged text the digest drops with no named reason is not hidden as
    "not listed": it shows up in ``settled_unreconciled`` (here, a digest
    that loses one signed variant)."""
    import playbook_engine.digest as digest_module
    from playbook_engine.scorecard import build_scorecard

    real = digest_module.digest_settled_counts

    def lossy(digest: dict[str, Any]) -> dict[str, dict[str, int]]:
        counts = real(digest)
        counts["signed_variants"]["agreed"] -= 1
        return counts

    monkeypatch.setattr(digest_module, "digest_settled_counts", lossy)
    doc = _document_with_both_exclusions()
    (tmp_path / "playbook.opf.json").write_text(json.dumps(doc), encoding="utf-8")
    card = build_scorecard(tmp_path)
    assert card["digest"]["settled_unreconciled"]["signed_variants"]["agreed"] == 1
    assert card["digest"]["settled_not_listed"]["signed_variants"]["agreed"] == 0
