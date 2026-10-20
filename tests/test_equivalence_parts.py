"""``vs_standard`` part by part (issue #243).

Our standard for a clause type is an ordered list of parts (the template
passages of the type); each distinct text is judged against EVERY part
(present with a relation, or omitted with the effect of the omission) plus
what it adds, and the label is DERIVED from that judgment. These tests cover
the derived-label function (every branch), part ids, the judgment shape, the
cache key, the producer of ``our_standard.parts``, the blind check's two
agreement halves, the digest part map, the dossier excerpt, the Review rows
and the owner overrides.

Fixture provenance (the producer of every shape used here):
- standards come from TEMPLATE ``Observation`` rows of the shape
  ``observation_builder`` writes, through ``compile_clause_positions`` ->
  ``template_standards`` (two template nodes of one clause type are a
  two-part standard) -> ``assemble_playbook``;
- judgments are the dicts ``playbook judge-apply --verdicts`` accepts
  (``equivalence.validate_draft_verdict`` is the function it runs), stored
  and replayed through ``VerdictStore`` / ``StoreBackedEquivalenceJudge``;
  check records are the dicts ``playbook judge-apply --check`` accepts;
- owner overrides are ``overrides.json`` entries ``apply_overrides`` parses.

SECURITY NOTE: synthetic text only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from playbook_engine.agent_judge import StoreBackedEquivalenceJudge, VerdictStore
from playbook_engine.clause_position_compiler import compile_clause_positions
from playbook_engine.digest import build_digest_v5
from playbook_engine.dossiers import build_derived_sections, excerpt_vs_standard
from playbook_engine.equivalence import (
    PART_MAP_LEGEND,
    canonical_judgment,
    collect_subjects,
    derive_label,
    equivalence_key,
    part_id_of,
    part_ids_of,
    part_map,
    validate_draft_verdict,
    vs_standard_of,
)
from playbook_engine.equivalence_check import (
    CHECKER_EFFORT,
    CHECKER_MODEL,
    apply_check_records,
)
from playbook_engine.observation_builder import Observation, ObservationCitation
from playbook_engine.overrides import apply_overrides, fold_overrides, parse_overrides
from playbook_engine.playbook_assembler import assemble_playbook
from playbook_engine.review_rows import build_review_rows
from playbook_engine.validator import validate_document

_PARTY = "Acme Corp"
_TYPE = "test-agreement"
_TID = "confidentiality"

#: Two template nodes of one clause type: a lead-in and its operative limb.
_P1 = "Each party shall keep the other party's information confidential."
_P2 = "The duty of confidentiality lasts three years after the Agreement ends."

_CAND_OMITS_P2 = "Each party shall keep the other party's information confidential."
_CAND_LONGER = (
    "Each party shall keep the other party's information confidential, "
    "and the duty lasts five years after the Agreement ends."
)
_CAND_UNRELATED = "Each party shall pay the other party's legal fees."

E, M, L, N = "equivalent", "more_protective", "less_protective", "neutral"


# ---------------------------------------------------------------------------
# The derived label — every branch
# ---------------------------------------------------------------------------


def _present(pid: str, relation: str) -> dict[str, Any]:
    return {"part_id": pid, "status": "present", "relation": relation}


def _omitted(pid: str, effect: str) -> dict[str, Any]:
    return {"part_id": pid, "status": "omitted", "effect": effect}


def _add(effect: str) -> dict[str, Any]:
    return {"summary": "Adds something.", "effect": effect}


@pytest.mark.parametrize(
    ("parts", "additions", "label"),
    [
        # different_concept: no part present, whatever the additions say.
        ([_omitted("a", N), _omitted("b", N)], [], "different_concept"),
        ([_omitted("a", L)], [_add(M)], "different_concept"),
        # equivalent: every present part equivalent, everything else neutral.
        ([_present("a", E)], [], "equivalent"),
        # the neutral omission: dropping a recital no longer makes it unrankable.
        ([_present("a", E), _omitted("b", N)], [], "equivalent"),
        ([_present("a", E)], [_add(N)], "equivalent"),
        # one direction, from a present part, an omission or an addition.
        ([_present("a", M), _present("b", E)], [], "more_protective"),
        ([_present("a", E), _omitted("b", M)], [], "more_protective"),
        ([_present("a", E)], [_add(M)], "more_protective"),
        ([_present("a", L), _omitted("b", N)], [_add(N)], "less_protective"),
        ([_present("a", E), _omitted("b", L)], [], "less_protective"),
        ([_present("a", E)], [_add(L)], "less_protective"),
        # mixed: both ways, across parts, omissions and additions.
        ([_present("a", L), _omitted("b", M)], [], "mixed"),
        ([_present("a", M), _present("b", L)], [], "mixed"),
        ([_present("a", L)], [_add(M)], "mixed"),
        ([_present("a", E), _omitted("b", L)], [_add(M)], "mixed"),
    ],
)
def test_derive_label_covers_every_branch(
    parts: list[dict[str, Any]], additions: list[dict[str, Any]], label: str
) -> None:
    assert derive_label(parts, additions) == label


def test_part_map_has_one_token_per_part_in_standard_order() -> None:
    vs = {
        "parts": [
            _present("a", E),
            _present("b", M),
            _present("c", L),
            _omitted("d", M),
            _omitted("e", L),
            _omitted("f", N),
        ]
    }
    assert part_map(vs) == ["=", "+", "-", "x+", "x-", "x0"]
    assert set(PART_MAP_LEGEND["part_map"]) == {"=", "+", "-", "x+", "x-", "x0"}
    assert set(PART_MAP_LEGEND["additions"]) == {"+", "-", "0"}


# ---------------------------------------------------------------------------
# Part ids
# ---------------------------------------------------------------------------


def test_part_ids_are_content_hashes_that_survive_an_edit_elsewhere() -> None:
    ids = part_ids_of([_P1, _P2], _PARTY)
    assert ids == [part_id_of(_P1, _PARTY), part_id_of(_P2, _PARTY)]
    assert all(len(pid) == 8 for pid in ids)
    # Whitespace and case are not part of the text identity.
    assert (
        part_id_of("  EACH party shall keep the other party's information confidential ", _PARTY)
        == ids[0]
    )
    # A part added, removed or edited elsewhere leaves the other ids alone.
    assert part_ids_of(["A new recital.", _P1, _P2 + " Extra."], _PARTY)[1] == ids[0]
    # Two parts with the same normalized text in one standard stay distinct.
    assert part_ids_of([_P1, _P2, _P1], _PARTY) == [ids[0], ids[1], f"{ids[0]}-2"]


# ---------------------------------------------------------------------------
# The judgment shape
# ---------------------------------------------------------------------------


def test_canonical_judgment_orders_parts_and_derives_the_label() -> None:
    got = canonical_judgment(
        {
            "parts": [_omitted("b", N), _present("a", E)],
            "additions": [],
            "reason": "Same effect; the recital is dropped.",
        },
        ["a", "b"],
    )
    assert [p["part_id"] for p in got["parts"]] == ["a", "b"]
    assert got["label"] == "equivalent"


@pytest.mark.parametrize(
    ("judgment", "match"),
    [
        (
            {"parts": [{"part_id": "a", "status": "present"}, _present("b", E)]},
            "needs a 'relation'",
        ),
        ({"parts": [{**_omitted("a", N), "relation": E}, _present("b", E)]}, "not a 'relation'"),
        ({"parts": [_present("a", L), _present("b", E)]}, "needs a one-sentence 'reason'"),
        ({"parts": [{**_present("a", E), "reason": "x"}, _present("b", E)]}, "only for a part"),
        ({"parts": [_present("a", E), _present("z", E)]}, "unknown part_id"),
        ({"parts": [_present("a", E)]}, "not judged"),
        ({"parts": [_present("a", E), _present("a", E)]}, "more than once"),
        ({"parts": [_present("a", E), _present("b", E)], "label": "mixed"}, "not what the parts"),
        ({"parts": [_present("a", E), _present("b", E)], "additions": [_add("worse")]}, "'effect'"),
    ],
)
def test_canonical_judgment_rejects_malformed_judgments(
    judgment: dict[str, Any], match: str
) -> None:
    body = {"additions": [], "reason": "One sentence.", **judgment}
    with pytest.raises(ValueError, match=match):
        canonical_judgment(body, ["a", "b"])


def test_a_draft_is_parts_additions_reason_and_basis() -> None:
    draft = {"parts": [_present("a", E)], "additions": [], "reason": "Same.", "basis": "agent"}
    assert validate_draft_verdict(draft, ["a"])["label"] == "equivalent"
    for bad in (
        {**draft, "basis": "guess"},
        {**draft, "check": {"agreed": True}},
        {k: v for k, v in draft.items() if k != "additions"},
        {k: v for k, v in draft.items() if k != "reason"},
    ):
        with pytest.raises(ValueError):
            validate_draft_verdict(bad, ["a"])


def test_owner_label_needs_owner_basis() -> None:
    stored = {
        "parts": [{**_present("a", L), "reason": "Narrower."}],
        "additions": [],
        "reason": "Worse.",
        "label": "equivalent",  # the effective label: the owner's
        "basis": "owner",
        "owner_label": {"label": "equivalent", "reason": "Immaterial for us."},
    }
    assert vs_standard_of(stored, ["a"])["label"] == "equivalent"
    with pytest.raises(ValueError, match="owner"):
        vs_standard_of({**stored, "basis": "agent"}, ["a"])


# ---------------------------------------------------------------------------
# The producer: our_standard.parts, the payload and the key
# ---------------------------------------------------------------------------


def _obs(
    doc_id: str,
    text: str,
    *,
    version: str = "v2",
    path: str = "8",
    outcome: str = "signed",
    standard: bool = False,
    opened_with: str | None = "non_standard",
    provenance: str = "our_paper",
) -> Observation:
    return Observation(
        observation_id=f"{doc_id}/{version}/{path}",
        taxonomy_id=_TID,
        text_summary=text,
        full_text=text,
        citation=ObservationCitation(
            document_id=doc_id, version=version, clause_path=path, char_span=None
        ),
        deviation="none" if standard else "substantive",
        risk_delta={"direction": "neutral", "magnitude": "none"},
        provenance=provenance,
        outcome=outcome,
        basis="deterministic",
        standard=standard,
        opened_with=opened_with,
    )


def _doc(doc_id: str, provenance: str) -> dict[str, Any]:
    return {
        "document_id": doc_id,
        "provenance": provenance,
        "in_scope": True,
        "versions": 2,
        "signed_version": 2,
        "signed_on": None,
        "signed_on_basis": None,
        "version_order_basis": "edit_distance_chain",
    }


_DEALS = {"deal_a": _CAND_OMITS_P2, "deal_b": _CAND_LONGER, "deal_c": _CAND_UNRELATED}


def _playbook(judge: StoreBackedEquivalenceJudge | None = None) -> dict[str, Any]:
    template = [
        _obs("template", _P1, version="template", path="5", standard=True, opened_with=None),
        _obs("template", _P2, version="template", path="6", standard=True, opened_with=None),
    ]
    observations = [_obs(d, text) for d, text in _DEALS.items()]
    positions, _, _ = compile_clause_positions(observations, template)
    return assemble_playbook(
        agreement_type={"id": _TYPE, "name": "Test Agreement"},
        baseline={"has_canonical_template": True},
        taxonomy={
            "source": "test",
            "entries": [
                {
                    "id": _TID,
                    "label": "Confidentiality",
                    "status": "active",
                    "cuad_origin": "x",
                    "description": "The duty.",
                }
            ],
        },
        clause_positions=positions,
        corpus_documents=[
            _doc("deal_a", "our_paper"),
            _doc("deal_b", "counterparty_paper"),
            _doc("deal_c", "our_paper"),
        ],
        generated_at="2026-10-10T00:00:00Z",
        observations=observations,
        perspective={"party": _PARTY, "counterparty_type": "vendor"},
        equivalence_judge=judge,
    )


def test_two_template_nodes_are_a_two_part_standard() -> None:
    playbook = _playbook()
    (clause,) = playbook["evidence"]["clauses"]
    std = clause["our_standard"]
    assert [p["text"] for p in std["parts"]] == [_P1, _P2]
    assert [p["order"] for p in std["parts"]] == [1, 2]
    assert [p["part_id"] for p in std["parts"]] == part_ids_of([_P1, _P2], _PARTY)
    assert std["text"] == f"{_P1}\n\n{_P2}"
    assert validate_document(playbook).ok


def test_the_payload_numbers_the_parts_and_names_no_deal_or_paper() -> None:
    subjects = collect_subjects(_playbook()["evidence"], _TYPE, _PARTY)
    assert len(subjects) == 3
    payload = subjects[0].payload
    assert payload["our_standard_parts"] == [
        {"n": 1, "part_id": part_id_of(_P1, _PARTY), "text": _P1},
        {"n": 2, "part_id": part_id_of(_P2, _PARTY), "text": _P2},
    ]
    assert "our_standard" not in payload
    blob = json.dumps([s.payload for s in subjects])
    for needle in ("deal_a", "deal_b", "deal_c", "our_paper", "counterparty_paper", "paper"):
        assert needle not in blob, needle


def test_the_key_is_over_the_ordered_parts_never_the_paper() -> None:
    key = equivalence_key(_TYPE, _TID, _PARTY, _CAND_LONGER, [_P1, _P2])
    assert key != equivalence_key(_TYPE, _TID, _PARTY, _CAND_LONGER, [_P2, _P1])
    assert key != equivalence_key(_TYPE, _TID, _PARTY, _CAND_LONGER, [f"{_P1}\n\n{_P2}"])
    assert key == equivalence_key(_TYPE, _TID, _PARTY, _CAND_LONGER, [_P1.upper(), _P2])
    with pytest.raises(TypeError):
        equivalence_key(_TYPE, _TID, _PARTY, _CAND_LONGER, _P1)  # type: ignore[arg-type]
    # The deal on counterparty paper is keyed exactly like one on ours.
    subjects = {
        s.payload["candidate"]: s.key
        for s in collect_subjects(_playbook()["evidence"], _TYPE, _PARTY)
    }
    assert subjects[_CAND_LONGER] == key


# ---------------------------------------------------------------------------
# Judged end to end: store -> playbook -> digest -> dossier -> review rows
# ---------------------------------------------------------------------------


def _judgments() -> dict[str, dict[str, Any]]:
    a, b = part_ids_of([_P1, _P2], _PARTY)
    return {
        _CAND_OMITS_P2: {
            "parts": [
                _present(a, E),
                {**_omitted(b, L), "reason": "No time limit is stated for the duty."},
            ],
            "additions": [],
            "reason": "Drops the three-year duration.",
            "basis": "agent",
        },
        _CAND_LONGER: {
            "parts": [
                _present(a, E),
                {
                    **_present(b, M),
                    "reason": "Five years instead of three.",
                },
            ],
            "additions": [
                {"summary": "Allows disclosure to advisers.", "effect": L},
            ],
            "reason": "A longer duty but disclosure to advisers is allowed.",
            "basis": "agent",
        },
        _CAND_UNRELATED: {
            "parts": [_omitted(a, L) | {"reason": "No confidentiality duty."}, _omitted(b, N)],
            "additions": [{"summary": "A fee-shifting duty.", "effect": N}],
            "reason": "Concerns fees, not confidentiality.",
            "basis": "agent",
        },
    }


def _judged(tmp_path: Path) -> tuple[dict[str, Any], VerdictStore]:
    store = VerdictStore(tmp_path / "judge" / "verdicts.jsonl")
    for subject in collect_subjects(_playbook()["evidence"], _TYPE, _PARTY):
        draft = _judgments()[subject.payload["candidate"]]
        store.put_by_key(subject.key, validate_draft_verdict(draft, subject.part_ids))
    return _playbook(StoreBackedEquivalenceJudge(store=store)), store


def test_judged_playbook_carries_parts_and_derived_labels(tmp_path: Path) -> None:
    playbook, store = _judged(tmp_path)
    by_deal = {
        p["document_id"]: p["signed_text"]["vs_standard"] for p in playbook["evidence"]["precedent"]
    }
    assert by_deal["deal_a"]["label"] == "less_protective"
    assert by_deal["deal_b"]["label"] == "mixed"
    assert by_deal["deal_c"]["label"] == "different_concept"
    assert part_map(by_deal["deal_b"]) == ["=", "+"]
    result = validate_document(playbook, verdict_store=store)
    assert result.ok, [str(e) for e in result.errors]


def test_digest_carries_the_compact_part_map_and_lists_mixed_and_less_first(
    tmp_path: Path,
) -> None:
    playbook, _ = _judged(tmp_path)
    digest = build_digest_v5(playbook)
    assert digest["part_map_legend"] == PART_MAP_LEGEND
    (clause,) = digest["clauses"]
    assert clause["our_standard"]["parts"] == [
        {"part_id": part_id_of(_P1, _PARTY), "text": _P1},
        {"part_id": part_id_of(_P2, _PARTY), "text": _P2},
    ]
    assert "text" not in clause["our_standard"]
    rows = [(v["label"], v["part_map"], v["additions"]) for v in clause["signed_variants"]]
    # less_protective and mixed lead (group order inside the tier), then different_concept.
    assert rows == [
        ("less_protective", ["=", "x-"], []),
        ("mixed", ["=", "+"], ["-"]),
        ("different_concept", ["x-", "x0"], ["0"]),
    ]
    assert clause["signed_variants"][1]["reason"] == (
        "A longer duty but disclosure to advisers is allowed."
    )
    assert clause["positions"]["mixed"] == 1
    # Part reasons stay out of the digest.
    assert "Five years instead of three." not in json.dumps(digest)


def test_dossier_excerpts_carry_the_part_map_and_part_reasons(tmp_path: Path) -> None:
    playbook, _ = _judged(tmp_path)
    dossier = build_derived_sections(playbook)["dossiers"][f"clause.{_TID}"]
    assert [p["text"] for p in dossier["our_standard"]["parts"]] == [_P1, _P2]
    excerpt = dossier["excerpts"][0]
    vs = excerpt["vs_standard"]
    assert vs["label"] == "less_protective" and vs["part_map"] == ["=", "x-"]
    assert vs["part_reasons"] == [
        {"part_id": part_id_of(_P2, _PARTY), "reason": "No time limit is stated for the duty."}
    ]
    assert vs["additions"] == []


def test_review_rows_show_the_part_map(tmp_path: Path) -> None:
    playbook, _ = _judged(tmp_path)
    rows = build_review_rows(playbook).rows
    assert [r["label"] for r in rows] == ["less_protective", "mixed", "different_concept"]
    mixed = rows[1]
    assert [p["token"] for p in mixed["parts"]] == ["=", "+"]
    assert mixed["parts"][1]["reason"] == "Five years instead of three."
    assert mixed["additions"] == [
        {"token": "-", "effect": L, "summary": "Allows disclosure to advisers."}
    ]
    assert mixed["derived_label"] == "mixed"


# ---------------------------------------------------------------------------
# The blind check: label and parts agreement, separately
# ---------------------------------------------------------------------------


def _record(key: str, judgment: dict[str, Any]) -> dict[str, Any]:
    body = {k: judgment[k] for k in ("parts", "additions", "reason")}
    return {"key": key, **body, "model": CHECKER_MODEL, "effort": CHECKER_EFFORT}


def test_check_agreement_needs_the_same_label_and_the_same_parts(tmp_path: Path) -> None:
    _, store = _judged(tmp_path)
    subjects = {
        s.payload["candidate"]: s for s in collect_subjects(_playbook()["evidence"], _TYPE, _PARTY)
    }
    a, b = part_ids_of([_P1, _P2], _PARTY)
    # Same label (less_protective) reached through a different part: label
    # agreed, parts disagreed -> a disagreement.
    other = {
        "parts": [
            {**_present(a, L), "reason": "Narrower duty."},
            _omitted(b, N),
        ],
        "additions": [],
        "reason": "A narrower duty.",
    }
    same = _judgments()[_CAND_LONGER]
    result = apply_check_records(
        store,
        [
            (1, _record(subjects[_CAND_OMITS_P2].key, other)),
            (2, _record(subjects[_CAND_LONGER].key, same)),
        ],
    )
    checks = {key: v["check"] for key, v in result.updates}
    assert checks[subjects[_CAND_OMITS_P2].key] == {
        "agreed": False,
        "label_agreed": True,
        "parts_agreed": False,
        "by": CHECKER_MODEL,
        "adjudicated": False,
    }
    assert checks[subjects[_CAND_LONGER].key]["agreed"] is True
    assert (result.agreed, result.disagreed, result.label_agreed, result.parts_agreed) == (
        1,
        1,
        2,
        1,
    )
    # A check about parts our standard does not have is rejected.
    with pytest.raises(ValueError, match="unknown part_id"):
        apply_check_records(
            store,
            [
                (
                    1,
                    _record(
                        subjects[_CAND_UNRELATED].key,
                        {
                            "parts": [_present("deadbeef", E), _present(a, E), _present(b, E)],
                            "additions": [],
                            "reason": "Same.",
                        },
                    ),
                )
            ],
        )


# ---------------------------------------------------------------------------
# Owner overrides: the per-part inputs, or the label with a reason
# ---------------------------------------------------------------------------


def _key_of(playbook: dict[str, Any], candidate: str) -> str:
    return next(
        s.key
        for s in collect_subjects(playbook["evidence"], _TYPE, _PARTY)
        if s.payload["candidate"] == candidate
    )


def test_an_owner_can_set_a_part_and_the_label_is_rederived(tmp_path: Path) -> None:
    playbook, _ = _judged(tmp_path)
    key = _key_of(playbook, _CAND_LONGER)
    b = part_id_of(_P2, _PARTY)
    entries = parse_overrides(
        {
            "overrides_version": 1,
            "overrides": [
                {
                    "target": "vs_standard",
                    "id": key,
                    "value": {
                        "parts": [
                            {"part_id": b, "status": "present", "relation": "less_protective"}
                        ]
                    },
                    "basis": "owner",
                }
            ],
        }
    )
    edited, result = fold_overrides(playbook, entries, tmp_path)
    vs = next(
        p["signed_text"]["vs_standard"]
        for p in edited["evidence"]["precedent"]
        if p["document_id"] == "deal_b"
    )
    assert vs["label"] == "less_protective" and vs["basis"] == "owner"
    assert vs["parts"][1] == {
        "part_id": b,
        "status": "present",
        "relation": "less_protective",
        "reason": "Set by the owner.",
    }
    assert "owner_label" not in vs
    # The judgment's reason explained the mixed map the owner changed.
    assert vs["reason"] == "Set by the owner."
    assert result.verdicts[key]["label"] == "less_protective"
    assert VerdictStore(tmp_path / "judge" / "verdicts.jsonl").get_by_key(key)["basis"] == "owner"


_SAME_EFFECT = "Same legal effect as our standard."


def _judged_equivalent(tmp_path: Path) -> tuple[dict[str, Any], str]:
    """The fixture with deal-b's text judged equivalent (``=, =``, no
    additions, the model's reason :data:`_SAME_EFFECT`), and that text's key."""
    a, b = part_ids_of([_P1, _P2], _PARTY)
    judgments = _judgments()
    judgments[_CAND_LONGER] = {
        "parts": [_present(a, E), _present(b, E)],
        "additions": [],
        "reason": _SAME_EFFECT,
        "basis": "agent",
    }
    store = VerdictStore(tmp_path / "judge" / "verdicts.jsonl")
    for subject in collect_subjects(_playbook()["evidence"], _TYPE, _PARTY):
        draft = judgments[subject.payload["candidate"]]
        store.put_by_key(subject.key, validate_draft_verdict(draft, subject.part_ids))
    playbook = _playbook(StoreBackedEquivalenceJudge(store=store))
    return playbook, _key_of(playbook, _CAND_LONGER)


def _set_part_two(key: str, relation: str, reason: str | None = None) -> list[Any]:
    value: dict[str, Any] = {
        "parts": [{"part_id": part_id_of(_P2, _PARTY), "status": "present", "relation": relation}]
    }
    if reason is not None:
        value["reason"] = reason
    return parse_overrides(
        {
            "overrides_version": 1,
            "overrides": [{"target": "vs_standard", "id": key, "value": value, "basis": "owner"}],
        }
    )


def _deal_b(playbook: dict[str, Any]) -> dict[str, Any]:
    return next(
        p["signed_text"] for p in playbook["evidence"]["precedent"] if p["document_id"] == "deal_b"
    )


def test_a_part_edit_that_changes_the_label_drops_the_models_reason(tmp_path: Path) -> None:
    """Issue #243 review: an owner who changes a part (here making an
    equivalent text less protective) without a reason must not leave the
    model's reason for the old label beside the re-derived one in any
    model-facing projection — the digest, the dossier, the Review rows."""
    playbook, key = _judged_equivalent(tmp_path)
    assert _deal_b(playbook)["vs_standard"]["label"] == "equivalent"
    edited, _ = fold_overrides(playbook, _set_part_two(key, L), tmp_path)

    vs = _deal_b(edited)["vs_standard"]
    assert (vs["label"], part_map(vs), vs["basis"]) == ("less_protective", ["=", "-"], "owner")
    assert vs["reason"] == "Set by the owner."

    digest = build_digest_v5(edited)
    assert edited["digest"] == digest
    (variant,) = [v for v in digest["clauses"][0]["signed_variants"] if v["part_map"] == ["=", "-"]]
    assert variant["label"] == "less_protective" and "label_by" not in variant
    assert variant["reason"] == "Set by the owner."
    assert _SAME_EFFECT not in json.dumps(digest)

    excerpt = excerpt_vs_standard(_deal_b(edited))
    assert excerpt is not None
    assert (excerpt["label"], excerpt["label_by"]) == ("less_protective", "parts")
    assert excerpt["reason"] == "Set by the owner."
    assert _SAME_EFFECT not in json.dumps(build_derived_sections(edited)["dossiers"])

    row = next(r for r in build_review_rows(edited).rows if r["key"] == key)
    assert row["reason"] == "Set by the owner."

    # Re-projecting replays the stored owner verdict and re-applies the same
    # override: the reason stays the owner's fallback, never the model's.
    store = VerdictStore(tmp_path / "judge" / "verdicts.jsonl")
    replayed = _playbook(StoreBackedEquivalenceJudge(store=store))
    again, _ = fold_overrides(replayed, _set_part_two(key, L), tmp_path)
    assert _deal_b(again)["vs_standard"]["reason"] == "Set by the owner."
    assert validate_document(again, verdict_store=store).ok


def test_a_part_edit_carries_the_owners_reason_to_the_model(tmp_path: Path) -> None:
    """The owner's ``value.reason`` (the Review tab's reason box for changed
    parts) is the reason the digest and the dossier carry."""
    playbook, key = _judged_equivalent(tmp_path)
    why = "The shorter duty leaves our information exposed."
    edited, _ = fold_overrides(playbook, _set_part_two(key, L, why), tmp_path)
    digest = build_digest_v5(edited)
    (variant,) = [v for v in digest["clauses"][0]["signed_variants"] if v["part_map"] == ["=", "-"]]
    assert variant["reason"] == why
    excerpt = excerpt_vs_standard(_deal_b(edited))
    assert excerpt is not None and excerpt["reason"] == why
    assert _SAME_EFFECT not in json.dumps(digest)


def test_a_part_edit_that_keeps_the_map_keeps_the_models_reason(tmp_path: Path) -> None:
    """Re-stating a part as judged changes neither the part map nor the
    label, so the model's reason still describes the map it was written for."""
    playbook, key = _judged_equivalent(tmp_path)
    edited, _ = fold_overrides(playbook, _set_part_two(key, E), tmp_path)
    vs = _deal_b(edited)["vs_standard"]
    assert (vs["label"], vs["basis"], vs["reason"]) == ("equivalent", "owner", _SAME_EFFECT)


def test_an_owner_label_override_keeps_the_parts_and_records_the_reason(tmp_path: Path) -> None:
    playbook, _ = _judged(tmp_path)
    key = _key_of(playbook, _CAND_OMITS_P2)
    entries = parse_overrides(
        {
            "overrides_version": 1,
            "overrides": [
                {
                    "target": "vs_standard",
                    "id": key,
                    "value": {"label": "equivalent", "reason": "Statute supplies the term."},
                    "basis": "owner",
                }
            ],
        }
    )
    edited, _ = fold_overrides(playbook, entries, tmp_path)
    vs = next(
        p["signed_text"]["vs_standard"]
        for p in edited["evidence"]["precedent"]
        if p["document_id"] == "deal_a"
    )
    assert vs["label"] == "equivalent"
    assert vs["owner_label"] == {"label": "equivalent", "reason": "Statute supplies the term."}
    assert part_map(vs) == ["=", "x-"]  # the model's facts stay
    # The owner verdict was written to the run's store, so the edit traces to it.
    store = VerdictStore(tmp_path / "judge" / "verdicts.jsonl")
    assert validate_document(edited, verdict_store=store).ok
    # Confirming the derived label records no owner_label.
    confirm = parse_overrides(
        {
            "overrides_version": 1,
            "overrides": [
                {
                    "target": "vs_standard",
                    "id": key,
                    "value": {"label": "less_protective"},
                    "basis": "owner",
                }
            ],
        }
    )
    copy = json.loads(json.dumps(playbook))
    apply_overrides(copy, confirm)
    vs = next(
        p["signed_text"]["vs_standard"]
        for p in copy["evidence"]["precedent"]
        if p["document_id"] == "deal_a"
    )
    assert vs["label"] == "less_protective" and vs["basis"] == "owner"
    assert "owner_label" not in vs


def test_an_owner_label_reaches_the_digest_dossier_and_review_rows_with_its_reason(
    tmp_path: Path,
) -> None:
    """Issue #243: when the owner sets a label over parts that derive another
    one, every model-facing projection carrying the part map marks the label
    owner-set and carries the owner's reason, never the judgment's reason for
    the label the owner overrode."""
    playbook, _ = _judged(tmp_path)
    key = _key_of(playbook, _CAND_LONGER)  # parts derive mixed: = +, addition -
    owner_reason = "The longer duty outweighs the adviser carve-out for us."
    entries = parse_overrides(
        {
            "overrides_version": 1,
            "overrides": [
                {
                    "target": "vs_standard",
                    "id": key,
                    "value": {"label": "more_protective", "reason": owner_reason},
                    "basis": "owner",
                }
            ],
        }
    )
    edited, _ = fold_overrides(playbook, entries, tmp_path)
    model_reason = "A longer duty but disclosure to advisers is allowed."

    digest = build_digest_v5(edited)
    assert edited["digest"] == digest
    variants = {v["label"]: v for v in digest["clauses"][0]["signed_variants"]}
    owned = variants["more_protective"]
    assert owned["label_by"] == "owner"
    assert owned["reason"] == owner_reason
    assert (owned["part_map"], owned["additions"]) == (["=", "+"], ["-"])
    assert "label_by" not in variants["less_protective"]  # derived from its parts: the default
    assert variants["less_protective"]["reason"] != owner_reason
    assert model_reason not in json.dumps(digest)
    assert digest["part_map_legend"]["label_by"]["owner"]

    # A dossier excerpt of that text carries the same (its two excerpts here are
    # the less protective and the different-concept variants, digest order).
    held = next(
        p["signed_text"] for p in edited["evidence"]["precedent"] if p["document_id"] == "deal_b"
    )
    excerpt = excerpt_vs_standard(held)
    assert excerpt is not None
    assert (excerpt["label_by"], excerpt["reason"]) == ("owner", owner_reason)
    assert excerpt["part_map"] == ["=", "+"]
    (dossier,) = build_derived_sections(edited)["dossiers"].values()
    assert [x["vs_standard"]["label_by"] for x in dossier["excerpts"]] == ["parts", "parts"]

    row = next(r for r in build_review_rows(edited).rows if r["key"] == key)
    assert row["label"] == "more_protective"
    assert row["owner_reason"] == owner_reason
    assert row["derived_label"] == "mixed"


def test_a_parts_override_naming_an_unknown_part_is_rejected(tmp_path: Path) -> None:
    from playbook_engine.overrides import OverridesError

    playbook, _ = _judged(tmp_path)
    key = _key_of(playbook, _CAND_LONGER)
    entries = parse_overrides(
        {
            "overrides_version": 1,
            "overrides": [
                {
                    "target": "vs_standard",
                    "id": key,
                    "value": {
                        "parts": [{"part_id": "deadbeef", "status": "omitted", "effect": "neutral"}]
                    },
                    "basis": "owner",
                }
            ],
        }
    )
    with pytest.raises(OverridesError, match="unknown part_id"):
        apply_overrides(json.loads(json.dumps(playbook)), entries)
