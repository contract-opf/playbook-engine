"""Instrument identity (QA sweep 2026-11-03 #5): an executed copy that is not the
negotiated agreement verifies no absence and records no struck-before-signing claim.

Every document is invented; the texts only carry the general shapes (an amendment,
a cover sheet over terms hosted at a URL, a full agreement).
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from playbook_engine import absence, instrument
from playbook_engine.agent_judge import VerdictStore
from playbook_engine.clause_position_compiler import compile_clause_positions
from playbook_engine.observation_builder import Observation
from playbook_engine.precedent import build_precedent_evidence
from playbook_engine.rubric import RubricStamp, rubric_version
from playbook_engine.scorecard import build_scorecard
from tests.test_absence_counts import (
    _DOCS,
    ASSIGN,
    ASSIGN_STD,
    TERM,
    TERM_STD,
    UNRELATED,
    _doc,
    _evidence,
    _obs,
    _observations,
    _playbook,
    _records,
    _template,
)

COVER_SHEET = (
    "Order Form. Acme Widgets LLC orders the Services listed below. This Order is governed by "
    "the Acme Widgets Terms of Service available at https://example.com/terms, which are "
    "incorporated by reference."
)
AMENDMENT = (
    "First Amendment to the Master Agreement. The parties amend Section 4 to extend the term "
    "by one year. Except as amended hereby, the Agreement remains in full force and effect."
)
FULL_AGREEMENT = " ".join([UNRELATED] * 80)  # long, no instrument cue


def _precedent(obs: list[Observation], doc_ids: list[dict[str, Any]], **kw: Any) -> dict[str, Any]:
    positions, _, _ = compile_clause_positions(obs, _template())
    return build_precedent_evidence(
        agreement_type_id="fixture",
        clause_positions=positions,
        observations=obs,
        corpus_documents=doc_ids,
        party=None,
        **kw,
    )


def test_a_cover_sheet_and_an_amendment_are_other_instruments() -> None:
    assert instrument.instrument_reason(COVER_SHEET) == instrument.INSTRUMENT_INCORPORATION
    assert instrument.instrument_reason(AMENDMENT) == instrument.INSTRUMENT_AMENDMENT
    incorporation = "The Services are governed by the Terms posted at https://example.com/terms."
    assert instrument.instrument_reason(incorporation) == instrument.INSTRUMENT_INCORPORATION


def test_a_plain_agreement_or_a_short_one_without_a_cue_is_not_flagged() -> None:
    assert instrument.instrument_reason(FULL_AGREEMENT) is None
    assert instrument.instrument_reason(UNRELATED) is None
    assert instrument.instrument_reason(TERM_STD) is None
    # A long agreement that merely says "remain in full force" (survival) is not an amendment.
    long_survival = FULL_AGREEMENT + " The rest shall remain in full force and effect."
    assert instrument.instrument_reason(long_survival) is None


def test_a_form_switch_needs_a_cue_and_a_much_shorter_copy() -> None:
    mid = (UNRELATED + " ") * 60 + " The parties ratify the above."
    assert len(mid) > instrument.SHORT_INSTRUMENT_CHARS
    assert instrument.instrument_reason(mid, earlier_chars=len(mid) * 5) == "form_switch"
    assert instrument.instrument_reason(mid, earlier_chars=len(mid)) is None


def test_the_flags_are_read_off_the_executed_copy_per_deal() -> None:
    obs = [
        _obs("deal-x", None, COVER_SHEET, path="1"),
        _obs("deal-x", TERM, TERM_STD, path="2", outcome="opening"),
        _obs("deal-ok", None, UNRELATED, path="1"),
    ]
    assert instrument.instrument_flags(obs) == {"deal-x": instrument.INSTRUMENT_INCORPORATION}


def test_an_other_instrument_copy_never_verifies_an_absence() -> None:
    obs = [
        *_observations(),
        _obs("deal-x", None, COVER_SHEET, path="1"),
        _obs("deal-y", TERM, TERM_STD, path="2"),
        _obs("deal-y", None, AMENDMENT, path="3"),
    ]
    records = _records(_precedent(obs, [*_DOCS, _doc("deal-x"), _doc("deal-y")]))
    for key in (("deal-x", ASSIGN), ("deal-x", TERM), ("deal-y", ASSIGN)):
        assert records[key]["status"] == absence.STATUS_POSSIBLE
        assert records[key]["basis"] == absence.BASIS_UNVERIFIABLE
    # an ordinary deal beside them is unchanged
    assert _records(_evidence())[("deal-b", ASSIGN)]["status"] == absence.STATUS_VERIFIED
    assert records[("deal-b", ASSIGN)]["status"] == absence.STATUS_VERIFIED


def _opening_case(signed_copy: str) -> list[Observation]:
    copy = _obs("deal-s", None, signed_copy, path="1")
    opening = replace(
        _obs("deal-s", ASSIGN, ASSIGN_STD, path="9", outcome="opening"), opened_with="standard"
    )
    refused = _obs(
        "deal-s",
        ASSIGN,
        "Either party may freely assign this Agreement to any affiliate or successor without notice or consent.",
        path="8",
        outcome="proposed_then_reversed",
    )
    return [*_observations(), copy, opening, refused, _obs("deal-s", TERM, TERM_STD, path="2")]


def _deal_s(obs: list[Observation]) -> dict[str, dict[str, Any]]:
    evidence = _precedent(obs, [*_DOCS, _doc("deal-s")], absence=False)
    return {p["taxonomy_id"]: p for p in evidence["precedent"] if p["document_id"] == "deal-s"}


def test_no_struck_before_signing_record_when_the_signed_instrument_is_another_document() -> None:
    plain = _deal_s(_opening_case(UNRELATED))
    assert plain[ASSIGN]["signed_text"] is None
    assert plain[ASSIGN]["opened_with"] == "standard"
    assert plain[ASSIGN]["refused_asks"]

    other = _deal_s(_opening_case(AMENDMENT))
    assert ASSIGN not in other  # no text in the amendment, nothing claimed about it
    assert other[TERM]["signed_text"] is not None  # what the amendment does say stays
    assert other[TERM]["opened_with"] is None
    assert other[TERM]["refused_asks"] == [] and other[TERM]["rounds"] == 0


def test_the_scorecard_scores_the_judges_reason_not_just_the_boolean(tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    playbook = _playbook()
    (out / "playbook.opf.json").write_text(json.dumps(playbook), encoding="utf-8")
    (out / "observations.jsonl").write_text(
        "".join(json.dumps(o.to_dict()) + "\n" for o in _observations()), encoding="utf-8"
    )
    subjects = absence.check_subjects(
        clauses=playbook["evidence"]["clauses"],
        records=playbook["evidence"]["absence"]["records"],
        observations=_observations(),
    )
    store = VerdictStore(out / "judge" / "verdicts.jsonl")
    current = RubricStamp(kind=absence.CHECK_KIND, version=rubric_version(absence.CHECK_KIND))
    store.put_by_key(subjects[0].key, {"absent": True, "reason": "No such clause."}, rubric=current)
    store.put_by_key(
        subjects[1].key,
        {"absent": True, "reason": "Short amendment; incorporates separate terms by reference."},
        rubric=current,
    )
    section = build_scorecard(out)["absence"]
    assert section["checked"] == 2
    assert section["reason_flagged"] == 1
    assert section["wrong"] == 1


def test_reason_wording() -> None:
    assert instrument.reason_flags_instrument("This is a cover sheet over hosted terms")
    assert not instrument.reason_flags_instrument("No assignment language appears.")
    assert not instrument.reason_flags_instrument(None)


def test_full_agreement_boilerplate_is_not_another_instrument() -> None:
    """Severability, an Amendment clause and a defined Order Form are boilerplate."""
    text = (
        "Mutual Non-Disclosure Agreement\n"
        "Amendment\n"
        "This Agreement may be changed only by written amendment signed by both parties.\n"
        "Amendment. No waiver is effective unless in writing.\n"
        "Order Form. Each Order Form placed by Customer forms part of this Agreement.\n"
        "If any provision is held invalid, the remaining provisions remain in full force "
        "and effect.\n"
        "The confidentiality obligations survive termination and remain in full force "
        "and effect."
    )
    assert instrument.instrument_reason(text) is None


def test_title_lines_flag_the_instrument() -> None:
    for title in (
        "Amendment No. 1",
        "First Amendment to Master Services Agreement",
        "Addendum to the Services Agreement",
        "ORDER FORM",
        "Statement of Work No. 3",
    ):
        assert instrument.instrument_reason(title + "\nThe fee is changed.") is not None, title


def test_reason_saying_no_amendment_clause_is_not_a_flag() -> None:
    assert not instrument.reason_flags_instrument("The text holds no amendment provision.")
    assert instrument.reason_flags_instrument("A short amendment that incorporates the main terms.")
