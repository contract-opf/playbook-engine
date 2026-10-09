"""The `vs_standard` equivalence label (issue #240).

Each distinct non-standard text of the precedent record (a deal's signed text,
a non-standard opening, a refused ask) is judged ONCE against our standard and
carries the answer. These tests cover the key, the queue, the store-backed
judge, the blind check and adjudication round, the projection into the
playbook, the validator rules and the scorecard section.

Fixture provenance (the producer of every shape used here):
- the evidence under test is built by the production path --
  ``assemble_playbook`` -> ``build_precedent_evidence`` over ``Observation``
  rows of the shapes ``observation_builder`` writes, or a real
  ``mine`` -> ``judge`` -> ``project`` run over the synthetic NDA example;
- verdicts are the dicts ``playbook judge-apply --verdicts`` accepts and check
  records the dicts ``playbook judge-apply --check`` accepts; the fake
  Anthropic client mimics the Message Batches surface the real one exposes.

SECURITY NOTE: synthetic text only (examples/nda and invented sentences).
"""

from __future__ import annotations

import dataclasses
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from click.testing import CliRunner
from click.testing import Result as CliResult

from playbook_engine.agent_judge import (
    PendingQueue,
    StoreBackedEquivalenceJudge,
    VerdictStore,
    infer_verdict_kind,
    validate_verdict,
)
from playbook_engine.clause_position_compiler import compile_clause_positions
from playbook_engine.cli import cli
from playbook_engine.equivalence import (
    collect_subjects,
    equivalence_key,
    iter_slots,
    label_evidence,
    summarize,
)
from playbook_engine.equivalence_check import (
    CHECKER_EFFORT,
    CHECKER_MODEL,
    apply_check_records,
    build_check_queues,
    check_via_api,
    load_check_records,
)
from playbook_engine.observation_builder import Observation, ObservationCitation
from playbook_engine.playbook_assembler import assemble_playbook
from playbook_engine.rubric import RubricPolicy, rubric_version
from playbook_engine.scorecard import build_scorecard
from playbook_engine.validator import validate_document

_ROOT = Path(__file__).resolve().parent.parent
_NDA_DIR = _ROOT / "examples" / "nda"
_CORPUS_DIR = _NDA_DIR / "corpus"
_SMOKE_CONFIG = _NDA_DIR / "config.smoke.yaml"
_CANNED = _NDA_DIR / "canned-verdicts.jsonl"

_PARTY = "Acme Corp"
_STANDARD = "Each party shall keep the other party's information confidential for three years."
_SIGNED_VARIANT = "Each party shall keep the other party's information confidential for five years."
_OPENING_VARIANT = "Each party shall keep the other party's information confidential."
_REFUSED_ASK = "The receiving party may use the information for any purpose it sees fit."


def _invoke(args: list[str]) -> CliResult:
    return CliRunner().invoke(cli, args)


# ---------------------------------------------------------------------------
# Evidence built by the production assembler
# ---------------------------------------------------------------------------


def _obs(
    doc_id: str,
    text: str,
    *,
    outcome: str = "signed",
    standard: bool = False,
    opened_with: str | None = None,
    version: str = "v2",
    taxonomy_id: str = "survival_period",
    provenance: str = "our_paper",
) -> Observation:
    return Observation(
        observation_id=f"{doc_id}/{version}/8",
        taxonomy_id=taxonomy_id,
        text_summary=text,
        full_text=text,
        citation=ObservationCitation(
            document_id=doc_id, version=version, clause_path="8", char_span=None
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
        "version_order_basis": "edit_distance_chain",
    }


_TAXONOMY = {
    "source": "test",
    "entries": [
        {
            "id": "survival_period",
            "label": "Survival Period",
            "status": "active",
            "cuad_origin": "x",
            "description": "How long the duty lasts.",
        }
    ],
}


def _assemble(
    observations: list[Observation],
    docs: list[dict[str, Any]],
    *,
    template: bool = True,
    judge: StoreBackedEquivalenceJudge | None = None,
) -> dict[str, Any]:
    template_obs = (
        [
            _obs("template", _STANDARD, version="template", standard=True),
        ]
        if template
        else []
    )
    positions, _, _ = compile_clause_positions(observations, template_obs)
    return assemble_playbook(
        agreement_type={"id": "test-agreement", "name": "Test Agreement"},
        baseline={"has_canonical_template": template},
        taxonomy=_TAXONOMY,
        clause_positions=positions,
        corpus_documents=docs,
        generated_at="2026-10-09T00:00:00Z",
        observations=observations,
        perspective={"party": _PARTY, "counterparty_type": "vendor"},
        equivalence_judge=judge,
    )


def _two_paper_playbook(judge: StoreBackedEquivalenceJudge | None = None) -> dict[str, Any]:
    """Three deals: one signed our standard, two signed the SAME variant text
    (one on our paper, one on counterparty paper)."""
    observations = [
        _obs("deal_a", _STANDARD, standard=True, opened_with="standard"),
        _obs("deal_b", _SIGNED_VARIANT, opened_with="non_standard"),
        _obs(
            "deal_c", _SIGNED_VARIANT, opened_with="non_standard", provenance="counterparty_paper"
        ),
    ]
    docs = [
        _doc("deal_a", "our_paper"),
        _doc("deal_b", "our_paper"),
        _doc("deal_c", "counterparty_paper"),
    ]
    return _assemble(observations, docs, judge=judge)


# ---------------------------------------------------------------------------
# Key
# ---------------------------------------------------------------------------


def test_key_is_over_the_five_fields_and_ignores_everything_else() -> None:
    key = equivalence_key("nda", "survival_period", _PARTY, _SIGNED_VARIANT, _STANDARD)
    assert key == equivalence_key("nda", "survival_period", _PARTY, _SIGNED_VARIANT, _STANDARD)
    # Whitespace, case and punctuation are not part of the text identity.
    assert key == equivalence_key(
        "nda",
        "survival_period",
        _PARTY,
        "  EACH party shall keep the other party's information confidential, for five years  ",
        _STANDARD,
    )
    # Our party's name is neutralized, a counterparty alias too.
    assert equivalence_key(
        "nda", "survival_period", _PARTY, f"{_PARTY} shall pay Counterparty-7.", _STANDARD
    ) == equivalence_key(
        "nda", "survival_period", _PARTY, f"{_PARTY} shall pay Counterparty-12.", _STANDARD
    )
    # Each of the five fields moves it.
    base = ("nda", "survival_period", _PARTY, _SIGNED_VARIANT, _STANDARD)
    for i, other in enumerate(("other", "other", "Other Corp", "A different text.", "Other std.")):
        args = list(base)
        args[i] = other
        assert equivalence_key(*args) != key, i  # type: ignore[arg-type]


def test_same_text_in_two_deals_on_both_paper_sides_is_one_subject_and_one_verdict(
    tmp_path: Path,
) -> None:
    """One queue item, one shared verdict, whichever paper the deal was on."""
    store = VerdictStore(tmp_path / "judge" / "verdicts.jsonl")
    pending = PendingQueue(tmp_path / "judge" / "pending.jsonl")
    judge = StoreBackedEquivalenceJudge(store=store, pending=pending)
    playbook = _two_paper_playbook()
    papers = {p["paper"] for p in playbook["evidence"]["precedent"]}
    assert papers == {"ours", "theirs"}

    subjects = collect_subjects(playbook["evidence"], "test-agreement", _PARTY)
    assert len(subjects) == 1
    assert subjects[0].roles == ("signed",)
    # Nothing that identifies a deal or a paper side is in what the judge sees.
    blob = json.dumps(subjects[0].payload)
    for needle in ("deal_a", "deal_b", "deal_c", "ours", "theirs", "paper"):
        assert needle not in blob, needle

    assert judge.judge(subjects[0]) is None
    assert judge.judge(subjects[0]) is None  # asked twice, queued once
    queued = [json.loads(line) for line in pending._queue_path.read_text().splitlines()]
    assert len(queued) == 1 and queued[0]["kind"] == "equivalence"

    store.put_by_key(
        subjects[0].key,
        {"label": "more_protective", "reason": "Five years instead of three.", "basis": "agent"},
    )
    labelled = _two_paper_playbook(StoreBackedEquivalenceJudge(store=store))
    by_deal = {p["document_id"]: p for p in labelled["evidence"]["precedent"]}
    assert (
        by_deal["deal_b"]["signed_text"]["vs_standard"]
        == (by_deal["deal_c"]["signed_text"]["vs_standard"])
    )
    assert by_deal["deal_b"]["signed_text"]["vs_standard"]["label"] == "more_protective"
    assert validate_document(labelled, verdict_store=store).ok


# ---------------------------------------------------------------------------
# What is never judged
# ---------------------------------------------------------------------------


def test_a_standard_true_text_is_never_queued_and_stays_null() -> None:
    playbook = _two_paper_playbook()
    by_deal = {p["document_id"]: p for p in playbook["evidence"]["precedent"]}
    assert by_deal["deal_a"]["standard"] is True
    assert by_deal["deal_a"]["signed_text"]["vs_standard"] is None
    keys = {s.key for s in collect_subjects(playbook["evidence"], "test-agreement", _PARTY)}
    standard_key = equivalence_key(
        "test-agreement", "survival_period", _PARTY, _STANDARD, _STANDARD
    )
    assert standard_key not in keys


def test_standard_true_beats_the_text_comparison() -> None:
    """A signed record can be ``standard: true`` while its §3.5.4 grouping key
    differs from our standard's, and ``standard: true`` is honoured then too.

    The producer sets a signed record's ``standard`` from the observation's
    standard fact (``observation_builder``'s ``_standard_fact``, carried by
    ``precedent.build_precedent_evidence``), which compares the text with the
    WHOLE joined template clause after ``normalize_for_standard``. That
    normalization neutralizes the configured ``provenance.our_party_aliases``
    and ``known_entities`` (for example "AlphaCorp"), a list the playbook does
    not record, so the §3.5.4 grouping key — which neutralizes only
    ``perspective.party`` and the ``Counterparty-<n>`` aliases — still tells
    the two texts apart."""
    from playbook_engine.precedent import restamp_evidence

    playbook = _two_paper_playbook()
    assert collect_subjects(playbook["evidence"], "test-agreement", _PARTY)
    for record in playbook["evidence"]["precedent"]:
        if record["document_id"] in ("deal_b", "deal_c"):
            record["standard"] = True
    restamp_evidence(playbook["evidence"], "test-agreement", party=_PARTY)
    assert validate_document(_rehash(playbook)).ok  # a state the validator accepts
    assert collect_subjects(playbook["evidence"], "test-agreement", _PARTY) == []


def test_emergent_mode_queues_nothing_and_every_label_is_null() -> None:
    observations = [_obs("deal_b", _SIGNED_VARIANT), _obs("deal_c", _SIGNED_VARIANT)]
    docs = [_doc("deal_b", "our_paper"), _doc("deal_c", "counterparty_paper")]
    playbook = _assemble(observations, docs, template=False)
    assert all(c["our_standard"] is None for c in playbook["evidence"]["clauses"])
    assert collect_subjects(playbook["evidence"], "test-agreement", _PARTY) == []
    slots = list(iter_slots(playbook["evidence"]))
    assert slots and all(slot.entry["vs_standard"] is None for slot in slots)
    assert validate_document(playbook).ok


def test_an_opening_that_is_our_standard_is_not_judged_but_a_non_standard_one_is() -> None:
    """opened_with decides: ``standard`` openings are equivalent by definition."""
    base = _two_paper_playbook()
    evidence = json.loads(json.dumps(base["evidence"]))
    record = next(p for p in evidence["precedent"] if p["document_id"] == "deal_b")
    record["opening_text"] = {"text": _OPENING_VARIANT, "ref": record["signed_text"]["ref"]}
    record["moved"] = True
    record["opened_with"] = "non_standard"
    roles = {s.roles for s in collect_subjects(evidence, "test-agreement", _PARTY)}
    assert ("opening",) in roles
    record["opened_with"] = "standard"
    assert ("opening",) not in {
        s.roles for s in collect_subjects(evidence, "test-agreement", _PARTY)
    }


def test_a_text_playing_two_roles_across_records_is_one_subject_with_both_roles() -> None:
    """One deal signed a text another deal opened with: the producer's shape."""
    observations = [
        _obs("deal_a", _STANDARD, standard=True, opened_with="standard"),
        _obs("deal_b", _SIGNED_VARIANT, opened_with="non_standard"),
        _obs(
            "deal_b",
            _REFUSED_ASK,
            outcome="proposed_then_reversed",
            opened_with="non_standard",
            version="v1",
        ),
        # deal_d signed something else but opened with deal_b's signed text.
        _obs("deal_d", _OPENING_VARIANT, opened_with="non_standard"),
        _obs(
            "deal_d",
            _SIGNED_VARIANT,
            outcome="opening",
            opened_with="non_standard",
            version="v1",
        ),
    ]
    docs = [_doc(d, "our_paper") for d in ("deal_a", "deal_b", "deal_d")]
    playbook = _assemble(observations, docs)
    assert validate_document(playbook).ok
    record_d = next(p for p in playbook["evidence"]["precedent"] if p["document_id"] == "deal_d")
    assert record_d["moved"] is True and record_d["opening_text"] is not None
    subjects = collect_subjects(playbook["evidence"], "test-agreement", _PARTY)
    assert len(subjects) == 3
    by_text = {s.payload["candidate"]: s for s in subjects}
    assert by_text[_SIGNED_VARIANT].roles == ("signed", "opening")
    assert by_text[_OPENING_VARIANT].roles == ("signed",)
    assert by_text[_REFUSED_ASK].roles == ("refused",)


# ---------------------------------------------------------------------------
# Unjudged is null, never guessed; checks never block
# ---------------------------------------------------------------------------


def test_an_unjudged_text_is_null_not_guessed(tmp_path: Path) -> None:
    store = VerdictStore(tmp_path / "judge" / "verdicts.jsonl")
    playbook = _two_paper_playbook(StoreBackedEquivalenceJudge(store=store))
    variants = [
        p["signed_text"]["vs_standard"]
        for p in playbook["evidence"]["precedent"]
        if p["standard"] is False
    ]
    assert variants == [None, None]
    summary = summarize(playbook["evidence"], "test-agreement", _PARTY)
    assert summary["totals"]["eligible"] == 1 and summary["totals"]["unjudged"] == 1
    assert validate_document(playbook).ok


def test_an_unchecked_draft_still_reaches_the_playbook_with_check_null(tmp_path: Path) -> None:
    store = VerdictStore(tmp_path / "judge" / "verdicts.jsonl")
    key = equivalence_key("test-agreement", "survival_period", _PARTY, _SIGNED_VARIANT, _STANDARD)
    store.put_by_key(
        key, {"label": "more_protective", "reason": "Longer survival.", "basis": "agent"}
    )
    playbook = _two_paper_playbook(StoreBackedEquivalenceJudge(store=store))
    vs = next(
        p["signed_text"]["vs_standard"]
        for p in playbook["evidence"]["precedent"]
        if p["document_id"] == "deal_b"
    )
    assert vs == {
        "label": "more_protective",
        "reason": "Longer survival.",
        "basis": "agent",
        "check": None,
    }
    totals = summarize(playbook["evidence"], "test-agreement", _PARTY)["totals"]
    assert (totals["drafted"], totals["checked"], totals["unchecked"]) == (1, 0, 1)


def test_a_stale_rubric_requeues_instead_of_replaying(tmp_path: Path) -> None:
    store = VerdictStore(tmp_path / "judge" / "verdicts.jsonl")
    playbook = _two_paper_playbook()
    (subject,) = collect_subjects(playbook["evidence"], "test-agreement", _PARTY)
    from playbook_engine.rubric import RubricStamp

    store.put_by_key(
        subject.key,
        {"label": "equivalent", "reason": "Same.", "basis": "agent"},
        rubric=RubricStamp(kind="equivalence", version="v0+stale"),
    )
    pending = PendingQueue(tmp_path / "pending.jsonl")
    judge = StoreBackedEquivalenceJudge(store=store, pending=pending, rubric=RubricPolicy())
    assert judge.judge(subject) is None
    record = json.loads((tmp_path / "pending.jsonl").read_text().splitlines()[0])
    assert record["rubric_version"] == rubric_version("equivalence")
    # Accepted staleness replays.
    accepting = StoreBackedEquivalenceJudge(store=store, rubric=RubricPolicy(accept_stale=True))
    assert accepting.judge(subject) is not None


# ---------------------------------------------------------------------------
# Apply-time verdict validation
# ---------------------------------------------------------------------------


def test_a_draft_verdict_needs_label_reason_and_basis_and_nothing_else() -> None:
    good = {"label": "equivalent", "reason": "Same effect.", "basis": "agent"}
    validate_verdict("equivalence", good)
    assert infer_verdict_kind(good) == "equivalence"
    for bad in (
        {**good, "label": "better"},
        {**good, "reason": "  "},
        {"label": "equivalent", "reason": "x"},  # basis missing
        {**good, "basis": "needs_review"},
        {**good, "basis": "guess"},
        {**good, "check": {"agreed": True, "by": "x", "adjudicated": False}},
    ):
        with pytest.raises(ValueError):
            validate_verdict("equivalence", bad)


# ---------------------------------------------------------------------------
# Blind check + adjudication (store level)
# ---------------------------------------------------------------------------


def _drafted_store(tmp_path: Path) -> tuple[VerdictStore, Any]:
    store = VerdictStore(tmp_path / "judge" / "verdicts.jsonl")
    playbook = _two_paper_playbook()
    (subject,) = collect_subjects(playbook["evidence"], "test-agreement", _PARTY)
    store.put_by_key(
        subject.key,
        {"label": "more_protective", "reason": "DRAFT-REASON-SENTINEL", "basis": "agent"},
    )
    return store, subject


def _answer(key: str, label: str, reason: str = "Checker reason.", **over: str) -> dict[str, str]:
    return {
        "key": key,
        "label": label,
        "reason": reason,
        "model": CHECKER_MODEL,
        "effort": CHECKER_EFFORT,
        **over,
    }


def test_a_check_item_never_contains_the_drafted_label_or_reason(tmp_path: Path) -> None:
    store, subject = _drafted_store(tmp_path)
    queues = build_check_queues(store, [subject])
    assert len(queues.check) == 1 and queues.adjudication == []
    item = queues.check[0]
    assert item["payload"] == subject.payload  # exactly what the drafter saw
    blob = json.dumps(item)
    assert "DRAFT-REASON-SENTINEL" not in blob
    assert "more_protective" not in blob
    assert not {"label", "reason", "basis", "draft", "check"} & set(item["payload"])


def test_check_records_with_the_wrong_model_or_effort_are_rejected_unless_overridden(
    tmp_path: Path,
) -> None:
    store, subject = _drafted_store(tmp_path)
    for bad in (
        _answer(subject.key, "more_protective", model="claude-sonnet-5-5"),
        _answer(subject.key, "more_protective", effort="high"),
    ):
        with pytest.raises(ValueError, match="allow-checker-model"):
            apply_check_records(store, [(1, bad)])
    result = apply_check_records(
        store,
        [(1, _answer(subject.key, "more_protective", model="other-model", effort="high"))],
        allow_checker_model=True,
    )
    ((_key, verdict),) = result.updates
    assert verdict["check"] == {"agreed": True, "by": "other-model", "adjudicated": False}
    assert store.get_by_key(subject.key)["reason"] == "DRAFT-REASON-SENTINEL"  # store untouched


def test_agreement_keeps_the_verdict_and_records_who_checked(tmp_path: Path) -> None:
    store, subject = _drafted_store(tmp_path)
    result = apply_check_records(store, [(1, _answer(subject.key, "more_protective"))])
    assert (result.agreed, result.disagreed) == (1, 0)
    ((_k, verdict),) = result.updates
    assert verdict["label"] == "more_protective" and verdict["reason"] == "DRAFT-REASON-SENTINEL"
    assert verdict["check"] == {"agreed": True, "by": CHECKER_MODEL, "adjudicated": False}


def test_a_disagreement_produces_an_adjudication_item_and_its_answer_becomes_the_verdict(
    tmp_path: Path,
) -> None:
    store, subject = _drafted_store(tmp_path)
    first = apply_check_records(store, [(1, _answer(subject.key, "equivalent", "CHECK-REASON"))])
    assert first.disagreed == 1
    for key, verdict in first.updates:
        store.put_by_key(key, verdict)
    # The draft still flows through, flagged as disputed.
    disputed = store.get_by_key(subject.key)
    assert disputed["label"] == "more_protective"
    assert disputed["check"] == {"agreed": False, "by": CHECKER_MODEL, "adjudicated": False}

    queues = build_check_queues(store, [subject])
    assert queues.check == [] and len(queues.adjudication) == 1
    payload = queues.adjudication[0]["payload"]
    assert payload["draft"] == {"label": "more_protective", "reason": "DRAFT-REASON-SENTINEL"}
    assert payload["check"] == {"label": "equivalent", "reason": "CHECK-REASON"}
    assert payload["our_standard"] == _STANDARD and payload["candidate"] == _SIGNED_VARIANT

    second = apply_check_records(
        store, [(1, _answer(subject.key, "less_protective", "ADJUDICATION-REASON"))]
    )
    assert second.adjudicated == 1
    for key, verdict in second.updates:
        store.put_by_key(key, verdict)
    final = store.get_by_key(subject.key)
    assert (final["label"], final["reason"]) == ("less_protective", "ADJUDICATION-REASON")
    assert final["check"] == {"agreed": False, "by": CHECKER_MODEL, "adjudicated": True}
    assert build_check_queues(store, [subject]).settled == 1
    # The draft and the checker's answer stay in the trail.
    assert final["draft"]["label"] == "more_protective"
    assert final["checker"]["label"] == "equivalent"
    # A settled verdict cannot be checked again.
    with pytest.raises(ValueError, match="already settled"):
        apply_check_records(store, [(1, _answer(subject.key, "equivalent"))])


def test_an_owner_correction_wins_and_is_never_checked(tmp_path: Path) -> None:
    store, subject = _drafted_store(tmp_path)
    store.put_by_key(
        subject.key, {"label": "equivalent", "reason": "Owner view.", "basis": "owner"}
    )
    assert build_check_queues(store, [subject]).check == []
    result = apply_check_records(store, [(1, _answer(subject.key, "less_protective"))])
    assert result.skipped_owner == 1 and result.updates == []


def test_a_check_for_an_undrafted_key_is_rejected_with_its_line_number(tmp_path: Path) -> None:
    store = VerdictStore(tmp_path / "judge" / "verdicts.jsonl")
    with pytest.raises(ValueError, match="line 3: no drafted equivalence verdict"):
        apply_check_records(store, [(3, _answer("0" * 64, "equivalent"))])


def test_a_key_repeated_in_one_check_file_is_rejected_and_nothing_is_written(
    tmp_path: Path,
) -> None:
    """One file can never take a verdict through both the blind check and the
    adjudication: a second line for the same key would otherwise be read as
    the adjudication, answered blind, with no adjudication item ever emitted."""
    store, subject = _drafted_store(tmp_path)  # the out-dir is tmp_path
    records = [
        (1, _answer(subject.key, "equivalent", "CHECK-REASON")),
        (2, _answer(subject.key, "less_protective", "NOT-AN-ADJUDICATION")),
    ]
    with pytest.raises(ValueError, match=r"line 2: key .* already answered on line 1"):
        apply_check_records(store, records)

    verdicts_path = tmp_path / "judge" / "verdicts.jsonl"
    before = verdicts_path.read_bytes()
    answers = tmp_path / "checks.jsonl"
    answers.write_text("".join(json.dumps(r) + "\n" for _, r in records), encoding="utf-8")
    result = _invoke(["judge-apply", str(tmp_path), "--check", str(answers)])
    assert result.exit_code == 1, result.output
    assert "line 2" in result.output and "already answered on line 1" in result.output
    assert verdicts_path.read_bytes() == before  # nothing written to the store
    stored = VerdictStore(verdicts_path).get_by_key(subject.key)
    assert stored == {
        "label": "more_protective",
        "reason": "DRAFT-REASON-SENTINEL",
        "basis": "agent",
    }
    # The check alone still applies, and leaves the dispute for the adjudication queue.
    first = apply_check_records(store, records[:1])
    assert (first.disagreed, first.adjudicated) == (1, 0)


def test_check_file_parsing_names_the_bad_line(tmp_path: Path) -> None:
    path = tmp_path / "check.jsonl"
    path.write_text(json.dumps({"key": "k", "label": "equivalent"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 1.*missing"):
        load_check_records(path)


# ---------------------------------------------------------------------------
# Optional keyed API path (fake client; CI never calls the API)
# ---------------------------------------------------------------------------


class _FakeBatches:
    def __init__(self, results: list[SimpleNamespace]) -> None:
        self.created: list[dict[str, Any]] = []
        self._results = results

    def create(self, *, requests: list[dict[str, Any]]) -> SimpleNamespace:
        self.created = requests
        return SimpleNamespace(id="batch_1", processing_status="ended")

    def retrieve(self, batch_id: str) -> SimpleNamespace:  # pragma: no cover - already ended
        return SimpleNamespace(id=batch_id, processing_status="ended")

    def results(self, batch_id: str) -> list[SimpleNamespace]:
        return self._results


def _message(text: str, *, stop_reason: str = "end_turn", model: str = CHECKER_MODEL) -> Any:
    return SimpleNamespace(
        type="succeeded",
        message=SimpleNamespace(
            stop_reason=stop_reason,
            model=model,
            content=[
                SimpleNamespace(type="thinking", thinking="..."),
                SimpleNamespace(type="text", text=text),
            ],
        ),
    )


def test_api_path_uses_the_pinned_settings_and_records_the_model_it_ran_on() -> None:
    answer = json.dumps({"label": "equivalent", "reason": "Same effect."})
    results = [
        SimpleNamespace(custom_id="k1", result=_message(answer, model="claude-opus-5-5")),
        SimpleNamespace(
            custom_id="k2", result=_message("", stop_reason="refusal", model="claude-opus-5-5")
        ),
    ]
    batches = _FakeBatches(results)
    client = SimpleNamespace(messages=SimpleNamespace(batches=batches))
    items = [
        {"key": "k1", "kind": "equivalence_check", "payload": {"candidate": "a"}},
        {"key": "k2", "kind": "equivalence_check", "payload": {"candidate": "b"}},
        {"key": "k3", "kind": "equivalence_check", "payload": {"candidate": "c"}},
    ]
    records, unchecked = check_via_api(items, client=client, poll_interval_s=0)
    assert records == [
        {
            "key": "k1",
            "label": "equivalent",
            "reason": "Same effect.",
            "model": "claude-opus-5-5",
            "effort": "xhigh",
        }
    ]
    assert unchecked == 2  # the refusal, and the item the batch never answered
    for request in batches.created:
        params = request["params"]
        assert params["model"] == "claude-opus-5-5"
        assert params["output_config"]["effort"] == "xhigh"
        assert params["output_config"]["format"]["type"] == "json_schema"
        # Forbidden on this model: a thinking parameter, a forced tool choice,
        # and any fallback model.
        assert not {"thinking", "tool_choice", "fallback", "fallbacks"} & set(params)


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


def test_validator_rejects_a_label_the_store_never_produced(tmp_path: Path) -> None:
    store = VerdictStore(tmp_path / "judge" / "verdicts.jsonl")
    key = equivalence_key("test-agreement", "survival_period", _PARTY, _SIGNED_VARIANT, _STANDARD)
    store.put_by_key(key, {"label": "equivalent", "reason": "Same.", "basis": "agent"})
    playbook = _two_paper_playbook(StoreBackedEquivalenceJudge(store=store))
    assert validate_document(playbook, verdict_store=store).ok
    assert validate_document(playbook).ok  # no store: the cross-check is skipped

    empty = VerdictStore(tmp_path / "empty" / "verdicts.jsonl")
    result = validate_document(playbook, verdict_store=empty)
    assert not result.ok
    assert any("verdict store has no equivalence verdict" in str(e) for e in result.errors)

    other = VerdictStore(tmp_path / "other" / "verdicts.jsonl")
    other.put_by_key(key, {"label": "less_protective", "reason": "x.", "basis": "agent"})
    result = validate_document(playbook, verdict_store=other)
    assert any("differs from the stored verdict" in str(e) for e in result.errors)


def test_validator_rejects_a_label_on_a_standard_true_text() -> None:
    playbook = _two_paper_playbook()
    record = next(p for p in playbook["evidence"]["precedent"] if p["standard"] is True)
    record["signed_text"]["vs_standard"] = {
        "label": "equivalent",
        "reason": "Identical.",
        "basis": "agent",
        "check": None,
    }
    result = validate_document(_rehash(playbook))
    assert any("standard fact is true" in str(e) for e in result.errors)


def test_validator_rejects_a_label_on_a_clause_with_no_our_standard() -> None:
    observations = [_obs("deal_b", _SIGNED_VARIANT)]
    playbook = _assemble(observations, [_doc("deal_b", "our_paper")], template=False)
    record = playbook["evidence"]["precedent"][0]
    record["signed_text"]["vs_standard"] = {
        "label": "equivalent",
        "reason": "x.",
        "basis": "agent",
        "check": None,
    }
    result = validate_document(_rehash(playbook))
    assert any("no our_standard" in str(e) for e in result.errors)


def test_validator_rejects_an_adjudicated_check_that_agreed() -> None:
    """OPF-SPEC §3.5.6: ``check.adjudicated`` true implies ``check.agreed`` false."""
    playbook = _two_paper_playbook()
    record = next(p for p in playbook["evidence"]["precedent"] if p["document_id"] == "deal_b")
    vs: dict[str, Any] = {
        "label": "less_protective",
        "reason": "Shorter protection than our standard.",
        "basis": "agent",
        "check": {"agreed": False, "by": CHECKER_MODEL, "adjudicated": True},
    }
    record["signed_text"]["vs_standard"] = vs
    assert validate_document(_rehash(playbook)).ok  # the legal adjudicated shape

    vs["check"] = {"agreed": True, "by": CHECKER_MODEL, "adjudicated": True}
    result = validate_document(_rehash(playbook))
    assert not result.ok
    assert any(
        "adjudicated but agreed is not false" in str(e)
        and "evidence.precedent[" in str(e)
        and "signed_text.vs_standard.check" in str(e)
        for e in result.errors
    ), [str(e) for e in result.errors]


def test_validator_schema_rejects_a_malformed_label() -> None:
    playbook = _two_paper_playbook()
    record = next(p for p in playbook["evidence"]["precedent"] if p["standard"] is False)
    record["signed_text"]["vs_standard"] = {"label": "better", "reason": "x", "basis": "agent"}
    result = validate_document(_rehash(playbook))
    assert not result.ok


def _rehash(playbook: dict[str, Any]) -> dict[str, Any]:
    """Re-stamp the sidecar, digest and identity after a test edited the evidence."""
    from playbook_engine.canonicalize import compute_section_digests, content_hash
    from playbook_engine.digest import build_digest
    from playbook_engine.opf_accessors import SIDECARS_KEY, precedent_sidecar_manifest

    playbook[SIDECARS_KEY] = precedent_sidecar_manifest(playbook)
    playbook["digest"] = build_digest(playbook)
    playbook["identity"]["content_hash"] = content_hash(playbook)
    playbook["identity"]["section_digests"] = compute_section_digests(playbook)
    return playbook


# ---------------------------------------------------------------------------
# End to end over the NDA example: mine -> judge -> draft -> check -> project
# ---------------------------------------------------------------------------


def _non_equivalence_canned() -> list[str]:
    return [
        line
        for line in _CANNED.read_text(encoding="utf-8").splitlines()
        if line.strip() and "label" not in json.loads(line)["verdict"]
    ]


@pytest.fixture(scope="module")
def judged_nda(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """An out-dir with every non-equivalence verdict applied, mined and judged,
    so its pending queue holds exactly the equivalence questions."""
    base = tmp_path_factory.mktemp("equivalence")
    out_dir = base / "out"
    out_dir.mkdir()
    canned = base / "canned-without-equivalence.jsonl"
    canned.write_text("\n".join(_non_equivalence_canned()) + "\n", encoding="utf-8")
    assert _invoke(["judge-apply", str(out_dir), "--verdicts", str(canned)]).exit_code == 0
    mine = _invoke(
        ["mine", str(_CORPUS_DIR), "--config", str(_SMOKE_CONFIG), "--out", str(out_dir)]
    )
    assert mine.exit_code == 0, mine.output
    judge = _invoke(
        ["judge", str(_CORPUS_DIR), "--config", str(_SMOKE_CONFIG), "--out", str(out_dir)]
    )
    assert judge.exit_code == 0, judge.output
    return out_dir


def _pending(out_dir: Path) -> list[dict[str, Any]]:
    path = out_dir / "judge" / "pending.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _copy(src: Path, dest: Path) -> Path:
    shutil.copytree(src, dest)
    return dest


def _draft(out_dir: Path, items: list[dict[str, Any]], *, label: str = "less_protective") -> Path:
    path = out_dir / "drafts.jsonl"
    lines = [
        json.dumps(
            {
                "key": item["key"],
                "verdict": {"label": label, "reason": f"DRAFT-{i}", "basis": "agent"},
            }
        )
        for i, item in enumerate(items)
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    result = _invoke(["judge-apply", str(out_dir), "--verdicts", str(path)])
    assert result.exit_code == 0, result.output
    return path


def test_judge_queues_one_equivalence_item_per_distinct_text(judged_nda: Path) -> None:
    items = [i for i in _pending(judged_nda) if i["kind"] == "equivalence"]
    assert items, "template mode queued no equivalence question"
    assert len(_pending(judged_nda)) == len(items)  # everything else was canned
    assert len({i["key"] for i in items}) == len(items)  # deduplicated by key
    expected_keys = {
        "stage",
        "agreement_type_id",
        "taxonomy_id",
        "taxonomy_title",
        "perspective_party",
        "our_standard",
        "candidate",
        "roles",
    }
    for item in items:
        assert set(item["payload"]) == expected_keys
        assert item["rubric_version"] == rubric_version("equivalence")
        assert item["payload"]["candidate"] != item["payload"]["our_standard"]
    # The example's document ids never reach a judge. (Its candidate texts can
    # still carry party names; this checks the document-id slugs only.)
    blob = json.dumps(items)
    for deal in ("beta-industries", "gamma-holdings", "theta-logistics", "zeta-diagnostics"):
        assert deal not in blob


def test_judge_plan_only_reports_equivalence_pending(tmp_path: Path) -> None:
    out_dir = tmp_path / "plan"
    out_dir.mkdir()
    result = _invoke(
        [
            "judge",
            str(_CORPUS_DIR),
            "--config",
            str(_SMOKE_CONFIG),
            "--out",
            str(out_dir),
            "--plan-only",
        ]
    )
    assert result.exit_code == 0, result.output
    import re

    assert re.search(r"equivalence: [1-9]", result.output), result.output


def test_project_without_a_store_leaves_every_label_null_and_reports_the_count(
    judged_nda: Path, tmp_path: Path
) -> None:
    out_dir = tmp_path / "no-store"
    out_dir.mkdir()
    for name in ("observations.jsonl", "corpus_manifest.json", "template_observations.jsonl"):
        shutil.copy(judged_nda / name, out_dir / name)
    for optional in ("round_moves.jsonl", "scope.json"):
        if (judged_nda / optional).exists():
            shutil.copy(judged_nda / optional, out_dir / optional)
    result = _invoke(["project", str(out_dir), "--config", str(_SMOKE_CONFIG)])
    assert result.exit_code == 0, result.output
    playbook = json.loads((out_dir / "playbook.opf.json").read_text(encoding="utf-8"))
    assert all(slot.entry["vs_standard"] is None for slot in iter_slots(playbook["evidence"]))
    n_pending = len([i for i in _pending(judged_nda) if i["kind"] == "equivalence"])
    assert f"{n_pending} unjudged" in result.output
    assert "playbook judge" in result.output
    assert _invoke(["validate", str(out_dir / "playbook.opf.json")]).exit_code == 0


def test_draft_check_adjudicate_project_end_to_end(judged_nda: Path, tmp_path: Path) -> None:
    out_dir = _copy(judged_nda, tmp_path / "e2e")
    items = [i for i in _pending(out_dir) if i["kind"] == "equivalence"]
    _draft(out_dir, items)

    # Drafted, not yet checked: the label still reaches the playbook, check null.
    assert _invoke(["project", str(out_dir), "--config", str(_SMOKE_CONFIG)]).exit_code == 0
    playbook = json.loads((out_dir / "playbook.opf.json").read_text(encoding="utf-8"))
    labelled = [
        s.entry["vs_standard"] for s in iter_slots(playbook["evidence"]) if s.entry["vs_standard"]
    ]
    assert labelled and all(v["check"] is None and v["basis"] == "agent" for v in labelled)
    totals = summarize(playbook["evidence"], "nda", "AlphaCorp Holdings, Inc.")["totals"]
    assert totals["unjudged"] == 0 and totals["unchecked"] == totals["drafted"] == len(items)
    assert _invoke(["validate", str(out_dir / "playbook.opf.json")]).exit_code == 0

    # The blind check queue: no draft, one item per drafted text.
    result = _invoke(
        ["judge", "--check", "equivalence", str(out_dir), "--config", str(_SMOKE_CONFIG)]
    )
    assert result.exit_code == 0, result.output
    check_path = out_dir / "judge" / "check-pending.jsonl"
    check_items = [json.loads(line) for line in check_path.read_text().splitlines()]
    assert len(check_items) == len(items)
    assert (
        "DRAFT-" not in check_path.read_text() and "less_protective" not in check_path.read_text()
    )

    # Answer: agree on the first, disagree on the second.
    answers = out_dir / "answers.jsonl"
    answers.write_text(
        "\n".join(
            json.dumps(
                {
                    "key": item["key"],
                    "label": "less_protective" if n == 0 else "equivalent",
                    "reason": f"CHECK-{n}",
                    "model": CHECKER_MODEL,
                    "effort": CHECKER_EFFORT,
                }
            )
            for n, item in enumerate(check_items[:2])
        )
        + "\n",
        encoding="utf-8",
    )
    wrong = _invoke(["judge-apply", str(out_dir), "--check", str(answers)])
    assert wrong.exit_code == 0, wrong.output
    assert "agreed 1, disagreed 1" in wrong.output

    # The disagreement is queued for adjudication; the draft still flows.
    assert (
        _invoke(
            ["judge", "--check", "equivalence", str(out_dir), "--config", str(_SMOKE_CONFIG)]
        ).exit_code
        == 0
    )
    adjudication = [
        json.loads(line)
        for line in (out_dir / "judge" / "adjudication-pending.jsonl").read_text().splitlines()
    ]
    assert len(adjudication) == 1
    assert adjudication[0]["payload"]["check"]["reason"] == "CHECK-1"
    adjudicated = out_dir / "adjudication.jsonl"
    adjudicated.write_text(
        json.dumps(
            {
                "key": adjudication[0]["key"],
                "label": "more_protective",
                "reason": "ADJUDICATED",
                "model": CHECKER_MODEL,
                "effort": CHECKER_EFFORT,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    assert _invoke(["judge-apply", str(out_dir), "--check", str(adjudicated)]).exit_code == 0

    assert _invoke(["project", str(out_dir), "--config", str(_SMOKE_CONFIG)]).exit_code == 0
    playbook = json.loads((out_dir / "playbook.opf.json").read_text(encoding="utf-8"))
    by_reason = {
        s.entry["vs_standard"]["reason"]: s.entry["vs_standard"]
        for s in iter_slots(playbook["evidence"])
        if s.entry["vs_standard"]
    }
    assert by_reason["DRAFT-0"]["check"] == {
        "agreed": True,
        "by": CHECKER_MODEL,
        "adjudicated": False,
    }
    assert by_reason["ADJUDICATED"]["label"] == "more_protective"
    assert by_reason["ADJUDICATED"]["check"] == {
        "agreed": False,
        "by": CHECKER_MODEL,
        "adjudicated": True,
    }
    assert _invoke(["validate", str(out_dir / "playbook.opf.json")]).exit_code == 0

    # Scorecard (counts only): drafted / checked / agreed / adjudicated / unchecked by role.
    card = build_scorecard(out_dir)["equivalence"]
    totals = card["totals"]
    assert (totals["checked"], totals["agreed"], totals["adjudicated"]) == (2, 1, 1)
    assert totals["unchecked"] == len(items) - 2
    assert card["agreement_rate"] == 0.5
    assert card["checker_models"] == {CHECKER_MODEL: 2}
    assert set(card["by_role"]) == {"signed", "opening", "refused"}
    flat = json.dumps(card)
    assert "ADJUDICATED" not in flat and "DRAFT-" not in flat


def test_judge_apply_check_rejects_the_wrong_checker_unless_overridden(
    judged_nda: Path, tmp_path: Path
) -> None:
    out_dir = _copy(judged_nda, tmp_path / "pin")
    items = [i for i in _pending(out_dir) if i["kind"] == "equivalence"]
    _draft(out_dir, items[:1])
    record = {
        "key": items[0]["key"],
        "label": "less_protective",
        "reason": "Checked.",
        "model": "claude-haiku-5-5",
        "effort": "xhigh",
    }
    answers = out_dir / "answers.jsonl"
    answers.write_text(json.dumps(record) + "\n", encoding="utf-8")
    rejected = _invoke(["judge-apply", str(out_dir), "--check", str(answers)])
    assert rejected.exit_code == 1
    assert "claude-opus-5-5" in rejected.output
    assert (
        VerdictStore(out_dir / "judge" / "verdicts.jsonl").get_by_key(items[0]["key"])["label"]
        == "less_protective"
    )
    assert "check" not in VerdictStore(out_dir / "judge" / "verdicts.jsonl").get_by_key(
        items[0]["key"]
    )

    record["model"], record["effort"] = CHECKER_MODEL, "medium"
    answers.write_text(json.dumps(record) + "\n", encoding="utf-8")
    assert _invoke(["judge-apply", str(out_dir), "--check", str(answers)]).exit_code == 1

    overridden = _invoke(
        ["judge-apply", str(out_dir), "--check", str(answers), "--allow-checker-model"]
    )
    assert overridden.exit_code == 0, overridden.output


def test_judge_apply_needs_exactly_one_of_verdicts_or_check(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    assert _invoke(["judge-apply", str(out_dir)]).exit_code == 1
    both = tmp_path / "x.jsonl"
    both.write_text("", encoding="utf-8")
    assert (
        _invoke(
            ["judge-apply", str(out_dir), "--verdicts", str(both), "--check", str(both)]
        ).exit_code
        == 1
    )


def test_every_precedent_text_entry_carries_a_vs_standard_key(
    judged_nda: Path, tmp_path: Path
) -> None:
    """The producer of the field: any projected playbook has the key on every
    signed_text, opening_text and refused ask (null when not judged)."""
    out_dir = _copy(judged_nda, tmp_path / "shape")
    assert _invoke(["project", str(out_dir), "--config", str(_SMOKE_CONFIG)]).exit_code == 0
    playbook = json.loads((out_dir / "playbook.opf.json").read_text(encoding="utf-8"))
    slots = list(iter_slots(playbook["evidence"]))
    assert slots and all("vs_standard" in slot.entry for slot in slots)
    # The sidecar records carry them too (it is the same record).
    sidecar = [
        json.loads(line)
        for line in (out_dir / "precedent.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert all("vs_standard" in (p["signed_text"] or {"vs_standard": None}) for p in sidecar)


def test_label_evidence_only_writes_labels_for_eligible_texts() -> None:
    playbook = _two_paper_playbook()
    evidence = json.loads(json.dumps(playbook["evidence"]))
    (subject,) = collect_subjects(evidence, "test-agreement", _PARTY)
    vs = {"label": "equivalent", "reason": "x.", "basis": "agent", "check": None}
    label_evidence(evidence, "test-agreement", _PARTY, {subject.key: vs})
    got = {p["document_id"]: p["signed_text"]["vs_standard"] for p in evidence["precedent"]}
    assert got == {"deal_a": None, "deal_b": vs, "deal_c": vs}
    assert dataclasses.is_dataclass(subject)


def test_playbook_validate_finds_the_verdict_store_beside_the_playbook(
    judged_nda: Path, tmp_path: Path
) -> None:
    """A label whose verdict is not in the run's store (judge/verdicts.jsonl next
    to the playbook) fails `playbook validate`; with the store it passes."""
    out_dir = _copy(judged_nda, tmp_path / "validate")
    items = [i for i in _pending(out_dir) if i["kind"] == "equivalence"]
    _draft(out_dir, items[:3])
    assert _invoke(["project", str(out_dir), "--config", str(_SMOKE_CONFIG)]).exit_code == 0
    playbook_path = out_dir / "playbook.opf.json"
    assert _invoke(["validate", str(playbook_path)]).exit_code == 0

    store_path = out_dir / "judge" / "verdicts.jsonl"
    kept = [
        line
        for line in store_path.read_text(encoding="utf-8").splitlines()
        if items[0]["key"] not in line
    ]
    store_path.write_text("\n".join(kept) + "\n", encoding="utf-8")
    result = _invoke(["validate", str(playbook_path)])
    assert result.exit_code == 1
    assert "verdict store has no equivalence verdict" in result.output
