"""When each deal was signed (OPF 0.6, issue #247).

Layers, from the pure readers up to the real producer:

- the four deterministic readers (document text, signature block, file name,
  file metadata) and what they refuse to read (two readings, an impossible
  date, a year outside 1980-2099);
- the evidence mining records and the judge question it derives
  (:func:`playbook_engine.deal_dates.extract_evidence`);
- the ``deal_date`` judge kind: grounding of its answer, the store-backed
  judge, ``judge-apply``;
- resolution in the order of trust, and the digest's months;
- the validator's signing-date MUSTs, on the REAL compiled NDA example;
- the engine end to end: ``mine`` -> ``judge`` -> ``judge-apply`` ->
  ``project`` over a copy of the synthetic NDA corpus, whose executed copies
  carry dates in their signature blocks (production writes every fixture
  here: ``deal_dates.extract_evidence`` -> the corpus manifest ->
  ``resolve_corpus_documents`` -> the assembler).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from playbook_engine import deal_dates as dd
from playbook_engine.agent_judge import (
    PendingQueue,
    StoreBackedDealDateJudge,
    VerdictStore,
    _payload_key,
    infer_verdict_kind,
    validate_verdict,
)
from playbook_engine.canonicalize import compute_section_digests, content_hash
from playbook_engine.clause_tree import ClauseNode, ClauseTree
from playbook_engine.cli import cli
from playbook_engine.digest import build_digest_v5
from playbook_engine.dossiers import build_derived_sections
from playbook_engine.rubric import RubricPolicy, RubricStamp, rubric_version
from playbook_engine.signed_detector import signing_texts
from playbook_engine.validator import validate_document

_ROOT = Path(__file__).resolve().parent.parent
_NDA_DIR = _ROOT / "examples" / "nda"
_NDA_PLAYBOOK = _NDA_DIR / "playbook.opf.json"

# ---------------------------------------------------------------------------
# The readers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # Cued dates in the preamble, in the shapes contracts use.
        (
            "This Agreement is made and entered into as of March 3, 2021 by and between",
            "2021-03-03",
        ),
        ("Dated as of 3 March 2021", "2021-03-03"),
        ("THIS AGREEMENT, dated MARCH 3rd, 2021, is made", "2021-03-03"),
        ("Effective Date: 2021-03-03", "2021-03-03"),
        ("made this 3rd day of March, 2021, between", "2021-03-03"),
        ("dated as of Sept. 9, 2021", "2021-09-09"),
        # A numeric date that cannot read two ways.
        ("dated as of 25/03/2021", "2021-03-25"),
        ("dated as of 03/25/2021", "2021-03-25"),
        ("dated as of 03/03/2021", "2021-03-03"),
        # Month precision: the document itself leaves the day blank.
        ("dated as of March ___, 2021", "2021-03"),
        ("made this ____ day of June, 2019, by", "2019-06"),
        ("effective March 2021", "2021-03"),
        (
            'This Agreement is made effective this June 1, 2022 (the "Effective Date") by '
            "and between the parties.",
            "2022-06-01",
        ),
        # A month-only and a full date of the same month agree; the day wins.
        ("dated as of March 2021 (the Effective Date March 3, 2021)", "2021-03-03"),
        # Ambiguous or unreadable input says nothing: null, never a guess.
        ("dated as of 03/04/2021", None),
        ("dated as of February 30, 2021", None),
        ("dated as of 2021-13-02", None),
        ("dated as of March 3, 1975", None),
        ("dated as of March 3, 2150", None),
        ("dated as of March 3, 2021, amending the agreement dated June 1, 2019", None),
        # Not the document's own date: no cue, or a number of days.
        ("The Term continues for 30 days after March 1, 2021.", None),
        ("within 30 days of March 1, 2021", None),
        # A defined term mentioned earlier, or an end or limit phrase, does not
        # cue the date after it: a term's end is not the document's own date.
        (
            '... (the "Effective Date") ... 1. Term. This Agreement shall begin on the '
            "Effective Date and end on June 30, 2026.",
            None,
        ),
        ("This Agreement shall be effective until December 31, 2025.", None),
        (
            '...(the "Effective Date") by and between the parties. Students placed on or '
            "after January 1, 2024 are covered.",
            None,
        ),
        ("Expiration Date: June 30, 2026", None),
        # A modal "may" is not a month.
        ("The parties may 2021 amend", None),
        ("", None),
    ],
)
def test_document_text_reader(text: str, expected: str | None) -> None:
    assert dd.date_from_document_text(text) == expected


def test_document_text_reader_only_reads_the_opening() -> None:
    body = "x" * (dd.PREAMBLE_CHARS + 50) + " dated as of March 3, 2021"
    assert dd.date_from_document_text(body) is None


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("By: /s/ A\nTitle: GC\nDate: March 3, 2021", "2021-03-03"),
        # Signatories dated their signatures apart: the deal was signed on the last.
        ("Date: March 3, 2021\nDate: March 9, 2021", "2021-03-09"),
        ("Date: 2021-03-09\nDate: 2021-03", "2021-03-09"),
        ("Date: 03/04/2021", None),
        ("Date: 03/04/2021\nDate: March 9, 2021", None),
        ("Date: February 30, 2021", None),
        ("By: /s/ A\nTitle: GC", None),
        ("", None),
    ],
)
def test_signature_block_reader(text: str, expected: str | None) -> None:
    assert dd.date_from_signature_block(text) == expected


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Acme NDA 2021-03-03 signed.pdf", "2021-03-03"),
        ("01__Acme NDA 2021-03-03.docx", "2021-03-03"),
        ("NDA_20210303_final.docx", "2021-03-03"),
        ("NDA 2021.03.03.pdf", "2021-03-03"),
        ("NDA March 3, 2021.pdf", "2021-03-03"),
        ("NDA 3 March 2021.pdf", "2021-03-03"),
        ("NDA March 2021.pdf", "2021-03"),
        ("NDA 2021-03.pdf", "2021-03"),
        ("NDA 25-03-2021.pdf", "2021-03-25"),
        # Ambiguous or several: nothing.
        ("NDA 03-04-2021.pdf", None),
        ("NDA 2021-03-03 2021-04-05.pdf", None),
        ("NDA 2021-02-30.pdf", None),
        ("Agreement 2019-45.pdf", None),
        ("NDA v2 final.pdf", None),
        ("NDA 2021.pdf", None),
        ("", None),
    ],
)
def test_filename_reader(name: str, expected: str | None) -> None:
    assert dd.date_from_filename(name) == expected


@pytest.mark.parametrize(
    ("timestamp", "expected"),
    [
        ("2021-03-03T10:20:30Z", "2021-03-03"),
        ("2021-03-03", "2021-03-03"),
        ("1970-01-01T00:00:00Z", None),  # before 1980: a format default
        ("2021-13-03T00:00:00Z", None),
        ("not a date", None),
        (None, None),
    ],
)
def test_file_metadata_reader(timestamp: str | None, expected: str | None) -> None:
    assert dd.date_from_file_metadata(timestamp) == expected


@pytest.mark.parametrize(
    "value",
    ["2021-03-03", "2021-03", "1980-01", "2099-12-31"],
)
def test_signed_on_values_that_are_well_formed(value: str) -> None:
    assert dd.is_signed_on(value)


@pytest.mark.parametrize(
    "value",
    ["2021", "2021-3-3", "2021-02-30", "2021-13", "2021-00", "2021-03-00", "1979-12", "2100-01",
     "2021-Q1", "2021-03-03T00:00:00Z", None, 20210303, ["2021-03"]],
)  # fmt: skip
def test_signed_on_values_that_are_not(value: Any) -> None:
    assert not dd.is_signed_on(value)


def test_a_month_sorts_before_every_day_in_it() -> None:
    assert (
        dd.signed_on_key("2021-03") < dd.signed_on_key("2021-03-01") < dd.signed_on_key("2021-04")
    )
    assert dd.signed_on_key("not a date") is None


# ---------------------------------------------------------------------------
# Evidence and the judge question
# ---------------------------------------------------------------------------

_PREAMBLE = "MUTUAL NDA. This Agreement is made as of March 3, 2021 between the parties. "
_BLOCK_WITH_NAMES = (
    "IN WITNESS WHEREOF the parties have executed this Agreement.\n"
    "AlphaCorp Holdings, Inc.\nBy: /s/ Dana Reyes\nName: Dana Reyes\nTitle: Vice President\n"
    "Date: 03/04/2023\nBeta Industries, LLC\nBy: /s/ Morgan Ellery\nDate: 03/04/2023"
)


def test_evidence_records_the_four_readings() -> None:
    evidence = dd.extract_evidence(
        document_text=_PREAMBLE + "Body.",
        signature_block_text="By: /s/ A\nDate: March 9, 2021",
        filename="01__NDA 2021-04-01.pdf",
        file_timestamp="2021-05-05T10:00:00Z",
    )
    assert evidence.to_dict() == {
        "document_text": "2021-03-03",
        "signature_block": "2021-03-09",
        "filename": "2021-04-01",
        "file_metadata": "2021-05-05",
    }
    assert evidence.passages is None  # the text spoke: nothing to ask a judge


def test_evidence_without_any_source_is_empty() -> None:
    evidence = dd.extract_evidence(
        document_text="No dates here.", signature_block_text="", filename=None, file_timestamp=None
    )
    assert evidence == dd.DealDateEvidence()
    assert dd.subject_of(evidence) is None


def test_an_ambiguous_signature_date_becomes_a_judge_question_without_names() -> None:
    evidence = dd.extract_evidence(
        document_text="No dates here.",
        signature_block_text=_BLOCK_WITH_NAMES,
        filename=None,
        file_timestamp=None,
    )
    assert evidence.signature_block is None
    subject = dd.subject_of(evidence)
    assert subject is not None
    assert subject.payload == {
        "stage": "deal_date",
        "document_text": [],
        # Only the lines holding a date: no party or signatory name reaches the judge.
        "signature_block": ["Date: 03/04/2023"],
    }


def test_conflicting_preamble_dates_are_a_judge_question_with_context() -> None:
    text = "Dated as of March 3, 2021, amending the agreement dated June 1, 2019. " + "x" * 5000
    evidence = dd.extract_evidence(
        document_text=text, signature_block_text="", filename=None, file_timestamp=None
    )
    assert evidence.document_text is None
    subject = dd.subject_of(evidence)
    assert subject is not None
    (passage,) = subject.payload["document_text"]
    assert "March 3, 2021" in passage and "June 1, 2019" in passage
    assert subject.payload["signature_block"] == []


def _executed_copy(*nodes: ClauseNode) -> ClauseTree:
    return ClauseTree(document_id="d", version="v1", source_file="d.docx", nodes=list(nodes))


def _node(path: str, heading: str | None, text: str, *kids: ClauseNode) -> ClauseNode:
    return ClauseNode(
        clause_path=path,
        heading=heading,
        text=text,
        char_span=(0, len(text) or 1),
        children=list(kids),
    )


_SHORT_BLOCK = (
    "IN WITNESS WHEREOF, the parties have executed this Agreement.\n"
    "By: /s/ Dana Reyes\nName: Dana Reyes\nDate: March 3, 2021"
)


def test_each_date_reader_sees_only_its_own_region_of_a_short_executed_copy() -> None:
    """A signature block inside the first PREAMBLE_CHARS is not "the document's own words"."""
    tree = _executed_copy(
        _node("1", "Term", "The Term is two years.\n\n" + _SHORT_BLOCK),
    )
    document_text, block_text = signing_texts(tree)
    assert len(document_text) + len(block_text) < dd.PREAMBLE_CHARS
    assert "Dana" not in document_text and "March 3" not in document_text
    assert "Dana Reyes" in block_text and "March 3, 2021" in block_text

    evidence = dd.extract_evidence(
        document_text=document_text,
        signature_block_text=block_text,
        filename=None,
        file_timestamp=None,
    )
    assert (evidence.document_text, evidence.signature_block) == (None, "2021-03-03")
    assert dd.resolve_signed_on(evidence) == ("2021-03-03", "signature_block")


def test_the_clause_body_before_an_absorbed_trailer_is_not_signature_block_text() -> None:
    """A clause that absorbed "IN WITNESS WHEREOF" gives the block only its text from the trailer on."""
    tree = _executed_copy(
        _node(
            "1",
            "Term",
            "The term of the Agreement is hereby extended through January 6, 2030.\n\n"
            + _SHORT_BLOCK
            + "\nBy: /s/ Morgan Ellery\nDate: March 5, 2021",
        ),
    )
    document_text, block_text = signing_texts(tree)
    assert "January 6, 2030" not in block_text and "extended" not in block_text
    assert "January 6, 2030" in document_text
    assert "March 3, 2021" in block_text and "March 5, 2021" in block_text

    evidence = dd.extract_evidence(
        document_text=document_text,
        signature_block_text=block_text,
        filename=None,
        file_timestamp=None,
    )
    assert (evidence.document_text, evidence.signature_block) == (None, "2021-03-05")
    assert dd.resolve_signed_on(evidence) == ("2021-03-05", "signature_block")


def test_a_preamble_date_is_not_made_ambiguous_by_the_signature_block() -> None:
    tree = _executed_copy(
        _node("0", None, "This Agreement is made as of March 1, 2021 between the parties."),
        _node("1", "Term", "The Term is two years.\n\n" + _SHORT_BLOCK),
    )
    evidence = dd.extract_evidence(
        document_text=signing_texts(tree)[0],
        signature_block_text=signing_texts(tree)[1],
        filename=None,
        file_timestamp=None,
    )
    assert (evidence.document_text, evidence.signature_block) == ("2021-03-01", "2021-03-03")
    assert dd.resolve_signed_on(evidence) == ("2021-03-01", "document_text")


def test_a_signature_headed_section_is_not_document_text_and_judge_passages_carry_no_names() -> (
    None
):
    tree = _executed_copy(
        _node("1", "Term", "The Term is two years."),
        _node(
            "2",
            "Signature Page",
            "",
            _node("2.1", None, "By: /s/ Dana Reyes\nName: Dana Reyes\nDate: 03/04/2023"),
        ),
    )
    document_text, block_text = signing_texts(tree)
    assert "Dana" not in document_text and "03/04/2023" not in document_text
    assert "Dana Reyes" in block_text
    evidence = dd.extract_evidence(
        document_text=document_text,
        signature_block_text=block_text,
        filename=None,
        file_timestamp=None,
    )
    subject = dd.subject_of(evidence)
    assert subject is not None
    assert subject.payload["document_text"] == []
    assert "Dana" not in json.dumps(subject.payload)


def test_evidence_round_trips_and_a_malformed_dict_reads_as_no_evidence() -> None:
    evidence = dd.extract_evidence(
        document_text="x",
        signature_block_text="Date: 03/04/2023",
        filename="NDA 2021-03.pdf",
        file_timestamp="2021-05-05T00:00:00Z",
    )
    assert dd.DealDateEvidence.from_dict(evidence.to_dict()) == evidence
    for bad in (None, "x", [], {"document_text": "2021-02-30", "filename": 5, "passages": 3}):
        assert dd.DealDateEvidence.from_dict(bad) == dd.DealDateEvidence()


# ---------------------------------------------------------------------------
# The deal_date verdict
# ---------------------------------------------------------------------------

_PAYLOAD = {
    "stage": "deal_date",
    "document_text": ["Dated the third of March in the year 2021, between the parties"],
    "signature_block": ["Date: 4 / 3 / 2023"],
}


def test_a_grounded_verdict_is_accepted() -> None:
    answer = dd.validate_verdict(
        {
            "signed_on": "2021-03-03",
            "basis": "document_text",
            "quote": "Dated the third of March in the year 2021",
        },
        _PAYLOAD,
    )
    assert answer == dd.DealDate(
        "2021-03-03", "document_text", "Dated the third of March in the year 2021"
    )


def test_a_null_verdict_is_a_judged_no() -> None:
    assert dd.validate_verdict(
        {"signed_on": None, "basis": None, "quote": None}, _PAYLOAD
    ) == dd.DealDate(None, None, None)


@pytest.mark.parametrize(
    ("verdict", "message"),
    [
        ({"signed_on": "2021-03-03"}, "basis"),
        ({"signed_on": "2021-03-03", "basis": "filename", "quote": "x"}, "basis"),
        ({"signed_on": "2021-03-03", "basis": "document_text"}, "quote"),
        ({"signed_on": "2021-03-03", "basis": "document_text", "quote": "  "}, "quote"),
        ({"signed_on": "2021-02-30", "basis": "document_text", "quote": "x"}, "signed_on"),
        ({"signed_on": "March 3, 2021", "basis": "document_text", "quote": "x"}, "signed_on"),
        ({"signed_on": "2021-Q1", "basis": "document_text", "quote": "x"}, "signed_on"),
        ({"signed_on": None, "basis": "document_text", "quote": None}, "null"),
        ({"signed_on": None, "basis": None, "quote": "x"}, "null"),
        # Grounding: the quote must occur in the named source's passages.
        (
            {"signed_on": "2021-03-03", "basis": "document_text", "quote": "third of April"},
            "does not occur",
        ),
        (
            {"signed_on": "2023-03-04", "basis": "signature_block", "quote": "Dated the third"},
            "does not occur",
        ),
    ],
)
def test_ungrounded_or_malformed_verdicts_are_rejected(
    verdict: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        dd.validate_verdict(verdict, _PAYLOAD)


def test_quote_matching_ignores_whitespace() -> None:
    """The payload is squashed (``_date_passages`` squashes it); the quote is the judge's own."""
    answer = dd.validate_verdict(
        {
            "signed_on": "2023-04-03",
            "basis": "signature_block",
            "quote": "Date:  4 /\n3 /  2023",
        },
        _PAYLOAD,
    )
    assert answer.signed_on == "2023-04-03"
    assert answer.quote == "Date: 4 / 3 / 2023"  # recorded squashed


def test_a_payload_built_by_mining_is_already_squashed() -> None:
    evidence = dd.extract_evidence(
        document_text="No dates here.",
        signature_block_text="By: /s/ A\nDate:   4  /\t3 / 2023",
        filename=None,
        file_timestamp=None,
    )
    subject = dd.subject_of(evidence)
    assert subject is not None
    assert subject.payload["signature_block"] == ["Date: 4 / 3 / 2023"]


def test_the_kind_is_registered_with_judge_apply() -> None:
    verdict = {"signed_on": "2021-03", "basis": "document_text", "quote": "March 2021"}
    assert infer_verdict_kind(verdict) == "deal_date"
    validate_verdict("deal_date", verdict)
    with pytest.raises(ValueError, match="signed_on"):
        validate_verdict("deal_date", {**verdict, "signed_on": "2021-02-30"})


# ---------------------------------------------------------------------------
# The store-backed judge
# ---------------------------------------------------------------------------


def _subject() -> dd.DealDateSubject:
    return dd.DealDateSubject(payload=dict(_PAYLOAD))


def test_a_miss_is_queued_and_never_guessed(tmp_path: Path) -> None:
    store = VerdictStore(tmp_path / "judge" / "verdicts.jsonl")
    queue = PendingQueue(tmp_path / "judge" / "pending.jsonl")
    judge = StoreBackedDealDateJudge(store=store, pending=queue)
    assert judge.judge(_subject()) is None
    assert judge.judge(_subject()) is None  # deduplicated by key
    (line,) = (tmp_path / "judge" / "pending.jsonl").read_text().splitlines()
    record = json.loads(line)
    assert record["kind"] == "deal_date"
    assert record["key"] == _payload_key(_PAYLOAD)
    assert record["payload"] == _PAYLOAD
    assert record["rubric_version"] == rubric_version("deal_date")


def test_a_hit_replays_and_a_read_only_judge_writes_nothing(tmp_path: Path) -> None:
    store = VerdictStore(tmp_path / "verdicts.jsonl")
    stamp = RubricStamp(kind="deal_date", version=rubric_version("deal_date"))
    store.put_by_key(
        _payload_key(_PAYLOAD),
        {"signed_on": "2021-03-03", "basis": "document_text", "quote": "Dated the third of March"},
        rubric=stamp,
    )
    judge = StoreBackedDealDateJudge(store=store, pending=None)
    assert judge.judge(_subject()) == dd.DealDate(
        "2021-03-03", "document_text", "Dated the third of March"
    )
    other = dd.DealDateSubject(payload={**_PAYLOAD, "signature_block": ["Date: 04/05/2023"]})
    assert judge.judge(other) is None  # unjudged, and no queue to write to
    assert not (tmp_path / "pending.jsonl").exists()


def test_a_stale_rubric_requeues_and_a_malformed_record_is_not_replayed(tmp_path: Path) -> None:
    store = VerdictStore(tmp_path / "verdicts.jsonl")
    queue = PendingQueue(tmp_path / "pending.jsonl")
    good = {
        "signed_on": "2021-03-03",
        "basis": "document_text",
        "quote": "Dated the third of March",
    }
    store.put_by_key(
        _payload_key(_PAYLOAD), good, rubric=RubricStamp(kind="deal_date", version="v0+stale")
    )
    judge = StoreBackedDealDateJudge(store=store, pending=queue)
    assert judge.judge(_subject()) is None
    assert (tmp_path / "pending.jsonl").read_text().strip()  # re-queued

    # A hand-edited store row whose quote does not occur in the passages.
    bad_store = VerdictStore(tmp_path / "bad.jsonl")
    bad_store.put_by_key(
        _payload_key(_PAYLOAD),
        {**good, "quote": "never shown to the judge"},
        rubric=RubricStamp(kind="deal_date", version=rubric_version("deal_date")),
    )
    assert (
        StoreBackedDealDateJudge(store=bad_store, rubric=RubricPolicy()).judge(_subject()) is None
    )


# ---------------------------------------------------------------------------
# Resolution: the order of trust
# ---------------------------------------------------------------------------


class _FixedJudge:
    """Answers like the store-backed judge: only a verdict grounded in the question's payload."""

    def __init__(self, verdict: dict[str, Any] | None) -> None:
        self.verdict = verdict
        self.asked: list[dd.DealDateSubject] = []

    def judge(self, subject: dd.DealDateSubject) -> dd.DealDate | None:
        self.asked.append(subject)
        if self.verdict is None:
            return None
        return dd.verdict_of(self.verdict, subject.payload)


def _evidence(**kw: Any) -> dd.DealDateEvidence:
    return dd.DealDateEvidence(**kw)


def test_the_order_of_trust() -> None:
    all_four = _evidence(
        document_text="2020-01-01",
        signature_block="2020-02-02",
        filename="2020-03-03",
        file_metadata="2020-04-04",
    )
    assert dd.resolve_signed_on(all_four) == ("2020-01-01", "document_text")
    assert dd.resolve_signed_on(_evidence(**{**all_four.__dict__, "document_text": None})) == (
        "2020-02-02",
        "signature_block",
    )
    assert dd.resolve_signed_on(_evidence(filename="2020-03-03", file_metadata="2020-04-04")) == (
        "2020-03-03",
        "filename",
    )
    assert dd.resolve_signed_on(_evidence(file_metadata="2020-04-04")) == (
        "2020-04-04",
        "file_metadata",
    )
    assert dd.resolve_signed_on(_evidence()) == (None, None)


def test_the_judge_is_read_between_the_text_and_the_file_name() -> None:
    evidence = dd.extract_evidence(
        document_text="No dates here.",
        signature_block_text="By: /s/ A\nDate: 03/04/2023",
        filename="2020-03-03.pdf",
        file_timestamp="2020-04-04T00:00:00Z",
    )
    assert evidence.passages == {"document_text": [], "signature_block": ["Date: 03/04/2023"]}
    grounded = {"signed_on": "2023-03-04", "basis": "signature_block", "quote": "Date: 03/04/2023"}
    judge = _FixedJudge(grounded)
    assert dd.resolve_signed_on(evidence, judge) == ("2023-03-04", "signature_block")
    assert len(judge.asked) == 1

    # A judged "no", no verdict yet, or a verdict whose quote the judge was never
    # shown (refused through grounding): the file name speaks.
    ungrounded = {**grounded, "quote": "Date: 04/03/2023"}
    for verdict in ({"signed_on": None, "basis": None, "quote": None}, None, ungrounded):
        assert dd.resolve_signed_on(evidence, _FixedJudge(verdict)) == ("2020-03-03", "filename")
    assert dd.resolve_signed_on(evidence, None) == ("2020-03-03", "filename")

    # Text that spoke leaves nothing to ask: mining records no passages, and the
    # judge is never consulted.
    spoke = dd.extract_evidence(
        document_text="Dated as of March 3, 2021, between the parties.",
        signature_block_text="By: /s/ A\nDate: 03/04/2023",
        filename=None,
        file_timestamp=None,
    )
    assert spoke.passages is None
    asked = _FixedJudge(grounded)
    assert dd.resolve_signed_on(spoke, asked) == ("2021-03-03", "document_text")
    assert asked.asked == []


def test_a_month_in_the_preamble_yields_to_a_day_in_that_month_from_the_block() -> None:
    evidence = dd.extract_evidence(
        document_text="This Agreement is made this ____ day of March, 2021, between the parties.",
        signature_block_text="By: /s/ A\nDate: March 15, 2021",
        filename="2021-03-01.pdf",
        file_timestamp=None,
    )
    assert (evidence.document_text, evidence.signature_block) == ("2021-03", "2021-03-15")
    assert dd.resolve_signed_on(evidence) == ("2021-03-15", "signature_block")

    # A different month's day does not refine it; the month stays, with its own basis.
    other = dd.extract_evidence(
        document_text="This Agreement is made this ____ day of March, 2021, between the parties.",
        signature_block_text="By: /s/ A\nDate: April 2, 2021",
        filename=None,
        file_timestamp=None,
    )
    assert dd.resolve_signed_on(other) == ("2021-03", "document_text")

    # A day in the preamble is never replaced.
    exact = dd.extract_evidence(
        document_text="Dated as of March 3, 2021, between the parties.",
        signature_block_text="By: /s/ A\nDate: March 15, 2021",
        filename=None,
        file_timestamp=None,
    )
    assert dd.resolve_signed_on(exact) == ("2021-03-03", "document_text")


def _doc(
    doc_id: str, *, signed_version: int | None, evidence: dd.DealDateEvidence | None, **kw: Any
) -> dict[str, Any]:
    doc: dict[str, Any] = {
        "document_id": doc_id,
        "provenance": "our_paper",
        "in_scope": True,
        "signed_version": signed_version,
        **kw,
    }
    if evidence is not None:
        doc["deal_date_evidence"] = evidence.to_dict()
    return doc


def test_only_a_deal_with_an_executed_copy_has_a_date() -> None:
    dated = _evidence(document_text="2021-03-03", filename="2021-04-04")
    docs = [
        _doc("signed", signed_version=3, evidence=dated),
        _doc("unsigned", signed_version=None, evidence=None),
        _doc("old-store", signed_version=3, evidence=None),
        _doc("out", signed_version=None, evidence=None, in_scope=False, scope_rationale="no"),
    ]
    resolved = {d["document_id"]: d for d in dd.resolve_corpus_documents(docs)}
    assert (resolved["signed"]["signed_on"], resolved["signed"]["signed_on_basis"]) == (
        "2021-03-03",
        "document_text",
    )
    for doc_id in ("unsigned", "old-store"):
        assert (resolved[doc_id]["signed_on"], resolved[doc_id]["signed_on_basis"]) == (None, None)
    assert "signed_on" not in resolved["out"]  # out-of-scope: no negotiation, no date
    assert "signed_on" not in docs[0]  # the input is not mutated


def test_coverage_counts_by_basis() -> None:
    docs = dd.resolve_corpus_documents(
        [
            _doc("a", signed_version=1, evidence=_evidence(document_text="2021-03-03")),
            _doc("b", signed_version=1, evidence=_evidence(filename="2021-04")),
            _doc("c", signed_version=1, evidence=_evidence()),
            _doc("d", signed_version=None, evidence=None),
        ]
    )
    cov = dd.coverage(docs)
    assert (cov.in_scope, cov.signed, cov.dated, cov.null_dates) == (4, 3, 2, 1)
    assert cov.by_basis == {"document_text": 1, "filename": 1}


def test_distinct_questions_are_asked_once() -> None:
    evidence = dd.extract_evidence(
        document_text="x",
        signature_block_text="Date: 03/04/2023",
        filename=None,
        file_timestamp=None,
    )
    docs = [
        _doc("a", signed_version=1, evidence=evidence),
        _doc("b", signed_version=1, evidence=evidence),
        _doc("c", signed_version=None, evidence=None),
        _doc("d", signed_version=1, evidence=_evidence()),
    ]
    assert len(dd.subjects_of(docs)) == 1


# ---------------------------------------------------------------------------
# The document: validator MUSTs on the real compiled example
# ---------------------------------------------------------------------------


def _nda() -> dict[str, Any]:
    doc: dict[str, Any] = json.loads(_NDA_PLAYBOOK.read_text(encoding="utf-8"))
    assert doc["opf_version"] == "0.6"
    return doc


def _refresh(doc: dict[str, Any]) -> dict[str, Any]:
    """Re-derive what depends on the dates, so a test isolates the rule it targets."""
    doc["digest"] = build_digest_v5(doc)
    doc.update(build_derived_sections(doc))
    doc["identity"]["content_hash"] = content_hash(doc)
    doc["identity"]["section_digests"] = compute_section_digests(doc)
    return doc


def _errors(doc: dict[str, Any]) -> list[str]:
    return [str(e) for e in validate_document(doc).errors if e.blocking]


def _document(doc: dict[str, Any], document_id: str) -> dict[str, Any]:
    return next(d for d in doc["corpus"]["documents"] if d["document_id"] == document_id)


def _set_date(
    doc: dict[str, Any], document_id: str, signed_on: str | None, basis: str | None
) -> None:
    """Set a document's date and every precedent record of the deal to match."""
    corpus_doc = _document(doc, document_id)
    corpus_doc["signed_on"], corpus_doc["signed_on_basis"] = signed_on, basis
    for record in doc["evidence"]["precedent"]:
        if record["document_id"] == document_id:
            record["signed_on"] = signed_on


def test_the_compiled_example_dates_its_deals_and_validates() -> None:
    doc = _nda()
    assert _errors(doc) == []
    dated = {
        d["document_id"]: (d["signed_on"], d["signed_on_basis"]) for d in doc["corpus"]["documents"]
    }
    assert dated == {
        "beta-industries": ("2022-03-16", "signature_block"),
        "delta-ventures": ("2022-06-02", "signature_block"),
        "epsilon-systems": ("2022-09-12", "signature_block"),
        "gamma-holdings": ("2022-11-30", "signature_block"),
        "theta-logistics": (None, None),
        "zeta-diagnostics": (None, None),
    }
    # Every precedent record carries its document's date.
    for record in doc["evidence"]["precedent"]:
        assert record["signed_on"] == dated[record["document_id"]][0]
    assert doc["digest"]["corpus"] == {
        "n_deals": 6,
        "n_signed": 6,
        "first_signed": "2022-03",
        "last_signed": "2022-11",
    }


@pytest.mark.parametrize(
    ("signed_on", "basis"),
    [
        ("2022-03", "filename"),
        ("2022-03-16", "document_text"),
        ("2022-03-16", "file_metadata"),
        (None, None),
    ],
)
def test_every_basis_and_both_precisions_validate(signed_on: str | None, basis: str | None) -> None:
    doc = _nda()
    _set_date(doc, "beta-industries", signed_on, basis)
    assert _errors(_refresh(doc)) == []


@pytest.mark.parametrize(
    ("signed_on", "basis", "message"),
    [
        ("2022-02-30", "filename", "not a real"),
        ("2022-13", "filename", "not a real"),
        ("2022-Q1", "filename", "not valid under any"),
        ("1979-12-01", "filename", "not a real"),
        ("2022-03-16", None, "signed_on_basis"),
        ("2022-03-16", "guess", "signed_on_basis"),
        (None, "filename", "signed_on is null"),
    ],
)
def test_a_malformed_document_date_is_rejected(
    signed_on: str | None, basis: str | None, message: str
) -> None:
    doc = _nda()
    _set_date(doc, "beta-industries", signed_on, basis)
    errors = _errors(_refresh(doc))
    assert any(message in e for e in errors), errors


def test_a_precedent_date_must_equal_its_documents() -> None:
    doc = _nda()
    record = next(r for r in doc["evidence"]["precedent"] if r["document_id"] == "beta-industries")
    record["signed_on"] = "2022-03-15"
    errors = _errors(_refresh(doc))
    assert any("precedent signed_on='2022-03-15'" in e for e in errors), errors

    doc = _nda()
    record = next(r for r in doc["evidence"]["precedent"] if r["document_id"] == "theta-logistics")
    record["signed_on"] = "2022-03-15"  # the document has none
    errors = _errors(_refresh(doc))
    assert any("precedent signed_on" in e for e in errors), errors


def test_a_precedent_record_must_carry_signed_on() -> None:
    doc = _nda()
    del doc["evidence"]["precedent"][0]["signed_on"]
    errors = _errors(_refresh(doc))
    assert any("signed_on" in e and "required" in e for e in errors), errors


@pytest.mark.parametrize(
    "missing", [["signed_on"], ["signed_on_basis"], ["signed_on", "signed_on_basis"]]
)
def test_an_in_scope_document_must_record_its_date_even_when_unknown(missing: list[str]) -> None:
    """null means "unknown"; an absent key means "never recorded", which 0.6 refuses."""
    doc = _nda()
    corpus_doc = _document(doc, "theta-logistics")  # in scope, signed, no date
    for key in missing:
        del corpus_doc[key]
    errors = _errors(_refresh(doc))
    assert any(k in e for k in missing for e in errors), errors
    assert any("missing" in e or "required" in e for e in errors), errors


def test_the_optional_signed_at_is_gone() -> None:
    doc = _nda()
    doc["evidence"]["precedent"][0]["signed_at"] = "2022-03-16"
    errors = _errors(_refresh(doc))
    assert any("signed_at" in e for e in errors), errors


def test_an_unsigned_deal_has_no_date() -> None:
    doc = _nda()
    corpus_doc = _document(doc, "beta-industries")
    corpus_doc["signed_version"] = None
    errors = _errors(_refresh(doc))
    assert any("signed_version is null" in e for e in errors), errors

    doc = _nda()
    for record in doc["evidence"]["precedent"]:
        if record["document_id"] == "beta-industries":
            record["signed"] = False
            record["opened_with"] = None
            record["opening_text"] = None
    errors = _errors(_refresh(doc))
    assert any("signed=false" in e for e in errors), errors


def test_an_edited_digest_span_is_rejected() -> None:
    doc = _refresh(_nda())
    doc["digest"]["clauses"][0]["first_signed"] = "2020-01"
    doc["identity"]["content_hash"] = content_hash(doc)
    assert any("digest" in e for e in _errors(doc))


# ---------------------------------------------------------------------------
# Digest 5: months
# ---------------------------------------------------------------------------


def test_digest_months_and_groups() -> None:
    doc = _nda()
    # Re-date the deals so variant and refused-ask groups span known months.
    _set_date(doc, "beta-industries", "2022-03", "filename")  # month precision
    _set_date(doc, "delta-ventures", "2023-07-31", "document_text")
    _set_date(doc, "epsilon-systems", None, None)
    _refresh(doc)
    digest = doc["digest"]
    assert digest["corpus"]["first_signed"] == "2022-03"
    assert digest["corpus"]["last_signed"] == "2023-07"
    for clause in digest["clauses"]:
        assert set(clause) >= {"first_signed", "last_signed"}
        for entry in clause["signed_variants"]:
            assert "first_signed" in entry and "last_signed" in entry
        for entry in clause["refused_asks"]:
            assert "first_signed" in entry and "last_signed" in entry
    month = r"^\d{4}-\d{2}$"
    import re

    for clause in digest["clauses"]:
        for value in (clause["first_signed"], clause["last_signed"]):
            assert value is None or re.match(month, value)
        assert (clause["first_signed"] is None) == (clause["last_signed"] is None)
        if clause["first_signed"]:
            assert clause["first_signed"] <= clause["last_signed"]
    assert _errors(doc) == []


def test_a_group_with_no_dated_deal_has_null_months() -> None:
    doc = _nda()
    for document_id in [d["document_id"] for d in doc["corpus"]["documents"]]:
        _set_date(doc, document_id, None, None)
    _refresh(doc)
    digest = doc["digest"]
    assert digest["corpus"]["first_signed"] is None and digest["corpus"]["last_signed"] is None
    for clause in digest["clauses"]:
        assert clause["first_signed"] is None and clause["last_signed"] is None
        for entry in [*clause["signed_variants"], *clause["refused_asks"]]:
            assert entry["first_signed"] is None and entry["last_signed"] is None
    assert _errors(doc) == []


def test_the_digest_stays_within_its_token_budget() -> None:
    from playbook_engine.digest import DIGEST_TOKEN_BUDGET, digest_token_estimate

    assert digest_token_estimate(_nda()["digest"]) <= DIGEST_TOKEN_BUDGET


def test_the_dossier_index_carries_the_date() -> None:
    doc = _nda()
    index = doc["provenance_index"]["documents"]
    dates = {d["document_id"]: d["signed_on"] for d in index}
    assert dates  # the NDA example indexes the deals behind its excerpts
    corpus_dates = {d["document_id"]: d["signed_on"] for d in doc["corpus"]["documents"]}
    for document_id, signed_on in dates.items():
        assert signed_on == corpus_dates[document_id]


# ---------------------------------------------------------------------------
# find_precedent ranks by the date
# ---------------------------------------------------------------------------


def test_find_precedent_ranks_by_the_signing_date() -> None:
    from playbook_engine.opf_accessors import find_precedent

    doc = _nda()
    ranked = find_precedent(doc, "governing_law", order=("last_signed",))
    dated = [r["signed_on"] for r in ranked if r["signed_on"] is not None]
    assert dated == sorted(dated, reverse=True)  # newest first
    seen_unknown = False
    for record in ranked:
        if record["signed_on"] is None:
            seen_unknown = True
        else:
            assert not seen_unknown, "an undated record outranked a dated one"


# ---------------------------------------------------------------------------
# Pseudonymization: the passages a judge is shown are born safe
# ---------------------------------------------------------------------------


def test_judge_passages_are_pseudonymized(tmp_path: Path) -> None:
    from playbook_engine.entity_registry import EntityRegistry
    from playbook_engine.pipeline import _pseudonymize_corpus_documents

    evidence = dd.extract_evidence(
        document_text=(
            "Dated as of March 3, 2021, amending the agreement dated June 1, 2019, between "
            "Acme Widgets Inc. and X."
        ),
        signature_block_text="",
        filename=None,
        file_timestamp=None,
    )
    assert evidence.passages is not None
    assert "Acme Widgets" in evidence.passages["document_text"][0]
    docs, _ = _pseudonymize_corpus_documents(
        [_doc("deal", signed_version=1, evidence=evidence)],
        ["Acme Widgets Inc."],
        EntityRegistry.load(tmp_path / "registry.json"),
    )
    shown = json.dumps(docs[0]["deal_date_evidence"])
    assert "Acme" not in shown and "Counterparty-" in shown


# ---------------------------------------------------------------------------
# The engine end to end over the synthetic NDA corpus
# ---------------------------------------------------------------------------


def _invoke(args: list[str]) -> Any:
    return CliRunner().invoke(cli, args)


@pytest.fixture(scope="module")
def mined(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Mine a copy of the NDA corpus in which two executed copies are altered and a draft is added:

    - ``theta-logistics`` has no date in its text; its signed file name gets one;
    - ``zeta-diagnostics`` has an ambiguous ``04/05/2023`` signature date (the
      committed example's ``03/04/2023`` already has a canned verdict, so this
      one is a question the judge has not seen);
    - ``eta-draft`` is a deal with no executed copy at all.
    """
    root = tmp_path_factory.mktemp("dates")
    corpus = root / "corpus"
    shutil.copytree(_NDA_DIR / "corpus", corpus)
    signed = corpus / "theta-logistics" / "v2.rtf"
    signed.rename(corpus / "theta-logistics" / "v2 signed 2022-12-01.rtf")
    zeta = corpus / "zeta-diagnostics" / "v3.rtf"
    zeta.write_text(
        zeta.read_text(encoding="utf-8").replace("03/04/2023", "04/05/2023"), encoding="utf-8"
    )
    # ``eta-draft`` never reached an executed copy: only a draft whose signature
    # lines are blank (a copy of an existing draft, so it adds only its scope
    # question and no segmentation or date question).
    # The engine finds no executed copy, so it must record no date for it.
    (corpus / "eta-draft").mkdir()
    shutil.copy(_NDA_DIR / "corpus" / "delta-ventures" / "v1.rtf", corpus / "eta-draft" / "v1.rtf")
    out = root / "out"
    out.mkdir()
    config = _NDA_DIR / "config.smoke.yaml"
    for args in (
        ["judge-apply", str(out), "--verdicts", str(_NDA_DIR / "canned-verdicts.jsonl")],
        ["mine", str(corpus), "--config", str(config), "--out", str(out)],
    ):
        result = _invoke(args)
        assert result.exit_code == 0, result.output
    _scope_the_added_draft(out, corpus, config)
    return {"corpus": corpus, "out": out, "config": config}


def _scope_the_added_draft(out: Path, corpus: Path, config: Path) -> None:
    """Answer the one scope question the added draft raises; it adds no other pending item.

    The canned verdicts cover the six committed deals; ``eta-draft`` is a new
    document, so ``judge`` queues its scope question, which is answered here
    (in scope, as the committed deals are) and mined again.
    """
    judged = _invoke(["judge", str(corpus), "--config", str(config), "--out", str(out)])
    assert judged.exit_code == 0, judged.output
    pending = [
        json.loads(line) for line in (out / "judge" / "pending.jsonl").read_text().splitlines()
    ]
    answers = [
        {
            "key": item["key"],
            "verdict": {
                "in_scope": True,
                "scope_confidence": 0.95,
                "scope_rationale": "Clause structure matches a Mutual Non-Disclosure Agreement.",
            },
        }
        for item in pending
        if item["kind"] == "scope"
    ]
    assert len(answers) == 1  # only the added draft is unknown
    verdicts = out / "scope-answers.jsonl"
    verdicts.write_text("".join(json.dumps(a) + "\n" for a in answers), encoding="utf-8")
    for args in (
        ["judge-apply", str(out), "--verdicts", str(verdicts)],
        ["mine", str(corpus), "--config", str(config), "--out", str(out)],
    ):
        result = _invoke(args)
        assert result.exit_code == 0, result.output


def _projected(paths: dict[str, Path]) -> dict[str, Any]:
    result = _invoke(["project", str(paths["out"]), "--config", str(paths["config"])])
    assert result.exit_code == 0, result.output
    playbook: dict[str, Any] = json.loads((paths["out"] / "playbook.opf.json").read_text())
    return playbook


def test_mining_records_evidence_for_executed_copies_only(mined: dict[str, Path]) -> None:
    manifest = json.loads((mined["out"] / "corpus_manifest.json").read_text())
    by_id = {d["document_id"]: d for d in manifest}
    # The six NDA deals are executed; the added draft never reached an executed copy.
    assert by_id["eta-draft"]["signed_version"] is None
    assert "deal_date_evidence" not in by_id["eta-draft"]
    for document_id, doc in by_id.items():
        if document_id != "eta-draft":
            assert "deal_date_evidence" in doc, document_id
    beta = dd.DealDateEvidence.from_dict(by_id["beta-industries"]["deal_date_evidence"])
    assert beta.signature_block == "2022-03-16" and beta.passages is None
    theta = dd.DealDateEvidence.from_dict(by_id["theta-logistics"]["deal_date_evidence"])
    assert (theta.document_text, theta.signature_block, theta.filename) == (
        None,
        None,
        "2022-12-01",
    )
    zeta = dd.DealDateEvidence.from_dict(by_id["zeta-diagnostics"]["deal_date_evidence"])
    assert zeta.signature_block is None and zeta.passages == {
        "document_text": [],
        "signature_block": ["Date: 04/05/2023"],
    }
    # The evidence carries the date read from the file name, never the name.
    assert "v2 signed" not in json.dumps(by_id["theta-logistics"]["deal_date_evidence"])


def test_project_reads_the_basis_and_the_file_name(mined: dict[str, Path]) -> None:
    playbook = _projected(mined)
    dated = {
        d["document_id"]: (d["signed_on"], d["signed_on_basis"])
        for d in playbook["corpus"]["documents"]
    }
    assert dated["beta-industries"] == ("2022-03-16", "signature_block")
    assert dated["theta-logistics"] == ("2022-12-01", "filename")
    # Ambiguous and no verdict yet: null, not a guess.
    assert dated["zeta-diagnostics"] == (None, None)
    # No executed copy: no date, and no precedent record of the deal claims one.
    assert dated["eta-draft"] == (None, None)
    assert all(
        r["signed_on"] is None
        for r in playbook["evidence"]["precedent"]
        if r["document_id"] == "eta-draft"
    )
    assert "deal_date_evidence" not in json.dumps(playbook["corpus"])
    assert validate_document(playbook).ok
    for record in playbook["evidence"]["precedent"]:
        assert record["signed_on"] == dated[record["document_id"]][0]


def test_judge_queues_the_ambiguous_date_and_judge_apply_resolves_it(
    mined: dict[str, Path],
) -> None:
    out = mined["out"]
    result = _invoke(
        ["judge", str(mined["corpus"]), "--config", str(mined["config"]), "--out", str(out)]
    )
    assert result.exit_code == 0, result.output
    assert "deal_date: 1" in result.output
    pending = [
        json.loads(line) for line in (out / "judge" / "pending.jsonl").read_text().splitlines()
    ]
    (item,) = [p for p in pending if p["kind"] == "deal_date"]
    assert item["payload"]["signature_block"] == ["Date: 04/05/2023"]
    assert item["rubric_version"] == rubric_version("deal_date")

    # An answer whose quote the judge was never shown is refused, with a line number.
    verdicts = out / "answers.jsonl"
    verdicts.write_text(
        json.dumps(
            {
                "key": item["key"],
                "verdict": {
                    "signed_on": "2023-05-04",
                    "basis": "signature_block",
                    "quote": "Date: 4 May 2023",
                },
            }
        )
        + "\n"
    )
    refused = _invoke(["judge-apply", str(out), "--verdicts", str(verdicts)])
    assert refused.exit_code == 1
    assert "does not occur" in refused.output

    verdicts.write_text(
        json.dumps(
            {
                "key": item["key"],
                "verdict": {
                    "signed_on": "2023-04-05",
                    "basis": "signature_block",
                    "quote": "Date: 04/05/2023",
                },
            }
        )
        + "\n"
    )
    applied = _invoke(["judge-apply", str(out), "--verdicts", str(verdicts)])
    assert applied.exit_code == 0, applied.output

    playbook = _projected(mined)
    zeta = next(
        d for d in playbook["corpus"]["documents"] if d["document_id"] == "zeta-diagnostics"
    )
    assert (zeta["signed_on"], zeta["signed_on_basis"]) == ("2023-04-05", "signature_block")
    assert validate_document(playbook).ok
    assert playbook["digest"]["corpus"]["last_signed"] == "2023-04"

    # The queue is drained: a second judge round asks nothing new.
    again = _invoke(
        ["judge", str(mined["corpus"]), "--config", str(mined["config"]), "--out", str(out)]
    )
    assert again.exit_code == 0, again.output
    assert "(0 pending items)" in again.output


def test_plan_only_counts_the_date_questions(mined: dict[str, Path]) -> None:
    """``judge --plan-only`` mines into a temp dir and must queue deal_date items too."""
    # A throwaway out-dir with the NDA canned verdicts so only the date item can be pending.
    plan_out = mined["out"].parent / "plan-out"
    plan_out.mkdir()
    applied = _invoke(
        ["judge-apply", str(plan_out), "--verdicts", str(_NDA_DIR / "canned-verdicts.jsonl")]
    )
    assert applied.exit_code == 0, applied.output
    result = _invoke(
        [
            "judge",
            str(mined["corpus"]),
            "--config",
            str(mined["config"]),
            "--out",
            str(plan_out),
            "--plan-only",
        ]
    )
    assert result.exit_code == 0, result.output
    assert "deal_date: 1" in result.output


def test_scorecard_counts_the_dates(mined: dict[str, Path]) -> None:
    from playbook_engine.scorecard import build_scorecard

    _projected(mined)
    card = build_scorecard(mined["out"])
    section = card["signed_on"]
    assert section["signed"] == 6
    assert section["dated"] + section["null_dates"] == section["signed"]
    assert set(section["by_basis"]) <= set(dd.BASES)
    # Counts only: no date value anywhere in the section.
    assert "20" not in json.dumps(section).replace('"in_scope"', "")


def test_the_committed_example_dates_four_of_its_six_deals() -> None:
    """The committed example carries a date on four deals and a recorded null on two."""
    documents = _nda()["corpus"]["documents"]
    assert all("signed_on" in d and "signed_on_basis" in d for d in documents)
    assert sum(1 for d in documents if d["signed_on"]) == 4
    assert sum(1 for d in documents if d["signed_on"] is None) == 2
