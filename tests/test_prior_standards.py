"""Prior standards: our former forms as a factual standard history (issue #249).

Past deals mostly opened on OLDER versions of our own form. Recording a clause
that opened on one of them as ``opened_with: prior_standard`` keeps our own
former wording from reading as the counterparty's, and gives the consumer a
factual history: per former form, its in-use dates, how its clause compares
with today's standard part by part, and how the deals that opened on it fared.

Fixture provenance (the producer of every shape used here):

- the end-to-end tests drive the REAL pipeline over a synthetic RTF corpus
  they write into ``tmp_path`` (the NDA example's standard form is the current
  template; the former form is that text with three clauses reworded; every
  deal is a v1 draft plus a signed v2, or a single signed version), through
  ``playbook mine`` -> ``playbook project`` -> ``playbook validate`` /
  ``view bundle``, so the observations, ``prior_templates.json``, the
  evidence, the digest and the page are what the engine writes;
- the builder-level tests hand-build the net-diff rows (``ClauseDiff``) the
  differ emits, exactly as ``tests/test_observation_builder.py`` does;
- equivalence verdicts are the dicts ``playbook judge-apply --verdicts``
  accepts (``equivalence.validate_draft_verdict`` is the function it runs),
  stored through ``VerdictStore`` as ``fold_overrides`` stores an owner's;
- the owner's note is an ``overrides.json`` entry ``parse_overrides`` parses.

SECURITY NOTE: synthetic text only; fictional parties.
"""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pytest
import yaml
from click.testing import CliRunner
from click.testing import Result as CliResult

import playbook_engine.pipeline as pipeline
from playbook_engine.agent_judge import VerdictStore
from playbook_engine.citation_resolver import (
    CitationResolutionError,
    ResolvedCitation,
    _resolve_ref,
)
from playbook_engine.clause_classifier import ClassifiedClause, ClauseClassification
from playbook_engine.clause_differ import ClauseDiff
from playbook_engine.clause_tree import ClauseNode, ClauseTree
from playbook_engine.cli import cli
from playbook_engine.config import load_config
from playbook_engine.deviation_classifier import DeviationResult
from playbook_engine.digest import build_digest_v5
from playbook_engine.equivalence import (
    EQUIVALENCE_KIND,
    ROLE_PRIOR_STANDARD,
    ROLE_SIGNED,
    collect_subjects,
    equivalence_key,
    summarize,
    validate_draft_verdict,
)
from playbook_engine.extraction import ExtractorLabel
from playbook_engine.observation_builder import (
    DROPPED_PRIOR_STANDARD_REMOVED,
    OUTCOME_CONCEDED_BEFORE_SIGNING,
    OUTCOME_OPENING,
    Observation,
    ObservationCitation,
    read_observations_jsonl,
)
from playbook_engine.observation_builder import build_observations as build_all_observations
from playbook_engine.overrides import (
    OverridesError,
    apply_overrides,
    fold_overrides,
    load_overrides,
    parse_overrides,
)
from playbook_engine.pipeline import (
    _restore_observations,
    _template_observations_from_classified,
    mine_corpus,
)
from playbook_engine.precedent import clause_counts, prior_form_counts
from playbook_engine.prior_standards import PriorForm, origin_standards
from playbook_engine.review_rows import build_review_rows
from playbook_engine.rtf_ingester import ingest_rtf
from playbook_engine.rubric import RubricStamp, rubric_version
from playbook_engine.scorecard import build_scorecard
from playbook_engine.segmentation_qa import SegmentationQAError
from playbook_engine.segmenter import segment
from playbook_engine.taxonomy import load_taxonomy
from playbook_engine.validator import validate_document

_REPO_ROOT = Path(__file__).resolve().parent.parent
_NDA_TEMPLATE = _REPO_ROOT / "examples" / "nda" / "standard-form.rtf"
_TAXONOMY = _REPO_ROOT / "spec" / "taxonomy" / "nda.yaml"

# ---------------------------------------------------------------------------
# Builder level: the opening of a clause is read against our former forms
# ---------------------------------------------------------------------------

_FORM_2018 = "2018 form"
_FORM_2020 = "2020 form"

_CURRENT = (
    "Neither party may assign this Agreement without the other party's prior written consent."
)
_PRIOR_2018 = "Either party may assign this Agreement to an affiliate without consent."
_PRIOR_2020 = "Neither party may assign this Agreement without written notice to the other party."
_THEIRS = "Assignment requires the written approval of the Disclosing Party's board."

_STD = {"assignment": [_CURRENT]}
_ORDINALS = {"draft_v1": 1, "draft_v2": 2, "signed_final": 3}


def _dr() -> DeviationResult:
    return DeviationResult(deviation="none")


def _node(path: str, text: str, tid: str = "assignment") -> ClassifiedClause:
    return ClassifiedClause(
        node=ClauseNode(clause_path=path, heading=None, text=text, char_span=(0, len(text))),
        classification=ClauseClassification(taxonomy_id=tid, confidence=0.9, basis="exact_match"),
    )


def _former_forms(*forms: tuple[str, dict[str, list[str]]]) -> dict[str, Any]:
    """Our former forms as the builder is handed them: ``forms`` are ``(label,
    {clause type: its clause nodes})`` in config order, and the mapping is the
    one the pipeline derives from them (the same template-observation producer
    ``mine`` runs on a form, then ``origin_standards``)."""
    prior_forms = []
    for label, clauses in forms:
        classified = [
            _node(f"{tid}.{i}", text, tid)
            for tid, texts in clauses.items()
            for i, text in enumerate(texts, start=1)
        ]
        prior_forms.append(
            PriorForm(
                label=label, observations=tuple(_template_observations_from_classified(classified))
            )
        )
    return origin_standards(prior_forms)


_PRIORS = _former_forms(
    (_FORM_2018, {"assignment": [_PRIOR_2018]}),
    (_FORM_2020, {"assignment": [_PRIOR_2020]}),
)


def _row(
    before: str | None, after: str | None, tid: str = "assignment", path: str = "9"
) -> ClauseDiff:
    """The net-diff row of one clause: its first-draft text, its signed text
    (``None``: struck before signing)."""
    kind = (
        "removed"
        if after is None
        else "added"
        if before is None
        else "unchanged"
        if before == after
        else "modified"
    )
    return ClauseDiff(
        taxonomy_id=tid,
        clause_path_before=None if before is None else path,
        clause_path_after=None if after is None else path,
        kind=kind,
        hunks=(),
        text_before=before or "",
        text_after=after or "",
        clause_version_before=None if before is None else "draft_v1",
        clause_version_after=None if after is None else "signed_final",
        char_span_before=None if before is None else (0, len(before)),
        char_span_after=None if after is None else (0, len(after)),
    )


def _run(
    before: str | None,
    after: str | None,
    *,
    priors: Any = _PRIORS,
    standards: Any = _STD,
    has_signed_copy: bool = True,
    dropped: dict[str, int] | None = None,
) -> list[Observation]:
    tree = [] if after is None else [_node("9", after)]
    return build_all_observations(
        "doc1",
        3,
        "our_paper",
        [(_row(before, after), _dr())],
        [],
        has_signed_copy=has_signed_copy,
        ordinal_by_vid=_ORDINALS,
        terminal_clauses=tree,
        terminal_version_id="signed_final",
        dropped=dropped,
        standard_text_by_tid=standards,
        prior_standards_by_tid=priors,
    )


def _terminal(obs: list[Observation]) -> Observation:
    (terminal,) = [o for o in obs if o.outcome in ("signed", "unsigned")]
    return terminal


def _opening(obs: list[Observation]) -> Observation:
    (opening,) = [o for o in obs if o.outcome == OUTCOME_OPENING]
    return opening


def test_opening_on_a_former_form_is_prior_standard_with_its_label() -> None:
    """Opened on the 2018 form, signed today's standard: the clause opened on
    OUR wording (`prior_standard`, `opened_with_form` the label), the signed
    text is today's standard, and the opening row carries the form's text."""
    obs = _run(_PRIOR_2018, _CURRENT)
    terminal, opening = _terminal(obs), _opening(obs)
    assert (terminal.opened_with, terminal.opened_with_form) == ("prior_standard", _FORM_2018)
    assert terminal.standard is True
    assert (opening.opened_with, opening.opened_with_form) == ("prior_standard", _FORM_2018)
    assert opening.full_text == _PRIOR_2018 and opening.standard is False


def test_signed_as_the_former_form_has_no_distinct_opening() -> None:
    obs = _run(_PRIOR_2020, _PRIOR_2020)
    terminal = _terminal(obs)
    assert (terminal.opened_with, terminal.opened_with_form) == ("prior_standard", _FORM_2020)
    assert terminal.standard is False
    assert not [o for o in obs if o.outcome == OUTCOME_OPENING]


def test_the_current_standard_wins_when_a_former_form_reads_the_same() -> None:
    """A former form whose clause is today's text: the clause opened with the
    current standard, with no form named."""
    priors = _former_forms((_FORM_2018, {"assignment": [_CURRENT]}))
    terminal = _terminal(_run(_CURRENT, _CURRENT, priors=priors))
    assert (terminal.opened_with, terminal.opened_with_form) == ("standard", None)


def test_a_counterparty_text_is_still_non_standard_with_formers_configured() -> None:
    terminal = _terminal(_run(_THEIRS, _THEIRS))
    assert (terminal.opened_with, terminal.opened_with_form) == ("non_standard", None)


def test_no_former_forms_changes_nothing() -> None:
    for priors in (None, {}, {"assignment": []}):
        terminal = _terminal(_run(_PRIOR_2018, _PRIOR_2018, priors=priors))
        assert (terminal.opened_with, terminal.opened_with_form) == ("non_standard", None)
        assert "opened_with_form" not in terminal.to_dict()


def test_a_former_form_is_matched_by_the_standard_checks_normalization() -> None:
    """The same normalization as the `standard` fact: case, spacing and
    punctuation are no difference."""
    shouty = _PRIOR_2018.upper().replace(" ", "  ")
    assert _terminal(_run(shouty, shouty)).opened_with_form == _FORM_2018


def test_a_former_form_needs_the_whole_clause_not_a_part_of_it() -> None:
    """A two-node former form: the first draft must be its whole clause (the
    standard fact's rule), not one node of it."""
    node_a, node_b = "The Supplier shall maintain insurance.", "Cover is for two million."
    priors = _former_forms((_FORM_2018, {"assignment": [node_a, node_b]}))
    assert _terminal(_run(node_a, node_a, priors=priors)).opened_with == "non_standard"
    assert (
        _terminal(_run(f"{node_a}\n{node_b}", f"{node_a}\n{node_b}", priors=priors)).opened_with
        == "prior_standard"
    )


def test_unsigned_deal_has_no_opening_origin() -> None:
    """A deal with no detected executed copy has no anchored opening, so no
    `opened_with` and no form."""
    obs = _run(_PRIOR_2018, _PRIOR_2018, has_signed_copy=False)
    assert [(o.opened_with, o.opened_with_form) for o in obs] == [(None, None)]


def test_our_former_wording_struck_before_signing_is_not_their_refused_ask() -> None:
    """Origin matters (issue #216): a clause of ours struck is no counterparty
    ask. Today's standard struck is our concession; a FORMER form's text struck
    produces no observation and is counted; a text neither of ours is still
    their refused ask."""
    dropped: dict[str, int] = {}
    obs = _run(_PRIOR_2018, None, dropped=dropped)
    assert [o.outcome for o in obs] == [OUTCOME_OPENING]
    assert obs[0].opened_with == "prior_standard"
    assert dropped == {DROPPED_PRIOR_STANDARD_REMOVED: 1}

    assert [o.outcome for o in _run(_CURRENT, None)] == [
        OUTCOME_CONCEDED_BEFORE_SIGNING,
        OUTCOME_OPENING,
    ]
    assert [o.outcome for o in _run(_THEIRS, None)] == ["proposed_then_reversed", OUTCOME_OPENING]


def test_opened_with_form_is_serialized_only_when_set_and_round_trips() -> None:
    terminal = _terminal(_run(_PRIOR_2018, _PRIOR_2018))
    d = terminal.to_dict()
    assert (d["opened_with"], d["opened_with_form"]) == ("prior_standard", _FORM_2018)
    (restored,) = _restore_observations([d])
    assert (restored.opened_with, restored.opened_with_form) == ("prior_standard", _FORM_2018)
    plain = _terminal(_run(_THEIRS, _THEIRS)).to_dict()
    assert "opened_with_form" not in plain
    (again,) = _restore_observations([plain])
    assert again.opened_with_form is None


def _record(
    doc: str,
    *,
    opened_with: str | None,
    form: str | None = None,
    opening: bool = False,
    standard: bool = False,
    signed_text: bool = True,
    signed: bool = True,
) -> dict[str, Any]:
    """A precedent record of the shape ``precedent.build_precedent_evidence``
    writes, for the counting function only (the end-to-end tests below build
    them through the pipeline)."""
    return {
        "document_id": doc,
        "signed": signed,
        "opened_with": opened_with,
        **({"opened_with_form": form} if form else {}),
        "opening_text": {"text": "x"} if opening else None,
        "signed_text": {"text": "y"} if signed_text else None,
        "standard": standard,
    }


def test_prior_form_counts_held_moved_and_conceded() -> None:
    records = [
        _record("kept-1", opened_with="prior_standard", form=_FORM_2018),
        _record("kept-2", opened_with="prior_standard", form=_FORM_2018),
        _record(
            "moved", opened_with="prior_standard", form=_FORM_2018, opening=True, standard=True
        ),
        _record(
            "variant", opened_with="prior_standard", form=_FORM_2020, opening=True, standard=False
        ),
        _record(
            "struck",
            opened_with="prior_standard",
            form=_FORM_2020,
            opening=True,
            signed_text=False,
        ),
        _record("current", opened_with="standard", standard=True),
        _record("theirs", opened_with="non_standard", opening=True),
        _record("unsigned", opened_with=None, signed=False),
    ]
    assert prior_form_counts(records) == {"n_opened": 5, "n_kept": 2, "n_moved_to_current": 1}
    assert prior_form_counts(records, _FORM_2018) == {
        "n_opened": 3,
        "n_kept": 2,
        "n_moved_to_current": 1,
    }
    assert prior_form_counts(records, _FORM_2020) == {
        "n_opened": 2,
        "n_kept": 0,
        "n_moved_to_current": 0,
    }
    # A deal opened on one form is one deal however many rows: distinct deals.
    assert prior_form_counts([*records, records[0]])["n_opened"] == 5
    counts = clause_counts(
        "assignment", [{**r, "taxonomy_id": "assignment"} for r in records], party=None
    )
    assert (counts["n_opened_prior"], counts["n_kept_prior"], counts["n_moved_to_current"]) == (
        5,
        2,
        1,
    )
    # Our own former wording is never a changed (counterparty) opening.
    assert counts["n_changed_openings"] == 1


# ---------------------------------------------------------------------------
# End to end: a synthetic corpus through mine -> project -> validate
# ---------------------------------------------------------------------------

_TERM_H = "9. Term"
_SURV_H = "10. Survival of Confidentiality Obligations"
_CARE_H = "6. Standard of Care"

_FORM_2018_LABEL = "2018 form"


def _body(rtf: str, heading: str) -> str:
    match = re.search(re.escape(heading) + r"\\par (.*?)\\par ", rtf)
    assert match is not None, heading
    return match.group(1)


_TEMPLATE_RTF = _NDA_TEMPLATE.read_text(encoding="utf-8")
_CARE_NOW = _body(_TEMPLATE_RTF, _CARE_H)
_TERM_NOW = _body(_TEMPLATE_RTF, _TERM_H)
_SURV_NOW = _body(_TEMPLATE_RTF, _SURV_H)
_CARE_OLD = (
    "Recipient shall protect Confidential Information using the same degree of care it uses "
    "for its own confidential information, and no less than reasonable care."
)
_TERM_OLD = (
    "This Agreement takes effect on the date of last signature and continues for one (1) year, "
    "unless earlier terminated by either party on sixty (60) days written notice."
)
_SURV_OLD = (
    "Recipient's obligations with respect to Confidential Information continue for five (5) "
    "years from the date of its disclosure, notwithstanding any expiration or termination of "
    "this Agreement."
)
_SURV_VARIANT = (
    "Recipient's obligations with respect to Confidential Information continue for seven (7) "
    "years from the date of its disclosure, notwithstanding any expiration or termination of "
    "this Agreement."
)
_TERM_THEIRS = (
    "This Agreement takes effect on the date of last signature and continues for six (6) "
    "months, unless earlier terminated by either party on ten (10) days written notice."
)

_BLANK_SIG = r"By: ______________________\par Name:\par Title:"


def _swap(rtf: str, old: str, new: str) -> str:
    assert old in rtf, old
    return rtf.replace(old, new, 1)


def _form(care: str, term: str, surv: str) -> str:
    rtf = _swap(_TEMPLATE_RTF, _CARE_NOW, care)
    rtf = _swap(rtf, _TERM_NOW, term)
    return _swap(rtf, _SURV_NOW, surv)


def _draft(form: str, counterparty: str) -> str:
    return form.replace("[Counterparty]", counterparty)


def _signed(form: str, counterparty: str, date: str) -> str:
    rtf = _draft(form, counterparty)
    for who in ("Dana Reyes", "Morgan Ellery"):
        rtf = _swap(
            rtf,
            _BLANK_SIG,
            rf"By: /s/ {who}\par Name: {who}\par Title: Counsel\par Date: {date}",
        )
    return _swap(
        rtf,
        "1. Parties and Recitals",
        r"DocuSign Envelope ID: 7f3a2c9e-1b4d\par 1. Parties and Recitals",
    )


_FORM_OLD = _form(_CARE_OLD, _TERM_OLD, _SURV_OLD)

#: Deal -> its versions (draft first), as RTF text. Who opened on what:
#:   alpha:   the 2018 form, signed as it stood (term, survival and care kept);
#:   bravo:   the 2018 form; term moved to today's, survival conceded to a
#:            variant, care kept;
#:   charlie: today's standard throughout;
#:   delta:   the counterparty's own term, signed as proposed;
#:   echo:    a single signed copy of the 2018 form (kept).
_DEALS: dict[str, list[str]] = {
    "alpha-labs": [
        _draft(_FORM_OLD, "Alpha Labs Inc."),
        _signed(_FORM_OLD, "Alpha Labs Inc.", "March 4, 2019"),
    ],
    "bravo-works": [
        _draft(_FORM_OLD, "Bravo Works LLC"),
        _signed(_form(_CARE_OLD, _TERM_NOW, _SURV_VARIANT), "Bravo Works LLC", "June 10, 2020"),
    ],
    "charlie-systems": [
        _draft(_TEMPLATE_RTF, "Charlie Systems Corp."),
        _signed(_TEMPLATE_RTF, "Charlie Systems Corp.", "January 20, 2022"),
    ],
    "delta-partners": [
        _draft(_swap(_TEMPLATE_RTF, _TERM_NOW, _TERM_THEIRS), "Delta Partners LP"),
        _signed(_swap(_TEMPLATE_RTF, _TERM_NOW, _TERM_THEIRS), "Delta Partners LP", "May 2, 2021"),
    ],
    "echo-group": [_signed(_FORM_OLD, "Echo Group Ltd.", "October 7, 2019")],
}


def _write_corpus(root: Path, deals: Iterable[str] | None = None) -> Path:
    corpus = root / "corpus"
    for deal, versions in _DEALS.items():
        if deals is not None and deal not in deals:
            continue
        (corpus / deal).mkdir(parents=True)
        for i, text in enumerate(versions, start=1):
            (corpus / deal / f"v{i}.rtf").write_text(text, encoding="utf-8")
    return corpus


def _write_config(
    root: Path, *, priors: list[dict[str, Any]] | None, name: str = "config.yaml"
) -> Path:
    (root / "standard-form.rtf").write_text(_TEMPLATE_RTF, encoding="utf-8")
    (root / "form-2018.rtf").write_text(_FORM_OLD, encoding="utf-8")
    baseline: dict[str, Any] = {"template": "standard-form.rtf"}
    if priors is not None:
        baseline["prior_templates"] = priors
    config = {
        "agreement_type": {"id": "nda", "name": "Mutual Non-Disclosure Agreement"},
        "perspective": {
            "party": "AlphaCorp Holdings, Inc.",
            "counterparty_type": "Prospective Commercial Counterparty",
        },
        "baseline": baseline,
        "taxonomy": str(_TAXONOMY),
        "provenance": {
            "our_party_aliases": ["AlphaCorp Holdings, Inc.", "AlphaCorp"],
            "known_entities": [],
        },
    }
    path = root / name
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


_PRIORS_2018 = [{"path": "form-2018.rtf", "label": _FORM_2018_LABEL, "in_use_from": 2018}]


def _invoke(args: list[str]) -> CliResult:
    return CliRunner().invoke(cli, args)


def _ok(args: list[str]) -> CliResult:
    result = _invoke(args)
    assert result.exit_code == 0, f"{' '.join(args)} failed:\n{result.output}"
    return result


def _derive(root: Path, config: Path, out: str = "out") -> Path:
    out_dir = root / out
    _ok(["mine", str(root / "corpus"), "--config", str(config), "--out", str(out_dir)])
    _ok(["project", str(out_dir), "--config", str(config)])
    return out_dir


@pytest.fixture(scope="module")
def derived(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """One derivation WITH the 2018 form and one WITHOUT (same corpus), plus the
    mine output of the first."""
    root = tmp_path_factory.mktemp("prior-standards")
    _write_corpus(root)
    with_cfg = _write_config(root, priors=_PRIORS_2018)
    without_cfg = _write_config(root, priors=None, name="config.none.yaml")
    mine = _ok(
        ["mine", str(root / "corpus"), "--config", str(with_cfg), "--out", str(root / "out")]
    )
    _ok(["project", str(root / "out"), "--config", str(with_cfg)])
    _derive(root, without_cfg, out="out-none")
    return {
        "root": root,
        "config": with_cfg,
        "out": root / "out",
        "out_none": root / "out-none",
        "mine_output": mine.output,
        "playbook": json.loads((root / "out" / "playbook.opf.json").read_text(encoding="utf-8")),
        "baseline_playbook": json.loads(
            (root / "out-none" / "playbook.opf.json").read_text(encoding="utf-8")
        ),
    }


def _clause(playbook: dict[str, Any], tid: str) -> dict[str, Any]:
    (clause,) = [c for c in playbook["evidence"]["clauses"] if c["taxonomy_id"] == tid]
    return clause


def _records(playbook: dict[str, Any], tid: str) -> dict[str, dict[str, Any]]:
    return {
        p["document_id"]: p for p in playbook["evidence"]["precedent"] if p["taxonomy_id"] == tid
    }


def _tids(playbook: dict[str, Any]) -> dict[str, str]:
    """Clause ids of the three reworded clauses, by the heading they carry."""
    by_text = {}
    for c in playbook["evidence"]["clauses"]:
        text = (c.get("our_standard") or {}).get("text") or ""
        for name, now in (("care", _CARE_NOW), ("term", _TERM_NOW), ("surv", _SURV_NOW)):
            if now in text:
                by_text[name] = c["taxonomy_id"]
    assert set(by_text) == {"care", "term", "surv"}, by_text
    return by_text


def test_mine_reports_the_former_form_and_persists_it(derived: dict[str, Any]) -> None:
    assert re.search(
        r"prior template '2018 form': \d+ clause type\(s\) classified", derived["mine_output"]
    )
    persisted = json.loads((derived["out"] / "prior_templates.json").read_text(encoding="utf-8"))
    (form,) = persisted["forms"]
    assert (form["label"], form["in_use_from"], form["in_use_until"]) == (
        _FORM_2018_LABEL,
        "2018",
        None,
    )
    assert form["sha256"] and form["observations"]
    assert not (derived["out_none"] / "prior_templates.json").exists()


def test_a_deal_that_opened_on_the_former_form_is_recorded_as_ours(derived: dict[str, Any]) -> None:
    playbook = derived["playbook"]
    tids = _tids(playbook)
    term = _records(playbook, tids["term"])
    for deal in ("alpha-labs", "bravo-works", "echo-group"):
        assert (term[deal]["opened_with"], term[deal]["opened_with_form"]) == (
            "prior_standard",
            _FORM_2018_LABEL,
        ), deal
    assert (
        term["charlie-systems"]["opened_with"],
        "opened_with_form" in term["charlie-systems"],
    ) == (
        "standard",
        False,
    )
    assert term["delta-partners"]["opened_with"] == "non_standard"
    # Signed as it stood: no distinct opening. Moved to today's: the opening is recorded.
    assert term["alpha-labs"]["opening_text"] is None and term["alpha-labs"]["standard"] is False
    assert term["bravo-works"]["standard"] is True
    assert term["bravo-works"]["opening_text"]["text"].strip() == _TERM_OLD


def test_counts_are_computed_against_the_form_we_sent(derived: dict[str, Any]) -> None:
    playbook = derived["playbook"]
    tids = _tids(playbook)
    expected = {
        # term: alpha and echo kept the 2018 term, bravo moved to today's
        "term": (3, 2, 1),
        # survival: alpha and echo kept it, bravo conceded to a variant
        "surv": (3, 2, 0),
        # standard of care: all three kept it
        "care": (3, 3, 0),
    }
    for name, counts in expected.items():
        clause = _clause(playbook, tids[name])
        assert (
            clause["n_opened_prior"],
            clause["n_kept_prior"],
            clause["n_moved_to_current"],
        ) == counts, name
    # Charlie opened with today's standard everywhere and kept it.
    term = _clause(playbook, tids["term"])
    assert (term["n_opened_standard"], term["n_kept_standard"]) == (1, 1)
    # Every clause still validates: the counts are what the records imply.
    assert validate_document(playbook).ok


def test_our_former_wording_is_neither_a_changed_opening_nor_a_refused_ask(
    derived: dict[str, Any],
) -> None:
    """WITHOUT the form, our own former wording reads as the counterparty's: the
    survival text bravo opened on (and did not sign) is a changed opening, and
    the term text it struck is a refused ask. WITH it, both are ours."""
    tids = _tids(derived["playbook"])
    surv, term = tids["surv"], tids["term"]
    before, after = derived["baseline_playbook"], derived["playbook"]
    assert _clause(before, surv)["n_changed_openings"] == 1
    assert _clause(after, surv)["n_changed_openings"] == 0
    assert _clause(before, term)["n_refused"] == 1
    assert _clause(after, term)["n_refused"] == 0
    # The refused ask is not lost silently: it is counted, by name.
    assert after["corpus"]["stats"]["dropped_observations"]["by_reason"] == {
        "removed_prior_standard": 1
    }
    assert "removed_prior_standard" not in json.dumps(before["corpus"]["stats"])
    digest = {c["taxonomy_id"]: c for c in after["digest"]["clauses"]}
    assert digest[surv]["changed_openings"] == [] and digest[term]["refused_asks"] == []
    # Their own term, signed as proposed, stays theirs either way.
    for playbook in (after, before):
        assert _records(playbook, term)["delta-partners"]["opened_with"] == "non_standard"


def test_zero_former_forms_leaves_the_output_as_it_was(derived: dict[str, Any]) -> None:
    baseline = derived["baseline_playbook"]
    assert "prior_templates" not in baseline["baseline"]
    for clause in baseline["evidence"]["clauses"]:
        assert clause["prior_standards"] == [] and clause["why_changed"] is None
        assert (clause["n_opened_prior"], clause["n_kept_prior"], clause["n_moved_to_current"]) == (
            0,
            0,
            0,
        )
    assert all("opened_with_form" not in p for p in baseline["evidence"]["precedent"])
    assert all(
        c["standard_history"] == {"forms": [], "why_changed": None}
        for c in baseline["digest"]["clauses"]
    )
    assert validate_document(baseline).ok


def test_each_former_clause_is_judged_once_and_shares_a_signed_variants_key(
    derived: dict[str, Any],
) -> None:
    """The 2018 form's term is a subject of role `prior_standard`; the same words
    signed by alpha and echo are the same text, so ONE subject plays both roles,
    and nothing in its payload names a deal, a counterparty or a paper side."""
    playbook = derived["playbook"]
    tid = _tids(playbook)["term"]
    subjects = collect_subjects(playbook["evidence"], "nda", "AlphaCorp Holdings, Inc.")
    prior = [s for s in subjects if ROLE_PRIOR_STANDARD in s.roles]
    assert prior and all(s.payload["taxonomy_id"] for s in prior)
    term = [s for s in prior if s.payload["taxonomy_id"] == tid]
    assert len(term) == 1
    assert term[0].roles == (ROLE_SIGNED, ROLE_PRIOR_STANDARD)
    assert term[0].payload["candidate"].strip() == _TERM_OLD
    blob = json.dumps(term[0].payload).lower()
    for word in ("alpha-labs", "bravo", "echo-group", "paper", "counterparty-"):
        assert word not in blob
    # An unchanged clause (the form reads as today's) is never a subject.
    unchanged = [
        c
        for c in playbook["evidence"]["clauses"]
        if c["prior_standards"] and c["prior_standards"][0]["text"] == c["our_standard"]["text"]
    ]
    assert unchanged
    keys = {s.key for s in subjects}
    for clause in unchanged:
        assert (
            equivalence_key(
                "nda",
                clause["taxonomy_id"],
                "AlphaCorp Holdings, Inc.",
                clause["prior_standards"][0]["text"],
                [p["text"] for p in clause["our_standard"]["parts"]],
            )
            not in keys
        )


def _store_verdict(out_dir: Path, subject_key: str, part_ids: list[str], relation: str) -> None:
    verdict = validate_draft_verdict(
        {
            "parts": [
                {
                    "part_id": pid,
                    "status": "present",
                    "relation": relation,
                    **({} if relation == "equivalent" else {"reason": "A synthetic difference."}),
                }
                for pid in part_ids
            ],
            "additions": [],
            "reason": "The synthetic difference is the single most important one.",
            "basis": "agent",
        },
        part_ids,
    )
    store = VerdictStore(out_dir / "judge" / "verdicts.jsonl")
    store.put_by_key(
        subject_key,
        verdict,
        rubric=RubricStamp(kind=EQUIVALENCE_KIND, version=rubric_version(EQUIVALENCE_KIND)),
    )


def test_a_judged_former_clause_reaches_the_digest_with_its_part_map(
    tmp_path: Path,
) -> None:
    _write_corpus(tmp_path)
    config = _write_config(tmp_path, priors=_PRIORS_2018)
    out = _derive(tmp_path, config)
    playbook = json.loads((out / "playbook.opf.json").read_text(encoding="utf-8"))
    tid = _tids(playbook)["term"]
    subject = next(
        s
        for s in collect_subjects(playbook["evidence"], "nda", "AlphaCorp Holdings, Inc.")
        if ROLE_PRIOR_STANDARD in s.roles and s.payload["taxonomy_id"] == tid
    )
    before = _clause(playbook, tid)["prior_standards"][0]
    assert before["vs_standard"] is None  # never guessed
    _store_verdict(out, subject.key, subject.part_ids, "less_protective")
    _ok(["project", str(out), "--config", str(config)])
    playbook = json.loads((out / "playbook.opf.json").read_text(encoding="utf-8"))
    assert validate_document(playbook).ok
    prior = _clause(playbook, tid)["prior_standards"][0]
    assert (prior["form_label"], prior["vs_standard"]["label"]) == (
        _FORM_2018_LABEL,
        "less_protective",
    )
    # The two signed variants of the same words carry the same judgment.
    for deal in ("alpha-labs", "echo-group"):
        assert _records(playbook, tid)[deal]["signed_text"]["vs_standard"] == prior["vs_standard"]

    history = {c["taxonomy_id"]: c for c in playbook["digest"]["clauses"]}[tid]["standard_history"]
    (form,) = history["forms"]
    assert form["form"] == _FORM_2018_LABEL
    assert (form["n_opened"], form["n_kept"], form["n_conceded"], form["n_moved_to_current"]) == (
        3,
        2,
        1,
        1,
    )
    assert (form["label"], form["part_map"], form["settled"]) == (
        "less_protective",
        ["-"],
        "unchecked",
    )
    assert form["unchanged"] is False and form["parts"][0]["text"].strip() == _TERM_OLD
    # Config gave the start; the end is inferred from the deals that opened on it.
    assert (form["in_use_from"], form["in_use_until"], form["in_use_basis"]) == (
        "2018",
        "2020-06",
        "mixed",
    )
    assert history["why_changed"] is None
    # An unchanged clause carries no part map.
    unchanged = [
        f
        for c in playbook["digest"]["clauses"]
        for f in c["standard_history"]["forms"]
        if f["unchanged"]
    ]
    assert unchanged and all(f["part_map"] is None and f["label"] is None for f in unchanged)

    # The Review tab lists the judgment as a row of its own (a former form's
    # clause belongs to no deal), shared with the signed variants that carry
    # the same words.
    rows = [r for r in build_review_rows(playbook).rows if r["taxonomy_id"] == tid]
    (row,) = rows
    assert ROLE_PRIOR_STANDARD in row["roles"] and row["label"] == "less_protective"
    assert ROLE_SIGNED in row["roles"] and row["key"] == subject.key

    # The scorecard counts it, and only counts.
    card = build_scorecard(out)
    section = card["prior_standards"]
    assert section["forms"] == 1 and section["judged"] == 1 and section["why_changed"] == 0
    assert section["entries"] >= 3 and section["unchanged"] == len(unchanged)
    assert (section["opened"], section["kept"], section["moved_to_current"]) == (9, 7, 1)
    assert card["equivalence"]["by_role"]["prior_standard"]["drafted"] == 1
    # The two signed variants (alpha, echo) are the SAME text as the form's term:
    # one distinct text in the signed role, one in the prior role.
    assert card["equivalence"]["by_role"]["signed"]["drafted"] == 1
    assert card["equivalence"]["totals"]["drafted"] == 1
    assert card["precedent"]["records_by_opened_with"]["prior_standard"] == 9
    summary = summarize(playbook["evidence"], "nda", "AlphaCorp Holdings, Inc.")
    assert (
        summary["by_role"]["prior_standard"]["eligible"]
        == summary["by_role"]["prior_standard"]["drafted"]
        + summary["by_role"]["prior_standard"]["unjudged"]
    )


def test_the_owners_why_changed_note_round_trips(tmp_path: Path) -> None:
    _write_corpus(tmp_path)
    config = _write_config(tmp_path, priors=_PRIORS_2018)
    out = _derive(tmp_path, config)
    playbook = json.loads((out / "playbook.opf.json").read_text(encoding="utf-8"))
    tid = _tids(playbook)["term"]
    note = "We moved to a two-year term because one year was too short to finish a pilot."
    (out / "overrides.json").write_text(
        json.dumps(
            {
                "overrides_version": 1,
                "overrides": [
                    {"target": "why_changed", "id": tid, "value": note, "basis": "owner"}
                ],
            }
        ),
        encoding="utf-8",
    )
    _ok(["project", str(out), "--config", str(config)])
    playbook = json.loads((out / "playbook.opf.json").read_text(encoding="utf-8"))
    assert validate_document(playbook).ok
    assert _clause(playbook, tid)["why_changed"] == {"text": note, "basis": "owner"}
    digest = {c["taxonomy_id"]: c for c in playbook["digest"]["clauses"]}
    assert digest[tid]["standard_history"]["why_changed"] == {"text": note, "basis": "owner"}
    # Every other clause stays null: the note is optional and per clause.
    assert [
        c["taxonomy_id"] for c in playbook["evidence"]["clauses"] if c["why_changed"] is not None
    ] == [tid]
    assert build_scorecard(out)["prior_standards"]["why_changed"] == 1
    assert playbook["digest"] == build_digest_v5(playbook)

    # The page: an optional edit box per clause with a former form, the page
    # script accepts the target, and the page carries the saved note.
    _ok(["view", "bundle", str(out)])
    page = (out / "index.html").read_text(encoding="utf-8")
    assert 'data-target="why_changed"' in page and "Why we changed this" in page
    assert "e.target === 'why_changed'" in page
    pf = json.loads(re.search(r'id="pf-data"[^>]*>(.*?)</script>', page, re.S).group(1))  # type: ignore[union-attr]
    assert pf["why_changed"] == {tid: note}

    # Applying the file again changes nothing; the strict command agrees.
    before = (out / "playbook.opf.json").read_text(encoding="utf-8")
    _ok(["apply-overrides", str(out)])
    assert (out / "playbook.opf.json").read_text(encoding="utf-8") == before


def test_why_changed_override_shape_and_unknown_clause() -> None:
    ok = {"target": "why_changed", "id": "term", "value": "Because.", "basis": "owner"}
    (entry,) = parse_overrides({"overrides_version": 1, "overrides": [ok]})
    assert (entry.target, entry.id, entry.value) == ("why_changed", "term", "Because.")
    for bad in (
        {**ok, "value": "  "},
        {**ok, "field": "statement"},
        {**ok, "basis": "agent"},
    ):
        with pytest.raises(OverridesError):
            parse_overrides({"overrides_version": 1, "overrides": [bad]})
    playbook = {"evidence": {"clauses": [{"taxonomy_id": "term", "why_changed": None}]}}
    with pytest.raises(OverridesError, match="no clause of this playbook has that taxonomy_id"):
        apply_overrides(
            copy.deepcopy(playbook),
            [parse_overrides({"overrides_version": 1, "overrides": [{**ok, "id": "nope"}]})[0]],
        )
    result = apply_overrides(playbook, [entry])
    assert result.changed["why_changed"] == 1
    assert playbook["evidence"]["clauses"][0]["why_changed"] == {
        "text": "Because.",
        "basis": "owner",
    }
    again = apply_overrides(playbook, [entry])
    assert again.unchanged == 1 and again.n_changed == 0


def test_a_stale_prior_templates_file_is_removed_when_no_form_is_configured(
    tmp_path: Path,
) -> None:
    _write_corpus(tmp_path)
    with_cfg = _write_config(tmp_path, priors=_PRIORS_2018)
    out = tmp_path / "out"
    _ok(["mine", str(tmp_path / "corpus"), "--config", str(with_cfg), "--out", str(out)])
    assert (out / "prior_templates.json").is_file()
    without = _write_config(tmp_path, priors=None, name="config.none.yaml")
    _ok(["mine", str(tmp_path / "corpus"), "--config", str(without), "--out", str(out)])
    assert not (out / "prior_templates.json").exists()
    _ok(["project", str(out), "--config", str(without)])
    playbook = json.loads((out / "playbook.opf.json").read_text(encoding="utf-8"))
    assert playbook["evidence"]["clauses"][0]["prior_standards"] == []
    assert all(p["opened_with"] != "prior_standard" for p in playbook["evidence"]["precedent"])


# ---------------------------------------------------------------------------
# The validator holds the new facts to account
# ---------------------------------------------------------------------------


def _messages(playbook: dict[str, Any]) -> list[str]:
    return [str(e) for e in validate_document(playbook).errors if e.blocking]


def test_validator_rejects_a_missing_or_stray_opened_with_form(derived: dict[str, Any]) -> None:
    playbook = copy.deepcopy(derived["playbook"])
    prior = next(
        p for p in playbook["evidence"]["precedent"] if p["opened_with"] == "prior_standard"
    )
    del prior["opened_with_form"]
    assert any(
        "opened_with='prior_standard' but opened_with_form is missing" in m
        for m in _messages(playbook)
    )

    playbook = copy.deepcopy(derived["playbook"])
    other = next(p for p in playbook["evidence"]["precedent"] if p["opened_with"] == "standard")
    other["opened_with_form"] = _FORM_2018_LABEL
    assert any(
        "opened_with_form is set but opened_with='standard'" in m for m in _messages(playbook)
    )


def test_validator_rejects_a_form_that_was_not_supplied(derived: dict[str, Any]) -> None:
    playbook = copy.deepcopy(derived["playbook"])
    next(p for p in playbook["evidence"]["precedent"] if p["opened_with"] == "prior_standard")[
        "opened_with_form"
    ] = "1999 form"
    assert any("not in baseline.prior_templates" in m for m in _messages(playbook))

    playbook = copy.deepcopy(derived["playbook"])
    next(c for c in playbook["evidence"]["clauses"] if c["prior_standards"])["prior_standards"][0][
        "form_label"
    ] = "1999 form"
    assert any("which is not in baseline.prior_templates" in m for m in _messages(playbook))


def test_validator_recomputes_the_prior_counts_and_parts(derived: dict[str, Any]) -> None:
    playbook = copy.deepcopy(derived["playbook"])
    clause = next(c for c in playbook["evidence"]["clauses"] if c["n_opened_prior"])
    clause["n_kept_prior"] += 1
    assert any("n_kept_prior" in m and "implies" in m for m in _messages(playbook))

    playbook = copy.deepcopy(derived["playbook"])
    clause = next(c for c in playbook["evidence"]["clauses"] if c["prior_standards"])
    clause["prior_standards"][0]["parts"][0]["part_id"] = "00000000"
    assert any("prior_standards entry part_id" in m for m in _messages(playbook))

    playbook = copy.deepcopy(derived["playbook"])
    clause = next(c for c in playbook["evidence"]["clauses"] if c["prior_standards"])
    clause["prior_standards"][0]["text"] += " extra"
    assert any(
        "prior_standards entry.text is not its parts joined" in m for m in _messages(playbook)
    )


def test_validator_rejects_a_judgment_on_a_form_that_reads_as_todays(
    derived: dict[str, Any],
) -> None:
    playbook = copy.deepcopy(derived["playbook"])
    clause = next(
        c
        for c in playbook["evidence"]["clauses"]
        if c["prior_standards"] and c["prior_standards"][0]["text"] == c["our_standard"]["text"]
    )
    pids = [p["part_id"] for p in clause["our_standard"]["parts"]]
    clause["prior_standards"][0]["vs_standard"] = {
        "label": "equivalent",
        "reason": "Same.",
        "parts": [{"part_id": pid, "status": "present", "relation": "equivalent"} for pid in pids],
        "additions": [],
        "basis": "agent",
        "check": None,
    }
    assert any("reads exactly as today's standard" in m for m in _messages(playbook))


def test_validator_rejects_a_repeated_form_label(derived: dict[str, Any]) -> None:
    playbook = copy.deepcopy(derived["playbook"])
    playbook["baseline"]["prior_templates"].append(
        copy.deepcopy(playbook["baseline"]["prior_templates"][0])
    )
    assert any("labels are unique" in m for m in _messages(playbook))


def test_baseline_records_the_supplied_form(derived: dict[str, Any]) -> None:
    (form,) = derived["playbook"]["baseline"]["prior_templates"]
    assert form["label"] == _FORM_2018_LABEL and form["in_use_from"] == "2018"
    assert "in_use_until" not in form and re.fullmatch(r"sha256:[0-9a-f]{64}", form["sha256"])


# ---------------------------------------------------------------------------
# A former form's clause is cited in the FORM's own bytes (OPF-SPEC §4.1)
# ---------------------------------------------------------------------------


def _flat(text: str) -> str:
    return " ".join(text.split())


def _cited_clause_text(resolved: ResolvedCitation) -> str:
    """The text of the clause a resolved citation names, read from the verified
    file the way the engine read it (ingest, segment, then the clause path)."""
    tree = segment(ingest_rtf(resolved.file_path, "cited", "cited").tree)
    assert resolved.clause_path is not None
    node = tree.resolve_path(resolved.clause_path)
    assert node is not None, resolved.clause_path
    if resolved.char_span is not None:
        assert tuple(node.char_span) == resolved.char_span
    return _flat(node.text)


def test_a_former_forms_citation_resolves_to_the_forms_own_file_and_text(
    derived: dict[str, Any],
) -> None:
    """The 2018 form's term clause, from ``prior_standards`` AND from the
    digest's ``standard_history``, resolves (hash-verified) to form-2018.rtf
    and its old term text, never to today's template and today's wording."""
    playbook = derived["playbook"]
    root: Path = derived["root"]
    term = _tids(playbook)["term"]
    (form,) = playbook["baseline"]["prior_templates"]
    (prior,) = _clause(playbook, term)["prior_standards"]
    (digest_clause,) = [c for c in playbook["digest"]["clauses"] if c["taxonomy_id"] == term]
    (history,) = digest_clause["standard_history"]["forms"]

    expected_id = {"document_id": "prior_template:2018 form", "version": "prior_template"}
    for ref in (prior["source_ref"], history["ref"]):
        assert {k: ref[k] for k in expected_id} == expected_id
        resolved = _resolve_ref(playbook, ref, "former form", root)
        assert resolved.file_path == root / "form-2018.rtf"
        assert resolved.sha256 == form["sha256"]
        assert resolved.sha256 != playbook["baseline"]["template_ref"]["sha256"]
        text = _cited_clause_text(resolved)
        assert _flat(_TERM_OLD) in text and _flat(_TERM_NOW) not in text
    assert history["ref"] == prior["source_ref"]

    # Today's standard still resolves to today's template and today's words.
    standard_ref = _clause(playbook, term)["our_standard"]["source_ref"]
    resolved = _resolve_ref(playbook, standard_ref, "our standard", root)
    assert resolved.file_path == root / "standard-form.rtf"
    assert _flat(_TERM_NOW) in _cited_clause_text(resolved)


def test_a_former_forms_citation_needs_a_listed_form_with_its_content_address(
    derived: dict[str, Any],
) -> None:
    playbook = copy.deepcopy(derived["playbook"])
    (prior,) = _clause(playbook, _tids(playbook)["term"])["prior_standards"]
    ref = dict(prior["source_ref"])

    with pytest.raises(CitationResolutionError, match="does not list"):
        _resolve_ref(
            playbook, {**ref, "document_id": "prior_template:1999 form"}, "x", derived["root"]
        )
    with pytest.raises(CitationResolutionError, match="pairs document_id"):
        _resolve_ref(playbook, {**ref, "version": "template"}, "x", derived["root"])
    del playbook["baseline"]["prior_templates"][0]["sha256"]
    with pytest.raises(CitationResolutionError, match="no sha256"):
        _resolve_ref(playbook, ref, "x", derived["root"])


def test_validator_rejects_a_former_form_cited_as_todays_template(
    derived: dict[str, Any],
) -> None:
    def prior_ref(playbook: dict[str, Any]) -> dict[str, Any]:
        ref: dict[str, Any] = _clause(playbook, _tids(playbook)["term"])["prior_standards"][0][
            "source_ref"
        ]
        return ref

    assert _messages(derived["playbook"]) == []

    playbook = copy.deepcopy(derived["playbook"])
    prior_ref(playbook).update(document_id="template", version="template")
    assert any("never today's template" in m for m in _messages(playbook))

    playbook = copy.deepcopy(derived["playbook"])
    prior_ref(playbook)["document_id"] = "prior_template:1999 form"
    messages = _messages(playbook)
    assert any("cites former form '1999 form', which is not in" in m for m in messages)

    playbook = copy.deepcopy(derived["playbook"])
    prior_ref(playbook)["version"] = "template"
    assert any("pairs document_id 'prior_template:2018 form'" in m for m in _messages(playbook))

    playbook = copy.deepcopy(derived["playbook"])
    del playbook["baseline"]["prior_templates"][0]["sha256"]
    assert any(
        "whose baseline.prior_templates entry carries no sha256" in m for m in _messages(playbook)
    )


def test_overrides_file_loader_reads_the_new_target(tmp_path: Path) -> None:
    path = tmp_path / "overrides.json"
    path.write_text(
        json.dumps(
            {
                "overrides_version": 1,
                "overrides": [{"target": "why_changed", "id": "t", "value": "x", "basis": "owner"}],
            }
        ),
        encoding="utf-8",
    )
    assert [e.target for e in load_overrides(path)] == ["why_changed"]
    playbook = {"evidence": {"clauses": []}, "agreement_type": {"id": "nda"}}
    with pytest.raises(OverridesError):
        fold_overrides(playbook, load_overrides(path), tmp_path)


def test_observation_citation_import_is_the_one_the_builder_uses() -> None:
    """A guard on the fixtures above: the Observation type they restore is the
    pipeline's own."""
    assert ObservationCitation.__module__ == "playbook_engine.observation_builder"


def test_a_former_clause_no_deal_signed_is_a_review_row_of_its_own(tmp_path: Path) -> None:
    """The Review row of a former form's clause that no deal carries has the
    form as its locator (the page shows its full text from the evidence).

    The state is reached through the real pipeline: the form is supplied, but
    the only deals in the corpus opened on today's standard or on the
    counterparty's own text, so no deal opened on or signed any of the form's
    wording. The judged verdict is stored the way the orchestrator's drain
    stores it, and `project` reads it; nothing in the playbook is hand-edited."""
    _write_corpus(tmp_path, deals=("charlie-systems", "delta-partners"))
    config = _write_config(tmp_path, priors=_PRIORS_2018)
    out = _derive(tmp_path, config)
    playbook = json.loads((out / "playbook.opf.json").read_text(encoding="utf-8"))
    tid = _tids(playbook)["term"]
    # No deal carries the form's term: not as a signed text, not as an opening.
    assert not [p for p in _records(playbook, tid).values() if p["opened_with"] == "prior_standard"]
    subject = next(
        s
        for s in collect_subjects(playbook["evidence"], "nda", "AlphaCorp Holdings, Inc.")
        if ROLE_PRIOR_STANDARD in s.roles and s.payload["taxonomy_id"] == tid
    )
    assert subject.roles == (ROLE_PRIOR_STANDARD,)
    _store_verdict(out, subject.key, subject.part_ids, "less_protective")
    _ok(["project", str(out), "--config", str(config)])
    playbook = json.loads((out / "playbook.opf.json").read_text(encoding="utf-8"))
    assert validate_document(playbook).ok

    clause = _clause(playbook, tid)
    (prior,) = clause["prior_standards"]
    assert prior["text"].strip() == _TERM_OLD and prior["vs_standard"] is not None
    rows = [r for r in build_review_rows(playbook).rows if r["taxonomy_id"] == tid]
    (row,) = [r for r in rows if r["roles"] == [ROLE_PRIOR_STANDARD]]
    assert row["loc"] == {"role": ROLE_PRIOR_STANDARD, "tid": tid, "form": prior["form_label"]}
    assert row["n_deals"] == 0 and row["label"] == prior["vs_standard"]["label"]


# ---------------------------------------------------------------------------
# One deal opens on ONE of our forms, not on whichever form each clause matches
# ---------------------------------------------------------------------------

_TERM_2018 = "The term is one year."
_TERM_2020 = "The term is two years."


def _deal_run(
    rows: list[tuple[str, str, str]],
    priors: dict[str, Any],
    standards: dict[str, list[str]],
) -> list[Observation]:
    """The observations of ONE deal; *rows* are ``(clause type, first-draft
    text, signed text)`` (the same differ rows ``_run`` hand-builds)."""
    return build_all_observations(
        "doc1",
        3,
        "our_paper",
        [(_row(before, after, tid, str(i)), _dr()) for i, (tid, before, after) in enumerate(rows)],
        [],
        has_signed_copy=True,
        ordinal_by_vid=_ORDINALS,
        terminal_clauses=[_node(str(i), after, tid) for i, (tid, _, after) in enumerate(rows)],
        terminal_version_id="signed_final",
        standard_text_by_tid=standards,
        prior_standards_by_tid=priors,
    )


def _forms_by_tid(obs: list[Observation]) -> dict[str, str | None]:
    return {o.taxonomy_id or "?": o.opened_with_form for o in obs if o.outcome == "signed"}


def test_a_deal_opens_on_the_form_most_of_its_clauses_read_as() -> None:
    """The 2018 and 2020 forms share their assignment text and differ on term.
    A deal that opened on the 2020 form (its term is the 2020 term) names the
    2020 form for its assignment too, whatever the config order: the form is
    decided per deal, so the deal does not contradict itself."""
    priors = _former_forms(
        (_FORM_2018, {"assignment": [_PRIOR_2018], "term": [_TERM_2018]}),
        (_FORM_2020, {"assignment": [_PRIOR_2018], "term": [_TERM_2020]}),
    )
    standards = {"assignment": [_CURRENT], "term": ["The term is three years."]}
    obs = _deal_run(
        [("assignment", _PRIOR_2018, _PRIOR_2018), ("term", _TERM_2020, _TERM_2020)],
        priors,
        standards,
    )
    assert _forms_by_tid(obs) == {"assignment": _FORM_2020, "term": _FORM_2020}

    # The 2018 deal of the same two forms stays on the 2018 form throughout.
    obs = _deal_run(
        [("assignment", _PRIOR_2018, _PRIOR_2018), ("term", _TERM_2018, _TERM_2018)],
        priors,
        standards,
    )
    assert _forms_by_tid(obs) == {"assignment": _FORM_2018, "term": _FORM_2018}

    # A clause shared by both forms and no other clause to tell them apart: the
    # config order breaks the tie.
    obs = _deal_run([("assignment", _PRIOR_2018, _PRIOR_2018)], priors, standards)
    assert _forms_by_tid(obs) == {"assignment": _FORM_2018}


_FORM_2016 = "2016 form"
_RECITALS_2016 = "The parties wish to exchange information for a possible transaction."
_RECITALS_2020 = "The parties intend to exchange information about a possible transaction."


def test_a_tie_between_forms_is_broken_by_their_config_order() -> None:
    """Three forms, listed 2016, 2018, 2020, with different clause sets: the 2016 and
    2020 forms both have recitals, the 2018 form has none, and the 2018 and 2020
    forms share their term. A deal whose only matching clause is that shared
    term ties between the 2018 and the 2020 form, and the form listed first
    (2018) takes it, however the clause types happen to be met: the rank is the
    form's real place in the config, not the order its clause types were seen in."""
    priors = _former_forms(
        (_FORM_2016, {"recitals": [_RECITALS_2016]}),
        (_FORM_2018, {"term": [_TERM_2018]}),
        (_FORM_2020, {"recitals": [_RECITALS_2020], "term": [_TERM_2018]}),
    )
    standards = {"recitals": ["The parties will talk."], "term": ["The term is three years."]}

    obs = _deal_run([("term", _TERM_2018, _TERM_2018)], priors, standards)
    assert _forms_by_tid(obs) == {"term": _FORM_2018}

    # Votes still come first: a deal that also reads as the 2020 form's recitals
    # opened on the 2020 form throughout, tie or no tie.
    obs = _deal_run(
        [("term", _TERM_2018, _TERM_2018), ("recitals", _RECITALS_2020, _RECITALS_2020)],
        priors,
        standards,
    )
    assert _forms_by_tid(obs) == {"term": _FORM_2020, "recitals": _FORM_2020}

    # And the 2016 recitals alone name the 2016 form: only it has that text.
    obs = _deal_run([("recitals", _RECITALS_2016, _RECITALS_2016)], priors, standards)
    assert _forms_by_tid(obs) == {"recitals": _FORM_2016}

    # The config decides: list the 2020 form ahead of the 2018 form and it wins the tie.
    reordered = _former_forms(
        (_FORM_2020, {"recitals": [_RECITALS_2020], "term": [_TERM_2018]}),
        (_FORM_2018, {"term": [_TERM_2018]}),
    )
    obs = _deal_run([("term", _TERM_2018, _TERM_2018)], reordered, standards)
    assert _forms_by_tid(obs) == {"term": _FORM_2020}

    # The mapping the builder is handed carries each form's config index.
    assert {label: index for forms in priors.values() for label, _nodes, index in forms} == {
        _FORM_2016: 0,
        _FORM_2018: 1,
        _FORM_2020: 2,
    }


def test_a_forms_in_use_dates_are_the_same_in_every_clause(derived: dict[str, Any]) -> None:
    """The dates belong to the form: the clauses the form changed and the ones it
    left as today's read the same dates, inferred once from every deal that
    opened on the form in any clause."""
    entries = [
        f
        for c in derived["playbook"]["digest"]["clauses"]
        for f in c["standard_history"]["forms"]
        if f["form"] == _FORM_2018_LABEL
    ]
    assert any(f["unchanged"] for f in entries) and any(not f["unchanged"] for f in entries)
    dates = {(f["in_use_from"], f["in_use_until"], f["in_use_basis"]) for f in entries}
    assert dates == {("2018", "2020-06", "mixed")}


# ---------------------------------------------------------------------------
# Agent / LLM segmentation: the former forms go through the same store-backed path
# ---------------------------------------------------------------------------

_AGENT_TAXONOMY = _REPO_ROOT / "spec" / "taxonomy" / "affiliation-agreement.yaml"
_RTF_PROLOGUE = r"{\rtf1\ansi\deff0{\fonttbl{\f0\froman\fcharset0 Times New Roman;}}\f0\fs24 "
_AGENT_NOW = "The service provider shall indemnify the institution against third-party claims."
_AGENT_OLD = "The service provider shall indemnify the institution against direct claims only."
_AGENT_TEXTS = {
    "governing_law": "Governed by the laws of Delaware.",
    "term": "The term is one year with renewal on notice.",
    "insurance": "Liability insurance of one million.",
}
_AGENT_SIGNATURES = (
    r"DocuSign Envelope ID: 7f3a2c9e-1b4d\par "
    r"By: /s/ Dana Reyes\par Name: Dana Reyes\par Title: Counsel\par Date: March 4, 2019\par "
    r"By: /s/ Morgan Ellery\par Name: Morgan Ellery\par Title: Counsel\par Date: March 4, 2019\par "
)


def _agent_setup(tmp_path: Path) -> tuple[Path, Path, Path]:
    """A one-deal corpus (a single signed copy that opened on the former form), a
    current template and a former form, as RTF files whose text the fake
    segmenter below replaces."""
    corpus = tmp_path / "corpus"
    (corpus / "deal-001").mkdir(parents=True)
    (corpus / "deal-001" / "v1.rtf").write_text(
        _RTF_PROLOGUE + r"1. Indemnification\par x\par " + _AGENT_SIGNATURES + "}",
        encoding="utf-8",
    )
    for name in ("template.rtf", "form-old.rtf"):
        (corpus / name).write_text(_RTF_PROLOGUE + r"1. Indemnification\par x\par }", "utf-8")
    config = {
        "agreement_type": {
            "id": "educational-affiliation",
            "name": "Educational Affiliation Agreement",
        },
        "baseline": {
            "template": str(corpus / "template.rtf"),
            "prior_templates": [
                {
                    "path": str(corpus / "form-old.rtf"),
                    "label": _FORM_2018_LABEL,
                    "in_use_from": 2018,
                }
            ],
        },
        "taxonomy": str(_AGENT_TAXONOMY),
        "provenance": {"our_party_aliases": ["Alpha Corp"]},
    }
    config_path = tmp_path / "playbook.config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return corpus, config_path, tmp_path / "out"


def _fake_segmenter(calls: list[str], *, pending_form: bool = False) -> Any:
    def fake(
        path: Path,
        document_id: str,
        version: str,
        taxonomy_ids: list[str],
        segment_fn: Any,
        segmentation_cache: Any = None,
        model: str = "test-model",
        extraction_cache: Any = None,
        refresh_extraction: bool = False,
        extractor: str = "auto",
    ) -> Any:
        calls.append(Path(path).name)
        if pending_form and Path(path).name == "form-old.rtf":
            raise SegmentationQAError("no verdict yet for this file")
        indemnity = _AGENT_NOW if Path(path).name == "template.rtf" else _AGENT_OLD
        specs = [("1", "indemnification", indemnity)] + [
            (str(i), tid, text) for i, (tid, text) in enumerate(_AGENT_TEXTS.items(), start=2)
        ]
        nodes = [
            ClauseNode(clause_path=p, heading=h, text=t, char_span=(0, len(t))) for p, h, t in specs
        ]
        if document_id != "template":
            # The deal is one executed copy: an e-signature certificate and two
            # filled signature blocks close it (unclassified, as a segmenter
            # leaves them).
            signed = (
                "DocuSign Envelope ID: 7f3a2c9e-1b4d\n"
                "By: /s/ Dana Reyes\nName: Dana Reyes\nTitle: Counsel\nDate: March 4, 2019\n"
                "By: /s/ Morgan Ellery\nName: Morgan Ellery\nTitle: Counsel\nDate: March 4, 2019"
            )
            nodes.append(
                ClauseNode(clause_path="5", heading=None, text=signed, char_span=(0, len(signed)))
            )
        tree = ClauseTree(
            document_id=document_id, version=version, source_file=Path(path).name, nodes=nodes
        )
        return tree, {p: h for p, h, _ in specs}, ExtractorLabel("legacy")

    return fake


def _agent_mine(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    pending_form: bool,
    messages: list[str] | None = None,
) -> tuple[list[str], Path, Path]:
    corpus, config_path, out = _agent_setup(tmp_path)
    calls: list[str] = []
    monkeypatch.setattr(
        pipeline, "_llm_segment_file", _fake_segmenter(calls, pending_form=pending_form)
    )
    mine_corpus(
        corpus_dir=corpus,
        config=load_config(config_path),
        taxonomy=load_taxonomy(_AGENT_TAXONOMY),
        out_dir=out,
        use_llm_segmentation=True,
        progress=(messages if messages is not None else []).append,
    )
    return calls, config_path, out


def test_a_former_form_is_segmented_and_classified_on_the_agent_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With agent segmentation the former form is read through the same
    store-backed path as the template; its clauses are classified by the
    verdict's ``taxonomy_by_path``, and a deal that opened on its wording is
    ``opened_with: prior_standard``."""
    calls, config_path, out = _agent_mine(tmp_path, monkeypatch, pending_form=False)
    assert "form-old.rtf" in calls and "template.rtf" in calls
    persisted = json.loads((out / "prior_templates.json").read_text(encoding="utf-8"))
    (form,) = persisted["forms"]
    assert form["label"] == _FORM_2018_LABEL
    by_tid = {o["taxonomy_id"]: o["full_text"] for o in form["observations"]}
    assert by_tid["indemnification"] == _AGENT_OLD
    deal = read_observations_jsonl(out / "observations.jsonl")
    (indemnity,) = [o for o in deal if o["taxonomy_id"] == "indemnification"]
    assert (indemnity["opened_with"], indemnity["opened_with_form"]) == (
        "prior_standard",
        _FORM_2018_LABEL,
    )
    _ok(["project", str(out), "--config", str(config_path)])
    playbook = json.loads((out / "playbook.opf.json").read_text(encoding="utf-8"))
    assert [t["label"] for t in playbook["baseline"]["prior_templates"]] == [_FORM_2018_LABEL]
    assert validate_document(playbook).ok


def test_a_former_form_whose_segmentation_is_pending_is_skipped_with_a_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Segmentation of a former form not yet in the store: mine warns and goes on
    without that form. No entry is written for it, and the playbook does not
    list it (the deal's opening is read against the forms it has)."""
    messages: list[str] = []
    calls, config_path, out = _agent_mine(
        tmp_path, monkeypatch, pending_form=True, messages=messages
    )
    assert "form-old.rtf" in calls
    warnings = [m for m in messages if "WARNING" in m and _FORM_2018_LABEL in m]
    assert len(warnings) == 1 and "segmentation pending" in warnings[0]
    path = out / "prior_templates.json"
    assert not path.is_file() or json.loads(path.read_text(encoding="utf-8"))["forms"] == []
    deal = read_observations_jsonl(out / "observations.jsonl")
    assert all(o.get("opened_with_form") is None for o in deal)
    _ok(["project", str(out), "--config", str(config_path)])
    playbook = json.loads((out / "playbook.opf.json").read_text(encoding="utf-8"))
    assert "prior_templates" not in playbook["baseline"]
    assert validate_document(playbook).ok
