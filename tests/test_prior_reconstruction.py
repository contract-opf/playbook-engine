"""Reconstructing our former forms from the corpus (issue #250).

When the owner supplies no older form files, the first drafts we sent are still
in the corpus. A wording that recurs as the first draft of several signed deals
on our paper, and is not today's standard, is rebuilt as a former form of ours
(``basis: reconstructed``) once the ``prior_form_origin`` judge (drafted by an
agent, blind-checked, adjudicated on disagreement) confirms it is OUR language
and not a counterparty's.

Fixture provenance (the producer of every shape used here):

- the pure tests hand-build :class:`FirstDraft` values, the shape
  ``first_drafts`` reads off ``evidence.precedent`` records;
- the end-to-end tests drive the REAL pipeline over a synthetic RTF corpus
  written into ``tmp_path`` (the NDA example's standard form is the current
  template; the former form is that text with three clauses reworded; a
  "counterparty" wording replaces one clause in other deals), through
  ``playbook mine`` -> ``playbook project`` -> ``playbook validate``, so the
  observations, the precedent record, the playbook and the sidecar are what the
  engine writes; the struck-wording tests derive the issue #249 corpus
  (``tests.test_prior_standards._DEALS``) the same way, once with that form
  supplied and once reconstructed, and the emergent test mines with a config
  that has no ``baseline.template`` (``load_config`` reads it as
  ``has_canonical_template: false``);
- the validator tests edit a projected playbook (a refused ask appended in the
  ``{text, round, ref}`` shape precedent writes);
- origin verdicts are the dicts ``playbook judge-apply --verdicts`` accepts
  (``validate_origin_draft`` is the function it runs) stored through
  ``VerdictStore``; check answers are the records ``judge-apply --check``
  accepts, with the pinned checker's model id and effort.

SECURITY NOTE: synthetic text only; fictional parties.
"""

from __future__ import annotations

import copy
import json
import random
import re
from pathlib import Path
from typing import Any

import pytest
import yaml
from click.testing import CliRunner

from playbook_engine.agent_judge import (
    PendingQueue,
    StoreBackedPriorFormOriginJudge,
    VerdictStore,
    validate_verdict,
)
from playbook_engine.cli import cli
from playbook_engine.config import load_config
from playbook_engine.equivalence_check import CHECKER_EFFORT, CHECKER_MODEL
from playbook_engine.observation_builder import Observation, ObservationCitation
from playbook_engine.pipeline import (
    PRIOR_RECONSTRUCTION_FILENAME,
    prior_form_origin_subjects,
    queue_prior_form_origin,
)
from playbook_engine.prior_form_origin_check import (
    apply_check_records,
    build_check_queues,
    load_check_records,
    write_check_queues,
)
from playbook_engine.prior_reconstruction import (
    CLUSTER_SIMILARITY,
    DOMINANT_SHARE,
    MIN_DEALS,
    ORIGIN_KIND,
    STATE_CONFIRMED,
    STATE_PENDING,
    STATE_REJECTED,
    Cluster,
    FirstDraft,
    _without_struck_former_wording,
    cluster_first_drafts,
    first_drafts,
    group_forms,
    outcome_of,
    similarity,
    validate_origin_draft,
)
from playbook_engine.prior_standards import PriorForm
from playbook_engine.rubric import rubric_version
from playbook_engine.scorecard import build_scorecard
from playbook_engine.taxonomy import load_taxonomy
from playbook_engine.validator import validate_document
from tests.test_prior_standards import (
    _CARE_NOW,
    _CARE_OLD,
    _FORM_OLD,
    _PRIORS_2018,
    _SURV_NOW,
    _SURV_OLD,
    _SURV_VARIANT,
    _TAXONOMY,
    _TEMPLATE_RTF,
    _TERM_OLD,
    _TERM_THEIRS,
    _clause,
    _draft,
    _form,
    _records,
    _signed,
    _tids,
    _write_config,
    _write_corpus,
)

# ---------------------------------------------------------------------------
# Pure: similarity, clustering, forms, outcomes
# ---------------------------------------------------------------------------

_BASE = (
    "recipient shall protect confidential information using the same degree of care "
    "it uses for its own confidential information and no less than reasonable care "
    "for a period of five years"
)


def _draft_of(deal: str, key: str, tid: str = "care", signed_on: str | None = None) -> FirstDraft:
    return FirstDraft(
        document_id=deal,
        taxonomy_id=tid,
        text=key,
        key=key,
        ref={"document_id": deal, "version": 1, "clause_path": "6"},
        signed_on=signed_on,
    )


def test_similarity_is_one_for_equal_text_and_high_for_a_near_copy() -> None:
    assert similarity(_BASE, _BASE) == 1.0
    near = _BASE.replace("confidential information", "confidential materials", 1)
    assert 0.5 < similarity(_BASE, near) < 1.0
    assert similarity(_BASE, "an entirely different clause about governing law and venue") < 0.2


def test_a_changed_number_or_a_flipped_negation_is_never_near_identical() -> None:
    assert similarity(_BASE, _BASE.replace("five years", "seven years")) == 0.0
    assert similarity(_BASE, _BASE.replace("shall protect", "shall not protect")) == 0.0


def test_a_recurring_wording_becomes_one_cluster_with_its_medoid() -> None:
    drafts = [
        _draft_of("d1", _BASE),
        _draft_of("d2", _BASE),
        _draft_of("d3", _BASE),
        _draft_of("d4", _BASE + " and"),  # a near copy
    ]
    (cluster,) = cluster_first_drafts(drafts, {})
    assert cluster.n_deals == 4
    assert cluster.medoid.key == _BASE
    assert cluster.n_wordings == 2
    assert cluster.dominant_share == 0.75


def test_fewer_than_min_deals_is_not_a_reconstructed_clause() -> None:
    drafts = [_draft_of(f"d{i}", _BASE) for i in range(MIN_DEALS - 1)]
    assert cluster_first_drafts(drafts, {}) == []
    assert len(cluster_first_drafts([*drafts, _draft_of("dx", _BASE)], {})) == 1


def test_a_cluster_without_a_dominant_wording_is_dropped() -> None:
    # Five deals, five different wordings that are each similar to the seed
    # but none is the first draft of half of them.
    words = [f"{_BASE} extra{suffix}" for suffix in "abcde"]
    drafts = [_draft_of(f"d{i}", w) for i, w in enumerate(words)]
    assert all(similarity(words[0], w) >= CLUSTER_SIMILARITY for w in words[1:])
    assert DOMINANT_SHARE > 1 / 5
    assert cluster_first_drafts(drafts, {}) == []


def test_a_wording_that_reads_as_todays_standard_is_dropped() -> None:
    drafts = [_draft_of(f"d{i}", _BASE) for i in range(4)]
    assert cluster_first_drafts(drafts, {"care": _BASE}) == []


def test_clustering_is_deterministic_whatever_the_input_order() -> None:
    drafts = [_draft_of(f"d{i}", _BASE) for i in range(4)]
    drafts += [_draft_of(f"e{i}", _BASE.replace("five", "ten")) for i in range(3)]
    drafts += [
        _draft_of(f"f{i}", "something else entirely about governing law", "law") for i in range(3)
    ]
    want = cluster_first_drafts(drafts, {})
    assert len(want) == 3
    for seed in range(5):
        shuffled = list(drafts)
        random.Random(seed).shuffle(shuffled)
        got = cluster_first_drafts(shuffled, {})
        assert [(c.taxonomy_id, c.medoid.key, c.deals) for c in got] == [
            (c.taxonomy_id, c.medoid.key, c.deals) for c in want
        ]


def _cluster(tid: str, deals: list[str], dates: list[str | None]) -> Cluster:
    members = tuple(
        _draft_of(d, f"{tid} wording", tid, signed_on=when)
        for d, when in zip(deals, dates, strict=True)
    )
    return Cluster(taxonomy_id=tid, medoid=members[0], members=members, n_wordings=1)


def test_clusters_that_co_occur_in_the_same_deals_are_one_form() -> None:
    deals = ["a", "b", "c", "d"]
    dates = ["2021-03-01", "2021-05-01", "2021-09-01", "2021-11-01"]
    forms = group_forms(
        [
            _cluster("care", deals, dates),
            _cluster("term", deals, dates),
            _cluster("law", deals, dates),
        ]
    )
    assert forms == [[0, 1, 2]]


def test_clusters_of_different_deals_or_eras_stay_per_clause() -> None:
    # Other deals: nothing co-occurs.
    care = _cluster("care", ["a", "b", "c"], ["2019-01-01"] * 3)
    other = _cluster("term", ["x", "y", "z"], ["2019-01-01"] * 3)
    assert group_forms([care, other]) == [[0], [1]]

    # Shared deals are one deal-date each, so clusters that share deals share
    # their dates: the era guard can only act when the shared deals are
    # undated and the rest of each cluster is dated in a different era.
    # Three of each cluster's four deals are shared (and undated), enough to
    # join them; the fourth deal of each is dated.
    shared = ["a", "b", "c"]
    care = _cluster("care", [*shared, "d"], [None, None, None, "2019-01-01"])
    same_era = _cluster("term", [*shared, "e"], [None, None, None, "2019-01-01"])
    later = _cluster("term", [*shared, "e"], [None, None, None, "2023-03-01"])
    assert group_forms([care, same_era]) == [[0, 1]]
    assert group_forms([care, later]) == [[0], [1]]


def test_a_form_never_holds_two_clusters_of_one_clause_type() -> None:
    # Reachable only through a third clause type that overlaps both: the first
    # care cluster joins term, and the second care cluster then cannot.
    care1 = _cluster("care", ["a", "b", "c"], [None] * 3)
    term = _cluster("term", list("abcdef"), [None] * 6)
    care2 = _cluster("care", ["d", "e", "f"], [None] * 3)
    clusters = [care1, term, care2]
    forms = group_forms(clusters)
    assert forms == [[0, 1], [2]]
    for form in forms:
        assert len({clusters[i].taxonomy_id for i in form}) == len(form)


def test_first_drafts_read_only_signed_our_paper_non_standard_openings() -> None:
    def record(**kw: Any) -> dict[str, Any]:
        base = {
            "document_id": "d1",
            "taxonomy_id": "care",
            "paper": "ours",
            "signed": True,
            "signed_on": "2021-03-01",
            "opened_with": "non_standard",
            "opening_text": None,
            "signed_text": {"text": "We shall protect it.", "ref": {"document_id": "d1"}},
        }
        return {**base, **kw}

    records = [
        record(),
        record(document_id="d2", paper="theirs"),
        record(document_id="d3", paper="unknown"),
        record(document_id="d4", signed=False),
        record(document_id="d5", opened_with="standard"),
        record(document_id="d6", opened_with="prior_standard"),
        record(document_id="d7", taxonomy_id="not-ours"),
        record(
            document_id="d8",
            opening_text={"text": "We shall protect it well.", "ref": {"document_id": "d8"}},
        ),
    ]
    got = first_drafts(records, {"care"}, party=None)
    assert [(d.document_id, d.text) for d in got] == [
        ("d1", "We shall protect it."),
        ("d8", "We shall protect it well."),
    ]


@pytest.mark.parametrize(
    ("verdict", "want"),
    [
        (None, STATE_PENDING),
        ({"ours": True, "reason": "x", "basis": "agent"}, STATE_PENDING),
        (
            {
                "ours": True,
                "reason": "x",
                "basis": "agent",
                "check": {"agreed": True, "by": "m", "adjudicated": False},
            },
            STATE_CONFIRMED,
        ),
        (
            {
                "ours": False,
                "reason": "x",
                "basis": "agent",
                "check": {"agreed": True, "by": "m", "adjudicated": False},
            },
            STATE_REJECTED,
        ),
        (
            {
                "ours": True,
                "reason": "x",
                "basis": "agent",
                "check": {"agreed": False, "by": "m", "adjudicated": False},
            },
            STATE_PENDING,
        ),
        (
            {
                "ours": True,
                "reason": "x",
                "basis": "agent",
                "check": {"agreed": False, "by": "m", "adjudicated": True},
            },
            STATE_CONFIRMED,
        ),
        ({"ours": False, "reason": "x", "basis": "owner"}, STATE_REJECTED),
        ({"ours": True, "reason": "x", "basis": "owner"}, STATE_CONFIRMED),
    ],
)
def test_only_a_settled_ours_verdict_confirms_a_cluster(
    verdict: dict[str, Any] | None, want: str
) -> None:
    assert outcome_of(verdict) == want


def test_origin_draft_validation() -> None:
    ok = {"ours": True, "reason": " Reads as the earlier drafting. ", "basis": "agent"}
    assert validate_origin_draft(ok) == {**ok, "reason": "Reads as the earlier drafting."}
    validate_verdict(ORIGIN_KIND, ok)
    for bad in (
        {**ok, "ours": "yes"},
        {**ok, "reason": ""},
        {**ok, "basis": "magic"},
        {**ok, "check": None},
    ):
        with pytest.raises(ValueError):
            validate_origin_draft(bad)


# ---------------------------------------------------------------------------
# End to end: a synthetic corpus through mine -> judge -> project
# ---------------------------------------------------------------------------

#: Our former form (old care, term and survival) was the first draft of four
#: deals; a "counterparty" term recurs as the first draft of three others;
#: today's standard opened one. Every deal is on the template, so every deal
#: reads as our paper.
_OLD_DEALS = {
    "alpha-labs": ("Alpha Labs Inc.", "March 4, 2019", _SURV_OLD),
    "bravo-works": ("Bravo Works LLC", "June 10, 2019", _SURV_VARIANT),
    "foxtrot-co": ("Foxtrot Co.", "August 19, 2019", _SURV_OLD),
    "golf-group": ("Golf Group Ltd.", "November 2, 2019", _SURV_OLD),
}
_THEIR_DEALS = {
    "hotel-inc": ("Hotel Inc.", "January 12, 2021"),
    "india-corp": ("India Corp.", "February 3, 2021"),
    "juliet-llc": ("Juliet LLC", "April 9, 2021"),
}


def _corpus(root: Path) -> Path:
    corpus = root / "corpus"
    deals: dict[str, list[str]] = {}
    for deal, (who, when, signed_surv) in _OLD_DEALS.items():
        deals[deal] = [
            _draft(_FORM_OLD, who),
            _signed(_form(_CARE_OLD, _TERM_OLD, signed_surv), who, when),
        ]
    for deal, (who, when) in _THEIR_DEALS.items():
        theirs = _form(_CARE_NOW, _TERM_THEIRS, _SURV_NOW)
        deals[deal] = [_draft(theirs, who), _signed(theirs, who, when)]
    deals["charlie-systems"] = [
        _draft(_TEMPLATE_RTF, "Charlie Systems Corp."),
        _signed(_TEMPLATE_RTF, "Charlie Systems Corp.", "January 20, 2022"),
    ]
    for deal, versions in deals.items():
        (corpus / deal).mkdir(parents=True)
        for i, text in enumerate(versions, start=1):
            (corpus / deal / f"v{i}.rtf").write_text(text, encoding="utf-8")
    return corpus


def _invoke(args: list[str]) -> Any:
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == 0, f"{' '.join(args)} failed:\n{result.output}"
    return result


@pytest.fixture(scope="module")
def mined(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """One mined out-dir (no form supplied) and one mined WITH a supplied form."""
    root = tmp_path_factory.mktemp("prior-reconstruction")
    _corpus(root)
    config = _write_config(root, priors=None)
    _invoke(["mine", str(root / "corpus"), "--config", str(config), "--out", str(root / "out")])
    supplied = _write_config(
        root,
        priors=[{"path": "form-2018.rtf", "label": "2018 form", "in_use_from": 2018}],
        name="config.supplied.yaml",
    )
    _invoke(
        [
            "mine",
            str(root / "corpus"),
            "--config",
            str(supplied),
            "--out",
            str(root / "out-supplied"),
        ]
    )
    return {"root": root, "config": config, "supplied": supplied, "out": root / "out"}


@pytest.fixture
def out(mined: dict[str, Any], tmp_path: Path) -> Path:
    """A fresh copy of the mined out-dir: each test writes its own verdicts."""
    import shutil

    dest = tmp_path / "out"
    shutil.copytree(mined["out"], dest)
    return dest


def _subjects(out: Path, mined: dict[str, Any]) -> list[Any]:
    cfg = load_config(mined["config"])
    return prior_form_origin_subjects(out, cfg, load_taxonomy(_TAXONOMY))


def _ours(subject: Any) -> bool:
    """Our former form (the old wording) is ours; the recurring term is a counterparty's."""
    return "ten (10) days" not in subject.payload["candidate"] and (
        "six (6) months" not in subject.payload["candidate"]
    )


def _draft_all(out: Path, subjects: list[Any], *, basis: str = "agent") -> None:
    store = VerdictStore(out / "judge" / "verdicts.jsonl")
    stamp = None
    for s in subjects:
        store.put_by_key(
            s.key,
            validate_origin_draft(
                {"ours": _ours(s), "reason": "Synthetic reading of the wording.", "basis": basis}
            ),
            rubric=stamp,
        )


def _check_all(out: Path, subjects: list[Any], *, ours: dict[str, bool] | None = None) -> Any:
    """Record the pinned checker's answers (agreeing unless *ours* overrides a key)."""
    store = VerdictStore(out / "judge" / "verdicts.jsonl")
    records = [
        (
            i,
            {
                "key": s.key,
                "ours": (ours or {}).get(s.key, _ours(s)),
                "reason": "Synthetic blind check.",
                "model": CHECKER_MODEL,
                "effort": CHECKER_EFFORT,
            },
        )
        for i, s in enumerate(subjects, start=1)
    ]
    result = apply_check_records(store, records)
    for key, verdict in result.updates:
        store.put_by_key(key, verdict)
    return result


def _project(out: Path, config: Path) -> dict[str, Any]:
    _invoke(["project", str(out), "--config", str(config)])
    return json.loads((out / "playbook.opf.json").read_text(encoding="utf-8"))


def _counts(out: Path) -> dict[str, Any]:
    return json.loads((out / PRIOR_RECONSTRUCTION_FILENAME).read_text(encoding="utf-8"))


def _normalised(playbook: dict[str, Any]) -> dict[str, Any]:
    clean = copy.deepcopy(playbook)
    clean.get("compiler", {}).pop("generated_at", None)
    return clean


def test_recurring_our_paper_first_drafts_are_the_questions(
    mined: dict[str, Any], out: Path
) -> None:
    subjects = _subjects(out, mined)
    by_candidate = {s.payload["candidate"]: s for s in subjects}
    # Three clauses of the former form (care, term, survival), one recurring
    # counterparty term.
    assert len(subjects) == 4
    recurs = sorted(s.payload["recurs"]["n_deals"] for s in subjects)
    assert recurs == [3, 4, 4, 4]
    assert any("six (6) months" in text for text in by_candidate)
    # Nothing that identifies a deal, a counterparty or a paper side is asked.
    blob = json.dumps([s.payload for s in subjects])
    for forbidden in ("our_paper", "counterparty_paper", "alpha-labs", "Alpha Labs", "document_id"):
        assert forbidden not in blob
    assert all(s.payload["stage"] == ORIGIN_KIND for s in subjects)


def test_without_a_confirming_verdict_nothing_changes(mined: dict[str, Any], out: Path) -> None:
    baseline = _project(out, mined["config"])
    assert baseline["baseline"].get("prior_templates") is None
    counts = _counts(out)
    assert counts["clusters"] == 4
    assert counts["pending"] == 4
    assert counts["pending_undrafted"] == 4
    assert counts["confirmed"] == counts["rejected"] == counts["forms"] == 0
    assert all(p.get("opened_with_form") is None for p in baseline["evidence"]["precedent"])

    # A draft nobody has blind-checked does not count either.
    subjects = _subjects(out, mined)
    _draft_all(out, subjects)
    drafted = _project(out, mined["config"])
    assert _normalised(drafted) == _normalised(baseline)
    counts = _counts(out)
    assert counts["pending_unchecked"] == 4 and counts["confirmed"] == 0


def test_a_confirmed_cluster_becomes_a_reconstructed_prior_form(
    mined: dict[str, Any], out: Path
) -> None:
    baseline = _project(out, mined["config"])
    subjects = _subjects(out, mined)
    _draft_all(out, subjects)
    result = _check_all(out, subjects)
    assert (result.agreed, result.disagreed) == (4, 0)
    playbook = _project(out, mined["config"])

    counts = _counts(out)
    assert (counts["confirmed"], counts["rejected"], counts["pending"]) == (3, 1, 0)
    # The three clauses of the old form co-occur in the same four deals: one form.
    assert counts["forms"] == 1 and counts["forms_multi_clause"] == 1
    assert counts["deals_in_forms"] == 4

    (form,) = playbook["baseline"]["prior_templates"]
    assert form["basis"] == "reconstructed"
    assert form["n_deals"] == 4
    assert form["first_signed"] == "2019-03" and form["last_signed"] == "2019-11"
    assert "sha256" not in form and "in_use_from" not in form
    assert form["label"].startswith("reconstructed 2019-03 to 2019-11")

    # The deals that opened on it are recorded as ours; the counterparty term
    # that recurs is not.
    opened = [p for p in playbook["evidence"]["precedent"] if p.get("opened_with_form")]
    assert {p["opened_with_form"] for p in opened} == {form["label"]}
    assert {p["document_id"] for p in opened} == set(_OLD_DEALS)
    assert all(p["opened_with"] == "prior_standard" for p in opened)
    changed = sum(c.get("n_opened_prior", 0) for c in playbook["evidence"]["clauses"])
    assert changed == len(opened)
    base_non_standard = sum(
        1 for p in baseline["evidence"]["precedent"] if p.get("opened_with") == "non_standard"
    )
    now_non_standard = sum(
        1 for p in playbook["evidence"]["precedent"] if p.get("opened_with") == "non_standard"
    )
    assert base_non_standard - now_non_standard == len(opened)

    # Each reconstructed clause is a prior_standards entry citing the corpus.
    entries = [
        (c["taxonomy_id"], e) for c in playbook["evidence"]["clauses"] for e in c["prior_standards"]
    ]
    assert len(entries) == 3
    docs = {d["document_id"] for d in playbook["corpus"]["documents"]}
    for _tid, entry in entries:
        assert entry["form_label"] == form["label"]
        assert entry["source_ref"]["document_id"] in docs
        assert isinstance(entry["source_ref"]["version"], int)

    result = validate_document(playbook)
    assert result.ok, [str(e) for e in result.errors]

    digest_forms = [
        f for c in playbook["digest"]["clauses"] for f in c["standard_history"]["forms"]
    ]
    assert digest_forms and {f["basis"] for f in digest_forms} == {"reconstructed"}


@pytest.mark.parametrize(
    "paper_edit",
    [{"provenance": "counterparty_paper"}, {"provenance_is_ambiguous": True}],
    ids=["their-paper", "unknown-paper"],
)
def test_a_deal_on_any_paper_that_opened_on_the_form_reads_prior_standard(
    mined: dict[str, Any], out: Path, paper_edit: dict[str, Any]
) -> None:
    # One deal whose first draft is exactly the former form's wording is not on
    # our paper (as detected). Paper side only chose the candidates the form
    # was built from (the other three), so it still opened on the form, just
    # as it would for a supplied form with the same text.
    manifest_path = out / "corpus_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for doc in manifest:
        if doc["document_id"] == "golf-group":
            doc.update(paper_edit)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    subjects = _subjects(out, mined)
    assert sorted(s.payload["recurs"]["n_deals"] for s in subjects) == [3, 3, 3, 3]
    _draft_all(out, subjects)
    _check_all(out, subjects)
    playbook = _project(out, mined["config"])

    (form,) = playbook["baseline"]["prior_templates"]
    assert form["n_deals"] == 4
    golf = [p for p in playbook["evidence"]["precedent"] if p["document_id"] == "golf-group"]
    assert golf and all(p["paper"] != "ours" for p in golf)
    reformed = [p for p in golf if p["opened_with"] == "prior_standard"]
    assert len(reformed) == 3
    assert {p["opened_with_form"] for p in reformed} == {form["label"]}
    assert _counts(out)["deals_in_forms"] == 4
    result = validate_document(playbook)
    assert result.ok, [str(e) for e in result.errors]


def test_a_rejected_cluster_changes_nothing(mined: dict[str, Any], out: Path) -> None:
    subjects = _subjects(out, mined)
    _draft_all(out, subjects)
    _check_all(out, subjects)
    playbook = _project(out, mined["config"])
    theirs = [p for p in playbook["evidence"]["precedent"] if p["document_id"].startswith("hotel")]
    assert any(p["opened_with"] == "non_standard" for p in theirs)
    assert all(p.get("opened_with_form") is None for p in theirs)


def test_a_disputed_check_stays_pending_until_adjudicated(mined: dict[str, Any], out: Path) -> None:
    subjects = _subjects(out, mined)
    _draft_all(out, subjects)
    old = next(s for s in subjects if _ours(s))
    result = _check_all(out, subjects, ours={old.key: False})
    assert (result.agreed, result.disagreed) == (3, 1)
    _project(out, mined["config"])
    counts = _counts(out)
    assert (counts["confirmed"], counts["rejected"], counts["pending"]) == (2, 1, 1)
    assert counts["forms"] == 1

    store = VerdictStore(out / "judge" / "verdicts.jsonl")
    queues = build_check_queues(store, subjects)
    assert len(queues.adjudication) == 1 and not queues.check
    item = queues.adjudication[0]
    assert item["payload"]["draft"]["ours"] is True and item["payload"]["check"]["ours"] is False

    adjudication = apply_check_records(
        store,
        [
            (
                1,
                {
                    "key": old.key,
                    "ours": True,
                    "reason": "Adjudicated: earlier drafting of ours.",
                    "model": CHECKER_MODEL,
                    "effort": CHECKER_EFFORT,
                },
            )
        ],
    )
    assert adjudication.adjudicated == 1
    for key, verdict in adjudication.updates:
        store.put_by_key(key, verdict)
    _project(out, mined["config"])
    assert _counts(out)["confirmed"] == 3


def test_the_checker_must_be_the_pinned_model(mined: dict[str, Any], out: Path) -> None:
    subjects = _subjects(out, mined)
    _draft_all(out, subjects)
    store = VerdictStore(out / "judge" / "verdicts.jsonl")
    record = {
        "key": subjects[0].key,
        "ours": True,
        "reason": "x",
        "model": "some-other-model",
        "effort": CHECKER_EFFORT,
    }
    with pytest.raises(ValueError, match="claude-opus-5-5"):
        apply_check_records(store, [(1, record)])
    assert apply_check_records(store, [(1, record)], allow_checker_model=True).updates


def test_a_supplied_form_suppresses_reconstruction(mined: dict[str, Any]) -> None:
    out = mined["root"] / "out-supplied"
    cfg = load_config(mined["supplied"])
    assert prior_form_origin_subjects(out, cfg, load_taxonomy(_TAXONOMY)) == []
    playbook = _project(out, mined["supplied"])
    assert _counts(out)["skipped"] == "forms_supplied"
    assert [t["label"] for t in playbook["baseline"]["prior_templates"]] == ["2018 form"]
    assert all(t.get("basis") != "reconstructed" for t in playbook["baseline"]["prior_templates"])


def test_an_emergent_playbook_reconstructs_nothing_and_queues_nothing(tmp_path: Path) -> None:
    # No canonical template: no clause has an our_standard for a first draft
    # to differ from, so nothing is asked, queued or reconstructed.
    _corpus(tmp_path)
    config = _write_config(tmp_path, priors=None, name="config.emergent.yaml")
    raw = yaml.safe_load(config.read_text(encoding="utf-8"))
    del raw["baseline"]["template"]
    config.write_text(yaml.safe_dump(raw), encoding="utf-8")
    cfg = load_config(config)
    assert cfg.baseline.has_canonical_template is False
    out = tmp_path / "out"
    _invoke(["mine", str(tmp_path / "corpus"), "--config", str(config), "--out", str(out)])
    taxonomy = load_taxonomy(_TAXONOMY)

    assert prior_form_origin_subjects(out, cfg, taxonomy) == []
    pending = out / "judge" / "pending.jsonl"
    judge = StoreBackedPriorFormOriginJudge(
        store=VerdictStore(out / "judge" / "verdicts.jsonl"), pending=PendingQueue(pending)
    )
    assert queue_prior_form_origin(out, cfg, taxonomy, judge) == (0, 0)
    queued = pending.read_text(encoding="utf-8").splitlines() if pending.is_file() else []
    assert all(json.loads(line)["kind"] != ORIGIN_KIND for line in queued)

    playbook = _project(out, config)
    assert playbook["baseline"]["has_canonical_template"] is False
    assert playbook["baseline"].get("prior_templates") is None
    counts = _counts(out)
    assert counts["skipped"] == "no_canonical_template"
    assert counts["clusters"] == counts["forms"] == 0
    assert build_scorecard(out)["prior_reconstruction"]["skipped"] == "no_canonical_template"

    result = _invoke(
        [
            "judge",
            str(out),
            "--config",
            str(config),
            "--out",
            str(out),
            "--check",
            "prior_form_origin",
        ]
    )
    assert "no canonical template configured" in result.output
    assert not (out / "judge" / "origin-check-pending.jsonl").exists()


def test_reconstruction_is_deterministic_across_runs(
    mined: dict[str, Any], out: Path, tmp_path: Path
) -> None:
    subjects = _subjects(out, mined)
    _draft_all(out, subjects)
    _check_all(out, subjects)
    first = _project(out, mined["config"])
    again = _project(out, mined["config"])
    assert _normalised(first) == _normalised(again)

    # A second, independent mine of the same corpus asks the same questions.
    other = tmp_path / "out2"
    _invoke(
        [
            "mine",
            str(mined["root"] / "corpus"),
            "--config",
            str(mined["config"]),
            "--out",
            str(other),
        ]
    )
    again_subjects = _subjects(other, mined)
    assert [s.key for s in again_subjects] == [s.key for s in subjects]
    assert [s.payload for s in again_subjects] == [s.payload for s in subjects]


def test_the_judge_queues_the_questions_and_replays_the_verdicts(
    mined: dict[str, Any], out: Path
) -> None:
    cfg = load_config(mined["config"])
    taxonomy = load_taxonomy(_TAXONOMY)
    store = VerdictStore(out / "judge" / "verdicts.jsonl")
    pending = PendingQueue(out / "judge" / "pending.jsonl")
    judge = StoreBackedPriorFormOriginJudge(store=store, pending=pending)
    assert queue_prior_form_origin(out, cfg, taxonomy, judge) == (4, 4)
    queued = [
        json.loads(line)
        for line in (out / "judge" / "pending.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert {q["kind"] for q in queued} == {ORIGIN_KIND}
    assert {q["rubric_version"] for q in queued} == {rubric_version(ORIGIN_KIND)}

    _draft_all(out, _subjects(out, mined))
    replaying = StoreBackedPriorFormOriginJudge(
        store=VerdictStore(out / "judge" / "verdicts.jsonl")
    )
    assert queue_prior_form_origin(out, cfg, taxonomy, replaying) == (4, 0)


def test_judge_apply_and_the_check_queue_round_trip_through_the_cli(
    mined: dict[str, Any], out: Path
) -> None:
    subjects = _subjects(out, mined)
    verdicts = out / "verdicts.jsonl"
    verdicts.write_text(
        "".join(
            json.dumps(
                {
                    "key": s.key,
                    "verdict": {"ours": _ours(s), "reason": "Synthetic.", "basis": "agent"},
                }
            )
            + "\n"
            for s in subjects
        ),
        encoding="utf-8",
    )
    _invoke(["judge-apply", str(out), "--verdicts", str(verdicts)])

    result = _invoke(
        [
            "judge",
            str(out),
            "--config",
            str(mined["config"]),
            "--out",
            str(out),
            "--check",
            "prior_form_origin",
        ]
    )
    assert "4 to check, 0 to adjudicate" in result.output
    check_items = [
        json.loads(line)
        for line in (out / "judge" / "origin-check-pending.jsonl").read_text().splitlines()
    ]
    assert len(check_items) == 4
    # A blind check never carries the draft.
    assert all(
        "draft" not in item["payload"] and "ours" not in item["payload"] for item in check_items
    )

    answers = out / "answers.jsonl"
    answers.write_text(
        "".join(
            json.dumps(
                {
                    "key": item["key"],
                    "ours": _ours(next(s for s in subjects if s.key == item["key"])),
                    "reason": "Synthetic blind check.",
                    "model": CHECKER_MODEL,
                    "effort": CHECKER_EFFORT,
                }
            )
            + "\n"
            for item in check_items
        ),
        encoding="utf-8",
    )
    applied = _invoke(["judge-apply", str(out), "--check", str(answers)])
    assert "agreed 4, disagreed 0" in applied.output
    _project(out, mined["config"])
    assert _counts(out)["confirmed"] == 3


def test_check_files_load_and_reject_malformed_records(tmp_path: Path) -> None:
    path = tmp_path / "answers.jsonl"
    path.write_text('{"key": "k", "ours": true}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="line 1"):
        load_check_records(path)
    path.write_text("not json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON"):
        load_check_records(path)
    queues = build_check_queues(VerdictStore(tmp_path / "v.jsonl"), [])
    check, adjudication = write_check_queues(tmp_path, queues)
    assert check.read_text() == "" and adjudication.read_text() == ""


def test_the_scorecard_counts_the_reconstruction_and_nothing_else(
    mined: dict[str, Any], out: Path
) -> None:
    subjects = _subjects(out, mined)
    _draft_all(out, subjects)
    _check_all(out, subjects)
    _project(out, mined["config"])
    card = build_scorecard(out)
    section = card["prior_reconstruction"]
    assert (section["confirmed"], section["rejected"], section["pending"]) == (3, 1, 0)
    assert section["forms"] == 1 and section["skipped"] is None
    assert card["prior_standards"]["forms_reconstructed"] == 1
    assert card["prior_standards"]["forms_supplied"] == 0
    blob = json.dumps(card)
    # Counts only: no date, label or wording leaves the out-dir.
    assert re.search(r"\d{4}-\d{2}", blob) is None
    assert "reconstructed 20" not in blob and "ten (10) days" not in blob


# ---------------------------------------------------------------------------
# Our former wording struck before signing is no refused ask (as for a
# supplied form)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def struck(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """The issue #249 corpus derived twice: with the 2018 form SUPPLIED, and with
    no form supplied and every origin cluster confirmed as ours (drafted and
    blind-checked). Bravo opened on the former form and struck its term for
    today's standard before signing."""
    root = tmp_path_factory.mktemp("struck-former-wording")
    _write_corpus(root)
    none_cfg = _write_config(root, priors=None)
    supplied_cfg = _write_config(root, priors=_PRIORS_2018, name="config.supplied.yaml")
    out, out_supplied = root / "out", root / "out-supplied"
    for cfg_path, dest in ((none_cfg, out), (supplied_cfg, out_supplied)):
        _invoke(["mine", str(root / "corpus"), "--config", str(cfg_path), "--out", str(dest)])
    subjects = prior_form_origin_subjects(out, load_config(none_cfg), load_taxonomy(_TAXONOMY))
    # The former form's care, term and survival each opened alpha, bravo and echo.
    assert len(subjects) == 3 and all(_ours(s) for s in subjects)
    _draft_all(out, subjects)
    assert _check_all(out, subjects).agreed == 3
    return {
        "reconstructed": _project(out, none_cfg),
        "supplied": _project(out_supplied, supplied_cfg),
        "out": out,
    }


def test_our_reconstructed_wording_struck_before_signing_is_no_refused_ask(
    struck: dict[str, Any],
) -> None:
    playbook = struck["reconstructed"]
    (form,) = playbook["baseline"]["prior_templates"]
    assert form["basis"] == "reconstructed"
    term = _tids(playbook)["term"]
    bravo = _records(playbook, term)["bravo-works"]
    assert (bravo["opened_with"], bravo["opened_with_form"]) == ("prior_standard", form["label"])
    assert bravo["standard"] is True and bravo["opening_text"]["text"].strip() == _TERM_OLD
    # The term bravo struck is our own former wording, not their refused ask...
    assert bravo["refused_asks"] == []
    assert _clause(playbook, term)["n_refused"] == 0
    digest = {c["taxonomy_id"]: c for c in playbook["digest"]["clauses"]}
    assert digest[term]["refused_asks"] == []
    # ...and it is not lost silently: it is counted, by name.
    dropped = playbook["corpus"]["stats"]["dropped_observations"]
    assert dropped["by_reason"] == {"removed_prior_standard": 1}
    assert dropped["by_document"] == {"bravo-works": 1}
    assert validate_document(playbook).ok, [str(e) for e in validate_document(playbook).errors]


def _row(
    deal: str, text: str, outcome: str, *, opened_with: str | None = None, standard: bool = False
) -> Observation:
    """A deal row in the shape ``build_observations`` writes: a removed row
    (``proposed_then_reversed``, no ``opened_with``), a reversal row (it carries
    the clause's ``opened_with``) or a terminal row."""
    return Observation(
        observation_id=f"{deal}/2/9",
        taxonomy_id="term",
        text_summary=text[:300],
        full_text=text,
        citation=ObservationCitation(document_id=deal, version=1, clause_path="9", char_span=None),
        deviation="none" if standard else "substantive",
        risk_delta={"direction": "neutral", "magnitude": "none"},
        provenance="our_paper",
        outcome=outcome,
        basis="deterministic",
        standard=standard,
        opened_with=opened_with,
    )


def test_every_row_of_our_struck_former_wording_is_dropped_and_counted() -> None:
    """The removed-row rule of a supplied form, applied to a reconstructed one:
    whole or a contiguous piece, under the mining names or the grouping key (a
    stored text carries each deal's own counterparty alias); a reversal of it
    too; their refused ask and every non-refused row are kept."""
    first, second = (s.strip() + "." for s in _TERM_OLD.rstrip(".").split(",", 1))
    form_text = f"{_TERM_OLD} Notices go to Counterparty-1."
    form = PriorForm(
        label="reconstructed 2019-03 to 2019-11",
        observations=(
            Observation(
                observation_id="reconstructed/term",
                taxonomy_id="term",
                text_summary=form_text,
                full_text=form_text,
                citation=ObservationCitation(
                    document_id="alpha-labs", version=1, clause_path="9", char_span=None
                ),
                deviation="substantive",
                risk_delta={"direction": "neutral", "magnitude": "none"},
                provenance="our_paper",
                outcome="signed",
            ),
        ),
        basis="reconstructed",
        n_deals=3,
    )
    rows = [
        _row("bravo", _TERM_OLD, "proposed_then_reversed"),  # a piece, removed
        _row("bravo", f"{_TERM_OLD} Notices go to Counterparty-4.", "proposed_then_reversed"),
        _row("foxtrot", first, "proposed_then_reversed", opened_with="non_standard"),  # reversal
        _row("delta", _TERM_THEIRS, "proposed_then_reversed"),  # their ask: kept
        _row("echo", _TERM_OLD, "signed", opened_with="prior_standard"),  # signed: kept
        _row("golf", second, "conceded_before_signing", standard=True),  # not a refused row
    ]
    kept, removed = _without_struck_former_wording(
        rows, [form], party="AlphaCorp Holdings, Inc.", party_names=["AlphaCorp"]
    )
    assert [(o.citation.document_id, o.outcome) for o in kept] == [
        ("delta", "proposed_then_reversed"),
        ("echo", "signed"),
        ("golf", "conceded_before_signing"),
    ]
    assert removed == {"bravo": 2, "foxtrot": 1}
    # Nothing to drop without a form.
    assert _without_struck_former_wording(rows, [], party=None, party_names=()) == (rows, {})


def test_a_reconstructed_form_feeds_the_counts_exactly_as_the_same_form_supplied(
    struck: dict[str, Any],
) -> None:
    """Every #249 effect of a former form, opened_with and the struck-wording
    drop alike, is the same whether the form was supplied or reconstructed."""
    rec, sup = struck["reconstructed"], struck["supplied"]
    assert rec["corpus"]["stats"].get("dropped_observations") == sup["corpus"]["stats"].get(
        "dropped_observations"
    )
    rec_digest = {c["taxonomy_id"]: c for c in rec["digest"]["clauses"]}
    sup_digest = {c["taxonomy_id"]: c for c in sup["digest"]["clauses"]}
    assert _tids(rec) == _tids(sup)
    for name, tid in _tids(rec).items():
        for key in (
            "n_deals",
            "n_variants",
            "n_refused",
            "n_changed_openings",
            "n_opened_standard",
            "n_kept_standard",
            "n_opened_prior",
            "n_kept_prior",
            "n_moved_to_current",
        ):
            assert _clause(rec, tid)[key] == _clause(sup, tid)[key], (name, key)
        for key in ("refused_asks", "changed_openings"):
            assert rec_digest[tid][key] == sup_digest[tid][key], (name, key)
        rec_records, sup_records = _records(rec, tid), _records(sup, tid)
        assert set(rec_records) == set(sup_records), name
        for deal, record in rec_records.items():
            other = sup_records[deal]
            for key in ("opened_with", "opening_text", "refused_asks", "standard", "moved"):
                assert record[key] == other[key], (name, deal, key)
            assert ("opened_with_form" in record) == ("opened_with_form" in other), (name, deal)


# ---------------------------------------------------------------------------
# The validator rejects a reconstructed form that breaks its MUSTs
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def reconstructed(
    mined: dict[str, Any], tmp_path_factory: pytest.TempPathFactory
) -> dict[str, Any]:
    """The playbook of the confirmed reconstruction (one reconstructed form)."""
    import shutil

    dest = tmp_path_factory.mktemp("reconstructed") / "out"
    shutil.copytree(mined["out"], dest)
    subjects = _subjects(dest, mined)
    _draft_all(dest, subjects)
    _check_all(dest, subjects)
    return _project(dest, mined["config"])


def _errors(playbook: dict[str, Any]) -> list[str]:
    return [str(e) for e in validate_document(playbook).errors if e.blocking]


def test_the_reconstructed_playbook_validates(reconstructed: dict[str, Any]) -> None:
    assert _errors(reconstructed) == []


def test_validator_rejects_a_reconstructed_form_with_a_file_or_owner_dates(
    reconstructed: dict[str, Any],
) -> None:
    for stray, value in (
        ("sha256", "sha256:" + "0" * 64),
        ("in_use_from", "2019"),
        ("in_use_until", "2020"),
    ):
        playbook = copy.deepcopy(reconstructed)
        playbook["baseline"]["prior_templates"][0][stray] = value
        assert any(
            f"carries {stray}, which only a supplied form has" in m for m in _errors(playbook)
        )


def test_validator_rejects_a_supplied_form_with_reconstructed_fields(
    reconstructed: dict[str, Any],
) -> None:
    for stray, value in (("n_deals", 3), ("first_signed", "2019-03"), ("last_signed", "2019-11")):
        playbook = copy.deepcopy(reconstructed)
        form = playbook["baseline"]["prior_templates"][0]
        form["basis"] = "supplied"
        form[stray] = value
        assert any(
            f"supplied form {form['label']!r} carries {stray}" in m for m in _errors(playbook)
        )


def test_validator_rejects_a_reconstructed_source_ref_into_a_form_file(
    reconstructed: dict[str, Any],
) -> None:
    label = reconstructed["baseline"]["prior_templates"][0]["label"]
    for document_id, version in (
        (f"prior_template:{label}", 1),
        ("template", 1),
        ("anything", "prior_template"),
    ):
        playbook = copy.deepcopy(reconstructed)
        clause = next(c for c in playbook["evidence"]["clauses"] if c["prior_standards"])
        ref = clause["prior_standards"][0]["source_ref"]
        ref["document_id"], ref["version"] = document_id, version
        assert any("a reconstructed form has no file of its own" in m for m in _errors(playbook)), (
            document_id,
            version,
        )


def test_validator_checks_n_deals_and_the_date_span_against_the_precedent(
    reconstructed: dict[str, Any],
) -> None:
    playbook = copy.deepcopy(reconstructed)
    playbook["baseline"]["prior_templates"][0]["n_deals"] += 1
    assert any("says n_deals" in m and "distinct deal(s) opened" in m for m in _errors(playbook))

    for name, wrong in (("first_signed", "2018-01"), ("last_signed", "2020-12")):
        playbook = copy.deepcopy(reconstructed)
        playbook["baseline"]["prior_templates"][0][name] = wrong
        assert any(f"says {name} {wrong!r}" in m for m in _errors(playbook)), name


def test_validator_rejects_supplied_and_reconstructed_forms_together(
    reconstructed: dict[str, Any],
) -> None:
    playbook = copy.deepcopy(reconstructed)
    playbook["baseline"]["prior_templates"].append(
        {"label": "2018 form", "in_use_from": "2018", "sha256": "sha256:" + "0" * 64}
    )
    assert any("never appear together" in m for m in _errors(playbook))


def test_validator_rejects_a_refused_ask_that_is_a_reconstructed_clause(
    reconstructed: dict[str, Any],
) -> None:
    """Our own former wording struck before signing is no refused ask: no
    refused ask may have the grouping key of a reconstructed form's clause."""
    label = reconstructed["baseline"]["prior_templates"][0]["label"]
    playbook = copy.deepcopy(reconstructed)
    clause = next(c for c in playbook["evidence"]["clauses"] if c["prior_standards"])
    prior = clause["prior_standards"][0]
    record = next(
        p
        for p in playbook["evidence"]["precedent"]
        if p["taxonomy_id"] == clause["taxonomy_id"] and p["signed"]
    )
    # A case and punctuation respelling still has the clause's grouping key.
    record["refused_asks"].append(
        {"text": prior["text"].upper().rstrip("."), "round": 1, "ref": prior["source_ref"]}
    )
    assert any(f"refused ask reads as reconstructed form {label!r}" in m for m in _errors(playbook))

    # A refused ask of other wording, or the same text under a SUPPLIED form,
    # is not this check's business.
    other = copy.deepcopy(reconstructed)
    other_record = next(
        p
        for p in other["evidence"]["precedent"]
        if p["taxonomy_id"] == clause["taxonomy_id"] and p["signed"]
    )
    other_record["refused_asks"].append(
        {"text": _TERM_THEIRS, "round": 1, "ref": prior["source_ref"]}
    )
    assert not any("reads as reconstructed form" in m for m in _errors(other))
    supplied = copy.deepcopy(playbook)
    supplied["baseline"]["prior_templates"][0]["basis"] = "supplied"
    assert not any("reads as reconstructed form" in m for m in _errors(supplied))
