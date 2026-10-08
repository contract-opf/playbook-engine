"""The consumer path through the real pipeline (issue #220).

With no deviation judge configured — ``mine_corpus``'s default — every
deviation is the deterministic standard check: each observation carries a
``standard`` fact (does its text match our template clause?) and a deviation
derived from it, never a judged or stub verdict, and ``project_playbook``
carries it into the precedent record — no stance is ever read out of the
placeholder risk_delta. A deviation judge is opt-in (the advisory layer):
its verdicts never enter ``evidence``.

Driven end-to-end through the real producers (``mine_corpus`` ->
``project_playbook``) over the wholly synthetic ``examples/nda`` corpus with
``config.smoke.yaml`` (deterministic segmentation, no network, no
``ANTHROPIC_API_KEY``). No hand-built observation shapes.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from playbook_engine.clause_differ import ClauseDiff
from playbook_engine.cli import cli
from playbook_engine.config import load_config
from playbook_engine.observation_builder import (
    Observation,
    build_observations,
    read_observations_jsonl,
)
from playbook_engine.pipeline import (
    _assess_deviations_with_standards,
    _NullDeviationJudge,
    mine_corpus,
    project_playbook,
)
from playbook_engine.run_manifest import (
    read_deviation_mode,
    read_run_manifest,
)
from playbook_engine.taxonomy import load_taxonomy
from playbook_engine.tracked_changes_overlay import HunkEnrichment
from playbook_engine.validator import validate_document

_REPO_ROOT = Path(__file__).resolve().parent.parent
_NDA_DIR = _REPO_ROOT / "examples" / "nda"
_CORPUS_DIR = _NDA_DIR / "corpus"
_SMOKE_CONFIG = _NDA_DIR / "config.smoke.yaml"


def _mine_and_project(out_dir: Path, **judges: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Mine + project (OPF 0.4, the one format the engine emits)."""
    cfg = load_config(_SMOKE_CONFIG)
    taxonomy = load_taxonomy(cfg.taxonomy_path)
    mine_corpus(_CORPUS_DIR, cfg, taxonomy, out_dir, no_cache=True, **judges)
    playbook = project_playbook(out_dir, cfg, taxonomy)
    return read_observations_jsonl(out_dir / "observations.jsonl"), playbook


@pytest.fixture(scope="module")
def consumer_run(tmp_path_factory: pytest.TempPathFactory) -> tuple[list[dict], dict]:
    return _mine_and_project(tmp_path_factory.mktemp("consumer"))


def test_default_mine_emits_deterministic_standard_facts(consumer_run: tuple) -> None:
    observations, _ = consumer_run
    assert observations
    for obs in observations:
        assert obs["basis"] == "deterministic", obs["observation_id"]
        assert obs["deviation"] != "needs_review"
        assert isinstance(obs["standard"], bool), obs["observation_id"]
        assert obs["deviation"] == ("none" if obs["standard"] else "substantive")
        assert obs["risk_delta"] == {"direction": "neutral", "magnitude": "none"}
    # Both answers occur on this corpus — the check is not trivially constant.
    assert {obs["standard"] for obs in observations if obs["outcome"] == "signed"} == {
        True,
        False,
    }


def test_identical_signed_text_always_gets_the_same_standard_fact(consumer_run: tuple) -> None:
    """The judged path gave identical signed text conflicting verdicts when it
    was reached from different opening drafts; the standard check reads only
    the signed text, so one (clause, text) pair has one answer."""
    observations, _ = consumer_run
    answers: dict[tuple[str, str], set[bool]] = defaultdict(set)
    deals: dict[tuple[str, str], set[str]] = defaultdict(set)
    for obs in observations:
        if obs["outcome"] != "signed" or obs["taxonomy_id"] is None:
            continue
        key = (obs["taxonomy_id"], " ".join(obs["full_text"].split()).casefold())
        answers[key].add(obs["standard"])
        deals[key].add(obs["citation"]["document_id"])
    assert any(len(ids) > 1 for ids in deals.values()), "corpus must repeat signed text"
    assert all(len(values) == 1 for values in answers.values())


def test_default_project_is_the_verdict_free_precedent(consumer_run: tuple) -> None:
    """Issue #223: the OPF 0.4 projection of the consumer-path store carries
    each terminal row's deterministic standard fact as the precedent's
    ``standard``, counts distinct deals, keeps refused asks, and has no
    stance, risk, deviation or x_judgments anywhere."""
    observations, playbook = consumer_run
    assert playbook["opf_version"] == "0.4"
    assert validate_document(playbook).ok
    precedent = playbook["evidence"]["precedent"]
    by_key = {(p["document_id"], p["taxonomy_id"]): p for p in precedent}
    terminal = [
        o
        for o in observations
        if o["outcome"] in ("signed", "unsigned") and o["taxonomy_id"] is not None
    ]
    assert terminal
    for obs in terminal:
        record = by_key.get((obs["citation"]["document_id"], obs["taxonomy_id"]))
        if record is None:  # a sub-sentence fragment, excluded from precedent
            assert len(obs["full_text"].strip()) < 25
            continue
        assert record["standard"] is obs["standard"]
        assert record["signed"] is (obs["outcome"] == "signed")
    for clause in playbook["evidence"]["clauses"]:
        signed_standard = {
            o["citation"]["document_id"]
            for o in terminal
            if o["taxonomy_id"] == clause["taxonomy_id"]
            and o["outcome"] == "signed"
            and o["standard"]
        }
        assert clause["n_signed_standard"] == len(signed_standard), clause["id"]
    # Refused asks are a deterministic outcome fact and survive.
    assert any(clause["n_refused"] for clause in playbook["evidence"]["clauses"])
    # Struck standard language is our concession: an opening text, no signed text
    # for that clause unless the deal signed replacement text, never a refused ask.
    conceded = [o for o in observations if o["outcome"] == "conceded_before_signing"]
    assert conceded
    for obs in conceded:
        record = by_key[(obs["citation"]["document_id"], obs["taxonomy_id"])]
        assert record["opening_text"]["text"] == obs["full_text"]
        assert obs["full_text"] not in {a["text"] for a in record["refused_asks"]}
    serialized = json.dumps({"evidence": playbook["evidence"], "digest": playbook["digest"]})
    for judged in ("historical_stance", "risk_delta", "deviation", "band", "full_text"):
        assert f'"{judged}"' not in serialized, judged
    assert "x_judgments" not in playbook


def test_opt_in_deviation_judge_never_enters_evidence(tmp_path: Path) -> None:
    """Wiring a deviation judge (here the stub) is the advisory layer: its
    verdicts replace the standard check's deviation in the store, the
    standard fact is still recorded, and the playbook watermarks the
    unjudged rows — but evidence.precedent still carries only the standard
    fact, and a stub verdict is not a judgment (no x_judgments)."""
    observations, playbook = _mine_and_project(tmp_path, deviation_judge=_NullDeviationJudge())
    bases = {obs["basis"] for obs in observations}
    assert "needs_review" in bases
    assert all(isinstance(obs.get("standard"), bool) for obs in observations)
    assert validate_document(playbook).ok
    assert playbook["compiler"]["stub_basis_present"] is True
    assert "x_judgments" not in playbook
    for judged in ("risk_delta", "deviation", "historical_stance"):
        assert f'"{judged}"' not in json.dumps(playbook["evidence"]), judged


# -- the deviation mode is recorded at mine time, not inferred (issue #230) --


def _all_template_corpus(root: Path) -> Path:
    """Two synthetic deals that each signed our template unchanged.

    Each deal is a copy of the NDA example's own ``standard-form.rtf`` with a
    ``hints.yaml`` naming it the executed copy (the production hint
    ``version_orderer.Hints`` reads), so every clause takes the unchanged
    fast path: on the opt-in judged run too, every row carries
    ``basis="deterministic"`` and a computed ``standard`` fact — the store
    shape that used to be indistinguishable from the consumer path.
    """
    corpus = root / "corpus"
    for deal in ("deal-one", "deal-two"):
        (corpus / deal).mkdir(parents=True)
        (corpus / deal / "v1.rtf").write_bytes((_NDA_DIR / "standard-form.rtf").read_bytes())
        (corpus / deal / "hints.yaml").write_text("signed_version: v1\n", encoding="utf-8")
    return corpus


def _mine_all_template(out_dir: Path, corpus: Path, **judges: Any) -> dict[str, Any]:
    cfg = load_config(_SMOKE_CONFIG)
    taxonomy = load_taxonomy(cfg.taxonomy_path)
    mine_corpus(corpus, cfg, taxonomy, out_dir, no_cache=True, **judges)
    return {"cfg": cfg, "taxonomy": taxonomy}


def test_mine_records_the_deviation_mode_whatever_the_rows_look_like(tmp_path: Path) -> None:
    """The #220 review's failure scenario: a judge was configured but every
    clause was unchanged from the template, so the store's rows look exactly
    like the consumer path's. The mode is recorded when they are written
    (issue #230), not inferred from them."""
    corpus = _all_template_corpus(tmp_path)

    judged_out = tmp_path / "judged"
    _mine_all_template(judged_out, corpus, deviation_judge=_NullDeviationJudge())
    rows = read_observations_jsonl(judged_out / "observations.jsonl")
    assert rows
    assert {o["outcome"] for o in rows} == {"signed"}
    assert all(o["basis"] == "deterministic" for o in rows)
    assert read_deviation_mode(judged_out) == "judged"

    consumer_out = tmp_path / "consumer"
    _mine_all_template(consumer_out, corpus)
    assert read_deviation_mode(consumer_out) == "deterministic"


def test_cli_mine_records_the_mode_and_project_succeeds(tmp_path: Path) -> None:
    """Through the CLI: ``mine --with-deviation-judge`` stamps the run
    manifest with its environment AND keeps the recorded mode (the end-of-run
    environment stamp must not erase it), and ``project`` (which has no
    ``--opf-version``: one format, issue #238) emits OPF 0.4."""
    corpus = _all_template_corpus(tmp_path)
    out_dir = tmp_path / "out"
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "mine",
            str(corpus),
            "--config",
            str(_SMOKE_CONFIG),
            "--out",
            str(out_dir),
            "--with-deviation-judge",
        ],
    )
    assert result.exit_code == 0, result.output
    manifest = read_run_manifest(out_dir)
    assert manifest is not None and manifest.written_by == "mine"
    assert manifest.deviation_mode == "judged"

    result = runner.invoke(cli, ["project", str(out_dir), "--config", str(_SMOKE_CONFIG)])
    assert result.exit_code == 0, result.output
    playbook = json.loads((out_dir / "playbook.opf.json").read_text(encoding="utf-8"))
    assert playbook["opf_version"] == "0.4"
    assert validate_document(playbook).ok


# -- dynamics follow the emitted deviation, not the per-row one --------------

_HALF_A = (
    "The Recipient shall hold all Confidential Information in strict confidence "
    "and shall not disclose it to any third party without prior written consent."
)
_HALF_B = (
    "The Recipient shall use the Confidential Information solely to evaluate the "
    "proposed transaction and for no other purpose whatsoever during the term."
)
_TID = "confidentiality_obligations"
_AUTHOR = HunkEnrichment(author="Counterparty Counsel", date="2024-03-15", tracked_type="insertion")


def _rows(texts: list[str]) -> list[ClauseDiff]:
    return [
        ClauseDiff(
            taxonomy_id=_TID,
            clause_path_before=f"{i + 1}",
            clause_path_after=f"{i + 1}",
            kind="modified",
            hunks=(),
            text_before=text,
            text_after=text,
        )
        for i, text in enumerate(texts)
    ]


def _consumer_observations(
    texts: list[str], std_nodes: list[str]
) -> tuple[list[Any], list[Observation]]:
    """The consumer path's own producers: the per-row deterministic check
    (``_assess_deviations_with_standards`` with no judge), then
    ``build_observations`` merging the deal's nodes into one observation."""
    assessed = _assess_deviations_with_standards(
        _rows(texts),
        {_TID: std_nodes[0]},
        None,
        template_std_nodes_by_tid={_TID: std_nodes},
    )
    observations = build_observations(
        "deal-1",
        1,
        "counterparty_paper",
        assessed,
        reversals=[],
        attributions=[_AUTHOR] * len(texts),
        our_party_aliases=["Our Company"],
        standard_text_by_tid={_TID: std_nodes},
        deterministic_deviations=True,
    )
    return assessed, observations


def test_split_standard_is_none_and_carries_no_proposal() -> None:
    """Our one-node standard split across two deal nodes: each row alone is
    "substantive", the merged clause is standard — so no proposal is
    attributed or dated."""
    assessed, observations = _consumer_observations([_HALF_A, _HALF_B], [f"{_HALF_A} {_HALF_B}"])
    assert {dr.deviation for _, dr in assessed} == {"substantive"}
    (obs,) = observations
    assert obs.standard is True
    assert obs.deviation == "none"
    assert obs.proposed_by is None
    assert obs.observed_at is None


def test_partial_multi_node_standard_is_substantive_and_attributed() -> None:
    """A deal that keeps one node of a two-node standard: the row alone
    matches a template node ("none"), the clause is not our standard — so
    the proposal is attributed and dated."""
    assessed, observations = _consumer_observations([_HALF_A], [_HALF_A, _HALF_B])
    assert [dr.deviation for _, dr in assessed] == ["none"]
    (obs,) = observations
    assert obs.standard is False
    assert obs.deviation == "substantive"
    assert obs.proposed_by == "unknown"
    assert obs.observed_at == "2024-03-15"
