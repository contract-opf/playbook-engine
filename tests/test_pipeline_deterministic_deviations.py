"""The consumer path through the real pipeline (issue #220).

With no deviation judge configured — ``mine_corpus``'s default — every
deviation is the deterministic standard check: each observation carries a
``standard`` fact (does its text match our template clause?) and a deviation
derived from it, never a judged or stub verdict, and ``project_playbook``
reads no stance out of the placeholder risk_delta. A deviation judge is
opt-in (the advisory layer), and wiring one keeps the judged derivation.

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
from playbook_engine.clause_position_compiler import (
    _normalize_for_dedup,
    deviations_are_deterministic,
)
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
    _restore_observations,
    mine_corpus,
    project_playbook,
)
from playbook_engine.run_manifest import (
    RUN_MANIFEST_FILENAME,
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


def _mine_and_project(
    out_dir: Path, *, opf_version: str = "0.3", **judges: Any
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Mine + project. Defaults to the OPF 0.3 projection: these tests pin how
    the 0.3 compiler reads the standard fact (stance_detail, summary), a
    shape kept for one release. The OPF 0.4 projection of the same store is
    pinned by ``test_default_project_04_is_the_verdict_free_precedent``."""
    cfg = load_config(_SMOKE_CONFIG)
    taxonomy = load_taxonomy(cfg.taxonomy_path)
    mine_corpus(_CORPUS_DIR, cfg, taxonomy, out_dir, no_cache=True, **judges)
    playbook = project_playbook(out_dir, cfg, taxonomy, opf_version=opf_version)
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
        key = (obs["taxonomy_id"], _normalize_for_dedup(obs["full_text"]))
        answers[key].add(obs["standard"])
        deals[key].add(obs["citation"]["document_id"])
    assert any(len(ids) > 1 for ids in deals.values()), "corpus must repeat signed text"
    assert all(len(values) == 1 for values in answers.values())


def test_default_project_reads_no_stance_and_validates(consumer_run: tuple) -> None:
    observations, playbook = consumer_run
    assert validate_document(playbook).ok
    clauses = playbook["evidence"]["clauses"]
    assert clauses
    for clause in clauses:
        summary = clause["summary"]
        assert summary["historical_stance"] == "no_signal", clause["id"]
        assert summary["acceptable_if"] == []
        assert summary["fallbacks"] == []
        detail = summary["stance_detail"]
        assert detail["basis"] == "all"
        signed_standard_deals = {
            obs["citation"]["document_id"]
            for obs in observations
            if obs["taxonomy_id"] == clause["taxonomy_id"]
            and obs["outcome"] == "signed"
            and obs["standard"]
        }
        assert detail["held"] == len(signed_standard_deals), clause["id"]
        assert detail["held"] <= detail["of"]
    # Refused asks are a deterministic outcome fact and survive.
    assert any(clause["summary"]["rejected"] for clause in clauses)
    # The digest carries no judged concession/variation category either.
    for clause in playbook["digest"]["clauses"]:
        assert clause["historical_stance"] == "no_signal"
        assert not clause.get("concessions")
        assert not clause.get("preferred_variations")


def test_default_project_04_is_the_verdict_free_precedent(tmp_path: Path) -> None:
    """Issue #223: the default OPF 0.4 projection of the consumer-path store
    carries each terminal row's deterministic standard fact as the
    precedent's ``standard``, counts distinct deals, and has no stance,
    risk, deviation or x_judgments anywhere."""
    observations, playbook = _mine_and_project(tmp_path, opf_version="0.4")
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
        if record is None:  # a sub-sentence fragment, excluded like 0.3 does
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


def test_opt_in_deviation_judge_keeps_the_judged_layer(tmp_path: Path) -> None:
    """Wiring a deviation judge (here the stub) is the advisory layer: its
    verdicts replace the standard check's deviation, and the compiler keeps
    the judged derivation."""
    observations, playbook = _mine_and_project(tmp_path, deviation_judge=_NullDeviationJudge())
    bases = {obs["basis"] for obs in observations}
    assert "needs_review" in bases
    # The standard fact is still recorded alongside the judged verdict.
    assert all(isinstance(obs.get("standard"), bool) for obs in observations)
    # Judged derivation: stance_detail is the judged held-rate on the
    # our-paper pool again, not the consumer path's deterministic facts.
    assert json.dumps(playbook)
    assert any(
        clause["summary"]["stance_detail"]["basis"] == "our_paper"
        for clause in playbook["evidence"]["clauses"]
    )


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


def _stances(playbook: dict[str, Any]) -> tuple[set[str], set[str]]:
    clauses = playbook["evidence"]["clauses"]
    assert clauses
    return (
        {c["summary"]["historical_stance"] for c in clauses},
        {c["summary"]["stance_detail"]["basis"] for c in clauses},
    )


def test_opt_in_run_where_every_clause_matches_the_template_compiles_judged(
    tmp_path: Path,
) -> None:
    """The #220 review's failure scenario: a judge was configured but every
    clause was unchanged from the template, so the store's rows look exactly
    like the consumer path's. The mode recorded at mine time — not the rows —
    decides: the opt-in run compiles the judged rollup, the default run the
    consumer path, over the very same corpus."""
    corpus = _all_template_corpus(tmp_path)

    judged_out = tmp_path / "judged"
    ctx = _mine_all_template(judged_out, corpus, deviation_judge=_NullDeviationJudge())
    rows = read_observations_jsonl(judged_out / "observations.jsonl")
    assert rows
    assert {o["outcome"] for o in rows} == {"signed"}
    # The rows alone would be inferred as the consumer path ...
    assert deviations_are_deterministic(_restore_observations(rows))
    # ... but the mode was recorded when they were written.
    assert read_deviation_mode(judged_out) == "judged"
    messages: list[str] = []
    playbook = project_playbook(
        judged_out, ctx["cfg"], ctx["taxonomy"], opf_version="0.3", progress=messages.append
    )
    assert not any("WARNING: no deviation mode" in m for m in messages)
    stances, bases = _stances(playbook)
    assert stances != {"no_signal"}, stances  # a judged stance is derived
    assert "consistently_held" in stances
    assert "our_paper" in bases  # the judged held-rate on the our-paper pool

    consumer_out = tmp_path / "consumer"
    ctx = _mine_all_template(consumer_out, corpus)
    assert read_deviation_mode(consumer_out) == "deterministic"
    playbook = project_playbook(consumer_out, ctx["cfg"], ctx["taxonomy"], opf_version="0.3")
    assert _stances(playbook) == ({"no_signal"}, {"all"})


def test_store_without_a_recorded_mode_falls_back_to_inference_with_a_warning(
    tmp_path: Path,
) -> None:
    """An out-dir mined before the mode was recorded has no ``deviation_mode``
    in its manifest (or no manifest at all): project infers the mode from the
    rows, as before, and says it did."""
    corpus = _all_template_corpus(tmp_path)
    out_dir = tmp_path / "legacy"
    ctx = _mine_all_template(out_dir, corpus, deviation_judge=_NullDeviationJudge())
    (out_dir / RUN_MANIFEST_FILENAME).unlink()
    messages: list[str] = []
    playbook = project_playbook(
        out_dir, ctx["cfg"], ctx["taxonomy"], opf_version="0.3", progress=messages.append
    )
    warnings = [m for m in messages if "WARNING: no deviation mode recorded" in m]
    assert len(warnings) == 1
    assert "inferred deterministic" in warnings[0]
    # Inference cannot see the opt-in here — exactly why the mode is recorded.
    assert _stances(playbook) == ({"no_signal"}, {"all"})


def test_cli_mine_records_the_mode_and_project_reads_it(tmp_path: Path) -> None:
    """Through the CLI: ``mine --with-deviation-judge`` stamps the run
    manifest with its environment AND keeps the recorded mode (the end-of-run
    environment stamp must not erase it), and ``project`` compiles judged."""
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

    result = runner.invoke(
        cli, ["project", str(out_dir), "--config", str(_SMOKE_CONFIG), "--opf-version", "0.3"]
    )
    assert result.exit_code == 0, result.output
    assert "WARNING: no deviation mode" not in result.output
    playbook = json.loads((out_dir / "playbook.opf.json").read_text(encoding="utf-8"))
    stances, _ = _stances(playbook)
    assert "consistently_held" in stances


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
