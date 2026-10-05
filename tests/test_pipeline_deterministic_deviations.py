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

from playbook_engine.clause_differ import ClauseDiff
from playbook_engine.clause_position_compiler import _normalize_for_dedup
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
from playbook_engine.taxonomy import load_taxonomy
from playbook_engine.tracked_changes_overlay import HunkEnrichment
from playbook_engine.validator import validate_document

_REPO_ROOT = Path(__file__).resolve().parent.parent
_NDA_DIR = _REPO_ROOT / "examples" / "nda"
_CORPUS_DIR = _NDA_DIR / "corpus"
_SMOKE_CONFIG = _NDA_DIR / "config.smoke.yaml"


def _mine_and_project(out_dir: Path, **judges: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
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
