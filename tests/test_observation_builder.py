"""Tests for observation builder (L4, issue #21).

SECURITY NOTE: All fixtures are programmatically constructed with synthetic
text.  No real agreements are referenced.  Fictional party/document names.
"""

from __future__ import annotations

import json

from playbook_engine.clause_differ import ClauseDiff
from playbook_engine.deviation_classifier import DeviationResult, RiskDelta
from playbook_engine.entity_registry import EntityRegistry, pseudonymize_text
from playbook_engine.observation_builder import (
    Observation,
    ObservationCitation,
    build_observations,
    read_observations_jsonl,
    write_observations_jsonl,
)
from playbook_engine.reversal_detector import ReversalRecord
from playbook_engine.tracked_changes_overlay import HunkEnrichment

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NEUTRAL = RiskDelta(direction="neutral", magnitude="none")
_WORSE = RiskDelta(direction="worse", magnitude="material")


def _cd(
    taxonomy_id: str | None,
    kind: str = "modified",
    text_before: str = "original text",
    text_after: str = "revised text",
    path: str = "1",
) -> ClauseDiff:
    return ClauseDiff(
        taxonomy_id=taxonomy_id,
        clause_path_before=path if kind != "added" else None,
        clause_path_after=path if kind != "removed" else None,
        kind=kind,
        hunks=(),
        text_before=text_before,
        text_after=text_after,
    )


def _dr(deviation: str = "none", basis: str = "deterministic") -> DeviationResult:
    rd = (
        _NEUTRAL
        if basis == "deterministic"
        else RiskDelta(
            direction="worse" if deviation == "substantive" else "neutral",
            magnitude="material" if deviation == "substantive" else "none",
        )
    )
    return DeviationResult(deviation=deviation, risk_delta=rd, basis=basis)


def _reversal(taxonomy_id: str | None, clause_path: str = "1") -> ReversalRecord:
    return ReversalRecord(
        taxonomy_id=taxonomy_id,
        clause_path=clause_path,
        version_inserted="v2",
        version_removed="v3",
        proposed_text="proposed text here",
    )


# ---------------------------------------------------------------------------
# build_observations: basic structure
# ---------------------------------------------------------------------------


def test_build_observations_one_per_diff() -> None:
    diffs = [(_cd("ind"), _dr()), (_cd("gov"), _dr())]
    obs = build_observations("doc1", "v2", "our_paper", diffs, [])
    assert len(obs) == 2


def test_build_observations_taxonomy_id_preserved() -> None:
    diffs = [(_cd("governing_law"), _dr())]
    obs = build_observations("doc1", "v2", "our_paper", diffs, [])
    assert obs[0].taxonomy_id == "governing_law"


def test_build_observations_none_taxonomy_id_preserved() -> None:
    diffs = [(_cd(None), _dr())]
    obs = build_observations("doc1", "v2", "our_paper", diffs, [])
    assert obs[0].taxonomy_id is None


def test_build_observations_citation_fields() -> None:
    diffs = [(_cd("ind", path="3"), _dr())]
    obs = build_observations("deal42", "v1", "our_paper", diffs, [])
    c = obs[0].citation
    assert c.document_id == "deal42"
    assert c.version == "v1"
    assert c.clause_path == "3"


def test_build_observations_provenance_stored() -> None:
    diffs = [(_cd("ind"), _dr())]
    obs = build_observations("doc1", "v2", "counterparty_paper", diffs, [])
    assert obs[0].provenance == "counterparty_paper"


def test_build_observations_outcome_signed_by_default() -> None:
    diffs = [(_cd("ind"), _dr())]
    obs = build_observations("doc1", "v2", "our_paper", diffs, [])
    assert obs[0].outcome == "signed"


def test_build_observations_outcome_proposed_then_reversed() -> None:
    diffs = [(_cd("ind"), _dr())]
    obs = build_observations("doc1", "v2", "our_paper", diffs, [_reversal("ind")])
    assert obs[0].outcome == "proposed_then_reversed"


def test_build_observations_non_reversed_clause_stays_signed() -> None:
    # Use distinct clause paths so reversal on path "1" does not bleed into "2".
    diffs = [(_cd("ind", path="1"), _dr()), (_cd("gov", path="2"), _dr())]
    obs = build_observations("doc1", "v2", "our_paper", diffs, [_reversal("ind", clause_path="1")])
    assert obs[0].outcome == "proposed_then_reversed"  # ind reversed
    assert obs[1].outcome == "signed"  # gov not reversed


# ---------------------------------------------------------------------------
# build_observations: has_signed_copy (issue #83)
#
# When no version was detected as the executed copy, non-reversed
# observations must carry outcome="unsigned", never a fabricated "signed".
# ---------------------------------------------------------------------------


def test_build_observations_unsigned_when_no_signed_copy() -> None:
    diffs = [(_cd("ind"), _dr())]
    obs = build_observations("doc1", "v2", "our_paper", diffs, [], has_signed_copy=False)
    assert obs[0].outcome == "unsigned"


def test_build_observations_no_signed_copy_reversed_clause_unaffected() -> None:
    # Reversal history is independent of whether the final draft was ever
    # executed: a reversed clause keeps its label even with has_signed_copy=False,
    # but a non-reversed clause in the same unsigned trail must not read "signed".
    diffs = [(_cd("ind", path="1"), _dr()), (_cd("gov", path="2"), _dr())]
    obs = build_observations(
        "doc1",
        "v2",
        "our_paper",
        diffs,
        [_reversal("ind", clause_path="1")],
        has_signed_copy=False,
    )
    assert obs[0].outcome == "proposed_then_reversed"  # ind reversed
    assert obs[1].outcome == "unsigned"  # gov not reversed, no signed copy detected


def test_build_observations_has_signed_copy_defaults_true() -> None:
    # Backward compatibility: omitting has_signed_copy preserves prior behavior.
    diffs = [(_cd("ind"), _dr())]
    obs = build_observations("doc1", "v2", "our_paper", diffs, [])
    assert obs[0].outcome == "signed"


# ---------------------------------------------------------------------------
# build_observations: text_summary
# ---------------------------------------------------------------------------


def test_build_observations_text_summary_from_after_text() -> None:
    diffs = [(_cd("ind", text_after="Alice shall indemnify Beta."), _dr())]
    obs = build_observations("doc1", "v2", "our_paper", diffs, [])
    assert obs[0].text_summary == "Alice shall indemnify Beta."


def test_build_observations_text_summary_uses_before_for_removed() -> None:
    diffs = [(_cd("ind", kind="removed", text_before="Removed clause text.", text_after=""), _dr())]
    obs = build_observations("doc1", "v2", "our_paper", diffs, [])
    assert obs[0].text_summary == "Removed clause text."


def test_build_observations_text_summary_truncated_at_200() -> None:
    long_text = "A" * 300
    diffs = [(_cd("ind", text_after=long_text), _dr())]
    obs = build_observations("doc1", "v2", "our_paper", diffs, [])
    assert len(obs[0].text_summary) == 200


def test_build_observations_full_text_not_truncated() -> None:
    """Regression (issue #105): full_text must carry the untruncated clause
    text even when text_summary is capped at 200 chars — any real
    indemnification/insurance clause exceeds 200 chars, and a truncated
    fragment is useless as a drafting standard or acceptable-alternative
    language."""
    long_text = "A" * 300
    diffs = [(_cd("ind", text_after=long_text), _dr())]
    obs = build_observations("doc1", "v2", "our_paper", diffs, [])
    assert obs[0].full_text == long_text
    assert len(obs[0].full_text) == 300


# ---------------------------------------------------------------------------
# build_observations: deviation + risk_delta
# ---------------------------------------------------------------------------


def test_build_observations_deviation_and_risk_delta() -> None:
    dr = DeviationResult(deviation="substantive", risk_delta=_WORSE, basis="judge")
    diffs = [(_cd("ind"), dr)]
    obs = build_observations("doc1", "v2", "our_paper", diffs, [])
    assert obs[0].deviation == "substantive"
    assert obs[0].risk_delta == {"direction": "worse", "magnitude": "material"}


# ---------------------------------------------------------------------------
# build_observations: observation_id uniqueness
# ---------------------------------------------------------------------------


def test_build_observations_ids_unique() -> None:
    diffs = [(_cd("ind", path="1"), _dr()), (_cd("gov", path="2"), _dr())]
    obs = build_observations("doc1", "v2", "our_paper", diffs, [])
    ids = [o.observation_id for o in obs]
    assert len(set(ids)) == len(ids)


def test_build_observations_same_clause_path_deduplicated() -> None:
    """Two clauses at the same path (split/merge) get distinct ids."""
    diffs = [(_cd("ind", path="1"), _dr()), (_cd("ind", path="1"), _dr())]
    obs = build_observations("doc1", "v2", "our_paper", diffs, [])
    ids = [o.observation_id for o in obs]
    assert len(set(ids)) == 2


# ---------------------------------------------------------------------------
# write/read observations.jsonl
# ---------------------------------------------------------------------------


def test_write_read_observations_jsonl_roundtrip(tmp_path) -> None:
    """Acceptance criterion: observations.jsonl matches hand-checked expectations."""
    diffs = [
        (_cd("ind", text_after="Alice shall indemnify Beta."), _dr(basis="deterministic")),
        (
            _cd("gov", kind="modified", text_before="Old law.", text_after="New York law governs."),
            DeviationResult(deviation="reworded_equivalent", risk_delta=_NEUTRAL, basis="judge"),
        ),
    ]
    obs = build_observations("deal1", "v2", "our_paper", diffs, [])
    path = tmp_path / "observations.jsonl"
    write_observations_jsonl(obs, path)

    rows = read_observations_jsonl(path)
    assert len(rows) == 2
    assert rows[0]["taxonomy_id"] == "ind"
    assert rows[0]["outcome"] == "signed"
    assert rows[0]["provenance"] == "our_paper"
    assert rows[1]["taxonomy_id"] == "gov"
    assert rows[1]["deviation"] == "reworded_equivalent"


def test_write_observations_jsonl_atomic_no_tmp_left(tmp_path) -> None:
    """No .jsonl.tmp file left after write."""
    obs = [
        Observation(
            observation_id="doc1/v1/1",
            taxonomy_id="ind",
            text_summary="text",
            citation=ObservationCitation("doc1", "v1", "1", None),
            deviation="none",
            risk_delta={"direction": "neutral", "magnitude": "none"},
            provenance="our_paper",
            outcome="signed",
        )
    ]
    path = tmp_path / "observations.jsonl"
    write_observations_jsonl(obs, path)

    assert path.exists()
    assert not (path.with_suffix(".jsonl.tmp")).exists()


def test_write_empty_observations_creates_empty_file(tmp_path) -> None:
    path = tmp_path / "observations.jsonl"
    write_observations_jsonl([], path)
    assert path.exists()
    assert read_observations_jsonl(path) == []


def test_read_nonexistent_file_returns_empty(tmp_path) -> None:
    assert read_observations_jsonl(tmp_path / "missing.jsonl") == []


def test_write_observations_jsonl_valid_json_lines(tmp_path) -> None:
    diffs = [(_cd("ind", text_after="Indemnification text."), _dr())]
    obs = build_observations("doc1", "v1", "our_paper", diffs, [])
    path = tmp_path / "observations.jsonl"
    write_observations_jsonl(obs, path)

    lines = path.read_text().strip().splitlines()
    for line in lines:
        parsed = json.loads(line)
        assert "observation_id" in parsed
        assert "taxonomy_id" in parsed
        assert "outcome" in parsed
        assert "citation" in parsed


# ---------------------------------------------------------------------------
# build_observations: clause-instance reversal matching (P1.2 acceptance)
# ---------------------------------------------------------------------------


def test_build_observations_same_taxonomy_id_only_reversed_instance_flagged() -> None:
    """Two clauses share a taxonomy_id; only the reversed instance gets the label.

    Acceptance criterion (P1.2): reversal matching must be clause-instance-level,
    not taxonomy-id-bucket-level.
    """
    # Clause path "1" was reversed; clause path "2" was signed as-is.
    diffs = [
        (_cd("ind", path="1"), _dr()),
        (_cd("ind", path="2"), _dr()),
    ]
    obs = build_observations("doc1", "v2", "our_paper", diffs, [_reversal("ind", clause_path="1")])
    assert obs[0].outcome == "proposed_then_reversed"  # path "1" — reversed
    assert obs[1].outcome == "signed"  # path "2" — must NOT be contaminated


def test_build_observations_none_taxonomy_id_no_cross_contamination() -> None:
    """Two unclassified clauses (taxonomy_id=None); only the reversed one is labeled.

    Acceptance criterion (P1.2): the None bucket must not cross-contaminate.
    """
    diffs = [
        (_cd(None, path="1"), _dr()),
        (_cd(None, path="2"), _dr()),
    ]
    obs = build_observations("doc1", "v2", "our_paper", diffs, [_reversal(None, clause_path="1")])
    assert obs[0].outcome == "proposed_then_reversed"  # path "1" — reversed
    assert obs[1].outcome == "signed"  # path "2" — must NOT be contaminated


# ---------------------------------------------------------------------------
# build_observations: whole-clause reversals (issue #106)
#
# A clause inserted mid-negotiation and removed again before the signed
# terminal never produces a net-diff row (clause_differ.diff_aligned skips any
# (before=None, after=None) pair) — so it never reaches deviation_results and
# was previously dropped from observations.jsonl entirely, along with the
# outcome=proposed_then_reversed / rollup.rejected / hold_firm signal it
# should have produced.
# ---------------------------------------------------------------------------


def test_reversal_record_yields_reversed_observation() -> None:
    """A reversal whose clause never appears in deviation_results still
    produces an Observation, built directly from the ReversalRecord."""
    diffs = [(_cd("gov", path="2"), _dr())]  # unrelated clause; no row for path "1"
    reversal = _reversal("ind", clause_path="1")
    obs = build_observations("doc1", "v2", "our_paper", diffs, [reversal])

    assert len(obs) == 2
    reversed_obs = next(o for o in obs if o.citation.clause_path == "1")
    assert reversed_obs.outcome == "proposed_then_reversed"
    assert reversed_obs.taxonomy_id == "ind"
    assert reversed_obs.full_text == "proposed text here"
    assert reversed_obs.provenance == "our_paper"
    # Must not carry a basis that caps clause_position_compiler's rollup
    # position to "negotiable" (see _UNJUDGED_BASES / _STUB_BASES).
    assert reversed_obs.basis == "deterministic"


def test_reversal_record_no_duplicate_when_clause_already_covered() -> None:
    """A reversal matching an existing deviation_results row must NOT also
    get a second, redundant Observation emitted directly from the record."""
    diffs = [(_cd("ind", path="1"), _dr())]
    reversal = _reversal("ind", clause_path="1")
    obs = build_observations("doc1", "v2", "our_paper", diffs, [reversal])

    assert len(obs) == 1
    assert obs[0].outcome == "proposed_then_reversed"


def test_reversal_record_citation_uses_document_version() -> None:
    """The synthetic reversal Observation's citation carries the caller's
    document_id/version, same as every other observation in the batch."""
    diffs: list[tuple] = []
    reversal = _reversal("ind", clause_path="7")
    obs = build_observations("deal9", "v3", "counterparty_paper", diffs, [reversal])

    assert len(obs) == 1
    assert obs[0].citation.document_id == "deal9"
    assert obs[0].citation.version == "v3"
    assert obs[0].citation.clause_path == "7"
    assert obs[0].provenance == "counterparty_paper"


# ---------------------------------------------------------------------------
# Observation / ObservationCitation dataclasses
# ---------------------------------------------------------------------------


def test_observation_citation_to_dict_with_span() -> None:
    c = ObservationCitation("doc1", "v2", "3.1", (100, 200))
    d = c.to_dict()
    assert d["char_span"] == [100, 200]


def test_observation_citation_to_dict_no_span() -> None:
    c = ObservationCitation("doc1", "v2", "3.1", None)
    d = c.to_dict()
    assert d["char_span"] is None


def test_observation_to_dict_structure() -> None:
    obs = Observation(
        observation_id="doc1/v2/1",
        taxonomy_id="ind",
        text_summary="text",
        citation=ObservationCitation("doc1", "v2", "1", None),
        deviation="none",
        risk_delta={"direction": "neutral", "magnitude": "none"},
        provenance="our_paper",
        outcome="signed",
    )
    d = obs.to_dict()
    required_keys = {
        "observation_id",
        "taxonomy_id",
        "text_summary",
        "citation",
        "deviation",
        "risk_delta",
        "provenance",
        "outcome",
    }
    assert required_keys.issubset(d.keys())


# ---------------------------------------------------------------------------
# Tracked-changes attribution (issue #88)
# ---------------------------------------------------------------------------


def test_build_observations_no_attributions_arg_all_none() -> None:
    """Default (no attributions passed): every observation's attribution is None."""
    diffs = [(_cd("ind"), _dr())]
    obs = build_observations("doc1", "v2", "our_paper", diffs, [])
    assert obs[0].attribution is None


def test_build_observations_threads_attribution_by_index() -> None:
    """attributions[idx] lands on Observation[idx].attribution, aligned with deviation_results."""
    diffs = [(_cd("ind", path="1"), _dr()), (_cd("gov", path="2"), _dr())]
    enrichment = HunkEnrichment(author="Alice", date="2024-03-15", tracked_type="insertion")
    obs = build_observations("doc1", "v2", "our_paper", diffs, [], attributions=[enrichment, None])
    assert obs[0].attribution is enrichment
    assert obs[1].attribution is None


def test_observation_to_dict_serializes_attribution() -> None:
    obs = Observation(
        observation_id="doc1/v2/1",
        taxonomy_id="ind",
        text_summary="text",
        citation=ObservationCitation("doc1", "v2", "1", None),
        deviation="substantive",
        risk_delta={"direction": "worse", "magnitude": "material"},
        provenance="counterparty_paper",
        outcome="signed",
        attribution=HunkEnrichment(author="Bob", date=None, tracked_type="deletion"),
    )
    d = obs.to_dict()
    assert d["attribution"] == {"author": "Bob", "date": None, "tracked_type": "deletion"}


def test_observation_full_text_defaults_to_text_summary() -> None:
    """Callers that don't pass full_text explicitly (e.g. existing tests with
    short synthetic text) get full_text == text_summary, not empty."""
    obs = Observation(
        observation_id="doc1/v2/1",
        taxonomy_id="ind",
        text_summary="short text",
        citation=ObservationCitation("doc1", "v2", "1", None),
        deviation="none",
        risk_delta={"direction": "neutral", "magnitude": "none"},
        provenance="our_paper",
        outcome="signed",
    )
    assert obs.full_text == "short text"


def test_observation_to_dict_includes_full_text() -> None:
    obs = Observation(
        observation_id="doc1/v2/1",
        taxonomy_id="ind",
        text_summary="short",
        full_text="the real, untruncated clause text",
        citation=ObservationCitation("doc1", "v2", "1", None),
        deviation="none",
        risk_delta={"direction": "neutral", "magnitude": "none"},
        provenance="our_paper",
        outcome="signed",
    )
    d = obs.to_dict()
    assert d["full_text"] == "the real, untruncated clause text"


# ---------------------------------------------------------------------------
# Citation carries version_id + char_span (issue #108)
#
# citation.version alone (the signed-ordinal display value) is not
# mechanically resolvable to a normalized-tree file, since those are stored
# under their original filename stem, not the ordinal. citation.version_id
# must carry that real stem, and char_span must be threaded from the
# ClauseNode rather than always None.
# ---------------------------------------------------------------------------


def test_citation_carries_version_id_and_char_span() -> None:
    cd = ClauseDiff(
        taxonomy_id="ind",
        clause_path_before="1",
        clause_path_after="1",
        kind="modified",
        hunks=(),
        text_before="original text",
        text_after="revised text",
        clause_version_before="draft_v1_2024_03_01",
        clause_version_after="signed_final",
        char_span_before=(0, 20),
        char_span_after=(0, 21),
    )
    obs = build_observations("doc1", 3, "our_paper", [(cd, _dr())], [])
    c = obs[0].citation
    # version is the caller's display ordinal — unchanged, kept for backward compat.
    assert c.version == 3
    # version_id is the real file stem the clause_path was actually read from —
    # the "after" side, since this diff has a clause_path_after.
    assert c.version_id == "signed_final"
    assert c.char_span == (0, 21)


def test_citation_removed_clause_cites_before_version() -> None:
    """A removed clause has no clause_path_after — its citation must cite the
    version/char_span the clause_path (the "before" side) actually came from,
    never the signed version this observation batch happens to be filed
    under (issue #108's "removed clauses compound this" concern)."""
    cd = ClauseDiff(
        taxonomy_id="ind",
        clause_path_before="2",
        clause_path_after=None,
        kind="removed",
        hunks=(),
        text_before="removed clause text",
        text_after="",
        clause_version_before="draft_v1_2024_03_01",
        clause_version_after=None,
        char_span_before=(10, 30),
        char_span_after=None,
    )
    obs = build_observations("doc1", 3, "our_paper", [(cd, _dr())], [])
    c = obs[0].citation
    assert c.clause_path == "2"
    assert c.version_id == "draft_v1_2024_03_01"
    assert c.char_span == (10, 30)


def test_reversal_observation_citation_uses_version_inserted() -> None:
    """The whole-clause-reversal Observation (issue #106) must cite
    version_inserted — the draft the clause_path actually belongs to — not
    the signed ordinal `version` the batch is filed under."""
    reversal = ReversalRecord(
        taxonomy_id="ind",
        clause_path="7",
        version_inserted="draft_v2",
        version_removed="signed_final",
        proposed_text="proposed text here",
        char_span=(5, 40),
    )
    obs = build_observations("doc1", 3, "our_paper", [], [reversal])
    c = obs[0].citation
    assert c.version == 3
    assert c.version_id == "draft_v2"
    assert c.char_span == (5, 40)


def test_observation_citation_to_dict_includes_version_id() -> None:
    c = ObservationCitation("doc1", 3, "3.1", (100, 200), version_id="signed_final")
    d = c.to_dict()
    assert d["version_id"] == "signed_final"


def test_observation_citation_version_id_defaults_none() -> None:
    """Backward compatibility: callers/tests that never had a real version id
    (e.g. the _cd() helper elsewhere in this file) get version_id=None."""
    diffs = [(_cd("ind"), _dr())]
    obs = build_observations("doc1", "v2", "our_paper", diffs, [])
    assert obs[0].citation.version_id is None


def test_observation_to_dict_attribution_none() -> None:
    obs = Observation(
        observation_id="doc1/v2/1",
        taxonomy_id="ind",
        text_summary="text",
        citation=ObservationCitation("doc1", "v2", "1", None),
        deviation="none",
        risk_delta={"direction": "neutral", "magnitude": "none"},
        provenance="our_paper",
        outcome="signed",
    )
    assert obs.to_dict()["attribution"] is None


# ---------------------------------------------------------------------------
# Verbatim, pseudonymized precedent text on fallback/acceptable_if-backing
# observations (issue #157)
#
# clause_position_compiler pulls fallback/acceptable_if entries straight from
# obs.full_text (never text_summary) — a fallback is a signed, our-paper,
# worse-risk_delta observation; an acceptable_if is a signed, neutral-risk_delta,
# actually-deviated (deviation != "none") observation with a real judge basis.
# Both must carry the untruncated clause text (issue #105) AND — since that
# text is exactly what a downstream compile step embeds verbatim into
# playbook.opf.json as drafting language for the runtime LLM — that text must
# be pseudonymizable to alias-only via the #153 born-safe path before it ever
# reaches the observation store (see pipeline._pseudonymize_observations,
# which applies entity_registry.pseudonymize_text to exactly this full_text
# field).
# ---------------------------------------------------------------------------

_KNOWN_ENTITY = "State University"


def test_fallback_backing_observation_carries_verbatim_pseudonymized_precedent_text(
    tmp_path,
) -> None:
    """A worse-risk, signed, our-paper observation (clause_position_compiler's
    fallback criteria) must carry the full untruncated precedent clause text,
    and that text must pseudonymize to alias-only — never the raw entity name
    — via the #153 born-safe path."""
    long_text = (
        f"Alpha Corp shall indemnify {_KNOWN_ENTITY} against third-party claims "
        "arising from the placement programme, provided that such claims are "
        "reported within thirty (30) days of discovery and " + "X" * 150
    )
    assert len(long_text) > 200  # must actually exceed the text_summary cap

    dr = DeviationResult(deviation="substantive", risk_delta=_WORSE, basis="judge")
    diffs = [(_cd("ind", text_after=long_text), dr)]
    obs = build_observations("doc1", "v2", "our_paper", diffs, [])[0]

    # Fallback-backing shape (clause_position_compiler._derive_rollup).
    assert obs.provenance == "our_paper"
    assert obs.outcome == "signed"
    assert obs.risk_delta["direction"] == "worse"

    # Verbatim precedent text, alongside the existing summary + citation.
    assert obs.full_text == long_text
    assert obs.text_summary == long_text[:200]
    assert obs.citation.document_id == "doc1"

    # Born-safe: pseudonymizing the carried full_text replaces the raw entity
    # name with its stable alias, never leaving the raw name behind.
    reg = EntityRegistry.load(tmp_path / "entity_registry.json")
    pseudonymized = pseudonymize_text(obs.full_text, [_KNOWN_ENTITY], reg)
    assert _KNOWN_ENTITY not in pseudonymized
    assert reg.alias_for(_KNOWN_ENTITY) in pseudonymized


def test_acceptable_if_backing_observation_carries_verbatim_pseudonymized_precedent_text(
    tmp_path,
) -> None:
    """A neutral-risk, signed, actually-deviated observation
    (clause_position_compiler's acceptable_if criteria) must carry the full
    untruncated precedent clause text, and that text must pseudonymize to
    alias-only via the #153 born-safe path."""
    long_text = (
        f"{_KNOWN_ENTITY} shall provide reasonable cooperation to Alpha Corp "
        "in connection with any third-party claim, reworded but materially "
        "equivalent to the standard cooperation clause, and " + "Y" * 150
    )
    assert len(long_text) > 200

    dr = DeviationResult(deviation="reworded_equivalent", risk_delta=_NEUTRAL, basis="judge")
    diffs = [(_cd("coop", text_after=long_text), dr)]
    obs = build_observations("doc1", "v2", "our_paper", diffs, [])[0]

    # Acceptable_if-backing shape (clause_position_compiler._derive_rollup).
    assert obs.outcome == "signed"
    assert obs.risk_delta["direction"] == "neutral"
    assert obs.deviation != "none"
    assert obs.basis == "judge"  # not in _UNJUDGED_BASES

    # Verbatim precedent text, alongside the existing summary + citation.
    assert obs.full_text == long_text
    assert obs.text_summary == long_text[:200]
    assert obs.citation.document_id == "doc1"

    # Born-safe: pseudonymizing the carried full_text replaces the raw entity
    # name with its stable alias, never leaving the raw name behind.
    reg = EntityRegistry.load(tmp_path / "entity_registry.json")
    pseudonymized = pseudonymize_text(obs.full_text, [_KNOWN_ENTITY], reg)
    assert _KNOWN_ENTITY not in pseudonymized
    assert reg.alias_for(_KNOWN_ENTITY) in pseudonymized
