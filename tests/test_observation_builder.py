"""Tests for observation builder (L4, issue #21).

SECURITY NOTE: All fixtures are programmatically constructed with synthetic
text.  No real agreements are referenced.  Fictional party/document names.
"""

from __future__ import annotations

import json

import pytest

from playbook_engine.clause_classifier import ClassifiedClause, ClauseClassification
from playbook_engine.clause_differ import ClauseDiff
from playbook_engine.clause_tree import ClauseNode
from playbook_engine.deviation_classifier import DeviationResult, RiskDelta
from playbook_engine.entity_registry import EntityRegistry, pseudonymize_text
from playbook_engine.observation_builder import (
    DROPPED_ORIGIN_UNDETERMINED,
    DROPPED_REFUSED_UNSIGNED,
    DROPPED_STANDARD_REMOVED_UNSIGNED,
    DROPPED_SURVIVES_IN_TERMINAL,
    OUTCOME_CONCEDED_BEFORE_SIGNING,
    Observation,
    ObservationCitation,
    build_observations,
    read_observations_jsonl,
    summarize_clause_text,
    truncate_search_snippets,
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


# Our standard (template) clause text per taxonomy_id — the origin reference
# for a clause removed before signing (issue #216). Every removed text in the
# tests below that is NOT one of these is non-standard (their language), so
# its removal is a refused ask.
_STD: dict[str, str] = {
    "ind": "Each party shall indemnify the other against third-party claims.",
    "governing_law": "This Agreement is governed by the laws of the State of Delaware.",
    "non_solicit": (
        "For twelve months neither party shall solicit the other's employees for hire."
    ),
}


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
    """Issue #216: a reversal inside a clause that survived to the terminal
    is its own proposed_then_reversed observation carrying the PROPOSED text;
    the terminal's own text stays the (single) signed observation — it is
    never relabeled as rejected."""
    diffs = [(_cd("ind"), _dr())]
    obs = build_observations("doc1", "v2", "our_paper", diffs, [_reversal("ind")])
    assert [(o.outcome, o.full_text) for o in obs] == [
        ("signed", "revised text"),
        ("proposed_then_reversed", "proposed text here"),
    ]


def test_build_observations_non_reversed_clause_stays_signed() -> None:
    # Use distinct clause paths so reversal on path "1" does not bleed into "2".
    diffs = [(_cd("ind", path="1"), _dr()), (_cd("gov", path="2"), _dr())]
    obs = build_observations("doc1", "v2", "our_paper", diffs, [_reversal("ind", clause_path="1")])
    outcomes = sorted((o.taxonomy_id, o.outcome) for o in obs)
    assert outcomes == [
        ("gov", "signed"),
        ("ind", "proposed_then_reversed"),
        ("ind", "signed"),
    ]


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


def test_build_observations_no_signed_copy_drops_reversals() -> None:
    # Issue #221: with no detected signed copy the terminal is only the last
    # draft, so a reversal passed in is not a refused ask — no
    # proposed_then_reversed observation, counted under
    # DROPPED_REFUSED_UNSIGNED — and no clause reads "signed".
    diffs = [(_cd("ind", path="1"), _dr()), (_cd("gov", path="2"), _dr())]
    dropped: dict[str, int] = {}
    obs = build_observations(
        "doc1",
        "v2",
        "our_paper",
        diffs,
        [_reversal("ind", clause_path="1")],
        has_signed_copy=False,
        dropped=dropped,
    )
    outcomes = sorted((o.taxonomy_id, o.outcome) for o in obs)
    assert outcomes == [
        ("gov", "unsigned"),  # no signed copy detected
        ("ind", "unsigned"),
    ]
    assert dropped == {DROPPED_REFUSED_UNSIGNED: 1}

    # Control: the same reversal against a signed terminal is a refused ask.
    signed = build_observations(
        "doc1", "v2", "our_paper", diffs, [_reversal("ind", clause_path="1")]
    )
    assert ("ind", "proposed_then_reversed") in [(o.taxonomy_id, o.outcome) for o in signed]


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
    # A removed row whose own (non-standard) text is absent from the terminal
    # is proposed_then_reversed and reads its text from the before side.
    diffs = [(_cd("ind", kind="removed", text_before="Removed clause text.", text_after=""), _dr())]
    obs = build_observations("doc1", "v2", "our_paper", diffs, [], standard_text_by_tid=_STD)
    assert len(obs) == 1
    assert obs[0].text_summary == "Removed clause text."
    assert obs[0].outcome == "proposed_then_reversed"


def test_build_observations_text_summary_hard_cut_only_for_unbroken_token() -> None:
    """A single token longer than the 300-char cap has no sentence or word
    boundary to cut at — the one case still hard-cut (issue #217)."""
    long_text = "A" * 400
    diffs = [(_cd("ind", text_after=long_text), _dr())]
    obs = build_observations("doc1", "v2", "our_paper", diffs, [])
    assert obs[0].text_summary == "A" * 300


def test_build_observations_text_summary_ends_on_sentence_boundary() -> None:
    """Issue #217: text_summary is the longest ≤ 300-char prefix ending on a
    sentence boundary — never a mid-word 200-char cut."""
    first = ("The Receiving Party shall hold all Confidential Information in strict " * 3).strip()
    first += "."
    second = ("It shall not disclose any of it to a third party without consent " * 4).strip()
    long_text = f"{first} {second}."
    assert 60 <= len(first) <= 300 < len(long_text)
    diffs = [(_cd("conf", text_after=long_text), _dr())]
    obs = build_observations("doc1", "v2", "our_paper", diffs, [])
    assert obs[0].text_summary == first
    assert obs[0].full_text == long_text


def test_build_observations_reversal_text_summary_ends_on_sentence_boundary() -> None:
    """The proposed_then_reversed observation's text_summary (built from the
    reversal's proposed_text, not a ClauseDiff) follows the same rule."""
    first = ("Recipient shall indemnify Discloser for any losses of any kind " * 3).strip() + "."
    proposed = f"{first} " + ("And for all costs and fees without limit " * 6).strip() + "."
    # Producer invariants (detect_reversals): version_inserted is a middle
    # draft (v2 of v1..v3), version_removed the signed terminal (v3), and a
    # whole-clause reversal has no net-diff row of its own.
    reversal = ReversalRecord(
        taxonomy_id="ind",
        clause_path="5",
        version_inserted="v2",
        version_removed="v3",
        proposed_text=proposed,
    )
    diffs = [(_cd("conf", text_after="Short clause."), _dr())]
    obs = build_observations("doc1", "v3", "our_paper", diffs, [reversal])
    rev = [o for o in obs if o.outcome == "proposed_then_reversed"]
    assert len(rev) == 1
    assert rev[0].text_summary == first
    assert rev[0].full_text == proposed


# ---------------------------------------------------------------------------
# summarize_clause_text (issue #217)
# ---------------------------------------------------------------------------


def test_summarize_short_text_returned_whole() -> None:
    assert summarize_clause_text("  Short clause, no full stop  ") == "Short clause, no full stop"


def test_summarize_cuts_after_last_sentence_in_window() -> None:
    s1 = ("Alpha " * 30).strip()  # 179 chars
    s2 = ("Beta " * 10).strip()  # 49 chars
    text = f"{s1}. {s2}. " + "Gamma " * 40
    out = summarize_clause_text(text)
    assert out == f"{s1}. {s2}."
    assert len(out) <= 300


def test_summarize_keeps_a_real_first_sentence_even_when_short() -> None:
    """Any sentence of real clause language (≥ 60 chars) is a valid cut, even
    when it leaves most of the window unused."""
    s1 = "Recipient shall keep all Confidential Information strictly secret."
    text = f"{s1} " + "and furthermore " * 30
    assert summarize_clause_text(text) == s1


def test_summarize_sentence_end_needs_following_whitespace() -> None:
    """A dotted number ("2.1") is not a sentence end."""
    text = ("See Section 2.1 of this Agreement for the terms " * 10).strip()
    out = summarize_clause_text(text)
    assert len(out) <= 300
    assert text.startswith(out)
    assert not out.endswith(".")
    assert text[len(out)] == " "


def test_summarize_short_first_sentence_falls_back_to_word_boundary() -> None:
    """A heading-like first sentence ("Confidentiality.") would make the
    summary merely restate the clause name — fall back to a word boundary
    that keeps real clause language."""
    text = "Confidentiality. " + ("The recipient shall protect all disclosed material " * 10)
    out = summarize_clause_text(text)
    assert out != "Confidentiality."
    assert 250 < len(out) <= 300
    assert text.startswith(out)
    assert text[len(out)] == " "  # ended on a word boundary, not mid-word


def test_summarize_window_edge_word_boundary() -> None:
    """When the char right after the window is whitespace the whole window is
    a clean word-boundary cut."""
    text = "x" * 299 + "y " + "z" * 50
    assert summarize_clause_text(text) == "x" * 299 + "y"


def test_summarize_sentence_end_exactly_at_window_edge() -> None:
    text = "w " * 149 + "." + " more text follows here"
    assert len("w " * 149 + ".") == 299
    out = summarize_clause_text(text)
    assert out == ("w " * 149 + ".")


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
    """A signed observation and a reversal at the same path get distinct ids."""
    diffs = [(_cd("ind", path="1"), _dr())]
    obs = build_observations("doc1", "v2", "our_paper", diffs, [_reversal("ind", clause_path="1")])
    ids = [o.observation_id for o in obs]
    assert len(obs) == 2
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
    # Both instances lost their terminal slot (issue #216). Path "1"'s text
    # survives verbatim in the terminal (relocated to path "3"), so it is
    # neither reversed nor signed — dropped and counted as surviving. Path
    # "2"'s text is gone from the terminal: reversed. The later-draft
    # reversal at the coincident path "1" is NOT claimed by the removed row —
    # it is its own observation carrying its own proposed text.
    diffs = [
        (_cd("ind", kind="removed", text_before="Alpha clause wording.", path="1"), _dr()),
        (_cd("ind", kind="removed", text_before="Beta clause wording.", path="2"), _dr()),
        (
            _cd(
                "ind",
                kind="unchanged",
                text_before="Alpha clause wording.",
                text_after="Alpha clause wording.",
                path="3",
            ),
            _dr(),
        ),
    ]
    dropped: dict[str, int] = {}
    obs = build_observations(
        "doc1",
        "v2",
        "our_paper",
        diffs,
        [_reversal("ind", clause_path="1")],
        dropped=dropped,
        standard_text_by_tid=_STD,
    )
    assert [(o.citation.clause_path, o.outcome, o.full_text) for o in obs] == [
        ("2", "proposed_then_reversed", "Beta clause wording."),
        ("3", "signed", "Alpha clause wording."),
        ("1", "proposed_then_reversed", "proposed text here"),
    ]
    assert dropped == {DROPPED_SURVIVES_IN_TERMINAL: 1}


def test_build_observations_none_taxonomy_id_no_cross_contamination() -> None:
    """Two unclassified clauses (taxonomy_id=None); only the reversal is labeled.

    Acceptance criterion (P1.2): the None bucket must not cross-contaminate.
    Issue #216: an unclassified clause removed before signing has no
    standard to tell our language from theirs, so it is neither reversed nor
    conceded — dropped and counted as origin-undetermined.
    """
    diffs = [
        (_cd(None, kind="removed", text_before="Alpha clause wording.", path="1"), _dr()),
        (_cd(None, kind="removed", text_before="Beta clause wording.", path="2"), _dr()),
        (
            _cd(
                None,
                kind="unchanged",
                text_before="Alpha clause wording.",
                text_after="Alpha clause wording.",
                path="3",
            ),
            _dr(),
        ),
    ]
    dropped: dict[str, int] = {}
    obs = build_observations(
        "doc1",
        "v2",
        "our_paper",
        diffs,
        [_reversal(None, clause_path="1")],
        dropped=dropped,
        standard_text_by_tid=_STD,
    )
    assert [(o.citation.clause_path, o.outcome, o.full_text) for o in obs] == [
        ("3", "signed", "Alpha clause wording."),
        ("1", "proposed_then_reversed", "proposed text here"),
    ]
    assert dropped == {DROPPED_SURVIVES_IN_TERMINAL: 1, DROPPED_ORIGIN_UNDETERMINED: 1}


# ---------------------------------------------------------------------------
# build_observations: whole-clause reversals (issue #106)
#
# A clause inserted mid-negotiation and removed again before the signed
# terminal never produces a net-diff row (clause_differ.diff_aligned skips any
# (before=None, after=None) pair) — so it never reaches deviation_results and
# was previously dropped from observations.jsonl entirely, along with the
# outcome=proposed_then_reversed refused-ask signal it should have produced.
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
    # Must not carry an unjudged placeholder basis (it would trip the
    # playbook's stub_basis_present watermark).
    assert reversed_obs.basis == "deterministic"


def test_removed_row_never_claims_reversal_record_sharing_its_path() -> None:
    """Issue #216: a removed row's path is a FIRST-version path, a
    ReversalRecord's is a later draft's (version_inserted), so sharing
    (taxonomy_id, clause_path) is a coincidence of numbering. The record is
    never swallowed by the removed row: both are emitted, each with its own
    text and citation."""
    diffs = [(_cd("ind", kind="removed", path="1"), _dr())]
    reversal = _reversal("ind", clause_path="1")
    obs = build_observations(
        "doc1", "v2", "our_paper", diffs, [reversal], standard_text_by_tid=_STD
    )

    assert [(o.outcome, o.full_text, o.citation.version_id) for o in obs] == [
        ("proposed_then_reversed", "original text", None),
        ("proposed_then_reversed", "proposed text here", "v2"),
    ]


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
# search_snippet defaulting + serialization (issue #95)
# ---------------------------------------------------------------------------


def test_observation_search_snippet_defaults_to_full_text() -> None:
    """Callers that don't pass search_snippet explicitly get search_snippet
    == full_text (untruncated), mirroring full_text's own default-from-
    text_summary cascade — this is what lets every existing Observation
    construction site (build_observations, pipeline.py's template path)
    populate a snippet source for free, with no changes at those sites."""
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
    assert obs.search_snippet == "the real, untruncated clause text"


def test_observation_search_snippet_defaults_to_text_summary_when_no_full_text() -> None:
    """With neither full_text nor search_snippet passed, both cascade from
    text_summary — the same two-step __post_init__ chain full_text alone
    already exercises."""
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
    assert obs.search_snippet == "short text"


def test_observation_search_snippet_explicit_value_not_overridden() -> None:
    obs = Observation(
        observation_id="doc1/v2/1",
        taxonomy_id="ind",
        text_summary="short",
        full_text="the real, untruncated clause text",
        search_snippet="a hand-picked excerpt",
        citation=ObservationCitation("doc1", "v2", "1", None),
        deviation="none",
        risk_delta={"direction": "neutral", "magnitude": "none"},
        provenance="our_paper",
        outcome="signed",
    )
    assert obs.search_snippet == "a hand-picked excerpt"


def test_observation_to_dict_includes_search_snippet_when_set() -> None:
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
    assert d["search_snippet"] == "the real, untruncated clause text"


def test_observation_to_dict_omits_search_snippet_when_no_clause_text() -> None:
    """No text_summary/full_text/search_snippet at all (empty clause) →
    search_snippet cascades to "" and to_dict() omits the key entirely,
    never a useless "" placeholder — same convention as proposed_by/
    observed_at/counterparty_ref above."""
    obs = Observation(
        observation_id="doc1/v2/1",
        taxonomy_id="ind",
        text_summary="",
        citation=ObservationCitation("doc1", "v2", "1", None),
        deviation="none",
        risk_delta={"direction": "neutral", "magnitude": "none"},
        provenance="our_paper",
        outcome="signed",
    )
    assert obs.search_snippet == ""
    assert "search_snippet" not in obs.to_dict()


def test_build_observations_search_snippet_matches_full_text_untruncated() -> None:
    """At construction time (build_observations), search_snippet must equal
    the untruncated full_text — NOT a pre-truncated fragment. Truncation to
    the short ~40-100 char phrase happens later, in
    observation_builder.truncate_search_snippets, which the pipeline calls
    strictly AFTER pseudonymization (see that function's docstring)."""
    long_text = f"Alpha Corp shall indemnify Acme University against claims. {'X' * 150}"
    diffs = [(_cd("ind", text_after=long_text), _dr())]
    obs = build_observations("doc1", "v2", "our_paper", diffs, [])
    assert obs[0].search_snippet == long_text
    assert obs[0].search_snippet == obs[0].full_text


def test_truncate_search_snippets_multiline_clause_stays_substring_of_full_text() -> None:
    """A multi-line clause (the norm for indemnification / limitation-of-
    liability clauses with sub-paragraphs -- segmenter.py's ClauseNode.text
    is built via ``"\\n".join(lines_list).strip()``) must still yield a
    search_snippet that is a genuine verbatim substring of full_text after
    truncate_search_snippets runs.

    Regression guard: the prior _shape_search_snippet collapsed ALL
    whitespace -- including embedded newlines -- via
    ``" ".join(text.split())``, which rebuilds the string from its words and
    is therefore not guaranteed to appear anywhere in the untouched source
    text. Word/PDF Ctrl+F does not match across a paragraph mark, so a
    collapsed multi-line snippet was unfindable in the source document,
    defeating this field's entire purpose. Every existing search_snippet
    fixture (elsewhere in this file and in test_dynamics_fields.py /
    test_entity_registry.py) is single-line, so none of them exercise this.
    """
    full_text = (
        "Alpha Corp shall indemnify and hold harmless the counterparty "
        "from and against:\n"
        "any and all third-party claims, losses, and expenses arising out "
        "of or relating to this agreement."
    )
    assert "\n" in full_text, "premise: fixture clause spans more than one line"

    obs = Observation(
        observation_id="doc1/v2/1",
        taxonomy_id="ind",
        text_summary="short",
        full_text=full_text,
        citation=ObservationCitation("doc1", "v2", "1", None),
        deviation="none",
        risk_delta={"direction": "neutral", "magnitude": "none"},
        provenance="our_paper",
        outcome="signed",
    )
    assert obs.search_snippet == full_text, "premise: search_snippet defaults to full_text"

    truncated = truncate_search_snippets([obs])
    snippet = truncated[0].search_snippet

    assert snippet, "expected a non-empty snippet"
    assert snippet in full_text, (
        "search_snippet must be a genuine verbatim substring of full_text "
        f"(so Ctrl+F in the source document finds it); got snippet={snippet!r} "
        f"which is not a substring of full_text={full_text!r}"
    )
    assert "\n" not in snippet, "a single-line excerpt should never itself carry a newline"


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
    obs = build_observations("doc1", 3, "our_paper", [(cd, _dr())], [], standard_text_by_tid=_STD)
    assert len(obs) == 1
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


# ---------------------------------------------------------------------------
# citation.version carries the cited draft's ordinal (ordinal_by_vid)
#
# version_id alone fixing resolvability (issue #108) is not enough: the OPF
# boundary (clause_position_compiler.OPFCitation) drops version_id, so
# citation.version — the only version field that survives into the playbook —
# must itself be the ordinal of the version the cited text actually lives in.
# Filing a reversal/removed-clause citation under the signed ordinal makes
# citation_resolver hand back a file the cited clause_path/char_span does not
# exist in.
# ---------------------------------------------------------------------------

_ORDINALS = {"draft_v1": 1, "draft_v2": 2, "signed_final": 3}


def test_reversal_citation_version_is_draft_ordinal() -> None:
    """A whole-clause-reversal Observation's citation.version must be the
    ordinal of version_inserted — the draft the proposed text lives in — not
    the signed ordinal the batch is filed under."""
    reversal = ReversalRecord(
        taxonomy_id="ind",
        clause_path="7",
        version_inserted="draft_v2",
        version_removed="signed_final",
        proposed_text="proposed text here",
        char_span=(5, 40),
    )
    obs = build_observations("doc1", 3, "our_paper", [], [reversal], ordinal_by_vid=_ORDINALS)
    c = obs[0].citation
    assert c.version == 2  # draft_v2's ordinal, NOT the signed ordinal 3
    assert c.version_id == "draft_v2"


def test_removed_clause_citation_version_is_draft_ordinal() -> None:
    """A removed clause's citation cites the "before" side (the last draft the
    clause existed in) — its version must be that draft's ordinal too."""
    cd = ClauseDiff(
        taxonomy_id="ind",
        clause_path_before="2",
        clause_path_after=None,
        kind="removed",
        hunks=(),
        text_before="removed clause text",
        text_after="",
        clause_version_before="draft_v1",
        clause_version_after=None,
        char_span_before=(10, 30),
        char_span_after=None,
    )
    obs = build_observations(
        "doc1",
        3,
        "our_paper",
        [(cd, _dr())],
        [],
        ordinal_by_vid=_ORDINALS,
        standard_text_by_tid=_STD,
    )
    assert len(obs) == 1
    c = obs[0].citation
    assert c.version == 1  # draft_v1's ordinal, NOT the signed ordinal 3
    assert c.version_id == "draft_v1"


def test_signed_clause_citation_version_keeps_signed_ordinal() -> None:
    """A clause read from the signed version keeps the signed ordinal — its
    version_id IS the signed version, so the map resolves to the same value."""
    cd = ClauseDiff(
        taxonomy_id="ind",
        clause_path_before="1",
        clause_path_after="1",
        kind="modified",
        hunks=(),
        text_before="original text",
        text_after="revised text",
        clause_version_before="draft_v1",
        clause_version_after="signed_final",
        char_span_before=(0, 20),
        char_span_after=(0, 21),
    )
    obs = build_observations("doc1", 3, "our_paper", [(cd, _dr())], [], ordinal_by_vid=_ORDINALS)
    assert obs[0].citation.version == 3
    assert obs[0].citation.version_id == "signed_final"


def test_citation_version_unmapped_vid_falls_back_to_caller_version() -> None:
    """A version_id absent from ordinal_by_vid falls back to the caller's
    version rather than raising — mirrors build_round_moves' .get() fallback."""
    reversal = ReversalRecord(
        taxonomy_id="ind",
        clause_path="7",
        version_inserted="not_in_map",
        version_removed="signed_final",
        proposed_text="proposed text here",
    )
    obs = build_observations("doc1", 3, "our_paper", [], [reversal], ordinal_by_vid=_ORDINALS)
    assert obs[0].citation.version == 3


def test_citation_version_no_ordinal_map_keeps_caller_version() -> None:
    """Legacy callers that pass no ordinal_by_vid keep the filed version on
    every citation, reversal or not — behavior unchanged."""
    diffs = [(_cd("gov", path="2"), _dr())]
    obs = build_observations("doc1", "v3", "our_paper", diffs, [_reversal("ind", clause_path="1")])
    assert all(o.citation.version == "v3" for o in obs)


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

    # A judged, worse-risk signed our-paper row.
    assert obs.provenance == "our_paper"
    assert obs.outcome == "signed"
    assert obs.risk_delta["direction"] == "worse"

    # Verbatim precedent text, alongside the existing summary + citation.
    assert obs.full_text == long_text
    # Issue #217: a word-boundary prefix, never a mid-word cut.
    assert obs.text_summary == summarize_clause_text(long_text)
    assert long_text.startswith(obs.text_summary)
    assert long_text[len(obs.text_summary)] == " "
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

    # A judged, neutral-risk signed row.
    assert obs.outcome == "signed"
    assert obs.risk_delta["direction"] == "neutral"
    assert obs.deviation != "none"
    assert obs.basis == "judge"  # not in _UNJUDGED_BASES

    # Verbatim precedent text, alongside the existing summary + citation.
    assert obs.full_text == long_text
    # Issue #217: a word-boundary prefix, never a mid-word cut.
    assert obs.text_summary == summarize_clause_text(long_text)
    assert long_text.startswith(obs.text_summary)
    assert long_text[len(obs.text_summary)] == " "
    assert obs.citation.document_id == "doc1"

    # Born-safe: pseudonymizing the carried full_text replaces the raw entity
    # name with its stable alias, never leaving the raw name behind.
    reg = EntityRegistry.load(tmp_path / "entity_registry.json")
    pseudonymized = pseudonymize_text(obs.full_text, [_KNOWN_ENTITY], reg)
    assert _KNOWN_ENTITY not in pseudonymized
    assert reg.alias_for(_KNOWN_ENTITY) in pseudonymized


# ---------------------------------------------------------------------------
# build_observations: the deal is the unit of precedent (issue #216)
#
# Exactly one terminal observation per (document, taxonomy_id), built from
# the terminal version's own tree; draft-only text is never "signed".
# ---------------------------------------------------------------------------


def _node_cc(
    path: str, taxonomy_id: str | None, text: str, span: tuple[int, int]
) -> ClassifiedClause:
    return ClassifiedClause(
        node=ClauseNode(clause_path=path, heading=None, text=text, char_span=span),
        classification=ClauseClassification(
            taxonomy_id=taxonomy_id, confidence=0.9, basis="exact_match"
        ),
    )


def _terminal_cd(
    taxonomy_id: str | None,
    path: str,
    text: str,
    span: tuple[int, int],
    kind: str = "unchanged",
    before: str | None = None,
) -> ClauseDiff:
    return ClauseDiff(
        taxonomy_id=taxonomy_id,
        clause_path_before=path,
        clause_path_after=path,
        kind=kind,
        hunks=(),
        text_before=before if before is not None else text,
        text_after=text,
        clause_version_before="draft_v1",
        clause_version_after="signed_final",
        char_span_before=span,
        char_span_after=span,
    )


def test_multi_node_clause_is_one_signed_observation_from_terminal_tree() -> None:
    """A clause spanning several terminal nodes is ONE precedent: full_text is
    the nodes' text in the terminal tree's document order joined by a single
    newline, cited to the first node in the signed version."""
    tree = [
        _node_cc("4", "confidentiality", "Each party shall keep it confidential.", (100, 140)),
        _node_cc("4.1", "confidentiality", "Including for three years.", (141, 170)),
        _node_cc("5", "governing_law", "Governed by the laws of Delaware.", (171, 205)),
        _node_cc("4.2", "confidentiality", "Return or destroy on request.", (206, 240)),
    ]
    # Rows deliberately NOT in document order — the tree is the authority.
    rows = [
        (
            _terminal_cd("confidentiality", "4.2", "Return or destroy on request.", (206, 240)),
            _dr(),
        ),
        (
            _terminal_cd("governing_law", "5", "Governed by the laws of Delaware.", (171, 205)),
            _dr(),
        ),
        (
            _terminal_cd(
                "confidentiality", "4", "Each party shall keep it confidential.", (100, 140)
            ),
            _dr(),
        ),
        (_terminal_cd("confidentiality", "4.1", "Including for three years.", (141, 170)), _dr()),
    ]
    obs = build_observations(
        "deal7",
        3,
        "our_paper",
        rows,
        [],
        ordinal_by_vid=_ORDINALS,
        terminal_clauses=tree,
        terminal_version_id="signed_final",
    )
    assert [o.taxonomy_id for o in obs] == ["confidentiality", "governing_law"]
    conf = obs[0]
    assert conf.outcome == "signed"
    assert conf.full_text == (
        "Each party shall keep it confidential.\n"
        "Including for three years.\n"
        "Return or destroy on request."
    )
    assert conf.citation.clause_path == "4"
    assert conf.citation.char_span == (100, 140)
    assert conf.citation.version == 3
    assert conf.citation.version_id == "signed_final"


def test_merged_observation_carries_worst_risk_and_weakest_basis() -> None:
    """Merging nodes never hides a concession (the worse-risk row is the
    representative) nor launders an unjudged node into a judged one (the
    weakest basis wins); confidence is the lowest among the rows."""
    worse = DeviationResult(
        deviation="substantive",
        risk_delta=RiskDelta(direction="worse", magnitude="minor"),
        basis="judge",
    )
    # deviation="needs_review" + basis="needs_review" is the sentinel
    # agent_judge._deviation_needs_review emits for a changed clause that has
    # no stored verdict.
    neutral_unjudged = DeviationResult(
        deviation="needs_review",
        risk_delta=RiskDelta(direction="neutral", magnitude="none"),
        basis="needs_review",
    )
    rows = [
        (
            _terminal_cd(
                "ind",
                "1",
                "First part of the clause.",
                (0, 25),
                kind="modified",
                before="First draft part of the clause.",
            ),
            neutral_unjudged,
        ),
        (
            _terminal_cd(
                "ind",
                "2",
                "Second part of the clause.",
                (26, 52),
                kind="modified",
                before="Second draft part of the clause.",
            ),
            worse,
        ),
        (_terminal_cd("ind", "3", "Third, unchanged part.", (53, 75)), _dr()),
    ]
    obs = build_observations(
        "doc1", 3, "our_paper", rows, [], classification_confidences=[0.8, 0.6, 0.95]
    )
    assert len(obs) == 1
    merged = obs[0]
    assert merged.deviation == "substantive"
    assert merged.risk_delta == {"direction": "worse", "magnitude": "minor"}
    assert merged.basis == "needs_review"
    assert merged.confidence == 0.6
    assert merged.citation.clause_path == "1"


def test_all_unchanged_group_keeps_deterministic_none() -> None:
    rows = [
        (_terminal_cd("ind", "1", "First part of the clause.", (0, 25)), _dr()),
        (_terminal_cd("ind", "2", "Second part of the clause.", (26, 52)), _dr()),
    ]
    obs = build_observations("doc1", 3, "our_paper", rows, [])
    assert len(obs) == 1
    assert (obs[0].deviation, obs[0].basis) == ("none", "deterministic")
    assert obs[0].risk_delta == {"direction": "neutral", "magnitude": "none"}


def _removed_row(taxonomy_id: str, text: str, path: str = "9") -> ClauseDiff:
    """A first-version clause with no terminal slot, read from draft_v1."""
    return ClauseDiff(
        taxonomy_id=taxonomy_id,
        clause_path_before=path,
        clause_path_after=None,
        kind="removed",
        hunks=(),
        text_before=text,
        text_after="",
        clause_version_before="draft_v1",
        clause_version_after=None,
        char_span_before=(10, 10 + len(text)),
    )


_THEIR_GOV_LAW = "This Agreement is governed by the laws of the State of New York."


def test_removed_before_signing_is_never_signed() -> None:
    """Issue #216: text removed before signing is never signed. Its ORIGIN
    decides what the removal means: their (non-standard) language struck is
    their refused ask — proposed_then_reversed, cited to the draft it was
    read from — whatever paper the deal is on."""
    removed = _removed_row("governing_law", _THEIR_GOV_LAW)
    kept = _terminal_cd("governing_law", "10", _STD["governing_law"], (60, 125))
    for provenance in ("our_paper", "counterparty_paper"):
        dropped: dict[str, int] = {}
        obs = build_observations(
            "doc1",
            3,
            provenance,
            [(removed, _dr()), (kept, _dr())],
            [],
            ordinal_by_vid=_ORDINALS,
            dropped=dropped,
            standard_text_by_tid=_STD,
        )
        assert [(o.outcome, o.full_text, o.citation.version) for o in obs] == [
            ("proposed_then_reversed", _THEIR_GOV_LAW, 1),
            ("signed", _STD["governing_law"], 3),
        ]
        assert dropped == {}


def test_our_standard_clause_replaced_before_signing_is_our_concession() -> None:
    """Issue #216 (owner decision 2026-09-13 (b)): OUR standard language
    struck before signing, replaced by their language, is our concession —
    conceded_before_signing, never proposed_then_reversed — on either
    paper side, since origin decides, not paper."""
    removed = _removed_row("governing_law", _STD["governing_law"])
    replacement = ClauseDiff(
        taxonomy_id="governing_law",
        clause_path_before=None,
        clause_path_after="10",
        kind="added",
        hunks=(),
        text_before="",
        text_after=_THEIR_GOV_LAW,
        clause_version_before=None,
        clause_version_after="signed_final",
        char_span_after=(60, 124),
    )
    for provenance in ("our_paper", "counterparty_paper"):
        obs = build_observations(
            "doc1",
            3,
            provenance,
            [(removed, _dr("substantive", basis="judge")), (replacement, _dr())],
            [],
            ordinal_by_vid=_ORDINALS,
            standard_text_by_tid=_STD,
        )
        assert [(o.outcome, o.full_text, o.citation.version) for o in obs] == [
            (OUTCOME_CONCEDED_BEFORE_SIGNING, _STD["governing_law"], 1),
            ("signed", _THEIR_GOV_LAW, 3),
        ]


def test_our_standard_clause_struck_outright_is_our_concession() -> None:
    """Our standard struck with no replacement: still our concession (and
    the only observation this deal has for the clause)."""
    removed = _removed_row("non_solicit", _STD["non_solicit"])
    other = _terminal_cd("governing_law", "10", _STD["governing_law"], (60, 125))
    obs = build_observations(
        "doc1",
        3,
        "our_paper",
        [(removed, _dr("substantive", basis="judge")), (other, _dr())],
        [],
        ordinal_by_vid=_ORDINALS,
        standard_text_by_tid=_STD,
    )
    assert [(o.taxonomy_id, o.outcome) for o in obs] == [
        ("non_solicit", OUTCOME_CONCEDED_BEFORE_SIGNING),
        ("governing_law", "signed"),
    ]


def test_unsigned_deal_removed_rows_never_concede_nor_refuse() -> None:
    """Issue #216 + #83: with no detected executed copy (has_signed_copy=
    False) a deal is never evidence of a concession. Our standard language
    struck is dropped and counted under DROPPED_STANDARD_REMOVED_UNSIGNED —
    never conceded_before_signing, whether struck outright or replaced.
    Issue #221: nor of a refused ask — their (non-standard) language struck
    is dropped and counted under DROPPED_REFUSED_UNSIGNED, never
    proposed_then_reversed (the terminal is only the last draft). Terminal
    rows are "unsigned"."""
    ours_struck = _removed_row("non_solicit", _STD["non_solicit"], path="8")
    ours_replaced = _removed_row("governing_law", _STD["governing_law"], path="9")
    theirs_struck = _removed_row("ind", "Beta shall indemnify Alpha for all losses.", path="7")
    replacement = _terminal_cd("governing_law", "10", _THEIR_GOV_LAW, (60, 124), kind="added")
    for provenance in ("our_paper", "counterparty_paper"):
        dropped: dict[str, int] = {}
        obs = build_observations(
            "doc1",
            3,
            provenance,
            [
                (ours_struck, _dr("substantive", basis="judge")),
                (ours_replaced, _dr("substantive", basis="judge")),
                (theirs_struck, _dr("substantive", basis="judge")),
                (replacement, _dr()),
            ],
            [],
            has_signed_copy=False,
            ordinal_by_vid=_ORDINALS,
            dropped=dropped,
            standard_text_by_tid=_STD,
        )
        assert [(o.taxonomy_id, o.outcome, o.citation.version) for o in obs] == [
            ("governing_law", "unsigned", 3),
        ]
        assert not [o for o in obs if o.outcome == OUTCOME_CONCEDED_BEFORE_SIGNING]
        assert dropped == {DROPPED_STANDARD_REMOVED_UNSIGNED: 2, DROPPED_REFUSED_UNSIGNED: 1}


def test_standard_node_split_from_a_longer_standard_is_our_language() -> None:
    """A first-version node that is a verbatim piece of our standard clause
    (the segmenter split the standard across nodes) is our language too."""
    piece = "neither party shall solicit the other's employees"
    removed = _removed_row("non_solicit", piece)
    obs = build_observations(
        "doc1", 3, "our_paper", [(removed, _dr())], [], standard_text_by_tid=_STD
    )
    assert [o.outcome for o in obs] == [OUTCOME_CONCEDED_BEFORE_SIGNING]


def test_removed_text_with_no_standard_is_dropped_as_origin_undetermined() -> None:
    """No standard to compare against (no template clause for the
    taxonomy_id, or no standards at all): the removal is neither their
    refused ask nor our concession — dropped and counted, never guessed."""
    removed = _removed_row("confidentiality_term", "Obligations last five years.")
    for standards in (None, {}, _STD, {"confidentiality_term": "   "}):
        dropped: dict[str, int] = {}
        obs = build_observations(
            "doc1",
            3,
            "our_paper",
            [(removed, _dr())],
            [],
            dropped=dropped,
            standard_text_by_tid=standards,
        )
        assert obs == []
        assert dropped == {DROPPED_ORIGIN_UNDETERMINED: 1}


def test_removed_row_whose_text_survives_in_terminal_is_neither_reversed_nor_signed() -> None:
    """Issue #216: a removed row whose own text still occurs in the signed
    terminal (a relocation the aligner left unpaired, basis="alignment") was
    not reversed — its text is already the terminal's signed observation. It
    is not proposed_then_reversed, and it is counted as surviving, never as
    removed before signing."""
    removed = ClauseDiff(
        taxonomy_id="survival",
        clause_path_before="4",
        clause_path_after=None,
        kind="removed",
        hunks=(),
        text_before="Obligations survive for three years.",
        text_after="",
        clause_version_before="draft_v1",
        clause_version_after=None,
        char_span_before=(10, 46),
    )
    kept = _terminal_cd("survival", "9", "Obligations survive for three years.", (80, 116))
    tree = [_node_cc("9", "survival", "Obligations survive for three years.", (80, 116))]
    dropped: dict[str, int] = {}
    obs = build_observations(
        "doc1",
        3,
        "our_paper",
        [(removed, _dr("none", basis="alignment")), (kept, _dr())],
        [],
        ordinal_by_vid=_ORDINALS,
        terminal_clauses=tree,
        terminal_version_id="signed_final",
        dropped=dropped,
    )
    assert [(o.outcome, o.citation.clause_path, o.citation.version) for o in obs] == [
        ("signed", "9", 3)
    ]
    assert dropped == {DROPPED_SURVIVES_IN_TERMINAL: 1}


def test_surviving_removed_row_without_counter_is_still_not_emitted() -> None:
    removed = _cd("ind", kind="removed", text_before="Struck text.", text_after="")
    kept = _cd(
        "gov",
        kind="unchanged",
        text_before="Struck text, kept.",
        text_after="Struck text, kept.",
        path="2",
    )
    obs = build_observations("doc1", "v2", "our_paper", [(removed, _dr()), (kept, _dr())], [])
    assert [(o.taxonomy_id, o.outcome) for o in obs] == [("gov", "signed")]


def test_unclassified_terminal_nodes_stay_one_per_node() -> None:
    """taxonomy_id=None is not a clause type — unclassified nodes are not
    merged into one observation (they feed unclassified coverage per node)."""
    rows = [
        (_terminal_cd(None, "0", "Title block text here.", (0, 22)), _dr()),
        (_terminal_cd(None, "20", "Signature block text.", (900, 921)), _dr()),
    ]
    obs = build_observations("doc1", 3, "our_paper", rows, [])
    assert [o.citation.clause_path for o in obs] == ["0", "20"]
    assert all(o.taxonomy_id is None for o in obs)


def test_unsigned_terminal_is_one_unsigned_observation_per_taxonomy() -> None:
    rows = [
        (_terminal_cd("ind", "1", "First part of the clause.", (0, 25)), _dr()),
        (_terminal_cd("ind", "2", "Second part of the clause.", (26, 52)), _dr()),
    ]
    obs = build_observations("doc1", 3, "our_paper", rows, [], has_signed_copy=False)
    assert [(o.taxonomy_id, o.outcome) for o in obs] == [("ind", "unsigned")]


def test_terminal_tree_node_without_net_diff_row_raises() -> None:
    """Every terminal node is the after slot of exactly one net-diff row
    (clause_differ._version_diff emits one per after slot), so a tree node no
    row reaches means the caller passed the wrong tree — raised, never
    papered over with a fabricated observation."""
    tree = [
        _node_cc("1", "ind", "Indemnity text.", (0, 15)),
        _node_cc("2", "gov", "Governing law text.", (16, 35)),
    ]
    rows = [(_terminal_cd("ind", "1", "Indemnity text.", (0, 15)), _dr())]
    with pytest.raises(ValueError, match="has no net-diff row"):
        build_observations(
            "doc1", 3, "our_paper", rows, [], terminal_clauses=tree, terminal_version_id="v3"
        )


def test_terminal_row_matching_no_tree_node_raises() -> None:
    tree = [_node_cc("1", "ind", "Indemnity text.", (0, 15))]
    rows = [
        (_terminal_cd("ind", "1", "Indemnity text.", (0, 15)), _dr()),
        (_terminal_cd("gov", "2", "Governing law text.", (16, 35)), _dr()),
    ]
    with pytest.raises(ValueError, match="match no terminal node"):
        build_observations(
            "doc1", 3, "our_paper", rows, [], terminal_clauses=tree, terminal_version_id="v3"
        )


def test_removed_row_differing_only_in_filled_in_blanks_survives() -> None:
    """A signature block whose blanks were filled in at signing is not a
    reversal: ``______`` tokenizes as a word under ``\\w+``, but a blank is
    not clause content, so the removed draft block still survives in the
    executed copy (counted, not emitted as rejected)."""
    removed = _cd(
        "counterparts",
        kind="removed",
        text_before="Executed in counterparts.\nBy: __________\nName:",
        text_after="",
        path="27",
    )
    kept = _cd(
        "counterparts",
        kind="added",
        text_after="Executed in counterparts.\nBy: /s/ Pat Doe\nName: Pat Doe",
        path="27",
    )
    dropped: dict[str, int] = {}
    obs = build_observations(
        "doc1", "v2", "our_paper", [(removed, _dr()), (kept, _dr())], [], dropped=dropped
    )
    assert [o.outcome for o in obs] == ["signed"]
    assert dropped == {DROPPED_SURVIVES_IN_TERMINAL: 1}


def test_replaced_clause_whose_words_recur_in_an_unrelated_clause_is_reversed() -> None:
    """Issue #216 fix round 1: survival is judged against ONE clause-sized,
    contiguous stretch of the signed terminal — never the whole document's
    vocabulary. Governing-law text replaced before signing is
    proposed_then_reversed even though every one of its words ("delaware"
    included) recurs somewhere in the signed copy — here in an unchanged
    recital that sits right next to the replacement clause."""
    removed = ClauseDiff(
        taxonomy_id="governing_law",
        clause_path_before="9",
        clause_path_after=None,
        kind="removed",
        hunks=(),
        text_before="This Agreement is governed by the laws of the State of Delaware.",
        text_after="",
        clause_version_before="draft_v1",
        clause_version_after=None,
        char_span_before=(300, 364),
    )
    recital_text = "Acme, Inc., a Delaware corporation, enters into this Agreement."
    replacement_text = "This Agreement is governed by the laws of the State of New York."
    recital = _terminal_cd("parties", "1", recital_text, (0, 63))
    replacement = ClauseDiff(
        taxonomy_id="governing_law",
        clause_path_before=None,
        clause_path_after="2",
        kind="added",
        hunks=(),
        text_before="",
        text_after=replacement_text,
        clause_version_before=None,
        clause_version_after="signed_final",
        char_span_after=(64, 128),
    )
    tree = [
        _node_cc("1", "parties", recital_text, (0, 63)),
        _node_cc("2", "governing_law", replacement_text, (64, 128)),
    ]
    rows = [(removed, _dr()), (recital, _dr()), (replacement, _dr())]
    for kwargs in ({}, {"terminal_clauses": tree, "terminal_version_id": "signed_final"}):
        dropped: dict[str, int] = {}
        obs = build_observations(
            "doc1",
            3,
            "our_paper",
            rows,
            [],
            ordinal_by_vid=_ORDINALS,
            dropped=dropped,
            # Our standard is English law, so the struck Delaware text is
            # their (non-standard) language: a refused ask once removed.
            standard_text_by_tid={"governing_law": "Governed by the laws of England and Wales."},
            **kwargs,
        )
        reversed_obs = [o for o in obs if o.outcome == "proposed_then_reversed"]
        assert [(o.taxonomy_id, o.full_text, o.citation.version) for o in reversed_obs] == [
            ("governing_law", removed.text_before, 1)
        ]
        assert dropped == {}


def test_removed_clause_split_across_adjacent_signed_nodes_survives() -> None:
    """A draft signature block the signed copy splits into several adjacent
    nodes (blanks filled in) survives in that contiguous stretch — the window
    spans more than one node, so a fixed per-node rule would miss it."""
    removed = _cd(
        "counterparts",
        kind="removed",
        text_before=(
            "Executed in counterparts by the parties.\n"
            "By: __________\nName: __________\nTitle: __________"
        ),
        text_after="",
        path="27",
    )
    split = [
        "Executed in counterparts by the parties.",
        "By: /s/ Pat Doe",
        "Name: Pat Doe",
        "Title: Director",
    ]
    rows = [(removed, _dr())] + [
        (_cd("counterparts", kind="added", text_after=text, path=str(28 + i)), _dr())
        for i, text in enumerate(split)
    ]
    dropped: dict[str, int] = {}
    obs = build_observations("doc1", "v2", "our_paper", rows, [], dropped=dropped)
    assert [o.outcome for o in obs] == ["signed"]
    assert dropped == {DROPPED_SURVIVES_IN_TERMINAL: 1}


def _added_row(taxonomy_id: str, text: str, path: str = "10") -> ClauseDiff:
    """A signed-version clause with no first-version slot."""
    return ClauseDiff(
        taxonomy_id=taxonomy_id,
        clause_path_before=None,
        clause_path_after=path,
        kind="added",
        hunks=(),
        text_before="",
        text_after=text,
        clause_version_before=None,
        clause_version_after="signed_final",
        char_span_after=(60, 60 + len(text)),
    )


_STD_COMPELLED = (
    "If compelled by law, the Recipient shall promptly notify the Discloser "
    "and shall disclose only that portion legally required."
)
_NARROWED_COMPELLED = (
    "If compelled by law, the Recipient shall disclose only that portion legally required."
)


def test_v1_narrowing_restored_to_our_standard_by_signing_is_a_refused_ask() -> None:
    """Issue #216 fix round 2: survival is decided on the TEXT, not its word
    set. A first-version clause that narrowed our standard (dropping the
    notice obligation) has every word recur in the signed clause, which
    restored our standard — but the narrowed text itself was changed before
    signing. It is not surviving: it is non-standard language struck, their
    refused ask."""
    removed = _removed_row("compelled_disclosure", _NARROWED_COMPELLED)
    restored = _added_row("compelled_disclosure", _STD_COMPELLED)
    dropped: dict[str, int] = {}
    obs = build_observations(
        "doc1",
        3,
        "our_paper",
        [(removed, _dr("substantive", basis="judge")), (restored, _dr())],
        [],
        ordinal_by_vid=_ORDINALS,
        dropped=dropped,
        standard_text_by_tid={"compelled_disclosure": _STD_COMPELLED},
    )
    assert [(o.outcome, o.full_text, o.citation.version) for o in obs] == [
        ("proposed_then_reversed", _NARROWED_COMPELLED, 1),
        ("signed", _STD_COMPELLED, 3),
    ]
    assert dropped == {}


def test_our_standard_replaced_by_a_superset_of_its_words_is_our_concession() -> None:
    """Our standard of care replaced before signing by a clause that repeats
    every one of its words (and more, in another order) was still replaced:
    the standard's text does not survive, so its removal is our concession,
    never a silent survives_in_terminal drop."""
    std = "The Recipient shall protect Confidential Information with reasonable care."
    replacement = (
        "The Recipient shall protect Confidential Information with the same "
        "degree of care it uses for its own information, and no less than "
        "reasonable care."
    )
    removed = _removed_row("standard_of_care", std)
    signed = _added_row("standard_of_care", replacement)
    dropped: dict[str, int] = {}
    obs = build_observations(
        "doc1",
        3,
        "our_paper",
        [(removed, _dr("substantive", basis="judge")), (signed, _dr())],
        [],
        ordinal_by_vid=_ORDINALS,
        dropped=dropped,
        standard_text_by_tid={"standard_of_care": std},
    )
    assert [(o.outcome, o.full_text) for o in obs] == [
        (OUTCOME_CONCEDED_BEFORE_SIGNING, std),
        ("signed", replacement),
    ]
    assert dropped == {}


def test_removed_text_verbatim_inside_a_signed_clause_still_survives() -> None:
    """The text test keeps a relocation surviving: the removed clause occurs
    verbatim (case and punctuation aside) inside the signed clause."""
    removed = _removed_row("survival", "Obligations survive for three years")
    signed = _added_row("survival", "OBLIGATIONS SURVIVE FOR THREE YEARS; return on request.")
    dropped: dict[str, int] = {}
    obs = build_observations(
        "doc1",
        3,
        "our_paper",
        [(removed, _dr("none", basis="alignment")), (signed, _dr())],
        [],
        dropped=dropped,
        standard_text_by_tid={"survival": "Obligations survive for five years."},
    )
    assert [o.outcome for o in obs] == ["signed"]
    assert dropped == {DROPPED_SURVIVES_IN_TERMINAL: 1}


def test_struck_text_equal_to_a_later_template_node_is_our_language() -> None:
    """Issue #216 fix round 2: our standard is EVERY template node carrying
    the taxonomy_id. Text struck before signing that equals the template's
    SECOND insurance node is our standard language — our concession — even
    though it shares nothing with the first node."""
    first = "Supplier shall maintain commercial general liability insurance."
    second = "Supplier shall name the Customer as an additional insured on every policy."
    removed = _removed_row("insurance", second)
    kept = _terminal_cd("insurance", "2", first, (0, len(first)))
    for standards in ({"insurance": [first, second]}, {"insurance": (first, second)}):
        obs = build_observations(
            "doc1",
            3,
            "our_paper",
            [(removed, _dr("substantive", basis="judge")), (kept, _dr())],
            [],
            ordinal_by_vid=_ORDINALS,
            standard_text_by_tid=standards,
        )
        assert [(o.outcome, o.full_text) for o in obs] == [
            (OUTCOME_CONCEDED_BEFORE_SIGNING, second),
            ("signed", first),
        ]
    # Against the first node alone, the same text would read as their ask.
    obs = build_observations(
        "doc1",
        3,
        "our_paper",
        [(removed, _dr("substantive", basis="judge")), (kept, _dr())],
        [],
        standard_text_by_tid={"insurance": first},
    )
    assert obs[0].outcome == "proposed_then_reversed"


def test_whole_multi_node_standard_reworded_in_one_node_is_our_language() -> None:
    """A struck clause rendering our whole multi-node standard as one node
    (near-identically) matches the standard's nodes joined in order."""
    nodes = [
        "Supplier shall maintain commercial general liability insurance.",
        "Supplier shall name the Customer as an additional insured on every policy.",
    ]
    struck = (
        "Supplier shall maintain commercial general liability insurance; "
        "Supplier shall name the Customer as an additional insured on every policy"
    )
    obs = build_observations(
        "doc1",
        3,
        "our_paper",
        [(_removed_row("insurance", struck), _dr())],
        [],
        standard_text_by_tid={"insurance": nodes},
    )
    assert [o.outcome for o in obs] == [OUTCOME_CONCEDED_BEFORE_SIGNING]


_ASSIGN_STD = (
    "Neither party may assign this Agreement, except to an Affiliate, without "
    "the prior written consent of the other party, and any attempted "
    "assignment in breach of this Section is void and of no effect as to an "
    "assignee."
)
_TRADE_SECRET_STD = (
    "Trade secrets of the Disclosing Party remain protected for as long as "
    "they remain trade secrets under applicable law, notwithstanding the "
    "expiry or termination of this Agreement for any reason whatsoever."
)


@pytest.mark.parametrize(
    ("tid", "standard", "struck"),
    [
        # Negation flip: "Neither" -> "Either".
        ("assignment", _ASSIGN_STD, _ASSIGN_STD.replace("Neither", "Either", 1)),
        # Deleted carve-out: the Affiliate exception struck from mid-clause.
        (
            "assignment",
            _ASSIGN_STD,
            _ASSIGN_STD.replace(", except to an Affiliate,", ""),
        ),
        # Protection negated: "remain protected" -> "are not protected".
        (
            "trade_secrets",
            _TRADE_SECRET_STD,
            _TRADE_SECRET_STD.replace("remain protected", "are not protected"),
        ),
    ],
    ids=["negation-flip", "deleted-carve-out", "protection-negated"],
)
def test_struck_first_version_text_that_edits_our_clause_is_their_ask(
    tid: str, standard: str, struck: str
) -> None:
    """Issue #229: the origin test is the exact standard check #220 put on
    the consumer path, not the order-blind token Jaccard it retired. A first
    draft that negates our clause or deletes its carve-out, struck before
    signing in favour of our standard, is THEIR refused ask — never our
    concession — even though the old 0.92 Jaccard called it our language."""
    from playbook_engine.deviation_classifier import (  # noqa: PLC0415
        REWORDED_EQUIVALENT_THRESHOLD,
        _text_jaccard,
    )

    # The case this test exists for: the retired similarity bar absorbed it.
    assert _text_jaccard(struck, standard) >= REWORDED_EQUIVALENT_THRESHOLD
    signed = _added_row(tid, standard)
    dropped: dict[str, int] = {}
    obs = build_observations(
        "doc1",
        3,
        "our_paper",
        [(_removed_row(tid, struck), _dr()), (signed, _dr())],
        [],
        ordinal_by_vid=_ORDINALS,
        dropped=dropped,
        standard_text_by_tid={tid: standard},
    )
    assert [(o.outcome, o.full_text) for o in obs] == [
        ("proposed_then_reversed", struck),
        ("signed", standard),
    ]
    assert dropped == {}


_INSURANCE_NODES = [
    "AlphaCorp shall maintain commercial general liability insurance.",
    "AlphaCorp shall name the Customer as an additional insured on every "
    "policy. AlphaCorp shall deliver a certificate of insurance on request.",
]


@pytest.mark.parametrize(
    ("struck", "party_names", "expected"),
    [
        # One sentence of the second node, verbatim: a fragment of our
        # standard, so its removal is our concession.
        (
            "AlphaCorp shall name the Customer as an additional insured on every policy.",
            (),
            OUTCOME_CONCEDED_BEFORE_SIGNING,
        ),
        # A run straddling the node boundary is still one contiguous
        # fragment of the standard (nodes joined in document order).
        (
            "AlphaCorp shall maintain commercial general liability insurance. "
            "AlphaCorp shall name the Customer as an additional insured on every policy.",
            (),
            OUTCOME_CONCEDED_BEFORE_SIGNING,
        ),
        # The deal names the party differently: party names are neutralized
        # in the fragment test exactly as in the exact check ...
        (
            "Beta Supplies LLC shall name the Customer as an additional insured on every policy.",
            ("AlphaCorp", "Beta Supplies LLC"),
            OUTCOME_CONCEDED_BEFORE_SIGNING,
        ),
        # ... and only for KNOWN party names.
        (
            "Beta Supplies LLC shall name the Customer as an additional insured on every policy.",
            (),
            "proposed_then_reversed",
        ),
        # A fragment must sit on word boundaries of the standard: "polic"
        # is not a fragment of "policy".
        (
            "AlphaCorp shall name the Customer as an additional insured on every polic",
            (),
            "proposed_then_reversed",
        ),
        # A fragment with a negator inserted is not contained in the standard.
        (
            "AlphaCorp shall not name the Customer as an additional insured on every policy.",
            (),
            "proposed_then_reversed",
        ),
    ],
    ids=[
        "one-sentence-of-node-2",
        "run-across-node-boundary",
        "known-party-name-neutralized",
        "unknown-party-name-not-neutralized",
        "partial-word-is-not-a-fragment",
        "inserted-negator",
    ],
)
def test_fragment_of_a_two_node_standard_is_our_language(
    struck: str, party_names: tuple[str, ...], expected: str
) -> None:
    """Issue #229: the containment rule for a fragment of a multi-node
    standard survives the switch to the exact check, and runs on
    ``normalize_for_standard`` output (same party names) so party names are
    handled exactly as in ``is_standard_text``."""
    kept = _terminal_cd("insurance", "2", "Customer may request proof of coverage.", (0, 40))
    obs = build_observations(
        "doc1",
        3,
        "our_paper",
        [(_removed_row("insurance", struck), _dr()), (kept, _dr())],
        [],
        ordinal_by_vid=_ORDINALS,
        standard_text_by_tid={"insurance": _INSURANCE_NODES},
        party_names=party_names,
    )
    assert [(o.outcome, o.full_text) for o in obs] == [
        (expected, struck),
        ("signed", "Customer may request proof of coverage."),
    ]


def test_whole_standard_with_other_party_names_is_our_language() -> None:
    """Issue #229: the exact branch of the origin test neutralizes known
    party names exactly as the consumer-path ``standard`` fact does."""
    std = "AlphaCorp shall keep the Confidential Information of the other party secret."
    struck = "Beta Supplies LLC shall keep the Confidential Information of the other party secret."
    for party_names, expected in (
        (("AlphaCorp", "Beta Supplies LLC"), OUTCOME_CONCEDED_BEFORE_SIGNING),
        ((), "proposed_then_reversed"),
    ):
        obs = build_observations(
            "doc1",
            3,
            "our_paper",
            [(_removed_row("confidentiality", struck), _dr())],
            [],
            standard_text_by_tid={"confidentiality": std},
            party_names=party_names,
        )
        assert [o.outcome for o in obs] == [expected]


def test_reversals_sharing_a_path_from_different_drafts_are_each_emitted() -> None:
    """Two distinct proposals reversed out of the same (taxonomy_id, path) in
    different drafts are separate evidence: both are emitted, each with its
    own proposed text and draft citation. Only a true duplicate record (same
    draft and text) is skipped. detect_reversals sets version_inserted to a
    diff's version_after, so it is always a middle draft — never the first
    version nor the signed terminal."""
    ordinals = {"draft_v1": 1, "draft_v2": 2, "draft_v3": 3, "signed_final": 4}
    v2 = ReversalRecord(
        taxonomy_id=None,
        clause_path="21",
        version_inserted="draft_v2",
        version_removed="signed_final",
        proposed_text="twelve 12 months non solicit",
    )
    v3 = ReversalRecord(
        taxonomy_id=None,
        clause_path="21",
        version_inserted="draft_v3",
        version_removed="signed_final",
        proposed_text="six 6 months directly involved",
    )
    obs = build_observations("doc1", 4, "our_paper", [], [v2, v3, v2], ordinal_by_vid=ordinals)
    assert [(o.outcome, o.full_text, o.citation.version) for o in obs] == [
        ("proposed_then_reversed", "twelve 12 months non solicit", 2),
        ("proposed_then_reversed", "six 6 months directly involved", 3),
    ]
    assert len({o.observation_id for o in obs}) == 2


# ---------------------------------------------------------------------------
# alignment_confidence reaches the observation (issue #222)
# ---------------------------------------------------------------------------


def test_alignment_confidence_carried_from_real_alignment_to_observations() -> None:
    """Drive the real producers (align_versions -> diff_aligned ->
    detect_reversals) and check the binding similarity lands on the
    observations, serialized as x_alignment_confidence and read back."""
    from playbook_engine.clause_aligner import ALIGNMENT_AMBIGUITY_THRESHOLD, align_versions
    from playbook_engine.clause_differ import diff_aligned
    from playbook_engine.pipeline import _restore_observations
    from playbook_engine.reversal_detector import detect_reversals

    def cc(path: str, tid: str, text: str) -> ClassifiedClause:
        return ClassifiedClause(
            node=ClauseNode(clause_path=path, heading=None, text=text, char_span=(0, len(text))),
            classification=ClauseClassification(
                taxonomy_id=tid, confidence=1.0, basis="exact_match"
            ),
        )

    base = "Alice Corp shall indemnify Beta Ltd for all losses"
    versions = [
        ("v1", [cc("1", "ind", base + ".")]),
        ("v2", [cc("1", "ind", base + ", including consequential damages.")]),
        (
            "v3",
            [
                cc("1", "ind", base + "."),
                cc("2", "notices", "Notices go to the registered office by courier."),
            ],
        ),
    ]
    order = [vid for vid, _ in versions]
    alignments = align_versions(versions)
    doc_diff = diff_aligned(alignments, order)
    reversals = detect_reversals(doc_diff)
    assert len(reversals) == 1

    net = list(doc_diff.net.diffs)
    obs = build_observations(
        "doc-1",
        3,
        "our_paper",
        [(cd, _dr()) for cd in net],
        reversals,
        ordinal_by_vid={vid: i + 1 for i, vid in enumerate(order)},
        terminal_clauses=versions[-1][1],
        terminal_version_id="v3",
        standard_text_by_tid=_STD,
        deterministic_deviations=True,
    )
    by_outcome = {(o.taxonomy_id, o.outcome): o for o in obs}

    signed_ind = by_outcome[("ind", "signed")]
    bound = next(cd for cd in net if cd.taxonomy_id == "ind").alignment_confidence
    assert bound is not None and bound >= ALIGNMENT_AMBIGUITY_THRESHOLD
    assert signed_ind.alignment_confidence == bound
    assert signed_ind.to_dict()["x_alignment_confidence"] == round(bound, 6)

    refused = by_outcome[("ind", "proposed_then_reversed")]
    assert refused.alignment_confidence == reversals[0].alignment_confidence
    assert refused.alignment_confidence is not None

    # A clause present only in the terminal binds nothing across versions.
    notices = by_outcome[("notices", "signed")]
    assert notices.alignment_confidence is None
    assert "x_alignment_confidence" not in notices.to_dict()

    restored = {o.observation_id: o for o in _restore_observations([o.to_dict() for o in obs])}
    for o in obs:
        assert restored[o.observation_id].alignment_confidence == (
            round(o.alignment_confidence, 6) if o.alignment_confidence is not None else None
        )


def test_appended_carve_out_stays_modified_and_keeps_tracked_attribution() -> None:
    """Fix round 1 regression (issue #222): our tracked carve-out appended to
    a clause roughly halves its token-set Jaccard, but the clause is the same
    clause — the localized-edit rescue (Jaccard >= 0.5, the two token
    sequences differing by exactly one contiguous insert) must keep it on one
    ``modified`` row, so the signed
    observation keeps the insertion's tracked-change attribution (author,
    date, type) and proposed_by "us" instead of degrading to an add/remove
    pair with no hunks and nothing to attribute."""
    from playbook_engine.clause_aligner import ALIGNMENT_AMBIGUITY_THRESHOLD, align_versions
    from playbook_engine.clause_differ import diff_aligned
    from playbook_engine.docx_ingester import TrackedChange, TrackedChanges
    from playbook_engine.pipeline import _attribution_for_diff

    def cc(text: str) -> ClassifiedClause:
        return ClassifiedClause(
            node=ClauseNode(clause_path="9", heading=None, text=text, char_span=(0, len(text))),
            classification=ClauseClassification(
                taxonomy_id="limitation_of_liability", confidence=1.0, basis="exact_match"
            ),
        )

    base = "Each party's aggregate liability under this Agreement shall not exceed $50,000"
    carve_out = (
        ", except that this limitation shall not apply to a breach of either "
        "party's confidentiality obligations or to claims of gross negligence"
    )
    v1_text, v2_text = base + ".", base + carve_out + "."
    from playbook_engine.clause_aligner import _jaccard, _tokens

    assert _jaccard(_tokens(v1_text), _tokens(v2_text)) < ALIGNMENT_AMBIGUITY_THRESHOLD

    versions = [("v1", [cc(v1_text)]), ("v2", [cc(v2_text)])]
    alignments = align_versions(versions)
    assert len(alignments) == 1, "the appended carve-out must not split the clause"
    net = list(diff_aligned(alignments, ["v1", "v2"]).net.diffs)
    assert [d.kind for d in net] == ["modified"]

    start = len(base)
    tracked = TrackedChanges(
        document_id="doc-1",
        version="v2",
        changes=[
            TrackedChange(
                change_type="insertion",
                author="Our Counsel",
                date="2024-02-03T10:00:00Z",
                text=carve_out,
                clause_path="9",
                char_span=(start, start + len(carve_out)),
            )
        ],
    )
    attribution = _attribution_for_diff(net[0], tracked, single_round=True)
    assert attribution is not None
    assert (attribution.author, attribution.tracked_type) == ("Our Counsel", "insertion")

    obs = build_observations(
        "doc-1",
        2,
        "our_paper",
        [(net[0], _dr("substantive"))],
        [],
        attributions=[attribution],
        our_party_aliases=["AlphaCorp"],
        our_authors=["Our Counsel"],
        ordinal_by_vid={"v1": 1, "v2": 2},
        terminal_clauses=versions[-1][1],
        terminal_version_id="v2",
        standard_text_by_tid={"limitation_of_liability": v1_text},
        deterministic_deviations=True,
    )
    signed = [o for o in obs if o.outcome == "signed"]
    assert len(signed) == 1
    assert signed[0].proposed_by == "us"
    assert signed[0].attribution is not None
    assert signed[0].attribution.author == "Our Counsel"
    assert signed[0].attribution.date == "2024-02-03T10:00:00Z"
    assert signed[0].attribution.tracked_type == "insertion"
    # The localized-edit rescue bound the row; its confidence is the honest
    # token-set Jaccard (below the primary threshold), never a rescue score.
    conf = signed[0].alignment_confidence
    assert conf is not None
    assert conf == pytest.approx(_jaccard(_tokens(v1_text), _tokens(v2_text)))
    assert conf < ALIGNMENT_AMBIGUITY_THRESHOLD
