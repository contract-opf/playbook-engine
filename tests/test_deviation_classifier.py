"""Tests for deviation + risk-delta classifier (L4, issue #20 / P2.6).

SECURITY NOTE: All fixtures are programmatically constructed with synthetic
text.  No real agreements are referenced.  Fictional party names only
('Alice Corp', 'Beta Ltd').
"""

from __future__ import annotations

import pytest

from playbook_engine.clause_differ import ClauseDiff, TextHunk
from playbook_engine.deviation_classifier import (
    REWORDED_EQUIVALENT_THRESHOLD,
    DeviationResult,
    RiskDelta,
    _text_jaccard,
    assess_deviations,
    assess_deviations_deterministic,
    is_standard_text,
    normalize_for_standard,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cd(
    kind: str,
    text_before: str = "",
    text_after: str = "",
    taxonomy_id: str = "ind",
    path: str = "1",
) -> ClauseDiff:
    hunk = (
        ()
        if kind in ("added", "removed", "unchanged")
        else (TextHunk(kind="replace", old_text=text_before, new_text=text_after),)
    )
    return ClauseDiff(
        taxonomy_id=taxonomy_id,
        clause_path_before=path if kind != "added" else None,
        clause_path_after=path if kind != "removed" else None,
        kind=kind,
        hunks=hunk,
        text_before=text_before,
        text_after=text_after,
    )


_NEUTRAL_ZERO = RiskDelta(direction="neutral", magnitude="none")
_WORSE_MATERIAL = RiskDelta(direction="worse", magnitude="material")
_NEUTRAL_JUDGE = DeviationResult(deviation="none", risk_delta=_NEUTRAL_ZERO, basis="judge")


# ---------------------------------------------------------------------------
# MockDeviationJudge — deterministic, text-based
# ---------------------------------------------------------------------------


class MockDeviationJudge:
    """Deterministic judge.

    Accepts the hunk-payload dict format (``{"hunk": "..."}``).
    Classifies by inspecting the hunk string:
    - If '[AFTER]' section contains 'consequential' → substantive, worse/material
    - Otherwise → reworded_equivalent, neutral/none
    """

    def __init__(self) -> None:
        self.received_items: list[dict[str, str]] = []

    def assess_batch(
        self,
        items: list[dict[str, str]],
        our_standard: str,
    ) -> list[DeviationResult]:
        self.received_items = list(items)
        results = []
        for item in items:
            hunk = item.get("hunk", "")
            after_section = hunk.split("[AFTER]", 1)[-1] if "[AFTER]" in hunk else hunk
            if "consequential" in after_section.lower():
                results.append(
                    DeviationResult(
                        deviation="substantive",
                        risk_delta=_WORSE_MATERIAL,
                        basis="judge",
                        rationale="Consequential damages expansion is material.",
                    )
                )
            else:
                results.append(
                    DeviationResult(
                        deviation="reworded_equivalent",
                        risk_delta=_NEUTRAL_ZERO,
                        basis="judge",
                        rationale="Phrasing differs but substance unchanged.",
                    )
                )
        return results


class RaisingJudge:
    def assess_batch(self, items: list[dict[str, str]], our_standard: str) -> list[DeviationResult]:
        raise RuntimeError("LLM unavailable")


class WrongLengthJudge:
    def assess_batch(self, items: list[dict[str, str]], our_standard: str) -> list[DeviationResult]:
        return []


class BadBasisJudge:
    def assess_batch(self, items: list[dict[str, str]], our_standard: str) -> list[DeviationResult]:
        return [
            DeviationResult(deviation="none", risk_delta=_NEUTRAL_ZERO, basis="deterministic")
            for _ in items
        ]


# ---------------------------------------------------------------------------
# RiskDelta dataclass
# ---------------------------------------------------------------------------


def test_risk_delta_valid() -> None:
    rd = RiskDelta(direction="worse", magnitude="material")
    assert rd.direction == "worse"
    assert rd.magnitude == "material"


def test_risk_delta_invalid_direction() -> None:
    with pytest.raises(ValueError, match="direction"):
        RiskDelta(direction="sideways", magnitude="none")


def test_risk_delta_invalid_magnitude() -> None:
    with pytest.raises(ValueError, match="magnitude"):
        RiskDelta(direction="neutral", magnitude="enormous")


def test_risk_delta_neutral_must_have_none_magnitude() -> None:
    with pytest.raises(ValueError):
        RiskDelta(direction="neutral", magnitude="minor")


def test_risk_delta_to_dict() -> None:
    rd = RiskDelta(direction="worse", magnitude="material")
    d = rd.to_dict()
    assert d == {"direction": "worse", "magnitude": "material"}


# ---------------------------------------------------------------------------
# DeviationResult dataclass
# ---------------------------------------------------------------------------


def test_deviation_result_valid() -> None:
    dr = DeviationResult(
        deviation="substantive",
        risk_delta=RiskDelta(direction="worse", magnitude="material"),
        basis="judge",
        rationale="Caps removed.",
    )
    assert dr.deviation == "substantive"
    assert dr.basis == "judge"


def test_deviation_result_invalid_deviation() -> None:
    with pytest.raises(ValueError, match="deviation"):
        DeviationResult(deviation="kind_of_bad", risk_delta=_NEUTRAL_ZERO, basis="judge")


def test_deviation_result_invalid_basis() -> None:
    with pytest.raises(ValueError, match="basis"):
        DeviationResult(deviation="none", risk_delta=_NEUTRAL_ZERO, basis="unknown")


def test_deviation_result_to_dict() -> None:
    dr = DeviationResult(
        deviation="substantive",
        risk_delta=RiskDelta(direction="worse", magnitude="minor"),
        basis="judge",
        rationale="Risk increased.",
    )
    d = dr.to_dict()
    assert d["deviation"] == "substantive"
    assert d["risk_delta"] == {"direction": "worse", "magnitude": "minor"}
    assert d["basis"] == "judge"
    assert d["rationale"] == "Risk increased."


# ---------------------------------------------------------------------------
# assess_deviations: deterministic fast path
# ---------------------------------------------------------------------------


def test_unchanged_clause_is_deterministic() -> None:
    """Unchanged clauses must not be sent to the judge."""
    cd = _cd("unchanged", "same text", "same text")
    results = assess_deviations([cd], "same text", MockDeviationJudge())
    assert len(results) == 1
    _, dr = results[0]
    assert dr.deviation == "none"
    assert dr.basis == "deterministic"
    assert dr.risk_delta == _NEUTRAL_ZERO


def test_multiple_unchanged_clauses_matching_template_all_deterministic() -> None:
    """Unchanged clauses that also match our_standard stay deterministic."""
    diffs = [
        _cd("unchanged", "same text", "same text", path="1"),
        _cd("unchanged", "same text", "same text", path="2"),
    ]
    results = assess_deviations(diffs, "same text", MockDeviationJudge())
    for _, dr in results:
        assert dr.basis == "deterministic"
        assert dr.deviation == "none"


def test_unchanged_clause_with_no_template_stays_deterministic() -> None:
    """Unchanged clause with an empty our_standard (no template configured,
    or no template clause for this taxonomy_id) has nothing to compare
    against — stays deterministic, not routed to the judge."""
    cd = _cd("unchanged", "some clause text", "some clause text")
    judge = MockDeviationJudge()
    results = assess_deviations([cd], "", judge)
    _, dr = results[0]
    assert dr.basis == "deterministic"
    assert dr.deviation == "none"
    assert judge.received_items == []


# ---------------------------------------------------------------------------
# Issue #103 — unchanged-in-negotiation clauses diffed against the template
# ---------------------------------------------------------------------------


def test_unchanged_clause_differing_from_template_routes_to_judge() -> None:
    """A clause unchanged across the whole negotiation trail (or the sole
    version of a single-version document) must still be checked against the
    canonical template — previously this was hardcoded deviation='none'
    without ever comparing to our_standard (issue #103)."""
    cd = _cd(
        "unchanged",
        "Alice Corp shall indemnify Beta Ltd for direct losses only.",
        "Alice Corp shall indemnify Beta Ltd for direct losses only.",
    )
    judge = MockDeviationJudge()
    results = assess_deviations(
        [cd], "Alice Corp shall indemnify Beta Ltd for consequential damages.", judge
    )
    _, dr = results[0]
    # Judge must have been called — the clause was never actually unassessed.
    assert judge.received_items, (
        "judge must be called for an unchanged clause that differs from the template"
    )
    assert dr.basis == "judge"


def test_unchanged_clause_differing_from_template_hunk_compares_template() -> None:
    """The hunk sent to the judge for this case must contain the template
    text, not a no-op before==after diff of the clause against itself."""
    clause_text = "Alice Corp shall indemnify Beta Ltd for direct losses only."
    template_text = "Alice Corp shall indemnify Beta Ltd for consequential damages."
    cd = _cd("unchanged", clause_text, clause_text)
    judge = MockDeviationJudge()
    assess_deviations([cd], template_text, judge)

    assert len(judge.received_items) == 1
    hunk = judge.received_items[0]["hunk"]
    assert template_text.split("\n")[0] in hunk or "consequential" in hunk.lower()
    assert clause_text.split("\n")[0] in hunk or "direct losses" in hunk.lower()


def test_unchanged_clause_matching_template_via_jaccard_stays_deterministic() -> None:
    """An unchanged clause that is a near-identical (Jaccard-above-threshold)
    match to our_standard stays deterministic — not every wording difference
    should trigger a judge call."""
    base = "Alice Corp shall indemnify Beta Ltd for all direct losses and damages arising"
    similar = (
        "Alice Corp shall indemnify Beta Ltd for all direct losses and damages arising hereunder"
    )
    assert _text_jaccard(base, similar) >= REWORDED_EQUIVALENT_THRESHOLD

    judge = MockDeviationJudge()
    cd = _cd("unchanged", base, base)
    results = assess_deviations([cd], similar, judge)
    _, dr = results[0]
    assert dr.basis == "deterministic"
    assert dr.deviation == "none"
    assert judge.received_items == []


# ---------------------------------------------------------------------------
# assess_deviations: LLM slow path
# ---------------------------------------------------------------------------


def test_modified_clause_goes_to_judge() -> None:
    cd = _cd("modified", "Alice Corp indemnifies.", "Alice Corp fully indemnifies.", path="1")
    results = assess_deviations([cd], "Alice Corp indemnifies.", MockDeviationJudge())
    _, dr = results[0]
    assert dr.basis == "judge"


def test_batch_items_carry_clause_context() -> None:
    """Issue #109: judge batch items must carry taxonomy_id and clause_path

    so a pending deviation record is traceable to the clause it came from,
    not just a bare BEFORE/AFTER hunk.
    """
    cd = _cd(
        "modified",
        "Alice Corp indemnifies.",
        "Alice Corp fully indemnifies.",
        taxonomy_id="indemnification",
        path="3.2",
    )
    judge = MockDeviationJudge()
    assess_deviations([cd], "Alice Corp indemnifies.", judge)

    assert len(judge.received_items) == 1
    item = judge.received_items[0]
    assert item["taxonomy_id"] == "indemnification"
    assert item["clause_path"] == "3.2"


def test_batch_items_carry_version_context() -> None:
    """Issue #166: judge batch items must carry version_from/version_to (the
    normalized-tree version ids) so a pending deviation record can be traced
    back to the specific $OUT/normalized/<document_id>/*.clauses.json files a
    relocation-triage reviewer needs to open — the pending hunk alone carries
    no version identifiers.
    """
    cd = ClauseDiff(
        taxonomy_id="indemnification",
        clause_path_before="3.2",
        clause_path_after="3.3",
        kind="modified",
        hunks=(
            TextHunk(
                kind="replace",
                old_text="Alice Corp indemnifies.",
                new_text="Alice Corp fully indemnifies.",
            ),
        ),
        text_before="Alice Corp indemnifies.",
        text_after="Alice Corp fully indemnifies.",
        clause_version_before="v1",
        clause_version_after="v2",
    )
    judge = MockDeviationJudge()
    assess_deviations([cd], "Alice Corp indemnifies.", judge)

    assert len(judge.received_items) == 1
    item = judge.received_items[0]
    assert item["version_from"] == "v1"
    assert item["version_to"] == "v2"


def test_batch_items_default_version_context_to_empty_string() -> None:
    """When a ClauseDiff carries no clause_version_before/after (e.g. built
    outside diff_aligned), batch items still carry version_from/version_to
    as empty strings rather than omitting the keys — same convention as
    taxonomy_id/clause_path."""
    cd = _cd("modified", "Alice Corp indemnifies.", "Alice Corp fully indemnifies.", path="1")
    judge = MockDeviationJudge()
    assess_deviations([cd], "Alice Corp indemnifies.", judge)

    item = judge.received_items[0]
    assert item["version_from"] == ""
    assert item["version_to"] == ""


def test_batch_items_carry_document_id_when_provided() -> None:
    """Issue #109: document_id is threaded onto batch items when the caller supplies one."""
    cd = _cd("modified", "Alice Corp indemnifies.", "Alice Corp fully indemnifies.", path="1")
    judge = MockDeviationJudge()
    assess_deviations([cd], "Alice Corp indemnifies.", judge, document_id="doc-42")

    assert judge.received_items[0]["document_id"] == "doc-42"


def test_batch_items_omit_document_id_when_not_provided() -> None:
    """document_id is absent from the item dict (not an empty string) when the
    caller passes no document_id — keeps existing callers/tests untouched."""
    cd = _cd("modified", "Alice Corp indemnifies.", "Alice Corp fully indemnifies.", path="1")
    judge = MockDeviationJudge()
    assess_deviations([cd], "Alice Corp indemnifies.", judge)

    assert "document_id" not in judge.received_items[0]


def test_modified_with_consequential_is_substantive_worse() -> None:
    """Acceptance criterion: clear worse/material concession detected."""
    cd = _cd(
        "modified",
        "Alice Corp shall indemnify Beta Ltd.",
        "Alice Corp shall indemnify Beta Ltd for consequential damages.",
    )
    results = assess_deviations([cd], "Alice Corp shall indemnify Beta Ltd.", MockDeviationJudge())
    _, dr = results[0]
    assert dr.deviation == "substantive"
    assert dr.risk_delta.direction == "worse"
    assert dr.risk_delta.magnitude == "material"


def test_modified_reword_is_neutral_reworded_equivalent() -> None:
    """Acceptance criterion: neutral reword does not flag as worse."""
    cd = _cd(
        "modified",
        "Alice Corp shall indemnify Beta Ltd for losses.",
        "Alice Corp shall fully indemnify Beta Ltd for losses.",
    )
    results = assess_deviations(
        [cd], "Alice Corp shall indemnify Beta Ltd for losses.", MockDeviationJudge()
    )
    _, dr = results[0]
    assert dr.deviation == "reworded_equivalent"
    assert dr.risk_delta.direction == "neutral"


def test_added_clause_goes_to_judge() -> None:
    cd = _cd("added", text_after="Newly added indemnification provision.", path="5")
    results = assess_deviations([cd], "Standard indemnification text.", MockDeviationJudge())
    _, dr = results[0]
    assert dr.basis == "judge"


def test_removed_clause_goes_to_judge() -> None:
    cd = _cd("removed", text_before="Standard indemnification text.", path="1")
    results = assess_deviations([cd], "Standard indemnification text.", MockDeviationJudge())
    _, dr = results[0]
    assert dr.basis == "judge"


# ---------------------------------------------------------------------------
# Alignment-artifact fast path (issue #167)
# ---------------------------------------------------------------------------


def test_removed_clause_present_in_counterpart_is_alignment_artifact() -> None:
    """A 'removed' clause whose text also occurs in the after-version's clause
    tree is a deterministic alignment artifact — no judge call."""
    cd = _cd(
        "removed",
        text_before="Notices shall be delivered by certified mail.",
        path="4",
    )
    judge = MockDeviationJudge()
    results = assess_deviations(
        [cd],
        "standard",
        judge,
        counterpart_clause_texts=(
            frozenset(),
            frozenset({"Section 9. Notices shall be delivered by certified mail. See Exhibit A."}),
        ),
    )
    assert judge.received_items == []
    _, dr = results[0]
    assert dr.deviation == "none"
    assert dr.basis == "alignment"
    assert dr.risk_delta.direction == "neutral"
    assert dr.risk_delta.magnitude == "none"


def test_added_clause_present_in_counterpart_is_alignment_artifact() -> None:
    """An 'added' clause whose text also occurs in the before-version's clause
    tree is a deterministic alignment artifact — no judge call."""
    cd = _cd(
        "added",
        text_after="Notices shall be delivered by certified mail.",
        path="7",
    )
    judge = MockDeviationJudge()
    results = assess_deviations(
        [cd],
        "standard",
        judge,
        counterpart_clause_texts=(
            frozenset({"1. Notices shall be delivered by certified mail."}),
            frozenset(),
        ),
    )
    assert judge.received_items == []
    _, dr = results[0]
    assert dr.deviation == "none"
    assert dr.basis == "alignment"


def test_removed_clause_normalization_tolerates_numbering_and_punctuation() -> None:
    """Containment survives renumbering/punctuation differences — the same
    tolerance clause_aligner._normalize applies for move-matching."""
    cd = _cd(
        "removed",
        text_before="Confidential Information; each party shall keep it secret.",
        path="2",
    )
    judge = MockDeviationJudge()
    results = assess_deviations(
        [cd],
        "standard",
        judge,
        counterpart_clause_texts=(
            frozenset(),
            frozenset({"3.2  confidential information  each party shall keep it secret!!"}),
        ),
    )
    assert judge.received_items == []
    _, dr = results[0]
    assert dr.basis == "alignment"


def test_removed_clause_not_present_in_counterpart_still_goes_to_judge() -> None:
    """counterpart_clause_texts supplied but no containment match → judge, as before."""
    cd = _cd("removed", text_before="Governing law is Delaware.", path="9")
    judge = MockDeviationJudge()
    results = assess_deviations(
        [cd],
        "standard",
        judge,
        counterpart_clause_texts=(frozenset(), frozenset({"Entirely unrelated clause text."})),
    )
    _, dr = results[0]
    assert dr.basis == "judge"
    assert len(judge.received_items) == 1


def test_added_removed_default_to_judge_when_counterpart_clause_texts_omitted() -> None:
    """Backward compatibility: omitting counterpart_clause_texts (the default)
    disables the fast path entirely, even for text that would otherwise match."""
    cd = _cd("removed", text_before="Notices shall be delivered by certified mail.", path="4")
    judge = MockDeviationJudge()
    results = assess_deviations([cd], "standard", judge)
    _, dr = results[0]
    assert dr.basis == "judge"
    assert len(judge.received_items) == 1


def test_alignment_artifact_empty_text_never_matches() -> None:
    """An empty removed-clause text must not vacuously match every counterpart
    (it would be a substring of everything)."""
    cd = _cd("removed", text_before="", path="1")
    judge = MockDeviationJudge()
    results = assess_deviations(
        [cd],
        "standard",
        judge,
        counterpart_clause_texts=(frozenset(), frozenset({"Some real clause text."})),
    )
    _, dr = results[0]
    assert dr.basis == "judge"


def test_alignment_artifact_requires_minimum_length_floor() -> None:
    """A short, generic added clause that merely happens to be a substring of
    an unrelated, longer counterpart clause must NOT be auto-suppressed as an
    alignment artifact (issue #167 fix-round-1) — it still goes to the judge,
    mirroring clause_aligner.MOVE_EXACT_MIN_CHARS's length floor."""
    cd = _cd(
        "added",
        text_after="Time is of the essence.",
        path="12",
    )
    judge = MockDeviationJudge()
    results = assess_deviations(
        [cd],
        "standard",
        judge,
        counterpart_clause_texts=(
            frozenset(
                {
                    "No failure or delay by either party in exercising any right under "
                    "this Agreement shall operate as a waiver, and the parties agree "
                    "that time is of the essence with respect to the delivery "
                    "obligations in Section 5."
                }
            ),
            frozenset(),
        ),
    )
    _, dr = results[0]
    assert dr.basis == "judge"
    assert len(judge.received_items) == 1


def test_mixed_unchanged_and_modified() -> None:
    """Unchanged → deterministic; modified → judge."""
    diffs = [
        _cd("unchanged", "same", "same", path="1"),
        _cd("modified", "old text here", "new text here", path="2"),
    ]
    results = assess_deviations(diffs, "same", MockDeviationJudge())
    assert results[0][1].basis == "deterministic"
    assert results[1][1].basis == "judge"


def test_result_order_matches_input_order() -> None:
    diffs = [
        _cd("modified", "first clause text", "first clause updated", path="1"),
        _cd("unchanged", "second clause same", "second clause same", path="2"),
        _cd("modified", "third clause text", "third clause consequential", path="3"),
    ]
    results = assess_deviations(diffs, "standard text", MockDeviationJudge())
    assert len(results) == 3
    assert results[0][0].clause_path_after == "1"
    assert results[1][0].clause_path_after == "2"
    assert results[2][0].clause_path_after == "3"


# ---------------------------------------------------------------------------
# assess_deviations: judge failures
# ---------------------------------------------------------------------------


def test_judge_raises_returns_judge_error_not_propagated() -> None:
    cd = _cd("modified", "old", "new")
    results = assess_deviations([cd], "standard", RaisingJudge())
    _, dr = results[0]
    assert dr.basis == "judge_error"
    # Must NOT be "none" — recording a changed clause as benign silently hides risk (§P1.5).
    assert dr.deviation == "needs_review"


def test_judge_raises_does_not_drop_clauses() -> None:
    cd = _cd("modified", "old", "new")
    results = assess_deviations([cd], "standard", RaisingJudge())
    assert len(results) == 1


def test_judge_raises_needs_review_is_not_neutral_benign() -> None:
    """Changed clause with judge_error must not masquerade as a benign unchanged clause."""
    cd = _cd("modified", "old text", "new text")
    results = assess_deviations([cd], "standard", RaisingJudge())
    _, dr = results[0]
    # deviation="none" would falsely signal "no change" — must be "needs_review" instead.
    assert dr.deviation != "none"
    assert dr.deviation == "needs_review"
    assert dr.basis == "judge_error"


def test_judge_raises_needs_review_observable_in_to_dict() -> None:
    """§P1.5 acceptance: judge_error state must be observable in the serialised output."""
    cd = _cd("modified", "old", "new")
    results = assess_deviations([cd], "standard", RaisingJudge())
    _, dr = results[0]
    d = dr.to_dict()
    assert d["deviation"] == "needs_review"
    assert d["basis"] == "judge_error"


def test_judge_wrong_length_raises_value_error() -> None:
    cd = _cd("modified", "old", "new")
    with pytest.raises(ValueError, match="assess_batch"):
        assess_deviations([cd], "standard", WrongLengthJudge())


def test_judge_bad_basis_raises_value_error() -> None:
    cd = _cd("modified", "old", "new")
    with pytest.raises(ValueError, match="basis"):
        assess_deviations([cd], "standard", BadBasisJudge())


# ---------------------------------------------------------------------------
# P2.6 — Jaccard pre-filter (acceptance criteria)
# ---------------------------------------------------------------------------


def test_jaccard_near_identical_skips_judge() -> None:
    """Fixture pair with Jaccard > 0.92: judge must NOT be called; result is
    deviation='none', basis='reworded_equivalent'."""
    # Build a pair that is clearly above the threshold (same text with one
    # word swapped — Jaccard = 9/10 = 0.9 is NOT above threshold, so use
    # a pair where only one token differs out of many identical ones).
    base = "Alice Corp shall indemnify Beta Ltd for all direct losses and damages arising"
    # Add a single extra word: 15 tokens before, 16 after; Jaccard = 15/16 ≈ 0.9375 > 0.92
    similar = (
        "Alice Corp shall indemnify Beta Ltd for all direct losses and damages arising hereunder"
    )
    assert _text_jaccard(base, similar) >= REWORDED_EQUIVALENT_THRESHOLD

    judge = MockDeviationJudge()
    cd = _cd("modified", base, similar)
    results = assess_deviations([cd], "standard indemnification text.", judge)
    assert len(results) == 1
    _, dr = results[0]
    # Judge must NOT have been called.
    assert judge.received_items == []
    assert dr.deviation == "none"
    assert dr.basis == "reworded_equivalent"
    assert dr.confidence is None


def test_jaccard_below_threshold_calls_judge_with_hunk() -> None:
    """Fixture pair with Jaccard < 0.92 and changed text: judge must be called
    and the payload item must contain a 'hunk' key, not the full before/after text."""
    before = "Alice Corp shall indemnify Beta Ltd."
    after = "Alice Corp shall indemnify Beta Ltd for consequential damages."
    assert _text_jaccard(before, after) < REWORDED_EQUIVALENT_THRESHOLD

    judge = MockDeviationJudge()
    cd = _cd("modified", before, after)
    results = assess_deviations([cd], before, judge)
    assert len(results) == 1

    # Judge must have been called with exactly one item containing 'hunk'.
    assert len(judge.received_items) == 1
    item = judge.received_items[0]
    assert "hunk" in item
    # The hunk must NOT contain the raw full before text as a flat string.
    # It should contain the structured [BEFORE]/[AFTER] markers.
    assert "[BEFORE]" in item["hunk"]
    assert "[AFTER]" in item["hunk"]

    _, dr = results[0]
    assert dr.basis == "judge"
    assert dr.deviation == "substantive"


# ---------------------------------------------------------------------------
# P2.6 — confidence field (acceptance criteria)
# ---------------------------------------------------------------------------


def test_deviation_result_has_confidence_field() -> None:
    """DeviationResult must carry a confidence field (may be None)."""
    dr = DeviationResult(deviation="none", risk_delta=_NEUTRAL_ZERO, basis="deterministic")
    assert hasattr(dr, "confidence")
    assert dr.confidence is None


def test_deviation_result_confidence_round_trips_to_dict() -> None:
    """confidence field must round-trip through to_dict() (including None)."""
    dr_none = DeviationResult(deviation="none", risk_delta=_NEUTRAL_ZERO, basis="deterministic")
    assert dr_none.to_dict()["confidence"] is None

    dr_val = DeviationResult(
        deviation="substantive",
        risk_delta=_WORSE_MATERIAL,
        basis="judge",
        confidence=0.87,
    )
    assert dr_val.to_dict()["confidence"] == pytest.approx(0.87)


def test_deterministic_paths_have_none_confidence() -> None:
    """Deterministic (unchanged) and Jaccard paths must have confidence=None."""
    unchanged = _cd("unchanged", "same text", "same text")
    results = assess_deviations([unchanged], "same text", MockDeviationJudge())
    _, dr_unchanged = results[0]
    assert dr_unchanged.confidence is None

    # Near-identical pair (Jaccard pre-filter)
    base = "Alice Corp shall indemnify Beta Ltd for all direct losses and damages arising"
    similar = (
        "Alice Corp shall indemnify Beta Ltd for all direct losses and damages arising hereunder"
    )
    cd_jaccard = _cd("modified", base, similar)
    results2 = assess_deviations([cd_jaccard], "standard", MockDeviationJudge())
    _, dr_jaccard = results2[0]
    assert dr_jaccard.confidence is None


# ---------------------------------------------------------------------------
# P2.6 — _text_jaccard helper
# ---------------------------------------------------------------------------


def test_text_jaccard_identical_strings() -> None:
    assert _text_jaccard("hello world", "hello world") == 1.0


def test_text_jaccard_completely_different() -> None:
    assert _text_jaccard("alice bob", "carol dave") == 0.0


def test_text_jaccard_both_empty() -> None:
    assert _text_jaccard("", "") == 1.0


def test_text_jaccard_one_empty() -> None:
    assert _text_jaccard("some text", "") == 0.0
    assert _text_jaccard("", "some text") == 0.0


def test_text_jaccard_partial_overlap() -> None:
    # "a b c" vs "a b d" → tokens {a,b,c} vs {a,b,d}: |intersection|=2, |union|=4 → 0.5
    assert _text_jaccard("a b c", "a b d") == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Issue #220: deterministic standard check — the consumer path, no judge
# ---------------------------------------------------------------------------

_STD = (
    "Each Party shall hold the Confidential Information of the other Party in "
    "strict confidence and shall not disclose it to any third party without "
    "the prior written consent of the disclosing Party."
)


class TestStandardCheck:
    def test_exact_match_after_whitespace_case_and_punctuation(self) -> None:
        rewrapped = _STD.upper().replace(" ", "\n  ").replace(".", "")
        assert is_standard_text(rewrapped, _STD)

    def test_leading_content_number_change_is_not_standard(self) -> None:
        # No clause-number stripping: a leading number is content.
        std = "30 days after written notice either party may terminate this Agreement."
        assert not is_standard_text(std.replace("30 days", "60 days"), std)
        mult = "1.5 times the fees paid in the twelve months before the claim."
        assert not is_standard_text(mult.replace("1.5 times", "2.5 times"), mult)
        assert not is_standard_text("30. days", "60. days")

    def test_one_extra_word_is_not_standard(self) -> None:
        # Exact after normalization: no similarity tolerance on top of it.
        assert not is_standard_text(
            _STD.replace("strict confidence", "strict and confidence"), _STD
        )

    def test_negation_flip_is_not_standard(self) -> None:
        # Token-set Jaccard 0.938 under the old 0.92 bar: a reversed
        # obligation used to be reported as signed standard language.
        std = (
            "Neither party may assign this Agreement without the other party's prior "
            "written consent, except to an affiliate or to a successor in connection "
            "with a merger, acquisition, or sale of substantially all of its assets."
        )
        assert not is_standard_text(std.replace("Neither party", "Either party"), std)

    def test_deleted_carve_out_is_not_standard(self) -> None:
        std = (
            "Neither party may assign this Agreement without the other party's prior "
            "written consent, except to an affiliate or to a successor in connection "
            "with a merger, acquisition, or sale of substantially all of its assets."
        )
        assert not is_standard_text(std.replace(" to an affiliate or", ""), std)

    def test_inserted_not_is_not_standard(self) -> None:
        std = (
            "The obligations in this Agreement survive for three years after "
            "termination, and trade secrets remain protected for as long as they "
            "remain trade secrets under applicable law."
        )
        assert not is_standard_text(
            std.replace("trade secrets remain protected", "trade secrets are not protected"), std
        )

    def test_changed_number_is_not_standard(self) -> None:
        std = "Each party's aggregate liability shall not exceed fifty thousand dollars ($50,000)."
        assert not is_standard_text(std.replace("$50,000", "$500,000"), std)
        assert not is_standard_text(std.replace("fifty", "five hundred"), std)

    def test_substantive_change_is_not_standard(self) -> None:
        changed = (
            "Each Party may disclose the Confidential Information of the other "
            "Party to its affiliates, advisers and financing sources without consent."
        )
        assert not is_standard_text(changed, _STD)

    def test_party_names_never_decide(self) -> None:
        std = "Alice Corp shall hold the information of the Counterparty in strict confidence."
        deal = "AliceCo shall hold the information of Beta Ltd in strict confidence."
        assert not is_standard_text(deal, std)
        names = ["Alice Corp", "AliceCo", "Beta Ltd", "the Counterparty"]
        assert normalize_for_standard(deal, names) == normalize_for_standard(std, names)
        assert is_standard_text(deal, std, names)

    def test_party_name_match_is_whole_word_only(self) -> None:
        # "Ace" must not be rewritten inside "Acetone".
        assert normalize_for_standard("Acetone supplied by Ace.", ["Ace"]) == (
            "acetone supplied by party"
        )

    def test_no_standard_is_never_standard(self) -> None:
        assert not is_standard_text(_STD, "")
        assert not is_standard_text(_STD, [])
        assert not is_standard_text("", _STD)

    def test_node_sequence_matches_whole_or_any_single_node(self) -> None:
        first, second = "Alpha clause text about notices.", "Beta clause text about venue."
        assert is_standard_text(second, [first, second])
        assert is_standard_text(f"{first} {second}", [first, second])
        # A bare string is ONE whole clause: a lone node of it is not standard.
        assert not is_standard_text(second, f"{first}\n{second}")

    def test_deterministic_rows_never_need_review_or_a_judge(self) -> None:
        diffs = [
            _cd("unchanged", _STD, _STD),
            _cd("modified", "An opening draft nobody signed.", _STD),
            _cd("modified", _STD, "Each Party may disclose anything to anyone at any time."),
            _cd("added", "", "A counterparty rider on residual knowledge of personnel."),
            _cd("removed", _STD, ""),
        ]
        results = assess_deviations_deterministic(diffs, _STD)
        assert [cd for cd, _ in results] == diffs
        assert [dr.deviation for _, dr in results] == [
            "none",
            "none",
            "substantive",
            "substantive",
            "none",  # removed row: its OWN (before) text is our standard
        ]
        for _, dr in results:
            assert dr.basis == "deterministic"
            assert dr.risk_delta == RiskDelta(direction="neutral", magnitude="none")
            assert dr.confidence is None

    def test_identical_signed_text_gets_identical_answer_whatever_the_opening(self) -> None:
        """The failure mode #220 retires: the judged path diffs the net
        first-to-last hunk, so the same signed text reached from different
        opening drafts could get different verdicts. The standard check reads
        only the signed text."""
        signed = "Each Party may share Confidential Information with its auditors only."
        openings = ["", _STD, "A wholly different counterparty opening position text."]
        answers = {
            assess_deviations_deterministic([_cd("modified" if o else "added", o, signed)], _STD)[
                0
            ][1]
            for o in openings
        }
        assert len(answers) == 1
