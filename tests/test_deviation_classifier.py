"""Tests for the standard check (L4, issue #20 / #220).

Every clause gets one deterministic question: is its text OUR standard
language? There is no deviation judge (issue #239).

SECURITY NOTE: All fixtures are programmatically constructed with synthetic
text.  No real agreements are referenced.  Fictional party names only
('Alice Corp', 'Beta Ltd').
"""

from __future__ import annotations

import pytest

from playbook_engine.clause_differ import ClauseDiff, TextHunk
from playbook_engine.deviation_classifier import (
    DeviationResult,
    assess_deviations_deterministic,
    is_standard_text,
    normalize_for_standard,
    standard_check_result,
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


# ---------------------------------------------------------------------------
# DeviationResult
# ---------------------------------------------------------------------------


def test_deviation_result_valid() -> None:
    dr = DeviationResult(deviation="none")
    assert dr.basis == "deterministic"
    assert dr.to_dict() == {"deviation": "none", "basis": "deterministic", "rationale": ""}


@pytest.mark.parametrize("judged", ["reworded_equivalent", "needs_review", "unsure"])
def test_deviation_result_rejects_a_judged_deviation(judged: str) -> None:
    with pytest.raises(ValueError, match="deviation"):
        DeviationResult(deviation=judged)


@pytest.mark.parametrize("basis", ["judge", "judge_error", "needs_review", "alignment"])
def test_deviation_result_rejects_a_judged_basis(basis: str) -> None:
    with pytest.raises(ValueError, match="basis"):
        DeviationResult(deviation="none", basis=basis)


def test_standard_check_result_maps_the_fact_to_a_deviation() -> None:
    assert standard_check_result(True).deviation == "none"
    assert standard_check_result(False).deviation == "substantive"
    assert (
        standard_check_result(True).basis == standard_check_result(False).basis == ("deterministic")
    )


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
