"""Tests for playbook_engine.export_profile (issue #146).

Verified entirely offline with FAKE ``RedactionJudge`` / ``VerifyJudge``
implementations — no LLM, no network. Covers the three required scenarios:

  (a) the independent verify pass ALWAYS runs, even when the redaction pass
      found nothing to rewrite;
  (b) a verify-pass leak finding is surfaced on the report (and logged),
      never silently dropped, and never raised as an error (best-effort,
      no human gate);
  (c) export preserves clause/stance structure — only the targeted free-text
      fields change, everything else (id, taxonomy_id, rollup, deviation,
      risk_delta, provenance, outcome, citations) is untouched.

SECURITY NOTE: All fixtures use synthetic text and fictional party/institution
names only. No real agreement text or real document paths are used.
"""

from __future__ import annotations

import copy

import pytest

from playbook_engine.export_profile import (
    ExportProfileError,
    RedactionFinding,
    TextSample,
    VerifyFinding,
    export_profile,
)

# ---------------------------------------------------------------------------
# Fixture: a minimal OPF-shaped doc with one clause, two observed positions
# ---------------------------------------------------------------------------


def _make_doc() -> dict:
    return {
        "opf_version": "0.2",
        "clauses": [
            {
                "id": "clause.indemnification",
                "taxonomy_id": "indemnification",
                "title": "Indemnification",
                "our_standard": {"text": "Each party shall indemnify the other."},
                "observed_positions": [
                    {
                        "text_summary": "Counterparty-1 demanded a mutual carve-out.",
                        "full_text": "Counterparty-1 demanded a mutual carve-out.",
                        "example_ref": {
                            "document_id": "Counterparty-1-2023",
                            "version": 3,
                            "clause_path": "8",
                        },
                        "deviation": "substantive",
                        "risk_delta": {"direction": "worse", "magnitude": "minor"},
                        "provenance": "counterparty_paper",
                        "outcome": "signed",
                        "precedent_count": 3,
                    },
                    {
                        "text_summary": (
                            "The large southeastern teaching-hospital university insisted "
                            "on capping liability."
                        ),
                        "full_text": (
                            "The large southeastern teaching-hospital university insisted "
                            "on capping liability."
                        ),
                        "example_ref": {
                            "document_id": "Counterparty-2-2022",
                            "version": 1,
                            "clause_path": "9.1",
                        },
                        "deviation": "none",
                        "risk_delta": {"direction": "neutral", "magnitude": "none"},
                        "provenance": "our_paper",
                        "outcome": "signed",
                        "precedent_count": 7,
                    },
                ],
                "rollup": {
                    "position": "negotiable",
                    "confidence": {"score": 0.7, "n_our_paper": 7, "n_counterparty_paper": 3},
                },
            }
        ],
    }


# ---------------------------------------------------------------------------
# Fake judges
# ---------------------------------------------------------------------------


class _FakeRedactionJudge:
    """Flags samples whose text contains a marker substring; rewrites them."""

    def __init__(self, flag_marker: str | None = None, drop_paths: frozenset[str] = frozenset()):
        self._flag_marker = flag_marker
        self._drop_paths = drop_paths

    def evaluate_batch(self, samples):  # noqa: ANN001
        findings = []
        for s in samples:
            if s.path in self._drop_paths:
                continue  # simulate a judge that silently drops a sample
            if self._flag_marker and self._flag_marker in s.text:
                findings.append(
                    RedactionFinding(
                        path=s.path,
                        has_residue=True,
                        rationale="Descriptive phrase still identifies the counterparty.",
                        rewritten_text="A counterparty raised concerns about this clause.",
                    )
                )
            else:
                findings.append(
                    RedactionFinding(path=s.path, has_residue=False, rationale="No residue found.")
                )
        return findings


class _CleanVerifyJudge:
    """Independent verify pass that always reports no leak."""

    def __init__(self):
        self.calls: list[list[str]] = []

    def evaluate_batch(self, samples):  # noqa: ANN001
        self.calls.append([s.text for s in samples])
        return [
            VerifyFinding(path=s.path, leaked=False, rationale="Independently confirmed clean.")
            for s in samples
        ]


class _FlaggingVerifyJudge:
    """Independent verify pass that flags one specific path as still leaked."""

    def __init__(self, leak_path: str):
        self._leak_path = leak_path

    def evaluate_batch(self, samples):  # noqa: ANN001
        return [
            VerifyFinding(
                path=s.path,
                leaked=(s.path == self._leak_path),
                rationale=(
                    "Rewrite still narrows the field enough to identify the counterparty."
                    if s.path == self._leak_path
                    else "Clean."
                ),
            )
            for s in samples
        ]


class _RaisingJudge:
    def evaluate_batch(self, samples):  # noqa: ANN001
        raise RuntimeError("LLM timeout")


class _DroppingRedactionJudge:
    """Simulates a judge that silently drops every sample (returns nothing)."""

    def evaluate_batch(self, samples):  # noqa: ANN001
        return []


# ---------------------------------------------------------------------------
# (a) independent verify pass ALWAYS runs, even when redaction found nothing
# ---------------------------------------------------------------------------


def test_verify_pass_always_runs_even_when_redaction_finds_nothing() -> None:
    doc = _make_doc()
    redaction = _FakeRedactionJudge(flag_marker=None)  # never flags anything
    verify = _CleanVerifyJudge()

    report = export_profile(doc, redaction_judge=redaction, verify_judge=verify)

    assert all(not f.has_residue for f in report.redaction_findings)
    # Verify pass must still have been called, once per free-text sample.
    assert len(verify.calls) == 1
    # 2 observed_positions x 2 fields (text_summary/full_text) + 1
    # our_standard.text (issue #188 extended sampling to this surface too).
    assert len(verify.calls[0]) == len(report.verify_findings) == 5
    assert report.leaked == ()


# ---------------------------------------------------------------------------
# (b) a verify-pass leak is surfaced, never silently emitted, never raised
# ---------------------------------------------------------------------------


def test_flagged_residual_leak_is_surfaced_not_silently_emitted() -> None:
    doc = _make_doc()
    redaction = _FakeRedactionJudge(flag_marker="large southeastern")
    # Independently flag the REWRITTEN text_summary as still leaking, even
    # though the redaction pass "fixed" it — proving the two passes are
    # decoupled.
    rewritten_path = "clauses[0:clause.indemnification].observed_positions[1].text_summary"
    verify = _FlaggingVerifyJudge(leak_path=rewritten_path)

    report = export_profile(doc, redaction_judge=redaction, verify_judge=verify)

    # Best-effort / no human gate: export_profile does not raise.
    assert isinstance(report.doc, dict)
    # ... but the leak is never silently dropped: it is on the report,
    assert len(report.leaked) == 1
    assert report.leaked[0].path == rewritten_path
    assert report.leaked[0].leaked is True
    assert report.leaked[0].rationale


def test_export_profile_does_not_raise_on_a_leaked_verdict() -> None:
    """A "leak found" verdict is a SUCCESSFUL evaluation, not a judge failure."""
    doc = _make_doc()
    redaction = _FakeRedactionJudge(flag_marker=None)
    leak_path = "clauses[0:clause.indemnification].observed_positions[0].text_summary"
    verify = _FlaggingVerifyJudge(leak_path=leak_path)

    # Must not raise.
    report = export_profile(doc, redaction_judge=redaction, verify_judge=verify)
    assert len(report.leaked) == 1


# ---------------------------------------------------------------------------
# (c) export preserves stance + clause structure
# ---------------------------------------------------------------------------


def test_export_preserves_clause_structure_and_only_rewrites_flagged_text() -> None:
    doc = _make_doc()
    original = copy.deepcopy(doc)
    redaction = _FakeRedactionJudge(flag_marker="large southeastern")
    verify = _CleanVerifyJudge()

    report = export_profile(doc, redaction_judge=redaction, verify_judge=verify)
    exported_clause = report.doc["clauses"][0]
    original_clause = original["clauses"][0]

    # Structure is byte-identical.
    assert exported_clause["id"] == original_clause["id"]
    assert exported_clause["taxonomy_id"] == original_clause["taxonomy_id"]
    assert exported_clause["rollup"] == original_clause["rollup"]
    for exp_obs, orig_obs in zip(
        exported_clause["observed_positions"], original_clause["observed_positions"], strict=True
    ):
        assert exp_obs["deviation"] == orig_obs["deviation"]
        assert exp_obs["risk_delta"] == orig_obs["risk_delta"]
        assert exp_obs["provenance"] == orig_obs["provenance"]
        assert exp_obs["outcome"] == orig_obs["outcome"]
        assert exp_obs["example_ref"] == orig_obs["example_ref"]

    # Observation 0 (no marker) is untouched.
    assert (
        exported_clause["observed_positions"][0]["text_summary"]
        == original_clause["observed_positions"][0]["text_summary"]
    )
    # Observation 1 (has the marker) is rewritten in BOTH free-text fields.
    assert (
        exported_clause["observed_positions"][1]["text_summary"]
        != (original_clause["observed_positions"][1]["text_summary"])
    )
    assert (
        exported_clause["observed_positions"][1]["full_text"]
        != (original_clause["observed_positions"][1]["full_text"])
    )
    # The input doc itself is never mutated.
    assert doc == original


# ---------------------------------------------------------------------------
# Duplicate clause ids must not collapse rewrites onto the last duplicate
# (issue #70) — a foreign/hand-edited doc may carry two clauses sharing an
# id (nothing in the schema or validator.py enforces uniqueness pre-#70).
# ---------------------------------------------------------------------------


def _make_duplicate_clause_id_doc() -> dict:
    """Two clauses sharing id 'clause.x', each with distinct residue-bearing text."""
    return {
        "opf_version": "0.2",
        "clauses": [
            {
                "id": "clause.x",
                "taxonomy_id": "indemnification",
                "title": "Indemnification",
                "observed_positions": [
                    {
                        "text_summary": "Northwind State University demanded a carve-out.",
                        "full_text": "Northwind State University demanded a carve-out.",
                        "deviation": "substantive",
                        "risk_delta": {"direction": "worse", "magnitude": "minor"},
                        "provenance": "counterparty_paper",
                        "outcome": "signed",
                        "precedent_count": 3,
                    }
                ],
                "rollup": {
                    "position": "negotiable",
                    "confidence": {"score": 0.5, "n_our_paper": 0, "n_counterparty_paper": 3},
                },
            },
            {
                "id": "clause.x",
                "taxonomy_id": "confidentiality",
                "title": "Confidentiality",
                "observed_positions": [
                    {
                        "text_summary": "Southridge Regional Medical Center capped liability.",
                        "full_text": "Southridge Regional Medical Center capped liability.",
                        "deviation": "none",
                        "risk_delta": {"direction": "neutral", "magnitude": "none"},
                        "provenance": "our_paper",
                        "outcome": "signed",
                        "precedent_count": 5,
                    }
                ],
                "rollup": {
                    "position": "standard",
                    "confidence": {"score": 0.9, "n_our_paper": 5, "n_counterparty_paper": 0},
                },
            },
        ],
    }


class _FlagAllRedactionJudge:
    """Flags and rewrites every sample it is handed, unconditionally."""

    def evaluate_batch(self, samples):  # noqa: ANN001
        return [
            RedactionFinding(
                path=s.path,
                has_residue=True,
                rationale="Flagged for rewrite.",
                rewritten_text="A counterparty raised concerns about this clause.",
            )
            for s in samples
        ]


def test_duplicate_clause_ids_both_get_rewritten_not_just_the_last() -> None:
    doc = _make_duplicate_clause_id_doc()
    # Flag every sample and rewrite it — a judge that flags+rewrites
    # everything it is handed.
    redaction = _FlagAllRedactionJudge()
    verify = _CleanVerifyJudge()

    report = export_profile(doc, redaction_judge=redaction, verify_judge=verify)

    exported_texts = [
        obs["text_summary"]
        for clause in report.doc["clauses"]
        for obs in clause["observed_positions"]
    ]
    # Pre-#70 bug: the two clauses' identical clause-id-keyed paths collided,
    # so only the LAST clause's location survived in `locations` — the first
    # clause's flagged text_summary shipped unmodified despite being flagged.
    assert "Northwind State University" not in " ".join(exported_texts)
    assert "Southridge Regional Medical Center" not in " ".join(exported_texts)
    # Both clauses were independently sampled (positionally-unique paths).
    sample_paths = {f.path for f in report.redaction_findings}
    assert "clauses[0:clause.x].observed_positions[0].text_summary" in sample_paths
    assert "clauses[1:clause.x].observed_positions[0].text_summary" in sample_paths


def _make_duplicate_other_sections_doc() -> dict:
    """Two entries sharing an id in each of clause_library/floor/corpus,
    each with distinct residue-bearing text (issue #70 round 2 — the same
    collision class as clauses, but for the other three path families
    export_profile.py touches)."""
    return {
        "opf_version": "0.2",
        "clauses": [],
        "clause_library": [
            {
                "concept_id": "concept.dup",
                "title": "Indemnification",
                "description": "Ridgeline Regional Utility pushed for a mutual carve-out.",
            },
            {
                "concept_id": "concept.dup",
                "title": "Confidentiality",
                "description": "Ashgrove Municipal Water District required a longer term.",
            },
        ],
        "floor": {
            "invariants": [
                {
                    "id": "invariant.dup",
                    "statement": "Brookhaven Transit Authority never accepts uncapped liability.",
                },
                {
                    "id": "invariant.dup",
                    "statement": "Fernwood County Hospital never waives audit rights.",
                },
            ]
        },
        "corpus": {
            "documents": [
                {"document_id": "doc.dup", "title": "Contract with Lakeside School District"},
                {"document_id": "doc.dup", "title": "Contract with Pinehollow Water Co-op"},
            ]
        },
    }


def test_duplicate_ids_in_other_sections_both_get_rewritten_not_just_the_last() -> None:
    # Sibling of test_duplicate_clause_ids_both_get_rewritten_not_just_the_last
    # covering the other three path families this issue changed:
    # clause_library[{li}:{concept_id}], floor.invariants[{fi}:{invariant_id}],
    # corpus.documents[{di}:{document_id}]. Without the index tag, the two
    # duplicates in each section collapse onto one `locations` key and only
    # the last one's flagged text gets rewritten.
    doc = _make_duplicate_other_sections_doc()
    redaction = _FlagAllRedactionJudge()
    verify = _CleanVerifyJudge()

    report = export_profile(doc, redaction_judge=redaction, verify_judge=verify)

    exported_descriptions = [c["description"] for c in report.doc["clause_library"]]
    exported_statements = [inv["statement"] for inv in report.doc["floor"]["invariants"]]
    exported_titles = [d["title"] for d in report.doc["corpus"]["documents"]]

    assert "Ridgeline Regional Utility" not in " ".join(exported_descriptions)
    assert "Ashgrove Municipal Water District" not in " ".join(exported_descriptions)
    assert "Brookhaven Transit Authority" not in " ".join(exported_statements)
    assert "Fernwood County Hospital" not in " ".join(exported_statements)
    assert "Lakeside School District" not in " ".join(exported_titles)
    assert "Pinehollow Water Co-op" not in " ".join(exported_titles)

    sample_paths = {f.path for f in report.redaction_findings}
    assert "clause_library[0:concept.dup].description" in sample_paths
    assert "clause_library[1:concept.dup].description" in sample_paths
    assert "floor.invariants[0:invariant.dup].statement" in sample_paths
    assert "floor.invariants[1:invariant.dup].statement" in sample_paths
    assert "corpus.documents[0:doc.dup].title" in sample_paths
    assert "corpus.documents[1:doc.dup].title" in sample_paths


# ---------------------------------------------------------------------------
# Coverage / contract failures — judge raises or silently drops a sample
# ---------------------------------------------------------------------------


def test_redaction_judge_raising_fails_loud() -> None:
    doc = _make_doc()
    with pytest.raises(ExportProfileError, match="RedactionJudge"):
        export_profile(doc, redaction_judge=_RaisingJudge(), verify_judge=_CleanVerifyJudge())


def test_verify_judge_raising_fails_loud() -> None:
    doc = _make_doc()
    with pytest.raises(ExportProfileError, match="VerifyJudge"):
        export_profile(
            doc, redaction_judge=_FakeRedactionJudge(flag_marker=None), verify_judge=_RaisingJudge()
        )


def test_redaction_judge_silently_dropping_a_sample_fails_loud() -> None:
    doc = _make_doc()
    with pytest.raises(ExportProfileError, match="unevaluated"):
        export_profile(
            doc, redaction_judge=_DroppingRedactionJudge(), verify_judge=_CleanVerifyJudge()
        )


def test_no_free_text_samples_never_calls_either_judge() -> None:
    doc = {"opf_version": "0.2", "clauses": []}

    class _NeverCallJudge:
        def evaluate_batch(self, samples):  # noqa: ANN001
            raise AssertionError("must not be called with zero samples")

    report = export_profile(doc, redaction_judge=_NeverCallJudge(), verify_judge=_NeverCallJudge())
    assert report.redaction_findings == ()
    assert report.verify_findings == ()
    assert report.leaked == ()


# ---------------------------------------------------------------------------
# Finding dataclass validation
# ---------------------------------------------------------------------------


def test_redaction_finding_requires_rewritten_text_when_flagged() -> None:
    with pytest.raises(ValueError, match="rewritten_text"):
        RedactionFinding(path="p", has_residue=True, rationale="found something")


def test_redaction_finding_rejects_unknown_basis() -> None:
    with pytest.raises(ValueError, match="basis"):
        RedactionFinding(path="p", has_residue=False, rationale="ok", basis="vibes")


def test_verify_finding_requires_rationale() -> None:
    with pytest.raises(ValueError, match="rationale"):
        VerifyFinding(path="p", leaked=False, rationale="")


def test_text_sample_is_a_plain_value_object() -> None:
    s = TextSample(path="p", text="hello")
    assert s.path == "p"
    assert s.text == "hello"
