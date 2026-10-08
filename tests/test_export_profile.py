"""Tests for playbook_engine.export_profile (issue #146).

Verified entirely offline with FAKE ``RedactionJudge`` / ``VerifyJudge``
implementations — no LLM, no network. Covers the three required scenarios:

  (a) the independent verify pass ALWAYS runs, even when the redaction pass
      found nothing to rewrite;
  (b) a verify-pass leak finding is surfaced on the report (and logged),
      never silently dropped, and never raised as an error (best-effort,
      no human gate);
  (c) export preserves clause structure — only the targeted free-text
      fields change, everything else (clause records, taxonomy_id, deal,
      paper, signed, standard, citations) is untouched.

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
# Fixture: a minimal OPF 0.4-shaped doc — one clause, two precedent records
# (no agreement_type, so precedent.refresh_derived leaves the ids as given)
# ---------------------------------------------------------------------------

_LARGE_SE = "The large southeastern teaching-hospital university"


def _record(pid: str, document_id: str, text: str, refused: list[str] | None = None) -> dict:
    ref = {"document_id": document_id, "version": 3, "clause_path": "8"}
    return {
        "id": pid,
        "taxonomy_id": "indemnification",
        "document_id": document_id,
        "paper": "theirs",
        "signed": True,
        "signed_text": {"text": text, "ref": ref},
        "opening_text": None,
        "standard": False,
        "refused_asks": [{"text": t, "round": 1, "ref": ref} for t in refused or []],
    }


def _make_doc() -> dict:
    return {
        "opf_version": "0.4",
        "evidence": {
            "clauses": [
                {
                    "id": "clause.indemnification",
                    "taxonomy_id": "indemnification",
                    "title": "Indemnification",
                    "our_standard": {"text": "Each party shall indemnify the other."},
                }
            ],
            "precedent": [
                _record(
                    "prec.a", "Counterparty-1-2023", "Counterparty-1 demanded a mutual carve-out."
                ),
                _record(
                    "prec.b",
                    "Counterparty-2-2022",
                    f"{_LARGE_SE} insisted on capping liability.",
                    refused=[f"{_LARGE_SE} asked for uncapped indemnity."],
                ),
            ],
        },
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
    # Verify pass must still have been called, once per free-text sample:
    # our_standard.text + 2 signed texts + 1 refused ask.
    assert len(verify.calls) == 1
    assert len(verify.calls[0]) == len(report.verify_findings) == 4
    assert report.leaked == ()


# ---------------------------------------------------------------------------
# (b) a verify-pass leak is surfaced, never silently emitted, never raised
# ---------------------------------------------------------------------------


def test_flagged_residual_leak_is_surfaced_not_silently_emitted() -> None:
    doc = _make_doc()
    redaction = _FakeRedactionJudge(flag_marker="large southeastern")
    # Independently flag the REWRITTEN signed text as still leaking, even
    # though the redaction pass "fixed" it — proving the two passes are
    # decoupled.
    rewritten_path = "precedent[1:prec.b].signed_text.text"
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
    leak_path = "precedent[0:prec.a].signed_text.text"
    verify = _FlaggingVerifyJudge(leak_path=leak_path)

    # Must not raise.
    report = export_profile(doc, redaction_judge=redaction, verify_judge=verify)
    assert len(report.leaked) == 1


# ---------------------------------------------------------------------------
# (c) export preserves clause structure
# ---------------------------------------------------------------------------


def test_export_preserves_clause_structure_and_only_rewrites_flagged_text() -> None:
    doc = _make_doc()
    original = copy.deepcopy(doc)
    redaction = _FakeRedactionJudge(flag_marker="large southeastern")
    verify = _CleanVerifyJudge()

    report = export_profile(doc, redaction_judge=redaction, verify_judge=verify)
    exported = report.doc["evidence"]
    before = original["evidence"]

    # Structure is byte-identical.
    assert exported["clauses"] == before["clauses"]
    for exp, orig in zip(exported["precedent"], before["precedent"], strict=True):
        for key in ("taxonomy_id", "document_id", "paper", "signed", "standard"):
            assert exp[key] == orig[key]
        assert exp["signed_text"]["ref"] == orig["signed_text"]["ref"]

    # Record 0 (no marker) is untouched.
    assert exported["precedent"][0]["signed_text"] == before["precedent"][0]["signed_text"]
    # Record 1 (has the marker) is rewritten in its signed text AND its ask.
    assert "large southeastern" not in exported["precedent"][1]["signed_text"]["text"]
    assert "large southeastern" not in exported["precedent"][1]["refused_asks"][0]["text"]
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
        "opf_version": "0.4",
        "evidence": {
            "clauses": [
                {
                    "id": "clause.x",
                    "taxonomy_id": "indemnification",
                    "our_standard": {"text": "Northwind State University demanded a carve-out."},
                },
                {
                    "id": "clause.x",
                    "taxonomy_id": "confidentiality",
                    "our_standard": {
                        "text": "Southridge Regional Medical Center capped liability."
                    },
                },
            ],
            "precedent": [],
        },
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

    exported_texts = [c["our_standard"]["text"] for c in report.doc["evidence"]["clauses"]]
    # Pre-#70 bug: the two clauses' identical clause-id-keyed paths collided,
    # so only the LAST clause's location survived in `locations` — the first
    # clause's flagged text shipped unmodified despite being flagged.
    assert "Northwind State University" not in " ".join(exported_texts)
    assert "Southridge Regional Medical Center" not in " ".join(exported_texts)
    # Both clauses were independently sampled (positionally-unique paths).
    sample_paths = {f.path for f in report.redaction_findings}
    assert "clauses[0:clause.x].our_standard.text" in sample_paths
    assert "clauses[1:clause.x].our_standard.text" in sample_paths


def _make_duplicate_other_sections_doc() -> dict:
    """Two entries sharing an id in each of precedent/floor/corpus, each
    with distinct residue-bearing text (issue #70 round 2 — the same
    collision class as clauses, but for the other three path families
    export_profile.py touches)."""
    return {
        "opf_version": "0.4",
        "evidence": {
            "clauses": [],
            "precedent": [
                _record(
                    "prec.dup", "d1", "Ridgeline Regional Utility pushed for a mutual carve-out."
                ),
                _record(
                    "prec.dup", "d2", "Ashgrove Municipal Water District required a longer term."
                ),
            ],
        },
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
    # precedent[{pi}:{id}], floor.invariants[{fi}:{invariant_id}],
    # corpus.documents[{di}:{document_id}]. Without the index tag, the two
    # duplicates in each section collapse onto one `locations` key and only
    # the last one's flagged text gets rewritten.
    doc = _make_duplicate_other_sections_doc()
    redaction = _FlagAllRedactionJudge()
    verify = _CleanVerifyJudge()

    report = export_profile(doc, redaction_judge=redaction, verify_judge=verify)

    exported_descriptions = [p["signed_text"]["text"] for p in report.doc["evidence"]["precedent"]]
    exported_statements = [inv["statement"] for inv in report.doc["floor"]["invariants"]]
    exported_titles = [d["title"] for d in report.doc["corpus"]["documents"]]

    assert "Ridgeline Regional Utility" not in " ".join(exported_descriptions)
    assert "Ashgrove Municipal Water District" not in " ".join(exported_descriptions)
    assert "Brookhaven Transit Authority" not in " ".join(exported_statements)
    assert "Fernwood County Hospital" not in " ".join(exported_statements)
    assert "Lakeside School District" not in " ".join(exported_titles)
    assert "Pinehollow Water Co-op" not in " ".join(exported_titles)

    sample_paths = {f.path for f in report.redaction_findings}
    assert "precedent[0:prec.dup].signed_text.text" in sample_paths
    assert "precedent[1:prec.dup].signed_text.text" in sample_paths
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
    doc = {"opf_version": "0.4", "evidence": {"clauses": [], "precedent": []}}

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


# ---------------------------------------------------------------------------
# OPF 0.4 (issue #223): precedent texts are residue surfaces, and a rewrite
# re-derives the digest — over the real compiled NDA example.
# ---------------------------------------------------------------------------


def test_export_profile_v04_samples_precedent_text_and_rederives_digest() -> None:
    import json as _json
    from pathlib import Path as _Path

    from playbook_engine.digest import build_digest
    from playbook_engine.export_profile import (
        RedactionFinding as _RF,
    )
    from playbook_engine.export_profile import (
        VerifyFinding as _VF,
    )
    from playbook_engine.export_profile import (
        _extract_text_samples,
        export_profile,
    )
    from playbook_engine.validator import validate_document as _validate

    doc = _json.loads(
        (_Path(__file__).parent.parent / "examples" / "nda" / "playbook.opf.json").read_text(
            encoding="utf-8"
        )
    )
    samples, _ = _extract_text_samples(doc)
    paths = {s.path for s in samples}
    assert any(".signed_text.text" in p for p in paths)
    assert any(".opening_text.text" in p for p in paths)
    assert any(".refused_asks[" in p for p in paths)
    assert not any(p.startswith("digest") for p in paths), "the digest is derived, not sampled"

    target = next(p for p in sorted(paths) if p.endswith(".signed_text.text"))

    class _Rewrite:
        def evaluate_batch(self, batch):  # noqa: ANN001
            return [
                _RF(
                    path=s.path,
                    has_residue=s.path == target,
                    rationale="identifying" if s.path == target else "clean",
                    rewritten_text="[REDACTED CLAUSE]" if s.path == target else None,
                )
                for s in batch
            ]

    class _Clean:
        def evaluate_batch(self, batch):  # noqa: ANN001
            return [_VF(path=s.path, leaked=False, rationale="clean") for s in batch]

    report = export_profile(doc, redaction_judge=_Rewrite(), verify_judge=_Clean())
    exported = report.doc
    assert "[REDACTED CLAUSE]" in _json.dumps(exported["evidence"])
    assert exported["digest"] == build_digest(exported)
    # export_profile does not re-stamp identity (publish does) — re-stamp it
    # so the check isolates the precedent ids, counts and digest.
    from playbook_engine.canonicalize import compute_section_digests, content_hash

    exported["identity"]["content_hash"] = content_hash(exported)
    exported["identity"]["section_digests"] = compute_section_digests(exported)
    errors = [str(e) for e in _validate(exported).errors if e.blocking]
    assert errors == [], errors
