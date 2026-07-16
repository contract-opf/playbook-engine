"""Tests for viewer.py — issue #68.

SECURITY NOTE: All fixtures use synthetic text and fictional party/institution
names only.  No real agreement text or real document paths are used.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from playbook_engine.cli import cli
from playbook_engine.viewer import _build_index, apply_feedback, render_review_html

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _make_opf(tmp_path: Path, clauses: list[dict] | None = None) -> dict:
    """Build a minimal valid OPF dict and write playbook.opf.json to tmp_path."""
    if clauses is None:
        clauses = [
            {
                "id": "clause.indemnification",
                "taxonomy_id": "indemnification",
                "title": "Indemnification",
                "our_standard": {
                    "text": "Each party shall indemnify the other.",
                    "source_ref": {
                        "document_id": "template",
                        "version": "template",
                        "clause_path": "8",
                    },
                },
                "observed_positions": [
                    {
                        "text_summary": "Mutual indemnification, negligence-based.",
                        "example_ref": {
                            "document_id": "state-university-2023",
                            "version": 3,
                            "clause_path": "8",
                        },
                        "deviation": "none",
                        "risk_delta": {"direction": "neutral", "magnitude": "none"},
                        "provenance": "our_paper",
                        "outcome": "signed",
                        "precedent_count": 7,
                    },
                    {
                        "text_summary": "Mutual limited to gross negligence — substantive.",
                        "example_ref": {
                            "document_id": "city-college-2022",
                            "version": 4,
                            "clause_path": "9.1",
                        },
                        "deviation": "substantive",
                        "risk_delta": {"direction": "worse", "magnitude": "minor"},
                        "provenance": "our_paper",
                        "outcome": "signed",
                        "precedent_count": 3,
                    },
                ],
                "rollup": {
                    "position": "negotiable",
                    "confidence": {
                        "score": 0.82,
                        "basis": "precedent_count",
                        "n_our_paper": 10,
                        "n_counterparty_paper": 0,
                    },
                },
            },
            {
                "id": "clause.governing_law",
                "taxonomy_id": "governing_law",
                "title": "Governing Law",
                "our_standard": None,
                "observed_positions": [
                    {
                        "text_summary": "Institution home-state law.",
                        "example_ref": {
                            "document_id": "pacific-state-college-2022",
                            "version": 3,
                            "clause_path": "12",
                        },
                        "deviation": "substantive",
                        "risk_delta": {"direction": "worse", "magnitude": "minor"},
                        "provenance": "our_paper",
                        "outcome": "signed",
                        "precedent_count": 2,
                    },
                ],
                "rollup": {
                    "position": "negotiable",
                    "confidence": {
                        "score": 0.55,
                        "basis": "precedent_count",
                        "n_our_paper": 9,
                        "n_counterparty_paper": 0,
                    },
                },
            },
        ]

    doc = {
        "opf_version": "0.1",
        "agreement_type": {"id": "educational-affiliation", "name": "Educational Affiliation"},
        "baseline": {"has_canonical_template": True},
        "taxonomy": {
            "source": "custom",
            "entries": [
                {"id": "indemnification", "label": "Indemnification", "status": "active"},
                {"id": "governing_law", "label": "Governing Law", "status": "active"},
            ],
        },
        "clauses": clauses,
        "corpus": {
            "documents": [
                {
                    "document_id": "state-university-2023",
                    "provenance": "our_paper",
                    "in_scope": True,
                },
            ],
            "stats": {},
        },
        "compiler": {
            "name": "playbook-engine",
            "version": "0.1.0",
            "generated_at": "2026-01-01T00:00:00Z",
        },
    }
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "playbook.opf.json").write_text(json.dumps(doc), encoding="utf-8")

    # Write normalized clause trees for every cited corpus document so that
    # classification feedback can source the full clause text the judge hashes
    # (issue #70). Node text/heading mirror what the engine would have produced.
    nodes_by_doc: dict[str, list[dict]] = {}
    for clause in clauses:
        title = clause.get("title", "")
        for obs in clause.get("observed_positions", []):
            ref = obs.get("example_ref") or {}
            cdoc, cpath = ref.get("document_id"), ref.get("clause_path")
            if not (cdoc and cpath):
                continue
            full_text = f"{obs.get('text_summary', '')} Full synthetic body for clause {cpath}."
            nodes_by_doc.setdefault(cdoc, []).append(
                {
                    "clause_path": cpath,
                    "heading": title,
                    "text": full_text,
                    "char_span": [0, len(full_text)],
                    "children": [],
                }
            )
    for cdoc, nodes in nodes_by_doc.items():
        tree = {"document_id": cdoc, "version": "v1", "source_file": "v1.rtf", "nodes": nodes}
        tree_path = out_dir / "normalized" / cdoc / "v1.clauses.json"
        tree_path.parent.mkdir(parents=True, exist_ok=True)
        tree_path.write_text(json.dumps(tree), encoding="utf-8")

    return doc


def _write_feedback(tmp_path: Path, feedback: dict) -> Path:
    p = tmp_path / "feedback.json"
    p.write_text(json.dumps(feedback), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# _build_index — item numbering
# ---------------------------------------------------------------------------


def test_build_index_clause_numbers(tmp_path: Path) -> None:
    """Clause items are numbered C1, C2, … in taxonomy+id sorted order."""
    _make_opf(tmp_path)
    doc = json.loads((tmp_path / "out" / "playbook.opf.json").read_text())
    index = _build_index(doc)
    clause_nums = [num for num, kind, _ in index if kind == "clause"]
    assert clause_nums == ["C1", "C2"]


def test_build_index_observation_numbers(tmp_path: Path) -> None:
    """Observation items are numbered C1.1, C2.1, C2.2, … under their clause.

    Clauses are sorted by (taxonomy_id, id): governing_law < indemnification.
    governing_law has 1 observation (C1.1); indemnification has 2 (C2.1, C2.2).
    """
    _make_opf(tmp_path)
    doc = json.loads((tmp_path / "out" / "playbook.opf.json").read_text())
    index = _build_index(doc)
    obs_nums = [num for num, kind, _ in index if kind == "observation"]
    assert "C1.1" in obs_nums
    assert "C2.1" in obs_nums
    assert "C2.2" in obs_nums


def test_build_index_deterministic(tmp_path: Path) -> None:
    """Same playbook → same item numbering on every call."""
    _make_opf(tmp_path)
    doc = json.loads((tmp_path / "out" / "playbook.opf.json").read_text())
    index1 = _build_index(doc)
    index2 = _build_index(doc)
    assert [(n, k) for n, k, _ in index1] == [(n, k) for n, k, _ in index2]


def test_build_index_clause_id_in_payload(tmp_path: Path) -> None:
    """Each clause item payload carries _clause_id and _clause_num."""
    _make_opf(tmp_path)
    doc = json.loads((tmp_path / "out" / "playbook.opf.json").read_text())
    index = _build_index(doc)
    for _num, kind, payload in index:
        if kind == "clause":
            assert "_clause_id" in payload
            assert "_clause_num" in payload


def test_build_index_observation_payload_has_clause_ref(tmp_path: Path) -> None:
    """Each observation item payload carries _clause_id and _obs_num."""
    _make_opf(tmp_path)
    doc = json.loads((tmp_path / "out" / "playbook.opf.json").read_text())
    index = _build_index(doc)
    for _num, kind, payload in index:
        if kind == "observation":
            assert "_clause_id" in payload
            assert "_obs_num" in payload


# ---------------------------------------------------------------------------
# render_review_html — HTML content
# ---------------------------------------------------------------------------


def test_render_review_html_returns_string(tmp_path: Path) -> None:
    """render_review_html returns a non-empty string."""
    _make_opf(tmp_path)
    html = render_review_html(tmp_path / "out")
    assert isinstance(html, str)
    assert len(html) > 100


def test_render_html_is_valid_html(tmp_path: Path) -> None:
    """Output starts with DOCTYPE."""
    _make_opf(tmp_path)
    html = render_review_html(tmp_path / "out")
    assert html.strip().startswith("<!DOCTYPE html>")


def test_render_html_contains_clause_number_c1(tmp_path: Path) -> None:
    """HTML contains the numbered clause item C1."""
    _make_opf(tmp_path)
    html = render_review_html(tmp_path / "out")
    assert "C1" in html


def test_render_html_contains_observation_number_c1_1(tmp_path: Path) -> None:
    """HTML contains the numbered observation item C1.1."""
    _make_opf(tmp_path)
    html = render_review_html(tmp_path / "out")
    assert "C1.1" in html


def test_render_html_contains_observation_number_c2_2(tmp_path: Path) -> None:
    """HTML contains the numbered observation C2.2 (indemnification's second obs)."""
    _make_opf(tmp_path)
    html = render_review_html(tmp_path / "out")
    assert "C2.2" in html


def test_render_html_embeds_json(tmp_path: Path) -> None:
    """HTML contains the embedded playbook JSON."""
    _make_opf(tmp_path)
    html = render_review_html(tmp_path / "out")
    assert 'type="application/json"' in html
    assert "playbook-data" in html
    # The embedded JSON should contain OPF version
    assert '"opf_version"' in html


def test_render_html_requires_no_network(tmp_path: Path) -> None:
    """HTML must not reference external CDN / fetch URLs."""
    _make_opf(tmp_path)
    html = render_review_html(tmp_path / "out")
    assert "https://cdn" not in html
    assert "http://cdn" not in html
    assert "fetch(" not in html


def test_render_html_contains_evidence_citations(tmp_path: Path) -> None:
    """HTML contains evidence citation (document_id from example_ref)."""
    _make_opf(tmp_path)
    html = render_review_html(tmp_path / "out")
    assert "state-university-2023" in html
    assert "city-college-2022" in html


def test_render_html_contains_clause_title(tmp_path: Path) -> None:
    """HTML contains clause titles."""
    _make_opf(tmp_path)
    html = render_review_html(tmp_path / "out")
    assert "Indemnification" in html
    assert "Governing Law" in html


def test_render_html_contains_rollup_position(tmp_path: Path) -> None:
    """HTML shows the rollup position."""
    _make_opf(tmp_path)
    html = render_review_html(tmp_path / "out")
    assert "negotiable" in html


def test_render_html_v02_historical_stance_gets_non_default_color(tmp_path: Path) -> None:
    """A v0.2 ``summary.historical_stance`` value renders a non-default
    (non-gray) stance color — issue #155. Before the fix, v0.2 stance values
    were absent from ``_POSITION_COLORS`` and fell through to the default
    gray (``#374151``)."""
    doc = {
        "opf_version": "0.2",
        "agreement_type": {"id": "educational-affiliation", "name": "Educational Affiliation"},
        "baseline": {"has_canonical_template": False},
        "taxonomy": {
            "source": "custom",
            "entries": [{"id": "governing_law", "label": "Governing Law", "status": "active"}],
        },
        "evidence": {
            "clauses": [
                {
                    "id": "clause.governing_law",
                    "taxonomy_id": "governing_law",
                    "title": "Governing Law",
                    "our_standard": None,
                    "observed_positions": [],
                    "summary": {
                        "historical_stance": "usually_conceded",
                        "confidence": {"score": 0.6},
                    },
                }
            ],
            "documents": [],
        },
        "corpus": {"documents": [], "stats": {}},
        "compiler": {
            "name": "playbook-engine",
            "version": "0.1.0",
            "generated_at": "2026-01-01T00:00:00Z",
        },
    }
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "playbook.opf.json").write_text(json.dumps(doc), encoding="utf-8")

    html = render_review_html(out_dir)
    assert "usually_conceded" in html
    # The mapped color for usually_conceded must appear, and the fallback
    # default gray must NOT be the color used for this clause's stance span.
    assert 'color:#dc2626;font-weight:700">usually_conceded' in html
    assert 'color:#374151;font-weight:700">usually_conceded' not in html


def test_render_html_contains_export_button(tmp_path: Path) -> None:
    """HTML contains the Export feedback button."""
    _make_opf(tmp_path)
    html = render_review_html(tmp_path / "out")
    assert "Export feedback" in html


def test_render_html_contains_comment_inputs(tmp_path: Path) -> None:
    """HTML contains per-item comment inputs."""
    _make_opf(tmp_path)
    html = render_review_html(tmp_path / "out")
    assert "comment-input" in html


def test_render_html_writes_file_when_out_file_given(tmp_path: Path) -> None:
    """render_review_html writes the HTML atomically when out_file is given."""
    _make_opf(tmp_path)
    out_file = tmp_path / "review.html"
    html = render_review_html(tmp_path / "out", out_file=out_file)
    assert out_file.exists()
    assert out_file.read_text(encoding="utf-8") == html


def test_render_html_no_tmp_file_left_behind(tmp_path: Path) -> None:
    """Atomic write leaves no .tmp file behind."""
    _make_opf(tmp_path)
    out_file = tmp_path / "review.html"
    render_review_html(tmp_path / "out", out_file=out_file)
    assert not (tmp_path / "review.tmp").exists()


def test_render_html_missing_opf_raises(tmp_path: Path) -> None:
    """FileNotFoundError raised when playbook.opf.json is absent."""
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        render_review_html(out_dir)


def test_render_html_our_standard_text_present(tmp_path: Path) -> None:
    """HTML shows our_standard text for clauses that have it."""
    _make_opf(tmp_path)
    html = render_review_html(tmp_path / "out")
    assert "Each party shall indemnify the other." in html


def test_render_html_numbering_stable_different_input_order(tmp_path: Path) -> None:
    """Clause ordering is deterministic (sorted by taxonomy_id, id), not input order."""
    clauses = [
        {
            "id": "clause.governing_law",
            "taxonomy_id": "governing_law",
            "title": "Governing Law",
            "our_standard": None,
            "observed_positions": [],
            "rollup": {"position": "standard", "confidence": {"score": 0.9}},
        },
        {
            "id": "clause.indemnification",
            "taxonomy_id": "indemnification",
            "title": "Indemnification",
            "our_standard": None,
            "observed_positions": [],
            "rollup": {"position": "negotiable", "confidence": {"score": 0.7}},
        },
    ]
    doc = {
        "opf_version": "0.1",
        "agreement_type": {"id": "test", "name": "Test"},
        "baseline": {"has_canonical_template": False},
        "taxonomy": {
            "source": "custom",
            "entries": [
                {"id": "governing_law", "label": "Governing Law", "status": "active"},
                {"id": "indemnification", "label": "Indemnification", "status": "active"},
            ],
        },
        "clauses": clauses,
        "corpus": {"documents": [], "stats": {}},
        "compiler": {"name": "pe", "version": "0.1.0", "generated_at": "2026-01-01T00:00:00Z"},
    }
    out_dir = tmp_path / "out2"
    out_dir.mkdir(parents=True)
    (out_dir / "playbook.opf.json").write_text(json.dumps(doc), encoding="utf-8")
    html = render_review_html(out_dir)
    # governing_law < indemnification lexicographically → C1 = governing_law
    idx_c1 = html.index(">C1<")
    idx_c2 = html.index(">C2<")
    idx_gov = html.index("Governing Law")
    idx_ind = html.index("Indemnification")
    # C1 appears before C2, and govering law should be C1
    assert idx_c1 < idx_c2
    # Governing Law header should appear before the C2 (Indemnification) section
    assert idx_gov < idx_ind


# ---------------------------------------------------------------------------
# apply_feedback — hints.yaml
# ---------------------------------------------------------------------------


def test_apply_feedback_provenance_writes_hints_yaml(tmp_path: Path) -> None:
    """Provenance correction in feedback → hints.yaml for cited document.

    C2.1 is the first observation of the indemnification clause, whose
    example_ref points to state-university-2023.
    """
    _make_opf(tmp_path)
    out_dir = tmp_path / "out"

    # Create a corpus directory for the cited doc
    doc_dir = tmp_path / "state-university-2023"
    doc_dir.mkdir()

    # C2.1 = first obs of indemnification clause → state-university-2023
    feedback = {
        "C2.1": {"provenance": "counterparty_paper"},
    }
    fp = _write_feedback(tmp_path, feedback)
    result = apply_feedback(out_dir, fp)

    # Check hints.yaml was written somewhere relevant
    assert "state-university-2023" in result.hints_written
    # Find the written hints.yaml
    hints_path = doc_dir / "hints.yaml"
    assert hints_path.exists()
    data = yaml.safe_load(hints_path.read_text(encoding="utf-8"))
    assert data["provenance"] == "counterparty_paper"


def test_apply_feedback_signed_version_writes_hints_yaml(tmp_path: Path) -> None:
    """signed_version correction → hints.yaml for cited document.

    C2.1 = first obs of indemnification → state-university-2023.
    """
    _make_opf(tmp_path)
    out_dir = tmp_path / "out"

    doc_dir = tmp_path / "state-university-2023"
    doc_dir.mkdir()

    feedback = {"C2.1": {"signed_version": "v3"}}
    fp = _write_feedback(tmp_path, feedback)
    result = apply_feedback(out_dir, fp)

    assert "state-university-2023" in result.hints_written
    hints_path = doc_dir / "hints.yaml"
    data = yaml.safe_load(hints_path.read_text(encoding="utf-8"))
    assert data["signed_version"] == "v3"


def test_apply_feedback_order_writes_hints_yaml(tmp_path: Path) -> None:
    """order correction → hints.yaml for cited document.

    C2.1 = first obs of indemnification → state-university-2023.
    """
    _make_opf(tmp_path)
    out_dir = tmp_path / "out"

    doc_dir = tmp_path / "state-university-2023"
    doc_dir.mkdir()

    feedback = {"C2.1": {"order": ["v1", "v2", "v3"]}}
    fp = _write_feedback(tmp_path, feedback)
    result = apply_feedback(out_dir, fp)

    assert "state-university-2023" in result.hints_written
    hints_path = doc_dir / "hints.yaml"
    data = yaml.safe_load(hints_path.read_text(encoding="utf-8"))
    assert data["order"] == ["v1", "v2", "v3"]


def test_apply_feedback_merges_with_existing_hints(tmp_path: Path) -> None:
    """Feedback merges with (does not overwrite) existing hints.yaml content.

    C2.1 = first obs of indemnification → state-university-2023.
    """
    _make_opf(tmp_path)
    out_dir = tmp_path / "out"

    doc_dir = tmp_path / "state-university-2023"
    doc_dir.mkdir()
    (doc_dir / "hints.yaml").write_text(
        yaml.dump({"signed_version": "v2", "timestamps": {"v1": "2022-01-01"}}),
        encoding="utf-8",
    )

    feedback = {"C2.1": {"provenance": "counterparty_paper"}}
    fp = _write_feedback(tmp_path, feedback)
    apply_feedback(out_dir, fp)

    data = yaml.safe_load((doc_dir / "hints.yaml").read_text(encoding="utf-8"))
    # Existing keys preserved
    assert data["signed_version"] == "v2"
    # New key added
    assert data["provenance"] == "counterparty_paper"


def test_apply_feedback_hints_fallback_to_out_dir_hints(tmp_path: Path) -> None:
    """When no corpus doc dir exists, hints go to out_dir/hints/<doc_id>.yaml.

    C2.1 = first obs of indemnification → state-university-2023.
    """
    _make_opf(tmp_path)
    out_dir = tmp_path / "out"
    # Do NOT create a doc dir this time

    feedback = {"C2.1": {"provenance": "our_paper"}}
    fp = _write_feedback(tmp_path, feedback)
    result = apply_feedback(out_dir, fp)

    assert "state-university-2023" in result.hints_written
    fallback_path = out_dir / "hints" / "state-university-2023.yaml"
    assert fallback_path.exists()
    data = yaml.safe_load(fallback_path.read_text(encoding="utf-8"))
    assert data["provenance"] == "our_paper"


# ---------------------------------------------------------------------------
# apply_feedback — VerdictStore
# ---------------------------------------------------------------------------


def test_apply_feedback_classification_writes_verdict_store(tmp_path: Path) -> None:
    """classification correction → VerdictStore entry in judge/verdicts.jsonl."""
    _make_opf(tmp_path)
    out_dir = tmp_path / "out"

    feedback = {"C1": {"classification": "governing_law"}}
    fp = _write_feedback(tmp_path, feedback)
    result = apply_feedback(out_dir, fp)

    assert result.verdicts_written >= 1
    verdicts_path = out_dir / "judge" / "verdicts.jsonl"
    assert verdicts_path.exists()
    lines = [
        json.loads(line)
        for line in verdicts_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(lines) >= 1
    # The verdict should carry the classification correction
    verdicts = [rec["verdict"] for rec in lines]
    assert any(v.get("taxonomy_id") == "governing_law" for v in verdicts)


def test_apply_feedback_classification_verdict_basis_is_judge(tmp_path: Path) -> None:
    """Classification verdict carries basis='judge' so classify_tree accepts it on replay."""
    _make_opf(tmp_path)
    out_dir = tmp_path / "out"

    feedback = {"C1": {"classification": "indemnification"}}
    fp = _write_feedback(tmp_path, feedback)
    apply_feedback(out_dir, fp)

    verdicts_path = out_dir / "judge" / "verdicts.jsonl"
    lines = [
        json.loads(line)
        for line in verdicts_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert lines, "expected a verdict to be written"
    assert all(rec["verdict"].get("basis") == "judge" for rec in lines)


def test_apply_feedback_classification_round_trips_through_judge(tmp_path: Path) -> None:
    """Regression for #70: a reclassification must replay through the judge.

    apply_feedback writes a verdict whose key must match exactly what
    StoreBackedClassificationJudge computes for the same clause node — otherwise
    the correction is a silent no-op. Apply a reclassification, then run the real
    judge over the same node and assert a store HIT (the human verdict), not a
    needs_review miss.
    """
    from types import SimpleNamespace  # noqa: PLC0415

    from playbook_engine.agent_judge import (  # noqa: PLC0415
        PendingQueue,
        StoreBackedClassificationJudge,
        VerdictStore,
    )
    from playbook_engine.clause_tree import ClauseTree  # noqa: PLC0415

    _make_opf(tmp_path)
    out_dir = tmp_path / "out"

    # Reclassify C1 (governing_law clause; cites pacific-state-college-2022/12).
    feedback = {"C1": {"classification": "indemnification"}}
    fp = _write_feedback(tmp_path, feedback)
    result = apply_feedback(out_dir, fp)
    assert result.verdicts_written >= 1

    # Load the exact node the engine would classify, and run the real judge.
    tree = ClauseTree.load(
        out_dir / "normalized" / "pacific-state-college-2022" / "v1.clauses.json"
    )
    node = tree.resolve_path("12")
    assert node is not None

    tax = SimpleNamespace(
        entries=[SimpleNamespace(id="governing_law"), SimpleNamespace(id="indemnification")]
    )
    judge = StoreBackedClassificationJudge(
        store=VerdictStore(out_dir / "judge" / "verdicts.jsonl"),
        pending=PendingQueue(tmp_path / "pending.jsonl"),
    )
    results = judge.classify_batch([node], tax)

    # Store HIT: the human verdict replays, with a judge-accepted basis.
    assert results[0].taxonomy_id == "indemnification"
    assert results[0].basis == "judge"
    # And nothing was queued for review (no miss).
    pending = tmp_path / "pending.jsonl"
    assert not pending.exists() or pending.read_text(encoding="utf-8").strip() == ""


# ---------------------------------------------------------------------------
# apply_feedback — notes
# ---------------------------------------------------------------------------


def test_apply_feedback_note_writes_viewer_notes(tmp_path: Path) -> None:
    """Free-text note → viewer_notes.md."""
    _make_opf(tmp_path)
    out_dir = tmp_path / "out"

    feedback = {"C1": {"note": "check this clause carefully"}}
    fp = _write_feedback(tmp_path, feedback)
    result = apply_feedback(out_dir, fp)

    assert result.notes_written is True
    notes_path = out_dir / "viewer_notes.md"
    assert notes_path.exists()
    content = notes_path.read_text(encoding="utf-8")
    assert "check this clause carefully" in content
    assert "C1" in content


def test_apply_feedback_note_appends_to_existing_notes(tmp_path: Path) -> None:
    """Notes are appended, not overwritten."""
    _make_opf(tmp_path)
    out_dir = tmp_path / "out"
    notes_path = out_dir / "viewer_notes.md"
    notes_path.write_text("existing notes\n", encoding="utf-8")

    feedback = {"C1": {"note": "new reviewer note"}}
    fp = _write_feedback(tmp_path, feedback)
    apply_feedback(out_dir, fp)

    content = notes_path.read_text(encoding="utf-8")
    assert "existing notes" in content
    assert "new reviewer note" in content


# ---------------------------------------------------------------------------
# apply_feedback — combined
# ---------------------------------------------------------------------------


def test_apply_feedback_combined_provenance_and_classification(tmp_path: Path) -> None:
    """Single feedback.json with both provenance flip and classification correction.

    C2.1 = first obs of indemnification → state-university-2023.
    C2 = indemnification clause → classification correction.
    """
    _make_opf(tmp_path)
    out_dir = tmp_path / "out"

    # Create corpus doc dir
    doc_dir = tmp_path / "state-university-2023"
    doc_dir.mkdir()

    feedback = {
        "C2.1": {"provenance": "counterparty_paper"},
        "C2": {"classification": "governing_law"},
    }
    fp = _write_feedback(tmp_path, feedback)
    result = apply_feedback(out_dir, fp)

    # hints.yaml written
    assert "state-university-2023" in result.hints_written
    hints_data = yaml.safe_load((doc_dir / "hints.yaml").read_text(encoding="utf-8"))
    assert hints_data["provenance"] == "counterparty_paper"

    # VerdictStore entry written
    assert result.verdicts_written >= 1
    verdicts_path = out_dir / "judge" / "verdicts.jsonl"
    lines = [
        json.loads(line)
        for line in verdicts_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert any(v["verdict"].get("taxonomy_id") == "governing_law" for v in lines)


def test_apply_feedback_unknown_item_number_skipped(tmp_path: Path) -> None:
    """Unknown item numbers (e.g. C99) are silently skipped — no crash."""
    _make_opf(tmp_path)
    out_dir = tmp_path / "out"

    feedback = {"C99": {"provenance": "counterparty_paper"}, "C99.5": {"note": "skip me"}}
    fp = _write_feedback(tmp_path, feedback)
    result = apply_feedback(out_dir, fp)
    # No crash; no hints written for unknown item
    assert "state-university-2023" not in result.hints_written


def test_apply_feedback_missing_opf_raises(tmp_path: Path) -> None:
    """FileNotFoundError raised when playbook.opf.json absent."""
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    fp = _write_feedback(tmp_path, {})
    with pytest.raises(FileNotFoundError):
        apply_feedback(out_dir, fp)


def test_apply_feedback_invalid_json_raises(tmp_path: Path) -> None:
    """ValueError raised when feedback.json is not valid JSON."""
    _make_opf(tmp_path)
    out_dir = tmp_path / "out"
    bad_fp = tmp_path / "bad.json"
    bad_fp.write_text("not json", encoding="utf-8")
    with pytest.raises(ValueError):
        apply_feedback(out_dir, bad_fp)


def test_apply_feedback_empty_feedback_no_changes(tmp_path: Path) -> None:
    """Empty feedback.json → no files written, no crash."""
    _make_opf(tmp_path)
    out_dir = tmp_path / "out"
    fp = _write_feedback(tmp_path, {})
    result = apply_feedback(out_dir, fp)
    assert result.hints_written == []
    assert result.verdicts_written == 0
    assert result.notes_written is False
    assert result.skipped == {}


# ---------------------------------------------------------------------------
# apply_feedback — issue #138: comment persistence + honest skip reporting
# ---------------------------------------------------------------------------


def test_apply_feedback_comment_writes_viewer_notes(tmp_path: Path) -> None:
    """A ``comment`` key (what the HTML viewer's Export button produces)
    is persisted to viewer_notes.md, same sink as ``note``."""
    _make_opf(tmp_path)
    out_dir = tmp_path / "out"

    feedback = {"C1": {"comment": "looks correct to me"}}
    fp = _write_feedback(tmp_path, feedback)
    result = apply_feedback(out_dir, fp)

    assert result.notes_written is True
    notes_path = out_dir / "viewer_notes.md"
    assert notes_path.exists()
    content = notes_path.read_text(encoding="utf-8")
    assert "looks correct to me" in content
    assert "C1" in content
    assert result.skipped == {}


def test_apply_feedback_comment_and_override_writes_both(tmp_path: Path) -> None:
    """Real viewer output {comment, override}: comment persists to
    viewer_notes.md AND override is embedded as a curation pin (issue #147)
    — neither is dropped."""
    _make_opf(tmp_path)
    out_dir = tmp_path / "out"

    feedback = {"C1": {"comment": "flag this", "override": "usually_conceded"}}
    fp = _write_feedback(tmp_path, feedback)
    result = apply_feedback(out_dir, fp)

    assert result.notes_written is True
    content = (out_dir / "viewer_notes.md").read_text(encoding="utf-8")
    assert "flag this" in content

    assert result.pins_written == ["C1"]
    assert result.skipped == {}

    doc = json.loads((out_dir / "playbook.opf.json").read_text(encoding="utf-8"))
    pins = doc["curation"]["pins"]
    assert len(pins) == 1
    assert pins[0]["position"] == "usually_conceded"
    assert pins[0]["clause_id"] == "clause.governing_law"


def test_apply_feedback_override_only_embeds_curation_pin(tmp_path: Path) -> None:
    """Feedback whose only key is ``override`` is embedded as a curation pin
    (issue #147) — no longer reported as skipped."""
    _make_opf(tmp_path)
    out_dir = tmp_path / "out"

    feedback = {"C1": {"override": "usually_conceded"}}
    fp = _write_feedback(tmp_path, feedback)
    result = apply_feedback(out_dir, fp)

    assert result.hints_written == []
    assert result.verdicts_written == 0
    assert result.notes_written is False
    assert result.skipped == {}
    assert result.pins_written == ["C1"]

    doc = json.loads((out_dir / "playbook.opf.json").read_text(encoding="utf-8"))
    pin = doc["curation"]["pins"][0]
    assert pin["position"] == "usually_conceded"
    assert pin["item_id"] == "C1"
    # baseline_stance records what the pin overrides FROM — this fixture's
    # governing_law clause carries a v0.1 rollup.position of "negotiable"
    # (clause_stance() falls back to rollup.position when summary is absent).
    assert pin["baseline_stance"] == "negotiable"
    assert "pinned_at" in pin


def test_apply_feedback_override_pin_updates_identity_when_present(tmp_path: Path) -> None:
    """A pin refreshes identity.content_hash/section_digests when the stored
    OPF already carries an identity block, but content_hash itself must be
    unchanged (curation is excluded from it — issue #147)."""
    doc = _make_opf(tmp_path)
    out_dir = tmp_path / "out"

    from playbook_engine.canonicalize import compute_section_digests, content_hash  # noqa: PLC0415

    doc["identity"] = {
        "content_hash": content_hash(doc),
        "section_digests": compute_section_digests(doc),
    }
    (out_dir / "playbook.opf.json").write_text(json.dumps(doc), encoding="utf-8")
    hash_before = doc["identity"]["content_hash"]
    curation_digest_before = doc["identity"]["section_digests"]["curation"]

    feedback = {"C1": {"override": "usually_conceded"}}
    fp = _write_feedback(tmp_path, feedback)
    apply_feedback(out_dir, fp)

    after = json.loads((out_dir / "playbook.opf.json").read_text(encoding="utf-8"))
    assert after["identity"]["content_hash"] == hash_before, (
        "adding a curation pin must not change content_hash"
    )
    assert after["identity"]["section_digests"]["curation"] != curation_digest_before, (
        "the curation section digest must change once a pin is added"
    )


# ---------------------------------------------------------------------------
# CLI — view render
# ---------------------------------------------------------------------------


def test_view_render_cmd_success(tmp_path: Path) -> None:
    """``playbook view render <out_dir>`` exits 0 and writes the HTML file."""
    _make_opf(tmp_path)
    runner = CliRunner()
    result = runner.invoke(cli, ["view", "render", str(tmp_path / "out")])
    assert result.exit_code == 0, result.output
    assert "OK" in result.output
    html_path = tmp_path / "out" / "playbook.review.html"
    assert html_path.exists()


def test_view_render_cmd_custom_out(tmp_path: Path) -> None:
    """``--out`` flag writes to the specified path."""
    _make_opf(tmp_path)
    custom_out = tmp_path / "my-review.html"
    runner = CliRunner()
    result = runner.invoke(cli, ["view", "render", str(tmp_path / "out"), "--out", str(custom_out)])
    assert result.exit_code == 0, result.output
    assert custom_out.exists()


def test_view_render_cmd_missing_opf_exits_nonzero(tmp_path: Path) -> None:
    """``view render`` exits non-zero when playbook.opf.json is absent."""
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    runner = CliRunner()
    result = runner.invoke(cli, ["view", "render", str(out_dir)])
    assert result.exit_code != 0


def test_view_render_html_contains_numbered_items(tmp_path: Path) -> None:
    """Rendered HTML from CLI contains C1 and C1.1 items."""
    _make_opf(tmp_path)
    runner = CliRunner()
    runner.invoke(cli, ["view", "render", str(tmp_path / "out")])
    html = (tmp_path / "out" / "playbook.review.html").read_text(encoding="utf-8")
    assert "C1" in html
    assert "C1.1" in html


# ---------------------------------------------------------------------------
# CLI — view apply
# ---------------------------------------------------------------------------


def test_view_apply_cmd_success(tmp_path: Path) -> None:
    """``playbook view apply <out_dir> <feedback.json>`` exits 0.

    C2.1 = first obs of indemnification → state-university-2023.
    """
    _make_opf(tmp_path)
    doc_dir = tmp_path / "state-university-2023"
    doc_dir.mkdir()

    feedback = {"C2.1": {"provenance": "counterparty_paper"}}
    fp = _write_feedback(tmp_path, feedback)

    runner = CliRunner()
    result = runner.invoke(cli, ["view", "apply", str(tmp_path / "out"), str(fp)])
    assert result.exit_code == 0, result.output
    assert "OK" in result.output


def test_view_apply_cmd_writes_hints(tmp_path: Path) -> None:
    """CLI apply writes hints.yaml for provenance correction.

    C2.1 = first obs of indemnification → state-university-2023.
    """
    _make_opf(tmp_path)
    doc_dir = tmp_path / "state-university-2023"
    doc_dir.mkdir()

    feedback = {"C2.1": {"provenance": "counterparty_paper"}}
    fp = _write_feedback(tmp_path, feedback)

    runner = CliRunner()
    runner.invoke(cli, ["view", "apply", str(tmp_path / "out"), str(fp)])

    hints_path = doc_dir / "hints.yaml"
    assert hints_path.exists()
    data = yaml.safe_load(hints_path.read_text(encoding="utf-8"))
    assert data["provenance"] == "counterparty_paper"


def test_view_apply_cmd_missing_opf_exits_nonzero(tmp_path: Path) -> None:
    """``view apply`` exits non-zero when playbook.opf.json is absent."""
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    fp = _write_feedback(tmp_path, {})
    runner = CliRunner()
    result = runner.invoke(cli, ["view", "apply", str(out_dir), str(fp)])
    assert result.exit_code != 0


def test_view_apply_cmd_invalid_feedback_exits_nonzero(tmp_path: Path) -> None:
    """``view apply`` exits non-zero when feedback.json is invalid JSON."""
    _make_opf(tmp_path)
    bad_fp = tmp_path / "bad.json"
    bad_fp.write_text("not valid json", encoding="utf-8")
    runner = CliRunner()
    result = runner.invoke(cli, ["view", "apply", str(tmp_path / "out"), str(bad_fp)])
    assert result.exit_code != 0


def test_view_apply_cmd_comment_only_reports_ok(tmp_path: Path) -> None:
    """A comment-only feedback file is honestly applied and reports OK."""
    _make_opf(tmp_path)
    feedback = {"C1": {"comment": "double check this"}}
    fp = _write_feedback(tmp_path, feedback)

    runner = CliRunner()
    result = runner.invoke(cli, ["view", "apply", str(tmp_path / "out"), str(fp)])
    assert result.exit_code == 0, result.output
    assert "OK" in result.output
    notes_path = tmp_path / "out" / "viewer_notes.md"
    assert notes_path.exists()
    assert "double check this" in notes_path.read_text(encoding="utf-8")


def test_view_apply_cmd_override_only_reports_ok_and_pins(tmp_path: Path) -> None:
    """Issue #147: an ``override`` correction is now honored (embedded
    curation pin), so it reports success, not "not applied"."""
    _make_opf(tmp_path)
    feedback = {"C1": {"override": "usually_conceded"}}
    fp = _write_feedback(tmp_path, feedback)

    runner = CliRunner()
    result = runner.invoke(cli, ["view", "apply", str(tmp_path / "out"), str(fp)])
    assert result.exit_code == 0, result.output
    assert "OK  feedback applied" in result.output
    assert "not applied" not in result.output
    assert "position pinned" in result.output


# ---------------------------------------------------------------------------
# Acceptance criteria — explicit AC checks
# ---------------------------------------------------------------------------


def test_ac_html_contains_numbered_clause_items(tmp_path: Path) -> None:
    """AC: HTML rendered from fixture contains numbered clause items C1, C1.1."""
    _make_opf(tmp_path)
    html = render_review_html(tmp_path / "out")
    assert "C1" in html
    assert "C1.1" in html


def test_ac_html_contains_embedded_json(tmp_path: Path) -> None:
    """AC: HTML contains the embedded OPF JSON."""
    _make_opf(tmp_path)
    html = render_review_html(tmp_path / "out")
    # The full JSON is embedded; opf_version is a marker
    assert "opf_version" in html


def test_ac_html_contains_evidence_citations(tmp_path: Path) -> None:
    """AC: HTML contains evidence citations from example_ref."""
    _make_opf(tmp_path)
    html = render_review_html(tmp_path / "out")
    assert "state-university-2023" in html


def test_ac_html_requires_no_network(tmp_path: Path) -> None:
    """AC: opening HTML requires no network (no external URLs)."""
    _make_opf(tmp_path)
    html = render_review_html(tmp_path / "out")
    import re

    # No external URLs in href or src
    external_urls = re.findall(r'(?:src|href)=["\']https?://', html)
    assert external_urls == [], f"Found external URLs: {external_urls}"


def test_ac_apply_writes_hints_yaml_for_provenance(tmp_path: Path) -> None:
    """AC: view --apply on a provenance flip writes the expected hints.yaml.

    C2.1 = first obs of indemnification → state-university-2023.
    """
    _make_opf(tmp_path)
    out_dir = tmp_path / "out"
    doc_dir = tmp_path / "state-university-2023"
    doc_dir.mkdir()

    feedback = {"C2.1": {"provenance": "counterparty_paper"}}
    fp = _write_feedback(tmp_path, feedback)
    result = apply_feedback(out_dir, fp)

    assert "state-university-2023" in result.hints_written
    data = yaml.safe_load((doc_dir / "hints.yaml").read_text(encoding="utf-8"))
    assert data["provenance"] == "counterparty_paper"


def test_ac_apply_writes_verdict_store_for_classification(tmp_path: Path) -> None:
    """AC: view --apply on a reclassification writes the expected VerdictStore entry."""
    _make_opf(tmp_path)
    out_dir = tmp_path / "out"

    feedback = {"C1": {"classification": "governing_law"}}
    fp = _write_feedback(tmp_path, feedback)
    result = apply_feedback(out_dir, fp)

    assert result.verdicts_written >= 1
    verdicts_path = out_dir / "judge" / "verdicts.jsonl"
    lines = [
        json.loads(line)
        for line in verdicts_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert any(rec["verdict"].get("taxonomy_id") == "governing_law" for rec in lines)


def test_ac_item_numbering_deterministic(tmp_path: Path) -> None:
    """AC: item numbering is deterministic and stable for the same playbook input."""
    _make_opf(tmp_path)
    doc = json.loads((tmp_path / "out" / "playbook.opf.json").read_text())
    result1 = [(n, k) for n, k, _ in _build_index(doc)]
    result2 = [(n, k) for n, k, _ in _build_index(doc)]
    assert result1 == result2


# ---------------------------------------------------------------------------
# Alias -> real name resolution at render time (issue #146)
# ---------------------------------------------------------------------------

_ALIAS_CLAUSES = [
    {
        "id": "clause.indemnification",
        "taxonomy_id": "indemnification",
        "title": "Indemnification",
        "our_standard": {
            "text": "Each party shall indemnify the other.",
            "source_ref": {"document_id": "template", "version": "template", "clause_path": "8"},
        },
        "observed_positions": [
            {
                "text_summary": "Counterparty-1 requested mutual indemnification.",
                "example_ref": {
                    "document_id": "Counterparty-1-2023",
                    "version": 3,
                    "clause_path": "8",
                },
                "deviation": "none",
                "risk_delta": {"direction": "neutral", "magnitude": "none"},
                "provenance": "counterparty_paper",
                "outcome": "signed",
                "precedent_count": 7,
            }
        ],
        "rollup": {
            "position": "standard",
            "confidence": {
                "score": 0.9,
                "basis": "precedent_count",
                "n_our_paper": 10,
                "n_counterparty_paper": 1,
            },
        },
    }
]

_ALIAS_MAP = {"Counterparty-1": "State University"}


def test_render_html_resolves_aliases_to_real_names_when_map_given(tmp_path: Path) -> None:
    """render_review_html substitutes aliases for real names when alias_map is given."""
    _make_opf(tmp_path, clauses=_ALIAS_CLAUSES)
    html = render_review_html(tmp_path / "out", alias_map=_ALIAS_MAP)
    assert "State University requested mutual indemnification." in html
    assert "Counterparty-1" not in html


def test_render_html_leaves_aliases_unresolved_without_map(tmp_path: Path) -> None:
    """Without alias_map, the rendered HTML still shows only aliases (default-safe)."""
    _make_opf(tmp_path, clauses=_ALIAS_CLAUSES)
    html = render_review_html(tmp_path / "out")
    assert "Counterparty-1 requested mutual indemnification." in html
    assert "State University" not in html


def test_render_html_alias_resolution_never_mutates_stored_opf(tmp_path: Path) -> None:
    """Rendering with alias_map does not rewrite playbook.opf.json on disk."""
    _make_opf(tmp_path, clauses=_ALIAS_CLAUSES)
    opf_path = tmp_path / "out" / "playbook.opf.json"
    before = opf_path.read_text(encoding="utf-8")

    render_review_html(tmp_path / "out", alias_map=_ALIAS_MAP)

    after = opf_path.read_text(encoding="utf-8")
    assert after == before
    assert "Counterparty-1" in after
    assert "State University" not in after


def test_render_html_alias_resolution_applies_to_embedded_drill_down_json(
    tmp_path: Path,
) -> None:
    """The embedded playbook-data script also resolves aliases (internal readability)."""
    _make_opf(tmp_path, clauses=_ALIAS_CLAUSES)
    html = render_review_html(tmp_path / "out", alias_map=_ALIAS_MAP)
    start = html.index('<script id="playbook-data"')
    end = html.index("</script>", start)
    embedded = html[start:end]
    assert "State University" in embedded
    assert "Counterparty-1" not in embedded


def test_load_alias_map_reads_json_file(tmp_path: Path) -> None:
    """load_alias_map reads the held-out alias->entity sidecar as a plain dict."""
    from playbook_engine.viewer import load_alias_map

    path = tmp_path / "alias_map.json"
    path.write_text(json.dumps(_ALIAS_MAP), encoding="utf-8")
    assert load_alias_map(path) == _ALIAS_MAP


def test_load_alias_map_missing_file_raises(tmp_path: Path) -> None:
    from playbook_engine.viewer import load_alias_map

    with pytest.raises(FileNotFoundError):
        load_alias_map(tmp_path / "does-not-exist.json")


def test_view_render_cmd_with_alias_map_resolves_names(tmp_path: Path) -> None:
    """CLI: `playbook view render --alias-map ...` resolves aliases in the output HTML."""
    _make_opf(tmp_path, clauses=_ALIAS_CLAUSES)
    out_dir = tmp_path / "out"
    alias_map_path = tmp_path / "alias_map.json"
    alias_map_path.write_text(json.dumps(_ALIAS_MAP), encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(
        cli, ["view", "render", str(out_dir), "--alias-map", str(alias_map_path)]
    )

    assert result.exit_code == 0, result.output
    html = (out_dir / "playbook.review.html").read_text(encoding="utf-8")
    assert "State University requested mutual indemnification." in html
    assert "Counterparty-1" not in html
    # Stored OPF is untouched.
    stored = (out_dir / "playbook.opf.json").read_text(encoding="utf-8")
    assert "Counterparty-1" in stored
