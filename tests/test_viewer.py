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
from playbook_engine.floor_candidates import _Q5_REJECTION_COMMENT, derive_interview_q4_candidates
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


def _floor_candidate(
    *,
    id: str = "cand-001",  # noqa: A002
    statement: str = "Never accept uncapped liability.",
    rationale: str = "Proposed then reversed before signing in 2 deals.",
    source: str = "reversal",
    citations: list[dict] | None = None,
    decision: str | None = None,
    comment: str | None = None,
) -> dict:
    """A synthetic floor.candidates.json candidate (issue #90 fixtures)."""
    if citations is None:
        citations = [{"document_id": "state-university-2023", "version": 3, "clause_path": "8.1"}]
    candidate: dict = {
        "id": id,
        "statement": statement,
        "rationale": rationale,
        "source": source,
        "citations": citations,
    }
    if decision is not None:
        candidate["decision"] = decision
    if comment is not None:
        candidate["comment"] = comment
    return candidate


def _write_floor_candidates(tmp_path: Path, candidates: list[dict]) -> Path:
    """Write out/floor.candidates.json — call _make_opf(tmp_path) first."""
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "floor.candidates.json"
    p.write_text(json.dumps({"candidates": candidates}), encoding="utf-8")
    return p


def _set_floor_invariants(tmp_path: Path, invariants: list[dict]) -> None:
    """Mutate out/playbook.opf.json (already written by _make_opf) to carry
    a floor.invariants list — for "already signed" candidate fixtures."""
    opf_path = tmp_path / "out" / "playbook.opf.json"
    doc = json.loads(opf_path.read_text(encoding="utf-8"))
    doc["floor"] = {"invariants": invariants}
    opf_path.write_text(json.dumps(doc), encoding="utf-8")


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


def test_render_html_escapes_script_breakout_in_embedded_json(tmp_path: Path) -> None:
    """A corpus-derived full_text containing '</script>' must not break out of
    the embedded playbook-data block (issue #31 — stored XSS).

    Covers the un-aliased path, where the embedded JSON is the raw on-disk
    bytes rather than a fresh json.dumps of a resolved copy.
    """
    breakout = "</script><script>window.x=1</script>"
    clauses = [
        {
            "id": "clause.indemnification",
            "taxonomy_id": "indemnification",
            "title": "Indemnification",
            "our_standard": None,
            "observed_positions": [
                {
                    "text_summary": "Mutual indemnification.",
                    "full_text": breakout,
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
    ]
    _make_opf(tmp_path, clauses=clauses)
    html = render_review_html(tmp_path / "out")

    # Exactly the two script tags we render ourselves must remain — none of
    # the corpus text's own "</script>" may have split the block open.
    # Only our own two script tags' closes may appear unescaped; the
    # corpus-derived "</script>" must be escaped to "<\/script>" so it can
    # never terminate the enclosing script element early.
    assert html.count("</script>") == 2
    assert breakout not in html
    assert "<\\/script><script>window.x=1<\\/script>" in html


def test_render_html_escapes_script_breakout_with_alias_map(tmp_path: Path) -> None:
    """Same breakout, but through the alias-resolved (json.dumps) path."""
    breakout = "</script><script>window.x=1</script>"
    clauses = [
        {
            "id": "clause.indemnification",
            "taxonomy_id": "indemnification",
            "title": "Indemnification",
            "our_standard": None,
            "observed_positions": [
                {
                    "text_summary": "Mutual indemnification.",
                    "full_text": breakout,
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
    ]
    _make_opf(tmp_path, clauses=clauses)
    html = render_review_html(tmp_path / "out", alias_map={"ALIAS_1": "Real Party Name"})

    assert html.count("</script>") == 2
    assert breakout not in html
    assert "<\\/script><script>window.x=1<\\/script>" in html


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
# render_review_html — "Proposed hard lines" checklist (issue #90)
# ---------------------------------------------------------------------------


def test_render_html_no_floor_candidates_file_no_section(tmp_path: Path) -> None:
    """No floor.candidates.json at all -> no section, no empty shell.

    The static User Guide overlay mentions "Proposed hard lines" and
    "already signed" unconditionally (it documents every page feature
    regardless of whether THIS render happens to use it — same as it
    always explains "preferred variations" even with none present), so the
    absence check here is the structural section marker, not that prose.
    """
    _make_opf(tmp_path)
    html = render_review_html(tmp_path / "out")
    assert 'id="floor-candidates"' not in html
    assert 'class="floor-decision"' not in html


def test_render_html_empty_floor_candidates_no_section(tmp_path: Path) -> None:
    """floor.candidates.json exists but has zero candidates -> still no
    section (acceptance criteria: no empty shell either way)."""
    _make_opf(tmp_path)
    _write_floor_candidates(tmp_path, [])
    html = render_review_html(tmp_path / "out")
    assert 'id="floor-candidates"' not in html
    assert 'class="floor-decision"' not in html


def test_render_html_all_malformed_floor_candidates_no_section(tmp_path: Path) -> None:
    """floor.candidates.json's candidates list holds only malformed
    (non-dict) entries -> still no section, no empty shell (issue #90
    review finding 5: the early return checked `not candidates`, true only
    for a genuinely EMPTY list -- a list of garbage is non-empty, so the
    section header + help paragraph rendered with zero actual rows)."""
    _make_opf(tmp_path)
    _write_floor_candidates(tmp_path, ["not-a-dict", 42])  # type: ignore[list-item]
    html = render_review_html(tmp_path / "out")
    assert 'id="floor-candidates"' not in html
    assert 'class="floor-decision"' not in html


def test_render_html_floor_candidates_section_present(tmp_path: Path) -> None:
    _make_opf(tmp_path)
    _write_floor_candidates(tmp_path, [_floor_candidate()])
    html = render_review_html(tmp_path / "out")
    assert "Proposed hard lines" in html
    assert 'id="floor-candidates"' in html
    assert "Never accept uncapped liability." in html
    assert 'data-candidate-id="cand-001"' in html


def test_render_html_undecided_candidate_has_three_way_control(tmp_path: Path) -> None:
    _make_opf(tmp_path)
    _write_floor_candidates(tmp_path, [_floor_candidate()])
    html = render_review_html(tmp_path / "out")
    assert 'class="floor-decision"' in html
    assert 'value="accept"' in html
    assert 'value="reject"' in html
    assert 'value="undecided" checked' in html  # default undecided
    # No status badge rendered for a still-pending candidate (the User
    # Guide's own static prose mentions both words unconditionally, so the
    # precise check is for the badge markup itself, not the bare words).
    assert ">already signed<" not in html
    assert ">rejected<" not in html


def test_render_html_floor_candidate_shows_reversal_citation(tmp_path: Path) -> None:
    _make_opf(tmp_path)
    _write_floor_candidates(
        tmp_path,
        [
            _floor_candidate(
                citations=[
                    {"document_id": "state-university-2023", "version": 3, "clause_path": "8.1"}
                ]
            )
        ],
    )
    html = render_review_html(tmp_path / "out")
    assert "state-university-2023" in html
    assert "v3" in html
    assert "8.1" in html


def test_render_html_floor_candidate_interview_q4_shows_marker_not_citation(
    tmp_path: Path,
) -> None:
    _make_opf(tmp_path)
    _write_floor_candidates(
        tmp_path,
        [
            _floor_candidate(
                id="cand-002",
                statement="Never accept IP assignment.",
                source="interview_q4",
                citations=[],
            )
        ],
    )
    html = render_review_html(tmp_path / "out")
    assert "Posture interview" in html
    assert "Q4" in html


def test_render_html_floor_candidate_already_signed_renders_inert(tmp_path: Path) -> None:
    """A candidate whose derived id is already present in floor.invariants
    renders as an inert 'already signed' row — no accept/reject control."""
    _make_opf(tmp_path)
    candidate = _floor_candidate()
    _write_floor_candidates(tmp_path, [candidate])
    _set_floor_invariants(
        tmp_path,
        [
            {
                "id": "never-accept-uncapped-liability",
                "statement": "Never accept uncapped liability.",
                "rationale": "Accepted via review feedback (floor candidate cand-001).",
            }
        ],
    )

    html = render_review_html(tmp_path / "out")

    assert "Proposed hard lines" in html
    assert ">already signed<" in html
    # No live control rendered for this (now inert) candidate.
    assert 'class="floor-decision"' not in html


def test_render_html_floor_candidate_rejected_renders_inert(tmp_path: Path) -> None:
    """A candidate recorded as rejected in floor.candidates.json renders as
    an inert 'rejected' row — no accept/reject control, and it is not
    re-proposed as an open item."""
    _make_opf(tmp_path)
    _write_floor_candidates(tmp_path, [_floor_candidate(decision="rejected")])

    html = render_review_html(tmp_path / "out")

    assert "Proposed hard lines" in html
    assert ">rejected<" in html
    assert 'class="floor-decision"' not in html


def test_render_html_floor_candidate_q5_auto_rejected_stays_live_control(
    tmp_path: Path,
) -> None:
    """issue #105 review finding 1: a candidate the Posture interview's Q5
    ("flexible_clauses") answer auto-rejected is NOT the same as a human
    rejection — it must render with its live accept/reject controls intact
    (recognizable by its ``decision: "rejected"`` PLUS the attributing
    :data:`_Q5_REJECTION_COMMENT`, vs. a bare human rejection which never
    carries that comment), unlike ``..._rejected_renders_inert`` above."""
    _make_opf(tmp_path)
    _write_floor_candidates(
        tmp_path,
        [_floor_candidate(decision="rejected", comment=_Q5_REJECTION_COMMENT)],
    )

    html = render_review_html(tmp_path / "out")

    assert "Proposed hard lines" in html
    assert 'class="floor-decision"' in html
    assert 'value="accept"' in html
    assert 'value="reject" checked' in html
    assert _Q5_REJECTION_COMMENT in html
    # Not rendered as the inert "already rejected" badge a human decision gets.
    assert ">rejected<" not in html


def test_render_html_floor_candidate_q5_auto_rejected_becomes_inert_after_human_confirms(
    tmp_path: Path,
) -> None:
    """issue #105 review round 2 finding 3: a reviewer who AGREES with the
    Q5 recommendation and explicitly rejects the candidate must be able to
    record that agreement -- the row must stop re-asking (stop rendering
    "Recommended reject" with live controls) and become the same inert
    "rejected" badge a plain human rejection gets."""
    _make_opf(tmp_path)
    out_dir = tmp_path / "out"
    _write_floor_candidates(
        tmp_path,
        [_floor_candidate(decision="rejected", comment=_Q5_REJECTION_COMMENT)],
    )

    feedback = {"floor": {"cand-001": {"decision": "reject"}}}
    fp = _write_feedback(tmp_path, feedback)
    result = apply_feedback(out_dir, fp)
    assert result.floor_rejected == ["cand-001"]

    html = render_review_html(out_dir)

    assert "Proposed hard lines" in html
    assert ">rejected<" in html
    assert _Q5_REJECTION_COMMENT not in html
    assert "Recommended reject" not in html
    assert 'class="floor-decision"' not in html


def test_render_html_floor_candidate_q5_auto_rejected_becomes_signed_after_human_accepts(
    tmp_path: Path,
) -> None:
    """issue #105 review round 2 finding 3 (generalized): a reviewer who
    DISAGREES with the Q5 recommendation and explicitly accepts the
    candidate must not have the Q5 attribution comment survive the
    override -- the row must stop re-asking and become the same inert
    "already signed" badge a plain human accept gets, never a "rejected"
    row still carrying "named as a willing concession"."""
    _make_opf(tmp_path)
    out_dir = tmp_path / "out"
    _write_floor_candidates(
        tmp_path,
        [_floor_candidate(decision="rejected", comment=_Q5_REJECTION_COMMENT)],
    )

    feedback = {"floor": {"cand-001": {"decision": "accept"}}}
    fp = _write_feedback(tmp_path, feedback)
    result = apply_feedback(out_dir, fp)
    assert result.floor_promoted == ["cand-001"]

    written = json.loads((out_dir / "floor.candidates.json").read_text(encoding="utf-8"))
    assert "comment" not in written["candidates"][0]

    html = render_review_html(out_dir)

    assert "Proposed hard lines" in html
    assert ">already signed<" in html
    assert _Q5_REJECTION_COMMENT not in html
    assert "Recommended reject" not in html
    assert 'class="floor-decision"' not in html


def test_render_html_floor_candidates_mixed_states(tmp_path: Path) -> None:
    """One undecided, one signed, one rejected — each candidate renders its
    own independent state (no cross-contamination)."""
    _make_opf(tmp_path)
    undecided = _floor_candidate(id="cand-001", statement="Never accept uncapped liability.")
    signed = _floor_candidate(id="cand-002", statement="Never accept IP assignment.", citations=[])
    rejected = _floor_candidate(
        id="cand-003",
        statement="Never accept broad non-compete.",
        citations=[],
        decision="rejected",
    )
    _write_floor_candidates(tmp_path, [undecided, signed, rejected])
    _set_floor_invariants(
        tmp_path,
        [
            {
                "id": "never-accept-ip-assignment",
                "statement": "Never accept IP assignment.",
                "rationale": "Accepted via review feedback (floor candidate cand-002).",
            }
        ],
    )

    html = render_review_html(tmp_path / "out")

    assert 'data-candidate-id="cand-001"' in html
    assert 'data-candidate-id="cand-002"' in html
    assert 'data-candidate-id="cand-003"' in html
    assert "already signed" in html
    assert ">rejected<" in html
    assert 'value="accept"' in html  # cand-001's live control


def test_render_html_floor_candidate_q4_already_signed_via_other_id_renders_inert(
    tmp_path: Path,
) -> None:
    """Issue #90 review finding 1 regression: an interview_q4-sourced
    candidate's item may already be signed into floor.invariants via
    promote_interview_q4_invariants (issue #89) -- under a DIFFERENT id
    (_slugify_statement_item's "liability-caps", not
    candidate_invariant_id's "never-accept-liability-caps") and different
    wording ("Do not concede on ..." vs this candidate's draft "Never
    accept ..."). The two must still be recognised as the same decided
    item -- an inert row, not a live control."""
    from playbook_engine.floor_candidates import promote_interview_q4_invariants

    _make_opf(tmp_path)
    item = "liability caps"
    invariants = promote_interview_q4_invariants(
        {"sacred_clauses": item}, posture_version=1, existing_invariants=[]
    )
    _set_floor_invariants(tmp_path, invariants)
    candidate = _floor_candidate(
        id="cand-001",
        # Built by the REAL producer, not a hardcoded string: this statement is
        # regex-parsed back into the Q4 item by `candidate_q4_invariant_id`, so
        # a hardcoded copy silently decouples from the producer and this test
        # keeps passing while the already-signed guard is dead in production.
        statement=derive_interview_q4_candidates({"sacred_clauses": item})[0].statement,
        rationale='Named as non-negotiable in the Posture interview (Q4 "sacred_clauses").',
        source="interview_q4",
        citations=[],
    )
    _write_floor_candidates(tmp_path, [candidate])

    html = render_review_html(tmp_path / "out")

    assert "Proposed hard lines" in html
    assert ">already signed<" in html
    assert 'class="floor-decision"' not in html


def test_render_html_floor_candidate_already_signed_with_alias_map_still_inert(
    tmp_path: Path,
) -> None:
    """Issue #90 review finding 2 regression: candidate_invariant_id must be
    derived from the RAW (unresolved) candidate -- the same input
    apply_feedback uses -- not the alias-resolved display copy. A reversal
    candidate whose statement quotes corpus text containing an alias (the
    case an unclassified reversal's text snippet can produce) must still
    render as inert 'already signed' when rendered WITH an alias map, using
    the SAME id an unaliased render (or an actual accept) would derive."""
    from playbook_engine.floor_candidates import candidate_invariant_id

    _make_opf(tmp_path)
    raw_statement = 'Never accept "ENTITY_1 shall bear all costs".'
    candidate = _floor_candidate(id="cand-001", statement=raw_statement, source="reversal")
    _write_floor_candidates(tmp_path, [candidate])
    inv_id = candidate_invariant_id(candidate)
    _set_floor_invariants(
        tmp_path,
        [
            {
                "id": inv_id,
                "statement": raw_statement,
                "rationale": "Accepted via review feedback (floor candidate cand-001).",
            }
        ],
    )

    html_no_map = render_review_html(tmp_path / "out")
    html_with_map = render_review_html(
        tmp_path / "out", alias_map={"ENTITY_1": "Statewide College"}
    )

    for html in (html_no_map, html_with_map):
        assert ">already signed<" in html
        assert 'class="floor-decision"' not in html
    assert "Statewide College" in html_with_map


def test_render_html_floor_candidate_taxonomy_already_signed_renders_inert(
    tmp_path: Path,
) -> None:
    """Issue #102 regression -- the real failure reported in the issue: a
    hand-authored invariant ("limitation-of-liability-not-unilateral")
    covers the same clause taxonomy as a reversal candidate's draft
    statement ("Do not concede on limitation of liability."), but the two
    share ZERO slug overlap -- neither candidate_invariant_id nor
    candidate_q4_invariant_id would ever match. Only the taxonomy_id /
    x_taxonomy_id signal can recognise this as already signed; without it
    the reviewer sees a live accept control for a hard line already on the
    books, and accepting it would append a second, conflicting invariant."""
    _make_opf(tmp_path)
    candidate = {
        "id": "cand-001",
        "statement": "Do not concede on limitation of liability.",
        "rationale": "Proposed then reversed before signing in 2 deals.",
        "source": "reversal",
        "citations": [{"document_id": "state-university-2023", "version": 3, "clause_path": "8.1"}],
        "taxonomy_id": "limitation_of_liability",
    }
    _write_floor_candidates(tmp_path, [candidate])
    _set_floor_invariants(
        tmp_path,
        [
            {
                "id": "limitation-of-liability-not-unilateral",
                "statement": "Limitation of liability, if present, must not be unilateral "
                "in the counterparty's favor.",
                "rationale": "Hand-authored via `playbook floor sign`.",
                "x_taxonomy_id": "limitation_of_liability",
            }
        ],
    )

    html = render_review_html(tmp_path / "out")

    assert "Proposed hard lines" in html
    assert ">already signed<" in html
    # No live control rendered for this (now inert) candidate.
    assert 'class="floor-decision"' not in html


def test_render_html_floor_candidate_different_taxonomy_still_live_control(
    tmp_path: Path,
) -> None:
    """A candidate's taxonomy_id must be an actual MATCH, not mere presence
    of the key on either side, to suppress the row."""
    _make_opf(tmp_path)
    candidate = {
        "id": "cand-001",
        "statement": "Do not concede on uncapped liability.",
        "rationale": "Proposed then reversed before signing in 2 deals.",
        "source": "reversal",
        "citations": [{"document_id": "state-university-2023", "version": 3, "clause_path": "8.1"}],
        "taxonomy_id": "uncapped_liability",
    }
    _write_floor_candidates(tmp_path, [candidate])
    _set_floor_invariants(
        tmp_path,
        [
            {
                "id": "some-other-hard-line",
                "statement": "IP assignment must be mutual.",
                "rationale": "Hand-authored via `playbook floor sign`.",
                "x_taxonomy_id": "ip_assignment",
            }
        ],
    )

    html = render_review_html(tmp_path / "out")

    assert ">already signed<" not in html
    assert 'value="accept"' in html


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
    """When no corpus doc dir can be found, the correction is parked at
    out_dir/hints/<doc_id>.yaml and reported as SKIPPED, not a success
    (issue #169) — a dead file no engine code reads back must never be
    counted in hints_written, which the CLI treats as "applied".

    C2.1 = first obs of indemnification → state-university-2023.
    """
    _make_opf(tmp_path)
    out_dir = tmp_path / "out"
    # Do NOT create a doc dir this time

    feedback = {"C2.1": {"provenance": "our_paper"}}
    fp = _write_feedback(tmp_path, feedback)
    result = apply_feedback(out_dir, fp)

    # Not a success: the doc_id must NOT appear in hints_written.
    assert "state-university-2023" not in result.hints_written
    # It IS reported as not-applied, with a message a human can act on.
    assert "hints:state-university-2023" in result.skipped
    message = " ".join(result.skipped["hints:state-university-2023"])
    assert "state-university-2023" in message
    assert "not read by any pipeline stage" in message.lower()

    # The best-effort fallback copy still exists, for manual recovery.
    fallback_path = out_dir / "hints" / "state-university-2023.yaml"
    assert fallback_path.exists()
    data = yaml.safe_load(fallback_path.read_text(encoding="utf-8"))
    assert data["provenance"] == "our_paper"


def test_apply_feedback_hints_uses_explicit_corpus_dir(tmp_path: Path) -> None:
    """corpus_dir, when given, is searched even when it is not out_dir's
    parent/grandparent — the documented Docker layout, where the corpus is
    mounted as a SIBLING of out_dir (/work/corpus vs. /work/out), not its
    parent (issue #169).

    C2.1 = first obs of indemnification → state-university-2023.
    """
    _make_opf(tmp_path)
    out_dir = tmp_path / "work" / "out"
    out_dir.mkdir(parents=True)
    corpus_dir = tmp_path / "work" / "corpus"
    doc_dir = corpus_dir / "state-university-2023"
    doc_dir.mkdir(parents=True)

    # Re-point the OPF we just wrote at the new out_dir location.
    opf_src = tmp_path / "out" / "playbook.opf.json"
    (out_dir / "playbook.opf.json").write_text(
        opf_src.read_text(encoding="utf-8"), encoding="utf-8"
    )

    feedback = {"C2.1": {"provenance": "counterparty_paper"}}
    fp = _write_feedback(tmp_path, feedback)
    result = apply_feedback(out_dir, fp, corpus_dir=corpus_dir)

    assert "state-university-2023" in result.hints_written
    assert result.skipped == {}
    data = yaml.safe_load((doc_dir / "hints.yaml").read_text(encoding="utf-8"))
    assert data["provenance"] == "counterparty_paper"


def test_apply_feedback_hints_reverses_pseudonymized_document_id(tmp_path: Path) -> None:
    """When the cited document_id is a pseudonymized alias (issue #153), the
    held-out out_dir/alias_map.json sidecar is used to reverse it back to the
    real corpus folder name before searching (issue #169) — without this,
    the alias matches no raw-named corpus folder anywhere and the correction
    silently no-ops into the dead out_dir/hints/ fallback.

    C2.1 = first obs of indemnification → state-university-2023, aliased to
    Counterparty-1-2023.
    """
    doc = _make_opf(tmp_path)
    doc["clauses"][0]["observed_positions"][0]["example_ref"]["document_id"] = "counterparty-1-2023"
    out_dir = tmp_path / "out"
    (out_dir / "playbook.opf.json").write_text(json.dumps(doc), encoding="utf-8")
    (out_dir / "alias_map.json").write_text(
        json.dumps({"Counterparty-1": "State University"}), encoding="utf-8"
    )

    # Raw-named corpus folder, exactly as a human would find it on disk.
    doc_dir = tmp_path / "state-university-2023"
    doc_dir.mkdir()

    feedback = {"C2.1": {"provenance": "our_paper"}}
    fp = _write_feedback(tmp_path, feedback)
    result = apply_feedback(out_dir, fp)

    assert "counterparty-1-2023" in result.hints_written
    assert result.skipped == {}
    data = yaml.safe_load((doc_dir / "hints.yaml").read_text(encoding="utf-8"))
    assert data["provenance"] == "our_paper"


def test_apply_feedback_hints_unwritable_corpus_dir_reported_as_skipped(
    tmp_path: Path,
) -> None:
    """A corpus document directory that exists but cannot be written (the
    Docker corpus mount is read-only) must be reported as skipped, not
    counted as a success — issue #169's other failure mode, distinct from
    "directory not found".

    C2.1 = first obs of indemnification → state-university-2023.
    """
    _make_opf(tmp_path)
    out_dir = tmp_path / "out"
    doc_dir = tmp_path / "state-university-2023"
    doc_dir.mkdir()
    doc_dir.chmod(0o500)  # read + execute, no write — simulates a ro mount
    try:
        feedback = {"C2.1": {"provenance": "our_paper"}}
        fp = _write_feedback(tmp_path, feedback)
        result = apply_feedback(out_dir, fp)

        assert "state-university-2023" not in result.hints_written
        assert "hints:state-university-2023" in result.skipped
        fallback_path = out_dir / "hints" / "state-university-2023.yaml"
        assert fallback_path.exists()
    finally:
        doc_dir.chmod(0o700)  # restore so tmp_path cleanup can remove it


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

    # Must mirror the taxonomy embedded in playbook.opf.json field-for-field:
    # apply_feedback stamps its verdict with the classify rubric digested from
    # the OPF's taxonomy (id + label + description of classifier-eligible
    # entries), and a mismatched rubric would re-queue the correction as stale
    # rather than replay it.
    tax = SimpleNamespace(
        entries=[
            SimpleNamespace(id="governing_law", label="Governing Law", status="active"),
            SimpleNamespace(id="indemnification", label="Indemnification", status="active"),
        ]
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
# apply_feedback — issue #174: stale _export.content_hash binding
# ---------------------------------------------------------------------------


def _set_identity_content_hash(tmp_path: Path, content_hash_value: str) -> None:
    """Mutate out/playbook.opf.json (already written by _make_opf) to carry
    an identity.content_hash — for #174 stale-export fixtures."""
    opf_path = tmp_path / "out" / "playbook.opf.json"
    doc = json.loads(opf_path.read_text(encoding="utf-8"))
    doc["identity"] = {"content_hash": content_hash_value}
    opf_path.write_text(json.dumps(doc), encoding="utf-8")


def test_apply_feedback_matching_content_hash_applies_normally(tmp_path: Path) -> None:
    """A feedback.json whose _export.content_hash matches the current
    playbook.opf.json applies exactly as it would with no binding at all."""
    _make_opf(tmp_path)
    _set_identity_content_hash(tmp_path, "sha256:abc123")
    out_dir = tmp_path / "out"
    doc_dir = tmp_path / "state-university-2023"
    doc_dir.mkdir()

    feedback = {
        "_export": {"content_hash": "sha256:abc123", "generated_at": "2026-01-01T00:00:00Z"},
        "C2.1": {"provenance": "counterparty_paper"},
    }
    fp = _write_feedback(tmp_path, feedback)
    result = apply_feedback(out_dir, fp)
    assert "state-university-2023" in result.hints_written
    assert "_export" not in result.skipped


def test_apply_feedback_mismatched_content_hash_raises(tmp_path: Path) -> None:
    """A stale feedback.json (content_hash no longer matches) is refused —
    the exact scenario from issue #174: item numbers may now point at
    different clauses after a re-mine/re-project."""
    _make_opf(tmp_path)
    _set_identity_content_hash(tmp_path, "sha256:current")
    out_dir = tmp_path / "out"
    doc_dir = tmp_path / "state-university-2023"
    doc_dir.mkdir()

    feedback = {
        "_export": {"content_hash": "sha256:stale", "generated_at": "2025-01-01T00:00:00Z"},
        "C2.1": {"provenance": "counterparty_paper"},
    }
    fp = _write_feedback(tmp_path, feedback)
    with pytest.raises(ValueError, match="stale"):
        apply_feedback(out_dir, fp)
    # Refused BEFORE any correction was applied — no hints.yaml written.
    assert not (doc_dir / "hints.yaml").exists()


def test_apply_feedback_mismatched_content_hash_force_applies(tmp_path: Path) -> None:
    """--force / force=True overrides the stale-export refusal."""
    _make_opf(tmp_path)
    _set_identity_content_hash(tmp_path, "sha256:current")
    out_dir = tmp_path / "out"
    doc_dir = tmp_path / "state-university-2023"
    doc_dir.mkdir()

    feedback = {
        "_export": {"content_hash": "sha256:stale", "generated_at": "2025-01-01T00:00:00Z"},
        "C2.1": {"provenance": "counterparty_paper"},
    }
    fp = _write_feedback(tmp_path, feedback)
    result = apply_feedback(out_dir, fp, force=True)
    assert "state-university-2023" in result.hints_written


def test_apply_feedback_no_export_key_applies_without_error(tmp_path: Path) -> None:
    """A pre-#174 feedback.json (no "_export" key at all) is unverifiable,
    not stale — it still applies, matching every pre-existing test fixture
    in this file that predates the binding."""
    _make_opf(tmp_path)
    _set_identity_content_hash(tmp_path, "sha256:current")
    out_dir = tmp_path / "out"
    doc_dir = tmp_path / "state-university-2023"
    doc_dir.mkdir()

    feedback = {"C2.1": {"provenance": "counterparty_paper"}}
    fp = _write_feedback(tmp_path, feedback)
    result = apply_feedback(out_dir, fp)
    assert "state-university-2023" in result.hints_written


def test_apply_feedback_no_identity_on_doc_applies_without_error(tmp_path: Path) -> None:
    """A playbook.opf.json with no identity block (e.g. an older/hand-built
    document) makes the binding unverifiable, not stale — still applies."""
    _make_opf(tmp_path)  # no identity block written
    out_dir = tmp_path / "out"
    doc_dir = tmp_path / "state-university-2023"
    doc_dir.mkdir()

    feedback = {
        "_export": {"content_hash": "sha256:whatever", "generated_at": "2026-01-01T00:00:00Z"},
        "C2.1": {"provenance": "counterparty_paper"},
    }
    fp = _write_feedback(tmp_path, feedback)
    result = apply_feedback(out_dir, fp)
    assert "state-university-2023" in result.hints_written


def test_render_html_export_button_embeds_content_hash_binding(tmp_path: Path) -> None:
    """The rendered page's exportFeedback() reads identity.content_hash off
    the embedded playbook-data script and stamps it onto the export as
    "_export" — the client-side half of the issue #174 fix."""
    _make_opf(tmp_path)
    out_dir = tmp_path / "out"
    html = render_review_html(out_dir)
    assert "fb._export" in html
    assert "playbook-data" in html
    assert "content_hash" in html


def test_view_apply_cmd_stale_export_exits_nonzero(tmp_path: Path) -> None:
    """CLI ``view apply`` refuses a stale feedback.json without --force."""
    _make_opf(tmp_path)
    _set_identity_content_hash(tmp_path, "sha256:current")
    doc_dir = tmp_path / "state-university-2023"
    doc_dir.mkdir()

    feedback = {
        "_export": {"content_hash": "sha256:stale"},
        "C2.1": {"provenance": "counterparty_paper"},
    }
    fp = _write_feedback(tmp_path, feedback)

    runner = CliRunner()
    result = runner.invoke(cli, ["view", "apply", str(tmp_path / "out"), str(fp)])
    assert result.exit_code != 0
    assert "stale" in result.output


def test_view_apply_cmd_stale_export_with_force_succeeds(tmp_path: Path) -> None:
    """CLI ``view apply --force`` applies a stale feedback.json anyway."""
    _make_opf(tmp_path)
    _set_identity_content_hash(tmp_path, "sha256:current")
    doc_dir = tmp_path / "state-university-2023"
    doc_dir.mkdir()

    feedback = {
        "_export": {"content_hash": "sha256:stale"},
        "C2.1": {"provenance": "counterparty_paper"},
    }
    fp = _write_feedback(tmp_path, feedback)

    runner = CliRunner()
    result = runner.invoke(cli, ["view", "apply", str(tmp_path / "out"), str(fp), "--force"])
    assert result.exit_code == 0, result.output
    assert "OK" in result.output


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
# apply_feedback — floor candidate accept/reject (issue #90)
# ---------------------------------------------------------------------------


def test_apply_feedback_floor_accept_promotes_invariant(tmp_path: Path) -> None:
    _make_opf(tmp_path)
    out_dir = tmp_path / "out"
    _write_floor_candidates(tmp_path, [_floor_candidate()])

    feedback = {"floor": {"cand-001": {"decision": "accept"}}}
    fp = _write_feedback(tmp_path, feedback)
    result = apply_feedback(out_dir, fp)

    assert result.floor_promoted == ["cand-001"]
    assert result.floor_rejected == []
    assert result.skipped == {}

    doc = json.loads((out_dir / "playbook.opf.json").read_text(encoding="utf-8"))
    invariants = doc["floor"]["invariants"]
    assert len(invariants) == 1
    assert invariants[0]["id"] == "never-accept-uncapped-liability"
    assert invariants[0]["statement"] == "Never accept uncapped liability."
    assert "Accepted via review feedback" in invariants[0]["rationale"]

    candidates = json.loads((out_dir / "floor.candidates.json").read_text(encoding="utf-8"))
    assert candidates["candidates"][0]["decision"] == "accepted"


def test_apply_feedback_floor_reject_records_rejection_never_touches_invariants(
    tmp_path: Path,
) -> None:
    _make_opf(tmp_path)
    out_dir = tmp_path / "out"
    _write_floor_candidates(tmp_path, [_floor_candidate()])

    feedback = {"floor": {"cand-001": {"decision": "reject"}}}
    fp = _write_feedback(tmp_path, feedback)
    result = apply_feedback(out_dir, fp)

    assert result.floor_rejected == ["cand-001"]
    assert result.floor_promoted == []

    doc = json.loads((out_dir / "playbook.opf.json").read_text(encoding="utf-8"))
    assert doc.get("floor", {}).get("invariants", []) == []  # never promoted

    candidates = json.loads((out_dir / "floor.candidates.json").read_text(encoding="utf-8"))
    assert candidates["candidates"][0]["decision"] == "rejected"


def test_apply_feedback_floor_accept_and_reject_together(tmp_path: Path) -> None:
    _make_opf(tmp_path)
    out_dir = tmp_path / "out"
    _write_floor_candidates(
        tmp_path,
        [
            _floor_candidate(id="cand-001", statement="Never accept uncapped liability."),
            _floor_candidate(
                id="cand-002", statement="Never accept broad non-compete.", citations=[]
            ),
        ],
    )

    feedback = {
        "floor": {
            "cand-001": {"decision": "accept"},
            "cand-002": {"decision": "reject", "comment": "too broad as worded"},
        }
    }
    fp = _write_feedback(tmp_path, feedback)
    result = apply_feedback(out_dir, fp)

    assert result.floor_promoted == ["cand-001"]
    assert result.floor_rejected == ["cand-002"]

    doc = json.loads((out_dir / "playbook.opf.json").read_text(encoding="utf-8"))
    assert len(doc["floor"]["invariants"]) == 1
    assert doc["floor"]["invariants"][0]["id"] == "never-accept-uncapped-liability"


def test_apply_feedback_floor_comment_written_to_viewer_notes(tmp_path: Path) -> None:
    _make_opf(tmp_path)
    out_dir = tmp_path / "out"
    _write_floor_candidates(tmp_path, [_floor_candidate()])

    feedback = {"floor": {"cand-001": {"decision": "reject", "comment": "too broad as worded"}}}
    fp = _write_feedback(tmp_path, feedback)
    result = apply_feedback(out_dir, fp)

    assert result.notes_written is True
    notes = (out_dir / "viewer_notes.md").read_text(encoding="utf-8")
    assert "too broad as worded" in notes
    assert "cand-001" in notes


def test_apply_feedback_floor_comment_without_decision_written_to_viewer_notes(
    tmp_path: Path,
) -> None:
    """Issue #90 review finding 3 regression: a floor entry with only a
    comment (no 'decision' key at all -- exactly the shape the page's
    Export JS produces when a reviewer types a note and leaves the radio on
    Undecided: viewer.py's collectFeedback skips 'undecided' radios but
    unconditionally attaches .floor-comment-input text) must still land in
    viewer_notes.md, the same as a comment on a clause item."""
    _make_opf(tmp_path)
    out_dir = tmp_path / "out"
    _write_floor_candidates(tmp_path, [_floor_candidate()])

    feedback = {"floor": {"cand-001": {"comment": "worth revisiting next cycle"}}}
    fp = _write_feedback(tmp_path, feedback)
    result = apply_feedback(out_dir, fp)

    assert result.notes_written is True
    notes = (out_dir / "viewer_notes.md").read_text(encoding="utf-8")
    assert "worth revisiting next cycle" in notes
    assert "cand-001" in notes
    # Nothing was accepted or rejected -- only the comment landed; the
    # missing decision is still (separately) reported as not applied.
    assert result.floor_promoted == []
    assert result.floor_rejected == []
    assert "floor:cand-001" in result.skipped


def test_apply_feedback_floor_accept_q4_already_signed_candidate_refused(
    tmp_path: Path,
) -> None:
    """Issue #90 review finding 1 regression: accepting a source=interview_q4
    candidate whose item was already promoted via
    promote_interview_q4_invariants (issue #89, a DIFFERENT id derivation)
    must be refused -- never append a second, opposite-polarity invariant
    for the same item."""
    from playbook_engine.floor_candidates import promote_interview_q4_invariants

    _make_opf(tmp_path)
    out_dir = tmp_path / "out"
    item = "liability caps"
    invariants = promote_interview_q4_invariants(
        {"sacred_clauses": item}, posture_version=1, existing_invariants=[]
    )
    _set_floor_invariants(tmp_path, invariants)
    candidate = _floor_candidate(
        id="cand-001",
        # Built by the REAL producer, not a hardcoded string: this statement is
        # regex-parsed back into the Q4 item by `candidate_q4_invariant_id`, so
        # a hardcoded copy silently decouples from the producer and this test
        # keeps passing while the already-signed guard is dead in production.
        statement=derive_interview_q4_candidates({"sacred_clauses": item})[0].statement,
        rationale='Named as non-negotiable in the Posture interview (Q4 "sacred_clauses").',
        source="interview_q4",
        citations=[],
    )
    _write_floor_candidates(tmp_path, [candidate])

    # Render: the candidate must show as already-signed, not a live control.
    html = render_review_html(out_dir)
    assert ">already signed<" in html
    assert 'class="floor-decision"' not in html

    # Accepting it anyway (e.g. a stale feedback.json) must be refused, not
    # append a second, contradictory invariant.
    feedback = {"floor": {"cand-001": {"decision": "accept"}}}
    fp = _write_feedback(tmp_path, feedback)
    result = apply_feedback(out_dir, fp)

    assert result.floor_promoted == []
    assert "floor:cand-001" in result.skipped

    doc = json.loads((out_dir / "playbook.opf.json").read_text(encoding="utf-8"))
    assert len(doc["floor"]["invariants"]) == 1  # still just the one Q4-promoted entry


def test_apply_feedback_floor_malformed_block_reported_not_applied(tmp_path: Path) -> None:
    """'floor' feedback that isn't {candidate_id: {...}} is reported via
    ApplyResult.skipped — never a false 'OK' (issue #138)."""
    _make_opf(tmp_path)
    out_dir = tmp_path / "out"
    _write_floor_candidates(tmp_path, [_floor_candidate()])

    feedback = {"floor": "accept everything"}
    fp = _write_feedback(tmp_path, feedback)
    result = apply_feedback(out_dir, fp)

    assert "floor" in result.skipped
    assert result.floor_promoted == []
    assert result.floor_rejected == []
    doc = json.loads((out_dir / "playbook.opf.json").read_text(encoding="utf-8"))
    assert doc.get("floor", {}).get("invariants", []) == []


def test_apply_feedback_floor_unknown_candidate_reported_not_applied(tmp_path: Path) -> None:
    _make_opf(tmp_path)
    out_dir = tmp_path / "out"
    _write_floor_candidates(tmp_path, [_floor_candidate()])

    feedback = {"floor": {"cand-999": {"decision": "accept"}}}
    fp = _write_feedback(tmp_path, feedback)
    result = apply_feedback(out_dir, fp)

    assert "floor:cand-999" in result.skipped
    assert result.floor_promoted == []


def test_apply_feedback_floor_unknown_decision_reported_not_applied(tmp_path: Path) -> None:
    _make_opf(tmp_path)
    out_dir = tmp_path / "out"
    _write_floor_candidates(tmp_path, [_floor_candidate()])

    feedback = {"floor": {"cand-001": {"decision": "maybe"}}}
    fp = _write_feedback(tmp_path, feedback)
    result = apply_feedback(out_dir, fp)

    assert "floor:cand-001" in result.skipped
    assert result.floor_promoted == []
    assert result.floor_rejected == []


def test_apply_feedback_floor_no_candidates_file_reports_not_applied(tmp_path: Path) -> None:
    """No floor.candidates.json on disk at all -> every named candidate id
    is 'unknown', reported honestly rather than silently ignored."""
    _make_opf(tmp_path)
    out_dir = tmp_path / "out"

    feedback = {"floor": {"cand-001": {"decision": "accept"}}}
    fp = _write_feedback(tmp_path, feedback)
    result = apply_feedback(out_dir, fp)

    assert "floor:cand-001" in result.skipped
    assert result.floor_promoted == []


def test_apply_feedback_floor_and_clause_item_together(tmp_path: Path) -> None:
    """The reserved top-level "floor" key coexists with ordinary Cx items in
    the same feedback.json — neither interferes with the other."""
    _make_opf(tmp_path)
    out_dir = tmp_path / "out"
    _write_floor_candidates(tmp_path, [_floor_candidate()])

    feedback = {
        "floor": {"cand-001": {"decision": "accept"}},
        "C1": {"comment": "double check this clause"},
    }
    fp = _write_feedback(tmp_path, feedback)
    result = apply_feedback(out_dir, fp)

    assert result.floor_promoted == ["cand-001"]
    assert result.notes_written is True
    notes = (out_dir / "viewer_notes.md").read_text(encoding="utf-8")
    assert "double check this clause" in notes
    doc = json.loads((out_dir / "playbook.opf.json").read_text(encoding="utf-8"))
    assert len(doc["floor"]["invariants"]) == 1


def test_apply_feedback_floor_accept_changes_content_hash_when_present(tmp_path: Path) -> None:
    """Unlike a curation pin, an accepted Floor candidate DOES change
    content_hash — floor.invariants is NOT excluded from it, unlike
    curation (canonicalize.py)."""
    doc = _make_opf(tmp_path)
    out_dir = tmp_path / "out"
    _write_floor_candidates(tmp_path, [_floor_candidate()])

    from playbook_engine.canonicalize import compute_section_digests, content_hash  # noqa: PLC0415

    doc["identity"] = {
        "content_hash": content_hash(doc),
        "section_digests": compute_section_digests(doc),
    }
    (out_dir / "playbook.opf.json").write_text(json.dumps(doc), encoding="utf-8")
    hash_before = doc["identity"]["content_hash"]
    floor_digest_before = doc["identity"]["section_digests"]["floor"]

    feedback = {"floor": {"cand-001": {"decision": "accept"}}}
    fp = _write_feedback(tmp_path, feedback)
    apply_feedback(out_dir, fp)

    after = json.loads((out_dir / "playbook.opf.json").read_text(encoding="utf-8"))
    assert after["identity"]["content_hash"] != hash_before
    assert after["identity"]["section_digests"]["floor"] != floor_digest_before


def test_apply_feedback_floor_reject_does_not_write_playbook_opf_json(tmp_path: Path) -> None:
    """A reject-only run never touches playbook.opf.json at all — only
    floor.candidates.json changes."""
    _make_opf(tmp_path)
    out_dir = tmp_path / "out"
    _write_floor_candidates(tmp_path, [_floor_candidate()])
    opf_path = out_dir / "playbook.opf.json"
    bytes_before = opf_path.read_bytes()

    feedback = {"floor": {"cand-001": {"decision": "reject"}}}
    fp = _write_feedback(tmp_path, feedback)
    apply_feedback(out_dir, fp)

    assert opf_path.read_bytes() == bytes_before


def test_apply_feedback_floor_idempotent_second_apply_unchanged(tmp_path: Path) -> None:
    """Round-trip requirement: re-applying identical feedback is a no-op —
    both playbook.opf.json and floor.candidates.json are byte-identical
    across the second apply."""
    _make_opf(tmp_path)
    out_dir = tmp_path / "out"
    _write_floor_candidates(tmp_path, [_floor_candidate()])

    feedback = {"floor": {"cand-001": {"decision": "accept"}}}
    fp = _write_feedback(tmp_path, feedback)

    first = apply_feedback(out_dir, fp)
    opf_bytes_after_first = (out_dir / "playbook.opf.json").read_bytes()
    candidates_bytes_after_first = (out_dir / "floor.candidates.json").read_bytes()

    second = apply_feedback(out_dir, fp)

    assert second.floor_promoted == first.floor_promoted == ["cand-001"]
    assert (out_dir / "playbook.opf.json").read_bytes() == opf_bytes_after_first
    assert (out_dir / "floor.candidates.json").read_bytes() == candidates_bytes_after_first

    doc = json.loads((out_dir / "playbook.opf.json").read_text(encoding="utf-8"))
    assert len(doc["floor"]["invariants"]) == 1  # never duplicated


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
    assert "not found" in result.output
    assert "could not parse" not in result.output


def test_view_render_cmd_truncated_opf_reports_error_no_traceback(tmp_path: Path) -> None:
    """A truncated/hand-edited playbook.opf.json fails cleanly (issue #57), not a traceback."""
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "playbook.opf.json").write_text('{"truncated":', encoding="utf-8")
    runner = CliRunner()
    result = runner.invoke(cli, ["view", "render", str(out_dir)])
    assert result.exit_code == 1
    assert "ERROR" in result.output
    # A raw JSONDecodeError propagating uncaught would surface as some other
    # exception type here; the handled path always exits via SystemExit(1).
    assert isinstance(result.exception, SystemExit)


def test_view_render_cmd_tolerates_non_utf8_floor_candidates_sidecar(tmp_path: Path) -> None:
    """Issue #90 review finding 6 regression: a non-UTF-8
    floor.candidates.json must not abort the render (nor get misattributed
    to playbook.opf.json, which is perfectly readable) -- it degrades to
    'no candidates', same as malformed JSON, exit 0."""
    _make_opf(tmp_path)
    out_dir = tmp_path / "out"
    (out_dir / "floor.candidates.json").write_bytes(b"\xff\xfe\x00\x01garbage")

    runner = CliRunner()
    result = runner.invoke(cli, ["view", "render", str(out_dir)])

    assert result.exit_code == 0, result.output
    assert "OK" in result.output
    assert 'id="floor-candidates"' not in (out_dir / "playbook.review.html").read_text(
        encoding="utf-8"
    )


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


def test_view_apply_cmd_floor_accept_reports_promoted(tmp_path: Path) -> None:
    """Issue #90: ApplyResult reporting includes counts of promoted/rejected
    Floor candidates."""
    _make_opf(tmp_path)
    _write_floor_candidates(tmp_path, [_floor_candidate()])
    feedback = {"floor": {"cand-001": {"decision": "accept"}}}
    fp = _write_feedback(tmp_path, feedback)

    runner = CliRunner()
    result = runner.invoke(cli, ["view", "apply", str(tmp_path / "out"), str(fp)])
    assert result.exit_code == 0, result.output
    assert "OK  feedback applied" in result.output
    assert "1 floor candidate(s) promoted" in result.output
    assert "cand-001: promoted" in result.output


def test_view_apply_cmd_floor_reject_reports_rejected(tmp_path: Path) -> None:
    _make_opf(tmp_path)
    _write_floor_candidates(tmp_path, [_floor_candidate()])
    feedback = {"floor": {"cand-001": {"decision": "reject"}}}
    fp = _write_feedback(tmp_path, feedback)

    runner = CliRunner()
    result = runner.invoke(cli, ["view", "apply", str(tmp_path / "out"), str(fp)])
    assert result.exit_code == 0, result.output
    assert "OK  feedback applied" in result.output
    assert "1 floor candidate(s) rejected" in result.output
    assert "cand-001: rejected" in result.output


def test_view_apply_cmd_floor_malformed_reports_not_applied(tmp_path: Path) -> None:
    _make_opf(tmp_path)
    _write_floor_candidates(tmp_path, [_floor_candidate()])
    feedback = {"floor": {"cand-999": {"decision": "accept"}}}
    fp = _write_feedback(tmp_path, feedback)

    runner = CliRunner()
    result = runner.invoke(cli, ["view", "apply", str(tmp_path / "out"), str(fp)])
    assert result.exit_code == 0, result.output
    assert "floor:cand-999: not applied" in result.output
    assert "NOTE  no feedback applied" in result.output


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


# ---------------------------------------------------------------------------
# User guide overlay + terminology hover help
# ---------------------------------------------------------------------------


def test_review_html_has_user_guide(tmp_path: Path) -> None:
    _make_opf(tmp_path)
    runner = CliRunner()
    result = runner.invoke(cli, ["view", "render", str(tmp_path / "out")])
    assert result.exit_code == 0, result.output
    html = (tmp_path / "out" / "playbook.review.html").read_text(encoding="utf-8")
    assert "User guide" in html
    assert "How to use this page" in html
    assert "toggleGuide" in html
    # variation vocabulary explained
    assert "preferred variations" in html
    assert "unacceptable variations" in html


def test_review_html_badges_carry_help_titles(tmp_path: Path) -> None:
    _make_opf(tmp_path)
    runner = CliRunner()
    result = runner.invoke(cli, ["view", "render", str(tmp_path / "out")])
    assert result.exit_code == 0, result.output
    html = (tmp_path / "out" / "playbook.review.html").read_text(encoding="utf-8")
    assert 'title="This form survived to the signed copy."' in html


def _add_clause_summary(tmp_path: Path) -> None:
    """Populate summary.* on the first clause so the Preferred / concessions /
    Unacceptable sections actually render (the base fixture has no summary)."""
    opf_path = tmp_path / "out" / "playbook.opf.json"
    doc = json.loads(opf_path.read_text(encoding="utf-8"))
    clauses = (doc.get("evidence") or {}).get("clauses") or doc["clauses"]
    clauses[0]["summary"] = {
        "historical_stance": "negotiable",
        "acceptable_if": [
            {
                "if": "Each party shall indemnify the other.",
                "to": "Each party shall indemnify the other for negligence.",
                "rationale": "signed at neutral risk",
                "observation_ref": {
                    "document_id": "state-university-2023",
                    "version": 3,
                    "clause_path": "8",
                },
            }
        ],
        "fallbacks": [
            {
                "text_summary": "Cap indemnity at fees paid.",
                "risk_delta": {"direction": "worse", "magnitude": "minor"},
            }
        ],
        "rejected": [
            {
                "text_summary": "Uncapped one-way indemnity in our disfavour.",
                "risk_delta": {"direction": "worse", "magnitude": "material"},
            }
        ],
    }
    opf_path.write_text(json.dumps(doc), encoding="utf-8")


def test_bundle_is_superset_of_document(tmp_path: Path) -> None:
    """The bundle is THE distribution artifact: it must carry every major
    section of the readable document. Guards the render_document_html ->
    render_bundle_html seam — a future edit to the document template must not
    be able to silently drop content out of the bundle."""
    _make_opf(tmp_path)
    _add_clause_summary(tmp_path)
    runner = CliRunner()
    result = runner.invoke(cli, ["view", "bundle", str(tmp_path / "out")])
    assert result.exit_code == 0, result.output
    html = (tmp_path / "out" / "playbook.opf.html").read_text(encoding="utf-8")

    # cover + stats block
    assert "Negotiation Playbook" in html
    assert '<div class="stats">' in html
    assert "agreements in scope" in html
    assert "clause concepts" in html

    # table of contents
    assert '<nav class="toc">' in html
    assert "<strong>Clauses</strong>" in html

    # per-clause sections, with stance chips and confidence hover help
    assert '<section class="clause" id="clause-1">' in html
    assert 'href="#clause-1"' in html
    assert "Indemnification" in html
    assert "Governing Law" in html
    assert "cursor:help" in html  # stance chip
    assert 'title="confidence = evidence-depth score' in html

    # variation terminology — new labels present, old ones gone
    assert "Preferred variations" in html
    assert "Acceptable variations — concessions" in html
    assert "Unacceptable variations — rejected/reversed asks" in html
    assert "Fallback positions" not in html
    assert "Rejected / reversed asks" not in html

    # posture / floor pending notices
    assert "Posture:</strong> pending" in html
    assert "Floor:</strong> pending" in html
    # the notices must say how to enable each section and that consumers work without them
    assert "playbook posture interview" in html
    assert "playbook floor propose" in html
    assert html.count("works fine") >= 2

    # Method & provenance panel
    assert "Method &amp; provenance" in html
    assert "compiled from evidence, not authored" in html
    assert "content hash" in html

    # ...and the bundle's own additions, after the document body
    assert 'id="digest"' in html
    assert '<script id="opf-canonical" type="application/json">' in html

    # No annotation machinery: the bundle is readable+machine, not the review UI
    assert "Export feedback" not in html


def test_bundle_embeds_document_body_verbatim(tmp_path: Path) -> None:
    """Structural form of the superset guarantee: the bundle is the document
    plus insertions — every document line survives, in order, byte-for-byte.

    Deliberately asserts no template text of its own (no "<footer>" marker):
    the point of the render_document_html -> render_bundle_html seam is that
    neither the bundle nor this test pattern-matches on the document template,
    so an edit to that template cannot silently break either.
    """
    import difflib

    from playbook_engine.document_renderer import render_bundle_html, render_document_html

    _make_opf(tmp_path)
    _add_clause_summary(tmp_path)
    out_dir = tmp_path / "out"
    document = render_document_html(out_dir)
    bundle = render_bundle_html(out_dir)

    matcher = difflib.SequenceMatcher(None, document.splitlines(), bundle.splitlines())
    ops = {tag for tag, *_ in matcher.get_opcodes()}
    # only 'equal' and 'insert' — a 'delete'/'replace' means the bundle dropped
    # or mangled part of the document body
    assert ops <= {"equal", "insert"}, f"bundle does not preserve the document: {ops}"
    assert "insert" in ops  # the bundle does add its digest + machine blocks
    assert len(bundle) > len(document)


def test_bundle_method_panel_counts_list_shaped_version_ingest(tmp_path: Path) -> None:
    """Real compiled docs carry version_ingest as a LIST of records."""
    _make_opf(tmp_path)
    opf_path = tmp_path / "out" / "playbook.opf.json"
    doc = json.loads(opf_path.read_text(encoding="utf-8"))
    doc["corpus"]["documents"][0]["version_ingest"] = [
        {"version": "01__a", "status": "ok", "error": None, "extractor": "docling"},
        {"version": "02__a", "status": "failed", "error": "no text", "extractor": "docling"},
    ]
    opf_path.write_text(json.dumps(doc), encoding="utf-8")
    runner = CliRunner()
    result = runner.invoke(cli, ["view", "bundle", str(tmp_path / "out")])
    assert result.exit_code == 0, result.output
    html = (tmp_path / "out" / "playbook.opf.html").read_text(encoding="utf-8")
    assert "1 version file(s) failed extraction" in html


# ---------------------------------------------------------------------------
# Round trip — issue #90 acceptance criteria
# ---------------------------------------------------------------------------


def test_floor_checklist_round_trip_accept_reject_idempotent(tmp_path: Path) -> None:
    """The full issue #90 acceptance-criteria round trip:

    render (both candidates show live controls) -> feedback.json accepting
    one and rejecting the other -> view apply (CLI) -> accepted is in
    floor.invariants and validate passes; rejected never reappears as
    proposed on re-render; re-applying the same feedback is a no-op.

    Uses a hand-built, schema-conformant OPF v0.2 doc rather than this
    file's v0.1-shaped ``_make_opf()`` fixture: OPF v0.1 predates the Floor
    section entirely (OPF-SPEC.md §3.7 is "NEW" in v0.2), so a v0.1 doc
    carrying a top-level ``"floor"`` key is schema-invalid by construction
    — every OTHER test in this file exercises ``apply_feedback``/
    ``render_review_html`` directly without ever running ``playbook
    validate``, but this test's acceptance criterion explicitly requires
    validation to pass.
    """
    doc = {
        "opf_version": "0.2",
        "agreement_type": {"id": "test-agreement", "name": "Test Agreement"},
        "baseline": {"has_canonical_template": False},
        "taxonomy": {"source": "custom", "entries": []},
        "evidence": {"clauses": [], "clause_library": []},
        "posture": {},
        "floor": {},
        "corpus": {"documents": [], "stats": {}},
        "compiler": {
            "name": "playbook-engine",
            "version": "0.1.0",
            "run_id": "run-abc",
            "generated_at": "2026-01-01T00:00:00Z",
        },
        "identity": {
            "content_hash": "sha256:" + "0" * 64,
            "section_digests": {
                "evidence": "sha256:" + "1" * 64,
                "posture": "sha256:" + "2" * 64,
                "floor": "sha256:" + "3" * 64,
                "curation": "sha256:" + "4" * 64,
            },
        },
    }
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True)
    (out_dir / "playbook.opf.json").write_text(json.dumps(doc), encoding="utf-8")

    keep = _floor_candidate(id="cand-001", statement="Never accept uncapped liability.")
    drop = _floor_candidate(
        id="cand-002", statement="Never accept broad non-compete.", citations=[]
    )
    _write_floor_candidates(tmp_path, [keep, drop])

    # 1. Render — both candidates are live, undecided controls.
    first_render = render_review_html(out_dir)
    assert "Proposed hard lines" in first_render
    assert 'data-candidate-id="cand-001"' in first_render
    assert 'data-candidate-id="cand-002"' in first_render
    assert first_render.count('value="undecided" checked') == 2

    # 2. Hand-write feedback.json: accept cand-001, reject cand-002.
    feedback = {
        "floor": {
            "cand-001": {"decision": "accept", "comment": "categorical, agreed"},
            "cand-002": {"decision": "reject", "comment": "too broad as worded"},
        }
    }
    fp = _write_feedback(tmp_path, feedback)

    # 3. view apply (CLI).
    runner = CliRunner()
    apply_result = runner.invoke(cli, ["view", "apply", str(out_dir), str(fp)])
    assert apply_result.exit_code == 0, apply_result.output
    assert "1 floor candidate(s) promoted" in apply_result.output
    assert "1 floor candidate(s) rejected" in apply_result.output

    applied_doc = json.loads((out_dir / "playbook.opf.json").read_text(encoding="utf-8"))
    invariants = applied_doc["floor"]["invariants"]
    assert len(invariants) == 1
    assert invariants[0]["id"] == "never-accept-uncapped-liability"

    # 4. playbook validate passes.
    validate_result = runner.invoke(cli, ["validate", str(out_dir / "playbook.opf.json")])
    assert validate_result.exit_code == 0, validate_result.output

    # 5. Re-render: accepted -> inert "already signed"; rejected -> inert
    #    "rejected", never re-proposed as an open item; no controls for
    #    either.
    second_render = render_review_html(out_dir)
    assert ">already signed<" in second_render
    assert ">rejected<" in second_render
    assert 'class="floor-decision"' not in second_render

    # 6. Re-applying the SAME feedback is a no-op: both files byte-identical.
    opf_bytes = (out_dir / "playbook.opf.json").read_bytes()
    candidates_bytes = (out_dir / "floor.candidates.json").read_bytes()
    second_apply = runner.invoke(cli, ["view", "apply", str(out_dir), str(fp)])
    assert second_apply.exit_code == 0, second_apply.output
    assert (out_dir / "playbook.opf.json").read_bytes() == opf_bytes
    assert (out_dir / "floor.candidates.json").read_bytes() == candidates_bytes


# ---------------------------------------------------------------------------
# opf_accessors.clause_is_thin — shared "thin evidence" trigger (issue #91)
# ---------------------------------------------------------------------------


def test_clause_is_thin_evidence_insufficient() -> None:
    from playbook_engine.opf_accessors import clause_is_thin

    clause = {
        "observed_positions": [{"precedent_count": 4}],
        "rollup": {"confidence": {"evidence_sufficient": False}},
    }
    assert clause_is_thin(clause) is True


def test_clause_is_thin_single_precedent_only() -> None:
    from playbook_engine.opf_accessors import clause_is_thin

    clause = {
        "observed_positions": [{"precedent_count": 1}, {"precedent_count": 1}],
        "rollup": {"confidence": {"evidence_sufficient": True}},
    }
    assert clause_is_thin(clause) is True


def test_clause_is_thin_false_when_recurring_and_sufficient() -> None:
    """A clause with at least one recurring (precedent_count > 1) position
    and evidence_sufficient not False is not thin, even though one of its
    OTHER positions was seen only once."""
    from playbook_engine.opf_accessors import clause_is_thin

    clause = {
        "observed_positions": [{"precedent_count": 4}, {"precedent_count": 1}],
        "rollup": {"confidence": {"evidence_sufficient": True}},
    }
    assert clause_is_thin(clause) is False


def test_clause_is_thin_false_when_no_positions_and_no_explicit_flag() -> None:
    """Vacuous case: zero observed positions and no explicit
    evidence_sufficient: false — not thin by this rule alone (a
    compiler-produced playbook already sets evidence_sufficient False
    whenever n_our_paper falls short; see clause_position_compiler.py)."""
    from playbook_engine.opf_accessors import clause_is_thin

    clause = {"observed_positions": [], "rollup": {"confidence": {"score": 0.9}}}
    assert clause_is_thin(clause) is False


def test_clause_is_thin_v02_summary_confidence_shape() -> None:
    """clause_is_thin is version-agnostic like every other opf_accessors
    helper — OPF v0.2's summary.confidence works the same as v0.1's
    rollup.confidence."""
    from playbook_engine.opf_accessors import clause_is_thin

    clause = {
        "observed_positions": [{"precedent_count": 1}],
        "summary": {"confidence": {"evidence_sufficient": True}},
    }
    assert clause_is_thin(clause) is True


# ---------------------------------------------------------------------------
# Control-ladder restructure — triage header, decisions, audit (issue #91)
# ---------------------------------------------------------------------------


def _set_posture_version(tmp_path: Path, version: int) -> None:
    """Mutate out/playbook.opf.json (already written by _make_opf) to carry
    a posture.version — for triage-header fixtures."""
    opf_path = tmp_path / "out" / "playbook.opf.json"
    doc = json.loads(opf_path.read_text(encoding="utf-8"))
    doc["posture"] = {"system_prompt": "Test posture.", "version": version}
    opf_path.write_text(json.dumps(doc), encoding="utf-8")


def test_render_html_triage_header_present_with_zero_state(tmp_path: Path) -> None:
    """Triage header renders even with no Floor/Posture state at all — real,
    honest zero counts, not a hidden/absent section (default _make_opf
    fixture: 2 clauses, neither thin, no floor.candidates.json, no posture)."""
    _make_opf(tmp_path)
    html = render_review_html(tmp_path / "out")
    assert 'id="triage-header"' in html
    assert "Hard lines: 0 signed" in html
    assert "0 proposed awaiting sign-off" in html
    assert "Posture: not authored" in html
    assert "Evidence: 2 clauses, 0 thin" in html
    # The "no empty shell" rule for the checklist itself (issue #90) is
    # unaffected: still no section when there are no candidates.
    assert 'id="floor-candidates"' not in html


def test_render_html_triage_header_counts_match_floor_and_posture_state(
    tmp_path: Path,
) -> None:
    """Header counts computed from playbook.opf.json + floor.candidates.json:
    1 candidate already signed (excluded from "proposed"), 2 still pending,
    Posture v1 — verified by hand against the fixture JSON, matching the
    ticket's acceptance criteria."""
    _make_opf(tmp_path)
    _set_posture_version(tmp_path, 1)
    signed = _floor_candidate(id="cand-001", statement="Never accept uncapped liability.")
    pending_a = _floor_candidate(
        id="cand-002", statement="Never accept broad non-compete.", citations=[]
    )
    pending_b = _floor_candidate(
        id="cand-003", statement="Never accept unlimited indemnity.", citations=[]
    )
    _write_floor_candidates(tmp_path, [signed, pending_a, pending_b])
    _set_floor_invariants(
        tmp_path,
        [
            {
                "id": "never-accept-uncapped-liability",
                "statement": "Never accept uncapped liability.",
                "rationale": "Accepted via review feedback (floor candidate cand-001).",
            }
        ],
    )

    html = render_review_html(tmp_path / "out")

    assert "Hard lines: 1 signed" in html
    assert "2 proposed awaiting sign-off" in html
    assert "Posture: v1" in html


def test_render_html_triage_header_shows_three_ways_to_act(tmp_path: Path) -> None:
    _make_opf(tmp_path)
    html = render_review_html(tmp_path / "out")
    assert "~10 min" in html
    assert "playbook posture interview" in html
    assert "~minutes" in html
    assert "sign proposed hard lines" in html
    assert "optional, open-ended" in html
    assert "audit the evidence record" in html


def test_render_html_regions_render_header_then_decisions_then_audit(tmp_path: Path) -> None:
    """The three page regions appear in order: triage header, then the
    Proposed hard lines checklist, then the per-clause audit section."""
    _make_opf(tmp_path)
    _write_floor_candidates(tmp_path, [_floor_candidate()])
    html = render_review_html(tmp_path / "out")

    idx_header = html.index('id="triage-header"')
    idx_decisions = html.index('id="floor-candidates"')
    idx_audit = html.index('class="audit-section-header"')
    idx_first_clause = html.index('<details class="clause"')

    assert idx_header < idx_decisions < idx_audit < idx_first_clause


def test_render_html_clauses_collapsed_by_default(tmp_path: Path) -> None:
    """Every clause renders as a <details> with no `open` attribute — the
    page must render fully collapsed, native-HTML, no JS required."""
    import re

    _make_opf(tmp_path)
    html = render_review_html(tmp_path / "out")
    assert '<details class="clause" id="C1"' in html
    assert '<details class="clause" id="C2"' in html
    assert not re.search(r"<details\b[^>]*\bopen\b", html)


_ATTENTION_SORT_CLAUSES = [
    {
        "id": "clause.alpha",
        "taxonomy_id": "aaa_alpha",
        "title": "Alpha Clean Clause",
        "our_standard": None,
        "observed_positions": [
            {
                "text_summary": "Alpha form, recurring.",
                "example_ref": {
                    "document_id": "state-university-2023",
                    "version": 1,
                    "clause_path": "1",
                },
                "deviation": "none",
                "risk_delta": {"direction": "neutral", "magnitude": "none"},
                "provenance": "our_paper",
                "outcome": "signed",
                "precedent_count": 5,
            },
        ],
        "rollup": {
            "position": "standard",
            "confidence": {
                "score": 0.9,
                "basis": "precedent_count",
                "n_our_paper": 5,
                "n_counterparty_paper": 0,
                "evidence_sufficient": True,
            },
        },
    },
    {
        "id": "clause.omega",
        "taxonomy_id": "zzz_omega",
        "title": "Omega Thin Clause",
        "our_standard": None,
        "observed_positions": [
            {
                "text_summary": "Omega form, seen once.",
                "example_ref": {
                    "document_id": "state-university-2023",
                    "version": 1,
                    "clause_path": "2",
                },
                "deviation": "none",
                "risk_delta": {"direction": "neutral", "magnitude": "none"},
                "provenance": "our_paper",
                "outcome": "signed",
                "precedent_count": 1,
            },
        ],
        "rollup": {
            "position": "standard",
            "confidence": {
                "score": 0.9,
                "basis": "precedent_count",
                "n_our_paper": 1,
                "n_counterparty_paper": 0,
                "evidence_sufficient": True,
            },
        },
    },
]


def test_render_html_attention_first_sort_overrides_taxonomy_order(tmp_path: Path) -> None:
    """A thin clause (Omega, taxonomy-sorted LAST -> numbered C2) must
    render BEFORE a clean clause (Alpha, taxonomy-sorted FIRST -> numbered
    C1) in the collapsed audit section — proving the attention-first sort
    actually reorders rendering, not just coincides with taxonomy order.
    Item numbering itself is untouched: Alpha is still C1, Omega still C2.
    """
    _make_opf(tmp_path, clauses=_ATTENTION_SORT_CLAUSES)
    html = render_review_html(tmp_path / "out")

    idx_c1_tag = html.index('<details class="clause" id="C1"')
    idx_c2_tag = html.index('<details class="clause" id="C2"')
    assert idx_c2_tag < idx_c1_tag, "thin clause C2 must render before clean clause C1"

    # Confirm titles inside each <details> block, scoped to that block —
    # NOT a bare html.index() on the title text, which would find the TOC's
    # sidebar link first: the TOC intentionally still lists clauses in
    # taxonomy order (issue #91 only reorders the audit section itself).
    end_c2 = html.index("</details>", idx_c2_tag)
    assert "Omega Thin Clause" in html[idx_c2_tag:end_c2]
    end_c1 = html.index("</details>", idx_c1_tag)
    assert "Alpha Clean Clause" in html[idx_c1_tag:end_c1]


def test_render_html_attention_reason_visible_on_collapsed_summary_line(tmp_path: Path) -> None:
    """The reason a clause wants attention is visible on its <summary> line
    without expanding; a clean clause's summary says "no flags"."""
    _make_opf(tmp_path, clauses=_ATTENTION_SORT_CLAUSES)
    html = render_review_html(tmp_path / "out")

    start_c2 = html.index('<details class="clause" id="C2"')
    end_c2 = html.index("</summary>", start_c2)
    assert "thin evidence" in html[start_c2:end_c2]

    start_c1 = html.index('<details class="clause" id="C1"')
    end_c1 = html.index("</summary>", start_c1)
    assert "no flags" in html[start_c1:end_c1]


def test_render_html_low_confidence_flags_attention(tmp_path: Path) -> None:
    """confidence.score < 0.6 is its own independent attention trigger,
    distinct from thin evidence — the default fixture's governing_law
    clause (score 0.55, not thin: 1 observation with precedent_count=2)
    demonstrates it in isolation."""
    _make_opf(tmp_path)
    html = render_review_html(tmp_path / "out")
    start = html.index('<details class="clause" id="C1"')  # governing_law, score 0.55
    end = html.index("</summary>", start)
    summary = html[start:end]
    assert "low confidence (55%)" in summary
    assert "thin evidence" not in summary


_PIN_CONFLICT_CLAUSES = [
    {
        "id": "clause.alpha",
        "taxonomy_id": "aaa_alpha",
        "title": "Alpha Clean Clause",
        "our_standard": None,
        "observed_positions": [
            {
                "text_summary": "Alpha form, recurring.",
                "example_ref": {
                    "document_id": "state-university-2023",
                    "version": 1,
                    "clause_path": "1",
                },
                "deviation": "none",
                "risk_delta": {"direction": "neutral", "magnitude": "none"},
                "provenance": "our_paper",
                "outcome": "signed",
                "precedent_count": 5,
            },
        ],
        "rollup": {
            "position": "standard",
            "confidence": {
                "score": 0.9,
                "basis": "precedent_count",
                "n_our_paper": 5,
                "n_counterparty_paper": 0,
                "evidence_sufficient": True,
            },
        },
    },
    {
        "id": "clause.omega",
        "taxonomy_id": "zzz_omega",
        "title": "Omega Pinned Clause",
        "our_standard": None,
        "observed_positions": [
            {
                "text_summary": "Omega form, recurring too.",
                "example_ref": {
                    "document_id": "state-university-2023",
                    "version": 1,
                    "clause_path": "2",
                },
                "deviation": "none",
                "risk_delta": {"direction": "neutral", "magnitude": "none"},
                "provenance": "our_paper",
                "outcome": "signed",
                "precedent_count": 4,
            },
        ],
        "rollup": {
            "position": "standard",
            "confidence": {
                "score": 0.95,
                "basis": "precedent_count",
                "n_our_paper": 4,
                "n_counterparty_paper": 0,
                "evidence_sufficient": True,
            },
        },
    },
]


def test_render_html_pin_conflict_flags_attention_and_sorts_first(tmp_path: Path) -> None:
    """A clause with NO thin/low-confidence trigger, but a curation pin
    whose conflict is set, still sorts to the front and says why — neither
    Alpha nor Omega is thin or low-confidence; only Omega's pin conflicts."""
    _make_opf(tmp_path, clauses=_PIN_CONFLICT_CLAUSES)
    opf_path = tmp_path / "out" / "playbook.opf.json"
    doc = json.loads(opf_path.read_text(encoding="utf-8"))
    doc["curation"] = {
        "pins": [
            {
                "clause_id": "clause.omega",
                "item_id": "C2",
                "position": "usually_conceded",
                "baseline_stance": "standard",
                "pinned_at": "2026-01-01T00:00:00+00:00",
                "conflict": {
                    "flagged_at": "2026-01-02T00:00:00+00:00",
                    "recomputed_historical_stance": "usually_held",
                    "reason": "historical_stance changed since this position was pinned",
                },
            }
        ]
    }
    opf_path.write_text(json.dumps(doc), encoding="utf-8")

    html = render_review_html(tmp_path / "out")

    idx_c1 = html.index('<details class="clause" id="C1"')
    idx_c2 = html.index('<details class="clause" id="C2"')
    assert idx_c2 < idx_c1

    summary_end = html.index("</summary>", idx_c2)
    assert "pinned position conflicts with evidence" in html[idx_c2:summary_end]


def test_control_ladder_restructure_preserves_clause_feedback_round_trip(tmp_path: Path) -> None:
    """A comment + pin correction on a (now collapsed) clause still applies
    correctly after the #91 restructure: the DOM wrapper changed
    (div -> details) but the data-item/data-clause-id hooks the Export JS
    and apply_feedback both depend on did not."""
    _make_opf(tmp_path)
    out_dir = tmp_path / "out"
    html = render_review_html(out_dir)

    assert 'data-item="C1"' in html
    assert 'data-clause-id="clause.governing_law"' in html

    feedback = {"C1": {"comment": "flag this", "override": "usually_conceded"}}
    fp = _write_feedback(tmp_path, feedback)
    result = apply_feedback(out_dir, fp)

    assert result.notes_written is True
    assert result.pins_written == ["C1"]
    doc = json.loads((out_dir / "playbook.opf.json").read_text(encoding="utf-8"))
    assert doc["curation"]["pins"][0]["clause_id"] == "clause.governing_law"


def test_review_html_guide_contains_ladder_framing(tmp_path: Path) -> None:
    """The rewritten guide explains the ladder (light-touch vs. granular,
    all optional) and states plainly that intent belongs in natural
    language, not on this page (issue #91 acceptance criteria)."""
    _make_opf(tmp_path)
    runner = CliRunner()
    result = runner.invoke(cli, ["view", "render", str(tmp_path / "out")])
    assert result.exit_code == 0, result.output
    html = (tmp_path / "out" / "playbook.review.html").read_text(encoding="utf-8")
    normalized = " ".join(html.split())

    assert "light-touch" in normalized
    assert "granular" in normalized
    assert "optional" in normalized
    assert "natural language" in normalized
    assert "signing proposals and correcting the record" in normalized
    # Pre-existing, still-required guide content (unchanged assertions).
    assert "preferred variations" in html
    assert "unacceptable variations" in html


def test_guide_html_is_under_half_its_pre_91_length() -> None:
    """Acceptance criteria: "Keep it under half its current length." The
    pre-#91 _GUIDE_HTML constant was 3643 characters."""
    from playbook_engine.viewer import _GUIDE_HTML

    assert len(_GUIDE_HTML) < 3643 / 2
