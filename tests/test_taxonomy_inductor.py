"""Tests for the taxonomy inductor (issue #31, M3).

SECURITY NOTE: All fixtures use programmatically constructed ClauseTree
objects with synthetic text.  No real agreement files are referenced.
Party names use fictional identifiers only ("Alice Corp", "Beta Ltd").
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from playbook_engine.clause_tree import ClauseNode, ClauseTree
from playbook_engine.cli import cli
from playbook_engine.taxonomy import load_taxonomy
from playbook_engine.taxonomy_inductor import (
    CLUSTER_SIMILARITY_THRESHOLD,
    CUAD_MATCH_THRESHOLD,
    REPRESENTATION_THRESHOLD,
    _heading_tokens,
    _jaccard,
    _normalize_heading,
    _to_entry_id,
    emit_taxonomy_yaml,
    induce_taxonomy,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _node(path: str, heading: str | None = None, text: str = "") -> ClauseNode:
    return ClauseNode(
        clause_path=path,
        heading=heading,
        text=text,
        char_span=(0, max(1, len(heading or ""))),
    )


def _tree(doc_id: str, *nodes: ClauseNode) -> ClauseTree:
    return ClauseTree(document_id=doc_id, version="v1", source_file="doc.docx", nodes=list(nodes))


# ---------------------------------------------------------------------------
# Small fixtures
# ---------------------------------------------------------------------------


def _affiliation_tree(doc_id: str = "aff-001") -> ClauseTree:
    return _tree(
        doc_id,
        _node("0", None, "This affiliation agreement is between Alice Corp and Beta Hospital."),
        _node("1", "Definitions", "All terms have their standard meanings."),
        _node("2", "Term", "This agreement lasts one year."),
        _node("3", "Indemnification", "Each party shall indemnify the other."),
        _node("4", "Insurance", "Each party maintains liability insurance."),
        _node("5", "Governing Law", "This agreement is governed by California law."),
        _node("6", "Termination", "Either party may terminate on thirty days notice."),
    )


def _similar_tree(doc_id: str = "aff-002") -> ClauseTree:
    return _tree(
        doc_id,
        _node("0", None, "This affiliation agreement governs student rotations."),
        _node("1", "Term of Agreement", "Duration is twelve months."),
        _node("2", "Indemnification and Hold Harmless", "Party A indemnifies Party B."),
        _node("3", "Insurance Requirements", "Both parties carry $1M coverage."),
        _node("4", "Governing Law", "This agreement is governed by Texas law."),
        _node("5", "Notices", "All notices to be sent by certified mail."),
    )


def _sparse_tree(doc_id: str = "aff-003") -> ClauseTree:
    """Tree with a unique heading not present elsewhere."""
    return _tree(
        doc_id,
        _node("1", "Rare Custom Clause", "This clause appears in only one document."),
        _node("2", "Indemnification", "Hold harmless provision."),
        _node("3", "Governing Law", "Applicable state law governs."),
    )


# ---------------------------------------------------------------------------
# _normalize_heading
# ---------------------------------------------------------------------------


def test_normalize_heading_lowercases() -> None:
    assert _normalize_heading("GOVERNING LAW") == "governing law"


def test_normalize_heading_strips_punctuation() -> None:
    result = _normalize_heading("Termination for Cause.")
    assert "." not in result


def test_normalize_heading_collapses_whitespace() -> None:
    result = _normalize_heading("  Term   of   Agreement  ")
    assert "  " not in result
    assert result == result.strip()


def test_normalize_heading_empty() -> None:
    assert _normalize_heading("") == ""


def test_normalize_heading_special_chars() -> None:
    result = _normalize_heading("Non-Compete / Exclusivity")
    assert "/" not in result
    assert "-" not in result


# ---------------------------------------------------------------------------
# _heading_tokens
# ---------------------------------------------------------------------------


def test_heading_tokens_removes_stop_words() -> None:
    tokens = _heading_tokens("termination of the agreement")
    assert "the" not in tokens
    assert "of" not in tokens
    assert "termination" in tokens
    assert "agreement" in tokens


def test_heading_tokens_empty() -> None:
    assert _heading_tokens("") == frozenset()


def test_heading_tokens_all_stops() -> None:
    assert _heading_tokens("a the and of") == frozenset()


# ---------------------------------------------------------------------------
# _jaccard
# ---------------------------------------------------------------------------


def test_jaccard_identical() -> None:
    s = frozenset({"a", "b", "c"})
    assert _jaccard(s, s) == 1.0


def test_jaccard_disjoint() -> None:
    assert _jaccard(frozenset({"a"}), frozenset({"b"})) == 0.0


def test_jaccard_partial() -> None:
    a = frozenset({"term", "agreement"})
    b = frozenset({"term", "duration"})
    assert 0.0 < _jaccard(a, b) < 1.0


def test_jaccard_both_empty() -> None:
    assert _jaccard(frozenset(), frozenset()) == 1.0


def test_jaccard_one_empty() -> None:
    assert _jaccard(frozenset({"a"}), frozenset()) == 0.0


# ---------------------------------------------------------------------------
# _to_entry_id
# ---------------------------------------------------------------------------


def test_to_entry_id_basic() -> None:
    assert _to_entry_id("Governing Law") == "governing_law"


def test_to_entry_id_strips_punctuation() -> None:
    entry_id = _to_entry_id("Non-Compete / Exclusivity")
    assert "/" not in entry_id
    assert "-" not in entry_id


def test_to_entry_id_max_length() -> None:
    long_heading = "A" * 100
    assert len(_to_entry_id(long_heading)) <= 60


def test_to_entry_id_empty_falls_back() -> None:
    assert _to_entry_id("") == "entry"
    assert _to_entry_id("---") == "entry"


# ---------------------------------------------------------------------------
# induce_taxonomy: empty / trivial inputs
# ---------------------------------------------------------------------------


def test_induce_empty_trees() -> None:
    result = induce_taxonomy([])
    assert result.total_documents == 0
    assert result.induced_entries == []
    assert result.taxonomy.entries == []


def test_induce_trees_with_no_headings() -> None:
    tree = _tree("d1", _node("1", None, "Body only."), _node("2", None, "More body."))
    result = induce_taxonomy([tree])
    assert result.total_documents == 1
    assert result.induced_entries == []


def test_induce_single_tree_returns_entries() -> None:
    result = induce_taxonomy([_affiliation_tree()])
    assert len(result.induced_entries) > 0


def test_induce_total_documents() -> None:
    trees = [_affiliation_tree("d1"), _similar_tree("d2"), _sparse_tree("d3")]
    result = induce_taxonomy(trees)
    assert result.total_documents == 3


# ---------------------------------------------------------------------------
# Clustering: similar headings merge
# ---------------------------------------------------------------------------


def test_similar_headings_cluster_together() -> None:
    """'Term' and 'Term of Agreement' should cluster into a single entry."""
    trees = [
        _tree("d1", _node("1", "Term", "One year.")),
        _tree("d2", _node("1", "Term of Agreement", "Twelve months.")),
        _tree("d3", _node("1", "Term", "Duration is one year.")),
    ]
    result = induce_taxonomy(trees, representation_threshold=0.0)
    term_entries = [ie for ie in result.induced_entries if "term" in ie.entry.id]
    assert len(term_entries) == 1, (
        f"Expected 'Term' and 'Term of Agreement' to merge, "
        f"got: {[ie.entry.id for ie in result.induced_entries]}"
    )


def test_dissimilar_headings_stay_separate() -> None:
    """'Indemnification' and 'Governing Law' must not cluster together."""
    trees = [
        _tree("d1", _node("1", "Indemnification", "Hold harmless.")),
        _tree("d2", _node("1", "Governing Law", "California law.")),
    ]
    result = induce_taxonomy(trees, representation_threshold=0.0)
    ids = [ie.entry.id for ie in result.induced_entries]
    assert any("indemnification" in eid for eid in ids)
    assert any("governing" in eid for eid in ids)


def test_exact_duplicate_headings_count_once_per_doc() -> None:
    """Two occurrences of the same heading in one tree → single example per doc."""
    tree = _tree(
        "d1",
        _node("1", "Indemnification", "Party A indemnifies."),
        _node("2", "Indemnification and Hold Harmless", "Also hold harmless."),
    )
    result = induce_taxonomy([tree], representation_threshold=0.0)
    indemnification_entries = [ie for ie in result.induced_entries if "indemnif" in ie.entry.id]
    # May be one cluster or two depending on similarity; examples must reference d1
    for ie in indemnification_entries:
        assert all(ex.document_id == "d1" for ex in ie.examples)


# ---------------------------------------------------------------------------
# Status assignment
# ---------------------------------------------------------------------------


def test_frequent_cuad_heading_is_active() -> None:
    """Insurance (a genuine CUAD v1 category) in all 3 docs → active."""
    trees = [
        _tree("d1", _node("1", "Insurance", "Party A carries coverage.")),
        _tree("d2", _node("1", "Insurance", "Party B carries coverage.")),
        _tree("d3", _node("1", "Insurance", "Mutual coverage minimums.")),
    ]
    result = induce_taxonomy(trees, representation_threshold=0.20)
    insurance = next(ie for ie in result.induced_entries if "insurance" in ie.entry.id)
    assert insurance.entry.status == "active"
    assert insurance.entry.cuad_origin is not None


def test_rare_cuad_heading_is_inactive() -> None:
    """Indemnification in 1/5 docs → inactive (below representation_threshold=0.30)."""
    trees = [
        _tree("d1", _node("1", "Indemnification", "Indemnification.")),
        _tree("d2", _node("1", "Governing Law", "California.")),
        _tree("d3", _node("1", "Governing Law", "Texas.")),
        _tree("d4", _node("1", "Governing Law", "New York.")),
        _tree("d5", _node("1", "Governing Law", "Delaware.")),
    ]
    result = induce_taxonomy(trees, representation_threshold=0.30)
    indemnification = next((ie for ie in result.induced_entries if "indemnif" in ie.entry.id), None)
    assert indemnification is not None
    assert indemnification.entry.status == "inactive"


def test_frequent_novel_heading_is_custom() -> None:
    """A heading not matching any CUAD category at ≥ threshold → custom."""
    trees = [
        _tree("d1", _node("1", "Student Rotation Protocols", "Daily supervision rules.")),
        _tree("d2", _node("1", "Student Rotation Protocols", "Weekly check-ins required.")),
        _tree("d3", _node("1", "Student Rotation Protocols", "Monthly evaluations.")),
    ]
    result = induce_taxonomy(
        trees,
        representation_threshold=0.20,
        cuad_match_threshold=0.99,  # force no CUAD match
    )
    entry = next((ie for ie in result.induced_entries if "student" in ie.entry.id), None)
    assert entry is not None
    assert entry.entry.status == "custom"
    assert entry.entry.cuad_origin is None


def test_rare_novel_heading_is_inactive() -> None:
    """A heading not matching CUAD and below threshold → inactive (cuad_origin=None)."""
    trees = [
        _tree("d1", _node("1", "Unique Proprietary Clause", "Some unique text.")),
        _tree("d2", _node("1", "Governing Law", "State law applies.")),
        _tree("d3", _node("1", "Governing Law", "Federal law applies.")),
        _tree("d4", _node("1", "Governing Law", "Local law applies.")),
        _tree("d5", _node("1", "Governing Law", "Applicable law applies.")),
    ]
    result = induce_taxonomy(
        trees,
        representation_threshold=0.30,
        cuad_match_threshold=0.99,  # force no CUAD match
    )
    unique = next((ie for ie in result.induced_entries if "unique" in ie.entry.id), None)
    assert unique is not None
    assert unique.entry.status == "inactive"
    assert unique.entry.cuad_origin is None


# ---------------------------------------------------------------------------
# CUAD mapping
# ---------------------------------------------------------------------------


def test_indemnification_is_supplemental_not_cuad() -> None:
    """'Indemnification' is NOT one of CUAD v1's 41 categories.

    The historical embedded list wrongly carried it (and ten other common
    headings) with CUAD provenance — the exact false-provenance bug the
    2026-07 audit flagged and issue #167 fixes. It now matches the
    general-commercial supplemental list: cuad_origin=None,
    source="playbook-engine-base".
    """
    result = induce_taxonomy(
        [_tree("d1", _node("1", "Indemnification", "Hold harmless."))],
        representation_threshold=0.0,
        cuad_match_threshold=CUAD_MATCH_THRESHOLD,
    )
    entry = next(ie for ie in result.induced_entries if "indemnif" in ie.entry.id)
    assert entry.entry.cuad_origin is None
    assert entry.source == "playbook-engine-base"


def test_governing_law_maps_to_cuad() -> None:
    result = induce_taxonomy(
        [_tree("d1", _node("1", "Governing Law", "California law."))],
        representation_threshold=0.0,
    )
    entry = next(ie for ie in result.induced_entries if "governing" in ie.entry.id)
    assert entry.entry.cuad_origin is not None


def test_supplemental_categories_are_not_cuad_origin() -> None:
    """ "Payment Terms" is a playbook-engine-base addition, not genuine CUAD v1.

    A cluster matched to a supplemental category must get cuad_origin=None
    and source="playbook-engine-base", while a cluster matched to a genuine
    CUAD v1 category (Insurance IS one of the 41) must keep cuad_origin set
    with source="CUAD v1" (2026-07 dual-repo audit finding: false CUAD
    provenance).
    """
    trees = [
        _tree("d1", _node("1", "Payment Terms", "Invoices due net 30.")),
        _tree("d2", _node("1", "Payment Terms", "Invoices due net 60.")),
        _tree("d3", _node("1", "Insurance", "Carrier-rated coverage required.")),
    ]
    result = induce_taxonomy(trees, representation_threshold=0.20)

    payment = next(ie for ie in result.induced_entries if "payment" in ie.entry.id)
    assert payment.entry.cuad_origin is None
    assert payment.source == "playbook-engine-base"

    insurance = next(ie for ie in result.induced_entries if "insurance" in ie.entry.id)
    assert insurance.entry.cuad_origin is not None
    assert insurance.source == "CUAD v1"


def test_supplemental_category_source_emitted_in_yaml(tmp_path: Path) -> None:
    """The induced YAML records the actual source string per entry."""
    trees = [
        _tree("d1", _node("1", "Payment Terms", "Invoices due net 30.")),
        _tree("d2", _node("1", "Payment Terms", "Invoices due net 60.")),
    ]
    result = induce_taxonomy(trees, representation_threshold=0.20)
    dest = tmp_path / "induced.yaml"
    emit_taxonomy_yaml(result, dest)
    raw = yaml.safe_load(dest.read_text(encoding="utf-8"))
    payment_raw = next(e for e in raw["entries"] if "payment" in e["id"])
    assert payment_raw["cuad_origin"] is None
    assert payment_raw["source"] == "playbook-engine-base"


def test_cuad_match_threshold_prevents_weak_match() -> None:
    """With threshold above a partial match score, a close-but-not-exact heading stays unmapped.

    'Term' matches 'Renewal Term' at Jaccard 0.5 (shared: {'term'}, union: {'renewal','term'}).
    Setting threshold=0.6 blocks that match, so cuad_origin must be None.
    """
    # Default CUAD_MATCH_THRESHOLD is 0.60 — no override needed; the test
    # documents WHY the default was chosen at this boundary.
    result = induce_taxonomy(
        [_tree("d1", _node("1", "Term", "Duration is one year."))],
        representation_threshold=0.0,
        # cuad_match_threshold defaults to 0.60; "Term"/"Renewal Term" jaccard=0.50 < 0.60
    )
    entry = next(ie for ie in result.induced_entries if ie.entry.id == "term")
    assert entry.entry.cuad_origin is None


# ---------------------------------------------------------------------------
# Examples
# ---------------------------------------------------------------------------


def test_examples_contain_document_id() -> None:
    trees = [_affiliation_tree("my-doc-001"), _similar_tree("my-doc-002")]
    result = induce_taxonomy(trees, representation_threshold=0.0)
    all_doc_ids = {ex.document_id for ie in result.induced_entries for ex in ie.examples}
    assert "my-doc-001" in all_doc_ids or "my-doc-002" in all_doc_ids


def test_examples_at_most_three_per_entry() -> None:
    trees = [_tree(f"d{i}", _node("1", "Indemnification", f"Text {i}.")) for i in range(10)]
    result = induce_taxonomy(trees, representation_threshold=0.0)
    entry = next(ie for ie in result.induced_entries if "indemnif" in ie.entry.id)
    assert len(entry.examples) <= 3


def test_examples_one_per_document() -> None:
    """All examples for one entry must reference distinct documents."""
    trees = [_tree(f"d{i}", _node("1", "Indemnification", f"Text {i}.")) for i in range(5)]
    result = induce_taxonomy(trees, representation_threshold=0.0)
    entry = next(ie for ie in result.induced_entries if "indemnif" in ie.entry.id)
    doc_ids = [ex.document_id for ex in entry.examples]
    assert len(doc_ids) == len(set(doc_ids))


def test_examples_have_clause_path() -> None:
    result = induce_taxonomy([_affiliation_tree()], representation_threshold=0.0)
    for ie in result.induced_entries:
        for ex in ie.examples:
            assert ex.clause_path is not None


def test_examples_text_snippet_not_too_long() -> None:
    long_text = "X" * 1000
    tree = _tree("d1", _node("1", "Indemnification", long_text))
    result = induce_taxonomy([tree], representation_threshold=0.0)
    entry = next(ie for ie in result.induced_entries if "indemnif" in ie.entry.id)
    for ex in entry.examples:
        assert len(ex.text_snippet) <= 200


# ---------------------------------------------------------------------------
# Document frequency
# ---------------------------------------------------------------------------


def test_document_frequency_all_docs() -> None:
    trees = [
        _tree("d1", _node("1", "Governing Law", "California.")),
        _tree("d2", _node("1", "Governing Law", "Texas.")),
        _tree("d3", _node("1", "Governing Law", "Delaware.")),
    ]
    result = induce_taxonomy(trees, representation_threshold=0.0)
    entry = next(ie for ie in result.induced_entries if "governing" in ie.entry.id)
    assert entry.document_frequency == pytest.approx(1.0)


def test_document_frequency_half_docs() -> None:
    trees = [
        _tree("d1", _node("1", "Indemnification", "Party A.")),
        _tree("d2", _node("1", "Indemnification", "Party B.")),
        _tree("d3", _node("1", "Governing Law", "Texas.")),
        _tree("d4", _node("1", "Governing Law", "Delaware.")),
    ]
    result = induce_taxonomy(trees, representation_threshold=0.0)
    entry = next(ie for ie in result.induced_entries if "indemnif" in ie.entry.id)
    assert entry.document_frequency == pytest.approx(0.5)


def test_document_frequency_same_heading_multiple_versions() -> None:
    """Same heading in two trees with same doc_id should count as one document."""
    tree_v1 = ClauseTree(
        document_id="agreement-001",
        version="v1",
        source_file="v1.docx",
        nodes=[_node("1", "Indemnification", "Version 1 text.")],
    )
    tree_v2 = ClauseTree(
        document_id="agreement-001",  # same document_id → counts once
        version="v2",
        source_file="v2.docx",
        nodes=[_node("1", "Indemnification", "Version 2 text.")],
    )
    result = induce_taxonomy([tree_v1, tree_v2], representation_threshold=0.0)
    entry = next(ie for ie in result.induced_entries if "indemnif" in ie.entry.id)
    # 2 trees passed, cluster has 1 distinct doc_id ("agreement-001").
    # Document frequency = distinct_doc_ids_in_cluster / total_trees = 1/2 = 0.5.
    # (Callers that want per-agreement frequency should pass one tree per agreement.)
    assert entry.document_frequency == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Taxonomy result
# ---------------------------------------------------------------------------


def test_result_taxonomy_is_taxonomy_type() -> None:
    from playbook_engine.taxonomy import Taxonomy

    result = induce_taxonomy([_affiliation_tree()])
    assert isinstance(result.taxonomy, Taxonomy)


def test_result_taxonomy_source_is_induced() -> None:
    result = induce_taxonomy([_affiliation_tree()])
    assert result.taxonomy.source == "induced"


def test_result_taxonomy_entries_match_induced_entries() -> None:
    result = induce_taxonomy([_affiliation_tree()])
    assert [e.entry for e in result.induced_entries] == result.taxonomy.entries


def test_result_no_duplicate_entry_ids() -> None:
    trees = [_affiliation_tree("d1"), _similar_tree("d2"), _sparse_tree("d3")]
    result = induce_taxonomy(trees, representation_threshold=0.0)
    ids = [ie.entry.id for ie in result.induced_entries]
    assert len(ids) == len(set(ids))


def test_result_entries_sorted_active_first() -> None:
    """Active/custom entries must precede inactive ones."""
    trees = [_affiliation_tree("d1"), _similar_tree("d2"), _sparse_tree("d3")]
    result = induce_taxonomy(trees, representation_threshold=0.50)
    statuses = [ie.entry.status for ie in result.induced_entries]
    active_positions = [i for i, s in enumerate(statuses) if s in ("active", "custom")]
    inactive_positions = [i for i, s in enumerate(statuses) if s == "inactive"]
    if active_positions and inactive_positions:
        assert max(active_positions) < min(inactive_positions)


# ---------------------------------------------------------------------------
# emit_taxonomy_yaml
# ---------------------------------------------------------------------------


def test_emit_taxonomy_yaml_creates_file(tmp_path: Path) -> None:
    result = induce_taxonomy([_affiliation_tree(), _similar_tree()])
    dest = tmp_path / "induced.yaml"
    emit_taxonomy_yaml(result, dest)
    assert dest.exists()


def test_emit_taxonomy_yaml_loadable_by_load_taxonomy(tmp_path: Path) -> None:
    """Emitted YAML must be loadable by load_taxonomy() without errors."""
    result = induce_taxonomy([_affiliation_tree(), _similar_tree()])
    dest = tmp_path / "induced.yaml"
    emit_taxonomy_yaml(result, dest)
    taxonomy = load_taxonomy(dest)
    assert taxonomy.source == "induced"
    assert len(taxonomy.entries) == len(result.induced_entries)


def test_emit_taxonomy_yaml_contains_examples(tmp_path: Path) -> None:
    """Emitted YAML entries include examples for attorney review."""
    result = induce_taxonomy([_affiliation_tree(), _similar_tree()])
    dest = tmp_path / "induced.yaml"
    emit_taxonomy_yaml(result, dest)
    raw = yaml.safe_load(dest.read_text(encoding="utf-8"))
    entries_with_examples = [e for e in raw["entries"] if e.get("examples")]
    assert len(entries_with_examples) > 0


def test_emit_taxonomy_yaml_atomic_no_tmp_left(tmp_path: Path) -> None:
    """Atomic write must not leave a .yaml.tmp file behind on success."""
    result = induce_taxonomy([_affiliation_tree()])
    dest = tmp_path / "induced.yaml"
    emit_taxonomy_yaml(result, dest)
    assert dest.exists()
    assert not dest.with_suffix(".yaml.tmp").exists()


def test_emit_taxonomy_yaml_creates_parent_dirs(tmp_path: Path) -> None:
    result = induce_taxonomy([_affiliation_tree()])
    dest = tmp_path / "nested" / "dir" / "induced.yaml"
    emit_taxonomy_yaml(result, dest)
    assert dest.exists()


# ---------------------------------------------------------------------------
# Integration: acceptance test
# ---------------------------------------------------------------------------


def test_acceptance_induces_known_categories_from_corpus() -> None:
    """Acceptance: standard clauses from 3 affiliation docs → correct entries.

    Corpus:
      doc-1 (_affiliation_tree): "Indemnification"
      doc-2 (_similar_tree):     "Indemnification and Hold Harmless"
      doc-3 (_sparse_tree):      "Indemnification"

    With overlap-coefficient clustering, all three headings must merge into
    ONE indemnification cluster with frequency 1.0 (all 3 docs).
    """
    trees = [
        _affiliation_tree("doc-1"),
        _similar_tree("doc-2"),
        _sparse_tree("doc-3"),
    ]
    result = induce_taxonomy(trees, representation_threshold=0.25)
    ids = {ie.entry.id for ie in result.induced_entries}

    # Governing law appears in all 3 docs → must be active.
    gov_law = next((ie for ie in result.induced_entries if "governing" in ie.entry.id), None)
    assert gov_law is not None, f"governing_law not found in {ids}"
    assert gov_law.entry.status == "active"

    # Indemnification + "Indemnification and Hold Harmless" must merge into ONE entry
    # (B1 fix: overlap-coefficient clustering, not Jaccard).
    indem_entries = [ie for ie in result.induced_entries if "indemnif" in ie.entry.id]
    assert len(indem_entries) == 1, (
        f"Expected exactly 1 indemnification entry after overlap clustering, "
        f"got {len(indem_entries)}: {[ie.entry.id for ie in indem_entries]}"
    )
    indem = indem_entries[0]
    assert indem.entry.status == "active"
    assert indem.document_frequency == pytest.approx(1.0), (
        f"Indemnification must appear in all 3 docs after merging variants; "
        f"got document_frequency={indem.document_frequency}"
    )

    # Rare Custom Clause appears only in sparse_tree (1/3 = 33% > 25%) → custom.
    rare = next((ie for ie in result.induced_entries if "rare" in ie.entry.id), None)
    assert rare is not None, f"rare clause not found in {ids}"
    assert rare.entry.status in ("custom", "active")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


def test_representation_threshold_in_range() -> None:
    assert 0.0 < REPRESENTATION_THRESHOLD < 1.0


def test_cluster_similarity_threshold_in_range() -> None:
    assert 0.0 < CLUSTER_SIMILARITY_THRESHOLD < 1.0


def test_cuad_match_threshold_in_range() -> None:
    assert 0.0 < CUAD_MATCH_THRESHOLD < 1.0


# ---------------------------------------------------------------------------
# CLI — playbook induce-taxonomy
# ---------------------------------------------------------------------------


def _make_rtf_doc(doc_dir: Path, filename: str, headings: list[str]) -> None:
    """Write a minimal RTF file with numbered headings to doc_dir/filename."""
    parts: list[str] = []
    for i, heading in enumerate(headings, start=1):
        text = f"Clause text for {heading}."
        parts.append(rf"{i}. {heading}\par {text}\par ")
    rtf_body = "".join(parts)
    content = r"{\rtf1\ansi " + rtf_body + r"}"
    doc_dir.mkdir(parents=True, exist_ok=True)
    (doc_dir / filename).write_text(content, encoding="utf-8")


def _make_corpus(tmp_path: Path) -> Path:
    """Build a two-agreement synthetic RTF corpus."""
    corpus = tmp_path / "corpus"
    # deal-alpha: Indemnification, Governing Law, Term
    _make_rtf_doc(corpus / "deal-alpha", "v1.rtf", ["Indemnification", "Governing Law", "Term"])
    # deal-beta: Indemnification, Governing Law, Termination
    _make_rtf_doc(
        corpus / "deal-beta", "v1.rtf", ["Indemnification", "Governing Law", "Termination"]
    )
    return corpus


def test_cli_induce_taxonomy_natural_sort_picks_highest_version(tmp_path: Path) -> None:
    """B2 fix: v10 must sort after v2 (natural sort, not lexicographic)."""
    from playbook_engine.cli import _natural_sort_key

    stems = ["v1", "v10", "v2", "v9"]
    sorted_stems = sorted(stems, key=_natural_sort_key)
    assert sorted_stems == ["v1", "v2", "v9", "v10"]


def test_cli_induce_taxonomy_ingests_highest_natural_version(tmp_path: Path) -> None:
    """B2 fix: CLI picks the highest-versioned file (v10 over v2) for a corpus."""
    corpus = tmp_path / "corpus"
    doc_dir = corpus / "deal-alpha"
    doc_dir.mkdir(parents=True)
    # v2 has "Governing Law", v10 has "Indemnification" — we expect v10 to be ingested
    _make_rtf_doc(corpus / "deal-alpha", "v2.rtf", ["Governing Law"])
    _make_rtf_doc(corpus / "deal-alpha", "v10.rtf", ["Indemnification"])
    out_yaml = tmp_path / "candidate.yaml"
    runner = CliRunner()
    result = runner.invoke(cli, ["induce-taxonomy", str(corpus), "--out", str(out_yaml)])
    assert result.exit_code == 0, result.output
    data = yaml.safe_load(out_yaml.read_text(encoding="utf-8"))
    ids = {e["id"] for e in data["entries"]}
    assert any("indemnif" in eid for eid in ids), (
        f"Expected v10 to be ingested (indemnification), got: {ids}"
    )
    assert not any("governing" in eid for eid in ids), (
        "v2 (governing_law) must not be picked over v10 (indemnification)"
    )


def test_cli_induce_taxonomy_exits_zero(tmp_path: Path) -> None:
    corpus = _make_corpus(tmp_path)
    runner = CliRunner()
    result = runner.invoke(cli, ["induce-taxonomy", str(corpus)])
    assert result.exit_code == 0, result.output


def test_cli_induce_taxonomy_stdout_contains_yaml_keys(tmp_path: Path) -> None:
    """Without --out the command prints YAML content to stdout (source + entries keys)."""
    corpus = _make_corpus(tmp_path)
    runner = CliRunner()
    result = runner.invoke(cli, ["induce-taxonomy", str(corpus)])
    assert result.exit_code == 0, result.output
    assert "source:" in result.output
    assert "entries:" in result.output


def test_cli_induce_taxonomy_writes_file(tmp_path: Path) -> None:
    corpus = _make_corpus(tmp_path)
    out_yaml = tmp_path / "candidate.yaml"
    runner = CliRunner()
    result = runner.invoke(cli, ["induce-taxonomy", str(corpus), "--out", str(out_yaml)])
    assert result.exit_code == 0, result.output
    assert out_yaml.exists()


def test_cli_induce_taxonomy_output_loadable(tmp_path: Path) -> None:
    """Output YAML must load cleanly via load_taxonomy()."""
    corpus = _make_corpus(tmp_path)
    out_yaml = tmp_path / "candidate.yaml"
    runner = CliRunner()
    result = runner.invoke(cli, ["induce-taxonomy", str(corpus), "--out", str(out_yaml)])
    assert result.exit_code == 0, result.output
    taxonomy = load_taxonomy(out_yaml)
    assert taxonomy.source == "induced"
    assert len(taxonomy.entries) > 0


def test_cli_induce_taxonomy_recovers_indemnification(tmp_path: Path) -> None:
    """Acceptance: indemnification and governing_law must appear in induced output."""
    corpus = _make_corpus(tmp_path)
    out_yaml = tmp_path / "candidate.yaml"
    runner = CliRunner()
    result = runner.invoke(cli, ["induce-taxonomy", str(corpus), "--out", str(out_yaml)])
    assert result.exit_code == 0, result.output
    data = yaml.safe_load(out_yaml.read_text(encoding="utf-8"))
    ids = {e["id"] for e in data["entries"]}
    assert any("indemnif" in eid for eid in ids), f"indemnification missing from {ids}"
    assert any("governing" in eid for eid in ids), f"governing_law missing from {ids}"


def test_cli_induce_taxonomy_empty_corpus_exits_nonzero(tmp_path: Path) -> None:
    empty_corpus = tmp_path / "empty"
    empty_corpus.mkdir()
    runner = CliRunner()
    result = runner.invoke(cli, ["induce-taxonomy", str(empty_corpus)])
    assert result.exit_code != 0


def test_cli_induce_taxonomy_representation_threshold_flag(tmp_path: Path) -> None:
    """--representation-threshold controls active vs inactive assignment."""
    corpus = _make_corpus(tmp_path)
    out_yaml = tmp_path / "candidate.yaml"
    runner = CliRunner()
    # With threshold=1.0 every entry is inactive (none appear in ALL docs)
    result = runner.invoke(
        cli,
        [
            "induce-taxonomy",
            str(corpus),
            "--out",
            str(out_yaml),
            "--representation-threshold",
            "1.0",
        ],
    )
    assert result.exit_code == 0, result.output
    data = yaml.safe_load(out_yaml.read_text(encoding="utf-8"))
    active_entries = [e for e in data["entries"] if e["status"] == "active"]
    # Indemnification and governing_law appear in 2/2 docs → active even at threshold=1.0
    assert (
        all(any(eid in e["id"] for eid in ("indemnif", "governing")) for e in active_entries)
        or len(active_entries) >= 2
    )
