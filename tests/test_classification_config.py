"""Tests for the producer-configurable classification thresholds (issue #168).

SECURITY NOTE: All fixtures use programmatically constructed ClauseTree
objects and synthetic taxonomy entries. No real agreement files are
referenced.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from playbook_engine.clause_classifier import (
    AMBIGUITY_THRESHOLD,
    AUTO_CLASSIFY_THRESHOLD,
    ClassificationHint,
    ClauseClassification,
    classify_tree,
)
from playbook_engine.clause_tree import ClauseNode, ClauseTree
from playbook_engine.config import ClassificationConfig, ConfigError, load_config
from playbook_engine.taxonomy import Taxonomy, TaxonomyEntry

TAXONOMY_PATH = Path(__file__).parent.parent / "spec" / "taxonomy" / "affiliation-agreement.yaml"


# ---------------------------------------------------------------------------
# Helpers (mirrors tests/test_config.py and tests/test_clause_classifier.py)
# ---------------------------------------------------------------------------


def _write_config(tmp_path: Path, content: str) -> Path:
    cfg = tmp_path / "playbook.config.yaml"
    cfg.write_text(content, encoding="utf-8")
    return cfg


def _minimal_config(tmp_path: Path, *, extra_yaml: str = "") -> Path:
    tax_dst = tmp_path / "taxonomy.yaml"
    tax_dst.write_text(TAXONOMY_PATH.read_text(), encoding="utf-8")
    return _write_config(
        tmp_path,
        f"""
agreement_type:
  id: test-type
  name: "Test Agreement"
baseline:
  template: null
taxonomy: taxonomy.yaml
provenance:
  our_party_aliases: ["FixtureCorp"]
{extra_yaml}""",
    )


def _entry(entry_id: str, label: str, status: str = "active") -> TaxonomyEntry:
    return TaxonomyEntry(id=entry_id, label=label, status=status, cuad_origin=None, description="")


def _taxonomy(*entries: TaxonomyEntry) -> Taxonomy:
    return Taxonomy(source="test", entries=list(entries))


def _node(path: str, heading: str | None = None, text: str = "") -> ClauseNode:
    return ClauseNode(
        clause_path=path,
        heading=heading,
        text=text,
        char_span=(0, max(1, len(heading or ""))),
    )


def _tree(*nodes: ClauseNode, doc_id: str = "doc") -> ClauseTree:
    return ClauseTree(document_id=doc_id, version="v1", source_file="doc.docx", nodes=list(nodes))


class _SpyJudge:
    """Judge that records whether it was called; always returns 'unclassified'."""

    def __init__(self) -> None:
        self.called = False

    def classify_batch(
        self,
        nodes: list[ClauseNode],
        taxonomy: Taxonomy,
        hints: list[ClassificationHint | None] | None = None,
    ) -> list[ClauseClassification]:
        self.called = True
        return [
            ClauseClassification(taxonomy_id=None, confidence=0.0, basis="unclassified")
            for _ in nodes
        ]


# ---------------------------------------------------------------------------
# test_defaults_unchanged
# ---------------------------------------------------------------------------


def test_defaults_unchanged(tmp_path: Path) -> None:
    """No ``classification:`` block -> config thresholds equal the classifier's
    own module constants (0.70/0.85), and threading those config values
    through ``classify_tree`` produces byte-identical results to calling it
    with no threshold override at all (today's pre-#168 behavior)."""
    path = _minimal_config(tmp_path)
    cfg = load_config(path)

    assert cfg.classification == ClassificationConfig()
    assert cfg.classification.ambiguity_threshold == AMBIGUITY_THRESHOLD == 0.70
    assert cfg.classification.auto_classify_threshold == AUTO_CLASSIFY_THRESHOLD == 0.85

    # In-band heading fixture (Jaccard = 7/9 ≈ 0.778, in [0.70, 0.85)).
    taxonomy = _taxonomy(_entry("entry_alpha", "alpha beta gamma delta epsilon zeta eta iota"))
    tree = _tree(_node("1", "alpha beta gamma delta epsilon zeta eta theta", "Some text."))

    judge_a = _SpyJudge()
    judge_b = _SpyJudge()
    no_override = classify_tree(tree, taxonomy, judge_a)
    with_config_defaults = classify_tree(
        tree,
        taxonomy,
        judge_b,
        ambiguity_threshold=cfg.classification.ambiguity_threshold,
        auto_classify_threshold=cfg.classification.auto_classify_threshold,
    )

    assert judge_a.called is True
    assert judge_b.called is True
    assert [r.classification.to_dict() for r in no_override] == [
        r.classification.to_dict() for r in with_config_defaults
    ]


# ---------------------------------------------------------------------------
# test_override_changes_banding
# ---------------------------------------------------------------------------


def test_override_changes_banding(tmp_path: Path) -> None:
    """``auto_classify_threshold: 0.99`` -> a heading that auto-classified at
    0.85-0.99 similarity (Jaccard = 0.9, above today's default of 0.85) now
    lands in the ambiguous band and is escalated to the judge."""
    path = _minimal_config(
        tmp_path,
        extra_yaml="classification:\n  auto_classify_threshold: 0.99\n",
    )
    cfg = load_config(path)
    assert cfg.classification.auto_classify_threshold == 0.99
    assert cfg.classification.ambiguity_threshold == AMBIGUITY_THRESHOLD  # unset -> default

    # Jaccard construction:
    #   heading tokens = {a1..a10}                       (10 tokens)
    #   entry tokens   = {a1..a9}                          (9 tokens, subset)
    #   intersection = 9, union = 10 -> Jaccard = 0.9
    heading = "a1 a2 a3 a4 a5 a6 a7 a8 a9 a10"
    label = "a1 a2 a3 a4 a5 a6 a7 a8 a9"
    taxonomy = _taxonomy(_entry("entry_a", label))
    tree = _tree(_node("1", heading, "Some text."))

    # With today's default (0.85), 0.9 auto-classifies without the judge.
    default_spy = _SpyJudge()
    default_results = classify_tree(tree, taxonomy, default_spy)
    assert default_spy.called is False
    assert default_results[0].classification.basis == "heading_similarity"
    assert default_results[0].classification.taxonomy_id == "entry_a"

    # With the config override (0.99), 0.9 falls below auto-classify and
    # (since 0.9 >= ambiguity_threshold=0.70) lands in the ambiguous band ->
    # judge is called.
    override_spy = _SpyJudge()
    classify_tree(
        tree,
        taxonomy,
        override_spy,
        ambiguity_threshold=cfg.classification.ambiguity_threshold,
        auto_classify_threshold=cfg.classification.auto_classify_threshold,
    )
    assert override_spy.called is True, "Judge MUST be called once auto_classify_threshold > 0.9"


# ---------------------------------------------------------------------------
# test_invalid_ordering_rejected
# ---------------------------------------------------------------------------


def test_invalid_ordering_rejected(tmp_path: Path) -> None:
    """``ambiguity_threshold: 0.9, auto_classify_threshold: 0.8`` (inverted
    ordering) -> ``ConfigError`` naming both keys."""
    path = _minimal_config(
        tmp_path,
        extra_yaml="classification:\n  ambiguity_threshold: 0.9\n  auto_classify_threshold: 0.8\n",
    )
    with pytest.raises(ConfigError, match="ambiguity_threshold") as exc_info:
        load_config(path)
    assert "auto_classify_threshold" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Additional validation edge cases
# ---------------------------------------------------------------------------


def test_classification_override_parsed(tmp_path: Path) -> None:
    path = _minimal_config(
        tmp_path,
        extra_yaml="classification:\n  ambiguity_threshold: 0.5\n  auto_classify_threshold: 0.9\n",
    )
    cfg = load_config(path)
    assert cfg.classification.ambiguity_threshold == 0.5
    assert cfg.classification.auto_classify_threshold == 0.9


def test_classification_threshold_out_of_range_rejected(tmp_path: Path) -> None:
    path = _minimal_config(
        tmp_path,
        extra_yaml="classification:\n  auto_classify_threshold: 1.5\n",
    )
    with pytest.raises(ConfigError, match="auto_classify_threshold"):
        load_config(path)


def test_classification_threshold_zero_rejected(tmp_path: Path) -> None:
    path = _minimal_config(
        tmp_path,
        extra_yaml="classification:\n  ambiguity_threshold: 0\n",
    )
    with pytest.raises(ConfigError, match="ambiguity_threshold"):
        load_config(path)


def test_classification_threshold_not_a_number_rejected(tmp_path: Path) -> None:
    path = _minimal_config(
        tmp_path,
        extra_yaml='classification:\n  ambiguity_threshold: "high"\n',
    )
    with pytest.raises(ConfigError, match="ambiguity_threshold must be a number"):
        load_config(path)


def test_classification_not_a_mapping_rejected(tmp_path: Path) -> None:
    path = _minimal_config(tmp_path, extra_yaml="classification: not-a-mapping\n")
    with pytest.raises(ConfigError, match="config.classification must be a mapping"):
        load_config(path)
