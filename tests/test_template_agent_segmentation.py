"""Template segmentation on the agent/LLM path (template-mode activation).

On the agent/LLM segmentation path the baseline template must be segmented
and classified through the SAME store-backed path as the corpus documents.
The deterministic segment+classify_tree fallback relies on heading
similarity, which on a real template routinely classifies nothing — leaving
``template_std_by_tid`` empty and silently degrading a template-mode run to
emergent mode (no per-clause ``our_standard``).

SECURITY NOTE: All fixtures are programmatically constructed with synthetic
text. No real agreements are referenced. Fictional party names only.
"""

from __future__ import annotations

from pathlib import Path

import yaml

import playbook_engine.pipeline as pipeline
from playbook_engine.clause_classifier import ClassifiedClause, ClauseClassification
from playbook_engine.clause_tree import ClauseNode, ClauseTree
from playbook_engine.config import load_config
from playbook_engine.deviation_classifier import DeviationResult, RiskDelta
from playbook_engine.pipeline import _template_observations_from_classified, mine_corpus
from playbook_engine.taxonomy import load_taxonomy

_TAXONOMY_PATH = Path(__file__).parent.parent / "spec" / "taxonomy" / "affiliation-agreement.yaml"

_RTF_PROLOGUE = (
    r"{\rtf1\ansi\deff0" r"{\fonttbl{\f0\froman\fcharset0 Times New Roman;}}" r"\f0\fs24 "
)


def _write_rtf(path: Path, body: str) -> None:
    path.write_text(_RTF_PROLOGUE + body + "}", encoding="utf-8")


_CORPUS_BODY = (
    r"1. Indemnification\par "
    r"Alpha Corp shall indemnify Beta University against third-party claims "
    r"arising from the placement programme.\par "
)

_TEMPLATE_BODY = (
    r"1. Indemnification\par "
    r"The service provider shall indemnify the institution against third-party claims.\par "
)


def _make_corpus_with_template(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    corpus_dir = tmp_path / "corpus"
    deal_dir = corpus_dir / "deal-001"
    deal_dir.mkdir(parents=True)
    _write_rtf(deal_dir / "v1.rtf", _CORPUS_BODY)

    template_path = corpus_dir / "template.rtf"
    _write_rtf(template_path, _TEMPLATE_BODY)

    cfg = {
        "agreement_type": {
            "id": "educational-affiliation",
            "name": "Educational Affiliation Agreement",
            "aliases": ["eiaa"],
        },
        "baseline": {"template": str(template_path)},
        "taxonomy": str(_TAXONOMY_PATH),
        "provenance": {"our_party_aliases": ["Alpha Corp"]},
    }
    config_path = tmp_path / "playbook.config.yaml"
    config_path.write_text(yaml.dump(cfg), encoding="utf-8")
    return corpus_dir, config_path, tmp_path / "out", template_path


def _cc(path: str, tid: str | None, text: str) -> ClassifiedClause:
    node = ClauseNode(clause_path=path, heading=tid, text=text, char_span=(0, max(1, len(text))))
    cls = (
        ClauseClassification(taxonomy_id=tid, confidence=0.9, basis="llm_segmenter")
        if tid
        else ClauseClassification(taxonomy_id=None, confidence=0.0, basis="unclassified")
    )
    return ClassifiedClause(node=node, classification=cls)


class _RecordingDeviationJudge:
    """Records the our_standard each assess_batch call received."""

    def __init__(self) -> None:
        self.standards: list[str] = []

    def assess_batch(self, items: list, our_standard: str) -> list:
        self.standards.append(our_standard)
        return [
            DeviationResult(
                deviation="none",
                risk_delta=RiskDelta(direction="neutral", magnitude="none"),
                basis="judge",
                rationale="test",
            )
            for _ in items
        ]


# ---------------------------------------------------------------------------
# _template_observations_from_classified
# ---------------------------------------------------------------------------


def test_template_observations_from_classified_basic() -> None:
    classified = [
        _cc("1", "indemnification", "The provider shall indemnify the institution."),
        _cc("2", None, "Unclassified boilerplate."),
        _cc("3", "governing_law", ""),  # classified but empty text — skipped
    ]
    obs = _template_observations_from_classified(classified)
    assert len(obs) == 1
    assert obs[0].taxonomy_id == "indemnification"
    assert obs[0].citation.document_id == "template"
    assert obs[0].full_text.startswith("The provider")


# ---------------------------------------------------------------------------
# mine_corpus: agent/LLM path routes the template through _llm_segment_file
# ---------------------------------------------------------------------------


def test_template_segmented_via_llm_path(tmp_path: Path, monkeypatch) -> None:
    corpus_dir, config_path, out_dir, template_path = _make_corpus_with_template(tmp_path)
    taxonomy = load_taxonomy(_TAXONOMY_PATH)
    cfg = load_config(config_path)

    calls: list[tuple[str, str, str]] = []

    def fake_llm_segment_file(
        path: Path,
        document_id: str,
        version: str,
        taxonomy_ids: list[str],
        segment_fn,
        segmentation_cache=None,
        model: str = "test-model",
        extraction_cache=None,
        refresh_extraction: bool = False,
    ):
        calls.append((document_id, version, Path(path).name))
        # Enough clauses to pass the scope gate's "too short" heuristic. The
        # deal text must DIFFER from the template text, or the deterministic
        # matches-template fast path never escalates to the deviation judge.
        flavor = "canonical" if document_id == "template" else "negotiated"
        specs = [
            ("1", "indemnification", f"The party shall indemnify the other ({flavor} form)."),
            ("2", "governing_law", f"Governed by the laws of Delaware ({flavor} form)."),
            ("3", "term", f"The term is one year with renewal on notice ({flavor} form)."),
            ("4", "insurance", f"Liability insurance of one million ({flavor} form)."),
        ]
        nodes = [
            ClauseNode(clause_path=p, heading=h, text=t, char_span=(0, len(t))) for p, h, t in specs
        ]
        tree = ClauseTree(
            document_id=document_id,
            version=version,
            source_file=Path(path).name,
            nodes=nodes,
        )
        return tree, {p: h for p, h, _ in specs}

    monkeypatch.setattr(pipeline, "_llm_segment_file", fake_llm_segment_file)

    judge = _RecordingDeviationJudge()
    mine_corpus(
        corpus_dir=corpus_dir,
        config=cfg,
        taxonomy=taxonomy,
        out_dir=out_dir,
        use_llm_segmentation=True,
        deviation_judge=judge,
    )

    # The template went through the SAME segmentation path as the documents.
    assert ("template", "template", "template.rtf") in calls
    # And its classified clause populated our_standard for deviation judging.
    assert any(s.strip() for s in judge.standards), (
        f"no non-empty our_standard reached the deviation judge: {judge.standards!r}"
    )


def test_template_deterministic_path_unchanged(tmp_path: Path, monkeypatch) -> None:
    """Without use_llm_segmentation the template stays on the deterministic
    segment+classify_tree path — _llm_segment_file is never called."""
    corpus_dir, config_path, out_dir, _ = _make_corpus_with_template(tmp_path)
    taxonomy = load_taxonomy(_TAXONOMY_PATH)
    cfg = load_config(config_path)

    def boom(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("_llm_segment_file must not be called on the deterministic path")

    monkeypatch.setattr(pipeline, "_llm_segment_file", boom)
    mine_corpus(corpus_dir=corpus_dir, config=cfg, taxonomy=taxonomy, out_dir=out_dir)
    assert (out_dir / "observations.jsonl").exists()
