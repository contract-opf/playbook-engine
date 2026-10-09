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
from playbook_engine.extraction import ExtractorLabel
from playbook_engine.observation_builder import read_observations_jsonl
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
        extractor: str = "auto",
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
        return tree, {p: h for p, h, _ in specs}, ExtractorLabel("legacy")

    monkeypatch.setattr(pipeline, "_llm_segment_file", fake_llm_segment_file)

    mine_corpus(
        corpus_dir=corpus_dir,
        config=cfg,
        taxonomy=taxonomy,
        out_dir=out_dir,
        use_llm_segmentation=True,
    )

    # The template went through the SAME segmentation path as the documents.
    assert ("template", "template", "template.rtf") in calls
    # And its classified clauses became our standard: the template
    # observations carry their full text.
    template_obs = read_observations_jsonl(out_dir / "template_observations.jsonl")
    standards = {o["taxonomy_id"]: o["full_text"] for o in template_obs}
    assert (
        standards.get("indemnification") == "The party shall indemnify the other (canonical form)."
    ), standards
    # The negotiated documents differ from it, so none is our standard.
    deal_obs = read_observations_jsonl(out_dir / "observations.jsonl")
    assert deal_obs and all(o["standard"] is False for o in deal_obs)


def test_template_signature_block_stripped_on_llm_path(tmp_path: Path, monkeypatch) -> None:
    """#217: on the agent/LLM segmentation path in template mode, the template's
    execution block never reaches template_observations.jsonl — the source of
    every our_standard (pipeline strips it before classifying the template)."""
    corpus_dir, config_path, out_dir, template_path = _make_corpus_with_template(tmp_path)
    taxonomy = load_taxonomy(_TAXONOMY_PATH)
    cfg = load_config(config_path)

    signature = (
        "\n\nIN WITNESS WHEREOF, the parties have executed this Agreement.\n"
        "By: ______________________\nName:\nTitle:"
    )

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
        extractor: str = "auto",
    ):
        flavor = "canonical" if document_id == "template" else "negotiated"
        specs = [
            ("1", "indemnification", f"The party shall indemnify the other ({flavor} form)."),
            ("2", "governing_law", f"Governed by the laws of Delaware ({flavor} form)."),
            ("3", "term", f"The term is one year with renewal on notice ({flavor} form)."),
            ("4", "insurance", f"Liability insurance of one million ({flavor} form)."),
        ]
        if document_id == "template":
            p_, h_, t_ = specs[-1]
            specs[-1] = (p_, h_, t_ + signature)
        nodes = [
            ClauseNode(clause_path=p, heading=h, text=t, char_span=(0, len(t))) for p, h, t in specs
        ]
        tree = ClauseTree(
            document_id=document_id,
            version=version,
            source_file=Path(path).name,
            nodes=nodes,
        )
        return tree, {p: h for p, h, _ in specs}, ExtractorLabel("legacy")

    monkeypatch.setattr(pipeline, "_llm_segment_file", fake_llm_segment_file)

    mine_corpus(
        corpus_dir=corpus_dir,
        config=cfg,
        taxonomy=taxonomy,
        out_dir=out_dir,
        use_llm_segmentation=True,
    )

    template_obs = (out_dir / "template_observations.jsonl").read_text(encoding="utf-8")
    assert "insurance" in template_obs, "the template's last clause must still be observed"
    for marker in ("IN WITNESS WHEREOF", "By:"):
        assert marker not in template_obs


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


# ---------------------------------------------------------------------------
# Issue #235: content similarity on the LLM path; never on the template
# ---------------------------------------------------------------------------

_INSURANCE_TEMPLATE = (
    "Each party shall maintain commercial general liability insurance coverage "
    "of at least one million dollars per occurrence throughout the term."
)
_INSURANCE_DEAL = (
    "Each party shall maintain commercial general liability insurance coverage "
    "of at least two million dollars per occurrence throughout the term."
)


def _fake_llm_segment_file_with_null_node(path, document_id, version, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
    """Template: four classified clauses. Deal: three classified clauses plus an
    insurance clause the LLM left unclassified (explicit null taxonomy)."""
    is_template = document_id == "template"
    flavor = "canonical" if is_template else "negotiated"
    specs: list[tuple[str, str | None, str]] = [
        ("1", "indemnification", f"The party shall indemnify the other ({flavor} form)."),
        ("2", "governing_law", f"Governed by the laws of Delaware ({flavor} form)."),
        ("3", "term", f"The term is one year with renewal on notice ({flavor} form)."),
        (
            "4",
            "insurance" if is_template else None,
            _INSURANCE_TEMPLATE if is_template else _INSURANCE_DEAL,
        ),
    ]
    nodes = [
        ClauseNode(clause_path=p, heading=h, text=t, char_span=(0, len(t))) for p, h, t in specs
    ]
    tree = ClauseTree(
        document_id=document_id, version=version, source_file=Path(path).name, nodes=nodes
    )
    return tree, {p: h for p, h, _ in specs}, ExtractorLabel("legacy")


def test_content_similarity_applies_on_the_llm_path(tmp_path: Path, monkeypatch) -> None:
    """Direction 3: the fallback runs on the LLM/agent segmentation path too. A
    deal clause the LLM left null, whose text matches a template standard, is
    assigned by content; a clause the LLM did classify keeps ``llm_segmenter``."""
    corpus_dir, config_path, out_dir, _ = _make_corpus_with_template(tmp_path)
    monkeypatch.setattr(pipeline, "_llm_segment_file", _fake_llm_segment_file_with_null_node)

    mine_corpus(
        corpus_dir=corpus_dir,
        config=load_config(config_path),
        taxonomy=load_taxonomy(_TAXONOMY_PATH),
        out_dir=out_dir,
        use_llm_segmentation=True,
    )

    deal_obs = read_observations_jsonl(out_dir / "observations.jsonl")
    by_tid = {o["taxonomy_id"]: o for o in deal_obs}
    assert by_tid["insurance"]["x_classification_basis"] == "content_similarity", by_tid
    assert by_tid["insurance"]["citation"]["clause_path"] == "4"
    for tid in ("indemnification", "governing_law", "term"):
        assert by_tid[tid]["x_classification_basis"] == "llm_segmenter", (tid, by_tid[tid])


def _spy_classification(monkeypatch) -> tuple[list[dict], list[list[str]]]:  # noqa: ANN001
    """Record every pipeline.classify_tree call (tree id + content_exemplars) and
    every pipeline.assign_by_content call (the clause texts it was handed)."""
    tree_calls: list[dict] = []
    content_calls: list[list[str]] = []
    real_classify, real_assign = pipeline.classify_tree, pipeline.assign_by_content

    def spy_classify(tree, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        tree_calls.append(
            {"document_id": tree.document_id, "content_exemplars": kwargs.get("content_exemplars")}
        )
        return real_classify(tree, *args, **kwargs)

    def spy_assign(classified, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        content_calls.append([cc.node.text or "" for cc in classified])
        return real_assign(classified, *args, **kwargs)

    monkeypatch.setattr(pipeline, "classify_tree", spy_classify)
    monkeypatch.setattr(pipeline, "assign_by_content", spy_assign)
    return tree_calls, content_calls


def test_template_is_never_content_classified_deterministic_path(
    tmp_path: Path, monkeypatch
) -> None:
    """The template is classified WITHOUT exemplars (they come from its own
    classification) and never goes through ``assign_by_content``; the deal's
    classification, by contrast, is given the template's exemplars."""
    corpus_dir, config_path, out_dir, template_path = _make_corpus_with_template(tmp_path)
    # Two headed clauses each, so the deal clears the scope gate's minimum
    # clause count and actually reaches L3 classification.
    _write_rtf(
        template_path,
        _TEMPLATE_BODY + r"2. Governing Law\par This Agreement is governed by Delaware law.\par ",
    )
    _write_rtf(
        corpus_dir / "deal-001" / "v1.rtf",
        _CORPUS_BODY + r"2. Governing Law\par This Agreement is governed by New York law.\par ",
    )
    tree_calls, content_calls = _spy_classification(monkeypatch)

    mine_corpus(
        corpus_dir=corpus_dir,
        config=load_config(config_path),
        taxonomy=load_taxonomy(_TAXONOMY_PATH),
        out_dir=out_dir,
    )

    template_calls = [c for c in tree_calls if c["document_id"] == "template"]
    deal_calls = [c for c in tree_calls if c["document_id"] != "template"]
    assert template_calls, "the template must be classified through classify_tree"
    assert all(not c["content_exemplars"] for c in template_calls), template_calls
    # Positive control: the spy sees the deal receiving the template's standards.
    assert deal_calls and all(c["content_exemplars"] for c in deal_calls), tree_calls
    assert not content_calls


def test_template_is_never_content_classified_llm_path(tmp_path: Path, monkeypatch) -> None:
    """On the LLM path the template's classification is the LLM's own, and no
    template clause text reaches ``assign_by_content`` (only the deal's does)."""
    corpus_dir, config_path, out_dir, _ = _make_corpus_with_template(tmp_path)
    monkeypatch.setattr(pipeline, "_llm_segment_file", _fake_llm_segment_file_with_null_node)
    tree_calls, content_calls = _spy_classification(monkeypatch)

    mine_corpus(
        corpus_dir=corpus_dir,
        config=load_config(config_path),
        taxonomy=load_taxonomy(_TAXONOMY_PATH),
        out_dir=out_dir,
        use_llm_segmentation=True,
    )

    assert not tree_calls  # LLM path bypasses classify_tree for template and deal alike
    assert content_calls, "positive control: the deal's classification reaches assign_by_content"
    for texts in content_calls:
        assert not any("canonical form" in t or t == _INSURANCE_TEMPLATE for t in texts), texts
