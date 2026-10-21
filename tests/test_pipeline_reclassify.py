"""A segmenter null gets a second, dedicated classification pass (issue #253).

The agent/LLM segmenter assigns ``taxonomy_id`` in the pass that cuts the
document, and the segmentation cache is keyed on the document alone. A clause
type added to the taxonomy after a document was segmented therefore never
reaches that document's text, unless the nodes the segmenter left ``null`` are
classified again under the taxonomy in force. These tests drive the real
producers end to end: ``mine_corpus`` over RTF fixtures, the production
``StoreBackedClassificationJudge`` and its ``PendingQueue``, and a verdict file
built from the queued payloads exactly as an agent answers ``judge-apply``.

SECURITY NOTE: synthetic, fictional text only.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from playbook_engine.agent_judge import (
    PendingQueue,
    StoreBackedClassificationJudge,
    VerdictStore,
)
from playbook_engine.clause_classifier import ClauseClassification
from playbook_engine.clause_tree import ClauseNode, ClauseTree
from playbook_engine.config import load_config
from playbook_engine.observation_builder import read_observations_jsonl
from playbook_engine.pipeline import _classify_llm_segmented, mine_corpus
from playbook_engine.rubric import RubricStamp
from playbook_engine.segmentation_grounding import Block, SegNode
from playbook_engine.taxonomy import load_taxonomy
from tests.test_pipeline_llm_seg import _TAXONOMY_PATH, _write_rtf

_BODY = (
    r"1. Indemnification\par "
    r"Alpha Corp shall indemnify Beta University against third-party claims "
    r"arising from the placement programme.\par "
    r"2. Compliance with Laws\par "
    r"Each side keeps to every statute that touches its own share of the placement "
    r"programme and confirms that no public health scheme has barred it.\par "
    r"3. Counterparts\par "
    r"This agreement may be signed on separate pages; the signed pages read "
    r"together make a single document.\par "
    r"4. Exhibit A\par "
    r"Exhibit A\par "
)

# What the segmenter said when these documents were segmented: it had no type
# for compliance or counterparts, and bare "Exhibit A" is not a clause.
_SEGMENTED_AS = {
    "1. Indemnification": "indemnification",
    "2. Compliance with Laws": None,
    "3. Counterparts": None,
    "4. Exhibit A": None,
}


def _segment_fn(canonical_text: str, blocks: list[Block]) -> list[SegNode]:
    del canonical_text
    nodes = []
    for n, i in enumerate(range(0, len(blocks), 2), start=1):
        head, body = blocks[i], blocks[i + 1]
        nodes.append(
            SegNode(
                node_id=f"n{n}",
                parent_id=None,
                order=n,
                heading=head.text,
                taxonomy_id=_SEGMENTED_AS[head.text],
                start_block_id=head.block_id,
                end_block_id=body.block_id,
                start_quote=head.text[:10],
                end_quote=body.text[-10:],
            )
        )
    return nodes


def _corpus(tmp_path: Path) -> tuple[Path, Path]:
    corpus = tmp_path / "corpus"
    (corpus / "deal-001").mkdir(parents=True, exist_ok=True)
    _write_rtf(corpus / "deal-001" / "v1.rtf", _BODY)
    cfg = {
        "agreement_type": {"id": "educational-affiliation", "name": "Educational Affiliation"},
        "baseline": {"template": None},
        "taxonomy": str(_TAXONOMY_PATH),
        "provenance": {"our_party_aliases": ["Alpha Corp"]},
    }
    config_path = tmp_path / "playbook.config.yaml"
    config_path.write_text(yaml.dump(cfg), encoding="utf-8")
    return corpus, config_path


def _mine(tmp_path: Path, judge: object | None, out: Path) -> list[dict]:  # type: ignore[type-arg]
    corpus, config_path = _corpus(tmp_path)
    mine_corpus(
        corpus_dir=corpus,
        config=load_config(config_path),
        taxonomy=load_taxonomy(_TAXONOMY_PATH),
        out_dir=out,
        use_llm_segmentation=True,
        llm_segment_fn=_segment_fn,
        classification_judge=judge,  # type: ignore[arg-type]
    )
    return read_observations_jsonl(out / "observations.jsonl")


def _judge(out: Path) -> tuple[StoreBackedClassificationJudge, VerdictStore]:
    store = VerdictStore(out / "judge" / "verdicts.jsonl")
    pending = PendingQueue(out / "judge" / "pending.jsonl")
    return StoreBackedClassificationJudge(store=store, pending=pending), store


def _queued(out: Path) -> list[dict]:  # type: ignore[type-arg]
    path = out / "judge" / "pending.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _basis_by_tid(rows: list[dict]) -> dict[str | None, str]:  # type: ignore[type-arg]
    return {r["taxonomy_id"]: r["x_classification_basis"] for r in rows}


def test_new_type_reaches_text_the_segmenter_left_null(tmp_path: Path) -> None:
    out = tmp_path / "out"
    judge, store = _judge(out)

    # Round 1: nothing judged yet. Only the two operative nulls are queued: the
    # segmenter's own assignment and the bare-heading node are not.
    first = _mine(tmp_path, judge, out)
    queued = _queued(out)
    assert {q["kind"] for q in queued} == {"classify"}
    assert sorted(q["payload"]["heading"] for q in queued) == [
        "2. Compliance with Laws",
        "3. Counterparts",
    ]
    assert "compliance_with_laws" in queued[0]["payload"]["taxonomy_ids"]
    assert _basis_by_tid(first)["indemnification"] == "llm_segmenter"
    assert None in _basis_by_tid(first)  # pending: still unclassified, run not blocked
    assert "compliance_with_laws" not in _basis_by_tid(first)

    # The agent answers the queue as `judge-apply` would bank it: one real
    # type, one explicit "no entry fits".
    for q in queued:
        if q["payload"]["heading"] == "2. Compliance with Laws":
            verdict = {"taxonomy_id": "compliance_with_laws", "confidence": 0.85, "basis": "judge"}
        else:
            verdict = {"taxonomy_id": None, "confidence": 0.0, "basis": "unclassified"}
        store.put_by_key(
            q["key"], verdict, rubric=RubricStamp(kind="classify", version=q["rubric_version"])
        )

    # Round 2: replayed from the store, nothing new queued.
    out2 = tmp_path / "out"
    judge2, _ = _judge(out2)
    pending_before = len(_queued(out2))
    second = _mine(tmp_path, judge2, out2)
    assert len(_queued(out2)) == pending_before

    basis = _basis_by_tid(second)
    assert basis["compliance_with_laws"] == "judge"
    assert basis["indemnification"] == "llm_segmenter"
    # The explicit null stays unclassified; it is not turned into a judge
    # verdict about nothing.
    assert basis[None] == "unclassified"


def test_no_classification_judge_leaves_segmenter_nulls_unclassified(tmp_path: Path) -> None:
    """Default (keyless) mining has no classify judge: nothing changes, nothing queues."""
    rows = _mine(tmp_path, None, tmp_path / "out")
    assert _basis_by_tid(rows) == {"indemnification": "llm_segmenter", None: "unclassified"}
    assert _queued(tmp_path / "out") == []


def test_llm_path_places_our_standards_clauses_before_asking_a_judge() -> None:
    """A segmenter null that is (nearly) a clause of our own form keeps the type
    our form gives it; only the rest reaches the judge."""
    taxonomy = load_taxonomy(_TAXONOMY_PATH)
    ours = ClauseNode(
        clause_path="1",
        heading="Lilac Card",
        text=(
            "A change to this arrangement counts only once both coordinators "
            "initial the lilac revision card."
        ),
        char_span=(0, 10),
    )
    other = ClauseNode(
        clause_path="2",
        heading="Compliance with Laws",
        text=(
            "Each side keeps to every statute that touches its own share of the "
            "work and is on no government debarment list."
        ),
        char_span=(11, 20),
    )
    tree = ClauseTree(document_id="d", version="v1", source_file="d.rtf", nodes=[ours, other])
    asked: list[str] = []

    class _Judge:
        def classify_batch(self, nodes, taxonomy, hints=None):  # type: ignore[no-untyped-def]
            asked.extend(n.clause_path or "" for n in nodes)
            return [
                ClauseClassification(
                    taxonomy_id="compliance_with_laws", confidence=0.8, basis="judge"
                )
                for _ in nodes
            ]

    out = _classify_llm_segmented(
        tree,
        {"1": None, "2": None},
        taxonomy,
        _Judge(),  # type: ignore[arg-type]
        {"amendments": ours.text},
        {e.id for e in taxonomy.classifier_entries()},
    )
    assert asked == ["2"]
    assert [(c.classification.taxonomy_id, c.classification.basis) for c in out] == [
        ("amendments", "content_similarity"),
        ("compliance_with_laws", "judge"),
    ]
