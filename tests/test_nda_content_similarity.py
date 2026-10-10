"""Ground truth for the keyless content-similarity classifier (issue #235).

Counterparty forms use their own headings, so the heading paths leave their
clauses unclassified and they never reach precedent. L3's content-similarity
fallback (``clause_classifier.assign_by_content``) compares each such node
with our standard's own clause text per taxonomy_id. This pins its behaviour
on the six synthetic NDA deals (four on our paper, two on counterparty paper;
see examples/nda/README.md) through the real CLI, hermetic and keyless:

- every "must be assigned" clause of the two counterparty-paper deals
  (theta-logistics, zeta-diagnostics) gets its taxonomy_id under basis
  ``content_similarity`` in the terminal version;
- no node of any version of any deal is assigned a taxonomy_id outside its
  allowed outcomes (the table below);
- our-paper behaviour is unchanged: no already-classified node moves, and the
  uncovered our-paper clauses stay unclassified.

Every fixture here is produced by the production path: ``playbook mine`` over
the committed corpus writes ``normalized/<doc>/<version>.clauses.json`` (the
trees L3 classifies) and ``template_observations.jsonl`` (the per-node
standards the pipeline joins into exemplars); the classification below calls
``classify_tree`` exactly as ``pipeline._compute_doc_result_l2l4`` does.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pytest
from click.testing import CliRunner

from playbook_engine.clause_classifier import AMBIGUITY_THRESHOLD, classify_tree
from playbook_engine.clause_tree import ClauseTree
from playbook_engine.cli import cli
from playbook_engine.pipeline import _NullClassificationJudge
from playbook_engine.taxonomy import load_taxonomy

_REPO_ROOT = Path(__file__).resolve().parent.parent
_NDA_DIR = _REPO_ROOT / "examples" / "nda"
_CORPUS_DIR = _NDA_DIR / "corpus"
_SMOKE_CONFIG = _NDA_DIR / "config.smoke.yaml"
_TAXONOMY_PATH = _REPO_ROOT / "spec" / "taxonomy" / "nda.yaml"

_THIRD_PARTY_DEALS = ("theta-logistics", "zeta-diagnostics")

# (text begins, taxonomy_id | None for "stays unclassified", must be assigned)
# for the counterparty-paper deals' fifteen unclassified nodes, in their order.
_THIRD_PARTY_TABLE: list[tuple[str, str | None, bool]] = [
    ("DocuSign Envelope ID:", None, False),
    ("This Confidentiality Agreement is made between", "parties_and_recitals", False),
    ('"Confidential Information" means all information', "definition_confidential_info", False),
    (
        "The obligations in this Agreement do not apply to information that",
        "exclusions_from_confidential",
        True,
    ),
    (
        "The receiving party shall use Confidential Information only to evaluate",
        "purpose_permitted_use",
        False,
    ),
    (
        "The receiving party shall protect Confidential Information using",
        "standard_of_care",
        True,
    ),
    (
        "The receiving party may disclose Confidential Information to its employees",
        "permitted_disclosure_reps",
        False,
    ),
    (
        "The receiving party may disclose Confidential Information to the extent required",
        "compelled_disclosure",
        True,
    ),
    ("The confidentiality obligations in this Agreement continue for", "survival_period", True),
    (
        "On written request, the receiving party shall promptly return or destroy",
        "return_or_destruction",
        False,
    ),
    (
        "Nothing in this Agreement grants any license or right under any intellectual property",
        "no_license_ip_ownership",
        False,
    ),
    (
        "The parties agree that a breach of this Agreement may cause irreparable",
        "injunctive_relief",
        True,
    ),
    (
        "This Agreement does not obligate either party to enter into any further",
        "no_obligation_to_proceed",
        False,
    ),
    (
        "The parties submit to the exclusive jurisdiction of the state and federal",
        "dispute_resolution_venue",
        True,
    ),
    ("This Agreement constitutes the entire agreement", "entire_agreement_amendment", True),
]

# The our-paper deals' unclassified nodes: a wrong type here is a false
# precedent, so only the listed outcomes are allowed (None == unclassified).
_OUR_PAPER_TABLE: list[tuple[str, set[str | None]]] = [
    ("DocuSign Envelope ID:", {None}),
    ("If Confidential Information includes personal data", {None}),
    ("For a period of twelve (12) months", {None}),
    ("For a period of six (6) months", {None}),
    ("With respect to information that constitutes a trade secret", {"survival_period", None}),
    (
        "Nothing in this Agreement grants Recipient the right to use",
        {"purpose_permitted_use", None},
    ),
    ("No license or other right in any patent", {"no_license_ip_ownership", None}),
]


def _allowed_outcomes(text: str, deal: str) -> set[str | None] | None:
    """The allowed taxonomy_ids (``None`` == unclassified) for a node's text, or
    ``None`` when the node is not one of the table's unclassified clauses."""
    for prefix, tid, _must in _THIRD_PARTY_TABLE if deal in _THIRD_PARTY_DEALS else []:
        if text.startswith(prefix):
            return {tid, None}
    for prefix, outcomes in _OUR_PAPER_TABLE if deal not in _THIRD_PARTY_DEALS else []:
        if text.startswith(prefix):
            return outcomes
    return None


@pytest.fixture(scope="module")
def mined(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out_dir = tmp_path_factory.mktemp("nda-content") / "out"
    result = CliRunner().invoke(
        cli, ["mine", str(_CORPUS_DIR), "--config", str(_SMOKE_CONFIG), "--out", str(out_dir)]
    )
    assert result.exit_code == 0, result.output
    # Issue #235: `playbook mine` prints one line of classification coverage by
    # basis and the same counts land in run_manifest.json.
    coverage_lines = [
        line for line in result.output.splitlines() if line.startswith("classification:")
    ]
    assert len(coverage_lines) == 1, result.output
    for basis in (
        "exact_match",
        "heading_similarity",
        "judge",
        "inherited",
        "content_similarity",
        "unclassified",
    ):
        assert f"{basis} " in coverage_lines[0], coverage_lines[0]
    manifest = json.loads((out_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["classification_coverage"]["content_similarity"] == 14
    # The coverage counts clause observations: the "opening" rows (issue #233)
    # restate a first-draft clause and are left out.
    assert sum(manifest["classification_coverage"].values()) == len(_observations(out_dir))
    return out_dir


def _observations(out_dir: Path, *, openings: bool = False) -> list[dict]:
    """The mined observations; the ``opening`` rows (issue #233), which restate
    a first-draft clause already classified through its own row, only on request."""
    rows = [
        json.loads(line)
        for line in (out_dir / "observations.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    return rows if openings else [o for o in rows if o["outcome"] != "opening"]


def _exemplars(out_dir: Path) -> dict[str, str]:
    """The pipeline's content exemplars: every template node joined per
    taxonomy_id (the synthetic NDA template has no form front matter, so this
    is its standard)."""
    nodes: dict[str, list[str]] = defaultdict(list)
    for line in (out_dir / "template_observations.jsonl").read_text(encoding="utf-8").splitlines():
        obs = json.loads(line)
        if obs["taxonomy_id"]:
            nodes[obs["taxonomy_id"]].append(obs["full_text"])
    return {tid: "\n".join(texts) for tid, texts in nodes.items()}


def test_every_must_assign_clause_is_assigned_in_the_terminal_version(mined: Path) -> None:
    by_deal: dict[str, dict[str, dict]] = defaultdict(dict)
    for obs in _observations(mined):
        deal = obs["citation"]["document_id"]
        if obs["taxonomy_id"]:
            by_deal[deal].setdefault(obs["taxonomy_id"], obs)
    for deal in _THIRD_PARTY_DEALS:
        for prefix, tid, must in _THIRD_PARTY_TABLE:
            if not must:
                continue
            assert tid is not None
            obs = by_deal[deal].get(tid)
            assert obs is not None, f"{deal}: {tid} not assigned"
            assert obs["x_classification_basis"] == "content_similarity", (deal, tid)
            assert obs["full_text"].startswith(prefix), (deal, tid, obs["full_text"][:60])
            assert obs["confidence"] < AMBIGUITY_THRESHOLD, (deal, tid)


def test_no_node_is_assigned_outside_its_allowed_outcomes(mined: Path) -> None:
    """Re-classify every version of every deal exactly as L3 does (no judge)."""
    taxonomy = load_taxonomy(_TAXONOMY_PATH)
    exemplars = _exemplars(mined)
    assert exemplars, "template mode produced no standards"
    assigned = 0
    for tree_path in sorted((mined / "normalized").rglob("*.clauses.json")):
        deal = tree_path.parent.name
        tree = ClauseTree.load(tree_path)
        for cc in classify_tree(
            tree, taxonomy, _NullClassificationJudge(), content_exemplars=exemplars
        ):
            cls = cc.classification
            if cls.basis != "content_similarity":
                continue
            assigned += 1
            allowed = _allowed_outcomes(cc.node.text, deal)
            assert allowed is not None, (
                f"{deal}/{tree_path.name}: content_similarity assigned a node outside "
                f"the ground-truth table: {cc.node.text[:60]!r} -> {cls.taxonomy_id}"
            )
            assert cls.taxonomy_id in allowed, (deal, tree_path.name, cc.node.text[:60])
    # Seven must-assign clauses per counterparty-paper deal in its terminal
    # version, plus whatever the table allows in earlier versions.
    assert assigned >= 2 * 7


def test_our_paper_deals_are_unchanged(mined: Path) -> None:
    """No our-paper node is assigned by content, and the uncovered our-paper
    clauses (DocuSign line, personal-data clause, non-solicits) stay
    unclassified in the observations."""
    for obs in _observations(mined):
        deal = obs["citation"]["document_id"]
        if deal in _THIRD_PARTY_DEALS:
            continue
        assert obs["x_classification_basis"] != "content_similarity", (deal, obs["taxonomy_id"])
        for prefix, outcomes in _OUR_PAPER_TABLE:
            if obs["full_text"].startswith(prefix):
                assert obs["taxonomy_id"] in outcomes, (deal, prefix, obs["taxonomy_id"])
