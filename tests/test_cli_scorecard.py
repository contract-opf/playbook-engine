"""`playbook scorecard` — the counts-only scorecard of a derivation out-dir (issue #237).

The NDA tests drive the real `mine` -> `project` pipeline over the committed
synthetic corpus (``examples/nda/config.smoke.yaml``, hermetic and keyless)
and score the out-dir it writes, so every artifact the scorecard reads is
the one production writes.

Two kinds of hand-built input appear below, and neither is a shape the
pipeline writes today:

- the *hostile* out-dir puts corpus-identifying strings where the scorecard
  reads labels (a dropped-observation reason, a pending-queue kind, a
  classification basis, a paper side, an OPF version). Production never
  writes these; the test exists to prove a corrupted or hand-edited out-dir
  still cannot leak them through the scorecard.
- the *OPF 0.5* case adds ``opened_with`` to precedent records (issue #233)
  and a ``dossiers`` map (issue #228) to a copy of the real NDA playbook.
  Nothing in the engine emits either yet; the case pins that the scorecard
  counts them once they exist, and reports ``null`` until then. It follows
  #233's rule: ``opening_text`` is non-null only when ``opened_with`` is
  ``standard`` or ``non_standard``.
- the *no-manifest* out-dir has a playbook but no ``corpus_manifest.json``,
  so the corpus counts fall back to ``corpus.stats``; strings, lists and
  dicts planted there prove that fallback is type-guarded too.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner
from click.testing import Result as CliResult

from playbook_engine.agent_judge import PendingQueue, VerdictStore
from playbook_engine.agent_segmenter import (
    AGENT_SEGMENTER_MODEL,
    AgentSegmentationPending,
    StoreBackedSegmentFn,
)
from playbook_engine.clause_classifier import ClauseClassification
from playbook_engine.cli import cli
from playbook_engine.digest import build_digest
from playbook_engine.llm_segmenter_batch import SegmentationVerdictCache
from playbook_engine.observation_builder import Observation, ObservationCitation
from playbook_engine.pipeline import _observation_classification_basis
from playbook_engine.rubric import RubricStamp, rubric_version
from playbook_engine.scorecard import (
    ENUM_LABELS,
    build_scorecard,
    compare_scorecards,
    flatten_scorecard,
    pending_queue_counts,
)

_ROOT = Path(__file__).resolve().parent.parent
_NDA_DIR = _ROOT / "examples" / "nda"
_CORPUS_DIR = _NDA_DIR / "corpus"
_SMOKE_CONFIG = _NDA_DIR / "config.smoke.yaml"

#: The scorecard's shape: every field name it may carry. Pinned here (not
#: imported) so a field added or renamed in the module fails this test.
_FIELD_NAMES = frozenset(
    {
        "scorecard_version",
        "opf_version",
        "corpus",
        "documents",
        "versions",
        "signed",
        "in_scope",
        "quarantined",
        "template",
        "has_canonical_template",
        "standards_classified",
        "clauses",
        "clauses_with_our_standard",
        "our_standard_coverage",
        "classification",
        "observations",
        "classified",
        "classified_share",
        "by_basis",
        "deals",
        "distinct_types_per_deal",
        "n",
        "mean",
        "median",
        "min",
        "max",
        "by_paper",
        "precedent",
        "records",
        "signed_standard",
        "signed_variants",
        "refused_asks",
        "openings",
        "openings_by_opened_with",
        "records_by_opened_with",
        "signed_opened_with_undetermined",
        "template_drift",
        "opening_drift",
        "below_half",
        "over_one_third_below_half",
        "shares",
        "dropped_observations",
        "count",
        "by_reason",
        "digest",
        "digest_version",
        "token_estimate",
        "capped_clauses",
        "signed_variant_groups",
        "refused_ask_groups",
        "queues",
        "judge_pending",
        "judge_pending_by_kind",
        "segment_pending",
        "dossiers",
        "max_tokens",
        "median_tokens",
        "equivalence",
        "by_role",
        "totals",
        "by_label",
        "agreement_rate",
        "checker_models",
        "eligible",
        "unjudged",
        "drafted",
        "checked",
        "agreed",
        "adjudicated",
        "disputed",
        "unchecked",
    }
)

_TOP_LEVEL = {
    "scorecard_version",
    "opf_version",
    "corpus",
    "template",
    "classification",
    "precedent",
    "template_drift",
    "opening_drift",
    "dropped_observations",
    "digest",
    "equivalence",
    "queues",
    "dossiers",
}


#: Every dotted leaf path of the NDA example's scorecard (``flatten_scorecard``).
#: Data-dependent labels (bases, paper sides, dropped reasons) are part of it
#: because the NDA run is deterministic; a field dropped or renamed fails here.
_NDA_FLAT_KEYS = frozenset(
    {
        "classification.by_basis.content_similarity",
        "classification.by_basis.exact_match",
        "classification.by_basis.heading_similarity",
        "classification.by_basis.unclassified",
        # Issue #235: the two counterparty-paper deals (paper "unknown" in this
        # stub run) are the only ones whose clauses are assigned by content.
        "classification.by_paper.unknown.by_basis.content_similarity",
        *(
            f"classification.by_paper.{paper}.{field}"
            for paper in ("our_paper", "unknown")
            for field in (
                "by_basis.exact_match",
                "by_basis.heading_similarity",
                "by_basis.unclassified",
                "classified",
                "classified_share",
                "deals",
                "distinct_types_per_deal.max",
                "distinct_types_per_deal.mean",
                "distinct_types_per_deal.median",
                "distinct_types_per_deal.min",
                "distinct_types_per_deal.n",
                "observations",
            )
        ),
        "classification.classified",
        "classification.classified_share",
        "classification.deals",
        "classification.distinct_types_per_deal.max",
        "classification.distinct_types_per_deal.mean",
        "classification.distinct_types_per_deal.median",
        "classification.distinct_types_per_deal.min",
        "classification.distinct_types_per_deal.n",
        "classification.observations",
        "corpus.documents",
        "corpus.in_scope",
        "corpus.quarantined",
        "corpus.signed",
        "corpus.versions",
        "digest.capped_clauses",
        "digest.clauses",
        "digest.digest_version",
        "digest.refused_ask_groups",
        "digest.signed_variant_groups",
        "digest.token_estimate",
        "dossiers",
        "dropped_observations.by_reason.removed_origin_undetermined",
        # Issue #240: no verdict store in this run, so every eligible text is
        # unjudged; the section is present (all zero) either way.
        "equivalence.agreement_rate",
        "equivalence.checker_models",
        *(
            f"equivalence.by_label.{label}"
            for label in (
                "equivalent",
                "more_protective",
                "less_protective",
                "different_concept",
            )
        ),
        *(
            f"equivalence.{scope}.{field}"
            for scope in ("totals", "by_role.signed", "by_role.opening", "by_role.refused")
            for field in (
                "eligible",
                "unjudged",
                "drafted",
                "checked",
                "agreed",
                "adjudicated",
                "disputed",
                "unchecked",
            )
        ),
        "dropped_observations.by_reason.survives_in_terminal",
        "dropped_observations.count",
        "opf_version",
        "precedent.openings",
        "precedent.openings_by_opened_with.non_standard",
        "precedent.openings_by_opened_with.standard",
        "precedent.records",
        "precedent.records_by_opened_with.absent",
        "precedent.records_by_opened_with.non_standard",
        "precedent.records_by_opened_with.standard",
        "precedent.refused_asks",
        "precedent.signed",
        "precedent.signed_opened_with_undetermined",
        "precedent.signed_standard",
        "precedent.signed_variants",
        "queues.judge_pending",
        "queues.judge_pending_by_kind",
        "queues.segment_pending",
        "scorecard_version",
        "template.clauses",
        "template.clauses_with_our_standard",
        "template.has_canonical_template",
        "template.our_standard_coverage",
        "template.standards_classified",
        *(
            f"{section}.{field}"
            for section in ("template_drift", "opening_drift")
            for field in (
                "below_half",
                "clauses",
                "max",
                "median",
                "min",
                "over_one_third_below_half",
                "shares[]",
            )
        ),
    }
)


def _invoke(args: list[str]) -> CliResult:
    return CliRunner().invoke(cli, args)


@pytest.fixture(scope="module")
def nda_out(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A real `mine` -> `project` out-dir over the NDA example."""
    out_dir = tmp_path_factory.mktemp("nda") / "out"
    mine = _invoke(
        ["mine", str(_CORPUS_DIR), "--config", str(_SMOKE_CONFIG), "--out", str(out_dir)]
    )
    assert mine.exit_code == 0, mine.output
    project = _invoke(["project", str(out_dir), "--config", str(_SMOKE_CONFIG)])
    assert project.exit_code == 0, project.output
    return out_dir


def _copy_out(src: Path, dest: Path) -> Path:
    shutil.copytree(src, dest)
    return dest


def _walk(node: Any, keys: list[str], strings: list[str]) -> None:
    if isinstance(node, dict):
        for k, v in node.items():
            keys.append(k)
            _walk(v, keys, strings)
    elif isinstance(node, list):
        for v in node:
            _walk(v, keys, strings)
    elif isinstance(node, str):
        strings.append(node)


def _assert_counts_only(card: dict[str, Any]) -> None:
    """The leak guard: keys are field names or enum labels; strings are enum labels."""
    keys: list[str] = []
    strings: list[str] = []
    _walk(card, keys, strings)
    stray_keys = sorted({k for k in keys if k not in _FIELD_NAMES and k not in ENUM_LABELS})
    assert stray_keys == [], f"non-field, non-label keys in scorecard: {stray_keys}"
    stray_values = sorted({s for s in strings if s not in ENUM_LABELS})
    assert stray_values == [], f"non-label string values in scorecard: {stray_values}"


def _corpus_strings(out_dir: Path) -> set[str]:
    """Document ids, clause types, file names and clause text of the out-dir."""
    found: set[str] = set()
    for line in (out_dir / "observations.jsonl").read_text(encoding="utf-8").splitlines():
        obs = json.loads(line)
        found.add(obs["citation"]["document_id"])
        if obs.get("taxonomy_id"):
            found.add(obs["taxonomy_id"])
        if len(obs.get("full_text") or "") >= 25:
            found.add(obs["full_text"][:25])
    found.update(p.name for p in _CORPUS_DIR.rglob("*.rtf"))
    return found


# ---------------------------------------------------------------------------
# NDA shape + leak guard
# ---------------------------------------------------------------------------


def test_scorecard_on_nda_writes_the_pinned_shape(nda_out: Path, tmp_path: Path) -> None:
    out = _copy_out(nda_out, tmp_path / "out")
    result = _invoke(["scorecard", str(out)])
    assert result.exit_code == 0, result.output
    card = json.loads((out / "scorecard.json").read_text(encoding="utf-8"))
    assert set(card) == _TOP_LEVEL
    assert set(flatten_scorecard(card)) == _NDA_FLAT_KEYS
    assert card["scorecard_version"] == 4
    assert card["opf_version"] == "0.5"
    assert card["corpus"] == {
        "documents": 6,
        "versions": 17,
        "signed": 6,
        "in_scope": 6,
        "quarantined": 0,
    }
    template = card["template"]
    assert template["has_canonical_template"] is True
    assert template["standards_classified"] > 0
    assert 0 < template["clauses_with_our_standard"] <= template["clauses"]
    assert template["our_standard_coverage"] == round(
        template["clauses_with_our_standard"] / template["clauses"], 4
    )

    cls = card["classification"]
    assert cls["observations"] > cls["classified"] > 0
    assert cls["deals"] == 6
    # Every NDA observation carries x_classification_basis (pipeline, #237),
    # so nothing is "unrecorded" and the bases add up to the observations.
    assert sum(cls["by_basis"].values()) == cls["observations"]
    assert "unrecorded" not in cls["by_basis"]
    assert cls["by_basis"]["unclassified"] == cls["observations"] - cls["classified"]
    # Paper side is a diagnostic split that partitions the same observations.
    assert set(cls["by_paper"]) <= {"our_paper", "counterparty_paper", "unknown"}
    assert sum(b["observations"] for b in cls["by_paper"].values()) == cls["observations"]
    assert sum(b["deals"] for b in cls["by_paper"].values()) == cls["deals"]

    precedent = card["precedent"]
    playbook = json.loads((out / "playbook.opf.json").read_text(encoding="utf-8"))
    assert precedent["records"] == len(playbook["evidence"]["precedent"])
    assert precedent["signed_standard"] + precedent["signed_variants"] <= precedent["signed"]
    assert (
        card["dropped_observations"]["count"]
        == (playbook["corpus"]["stats"]["dropped_observations"]["count"])
    )
    assert card["digest"]["digest_version"] == "3"
    assert card["digest"]["token_estimate"] > 0
    assert card["digest"]["clauses"] == len(playbook["digest"]["clauses"])
    digest_clauses = playbook["digest"]["clauses"]
    assert card["digest"]["capped_clauses"] == 0
    assert card["digest"]["signed_variant_groups"] == sum(
        c.get("n_variants_total", 0) for c in digest_clauses
    )
    assert card["digest"]["refused_ask_groups"] == sum(
        c.get("n_refused_total", 0) for c in digest_clauses
    )
    assert precedent["openings"] == sum(
        1 for r in playbook["evidence"]["precedent"] if r.get("opening_text") is not None
    )
    assert card["queues"] == {"judge_pending": 0, "judge_pending_by_kind": {}, "segment_pending": 0}
    drift = card["template_drift"]
    assert drift["clauses"] == len(drift["shares"]) > 0
    assert drift["shares"] == sorted(drift["shares"])
    assert all(0.0 <= s <= 1.0 for s in drift["shares"])
    assert drift["below_half"] == sum(1 for s in drift["shares"] if s < 0.5)
    per_deal = cls["distinct_types_per_deal"]
    assert per_deal["n"] == cls["deals"]
    assert per_deal["min"] <= per_deal["median"] <= per_deal["max"]

    # OPF 0.5 carries opened_with (#233) but not yet dossiers (#228): null, not an error.
    assert sum(precedent["records_by_opened_with"].values()) == precedent["records"]
    assert sum(precedent["openings_by_opened_with"].values()) == precedent["openings"] > 0
    assert set(precedent["openings_by_opened_with"]) <= {"standard", "non_standard"}
    # Every signed record of a fresh store has its opening determined (#233).
    assert precedent["signed_opened_with_undetermined"] == 0
    assert card["dossiers"] is None
    # The opening drift is the template drift's twin over `opened_with`.
    opening = card["opening_drift"]
    assert opening["clauses"] == drift["clauses"]
    assert opening["shares"] == sorted(opening["shares"])
    assert all(0.0 <= s <= 1.0 for s in opening["shares"])
    assert opening["below_half"] == sum(1 for s in opening["shares"] if s < 0.5)

    # The printed table carries the same numbers.
    assert "corpus.documents" in result.output
    assert "template.standards_classified" in result.output


def test_scorecard_on_nda_is_counts_only(nda_out: Path, tmp_path: Path) -> None:
    out = _copy_out(nda_out, tmp_path / "out")
    result = _invoke(["scorecard", str(out)])
    assert result.exit_code == 0, result.output
    card = json.loads((out / "scorecard.json").read_text(encoding="utf-8"))
    _assert_counts_only(card)
    keys: list[str] = []
    strings: list[str] = []
    _walk(card, keys, strings)
    tokens = set(keys) | set(strings)
    # Table rows are dotted field paths: split them back into their parts.
    for line in result.output.split("OK  ")[0].splitlines():
        for cell in line.split():
            tokens.update(cell.split("."))
    text = (out / "scorecard.json").read_text(encoding="utf-8") + result.output.split("OK  ")[0]
    corpus = _corpus_strings(out)
    # Exact for every string (a clause type like "term" is a substring of the
    # enum label "survives_in_terminal", not a leak); substring for long ones.
    leaked = sorted(s for s in corpus if s in tokens or (len(s) >= 12 and s in text))
    assert leaked == [], f"corpus strings leaked into the scorecard: {leaked}"


def test_hostile_out_dir_cannot_leak_through_labels(tmp_path: Path) -> None:
    """Corpus strings planted where the scorecard reads labels come out as "other"."""
    secret = "Acme University Hospital"
    out = tmp_path / "out"
    (out / "judge").mkdir(parents=True)
    (out / "segment").mkdir()
    playbook = {
        "opf_version": secret,
        "baseline": {"has_canonical_template": True},
        "evidence": {
            "clauses": [
                {"id": "clause.x", "taxonomy_id": secret, "our_standard": {"text": secret}}
            ],
            "precedent": [
                {
                    "id": "prec.1",
                    "taxonomy_id": secret,
                    "document_id": secret,
                    "paper": "ours",
                    "signed": True,
                    "standard": False,
                    "signed_text": {"text": secret},
                    "opening_text": {"text": secret},
                    "opened_with": secret,
                    "refused_asks": [{"text": secret}],
                }
            ],
        },
        "corpus": {
            "stats": {"dropped_observations": {"count": 3, "by_reason": {secret: 3}}},
            "documents": [{"document_id": secret, "signed_version": 1}],
        },
        "digest": {"digest_version": secret, "clauses": [{"id": secret, "title": secret}]},
        "dossiers": {secret: {"rationale": secret}},
    }
    (out / "playbook.opf.json").write_text(json.dumps(playbook), encoding="utf-8")
    obs = {
        "taxonomy_id": secret,
        "full_text": secret,
        "provenance": secret,
        "x_classification_basis": secret,
        "citation": {"document_id": secret},
    }
    (out / "observations.jsonl").write_text(json.dumps(obs) + "\n", encoding="utf-8")
    (out / "corpus_manifest.json").write_text(
        json.dumps([{"document_id": secret, "versions": 2, "in_scope": True}]), encoding="utf-8"
    )
    (out / "judge" / "pending.jsonl").write_text(
        json.dumps({"key": "k1", "kind": secret, "payload": {"text": secret}}) + "\n",
        encoding="utf-8",
    )
    (out / "segment" / "pending.jsonl").write_text(
        json.dumps({"key": "k2", "kind": "segment", "payload": {"text": secret}}) + "\n",
        encoding="utf-8",
    )

    result = _invoke(["scorecard", str(out)])
    assert result.exit_code == 0, result.output
    card = json.loads((out / "scorecard.json").read_text(encoding="utf-8"))
    _assert_counts_only(card)
    assert secret not in (out / "scorecard.json").read_text(encoding="utf-8")
    assert secret not in result.output
    assert card["opf_version"] == "other"
    assert card["dropped_observations"]["by_reason"] == {"other": 3}
    assert card["queues"]["judge_pending_by_kind"] == {"other": 1}
    assert card["queues"]["segment_pending"] == 1
    assert card["classification"]["by_basis"] == {"other": 1}
    assert set(card["classification"]["by_paper"]) == {"other"}
    assert card["precedent"]["openings"] == 1
    assert card["precedent"]["openings_by_opened_with"] == {"other": 1}
    assert card["precedent"]["records_by_opened_with"] == {"other": 1}
    assert card["digest"]["digest_version"] == "other"
    assert card["dossiers"]["count"] == 1


_SECRET = "Acme University Hospital"


@pytest.mark.parametrize(
    ("documents_total", "versions_total", "documents_in_scope", "expected"),
    [
        pytest.param(
            _SECRET,
            {_SECRET: _SECRET},
            [_SECRET],
            (None, None, None),
            id="string-dict-list",
        ),
        pytest.param(
            {_SECRET: 3},
            [_SECRET, 2],
            _SECRET,
            (None, None, None),
            id="dict-list-string",
        ),
        pytest.param(
            [{_SECRET: _SECRET}],
            _SECRET,
            {"by_document": {_SECRET: True}},
            (None, None, None),
            id="list-string-dict",
        ),
        pytest.param(True, 2.5, False, (None, None, None), id="bool-float-bool"),
        pytest.param(4, 9, 3, (4, 9, 3), id="counts-pass-through"),
    ],
)
def test_corpus_stats_fallback_without_a_manifest_passes_only_counts(
    tmp_path: Path,
    documents_total: Any,
    versions_total: Any,
    documents_in_scope: Any,
    expected: tuple[int | None, int | None, int | None],
) -> None:
    """No corpus_manifest.json: the corpus counts come from the playbook's stats.

    They are copied straight through, so anything but a count is ``null``.
    """
    out = tmp_path / "out"
    out.mkdir()
    playbook = {
        "opf_version": "0.5",
        "corpus": {
            "stats": {
                "documents_total": documents_total,
                "versions_total": versions_total,
                "documents_in_scope": documents_in_scope,
            },
            "documents": [{"document_id": _SECRET, "signed_version": 1}],
        },
    }
    (out / "playbook.opf.json").write_text(json.dumps(playbook), encoding="utf-8")
    assert not (out / "corpus_manifest.json").exists()

    result = _invoke(["scorecard", str(out)])
    assert result.exit_code == 0, result.output
    text = (out / "scorecard.json").read_text(encoding="utf-8")
    card = json.loads(text)
    _assert_counts_only(card)
    assert _SECRET not in text
    assert _SECRET not in result.output
    corpus = card["corpus"]
    assert (corpus["documents"], corpus["versions"], corpus["in_scope"]) == expected
    # Signed is counted from the playbook's documents, never copied.
    assert corpus["signed"] == 1
    # Every corpus field is a flat leaf: nothing nested under it.
    assert {k for k in flatten_scorecard(card) if k.startswith("corpus.")} == {
        "corpus.documents",
        "corpus.versions",
        "corpus.signed",
        "corpus.in_scope",
        "corpus.quarantined",
    }


# ---------------------------------------------------------------------------
# Missing artifacts, future fields, queues, comparison
# ---------------------------------------------------------------------------


def test_empty_out_dir_scores_null_never_errors(tmp_path: Path) -> None:
    result = _invoke(["scorecard", str(tmp_path)])
    assert result.exit_code == 0, result.output
    card = json.loads((tmp_path / "scorecard.json").read_text(encoding="utf-8"))
    assert set(card) == _TOP_LEVEL
    assert card["opf_version"] is None
    assert set(card["corpus"].values()) == {None}
    assert card["template"]["standards_classified"] is None
    for section in (
        "classification",
        "precedent",
        "template_drift",
        "opening_drift",
        "digest",
        "equivalence",
        "dossiers",
    ):
        assert card[section] is None, section
    assert card["dropped_observations"] is None
    assert card["queues"] == {"judge_pending": 0, "judge_pending_by_kind": {}, "segment_pending": 0}
    _assert_counts_only(card)


def test_missing_out_dir_is_an_error(tmp_path: Path) -> None:
    result = _invoke(["scorecard", str(tmp_path / "nope")])
    assert result.exit_code == 1


def test_opf_05_fields_are_counted_once_carried(nda_out: Path, tmp_path: Path) -> None:
    out = _copy_out(nda_out, tmp_path / "out")
    path = out / "playbook.opf.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    records = doc["evidence"]["precedent"]
    # The pipeline's own openings today are our standard struck or edited
    # before signing: under #233 they opened with "standard".
    struck_standard = [r for r in records if r.get("opening_text") is not None]
    assert struck_standard, "the NDA example records openings today"
    # #233: opening_text is non-null only with opened_with standard or
    # non_standard (and only when it differs from what was signed); "absent"
    # and null never carry one, and an unsigned deal's opened_with is null.
    cycle = ["standard", "non_standard", "absent", None]
    expected_openings = {"standard": len(struck_standard), "non_standard": 0}
    expected_records = {"standard": 0, "non_standard": 0, "absent": 0, "undetermined": 0}
    signed_null = 0
    for i, rec in enumerate(records):
        if rec.get("opening_text") is not None:
            value: str | None = "standard"
        elif rec.get("signed") is not True:
            value = None
        else:
            value = cycle[i % len(cycle)]
            # Every other standard / non_standard record opened with a
            # distinct first draft that we moved off before signing.
            if value in ("standard", "non_standard") and (i // len(cycle)) % 2 == 0:
                rec["opening_text"] = {
                    "text": "a distinct first draft",
                    "ref": {"document_id": rec["document_id"], "version": 1, "clause_path": "1"},
                }
                rec["moved"] = True
                expected_openings[value] += 1
        rec["opened_with"] = value
        expected_records["undetermined" if value is None else value] += 1
        if value is None and rec.get("signed") is True:
            signed_null += 1
    for rec in records:
        assert rec["opening_text"] is None or rec["opened_with"] in ("standard", "non_standard")
    doc["dossiers"] = {
        "clause.a": {"excerpts": ["x" * 400]},
        "clause.b": {"excerpts": ["y" * 800]},
        "clause.c": {"excerpts": []},
    }
    path.write_text(json.dumps(doc), encoding="utf-8")

    card = build_scorecard(out)
    precedent = card["precedent"]
    # Openings are the records with a non-null opening_text, split by what
    # they opened with; the split covers exactly the openings.
    assert precedent["openings"] == sum(expected_openings.values())
    assert precedent["openings_by_opened_with"] == {k: v for k, v in expected_openings.items() if v}
    assert expected_openings["non_standard"] > 0
    assert sum(precedent["openings_by_opened_with"].values()) == precedent["openings"]
    # The all-records distribution is a separate field that sums to records.
    assert precedent["records_by_opened_with"] == {k: v for k, v in expected_records.items() if v}
    assert sum(precedent["records_by_opened_with"].values()) == precedent["records"]
    assert precedent["records"] > precedent["openings"]
    assert signed_null > 0
    assert precedent["signed_opened_with_undetermined"] == signed_null
    dossiers = card["dossiers"]
    assert dossiers["count"] == 3
    assert dossiers["max_tokens"] > dossiers["median_tokens"] > 0
    _assert_counts_only(card)


def test_observations_from_an_older_store_count_as_unrecorded(
    nda_out: Path, tmp_path: Path
) -> None:
    out = _copy_out(nda_out, tmp_path / "out")
    path = out / "observations.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    for row in rows[:10]:
        row.pop("x_classification_basis", None)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    card = build_scorecard(out)
    assert card["classification"]["by_basis"]["unrecorded"] == 10
    for row in rows:
        row.pop("x_classification_basis", None)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    assert build_scorecard(out)["classification"]["by_basis"] is None


def test_digest_capped_clauses_count_lists_shorter_than_their_totals(
    nda_out: Path, tmp_path: Path
) -> None:
    out = _copy_out(nda_out, tmp_path / "out")
    path = out / "playbook.opf.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert build_scorecard(out)["digest"]["capped_clauses"] == 0
    # The real digest builder under a budget it cannot meet tightens every
    # list to its floor of one entry, so clauses with more groups than that
    # show fewer than their uncapped *_total.
    doc["digest"] = build_digest(doc, token_budget=1)
    path.write_text(json.dumps(doc), encoding="utf-8")
    clauses = doc["digest"]["clauses"]
    capped = [
        c
        for c in clauses
        if len(c.get("signed_variants") or []) < c["n_variants_total"]
        or len(c.get("refused_asks") or []) < c["n_refused_total"]
    ]
    # Issue #235: the counterparty-paper deals' content-similarity assignments
    # add signed variants to five clauses, so five (was two) exceed the floor
    # of one entry per list.
    assert len(capped) == 5

    digest = build_scorecard(out)["digest"]
    assert digest["capped_clauses"] == len(capped)
    assert digest["signed_variant_groups"] == sum(c["n_variants_total"] for c in clauses)
    assert digest["refused_ask_groups"] == sum(c["n_refused_total"] for c in clauses)


def test_queues_count_only_items_the_store_has_answered_under_their_rubric(
    tmp_path: Path,
) -> None:
    """Queue and store files built by the real writers, re-queues included.

    The store-backed judges re-queue an item under the SAME key when its
    banked verdict's rubric is stale (or unstamped under --strict-rubric), so
    a key present in ``verdicts.jsonl`` is not by itself an answer.
    """
    classify_v = rubric_version("classify", taxonomy={})
    scope_v = rubric_version("scope")
    verdict = {"taxonomy_id": None, "confidence": 0.0, "basis": "judge"}
    store = VerdictStore(tmp_path / "judge" / "verdicts.jsonl")
    # Answered: banked under the rubric the item was queued under.
    store.put_by_key("a", verdict, rubric=RubricStamp("classify", classify_v))
    # Stale: banked under an older rubric, then re-queued by the judge.
    store.put_by_key("d", verdict, rubric=RubricStamp("classify", "v0+000000000000"))
    # Legacy: banked unstamped, then re-queued (--strict-rubric).
    store.put_by_key("e", verdict)
    # Stale, then re-answered by judge-apply: the last record wins.
    store.put_by_key("f", verdict, rubric=RubricStamp("classify", "v0+000000000000"))
    store.put_by_key("f", verdict, rubric=RubricStamp("classify", classify_v))

    round_one = PendingQueue(tmp_path / "judge" / "pending.jsonl")
    for key in ("a", "b", "d", "e", "f"):
        round_one.add(key, "classify", {"key": key}, classify_v)
    round_one.add("c", "scope", {"key": "c"}, scope_v)
    # A second writer (a later round) appends "b" again: counted once.
    PendingQueue(tmp_path / "judge" / "pending.jsonl").add("b", "classify", {}, classify_v)

    segment_queue = PendingQueue(tmp_path / "segment" / "pending.jsonl")
    segment_fn = StoreBackedSegmentFn(pending=segment_queue, taxonomy_ids=["term"])
    for text in ("first document", "second document", "third document"):
        with pytest.raises(AgentSegmentationPending):
            segment_fn(text, [])
    cache = SegmentationVerdictCache(tmp_path / "segment" / "cache.jsonl")
    # Answered: what `playbook segment-apply` banks.
    cache.put("second document", [], model=AGENT_SEGMENTER_MODEL)
    # Banked for a different segmenter model: not this queue's answer.
    cache.put("third document", [], model="some-other-model")

    queues = build_scorecard(tmp_path)["queues"]
    assert queues == {
        "judge_pending": 4,
        "judge_pending_by_kind": {"classify": 3, "scope": 1},
        "segment_pending": 2,
    }
    assert pending_queue_counts(tmp_path) == (4, 2)

    # Once every item is answered under its rubric, nothing is owed.
    for key in ("b", "d", "e"):
        store.put_by_key(key, verdict, rubric=RubricStamp("classify", classify_v))
    store.put_by_key("c", {"in_scope": True}, rubric=RubricStamp("scope", scope_v))
    cache.put("first document", [], model=AGENT_SEGMENTER_MODEL)
    cache.put("third document", [], model=AGENT_SEGMENTER_MODEL)
    assert pending_queue_counts(tmp_path) == (0, 0)


def test_compare_prints_deltas_against_an_earlier_scorecard(nda_out: Path, tmp_path: Path) -> None:
    out = _copy_out(nda_out, tmp_path / "out")
    first = _invoke(["scorecard", str(out), "--out", str(tmp_path / "baseline.json")])
    assert first.exit_code == 0, first.output
    baseline = json.loads((tmp_path / "baseline.json").read_text(encoding="utf-8"))
    baseline["precedent"]["records"] -= 4
    baseline["classification"]["classified_share"] = 0.5
    (tmp_path / "baseline.json").write_text(json.dumps(baseline), encoding="utf-8")

    result = _invoke(["scorecard", str(out), "--compare", str(tmp_path / "baseline.json")])
    assert result.exit_code == 0, result.output
    header = result.output.splitlines()[0].split()
    assert header == ["field", "baseline", "current", "delta"]
    line = next(x for x in result.output.splitlines() if x.startswith("precedent.records "))
    assert line.split()[-1] == "+4"

    card = json.loads((out / "scorecard.json").read_text(encoding="utf-8"))
    rows = {r[0]: r for r in compare_scorecards(card, baseline)}
    assert rows["precedent.records"][3] == 4
    assert rows["classification.classified_share"][3] == round(
        card["classification"]["classified_share"] - 0.5, 4
    )
    assert rows["corpus.documents"][3] == 0
    assert rows["opf_version"][3] is None


def test_compare_rejects_an_unreadable_baseline(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("not json", encoding="utf-8")
    result = _invoke(["scorecard", str(tmp_path), "--compare", str(bad)])
    assert result.exit_code == 1


# ---------------------------------------------------------------------------
# pipeline: x_classification_basis
# ---------------------------------------------------------------------------


def _obs(taxonomy_id: str | None, version_id: str | None, clause_path: str) -> Observation:
    return Observation(
        observation_id="d/1/1",
        taxonomy_id=taxonomy_id,
        text_summary="text",
        citation=ObservationCitation(
            document_id="d",
            version=1,
            clause_path=clause_path,
            char_span=None,
            version_id=version_id,
        ),
        deviation="none",
        risk_delta={"direction": "neutral", "magnitude": "none"},
        provenance="our_paper",
        outcome="signed",
    )


def test_observation_classification_basis_reads_the_cited_node() -> None:
    by_node = {
        ("v1", "1"): ClauseClassification("governing_law", 1.0, "exact_match"),
        ("v1", "2"): ClauseClassification(None, 0.0, "unclassified"),
        ("v2", "1.1"): ClauseClassification("term", 0.6, "inherited"),
    }
    assert _observation_classification_basis(_obs("governing_law", "v1", "1"), by_node) == (
        "exact_match"
    )
    assert _observation_classification_basis(_obs(None, "v1", "2"), by_node) == "unclassified"
    assert _observation_classification_basis(_obs("term", "v2", "1.1"), by_node) == "inherited"
    # The row's taxonomy_id, not the node's own classification.
    assert _observation_classification_basis(_obs("notices", "v1", "2"), by_node) == "aligned"
    # An unclassified row is "unclassified" even when its cited node is
    # classified: "aligned" only means "classified via the aligned row".
    assert _observation_classification_basis(_obs(None, "v1", "1"), by_node) == "unclassified"
    assert _observation_classification_basis(_obs(None, "v2", "1.1"), by_node) == "unclassified"
    assert _observation_classification_basis(_obs("term", "v9", "1"), by_node) is None
    assert _observation_classification_basis(_obs("term", None, "1"), by_node) is None
    assert _obs("term", "v1", "1").to_dict().get("x_classification_basis") is None
