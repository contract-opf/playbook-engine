"""Tests for intake_plan.py — universal corpus intake (issue #186).

Builds a scrambled, flat-loose-files copy of the persistent
``examples/judge-fixture/corpus`` fixture (deal-alpha: v1/v2 rtf; deal-beta:
v1 rtf) in-test — renamed to generic ``doc_NNN`` names with no per-agreement
subfolders (an "unknown" layout: staging.detect_layout would refuse to guess
this one). No second fixture corpus is committed.

Original chronological order within a cluster is recovered via filesystem
mtime (the weakest evidence tier — see ``intake_plan._fallback_timestamp``),
which this test controls explicitly since the RTF fixture carries no
embedded-metadata dates for the stronger tiers to key off.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import pytest
from click.testing import CliRunner
from docx import Document

from playbook_engine.cli import cli
from playbook_engine.intake_plan import build_staging_plan, execute_staging_plan
from playbook_engine.staging import UnknownLayoutError, stage

FIXTURE_CORPUS = Path(__file__).parent.parent / "examples" / "judge-fixture" / "corpus"
_DAY = 86400.0


def _scrambled_corpus(dest: Path) -> None:
    """Write a flat, loosely-named, scrambled copy of FIXTURE_CORPUS into *dest*.

    Mapping (scrambled name -> original) is deliberately out of order and
    stripped of any agreement-identifying folder/name signal:
      doc_001.rtf <- deal-beta/v1.rtf   (mtime: -3 days)
      doc_002.rtf <- deal-alpha/v2.rtf  (mtime: -1 day)
      doc_003.rtf <- deal-alpha/v1.rtf  (mtime: -2 days)

    mtimes are set explicitly (not just copy order) so version_orderer's
    timestamp tie-break can recover alpha's original v1-before-v2 order from
    content-tied permutations (two non-signed versions with no signed
    anchor have exactly one pairwise distance, so cost alone can't order
    them — see version_orderer.order_versions).
    """
    dest.mkdir(parents=True, exist_ok=True)
    base = time.time()
    mapping = [
        (FIXTURE_CORPUS / "deal-beta" / "v1.rtf", "doc_001.rtf", -3 * _DAY),
        (FIXTURE_CORPUS / "deal-alpha" / "v2.rtf", "doc_002.rtf", -1 * _DAY),
        (FIXTURE_CORPUS / "deal-alpha" / "v1.rtf", "doc_003.rtf", -2 * _DAY),
    ]
    for original, name, offset in mapping:
        target = dest / name
        shutil.copy2(original, target)
        t = base + offset
        os.utime(target, (t, t))


def _add_lorem_ipsum_docx(dest: Path, name: str = "doc_004.docx") -> None:
    """Write an unrelated lorem-ipsum DOCX into *dest* — no vocabulary overlap
    with the legal-boilerplate fixture corpus, so its nearest-neighbour
    content distance to every real file is ~1.0 (see intake_plan.UNRELATED_CEILING)."""
    doc = Document()
    doc.add_paragraph("Lorem Ipsum")
    doc.add_paragraph(
        "Lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod "
        "tempor incididunt ut labore et dolore magna aliqua."
    )
    doc.save(str(dest / name))


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolved_files_by_deal(out_dir: Path) -> dict[str, list[str]]:
    """Map each staged agreement folder to its ordered list of content hashes
    (following symlinks) — for comparing staged output byte-for-byte,
    independent of deal_id naming AND of which source tree (scrambled temp
    copy vs. the persistent fixture) the files were staged from."""
    result: dict[str, list[str]] = {}
    for agreement_dir in sorted(p for p in out_dir.iterdir() if p.is_dir()):
        files = sorted(
            p
            for p in agreement_dir.iterdir()
            if (p.is_file() or p.is_symlink()) and p.name != "hints.yaml"
        )
        result[agreement_dir.name] = [_hash(f) for f in files]
    return result


# ---------------------------------------------------------------------------
# 1. Reconstruction
# ---------------------------------------------------------------------------


class TestBuildStagingPlan:
    def test_plan_reconstructs_scrambled_fixture(self, tmp_path: Path) -> None:
        scrambled = tmp_path / "scrambled"
        _scrambled_corpus(scrambled)

        plan = build_staging_plan(scrambled)

        assert plan["layout"] == "unknown"
        assert plan["unassigned"] == []
        assert len(plan["deals"]) == 2

        by_size = sorted(plan["deals"], key=lambda d: len(d["files"]))
        singleton, pair = by_size[0], by_size[1]

        # deal-beta: exactly one version, unsigned.
        assert len(singleton["files"]) == 1
        assert singleton["files"][0]["path"] == "doc_001.rtf"
        assert singleton["files"][0]["signed"] is False

        # deal-alpha: two versions, original chronological order recovered
        # (v1 -> doc_003.rtf, v2 -> doc_002.rtf) despite scrambled naming,
        # and neither is signed (matches the un-scrambled fixture — same
        # signed anchors: both None).
        assert [f["path"] for f in pair["files"]] == ["doc_003.rtf", "doc_002.rtf"]
        assert [f["proposed_version"] for f in pair["files"]] == [1, 2]
        assert all(f["signed"] is False for f in pair["files"])

    def test_unrelated_file_unassigned(self, tmp_path: Path) -> None:
        scrambled = tmp_path / "scrambled"
        _scrambled_corpus(scrambled)
        _add_lorem_ipsum_docx(scrambled)

        plan = build_staging_plan(scrambled)

        assert plan["unassigned"], "lorem-ipsum file should land in unassigned"
        unassigned_paths = [u["path"] for u in plan["unassigned"]]
        assert "doc_004.docx" in unassigned_paths
        for u in plan["unassigned"]:
            assert u["reason"]

        # The two real deals are still correctly assembled (the junk file
        # didn't merge into or otherwise disturb either cluster).
        assert len(plan["deals"]) == 2

        # CLI plan-only run over the same scrambled+junk corpus exits 0.
        runner = CliRunner()
        result = runner.invoke(
            cli, ["stage", str(scrambled), "--out", str(tmp_path / "out"), "--plan-only"]
        )
        assert result.exit_code == 0, result.output

    def test_evidence_recorded_per_file(self, tmp_path: Path) -> None:
        scrambled = tmp_path / "scrambled"
        _scrambled_corpus(scrambled)
        _add_lorem_ipsum_docx(scrambled)

        plan = build_staging_plan(scrambled)

        for deal in plan["deals"]:
            for f in deal["files"]:
                assert f["evidence"], f"{f['path']} has no evidence recorded"


# ---------------------------------------------------------------------------
# 2. stage() refuses to guess on an unknown layout
# ---------------------------------------------------------------------------


class TestPlanNeverStagesDirectly:
    def test_stage_raises_on_unknown_layout(self, tmp_path: Path) -> None:
        scrambled = tmp_path / "scrambled"
        _scrambled_corpus(scrambled)

        with pytest.raises(UnknownLayoutError, match="staging_plan"):
            stage(scrambled, tmp_path / "out")

    def test_plan_never_stages_directly(self, tmp_path: Path) -> None:
        scrambled = tmp_path / "scrambled"
        _scrambled_corpus(scrambled)

        runner = CliRunner()
        result = runner.invoke(cli, ["stage", str(scrambled), "--out", str(tmp_path / "out")])

        assert result.exit_code != 0
        assert "staging_plan" in result.output


# ---------------------------------------------------------------------------
# 3. execute_staging_plan matches direct staging of the un-scrambled fixture
# ---------------------------------------------------------------------------


class TestExecuteStagingPlan:
    def test_execute_plan_equals_canonical(self, tmp_path: Path) -> None:
        scrambled = tmp_path / "scrambled"
        _scrambled_corpus(scrambled)

        plan = build_staging_plan(scrambled)
        out_from_plan = tmp_path / "out_plan"
        plan_result = execute_staging_plan(plan, scrambled, out_from_plan)

        out_canonical = tmp_path / "out_canonical"
        canonical_result = stage(FIXTURE_CORPUS, out_canonical)

        assert plan_result.agreement_count == canonical_result.agreement_count == 2
        assert plan_result.staged_count == canonical_result.staged_count == 3

        # Compare by resolved source file content, not by deal_id/agreement
        # folder naming (the plan can't recover "deal-alpha"/"deal-beta" as
        # names — only their content/order/signed-ness).
        plan_by_deal = _resolved_files_by_deal(out_from_plan)
        canonical_by_deal = _resolved_files_by_deal(out_canonical)

        plan_groups = {frozenset(files) for files in plan_by_deal.values()}
        canonical_groups = {frozenset(files) for files in canonical_by_deal.values()}
        assert plan_groups == canonical_groups

        # Version order within each group also matches (list, not just set).
        plan_order_by_group = {frozenset(files): files for files in plan_by_deal.values()}
        canonical_order_by_group = {frozenset(files): files for files in canonical_by_deal.values()}
        for group in plan_groups:
            assert plan_order_by_group[group] == canonical_order_by_group[group]

    def test_execute_writes_hints_with_ordering(self, tmp_path: Path) -> None:
        scrambled = tmp_path / "scrambled"
        _scrambled_corpus(scrambled)

        plan = build_staging_plan(scrambled)
        out_dir = tmp_path / "out"
        execute_staging_plan(plan, scrambled, out_dir)

        for deal in plan["deals"]:
            hints_path = out_dir / deal["deal_id"] / "hints.yaml"
            assert hints_path.exists()

    def test_plan_json_roundtrip(self, tmp_path: Path) -> None:
        """A plan serialized to JSON and read back executes identically —
        the real CLI workflow (--plan-only writes it, --plan reads it back)."""
        scrambled = tmp_path / "scrambled"
        _scrambled_corpus(scrambled)

        plan = build_staging_plan(scrambled)
        plan_path = tmp_path / "staging_plan.json"
        plan_path.write_text(json.dumps(plan), encoding="utf-8")

        reloaded = json.loads(plan_path.read_text(encoding="utf-8"))
        out_dir = tmp_path / "out"
        result = execute_staging_plan(reloaded, scrambled, out_dir)
        assert result.agreement_count == 2
        assert result.staged_count == 3
