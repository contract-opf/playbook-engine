"""Tests for pipeline._discover_versions / _discover_legacy_doc_files ordering.

Issue #37: natural_sort.py's docstring promises it is "used wherever version
files must be listed in a stable, human-intuitive order" so that ``v2`` sorts
before ``v10`` — but _discover_versions and _discover_legacy_doc_files used
plain lexicographic ``sorted()``, which for unpadded version names (common in
real negotiation folders, e.g. draft-1 ... draft-12) puts draft-10 before
draft-2.

SECURITY NOTE: fixtures use synthetic filenames only; no real corpus content.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from playbook_engine.pipeline import _discover_legacy_doc_files, _discover_versions


def _touch(path: Path) -> None:
    path.write_text("stub", encoding="utf-8")


def test_discover_versions_uses_natural_sort_not_lexicographic(tmp_path: Path) -> None:
    _touch(tmp_path / "draft-1.docx")
    _touch(tmp_path / "draft-2.docx")
    _touch(tmp_path / "draft-10.docx")

    result = _discover_versions(tmp_path)

    assert [p.name for p in result] == ["draft-1.docx", "draft-2.docx", "draft-10.docx"]


def test_discover_legacy_doc_files_uses_natural_sort_not_lexicographic(tmp_path: Path) -> None:
    _touch(tmp_path / "draft-1.doc")
    _touch(tmp_path / "draft-2.doc")
    _touch(tmp_path / "draft-10.doc")

    result = _discover_legacy_doc_files(tmp_path)

    assert [p.name for p in result] == ["draft-1.doc", "draft-2.doc", "draft-10.doc"]


@pytest.mark.parametrize(
    "creation_order",
    [
        ["a.docx", "a.pdf", "a.rtf"],
        ["a.rtf", "a.pdf", "a.docx"],
        ["a.pdf", "a.docx", "a.rtf"],
    ],
)
def test_discover_versions_tie_group_is_totally_and_deterministically_ordered(
    tmp_path: Path, creation_order: list[str]
) -> None:
    """Issue #37: files sharing a stem (mixed formats in one folder — blessed
    by docs/CORPUS-LAYOUT.md:24) must not compare EQUAL under the sort key,
    or Python's stable sort falls back to filesystem readdir order, which
    varies by filesystem/host and is exactly what this ticket exists to
    eliminate from the staged ``order`` hints. The key must be total (tie-broken
    on the full filename) so the result is identical regardless of on-disk
    creation/iteration order.
    """
    for name in creation_order:
        _touch(tmp_path / name)

    result = _discover_versions(tmp_path)

    assert [p.name for p in result] == ["a.docx", "a.pdf", "a.rtf"]
