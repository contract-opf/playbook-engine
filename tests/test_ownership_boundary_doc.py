"""Tests for the OPF-vs-bundle ownership-boundary doc (issue #150).

Verifies:
  - `docs/OPF-BUNDLE-BOUNDARY.md` exists.
  - It carries the OPF-owns and bundle-owns headings.
  - It states the canonical-format decision (OPF v0.2 IS the format; a
    consuming app's playbook = OPF document + bundle wrapper), superseding #115's
    converter framing.
  - `docs/OPF-SPEC.md` (the v0.2 spec, promoted to this filename by #172)
    cross-links to it.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).parent.parent
BOUNDARY_DOC_PATH = ROOT / "docs" / "OPF-BUNDLE-BOUNDARY.md"
SPEC_DRAFT_PATH = ROOT / "docs" / "OPF-SPEC.md"


def _boundary_text() -> str:
    return BOUNDARY_DOC_PATH.read_text(encoding="utf-8")


def test_boundary_doc_exists() -> None:
    assert BOUNDARY_DOC_PATH.is_file()


def test_boundary_doc_has_ownership_headings() -> None:
    text = _boundary_text()
    assert "## What OPF owns" in text
    assert "## What the bundle owns" in text


def test_boundary_doc_states_canonical_format() -> None:
    text = _boundary_text()
    assert "OPF v0.2 is the canonical playbook format" in text
    assert "supersedes" in text.lower()
    assert "#115" in text


def test_boundary_doc_covers_owned_concerns() -> None:
    text = _boundary_text()
    for term in [
        "knowledge",
        "intent",
        "red lines",
        "perspective",
        "de_minimis" if "de_minimis" in text else "de minimis",
        # The anti-rubric decision (#178, owner 2026-07-12) is itself an
        # ownership concern the doc must keep documenting: there is
        # deliberately NO structured accept/reject list alongside the
        # Posture prose.
        "deliberately no structured accept/reject",
        "model policy",
        "release",
    ]:
        assert term.lower() in text.lower(), f"missing owned concern: {term!r}"


def test_spec_draft_cross_links_to_boundary_doc() -> None:
    text = SPEC_DRAFT_PATH.read_text(encoding="utf-8")
    assert "OPF-BUNDLE-BOUNDARY.md" in text
