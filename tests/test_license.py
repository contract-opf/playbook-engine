"""Tests for the repo LICENSE licensing hygiene (issue #148).

Verifies:
  - The Apache License, Version 2.0 body is vendored verbatim into LICENSE
    (not just referenced by URL).
  - The old "to be vendored" placeholder is gone.
  - The existing CC-BY grant note for OPF spec text is preserved.

Schema `$id` conformance (canonical https:// URLs, restored by issue #160
superseding #148's interim repo-relative decision) is covered by
tests/test_schema_ids.py.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).parent.parent
LICENSE_PATH = ROOT / "LICENSE"

# The canonical Apache License 2.0 body (TERMS AND CONDITIONS FOR USE,
# REPRODUCTION, AND DISTRIBUTION, Sections 1-9), as published at
# https://www.apache.org/licenses/LICENSE-2.0.txt
APACHE_2_0_BODY_MARKERS = [
    "                                 Apache License",
    "                           Version 2.0, January 2004",
    "   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION",
    '      "License" shall mean the terms and conditions for use, reproduction,',
    "   2. Grant of Copyright License. Subject to the terms and conditions of",
    "   3. Grant of Patent License. Subject to the terms and conditions of",
    "   4. Redistribution. You may reproduce and distribute copies of the",
    "   5. Submission of Contributions. Unless You explicitly state otherwise,",
    "   6. Trademarks. This License does not grant permission to use the trade",
    "   7. Disclaimer of Warranty. Unless required by applicable law or",
    "   8. Limitation of Liability. In no event and under no legal theory,",
    "   9. Accepting Warranty or Additional Liability. While redistributing",
    "   END OF TERMS AND CONDITIONS",
]


def _license_text() -> str:
    return LICENSE_PATH.read_text(encoding="utf-8")


def test_apache_2_0_body_present_verbatim() -> None:
    text = _license_text()
    for marker in APACHE_2_0_BODY_MARKERS:
        assert marker in text, f"missing Apache-2.0 body text: {marker!r}"


def test_vendored_placeholder_is_gone() -> None:
    text = _license_text()
    assert "to be vendored" not in text.lower()


def test_cc_by_grant_note_preserved() -> None:
    text = _license_text()
    assert "CC-BY-4.0" in text
    assert "creativecommons.org/licenses/by/4.0" in text
    assert "OPEN PLAYBOOK FORMAT (OPF) SPECIFICATION TEXT" in text


def test_opf_v02_spec_has_cc_by_marker() -> None:
    # #172 promoted the v0.2 draft to the canonical docs/OPF-SPEC.md filename.
    spec_path = ROOT / "docs" / "OPF-SPEC.md"
    text = spec_path.read_text(encoding="utf-8")
    assert "CC-BY-4.0" in text
