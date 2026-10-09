"""CI guard for the shipped examples (issue #164).

The reference playbook is the artifact adopters pattern-match against, so it
must pass the engine's own validator, actually demonstrate the headline
sections (Posture, Floor, precedent with moves, openings and refused asks),
keep its internal counts consistent with its precedent, and carry no real
company branding or machine paths.

The reference playbook is the NDA worked example (issue #9),
`examples/nda/playbook.opf.json`. The OPF 0.1 and 0.2 worked examples were
retired with those formats (issue #238); any future top-level
`examples/*.playbook.json` is covered by the generic checks too.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from playbook_engine.canonicalize import compute_section_digests, content_hash
from playbook_engine.digest import build_digest_v4
from playbook_engine.validator import validate_document

ROOT = Path(__file__).parent.parent
NDA_PLAYBOOK = ROOT / "examples" / "nda" / "playbook.opf.json"
EXAMPLE_PATHS = sorted((ROOT / "examples").glob("*.playbook.json")) + [NDA_PLAYBOOK]


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_examples_exist() -> None:
    assert NDA_PLAYBOOK in EXAMPLE_PATHS
    assert NDA_PLAYBOOK.exists(), (
        "examples/nda/playbook.opf.json is missing — the NDA second-agreement-type "
        "worked example must ship a derived playbook, not just a corpus (issue #9)"
    )


@pytest.mark.parametrize("path", EXAMPLE_PATHS, ids=lambda p: p.name)
def test_all_examples_validate(path: Path) -> None:
    """Every shipped example validates under its declared opf_version —
    the guard that would have caught the flagship being non-conformant."""
    doc = _load(path)
    result = validate_document(doc)
    blocking = [str(e) for e in result.errors if e.blocking]
    assert result.ok, f"{path.name} fails its own engine's validation: {blocking}"


def test_retired_format_examples_are_gone() -> None:
    """Issue #238: the 0.1/0.2 worked examples were retired with those formats."""
    for name in (
        "our-paper-baseline.v0.2.playbook.json",
        "our-paper-baseline.playbook.json",
        "emergent-no-template.playbook.json",
    ):
        assert not (ROOT / "examples" / name).exists(), name


def test_nda_example_has_populated_posture_and_floor() -> None:
    """The NDA second-agreement-type example (issue #9) must demonstrate a
    genuinely worked playbook — not the evidence-only, empty-posture/floor
    state the no-LLM smoke run (`make smoke-nda`) deliberately produces.

    An installed playbook with `posture {}` / `floor {}` is exactly the
    stale-example failure mode this ticket exists to avoid — so this guard
    is load-bearing, not decorative.
    """
    doc = _load(NDA_PLAYBOOK)
    assert doc["agreement_type"]["id"] == "nda"

    posture = doc["posture"]
    assert posture.get("system_prompt", "").strip(), "NDA example posture must be populated"
    interview = posture.get("generation", {}).get("interview", [])
    assert len(interview) >= 3, "NDA example must carry >=3 interview entries"

    invariants = doc["floor"].get("invariants", [])
    assert len(invariants) >= 2, "NDA example must demonstrate >=2 floor.invariants"

    # Issue #223: the NDA example is the OPF 0.5 reference artifact — the
    # verdict-free per-deal precedent record with a digest_version 4 digest.
    assert doc["opf_version"] == "0.5"
    assert doc["digest"]["digest_version"] == "4"
    assert doc["digest"] == build_digest_v4(doc), "NDA example digest is stale"
    assert doc["digest"]["perspective"] == doc["perspective"]

    clauses = doc["evidence"]["clauses"]
    precedent = doc["evidence"]["precedent"]
    assert len(clauses) >= 5, "NDA example must demonstrate real clause coverage"
    assert any(p.get("signed_text") for p in precedent), (
        "NDA example must demonstrate signed text on at least one precedent"
    )
    assert any(p["rounds"] >= 1 for p in precedent), (
        "NDA example must demonstrate a clause that moved across rounds"
    )
    # Our standard struck before signing is a precedent with an opening text
    # and no refused ask for it (issue #216's origin rule).
    assert any(p["opening_text"] for p in precedent), (
        "NDA example must demonstrate our standard struck before signing"
    )

    # The corpus was deliberately built with >=3 versions on one deal so a
    # genuine proposed-then-reversed round-trip is observable (see the
    # issue's sequencing-note comment on the reversal_detector's >=3-version
    # requirement) — assert it actually fired rather than trusting the
    # corpus shape alone.
    assert any(p["refused_asks"] for p in precedent), (
        "NDA example must demonstrate at least one refused ask"
    )

    identity = doc.get("identity", {})
    assert identity.get("content_hash") == content_hash(doc), (
        "NDA example identity.content_hash is stale — regenerate with "
        "playbook_engine.canonicalize.content_hash() after any content edit"
    )
    assert identity.get("section_digests") == compute_section_digests(doc), (
        "NDA example identity.section_digests is stale — regenerate with "
        "playbook_engine.canonicalize.compute_section_digests() after any content edit"
    )


def test_nda_example_precedent_counts_consistent() -> None:
    """OPF 0.5 successor of the confidence-count guard (issue #223): each
    clause's n_* counts equal what its precedent implies, every count is
    distinct deals (issue #216), and there is one precedent per (deal,
    clause)."""
    from playbook_engine.opf_accessors import perspective_party
    from playbook_engine.precedent import clause_counts

    doc = _load(NDA_PLAYBOOK)
    precedent = doc["evidence"]["precedent"]
    pairs = [(p["document_id"], p["taxonomy_id"]) for p in precedent]
    assert len(pairs) == len(set(pairs))
    for clause in doc["evidence"]["clauses"]:
        expected = clause_counts(clause["taxonomy_id"], precedent, party=perspective_party(doc))
        assert {k: clause[k] for k in expected} == expected, clause["id"]
        deals = {p["document_id"] for p in precedent if p["taxonomy_id"] == clause["taxonomy_id"]}
        assert clause["n_deals"] == len(deals), clause["id"]


@pytest.mark.parametrize("path", EXAMPLE_PATHS, ids=lambda p: p.name)
def test_examples_carry_no_real_branding(path: Path) -> None:
    """Examples must not read as a real company's positions (#164/#170)."""
    text = path.read_text(encoding="utf-8")
    assert not re.search(r"exos", text, flags=re.IGNORECASE), (
        f"{path.name} carries real branding — use the fictional FixtureCorp"
    )


@pytest.mark.parametrize("path", EXAMPLE_PATHS, ids=lambda p: p.name)
def test_examples_carry_no_absolute_filesystem_path(path: Path) -> None:
    """Examples must not leak the authoring machine's directory structure
    (issue #9 fix round 1 finding 2): a committed `playbook.opf.json` is a
    public artifact, and fields like `baseline.template_ref.source` are
    populated at derivation time with whatever path the deriving machine
    happened to resolve the template against. A shipped example must ship
    already scrubbed: there is no downstream step that scrubs it.
    """
    text = path.read_text(encoding="utf-8")
    # The Windows-drive branch requires the drive letter not be preceded by
    # another word character, so it doesn't false-positive on ordinary text
    # ending "...e:" immediately before a JSON-escaped "\n" (e.g. a
    # signature-block placeholder like "Title:\nSignature:") -- that's a
    # single backslash after a letter-colon, not a drive-letter path.
    assert not re.search(r"/Users/|/home/|(?<![A-Za-z0-9])[A-Za-z]:\\+", text), (
        f"{path.name} carries an absolute filesystem path — this leaks the "
        "authoring machine's home directory/username into a public example; "
        "scrub it (e.g. strip baseline.template_ref.source, keeping sha256)"
    )
