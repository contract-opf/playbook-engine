"""CI guard for the shipped examples (issue #164).

The flagship examples are the reference artifacts adopters pattern-match
against, so every `examples/*.playbook.json` must pass the engine's own
validator under its declared `opf_version`, the v0.2 flagship must actually
demonstrate the headline sections (Posture, Floor, curation, dynamics), its
internal counts must agree with its lists, and no example may carry real
company branding.

The NDA second-agreement-type worked example (issue #9) is committed as
`examples/nda/playbook.opf.json` rather than `examples/*.playbook.json` (it
lives alongside its corpus/config/canned-verdicts, not at the examples/ top
level), so it is added to `EXAMPLE_PATHS` explicitly below — every generic
check in this file (schema validation, no-real-branding) then covers it for
free, and `test_nda_example_has_populated_posture_and_floor` /
`test_nda_example_confidence_counts_consistent` add the NDA-specific
headline-section guard the v0.2 flagship already gets above.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from playbook_engine.canonicalize import compute_section_digests, content_hash
from playbook_engine.validator import validate_document

ROOT = Path(__file__).parent.parent
V02_FLAGSHIP = ROOT / "examples" / "our-paper-baseline.v0.2.playbook.json"
NDA_PLAYBOOK = ROOT / "examples" / "nda" / "playbook.opf.json"
EXAMPLE_PATHS = sorted((ROOT / "examples").glob("*.playbook.json")) + [NDA_PLAYBOOK]


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_examples_exist() -> None:
    assert EXAMPLE_PATHS, "no examples/*.playbook.json found"
    assert V02_FLAGSHIP in EXAMPLE_PATHS
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


def test_v02_example_demonstrates_headline_sections() -> None:
    """The v0.2 flagship must demonstrate what defines v0.2 — not ship
    empty posture/floor."""
    doc = _load(V02_FLAGSHIP)

    invariants = doc["floor"].get("invariants", [])
    assert len(invariants) >= 2, "flagship must demonstrate >=2 floor.invariants"

    posture = doc["posture"]
    assert posture.get("system_prompt", "").strip(), "flagship posture must be populated"
    interview = posture.get("generation", {}).get("interview", [])
    assert len(interview) >= 3, "flagship must carry >=3 interview entries"

    pins = doc.get("curation", {}).get("pins", [])
    assert len(pins) >= 1, "flagship must demonstrate a curation pin"

    clauses = doc["evidence"]["clauses"]
    assert any(
        obs.get("full_text") for clause in clauses for obs in clause.get("observed_positions", [])
    ), "flagship must demonstrate full_text on at least one observation"
    assert any(clause.get("negotiation_trail") for clause in clauses), (
        "flagship must demonstrate a negotiation_trail (§3.5.3)"
    )
    # Recompute and compare — not just prefix-match — so a future prose edit
    # to the flagship that forgets to regenerate identity can't silently ship
    # a document that fails its own published integrity contract (issue #88
    # fix round 1 finding 1; OPF-SPEC §3.4 rule 1 is fail-closed on mismatch).
    identity = doc.get("identity", {})
    assert identity.get("content_hash") == content_hash(doc), (
        "flagship identity.content_hash is stale — regenerate with "
        "playbook_engine.canonicalize.content_hash() after any content edit"
    )
    assert identity.get("section_digests") == compute_section_digests(doc), (
        "flagship identity.section_digests is stale — regenerate with "
        "playbook_engine.canonicalize.compute_section_digests() after any content edit"
    )


def test_v02_example_confidence_counts_consistent() -> None:
    """confidence.n_our_paper / n_counterparty_paper must equal the actual
    provenance counts of observed_positions — the flagship previously
    claimed counts its own lists contradicted."""
    doc = _load(V02_FLAGSHIP)
    for clause in doc["evidence"]["clauses"]:
        confidence = clause["summary"]["confidence"]
        observed = clause.get("observed_positions", [])
        n_ours = sum(1 for o in observed if o.get("provenance") == "our_paper")
        n_theirs = sum(1 for o in observed if o.get("provenance") == "counterparty_paper")
        assert confidence.get("n_our_paper") == n_ours, clause["id"]
        assert confidence.get("n_counterparty_paper") == n_theirs, clause["id"]


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

    clauses = doc["evidence"]["clauses"]
    assert len(clauses) >= 5, "NDA example must demonstrate real clause coverage"
    assert any(
        obs.get("full_text") for clause in clauses for obs in clause.get("observed_positions", [])
    ), "NDA example must demonstrate full_text on at least one observation"
    assert any(clause.get("negotiation_trail") for clause in clauses), (
        "NDA example must demonstrate a negotiation_trail (§3.5.3)"
    )

    # The corpus was deliberately built with >=3 versions on one deal so a
    # genuine proposed-then-reversed round-trip is observable (see the
    # issue's sequencing-note comment on the reversal_detector's >=3-version
    # requirement) — assert it actually fired rather than trusting the
    # corpus shape alone.
    reversed_obs = [
        obs
        for clause in clauses
        for obs in clause.get("observed_positions", [])
        if obs.get("outcome") == "proposed_then_reversed"
    ]
    assert reversed_obs, "NDA example must demonstrate at least one proposed_then_reversed clause"

    identity = doc.get("identity", {})
    assert identity.get("content_hash") == content_hash(doc), (
        "NDA example identity.content_hash is stale — regenerate with "
        "playbook_engine.canonicalize.content_hash() after any content edit"
    )
    assert identity.get("section_digests") == compute_section_digests(doc), (
        "NDA example identity.section_digests is stale — regenerate with "
        "playbook_engine.canonicalize.compute_section_digests() after any content edit"
    )


def test_nda_example_confidence_counts_consistent() -> None:
    """Same guard as `test_v02_example_confidence_counts_consistent`, for
    the NDA example: `confidence.n_our_paper` / `n_counterparty_paper` must
    equal the actual provenance counts of `observed_positions`."""
    doc = _load(NDA_PLAYBOOK)
    for clause in doc["evidence"]["clauses"]:
        confidence = clause["summary"]["confidence"]
        observed = clause.get("observed_positions", [])
        n_ours = sum(1 for o in observed if o.get("provenance") == "our_paper")
        n_theirs = sum(1 for o in observed if o.get("provenance") == "counterparty_paper")
        assert confidence.get("n_our_paper") == n_ours, clause["id"]
        assert confidence.get("n_counterparty_paper") == n_theirs, clause["id"]


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
    happened to resolve the template against. `publisher.py` treats this
    exact leak as load-bearing enough to strip unconditionally before
    publication (see its `_strip_source_paths` step) — a shipped example
    must ship already scrubbed, not rely on a downstream `publish` call
    that never runs on it.
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
