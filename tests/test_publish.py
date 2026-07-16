"""Tests for playbook_engine.publisher (issue #188).

Verified entirely offline with FAKE ``RedactionJudge``/``VerifyJudge``
implementations — no LLM, no network (per issue #188's required
verification: "NO live LLM anywhere in this issue").

SECURITY NOTE: all fixtures use synthetic text and fictional party/
institution names only. No real agreement text or real document paths are
used. Where a fixture needs a "real entity name" to prove the deterministic
backstop catches it, a made-up institution name is used.
"""

from __future__ import annotations

import copy
import re
from typing import Any

import pytest

from playbook_engine.export_profile import RedactionFinding, VerifyFinding
from playbook_engine.publisher import (
    DEFAULT_COUNTERPARTY_LABEL,
    DEFAULT_PARTY_LABEL,
    PublishError,
    publish_playbook,
)
from playbook_engine.validator import validate_document

_FAKE_HASH = "a" * 64

REAL_NAME = "Northwind State University"


# ---------------------------------------------------------------------------
# Fixture: a full, schema-valid OPF v0.2 playbook with every free-text
# surface issue #188 extended residue sampling to, populated.
# ---------------------------------------------------------------------------


def _make_doc(*, real_name: str | None = None, alias: str = "Counterparty-7") -> dict[str, Any]:
    """Return a schema-valid OPF v0.2 doc.

    If *real_name* is given, it is seeded verbatim into EVERY free-text
    surface issue #188 added (plus the two pre-existing ones), for
    :func:`test_deterministic_backstop_catches_every_surface`. *alias* is a
    stable per-deal entity-registry-style pseudonym embedded alongside —
    always present, to prove step 1 never touches it.
    """
    residue_suffix = f" (mentions {real_name})" if real_name else ""

    return {
        "opf_version": "0.2",
        "agreement_type": {"id": "test-agreement", "name": "Test Agreement"},
        "taxonomy": {
            "source": "test",
            "entries": [{"id": "indemnification", "label": "Indemnification", "status": "active"}],
        },
        "perspective": {
            "party": real_name or "Our Real Company Name Inc.",
            "counterparty_type": "Educational Institution",
        },
        "baseline": {
            "has_canonical_template": True,
            "template_ref": {
                "document_id": "template",
                "title": f"Template MSA{residue_suffix}",
                "source": "file:///dms/templates/msa-template.docx",
                "sha256": f"sha256:{_FAKE_HASH}",
            },
        },
        "posture": {
            "system_prompt": (
                f"Push back when counterparty proposes uncapped liability{residue_suffix}."
            ),
            "version": 1,
            "generation": {
                "generated_by": "engine",
                "generated_at": "2026-07-01T00:00:00Z",
                "interview": [
                    {
                        "q": "q1",
                        "question": "What is our risk tolerance?",
                        "answer": f"Low tolerance for uncapped liability{residue_suffix}.",
                    }
                ],
            },
        },
        "floor": {
            "invariants": [
                {
                    "id": "inv.cap",
                    "statement": "Liability must be capped at fees paid.",
                    "rationale": f"Board-mandated risk ceiling{residue_suffix}.",
                }
            ]
        },
        "curation": {
            "pins": [
                {
                    "clause_id": "clause.indemnification",
                    "item_id": "C1",
                    "position": "hold firm",
                    "baseline_stance": "no_signal",
                    "pinned_at": "2026-07-01T00:00:00Z",
                    "comment": f"Attorney override{residue_suffix}.",
                }
            ]
        },
        "evidence": {
            "clauses": [
                {
                    "id": "clause.indemnification",
                    "taxonomy_id": "indemnification",
                    "title": "Indemnification",
                    "our_standard": {
                        "text": f"Each party shall indemnify the other{residue_suffix}.",
                        "source_ref": {
                            "document_id": "template",
                            "version": "template",
                            "clause_path": "8",
                        },
                    },
                    "observed_positions": [
                        {
                            "text_summary": f"{alias} demanded a mutual carve-out.",
                            "full_text": f"{alias} demanded a mutual carve-out.",
                            "example_ref": {
                                "document_id": "counterparty-deal-1",
                                "version": 1,
                                "clause_path": "8",
                            },
                            "deviation": "substantive",
                            "risk_delta": {"direction": "worse", "magnitude": "minor"},
                            "provenance": "counterparty_paper",
                            "outcome": "signed",
                            "observed_at": "2023-06-15",
                            "counterparty_ref": {"alias": alias, "counterparty_type": "University"},
                        },
                        {
                            "text_summary": "A second counterparty insisted on capping liability.",
                            "full_text": "A second counterparty insisted on capping liability.",
                            "example_ref": {
                                "document_id": "counterparty-deal-1",
                                "version": 1,
                                "clause_path": "9",
                            },
                            "deviation": "none",
                            "risk_delta": {"direction": "neutral", "magnitude": "none"},
                            "provenance": "counterparty_paper",
                            "outcome": "signed",
                            "observed_at": "2022-01-10",
                        },
                    ],
                    "negotiation_trail": [
                        {
                            "document_id": "counterparty-deal-1",
                            "round": 1,
                            "moved_by": "counterparty",
                            "change_summary": "Cap raised from 1x fees to 2x fees.",
                            "ref": {
                                "document_id": "counterparty-deal-1",
                                "version": 1,
                                "clause_path": "8",
                            },
                        }
                    ],
                    "summary": {
                        "historical_stance": "no_signal",
                        "confidence": {"score": 0.5, "n_our_paper": 0, "n_counterparty_paper": 2},
                    },
                }
            ],
            "clause_library": [
                {
                    "concept_id": "concept.indemnification.mutual",
                    "taxonomy_id": "indemnification",
                    "description": f"Mutual indemnification clause{residue_suffix}.",
                    "notes": f"Observed in one deal so far{residue_suffix}.",
                    "accepted_forms": [
                        {
                            "text_summary": "Mutual indemnification, capped at fees paid.",
                            "example_ref": {
                                "document_id": "counterparty-deal-1",
                                "version": 1,
                                "clause_path": "8",
                            },
                            "provenance": "counterparty_paper",
                        }
                    ],
                }
            ],
        },
        "corpus": {
            "documents": [
                {
                    "document_id": "counterparty-deal-1",
                    "title": f"Master Services Agreement{residue_suffix}",
                    "provenance": "counterparty_paper",
                    "in_scope": True,
                    "versions": 1,
                    "version_files": [
                        {
                            "version": 1,
                            "sha256": f"sha256:{_FAKE_HASH}",
                            "media_type": "application/pdf",
                            "source_uri": "file:///dms/matters/12345/deal-1/v1.pdf",
                        }
                    ],
                }
            ],
            "stats": {"documents_total": 1, "documents_in_scope": 1, "versions_total": 1},
        },
        "compiler": {
            "name": "playbook-engine",
            "version": "0.1.0",
            "generated_at": "2026-07-01T00:00:00Z",
            "stub_basis_present": False,
        },
        "identity": {
            "content_hash": f"sha256:{_FAKE_HASH}",
            "section_digests": {
                "evidence": f"sha256:{_FAKE_HASH}",
                "posture": f"sha256:{_FAKE_HASH}",
                "floor": f"sha256:{_FAKE_HASH}",
                "curation": f"sha256:{_FAKE_HASH}",
            },
        },
    }


# ---------------------------------------------------------------------------
# Fake judges
# ---------------------------------------------------------------------------


class _CleanRedactionJudge:
    """Never flags anything (used once residue has already been dealt with)."""

    def evaluate_batch(self, samples):  # noqa: ANN001
        return [
            RedactionFinding(path=s.path, has_residue=False, rationale="No residue found.")
            for s in samples
        ]


class _CleanVerifyJudge:
    def evaluate_batch(self, samples):  # noqa: ANN001
        return [
            VerifyFinding(path=s.path, leaked=False, rationale="Independently confirmed clean.")
            for s in samples
        ]


class _FlaggingVerifyJudge:
    def __init__(self, leak_path: str):
        self._leak_path = leak_path

    def evaluate_batch(self, samples):  # noqa: ANN001
        return [
            VerifyFinding(
                path=s.path,
                leaked=(s.path == self._leak_path),
                rationale=(
                    "Still identifies the counterparty." if s.path == self._leak_path else "Clean."
                ),
            )
            for s in samples
        ]


class _NeverCallJudge:
    def evaluate_batch(self, samples):  # noqa: ANN001
        raise AssertionError(
            "the deterministic backstop must short-circuit before any judge is called"
        )


def _walk_strings(node):
    if isinstance(node, str):
        yield node
    elif isinstance(node, dict):
        for v in node.values():
            yield from _walk_strings(v)
    elif isinstance(node, list):
        for v in node:
            yield from _walk_strings(v)


def _contains_key(node, key: str) -> bool:
    if isinstance(node, dict):
        if key in node:
            return True
        return any(_contains_key(v, key) for v in node.values())
    if isinstance(node, list):
        return any(_contains_key(v, key) for v in node)
    return False


# ---------------------------------------------------------------------------
# 1. deterministic backstop catches a real name on EVERY seeded surface
# ---------------------------------------------------------------------------


def test_deterministic_backstop_catches_every_surface() -> None:
    doc = _make_doc(real_name=REAL_NAME)

    with pytest.raises(PublishError) as exc_info:
        publish_playbook(
            doc,
            redaction_judge=_NeverCallJudge(),
            verify_judge=_NeverCallJudge(),
            known_entity_names=[REAL_NAME],
            published_at="2026-07-13T00:00:00Z",
        )

    message = str(exc_info.value)
    expected_surfaces = [
        "posture.system_prompt",
        "posture.generation.interview[0].answer",
        "floor.invariants[0].rationale",
        "curation.pins[0].comment",
        "evidence.clauses[0].our_standard.text",
        "evidence.clause_library[0].description",
        "evidence.clause_library[0].notes",
        "corpus.documents[0].title",
        "baseline.template_ref.title",
    ]
    for surface in expected_surfaces:
        assert surface in message, f"missing {surface!r} in backstop failure message: {message}"


# ---------------------------------------------------------------------------
# 2. happy path
# ---------------------------------------------------------------------------


def test_publish_happy_path() -> None:
    doc = _make_doc()
    original = copy.deepcopy(doc)

    report = publish_playbook(
        doc,
        redaction_judge=_CleanRedactionJudge(),
        verify_judge=_CleanVerifyJudge(),
        known_entity_names=[],
        published_at="2026-07-13T00:00:00Z",
    )

    assert report.doc["perspective"]["party"] == DEFAULT_PARTY_LABEL == "the company"
    assert not _contains_key(report.doc, "source_uri")
    assert "source" not in report.doc["baseline"]["template_ref"]

    for clause in report.doc["evidence"]["clauses"]:
        for obs in clause["observed_positions"]:
            observed_at = obs.get("observed_at")
            assert observed_at is not None
            assert re.match(r"^\d{4}-Q[1-4]$", observed_at), observed_at

    assert report.doc["identity"]["content_hash"] != original["identity"]["content_hash"]
    assert report.doc["identity"]["supersedes"] == original["identity"]["content_hash"]
    assert report.doc["x_publication"] == {
        "profile": "public",
        "published_at": "2026-07-13T00:00:00Z",
    }
    # The input doc is never mutated.
    assert doc == original


# ---------------------------------------------------------------------------
# 3. per-deal numbered aliases survive unchanged (NOT collapsed to the label)
# ---------------------------------------------------------------------------


def test_per_deal_aliases_preserved() -> None:
    doc = _make_doc(alias="Counterparty-7")

    report = publish_playbook(
        doc,
        redaction_judge=_CleanRedactionJudge(),
        verify_judge=_CleanVerifyJudge(),
        known_entity_names=[],
        published_at="2026-07-13T00:00:00Z",
    )

    clause = report.doc["evidence"]["clauses"][0]
    tagged_obs = clause["observed_positions"][0]
    assert tagged_obs["counterparty_ref"]["alias"] == "Counterparty-7"
    assert tagged_obs["text_summary"] == "Counterparty-7 demanded a mutual carve-out."
    assert tagged_obs["full_text"] == "Counterparty-7 demanded a mutual carve-out."

    # Meanwhile a GENERIC (unnumbered) mention in free text IS normalized to
    # the counterparty label — proving the two are distinguished, not that
    # nothing ever changes.
    assert report.doc["posture"]["system_prompt"] == (
        f"Push back when {DEFAULT_COUNTERPARTY_LABEL} proposes uncapped liability."
    )


# ---------------------------------------------------------------------------
# 4. --keep-dates leaves ISO dates intact
# ---------------------------------------------------------------------------


def test_keep_dates_flag() -> None:
    doc = _make_doc()

    report = publish_playbook(
        doc,
        redaction_judge=_CleanRedactionJudge(),
        verify_judge=_CleanVerifyJudge(),
        known_entity_names=[],
        published_at="2026-07-13T00:00:00Z",
        keep_dates=True,
    )

    dates = [
        obs["observed_at"]
        for clause in report.doc["evidence"]["clauses"]
        for obs in clause["observed_positions"]
        if obs.get("observed_at")
    ]
    assert dates == ["2023-06-15", "2022-01-10"]


# ---------------------------------------------------------------------------
# 5. a residue finding blocks publish unless accept_residue_risk=True
# ---------------------------------------------------------------------------


def test_residue_finding_blocks_without_flag() -> None:
    doc = _make_doc()
    leak_path = "clauses[clause.indemnification].observed_positions[0].text_summary"

    with pytest.raises(PublishError, match="leaking semantic"):
        publish_playbook(
            doc,
            redaction_judge=_CleanRedactionJudge(),
            verify_judge=_FlaggingVerifyJudge(leak_path=leak_path),
            known_entity_names=[],
            published_at="2026-07-13T00:00:00Z",
        )

    report = publish_playbook(
        doc,
        redaction_judge=_CleanRedactionJudge(),
        verify_judge=_FlaggingVerifyJudge(leak_path=leak_path),
        known_entity_names=[],
        published_at="2026-07-13T00:00:00Z",
        accept_residue_risk=True,
    )
    assert len(report.leaked) == 1
    assert report.leaked[0].path == leak_path


# ---------------------------------------------------------------------------
# 6. the published document validates
# ---------------------------------------------------------------------------


def test_publish_output_validates() -> None:
    doc = _make_doc()

    report = publish_playbook(
        doc,
        redaction_judge=_CleanRedactionJudge(),
        verify_judge=_CleanVerifyJudge(),
        known_entity_names=[],
        published_at="2026-07-13T00:00:00Z",
    )

    result = validate_document(report.doc)
    assert result.ok, [str(e) for e in result.errors if e.blocking]


# ---------------------------------------------------------------------------
# 7. list-independent proper-noun residue sweep (issue #211)
# ---------------------------------------------------------------------------


def _doc_with_prose(text: str) -> dict[str, Any]:
    """Minimal schema-agnostic doc carrying *text* in a scanned free-text field."""
    return {
        "opf_version": "0.2",
        "evidence": {
            "clauses": [{"id": "c1", "our_standard": {"text": text}}],
        },
    }


def test_proper_noun_residue_flags_unknown_names_without_a_list() -> None:
    """The sweep needs no entity list: it surfaces name-shaped strings so a
    reviewer can confirm none is a counterparty (issue #211)."""
    from playbook_engine.publisher import proper_noun_residue

    doc = _doc_with_prose(
        "This Agreement is governed by the laws of the State of New York. "
        "Alpha University shall indemnify the Provider. Ashland signed."
    )
    texts = [f.text for f in proper_noun_residue(doc)]

    assert "Alpha University" in texts
    assert "Ashland" in texts
    # A sentence boundary must split names, not merge them across the period.
    assert "State of New York" in texts
    assert not any("Alpha" in t and "New York" in t for t in texts)


def test_proper_noun_residue_drops_boilerplate_and_role_labels() -> None:
    from playbook_engine.publisher import proper_noun_residue

    doc = _doc_with_prose(
        "This Agreement is confidential. The Provider and the Institution agree. "
        "The Effective Date is fixed. Acme Corp conceded."
    )
    # Default run: boilerplate/role words gone; the one real name remains.
    assert [f.text for f in proper_noun_residue(doc)] == ["Acme Corp"]
    # When "Acme Corp" IS the configured party label, it is excluded too.
    assert proper_noun_residue(doc, party_label="Acme Corp") == ()


def test_proper_noun_residue_dedups_with_counts_and_paths() -> None:
    from playbook_engine.publisher import proper_noun_residue

    doc = {
        "opf_version": "0.2",
        "evidence": {
            "clauses": [
                {
                    "id": "c1",
                    "our_standard": {"text": "Ashland proposed the change."},
                    "observed_positions": [{"full_text": "Ashland later signed."}],
                }
            ]
        },
    }
    findings = proper_noun_residue(doc)
    assert len(findings) == 1
    assert findings[0].text == "Ashland"
    assert findings[0].count == 2
    assert len(findings[0].sample_paths) == 2


def test_publish_report_carries_proper_noun_findings() -> None:
    """End to end: a surviving unknown name shows up on the PublishReport (it is
    advisory — it does NOT block, unlike a KNOWN-name backstop hit)."""
    doc = _make_doc()
    # Inject an unknown name into a free-text surface the sweep scans. It is
    # NOT in known_entity_names, so step-4's backstop cannot catch it — the
    # proper-noun sweep is the layer that surfaces it.
    doc["evidence"]["clauses"][0]["our_standard"]["text"] += " Ashland University drafted this."

    report = publish_playbook(
        doc,
        redaction_judge=_CleanRedactionJudge(),
        verify_judge=_CleanVerifyJudge(),
        known_entity_names=[],
        published_at="2026-07-13T00:00:00Z",
    )

    texts = [f.text for f in report.proper_noun_findings]
    assert "Ashland University" in texts
    # Advisory only — publish still succeeded and produced a doc.
    assert report.doc is not None
    assert "proper_noun_findings" in report.to_dict()
