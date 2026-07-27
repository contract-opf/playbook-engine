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
    _apply_redact_terms,
    _scrub_publication_noise,
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
# 1b. the backstop is re-run on the doc step 5's judge rewrote (issue #50):
#     a redaction judge's rewritten_text can itself reintroduce a known real
#     entity name, and that must be caught even though step 4 already ran
#     (and passed) on the PRE-judge doc.
# ---------------------------------------------------------------------------


class _ReintroducingRedactionJudge:
    """Flags exactly one sample and "rewrites" it to text that itself
    reintroduces a known real entity name — simulating a judge that
    hallucinates or echoes the very name it was meant to scrub (issue #50)."""

    def __init__(self, target_path: str, rewritten_text: str):
        self._target_path = target_path
        self._rewritten_text = rewritten_text

    def evaluate_batch(self, samples):  # noqa: ANN001
        findings = []
        for s in samples:
            if s.path == self._target_path:
                findings.append(
                    RedactionFinding(
                        path=s.path,
                        has_residue=True,
                        rationale="Rewritten for residue.",
                        rewritten_text=self._rewritten_text,
                    )
                )
            else:
                findings.append(
                    RedactionFinding(path=s.path, has_residue=False, rationale="No residue found.")
                )
        return findings


def test_step5_judge_rewrite_reintroducing_known_name_blocks_publish() -> None:
    doc = _make_doc()
    target_path = "posture.system_prompt"
    # Deliberately NOT an institution-name shape (no "University"/"College"/
    # "Regents"/"Board of Trustees") and not an address/contact pattern, so
    # this proves the step-4b deterministic re-scan catches it — not that the
    # unrelated shape-based 5.5/5.6 gates happen to also fire on this name.
    reintroduced_name = "Brightwater Logistics Inc."

    with pytest.raises(PublishError, match="step-5 judge rewrite"):
        publish_playbook(
            doc,
            redaction_judge=_ReintroducingRedactionJudge(
                target_path=target_path,
                rewritten_text=f"Push back when {reintroduced_name} proposes uncapped liability.",
            ),
            # A clean verify judge proves the deterministic backstop — not
            # the best-effort, bypassable verify pass — is what catches this.
            verify_judge=_CleanVerifyJudge(),
            known_entity_names=[reintroduced_name],
            published_at="2026-07-13T00:00:00Z",
        )


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
    # Inject an unknown name into a free-text surface the sweep scans. A
    # signatory-style PERSONAL name is advisory: not in known_entity_names, and
    # not an institution shape, so neither step-4 nor the step-5.5 gate blocks —
    # the proper-noun sweep is the layer that surfaces it. (An institution name
    # like "Ashland University" would now HARD-block; see the gate tests below.)
    doc["evidence"]["clauses"][0]["our_standard"]["text"] += " Dana Ashland drafted this."

    report = publish_playbook(
        doc,
        redaction_judge=_CleanRedactionJudge(),
        verify_judge=_CleanVerifyJudge(),
        known_entity_names=[],
        published_at="2026-07-13T00:00:00Z",
    )

    texts = [f.text for f in report.proper_noun_findings]
    assert "Dana Ashland" in texts
    # Advisory only — publish still succeeded and produced a doc.
    assert report.doc is not None
    assert "proper_noun_findings" in report.to_dict()


# ---------------------------------------------------------------------------
# Issue #234: publication-noise scrub + GC redact list
# ---------------------------------------------------------------------------


def _publish_with_text(text: str, redact_terms: list[str] | None = None) -> dict:
    doc = _make_doc()
    doc["evidence"]["clauses"][0]["observed_positions"][0]["full_text"] = text
    report = publish_playbook(
        doc,
        redaction_judge=_CleanRedactionJudge(),
        verify_judge=_CleanVerifyJudge(),
        known_entity_names=[],
        published_at="2026-07-16T00:00:00Z",
        redact_terms=redact_terms or (),
    )
    return report.doc["evidence"]["clauses"][0]["observed_positions"][0]["full_text"]  # type: ignore[no-any-return]


def test_publish_strips_esign_audit_lines() -> None:
    text = (
        "Notices shall be sent to the parties.\n"
        "DocuSigned by: Pat Example\n"
        "Envelope Id: CBJCHBCAABAAdvXXQkglEeVYM\n"
        "Signed 07/01/2024 10:32:11 AM PDT\n"
        "IP Address: 10.0.0.1\n"
        "The remainder of the clause survives."
    )
    out = _publish_with_text(text)
    assert "Notices shall be sent" in out
    assert "remainder of the clause survives" in out
    assert "DocuSigned" not in out
    assert "Envelope" not in out
    assert "IP Address" not in out
    assert "10:32:11" not in out


def test_publish_scrub_does_not_drop_ambiguous_contract_language() -> None:
    """issue #43: the e-sign/opaque-token markers were unanchored enough to
    drop real contract sentences (and redact a real contract term) wholesale.
    Every contract sentence below must survive verbatim; the accompanying
    genuine DocuSign audit block must still be scrubbed.
    """
    contract_sentences = [
        "Policyholder: obligations survive termination.",
        "The Certificate Holder: shall be named as additional insured.",
        "Each record shall bear a timestamp for audit purposes.",
        "Signed counterparts delivered by 5:00 pm shall be effective.",
        "This is a Work-Made-For-Hire under the Most-Favored-Nation clause.",
    ]
    audit_block = [
        "Envelope Id: CBJCHBCAABAAdvXXQkglEeVYM",
        "Signer Events",
        "Signed 7/1/2024 3:02:11 PM",
        "Holder: Pat Example",
        "Time Stamp: 7/1/2024 3:02:11 PM",
    ]
    text = "\n".join(contract_sentences + audit_block)
    out = _publish_with_text(text)

    for sentence in contract_sentences:
        assert sentence in out, f"contract sentence was corrupted: {sentence!r}"
    assert "Work-Made-For-Hire" in out
    assert "Most-Favored-Nation" in out

    assert "Envelope Id" not in out
    assert "Signer Events" not in out
    assert "3:02:11" not in out
    assert "Holder: Pat Example" not in out
    assert "Time Stamp: 7/1/2024 3:02:11 PM" not in out


def test_publish_redacts_opaque_token_in_ordinary_prose() -> None:
    """issue #43: the mixed-case-plus-digit tightening on ``_OPAQUE_TOKEN_RE``
    must still catch an envelope-id/base64-shaped token that appears in a
    plain prose line with no e-sign vendor marker nearby (so the line isn't
    dropped by ``_ESIGN_LINE_RE`` instead) — the redaction must come from the
    opaque-token rule itself, not incidentally from the line-drop rule.
    """
    text = (
        "The parties executed this Agreement under envelope CBJCHBCAABAAdvXXQkglEeVYM7Vg on file."
    )
    out = _publish_with_text(text)
    assert "CBJCHBCAABAAdvXXQkglEeVYM7Vg" not in out
    assert "[redacted]" in out


def test_publish_redacts_address_spans() -> None:
    text = (
        "Notices to the Institution at 1234 Campus Garden Lane, Springfield, "
        "IL 62704, Suite 400, or P.O. Box 6186, with a copy to Legal."
    )
    out = _publish_with_text(text)
    assert "1234" not in out
    assert "Campus Garden Lane" not in out
    assert "62704" not in out
    assert "P.O. Box 6186" not in out
    assert "[address redacted]" in out
    assert "with a copy to Legal" in out


def test_publish_never_scrubs_content_addresses() -> None:
    doc = _make_doc()
    report = publish_playbook(
        doc,
        redaction_judge=_CleanRedactionJudge(),
        verify_judge=_CleanVerifyJudge(),
        known_entity_names=[],
        published_at="2026-07-16T00:00:00Z",
    )
    published = report.doc
    for document in published["corpus"]["documents"]:
        for vf in document.get("version_files", []):
            assert vf["sha256"].startswith("sha256:")
            assert "[redacted]" not in vf["sha256"]
    assert published["identity"]["content_hash"].startswith("sha256:")


def test_publish_redact_terms_replace_and_join_backstop() -> None:
    # The institution name must be on the redact list now — the step-5.5 gate
    # hard-blocks a surviving "... University". With it listed, redaction takes
    # it out (joining the backstop) while a non-listed term ("Provost") stays.
    text = "Signature: Pat Q. Example, Provost, Example State University (KSU)."
    out = _publish_with_text(
        text, redact_terms=["Pat Q. Example", "Example State University", "KSU"]
    )
    assert "Pat Q. Example" not in out
    assert "KSU" not in out
    assert "State University" not in out
    assert out.count("[redacted]") >= 2
    assert "Provost" in out  # a non-listed term still survives


def test_publish_version_ingest_stems_become_ordinals() -> None:
    doc = _make_doc()
    doc["corpus"]["documents"][0]["version_ingest"] = [
        {"version": "01__Real Filename Stem- Some University (002)", "status": "ok"},
        {"version": "02__Real Filename Stem- Some University (002)", "status": "ok"},
    ]
    report = publish_playbook(
        doc,
        redaction_judge=_CleanRedactionJudge(),
        verify_judge=_CleanVerifyJudge(),
        known_entity_names=[],
        published_at="2026-07-16T00:00:00Z",
    )
    ingests = report.doc["corpus"]["documents"][0]["version_ingest"]
    assert [i["version"] for i in ingests] == ["v1", "v2"]


def test_publish_scrubs_emails_uuids_urls() -> None:
    text = (
        "Contact pat.example@someuniversity.edu with questions.\n"
        "Transaction ref 21345135-f8f5-4497-a929-6e64db805306 follows.\n"
        "See https://dms.example.edu/contracts/123 and www.example.edu for details.\n"
        "The clause text itself survives."
    )
    out = _publish_with_text(text)
    assert "someuniversity" not in out
    assert "pat.example" not in out
    assert "21345135" not in out
    assert "https://" not in out and "www.example.edu" not in out
    assert "The clause text itself survives." in out


def test_publish_redact_terms_match_across_punctuation() -> None:
    """A redact term must hit slugs and e-mail localparts, not just prose —
    the same normalization class as the step-4 backstop."""
    text = (
        "Deal id fwd-legaladmin-counterparty-9-chapel-hill-agreem-773a "
        "and mail to amanda.wynn@example.com about it."
    )
    out = _publish_with_text(text, redact_terms=["Chapel Hill", "Amanda Wynn"])
    assert "chapel-hill" not in out and "Chapel Hill" not in out
    assert "amanda" not in out.lower()
    assert "counterparty-9" in out  # surrounding slug context survives


def test_backstop_and_redaction_cover_dict_keys() -> None:
    """document_id slugs appear as dict KEYS in stats tallies — a
    counterparty fragment there must trip the backstop, and a redact term
    must rewrite it consistently."""
    doc = _make_doc()
    doc["corpus"]["stats"] = {"observations_by_document": {"deal-somewhere-chapel-hill-773a": 3}}

    # 1. backstop sees the key
    with pytest.raises(PublishError, match="chapel"):
        publish_playbook(
            doc,
            redaction_judge=_NeverCallJudge(),
            verify_judge=_NeverCallJudge(),
            known_entity_names=["Chapel Hill"],
            published_at="2026-07-17T00:00:00Z",
        )

    # 2. a redact term rewrites the key, so the backstop then passes
    report = publish_playbook(
        doc,
        redaction_judge=_CleanRedactionJudge(),
        verify_judge=_CleanVerifyJudge(),
        known_entity_names=["Chapel Hill"],
        published_at="2026-07-17T00:00:00Z",
        redact_terms=["Chapel Hill"],
    )
    keys = list(report.doc["corpus"]["stats"]["observations_by_document"])
    assert keys == ["deal-somewhere-[redacted]-773a"]


def test_redact_terms_with_punctuation_match_normalized_text() -> None:
    """A term written with punctuation ('Kansas City, Missouri') must match
    however the extraction rendered it (double spaces, no comma, case)."""
    text = "THE JUNIOR COLLEGE DISTRICT OF METROPOLITAN  KANSAS CITY MISSOURI agrees."
    out = _publish_with_text(
        text, redact_terms=["Junior College District of Metropolitan Kansas City, Missouri"]
    )
    assert "KANSAS" not in out and "Kansas" not in out
    assert "[redacted] agrees." in out


# ---------------------------------------------------------------------------
# Final institution-identity gate (list-independent, full-surface, hard):
# the fix for the 2026-07-22 public example-playbook leak class — a real
# counterparty name that was never registered, surviving in extracted prose,
# a stats dict key, or a filename-derived document_id slug.
# ---------------------------------------------------------------------------

_INST_NAME = "Wexford University"  # fictional; registered NOWHERE


def _doc_leaking_institution_everywhere() -> dict[str, Any]:
    """A clean base doc with a fictional institution name planted in the three
    surfaces the born-safe pass historically missed: signature-block prose, a
    corpus.stats dict KEY, and a filename-derived document_id slug value."""
    doc = _make_doc()
    doc["evidence"]["clauses"][0]["observed_positions"][0]["full_text"] = (
        f"IN WITNESS WHEREOF, signed on behalf of {_INST_NAME} by its officer."
    )
    slug = "affiliation-agreement-wexford-university-0212e146"
    doc["corpus"]["documents"][0]["document_id"] = slug
    doc["corpus"]["stats"]["observations_by_document"] = {slug: 3}
    return doc


def test_institution_gate_blocks_unregistered_name_across_surfaces() -> None:
    """An institution name never given to the backstop still HARD-blocks —
    whether it hides in prose, a dict key, or a document_id slug."""
    doc = _doc_leaking_institution_everywhere()

    with pytest.raises(PublishError, match="institution") as exc_info:
        publish_playbook(
            doc,
            redaction_judge=_CleanRedactionJudge(),
            verify_judge=_CleanVerifyJudge(),
            known_entity_names=[],  # NOT registered — step-4 backstop is blind to it
            published_at="2026-07-22T00:00:00Z",
        )
    message = str(exc_info.value)
    assert "wexford university" in message.lower()
    # every leaking surface is named in the failure
    assert "observed_positions" in message
    assert "document_id" in message
    assert "observations_by_document" in message


def test_institution_gate_passes_once_name_is_redacted() -> None:
    """Naming the survivor in redact_terms clears the gate on every surface —
    prose, key, and slug transform identically, so publish succeeds clean."""
    doc = _doc_leaking_institution_everywhere()

    report = publish_playbook(
        doc,
        redaction_judge=_CleanRedactionJudge(),
        verify_judge=_CleanVerifyJudge(),
        known_entity_names=[],
        published_at="2026-07-22T00:00:00Z",
        redact_terms=[_INST_NAME],
    )
    blob = " ".join(_walk_strings(report.doc))
    assert "wexford" not in blob.lower()
    # the counterparty fragment is gone from BOTH the slug value and the key
    stats_keys = list(report.doc["corpus"]["stats"]["observations_by_document"])
    assert stats_keys == ["affiliation-agreement-[redacted]-0212e146"]
    assert (
        report.doc["corpus"]["documents"][0]["document_id"]
        == "affiliation-agreement-[redacted]-0212e146"
    )


# ---------------------------------------------------------------------------
# Issue #33: concatenated (no-separator / camelCase-derived) entity-name
# slugs evade every identity layer's word-boundary matching. A DMS folder
# named "WestmoorUniversity 2023" becomes document_id
# "westmooruniversity-2023" (lowercased, hyphen-joined) — there is no
# separator between "westmoor" and "university" for a word-boundary check to
# anchor on, so the token-match pseudonymizer, the padded-substring step-4
# backstop, and the institution-name regexes all miss it even though the
# name IS registered.
# ---------------------------------------------------------------------------

_CONCAT_INST_NAME = "Westmoor University"  # fictional; illustrative per issue #33
_CONCAT_SLUG = "westmooruniversity-2023"


def test_backstop_catches_concatenated_slug_value_and_dict_key() -> None:
    """A no-separator document_id slug must trip the step-4 backstop, both
    as a value and as a corpus.stats dict key, once the name is registered
    — even though it has no word boundary for the padded-substring check."""
    doc = _make_doc()
    doc["corpus"]["documents"][0]["document_id"] = _CONCAT_SLUG
    doc["corpus"]["stats"] = {"observations_by_document": {_CONCAT_SLUG: 1}}

    with pytest.raises(PublishError, match="westmoor"):
        publish_playbook(
            doc,
            redaction_judge=_NeverCallJudge(),
            verify_judge=_NeverCallJudge(),
            known_entity_names=[_CONCAT_INST_NAME],
            published_at="2026-07-27T00:00:00Z",
        )


def test_backstop_ignores_short_names_in_concatenated_form() -> None:
    """The collapsed-form check requires a length floor so a short registered
    name (e.g. an acronym) doesn't false-positive on an ordinary substring."""
    doc = _make_doc()
    doc["corpus"]["documents"][0]["document_id"] = "abc-glossary-2023"

    report = publish_playbook(
        doc,
        redaction_judge=_CleanRedactionJudge(),
        verify_judge=_CleanVerifyJudge(),
        known_entity_names=["A B C"],  # collapses to "abc" — only 3 chars, below the floor
        published_at="2026-07-27T00:00:00Z",
    )
    assert report.doc["corpus"]["documents"][0]["document_id"] == "abc-glossary-2023"


def test_apply_redact_terms_rewrites_concatenated_slug() -> None:
    """The documented remedy — add the name to --redact-terms and re-run —
    must actually redact the no-separator (concatenated) slug form, not just
    the punctuated/spaced form the pre-fix ``[\\W_]+`` joiner required."""
    doc = _make_doc()
    doc["corpus"]["documents"][0]["document_id"] = _CONCAT_SLUG
    doc["corpus"]["stats"] = {"observations_by_document": {_CONCAT_SLUG: 1}}

    redacted = _apply_redact_terms(doc, [_CONCAT_INST_NAME])
    assert redacted["corpus"]["documents"][0]["document_id"] == "[redacted]-2023"
    assert list(redacted["corpus"]["stats"]["observations_by_document"]) == ["[redacted]-2023"]

    # End to end: once the redact term is supplied, publish succeeds clean —
    # the slug no longer contains the registered name in any form.
    report = publish_playbook(
        doc,
        redaction_judge=_CleanRedactionJudge(),
        verify_judge=_CleanVerifyJudge(),
        known_entity_names=[_CONCAT_INST_NAME],
        published_at="2026-07-27T00:00:00Z",
        redact_terms=[_CONCAT_INST_NAME],
    )
    assert report.doc["corpus"]["documents"][0]["document_id"] == "[redacted]-2023"


def test_institution_gate_ignores_governing_law_and_generic_descriptors() -> None:
    """The gate must NOT fire on governing-law states or generic org
    descriptors — those stay the advisory sweep's job, so a clean publish is
    never falsely blocked."""
    doc = _make_doc()
    doc["evidence"]["clauses"][0]["our_standard"]["text"] = (
        "This Agreement is governed by the laws of the State of New York. "
        "The University shall provide services through its College of Health "
        "Professions and the Board of Directors shall meet annually."
    )
    report = publish_playbook(
        doc,
        redaction_judge=_CleanRedactionJudge(),
        verify_judge=_CleanVerifyJudge(),
        known_entity_names=[],
        published_at="2026-07-22T00:00:00Z",
    )
    assert report.doc is not None  # no PublishError


def test_publish_scrub_redacts_city_full_state_zip() -> None:
    """A notice block written 'City, <FullStateName> <ZIP>' (which the
    two-letter ``ST`` rule misses) is redacted; the gate confirms none
    survives."""
    text = "Notices to the Institution at New York, New York 10017, attn: Legal."
    out = _publish_with_text(text)
    assert "10017" not in out
    assert "[address redacted]" in out
    assert "attn: Legal" in out


# ---------------------------------------------------------------------------
# Final contact-identity gate (list-independent, full-surface, hard): phones /
# e-mail domains / URLs — the contact-info sibling of the institution gate,
# closing the 2026-07-24 example-playbook leak class (counterparty area codes,
# a redacted-localpart e-mail domain, and a whitespace-normalized URL that the
# scrub's e-mail/URL rules missed).
# ---------------------------------------------------------------------------


def test_publish_scrub_removes_phone_email_domain_and_url() -> None:
    """The scrub strips phone numbers, bare/redacted-localpart e-mail domains,
    and whitespace-normalized URLs from prose — the exact forms the original
    e-mail/URL rules missed."""
    text = (
        "Contact the Institution at (555) 867-5309 or 555-123-4567; "
        "e-mail [redacted]@wexford.edu or @wex; portal https www wexford edu login. "
        "The clause text itself survives."
    )
    out = _publish_with_text(text)
    assert "867-5309" not in out and "555-123-4567" not in out
    assert "wexford.edu" not in out and "@wex" not in out
    assert "https" not in out
    assert "The clause text itself survives." in out


def test_contact_identity_hits_flags_phone_email_url_in_values_and_keys() -> None:
    """The gate detects contact info in string VALUES and dict KEYS alike, and
    does not fire on governing law or a bare emergency number."""
    from playbook_engine.publisher import _contact_identity_hits

    doc = {
        "a": "call (555) 246-8010 today",
        "b": "mail [redacted]@wexford.edu now",
        "c": "see https www example org path",
        "stats": {"[redacted]@wex": 3},  # contact info hiding in a dict KEY
        "safe": "governed by the laws of New York; dial 911 in an emergency",
    }
    hits = _contact_identity_hits(doc)
    kinds = {kind for _p, _m, kind in hits}
    matched = " ".join(m for _p, m, _k in hits).lower()
    assert {"phone", "email", "url"} <= kinds
    assert "246-8010" in matched and "wexford.edu" in matched and "@wex" in matched
    assert "new york" not in matched and "911" not in matched


def test_contact_gate_blocks_email_domain_in_dict_key() -> None:
    """Contact info in a dict KEY survives the value-only scrub, so the
    fail-closed contact gate is the backstop that catches it end-to-end."""
    doc = _make_doc()
    doc["corpus"]["stats"]["observations_by_document"] = {"[redacted]@wexford.edu": 3}
    with pytest.raises(PublishError, match="contact"):
        publish_playbook(
            doc,
            redaction_judge=_CleanRedactionJudge(),
            verify_judge=_CleanVerifyJudge(),
            known_entity_names=[],
            published_at="2026-07-24T00:00:00Z",
        )


def test_contact_gate_does_not_fire_on_clean_prose() -> None:
    """No false positive: '$50 @2x' and a bare '911' are not contact info, so a
    clean publish is never falsely blocked."""
    doc = _make_doc()
    doc["evidence"]["clauses"][0]["our_standard"]["text"] = (
        "Rates apply at $50 @2x after hours; dial 911 in an emergency."
    )
    report = publish_playbook(
        doc,
        redaction_judge=_CleanRedactionJudge(),
        verify_judge=_CleanVerifyJudge(),
        known_entity_names=[],
        published_at="2026-07-24T00:00:00Z",
    )
    assert report.doc is not None  # no PublishError


# ---------------------------------------------------------------------------
# Issue #49: the sha256 exemption in the scrub and both hard identity gates
# was PREFIX-based (``str.startswith("sha256:")``), so any string that merely
# BEGAN with that prefix — not just a bare content-address hash — was exempt
# from scrubbing/scanning wholesale. A clause/exhibit block whose extracted
# full_text opens with a hash label (security schedules, e-signature
# certificates) could smuggle contact info and institution names past every
# fail-closed gate. Only a string that IS EXACTLY "sha256:" + 64 hex chars
# (canonicalize.py's guaranteed shape) may skip.
# ---------------------------------------------------------------------------


def test_sha256_prefix_does_not_exempt_trailing_leak_from_contact_gate() -> None:
    """Pins the CONTACT GATE via a dict KEY — a surface the value-only scrub
    never touches, so this exercises the gate itself, not the scrub. A hash
    LABEL followed by contact info is not a bare content address; it must
    still trip on the trailing digits/domain rather than ride the sha256
    prefix to safety. (The security-schedule/e-sign VALUE shape from the bug
    report, where the scrub runs first, is covered separately by
    ``test_scrub_publication_noise_does_not_exempt_sha256_label_prefix``.)"""
    doc = _make_doc()
    leaking_key = (
        f"sha256:{_FAKE_HASH} shall be delivered to security@westmoor.edu; call 919-555-1234"
    )
    doc["corpus"]["stats"]["observations_by_document"] = {leaking_key: 3}
    with pytest.raises(PublishError, match="contact"):
        publish_playbook(
            doc,
            redaction_judge=_CleanRedactionJudge(),
            verify_judge=_CleanVerifyJudge(),
            known_entity_names=[],
            published_at="2026-07-27T00:00:00Z",
        )


def test_bare_sha256_hash_still_exempt_everywhere() -> None:
    """The narrowed, fullmatch-based exemption still protects an EXACT content
    address end-to-end — identity hashes are unaffected by the fix — while a
    hash immediately followed by more text is no longer treated as bare."""
    from playbook_engine.publisher import _contact_identity_hits, _institution_identity_hits

    bare_hash = f"sha256:{_FAKE_HASH}"
    assert _contact_identity_hits({"a": bare_hash}) == []
    assert _institution_identity_hits({"a": bare_hash}) == []

    # a hash immediately followed by more text (no separating space) is NOT a
    # bare address: the institution gate must still see what comes after it.
    trailing = {"b": f"{bare_hash} University of Westmoor"}
    assert any(kind == "institution" for _p, _m, kind in _institution_identity_hits(trailing))

    report = publish_playbook(
        _make_doc(),
        redaction_judge=_CleanRedactionJudge(),
        verify_judge=_CleanVerifyJudge(),
        known_entity_names=[],
        published_at="2026-07-27T00:00:00Z",
    )
    assert report.doc is not None  # no PublishError; baseline sha256 untouched
    assert report.doc["baseline"]["template_ref"]["sha256"] == bare_hash


def test_scrub_publication_noise_does_not_exempt_sha256_label_prefix() -> None:
    """The security-schedule/e-sign VALUE shape from the bug report: a
    string that BEGINS with "sha256:" but is not a bare content address
    (there's prose after the hash) must still go through the full scrub —
    the e-sign audit line dropped and the contact spans redacted — rather
    than riding the old prefix-based exemption to a byte-identical pass-
    through. Exercises ``_scrub_publication_noise`` directly, not the
    identity gates, since the scrub runs before either gate sees the text
    and previously erased this evidence before the gates ever ran."""
    leaking_text = (
        f"sha256: {_FAKE_HASH} shall be delivered to security@westmoor.edu, "
        "Regents of the University of Westmoor, call 919-555-1234\n"
        "DocuSign Envelope ID: 9AB12CD3-4E56-7890-ABCD-EF1234567890"
    )
    doc = {"clause": {"full_text": leaking_text}}

    scrubbed = _scrub_publication_noise(doc)

    scrubbed_text = scrubbed["clause"]["full_text"]
    assert "DocuSign Envelope ID" not in scrubbed_text
    assert "security@westmoor.edu" not in scrubbed_text
    assert "919-555-1234" not in scrubbed_text
    assert (
        scrubbed_text == f"sha256: {_FAKE_HASH} shall be delivered to [redacted], "
        "Regents of the University of Westmoor, call [redacted]"
    )

    # Paired negative: a bare content address (no trailing text) still skips
    # the scrub wholesale and passes through byte-identical.
    bare_hash = f"sha256:{_FAKE_HASH}"
    assert _scrub_publication_noise({"a": bare_hash})["a"] == bare_hash
