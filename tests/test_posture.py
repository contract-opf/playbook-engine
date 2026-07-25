"""Tests for Posture generation — GC interview -> governed, versioned prose
block (issue #156).

Acceptance criteria verified here (mirrors the issue's Required verification):

  - Given fixture interview answers, the OPF ``posture`` block is populated
    (``system_prompt`` assembled from the answers, ``generation.interview``
    recorded).
  - The Posture is versioned: a first generation starts at ``version=1``;
    re-running the interview against the resulting Posture bumps the version.
  - The SHOULD-warn (``check_posture_floor_conflict``) fires when the Posture
    softens language around a Floor-protected concept, and is wired into
    ``validate_document()`` as a non-blocking warning (never a hard error, per
    the issue's Direction).

SECURITY NOTE: All fixtures are synthetic, minimal dicts/answers — no real
legal text, no real parties.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from playbook_engine.canonicalize import content_hash
from playbook_engine.cli import cli
from playbook_engine.posture import (
    INTERVIEW_QUESTIONS,
    PostureApplyResult,
    PostureError,
    apply_posture_interview,
    check_posture_floor_conflict,
    generate_posture,
)
from playbook_engine.validator import validate_document

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_ANSWERS: dict[str, str] = {
    "rounds": "Usually 2 rounds before escalating.",
    "leverage": "Collaborative; we often want the deal.",
    "risk_appetite": "Default to accept-to-close on non-material changes.",
    "sacred_clauses": "Liability cap and indemnification.",
    "flexible_clauses": "Term length and renewal mechanics.",
    "audience": "Terse rationale for a GC audience.",
}


def _minimal_v02_doc(**overrides: Any) -> dict[str, Any]:
    doc: dict[str, Any] = {
        "opf_version": "0.2",
        "agreement_type": {"id": "test-agreement", "name": "Test Agreement"},
        "baseline": {"has_canonical_template": False},
        "taxonomy": {"source": "custom", "entries": []},
        "evidence": {"clauses": [], "clause_library": []},
        "posture": {},
        "floor": {},
        "corpus": {"documents": [], "stats": {}},
        "compiler": {
            "name": "playbook-engine",
            "version": "0.1.0",
            "run_id": "run-abc",
            "generated_at": "2026-01-01T00:00:00Z",
        },
        "identity": {
            "content_hash": "sha256:" + "0" * 64,
            "section_digests": {
                "evidence": "sha256:" + "1" * 64,
                "posture": "sha256:" + "2" * 64,
                "floor": "sha256:" + "3" * 64,
                "curation": "sha256:" + "4" * 64,
            },
        },
    }
    doc.update(overrides)
    return doc


# ---------------------------------------------------------------------------
# generate_posture — population + versioning
# ---------------------------------------------------------------------------


def test_generate_posture_populates_system_prompt_and_interview() -> None:
    posture = generate_posture(_ANSWERS, generated_at="2026-07-10T00:00:00Z")

    assert posture["version"] == 1
    assert posture["system_prompt"].strip()
    # Every answered question shows up somewhere in the assembled prose.
    for answer in _ANSWERS.values():
        assert answer in posture["system_prompt"]

    interview = posture["generation"]["interview"]
    assert len(interview) == len(_ANSWERS)
    ids = {rec["q"] for rec in interview}
    assert ids == set(_ANSWERS)
    for rec in interview:
        assert rec["question"]
        assert rec["answer"]

    assert posture["generation"]["generated_by"] == "playbook-engine"
    assert posture["generation"]["generated_at"] == "2026-07-10T00:00:00Z"


def test_generate_posture_interview_order_is_canonical_not_dict_order() -> None:
    # Feed answers in reverse-canonical dict-insertion order; the assembled
    # prose must still follow INTERVIEW_QUESTIONS order, not dict order.
    reversed_answers = dict(reversed(list(_ANSWERS.items())))
    posture = generate_posture(reversed_answers, generated_at="2026-07-10T00:00:00Z")

    canonical_ids = [iq.q for iq in INTERVIEW_QUESTIONS if iq.q in _ANSWERS]
    interview_ids = [rec["q"] for rec in posture["generation"]["interview"]]
    assert interview_ids == canonical_ids


def test_generate_posture_first_run_starts_at_version_1() -> None:
    posture = generate_posture(_ANSWERS, generated_at="2026-07-10T00:00:00Z", existing_posture=None)
    assert posture["version"] == 1

    posture_empty_prior = generate_posture(
        _ANSWERS, generated_at="2026-07-10T00:00:00Z", existing_posture={}
    )
    assert posture_empty_prior["version"] == 1


def test_generate_posture_rerun_bumps_version() -> None:
    v1 = generate_posture(_ANSWERS, generated_at="2026-07-10T00:00:00Z")
    v2 = generate_posture(_ANSWERS, generated_at="2026-07-11T00:00:00Z", existing_posture=v1)
    v3 = generate_posture(_ANSWERS, generated_at="2026-07-12T00:00:00Z", existing_posture=v2)

    assert v1["version"] == 1
    assert v2["version"] == 2
    assert v3["version"] == 3


def test_generate_posture_grounded_in_recorded_when_supplied() -> None:
    posture = generate_posture(
        _ANSWERS,
        generated_at="2026-07-10T00:00:00Z",
        grounded_in="evidence@sha256:" + "a" * 64,
    )
    assert posture["generation"]["grounded_in"] == "evidence@sha256:" + "a" * 64


def test_generate_posture_omits_grounded_in_when_not_supplied() -> None:
    posture = generate_posture(_ANSWERS, generated_at="2026-07-10T00:00:00Z")
    assert "grounded_in" not in posture["generation"]


def test_generate_posture_requires_at_least_3_answers() -> None:
    with pytest.raises(PostureError, match="at least 3"):
        generate_posture(
            {"rounds": "2 rounds.", "leverage": "Collaborative."},
            generated_at="2026-07-10T00:00:00Z",
        )


def test_generate_posture_blank_answers_do_not_count_toward_minimum() -> None:
    answers = {**_ANSWERS, "audience": "   "}  # blank after strip
    # Still 5 non-blank answers, so this must succeed and omit the blank one.
    posture = generate_posture(answers, generated_at="2026-07-10T00:00:00Z")
    ids = {rec["q"] for rec in posture["generation"]["interview"]}
    assert "audience" not in ids


def test_generate_posture_rejects_unknown_question_id() -> None:
    answers = {**_ANSWERS, "bogus_question": "nonsense"}
    with pytest.raises(PostureError, match="unrecognized"):
        generate_posture(answers, generated_at="2026-07-10T00:00:00Z")


def test_generate_posture_allows_pruned_subset_of_3() -> None:
    minimal_answers = {
        "rounds": "2 rounds.",
        "leverage": "Collaborative.",
        "risk_appetite": "Accept-to-close on non-material changes.",
    }
    posture = generate_posture(minimal_answers, generated_at="2026-07-10T00:00:00Z")
    assert posture["version"] == 1
    assert len(posture["generation"]["interview"]) == 3


# ---------------------------------------------------------------------------
# check_posture_floor_conflict — deterministic SHOULD-warn
# ---------------------------------------------------------------------------

_LIABILITY_INVARIANT = {
    "id": "no-uncapped-liability",
    "statement": "Never accept uncapped liability on the liability cap.",
    "rationale": "Categorically unacceptable regardless of deal value.",
}


def test_check_posture_floor_conflict_fires_on_softening_language() -> None:
    system_prompt = "The liability cap is flexible to close a deal."
    warnings = check_posture_floor_conflict(system_prompt, [_LIABILITY_INVARIANT])
    assert warnings
    assert "no-uncapped-liability" in warnings[0]


def test_check_posture_floor_conflict_silent_without_softening_language() -> None:
    system_prompt = "Hold firm on the liability cap; see Floor."
    warnings = check_posture_floor_conflict(system_prompt, [_LIABILITY_INVARIANT])
    assert warnings == []


def test_check_posture_floor_conflict_silent_without_concept_overlap() -> None:
    # Softening language present, but about an unrelated concept (renewal
    # terms), not the liability cap the invariant protects.
    system_prompt = "Renewal terms and notice periods are flexible to close a deal."
    warnings = check_posture_floor_conflict(system_prompt, [_LIABILITY_INVARIANT])
    assert warnings == []


def test_check_posture_floor_conflict_no_invariants_no_warnings() -> None:
    system_prompt = "The liability cap is flexible to close a deal."
    assert check_posture_floor_conflict(system_prompt, []) == []
    assert check_posture_floor_conflict(system_prompt, None) == []


def test_check_posture_floor_conflict_empty_prompt_no_warnings() -> None:
    assert check_posture_floor_conflict("", [_LIABILITY_INVARIANT]) == []


# ---------------------------------------------------------------------------
# validator.py wiring — non-blocking, never a hard error
# ---------------------------------------------------------------------------


def test_validator_surfaces_posture_floor_conflict_as_non_blocking_warning() -> None:
    doc = _minimal_v02_doc(
        posture={"system_prompt": "The liability cap is flexible to close a deal."},
        floor={"invariants": [_LIABILITY_INVARIANT]},
    )
    result = validate_document(doc)

    # SHOULD-warn, never a hard error (issue #156 Direction).
    assert result.ok, [str(e) for e in result.errors if e.blocking]
    warn_messages = [e.message for e in result.errors if not e.blocking]
    assert any("no-uncapped-liability" in m for m in warn_messages)


def test_validator_clean_posture_raises_no_warning() -> None:
    doc = _minimal_v02_doc(
        posture={"system_prompt": "Hold firm on the liability cap; see Floor."},
        floor={"invariants": [_LIABILITY_INVARIANT]},
    )
    result = validate_document(doc)
    assert result.ok
    assert result.errors == []


def test_validator_empty_posture_and_floor_still_valid() -> None:
    doc = _minimal_v02_doc()
    result = validate_document(doc)
    assert result.ok
    assert result.errors == []


# ---------------------------------------------------------------------------
# apply_posture_interview — I/O orchestration (read-modify-write)
# ---------------------------------------------------------------------------


def test_apply_posture_interview_writes_versioned_posture(tmp_path: Path) -> None:
    doc = _minimal_v02_doc()
    opf_path = tmp_path / "playbook.opf.json"
    opf_path.write_text(json.dumps(doc), encoding="utf-8")

    result = apply_posture_interview(tmp_path, _ANSWERS, generated_at="2026-07-10T00:00:00Z")

    assert isinstance(result, PostureApplyResult)
    assert result.version == 1
    assert result.warnings == ()
    assert result.path == opf_path

    written = json.loads(opf_path.read_text(encoding="utf-8"))
    assert written["posture"]["version"] == 1
    assert written["posture"]["system_prompt"].strip()
    # grounded_in derived from identity.section_digests.evidence.
    assert written["posture"]["generation"]["grounded_in"] == ("evidence@sha256:" + "1" * 64)


def test_apply_posture_interview_rerun_bumps_version(tmp_path: Path) -> None:
    doc = _minimal_v02_doc()
    opf_path = tmp_path / "playbook.opf.json"
    opf_path.write_text(json.dumps(doc), encoding="utf-8")

    apply_posture_interview(tmp_path, _ANSWERS, generated_at="2026-07-10T00:00:00Z")
    result2 = apply_posture_interview(tmp_path, _ANSWERS, generated_at="2026-07-11T00:00:00Z")

    assert result2.version == 2
    written = json.loads(opf_path.read_text(encoding="utf-8"))
    assert written["posture"]["version"] == 2


def test_apply_posture_interview_refreshes_identity_content_hash(tmp_path: Path) -> None:
    doc = _minimal_v02_doc()
    opf_path = tmp_path / "playbook.opf.json"
    opf_path.write_text(json.dumps(doc), encoding="utf-8")

    apply_posture_interview(tmp_path, _ANSWERS, generated_at="2026-07-10T00:00:00Z")

    written = json.loads(opf_path.read_text(encoding="utf-8"))
    expected_doc = copy.deepcopy(written)
    # content_hash() is a pure function of the doc minus identity/curation —
    # the written identity.content_hash must match recomputing it.
    assert written["identity"]["content_hash"] == content_hash(expected_doc)
    assert written["identity"]["content_hash"] != doc["identity"]["content_hash"]


def test_apply_posture_interview_surfaces_floor_conflict_warning(tmp_path: Path) -> None:
    doc = _minimal_v02_doc(floor={"invariants": [_LIABILITY_INVARIANT]})
    opf_path = tmp_path / "playbook.opf.json"
    opf_path.write_text(json.dumps(doc), encoding="utf-8")

    answers = {**_ANSWERS, "sacred_clauses": "The liability cap is flexible to close a deal."}
    result = apply_posture_interview(tmp_path, answers, generated_at="2026-07-10T00:00:00Z")

    assert result.warnings
    assert any("no-uncapped-liability" in w for w in result.warnings)


def test_apply_posture_interview_missing_playbook_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        apply_posture_interview(tmp_path, _ANSWERS, generated_at="2026-07-10T00:00:00Z")


# ---------------------------------------------------------------------------
# CLI — playbook posture interview / questions
# ---------------------------------------------------------------------------


def _invoke(*args: str) -> tuple[int, str]:
    runner = CliRunner()
    result = runner.invoke(cli, list(args))
    return result.exit_code, result.output


def test_cli_posture_questions_lists_canonical_ids() -> None:
    exit_code, output = _invoke("posture", "questions")
    assert exit_code == 0
    for iq in INTERVIEW_QUESTIONS:
        assert iq.q in output


def test_cli_posture_interview_answers_file_round_trip(tmp_path: Path) -> None:
    doc = _minimal_v02_doc()
    opf_path = tmp_path / "playbook.opf.json"
    opf_path.write_text(json.dumps(doc), encoding="utf-8")

    answers_path = tmp_path / "answers.json"
    answers_path.write_text(json.dumps(_ANSWERS), encoding="utf-8")

    exit_code, output = _invoke(
        "posture", "interview", str(tmp_path), "--answers-file", str(answers_path)
    )
    assert exit_code == 0, output
    assert "posture.version=1" in output

    exit_code2, output2 = _invoke(
        "posture", "interview", str(tmp_path), "--answers-file", str(answers_path)
    )
    assert exit_code2 == 0, output2
    assert "posture.version=2" in output2


def test_cli_posture_interview_missing_out_dir_playbook_fails(tmp_path: Path) -> None:
    answers_path = tmp_path / "answers.json"
    answers_path.write_text(json.dumps(_ANSWERS), encoding="utf-8")

    exit_code, output = _invoke(
        "posture", "interview", str(tmp_path), "--answers-file", str(answers_path)
    )
    assert exit_code != 0
    assert "not found" in output.lower() or "error" in output.lower()


def test_cli_posture_interview_too_few_answers_fails(tmp_path: Path) -> None:
    doc = _minimal_v02_doc()
    opf_path = tmp_path / "playbook.opf.json"
    opf_path.write_text(json.dumps(doc), encoding="utf-8")

    answers_path = tmp_path / "answers.json"
    answers_path.write_text(json.dumps({"rounds": "2 rounds."}), encoding="utf-8")

    exit_code, output = _invoke(
        "posture", "interview", str(tmp_path), "--answers-file", str(answers_path)
    )
    assert exit_code != 0
    assert "error" in output.lower()
