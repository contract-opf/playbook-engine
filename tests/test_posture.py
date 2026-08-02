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
import re
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from playbook_engine.canonicalize import content_hash
from playbook_engine.cli import cli
from playbook_engine.floor_candidates import write_floor_candidates
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
# _assemble_system_prompt / generate_posture — answer-splice grammar (#89)
#
# Pre-fix, two of the six _PROSE_TEMPLATES interpolated {answer} MID-sentence
# with template text trailing it ("rounds": "...goes {answer} negotiation
# round(s)..."; "sacred_clauses": "...{answer} (see Floor)."). A FRAGMENT
# answer ("Two rounds") read fine; a FULL-SENTENCE answer ("Two rounds, then
# escalate to the GC.") produced a garbled double-punctuated splice. These
# tests cover both answer shapes for both formerly-broken templates.
# ---------------------------------------------------------------------------


def test_rounds_full_sentence_answer_has_no_splice_artifact() -> None:
    answers = {**_ANSWERS, "rounds": "Two rounds, then escalate to the GC."}
    posture = generate_posture(answers, generated_at="2026-07-10T00:00:00Z")

    assert ". negotiation round(s)" not in posture["system_prompt"]
    assert "Two rounds, then escalate to the GC. negotiation" not in posture["system_prompt"]
    assert (
        "Rounds before escalating: Two rounds, then escalate to the GC."
        in (posture["system_prompt"])
    )


def test_rounds_fragment_answer_renders_cleanly() -> None:
    answers = {**_ANSWERS, "rounds": "Two rounds"}
    posture = generate_posture(answers, generated_at="2026-07-10T00:00:00Z")

    assert "Rounds before escalating: Two rounds" in posture["system_prompt"]
    assert "negotiation round(s)" not in posture["system_prompt"]


def test_sacred_clauses_full_sentence_answer_has_no_splice_artifact() -> None:
    answers = {**_ANSWERS, "sacred_clauses": "Liability cap and indemnification are sacred."}
    posture = generate_posture(answers, generated_at="2026-07-10T00:00:00Z")

    assert ". (see Floor)" not in posture["system_prompt"]
    assert (
        "Hard lines named in interview (see Floor): "
        "Liability cap and indemnification are sacred." in posture["system_prompt"]
    )


def test_sacred_clauses_fragment_answer_renders_cleanly() -> None:
    answers = {**_ANSWERS, "sacred_clauses": "Liability cap"}
    posture = generate_posture(answers, generated_at="2026-07-10T00:00:00Z")

    assert "Hard lines named in interview (see Floor): Liability cap" in posture["system_prompt"]


# A field's "trailing" text (everything right after the raw *answer*
# substring) alone can't tell "properly terminated" from "run-on": a
# full-sentence answer's own "." is INSIDE the answer substring, so a
# correctly-formed trailing looks like " NextLabel..." (space, uppercase) —
# but that is EXACTLY what an unterminated FRAGMENT run straight into the
# next label ALSO looks like, since nothing was inserted between them
# either. The two are only distinguishable by ALSO checking whether the
# answer itself already ended in terminal punctuation — hence two patterns,
# selected by that check, not one pattern applied uniformly (a single
# "punctuation-is-optional" regex here would silently accept the very
# run-on issue #89 review finding 1 reported — caught while writing this
# fix's own regression test, against the pre-fix source, before it ever
# reached review).
_ALREADY_TERMINATED_TRAILING_RE = re.compile(r"^(?:\s[A-Z].*)?$")  # "" or " NextLabel..."
_FRAGMENT_TRAILING_RE = re.compile(r"^\.(?:\s[A-Z].*)?$")  # "." or ". NextLabel..."

_FULL_SENTENCE_ANSWERS: dict[str, str] = {
    "rounds": "We go two rounds, then escalate.",
    "leverage": "We favor a collaborative posture.",
    "risk_appetite": "We accept to close on non-material changes.",
    "sacred_clauses": "We hold firm on the liability cap.",
    "flexible_clauses": "We concede on renewal mechanics.",
    "audience": "We write for a GC audience.",
}

# No trailing '.'/'!'/'?' on any answer — the FRAGMENT case (issue #89
# review finding 1's regression: pre-fix, a fragment ran straight into the
# next field's label with no sentence boundary at all).
_FRAGMENT_ANSWERS: dict[str, str] = {
    "rounds": "Two rounds before escalating",
    "leverage": "Collaborative",
    "risk_appetite": "Accept-to-close on non-material changes",
    "sacred_clauses": "Liability caps and student-data protection",
    "flexible_clauses": "Term length and renewal mechanics",
    "audience": "Terse rationale for a GC audience",
}


@pytest.mark.parametrize(
    "answers", [_FULL_SENTENCE_ANSWERS, _FRAGMENT_ANSWERS], ids=["full_sentence", "fragment"]
)
def test_all_six_templates_terminate_each_field_before_the_next_label(
    answers: dict[str, str],
) -> None:
    """One consistent, grammatical scheme for all six templates (#89): every
    assembled field ends with the answer, immediately followed by
    sentence-final punctuation, before the next field's label begins — for
    BOTH a full-sentence answer (already ends in ./!/?, left untouched) and
    a FRAGMENT answer (no terminal punctuation — issue #89 review finding
    1's regression: pre-fix, a fragment ran straight into the next label
    with no sentence boundary at all, e.g. "...protection Flexible to
    close a deal:...", which also defeated ``_SENTENCE_RE``'s sentence
    splitting in ``check_posture_floor_conflict``).

    Distinguishes "legitimately followed by the next label" from "template
    junk trails the answer" (pre-fix: a space then lowercase "negotiation"
    or a literal "(") — and, for a fragment answer, from a bare run-on with
    NO terminator at all — using whichever of
    ``_ALREADY_TERMINATED_TRAILING_RE`` / ``_FRAGMENT_TRAILING_RE`` applies
    to that answer; see the comment above them.
    """
    posture = generate_posture(answers, generated_at="2026-07-10T00:00:00Z")
    system_prompt = posture["system_prompt"]

    for iq in INTERVIEW_QUESTIONS:
        answer = answers[iq.q]
        assert system_prompt.count(answer) == 1, iq.q
        trailing = system_prompt[system_prompt.index(answer) + len(answer) :]
        pattern = _ALREADY_TERMINATED_TRAILING_RE if answer[-1] in ".!?" else _FRAGMENT_TRAILING_RE
        assert pattern.match(trailing), (
            f"{iq.q}: field not properly terminated before the next label — {trailing[:30]!r}"
        )


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


def test_check_posture_floor_conflict_fires_on_red_line_and_hard_line_phrasing() -> None:
    # Issue #88 renamed the *rendered* term "red line" -> "hard line", but
    # "not a red line" stays a recognized _SOFTENING_TERMS entry alongside
    # "not a hard line": it is matched against user-authored
    # posture.system_prompt prose, not our own rendered output, and GCs keep
    # typing the ordinary-English "red line" regardless of this repo's
    # vocabulary (every already-derived playbook uses the old term too).
    # Both spellings must fire so this SHOULD-warn's recall doesn't silently
    # drop for pre-existing prose.
    for phrase in ("not a red line", "not a hard line"):
        system_prompt = f"The liability cap is {phrase} for us."
        warnings = check_posture_floor_conflict(system_prompt, [_LIABILITY_INVARIANT])
        assert warnings, phrase
        assert "no-uncapped-liability" in warnings[0], phrase


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
# apply_posture_interview — Q4 -> Floor promotion (issue #89)
# ---------------------------------------------------------------------------


def test_apply_posture_interview_promotes_q4_into_floor_invariants(tmp_path: Path) -> None:
    doc = _minimal_v02_doc()
    opf_path = tmp_path / "playbook.opf.json"
    opf_path.write_text(json.dumps(doc), encoding="utf-8")

    answers = {**_ANSWERS, "sacred_clauses": "Liability caps and student-data protection"}
    apply_posture_interview(tmp_path, answers, generated_at="2026-07-10T00:00:00Z")

    written = json.loads(opf_path.read_text(encoding="utf-8"))
    invariants = written["floor"]["invariants"]
    assert len(invariants) == 1
    inv = invariants[0]
    assert inv["id"] == "liability-caps-and-student-data-protection"
    assert inv["statement"] == "Do not concede on Liability caps and student-data protection."
    assert "posture interview v1" in inv["rationale"]
    assert "sacred_clauses" in inv["rationale"]


def test_apply_posture_interview_ticket_demo_answers_yield_zero_warnings(tmp_path: Path) -> None:
    """Issue #89 review finding 1 regression: the ticket's own required-
    verification demo answers (``sacred_clauses`` = "Liability caps and
    student-data protection" — a FRAGMENT, no terminal punctuation) must
    not trip ``check_posture_floor_conflict`` against the very invariant
    this same interview run promotes from that answer.

    Pre-fix, the missing sentence terminator let ``_SENTENCE_RE`` merge the
    sacred_clauses field with the next field into one run-on sentence
    ("...student-data protection Flexible to close a deal: Term length and
    renewal mechanics."), which contains the softening term "flexible" and
    overlaps the freshly-promoted invariant's content words — a
    false-positive SHOULD-warn on the exact invariant just promoted."""
    doc = _minimal_v02_doc()
    opf_path = tmp_path / "playbook.opf.json"
    opf_path.write_text(json.dumps(doc), encoding="utf-8")

    answers = {**_ANSWERS, "sacred_clauses": "Liability caps and student-data protection"}
    result = apply_posture_interview(tmp_path, answers, generated_at="2026-07-10T00:00:00Z")

    assert result.warnings == ()

    written = json.loads(opf_path.read_text(encoding="utf-8"))
    validated = validate_document(written)
    warn_messages = [e.message for e in validated.errors if not e.blocking]
    assert warn_messages == []


def test_apply_posture_interview_concede_softening_phrase_does_not_warn_on_unrelated_invariants(
    tmp_path: Path,
) -> None:
    """Issue #89 review, fix round 2: every interview-promoted invariant's
    statement is built from the same boilerplate — "Do not concede on
    {item}." (``floor_candidates._q4_promoted_statement``) — so "concede"
    is a content word of EVERY promoted invariant regardless of which
    clause type was actually named. "concede" is also a constituent word
    of three ``_SOFTENING_TERMS`` phrases ("willing to concede", "happy to
    concede", "concede to move"). Pre-fix, an ordinary ``risk_appetite``
    answer using one of those phrases content-word-overlapped on "concede"
    alone with every promoted invariant, regardless of the concept either
    one actually named — a pure template artifact, not a shared concept.

    Reproduces the reviewer's own end-to-end repro: a risk_appetite answer
    using "willing to concede" alongside three unrelated sacred_clauses
    items must promote three invariants and warn about none of them."""
    doc = _minimal_v02_doc()
    opf_path = tmp_path / "playbook.opf.json"
    opf_path.write_text(json.dumps(doc), encoding="utf-8")

    answers = {
        **_ANSWERS,
        "risk_appetite": "We are willing to concede on non-material drafting changes",
        "sacred_clauses": "Student-data protection; audit rights; source-code escrow",
    }
    result = apply_posture_interview(tmp_path, answers, generated_at="2026-07-10T00:00:00Z")

    written = json.loads(opf_path.read_text(encoding="utf-8"))
    invariants = written["floor"]["invariants"]
    assert len(invariants) == 3  # the promotion itself is unaffected by this fix

    assert result.warnings == ()

    validated = validate_document(written)
    warn_messages = [e.message for e in validated.errors if not e.blocking]
    assert warn_messages == []  # validator.py re-derives the same check on every run


def test_apply_posture_interview_colliding_hand_authored_id_raises_postureerror(
    tmp_path: Path,
) -> None:
    """Issue #89 review finding 2, end-to-end: a hand-authored invariant
    whose id collides with a freshly Q4-named item's slug must never be
    silently overwritten by 'playbook posture interview' — the conflict
    surfaces as a PostureError (the public type this module's callers
    already catch — see cli.py), and nothing is written."""
    hand_authored = {
        "id": "ip-assignment",
        "statement": "Never accept present-tense assignment of pre-existing IP.",
        "rationale": "Signed off by the GC 2026-03-01 after board review.",
        "x_signed_by": "gc@example.com",
    }
    doc = _minimal_v02_doc(floor={"invariants": [hand_authored]})
    opf_path = tmp_path / "playbook.opf.json"
    original_bytes = json.dumps(doc).encode("utf-8")
    opf_path.write_bytes(original_bytes)

    answers = {**_ANSWERS, "sacred_clauses": "IP assignment"}
    with pytest.raises(PostureError, match="ip-assignment"):
        apply_posture_interview(tmp_path, answers, generated_at="2026-07-10T00:00:00Z")

    # Nothing was written — the atomic tmp+replace write never ran.
    assert opf_path.read_bytes() == original_bytes


def test_apply_posture_interview_promotion_is_idempotent_on_rerun(tmp_path: Path) -> None:
    doc = _minimal_v02_doc()
    opf_path = tmp_path / "playbook.opf.json"
    opf_path.write_text(json.dumps(doc), encoding="utf-8")

    answers = {**_ANSWERS, "sacred_clauses": "Liability caps and student-data protection"}
    apply_posture_interview(tmp_path, answers, generated_at="2026-07-10T00:00:00Z")
    first_invariants = json.loads(opf_path.read_text(encoding="utf-8"))["floor"]["invariants"]

    # Re-run with the SAME answers file — posture.version bumps, but the
    # promoted invariant must not duplicate or otherwise change.
    apply_posture_interview(tmp_path, answers, generated_at="2026-07-11T00:00:00Z")
    second_invariants = json.loads(opf_path.read_text(encoding="utf-8"))["floor"]["invariants"]

    assert len(second_invariants) == 1
    assert second_invariants == first_invariants  # byte-for-byte unchanged
    written = json.loads(opf_path.read_text(encoding="utf-8"))
    assert written["posture"]["version"] == 2  # posture itself did advance


def test_apply_posture_interview_promotion_preserves_hand_authored_invariants(
    tmp_path: Path,
) -> None:
    doc = _minimal_v02_doc(floor={"invariants": [_LIABILITY_INVARIANT]})
    opf_path = tmp_path / "playbook.opf.json"
    opf_path.write_text(json.dumps(doc), encoding="utf-8")

    answers = {**_ANSWERS, "sacred_clauses": "IP assignment"}
    apply_posture_interview(tmp_path, answers, generated_at="2026-07-10T00:00:00Z")

    written = json.loads(opf_path.read_text(encoding="utf-8"))
    invariants = written["floor"]["invariants"]
    assert _LIABILITY_INVARIANT in invariants
    assert len(invariants) == 2


def test_apply_posture_interview_promoted_invariants_pass_validation(tmp_path: Path) -> None:
    doc = _minimal_v02_doc()
    opf_path = tmp_path / "playbook.opf.json"
    opf_path.write_text(json.dumps(doc), encoding="utf-8")

    answers = {**_ANSWERS, "sacred_clauses": "Liability caps and student-data protection"}
    apply_posture_interview(tmp_path, answers, generated_at="2026-07-10T00:00:00Z")
    # Re-run twice more to prove the validator's duplicate-id check
    # (OPF-SPEC.md §3.13) never trips on repeated promotion of the same
    # statement.
    apply_posture_interview(tmp_path, answers, generated_at="2026-07-11T00:00:00Z")
    apply_posture_interview(tmp_path, answers, generated_at="2026-07-12T00:00:00Z")

    written = json.loads(opf_path.read_text(encoding="utf-8"))
    result = validate_document(written)
    blocking_errors = [str(e) for e in result.errors if e.blocking]
    assert result.ok, blocking_errors
    assert not any("duplicate" in e.lower() for e in blocking_errors)


def test_apply_posture_interview_no_sacred_clauses_answer_leaves_floor_untouched(
    tmp_path: Path,
) -> None:
    doc = _minimal_v02_doc(floor={"invariants": [_LIABILITY_INVARIANT]})
    opf_path = tmp_path / "playbook.opf.json"
    opf_path.write_text(json.dumps(doc), encoding="utf-8")

    answers = {k: v for k, v in _ANSWERS.items() if k != "sacred_clauses"}
    apply_posture_interview(tmp_path, answers, generated_at="2026-07-10T00:00:00Z")

    written = json.loads(opf_path.read_text(encoding="utf-8"))
    assert written["floor"]["invariants"] == [_LIABILITY_INVARIANT]


def test_apply_posture_interview_no_promotion_leaves_floor_section_untouched(
    tmp_path: Path,
) -> None:
    """Issue #89 review finding 6 regression: when Q4 wasn't answered (so
    promotion is a pure pass-through, producing the identical invariants
    list), the ``floor`` section itself must not be rewritten — an empty
    ``floor: {}`` must stay exactly ``{}``, not gain a fabricated
    ``{"invariants": []}`` that changes identity.content_hash for no
    reason."""
    doc = _minimal_v02_doc()  # floor={}
    opf_path = tmp_path / "playbook.opf.json"
    opf_path.write_text(json.dumps(doc), encoding="utf-8")

    answers = {k: v for k, v in _ANSWERS.items() if k != "sacred_clauses"}
    apply_posture_interview(tmp_path, answers, generated_at="2026-07-10T00:00:00Z")

    written = json.loads(opf_path.read_text(encoding="utf-8"))
    assert written["floor"] == {}


def test_apply_posture_interview_never_promotes_reversal_candidates(tmp_path: Path) -> None:
    """Issue #89 acceptance criterion: an out dir whose observations contain
    ``proposed_then_reversed`` still gets those ONLY as candidates in
    ``floor.candidates.json`` (via 'playbook floor propose'), never
    auto-promoted — posture interview's Floor promotion is scoped to the Q4
    answer only and never reads observations.jsonl."""
    doc = _minimal_v02_doc()
    opf_path = tmp_path / "playbook.opf.json"
    opf_path.write_text(json.dumps(doc), encoding="utf-8")

    obs_path = tmp_path / "observations.jsonl"
    reversal_obs = {
        "observation_id": "doc-a/2/8.1",
        "taxonomy_id": "uncapped_liability",
        "text_summary": "The Vendor's liability shall be uncapped for any breach.",
        "full_text": "The Vendor's liability shall be uncapped for any breach.",
        "citation": {
            "document_id": "doc-a",
            "version": 2,
            "clause_path": "8.1",
            "char_span": None,
            "version_id": None,
        },
        "deviation": "substantive",
        "risk_delta": {"direction": "neutral", "magnitude": "none"},
        "provenance": "counterparty_paper",
        "outcome": "proposed_then_reversed",
        "confidence": None,
        "basis": "deterministic",
    }
    obs_path.write_text(json.dumps(reversal_obs) + "\n", encoding="utf-8")

    answers = {**_ANSWERS, "sacred_clauses": "IP assignment"}
    apply_posture_interview(tmp_path, answers, generated_at="2026-07-10T00:00:00Z")

    written = json.loads(opf_path.read_text(encoding="utf-8"))
    invariant_statements = [inv["statement"] for inv in written["floor"]["invariants"]]
    # Only the human-authored Q4 item was promoted — the reversal never was.
    assert invariant_statements == ["Do not concede on IP assignment."]
    assert not any("uncapped" in s.lower() for s in invariant_statements)

    # The reversal is still available as a pending PROPOSAL, untouched.
    candidates_path = write_floor_candidates(tmp_path)
    candidates = json.loads(candidates_path.read_text(encoding="utf-8"))["candidates"]
    reversal_candidates = [c for c in candidates if c["source"] == "reversal"]
    assert len(reversal_candidates) == 1
    assert "uncapped liability" in reversal_candidates[0]["statement"].lower()


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


def test_cli_posture_interview_rerun_does_not_duplicate_floor_invariants(tmp_path: Path) -> None:
    """Issue #89 required verification: run the interview a second time with
    the same answers file; floor.invariants must be unchanged (no
    duplicates)."""
    doc = _minimal_v02_doc()
    opf_path = tmp_path / "playbook.opf.json"
    opf_path.write_text(json.dumps(doc), encoding="utf-8")

    answers = {**_ANSWERS, "sacred_clauses": "Liability caps and student-data protection"}
    answers_path = tmp_path / "answers.json"
    answers_path.write_text(json.dumps(answers), encoding="utf-8")

    exit_code, output = _invoke(
        "posture", "interview", str(tmp_path), "--answers-file", str(answers_path)
    )
    assert exit_code == 0, output
    first_invariants = json.loads(opf_path.read_text(encoding="utf-8"))["floor"]["invariants"]
    assert len(first_invariants) == 1

    exit_code2, output2 = _invoke(
        "posture", "interview", str(tmp_path), "--answers-file", str(answers_path)
    )
    assert exit_code2 == 0, output2
    second_invariants = json.loads(opf_path.read_text(encoding="utf-8"))["floor"]["invariants"]

    assert second_invariants == first_invariants


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
