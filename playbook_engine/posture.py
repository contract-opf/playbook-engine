"""Posture generation — GC interview -> governed, versioned prose block (issue #156).

Implements the decided slice of #116 (2026-07-10, "treat as settled"):
Posture is authored via a short GC interview that the compiler assembles
deterministically into a Posture prose block (OPF-SPEC.md §3.6/§7).
Live LLM generation quality is explicitly out of scope for this slice — the
prose is templated/assembled from the interview answers, not model-written.

This module is the *mechanism*: the canonical question set, the deterministic
assembly of ``posture.system_prompt`` + ``posture.generation.interview`` from
answers, a governed ``version`` counter that bumps every time the interview is
re-run against an existing posture, and a deterministic (non-LLM) SHOULD-warn
check for a Posture that softens language around a Floor-protected concept —
per the issue's Direction, this is advisory (judgment-first), never a hard
error; ``validator.py`` wires it in as a non-blocking ``ValidationError``.

The GC actually answering the interview is a runtime step; this module is
exercised in tests with fixture answers (issue #156's Out of scope note).

API
---
``INTERVIEW_QUESTIONS``       — the canonical 6-question set (OPF §7).
``generate_posture()``        — answers -> a schema-0.2 ``posture`` dict,
                                 versioned (bumped from ``existing_posture``).
``check_posture_floor_conflict()`` — deterministic SHOULD-warn: does the
                                 Posture prose name a Floor invariant's
                                 concept alongside softening language?
``apply_posture_interview()`` — I/O orchestration: read a prior compile's
                                 ``playbook.opf.json``, generate the new
                                 (versioned) Posture, refresh ``identity``,
                                 write back atomically. The CLI's thin
                                 ``playbook posture interview`` command calls
                                 this.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from playbook_engine.canonicalize import compute_section_digests, content_hash
from playbook_engine.playbook_assembler import write_playbook
from playbook_engine.validator import load_opf_file

__all__ = [
    "INTERVIEW_QUESTIONS",
    "InterviewQuestion",
    "PostureApplyResult",
    "PostureError",
    "apply_posture_interview",
    "check_posture_floor_conflict",
    "generate_posture",
]


class PostureError(Exception):
    """Raised when interview answers can't be assembled into a Posture."""


@dataclass(frozen=True)
class InterviewQuestion:
    """One question in the canonical interview set (OPF-SPEC.md §7)."""

    q: str
    question: str
    seeds_floor_candidates: bool = False


# The canonical starter set (3-6 questions; a producer MAY prune or extend —
# OPF §7). Q4 ("sacred_clauses") seeds Floor candidates; the rest shape the
# Posture prose directly.
INTERVIEW_QUESTIONS: tuple[InterviewQuestion, ...] = (
    InterviewQuestion(
        "rounds",
        "How many negotiation rounds do you typically go on this agreement "
        "type before escalating or walking?",
    ),
    InterviewQuestion(
        "leverage",
        "What's your default leverage posture? (take-it-or-leave-it standard "
        "form / collaborative / we usually need the deal more than they do)",
    ),
    InterviewQuestion(
        "risk_appetite",
        "When a counterparty change is non-material, do you default to "
        "accept-to-close, or hold the line?",
    ),
    InterviewQuestion(
        "sacred_clauses",
        "Which clause types are non-negotiable regardless of deal value?",
        seeds_floor_candidates=True,
    ),
    InterviewQuestion(
        "flexible_clauses",
        "Which clause types are you happy to concede to move a deal?",
    ),
    InterviewQuestion(
        "audience",
        "Does your posture change above a deal-value threshold? Who reads "
        "the output — a GC who wants terse rationale, or a junior reviewer "
        "who needs it explained?",
    ),
)

_QUESTIONS_BY_ID: dict[str, InterviewQuestion] = {iq.q: iq for iq in INTERVIEW_QUESTIONS}

# OPF §7: "a short (3-6 question) GC interview" — fewer than 3 answers isn't
# a real interview, it's a couple of stray facts; more than the canonical 6
# isn't recognized by this producer's fixed question set.
_MIN_ANSWERS = 3

# Deterministic prose templates, one per question id, in canonical order.
# Assembled (not LLM-generated) per issue #156's Out-of-scope note.
_PROSE_TEMPLATES: dict[str, str] = {
    "leverage": "Default leverage posture: {answer}",
    "rounds": "Typically goes {answer} negotiation round(s) before escalating or walking.",
    "risk_appetite": "On a non-material counterparty change: {answer}",
    "sacred_clauses": "Hold firm on: {answer} (see Floor).",
    "flexible_clauses": "Flexible to close a deal: {answer}",
    "audience": "Deal-size sensitivity / output audience: {answer}",
}


def _assemble_system_prompt(answers: dict[str, str]) -> str:
    """Deterministically assemble ``system_prompt`` from *answers*.

    One templated sentence per answered question, in ``INTERVIEW_QUESTIONS``
    order (not answer-dict order, so the prose reads the same regardless of
    what order the caller's dict happens to iterate in).
    """
    sentences = [
        _PROSE_TEMPLATES[iq.q].format(answer=answers[iq.q].strip())
        for iq in INTERVIEW_QUESTIONS
        if iq.q in answers
    ]
    return " ".join(sentences)


def generate_posture(
    answers: dict[str, str],
    *,
    generated_at: str,
    generated_by: str = "playbook-engine",
    grounded_in: str | None = None,
    existing_posture: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble a schema-0.2 ``posture`` dict from interview *answers*.

    Args:
        answers:          ``{question_id: answer_text}`` for >= 3 of the
                          questions in ``INTERVIEW_QUESTIONS``. Every key must
                          be a recognized question id.
        generated_at:     ISO-8601 datetime string (supplied by caller — this
                          module stays deterministic/testable, same
                          convention as ``playbook_assembler``).
        generated_by:     Recorded in ``posture.generation.generated_by``.
        grounded_in:      Optional ``"evidence@<digest>"`` string (OPF §7) —
                          the Evidence state the draft was written against.
                          Omitted from ``generation`` when not supplied.
        existing_posture: The prior compile's ``playbook["posture"]`` dict, or
                          ``None``/``{}`` for a first-ever interview. Its
                          ``version`` (if present) is incremented by 1; a
                          missing/absent prior version starts at 1. This is
                          the governed-versioning mechanism the issue asks
                          for: re-running the interview bumps the version.

    Returns:
        A ``posture`` dict: ``{system_prompt, version, generation: {
        generated_by, generated_at, interview, grounded_in?}}``.

    Raises:
        PostureError: fewer than 3 answers, or an answer keyed by an
            unrecognized question id.
    """
    unknown = sorted(set(answers) - set(_QUESTIONS_BY_ID))
    if unknown:
        raise PostureError(
            f"unrecognized interview question id(s): {unknown!r} — must be one of "
            f"{sorted(_QUESTIONS_BY_ID)!r}"
        )
    answered = {q: a for q, a in answers.items() if a is not None and str(a).strip()}
    if len(answered) < _MIN_ANSWERS:
        raise PostureError(
            f"only {len(answered)} answer(s) given; the Posture interview requires "
            f"at least {_MIN_ANSWERS} (OPF §7)"
        )

    system_prompt = _assemble_system_prompt(answered)

    interview = [
        {"q": iq.q, "question": iq.question, "answer": answered[iq.q].strip()}
        for iq in INTERVIEW_QUESTIONS
        if iq.q in answered
    ]

    prior_version = (existing_posture or {}).get("version")
    version = prior_version + 1 if isinstance(prior_version, int) else 1

    generation: dict[str, Any] = {
        "generated_by": generated_by,
        "generated_at": generated_at,
        "interview": interview,
    }
    if grounded_in is not None:
        generation["grounded_in"] = grounded_in

    return {
        "system_prompt": system_prompt,
        "version": version,
        "generation": generation,
    }


# ---------------------------------------------------------------------------
# SHOULD-warn: Posture softening language vs. a Floor-protected concept
# ---------------------------------------------------------------------------

# Deliberately deterministic/lexical (never an LLM judge) — mirrors the
# Floor's own "detectable" admission-test half (OPF §3.7.1) and this issue's
# Out-of-scope note that generation quality (and, by the same logic, this
# check) is templated/assembled, not model-judged, in this slice.
_SOFTENING_TERMS: tuple[str, ...] = (
    "flexible",
    "negotiable",
    "willing to concede",
    "happy to concede",
    "may waive",
    "open to waiving",
    "not a red line",
    "can be adjusted",
    "room to move",
    "willing to soften",
    "concede to move",
)

_STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "of",
        "on",
        "in",
        "to",
        "is",
        "are",
        "we",
        "our",
        "us",
        "for",
        "regardless",
        "never",
        "always",
        "not",
        "with",
        "see",
        "floor",
        "posture",
    }
)

_WORD_RE = re.compile(r"[a-z0-9]+")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def _content_words(text: str) -> set[str]:
    return {w for w in _WORD_RE.findall(text.lower()) if len(w) > 2 and w not in _STOPWORDS}


def check_posture_floor_conflict(
    system_prompt: str,
    floor_invariants: list[dict[str, Any]] | None,
) -> list[str]:
    """Deterministic SHOULD-warn: does *system_prompt* name a Floor-protected
    concept in the same sentence as softening language?

    Per issue #156's Direction: "Posture-vs-Floor contradiction is a
    SHOULD-warn (judgment-first), not a hard error" — this supersedes
    OPF-SPEC.md §3.6 rule 3's "validation error" language for this
    slice. Callers (``validator.py``) surface the returned messages as
    non-blocking warnings, never as blocking errors.

    Heuristic (deterministic, no LLM): for each Floor invariant, take the
    content words (stopwords/short words stripped) shared between the
    invariant's ``statement`` and *system_prompt*. If a sentence of
    *system_prompt* contains both a shared content word and one of
    ``_SOFTENING_TERMS``, the Posture may be softening a concept the Floor
    protects — flag it. This is advisory pattern-matching, not proof of an
    actual conflict; a human reviews every flagged case.

    Args:
        system_prompt:    ``posture.system_prompt`` text (may be empty).
        floor_invariants: ``floor.invariants`` list (schema-0.2 shape: dicts
                          with ``id``/``statement``), or ``None``/``[]``.

    Returns:
        Human-readable warning strings, one per (invariant, sentence) match.
        Empty when there's nothing to warn about (no invariants, no
        softening language present at all, or no overlap).
    """
    if not floor_invariants or not system_prompt.strip():
        return []

    prompt_lower = system_prompt.lower()
    if not any(term in prompt_lower for term in _SOFTENING_TERMS):
        return []

    sentences = [s for s in _SENTENCE_RE.split(system_prompt.strip()) if s.strip()]

    warnings: list[str] = []
    for inv in floor_invariants:
        statement = inv.get("statement", "")
        inv_id = inv.get("id", "<unknown>")
        invariant_words = _content_words(statement)
        if not invariant_words:
            continue

        for sentence in sentences:
            sentence_lower = sentence.lower()
            softening_hits = [t for t in _SOFTENING_TERMS if t in sentence_lower]
            if not softening_hits:
                continue
            overlap = invariant_words & _content_words(sentence)
            if overlap:
                warnings.append(
                    f"Posture sentence {sentence.strip()!r} names Floor invariant "
                    f"{inv_id!r} ({statement!r}) alongside softening language "
                    f"({sorted(softening_hits)!r}) — possible Posture-vs-Floor "
                    "conflict; SHOULD be reviewed (OPF §3.6 rule 3, issue #156)."
                )

    return warnings


# ---------------------------------------------------------------------------
# I/O orchestration — read-modify-write playbook.opf.json (mirrors
# viewer.apply_feedback's curation-pin write path).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PostureApplyResult:
    """Result of writing a freshly generated Posture into ``playbook.opf.json``."""

    version: int
    warnings: tuple[str, ...]
    path: Path


def apply_posture_interview(
    out_dir: Path,
    answers: dict[str, str],
    *,
    generated_at: str,
    generated_by: str = "playbook-engine",
) -> PostureApplyResult:
    """Read ``{out_dir}/playbook.opf.json``, write a freshly generated,
    versioned Posture into it, and return the result.

    Mirrors ``viewer.apply_feedback``'s curation-pin write path: reads the
    existing document, replaces one section, refreshes ``identity`` (since —
    unlike ``curation`` — ``posture`` IS part of ``content_hash``; see
    ``canonicalize.py``), and writes back atomically via
    ``playbook_assembler.write_playbook``.

    Args:
        out_dir:       Directory containing ``playbook.opf.json`` (produced
                       by ``playbook compile``/``project``).
        answers:       Interview answers — see ``generate_posture()``.
        generated_at:  ISO-8601 datetime (supplied by caller).
        generated_by:  Recorded in ``posture.generation.generated_by``.

    Returns:
        ``PostureApplyResult`` — the new version number, any SHOULD-warn
        messages from ``check_posture_floor_conflict()``, and the path
        written.

    Raises:
        FileNotFoundError: no ``playbook.opf.json`` in *out_dir*.
        PostureError:      see ``generate_posture()``.
    """
    opf_path = out_dir / "playbook.opf.json"
    if not opf_path.exists():
        raise FileNotFoundError(f"{opf_path} not found — run 'playbook compile'/'project' first.")
    doc = load_opf_file(opf_path)

    existing_posture = doc.get("posture") or {}
    grounded_in = None
    evidence_digest = (doc.get("identity") or {}).get("section_digests", {}).get("evidence")
    if evidence_digest:
        grounded_in = f"evidence@{evidence_digest}"

    posture = generate_posture(
        answers,
        generated_at=generated_at,
        generated_by=generated_by,
        grounded_in=grounded_in,
        existing_posture=existing_posture,
    )

    floor_invariants = (doc.get("floor") or {}).get("invariants") or []
    warnings = check_posture_floor_conflict(posture["system_prompt"], floor_invariants)

    doc["posture"] = posture
    if "identity" in doc:
        doc["identity"]["content_hash"] = content_hash(doc)
        doc["identity"]["section_digests"] = compute_section_digests(doc)

    write_playbook(doc, opf_path)

    return PostureApplyResult(version=posture["version"], warnings=tuple(warnings), path=opf_path)
