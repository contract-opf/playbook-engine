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

Issue #89 adds one more effect to ``apply_posture_interview()``: the Q4
("sacred_clauses") answer is promoted directly into ``floor.invariants`` —
see ``floor_candidates.promote_interview_q4_invariants()`` for why that is
not the auto-promotion OPF-SPEC.md §3.7 rule 4 forbids (short version: a Q4
answer is human-authored, not a compiler-derived candidate — the human act
of authorship already is the sign-off).

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
                                 (versioned) Posture, promote the Q4 answer
                                 into ``floor.invariants``, refresh
                                 ``identity``, write back atomically. The
                                 CLI's thin ``playbook posture interview``
                                 command calls this. Accepts an optional
                                 ``base_version`` (issue #126) so the
                                 governed version counter can continue
                                 across a re-derivation into an out-dir
                                 whose own document doesn't carry the prior
                                 Posture, instead of silently restarting at
                                 1 — see ``generate_posture()``'s docstring.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from playbook_engine.canonicalize import compute_section_digests, content_hash
from playbook_engine.floor_candidates import (
    FloorCandidateError,
    promote_interview_q4_invariants,
    q4_sentence_shaped_items,
)
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


# The canonical starter set (3-6 questions; a producer MAY prune or extend —
# OPF §7). Q4 ("sacred_clauses") seeds Floor candidates (promoted directly by
# floor_candidates.promote_interview_q4_invariants) and Q5 ("flexible_clauses")
# auto-rejects matching reversal-derived Floor candidates
# (floor_candidates.propose_floor_candidates, issue #105) — both identified
# by their local id constants (INTERVIEW_Q4_ID / INTERVIEW_Q5_ID) in
# floor_candidates.py, not by any flag on this dataclass; the rest shape the
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
#
# One consistent, grammatical scheme for all six (issue #89): a short label,
# a colon, then the answer verbatim with NOTHING template-authored appended
# after it. Two of these used to interpolate {answer} mid-sentence with
# template text trailing it ("rounds": "...goes {answer} negotiation
# round(s)..."; "sacred_clauses": "...{answer} (see Floor)."), which
# produced a double-punctuated splice the instant an answer was a full
# sentence rather than a bare fragment (e.g. answer = "Two rounds, then
# escalate to the GC." rendered as "...goes Two rounds, then escalate to
# the GC. negotiation round(s) before..."). Putting {answer} last removes
# that particular splice — but is NOT sufficient on its own: a FRAGMENT
# answer (no trailing ./!/?) then leaves no sentence boundary before the
# next field's label, so e.g. a sacred_clauses fragment ran straight into
# "Flexible to close a deal: ..." as one unreadable, unsplittable run-on
# sentence (issue #89 review finding 1). ``_assemble_system_prompt`` closes
# that gap by passing every assembled field through ``_terminate`` before
# joining — see its docstring.
_PROSE_TEMPLATES: dict[str, str] = {
    "rounds": "Rounds before escalating: {answer}",
    "leverage": "Default leverage posture: {answer}",
    "risk_appetite": "Non-material counterparty change: {answer}",
    "sacred_clauses": "Hard lines named in interview (see Floor): {answer}",
    "flexible_clauses": "Flexible to close a deal: {answer}",
    "audience": "Deal-size sensitivity / output audience: {answer}",
}


def _terminate(field: str) -> str:
    """Ensure *field* ends with sentence-final punctuation.

    Every ``_PROSE_TEMPLATES`` entry puts the answer LAST with nothing
    template-authored appended after it (see the comment above
    ``_PROSE_TEMPLATES``), which fixed the original mid-sentence splice —
    but a FRAGMENT answer (no trailing ``.``/``!``/``?``) then has no
    sentence boundary before the next field's label: "Hard lines named in
    interview (see Floor): Liability caps and student-data protection
    Flexible to close a deal: ..." reads (and, worse, PARSES) as one run-on
    sentence — invisible to ``_SENTENCE_RE``'s ``(?<=[.!?])\\s+`` boundary
    detector, which is exactly what ``check_posture_floor_conflict`` splits
    on to reason about "one sentence at a time" (issue #89 review finding
    1). Appending "." only when *field* doesn't already end in terminal
    punctuation leaves a full-sentence answer's own punctuation untouched
    (no double punctuation), while guaranteeing every field — fragment or
    full sentence — ends its own sentence.
    """
    field = field.rstrip()
    if field and field[-1] not in ".!?":
        field += "."
    return field


def _assemble_system_prompt(answers: dict[str, str]) -> str:
    """Deterministically assemble ``system_prompt`` from *answers*.

    One templated sentence per answered question, in ``INTERVIEW_QUESTIONS``
    order (not answer-dict order, so the prose reads the same regardless of
    what order the caller's dict happens to iterate in). Each assembled
    field is passed through :func:`_terminate` before joining, so a
    fragment answer still ends its sentence — see that function's
    docstring.
    """
    sentences = [
        _terminate(_PROSE_TEMPLATES[iq.q].format(answer=answers[iq.q].strip()))
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
    base_version: int | None = None,
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
                          ``version`` (if present) is one input to the
                          governed-versioning bump below.
        base_version:     Issue #126 — the last known posture version from a
                          playbook this run's document does NOT itself carry
                          (e.g. the out-dir was wiped or the interview is
                          being applied to a freshly re-derived, previously
                          unrelated out-dir during a re-derivation). When the
                          document was recompiled in place, ``existing_posture``
                          alone is sufficient and this can be omitted. When
                          supplied, the effective prior version is
                          ``max(existing_posture.version, base_version)`` so
                          the counter can never go backwards across a
                          re-derivation — only forward, and never below what
                          the document itself already carries.

    Returns:
        A ``posture`` dict: ``{system_prompt, version, generation: {
        generated_by, generated_at, interview, grounded_in?}}``.

    Raises:
        PostureError: fewer than 3 answers, or an answer keyed by an
            unrecognized question id, or a non-positive ``base_version``.
    """
    unknown = sorted(set(answers) - set(_QUESTIONS_BY_ID))
    if unknown:
        raise PostureError(
            f"unrecognized interview question id(s): {unknown!r} — must be one of "
            f"{sorted(_QUESTIONS_BY_ID)!r}"
        )
    if base_version is not None and base_version < 1:
        raise PostureError(f"base_version must be >= 1 if supplied; got {base_version!r}")
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

    # Issue #126: the bump must never regress across a re-derivation. The
    # document's own carried-forward posture version and an externally
    # supplied `base_version` (the last known version from a playbook this
    # freshly re-derived document doesn't itself carry — see the docstring)
    # are both candidate "prior" versions; the effective prior is whichever
    # is higher, so the new version is always > either.
    doc_version = (existing_posture or {}).get("version")
    doc_version = doc_version if isinstance(doc_version, int) else None
    candidates = [v for v in (doc_version, base_version) if v is not None]
    version = max(candidates) + 1 if candidates else 1

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
    "not a hard line",
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
        # "concede" is the boilerplate verb every interview-promoted
        # invariant's statement is built from (floor_candidates.py's
        # ``_q4_promoted_statement`` — "Do not concede on {item}."), so it
        # is a CONTENT word of every one of those statements regardless of
        # which clause type the legal owner actually named. It is also a
        # constituent word of three ``_SOFTENING_TERMS`` phrases above
        # ("willing to concede", "happy to concede", "concede to move").
        # Left as a content word, ANY Posture sentence using one of those
        # phrases would content-word-overlap EVERY promoted invariant on
        # "concede" alone, regardless of what concept the sentence and the
        # invariant actually name — a pure template artifact, not a shared
        # concept (issue #89 review, fix round 2). Treating it as a
        # stopword removes it from the overlap computation on both sides
        # (invariant statement and Posture sentence) without touching
        # ``_SOFTENING_TERMS`` softening-phrase detection itself, which
        # matches those phrases as substrings independently of
        # ``_content_words`` — see ``check_posture_floor_conflict`` below.
        "concede",
        # "agreement" is the single most domain-universal noun here: every
        # Posture sentence and essentially every hand-authored Floor statement
        # ("If the agreement contains a limitation of liability...") uses it,
        # so left as a content word it makes ANY softening Posture sentence
        # overlap ANY invariant on "agreement" alone — a domain artifact, not a
        # shared concept. Same reasoning as "concede" above, and the same
        # reasoning that kept "deal" out of `_q4_promoted_statement`'s wording.
        # Verified against a real derived playbook: two invariants each warned
        # against the "Flexible to close a deal: governing law; notices; ..."
        # sentence whose ONLY overlap with them was the word "agreement".
        "agreement",
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
    base_version: int | None = None,
) -> PostureApplyResult:
    """Read ``{out_dir}/playbook.opf.json``, write a freshly generated,
    versioned Posture into it — promoting the Q4 ("sacred_clauses") answer
    directly into ``floor.invariants`` along the way — and return the result.

    Mirrors ``viewer.apply_feedback``'s curation-pin write path: reads the
    existing document, replaces sections, refreshes ``identity`` (since —
    unlike ``curation`` — both ``posture`` and ``floor`` ARE part of
    ``content_hash``; see ``canonicalize.py``), and writes back atomically
    via ``playbook_assembler.write_playbook``. ``posture`` is always
    replaced (its ``version`` always advances); ``floor`` is only replaced
    when the Q4 promotion below actually changed ``floor.invariants`` — a
    run that doesn't touch the Floor doesn't fabricate a
    ``floor.invariants: []`` or otherwise perturb ``identity`` for no
    reason (issue #89 review finding 6).

    The Floor promotion (issue #89) is not the auto-promotion OPF-SPEC.md
    §3.7 rule 4 forbids — see ``floor_candidates.promote_interview_q4_invariants``'s
    docstring for why — and is idempotent per statement: re-running the
    interview with an unchanged ``sacred_clauses`` answer leaves
    ``floor.invariants`` unchanged (same id, same statement, no duplicate;
    OPF-SPEC.md §3.13 makes a duplicate sibling id a blocking validator
    error). Reversal-derived candidates never take this path — they stay on
    ``floor_candidates.derive_reversal_candidates``'s propose-then-sign-off
    path (``playbook floor propose`` / ``floor.candidates.json``), which
    this function never touches.

    A SENTENCE-SHAPED Q4 item (``floor_candidates.is_q4_item_sentence_shaped``,
    issue #104) — a conditional hard line typed as prose, not a bare
    clause-type name, e.g. "limitation of liability, if present, must not
    be unilateral in the counterparty's favor" — is never templated or
    promoted (``promote_interview_q4_invariants`` itself skips it); this
    function additionally surfaces one actionable warning per such item,
    through the same warnings channel as ``check_posture_floor_conflict()``,
    naming the exact command to record it verbatim instead: ``playbook
    floor sign {out_dir} --statement "<the item>" --signed-by "<your
    name>"``. The item itself stays
    exactly as typed in the recorded interview
    (``posture.generation.interview`` — see ``generate_posture()``); only
    the Floor-promotion and templating steps skip it.

    Args:
        out_dir:       Directory containing ``playbook.opf.json`` (produced
                       by ``playbook mine``/``project``). Also the
                       directory named in a sentence-shaped item's
                       ``playbook floor sign`` warning, above.
        answers:       Interview answers — see ``generate_posture()``.
        generated_at:  ISO-8601 datetime (supplied by caller).
        generated_by:  Recorded in ``posture.generation.generated_by``.
        base_version:  Issue #126 — see ``generate_posture()``'s docstring.
                       Lets the governed version counter continue across a
                       re-derivation into an out-dir whose own
                       ``playbook.opf.json`` doesn't carry the prior
                       Posture (a wiped or freshly re-derived out-dir), by
                       naming the last known version explicitly instead of
                       silently restarting at 1.

    Returns:
        ``PostureApplyResult`` — the new version number, any SHOULD-warn
        messages (one ``playbook floor sign`` warning per sentence-shaped
        Q4 item, issue #104, followed by any messages from
        ``check_posture_floor_conflict()`` checked against the Floor's
        post-promotion state), and the path written.

    Raises:
        FileNotFoundError: no ``playbook.opf.json`` in *out_dir*.
        PostureError:      see ``generate_posture()``; also raised (wrapping
                           a ``floor_candidates.FloorCandidateError``) when
                           a Q4 item's derived id collides with an existing
                           ``floor.invariants`` entry this function's Q4
                           promotion did not itself author — that entry is
                           never silently overwritten, and nothing is
                           written (issue #89 review finding 2).
    """
    opf_path = out_dir / "playbook.opf.json"
    if not opf_path.exists():
        raise FileNotFoundError(
            f"{opf_path} not found — run 'playbook mine' and 'playbook project' first."
        )
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
        base_version=base_version,
    )

    existing_invariants = (doc.get("floor") or {}).get("invariants") or []
    try:
        floor_invariants = promote_interview_q4_invariants(
            answers,
            posture_version=posture["version"],
            existing_invariants=existing_invariants,
        )
    except FloorCandidateError as exc:
        raise PostureError(str(exc)) from exc
    # issue #104: a sentence-shaped Q4 item was skipped, silently, by
    # promote_interview_q4_invariants above (it neither templates nor
    # promotes one) — surface it here as an actionable warning naming the
    # exact `playbook floor sign` command instead of letting it vanish.
    sentence_shaped_warnings = [
        (
            f'Posture interview Q4 ("sacred_clauses") item {item!r} reads as a full '
            "sentence, not a clause-type name — not templated or promoted into "
            f"floor.invariants. To record it verbatim, run: playbook floor sign {out_dir} "
            f'--statement "{item}" --signed-by "<your name>"'
        )
        for item in q4_sentence_shaped_items(answers)
    ]
    warnings = sentence_shaped_warnings + check_posture_floor_conflict(
        posture["system_prompt"], floor_invariants
    )

    doc["posture"] = posture
    # Only rewrite `floor` when promotion actually changed it — e.g. no
    # `sacred_clauses` answer this run, or a true no-op re-run — so a
    # document with no Floor section doesn't gain a fabricated
    # `{"invariants": []}` (and a changed identity.content_hash /
    # section_digests) for no reason. OPF-SPEC.md §3.7 rule 3 makes the
    # section optional; prompt_renderer.py already treats `[]` and absent
    # identically (issue #89 review finding 6).
    if floor_invariants != existing_invariants:
        floor_section = dict(doc.get("floor") or {})
        floor_section["invariants"] = floor_invariants
        doc["floor"] = floor_section
    if "identity" in doc:
        doc["identity"]["content_hash"] = content_hash(doc)
        doc["identity"]["section_digests"] = compute_section_digests(doc)

    write_playbook(doc, opf_path)

    return PostureApplyResult(version=posture["version"], warnings=tuple(warnings), path=opf_path)
