"""Floor-candidate proposal — derive review candidates from reversals + the
Posture interview's Q4 answer (issue #166) — and direct Floor promotion of
that same Q4 answer's human-authored statements (issue #89), plus the
review-checklist accept/reject path for the OTHER (compiler-derived) kind of
candidate (issue #90).

OPF-SPEC.md §3.7 rule 4: the compiler MAY propose Floor candidates
("every ``outcome: proposed_then_reversed`` in the Evidence is a candidate
hard line") and §7 marks the interview's Q4 ("sacred_clauses") as seeding
Floor candidates (see ``posture.py``'s ``seeds_floor_candidates=True``) —
but the legal owner finalizes, and a *compiler-derived* candidate must
NEVER be auto-promoted into the signed OPF ``floor.invariants`` (spec rule
4, "never auto-promote"). Reversal-derived candidates are compiler-derived
— machine inferences read off the Evidence that no human has seen or
endorsed — so they stay firmly on the propose-then-sign-off path.

A Q4 answer is different in kind: it is *human-authored* prose the legal
owner typed directly into the Posture interview, naming their own hard
line. There is no separate machine inference to approve there — the human
act of authorship already IS the sign-off (Marc Mandel, spec owner,
approved this distinction 2026-07-31; OPF-SPEC.md §3.7 rule 4 records it).
So a Q4 answer gets two independent, simultaneously-true treatments here:
it is still offered as PROPOSAL material (``floor.candidates.json``, via
:func:`derive_interview_q4_candidates` below, unchanged since #166 — a
legal owner reviewing that sidecar sees every named clause type in one
place, alongside reversal candidates), and it is ALSO promoted directly
into ``floor.invariants`` (via :func:`promote_interview_q4_invariants`,
called from ``posture.apply_posture_interview`` every time ``playbook
posture interview`` completes — no separate accept step, because the
interview answer already carries the legal owner's sign-off).

This module implements:

  - :func:`derive_reversal_candidates` — one candidate per distinct reversed
    concept (grouped by ``taxonomy_id``, falling back to per-observation
    grouping when unclassified), citing every reversal observation that
    contributed to it. PROPOSAL only — never promoted by this module.
  - :func:`derive_interview_q4_candidates` — one candidate per semicolon-
    separated clause type named in the Q4 ("sacred_clauses") interview
    answer, uncited (the interview names a clause TYPE, not a specific
    document/clause instance).
  - :func:`propose_floor_candidates` — combines both into the locked
    ``floor.candidates.json`` shape (see the issue's "Candidate shape"
    section), pure and deterministic given its inputs.
  - :func:`write_floor_candidates` — I/O orchestration: reads
    ``observations.jsonl`` + the prior compile's ``playbook.opf.json``
    (for the Posture interview's Q4 answer, if a Posture interview has been
    run) from an output directory, and writes ``floor.candidates.json`` next
    to the playbook. This file is a sidecar ENGINE OUTPUT artifact, never
    written into the OPF document itself. :func:`write_floor_candidates`
    itself never writes to ``floor.invariants`` — accepting a REVERSAL
    candidate is always a separate, later, explicit human act (editing
    ``floor.invariants`` directly, via the curation CLI, or via the
    review-checklist path below).
  - :func:`promote_interview_q4_invariants` — one of two exceptions to
    "never writes to floor.invariants" (the other is
    :func:`promote_floor_candidate` below): a pure merge function (the
    caller, ``posture.apply_posture_interview``, handles I/O) that promotes
    each Q4-named item directly into a ``floor.invariants`` list,
    idempotently (OPF-SPEC.md §3.13 — a duplicate sibling id is a blocking
    validator error, so re-running the interview must never append a
    duplicate) and raises :class:`FloorCandidateError` rather than silently
    overwriting an entry it did not itself promote (issue #89 review
    finding 2).
  - :func:`candidate_invariant_id` / :func:`promote_floor_candidate` — issue
    #90's counterpart to the Q4 path, for the review-checklist route into
    ``floor.invariants``: a reversal (or interview_q4-drafted) candidate a
    human reviewer explicitly accepted in ``playbook.review.html``'s
    "Proposed hard lines" checklist. Idempotent by the same §3.13 argument
    as the Q4 path, keyed on a slug of the candidate's ``statement`` rather
    than a Q4 item fragment.
  - :func:`resolve_floor_candidate_decisions` / :func:`apply_floor_review` —
    issue #90's pure resolver and I/O wrapper (respectively) for a
    ``feedback.json`` ``"floor"`` block (``{candidate_id: {"decision":
    "accept"|"reject", "comment": "..."}}``), called from
    ``viewer.apply_feedback``. ``accept`` calls
    :func:`promote_floor_candidate`; ``reject`` records a ``"decision":
    "rejected"`` flag directly on the candidate in ``floor.candidates.json``
    so a later re-render shows it as rejected instead of re-proposing it —
    rejections have no ``floor.invariants`` counterpart to promote into.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from playbook_engine.observation_builder import read_observations_jsonl
from playbook_engine.validator import load_opf_file

__all__ = [
    "FloorCandidate",
    "FloorCandidateCitation",
    "FloorCandidateError",
    "FloorFeedbackResult",
    "apply_floor_review",
    "candidate_invariant_id",
    "candidate_q4_invariant_id",
    "derive_interview_q4_candidates",
    "derive_reversal_candidates",
    "promote_floor_candidate",
    "promote_interview_q4_invariants",
    "propose_floor_candidates",
    "read_floor_candidates",
    "resolve_floor_candidate_decisions",
    "write_floor_candidates",
]

# The Posture interview question that seeds Floor candidates (OPF §7,
# posture.py's INTERVIEW_QUESTIONS — "sacred_clauses", seeds_floor_candidates
# =True). Kept as a local constant rather than importing ``posture`` — this
# module only needs the id string, and staying decoupled from posture.py's
# templating avoids a needless import-time dependency.
INTERVIEW_Q4_ID = "sacred_clauses"

_TEXT_SNIPPET_MAX = 80


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FloorCandidateCitation:
    """One citation in the locked ``floor.candidates.json`` shape (issue #166)."""

    document_id: str
    version: int | str
    clause_path: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "version": self.version,
            "clause_path": self.clause_path,
        }


@dataclass(frozen=True)
class FloorCandidate:
    """One proposed Floor invariant, pending human review (issue #166).

    Attributes:
        id:         ``"cand-NNN"``, 1-indexed, assigned in derivation order
                    (reversal candidates before interview_q4 candidates).
        statement:  NL invariant draft, imperative "Never ..." form.
        rationale:  Human-readable justification for the proposal.
        source:     ``"reversal"`` or ``"interview_q4"``.
        citations:  >=1 for ``source == "reversal"``; ``[]`` for
                    ``source == "interview_q4"`` (the interview names a
                    clause TYPE, not a specific document/clause instance).
    """

    id: str
    statement: str
    rationale: str
    source: str
    citations: list[FloorCandidateCitation] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "statement": self.statement,
            "rationale": self.rationale,
            "source": self.source,
            "citations": [c.to_dict() for c in self.citations],
        }


# ---------------------------------------------------------------------------
# Reversal-sourced candidates
# ---------------------------------------------------------------------------


def _humanize_taxonomy_id(taxonomy_id: str) -> str:
    return taxonomy_id.replace("_", " ").replace("-", " ").strip()


def _text_snippet(text: str) -> str:
    text = " ".join(text.split())  # collapse whitespace
    if len(text) <= _TEXT_SNIPPET_MAX:
        return text
    return text[:_TEXT_SNIPPET_MAX].rsplit(" ", 1)[0] + "..."


def derive_reversal_candidates(
    observations: list[dict[str, Any]],
) -> list[FloorCandidate]:
    """Derive Floor candidates from ``outcome: proposed_then_reversed`` observations.

    Groups observations by ``taxonomy_id`` (an unclassified observation —
    ``taxonomy_id`` is ``None`` — gets its own singleton group, keyed by
    ``observation_id``, so distinct unclassified reversals never collapse
    into one candidate). One candidate per group, citing every reversal
    observation contributing to it (deduplicated, order-preserving).

    Args:
        observations: Raw observation dicts, as returned by
                      ``read_observations_jsonl`` (or ``Observation.to_dict()``).
                      Only ``outcome == "proposed_then_reversed"`` entries
                      contribute; everything else is ignored.

    Returns:
        Candidates in first-seen group order. Empty when there are no
        ``proposed_then_reversed`` observations.
    """
    groups: dict[str, dict[str, Any]] = {}
    order: list[str] = []

    for obs in observations:
        if obs.get("outcome") != "proposed_then_reversed":
            continue
        taxonomy_id = obs.get("taxonomy_id")
        group_key = f"taxonomy:{taxonomy_id}" if taxonomy_id else f"obs:{obs.get('observation_id')}"

        if group_key not in groups:
            groups[group_key] = {
                "taxonomy_id": taxonomy_id,
                "text": obs.get("full_text") or obs.get("text_summary") or "",
                "document_ids": set(),
                "citations": [],
                "seen_citations": set(),
            }
            order.append(group_key)

        group = groups[group_key]
        citation = obs.get("citation") or {}
        document_id = citation.get("document_id")
        version = citation.get("version")
        clause_path = citation.get("clause_path")
        if document_id is not None:
            group["document_ids"].add(document_id)
        if document_id is not None and clause_path is not None and version is not None:
            cite_key = (document_id, version, clause_path)
            if cite_key not in group["seen_citations"]:
                group["seen_citations"].add(cite_key)
                group["citations"].append(
                    FloorCandidateCitation(
                        document_id=document_id, version=version, clause_path=clause_path
                    )
                )

    candidates: list[FloorCandidate] = []
    for group_key in order:
        group = groups[group_key]
        taxonomy_id = group["taxonomy_id"]
        summary = (
            _humanize_taxonomy_id(taxonomy_id)
            if taxonomy_id
            else f'"{_text_snippet(group["text"])}"'
        )
        n_deals = len(group["document_ids"]) or 1
        deal_word = "deal" if n_deals == 1 else "deals"
        candidates.append(
            FloorCandidate(
                id="",  # assigned by propose_floor_candidates
                statement=f"Never accept {summary}.",
                rationale=f"Proposed then reversed before signing in {n_deals} {deal_word}.",
                source="reversal",
                citations=group["citations"],
            )
        )

    return candidates


# ---------------------------------------------------------------------------
# Interview-Q4-sourced candidates
# ---------------------------------------------------------------------------


def _q4_items(answer: str | None) -> list[str]:
    """Split a Q4 ("sacred_clauses") answer into its named clause-type items.

    The interview names clause TYPES as semicolon-separated free text (e.g.
    "uncapped liability; IP assignment") — this is the one splitting rule
    shared by every consumer of the Q4 answer
    (:func:`derive_interview_q4_candidates` below, and
    :func:`promote_interview_q4_invariants`), so they always agree on what
    counts as "one named item."

    Returns:
        Items in the order named, stripped, blank items dropped. Empty when
        *answer* is ``None`` or blank.
    """
    if not answer or not answer.strip():
        return []
    return [item.strip() for item in answer.split(";") if item.strip()]


def _q4_statement(item: str) -> str:
    """The single checkable Floor sentence one Q4-named *item* becomes in a
    PROPOSED candidate (:func:`derive_interview_q4_candidates`, written to
    ``floor.candidates.json``) — a draft a human reviews and rewrites
    before ever signing it into ``floor.invariants``.

    NOT used for the direct ACTIVE-promotion path
    (:func:`promote_interview_q4_invariants`) — see
    :func:`_q4_promoted_statement` for that one. The "Never accept
    {item}." phrasing here inverts a sacred-clause answer's intent (issue
    #89 review finding 3: Q4 names things to KEEP, not things to reject),
    which is tolerable in a draft a human is expected to rewrite, but not
    in a statement that ships live, unedited, with fail-closed consequence
    semantics (OPF-SPEC.md §3.7 rule 1).
    """
    return f"Never accept {item}."


def _q4_promoted_statement(item: str) -> str:
    """The single checkable Floor sentence a Q4-named *item* becomes when
    promoted directly, unedited, into the ACTIVE ``floor.invariants``
    (:func:`promote_interview_q4_invariants`) — deliberately different
    wording from :func:`_q4_statement`'s candidate-draft phrasing.

    Q4 asks which clause TYPES are non-negotiable — i.e. things the legal
    owner insists on KEEPING regardless of deal value, not things to
    reject. :func:`_q4_statement`'s "Never accept {item}." (fine for a
    PROPOSED candidate a human reviews and rewrites before ever signing it)
    inverts that intent when shipped verbatim as an ACTIVE invariant:
    "Never accept Liability caps and student-data protection." tells the
    Floor judge to force negotiation-unacceptable on any clause that
    CONTAINS a liability cap — backwards (issue #89 review finding 3).
    "Do not concede on X." keeps the same imperative, judge-checkable
    register without inverting the polarity.

    Deliberately does NOT echo the interview question's own "regardless of
    deal value" phrase, tempting as that callback is: ``_PROSE_TEMPLATES``'s
    fixed "Flexible to close a deal: {answer}" label means every assembled
    Posture ``system_prompt`` unconditionally contains the words
    "close"/"deal" alongside the softening term "flexible" — a promoted
    statement containing "deal" would then spuriously content-word-overlap
    with that ALWAYS-present sentence, tripping
    ``posture.check_posture_floor_conflict`` on every playbook regardless
    of what was actually said (verified against the ticket's own demo
    answers while fixing review finding 3 — dropping "deal" from this
    wording is what keeps that demo warning-free).
    """
    return f"Do not concede on {item}."


def derive_interview_q4_candidates(
    interview_answers: dict[str, str] | None,
) -> list[FloorCandidate]:
    """Derive Floor candidates from the Posture interview's Q4 ("sacred_clauses") answer.

    Args:
        interview_answers: ``{question_id: answer}`` (the same shape
                           ``posture.generate_posture`` takes, or extracted
                           from a compiled playbook's
                           ``posture.generation.interview``), or ``None``
                           when no Posture interview has been run yet.

    Returns:
        One candidate per semicolon-separated clause type named in the Q4
        answer, in the order named. Empty when Q4 was not answered (missing,
        ``None``, or blank).
    """
    if not interview_answers:
        return []
    items = _q4_items(interview_answers.get(INTERVIEW_Q4_ID))

    return [
        FloorCandidate(
            id="",  # assigned by propose_floor_candidates
            statement=_q4_statement(item),
            rationale=(
                f'Named as non-negotiable in the Posture interview (Q4 "{INTERVIEW_Q4_ID}").'
            ),
            source="interview_q4",
            citations=[],
        )
        for item in items
    ]


# ---------------------------------------------------------------------------
# Combined proposal (pure)
# ---------------------------------------------------------------------------


def propose_floor_candidates(
    observations: list[dict[str, Any]],
    interview_answers: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Assemble the locked ``floor.candidates.json`` shape (issue #166).

    Pure derivation, deterministic given its inputs — no I/O, no LLM. Never
    writes/reads ``floor.invariants``; the caller (:func:`write_floor_candidates`)
    handles I/O, and only a human (or the curation CLI) ever promotes a
    candidate into the signed OPF Floor.

    Args:
        observations:      Raw observation dicts (see
                           :func:`derive_reversal_candidates`).
        interview_answers: See :func:`derive_interview_q4_candidates`.

    Returns:
        ``{"candidates": [...]}`` — reversal-sourced candidates first (in
        first-seen group order), then interview_q4-sourced candidates (in
        answer order), each assigned a stable ``"cand-NNN"`` id.
    """
    all_candidates = derive_reversal_candidates(observations) + derive_interview_q4_candidates(
        interview_answers
    )
    numbered = [
        FloorCandidate(
            id=f"cand-{i:03d}",
            statement=c.statement,
            rationale=c.rationale,
            source=c.source,
            citations=c.citations,
        )
        for i, c in enumerate(all_candidates, start=1)
    ]
    return {"candidates": [c.to_dict() for c in numbered]}


# ---------------------------------------------------------------------------
# Interview-Q4 direct promotion (issue #89) — the one exception to
# "never writes to floor.invariants"
# ---------------------------------------------------------------------------


class FloorCandidateError(Exception):
    """Raised when a Q4 promotion would silently overwrite a
    ``floor.invariants`` entry this module did not itself author (issue
    #89 review finding 2).

    ``posture.apply_posture_interview`` catches this and re-raises it as a
    ``PostureError`` (the CLI already surfaces that type as a clean
    ``ERROR:`` line — see ``cli.py``'s ``posture_interview_cmd``). Defined
    here rather than imported from ``posture.py`` to avoid a circular
    import: ``posture.py`` already imports from this module, not the
    reverse.
    """


_ID_SEP_RE = re.compile(r"[^a-z0-9]+")


def _slugify(text: str, *, fallback: str) -> str:
    """Generic stable kebab-case slug of *text*, or *fallback* if that
    normalizes to nothing (e.g. *text* is empty or all punctuation).

    Shared by :func:`_slugify_statement_item` (Q4 item -> id, issue #89) and
    :func:`candidate_invariant_id` (Floor candidate statement -> id, issue
    #90) — both need the exact same "same text -> same id, every run"
    stability (OPF-SPEC.md §3.13: a duplicate sibling id in
    ``floor.invariants`` is a blocking validator error), just applied to
    different source text.
    """
    slug = _ID_SEP_RE.sub("-", text.strip().lower()).strip("-")
    return slug or fallback


def _slugify_statement_item(item: str) -> str:
    """Stable kebab-case id for one Q4-named clause type.

    Derived from *item* — the semicolon-separated fragment of the Q4
    answer, which is also the variable part of the promoted ``statement``
    (:func:`_q4_statement`) — so the same wording always resolves to the
    same id, this run and every future rerun. That stability is what makes
    :func:`promote_interview_q4_invariants` idempotent: OPF-SPEC.md §3.13
    makes a duplicate sibling id in ``floor.invariants`` a blocking
    validator error, so re-deriving a different id for the same item on a
    later run would grow the list forever instead of updating in place.
    """
    return _slugify(item, fallback="sacred-clause")


_Q4_ATTRIBUTION_RE = re.compile(
    r"^Authored by the legal owner in posture interview v\d+, question "
    + re.escape(INTERVIEW_Q4_ID)
    + r"\.$"
)


def _is_own_q4_promotion(entry: dict[str, Any]) -> bool:
    """Whether *entry* carries THIS function's own attribution marker.

    Distinguishes "an invariant :func:`promote_interview_q4_invariants`
    itself promoted on an earlier run" (safe to update in place on a
    rerun — same id, refreshed ``statement``/``rationale``) from "an
    existing invariant whose id merely happens to collide with a freshly
    Q4-named item's slug" — e.g. a hand-authored, signed-off statement, or
    one written by a different producer entirely (never safe to overwrite;
    issue #89 review finding 2). Matches the EXACT ``rationale`` text
    :func:`promote_interview_q4_invariants` itself writes (below) —
    deliberately narrow, so a hand-written rationale that merely sounds
    similar (or names a different question id) is never mistaken for our
    own marker.
    """
    rationale = entry.get("rationale")
    return isinstance(rationale, str) and bool(_Q4_ATTRIBUTION_RE.match(rationale))


def promote_interview_q4_invariants(
    interview_answers: dict[str, str] | None,
    *,
    posture_version: int,
    existing_invariants: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Promote each Q4 ("sacred_clauses") item directly into ``floor.invariants``.

    Why this is not the auto-promotion OPF-SPEC.md §3.7 rule 4 forbids: that
    rule bars promoting a *compiler-derived* candidate — a machine
    inference read off the Evidence (see :func:`derive_reversal_candidates`)
    that no human has seen — without an explicit accept decision. A Q4
    answer is different in kind: the legal owner typed this exact statement
    into the Posture interview themselves. There is no separate machine
    inference to approve; the human act of authorship already IS the
    sign-off. (Marc Mandel, spec owner, approved this distinction
    2026-07-31; OPF-SPEC.md §3.7 rule 4 records it.) Reversal-derived
    candidates are NOT human-authored — they never take this path; they
    stay on :func:`derive_reversal_candidates`'s propose-then-sign-off path
    via ``floor.candidates.json``, entirely untouched by this function.

    Idempotent per statement — never appends a duplicate ``id`` (OPF-SPEC.md
    §3.13 makes a duplicate sibling id a blocking validator error):

      - an item whose derived id already exists in *existing_invariants*,
        AUTHORED BY AN EARLIER RUN OF THIS SAME FUNCTION (its ``rationale``
        carries this function's own attribution marker — see
        :func:`_is_own_q4_promotion`), with the exact same ``statement`` is
        left byte-for-byte untouched — a true no-op, so re-running the
        interview with unchanged answers leaves ``floor.invariants``
        unchanged;
      - such an entry whose statement has changed (the wording was edited
        but slugifies the same) is updated in place, at the same list
        position, with a fresh ``rationale``;
      - an existing id that collides with a freshly Q4-named item's slug
        but was NOT written by an earlier run of this function (no
        matching attribution marker — e.g. hand-authored and signed off,
        possibly carrying its own ``rationale``/``x_*`` fields) is NEVER
        overwritten: promotion raises :class:`FloorCandidateError` naming
        the conflicting id instead, so the conflict surfaces to the legal
        owner rather than silently destroying a signed statement (issue
        #89 review finding 2);
      - a new item (no id collision at all) is appended, in Q4-answer
        order.

    Invariants this run's Q4 answer doesn't name — hand-authored, or
    promoted by an earlier interview run naming a since-dropped item — are
    left exactly as they are: this is an upsert, never a delete, so a legal
    owner's hand edits (or an earlier run's promotions) are never silently
    discarded. Non-dict entries (a hand-authored playbook MAY carry a bare
    ``floor.invariants[]`` string — ``prompt_renderer.py`` and
    ``document_renderer.py`` both tolerate that shape) are likewise left
    untouched; they simply never match a Q4-derived id (``index_by_id``
    only ever indexes dict entries with a string ``id``).

    Args:
        interview_answers:   ``{question_id: answer}`` — see
                             :func:`derive_interview_q4_candidates`.
        posture_version:     The just-generated ``posture.version`` a
                             freshly-written or freshly-updated invariant's
                             ``rationale`` is attributed to.
        existing_invariants: The playbook's current ``floor.invariants``
                             list (schema-0.2/0.3 shape: dicts with
                             ``id``/``statement``/``rationale``), or
                             ``None``/``[]`` for a first-ever promotion.

    Returns:
        The full, merged ``floor.invariants`` list: pre-existing entries in
        their original order (untouched, or updated in place), then any
        newly-named items appended in Q4-answer order. Equal to
        *existing_invariants* (same order, same content) when Q4 was not
        answered this run.

    Raises:
        FloorCandidateError: a freshly Q4-named item's derived id collides
            with an *existing_invariants* entry this function did not
            itself promote (see above) — never overwritten.
    """
    items = _q4_items((interview_answers or {}).get(INTERVIEW_Q4_ID))
    merged: list[dict[str, Any]] = list(existing_invariants or [])
    if not items:
        return merged

    index_by_id: dict[str, int] = {
        inv["id"]: i
        for i, inv in enumerate(merged)
        if isinstance(inv, dict) and isinstance(inv.get("id"), str)
    }

    for item in items:
        inv_id = _slugify_statement_item(item)
        statement = _q4_promoted_statement(item)
        existing_index = index_by_id.get(inv_id)
        if existing_index is not None:
            # existing_index only ever comes from a dict entry with a
            # string "id" (see index_by_id's construction above) — never a
            # bare-string invariant.
            existing_entry = merged[existing_index]
            if not _is_own_q4_promotion(existing_entry):
                raise FloorCandidateError(
                    f"floor.invariants already has an entry with id {inv_id!r} "
                    f"(statement={existing_entry.get('statement')!r}, rationale="
                    f"{existing_entry.get('rationale')!r}) that this posture "
                    "interview's Q4 promotion did not itself author — refusing "
                    "to silently overwrite it. Rename or remove the conflicting "
                    "invariant, or reword the sacred_clauses answer so it "
                    "slugifies to a different id."
                )
            if existing_entry.get("statement") == statement:
                continue  # true no-op: same id, same statement, nothing changed
        entry = {
            "id": inv_id,
            "statement": statement,
            "rationale": (
                f"Authored by the legal owner in posture interview v{posture_version}, "
                f"question {INTERVIEW_Q4_ID}."
            ),
        }
        if existing_index is not None:
            merged[existing_index] = entry
        else:
            index_by_id[inv_id] = len(merged)
            merged.append(entry)

    return merged


# ---------------------------------------------------------------------------
# Review-checklist candidate promotion (issue #90) — the OTHER route into
# floor.invariants: a reversal (or interview_q4-drafted) candidate a human
# reviewer explicitly accepted in playbook.review.html's "Proposed hard
# lines" checklist, via feedback.json's "floor" key.
# ---------------------------------------------------------------------------


def candidate_invariant_id(candidate: dict[str, Any]) -> str:
    """The stable ``floor.invariants[].id`` a Floor candidate gets if accepted.

    A kebab-case slug of the candidate's ``statement`` — the "same
    statement -> same id, every run" contract :func:`promote_floor_candidate`
    needs to be idempotent (OPF-SPEC.md §3.13: a duplicate sibling id in
    ``floor.invariants`` is a blocking validator error), mirroring
    :func:`_slugify_statement_item`'s role for the Q4 path except keyed on
    the candidate's full statement rather than a Q4 item fragment — a
    reversal candidate has no separate "item" text distinct from its
    statement.

    One of TWO ground-truth signals ``viewer.render_review_html`` uses to
    detect whether a candidate is ALREADY present in ``floor.invariants``
    — the id an acceptance would produce is looked up directly in the
    playbook's current invariant ids, so the "already signed" state is
    always read off the playbook itself, never a possibly-stale flag. For
    a ``source: interview_q4`` candidate specifically, this id ALONE is not
    enough — see :func:`candidate_q4_invariant_id` (issue #90 review
    finding 1).
    """
    candidate_id = candidate.get("id") or "candidate"
    return _slugify(str(candidate.get("statement", "")), fallback=f"floor-{candidate_id}")


_Q4_CANDIDATE_STATEMENT_RE = re.compile(r"^Never accept (?P<item>.+)\.$")


def candidate_q4_invariant_id(candidate: dict[str, Any]) -> str | None:
    """For a ``source: interview_q4`` candidate, the ``floor.invariants[].id``
    its underlying Posture-interview item ALREADY carries if promoted
    directly via :func:`promote_interview_q4_invariants` (issue #89) — the
    OTHER, independently-derived id the exact same Q4 answer item can be
    signed under (issue #90 review finding 1).

    :func:`candidate_invariant_id` slugs THIS candidate's own draft
    ``statement`` (:func:`_q4_statement`'s "Never accept {item}." wording);
    :func:`promote_interview_q4_invariants` instead slugs the bare *item*
    text via :func:`_slugify_statement_item` and writes
    :func:`_q4_promoted_statement`'s "Do not concede on {item}." wording.
    Same underlying Q4 item, two unrelated ids and two unrelated statement
    strings — so ``candidate_invariant_id(candidate) in invariant_ids``
    alone can never detect that this candidate's item was already signed
    via the OTHER (interview-authored) route: a caller that checks only
    :func:`candidate_invariant_id` would let an already-decided item
    through as a live control, and accepting it would append a SECOND,
    opposite-polarity invariant for the same item instead of recognising
    it as already settled. This function reconstructs *item* from the
    candidate's own statement (the exact inverse of :func:`_q4_statement`)
    and re-derives the id :func:`promote_interview_q4_invariants` would use
    for it, so callers can check both ids.

    Returns ``None`` for a ``source: reversal`` candidate (no Q4 item to
    derive from — reversal candidates never take the Q4 promotion path at
    all), or when *candidate*'s ``statement`` doesn't match
    :func:`_q4_statement`'s exact "Never accept {item}." shape (defensive;
    every real ``interview_q4`` candidate does, by construction of
    :func:`derive_interview_q4_candidates`).
    """
    if candidate.get("source") != "interview_q4":
        return None
    match = _Q4_CANDIDATE_STATEMENT_RE.match(str(candidate.get("statement", "")))
    if not match:
        return None
    return _slugify_statement_item(match.group("item"))


_CANDIDATE_ATTRIBUTION_RE = re.compile(
    r"Accepted via review feedback \(floor candidate (?P<cand_id>[^)]+)\)\.$"
)


def _is_own_candidate_promotion(entry: dict[str, Any], candidate_id: str) -> bool:
    """Whether *entry* carries THIS function's own attribution marker for
    *candidate_id* — the :func:`promote_floor_candidate` analogue of
    :func:`_is_own_q4_promotion` (issue #89 review finding 2's foreign-
    collision guard, applied here to the candidate-acceptance path).
    Distinguishes "an invariant a previous acceptance of THIS candidate
    itself wrote" (safe to update in place) from "an existing invariant
    whose id merely happens to collide with this candidate's slug" — a
    hand-authored entry, a Q4 promotion, or an acceptance of a DIFFERENT
    candidate that happened to slugify the same (never safe to overwrite).
    """
    rationale = entry.get("rationale")
    if not isinstance(rationale, str):
        return False
    match = _CANDIDATE_ATTRIBUTION_RE.search(rationale)
    return match is not None and match.group("cand_id") == candidate_id


def _promoted_candidate_rationale(candidate: dict[str, Any]) -> str:
    """The ``floor.invariants[].rationale`` text an ACCEPTED candidate gets.

    Its own proposal rationale (already source-aware — see
    :func:`derive_reversal_candidates`/:func:`derive_interview_q4_candidates`),
    plus its evidence citation(s) when present (reversal-sourced candidates
    only; interview_q4 candidates carry none), plus the "Accepted via review
    feedback" attribution :func:`_is_own_candidate_promotion` matches back
    against on a later idempotent re-apply.
    """
    base = str(candidate.get("rationale", "")).rstrip()
    citations = candidate.get("citations") or []
    evidence = ""
    if citations:
        cite_strs = [
            f"{c.get('document_id')} v{c.get('version')} §{c.get('clause_path')}"
            for c in citations
            if isinstance(c, dict)
        ]
        if cite_strs:
            evidence = f" Evidence: {'; '.join(cite_strs)}."
    candidate_id = candidate.get("id")
    return f"{base}{evidence} Accepted via review feedback (floor candidate {candidate_id})."


def promote_floor_candidate(
    candidate: dict[str, Any],
    *,
    existing_invariants: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Promote ONE explicitly-ACCEPTED Floor candidate into ``floor.invariants``
    (issue #90) — the review-checklist counterpart to
    :func:`promote_interview_q4_invariants`.

    This is not the auto-promotion OPF-SPEC.md §3.7 rule 4 forbids: rule 4
    bars promoting a *compiler-derived* candidate WITHOUT an explicit accept
    decision. This function is only ever called after exactly that decision
    — a reviewer's ``"decision": "accept"`` on a ``feedback.json`` ``"floor"``
    entry (see :func:`resolve_floor_candidate_decisions`, called from
    ``viewer.apply_feedback``). A candidate this function is not explicitly
    told to promote is never written here, or anywhere else in this module.

    Idempotent per candidate (OPF-SPEC.md §3.13 — a duplicate sibling id is
    a blocking validator error, so accepting the same candidate twice, or
    re-applying the same ``feedback.json``, must never append a duplicate):
    the derived id (:func:`candidate_invariant_id`) is stable for a given
    ``statement``, and an existing entry at that id carrying THIS function's
    own attribution marker (:func:`_is_own_candidate_promotion`) is updated
    in place — or, when the statement is unchanged, left byte-for-byte
    untouched (a true no-op) — rather than duplicated. An existing entry at
    that id NOT carrying the marker — hand-authored, written by a Q4
    promotion, or an earlier acceptance of a DIFFERENT candidate that
    happened to slugify the same — is never silently overwritten: promotion
    raises :class:`FloorCandidateError` instead (mirrors issue #89 review
    finding 2).

    A ``source: interview_q4`` candidate gets one MORE guard, on top of the
    above (issue #90 review finding 1): its underlying Posture-interview
    item may ALREADY be signed into ``floor.invariants`` via the OTHER,
    independent promotion route (:func:`promote_interview_q4_invariants`,
    issue #89), under a DIFFERENT id — see :func:`candidate_q4_invariant_id`.
    That id never collides with :func:`candidate_invariant_id`'s (the two
    functions slug different text), so it is checked separately; a
    collision there is refused the same way, rather than appending a
    second, opposite-polarity invariant for an item that is already
    settled.

    Args:
        candidate:            One candidate dict from ``floor.candidates.json``
                              (``id``/``statement``/``rationale``/``source``/
                              ``citations`` — see :class:`FloorCandidate`).
        existing_invariants:  The playbook's current ``floor.invariants``
                              list, or ``None``/``[]`` for a first-ever
                              promotion.

    Returns:
        The full, merged ``floor.invariants`` list. Equal to
        *existing_invariants* (same order, same content) when this exact
        candidate was already promoted with an unchanged statement.

    Raises:
        FloorCandidateError: the derived id collides with an
            *existing_invariants* entry this function did not itself
            promote for THIS candidate — never overwritten; or (issue #90
            review finding 1) a ``source: interview_q4`` candidate's item is
            already signed via :func:`promote_interview_q4_invariants` under
            its own, differently-derived id.
    """
    candidate_id = str(candidate.get("id", ""))
    inv_id = candidate_invariant_id(candidate)
    q4_inv_id = candidate_q4_invariant_id(candidate)
    statement = str(candidate.get("statement", ""))
    rationale = _promoted_candidate_rationale(candidate)

    merged: list[dict[str, Any]] = list(existing_invariants or [])
    index_by_id: dict[str, int] = {
        inv["id"]: i
        for i, inv in enumerate(merged)
        if isinstance(inv, dict) and isinstance(inv.get("id"), str)
    }

    existing_index = index_by_id.get(inv_id)
    if existing_index is not None:
        existing_entry = merged[existing_index]
        if not _is_own_candidate_promotion(existing_entry, candidate_id):
            raise FloorCandidateError(
                f"floor.invariants already has an entry with id {inv_id!r} "
                f"(statement={existing_entry.get('statement')!r}, rationale="
                f"{existing_entry.get('rationale')!r}) that accepting floor "
                f"candidate {candidate_id!r} did not itself author — refusing "
                "to silently overwrite it. Rename or remove the conflicting "
                "invariant, or edit the candidate's statement so it "
                "slugifies to a different id."
            )
        if existing_entry.get("statement") == statement:
            return merged  # true no-op: same id, same statement, nothing changed
        entry = {"id": inv_id, "statement": statement, "rationale": rationale}
        merged[existing_index] = entry
        return merged

    # issue #90 review finding 1: this candidate's own id (inv_id, above)
    # doesn't collide with anything, but a source: interview_q4 candidate's
    # underlying item may STILL already be signed — under the OTHER id
    # promote_interview_q4_invariants (issue #89) would have used for the
    # exact same item. Appending here regardless would add a SECOND,
    # opposite-polarity invariant for an item that is already settled
    # ("Never accept X." alongside an existing "Do not concede on X.") —
    # refuse instead, the same never-silently-duplicate-or-conflict
    # principle as the foreign-collision guard above.
    if q4_inv_id is not None and q4_inv_id in index_by_id:
        q4_entry = merged[index_by_id[q4_inv_id]]
        raise FloorCandidateError(
            f"floor.invariants already has an entry with id {q4_inv_id!r} "
            f"(statement={q4_entry.get('statement')!r}) for the same Posture "
            f"interview Q4 item as floor candidate {candidate_id!r} — already "
            "signed; refusing to add a duplicate, opposite-polarity invariant "
            "for the same item."
        )

    entry = {"id": inv_id, "statement": statement, "rationale": rationale}
    merged.append(entry)
    return merged


# ---------------------------------------------------------------------------
# feedback.json "floor" block resolution (issue #90)
# ---------------------------------------------------------------------------

# Keys recognised inside one feedback.json floor[candidate_id] entry. Mirrors
# viewer.py's _RECOGNIZED_FEEDBACK_KEYS convention: anything else is reported
# as not-applied rather than silently dropped (issue #138).
_FLOOR_ENTRY_KEYS = frozenset({"decision", "comment"})


@dataclass
class FloorFeedbackResult:
    """Result of resolving a ``feedback.json`` ``"floor"`` block (issue #90).

    Attributes:
        invariants:          The (possibly updated) ``floor.invariants``
                             list. The caller writes this into
                             ``doc["floor"]["invariants"]`` iff
                             ``invariants_changed``.
        invariants_changed:  Whether ``invariants`` differs from the
                             *existing_invariants* it was resolved against.
        candidates:           The (possibly updated) ``floor.candidates.json``
                             ``candidates`` list — each accepted/rejected
                             entry now carries a ``"decision"`` field. The
                             caller writes this back iff
                             ``candidates_changed``.
        candidates_changed:   Whether any candidate's ``"decision"`` changed.
        promoted:             Candidate ids successfully accepted (promoted)
                             this run — including a true no-op re-accept, so
                             re-applying identical feedback reports the same
                             counts both times.
        rejected:             Candidate ids marked rejected this run
                             (including a no-op re-reject).
        skipped:              ``candidate_id`` (or the bare ``"floor"`` key
                             for a malformed top-level block) -> human-
                             readable "not applied" messages — the same
                             convention as ``viewer.ApplyResult.skipped``
                             (issue #138).
        comments:             ``(candidate_id, statement, comment)`` triples
                             for the caller to fold into ``viewer_notes.md``,
                             the same free-text sink every other comment
                             uses.
    """

    invariants: list[dict[str, Any]]
    invariants_changed: bool = False
    candidates: list[dict[str, Any]] = field(default_factory=list)
    candidates_changed: bool = False
    promoted: list[str] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)
    skipped: dict[str, list[str]] = field(default_factory=dict)
    comments: list[tuple[str, str, str]] = field(default_factory=list)


def resolve_floor_candidate_decisions(
    floor_feedback: Any,
    candidates: list[dict[str, Any]],
    existing_invariants: list[dict[str, Any]] | None = None,
) -> FloorFeedbackResult:
    """Pure resolution of a ``feedback.json`` ``"floor"`` block (issue #90).

    Resolves *floor_feedback* (the ``feedback["floor"]`` value:
    ``{candidate_id: {"decision": "accept"|"reject", "comment": "..."}}``)
    against *candidates* (``floor.candidates.json``'s ``candidates`` list)
    and *existing_invariants* (the playbook's current ``floor.invariants``).

    No I/O — :func:`apply_floor_review` is the file-reading/-writing wrapper
    ``viewer.apply_feedback`` actually calls. Kept separate and pure so it is
    directly unit-testable with plain dicts, matching every other function
    in this module.

    Per issue #138, nothing is ever silently dropped — anything this
    function cannot honor lands in the result's ``skipped``:

      - *floor_feedback* itself is not an object -> ``skipped["floor"]``.
      - one entry is not an object, names a key other than
        ``decision``/``comment``, or names a ``decision`` other than
        ``"accept"``/``"reject"`` -> ``skipped[f"floor:{candidate_id}"]``.
      - one entry names a ``candidate_id`` absent from *candidates* ->
        ``skipped[f"floor:{candidate_id}"]``.
      - an ``accept`` whose derived id collides with an
        *existing_invariants* entry :func:`promote_floor_candidate` did not
        itself author -> ``skipped[f"floor:{candidate_id}"]`` (the
        :class:`FloorCandidateError` message).

    A ``comment`` is captured into the result's ``comments`` (issue #90
    review finding 3) whenever the named ``candidate_id`` is known,
    REGARDLESS of whether ``decision`` is present/valid — a comment-only
    entry (no ``decision`` key at all) still lands there, even though it is
    ALSO reported via ``skipped`` for the missing decision; the two are
    independent outcomes of the same entry.

    ``accept`` promotes the candidate into the returned ``invariants`` (via
    :func:`promote_floor_candidate`) and marks the candidate's own
    ``"decision": "accepted"`` in the returned ``candidates``. ``reject``
    marks ``"decision": "rejected"`` and never touches ``invariants`` —
    Floor rejections have no ``floor.invariants`` counterpart; the candidate's
    own ``"decision"`` field is the only record of a rejection, and is what
    lets a later re-render show it as rejected instead of re-proposing it.

    Returns:
        :class:`FloorFeedbackResult`.
    """
    result = FloorFeedbackResult(
        invariants=list(existing_invariants or []),
        candidates=list(candidates),
    )

    if not isinstance(floor_feedback, dict):
        result.skipped.setdefault("floor", []).append(
            "'floor' feedback must be an object of "
            '{candidate_id: {"decision": "accept"|"reject", "comment": "..."}}'
        )
        return result

    candidates_by_id: dict[str, dict[str, Any]] = {
        c["id"]: c
        for c in result.candidates
        if isinstance(c, dict) and isinstance(c.get("id"), str)
    }

    for candidate_id, entry in floor_feedback.items():
        if not isinstance(entry, dict):
            result.skipped.setdefault(f"floor:{candidate_id}", []).append(
                "malformed floor feedback entry: expected an object with a 'decision' key"
            )
            continue

        for key in entry:
            if key not in _FLOOR_ENTRY_KEYS:
                result.skipped.setdefault(f"floor:{candidate_id}", []).append(
                    f"{key!r} not yet supported"
                )

        candidate = candidates_by_id.get(candidate_id)
        if candidate is None:
            result.skipped.setdefault(f"floor:{candidate_id}", []).append(
                f"unknown floor candidate id {candidate_id!r} (not present in floor.candidates.json)"
            )
            continue

        # Captured before the decision guard below (issue #90 review
        # finding 3): the page's own Export JS can produce a floor entry
        # with a comment but NO "decision" key at all — a reviewer types a
        # note and leaves the radio on Undecided (viewer.py's
        # collectFeedback skips 'undecided' radios but unconditionally
        # attaches .floor-comment-input text). That comment must still
        # reach viewer_notes.md, the same free-text sink every other
        # comment uses, independent of whether a decision was also given.
        comment = entry.get("comment")
        if comment and str(comment).strip():
            result.comments.append(
                (candidate_id, str(candidate.get("statement", "")), str(comment).strip())
            )

        decision = entry.get("decision")
        if decision not in ("accept", "reject"):
            result.skipped.setdefault(f"floor:{candidate_id}", []).append(
                f"unknown or missing decision {decision!r} (expected 'accept' or 'reject')"
            )
            continue

        if decision == "accept":
            try:
                result.invariants = promote_floor_candidate(
                    candidate, existing_invariants=result.invariants
                )
            except FloorCandidateError as exc:
                result.skipped.setdefault(f"floor:{candidate_id}", []).append(str(exc))
                continue
            result.promoted.append(candidate_id)
            if candidate.get("decision") != "accepted":
                candidate["decision"] = "accepted"
                result.candidates_changed = True
        else:  # "reject"
            result.rejected.append(candidate_id)
            if candidate.get("decision") != "rejected":
                candidate["decision"] = "rejected"
                result.candidates_changed = True

    result.invariants_changed = result.invariants != list(existing_invariants or [])
    return result


# ---------------------------------------------------------------------------
# I/O orchestration
# ---------------------------------------------------------------------------


def _extract_interview_answers(doc: dict[str, Any]) -> dict[str, str] | None:
    """Pull ``{question_id: answer}`` out of a compiled playbook's Posture.

    Returns ``None`` when no Posture interview has been recorded (empty
    ``posture``/``generation``/``interview`` — never a fabricated answer).
    """
    interview = ((doc.get("posture") or {}).get("generation") or {}).get("interview")
    if not interview:
        return None
    return {
        entry["q"]: entry["answer"]
        for entry in interview
        if isinstance(entry, dict) and "q" in entry and "answer" in entry
    }


def write_floor_candidates(out_dir: Path) -> Path:
    """Read ``observations.jsonl`` (+ Posture Q4, if present) from *out_dir*
    and write ``floor.candidates.json`` next to it.

    This is the ``playbook floor propose`` CLI command's I/O layer. The
    playbook's ``floor.invariants`` is never read for input nor written —
    proposals never appear there automatically (spec rule 4).

    Each freshly-derived candidate's ``decision`` (issue #90's accept/reject
    review-checklist field — the ONLY record of a REJECTION, since an
    accepted candidate is separately protected by its presence in
    ``floor.invariants``) is carried over from any prior
    ``floor.candidates.json`` already in *out_dir*, matched by
    ``statement``: ids are reassigned positionally in derivation order on
    every call (see :func:`propose_floor_candidates`), so a newly added or
    removed candidate can shift every later id, but the statement text is
    the stable identity a re-derived candidate is "the same" proposal by
    (:func:`candidate_invariant_id` treats it exactly the same way).
    Without this, re-running ``playbook floor propose`` would silently
    resurrect every previously-rejected (or previously-accepted) candidate
    as a fresh, undecided proposal (issue #90 review finding 4). A
    candidate whose statement no longer recurs (e.g. its reversal evidence
    disappeared) simply drops its stale decision along with itself — there
    is nothing left to carry it to.

    Args:
        out_dir: Output directory produced by ``playbook compile``/``project``.

    Returns:
        Path to the written ``floor.candidates.json``.
    """
    obs_path = out_dir / "observations.jsonl"
    observations = read_observations_jsonl(obs_path)

    interview_answers: dict[str, str] | None = None
    opf_path = out_dir / "playbook.opf.json"
    if opf_path.exists():
        doc = load_opf_file(opf_path)
        interview_answers = _extract_interview_answers(doc)

    result = propose_floor_candidates(observations, interview_answers)

    prior_decisions: dict[str, str] = {
        c["statement"]: c["decision"]
        for c in read_floor_candidates(out_dir)
        if isinstance(c, dict)
        and isinstance(c.get("statement"), str)
        and isinstance(c.get("decision"), str)
    }
    if prior_decisions:
        for candidate in result["candidates"]:
            decision = prior_decisions.get(candidate["statement"])
            if decision is not None:
                candidate["decision"] = decision

    out_path = out_dir / "floor.candidates.json"
    tmp = out_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, out_path)

    return out_path


def read_floor_candidates(out_dir: Path) -> list[dict[str, Any]]:
    """Return the ``candidates`` list from ``out_dir/floor.candidates.json``.

    Empty when the file is absent (no ``playbook floor propose`` run yet) or
    its top-level shape is not the locked ``{"candidates": [...]}`` — an
    unparseable/malformed file is treated as "no candidates" rather than
    raised, so a caller like ``viewer.render_review_html`` degrades to
    rendering no "Proposed hard lines" section instead of crashing the whole
    page render over a corrupt sidecar. That degradation covers a
    non-UTF-8 or otherwise unreadable file too (``UnicodeDecodeError`` /
    ``OSError``), not just invalid JSON — issue #90 review finding 6: before
    the fix, a non-UTF-8 sidecar raised past this function entirely, and the
    CLI's generic ``ValueError`` handler (``UnicodeDecodeError`` IS-A
    ``ValueError``) then misattributed the failure to ``playbook.opf.json``.
    """
    candidates_path = out_dir / "floor.candidates.json"
    if not candidates_path.exists():
        return []
    try:
        raw = json.loads(candidates_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError, OSError):
        return []
    candidates = raw.get("candidates") if isinstance(raw, dict) else None
    return candidates if isinstance(candidates, list) else []


def apply_floor_review(
    out_dir: Path,
    floor_feedback: Any,
    existing_invariants: list[dict[str, Any]] | None = None,
) -> FloorFeedbackResult:
    """I/O wrapper around :func:`resolve_floor_candidate_decisions` (issue #90).

    Reads ``out_dir/floor.candidates.json`` (via :func:`read_floor_candidates`),
    resolves *floor_feedback* against it, and writes ``floor.candidates.json``
    back — atomic tmp+replace, mirroring :func:`write_floor_candidates` —
    iff any candidate's ``decision`` changed.

    Deliberately does NOT write ``playbook.opf.json``: the caller
    (``viewer.apply_feedback``) owns that write, shared with curation-pin
    handling, so ``identity`` is refreshed and the file is written at most
    once per ``apply_feedback`` call regardless of how many sections
    changed this run.

    Args:
        out_dir:              Directory that may contain
                              ``floor.candidates.json``. Absent -> resolved
                              against an empty candidate list, so every
                              ``"floor"`` feedback entry reports "unknown
                              candidate id" rather than crashing (issue
                              #138).
        floor_feedback:       ``feedback["floor"]`` — see
                              :func:`resolve_floor_candidate_decisions`.
        existing_invariants:  The playbook's current ``floor.invariants``
                              list.

    Returns:
        :class:`FloorFeedbackResult`.
    """
    candidates_path = out_dir / "floor.candidates.json"
    candidates = read_floor_candidates(out_dir)

    result = resolve_floor_candidate_decisions(floor_feedback, candidates, existing_invariants)

    if result.candidates_changed:
        tmp = candidates_path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps({"candidates": result.candidates}, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(tmp, candidates_path)

    return result
