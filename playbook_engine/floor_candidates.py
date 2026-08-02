"""Floor-candidate proposal — derive review candidates from reversals + the
Posture interview's Q4 answer (issue #166) — and direct Floor promotion of
that same Q4 answer's human-authored statements (issue #89).

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
    written into the OPF document itself. Accepting a REVERSAL candidate
    stays a human act: editing ``floor.invariants`` directly, or via the
    curation CLI — :func:`write_floor_candidates` never writes to it.
  - :func:`promote_interview_q4_invariants` — the one exception to "never
    writes to floor.invariants": a pure merge function (the caller,
    ``posture.apply_posture_interview``, handles I/O) that promotes each
    Q4-named item directly into a ``floor.invariants`` list, idempotently
    (OPF-SPEC.md §3.13 — a duplicate sibling id is a blocking validator
    error, so re-running the interview must never append a duplicate) and
    raises :class:`FloorCandidateError` rather than silently overwriting an
    entry it did not itself promote (issue #89 review finding 2).
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
    "derive_interview_q4_candidates",
    "derive_reversal_candidates",
    "promote_interview_q4_invariants",
    "propose_floor_candidates",
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
    slug = _ID_SEP_RE.sub("-", item.strip().lower()).strip("-")
    return slug or "sacred-clause"


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

    out_path = out_dir / "floor.candidates.json"
    tmp = out_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, out_path)

    return out_path
