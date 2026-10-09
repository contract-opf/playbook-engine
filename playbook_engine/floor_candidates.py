"""Floor-candidate proposal — derive review candidates from reversals + the
Posture interview's Q4 answer (issue #166) — and direct Floor promotion of
that same Q4 answer's human-authored statements (issue #89).

OPF-SPEC.md §3.7 rule 4: the compiler MAY propose Floor candidates
("every ``outcome: proposed_then_reversed`` in the Evidence is a candidate
hard line") and §7 marks the interview's Q4 ("sacred_clauses") as seeding
Floor candidates — but the legal owner finalizes, and a *compiler-derived*
candidate must NEVER be auto-promoted into the signed OPF
``floor.invariants`` (spec rule 4, "never auto-promote"). Reversal-derived
candidates are compiler-derived — machine inferences read off the Evidence
that no human has seen or endorsed — so they stay firmly on the
propose-then-sign-off path: the legal owner reads ``floor.candidates.json``
(or the table ``playbook floor propose`` prints) and records each hard line
they accept with ``playbook floor sign``.

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

The interview's Q5 ("flexible_clauses" — "which clause types are you happy
to concede") is the mirror-image, human-authored signal (issue #105): where
Q4 names a clause type the legal owner will never give up, Q5 names one
they've already decided is concession material. A freshly re-derived
REVERSAL candidate (:func:`derive_reversal_candidates`) whose ``taxonomy_id``
normalizes equal to a Q5 item is auto-rejected by
:func:`propose_floor_candidates` — not promoted anywhere, never touching
``floor.invariants``, just pre-marked ``"decision": "rejected"`` in
``floor.candidates.json`` so the reader isn't asked to re-litigate a
concession their own interview answer already settled. This is symmetric
with Q4's direct promotion, not a new exception to spec rule 4: the
authority for the rejection is the human-authored Q5 answer, not a
compiler inference.

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
    candidate is always a separate, later, explicit human act
    (``playbook floor sign``).
  - :func:`promote_interview_q4_invariants` — a pure merge function (the
    caller, ``posture.apply_posture_interview``, handles I/O) that promotes
    each Q4-named item directly into a ``floor.invariants`` list,
    idempotently (OPF-SPEC.md §3.13 — a duplicate sibling id is a blocking
    validator error, so re-running the interview must never append a
    duplicate) and raises :class:`FloorCandidateError` rather than silently
    overwriting an entry it did not itself promote (issue #89 review
    finding 2).
  - :func:`sign_floor_invariant` / :func:`sign_invariant_id` — issue #103's
    route into ``floor.invariants`` for a verbatim, hand-authored statement
    (``playbook floor sign``) that has no Q4 item to template — the caller's
    exact wording is written unchanged, e.g. a conditional hard line Q4's
    semicolon-split templating would otherwise garble.
"""

from __future__ import annotations

import datetime
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
    "is_q4_item_sentence_shaped",
    "promote_interview_q4_invariants",
    "propose_floor_candidates",
    "q4_q5_contradictions",
    "q4_sentence_shaped_items",
    "sign_floor_invariant",
    "sign_invariant_id",
    "write_floor_candidates",
]

# The Posture interview question that seeds Floor candidates (OPF §7,
# posture.py's INTERVIEW_QUESTIONS — "sacred_clauses"). Kept as a local
# constant rather than importing ``posture`` — this module only needs the id
# string, and staying decoupled from posture.py's templating avoids a
# needless import-time dependency.
INTERVIEW_Q4_ID = "sacred_clauses"

# The Posture interview question that names willing concessions (OPF §7,
# posture.py's INTERVIEW_QUESTIONS — "flexible_clauses", issue #105). Same
# local-constant rationale as INTERVIEW_Q4_ID above.
INTERVIEW_Q5_ID = "flexible_clauses"

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
        id:           ``"cand-NNN"``, 1-indexed, assigned in derivation order
                      (reversal candidates before interview_q4 candidates).
        statement:    NL invariant draft, imperative "Never ..." form.
        rationale:    Human-readable justification for the proposal.
        source:       ``"reversal"`` or ``"interview_q4"``.
        citations:    >=1 for ``source == "reversal"``; ``[]`` for
                      ``source == "interview_q4"`` (the interview names a
                      clause TYPE, not a specific document/clause instance).
        taxonomy_id:  Clause-taxonomy id this candidate is about (issue
                      #102), for ``source == "reversal"`` — the group key
                      :func:`derive_reversal_candidates` grouped its
                      contributing observations by, BEFORE it gets flattened
                      into this candidate's humanized-prose ``statement``.
                      ``None`` for ``source == "interview_q4"`` (a Q4-named
                      item carries no taxonomy) and for any candidate
                      produced before this field existed — an ADDITIVE key,
                      absent on old ``floor.candidates.json`` files, which
                      must keep loading exactly as before. Lets a
                      taxonomy-aware "already signed" check
                      (``candidate_invariant_id``'s exact-slug match can
                      miss a hand-authored invariant that covers the same
                      taxonomy under different wording) recognise the
                      candidate without needing the two statements to
                      slugify the same.
    """

    id: str
    statement: str
    rationale: str
    source: str
    citations: list[FloorCandidateCitation] = field(default_factory=list)
    taxonomy_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": self.id,
            "statement": self.statement,
            "rationale": self.rationale,
            "source": self.source,
            "citations": [c.to_dict() for c in self.citations],
        }
        # Additive, omitted key rather than an explicit `null` when absent —
        # keeps a candidate with no taxonomy_id (every interview_q4 one, and
        # every reversal one derived before this field existed) byte-
        # identical to the pre-#102 output.
        if self.taxonomy_id is not None:
            d["taxonomy_id"] = self.taxonomy_id
        return d


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


def _group_reversal_observations(
    observations: list[dict[str, Any]],
    *,
    structural_ids: frozenset[str],
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Group ``outcome: proposed_then_reversed`` observations by ``taxonomy_id``.

    Shared grouping step for :func:`derive_reversal_candidates` and
    :func:`count_below_min_deals_reversals` (issue #106) — extracted so both
    call sites agree exactly on what counts as "one group" and its citing
    document count; a second, independently-written grouping loop could
    silently drift from what :func:`derive_reversal_candidates` actually
    dropped.

    An UNCLASSIFIED reversal (``taxonomy_id`` is ``None``) is excluded here —
    see :func:`derive_reversal_candidates`'s docstring. A STRUCTURAL reversal
    (``taxonomy_id`` in *structural_ids*, issue #106 — administrative/
    boilerplate framing curated per taxonomy YAML, e.g. parties-and-recitals)
    is excluded the same way: neither is a proposable hard line.

    Returns:
        ``(groups, order)`` — ``groups`` keyed by ``f"taxonomy:{taxonomy_id}"``,
        each holding ``taxonomy_id``/``text``/``document_ids``/``citations``/
        ``seen_citations``; ``order`` lists group keys in first-seen order.
    """
    groups: dict[str, dict[str, Any]] = {}
    order: list[str] = []

    for obs in observations:
        if obs.get("outcome") != "proposed_then_reversed":
            continue
        taxonomy_id = obs.get("taxonomy_id")
        if not taxonomy_id:
            # Unclassified — excluded by design; see derive_reversal_candidates.
            continue
        if taxonomy_id in structural_ids:
            # Structural (issue #106) — excluded by design; see this
            # function's docstring.
            continue
        group_key = f"taxonomy:{taxonomy_id}"

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

    return groups, order


def derive_reversal_candidates(
    observations: list[dict[str, Any]],
    *,
    structural_ids: frozenset[str] = frozenset(),
    min_deals: int = 1,
) -> list[FloorCandidate]:
    """Derive Floor candidates from ``outcome: proposed_then_reversed`` observations.

    Groups observations by ``taxonomy_id``. One candidate per group, citing
    every reversal observation contributing to it (deduplicated,
    order-preserving).

    An UNCLASSIFIED reversal (``taxonomy_id`` is ``None``) is EXCLUDED. It
    used to get its own singleton candidate keyed by ``observation_id``, on
    the reasoning that distinct unclassified reversals must not collapse into
    one candidate — right about the collapsing, wrong about surfacing them:
    with no taxonomy id the statement is built by quoting the raw clause text
    (see ``_text_snippet`` below), so the legal owner is asked to sign hard
    lines reading ``Do not concede on "shall".`` / ``... "3 3".`` /
    ``... "4 1".`` — segmentation debris, not legal positions. Measured on
    a real 43-document corpus: 67 of 530 reversals were unclassified and produced 67 of
    the 90 candidates, so three-quarters of the review checklist was noise,
    against OPF-SPEC.md §3.7.1's "keep the Floor minimal" admission test.

    A STRUCTURAL reversal (``taxonomy_id`` curated ``structural: true`` in
    the taxonomy YAML, issue #106) is EXCLUDED the same way — administrative/
    boilerplate framing (e.g. parties-and-recitals), not a legal position.
    Measured on that same real corpus: ``parties_and_recitals`` alone
    produced 54 reversal citations, the single most-cited proposed hard
    line, none of it a real negotiating position.

    A group cited by FEWER than *min_deals* distinct documents is also
    EXCLUDED (issue #106) — a single-document reversal is a plausible fluke,
    not a corroborated pattern across the corpus.

    They are never silently dropped: :func:`write_floor_candidates` records
    the omitted counts in ``floor.candidates.json`` so the reviewer can see
    that reversals were set aside and how many. The fix for a corpus with
    many unclassified/structural reversals is better segmentation/
    classification/curation, not a longer checklist.

    Args:
        observations: Raw observation dicts, as returned by
                      ``read_observations_jsonl`` (or ``Observation.to_dict()``).
                      Only ``outcome == "proposed_then_reversed"`` entries
                      contribute; everything else is ignored.
        structural_ids: Taxonomy ids curated ``structural: true`` (issue
                      #106) — see above. Config/taxonomy-owned; the caller
                      (:func:`write_floor_candidates`) derives this set from
                      a loaded ``Taxonomy``. Empty by default, so every
                      pre-#106 call site is unaffected.
        min_deals:    Minimum number of distinct documents that must cite a
                      group before it becomes a candidate (issue #106).
                      Defaults to 1 (no filtering), so every pre-#106 call
                      site is unaffected; the CLI (``playbook floor
                      propose``) raises this to 2 by default (``--min-deals``).
                      See :func:`count_below_min_deals_reversals` for the
                      omitted count.

    Returns:
        Candidates in first-seen group order. Empty when there are no
        ``proposed_then_reversed`` observations (after exclusions/threshold).
    """
    groups, order = _group_reversal_observations(observations, structural_ids=structural_ids)

    candidates: list[FloorCandidate] = []
    for group_key in order:
        group = groups[group_key]
        n_deals = len(group["document_ids"]) or 1
        if n_deals < min_deals:
            continue
        taxonomy_id = group["taxonomy_id"]
        # Always a humanized taxonomy id now: an unclassified reversal never
        # reaches here, so a candidate can no longer quote raw clause text.
        summary = _humanize_taxonomy_id(taxonomy_id)
        deal_word = "deal" if n_deals == 1 else "deals"
        candidates.append(
            FloorCandidate(
                id="",  # assigned by propose_floor_candidates
                statement=f"Do not concede on {summary}.",
                rationale=f"Proposed then reversed before signing in {n_deals} {deal_word}.",
                source="reversal",
                citations=group["citations"],
                taxonomy_id=taxonomy_id,
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


_Q4_SENTENCE_MARKERS: tuple[str, ...] = (" if ", " unless ", " must ", " shall ", " provided ")


def is_q4_item_sentence_shaped(item: str) -> bool:
    """Whether a Q4 ("sacred_clauses") *item* reads as a full sentence or
    conditional clause rather than a bare clause-TYPE name (issue #104).

    :func:`promote_interview_q4_invariants` and :func:`_q4_promoted_statement`
    both assume a Q4 item is a short noun phrase naming a clause type
    ("uncapped liability", "IP assignment") and wrap it verbatim in "Do not
    concede on {item}." — a template built for names, not sentences. A
    *conditional* item the legal owner types free-form — e.g. "limitation
    of liability, if present, must not be unilateral in the counterparty's
    favor" — templates into a nonsense invariant that a fail-closed Floor
    judge then evaluates on every review (issue #104's Problem). This
    heuristic flags that shape deterministically, with no LLM call, so
    :func:`promote_interview_q4_invariants` can refuse to template it and
    ``posture.apply_posture_interview`` can route the legal owner to
    ``playbook floor sign`` instead — the command that records a statement
    exactly as typed, no templating (issue #103).

    Heuristic (either condition trips it):

      - *item* is more than 7 words long (``len(item.split()) > 7``) — a
        bare clause-type name is short; a full sentence usually isn't. The
        threshold is deliberately ``> 7``, not ``>= 7``: a legitimate, if
        wordy, clause-TYPE name ("Limitation of liability and
        consequential damages waiver" — exactly 7 words, no conditional
        marker) must stay name-shaped (issue #104 reviewer gate: this
        exact boundary is probed in tests).
      - *item* contains one of " if ", " unless ", " must ", " shall ",
        " provided " (case-insensitive; *item* is padded with a leading/
        trailing space before matching, so a marker at the very start or
        end of *item* — e.g. an item starting "If present, ...") still
        matches) — words that show up in conditional/imperative prose, not
        in a bare clause-type name.

    Deliberately simple and over-inclusive by design: a false positive
    here costs the legal owner one extra ``playbook floor sign`` command
    (issue #104's Fix); a false negative ships a garbled invariant into a
    fail-closed judge, silently, on every review — the asymmetric cost the
    ticket's Problem section describes.
    """
    if len(item.split()) > 7:
        return True
    padded = f" {item.lower()} "
    return any(marker in padded for marker in _Q4_SENTENCE_MARKERS)


def q4_sentence_shaped_items(interview_answers: dict[str, str] | None) -> list[str]:
    """The Q4 ("sacred_clauses") items in *interview_answers* that
    :func:`is_q4_item_sentence_shaped` flags (issue #104), in the order
    named — using the exact same split as every other Q4 consumer (see
    :func:`_q4_items`), so this always agrees with what
    :func:`promote_interview_q4_invariants` itself skips.

    ``posture.apply_posture_interview`` calls this (separately from
    :func:`promote_interview_q4_invariants`, which independently applies
    the same per-item skip while merging into ``floor.invariants``) to
    build the actionable ``playbook floor sign`` warning for each skipped
    item — assembled in ``posture.py`` rather than here because that
    warning needs the run's ``out_dir``, which this module never sees.

    Returns:
        Sentence-shaped items, in Q4-answer order. Empty when Q4 was not
        answered (missing, ``None``, or blank), or every named item is
        name-shaped.
    """
    items = _q4_items((interview_answers or {}).get(INTERVIEW_Q4_ID))
    return [item for item in items if is_q4_item_sentence_shaped(item)]


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
    return f"Do not concede on {item}."


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
# Interview-Q5-sourced auto-rejection (issue #105)
# ---------------------------------------------------------------------------

# The exact attribution comment stamped onto a REVERSAL candidate
# propose_floor_candidates auto-rejects because its taxonomy was named in
# the Posture interview's Q5 ("flexible_clauses") answer. Fixed text (no
# per-candidate interpolation needed — the candidate's own statement/
# rationale already say which clause type) so a later caller can recognise
# "this rejection came from Q5" rather than a human's own reject decision,
# if that distinction is ever needed.
_Q5_REJECTION_COMMENT = "Posture interview Q5 (flexible_clauses): named as a willing concession."


def _q5_flexible_taxonomy_slugs(interview_answers: dict[str, str] | None) -> set[str]:
    """Normalized (kebab-slug) forms of every clause type named in the
    Posture interview's Q5 ("flexible_clauses") answer.

    Same semicolon-split rule as Q4 (:func:`_q4_items`, reused directly
    rather than duplicated, so Q4 and Q5 always agree on what counts as
    "one named item") — but Q5 items are compared by NORMALIZED slug, not
    verbatim text, because the thing they're matched against
    (:func:`propose_floor_candidates`'s freshly-derived REVERSAL
    candidates) is a machine-humanized ``taxonomy_id`` like
    ``"renewal_notice"`` -> ``"renewal notice"``, not free-form prose a
    legal owner typed. Blank items normalize to nothing and are dropped —
    an all-punctuation Q5 item can never spuriously match every candidate.
    """
    items = _q4_items((interview_answers or {}).get(INTERVIEW_Q5_ID))
    return {slug for item in items if (slug := _slugify(item, fallback=""))}


def q4_q5_contradictions(interview_answers: dict[str, str] | None) -> list[str]:
    """Human-readable warnings for a clause type named in BOTH the Posture
    interview's Q4 ("sacred_clauses" — non-negotiable) and Q5
    ("flexible_clauses" — happy to concede) answers: a self-contradictory
    interview (issue #105 reviewer gate).

    Deliberately advisory, not a hard error or a behavior override — the
    same SHOULD-warn, judgment-first philosophy as
    ``posture.check_posture_floor_conflict`` (Marc Mandel, spec owner,
    approved 2026-07-31 for the sibling Q4-authorship distinction; see this
    module's top docstring). Both of Q4's and Q5's OWN, independent effects
    still happen exactly as if this function didn't exist: Q4's item is
    still promoted into ``floor.invariants``
    (``posture.promote_interview_q4_invariants``) and Q5's item still
    auto-rejects any matching reversal candidate
    (:func:`propose_floor_candidates`) — this function only surfaces the
    tension so the legal owner notices and resolves it (reword one of the
    two answers, or sign the auto-rejected candidate's hard line anyway with
    ``playbook floor sign`` — a human can always override an auto-rejection).

    Matching is EXACT normalized-slug equality (:func:`_slugify`), never
    fuzzy/substring — the same discipline :func:`propose_floor_candidates`
    itself uses for the Q5 rejection match.

    Returns:
        One message per Q4 item whose slug also appears among the Q5
        items, in Q4-answer order. Empty when Q4 or Q5 was not answered, or
        they share no item.
    """
    q4_items = _q4_items((interview_answers or {}).get(INTERVIEW_Q4_ID))
    q5_slugs = _q5_flexible_taxonomy_slugs(interview_answers)
    if not q4_items or not q5_slugs:
        return []

    warnings: list[str] = []
    for item in q4_items:
        slug = _slugify(item, fallback="")
        if slug and slug in q5_slugs:
            warnings.append(
                f'Posture interview names {item!r} in both Q4 ("{INTERVIEW_Q4_ID}", '
                f'non-negotiable) and Q5 ("{INTERVIEW_Q5_ID}", happy to concede) — '
                "contradictory answers. Q4's item is still promoted into "
                "floor.invariants and Q5 still auto-rejects any matching reversal "
                "candidate; resolve by editing one of the two answers."
            )
    return warnings


# ---------------------------------------------------------------------------
# Combined proposal (pure)
# ---------------------------------------------------------------------------


def count_unclassified_reversals(observations: list[dict[str, Any]]) -> int:
    """How many ``proposed_then_reversed`` observations
    :func:`derive_reversal_candidates` set aside for having no
    ``taxonomy_id``.

    Kept as a separate pure function rather than a second return value so
    the derivation's signature stays a plain ``list[FloorCandidate]``. Its
    only purpose is honesty: the exclusion is deliberate (see that
    function's docstring), but a reviewer must be able to see that reversals
    were set aside and how many, rather than reading a short checklist as
    "this is everything the corpus proposed".
    """
    return sum(
        1
        for obs in observations
        if obs.get("outcome") == "proposed_then_reversed" and not obs.get("taxonomy_id")
    )


def count_structural_reversals_omitted(
    observations: list[dict[str, Any]], structural_ids: frozenset[str]
) -> int:
    """How many ``proposed_then_reversed`` observations
    :func:`derive_reversal_candidates` set aside for being classified under
    a taxonomy entry curated ``structural: true`` (issue #106).

    Honesty counterpart to :func:`count_unclassified_reversals`: the
    exclusion is deliberate (see :func:`derive_reversal_candidates`'s
    docstring), but a reviewer must be able to see that boilerplate churn
    was set aside, and how much, rather than reading a short checklist as
    "this is everything the corpus proposed".
    """
    return sum(
        1
        for obs in observations
        if obs.get("outcome") == "proposed_then_reversed"
        and obs.get("taxonomy_id") in structural_ids
    )


def count_below_min_deals_reversals(
    observations: list[dict[str, Any]],
    *,
    min_deals: int,
    structural_ids: frozenset[str] = frozenset(),
) -> int:
    """How many groups :func:`derive_reversal_candidates` would otherwise
    have proposed as a candidate, but dropped, because fewer than
    *min_deals* distinct documents cited them (issue #106).

    Reuses :func:`_group_reversal_observations` — the exact same grouping
    :func:`derive_reversal_candidates` itself uses — so this count can never
    drift from what was actually dropped. A single aggregate count (not a
    per-candidate breakdown), matching :func:`count_unclassified_reversals`'s
    shape.
    """
    groups, order = _group_reversal_observations(observations, structural_ids=structural_ids)
    return sum(1 for key in order if (len(groups[key]["document_ids"]) or 1) < min_deals)


def propose_floor_candidates(
    observations: list[dict[str, Any]],
    interview_answers: dict[str, str] | None = None,
    *,
    structural_ids: frozenset[str] = frozenset(),
    min_deals: int = 1,
) -> dict[str, Any]:
    """Assemble the locked ``floor.candidates.json`` shape (issue #166).

    Pure derivation, deterministic given its inputs — no I/O, no LLM. Never
    writes/reads ``floor.invariants``; the caller (:func:`write_floor_candidates`)
    handles I/O, and only a human (``playbook floor sign``) ever promotes a
    candidate into the signed OPF Floor.

    A freshly-derived REVERSAL candidate (never an interview_q4 one — those
    carry no ``taxonomy_id`` at all, see :class:`FloorCandidate`) whose
    ``taxonomy_id`` normalizes equal to a clause type named in the Q5
    ("flexible_clauses") answer is pre-marked ``"decision": "rejected"``,
    with an attributing ``"comment"`` (issue #105) — matched by the
    ``taxonomy_id`` slug. EXACT normalized-slug equality only, never
    substring/fuzzy — a Q5 item that merely shares a WORD with a taxonomy
    label (e.g. "renewal" vs. "auto-renewal option") must never match. Spec
    note (OPF-SPEC.md §3.7 rule 4): this is not the promotion rule 4 bars —
    nothing is written to ``floor.invariants`` here, and the candidate stays
    listed with its attributing comment; the auto-rejection's authority is
    the human-authored Q5 answer, symmetric with Q4's direct-promotion
    authority (see this module's top docstring).

    Args:
        observations:      Raw observation dicts (see
                           :func:`derive_reversal_candidates`).
        interview_answers: See :func:`derive_interview_q4_candidates`. Also
                           supplies Q5 ("flexible_clauses") for the
                           auto-rejection above.
        structural_ids:    Passed straight through to
                           :func:`derive_reversal_candidates` (issue #106).
        min_deals:         Passed straight through to
                           :func:`derive_reversal_candidates` (issue #106).

    Returns:
        ``{"candidates": [...]}`` — reversal-sourced candidates first (in
        first-seen group order), then interview_q4-sourced candidates (in
        answer order), each assigned a stable ``"cand-NNN"`` id. Also
        carries an additive ``"warnings"`` key (issue #105's Q4/Q5
        contradiction messages, see :func:`q4_q5_contradictions`) — omitted
        entirely, not written as ``[]``, when there's nothing to warn
        about, so the locked empty-input shape (``{"candidates": []}``)
        stays byte-identical to before this key existed.
    """
    all_candidates = derive_reversal_candidates(
        observations, structural_ids=structural_ids, min_deals=min_deals
    ) + derive_interview_q4_candidates(interview_answers)
    numbered = [
        FloorCandidate(
            id=f"cand-{i:03d}",
            statement=c.statement,
            rationale=c.rationale,
            source=c.source,
            citations=c.citations,
            taxonomy_id=c.taxonomy_id,
        )
        for i, c in enumerate(all_candidates, start=1)
    ]
    candidate_dicts = [c.to_dict() for c in numbered]

    q5_slugs = _q5_flexible_taxonomy_slugs(interview_answers)
    if q5_slugs:
        for candidate, candidate_dict in zip(numbered, candidate_dicts, strict=True):
            if candidate.source != "reversal" or not candidate.taxonomy_id:
                continue
            taxonomy_slug = _slugify(candidate.taxonomy_id, fallback="")
            if taxonomy_slug in q5_slugs:
                candidate_dict["decision"] = "rejected"
                candidate_dict["comment"] = _Q5_REJECTION_COMMENT

    result: dict[str, Any] = {"candidates": candidate_dicts}
    warnings = q4_q5_contradictions(interview_answers)
    if warnings:
        result["warnings"] = warnings
    return result


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

    A SENTENCE-SHAPED item (:func:`is_q4_item_sentence_shaped`, issue #104)
    — a conditional hard line the legal owner typed as prose ("limitation
    of liability, if present, must not be unilateral in the counterparty's
    favor"), not a bare clause-type name — is skipped entirely: neither
    templated nor promoted, and treated exactly like an item this run's
    answer simply didn't name (see the upsert-never-delete paragraph
    below), so an EARLIER run's promotion of the same item under a
    now-sentence-shaped rewording is left untouched rather than deleted.
    ``posture.apply_posture_interview`` separately calls
    :func:`q4_sentence_shaped_items` on the same answer to surface an
    actionable warning naming the exact ``playbook floor sign`` command —
    this function itself only ever silently skips; it raises no error and
    emits no warning of its own.

    Invariants this run's Q4 answer doesn't name — hand-authored, or
    promoted by an earlier interview run naming a since-dropped item — are
    left exactly as they are: this is an upsert, never a delete, so a legal
    owner's hand edits (or an earlier run's promotions) are never silently
    discarded. Non-dict entries (a hand-authored playbook MAY carry a bare
    ``floor.invariants[]`` string — ``document_renderer.py``
    tolerates that shape) are likewise left
    untouched; they simply never match a Q4-derived id (``index_by_id``
    only ever indexes dict entries with a string ``id``).

    Args:
        interview_answers:   ``{question_id: answer}`` — see
                             :func:`derive_interview_q4_candidates`.
        posture_version:     The just-generated ``posture.version`` a
                             freshly-written or freshly-updated invariant's
                             ``rationale`` is attributed to.
        existing_invariants: The playbook's current ``floor.invariants``
                             list (schema shape: dicts with
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
        if is_q4_item_sentence_shaped(item):
            # issue #104: a sentence-shaped item is never templated or
            # promoted — treated as unnamed this run (see docstring above);
            # posture.apply_posture_interview surfaces the actionable
            # warning separately, via q4_sentence_shaped_items().
            continue
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
# Hand-authored verbatim signing (issue #103) — the route into
# floor.invariants for a hard line the legal owner words themselves
# (including an accepted floor candidate), alongside
# promote_interview_q4_invariants (Q4 templating). Nothing here is derived or
# templated: the caller supplies the exact
# statement text, and it is written unchanged. This is the path for a
# conditional hard line ("limitation of liability, if present, must not be
# unilateral in the counterparty's favor") that Q4's semicolon-split
# "Do not concede on {item}." templating would otherwise garble into
# nonsense.
# ---------------------------------------------------------------------------

# Storing a bare "taxonomy_id" key on a floor.invariants entry would violate
# spec/playbook.schema-0.5.json's `additionalProperties: false` (only
# id/statement/rationale/`^x_.*` are allowed there) — and OPF-SPEC.md's
# versioning policy (spec/CHANGELOG.md: a published opf_version is frozen;
# any further spec-affecting change goes to a new version) forbids widening
# that FROZEN schema in place. The schema already ships an escape hatch for exactly this
# situation (`patternProperties: {"^x_": true}`), the same one
# pipeline.py's
# `x_quarantined` already uses — so the
# clause taxonomy id this function records goes in under that prefix,
# `x_taxonomy_id`, never a bare `taxonomy_id`. A schema-version bump (or a
# genuine widening of a NOT-yet-frozen version) to promote this to a first-
# class `taxonomy_id` property is out of this ticket's scope.
_SIGN_DEFAULT_RATIONALE = "Hand-authored via `playbook floor sign`."


def sign_invariant_id(statement: str, invariant_id: str | None = None) -> str:
    """The ``floor.invariants[].id`` :func:`sign_floor_invariant` will use for
    *statement*, given an optional caller-supplied *invariant_id* override.

    Exposed separately so a caller — the CLI — can report the id it just
    signed without re-deriving the slug logic itself.

    Args:
        statement:     The verbatim invariant text (see :func:`sign_floor_invariant`).
        invariant_id:  ``--id`` override, or ``None``/blank to derive one from
                       *statement* via :func:`_slugify`.

    Returns:
        *invariant_id* verbatim if non-blank, else a kebab-case slug of
        *statement* (``"floor-invariant"`` if that normalizes to nothing).
    """
    return (
        invariant_id
        if invariant_id and invariant_id.strip()
        else _slugify(statement, fallback="floor-invariant")
    )


def sign_floor_invariant(
    statement: str,
    *,
    invariant_id: str | None = None,
    taxonomy_id: str | None = None,
    rationale: str | None = None,
    signed_by: str | None = None,
    signed_at: str | None = None,
    existing_invariants: list[dict[str, Any]] | None = None,
    required_presence: bool | None = None,
    condition: Any = None,
    permissible_proof: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Record ONE verbatim, hand-authored Floor invariant (issue #103).

    Pure merge function — no I/O, no LLM, no templating. *statement* is
    written into the returned ``floor.invariants`` list byte-for-byte: unlike
    :func:`promote_interview_q4_invariants` (which wraps each Q4-named item in
    "Do not concede on {item}." — a template that garbles a conditional hard
    line) or :func:`promote_floor_candidate` (whose statement started life as
    a compiler-drafted proposal), the caller here is the legal owner
    authoring the exact sentence themselves; there is nothing to reword.

    Collision semantics (simpler than the other two producers' — this
    function never needs an attribution-marker guard, because EVERY entry it
    is asked to write is, by construction, already a human's own considered
    statement, not a machine draft someone else might have authored around):

      - *invariant_id* (or, if omitted, ``slugify(statement)`` via
        :func:`sign_invariant_id`) not present in *existing_invariants* ->
        appended.
      - present, with the SAME ``statement`` already recorded -> idempotent
        no-op: *existing_invariants* is returned completely unchanged (same
        object contents), including its ``rationale``/``x_taxonomy_id`` —
        re-running the identical ``floor sign`` command twice must never
        perturb the document. A run that supplies a different
        ``--rationale``/``--clause`` for an otherwise-unchanged statement is
        ALSO a no-op by this rule; there is no in-place-update path here (by
        design — this command is for adding a new hard line, not editing an
        existing one).
      - present, with a DIFFERENT ``statement`` -> refused
        (:class:`FloorCandidateError`) — never silently overwritten. Pick a
        different ``--id``, or edit/remove the conflicting invariant first.

    Args:
        statement:            The invariant, verbatim. Must be non-blank.
        invariant_id:         ``--id`` override; see :func:`sign_invariant_id`.
        taxonomy_id:          Clause-taxonomy id this invariant is about (CLI-
                              validated against the config-resolved taxonomy
                              before this is called — this function trusts its
                              caller and does no taxonomy lookup itself).
                              Stored as ``x_taxonomy_id`` (see the module-level
                              comment above this function for why not a bare
                              ``taxonomy_id``), omitted entirely when
                              ``None``/blank.
        rationale:            Legal justification text ONLY — never the
                              signer's name or a sign-off date; *signed_by*/
                              *signed_at* already record that structurally,
                              and rationale ships verbatim into every
                              consumer's model-facing review prompt (issue
                              #209). Defaults to :data:`_SIGN_DEFAULT_RATIONALE`
                              when ``None``/blank.
        signed_by:            Name of the human legal owner signing this
                              statement — REQUIRED, not optional (issue
                              #127): unlike *rationale*, which is free-form
                              text an agent could type to merely resemble a
                              genuine sign-off, this is the structural
                              attribution ``validate`` checks for. Stored
                              verbatim as ``x_signed_by`` (see the
                              module-level comment above this function for
                              why ``x_``-prefixed, not a bare property).
        signed_at:            ISO-8601 timestamp this statement was signed.
                              ``None``/blank defaults to now (UTC) — callers
                              that need a fixed value (tests, replaying a
                              past sign-off) pass one explicitly. Stored
                              verbatim as ``x_signed_at``.
        existing_invariants:  The playbook's current ``floor.invariants``
                              list, or ``None``/``[]`` for a first-ever sign.
        required_presence:    Whether the clause's absence or deletion is a hard
                              rejection (issue #228). Stored as
                              ``x_required_presence`` when given; the hard-rule
                              manifest reads it (default there: ``False``).
        condition:            A machine-evaluable predicate spec
                              (``dossiers.validate_condition``) or ``"judged"``.
                              Stored as ``x_condition`` when given.
        permissible_proof:    What a reviewed document may show to satisfy the
                              rule. Stored as ``x_permissible_proof`` when
                              non-empty.

    Returns:
        The full, merged ``floor.invariants`` list: pre-existing entries
        untouched, in their original order, plus this statement appended (or
        nothing appended, for the no-op case above).

    Raises:
        FloorCandidateError: *statement* is blank, *signed_by* is blank,
            *invariant_id* (or its derived slug) collides with an existing
            entry carrying a different statement, or *rationale* names the
            signer (issue #209) for a NEW entry (never raised on the
            idempotent-no-op path above, so an exact rerun of a prior,
            already-recorded call never fails on this check).
    """
    if not isinstance(statement, str) or not statement.strip():
        raise FloorCandidateError("floor sign: --statement must be a non-empty string")
    if not isinstance(signed_by, str) or not signed_by.strip():
        raise FloorCandidateError(
            "floor sign: --signed-by must name the human legal owner signing this "
            "hard line — a Floor invariant must carry a structural record of who "
            "signed it, not just free-form rationale text."
        )

    if condition is not None:
        from playbook_engine.dossiers import validate_condition  # noqa: PLC0415

        why = validate_condition(condition)
        if why:
            raise FloorCandidateError(f"floor sign: --condition is not valid: {why}")
    proof = [p for p in (permissible_proof or []) if isinstance(p, str) and p.strip()]

    inv_id = sign_invariant_id(statement, invariant_id)
    rationale_text = rationale if rationale and rationale.strip() else _SIGN_DEFAULT_RATIONALE

    merged: list[dict[str, Any]] = list(existing_invariants or [])
    index_by_id: dict[str, int] = {
        inv["id"]: i
        for i, inv in enumerate(merged)
        if isinstance(inv, dict) and isinstance(inv.get("id"), str)
    }

    existing_index = index_by_id.get(inv_id)
    if existing_index is not None:
        # existing_index only ever comes from a dict entry with a string
        # "id" (see index_by_id's construction above) — never a bare-string
        # invariant.
        existing_entry = merged[existing_index]
        if existing_entry.get("statement") == statement:
            return merged  # true no-op: same id, same statement, nothing changed
        raise FloorCandidateError(
            f"floor.invariants already has an entry with id {inv_id!r} "
            f"(statement={existing_entry.get('statement')!r}) that does not "
            f"match the statement being signed ({statement!r}) — refusing to "
            "overwrite. Pass a different --id, or edit/remove the conflicting "
            "invariant first if it should be replaced."
        )

    # issue #209 (2026-08-24 skill QA audit, finding #92): --signed-by is the
    # ONE structural home for sign-off attribution — rationale is legal
    # justification only. rationale ships verbatim into every consumer's
    # model-facing review prompt (a consumer renders each invariant as
    # "{statement} ({rationale})"), while x_signed_by/
    # x_signed_at are never sent to a model. A rationale that repeats the
    # signer's name (e.g. "Hand-authored and signed by the legal owner
    # (Jane Doe, GC), 2026-08-21.") duplicates that attribution into the
    # one field this engine cannot keep confidential — refuse it instead of
    # writing it, rather than relying on every consumer to notice.
    if signed_by.strip().casefold() in rationale_text.casefold():
        raise FloorCandidateError(
            f"floor sign: --rationale must not name who signed this invariant "
            f"— it contains {signed_by.strip()!r}, which --signed-by already "
            "records structurally (as x_signed_by/x_signed_at, never rendered "
            "into a review prompt). Rewrite --rationale to state the legal "
            "justification only, with no name or sign-off date."
        )

    signed_at_text = (
        signed_at.strip()
        if isinstance(signed_at, str) and signed_at.strip()
        else datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")
    )
    entry: dict[str, Any] = {
        "id": inv_id,
        "statement": statement,
        "rationale": rationale_text,
        "x_signed_by": signed_by.strip(),
        "x_signed_at": signed_at_text,
    }
    if taxonomy_id and taxonomy_id.strip():
        entry["x_taxonomy_id"] = taxonomy_id
    # The hard-rule manifest's fields (issue #228); each only when stated.
    if required_presence is not None:
        entry["x_required_presence"] = required_presence
    if condition is not None:
        entry["x_condition"] = condition
    if proof:
        entry["x_permissible_proof"] = proof
    from playbook_engine.dossiers import floor_rule_errors  # noqa: PLC0415

    rule_errors = floor_rule_errors(entry)
    if rule_errors:
        raise FloorCandidateError(
            "floor sign: " + "; ".join(rule_errors) + " (pass --clause to name the clause)"
        )
    merged.append(entry)
    return merged


def floor_invariant_attribution(entry: dict[str, Any]) -> str | None:
    """WHO structurally attributes *entry* (one ``floor.invariants[]``
    item) — or ``None`` if nothing does (issue #127).

    Two producers write into ``floor.invariants``, each leaving a
    distinct, mechanically-checkable trace:

      - ``playbook floor sign`` (:func:`sign_floor_invariant`) stamps
        ``x_signed_by`` with the human name the CLI now REQUIRES at sign
        time — returns ``"signed"``.
      - the Posture interview's Q4 answer
        (:func:`promote_interview_q4_invariants`) stamps a ``rationale``
        matching :data:`_Q4_ATTRIBUTION_RE` — returns ``"posture_interview"``.

    Returns ``None`` when neither marker is present — an
    invariant that never travelled through any of this engine's own
    attribution paths, which ``validate`` (:mod:`playbook_engine.validator`)
    surfaces as a non-blocking warning so a human review can look at it.

    Deliberately NOT a security boundary on its own for the interview case:
    that ``rationale`` marker is lexical, and text that merely resembles it
    could in principle be typed by hand. The path
    this genuinely closes is ``floor sign``, which no longer accepts a
    human sign-off as free-form ``rationale`` text — it now REQUIRES the
    structural ``x_signed_by`` field, refusing to run without it (see
    :func:`sign_floor_invariant`).
    """
    if not isinstance(entry, dict):
        return None
    signed_by = entry.get("x_signed_by")
    if isinstance(signed_by, str) and signed_by.strip():
        return "signed"
    rationale = entry.get("rationale")
    if not isinstance(rationale, str):
        return None
    if _Q4_ATTRIBUTION_RE.match(rationale):
        return "posture_interview"
    return None


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


def write_floor_candidates(
    out_dir: Path,
    *,
    structural_ids: frozenset[str] = frozenset(),
    min_deals: int = 1,
) -> Path:
    """Read ``observations.jsonl`` (+ Posture Q4, if present) from *out_dir*
    and write ``floor.candidates.json`` next to it.

    This is the ``playbook floor propose`` CLI command's I/O layer. The
    playbook's ``floor.invariants`` is never read for input nor written —
    proposals never appear there automatically (spec rule 4).

    Args:
        out_dir:        Output directory produced by ``playbook mine``/``project``.
        structural_ids: Taxonomy ids curated ``structural: true`` (issue
                       #106), passed straight through to
                       :func:`propose_floor_candidates`. Empty by default
                       (no structural exclusion) so every pre-#106 caller
                       is unaffected; the CLI (``playbook floor propose
                       --config``) is the only caller that supplies this.
        min_deals:      Minimum distinct-document citation threshold (issue
                       #106), passed straight through to
                       :func:`propose_floor_candidates`. Defaults to 1 (no
                       filtering) here so every pre-#106 direct caller of
                       this function is unaffected; the CLI's own
                       ``--min-deals`` default is 2 (``playbook floor
                       propose``), applied by the caller, not by this
                       function's default.

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

    result = propose_floor_candidates(
        observations, interview_answers, structural_ids=structural_ids, min_deals=min_deals
    )
    # Additive sibling keys — `read_floor_candidates` reads `raw.get("candidates")`
    # and tolerates siblings. Always present (0 when nothing was set aside), so
    # their absence never has to be interpreted.
    result["unclassified_reversals_omitted"] = count_unclassified_reversals(observations)
    result["structural_reversals_omitted"] = count_structural_reversals_omitted(
        observations, structural_ids
    )
    result["below_min_deals_omitted"] = count_below_min_deals_reversals(
        observations, min_deals=min_deals, structural_ids=structural_ids
    )

    out_path = out_dir / "floor.candidates.json"
    tmp = out_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, out_path)

    return out_path
