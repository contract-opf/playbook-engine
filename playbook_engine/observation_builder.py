"""Observation builder — L4 → L5 bridge.

Assembles one inspectable row per clause observation, writing them to
``observations.jsonl`` (one JSON object per line). The deal is the unit of
precedent (issue #216): each document contributes exactly one terminal
(signed/unsigned) observation per taxonomy_id, built from the terminal
version's own text, plus one ``proposed_then_reversed`` row per reversal.
A clause removed before signing (no terminal slot) is classified by the
ORIGIN of its text: our standard language struck is one
``conceded_before_signing`` row (only when the deal has a detected
executed copy), non-standard language struck is one
``proposed_then_reversed`` row, and removed text that survives in the
terminal, whose origin is undetermined, or that is our standard in a deal
with no detected executed copy produces no row at all — it is counted in
``corpus.stats.dropped_observations``. See ``build_observations``.

Each observation captures:
  - What was observed: taxonomy_id, text_summary (a ≤ 300-char prefix of the
    clause text ending on a sentence boundary — see summarize_clause_text —
    display-only; full_text carries the untruncated clause text used by
    downstream judges/standards, issue #105)
  - Citation: document_id, version, version_id, clause_path, char_span (see
    ObservationCitation for why version alone is not file-resolvable — issue #108)
  - Deviation assessment: deviation, risk_delta (from the deviation classifier)
  - Provenance: whose paper the document is on (OPF §2.2)
  - Outcome: "signed", "unsigned", "proposed_then_reversed" (from reversal
    detector, or a non-standard clause removed before signing), or
    "conceded_before_signing" (our standard language removed before signing
    in a deal with a detected executed copy; engine-internal, never an OPF
    outcome). A terminal row is "unsigned" when no version was detected as
    the executed copy — see build_observations' has_signed_copy
  - Source: document_id + version for traceability

Only in-scope documents are included; out-of-scope decisions from the scope
gate are respected (scope_decision.in_scope must be True).

The file is written atomically via ``os.replace()`` to prevent partial writes.
"""

from __future__ import annotations

import dataclasses
import datetime
import json
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from playbook_engine.clause_classifier import ClassifiedClause
from playbook_engine.clause_differ import ClauseDiff, DocumentDiff
from playbook_engine.deviation_classifier import (
    REWORDED_EQUIVALENT_THRESHOLD,
    DeviationResult,
    _text_jaccard,
)
from playbook_engine.deviation_classifier import (
    _normalize_for_containment as _normalize_for_origin,
)
from playbook_engine.docx_ingester import TrackedChanges
from playbook_engine.reversal_detector import ReversalRecord
from playbook_engine.reversal_detector import _tokens as _reversal_tokens
from playbook_engine.tracked_changes_overlay import (
    HunkEnrichment,
    enrich_clause_diff,
    round_level_fallback_attribution,
)

# Cap for RoundMove.change_summary (see truncate_move_summaries) — the
# born-safe store's what-moved line, a separate field from text_summary.
_CHANGE_SUMMARY_MAX = 200

# Cap for Observation.text_summary (issue #217) — see summarize_clause_text.
_TEXT_SUMMARY_MAX = 300

# A sentence cut shorter than this is a heading-like fragment
# ("Confidentiality.", "5.") that would make the summary merely restate the
# clause name, so summarize_clause_text falls back to a word boundary.
_TEXT_SUMMARY_MIN_SENTENCE = 60

# A sentence end: terminal punctuation, optionally followed by closing
# quotes/brackets, then whitespace or end of text. "2.1" (no whitespace
# after the dot) is not a sentence end; "Inc. " is — an accepted heuristic
# cost, since a cut there is still a clean word boundary.
_SENTENCE_END = re.compile(r"[.!?][\"'\u201d\u2019)\]]*(?=\s|$)")


def summarize_clause_text(text: str, limit: int = _TEXT_SUMMARY_MAX) -> str:
    """Display summary of a clause's text: its first ≤ *limit* chars, ending
    on a sentence boundary (issue #217).

    Replaces the old hard ``text[:200]`` cut, which ended mid-word. The
    result is always a verbatim prefix of the stripped text:

    1. Text that fits within *limit* is returned whole.
    2. Otherwise, cut after the LAST sentence end inside the first *limit*
       chars — unless that keeps fewer than ``_TEXT_SUMMARY_MIN_SENTENCE``
       chars (a heading-like fragment such as "Confidentiality."), in which
       case, as when the window holds no sentence end at all:
    3. fall back to the last word boundary inside the window;
    4. only a single unbroken token longer than *limit* is hard-cut.
    """
    text = text.strip()
    if len(text) <= limit:
        return text
    sentence_cut = 0
    # Search one char past the window so a sentence end AT the window edge
    # is recognised by the lookahead against the real following char.
    for m in _SENTENCE_END.finditer(text, 0, limit + 1):
        if m.end() <= limit:
            sentence_cut = m.end()
    if sentence_cut >= min(_TEXT_SUMMARY_MIN_SENTENCE, limit):
        return text[:sentence_cut]
    window = text[:limit]
    if text[limit].isspace():
        return window.rstrip()
    word_cut = max(window.rfind(" "), window.rfind("\n"), window.rfind("\t"))
    if word_cut > 0:
        return window[:word_cut].rstrip()
    return window


# Target cap for Observation.search_snippet (issue #95) — "a phrase," not a
# paragraph: ~40-100 chars / roughly 5-15 words is enough for a reviewer to
# Ctrl+F the source document, short enough to read as a search anchor rather
# than a second text_summary. See truncate_search_snippets() for why this cap
# is applied separately from — and strictly after — full_text/search_snippet
# pseudonymization, never at construction time.
_SEARCH_SNIPPET_MAX = 100


# Minimum author-string length for the author-in-alias containment direction.
# DOCX w:author values are frequently initials or short handles ("Al", "IT");
# a 1-3 char author is a substring of almost any alias, so matching it would
# systematically flip counterparty edits to "us".
_MIN_AUTHOR_CONTAINMENT_LEN = 4


def party_side_for_author(
    author: str | None,
    our_party_aliases: list[str],
    our_authors: list[str] | None = None,
) -> str:
    """Map a tracked-changes author name to a negotiation side (issue #177).

    Checked against two distinct config lists — ``config.provenance.
    our_party_aliases`` (entity/org names, e.g. "FixtureCorp") and
    ``config.provenance.our_authors`` (people: personal names, initials,
    and/or email addresses actually found in DOCX ``w:author`` metadata,
    issue #119) — because a tracked-change author is a *person*, a
    fundamentally different namespace from an org alias that was never
    going to match it by containment. Case-insensitive containment applies
    to both lists identically ("FixtureCorp Legal" matches alias
    "FixtureCorp"; "J. Smith" matches author "J. Smith (Legal)"). The
    reverse direction (candidate contained in the author string) only
    applies to authors of ``_MIN_AUTHOR_CONTAINMENT_LEN``+ chars — Word
    author strings are often initials, and "IT" ⊂ "Summit Health" must not
    read as "us".

    Returns "us" on a match against either list, else "unknown" — never
    "counterparty". Not matching our side is not evidence of matching
    theirs: a corpus can easily have real counterparty-side authors it has
    never seen before, or "us"-side authors missing from either list. This
    holds symmetrically whether both lists are empty (nothing configured to
    discriminate against) or non-empty with no match (issue #119 — an
    unconfigured or under-configured corpus must not publish our own
    attorneys' edits, or unrecognized authors of either side, as
    counterparty asks) — a side is never guessed (§3.5.3).
    """
    if not author:
        return "unknown"
    author_lower = author.lower()
    for candidates in (our_party_aliases, our_authors or []):
        for candidate in candidates:
            candidate_lower = candidate.lower()
            if not candidate_lower:
                continue
            if candidate_lower in author_lower:
                return "us"
            if len(author_lower) >= _MIN_AUTHOR_CONTAINMENT_LEN and author_lower in candidate_lower:
                return "us"
    return "unknown"


def _date_from_tracked(date_str: str | None) -> str | None:
    """Extract a plain ISO date from a tracked-change ``w:date`` timestamp.

    Returns ``None`` unless the first 10 characters parse as a real ISO-8601
    date — dynamics fields are omitted, never fabricated (issue #177).
    """
    if not date_str or len(date_str) < 10:
        return None
    candidate = date_str[:10]
    try:
        datetime.date.fromisoformat(candidate)
    except ValueError:
        return None
    return candidate


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ObservationCitation:
    """Traceability reference (OPF §4).

    ``version`` is a display ordinal (e.g. ``3`` meaning "3rd version in
    negotiation order", or ``"template"``) — it is NOT mechanically
    resolvable to a file on disk, since normalized trees are stored under
    their original filename stem (``normalized/<doc>/<stem>.clauses.json``),
    not under the ordinal (issue #108). ``version_id`` carries that actual
    stem alongside the ordinal so a citation like "doc v3 §5.2" can be
    resolved to ``normalized/doc/<version_id>.clauses.json`` directly, with
    no glob-every-version workaround required. ``None`` when the clause_path
    this citation points at has no known source version (should not happen
    for real corpus documents; kept optional for callers — e.g. some tests —
    that never had a real version id to give).
    """

    document_id: str
    version: int | str
    clause_path: str
    char_span: tuple[int, int] | None
    version_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "version": self.version,
            "clause_path": self.clause_path,
            "char_span": list(self.char_span) if self.char_span else None,
            "version_id": self.version_id,
        }


@dataclass(frozen=True)
class Observation:
    """One clause observation feeding the L5 compiler.

    Attributes:
        observation_id:  Unique string id for this observation (caller-supplied).
        taxonomy_id:     Taxonomy entry, or ``None`` for unclassified clauses.
        text_summary:    A ≤ 300-char prefix of the clause text ending on a
                         sentence boundary (``summarize_clause_text``,
                         issue #217). Display-only —
                         a human-scanning summary. NOT the source for
                         our_standard.text, acceptable_if, fallback/rejected
                         language, or any judge payload; use full_text for
                         those (issue #105 — a 200-char fragment is not a
                         usable drafting standard or negotiable-alternative
                         text for any real indemnification/insurance clause).
        citation:        Traceability reference to the source version.
        deviation:       How the clause deviates from our standard.
        risk_delta:      Direction and magnitude of risk shift.
        provenance:      ``"our_paper"`` or ``"counterparty_paper"``.
        outcome:         ``"signed"``, ``"unsigned"``, ``"proposed_then_reversed"``,
                         or ``"conceded_before_signing"`` (issue #216: OUR
                         standard language removed before signing — our
                         concession, never a refused ask; engine-internal
                         like ``"unsigned"``, so it never reaches
                         ``observed_positions``, and the position compiler
                         counts it only as a conceded deal).
                         ``"unsigned"`` marks a clause from a document with no
                         detected executed copy (issue #83) — the position
                         compiler and clause library only ever treat
                         ``outcome == "signed"`` as accepted-position evidence,
                         so ``"unsigned"`` observations are excluded from
                         those rollups by construction, not by a separate
                         filter.
        confidence:      Classification confidence in [0, 1], or ``None`` when
                         the clause is unclassified or confidence is unavailable.
        basis:           How the deviation assessment was reached (e.g.
                         ``"deterministic"``, ``"judge"``), or ``None`` for
                         observations that bypass the deviation classifier
                         (e.g. template observations).
        attribution:     Word tracked-changes author/date attribution for this
                         clause's hunks (issue #88), or ``None`` when no
                         tracked-changes side-channel matched — PDF/RTF, a
                         clean DOCX, a DOCX ``python-docx`` could not open, or
                         a DOCX redline whose text didn't match closely
                         enough (see ``tracked_changes_overlay``). Captured
                         the same way regardless of segmentation mode (issue
                         #85). This is a bonus signal, never a requirement —
                         most observations will have ``attribution=None``.
        full_text:       The untruncated clause text (issue #105). Defaults
                         to ``text_summary`` when not supplied (via
                         ``__post_init__``) so existing callers/tests that
                         only ever dealt in short synthetic text keep
                         working unchanged; real callers pass the actual
                         full clause text explicitly. This is the field
                         our_standard.text, acceptable_if, and fallback/
                         rejected language must resolve from — never
                         text_summary.
        search_snippet:  Short verbatim excerpt near the citation's location,
                         for a reviewer to Ctrl+F in the source document
                         (issue #95 — replaces the page-number approach from
                         #86, since docling/DOCX/RTF extraction never
                         populates a real page). Defaults to ``full_text``
                         when not supplied (via ``__post_init__``), mirroring
                         ``full_text``'s own default-from-``text_summary``
                         cascade — every construction site that already sets
                         ``full_text`` gets a snippet source for free. Kept
                         UNTRUNCATED at construction, exactly like
                         ``RoundMove.change_summary`` (see
                         ``_summarize_move``'s docstring): truncating a raw
                         clause-text excerpt before it is pseudonymized can
                         bisect a counterparty name mid-word and defeat the
                         whole-word aliasing match, leaking the fragment. The
                         pipeline pseudonymizes this field alongside
                         ``full_text`` and only then calls
                         ``truncate_search_snippets`` to cap its length —
                         never the other way around.
    """

    observation_id: str
    taxonomy_id: str | None
    text_summary: str
    citation: ObservationCitation
    deviation: str
    risk_delta: dict[str, str]  # {"direction": ..., "magnitude": ...}
    provenance: str
    outcome: str
    confidence: float | None = None
    basis: str | None = None
    attribution: HunkEnrichment | None = None
    full_text: str = ""
    search_snippet: str = ""
    # Negotiation dynamics (issue #177, OPF §3.5.3) — all optional-when-
    # underivable, never fabricated. proposed_by/observed_at derive from the
    # tracked-changes side-channel in build_observations; counterparty_ref
    # ({"alias": ...}) is attached by the pipeline's pseudonymization pass,
    # the only place a deal→known-entity match exists.
    proposed_by: str | None = None
    observed_at: str | None = None
    counterparty_ref: dict[str, str] | None = None

    def __post_init__(self) -> None:
        if not self.full_text:
            # frozen dataclass — object.__setattr__ is the sanctioned escape hatch.
            object.__setattr__(self, "full_text", self.text_summary)
        if not self.search_snippet:
            object.__setattr__(self, "search_snippet", self.full_text)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "observation_id": self.observation_id,
            "taxonomy_id": self.taxonomy_id,
            "text_summary": self.text_summary,
            "full_text": self.full_text,
            "citation": self.citation.to_dict(),
            "deviation": self.deviation,
            "risk_delta": self.risk_delta,
            "provenance": self.provenance,
            "outcome": self.outcome,
            "confidence": self.confidence,
            "basis": self.basis,
            "attribution": self.attribution.to_dict() if self.attribution is not None else None,
        }
        # Dynamics keys are present only when derived — an absent key is the
        # "underivable" signal itself (issue #177), so no null placeholders.
        if self.proposed_by is not None:
            d["proposed_by"] = self.proposed_by
        if self.observed_at is not None:
            d["observed_at"] = self.observed_at
        if self.counterparty_ref is not None:
            d["counterparty_ref"] = self.counterparty_ref
        # search_snippet (issue #95): present only when there is real clause
        # text to excerpt from — an empty string (no clause text available)
        # is omitted rather than round-tripped as a useless "" entry.
        if self.search_snippet:
            d["search_snippet"] = self.search_snippet
        return d


@dataclass(frozen=True)
class RoundMove:
    """One round-scoped clause move for ``negotiation_trail`` (issue #177).

    Built from ``DocumentDiff.consecutive`` — the per-round diffs the
    pipeline previously computed and discarded. ``taxonomy_id`` exists for
    L5 grouping only and is NOT part of the OPF trail-entry shape (the
    entry already lives inside the taxonomy-anchored ClausePosition);
    ``to_opf_dict()`` is the schema-conformant serialization.

    ``citation`` is the post-move state for added/modified clauses; for a
    removed clause the post-move state does not exist, so it cites the last
    state where the clause did (the before side) — resolvability beats a
    dangling post-move ref.
    """

    document_id: str
    round: int  # version-transition ordinal (v2→v3 = round 2)
    taxonomy_id: str | None
    moved_by: str  # "us" | "counterparty" | "unknown"
    change_summary: str
    citation: ObservationCitation
    risk_delta: dict[str, str] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Full internal shape — round_moves.jsonl / cache serialization."""
        return {
            "document_id": self.document_id,
            "round": self.round,
            "taxonomy_id": self.taxonomy_id,
            "moved_by": self.moved_by,
            "change_summary": self.change_summary,
            "citation": self.citation.to_dict(),
            "risk_delta": self.risk_delta,
        }

    def to_opf_dict(self) -> dict[str, Any]:
        """OPF §3.5.3 ``negotiation_trail`` entry shape."""
        ref: dict[str, Any] = {
            "document_id": self.citation.document_id,
            "version": self.citation.version,
            "clause_path": self.citation.clause_path,
        }
        if self.citation.char_span is not None:
            ref["char_span"] = list(self.citation.char_span)
        d: dict[str, Any] = {
            "document_id": self.document_id,
            "round": self.round,
            "moved_by": self.moved_by,
            "change_summary": self.change_summary,
            "ref": ref,
        }
        if self.risk_delta is not None:
            d["risk_delta"] = self.risk_delta
        return d


def _summarize_move(diff: ClauseDiff) -> str:
    """One-line what-moved summary for a round-scoped ClauseDiff.

    Deliberately UNTRUNCATED: truncating raw clause text here can cut a
    counterparty name mid-word, after which the pseudonymization pass's
    whole-word matching no longer recognizes it and the fragment leaks into
    the born-safe store. Truncation happens in ``truncate_move_summaries``,
    which the pipeline applies AFTER pseudonymization.
    """
    if diff.kind == "added":
        return f"Clause added: {diff.text_after}"
    if diff.kind == "removed":
        return f"Clause removed (was: {diff.text_before})"
    hunk = diff.hunks[0] if diff.hunks else None
    if hunk is None:
        return f"Clause modified: {diff.text_after}"
    if hunk.kind == "insert":
        detail = f"added '{hunk.new_text}'"
    elif hunk.kind == "delete":
        detail = f"removed '{hunk.old_text}'"
    else:
        detail = f"'{hunk.old_text}' → '{hunk.new_text}'"
    more = len(diff.hunks) - 1
    suffix = f" (+{more} more change{'s' if more > 1 else ''})" if more > 0 else ""
    return f"Clause modified: {detail}{suffix}"


def truncate_move_summaries(
    moves: list[RoundMove], limit: int = _CHANGE_SUMMARY_MAX
) -> list[RoundMove]:
    """Cap each move's ``change_summary`` at *limit* chars for the store.

    Runs after the pipeline's pseudonymization pass (see ``_summarize_move``
    for why the order matters — a post-aliasing slice cannot expose a raw
    name, because the name is already gone).
    """
    return [
        dataclasses.replace(m, change_summary=m.change_summary[:limit])
        if len(m.change_summary) > limit
        else m
        for m in moves
    ]


def _shape_search_snippet(text: str, limit: int) -> str:
    """Pick one line of *text* and cap it at *limit* chars (issue #95).

    ``full_text``/``search_snippet`` routinely spans multiple lines —
    ``segmenter.py``'s ``"\\n".join(lines_list).strip()`` is the norm for a
    clause with sub-paragraphs (indemnification, limitation-of-liability),
    not the exception. Collapsing the WHOLE block onto one line via
    ``" ".join(text.split())`` (the prior behaviour) rebuilds the string
    from its words and stops being a literal substring of the source the
    moment there is more than one line: Word/PDF Ctrl+F does not match
    across a paragraph mark, so a collapsed multi-line snippet is
    unfindable in the source document — defeating the whole point of this
    field.

    Instead this takes the FIRST NON-EMPTY LINE verbatim: only
    ``str.strip()`` is applied to it, which trims solely from the ends, so
    the result is always a contiguous slice of *text*, never a rebuilt
    string. Only if that line still exceeds *limit* is it trimmed back to
    the last word boundary within the cap, so the excerpt reads as "a
    phrase," not a word fragment — that trim is itself a plain slice, so
    the result is a genuine substring of *text* all the way through (Ctrl+F
    really does find it). Falls back to a hard cut only when the first
    token alone exceeds *limit* (no space to trim back to).
    """
    line = ""
    for candidate in text.splitlines():
        stripped = candidate.strip()
        if stripped:
            line = stripped
            break
    if len(line) <= limit:
        return line
    truncated = line[:limit]
    last_space = truncated.rfind(" ")
    if last_space > 0:
        truncated = truncated[:last_space]
    return truncated.rstrip()


def truncate_search_snippets(
    observations: list[Observation], limit: int = _SEARCH_SNIPPET_MAX
) -> list[Observation]:
    """Cap each observation's ``search_snippet`` at *limit* chars for the store.

    Mirrors ``truncate_move_summaries`` exactly, including WHY this is a
    separate, unconditional step rather than folded into
    ``_pseudonymize_observations``: pseudonymization only runs when
    ``config.provenance.known_entities`` is configured, but every corpus —
    aliased or not — still needs its snippet capped down to a short phrase,
    so the pipeline calls this unconditionally, after (never as part of) the
    conditional pseudonymization pass. Per ``Observation.search_snippet``'s
    docstring and ``_summarize_move``'s (the analogous ``change_summary``
    field): a post-aliasing slice cannot expose a raw counterparty name,
    because by the time this runs the name is already gone from
    ``search_snippet`` — truncating before aliasing is what bisects a name
    and defeats the whole-word match.
    """
    return [
        dataclasses.replace(o, search_snippet=_shape_search_snippet(o.search_snippet, limit))
        for o in observations
    ]


def build_round_moves(
    document_id: str,
    doc_diff: DocumentDiff,
    tracked_by_vid: dict[str, TrackedChanges | None] | None = None,
    our_party_aliases: list[str] | None = None,
    our_authors: list[str] | None = None,
) -> list[RoundMove]:
    """Surface ``doc_diff.consecutive`` as ``RoundMove`` records (issue #177).

    One record per changed clause per negotiation round. ``moved_by`` is
    attributed from the destination version's own tracked-changes
    side-channel when a per-hunk match exists (each author's edits are
    tracked against the file they received, so the post-move version
    carries the mover's w:ins/w:del) via ``enrich_clause_diff``; when that
    finds no match, ``tracked_changes_overlay.round_level_fallback_
    attribution`` (issue #118 fix round 2, finding 2) gets one last try —
    it fires only when the side-channel carries exactly one distinct
    author, attributing every real content change in the round to them
    without per-hunk matching, and refuses outright when two or more
    distinct authors are present. Either way the resolved author is mapped
    through *our_party_aliases* and *our_authors* (issue #119) via
    ``party_side_for_author``; ``"unknown"`` when neither tier matches — never
    guessed.

    Unlike ``pipeline._attribution_for_diff`` (which enriches the NET diff
    and must gate its round-level fallback to single-round documents so it
    never attributes an earlier round's change to a later round's sole
    author — see that function's docstring), this loop iterates
    ``doc_diff.consecutive`` directly: ``side_channel`` here is always
    genuinely THIS round's own destination-version side channel, so the
    fallback is safe to apply unconditionally per round.
    """
    aliases = our_party_aliases or []
    authors = our_authors or []
    tracked = tracked_by_vid or {}
    moves: list[RoundMove] = []
    # doc_diff.version_order is ordered oldest-first; consecutive[i] diffs
    # version_order[i] → version_order[i+1], i.e. negotiation round i+1.
    ordinal_by_vid = {vid: i + 1 for i, vid in enumerate(doc_diff.version_order)}

    for round_idx, version_diff in enumerate(doc_diff.consecutive, start=1):
        for diff in version_diff.changed():
            if diff.clause_path_after is not None:
                cite_version_id = diff.clause_version_after or version_diff.version_after
                cite_path = diff.clause_path_after
                cite_span = diff.char_span_after
            else:
                # Removed clause: cite the last state where it existed.
                cite_version_id = diff.clause_version_before or version_diff.version_before
                cite_path = diff.clause_path_before or "?"
                cite_span = diff.char_span_before

            moved_by = "unknown"
            if diff.hunks:
                side_channel = tracked.get(version_diff.version_after)
                if side_channel is not None:
                    enriched = enrich_clause_diff(diff, side_channel)
                    enrichment = next(
                        (eh.enrichment for eh in enriched if eh.enrichment is not None), None
                    )
                    if enrichment is None:
                        enrichment = round_level_fallback_attribution(diff.hunks[0], side_channel)
                    if enrichment is not None:
                        moved_by = party_side_for_author(enrichment.author, aliases, authors)

            moves.append(
                RoundMove(
                    document_id=document_id,
                    round=round_idx,
                    taxonomy_id=diff.taxonomy_id,
                    moved_by=moved_by,
                    change_summary=_summarize_move(diff),
                    citation=ObservationCitation(
                        document_id=document_id,
                        version=ordinal_by_vid.get(cite_version_id, round_idx + 1),
                        clause_path=cite_path,
                        char_span=cite_span,
                        version_id=cite_version_id,
                    ),
                )
            )
    return moves


def write_round_moves_jsonl(moves: list[RoundMove], path: Path) -> None:
    """Write *moves* to *path* as JSONL, atomically (mirrors observations)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for move in moves:
            f.write(json.dumps(move.to_dict(), ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def round_move_from_dict(raw: dict[str, Any]) -> RoundMove:
    """Reconstruct a ``RoundMove`` from its ``to_dict()`` form."""
    cit = raw["citation"]
    cs_raw = cit.get("char_span")
    return RoundMove(
        document_id=raw["document_id"],
        round=raw["round"],
        taxonomy_id=raw.get("taxonomy_id"),
        moved_by=raw["moved_by"],
        change_summary=raw["change_summary"],
        citation=ObservationCitation(
            document_id=cit["document_id"],
            version=cit["version"],
            clause_path=cit["clause_path"],
            char_span=tuple(cs_raw) if cs_raw else None,
            version_id=cit.get("version_id"),
        ),
        risk_delta=raw.get("risk_delta"),
    )


def read_round_moves_jsonl(path: Path) -> list[RoundMove]:
    """Read ``RoundMove`` records back from *path*; ``[]`` when absent."""
    if not path.exists():
        return []
    moves: list[RoundMove] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            moves.append(round_move_from_dict(json.loads(line)))
    return moves


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

#: ``dropped`` counter key (issue #216): a net-diff row whose clause has no
#: terminal slot (``clause_path_after is None``) but whose own text still
#: survives in the terminal — its normalized text occurs verbatim in one
#: clause-sized, contiguous stretch of the terminal version (fill-in blanks
#: aside — see ``_survives_in_terminal``), e.g. a relocated clause the
#: aligner left unpaired (``basis="alignment"``). Survival is decided on the
#: text itself, never on its words recurring: text narrowed, restored or
#: replaced before signing does not survive. A surviving row was not
#: removed, and its text is already represented by the terminal's own
#: signed observation, so it produces no observation of its own; it is
#: counted here instead of being dropped silently (surfaced in
#: ``corpus.stats.dropped_observations``).
DROPPED_SURVIVES_IN_TERMINAL = "survives_in_terminal"

#: ``dropped`` counter key (issue #216): a net-diff row whose clause has no
#: terminal slot and whose text is absent from the terminal (removed before
#: signing), but whose ORIGIN cannot be determined — there is no standard
#: text for its clause (unclassified, or no template clause for its
#: taxonomy_id) to tell our standard language from a counterparty ask. It is
#: neither a refused ask (``proposed_then_reversed``) nor our concession
#: (``conceded_before_signing``), so it produces no observation and is
#: counted here (surfaced in ``corpus.stats.dropped_observations``).
DROPPED_ORIGIN_UNDETERMINED = "removed_origin_undetermined"

#: ``dropped`` counter key (issue #216, issue #83): a removed row whose text
#: is OUR standard language, in a deal with no detected executed copy
#: (``has_signed_copy=False``). In a signed deal it would be our concession
#: (``conceded_before_signing``), but a deal never shown to be executed is
#: not evidence of what we accept or concede — exactly as its terminal rows
#: are ``"unsigned"`` rather than ``"signed"``. It produces no observation,
#: so it never reaches ``stance_detail`` or the position, and is counted
#: here (surfaced in ``corpus.stats.dropped_observations``).
DROPPED_STANDARD_REMOVED_UNSIGNED = "removed_standard_no_signed_copy"

#: Outcome of a removed-before-signing row whose text is OUR standard
#: language (issue #216, owner decision 2026-09-13 (b): a provision's origin
#: decides, never the deal's paper side). Striking our own standard before
#: signing is a concession we made, not an ask we refused, so it is never
#: ``proposed_then_reversed``. Engine-internal like ``"unsigned"``: the OPF
#: ``observation.outcome`` enum does not carry it, so the position compiler
#: keeps it out of ``observed_positions`` / ``rollup.rejected`` (and hence
#: the digest's ``unacceptable`` list and Floor candidates) and counts it
#: only as a conceded deal in ``stance_detail`` and the position. Emitted
#: only for a deal with a detected executed copy; otherwise the row is
#: dropped under ``DROPPED_STANDARD_REMOVED_UNSIGNED`` (issue #83).
OUTCOME_CONCEDED_BEFORE_SIGNING = "conceded_before_signing"

# Severity ranks used to pick the representative net-diff row when several
# terminal nodes share one taxonomy_id (issue #216). The representative's
# deviation + risk_delta pair is carried by the merged observation as one
# consistent assessment; the worst risk dominates so a concession
# (direction="worse") is never masked by an unchanged sibling node.
_DEVIATION_RANK: dict[str, int] = {
    "none": 0,
    "reworded_equivalent": 1,
    "needs_review": 2,
    "substantive": 3,
}
_MAGNITUDE_RANK: dict[str, int] = {"none": 0, "minor": 1, "material": 2}
_DIRECTION_RANK: dict[str, int] = {"neutral": 0, "better": 1, "worse": 2}

# Bases meaning a row's assessment is weaker than a judge verdict, weakest
# last. Mirrors clause_position_compiler._UNJUDGED_BASES/_STUB_BASES: if ANY
# node of a merged taxonomy group carries one, the merged observation carries
# the weakest of them, so merging never launders an unjudged node into a
# judged one (the stub cap and the acceptable_if basis filter keep holding).
_WEAK_BASIS_RANK: dict[str, int] = {"needs_review": 1, "judge_error": 2, "stub": 3}


def _content_tokens(text: str) -> frozenset[str]:
    """``detect_reversals``' token set for *text*, minus fill-in blanks.

    ``\\w`` matches ``_``, so a signature-block placeholder (``By: ______``)
    tokenizes as a "word" that the executed copy — where the blank was
    filled in — never contains. A blank carries no clause content, so it
    must not make a removed clause look absent from the signed terminal.
    """
    return frozenset(t for t in _reversal_tokens(text) if t.strip("_"))


#: A removed row survives only inside a contiguous run of terminal nodes whose
#: combined content-token count is at most this multiple of the row's own —
#: a clause-sized stretch of the terminal, never its whole vocabulary.
_SURVIVAL_WINDOW_FACTOR = 3

#: A fill-in blank (``By: ______``): a run of two or more underscores.
_FILL_IN_BLANK_RE = re.compile(r"_{2,}")


@dataclass(frozen=True)
class _SurvivalNode:
    """One terminal node as ``_survives_in_terminal`` reads it: its text,
    normalized like the origin test, its content-token count (the window
    size unit), and whether its net-diff row is ``unchanged``."""

    normalized: str
    size: int
    unchanged: bool


def _survival_node(text: str, unchanged: bool) -> _SurvivalNode:
    return _SurvivalNode(_normalize_for_origin(text), len(_content_tokens(text)), unchanged)


def _survival_pattern(text: str) -> re.Pattern[str] | None:
    """Matcher for a removed row's OWN text inside terminal text (issue #216).

    The text is normalized like the origin test (case-, punctuation- and
    whitespace-insensitive) and must occur verbatim, contiguous and in
    order — words that merely recur, in another order or with other words
    inserted between them, do not match, so text narrowed or replaced
    before signing never passes as surviving. The one allowance is a
    fill-in blank: it stands for whatever run of words (possibly none) the
    executed copy filled it with. ``None`` when the text has no content
    beyond blanks.
    """
    segments = [_normalize_for_origin(seg) for seg in _FILL_IN_BLANK_RE.split(text)]
    segments = [seg for seg in segments if seg]
    if not segments:
        return None
    body = r"(?: .*?)? ".join(re.escape(seg) for seg in segments)
    return re.compile(r"(?<!\S)" + body + r"(?!\S)")


def _survives_in_terminal(text: str, nodes: Sequence[_SurvivalNode]) -> bool:
    """Whether a removed row's own *text* survives, as text, in ONE
    clause-sized, contiguous stretch of the terminal (issue #216).

    Survival is decided on the text itself (``_survival_pattern``: its
    normalized text occurs verbatim in the stretch, fill-in blanks aside),
    never on word-set membership — a clause narrowed, restored or replaced
    before signing keeps most of its words but not its text, and falls
    through to the origin test instead. *nodes* are the terminal nodes in
    document order (see ``_survival_node``). A window of adjacent nodes
    qualifies when their combined content-token count is at most
    ``_SURVIVAL_WINDOW_FACTOR`` times the row's own, so a clause split
    across several signed nodes (e.g. an executed signature block) still
    survives, while a clause whose text recurs by coincidence inside a much
    larger signed clause does not. An ``unchanged`` node's text is wholly
    its own first-version clause, so it can be a single-node match (a
    duplicate of the removed text) but never joins a multi-node window —
    otherwise an adjacent unchanged clause could supply part of a replaced
    clause's text.
    """
    tokens = _content_tokens(text)
    pattern = _survival_pattern(text)
    if not tokens or pattern is None:
        return True
    cap = _SURVIVAL_WINDOW_FACTOR * len(tokens)
    for i, start in enumerate(nodes):
        size = 0
        parts: list[str] = []
        for j in range(i, len(nodes)):
            node = nodes[j]
            if j > i and (node.unchanged or start.unchanged):
                break
            size += node.size
            if size > cap:
                break
            parts.append(node.normalized)
            if pattern.search(" ".join(parts)):
                return True
    return False


def _standard_nodes(standard: str | Sequence[str]) -> list[str]:
    """Our standard for one taxonomy_id as its non-empty template nodes, in
    document order (a bare string is a single-node standard)."""
    nodes = [standard] if isinstance(standard, str) else list(standard)
    return [node for node in nodes if node.strip()]


def _is_standard_language(text: str, standard: str | Sequence[str]) -> bool:
    """Whether *text* is OUR standard language for its clause (issue #216).

    The origin test for a clause removed before signing. *standard* is our
    standard text for the clause's taxonomy_id — EVERY template node
    carrying that taxonomy_id, in document order (a bare string is a
    single-node standard). *text* matches when it is a near-identical
    rendering (token Jaccard at or above ``REWORDED_EQUIVALENT_THRESHOLD`` —
    the same bar ``assess_deviations`` uses to call a clause "matches our
    standard") of the whole standard or of any one of its nodes, or when its
    normalized text occurs verbatim inside the normalized standard (one
    fragment of a standard the deal or the template split across several
    nodes — including any later template node, never only the first).
    """
    nodes = _standard_nodes(standard)
    if not text.strip() or not nodes:
        return False
    whole = "\n".join(nodes)
    candidates = [whole] + (nodes if len(nodes) > 1 else [])
    if any(_text_jaccard(text, cand) >= REWORDED_EQUIVALENT_THRESHOLD for cand in candidates):
        return True
    normalized = _normalize_for_origin(text)
    return bool(normalized) and normalized in _normalize_for_origin(whole)


def _row_severity(dr: DeviationResult) -> tuple[int, int, int]:
    risk = dr.risk_delta
    return (
        _DIRECTION_RANK.get(risk.direction, 0),
        _MAGNITUDE_RANK.get(risk.magnitude, 0),
        _DEVIATION_RANK.get(dr.deviation, 0),
    )


@dataclass
class _TerminalNode:
    """One clause node of the terminal (signed/last) version, plus the
    net-diff row (index, ClauseDiff, DeviationResult) whose after slot it is."""

    taxonomy_id: str | None
    clause_path: str
    text: str
    char_span: tuple[int, int] | None
    version_id: str | None
    row: tuple[int, ClauseDiff, DeviationResult]


def build_observations(
    document_id: str,
    version: int | str,
    provenance: str,
    deviation_results: list[tuple[Any, DeviationResult]],  # (ClauseDiff, DeviationResult)
    reversals: list[ReversalRecord],
    classification_confidences: list[float | None] | None = None,
    has_signed_copy: bool = True,
    attributions: list[HunkEnrichment | None] | None = None,
    our_party_aliases: list[str] | None = None,
    our_authors: list[str] | None = None,
    ordinal_by_vid: dict[str, int] | None = None,
    terminal_clauses: Sequence[ClassifiedClause] | None = None,
    terminal_version_id: str | None = None,
    dropped: dict[str, int] | None = None,
    standard_text_by_tid: Mapping[str, str | Sequence[str]] | None = None,
) -> list[Observation]:
    """Assemble ``Observation`` objects for one document (one deal).

    **The deal is the unit of precedent (issue #216).** The terminal
    (signed, or last when unsigned) version contributes EXACTLY ONE
    observation per taxonomy_id: its ``full_text`` is that version's nodes
    carrying the taxonomy_id, concatenated in document order with a single
    newline, cited to the first such node. A clause that spans several nodes
    in one deal is one precedent, not several. Unclassified nodes
    (``taxonomy_id=None``) are not a clause type and stay one observation per
    node — they never reach a ClausePosition, only unclassified coverage.

    Text that never reached the terminal is never ``outcome="signed"``: a
    net-diff row whose clause has no terminal slot
    (``clause_path_after is None``) is tested on its OWN text against one
    clause-sized, contiguous stretch of the terminal version: it survives
    only when its normalized text occurs there verbatim (fill-in blanks such
    as ``By: ______`` aside), never merely because its words recur — see
    ``_survives_in_terminal``. Text that survives in the terminal (a
    relocation the aligner left unpaired) is not removed at all: it produces
    no observation and is counted in *dropped* under
    ``DROPPED_SURVIVES_IN_TERMINAL``. Text absent from the terminal was
    removed before signing, and what that means is decided by the ORIGIN of
    the text, never by the deal's paper side (owner decision 2026-09-13
    (b)), against *standard_text_by_tid* (see ``_is_standard_language``):

    - OUR standard language → ``OUTCOME_CONCEDED_BEFORE_SIGNING``: our
      concession, never a refused ask — but only when *has_signed_copy*.
      In a deal with no detected executed copy (issue #83) it produces no
      observation and is counted in *dropped* under
      ``DROPPED_STANDARD_REMOVED_UNSIGNED``: a deal never shown to be
      executed is never evidence of a concession;
    - non-standard (counterparty-originated) language →
      ``"proposed_then_reversed"``: their refused ask. Like a
      ``ReversalRecord`` this is within-trail negotiation history, so it
      holds whether or not the deal has a detected executed copy;
    - no standard to compare against (unclassified, or no standard text for
      the taxonomy_id) → no observation, counted in *dropped* under
      ``DROPPED_ORIGIN_UNDETERMINED``.

    Both emitted kinds cite the first version the text was read from.
    Removed rows never claim a
    ``ReversalRecord``: a removed row's path is a FIRST-version path, a
    reversal's is a later draft's, so a path match between them is a
    coincidence of numbering, not the same text. Every ``ReversalRecord`` is
    emitted as its own ``"proposed_then_reversed"`` observation carrying its
    PROPOSED text and the draft citation — including a reversal inside a
    clause that survived to the terminal, whose signed text stays the
    (single) signed observation.

    Args:
        document_id:               Source document identifier.
        version:                   Source version identifier.
        provenance:                ``"our_paper"`` or ``"counterparty_paper"``.
        deviation_results:         Output of ``assess_deviations()`` — list of
                                  ``(ClauseDiff, DeviationResult)`` pairs.  Only
                                  changed clauses carry meaningful deviation data;
                                  unchanged clauses are also included (outcome
                                  defaults to signed/unsigned per
                                  ``has_signed_copy``, deviation=none).
        reversals:                 Output of ``detect_reversals()`` for this document.
        classification_confidences: Per-diff classification confidence values in the
                                  same order as ``deviation_results``.  Each entry
                                  is a float in [0, 1] or ``None`` when unavailable.
                                  When omitted, all observations have
                                  ``confidence=None``. A merged terminal
                                  observation (issue #216) carries the lowest
                                  confidence among its rows.
        has_signed_copy:            Whether the caller's version-ordering step
                                  (``order_versions``) actually identified a
                                  version as the executed copy of this
                                  document. Defaults to True for backward
                                  compatibility with existing callers/tests
                                  that don't model signed-copy detection.
                                  When False, every terminal observation's
                                  ``outcome`` is ``"unsigned"`` instead of
                                  ``"signed"`` — reporting a clause from a
                                  document with no detected signed copy as an
                                  accepted, signed position is exactly the
                                  fabrication issue #83 closes — and a
                                  removed row whose text is our standard
                                  language is never
                                  ``"conceded_before_signing"``: it produces
                                  no observation and is counted under
                                  ``DROPPED_STANDARD_REMOVED_UNSIGNED``, so
                                  such a deal never contributes accepted or
                                  conceded evidence. Reversals, and removed
                                  non-standard (their) language, keep
                                  ``"proposed_then_reversed"`` regardless —
                                  that label describes within-trail
                                  negotiation history (something was
                                  proposed, then struck by a later draft),
                                  which holds independent of whether the
                                  final draft was ever executed.
        attributions:               Per-diff tracked-changes attribution (issue #88),
                                  in the same order as ``deviation_results`` — see
                                  ``playbook_engine.tracked_changes_overlay``. Each
                                  entry is a ``HunkEnrichment`` or ``None`` when no
                                  DOCX tracked-changes side-channel matched that
                                  clause. When omitted, every observation's
                                  ``attribution`` is ``None``.
        our_party_aliases:          ``config.provenance.our_party_aliases`` (issue
                                  #177) — enables the dynamics fields: a changed
                                  clause's ``proposed_by`` derives from its
                                  attribution author mapped through these aliases
                                  ("unknown" when unattributed), and
                                  ``observed_at`` from the tracked-change date.
                                  Unchanged clauses (deviation "none") carry no
                                  ``proposed_by`` — nothing was proposed. When
                                  ``None`` (legacy callers/tests), no dynamics
                                  fields are derived at all.
        our_authors:                ``config.provenance.our_authors`` (issue #119)
                                  — the people-namespace counterpart to
                                  ``our_party_aliases`` (personal names/initials/
                                  emails, as opposed to entity/org names), checked
                                  alongside it by ``party_side_for_author``. An
                                  author matching NEITHER list is "unknown", never
                                  "counterparty" — absence of a match is not
                                  evidence, in either direction.
        ordinal_by_vid:             Version-id → negotiation-ordinal map (the
                                  1-based position of each version id in
                                  ``version_order``, exactly as
                                  ``build_round_moves`` derives it). When
                                  supplied, each citation's ``version`` is the
                                  ordinal of the version its clause text was
                                  actually read from (its ``version_id``) —
                                  a removed clause or a reversal cites the
                                  DRAFT's ordinal, never the signed ordinal
                                  this observation batch is filed under, so
                                  ``citation.version`` resolves (via
                                  ``version_files`` / citation_resolver) to
                                  the file that really contains the cited
                                  clause_path/char_span. Signed-clause
                                  observations are unaffected: their
                                  version_id IS the signed version, whose
                                  ordinal is the caller's *version*. When
                                  ``None`` (legacy callers/tests), every
                                  citation keeps the caller's *version*
                                  unchanged.
        terminal_clauses:           The terminal version's classified clause tree
                                  in document order. When supplied it is the
                                  authority for which taxonomy_ids the terminal
                                  carries, their text, and the first node each
                                  observation cites; each net-diff row that
                                  reached the terminal attaches to exactly one
                                  node by (``clause_path_after``,
                                  ``char_span_after``), and a node with no row
                                  (or a row with no node) raises ``ValueError``
                                  — the tree must be the net diff's after
                                  side. When ``None`` (legacy callers/tests),
                                  each row that reached the terminal stands for
                                  its own node, in row order.
        terminal_version_id:        Normalized-tree id of the terminal version,
                                  cited as ``version_id`` for tree-derived
                                  nodes (issue #108).
        dropped:                    Optional counter, incremented per dropped
                                  row by reason (see
                                  ``DROPPED_SURVIVES_IN_TERMINAL``,
                                  ``DROPPED_ORIGIN_UNDETERMINED`` and
                                  ``DROPPED_STANDARD_REMOVED_UNSIGNED``).
        standard_text_by_tid:       Our standard (template) clause text per
                                  taxonomy_id — the origin reference for a
                                  clause removed before signing (issue
                                  #216): EVERY template node carrying the
                                  taxonomy_id, in document order (a bare
                                  string is a single-node standard; see
                                  ``_is_standard_language``). ``None`` or a
                                  missing/empty entry means the origin of
                                  that clause's removed text cannot be
                                  determined.

    Returns:
        One ``Observation`` per terminal taxonomy_id (plus one per
        unclassified terminal node), one per removed row whose text is
        absent from the terminal and whose origin is determined (except our
        standard language in a deal with no detected executed copy, which
        is dropped), and one per distinct ``ReversalRecord``.
    """
    default_outcome = "signed" if has_signed_copy else "unsigned"

    observations: list[Observation] = []
    obs_counter: dict[str, int] = {}

    def _next_id(clause_path: str) -> str:
        base_id = f"{document_id}/{version}/{clause_path}"
        obs_counter[base_id] = obs_counter.get(base_id, 0) + 1
        count = obs_counter[base_id]
        return base_id if count == 1 else f"{base_id}#{count}"

    def _cite_version(version_id: str | None) -> int | str:
        # citation.version is the ordinal of the version the cited text was
        # actually read from, never blanket the terminal ordinal (issue #108).
        if ordinal_by_vid is not None and version_id is not None:
            return ordinal_by_vid.get(version_id, version)
        return version

    def _confidence(idx: int) -> float | None:
        if classification_confidences is not None and idx < len(classification_confidences):
            return classification_confidences[idx]
        return None

    def _attribution(idx: int) -> HunkEnrichment | None:
        if attributions is not None and idx < len(attributions):
            return attributions[idx]
        return None

    def _dynamics(
        deviation: str, attribution: HunkEnrichment | None
    ) -> tuple[str | None, str | None]:
        # Negotiation dynamics (issue #177). Only derived when the caller
        # opted in via our_party_aliases; only for clauses where something
        # actually moved (a deviation="none" row records absence of change —
        # there is no proposal to attribute or date).
        if our_party_aliases is None or deviation == "none":
            return None, None
        if attribution is None:
            return "unknown", None
        return (
            party_side_for_author(attribution.author, our_party_aliases, our_authors),
            _date_from_tracked(attribution.date),
        )

    # --- 1. Partition rows: terminal (reached the terminal) vs removed ---
    terminal_rows: list[tuple[int, ClauseDiff, DeviationResult]] = []
    removed_rows: list[tuple[int, ClauseDiff, DeviationResult]] = []
    for idx, (clause_diff, dr) in enumerate(deviation_results):
        if clause_diff.clause_path_after is not None:
            terminal_rows.append((idx, clause_diff, dr))
        else:
            removed_rows.append((idx, clause_diff, dr))

    # --- 2. Terminal nodes, in document order, with their rows attached ---
    nodes: list[_TerminalNode] = []
    if terminal_clauses is not None:
        # Rows attach to tree nodes by (clause_path, char_span) — the net
        # diff's after slot IS the node (clause_differ._version_diff copies
        # both from it), and the span disambiguates two nodes that share a
        # path. Each node consumes exactly one row: the net diff emits one row
        # per after-slot, and every terminal node sits in exactly one
        # alignment, so a node no row reaches, or a row no node claims, means
        # the caller passed a tree that is not the net diff's after side.
        pending: dict[
            tuple[str, tuple[int, int] | None], list[tuple[int, ClauseDiff, DeviationResult]]
        ] = {}
        for row in terminal_rows:
            cd = row[1]
            pending.setdefault((cd.clause_path_after or "?", cd.char_span_after), []).append(row)
        for cc in terminal_clauses:
            path = cc.node.clause_path or "?"
            queue = pending.get((path, cc.node.char_span))
            if not queue:
                raise ValueError(
                    f"{document_id}: terminal node {path!r} has no net-diff row — "
                    "terminal_clauses must be the net diff's after-side tree"
                )
            nodes.append(
                _TerminalNode(
                    taxonomy_id=cc.classification.taxonomy_id,
                    clause_path=path,
                    text=cc.node.text or "",
                    char_span=cc.node.char_span,
                    version_id=terminal_version_id,
                    row=queue.pop(0),
                )
            )
        unclaimed = [row[1].clause_path_after for rows in pending.values() for row in rows]
        if unclaimed:
            raise ValueError(
                f"{document_id}: net-diff rows {unclaimed!r} match no terminal node — "
                "terminal_clauses must be the net diff's after-side tree"
            )
    else:
        # Legacy callers/tests pass no tree: each row that reached the
        # terminal stands for its own after-side node (exactly what the net
        # diff emits — one row per after-slot), in row order.
        for row in terminal_rows:
            cd = row[1]
            nodes.append(
                _TerminalNode(
                    taxonomy_id=cd.taxonomy_id,
                    clause_path=cd.clause_path_after or "?",
                    text=cd.text_after,
                    char_span=cd.char_span_after,
                    version_id=cd.clause_version_after,
                    row=row,
                )
            )

    # --- 2b. Removed rows: surviving, conceded, refused, or undetermined ---
    # A removed row's own text, tested as text against ONE clause-sized
    # stretch of the terminal (see _survives_in_terminal) — never against
    # the whole signed document, and never by word-set membership, where a
    # narrowed or replaced clause's words can all recur in the signed copy.
    survival_nodes = [_survival_node(n.text, n.row[1].kind == "unchanged") for n in nodes]
    for idx, clause_diff, dr in removed_rows:
        # No terminal slot: the cited text is read from the FIRST version, so
        # it is never the default (signed/unsigned) outcome. What its removal
        # means is decided on its own text — never by matching a
        # ReversalRecord, whose path belongs to a later draft, and never by
        # the deal's paper side.
        raw_text = clause_diff.text_before
        if _survives_in_terminal(raw_text, survival_nodes):
            # The text survives in the terminal (e.g. basis="alignment") —
            # not reversed, and not draft-only.
            if dropped is not None:
                dropped[DROPPED_SURVIVES_IN_TERMINAL] = (
                    dropped.get(DROPPED_SURVIVES_IN_TERMINAL, 0) + 1
                )
            continue
        tid = clause_diff.taxonomy_id
        standard = (standard_text_by_tid or {}).get(tid or "", "") if tid is not None else ""
        if not _standard_nodes(standard):
            # No standard to tell our language from theirs: neither a
            # refused ask nor our concession — counted, never guessed.
            if dropped is not None:
                dropped[DROPPED_ORIGIN_UNDETERMINED] = (
                    dropped.get(DROPPED_ORIGIN_UNDETERMINED, 0) + 1
                )
            continue
        if _is_standard_language(raw_text, standard):
            if not has_signed_copy:
                # Our standard struck in a deal with no detected executed
                # copy (issue #83): never a concession at L5 — counted,
                # exactly as its terminal rows are "unsigned", not "signed".
                if dropped is not None:
                    dropped[DROPPED_STANDARD_REMOVED_UNSIGNED] = (
                        dropped.get(DROPPED_STANDARD_REMOVED_UNSIGNED, 0) + 1
                    )
                continue
            removed_outcome = OUTCOME_CONCEDED_BEFORE_SIGNING
        else:
            # Their (non-standard) language struck is their refused ask —
            # within-trail negotiation history, so, like a ReversalRecord,
            # it holds whether or not the deal was executed.
            removed_outcome = "proposed_then_reversed"
        clause_path = clause_diff.clause_path_before or "?"
        attribution = _attribution(idx)
        proposed_by, observed_at = _dynamics(dr.deviation, attribution)
        observations.append(
            Observation(
                observation_id=_next_id(clause_path),
                taxonomy_id=tid,
                text_summary=summarize_clause_text(raw_text),
                full_text=raw_text,
                citation=ObservationCitation(
                    document_id=document_id,
                    version=_cite_version(clause_diff.clause_version_before),
                    clause_path=clause_path,
                    char_span=clause_diff.char_span_before,
                    version_id=clause_diff.clause_version_before,
                ),
                deviation=dr.deviation,
                risk_delta=dr.risk_delta.to_dict(),
                provenance=provenance,
                outcome=removed_outcome,
                confidence=_confidence(idx),
                basis=dr.basis,
                attribution=attribution,
                proposed_by=proposed_by,
                observed_at=observed_at,
            )
        )

    # --- 3. One observation per terminal taxonomy_id (per node if None) ---
    groups: list[list[_TerminalNode]] = []
    group_by_tid: dict[str, list[_TerminalNode]] = {}
    for node in nodes:
        if node.taxonomy_id is None:
            groups.append([node])
            continue
        if node.taxonomy_id not in group_by_tid:
            group_by_tid[node.taxonomy_id] = []
            groups.append(group_by_tid[node.taxonomy_id])
        group_by_tid[node.taxonomy_id].append(node)

    for group in groups:
        first = group[0]
        full_text = "\n".join(n.text for n in group if n.text)
        rows = [n.row for n in group]
        rep_idx, _rep_cd, rep_dr = max(rows, key=lambda r: _row_severity(r[2]))
        deviation = rep_dr.deviation
        risk_delta = rep_dr.risk_delta.to_dict()
        basis = rep_dr.basis
        weakest = max(rows, key=lambda r: _WEAK_BASIS_RANK.get(r[2].basis or "", 0))[2].basis
        if _WEAK_BASIS_RANK.get(weakest or "", 0) > _WEAK_BASIS_RANK.get(basis or "", 0):
            basis = weakest
        confidences = [c for c in (_confidence(r[0]) for r in rows) if c is not None]
        conf: float | None = min(confidences) if confidences else None
        attribution = _attribution(rep_idx)
        if attribution is None:
            attribution = next(
                (a for a in (_attribution(r[0]) for r in rows) if a is not None), None
            )

        proposed_by, observed_at = _dynamics(deviation, attribution)
        observations.append(
            Observation(
                observation_id=_next_id(first.clause_path),
                taxonomy_id=first.taxonomy_id,
                text_summary=summarize_clause_text(full_text),
                full_text=full_text,
                citation=ObservationCitation(
                    document_id=document_id,
                    version=_cite_version(first.version_id),
                    clause_path=first.clause_path,
                    char_span=first.char_span,
                    version_id=first.version_id,
                ),
                deviation=deviation,
                risk_delta=risk_delta,
                provenance=provenance,
                outcome=default_outcome,
                confidence=conf,
                basis=basis,
                attribution=attribution,
                proposed_by=proposed_by,
                observed_at=observed_at,
            )
        )

    # --- 4. Reversals (issue #106) ---
    # A clause inserted mid-negotiation and removed again before the signed
    # terminal is the cleanest "we rejected this ask" signal available. Emit
    # an Observation directly from each ReversalRecord — it carries
    # taxonomy_id, proposed_text, and the draft citation. Removed net-diff
    # rows above never claim one (their paths are first-version paths), so
    # every reversal keeps its own proposed text. This also covers a
    # reversal inside a clause that survived to the terminal: the terminal's
    # signed text is the signed observation above, and the proposal that was
    # reversed out of it is this one.
    # Only a true duplicate record (same clause, draft AND proposed text) is
    # skipped: two proposals from different drafts that share a path number
    # are distinct evidence, each emitted with its own proposed text.
    emitted_reversals: set[tuple[str | None, str, str, str]] = set()
    for r in reversals:
        key = (r.taxonomy_id, r.clause_path, r.version_inserted, r.proposed_text)
        if key in emitted_reversals:
            continue
        emitted_reversals.add(key)

        observations.append(
            Observation(
                observation_id=_next_id(r.clause_path),
                taxonomy_id=r.taxonomy_id,
                text_summary=summarize_clause_text(r.proposed_text),
                full_text=r.proposed_text,
                citation=ObservationCitation(
                    document_id=document_id,
                    # The citation's version is the DRAFT's ordinal (the
                    # version the proposed text actually lives in), never the
                    # signed ordinal — the proposal was, by definition,
                    # reversed out of the signed version.
                    version=_cite_version(r.version_inserted),
                    clause_path=r.clause_path,
                    char_span=r.char_span,
                    # r.clause_path is the clause instance path in the DRAFT
                    # version the proposal first appeared in, not the signed
                    # terminal (see ReversalRecord.clause_path's docstring) —
                    # version_id must cite version_inserted, never the signed
                    # version this observation batch is filed under, or the
                    # citation resolves to the wrong file (issue #108).
                    version_id=r.version_inserted,
                ),
                # "substantive": the proposed text genuinely differed from the
                # signed terminal (that is exactly what detect_reversals
                # verified via its token-subset check) — never "none".
                deviation="substantive",
                # No DeviationJudge ever assessed the proposal (it never
                # entered deviation_results) — a neutral placeholder, not a
                # real risk judgment. clause_position_compiler's hold_firm
                # derivation keys off outcome/provenance for rejected
                # observations, not risk_delta, so this placeholder does not
                # distort position derivation.
                risk_delta={"direction": "neutral", "magnitude": "none"},
                provenance=provenance,
                outcome="proposed_then_reversed",
                # A reversal is by definition a proposed change, but the
                # ReversalRecord carries no attribution — "unknown", never
                # guessed (issue #177). Omitted entirely for legacy callers.
                proposed_by="unknown" if our_party_aliases is not None else None,
                confidence=None,
                # "deterministic": detected by detect_reversals' token-subset
                # comparison, not a judge call — but NOT one of the
                # _UNJUDGED_BASES/_STUB_BASES values, since this is a real,
                # fully-verified signal (unlike the stub judges' placeholder
                # basis values) and must not cap the clause's rollup position
                # to "negotiable".
                basis="deterministic",
            )
        )

    return observations


def write_observations_jsonl(observations: list[Observation], path: Path) -> None:
    """Write *observations* to *path* as JSONL, atomically.

    Each line is a JSON object.  The file is written via a temp file and
    ``os.replace()`` to prevent partial writes.

    Args:
        observations: List of ``Observation`` objects.
        path:         Destination path (parent directories created if needed).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".jsonl.tmp")
    lines = [json.dumps(obs.to_dict(), ensure_ascii=False) for obs in observations]
    tmp.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    os.replace(tmp, path)


def read_observations_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSONL file, returning raw dicts (for inspection / testing)."""
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]
