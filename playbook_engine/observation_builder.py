"""Observation builder — L4 → L5 bridge.

Assembles one inspectable row per clause observation, writing them to
``observations.jsonl`` (one JSON object per line).

Each observation captures:
  - What was observed: taxonomy_id, text_summary (≤ 200 chars of clause text,
    display-only — see full_text for the untruncated clause text used by
    downstream judges/standards, issue #105)
  - Citation: document_id, version, version_id, clause_path, char_span (see
    ObservationCitation for why version alone is not file-resolvable — issue #108)
  - Deviation assessment: deviation, risk_delta (from the deviation classifier)
  - Provenance: whose paper the document is on (OPF §2.2)
  - Outcome: "signed", "unsigned", or "proposed_then_reversed" (from reversal
    detector; "unsigned" when no version was detected as the executed copy —
    see build_observations' has_signed_copy)
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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from playbook_engine.clause_differ import ClauseDiff, DocumentDiff
from playbook_engine.deviation_classifier import DeviationResult
from playbook_engine.docx_ingester import TrackedChanges
from playbook_engine.reversal_detector import ReversalRecord
from playbook_engine.tracked_changes_overlay import HunkEnrichment, enrich_clause_diff

_TEXT_SUMMARY_MAX = 200


# Minimum author-string length for the author-in-alias containment direction.
# DOCX w:author values are frequently initials or short handles ("Al", "IT");
# a 1-3 char author is a substring of almost any alias, so matching it would
# systematically flip counterparty edits to "us".
_MIN_AUTHOR_CONTAINMENT_LEN = 4


def party_side_for_author(author: str | None, our_party_aliases: list[str]) -> str:
    """Map a tracked-changes author name to a negotiation side (issue #177).

    Case-insensitive containment against ``config.provenance.
    our_party_aliases`` ("FixtureCorp Legal" matches alias "FixtureCorp"). The reverse
    direction (author contained in an alias) only applies to authors of
    ``_MIN_AUTHOR_CONTAINMENT_LEN``+ chars — Word author strings are often
    initials, and "IT" ⊂ "Summit Health" must not read as "us". With NO
    aliases configured there is nothing to discriminate against, so every
    author maps to "unknown" — a side is never guessed (§3.5.3), and in
    particular an unconfigured corpus must not publish our own attorneys'
    edits as counterparty asks.
    """
    if not author or not any(a for a in our_party_aliases):
        return "unknown"
    author_lower = author.lower()
    for alias in our_party_aliases:
        alias_lower = alias.lower()
        if not alias_lower:
            continue
        if alias_lower in author_lower:
            return "us"
        if len(author_lower) >= _MIN_AUTHOR_CONTAINMENT_LEN and author_lower in alias_lower:
            return "us"
    return "counterparty"


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
        text_summary:    First ≤ 200 chars of the clause text. Display-only —
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
        outcome:         ``"signed"``, ``"unsigned"``, or ``"proposed_then_reversed"``.
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
                         clean DOCX, an LLM-segmented document (no
                         w:ins/w:del capture on that path yet), or a DOCX
                         redline whose text didn't match closely enough (see
                         ``tracked_changes_overlay``). This is a bonus
                         signal, never a requirement — most observations
                         will have ``attribution=None``.
        full_text:       The untruncated clause text (issue #105). Defaults
                         to ``text_summary`` when not supplied (via
                         ``__post_init__``) so existing callers/tests that
                         only ever dealt in short synthetic text keep
                         working unchanged; real callers pass the actual
                         full clause text explicitly. This is the field
                         our_standard.text, acceptable_if, and fallback/
                         rejected language must resolve from — never
                         text_summary.
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
    moves: list[RoundMove], limit: int = _TEXT_SUMMARY_MAX
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


def build_round_moves(
    document_id: str,
    doc_diff: DocumentDiff,
    tracked_by_vid: dict[str, TrackedChanges | None] | None = None,
    our_party_aliases: list[str] | None = None,
) -> list[RoundMove]:
    """Surface ``doc_diff.consecutive`` as ``RoundMove`` records (issue #177).

    One record per changed clause per negotiation round. ``moved_by`` is
    attributed from the destination version's own tracked-changes
    side-channel when one matches (each author's edits are tracked against
    the file they received, so the post-move version carries the mover's
    w:ins/w:del), mapped through *our_party_aliases*; ``"unknown"``
    otherwise — never guessed.
    """
    aliases = our_party_aliases or []
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
                    if enrichment is not None:
                        moved_by = party_side_for_author(enrichment.author, aliases)

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
) -> list[Observation]:
    """Assemble ``Observation`` objects for one document version.

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
                                  ``confidence=None``.
        has_signed_copy:            Whether the caller's version-ordering step
                                  (``order_versions``) actually identified a
                                  version as the executed copy of this
                                  document. Defaults to True for backward
                                  compatibility with existing callers/tests
                                  that don't model signed-copy detection.
                                  When False, every non-reversed observation's
                                  ``outcome`` is ``"unsigned"`` instead of
                                  ``"signed"`` — reporting a clause from a
                                  document with no detected signed copy as an
                                  accepted, signed position is exactly the
                                  fabrication issue #83 closes. Reversed
                                  clauses keep ``"proposed_then_reversed"``
                                  regardless — that label describes
                                  within-trail negotiation history
                                  (something was proposed, then reverted by a
                                  later draft), which holds independent of
                                  whether the final draft was ever executed.
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

    Returns:
        One ``Observation`` per entry in ``deviation_results``, PLUS one
        additional ``Observation`` per ``reversals`` entry whose clause never
        appears in ``deviation_results`` at all (issue #106 — see the
        "whole-clause reversals" block below).
    """
    # Match reversals on (taxonomy_id, clause_path) — not bare clause_path.
    # Bare-path matching (the previous behavior) can false-mark an unrelated
    # clause instance in the signed version that merely happens to land at
    # the same path number as a reversal detected in an earlier negotiation
    # round (e.g. after intervening clauses were added/removed and the
    # document renumbered) — two genuinely different clauses essentially
    # never also share a taxonomy_id, so pairing the two keys removes that
    # cross-contamination risk (issue #106). This mirrors the existing
    # clause-instance precision rationale in ReversalRecord.clause_path's
    # docstring.
    reversed_keys: set[tuple[str | None, str]] = {(r.taxonomy_id, r.clause_path) for r in reversals}
    default_outcome = "signed" if has_signed_copy else "unsigned"

    observations: list[Observation] = []
    obs_counter: dict[str, int] = {}
    # Tracks which (taxonomy_id, clause_path) keys were already represented by
    # a deviation_results row, so the whole-clause-reversal pass below never
    # double-emits for a reversal that also matched an existing observation.
    covered_keys: set[tuple[str | None, str]] = set()

    for idx, (clause_diff, dr) in enumerate(deviation_results):
        tid = clause_diff.taxonomy_id
        clause_path = clause_diff.clause_path_after or clause_diff.clause_path_before or "?"
        covered_keys.add((tid, clause_path))

        # citation.version_id / char_span (issue #108) must come from whichever
        # side of the diff clause_path was actually read from — a removed
        # clause's clause_path is the "before" side (clause_path_after is
        # None), so its version_id/char_span must be the "before" side too,
        # never the signed/last version this observation batch is filed
        # under. Mirrors the clause_path fallback above exactly.
        if clause_diff.clause_path_after is not None:
            cite_version_id = clause_diff.clause_version_after
            cite_char_span = clause_diff.char_span_after
        else:
            cite_version_id = clause_diff.clause_version_before
            cite_char_span = clause_diff.char_span_before

        # Build text summary from the "after" text (or "before" for removed clauses).
        # text_summary is a display-only truncation; full_text (issue #105) carries
        # the untruncated clause text for our_standard / acceptable_if / fallback
        # resolution and judge payloads downstream.
        raw_text = clause_diff.text_after or clause_diff.text_before
        text_summary = raw_text[:_TEXT_SUMMARY_MAX]

        outcome = (
            "proposed_then_reversed" if (tid, clause_path) in reversed_keys else default_outcome
        )

        # Stable observation_id: document_id + version + clause_path (deduplicated).
        base_id = f"{document_id}/{version}/{clause_path}"
        obs_counter[base_id] = obs_counter.get(base_id, 0) + 1
        count = obs_counter[base_id]
        obs_id = base_id if count == 1 else f"{base_id}#{count}"

        conf: float | None = (
            classification_confidences[idx]
            if classification_confidences is not None and idx < len(classification_confidences)
            else None
        )

        attribution: HunkEnrichment | None = (
            attributions[idx] if attributions is not None and idx < len(attributions) else None
        )

        # Negotiation dynamics (issue #177). Only derived when the caller
        # opted in via our_party_aliases; only for clauses where something
        # actually moved (a deviation="none" row records absence of change —
        # there is no proposal to attribute or date).
        proposed_by: str | None = None
        observed_at: str | None = None
        if our_party_aliases is not None and dr.deviation != "none":
            if attribution is not None:
                proposed_by = party_side_for_author(attribution.author, our_party_aliases)
                observed_at = _date_from_tracked(attribution.date)
            else:
                proposed_by = "unknown"

        observations.append(
            Observation(
                observation_id=obs_id,
                taxonomy_id=tid,
                text_summary=text_summary,
                full_text=raw_text,
                citation=ObservationCitation(
                    document_id=document_id,
                    version=version,
                    clause_path=clause_path,
                    char_span=cite_char_span,
                    version_id=cite_version_id,
                ),
                deviation=dr.deviation,
                risk_delta=dr.risk_delta.to_dict(),
                provenance=provenance,
                outcome=outcome,
                confidence=conf,
                basis=dr.basis,
                attribution=attribution,
                proposed_by=proposed_by,
                observed_at=observed_at,
            )
        )

    # Whole-clause reversals (issue #106): a clause inserted mid-negotiation
    # and removed again before the signed terminal is the cleanest "we
    # rejected this ask" signal available — but a clause absent from BOTH the
    # first and signed versions never produces a net-diff row at all
    # (clause_differ.diff_aligned skips any (before=None, after=None) pair),
    # so it never reaches ``deviation_results`` and was previously dropped
    # silently. Emit an Observation directly from each such ReversalRecord —
    # it already carries taxonomy_id, proposed_text, and version citations —
    # instead of trying to join it back through a net-diff row that does not
    # exist. Reversals that DID match an existing deviation_results row
    # (in-clause reversal — the clause instance persists to the signed
    # version, just with different final text) are skipped here: they were
    # already labeled ``proposed_then_reversed`` above.
    for r in reversals:
        key = (r.taxonomy_id, r.clause_path)
        if key in covered_keys:
            continue
        covered_keys.add(key)  # a repeated ReversalRecord for the same clause is not re-emitted

        base_id = f"{document_id}/{version}/{r.clause_path}"
        obs_counter[base_id] = obs_counter.get(base_id, 0) + 1
        count = obs_counter[base_id]
        obs_id = base_id if count == 1 else f"{base_id}#{count}"

        observations.append(
            Observation(
                observation_id=obs_id,
                taxonomy_id=r.taxonomy_id,
                text_summary=r.proposed_text[:_TEXT_SUMMARY_MAX],
                full_text=r.proposed_text,
                citation=ObservationCitation(
                    document_id=document_id,
                    version=version,
                    clause_path=r.clause_path,
                    char_span=r.char_span,
                    # r.clause_path is the clause instance path in the DRAFT
                    # version the proposal first appeared in, not the signed
                    # terminal (see ReversalRecord.clause_path's docstring) —
                    # version_id must cite version_inserted, never `version`
                    # (the signed ordinal this observation batch is filed
                    # under), or the citation resolves to the wrong file
                    # (issue #108).
                    version_id=r.version_inserted,
                ),
                # "substantive": the proposed text genuinely differed from the
                # signed terminal (that is exactly what detect_reversals
                # verified via its token-subset check) — never "none".
                deviation="substantive",
                # No DeviationJudge ever assessed this clause (it never
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
