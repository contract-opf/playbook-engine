"""Clause-type compiler — L5 pipeline stage.

Decides which clause types a playbook carries and what each one's standard
is. One ``ClausePosition`` per clause type — ``{id, taxonomy_id, title,
our_standard}`` — which ``playbook_engine.precedent`` turns into
``evidence.clauses`` and uses to key ``evidence.precedent``. Nothing here is
derived from risk direction, paper side or a judged verdict: the per-deal
facts live in the precedent record, and the consumer does the judging.

Design invariants:
  - ``our_standard`` comes only from the canonical template (``document_id
    "template"``, ``provenance="our_paper"``); a clause type with no
    non-empty template clause carries ``our_standard: null``.
  - Every asserted text carries a citation (``source_ref``).
  - Minimum-viable-observation floor (issue #210): an observation whose
    ``full_text`` is under ``MIN_OBSERVATION_TEXT_LEN`` characters after
    stripping leading/trailing whitespace (a segmentation fragment — a
    page-number artifact, a bare heading) never makes a clause type appear
    and never reaches precedent — never silently: it is counted and surfaced
    as a ``CoherenceFlag`` (severity ``"warn"``).
  - Observations with ``taxonomy_id=None`` (unclassified clauses) cannot be
    anchored to a template clause, so they are excluded from the clause
    list — but they are never silently dropped (issue #113): every call
    also returns an ``UnclassifiedCoverage`` summary (count, per-document
    breakdown, example citations) so a consumer can see corpus coverage
    without cross-referencing the AAR.

OPF 0.1-0.3's derived surfaces (observed positions, the clause library, the
historical-stance rollup and its tolerances, fallbacks and negotiation
trail) were retired with those formats (issue #238); git history has them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from playbook_engine.observation_builder import (
    OUTCOME_CONCEDED_BEFORE_SIGNING,
    Observation,
)

# Minimum length (issue #210), after stripping leading/trailing whitespace,
# for an observation's `full_text` to count as clause language. Segmentation
# occasionally emits sub-sentence fragments as their own clause node —
# page-number artifacts ("1 6"), a bare section heading with no body
# ("indemnification") — which would otherwise ride into the compiled
# playbook as if they were real observed clause language. This is a hard
# exclusion (precedent applies the same floor), but it is never silent:
# every quarantined observation is counted per taxonomy_id and surfaced as a
# ``CoherenceFlag`` (severity "warn") in `coherence_flags.json`, mirroring
# how `compute_unclassified_coverage` below surfaces taxonomy_id=None
# observations instead of just dropping them.
MIN_OBSERVATION_TEXT_LEN: int = 25


def _is_degenerate_observation_text(text: str) -> bool:
    """True when *text* is too short to be a usable observed clause (#210)."""
    return len(text.strip()) < MIN_OBSERVATION_TEXT_LEN


@dataclass(frozen=True)
class CoherenceFlag:
    """A compile-time warning about one clause type (``coherence_flags.json``).

    Attributes:
        clause_id:  The ClausePosition.id (e.g. ``"clause.indemnification"``).
        reason:     Human-readable explanation.
        severity:   ``"warn"`` — surfaced in the inspection report but does not
                    block the playbook; ``"block"`` — the playbook should not be
                    published without human review of this clause.
    """

    clause_id: str
    reason: str
    severity: Literal["warn", "block"]

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict for JSON persistence."""
        return {
            "clause_id": self.clause_id,
            "reason": self.reason,
            "severity": self.severity,
        }


# Observation outcomes that make a clause type appear in the playbook: a
# signed (or last-draft) text and a refused ask. "unsigned" rows (issue #83 —
# no version of the document was detected as the executed copy) are real
# corpus evidence but never on their own make a clause type appear;
# ``conceded_before_signing`` rows (our standard struck before signing,
# issue #216) do, and are handled separately below.
_CLAUSE_DEFINING_OUTCOMES = frozenset({"signed", "proposed_then_reversed"})


@dataclass(frozen=True)
class OPFCitation:
    """Citation anchor (OPF §4).

    ``version`` should be an integer for deal documents or the string
    ``"template"`` for the canonical template.
    """

    document_id: str
    version: str | int
    clause_path: str | None = None
    char_span: tuple[int, int] | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "document_id": self.document_id,
            "version": _normalize_version(self.version),
        }
        if self.clause_path is not None:
            d["clause_path"] = self.clause_path
        if self.char_span is not None:
            d["char_span"] = list(self.char_span)
        return d


@dataclass(frozen=True)
class OurStandard:
    """Our canonical clause text and its source citation."""

    text: str
    source_ref: OPFCitation

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "source_ref": self.source_ref.to_dict()}


# Maximum number of example citations surfaced per UnclassifiedCoverage
# summary (issue #113). The count/by_document fields already give exact
# totals; examples exist so a human can spot-check a handful of citations
# without the summary growing unbounded on a large corpus.
UNCLASSIFIED_EXAMPLE_LIMIT: int = 5


@dataclass(frozen=True)
class UnclassifiedCoverage:
    """Coverage summary for observations that could not be classified.

    Issue #113: ``taxonomy_id=None`` observations (unclassified clauses) are
    excluded from ``evidence.clauses`` because they cannot be anchored to a
    taxonomy entry — but that exclusion must never be silent. This summary
    is returned alongside the compiled output so a consumer can see corpus
    coverage (counts, per-document breakdown, example citations) without
    hunting through the AAR.
    """

    count: int
    by_document: dict[str, int]
    example_citations: tuple[OPFCitation, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "by_document": dict(self.by_document),
            "example_citations": [c.to_dict() for c in self.example_citations],
        }


def compute_unclassified_coverage(observations: list[Observation]) -> UnclassifiedCoverage:
    """Summarise the ``taxonomy_id=None`` observations in *observations*."""
    unclassified = [obs for obs in observations if obs.taxonomy_id is None]
    by_document: dict[str, int] = {}
    for obs in unclassified:
        doc_id = obs.citation.document_id
        by_document[doc_id] = by_document.get(doc_id, 0) + 1
    example_citations = tuple(
        OPFCitation(
            document_id=obs.citation.document_id,
            version=obs.citation.version,
            clause_path=obs.citation.clause_path,
            char_span=obs.citation.char_span,
        )
        for obs in unclassified[:UNCLASSIFIED_EXAMPLE_LIMIT]
    )
    return UnclassifiedCoverage(
        count=len(unclassified),
        by_document=by_document,
        example_citations=example_citations,
    )


@dataclass(frozen=True)
class ClausePosition:
    """One clause type of the playbook: id, title and our standard (if any)."""

    id: str
    taxonomy_id: str
    title: str
    our_standard: OurStandard | None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compile_clause_positions(
    observations: list[Observation],
    template_observations: list[Observation],
    taxonomy_titles: dict[str, str] | None = None,
) -> tuple[list[ClausePosition], list[CoherenceFlag], UnclassifiedCoverage]:
    """Decide the playbook's clause types and each one's ``our_standard``.

    A clause type appears when the template has a clause of it, or a deal
    observation of it is a signed text, a refused ask
    (``proposed_then_reversed``) or our standard struck before signing
    (``conceded_before_signing``) — sub-sentence fragments excluded.

    Args:
        observations:        All L4 observations from the deal corpus.
        template_observations: Observations extracted from the canonical
                             template document.  Must have
                             ``citation.document_id`` set to ``"template"``
                             and ``provenance="our_paper"``.  Used as source
                             for ``our_standard``.
        taxonomy_titles:     Optional mapping ``{taxonomy_id: human_title}``.
                             Falls back to ``_title_from_id()`` when absent.

    Returns:
        Tuple of (positions, coherence_flags, unclassified_coverage):
          - ``positions``: One ``ClausePosition`` per clause type, sorted by
            ``taxonomy_id``.
          - ``coherence_flags``: one ``"warn"`` flag per clause type that had
            sub-sentence fragments quarantined (issue #210).
          - ``unclassified_coverage``: Summary of ``observations`` entries
            with ``taxonomy_id=None`` (issue #113).

    Raises:
        ValueError:  If a template observation has provenance other than
                     ``"our_paper"``.
    """
    for tmpl_obs in template_observations:
        if tmpl_obs.provenance != "our_paper":
            raise ValueError(
                f"Template observation must have provenance='our_paper'; "
                f"got {tmpl_obs.provenance!r} for taxonomy_id={tmpl_obs.taxonomy_id!r}."
            )

    # --- template map (taxonomy_id → first template observation) ---
    template_map: dict[str, Observation] = {}
    for tmpl in template_observations:
        if tmpl.taxonomy_id is not None and tmpl.taxonomy_id not in template_map:
            template_map[tmpl.taxonomy_id] = tmpl

    # --- clause types evidenced by the deals (skip None and fragments) ---
    deal_tids: set[str] = set()
    # Sub-sentence fragments (issue #210) are counted per taxonomy_id so a
    # CoherenceFlag can name each affected clause rather than dropping them
    # silently.
    quarantined_by_tid: dict[str, int] = {}
    for obs in observations:
        if obs.taxonomy_id is None:
            continue
        if obs.outcome == OUTCOME_CONCEDED_BEFORE_SIGNING:
            if not _is_degenerate_observation_text(obs.full_text):
                deal_tids.add(obs.taxonomy_id)
            continue
        if obs.outcome not in _CLAUSE_DEFINING_OUTCOMES:
            continue
        if _is_degenerate_observation_text(obs.full_text):
            quarantined_by_tid[obs.taxonomy_id] = quarantined_by_tid.get(obs.taxonomy_id, 0) + 1
            continue
        deal_tids.add(obs.taxonomy_id)

    unclassified_coverage = compute_unclassified_coverage(observations)

    coherence_flags = [
        CoherenceFlag(
            clause_id=f"clause.{tid}",
            reason=(
                f"{count} observation(s) excluded from precedent: "
                f"full_text shorter than {MIN_OBSERVATION_TEXT_LEN} "
                "character(s) — a segmentation fragment "
                "(e.g. a page-number artifact or bare heading), not usable "
                "clause language"
            ),
            severity="warn",
        )
        for tid, count in sorted(quarantined_by_tid.items())
    ]

    positions: list[ClausePosition] = []
    for tid in sorted(deal_tids | template_map.keys()):
        t_obs = template_map.get(tid)
        our_standard: OurStandard | None = None
        # An empty-text template observation is not a usable standard
        # (issue #182).
        if t_obs is not None and t_obs.full_text.strip():
            our_standard = OurStandard(
                # Full clause text (issue #105).
                text=t_obs.full_text,
                source_ref=OPFCitation(
                    document_id=t_obs.citation.document_id,
                    version=t_obs.citation.version,
                    clause_path=t_obs.citation.clause_path,
                    char_span=t_obs.citation.char_span,
                ),
            )
        positions.append(
            ClausePosition(
                id=f"clause.{tid}",
                taxonomy_id=tid,
                title=(taxonomy_titles or {}).get(tid) or _title_from_id(tid),
                our_standard=our_standard,
            )
        )

    return positions, coherence_flags, unclassified_coverage


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _normalize_version(version: str | int) -> str | int:
    """Normalize a version value to match OPF citation schema (integer | "template").

    Converts string versions like "v2", "v3", "2" to integers.
    Passes integers and the literal "template" through unchanged.
    Non-parseable strings are passed through (let the schema validator catch them).
    """
    if isinstance(version, int) or version == "template":
        return version
    # Strip leading "v"/"V" and try integer parse.
    stripped = version.lstrip("vV")
    try:
        return int(stripped)
    except ValueError:
        return version  # pass through; schema validator will report if invalid


def _title_from_id(taxonomy_id: str) -> str:
    """Convert ``snake_case_id`` → ``Title Case Words``."""
    return " ".join(word.capitalize() for word in taxonomy_id.split("_"))
