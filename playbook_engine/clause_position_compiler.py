"""ClausePosition compiler — L5 pipeline stage.

Aggregates ``Observation`` objects (L4 output) into ``ClausePosition`` records
that serialise to the OPF v0.2 ``clausePosition`` shape (§3.5). Internally,
each clause type's derivation still runs the v0.1-era 4-way ``position``
cascade documented below (it already encodes every cap this module enforces);
``ClausePosition.to_dict()`` translates the concluded ``position`` into the
OPF v0.2-facing, descriptive ``summary.historical_stance`` via
``_historical_stance()`` — see that function's docstring for the mapping.

Design invariants:
  - OPF §2.2 provenance rule is structurally enforced: no code path can emit
    ``our_standard`` or a ``rollup.position`` stronger than ``"negotiable"``
    for a taxonomy_id that has zero our-paper observations.
  - Mirroring §2.2: no code path can emit a ``rollup.position`` stronger than
    ``"negotiable"`` for a taxonomy_id with any ``basis="stub"`` observation
    (no judge configured at all — see ``_STUB_BASES``).
  - Evidence-depth cap (issue #107): no code path can emit a
    ``rollup.position`` stronger than ``"negotiable"`` for a taxonomy_id with
    fewer than ``MIN_EVIDENCE_N`` our-paper observations. Confidence
    ``score`` measures provenance quality, not evidence depth — a single
    our-paper observation scores 1.0 the same as a hundred would, so the
    position cap (not the score) is what protects against sparse-evidence
    over-confidence.
  - Every asserted text carries a citation (``source_ref`` or ``example_ref``).
  - Observations with ``taxonomy_id=None`` (unclassified clauses) cannot be
    anchored to a template clause, so they are excluded from the ``clauses``
    array — but they are never silently dropped (issue #113): every call
    also returns an ``UnclassifiedCoverage`` summary (count, per-document
    breakdown, example citations) so a consumer can see corpus coverage
    without cross-referencing the AAR.

Position derivation (from our-paper observations only):
  ``"standard"``                — all signed our-paper observations have
                                  deviation ``"none"`` and neutral/better risk.
  ``"acceptable_variants_exist"`` — neutral-risk signed variants exist.
  ``"negotiable"``              — worse-risk signed observations exist (we have
                                  conceded before), or no our-paper at all.
  ``"hold_firm"``               — proposed_then_reversed observations exist with
                                  no concessions.

Confidence score:
  ``(n_our_paper * 1.0 + n_counterparty_paper * 0.5) / total`` — our-paper
  weighted 2× counterparty-paper; recorded in ``confidence.basis``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

from playbook_engine.observation_builder import Observation, RoundMove

# ---------------------------------------------------------------------------
# CoherenceJudge — LLM seam for flagging unreliable clause positions
# ---------------------------------------------------------------------------

# Minimum number of our-paper citations for a clause to be considered
# well-grounded; clauses below this threshold trigger CoherenceJudge.
COHERENCE_MIN_CITATIONS: int = 3

# Default minimum number of distinct our-paper observations required before a
# clause may be assigned a position stronger than "negotiable" (issue #107).
# Below this floor there is not enough independent evidence to call a clause
# "standard"/"acceptable_variants_exist"/"hold_firm" — a single our-paper
# agreement is one data point, not a pattern. This is a hard structural cap,
# mirroring the §2.2 provenance cap and the stub-basis cap below: it applies
# regardless of what CoherenceJudge (if any) would otherwise conclude.
#
# Producer-configurable (issue #144, config.provenance.min_evidence_n) —
# this module-level constant is only the DEFAULT used when a caller does not
# supply its own ``min_evidence_n``. ``validator.py`` imports this same
# constant as its own default so the two stay aligned on one rule absent an
# explicit override.
MIN_EVIDENCE_N: int = 2


@dataclass(frozen=True)
class CoherenceFlag:
    """Flag emitted by CoherenceJudge for an unreliable clause position.

    Attributes:
        clause_id:  The ClausePosition.id (e.g. ``"clause.indemnification"``).
        reason:     Human-readable explanation of the incoherence.
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


@runtime_checkable
class CoherenceJudge(Protocol):
    """Protocol for LLM-assisted coherence review of assembled clause positions.

    Implementations receive a minimal clause summary dict (not raw citation
    text) and return ``None`` if the clause position is coherent, or a
    ``CoherenceFlag`` describing the incoherence.  The judge is called only
    for flagged clauses (10–20% of the playbook), not a full re-read.

    The summary dict passed to ``judge()`` contains:
        ``clause_id``     str   — ClausePosition.id
        ``position``      str   — rollup position (standard/negotiable/…)
        ``n_our_paper``   int   — number of our-paper observations
        ``risk_delta_directions``  list[str]  — risk_delta.direction values across citations
        ``is_fallback``   bool  — True when position is "negotiable" due to a
                                  fallback (position-vs-fallback tension indicator)
    """

    def judge(self, clause_summary: dict[str, Any]) -> CoherenceFlag | None:
        """Review one clause summary and return a flag, or None if coherent."""
        ...


# ---------------------------------------------------------------------------
# OPF output types (mirror the JSON shapes defined in spec/playbook.schema.json)
# ---------------------------------------------------------------------------

_POSITION_STRONGER_THAN_NEGOTIABLE = frozenset(
    {"standard", "acceptable_variants_exist", "hold_firm"}
)

_VALID_POSITIONS = frozenset({"standard", "acceptable_variants_exist", "negotiable", "hold_firm"})

# A signed clause counts as an "acceptable variant" only when its deviation was
# actually assessed by a judge. These bases mean it was NOT: ``"needs_review"``
# (judge raised / low-confidence), ``"judge_error"`` (judge raised), and
# ``"stub"`` (no judge configured at all — see ``_STUB_BASES`` below). An
# observation carrying one of these must never manufacture an
# ``acceptable_variants_exist`` position or an ``acceptable_if`` entry — its
# neutral risk_delta is a placeholder, not a real assessment.
_UNJUDGED_BASES = frozenset({"needs_review", "judge_error", "stub"})

# "stub" observations are stricter than the other _UNJUDGED_BASES values:
# a "needs_review"/"judge_error" observation may still fall through to
# "standard" when it's the only signal for a clause (nothing else claims a
# deviation, so "standard" is the intentional fallback — see
# test_unjudged_deviation_does_not_fabricate_acceptable_variants). A "stub"
# observation gets no such benefit of the doubt: it means NO judge was
# configured at all (systemic, not a one-off failure), so any clause type
# with a stub-basis observation is capped at "negotiable" — mirroring the
# §2.2 provenance cap — and can never reach "standard", "acceptable_variants_
# exist", or "hold_firm".
_STUB_BASES = frozenset({"stub"})

# The only OPF-conformant observation outcomes (spec/playbook.schema.json's
# observation.outcome enum). An internal Observation may carry a third value
# such as "unsigned" (issue #83 — no version of the document was detected as
# the executed copy); such observations are real corpus evidence but not
# position-defining OPF outcomes and must be withheld from observed_positions
# / rollups rather than reach the schema-validated playbook output.
_OPF_OUTCOMES = frozenset({"signed", "proposed_then_reversed"})


@dataclass(frozen=True)
class OPFCitation:
    """Citation anchor (OPF §4).

    ``version`` should be an integer for deal documents or the string
    ``"template"`` for the canonical template.  The schema validator (issue
    #25) will enforce this; passing a raw version string through is acceptable
    for internal use.
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
class ObservedPosition:
    """One observed clause variant for an OPF ClausePosition.

    Mirrors the ``observation`` sub-schema defined in §3.4.

    ``full_text`` (issue #105) carries the untruncated clause text alongside
    the 200-char ``text_summary`` — fallback/rejected language in particular
    is exactly the "acceptable alternative language" lawyers need verbatim,
    not a fragment. Optional in the OPF schema; defaults to ``text_summary``
    when not supplied.
    """

    text_summary: str
    example_ref: OPFCitation
    deviation: str
    risk_delta: dict[str, str]
    provenance: str
    outcome: str
    precedent_count: int = 1
    full_text: str = ""
    # Negotiation dynamics (issue #177, OPF §3.5.3) — copied from the source
    # Observation; optional-when-underivable, so to_dict() emits the keys
    # only when set (never null placeholders).
    proposed_by: str | None = None
    observed_at: str | None = None
    counterparty_ref: dict[str, str] | None = None

    def __post_init__(self) -> None:
        if not self.full_text:
            object.__setattr__(self, "full_text", self.text_summary)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "text_summary": self.text_summary,
            "full_text": self.full_text,
            "example_ref": self.example_ref.to_dict(),
            "deviation": self.deviation,
            "risk_delta": self.risk_delta,
            "provenance": self.provenance,
            "outcome": self.outcome,
            "precedent_count": self.precedent_count,
        }
        if self.proposed_by is not None:
            d["proposed_by"] = self.proposed_by
        if self.observed_at is not None:
            d["observed_at"] = self.observed_at
        if self.counterparty_ref is not None:
            d["counterparty_ref"] = self.counterparty_ref
        return d


@dataclass(frozen=True)
class AcceptableIfEntry:
    """Structured tolerance-condition triple (issue #141, OPF v0.2 §3.5).

    Replaces v0.1/early-v0.2's free-text ``acceptable_if`` entries with the
    ``{if, to, rationale}`` shape consuming review applications already
    prove out as ``acceptable_variations`` — this is what lets a consumer
    run the acceptable-variations-vs-Floor consistency lint (see issue #127)
    instead of reasoning over bare strings. ``observation_ref`` cites the observation
    this entry was derived from, so the tolerance is traceable evidence, not
    an assertion.

    ``if_`` (not ``if``, a Python keyword) serialises to the schema's ``if``
    key in ``to_dict()``.
    """

    if_: str
    to: str
    rationale: str
    observation_ref: OPFCitation

    def to_dict(self) -> dict[str, Any]:
        return {
            "if": self.if_,
            "to": self.to,
            "rationale": self.rationale,
            "observation_ref": self.observation_ref.to_dict(),
        }


@dataclass(frozen=True)
class OurStandard:
    """Our canonical clause text and its source citation."""

    text: str
    source_ref: OPFCitation

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "source_ref": self.source_ref.to_dict()}


@dataclass(frozen=True)
class ClauseRollup:
    """Internal derivation state for one clause type (v0.1-era 4-way cascade).

    Not emitted directly. OPF v0.2's OUTPUT shape is the descriptive
    ``summary.historical_stance`` (see ``_historical_stance()`` below and
    ``ClausePosition.to_dict()``, which translates this internal state into
    that OPF-facing block) rather than this prescriptive ``position`` enum.
    The cascade itself is retained unchanged because it already encodes every
    cap this module must enforce (§2.2 provenance, stub-basis, evidence-depth)
    — the translation only relabels the *conclusion*, not the derivation.
    """

    position: str  # standard|acceptable_variants_exist|negotiable|hold_firm
    acceptable_if: tuple[AcceptableIfEntry, ...]
    fallbacks: tuple[ObservedPosition, ...]
    rejected: tuple[ObservedPosition, ...]
    confidence: dict[str, Any]
    # Held-rate behind the historical_stance enum (issue #177, resolves spec
    # Appendix A.3): {"held": H, "of": N, "basis": "our_paper"|"all"} —
    # computed in _derive_rollup from the same observation group the stance
    # derives from. None only for legacy construction paths (tests) that
    # never computed it.
    stance_detail: dict[str, Any] | None = None
    # Set when >=1 observation in this clause's group had basis="stub" (no
    # judge configured at all). Mirrors the cap already applied to `position`
    # in `_derive_rollup`; carried here too because `_historical_stance()`
    # needs it to distinguish "no reliable signal" (no_signal) from a genuine
    # observed concession pattern (usually_conceded/mixed) when translating.
    stub_basis_present: bool = False

    def __post_init__(self) -> None:
        if self.position not in _VALID_POSITIONS:
            raise ValueError(
                f"ClauseRollup.position must be one of {sorted(_VALID_POSITIONS)!r}; "
                f"got {self.position!r}"
            )


# historical_stance values stronger than "mixed" (mirrors
# validator._STANCES_STRONGER_THAN_MIXED) — kept local so this module does not
# import from validator.py just to reuse a literal set.
_STANCE_FOR_POSITION: dict[str, str] = {
    # "standard" (all signed our-paper observations match our_standard, no
    # deviation) and "hold_firm" (asks were made and refused, no concessions)
    # both describe the same historical fact: our position was never
    # conceded — so both collapse to "consistently_held".
    "standard": "consistently_held",
    "hold_firm": "consistently_held",
    # Neutral-risk signed variants exist alongside our_standard — we usually
    # hold, but some tolerated variation is on record.
    "acceptable_variants_exist": "usually_held",
}


# historical_stance values that OPF §2.2 requires an ``our_standard`` to back
# (mirrors validator._STANCES_STRONGER_THAN_MIXED). A clause with no our_standard
# (no template clause for this taxonomy) may not emit any of these.
_STANCES_REQUIRING_OUR_STANDARD: frozenset[str] = frozenset(
    {"usually_conceded", "usually_held", "consistently_held"}
)


def _historical_stance(rollup: ClauseRollup, *, has_our_standard: bool = True) -> str:
    """Translate ``rollup`` (internal 4-way position cascade) into OPF v0.2's
    descriptive ``summary.historical_stance`` (§3.5, §2.2/§2.3).

    ``has_our_standard`` (issue #182): OPF §2.2 requires any stance stronger than
    "mixed" to reference an ``our_standard`` clause. When this clause has none
    (our-paper observations exist but no template clause covers this taxonomy),
    such a stance is capped to "no_signal" — we have evidence but no standard to
    characterise a settled stance against. Defaults ``True`` so template-grounded
    callers are unaffected.

    historical_stance answers "what has the corpus shown", never "what must
    you do" — see OPF-SPEC.md §2.2 and the opf-v0.2-redesign
    rationale. The five values:

      "no_signal"           — no reliable our-paper signal: either the §2.2
                              provenance cap applies (zero/insufficient
                              our-paper evidence — ``evidence_sufficient`` is
                              False) or a stub-basis observation means no
                              judge ever assessed this clause type. Both are
                              carried on ``rollup`` already (confidence dict
                              and ``stub_basis_present``).
      "usually_conceded"    — has_our_paper evidence exists, a real judge
                              assessed it, and the corpus shows we have
                              conceded before (fallbacks present) with no
                              our-paper rejections on record.
      "mixed"               — genuinely contradictory evidence: the corpus
                              shows BOTH a concession (fallback) AND an
                              our-paper rejection (proposed_then_reversed)
                              for the same clause type.
      "consistently_held" / "usually_held" — see ``_STANCE_FOR_POSITION``.

    Note: within the reachable ``position == "negotiable"`` branch of
    ``_derive_rollup``/``_derive_position``, "negotiable" is *only* ever
    returned once evidence is sufficient and non-stub for a genuine
    concession pattern (``fallback_obs`` truthy) — the "no signal" reasons
    for "negotiable" are already filtered out by the ``evidence_sufficient``/
    ``stub_basis_present`` checks above, so ``rollup.fallbacks`` is
    guaranteed non-empty whenever this function reaches that branch.
    """
    evidence_sufficient = bool(rollup.confidence.get("evidence_sufficient", False))
    if rollup.stub_basis_present or not evidence_sufficient:
        return "no_signal"
    if rollup.position == "negotiable":
        our_paper_rejected = any(op.provenance == "our_paper" for op in rollup.rejected)
        stance = "mixed" if our_paper_rejected else "usually_conceded"
    else:
        stance = _STANCE_FOR_POSITION.get(rollup.position, "no_signal")
    # §2.2 (issue #182): a stronger-than-"mixed" stance needs an our_standard.
    if not has_our_standard and stance in _STANCES_REQUIRING_OUR_STANDARD:
        return "no_signal"
    return stance


# Maximum number of example citations surfaced per UnclassifiedCoverage
# summary (issue #113). The count/by_document fields already give exact
# totals; examples exist so a human can spot-check a handful of citations
# without the summary growing unbounded on a large corpus.
UNCLASSIFIED_EXAMPLE_LIMIT: int = 5


@dataclass(frozen=True)
class UnclassifiedCoverage:
    """Coverage summary for observations that could not be classified.

    Issue #113: ``taxonomy_id=None`` observations (unclassified clauses) are
    excluded from ``clauses``/``clause_library`` because they cannot be
    anchored to a taxonomy entry — but that exclusion must never be silent.
    This summary is returned alongside the compiled output so a consumer can
    see corpus coverage (counts, per-document breakdown, example citations)
    without hunting through the AAR.
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
    """Summarise the ``taxonomy_id=None`` observations in *observations*.

    Shared by ``compile_clause_positions`` and ``compile_clause_library``
    (issue #113) so both L5 stages surface the same coverage shape instead of
    each silently filtering unclassified observations out of its own output.
    """
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
    """Template-anchored clause position (OPF §3.4)."""

    id: str
    taxonomy_id: str
    title: str
    our_standard: OurStandard | None
    observed_positions: tuple[ObservedPosition, ...]
    rollup: ClauseRollup
    # Round-by-round ask→landing trajectory for this clause type (issue
    # #177, OPF §3.5.3) — RoundMove records grouped by taxonomy_id in
    # compile_clause_positions. Empty for single-version deals or legacy
    # stores with no round_moves.jsonl.
    negotiation_trail: tuple[RoundMove, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Serialise to the OPF v0.2 ``clausePosition`` shape (§3.5).

        ``rollup`` (this module's internal 4-way position cascade) is
        translated into the OPF-facing ``summary`` block here —
        ``historical_stance`` replaces the prescriptive ``position``;
        ``fallbacks``/``rejected``/``confidence`` carry over unchanged (only
        their wrapper key changed, v0.1's ``rollup`` → v0.2's ``summary``).
        ``acceptable_if`` entries are serialised as ``{if,to,rationale,
        observation_ref}`` triples (issue #141), not bare strings.
        """
        summary: dict[str, Any] = {
            "historical_stance": _historical_stance(
                self.rollup, has_our_standard=self.our_standard is not None
            ),
            "acceptable_if": [entry.to_dict() for entry in self.rollup.acceptable_if],
            "fallbacks": [fb.to_dict() for fb in self.rollup.fallbacks],
            "rejected": [rej.to_dict() for rej in self.rollup.rejected],
            "confidence": self.rollup.confidence,
        }
        # stance_detail / negotiation_trail (issue #177, §3.5.3): emitted
        # only when derived — absent keys signal "not computed", never null.
        if self.rollup.stance_detail is not None:
            summary["stance_detail"] = self.rollup.stance_detail
        d: dict[str, Any] = {
            "id": self.id,
            "taxonomy_id": self.taxonomy_id,
            "title": self.title,
            "our_standard": self.our_standard.to_dict() if self.our_standard else None,
            "observed_positions": [op.to_dict() for op in self.observed_positions],
            "summary": summary,
        }
        if self.negotiation_trail:
            d["negotiation_trail"] = [m.to_opf_dict() for m in self.negotiation_trail]
        return d


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compile_clause_positions(
    observations: list[Observation],
    template_observations: list[Observation],
    taxonomy_titles: dict[str, str] | None = None,
    *,
    coherence_judge: CoherenceJudge | None = None,
    min_evidence_n: int = MIN_EVIDENCE_N,
    round_moves: list[RoundMove] | None = None,
) -> tuple[list[ClausePosition], list[CoherenceFlag], UnclassifiedCoverage]:
    """Aggregate observations into OPF-conformant ClausePosition records.

    Args:
        observations:        All L4 observations from deal corpus (signed +
                             proposed_then_reversed), any provenance mix.
        template_observations: Observations extracted from the canonical
                             template document.  Must have
                             ``citation.document_id`` set to ``"template"``
                             and ``provenance="our_paper"``.  Used as source
                             for ``our_standard``.
        taxonomy_titles:     Optional mapping ``{taxonomy_id: human_title}``.
                             Falls back to ``_title_from_id()`` when absent.
        coherence_judge:     Optional judge called for clauses where the
                             position cascade is unreliable (low n_our_paper,
                             contradictory risk_delta, or fallback tension).
                             When ``None``, coherence checks are skipped.
        min_evidence_n:      Producer-configurable evidence-depth floor
                             (issue #144, config.provenance.min_evidence_n) —
                             the minimum number of distinct our-paper
                             observations required before a clause may be
                             assigned a position/historical_stance stronger
                             than "negotiable"/"mixed". Defaults to
                             ``MIN_EVIDENCE_N`` (2).
        round_moves:         Per-round clause moves from the L4 store
                             (``round_moves.jsonl``, issue #177) — grouped by
                             taxonomy_id into each ClausePosition's
                             ``negotiation_trail``, ordered by (document_id,
                             round). ``None``/empty (single-version corpora,
                             legacy stores) emits no trail.

    Returns:
        Tuple of (positions, coherence_flags, unclassified_coverage):
          - ``positions``: One ``ClausePosition`` per distinct non-None
            ``taxonomy_id`` in sorted order.
          - ``coherence_flags``: ``CoherenceFlag`` entries emitted by the
            judge for risky clauses; empty list when no judge is configured
            or all clauses pass.
          - ``unclassified_coverage``: Summary (count, per-document
            breakdown, example citations) of ``observations`` entries with
            ``taxonomy_id=None`` (issue #113) — these are excluded from
            ``positions`` but never silently dropped from the return value.

    Raises:
        ValueError:  If a template observation has provenance other than
                     ``"our_paper"``.

    OPF §2.2 guarantee:
        taxonomy_ids with zero our-paper observations will have
        ``our_standard=None`` and ``rollup.position="negotiable"``.
        No code path can produce a stronger position for them.
    """
    for tmpl_obs in template_observations:
        if tmpl_obs.provenance != "our_paper":
            raise ValueError(
                f"Template observation must have provenance='our_paper'; "
                f"got {tmpl_obs.provenance!r} for taxonomy_id={tmpl_obs.taxonomy_id!r}."
            )

    # --- build template map (taxonomy_id → first template observation) ---
    template_map: dict[str, Observation] = {}
    for tmpl in template_observations:
        if tmpl.taxonomy_id is not None and tmpl.taxonomy_id not in template_map:
            template_map[tmpl.taxonomy_id] = tmpl

    # --- group deal observations by taxonomy_id (skip None) ---
    #
    # Only "signed" and "proposed_then_reversed" are OPF-conformant outcomes
    # (spec/playbook.schema.json's observation.outcome enum) — the two
    # position-defining buckets the spec actually models. An observation with
    # any other outcome (e.g. "unsigned" — issue #83: no version of this
    # document was detected as the executed copy) is real corpus evidence but
    # not a position-defining OPF outcome; it must be withheld here rather
    # than fabricate/relabel it to fit the schema. It remains visible in
    # observations.jsonl and the inspection report for human review.
    groups: dict[str, list[Observation]] = {}
    for obs in observations:
        if obs.taxonomy_id is None:
            continue
        if obs.outcome not in _OPF_OUTCOMES:
            continue
        groups.setdefault(obs.taxonomy_id, []).append(obs)

    # taxonomy_id=None observations are excluded from `groups` above (they
    # cannot be anchored to a template clause) but are never silently
    # dropped — issue #113: summarised into unclassified_coverage below and
    # returned alongside positions/coherence_flags.
    unclassified_coverage = compute_unclassified_coverage(observations)

    # --- group round moves by taxonomy_id (issue #177) ---
    # Unclassified moves (taxonomy_id=None) cannot anchor to a ClausePosition;
    # like unclassified observations they stay visible in the L4 store
    # (round_moves.jsonl) rather than reach the playbook.
    trail_by_tid: dict[str, list[RoundMove]] = {}
    for move in round_moves or []:
        if move.taxonomy_id is None:
            continue
        trail_by_tid.setdefault(move.taxonomy_id, []).append(move)
    for moves in trail_by_tid.values():
        moves.sort(key=lambda m: (m.document_id, m.round))

    # Collect all taxonomy_ids (from both deal observations and template).
    all_tids = sorted(groups.keys() | template_map.keys())

    positions: list[ClausePosition] = []
    coherence_flags: list[CoherenceFlag] = []

    for tid in all_tids:
        group = groups.get(tid, [])
        t_obs = template_map.get(tid)

        our_paper_obs = [obs for obs in group if obs.provenance == "our_paper"]
        # An empty-text template observation is not a usable standard (issue
        # #182), so it must not count as "we have our paper" for this clause —
        # otherwise the rollup could claim a standard the playbook can't show.
        has_our_paper = bool(our_paper_obs) or (t_obs is not None and t_obs.full_text.strip() != "")

        # ---------------------------------------------------------------
        # §2.2 enforcement — our_standard
        # ---------------------------------------------------------------
        our_standard: OurStandard | None = None
        if has_our_paper and t_obs is not None and t_obs.full_text.strip():
            our_standard = OurStandard(
                # Full clause text (issue #105) — text_summary is a 200-char
                # display fragment, useless as a drafting standard for any
                # real indemnification/insurance clause.
                text=t_obs.full_text,
                source_ref=OPFCitation(
                    document_id=t_obs.citation.document_id,
                    version=t_obs.citation.version,
                    clause_path=t_obs.citation.clause_path,
                    char_span=t_obs.citation.char_span,
                ),
            )
        # If has_our_paper is False, our_standard stays None — §2.2 enforced.

        # ---------------------------------------------------------------
        # Build observed_positions
        # ---------------------------------------------------------------
        # precedent_count (issue #107): identical clause texts across deals
        # are aggregated by normalized text rather than left at the dataclass
        # default of 1 for every observation.
        precedent_counts = _count_precedents(group)
        observed_positions = tuple(
            _obs_to_observed_position(obs, precedent_counts) for obs in group
        )

        # ---------------------------------------------------------------
        # Derive rollup
        # ---------------------------------------------------------------
        rollup = _derive_rollup(
            group,
            our_paper_obs,
            has_our_paper,
            precedent_counts,
            min_evidence_n=min_evidence_n,
            # §2.2 (issue #182): a strong position needs an our_standard to point
            # at; without one (no template clause for this taxonomy) cap at
            # negotiable so historical_stance stays validator-consistent.
            has_our_standard=our_standard is not None,
        )

        title = (taxonomy_titles or {}).get(tid) or _title_from_id(tid)
        clause_id = f"clause.{tid}"
        positions.append(
            ClausePosition(
                id=clause_id,
                taxonomy_id=tid,
                title=title,
                our_standard=our_standard,
                observed_positions=observed_positions,
                rollup=rollup,
                negotiation_trail=tuple(trail_by_tid.get(tid, ())),
            )
        )

        # ---------------------------------------------------------------
        # CoherenceJudge gate — call for risky ~10–20% of clauses only
        # ---------------------------------------------------------------
        if coherence_judge is not None:
            n_our_paper = rollup.confidence["n_our_paper"]
            risk_directions = [obs.risk_delta.get("direction", "neutral") for obs in group]
            # Trigger conditions (issue #54):
            # 1. Low citation count — position is under-grounded.
            # 2. Contradictory risk_delta directions across citations.
            # 3. Position is "negotiable" due to a fallback despite high-
            #    confidence citations (position-vs-fallback tension).
            low_citations = n_our_paper < COHERENCE_MIN_CITATIONS
            contradictory_risk = (
                len(set(risk_directions)) > 1
                and "worse" in risk_directions
                and "better" in risk_directions
            )
            fallback_tension = rollup.position == "negotiable" and bool(rollup.fallbacks)

            if low_citations or contradictory_risk or fallback_tension:
                summary: dict[str, Any] = {
                    "clause_id": clause_id,
                    "position": rollup.position,
                    "n_our_paper": n_our_paper,
                    "risk_delta_directions": risk_directions,
                    "is_fallback": fallback_tension,
                }
                flag = coherence_judge.judge(summary)
                if flag is not None:
                    coherence_flags.append(flag)

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


def _normalize_for_dedup(text: str) -> str:
    """Collapse whitespace and casefold for identical-precedent matching.

    Deliberately conservative: only whitespace and case differences are
    treated as "identical" (e.g. line-wrapping artifacts from extraction).
    Genuinely different wording is left as a distinct precedent.
    """
    return " ".join(text.split()).casefold()


def _count_precedents(group: list[Observation]) -> dict[str, int]:
    """Count observations per normalized full_text within one taxonomy group.

    Issue #107: ``ObservedPosition.precedent_count`` was always 1 — identical
    clause texts across deals were never aggregated into a strength signal.
    This counts, across ALL observations in the group (any provenance), how
    many share the same normalized text, so a variant seen in 5 agreements
    is distinguishable from one seen in exactly 1.
    """
    counts: dict[str, int] = {}
    for obs in group:
        key = _normalize_for_dedup(obs.full_text)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _obs_to_observed_position(
    obs: Observation, precedent_counts: dict[str, int] | None = None
) -> ObservedPosition:
    precedent_count = 1
    if precedent_counts is not None:
        precedent_count = precedent_counts.get(_normalize_for_dedup(obs.full_text), 1)
    return ObservedPosition(
        text_summary=obs.text_summary,
        full_text=obs.full_text,
        example_ref=OPFCitation(
            document_id=obs.citation.document_id,
            version=obs.citation.version,
            clause_path=obs.citation.clause_path,
            char_span=obs.citation.char_span,
        ),
        deviation=obs.deviation,
        risk_delta=obs.risk_delta,
        provenance=obs.provenance,
        outcome=obs.outcome,
        precedent_count=precedent_count,
        proposed_by=obs.proposed_by,
        observed_at=obs.observed_at,
        counterparty_ref=obs.counterparty_ref,
    )


_MAGNITUDE_ORDER: dict[str, int] = {"minor": 0, "material": 1}


def _derive_rollup(
    group: list[Observation],
    our_paper_obs: list[Observation],
    has_our_paper: bool,
    precedent_counts: dict[str, int] | None = None,
    *,
    min_evidence_n: int = MIN_EVIDENCE_N,
    has_our_standard: bool = True,
) -> ClauseRollup:
    """Derive rollup guidance from the observation group.

    §2.2 cap: if not has_our_paper, position is capped at ``"negotiable"``.

    ``has_our_standard`` (issue #182): a strong position/historical_stance
    ("standard"/"hold_firm" → "consistently_held") must be backed by an
    ``our_standard`` clause per OPF §2.2 — the validator rejects a strong
    stance with ``our_standard: null``. our-paper observations can exist for a
    taxonomy that has NO template clause (emergent, or a template that doesn't
    cover this clause type), which yields ``has_our_paper=True`` but no
    ``our_standard``; capping at "negotiable" in that case keeps the rollup
    §2.2-consistent. Defaults to ``True`` so callers grounding every strong
    position in a template are unaffected.

    Confidence (§6) is published as two orthogonal numbers (issue #107):
      - ``score`` — provenance QUALITY: weighted = n_our_paper * 1.0 +
        n_counterparty_paper * 0.5; score = weighted / total. Our-paper
        observations are weighted 2× counterparty-paper because they
        represent positive choices (we proposed or accepted against our own
        template), not mere tolerance. This is NOT a measure of how much
        evidence exists — a single our-paper observation scores 1.0 here,
        same as a hundred.
      - ``evidence_sufficient`` — evidence DEPTH: whether ``n_our_paper``
        meets ``min_evidence_n`` (issue #144: producer-configurable,
        default ``MIN_EVIDENCE_N``=2). This is the number that gates
        ``rollup.position`` below (see the position-cap section) — a high
        ``score`` cannot rescue a clause with too few observations. The
        threshold actually applied is recorded in ``confidence.basis`` so a
        consumer reading one playbook document (without the producer's
        config) can see what N was enforced.
    """
    n_our_paper = sum(1 for obs in group if obs.provenance == "our_paper")
    n_counterparty_paper = sum(1 for obs in group if obs.provenance == "counterparty_paper")
    total = n_our_paper + n_counterparty_paper
    if total > 0:
        weighted = n_our_paper * 1.0 + n_counterparty_paper * 0.5
        confidence_score = round(weighted / total, 3)
    else:
        confidence_score = 0.0

    confidence: dict[str, Any] = {
        "score": confidence_score,
        "basis": (
            "provenance_mix (quality, not depth); raw counts in "
            "n_our_paper / n_counterparty_paper; evidence_sufficient requires "
            f"n_our_paper >= min_evidence_n={min_evidence_n} "
            "(producer-configurable, see config.provenance.min_evidence_n)"
        ),
        "n_our_paper": n_our_paper,
        "n_counterparty_paper": n_counterparty_paper,
        "evidence_sufficient": n_our_paper >= min_evidence_n,
    }

    # ---------------------------------------------------------------
    # acceptable_if: neutral-risk signed variants (OPF §2.1)
    # Our-paper first (stronger endorsement signal), then counterparty-paper.
    # §2.2 Appendix B: counterparty-paper MAY inform tolerance bounds here.
    # ---------------------------------------------------------------
    neutral_signed = sorted(
        (
            obs
            for obs in group
            if obs.outcome == "signed"
            and obs.risk_delta.get("direction") == "neutral"
            and obs.deviation != "none"
            and obs.basis not in _UNJUDGED_BASES
        ),
        key=lambda o: 0 if o.provenance == "our_paper" else 1,
    )
    # Full clause text (issue #105) — acceptable_if.to is the "acceptable
    # alternative language" lawyers act on directly; a 200-char text_summary
    # fragment is not actually usable drafting language.
    #
    # issue #141: each entry is a {if,to,rationale} triple, not a bare
    # string — `to` is the accepted language (obs.full_text); `if` is the
    # recognizable short-form pattern a reviewer matches a counterparty draft
    # against (obs.text_summary — deliberately distinct from `to` so the
    # entry reads as "if you see something like THIS, THIS full text is what
    # we've accepted before"); `rationale` cites the precedent basis
    # (deviation + precedent_count) that grounds the tolerance; observation_ref
    # is the citation this entry is derived from (OPF §4 — resolvable evidence,
    # not an assertion).
    seen_texts: set[str] = set()
    acceptable_if_list: list[AcceptableIfEntry] = []
    for obs in neutral_signed:
        if obs.full_text not in seen_texts:
            seen_texts.add(obs.full_text)
            precedent_count = 1
            if precedent_counts is not None:
                precedent_count = precedent_counts.get(_normalize_for_dedup(obs.full_text), 1)
            acceptable_if_list.append(
                AcceptableIfEntry(
                    if_=obs.text_summary,
                    to=obs.full_text,
                    rationale=(
                        f"Signed with neutral risk_delta (deviation={obs.deviation}); "
                        f"{precedent_count}x precedent in the corpus."
                    ),
                    observation_ref=OPFCitation(
                        document_id=obs.citation.document_id,
                        version=obs.citation.version,
                        clause_path=obs.citation.clause_path,
                        char_span=obs.citation.char_span,
                    ),
                )
            )

    # ---------------------------------------------------------------
    # fallbacks: worse-risk signed our-paper, ordered least→most costly
    # ---------------------------------------------------------------
    fallback_obs = tuple(
        sorted(
            (
                _obs_to_observed_position(obs, precedent_counts)
                for obs in group
                if obs.provenance == "our_paper"
                and obs.outcome == "signed"
                and obs.risk_delta.get("direction") == "worse"
            ),
            key=lambda op: _MAGNITUDE_ORDER.get(op.risk_delta.get("magnitude", "minor"), 0),
        )
    )

    # ---------------------------------------------------------------
    # rejected: proposed_then_reversed observations
    # ---------------------------------------------------------------
    rejected_obs = tuple(
        _obs_to_observed_position(obs, precedent_counts)
        for obs in group
        if obs.outcome == "proposed_then_reversed"
    )

    # ---------------------------------------------------------------
    # stance_detail — held-rate behind the historical_stance enum
    # (issue #177, OPF §3.5.3, resolves spec Appendix A.3)
    # ---------------------------------------------------------------
    # "Held" mirrors the cascade's own semantics: an observation where our
    # position did NOT give ground — a refusal (proposed_then_reversed) or a
    # signing with no worse-risk shift. "Conceded" is exactly the fallbacks
    # definition above (signed, direction="worse"). Basis is our_paper when
    # any our-paper evidence exists (the §2.2-preferred pool the stance is
    # actually derived from), else the whole group. Outcomes outside the OPF
    # enum (e.g. "unsigned", issue #83) are not opportunities and count in
    # neither number.
    # Basis follows the DEAL evidence, not has_our_paper: has_our_paper is
    # true for template-only grounding (t_obs), which would emit a vacuous
    # "held 0 of 0, basis our_paper" while real counterparty concessions
    # exist in the group — basis "all" over the group is the derivable truth
    # there (review finding, 2026-07-13).
    detail_basis = "our_paper" if our_paper_obs else "all"
    detail_pool = [
        obs
        for obs in (our_paper_obs if detail_basis == "our_paper" else group)
        if obs.outcome in _OPF_OUTCOMES
    ]
    held = sum(
        1
        for obs in detail_pool
        if obs.outcome == "proposed_then_reversed" or obs.risk_delta.get("direction") != "worse"
    )
    stance_detail: dict[str, Any] = {"held": held, "of": len(detail_pool), "basis": detail_basis}

    # ---------------------------------------------------------------
    # §2.2 enforcement — position cap
    # ---------------------------------------------------------------
    has_stub_basis = any(obs.basis in _STUB_BASES for obs in group)
    has_insufficient_evidence = n_our_paper < min_evidence_n
    if not has_our_paper:
        # Provenance rule: counterparty-paper-only → must not exceed "negotiable".
        position = "negotiable"
    elif not has_our_standard:
        # §2.2 (issue #182): a strong position/stance must point at an
        # our_standard clause. our-paper observations exist here but no template
        # clause covers this taxonomy, so there is no standard to hold — cap at
        # negotiable rather than emit a "consistently_held" stance the validator
        # (correctly) rejects for having our_standard: null.
        position = "negotiable"
    elif has_stub_basis:
        # Stub cap: at least one observation in this clause type was never
        # assessed by any judge (no judge configured at all) — the position
        # cannot be trusted beyond "negotiable", regardless of what the other
        # (possibly judged) observations in the group would otherwise derive.
        position = "negotiable"
    elif has_insufficient_evidence:
        # Evidence-depth cap (issue #107): fewer than MIN_EVIDENCE_N our-paper
        # observations is not enough independent evidence to call a clause
        # "standard"/"acceptable_variants_exist"/"hold_firm" — one agreement
        # is one data point, not a pattern, regardless of confidence.score.
        position = "negotiable"
    else:
        position = _derive_position(our_paper_obs, fallback_obs, rejected_obs)

    # Post-condition: structural guarantee that §2.2 was not violated.
    assert not (not has_our_paper and position in _POSITION_STRONGER_THAN_NEGOTIABLE), (
        f"BUG: §2.2 violated — position={position!r} emitted with no our-paper signal"
    )
    assert not (not has_our_standard and position in _POSITION_STRONGER_THAN_NEGOTIABLE), (
        f"BUG: §2.2 violated — position={position!r} emitted with our_standard: null"
    )
    assert not (has_stub_basis and position in _POSITION_STRONGER_THAN_NEGOTIABLE), (
        f"BUG: stub-basis cap violated — position={position!r} emitted from a "
        "clause type with a stub-basis observation"
    )
    assert not (has_insufficient_evidence and position in _POSITION_STRONGER_THAN_NEGOTIABLE), (
        f"BUG: evidence-depth cap violated — position={position!r} emitted with "
        f"n_our_paper={n_our_paper} < min_evidence_n={min_evidence_n}"
    )

    return ClauseRollup(
        position=position,
        acceptable_if=tuple(acceptable_if_list),
        fallbacks=fallback_obs,
        rejected=rejected_obs,
        confidence=confidence,
        stub_basis_present=has_stub_basis,
        stance_detail=stance_detail,
    )


def _derive_position(
    our_paper_obs: list[Observation],
    fallback_obs: tuple[ObservedPosition, ...],
    rejected_obs: tuple[ObservedPosition, ...],
) -> str:
    """Derive position string from our-paper deal signal.

    Note: when ``our_paper_obs`` is empty (only template grounding, no deal
    evidence), we conservatively return ``"negotiable"`` — a template alone
    is not sufficient to assert a strong position without deal confirmation.
    The validator enforces the same constraint (§2.2).
    """
    # Strong positions (> negotiable) require our-paper DEAL signal (§2.2).
    # Counterparty-paper reversals inform tolerance/clause_library only; they
    # must not, together with a template match, manufacture a hold_firm.
    our_paper_rejected = [op for op in rejected_obs if op.provenance == "our_paper"]
    if not our_paper_obs:
        # No our-paper deal evidence (template-only or counterparty-only) — cap.
        return "negotiable"

    if fallback_obs:
        # We have conceded before — negotiable.
        return "negotiable"

    # Check for neutral-risk signed variants whose deviation was actually judged.
    signed_our = [obs for obs in our_paper_obs if obs.outcome == "signed"]
    neutral_variants = [
        obs
        for obs in signed_our
        if obs.risk_delta.get("direction") == "neutral"
        and obs.deviation != "none"
        and obs.basis not in _UNJUDGED_BASES
    ]
    if neutral_variants:
        return "acceptable_variants_exist"

    if our_paper_rejected:
        # We have explicitly rejected asks on our paper with no concessions.
        return "hold_firm"

    # All signed our-paper observations are deviation=none or strictly better-risk.
    # A "better" risk_delta with deviation != "none" still maps to "standard" here —
    # it means we achieved a more favourable variant, not that there are variants to
    # negotiate. Issue #24 may surface these in acceptable_if as positive examples.
    return "standard"


def _title_from_id(taxonomy_id: str) -> str:
    """Convert ``snake_case_id`` → ``Title Case Words``."""
    return " ".join(word.capitalize() for word in taxonomy_id.split("_"))
