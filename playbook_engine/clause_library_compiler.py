"""Clause-library compiler — L5 pipeline stage.

Builds the concept-indexed clause library (OPF §3.5) from signed observations.
The library is used to redline counterparty paper when there is no baseline to
diff against: instead of detecting deviations from our template, a reviewer
looks up observed accepted forms by concept and risk profile.

Only signed observations are included (``outcome="signed"``).  Proposed-
then-reversed forms are never acceptable, so they do not belong here.

Counterparty-paper observations are the primary input: the library captures
clauses we tolerated in deals where the other side drafted.  Our-paper signed
observations may also appear if they carry useful accepted-form signal.

Grouping: one ``ClauseConcept`` per ``taxonomy_id``.  Observations with
``taxonomy_id=None`` (unclassified) are excluded from ``accepted_forms`` —
they cannot be anchored to a concept — but they are never silently dropped
(issue #113): ``compile_clause_library`` also returns an
``UnclassifiedCoverage`` summary (count, per-document breakdown, example
citations).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from playbook_engine.clause_position_compiler import (
    OPFCitation,
    UnclassifiedCoverage,
    compute_unclassified_coverage,
)
from playbook_engine.observation_builder import Observation

# ---------------------------------------------------------------------------
# OPF output types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AcceptedForm:
    """One accepted clause variant in the concept library (OPF §3.5)."""

    text_summary: str
    example_ref: OPFCitation
    provenance: str  # "our_paper" | "counterparty_paper"
    risk_delta_vs_our_standard: dict[str, str] | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "text_summary": self.text_summary,
            "example_ref": self.example_ref.to_dict(),
            "provenance": self.provenance,
        }
        if self.risk_delta_vs_our_standard is not None:
            d["risk_delta_vs_our_standard"] = self.risk_delta_vs_our_standard
        return d


@dataclass(frozen=True)
class ClauseConcept:
    """Concept-indexed clause entry for redlining counterparty paper (OPF §3.5)."""

    concept_id: str  # "concept.<taxonomy_id>"
    taxonomy_id: str
    description: str
    accepted_forms: tuple[AcceptedForm, ...]
    risk_profile: str | None = None
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "concept_id": self.concept_id,
            "taxonomy_id": self.taxonomy_id,
            "description": self.description,
            "accepted_forms": [af.to_dict() for af in self.accepted_forms],
        }
        if self.risk_profile is not None:
            d["risk_profile"] = self.risk_profile
        if self.notes is not None:
            d["notes"] = self.notes
        return d


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compile_clause_library(
    observations: list[Observation],
    taxonomy_metadata: dict[str, dict[str, str]] | None = None,
) -> tuple[list[ClauseConcept], UnclassifiedCoverage]:
    """Build the concept-indexed clause library from signed observations.

    Args:
        observations:      All L4 observations (any provenance, any outcome).
                          Proposed-then-reversed and unclassified entries are
                          excluded from ``accepted_forms``.
        taxonomy_metadata: Optional ``{taxonomy_id: {"description": ...,
                          "risk_profile": ...}}`` override.  Falls back to
                          ``_description_from_id()`` / ``None`` when absent.

    Returns:
        Tuple of (concepts, unclassified_coverage):
          - ``concepts``: One ``ClauseConcept`` per distinct non-None
            ``taxonomy_id`` that has at least one signed observation, in
            taxonomy_id sorted order.
          - ``unclassified_coverage``: Summary (count, per-document
            breakdown, example citations) of ``observations`` entries with
            ``taxonomy_id=None`` (issue #113) — excluded from ``concepts``
            but never silently dropped from the return value.
    """
    unclassified_coverage = compute_unclassified_coverage(observations)

    # Filter to signed observations only; skip unclassified.
    signed: list[Observation] = [
        obs for obs in observations if obs.taxonomy_id is not None and obs.outcome == "signed"
    ]

    # Group by taxonomy_id.  taxonomy_id is guaranteed non-None by the filter above.
    groups: dict[str, list[Observation]] = {}
    for obs in signed:
        tid_key: str = obs.taxonomy_id  # type: ignore[assignment]
        groups.setdefault(tid_key, []).append(obs)

    meta = taxonomy_metadata or {}
    concepts: list[ClauseConcept] = []

    for tid in sorted(groups.keys()):
        group = groups[tid]
        tid_meta = meta.get(tid, {})

        description = tid_meta.get("description") or _description_from_id(tid)
        risk_profile: str | None = tid_meta.get("risk_profile") or None

        accepted_forms = tuple(_obs_to_accepted_form(obs) for obs in group)

        n_cp = sum(1 for obs in group if obs.provenance == "counterparty_paper")
        notes: str | None = (
            f"Accepted in {n_cp} signed counterparty-paper observation(s)." if n_cp > 0 else None
        )

        concepts.append(
            ClauseConcept(
                concept_id=f"concept.{tid}",
                taxonomy_id=tid,
                description=description,
                accepted_forms=accepted_forms,
                risk_profile=risk_profile,
                notes=notes,
            )
        )

    return concepts, unclassified_coverage


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _obs_to_accepted_form(obs: Observation) -> AcceptedForm:
    return AcceptedForm(
        text_summary=obs.text_summary,
        example_ref=OPFCitation(
            document_id=obs.citation.document_id,
            version=obs.citation.version,
            clause_path=obs.citation.clause_path,
            char_span=obs.citation.char_span,
        ),
        provenance=obs.provenance,
        risk_delta_vs_our_standard=obs.risk_delta,
    )


def _description_from_id(taxonomy_id: str) -> str:
    """Convert ``snake_case_id`` → sentence-style description placeholder."""
    words = " ".join(word.lower() for word in taxonomy_id.split("_"))
    return f"Clause governing {words}."
