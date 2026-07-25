"""Floor-candidate proposal — derive review candidates from reversals + the
Posture interview's Q4 answer (issue #166).

OPF-SPEC.md §3.7 rule 4: the compiler MAY propose Floor candidates
("every ``outcome: proposed_then_reversed`` in the Evidence is a candidate red
line") and §7 marks the interview's Q4 ("sacred_clauses") as seeding Floor
candidates (see ``posture.py``'s ``seeds_floor_candidates=True``) — but the
legal owner finalizes, and a proposal must NEVER be auto-promoted into the
signed OPF ``floor.invariants`` (spec rule 4, "never auto-promote").

This module implements PROPOSAL only:

  - :func:`derive_reversal_candidates` — one candidate per distinct reversed
    concept (grouped by ``taxonomy_id``, falling back to per-observation
    grouping when unclassified), citing every reversal observation that
    contributed to it.
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
    written into the OPF document itself — the OPF ``floor`` section stays
    authored-and-signed only. Accepting a candidate is a human act: editing
    ``floor.invariants`` directly, or via the curation CLI. This module
    never writes to ``floor.invariants``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from playbook_engine.observation_builder import read_observations_jsonl
from playbook_engine.validator import load_opf_file

__all__ = [
    "FloorCandidate",
    "FloorCandidateCitation",
    "derive_interview_q4_candidates",
    "derive_reversal_candidates",
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
    answer = interview_answers.get(INTERVIEW_Q4_ID)
    if not answer or not answer.strip():
        return []

    items = [item.strip() for item in answer.split(";") if item.strip()]

    return [
        FloorCandidate(
            id="",  # assigned by propose_floor_candidates
            statement=f"Never accept {item}.",
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
