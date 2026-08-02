"""Shape-agnostic OPF accessors — issue #154.

Issue #140 migrated ``playbook_assembler`` to emit OPF **v0.2** documents:
clauses moved from a top-level ``clauses`` array to ``evidence.clauses``, and
each clause's prescriptive ``rollup.position`` was replaced by a descriptive
``summary.historical_stance`` (``rollup.confidence`` -> ``summary.confidence``
likewise). Consumers that read the v0.1 shape directly (``aar.py``,
``viewer.py``) silently degraded to empty output against a real compiled
v0.2 playbook — no exception, just zero clauses.

This module is the single place that knows both shapes. Every consumer that
needs a playbook's clauses or a clause's historical stance/confidence MUST
go through these accessors rather than reaching into ``doc["clauses"]`` or
``clause["rollup"]`` directly, so a future OPF version only needs to change
one file.

v0.1 fixtures (hand-authored in existing test suites) continue to work
unchanged — every accessor here falls back to the v0.1 shape when the v0.2
key is absent.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "playbook_clauses",
    "playbook_clause_library",
    "clause_stance",
    "clause_confidence",
    "clause_is_thin",
    "observation_dynamics",
    "clause_trail",
]


def playbook_clauses(doc: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the playbook's clause list, regardless of OPF version.

    OPF v0.2 nests clauses under ``evidence.clauses``; OPF v0.1 (and hand
    -authored test fixtures) keep them at the top-level ``clauses`` key.
    v0.2 takes precedence when ``evidence`` is present as a dict — a v0.2
    document never carries a top-level ``clauses`` key (see
    ``playbook_assembler.assemble_playbook``), so there is no ambiguity in
    practice.

    Args:
        doc: A parsed ``playbook.opf.json`` dict (either OPF version).

    Returns:
        The clause list, or ``[]`` if neither shape is present.
    """
    evidence = doc.get("evidence")
    if isinstance(evidence, dict) and "clauses" in evidence:
        clauses = evidence.get("clauses")
        return clauses if isinstance(clauses, list) else []

    clauses = doc.get("clauses")
    return clauses if isinstance(clauses, list) else []


def playbook_clause_library(doc: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the playbook's clause-concept library, regardless of OPF version.

    Mirrors :func:`playbook_clauses`: OPF v0.2 nests the library under
    ``evidence.clause_library``; OPF v0.1 (and hand-authored test fixtures)
    keep it at the top-level ``clause_library`` key. v0.2 takes precedence
    when ``evidence`` is present as a dict (issue #188 — the publish
    transform and export-profile residue sampling both need version-agnostic
    access to ``ClauseConcept.description``/``notes``).

    Args:
        doc: A parsed ``playbook.opf.json`` dict (either OPF version).

    Returns:
        The clause-concept list, or ``[]`` if neither shape is present.
    """
    evidence = doc.get("evidence")
    if isinstance(evidence, dict) and "clause_library" in evidence:
        library = evidence.get("clause_library")
        return library if isinstance(library, list) else []

    library = doc.get("clause_library")
    return library if isinstance(library, list) else []


def clause_stance(clause: dict[str, Any]) -> str:
    """Return one clause's historical stance / rollup position, version-agnostic.

    OPF v0.2 carries this as ``summary.historical_stance`` (descriptive: "what
    has the corpus shown"); OPF v0.1 carried it as ``rollup.position``
    (prescriptive). v0.2 takes precedence when ``summary`` is present as a
    dict.

    Args:
        clause: One clause dict from ``playbook_clauses()``.

    Returns:
        The stance/position string, or ``"unknown"`` if neither shape is
        present.
    """
    summary = clause.get("summary")
    if isinstance(summary, dict) and "historical_stance" in summary:
        return str(summary.get("historical_stance") or "unknown")

    rollup = clause.get("rollup")
    if isinstance(rollup, dict):
        return str(rollup.get("position") or "unknown")

    return "unknown"


def clause_confidence(clause: dict[str, Any]) -> dict[str, Any]:
    """Return one clause's confidence block, version-agnostic.

    OPF v0.2 carries this as ``summary.confidence``; OPF v0.1 carried it as
    ``rollup.confidence``. Both shapes carry the same inner keys (``score``,
    ``n_our_paper``, ``n_counterparty_paper``, ``evidence_sufficient``,
    ...) — only the wrapper key changed.

    Args:
        clause: One clause dict from ``playbook_clauses()``.

    Returns:
        The confidence dict, or ``{}`` if neither shape is present.
    """
    summary = clause.get("summary")
    if isinstance(summary, dict) and "confidence" in summary:
        confidence = summary.get("confidence")
        return confidence if isinstance(confidence, dict) else {}

    rollup = clause.get("rollup")
    if isinstance(rollup, dict):
        confidence = rollup.get("confidence")
        return confidence if isinstance(confidence, dict) else {}

    return {}


def clause_is_thin(clause: dict[str, Any]) -> bool:
    """Whether one clause's evidence is "thin" — the shared trigger issue
    #91 (review-HTML attention sort) and issue #92 (prompt-renderer heading
    marker) both key off.

    ``True`` when ``confidence.evidence_sufficient`` is explicitly
    ``False``, OR every ``observed_positions`` entry on record has
    ``precedent_count == 1`` (nothing behind this clause has ever recurred
    in the corpus). A clause with NO observed positions at all is not, by
    the second branch alone, "thin" — that case is caught by the first
    branch in any playbook the compiler itself produced, since the
    compiler sets ``evidence_sufficient`` False whenever ``n_our_paper``
    falls short of its configured minimum (``clause_position_compiler.py``).

    Mirrors ``prompt_renderer._thin_marker``'s trigger condition exactly;
    kept here as the version-agnostic, publicly reusable primitive so a
    second consumer (issue #91) never has to re-derive or drift from the
    same definition.

    Args:
        clause: One clause dict from ``playbook_clauses()``.

    Returns:
        ``True`` if the clause's evidence is thin by either trigger.
    """
    confidence = clause_confidence(clause)
    positions = [p for p in (clause.get("observed_positions") or []) if isinstance(p, dict)]
    evidence_insufficient = confidence.get("evidence_sufficient") is False
    single_precedent_only = bool(positions) and all(
        p.get("precedent_count") == 1 for p in positions
    )
    return evidence_insufficient or single_precedent_only


def observation_dynamics(obs: dict[str, Any]) -> dict[str, Any]:
    """Return one observation's negotiation-dynamics fields (issue #177).

    OPF v0.2 §3.5.3 fields are optional-when-underivable, so a v0.2 document
    without dynamics (or any v0.1 observation) simply yields ``{}`` — a key
    appears in the result only when the observation actually carries it.

    Args:
        obs: One entry of a clause's ``observed_positions`` (either OPF
             version).

    Returns:
        Dict with any of ``proposed_by`` / ``observed_at`` /
        ``counterparty_ref`` that are present; ``{}`` otherwise.
    """
    dynamics: dict[str, Any] = {}
    for key in ("proposed_by", "observed_at", "counterparty_ref"):
        value = obs.get(key)
        if value is not None:
            dynamics[key] = value
    return dynamics


def clause_trail(clause: dict[str, Any]) -> list[dict[str, Any]]:
    """Return one clause's ``negotiation_trail`` (issue #177), or ``[]``.

    v0.2 documents compiled before §3.5.3 (and every v0.1 document) carry no
    trail; they read cleanly as an empty list.

    Args:
        clause: One clause dict (either OPF version).

    Returns:
        The trail entry list, or ``[]`` when absent/malformed.
    """
    trail = clause.get("negotiation_trail")
    return trail if isinstance(trail, list) else []
