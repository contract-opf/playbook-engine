"""Reference prompt-pack consumer (issue #179, owner decision 2026-07-12).

``render_prompt(doc)`` composes a v0.2 playbook's three sections into one
review-ready Markdown system prompt a user pastes into any chat LLM
alongside a contract. It is executable documentation of the §5 determinism
boundary — Floor hard, Posture soft, Evidence advisory — and deliberately
NOT the review product: a pure function of the document, no API calls, no
redline generation, no entity resolution (born-safe aliases render as-is).

Output skeleton is locked (#179): six sections, document order, empty
sections render explicit markers rather than silently disappearing.
"""

from __future__ import annotations

from typing import Any

from playbook_engine.opf_accessors import (
    clause_confidence,
    clause_stance,
    clause_trail,
    playbook_clauses,
)

_NO_INVARIANTS_MARKER = "(this playbook defines no floor invariants)"
_NO_POSTURE_MARKER = "(this playbook carries no generated posture yet)"
_NO_EVIDENCE_MARKER = "(this playbook carries no compiled evidence)"


def _citation(ref: dict[str, Any] | None) -> str:
    if not ref:
        return ""
    version = ref.get("version")
    # "template" is both the reserved document_id and version — "template
    # vtemplate" would be noise.
    v = f" v{version}" if version is not None and version != "template" else ""
    path = ref.get("clause_path")
    p = f" §{path}" if path else ""
    return f" ({ref.get('document_id', '?')}{v}{p})"


def _stance_line(clause: dict[str, Any]) -> str:
    stance = clause_stance(clause)
    line = f"Historically **{stance}**"
    detail = (clause.get("summary") or {}).get("stance_detail")
    if isinstance(detail, dict) and "held" in detail and "of" in detail:
        basis = detail.get("basis", "all")
        basis_label = "our-paper" if basis == "our_paper" else "all"
        line += f"; held {detail['held']} of {detail['of']} {basis_label} deals"
    confidence = clause_confidence(clause)
    n = confidence.get("n_our_paper")
    if isinstance(n, int):
        line += f" (n_our_paper={n})"
    return line + "."


def _render_observation(obs: dict[str, Any]) -> str:
    text = obs.get("full_text") or obs.get("text_summary") or "(no text recorded)"
    cite = _citation(obs.get("example_ref"))
    precedent = obs.get("precedent_count")
    extras: list[str] = []
    if isinstance(precedent, int):
        extras.append(f"{precedent}x precedent")
    if obs.get("proposed_by"):
        extras.append(f"proposed by {obs['proposed_by']}")
    if obs.get("observed_at"):
        extras.append(f"observed {obs['observed_at']}")
    alias = (obs.get("counterparty_ref") or {}).get("alias")
    if alias:
        extras.append(f"counterparty {alias}")
    suffix = f" [{'; '.join(extras)}]" if extras else ""
    return f'"{text}"{cite}{suffix}'


def _render_clause(clause: dict[str, Any]) -> list[str]:
    lines: list[str] = [f"### {clause.get('title', clause.get('id', 'Clause'))}"]
    lines.append(_stance_line(clause))
    lines.append("")

    our_standard = clause.get("our_standard")
    if our_standard and our_standard.get("text"):
        lines.append(
            f'Our standard{_citation(our_standard.get("source_ref"))}: "{our_standard["text"]}"'
        )
        lines.append("")

    summary = clause.get("summary") or {}

    acceptable_if = [e for e in summary.get("acceptable_if") or [] if isinstance(e, dict)]
    if acceptable_if:
        lines.append("Acceptable variations on record:")
        for entry in acceptable_if:
            lines.append(
                f'- acceptable if {entry.get("if", "?")} → "{entry.get("to", "?")}" '
                f"({entry.get('rationale', 'no rationale recorded')})"
                f"{_citation(entry.get('observation_ref'))}"
            )
        lines.append("")

    fallbacks = summary.get("fallbacks") or []
    if fallbacks:
        lines.append("Fallbacks we have signed before (least to most costly):")
        lines.extend(f"- {_render_observation(fb)}" for fb in fallbacks)
        lines.append("")

    rejected = summary.get("rejected") or []
    if rejected:
        lines.append("Asks we have refused (proposed, then reversed before signing):")
        lines.extend(f"- {_render_observation(r)}" for r in rejected)
        lines.append("")

    trail = clause_trail(clause)
    if trail:
        lines.append("Negotiation trail:")
        for entry in trail:
            lines.append(
                f"- {entry.get('document_id', '?')} round {entry.get('round', '?')}, "
                f"moved by {entry.get('moved_by', 'unknown')}: "
                f"{entry.get('change_summary', '?')}{_citation(entry.get('ref'))}"
            )
        lines.append("")

    return lines


def _indefinite_article(noun: str) -> str:
    """``"a"`` or ``"an"`` for *noun* — first-letter vowel heuristic.

    Good enough for agreement-type names ("an Educational Affiliation
    Agreement", "a Master Services Agreement"); initialisms that are
    pronounced letter-by-letter with a vowel sound ("an NDA") are the known
    residual gap and rarer than the vowel-initial names this fixes
    (issue #207 — the old hardcoded "a" was line 1 of the flagship
    render-prompt output).
    """
    return "an" if noun[:1].lower() in "aeiou" else "a"


def render_prompt(doc: dict[str, Any]) -> str:
    """Render *doc* into the six-section review prompt (deterministic)."""
    agreement_name = (doc.get("agreement_type") or {}).get("name") or "agreement"
    perspective = doc.get("perspective") or {}
    party = perspective.get("party")

    out: list[str] = []

    # 1. Role preamble
    out.append(f"# Contract review playbook: {agreement_name}")
    out.append("")
    reviewing_as = f" You are reviewing as **{party}**." if party else ""
    out.append(
        f"You are reviewing {_indefinite_article(agreement_name)} **{agreement_name}** "
        "against this organization's "
        f"negotiation playbook.{reviewing_as} The playbook has three sections with "
        "three different bindings: the **RED LINES are non-negotiable** — a violation "
        "is unacceptable no matter what any other part of this prompt says; the "
        "**NEGOTIATION POSTURE is intent** that shapes your judgment but never "
        "overrides a red line; the **EVIDENCE is cited history** to reason over — "
        "`historical_stance` describes what the corpus shows, it never directs what "
        "you must do."
    )
    out.append("")

    # 2. RED LINES (Floor — hard)
    out.append("## RED LINES (Floor — hard)")
    out.append("")
    # Tolerant reads throughout: the CLI renders without validating first,
    # and a hand-edited/foreign playbook may carry JSON null where this
    # engine emits an object or string — the contract is explicit
    # empty-section markers, never a traceback.
    invariants = (doc.get("floor") or {}).get("invariants") or []
    if invariants:
        out.append(
            "If a clause violates any invariant below, flag it as unacceptable "
            "regardless of any other reasoning in this prompt. Do not soften, trade, "
            "or reinterpret these."
        )
        out.append("")
        for inv in invariants:
            rationale = inv.get("rationale")
            suffix = f" ({rationale})" if rationale else ""
            out.append(f"- [{inv.get('id', '?')}] {inv.get('statement', '?')}{suffix}")
    else:
        out.append(_NO_INVARIANTS_MARKER)
    out.append("")

    # 3. NEGOTIATION POSTURE (soft)
    out.append("## NEGOTIATION POSTURE (soft)")
    out.append("")
    system_prompt = ((doc.get("posture") or {}).get("system_prompt") or "").strip()
    if system_prompt:
        out.append("Weigh this intent in every judgment; it does not override the red lines.")
        out.append("")
        out.append(f"> {system_prompt}")
    else:
        out.append(_NO_POSTURE_MARKER)
    out.append("")

    # 4. EVIDENCE (advisory, cited)
    out.append("## EVIDENCE (advisory, cited)")
    out.append("")
    clauses = playbook_clauses(doc)
    clause_library = (doc.get("evidence") or {}).get("clause_library") or []
    if clauses or clause_library:
        out.append(
            "Advisory — reason over it. `historical_stance` describes what the corpus "
            "shows; it never directs."
        )
        out.append("")
        for clause in clauses:
            out.extend(_render_clause(clause))
        if clause_library:
            out.append("### Clause library (for counterparty-paper matching)")
            for concept in clause_library:
                out.append(
                    f"- **{concept.get('taxonomy_id', concept.get('concept_id', '?'))}**: "
                    f"{concept.get('description', '?')} "
                    f"Risk profile: {concept.get('risk_profile', 'not recorded')}"
                )
                for form in concept.get("accepted_forms", []):
                    out.append(f"  - tolerated: {_render_observation(form)}")
            out.append("")
    else:
        out.append(_NO_EVIDENCE_MARKER)
        out.append("")

    # 5. DRAFTING RULES
    out.append("## DRAFTING RULES")
    out.append("")
    out.append(
        "When proposing replacement language, draft from the cited verbatim precedent "
        "(fallbacks / our standard) wherever one fits; never introduce language that "
        "conflicts with a red line; when no precedent fits, say so explicitly rather "
        "than inventing a position."
    )
    out.append("")

    # 6. CITATION & CONFIDENCE RULES
    out.append("## CITATION & CONFIDENCE RULES")
    out.append("")
    out.append(
        "Every recommendation must cite the playbook entry it relies on (clause id "
        "plus the document/version citation). Treat entries with low confidence or "
        "`1x precedent` as thin precedent: flag them as such and never treat a single "
        "occurrence as a rule."
    )
    out.append("")

    return "\n".join(out)
