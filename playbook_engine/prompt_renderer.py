"""Reference prompt-pack consumer (issue #179, owner decision 2026-07-12).

``render_prompt(doc)`` composes a playbook's three sections into one
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

from playbook_engine.digest import clause_precedent_groups
from playbook_engine.opf_accessors import (
    perspective_party,
    playbook_clauses,
    playbook_precedent,
)

_NO_INVARIANTS_MARKER = (
    "(no hard lines defined — author them via 'playbook posture interview' "
    "(sacred-clauses question) or review proposals via 'playbook floor propose')"
)
_NO_POSTURE_MARKER = "(no posture yet — run 'playbook posture interview')"
_NO_EVIDENCE_MARKER = "(this playbook carries no compiled evidence)"

# Loud, hard-to-miss block prepended by render_prompt() when the playbook is
# advisory-only (issue #92) — both Floor and Posture are empty, so nothing in
# the rendered prompt is binding. Kept as one module-level constant (not
# assembled inline) so a later wording pass is a one-file, one-string edit.
_ADVISORY_BANNER = (
    "> **ADVISORY ONLY — NOTHING BELOW IS BINDING.**\n"
    ">\n"
    "> This playbook defines no hard lines and carries no negotiation posture "
    "yet — every section below is historical evidence to reason over, not an "
    "instruction to follow.\n"
    ">\n"
    "> To make part of this playbook binding: run `playbook posture "
    "interview` to add a negotiation posture, or `playbook floor propose` "
    "and have your reviewer sign off on the proposed invariants to add hard "
    "lines."
)


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


def _deal_count(n: Any) -> str:
    n = n if isinstance(n, int) else 0
    return f"{n} deal{'s' if n != 1 else ''}"


def _render_clause(
    clause: dict[str, Any], precedent: list[dict[str, Any]], *, party: str | None
) -> list[str]:
    """One clause (issue #223): facts only — no stance, no verdict.

    Our standard and how many deals signed it, then every non-standard
    variant signed and every ask refused before signing, each with its
    distinct-deal count and a citation (``digest.clause_precedent_groups``,
    uncapped). The reviewing model does the judging.
    """
    title = clause.get("title", clause.get("id", "Clause"))
    lines: list[str] = [f"### {title}"]
    n_deals = clause.get("n_deals")
    our_standard = clause.get("our_standard")
    if isinstance(our_standard, dict) and our_standard.get("text"):
        lines.append(
            f"{clause.get('n_signed_standard', 0)} of {_deal_count(n_deals)} on record "
            "signed our standard language."
        )
        lines.append("")
        lines.append(
            f'Our standard{_citation(our_standard.get("source_ref"))}: "{our_standard["text"]}"'
        )
    else:
        lines.append(f"No standard language on record; {_deal_count(n_deals)} on record.")
    lines.append("")

    groups = clause_precedent_groups(clause.get("taxonomy_id"), precedent, party=party)
    if groups["signed_variants"]:
        lines.append("Non-standard language we have signed:")
        for v in groups["signed_variants"]:
            lines.append(f'- "{v["text"]}"{_citation(v.get("ref"))} [{_deal_count(v["n_deals"])}]')
        lines.append("")
    if groups["refused_asks"]:
        lines.append("Asks refused before signing (proposed, then struck):")
        for a in groups["refused_asks"]:
            lines.append(f'- "{a["text"]}"{_citation(a.get("ref"))} [{_deal_count(a["n_deals"])}]')
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


def is_advisory_only(doc: dict[str, Any]) -> bool:
    """True when *doc* carries neither Floor invariants nor a Posture brief.

    Single source of truth for "advisory-only" (issue #92), shared between
    :func:`render_prompt` (prepends :data:`_ADVISORY_BANNER`) and the CLI's
    ``render-prompt`` command (emits a one-line stderr WARN) — both must
    agree on the same definition rather than drifting apart.
    """
    invariants = (doc.get("floor") or {}).get("invariants") or []
    system_prompt = ((doc.get("posture") or {}).get("system_prompt") or "").strip()
    return not invariants and not system_prompt


def render_prompt(doc: dict[str, Any]) -> str:
    """Render *doc* into the six-section review prompt (deterministic)."""
    agreement_name = (doc.get("agreement_type") or {}).get("name") or "agreement"
    perspective = doc.get("perspective") or {}
    party = perspective.get("party")

    out: list[str] = []

    if is_advisory_only(doc):
        out.append(_ADVISORY_BANNER)
        out.append("")

    # 1. Role preamble
    out.append(f"# Contract review playbook: {agreement_name}")
    out.append("")
    reviewing_as = f" You are reviewing as **{party}**." if party else ""
    out.append(
        f"You are reviewing {_indefinite_article(agreement_name)} **{agreement_name}** "
        "against this organization's "
        f"negotiation playbook.{reviewing_as} The playbook has three sections with "
        "three different bindings: the **HARD LINES are non-negotiable** — a violation "
        "is unacceptable no matter what any other part of this prompt says; the "
        "**NEGOTIATION POSTURE is intent** that shapes your judgment but never "
        "overrides a hard line; the **EVIDENCE is cited history** to reason over — "
        "it describes what the corpus has shown, never what you must do."
    )
    out.append("")

    # 2. HARD LINES (Floor)
    out.append("## HARD LINES (Floor)")
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
            if isinstance(inv, dict):
                rationale = inv.get("rationale")
                suffix = f" ({rationale})" if rationale else ""
                out.append(f"- [{inv.get('id', '?')}] {inv.get('statement', '?')}{suffix}")
            else:
                out.append(f"- {inv}")
    else:
        out.append(_NO_INVARIANTS_MARKER)
    out.append("")

    # 3. NEGOTIATION POSTURE (soft)
    out.append("## NEGOTIATION POSTURE (soft)")
    out.append("")
    system_prompt = ((doc.get("posture") or {}).get("system_prompt") or "").strip()
    if system_prompt:
        out.append("Weigh this intent in every judgment; it does not override the hard lines.")
        out.append("")
        out.append(f"> {system_prompt}")
    else:
        out.append(_NO_POSTURE_MARKER)
    out.append("")

    # 4. EVIDENCE (advisory, cited)
    out.append("## EVIDENCE (advisory, cited)")
    out.append("")
    clauses = playbook_clauses(doc)
    if clauses:
        out.append(
            "Advisory — reason over it. Each entry describes what the corpus has "
            "shown; it never directs what you must do."
        )
        out.append("")
        precedent = playbook_precedent(doc)
        party = perspective_party(doc)
        for clause in clauses:
            out.extend(_render_clause(clause, precedent, party=party))
    else:
        out.append(_NO_EVIDENCE_MARKER)
        out.append("")

    # 5. DRAFTING RULES
    out.append("## DRAFTING RULES")
    out.append("")
    out.append(
        "When proposing replacement language, draft from the cited verbatim precedent "
        "(signed language / our standard) wherever one fits; never introduce language that "
        "conflicts with a hard line; when no precedent fits, say so explicitly rather "
        "than inventing a position."
    )
    out.append("")

    # 6. CITATION & CONFIDENCE RULES
    out.append("## CITATION & CONFIDENCE RULES")
    out.append("")
    out.append(
        "Every recommendation must cite the playbook entry it relies on (clause id "
        "plus the document/version citation). An entry backed by `1 deal` is a single "
        "occurrence — flag any recommendation drawn from it as such, and never treat "
        "a single occurrence as a rule."
    )
    out.append("")

    return "\n".join(out)
