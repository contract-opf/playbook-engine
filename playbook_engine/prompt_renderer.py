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

# Inline heading marker for a clause resting on thin evidence (issue #92) —
# named identically wherever it appears (the clause heading and the CITATION
# & CONFIDENCE RULES section) so the model can actually correlate the two.
_THIN_TERM = "THIN PRECEDENT"

# Plain-language stance sentences (issue #92) — one per OPF v0.2
# historical_stance enum value except "no_signal" (handled separately in
# _stance_line: it describes an absence of pattern, not a pattern, so it
# earns its own construction rather than forcing a "held/of" framing onto
# zero signal). JSON enum values themselves are unchanged (OPF-SPEC.md) —
# this dict only controls rendering.
_STANCE_SENTENCES: dict[str, str] = {
    "consistently_held": "We have consistently held this position",
    "usually_held": "We have usually held this position",
    "mixed": "Our history on this clause is mixed — we have both held and conceded it",
    "usually_conceded": "We have usually conceded this position when it was contested",
}

# Plain-language sentences for OPF v0.1's prescriptive `rollup.position`
# enum (issue #92 fix round 2 — spec/playbook.schema.json
# /$defs/clausePosition/properties/rollup/properties/position). clause_stance()
# (opf_accessors.py) falls back to this literal value for a v0.1-shaped
# clause (no "summary.historical_stance" key) — that value reaches
# _stance_line() exactly like a v0.2 historical_stance does, and both
# shipped examples/our-paper-baseline.playbook.json and
# examples/emergent-no-template.playbook.json are `playbook validate`-
# passing v0.1 documents that hit this path, so it needs its own prose
# rather than falling through to the "could not be determined" fallback.
# Wording follows the position semantics documented in
# clause_position_compiler.py's module docstring (the derivation these
# values come from in this engine's own output) — kept local rather than
# imported, since this table only needs the four enum strings, not the
# derivation logic.
_V01_POSITION_SENTENCES: dict[str, str] = {
    "standard": (
        "This is our standard position — every signed deal on record has matched it "
        "without deviation"
    ),
    "acceptable_variants_exist": (
        "This is our standard position, though neutral-risk variations have been accepted before"
    ),
    "negotiable": "This position has historically been treated as negotiable",
    "hold_firm": "We have held firm on this position whenever it was contested",
}

# Reserved for clause_stance()'s literal "unknown" sentinel (opf_accessors.py)
# — returned ONLY when a clause carries neither a v0.2
# `summary.historical_stance` nor a v0.1 `rollup.position` (or the key that
# is present is empty/falsy), i.e. no stance was recorded at all. A stance
# string that IS recorded but isn't a key in _STANCE_SENTENCES or
# _V01_POSITION_SENTENCES above (e.g. a future enum addition) must still
# surface that recorded value rather than claim it is undeterminable —
# see the residual branch in _stance_line() (issue #92 fix round 2).
_UNKNOWN_STANCE_SENTENCE = "This clause's historical stance could not be determined"
_NO_SIGNAL_SENTENCE = "Not enough history to establish a stance"

# Renders in place of a negotiation_trail `moved_by` that isn't `"us"` or
# `"counterparty"` (issue #92 jargon strip — was "moved by unknown"). Both
# OPF schemas make `moved_by` REQUIRED with enum ["us", "counterparty",
# "unknown"], and the engine writes the literal string "unknown" whenever
# tracked-changes attribution fails (observation_builder.build_round_moves) —
# that schema-valid "unknown" is the real-world case this covers, not just a
# missing/empty key from a hand-edited or foreign playbook (tolerated too).
# Deliberately articleless, matching the other values in this slot ("us",
# "counterparty") rather than reading as a full sentence fragment.
_UNRECORDED_MOVER = "unrecorded party"


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


def _no_signal_detail(clause: dict[str, Any]) -> str:
    """Occurrence-count + provenance phrase for a ``no_signal`` stance.

    e.g. ``"1 occurrence, counterparty paper only"`` — conveys the same two
    numbers the raw ``n_our_paper``/``n_counterparty_paper`` fields carry,
    in prose, without ever printing either field name on the surface
    (issue #92 jargon strip).
    """
    confidence = clause_confidence(clause)
    n_our = confidence.get("n_our_paper")
    n_cp = confidence.get("n_counterparty_paper")

    total: int | None = None
    if isinstance(n_our, int) and isinstance(n_cp, int):
        total = n_our + n_cp
    else:
        detail = (clause.get("summary") or {}).get("stance_detail")
        if isinstance(detail, dict) and isinstance(detail.get("of"), int):
            total = detail["of"]

    count_phrase = (
        f"{total} occurrence{'s' if total != 1 else ''}" if total else ("no recorded occurrences")
    )

    if isinstance(n_our, int) and isinstance(n_cp, int):
        if n_our == 0 and n_cp > 0:
            provenance_phrase = "counterparty paper only"
        elif n_our > 0 and n_cp == 0:
            provenance_phrase = "our paper only"
        elif n_our > 0 and n_cp > 0:
            provenance_phrase = "our paper and counterparty paper"
        else:
            provenance_phrase = "no provenance recorded"
    else:
        provenance_phrase = "provenance not recorded"

    return f"{count_phrase}, {provenance_phrase}"


def _stance_line(clause: dict[str, Any]) -> str:
    stance = clause_stance(clause)

    if stance == "no_signal":
        return f"{_NO_SIGNAL_SENTENCE} ({_no_signal_detail(clause)})."

    if stance == "unknown":
        # clause_stance()'s sentinel: neither shape recorded a stance at all.
        sentence = _UNKNOWN_STANCE_SENTENCE
    elif stance in _STANCE_SENTENCES:
        sentence = _STANCE_SENTENCES[stance]
    elif stance in _V01_POSITION_SENTENCES:
        sentence = _V01_POSITION_SENTENCES[stance]
    else:
        # A stance IS recorded but isn't one of the enums above (issue #92
        # fix round 2) — surface it rather than claiming it's
        # undeterminable; e.g. a future OPF enum value, or a foreign/hand
        # -edited playbook this renderer reads tolerantly regardless.
        sentence = f"Recorded stance: {stance}"
    detail = (clause.get("summary") or {}).get("stance_detail")
    if isinstance(detail, dict) and "held" in detail and "of" in detail:
        basis = detail.get("basis", "all")
        basis_label = "our-paper" if basis == "our_paper" else "all"
        sentence += f" (held {detail['held']} of {detail['of']} {basis_label} deals)"
    return sentence + "."


def _thin_marker(clause: dict[str, Any]) -> str:
    """Inline heading marker for a clause resting on thin evidence (issue #92).

    Triggers when ``summary.confidence.evidence_sufficient`` is explicitly
    ``False``, OR every observed position on record has
    ``precedent_count == 1`` (nothing behind this clause has ever recurred
    in the corpus). The CITATION & CONFIDENCE RULES section names the same
    ``THIN PRECEDENT`` term so the model can actually locate what that rule
    is talking about.

    Returns ``""`` when the clause is not thin (appended directly onto the
    heading line, so the empty string is a no-op suffix).

    The parenthetical names the evidence that actually triggered the
    marker rather than counting ``observed_positions`` entries (issue #92
    fix round 1 — occurrences live in each position's ``precedent_count``,
    not in how many position entries exist): "single occurrence" only when
    the sole observed position was itself seen once (``precedent_count ==
    1``); otherwise "low confidence" whenever ``evidence_sufficient`` is
    the trigger (covers a lone high-precedent-but-insufficient position and
    the no-positions-recorded case alike); "thin evidence" as the residual
    fallback (e.g. several positions that have each individually never
    recurred, so no single position can be named as "the" occurrence).
    """
    confidence = clause_confidence(clause)
    positions = [p for p in (clause.get("observed_positions") or []) if isinstance(p, dict)]
    evidence_insufficient = confidence.get("evidence_sufficient") is False
    single_precedent_only = bool(positions) and all(
        p.get("precedent_count") == 1 for p in positions
    )
    if not (evidence_insufficient or single_precedent_only):
        return ""
    if len(positions) == 1 and positions[0].get("precedent_count") == 1:
        detail = "single occurrence"
    elif evidence_insufficient:
        detail = "low confidence"
    else:
        detail = "thin evidence"
    return f" — {_THIN_TERM} ({detail})"


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
    title = clause.get("title", clause.get("id", "Clause"))
    lines: list[str] = [f"### {title}{_thin_marker(clause)}"]
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
            # "us" / "counterparty" render as-is; anything else — the
            # schema-valid literal "unknown", an empty string, or a missing
            # key — renders as _UNRECORDED_MOVER (issue #92 fix round 1).
            raw_moved_by = entry.get("moved_by")
            moved_by = raw_moved_by if raw_moved_by in ("us", "counterparty") else _UNRECORDED_MOVER
            lines.append(
                f"- {entry.get('document_id', '?')} round {entry.get('round', '?')}, "
                f"moved by {moved_by}: "
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
    clause_library = (doc.get("evidence") or {}).get("clause_library") or []
    if clauses or clause_library:
        out.append(
            "Advisory — reason over it. Each entry describes what the corpus has "
            "shown; it never directs what you must do."
        )
        out.append("")
        for clause in clauses:
            out.extend(_render_clause(clause))
        if clause_library:
            out.append("### Clause library (for counterparty-paper matching)")
            for concept in clause_library:
                line = (
                    f"- **{concept.get('taxonomy_id', concept.get('concept_id', '?'))}**: "
                    f"{concept.get('description', '?')}"
                )
                risk_profile = concept.get("risk_profile")
                if risk_profile:
                    line += f" Risk profile: {risk_profile}"
                out.append(line)
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
        "conflicts with a hard line; when no precedent fits, say so explicitly rather "
        "than inventing a position."
    )
    out.append("")

    # 6. CITATION & CONFIDENCE RULES
    out.append("## CITATION & CONFIDENCE RULES")
    out.append("")
    out.append(
        "Every recommendation must cite the playbook entry it relies on (clause id "
        f"plus the document/version citation). A clause heading marked **{_THIN_TERM}** "
        "rests on low confidence or a single occurrence — flag any recommendation "
        "drawn from it as such, and never treat a single occurrence as a rule. The "
        "same caution applies to any individual entry marked `1x precedent`, even "
        "under a clause heading with stronger overall evidence."
    )
    out.append("")

    return "\n".join(out)
