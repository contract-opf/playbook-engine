"""Provenance detector — L2 structure layer.

Decides whether a document is drafted on *our paper* (our standard form / our
template) or *counterparty paper* (the counterparty's form).  This feeds the
provenance rule in the OPF: only our-paper observations may define an opening
position; counterparty-paper observations inform tolerance bounds only.

Detection signals (applied in priority order):
1. **Template similarity** (highest fidelity) — if a canonical template is
   supplied, compute the normalised edit-distance between the document's text
   fingerprint and the template's.  High similarity → ``our_paper``.  Low
   similarity → ``counterparty_paper`` (or heavy redline, flagged as
   low-confidence).
2. **Alias position in opening recital** — locate a "between X and Y" or
   "by and between X ... and Y" pattern in the first visible text.  If one of
   our party aliases appears as the *first*-named party (before "and"), it is
   our paper.  If it appears as the *second*-named party (after "and"), it is
   counterparty paper.
3. **Alias present anywhere in first section** — weaker signal used when the
   "between...and" pattern is absent.
4. **Alias absent entirely** — no alias found anywhere → ``counterparty_paper``
   at low confidence.
5. **No aliases configured** — returns ``counterparty_paper`` at 0.50 (unknown;
   defaults toward ``counterparty_paper`` because that is the safe direction
   per OPF §2.2: it is better to miss an opening position than to falsely
   attribute one).

Confidence levels:
- ≥ 0.85  high   — template_similarity with strong match or alias_first_party
- 0.70–0.84  medium — alias_second_party
- < 0.70  low    — template dissimilarity, alias_present, alias_absent, no_aliases_configured

``AMBIGUITY_THRESHOLD = 0.70`` — callers (L5 compiler) MUST NOT use a result
with ``confidence < AMBIGUITY_THRESHOLD`` to define an opening position.
``ProvenanceResult.is_ambiguous`` exposes this check directly.

Alias matching uses whole-word boundaries (``\\b``) to prevent substring
false-positives (e.g. alias ``"Acme"`` must not match ``"Acmeseal Technologies"``).
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from playbook_engine.clause_tree import ClauseTree
from playbook_engine.config import ProvenanceConfig

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AMBIGUITY_THRESHOLD: float = 0.70
"""Confidence below this level warrants manual or LLM review.

Callers MUST check ``ProvenanceResult.is_ambiguous`` (or
``result.confidence >= AMBIGUITY_THRESHOLD``) before using the result to
define an opening position per OPF §2.2.
"""

PROVENANCE_JUDGE_BASES: frozenset[str] = frozenset(
    {
        "alias_second_party",  # name-order heuristic — weak for complex MSAs
        "no_aliases_configured",  # unknown / no signal
    }
)
"""Basis values that trigger escalation to ``ProvenanceJudge``, even when
``is_ambiguous`` is False.

``alias_second_party`` (name-order): placement of our alias as the
second-named party is a genuine signal but can be inverted by MSAs where a
counterparty reuses our form with parties swapped.  ``no_aliases_configured``:
there is no deterministic signal at all — the result is purely a safe-direction
default.  Both warrant LLM arbitration when a judge is available.
"""

# Similarity to our template above which we call it our_paper.
_OUR_PAPER_SIMILARITY_THRESHOLD: float = 0.60
# Similarity below which we call it counterparty_paper.
_COUNTERPARTY_SIMILARITY_THRESHOLD: float = 0.40
# Confidence for a DISSIMILAR-to-template result. Deliberately below
# AMBIGUITY_THRESHOLD: low line-similarity is unreliable (an .rtf template vs an
# .docx/.pdf corpus can drive even our-paper docs to ~0 overlap), so a dissimilar
# verdict must escalate to the ProvenanceJudge, not act on false confidence.
_LOW_SIMILARITY_CONFIDENCE: float = 0.60

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Opening recital "between ... and ...":
# Captures the first-named party and the second-named party.
# Limitation: assumes two-party recitals; multi-party agreements (3+) will
# fold everything after the first "and" into the second-party group.
_BETWEEN_AND = re.compile(
    r"(?:by\s+and\s+)?between\s+"
    r"([\w\s,.()\-&'\"]{3,120}?)"
    r"\s+and\s+"
    r"([\w\s,.()\-&'\"]{3,120}?)"
    r"\s*(?:[,(]|$)",
    re.IGNORECASE | re.DOTALL,
)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

_PROVENANCE_VALUES = frozenset({"our_paper", "counterparty_paper"})

_BASIS_VALUES = frozenset(
    {
        "template_similarity",
        "alias_first_party",
        "alias_second_party",
        "alias_present",
        "alias_absent",
        "no_aliases_configured",
        "llm",  # result produced by a ProvenanceJudge implementation
        "hint",  # override from hints.yaml
        "needs_review",  # store-backed judge: verdict pending human review
    }
)


@dataclass(frozen=True)
class ProvenanceResult:
    """The provenance determination for a document.

    Attributes:
        provenance:  ``"our_paper"`` or ``"counterparty_paper"``.
        confidence:  Float in [0, 1].  Values below ``AMBIGUITY_THRESHOLD``
                     indicate the determination is uncertain.
        basis:       Machine-readable reason code (one of ``_BASIS_VALUES``).

    Use ``is_ambiguous`` to gate whether this result is reliable enough to
    define an opening position (OPF §2.2).
    """

    provenance: str
    confidence: float
    basis: str

    def __post_init__(self) -> None:
        if self.provenance not in _PROVENANCE_VALUES:
            raise ValueError(
                f"Unknown provenance: {self.provenance!r}. "
                f"Must be one of {sorted(_PROVENANCE_VALUES)}"
            )
        if self.basis not in _BASIS_VALUES:
            raise ValueError(
                f"Unknown basis: {self.basis!r}. Must be one of {sorted(_BASIS_VALUES)}"
            )
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")

    @property
    def is_ambiguous(self) -> bool:
        """True when confidence < AMBIGUITY_THRESHOLD.

        An ambiguous result MUST NOT be used to define an opening position
        (OPF §2.2).  Callers should flag it for manual review or LLM
        arbitration before acting on it.
        """
        return self.confidence < AMBIGUITY_THRESHOLD


# ---------------------------------------------------------------------------
# ProvenanceJudge protocol
# ---------------------------------------------------------------------------

_PREAMBLE_MAX_LINES: int = 5
"""Maximum number of text lines extracted from the document head for the judge payload."""


@runtime_checkable
class ProvenanceJudge(Protocol):
    """Protocol for LLM-assisted provenance arbitration.

    Invoked only when the deterministic detector is ambiguous (confidence
    below ``AMBIGUITY_THRESHOLD``) or when the basis falls in
    ``PROVENANCE_JUDGE_BASES`` (e.g. name-order / no-aliases).

    Implementations may call an LLM, apply heuristics, or both.
    Contract: MUST return a ``ProvenanceResult`` with ``basis="llm"``.
    """

    def judge(
        self,
        preamble: str,
        letterhead: str,
        agreement_type: str,
    ) -> ProvenanceResult:
        """Return a provenance determination for the given document slices.

        Args:
            preamble:       The first few lines of the document body (the
                            recital / "by and between" block, ≤5 lines).
            letterhead:     The document's title / heading block (first
                            heading node, if any; else empty string).
            agreement_type: Human-readable agreement type label from the
                            engine config (e.g. ``"Master Services Agreement"``).

        Returns:
            ``ProvenanceResult`` with ``basis="llm"``.
        """
        ...  # pragma: no cover


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_provenance(
    tree: ClauseTree,
    config: ProvenanceConfig,
    *,
    template_tree: ClauseTree | None = None,
    provenance_judge: ProvenanceJudge | None = None,
    agreement_type: str = "",
) -> ProvenanceResult:
    """Determine whether *tree* is drafted on our paper or counterparty paper.

    Args:
        tree:             The document to classify (typically the first/oldest
                          version, or the template-most version from the trail).
        config:           Provenance configuration with ``our_party_aliases``.
        template_tree:    Our canonical template tree, if available.  When
                          supplied, template-similarity is the primary signal.
        provenance_judge: Optional LLM judge for ambiguous / name-order cases.
                          When provided, called if ``result.is_ambiguous`` or
                          ``result.basis in PROVENANCE_JUDGE_BASES``; its
                          ``ProvenanceResult`` replaces the heuristic result.
        agreement_type:   Agreement type label forwarded to the judge payload.

    Returns:
        A ``ProvenanceResult`` with ``provenance``, ``confidence``, and ``basis``.
        Check ``result.is_ambiguous`` before using it to define an opening
        position (OPF §2.2).
    """
    result = _detect_provenance_deterministic(tree, config, template_tree=template_tree)

    # LLM escalation: call the judge when the heuristic result is ambiguous or
    # based on a weak/unreliable signal (see PROVENANCE_JUDGE_BASES).
    if provenance_judge is not None and (
        result.is_ambiguous or result.basis in PROVENANCE_JUDGE_BASES
    ):
        preamble = _extract_preamble(tree)
        letterhead = _extract_letterhead(tree)
        result = provenance_judge.judge(preamble, letterhead, agreement_type)

    return result


def _detect_provenance_deterministic(
    tree: ClauseTree,
    config: ProvenanceConfig,
    *,
    template_tree: ClauseTree | None = None,
) -> ProvenanceResult:
    """Run the deterministic provenance heuristics; returns a ``ProvenanceResult``.

    Internal helper — external callers should use ``detect_provenance()``.
    """
    if not config.our_party_aliases:
        # B1 fix: default to counterparty_paper (safe direction per §2.2).
        # Without aliases we cannot determine provenance; defaulting toward
        # counterparty_paper means the result can only inform tolerance bounds
        # — it will NOT define a false opening position.
        return ProvenanceResult(
            provenance="counterparty_paper",
            confidence=0.50,
            basis="no_aliases_configured",
        )

    # Signal 1: template similarity (highest fidelity).
    if template_tree is not None:
        fp_doc = _fingerprint(tree)
        fp_tpl = _fingerprint(template_tree)
        similarity = _similarity(fp_doc, fp_tpl)
        if similarity >= _OUR_PAPER_SIMILARITY_THRESHOLD:
            confidence = 0.70 + 0.25 * (
                (similarity - _OUR_PAPER_SIMILARITY_THRESHOLD)
                / (1.0 - _OUR_PAPER_SIMILARITY_THRESHOLD)
            )
            return ProvenanceResult(
                provenance="our_paper",
                confidence=min(confidence, 0.95),
                basis="template_similarity",
            )
        if similarity <= _COUNTERPARTY_SIMILARITY_THRESHOLD:
            # Dissimilar to our template. This leans counterparty, but low
            # line-similarity is an unreliable signal — extraction differences
            # alone (.rtf template vs .docx/.pdf corpus) drive even our-paper
            # documents to near-zero overlap — so we return it BELOW the ambiguity
            # threshold. The caller escalates to the ProvenanceJudge instead of
            # acting on false high confidence. (High similarity, by contrast, is a
            # reliable our_paper signal and keeps its strong confidence above.)
            return ProvenanceResult(
                provenance="counterparty_paper",
                confidence=_LOW_SIMILARITY_CONFIDENCE,
                basis="template_similarity",
            )
        # Similarity is in the ambiguous middle band — fall through to alias signals.

    # Signal 2 & 3: alias position in opening text.
    opening_text = _opening_text(tree)
    alias_position = _alias_position_in_recital(opening_text, config.our_party_aliases)

    if alias_position == "first":
        return ProvenanceResult(
            provenance="our_paper",
            confidence=0.85,
            basis="alias_first_party",
        )
    if alias_position == "second":
        return ProvenanceResult(
            provenance="counterparty_paper",
            confidence=0.75,
            basis="alias_second_party",
        )

    # Signal 4: alias present anywhere in full text (weak).
    full_text = _full_text(tree)
    if _any_alias_in_text(full_text, config.our_party_aliases):
        return ProvenanceResult(
            provenance="our_paper",
            confidence=0.65,
            basis="alias_present",
        )

    # Signal 5: alias absent.
    return ProvenanceResult(
        provenance="counterparty_paper",
        confidence=0.65,
        basis="alias_absent",
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _fingerprint(tree: ClauseTree) -> list[str]:
    """Extract ordered, stripped text lines from a ClauseTree."""
    lines: list[str] = []
    for node in tree.all_nodes():
        if node.heading:
            lines.append(node.heading.strip())
        if node.text:
            for line in node.text.splitlines():
                s = line.strip()
                if s:
                    lines.append(s)
    return lines


def _similarity(a: list[str], b: list[str]) -> float:
    """Normalised similarity in [0, 1]; 1 = identical, 0 = no common lines."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def _full_text(tree: ClauseTree) -> str:
    """Concatenate all heading + body text."""
    parts: list[str] = []
    for node in tree.all_nodes():
        if node.heading:
            parts.append(node.heading)
        if node.text:
            parts.append(node.text)
    return "\n".join(parts)


def _opening_text(tree: ClauseTree, max_chars: int = 1000) -> str:
    """Return the first ``max_chars`` chars of the document's text.

    Limitation: if the document has a long title page or preamble, the
    "between ... and ..." recital may fall outside the window.  In that
    case detection falls through to the alias-presence signals.
    """
    return _full_text(tree)[:max_chars]


def _alias_re(alias: str) -> re.Pattern[str]:
    """Compile a word-boundary regex for *alias* (cached implicitly by Python's re module)."""
    return re.compile(r"\b" + re.escape(alias) + r"\b", re.IGNORECASE)


def _alias_position_in_recital(text: str, aliases: list[str]) -> str:
    """Return 'first', 'second', or 'unknown' based on alias position in a
    "between X and Y" recital pattern.

    Uses whole-word matching (``\\b``) to prevent false positives from
    substring containment (e.g. alias "Acme" must not match "Acmeseal").

    'first'  — our alias appears as the first-named party (before "and").
    'second' — our alias appears as the second-named party (after "and").
    'unknown' — no "between...and" pattern found, or alias not in either slot.

    NOTE: designed for two-party recitals.  Multi-party recitals (3+ parties)
    will fold everything after the first "and" into the second-party group.
    """
    m = _BETWEEN_AND.search(text)
    if not m:
        return "unknown"

    first_party = m.group(1).strip()
    second_party = m.group(2).strip()

    first_has_alias = any(_alias_re(alias).search(first_party) for alias in aliases)
    second_has_alias = any(_alias_re(alias).search(second_party) for alias in aliases)

    if first_has_alias and not second_has_alias:
        return "first"
    if second_has_alias and not first_has_alias:
        return "second"
    return "unknown"


def _any_alias_in_text(text: str, aliases: list[str]) -> bool:
    """Return True if any alias appears as a whole word in *text*.

    Uses word-boundary matching to prevent false positives from alias names
    that appear as substrings of longer entity names.
    """
    return any(_alias_re(alias).search(text) for alias in aliases)


def _extract_preamble(tree: ClauseTree) -> str:
    """Return the first few lines of document body text (the recital block).

    Extracts up to ``_PREAMBLE_MAX_LINES`` non-empty lines from the body text
    of the leading nodes (those with no heading).  This is the minimal payload
    that the ProvenanceJudge needs: the "by and between" recital that names the
    parties.

    The full document tree is NOT sent to the judge — only this carved slice.
    """
    lines: list[str] = []
    for node in tree.all_nodes():
        if node.text:
            for line in node.text.splitlines():
                stripped = line.strip()
                if stripped:
                    lines.append(stripped)
                    if len(lines) >= _PREAMBLE_MAX_LINES:
                        return "\n".join(lines)
        if len(lines) >= _PREAMBLE_MAX_LINES:
            break
    return "\n".join(lines)


def _extract_letterhead(tree: ClauseTree) -> str:
    """Return the document's title / heading block (letterhead).

    Returns the heading of the first node that has a heading, or an empty
    string if no heading node is found.  This gives the judge the document
    title (e.g. "Master Services Agreement") without sending the full tree.

    The full document tree is NOT sent to the judge — only this carved slice.
    """
    for node in tree.all_nodes():
        if node.heading and node.heading.strip():
            return node.heading.strip()
    return ""
