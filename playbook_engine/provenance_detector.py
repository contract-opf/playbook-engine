"""Provenance detector — L2 structure layer.

Decides whether a document is drafted on *our paper* (our standard form / our
template) or *counterparty paper* (the counterparty's form) — or that the
side cannot be determined (``"unknown"``).  In OPF 0.4 the result is deal
metadata only (the precedent record's three-valued ``paper``): it never
partitions, gates or weights anything, and ``our_standard`` comes only from
the configured template (owner decision 2026-09-13 (b), issue #225).

Detection signals (applied in priority order):
1. **Template similarity** (highest fidelity) — if a canonical template is
   supplied, compare the document's node fingerprint with the template's
   (``version_orderer.node_fingerprint``: one whitespace-collapsed element per
   node, so a DOCX paragraph and the same paragraph wrapped across PDF/RTF
   lines are the same element; ``SequenceMatcher`` with ``autojunk=False`` so
   repeated boilerplate elements are never silently discarded).  High
   similarity → ``our_paper``.  Low similarity → ``counterparty_paper`` at
   low confidence (or heavy redline — ambiguous, escalated).
2. **Alias position in opening recital** — locate a "between X and Y" or
   "by and between X ... and Y" pattern in the first visible text.  If one of
   our party aliases appears as the *first*-named party (before "and"), it
   leans our paper — but many counterparty forms name the customer first, so
   this name-order signal is escalated to the judge exactly like the
   second-named case.  If it appears as the *second*-named party (after
   "and"), it leans counterparty paper.
3. **Alias present anywhere in first section** — weaker signal used when the
   "between...and" pattern is absent.
4. **Alias absent entirely** — no alias found anywhere → ``counterparty_paper``
   at low confidence.
5. **No aliases configured** — returns ``counterparty_paper`` at 0.50: no
   signal at all, so the lean is below the ambiguity threshold and the deal's
   paper is recorded ``"unknown"``.

Confidence levels:
- ≥ 0.85  high   — template_similarity with strong match
- 0.70–0.84  medium — alias_second_party (0.75), alias_first_party (0.70)
- < 0.70  low    — template dissimilarity, alias_present, alias_absent, no_aliases_configured

``"unknown"`` provenance (a store-backed judge with no verdict yet,
``basis="needs_review"``, confidence 0.0) is always ambiguous.

``AMBIGUITY_THRESHOLD = 0.70`` — an ambiguous result
(``ProvenanceResult.is_ambiguous``: confidence below the threshold, or no side
at all) is recorded as paper ``"unknown"``, never coerced to a side (issue
#225).

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
        "alias_first_party",  # name-order heuristic — counterparty forms often name the customer first
        "alias_second_party",  # name-order heuristic — weak for complex MSAs
        "no_aliases_configured",  # unknown / no signal
    }
)
"""Basis values that trigger escalation to ``ProvenanceJudge``, even when
``is_ambiguous`` is False.

``alias_first_party`` / ``alias_second_party`` (name-order): the recital's
party order is a genuine signal but an unreliable one — many counterparty
forms name the customer first, and MSAs where a counterparty reuses our form
can swap the parties.  ``no_aliases_configured``: there is no deterministic
signal at all — the result is purely a default.  All three warrant LLM
arbitration when a judge is available.
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
"""The two paper SIDES — what a hints.yaml ``provenance`` override may name
(``version_orderer.Hints``) and what OPF's two-valued
``corpus.documents[].provenance`` can carry."""

PROVENANCE_UNKNOWN: str = "unknown"
"""A determination with no side: a store-backed judge with no verdict yet
(``basis="needs_review"``). Always ambiguous; recorded as paper
``"unknown"``, never coerced to a side (issue #225)."""

_RESULT_PROVENANCE_VALUES = _PROVENANCE_VALUES | {PROVENANCE_UNKNOWN}


def two_valued_side(provenance: str) -> str:
    """The value a frozen two-valued OPF provenance field carries for *provenance*.

    ``corpus.documents[].provenance`` has no ``"unknown"`` — the published
    enum is frozen. A side passes through unchanged; ``"unknown"`` (no side
    at all) is written as ``counterparty_paper``, the §2.3 direction that
    never lets it define an opening position. It is only ever written next to the honest
    record of the undetermined side: ``provenance_is_ambiguous: true`` on the
    corpus document and ``paper: "unknown"`` on the OPF 0.4 precedent record
    (issue #225).
    """
    return provenance if provenance in _PROVENANCE_VALUES else "counterparty_paper"


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
        provenance:  ``"our_paper"``, ``"counterparty_paper"``, or
                     ``"unknown"`` (no side determined — always ambiguous).
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
        if self.provenance not in _RESULT_PROVENANCE_VALUES:
            raise ValueError(
                f"Unknown provenance: {self.provenance!r}. "
                f"Must be one of {sorted(_RESULT_PROVENANCE_VALUES)}"
            )
        if self.basis not in _BASIS_VALUES:
            raise ValueError(
                f"Unknown basis: {self.basis!r}. Must be one of {sorted(_BASIS_VALUES)}"
            )
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")

    @property
    def is_ambiguous(self) -> bool:
        """True when confidence < AMBIGUITY_THRESHOLD, or no side was determined.

        An ambiguous result is recorded as paper ``"unknown"`` — never
        coerced to a side (issue #225) — and MUST NOT be used to define an
        opening position (OPF §2.2).
        """
        return self.provenance == PROVENANCE_UNKNOWN or self.confidence < AMBIGUITY_THRESHOLD


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
            # similarity is still a weak signal: the node fingerprint removes
            # the line-shape artefact (a PDF/RTF extraction wrapping lines
            # differently from the DOCX template), yet a heavily edited
            # our-paper draft, or node text altered by extraction (e.g. OCR
            # noise), can still score low — so we return it BELOW the ambiguity
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
        # Issue #225: name order is weak — many counterparty forms name the
        # customer first — so this sits at the ambiguity threshold (not
        # ambiguous on its own) and is in PROVENANCE_JUDGE_BASES (escalated
        # whenever a judge is configured).
        return ProvenanceResult(
            provenance="our_paper",
            confidence=0.70,
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
    """The version orderer's format-independent node fingerprint (issue #225).

    One whitespace-collapsed element per node heading and per node body —
    never one per physical line, which made a PDF/RTF extraction of our own
    template (wrapped across lines differently from the DOCX template)
    collapse toward zero similarity purely on line shape.
    """
    # Deferred: version_orderer imports this module (hints validation).
    from playbook_engine.version_orderer import node_fingerprint  # noqa: PLC0415

    return node_fingerprint(tree)


def _similarity(a: list[str], b: list[str]) -> float:
    """Normalised similarity in [0, 1]; 1 = identical, 0 = no common elements.

    ``autojunk=False``: SequenceMatcher's default heuristic treats any element
    repeated in more than 1% of a 200+-element sequence as junk and drops it
    from matching, so a long agreement's repeated boilerplate (a recurring
    heading or sentence) would silently stop counting toward similarity.
    """
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


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
