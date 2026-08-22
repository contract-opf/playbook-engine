"""Signed-copy detector — L2 structure layer.

Determines whether a ClauseTree represents an executed (signed) copy of an
agreement.  Detection is fully deterministic; LLM arbitration is reserved for
cases whose confidence falls below ``AMBIGUITY_THRESHOLD``.

Detection heuristics (applied in priority order):
1. DocuSign / Adobe Sign certificate page — highest-confidence signal.
2. Two or more *filled* signature blocks (dual-party execution).
3. One filled signature block.
4. Electronic-signature (``/s/ …``) markers.
5. Signature section found but all blocks blank → *not* signed.
6. Signature section found with zero evidence (no filled/blank ``By:`` line,
   no ``/s/``) → *not* signed; confidence depends on how the section was
   found — see "Signature sections that never became their own node" below.
7. No signature section found → *not* signed.

A "filled" signature block is one where a ``By:`` line is followed by a value
that contains at least one letter (i.e. not blank, not underscores/dashes
only).  Template placeholders such as ``By: _____________________`` are treated
as blank (unsigned).

Confidence levels:
- ≥ 0.90  high — docusign_cert, or a dual-signature block localized to a
  signature section
- 0.75–0.89  medium — single filled block, or /s/ markers found only in
  document-wide text with no section to localize them
- 0.60–0.74  low — ambiguous; LLM arbitration recommended

``AMBIGUITY_THRESHOLD = 0.70`` marks the boundary below which callers may
forward the document to an LLM for confirmation.

Signature section vs. business-term "execution":
  ``_SIG_HEADING`` is anchored at end-of-string so that ``execution`` only
  matches when it IS the heading (e.g. "EXECUTION", "COUNTERPART EXECUTION")
  and not when it appears as a noun in business headings like
  "Execution of Services".

Signature sections that never became their own node:
  Ingesters that only start a clause on a *numbered* paragraph (RTF, PDF)
  append an unnumbered execution trailer to the body of whatever clause
  preceded it, so no node's *heading* is a signature heading and the document
  reads as ``no_signature_section`` at 0.85 — a confident wrong answer for an
  executed copy, which then withholds every observation downstream.  Two
  defences: ``_SIG_TRAILER`` matches a signature section from *body* text, and
  ``detect_signed`` falls back to document-wide ``/s/`` markers when no
  signature section is found at all.

  ``_SIG_TRAILER`` matches execution-page boilerplate ("in witness whereof",
  "executed as a deed") in ordinary body prose too — a clause that merely
  *mentions* execution in passing, not a real signature block.  When that is
  the ONLY reason a node matched (no heading anywhere) and nothing filled in
  turns up, treating it the same as a genuine empty heading would send it to
  LLM arbitration for content that's almost always incidental boilerplate.
  ``_signature_nodes`` therefore tags each match's provenance (``"heading"``
  vs. ``"trailer"``), and ``detect_signed``'s final branch uses it: any
  heading match keeps ``basis="empty_signature_section"`` at the ambiguous
  0.60 (escalates); trailer-only matches get the confident
  ``basis="unsigned_trailer_reference"`` at 0.80 (no escalation).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

from playbook_engine.clause_tree import ClauseNode, ClauseTree

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AMBIGUITY_THRESHOLD: float = 0.70
"""Confidence below this level warrants LLM arbitration."""

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# UUID pattern — 8-4-4-4-12 hex digits (case-insensitive).
_UUID_PAT = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"

# Certificate page from DocuSign or Adobe Sign.
# Requires DocuSign Envelope ID to be an actual UUID so that instructional
# mentions like "an Envelope ID will be assigned" do not fire.
_DOCUSIGN_CERT = re.compile(
    rf"(docusign\s+envelope\s+id\s*:\s*{_UUID_PAT}|"
    r"certificate\s+of\s+completion\s+.*\s+docusign|"
    r"electronically\s+signed\s+by\s.*\s+docusign)",
    re.IGNORECASE | re.DOTALL,
)

# Loose DocuSign/Adobe indicator (lower confidence — used as tie-breaker).
_ESIGN_PLATFORM = re.compile(
    r"\b(docusign|adobe\s+sign|hellosign)\b",
    re.IGNORECASE,
)

# Execution-trailer phrases strong enough to identify a signature section from
# *body* text alone — see _signature_nodes.
#
# Deliberately a strict subset of _SIG_HEADING: it omits "signatures?" and
# "execution$" because those are safe only as headings.  Ordinary body prose
# says "signature" constantly ("signature pages may be delivered by facsimile",
# "an authorized signature is required"), so matching it against body text
# would turn half a contract into signature sections.  These two phrases are
# execution-page boilerplate and essentially never appear elsewhere.
_TRAILER_ALTS = r"\bin\s+witness\s+whereof\b|\bexecuted\s+as\s+a\s+deed\b"

_SIG_TRAILER = re.compile(rf"(?:{_TRAILER_ALTS})", re.IGNORECASE)

# Heading that introduces a signature block.
#
# "execution" is anchored to end-of-string (after optional whitespace) so it
# does NOT match "Execution of Services", "Execution Schedule", etc.
# Other patterns use word-boundary matching because they are multi-word
# phrases specific enough to avoid false positives.
_SIG_HEADING = re.compile(
    r"(?:"
    r"\bsignatures?\b"
    r"|execution\s*$"
    rf"|{_TRAILER_ALTS}"
    r")",
    re.IGNORECASE,
)

# "By:" label used in signature blocks.
#
# Matches "By:" either at the start of a line OR right after a "|" cell
# separator, so that a table-laid-out signature block — flattened by
# docx_ingester._flatten_table / extraction.py's Markdown table-row parsing
# into a single "By: | By: " pipe-joined line — still yields one match per
# signature cell instead of only the line-initial one (issue #94). The
# captured value stops at the next "|" (or end of line) so it never bleeds
# into an adjacent cell's text.
_BY_LINE = re.compile(r"(?:^[ \t]*|\|\s*)By\s*:\s*([^|\r\n]*)", re.IGNORECASE | re.MULTILINE)

# Blank / template placeholder value on a "By:" line.
_BLANK_VALUE = re.compile(r"^[_ \t\-–—]*$")

# Electronic /s/ signature.
_SLASH_S = re.compile(r"/s/\s+(\S[\w ,.\-]{0,80})")


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

_BASIS_VALUES = frozenset(
    {
        "docusign_cert",
        "dual_signatures",
        "single_signature",
        "electronic_signature",
        "blank_signature_blocks",
        "empty_signature_section",
        "unsigned_trailer_reference",
        "no_signature_section",
        "llm",
        "hint",
    }
)


@dataclass(frozen=True)
class SignedStatus:
    """The signed-copy determination for a single document version.

    Attributes:
        signed:      True if the version is judged to be an executed copy.
        basis:       Machine-readable reason code (one of ``_BASIS_VALUES``).
        confidence:  Float in [0, 1]; values below ``AMBIGUITY_THRESHOLD``
                     suggest LLM review is warranted.
    """

    signed: bool
    basis: str
    confidence: float

    def __post_init__(self) -> None:
        if self.basis not in _BASIS_VALUES:
            raise ValueError(
                f"Unknown basis: {self.basis!r}. Must be one of {sorted(_BASIS_VALUES)}"
            )
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")


# ---------------------------------------------------------------------------
# Judge protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class SignedJudge(Protocol):
    """Protocol for LLM-assisted signed-copy arbitration.

    Called only when ``detect_signed`` returns a ``confidence`` value below
    ``AMBIGUITY_THRESHOLD``.  The caller passes the carved signature section
    text (already extracted during detection) so the LLM receives a focused,
    ~300-token payload rather than the full document.

    Contract:
    - Implementations MUST return a ``SignedStatus`` with ``basis="llm"``.
    - Implementations MUST NOT raise; on any error they should return a
      conservative ``SignedStatus(signed=False, basis="llm", confidence=0.0)``.
    """

    def judge(self, signature_subtree: str) -> SignedStatus:
        """Return a signed determination for the given signature section text.

        Args:
            signature_subtree: The extracted signature section text (heading +
                               body of all signature nodes and their descendants).

        Returns:
            ``SignedStatus`` with ``basis="llm"``.
        """
        ...  # pragma: no cover


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_signed(tree: ClauseTree, *, signed_judge: SignedJudge | None = None) -> SignedStatus:
    """Determine whether *tree* represents an executed (signed) copy.

    Args:
        tree:         A ``ClauseTree`` produced by any ingester (DOCX, PDF, RTF).
                      May have been passed through ``segment()`` first — this
                      function recurses into children to find signature content.
        signed_judge: Optional ``SignedJudge`` for LLM arbitration.  When
                      provided and the deterministic result has
                      ``confidence < AMBIGUITY_THRESHOLD``, the judge is called
                      with the carved signature section text and its verdict
                      replaces the low-confidence result.

    Returns:
        A ``SignedStatus`` with ``signed``, ``basis``, and ``confidence``.
    """
    full_text = _full_text(tree)

    # Priority 1: strong e-sign certificate signal.
    if _DOCUSIGN_CERT.search(full_text):
        return SignedStatus(signed=True, basis="docusign_cert", confidence=0.95)

    # Locate nodes that belong to a signature section, tagged with how each
    # one qualified (a real heading, or a body-text trailer match).
    sig_matches = _signature_nodes(tree)

    if not sig_matches:
        # No signature section to carve — fall back to document-wide markers.
        #
        # Confidences here are degraded relative to the localized path below
        # (0.90 / 0.80) because nothing corroborates *where* the markers sit:
        # a stray "/s/" in an unexecuted exhibit form reads the same as a real
        # execution page.  This is the same discount the _ESIGN_PLATFORM
        # fallback already takes.  A lone unlocalized marker lands under
        # AMBIGUITY_THRESHOLD so it escalates instead of asserting.
        slash_s_count = len(_SLASH_S.findall(full_text))
        if slash_s_count >= 2:
            return SignedStatus(signed=True, basis="dual_signatures", confidence=0.85)
        if slash_s_count == 1:
            result = SignedStatus(signed=True, basis="electronic_signature", confidence=0.65)
            if signed_judge is not None and result.confidence < AMBIGUITY_THRESHOLD:
                return signed_judge.judge(full_text)
            return result

        # Weak e-sign platform mention in body text — low confidence.
        if _ESIGN_PLATFORM.search(full_text):
            result = SignedStatus(signed=True, basis="electronic_signature", confidence=0.65)
            if signed_judge is not None and result.confidence < AMBIGUITY_THRESHOLD:
                return signed_judge.judge(full_text)
            return result
        return SignedStatus(signed=False, basis="no_signature_section", confidence=0.85)

    # Collect the full text of each signature node, including all descendants,
    # because the segmenter may have promoted inline sub-clauses to children.
    sig_nodes = [node for node, _provenance in sig_matches]
    sig_text = "\n".join(_node_subtree_text(n) for n in sig_nodes)

    # Count filled and blank "By:" lines.
    filled_count, blank_count = _count_by_lines(sig_text)

    # Count /s/ markers.
    slash_s_count = len(_SLASH_S.findall(sig_text))

    total_sig_count = filled_count + slash_s_count

    if total_sig_count >= 2:
        return SignedStatus(signed=True, basis="dual_signatures", confidence=0.90)
    if filled_count == 1:
        return SignedStatus(signed=True, basis="single_signature", confidence=0.75)
    if slash_s_count == 1:
        return SignedStatus(signed=True, basis="electronic_signature", confidence=0.80)

    # Signature section found but no filled blocks.
    if blank_count > 0:
        return SignedStatus(signed=False, basis="blank_signature_blocks", confidence=0.80)

    # No filled/blank By: lines and no /s/ marker anywhere in the matched
    # nodes.  Provenance decides how confident "not signed" is:
    #
    # - Any node matched via a real *heading* → a signature section genuinely
    #   exists (found on purpose, not incidentally); it could be an unsigned
    #   template or a signature page whose blocks failed to extract either
    #   way, so stay ambiguous and keep escalating to the judge.
    # - All matches are _SIG_TRAILER hits in body text (no heading at all) →
    #   this is boilerplate like "IN WITNESS WHEREOF" mentioned in passing
    #   with zero corroborating evidence.  Confident "not signed" — no
    #   escalation, so this doesn't pay for LLM arbitration on content that's
    #   almost always incidental.
    if any(provenance == "heading" for _node, provenance in sig_matches):
        result = SignedStatus(signed=False, basis="empty_signature_section", confidence=0.60)
        if signed_judge is not None and result.confidence < AMBIGUITY_THRESHOLD:
            return signed_judge.judge(sig_text)
        return result

    return SignedStatus(signed=False, basis="unsigned_trailer_reference", confidence=0.80)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _full_text(tree: ClauseTree) -> str:
    """Concatenate all heading + body text in the tree (all nodes)."""
    parts: list[str] = []
    for node in tree.all_nodes():
        if node.heading:
            parts.append(node.heading)
        if node.text:
            parts.append(node.text)
    return "\n".join(parts)


def _node_subtree_text(node: ClauseNode) -> str:
    """Return heading + body text for *node* and all its descendants.

    Recurses into children so that signature content promoted to child nodes
    by the segmenter is captured.
    """
    parts: list[str] = []
    if node.heading:
        parts.append(node.heading)
    if node.text:
        parts.append(node.text)
    for child in node.children:
        child_text = _node_subtree_text(child)
        if child_text:
            parts.append(child_text)
    return "\n".join(parts)


def _signature_nodes(tree: ClauseTree) -> list[tuple[ClauseNode, Literal["heading", "trailer"]]]:
    """Return nodes that belong to a signature section, tagged by provenance.

    A node qualifies on either of two signals, recorded alongside it so
    callers can weigh the two differently when no filled/blank evidence turns
    up in the matched text (see ``detect_signed``'s final branch):

    - its *heading* matches ``_SIG_HEADING`` — the normal case, where the
      ingester gave the execution page its own node — tagged ``"heading"``; or
    - its *body* matches ``_SIG_TRAILER`` — the absorbed-trailer case, where
      an unnumbered "IN WITNESS WHEREOF …" block was appended to the preceding
      numbered clause instead of starting one of its own, leaving no heading to
      match — tagged ``"trailer"``.

    A node whose heading matches is tagged ``"heading"`` even if its body also
    happens to match ``_SIG_TRAILER``: a real heading is the stronger signal a
    signature section actually exists.

    Matching the body only against the strict ``_SIG_TRAILER`` subset keeps the
    "Execution of Services" guard intact: the loose ``signatures?`` /
    ``execution$`` alternatives stay heading-only.

    Widening this can only *add* text to the caller's ``sig_text``, and both
    the ``By:`` and ``/s/`` counts are monotone in that text, so a document
    already judged signed cannot be flipped to unsigned by a trailer match.
    """
    result: list[tuple[ClauseNode, Literal["heading", "trailer"]]] = []
    for node in tree.all_nodes():
        if _SIG_HEADING.search(node.heading or ""):
            result.append((node, "heading"))
        elif _SIG_TRAILER.search(node.text or ""):
            result.append((node, "trailer"))
    return result


def _count_by_lines(text: str) -> tuple[int, int]:
    """Return (filled_count, blank_count) for 'By:' lines in *text*.

    Filled: the value after 'By:' contains at least one letter.
    Blank:  the value is empty, underscores, dashes, or whitespace only.
    """
    filled = 0
    blank = 0
    for m in _BY_LINE.finditer(text):
        value = m.group(1).strip()
        if not value or _BLANK_VALUE.match(value):
            blank += 1
        else:
            filled += 1
    return filled, blank
