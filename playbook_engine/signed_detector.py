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
that is not a template placeholder (issue #221).  A value is *blank* when,
after dropping bracketed placeholders (``[Name]``, ``<Signature>``,
``{Signatory}``, ``«Name»``) and any ``Name:``/``Title:``/``Date:``/``Its:``/
``Signature:`` label residue (a flattened ``By: Name:`` line), no run of two
or more letters is left — so ``By: _____``, ``By: [Name]``, ``By: Name:`` and
``By: X`` are all blank (unsigned).  A value whose every word is generic
signature-block vocabulary (``By: Authorized Signatory``) is
*placeholder-like*: it never counts toward a signature on its own.  When the
only filled values in the section are placeholder-like (the case that used to
read as ``dual_signatures`` at 0.90), the result is withheld as ambiguous
(``blank_signature_blocks`` at 0.60, not signed) and escalates to the
``signed_judge``.  ``/s/`` markers are filtered the same way (``/s/ [Name]``
is not an electronic signature).

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

# Blank / template placeholder value on a "By:" line: underscores, dashes
# and whitespace only. ``_is_blank_value`` widens this (issue #221) to
# bracketed placeholders, label residue and values with no letter-run.
_BLANK_VALUE = re.compile(r"^[_ \t\-–—]*$")

# A bracketed template placeholder: "[Name]", "<Signature>", "{Signatory}",
# "«Name»". An unclosed bracket runs to the end of the value ("[Name" when a
# capture stopped at the closing bracket).
_BRACKETED = re.compile(r"\[[^\]]*\]?|<[^>]*>?|\{[^}]*\}?|«[^»]*»?")

# Signature-field label residue inside a "By:" value — a flattened
# "By: Name: Title:" line, or "By: ____ Name: Alice Smith". Everything from
# the first label on belongs to the NEXT field, never to the By: value.
_LABEL_RESIDUE = re.compile(
    r"(?<![^\W\d_])(?:print(?:ed)?\s+name|name|title|date|its|signature)\s*:.*$",
    re.IGNORECASE | re.DOTALL,
)

# A run of two or more letters — a value without one ("X", "J.", "/ /") is
# not a signature.
_LETTER_RUN = re.compile(r"[^\W\d_]{2,}")

# Generic signature-block vocabulary. A value made only of these words
# ("Authorized Signatory", "Signature of Authorized Representative") is a
# printed caption, not a signature — placeholder-like (issue #221).
_PLACEHOLDER_WORDS = frozenset(
    {
        "authorized",
        "authorised",
        "signatory",
        "signature",
        "signer",
        "sign",
        "here",
        "representative",
        "officer",
        "name",
        "title",
        "print",
        "printed",
        "type",
        "typed",
        "insert",
        "party",
        "company",
        "counterparty",
        "duly",
        "its",
        "of",
        "the",
        "and",
        "by",
        "for",
        "on",
        "behalf",
    }
)

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
        slash_s_count = _count_slash_s(full_text)
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

    # Count filled, placeholder-like and blank "By:" lines (issue #221).
    filled_count, placeholder_count, blank_count = _classify_by_lines(sig_text)

    # Count /s/ markers carrying a real name.
    slash_s_count = _count_slash_s(sig_text)

    total_sig_count = filled_count + slash_s_count

    if total_sig_count >= 2:
        return SignedStatus(signed=True, basis="dual_signatures", confidence=0.90)
    if filled_count == 1:
        return SignedStatus(signed=True, basis="single_signature", confidence=0.75)
    if slash_s_count == 1:
        return SignedStatus(signed=True, basis="electronic_signature", confidence=0.80)

    # Issue #221: the only "filled" values are placeholder-like captions
    # ("By: Authorized Signatory" twice used to read as dual_signatures at
    # 0.90 — an unsigned template becoming the signed terminal). Never
    # signed on that evidence: withheld as ambiguous and escalated, so a
    # wired signed_judge reads the section; without one it stays unsigned.
    if placeholder_count > 0:
        result = SignedStatus(signed=False, basis="blank_signature_blocks", confidence=0.60)
        if signed_judge is not None and result.confidence < AMBIGUITY_THRESHOLD:
            return signed_judge.judge(sig_text)
        return result

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


def _signature_value_kind(value: str) -> Literal["blank", "placeholder", "filled"]:
    """Classify the value of a ``By:`` (or ``/s/``) signature field (issue #221).

    - ``"blank"``: empty, underscores/dashes only, or — once bracketed
      placeholders (``[Name]``, ``<Signature>``) and ``Name:``/``Title:``/
      ``Date:``/``Its:``/``Signature:`` label residue are dropped — no run of
      two or more letters is left (``By: [Name]``, ``By: Name:``, ``By: X``).
    - ``"placeholder"``: letters remain, but every word of two or more
      letters is generic signature-block vocabulary
      (``By: Authorized Signatory``) — a printed caption, not a signature.
    - ``"filled"``: anything else.
    """
    value = value.strip()
    if not value or _BLANK_VALUE.match(value):
        return "blank"
    value = _LABEL_RESIDUE.sub("", value)
    value = _BRACKETED.sub(" ", value)
    if not _LETTER_RUN.search(value):
        return "blank"
    words = [w.lower() for w in _LETTER_RUN.findall(value)]
    if all(w in _PLACEHOLDER_WORDS for w in words):
        return "placeholder"
    return "filled"


def _count_by_lines(text: str) -> tuple[int, int]:
    """Return (filled_count, blank_count) for 'By:' lines in *text*.

    Filled: the value after 'By:' is not a template placeholder — see
            :func:`_signature_value_kind`. Placeholder-like values
            (``By: Authorized Signatory``) count as filled here; use
            :func:`_classify_by_lines` to tell them apart.
    Blank:  the value is empty, underscores/dashes, a bracketed placeholder,
            label residue, or has no run of two or more letters.
    """
    filled, placeholder, blank = _classify_by_lines(text)
    return filled + placeholder, blank


def _classify_by_lines(text: str) -> tuple[int, int, int]:
    """Return (filled, placeholder_like, blank) counts for 'By:' lines in *text*."""
    counts = {"filled": 0, "placeholder": 0, "blank": 0}
    for m in _BY_LINE.finditer(text):
        counts[_signature_value_kind(m.group(1))] += 1
    return counts["filled"], counts["placeholder"], counts["blank"]


def _count_slash_s(text: str) -> int:
    """Count ``/s/`` electronic signatures in *text* whose name is real.

    ``/s/ [Name]`` or ``/s/ ________`` in an unexecuted form is a placeholder
    (issue #221): the value — the rest of the line after the marker — must
    classify as ``"filled"`` to count.
    """
    count = 0
    for m in _SLASH_S.finditer(text):
        end = text.find("\n", m.start(1))
        rest = text[m.start(1) : end if end != -1 else len(text)]
        rest = rest.split("|", 1)[0]
        if _signature_value_kind(rest) == "filled":
            count += 1
    return count


# ---------------------------------------------------------------------------
# Signature-block stripping (issue #217)
# ---------------------------------------------------------------------------
#
# The ingesters that start a clause only on a numbered/heading paragraph
# (RTF, PDF, and DOCX without a heading-styled signature page) append the
# execution trailer — "IN WITNESS WHEREOF", party captions, By:/Name:/Title:
# lines, signatory names — to the body of whatever clause preceded it,
# usually the last one (counterparts, entire agreement). That text is not
# clause language: it is prompt noise for every judge that reads the clause,
# it rides into our_standard / signed_text / the digest, and the
# signatories' names are a pseudonymization residue path (people's names are
# not in known_entities). ``strip_signature_block`` cuts it out of the clause
# text deterministically and reports where it was, so the pipeline can
# record the span in version_ingest.
#
# Signed-copy detection (``detect_signed`` above) and party-alias scans NEED
# this text — a filled "By:" line is the signal — so the pipeline runs them
# on the unstripped tree and uses the stripped one for everything that
# treats node text as clause language.

# An execution-trailer phrase ("IN WITNESS WHEREOF, the parties have
# executed…") that OPENS the block: at the start of a line, or right after a
# sentence end on the same line (a PDF text layer can run the trailer onto
# the last clause sentence's line). A clause that merely mentions the phrase
# mid-sentence is never cut.
_SIG_BLOCK_TRAILER = re.compile(
    rf"(?:^|(?<=[.!?]))[ \t]*(?P<trailer>{_TRAILER_ALTS})", re.IGNORECASE | re.MULTILINE
)

# A line that opens a signature field group when no trailer phrase exists.
_SIG_BLOCK_START_LINE = re.compile(r"^[ \t]*(?:By[ \t]*:|/s/)", re.IGNORECASE)

# Any signature-field line: By:/Name:/Title:/Its:/Date:/Signature: or /s/.
_SIG_FIELD_LINE = re.compile(
    r"^[ \t]*(?:(?:By|Name|Title|Its|Date|Signature)[ \t]*:|/s/)", re.IGNORECASE
)

# A field-group start only counts as a signature block when at least this
# many signature-field lines follow from it (itself included) — one stray
# "By:" line in ordinary prose is not a block.
_MIN_SIG_FIELD_LINES = 2

# Party-caption lines directly above a field group ("AlphaCorp Holdings,
# Inc.", "[Counterparty]", "BETA INDUSTRIES LLC") belong to the block too.
_CAPTION_MAX_WORDS = 8
_CAPTION_MAX_LINES = 3
_CORPORATE_SUFFIX = re.compile(
    r"\b(?:inc|llc|l\.l\.c|ltd|limited|corp|corporation|company|co|lp|l\.p|llp|plc|gmbh|ag|s\.a|n\.v|b\.v)\.?$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SignatureBlock:
    """Where a version's signature block was cut from its clause text.

    Attributes:
        clause_path:  The clause whose ``text`` held the block (and was
                      truncated at the block start).
        char_span:    ``[start, end)`` of the removed block in the document's
                      full normalized text (same coordinates as
                      ``ClauseNode.char_span``), or ``None`` when the node's
                      spans could not be related to its text (a legacy tree
                      with heading-only spans) — the text is still cut.
        basis:        ``"trailer"`` (an "IN WITNESS WHEREOF"/"executed as a
                      deed" phrase opens the block) or ``"signature_fields"``
                      (a By:/"/s/" field group in the last clause, plus the
                      party-caption lines directly above it).
    """

    clause_path: str
    char_span: tuple[int, int] | None
    basis: Literal["trailer", "signature_fields"]


def _lines_with_offsets(text: str) -> list[tuple[int, str]]:
    out: list[tuple[int, str]] = []
    offset = 0
    for line in text.split("\n"):
        out.append((offset, line))
        offset += len(line) + 1
    return out


def _is_caption_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped or len(stripped.split()) > _CAPTION_MAX_WORDS:
        return False
    if stripped.startswith("[") and stripped.endswith("]"):
        return True
    if any(c.isalpha() for c in stripped) and stripped == stripped.upper():
        return True
    return bool(_CORPORATE_SUFFIX.search(stripped.rstrip(",:")))


def _trailer_offset(text: str) -> int | None:
    m = _SIG_BLOCK_TRAILER.search(text)
    return m.start("trailer") if m is not None else None


def _signature_fields_offset(text: str) -> int | None:
    lines = _lines_with_offsets(text)
    for i, (_offset, line) in enumerate(lines):
        if not _SIG_BLOCK_START_LINE.match(line):
            continue
        n_fields = sum(1 for _, rest in lines[i:] if _SIG_FIELD_LINE.match(rest))
        if n_fields < _MIN_SIG_FIELD_LINES:
            return None
        j = i
        while j > 0 and i - j < _CAPTION_MAX_LINES and _is_caption_line(lines[j - 1][1]):
            j -= 1
        return lines[j][0]
    return None


def _locate_signature_block(
    nodes: list[ClauseNode],
) -> tuple[int, int, Literal["trailer", "signature_fields"]] | None:
    """Return ``(node_index, offset_in_text, basis)`` or ``None``.

    The LAST node carrying a trailer phrase wins (a signature page followed
    by an exhibit keeps the exhibit) — in its body text, or as its whole
    heading (an ingester that reads a short ALL-CAPS "IN WITNESS WHEREOF"
    line as a heading puts the block in that node's body, all of which is
    cut). Failing that, a By:/"/s/" field group is looked for only in the
    last clause that has body text — the one place an unheaded execution
    block lands.
    """
    for idx in range(len(nodes) - 1, -1, -1):
        node = nodes[idx]
        offset = _trailer_offset(node.text or "")
        if offset is None and node.heading and _SIG_BLOCK_TRAILER.match(node.heading.strip()):
            offset = 0 if (node.text or "").strip() else None
        if offset is not None:
            return idx, offset, "trailer"
    for idx in range(len(nodes) - 1, -1, -1):
        if (nodes[idx].text or "").strip():
            offset = _signature_fields_offset(nodes[idx].text)
            return (idx, offset, "signature_fields") if offset is not None else None
    return None


def strip_signature_block(tree: ClauseTree) -> tuple[ClauseTree, SignatureBlock | None]:
    """Cut the signature block out of *tree*'s clause text (issue #217).

    Returns ``(tree, None)`` unchanged when no block is found. Otherwise
    returns a NEW tree (the input is never mutated) whose block-holding node
    has its ``text`` truncated at the block start (trailing whitespace
    dropped) and its ``char_span`` end pulled back to the end of the kept
    text — or to the end of its heading line when nothing is kept — plus the
    :class:`SignatureBlock` describing what was removed. Only that node's
    own text is cut; nodes after it (an exhibit) are left alone.

    Deterministic, no LLM. Callers run ``detect_signed`` on the UNSTRIPPED
    tree — the block is exactly the evidence that detector reads.
    """
    located = _locate_signature_block(list(tree.all_nodes()))
    if located is None:
        return tree, None
    node_index, offset, basis = located

    stripped = ClauseTree.from_dict(tree.to_dict())
    target = list(stripped.all_nodes())[node_index]
    text = target.text
    kept = text[:offset].rstrip()

    # The node's own text ends at char_span[1] on every current producer
    # (ingester whole-clause spans, segmenter-promoted sub-clauses, grounded
    # LLM nodes), so its start is char_span[1] - len(text). A legacy tree's
    # heading-only span fails this check — cut the text, report no span.
    text_start = target.char_span[1] - len(text)
    heading_end = target.heading_span[1] if target.heading_span is not None else None
    block_span: tuple[int, int] | None = None
    if text_start >= target.char_span[0] and (heading_end is None or text_start > heading_end):
        block_span = (text_start + offset, target.char_span[1])
        if kept:
            new_end = text_start + len(kept)
        elif heading_end is not None:
            new_end = heading_end
        else:
            new_end = target.char_span[0]
        target.char_span = (target.char_span[0], new_end)
    target.text = kept
    return stripped, SignatureBlock(
        clause_path=target.clause_path, char_span=block_span, basis=basis
    )
