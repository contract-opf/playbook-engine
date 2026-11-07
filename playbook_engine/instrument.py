"""Per-deal instrument identity: is the executed copy the document that was negotiated?

(QA sweep 2026-11-03 #5.) The precedent record and the absence check both read a
deal's executed copy as the agreement that was negotiated. That is wrong for
general contract structure: an *amendment* to an agreement signed earlier, a
*cover sheet* or *order form* that incorporates its terms by reference or by URL,
a *statement of work* under a master agreement, or a form switch inside one deal
folder (a full agreement followed by "Amendment No. 1"). A clause missing from
such a copy is not a clause the parties left out (the terms live in another
document), and an opening draft that was a different document was not "struck
before signing".

:func:`instrument_flags` finds those deals from the text alone, never from a
corpus-specific name. A flagged deal records absence as ``unverifiable`` and
records no struck / opening / refused-ask claim (an unflagged deal is unchanged).
A flag costs real evidence (the deal's refused asks and opening facts), so each cue
is one a full agreement's boilerplate does not carry: a title line, not the word
"amendment"; ratification, not severability.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from playbook_engine.observation_builder import Observation

__all__ = [
    "AMENDMENT_TITLE_CHARS",
    "INSTRUMENT_AMENDMENT",
    "INSTRUMENT_FORM_SWITCH",
    "INSTRUMENT_INCORPORATION",
    "SHORT_INSTRUMENT_CHARS",
    "instrument_flags",
    "instrument_reason",
    "reason_flags_instrument",
]

INSTRUMENT_AMENDMENT = "amendment"
INSTRUMENT_INCORPORATION = "incorporation"
INSTRUMENT_FORM_SWITCH = "form_switch"

SHORT_INSTRUMENT_CHARS = 6000
"""An executed copy shorter than this, with an incorporation or amendment cue, is
a cover sheet or amendment rather than the agreement."""

AMENDMENT_TITLE_CHARS = 15000
"""A copy with an instrument title up front is the instrument itself only when it
is shorter than this (an amended and restated agreement is a full document)."""

_FORM_SWITCH_RATIO = 4

# Any mention of an instrument: enough only beside a form switch's size evidence.
_CUE_RE = re.compile(
    r"\b(?:amendment|addendum|amending agreement|order form|statement of work|work order"
    r"|cover sheet|joinder|side letter|supplemental agreement|renewal form)\b",
    re.IGNORECASE,
)
# A title line. Boilerplate never matches: an "Amendment." clause heading, "Order
# Form" as a defined term, "may be changed by written amendment".
_ORDINAL = (
    r"(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|\d+(?:st|nd|rd|th))"
)
_TITLE_RE = re.compile(
    r"^[ \t]*(?:"
    rf"(?:{_ORDINAL}\s+)(?:amendment|addendum)\b"
    r"|(?:amendment|addendum)\s+(?:no\b|number\b|#|\d|to\s+(?!this\b))"
    r"|(?:amending agreement|order form|statement of work|work order|cover sheet|joinder"
    r"|side letter|supplemental agreement|renewal form)(?:\s+(?:no\.?\s*)?[\w-]+)?[ \t]*$"
    r")",
    re.IGNORECASE | re.MULTILINE,
)
# Language that keeps another agreement alive. Severability says "remain in full
# force and effect" too, of this agreement's own terms: see _ratifies.
_RATIFY_RE = re.compile(
    r"\b(?:remain(?:s)? in full force and effect"
    r"|except as (?:expressly )?(?:amended|modified|supplemented)"
    r"|all other terms[^.]{0,60}(?:remain|unchanged)"
    r"|(?:hereby )?ratif(?:y|ies)(?:\s+and\s+confirm\w*)?\s+(?:the|this|that|all|each|its|such)?"
    r"[^.]{0,60}\b(?:agreement|contract|terms)\b)",
    re.IGNORECASE,
)
_SEVERABILITY_RE = re.compile(
    r"\b(?:invalid|unenforceable|illegal|severab\w*|void|terminat\w*|expir\w*|surviv\w*)\b",
    re.IGNORECASE,
)
_INCORPORATION_RE = re.compile(
    r"\bincorporated\s+(?:herein\s+|into\s+[^.]{0,40}\s+)?by\s+reference\b"
    r"|\bgoverned\s+by\s+(?:the\s+)?(?:(?!laws?\b)[^.]){0,80}\bterms\b",
    re.IGNORECASE,
)
# A URL and the word "terms" together are incorporation only when they sit close
# on one line (terms hosted at an address), not anywhere in a short agreement
# that has a web address in its notice block and says "terms" in its definitions.
_URL_TERMS_RE = re.compile(
    r"\bterms\b[^\n]{0,80}?(?:https?://|\bwww\.)|(?:https?://|\bwww\.)[^\n]{0,120}?\bterms\b",
    re.IGNORECASE,
)

_REASON_RE = re.compile(
    r"(?<!\bno )(?<!\bany )\bamendment\b|\baddendum\b|\bcover sheet\b|\border form\b|\bstatement of work\b"
    r"|\bincorporat\w*\b|\bby reference\b|\bseparate(?:ly)?\s+(?:terms|agreement|document)"
    r"|\bhosted\b|\bdifferent (?:document|agreement|instrument)\b|\bnot the (?:full|whole)\b"
    r"|\bsow\b",
    re.IGNORECASE,
)


def _ratifies(text: str) -> bool:
    """Whether a sentence of *text* keeps another agreement alive (not severability
    or survival language, which uses the same words of this agreement's terms)."""
    return any(
        _RATIFY_RE.search(sentence) and not _SEVERABILITY_RE.search(sentence)
        for sentence in re.split(r"(?<=[.;])\s+", text)
    )


def instrument_reason(text: str, *, earlier_chars: int = 0) -> str | None:
    """Why *text* (one deal's whole executed copy) is not the negotiated agreement.

    ``None`` when it reads as the agreement itself. *earlier_chars* is the size of
    the deal's largest earlier draft (``0`` when unknown); an executed copy a
    fraction of that size and carrying any instrument cue is a form switch.
    """
    chars = len(text)
    if not chars:
        return None
    title = bool(_TITLE_RE.search(text))
    ratify = _ratifies(text)
    incorporates = bool(_INCORPORATION_RE.search(text)) or bool(_URL_TERMS_RE.search(text))
    if title and chars < AMENDMENT_TITLE_CHARS:
        return INSTRUMENT_AMENDMENT
    if chars < SHORT_INSTRUMENT_CHARS:
        if incorporates:
            return INSTRUMENT_INCORPORATION
        if ratify:
            return INSTRUMENT_AMENDMENT
    if (
        earlier_chars
        and chars * _FORM_SWITCH_RATIO <= earlier_chars
        and chars < AMENDMENT_TITLE_CHARS
        and (ratify or incorporates or _CUE_RE.search(text))
    ):
        return INSTRUMENT_FORM_SWITCH
    return None


def instrument_flags(observations: Iterable[Observation]) -> dict[str, str]:
    """``{document_id: reason}`` for every deal whose executed copy is another instrument.

    The executed copy is the deal's ``outcome == "signed"`` rows in the order
    given; a deal with none is not flagged. A pure function of the rows.
    """
    signed: dict[str, list[str]] = {}
    drafts: dict[str, dict[object, int]] = {}
    for obs in observations:
        document_id = obs.citation.document_id
        if document_id == "template":
            continue
        if obs.outcome == "signed":
            signed.setdefault(document_id, []).append(obs.full_text)
        else:
            by_version = drafts.setdefault(document_id, {})
            by_version[obs.citation.version] = by_version.get(obs.citation.version, 0) + len(
                obs.full_text
            )
    flags: dict[str, str] = {}
    for document_id in sorted(signed):
        earlier = max(drafts.get(document_id, {}).values(), default=0)
        reason = instrument_reason("\n".join(signed[document_id]), earlier_chars=earlier)
        if reason is not None:
            flags[document_id] = reason
    return flags


def reason_flags_instrument(reason: object) -> bool:
    """Whether an ``absence_check`` judge's stated *reason* says the signed text is
    an amendment, cover sheet or other instrument that incorporates its terms
    elsewhere (so a clause missing from it is not a clause the deal omitted)."""
    return isinstance(reason, str) and bool(_REASON_RE.search(reason))
