"""When each deal was signed — a fact recorded at compile time (OPF 0.6, issue #247).

The toaster weighs recency and the engine can tell which of our forms was
current only if every deal carries the date it was signed. This module reads
that date off the deal's executed copy and records where it came from:

``corpus.documents[].signed_on``
    An ISO-8601 date (``YYYY-MM-DD``) or, when only the month is known,
    ``YYYY-MM``. ``null`` when the date is unknown — a date is never invented.
``corpus.documents[].signed_on_basis``
    Where the date was read, tried in this order of trust:

    1. ``document_text`` — a date the document states for itself in its
       preamble ("dated as of …", "made and entered into on …", "Effective
       Date: …", "this 3rd day of March, 2021");
    2. ``signature_block`` — a date inside the signature block ("Date: …");
    3. (the agent, between 2 and 4: when the deterministic reading of the
       text finds nothing, the store-backed ``deal_date`` judge reads the date
       passages and answers with a date, which of the two text sources it
       came from, and a verbatim quote that must occur in the passages it was
       shown — so the judge cannot invent a date either);
    4. ``filename`` — a date in the executed copy's file name;
    5. ``file_metadata`` — the executed copy's own authorship timestamp
       (DOCX core properties, PDF info dictionary).

Only a deal with a detected executed copy has a date: an unsigned deal has
none (``signed_on`` null, ``signed_on_basis`` null). A source that finds two
different dates, a numeric date that reads two ways (``03/04/2021``), an
impossible date (``February 30``) or a date outside 1980-2099 yields nothing —
the next source is tried and, if none speaks, the date is ``null``.

Two halves, like :mod:`playbook_engine.equivalence`: the pure, store-free part
(parsing, the four deterministic readers, the judge payload, resolution) lives
here; the store-backed judge that queues on a miss and replays on a hit is
:class:`playbook_engine.agent_judge.StoreBackedDealDateJudge`.

Mining records the *evidence* once per deal (:class:`DealDateEvidence`, in the
engine-internal ``deal_date_evidence`` key of the corpus manifest); projecting
*derives* the meaning (:func:`resolve_deal_dates`) from that evidence and the
verdict store, so a verdict applied after mining takes effect without re-mining.
"""

from __future__ import annotations

import datetime
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol

__all__ = [
    "BASES",
    "BASIS_DOCUMENT_TEXT",
    "BASIS_FILENAME",
    "BASIS_FILE_METADATA",
    "BASIS_SIGNATURE_BLOCK",
    "Coverage",
    "TEXT_BASES",
    "DEAL_DATE_KIND",
    "DealDate",
    "DealDateEvidence",
    "DealDateSubject",
    "coverage",
    "date_from_document_text",
    "date_from_file_metadata",
    "date_from_filename",
    "date_from_signature_block",
    "extract_evidence",
    "find_mentions",
    "is_signed_on",
    "resolve_corpus_documents",
    "resolve_signed_on",
    "signed_on_key",
    "subject_of",
    "subjects_of",
    "validate_verdict",
    "verdict_of",
]

#: The pending-item kind of the store-backed judge (``playbook judge``).
DEAL_DATE_KIND = "deal_date"

BASIS_DOCUMENT_TEXT = "document_text"
BASIS_SIGNATURE_BLOCK = "signature_block"
BASIS_FILENAME = "filename"
BASIS_FILE_METADATA = "file_metadata"

#: ``signed_on_basis`` values, in order of trust.
BASES: tuple[str, ...] = (
    BASIS_DOCUMENT_TEXT,
    BASIS_SIGNATURE_BLOCK,
    BASIS_FILENAME,
    BASIS_FILE_METADATA,
)

#: The two text sources a ``deal_date`` verdict may name.
TEXT_BASES = (BASIS_DOCUMENT_TEXT, BASIS_SIGNATURE_BLOCK)

#: Years outside this range are corrupt or placeholder values, never a deal's
#: signing year (1980 matches the file-timestamp floor in ``extraction``).
MIN_YEAR = 1980
MAX_YEAR = 2099

#: How much of the document's opening the ``document_text`` reader looks at.
PREAMBLE_CHARS = 3000

#: The date-passage excerpt a judge sees: characters of context either side of
#: a date-like token, passages per source and characters per passage.
_CONTEXT_CHARS = 160
_MAX_PASSAGES = 6
_MAX_PASSAGE_CHARS = 420


# ---------------------------------------------------------------------------
# Date values
# ---------------------------------------------------------------------------

_SIGNED_ON_RE = re.compile(r"^(\d{4})-(\d{2})(?:-(\d{2}))?$")


def is_signed_on(value: Any) -> bool:
    """``True`` for a well-formed, real ``YYYY-MM-DD`` or ``YYYY-MM`` string.

    Never raises: an impossible date (``2021-02-30``), a month 13 and a year
    outside :data:`MIN_YEAR`-:data:`MAX_YEAR` are simply not well formed.
    """
    return _parse_signed_on(value) is not None


def _parse_signed_on(value: Any) -> tuple[int, int, int] | None:
    """``(year, month, day)`` of a well-formed value (``day`` 0 for a month), else ``None``."""
    if not isinstance(value, str):
        return None
    m = _SIGNED_ON_RE.match(value)
    if m is None:
        return None
    year, month = int(m.group(1)), int(m.group(2))
    day = int(m.group(3)) if m.group(3) is not None else 0
    if not MIN_YEAR <= year <= MAX_YEAR or not 1 <= month <= 12:
        return None
    if day:
        try:
            datetime.date(year, month, day)
        except ValueError:
            return None
    elif m.group(3) is not None:  # "-00"
        return None
    return (year, month, day)


def signed_on_key(value: Any) -> tuple[int, int, int] | None:
    """Sortable key of a ``signed_on`` (a month sorts before every day in it); ``None`` if unknown."""
    return _parse_signed_on(value)


def _format(year: int, month: int, day: int | None) -> str | None:
    """``YYYY-MM-DD`` / ``YYYY-MM`` for a real date within range, else ``None``."""
    if not MIN_YEAR <= year <= MAX_YEAR or not 1 <= month <= 12:
        return None
    if day is None:
        return f"{year:04d}-{month:02d}"
    try:
        datetime.date(year, month, day)
    except ValueError:
        return None
    return f"{year:04d}-{month:02d}-{day:02d}"


# ---------------------------------------------------------------------------
# Date mentions in free text
# ---------------------------------------------------------------------------

_MONTHS = {
    "january": 1,
    "jan": 1,
    "february": 2,
    "feb": 2,
    "march": 3,
    "mar": 3,
    "april": 4,
    "apr": 4,
    "may": 5,
    "june": 6,
    "jun": 6,
    "july": 7,
    "jul": 7,
    "august": 8,
    "aug": 8,
    "september": 9,
    "sept": 9,
    "sep": 9,
    "october": 10,
    "oct": 10,
    "november": 11,
    "nov": 11,
    "december": 12,
    "dec": 12,
}
_MONTH = (
    r"(?P<mon>jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?"
    r"|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
)
_MONTH_NAMED = _MONTH.replace("(?P<mon>", "(?P<mon{n}>")
_YEAR = r"(?P<year>\d{4})"
_ORD = r"(?:st|nd|rd|th)?"
#: A blank where a day belongs: "March ___, 2021", "this ____ day of March".
_BLANK_DAY = r"(?:_{2,}|\.{3,}|\[\s*_*\s*\]|—+|-{3,})"

#: The shapes read from running text. ``ym`` ("2021-03") is a file-name shape:
#: in a document it would read case numbers and section numbers as dates.
_TEXT_SHAPES = frozenset({"iso", "mdy", "dmy", "blank_day", "month_year", "numeric"})

# Order matters: the more specific shapes mask their span before the shorter
# ones run, so "March 3, 2021" is never also read as "March 3".
_MENTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("iso", re.compile(r"(?<!\d)(?P<year>\d{4})[-/.](?P<a>\d{1,2})[-/.](?P<b>\d{1,2})(?!\d)")),
    (
        "mdy",
        re.compile(
            rf"(?<![A-Za-z]){_MONTH}\.?\s+(?P<day>\d{{1,2}}){_ORD}\s*,?\s*{_YEAR}(?!\d)",
            re.IGNORECASE,
        ),
    ),
    (
        "dmy",
        re.compile(
            rf"(?<![\d.])(?P<day>\d{{1,2}}){_ORD}\s+(?:day\s+of\s+)?{_MONTH}\.?\s*,?\s*{_YEAR}(?!\d)",
            re.IGNORECASE,
        ),
    ),
    (
        "blank_day",
        re.compile(
            rf"(?:(?<![A-Za-z]){_MONTH}\.?\s*{_BLANK_DAY}\s*,?\s*{_YEAR}"
            rf"|{_BLANK_DAY}\s+day\s+of\s+{_MONTH_NAMED.format(n='2')}\.?\s*,?\s*(?P<year2>\d{{4}}))(?!\d)",
            re.IGNORECASE,
        ),
    ),
    (
        "month_year",
        re.compile(rf"(?<![A-Za-z]){_MONTH}\.?,?\s+{_YEAR}(?!\d)", re.IGNORECASE),
    ),
    (
        "numeric",
        re.compile(r"(?<![\d/.\-])(?P<a>\d{1,2})[-/.](?P<b>\d{1,2})[-/.](?P<year>\d{4})(?![\d/])"),
    ),
    ("ym", re.compile(r"(?<![\d-])(?P<year>\d{4})-(?P<a>\d{2})(?![\d-])")),
)


@dataclass(frozen=True)
class Mention:
    """One date found in text: ``value`` is ``None`` when it does not read one way."""

    value: str | None
    start: int
    end: int
    shape: str


def _month_of(raw: str) -> int | None:
    """Month number of a matched month token; a lowercase modal "may" is not a month."""
    if raw.lower() == "may" and raw not in ("May", "MAY"):
        return None
    return _MONTHS.get(raw.lower())


def _read_mention(shape: str, m: re.Match[str]) -> tuple[bool, str | None]:
    """``(is_a_date_mention, value)`` for one regex match.

    ``is_a_date_mention`` is ``False`` for a match that is not a date at all
    (a lowercase "may 2021"); ``True`` with ``value`` ``None`` for a date that
    cannot be read one way (an ambiguous or impossible one).
    """
    g = m.groupdict()
    if shape == "iso":
        return True, _format(int(g["year"]), int(g["a"]), int(g["b"]))
    if shape in ("mdy", "dmy"):
        month = _month_of(g["mon"])
        if month is None:
            return False, None
        return True, _format(int(g["year"]), month, int(g["day"]))
    if shape in ("blank_day", "month_year"):
        raw = g.get("mon") or g.get("mon2")
        year = g.get("year") or g.get("year2")
        month = _month_of(raw or "")
        if month is None or year is None:
            return False, None
        return True, _format(int(year), month, None)
    if shape == "numeric":
        a, b, year = int(g["a"]), int(g["b"]), int(g["year"])
        if a > 12 and b <= 12:
            return True, _format(year, b, a)  # day-first: 25/03/2021
        if b > 12 and a <= 12:
            return True, _format(year, a, b)  # month-first: 03/25/2021
        if a == b:
            return True, _format(year, a, b)
        return True, None  # 03/04/2021 reads two ways
    if shape == "ym":
        return True, _format(int(g["year"]), int(g["a"]), None)
    return False, None


def find_mentions(text: str, *, shapes: Iterable[str] | None = None) -> list[Mention]:
    """Every date mentioned in *text*, left to right, each span read once.

    A mention that cannot be read one way has ``value`` ``None`` — callers
    treat it as "this source is ambiguous", never skip it silently.
    """
    wanted = set(shapes) if shapes is not None else None
    taken: list[tuple[int, int]] = []
    found: list[Mention] = []
    for shape, pattern in _MENTION_PATTERNS:
        if wanted is not None and shape not in wanted:
            continue
        for m in pattern.finditer(text):
            if any(m.start() < e and s < m.end() for s, e in taken):
                continue
            is_date, value = _read_mention(shape, m)
            if not is_date:
                continue
            taken.append((m.start(), m.end()))
            found.append(Mention(value=value, start=m.start(), end=m.end(), shape=shape))
    found.sort(key=lambda mention: mention.start)
    return found


def _reconcile(mentions: list[Mention], *, latest: bool) -> str | None:
    """One value from a source's mentions, or ``None`` when it does not speak clearly.

    Any unreadable mention makes the whole source ambiguous. Month-precision
    and day-precision mentions of the same month agree (the day wins). Two
    different dates agree only with ``latest``: a signature block carries one
    date per signatory and the deal was signed when the last one signed.
    """
    if not mentions or any(mention.value is None for mention in mentions):
        return None
    values = sorted({mention.value for mention in mentions if mention.value is not None})
    if latest:
        return max(values, key=lambda v: _parse_signed_on(v) or (0, 0, 0))
    best = max(values, key=len)
    if all(best.startswith(v) for v in values):
        return best
    return None


# ---------------------------------------------------------------------------
# The four deterministic readers
# ---------------------------------------------------------------------------

#: A cue that says the date right after it is the document's own date. It
#: must sit IMMEDIATELY before the date, with only spacing and punctuation
#: between ("dated (as of / this)", "effective (as of / on / this)", "the
#: Effective Date (is / of)", "as of", "entered into (on / as of / this)",
#: "made / executed / signed (on / as of / this)", a "Date:" label): a
#: defined term mentioned a few words earlier ("begin on the Effective Date
#: and end on June 30, 2026") or an end or limit phrase ("effective until",
#: "through", "on or after") is not one.
_CUE_BEFORE_RE = re.compile(
    r"\b(?:(?:dated|executed|signed|made|entered\s+into)(?:\s+(?:as\s+of|on|this))?"
    r"|effective(?:\s+(?:as\s+of|on|this|date(?:\s+(?:is|of))?))?"
    r"|as\s+of"
    r"|date\s*:"
    r")[\s,:;\"'\u201c\u201d\u2018\u2019(\[]*$",
    re.IGNORECASE,
)
#: A "Date:" label of the term's end, never the document's own date.
_END_DATE_LABEL_RE = re.compile(
    r"\b(?:termination|expiration|expiry|end|renewal)\s+date\s*:[\s,;\"'\u201c\u201d(\[]*$",
    re.IGNORECASE,
)
#: "this 3rd day of March, 2021": the cue is inside the date itself.
_DAY_OF_RE = re.compile(r"\bday\s+of\b", re.IGNORECASE)
_CUE_WINDOW = 70


def _is_cued(head: str, mention: Mention) -> bool:
    """Whether *mention* is introduced as the document's own date (see :data:`_CUE_BEFORE_RE`)."""
    if mention.shape == "blank_day" or _DAY_OF_RE.search(head[mention.start : mention.end]):
        return True
    before = head[max(0, mention.start - _CUE_WINDOW) : mention.start]
    return bool(_CUE_BEFORE_RE.search(before)) and not _END_DATE_LABEL_RE.search(before)


def date_from_document_text(text: str) -> str | None:
    """The date the document states for itself in its opening, or ``None``.

    Reads the first :data:`PREAMBLE_CHARS` characters and keeps only dates
    that a cue introduces right before them ("dated", "effective", "as of",
    "entered into", "made", "executed", "Date:", "day of"): a date in the
    body that is not the document's own ("within 30 days of March 1, 2021",
    "effective until December 31, 2025", "begin on the Effective Date and
    end on June 30, 2026") is not read. Two different cued dates ("dated …
    amending the agreement dated …") are ambiguous: ``None``.
    """
    head = text[:PREAMBLE_CHARS]
    cued = [
        mention for mention in find_mentions(head, shapes=_TEXT_SHAPES) if _is_cued(head, mention)
    ]
    return _reconcile(cued, latest=False)


def date_from_signature_block(text: str) -> str | None:
    """The date in the signature block, or ``None``.

    Every date in the block counts (it holds nothing but the execution): the
    latest wins when signatories dated their signatures differently. A block
    with an unreadable date is ambiguous: ``None``.
    """
    return _reconcile(find_mentions(text, shapes=_TEXT_SHAPES), latest=True)


_STAGING_PREFIX_RE = re.compile(r"^\d+__")
_EXTENSION_RE = re.compile(r"\.[A-Za-z0-9]{2,5}$")
_FILENAME_DIGITS_RE = re.compile(r"(?<!\d)(?P<year>(?:19|20)\d{2})(?P<m>\d{2})(?P<d>\d{2})(?!\d)")


def date_from_filename(name: str) -> str | None:
    """The date in a file name, or ``None``.

    Reads ``2021-03-03`` / ``2021.03.03`` / ``20210303`` / ``2021-03`` /
    ``March 3, 2021`` / ``3 March 2021`` / ``March 2021`` shapes (underscores
    count as spaces); the staging ``01__`` prefix and the extension are
    dropped first. A name with two different dates, or a numeric one that
    reads two ways, gives ``None``.
    """
    stem = _EXTENSION_RE.sub("", name.strip())
    stem = _STAGING_PREFIX_RE.sub("", stem)
    spaced = stem.replace("_", " ")
    mentions = find_mentions(spaced)
    taken = [(m.start, m.end) for m in mentions]
    for m in _FILENAME_DIGITS_RE.finditer(spaced):
        if any(m.start() < e and s < m.end() for s, e in taken):
            continue
        mentions.append(
            Mention(
                value=_format(int(m.group("year")), int(m.group("m")), int(m.group("d"))),
                start=m.start(),
                end=m.end(),
                shape="compact",
            )
        )
    return _reconcile(mentions, latest=False)


_TIMESTAMP_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})(?:T|\s|$)")


def date_from_file_metadata(timestamp: str | None) -> str | None:
    """The date part of a file's authorship timestamp (``YYYY-MM-DDTHH:MM:SSZ``), or ``None``."""
    if not isinstance(timestamp, str):
        return None
    m = _TIMESTAMP_RE.match(timestamp.strip())
    if m is None:
        return None
    return _format(int(m.group(1)), int(m.group(2)), int(m.group(3)))


# ---------------------------------------------------------------------------
# Evidence (recorded once at mine time)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DealDateEvidence:
    """What mining read about one deal's signing date — facts only, no meaning.

    ``document_text`` / ``signature_block`` / ``filename`` / ``file_metadata``
    are the four deterministic readings (``None`` = that source said nothing
    clear). ``passages`` is the judge payload — the date passages of the two
    text sources — present only when neither text reader found a date and
    there is something date-like to read (``None`` otherwise: nothing to ask).
    """

    document_text: str | None = None
    signature_block: str | None = None
    filename: str | None = None
    file_metadata: str | None = None
    passages: dict[str, list[str]] | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            BASIS_DOCUMENT_TEXT: self.document_text,
            BASIS_SIGNATURE_BLOCK: self.signature_block,
            BASIS_FILENAME: self.filename,
            BASIS_FILE_METADATA: self.file_metadata,
        }
        if self.passages is not None:
            out["passages"] = {k: list(v) for k, v in self.passages.items()}
        return out

    @classmethod
    def from_dict(cls, raw: Any) -> DealDateEvidence:
        """Read an evidence dict back; anything malformed reads as "no evidence"."""
        if not isinstance(raw, dict):
            return cls()

        def _date(key: str) -> str | None:
            value = raw.get(key)
            return value if is_signed_on(value) else None

        passages = raw.get("passages")
        clean: dict[str, list[str]] | None = None
        if isinstance(passages, dict):
            clean = {
                src: [p for p in passages.get(src, []) if isinstance(p, str)]
                for src in TEXT_BASES
                if isinstance(passages.get(src, []), list)
            }
            if not any(clean.values()):
                clean = None
        return cls(
            document_text=_date(BASIS_DOCUMENT_TEXT),
            signature_block=_date(BASIS_SIGNATURE_BLOCK),
            filename=_date(BASIS_FILENAME),
            file_metadata=_date(BASIS_FILE_METADATA),
            passages=clean,
        )


#: A token a person would read a date from: a month name beside a number, a
#: year, or "day of".
_DATE_LIKE_RE = re.compile(
    rf"(?:{_MONTH.replace('(?P<mon>', '(?:')}\.?\s+\d)|(?<!\d)(?:19|20)\d{{2}}(?!\d)|\bday\s+of\b",
    re.IGNORECASE,
)


def _date_passages(text: str, *, by_line: bool = False) -> list[str]:
    """What a judge is shown of *text*: only the passages that hold a date-like token.

    By default, windows of context around each date-like token, merged. With
    ``by_line`` (the signature block), just the lines holding one: a signature
    block is names and titles around its dates, and the judge needs none of
    them. Duplicates are dropped; the passages are capped. This is all a judge
    needs to read a date, and it keeps names and the rest of the document out
    of the payload.
    """
    passages: list[str] = []
    if by_line:
        for line in text.splitlines():
            if _DATE_LIKE_RE.search(line):
                passage = " ".join(line.split())[:_MAX_PASSAGE_CHARS]
                if passage not in passages:
                    passages.append(passage)
        return passages[:_MAX_PASSAGES]
    spans: list[list[int]] = []
    for m in _DATE_LIKE_RE.finditer(text):
        start, end = max(0, m.start() - _CONTEXT_CHARS), min(len(text), m.end() + _CONTEXT_CHARS)
        if spans and start <= spans[-1][1]:
            spans[-1][1] = max(spans[-1][1], end)
        else:
            spans.append([start, end])
    for start, end in spans[:_MAX_PASSAGES]:
        passage = " ".join(text[start:end].split())[:_MAX_PASSAGE_CHARS]
        if passage not in passages:
            passages.append(passage)
    return passages


def extract_evidence(
    *,
    document_text: str,
    signature_block_text: str,
    filename: str | None,
    file_timestamp: str | None,
) -> DealDateEvidence:
    """Read the four deterministic sources of an executed copy, and the judge passages.

    *document_text* is the executed copy's full text, *signature_block_text*
    the text of its signature block (``""`` when none was found).
    """
    from_text = date_from_document_text(document_text)
    from_block = date_from_signature_block(signature_block_text) if signature_block_text else None
    passages: dict[str, list[str]] | None = None
    if from_text is None and from_block is None:
        found = {
            BASIS_DOCUMENT_TEXT: _date_passages(document_text[:PREAMBLE_CHARS]),
            BASIS_SIGNATURE_BLOCK: _date_passages(signature_block_text, by_line=True),
        }
        passages = found if any(found.values()) else None
    return DealDateEvidence(
        document_text=from_text,
        signature_block=from_block,
        filename=date_from_filename(filename) if filename else None,
        file_metadata=date_from_file_metadata(file_timestamp),
        passages=passages,
    )


# ---------------------------------------------------------------------------
# The judge: payload, verdict
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DealDateSubject:
    """One question for the ``deal_date`` judge: the date passages of an executed copy.

    ``payload`` is exactly what the judge sees and the only thing the store
    key is derived from (content, not a document id or file name): two deals
    whose passages are identical share one verdict.
    """

    payload: dict[str, Any]


def subject_of(evidence: DealDateEvidence) -> DealDateSubject | None:
    """The judge question for *evidence*, or ``None`` when there is nothing to ask."""
    if evidence.passages is None:
        return None
    return DealDateSubject(
        payload={
            "stage": DEAL_DATE_KIND,
            BASIS_DOCUMENT_TEXT: list(evidence.passages.get(BASIS_DOCUMENT_TEXT, [])),
            BASIS_SIGNATURE_BLOCK: list(evidence.passages.get(BASIS_SIGNATURE_BLOCK, [])),
        }
    )


@dataclass(frozen=True)
class DealDate:
    """A ``deal_date`` judge's answer: the date, which text it came from, and the quote."""

    signed_on: str | None
    basis: str | None
    quote: str | None = None


def _squash(text: str) -> str:
    return " ".join(text.split())


def validate_verdict(verdict: dict[str, Any], payload: dict[str, Any] | None = None) -> DealDate:
    """Check a ``deal_date`` verdict; return it as a :class:`DealDate` or raise ``ValueError``.

    ``{"signed_on": "2021-03-03" | "2021-03" | null, "basis": "document_text" |
    "signature_block" | null, "quote": "<verbatim from that source's passages>" |
    null}``. A date needs its basis and its quote; ``null`` needs neither
    (the text says nothing about when the deal was signed). With the question's
    *payload* the quote must occur, whitespace aside, in the passages of the
    named source — a date the judge cannot point to is not recorded.
    """
    if not isinstance(verdict, dict):
        raise ValueError("a deal_date verdict must be a JSON object")
    signed_on = verdict.get("signed_on")
    basis = verdict.get("basis")
    quote = verdict.get("quote")
    if signed_on is None:
        if basis is not None or quote is not None:
            raise ValueError("'basis' and 'quote' must be null when 'signed_on' is null")
        return DealDate(None, None, None)
    if not is_signed_on(signed_on):
        raise ValueError(
            f"'signed_on' must be a real YYYY-MM-DD or YYYY-MM date in {MIN_YEAR}-{MAX_YEAR}, "
            f"got {signed_on!r} (use null when the text does not say)"
        )
    if basis not in TEXT_BASES:
        raise ValueError(f"'basis' must be one of {list(TEXT_BASES)} when a date is given")
    if not isinstance(quote, str) or not quote.strip():
        raise ValueError("'quote' must be the verbatim words the date was read from")
    if payload is not None:
        shown = " ".join(_squash(p) for p in payload.get(basis) or [])
        if _squash(quote) not in shown:
            raise ValueError(
                f"'quote' does not occur in the {basis} passages the judge was shown — a "
                "date it cannot point to is not recorded"
            )
    return DealDate(signed_on, basis, _squash(quote))


def verdict_of(record: dict[str, Any], payload: dict[str, Any]) -> DealDate | None:
    """A stored verdict as a :class:`DealDate`, or ``None`` when it is malformed for *payload*."""
    try:
        return validate_verdict(record, payload)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Resolution (derived at project time)
# ---------------------------------------------------------------------------


class _Judge(Protocol):
    """What :func:`resolve_signed_on` needs of a judge (``StoreBackedDealDateJudge``)."""

    def judge(self, subject: DealDateSubject) -> DealDate | None: ...


def resolve_signed_on(
    evidence: DealDateEvidence, judge: _Judge | None = None
) -> tuple[str | None, str | None]:
    """``(signed_on, signed_on_basis)`` of a deal from its evidence — ``(None, None)`` if unknown.

    Document text (unless it gives only a month and the signature block gives
    a day in that month: the day wins, with the block as its basis), then
    signature block, then the judge's reading of the date
    passages (when the deterministic readers found nothing and a verdict is
    stored), then the file name, then the file metadata.
    """
    if evidence.document_text is not None:
        # A month-precision reading ("this ___ day of March, 2021") yields to
        # a day in that same month from the signature block: the date is
        # recorded at the finest precision known, with the basis of the
        # source that supplied it.
        month, block = evidence.document_text, evidence.signature_block
        if len(month) == 7 and block is not None and len(block) == 10 and block.startswith(month):
            return block, BASIS_SIGNATURE_BLOCK
        return evidence.document_text, BASIS_DOCUMENT_TEXT
    if evidence.signature_block is not None:
        return evidence.signature_block, BASIS_SIGNATURE_BLOCK
    subject = subject_of(evidence)
    if subject is not None and judge is not None:
        answer = judge.judge(subject)
        if answer is not None and answer.signed_on is not None:
            return answer.signed_on, answer.basis
    if evidence.filename is not None:
        return evidence.filename, BASIS_FILENAME
    if evidence.file_metadata is not None:
        return evidence.file_metadata, BASIS_FILE_METADATA
    return None, None


def resolve_corpus_documents(
    corpus_documents: list[dict[str, Any]], judge: _Judge | None = None
) -> list[dict[str, Any]]:
    """*corpus_documents* with ``signed_on`` / ``signed_on_basis`` derived on every in-scope document.

    A document with no detected executed copy (``signed_version`` null), and
    a store mined before dates were recorded (no ``deal_date_evidence``), get
    ``null`` / ``null``. Out-of-scope documents are left as they are (they
    carry no negotiation, hence no date). The input is not mutated.
    """
    out: list[dict[str, Any]] = []
    for doc in corpus_documents:
        if not doc.get("in_scope", True):
            out.append(doc)
            continue
        new_doc = dict(doc)
        signed_on: str | None = None
        basis: str | None = None
        if doc.get("signed_version") is not None:
            signed_on, basis = resolve_signed_on(
                DealDateEvidence.from_dict(doc.get("deal_date_evidence")), judge
            )
        new_doc["signed_on"] = signed_on
        new_doc["signed_on_basis"] = basis
        out.append(new_doc)
    return out


def subjects_of(corpus_documents: list[dict[str, Any]]) -> list[DealDateSubject]:
    """The distinct judge questions of a store's corpus documents (payload order of first sight)."""
    seen: set[str] = set()
    subjects: list[DealDateSubject] = []
    for doc in corpus_documents:
        if not doc.get("in_scope", True) or doc.get("signed_version") is None:
            continue
        subject = subject_of(DealDateEvidence.from_dict(doc.get("deal_date_evidence")))
        if subject is None:
            continue
        marker = repr(sorted(subject.payload.items()))
        if marker in seen:
            continue
        seen.add(marker)
        subjects.append(subject)
    return subjects


@dataclass
class Coverage:
    """Counts only: how many in-scope deals carry a date, by basis."""

    in_scope: int = 0
    signed: int = 0
    dated: int = 0
    by_basis: dict[str, int] = field(default_factory=dict)

    @property
    def null_dates(self) -> int:
        return self.signed - self.dated


def coverage(corpus_documents: list[dict[str, Any]]) -> Coverage:
    """Counts of resolved ``signed_on`` by basis over *corpus_documents* (resolved ones)."""
    cov = Coverage()
    for doc in corpus_documents:
        if not doc.get("in_scope", True):
            continue
        cov.in_scope += 1
        if doc.get("signed_version") is None:
            continue
        cov.signed += 1
        if doc.get("signed_on") is not None:
            cov.dated += 1
            basis = str(doc.get("signed_on_basis"))
            cov.by_basis[basis] = cov.by_basis.get(basis, 0) + 1
    cov.by_basis = {b: cov.by_basis[b] for b in BASES if b in cov.by_basis}
    return cov
