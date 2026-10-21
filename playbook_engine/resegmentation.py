"""Re-segmentation — L3 pipeline stage (issue #255).

On the engine's real-corpus evaluation 61% of the clause texts the playbook compared
were segmentation artefacts: a deal's "clause" either held provisions of other
clause types (a preamble that swallowed an insurance paragraph) or was cut
short (a sentence broken at a page boundary). That pollutes precedent, the
``vs_standard`` part maps and absence counts. This stage runs after a version
is segmented and classified and repairs both, conservatively and
deterministically, with no judge and no key:

1. **Split pass** (:func:`resegment`). Each classified node is cut into units
   (paragraphs, then sentences and list items, *inside* the node: an agent
   segmenter can only cut at extractor block boundaries, and many single
   blocks hold several topics). Every unit is scored against every clause
   type's :class:`TypeProfiles` (the words characteristic of that type,
   learned from our standard and from the classified exemplars of the type
   across the corpus, so types with no ``our_standard`` such as insurance are
   covered too). A unit is re-homed only when it clears a threshold AND leads
   the node's own type and the runner-up by a margin; a weak score never
   moves anything. Contiguous re-homed units become a new node of the target
   type (a child of the node they were cut from, so document order and the
   tree invariants hold); the rest stay.
2. **Join pass.** A node that ends mid-sentence or mid-list is merged with the
   next sibling when that one starts mid-sentence (lowercase) or continues the
   list numbering, and both share a type (or the next is unclassified). Segments
   split at a redline boundary are stitched too: when one author's tracked
   insertion runs across the cut (the extractor records one span per
   paragraph-local insertion, so the pieces meet at the edge of the two nodes)
   and the next piece is positively of the node's type (typed so, or
   unclassified with at least :data:`JOIN_MIN_TYPE_HITS` of that type's words
   and more of them than of any other type's; a lead-in, which has no topic of
   its own, when the piece after it is) AND the text continues across the cut
   (the piece before ends open, or the next opens lower case or with a
   back-reference; a list marker on the next, of any style, counts only when
   the piece before ends open: :func:`_continues_across`), the pieces are
   one passage, with no chain cap (the size cap alone). An insertion alone is
   no such evidence: a counterparty often inserts several whole clauses in one
   go, even several of one type, so a node with no type never extends on it and
   two pieces that each read as complete stay two. On
   every basis a piece that opens with a heading of its own (one the extractor
   found, a title line, or the segmenter's label when the text begins with it)
   is never joined; the segmenter's label alone is its name for the node, not a
   clause start. Every basis re-applies step 3's not-a-clause test to the
   pieces, typed or not (a lead-in or a fragment aside, no piece it rejects is
   joined to or from), and refuses an unclassified piece the split pass's
   scorer reads as another type than the node's.
3. **Not-a-clause filter** (:func:`mark_non_clauses`). A node still
   unclassified that is a bare section number, a signature block or e-signature
   certificate, a bare heading or "intentionally omitted" placeholder, a running
   header or footer, a leftover mid-sentence fragment, a lead-in, OCR junk or an
   email header is marked ``x_non_clause`` with a reason. It is never deleted,
   so citations still resolve; it is left out of classification, precedent and
   absence counts.

Provenance. Every split and join is recorded on the node it produced
(``ClauseNode.x_resegmented``: ``split_from`` or ``joined_with``, ``basis``,
``scores`` (empty for a join and for the node a passage was cut out of),
``origin_span`` / ``origin_spans``; a join also records ``at``, the offsets in
the node's text where each joined passage starts, which
:func:`rebase_join_offsets` keeps true when the text is pseudonymized). A
node's ``char_span`` always lies inside the original block span (or covers the
two joined ones), so an OPF citation resolves to the original bytes. The pass is a pure function of the version's
tree, the classification, the profiles and the tracked changes, run once in
``mine``; ``project`` reads the recorded result, so both see the same answer
(as for form front matter, issue #242).

Agent check. :func:`check_subjects` turns a deterministic sample of the
recorded splits and joins (all of them when there are
:data:`CHECK_SAMPLE_MAX` or fewer) into ``resegment_check`` questions; the
verdicts never gate anything, they measure the thresholds.

The thresholds below were set on the shape of the problem (counts only, no
corpus text leaves the machine); each constant's docstring says what it
guards.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from playbook_engine.clause_classifier import (
    ClassifiedClause,
    ClauseClassification,
    _content_tokens,
)
from playbook_engine.clause_tree import ClauseNode, ClauseTree
from playbook_engine.docx_ingester import TrackedChanges

__all__ = [
    "CHECK_KIND",
    "CHECK_SAMPLE_MAX",
    "EMPTY_PROFILES",
    "NON_CLAUSE_REASONS",
    "ResegmentCheckSubject",
    "ResegmentResult",
    "TypeProfiles",
    "build_profiles",
    "check_subjects",
    "looks_like_heading",
    "mark_non_clauses",
    "non_clause_reason",
    "rebase_join_offsets",
    "resegment",
    "split_units",
    "validate_check_verdict",
]

#: The pending-item kind of the agent check (``judge/pending.jsonl``).
CHECK_KIND = "resegment_check"

#: A deterministic sample of at most this many recorded splits and joins is
#: queued for the check (all of them when there are this many or fewer).
CHECK_SAMPLE_MAX = 60

#: ``x_resegmented["basis"]`` values: how the pass decided.
BASIS_BLOCK_PROFILE = "block_profile"
BASIS_LOWERCASE_START = "lowercase_start"
BASIS_NUMBERING = "numbering"
BASIS_REDLINE = "redline"
JOIN_BASES = (BASIS_LOWERCASE_START, BASIS_NUMBERING, BASIS_REDLINE)

#: ``ClauseClassification.basis`` of a node re-homed by the split pass.
CLASSIFICATION_BASIS = "resegmented"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SPLIT_SCORE_THRESHOLD: float = 0.5
"""A unit is re-homed only if at least this share of its content words are
characteristic of the target type (:class:`TypeProfiles`). A sentence that is
half another clause type's vocabulary is that type's sentence; anything
weaker is left where the segmenter put it."""

SPLIT_OWN_MARGIN_RATIO: float = 2.0
"""...and the target's score is at least this many times the node's own
type's score for the same unit."""

SPLIT_OWN_MARGIN_ABS: float = 0.25
"""...and exceeds the node's own type's score by at least this much (the ratio
alone is cheap to satisfy when the own score is near zero)."""

SPLIT_RUNNER_UP_RATIO: float = 1.5
"""...and leads every other candidate type by this factor: a unit that fits
two foreign types about equally is a guess, and a wrong re-homing plants a
false precedent."""

MIN_UNIT_TOKENS: int = 6
"""Fewer content words than this and a unit is too short to score; it follows
the unit before it (a short tail never starts a new node on its own)."""

MIN_UNIT_HITS: int = 5
"""A re-homed unit must contain at least this many profile words of the
target, whatever its share: four generic words of a type's profile do not make
a sentence that type's."""

MIN_PROFILE_WEIGHT: int = 3
"""A word is characteristic of a type only if it occurs in the type's
exemplars of at least this many deals (our standard counts as
:data:`TEMPLATE_WEIGHT` deals)."""

PROFILE_DOMINANCE: float = 0.5
"""...and the type holds at least this share of the word's occurrences over
all types: a word spread over many types (``notice``, ``written``) is no
mark of one."""

MIN_TYPE_DEALS: int = 3
"""A type is a re-homing target only if its exemplars span at least this many
deals of the corpus; our standard alone never makes a target (a type seen in
one document is no evidence of what the type looks like elsewhere)."""

MIN_PROFILE_TOKENS: int = 8
"""...and its profile holds at least this many characteristic words."""

TEMPLATE_WEIGHT: int = 3
"""Our standard's words count as this many deals each (enough, with nothing
else, to clear :data:`MIN_PROFILE_WEIGHT`)."""

EXEMPLAR_MIN_CHARS: int = 60
EXEMPLAR_MAX_CHARS: int = 2500
"""Corpus exemplar nodes outside this length are not used: a stub teaches
nothing, and an oversized node is the very mixture this pass repairs."""

MIN_RETAINED_TOKENS: int = 8
"""A node is split only if the part that stays holds at least this many content
words beyond its heading line. A node whose only body is the passage that would
move is not a mixture, it is a node the segmenter typed differently, and that
call is the segmenter's (it sees the heading and the exhibit the text sits in)."""

CONFIDENCE_CAP: float = 0.5
"""Ceiling on the confidence of a ``resegmented`` classification: below
``AMBIGUITY_THRESHOLD``, so it never reads as a verified judge verdict."""

JOIN_MAX_CHARS: int = 8000
"""A join that would make a node longer than this is refused."""

JOIN_MAX_CHAIN: int = 4
"""At most this many nodes are merged into one on a textual basis (lower-case
start, list numbering). A redline join is bounded by :data:`JOIN_MAX_CHARS`
alone: the pieces of one tracked insertion are one passage however many
paragraphs it was typed over, and the producer cuts it at every block."""

JOIN_MIN_TYPE_HITS: int = 2
"""On the redline basis an unclassified piece joins a typed node only with
positive evidence that it is about the same thing: at least this many of its
content words are characteristic of the node's type (:class:`TypeProfiles`),
and the node's type scores higher on it than every other type. A tracked
insertion says one author typed the pieces in one go, not that they are one
clause: a counterparty often inserts several whole clauses at once. One shared
word is chance."""

FRAGMENT_MAX_WORDS: int = 14
"""A leftover unclassified node of at most this many words that starts
mid-sentence or ends without sentence punctuation is a ``fragment``."""

LEAD_IN_MAX_WORDS: int = 25
"""An unclassified node ending in ``:`` of at most this many words, with no
modal verb (a sentence that says what must or may happen is a clause), is a
``lead_in``."""

# ---------------------------------------------------------------------------
# Type profiles
# ---------------------------------------------------------------------------

_ALPHA_WORD = re.compile(r"[a-z]+")


def _unit_tokens(text: str) -> frozenset[str]:
    """Content words of *text* for scoring: alphabetic, three letters or more."""
    return frozenset(t for t in _content_tokens(text) if len(t) >= 3 and _ALPHA_WORD.fullmatch(t))


@dataclass(frozen=True)
class TypeProfiles:
    """The words characteristic of each clause type, learned once per corpus.

    ``tokens[tid]`` is the type's characteristic word set; ``targets`` is the
    subset of types units may be re-homed to (enough deals, enough words).
    Immutable and hashable by :meth:`fingerprint`, which keys the stage cache.
    """

    tokens: Mapping[str, frozenset[str]] = field(default_factory=dict)
    targets: frozenset[str] = frozenset()

    def fingerprint(self) -> str:
        surface = {
            "targets": sorted(self.targets),
            "tokens": {t: sorted(w) for t, w in sorted(self.tokens.items())},
            "constants": [
                SPLIT_SCORE_THRESHOLD,
                SPLIT_OWN_MARGIN_RATIO,
                SPLIT_OWN_MARGIN_ABS,
                SPLIT_RUNNER_UP_RATIO,
                MIN_UNIT_TOKENS,
                MIN_UNIT_HITS,
            ],
        }
        raw = json.dumps(surface, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]


EMPTY_PROFILES = TypeProfiles()


def build_profiles(
    exemplars: Iterable[tuple[str, str, str]],
    template: Mapping[str, str] | None = None,
) -> TypeProfiles:
    """Learn :class:`TypeProfiles` from classified exemplars.

    *exemplars* are ``(deal_id, taxonomy_id, text)`` triples: the classified
    nodes of the corpus (a deal counts once per word however many versions
    carry it). *template* is ``{taxonomy_id: our standard's clause text}``
    (every template node of the type joined, front matter aside, issue #243);
    its words weigh :data:`TEMPLATE_WEIGHT` deals. A word is characteristic of
    a type when it occurs in at least :data:`MIN_PROFILE_WEIGHT` deals of the
    type and the type holds at least :data:`PROFILE_DOMINANCE` of its
    occurrences over all types. A type is a re-homing target only when its
    corpus exemplars span :data:`MIN_TYPE_DEALS` deals and it has
    :data:`MIN_PROFILE_TOKENS` characteristic words.

    Deterministic: a pure function of its inputs, independent of their order.
    """
    deals_of: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    type_deals: dict[str, set[str]] = defaultdict(set)
    for deal_id, tid, text in exemplars:
        if not EXEMPLAR_MIN_CHARS <= len(text or "") <= EXEMPLAR_MAX_CHARS:
            continue
        type_deals[tid].add(deal_id)
        for tok in _unit_tokens(text):
            deals_of[tok][tid].add(deal_id)
    template_weight: dict[str, set[str]] = defaultdict(set)
    for tid, text in (template or {}).items():
        for tok in _unit_tokens(text):
            template_weight[tok].add(tid)

    tokens: dict[str, set[str]] = defaultdict(set)
    for tok in set(deals_of) | set(template_weight):
        weights: dict[str, int] = {}
        for tid, deals in deals_of.get(tok, {}).items():
            weights[tid] = len(deals)
        for tid in template_weight.get(tok, ()):
            weights[tid] = weights.get(tid, 0) + TEMPLATE_WEIGHT
        total = sum(weights.values())
        for tid, w in weights.items():
            if w >= MIN_PROFILE_WEIGHT and w >= PROFILE_DOMINANCE * total:
                tokens[tid].add(tok)

    targets = frozenset(
        tid
        for tid, deals in type_deals.items()
        if len(deals) >= MIN_TYPE_DEALS and len(tokens.get(tid, ())) >= MIN_PROFILE_TOKENS
    )
    return TypeProfiles(
        tokens={tid: frozenset(words) for tid, words in sorted(tokens.items())},
        targets=targets,
    )


# ---------------------------------------------------------------------------
# Units: paragraphs, then sentences and list items
# ---------------------------------------------------------------------------

_ABBREVIATIONS = frozenset(
    {
        "inc",
        "ltd",
        "llc",
        "co",
        "corp",
        "no",
        "nos",
        "st",
        "mr",
        "mrs",
        "ms",
        "dr",
        "jr",
        "sr",
        "vs",
        "etc",
        "e.g",
        "i.e",
        "u.s",
        "sec",
        "art",
        "para",
        "approx",
        "dept",
        "univ",
        "cf",
    }
)
_END_RE = re.compile(r"([.!?;:])([\"”’')\]]*)(\s+)")
_BARE_NUMBER_RE = re.compile(r"^\(?\d+(?:\.\d+)*\)?$")


def _line_units(line: str, base: int) -> list[tuple[int, int]]:
    """Sentence and list-item spans of one non-empty line (offsets from *base*)."""
    spans: list[tuple[int, int]] = []
    start = 0
    for m in _END_RE.finditer(line):
        punct = m.group(1)
        nxt = m.end(3)
        if nxt >= len(line) or m.start() < start:
            continue
        c = line[nxt]
        if punct == ":":
            ok = c == "("
        elif punct == ";":
            ok = c == "(" or c.isdigit() or c.isupper()
        else:
            ok = c.isupper() or c.isdigit() or c in '("“['
        if not ok:
            continue
        if punct in ".?!":
            head = line[start : m.start()].split()
            prev = head[-1] if head else ""
            if punct == "." and (
                prev.lower().strip("()") in _ABBREVIATIONS
                or (len(prev) == 1 and prev.isalpha())
                or _BARE_NUMBER_RE.match(prev)
            ):
                continue
        end = m.end(2)
        spans.append((base + start, base + end))
        start = nxt
    spans.append((base + start, base + len(line)))
    return [(s, e) for s, e in spans if e > s]


def split_units(text: str) -> list[tuple[int, int]]:
    """Cut *text* into units: paragraphs (lines), each into sentences / list items.

    Returns ``[(start, end), ...]`` offsets into *text*, in order, each with no
    leading or trailing whitespace; blank lines yield nothing.
    """
    units: list[tuple[int, int]] = []
    pos = 0
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped:
            lead = len(line) - len(line.lstrip())
            units.extend(_line_units(stripped, pos + lead))
        pos += len(line) + 1
    return units


# ---------------------------------------------------------------------------
# Split pass
# ---------------------------------------------------------------------------


def _score_unit(
    tokens: frozenset[str], own_tid: str, profiles: TypeProfiles
) -> tuple[str, dict[str, float]] | None:
    """The type a unit with *tokens* should be re-homed to, with its scores, or ``None``."""
    n = len(tokens)
    own_profile = profiles.tokens.get(own_tid)
    own = len(tokens & own_profile) / n if own_profile else 0.0
    scored: list[tuple[float, int, str]] = []
    for tid in sorted(profiles.targets):
        if tid == own_tid:
            continue
        hits = len(tokens & profiles.tokens[tid])
        scored.append((hits / n, hits, tid))
    if not scored:
        return None
    scored.sort(key=lambda s: (-s[0], -s[1], s[2]))
    best, hits, tid = scored[0]
    runner_up = scored[1][0] if len(scored) > 1 else 0.0
    if (
        hits < MIN_UNIT_HITS
        or best < SPLIT_SCORE_THRESHOLD
        or best < SPLIT_OWN_MARGIN_RATIO * own
        or best - own < SPLIT_OWN_MARGIN_ABS
        or best < SPLIT_RUNNER_UP_RATIO * runner_up
    ):
        return None
    return tid, {"score": best, "own": own, "runner_up": runner_up}


#: Words a title keeps in lower case.
_SMALL_WORDS = frozenset(
    {"a", "an", "and", "as", "at", "by", "for", "from", "in", "of", "on", "or", "the", "to", "with"}
)
_NUMBER_PREFIX_RE = re.compile(r"^\W*(?:\d+(?:\.\d+)*[.)]?|[A-Za-z][.)])\s+")


def looks_like_heading(text: str) -> bool:
    """Whether *text* is a title or section heading rather than a sentence.

    At most ten words after any section number, every word capitalised (or
    upper case, or a number) apart from the small words a title keeps lower
    case, no sentence punctuation inside, and no colon or comma at the end.
    """
    t = _NUMBER_PREFIX_RE.sub("", text.strip().strip("[]()").strip())
    t = t.rstrip(".").strip()
    words = t.split()
    if not 0 < len(words) <= 10 or any(c in t for c in ".;:?!,"):
        return False
    if not (words[0][0].isupper() or words[0][0].isdigit()):
        return False
    return all(w[0].isupper() or not w[0].isalpha() or w.lower() in _SMALL_WORDS for w in words)


_LIST_MARKER_RE = re.compile(r"^\(?([A-Za-z]{1,4}|\d{1,2})[.)]\s")
_FIRST_MARKERS = frozenset({"a", "1", "i"})
_DEPENDENT_OPENERS = (
    "such ",
    "said ",
    "any such ",
    "all such ",
    "the foregoing",
    "notwithstanding the foregoing",
    "failure to ",
    "in such ",
    "in addition",
    "however",
    "otherwise",
    "likewise",
    "accordingly",
    "thereafter",
    "the same ",
    "if so",
    "if not",
)


def _follows_previous(unit: str) -> bool:
    """Whether *unit* depends on the unit before it: a list item after the first, or an
    opener that refers back ("Such ...", "Failure to do so ...", "However, ...")."""
    m = _LIST_MARKER_RE.match(unit)
    if m is not None and m.group(1).lower() not in _FIRST_MARKERS:
        return True
    body = _LIST_MARKER_RE.sub("", unit).lstrip().lower()
    return body.startswith(_DEPENDENT_OPENERS)


@dataclass
class ResegmentResult:
    """What :func:`resegment` hands back for one version."""

    tree: ClauseTree
    classified: list[ClassifiedClause]
    splits: int = 0
    joins: int = 0


def _verbatim_base(node: ClauseNode) -> int | None:
    """Absolute offset of ``node.text[0]``, or ``None`` when the text is not
    known to sit verbatim flush with the node's span end (the producers' shared
    contract; see ``signed_detector.strip_signature_block``)."""
    base = node.char_span[1] - len(node.text)
    return base if base >= node.char_span[0] else None


def _split_node(
    node: ClauseNode,
    own_tid: str,
    own_cls: ClauseClassification,
    profiles: TypeProfiles,
    labels: Mapping[str, str],
) -> tuple[list[ClauseNode], dict[str, ClauseClassification], int] | None:
    """Split *node* in place; return ``(new children, classification by path, splits)``.

    ``None`` (and *node* untouched) when no unit clears the thresholds.
    """
    base = _verbatim_base(node)
    if base is None:
        return None
    text = node.text
    units = split_units(text)
    if len(units) < 2:
        return None
    anchor = 2 if len(units) > 2 and looks_like_heading(text[units[0][0] : units[0][1]]) else 1
    owner: list[str | None] = [None] * len(units)
    detail: list[dict[str, float] | None] = [None] * len(units)
    decided = False
    for i in range(anchor, len(units)):
        unit = text[units[i][0] : units[i][1]]
        toks = _unit_tokens(unit)
        if len(toks) < MIN_UNIT_TOKENS or _follows_previous(unit):
            owner[i], detail[i] = (
                owner[i - 1],
                None,
            )  # a short tail, a later list item follows its neighbour
            continue
        verdict = _score_unit(toks, own_tid, profiles)
        if verdict is not None:
            owner[i], detail[i] = verdict[0], verdict[1]
            decided = True
    if not decided:
        return None

    runs: list[tuple[int, int, str | None]] = []  # (first unit, last unit, target or None=own)
    for i, tid in enumerate(owner):
        if runs and runs[-1][2] == tid:
            runs[-1] = (runs[-1][0], i, tid)
        else:
            runs.append((i, i, tid))
    if len(runs) < 2 or runs[0][2] is not None:
        return None

    retained: set[str] = set()
    for i, tid in enumerate(owner):
        if tid is None and not (i == 0 and looks_like_heading(text[units[0][0] : units[0][1]])):
            retained |= _unit_tokens(text[units[i][0] : units[i][1]])
    if len(retained) < MIN_RETAINED_TOKENS:
        return None

    first_end = units[runs[0][1]][1]
    if node.heading_span is not None and node.heading_span[1] > base + first_end:
        return None

    origin = [node.char_span[0], node.char_span[1]]
    own_path = node.clause_path
    new_children: list[ClauseNode] = []
    cls_by_path: dict[str, ClauseClassification] = {}
    splits = 0
    for k, (i0, i1, tid) in enumerate(runs[1:], start=1):
        s, e = units[i0][0], units[i1][1]
        path = f"{own_path}.r{k}"
        meta: dict[str, Any] = {
            "split_from": own_path,
            "basis": BASIS_BLOCK_PROFILE,
            "origin_span": origin,
            "scores": {},
        }
        if tid is None:
            cls_by_path[path] = own_cls
        else:
            ds = [d for d in detail[i0 : i1 + 1] if d is not None]
            scores = {
                "score": round(min(d["score"] for d in ds), 4),
                "own": round(max(d["own"] for d in ds), 4),
                "runner_up": round(max(d["runner_up"] for d in ds), 4),
            }
            meta["scores"] = scores
            meta["to"] = {"id": tid, "label": labels.get(tid, tid)}
            cls_by_path[path] = ClauseClassification(
                taxonomy_id=tid,
                confidence=min(scores["score"], CONFIDENCE_CAP),
                basis=CLASSIFICATION_BASIS,
            )
            splits += 1
        new_children.append(
            ClauseNode(
                clause_path=path,
                heading=None,
                text=text[s:e],
                char_span=(base + s, base + e),
                page=node.page,
                x_resegmented=meta,
            )
        )
    node.text = text[:first_end]
    node.char_span = (node.char_span[0], base + first_end)
    node.x_resegmented = {
        "split_from": own_path,
        "basis": BASIS_BLOCK_PROFILE,
        "origin_span": origin,
        "scores": {},
    }
    return new_children, cls_by_path, splits


# ---------------------------------------------------------------------------
# Join pass
# ---------------------------------------------------------------------------

_TERMINAL = ".!?"
_CLOSERS = "\"”’')]"
_MARKER_AT_START = re.compile(r"^\(?([A-Za-z]{1,4}|\d{1,2})[.)]\s")
_MARKER_INLINE = re.compile(r"(?:^|[\n;:.]\s*)\(?([A-Za-z]{1,4}|\d{1,2})[.)]\s")
_ROMAN = ("i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x", "xi", "xii")


def _successor(a: str, b: str) -> bool:
    """Whether list marker *b* directly follows *a* (``a``->``b``, ``2``->``3``, ``ii``->``iii``)."""
    a, b = a.lower(), b.lower()
    if a.isdigit() and b.isdigit():
        return int(b) == int(a) + 1
    if len(a) == 1 and len(b) == 1 and a.isalpha() and b.isalpha() and a != "i":
        return ord(b) == ord(a) + 1
    if a in _ROMAN and b in _ROMAN:
        return _ROMAN.index(b) == _ROMAN.index(a) + 1
    return len(a) == 1 and len(b) == 1 and ord(b) == ord(a) + 1


def _ends_mid_sentence(text: str) -> bool:
    t = text.rstrip()
    while t and t[-1] in _CLOSERS:
        t = t[:-1].rstrip()
    return bool(t) and t[-1] not in _TERMINAL


def _insertion_spans(tracked: TrackedChanges | None) -> list[tuple[int, int, str]]:
    """``(start, end, author)`` of every tracked insertion with a span.

    The extractor records one span per paragraph-local ``w:ins``, so one
    author's insertion that runs over several paragraphs is several spans that
    meet at the paragraph boundaries.
    """
    if tracked is None:
        return []
    return [
        (c.char_span[0], c.char_span[1], c.author or "")
        for c in tracked.changes
        if c.change_type == "insertion" and c.char_span is not None
    ]


def _starts_clause(node: ClauseNode) -> bool:
    """Whether *node* opens with a heading of its own, as a clause starts.

    A heading the extractor found (``heading_span``) or a title line leading the
    text counts; so does the segmenter's label when the text begins with it. A
    segmenter label that is not in the text (``heading`` set, ``heading_span``
    ``None``, text a continuation) is the segmenter's name for the node, not a
    clause start: an agent segmenter gives every node one.
    """
    if node.heading_span is not None:
        return True
    text = (node.text or "").lstrip()
    first = text.split("\n", 1)[0].strip()
    if first and looks_like_heading(first):
        return True
    label = _NUMBER_PREFIX_RE.sub("", (node.heading or "").strip())
    label_words = re.findall(r"[a-z0-9]+", label.lower())
    if not label_words:
        return False
    return re.findall(r"[a-z0-9]+", text.lower())[: len(label_words)] == label_words


def _redline_across_cut(
    a: ClauseNode, b: ClauseNode, insertions: list[tuple[int, int, str]]
) -> bool:
    """Whether one tracked insertion runs across the cut between *a* and *b*.

    Either a single insertion straddles the cut, or an insertion ends at the
    edge of *a* and one by the same author starts at the edge of *b*: the
    shape the extractor gives a multi-paragraph insertion. An author the
    document does not name never links two insertions.
    """
    cut = a.char_span[1]
    if any(s < cut and e > b.char_span[0] for s, e, _ in insertions):
        return True
    a_content_end = cut - (len(a.text) - len(a.text.rstrip()))
    b_content_start = b.char_span[0] + (len(b.text) - len(b.text.lstrip()))
    enders = {au for s, e, au in insertions if au and s < e and a_content_end <= e <= cut}
    return any(
        au in enders
        for s, e, au in insertions
        if au and s < e and b.char_span[0] <= s <= b_content_start
    )


def _unjoinable(node: ClauseNode) -> bool:
    """Whether *node* is text the not-a-clause filter excludes (:data:`_UNJOINABLE_REASONS`)."""
    return non_clause_reason(node) in _UNJOINABLE_REASONS


def _unclassified(cls: ClauseClassification) -> bool:
    return cls.taxonomy_id is None and cls.basis == "unclassified"


def _reads_as_other_type(node: ClauseNode, tid: str, profiles: TypeProfiles) -> bool:
    """Whether *node*'s words make it, by the split pass's own scorer
    (:func:`_score_unit`), a clause of a split target other than *tid*."""
    toks = _unit_tokens(node.text)
    return bool(toks) and _score_unit(toks, tid, profiles) is not None


def _profile_evidence(node: ClauseNode, tid: str, profiles: TypeProfiles) -> bool:
    """Whether *node*'s own words say it is about type *tid* (:data:`JOIN_MIN_TYPE_HITS`)."""
    toks = _unit_tokens(node.text)
    own_profile = profiles.tokens.get(tid)
    if not toks or not own_profile:
        return False
    hits = len(toks & own_profile)
    if hits < JOIN_MIN_TYPE_HITS:
        return False
    return all(len(toks & words) < hits for other, words in profiles.tokens.items() if other != tid)


def _type_evidence(
    node: ClauseNode,
    tid: str,
    cls_by_path: Mapping[str, ClauseClassification],
    profiles: TypeProfiles,
) -> bool:
    """Whether *node* is positively of type *tid*: typed so, or unclassified with
    :func:`_profile_evidence` for it."""
    cls = cls_by_path.get(node.clause_path)
    if cls is None:
        return False
    if cls.taxonomy_id is not None:
        return cls.taxonomy_id == tid
    return _unclassified(cls) and _profile_evidence(node, tid, profiles)


def _continues_across(a: ClauseNode, b: ClauseNode) -> bool:
    """Whether the text itself says *b* continues *a* at the cut between them.

    *a* ends open (no sentence-final punctuation: mid-sentence, or a colon or
    semicolon that hands over to what follows), or *b* opens lower case, or
    with a back-reference ("Such ...", "However, ..."). A list marker of any
    style ("2.", "(b)", "b.", "b)", "ii.") on *b* counts only when *a* ends
    open: after a complete sentence it opens a numbered clause of its own, and
    a lower-case marker is not a lower-case opening.
    Two pieces that each read as complete, the second opening a sentence of its
    own, are two paragraphs, and a tracked insertion alone does not make them
    one clause: a counterparty often inserts a whole run of clauses of one type
    in one go (the blind check found one such run joined into a single clause).
    """
    ta, tb = a.text.rstrip(), b.text.lstrip()
    if not ta or not tb or len(re.findall(r"[A-Za-z]{2,}", ta)) < 3:
        return False
    if _ends_mid_sentence(ta):
        return True
    if _LIST_MARKER_RE.match(tb):
        # A numbered opener after a complete sentence is a numbered clause of its
        # own, not a continuation: the marker is evidence only when *a* ends open.
        # Checked before the lower-case test, which "b." or "ii)" would pass.
        return False
    return tb[0].islower() or _follows_previous(tb)


def _redline_piece(
    a: ClauseNode,
    b: ClauseNode,
    c: ClauseNode | None,
    cls_by_path: Mapping[str, ClauseClassification],
    insertions: list[tuple[int, int, str]],
    profiles: TypeProfiles,
) -> bool:
    """Whether *b* is the next piece of a tracked insertion that *a* is typed by.

    One author's insertion runs across the cut, *b* opens with no heading of its
    own, the text continues across the cut (:func:`_continues_across`), and *b*
    is positively of *a*'s type (:func:`_type_evidence`). A lead-in has no topic
    of its own: it joins when the piece after it (*c*) is the next piece of the
    same insertion, continues the lead-in, and is positively of *a*'s type. An
    unclassified *a* gives no type to test, so it never extends on this basis.
    """
    ca = cls_by_path.get(a.clause_path)
    if ca is None or ca.taxonomy_id is None:
        return False
    if (
        not _redline_across_cut(a, b, insertions)
        or _starts_clause(b)
        or not _continues_across(a, b)
    ):
        return False
    tid = ca.taxonomy_id
    if _type_evidence(b, tid, cls_by_path, profiles):
        return True
    if non_clause_reason(b) != "lead_in" or c is None:
        return False
    gap_c = c.char_span[0] - b.char_span[1]
    length = len(a.text) + (b.char_span[0] - a.char_span[1]) + len(b.text) + gap_c + len(c.text)
    return (
        not c.children
        and not c.x_resegmented
        and gap_c in (0, 1)
        and _verbatim_base(c) == c.char_span[0]
        and length <= JOIN_MAX_CHARS
        and not _unjoinable(c)
        and _redline_across_cut(b, c, insertions)
        and not _starts_clause(c)
        and _continues_across(b, c)
        and _type_evidence(c, tid, cls_by_path, profiles)
    )


def _join_basis(
    a: ClauseNode,
    b: ClauseNode,
    cls_by_path: Mapping[str, ClauseClassification],
    insertions: list[tuple[int, int, str]],
    profiles: TypeProfiles,
    c: ClauseNode | None,
) -> str | None:
    """Why *b* continues *a* (one of :data:`JOIN_BASES`), or ``None``.

    *c* is the node after *b* (or ``None``); it is consulted only when *b* is a
    lead-in on the redline basis. Every basis re-applies the not-a-clause filter
    to the pieces (:data:`_UNJOINABLE_REASONS`, on *b* and on an *a* no join has
    extended yet) and refuses an unclassified *b* that the split pass's scorer
    reads as another type.
    """
    if a.children or b.children or b.x_resegmented:
        return None
    if a.x_resegmented is not None and "joined_with" not in a.x_resegmented:
        return None  # a node the split pass made is never merged on
    ca, cb = cls_by_path.get(a.clause_path), cls_by_path.get(b.clause_path)
    if ca is None or cb is None:
        return None
    if ca.taxonomy_id is None:
        same_type = _unclassified(ca) and _unclassified(cb)
    else:
        same_type = cb.taxonomy_id == ca.taxonomy_id or _unclassified(cb)
    if not same_type:
        return None
    base_b = _verbatim_base(b)
    if base_b is None or base_b != b.char_span[0] or _verbatim_base(a) is None:
        return None
    gap = b.char_span[0] - a.char_span[1]
    if gap not in (0, 1):
        return None
    if len(a.text) + gap + len(b.text) > JOIN_MAX_CHARS:
        return None
    ta, tb = a.text.rstrip(), b.text.lstrip()
    if not ta or not tb:
        return None
    # Text that is not a clause (a signature block, a running header, OCR junk)
    # is never folded into one, nor one into it.
    if _unjoinable(b) or (a.x_resegmented is None and _unjoinable(a)):
        return None
    # An unclassified piece whose own words are another clause type's is that
    # clause, not the rest of this one.
    if (
        ca.taxonomy_id is not None
        and _unclassified(cb)
        and _reads_as_other_type(b, ca.taxonomy_id, profiles)
    ):
        return None
    words_a = len(re.findall(r"[A-Za-z]{2,}", ta))
    if not _starts_clause(b) and words_a >= 3 and _ends_mid_sentence(ta):
        if tb[0].islower():
            return BASIS_LOWERCASE_START
        markers = list(_MARKER_INLINE.finditer(ta + " "))
        m_next = _MARKER_AT_START.match(tb)
        if markers and m_next and _successor(markers[-1].group(1), m_next.group(1)):
            return BASIS_NUMBERING
    # One tracked insertion cut into pieces: its pieces are one passage whether
    # or not each ends a sentence, and the segmenter's label on a piece is not a
    # clause start (only a heading in the text itself is) -- but only when each
    # piece is positively of the passage's type.
    if _redline_piece(a, b, c, cls_by_path, insertions, profiles):
        return BASIS_REDLINE
    return None


def _merge(a: ClauseNode, b: ClauseNode, basis: str) -> None:
    gap = b.char_span[0] - a.char_span[1]
    meta = a.x_resegmented or {
        "joined_with": [],
        "basis": [],
        "at": [],
        "origin_spans": [list(a.char_span)],
        "scores": {},
    }
    meta["joined_with"].append(b.clause_path)
    meta["basis"].append(basis)
    meta["at"].append(len(a.text) + gap)
    meta["origin_spans"].append(list(b.char_span))
    a.text = a.text + ("\n" * gap) + b.text
    a.char_span = (a.char_span[0], b.char_span[1])
    a.x_resegmented = meta


def rebase_join_offsets(
    meta: Mapping[str, Any], raw_text: str, alias: Callable[[str], str]
) -> dict[str, Any]:
    """*meta* (an ``x_resegmented`` record) with a join's ``at`` offsets true for aliased text.

    :func:`_merge` records ``at`` on the raw text. When the node's text is then
    rewritten (pseudonymization replaces a name with an alias of another
    length), every offset after the first rewrite points at the wrong place.
    Each offset is moved to the length of its raw prefix once *alias* has been
    applied to it. A record with no ``at`` is returned as a copy.
    """
    out = dict(meta)
    at = meta.get("at")
    if isinstance(at, list):
        out["at"] = [len(alias(raw_text[: int(a)])) for a in at]
    return out


def _join_level(
    nodes: list[ClauseNode],
    cls_by_path: dict[str, ClauseClassification],
    insertions: list[tuple[int, int, str]],
    profiles: TypeProfiles,
) -> int:
    joins = 0
    i = 0
    chain = 0
    while i < len(nodes) - 1:
        a, b = nodes[i], nodes[i + 1]
        c = nodes[i + 2] if i + 2 < len(nodes) else None
        basis = _join_basis(a, b, cls_by_path, insertions, profiles, c)
        if basis != BASIS_REDLINE and chain >= JOIN_MAX_CHAIN - 1:
            basis = None
        if basis is not None:
            _merge(a, b, basis)
            cls_by_path.pop(b.clause_path, None)
            del nodes[i + 1]
            joins += 1
            if basis != BASIS_REDLINE:
                chain += 1
            continue
        chain = 0
        i += 1
    for node in nodes:
        joins += _join_level(node.children, cls_by_path, insertions, profiles)
    return joins


def resegment(
    tree: ClauseTree,
    classified: list[ClassifiedClause],
    profiles: TypeProfiles = EMPTY_PROFILES,
    *,
    labels: Mapping[str, str] | None = None,
    tracked: TrackedChanges | None = None,
) -> ResegmentResult:
    """Split mixed nodes and rejoin truncated ones (see the module docstring).

    *tree* and *classified* are one version's clause tree and its classification
    (``classified[i]`` for the ``i``-th node of ``tree.all_nodes()``; matched by
    ``clause_path``). Neither is mutated: the result carries a new tree (nodes
    the pass did not touch are equal copies) and the matching classification
    in the new tree's order. A node the split pass re-homes gets
    ``basis="resegmented"`` at a confidence capped at :data:`CONFIDENCE_CAP`;
    every other node keeps its classification.
    """
    labels = labels or {}
    new_tree = ClauseTree.from_dict(tree.to_dict())
    cls_by_path = {cc.node.clause_path: cc.classification for cc in classified}

    splits = 0
    for node in list(new_tree.all_nodes()):
        own = cls_by_path.get(node.clause_path)
        if own is None or own.taxonomy_id is None or not profiles.targets:
            continue
        outcome = _split_node(node, own.taxonomy_id, own, profiles, labels)
        if outcome is None:
            continue
        children, added, n = outcome
        node.children = [*children, *node.children]
        cls_by_path.update(added)
        splits += n

    joins = _join_level(new_tree.nodes, cls_by_path, _insertion_spans(tracked), profiles)

    out = [
        ClassifiedClause(node=node, classification=cls_by_path[node.clause_path])
        for node in new_tree.all_nodes()
    ]
    return ResegmentResult(tree=new_tree, classified=out, splits=splits, joins=joins)


# ---------------------------------------------------------------------------
# Not-a-clause filter
# ---------------------------------------------------------------------------

#: Closed vocabulary of ``x_non_clause`` reasons.
NON_CLAUSE_REASONS: tuple[str, ...] = (
    "numbering_only",
    "signature_block",
    "heading_or_placeholder",
    "header_footer",
    "fragment",
    "lead_in",
    "ocr_junk",
    "email",
)

#: Reasons of a piece no join takes in or extends: text that cannot be part of a
#: clause. A lead-in (it introduces what follows) and a fragment (the cut-off
#: rest of a sentence) are pieces of a passage, so they may join.
_UNJOINABLE_REASONS: frozenset[str] = frozenset(NON_CLAUSE_REASONS) - {"lead_in", "fragment"}

_SIG_FIELD = (
    r"(?:by|name|title|date|signature|signed|printed name|its|address|phone(?: number)?|"
    r"authorized signatory)"
)
_SIG_LINE_RE = re.compile(rf"(?im)^\s*{_SIG_FIELD}\s*(?:[:;_\\|]|$)")
#: Phrases only a signature page, an execution line or an e-signature
#: certificate carries.
_SIG_STRONG_RE = re.compile(
    r"(?i)\b(?:docusign|envelope id|certificate of completion|signature certificate|"
    r"in witness whereof|adobe sign|hellosign|signed by:|/s/)"
)
#: Phrases a clause about signing (counterparts, e-signature) uses as well; they
#: mark a signature block only when the text states nothing operative.
_SIG_WEAK_RE = re.compile(
    r"(?i)\b(?:electronically signed|e-?signature|digital signature|signature pages?)"
)
#: A sentence that says what must, may or is so: a clause, not a signature block.
_OPERATIVE_RE = re.compile(
    r"(?i)\b(?:shall|must|may|will|should|can|cannot|required|deemed|binding|enforceable|"
    r"constitutes?|agrees?|counterparts?)\b"
)
_EMAIL_HEADER_RE = re.compile(r"(?im)^\s*(?:from|to|cc|bcc|subject|sent)\s*:")
_EMAIL_ONLY_RE = re.compile(r"^\s*<?[\w.+-]+@[\w-]+(?:\.[\w-]+)+>?\s*$")
_HEADER_FOOTER_RE = re.compile(
    r"(?i)^\W*(?:page\s+\d+(?:\s+of\s+\d+)?|\d+\s+of\s+\d+|confidential(?:\s+and\s+proprietary)?|"
    r"privileged(?:\s*(?:&|and)\s*confidential)?|draft|\[?logo\]?|all\s+rights\s+reserved\.?|"
    r"(?:copyright|©).{0,60}|version\s+[\d.]+|v\d+(?:\.\d+)*|rev(?:ision)?\.?\s*[\w.]+)\W*$"
)
_PLACEHOLDER_RE = re.compile(
    r"(?i)^\W*(?:\d+(?:\.\d+)*\.?\s*)?[\[(]?\s*(?:intentionally\s+(?:omitted|left\s+blank|blank)|"
    r"reserved|omitted|not\s+used|n/?a)\s*[\])]?\W*$"
)
_PLACEHOLDER_TAIL_RE = re.compile(
    r"(?i)[\[(]\s*(?:intentionally\s+(?:omitted|left\s+blank|blank)|reserved|omitted|not\s+used)"
    r"\s*[\])]\W*$"
)
_ROMAN_MARKER_RE = re.compile(
    r"^\W*\(?(?=[ivx])x{0,3}(?:ix|iv|v?i{0,3})\)?[.)]?\W*$", re.IGNORECASE
)
#: A text that stops on one of these (or a comma) was cut off mid-sentence.
_DANGLING_WORDS = frozenset(
    {"a", "an", "and", "as", "at", "by", "for", "from", "in", "of", "on", "or", "that", "the",
     "to", "with", "which", "who", "whose", "if", "than", "but", "nor", "any", "each", "its", "their"}
)  # fmt: skip
_MODAL_RE = re.compile(r"(?i)\b(?:shall|must|may|will|should|can|cannot|required)\b")


def non_clause_reason(node: ClauseNode) -> str | None:
    """Why *node* is not a clause (one of :data:`NON_CLAUSE_REASONS`), or ``None``.

    Judged on the node's own text and heading alone; the caller applies it to
    nodes no classification path assigned a type. Conservative on purpose: a
    node that is flagged is kept out of the classify queue, so only text that
    states nothing a clause could state is flagged (a sentence with a modal
    verb is never a lead-in; a heading is a title, not a sentence; a phrase about
    signing marks a signature block only when nothing operative is stated, so a
    counterparts or e-signature clause stays a clause).
    """
    text = (node.text or "").strip()
    heading = (node.heading or "").strip()
    whole = text or heading
    if not whole:
        return None if node.children else "heading_or_placeholder"
    words = whole.split()
    letters = re.findall(r"[A-Za-z]{2,}", whole)
    sig_lines = len(_SIG_LINE_RE.findall(whole))
    if (sig_lines >= 2 and len(words) <= 80) or (sig_lines >= 1 and len(words) <= 12):
        return "signature_block"
    if len(words) <= 150 and _SIG_STRONG_RE.search(whole):
        return "signature_block"
    if len(words) <= 150 and _SIG_WEAK_RE.search(whole):
        # Outside the signature-field lines, does the text state anything?
        body = "\n".join(ln for ln in whole.split("\n") if not _SIG_LINE_RE.match(ln))
        if not _OPERATIVE_RE.search(body):
            return "signature_block"
    if len(_EMAIL_HEADER_RE.findall(whole)) >= 2 or _EMAIL_ONLY_RE.match(whole):
        return "email"
    if _HEADER_FOOTER_RE.match(whole):
        return "header_footer"
    if _PLACEHOLDER_RE.match(whole) or (len(words) <= 14 and _PLACEHOLDER_TAIL_RE.search(whole)):
        return "heading_or_placeholder"
    non_space = [c for c in whole if not c.isspace()]
    if not letters or _ROMAN_MARKER_RE.match(whole):
        # A bare section number or list marker; a longer run of stray glyphs
        # and single letters with no digit in it is extraction junk.
        if any(c.isdigit() for c in whole) or len(non_space) <= 6:
            return "numbering_only"
        return "ocr_junk"
    alpha = sum(c.isalpha() for c in non_space)
    avg = sum(len(w) for w in words) / len(words)
    if len(non_space) >= 4 and (alpha / len(non_space) < 0.5 or (len(words) >= 6 and avg < 2.0)):
        return "ocr_junk"
    if not text or looks_like_heading(whole):
        return "heading_or_placeholder"
    ends_colon = whole.rstrip().endswith(":")
    if ends_colon and len(words) <= LEAD_IN_MAX_WORDS and not _MODAL_RE.search(whole):
        return "lead_in"
    dangling = words[-1].lower().strip(",") in _DANGLING_WORDS or whole.rstrip().endswith(",")
    if len(words) <= FRAGMENT_MAX_WORDS and not ends_colon and (whole[0].islower() or dangling):
        return "fragment"
    return None


def mark_non_clauses(classified: list[ClassifiedClause]) -> list[ClassifiedClause]:
    """Set ``x_non_clause`` on every node no path classified that is not a clause.

    Only ``basis="unclassified"`` nodes with no type are considered (a pending
    or failed judge verdict is not an answer). Nodes are marked in place and
    returned in the same order; none is removed.
    """
    for cc in classified:
        cls = cc.classification
        if cls.taxonomy_id is not None or cls.basis != "unclassified":
            continue
        reason = non_clause_reason(cc.node)
        if reason is not None:
            cc.node.x_non_clause = reason
    return classified


# ---------------------------------------------------------------------------
# Agent check (resegment_check)
# ---------------------------------------------------------------------------

_CONTEXT_CHARS = 400


@dataclass(frozen=True)
class ResegmentCheckSubject:
    """One ``resegment_check`` question; ``payload`` is exactly what the judge sees."""

    key: str
    payload: dict[str, Any]


def _payload_key(payload: Any) -> str:
    """The pending-queue key of *payload* (same hash as ``agent_judge._payload_key``)."""
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def _tree_subjects(tree: ClauseTree) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Every recorded split and join of *tree*, as ``(payload, identity)``.

    The payload is the question exactly as the judge sees it (its hash is the
    verdict key). The identity is the same question built from the
    whitespace-squashed source text, a split's context window cut only after
    squashing: two versions whose texts differ only in whitespace, even inside
    that window (where the raw cut would shift), share it.
    """
    payloads: list[tuple[dict[str, Any], dict[str, Any]]] = []
    by_path = {n.clause_path: n for n in tree.all_nodes()}
    for node in tree.all_nodes():
        meta = node.x_resegmented
        if not isinstance(meta, dict):
            continue
        if "split_from" in meta and isinstance(meta.get("to"), dict):
            origin = by_path.get(str(meta["split_from"]))
            siblings = origin.children if origin is not None else []
            before = origin.text if origin is not None else ""
            for sib in siblings:
                if sib is node:
                    break
                before = sib.text
            payload: dict[str, Any] = {
                "stage": CHECK_KIND,
                "check": "split",
                "question": (
                    "Is the text below really a provision of the clause type given, "
                    "rather than of the clause it was cut out of?"
                ),
                "clause_type": {
                    "id": str(meta["to"].get("id")),
                    "label": str(meta["to"].get("label")),
                },
                "text": node.text,
                "context_before": before[-_CONTEXT_CHARS:],
            }
            identity = {
                **_squash_whitespace(payload),
                "context_before": _squash_whitespace(before)[-_CONTEXT_CHARS:],
            }
            payloads.append((payload, identity))
        elif isinstance(meta.get("at"), list) and meta.get("joined_with"):
            cuts = [0, *[int(a) for a in meta["at"]], len(node.text)]
            parts = [node.text[a:b].strip() for a, b in zip(cuts, cuts[1:], strict=False)]
            payload = {
                "stage": CHECK_KIND,
                "check": "join",
                "question": (
                    "Do these consecutive passages form ONE clause, each continuing the one before?"
                ),
                "passages": parts,
            }
            payloads.append((payload, _squash_whitespace(payload)))
    return payloads


def _squash_whitespace(value: Any) -> Any:
    """*value* with every run of whitespace (a no-break space, a double space, a
    line break) in every string collapsed to one space; lists and dicts recursively."""
    if isinstance(value, str):
        return " ".join(value.split())
    if isinstance(value, list):
        return [_squash_whitespace(v) for v in value]
    if isinstance(value, dict):
        return {k: _squash_whitespace(v) for k, v in value.items()}
    return value


def check_subjects(out_dir: Path, *, limit: int = CHECK_SAMPLE_MAX) -> list[ResegmentCheckSubject]:
    """The ``resegment_check`` sample of a mined *out_dir*.

    Reads the (born-safe, already pseudonymized) ``normalized/`` clause trees,
    turns every recorded split and join into a question, drops duplicates and
    keeps the *limit* questions with the lowest payload hash -- all of them when
    there are *limit* or fewer. A duplicate is the same text in several
    versions, or the same text differing only in whitespace (the versions of a
    document extract a no-break space or a double space differently), also
    inside a split's context window: two questions are one passage when their
    payloads are equal or their identities are (:func:`_tree_subjects`), and
    one passage is checked once, as the variant with the lowest hash. A pure
    function of the trees: the sample is the same on every run over the same
    store.
    """
    payloads: dict[str, dict[str, Any]] = {}
    # Union-find over payload keys, always rooted at the lower key: a passage's
    # root is its variant with the lowest hash.
    parent: dict[str, str] = {}
    first_of: dict[str, str] = {}  # identity key -> the first payload key seen with it

    def passage(key: str) -> str:
        while parent[key] != key:
            key = parent[key]
        return key

    for path in sorted((out_dir / "normalized").glob("*/*.clauses.json")):
        try:
            tree = ClauseTree.from_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for payload, identity in _tree_subjects(tree):
            key = _payload_key(payload)
            payloads.setdefault(key, payload)
            parent.setdefault(key, key)
            a, b = passage(key), passage(first_of.setdefault(_payload_key(identity), key))
            if a != b:
                parent[max(a, b)] = min(a, b)
    keys = sorted({passage(key) for key in payloads})[: max(0, limit)]
    return [ResegmentCheckSubject(key=k, payload=payloads[k]) for k in keys]


def validate_check_verdict(verdict: Mapping[str, Any]) -> None:
    """Check a producer-supplied ``resegment_check`` verdict.

    ``{"resegment_ok": true | false, "reason": "<optional short text>"}`` —
    ``true`` when the split puts the text under the right type (or the passages
    are one clause), ``false`` when it does not.
    """
    ok = verdict.get("resegment_ok")
    if not isinstance(ok, bool):
        raise ValueError("'resegment_ok' must be a JSON boolean")
    reason = verdict.get("reason")
    if reason is not None and not isinstance(reason, str):
        raise ValueError("'reason' must be a string when given")
