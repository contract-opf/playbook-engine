"""Agent-judged sentence-level sub-split of multi-topic nodes (issue #257).

The re-segmentation pass (:mod:`playbook_engine.resegmentation`, issues #255
and #256) splits a node only when a sentence is overwhelmingly another clause
type's vocabulary: a false clause boundary is worse than a mixed node, so its
word-count thresholds are set high. That leaves the common co-located pairs
mixed: a term clause that also states the renewal, an entire-agreement clause
followed by severability and amendments in the same block, termination for
cause and for convenience in one paragraph. The block-aligned agent segmenter
cannot cut inside a block, so those stay one node.

This module repairs them with a judgment, where words are not enough:

1. **Candidates** (:func:`candidate_types`, deterministic and cheap). A
   classified node is a candidate when at least one of its sentences or list
   items gives a second clause type a meaningful score: at least
   :data:`CANDIDATE_HITS` of the unit's content words, and at least
   :data:`CANDIDATE_SCORE` of them, are characteristic of a split-target type
   other than the node's own (:class:`~playbook_engine.resegmentation.TypeProfiles`:
   the words of our standard and of the classified exemplars across the
   corpus). The bar is low on purpose: the agent decides, the bar only bounds
   how many questions are asked.
2. **The question** (``subsplit`` judge kind, :func:`subject_for`). The node's
   text cut into numbered units (sentences and list items,
   :func:`subsplit_units`) plus the allowed clause types (every
   classifier-eligible taxonomy entry, id and label). The payload is keyed on
   the text and the type list only. The agent answers a partition of the units
   into contiguous spans, each with a ``taxonomy_id`` or ``null``, or "no
   split" (:func:`validate_draft`).
3. **Grounding** (:func:`ground`). As the segmentation gates do for a
   segmenter's block references: the spans must cover the units exactly, in
   order, none empty, each boundary on a unit boundary (a sentence or list
   boundary by construction), each id from the allowed list, two neighbouring
   spans of different types, and the spans' text must reconstruct the node's
   text. Any failure raises :class:`~playbook_engine.segmentation_grounding.GroundingError`;
   a verdict that does not ground is never applied.
4. **Apply** (:func:`apply_subsplits`), in the re-segmentation stage after
   the split and join passes. Only a SETTLED verdict is applied: an owner's,
   or a draft whose blind check agreed or that was adjudicated
   (:mod:`playbook_engine.subsplit_check`); a draft nobody checked yet, or a
   disputed one, changes nothing. The first span stays the node it was cut
   from (its path, heading and, when the span keeps the node's type, its
   classification); every later span becomes a child node numbered
   ``<path>.s<k>``. Each node records ``x_resegmented: {subsplit_from, basis:
   "agent", origin_span, span, of, key}`` and cites a ``char_span`` inside
   the original block, so a citation still resolves to the original bytes.
   A pure function of the tree, the classification, the profiles and the
   verdict store.

A node the split or join pass already made (``x_resegmented`` set) is not a
candidate: it was cut or stitched on evidence of its own, and its provenance
record would be overwritten.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from playbook_engine.clause_classifier import ClassifiedClause, ClauseClassification
from playbook_engine.clause_tree import ClauseNode, ClauseTree
from playbook_engine.resegmentation import (
    MIN_UNIT_TOKENS,
    TypeProfiles,
    _unit_tokens,
    _verbatim_base,
    looks_like_heading,
    split_units,
)
from playbook_engine.segmentation_grounding import GroundingError

__all__ = [
    "BASIS_AGENT",
    "CANDIDATE_DOMINANCE",
    "CANDIDATE_HITS",
    "CANDIDATE_SCORE",
    "CLASSIFICATION_BASIS",
    "MAX_UNITS",
    "SUBSPLIT_CONFIDENCE",
    "SUBJECTS_FILENAME",
    "SUBSPLIT_KIND",
    "VERDICT_BASES",
    "SubsplitJudge",
    "SubsplitOutcome",
    "SubsplitSubject",
    "agreement_signature",
    "apply_subsplits",
    "candidate_types",
    "fingerprint_constants",
    "ground",
    "is_settled",
    "load_subjects",
    "subject_for",
    "subsplit_units",
    "validate_draft",
    "validate_replayable",
]

#: The pending-item kind (``judge/pending.jsonl``) and verdict kind.
SUBSPLIT_KIND = "subsplit"

#: Under ``<out>/judge/``: every question the last mine asked, one
#: ``{key, payload, rubric_version}`` per line (read by the blind check).
SUBJECTS_FILENAME = "subsplit-subjects.jsonl"

#: ``x_resegmented["basis"]`` of a node a sub-split made.
BASIS_AGENT = "agent"

#: ``ClauseClassification.basis`` of a span the sub-split typed.
CLASSIFICATION_BASIS = "subsplit"

#: ``basis`` values a drafted verdict may carry.
VERDICT_BASES: tuple[str, ...] = ("agent", "judge", "owner")

CANDIDATE_DOMINANCE: float = 0.3
"""The candidate test scores units against profiles learned from the same
exemplars as the split pass's, but a word is characteristic of a type when the
type holds at least this share of its occurrences (the split pass asks 0.5:
``resegmentation.PROFILE_DOMINANCE``). Shared co-located vocabulary (a term
clause's "renew", an entire-agreement clause's "amend") is exactly what a
mixed node holds, and the strict profiles drop it."""

CANDIDATE_SCORE: float = 0.2
"""A unit makes its node a candidate when at least this share of its content
words is characteristic of a split-target type other than the node's own. A low
bar on purpose (the split pass itself moves a unit only at 0.5 and a margin):
the agent decides, this only bounds how many questions are asked. Set on the
private acceptance corpus's double-rated segmentation sample (counts only):
with these constants about three in four of its mixed texts are candidates and
about one in four of its clean ones, which the agent answers "no split"."""

CANDIDATE_HITS: int = 2
"""...and at least this many of the unit's content words are (one shared word
is chance)."""

MAX_UNITS: int = 60
"""A node of more units than this is not asked about: a passage that long is a
segmentation failure of another kind, and a partition of it is no longer one
judgment."""

SUBSPLIT_CONFIDENCE: float = 0.9
"""Confidence of a span the sub-split typed: a settled agent verdict (an
owner's, or blind-checked), so above ``AMBIGUITY_THRESHOLD``."""

_QUESTION = (
    "Does this clause text state provisions of more than one clause type? If it does, "
    "partition its units into consecutive spans, each stating one clause type, and give "
    "each span its taxonomy_id from the list (null when none fits); otherwise answer no split."
)

_UNIT_ID_RE = re.compile(r"^u(0|[1-9]\d*)$")
_DRAFT_KEYS = frozenset({"split", "spans", "reason", "basis"})
_SPAN_KEYS = frozenset({"start", "end", "taxonomy_id"})


# ---------------------------------------------------------------------------
# Units and candidates
# ---------------------------------------------------------------------------


def subsplit_units(text: str) -> list[tuple[int, int]]:
    """The units of *text* the question numbers: sentences and list items.

    :func:`~playbook_engine.resegmentation.split_units`, except that a leading
    heading line is folded into the unit after it, so no span can be a bare
    heading. Offsets into *text*, in order, each without surrounding
    whitespace; what lies between two units is whitespace only.
    """
    units = split_units(text)
    if len(units) > 1 and looks_like_heading(text[units[0][0] : units[0][1]]):
        units = [(units[0][0], units[1][1]), *units[2:]]
    return units


def fingerprint_constants() -> list[Any]:
    """The constants the candidate test and the question depend on (folded into
    the stage cache key)."""
    return [CANDIDATE_DOMINANCE, CANDIDATE_SCORE, CANDIDATE_HITS, MAX_UNITS, MIN_UNIT_TOKENS]


def candidate_types(node: ClauseNode, own_tid: str, profiles: TypeProfiles) -> list[str]:
    """The split-target types other than *own_tid* that a unit of *node* scores
    meaningfully for (:data:`CANDIDATE_SCORE`, :data:`CANDIDATE_HITS`), sorted.

    Empty when *node* is no candidate: it has fewer than two units or more than
    :data:`MAX_UNITS`, its text does not sit verbatim at the end of its span,
    the split or join pass already made it, or no unit scores.
    """
    if node.x_resegmented is not None or node.x_non_clause is not None:
        return []
    text = node.text or ""
    if not text.strip() or _verbatim_base(node) is None:
        return []
    units = subsplit_units(text)
    if not 2 <= len(units) <= MAX_UNITS:
        return []
    found: set[str] = set()
    for s, e in units:
        toks = _unit_tokens(text[s:e])
        if len(toks) < MIN_UNIT_TOKENS:
            continue
        for tid in profiles.targets:
            if tid == own_tid:
                continue
            hits = len(toks & profiles.tokens.get(tid, frozenset()))
            if hits >= CANDIDATE_HITS and hits / len(toks) >= CANDIDATE_SCORE:
                found.add(tid)
    return sorted(found)


# ---------------------------------------------------------------------------
# The question
# ---------------------------------------------------------------------------


def _payload_key(payload: Any) -> str:
    """The verdict key of *payload* (same hash as ``agent_judge._payload_key``)."""
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()


@dataclass(frozen=True)
class SubsplitSubject:
    """One ``subsplit`` question: ``payload`` is exactly what the judge sees.

    ``rubric_version`` is the rubric in force when it was asked (the caller
    computes it once from the taxonomy; see ``rubric.rubric_version``).
    """

    key: str
    payload: dict[str, Any]
    rubric_version: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"key": self.key, "payload": self.payload, "rubric_version": self.rubric_version}


def load_subjects(judge_dir: Path) -> list[SubsplitSubject]:
    """The questions in ``<judge_dir>/subsplit-subjects.jsonl``, in key order
    (empty when the file is absent; a malformed line is skipped)."""
    path = judge_dir / SUBJECTS_FILENAME
    if not path.is_file():
        return []
    out: dict[str, SubsplitSubject] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            raw = json.loads(line)
        except ValueError:
            continue
        if not isinstance(raw, dict):
            continue
        key, payload = raw.get("key"), raw.get("payload")
        if not isinstance(key, str) or not isinstance(payload, dict):
            continue
        rv = raw.get("rubric_version")
        out[key] = SubsplitSubject(
            key=key, payload=payload, rubric_version=rv if isinstance(rv, str) else None
        )
    return [out[k] for k in sorted(out)]


def allowed_types(labels: Mapping[str, str]) -> list[dict[str, str]]:
    """The allowed clause types, ``[{id, label}]`` sorted by id."""
    return [{"id": tid, "label": labels[tid]} for tid in sorted(labels)]


def subject_for(
    text: str, allowed: Sequence[Mapping[str, str]], rubric_version: str | None = None
) -> SubsplitSubject:
    """The ``subsplit`` question about *text* under the *allowed* types."""
    units = subsplit_units(text)
    payload: dict[str, Any] = {
        "stage": SUBSPLIT_KIND,
        "question": _QUESTION,
        "text": text,
        "units": [{"id": f"u{i}", "text": text[s:e]} for i, (s, e) in enumerate(units)],
        "taxonomy": [{"id": str(t["id"]), "label": str(t["label"])} for t in allowed],
    }
    return SubsplitSubject(
        key=_payload_key(payload), payload=payload, rubric_version=rubric_version
    )


# ---------------------------------------------------------------------------
# Verdicts: shape, grounding, settlement
# ---------------------------------------------------------------------------


def _unit_index(value: Any, what: str) -> int:
    if not isinstance(value, str) or not _UNIT_ID_RE.match(value):
        raise ValueError(f"{what} must be a unit id like 'u0', got {value!r}")
    return int(value[1:])


def _validate_shape(verdict: Mapping[str, Any]) -> dict[str, Any]:
    """The judgment half of *verdict* (``split`` and ``spans``), shape-checked.

    Context-free: what can be checked without the question. Returns
    ``{"split": False}`` or ``{"split": True, "spans": [...]}``.
    """
    split = verdict.get("split")
    if not isinstance(split, bool):
        raise ValueError("'split' must be a JSON boolean")
    if not split:
        if verdict.get("spans") not in (None, []):
            raise ValueError("a 'no split' answer ('split': false) carries no spans")
        return {"split": False}
    spans = verdict.get("spans")
    if not isinstance(spans, list) or len(spans) < 2:
        raise ValueError("a split ('split': true) needs 'spans', a list of at least two spans")
    out: list[dict[str, Any]] = []
    expected = 0
    for n, span in enumerate(spans):
        if not isinstance(span, Mapping) or set(span) != _SPAN_KEYS:
            raise ValueError(f"span {n} must be exactly {{start, end, taxonomy_id}}")
        start = _unit_index(span["start"], f"span {n} 'start'")
        end = _unit_index(span["end"], f"span {n} 'end'")
        if start != expected:
            raise ValueError(
                f"span {n} starts at u{start}; spans must be contiguous, starting at u0 "
                f"(expected u{expected})"
            )
        if end < start:
            raise ValueError(f"span {n} is empty (ends at u{end} before it starts at u{start})")
        tid = span["taxonomy_id"]
        if tid is not None and (not isinstance(tid, str) or not tid):
            raise ValueError(f"span {n} 'taxonomy_id' must be a taxonomy id or null")
        if out and out[-1]["taxonomy_id"] == tid:
            raise ValueError(
                f"spans {n - 1} and {n} have the same taxonomy_id {tid!r}; one span covers "
                "a run of one type"
            )
        out.append({"start": f"u{start}", "end": f"u{end}", "taxonomy_id": tid})
        expected = end + 1
    return {"split": True, "spans": out}


def validate_draft(
    verdict: Mapping[str, Any], payload: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Validate a producer-supplied ``subsplit`` verdict (``judge-apply --verdicts``).

    ``{"split": false, "reason", "basis"}`` or ``{"split": true, "spans":
    [{"start": "u0", "end": "u2", "taxonomy_id": "term"}, ...], "reason",
    "basis"}`` and nothing else (a check is recorded only by ``judge-apply
    --check``). With the question's *payload* (from ``pending.jsonl``) the
    spans are grounded against it too (:func:`ground`). Returns the verdict as
    it is stored; raises ``ValueError`` (a :class:`GroundingError` is
    re-raised as one) naming the first problem.
    """
    extra = sorted(set(verdict) - _DRAFT_KEYS)
    if extra:
        raise ValueError(
            f"unexpected field(s) {extra} on a subsplit verdict; a verdict carries only "
            "split, spans, reason and basis (checks are recorded with `judge-apply --check`)"
        )
    judgment = _validate_shape(verdict)
    reason = verdict.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("'reason' must be a non-empty string (one sentence)")
    basis = verdict.get("basis")
    if basis not in VERDICT_BASES:
        raise ValueError(f"'basis' must be one of {list(VERDICT_BASES)}, got {basis!r}")
    if payload is not None:
        try:
            ground(judgment, payload)
        except GroundingError as exc:
            raise ValueError(f"grounding: {exc}") from exc
    return {**judgment, "reason": reason.strip(), "basis": basis}


def validate_replayable(verdict: Mapping[str, Any]) -> None:
    """Raise ``ValueError`` when a stored *verdict* could not be replayed (any
    trail fields a check added are allowed)."""
    _validate_shape(verdict)
    if verdict.get("basis") not in VERDICT_BASES:
        raise ValueError(f"'basis' must be one of {list(VERDICT_BASES)}")


def ground(
    judgment: Mapping[str, Any], payload: Mapping[str, Any]
) -> list[tuple[int, int, str | None]]:
    """Ground a split *judgment* against the question *payload*.

    Returns ``[(first unit, last unit, taxonomy_id), ...]`` (empty for "no
    split"). Raises :class:`GroundingError` when the spans do not cover the
    question's units exactly and in order, a span is empty, an id is not in the
    allowed list, two neighbouring spans share a type, or the spans' text does
    not reconstruct the node's text.
    """
    try:
        shaped = _validate_shape(judgment)
    except ValueError as exc:
        raise GroundingError(str(exc)) from exc
    if not shaped["split"]:
        return []
    units = payload.get("units")
    text = payload.get("text")
    allowed = {str(t.get("id")) for t in payload.get("taxonomy") or [] if isinstance(t, Mapping)}
    if not isinstance(units, list) or not isinstance(text, str):
        raise GroundingError("the question carries no units to ground against")
    spans = [(int(s["start"][1:]), int(s["end"][1:]), s["taxonomy_id"]) for s in shaped["spans"]]
    if spans[-1][1] != len(units) - 1:
        raise GroundingError(
            f"the spans end at u{spans[-1][1]} but the text has units u0..u{len(units) - 1}; "
            "they must cover the text exactly"
        )
    for _start, _end, tid in spans:
        if tid is not None and tid not in allowed:
            raise GroundingError(f"taxonomy_id {tid!r} is not in the allowed list")
    unit_texts = [u.get("text") if isinstance(u, Mapping) else None for u in units]
    # Reconstruction gate (as segmentation_qa's): the units, whitespace aside,
    # are the text; and they are exactly its sentences and list items, so
    # every span boundary is a sentence or list boundary.
    rebuilt = "".join(t for t in unit_texts if isinstance(t, str))
    if re.sub(r"\s+", "", rebuilt) != re.sub(r"\s+", "", text):
        raise GroundingError("the units do not reconstruct the text")
    if unit_texts != [text[s:e] for s, e in subsplit_units(text)]:
        raise GroundingError("the units are not the text's sentences and list items")
    return spans


def agreement_signature(verdict: Mapping[str, Any]) -> tuple[Any, ...]:
    """What two judgments must share to agree: the split decision and, for a
    split, every span's boundaries and type."""
    judgment = _validate_shape(verdict)
    if not judgment["split"]:
        return (False,)
    return (True, *((s["start"], s["end"], s["taxonomy_id"]) for s in judgment["spans"]))


def is_settled(verdict: Mapping[str, Any]) -> bool:
    """Whether a stored *verdict* is applied: an owner's, or a draft whose blind
    check agreed or that was adjudicated (``equivalence.check_state``)."""
    if verdict.get("basis") == "owner":
        return True
    check = verdict.get("check")
    if not isinstance(check, Mapping):
        return False
    return check.get("adjudicated") is True or check.get("agreed") is True


# ---------------------------------------------------------------------------
# Apply
# ---------------------------------------------------------------------------


class SubsplitJudge(Protocol):
    """Answers a ``subsplit`` question: the settled verdict, or ``None``."""

    def judge(self, subject: SubsplitSubject) -> dict[str, Any] | None: ...


@dataclass
class SubsplitOutcome:
    """What :func:`apply_subsplits` hands back for one version."""

    tree: ClauseTree
    classified: list[ClassifiedClause]
    #: Nodes asked about (or that would be, with a judge).
    candidates: int = 0
    #: Nodes a settled verdict split.
    split: int = 0
    #: Nodes a settled verdict left whole ("no split").
    no_split: int = 0
    #: Nodes added (the spans after the first of every split node).
    spans_added: int = 0
    #: The questions asked, one per distinct key, in key order.
    subjects: list[SubsplitSubject] = field(default_factory=list)


def _span_classification(
    tid: str | None, own_tid: str, own: ClauseClassification
) -> ClauseClassification:
    if tid is None:
        return ClauseClassification(taxonomy_id=None, confidence=0.0, basis="unclassified")
    if tid == own_tid:
        return own
    return ClauseClassification(
        taxonomy_id=tid, confidence=SUBSPLIT_CONFIDENCE, basis=CLASSIFICATION_BASIS
    )


def _split_node(
    node: ClauseNode,
    own_tid: str,
    own: ClauseClassification,
    spans: list[tuple[int, int, str | None]],
    key: str,
) -> tuple[list[ClauseNode], dict[str, ClauseClassification]] | None:
    """Cut *node* in place at the grounded *spans*; ``None`` (untouched) when the
    node's own heading would be cut."""
    base = _verbatim_base(node)
    if base is None:
        return None
    text = node.text
    units = subsplit_units(text)
    if spans[-1][1] != len(units) - 1:
        return None
    first_end = units[spans[0][1]][1]
    if node.heading_span is not None and node.heading_span[1] > base + first_end:
        return None
    origin = [node.char_span[0], node.char_span[1]]
    path = node.clause_path

    def meta(k: int) -> dict[str, Any]:
        return {
            "subsplit_from": path,
            "basis": BASIS_AGENT,
            "origin_span": list(origin),
            "span": k,
            "of": len(spans),
            "key": key,
        }

    children: list[ClauseNode] = []
    cls_by_path: dict[str, ClauseClassification] = {
        path: _span_classification(spans[0][2], own_tid, own)
    }
    for k, (i0, i1, tid) in enumerate(spans[1:], start=1):
        s, e = units[i0][0], units[i1][1]
        child_path = f"{path}.s{k}"
        children.append(
            ClauseNode(
                clause_path=child_path,
                heading=None,
                text=text[s:e],
                char_span=(base + s, base + e),
                page=node.page,
                x_resegmented=meta(k),
            )
        )
        cls_by_path[child_path] = _span_classification(tid, own_tid, own)
    node.text = text[:first_end]
    node.char_span = (node.char_span[0], base + first_end)
    node.x_resegmented = meta(0)
    return children, cls_by_path


def apply_subsplits(
    tree: ClauseTree,
    classified: list[ClassifiedClause],
    profiles: TypeProfiles,
    *,
    labels: Mapping[str, str],
    judge: SubsplitJudge | None = None,
    rubric_version: str | None = None,
) -> SubsplitOutcome:
    """Ask about every candidate node of *tree* and apply the settled answers.

    *classified* is the version's classification (matched by ``clause_path``);
    *labels* the allowed clause types (``{id: label}``, every
    classifier-eligible entry). Without a *judge* the candidates are counted
    and nothing changes. Neither input is mutated: the outcome carries a new
    tree and the classification in its node order. Deterministic: a pure
    function of its inputs and the judge's answers.
    """
    new_tree = ClauseTree.from_dict(tree.to_dict())
    cls_by_path = {cc.node.clause_path: cc.classification for cc in classified}
    allowed = allowed_types(labels)
    outcome = SubsplitOutcome(tree=new_tree, classified=[])
    subjects: dict[str, SubsplitSubject] = {}
    if profiles.targets:
        for node in list(new_tree.all_nodes()):
            own = cls_by_path.get(node.clause_path)
            if own is None or own.taxonomy_id is None:
                continue
            if not candidate_types(node, own.taxonomy_id, profiles):
                continue
            outcome.candidates += 1
            if judge is None:
                continue
            subject = subject_for(node.text, allowed, rubric_version)
            subjects.setdefault(subject.key, subject)
            verdict = judge.judge(subject)
            if verdict is None:
                continue
            try:
                spans = ground(verdict, subject.payload)
            except GroundingError:
                continue
            if not spans:
                outcome.no_split += 1
                continue
            cut = _split_node(node, own.taxonomy_id, own, spans, subject.key)
            if cut is None:
                continue
            children, added = cut
            node.children = [*children, *node.children]
            cls_by_path.update(added)
            outcome.split += 1
            outcome.spans_added += len(children)
    outcome.classified = [
        ClassifiedClause(node=node, classification=cls_by_path[node.clause_path])
        for node in new_tree.all_nodes()
    ]
    outcome.subjects = [subjects[k] for k in sorted(subjects)]
    return outcome
