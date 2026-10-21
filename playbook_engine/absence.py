"""Verified absence counts: "we signed N deals without clause X" (issue #251, OPF 0.6).

The toaster reads a new contract that lacks a clause type entirely and asks how
often we have signed without it. Counting a deal that has no precedent record
for the clause as "signed without it" is wrong whenever the clause was in the
deal but never classified (on the private acceptance corpus about one
observation in six is still unclassified), so a missed classification would read
as an absence. This module splits every (signed deal, clause type) with no
signed precedent into two facts, recorded in ``evidence.absence``:

``absent_verified``
    None of the deal's signed-text nodes scores at or above :data:`ABSENCE_GUARD`
    against the clause type, so there is no passage of the signed copy that could
    be the clause and was filed elsewhere or not at all.
``possibly_present``
    Otherwise: at least one signed-text node scores at or above the guard (the
    record cites up to :data:`MAX_CANDIDATES` of them, best first, each with the
    clause type the classifier filed it under or ``null``), or nothing could be
    checked (``basis: "unverifiable"``: the deal's executed copy produced no
    readable text or has big passages that extracted as junk, or the clause type
    has no reference text).

The nodes scored are every clause node of the executed copy: the ones the
classifier left unclassified AND the ones it filed under another clause type
(an unclassified node that is no clause, ``x_non_clause``, is left out). The
second kind matters: a clause stated inside a block the classifier filed under
a neighbouring type (a removal right inside the students' responsibilities) is
as good as unclassified for the question "is it in this deal?". The price is
that a neighbouring clause's shared vocabulary can keep a deal out of the verified
count; that errs towards ``possibly_present``, the safe side.

The score is the content-word Jaccard similarity the content-similarity
classifier uses (issue #235, :func:`~playbook_engine.clause_classifier._content_tokens`),
the best of the node as a whole and each of its paragraphs, sentences and list
items (:func:`~playbook_engine.resegmentation.split_units`; a big node that holds
the clause among others would otherwise dilute it below any threshold), against
each reference text of the clause type: the parts of our standard and of every
former form of ours, and the classified exemplars of the clause (every text
any deal signed, opened with or had refused for it). :data:`ABSENCE_GUARD` is
well below the assign threshold (:data:`~playbook_engine.clause_classifier.CONTENT_ASSIGN_THRESHOLD`,
0.25): a node that is merely *plausibly* the clause keeps a deal out of the
verified count, while only a node that is confidently the clause is assigned
to it. A node the classifier assigned to the very clause type but that was too
short to be precedent (it never reached ``evidence.precedent``) scores 1.0: it
IS the classifier's finding of the clause.

Everything is a fact, never a verdict, and a pure function of the store: two
compiles of the same store produce byte-identical records. Paper side takes no
part in it.

Agent spot-check. :func:`check_subjects` turns a deterministic sample of the
verified absences (all of them when there are :data:`CHECK_SAMPLE_MAX` or
fewer) into ``absence_check`` questions, "is clause type X really absent from
this signed text?"; the verdicts gate nothing, they measure the guard. A
confirmed-present item means the guard is too weak (a clause scoring just below
it was counted as absent): tighten it, that is, lower it, until the checked sample
has none.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from playbook_engine.canonicalize import canonicalize
from playbook_engine.clause_classifier import _content_tokens
from playbook_engine.clause_position_compiler import OPFCitation, _is_degenerate_observation_text
from playbook_engine.observation_builder import Observation
from playbook_engine.resegmentation import split_units

__all__ = [
    "ABSENCE_GUARD",
    "ABSENCE_ID_PREFIX",
    "BASES",
    "BASIS_SCORED",
    "BASIS_UNVERIFIABLE",
    "CHECK_KIND",
    "CHECK_SAMPLE_MAX",
    "MAX_CANDIDATES",
    "SCORE_DECIMALS",
    "STATUSES",
    "STATUS_POSSIBLE",
    "STATUS_VERIFIED",
    "AbsenceSubject",
    "absence_id",
    "absence_records",
    "build_absence",
    "check_subjects",
    "counts_by_clause",
    "signed_deal_ids",
    "validate_check_verdict",
]

#: The pending-item kind of the agent spot-check (``judge/pending.jsonl``).
CHECK_KIND = "absence_check"

#: A deterministic sample of at most this many verified absences is queued for
#: the check (all of them when there are this many or fewer).
CHECK_SAMPLE_MAX = 40

#: Every absence id starts with this prefix (OPF-SPEC §3.5.8).
ABSENCE_ID_PREFIX = "abs."
_ABSENCE_ID_HEX = 16

STATUS_VERIFIED = "absent_verified"
STATUS_POSSIBLE = "possibly_present"
STATUSES = (STATUS_VERIFIED, STATUS_POSSIBLE)

#: ``basis`` of a record: how the status was reached.
BASIS_SCORED = "scored"
BASIS_UNVERIFIABLE = "unverifiable"
BASES = (BASIS_SCORED, BASIS_UNVERIFIABLE)

ABSENCE_GUARD: float = 0.10
"""A signed-text node that scores at least this against a clause type keeps the
deal out of that clause's verified absences (it becomes ``possibly_present``).

Well below the content-similarity assign threshold (0.25,
``clause_classifier.CONTENT_ASSIGN_THRESHOLD``, issue #235): assigning a node to
a clause plants a precedent record, so it needs confidence; declining to *verify
an absence* only needs plausibility, and a missed classification read as an
absence is the error this guard exists to prevent. A higher guard
verifies more absences and a lower one fewer, so the spot-check (``absence_check``)
is the arbiter: lower it the moment a confirmed-present item turns up, and
relax it only while the checked sample has none. Scores are rounded to :data:`SCORE_DECIMALS` places
before they are compared with it, so the number a record shows is the number
that decided."""

UNREADABLE_NODE_CHARS = 200
"""A signed-text node of at least this many characters is *unreadable* when fewer
than :data:`UNREADABLE_MAX_WORD_SHARE` of its whitespace-separated tokens are
words (three letters or more): a font-encoding failure that extracts a page as
columns of digits. A shorter node (a signature line, a table cell) is never
unreadable."""

UNREADABLE_MAX_WORD_SHARE = 0.3

UNREADABLE_DEAL_CHARS = 1000
"""A signed copy with at least this many characters of unreadable nodes cannot
have a clause verified absent from it (the clause may sit in the text that did not
extract): every record of the deal is ``possibly_present`` with ``basis``
``unverifiable``. Measured on the private acceptance corpus: 2 of 43 signed copies
(5,030 and 19,496 unreadable characters)."""

MAX_CANDIDATES = 3
"""A ``possibly_present`` record cites at most this many candidate nodes, best
score first."""

SCORE_DECIMALS = 3
"""Scores are rounded to this many decimal places."""

MIN_WINDOW_TOKENS = 3
"""A paragraph, sentence or list item with fewer content words than this is not
scored on its own (the node as a whole always is)."""

_CLASSIFIED_SCORE = 1.0


def absence_id(agreement_type_id: str, document_id: str, taxonomy_id: str) -> str:
    """The stable id of one absence record.

    ``"abs." + sha256(canonical([agreement_type_id, document_id, taxonomy_id]))[:16]``;
    (document_id, taxonomy_id) is unique per record, and the status is a fact
    about it that may change when the corpus is re-mined, so it is not hashed."""
    payload = canonicalize([agreement_type_id, document_id, taxonomy_id])
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"{ABSENCE_ID_PREFIX}{digest[:_ABSENCE_ID_HEX]}"


def signed_deal_ids(
    corpus_documents: Iterable[Mapping[str, Any]], precedent: Iterable[Mapping[str, Any]]
) -> list[str]:
    """The in-scope deals with a detected executed copy, sorted.

    ``corpus.documents[].signed_version`` is authoritative when recorded (the
    digest's ``corpus.n_signed`` rule); a document that records none counts as
    signed when one of its precedent records is signed."""
    signed_by_precedent = {
        p.get("document_id")
        for p in precedent
        if isinstance(p, Mapping) and p.get("signed") is True
    }
    ids: set[str] = set()
    for doc in corpus_documents:
        if not isinstance(doc, Mapping) or not doc.get("in_scope", True):
            continue
        document_id = doc.get("document_id")
        if not isinstance(document_id, str):
            continue
        signed = (
            doc.get("signed_version") is not None
            if "signed_version" in doc
            else document_id in signed_by_precedent
        )
        if signed:
            ids.add(document_id)
    return sorted(ids)


def _is_unreadable(text: str) -> bool:
    """Whether *text* is a big node that extracted as junk (see :data:`UNREADABLE_NODE_CHARS`)."""
    if len(text) < UNREADABLE_NODE_CHARS:
        return False
    tokens = text.split()
    words = sum(1 for t in tokens if _WORD.fullmatch(t.strip(".,;:()[]\"'")))
    return words < UNREADABLE_MAX_WORD_SHARE * len(tokens)


_WORD = re.compile(r"[A-Za-z]{3,}")


def _windows(text: str) -> list[frozenset[str]]:
    """The token sets a node is scored on: the whole node, then each unit of it."""
    whole = _content_tokens(text)
    windows = [whole] if whole else []
    units = split_units(text)
    if len(units) > 1:
        for start, end in units:
            tokens = _content_tokens(text[start:end])
            if len(tokens) >= MIN_WINDOW_TOKENS and tokens != whole:
                windows.append(tokens)
    return windows


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter) if inter else 0.0


def _reference_texts(clause: Mapping[str, Any]) -> list[str]:
    """The standard texts of one evidence clause: our standard's parts and whole,
    and the same for every former form of ours."""
    texts: list[str] = []
    for std in (clause.get("our_standard"), *(clause.get("prior_standards") or [])):
        if not isinstance(std, Mapping):
            continue
        parts = [str(p.get("text") or "") for p in std.get("parts") or [] if isinstance(p, Mapping)]
        texts.extend(parts)
        if isinstance(std.get("text"), str):
            texts.append(std["text"])
    return [t for t in texts if t.strip()]


def _citation(obs: Observation) -> dict[str, Any]:
    return OPFCitation(
        document_id=obs.citation.document_id,
        version=obs.citation.version,
        clause_path=obs.citation.clause_path,
        char_span=obs.citation.char_span,
    ).to_dict()


def build_absence(
    *,
    agreement_type_id: str,
    clauses: Sequence[Mapping[str, Any]],
    precedent: Sequence[Mapping[str, Any]],
    observations: Sequence[Observation],
    corpus_documents: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """The ``evidence.absence`` section: ``{guard, records}``.

    One record per (signed deal, clause type) with no signed precedent, in
    (document_id, taxonomy_id) order. *clauses* are the ``evidence.clauses``
    entries (their standards are the reference texts); *precedent* the
    ``evidence.precedent`` records (what the deals signed, and the exemplars'
    signed texts); *observations* the store's rows (the signed-text nodes).
    """
    tids = [c["taxonomy_id"] for c in clauses if isinstance(c.get("taxonomy_id"), str)]
    tid_set = set(tids)
    signed_deals = signed_deal_ids(corpus_documents, precedent)
    signed_set = set(signed_deals)
    has_signed_text = {
        (p.get("document_id"), p.get("taxonomy_id"))
        for p in precedent
        if p.get("signed") is True and isinstance(p.get("signed_text"), Mapping)
    }

    # Reference token sets per clause type, deduplicated.
    references: dict[str, set[frozenset[str]]] = {tid: set() for tid in tids}
    for clause in clauses:
        tid = clause.get("taxonomy_id")
        if tid not in references:
            continue
        for text in _reference_texts(clause):
            tokens = _content_tokens(text)
            if tokens:
                references[tid].add(tokens)
    for p in precedent:
        p_tid = p.get("taxonomy_id")
        if p_tid not in references:
            continue
        for entry in (p.get("signed_text"), p.get("opening_text"), *(p.get("refused_asks") or [])):
            entry_text = entry.get("text") if isinstance(entry, Mapping) else None
            entry_tokens = (
                _content_tokens(entry_text) if isinstance(entry_text, str) else frozenset()
            )
            if entry_tokens:
                references[p_tid].add(entry_tokens)
    flat_refs = [(tid, ref) for tid in tids for ref in references[tid]]

    # The signed-text nodes of each signed deal: every row of the executed copy
    # that is a clause (an unclassified node that is no clause is left out).
    nodes: dict[str, list[Observation]] = {d: [] for d in signed_deals}
    unreadable_chars: dict[str, int] = dict.fromkeys(signed_deals, 0)
    for obs in observations:
        document_id = obs.citation.document_id
        if document_id not in signed_set or obs.outcome != "signed":
            continue
        if _is_unreadable(obs.full_text):
            unreadable_chars[document_id] += len(obs.full_text)
        if obs.non_clause is None and obs.full_text.strip():
            nodes[document_id].append(obs)

    # A signed copy with big unreadable passages cannot be verified at all.
    unreadable_deals = {
        document_id
        for document_id, chars in unreadable_chars.items()
        if chars >= UNREADABLE_DEAL_CHARS
    }

    # Score each distinct node text against every clause type once (boilerplate
    # repeats across deals).
    scores_by_text: dict[str, dict[str, float]] = {}

    def scores_of(text: str) -> dict[str, float]:
        cached = scores_by_text.get(text)
        if cached is None:
            best: dict[str, float] = {}
            for window in _windows(text):
                for tid, ref in flat_refs:
                    s = _jaccard(window, ref)
                    if s > best.get(tid, 0.0):
                        best[tid] = s
            cached = {tid: round(s, SCORE_DECIMALS) for tid, s in best.items()}
            scores_by_text[text] = cached
        return cached

    records: list[dict[str, Any]] = []
    for document_id in signed_deals:
        for tid in sorted(tid_set):
            if (document_id, tid) in has_signed_text:
                continue
            candidates: list[tuple[float, Observation]] = []
            checked = 0
            for node in nodes[document_id]:
                if node.taxonomy_id == tid:
                    # Filed under this very clause but too short to be precedent
                    # (a longer one has a precedent record and no absence record).
                    if _is_degenerate_observation_text(node.full_text):
                        candidates.append((_CLASSIFIED_SCORE, node))
                        checked += 1
                    continue
                checked += 1
                score = scores_of(node.full_text).get(tid, 0.0)
                if score >= ABSENCE_GUARD:
                    candidates.append((score, node))
            candidates.sort(
                key=lambda c: (-c[0], c[1].citation.clause_path, str(c[1].citation.version))
            )
            unverifiable = (
                not nodes[document_id] or not references[tid] or document_id in unreadable_deals
            )
            records.append(
                {
                    "id": absence_id(agreement_type_id, document_id, tid),
                    "taxonomy_id": tid,
                    "document_id": document_id,
                    "status": STATUS_VERIFIED
                    if not (candidates or unverifiable)
                    else STATUS_POSSIBLE,
                    "basis": BASIS_UNVERIFIABLE if unverifiable else BASIS_SCORED,
                    "n_checked": checked,
                    "candidates": [
                        {
                            "ref": _citation(node),
                            "score": score,
                            "filed_under": node.taxonomy_id,
                        }
                        for score, node in candidates[:MAX_CANDIDATES]
                    ],
                }
            )
    return {"guard": ABSENCE_GUARD, "records": records}


def absence_records(playbook: Mapping[str, Any]) -> list[dict[str, Any]]:
    """``evidence.absence.records`` of a document (``[]`` when absent)."""
    evidence = playbook.get("evidence")
    section = evidence.get("absence") if isinstance(evidence, Mapping) else None
    records = section.get("records") if isinstance(section, Mapping) else None
    return [r for r in records if isinstance(r, dict)] if isinstance(records, list) else []


def counts_by_clause(records: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    """``{taxonomy_id: {verified, possible}}`` over *records* (distinct deals)."""
    deals: dict[str, dict[str, set[Any]]] = {}
    for record in records:
        tid = record.get("taxonomy_id")
        if not isinstance(tid, str):
            continue
        bucket = deals.setdefault(tid, {STATUS_VERIFIED: set(), STATUS_POSSIBLE: set()})
        if record.get("status") in bucket:
            bucket[record["status"]].add(record.get("document_id"))
    return {
        tid: {
            "verified": len(b[STATUS_VERIFIED]),
            "possible": len(b[STATUS_POSSIBLE]),
        }
        for tid, b in sorted(deals.items())
    }


# ---------------------------------------------------------------------------
# Agent spot-check (absence_check)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AbsenceSubject:
    """One ``absence_check`` question; ``payload`` is exactly what the judge sees.

    The payload carries the clause type, our standard and the deal's whole signed
    text, and no document id, counterparty or paper side. ``absence_id`` names
    the record the question was built from, for the operator's own bookkeeping;
    it is not part of the payload or the key."""

    key: str
    payload: dict[str, Any]
    absence_id: str


def _payload_key(payload: Any) -> str:
    """The pending-queue key of *payload* (same hash as ``agent_judge._payload_key``)."""
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def check_subjects(
    *,
    clauses: Sequence[Mapping[str, Any]],
    records: Iterable[Mapping[str, Any]],
    observations: Sequence[Observation],
    limit: int = CHECK_SAMPLE_MAX,
) -> list[AbsenceSubject]:
    """The ``absence_check`` sample of one compile.

    Every ``absent_verified`` record becomes a question (the clause type, our
    standard, and the deal's signed text as an ordered list of its nodes, each
    with the clause-type label the classifier gave it or ``null``); duplicates
    (the same question for two deals with the same signed text) are asked once,
    and the *limit* questions with the lowest payload hash are kept, all of them
    when there are *limit* or fewer. A pure function of the compile: the sample is
    the same on every run over the same store.
    """
    records = list(records)
    clause_by_tid = {c["taxonomy_id"]: c for c in clauses if isinstance(c.get("taxonomy_id"), str)}
    label_of = {tid: str(c.get("title") or tid) for tid, c in clause_by_tid.items()}
    nodes_by_doc: dict[str, list[dict[str, Any]]] = {}
    wanted = {r.get("document_id") for r in records if r.get("status") == STATUS_VERIFIED}
    for obs in observations:
        document_id = obs.citation.document_id
        if document_id not in wanted or obs.outcome != "signed" or not obs.full_text.strip():
            continue
        nodes_by_doc.setdefault(document_id, []).append(
            {
                "clause_type": label_of.get(obs.taxonomy_id, obs.taxonomy_id)
                if obs.taxonomy_id
                else None,
                "text": obs.full_text,
            }
        )
    questions: dict[str, AbsenceSubject] = {}
    for record in records:
        if record.get("status") != STATUS_VERIFIED:
            continue
        rec_tid, rec_doc = record.get("taxonomy_id"), record.get("document_id")
        clause = clause_by_tid.get(rec_tid) if isinstance(rec_tid, str) else None
        if clause is None or not isinstance(rec_doc, str):
            continue
        std = clause.get("our_standard")
        payload = {
            "stage": CHECK_KIND,
            "question": (
                "Is the clause type below really absent from this signed text, "
                "that is, does no passage of it state this clause?"
            ),
            "clause_type": {"id": str(rec_tid), "label": label_of[str(rec_tid)]},
            "our_standard": std.get("text") if isinstance(std, Mapping) else None,
            "signed_text": nodes_by_doc.get(rec_doc, []),
        }
        key = _payload_key(payload)
        questions.setdefault(
            key, AbsenceSubject(key=key, payload=payload, absence_id=str(record.get("id")))
        )
    keys = sorted(questions)[: max(0, limit)]
    return [questions[k] for k in keys]


def validate_check_verdict(verdict: Mapping[str, Any]) -> None:
    """Check a producer-supplied ``absence_check`` verdict.

    ``{"absent": true | false, "reason": "<optional short text>"}`` -- ``true``
    when the clause type is really absent from the signed text, ``false`` when a
    passage of it states the clause (a confirmed-present item: the guard is too
    high).
    """
    absent = verdict.get("absent")
    if not isinstance(absent, bool):
        raise ValueError("'absent' must be a JSON boolean")
    reason = verdict.get("reason")
    if reason is not None and not isinstance(reason, str):
        raise ValueError("'reason' must be a string when given")
