"""Reconstructing our former forms from the corpus (issue #250, OPF 0.6).

When the owner supplies no older form files (``baseline.prior_templates``,
issue #246) our former wording is still in the corpus: it is what we SENT. A
deal that opened on last year's indemnity has that indemnity as its first
draft, and several deals opened on the same one. This module rebuilds those
forms from the first drafts, flags them ``basis: "reconstructed"`` and hands
them to the same machinery as supplied forms (issue #249): the clause becomes
a ``prior_standards[]`` entry, judged against today's standard, and the deals
that opened on it read ``opened_with: "prior_standard"``.

What is a reconstructed prior form clause
-----------------------------------------

Per clause type with an ``our_standard`` the candidates are the first drafts
of signed deals on OUR paper that are not today's standard (``opened_with``
is ``non_standard``). A first draft is the opening row's text, else the signed
text (a deal that signed what it opened with has no distinct opening). They are
clustered by a deterministic method (:func:`cluster_first_drafts`):

* the text of a draft is its OPF-SPEC §3.5.4 grouping key
  (:func:`~playbook_engine.precedent.normalize_variant_text`: case,
  punctuation and whitespace dropped, counterparty aliases and our party
  neutralized), split into words;
* two wordings are near-identical when the Jaccard similarity of their word
  3-gram sets is at least :data:`CLUSTER_SIMILARITY` AND they carry the same
  numbers and the same count of negators (a changed amount or a flipped
  negation is never near-identical, whatever the rest says);
* wordings are visited most-deals-first (then by text) and each joins the
  first cluster whose seed it matches, else seeds a new one — no randomness,
  no dependence on input order.

A cluster becomes a reconstructed clause when it appears as the first draft of
at least :data:`MIN_DEALS` distinct deals and its **medoid** (the member
wording closest to all the others, weighted by deals) is the first draft of at
least :data:`DOMINANT_SHARE` of the cluster's deals — the members share one
dominant text. The medoid is the clause's text; it differs from today's
standard by construction (and is checked).

Forms
-----

Clause-level reconstructions that co-occur in the same deals are one form
(:func:`group_forms`): two clusters of different clause types join when the
smaller one's deals are at least :data:`FORM_OVERLAP` inside the other's and
their signing-date spans overlap (or one has no dates), as long as the form
never holds two clusters of one clause type. Where nothing co-occurs a cluster
stays a form of its own, per clause. A form is labelled by what is known of it
(``reconstructed 2021-03 to 2022-11``): never a year the data does not give,
and no claim of why it changed.

Which deals opened on a form
----------------------------

Paper side chooses the CANDIDATES a form is built from and nothing else. Once a
cluster is confirmed, a deal opened on the form when its first draft of the
clause has the same grouping key as the medoid: the exact check a supplied form
uses (issue #249), applied to every signed ``non_standard`` opening of the
clause type on ANY paper side. The form's ``n_deals`` and date span count those
deals.

Our former wording struck before signing
----------------------------------------

The other thing a supplied form does when mining (issue #249) is done here too:
a clause of ours struck before signing is no refused ask. A
``proposed_then_reversed`` row that is not today's standard and whose text is a
confirmed form clause's wording, whole or one contiguous piece of it, is
dropped and counted under ``corpus.stats.dropped_observations``
``removed_prior_standard`` (:meth:`Reconstruction.corpus_documents_with_drops`).
The test is the one mining applies to a supplied form
(:func:`~playbook_engine.observation_builder.is_former_form_language`, the same
party names), and the same test under the §3.5.4 grouping key, the
reconstructed wording's own identity: ``project`` reads the stored text, where
each deal's counterparty is its own alias. A reversal row of that wording is
dropped too, so no refused ask reads as a reconstructed clause (the validator
enforces it).

Origin check
------------

A recurring first draft on our paper might still be a counterparty's wording
(a form we agreed to reuse, a paper side misread). Each qualifying cluster is
put to the ``prior_form_origin`` judge kind (:class:`OriginSubject`): "is this
really OUR former language?". It is drafted by an agent, blind-checked and, on
disagreement, adjudicated (:mod:`playbook_engine.prior_form_origin_check`).
**Only confirmed clusters become prior standards** (:func:`outcome_of`): a
cluster with no verdict, a drafted-but-unchecked one, or a disputed one is
reported as pending and changes nothing in the playbook. The payload and the
key carry no document, counterparty or paper side.

Supplied forms always win: when any form was supplied nothing is
reconstructed (:func:`reconstruct`).
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from playbook_engine.deal_dates import signed_on_key
from playbook_engine.equivalence import (
    BASIS_AGENT,
    BASIS_JUDGE,
    BASIS_OWNER,
    SETTLED_ADJUDICATED,
    SETTLED_AGREED,
    SETTLED_OWNER,
    check_state,
    standard_parts,
)
from playbook_engine.observation_builder import (
    DROPPED_PRIOR_STANDARD_REMOVED,
    OPENED_NON_STANDARD,
    OPENED_PRIOR_STANDARD,
    Observation,
    ObservationCitation,
    is_former_form_language,
    summarize_clause_text,
)
from playbook_engine.precedent import PAPER_OURS, build_precedent_evidence, normalize_variant_text
from playbook_engine.prior_standards import BASIS_RECONSTRUCTED, PriorForm, origin_standards

__all__ = [
    "CLUSTER_SIMILARITY",
    "DOMINANT_SHARE",
    "FORM_OVERLAP",
    "MIN_DEALS",
    "ORIGIN_BASES",
    "ORIGIN_KIND",
    "SKIPS",
    "SKIP_NO_TEMPLATE",
    "SKIP_SUPPLIED",
    "STATE_CONFIRMED",
    "STATE_PENDING",
    "STATE_REJECTED",
    "Cluster",
    "ClusterOutcome",
    "FirstDraft",
    "OriginSubject",
    "Reconstruction",
    "cluster_first_drafts",
    "first_drafts",
    "group_forms",
    "origin_key",
    "origin_subjects",
    "outcome_of",
    "reconstruct",
    "similarity",
    "validate_origin_draft",
    "verdict_of",
]

#: The pending-item / rubric kind of the origin judge.
ORIGIN_KIND = "prior_form_origin"

#: A reconstructed clause needs at least this many distinct deals whose first
#: draft is (near-identically) its wording.
MIN_DEALS = 3

#: Two wordings are near-identical at this Jaccard similarity of their word
#: 3-gram sets or above (and the same numbers and negators).
CLUSTER_SIMILARITY = 0.85

#: The medoid wording must be the first draft of at least this share of the
#: cluster's deals.
DOMINANT_SHARE = 0.5

#: Clusters of different clause types are one form when the smaller one's deals
#: are at least this share inside the other's.
FORM_OVERLAP = 0.6

#: Word n-gram size of the similarity (a text shorter than this is one gram).
_SHINGLE = 3

#: Words that flip a clause's meaning: the same count is part of "near-identical".
_NEGATORS = frozenset({"not", "no", "never", "nor", "neither", "without", "none", "cannot"})

#: ``basis`` vocabulary of an origin verdict (``agent``: a coder acting as the
#: judge, ``judge``: the store-backed judge, ``owner``: the owner's correction).
ORIGIN_BASES = (BASIS_AGENT, BASIS_JUDGE, BASIS_OWNER)
_DRAFT_KEYS = frozenset({"ours", "reason", "basis"})

#: Outcome of a cluster's origin check.
STATE_CONFIRMED = "confirmed"
STATE_REJECTED = "rejected"
STATE_PENDING = "pending"

#: How settled a verdict must be to count (the owner's, an agreed blind check,
#: an adjudication); anything weaker is pending.
_COUNTING = frozenset({SETTLED_AGREED, SETTLED_ADJUDICATED, SETTLED_OWNER})

#: Why nothing was attempted (``Reconstruction.skipped``): an emergent playbook
#: has no ``our_standard`` for a first draft to differ from, and a supplied form
#: always wins.
SKIP_NO_TEMPLATE = "no_canonical_template"
SKIP_SUPPLIED = "forms_supplied"
SKIPS = (SKIP_NO_TEMPLATE, SKIP_SUPPLIED)


# ---------------------------------------------------------------------------
# First drafts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FirstDraft:
    """The first draft of one clause type in one signed deal on our paper."""

    document_id: str
    taxonomy_id: str
    text: str
    key: str  # the §3.5.4 grouping key of text
    ref: Mapping[str, Any]  # the OPF citation of the first-draft node
    signed_on: str | None


def first_drafts(
    records: Iterable[Mapping[str, Any]],
    tids: Iterable[str],
    *,
    party: str | None,
    any_paper: bool = False,
) -> list[FirstDraft]:
    """The first drafts of *records* (``evidence.precedent``) for *tids*.

    A record qualifies when its deal is signed, the clause type is one of
    *tids* (those with an ``our_standard``) and it opened ``non_standard``.
    Its first draft is the distinct opening text, else the signed text. Order
    follows the records.

    By default only deals on OUR paper qualify: those are the CANDIDATES a
    former form is built from. With *any_paper* every signed deal does: that
    is the set a confirmed form is matched against, because ``opened_with``
    is a fact about a deal's first-draft text whatever its paper side.
    """
    wanted = set(tids)
    out: list[FirstDraft] = []
    for record in records:
        if (
            record.get("signed") is not True
            or (not any_paper and record.get("paper") != PAPER_OURS)
            or record.get("opened_with") != OPENED_NON_STANDARD
            or record.get("taxonomy_id") not in wanted
        ):
            continue
        entry = record.get("opening_text") or record.get("signed_text")
        if not isinstance(entry, Mapping) or not isinstance(entry.get("text"), str):
            continue
        text = str(entry["text"])
        key = normalize_variant_text(text, party=party)
        if not key:
            continue
        out.append(
            FirstDraft(
                document_id=str(record["document_id"]),
                taxonomy_id=str(record["taxonomy_id"]),
                text=text,
                key=key,
                ref=dict(entry.get("ref") or {}),
                signed_on=record.get("signed_on"),
            )
        )
    return out


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------


def _grams(tokens: Sequence[str]) -> frozenset[tuple[str, ...]]:
    if len(tokens) < _SHINGLE:
        return frozenset({tuple(tokens)})
    return frozenset(tuple(tokens[i : i + _SHINGLE]) for i in range(len(tokens) - _SHINGLE + 1))


#: Spelled-out numbers: "five years" and "seven years" differ in a number.
_NUMBER_WORDS = frozenset(
    [
        "zero",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "eleven",
        "twelve",
        "thirteen",
        "fourteen",
        "fifteen",
        "sixteen",
        "seventeen",
        "eighteen",
        "nineteen",
        "twenty",
        "thirty",
        "forty",
        "fifty",
        "sixty",
        "seventy",
        "eighty",
        "ninety",
        "hundred",
        "thousand",
        "million",
        "billion",
        "first",
        "second",
        "third",
        "fourth",
        "fifth",
    ]
)


def _guard(tokens: Sequence[str]) -> tuple[tuple[str, ...], int]:
    """The numbers (sorted, digits or spelled out) and the count of negators of a wording."""
    numbers = tuple(sorted(t for t in tokens if t in _NUMBER_WORDS or any(c.isdigit() for c in t)))
    return numbers, sum(1 for t in tokens if t in _NEGATORS)


def similarity(key_a: str, key_b: str) -> float:
    """Near-identity of two grouping keys: the Jaccard similarity of their word
    3-gram sets, or ``0.0`` when they differ in a number or in how many
    negators they hold. ``1.0`` for equal keys."""
    if key_a == key_b:
        return 1.0
    a, b = key_a.split(), key_b.split()
    if _guard(a) != _guard(b):
        return 0.0
    grams_a, grams_b = _grams(a), _grams(b)
    union = len(grams_a | grams_b)
    return len(grams_a & grams_b) / union if union else 0.0


def _month(values: Iterable[Any]) -> tuple[str | None, str | None]:
    """The earliest and latest signing month (``YYYY-MM``) among *values*."""
    keyed = sorted(k for v in values for k in (signed_on_key(v),) if k is not None)
    if not keyed:
        return None, None
    return f"{keyed[0][0]:04d}-{keyed[0][1]:02d}", f"{keyed[-1][0]:04d}-{keyed[-1][1]:02d}"


@dataclass(frozen=True)
class Cluster:
    """Near-identical first drafts of one clause type that qualify as a reconstructed clause."""

    taxonomy_id: str
    medoid: FirstDraft  # the representative draft of the medoid wording
    members: tuple[FirstDraft, ...]
    n_wordings: int  # distinct grouping keys among the members

    @property
    def deals(self) -> tuple[str, ...]:
        return tuple(sorted({d.document_id for d in self.members}))

    @property
    def n_deals(self) -> int:
        return len(self.deals)

    @property
    def span(self) -> tuple[str | None, str | None]:
        return _month(d.signed_on for d in self.members)

    @property
    def dominant_share(self) -> float:
        n = sum(1 for d in self.members if d.key == self.medoid.key)
        return n / len(self.members)


@dataclass
class _Tally:
    """What clustering left out, for the report."""

    below_min_deals: int = 0
    not_dominant: int = 0


def _medoid_key(counts: Mapping[str, int]) -> str:
    """The wording closest to all the others, weighted by deals (ties: the more
    deals, then the text)."""

    def score(k: str) -> tuple[float, int, str]:
        total = sum(n * similarity(k, other) for other, n in counts.items())
        return (-total, -counts[k], k)

    return min(counts, key=score)


def cluster_first_drafts(
    drafts: Iterable[FirstDraft],
    standard_keys: Mapping[str, str],
    *,
    min_deals: int = MIN_DEALS,
    tally: _Tally | None = None,
) -> list[Cluster]:
    """The qualifying clusters of *drafts* (module docstring), by clause type then size.

    *standard_keys* maps a clause type to the grouping key of today's
    standard; a cluster whose medoid reads as it is dropped (it would not be a
    prior form). Deterministic: the result depends on the drafts as a set.
    """
    tally = tally if tally is not None else _Tally()
    by_tid: dict[str, list[FirstDraft]] = defaultdict(list)
    for draft in drafts:
        by_tid[draft.taxonomy_id].append(draft)
    out: list[Cluster] = []
    for tid in sorted(by_tid):
        by_key: dict[str, list[FirstDraft]] = defaultdict(list)
        for draft in by_tid[tid]:
            by_key[draft.key].append(draft)
        # Most deals first, then text: the visiting order that makes a cluster's
        # seed its most common wording.
        order = sorted(by_key, key=lambda k: (-len({d.document_id for d in by_key[k]}), k))
        seeds: list[str] = []
        grouped: list[list[str]] = []
        for key in order:
            for i, seed in enumerate(seeds):
                if similarity(seed, key) >= CLUSTER_SIMILARITY:
                    grouped[i].append(key)
                    break
            else:
                seeds.append(key)
                grouped.append([key])
        for keys in grouped:
            members = tuple(
                sorted(
                    (d for k in keys for d in by_key[k]),
                    key=lambda d: (d.document_id, d.key),
                )
            )
            deals = {d.document_id for d in members}
            if len(deals) < min_deals:
                tally.below_min_deals += 1
                continue
            counts = {k: len({d.document_id for d in by_key[k]}) for k in keys}
            medoid_key = _medoid_key(counts)
            if counts[medoid_key] / len(deals) < DOMINANT_SHARE:
                tally.not_dominant += 1
                continue
            if medoid_key == standard_keys.get(tid):
                continue
            representative = min(by_key[medoid_key], key=lambda d: d.document_id)
            out.append(
                Cluster(
                    taxonomy_id=tid, medoid=representative, members=members, n_wordings=len(keys)
                )
            )
    out.sort(key=lambda c: (c.taxonomy_id, -c.n_deals, c.medoid.key))
    return out


# ---------------------------------------------------------------------------
# Forms
# ---------------------------------------------------------------------------


def _spans_overlap(a: tuple[str | None, str | None], b: tuple[str | None, str | None]) -> bool:
    """Whether two date spans overlap; a span with no dates overlaps anything."""
    if a[0] is None or b[0] is None:
        return True
    assert a[1] is not None and b[1] is not None
    return a[0] <= b[1] and b[0] <= a[1]


def group_forms(clusters: Sequence[Cluster]) -> list[list[int]]:
    """Group *clusters* into forms: lists of indexes into *clusters*.

    Clusters of different clause types join when the smaller one's deals are at
    least :data:`FORM_OVERLAP` inside the other's and their date spans overlap,
    strongest overlap first, never putting two clusters of one clause type in a
    form. Forms are ordered by their first signing month (undated last), then
    their first clause type; indexes inside a form follow *clusters*.
    """
    parent = list(range(len(clusters)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    tids_of: dict[int, set[str]] = {i: {c.taxonomy_id} for i, c in enumerate(clusters)}
    deals_of = [set(c.deals) for c in clusters]
    pairs: list[tuple[float, int, int]] = []
    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            if clusters[i].taxonomy_id == clusters[j].taxonomy_id:
                continue
            shared = len(deals_of[i] & deals_of[j])
            overlap = shared / min(len(deals_of[i]), len(deals_of[j]))
            if overlap >= FORM_OVERLAP and _spans_overlap(clusters[i].span, clusters[j].span):
                pairs.append((overlap, i, j))
    for _overlap, i, j in sorted(pairs, key=lambda p: (-p[0], p[1], p[2])):
        ri, rj = find(i), find(j)
        if ri == rj or tids_of[ri] & tids_of[rj]:
            continue
        parent[rj] = ri
        tids_of[ri] |= tids_of[rj]
    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(len(clusters)):
        groups[find(i)].append(i)

    def order(members: list[int]) -> tuple[bool, str, str]:
        firsts = [f for i in members for f in (clusters[i].span[0],) if f is not None]
        return (not firsts, min(firsts) if firsts else "", clusters[members[0]].taxonomy_id)

    return sorted((sorted(m) for m in groups.values()), key=order)


def _form_label(
    members: Sequence[Cluster], span: tuple[str | None, str | None], taken: set[str]
) -> str:
    """A label from what is known of the form: its clause (when it is one
    clause alone) and the span of its deals' signing months."""
    first, last = span
    when = "undated" if first is None else first if first == last else f"{first} to {last}"
    base = (
        f"reconstructed {members[0].taxonomy_id} {when}"
        if len(members) == 1
        else f"reconstructed {when}"
    )
    label, n = base, 1
    while label in taken:
        n += 1
        label = f"{base} #{n}"
    taken.add(label)
    return label


# ---------------------------------------------------------------------------
# The origin judgment
# ---------------------------------------------------------------------------


def origin_key(
    agreement_type_id: str,
    taxonomy_id: str,
    party: str | None,
    text: str,
    standard_part_texts: Sequence[str],
) -> str:
    """The verdict-store key of a cluster's medoid *text* against today's standard.

    ``sha256`` over ``[kind, agreement_type.id, taxonomy_id, perspective.party,
    grouping_key(text), [grouping_key(part) ...]]``: no document, no counterparty,
    no paper side, no count, no date — one question per distinct wording."""
    raw = json.dumps(
        [
            ORIGIN_KIND,
            agreement_type_id,
            taxonomy_id,
            party or "",
            normalize_variant_text(text, party=party),
            [normalize_variant_text(p, party=party) for p in standard_part_texts],
        ],
        ensure_ascii=False,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class OriginSubject:
    """One cluster awaiting (or carrying) an origin verdict.

    ``payload`` is exactly what the drafter and the blind checker see: the
    clause type, today's standard as numbered parts, the recurring wording and
    how often and between when it was the first draft. No document,
    counterparty or paper side."""

    key: str
    payload: dict[str, Any]
    cluster: Cluster


def origin_subjects(
    clusters: Iterable[Cluster],
    clauses: Mapping[str, Mapping[str, Any]],
    *,
    agreement_type_id: str,
    party: str | None,
) -> list[OriginSubject]:
    """The origin question of each cluster, in cluster order. *clauses* maps a
    clause type to its ``evidence.clauses[]`` entry (title and standard parts).
    Clusters partition the wordings of a clause type, so no two share a key."""
    out: list[OriginSubject] = []
    for cluster in clusters:
        clause = clauses.get(cluster.taxonomy_id) or {}
        parts = standard_parts(clause)
        key = origin_key(
            agreement_type_id,
            cluster.taxonomy_id,
            party,
            cluster.medoid.text,
            [str(p.get("text", "")) for p in parts],
        )
        first, last = cluster.span
        out.append(
            OriginSubject(
                key=key,
                payload={
                    "stage": ORIGIN_KIND,
                    "agreement_type_id": agreement_type_id,
                    "taxonomy_id": cluster.taxonomy_id,
                    "taxonomy_title": clause.get("title") or "",
                    "perspective_party": party,
                    "our_standard_parts": [
                        {"n": i + 1, "part_id": p.get("part_id"), "text": p.get("text")}
                        for i, p in enumerate(parts)
                    ],
                    "candidate": cluster.medoid.text,
                    "recurs": {
                        "n_deals": cluster.n_deals,
                        "n_wordings": cluster.n_wordings,
                        "first_signed": first,
                        "last_signed": last,
                    },
                },
                cluster=cluster,
            )
        )
    return out


def validate_origin_draft(verdict: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a producer-supplied origin verdict (``judge-apply --verdicts``).

    ``{ours, reason, basis}`` and nothing else: ``ours`` a JSON boolean (is the
    recurring wording OUR former language), ``reason`` one sentence, ``basis``
    one of :data:`ORIGIN_BASES`. A check is recorded only by
    ``judge-apply --check``. Returns the verdict as it is stored.
    """
    extra = sorted(set(verdict) - _DRAFT_KEYS)
    if extra:
        raise ValueError(
            f"unexpected field(s) {extra} on a prior_form_origin verdict; a verdict carries "
            "only ours, reason and basis (checks are recorded with `judge-apply --check`)"
        )
    if not isinstance(verdict.get("ours"), bool):
        raise ValueError("'ours' must be a JSON boolean")
    reason = verdict.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("'reason' must be a non-empty string (one sentence)")
    basis = verdict.get("basis")
    if basis not in ORIGIN_BASES:
        raise ValueError(f"'basis' must be one of {list(ORIGIN_BASES)}, got {basis!r}")
    return {"ours": verdict["ours"], "reason": reason.strip(), "basis": basis}


def verdict_of(stored: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """The stored origin verdict *stored* as is, or ``None`` when it is absent or malformed
    (then treated as unjudged and never replayed)."""
    if not isinstance(stored, Mapping):
        return None
    if not isinstance(stored.get("ours"), bool) or not isinstance(stored.get("reason"), str):
        return None
    if stored.get("basis") not in ORIGIN_BASES:
        return None
    return dict(stored)


def outcome_of(verdict: Mapping[str, Any] | None) -> str:
    """``confirmed`` / ``rejected`` / ``pending`` for a stored origin verdict.

    Only an owner's, an agreed blind check's or an adjudicated verdict counts:
    no verdict, a draft nobody checked and a disputed one are ``pending``."""
    if verdict is None:
        return STATE_PENDING
    if check_state(verdict) not in _COUNTING:
        return STATE_PENDING
    return STATE_CONFIRMED if verdict.get("ours") is True else STATE_REJECTED


# ---------------------------------------------------------------------------
# The reconstruction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClusterOutcome:
    """A qualifying cluster, its question, its stored verdict and where that leaves it."""

    subject: OriginSubject
    verdict: dict[str, Any] | None
    state: str
    #: The label of the form the cluster is part of (confirmed clusters only).
    form: str | None = None

    @property
    def drafted(self) -> bool:
        return self.verdict is not None


@dataclass(frozen=True)
class Reconstruction:
    """What :func:`reconstruct` found.

    ``observations`` are the deal observations with ``opened_with`` re-read for
    the deals that opened on a confirmed reconstructed clause, less the rows of
    our former wording struck before signing (counted per deal in
    ``removed_by_document``); ``forms`` are the reconstructed forms (confirmed
    clusters only), in the order of ``baseline.prior_templates``. ``outcomes``
    is every qualifying cluster. ``skipped`` says why nothing was attempted (a
    supplied form wins, no canonical template) or is ``None``.
    """

    observations: list[Observation]
    forms: list[PriorForm]
    outcomes: list[ClusterOutcome] = field(default_factory=list)
    n_candidate_drafts: int = 0
    below_min_deals: int = 0
    not_dominant: int = 0
    skipped: str | None = None
    #: Distinct deals that opened on any reconstructed form.
    n_deals_opened: int = 0
    #: Per deal, the rows dropped as our former wording struck before signing
    #: (``removed_prior_standard``), as mining counts them for a supplied form.
    removed_by_document: dict[str, int] = field(default_factory=dict)

    def corpus_documents_with_drops(
        self, corpus_documents: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """*corpus_documents* with each deal's ``dropped_observations`` counting
        its rows of our reconstructed former wording struck before signing under
        ``removed_prior_standard`` (summed into ``corpus.stats`` by the
        assembler), exactly where mining counts them for a supplied form. The
        documents are copied, never changed in place."""
        if not self.removed_by_document:
            return corpus_documents
        out: list[dict[str, Any]] = []
        for doc in corpus_documents:
            n = self.removed_by_document.get(str(doc.get("document_id")), 0)
            if n:
                dropped = dict(doc.get("dropped_observations") or {})
                dropped[DROPPED_PRIOR_STANDARD_REMOVED] = (
                    dropped.get(DROPPED_PRIOR_STANDARD_REMOVED, 0) + n
                )
                doc = {**doc, "dropped_observations": dict(sorted(dropped.items()))}
            out.append(doc)
        return out

    def counts(self) -> dict[str, Any]:
        """Counts only (no text): the clusters by outcome, drafted ones, forms."""
        by_state = dict.fromkeys((STATE_CONFIRMED, STATE_REJECTED, STATE_PENDING), 0)
        for outcome in self.outcomes:
            by_state[outcome.state] += 1
        return {
            "skipped": self.skipped,
            "candidate_first_drafts": self.n_candidate_drafts,
            "clusters": len(self.outcomes),
            "clusters_below_min_deals": self.below_min_deals,
            "clusters_not_dominant": self.not_dominant,
            **by_state,
            "pending_undrafted": sum(
                1 for o in self.outcomes if o.state == STATE_PENDING and not o.drafted
            ),
            "pending_unchecked": sum(
                1 for o in self.outcomes if o.state == STATE_PENDING and o.drafted
            ),
            "forms": len(self.forms),
            "forms_single_clause": sum(1 for f in self.forms if len(f.observations) == 1),
            "forms_multi_clause": sum(1 for f in self.forms if len(f.observations) > 1),
            "forms_dated": sum(1 for f in self.forms if f.first_signed is not None),
            "forms_undated": sum(1 for f in self.forms if f.first_signed is None),
            "deals_in_forms": self.n_deals_opened,
            # Year-level spans, for the local sidecar only (the scorecard copies
            # integers): how many forms span each range of signing years.
            "forms_by_span": _span_years(self.forms),
        }


def _span_years(forms: Iterable[PriorForm]) -> dict[str, int]:
    """How many forms span each range of signing years (``"2021-2022"``,
    ``"2021"``, or ``"undated"``): counts only, no date finer than a year."""
    counts: dict[str, int] = defaultdict(int)
    for form in forms:
        if form.first_signed is None or form.last_signed is None:
            counts["undated"] += 1
            continue
        first, last = form.first_signed[:4], form.last_signed[:4]
        counts[first if first == last else f"{first}-{last}"] += 1
    return dict(sorted(counts.items()))


def _reconstructed_observation(cluster: Cluster, form_label: str) -> Observation:
    """The template-style observation that carries a reconstructed clause: the
    medoid wording cited where it sits in its corpus document."""
    ref = cluster.medoid.ref
    text = cluster.medoid.text
    span = ref.get("char_span")
    return Observation(
        observation_id=f"reconstructed/{form_label}/{cluster.taxonomy_id}",
        taxonomy_id=cluster.taxonomy_id,
        text_summary=summarize_clause_text(text),
        full_text=text,
        citation=ObservationCitation(
            document_id=str(ref.get("document_id") or cluster.medoid.document_id),
            version=ref.get("version", 1),
            clause_path=str(ref.get("clause_path") or "?"),
            char_span=(int(span[0]), int(span[1])) if span else None,
        ),
        deviation="substantive",
        risk_delta={"direction": "neutral", "magnitude": "none"},
        provenance="our_paper",
        outcome="signed",
    )


def reconstruct(
    *,
    observations: Sequence[Observation],
    corpus_documents: list[dict[str, Any]],
    clause_positions: list[Any],
    round_moves: list[Any] | None,
    agreement_type_id: str,
    party: str | None,
    supplied_forms: Sequence[PriorForm],
    judge: Callable[[OriginSubject], dict[str, Any] | None] | None,
    has_canonical_template: bool = True,
    party_names: Sequence[str] = (),
) -> Reconstruction:
    """Reconstruct our former forms from *observations* (module docstring).

    Nothing is attempted (``skipped`` says why) when the playbook is emergent
    (no canonical template, so no ``our_standard`` to differ from) or when any
    form was supplied — a supplied form always wins. *judge* returns the stored
    origin verdict of a subject, or ``None`` (and may queue the question);
    ``None`` itself means no verdict store exists yet, so every cluster is
    pending. *party_names* are the names mining's standard check neutralizes
    (our aliases and the known entities): the removed-row test of a confirmed
    form uses them as mining does for a supplied one.
    """
    unchanged = list(observations)
    if not has_canonical_template:
        return Reconstruction(unchanged, [], skipped=SKIP_NO_TEMPLATE)
    if supplied_forms:
        return Reconstruction(unchanged, [], skipped=SKIP_SUPPLIED)
    evidence = build_precedent_evidence(
        agreement_type_id=agreement_type_id,
        clause_positions=clause_positions,
        observations=list(observations),
        corpus_documents=corpus_documents,
        round_moves=round_moves,
        party=party,
        prior_forms=(),
    )
    clauses = {str(c["taxonomy_id"]): c for c in evidence["clauses"]}
    standard_keys = {
        tid: normalize_variant_text(str(c["our_standard"]["text"]), party=party)
        for tid, c in clauses.items()
        if isinstance(c.get("our_standard"), dict)
    }
    drafts = first_drafts(evidence["precedent"], standard_keys, party=party)
    tally = _Tally()
    clusters = cluster_first_drafts(drafts, standard_keys, tally=tally)
    outcomes: list[ClusterOutcome] = []
    for subject in origin_subjects(
        clusters, clauses, agreement_type_id=agreement_type_id, party=party
    ):
        verdict = verdict_of(judge(subject)) if judge is not None else None
        outcomes.append(ClusterOutcome(subject, verdict, outcome_of(verdict)))
    confirmed = [i for i, o in enumerate(outcomes) if o.state == STATE_CONFIRMED]
    forms: list[PriorForm] = []
    relabel: dict[tuple[str, str], str] = {}
    if confirmed:
        kept = [clusters[i] for i in confirmed]
        groups = group_forms(kept)
        # Which signed deals opened on a confirmed form is the same exact fact
        # a supplied form gives (issue #249): the deal's first draft of the
        # clause has the form clause's grouping key. It is read off EVERY
        # signed non_standard opening of that clause type, whatever its paper
        # side; paper side only chose the candidates the form was built from.
        # A wording is the medoid of one cluster of its clause type, so a
        # clause reads as at most one form and no per-clause vote is needed.
        group_of = {i: gi for gi, members in enumerate(groups) for i in members}
        gi_by_wording = {(c.taxonomy_id, c.medoid.key): group_of[i] for i, c in enumerate(kept)}
        openings = first_drafts(
            evidence["precedent"], {c.taxonomy_id for c in kept}, party=party, any_paper=True
        )
        opened_on: dict[tuple[str, str], int] = {}
        dated: dict[int, list[str | None]] = defaultdict(list)
        deals_of: dict[int, set[str]] = defaultdict(set)
        for draft in openings:
            gi = gi_by_wording.get((draft.taxonomy_id, draft.key))
            if gi is None:
                continue
            opened_on[(draft.document_id, draft.taxonomy_id)] = gi
            deals_of[gi].add(draft.document_id)
            dated[gi].append(draft.signed_on)
        taken: set[str] = set()
        label_of: dict[int, str] = {}
        for gi, members in enumerate(groups):
            member_clusters = [kept[i] for i in members]
            span = _month(dated[gi])
            label = _form_label(member_clusters, span, taken)
            label_of[gi] = label
            forms.append(
                PriorForm(
                    label=label,
                    observations=tuple(
                        _reconstructed_observation(c, label)
                        for c in sorted(member_clusters, key=lambda c: c.taxonomy_id)
                    ),
                    basis=BASIS_RECONSTRUCTED,
                    n_deals=len(deals_of[gi]),
                    first_signed=span[0],
                    last_signed=span[1],
                )
            )
            for i in members:
                outcomes[confirmed[i]] = replace(outcomes[confirmed[i]], form=label)
        relabel = {key: label_of[gi] for key, gi in opened_on.items()}
    kept_observations, removed = _without_struck_former_wording(
        [_relabelled(o, relabel) for o in unchanged],
        forms,
        party=party,
        party_names=party_names,
    )
    return Reconstruction(
        observations=kept_observations,
        forms=forms,
        outcomes=outcomes,
        n_candidate_drafts=len(drafts),
        below_min_deals=tally.below_min_deals,
        not_dominant=tally.not_dominant,
        n_deals_opened=len({doc for doc, _tid in relabel}),
        removed_by_document=removed,
    )


#: The outcome of a row proposed and struck before signing (a removed row or a
#: reversal): precedent reads it as a refused ask.
_REFUSED_OUTCOME = "proposed_then_reversed"


def _key_as_shipped(text: str, party: str | None) -> str:
    """The §3.5.4 grouping key of *text* as the playbook carries it (invisible
    characters stripped, as the assembler strips them)."""
    from playbook_engine.playbook_assembler import _strip_invisible  # noqa: PLC0415

    return normalize_variant_text(str(_strip_invisible(text)), party=party)


def _is_piece_by_key(text: str, nodes: Sequence[str], party: str | None) -> bool:
    """:func:`~playbook_engine.observation_builder.is_former_form_language`
    under the §3.5.4 grouping key: *text* is a node of *nodes*, their whole or
    one contiguous piece of it."""
    key = _key_as_shipped(text, party)
    if not key:
        return False
    whole = _key_as_shipped("\n".join(nodes), party)
    return f" {key} " in f" {whole} " or any(key == _key_as_shipped(n, party) for n in nodes)


def _without_struck_former_wording(
    observations: list[Observation],
    forms: Sequence[PriorForm],
    *,
    party: str | None,
    party_names: Sequence[str],
) -> tuple[list[Observation], dict[str, int]]:
    """*observations* less the rows of our former wording struck before
    signing, and how many were dropped per deal (module docstring).

    The supplied-form rule of mining (issue #249), applied to the confirmed
    reconstructed *forms*: a ``proposed_then_reversed`` row that is not today's
    standard and whose text is a form clause's wording, whole or one contiguous
    piece of it, is no refused ask of the counterparty. (A removed row of
    today's standard language is never ``proposed_then_reversed``: mining made
    it ``conceded_before_signing``, so the current standard still wins.) A
    reversal row of that wording is dropped too: the playbook records no refused
    ask that reads as a reconstructed clause (OPF-SPEC §3.5.7, which the
    validator enforces)."""
    if not forms:
        return observations, {}
    nodes_by_tid = origin_standards(forms)
    kept: list[Observation] = []
    removed: dict[str, int] = defaultdict(int)
    for obs in observations:
        candidates = (
            nodes_by_tid.get(obs.taxonomy_id, ())
            if obs.taxonomy_id is not None and obs.outcome == _REFUSED_OUTCOME and not obs.standard
            else ()
        )
        if any(
            is_former_form_language(obs.full_text, nodes, party_names)
            or _is_piece_by_key(obs.full_text, nodes, party)
            for _label, nodes, _index in candidates
        ):
            removed[obs.citation.document_id] += 1
            continue
        kept.append(obs)
    return kept, dict(sorted(removed.items()))


def _relabelled(obs: Observation, relabel: Mapping[tuple[str, str], str]) -> Observation:
    """*obs* with ``opened_with`` read as the reconstructed form its deal opened
    on, when it is a ``non_standard`` opening that is a confirmed form's wording."""
    if obs.opened_with != OPENED_NON_STANDARD or obs.taxonomy_id is None:
        return obs
    label = relabel.get((obs.citation.document_id, obs.taxonomy_id))
    if label is None:
        return obs
    return replace(obs, opened_with=OPENED_PRIOR_STANDARD, opened_with_form=label)
