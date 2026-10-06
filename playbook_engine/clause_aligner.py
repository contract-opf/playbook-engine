"""Cross-version clause alignment — L3 pipeline stage.

Matches 'the same clause' across versions of one document so that downstream
diff stages operate clause-scoped rather than whole-document.

Algorithm (fully deterministic, no LLM):

0. **Global move matching** (issue: relocation artifacts): pair clauses
   across adjacent versions by content similarity anywhere in the document —
   normalized-text near-exact first, then high-Jaccard mutual-best — and
   chain the pairs across versions. Each chain becomes one aligned row
   (``match_basis`` = ``"content_exact"`` / ``"content_jaccard"``), so a
   clause that merely moves position (or whose classification flaps between
   versions) aligns to itself instead of degenerating into a delete+add
   pair. Only substantial clauses participate (length/token minimums below);
   short boilerplate is left to the positional path.
0b. **Backward extension** (issue #232): a move row that starts at a later
   draft is offered, newest version pair first, the earlier draft's
   unmatched same-taxonomy clauses under the bucket path's bind rule (step
   3), so a clause edited in round one and carried unchanged into the
   signed copy stays one row instead of a removed + added pair (see
   :func:`_extend_moves_backward`).
1. Group each remaining (unmatched) clause list by ``taxonomy_id``.
2. Collect all taxonomy_ids in first-appearance order (v0 → v1 → ...).
3. For each taxonomy_id bucket, align the per-version clause sequences by
   text similarity (issue #222) — never by position alone, and whether or
   not the per-version counts agree. Two clauses bind as one logical clause
   only when their token-set Jaccard is at least
   ``ALIGNMENT_AMBIGUITY_THRESHOLD``, or — a narrow rescue for a localized
   edit such as a changed number or an appended proviso — when the Jaccard
   is at least ``ALIGNMENT_RESCUE_MIN_JACCARD``, the shorter clause has at
   least ``ALIGNMENT_RESCUE_MIN_TOKENS`` content tokens, and the two token
   sequences differ by exactly one contiguous replace, insert or delete
   (see :func:`_bind_by_similarity`). The version with the most clauses (the
   first one on a tie) is the reference frame; every other version, nearest
   the reference first, binds each of its clauses to an open row, highest
   Jaccard first (ties broken by position). A clause that binds nowhere
   opens its own row, so two unrelated clauses that merely share a
   taxonomy_id become separate removed/added rows instead of one fabricated
   ``modified`` diff. A row's slots carry its worst binding Jaccard as
   ``alignment_confidence``.
4. Return one ``ClauseAlignment`` per logical clause, preserving the
   first-appearance order of taxonomy_ids.

Handles:
  - Renumbering: §1→§2 is irrelevant; taxonomy_id is the key.
  - Relocation: a moved-but-unchanged clause pairs with itself via the global
    move phase (near-exact content match) and diffs as ``unchanged``; a
    moved-and-edited clause pairs via high-Jaccard and diffs as one
    ``modified`` row with the true before/after.
  - Insertions: new taxonomy_id in a later version → AlignmentSlot(clause=None)
    for earlier versions.
  - Deletions: taxonomy_id absent in a later version → AlignmentSlot(clause=None)
    for that version.
  - Splits/merges: handled by similarity matching within the bucket; a part
    similar enough to the original stays on its row, the rest become new
    rows. Two same-taxonomy_id clauses that swap positions are matched by
    similarity (and, when substantial enough, by the global move phase
    first), never zipped by position.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Protocol, runtime_checkable

from playbook_engine.clause_classifier import ClassifiedClause

# ---------------------------------------------------------------------------
# Stop words (same set as clause_classifier for consistency)
# ---------------------------------------------------------------------------

_STOP_WORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "in",
        "is",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "with",
    }
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Minimum token-set Jaccard for the bucket path to bind two clauses as the
#: same logical clause (issue #222). Below it the clauses are separate rows
#: (removed + added) unless the narrow localized-edit rescue below applies,
#: and a row left with an empty slot triggers an ``AlignmentJudge`` call (if
#: a judge is configured). One shared token is not evidence that two clauses
#: are the same clause.
ALIGNMENT_AMBIGUITY_THRESHOLD: float = 0.70

#: Localized-edit rescue (issue #222): a pair below
#: ``ALIGNMENT_AMBIGUITY_THRESHOLD`` still binds when ALL of these hold —
#: its Jaccard is at least ``ALIGNMENT_RESCUE_MIN_JACCARD``; the shorter
#: clause has at least ``ALIGNMENT_RESCUE_MIN_TOKENS`` content
#: (non-stop-word) tokens; and
#: ``difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes()`` over
#: the two content-token sequences has EXACTLY ONE non-``equal`` opcode — one
#: contiguous replace, insert or delete. That is the shape of a localized
#: edit (a changed number, an appended carve-out) that costs a short clause
#: its Jaccard; short boilerplate that merely shares a skeleton with an
#: unrelated longer clause differs in several separate spans (or falls below
#: the Jaccard floor) and is never rescued. A rescue bind may not take a
#: partner from a competing pair whose Jaccard is within
#: ``ALIGNMENT_RESCUE_MARGIN`` of its own or higher, and the confidence it
#: reports is the Jaccard.
ALIGNMENT_RESCUE_MIN_JACCARD: float = 0.5
ALIGNMENT_RESCUE_MIN_TOKENS: int = 3
ALIGNMENT_RESCUE_MARGIN: float = 0.1

#: Global move matching — near-exact phase: normalized clause text must be at
#: least this many characters to participate. Short boilerplate ("Notices",
#: "Reserved") would otherwise cross-pair unrelated slots document-wide.
MOVE_EXACT_MIN_CHARS: int = 30

#: Global move matching — Jaccard phase: mutual-best pairs at or above this
#: similarity are treated as the same clause relocated-and-edited. Stricter
#: than ALIGNMENT_AMBIGUITY_THRESHOLD because a document-wide match has no
#: positional prior backing it up.
MOVE_JACCARD_THRESHOLD: float = 0.80

#: Global move matching — Jaccard phase: both clauses must have at least this
#: many non-stop-word tokens. Tiny token sets reach high Jaccard by accident.
MOVE_JACCARD_MIN_TOKENS: int = 8

# ---------------------------------------------------------------------------
# Judge protocol (LLM integration point)
# ---------------------------------------------------------------------------


@runtime_checkable
class AlignmentJudge(Protocol):
    """Protocol for LLM-assisted clause alignment disambiguation.

    Called only on ambiguous rows of the bucket path — a row with a
    ``None`` slot (a version whose clauses did not bind to the row, issue
    #222), a row opened by a non-reference version's unbound clause, or a
    row bound by the localized-edit rescue (Jaccard below
    ``ALIGNMENT_AMBIGUITY_THRESHOLD``).

    Contract:
    - Return one ``(before_idx | None, after_idx | None)`` pair per *logical*
      clause the judge resolves. ``None`` indices mean no match on that side.
      The caller emits exactly one output row per returned pair — a judge
      that finds a split or merge should return multiple pairs (e.g. two
      pairs both referencing the same ``before_idx`` for a one-into-two
      split), and every pair is preserved in the output.
    - When the bucket spans more than two versions, ``after_clauses`` is the
      concatenation of every non-reference version's clauses for this row
      (in ascending version-index order); the caller tracks which original
      version each ``after_idx`` came from when reconstructing rows, so
      judges do not need to know version identity — only position.
    - Implementations MUST NOT raise; on any error they should return the
      identity pairing ``[(i, i) for i in range(len(before_clauses))]`` to
      preserve the deterministic fallback.
    """

    def judge_bucket(
        self,
        before_clauses: list[ClassifiedClause],
        after_clauses: list[ClassifiedClause],
    ) -> list[tuple[int | None, int | None]]:
        """Resolve one ambiguous alignment bucket.

        Args:
            before_clauses: Clauses from the reference (longest) version.
            after_clauses:  Clauses from the non-reference version(s) being
                            matched (concatenated in version-index order when
                            more than one non-reference version is present).

        Returns:
            A pairing/split/merge map — one tuple per logical clause row.
            Each tuple is ``(before_idx | None, after_idx | None)``. Return
            multiple tuples to represent a split or merge; the caller emits
            one output row per tuple.
        """
        ...  # pragma: no cover


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AlignmentSlot:
    """One version's contribution to a logical clause alignment."""

    version: str
    clause: ClassifiedClause | None
    alignment_confidence: float | None = field(default=None)


@dataclass(frozen=True)
class ClauseAlignment:
    """One logical clause aligned across all versions.

    ``slots`` is parallel to the ``classified_versions`` input list: ``slots[i]``
    corresponds to ``classified_versions[i]``.  A slot's ``clause`` is ``None``
    when the logical clause is absent in that version (insertion or deletion).

    ``match_basis`` records how the row was paired: ``"content_exact"`` /
    ``"content_jaccard"`` for rows produced by the global move phase (the
    clause was matched by content anywhere in the document — a relocation or
    classification flap), ``None`` for rows from the positional bucket path.
    A move row is ``"content_exact"`` only when every link is near-exact; a
    row the backward extension (issue #232) grew by bind similarity is
    ``"content_jaccard"``, and its ``alignment_confidence`` is its worst
    link's Jaccard.
    """

    taxonomy_id: str | None
    slots: tuple[AlignmentSlot, ...]
    match_basis: str | None = field(default=None)

    @property
    def is_present_in_all(self) -> bool:
        """True when every slot has a non-None clause."""
        return all(s.clause is not None for s in self.slots)

    @property
    def version_count(self) -> int:
        """Number of versions in this alignment."""
        return len(self.slots)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def align_versions(
    classified_versions: list[tuple[str, list[ClassifiedClause]]],
    *,
    alignment_judge: AlignmentJudge | None = None,
) -> list[ClauseAlignment]:
    """Align classified clauses across versions of the same document.

    Args:
        classified_versions: ``[(version_id, classified_clauses), ...]`` in
                             version order (oldest first).  Version ids must
                             be unique.
        alignment_judge:     Optional judge called on ambiguous rows (a
                             ``None`` slot, a row opened by a non-reference
                             version's unbound clause, or a rescue bind
                             below ``ALIGNMENT_AMBIGUITY_THRESHOLD``).  When
                             ``None`` the deterministic similarity matching
                             is used for all buckets.

    Returns:
        One ``ClauseAlignment`` per logical clause, in first-appearance order
        of taxonomy_ids.

    Raises:
        ValueError: if ``classified_versions`` contains duplicate version ids.
    """
    if not classified_versions:
        return []

    version_ids = [vid for vid, _ in classified_versions]
    if len(version_ids) != len(set(version_ids)):
        raise ValueError(f"Duplicate version ids in classified_versions: {version_ids!r}")

    if len(classified_versions) == 1:
        vid, clauses = classified_versions[0]
        return [
            ClauseAlignment(
                taxonomy_id=c.classification.taxonomy_id,
                slots=(AlignmentSlot(version=vid, clause=c),),
            )
            for c in clauses
        ]

    # 0. Global move matching: pair substantial clauses by content anywhere
    #    in the document, chain across versions, and take those rows out of
    #    the positional path entirely.
    full_lists = [clauses for _, clauses in classified_versions]
    chains, matched_keys = _match_moves(full_lists)
    # 0b. Backward extension (issue #232): an earlier draft's copy of a
    #     clause the move phase chained only from a later draft onwards
    #     joins that row by bind similarity instead of being stranded.
    _extend_moves_backward(full_lists, chains, matched_keys)
    move_rows = [_move_row(version_ids, full_lists, chain) for chain in chains]

    # 1. Build per-version groups over the REMAINDER only:
    #    {taxonomy_id: [ClassifiedClause, ...]}
    VersionGroups = dict[str | None, list[ClassifiedClause]]
    per_version: list[VersionGroups] = []
    for vi, clauses in enumerate(full_lists):
        groups: VersionGroups = {}
        for ci, c in enumerate(clauses):
            if (vi, ci) in matched_keys:
                continue
            tid = c.classification.taxonomy_id
            groups.setdefault(tid, []).append(c)
        per_version.append(groups)

    # 2. Collect all taxonomy_ids in first-appearance order — over the FULL
    #    clause lists, so move rows slot into the same ordering frame.
    seen_tids: set[str | None] = set()
    ordered_tids: list[str | None] = []
    for clauses in full_lists:
        for c in clauses:
            tid = c.classification.taxonomy_id
            if tid not in seen_tids:
                seen_tids.add(tid)
                ordered_tids.append(tid)

    # 3. For each taxonomy_id, emit that bucket's move rows (document order)
    #    followed by the positional alignment of the remainder.
    move_rows_by_tid: dict[str | None, list[ClauseAlignment]] = {}
    for row in move_rows:
        move_rows_by_tid.setdefault(row.taxonomy_id, []).append(row)

    result: list[ClauseAlignment] = []
    for tid in ordered_tids:
        result.extend(move_rows_by_tid.get(tid, []))
        seqs = [groups.get(tid, []) for groups in per_version]
        if any(seqs):
            result.extend(_align_seqs(tid, version_ids, seqs, alignment_judge=alignment_judge))

    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


@dataclass
class _MoveChain:
    """One move row under construction: ``members`` maps each version index
    to the index of its clause in that version (always a contiguous run of
    versions); ``sims``/``bases`` record every link's similarity and basis."""

    members: dict[int, int]
    sims: list[float]
    bases: list[str]


def _match_moves(
    full_lists: list[list[ClassifiedClause]],
) -> tuple[list[_MoveChain], set[tuple[int, int]]]:
    """Global move-matching phase: pair clauses across adjacent versions by
    content similarity anywhere in the document, then chain the pairs across
    versions into move rows.

    Returns ``(chains, matched_keys)`` where ``matched_keys`` is the set of
    ``(version_index, clause_index)`` coordinates consumed by a move row —
    the positional path must skip exactly those clauses.
    """
    n_versions = len(full_lists)

    # links[i]: {a_idx: (b_idx, basis, sim)} pairing version i → version i+1.
    links: list[dict[int, tuple[int, str, float]]] = []
    for i in range(n_versions - 1):
        links.append(_match_pair(full_lists[i], full_lists[i + 1]))

    # Chain links: a row starts at any clause that is not the target of a
    # link from the previous version, and follows links forward.
    incoming: list[set[int]] = [set() for _ in range(n_versions)]
    for i, pair_map in enumerate(links):
        incoming[i + 1].update(b for b, _, _ in pair_map.values())

    chains: list[_MoveChain] = []
    matched_keys: set[tuple[int, int]] = set()
    for vi in range(n_versions - 1):
        for ci in range(len(full_lists[vi])):
            if vi > 0 and ci in incoming[vi]:
                continue  # continuation of an earlier chain
            if ci not in links[vi]:
                continue  # no content match — positional path handles it
            # Follow the chain forward from (vi, ci).
            chain = _MoveChain(members={vi: ci}, sims=[], bases=[])
            v, c = vi, ci
            while v < n_versions - 1 and c in links[v]:
                nxt, basis, sim = links[v][c]
                chain.bases.append(basis)
                chain.sims.append(sim)
                chain.members[v + 1] = nxt
                v, c = v + 1, nxt
            matched_keys.update(chain.members.items())
            chains.append(chain)

    return chains, matched_keys


def _extend_moves_backward(
    full_lists: list[list[ClassifiedClause]],
    chains: list[_MoveChain],
    matched_keys: set[tuple[int, int]],
) -> None:
    """Extend move rows backwards by bind similarity (issue #232).

    The move phase links two drafts only at a near-exact or a
    ``MOVE_JACCARD_THRESHOLD`` match, and it takes every clause it chains out
    of the bucket path. When a clause is edited in round one and then carried
    unchanged into the signed copy (v1 "five (5) years" → v2 "three (3)
    years" == v3), the move phase chains v2 to v3 only, and v1's copy is
    left with nothing to bind to: the deal shows a same-round removed + added
    pair and a fabricated refused ask downstream.

    For each adjacent version pair (v_i, v_i+1), NEWEST first, every clause
    of v_i that no row holds is offered to the move rows that have a v_i+1
    member but no v_i member — under the bucket path's own bind rule
    (:func:`_bind_by_similarity`: Jaccard >= ``ALIGNMENT_AMBIGUITY_THRESHOLD``
    or the localized-edit rescue, ranked Jaccard first). Offers stay within
    one taxonomy bucket: a v_i clause is compared only with v_i+1 clauses of
    its own ``taxonomy_id``. The free (unmatched) v_i+1 clauses of the bucket
    compete in the same ranking, so a row never takes a clause whose better
    partner is a free clause (that clause is left to the bucket path), and
    they contest a rescue bind just as they do in the bucket path; only
    binds that land on a move row are kept. Newest first, so a row extended to v_i can be extended
    again to v_i-1 — a clause edited in two successive rounds and then
    carried into the signed copy stays one row.

    A clause already held by a row is never re-bound. A bound clause joins
    the row (``matched_keys`` gains it, so the bucket path skips it) and its
    Jaccard is recorded as a link of the row, so the row's
    ``alignment_confidence`` is its worst link. The row's taxonomy_id still
    follows its latest member, which an extension never changes.
    """
    for vi in range(len(full_lists) - 2, -1, -1):
        nxt = vi + 1
        # Move rows starting at v_i+1, keyed by their v_i+1 member's index.
        chain_by_ci = {
            chain.members[nxt]: chain
            for chain in chains
            if nxt in chain.members and vi not in chain.members
        }
        if not chain_by_ci:
            continue
        free_older = [ci for ci in range(len(full_lists[vi])) if (vi, ci) not in matched_keys]
        if not free_older:
            continue
        free_newer = [ci for ci in range(len(full_lists[nxt])) if (nxt, ci) not in matched_keys]
        # Bucket by taxonomy_id — clause and row member must share one.
        targets_by_tid: dict[str | None, list[int]] = {}
        for ci in sorted([*chain_by_ci, *free_newer]):
            tid = full_lists[nxt][ci].classification.taxonomy_id
            targets_by_tid.setdefault(tid, []).append(ci)
        offers_by_tid: dict[str | None, list[int]] = {}
        for ci in free_older:
            tid = full_lists[vi][ci].classification.taxonomy_id
            offers_by_tid.setdefault(tid, []).append(ci)
        for tid, offers in offers_by_tid.items():
            targets = targets_by_tid.get(tid, [])
            if not any(t in chain_by_ci for t in targets):
                continue
            offer_tokens = [_clause_tokens(full_lists[vi][ci].node.text or "") for ci in offers]
            target_tokens = [_clause_tokens(full_lists[nxt][t].node.text or "") for t in targets]
            for oi, ti, sim in _bind_by_similarity(
                offer_tokens, target_tokens, list(range(len(targets)))
            ):
                chain = chain_by_ci.get(targets[ti])
                if chain is None:
                    continue  # a free clause — the bucket path pairs it
                chain.members[vi] = offers[oi]
                chain.sims.append(sim)
                chain.bases.append("bind_similarity")
                matched_keys.add((vi, offers[oi]))


def _move_row(
    version_ids: list[str],
    full_lists: list[list[ClassifiedClause]],
    chain: _MoveChain,
) -> ClauseAlignment:
    """Build one move row's ``ClauseAlignment`` from its chain."""
    clause_by_version: dict[int, ClassifiedClause] = {
        v_idx: full_lists[v_idx][c_idx] for v_idx, c_idx in chain.members.items()
    }
    confidence = min(chain.sims)
    slots = tuple(
        AlignmentSlot(
            version=version_ids[i],
            clause=clause_by_version.get(i),
            alignment_confidence=confidence if i in clause_by_version else None,
        )
        for i in range(len(version_ids))
    )
    # The row's taxonomy_id follows the latest version's classification
    # (deviation-vs-template is assessed on the net/signed side).
    last_clause = clause_by_version[max(clause_by_version)]
    return ClauseAlignment(
        taxonomy_id=last_clause.classification.taxonomy_id,
        slots=slots,
        match_basis=(
            "content_exact" if all(b == "content_exact" for b in chain.bases) else "content_jaccard"
        ),
    )


def _match_pair(
    a_clauses: list[ClassifiedClause],
    b_clauses: list[ClassifiedClause],
) -> dict[int, tuple[int, str, float]]:
    """Content-match one adjacent version pair, document-wide.

    Near-exact normalized text first (substantial clauses only, duplicates
    paired greedily in document order), then high-Jaccard mutual-best on the
    remainder. Returns ``{a_idx: (b_idx, basis, similarity)}``.
    """
    result: dict[int, tuple[int, str, float]] = {}
    a_free = set(range(len(a_clauses)))
    b_free = set(range(len(b_clauses)))

    # Phase 1 — near-exact: identical normalized text, length-gated.
    def _exact_groups(clauses: list[ClassifiedClause], free: set[int]) -> dict[str, list[int]]:
        groups: dict[str, list[int]] = {}
        for idx in sorted(free):
            norm = _normalize(clauses[idx].node.text or "")
            if len(norm) >= MOVE_EXACT_MIN_CHARS:
                groups.setdefault(norm, []).append(idx)
        return groups

    a_groups = _exact_groups(a_clauses, a_free)
    b_groups = _exact_groups(b_clauses, b_free)
    for norm, a_idxs in a_groups.items():
        # strict=False: duplicate groups may have unequal counts across the
        # two versions; leftover duplicates fall to the positional path.
        for a_idx, b_idx in zip(a_idxs, b_groups.get(norm, []), strict=False):
            result[a_idx] = (b_idx, "content_exact", 1.0)
            a_free.discard(a_idx)
            b_free.discard(b_idx)

    # Phase 2 — high-Jaccard mutual-best on the remainder. Greedy global-max
    # acceptance: each accepted pair is mutual-best among clauses still free.
    a_tokens = {
        i: toks
        for i in a_free
        if len(toks := _tokens(a_clauses[i].node.text or "")) >= MOVE_JACCARD_MIN_TOKENS
    }
    b_tokens = {
        j: toks
        for j in b_free
        if len(toks := _tokens(b_clauses[j].node.text or "")) >= MOVE_JACCARD_MIN_TOKENS
    }
    candidates: list[tuple[float, int, int]] = []
    for i, ta in a_tokens.items():
        for j, tb in b_tokens.items():
            sim = len(ta & tb) / len(ta | tb)
            if sim >= MOVE_JACCARD_THRESHOLD:
                candidates.append((sim, i, j))
    # Sort by similarity desc, then document order for determinism on ties.
    candidates.sort(key=lambda t: (-t[0], t[1], t[2]))
    for sim, i, j in candidates:
        if i in a_free and j in b_free:
            result[i] = (j, "content_jaccard", sim)
            a_free.discard(i)
            b_free.discard(j)

    return result


def _align_seqs(
    taxonomy_id: str | None,
    version_ids: list[str],
    seqs: list[list[ClassifiedClause]],
    *,
    alignment_judge: AlignmentJudge | None = None,
) -> list[ClauseAlignment]:
    """Align per-version clause sequences within one taxonomy_id bucket.

    Similarity matching, never position alone (issue #222): the longest
    sequence (the first one on a tie) is the reference frame and opens one
    row per clause. Every other version, nearest the reference first, binds
    its clauses to open rows (see :func:`_bind_by_similarity`) — a clause is
    compared against the row's member nearest it in version order (its
    *frontier* on that side of the reference), so a clause that drifts
    gradually across many drafts stays on one row. Only a Jaccard of at
    least ``ALIGNMENT_AMBIGUITY_THRESHOLD``, or the narrow localized-edit
    rescue, binds; a clause that binds nowhere opens its own row. Each row's
    slots carry the row's worst binding Jaccard as ``alignment_confidence``
    (``None`` for a single-member row).
    """
    ref_idx = max(range(len(seqs)), key=lambda i: len(seqs[i]))
    ref_clauses = seqs[ref_idx]

    # Each clause is tokenized exactly once (issue #246).
    ref_tokens = [_clause_tokens(rc.node.text or "") for rc in ref_clauses]

    # rows[r]: {version_index: clause}; row_sim[r]: the worst binding
    # similarity on the row; left/right[r]: the tokens of the row's member
    # nearest version 0 / the last version (the frontier a further version on
    # that side binds against); row_pos[r]: the positional prior used to
    # break similarity ties (the opening clause's index in its version).
    rows: list[dict[int, ClassifiedClause]] = [{ref_idx: c} for c in ref_clauses]
    row_sim: list[float | None] = [None] * len(ref_clauses)
    left: list[_ClauseTokens] = list(ref_tokens)
    right: list[_ClauseTokens] = list(ref_tokens)
    row_pos: list[int] = list(range(len(ref_clauses)))

    order = sorted(
        (i for i in range(len(seqs)) if i != ref_idx),
        key=lambda i: (abs(i - ref_idx), i),
    )
    for i in order:
        ver_clauses = seqs[i]
        if not ver_clauses:
            continue
        frontier = left if i < ref_idx else right
        ver_tokens = [_clause_tokens(c.node.text or "") for c in ver_clauses]
        # Every row open so far lacks a version-i member; rows opened by this
        # version's own unbound clauses (below) are not candidates for it.
        n_open = len(rows)
        bound: set[int] = set()
        for ci, r, sim in _bind_by_similarity(ver_tokens, frontier[:n_open], row_pos[:n_open]):
            rows[r][i] = ver_clauses[ci]
            prev = row_sim[r]
            row_sim[r] = sim if prev is None else min(prev, sim)
            frontier[r] = ver_tokens[ci]
            bound.add(ci)
        for ci, clause in enumerate(ver_clauses):
            if ci in bound:
                continue
            rows.append({i: clause})
            row_sim.append(None)
            left.append(ver_tokens[ci])
            right.append(ver_tokens[ci])
            row_pos.append(ci)

    alignments: list[ClauseAlignment] = []
    for row_idx, row_members in enumerate(rows):
        row_dict: dict[int, ClassifiedClause | None] = dict(row_members)
        is_extra = row_idx >= len(ref_clauses)
        sim_score: float | None = row_sim[row_idx]

        # Determine if this bucket is ambiguous and needs judge intervention.
        has_none_slot = any(row_dict.get(i) is None for i in range(len(version_ids)))
        low_confidence = sim_score is not None and sim_score < ALIGNMENT_AMBIGUITY_THRESHOLD
        is_flagged = has_none_slot or low_confidence or is_extra

        if is_flagged and alignment_judge is not None:
            # Collect before/after clause lists for the judge.
            # "before" = ref version clause(s), "after" = non-ref clauses,
            # concatenated in version-index order. after_items keeps track of
            # which actual version each after_clause came from, so a bucket
            # spanning more than two versions is reconstructed correctly.
            before_clauses = [c for c in [row_dict.get(ref_idx)] if c is not None]
            after_items: list[tuple[int, ClassifiedClause]] = [
                (i, c) for i, c in sorted(row_dict.items()) if i != ref_idx and c is not None
            ]
            after_clauses = [c for _, c in after_items]

            # Judge returns a pairing/split/merge map; each pairing becomes
            # its own output row so multi-pair verdicts (splits/merges)
            # aren't collapsed into a single overwritten row. Fall back to
            # the deterministic result if the judge fails or returns nothing.
            new_rows: list[ClauseAlignment] | None = None
            try:
                pairing = alignment_judge.judge_bucket(before_clauses, after_clauses)
                if not pairing:
                    raise ValueError("alignment judge returned no pairings")
                built: list[ClauseAlignment] = []
                for before_i, after_i in pairing:
                    judge_row: dict[int, ClassifiedClause | None] = {}
                    if before_i is not None and 0 <= before_i < len(before_clauses):
                        judge_row[ref_idx] = before_clauses[before_i]
                    if after_i is not None and 0 <= after_i < len(after_clauses):
                        version_idx, other_clause = after_items[after_i]
                        judge_row[version_idx] = other_clause
                    row_slots = tuple(
                        AlignmentSlot(
                            version=version_ids[i],
                            clause=judge_row.get(i),
                            alignment_confidence=sim_score,
                        )
                        for i in range(len(version_ids))
                    )
                    built.append(ClauseAlignment(taxonomy_id=taxonomy_id, slots=row_slots))
                new_rows = built
            except Exception:  # noqa: BLE001
                new_rows = None

            if new_rows is not None:
                alignments.extend(new_rows)
                continue

            # Judge failed or returned nothing; fall back to deterministic result.
            slots = tuple(
                AlignmentSlot(
                    version=version_ids[i],
                    clause=row_dict.get(i),
                    alignment_confidence=sim_score,
                )
                for i in range(len(version_ids))
            )
        else:
            slots = tuple(
                AlignmentSlot(
                    version=version_ids[i],
                    clause=row_dict.get(i),
                    alignment_confidence=sim_score,
                )
                for i in range(len(version_ids))
            )
        alignments.append(ClauseAlignment(taxonomy_id=taxonomy_id, slots=slots))

    return alignments


def _bind_by_similarity(
    clause_tokens: list[_ClauseTokens],
    row_tokens: list[_ClauseTokens],
    row_pos: list[int],
) -> list[tuple[int, int, float]]:
    """Bind clauses to rows (issue #222).

    A pair is a candidate when its token-set Jaccard is at least
    ``ALIGNMENT_AMBIGUITY_THRESHOLD`` (a *primary* bind) or it passes the
    localized-edit rescue (:func:`_is_localized_edit`). Candidates are
    ranked by Jaccard, highest first, then positional distance
    (``|clause index - row_pos|``), then index, so identical duplicates keep
    their document order; a pair binds only when both sides are still free.
    A rescue bind is additionally refused when a competing pair that is
    still free — another free clause for the same row, or another free row
    for the same clause — has a Jaccard within ``ALIGNMENT_RESCUE_MARGIN`` of
    its own or higher: the rescue never settles an ambiguous choice.
    Identical token sets (Jaccard 1.0) are paired by hashing first, so a
    bucket of unchanged clauses never pays for the pairwise matrix. Returns
    ``[(clause_index, row_index, jaccard), ...]`` — the reported similarity
    is always the Jaccard, never a rescue score.
    """
    out: list[tuple[int, int, float]] = []
    free_rows = set(range(len(row_tokens)))
    free_clauses = set(range(len(clause_tokens)))

    rows_by_tokens: dict[frozenset[str], list[int]] = {}
    for r, toks in enumerate(row_tokens):
        rows_by_tokens.setdefault(toks.vocab, []).append(r)
    for ci, toks in enumerate(clause_tokens):
        same = rows_by_tokens.get(toks.vocab)
        if not same:
            continue
        r = min(same, key=lambda k: (abs(ci - row_pos[k]), k))
        same.remove(r)
        out.append((ci, r, 1.0))
        free_rows.discard(r)
        free_clauses.discard(ci)

    # Jaccard of every remaining pair that could matter: as a candidate
    # (>= ALIGNMENT_RESCUE_MIN_JACCARD) or as a competitor that blocks a
    # rescue bind (>= that minus ALIGNMENT_RESCUE_MARGIN). Pairs whose set
    # sizes alone bound the Jaccard below that floor are skipped.
    floor = ALIGNMENT_RESCUE_MIN_JACCARD - ALIGNMENT_RESCUE_MARGIN
    jac: dict[tuple[int, int], float] = {}
    candidates: list[tuple[float, int, int, bool]] = []
    for ci in sorted(free_clauses):
        ta = clause_tokens[ci]
        for r in sorted(free_rows):
            tb = row_tokens[r]
            small, large = sorted((len(ta.vocab), len(tb.vocab)))
            if large == 0 or small / large < floor:
                continue
            j = _jaccard(ta.vocab, tb.vocab)
            if j < floor:
                continue
            jac[(ci, r)] = j
            if j >= ALIGNMENT_AMBIGUITY_THRESHOLD:
                candidates.append((j, ci, r, False))
            elif j >= ALIGNMENT_RESCUE_MIN_JACCARD and _is_localized_edit(ta, tb):
                candidates.append((j, ci, r, True))
    candidates.sort(key=lambda t: (-t[0], abs(t[1] - row_pos[t[2]]), t[1], t[2]))
    for j, ci, r, rescue in candidates:
        if ci not in free_clauses or r not in free_rows:
            continue
        if rescue and _rescue_is_contested(j, ci, r, jac, free_clauses, free_rows):
            continue
        out.append((ci, r, j))
        free_clauses.discard(ci)
        free_rows.discard(r)
    return out


def _rescue_is_contested(
    j: float,
    ci: int,
    r: int,
    jac: dict[tuple[int, int], float],
    free_clauses: set[int],
    free_rows: set[int],
) -> bool:
    """True when a rescue bind of clause ``ci`` to row ``r`` (Jaccard ``j``)
    would take a partner from a still-free competing pair whose Jaccard is
    within ``ALIGNMENT_RESCUE_MARGIN`` of ``j`` or higher."""
    bar = j - ALIGNMENT_RESCUE_MARGIN
    if any(jac.get((c, r), 0.0) >= bar for c in free_clauses if c != ci):
        return True
    return any(jac.get((ci, k), 0.0) >= bar for k in free_rows if k != r)


@dataclass(frozen=True, eq=False)
class _ClauseTokens:
    """One clause's non-stop-word tokens, in order and as a set."""

    seq: tuple[str, ...]
    vocab: frozenset[str]


def _clause_tokens(text: str) -> _ClauseTokens:
    seq = tuple(w for w in _normalize(text).split() if w not in _STOP_WORDS)
    return _ClauseTokens(seq=seq, vocab=frozenset(seq))


def _is_localized_edit(a: _ClauseTokens, b: _ClauseTokens) -> bool:
    """The localized-edit test of the bucket path's rescue bind (issue #222).

    True when the shorter clause has at least ``ALIGNMENT_RESCUE_MIN_TOKENS``
    content tokens and ``difflib.SequenceMatcher(autojunk=False)`` turns one
    token sequence into the other with EXACTLY ONE non-``equal`` opcode — a
    single contiguous replace, insert or delete, the shape of a changed
    number or an appended (or inserted) proviso. Short boilerplate that
    shares a skeleton with an unrelated longer clause differs in several
    separate spans and fails. The Jaccard floor is the caller's check.
    """
    if min(len(a.seq), len(b.seq)) < ALIGNMENT_RESCUE_MIN_TOKENS:
        return False
    opcodes = SequenceMatcher(None, a.seq, b.seq, autojunk=False).get_opcodes()
    return sum(1 for tag, *_ in opcodes if tag != "equal") == 1


def _normalize(text: str) -> str:
    s = text.lower()
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _tokens(text: str) -> frozenset[str]:
    return frozenset(w for w in _normalize(text).split() if w not in _STOP_WORDS)


def _jaccard(ta: frozenset[str], tb: frozenset[str]) -> float:
    """Jaccard similarity between two precomputed token sets."""
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)
