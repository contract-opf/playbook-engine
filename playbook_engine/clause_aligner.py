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
1. Group each remaining (unmatched) clause list by ``taxonomy_id``.
2. Collect all taxonomy_ids in first-appearance order (v0 → v1 → ...).
3. For each taxonomy_id bucket, align the per-version clause sequences:
   a. Identical counts across all versions → zip in order (exact alignment).
   b. Differing counts → greedy text-Jaccard matching against the
      version with the most clauses; unmatched slots become ``None``.
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
  - Splits/merges: handled by greedy match within the bucket; extra clauses
    become new rows.  When counts are identical, remaining clauses are zipped
    by position — two same-taxonomy_id clauses that swap positions are caught
    by the global move phase when their text is substantial enough to match.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
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

#: Jaccard similarity below this threshold triggers an ``AlignmentJudge`` call
#: for the corresponding bucket (if a judge is configured).
ALIGNMENT_AMBIGUITY_THRESHOLD: float = 0.70

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

    Called only on ambiguous buckets — those containing a ``None``-matched
    slot, a low-Jaccard pair (below ``ALIGNMENT_AMBIGUITY_THRESHOLD``), an
    ``extra_rows`` bucket (count mismatch), or a detected reorder.

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
        alignment_judge:     Optional judge called on ambiguous buckets (those
                             with a ``None``-matched slot, low-Jaccard score,
                             count mismatch, or detected reorder).  When
                             ``None`` the deterministic greedy algorithm is
                             used for all buckets.

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
    move_rows, matched_keys = _match_moves(version_ids, full_lists)

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


def _match_moves(
    version_ids: list[str],
    full_lists: list[list[ClassifiedClause]],
) -> tuple[list[ClauseAlignment], set[tuple[int, int]]]:
    """Global move-matching phase: pair clauses across adjacent versions by
    content similarity anywhere in the document, then chain the pairs across
    versions into aligned rows.

    Returns ``(move_rows, matched_keys)`` where ``matched_keys`` is the set of
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

    move_rows: list[ClauseAlignment] = []
    matched_keys: set[tuple[int, int]] = set()
    for vi in range(n_versions - 1):
        for ci in range(len(full_lists[vi])):
            if vi > 0 and ci in incoming[vi]:
                continue  # continuation of an earlier chain
            if ci not in links[vi]:
                continue  # no content match — positional path handles it
            # Follow the chain forward from (vi, ci).
            chain: list[tuple[int, int]] = [(vi, ci)]
            bases: list[str] = []
            sims: list[float] = []
            v, c = vi, ci
            while v < n_versions - 1 and c in links[v]:
                nxt, basis, sim = links[v][c]
                bases.append(basis)
                sims.append(sim)
                chain.append((v + 1, nxt))
                v, c = v + 1, nxt
            clause_by_version: dict[int, ClassifiedClause] = {
                v_idx: full_lists[v_idx][c_idx] for v_idx, c_idx in chain
            }
            matched_keys.update(chain)
            confidence = min(sims)
            slots = tuple(
                AlignmentSlot(
                    version=version_ids[i],
                    clause=clause_by_version.get(i),
                    alignment_confidence=confidence if i in clause_by_version else None,
                )
                for i in range(n_versions)
            )
            # The row's taxonomy_id follows the latest version's classification
            # (deviation-vs-template is assessed on the net/signed side).
            last_clause = clause_by_version[max(clause_by_version)]
            move_rows.append(
                ClauseAlignment(
                    taxonomy_id=last_clause.classification.taxonomy_id,
                    slots=slots,
                    match_basis=(
                        "content_exact"
                        if all(b == "content_exact" for b in bases)
                        else "content_jaccard"
                    ),
                )
            )

    return move_rows, matched_keys


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
    """Align per-version clause sequences within one taxonomy_id bucket."""
    # Fast path: all versions have identical clause counts → zip in order.
    counts = {len(s) for s in seqs}
    if len(counts) == 1:
        n = len(seqs[0])
        return [
            ClauseAlignment(
                taxonomy_id=taxonomy_id,
                slots=tuple(
                    AlignmentSlot(version=version_ids[i], clause=seqs[i][j])
                    for i in range(len(version_ids))
                ),
            )
            for j in range(n)
        ]

    # Slow path: differing counts.  Use the longest sequence as the reference
    # frame; greedy-match other versions to it by text-Jaccard similarity.
    ref_idx = max(range(len(seqs)), key=lambda i: len(seqs[i]))
    ref_clauses = seqs[ref_idx]

    # matched[j] tracks the per-version clause mapped to reference slot j.
    # matched_sim[j] tracks the best Jaccard score for reference slot j.
    matched: list[dict[int, ClassifiedClause | None]] = [{ref_idx: c} for c in ref_clauses]
    matched_sim: list[float | None] = [None] * len(ref_clauses)
    extra_rows: list[dict[int, ClassifiedClause | None]] = []

    # Precompute reference-clause token sets once per bucket (not per pair, and
    # not per non-reference version) to avoid O(n^2) re-tokenization below.
    ref_tokens = [_tokens(rc.node.text or "") for rc in ref_clauses]

    for i, ver_clauses in enumerate(seqs):
        if i == ref_idx:
            continue

        unmatched_ref: list[int] = list(range(len(ref_clauses)))

        for clause in ver_clauses:
            if not unmatched_ref:
                extra_rows.append({i: clause})
                continue

            clause_tokens = _tokens(clause.node.text or "")
            best_j = unmatched_ref[0]
            best_sim = _jaccard(clause_tokens, ref_tokens[best_j])
            for j in unmatched_ref[1:]:
                candidate_sim = _jaccard(clause_tokens, ref_tokens[j])
                if candidate_sim > best_sim:
                    best_sim = candidate_sim
                    best_j = j
            sim = best_sim
            if sim > 0.0:
                matched[best_j][i] = clause
                # Store the minimum Jaccard seen for this reference slot across
                # all non-reference versions (worst-case confidence for the row).
                prev = matched_sim[best_j]
                matched_sim[best_j] = sim if prev is None else min(prev, sim)
                unmatched_ref.remove(best_j)
            else:
                extra_rows.append({i: clause})

        # Reference slots still unmatched → this version has no clause there.
        for j in unmatched_ref:
            matched[j].setdefault(i, None)

    alignments: list[ClauseAlignment] = []
    for row_idx, row_dict in enumerate(matched + extra_rows):
        is_extra = row_idx >= len(matched)
        sim_score: float | None = None if is_extra else matched_sim[row_idx]

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
