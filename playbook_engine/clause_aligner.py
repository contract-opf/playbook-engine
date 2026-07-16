"""Cross-version clause alignment — L3 pipeline stage.

Matches 'the same clause' across versions of one document so that downstream
diff stages operate clause-scoped rather than whole-document.

Algorithm (fully deterministic, no LLM):

1. Group each version's ``ClassifiedClause`` list by ``taxonomy_id``.
2. Collect all taxonomy_ids in first-appearance order (v0 → v1 → ...).
3. For each taxonomy_id bucket, align the per-version clause sequences:
   a. Identical counts across all versions → zip in order (exact alignment).
   b. Differing counts → greedy text-Jaccard matching against the
      version with the most clauses; unmatched slots become ``None``.
4. Return one ``ClauseAlignment`` per logical clause, preserving the
   first-appearance order of taxonomy_ids.

Handles:
  - Renumbering: §1→§2 is irrelevant; taxonomy_id is the key.
  - Insertions: new taxonomy_id in a later version → AlignmentSlot(clause=None)
    for earlier versions.
  - Deletions: taxonomy_id absent in a later version → AlignmentSlot(clause=None)
    for that version.
  - Splits/merges: handled by greedy match within the bucket; extra clauses
    become new rows.  When counts are identical, clauses are zipped by position —
    two same-taxonomy_id clauses that swap positions would be mis-paired.
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
    """

    taxonomy_id: str | None
    slots: tuple[AlignmentSlot, ...]

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

    # 1. Build per-version groups: {taxonomy_id: [ClassifiedClause, ...]}
    VersionGroups = dict[str | None, list[ClassifiedClause]]
    per_version: list[VersionGroups] = []
    for _, clauses in classified_versions:
        groups: VersionGroups = {}
        for c in clauses:
            tid = c.classification.taxonomy_id
            groups.setdefault(tid, []).append(c)
        per_version.append(groups)

    # 2. Collect all taxonomy_ids in first-appearance order.
    seen_tids: set[str | None] = set()
    ordered_tids: list[str | None] = []
    for groups in per_version:
        for tid in groups:
            if tid not in seen_tids:
                seen_tids.add(tid)
                ordered_tids.append(tid)

    # 3. For each taxonomy_id, build alignment rows.
    result: list[ClauseAlignment] = []
    for tid in ordered_tids:
        seqs = [groups.get(tid, []) for groups in per_version]
        result.extend(_align_seqs(tid, version_ids, seqs, alignment_judge=alignment_judge))

    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


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

    for i, ver_clauses in enumerate(seqs):
        if i == ref_idx:
            continue

        unmatched_ref: list[int] = list(range(len(ref_clauses)))

        for clause in ver_clauses:
            if not unmatched_ref:
                extra_rows.append({i: clause})
                continue

            best_j = max(
                unmatched_ref,
                key=lambda j: _text_jaccard(clause, ref_clauses[j]),
            )
            sim = _text_jaccard(clause, ref_clauses[best_j])
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


def _text_jaccard(a: ClassifiedClause, b: ClassifiedClause) -> float:
    """Token-Jaccard similarity on clause text (stop-words removed)."""
    ta = _tokens(a.node.text or "")
    tb = _tokens(b.node.text or "")
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)
