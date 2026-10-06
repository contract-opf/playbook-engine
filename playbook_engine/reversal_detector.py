"""Reversal detection — L4 pipeline stage.

Identifies text spans that were **inserted in a draft but removed before the
signed terminal** — the cleanest "explicitly rejected" signal derivable from
diffs alone (no status labels required).

These correspond to OPF ``outcome: proposed_then_reversed`` (§2.1).

Algorithm (fully deterministic, no LLM):

1. Build a token set for each clause in the signed (final) version from the
   ``net`` diff's ``text_after`` values.
2. Scan every consecutive diff for ``"modified"`` or ``"added"`` clauses.
3. Each ``"insert"`` or ``"replace"`` hunk is one proposal (the whole clause
   is one proposal for ``"added"``); collect its word tokens.
4. Per proposal (per hunk), compute
   ``retained = |tokens(hunk) ∩ tokens(signed clause)| / |tokens(hunk)|``
   against the signed token set for the same logical clause (matched by
   ``ClauseDiff.alignment_index``, not by per-version ``clause_path`` — see
   below). That hunk was reversed only when
   ``retained < REVERSAL_RETAINED_THRESHOLD`` (0.5) → emit a
   ``ReversalRecord`` carrying that hunk's text and ``retained`` (issue #222).
   Hunks are never pooled: pooling would let an accepted sibling edit in the
   same round dilute a refused one below detection.

Matching is keyed by ``alignment_index`` rather than ``clause_path`` because
``clause_path`` is per-version dotted numbering: the aligner
(``clause_aligner._match_moves``, the v5 relocation feature) can legitimately
align the same logical clause under a different path in each version (e.g. a
later insertion renumbers everything after it). Keying the signed-token
lookup by the draft version's path would then compare a proposal against the
wrong clause's signed text, both fabricating reversals for clauses that
actually survived and, in the mirror case, masking real reversals.

Retained-token ratio (issue #222): a proposal is *accepted* when at least half
of its content words survive in the signed clause, even if the exact phrasing
changed — a counter-proposal accepted with one word changed is not a refusal.
A proposal most of whose content words are absent from the signed clause was
reversed. The previous rule (reversed unless EVERY proposed word survived) read
any accepted-with-edits proposal as refused.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from playbook_engine.clause_differ import DocumentDiff

# ---------------------------------------------------------------------------
# Stop words (same set as rest of pipeline)
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

#: A proposal is a reversal only when less than this share of its content
#: tokens (stop words excluded) survives in the signed clause (issue #222):
#: ``retained = |tokens(proposal) ∩ tokens(signed clause)| / |tokens(proposal)|``.
#: At or above it the proposal was accepted, possibly with edits.
REVERSAL_RETAINED_THRESHOLD: float = 0.5

# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReversalRecord:
    """A text proposal that was inserted in a draft and removed before signing.

    Corresponds to OPF ``outcome: proposed_then_reversed``.

    Attributes:
        taxonomy_id:       Taxonomy entry of the affected clause, or ``None``.
        clause_path:       Clause instance path in the draft version (e.g.
                           ``"3.1"``).  Used to match the specific clause
                           instance rather than the taxonomy bucket, so that
                           two clauses sharing a ``taxonomy_id`` (or both
                           ``None``) cannot cross-contaminate outcomes.
        version_inserted:  Version in which the proposed text first appeared.
        version_removed:   The signed terminal (last version, a detected signed
                           copy — see ``detect_reversals``' ``has_signed_copy``)
                           — the proposal is confirmed absent from the
                           executed text.
        proposed_text:     The text of the one proposal (insert/replace hunk,
                           or whole added clause) that was reversed.
        char_span:         ``ClauseNode.char_span`` of ``clause_path`` in
                           ``version_inserted`` (issue #108), or ``None`` when
                           unavailable. Threaded from the ``ClauseDiff`` this
                           reversal was detected on so the citation built from
                           this record is one-click verifiable, not just a
                           clause-path/ordinal pair.
        retained:          Share of this hunk's content tokens present in
                           the signed clause (issue #222) — always below
                           ``REVERSAL_RETAINED_THRESHOLD`` for a detected
                           reversal; ``None`` only for records built outside
                           ``detect_reversals``.
        alignment_confidence: The ``ClauseDiff.alignment_confidence`` of the
                           diff the reversal was detected on — how strongly
                           the draft clause was bound to the clause it is
                           compared with — or ``None`` when that row binds
                           nothing across versions.
    """

    taxonomy_id: str | None
    clause_path: str
    version_inserted: str
    version_removed: str
    proposed_text: str
    char_span: tuple[int, int] | None = None
    retained: float | None = None
    alignment_confidence: float | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "taxonomy_id": self.taxonomy_id,
            "clause_path": self.clause_path,
            "version_inserted": self.version_inserted,
            "version_removed": self.version_removed,
            "proposed_text": self.proposed_text,
            "char_span": list(self.char_span) if self.char_span else None,
        }
        # Omitted when never computed, so an absent key reads back as None.
        if self.retained is not None:
            d["retained"] = round(self.retained, 6)
        if self.alignment_confidence is not None:
            d["alignment_confidence"] = round(self.alignment_confidence, 6)
        return d


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_reversals(
    doc_diff: DocumentDiff, *, has_signed_copy: bool = True
) -> list[ReversalRecord]:
    """Detect text spans inserted in a draft but absent from the signed terminal.

    Args:
        doc_diff:        ``DocumentDiff`` produced by ``diff_aligned()``.
        has_signed_copy: Whether the last version of ``doc_diff.version_order``
                         is a DETECTED signed copy (``VersionOrder.signed_id``
                         anchors the chain there). When False (issue #221)
                         there is no signed terminal — the last version is
                         only the last draft in a content-derived (for an
                         unsigned deal, often tie-broken) order — so text
                         absent from it was not "refused before signing" and
                         no reversal is reported: ``[]``.

    Returns:
        One ``ReversalRecord`` per distinct reversal event.  Empty list if no
        reversals are found, if there are no consecutive diffs, or if the deal
        has no detected signed copy.
    """
    if not has_signed_copy or not doc_diff.consecutive:
        return []

    signed_version = doc_diff.version_order[-1]

    # Build token sets keyed by alignment_index (logical-clause identity) in
    # the signed (final) version, not by clause_path. clause_path is
    # per-version numbering and is not stable across versions when clauses
    # are inserted/removed elsewhere in the document (see module docstring);
    # alignment_index is the stable row position in the alignments list both
    # the net diff and every consecutive diff were built from
    # (clause_differ._version_diff), so it identifies the same logical clause
    # on both sides of the lookup below regardless of renumbering. This still
    # preserves clause-instance precision (two clauses sharing a taxonomy_id,
    # or both ``None``, get distinct alignment_index values and are tracked
    # independently).
    signed_tokens: dict[int, frozenset[str]] = {}
    for cd in doc_diff.net.diffs:
        if (
            cd.kind != "removed"
            and cd.text_after
            and cd.clause_path_after
            and cd.alignment_index is not None
        ):
            idx = cd.alignment_index
            existing = signed_tokens.get(idx, frozenset())
            signed_tokens[idx] = existing | _tokens(cd.text_after)

    reversals: list[ReversalRecord] = []

    for vdiff in doc_diff.consecutive:
        for cd in vdiff.diffs:
            if cd.kind not in ("modified", "added"):
                continue

            # The clause instance path in the draft (the "after" side of this diff).
            clause_path = cd.clause_path_after or cd.clause_path_before or "?"

            # Compare against signed version's tokens for this clause instance,
            # keyed by alignment_index (logical-clause identity) — not by
            # clause_path, which is per-version and may have been renumbered
            # by the time the signed version was reached (see module docstring).
            signed = (
                signed_tokens.get(cd.alignment_index, frozenset())
                if cd.alignment_index is not None
                else frozenset()
            )

            # The unit of proposal is the HUNK (issue #222): each insert/replace
            # hunk is scored on its own, so an accepted sibling edit in the same
            # round cannot dilute a refused one below detection. An ``added``
            # clause is a single proposal.
            if cd.kind == "added":
                proposals = [cd.text_after]
            else:
                proposals = [
                    hunk.new_text
                    for hunk in cd.hunks
                    if hunk.kind in ("insert", "replace") and hunk.new_text
                ]

            for proposed_text in proposals:
                proposed_toks = _tokens(proposed_text)
                if not proposed_toks:
                    continue
                retained = len(proposed_toks & signed) / len(proposed_toks)
                if retained < REVERSAL_RETAINED_THRESHOLD:
                    reversals.append(
                        ReversalRecord(
                            taxonomy_id=cd.taxonomy_id,
                            clause_path=clause_path,
                            version_inserted=vdiff.version_after,
                            version_removed=signed_version,
                            proposed_text=proposed_text,
                            char_span=(
                                cd.char_span_after
                                if cd.clause_path_after is not None
                                else cd.char_span_before
                            ),
                            retained=retained,
                            alignment_confidence=cd.alignment_confidence,
                        )
                    )

    return reversals


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _tokens(text: str) -> frozenset[str]:
    words = re.findall(r"\w+", text.lower())
    return frozenset(w for w in words if w not in _STOP_WORDS)
