"""Reversal detection — L4 pipeline stage.

Identifies text spans that were **inserted in a draft but removed before the
signed terminal** — the cleanest "explicitly rejected" signal derivable from
diffs alone (no status labels required).

These correspond to OPF ``outcome: proposed_then_reversed`` (§2.1).

Algorithm (fully deterministic, no LLM):

1. Build a token set for each clause in the signed (final) version from the
   ``net`` diff's ``text_after`` values.
2. Scan every consecutive diff for ``"modified"`` or ``"added"`` clauses.
3. For each ``"insert"`` or ``"replace"`` hunk (or the whole clause for
   ``"added"``), collect the proposed word tokens.
4. If those tokens are **not a subset** of the signed token set for the same
   ``taxonomy_id``, the proposal was reversed → emit a ``ReversalRecord``.

Token-subset check: conservative but correct.  A proposed phrase whose content
words are all retained in the signed text is treated as *accepted* even if the
exact phrasing changed.  A phrase whose content words are wholly absent from
the signed text is unambiguously reversed.
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
        version_removed:   The signed terminal (last version) — the proposal
                           is confirmed absent from the executed text.
        proposed_text:     The word tokens that were proposed then reversed.
        char_span:         ``ClauseNode.char_span`` of ``clause_path`` in
                           ``version_inserted`` (issue #108), or ``None`` when
                           unavailable. Threaded from the ``ClauseDiff`` this
                           reversal was detected on so the citation built from
                           this record is one-click verifiable, not just a
                           clause-path/ordinal pair.
    """

    taxonomy_id: str | None
    clause_path: str
    version_inserted: str
    version_removed: str
    proposed_text: str
    char_span: tuple[int, int] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "taxonomy_id": self.taxonomy_id,
            "clause_path": self.clause_path,
            "version_inserted": self.version_inserted,
            "version_removed": self.version_removed,
            "proposed_text": self.proposed_text,
            "char_span": list(self.char_span) if self.char_span else None,
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_reversals(doc_diff: DocumentDiff) -> list[ReversalRecord]:
    """Detect text spans inserted in a draft but absent from the signed terminal.

    Args:
        doc_diff: ``DocumentDiff`` produced by ``diff_aligned()``.

    Returns:
        One ``ReversalRecord`` per distinct reversal event.  Empty list if no
        reversals are found or if there are no consecutive diffs.
    """
    if not doc_diff.consecutive:
        return []

    signed_version = doc_diff.version_order[-1]

    # Build token sets keyed by clause_path in the signed (final) version.
    # Using clause_path (not taxonomy_id) ensures clause-instance precision:
    # two clauses that share a taxonomy_id (or both have None) are tracked
    # independently, preventing cross-contamination of reversal outcomes.
    signed_tokens: dict[str, frozenset[str]] = {}
    for cd in doc_diff.net.diffs:
        if cd.kind != "removed" and cd.text_after and cd.clause_path_after:
            path = cd.clause_path_after
            existing = signed_tokens.get(path, frozenset())
            signed_tokens[path] = existing | _tokens(cd.text_after)

    reversals: list[ReversalRecord] = []

    for vdiff in doc_diff.consecutive:
        for cd in vdiff.diffs:
            if cd.kind not in ("modified", "added"):
                continue

            # The clause instance path in the draft (the "after" side of this diff).
            clause_path = cd.clause_path_after or cd.clause_path_before or "?"

            # Collect proposed tokens for this clause diff.
            if cd.kind == "added":
                proposed_text = cd.text_after
                proposed_toks = _tokens(proposed_text)
            else:
                proposed_parts: list[str] = []
                proposed_toks = frozenset()
                for hunk in cd.hunks:
                    if hunk.kind in ("insert", "replace") and hunk.new_text:
                        proposed_parts.append(hunk.new_text)
                        proposed_toks |= _tokens(hunk.new_text)
                proposed_text = " ".join(proposed_parts)

            if not proposed_toks:
                continue

            # Compare against signed version's tokens for this clause instance.
            signed = signed_tokens.get(clause_path, frozenset())
            if not proposed_toks.issubset(signed):
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
                    )
                )

    return reversals


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _tokens(text: str) -> frozenset[str]:
    words = re.findall(r"\w+", text.lower())
    return frozenset(w for w in words if w not in _STOP_WORDS)
