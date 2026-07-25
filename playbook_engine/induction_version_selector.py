"""Content-based representative-version selection for taxonomy induction.

``induce-taxonomy`` picks one version per agreement to represent that
agreement's clause headings.  Previously this was pure filename
natural-sort ("the highest-versioned file wins"), which contradicts the
engine's documented "filenames don't matter" contract
(``docs/CORPUS-LAYOUT.md:23``) and silently degrades induction on corpora
with uninformative filenames (issue #169).

This module reuses the same content-based path the rest of the engine uses
to reconstruct a negotiation trail (``signed_detector`` +
``version_orderer``) instead of re-deriving version order from names:

1. Prefer the version ``signed_detector.detect_signed`` judges to be the
   executed (signed) copy — basis ``"signed"``.
2. Otherwise fall back to the terminal of the minimum-edit-distance chain
   (``version_orderer``) — basis ``"chain_terminal"`` — PROVIDED that
   terminal is unambiguous (see ``_terminal_candidates``).
3. When the content evidence is genuinely tied — e.g. exactly two versions
   with no signed cues, where the pairwise edit distance is symmetric so
   either version is an equally valid chain terminal — fall back to
   filename natural-sort as a last-resort tiebreak — basis
   ``"filename_tiebreak"``.
4. A single version file needs no arbitration — basis ``"single"``.

Induction reuses the clause trees the caller already extracted (ingestion
happens once, in the caller); this module only decides which of those
already-extracted trees is representative.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from pathlib import Path

from playbook_engine.clause_tree import ClauseTree
from playbook_engine.natural_sort import natural_sort_key
from playbook_engine.signed_detector import SignedStatus, detect_signed
from playbook_engine.version_orderer import (
    EXHAUSTIVE_THRESHOLD,
    VersionInput,
    order_versions,
    pairwise_distances,
)

_COST_TOL = 1e-9


@dataclass(frozen=True)
class VersionCandidate:
    """One already-ingested version file for a single agreement."""

    path: Path
    tree: ClauseTree


@dataclass(frozen=True)
class SelectedVersion:
    """The chosen representative version for one agreement.

    Attributes:
        path:  The winning version file.
        tree:  Its already-extracted ``ClauseTree`` (reused, not re-ingested).
        basis: One of ``"single"``, ``"signed"``, ``"chain_terminal"``, or
               ``"filename_tiebreak"``.
    """

    path: Path
    tree: ClauseTree
    basis: str


def select_representative_version(candidates: list[VersionCandidate]) -> SelectedVersion:
    """Pick the representative version among *candidates* for one agreement.

    Args:
        candidates: All successfully-ingested version files for one
                    agreement (any order); must be non-empty.

    Returns:
        A ``SelectedVersion`` naming the winner and the basis for the choice.
    """
    if not candidates:
        raise ValueError("select_representative_version requires at least one candidate")
    if len(candidates) == 1:
        c = candidates[0]
        return SelectedVersion(path=c.path, tree=c.tree, basis="single")

    statuses: dict[str, SignedStatus] = {c.path.stem: detect_signed(c.tree) for c in candidates}

    # 1. Prefer the detected signed copy.
    signed_candidates = [c for c in candidates if statuses[c.path.stem].signed]
    if signed_candidates:
        best_conf = max(statuses[c.path.stem].confidence for c in signed_candidates)
        tied = [
            c
            for c in signed_candidates
            if math.isclose(statuses[c.path.stem].confidence, best_conf, abs_tol=_COST_TOL)
        ]
        winner = (
            tied[0] if len(tied) == 1 else max(tied, key=lambda c: natural_sort_key(c.path.stem))
        )
        return SelectedVersion(path=winner.path, tree=winner.tree, basis="signed")

    # 2/3. No signed version — fall back to the edit-distance chain
    # terminal, but only when it is unambiguous; otherwise filename
    # natural-sort is the last-resort tiebreak.
    version_inputs = [
        VersionInput(version_id=c.path.stem, tree=c.tree, signed=statuses[c.path.stem])
        for c in candidates
    ]
    terminal_ids = _terminal_candidates(version_inputs)

    if len(terminal_ids) == 1:
        winner_id = next(iter(terminal_ids))
        winner = next(c for c in candidates if c.path.stem == winner_id)
        return SelectedVersion(path=winner.path, tree=winner.tree, basis="chain_terminal")

    tied_candidates = [c for c in candidates if c.path.stem in terminal_ids]
    winner = max(tied_candidates, key=lambda c: natural_sort_key(c.path.stem))
    return SelectedVersion(path=winner.path, tree=winner.tree, basis="filename_tiebreak")


def _terminal_candidates(version_inputs: list[VersionInput]) -> set[str]:
    """Return the set of version ids that could be the chain terminal.

    With no signed anchor, ``order_versions`` optimises over an inherently
    *undirected* path (edit distance is symmetric), so the ``ordered_ids[-1]``
    it returns is only meaningful as "the" terminal when it is the UNIQUE
    version that can occupy that position in an optimal (minimum total cost)
    permutation. With exactly two versions, for example, the two possible
    chains have identical cost by construction (there's only one pairwise
    distance to sum either way), so either end is an equally valid terminal
    — that is a genuine tie, not a content-derived answer.

    This performs the same exhaustive permutation search ``order_versions``
    does internally (bounded by the same ``EXHAUSTIVE_THRESHOLD``) and
    returns every id that wins the terminal slot under some cost-optimal
    permutation, so the caller can detect a real tie instead of silently
    trusting a lexicographic/greedy artifact as "the" terminal.
    """
    ids = [vi.version_id for vi in version_inputs]
    dist = pairwise_distances(version_inputs)

    if len(ids) > EXHAUSTIVE_THRESHOLD:
        # Too many versions to search exhaustively — trust order_versions's
        # own (greedy) terminal rather than pay for a combinatorial search;
        # induction corpora this large are already an edge case.
        order = order_versions(version_inputs)
        return {order.ordered_ids[-1]} if order.ordered_ids else set()

    best_cost = math.inf
    terminals: set[str] = set()
    for perm in itertools.permutations(ids):
        cost = sum(dist[(perm[i], perm[i + 1])] for i in range(len(perm) - 1))
        if cost < best_cost - _COST_TOL:
            best_cost = cost
            terminals = {perm[-1]}
        elif math.isclose(cost, best_cost, abs_tol=_COST_TOL):
            terminals.add(perm[-1])
    return terminals
