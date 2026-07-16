"""Version orderer — L2 structure layer.

Orders versions of one document as the minimum-edit-distance chain anchored
at the signed (executed) copy.  No status labels are required.

Algorithm:
1. Extract a text fingerprint from each ``ClauseTree``: one whitespace-
   collapsed element per node (heading + body), in clause-path order —
   format-independent, so DOCX/PDF/RTF renderings of identical content
   don't get inflated distances from differing physical line-wrap shapes
   (see ``_fingerprint``).
2. Compute all pairwise edit distances using ``difflib.SequenceMatcher``;
   distance = 1 − similarity_ratio ∈ [0, 1].
3. Anchor: the version with ``SignedStatus.signed=True`` is fixed as the last
   element of the chain.
4. Order: find the permutation of the remaining versions that minimises the
   sum of consecutive pairwise distances.  Exhaustive search for
   N ≤ ``EXHAUSTIVE_THRESHOLD`` (=8 non-signed versions), greedy
   nearest-neighbour otherwise.
5. Tie-breaking: version timestamps from ``VersionInput.timestamp`` (or
   ``Hints.timestamps``) may break equal-cost orderings but cannot override a
   strictly better content-derived order.

When no signed version is present the chain is still produced; the signed
anchor is ``None`` and the first version in the chain is simply the one most
distant from all others on average (likely the earliest template).

Hints (``hints.yaml`` / ``Hints`` dataclass):
  - ``Hints.order`` provides a weakly trusted explicit ordering; used only
    when content-derived distances are all equal (e.g. identical versions).
  - ``Hints.timestamps`` provides ISO-date strings for tie-breaking.

Output: ``VersionOrder`` — ordered IDs, signed ID, basis label, total
chain distance, and per-step pairwise distances.  Callers may persist this
to ``trail/<doc>.json`` via ``VersionOrder.to_dict()``.
"""

from __future__ import annotations

import difflib
import itertools
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

import yaml

from playbook_engine.clause_tree import ClauseTree
from playbook_engine.signed_detector import SignedStatus

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXHAUSTIVE_THRESHOLD: int = 8
"""Max non-signed versions for exhaustive permutation search."""

FORK_GAP_RATIO: float = 2.0
"""Bimodal-fork detector: cost jump is a 'fork' if the largest gap between
consecutive sorted pairwise costs is at least this multiple of the median cost."""

GAP_THRESHOLD: float = 0.4
"""Consecutive-step gap detector: a single step cost above this threshold
marks the chain shape as 'gap' (a missing intermediate draft)."""


# ---------------------------------------------------------------------------
# Input / output types
# ---------------------------------------------------------------------------


@dataclass
class VersionInput:
    """A single document version ready for ordering.

    Attributes:
        version_id:  Unique identifier for this version.
        tree:        The ``ClauseTree`` emitted by an ingester (+ segmenter).
        signed:      The signed-copy determination for this version.
        timestamp:   Optional ISO-8601 date string (e.g. ``"2025-01-15"``).
    """

    version_id: str
    tree: ClauseTree
    signed: SignedStatus
    timestamp: str | None = None


@dataclass(frozen=True)
class PairwiseDistance:
    """Edit distance between two adjacent versions in the chain."""

    from_id: str
    to_id: str
    distance: float


@runtime_checkable
class TrailJudge(Protocol):
    """Protocol for LLM-assisted version-chain arbitration.

    Called only when ``order_versions`` produces an uncertain chain:
    ``basis=="greedy"`` (large N, approximate ordering) or
    ``shape != "linear"`` (bimodal costs or a large single-step gap suggesting
    a missing intermediate draft or parallel redlines).

    The judge receives lightweight per-version summaries (heading + clause count
    + date hint if present), not full tree text, keeping payload < 500 tokens per
    chain.

    Contract:
    - Implementations MUST return a ``VersionOrder`` with ``basis="llm"``.
    - Implementations MUST NOT raise; on any error they should return the
      input ``VersionOrder`` unchanged (pass-through fallback).
    """

    def judge(
        self,
        version_summaries: list[str],
        pairwise_distances: dict[tuple[str, str], float],
        current_order: VersionOrder,
    ) -> VersionOrder:
        """Arbitrate an ambiguous version chain.

        Args:
            version_summaries:  One summary string per version (heading +
                                clause count + optional date hint).
            pairwise_distances: All pairwise edit distances (symmetric dict).
            current_order:      The deterministic result to refine or confirm.

        Returns:
            A ``VersionOrder``; pass ``current_order`` through on error.
        """
        ...


@dataclass(frozen=True)
class VersionOrder:
    """The inferred version ordering for one document.

    Attributes:
        ordered_ids:        Version IDs in inferred chronological order.
                            The signed version (if any) is always last.
        signed_id:          ID of the signed version, or ``None``.
        basis:              How the order was derived:
                            ``"single"`` — only one version present.
                            ``"exhaustive"`` — optimal by exhaustive search.
                            ``"greedy"`` — approximate by nearest-neighbour.
                            ``"hints"`` — content tie broken by hints.
                            ``"llm"`` — refined by a ``TrailJudge``.
        total_distance:     Sum of pairwise edit distances along the chain.
        pairwise_distances: Per-step distances (length = len(ordered_ids)−1).
        shape:              Chain shape derived from pairwise cost distribution:
                            ``"linear"`` — costs are roughly uniform.
                            ``"fork"`` — bimodal cost distribution (two clusters,
                                suggesting parallel redlines or a missing draft).
                            ``"gap"`` — a single large cost jump between adjacent
                                steps (missing intermediate version).
    """

    ordered_ids: tuple[str, ...]
    signed_id: str | None
    basis: str
    total_distance: float
    pairwise_distances: tuple[PairwiseDistance, ...]
    shape: str = "linear"

    def to_dict(self) -> dict[str, Any]:
        return {
            "ordered_versions": list(self.ordered_ids),
            "signed_version": self.signed_id,
            "basis": self.basis,
            "total_distance": round(self.total_distance, 6),
            "pairwise_distances": [
                {"from": p.from_id, "to": p.to_id, "distance": round(p.distance, 6)}
                for p in self.pairwise_distances
            ],
            "shape": self.shape,
        }


# ---------------------------------------------------------------------------
# Hints (optional seeding)
# ---------------------------------------------------------------------------


class HintsError(ValueError):
    """Raised by ``Hints.load`` when hints.yaml exists but is malformed.

    Deliberately distinct from a missing file (empty ``Hints`` — hints.yaml
    is optional, so absence is not an error).  hints.yaml is the documented
    remediation channel for correcting the engine's inferences (see
    ``cli.py``'s ``inspect`` command docstring); a YAML typo that silently
    discarded the correction (the previous ``return cls()`` on any
    exception) would defeat that whole human-in-the-loop story without any
    signal to the person who wrote the file.
    """


# Real document extensions the engine ingests. Only these are stripped from a
# hint value — a trailing dotted segment that is NOT one of these (a date like
# "6.14.23", a form number like "01.29.25") must be preserved, or Path.stem
# would eat it and the hint would never match the real version_id (issue #182).
_HINT_EXTENSIONS = frozenset({".docx", ".pdf", ".rtf"})


def _strip_hint_ext(value: str) -> str:
    """Strip a real document extension so a hint value matches a ``version_id``.

    ``version_id`` is always a file stem (``vf.stem`` — see
    ``pipeline.py``'s per-version loop), but docs/CORPUS-LAYOUT.md's
    documented ``hints.yaml`` example names ``order``/``signed_version``
    entries WITH extensions (e.g. ``fully-executed.pdf``).  Without this
    normalisation those hints never match a real version id and silently
    have no effect.

    Only a genuine ``.docx``/``.pdf``/``.rtf`` suffix is removed.
    ``Path.stem`` alone would strip a trailing dotted segment even when it is
    part of the name (``...6.14.23`` -> ``...6.14``), so a hint written by
    ``playbook stage`` for a dotted filename would double-strip and match
    nothing — the exact silent-no-op this function exists to prevent
    (issue #182).
    """
    p = Path(value)
    if p.suffix.lower() in _HINT_EXTENSIONS:
        return p.stem
    return value


@dataclass
class Hints:
    """Weakly trusted external ordering hints.

    Loaded from a ``hints.yaml`` file alongside the corpus.  Used only as
    a tie-breaker when content evidence is ambiguous, except for
    ``signed_version`` and ``provenance`` which are applied as hard overrides.

    Attributes:
        order:          Explicit version ordering; used only as a tie-breaker.
        timestamps:     ISO-date strings for tie-breaking; keyed by version ID.
        signed_version: When set, this version ID is unconditionally treated as
                        the signed copy, bypassing ``detect_signed`` heuristics.
        provenance:     When set (``"our_paper"`` or ``"counterparty_paper"``),
                        overrides the ``detect_provenance`` result for this doc.
    """

    order: list[str] | None = None
    timestamps: dict[str, str] = field(default_factory=dict)
    signed_version: str | None = None
    provenance: str | None = None

    @classmethod
    def load(cls, path: Path) -> Hints:
        """Load hints from a YAML file.

        Returns empty ``Hints`` when *path* does not exist — hints.yaml is
        optional, so a missing file is not an error.  When the file DOES
        exist but fails to parse (invalid YAML, a non-mapping document, or
        an ``order`` that isn't a list), raises ``HintsError`` instead of
        silently discarding the lawyer's corrections.

        ``order`` entries and ``signed_version`` are normalised by stripping
        any file extension (see ``_strip_hint_ext``) so hints written per
        docs/CORPUS-LAYOUT.md's example (extensions included) actually match
        the engine's file-stem version ids.
        """
        if not path.exists():
            return cls()
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise HintsError(f"{path}: not valid YAML: {exc}") from exc
        if data is None:
            data = {}
        if not isinstance(data, dict):
            raise HintsError(
                f"{path}: expected a YAML mapping at the top level, got {type(data).__name__}"
            )
        order = data.get("order")
        if order is not None:
            if not isinstance(order, list):
                raise HintsError(f"{path}: 'order' must be a list, got {type(order).__name__}")
            order = [_strip_hint_ext(v) for v in order]
        signed_version = data.get("signed_version") or None
        if signed_version is not None:
            signed_version = _strip_hint_ext(signed_version)
        return cls(
            order=order,
            timestamps=data.get("timestamps") or {},
            signed_version=signed_version,
            provenance=data.get("provenance") or None,
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def order_versions(
    versions: list[VersionInput],
    hints: Hints | None = None,
    *,
    trail_judge: TrailJudge | None = None,
) -> VersionOrder:
    """Infer the chronological ordering of *versions*.

    Args:
        versions:    List of ``VersionInput`` items (any order).
        hints:       Optional external seeding hints.  Never override strong
                     content evidence.
        trail_judge: Optional judge called when the chain is ambiguous
                     (``basis=="greedy"`` or ``shape != "linear"``).  When
                     provided the judge may return a refined ``VersionOrder``.

    Returns:
        A ``VersionOrder`` with ordered IDs, the signed version, and the
        basis/cost/shape metadata.
    """
    if not versions:
        return VersionOrder(
            ordered_ids=(),
            signed_id=None,
            basis="single",
            total_distance=0.0,
            pairwise_distances=(),
            shape="linear",
        )

    if len(versions) == 1:
        v = versions[0]
        return VersionOrder(
            ordered_ids=(v.version_id,),
            signed_id=v.version_id if v.signed.signed else None,
            basis="single",
            total_distance=0.0,
            pairwise_distances=(),
            shape="linear",
        )

    # Identify signed version (highest-confidence signed copy).
    signed_v = _pick_signed(versions)
    signed_id = signed_v.version_id if signed_v else None

    # B1 fix: build effective hints by merging VersionInput.timestamp values
    # with any explicitly supplied Hints (explicit takes precedence per-key).
    effective_hints = _merge_hints(versions, hints)

    # Compute fingerprints and pairwise distances.
    fingerprints = {v.version_id: _fingerprint(v.tree) for v in versions}
    dist = _all_distances(fingerprints)

    ids = [v.version_id for v in versions]
    non_signed_ids = [vid for vid in ids if vid != signed_id]

    if not non_signed_ids:
        # Only one version and it's signed — already handled above, but guard.
        return VersionOrder(
            ordered_ids=(signed_id,) if signed_id else (),
            signed_id=signed_id,
            basis="single",
            total_distance=0.0,
            pairwise_distances=(),
            shape="linear",
        )

    # Find the optimal chain.
    if len(non_signed_ids) <= EXHAUSTIVE_THRESHOLD:
        chain, basis = _exhaustive_chain(non_signed_ids, signed_id, dist, effective_hints)
    else:
        chain, basis = _greedy_chain(non_signed_ids, signed_id, dist, effective_hints)

    # Build output.
    total, pairs = _chain_cost(chain, dist)
    shape = chain_shape(chain, dist)
    result = VersionOrder(
        ordered_ids=tuple(chain),
        signed_id=signed_id,
        basis=basis,
        total_distance=total,
        pairwise_distances=tuple(pairs),
        shape=shape,
    )

    # Invoke TrailJudge when the chain is ambiguous.
    if trail_judge is not None and (basis == "greedy" or shape != "linear"):
        summaries = _version_summaries(versions, chain)
        result = trail_judge.judge(summaries, dist, result)

    return result


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def pairwise_distances(versions: list[VersionInput]) -> dict[tuple[str, str], float]:
    """Compute all pairwise content edit distances between *versions*.

    Public wrapper around the same fingerprint + edit-distance machinery
    ``order_versions`` uses internally (see ``_fingerprint``), exposed so
    callers that need to reason about chain ambiguity — e.g.
    ``induction_version_selector.select_representative_version``, which must
    tell a genuinely tied chain (no signed anchor, symmetric edit distance)
    apart from a real content-derived terminal — don't have to duplicate the
    edit-distance logic.
    """
    fingerprints = {v.version_id: _fingerprint(v.tree) for v in versions}
    return _all_distances(fingerprints)


def chain_shape(
    ordered_ids: list[str],
    pairwise_distances: dict[tuple[str, str], float],
) -> Literal["linear", "fork", "gap"]:
    """Classify the distribution of pairwise costs along *ordered_ids*.

    Rules (applied in order):
    1. ``"fork"`` — bimodal cost distribution: the largest gap between any two
       consecutive sorted costs is at least ``FORK_GAP_RATIO × median_cost``.
       Signals parallel redlines or a missing intermediate draft.
    2. ``"gap"`` — a single step cost exceeds ``GAP_THRESHOLD``.
       Signals a missing intermediate version between two adjacent steps.
    3. ``"linear"`` — costs are roughly uniform (no bimodality or large jump).

    Args:
        ordered_ids:        Ordered version IDs (the full chain including signed).
        pairwise_distances: All pairwise edit distances (symmetric dict from
                            ``_all_distances``).

    Returns:
        One of ``"linear"``, ``"fork"``, or ``"gap"``.
    """
    if len(ordered_ids) < 2:
        return "linear"

    step_costs = [
        pairwise_distances[(ordered_ids[i], ordered_ids[i + 1])]
        for i in range(len(ordered_ids) - 1)
    ]

    if len(step_costs) == 0:
        return "linear"

    # Check for a "gap": any single step cost exceeds GAP_THRESHOLD.
    if any(c > GAP_THRESHOLD for c in step_costs):
        return "gap"

    if len(step_costs) < 2:
        # Cannot detect bimodality with a single step.
        return "linear"

    # Check for "fork": bimodal distribution — largest inter-cost gap is large
    # relative to the median cost.
    sorted_costs = sorted(step_costs)
    median_cost = sorted_costs[len(sorted_costs) // 2]

    if median_cost < 1e-9:
        # All costs near zero — identical or nearly identical versions; linear.
        return "linear"

    inter_gaps = [sorted_costs[i + 1] - sorted_costs[i] for i in range(len(sorted_costs) - 1)]
    largest_gap = max(inter_gaps)

    if largest_gap >= FORK_GAP_RATIO * median_cost:
        return "fork"

    return "linear"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _version_summaries(versions: list[VersionInput], ordered_chain: list[str]) -> list[str]:
    """Build lightweight per-version summaries for the TrailJudge payload.

    Each summary is a single string: heading of the first clause node +
    clause count + optional date hint from the version's timestamp.  This
    keeps the payload well below 500 tokens per chain.

    Args:
        versions:      All ``VersionInput`` objects.
        ordered_chain: Chain ordering (from orderer); summaries follow this order.

    Returns:
        List of summary strings, one per version in ``ordered_chain`` order.
    """
    by_id = {v.version_id: v for v in versions}
    summaries: list[str] = []
    for vid in ordered_chain:
        v = by_id.get(vid)
        if v is None:
            summaries.append(f"{vid}: (unknown)")
            continue
        nodes = list(v.tree.all_nodes())
        clause_count = len(nodes)
        first_heading = ""
        for node in nodes:
            if node.heading:
                first_heading = node.heading.strip()
                break
        date_hint = f", date={v.timestamp}" if v.timestamp else ""
        summaries.append(f"{vid}: heading={first_heading!r}, clauses={clause_count}{date_hint}")
    return summaries


def _pick_signed(versions: list[VersionInput]) -> VersionInput | None:
    """Return the highest-confidence signed version, or None."""
    signed = [v for v in versions if v.signed.signed]
    if not signed:
        return None
    return max(signed, key=lambda v: v.signed.confidence)


def _fingerprint(tree: ClauseTree) -> list[str]:
    """Extract a format-independent text fingerprint from a ClauseTree.

    One element per node — heading (if present) plus a single
    whitespace-collapsed body-text element — rather than one element per
    *physical* line.  Ingesters disagree on physical line shape for
    identical content: a DOCX ingester yields a whole paragraph as one
    ``node.text`` line, while a PDF ingester (pdfplumber) wraps that same
    paragraph across several embedded physical lines, and RTF differs
    again.  Splitting on ``str.splitlines()`` (the previous approach) let
    edit distance track extraction artifacts instead of content: the exact
    signed PDF of a DOCX draft could come out looking maximally distant
    from that draft purely because the two formats wrap lines differently
    (#97) — mixed-format trails are explicitly supported (see
    docs/CORPUS-LAYOUT.md:22). ``provenance_detector.py``'s ``_fingerprint``
    documents the identical physics for the template-similarity signal.

    Collapsing all internal whitespace (including embedded newlines) to
    single spaces per node makes the fingerprint depend only on a node's
    words, not on how the source ingester wrapped them across physical
    lines, so the same content in two different formats/line-shapes
    produces the same fingerprint element.
    """
    lines: list[str] = []
    for node in tree.all_nodes():
        if node.heading:
            heading = " ".join(node.heading.split())
            if heading:
                lines.append(heading)
        if node.text:
            body = " ".join(node.text.split())
            if body:
                lines.append(body)
    return lines


def _edit_distance(a: list[str], b: list[str]) -> float:
    """Normalised edit distance in [0, 1]; 0 = identical, 1 = completely different."""
    if not a and not b:
        return 0.0
    ratio = difflib.SequenceMatcher(None, a, b).ratio()
    return 1.0 - ratio


def _all_distances(fingerprints: dict[str, list[str]]) -> dict[tuple[str, str], float]:
    """Compute all pairwise edit distances."""
    ids = list(fingerprints.keys())
    dist: dict[tuple[str, str], float] = {}
    for i, a in enumerate(ids):
        for b in ids[i + 1 :]:
            d = _edit_distance(fingerprints[a], fingerprints[b])
            dist[(a, b)] = d
            dist[(b, a)] = d
        dist[(a, a)] = 0.0
    return dist


def _chain_cost(
    chain: list[str], dist: dict[tuple[str, str], float]
) -> tuple[float, list[PairwiseDistance]]:
    """Return total cost and per-step distances for a chain."""
    total = 0.0
    pairs: list[PairwiseDistance] = []
    for i in range(len(chain) - 1):
        d = dist[(chain[i], chain[i + 1])]
        total += d
        pairs.append(PairwiseDistance(from_id=chain[i], to_id=chain[i + 1], distance=d))
    return total, pairs


def _exhaustive_chain(
    non_signed: list[str],
    signed_id: str | None,
    dist: dict[tuple[str, str], float],
    hints: Hints | None,
) -> tuple[list[str], str]:
    """Return (chain, basis) via exhaustive permutation search."""
    best_chain: list[str] | None = None
    best_cost = float("inf")
    hint_decided = False

    anchor = [signed_id] if signed_id else []

    for perm in itertools.permutations(non_signed):
        # NB4 fix: sort within equal-score permutations lexicographically so
        # output is deterministic regardless of caller input order.
        chain = list(perm) + anchor
        cost, _ = _chain_cost(chain, dist)
        if cost < best_cost - 1e-9:  # NB3: float-tolerance comparison
            best_cost = cost
            best_chain = chain
            hint_decided = False
        elif math.isclose(cost, best_cost, abs_tol=1e-9) and hints:
            # Tie-break by hint score; then lexicographic for full determinism.
            new_score = _hint_score(chain, hints)
            old_score = _hint_score(best_chain, hints) if best_chain else float("-inf")
            if new_score > old_score or (
                math.isclose(new_score, old_score) and chain < (best_chain or [])
            ):
                best_chain = chain
                hint_decided = new_score > old_score
        elif math.isclose(cost, best_cost, abs_tol=1e-9) and (not hints):
            # Lexicographic tie-break when no hints available.
            if not best_chain or chain < best_chain:
                best_chain = chain

    if not best_chain:
        best_chain = list(non_signed) + anchor

    # NB2 fix: label basis as "hints" whenever hints actually decided the order.
    basis = "hints" if hint_decided else "exhaustive"

    return best_chain, basis


def _greedy_chain(
    non_signed: list[str],
    signed_id: str | None,
    dist: dict[tuple[str, str], float],
    hints: Hints | None = None,
) -> tuple[list[str], str]:
    """Return (chain, basis) via greedy nearest-neighbour.

    Hint tie-breaking applies to the start-node selection and nearest-
    neighbour step so timestamps/order seeding still has effect for large
    version sets (NB1).

    NOTE: distance values are computed from _all_distances, so all expected
    keys are always present; direct indexing is used to fail loud on bugs.
    """
    # Start from the version farthest from the signed anchor (likely the template).
    if signed_id:
        start = max(non_signed, key=lambda v: dist[(v, signed_id)])
    else:
        # No signed anchor: start from the version most different from all others.
        avg_dist = {v: sum(dist[(v, u)] for u in non_signed if u != v) for v in non_signed}
        start = max(non_signed, key=lambda v: avg_dist[v])

    remaining = set(non_signed)
    remaining.discard(start)
    chain = [start]

    while remaining:
        current = chain[-1]
        nearest = min(remaining, key=lambda v: dist[(current, v)])
        chain.append(nearest)
        remaining.discard(nearest)

    if signed_id:
        chain.append(signed_id)

    return chain, "greedy"


def _merge_hints(versions: list[VersionInput], explicit: Hints | None) -> Hints | None:
    """B1 fix: merge VersionInput.timestamp values into effective Hints.

    Returns None if neither source provides any timestamps/order data.
    Explicit Hints take precedence over VersionInput.timestamp per key.
    """
    auto_ts = {v.version_id: v.timestamp for v in versions if v.timestamp}
    if not auto_ts and explicit is None:
        return None
    if explicit is None:
        return Hints(timestamps=auto_ts)
    # Explicit takes precedence: start from auto, overwrite with explicit.
    merged_ts = {**auto_ts, **explicit.timestamps}
    return Hints(order=explicit.order, timestamps=merged_ts)


def _hint_score(chain: list[str], hints: Hints) -> float:
    """Score a chain against available hints (higher = better agreement)."""
    score = 0.0
    # Timestamp agreement: earlier timestamps should appear earlier.
    if hints.timestamps:
        for i in range(len(chain) - 1):
            ts_a = hints.timestamps.get(chain[i])
            ts_b = hints.timestamps.get(chain[i + 1])
            if ts_a and ts_b:
                score += 1.0 if ts_a <= ts_b else -1.0
    # Explicit order agreement: count how many adjacent pairs respect the hint.
    if hints.order:
        pos = {vid: idx for idx, vid in enumerate(hints.order)}
        for i in range(len(chain) - 1):
            p_a = pos.get(chain[i])
            p_b = pos.get(chain[i + 1])
            if p_a is not None and p_b is not None:
                score += 1.0 if p_a < p_b else -1.0
    return score
