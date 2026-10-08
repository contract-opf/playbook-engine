"""Deviation classifier — L4 pipeline stage (LLM).

Assesses how a changed clause deviates from our standard position and in
which direction risk moves.

Fast path (deterministic):
  - Unchanged clauses (``kind="unchanged"``) → ``deviation="none"``,
    ``risk_delta=neutral/none``, ``basis="deterministic"`` — but ONLY when the
    clause text matches (or Jaccard-nears) ``our_standard``, the canonical
    template clause for this taxonomy_id. When ``our_standard`` is non-empty
    and differs beyond ``REWORDED_EQUIVALENT_THRESHOLD``, the clause is routed
    to the judge instead: a clause that never changed during negotiation was
    never actually compared to the template before (issue #103) — "unchanged"
    describes the negotiation trail, not agreement with our standard. An empty
    ``our_standard`` (no template, or no template clause for this taxonomy_id)
    means there is nothing to compare against, so the clause stays
    deterministic, same as before.
  - Near-identical rewrites (Jaccard ≥ ``REWORDED_EQUIVALENT_THRESHOLD``) →
    ``deviation="none"``, ``basis="reworded_equivalent"``, no judge call.
  - Added/removed clauses whose normalized text occurs verbatim somewhere in
    the counterpart version's clause tree (issue #167) → ``deviation="none"``,
    ``basis="alignment"``, no judge call. This is the same containment check
    a human relocation-triage reviewer performs by hand (see
    ``.claude/skills/playbook-from-corpus/REFERENCE.md``'s "Relocation triage
    FIRST" bullet): the clause_aligner's global move-matching phase
    (``clause_aligner._match_moves``) already pairs most relocated clauses
    back into a single ``unchanged``/``modified`` row, but a residual set
    still surfaces as an added/removed hunk (short clauses below the
    move-matcher's length/token gates, or one side matched into a *different*
    slot by the positional bucket path) whose text nonetheless still appears,
    unchanged, in the other version. Only engaged when the caller supplies
    ``counterpart_clause_texts``; callers that omit it (or pre-#167 callers)
    keep routing every added/removed clause to the judge, unchanged.

Consumer path (issue #220 — the default; no deviation judge at all):
  - :func:`assess_deviations_deterministic` answers one deterministic
    question per clause instead of a judged one: is its text OUR standard
    (:func:`is_standard_text` — after normalization, does its text equal the
    template clause for its taxonomy_id)? ``deviation="none"`` when it is, ``"substantive"``
    otherwise, always ``basis="deterministic"`` with a neutral/none
    ``risk_delta`` placeholder (the observation store's shape; it is not a
    risk assessment). Never ``needs_review``, never a judge call. The
    consumer (a capable review model) does the judging; the playbook supplies
    precedent (owner decision 2026-09-13 (c)).

Slow path (opt-in advisory layer — injected ``DeviationJudge``, only under
``--with-deviation-judge``):
  - Changed clauses (added/removed/modified) that do not pass a deterministic
    fast path above are batched and passed to the judge.  The judge receives
    a compact hunk payload (not the full clause text) and the
    ``our_standard`` text for context.

``DeviationResult.deviation`` values:
  ``"none"``                — clause unchanged from our standard.
  ``"reworded_equivalent"`` — phrasing differs, substantive effect does not.
  ``"substantive"``         — material change in rights, obligations, or risk.
  ``"needs_review"``        — judge raised; clause is quarantined for human review
                              rather than silently recorded as benign (``"none"``).

``DeviationResult.risk_delta.direction`` values:
  ``"better"``  — more favourable than our standard.
  ``"neutral"`` — equivalent risk.
  ``"worse"``   — less favourable than our standard.

``DeviationResult.risk_delta.magnitude`` values:
  ``"none"``     — no risk shift (direction must be "neutral").
  ``"minor"``    — small risk shift.
  ``"material"`` — significant risk shift.

``DeviationResult.basis`` values:
  ``"deterministic"``       — decided without LLM (unchanged clause, or the
                              issue #220 standard check on the consumer path).
  ``"reworded_equivalent"`` — Jaccard pre-filter; no judge call needed.
  ``"alignment"``           — added/removed clause whose text also occurs in
                              the counterpart version's clause tree (issue
                              #167); an alignment/relocation artifact, not a
                              negotiated change. No judge call needed.
  ``"judge"``               — decided by injected ``DeviationJudge``.
  ``"judge_error"``         — judge raised; clause assessed as unknown deviation.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from playbook_engine.clause_aligner import MOVE_EXACT_MIN_CHARS
from playbook_engine.clause_differ import ClauseDiff

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEVIATION_VALUES = frozenset({"none", "reworded_equivalent", "substantive", "needs_review"})
_DIRECTION_VALUES = frozenset({"better", "neutral", "worse"})
_MAGNITUDE_VALUES = frozenset({"none", "minor", "material"})
_BASIS_VALUES = frozenset(
    {"deterministic", "reworded_equivalent", "alignment", "judge", "judge_error", "needs_review"}
)

# Jaccard similarity threshold above which a modified clause is treated as a
# near-identical reword and classified without calling the judge.
REWORDED_EQUIVALENT_THRESHOLD: float = 0.92

# The deterministic standard check (issue #220) is an EXACT match after
# ``normalize_for_standard`` — never a similarity score. A token-set Jaccard
# is order-blind and absorbs a one-token swap in any clause over ~25 tokens,
# so at the old 0.92 bar "Neither party may assign" -> "Either party may
# assign", a deleted carve-out, or "remain protected" -> "are not protected"
# all scored as our standard and reached the consumer as signed standard
# language. Normalization already absorbs everything "near-equal" is meant to
# (rewrapping, case, punctuation, party names), so
# the check needs no tolerance on top of it.

# The neutral token every known party name is rewritten to before the
# standard comparison, so "AlphaCorp" in a deal and "AlphaCorp Holdings,
# Inc." (or a counterparty's own name) in the template never decide it.
_PARTY_TOKEN = "party"

# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RiskDelta:
    """Direction and magnitude of risk shift relative to our standard."""

    direction: str  # "better", "neutral", "worse"
    magnitude: str  # "none", "minor", "material"

    def __post_init__(self) -> None:
        if self.direction not in _DIRECTION_VALUES:
            raise ValueError(
                f"RiskDelta.direction must be one of {sorted(_DIRECTION_VALUES)!r}; "
                f"got {self.direction!r}"
            )
        if self.magnitude not in _MAGNITUDE_VALUES:
            raise ValueError(
                f"RiskDelta.magnitude must be one of {sorted(_MAGNITUDE_VALUES)!r}; "
                f"got {self.magnitude!r}"
            )
        if self.direction == "neutral" and self.magnitude != "none":
            raise ValueError(
                "RiskDelta with direction='neutral' must have magnitude='none'; "
                f"got magnitude={self.magnitude!r}"
            )

    def to_dict(self) -> dict[str, str]:
        return {"direction": self.direction, "magnitude": self.magnitude}


@dataclass(frozen=True)
class DeviationResult:
    """Assessment of how a changed clause deviates from our standard.

    Attributes:
        deviation:   Degree of change from our standard.
        risk_delta:  Direction and magnitude of risk shift.
        basis:       How the assessment was reached.
        rationale:   Brief natural-language explanation (empty for
                     deterministic results).
        confidence:  Judge confidence in [0.0, 1.0], or ``None`` for
                     deterministic paths (Jaccard pre-filter, unchanged
                     clauses, and judge errors).
    """

    deviation: str
    risk_delta: RiskDelta
    basis: str
    rationale: str = ""
    confidence: float | None = None

    def __post_init__(self) -> None:
        if self.deviation not in _DEVIATION_VALUES:
            raise ValueError(
                f"DeviationResult.deviation must be one of "
                f"{sorted(_DEVIATION_VALUES)!r}; got {self.deviation!r}"
            )
        if self.basis not in _BASIS_VALUES:
            raise ValueError(
                f"DeviationResult.basis must be one of {sorted(_BASIS_VALUES)!r}; "
                f"got {self.basis!r}"
            )
        if self.confidence is not None:
            if isinstance(self.confidence, bool) or not isinstance(self.confidence, (int, float)):
                raise ValueError(
                    "DeviationResult.confidence must be a number in [0, 1] or None; "
                    f"got {self.confidence!r}"
                )
            if not 0.0 <= self.confidence <= 1.0:
                raise ValueError(
                    f"DeviationResult.confidence must be in [0, 1] or None; got {self.confidence!r}"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "deviation": self.deviation,
            "risk_delta": self.risk_delta.to_dict(),
            "basis": self.basis,
            "rationale": self.rationale,
            "confidence": self.confidence,
        }


# ---------------------------------------------------------------------------
# Judge protocol (LLM integration point)
# ---------------------------------------------------------------------------

_NEUTRAL_ZERO = RiskDelta(direction="neutral", magnitude="none")


@runtime_checkable
class DeviationJudge(Protocol):
    """Protocol for LLM-based deviation and risk-delta assessment.

    The judge receives batches of changed clauses as compact hunk payloads
    (not the full clause text) and the ``our_standard`` reference text, and
    returns one ``DeviationResult`` per item **in the same order**.

    Contract:
    - Return exactly ``len(items)`` results.
    - Each result must have ``basis="judge"``.
    - ``deviation`` and ``risk_delta`` must use the defined vocabulary.
    """

    def assess_batch(
        self,
        items: list[dict[str, str]],  # [{"hunk": "...", ...}, ...]
        our_standard: str,
    ) -> list[DeviationResult]:
        """Assess deviation for each hunk payload against *our_standard*.

        Args:
            items:        List of hunk payload dicts.  Each dict contains a
                         ``"hunk"`` key with a compact
                         ``[BEFORE]\\n<text>\\n[AFTER]\\n<text>`` diff
                         representation.  For added clauses, the
                         ``[BEFORE]`` section is empty; for removed
                         clauses the ``[AFTER]`` section is empty.  Each dict
                         also carries ``"taxonomy_id"`` and ``"clause_path"``
                         (and ``"document_id"`` when the caller supplied one)
                         so a human or downstream store can trace the item
                         back to the actual clause it describes (issue #109)
                         — these are traceability context, not judgment
                         content, and must NOT be folded into any content-hash
                         cache key derived from this payload (that would break
                         cross-document dedup of identical hunk/standard
                         pairs).
            our_standard: The canonical text from our standard playbook for
                         this clause type.

        Returns:
            One ``DeviationResult(basis='judge')`` per item, same order.
        """
        ...  # pragma: no cover


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _text_jaccard(a: str, b: str) -> float:
    """Compute token-level Jaccard similarity between two strings.

    Tokenises by splitting on whitespace and punctuation (non-word chars).
    Returns 1.0 if both strings are empty; 0.0 if only one is empty.
    """

    def _tokens(text: str) -> frozenset[str]:
        return frozenset(t.lower() for t in re.split(r"\W+", text) if t)

    tokens_a = _tokens(a)
    tokens_b = _tokens(b)
    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    return intersection / union


def _normalize_for_containment(text: str) -> str:
    """Normalize text for the alignment-artifact containment check (issue #167).

    Lowercase, strip punctuation/numbering to single spaces, collapse
    whitespace. Deliberately the same shape as ``clause_aligner._normalize``
    so a clause whose only difference from its counterpart is a renumbered
    heading prefix ("2.6 Notices..." vs "Notices...") or trailing punctuation
    still matches as a substring.
    """
    s = text.lower()
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _is_alignment_artifact(clause_text: str, normalized_counterpart_texts: frozenset[str]) -> bool:
    """True if *clause_text*, normalized, occurs verbatim in any counterpart clause.

    ``normalized_counterpart_texts`` must already be normalized (via
    ``_normalize_for_containment``) — callers normalize the counterpart set
    once per ``assess_deviations`` call rather than per candidate clause.

    Deliberately a plain substring test (containment), not equality: an
    added/removed clause frequently reappears as a sub-span of a merged or
    renumbered clause in the counterpart version, not as a standalone
    identical clause. Empty/whitespace-only text never matches — it would be
    a substring of everything, producing false suppressions.

    Also requires the normalized text to be at least ``MOVE_EXACT_MIN_CHARS``
    long, mirroring the length floor ``clause_aligner``'s own near-exact move
    matcher applies to its containment-shaped comparison. Below that floor a
    short, generic clause (e.g. a one-line addition) can coincidentally be a
    substring of an unrelated, longer counterpart clause; without the floor
    that coincidence would silently suppress a genuine negotiated change.
    """
    norm = _normalize_for_containment(clause_text)
    if len(norm) < MOVE_EXACT_MIN_CHARS:
        return False
    return any(norm in counterpart for counterpart in normalized_counterpart_texts)


def normalize_for_standard(text: str, party_names: Sequence[str] = ()) -> str:
    """Normalize *text* for the deterministic standard check (issue #220).

    Known party names (``party_names`` — the configured
    ``provenance.our_party_aliases`` plus ``provenance.known_entities``, the
    same names the entity registry aliases) are rewritten to one neutral
    token first, longest name first and case-insensitively on word
    boundaries, so the parties' own names never decide whether a clause is
    our standard. Then whitespace is collapsed exactly the way
    ``version_orderer``'s fingerprint does it
    (:func:`~playbook_engine.version_orderer.collapse_whitespace`), and case
    and punctuation are dropped (``_normalize_for_containment``). Word order
    and every content token — negators, modals, numerals — survive, so the
    exact comparison in :func:`is_standard_text` stays sensitive to them.
    Clause numbering is deliberately NOT stripped: the segmenter keeps it out
    of node text, and any prefix stripper also eats leading content numbers
    ("30 days" vs "60 days", "1.5 times" vs "2.5 times").
    """
    # Imported here, not at module top: version_orderer -> provenance_detector
    # -> config -> clause_position_compiler -> observation_builder imports
    # this module, so a top-level import would be circular.
    from playbook_engine.version_orderer import collapse_whitespace  # noqa: PLC0415

    s = collapse_whitespace(text)
    for name in sorted({collapse_whitespace(n) for n in party_names if n.strip()}, key=len)[::-1]:
        s = re.sub(rf"(?<!\w){re.escape(name)}(?!\w)", _PARTY_TOKEN, s, flags=re.IGNORECASE)
    return _normalize_for_containment(s)


def is_standard_text(
    text: str, standard: str | Sequence[str], party_names: Sequence[str] = ()
) -> bool:
    """Whether *text* is OUR standard language for its clause (issue #220).

    *standard* is the template text for the clause's taxonomy_id: a string is
    compared as one whole clause; a sequence is the template's nodes for that
    taxonomy_id, in document order, and *text* matches the whole (the nodes
    joined) OR any single node — the node-granular comparison a single
    clause-tree row needs when the template splits one clause across several
    nodes. A match is an exact match after ``normalize_for_standard`` — no
    similarity tolerance (see the comment above ``_PARTY_TOKEN`` for why a
    token-set score is unsafe here). Empty text,
    or no standard text at all (no template clause for this taxonomy_id), is
    never a match: there is nothing to be standard against.
    """
    nodes = [standard] if isinstance(standard, str) else list(standard)
    nodes = [node for node in nodes if node.strip()]
    if not text.strip() or not nodes:
        return False
    candidates = ["\n".join(nodes)] + (nodes if len(nodes) > 1 else [])
    norm_text = normalize_for_standard(text, party_names)
    for cand in candidates:
        norm_cand = normalize_for_standard(cand, party_names)
        if norm_text and norm_text == norm_cand:
            return True
    return False


_STANDARD_RATIONALE = "deterministic standard check: text matches our template clause"
_NON_STANDARD_RATIONALE = (
    "deterministic standard check: text does not match our template clause "
    "(or there is no template clause for this taxonomy_id)"
)


def standard_check_result(standard: bool) -> DeviationResult:
    """The consumer-path ``DeviationResult`` for one standard-check outcome
    (issue #220): ``"none"`` for standard text, ``"substantive"`` otherwise,
    always ``basis="deterministic"`` with the neutral/none placeholder
    ``risk_delta`` the observation store carries. Never ``needs_review``."""
    return DeviationResult(
        deviation="none" if standard else "substantive",
        risk_delta=_NEUTRAL_ZERO,
        basis="deterministic",
        rationale=_STANDARD_RATIONALE if standard else _NON_STANDARD_RATIONALE,
    )


def assess_deviations_deterministic(
    clause_diffs: list[ClauseDiff],
    our_standard: str | Sequence[str],
    party_names: Sequence[str] = (),
) -> list[tuple[ClauseDiff, DeviationResult]]:
    """Consumer-path deviation assessment — no judge, ever (issue #220).

    Each row's own text (the after side; the before side for a removed row)
    is checked against *our_standard* with :func:`is_standard_text`, and the
    row gets :func:`standard_check_result` for the answer. Unlike
    :func:`assess_deviations` this never compares the net first-to-last hunk
    and never consults the opening draft: identical signed text gets the
    identical answer whatever draft it was reached from.

    Returns one ``(ClauseDiff, DeviationResult)`` pair per input diff, same
    order.
    """
    return [
        (
            cd,
            standard_check_result(
                is_standard_text(
                    cd.text_before if cd.kind == "removed" else (cd.text_after or cd.text_before),
                    our_standard,
                    party_names,
                )
            ),
        )
        for cd in clause_diffs
    ]


_ALIGNMENT_ARTIFACT_RATIONALE = (
    "alignment artifact — this clause's text is present in two or more "
    "versions of the same document, so the aligner paired it differently "
    "between versions rather than the clause being added or removed. "
    "No negotiated change."
)

_HUNK_CONTEXT_LINES = 3


def _build_hunk(before_text: str, after_text: str) -> str:
    """Build a compact ``[BEFORE]\\n<text>\\n[AFTER]\\n<text>`` hunk.

    Includes at most ``_HUNK_CONTEXT_LINES`` leading and trailing lines from
    each section to keep the payload small.  Does not require a diff library.
    """

    def _trim(text: str) -> str:
        lines = text.splitlines()
        if len(lines) <= _HUNK_CONTEXT_LINES * 2:
            return text
        kept = lines[:_HUNK_CONTEXT_LINES] + ["..."] + lines[-_HUNK_CONTEXT_LINES:]
        return "\n".join(kept)

    return f"[BEFORE]\n{_trim(before_text)}\n[AFTER]\n{_trim(after_text)}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def assess_deviations(
    clause_diffs: list[ClauseDiff],
    our_standard: str,
    judge: DeviationJudge,
    document_id: str | None = None,
    counterpart_clause_texts: tuple[frozenset[str], frozenset[str]] | None = None,
) -> list[tuple[ClauseDiff, DeviationResult]]:
    """Assess deviation for each changed clause in *clause_diffs*.

    Unchanged clauses are handled deterministically.  Changed clauses
    (added, removed, modified) are batched and sent to the judge.  If the
    judge raises, all batch items receive ``basis="judge_error"``.

    Args:
        clause_diffs:  Diffs from the diff engine (any mix of kinds).
        our_standard:  Canonical clause text from our playbook standard.
                      May be empty for newly observed clause types.
        judge:         Injected ``DeviationJudge`` for semantic assessment.
        document_id:   Caller's document identifier, threaded onto each judge
                      batch item as ``"document_id"`` for traceability
                      (issue #109). Omitted from batch items entirely when
                      ``None`` (single-document callers/tests that have no
                      document context to give).
        counterpart_clause_texts: Optional ``(before_texts, after_texts)`` —
                      the full, un-normalized clause text of every clause in
                      the diff's "before" version and "after" version,
                      respectively (issue #167). When supplied, an
                      added/removed clause whose normalized text occurs in
                      the *other* side's set is classified deterministically
                      as an alignment artifact (``basis="alignment"``)
                      instead of being sent to the judge — see module
                      docstring. ``None`` (the default) disables this fast
                      path entirely; every added/removed clause is judged,
                      same as before issue #167.

    Each batch item also carries ``"version_from"``/``"version_to"``
    (``ClauseDiff.clause_version_before``/``clause_version_after``, issue
    #166) — traceability context only, same as ``document_id`` above.

    Returns:
        One ``(ClauseDiff, DeviationResult)`` pair per input diff, same order.

    Raises:
        ValueError: if the judge returns wrong-length results or invalid
                    ``basis`` / vocabulary values.
    """
    results: list[tuple[ClauseDiff, DeviationResult] | None] = [None] * len(clause_diffs)
    before_texts, after_texts = counterpart_clause_texts or (frozenset(), frozenset())
    norm_before_texts = frozenset(_normalize_for_containment(t) for t in before_texts)
    norm_after_texts = frozenset(_normalize_for_containment(t) for t in after_texts)

    # Fast path 1 (deterministic): unchanged clauses need no LLM call ONLY if
    # they actually match our_standard (the template clause for this
    # taxonomy_id) — see module docstring. against_template tracks which
    # judge_indices entries need a template-vs-clause hunk below rather than
    # the (identical, hence useless) before/after hunk.
    # Fast path 2 (Jaccard): near-identical rewrites are classified without the judge.
    # Fast path 3 (alignment, issue #167): added/removed clauses whose text
    # also occurs in the counterpart version are alignment artifacts, not
    # negotiated changes. Only engaged when the caller supplied
    # counterpart_clause_texts.
    judge_indices: list[int] = []
    against_template: set[int] = set()
    for i, cd in enumerate(clause_diffs):
        if cd.kind == "unchanged":
            clause_text = cd.text_after or cd.text_before
            if not our_standard or _text_jaccard(clause_text, our_standard) >= (
                REWORDED_EQUIVALENT_THRESHOLD
            ):
                results[i] = (
                    cd,
                    DeviationResult(
                        deviation="none",
                        risk_delta=_NEUTRAL_ZERO,
                        basis="deterministic",
                    ),
                )
            else:
                # Unchanged across the negotiation trail (or the only version
                # of a single-version document), but differs from our
                # canonical template — never actually checked against the
                # standard until now (issue #103). Route through the judge
                # instead of silently recording "matches our standard".
                judge_indices.append(i)
                against_template.add(i)
        elif (
            cd.text_before
            and cd.text_after
            and _text_jaccard(cd.text_before, cd.text_after) >= REWORDED_EQUIVALENT_THRESHOLD
        ):
            # Near-identical reword: skip the judge; confidence is None (deterministic path).
            results[i] = (
                cd,
                DeviationResult(
                    deviation="none",
                    risk_delta=_NEUTRAL_ZERO,
                    basis="reworded_equivalent",
                ),
            )
        elif (
            cd.kind == "removed" and _is_alignment_artifact(cd.text_before, norm_after_texts)
        ) or (cd.kind == "added" and _is_alignment_artifact(cd.text_after, norm_before_texts)):
            results[i] = (
                cd,
                DeviationResult(
                    deviation="none",
                    risk_delta=_NEUTRAL_ZERO,
                    basis="alignment",
                    rationale=_ALIGNMENT_ARTIFACT_RATIONALE,
                ),
            )
        else:
            judge_indices.append(i)

    if not judge_indices:
        return [r for r in results if r is not None]

    # Slow path: batch changed clauses to judge using compact hunk payloads.
    # For indices in against_template (unchanged-vs-negotiation but differs
    # from our_standard), text_before == text_after == the clause text — a
    # hunk built from those would show a no-op diff. Build the hunk against
    # our_standard instead so the judge actually sees what changed.
    batch_items = [
        {
            "hunk": (
                _build_hunk(our_standard, clause_diffs[i].text_after or clause_diffs[i].text_before)
                if i in against_template
                else _build_hunk(clause_diffs[i].text_before, clause_diffs[i].text_after)
            ),
            # Traceability context (issue #109) — NOT judgment content. A
            # store-backed judge must hash only "hunk" + our_standard for its
            # cache key so identical hunk/standard pairs from different
            # clauses/documents still dedup; these keys exist so a human (or
            # judge-apply tooling) reviewing a pending verdict can see which
            # actual clause the bare BEFORE/AFTER text pair came from.
            "taxonomy_id": clause_diffs[i].taxonomy_id or "",
            "clause_path": clause_diffs[i].clause_path_after
            or clause_diffs[i].clause_path_before
            or "",
            # version_from/version_to (issue #166) — same traceability-only
            # pattern: the normalized-tree version ids (ClauseDiff.
            # clause_version_before/after) a relocation-triage reviewer needs
            # to open the right $OUT/normalized/<document_id>/*.clauses.json
            # files. Never part of the content hash (see above).
            "version_from": clause_diffs[i].clause_version_before or "",
            "version_to": clause_diffs[i].clause_version_after or "",
            **({"document_id": document_id} if document_id else {}),
        }
        for i in judge_indices
    ]

    try:
        judge_results = judge.assess_batch(batch_items, our_standard)
    except Exception:  # noqa: BLE001
        # Use deviation="needs_review" rather than "none" so a judge failure on a
        # changed clause is visibly quarantined — not silently recorded as benign.
        # Callers must inspect basis="judge_error" to route for human review (§P1.5).
        judge_results = [
            DeviationResult(
                deviation="needs_review",
                risk_delta=_NEUTRAL_ZERO,
                basis="judge_error",
                rationale="Deviation judge raised an unexpected error; clause flagged for review.",
            )
        ] * len(batch_items)

    if len(judge_results) != len(batch_items):
        raise ValueError(
            f"DeviationJudge.assess_batch() returned {len(judge_results)} results "
            f"for {len(batch_items)} items."
        )

    for idx, dr in zip(judge_indices, judge_results, strict=True):
        if dr.basis not in ("judge", "judge_error", "needs_review"):
            raise ValueError(
                f"DeviationJudge must return basis='judge'; "
                f"got {dr.basis!r} for clause {clause_diffs[idx].clause_path_after!r}."
            )
        results[idx] = (clause_diffs[idx], dr)

    return [r for r in results if r is not None]
