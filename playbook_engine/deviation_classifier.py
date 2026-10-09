"""Standard check — the deterministic answer to "is this OUR language?".

Every clause gets one deterministic question, never a judged one: is its text
OUR standard (:func:`is_standard_text` — after normalization, does its text
equal the template clause for its taxonomy_id)? :func:`assess_deviations_deterministic`
answers it per ``ClauseDiff`` row: ``deviation="none"`` when the text is our
standard, ``"substantive"`` otherwise, always ``basis="deterministic"``. There
is no deviation judge, no risk assessment and nothing to queue: the consumer (a
capable review model) does the judging; the playbook supplies precedent
(owner decision 2026-09-13 (c)).

``DeviationResult.deviation`` values:
  ``"none"``        — the clause text is our standard.
  ``"substantive"`` — it is not (or there is no template clause for its
                      taxonomy_id to be standard against).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from playbook_engine.clause_differ import ClauseDiff

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEVIATION_VALUES = frozenset({"none", "substantive"})
_BASIS_VALUES = frozenset({"deterministic"})

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
# Result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DeviationResult:
    """The standard-check outcome for one clause row.

    Attributes:
        deviation:   ``"none"`` (our standard) or ``"substantive"``.
        basis:       Always ``"deterministic"``.
        rationale:   Brief explanation of the check's outcome.
    """

    deviation: str
    basis: str = "deterministic"
    rationale: str = ""

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

    def to_dict(self) -> dict[str, Any]:
        return {
            "deviation": self.deviation,
            "basis": self.basis,
            "rationale": self.rationale,
        }


# ---------------------------------------------------------------------------
# Normalization and the standard check
# ---------------------------------------------------------------------------


def _normalize_for_containment(text: str) -> str:
    """Lowercase, strip punctuation to single spaces and collapse whitespace.

    The comparison shape for the standard check (:func:`normalize_for_standard`)
    and for the survival test in ``observation_builder``. Deliberately the
    same shape as ``clause_aligner._normalize``.
    """
    s = text.lower()
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


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
    """The ``DeviationResult`` for one standard-check outcome (issue #220):
    ``"none"`` for standard text, ``"substantive"`` otherwise, always
    ``basis="deterministic"``."""
    return DeviationResult(
        deviation="none" if standard else "substantive",
        basis="deterministic",
        rationale=_STANDARD_RATIONALE if standard else _NON_STANDARD_RATIONALE,
    )


def assess_deviations_deterministic(
    clause_diffs: list[ClauseDiff],
    our_standard: str | Sequence[str],
    party_names: Sequence[str] = (),
) -> list[tuple[ClauseDiff, DeviationResult]]:
    """Deviation assessment — no judge, ever (issue #220).

    Each row's own text (the after side; the before side for a removed row)
    is checked against *our_standard* with :func:`is_standard_text`, and the
    row gets :func:`standard_check_result` for the answer. It never compares
    the net first-to-last hunk and never consults the opening draft:
    identical signed text gets the identical answer whatever draft it was
    reached from.

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
