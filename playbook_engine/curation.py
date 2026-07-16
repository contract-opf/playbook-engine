"""Embedded attorney-pinned positions (curation overlay) — issue #147.

Makes the viewer feedback loop real: an attorney's pinned position on a
clause is embedded directly in the OPF (not a sidecar file), wins over the
recomputed ``summary.historical_stance`` for display purposes, survives a
recompile, and is checked for contradiction against fresh evidence on every
recompile.

Design
------
A pin records BOTH the attorney's asserted ``position`` AND the
``baseline_stance`` — the engine's own ``historical_stance`` for that clause
at the moment the pin was made (i.e. what the attorney was overriding
*from*). This is deliberate: a pin's whole purpose is usually to assert a
position that already differs from the corpus rollup, so comparing the
*recomputed* stance against ``position`` would flag a "conflict" on every
single recompile, even when nothing about the underlying evidence changed.
Comparing the recomputed stance against ``baseline_stance`` instead answers
the right question — "has the evidence-driven signal moved since this pin
was made" — which is what "new evidence contradicts a pinned position"
means in the issue.

``merge_curation()`` is the pipeline's recompile merge layer (called from
``playbook_assembler.assemble_playbook``): given the curation section of the
*prior* compile and the freshly recomputed ``historical_stance`` per clause,
it preserves every pin, and flags/clears ``conflict`` deterministically:

- ``recomputed_stance == baseline_stance``   → evidence unchanged; conflict
  cleared (or was never set). The pin survives quietly.
- ``recomputed_stance != baseline_stance``   → evidence moved since the pin
  was made; conflict flagged with the new recomputed stance recorded.
- clause no longer present in this compile  → pin preserved as-is (nothing
  to check against).

This is a simple deterministic contradiction check (no judge/LLM call) — see
issue #147's Out of scope note; a live-judgment version is future work.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["CurationPin", "merge_curation"]


@dataclass(frozen=True)
class CurationPin:
    """One attorney-pinned clause position, embedded in ``curation.pins``."""

    clause_id: str
    item_id: str
    position: str
    baseline_stance: str
    pinned_at: str
    pinned_by: str | None = None
    comment: str | None = None
    conflict: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "clause_id": self.clause_id,
            "item_id": self.item_id,
            "position": self.position,
            "baseline_stance": self.baseline_stance,
            "pinned_at": self.pinned_at,
        }
        if self.pinned_by is not None:
            d["pinned_by"] = self.pinned_by
        if self.comment is not None:
            d["comment"] = self.comment
        if self.conflict is not None:
            d["conflict"] = self.conflict
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CurationPin:
        return cls(
            clause_id=d["clause_id"],
            item_id=d.get("item_id", ""),
            position=d["position"],
            baseline_stance=d.get("baseline_stance", "unknown"),
            pinned_at=d.get("pinned_at", ""),
            pinned_by=d.get("pinned_by"),
            comment=d.get("comment"),
            conflict=d.get("conflict"),
        )

    def with_conflict(self, conflict: dict[str, Any] | None) -> CurationPin:
        """Return a copy of this pin with ``conflict`` replaced."""
        return CurationPin(
            clause_id=self.clause_id,
            item_id=self.item_id,
            position=self.position,
            baseline_stance=self.baseline_stance,
            pinned_at=self.pinned_at,
            pinned_by=self.pinned_by,
            comment=self.comment,
            conflict=conflict,
        )


def merge_curation(
    existing_curation: dict[str, Any] | None,
    clause_stances: dict[str, str],
    checked_at: str,
) -> dict[str, Any]:
    """Merge a prior compile's ``curation`` section over freshly recomputed stances.

    Args:
        existing_curation: The prior compile's ``playbook["curation"]`` dict
                          (``{"pins": [...]}``), or ``None``/``{}`` if this is
                          a first compile / no pins exist yet.
        clause_stances:   ``{clause_id: historical_stance}`` for every clause
                          in the FRESHLY assembled playbook (this compile).
        checked_at:       ISO-8601 timestamp to stamp on any conflict raised
                          or cleared this run (the caller's ``generated_at``).

    Returns:
        ``{"pins": [...]}`` with every pin preserved and its ``conflict``
        field updated per the module docstring's deterministic check, or
        ``{}`` if there were no pins to carry forward.
    """
    if not existing_curation:
        return {}
    raw_pins = existing_curation.get("pins") or []
    if not raw_pins:
        return {}

    merged: list[dict[str, Any]] = []
    for raw in raw_pins:
        pin = CurationPin.from_dict(raw)
        recomputed = clause_stances.get(pin.clause_id)
        if recomputed is None:
            # Clause no longer present in this compile — nothing to check
            # against; preserve the pin (and any prior conflict) unchanged.
            merged.append(pin.to_dict())
            continue
        if recomputed == pin.baseline_stance:
            # Evidence unchanged since the pin was made — no conflict.
            merged.append(pin.with_conflict(None).to_dict())
        else:
            merged.append(
                pin.with_conflict(
                    {
                        "flagged_at": checked_at,
                        "recomputed_historical_stance": recomputed,
                        "reason": (
                            f"historical_stance changed from {pin.baseline_stance!r} to "
                            f"{recomputed!r} since this position was pinned"
                        ),
                    }
                ).to_dict()
            )
    return {"pins": merged}
