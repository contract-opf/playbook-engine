"""Taxonomy loader and CUAD-merge utility.

A taxonomy YAML has the shape:
  source: CUAD-v1          # upstream taxonomy + version, or "custom"
  entries:
    - id: indemnification
      label: Indemnification
      status: active          # active | inactive | custom
      cuad_origin: Indemnification   # or null for custom entries
      description: "..."

Curation rules (OPF §5):
  - Pruning is done by setting status: inactive (never by deletion).
  - Custom additions use status: custom with cuad_origin: null.
  - When merging a newer upstream release, new categories enter as inactive;
    existing curation choices (any known id) are preserved.
  - A compiler MUST only classify clauses into active or custom entries.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

# Valid status values per OPF §3.3
VALID_STATUSES = frozenset({"active", "inactive", "custom"})


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TaxonomyEntry:
    id: str
    label: str
    status: str  # active | inactive | custom
    cuad_origin: str | None  # None for custom entries
    description: str = ""

    @property
    def is_classifier_eligible(self) -> bool:
        """True if a compiler may tag clauses with this entry (OPF §5)."""
        return self.status in ("active", "custom")


@dataclass
class Taxonomy:
    source: str
    entries: list[TaxonomyEntry]

    def classifier_entries(self) -> list[TaxonomyEntry]:
        """Return only active + custom entries — those a classifier may use."""
        return [e for e in self.entries if e.is_classifier_eligible]

    def get(self, entry_id: str) -> TaxonomyEntry | None:
        for e in self.entries:
            if e.id == entry_id:
                return e
        return None


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


class TaxonomyError(ValueError):
    """Friendly taxonomy error."""


def _parse_entry(raw: dict[str, Any], index: int) -> TaxonomyEntry:
    entry_id = raw.get("id")
    if not isinstance(entry_id, str) or not entry_id:
        raise TaxonomyError(f"entries[{index}]: 'id' must be a non-empty string")
    label = raw.get("label")
    if not isinstance(label, str) or not label:
        raise TaxonomyError(f"entries[{index}] ({entry_id!r}): 'label' must be a non-empty string")
    status = raw.get("status")
    if status not in VALID_STATUSES:
        raise TaxonomyError(
            f"entries[{index}] ({entry_id!r}): 'status' must be one of {sorted(VALID_STATUSES)}, got {status!r}"
        )
    cuad_origin = raw.get("cuad_origin")
    if status == "custom" and cuad_origin is not None:
        raise TaxonomyError(
            f"entries[{index}] ({entry_id!r}): custom entries must have cuad_origin: null"
        )
    description = str(raw.get("description") or "")
    return TaxonomyEntry(
        id=entry_id,
        label=label,
        status=status,
        cuad_origin=cuad_origin if cuad_origin is not None else None,
        description=description,
    )


def load_taxonomy(path: Path) -> Taxonomy:
    """Load and validate a taxonomy YAML file."""
    if not path.is_file():
        raise TaxonomyError(f"Taxonomy file not found: {path}")

    try:
        raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise TaxonomyError(f"Taxonomy file is not valid YAML: {exc}") from exc

    if not isinstance(raw, dict):
        raise TaxonomyError(f"Taxonomy file must be a YAML mapping, got {type(raw).__name__}")

    source = raw.get("source")
    if not isinstance(source, str) or not source:
        raise TaxonomyError("taxonomy.source must be a non-empty string")

    entries_raw = raw.get("entries", [])
    if not isinstance(entries_raw, list):
        raise TaxonomyError("taxonomy.entries must be a list")

    seen_ids: set[str] = set()
    entries: list[TaxonomyEntry] = []
    for i, entry_raw in enumerate(entries_raw):
        if not isinstance(entry_raw, dict):
            raise TaxonomyError(f"entries[{i}]: must be a mapping, got {type(entry_raw).__name__}")
        entry = _parse_entry(entry_raw, i)
        if entry.id in seen_ids:
            raise TaxonomyError(f"entries[{i}]: duplicate id {entry.id!r}")
        seen_ids.add(entry.id)
        entries.append(entry)

    return Taxonomy(source=source, entries=entries)


# ---------------------------------------------------------------------------
# CUAD-merge utility  (OPF §5)
# ---------------------------------------------------------------------------


def merge_taxonomy(
    existing: Taxonomy,
    upstream_entries: list[dict[str, Any]],
    new_source: str | None = None,
) -> Taxonomy:
    """Merge new entries from an upstream CUAD release into an existing taxonomy.

    Rules (OPF §5):
    - Known ids: preserve existing entry unchanged (including status).
    - New ids: add as inactive, preserving cuad_origin from the upstream data.
    - The resulting source is updated to ``new_source`` only when ``new_source``
      is a non-empty string AND at least one new entry is actually added.  If
      ``new_source`` is ``None`` or an empty string, or if no new entries are
      added, ``existing.source`` is preserved.  Pass ``new_source=None`` (the
      default) to never update the source field.

    Returns a new Taxonomy; does not mutate the originals.
    """
    existing_ids = {e.id: e for e in existing.entries}
    result_entries = list(existing.entries)  # start from all existing entries
    added = 0

    for raw in upstream_entries:
        if not isinstance(raw, dict):
            continue
        entry_id = raw.get("id")
        if not isinstance(entry_id, str) or not entry_id:
            continue
        if entry_id in existing_ids:
            continue  # preserve existing curation
        # New entry — add as inactive
        label = str(raw.get("label") or entry_id)
        cuad_origin = raw.get("cuad_origin") or raw.get("id")
        description = str(raw.get("description") or "")
        result_entries.append(
            TaxonomyEntry(
                id=entry_id,
                label=label,
                status="inactive",
                cuad_origin=cuad_origin if isinstance(cuad_origin, str) else None,
                description=description,
            )
        )
        added += 1

    effective_source = new_source if (new_source and added > 0) else existing.source
    return Taxonomy(source=effective_source, entries=result_entries)
