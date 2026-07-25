"""Shared natural-sort key for version filenames.

Used wherever version files must be listed in a stable, human-intuitive
order so that ``"v2"`` sorts before ``"v10"`` (plain lexicographic sort
would put ``"v10"`` before ``"v2"``).

Filenames are NOT dispositive for version *ordering* in this engine — see
``docs/CORPUS-LAYOUT.md`` and the content-based ``signed_detector`` /
``version_orderer`` modules, which are the real source of truth. Natural
sort here is used only for stable display/discovery ordering, and as a
last-resort tiebreak when content evidence is genuinely ambiguous (see
``induction_version_selector.py``).
"""

from __future__ import annotations

import re


def natural_sort_key(stem: str) -> list[int | str]:
    """Natural sort key so v1 < v2 < v10 (not v1 < v10 < v2)."""
    return [int(p) if p.isdigit() else p.lower() for p in re.split(r"(\d+)", stem)]
