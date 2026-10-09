"""An independent no-known-entity scan for the born-safe tests (issue #239).

The born-safe tests assert that pseudonymization at ingest leaves no real
counterparty name anywhere in a compiled artifact. The check is
deliberately independent of the pseudonymizer under test, so the oracle
lives here, in the tests: it normalizes
(casefold, punctuation to whitespace) and matches on word boundaries — and,
for names of eight or more characters, on the whitespace-collapsed form, which
catches a no-separator folder-name slug ("westmooruniversity-2023") that
destroys the word boundaries the token-based pseudonymizer relies on.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

_PUNCT_RE = re.compile(r"[^\w\s]")
_WS_RE = re.compile(r"\s+")


def normalize(text: str) -> str:
    return _WS_RE.sub(" ", _PUNCT_RE.sub(" ", text.casefold())).strip()


def walk_strings(node: Any, path: str = "$") -> list[tuple[str, str]]:
    """Every ``(json path, string)`` in *node*, dict keys included."""
    found: list[tuple[str, str]] = []
    if isinstance(node, str):
        found.append((path, node))
    elif isinstance(node, dict):
        for key, value in node.items():
            if isinstance(key, str):
                found.append((f"{path}.{{{key}}}", key))
            found.extend(walk_strings(value, f"{path}.{key}"))
    elif isinstance(node, list):
        for idx, value in enumerate(node):
            found.extend(walk_strings(value, f"{path}[{idx}]"))
    return found


def entity_backstop_scan(doc: Any, known_entity_names: Sequence[str]) -> list[tuple[str, str]]:
    """Every ``(path, real_name)`` where a known entity name survives in *doc*."""
    names = [(n, normalize(n)) for n in known_entity_names if n and n.strip()]
    hits: list[tuple[str, str]] = []
    for path, text in walk_strings(doc):
        if not text:
            continue
        haystack = f" {normalize(text)} "
        collapsed = haystack.replace(" ", "")
        for real_name, normalized in names:
            if not normalized:
                continue
            if f" {normalized} " in haystack:
                hits.append((path, real_name))
                continue
            collapsed_name = normalized.replace(" ", "")
            if len(collapsed_name) >= 8 and collapsed_name in collapsed:
                hits.append((path, real_name))
    return hits
