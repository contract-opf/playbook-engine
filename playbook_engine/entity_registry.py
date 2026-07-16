"""Entity registry — deterministic pseudonymization of known entities (issue #153).

Goal: the stored OPF must be confidentiality-safe **by construction**, not by a
later cleanup pass. Known entity names (counterparty institutions, etc. —
curated by a human from the corpus manifest/folder names, the same workflow
``provenance.our_party_aliases`` already uses for "our" party) are
deterministically replaced with stable aliases before clause text, summaries,
document_ids, or citations ever reach the observation store — the persisted
artifact (``observations.jsonl`` / ``corpus_manifest.json``) that
``playbook.opf.json`` is compiled from never carries a raw entity name.

This module owns exactly two things:

1. :class:`EntityRegistry` — a corpus-wide, disk-persisted ``entity -> alias``
   map. Persisted (write-through on first sight of a name) so the SAME entity
   gets the SAME alias across two runs and across two documents/playbooks —
   the whole point of "stable" pseudonymization. Default location is a
   user-owned cache dir (mirrors ``staging.DEFAULT_STAGING_ROOT``), not a
   per-out_dir file, precisely so alias stability survives across playbooks.
2. :func:`pseudonymize_text` / :func:`pseudonymize_document_id` — apply the
   registry's aliases to a string / a directory-name-shaped document id.

The registry, once inverted (``alias -> entity``), is now the SENSITIVE
artifact — it reverses pseudonymization back to a real name. Callers must
write it out via :func:`write_holdout_map` to a restricted-permission
sidecar, kept OUTSIDE the OPF artifact (never embedded in
``playbook.opf.json``) — see ``pipeline.mine_corpus``.

Security: this module never reads or writes real agreement content itself —
it only transforms strings/registries handed to it by callers.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

# Corpus-wide default location — a user-owned cache dir (not world-readable
# /tmp; same rationale as staging.DEFAULT_STAGING_ROOT, issue #135) so the
# SAME registry file is reused across mine_corpus runs/out_dirs by default,
# giving "same entity -> same alias everywhere" (cross-playbook stability)
# without any wiring beyond accepting the default.
DEFAULT_REGISTRY_PATH = Path.home() / ".cache" / "playbook-engine" / "entity_registry.json"

_ALIAS_PREFIX = "Counterparty"

_WS_RE = re.compile(r"\s+")
_SLUG_SEP_RE = re.compile(r"[^a-z0-9]+")


def _normalize(name: str) -> str:
    """Case/whitespace-insensitive registry key for *name*."""
    return _WS_RE.sub(" ", name.strip()).casefold()


def entity_slug(s: str) -> str:
    """Lowercase, ``-``-separated slug form of *s* (matches staging's naming)."""
    return _SLUG_SEP_RE.sub("-", s.lower()).strip("-")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


@dataclass
class EntityRegistry:
    """Persisted ``entity name -> stable alias`` map.

    Construct via :meth:`load`, not directly, so an existing on-disk registry
    is always picked up (cross-run stability). ``alias_for`` writes through to
    disk immediately on first sight of a new name, so stability holds even if
    a caller never calls :meth:`save` explicitly.
    """

    path: Path
    _aliases: dict[str, str] = field(default_factory=dict)  # normalized name -> alias
    _canonical: dict[str, str] = field(
        default_factory=dict
    )  # normalized name -> first-seen spelling

    @classmethod
    def load(cls, path: Path) -> EntityRegistry:
        """Load the registry at *path*, or return an empty one if absent."""
        reg = cls(path=path)
        if path.exists():
            raw = json.loads(path.read_text(encoding="utf-8"))
            reg._aliases = dict(raw.get("aliases", {}))
            reg._canonical = dict(raw.get("canonical", {}))
        return reg

    def save(self) -> None:
        """Persist the registry to :attr:`path`, atomically."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        payload = {"aliases": self._aliases, "canonical": self._canonical}
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp, self.path)

    def alias_for(self, entity_name: str) -> str:
        """Return the stable alias for *entity_name*, assigning one if new.

        Assignment is deterministic given the registry's existing contents:
        the next unused ``Counterparty-<n>`` slot, where ``n`` is one more
        than the number of distinct entities already registered. A freshly
        assigned alias is written through to disk immediately (cross-run
        stability without requiring the caller to call :meth:`save`).
        """
        key = _normalize(entity_name)
        if key in self._aliases:
            return self._aliases[key]
        alias = f"{_ALIAS_PREFIX}-{len(self._aliases) + 1}"
        self._aliases[key] = alias
        self._canonical[key] = entity_name
        self.save()
        return alias

    def alias_map(self) -> dict[str, str]:
        """Return ``alias -> canonical entity name`` for every registered entity.

        This is the SENSITIVE direction (reverses pseudonymization) — see
        :func:`write_holdout_map`.
        """
        return {alias: self._canonical[key] for key, alias in self._aliases.items()}


# ---------------------------------------------------------------------------
# Pseudonymization
# ---------------------------------------------------------------------------


def pseudonymize_text(text: str, known_entities: list[str], registry: EntityRegistry) -> str:
    """Replace every whole-word occurrence of a known entity name in *text* with its alias.

    Longest names are matched first so a shorter known name that is a prefix/
    substring of a longer one (e.g. "State" vs. "State University") never
    partially shadows the longer, more specific match. Matching is
    case-insensitive but whole-word (``\\b``-bounded) to avoid corrupting an
    unrelated word that merely contains a known name as a substring.
    """
    if not text or not known_entities:
        return text
    result = text
    for name in sorted((n for n in known_entities if n), key=len, reverse=True):
        alias = registry.alias_for(name)
        pattern = re.compile(r"\b" + re.escape(name) + r"\b", re.IGNORECASE)
        result = pattern.sub(alias, result)
    return result


def pseudonymize_document_id(
    document_id: str, known_entities: list[str], registry: EntityRegistry
) -> str:
    """Replace a known entity's slug form embedded in *document_id* with its alias slug.

    ``document_id`` is typically a directory-name slug (e.g.
    ``"state-university-2023"`` — see issue #123's evidence). This matches a
    known entity name's normalized token sequence against *document_id*'s
    normalized tokens and replaces only the matched span, leaving the rest of
    the slug (e.g. a trailing year) untouched. Returns *document_id* unchanged
    when no known entity's slug form appears in it.
    """
    if not document_id or not known_entities:
        return document_id
    tokens = entity_slug(document_id).split("-")
    for name in sorted((n for n in known_entities if n), key=len, reverse=True):
        name_tokens = entity_slug(name).split("-")
        n = len(name_tokens)
        if n == 0:
            continue
        for i in range(len(tokens) - n + 1):
            if tokens[i : i + n] == name_tokens:
                alias = registry.alias_for(name)
                tokens = tokens[:i] + [entity_slug(alias)] + tokens[i + n :]
                break
    return "-".join(tokens)


# ---------------------------------------------------------------------------
# Held-out alias -> entity map (the sensitive sidecar)
# ---------------------------------------------------------------------------


def write_holdout_map(path: Path, registry: EntityRegistry) -> None:
    """Write the ``alias -> real entity name`` map to *path*, access-restricted.

    This is the held-out, access-controlled sidecar the Goal describes: it
    lives OUTSIDE the OPF artifact (a caller must never embed its contents in
    ``playbook.opf.json``) and is chmod'd ``0600`` (owner read/write only) —
    once entity names are pseudonymized at ingest, this map is the sensitive
    asset that needs protecting, not the (now born-safe) OPF.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(registry.alias_map(), indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)
    os.chmod(path, 0o600)
