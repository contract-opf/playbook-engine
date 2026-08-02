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
        """Persist the registry to :attr:`path`, atomically and access-restricted.

        The registry carries real entity names (``canonical``) keyed by
        alias, so it is exactly as sensitive as the held-out map written by
        :func:`write_holdout_map` — the tmp file's mode is enforced 0600
        (owner read/write only) on the open file descriptor via
        ``os.fchmod`` *before* any content is written, regardless of whether
        the tmp path pre-existed (e.g. as a crash leftover with a looser
        mode), and that mode travels with the inode across ``os.replace``
        onto the final path, so there is no window where the registry (or a
        crash-leftover tmp file) is world-readable.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        payload = {"aliases": self._aliases, "canonical": self._canonical}
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        os.fchmod(fd, 0o600)  # enforce on the fd: O_CREAT's mode is ignored if tmp pre-existed
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(json.dumps(payload, indent=2, sort_keys=True))
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


def _fuzzy_name_pattern(name: str) -> re.Pattern[str] | None:
    """Compile a case/whitespace-tolerant whole-word pattern for *name*.

    Mirrors ``publisher._normalize_for_scan``'s normalization (casefold +
    whitespace collapse, issue #29) so a known name is matched the same way
    at ingest as it is scanned for at publish time: ``re.IGNORECASE`` handles
    casefold (an ALL-CAPS or mixed-case rendering matches, as it already
    did), and joining the name's own space-delimited words with ``\\s+``
    (rather than the literal whitespace ``re.escape`` would produce) makes
    the match tolerant of doubled or irregular (tab/newline) extraction
    whitespace between them — a single space in the registry name now
    matches ANY run of whitespace in the source, not only an exact single
    space, which was letting e.g. a double-spaced notices-clause rendering
    of a counterparty's name slip through un-aliased.

    Deliberately narrower than a full ``_normalize_for_scan`` mirror:
    punctuation embedded IN or adjacent to a word (e.g. the trailing period
    of "Acme Corp.") is left exactly as ``re.escape`` would match it, same as
    before (see the boundary-lookaround note below) — treating punctuation
    as an inter-word separator too, the way ``_normalize_for_scan`` does for
    its whole-document scan, would let two unrelated, adjacent sentence
    fragments (e.g. "...our vendor is Acme. Corp policy states...") collapse
    into a single false match; a detection-only scan can afford that
    over-triggering, but this function performs the actual substitution, so
    it does not take on that risk.

    Boundary-checked with ``(?<!\\w)``/``(?!\\w)`` lookarounds rather than
    ``\\b`` — these are equivalent to ``\\b`` when the name starts/ends with a
    word character, but (unlike ``\\b``) remain satisfied when it starts/ends
    with punctuation (e.g. "Acme Corp." or "Acme, Inc."), so substring
    protection is preserved without silently failing on the common case of
    legal names ending in "Inc." or "Corp." Longest names are matched first
    by the caller so a shorter known name that is a prefix/substring of a
    longer one (e.g. "State" vs. "State University") never partially shadows
    the longer, more specific match.

    Returns ``None`` for a name with no word tokens (e.g. all whitespace) —
    the caller must skip it rather than match on an empty, unbounded pattern.
    """
    words = [w for w in name.split() if w]
    if not words:
        return None
    body = r"\s+".join(re.escape(w) for w in words)
    return re.compile(r"(?<!\w)" + body + r"(?!\w)", re.IGNORECASE)


def pseudonymize_text(text: str, known_entities: list[str], registry: EntityRegistry) -> str:
    """Replace every whole-word occurrence of a known entity name in *text* with its alias.

    See :func:`_fuzzy_name_pattern` for the matching rules (case/whitespace
    tolerance, boundary handling).
    """
    if not text or not known_entities:
        return text
    result = text
    for name in sorted((n for n in known_entities if n), key=len, reverse=True):
        pattern = _fuzzy_name_pattern(name)
        if pattern is None:
            continue
        alias = registry.alias_for(name)
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
    ``playbook.opf.json``) and is created ``0600`` (owner read/write only) —
    once entity names are pseudonymized at ingest, this map is the sensitive
    asset that needs protecting, not the (now born-safe) OPF.

    The tmp file's mode is enforced 0600 on the open file descriptor via
    ``os.fchmod`` *before* any content is written (rather than chmod'd after
    ``os.replace``), and this holds regardless of whether the tmp path
    pre-existed (e.g. as a crash leftover with a looser mode), so there is no
    window where the live path — or a crash-leftover tmp file — is
    world-readable; the mode travels with the inode across the rename.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    os.fchmod(fd, 0o600)  # enforce on the fd: O_CREAT's mode is ignored if tmp pre-existed
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(json.dumps(registry.alias_map(), indent=2, sort_keys=True))
    os.replace(tmp, path)
