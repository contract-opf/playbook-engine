"""Scoped cache-format invalidation — migrate what can be migrated, discard only
what a change actually broke.

Why this module exists
----------------------
A content-addressed cache whose key carries a ``format_version`` component
(``extraction._extraction_cache_payload``) has exactly one invalidation move
available to it: bump the number and lose *everything*. That is the correct
move only when a change invalidates every entry. It was used twice inside one
month for changes that did not:

  - the issue #81 bump was **representational** — the stored ``extractor``
    string grew into a structured label. The old bytes were still right; only
    their shape was wrong.
  - the issue #84 bump was **semantic but narrow** — output changed for
    redline DOCX that had hit the docling->legacy fallback, a minority of any
    corpus. Every other entry was still correct and was discarded anyway.

For the extraction cache the bill for a discard is not abstract: a cold cache
over a 44-document / 161-version corpus is 1h45m-5h17m of docling/pdfplumber
wall-clock, a warm one is ~0. So a bump that throws away entries it did not
have to is a real, measurable cost.

The model
---------
A format version is no longer a bare number but a rung on a **ladder**: an
ordered tuple of :class:`CacheFormatStep`, one per bump, each declaring what
its change did to already-stored entries:

  - ``migrate``  — rewrite an old entry into the new shape (representational
    changes). Returning ``None`` from the callback means "this particular
    entry cannot be rewritten" and invalidates just that entry.
  - ``affects``  — an applicability predicate: ``True`` when THIS entry is one
    the change actually broke (invalidate it), ``False`` when it is not (keep
    it). Returning ``None`` means "cannot decide from what is stored", which
    is treated as ``True`` — see "Conservatism" below.
  - ``discard_all`` — the old blunt behavior, still available, but now a
    deliberate declaration for a bump that genuinely invalidates everything
    rather than the silent default.

A step may declare both hooks: ``affects`` is evaluated first (an entry the
change broke is dropped, not rewritten), then ``migrate`` reshapes whatever
survived.

:func:`carry_forward` walks an entry stored at some older version up the
ladder one rung at a time, applying each step in turn, and reports either the
migrated value or the exact rung that invalidated it.

Conservatism
------------
Migration is never allowed to *guess* an entry into looking current. The rules:

  - A predicate that cannot decide (``None``) invalidates. An unnecessary
    re-extraction costs minutes; a wrongly-preserved entry silently poisons
    every downstream artifact derived from it, potentially forever (the entry
    is now filed under the CURRENT version, so no later bump will catch it).
  - A migration callback that cannot faithfully reconstruct a field returns
    ``None`` rather than inventing a plausible default.
  - Steps see only the stored value plus a caller-supplied *context* mapping
    (for extraction: the source file's extension and the key's extractor
    environment). Anything not derivable from those is undecidable by
    definition.

This module is deliberately free of any knowledge of extraction, paths, or
I/O: it is pure ladder-walking over dicts, so the same mechanism can serve
the other format-version constants in the engine (e.g.
``pipeline._VERSION_INGEST_REASON_VERSION``) without inheriting extraction's
vocabulary.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

#: A stored cache value, as it round-trips through the JSONL store.
CacheValue = dict[str, Any]

#: Extra facts a step may consult that are not inside the stored value itself
#: (for the extraction cache: ``suffix``, ``extractor_env``, ``legacy_reason``).
#: Deliberately an opaque mapping here — this module never interprets it.
Context = Mapping[str, Any]

#: Returns ``True`` if *this* entry is one the bump's change actually broke
#: (invalidate), ``False`` if the entry is unaffected (keep), ``None`` if the
#: stored value plus context cannot decide (treated as ``True``).
AffectsFn = Callable[[CacheValue, Context], "bool | None"]

#: Rewrites an old-shaped entry into the new shape, or returns ``None`` when
#: this particular entry cannot be faithfully rewritten (invalidate it).
MigrateFn = Callable[[CacheValue, Context], "CacheValue | None"]


@dataclass(frozen=True)
class CacheFormatStep:
    """One rung of a cache-format ladder: what bumping to *version* did.

    Attributes:
        version: The format version this step PRODUCES (the value that ends up
            in the cache key after the bump). Steps are ordered oldest-first;
            the last step's version is the current format version.
        issue: Issue reference for the change (e.g. ``"#84"``), surfaced in
            log lines so an operator can see WHY their cache moved.
        summary: One-line human description of the change.
        discard_all: ``True`` for a bump that genuinely invalidates every
            entry. Short-circuits both hooks. This is the old global-bump
            behavior, kept as an explicit, greppable declaration rather than
            the default.
        affects: Applicability predicate — see :data:`AffectsFn`. ``None``
            means "this change broke no pre-existing entry".
        migrate: Shape rewriter — see :data:`MigrateFn`. ``None`` means "the
            stored shape did not change".
    """

    version: str
    issue: str
    summary: str
    discard_all: bool = False
    affects: AffectsFn | None = None
    migrate: MigrateFn | None = None

    def __post_init__(self) -> None:
        if self.discard_all and (self.affects is not None or self.migrate is not None):
            raise ValueError(
                f"format step {self.version!r} declares discard_all together with "
                "affects/migrate; a full discard makes both unreachable — drop "
                "discard_all, or drop the hooks"
            )


@dataclass(frozen=True)
class MigrationResult:
    """Outcome of walking one stored entry up the ladder.

    Attributes:
        value: The migrated value, ready to be re-filed under the current
            format version — or ``None`` when the entry was invalidated.
        from_version: The version the entry was stored under.
        applied: Versions whose steps were applied successfully, in order.
        invalidated_by: Version of the step that invalidated the entry, or
            ``None`` when it survived.
        detail: Human-readable reason, suitable for a log line.
    """

    value: CacheValue | None
    from_version: str
    applied: tuple[str, ...]
    invalidated_by: str | None
    detail: str

    @property
    def migrated(self) -> bool:
        """``True`` when the entry survived the whole ladder."""
        return self.value is not None


def ladder_versions(ladder: tuple[CacheFormatStep, ...], origin: str) -> tuple[str, ...]:
    """Every format version this ladder knows about, oldest first.

    *origin* is the version that existed before the ladder's first bump (the
    ladder itself only names versions that bumps PRODUCED). Used by a cache to
    enumerate the older keys worth probing on a miss.
    """
    return (origin, *(step.version for step in ladder))


def current_version(ladder: tuple[CacheFormatStep, ...], origin: str) -> str:
    """The version an entry written today is filed under (the ladder's head)."""
    return ladder[-1].version if ladder else origin


def carry_forward(
    value: CacheValue,
    *,
    from_version: str,
    ladder: tuple[CacheFormatStep, ...],
    context: Context,
) -> MigrationResult:
    """Walk *value* — stored under *from_version* — up to the ladder's head.

    Applies every step that produces a version newer than *from_version*, in
    order. The first step that invalidates the entry stops the walk.

    Args:
        value: The stored cache value (not mutated; steps receive copies).
        from_version: The format version *value* was filed under.
        ladder: The ordered format ladder, oldest bump first.
        context: Extra facts steps may consult (see :data:`Context`).

    Returns:
        A :class:`MigrationResult`. ``result.value is None`` means the entry
        must be discarded and recomputed.

    An entry whose *from_version* is not a version any rung PRODUCED — the
    pre-ladder origin, or a format from code we are not running — walks the
    FULL ladder, so it is judged by every predicate rather than waved through.
    """
    known = {step.version: index for index, step in enumerate(ladder)}
    remaining = ladder[known[from_version] + 1 :] if from_version in known else ladder

    carried = dict(value)
    applied: list[str] = []

    for step in remaining:
        if step.discard_all:
            return MigrationResult(
                value=None,
                from_version=from_version,
                applied=tuple(applied),
                invalidated_by=step.version,
                detail=(
                    f"format {step.version} ({step.issue}) declares a full discard: {step.summary}"
                ),
            )

        if step.affects is not None:
            verdict = step.affects(carried, context)
            if verdict is None:
                return MigrationResult(
                    value=None,
                    from_version=from_version,
                    applied=tuple(applied),
                    invalidated_by=step.version,
                    detail=(
                        f"format {step.version} ({step.issue}): cannot determine whether this "
                        f"entry is affected — invalidated conservatively ({step.summary})"
                    ),
                )
            if verdict:
                return MigrationResult(
                    value=None,
                    from_version=from_version,
                    applied=tuple(applied),
                    invalidated_by=step.version,
                    detail=f"format {step.version} ({step.issue}) changes this entry: {step.summary}",
                )

        if step.migrate is not None:
            rewritten = step.migrate(carried, context)
            if rewritten is None:
                return MigrationResult(
                    value=None,
                    from_version=from_version,
                    applied=tuple(applied),
                    invalidated_by=step.version,
                    detail=(
                        f"format {step.version} ({step.issue}): entry cannot be rewritten into "
                        f"the new shape — invalidated ({step.summary})"
                    ),
                )
            carried = rewritten

        applied.append(step.version)

    return MigrationResult(
        value=carried,
        from_version=from_version,
        applied=tuple(applied),
        invalidated_by=None,
        detail=(
            f"migrated {from_version} -> {'/'.join(applied)}"
            if applied
            else f"no migration needed from {from_version}"
        ),
    )
