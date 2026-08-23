"""Tests for the cache-format ladder — the scoped-invalidation mechanism that
replaces "bump the format_version and discard the whole cache".

These are pure ladder-walking tests over synthetic dicts; the extraction
cache's real rungs (#81 migration, #84 predicate) are exercised end-to-end in
tests/test_extraction.py.
"""

from __future__ import annotations

import pytest

from playbook_engine.cache_format import (
    CacheFormatStep,
    CacheValue,
    Context,
    carry_forward,
    current_version,
    ladder_versions,
)


def _step(version: str, **kwargs: object) -> CacheFormatStep:
    return CacheFormatStep(
        version=version,
        issue=f"#{version}00",
        summary=f"change {version}",
        **kwargs,  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# Ladder bookkeeping
# ---------------------------------------------------------------------------


def test_ladder_versions_lists_origin_then_every_bump() -> None:
    ladder = (_step("2"), _step("3"))
    assert ladder_versions(ladder, "1") == ("1", "2", "3")


def test_current_version_is_the_ladder_head() -> None:
    assert current_version((_step("2"), _step("3")), "1") == "3"
    assert current_version((), "1") == "1", "an empty ladder leaves the origin current"


# ---------------------------------------------------------------------------
# Migration (representational changes)
# ---------------------------------------------------------------------------


def test_migration_applies_every_rung_above_the_stored_version() -> None:
    def add_a(value: CacheValue, _: Context) -> CacheValue:
        return {**value, "a": True}

    def add_b(value: CacheValue, _: Context) -> CacheValue:
        return {**value, "b": True}

    ladder = (_step("2", migrate=add_a), _step("3", migrate=add_b))

    result = carry_forward({"x": 1}, from_version="1", ladder=ladder, context={})
    assert result.value == {"x": 1, "a": True, "b": True}
    assert result.applied == ("2", "3")
    assert result.invalidated_by is None
    assert result.migrated is True


def test_migration_skips_rungs_at_or_below_the_stored_version() -> None:
    """An entry stored at "2" must not be re-run through the "2" rung — that
    rung's rewrite already happened when the entry was written."""

    def add_a(value: CacheValue, _: Context) -> CacheValue:
        return {**value, "seen": [*value.get("seen", []), "2"]}

    def add_b(value: CacheValue, _: Context) -> CacheValue:
        return {**value, "seen": [*value.get("seen", []), "3"]}

    ladder = (_step("2", migrate=add_a), _step("3", migrate=add_b))

    result = carry_forward({}, from_version="2", ladder=ladder, context={})
    assert result.value == {"seen": ["3"]}
    assert result.applied == ("3",)


def test_entry_already_at_the_head_needs_no_migration() -> None:
    ladder = (_step("2"), _step("3"))
    result = carry_forward({"x": 1}, from_version="3", ladder=ladder, context={})
    assert result.value == {"x": 1}
    assert result.applied == ()
    assert "no migration needed" in result.detail


def test_carry_forward_does_not_mutate_the_stored_value() -> None:
    stored = {"x": 1}
    ladder = (_step("2", migrate=lambda value, _: {**value, "x": 2}),)
    result = carry_forward(stored, from_version="1", ladder=ladder, context={})
    assert stored == {"x": 1}, "the caller's stored dict must be left alone"
    assert result.value == {"x": 2}


def test_migration_returning_none_invalidates_only_that_entry() -> None:
    """A migration that cannot faithfully rewrite THIS entry discards it —
    without that verdict leaking to any other entry."""
    ladder = (_step("2", migrate=lambda value, _: None if value.get("odd") else dict(value)),)

    dropped = carry_forward({"odd": True}, from_version="1", ladder=ladder, context={})
    kept = carry_forward({"odd": False}, from_version="1", ladder=ladder, context={})

    assert dropped.value is None
    assert dropped.invalidated_by == "2"
    assert dropped.migrated is False
    assert "cannot be rewritten" in dropped.detail
    assert kept.value == {"odd": False}


def test_migration_context_reaches_the_callback() -> None:
    ladder = (_step("2", migrate=lambda value, ctx: {**value, "env": ctx["extractor_env"]}),)
    result = carry_forward(
        {}, from_version="1", ladder=ladder, context={"extractor_env": "docling"}
    )
    assert result.value == {"env": "docling"}


# ---------------------------------------------------------------------------
# Applicability predicate (narrow semantic changes)
# ---------------------------------------------------------------------------


def test_predicate_invalidates_only_the_entries_it_names() -> None:
    ladder = (_step("2", affects=lambda value, _: bool(value["redline"])),)

    affected = carry_forward({"redline": True}, from_version="1", ladder=ladder, context={})
    untouched = carry_forward({"redline": False}, from_version="1", ladder=ladder, context={})

    assert affected.value is None
    assert affected.invalidated_by == "2"
    assert "changes this entry" in affected.detail
    assert untouched.value == {"redline": False}, "an unaffected entry survives the bump"


def test_undecidable_predicate_invalidates_conservatively() -> None:
    ladder = (_step("2", affects=lambda value, _: None),)
    result = carry_forward({"x": 1}, from_version="1", ladder=ladder, context={})
    assert result.value is None
    assert result.invalidated_by == "2"
    assert "cannot determine" in result.detail


def test_predicate_runs_before_migrate_on_the_same_rung() -> None:
    """An entry the change broke is dropped, never rewritten into looking
    current — so a rung that both reshapes and breaks entries is safe."""
    calls: list[str] = []

    def migrate(value: CacheValue, _: Context) -> CacheValue:
        calls.append("migrate")
        return dict(value)

    ladder = (_step("2", affects=lambda value, _: True, migrate=migrate),)
    result = carry_forward({}, from_version="1", ladder=ladder, context={})

    assert result.value is None
    assert calls == [], "migrate must not run for an entry the predicate rejected"


def test_a_later_rung_can_invalidate_what_an_earlier_rung_migrated() -> None:
    ladder = (
        _step("2", migrate=lambda value, _: {**value, "reason": "backend-error"}),
        _step("3", affects=lambda value, _: value.get("reason") == "backend-error"),
    )
    result = carry_forward({}, from_version="1", ladder=ladder, context={})
    assert result.value is None
    assert result.applied == ("2",), "the #2 rung applied; the #3 rung then rejected the result"
    assert result.invalidated_by == "3"


# ---------------------------------------------------------------------------
# discard_all — the blunt option, kept but made deliberate
# ---------------------------------------------------------------------------


def test_discard_all_invalidates_every_entry() -> None:
    ladder = (_step("2", discard_all=True),)
    for value in ({"extractor": "docling"}, {"error": "boom"}, {}):
        result = carry_forward(value, from_version="1", ladder=ladder, context={})
        assert result.value is None
        assert result.invalidated_by == "2"
        assert "full discard" in result.detail


def test_discard_all_short_circuits_later_rungs() -> None:
    later: list[str] = []
    ladder = (
        _step("2", discard_all=True),
        _step("3", migrate=lambda value, _: later.append("3") or dict(value)),  # type: ignore[func-returns-value]
    )
    assert carry_forward({}, from_version="1", ladder=ladder, context={}).value is None
    assert later == []


def test_discard_all_cannot_be_combined_with_hooks() -> None:
    with pytest.raises(ValueError, match="discard_all"):
        CacheFormatStep(
            version="2",
            issue="#1",
            summary="both",
            discard_all=True,
            migrate=lambda value, _: dict(value),
        )


# ---------------------------------------------------------------------------
# Unknown provenance
# ---------------------------------------------------------------------------


def test_unknown_stored_version_walks_the_whole_ladder() -> None:
    """An entry from a format no rung produced (a parallel branch, a hand-edit)
    is judged by EVERY predicate rather than waved through."""
    seen: list[str] = []
    ladder = (
        _step("2", affects=lambda value, _: seen.append("2") or False),  # type: ignore[func-returns-value]
        _step("3", affects=lambda value, _: seen.append("3") or False),  # type: ignore[func-returns-value]
    )
    result = carry_forward({"x": 1}, from_version="99", ladder=ladder, context={})
    assert seen == ["2", "3"]
    assert result.value == {"x": 1}
