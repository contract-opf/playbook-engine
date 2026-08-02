"""The skill's pre-flight estimator must detect a warm extraction cache.

``.claude/skills/playbook-from-corpus/estimate_runtime.py`` is a standalone,
dependency-light script (stdlib only — no docling/torch import) that replicates
the engine's extraction-cache key recipe so it can report "already extracted,
0 wall-clock" without importing the engine. That replication is the fragile
part: if ``extraction``/``agent_judge`` ever change the key recipe or the
format version, the estimator would silently stop detecting cache hits and
scare the operator into a needless multi-hour re-OCR. This test pins the
coupling by proving the standalone key matches a real cache entry the ENGINE
wrote.

The key now includes the extractor environment (docling vs. legacy — issue
#77), and the documented pipeline routinely extracts under a DIFFERENT
environment (docker container, docling installed) than the one this
estimator itself runs in (host venv, no docling — see SKILL.md). A second
group of tests below pins that: the estimator must detect a cache entry the
engine wrote under docling even though nothing here gives the estimator any
signal that docling was ever involved.

Detecting a hit is not the same as it being safe to treat as 0 wall-clock,
though: the documented run always extracts under docling (the container),
so a hit that is cached under ``legacy`` ONLY will genuinely miss and
re-extract under that run — crediting it as "already extracted" anyway
(the issue #77 fix-round-2 finding) would silently turn a real multi-hour
re-OCR into a reported ``~0m``. A third group of tests below pins the
``main()``-level ETA accounting that decides this (previously uncovered —
the round-2 diff's only test exercised ``_cached_env`` directly).

A file can also legitimately be cached under BOTH environments at once
(e.g. a legacy host run followed later by a docling container run over the
same ``$OUT``) — a naive "first match wins" probe mis-reports such a file
as missing whenever the target environment isn't the one checked first. A
fourth group of tests below pins that this is credited as a hit under
EITHER target-environment selection (the issue #77 fix-round-3 finding).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from playbook_engine import extraction as engine_extraction
from playbook_engine.extraction import ExtractionCache, extract_blocks

_SCRIPT = (
    Path(__file__).parent.parent
    / ".claude"
    / "skills"
    / "playbook-from-corpus"
    / "estimate_runtime.py"
)


def _load_estimator():
    spec = importlib.util.spec_from_file_location("estimate_runtime", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_estimator_key_matches_engine_cache(tmp_path: Path) -> None:
    est = _load_estimator()

    src = tmp_path / "deal" / "v1.rtf"
    src.parent.mkdir(parents=True)
    src.write_text(r"{\rtf1 Fictional agreement text.}", encoding="utf-8")

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    cache = ExtractionCache(out_dir / "extraction_cache.jsonl")
    # The engine extracts and writes the cache entry under its OWN key, for
    # whichever extractor environment this host actually has.
    extract_blocks(src, cache=cache)

    cached_keys = est.load_cached_keys(str(out_dir))
    assert cached_keys, "estimator read no keys from a populated cache"
    # Key off the extractor ENVIRONMENT (``detect_extractor``), not the
    # adapter label ``extract_blocks`` returns: on a docling-per-file
    # fallback (extraction.py's docling->legacy fallback) those diverge —
    # the cache KEY is filed under the environment (``"docling"``) while the
    # returned/stored label is the post-fallback adapter (``"legacy"``).
    # Asserting against the returned label would key off the wrong value on
    # any docling-equipped host that hits that fallback (issue #77
    # fix-round-2 finding).
    expected_key = est._extraction_cache_key(str(src), engine_extraction.detect_extractor(src))
    assert expected_key in cached_keys, (
        "estimator's standalone key recipe no longer matches the engine's "
        "extraction-cache key — pre-flight would miss warm-cache hits and "
        "report a needless full re-OCR"
    )


def test_estimator_cold_cache_is_empty(tmp_path: Path) -> None:
    est = _load_estimator()
    # No cache file present → no keys, nothing treated as pre-extracted.
    assert est.load_cached_keys(str(tmp_path)) == set()


def test_estimator_detects_docling_written_cache_from_docling_less_estimator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression test for the issue #77 fix-round-2 finding.

    ``_detect_extractor`` used to resolve the extractor environment from
    THIS SCRIPT's own host PATH. But the documented pipeline runs
    extraction inside the docker container (Dockerfile installs docling),
    while the estimator itself is documented to run on the host venv (no
    docling — SKILL.md). A host-detected estimator therefore builds a
    ``legacy``-keyed lookup that can never match a container-written
    ``docling``-keyed entry, silently stops detecting cache hits, and
    reports a needless multi-hour re-OCR ETA for an already-extracted
    corpus — exactly what this test file exists to prevent (see the module
    docstring).

    This is proven directly against the engine's real ``ExtractionCache``:
    the entry below is written with ``detect_extractor`` pinned to
    ``"docling"`` (simulating the container), and the estimator — with no
    patching on its side at all — must still find it.
    """
    est = _load_estimator()

    src = tmp_path / "deal" / "v1.rtf"
    src.parent.mkdir(parents=True)
    src.write_text(r"{\rtf1 Fictional agreement text.}", encoding="utf-8")

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    cache = ExtractionCache(out_dir / "extraction_cache.jsonl")

    # Simulate the CONTAINER: docling is "on PATH" there and produces the
    # text, regardless of whatever this test process's real PATH has.
    monkeypatch.setattr(engine_extraction, "detect_extractor", lambda p: "docling")
    monkeypatch.setattr(
        engine_extraction,
        "_extract_docling_lines",
        lambda p: [("Docling-extracted text.", 0)],
    )
    _, _, extractor = extract_blocks(src, cache=cache)
    assert extractor == "docling"

    # The estimator gets no monkeypatch at all here — it must not need one:
    # post-fix it probes both key variants rather than trusting its own
    # host's docling availability.
    cached_keys = est.load_cached_keys(str(out_dir))
    assert cached_keys, "estimator read no keys from a populated cache"
    assert est._cached_envs(str(src), cached_keys) == {"docling"}, (
        "estimator failed to detect a docling-written cache entry — would "
        "report a needless full re-OCR ETA for an already-extracted corpus "
        "mined inside the docker container (issue #77 regression)"
    )


# ---------------------------------------------------------------------------
# main()'s ETA accounting must be scoped to the TARGET extractor environment
# (issue #77 fix-round-2, finding #1)
# ---------------------------------------------------------------------------


def _write_deal_with_legacy_only_cache_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path]:
    """Write one corpus file plus an ``ExtractionCache`` entry for it that
    was produced entirely under the ``legacy`` environment.

    Mirrors the real incident cited in the issue: the Jul 14 host run (no
    docling on that host) wrote a 150-entry all-legacy
    ``extraction_cache.jsonl``. Returns ``(corpus_dir, out_dir)``.
    """
    corpus = tmp_path / "corpus"
    deal = corpus / "deal1"
    deal.mkdir(parents=True)
    src = deal / "v1.rtf"
    src.write_text(r"{\rtf1 Fictional agreement text.}", encoding="utf-8")

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    cache = ExtractionCache(out_dir / "extraction_cache.jsonl")

    # Simulate the documented host-run-without-docling scenario: every
    # entry is written under the "legacy" environment, regardless of
    # whatever this test process's own PATH actually has.
    monkeypatch.setattr(engine_extraction, "detect_extractor", lambda p: "legacy")
    extract_blocks(src, cache=cache)

    return corpus, out_dir


def test_estimator_legacy_only_cache_not_credited_under_default_docling_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Regression test for the issue #77 fix-round-2 finding #1.

    A cache hit under EITHER environment used to be credited as 0
    wall-clock in ``main()``'s ETA accounting, so a legacy-only warm cache
    (e.g. the real Jul 14 host run's all-legacy ``extraction_cache.jsonl``
    cited in the issue) was reported as "already extracted" even though the
    documented run (``make docker-run ... mine/judge`` — the container has
    docling per the Dockerfile) targets docling and will MISS every one of
    those entries, silently re-OCR-ing the whole corpus for hours despite
    the pre-flight report. With the default target environment (docling,
    no override), a legacy-only cache must NOT produce the "~0m — corpus
    already extracted" line, and the legacy-only entry must instead be
    reported as needing re-extraction under docling.
    """
    est = _load_estimator()
    corpus, out_dir = _write_deal_with_legacy_only_cache_entry(tmp_path, monkeypatch)

    monkeypatch.delenv("PLAYBOOK_ESTIMATE_TARGET_ENV", raising=False)
    monkeypatch.setattr(sys, "argv", ["estimate_runtime.py", str(corpus), str(out_dir)])
    est.main()

    out = capsys.readouterr().out
    assert "already extracted (cache hit)" not in out, (
        "legacy-only cache was credited as already-extracted under the default "
        "docling target — the upcoming docker/docling run would actually MISS "
        "and silently re-OCR the whole corpus despite the ~0m pre-flight report "
        "(issue #77 fix-round-2 regression)"
    )
    assert "cached under legacy only — will be re-extracted under docling" in out, (
        "estimator did not separately report the legacy-only cache entry as "
        "needing re-extraction under the target (docling) environment"
    )


def test_estimator_target_env_override_credits_matching_cache_as_zero_wallclock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The target-env scoping above must not overcorrect: a cache hit that
    DOES match the (possibly operator-overridden) target environment must
    still be reported as a genuine 0-wall-clock no-op — an operator who
    really runs extraction on the host can opt out of the docling default
    via ``PLAYBOOK_ESTIMATE_TARGET_ENV=legacy`` and get correct credit for
    their own warm cache.
    """
    est = _load_estimator()
    corpus, out_dir = _write_deal_with_legacy_only_cache_entry(tmp_path, monkeypatch)

    monkeypatch.setenv("PLAYBOOK_ESTIMATE_TARGET_ENV", "legacy")
    monkeypatch.setattr(sys, "argv", ["estimate_runtime.py", str(corpus), str(out_dir)])
    est.main()

    out = capsys.readouterr().out
    assert "~0m — corpus already extracted (cache hit)" in out, (
        "a cache hit matching the (overridden) target environment must still "
        "short-circuit to 0 wall-clock"
    )


# ---------------------------------------------------------------------------
# A file cached under BOTH environments must be credited as a hit under
# EITHER target-environment selection (issue #77 fix-round-3 finding).
# ---------------------------------------------------------------------------


def _write_deal_with_dual_cache_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path]:
    """Write one corpus file plus TWO ``ExtractionCache`` entries for it —
    one under ``legacy``, one under ``docling`` — simulating a legacy host
    run followed later by a docling container run over the same ``$OUT``
    (the scenario named in the module docstring). Returns ``(corpus_dir,
    out_dir)``.
    """
    corpus = tmp_path / "corpus"
    deal = corpus / "deal1"
    deal.mkdir(parents=True)
    src = deal / "v1.rtf"
    src.write_text(r"{\rtf1 Fictional agreement text.}", encoding="utf-8")

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    cache = ExtractionCache(out_dir / "extraction_cache.jsonl")

    monkeypatch.setattr(engine_extraction, "detect_extractor", lambda p: "legacy")
    extract_blocks(src, cache=cache)

    monkeypatch.setattr(engine_extraction, "detect_extractor", lambda p: "docling")
    monkeypatch.setattr(
        engine_extraction,
        "_extract_docling_lines",
        lambda p: [("Docling-extracted text.", 0)],
    )
    extract_blocks(src, cache=cache)

    return corpus, out_dir


def test_cached_envs_reports_both_for_dual_cached_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unit-level pin: ``_cached_envs`` must return BOTH environments for a
    file cached under both, not just whichever ``_EXTRACTOR_ENVS`` happens
    to check first.

    This must fail against the pre-fix ``_cached_env`` (singular, returns
    the first match only): it would return just ``"docling"`` here since
    ``_EXTRACTOR_ENVS = ("docling", "legacy")`` checks docling first, even
    though the file is ALSO cached under legacy.
    """
    est = _load_estimator()
    corpus, out_dir = _write_deal_with_dual_cache_entry(tmp_path, monkeypatch)
    file_path = corpus / "deal1" / "v1.rtf"

    cached_keys = est.load_cached_keys(str(out_dir))
    assert cached_keys, "estimator read no keys from a populated cache"
    assert est._cached_envs(str(file_path), cached_keys) == {"docling", "legacy"}, (
        "estimator failed to report BOTH environments for a dual-cached file — "
        "a first-match-only probe would mis-report it as missing whenever the "
        "target environment isn't the one checked first (issue #77 fix-round-3)"
    )


@pytest.mark.parametrize("target_env", ["docling", "legacy"])
def test_estimator_dual_cached_file_credited_as_hit_under_either_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    target_env: str,
) -> None:
    """End-to-end pin for the issue #77 fix-round-3 finding: a file cached
    under BOTH environments must be reported as a genuine 0-wall-clock hit
    regardless of which environment is selected as the target — NOT just
    whichever one a first-match probe happens to check first.

    This must fail against the pre-fix ``main()``/``_cached_env`` when
    ``target_env == "legacy"``: ``_EXTRACTOR_ENVS`` checks ``"docling"``
    first, so the old first-match probe would report ``"docling"`` for this
    file, ``"docling" != "legacy"``, and the dual-cached file would be
    wrongly charged its full per-file re-extraction cost instead of being
    credited as already extracted.
    """
    est = _load_estimator()
    corpus, out_dir = _write_deal_with_dual_cache_entry(tmp_path, monkeypatch)

    monkeypatch.setenv("PLAYBOOK_ESTIMATE_TARGET_ENV", target_env)
    monkeypatch.setattr(sys, "argv", ["estimate_runtime.py", str(corpus), str(out_dir)])
    est.main()

    out = capsys.readouterr().out
    assert "~0m — corpus already extracted (cache hit)" in out, (
        f"a file cached under BOTH environments was not credited as a 0 "
        f"wall-clock hit under target_env={target_env!r} — a first-match-only "
        f"probe mis-reports a dual-cached file as missing whenever the target "
        f"isn't the environment checked first (issue #77 fix-round-3 regression)"
    )
    assert "will be re-extracted under" not in out, (
        "a dual-cached file (hit under the target env) must not also be "
        "reported in the 'cached under X only' bucket"
    )
