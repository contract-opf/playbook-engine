"""Tests for issue #54: mine_corpus must fail loud on a corpus with zero
minable documents instead of silently writing an empty observation store
and exiting 0.

Before this fix, ``mine_corpus`` iterated only subdirectories of
``corpus_dir`` with no guard for "zero documents yielded a supported
version file" — a loose-files-in-corpus-root layout mistake (the most
likely first-run mistake) produced a green "OK" with an empty store, and
the failure only surfaced later, misleadingly, at project/compile time.

Merged finding (same defect, second report): mine_corpus treated
dot-directories (.cache, .git, .DS_Store) as agreement folders, unlike
corpus_linter and cli.segment_cmd, producing spurious per-run "no supported
files — skipping" lines and a document count that disagreed with
lint-corpus.

SECURITY NOTE: all fixtures use programmatically constructed, synthetic
RTF text with fictional party names only (e.g. "Alpha Corp"). No real
agreement files are referenced.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from playbook_engine.config import load_config
from playbook_engine.corpus_linter import lint_corpus
from playbook_engine.pipeline import PipelineError, mine_corpus
from playbook_engine.taxonomy import load_taxonomy

_RTF_BODY = (
    r"{\rtf1\ansi\deff0"
    r"{\fonttbl{\f0\froman\fcharset0 Times New Roman;}}"
    r"\f0\fs24 "
    r"1. Indemnification\par "
    r"Alpha Corp shall indemnify Beta University against third-party claims.\par "
    r"}"
)


def _make_taxonomy(tmp_path: Path) -> Path:
    tax_path = tmp_path / "taxonomy.yaml"
    tax_path.write_text(
        "source: fictional\nentries:\n  - id: TERM\n    label: Term\n    status: active\n",
        encoding="utf-8",
    )
    return tax_path


def _make_config(tmp_path: Path, taxonomy_path: Path) -> Path:
    config_path = tmp_path / "playbook.config.yaml"
    config_path.write_text(
        yaml.dump(
            {
                "agreement_type": {"id": "fictional-agreement", "name": "Fictional Agreement"},
                "baseline": {"template": None},
                "taxonomy": str(taxonomy_path),
            }
        ),
        encoding="utf-8",
    )
    return config_path


def _load(tmp_path: Path) -> tuple[object, object]:
    taxonomy_path = _make_taxonomy(tmp_path)
    config_path = _make_config(tmp_path, taxonomy_path)
    return load_config(config_path), load_taxonomy(taxonomy_path)


def test_mine_corpus_on_empty_dir_raises_pipeline_error(tmp_path: Path) -> None:
    """An entirely empty corpus dir (no subdirectories at all) must fail loud."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    cfg, taxonomy = _load(tmp_path)

    with pytest.raises(PipelineError, match="one subfolder per agreement"):
        mine_corpus(
            corpus_dir=corpus_dir,
            config=cfg,
            taxonomy=taxonomy,
            out_dir=tmp_path / "out",
        )


def test_mine_corpus_on_loose_files_raises_pipeline_error_naming_the_cause(
    tmp_path: Path,
) -> None:
    """The classic first-run mistake: agreement files dropped directly in the
    corpus root instead of one-subfolder-per-agreement. Repro from the issue:
    `playbook mine loosedir --config cfg.yaml` must exit 1, not print a green
    OK with 0 observations."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "agreement-v1.rtf").write_text(_RTF_BODY, encoding="utf-8")
    cfg, taxonomy = _load(tmp_path)

    with pytest.raises(PipelineError, match="playbook stage") as exc_info:
        mine_corpus(
            corpus_dir=corpus_dir,
            config=cfg,
            taxonomy=taxonomy,
            out_dir=tmp_path / "out",
        )
    # Names the actual cause and the fix, not just "store is empty".
    assert "supported file(s) sit directly in the corpus root" in str(exc_info.value)
    assert "lint-corpus" in str(exc_info.value)
    # Must not write any store artifacts on this fail-fast path.
    assert not (tmp_path / "out" / "observations.jsonl").exists()


def test_mine_corpus_subdirs_with_only_unsupported_files_raises_pipeline_error(
    tmp_path: Path,
) -> None:
    """Sibling case flagged by the verifier: subdirectories exist but none of
    them contain a supported (.docx/.pdf/.rtf) file — must trigger the same
    guard, without mentioning the loose-file hint (nothing sits in the root)."""
    corpus_dir = tmp_path / "corpus"
    deal_dir = corpus_dir / "deal-alice"
    deal_dir.mkdir(parents=True)
    (deal_dir / "notes.txt").write_text("stray notes, not an agreement", encoding="utf-8")
    cfg, taxonomy = _load(tmp_path)

    with pytest.raises(PipelineError, match="one subfolder per agreement") as exc_info:
        mine_corpus(
            corpus_dir=corpus_dir,
            config=cfg,
            taxonomy=taxonomy,
            out_dir=tmp_path / "out",
        )
    assert "corpus root" not in str(exc_info.value)


def test_mine_corpus_does_not_raise_when_some_docs_succeed_and_others_are_skipped(
    tmp_path: Path,
) -> None:
    """The guard must trigger ONLY when zero doc dirs yield any supported
    version file — a corpus mixing one real agreement with one doc dir that
    has no supported files must mine normally, not raise (verifier
    correction: don't conflate "zero docs yielded versions" with "some docs
    were merely skipped")."""
    corpus_dir = tmp_path / "corpus"
    good_dir = corpus_dir / "deal-alice"
    good_dir.mkdir(parents=True)
    (good_dir / "v1.rtf").write_text(_RTF_BODY, encoding="utf-8")
    empty_dir = corpus_dir / "deal-bob"
    empty_dir.mkdir(parents=True)
    (empty_dir / "notes.txt").write_text("stray", encoding="utf-8")
    cfg, taxonomy = _load(tmp_path)

    # Must not raise.
    mine_corpus(
        corpus_dir=corpus_dir,
        config=cfg,
        taxonomy=taxonomy,
        out_dir=tmp_path / "out",
    )
    assert (tmp_path / "out" / "observations.jsonl").exists()


def test_mine_corpus_ignores_dot_directories_no_skip_line_and_matches_linter_count(
    tmp_path: Path,
) -> None:
    """Merged finding: a `.hidden/` subdirectory (e.g. .cache, .git,
    .DS_Store) must not be treated as an agreement folder — no "skipping"
    progress line for it, and the mined document count must match
    lint-corpus's HAS_DOCUMENTS count."""
    corpus_dir = tmp_path / "corpus"
    good_dir = corpus_dir / "deal-alice"
    good_dir.mkdir(parents=True)
    (good_dir / "v1.rtf").write_text(_RTF_BODY, encoding="utf-8")
    hidden_dir = corpus_dir / ".hidden"
    hidden_dir.mkdir(parents=True)
    (hidden_dir / "junk").write_text("not an agreement", encoding="utf-8")
    cfg, taxonomy = _load(tmp_path)

    messages: list[str] = []
    mine_corpus(
        corpus_dir=corpus_dir,
        config=cfg,
        taxonomy=taxonomy,
        out_dir=tmp_path / "out",
        progress=messages.append,
    )

    assert not any(".hidden" in m for m in messages)

    manifest = (tmp_path / "out" / "corpus_manifest.json").read_text(encoding="utf-8")
    import json

    manifest_docs = json.loads(manifest)
    assert len(manifest_docs) == 1

    report = lint_corpus(corpus_dir)
    has_documents_items = [i for i in report.items if i.code == "HAS_DOCUMENTS"]
    assert len(has_documents_items) == 1
    assert "1 document subdirectory" in has_documents_items[0].message
    assert len(manifest_docs) == len(
        [d for d in corpus_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
    )
