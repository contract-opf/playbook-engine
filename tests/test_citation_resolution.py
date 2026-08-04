"""Tests for resolvable citations (issue #185, OPF §4/§3.8).

Per-version content addresses (`corpus.documents[].version_files`), the
corpus snapshot hash, the validator's content-address rule, and the
`resolve-citation` reference resolver.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml
from click.testing import CliRunner

from playbook_engine.citation_resolver import (
    CitationResolutionError,
    ResolvedCitation,
    resolve_citation,
)
from playbook_engine.cli import cli
from playbook_engine.config import load_config
from playbook_engine.pipeline import compile_corpus
from playbook_engine.taxonomy import load_taxonomy
from playbook_engine.validator import validate_document

FIXTURES = Path(__file__).parent.parent / "examples" / "fixtures"

_RTF_PROLOGUE = (
    r"{\rtf1\ansi\deff0"
    r"{\fonttbl{\f0\froman\fcharset0 Times New Roman;}}"
    r"\f0\fs24 "
)

_V1 = (
    r"1. Indemnification\par "
    r"Each party shall indemnify the other against claims arising from its own negligence.\par "
    r"2. Governing Law\par "
    r"This Agreement is governed by the laws of the State of Delaware.\par "
)

_V2 = (
    r"1. Indemnification\par "
    r"Each party shall indemnify the other against third-party claims arising from its own negligence.\par "
    r"2. Governing Law\par "
    r"This Agreement is governed by the laws of the State of Delaware.\par "
    r"3. Signatures\par "
    r"By: Jane Roe, VP Operations, Alpha Corp\par "
    r"By: John Doe, Provost, University of Example\par "
)


def _write_rtf(path: Path, body: str) -> None:
    path.write_text(_RTF_PROLOGUE + body + "}", encoding="utf-8")


def _make_corpus(tmp_path: Path) -> tuple[Path, Path]:
    corpus_dir = tmp_path / "corpus"
    deal = corpus_dir / "deal-gamma"
    deal.mkdir(parents=True)
    _write_rtf(deal / "v1.rtf", _V1)
    _write_rtf(deal / "v2.rtf", _V2)
    (deal / "hints.yaml").write_text("order:\n  - v1\n  - v2\n", encoding="utf-8")

    taxonomy_path = (
        Path(__file__).parent.parent / "spec" / "taxonomy" / "affiliation-agreement.yaml"
    )
    cfg = {
        "agreement_type": {
            "id": "educational-affiliation",
            "name": "Educational Affiliation Agreement",
        },
        "baseline": {"template": None},
        "taxonomy": str(taxonomy_path),
        "provenance": {"our_party_aliases": ["Alpha Corp"]},
    }
    config_path = tmp_path / "playbook.config.yaml"
    config_path.write_text(yaml.dump(cfg), encoding="utf-8")
    return corpus_dir, config_path


def _compile(corpus_dir: Path, config_path: Path, out_dir: Path) -> dict[str, Any]:
    cfg = load_config(config_path)
    taxonomy = load_taxonomy(cfg.taxonomy_path)
    compile_corpus(corpus_dir=corpus_dir, config=cfg, taxonomy=taxonomy, out_dir=out_dir)
    return json.loads((out_dir / "playbook.opf.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def compiled(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict[str, Any], Path]:
    tmp_path = tmp_path_factory.mktemp("citation-resolution")
    corpus_dir, config_path = _make_corpus(tmp_path)
    playbook = _compile(corpus_dir, config_path, tmp_path / "out")
    playbook_path = tmp_path / "out" / "playbook.opf.json"
    return corpus_dir, playbook, playbook_path


_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


def test_version_files_emitted(tmp_path: Path) -> None:
    """Every mined corpus document carries version_files with valid digests;
    snapshot.manifest_hash is present and stable across identical compiles."""
    corpus_dir, config_path = _make_corpus(tmp_path)
    playbook_a = _compile(corpus_dir, config_path, tmp_path / "out-a")
    playbook_b = _compile(corpus_dir, config_path, tmp_path / "out-b")

    for doc in playbook_a["corpus"]["documents"]:
        version_files = doc.get("version_files")
        assert version_files, f"{doc['document_id']} missing version_files"
        assert len(version_files) == doc["versions"]
        for entry in version_files:
            assert _SHA256_RE.match(entry["sha256"]), entry
            assert entry["version"] >= 1
            assert entry["media_type"] == "application/rtf"

    snap_a = playbook_a["corpus"]["snapshot"]["manifest_hash"]
    snap_b = playbook_b["corpus"]["snapshot"]["manifest_hash"]
    assert _SHA256_RE.match(snap_a)
    assert snap_a == snap_b, "snapshot must be stable across identical compiles"


def test_resolve_citation_roundtrip(compiled: tuple[Path, dict[str, Any], Path]) -> None:
    """Observation 0 of the first clause resolves to the staged file whose
    sha256 matches, carrying the citation's clause_path/char_span."""
    corpus_dir, playbook, _ = compiled
    clause = playbook["evidence"]["clauses"][0]
    ref = clause["observed_positions"][0]["example_ref"]

    resolved = resolve_citation(playbook, clause["id"], 0, corpus_dir)

    assert resolved.file_path.is_file()
    assert corpus_dir in resolved.file_path.parents
    actual = "sha256:" + hashlib.sha256(resolved.file_path.read_bytes()).hexdigest()
    assert actual == resolved.sha256
    assert resolved.clause_path == ref["clause_path"]
    if ref.get("char_span"):
        assert list(resolved.char_span) == ref["char_span"]
    # #86: the compiled citation object (spec/playbook.schema-0.3.json's
    # closed $defs.citation) never carries a "page" key (see
    # citation_resolver's module docstring) — resolve_citation() must
    # default to None rather than crash or fabricate a value.
    assert resolved.page is None


def test_resolve_citation_tamper_fails(
    compiled: tuple[Path, dict[str, Any], Path], tmp_path: Path
) -> None:
    """Tampering with the cited file (append one byte) → exit 1, 'hash mismatch'."""
    corpus_dir, playbook, playbook_path = compiled
    clause_id = playbook["evidence"]["clauses"][0]["id"]

    resolved = resolve_citation(playbook, clause_id, 0, corpus_dir)
    tampered_dir = tmp_path / "tampered-corpus"
    shutil.copytree(corpus_dir, tampered_dir)
    tampered_file = tampered_dir / resolved.file_path.relative_to(corpus_dir)
    with tampered_file.open("ab") as f:
        f.write(b"\n")

    with pytest.raises(CitationResolutionError, match="hash mismatch"):
        resolve_citation(playbook, clause_id, 0, tampered_dir)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "resolve-citation",
            str(playbook_path),
            "--clause",
            clause_id,
            "--obs",
            "0",
            "--corpus-dir",
            str(tampered_dir),
        ],
    )
    assert result.exit_code == 1
    assert "hash mismatch" in result.output


def test_resolve_citation_cli_roundtrip(compiled: tuple[Path, dict[str, Any], Path]) -> None:
    corpus_dir, playbook, playbook_path = compiled
    clause_id = playbook["evidence"]["clauses"][0]["id"]
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "resolve-citation",
            str(playbook_path),
            "--clause",
            clause_id,
            "--obs",
            "0",
            "--corpus-dir",
            str(corpus_dir),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "verified sha256:" in result.output
    assert "file:" in result.output


# ---------------------------------------------------------------------------
# #86 — ResolvedCitation.page / describe()
# ---------------------------------------------------------------------------

_SHA_A = "sha256:" + "a" * 64


def _resolved(*, page: int | None) -> ResolvedCitation:
    return ResolvedCitation(
        document_id="doc",
        version=1,
        sha256=_SHA_A,
        file_path=Path("/corpus/doc/v1.pdf"),
        clause_path="1.2",
        char_span=(10, 20),
        page=page,
    )


def test_resolved_citation_describe_includes_page_when_present() -> None:
    assert _resolved(page=5).describe() == (
        f"doc v1 -> /corpus/doc/v1.pdf clause 1.2 chars [10, 20] (verified {_SHA_A}) page 5"
    )


def test_resolved_citation_describe_unchanged_when_page_absent() -> None:
    """Byte-identical to the pre-#86 format when page is None."""
    assert _resolved(page=None).describe() == (
        f"doc v1 -> /corpus/doc/v1.pdf clause 1.2 chars [10, 20] (verified {_SHA_A})"
    )


def _with_example_ref_page(playbook: dict[str, Any], page_value: Any) -> dict[str, Any]:
    """Deep-copy *playbook* with a raw ``page`` key injected into the first
    clause's first observation's ``example_ref`` — simulating a hand-edited/
    foreign playbook dict (the compiled schema is closed, so no producer in
    this codebase ever writes this key) to exercise resolve_citation()'s
    defensive page gate."""
    doc: dict[str, Any] = json.loads(json.dumps(playbook))
    doc["evidence"]["clauses"][0]["observed_positions"][0]["example_ref"]["page"] = page_value
    return doc


@pytest.mark.parametrize("bad_page", [0, -3, True, "5"])
def test_resolve_citation_rejects_invalid_page(
    compiled: tuple[Path, dict[str, Any], Path], bad_page: Any
) -> None:
    """fix-round-1 (#86): a hand-edited/foreign citation carrying a
    zero/negative/wrong-typed 'page' — including ``True``, which
    ``isinstance(x, int)`` alone accepts because ``bool`` subclasses
    ``int`` — must resolve to ``page=None``, never be trusted verbatim."""
    corpus_dir, playbook, _ = compiled
    clause_id = playbook["evidence"]["clauses"][0]["id"]
    tampered = _with_example_ref_page(playbook, bad_page)

    resolved = resolve_citation(tampered, clause_id, 0, corpus_dir)

    assert resolved.page is None


def test_resolve_citation_accepts_valid_page(
    compiled: tuple[Path, dict[str, Any], Path],
) -> None:
    """A hand-edited/foreign citation carrying a valid positive-int page is
    passed through unchanged."""
    corpus_dir, playbook, _ = compiled
    clause_id = playbook["evidence"]["clauses"][0]["id"]
    tampered = _with_example_ref_page(playbook, 7)

    resolved = resolve_citation(tampered, clause_id, 0, corpus_dir)

    assert resolved.page == 7


def _load_minimal() -> dict[str, Any]:
    with (FIXTURES / "valid_v0_2_minimal.json").open() as f:
        return json.load(f)


def test_validator_rejects_unlisted_citation() -> None:
    """A citation to a (doc, version) absent from version_files must fail."""
    doc = _load_minimal()
    # The fixture's citations cite university-of-example v2/v3; publish
    # version_files listing only v1 → every cited version is unaddressable.
    doc["corpus"]["documents"][0]["version_files"] = [
        {"version": 1, "sha256": "sha256:" + "a" * 64, "media_type": "application/pdf"}
    ]
    result = validate_document(doc)
    assert not result.ok
    assert any("version_files" in str(e) for e in result.errors)


def test_source_uri_optional() -> None:
    """A version_files entry validates with and without source_uri."""
    base = _load_minimal()
    versions = base["corpus"]["documents"][0]["versions"]
    entries = [
        {"version": i + 1, "sha256": "sha256:" + "b" * 64, "media_type": "application/pdf"}
        for i in range(versions)
    ]
    with_uri = json.loads(json.dumps(base))
    with_uri["corpus"]["documents"][0]["version_files"] = [
        {**e, "source_uri": f"dms://example/{e['version']}"} for e in entries
    ]
    without_uri = json.loads(json.dumps(base))
    without_uri["corpus"]["documents"][0]["version_files"] = entries

    for candidate in (with_uri, without_uri):
        result = validate_document(candidate)
        assert result.ok, [str(e) for e in result.errors]
