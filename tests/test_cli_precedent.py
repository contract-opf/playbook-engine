"""`playbook precedent`, `resolve-citation --precedent-id`, and the
``precedent.jsonl`` sidecar `playbook project` writes (issue #224).

Everything here reads the real compiled NDA example
(``examples/nda/playbook.opf.json``) and its committed synthetic corpus, the
committed v0.2 validator fixture, or drives the real `mine` -> `project`
pipeline over that corpus. No playbook is built or mutated here; the
mutated-document cases (unsigned drafts, flipped paper side) live in
``tests/test_opf_accessors.py``, which re-stamps each one and asserts it
passes ``validator.validate_document``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner
from click.testing import Result as CliResult

from playbook_engine.canonicalize import canonicalize
from playbook_engine.cli import cli
from playbook_engine.opf_accessors import (
    PRECEDENT_SIDECAR,
    SIDECARS_KEY,
    find_precedent,
    playbook_clauses,
    precedent_by_id,
    precedent_jsonl,
    verify_precedent_sidecar,
)

_ROOT = Path(__file__).resolve().parent.parent
_NDA_DIR = _ROOT / "examples" / "nda"
_NDA = _NDA_DIR / "playbook.opf.json"
_NDA_SIDECAR = _NDA_DIR / "precedent.jsonl"
_CORPUS_DIR = _NDA_DIR / "corpus"
_SMOKE_CONFIG = _NDA_DIR / "config.smoke.yaml"
_V02_FIXTURE = _ROOT / "examples" / "fixtures" / "valid_v0_2_minimal.json"


def _invoke(args: list[str]) -> CliResult:
    return CliRunner().invoke(cli, args)


def _doc() -> dict[str, Any]:
    doc: dict[str, Any] = json.loads(_NDA.read_text(encoding="utf-8"))
    return doc


# ---------------------------------------------------------------------------
# playbook precedent
# ---------------------------------------------------------------------------


def test_precedent_clause_jsonl_prints_the_ranked_records_verbatim() -> None:
    result = _invoke(["precedent", str(_NDA), "--clause", "governing_law", "--format", "jsonl"])
    assert result.exit_code == 0, result.output
    lines = result.output.splitlines()
    doc = _doc()
    expected = find_precedent(doc, "governing_law")
    assert lines == [canonicalize(p) for p in expected]
    assert len(lines) == 6
    for line in lines:
        record = json.loads(line)
        assert record == precedent_by_id(doc, record["id"])
        assert record["taxonomy_id"] == "governing_law"


def test_precedent_limit_and_clause_id_alias() -> None:
    full = _invoke(["precedent", str(_NDA), "--clause", "governing_law", "--format", "jsonl"])
    limited = _invoke(
        [
            "precedent",
            str(_NDA),
            "--clause",
            "clause.governing_law",
            "--limit",
            "2",
            "--format",
            "jsonl",
        ]
    )
    assert limited.exit_code == 0, limited.output
    assert limited.output.splitlines() == full.output.splitlines()[:2]


def test_precedent_with_no_filter_reproduces_the_sidecar_bytes() -> None:
    result = _invoke(["precedent", str(_NDA), "--format", "jsonl"])
    assert result.exit_code == 0, result.output
    assert result.output == _NDA_SIDECAR.read_text(encoding="utf-8")
    assert result.output == precedent_jsonl(_doc())


def test_precedent_by_id_and_refused() -> None:
    doc = _doc()
    record = find_precedent(doc, "compelled_disclosure", refused=True)[0]
    pid = record["precedent_id"]

    one = _invoke(["precedent", str(_NDA), "--id", pid, "--format", "jsonl"])
    assert one.exit_code == 0, one.output
    assert one.output.splitlines() == [canonicalize(precedent_by_id(doc, pid))]

    asks = _invoke(["precedent", str(_NDA), "--id", pid, "--refused", "--format", "jsonl"])
    assert asks.exit_code == 0, asks.output
    assert [json.loads(line) for line in asks.output.splitlines()] == [record]

    by_clause = _invoke(
        [
            "precedent",
            str(_NDA),
            "--clause",
            "compelled_disclosure",
            "--refused",
            "--format",
            "jsonl",
        ]
    )
    assert by_clause.exit_code == 0, by_clause.output
    assert [json.loads(line) for line in by_clause.output.splitlines()] == [record]

    none = _invoke(
        ["precedent", str(_NDA), "--clause", "governing_law", "--refused", "--format", "jsonl"]
    )
    assert none.exit_code == 0
    assert none.stdout == ""


def test_precedent_table_format() -> None:
    result = _invoke(["precedent", str(_NDA), "--clause", "limitation_of_liability"])
    assert result.exit_code == 0, result.output
    header, *rows = result.output.splitlines()
    assert header.split()[:3] == ["id", "document_id", "signed"]
    assert [row.split()[0] for row in rows] == [
        p["id"] for p in find_precedent(_doc(), "limitation_of_liability")
    ]
    refused = _invoke(["precedent", str(_NDA), "--clause", "residuals", "--refused"])
    assert refused.exit_code == 0, refused.output
    assert refused.output.splitlines()[0].split()[:3] == ["precedent_id", "document_id", "round"]


@pytest.mark.parametrize(
    ("args", "exit_code", "message"),
    [
        (["--clause", "no_such_clause"], 1, "no clause 'no_such_clause'"),
        (["--id", "prec.0000000000000000"], 1, "no precedent record"),
        (["--clause", "governing_law", "--id", "prec.x"], 2, "not both"),
        (["--refused"], 2, "--refused needs --clause or --id"),
        (["--clause", "governing_law", "--limit", "-1"], 2, "--limit"),
    ],
)
def test_precedent_errors(args: list[str], exit_code: int, message: str) -> None:
    result = _invoke(["precedent", str(_NDA), *args])
    assert result.exit_code == exit_code, result.output
    assert message in result.output


def test_precedent_refuses_a_pre_0_4_playbook() -> None:
    result = _invoke(["precedent", str(_V02_FIXTURE), "--clause", "x"])
    assert result.exit_code == 1
    assert "OPF 0.4" in result.output


# ---------------------------------------------------------------------------
# resolve-citation --precedent-id
# ---------------------------------------------------------------------------


def test_resolve_citation_by_precedent_id_matches_the_index_form() -> None:
    doc = _doc()
    clause = next(c for c in playbook_clauses(doc) if c["taxonomy_id"] == "governing_law")
    own = [p for p in doc["evidence"]["precedent"] if p["taxonomy_id"] == "governing_law"]
    for index, record in enumerate(own):
        by_id = _invoke(
            [
                "resolve-citation",
                str(_NDA),
                "--precedent-id",
                record["id"],
                "--corpus-dir",
                str(_CORPUS_DIR),
            ]
        )
        by_index = _invoke(
            [
                "resolve-citation",
                str(_NDA),
                "--clause",
                clause["id"],
                "--obs",
                str(index),
                "--corpus-dir",
                str(_CORPUS_DIR),
            ]
        )
        assert by_id.exit_code == 0, by_id.output
        assert by_id.output == by_index.output
        assert f"{record['document_id']} v{record['signed_text']['ref']['version']}" in by_id.output


@pytest.mark.parametrize(
    ("args", "exit_code", "message"),
    [
        (["--precedent-id", "prec.0000000000000000"], 1, "no precedent record"),
        (["--precedent-id", "prec.x", "--clause", "clause.governing_law"], 2, "not both"),
        (["--clause", "clause.governing_law"], 2, "--clause and --obs together"),
        ([], 2, "--precedent-id"),
    ],
)
def test_resolve_citation_argument_errors(args: list[str], exit_code: int, message: str) -> None:
    result = _invoke(["resolve-citation", str(_NDA), *args, "--corpus-dir", str(_CORPUS_DIR)])
    assert result.exit_code == exit_code, result.output
    assert message in result.output


# ---------------------------------------------------------------------------
# project writes precedent.jsonl (the real producer)
# ---------------------------------------------------------------------------


def test_project_writes_the_precedent_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`project` writes precedent.jsonl next to playbook.opf.json, its sha256
    is the one x_sidecars records, and a 0.3 projection into the same
    directory removes the sidecar rather than leaving a stale one beside a
    playbook it does not belong to."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    out_dir = tmp_path / "out"
    mine = _invoke(
        ["mine", str(_CORPUS_DIR), "--config", str(_SMOKE_CONFIG), "--out", str(out_dir)]
    )
    assert mine.exit_code == 0, mine.output

    project = _invoke(["project", str(out_dir), "--config", str(_SMOKE_CONFIG)])
    assert project.exit_code == 0, project.output
    playbook = json.loads((out_dir / "playbook.opf.json").read_text(encoding="utf-8"))
    sidecar = out_dir / PRECEDENT_SIDECAR
    assert sidecar.is_file()
    assert sidecar.read_bytes() == precedent_jsonl(playbook).encode("utf-8")
    assert verify_precedent_sidecar(playbook, sidecar)
    assert playbook[SIDECARS_KEY][PRECEDENT_SIDECAR]["records"] == len(
        playbook["evidence"]["precedent"]
    )
    lines = sidecar.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line) for line in lines] == sorted(
        playbook["evidence"]["precedent"], key=lambda p: p["id"]
    )

    older = _invoke(
        ["project", str(out_dir), "--config", str(_SMOKE_CONFIG), "--opf-version", "0.3"]
    )
    assert older.exit_code == 0, older.output
    playbook_03 = json.loads((out_dir / "playbook.opf.json").read_text(encoding="utf-8"))
    assert playbook_03["opf_version"] == "0.3"
    assert SIDECARS_KEY not in playbook_03
    assert not sidecar.exists()
