"""Pipeline wiring for signature-block stripping (issue #217).

``_compute_doc_result`` cuts each version's signature block out of the clause
text, but the block is still EVIDENCE for two readers, which therefore get
the unstripped tree:

- the our-party alias scan (issue #201: "so recitals/preambles and signature
  blocks count"). An alias that appears only in a signature-block caption
  must still count as a match;
- ``detect_signed``. An executed copy whose only execution evidence is its
  ``By: /s/`` block (basis ``dual_signatures``) must still read as signed.

Both are driven through ``mine_corpus`` on one synthetic deal. Its executed
version's signed verdict and its alias match both depend on the block itself
(no DocuSign certificate, no signature heading, the alias nowhere else), so
pointing either reader at the stripped tree turns these tests red.

SECURITY NOTE: synthetic RTF fixtures written at test runtime. Fictional
names only ("ACME Works", "Alpha Corp", "Sam Signer", "Robin Roe").
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from playbook_engine.config import load_config
from playbook_engine.pipeline import mine_corpus
from playbook_engine.rtf_ingester import ingest_rtf
from playbook_engine.signed_detector import detect_signed, strip_signature_block
from playbook_engine.taxonomy import load_taxonomy

_TAXONOMY_PATH = Path(__file__).parent.parent / "spec" / "taxonomy" / "nda.yaml"

_RTF_PROLOGUE = (
    r"{\rtf1\ansi\deff0"
    r"{\fonttbl{\f0\froman\fcharset0 Times New Roman;}}"
    r"\f0\fs24 "
)
_RTF_EPILOGUE = r"}"

_ALIAS = "ACME Works"
_ALIAS_WARNING_FRAGMENT = "provenance.our_party_aliases"

# The parties clause names "us" by role only, so the alias appears nowhere in
# either version's clause language.
_CLAUSES = (
    r"1. Parties\par "
    r"This Agreement is entered into by the Service Provider and Alpha Corp.\par "
    r"2. Confidentiality\par "
    r"Each party shall keep the other party's confidential information secret "
    r"for {years} years.\par "
    r"3. Counterparts\par "
    r"This Agreement may be executed in counterparts.\par "
)

# v2's execution page: no heading (the RTF ingester absorbs it into clause 3),
# no DocuSign certificate. The two /s/ lines are its only execution evidence,
# and the "ACME Works" caption is the only place the alias appears.
_EXECUTION_PAGE = (
    r"IN WITNESS WHEREOF, the parties have executed this Agreement.\par "
    r"ACME Works\par "
    r"By: /s/ Sam Signer\par "
    r"Name: Sam Signer\par "
    r"Alpha Corp\par "
    r"By: /s/ Robin Roe\par "
    r"Name: Robin Roe\par "
)

_V1_DRAFT = _CLAUSES.replace("{years}", "three")
_V2_EXECUTED = _CLAUSES.replace("{years}", "five") + _EXECUTION_PAGE


def _write_deal(tmp_path: Path) -> Path:
    corpus_dir = tmp_path / "corpus"
    deal_dir = corpus_dir / "deal-x"
    deal_dir.mkdir(parents=True)
    for stem, body in (("v1", _V1_DRAFT), ("v2", _V2_EXECUTED)):
        (deal_dir / f"{stem}.rtf").write_text(
            _RTF_PROLOGUE + body + _RTF_EPILOGUE, encoding="utf-8"
        )
    return corpus_dir


def _mine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, list[str]]:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    corpus_dir = _write_deal(tmp_path)
    cfg = {
        "agreement_type": {"id": "nda", "name": "Mutual Non-Disclosure Agreement"},
        "baseline": {"template": None},
        "taxonomy": str(_TAXONOMY_PATH),
        "provenance": {"our_party_aliases": [_ALIAS]},
    }
    config_path = tmp_path / "playbook.config.yaml"
    config_path.write_text(yaml.dump(cfg), encoding="utf-8")
    out_dir = tmp_path / "out"
    lines: list[str] = []
    mine_corpus(
        corpus_dir=corpus_dir,
        config=load_config(config_path),
        taxonomy=load_taxonomy(_TAXONOMY_PATH),
        out_dir=out_dir,
        progress=lines.append,
    )
    return out_dir, lines


def _assert_block_was_stripped(out_dir: Path) -> None:
    """The block really left the clause text, so the alias and the /s/
    lines survive ONLY in the unstripped trees. Without this, the tests
    below could pass because stripping silently stopped happening."""
    observations = (out_dir / "observations.jsonl").read_text(encoding="utf-8")
    assert observations.strip(), "no observations mined"
    trees = [p.read_text(encoding="utf-8") for p in (out_dir / "normalized").rglob("*.json")]
    for residue in (_ALIAS, "By: /s/", "IN WITNESS WHEREOF"):
        assert residue not in observations, f"{residue!r} left in observations.jsonl"
        assert all(residue not in t for t in trees), f"{residue!r} left in normalized/"
    manifest = json.loads((out_dir / "corpus_manifest.json").read_text(encoding="utf-8"))
    spans = {row["version"]: row["signature_block_span"] for row in manifest[0]["version_ingest"]}
    assert spans["v1"] is None
    assert isinstance(spans["v2"], list)


def test_alias_only_in_signature_block_still_matches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The alias scan reads the UNSTRIPPED trees: an our-party alias whose
    only appearance is a signature-block caption is still a match, so the
    zero-match warning must not fire."""
    out_dir, lines = _mine(tmp_path, monkeypatch)
    _assert_block_was_stripped(out_dir)

    offending = [ln for ln in lines if _ALIAS_WARNING_FRAGMENT in ln]
    assert offending == [], (
        f"alias sanity warning fired although {_ALIAS!r} is in v2's signature block "
        f"(the alias scan read the stripped tree?): {offending}"
    )


def test_signed_verdict_that_depends_on_the_block_survives_stripping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """detect_signed reads the UNSTRIPPED tree: an executed copy whose only
    execution evidence is its By: /s/ block still reads as signed."""
    # Premise: the verdict depends on the block itself. Unstripped it is
    # dual_signatures; stripped, nothing is left to read.
    v2_tree = ingest_rtf(_write_deal(tmp_path / "premise") / "deal-x" / "v2.rtf", "d", "v2").tree
    unstripped = detect_signed(v2_tree)
    assert (unstripped.signed, unstripped.basis) == (True, "dual_signatures")
    assert detect_signed(strip_signature_block(v2_tree)[0]).signed is False

    out_dir, _lines = _mine(tmp_path, monkeypatch)
    _assert_block_was_stripped(out_dir)

    manifest = json.loads((out_dir / "corpus_manifest.json").read_text(encoding="utf-8"))
    assert manifest[0]["signed_version"] == 2, (
        "the executed v2 read as unsigned: signed-copy detection lost the block"
    )
    outcomes = {
        json.loads(line)["outcome"]
        for line in (out_dir / "observations.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    }
    assert outcomes == {"signed"}
