"""Regression tests for the our_party_aliases sanity warning — issue #201.

The corpus-level check used to scan only mined observation texts (head-version
clause bodies). Party names live in recitals/preambles and signature blocks —
and often only in a NON-head version — so the warning false-alarmed on
correctly configured corpora (including the committed quickstart fixture,
where "FixtureCorp" sits in deal-alpha/v1.rtf's parties clause). The primary
signal is now a per-document full-tree scan (every version, headings + body
text) computed in ``_compute_doc_result`` and cached with the doc result.

SECURITY NOTE: All fixtures use programmatically constructed RTF text with
synthetic, fictional content. Fictional party names only ("ACME Works",
"Alpha Corp", "Beta University").
"""

from __future__ import annotations

from pathlib import Path

import yaml

from playbook_engine.config import load_config
from playbook_engine.pipeline import mine_corpus
from playbook_engine.taxonomy import load_taxonomy

_TAXONOMY_PATH = Path(__file__).parent.parent / "spec" / "taxonomy" / "affiliation-agreement.yaml"

_RTF_PROLOGUE = (
    r"{\rtf1\ansi\deff0"
    r"{\fonttbl{\f0\froman\fcharset0 Times New Roman;}}"
    r"\f0\fs24 "
)
_RTF_EPILOGUE = r"}"

_WARNING_FRAGMENT = "provenance.our_party_aliases"

# v1 — our opening draft. The ONLY place the ACME alias appears in the whole
# corpus is this version's parties recital: not in any other clause body, and
# not anywhere in the signed head version below. The pre-#201 observation-text
# scan never saw it.
_V1_ALIAS_IN_RECITAL = (
    r"1. Parties\par "
    r"This Agreement is by and between ACME Works (Company) "
    r"and Alpha Corp (Client).\par "
    r"2. Governing Law\par "
    r"This Agreement is governed by California law.\par "
    r"3. Term\par "
    r"One year.\par "
)

# v2 — executed signed copy; the parties clause names roles only (no ACME
# anywhere), plus a filled signature block so detect_signed() fires and the
# observations actually compile.
_V2_SIGNED_NO_ALIAS = (
    r"1. Parties\par "
    r"This Agreement is by and between the Service Provider "
    r"and Alpha Corp (Client).\par "
    r"2. Governing Law\par "
    r"This Agreement is governed by New York law.\par "
    r"3. Term\par "
    r"Two years.\par "
    r"4. Signatures\par "
    r"By: Maria Garcia, General Counsel\par "
    r"By: David Kim, Managing Director\par "
)

# Control corpus: the alias appears nowhere in any version.
_NO_ALIAS_ANYWHERE = (
    r"1. Parties\par "
    r"This Agreement is between Alpha Corp and Beta University.\par "
    r"2. Governing Law\par "
    r"This Agreement is governed by Delaware law.\par "
    r"3. Signatures\par "
    r"By: Maria Garcia, General Counsel\par "
    r"By: David Kim, Managing Director\par "
)


def _write_rtf(path: Path, body: str) -> None:
    path.write_text(_RTF_PROLOGUE + body + _RTF_EPILOGUE, encoding="utf-8")


def _mine(tmp_path: Path, deal_bodies: dict[str, str]) -> list[str]:
    """Build a one-deal corpus from {version_stem: rtf_body}, mine it, and
    return the captured progress lines."""
    corpus_dir = tmp_path / "corpus"
    deal_dir = corpus_dir / "deal-x"
    deal_dir.mkdir(parents=True)
    for stem, body in deal_bodies.items():
        _write_rtf(deal_dir / f"{stem}.rtf", body)

    cfg = {
        "agreement_type": {
            "id": "educational-affiliation",
            "name": "Educational Affiliation Agreement",
        },
        "baseline": {"template": None},
        "taxonomy": str(_TAXONOMY_PATH),
        "provenance": {"our_party_aliases": ["ACME Works", "ACME"]},
    }
    config_path = tmp_path / "playbook.config.yaml"
    config_path.write_text(yaml.dump(cfg), encoding="utf-8")

    lines: list[str] = []
    mine_corpus(
        corpus_dir=corpus_dir,
        config=load_config(config_path),
        taxonomy=load_taxonomy(_TAXONOMY_PATH),
        out_dir=tmp_path / "out",
        progress=lines.append,
    )
    return lines


def test_alias_only_in_non_head_recital_does_not_warn(tmp_path: Path) -> None:
    """An alias present ONLY in the recital of a non-head version is a
    correctly configured corpus — the warning must NOT fire (issue #201:
    the old observation-text scan false-alarmed on exactly this shape).
    """
    lines = _mine(
        tmp_path,
        {"v1": _V1_ALIAS_IN_RECITAL, "v2": _V2_SIGNED_NO_ALIAS},
    )
    offending = [ln for ln in lines if _WARNING_FRAGMENT in ln]
    assert offending == [], (
        f"alias sanity warning fired despite 'ACME Works' in v1's recital: {offending}"
    )


def test_alias_absent_everywhere_still_warns(tmp_path: Path) -> None:
    """The warning must still fire when the aliases genuinely appear nowhere —
    the fix widens the scanned surface, it must not neuter the check."""
    lines = _mine(tmp_path, {"v1": _NO_ALIAS_ANYWHERE})
    assert any(_WARNING_FRAGMENT in ln for ln in lines), (
        "alias sanity warning did not fire on a corpus with no alias match anywhere"
    )
