"""Regression test for the known_entities zero-match warning — issue #136.

Skill-QA finding #57 (2026-08-24): the born-safe pseudonymization pass
(entity_registry.pseudonymize_text) only ever redacts a known_entities name
that its contiguous word-sequence matcher can actually find in the raw text.
A misconfigured entry — e.g. spelled differently than the corpus's own
recitals, or written in a stopword-stripped form like "University <City>"
that skips over the "of" an actual "University of <City>" recital uses —
silently matches nothing, and the real name survives unredacted into
observations.jsonl / playbook.opf.json. mine_corpus already warns when NONE
of provenance.our_party_aliases match anywhere in the corpus (issue #182);
this test covers the analogous warning for provenance.known_entities, fired
per-entry so a corpus with a mix of matching and non-matching names still
flags the non-matching ones.

SECURITY NOTE: All fixtures use programmatically constructed RTF text with
synthetic, fictional content. "Fictional State University" / "Example
Institute" are stand-in fictional counterparty names, not real institutions.
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

_WARNING_FRAGMENT = "provenance.known_entities"


def _write_rtf(path: Path, body: str) -> None:
    path.write_text(_RTF_PROLOGUE + body + _RTF_EPILOGUE, encoding="utf-8")


def _mine(tmp_path: Path, body: str, known_entities: list[str]) -> list[str]:
    """Build a one-deal corpus, mine it with *known_entities* configured, and
    return the captured progress lines."""
    corpus_dir = tmp_path / "corpus"
    deal_dir = corpus_dir / "deal-x"
    deal_dir.mkdir(parents=True)
    _write_rtf(deal_dir / "v1.rtf", body)

    cfg = {
        "agreement_type": {
            "id": "educational-affiliation",
            "name": "Educational Affiliation Agreement",
        },
        "baseline": {"template": None},
        "taxonomy": str(_TAXONOMY_PATH),
        "provenance": {
            "our_party_aliases": ["Alpha Corp"],
            "known_entities": known_entities,
        },
    }
    config_path = tmp_path / "playbook.config.yaml"
    config_path.write_text(yaml.dump(cfg), encoding="utf-8")

    lines: list[str] = []
    mine_corpus(
        corpus_dir=corpus_dir,
        config=load_config(config_path),
        taxonomy=load_taxonomy(_TAXONOMY_PATH),
        out_dir=tmp_path / "out",
        entity_registry_path=tmp_path / "entity_registry.json",
        progress=lines.append,
    )
    return lines


_BODY = (
    r"1. Indemnification\par "
    r"Alpha Corp shall indemnify Example Institute of Fictional City against "
    r"third-party claims arising from the placement programme.\par "
    r"2. Governing Law\par "
    r"This agreement is governed by the laws of the State of Delaware.\par "
)


def test_known_entity_present_verbatim_does_not_warn(tmp_path: Path) -> None:
    """A known_entities entry that matches the corpus text verbatim must NOT
    trigger the warning — this is the healthy, correctly configured case."""
    lines = _mine(tmp_path, _BODY, ["Example Institute of Fictional City"])
    offending = [ln for ln in lines if _WARNING_FRAGMENT in ln]
    assert offending == [], (
        f"known_entities warning fired despite a verbatim match in the corpus: {offending}"
    )


def test_known_entity_absent_everywhere_warns(tmp_path: Path) -> None:
    """A known_entities entry that appears nowhere in the corpus (a plain
    config typo) must warn — nothing will be pseudonymized for it."""
    lines = _mine(tmp_path, _BODY, ["Totally Different Institution"])
    offending = [ln for ln in lines if _WARNING_FRAGMENT in ln]
    assert offending, "known_entities warning did not fire for a name absent from the corpus"
    assert any("Totally Different Institution" in ln for ln in offending)


def test_known_entity_stopword_stripped_form_warns(tmp_path: Path) -> None:
    """The exact leak shape from skill-QA finding #57: a known_entities entry
    written with a stopword ("of") stripped out never matches the corpus's
    "University of X" form, so pseudonymization silently does nothing for it
    — the warning must fire so this is caught before anything is shared."""
    lines = _mine(tmp_path, _BODY, ["Example Institute Fictional City"])
    offending = [ln for ln in lines if _WARNING_FRAGMENT in ln]
    assert offending, (
        "known_entities warning did not fire for a stopword-stripped entry that "
        "can never match the corpus's actual 'X of Y' spelling"
    )
    assert any("Example Institute Fictional City" in ln for ln in offending)


def test_partial_match_only_warns_for_the_unmatched_entry(tmp_path: Path) -> None:
    """With two configured entries where only one matches, the warning must
    name only the unmatched one — a correctly configured entry must not be
    swept up in the same warning."""
    lines = _mine(
        tmp_path,
        _BODY,
        ["Example Institute of Fictional City", "Nonexistent Counterparty LLC"],
    )
    offending = [ln for ln in lines if _WARNING_FRAGMENT in ln]
    assert offending, "known_entities warning did not fire for the unmatched entry"
    joined = " ".join(offending)
    assert "Nonexistent Counterparty LLC" in joined
    assert "Example Institute of Fictional City" not in joined
