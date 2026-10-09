"""Tests for the digest section (digest_version 4), `playbook digest`, and `view bundle`.

The digest_version 2 builder was retired with OPF 0.3 (issue #238); digest 3 was
replaced in place by digest 4 (issue #234).

SECURITY NOTE: All fixtures are programmatically constructed or drawn from
the synthetic examples/ fixtures. No real agreements are referenced.
"""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from playbook_engine.canonicalize import content_hash
from playbook_engine.cli import cli
from playbook_engine.digest import (
    DIGEST_VERSION_V4,
    EXEMPLAR_TOP_N,
    build_digest_v4,
    digest_token_estimate,
)
from playbook_engine.validator import validate_document

# ---------------------------------------------------------------------------
# CLI: the digest inside the OPF + view bundle on a real compile
# ---------------------------------------------------------------------------


def _compiled_out_dir(tmp_path: Path) -> Path:
    from tests.test_cli import _make_corpus  # reuse the synthetic corpus builder

    corpus_dir, config_path, out_dir = _make_corpus(tmp_path)
    runner = CliRunner()
    mine_result = runner.invoke(
        cli,
        ["mine", str(corpus_dir), "--config", str(config_path), "--out", str(out_dir)],
    )
    assert mine_result.exit_code == 0, mine_result.output
    project_result = runner.invoke(
        cli,
        ["project", str(out_dir), "--config", str(config_path)],
    )
    assert project_result.exit_code == 0, project_result.output
    return out_dir


def test_compiled_playbook_is_v05_with_digest_v4(tmp_path: Path) -> None:
    """Issue #223: the reference compiler emits OPF 0.5 with a digest_version 4
    digest equal to build_digest_v4 over the shipped document."""
    out_dir = _compiled_out_dir(tmp_path)
    pb = json.loads((out_dir / "playbook.opf.json").read_text())
    assert pb["opf_version"] == "0.5"
    assert pb["digest"]["digest_version"] == DIGEST_VERSION_V4 == "4"
    assert len(pb["digest"]["clauses"]) == len(pb["evidence"]["clauses"])
    assert "clause_count" not in pb["digest"]
    assert pb["digest"] == build_digest_v4(pb)
    # digest participates in content_hash: recompute and compare
    assert pb["identity"]["content_hash"] == content_hash(pb)
    # The digest lives inside the OPF; there is no standalone sidecar (the
    # `digest` command was retired, issue #239).
    assert not (out_dir / "playbook.digest.json").exists()
    assert "digest" not in cli.commands


def _retired_0_3_out_dir(tmp_path: Path) -> Path:
    """An out-dir holding a retired OPF 0.3-shaped playbook (issue #238).

    Carries a digest_version "2" section with the retired stance fields, the
    shape a pre-#238 0.3 out-dir on disk still has.
    """
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    doc = {
        "opf_version": "0.3",
        "evidence": {"clauses": [], "clause_library": []},
        "digest": {
            "digest_version": "2",
            "clauses": [
                {
                    "id": "clause.x",
                    "historical_stance": "accepted",
                    "preferred_variations": [],
                    "concessions": [],
                }
            ],
        },
    }
    (out_dir / "playbook.opf.json").write_text(json.dumps(doc), encoding="utf-8")
    return out_dir


def test_view_bundle_refuses_a_retired_opf_version(tmp_path: Path) -> None:
    """`playbook view bundle` refuses a 0.3 document instead of rendering it
    as an empty precedent record (issue #238)."""
    out_dir = _retired_0_3_out_dir(tmp_path)
    result = CliRunner().invoke(cli, ["view", "bundle", str(out_dir)])
    assert result.exit_code == 1
    assert "unsupported opf_version '0.3'" in result.output
    assert not (out_dir / "index.html").exists()


def test_view_bundle_embeds_canonical_json_and_digest(tmp_path: Path) -> None:
    out_dir = _compiled_out_dir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(cli, ["view", "bundle", str(out_dir)])
    assert result.exit_code == 0, result.output

    html = (out_dir / "index.html").read_text(encoding="utf-8")
    assert '<script id="opf-canonical" type="application/json">' in html
    assert '<script id="opf-digest" type="application/json">' in html

    # Extract the canonical block the way a consumer would and verify the
    # content hash over the canonical serialization.
    start = html.index('<script id="opf-canonical" type="application/json">')
    start = html.index(">", start) + 1
    end = html.index("</script>", start)
    embedded = json.loads(html[start:end])
    on_disk = json.loads((out_dir / "playbook.opf.json").read_text())
    assert embedded == on_disk
    assert embedded["identity"]["content_hash"] == content_hash(embedded)

    # Digest block parses and matches the document's digest section.
    dstart = html.index('<script id="opf-digest" type="application/json">')
    dstart = html.index(">", dstart) + 1
    dend = html.index("</script>", dstart)
    assert json.loads(html[dstart:dend]) == on_disk["digest"]


def test_view_bundle_escapes_script_closers(tmp_path: Path) -> None:
    out_dir = _compiled_out_dir(tmp_path)
    # Inject a hostile </script> into a text field of the on-disk playbook,
    # recompute nothing (bundle embeds verbatim) — the bundle must escape it.
    pb_path = out_dir / "playbook.opf.json"
    pb = json.loads(pb_path.read_text())
    pb["agreement_type"]["name"] = 'x</script><script>alert("pwned")</script>'
    pb_path.write_text(json.dumps(pb, indent=2, ensure_ascii=False))

    runner = CliRunner()
    result = runner.invoke(cli, ["view", "bundle", str(out_dir)])
    assert result.exit_code == 0, result.output
    html = (out_dir / "index.html").read_text(encoding="utf-8")
    start = html.index('<script id="opf-canonical" type="application/json">')
    end = html.index("</script>", start)
    block = html[start:end]
    assert "</script" not in block[1:], "unescaped </script> inside the JSON block"


# ---------------------------------------------------------------------------
# digest_version 4 (OPF 0.5, issues #223, #234). Every document here is built by
# the real producer — Observation rows (the shapes observation_builder
# writes: signed / unsigned / proposed_then_reversed /
# conceded_before_signing / opening, with the deterministic `standard` fact
# and `opened_with`) through
# compile_clause_positions + assemble_playbook — never a hand-built
# precedent record.
# ---------------------------------------------------------------------------


def _v05_obs(
    doc_id: str,
    text: str,
    *,
    tid: str = "governing_law",
    outcome: str = "signed",
    standard: bool = False,
    version: int = 3,
    opened_with: str | None = None,
):
    from playbook_engine.observation_builder import Observation, ObservationCitation

    return Observation(
        observation_id=f"{doc_id}/{tid}/{outcome}/{version}",
        taxonomy_id=tid,
        text_summary=text[:300],
        full_text=text,
        citation=ObservationCitation(
            document_id=doc_id, version=version, clause_path="4", char_span=(0, len(text))
        ),
        deviation="none" if standard else "substantive",
        risk_delta={"direction": "neutral", "magnitude": "none"},
        provenance="our_paper",
        outcome=outcome,
        basis="deterministic",
        standard=standard,
        opened_with=opened_with,
    )


_V05_STANDARD = "This Agreement is governed by the laws of the State of New York."


def _v05_playbook(observations: list, *, signed: dict[str, bool], perspective=None) -> dict:
    from playbook_engine.clause_position_compiler import compile_clause_positions
    from playbook_engine.observation_builder import Observation, ObservationCitation
    from playbook_engine.playbook_assembler import assemble_playbook

    template = Observation(
        observation_id="template/governing_law",
        taxonomy_id="governing_law",
        text_summary=_V05_STANDARD,
        citation=ObservationCitation(
            document_id="template", version="template", clause_path="4", char_span=None
        ),
        deviation="none",
        risk_delta={"direction": "neutral", "magnitude": "none"},
        provenance="our_paper",
        outcome="signed",
    )
    positions, _, _ = compile_clause_positions(observations, [template])
    documents = [
        {
            "document_id": doc_id,
            "provenance": "our_paper",
            "in_scope": True,
            "versions": 3,
            "signed_version": 3 if is_signed else None,
        }
        for doc_id, is_signed in sorted(signed.items())
    ]
    return assemble_playbook(
        agreement_type={"id": "nda", "name": "Mutual NDA"},
        baseline={"has_canonical_template": True},
        taxonomy={"source": "custom", "entries": []},
        clause_positions=positions,
        corpus_documents=documents,
        generated_at="2026-01-01T00:00:00Z",
        observations=observations,
        perspective=perspective,
    )


def test_v4_digest_groups_variants_by_exact_normalization() -> None:
    """Case/punctuation respellings merge; a negator never does; our standard
    text and an unsigned deal's text are never signed variants."""
    variant = "This Agreement is governed by the laws of the State of Delaware."
    observations = [
        _v05_obs("d1", _V05_STANDARD, standard=True),
        _v05_obs("d2", variant),
        _v05_obs("d3", "THIS AGREEMENT is governed by the laws of the State of Delaware"),
        _v05_obs("d4", "This Agreement is not governed by the laws of the State of Delaware."),
        _v05_obs("d5", "This Agreement is governed by the laws of Texas.", outcome="unsigned"),
    ]
    pb = _v05_playbook(
        observations, signed={"d1": True, "d2": True, "d3": True, "d4": True, "d5": False}
    )
    (clause,) = pb["digest"]["clauses"]
    assert clause["n_deals"] == 5
    assert clause["n_signed_standard"] == 1
    assert [v["n_deals"] for v in clause["signed_variants"]] == [2, 1]
    assert " not " in clause["signed_variants"][1]["text"]
    assert clause["n_variants_total"] == 2
    assert all("Texas" not in v["text"] for v in clause["signed_variants"])
    assert len(clause["signed_variants"][0]["precedent_ids"]) == 2
    # The digest is exactly what build_digest_v4 recomputes over the document.
    assert pb["digest"] == build_digest_v4(pb)


def test_v4_grouping_neutralizes_counterparty_aliases_and_perspective_party() -> None:
    """Born-safe pseudonymization writes each deal's counterparty into the
    clause text as its own ``Counterparty-<n>`` alias, so two deals that
    signed the same words differ only by alias. They are ONE variant with
    n_deals 2 — in the clause counts the validator recomputes and in the
    digest — and the same holds for refused asks. The parties' places
    swapped is a different text and never merges (OPF-SPEC §3.5.4
    "Grouping key")."""
    import dataclasses

    def aliased(obs, alias: str):
        return dataclasses.replace(obs, counterparty_ref={"alias": alias})

    signed = "{} and AlphaCorp agree that the laws of Delaware govern this Agreement."
    ask = "The laws of the home state of {} govern this Agreement."
    observations = [
        aliased(_v05_obs("d1", signed.format("Counterparty-2")), "Counterparty-2"),
        aliased(
            _v05_obs(
                "d1", ask.format("Counterparty-2"), outcome="proposed_then_reversed", version=2
            ),
            "Counterparty-2",
        ),
        aliased(_v05_obs("d2", signed.format("Counterparty-5")), "Counterparty-5"),
        aliased(
            _v05_obs(
                "d2", ask.format("COUNTERPARTY-5"), outcome="proposed_then_reversed", version=2
            ),
            "Counterparty-5",
        ),
        aliased(
            _v05_obs(
                "d3",
                "AlphaCorp and Counterparty-3 agree that the laws of Delaware govern "
                "this Agreement.",
            ),
            "Counterparty-3",
        ),
    ]
    pb = _v05_playbook(
        observations,
        signed={"d1": True, "d2": True, "d3": True},
        perspective={"party": "AlphaCorp", "counterparty_type": "Vendor"},
    )
    (clause,) = pb["evidence"]["clauses"]
    assert (clause["n_deals"], clause["n_variants"], clause["n_refused"]) == (3, 2, 1)
    (entry,) = pb["digest"]["clauses"]
    assert [(v["n_deals"], v["ref"]["document_id"]) for v in entry["signed_variants"]] == [
        (2, "d1"),
        (1, "d3"),
    ]
    assert [(a["n_deals"], len(a["precedent_ids"])) for a in entry["refused_asks"]] == [(2, 2)]
    assert (entry["n_variants_total"], entry["n_refused_total"]) == (2, 1)
    assert pb["digest"] == build_digest_v4(pb)
    result = validate_document(pb)
    assert result.ok, [str(e) for e in result.errors if e.blocking]

    # The validator's count MUST groups the same way: a per-alias count (d1
    # and d2 as two one-deal variants, plus d3 — 3) is rejected.
    stale = json.loads(json.dumps(pb))
    stale["evidence"]["clauses"][0]["n_variants"] = 3
    errors = [str(e) for e in validate_document(stale).errors if e.blocking]
    assert any("n_variants=3 but evidence.precedent implies 2" in e for e in errors), errors


def test_normalize_variant_text_party_tokens() -> None:
    """The grouping key, rule by rule (OPF-SPEC §3.5.4)."""
    from playbook_engine.precedent import normalize_variant_text as key

    # 1. Any Counterparty-<n> alias, any case or zero-padding, is one token.
    assert key("Counterparty-2 shall pay.", party=None) == key(
        "counterparty-007 shall pay", party=None
    )
    assert key("Counterparty-2 shall pay.", party=None) == "counterparty shall pay"
    # ... but only a whole alias: a trailing word character is not one.
    assert key("Counterparty-2a shall pay.", party=None) == "counterparty 2a shall pay"
    # 3. perspective.party, case-insensitively, whitespace-tolerant, on word
    #    boundaries, becomes its own token — never the counterparty's.
    assert key("ALPHACORP   Holdings, Inc. shall pay.", party="AlphaCorp Holdings, Inc.") == (
        "party shall pay"
    )
    assert key("AlphaCorpX shall pay.", party="AlphaCorp") == "alphacorpx shall pay"
    swapped = key("AlphaCorp shall pay Counterparty-4.", party="AlphaCorp")
    assert swapped == "party shall pay counterparty"
    assert swapped != key("Counterparty-4 shall pay AlphaCorp.", party="AlphaCorp")
    # No perspective: no party name is applied.
    assert key("AlphaCorp shall pay.", party=None) == "alphacorp shall pay"
    # Content tokens survive exactly as in the standard check.
    assert key("Counterparty-1 shall not pay.", party=None) != key(
        "Counterparty-1 shall pay.", party=None
    )


def test_v4_digest_refused_asks_group_across_deals_and_conceded_is_not_refused() -> None:
    ask = "Either party may terminate this Agreement at any time without notice."
    observations = [
        _v05_obs("d1", _V05_STANDARD, standard=True),
        _v05_obs("d1", ask, outcome="proposed_then_reversed", version=2),
        _v05_obs("d2", _V05_STANDARD, standard=True),
        _v05_obs("d2", ask.upper(), outcome="proposed_then_reversed", version=3),
        # Our standard struck in d3: a concession and an opening (the two rows
        # observation_builder writes for it) — opening text, never refused.
        _v05_obs("d3", _V05_STANDARD, standard=True, outcome="conceded_before_signing", version=1),
        _v05_obs(
            "d3",
            _V05_STANDARD,
            standard=True,
            outcome="opening",
            version=1,
            opened_with="standard",
        ),
    ]
    pb = _v05_playbook(observations, signed={"d1": True, "d2": True, "d3": True})
    (clause,) = pb["digest"]["clauses"]
    assert [(a["n_deals"], len(a["precedent_ids"])) for a in clause["refused_asks"]] == [(2, 2)]
    # The representative is the earliest-round ask (d1's v2 → round 1).
    assert clause["refused_asks"][0]["ref"]["document_id"] == "d1"
    d3 = next(p for p in pb["evidence"]["precedent"] if p["document_id"] == "d3")
    assert d3["signed_text"] is None and d3["opening_text"]["text"] == _V05_STANDARD
    assert d3["opened_with"] == "standard"
    assert d3["refused_asks"] == [] and d3["standard"] is False and d3["moved"] is True
    assert clause["n_signed_standard"] == 2


def test_v4_digest_carries_perspective_or_null() -> None:
    observations = [_v05_obs("d1", _V05_STANDARD, standard=True)]
    with_p = _v05_playbook(
        observations,
        signed={"d1": True},
        perspective={"party": "Fixture Co", "counterparty_type": "Vendor"},
    )
    without_p = _v05_playbook(observations, signed={"d1": True})
    assert with_p["digest"]["perspective"] == with_p["perspective"]
    assert without_p["digest"]["perspective"] is None
    assert "perspective" in without_p["digest"]


def test_v4_digest_budget_tightens_cap_but_keeps_totals() -> None:
    observations = [
        _v05_obs(f"d{i}", f"This Agreement is governed by the laws of jurisdiction {i}.")
        for i in range(8)
    ]
    pb = _v05_playbook(observations, signed={f"d{i}": True for i in range(8)})
    loose = build_digest_v4(pb, token_budget=None)
    tight = build_digest_v4(pb, token_budget=1)
    assert len(loose["clauses"][0]["signed_variants"]) == EXEMPLAR_TOP_N
    assert len(tight["clauses"][0]["signed_variants"]) == 1
    for d in (loose, tight):
        assert d["clauses"][0]["n_variants_total"] == 8
        assert d["clauses"][0]["n_variants_total"] == pb["evidence"]["clauses"][0]["n_variants"]
    assert digest_token_estimate(tight) < digest_token_estimate(loose)


def test_v4_digest_summaries_never_exceed_300_chars_and_no_judged_fields() -> None:
    long_text = ("This Agreement is governed by the laws of Delaware. " * 12).strip()
    pb = _v05_playbook([_v05_obs("d1", long_text)], signed={"d1": True})
    (variant,) = pb["digest"]["clauses"][0]["signed_variants"]
    assert len(variant["text"]) <= 300 and long_text.startswith(variant["text"])
    serialized = json.dumps(pb["digest"])
    for key in ("full_text", "historical_stance", "band", "risk_delta", "deviation"):
        assert f'"{key}"' not in serialized, key


# ---------------------------------------------------------------------------
# Validator: digest consistency on a 0.5 document
# ---------------------------------------------------------------------------


def _small_v05_playbook() -> dict:
    variant = "This Agreement is governed by the laws of the State of Delaware."
    return _v05_playbook(
        [_v05_obs("d1", _V05_STANDARD, standard=True), _v05_obs("d2", variant)],
        signed={"d1": True, "d2": True},
    )


def test_digest_never_contains_full_text_and_estimates_positive() -> None:
    pb = _small_v05_playbook()
    assert "full_text" not in json.dumps(pb["digest"])
    assert digest_token_estimate(pb["digest"]) > 0


def test_validator_accepts_v05_with_and_without_digest() -> None:
    pb = _small_v05_playbook()
    assert validate_document(pb).ok
    del pb["digest"]
    del pb["identity"]
    result = validate_document(pb)
    assert result.ok, [str(e) for e in result.errors]


def test_validator_rejects_digest_id_mismatch() -> None:
    pb = _small_v05_playbook()
    del pb["identity"]
    pb["digest"]["clauses"][0]["id"] = "clause.some_other_clause"
    result = validate_document(pb)
    assert not result.ok
    assert any("digest.clauses ids do not match" in str(e) for e in result.errors)


def test_validator_rejects_full_text_in_digest() -> None:
    pb = _small_v05_playbook()
    del pb["identity"]
    pb["digest"]["clauses"][0]["x_extra"] = {"full_text": "leaked verbatim clause"}
    result = validate_document(pb)
    assert not result.ok
    assert any("full_text" in str(e) for e in result.errors)


# ---------------------------------------------------------------------------
# digest_version 4 (issue #234): held/conceded counts, variant provenance,
# changed openings, uncovered clause types and the #240 label. The documents
# are built by the real producer (``assemble_playbook`` over Observation rows
# of the shapes observation_builder writes), never hand-edited records.
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parent.parent
_STD = _V05_STANDARD
_V_A = "This Agreement is governed by the laws of the State of Delaware."
_V_B = "This Agreement is governed by the laws of the State of Texas."
_N_1 = "Disputes are governed by the laws of the State of Oregon."
_N_2 = "Disputes are governed by the laws of the State of Nevada."
_N_3 = "Disputes are governed by the laws of the State of Ohio."
_N_4 = "Disputes are governed by the laws of the State of Utah."


def _opening_rows() -> tuple[list, dict[str, bool]]:
    """Nine deals, one per branch of the opening rules (ticket #234)."""
    o = _v05_obs
    rows = [
        # d1: opened with our standard, signed an edit of it (a concession).
        o("d1", _V_A, opened_with="standard"),
        o("d1", _STD, standard=True, outcome="opening", version=1, opened_with="standard"),
        # d2: opened with our standard and it was struck before signing.
        o("d2", _STD, standard=True, outcome="opening", version=1, opened_with="standard"),
        # d3: opened non-standard, ended at our standard.
        o("d3", _STD, standard=True, opened_with="non_standard"),
        o("d3", _N_1, outcome="opening", version=1, opened_with="non_standard"),
        # d4: opened with our standard and kept it (no distinct opening).
        o("d4", _STD, standard=True, opened_with="standard"),
        # d5: opened non-standard, signed exactly as it opened.
        o("d5", _V_B, opened_with="non_standard"),
        # d6: opened non-standard, struck, and the same text is a refused ask.
        o("d6", _N_2, outcome="opening", version=1, opened_with="non_standard"),
        o("d6", _N_2, outcome="proposed_then_reversed", version=1, opened_with="non_standard"),
        # d7: opened non-standard and struck; no refused ask.
        o("d7", _N_3, outcome="opening", version=1, opened_with="non_standard"),
        # d8: clause added in round 2.
        o("d8", _V_A, opened_with="absent"),
        # d9: unsigned deal; its text and any opening are ignored.
        o("d9", _N_4, outcome="unsigned", opened_with=None),
    ]
    signed = {f"d{i}": i != 9 for i in range(1, 10)}
    return rows, signed


def test_v4_opening_counts_and_variant_provenance() -> None:
    rows, signed = _opening_rows()
    pb = _v05_playbook(rows, signed=signed)
    result = validate_document(pb)
    assert result.ok, [str(e) for e in result.errors if e.blocking]
    (clause,) = pb["digest"]["clauses"]
    ev = pb["evidence"]["clauses"][0]
    # d1, d2, d4 opened with our standard; d4 kept it (d1 edited, d2 struck).
    assert (clause["n_opened_standard"], clause["n_kept_standard"]) == (3, 1)
    assert (ev["n_opened_standard"], ev["n_kept_standard"]) == (3, 1)
    by_text = {v["text"]: v for v in clause["signed_variants"]}
    # d1's edit is a concession; d5's text was signed as proposed; d8 added it.
    assert (by_text[_V_A]["n_deals"], by_text[_V_A]["n_from_standard"]) == (2, 1)
    assert by_text[_V_A]["n_unchanged"] == 0
    assert (by_text[_V_B]["n_from_standard"], by_text[_V_B]["n_unchanged"]) == (0, 1)
    # Changed openings: d3 (to standard) and d7 (struck). d6 is a refused ask
    # already; d2 opened with our standard so is not "non-standard language";
    # d9 is unsigned.
    assert {e["text"] for e in clause["changed_openings"]} == {_N_1, _N_3}
    by_open = {e["text"]: e for e in clause["changed_openings"]}
    assert (by_open[_N_1]["n_to_standard"], by_open[_N_1]["n_struck"]) == (1, 0)
    assert (by_open[_N_3]["n_to_standard"], by_open[_N_3]["n_struck"]) == (0, 1)
    assert clause["n_changed_openings_total"] == ev["n_changed_openings"] == 2
    assert [a["text"] for a in clause["refused_asks"]] == [_N_2]
    # positions: d4 and d3 signed our standard; d1/d8 signed V_A, d5 V_B (unjudged).
    assert clause["positions"] == {
        "standard": 2,
        "equivalent": 0,
        "more_protective": 0,
        "less_protective": 0,
        "different_concept": 0,
        "unjudged": 3,
    }
    assert clause["positions"]["standard"] == clause["n_signed_standard"]


def test_v4_changed_openings_group_across_deals_and_cite_the_lowest_document() -> None:
    o = _v05_obs
    rows = [
        o("d2", _N_1.upper(), outcome="opening", version=1, opened_with="non_standard"),
        o("d1", _N_1, outcome="opening", version=1, opened_with="non_standard"),
        o("d3", _STD, standard=True, opened_with="non_standard"),
        o("d3", _N_1, outcome="opening", version=1, opened_with="non_standard"),
    ]
    pb = _v05_playbook(rows, signed={"d1": True, "d2": True, "d3": True})
    assert validate_document(pb).ok
    (entry,) = pb["digest"]["clauses"][0]["changed_openings"]
    assert (entry["n_deals"], entry["n_to_standard"], entry["n_struck"]) == (3, 1, 2)
    assert entry["ref"]["document_id"] == "d1"
    assert len(entry["precedent_ids"]) == 3


def test_v4_budget_caps_changed_openings_and_the_scorecard_counts_the_clause(
    tmp_path: Path,
) -> None:
    """The cap reaches ``changed_openings`` on its own: one clause with three
    distinct changed openings and no signed variant or refused ask shows one
    opening under a budget it cannot meet, keeps the uncapped total, and
    ``playbook scorecard`` counts it as a capped clause."""
    o = _v05_obs
    rows = [
        # d1, d2: opened non-standard, ended at our standard.
        o("d1", _STD, standard=True, opened_with="non_standard"),
        o("d1", _N_1, outcome="opening", version=1, opened_with="non_standard"),
        o("d2", _STD, standard=True, opened_with="non_standard"),
        o("d2", _N_2, outcome="opening", version=1, opened_with="non_standard"),
        # d3: opened non-standard and struck; no refused ask.
        o("d3", _N_3, outcome="opening", version=1, opened_with="non_standard"),
    ]
    pb = _v05_playbook(rows, signed={"d1": True, "d2": True, "d3": True})
    result = validate_document(pb)
    assert result.ok, [str(e) for e in result.errors if e.blocking]
    ev = pb["evidence"]["clauses"][0]
    loose = build_digest_v4(pb, token_budget=None)["clauses"][0]
    tight = build_digest_v4(pb, token_budget=1)["clauses"][0]
    # Only the changed openings outnumber the floor of one entry per list.
    for clause in (loose, tight):
        assert (clause["n_variants_total"], clause["n_refused_total"]) == (0, 0)
        assert clause["signed_variants"] == [] and clause["refused_asks"] == []
        assert clause["n_changed_openings_total"] == ev["n_changed_openings"] == 3
    assert {e["text"] for e in loose["changed_openings"]} == {_N_1, _N_2, _N_3}
    assert len(tight["changed_openings"]) == 1

    out = tmp_path / "out"
    out.mkdir()
    path = out / "playbook.opf.json"

    def _scorecard_digest(doc: dict) -> dict:
        path.write_text(json.dumps(doc), encoding="utf-8")
        run = CliRunner().invoke(cli, ["scorecard", str(out)])
        assert run.exit_code == 0, run.output
        card = json.loads((out / "scorecard.json").read_text(encoding="utf-8"))
        return card["digest"]

    # The compiled digest fits the default budget, so nothing is capped.
    assert len(pb["digest"]["clauses"][0]["changed_openings"]) == 3
    assert _scorecard_digest(pb)["capped_clauses"] == 0
    capped = _scorecard_digest({**pb, "digest": build_digest_v4(pb, token_budget=1)})
    assert capped["capped_clauses"] == 1
    assert capped["changed_openings"] == 3


_LABEL_TEXTS = {
    "eq1": "This Agreement is governed by the laws of the State of Delaware.",
    "eq2": "Delaware law governs this Agreement.",
    "less": "This Agreement is governed by the laws of Mars.",
    "diff": "This Agreement is governed by the laws of the State of Texas.",
    "more": "This Agreement is governed by the laws of the State of New York, "
    "and each party waives any objection to that forum.",
    "none": "The laws of Utah govern this Agreement.",
}

_LABELS = {
    "eq1": "equivalent",
    "eq2": "equivalent",
    "less": "less_protective",
    "diff": "different_concept",
    "more": "more_protective",
}


def _labelled_playbook(tmp_path: Path, labels: dict[str, str]) -> dict:
    """A playbook whose texts carry ``vs_standard`` labels from a verdict store.

    Production path: ``assemble_playbook`` with the store-backed equivalence
    judge, which reads verdicts the way ``playbook judge-apply`` banks them.
    """
    from playbook_engine.agent_judge import StoreBackedEquivalenceJudge, VerdictStore
    from playbook_engine.clause_position_compiler import compile_clause_positions
    from playbook_engine.equivalence import equivalence_key
    from playbook_engine.observation_builder import Observation, ObservationCitation
    from playbook_engine.playbook_assembler import assemble_playbook

    o = _v05_obs
    rows = [o("s1", _STD, standard=True, opened_with="standard")]
    deals = {"s1": True}
    n = 2
    for name, count in (("eq1", 2), ("eq2", 1), ("less", 2), ("diff", 1), ("more", 1), ("none", 1)):
        for _ in range(count):
            doc = f"s{n}"
            n += 1
            # s3 is the deal whose clause opened non-standard (its opening row
            # is added below); every row of a deal carries the same value.
            opened_with = "standard" if name == "eq2" else "non_standard" if doc == "s3" else None
            rows.append(o(doc, _LABEL_TEXTS[name], opened_with=opened_with))
            if name == "eq2":
                # Opened with our standard and signed different text: the
                # producer always emits the opening row for that shape.
                rows.append(
                    o(
                        doc,
                        _STD,
                        standard=True,
                        outcome="opening",
                        version=1,
                        opened_with="standard",
                    )
                )
            deals[doc] = True
    # A refused ask and a changed opening, both labelled.
    rows.append(o("s1", _LABEL_TEXTS["less"], outcome="proposed_then_reversed", version=2))
    rows.append(
        o("s3", _LABEL_TEXTS["diff"], outcome="opening", version=1, opened_with="non_standard")
    )

    template = Observation(
        observation_id="template/governing_law",
        taxonomy_id="governing_law",
        text_summary=_STD,
        citation=ObservationCitation(
            document_id="template", version="template", clause_path="4", char_span=None
        ),
        deviation="none",
        risk_delta={"direction": "neutral", "magnitude": "none"},
        provenance="our_paper",
        outcome="signed",
    )
    positions, _, _ = compile_clause_positions(rows, [template])
    store = VerdictStore(tmp_path / "judge" / "verdicts.jsonl")
    for name, label in labels.items():
        key = equivalence_key("nda", "governing_law", None, _LABEL_TEXTS[name], _STD)
        store.put_by_key(key, {"label": label, "reason": "Test reason.", "basis": "agent"})
    documents = [
        {
            "document_id": d,
            "provenance": "our_paper",
            "in_scope": True,
            "versions": 3,
            "signed_version": 3,
        }
        for d in sorted(deals)
    ]
    pb = assemble_playbook(
        agreement_type={"id": "nda", "name": "Mutual NDA"},
        baseline={"has_canonical_template": True},
        taxonomy={"source": "custom", "entries": []},
        clause_positions=positions,
        corpus_documents=documents,
        generated_at="2026-01-01T00:00:00Z",
        observations=rows,
        equivalence_judge=StoreBackedEquivalenceJudge(store=store),
    )
    assert validate_document(pb, verdict_store=store).ok
    return pb


def test_v4_equivalent_variants_collapse_and_worse_ones_lead(tmp_path: Path) -> None:
    pb = _labelled_playbook(tmp_path, _LABELS)
    (clause,) = pb["digest"]["clauses"]
    listed = clause["signed_variants"]
    assert [v["label"] for v in listed] == [
        "less_protective",  # 2 deals
        "different_concept",  # 1 deal
        None,  # unjudged, after the worse ones
        "more_protective",
        "equivalent",  # the one collapsed entry, last
    ]
    collapsed = listed[-1]
    assert (collapsed["n_deals"], collapsed["n_texts"]) == (3, 2)
    assert collapsed["n_from_standard"] == 1  # the eq2 deal opened with our standard
    assert len(collapsed["exemplars"]) == 2
    assert {"text", "ref"} == set(collapsed["exemplars"][0])
    assert len(collapsed["precedent_ids"]) == 3
    assert "text" not in collapsed
    # Totals count distinct texts, not entries: 4 individual + 2 equivalent.
    assert clause["n_variants_total"] == 6
    assert clause["positions"] == {
        "standard": 1,
        "equivalent": 3,
        "more_protective": 1,
        "less_protective": 2,
        "different_concept": 1,
        "unjudged": 1,
    }
    # Labels travel on the refused ask and the changed opening too.
    assert [a["label"] for a in clause["refused_asks"]] == ["less_protective"]
    assert [e["label"] for e in clause["changed_openings"]] == ["different_concept"]


def test_v4_the_cap_applies_after_collapsing(tmp_path: Path) -> None:
    pb = _labelled_playbook(tmp_path, _LABELS)
    tight = build_digest_v4(pb, token_budget=1)["clauses"][0]
    assert [v["label"] for v in tight["signed_variants"]] == ["less_protective"]
    assert tight["n_variants_total"] == 6
    loose = build_digest_v4(pb, token_budget=None)["clauses"][0]
    assert [v["label"] for v in loose["signed_variants"]][-1] == "equivalent"
    # An unjudged playbook collapses nothing and orders as digest 3 did.
    unjudged = _labelled_playbook(tmp_path / "none", {})
    listed = unjudged["digest"]["clauses"][0]["signed_variants"]
    assert all("exemplars" not in v for v in listed)
    assert [v["n_deals"] for v in listed] == [2, 2, 1, 1, 1]  # top 5 of 6 distinct texts
    assert unjudged["digest"]["clauses"][0]["n_variants_total"] == 6


def test_v4_uncovered_clause_types_are_the_eligible_taxonomy_without_evidence() -> None:
    from playbook_engine.digest import uncovered_clause_types
    from playbook_engine.taxonomy import load_taxonomy

    pb = _small_v05_playbook()
    entries = [
        {"id": "governing_law", "label": "Governing Law", "status": "active"},
        {"id": "zeta", "label": "Zeta", "status": "custom"},
        {"id": "alpha", "label": "Alpha", "status": "active"},
        {"id": "retired", "label": "Retired", "status": "inactive"},
    ]
    pb["taxonomy"] = {"source": "custom", "entries": entries}
    assert uncovered_clause_types(pb) == [
        {"taxonomy_id": "alpha", "label": "Alpha"},
        {"taxonomy_id": "zeta", "label": "Zeta"},
    ]
    assert build_digest_v4(pb)["uncovered_clause_types"] == uncovered_clause_types(pb)

    # The NDA example: exactly the eligible entries of spec/taxonomy/nda.yaml
    # that no evidence clause covers.
    nda = json.loads((_ROOT / "examples" / "nda" / "playbook.opf.json").read_text())
    covered = {c["taxonomy_id"] for c in nda["evidence"]["clauses"]}
    expected = sorted(
        e.id
        for e in load_taxonomy(_ROOT / "spec" / "taxonomy" / "nda.yaml").entries
        if e.is_classifier_eligible and e.id not in covered
    )
    assert expected, "the NDA example is expected to leave some clause types uncovered"
    assert [u["taxonomy_id"] for u in nda["digest"]["uncovered_clause_types"]] == expected


def test_v4_nda_example_acceptance() -> None:
    """The ticket's acceptance numbers, read off the committed worked example."""
    nda = json.loads((_ROOT / "examples" / "nda" / "playbook.opf.json").read_text())
    assert nda["digest"]["digest_version"] == "4"
    assert nda["digest"] == build_digest_v4(nda)
    clauses = {c["taxonomy_id"]: c for c in nda["digest"]["clauses"]}

    def variant(clause: str, needle: str) -> dict:
        (found,) = [v for v in clauses[clause]["signed_variants"] if needle in v.get("text", "")]
        return found

    gl = clauses["governing_law"]
    assert (gl["n_opened_standard"], gl["n_kept_standard"]) == (4, 2)
    new_york = variant("governing_law", "New York")
    assert (new_york["n_deals"], new_york["n_from_standard"]) == (2, 2)
    california = variant("governing_law", "California")
    assert (california["n_deals"], california["n_unchanged"], california["n_from_standard"]) == (
        2,
        2,
        0,
    )
    assert variant("dispute_resolution_venue", "New York")["n_from_standard"] == 2

    lol = clauses["limitation_of_liability"]
    (cap,) = lol["changed_openings"]
    assert "$50,000" in cap["text"] and cap["n_deals"] == 3 and cap["n_struck"] == 1
    evidence = {c["taxonomy_id"]: c for c in nda["evidence"]["clauses"]}
    for tid, clause in clauses.items():
        assert clause["n_changed_openings_total"] == evidence[tid]["n_changed_openings"], tid

    (opening,) = clauses["compelled_disclosure"]["changed_openings"]
    assert opening["ref"]["document_id"] == "epsilon-systems" and opening["n_to_standard"] == 1


def test_v4_validator_rejects_a_digest_that_disagrees_with_the_reference() -> None:
    rows, signed = _opening_rows()
    base = _v05_playbook(rows, signed=signed)
    del base["identity"]
    for edit in (
        lambda c: c["signed_variants"][0].__setitem__("n_from_standard", 5),
        lambda c: c["changed_openings"][0].__setitem__("n_struck", 9),
        lambda c: c.__setitem__("n_kept_standard", 3),
        lambda c: c["positions"].__setitem__("equivalent", 1),
        lambda c: c.__setitem__("changed_openings", []),
    ):
        tampered = json.loads(json.dumps(base))
        edit(tampered["digest"]["clauses"][0])
        errors = [str(e) for e in validate_document(tampered).errors if e.blocking]
        assert any("digest does not equal build_digest_v4" in e for e in errors), errors
    tampered = json.loads(json.dumps(base))
    tampered["digest"]["uncovered_clause_types"].append({"taxonomy_id": "x", "label": "X"})
    errors = [str(e) for e in validate_document(tampered).errors if e.blocking]
    assert any("digest does not equal build_digest_v4" in e for e in errors), errors


def test_v4_validator_recomputes_the_new_clause_counts() -> None:
    rows, signed = _opening_rows()
    base = _v05_playbook(rows, signed=signed)
    del base["identity"]
    for key in ("n_opened_standard", "n_kept_standard", "n_changed_openings"):
        stale = json.loads(json.dumps(base))
        stale["evidence"]["clauses"][0][key] += 1
        errors = [str(e) for e in validate_document(stale).errors if e.blocking]
        assert any(f"{key}=" in e and "evidence.precedent implies" in e for e in errors), (
            key,
            errors,
        )


def test_v4_view_bundle_states_the_new_facts_in_plain_words(tmp_path: Path) -> None:
    """Issue #234 (page rebuilt in #241): the Playbook and Evidence tabs of
    index.html say how many deals opened with our standard and kept it, how
    each variant came to be signed, which openings were not signed as proposed
    and which clause types have no evidence — and carry no judged label (the
    labels live in the digest JSON and the Review tab)."""
    import re
    import shutil

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    shutil.copy(_ROOT / "examples" / "nda" / "playbook.opf.json", out_dir / "playbook.opf.json")
    result = CliRunner().invoke(cli, ["view", "bundle", str(out_dir)])
    assert result.exit_code == 0, result.output
    html = (out_dir / "index.html").read_text(encoding="utf-8")

    def panel(tab: str) -> str:
        m = re.search(
            rf'<div role="tabpanel" id="tab-{tab}".*?(?=<div role="tabpanel"|<footer)', html, re.S
        )
        assert m, tab
        return m.group(0)

    playbook_tab, evidence_tab = panel("playbook"), panel("evidence")
    assert "opened with our standard in 4; kept it in 2" in playbook_tab
    assert "from our standard in 2" in evidence_tab
    assert "signed as proposed in 2" in evidence_tab
    assert "Openings not signed as proposed" in evidence_tab
    assert "struck in 1" in evidence_tab
    assert "No evidence in this corpus for: " in playbook_tab
    assert "No evidence for" in evidence_tab
    assert "Trade Secret Carve-Out" in playbook_tab
    for tab_html in (playbook_tab, evidence_tab):
        assert "judged equivalent to our standard" not in tab_html
        assert "protective than our standard" not in tab_html
