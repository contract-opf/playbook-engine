"""Tests for the digest section (digest_version 3), `playbook digest`, and `view bundle`.

The digest_version 2 builder was retired with OPF 0.3 (issue #238).

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
    DIGEST_VERSION,
    EXEMPLAR_TOP_N,
    build_digest,
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


def test_compiled_playbook_is_v04_with_digest_v3(tmp_path: Path) -> None:
    """Issue #223: the reference compiler emits OPF 0.4 with a digest_version 3
    digest equal to build_digest over the shipped document."""
    out_dir = _compiled_out_dir(tmp_path)
    pb = json.loads((out_dir / "playbook.opf.json").read_text())
    assert pb["opf_version"] == "0.4"
    assert pb["digest"]["digest_version"] == DIGEST_VERSION == "3"
    assert len(pb["digest"]["clauses"]) == len(pb["evidence"]["clauses"])
    assert "clause_count" not in pb["digest"]
    assert pb["digest"] == build_digest(pb)
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
    assert not (out_dir / "playbook.opf.html").exists()


def test_view_bundle_embeds_canonical_json_and_digest(tmp_path: Path) -> None:
    out_dir = _compiled_out_dir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(cli, ["view", "bundle", str(out_dir)])
    assert result.exit_code == 0, result.output

    html = (out_dir / "playbook.opf.html").read_text(encoding="utf-8")
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
    html = (out_dir / "playbook.opf.html").read_text(encoding="utf-8")
    start = html.index('<script id="opf-canonical" type="application/json">')
    end = html.index("</script>", start)
    block = html[start:end]
    assert "</script" not in block[1:], "unescaped </script> inside the JSON block"


# ---------------------------------------------------------------------------
# digest_version 3 (OPF 0.4, issue #223). Every document here is built by
# the real producer — Observation rows (the shapes observation_builder
# writes: signed / unsigned / proposed_then_reversed /
# conceded_before_signing, with the deterministic `standard` fact) through
# compile_clause_positions + assemble_playbook — never a hand-built
# precedent record.
# ---------------------------------------------------------------------------


def _v04_obs(
    doc_id: str,
    text: str,
    *,
    tid: str = "governing_law",
    outcome: str = "signed",
    standard: bool = False,
    version: int = 3,
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
    )


_V04_STANDARD = "This Agreement is governed by the laws of the State of New York."


def _v04_playbook(observations: list, *, signed: dict[str, bool], perspective=None) -> dict:
    from playbook_engine.clause_position_compiler import compile_clause_positions
    from playbook_engine.observation_builder import Observation, ObservationCitation
    from playbook_engine.playbook_assembler import assemble_playbook

    template = Observation(
        observation_id="template/governing_law",
        taxonomy_id="governing_law",
        text_summary=_V04_STANDARD,
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


def test_v3_digest_groups_variants_by_exact_normalization() -> None:
    """Case/punctuation respellings merge; a negator never does; our standard
    text and an unsigned deal's text are never signed variants."""
    variant = "This Agreement is governed by the laws of the State of Delaware."
    observations = [
        _v04_obs("d1", _V04_STANDARD, standard=True),
        _v04_obs("d2", variant),
        _v04_obs("d3", "THIS AGREEMENT is governed by the laws of the State of Delaware"),
        _v04_obs("d4", "This Agreement is not governed by the laws of the State of Delaware."),
        _v04_obs("d5", "This Agreement is governed by the laws of Texas.", outcome="unsigned"),
    ]
    pb = _v04_playbook(
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
    # The digest is exactly what build_digest recomputes over the document.
    assert pb["digest"] == build_digest(pb)


def test_v3_grouping_neutralizes_counterparty_aliases_and_perspective_party() -> None:
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
        aliased(_v04_obs("d1", signed.format("Counterparty-2")), "Counterparty-2"),
        aliased(
            _v04_obs(
                "d1", ask.format("Counterparty-2"), outcome="proposed_then_reversed", version=2
            ),
            "Counterparty-2",
        ),
        aliased(_v04_obs("d2", signed.format("Counterparty-5")), "Counterparty-5"),
        aliased(
            _v04_obs(
                "d2", ask.format("COUNTERPARTY-5"), outcome="proposed_then_reversed", version=2
            ),
            "Counterparty-5",
        ),
        aliased(
            _v04_obs(
                "d3",
                "AlphaCorp and Counterparty-3 agree that the laws of Delaware govern "
                "this Agreement.",
            ),
            "Counterparty-3",
        ),
    ]
    pb = _v04_playbook(
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
    assert pb["digest"] == build_digest(pb)
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


def test_v3_digest_refused_asks_group_across_deals_and_conceded_is_not_refused() -> None:
    ask = "Either party may terminate this Agreement at any time without notice."
    observations = [
        _v04_obs("d1", _V04_STANDARD, standard=True),
        _v04_obs("d1", ask, outcome="proposed_then_reversed", version=2),
        _v04_obs("d2", _V04_STANDARD, standard=True),
        _v04_obs("d2", ask.upper(), outcome="proposed_then_reversed", version=3),
        # Our standard struck in d3: a concession — opening text, never refused.
        _v04_obs("d3", _V04_STANDARD, outcome="conceded_before_signing", version=1),
    ]
    pb = _v04_playbook(observations, signed={"d1": True, "d2": True, "d3": True})
    (clause,) = pb["digest"]["clauses"]
    assert [(a["n_deals"], len(a["precedent_ids"])) for a in clause["refused_asks"]] == [(2, 2)]
    # The representative is the earliest-round ask (d1's v2 → round 1).
    assert clause["refused_asks"][0]["ref"]["document_id"] == "d1"
    d3 = next(p for p in pb["evidence"]["precedent"] if p["document_id"] == "d3")
    assert d3["signed_text"] is None and d3["opening_text"]["text"] == _V04_STANDARD
    assert d3["refused_asks"] == [] and d3["standard"] is False and d3["moved"] is True
    assert clause["n_signed_standard"] == 2


def test_v3_digest_carries_perspective_or_null() -> None:
    observations = [_v04_obs("d1", _V04_STANDARD, standard=True)]
    with_p = _v04_playbook(
        observations,
        signed={"d1": True},
        perspective={"party": "Fixture Co", "counterparty_type": "Vendor"},
    )
    without_p = _v04_playbook(observations, signed={"d1": True})
    assert with_p["digest"]["perspective"] == with_p["perspective"]
    assert without_p["digest"]["perspective"] is None
    assert "perspective" in without_p["digest"]


def test_v3_digest_budget_tightens_cap_but_keeps_totals() -> None:
    observations = [
        _v04_obs(f"d{i}", f"This Agreement is governed by the laws of jurisdiction {i}.")
        for i in range(8)
    ]
    pb = _v04_playbook(observations, signed={f"d{i}": True for i in range(8)})
    loose = build_digest(pb, token_budget=None)
    tight = build_digest(pb, token_budget=1)
    assert len(loose["clauses"][0]["signed_variants"]) == EXEMPLAR_TOP_N
    assert len(tight["clauses"][0]["signed_variants"]) == 1
    for d in (loose, tight):
        assert d["clauses"][0]["n_variants_total"] == 8
        assert d["clauses"][0]["n_variants_total"] == pb["evidence"]["clauses"][0]["n_variants"]
    assert digest_token_estimate(tight) < digest_token_estimate(loose)


def test_v3_digest_summaries_never_exceed_300_chars_and_no_judged_fields() -> None:
    long_text = ("This Agreement is governed by the laws of Delaware. " * 12).strip()
    pb = _v04_playbook([_v04_obs("d1", long_text)], signed={"d1": True})
    (variant,) = pb["digest"]["clauses"][0]["signed_variants"]
    assert len(variant["text"]) <= 300 and long_text.startswith(variant["text"])
    serialized = json.dumps(pb["digest"])
    for key in ("full_text", "historical_stance", "band", "risk_delta", "deviation"):
        assert f'"{key}"' not in serialized, key


# ---------------------------------------------------------------------------
# Validator: digest consistency on a 0.4 document
# ---------------------------------------------------------------------------


def _small_v04_playbook() -> dict:
    variant = "This Agreement is governed by the laws of the State of Delaware."
    return _v04_playbook(
        [_v04_obs("d1", _V04_STANDARD, standard=True), _v04_obs("d2", variant)],
        signed={"d1": True, "d2": True},
    )


def test_digest_never_contains_full_text_and_estimates_positive() -> None:
    pb = _small_v04_playbook()
    assert "full_text" not in json.dumps(pb["digest"])
    assert digest_token_estimate(pb["digest"]) > 0


def test_validator_accepts_v04_with_and_without_digest() -> None:
    pb = _small_v04_playbook()
    assert validate_document(pb).ok
    del pb["digest"]
    del pb["identity"]
    result = validate_document(pb)
    assert result.ok, [str(e) for e in result.errors]


def test_validator_rejects_digest_id_mismatch() -> None:
    pb = _small_v04_playbook()
    del pb["identity"]
    pb["digest"]["clauses"][0]["id"] = "clause.some_other_clause"
    result = validate_document(pb)
    assert not result.ok
    assert any("digest.clauses ids do not match" in str(e) for e in result.errors)


def test_validator_rejects_full_text_in_digest() -> None:
    pb = _small_v04_playbook()
    del pb["identity"]
    pb["digest"]["clauses"][0]["x_extra"] = {"full_text": "leaked verbatim clause"}
    result = validate_document(pb)
    assert not result.ok
    assert any("full_text" in str(e) for e in result.errors)
