"""Tests for the OPF 0.3 digest section, `playbook digest`, and `view bundle`.

SECURITY NOTE: All fixtures are programmatically constructed or drawn from
the synthetic examples/ fixtures. No real agreements are referenced.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from playbook_engine.canonicalize import canonicalize, content_hash
from playbook_engine.cli import cli
from playbook_engine.digest import (
    DIGEST_VERSION,
    DIGEST_VERSION_V2,
    EXEMPLAR_TOP_N,
    _exemplar_forms,
    build_digest,
    digest_token_estimate,
)
from playbook_engine.validator import validate_document

_FIXTURES = Path(__file__).parent.parent / "examples" / "fixtures"


def _load_fixture(name: str) -> dict:
    return json.loads((_FIXTURES / name).read_text(encoding="utf-8"))


def _obs(
    text: str,
    *,
    full_text: str | None = None,
    n: int = 1,
    magnitude: str = "none",
    direction: str = "neutral",
    deviation: str = "none",
    doc_id: str = "deal-001",
) -> dict:
    return {
        "text_summary": text,
        "full_text": full_text or text,
        "example_ref": {"document_id": doc_id, "version": 1, "clause_path": "1"},
        "deviation": deviation,
        "risk_delta": {"direction": direction, "magnitude": magnitude},
        "provenance": "our_paper",
        "outcome": "signed",
        "precedent_count": n,
    }


def _deals(text: str, k: int, start: int = 0, **kw: Any) -> list[dict]:
    """*k* observed positions of *text*, one per distinct deal (deal-<start>
    onward) — the shape the compiler emits after issue #216: one row per
    (deal, clause), every row of a text stamped with that text's
    distinct-deal ``precedent_count``."""
    return [_obs(text, n=k, doc_id=f"deal-{j:03d}", **kw) for j in range(start, start + k)]


# ---------------------------------------------------------------------------
# build_digest
# ---------------------------------------------------------------------------


def test_build_digest_from_v02_fixture() -> None:
    doc = _load_fixture("valid_v0_2_minimal.json")
    digest = build_digest(doc)
    # A pre-0.4 document always gets the frozen digest_version 2 shape.
    assert digest["digest_version"] == DIGEST_VERSION_V2
    assert digest["clause_count"] == len(doc["evidence"]["clauses"])
    entry = digest["clauses"][0]
    src = doc["evidence"]["clauses"][0]
    assert entry["id"] == src["id"]
    assert entry["taxonomy_id"] == src["taxonomy_id"]
    assert entry["historical_stance"] == src["summary"]["historical_stance"]
    # surviving acceptable_if entries carry if/to VERBATIM (no rationale in
    # the digest projection) plus n/band
    src_pv = src["summary"].get("acceptable_if", [])
    out_pv = entry["preferred_variations"]
    assert len(out_pv) == len(src_pv)  # tiny fixture — nothing capped away
    for out, orig in zip(out_pv, src_pv, strict=True):
        if isinstance(orig, str):
            assert out == orig
        else:
            assert out["if"] == orig["if"]
            assert out["to"] == orig["to"]
            assert out["observation_ref"] == orig["observation_ref"]
            assert "rationale" not in out
            assert out["n"] >= 1 and out["band"] in ("often", "sometimes", "rare")


def test_digest_never_contains_full_text() -> None:
    doc = _load_fixture("valid_v0_2_minimal.json")
    digest = build_digest(doc)

    def walk(value: object) -> None:
        if isinstance(value, dict):
            assert "full_text" not in value
            for v in value.values():
                walk(v)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(digest)


def test_digest_token_estimate_positive() -> None:
    doc = _load_fixture("valid_v0_2_minimal.json")
    assert digest_token_estimate(build_digest(doc)) > 0


# ---------------------------------------------------------------------------
# _exemplar_forms: dedupe, n-weighting, bands, top-N + material
# ---------------------------------------------------------------------------


def test_exemplar_dedupe_by_normalized_text_counts_deals() -> None:
    obs = _deals("Indemnification survives termination.", 7)
    # Same text modulo case/punct/whitespace, in 5 other deals — must merge
    # into one form whose n is all 12 deals.
    obs += _deals("indemnification  survives termination", 5, start=7)
    forms = _exemplar_forms(obs)
    assert len(forms) == 1
    assert forms[0]["n"] == 12
    assert forms[0]["band"] == "often"


def test_exemplar_n_is_deal_count_not_rows_times_precedent_count() -> None:
    """Issue #216 regression: a text signed in k deals arrives as k rows,
    each stamped precedent_count=k; summing per row reported n=k*k (the
    NDA example's assignment clause showed n=16 on a 4-deal group)."""
    forms = _exemplar_forms(_deals("Neither party may assign this Agreement.", 4))
    assert [(f["n"], f["band"]) for f in forms] == [(4, "sometimes")]


def test_exemplar_merged_texts_count_each_deal_once() -> None:
    """Two texts the compiler counts separately (they differ in punctuation)
    but the digest merges: n is the distinct deals across both, so a deal
    carrying both texts counts once (never 3 + 2 summed precedent_counts)."""
    forms = _exemplar_forms(_deals("Term: two (2) years.", 3) + _deals("Term two 2 years", 2))
    # deal-000..002 and deal-000..001 — 3 distinct deals.
    assert [f["n"] for f in forms] == [3]


def test_row_without_document_id_raises_never_counted_as_a_guessed_deal() -> None:
    """Every playbook schema requires example_ref.document_id, so a row
    without one is non-conforming: the digest raises (spec/CHANGELOG.md,
    2026-09-25 entry) rather than count it as a deal of its own."""
    row = _obs("Neither party may assign this Agreement.")
    del row["example_ref"]["document_id"]
    doc = {
        "opf_version": "0.2",
        "evidence": {
            "clauses": [
                {
                    "id": "clause.assignment",
                    "taxonomy_id": "assignment",
                    "title": "Assignment",
                    "observed_positions": [row],
                    "summary": {
                        "historical_stance": "mixed",
                        "acceptable_if": [],
                        "fallbacks": [],
                        "rejected": [],
                        "confidence": {"score": 0.5},
                    },
                }
            ],
            "clause_library": [],
        },
    }
    with pytest.raises(ValueError, match="no example_ref.document_id"):
        build_digest(doc)


def test_preferred_variation_n_taken_once_per_text() -> None:
    """acceptable_if entries whose `to` text is the same (whitespace/case)
    count that text's deals once, not once per entry."""
    text = "Either party may disclose to its professional advisers."
    positions = _deals(text, 3, deviation="substantive")

    def _entry(to: str, doc: str) -> dict:
        return {
            "if": text[:30],
            "to": to,
            "rationale": "engine narration",
            "observation_ref": {"document_id": doc, "version": 1, "clause_path": "1"},
        }

    doc = {
        "opf_version": "0.2",
        "evidence": {
            "clauses": [
                {
                    "id": "clause.x",
                    "taxonomy_id": "x",
                    "title": "X",
                    "observed_positions": positions,
                    "summary": {
                        "historical_stance": "mixed",
                        "acceptable_if": [
                            _entry(text, "deal-000"),
                            _entry(text.upper(), "deal-001"),
                        ],
                        "fallbacks": [],
                        "rejected": [],
                        "confidence": {"score": 0.5},
                    },
                }
            ],
            "clause_library": [],
        },
    }
    pv = build_digest(doc)["clauses"][0]["preferred_variations"]
    assert [(p["n"], p["band"]) for p in pv] == [(3, "sometimes")]


def test_preferred_variation_n_counts_only_signed_deals() -> None:
    """Issue #216: a deal where the accepted `to` text was
    proposed_then_reversed REFUSED it — it is unacceptable precedent, never
    acceptance precedent. The text signed in one deal and reversed in two
    others is preferred n=1 and unacceptable n=2."""
    text = "Recipient may retain one archival copy of Confidential Information."
    signed = _obs(text, deviation="substantive", doc_id="deal-000")
    reversed_rows = [
        {**_obs(text, doc_id=f"deal-{j:03d}"), "outcome": "proposed_then_reversed"} for j in (1, 2)
    ]
    doc = {
        "opf_version": "0.2",
        "evidence": {
            "clauses": [
                {
                    "id": "clause.x",
                    "taxonomy_id": "x",
                    "title": "X",
                    "observed_positions": [signed, *reversed_rows],
                    "summary": {
                        "historical_stance": "mixed",
                        "acceptable_if": [
                            {
                                "if": text[:30],
                                "to": text,
                                "rationale": "engine narration",
                                "observation_ref": signed["example_ref"],
                            }
                        ],
                        "fallbacks": [],
                        "rejected": reversed_rows,
                        "confidence": {"score": 0.5},
                    },
                }
            ],
            "clause_library": [],
        },
    }
    clause = build_digest(doc)["clauses"][0]
    assert [(p["n"], p["band"]) for p in clause["preferred_variations"]] == [(1, "rare")]
    assert [u["n"] for u in clause["unacceptable"]] == [2]


def test_exemplar_bands() -> None:
    forms = _exemplar_forms(
        _deals("alpha clause text", 10) + _deals("beta clause text", 2) + _deals("gamma clause", 1)
    )
    by_text = {f["text_summary"]: f["band"] for f in forms}
    assert by_text["alpha clause text"] == "often"
    assert by_text["beta clause text"] == "sometimes"
    assert by_text["gamma clause"] == "rare"


def test_exemplar_top_n_plus_material() -> None:
    obs = [
        o for i in range(EXEMPLAR_TOP_N) for o in _deals(f"common form variant number {i}", 20 - i)
    ]
    obs.append(_obs("rare but material risk form", n=1, magnitude="material", direction="worse"))
    obs.append(_obs("rare and boring form", n=1))
    forms = _exemplar_forms(obs)
    texts = [f["text_summary"] for f in forms]
    assert len(forms) == EXEMPLAR_TOP_N + 1
    assert "rare but material risk form" in texts
    assert "rare and boring form" not in texts


def test_exemplar_forms_carry_example_ref_and_deviation() -> None:
    forms = _exemplar_forms([_obs("some clause text", deviation="substantive")])
    assert forms[0]["example_ref"]["document_id"] == "deal-001"
    assert forms[0]["deviation"] == "substantive"


def test_preferred_variations_deduped_capped_and_budgeted() -> None:
    """acceptable_if gets the same dedupe/top-N+material discipline; the
    token budget tightens caps when the digest would blow it."""

    def _acc(i: int, text_len: int = 400) -> dict:
        body = f"variation {i} " + ("lorem ipsum dolor sit amet " * (text_len // 27))
        return {
            "if": body,
            "to": body + " revised",
            "rationale": "engine narration",
            "observation_ref": {"document_id": "deal-001", "version": 1, "clause_path": str(i)},
        }

    clauses = []
    for c in range(6):
        clauses.append(
            {
                "id": f"clause.c{c}",
                "taxonomy_id": f"c{c}",
                "title": f"C{c}",
                "observed_positions": [],
                "summary": {
                    "historical_stance": "mixed",
                    "acceptable_if": [_acc(i) for i in range(12)],
                    "fallbacks": [],
                    "rejected": [],
                    "confidence": {"score": 0.5},
                },
            }
        )
    doc = {"opf_version": "0.2", "evidence": {"clauses": clauses, "clause_library": []}}

    # No budget: loosest cap applies (top-5; nothing material here).
    loose = build_digest(doc, token_budget=None)
    assert all(len(c["preferred_variations"]) == EXEMPLAR_TOP_N for c in loose["clauses"])
    for pv in loose["clauses"][0]["preferred_variations"]:
        assert "rationale" not in pv and pv["n"] == 1 and pv["band"] == "rare"

    # A tight budget forces the caps down to the floor of 3.
    tight = build_digest(doc, token_budget=1)
    assert all(len(c["preferred_variations"]) == 3 for c in tight["clauses"])


def test_rejected_and_fallbacks_deduped_and_capped() -> None:
    """concessions/unacceptable get the same dedupe/top-N+material discipline
    as exemplar forms — a raw rejected list of hundreds of near-duplicates
    must not flood the digest."""
    rejected = [_obs(f"rejected ask variant {i}", n=1) for i in range(1, 20)]
    rejected += _deals("rejected ask variant 0", 5)  # the same ask refused in 5 deals
    rejected += [_obs("rare but material ask", n=1, magnitude="material", direction="worse")]
    doc = {
        "opf_version": "0.2",
        "evidence": {
            "clauses": [
                {
                    "id": "clause.x",
                    "taxonomy_id": "x",
                    "title": "X",
                    "observed_positions": [],
                    "summary": {
                        "historical_stance": "mixed",
                        "acceptable_if": [],
                        "fallbacks": rejected[:3],
                        "rejected": rejected,
                        "confidence": {"score": 0.5},
                    },
                }
            ],
            "clause_library": [],
        },
    }
    digest = build_digest(doc)
    entry = digest["clauses"][0]
    unacceptable = entry["unacceptable"]
    assert len(unacceptable) == EXEMPLAR_TOP_N + 1  # top-5 + the material one
    texts = [u["text_summary"] for u in unacceptable]
    assert "rare but material ask" in texts
    # the 5 rows of variant 0 merge into one entry whose n is its 5 deals
    # (never 5 rows x precedent_count 5 = 25) and it ranks first
    top = unacceptable[0]
    assert top["text_summary"] == "rejected ask variant 0"
    assert top["n"] == 5
    assert top["band"] == "sometimes"
    assert all("n" in u and "band" in u for u in unacceptable)
    assert all("deviation" not in u for u in unacceptable)
    assert len(entry["concessions"]) <= EXEMPLAR_TOP_N + 1


# ---------------------------------------------------------------------------
# Validator: 0.3 acceptance + digest consistency
# ---------------------------------------------------------------------------


def _as_v03(doc: dict) -> dict:
    doc = json.loads(json.dumps(doc))  # deep copy
    doc["opf_version"] = "0.3"
    doc["digest"] = build_digest(doc)
    return doc


def test_validator_accepts_v03_with_digest() -> None:
    doc = _as_v03(_load_fixture("valid_v0_2_minimal.json"))
    result = validate_document(doc)
    assert result.ok, [str(e) for e in result.errors]


def test_validator_accepts_v03_without_digest() -> None:
    doc = _as_v03(_load_fixture("valid_v0_2_minimal.json"))
    del doc["digest"]
    result = validate_document(doc)
    assert result.ok, [str(e) for e in result.errors]


def test_validator_still_accepts_v02() -> None:
    doc = _load_fixture("valid_v0_2_minimal.json")
    result = validate_document(doc)
    assert result.ok, [str(e) for e in result.errors]


def test_validator_rejects_v03_digest_id_mismatch() -> None:
    doc = _as_v03(_load_fixture("valid_v0_2_minimal.json"))
    doc["digest"]["clauses"][0]["id"] = "clause.some_other_clause"
    result = validate_document(doc)
    assert not result.ok
    assert any("digest" in str(e) for e in result.errors)


def test_validator_rejects_full_text_in_digest() -> None:
    doc = _as_v03(_load_fixture("valid_v0_2_minimal.json"))
    doc["digest"]["clauses"][0]["exemplar_forms"] = [
        {
            "text_summary": "t",
            "n": 1,
            "band": "rare",
            "x_extra": {"full_text": "leaked verbatim clause"},
        }
    ]
    result = validate_document(doc)
    assert not result.ok
    assert any("full_text" in str(e) for e in result.errors)


# ---------------------------------------------------------------------------
# CLI: digest + view bundle on a real compile
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


def test_digest_cmd_writes_sidecar(tmp_path: Path) -> None:
    out_dir = _compiled_out_dir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(cli, ["digest", str(out_dir)])
    assert result.exit_code == 0, result.output
    sidecar = json.loads((out_dir / "playbook.digest.json").read_text())
    pb = json.loads((out_dir / "playbook.opf.json").read_text())
    assert sidecar == pb["digest"]
    assert "tokens" in result.output


def test_digest_cmd_sidecar_matches_canonical_size(tmp_path: Path) -> None:
    """The on-disk sidecar must not exceed the canonical-chars/4 estimate it reports.

    Regression for issue #211: the sidecar used to be pretty-printed
    (indent=1), so a consumer measuring the actual file on disk saw ~20%+
    more bytes than the reported "~N tokens" estimate — the ~40K-token
    budget promise silently depended on the consumer re-canonicalizing
    rather than reading the artifact as shipped.
    """
    out_dir = _compiled_out_dir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(cli, ["digest", str(out_dir)])
    assert result.exit_code == 0, result.output

    raw = (out_dir / "playbook.digest.json").read_text(encoding="utf-8")
    sidecar = json.loads(raw)
    expected = canonicalize(sidecar)
    # Exactly the canonical form plus a single trailing newline for the
    # on-disk file — no extra whitespace from pretty-printing.
    assert raw == expected + "\n"
    assert len(raw.rstrip("\n")) == len(expected)


def test_digest_cmd_truncated_opf_reports_error_no_traceback(tmp_path: Path) -> None:
    """A hand-edited/truncated playbook.opf.json fails cleanly (issue #57), not a traceback."""
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "playbook.opf.json").write_text('{"truncated":', encoding="utf-8")
    runner = CliRunner()
    result = runner.invoke(cli, ["digest", str(out_dir)])
    assert result.exit_code == 1
    assert "ERROR" in result.output
    # A raw JSONDecodeError propagating uncaught would surface as some other
    # exception type here; the handled path always exits via SystemExit(1).
    assert isinstance(result.exception, SystemExit)


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


def test_preferred_variation_with_no_matching_position_counts_its_own_deal() -> None:
    doc = {
        "opf_version": "0.2",
        "evidence": {
            "clauses": [
                {
                    "id": "clause.x",
                    "taxonomy_id": "x",
                    "title": "X",
                    "observed_positions": _deals("Some other signed text.", 4),
                    "summary": {
                        "historical_stance": "mixed",
                        "acceptable_if": [
                            {
                                "if": "Accepted language",
                                "to": "Accepted language, verbatim.",
                                "rationale": "engine narration",
                                "observation_ref": {
                                    "document_id": "deal-009",
                                    "version": 1,
                                    "clause_path": "3",
                                },
                            }
                        ],
                        "fallbacks": [],
                        "rejected": [],
                        "confidence": {"score": 0.5},
                    },
                }
            ],
            "clause_library": [],
        },
    }
    pv = build_digest(doc)["clauses"][0]["preferred_variations"]
    assert [(p["n"], p["band"]) for p in pv] == [(1, "rare")]


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
    from playbook_engine.clause_library_compiler import compile_clause_library
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
    library, _ = compile_clause_library(observations)
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
        clause_library=library,
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


def test_v3_dispatch_leaves_0_3_digest_unchanged() -> None:
    """build_digest on a 0.3 document is the frozen digest_version 2 shape."""
    doc = _load_fixture("valid_v0_2_minimal.json")
    doc["opf_version"] = "0.3"
    digest = build_digest(doc)
    assert digest["digest_version"] == "2"
    assert "clause_count" in digest
