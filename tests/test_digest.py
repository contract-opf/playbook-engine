"""Tests for the OPF 0.3 digest section, `playbook digest`, and `view bundle`.

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
) -> dict:
    return {
        "text_summary": text,
        "full_text": full_text or text,
        "example_ref": {"document_id": "deal-001", "version": 1, "clause_path": "1"},
        "deviation": deviation,
        "risk_delta": {"direction": direction, "magnitude": magnitude},
        "provenance": "our_paper",
        "outcome": "signed",
        "precedent_count": n,
    }


# ---------------------------------------------------------------------------
# build_digest
# ---------------------------------------------------------------------------


def test_build_digest_from_v02_fixture() -> None:
    doc = _load_fixture("valid_v0_2_minimal.json")
    digest = build_digest(doc)
    assert digest["digest_version"] == DIGEST_VERSION
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


def test_exemplar_dedupe_by_normalized_text_sums_precedent_counts() -> None:
    obs = [
        _obs("Indemnification survives termination.", n=7),
        # Same text modulo case/punct/whitespace — must merge into one form.
        _obs("indemnification  survives termination", n=5),
    ]
    forms = _exemplar_forms(obs)
    assert len(forms) == 1
    assert forms[0]["n"] == 12
    assert forms[0]["band"] == "often"


def test_exemplar_bands() -> None:
    forms = _exemplar_forms(
        [_obs("alpha clause text", n=10), _obs("beta clause text", n=2), _obs("gamma clause", n=1)]
    )
    by_text = {f["text_summary"]: f["band"] for f in forms}
    assert by_text["alpha clause text"] == "often"
    assert by_text["beta clause text"] == "sometimes"
    assert by_text["gamma clause"] == "rare"


def test_exemplar_top_n_plus_material() -> None:
    obs = [_obs(f"common form variant number {i}", n=20 - i) for i in range(EXEMPLAR_TOP_N)]
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
    rejected = [_obs(f"rejected ask variant {i}", n=1) for i in range(20)]
    rejected += [_obs("rejected ask variant 0", n=4)]  # duplicate of variant 0
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
    # the duplicated variant merged with summed n and ranks first
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


def test_compiled_playbook_is_v03_with_digest(tmp_path: Path) -> None:
    out_dir = _compiled_out_dir(tmp_path)
    pb = json.loads((out_dir / "playbook.opf.json").read_text())
    assert pb["opf_version"] == "0.3"
    assert pb["digest"]["digest_version"] == DIGEST_VERSION
    assert pb["digest"]["clause_count"] == len(pb["evidence"]["clauses"])
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
