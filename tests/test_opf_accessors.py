"""OPF accessors (issue #154; OPF 0.4 precedent, issue #223).

The cases read the real compiled NDA example
(``examples/nda/playbook.opf.json``, produced by ``playbook project``). The
0.1-0.3 accessors were retired with those formats (issue #238). Where a test mutates
one of those documents (unsigned drafts, flipped paper side), the mutated
document is re-stamped with ``precedent.refresh_derived`` and asserted to pass
``validator.validate_document`` -- never a shape the engine itself rejects.

The query surface (issue #224) is also exercised on the frozen OPF 0.4
conformance vector 002 (``spec/conformance/0.4/vectors/``), read-only: the
reference compiler never sets ``signed_at`` (OPF-SPEC §3.5.4), so that
vector — which models a third-party producer recording signing dates, with
multi-deal variant groups, a struck clause and a refused ask shared by
three deals — is the only committed 0.4 document on which recency and
group ranking can be observed.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from playbook_engine.canonicalize import canonicalize, file_sha256
from playbook_engine.digest import build_digest
from playbook_engine.opf_accessors import (
    PRECEDENT_SIDECAR,
    SIDECARS_KEY,
    clause_precedent,
    find_precedent,
    is_precedent_shape,
    playbook_clauses,
    playbook_precedent,
    precedent_by_id,
    precedent_jsonl,
    precedent_sidecar_manifest,
    verify_precedent_sidecar,
)
from playbook_engine.precedent import refresh_derived
from playbook_engine.validator import validate_document

_ROOT = Path(__file__).parent.parent
_NDA = _ROOT / "examples" / "nda" / "playbook.opf.json"
_NDA_SIDECAR = _ROOT / "examples" / "nda" / "precedent.jsonl"
_VECTOR_002 = (
    _ROOT / "spec" / "conformance" / "0.4" / "vectors" / "002-variants-refused-and-exclusions.json"
)


def _load(path: Path) -> dict[str, Any]:
    doc: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return doc


def test_precedent_accessors_read_a_compiled_0_4_document() -> None:
    doc = _load(_NDA)
    assert doc["opf_version"] == "0.4"
    assert is_precedent_shape(doc)
    records = playbook_precedent(doc)
    assert records == doc["evidence"]["precedent"]
    for clause in playbook_clauses(doc):
        own = clause_precedent(doc, clause)
        assert all(p["taxonomy_id"] == clause["taxonomy_id"] for p in own)
        assert len({p["document_id"] for p in own}) == clause["n_deals"]
    assert sum(len(clause_precedent(doc, c)) for c in playbook_clauses(doc)) == len(records)


def test_retired_0_3_accessors_are_gone() -> None:
    """The 0.1-0.3 accessors were retired with those formats (issue #238)."""
    import playbook_engine.opf_accessors as acc

    for name in (
        "clause_stance",
        "clause_confidence",
        "clause_is_thin",
        "clause_trail",
        "observation_dynamics",
        "playbook_clause_library",
    ):
        assert not hasattr(acc, name), name


def test_accessors_are_empty_on_a_document_without_evidence() -> None:
    """A top-level ``clauses`` list (the retired 0.1 shape) is not read."""
    doc = {"opf_version": "0.4", "clauses": [{"id": "clause.x", "taxonomy_id": "x"}]}
    assert not is_precedent_shape(doc)
    assert playbook_clauses(doc) == []
    assert playbook_precedent(doc) == []


def test_precedent_accessor_drops_non_dict_entries() -> None:
    doc = _load(_NDA)
    doc["evidence"]["precedent"] = [None, *doc["evidence"]["precedent"][:1], "x"]
    assert playbook_precedent(doc) == doc["evidence"]["precedent"][1:2]


# ---------------------------------------------------------------------------
# Query surface (issue #224)
# ---------------------------------------------------------------------------


def _vector_002() -> dict[str, Any]:
    doc: dict[str, Any] = _load(_VECTOR_002)["input"]
    return doc


def test_find_precedent_ranks_by_deal_count_then_recency() -> None:
    """Vector 002's assignment clause: the 3-deal affiliate-assignment group
    (signed 2025-03-01, 2025-Q4 and undated) outranks the 2-deal standard
    group, which outranks the 1-deal groups ordered newest first with the
    undated struck clause after the dated one. The unsigned deal-f's
    on-notice draft is not signed precedent, so it comes last. Inside a
    group the record's own signing date decides."""
    doc = _vector_002()
    ranked = [p["document_id"] for p in find_precedent(doc, "assignment")]
    assert ranked == [
        "deal-d",
        "deal-c",
        "deal-e",
        "deal-b",
        "deal-a",
        "deal-g",
        "deal-h",
        "deal-f",
    ]
    # Recency alone: groups by their latest signing date (2025-Q4 group,
    # then the 2025-02-10 standard group, then 2024-06-30), undated last.
    by_date = [p["document_id"] for p in find_precedent(doc, "assignment", order=("last_signed",))]
    assert by_date == [
        "deal-d",
        "deal-c",
        "deal-e",
        "deal-b",
        "deal-a",
        "deal-g",
        "deal-h",
        "deal-f",
    ]
    # No ranking keys: grouping key, then each record's own date; the
    # unsigned record still comes after every signed one.
    unranked = [p["document_id"] for p in find_precedent(doc, "assignment", order=())]
    assert unranked[0] == "deal-h"
    assert unranked[-1] == "deal-f"
    # Records are the document's own dicts, never copies or projections.
    assert all(
        any(p is q for q in doc["evidence"]["precedent"]) for p in find_precedent(doc, "assignment")
    )


def test_find_precedent_group_order_matches_the_digest() -> None:
    """The non-standard signed groups appear in find_precedent in exactly the
    order the digest ranks its signed_variants (both rank n_deals, then
    last_signed) — on the dated vector and on the real compiled NDA."""
    for doc in (_vector_002(), _load(_NDA)):
        digest = build_digest(doc)
        for clause in digest["clauses"]:
            variant_of = {
                pid: i
                for i, v in enumerate(clause["signed_variants"])
                for pid in v["precedent_ids"]
            }
            seen = [
                variant_of[p["id"]]
                for p in find_precedent(doc, clause["taxonomy_id"])
                if p["id"] in variant_of
            ]
            groups = [g for i, g in enumerate(seen) if i == 0 or seen[i - 1] != g]
            assert groups == list(range(len(clause["signed_variants"]))), clause["taxonomy_id"]


def test_find_precedent_never_counts_unsigned_drafts() -> None:
    """An unsigned deal's signed_text is its last draft (OPF-SPEC §3.5.4),
    never signed precedent (owner decision 2026-09-13 (a)): four unsigned
    records carrying deal-g's text neither lift deal-g's 1-deal group above
    the 3-deal affiliate group nor count toward it; they rank after every
    signed record, and the digest's signed_variants order still matches."""
    doc = copy.deepcopy(_vector_002())
    documents = doc["corpus"]["documents"]
    precedent = doc["evidence"]["precedent"]
    deal_f_doc = next(d for d in documents if d["document_id"] == "deal-f")
    deal_f = next(p for p in precedent if p["document_id"] == "deal-f")
    deal_g = next(p for p in precedent if p["document_id"] == "deal-g")
    drafts = [f"deal-u{i}" for i in range(4)]
    for deal_id in drafts:
        # Built as the reference compiler writes an unsigned deal: a
        # corpus.documents entry with no signed version, signed_at omitted
        # (never null, OPF-SPEC §3.5.4), the paper of an unsigned deal, and
        # ids/counts/digest re-stamped below so the validator accepts it.
        entry = copy.deepcopy(deal_f_doc)
        entry["document_id"] = deal_id
        entry["signed_version"] = None
        documents.append(entry)
        draft = copy.deepcopy(deal_g)
        draft["document_id"] = deal_id
        draft["signed_text"]["ref"]["document_id"] = deal_id
        draft["signed"] = False
        draft.pop("signed_at", None)
        for key in ("paper", "paper_basis", "paper_confidence"):
            draft[key] = deal_f[key]
        precedent.append(draft)
    refresh_derived(doc)
    result = validate_document(doc)
    assert result.ok, result.errors

    ranked = [p["document_id"] for p in find_precedent(doc, "assignment")]
    assert ranked[:3] == ["deal-d", "deal-c", "deal-e"]
    assert ranked.index("deal-g") < ranked.index("deal-f")
    signed = [p["signed"] is True for p in find_precedent(doc, "assignment")]
    assert signed == sorted(signed, reverse=True)
    assert set(ranked[-5:]) == {"deal-f", *drafts}

    digest = build_digest(doc)
    clause = next(c for c in digest["clauses"] if c["taxonomy_id"] == "assignment")
    variant_of = {
        pid: i for i, v in enumerate(clause["signed_variants"]) for pid in v["precedent_ids"]
    }
    seen = [variant_of[p["id"]] for p in find_precedent(doc, "assignment") if p["id"] in variant_of]
    groups = [g for i, g in enumerate(seen) if i == 0 or seen[i - 1] != g]
    assert groups == list(range(len(clause["signed_variants"])))


def test_find_precedent_refused_returns_asks_with_their_record() -> None:
    doc = _vector_002()
    asks = find_precedent(doc, "assignment", refused=True)
    assert [(a["document_id"], a["round"]) for a in asks] == [
        ("deal-h", 0),
        ("deal-d", 1),
        ("deal-f", 2),
    ]
    by_id = {p["id"]: p for p in doc["evidence"]["precedent"]}
    for ask in asks:
        record = by_id[ask["precedent_id"]]
        assert ask["document_id"] == record["document_id"]
        assert ask["taxonomy_id"] == "assignment"
        own = {k: ask[k] for k in ("text", "round", "ref")}
        assert own in record["refused_asks"]
    # A clause with no refused asks yields none.
    nda = _load(_NDA)
    assert find_precedent(nda, "governing_law", refused=True) == []
    assert len(find_precedent(nda, "residuals", refused=True)) == 1


def test_find_precedent_limit_clause_id_and_errors() -> None:
    doc = _load(_NDA)
    full = find_precedent(doc, "governing_law")
    assert len(full) == 6
    assert find_precedent(doc, "governing_law", limit=2) == full[:2]
    assert find_precedent(doc, "governing_law", limit=0) == []
    assert find_precedent(doc, "clause.governing_law") == full
    assert find_precedent(doc, "no_such_clause") == []
    with pytest.raises(ValueError, match="unknown order key"):
        find_precedent(doc, "governing_law", order=("paper",))
    with pytest.raises(ValueError, match="limit"):
        find_precedent(doc, "governing_law", limit=-1)
    no_precedent = copy.deepcopy(doc)
    no_precedent["evidence"]["precedent"] = []
    for clause in playbook_clauses(no_precedent):
        assert find_precedent(no_precedent, str(clause.get("taxonomy_id"))) == []


def test_find_precedent_ignores_paper_side() -> None:
    """Paper side is metadata only (owner decision 2026-09-13 (b)): flipping
    every record's paper never changes the ranking."""
    doc = _vector_002()
    before = [p["document_id"] for p in find_precedent(doc, "assignment")]
    flipped = copy.deepcopy(doc)
    for d in flipped["corpus"]["documents"]:
        d["provenance"] = {
            "our_paper": "counterparty_paper",
            "counterparty_paper": "our_paper",
        }.get(d["provenance"], d["provenance"])
    for p in flipped["evidence"]["precedent"]:
        p["paper"] = {"ours": "theirs", "theirs": "ours"}.get(p["paper"], p["paper"])
    refresh_derived(flipped)
    result = validate_document(flipped)
    assert result.ok, result.errors
    assert [p["document_id"] for p in find_precedent(flipped, "assignment")] == before


def test_precedent_by_id_round_trips_every_record() -> None:
    doc = _load(_NDA)
    for record in playbook_precedent(doc):
        assert precedent_by_id(doc, record["id"]) is record
    assert precedent_by_id(doc, "prec.0000000000000000") is None
    assert precedent_by_id({"opf_version": "0.4", "evidence": {"clauses": []}}, "prec.x") is None


def test_precedent_jsonl_is_the_records_sorted_by_id() -> None:
    doc = _load(_NDA)
    text = precedent_jsonl(doc)
    lines = text.splitlines()
    assert text.endswith("\n") and len(lines) == len(playbook_precedent(doc))
    parsed = [json.loads(line) for line in lines]
    assert [p["id"] for p in parsed] == sorted(p["id"] for p in playbook_precedent(doc))
    for line, record in zip(lines, parsed, strict=True):
        assert record == precedent_by_id(doc, record["id"])
        assert line == canonicalize(precedent_by_id(doc, record["id"]))
    assert precedent_jsonl({"opf_version": "0.4", "evidence": {"clauses": []}}) == ""


def test_committed_nda_sidecar_belongs_to_its_playbook() -> None:
    """examples/nda/precedent.jsonl is what `playbook project` writes next to
    the committed playbook, and the playbook's x_sidecars names its sha256."""
    doc = _load(_NDA)
    assert doc[SIDECARS_KEY] == precedent_sidecar_manifest(doc)
    entry = doc[SIDECARS_KEY][PRECEDENT_SIDECAR]
    assert entry["records"] == len(playbook_precedent(doc))
    assert _NDA_SIDECAR.read_bytes() == precedent_jsonl(doc).encode("utf-8")
    assert file_sha256(_NDA_SIDECAR) == entry["sha256"]
    assert verify_precedent_sidecar(doc, _NDA_SIDECAR)


def test_verify_precedent_sidecar_rejects_a_foreign_file(tmp_path: Path) -> None:
    doc = _load(_NDA)
    tampered = tmp_path / PRECEDENT_SIDECAR
    lines = _NDA_SIDECAR.read_text(encoding="utf-8").splitlines(keepends=True)
    tampered.write_text("".join(lines[1:]), encoding="utf-8")
    assert not verify_precedent_sidecar(doc, tampered)
    assert not verify_precedent_sidecar(doc, tmp_path / "missing.jsonl")
    no_manifest = {k: v for k, v in doc.items() if k != SIDECARS_KEY}
    assert not verify_precedent_sidecar(no_manifest, _NDA_SIDECAR)


def test_refresh_derived_restamps_the_sidecar_hash() -> None:
    """A transform that rewrites precedent text after assembly (publish's
    scrub, export_profile's residue rewrites) re-derives the sidecar hash
    along with the ids and digest, so x_sidecars never names stale bytes."""
    doc = _load(_NDA)
    before = doc[SIDECARS_KEY][PRECEDENT_SIDECAR]["sha256"]
    record = find_precedent(doc, "governing_law")[0]
    record["signed_text"]["text"] = record["signed_text"]["text"] + " Rewritten."
    refresh_derived(doc)
    after = doc[SIDECARS_KEY][PRECEDENT_SIDECAR]["sha256"]
    assert after != before
    assert doc[SIDECARS_KEY] == precedent_sidecar_manifest(doc)
    # A document recording no sidecar is not given one.
    bare = {k: v for k, v in _load(_NDA).items() if k != SIDECARS_KEY}
    refresh_derived(bare)
    assert SIDECARS_KEY not in bare
