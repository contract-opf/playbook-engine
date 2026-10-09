"""Conformance vector suite for canonicalize.py + digest.py — issue #115.

``spec/conformance/0.5/`` is the frozen, standalone-consumable (plain JSON,
no Python import required) normative definition of canonicalization,
content hashing, and digest construction for OPF 0.5 / digest_version 4 —
the one format the engine reads and writes (issue #238; the OPF 0.3 /
digest 2 set was retired with that format). This suite is the reference
check: for every vector, recompute ``canonicalize_playbook``/
``content_hash``/``compute_section_digests``/``build_digest_v4`` from the
vector's ``input`` using THIS engine and assert the result equals the
vector's FROZEN ``expected.*`` values byte-for-byte.

The ``expected.*`` values are generated once by
``scripts/generate_conformance_vectors.py`` and committed — this test never
recomputes them at verification time from anywhere but the fixture file, so
a bug that changed canonicalize.py's/digest.py's output would actually be
caught here, not just asserted "self-consistent" against itself.
The ``test_mutated_*`` tests below prove that directly (in-memory
tampering, never touching the fixture file on disk).

See ``spec/conformance/0.5/README.md`` for the vector-by-vector rationale.

SECURITY NOTE: every vector's ``input`` is a synthetic, hand-built minimal
document — no real agreement content.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from playbook_engine.canonicalize import (
    canonicalize_playbook,
    compute_section_digests,
    content_hash,
)
from playbook_engine.digest import build_digest_v4
from playbook_engine.dossiers import (
    build_derived_sections,
    dossier_budget,
    dossier_tokens,
)

ROOT = Path(__file__).parent.parent
CONFORMANCE_DIR = ROOT / "spec" / "conformance"
#: The OPF 0.5 / digest_version 4 set (issues #223, #234).
CONFORMANCE_DIR_V05 = CONFORMANCE_DIR / "0.5"


def test_retired_0_3_vector_set_is_gone() -> None:
    """The OPF 0.3 / digest 2 set was retired with that format (issue #238)."""
    assert not (CONFORMANCE_DIR / "manifest.json").exists()
    assert not (CONFORMANCE_DIR / "vectors").exists()


# ---------------------------------------------------------------------------
# OPF 0.5 / digest_version 4 set (issues #223, #234) — spec/conformance/0.5/
# ---------------------------------------------------------------------------


def _load_manifest_v05() -> dict[str, Any]:
    return json.loads((CONFORMANCE_DIR_V05 / "manifest.json").read_text(encoding="utf-8"))


def _vector_files_v05() -> list[str]:
    files = [entry["file"] for entry in _load_manifest_v05()["vectors"]]
    assert files, "0.5/manifest.json lists no vectors"
    return files


def _load_vector_v05(relative_file: str) -> dict[str, Any]:
    return json.loads((CONFORMANCE_DIR_V05 / relative_file).read_text(encoding="utf-8"))


def test_v05_manifest_lists_every_vector_file_on_disk() -> None:
    manifest_files = {entry["file"] for entry in _load_manifest_v05()["vectors"]}
    on_disk = {f"vectors/{p.name}" for p in (CONFORMANCE_DIR_V05 / "vectors").glob("*.json")}
    assert manifest_files == on_disk


def test_v05_manifest_is_stamped_0_5_digest_4() -> None:
    fv = _load_manifest_v05()["format_version"]
    assert fv["opf_version"] == "0.5"
    assert fv["digest_version"] == "4"
    assert fv["engine_version"]


@pytest.mark.parametrize("filename", _vector_files_v05())
def test_v05_vector_reproduces_exactly(filename: str) -> None:
    vector = _load_vector_v05(filename)
    doc = vector["input"]
    expected = vector["expected"]
    assert vector["opf_version"] == "0.5" and doc["opf_version"] == "0.5"
    assert canonicalize_playbook(doc) == expected["canonical"], filename
    assert content_hash(doc) == expected["content_hash"], filename
    assert compute_section_digests(doc) == expected["section_digests"], filename
    assert build_digest_v4(doc) == expected["digest"], filename
    # Issue #228: the manifest, dossiers and provenance index are pinned too.
    derived = build_derived_sections(doc)
    for name in ("manifest", "dossiers", "provenance_index"):
        assert derived[name] == expected[name], (filename, name)
    assert expected["digest"]["digest_version"] == "4"
    assert vector["digest_version"] == "4"
    assert "full_text" not in json.dumps(expected["digest"])


@pytest.mark.parametrize("filename", _vector_files_v05())
def test_v05_vector_inputs_are_valid_0_5_documents(filename: str) -> None:
    """Every 0.5 input is a self-consistent document the validator accepts —
    precedent ids, clause counts, and (once embedded) the digest all agree."""
    from playbook_engine.validator import validate_document

    vector = _load_vector_v05(filename)
    doc = copy.deepcopy(vector["input"])
    doc["digest"] = vector["expected"]["digest"]
    for name in ("manifest", "dossiers", "provenance_index"):
        doc[name] = vector["expected"][name]
    result = validate_document(doc)
    assert result.ok, [str(e) for e in result.errors if e.blocking]


def test_v05_mutated_digest_is_detected() -> None:
    """A tampered expected digest (one deal count off) must not match."""
    vector = _load_vector_v05("vectors/002-variants-refused-and-exclusions.json")
    tampered = copy.deepcopy(vector["expected"]["digest"])
    tampered["clauses"][0]["signed_variants"][0]["n_deals"] += 1
    recomputed = build_digest_v4(vector["input"])
    assert recomputed != tampered
    assert recomputed == vector["expected"]["digest"]


def test_v05_negator_is_never_merged_into_a_variant() -> None:
    """002 pins exact normalization: 'may not assign' is its own group, and the
    case/punctuation respelling of variant A is merged into A (3 deals)."""
    digest = _load_vector_v05("vectors/002-variants-refused-and-exclusions.json")["expected"][
        "digest"
    ]
    variants = digest["clauses"][0]["signed_variants"]
    assert [v["n_deals"] for v in variants] == [3, 1]
    assert " not " in variants[1]["text"]
    assert digest["clauses"][0]["refused_asks"][0]["n_deals"] == 3


def test_v05_counterparty_alias_never_splits_a_variant() -> None:
    """004 pins the grouping key's party neutralization (OPF-SPEC §3.5.4):
    texts differing only by the counterparty's Counterparty-<n> alias are one
    group with n_deals 2 (signed variants and refused asks alike), the
    parties' places swapped stays its own group, and the one-deal variants'
    order shows perspective.party was rewritten to 'party'."""
    vector = _load_vector_v05("vectors/004-party-alias-grouping.json")
    entry = vector["expected"]["digest"]["clauses"][0]
    assert [(v["n_deals"], v["ref"]["document_id"]) for v in entry["signed_variants"]] == [
        (2, "deal-a"),
        (1, "deal-d"),
        (1, "deal-c"),
    ]
    assert [a["n_deals"] for a in entry["refused_asks"]] == [2]
    clause = vector["input"]["evidence"]["clauses"][0]
    assert (clause["n_variants"], clause["n_refused"]) == (3, 1)


def test_v05_opening_rules_vector_pins_every_branch() -> None:
    """006 (issue #234): edited standard, struck standard, non-standard changed
    to standard, non-standard unchanged, struck + own refused ask (excluded
    from changed_openings), struck with no refused ask (included, n_struck 1),
    "absent" and an unsigned deal (ignored)."""
    digest = _load_vector_v05("vectors/006-opening-rules.json")["expected"]["digest"]
    clause = digest["clauses"][0]
    assert (clause["n_opened_standard"], clause["n_kept_standard"]) == (3, 1)
    by_text = {v["text"]: v for v in clause["signed_variants"]}
    new_york = next(v for t, v in by_text.items() if "New York" in t)
    texas = next(v for t, v in by_text.items() if "Texas" in t)
    assert (new_york["n_deals"], new_york["n_from_standard"], new_york["n_unchanged"]) == (2, 1, 0)
    assert (texas["n_deals"], texas["n_from_standard"], texas["n_unchanged"]) == (1, 0, 1)
    assert [
        (e["n_deals"], e["n_to_standard"], e["n_struck"]) for e in clause["changed_openings"]
    ] == [
        (2, 1, 0),
        (1, 0, 1),
    ]
    assert clause["n_changed_openings_total"] == 2
    assert len(clause["refused_asks"]) == 1
    # The excluded opening is the refused ask's text; it is not also a changed opening.
    assert clause["refused_asks"][0]["text"].lower().rstrip(".") not in {
        e["text"].lower().rstrip(".") for e in clause["changed_openings"]
    }


def test_v05_label_vector_collapses_equivalent_variants_and_lists_uncovered() -> None:
    """007 (issues #240, #234): equivalent variants are ONE entry, the rest
    keep the tier order, and the uncovered list is the eligible taxonomy
    entries with no evidence clause."""
    digest = _load_vector_v05("vectors/007-equivalence-label-and-coverage.json")["expected"][
        "digest"
    ]
    listed = digest["clauses"][0]["signed_variants"]
    assert [v["label"] for v in listed] == [
        "less_protective",
        "different_concept",
        None,
        "more_protective",
        "equivalent",
    ]
    collapsed = listed[-1]
    assert (collapsed["n_deals"], collapsed["n_texts"], len(collapsed["exemplars"])) == (3, 2, 2)
    assert "text" not in collapsed
    assert [u["taxonomy_id"] for u in digest["uncovered_clause_types"]] == ["audit_rights", "term"]
    assert digest["clauses"][0]["n_variants_total"] == 6


# ---------------------------------------------------------------------------
# Proof the suite has teeth (issue #115 reviewer gate): a tampered expected
# value must be DETECTED, not silently pass. These mutate only an in-memory
# copy — never the fixture file on disk.
# ---------------------------------------------------------------------------


def _flip_last_hex_char(sha: str) -> str:
    prefix, hexdigest = sha.split(":", 1)
    last = hexdigest[-1]
    flipped = "0" if last != "0" else "1"
    return f"{prefix}:{hexdigest[:-1]}{flipped}"


@pytest.mark.parametrize("filename", _vector_files_v05())
def test_mutated_content_hash_is_detected(filename: str) -> None:
    vector = _load_vector_v05(filename)
    tampered_expected = _flip_last_hex_char(vector["expected"]["content_hash"])
    recomputed = content_hash(vector["input"])
    assert recomputed != tampered_expected
    assert recomputed == vector["expected"]["content_hash"]


def test_mutated_canonical_bytes_are_detected() -> None:
    vector = _load_vector_v05("vectors/001-minimal-no-perspective.json")
    tampered_expected = vector["expected"]["canonical"].replace(
        '"opf_version":"0.5"', '"opf_version":"9.9"'
    )
    assert tampered_expected != vector["expected"]["canonical"]
    assert canonicalize_playbook(vector["input"]) != tampered_expected
    assert canonicalize_playbook(vector["input"]) == vector["expected"]["canonical"]


def test_v05_hard_rules_and_dossiers_vector_pins_selection_and_the_bound() -> None:
    """008 (issue #228): rules in Floor order with their defaults, the
    concession-then-changed-opening excerpt order over more groups than the
    cap, a rule that demands presence or carries a predicate naming its clause,
    a dossier over its budget that drops its second excerpt WHOLE (never a cut)
    and then its Floor rule, and a dossier with no excerpt that drops its last
    Floor rule whole to fit."""
    vector = _load_vector_v05("vectors/008-hard-rules-and-dossiers.json")
    expected = vector["expected"]
    rules = {r["rule_id"]: r for r in expected["manifest"]["hard_rules"]}
    assert list(rules) == [i["id"] for i in vector["input"]["floor"]["invariants"]]
    assert rules["venue-holds"]["required_presence"] is True
    assert rules["venue-holds"]["fallback_language"].startswith(
        "The courts of the State of Delaware"
    )
    assert (
        rules["venue-no-arbitration"]["condition"],
        rules["venue-no-arbitration"]["required_presence"],
    ) == (
        "judged",
        False,
    )
    assert rules["term-cap"]["condition"]["type"] == "numeric_bound"
    assert rules["term-cap"]["fallback_language"] is None
    assert rules["survival-reference"]["clause_id"] == "clause.survival"
    assert rules["survival-reference"]["required_presence"] is True
    # A rule that demands presence or carries a predicate names its clause.
    for rule in rules.values():
        if rule["required_presence"] or isinstance(rule["condition"], dict):
            assert isinstance(rule["taxonomy_id"], str), rule["rule_id"]
    assert rules["no-ghost-clause"]["clause_id"] is None
    assert rules["no-ghost-clause"]["taxonomy_id"] == "ghost_clause"

    venue = expected["dossiers"]["clause.venue"]
    assert [(e["kind"], e["precedent_id"]) for e in venue["excerpts"]] == [
        (
            "signed_variant",
            next(
                p["id"]
                for p in vector["input"]["evidence"]["precedent"]
                if p["document_id"] == "deal-a" and p["taxonomy_id"] == "venue"
            ),
        ),
        ("changed_opening", venue["excerpts"][1]["precedent_id"]),
    ]
    assert (
        venue["excerpts"][0]["opening"] is not None
        and "Texas" not in venue["excerpts"][0]["signed"]
    )
    # Term: no standard (1,000-token budget). Its first excerpt alone is over
    # budget and is kept WHOLE; the second is dropped whole and named, then its
    # Floor rule is dropped whole (n_floor_rules still counts it).
    term = expected["dossiers"]["clause.term"]
    assert term["n_floor_rules"] == 1 and term["floor_rules"] == []
    records = {p["id"]: p for p in vector["input"]["evidence"]["precedent"]}
    assert len(term["excerpts"]) == 1 and term["n_omitted"] == 1
    kept = term["excerpts"][0]
    assert kept["signed"] == records[kept["precedent_id"]]["signed_text"]["text"]
    assert "…" not in kept["signed"]
    assert term["omitted_precedent_ids"] == [
        p["id"]
        for p in records.values()
        if p["taxonomy_id"] == "term" and p["id"] != kept["precedent_id"]
    ]
    assert dossier_tokens(term) > dossier_budget(None)
    for clause_id, dossier in expected["dossiers"].items():
        standard = dossier["our_standard"]
        budget = dossier_budget(standard["text"] if standard else None)
        # Only a dossier holding its single kept excerpt (no Floor rule left
        # listed) may be over its budget.
        assert dossier_tokens(dossier) <= budget or (
            len(dossier["excerpts"]) == 1 and dossier["floor_rules"] == []
        ), clause_id
        assert dossier["n_omitted"] == len(dossier["omitted_precedent_ids"])
        for ex in dossier["excerpts"]:  # every excerpt text is a record's own text, whole
            record = records[ex["precedent_id"]]
            assert ex["signed"] in (None, record["signed_text"]["text"])
    # Survival: no excerpt; its three Floor rules exceed the budget, so the last
    # listed is dropped whole and the dossier fits.
    survival = expected["dossiers"]["clause.survival"]
    assert survival["excerpts"] == [] and survival["n_floor_rules"] == 3
    assert [r["rule_id"] for r in survival["floor_rules"]] == [
        "survival-reference",
        "survival-scope",
    ]
    assert dossier_tokens(survival) <= dossier_budget(survival["our_standard"]["text"])
    index = expected["provenance_index"]["dossiers"]["clause.venue"]
    assert [row["precedent_id"] for row in index] == [e["precedent_id"] for e in venue["excerpts"]]
    assert not any(row["omitted"] for row in index)
    # The term dossier's dropped excerpt is indexed as omitted, with its deal.
    term_rows = expected["provenance_index"]["dossiers"]["clause.term"]
    assert [(row["precedent_id"], row["omitted"]) for row in term_rows] == [
        (kept["precedent_id"], False),
        (term["omitted_precedent_ids"][0], True),
    ]
    assert term_rows[1]["document_id"] in {
        d["document_id"] for d in expected["provenance_index"]["documents"]
    }
