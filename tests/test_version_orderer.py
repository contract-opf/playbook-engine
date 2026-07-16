"""Tests for the version orderer.

SECURITY NOTE: All fixtures use programmatically constructed ClauseTree
objects with synthetic text.  No real agreement files are referenced.
Party names use fictional identifiers only ("Alice Corp", "Beta Ltd",
"Party A", "Party B").
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from playbook_engine.clause_tree import ClauseNode, ClauseTree
from playbook_engine.signed_detector import SignedStatus
from playbook_engine.version_orderer import (
    EXHAUSTIVE_THRESHOLD,
    Hints,
    HintsError,
    PairwiseDistance,
    VersionInput,
    VersionOrder,
    _chain_cost,
    _edit_distance,
    _fingerprint,
    _pick_signed,
    chain_shape,
    order_versions,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tree(text: str, doc_id: str = "doc") -> ClauseTree:
    """Build a minimal ClauseTree from a block of text (one node per line)."""
    nodes = []
    for i, line in enumerate(text.strip().splitlines(), start=1):
        stripped = line.strip()
        if stripped:
            nodes.append(
                ClauseNode(
                    clause_path=str(i),
                    heading=None,
                    text=stripped,
                    char_span=(0, len(stripped)),
                )
            )
    return ClauseTree(document_id=doc_id, version="v1", source_file="doc.docx", nodes=nodes)


def _signed_status(signed: bool, confidence: float = 0.90) -> SignedStatus:
    if signed:
        return SignedStatus(signed=True, basis="dual_signatures", confidence=confidence)
    return SignedStatus(signed=False, basis="no_signature_section", confidence=0.85)


def _vi(
    vid: str,
    text: str,
    signed: bool = False,
    timestamp: str | None = None,
) -> VersionInput:
    return VersionInput(
        version_id=vid,
        tree=_tree(text, vid),
        signed=_signed_status(signed),
        timestamp=timestamp,
    )


# ---------------------------------------------------------------------------
# Three-version negotiation fixture
# ---------------------------------------------------------------------------
# v1 = template (most different from final)
# v2 = counterparty redline
# v3 = final signed (executed copy)

_V1_TEXT = """\
1. Definitions
"Agreement" means this Services Agreement.
2. Services
Alice Corp shall provide software development services.
3. Payment
Payment shall be thirty (30) days after invoice.
4. Term
This agreement shall be for one (1) year.
5. Termination
Either party may terminate on sixty (60) days notice.
"""

_V2_TEXT = """\
1. Definitions
"Agreement" means this Services Agreement dated January 2025.
2. Services
Alice Corp shall provide software development services as described in Exhibit A.
3. Payment
Payment shall be forty-five (45) days after invoice.
4. Term
This agreement shall be for two (2) years.
5. Termination
Either party may terminate on thirty (30) days notice.
6. Limitation of Liability
Beta Ltd shall not be liable for consequential damages.
"""

_V3_TEXT = """\
1. Definitions
"Agreement" means this Services Agreement dated January 15, 2025.
2. Services
Alice Corp shall provide software development services as described in Exhibit A.
3. Payment
Payment shall be forty-five (45) days after invoice, with a three-day grace period.
4. Term
This agreement shall be for two (2) years, automatically renewable.
5. Termination
Either party may terminate on thirty (30) days written notice.
6. Limitation of Liability
Beta Ltd shall not be liable for consequential or indirect damages.
7. Signatures
By: Alice Smith, CEO
By: Bob Jones, President
"""


# ---------------------------------------------------------------------------
# _fingerprint
# ---------------------------------------------------------------------------


def test_fingerprint_returns_list_of_strings() -> None:
    tree = _tree("Line one\nLine two")
    result = _fingerprint(tree)
    assert isinstance(result, list)
    assert all(isinstance(s, str) for s in result)


def test_fingerprint_includes_heading_text() -> None:
    node = ClauseNode(clause_path="1", heading="Definitions", text="Body text.", char_span=(0, 11))
    tree = ClauseTree(document_id="d", version="v1", source_file="f", nodes=[node])
    fp = _fingerprint(tree)
    assert "Definitions" in fp
    assert "Body text." in fp


def test_fingerprint_empty_tree() -> None:
    tree = ClauseTree(document_id="d", version="v1", source_file="f")
    assert _fingerprint(tree) == []


def test_fingerprint_strips_whitespace() -> None:
    tree = _tree("  line with spaces  ")
    fp = _fingerprint(tree)
    assert all(s == s.strip() for s in fp)


def test_fingerprint_collapses_embedded_newlines() -> None:
    """A node whose body text is wrapped across several physical lines (as a
    pdfplumber-style PDF ingester would emit) must fingerprint identically to
    the same content emitted as one DOCX-style paragraph line (#97)."""
    docx_node = ClauseNode(
        clause_path="1",
        heading=None,
        text="Alice Corp shall provide software development services under this Agreement.",
        char_span=(0, 10),
    )
    pdf_node = ClauseNode(
        clause_path="1",
        heading=None,
        text="Alice Corp shall provide software\ndevelopment services under this\nAgreement.",
        char_span=(0, 10),
    )
    docx_tree = ClauseTree(document_id="d", version="v1", source_file="f.docx", nodes=[docx_node])
    pdf_tree = ClauseTree(document_id="d", version="v2", source_file="f.pdf", nodes=[pdf_node])
    assert _fingerprint(docx_tree) == _fingerprint(pdf_tree)


def test_mixed_format_fingerprint_normalization() -> None:
    """Same content in two line-shapes must produce a near-zero edit distance
    (not a maximally-distant pair) — a DOCX draft vs the same content
    extracted from a PDF (pdfplumber wraps physical lines differently than a
    DOCX paragraph) must not look artificially unrelated (#97).

    version_orderer.py:430-449 previously fingerprinted per *physical* line
    (``str.splitlines()``), so identical content produced differently-shaped
    line sequences across formats and inflated the edit distance — exactly
    the physics provenance_detector.py:83-87 already documents for template
    similarity.
    """
    docx_text = (
        "1. Definitions\n"
        '"Agreement" means this Services Agreement between Alice Corp and Beta Ltd.\n'
        "2. Services\n"
        "Alice Corp shall provide software development services described in Exhibit A.\n"
        "3. Payment\n"
        "Payment shall be forty-five (45) days after invoice, with a grace period.\n"
    )
    # Same logical clauses, but each body wrapped across several physical
    # lines the way a PDF ingester (pdfplumber) would extract a rendered page
    # — mid-sentence line breaks that a DOCX ingester would never produce.
    pdf_nodes = [
        ClauseNode(clause_path="1", heading=None, text="1. Definitions", char_span=(0, 1)),
        ClauseNode(
            clause_path="2",
            heading=None,
            text='"Agreement" means this Services\nAgreement between Alice Corp\nand Beta Ltd.',
            char_span=(0, 1),
        ),
        ClauseNode(clause_path="3", heading=None, text="2. Services", char_span=(0, 1)),
        ClauseNode(
            clause_path="4",
            heading=None,
            text="Alice Corp shall provide software\ndevelopment services described in\nExhibit A.",
            char_span=(0, 1),
        ),
        ClauseNode(clause_path="5", heading=None, text="3. Payment", char_span=(0, 1)),
        ClauseNode(
            clause_path="6",
            heading=None,
            text="Payment shall be forty-five (45)\ndays after invoice, with a grace\nperiod.",
            char_span=(0, 1),
        ),
    ]
    docx_tree = _tree(docx_text, doc_id="draft")
    pdf_tree = ClauseTree(
        document_id="signed", version="v1", source_file="signed.pdf", nodes=pdf_nodes
    )

    d = _edit_distance(_fingerprint(docx_tree), _fingerprint(pdf_tree))
    assert d < 0.05, (
        f"expected near-zero distance for identical content across formats, got {d:.3f}"
    )

    # And at the chain level: the mixed-format identical-content pair must be
    # ordered adjacent with a tiny step cost, not treated as maximally distant.
    versions = [
        VersionInput("draft", docx_tree, SignedStatus(False, "no_signature_section", 0.85)),
        VersionInput("signed", pdf_tree, SignedStatus(True, "dual_signatures", 0.90)),
    ]
    result = order_versions(versions)
    assert result.ordered_ids == ("draft", "signed")
    assert result.total_distance < 0.05, (
        f"expected near-zero chain cost for a mixed-format identical-content pair, "
        f"got {result.total_distance:.3f}"
    )


# ---------------------------------------------------------------------------
# _edit_distance
# ---------------------------------------------------------------------------


def test_edit_distance_identical() -> None:
    a = ["line one", "line two", "line three"]
    assert _edit_distance(a, a) == 0.0


def test_edit_distance_completely_different() -> None:
    a = ["alpha", "beta", "gamma"]
    b = ["delta", "epsilon", "zeta"]
    d = _edit_distance(a, b)
    assert d > 0.5


def test_edit_distance_symmetric() -> None:
    a = ["line one", "line two"]
    b = ["line one", "different"]
    assert _edit_distance(a, b) == _edit_distance(b, a)


def test_edit_distance_in_range() -> None:
    a = _fingerprint(_tree(_V1_TEXT))
    b = _fingerprint(_tree(_V3_TEXT))
    d = _edit_distance(a, b)
    assert 0.0 <= d <= 1.0


def test_edit_distance_empty_both() -> None:
    assert _edit_distance([], []) == 0.0


def test_edit_distance_closer_versions() -> None:
    """v2→v3 should be a smaller distance than v1→v3 (v2 is closer to final)."""
    fp1 = _fingerprint(_tree(_V1_TEXT))
    fp2 = _fingerprint(_tree(_V2_TEXT))
    fp3 = _fingerprint(_tree(_V3_TEXT))
    d13 = _edit_distance(fp1, fp3)
    d23 = _edit_distance(fp2, fp3)
    assert d23 < d13, f"Expected d(v2,v3)={d23:.3f} < d(v1,v3)={d13:.3f}"


# ---------------------------------------------------------------------------
# _pick_signed
# ---------------------------------------------------------------------------


def test_pick_signed_returns_signed_version() -> None:
    versions = [
        _vi("v1", _V1_TEXT),
        _vi("v2", _V2_TEXT),
        _vi("v3", _V3_TEXT, signed=True),
    ]
    result = _pick_signed(versions)
    assert result is not None
    assert result.version_id == "v3"


def test_pick_signed_returns_none_when_none_signed() -> None:
    versions = [_vi("v1", _V1_TEXT), _vi("v2", _V2_TEXT)]
    assert _pick_signed(versions) is None


def test_pick_signed_prefers_higher_confidence() -> None:
    v1 = VersionInput("v1", _tree(_V1_TEXT), SignedStatus(True, "single_signature", 0.75))
    v2 = VersionInput("v2", _tree(_V2_TEXT), SignedStatus(True, "dual_signatures", 0.90))
    result = _pick_signed([v1, v2])
    assert result is not None
    assert result.version_id == "v2"


# ---------------------------------------------------------------------------
# _chain_cost
# ---------------------------------------------------------------------------


def test_chain_cost_single_step() -> None:
    dist = {("a", "b"): 0.3, ("b", "a"): 0.3}
    total, pairs = _chain_cost(["a", "b"], dist)
    assert total == 0.3
    assert len(pairs) == 1
    assert pairs[0].from_id == "a"
    assert pairs[0].to_id == "b"


def test_chain_cost_two_steps() -> None:
    dist = {("a", "b"): 0.2, ("b", "a"): 0.2, ("b", "c"): 0.15, ("c", "b"): 0.15}
    total, pairs = _chain_cost(["a", "b", "c"], dist)
    assert abs(total - 0.35) < 1e-9


def test_chain_cost_empty_chain() -> None:
    total, pairs = _chain_cost([], {})
    assert total == 0.0
    assert pairs == []


# ---------------------------------------------------------------------------
# order_versions: core behaviour
# ---------------------------------------------------------------------------


def test_order_versions_empty() -> None:
    result = order_versions([])
    assert result.ordered_ids == ()
    assert result.signed_id is None
    assert result.basis == "single"


def test_order_versions_single() -> None:
    result = order_versions([_vi("v1", _V1_TEXT, signed=True)])
    assert result.ordered_ids == ("v1",)
    assert result.signed_id == "v1"
    assert result.basis == "single"


def test_order_versions_single_unsigned() -> None:
    result = order_versions([_vi("v1", _V1_TEXT)])
    assert result.ordered_ids == ("v1",)
    assert result.signed_id is None


def test_order_versions_two_signed_last() -> None:
    """Signed version must always be the last element."""
    versions = [_vi("v2", _V2_TEXT, signed=True), _vi("v1", _V1_TEXT)]
    result = order_versions(versions)
    assert result.ordered_ids[-1] == "v2"


def test_order_versions_three_correct_order() -> None:
    """Core acceptance test: reconstructs v1→v2→v3(signed) without timestamps."""
    versions = [
        _vi("v3", _V3_TEXT, signed=True),
        _vi("v1", _V1_TEXT),
        _vi("v2", _V2_TEXT),
    ]
    result = order_versions(versions)
    assert result.ordered_ids == ("v1", "v2", "v3"), f"Expected v1→v2→v3, got {result.ordered_ids}"
    assert result.signed_id == "v3"


def test_order_versions_basis_exhaustive_for_small_n() -> None:
    versions = [
        _vi("v3", _V3_TEXT, signed=True),
        _vi("v1", _V1_TEXT),
        _vi("v2", _V2_TEXT),
    ]
    result = order_versions(versions)
    assert result.basis == "exhaustive"


def test_order_versions_pairwise_distances_populated() -> None:
    versions = [_vi("v1", _V1_TEXT), _vi("v2", _V2_TEXT, signed=True)]
    result = order_versions(versions)
    assert len(result.pairwise_distances) == 1
    assert isinstance(result.pairwise_distances[0], PairwiseDistance)
    assert 0.0 <= result.pairwise_distances[0].distance <= 1.0


def test_order_versions_total_distance_sum_of_pairs() -> None:
    versions = [
        _vi("v1", _V1_TEXT),
        _vi("v2", _V2_TEXT),
        _vi("v3", _V3_TEXT, signed=True),
    ]
    result = order_versions(versions)
    expected = sum(p.distance for p in result.pairwise_distances)
    assert abs(result.total_distance - expected) < 1e-9


def test_order_versions_no_signed_still_produces_chain() -> None:
    """When no version is signed, ordering still produces a chain."""
    versions = [_vi("v1", _V1_TEXT), _vi("v2", _V2_TEXT), _vi("v3", _V3_TEXT)]
    result = order_versions(versions)
    assert len(result.ordered_ids) == 3
    assert result.signed_id is None


# ---------------------------------------------------------------------------
# VersionOrder.to_dict (trail output)
# ---------------------------------------------------------------------------


def test_to_dict_keys() -> None:
    versions = [_vi("v1", _V1_TEXT), _vi("v2", _V2_TEXT, signed=True)]
    result = order_versions(versions)
    d = result.to_dict()
    assert "ordered_versions" in d
    assert "signed_version" in d
    assert "basis" in d
    assert "total_distance" in d
    assert "pairwise_distances" in d


def test_to_dict_json_serialisable(tmp_path: Path) -> None:
    versions = [
        _vi("v1", _V1_TEXT),
        _vi("v2", _V2_TEXT),
        _vi("v3", _V3_TEXT, signed=True),
    ]
    result = order_versions(versions)
    dest = tmp_path / "trail.json"
    dest.write_text(json.dumps(result.to_dict()), encoding="utf-8")
    loaded = json.loads(dest.read_text(encoding="utf-8"))
    assert loaded["ordered_versions"] == list(result.ordered_ids)


def test_to_dict_pairwise_distances_entries() -> None:
    versions = [_vi("v1", _V1_TEXT), _vi("v2", _V2_TEXT, signed=True)]
    result = order_versions(versions)
    d = result.to_dict()
    assert len(d["pairwise_distances"]) == 1
    entry = d["pairwise_distances"][0]
    assert "from" in entry
    assert "to" in entry
    assert "distance" in entry


# ---------------------------------------------------------------------------
# Hints
# ---------------------------------------------------------------------------


def test_hints_load_missing_file(tmp_path: Path) -> None:
    hints = Hints.load(tmp_path / "nofile.yaml")
    assert hints.order is None
    assert hints.timestamps == {}
    assert hints.signed_version is None
    assert hints.provenance is None


def test_hints_load_valid(tmp_path: Path) -> None:
    yaml_content = (
        "order:\n  - v1\n  - v2\n  - v3\ntimestamps:\n  v1: '2025-01-01'\n  v2: '2025-01-10'\n"
    )
    p = tmp_path / "hints.yaml"
    p.write_text(yaml_content, encoding="utf-8")
    hints = Hints.load(p)
    assert hints.order == ["v1", "v2", "v3"]
    assert hints.timestamps["v1"] == "2025-01-01"
    assert hints.signed_version is None
    assert hints.provenance is None


def test_hints_load_signed_version(tmp_path: Path) -> None:
    """signed_version key in hints.yaml is parsed into Hints.signed_version."""
    yaml_content = "signed_version: v3\n"
    p = tmp_path / "hints.yaml"
    p.write_text(yaml_content, encoding="utf-8")
    hints = Hints.load(p)
    assert hints.signed_version == "v3"
    assert hints.provenance is None


def test_hints_load_provenance(tmp_path: Path) -> None:
    """provenance key in hints.yaml is parsed into Hints.provenance."""
    yaml_content = "provenance: our_paper\n"
    p = tmp_path / "hints.yaml"
    p.write_text(yaml_content, encoding="utf-8")
    hints = Hints.load(p)
    assert hints.provenance == "our_paper"


def test_strip_hint_ext_preserves_dotted_stem() -> None:
    """A dotted version stem (date/form number) must not be truncated (issue #182)."""
    from playbook_engine.version_orderer import _strip_hint_ext

    # Real document extensions ARE stripped.
    assert _strip_hint_ext("fully-executed.pdf") == "fully-executed"
    assert _strip_hint_ext("01__Agreement.docx") == "01__Agreement"
    # A dotted stem WITHOUT a real extension is preserved verbatim — Path.stem
    # alone would eat the trailing ".25"/".23".
    assert _strip_hint_ext("01__Form_01.29.25") == "01__Form_01.29.25"
    assert _strip_hint_ext("03__Deal 6.14.23") == "03__Deal 6.14.23"
    # A dotted stem WITH a real extension loses only the extension.
    assert _strip_hint_ext("03__Deal 6.14.23.pdf") == "03__Deal 6.14.23"


def test_hints_load_dotted_names_not_truncated(tmp_path: Path) -> None:
    """order/signed_version hints for dotted filenames survive load (issue #182).

    Regression: version ids legitimately contain dots (dates like 6.14.23, form
    numbers like Form_01.29.25). A double extension-strip truncated the hint at
    the first dot from the right, so it matched no discovered version and was
    silently ignored — signed-copy + ordering hints had no effect.
    """
    yaml_content = (
        'order:\n  - "01__Form_01.29.25"\n  - "02__Form_01.29.25"\n'
        'signed_version: "02__Form_01.29.25"\n'
    )
    p = tmp_path / "hints.yaml"
    p.write_text(yaml_content, encoding="utf-8")
    hints = Hints.load(p)
    assert hints.order == ["01__Form_01.29.25", "02__Form_01.29.25"]
    assert hints.signed_version == "02__Form_01.29.25"


def test_hints_load_all_fields(tmp_path: Path) -> None:
    """All four hint keys are parsed together without interference."""
    yaml_content = (
        "order:\n  - v1\n  - v2\n"
        "timestamps:\n  v1: '2025-01-01'\n"
        "signed_version: v2\n"
        "provenance: counterparty_paper\n"
    )
    p = tmp_path / "hints.yaml"
    p.write_text(yaml_content, encoding="utf-8")
    hints = Hints.load(p)
    assert hints.order == ["v1", "v2"]
    assert hints.timestamps["v1"] == "2025-01-01"
    assert hints.signed_version == "v2"
    assert hints.provenance == "counterparty_paper"


# ---------------------------------------------------------------------------
# Issue #84: hints.yaml documented WITH extensions (docs/CORPUS-LAYOUT.md)
# must still match version ids, which are file STEMS (vf.stem); and a
# malformed hints.yaml must not silently discard the lawyer's corrections.
# ---------------------------------------------------------------------------


def test_hints_load_strips_extension_from_order(tmp_path: Path) -> None:
    """order entries written WITH extensions (as CORPUS-LAYOUT.md documents)
    must be normalised to bare stems so they match real version ids.
    """
    yaml_content = (
        "order:\n  - draft-we-sent.docx\n  - their-redline.docx\n  - fully-executed.pdf\n"
    )
    p = tmp_path / "hints.yaml"
    p.write_text(yaml_content, encoding="utf-8")
    hints = Hints.load(p)
    assert hints.order == ["draft-we-sent", "their-redline", "fully-executed"]


def test_hints_load_strips_extension_from_signed_version(tmp_path: Path) -> None:
    """signed_version written WITH an extension (as documented) must be
    normalised to the bare stem so it anchors the correct version_id.
    """
    yaml_content = "signed_version: fully-executed.pdf\n"
    p = tmp_path / "hints.yaml"
    p.write_text(yaml_content, encoding="utf-8")
    hints = Hints.load(p)
    assert hints.signed_version == "fully-executed"


def test_hints_load_no_extension_still_works(tmp_path: Path) -> None:
    """Bare stems (no extension) continue to load unchanged — stripping an
    extension that isn't there is a no-op.
    """
    yaml_content = "signed_version: v3\norder:\n  - v1\n  - v2\n"
    p = tmp_path / "hints.yaml"
    p.write_text(yaml_content, encoding="utf-8")
    hints = Hints.load(p)
    assert hints.signed_version == "v3"
    assert hints.order == ["v1", "v2"]


def test_hints_load_malformed_yaml_raises(tmp_path: Path) -> None:
    """Invalid YAML must raise HintsError, not silently return empty Hints —
    a typo in the lawyer's correction file must never disappear unnoticed.
    """
    p = tmp_path / "hints.yaml"
    p.write_text("order: [v1, v2\nsigned_version: v3\n", encoding="utf-8")  # unbalanced bracket
    with pytest.raises(HintsError):
        Hints.load(p)


def test_hints_load_non_mapping_raises(tmp_path: Path) -> None:
    """A hints.yaml that parses but isn't a mapping (e.g. a bare list) must
    raise HintsError rather than being silently ignored.
    """
    p = tmp_path / "hints.yaml"
    p.write_text("- v1\n- v2\n", encoding="utf-8")
    with pytest.raises(HintsError):
        Hints.load(p)


def test_hints_load_non_list_order_raises(tmp_path: Path) -> None:
    """order must be a list; a scalar value is malformed and must raise."""
    p = tmp_path / "hints.yaml"
    p.write_text("order: v1\n", encoding="utf-8")
    with pytest.raises(HintsError):
        Hints.load(p)


def test_hints_load_empty_file_returns_empty_hints(tmp_path: Path) -> None:
    """An existing-but-empty hints.yaml (yaml.safe_load returns None) is not
    malformed — it's a valid, empty document — so it must return empty Hints,
    not raise.
    """
    p = tmp_path / "hints.yaml"
    p.write_text("", encoding="utf-8")
    hints = Hints.load(p)
    assert hints.order is None
    assert hints.signed_version is None


def test_hints_timestamps_tiebreak() -> None:
    """Timestamps should break a tie between equally costly orderings."""
    # Construct two identical texts — distances will all be 0.
    identical = "Body text that is the same in all versions.\n"
    versions = [
        _vi("v1", identical, timestamp="2025-01-01"),
        _vi("v2", identical, timestamp="2025-01-10"),
        _vi("v3", identical, signed=True, timestamp="2025-01-20"),
    ]
    hints = Hints(timestamps={"v1": "2025-01-01", "v2": "2025-01-10", "v3": "2025-01-20"})
    result = order_versions(versions, hints=hints)
    # Signed must still be last.
    assert result.ordered_ids[-1] == "v3"
    # Order consistent with timestamps.
    idx_v1 = result.ordered_ids.index("v1")
    idx_v2 = result.ordered_ids.index("v2")
    assert idx_v1 < idx_v2


# ---------------------------------------------------------------------------
# Constant
# ---------------------------------------------------------------------------


def test_exhaustive_threshold_positive() -> None:
    assert EXHAUSTIVE_THRESHOLD > 0


# ---------------------------------------------------------------------------
# Regression: B1 — VersionInput.timestamp must seed ordering without Hints
# ---------------------------------------------------------------------------


def test_b1_version_input_timestamp_seeds_tiebreak() -> None:
    """VersionInput.timestamp must influence ordering even without explicit Hints.

    Before B1 fix: the timestamp field was documented as seeding tie-breaking
    but order_versions never read it, so identical-content versions were ordered
    by input position only.
    """
    identical = "Body text that is the same in all versions.\n"
    # Supply timestamps directly on VersionInput — no explicit Hints.
    versions = [
        _vi("vLate", identical, timestamp="2025-03-01"),
        _vi("vEarly", identical, timestamp="2025-01-01"),
        _vi("vSigned", identical, signed=True, timestamp="2025-06-01"),
    ]
    result = order_versions(versions)
    # Signed must still be last.
    assert result.ordered_ids[-1] == "vSigned"
    # vEarly (earlier timestamp) must appear before vLate.
    idx_early = result.ordered_ids.index("vEarly")
    idx_late = result.ordered_ids.index("vLate")
    assert idx_early < idx_late, f"Expected vEarly before vLate, got {result.ordered_ids}"


# ---------------------------------------------------------------------------
# chain_shape helper
# ---------------------------------------------------------------------------


def _uniform_dist(ids: list[str], cost: float) -> dict[tuple[str, str], float]:
    """Return a symmetric distance dict where every pair has the same cost."""
    dist: dict[tuple[str, str], float] = {}
    for a in ids:
        for b in ids:
            dist[(a, b)] = 0.0 if a == b else cost
    return dist


def test_chain_shape_linear_uniform_costs() -> None:
    """Uniform pairwise costs → shape is 'linear'."""
    ids = ["v1", "v2", "v3"]
    dist = _uniform_dist(ids, 0.1)
    assert chain_shape(ids, dist) == "linear"


def test_chain_shape_linear_single_pair() -> None:
    """Only two versions (one step) with low cost → cannot be bimodal → 'linear'."""
    ids = ["v1", "v2"]
    dist = {("v1", "v2"): 0.1, ("v2", "v1"): 0.1, ("v1", "v1"): 0.0, ("v2", "v2"): 0.0}
    assert chain_shape(ids, dist) == "linear"


def test_chain_shape_gap_large_single_step() -> None:
    """A single step cost above GAP_THRESHOLD → shape is 'gap'."""
    ids = ["v1", "v2", "v3"]
    # v1→v2: small; v2→v3: very large (> 0.4)
    dist: dict[tuple[str, str], float] = {
        ("v1", "v2"): 0.05,
        ("v2", "v1"): 0.05,
        ("v2", "v3"): 0.8,
        ("v3", "v2"): 0.8,
        ("v1", "v3"): 0.8,
        ("v3", "v1"): 0.8,
        ("v1", "v1"): 0.0,
        ("v2", "v2"): 0.0,
        ("v3", "v3"): 0.0,
    }
    assert chain_shape(ids, dist) == "gap"


def test_chain_shape_fork_bimodal_costs() -> None:
    """Bimodal cost distribution → shape is 'fork'.

    Three steps: two cheap (0.05) and one very expensive (0.9).
    The gap between clusters is 0.9 - 0.05 = 0.85; median = 0.05 (middle of sorted).
    0.85 / 0.05 = 17 >> FORK_GAP_RATIO=2.0.

    NOTE: 'gap' is checked first in chain_shape, but since any cost > 0.4 triggers
    'gap', we need bimodal costs where the expensive step is <= 0.4 to see 'fork'.
    Use costs 0.05, 0.05, 0.35 so no single step exceeds GAP_THRESHOLD=0.4.
    Gap between clusters: 0.35 - 0.05 = 0.30; median = 0.05; ratio = 6.0 >> 2.0.
    """
    ids = ["v1", "v2", "v3", "v4"]
    # Steps: v1→v2=0.05, v2→v3=0.05, v3→v4=0.35
    dist: dict[tuple[str, str], float] = {}
    for a in ids:
        for b in ids:
            dist[(a, b)] = 0.0
    dist[("v1", "v2")] = dist[("v2", "v1")] = 0.05
    dist[("v2", "v3")] = dist[("v3", "v2")] = 0.05
    dist[("v3", "v4")] = dist[("v4", "v3")] = 0.35
    # Non-adjacent pairs (not used by chain_shape but required by _all_distances contract)
    dist[("v1", "v3")] = dist[("v3", "v1")] = 0.1
    dist[("v1", "v4")] = dist[("v4", "v1")] = 0.4
    dist[("v2", "v4")] = dist[("v4", "v2")] = 0.38
    assert chain_shape(ids, dist) == "fork"


def test_chain_shape_single_id() -> None:
    """Single-element chain → 'linear' (no steps)."""
    assert chain_shape(["v1"], {("v1", "v1"): 0.0}) == "linear"


def test_chain_shape_empty() -> None:
    """Empty chain → 'linear'."""
    assert chain_shape([], {}) == "linear"


# ---------------------------------------------------------------------------
# VersionOrder.shape field
# ---------------------------------------------------------------------------


def test_version_order_has_shape_field() -> None:
    """VersionOrderResult carries a shape field."""
    versions = [
        _vi("v3", _V3_TEXT, signed=True),
        _vi("v1", _V1_TEXT),
        _vi("v2", _V2_TEXT),
    ]
    result = order_versions(versions)
    assert hasattr(result, "shape")
    assert result.shape in ("linear", "fork", "gap")


def test_to_dict_includes_shape() -> None:
    """trail/<doc>.json includes 'shape' field."""
    versions = [_vi("v1", _V1_TEXT), _vi("v2", _V2_TEXT, signed=True)]
    result = order_versions(versions)
    d = result.to_dict()
    assert "shape" in d
    assert d["shape"] in ("linear", "fork", "gap")


def test_version_order_uniform_costs_shape_linear() -> None:
    """Fixture with uniform pairwise costs → shape=='linear'; judge not called."""
    call_log: list[str] = []

    class _StubJudge:
        def judge(
            self,
            version_summaries: list[str],
            pairwise_distances: dict,
            current_order: VersionOrder,
        ) -> VersionOrder:
            call_log.append("called")
            return current_order

    # Identical content → all pairwise costs zero (uniform).
    identical = "Same text in every version for uniform cost test.\n"
    versions = [
        _vi("va", identical),
        _vi("vb", identical),
        _vi("vc", identical, signed=True),
    ]
    result = order_versions(versions, trail_judge=_StubJudge())
    assert result.shape == "linear"
    # Judge must NOT be called when shape is linear and basis is not greedy.
    assert call_log == [], f"Judge was called unexpectedly: {call_log}"


def test_trail_judge_called_on_greedy_basis() -> None:
    """Fixture with basis=='greedy' → judge called with non-empty version_summaries."""
    call_log: list[list[str]] = []

    class _StubJudge:
        def judge(
            self,
            version_summaries: list[str],
            pairwise_distances: dict,
            current_order: VersionOrder,
        ) -> VersionOrder:
            call_log.append(version_summaries)
            return current_order

    # Build EXHAUSTIVE_THRESHOLD + 2 non-signed versions so the greedy path fires.
    # Each version gets distinct synthetic text so distances are non-trivial.
    base = "Clause text for version {n}. Unique content line {n} here.\n"
    versions = [_vi(f"vg{i}", base.format(n=i)) for i in range(EXHAUSTIVE_THRESHOLD + 2)] + [
        _vi("vgSigned", _V3_TEXT, signed=True)
    ]

    result = order_versions(versions, trail_judge=_StubJudge())
    assert result.basis == "greedy"
    assert len(call_log) == 1, f"Expected judge called once, got {len(call_log)}"
    summaries = call_log[0]
    assert len(summaries) > 0, "Judge received empty version_summaries"


def test_trail_judge_called_on_fork_shape() -> None:
    """Fixture with bimodal costs (fork shape) → judge called."""
    call_log: list[str] = []

    class _StubJudge:
        def judge(
            self,
            version_summaries: list[str],
            pairwise_distances: dict,
            current_order: VersionOrder,
        ) -> VersionOrder:
            call_log.append("called")
            return current_order

    # v1 and v2 are very similar; v3 (signed) is very different from v2.
    # This should produce a large cost jump that triggers fork or gap shape.
    v1_text = "Short agreement template clause.\nBasic terms apply.\n"
    v2_text = "Short agreement template clause.\nBasic terms apply.\nMinor edit.\n"
    # v3_text is the full V3 text with signatures — very different from v1/v2.
    versions = [
        _vi("vf1", v1_text),
        _vi("vf2", v2_text),
        _vi("vf3", _V3_TEXT, signed=True),
    ]
    result = order_versions(versions, trail_judge=_StubJudge())
    # Shape must be non-linear (fork or gap) AND judge must have been called.
    assert result.shape in ("fork", "gap"), f"Expected non-linear shape, got {result.shape!r}"
    assert call_log == ["called"], "Judge was not called on non-linear chain"
