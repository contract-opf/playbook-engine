"""Tests for the scope gate (L1b).

SECURITY NOTE: All fixtures use programmatically constructed ClauseTree
objects with synthetic text.  No real agreement files are referenced.
Party names use fictional identifiers only ("Alice Corp", "Beta Ltd",
"Party A").
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from playbook_engine.clause_tree import ClauseNode, ClauseTree
from playbook_engine.config import AgreementType
from playbook_engine.scope_gate import (
    MIN_CLAUSE_COUNT,
    ScopeDecision,
    ScopeJudge,
    ScopeLog,
    scope_gate,
    scope_gate_and_record,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_AFFILIATION_TYPE = AgreementType(
    id="educational-affiliation",
    name="Educational Affiliation Agreement",
)


def _node(path: str, heading: str | None = None, text: str = "") -> ClauseNode:
    return ClauseNode(
        clause_path=path,
        heading=heading,
        text=text,
        char_span=(0, max(1, len(heading or ""))),
    )


def _tree(*nodes: ClauseNode, doc_id: str = "doc") -> ClauseTree:
    return ClauseTree(document_id=doc_id, version="v1", source_file="doc.docx", nodes=list(nodes))


# ---------------------------------------------------------------------------
# MockScopeJudge — deterministic judge for tests
# ---------------------------------------------------------------------------

# Keyword sets that indicate agreement type for the synthetic fixtures.
# NOTE: _DATA_SHARING_KEYWORDS is checked first, so affiliation fixtures must
# not contain data-sharing terms.
_AFFILIATION_KEYWORDS = {
    "affiliation",
    "student",
    "clinical",
    "rotation",
    "placement",
    "intern",
    "preceptor",
    "healthcare",
    "hospital",
}
_DATA_SHARING_KEYWORDS = {
    "data sharing",
    "data processing",
    "gdpr",
    "personal data",
    "controller",
    "processor",
    "data subject",
}


class MockScopeJudge:
    """Keyword-based scope judge for tests.

    Classifies a document as in-scope for an affiliation agreement if it
    contains affiliation-related keywords; returns out-of-scope for documents
    containing data-sharing-specific keywords.
    """

    def judge(self, tree: ClauseTree, agreement_type: AgreementType) -> ScopeDecision:
        full_text = " ".join(
            (node.heading or "") + " " + (node.text or "") for node in tree.all_nodes()
        ).lower()

        # Off-type: data sharing agreement keywords.
        if any(kw in full_text for kw in _DATA_SHARING_KEYWORDS):
            return ScopeDecision(
                in_scope=False,
                scope_rationale=(
                    "Document appears to be a data-sharing/data-processing agreement, "
                    f"not a {agreement_type.name}. "
                    "Clause profile contains data-processing terms (GDPR, controller/processor)."
                ),
                scope_confidence=0.88,
                basis="judge",
            )

        # In-scope: affiliation-related keywords present.
        if any(kw in full_text for kw in _AFFILIATION_KEYWORDS):
            return ScopeDecision(
                in_scope=True,
                scope_rationale=(
                    f"Document matches expected clause profile of a {agreement_type.name}."
                ),
                scope_confidence=0.90,
                basis="judge",
            )

        # Borderline: neither set of keywords found — accept with lower confidence.
        return ScopeDecision(
            in_scope=True,
            scope_rationale=(
                "Document clause profile is borderline; no off-type keywords detected. "
                f"Treated as in-scope {agreement_type.name} at reduced confidence."
            ),
            scope_confidence=0.60,
            basis="judge",
        )


class BadBasisJudge:
    """Judge that returns a reserved pre-check basis — for error-path testing.

    A judge returning a ``deterministic_*`` basis is a genuine programming
    error: those values are reserved for ``scope_gate``'s own pre-checks.
    (``judge_error`` is NOT a bad basis — it is a legitimate sentinel produced
    by caching wrappers such as ``BatchedScopeJudge``.)
    """

    def judge(self, tree: ClauseTree, agreement_type: AgreementType) -> ScopeDecision:
        return ScopeDecision(
            in_scope=True,
            scope_rationale="Deliberate reserved pre-check basis for testing.",
            scope_confidence=0.80,
            basis="deterministic_empty",  # reserved for scope_gate — must NOT come from a judge
        )


class RaisingJudge:
    """Judge that always raises — simulates LLM timeout / network failure."""

    def judge(self, tree: ClauseTree, agreement_type: AgreementType) -> ScopeDecision:
        raise RuntimeError("LLM service unavailable")


class StubScopeJudge:
    """Judge stand-in for a no-LLM stub (e.g. ``_AllInScopeJudge``) — returns
    ``basis="stub"`` rather than ``basis="judge"``."""

    def judge(self, tree: ClauseTree, agreement_type: AgreementType) -> ScopeDecision:
        return ScopeDecision(
            in_scope=True,
            scope_rationale="Accepted without LLM judgment (stub mode).",
            scope_confidence=0.5,
            basis="stub",
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _affiliation_tree() -> ClauseTree:
    return _tree(
        _node("0", None, "This Affiliation Agreement is between Alice Corp and Beta Hospital."),
        _node("1", "Purpose", "This agreement governs the clinical rotation of students."),
        _node("2", "Student Placement", "Alice Corp places interns at Beta Hospital."),
        _node("3", "Preceptor", "A qualified preceptor shall supervise each student."),
        _node("4", "Term", "This agreement lasts one year."),
        _node("5", "Termination", "Either party may terminate on thirty days notice."),
    )


def _data_sharing_tree() -> ClauseTree:
    return _tree(
        _node("0", None, "This Data Sharing Agreement governs the exchange of personal data."),
        _node("1", "Definitions", '"Personal Data" has the meaning given in the GDPR.'),
        _node("2", "Controller", "Alice Corp acts as the data controller."),
        _node("3", "Processor", "Beta Ltd acts as the data processor."),
        _node(
            "4", "Data Subject Rights", "Data subjects may exercise rights under applicable law."
        ),
        _node("5", "Security", "Each party shall maintain appropriate technical measures."),
    )


def _borderline_tree() -> ClauseTree:
    """Generic services agreement — no off-type keywords, borderline acceptable."""
    return _tree(
        _node("0", None, "This Agreement governs services between Party A and Party B."),
        _node("1", "Services", "Party A shall provide professional services."),
        _node("2", "Payment", "Party B shall pay within thirty days."),
        _node("3", "Term", "This agreement lasts for the duration of the project."),
    )


# ---------------------------------------------------------------------------
# ScopeDecision dataclass
# ---------------------------------------------------------------------------


def test_scope_decision_fields() -> None:
    d = ScopeDecision(
        in_scope=True, scope_rationale="Matches.", scope_confidence=0.90, basis="judge"
    )
    assert d.in_scope is True
    assert d.scope_rationale == "Matches."
    assert d.scope_confidence == 0.90
    assert d.basis == "judge"


def test_scope_decision_frozen() -> None:
    d = ScopeDecision(in_scope=True, scope_rationale="R.", scope_confidence=0.9, basis="judge")
    with pytest.raises((AttributeError, TypeError)):
        d.in_scope = False  # type: ignore[misc]


def test_scope_decision_invalid_basis() -> None:
    with pytest.raises(ValueError, match="Unknown basis"):
        ScopeDecision(in_scope=True, scope_rationale="R.", scope_confidence=0.9, basis="bad")


def test_scope_decision_confidence_out_of_range() -> None:
    with pytest.raises(ValueError, match="scope_confidence"):
        ScopeDecision(in_scope=True, scope_rationale="R.", scope_confidence=1.5, basis="judge")


def test_scope_decision_empty_rationale() -> None:
    with pytest.raises(ValueError, match="scope_rationale"):
        ScopeDecision(in_scope=True, scope_rationale="   ", scope_confidence=0.9, basis="judge")


def test_scope_decision_judge_error_basis_valid() -> None:
    d = ScopeDecision(
        in_scope=False,
        scope_rationale="Judge raised RuntimeError: timeout",
        scope_confidence=0.0,
        basis="judge_error",
    )
    assert d.basis == "judge_error"


def test_scope_decision_stub_basis_valid() -> None:
    """issue #101: "stub" is a distinct, valid basis — not "judge"."""
    d = ScopeDecision(
        in_scope=True,
        scope_rationale="Accepted without LLM judgment (stub mode).",
        scope_confidence=0.5,
        basis="stub",
    )
    assert d.basis == "stub"


def test_scope_decision_to_dict() -> None:
    d = ScopeDecision(
        in_scope=False, scope_rationale="Wrong type.", scope_confidence=0.88, basis="judge"
    )
    result = d.to_dict()
    assert result["in_scope"] is False
    assert result["scope_rationale"] == "Wrong type."
    assert "scope_confidence" in result
    assert result["basis"] == "judge"


# ---------------------------------------------------------------------------
# ScopeJudge protocol
# ---------------------------------------------------------------------------


def test_mock_judge_is_scope_judge_protocol() -> None:
    judge = MockScopeJudge()
    assert isinstance(judge, ScopeJudge)


# ---------------------------------------------------------------------------
# scope_gate: deterministic pre-checks
# ---------------------------------------------------------------------------


def test_scope_gate_empty_tree_excluded() -> None:
    tree = ClauseTree(document_id="empty", version="v1", source_file="empty.docx")
    decision = scope_gate(tree, _AFFILIATION_TYPE, MockScopeJudge())
    assert decision.in_scope is False
    assert decision.basis == "deterministic_empty"
    assert decision.scope_confidence >= 0.90


def test_scope_gate_trivial_tree_excluded() -> None:
    tree = _tree(_node("1", "Term", "One year."))  # only 1 node, < MIN_CLAUSE_COUNT
    decision = scope_gate(tree, _AFFILIATION_TYPE, MockScopeJudge())
    assert decision.in_scope is False
    assert decision.basis == "deterministic_trivial"


def test_scope_gate_trivial_rationale_mentions_count() -> None:
    tree = _tree(_node("1", "Term", "One year."))
    decision = scope_gate(tree, _AFFILIATION_TYPE, MockScopeJudge())
    assert "1" in decision.scope_rationale


def test_scope_gate_empty_rationale_present() -> None:
    tree = ClauseTree(document_id="d", version="v1", source_file="f")
    decision = scope_gate(tree, _AFFILIATION_TYPE, MockScopeJudge())
    assert decision.scope_rationale.strip()  # never empty


def test_scope_gate_at_threshold_delegates() -> None:
    """MIN_CLAUSE_COUNT nodes is enough to pass pre-check and delegate to judge."""
    nodes = [_node(str(i), heading=f"Section {i}", text="Body.") for i in range(MIN_CLAUSE_COUNT)]
    tree = _tree(*nodes)
    decision = scope_gate(tree, _AFFILIATION_TYPE, MockScopeJudge())
    assert decision.basis == "judge"  # passed pre-checks, delegated to judge


# ---------------------------------------------------------------------------
# scope_gate: judge error path (§3.6 — never silently drop)
# ---------------------------------------------------------------------------


def test_scope_gate_judge_raises_returns_judge_error() -> None:
    """When the judge raises, scope_gate retains the document (in_scope=True) with judge_error basis."""
    decision = scope_gate(_affiliation_tree(), _AFFILIATION_TYPE, RaisingJudge())
    assert decision.basis == "judge_error"
    # Document must be RETAINED (not dropped) — fail-closed is the bug P1.5 fixes.
    assert decision.in_scope is True
    assert decision.scope_confidence == 0.0
    assert "LLM service unavailable" in decision.scope_rationale


def test_scope_gate_judge_error_rationale_not_empty() -> None:
    """§3.6: judge_error decision must still carry a rationale."""
    decision = scope_gate(_affiliation_tree(), _AFFILIATION_TYPE, RaisingJudge())
    assert len(decision.scope_rationale) > 10


def test_scope_gate_judge_error_document_retained_not_dropped() -> None:
    """§P1.5 acceptance: document retained (in_scope=True) on judge error — not silently deleted."""
    decision = scope_gate(_affiliation_tree(), _AFFILIATION_TYPE, RaisingJudge())
    assert decision.in_scope is True
    assert decision.basis == "judge_error"


def test_scope_log_judge_error_ids() -> None:
    """judge_error_ids must expose retained-but-flagged documents for routing."""
    log = ScopeLog(agreement_type_id="test")
    scope_gate_and_record(_affiliation_tree(), "doc-ok", _AFFILIATION_TYPE, MockScopeJudge(), log)
    scope_gate_and_record(_affiliation_tree(), "doc-err", _AFFILIATION_TYPE, RaisingJudge(), log)
    assert "doc-err" in log.judge_error_ids
    assert "doc-ok" not in log.judge_error_ids
    # judge_error doc is retained → not in out_of_scope_ids
    assert "doc-err" not in log.out_of_scope_ids
    # judge_error doc is NOT in clean in_scope_ids (it's quarantined for review)
    assert "doc-err" not in log.in_scope_ids


def test_scope_gate_bad_basis_raises() -> None:
    """Judge returning a reserved pre-check basis raises ValueError (programming error)."""
    with pytest.raises(ValueError, match="reserved pre-check basis"):
        scope_gate(_affiliation_tree(), _AFFILIATION_TYPE, BadBasisJudge())


def test_scope_gate_wrapped_raising_delegate_returns_judge_error(tmp_path: Path) -> None:
    """A raising delegate wrapped in BatchedScopeJudge (the no_cache=False path)
    surfaces basis='judge_error' as a NORMAL return; scope_gate must accept that
    sentinel and retain the document rather than crashing the compile.

    Regression: scope_gate's basis guard previously rejected 'judge_error',
    raising ValueError on legitimate wrapper output.
    """
    from playbook_engine.judgment import BatchedScopeJudge, JudgmentCache

    cache = JudgmentCache(tmp_path / "verdicts.jsonl", model_id="test-v1")
    wrapped = BatchedScopeJudge(delegate=RaisingJudge(), cache=cache)

    decision = scope_gate(_affiliation_tree(), _AFFILIATION_TYPE, wrapped)

    assert decision.basis == "judge_error"
    assert decision.in_scope is True  # retained, not dropped
    assert decision.scope_confidence == 0.0
    assert decision.scope_rationale.strip()


# ---------------------------------------------------------------------------
# scope_gate: stub basis (issue #101) — distinct from a real judge's "judge"
# ---------------------------------------------------------------------------


def test_scope_gate_stub_judge_returns_stub_basis() -> None:
    """A stub judge (no LLM configured) must surface basis='stub', not 'judge' —
    downstream consumers need to tell a rubber-stamped decision from a real one."""
    decision = scope_gate(_affiliation_tree(), _AFFILIATION_TYPE, StubScopeJudge())
    assert decision.basis == "stub"
    assert decision.in_scope is True


def test_scope_gate_stub_basis_does_not_raise() -> None:
    """basis='stub' is a legitimate judge return value — the programming-error
    guard must accept it alongside 'judge'."""
    # No exception should propagate.
    scope_gate(_affiliation_tree(), _AFFILIATION_TYPE, StubScopeJudge())


def test_scope_gate_and_record_stub_basis_logged(tmp_path: Path) -> None:
    """A stub scope decision is recorded in the log/scope.json like any other,
    with its basis preserved for audit."""
    log = ScopeLog(agreement_type_id="test")
    scope_gate_and_record(_affiliation_tree(), "doc-stub", _AFFILIATION_TYPE, StubScopeJudge(), log)
    dest = tmp_path / "scope.json"
    log.write(dest)
    loaded = json.loads(dest.read_text(encoding="utf-8"))
    entry = next(e for e in loaded["documents"] if e["document_id"] == "doc-stub")
    assert entry["basis"] == "stub"
    assert entry["in_scope"] is True


# ---------------------------------------------------------------------------
# scope_gate: acceptance test (LLM judge path via MockScopeJudge)
# ---------------------------------------------------------------------------


def test_acceptance_affiliation_doc_in_scope() -> None:
    """Core acceptance: an affiliation agreement is classified in-scope."""
    decision = scope_gate(_affiliation_tree(), _AFFILIATION_TYPE, MockScopeJudge())
    assert decision.in_scope is True
    assert decision.basis == "judge"
    assert decision.scope_rationale.strip()


def test_acceptance_data_sharing_doc_out_of_scope() -> None:
    """Core acceptance: a data-sharing agreement is excluded with a rationale."""
    decision = scope_gate(_data_sharing_tree(), _AFFILIATION_TYPE, MockScopeJudge())
    assert decision.in_scope is False
    assert decision.basis == "judge"
    assert "data" in decision.scope_rationale.lower()


def test_acceptance_data_sharing_rationale_logged() -> None:
    """Out-of-scope docs must carry a scope_rationale — never silently dropped."""
    decision = scope_gate(_data_sharing_tree(), _AFFILIATION_TYPE, MockScopeJudge())
    assert len(decision.scope_rationale) > 10  # non-trivial explanation


def test_acceptance_borderline_doc_kept() -> None:
    """A borderline-but-valid doc is accepted (reduced confidence)."""
    decision = scope_gate(_borderline_tree(), _AFFILIATION_TYPE, MockScopeJudge())
    assert decision.in_scope is True
    assert decision.basis == "judge"
    assert decision.scope_confidence < 0.85  # reduced confidence, but still in scope


def test_acceptance_confidence_in_range() -> None:
    for tree in [_affiliation_tree(), _data_sharing_tree(), _borderline_tree()]:
        d = scope_gate(tree, _AFFILIATION_TYPE, MockScopeJudge())
        assert 0.0 <= d.scope_confidence <= 1.0


# ---------------------------------------------------------------------------
# §3.6 acceptance: out-of-scope doc must survive into the log with rationale
# ---------------------------------------------------------------------------


def test_s36_out_of_scope_doc_survives_in_log(tmp_path: Path) -> None:
    """OPF §3.6: out-of-scope document must appear in scope.json with its rationale."""
    log = ScopeLog(agreement_type_id="educational-affiliation")
    scope_gate_and_record(_affiliation_tree(), "doc-in", _AFFILIATION_TYPE, MockScopeJudge(), log)
    scope_gate_and_record(_data_sharing_tree(), "doc-out", _AFFILIATION_TYPE, MockScopeJudge(), log)
    dest = tmp_path / "scope.json"
    log.write(dest)

    loaded = json.loads(dest.read_text(encoding="utf-8"))
    doc_out_entry = next(e for e in loaded["documents"] if e["document_id"] == "doc-out")
    assert not doc_out_entry["in_scope"]
    assert len(doc_out_entry["scope_rationale"]) > 10


def test_s36_judge_error_doc_survives_in_log(tmp_path: Path) -> None:
    """§3.6: judge-error docs must be retained in the pipeline and routable via judge_error_ids."""
    log = ScopeLog(agreement_type_id="test")
    scope_gate_and_record(_affiliation_tree(), "doc-error", _AFFILIATION_TYPE, RaisingJudge(), log)
    # Document is retained (in_scope=True), so it appears in judge_error_ids, not out_of_scope_ids.
    assert "doc-error" in log.judge_error_ids
    assert "doc-error" not in log.out_of_scope_ids
    d = log.to_dict()
    assert d["stats"]["judge_error"] == 1
    assert d["stats"]["out_of_scope"] == 0
    entry = d["documents"][0]
    assert entry["basis"] == "judge_error"
    assert entry["in_scope"] is True
    assert entry["scope_rationale"]


# ---------------------------------------------------------------------------
# scope_gate_and_record
# ---------------------------------------------------------------------------


def test_scope_gate_and_record_logs_all_decisions() -> None:
    """scope_gate_and_record must log every document regardless of decision."""
    log = ScopeLog(agreement_type_id="test")
    trees_and_ids = [
        (_affiliation_tree(), "d1"),
        (_data_sharing_tree(), "d2"),
        (_borderline_tree(), "d3"),
    ]
    for tree, doc_id in trees_and_ids:
        scope_gate_and_record(tree, doc_id, _AFFILIATION_TYPE, MockScopeJudge(), log)

    assert log.to_dict()["stats"]["total"] == 3
    recorded_ids = [e.document_id for e in log.entries]
    assert recorded_ids == ["d1", "d2", "d3"]


def test_scope_gate_and_record_returns_decision() -> None:
    log = ScopeLog(agreement_type_id="test")
    decision = scope_gate_and_record(
        _affiliation_tree(), "d1", _AFFILIATION_TYPE, MockScopeJudge(), log
    )
    assert isinstance(decision, ScopeDecision)
    assert decision.in_scope is True


# ---------------------------------------------------------------------------
# ScopeLog
# ---------------------------------------------------------------------------


def test_scope_log_record_and_ids() -> None:
    log = ScopeLog(agreement_type_id="educational-affiliation")
    d_in = scope_gate(_affiliation_tree(), _AFFILIATION_TYPE, MockScopeJudge())
    d_out = scope_gate(_data_sharing_tree(), _AFFILIATION_TYPE, MockScopeJudge())
    log.record("doc-001", d_in)
    log.record("doc-002", d_out)
    assert "doc-001" in log.in_scope_ids
    assert "doc-002" in log.out_of_scope_ids
    assert "doc-002" not in log.in_scope_ids


def test_scope_log_to_dict_keys() -> None:
    log = ScopeLog(agreement_type_id="educational-affiliation")
    log.record("doc-001", scope_gate(_affiliation_tree(), _AFFILIATION_TYPE, MockScopeJudge()))
    d = log.to_dict()
    assert "agreement_type_id" in d
    assert "stats" in d
    assert "documents" in d
    assert d["stats"]["total"] == 1


def test_scope_log_to_dict_stats() -> None:
    log = ScopeLog(agreement_type_id="educational-affiliation")
    log.record("d1", scope_gate(_affiliation_tree(), _AFFILIATION_TYPE, MockScopeJudge()))
    log.record("d2", scope_gate(_data_sharing_tree(), _AFFILIATION_TYPE, MockScopeJudge()))
    stats = log.to_dict()["stats"]
    assert stats["in_scope"] == 1
    assert stats["out_of_scope"] == 1
    assert stats["total"] == 2


def test_scope_log_document_entries_contain_document_id() -> None:
    log = ScopeLog(agreement_type_id="educational-affiliation")
    log.record("my-doc", scope_gate(_affiliation_tree(), _AFFILIATION_TYPE, MockScopeJudge()))
    docs = log.to_dict()["documents"]
    assert docs[0]["document_id"] == "my-doc"


def test_scope_log_write_and_load(tmp_path: Path) -> None:
    log = ScopeLog(agreement_type_id="educational-affiliation")
    log.record("d1", scope_gate(_affiliation_tree(), _AFFILIATION_TYPE, MockScopeJudge()))
    log.record("d2", scope_gate(_data_sharing_tree(), _AFFILIATION_TYPE, MockScopeJudge()))
    dest = tmp_path / "scope.json"
    log.write(dest)
    loaded = json.loads(dest.read_text(encoding="utf-8"))
    assert loaded["stats"]["total"] == 2
    assert any(e["document_id"] == "d2" and not e["in_scope"] for e in loaded["documents"])


def test_scope_log_write_creates_parent_dirs(tmp_path: Path) -> None:
    log = ScopeLog(agreement_type_id="test")
    log.record("d", scope_gate(_affiliation_tree(), _AFFILIATION_TYPE, MockScopeJudge()))
    dest = tmp_path / "out" / "subdir" / "scope.json"
    log.write(dest)
    assert dest.exists()


def test_scope_log_write_atomic_no_tmp_left(tmp_path: Path) -> None:
    """Atomic write must not leave a .json.tmp file behind on success."""
    log = ScopeLog(agreement_type_id="test")
    log.record("d", scope_gate(_affiliation_tree(), _AFFILIATION_TYPE, MockScopeJudge()))
    dest = tmp_path / "scope.json"
    log.write(dest)
    assert dest.exists()
    assert not dest.with_suffix(".json.tmp").exists()


def test_scope_log_empty() -> None:
    log = ScopeLog(agreement_type_id="test")
    d = log.to_dict()
    assert d["stats"]["total"] == 0
    assert d["documents"] == []


# ---------------------------------------------------------------------------
# Constant
# ---------------------------------------------------------------------------


def test_min_clause_count_positive() -> None:
    assert MIN_CLAUSE_COUNT > 0
