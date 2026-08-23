"""Tests for rubric/judgment versioning of the store-backed judges.

The hazard being closed: ``agent_judge._payload_key`` hashes clause CONTENT
only, so a change to the judging *criteria* — the taxonomy, the deviation
vocabulary, the prose rubric in the ``playbook-from-corpus`` skill — left
every previously banked verdict replaying forever. A live re-derivation
seeded ~1,444 stored verdicts and re-queued only 246; nothing could say
whether the ~1,200 replays were still valid.

Acceptance criteria verified here:

  AC-1: A verdict is stamped with the rubric it was produced under, and the
        stamp survives a store round-trip.
  AC-2: A store hit whose stamp matches the current rubric replays.
  AC-3: A store hit whose stamp does NOT match re-queues instead of replaying
        — for all four judge kinds.
  AC-4: An UNSTAMPED (pre-versioning) verdict still replays by default — no
        banked human judgment is thrown away on upgrade — but is counted and
        reported, and re-queues under --strict-rubric.
  AC-5: --accept-stale replays known-stale verdicts instead of re-queueing.
  AC-6: The derived half of the version tracks the taxonomy (classify) and the
        agreement-type definition (scope), and is identical whether digested
        from the loaded YAML or from the taxonomy embedded in an OPF document.
  AC-7: ``judge --plan-only`` reports "N stored verdicts were made under an
        older rubric".
  AC-8: ``judge-migrate`` adopts unstamped verdicts (preserving the verdict),
        and only re-stamps known-stale ones under --accept-stale.

SECURITY NOTE: every fixture here is synthetic — programmatically built
clause text, or the pre-committed ``examples/judge-fixture/`` corpus. No real
agreement content and no ``*-corpus/`` data is used.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from click.testing import CliRunner

from playbook_engine.agent_judge import (
    PendingQueue,
    ScopeNeedsReviewError,
    StoreBackedClassificationJudge,
    StoreBackedDeviationJudge,
    StoreBackedProvenanceJudge,
    StoreBackedScopeJudge,
    VerdictStore,
)
from playbook_engine.clause_tree import ClauseNode, ClauseTree
from playbook_engine.cli import cli
from playbook_engine.config import AgreementType
from playbook_engine.rubric import (
    JUDGE_KINDS,
    RUBRIC_PROMPT_VERSIONS,
    STATE_CURRENT,
    STATE_LEGACY,
    STATE_STALE,
    RubricError,
    RubricPolicy,
    RubricStamp,
    current_versions,
    rubric_version,
    taxonomy_digest,
)

_FIXTURE_DIR = Path(__file__).parent.parent / "examples" / "judge-fixture"
_CORPUS_DIR = _FIXTURE_DIR / "corpus"
_CONFIG_PATH = _FIXTURE_DIR / "config.yaml"


# ---------------------------------------------------------------------------
# Helpers — synthetic fixtures only
# ---------------------------------------------------------------------------


def _tax(*entries: tuple[str, str, str]) -> SimpleNamespace:
    """Build a duck-typed taxonomy from ``(id, label, description)`` triples."""
    return SimpleNamespace(
        entries=[
            SimpleNamespace(id=i, label=lbl, description=desc, status="active")
            for i, lbl, desc in entries
        ]
    )


_TAX_A = _tax(
    ("confidentiality", "Confidentiality", "Obligations of confidence."),
    ("governing_law", "Governing Law", "Choice of law and venue."),
)

_AGREEMENT_A = AgreementType(id="nda", name="NDA", description="Mutual NDA.", aliases=["mnda"])


def _node(heading: str, text: str, clause_path: str = "1") -> ClauseNode:
    return ClauseNode(heading=heading, text=text, clause_path=clause_path, char_span=(0, len(text)))


def _tree(document_id: str, headings: list[str]) -> ClauseTree:
    return ClauseTree(
        document_id=document_id,
        version="v1",
        source_file="v1.rtf",
        nodes=[_node(h, f"Body of {h}.", str(i + 1)) for i, h in enumerate(headings)],
    )


def _store_and_queue(tmp_path: Path) -> tuple[VerdictStore, PendingQueue]:
    return (
        VerdictStore(tmp_path / "verdicts.jsonl"),
        PendingQueue(tmp_path / "pending.jsonl"),
    )


def _queued(tmp_path: Path) -> list[dict[str, Any]]:
    path = tmp_path / "pending.jsonl"
    if not path.exists():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


# ---------------------------------------------------------------------------
# AC-6: version construction
# ---------------------------------------------------------------------------


class TestRubricVersion:
    def test_shape_is_manual_plus_derived(self) -> None:
        version = rubric_version("deviation")
        manual, _, derived = version.partition("+")
        assert manual == RUBRIC_PROMPT_VERSIONS["deviation"]
        assert len(derived) == 12 and derived.isalnum()

    def test_every_kind_has_a_version(self) -> None:
        versions = current_versions(taxonomy=_TAX_A, agreement_type=_AGREEMENT_A)
        assert set(versions) == set(JUDGE_KINDS)
        assert all(v for v in versions.values())

    def test_unknown_kind_raises(self) -> None:
        with pytest.raises(RubricError):
            rubric_version("segment")

    def test_manual_bump_changes_the_version(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A prose-rubric bump is the hand-maintained lever; it must bite."""
        before = rubric_version("provenance")
        monkeypatch.setitem(RUBRIC_PROMPT_VERSIONS, "provenance", "v99")
        assert rubric_version("provenance") != before

    def test_classify_version_tracks_taxonomy_description(self) -> None:
        """The failure the content key cannot see: same ids, different meaning."""
        reworded = _tax(
            ("confidentiality", "Confidentiality", "Obligations of confidence."),
            ("governing_law", "Governing Law", "Venue ONLY — choice of law moved out."),
        )
        assert rubric_version("classify", taxonomy=_TAX_A) != rubric_version(
            "classify", taxonomy=reworded
        )

    def test_classify_version_ignores_inactive_entries(self) -> None:
        """A compiler may not classify into an inactive entry, so its wording
        is not part of the rubric — and must not churn banked verdicts."""
        with_inactive = SimpleNamespace(
            entries=[
                *_TAX_A.entries,
                SimpleNamespace(
                    id="assignment",
                    label="Assignment",
                    description="Pruned from this playbook.",
                    status="inactive",
                ),
            ]
        )
        assert rubric_version("classify", taxonomy=with_inactive) == rubric_version(
            "classify", taxonomy=_TAX_A
        )

    def test_taxonomy_digest_agrees_across_dataclass_and_opf_dict(self) -> None:
        """The viewer digests the OPF-embedded taxonomy; mining digests the
        loaded YAML. They must agree or a reviewer correction lands stale."""
        as_opf = {
            "source": "custom",
            "entries": [
                {
                    "id": "confidentiality",
                    "label": "Confidentiality",
                    "status": "active",
                    "description": "Obligations of confidence.",
                },
                {
                    "id": "governing_law",
                    "label": "Governing Law",
                    "status": "active",
                    "description": "Choice of law and venue.",
                },
            ],
        }
        assert taxonomy_digest(as_opf) == taxonomy_digest(_TAX_A)

    def test_scope_version_tracks_agreement_type_definition(self) -> None:
        widened = AgreementType(
            id="nda", name="NDA", description="Mutual NDA.", aliases=["mnda", "cda"]
        )
        assert rubric_version("scope", agreement_type=_AGREEMENT_A) != rubric_version(
            "scope", agreement_type=widened
        )

    def test_deviation_version_is_taxonomy_independent(self) -> None:
        """Deviation judging does not consult the taxonomy, so a taxonomy edit
        must not invalidate deviation verdicts ("and only those")."""
        assert rubric_version("deviation", taxonomy=_TAX_A) == rubric_version("deviation")


# ---------------------------------------------------------------------------
# RubricPolicy
# ---------------------------------------------------------------------------


class TestRubricPolicy:
    def test_matching_stamp_is_current_and_replays(self) -> None:
        policy = RubricPolicy()
        decision = policy.evaluate("classify", "v1+aaa", "v1+aaa")
        assert (decision.state, decision.replay) == (STATE_CURRENT, True)

    def test_mismatched_stamp_is_stale_and_does_not_replay(self) -> None:
        policy = RubricPolicy()
        decision = policy.evaluate("classify", "v1+aaa", "v1+bbb")
        assert (decision.state, decision.replay) == (STATE_STALE, False)

    def test_absent_stamp_is_legacy_and_replays_by_default(self) -> None:
        policy = RubricPolicy()
        decision = policy.evaluate("classify", None, "v1+aaa")
        assert (decision.state, decision.replay) == (STATE_LEGACY, True)

    def test_strict_legacy_stops_unstamped_replay(self) -> None:
        policy = RubricPolicy(strict_legacy=True)
        assert policy.evaluate("classify", None, "v1+aaa").replay is False

    def test_accept_stale_allows_stale_replay(self) -> None:
        policy = RubricPolicy(accept_stale=True)
        assert policy.evaluate("classify", "v1+aaa", "v1+bbb").replay is True

    def test_tally_and_breakdown(self) -> None:
        policy = RubricPolicy()
        policy.evaluate("classify", "old", "new")
        policy.evaluate("classify", "old", "new")
        policy.evaluate("deviation", None, "new")
        assert policy.total(STATE_STALE) == 2
        assert policy.breakdown(STATE_STALE) == {"classify": 2}
        assert policy.format_breakdown(STATE_LEGACY) == "deviation: 1"

    def test_would_replay_does_not_tally(self) -> None:
        policy = RubricPolicy()
        assert policy.would_replay("old", "new") is False
        assert policy.counts == {}

    def test_unknown_current_version_is_not_evidence_of_staleness(self) -> None:
        """An old pending.jsonl carries no rubric_version; that is missing
        information, not a stale stamp."""
        assert RubricPolicy().would_replay("v1+aaa", None) is True


# ---------------------------------------------------------------------------
# AC-1: VerdictStore stamping
# ---------------------------------------------------------------------------


class TestVerdictStoreStamping:
    def test_stamp_round_trips_across_instances(self, tmp_path: Path) -> None:
        path = tmp_path / "v.jsonl"
        stamp = RubricStamp(kind="classify", version="v1+abc123abc123")
        VerdictStore(path).put({"stage": "classify"}, {"taxonomy_id": "x"}, rubric=stamp)

        record = VerdictStore(path).get_record({"stage": "classify"})
        assert record is not None
        assert record.verdict == {"taxonomy_id": "x"}
        assert record.rubric == stamp
        assert record.rubric_version == "v1+abc123abc123"
        assert record.rubric_kind == "classify"

    def test_unstamped_record_loads_as_legacy(self, tmp_path: Path) -> None:
        """A store written before rubric versioning must keep loading."""
        path = tmp_path / "v.jsonl"
        path.write_text('{"key":"k1","verdict":{"r":1}}\n', encoding="utf-8")
        record = VerdictStore(path).get_record_by_key("k1")
        assert record is not None
        assert record.verdict == {"r": 1}
        assert record.rubric is None

    def test_malformed_stamp_degrades_to_legacy(self, tmp_path: Path) -> None:
        path = tmp_path / "v.jsonl"
        path.write_text('{"key":"k1","verdict":{"r":1},"rubric":"nonsense"}\n', encoding="utf-8")
        record = VerdictStore(path).get_record_by_key("k1")
        assert record is not None and record.rubric is None

    def test_unstamped_put_writes_no_rubric_member(self, tmp_path: Path) -> None:
        path = tmp_path / "v.jsonl"
        VerdictStore(path).put({"a": 1}, {"v": 1})
        line = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
        assert "rubric" not in line

    def test_restamp_keeps_verdict_and_preserves_the_prior_line(self, tmp_path: Path) -> None:
        path = tmp_path / "v.jsonl"
        store = VerdictStore(path)
        store.put_by_key("k1", {"taxonomy_id": "confidentiality"})
        assert store.restamp("k1", RubricStamp(kind="classify", version="v1+new")) is True

        lines = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
        assert len(lines) == 2, "re-stamp must append, leaving an audit trail"
        assert "rubric" not in lines[0]
        assert lines[1]["rubric"] == {"kind": "classify", "version": "v1+new"}

        reloaded = VerdictStore(path).get_record_by_key("k1")
        assert reloaded is not None
        assert reloaded.verdict == {"taxonomy_id": "confidentiality"}, "verdict must survive"
        assert reloaded.rubric_version == "v1+new"

    def test_restamp_unknown_key_is_a_no_op(self, tmp_path: Path) -> None:
        assert VerdictStore(tmp_path / "v.jsonl").restamp("nope", RubricStamp("classify", "v")) is (
            False
        )

    def test_records_and_len(self, tmp_path: Path) -> None:
        store = VerdictStore(tmp_path / "v.jsonl")
        store.put_by_key("k2", {"v": 2})
        store.put_by_key("k1", {"v": 1})
        assert len(store) == 2
        assert [k for k, _ in store.records()] == ["k1", "k2"]


# ---------------------------------------------------------------------------
# AC-2/AC-3/AC-4/AC-5: judge replay behaviour, per kind
# ---------------------------------------------------------------------------


class TestClassifyJudgeRubric:
    def _seed(self, tmp_path: Path, version: str | None) -> tuple[VerdictStore, PendingQueue]:
        store, pending = _store_and_queue(tmp_path)
        payload = {
            "stage": "classify",
            "text": "Body of Confidentiality.",
            "heading": "Confidentiality",
            "taxonomy_ids": sorted(e.id for e in _TAX_A.entries),
        }
        stamp = RubricStamp(kind="classify", version=version) if version else None
        store.put(payload, {"taxonomy_id": "confidentiality", "confidence": 0.9}, rubric=stamp)
        return store, pending

    def test_current_stamp_replays(self, tmp_path: Path) -> None:
        store, pending = self._seed(tmp_path, rubric_version("classify", taxonomy=_TAX_A))
        judge = StoreBackedClassificationJudge(store=store, pending=pending)
        [result] = judge.classify_batch(
            [_node("Confidentiality", "Body of Confidentiality.")], _TAX_A
        )
        assert result.taxonomy_id == "confidentiality"
        assert judge.rubric.total(STATE_CURRENT) == 1
        assert _queued(tmp_path) == []

    def test_stale_stamp_requeues_instead_of_replaying(self, tmp_path: Path) -> None:
        store, pending = self._seed(tmp_path, "v0+deadbeefcafe")
        judge = StoreBackedClassificationJudge(store=store, pending=pending)
        [result] = judge.classify_batch(
            [_node("Confidentiality", "Body of Confidentiality.")], _TAX_A
        )
        assert result.basis == "needs_review"
        assert result.taxonomy_id is None
        assert judge.rubric.total(STATE_STALE) == 1
        queued = _queued(tmp_path)
        assert [q["kind"] for q in queued] == ["classify"]
        assert queued[0]["rubric_version"] == rubric_version("classify", taxonomy=_TAX_A)

    def test_taxonomy_reword_alone_invalidates(self, tmp_path: Path) -> None:
        """End-to-end of the real trigger: same clause text, same taxonomy ids
        (so the same content key), but a rewritten description."""
        store, pending = self._seed(tmp_path, rubric_version("classify", taxonomy=_TAX_A))
        reworded = _tax(
            ("confidentiality", "Confidentiality", "NARROWED: trade secrets only."),
            ("governing_law", "Governing Law", "Choice of law and venue."),
        )
        judge = StoreBackedClassificationJudge(store=store, pending=pending)
        [result] = judge.classify_batch(
            [_node("Confidentiality", "Body of Confidentiality.")], reworded
        )
        assert result.basis == "needs_review"

    def test_legacy_verdict_still_replays_but_is_counted(self, tmp_path: Path) -> None:
        """AC-4: banked human judgment is not discarded on upgrade."""
        store, pending = self._seed(tmp_path, None)
        judge = StoreBackedClassificationJudge(store=store, pending=pending)
        [result] = judge.classify_batch(
            [_node("Confidentiality", "Body of Confidentiality.")], _TAX_A
        )
        assert result.taxonomy_id == "confidentiality"
        assert judge.rubric.total(STATE_LEGACY) == 1
        assert _queued(tmp_path) == []

    def test_strict_rubric_requeues_legacy(self, tmp_path: Path) -> None:
        store, pending = self._seed(tmp_path, None)
        judge = StoreBackedClassificationJudge(
            store=store, pending=pending, rubric=RubricPolicy(strict_legacy=True)
        )
        [result] = judge.classify_batch(
            [_node("Confidentiality", "Body of Confidentiality.")], _TAX_A
        )
        assert result.basis == "needs_review"
        assert [q["kind"] for q in _queued(tmp_path)] == ["classify"]

    def test_accept_stale_replays(self, tmp_path: Path) -> None:
        store, pending = self._seed(tmp_path, "v0+deadbeefcafe")
        judge = StoreBackedClassificationJudge(
            store=store, pending=pending, rubric=RubricPolicy(accept_stale=True)
        )
        [result] = judge.classify_batch(
            [_node("Confidentiality", "Body of Confidentiality.")], _TAX_A
        )
        assert result.taxonomy_id == "confidentiality"
        assert judge.rubric.total(STATE_STALE) == 1
        assert _queued(tmp_path) == []


class TestDeviationJudgeRubric:
    _ITEM = {"hunk": "[BEFORE] 30 days\n[AFTER] 90 days", "taxonomy_id": "term"}
    _STANDARD = "Notice period of 30 days."

    def _seed(self, tmp_path: Path, version: str | None) -> tuple[VerdictStore, PendingQueue]:
        store, pending = _store_and_queue(tmp_path)
        payload = {"stage": "deviation", "hunk": self._ITEM["hunk"], "our_standard": self._STANDARD}
        stamp = RubricStamp(kind="deviation", version=version) if version else None
        store.put(
            payload,
            {
                "deviation": "substantive",
                "risk_delta": {"direction": "worse", "magnitude": "material"},
                "basis": "judge",
                "rationale": "Tripled the notice period.",
            },
            rubric=stamp,
        )
        return store, pending

    def test_current_stamp_replays(self, tmp_path: Path) -> None:
        store, pending = self._seed(tmp_path, rubric_version("deviation"))
        judge = StoreBackedDeviationJudge(store=store, pending=pending)
        [result] = judge.assess_batch([dict(self._ITEM)], self._STANDARD)
        assert result.deviation == "substantive"
        assert _queued(tmp_path) == []

    def test_stale_stamp_requeues_with_traceability_context(self, tmp_path: Path) -> None:
        store, pending = self._seed(tmp_path, "v1+staleaaaaaa")
        judge = StoreBackedDeviationJudge(store=store, pending=pending)
        [result] = judge.assess_batch([dict(self._ITEM)], self._STANDARD)
        assert result.deviation == "needs_review"
        [queued] = _queued(tmp_path)
        assert queued["kind"] == "deviation"
        assert queued["payload"]["taxonomy_id"] == "term"
        assert queued["rubric_version"] == rubric_version("deviation")

    def test_legacy_replays(self, tmp_path: Path) -> None:
        store, pending = self._seed(tmp_path, None)
        judge = StoreBackedDeviationJudge(store=store, pending=pending)
        [result] = judge.assess_batch([dict(self._ITEM)], self._STANDARD)
        assert result.deviation == "substantive"
        assert judge.rubric.total(STATE_LEGACY) == 1


class TestProvenanceJudgeRubric:
    _ARGS = ("Between AlphaCo and BetaCo.", "MUTUAL NDA", "NDA")

    def _seed(self, tmp_path: Path, version: str | None) -> tuple[VerdictStore, PendingQueue]:
        store, pending = _store_and_queue(tmp_path)
        payload = {
            "stage": "provenance",
            "preamble": self._ARGS[0],
            "letterhead": self._ARGS[1],
            "agreement_type": self._ARGS[2],
        }
        stamp = RubricStamp(kind="provenance", version=version) if version else None
        store.put(
            payload, {"provenance": "our_paper", "confidence": 0.9, "basis": "llm"}, rubric=stamp
        )
        return store, pending

    def test_current_stamp_replays(self, tmp_path: Path) -> None:
        store, pending = self._seed(tmp_path, rubric_version("provenance"))
        result = StoreBackedProvenanceJudge(store=store, pending=pending).judge(*self._ARGS)
        assert result.provenance == "our_paper"
        assert _queued(tmp_path) == []

    def test_stale_stamp_requeues(self, tmp_path: Path) -> None:
        store, pending = self._seed(tmp_path, "v0+000000000000")
        result = StoreBackedProvenanceJudge(store=store, pending=pending).judge(*self._ARGS)
        assert result.basis == "needs_review"
        assert [q["kind"] for q in _queued(tmp_path)] == ["provenance"]

    def test_manual_bump_invalidates(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        store, pending = self._seed(tmp_path, rubric_version("provenance"))
        monkeypatch.setitem(RUBRIC_PROMPT_VERSIONS, "provenance", "v2")
        result = StoreBackedProvenanceJudge(store=store, pending=pending).judge(*self._ARGS)
        assert result.basis == "needs_review"


class TestScopeJudgeRubric:
    def _seed(
        self, tmp_path: Path, version: str | None
    ) -> tuple[VerdictStore, PendingQueue, ClauseTree]:
        store, pending = _store_and_queue(tmp_path)
        tree = _tree("deal-synthetic", ["Confidentiality", "Governing Law"])
        payload = {
            "stage": "scope",
            "agreement_type_id": _AGREEMENT_A.id,
            "document_id": tree.document_id,
            "clause_heads": [n.heading or "" for n in tree.all_nodes()],
        }
        stamp = RubricStamp(kind="scope", version=version) if version else None
        store.put(
            payload,
            {"in_scope": True, "scope_rationale": "Reads as an NDA.", "scope_confidence": 0.95},
            rubric=stamp,
        )
        return store, pending, tree

    def test_current_stamp_replays(self, tmp_path: Path) -> None:
        store, pending, tree = self._seed(
            tmp_path, rubric_version("scope", agreement_type=_AGREEMENT_A)
        )
        decision = StoreBackedScopeJudge(store=store, pending=pending).judge(tree, _AGREEMENT_A)
        assert decision.in_scope is True
        assert _queued(tmp_path) == []

    def test_stale_stamp_raises_needs_review_and_requeues(self, tmp_path: Path) -> None:
        store, pending, tree = self._seed(tmp_path, "v0+aaaaaaaaaaaa")
        judge = StoreBackedScopeJudge(store=store, pending=pending)
        with pytest.raises(ScopeNeedsReviewError, match="older rubric"):
            judge.judge(tree, _AGREEMENT_A)
        assert [q["kind"] for q in _queued(tmp_path)] == ["scope"]

    def test_widened_agreement_type_invalidates(self, tmp_path: Path) -> None:
        """The scope question is "is this one of THESE?" — widening the alias
        set changes the question without touching the document."""
        store, pending, tree = self._seed(
            tmp_path, rubric_version("scope", agreement_type=_AGREEMENT_A)
        )
        widened = AgreementType(
            id="nda", name="NDA", description="Mutual NDA.", aliases=["mnda", "cda"]
        )
        with pytest.raises(ScopeNeedsReviewError):
            StoreBackedScopeJudge(store=store, pending=pending).judge(tree, widened)


# ---------------------------------------------------------------------------
# Shared policy across judges (what the CLI reports from)
# ---------------------------------------------------------------------------


def test_one_policy_tallies_every_kind(tmp_path: Path) -> None:
    policy = RubricPolicy()
    store, pending = _store_and_queue(tmp_path)

    classify_payload = {
        "stage": "classify",
        "text": "Body of Confidentiality.",
        "heading": "Confidentiality",
        "taxonomy_ids": sorted(e.id for e in _TAX_A.entries),
    }
    store.put(classify_payload, {"taxonomy_id": "confidentiality"}, rubric=None)
    store.put(
        {"stage": "deviation", "hunk": "h", "our_standard": "s"},
        {
            "deviation": "none",
            "risk_delta": {"direction": "neutral", "magnitude": "none"},
            "basis": "judge",
        },
        rubric=RubricStamp("deviation", "v0+obsolete0000"),
    )

    StoreBackedClassificationJudge(store=store, pending=pending, rubric=policy).classify_batch(
        [_node("Confidentiality", "Body of Confidentiality.")], _TAX_A
    )
    StoreBackedDeviationJudge(store=store, pending=pending, rubric=policy).assess_batch(
        [{"hunk": "h"}], "s"
    )

    assert policy.breakdown(STATE_LEGACY) == {"classify": 1}
    assert policy.breakdown(STATE_STALE) == {"deviation": 1}


def test_pending_queue_records_rubric_version_only_when_given(tmp_path: Path) -> None:
    queue = PendingQueue(tmp_path / "pending.jsonl")
    queue.add("k1", "classify", {"stage": "classify"}, "v1+abc")
    queue.add("k2", "classify", {"stage": "classify"})
    records = _queued(tmp_path)
    assert records[0]["rubric_version"] == "v1+abc"
    assert "rubric_version" not in records[1]


# ---------------------------------------------------------------------------
# AC-7/AC-8: CLI — synthetic examples/judge-fixture corpus only
# ---------------------------------------------------------------------------


def _invoke(*args: str) -> Any:
    return CliRunner().invoke(cli, list(args))


@pytest.fixture
def judged_out(tmp_path: Path) -> Path:
    """Run judge → judge-apply so the store holds real, freshly stamped verdicts."""
    out = tmp_path / "out"
    result = _invoke("judge", str(_CORPUS_DIR), "--config", str(_CONFIG_PATH), "--out", str(out))
    assert result.exit_code == 0, result.output

    pending = [
        json.loads(line)
        for line in (out / "judge" / "pending.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert pending, "fixture corpus should queue at least one item"

    verdicts = tmp_path / "verdicts-in.jsonl"
    canned = {
        "classify": {"taxonomy_id": None, "confidence": 0.4, "basis": "unclassified"},
        "deviation": {
            "deviation": "none",
            "risk_delta": {"direction": "neutral", "magnitude": "none"},
            "basis": "judge",
            "rationale": "No material change.",
        },
        "provenance": {"provenance": "counterparty_paper", "confidence": 0.8, "basis": "llm"},
        "scope": {
            "in_scope": True,
            "scope_rationale": "Affiliation agreement.",
            "scope_confidence": 0.9,
        },
    }
    verdicts.write_text(
        "".join(
            json.dumps({"key": item["key"], "verdict": canned[item["kind"]]}) + "\n"
            for item in pending
        ),
        encoding="utf-8",
    )
    result = _invoke("judge-apply", str(out), "--verdicts", str(verdicts))
    assert result.exit_code == 0, result.output
    return out


def test_judge_apply_stamps_from_the_pending_queue(judged_out: Path) -> None:
    """AC-1 end-to-end: the stamp comes from the queue, so judge-apply needs no config."""
    lines = [
        json.loads(x)
        for x in (judged_out / "judge" / "verdicts.jsonl").read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]
    assert lines
    assert all("rubric" in rec for rec in lines)
    assert all(rec["rubric"]["kind"] in JUDGE_KINDS for rec in lines)


def test_judge_reports_current_rubric_versions(judged_out: Path) -> None:
    result = _invoke(
        "judge",
        str(_CORPUS_DIR),
        "--config",
        str(_CONFIG_PATH),
        "--out",
        str(judged_out),
        "--plan-only",
    )
    assert result.exit_code == 0
    assert "rubric  :" in result.output
    assert "deviation=" in result.output


def test_plan_only_reports_stale_verdicts(
    judged_out: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC-7: the required visibility — "N stored verdicts were made under an older rubric"."""
    for kind in JUDGE_KINDS:
        monkeypatch.setitem(RUBRIC_PROMPT_VERSIONS, kind, "v99")

    result = _invoke(
        "judge",
        str(_CORPUS_DIR),
        "--config",
        str(_CONFIG_PATH),
        "--out",
        str(judged_out),
        "--plan-only",
    )
    assert result.exit_code == 0, result.output
    assert "were made under an older rubric" in result.stderr
    assert "re-queued" in result.stderr


def test_plan_only_is_quiet_when_every_stamp_is_current(judged_out: Path) -> None:
    result = _invoke(
        "judge",
        str(_CORPUS_DIR),
        "--config",
        str(_CONFIG_PATH),
        "--out",
        str(judged_out),
        "--plan-only",
    )
    assert result.exit_code == 0
    assert "older rubric" not in result.stderr
    assert "carry no rubric version" not in result.stderr


def test_judge_reports_legacy_store(judged_out: Path) -> None:
    """AC-4: the pre-versioning store — the actual 1,444-verdict situation."""
    store_path = judged_out / "judge" / "verdicts.jsonl"
    stripped = []
    for line in store_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        rec.pop("rubric", None)
        stripped.append(json.dumps(rec))
    store_path.write_text("\n".join(stripped) + "\n", encoding="utf-8")

    result = _invoke(
        "judge",
        str(_CORPUS_DIR),
        "--config",
        str(_CONFIG_PATH),
        "--out",
        str(judged_out),
        "--plan-only",
    )
    assert result.exit_code == 0, result.output
    assert "carry no rubric version" in result.stderr
    assert "judge-migrate" in result.stderr


def test_judge_migrate_dry_run_reports_without_writing(judged_out: Path) -> None:
    store_path = judged_out / "judge" / "verdicts.jsonl"
    before = store_path.read_text(encoding="utf-8")
    store_path.write_text(
        "\n".join(
            json.dumps({k: v for k, v in json.loads(x).items() if k != "rubric"})
            for x in before.splitlines()
            if x.strip()
        )
        + "\n",
        encoding="utf-8",
    )
    after_strip = store_path.read_text(encoding="utf-8")

    result = _invoke("judge-migrate", str(judged_out), "--config", str(_CONFIG_PATH), "--dry-run")
    assert result.exit_code == 0, result.output
    assert "DRY RUN" in result.output
    assert "legacy" in result.output
    assert store_path.read_text(encoding="utf-8") == after_strip, "dry run must not write"


def test_judge_migrate_adopts_legacy_and_silences_the_warning(judged_out: Path) -> None:
    """AC-8: migration preserves the verdict and converts unknown → current."""
    store_path = judged_out / "judge" / "verdicts.jsonl"
    original = [
        json.loads(x) for x in store_path.read_text(encoding="utf-8").splitlines() if x.strip()
    ]
    store_path.write_text(
        "".join(
            json.dumps({k: v for k, v in rec.items() if k != "rubric"}) + "\n" for rec in original
        ),
        encoding="utf-8",
    )

    result = _invoke("judge-migrate", str(judged_out), "--config", str(_CONFIG_PATH))
    assert result.exit_code == 0, result.output
    assert "re-stamped" in result.output

    migrated = VerdictStore(store_path)
    assert len(migrated) == len(original)
    for rec in original:
        stored = migrated.get_record_by_key(rec["key"])
        assert stored is not None
        assert stored.verdict == rec["verdict"], "migration must not alter the judgment"
        assert stored.rubric_version is not None

    result = _invoke(
        "judge",
        str(_CORPUS_DIR),
        "--config",
        str(_CONFIG_PATH),
        "--out",
        str(judged_out),
        "--plan-only",
    )
    assert "carry no rubric version" not in result.stderr


def test_judge_migrate_leaves_stale_alone_without_accept_stale(
    judged_out: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for kind in JUDGE_KINDS:
        monkeypatch.setitem(RUBRIC_PROMPT_VERSIONS, kind, "v99")
    store_path = judged_out / "judge" / "verdicts.jsonl"
    before = store_path.read_text(encoding="utf-8")

    result = _invoke("judge-migrate", str(judged_out), "--config", str(_CONFIG_PATH))
    assert result.exit_code == 0, result.output
    assert "nothing to re-stamp" in result.output
    assert store_path.read_text(encoding="utf-8") == before


def test_judge_migrate_accept_stale_restamps(
    judged_out: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for kind in JUDGE_KINDS:
        monkeypatch.setitem(RUBRIC_PROMPT_VERSIONS, kind, "v99")

    result = _invoke(
        "judge-migrate", str(judged_out), "--config", str(_CONFIG_PATH), "--accept-stale"
    )
    assert result.exit_code == 0, result.output
    assert "re-stamped" in result.output

    result = _invoke(
        "judge",
        str(_CORPUS_DIR),
        "--config",
        str(_CONFIG_PATH),
        "--out",
        str(judged_out),
        "--plan-only",
    )
    assert "older rubric" not in result.stderr


def test_judge_migrate_kind_filter(judged_out: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for kind in JUDGE_KINDS:
        monkeypatch.setitem(RUBRIC_PROMPT_VERSIONS, kind, "v99")

    result = _invoke(
        "judge-migrate",
        str(judged_out),
        "--config",
        str(_CONFIG_PATH),
        "--accept-stale",
        "--kind",
        "provenance",
    )
    assert result.exit_code == 0, result.output

    # Only provenance may have been advanced; every other kind keeps its old stamp.
    store = VerdictStore(judged_out / "judge" / "verdicts.jsonl")
    advanced = {
        rec.rubric_kind
        for _k, rec in store.records()
        if rec.rubric_version and rec.rubric_version.startswith("v99+")
    }
    assert advanced <= {"provenance"}


def test_judge_migrate_without_store_exits_nonzero(tmp_path: Path) -> None:
    (tmp_path / "out").mkdir()
    result = _invoke("judge-migrate", str(tmp_path / "out"), "--config", str(_CONFIG_PATH))
    assert result.exit_code != 0
    assert "no verdict store" in result.stderr


def test_stale_requeue_is_not_reported_as_a_malformed_verdict_loop(
    judged_out: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The #182 warning ("failed replay validation") must not fire for an
    expected rubric-driven re-queue."""
    for kind in JUDGE_KINDS:
        monkeypatch.setitem(RUBRIC_PROMPT_VERSIONS, kind, "v99")
    result = _invoke(
        "judge", str(_CORPUS_DIR), "--config", str(_CONFIG_PATH), "--out", str(judged_out)
    )
    assert result.exit_code == 0, result.output
    assert "failed replay validation" not in result.stderr
    assert "were made under an older rubric" in result.stderr
