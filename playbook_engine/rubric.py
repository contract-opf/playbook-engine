"""Rubric versioning for the store-backed judges.

Closes the silent-replay hazard in :mod:`playbook_engine.agent_judge`: the
verdict store is keyed purely by *content* (clause text, hunk, preamble …),
so when the **judging criteria** change — the taxonomy a clause is classified
into, the deviation vocabulary, the prose rubric in the
``playbook-from-corpus`` skill — every previously banked verdict keeps
replaying unchanged. Nothing re-queues, because the clause text did not move.
A re-derivation observed 1,444 verdicts seeded from an earlier run with only
246 items re-queued, and there was no way to tell which of the 1,198 replays
were still answers to the question currently being asked.

Model
-----

Every verdict is **stamped**, at the moment it enters the store, with the
rubric version that was current when the question was posed. On replay the
stamp is compared against the version current *now*:

======== ============================================ =========================
state    meaning                                       default behaviour
======== ============================================ =========================
current  stamp == current version                      replay
legacy   no stamp at all (banked pre-versioning)       replay, reported loudly
stale    stamp != current version                      re-queue for re-judgement
======== ============================================ =========================

The version is **not** folded into the content hash. That is deliberate:
folding it in would silently orphan every banked verdict on the first rubric
bump (the key would simply never match again), which is exactly the "throw
away the human judgment" outcome this is meant to avoid. Keeping the key
content-only and the version in the record means a rubric bump is a visible,
countable, reversible event — and cross-document dedup of identical clause
text keeps working.

Version shape: ``"<manual>+<derived>"``, e.g. ``"v2+9f1c0a4b21de"``.

- The **manual** half (:data:`RUBRIC_PROMPT_VERSIONS`) covers the part of the
  rubric that lives in prose and cannot be hashed usefully: the judge prompts
  and rules in ``.claude/skills/playbook-from-corpus/REFERENCE.md``. Hashing
  that file directly was rejected — it is a large operator-facing document
  where a typo fix, a reordered bullet, or an added failure-mode note would
  invalidate thousands of verdicts that no human would consider stale. A
  human decides when a prose change is *semantic* and bumps the constant.

- The **derived** half is a digest of the machine-readable rubric surface for
  that kind — the answer vocabulary and, where the judge seam actually has it
  in hand, the semantic input the operator edits (the taxonomy for
  ``classify``; the agreement-type definition for ``scope``). This half needs
  no discipline: editing ``spec/taxonomy/nda.yaml`` moves it automatically.

This mirrors the split already proven in
:mod:`playbook_engine.llm_segmenter_batch`, whose segmentation cache is keyed
on a hand-maintained ``PROMPT_VERSION`` alongside a derived ``SCHEMA_HASH``.

What is deliberately *excluded* from the derived half
-----------------------------------------------------

- **Engine-internal basis values.** ``clause_classifier._BASIS_VALUES`` and
  friends carry values a judge may never emit (``exact_match``,
  ``llm_segmenter``, ``deterministic``). Adding one is an engine change, not
  a rubric change, and must not churn banked verdicts. Only the values a
  judge is actually allowed to answer with are hashed.

- **Thresholds** (``AUTO_CLASSIFY_THRESHOLD``, ``AMBIGUITY_THRESHOLD``,
  ``REWORDED_EQUIVALENT_THRESHOLD``). These change *which* clauses reach a
  judge and how much a downstream consumer trusts the answer — not what a
  past answer means. A verdict is still a correct answer to the question it
  was asked after a threshold moves.

- **``TaxonomyEntry.structural``.** It gates floor-candidate derivation, not
  classification, and is not serialised into ``playbook.opf.json`` — so
  including it would make the digest computed from an OPF-embedded taxonomy
  (see :func:`taxonomy_digest`) disagree with the one computed from the YAML.

- **Provenance aliases.** ``provenance.our_party_aliases`` is genuinely part
  of the provenance rubric, but the ``ProvenanceJudge`` seam never receives
  the config, so it cannot be hashed without widening that protocol. It is
  covered by the manual half instead; bump ``RUBRIC_PROMPT_VERSIONS
  ["provenance"]`` when the alias set changes materially.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from playbook_engine.deviation_classifier import (
    _DEVIATION_VALUES,
    _DIRECTION_VALUES,
    _MAGNITUDE_VALUES,
)
from playbook_engine.provenance_detector import _PROVENANCE_VALUES

# ---------------------------------------------------------------------------
# Kinds
# ---------------------------------------------------------------------------

#: The four pending-item kinds a store-backed judge can produce.
JUDGE_KINDS: tuple[str, ...] = ("classify", "deviation", "provenance", "scope")

# ---------------------------------------------------------------------------
# Manual half — bump by hand when the PROSE rubric changes semantically
# ---------------------------------------------------------------------------

#: Hand-maintained prompt/criteria version per judge kind. Bump the entry for
#: a kind when the corresponding section of
#: ``.claude/skills/playbook-from-corpus/REFERENCE.md`` changes in a way that
#: could change a reasonable judge's answer — new/removed answer categories,
#: a reversed default, a changed definition of "material" — and NOT for
#: typo fixes, reworded examples, or added guardrails that only restate
#: existing rules.
#:
#: ``deviation`` starts at ``"v2"``: issues #117/#118/#119 changed extraction
#: and tracked-change attribution, so the ``[BEFORE]``/``[AFTER]`` hunks the
#: deviation judge reasons over are no longer assembled the way they were for
#: the July-2026 verdict bank. Verdicts stamped ``v1`` (or stamped by a
#: migration as pre-#117) are answers to a materially different question.
RUBRIC_PROMPT_VERSIONS: dict[str, str] = {
    "classify": "v1",
    "deviation": "v2",
    "provenance": "v1",
    "scope": "v1",
}

#: Truncation length for the derived digest. 12 hex chars = 48 bits; these
#: are equality-compared identifiers, never security tokens.
_DIGEST_LEN = 12


class RubricError(ValueError):
    """Raised for an unknown judge kind or a malformed rubric stamp."""


# ---------------------------------------------------------------------------
# Derived half — digests of the machine-readable rubric surface
# ---------------------------------------------------------------------------


def _digest(surface: Any) -> str:
    """Stable short SHA-256 over a JSON-serialisable rubric *surface*."""
    raw = json.dumps(surface, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()[:_DIGEST_LEN]


def _entry_field(entry: Any, name: str, default: Any = "") -> Any:
    """Read *name* off a ``TaxonomyEntry`` dataclass **or** an OPF entry dict.

    The same taxonomy reaches this module two ways: as a
    :class:`~playbook_engine.taxonomy.Taxonomy` (CLI / mining path) and as the
    ``taxonomy.entries`` list embedded in ``playbook.opf.json`` (viewer
    feedback path). Both must digest identically or a reviewer correction
    would land already-stale.
    """
    if isinstance(entry, Mapping):
        value = entry.get(name, default)
    else:
        value = getattr(entry, name, default)
    return default if value is None else value


def taxonomy_entries_of(taxonomy: Any) -> list[Any]:
    """Return the entry list of *taxonomy* (dataclass, Mapping, or bare list)."""
    if taxonomy is None:
        return []
    if isinstance(taxonomy, Mapping):
        entries = taxonomy.get("entries", [])
    elif hasattr(taxonomy, "entries"):
        entries = taxonomy.entries
    elif isinstance(taxonomy, Iterable):
        entries = list(taxonomy)
    else:  # pragma: no cover - defensive
        entries = []
    return list(entries)


def taxonomy_digest(taxonomy: Any) -> str:
    """Digest the classification rubric carried by *taxonomy*.

    Surface: ``id`` + ``label`` + ``description`` of every classifier-eligible
    entry (``status`` in ``active``/``custom`` — a compiler may not classify
    into an ``inactive`` entry, so its wording is not part of the rubric),
    sorted by id.

    Note the id **set** is already inside the classify payload hash, so an
    added or removed id re-queues by content anyway. What this digest adds is
    the part the key cannot see: a re-worded label or a rewritten description,
    which changes what the judge is being asked without changing the payload
    by a single byte.
    """
    surface = sorted(
        (
            str(_entry_field(e, "id")),
            str(_entry_field(e, "label")),
            str(_entry_field(e, "description")),
        )
        for e in taxonomy_entries_of(taxonomy)
        if str(_entry_field(e, "status", "active")) in ("active", "custom")
    )
    return _digest(surface)


def _agreement_type_surface(agreement_type: Any) -> Any:
    """Scope rubric surface: the agreement-type definition being gated on.

    ``scope`` asks "is this document one of *these*?" — so the id, name,
    description and alias set of the target agreement type ARE the rubric.
    Only the id is in the scope payload hash; editing the description or
    aliases in the config silently changes the question otherwise.
    """
    if agreement_type is None:
        return None
    return {
        "id": str(_entry_field(agreement_type, "id")),
        "name": str(_entry_field(agreement_type, "name")),
        "description": str(_entry_field(agreement_type, "description")),
        "aliases": sorted(str(a) for a in (_entry_field(agreement_type, "aliases", []) or [])),
    }


#: Answer vocabularies — the values a judge of each kind is permitted to
#: emit. Engine-internal sentinels are stripped (see module docstring).
_DEVIATION_ANSWERS = sorted(_DEVIATION_VALUES - {"needs_review"})
_PROVENANCE_ANSWERS = sorted(_PROVENANCE_VALUES)


def _derived_surface(kind: str, *, taxonomy: Any, agreement_type: Any) -> Any:
    if kind == "classify":
        return {"taxonomy": taxonomy_digest(taxonomy)}
    if kind == "deviation":
        return {
            "deviation": _DEVIATION_ANSWERS,
            "direction": sorted(_DIRECTION_VALUES),
            "magnitude": sorted(_MAGNITUDE_VALUES),
        }
    if kind == "provenance":
        return {"provenance": _PROVENANCE_ANSWERS}
    if kind == "scope":
        return {"agreement_type": _agreement_type_surface(agreement_type)}
    raise RubricError(f"unknown judge kind {kind!r}; expected one of {list(JUDGE_KINDS)}")


def rubric_version(
    kind: str,
    *,
    taxonomy: Any = None,
    agreement_type: Any = None,
) -> str:
    """Return the rubric version currently in force for *kind*.

    Args:
        kind:            One of :data:`JUDGE_KINDS`.
        taxonomy:        Taxonomy in force (``classify`` only). A
                         :class:`~playbook_engine.taxonomy.Taxonomy`, an OPF
                         ``taxonomy`` mapping, or a bare entry list. Omitting
                         it yields the empty-taxonomy digest, which is a
                         *different* version from any real taxonomy — so a
                         caller that forgets it produces stale-looking
                         stamps rather than silently-compatible ones.
        agreement_type:  Agreement type in force (``scope`` only); same
                         duck-typing.

    Returns:
        ``"<manual>+<derived>"``, e.g. ``"v1+3ad9f0c1b877"``.

    Raises:
        RubricError: if *kind* is not a known judge kind.
    """
    if kind not in RUBRIC_PROMPT_VERSIONS:
        raise RubricError(f"unknown judge kind {kind!r}; expected one of {list(JUDGE_KINDS)}")
    surface = _derived_surface(kind, taxonomy=taxonomy, agreement_type=agreement_type)
    return f"{RUBRIC_PROMPT_VERSIONS[kind]}+{_digest(surface)}"


def current_versions(*, taxonomy: Any = None, agreement_type: Any = None) -> dict[str, str]:
    """Return ``{kind: version}`` for all four kinds (for CLI reporting)."""
    return {
        kind: rubric_version(kind, taxonomy=taxonomy, agreement_type=agreement_type)
        for kind in JUDGE_KINDS
    }


# ---------------------------------------------------------------------------
# Stamps
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RubricStamp:
    """The rubric a stored verdict was produced under.

    Serialised into each ``verdicts.jsonl`` record as
    ``{"rubric": {"kind": …, "version": …}}``. Absent on every record written
    before rubric versioning existed — that absence is the ``legacy`` state,
    and is treated as information ("we do not know"), never as agreement.
    """

    kind: str
    version: str

    def to_dict(self) -> dict[str, str]:
        return {"kind": self.kind, "version": self.version}

    @classmethod
    def from_dict(cls, raw: Any) -> RubricStamp | None:
        """Parse a stamp from a store record; ``None`` if absent or malformed.

        Malformed stamps degrade to ``None`` (legacy) rather than raising:
        the store's whole contract is that a bad line never crashes startup.
        """
        if not isinstance(raw, Mapping):
            return None
        kind = raw.get("kind")
        version = raw.get("version")
        if not isinstance(kind, str) or not isinstance(version, str) or not version:
            return None
        return cls(kind=kind, version=version)


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------

#: Replay states, worst-first for reporting.
STATE_STALE = "stale"
STATE_LEGACY = "legacy"
STATE_CURRENT = "current"


@dataclass(frozen=True)
class RubricDecision:
    """Outcome of comparing a stored stamp against the current rubric."""

    state: str
    replay: bool
    stored_version: str | None
    current_version: str

    @property
    def is_current(self) -> bool:
        return self.state == STATE_CURRENT


@dataclass
class RubricPolicy:
    """Staleness policy + run-scoped tally, shared by all four judges.

    One instance is handed to every store-backed judge in a run, so the CLI
    can report a single coherent picture afterwards.

    Defaults encode the intended posture:

    - **stale ⇒ re-queue.** A verdict whose rubric provably moved is not an
      answer to the current question. It goes back on the pending queue and
      gets re-judged. ``accept_stale=True`` overrides this for an operator
      who has decided the change is immaterial for this run.
    - **legacy ⇒ replay, loudly.** Pre-versioning verdicts are the entire
      existing bank; auto-invalidating them would discard exactly the human
      judgment this design is protecting. They replay, are counted, and are
      reported on every run until an operator stamps them with
      ``playbook judge-migrate``. ``strict_legacy=True`` re-queues them
      instead, for an operator who wants a clean slate.
    """

    strict_legacy: bool = False
    accept_stale: bool = False
    #: ``(kind, state) -> count`` for verdicts actually replayed/re-queued
    #: during this run. Only store *hits* are tallied — a plain miss is not a
    #: rubric event.
    counts: dict[tuple[str, str], int] = field(default_factory=dict)

    def classify(self, stored_version: str | None, current_version: str | None) -> tuple[str, bool]:
        """Return ``(state, replay)`` for a stamp pair, WITHOUT tallying.

        Pure counterpart of :meth:`evaluate`, for callers that need to ask
        "would this have replayed?" after the fact — e.g. ``playbook judge``
        distinguishing a rubric-driven re-queue from a malformed-verdict
        re-queue. A ``current_version`` of ``None`` means the caller could not
        determine it (an old ``pending.jsonl`` with no ``rubric_version``
        field), which is not evidence of staleness — treat as replay.
        """
        if current_version is None:
            return STATE_CURRENT, True
        if stored_version is None:
            return STATE_LEGACY, not self.strict_legacy
        if stored_version == current_version:
            return STATE_CURRENT, True
        return STATE_STALE, self.accept_stale

    def would_replay(self, stored_version: str | None, current_version: str | None) -> bool:
        """``True`` if this stamp pair replays under the current policy."""
        return self.classify(stored_version, current_version)[1]

    def evaluate(
        self, kind: str, stored_version: str | None, current_version: str
    ) -> RubricDecision:
        """Classify a store hit and record it. Returns the replay decision."""
        state, replay = self.classify(stored_version, current_version)
        self.counts[(kind, state)] = self.counts.get((kind, state), 0) + 1
        return RubricDecision(
            state=state,
            replay=replay,
            stored_version=stored_version,
            current_version=current_version,
        )

    # -- reporting ------------------------------------------------------

    def total(self, state: str) -> int:
        """Total hits in *state* across all kinds."""
        return sum(n for (_kind, s), n in self.counts.items() if s == state)

    def breakdown(self, state: str) -> dict[str, int]:
        """``{kind: count}`` for *state*, kinds with zero hits omitted."""
        return {kind: n for (kind, s), n in sorted(self.counts.items()) if s == state and n}

    def format_breakdown(self, state: str) -> str:
        """``"classify: 12, deviation: 3"`` — empty string when nothing in *state*."""
        return ", ".join(f"{k}: {n}" for k, n in self.breakdown(state).items())
