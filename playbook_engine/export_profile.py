"""Export profile — semantic-residue redaction judgment pass (issue #146).

Re-envisions issue #123's "normative export profile" (strip/alias
document_ids, text_summaries, citation details while preserving
stance/structure) as the **born-safe** variant: known entity names are
already pseudonymized at ingest (:mod:`playbook_engine.entity_registry`,
issue #153), so every stored artifact holds only stable aliases before this
module ever runs. What is left for THIS module to catch is SEMANTIC
residue — free text that still identifies a counterparty despite aliasing
(e.g. "the large southeastern teaching-hospital university" never uses the
institution's name, yet still identifies it).

Per the Q6 (revised 2026-07-09) direction, this is a **judgment**, not a
lexical, pass:

  - :class:`RedactionJudge` reviews every free-text sample and may flag +
    rewrite it. Issue #188 extended sampling from just ``text_summary``/
    ``full_text`` on each observed position to EVERY free-text surface in
    the document — ``posture.system_prompt``, interview answers, Floor
    invariant statements/rationale, curation pin comments,
    ``our_standard.text``, clause-concept description/notes, corpus
    document titles, and the baseline template's title/source — since a
    counterparty can be identified from prose anywhere in the doc, not only
    an observation's text. See :func:`_extract_text_samples` for the full
    list.
  - An INDEPENDENT :class:`VerifyJudge` reviews the (possibly rewritten)
    output afterwards. It always runs — regardless of what the redaction
    pass found — because a rewrite can itself leave (or introduce) residue,
    and a single judge marking its own homework is not independent
    verification.
  - This is best-effort: there is NO human release gate in code (see
    issue #152 for live-eval of residue quality). A verify-pass finding of
    leaked residue does **not** block export — but it is never silently
    dropped either: it is always surfaced on :attr:`ExportProfileReport.leaked`
    and logged, so a caller can act on it (alert, quarantine, re-run) even
    though :func:`export_profile` itself does not gate.

Architecture mirrors the codebase's other LLM-integration seams
(:mod:`playbook_engine.floor_judge`, :mod:`playbook_engine.scope_gate`):
frozen result dataclasses, ``Protocol`` judges (real judges call an LLM;
tests inject a fake), and a coverage gate around the JUDGE CONTRACT (a judge
must attempt every sample it was handed, or the run fails loud) — this is
distinct from a "residue found" verdict, which never fails loud, per above.

Scope note: this module is the *mechanism* only, verified offline with fake
judges (per issue #146's required verification). :mod:`playbook_engine.publisher`
(issue #188) is the first real call site (the ``playbook publish`` CLI
command) — but it too defaults to stub judges; authoring the real LLM
prompts is still out of scope here, see the module docstrings of
:mod:`playbook_engine.floor_judge` and :mod:`playbook_engine.scope_gate` for
the same pattern.
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from playbook_engine.opf_accessors import playbook_clause_library, playbook_clauses

_log = logging.getLogger(__name__)

_BASIS_VALUES = frozenset({"judge", "stub", "judge_error"})

# Free-text fields on each ``observed_positions[]`` entry that may carry
# semantic residue. ``full_text`` (issue #105) is the untruncated clause
# text; ``text_summary`` is the ≤200-char display truncation. Both are
# judged independently since a rewrite of one must not silently leave the
# other unexamined.
_FREE_TEXT_FIELDS: tuple[str, ...] = ("text_summary", "full_text")

# ClauseConcept (``evidence.clause_library[]``) free-text fields (issue #188
# gap analysis item 1) — human-authored prose describing an accepted clause
# form, exactly as residue-prone as an observation's text_summary/full_text.
_CLAUSE_CONCEPT_TEXT_FIELDS: tuple[str, ...] = ("description", "notes")

# One ``floor.invariants[]`` entry's free-text fields (issue #188) — a red
# line's ``statement``/``rationale`` are human-authored natural language and
# may quote or paraphrase counterparty-identifying context.
_INVARIANT_TEXT_FIELDS: tuple[str, ...] = ("statement", "rationale")

# ``baseline.template_ref`` free-text fields (issue #188) — ``title`` is
# prose; ``source`` is a path/URI (stripped outright by ``publisher.publish``
# step 2, but still sampled here since a caller of ``export_profile`` alone,
# without going through ``publish``, gets no other protection for it).
_TEMPLATE_REF_TEXT_FIELDS: tuple[str, ...] = ("title", "source")


# ---------------------------------------------------------------------------
# TextSample — one free-text field pulled out of the OPF doc for judgment
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TextSample:
    """One free-text field extracted from the OPF doc for judgment.

    Attributes:
        path: Stable locator back into the doc, e.g.
              ``"clauses[clause.indemnification].observed_positions[0].text_summary"``.
              Opaque to callers — only used to match a judge's findings back
              to the sample it answers for and to apply a rewrite in place.
        text: The text content to judge.
    """

    path: str
    text: str


# A clause-scoped location: (clause_index, container key, item index, field
# name) into ``playbook_clauses(doc)`` — the ORIGINAL, narrower location
# shape (issue #146/#177). A setter is a callable taking the doc being
# rewritten and the replacement text and mutating the target field in place
# — used for every free-text surface OUTSIDE a clause's observed_positions/
# negotiation_trail (issue #188's full-surface extension: posture, floor,
# curation, corpus, baseline, clause_library, our_standard all live at
# different nesting depths that don't fit the 4-tuple shape). Both forms
# close over integer indices/field names only (never over a specific dict
# object), so the SAME location map is safe to apply to any deep copy of the
# doc it was extracted from — see :func:`_apply_rewrites`.
_ClauseLocation = tuple[int, str, int, str]
_Setter = Callable[[dict[str, Any], str], None]


def _extract_text_samples(
    doc: dict[str, Any],
) -> tuple[list[TextSample], dict[str, _ClauseLocation | _Setter]]:
    """Return every free-text sample in *doc*, plus a path -> location map.

    Full free-text surface (issue #188 gap analysis item 1 — everything
    residue sampling missed before this issue is now covered):

      - ``observed_positions[].text_summary`` / ``.full_text`` (issue #146)
      - ``negotiation_trail[].change_summary`` (issue #177 — quotes raw
        clause text verbatim, exactly as residue-prone as an observation)
      - ``clauses[].our_standard.text``
      - ``clause_library[].description`` / ``.notes``
      - ``posture.system_prompt`` and ``posture.generation.interview[].answer``
        (operators type real names/context into interview answers)
      - ``floor.invariants[].statement`` / ``.rationale``
      - ``curation.pins[].comment``
      - ``corpus.documents[].title``
      - ``baseline.template_ref.title`` / ``.source``

    Every surface no-ops cleanly when absent (optional sections, or a v0.1
    fixture / pre-#177 store with none of the newer fields) — there is no
    ordering dependency on which OPF sections a given doc happens to carry.
    """
    samples: list[TextSample] = []
    locations: dict[str, _ClauseLocation | _Setter] = {}

    for ci, clause in enumerate(playbook_clauses(doc)):
        clause_id = clause.get("id", str(ci))
        for oi, obs in enumerate(clause.get("observed_positions", [])):
            for field_name in _FREE_TEXT_FIELDS:
                text = obs.get(field_name, "")
                if not text:
                    continue
                path = f"clauses[{clause_id}].observed_positions[{oi}].{field_name}"
                samples.append(TextSample(path=path, text=text))
                locations[path] = (ci, "observed_positions", oi, field_name)
        for ti, entry in enumerate(clause.get("negotiation_trail", [])):
            text = entry.get("change_summary", "")
            if not text:
                continue
            path = f"clauses[{clause_id}].negotiation_trail[{ti}].change_summary"
            samples.append(TextSample(path=path, text=text))
            locations[path] = (ci, "negotiation_trail", ti, "change_summary")

        our_standard = clause.get("our_standard")
        if isinstance(our_standard, dict):
            text = our_standard.get("text", "")
            if text:
                path = f"clauses[{clause_id}].our_standard.text"
                samples.append(TextSample(path=path, text=text))
                locations[path] = _clause_our_standard_setter(ci)

    for li, concept in enumerate(playbook_clause_library(doc)):
        concept_id = concept.get("concept_id", str(li))
        for field_name in _CLAUSE_CONCEPT_TEXT_FIELDS:
            text = concept.get(field_name, "")
            if not text:
                continue
            path = f"clause_library[{concept_id}].{field_name}"
            samples.append(TextSample(path=path, text=text))
            locations[path] = _clause_library_setter(li, field_name)

    posture = doc.get("posture")
    if isinstance(posture, dict):
        text = posture.get("system_prompt", "")
        if text:
            path = "posture.system_prompt"
            samples.append(TextSample(path=path, text=text))
            locations[path] = _posture_field_setter("system_prompt")
        generation = posture.get("generation")
        if isinstance(generation, dict):
            for ii, entry in enumerate(generation.get("interview", [])):
                text = entry.get("answer", "")
                if not text:
                    continue
                path = f"posture.generation.interview[{ii}].answer"
                samples.append(TextSample(path=path, text=text))
                locations[path] = _interview_answer_setter(ii)

    floor = doc.get("floor")
    if isinstance(floor, dict):
        for fi, invariant in enumerate(floor.get("invariants", [])):
            invariant_id = invariant.get("id", str(fi))
            for field_name in _INVARIANT_TEXT_FIELDS:
                text = invariant.get(field_name, "")
                if not text:
                    continue
                path = f"floor.invariants[{invariant_id}].{field_name}"
                samples.append(TextSample(path=path, text=text))
                locations[path] = _invariant_setter(fi, field_name)

    curation = doc.get("curation")
    if isinstance(curation, dict):
        for pi, pin in enumerate(curation.get("pins", [])):
            text = pin.get("comment", "")
            if not text:
                continue
            path = f"curation.pins[{pi}].comment"
            samples.append(TextSample(path=path, text=text))
            locations[path] = _pin_comment_setter(pi)

    corpus = doc.get("corpus")
    if isinstance(corpus, dict):
        for di, document in enumerate(corpus.get("documents", [])):
            text = document.get("title", "")
            if not text:
                continue
            document_id = document.get("document_id", str(di))
            path = f"corpus.documents[{document_id}].title"
            samples.append(TextSample(path=path, text=text))
            locations[path] = _corpus_document_title_setter(di)

    baseline = doc.get("baseline")
    if isinstance(baseline, dict):
        template_ref = baseline.get("template_ref")
        if isinstance(template_ref, dict):
            for field_name in _TEMPLATE_REF_TEXT_FIELDS:
                text = template_ref.get(field_name, "")
                if not text:
                    continue
                path = f"baseline.template_ref.{field_name}"
                samples.append(TextSample(path=path, text=text))
                locations[path] = _template_ref_setter(field_name)

    return samples, locations


# ---------------------------------------------------------------------------
# Setter factories for every free-text surface outside a clause's
# observed_positions/negotiation_trail (issue #188). Each closes over
# integer indices/field names only, never over a specific dict object, so it
# is safe to apply against ANY structurally-equivalent deep copy of the doc
# it was built from (see the comment above ``_ClauseLocation``/``_Setter``).
# ---------------------------------------------------------------------------


def _clause_our_standard_setter(ci: int) -> _Setter:
    def setter(target: dict[str, Any], text: str) -> None:
        playbook_clauses(target)[ci]["our_standard"]["text"] = text

    return setter


def _clause_library_setter(li: int, field_name: str) -> _Setter:
    def setter(target: dict[str, Any], text: str) -> None:
        playbook_clause_library(target)[li][field_name] = text

    return setter


def _posture_field_setter(field_name: str) -> _Setter:
    def setter(target: dict[str, Any], text: str) -> None:
        target["posture"][field_name] = text

    return setter


def _interview_answer_setter(ii: int) -> _Setter:
    def setter(target: dict[str, Any], text: str) -> None:
        target["posture"]["generation"]["interview"][ii]["answer"] = text

    return setter


def _invariant_setter(fi: int, field_name: str) -> _Setter:
    def setter(target: dict[str, Any], text: str) -> None:
        target["floor"]["invariants"][fi][field_name] = text

    return setter


def _pin_comment_setter(pi: int) -> _Setter:
    def setter(target: dict[str, Any], text: str) -> None:
        target["curation"]["pins"][pi]["comment"] = text

    return setter


def _corpus_document_title_setter(di: int) -> _Setter:
    def setter(target: dict[str, Any], text: str) -> None:
        target["corpus"]["documents"][di]["title"] = text

    return setter


def _template_ref_setter(field_name: str) -> _Setter:
    def setter(target: dict[str, Any], text: str) -> None:
        target["baseline"]["template_ref"][field_name] = text

    return setter


def _apply_rewrites(
    doc: dict[str, Any],
    findings: Sequence[RedactionFinding],
    locations: dict[str, _ClauseLocation | _Setter],
) -> dict[str, Any]:
    """Return a deep copy of *doc* with every flagged sample's text replaced.

    Only the targeted free-text field is mutated — clause id, taxonomy_id,
    rollup (position/confidence), deviation, risk_delta, provenance, outcome,
    and every other structural field are copied through unchanged. This is
    what "export preserves stance + clause structure" means in practice.
    """
    exported = copy.deepcopy(doc)
    exported_clauses = playbook_clauses(exported)
    for finding in findings:
        if not finding.has_residue:
            continue
        loc = locations.get(finding.path)
        if loc is None:
            # Judge returned a finding under a path we never handed it.
            # Nothing to rewrite; the coverage check on samples (not
            # findings) is what guards against silent gaps.
            continue
        # Guaranteed non-None by RedactionFinding.__post_init__ whenever
        # has_residue is True (checked above).
        rewritten_text = finding.rewritten_text
        assert rewritten_text is not None
        if isinstance(loc, tuple):
            ci, container, idx, field_name = loc
            exported_clauses[ci][container][idx][field_name] = rewritten_text
        else:
            loc(exported, rewritten_text)
    return exported


# ---------------------------------------------------------------------------
# RedactionJudge — flags + rewrites semantic residue
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RedactionFinding:
    """One :class:`RedactionJudge` verdict for one :class:`TextSample`.

    Attributes:
        path:           The :attr:`TextSample.path` this finding answers for.
        has_residue:     ``True`` if the judge found the text still identifies
                         the counterparty despite aliasing.
        rewritten_text: The residue-free replacement text. Required when
                        ``has_residue`` is ``True``; ``None`` otherwise (the
                        original text is kept as-is).
        rationale:      Human-readable explanation — always required, mirroring
                        every other judge result in this codebase.
        basis:          ``"judge"`` (real LLM-backed judge), ``"stub"`` (no LLM
                        configured — a rubber-stamped "no residue" verdict, not
                        evidence of genuine judgment), or ``"judge_error"``
                        (the judge raised and a wrapper caught it while still
                        logging a verdict).
    """

    path: str
    has_residue: bool
    rationale: str
    rewritten_text: str | None = None
    basis: str = "judge"

    def __post_init__(self) -> None:
        if self.basis not in _BASIS_VALUES:
            raise ValueError(
                f"RedactionFinding.basis must be one of {sorted(_BASIS_VALUES)!r}; "
                f"got {self.basis!r}"
            )
        if not self.rationale.strip():
            raise ValueError("RedactionFinding.rationale must not be empty")
        if self.has_residue and not (self.rewritten_text or "").strip():
            raise ValueError(
                "RedactionFinding.rewritten_text is required when has_residue=True — "
                "a flagged sample must never be exported without its replacement text."
            )

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "path": self.path,
            "has_residue": self.has_residue,
            "rationale": self.rationale,
            "basis": self.basis,
        }
        if self.rewritten_text is not None:
            d["rewritten_text"] = self.rewritten_text
        return d


@runtime_checkable
class RedactionJudge(Protocol):
    """Protocol for the semantic-residue redaction pass — the LLM integration point.

    Implementations MAY call an LLM; tests inject a fake that returns
    pre-canned findings.
    """

    def evaluate_batch(self, samples: Sequence[TextSample]) -> list[RedactionFinding]:
        """Evaluate every sample in *samples* for semantic residue.

        Returns:
            One :class:`RedactionFinding` per sample it was able to evaluate,
            keyed by :attr:`RedactionFinding.path`. MAY be a strict subset of
            *samples* — that gap is exactly what :func:`export_profile`'s
            coverage check exists to catch.
        """
        ...  # pragma: no cover


# ---------------------------------------------------------------------------
# VerifyJudge — independent second pass over the (possibly rewritten) output
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VerifyFinding:
    """One independent :class:`VerifyJudge` verdict for one :class:`TextSample`.

    Attributes:
        path:      The :attr:`TextSample.path` this finding answers for
                   (evaluated against the POST-redaction text).
        leaked:    ``True`` if this independent pass still finds the
                   counterparty identifiable. A ``True`` here is never
                   silently dropped — see :attr:`ExportProfileReport.leaked`.
        rationale: Human-readable explanation — always required.
        basis:     ``"judge"``, ``"stub"``, or ``"judge_error"`` — same
                   convention as :attr:`RedactionFinding.basis`.
    """

    path: str
    leaked: bool
    rationale: str
    basis: str = "judge"

    def __post_init__(self) -> None:
        if self.basis not in _BASIS_VALUES:
            raise ValueError(
                f"VerifyFinding.basis must be one of {sorted(_BASIS_VALUES)!r}; got {self.basis!r}"
            )
        if not self.rationale.strip():
            raise ValueError("VerifyFinding.rationale must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "leaked": self.leaked,
            "rationale": self.rationale,
            "basis": self.basis,
        }


@runtime_checkable
class VerifyJudge(Protocol):
    """Protocol for the independent verify pass — the LLM integration point.

    Deliberately a SEPARATE protocol from :class:`RedactionJudge` (even though
    the method shape mirrors it) so a real implementation cannot accidentally
    reuse the same judge instance/prompt for both passes — the whole point is
    that the verify pass is independent of the redaction pass.
    """

    def evaluate_batch(self, samples: Sequence[TextSample]) -> list[VerifyFinding]:
        """Evaluate every sample in *samples* (post-redaction text) for leakage.

        Returns:
            One :class:`VerifyFinding` per sample it was able to evaluate,
            keyed by :attr:`VerifyFinding.path`. MAY be a strict subset of
            *samples* — see :class:`RedactionJudge.evaluate_batch`.
        """
        ...  # pragma: no cover


# ---------------------------------------------------------------------------
# Coverage gate + export_profile entry point
# ---------------------------------------------------------------------------


class ExportProfileError(Exception):
    """Raised when a judge's CONTRACT is broken — never for a "residue found" verdict.

    Mirrors ``FloorCoverageError``/``SegmentationQAError``'s fail-loud
    contract: a judge that raises, or that silently drops a sample it was
    handed (returns fewer findings than samples), leaves a free-text field
    NEVER EVALUATED — exactly the failure mode this module exists to
    prevent. This is orthogonal to whether residue/leakage was *found*: a
    "yes, this leaks" verdict is a successful evaluation and is surfaced via
    :attr:`ExportProfileReport.leaked`, never raised as an error.
    """


@dataclass(frozen=True)
class ExportProfileReport:
    """Result of :func:`export_profile`: the exported doc plus both judge passes.

    Attributes:
        doc:                The exported doc — a deep copy of the input with
                            every flagged free-text field replaced by its
                            judge-provided rewrite. All other fields (clause
                            id, taxonomy_id, rollup, deviation, risk_delta,
                            provenance, outcome, citations, ...) are
                            byte-identical to the input.
        redaction_findings: Every :class:`RedactionFinding`, one per free-text
                            sample in the input doc.
        verify_findings:    Every :class:`VerifyFinding`, one per free-text
                            sample in :attr:`doc` (the POST-redaction text) —
                            always populated when there is any free text to
                            check, independent of what ``redaction_findings``
                            found.
        leaked:             The subset of ``verify_findings`` with
                            ``leaked=True`` — the caller's escalation signal.
                            Best-effort/no-gate (Q6): a non-empty ``leaked``
                            does not block export, but is never silently
                            dropped either.
    """

    doc: dict[str, Any]
    redaction_findings: tuple[RedactionFinding, ...]
    verify_findings: tuple[VerifyFinding, ...]
    leaked: tuple[VerifyFinding, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "redaction_findings": [f.to_dict() for f in self.redaction_findings],
            "verify_findings": [f.to_dict() for f in self.verify_findings],
            "leaked": [f.to_dict() for f in self.leaked],
        }


def _run_pass(
    samples: Sequence[TextSample],
    judge: RedactionJudge | VerifyJudge,
    pass_name: str,
) -> list[Any]:
    """Run one judgment pass over *samples* and enforce full sample coverage."""
    if not samples:
        return []
    try:
        findings = judge.evaluate_batch(samples)
    except Exception as exc:  # noqa: BLE001
        raise ExportProfileError(
            f"{pass_name} judge raised {type(exc).__name__}: {exc}. "
            f"{len(samples)} sample(s) went unevaluated."
        ) from exc

    covered = {f.path for f in findings}
    missing = [s.path for s in samples if s.path not in covered]
    if missing:
        raise ExportProfileError(
            f"{pass_name} pass left {len(missing)} sample(s) unevaluated: {missing!r}. "
            "Every free-text sample must be evaluated (issue #146)."
        )
    return list(findings)


def export_profile(
    doc: dict[str, Any],
    redaction_judge: RedactionJudge,
    verify_judge: VerifyJudge,
) -> ExportProfileReport:
    """Run the semantic-residue redaction pass, then an independent verify pass.

    Args:
        doc:             A compiled OPF doc (already born-safe per #153 — known
                         entity names are aliases, never raw names).
        redaction_judge: Flags + rewrites semantic residue in every free-text
                         sample (``text_summary`` / ``full_text``).
        verify_judge:    Independently re-checks the (possibly rewritten)
                         samples. ALWAYS runs when there is free text to check
                         — regardless of what ``redaction_judge`` found —
                         because a rewrite can itself leave or introduce
                         residue, and one judge cannot verify itself.

    Returns:
        :class:`ExportProfileReport`. Never raises on a "leak found" verdict
        (best-effort, no human gate — Q6); only raises
        :class:`ExportProfileError` if a judge breaks its coverage contract
        (raises, or silently drops a sample).
    """
    samples, locations = _extract_text_samples(doc)

    redaction_findings = _run_pass(samples, redaction_judge, "RedactionJudge")
    exported_doc = _apply_rewrites(doc, redaction_findings, locations)

    # Independent verify pass: re-extract from the EXPORTED doc so it judges
    # the post-redaction text, and runs even when redaction_findings is all
    # has_residue=False — this is not conditioned on the redaction outcome.
    post_samples, _post_locations = _extract_text_samples(exported_doc)
    verify_findings = _run_pass(post_samples, verify_judge, "VerifyJudge")

    leaked = tuple(f for f in verify_findings if f.leaked)
    if leaked:
        # Surfaced, never silently emitted (issue #146 required verification):
        # logged here in addition to being returned on the report, so a
        # caller that only checks logs (not the report) still sees it.
        _log.warning(
            "export_profile: independent verify pass flagged %d sample(s) as still "
            "leaking semantic residue after redaction: %s",
            len(leaked),
            [f.path for f in leaked],
        )

    return ExportProfileReport(
        doc=exported_doc,
        redaction_findings=tuple(redaction_findings),
        verify_findings=tuple(verify_findings),
        leaked=leaked,
    )
