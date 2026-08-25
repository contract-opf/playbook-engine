"""OPF playbook validator.

Validates a playbook JSON/YAML against:
  1. spec/playbook.schema.json (JSON Schema draft 2020-12)
  2. Normative rules the schema cannot express (OPF §2.2, §3.6, §4)
"""

from __future__ import annotations

import datetime
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jsonschema
import yaml

from playbook_engine.clause_position_compiler import MIN_EVIDENCE_N
from playbook_engine.opf_accessors import playbook_clause_library, playbook_clauses

# ``playbook publish`` (issue #188) coarsens ``observed_at`` from an ISO-8601
# date to ``YYYY-Qn`` (exact dates can identify a counterparty). A published
# document must still pass this validator, so the dynamics check below
# accepts either form — a malformed value in neither shape is still rejected.
_QUARTER_DATE_RE = re.compile(r"^\d{4}-Q[1-4]$")

_SCHEMA_PATH_V1 = Path(__file__).parent.parent / "spec" / "playbook.schema.json"
_SCHEMA_PATH_V2 = Path(__file__).parent.parent / "spec" / "playbook.schema-0.2.json"
_SCHEMA_PATH_V3 = Path(__file__).parent.parent / "spec" / "playbook.schema-0.3.json"

# Versions this validator's normative checks know how to enforce:
#   "0.1" — clauses/clause_library top-level, rollup.position (§2.2, §3.6, §4)
#   "0.2" — clauses/clause_library under `evidence`, summary.historical_stance
#           (OPF-SPEC.md §2.2, §3.5, §4)
#   "0.3" — the 0.2 shape plus an optional top-level `digest` section (the
#           compact model-facing projection); all 0.2 normative checks apply
#           unchanged, plus the digest consistency check.
# Anything else (missing, or an unrecognized version) must fail loud rather
# than silently pass with an empty `doc.get("clauses", [])`.
_SUPPORTED_OPF_VERSIONS = {"0.1", "0.2", "0.3"}

_SCHEMA_PATH_BY_VERSION = {"0.2": _SCHEMA_PATH_V2, "0.3": _SCHEMA_PATH_V3}

# Public alias — `playbook --version` (cli.py) reports these alongside the
# engine version so bug reports carry both, since engine version and OPF
# version drift independently (issue #176).
SUPPORTED_OPF_VERSIONS = _SUPPORTED_OPF_VERSIONS

# Zero-width/bidi-control characters. A downstream review engine's fail-closed
# injection scan rejects a document containing any of these, so their
# presence is a blocking error here — the assembler strips them
# (``playbook_assembler._strip_invisible``) and this check keeps hand-edited
# or third-party documents honest.
_INVISIBLE_CHARS_RE = re.compile("[\u200b-\u200d\ufeff\u202a-\u202e]")


@dataclass
class ValidationError:
    message: str
    path: str = ""
    blocking: bool = True

    def __str__(self) -> str:
        loc = f" [{self.path}]" if self.path else ""
        return f"{'ERROR' if self.blocking else 'WARN '}{loc}: {self.message}"


@dataclass
class ValidationResult:
    errors: list[ValidationError] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(e.blocking for e in self.errors)

    def add(self, message: str, path: str = "", blocking: bool = True) -> None:
        self.errors.append(ValidationError(message, path, blocking))


def _load_schema(opf_version: Any) -> dict[str, Any]:
    """Select the schema to validate against based on `opf_version`.

    "0.2" gets the v0.2 (Evidence/Posture/Floor) schema; "0.3" gets the v0.3
    schema (0.2 + digest); everything else (including missing/unrecognized
    versions) is validated against the v0.1 schema, whose `const: "0.1"` on
    `opf_version` produces the expected schema error for those cases —
    preserving pre-existing v0.1 behavior byte-for-byte.
    """
    schema_path = _SCHEMA_PATH_BY_VERSION.get(opf_version, _SCHEMA_PATH_V1)
    with schema_path.open() as f:
        return json.load(f)  # type: ignore[no-any-return]


def _path_str(path: list[str | int]) -> str:
    return ".".join(str(p) for p in path) if path else "<root>"


def _corpus_docs(doc: dict[str, Any]) -> dict[str, dict[str, Any]]:
    # A hand-edited/foreign corpus document may omit document_id entirely (or
    # carry a non-string one) — skip it rather than KeyError; the schema pass
    # already reports the missing/invalid field, and a document that can't be
    # keyed can't be cited by document_id anyway (see issue #72).
    return {
        d["document_id"]: d
        for d in doc.get("corpus", {}).get("documents", [])
        if isinstance(d.get("document_id"), str)
    }


def _check_opf_version(doc: dict[str, Any], result: ValidationResult) -> bool:
    """Fail loud on an opf_version this validator doesn't understand.

    Returns True if the version is supported and the version-appropriate
    normative checks may safely proceed; False if they must be skipped because
    the document's shape is unknown (e.g. an unrecognized version whose
    clauses live somewhere this validator does not know to look — iterating
    an empty top-level `clauses` would silently no-op every normative check).
    """
    version = doc.get("opf_version")
    if version not in _SUPPORTED_OPF_VERSIONS:
        supported = ", ".join(sorted(_SUPPORTED_OPF_VERSIONS))
        result.add(
            f"unsupported opf_version {version!r} (supported: {supported}) — "
            "unrecognized document shape, normative checks (§2.2/§3.6/§4) skipped",
            path="opf_version",
        )
        return False
    return True


def _check_provenance_rule(doc: dict[str, Any], result: ValidationResult) -> None:
    """OPF §2.2: counterparty-paper observations must not set our_standard."""
    corpus_docs = _corpus_docs(doc)

    for i, clause in enumerate(doc.get("clauses", [])):
        # our_standard must not be sourced from a counterparty_paper document.
        # (Dangling/unknown document_id is caught generically by _check_citations.)
        our_std = clause.get("our_standard")
        if our_std is not None:
            source = our_std.get("source_ref") or {}
            doc_id = source.get("document_id", "")
            if (
                doc_id
                and doc_id != "template"
                and doc_id in corpus_docs
                and corpus_docs[doc_id].get("provenance") == "counterparty_paper"
            ):
                result.add(
                    "our_standard sourced from a counterparty_paper document — violates OPF §2.2",
                    path=f"clauses[{i}].our_standard.source_ref",
                )

        # rollup.position must not be strong when ALL observations are counterparty_paper.
        observations = clause.get("observed_positions", [])
        if observations and all(
            obs.get("provenance") == "counterparty_paper" for obs in observations
        ):
            rollup = clause.get("rollup", {})
            position = rollup.get("position")
            if position in ("standard", "hold_firm", "acceptable_variants_exist"):
                result.add(
                    f"rollup.position={position!r} but all observations are counterparty_paper — violates OPF §2.2",
                    path=f"clauses[{i}].rollup.position",
                )


def _check_out_of_scope_rationale(doc: dict[str, Any], result: ValidationResult) -> None:
    """OPF §3.6: out-of-scope docs must carry scope_rationale."""
    for i, corpus_doc in enumerate(doc.get("corpus", {}).get("documents", [])):
        if not corpus_doc.get("in_scope", True) and not corpus_doc.get("scope_rationale"):
            result.add(
                "out-of-scope document missing scope_rationale — violates OPF §3.6",
                path=f"corpus.documents[{i}]",
            )


def _check_citation_ref(
    ref: dict[str, Any] | None,
    path: str,
    result: ValidationResult,
    corpus_docs: dict[str, dict[str, Any]],
) -> None:
    """Assert a citation resolves: non-null, non-empty document_id, present in
    corpus.documents (unless 'template'), and version within that document's
    known version count. OPF §4 — every asserted citation MUST be traceable;
    a citation that cannot be resolved against the corpus is dangling.
    """
    if not ref:
        result.add("missing citation — violates OPF §4", path=path)
        return

    # document_id may be JSON null (hand-edited/foreign document) or, in
    # principle, any non-string type — guard the type before .strip() rather
    # than assuming the schema already rejected it (issue #72: normative
    # checks run even when schema validation already found errors).
    doc_id = ref.get("document_id") or ""
    if not isinstance(doc_id, str) or not doc_id.strip():
        result.add(
            "citation document_id is empty — citation is unresolvable, violates OPF §4",
            path=f"{path}.document_id",
        )
        return

    if doc_id == "template":
        return

    corpus_doc = corpus_docs.get(doc_id)
    if corpus_doc is None:
        result.add(
            f"citation cites unknown document_id {doc_id!r} "
            "(not 'template' and not present in corpus) — dangling citation, violates OPF §4",
            path=path,
        )
        return

    version = ref.get("version")
    if isinstance(version, int):
        max_version = corpus_doc.get("versions")
        if isinstance(max_version, int) and version > max_version:
            result.add(
                f"citation version {version} exceeds corpus.documents[{doc_id!r}].versions "
                f"({max_version}) — dangling citation, violates OPF §4",
                path=f"{path}.version",
            )
            return
        # Content-address rule (issue #185, §4): when the document publishes
        # version_files, every cited version MUST have an entry — otherwise
        # the citation names bytes no consumer can ever verify it holds.
        version_files = corpus_doc.get("version_files")
        if isinstance(version_files, list) and version_files:
            listed = {vf.get("version") for vf in version_files if isinstance(vf, dict)}
            if version not in listed:
                result.add(
                    f"citation version {version} has no version_files entry on "
                    f"corpus.documents[{doc_id!r}] — content-unaddressable citation, "
                    "violates OPF §4",
                    path=f"{path}.version",
                )


def _check_citations(doc: dict[str, Any], result: ValidationResult) -> None:
    """OPF §4: every asserted clause text must have a citation, and every citation
    must resolve against corpus.documents."""
    corpus_docs = _corpus_docs(doc)

    for i, clause in enumerate(doc.get("clauses", [])):
        if clause.get("our_standard") is not None:
            std = clause["our_standard"]
            # text may be JSON null on a hand-edited document — guard the
            # type before .strip() (issue #72).
            text = std.get("text") or ""
            if not isinstance(text, str) or not text.strip():
                result.add(
                    "our_standard.text is empty",
                    path=f"clauses[{i}].our_standard.text",
                )
            _check_citation_ref(
                std.get("source_ref"),
                f"clauses[{i}].our_standard.source_ref",
                result,
                corpus_docs,
            )

        for j, obs in enumerate(clause.get("observed_positions", [])):
            _check_citation_ref(
                obs.get("example_ref"),
                f"clauses[{i}].observed_positions[{j}].example_ref",
                result,
                corpus_docs,
            )

        rollup = clause.get("rollup", {})
        for j, obs in enumerate(rollup.get("fallbacks", [])):
            _check_citation_ref(
                obs.get("example_ref"),
                f"clauses[{i}].rollup.fallbacks[{j}].example_ref",
                result,
                corpus_docs,
            )
        for j, obs in enumerate(rollup.get("rejected", [])):
            _check_citation_ref(
                obs.get("example_ref"),
                f"clauses[{i}].rollup.rejected[{j}].example_ref",
                result,
                corpus_docs,
            )

    for i, concept in enumerate(doc.get("clause_library", [])):
        for j, form in enumerate(concept.get("accepted_forms", [])):
            _check_citation_ref(
                form.get("example_ref"),
                f"clause_library[{i}].accepted_forms[{j}].example_ref",
                result,
                corpus_docs,
            )


# ---------------------------------------------------------------------------
# v0.2 normative checks — same rules (§2.2, §3.6, §4), evidence-wrapped shape.
# `evidence.clauses` / `evidence.clause_library` replace top-level
# `clauses` / `clause_library`; `summary.historical_stance` (descriptive)
# replaces `rollup.position` (prescriptive). See OPF-SPEC.md §2.2, §3.5.
# ---------------------------------------------------------------------------

# historical_stance values stronger than "mixed" — i.e. everything except the
# two values that concede no operative signal (`mixed`, `no_signal`). Only
# our-paper drafting may license one of these (§2.2/§2.3).
_STANCES_STRONGER_THAN_MIXED = {"consistently_held", "usually_held", "usually_conceded"}


def _check_provenance_rule_v2(doc: dict[str, Any], result: ValidationResult) -> None:
    """OPF v0.2 §2.2: counterparty-paper observations must not set our_standard.

    The evidence-depth half of §2.2 (a historical_stance stronger than
    'mixed' requires >= min_evidence_n our-paper observations — issue #144)
    is checked separately by ``_check_evidence_depth_rule_v2``, which
    generalizes (and subsumes) the old "all observations are
    counterparty_paper" special case: zero our-paper observations is just the
    n_our_paper < min_evidence_n boundary for any min_evidence_n >= 1.
    """
    corpus_docs = _corpus_docs(doc)

    for i, clause in enumerate(doc.get("evidence", {}).get("clauses", [])):
        our_std = clause.get("our_standard")
        if our_std is not None:
            source = our_std.get("source_ref") or {}
            doc_id = source.get("document_id", "")
            if (
                doc_id
                and doc_id != "template"
                and doc_id in corpus_docs
                and corpus_docs[doc_id].get("provenance") == "counterparty_paper"
            ):
                result.add(
                    "our_standard sourced from a counterparty_paper document — violates OPF §2.2",
                    path=f"evidence.clauses[{i}].our_standard.source_ref",
                )


def _check_evidence_depth_rule_v2(
    doc: dict[str, Any], result: ValidationResult, min_evidence_n: int
) -> None:
    """OPF v0.2 §2.2 (issue #144): a historical_stance stronger than 'mixed'
    requires >= min_evidence_n distinct our-paper observed_positions AND a
    present our_standard — mirrors clause_position_compiler._derive_rollup's
    evidence-depth cap so the validator and compiler enforce the identical
    rule. One our-paper observation among many counterparty-paper ones must
    not license 'consistently_held'/'usually_held'/'usually_conceded' at the
    default min_evidence_n=2 (nor must zero our-paper observations at any
    min_evidence_n >= 1 — this subsumes the old "all counterparty" check).

    our_standard being sourced from a counterparty_paper document is already
    caught unconditionally by ``_check_provenance_rule_v2`` above; this
    function only adds the case that check does not cover — our_standard
    being entirely absent while the stance claims a settled position.
    """
    for i, clause in enumerate(doc.get("evidence", {}).get("clauses", [])):
        summary = clause.get("summary", {})
        stance = summary.get("historical_stance")
        if stance not in _STANCES_STRONGER_THAN_MIXED:
            continue

        observations = clause.get("observed_positions", [])
        n_our_paper = sum(1 for obs in observations if obs.get("provenance") == "our_paper")
        if n_our_paper < min_evidence_n:
            result.add(
                f"summary.historical_stance={stance!r} but only {n_our_paper} "
                f"our_paper observation(s) (< min_evidence_n={min_evidence_n}) — "
                "violates OPF §2.2",
                path=f"evidence.clauses[{i}].summary.historical_stance",
            )
            continue

        if clause.get("our_standard") is None:
            result.add(
                f"summary.historical_stance={stance!r} but our_standard is absent — "
                "violates OPF §2.2",
                path=f"evidence.clauses[{i}].our_standard",
            )


def _check_citations_v2(doc: dict[str, Any], result: ValidationResult) -> None:
    """OPF v0.2 §4: every asserted clause text must have a citation, and every
    citation must resolve against corpus.documents. Same rule as v0.1, applied
    to the evidence-wrapped shape (evidence.clauses[].summary.* instead of
    clauses[].rollup.*)."""
    corpus_docs = _corpus_docs(doc)

    for i, clause in enumerate(doc.get("evidence", {}).get("clauses", [])):
        if clause.get("our_standard") is not None:
            std = clause["our_standard"]
            # text may be JSON null on a hand-edited document — guard the
            # type before .strip() (issue #72).
            text = std.get("text") or ""
            if not isinstance(text, str) or not text.strip():
                result.add(
                    "our_standard.text is empty",
                    path=f"evidence.clauses[{i}].our_standard.text",
                )
            _check_citation_ref(
                std.get("source_ref"),
                f"evidence.clauses[{i}].our_standard.source_ref",
                result,
                corpus_docs,
            )

        for j, obs in enumerate(clause.get("observed_positions", [])):
            _check_citation_ref(
                obs.get("example_ref"),
                f"evidence.clauses[{i}].observed_positions[{j}].example_ref",
                result,
                corpus_docs,
            )

        summary = clause.get("summary", {})
        for j, obs in enumerate(summary.get("fallbacks", [])):
            _check_citation_ref(
                obs.get("example_ref"),
                f"evidence.clauses[{i}].summary.fallbacks[{j}].example_ref",
                result,
                corpus_docs,
            )
        for j, obs in enumerate(summary.get("rejected", [])):
            _check_citation_ref(
                obs.get("example_ref"),
                f"evidence.clauses[{i}].summary.rejected[{j}].example_ref",
                result,
                corpus_docs,
            )
        # issue #141: acceptable_if entries MAY be the legacy bare-string form
        # (schema still accepts it on input) or the {if,to,rationale,
        # observation_ref} triple a compiler emits. Only the triple form
        # carries a citation to check — a bare string has nothing to resolve.
        for j, entry in enumerate(summary.get("acceptable_if", [])):
            if isinstance(entry, dict):
                _check_citation_ref(
                    entry.get("observation_ref"),
                    f"evidence.clauses[{i}].summary.acceptable_if[{j}].observation_ref",
                    result,
                    corpus_docs,
                )

    for i, concept in enumerate(doc.get("evidence", {}).get("clause_library", [])):
        for j, form in enumerate(concept.get("accepted_forms", [])):
            _check_citation_ref(
                form.get("example_ref"),
                f"evidence.clause_library[{i}].accepted_forms[{j}].example_ref",
                result,
                corpus_docs,
            )


def _check_dynamics_v2(doc: dict[str, Any], result: ValidationResult) -> None:
    """OPF v0.2 §3.5.3 (issue #177) — negotiation-dynamics consistency.

    - ``summary.stance_detail``: ``held <= of`` (blocking — a held-rate
      exceeding its opportunity count is arithmetically impossible), plus a
      non-blocking coarse-consistency warning against the historical_stance
      bucket (a "consistently_held" clause whose own counts show concessions,
      or a "usually_conceded" clause whose counts show none, signals a
      producer bug even though neither is provably wrong on its own).
    - ``negotiation_trail[].ref``: must resolve against corpus.documents
      like every other citation (§4).
    - ``observed_at``: must parse as an ISO-8601 date, OR match the
      publish-time coarsened ``YYYY-Qn`` form (issue #188), when present — a
      malformed date is fabrication-adjacent, exactly what §3.5.3's
      never-fabricate rule exists to keep out.
    """
    corpus_docs = _corpus_docs(doc)

    for i, clause in enumerate(doc.get("evidence", {}).get("clauses", [])):
        summary = clause.get("summary", {})
        detail = summary.get("stance_detail")
        if isinstance(detail, dict):
            held, of = detail.get("held"), detail.get("of")
            if isinstance(held, int) and isinstance(of, int):
                if held > of:
                    result.add(
                        f"stance_detail.held={held} exceeds of={of} — "
                        "a held-rate cannot exceed its opportunity count (§3.5.3)",
                        path=f"evidence.clauses[{i}].summary.stance_detail",
                    )
                else:
                    stance = summary.get("historical_stance")
                    if stance == "consistently_held" and held < of:
                        result.add(
                            f"historical_stance='consistently_held' but stance_detail "
                            f"shows {of - held} concession(s) ({held}/{of}) — "
                            "counts contradict the enum bucket (§3.5.3)",
                            path=f"evidence.clauses[{i}].summary.stance_detail",
                            blocking=False,
                        )
                    elif stance == "usually_conceded" and of > 0 and held == of:
                        result.add(
                            f"historical_stance='usually_conceded' but stance_detail "
                            f"shows no concessions ({held}/{of}) — "
                            "counts contradict the enum bucket (§3.5.3)",
                            path=f"evidence.clauses[{i}].summary.stance_detail",
                            blocking=False,
                        )

        for j, entry in enumerate(clause.get("negotiation_trail", [])):
            _check_citation_ref(
                entry.get("ref"),
                f"evidence.clauses[{i}].negotiation_trail[{j}].ref",
                result,
                corpus_docs,
            )

        for j, obs in enumerate(clause.get("observed_positions", [])):
            observed_at = obs.get("observed_at")
            if observed_at is not None and not (
                isinstance(observed_at, str) and _QUARTER_DATE_RE.match(observed_at)
            ):
                try:
                    datetime.date.fromisoformat(observed_at)
                except (TypeError, ValueError):
                    result.add(
                        f"observed_at={observed_at!r} is not an ISO-8601 date (§3.5.3)",
                        path=f"evidence.clauses[{i}].observed_positions[{j}].observed_at",
                    )


def _check_posture_floor_conflict_v2(doc: dict[str, Any], result: ValidationResult) -> None:
    """OPF v0.2 §3.6 rule 3, issue #156: a Posture that softens language
    around a Floor-protected concept is a SHOULD-warn, not a hard error (the
    issue's Direction settles this, superseding §3.6 rule 3's older
    "validation error" wording for this slice). Non-blocking — surfaced via
    ``ValidationError(blocking=False)``, same convention as every other
    advisory finding in this validator.

    Deliberately deterministic/lexical (``posture.check_posture_floor_conflict``)
    — no LLM judge — mirrors this issue's "templated/assembled, not
    LLM-generated" scope boundary.
    """
    from playbook_engine.posture import check_posture_floor_conflict  # noqa: PLC0415

    system_prompt = doc.get("posture", {}).get("system_prompt", "")
    floor_invariants = doc.get("floor", {}).get("invariants", [])
    for message in check_posture_floor_conflict(system_prompt, floor_invariants):
        result.add(message, path="posture.system_prompt", blocking=False)


def _check_posture_interview_provenance_v2(doc: dict[str, Any], result: ValidationResult) -> None:
    """Non-blocking SHOULD-warn: a drafted ``posture.system_prompt`` with no
    ``posture.generation.interview`` record behind it (issue #133, skill QA
    audit finding #85).

    OPF-SPEC.md §3.6 rule 2 makes the interview record provenance and says
    "It MUST be retained" -- it lets an auditor see *why* the Posture says
    what it says. Neither the schema (``generation``/``interview`` are
    optional, with no conditional requirement) nor any prior validator check
    enforces that MUST, so a hand-edited or third-party playbook can strip
    the interview and still validate clean. Advisory only, same convention
    as every other SHOULD finding in this validator (see
    :func:`_check_floor_attribution`) -- this exists so a human reading
    ``validate``'s output sees a Posture with no traceable provenance,
    rather than assuming every ``system_prompt`` was genuinely interview-
    derived.

    An *empty* Posture (no ``system_prompt`` at all) is untouched by this
    check -- the schema explicitly allows that shape for a corpus-only
    compile with no interview yet run (see
    ``spec/playbook.schema-0.3.json``'s posture description, and
    ``test_v0_2_empty_posture_and_floor_are_valid``); there is nothing to
    provide provenance for until prose actually exists.
    """
    posture = doc.get("posture")
    if not isinstance(posture, dict):
        return
    system_prompt = posture.get("system_prompt")
    if not (isinstance(system_prompt, str) and system_prompt.strip()):
        return

    generation = posture.get("generation")
    interview = generation.get("interview") if isinstance(generation, dict) else None
    if isinstance(interview, list) and len(interview) > 0:
        return

    result.add(
        "posture.system_prompt is drafted but posture.generation.interview "
        "is absent/empty -- the interview record MUST be retained as "
        "provenance for the drafted Posture (OPF-SPEC.md §3.6 rule 2).",
        path="posture.generation.interview",
        blocking=False,
    )


def _check_invisible_chars(doc: dict[str, Any], result: ValidationResult) -> None:
    """Blocking error for zero-width/bidi-control characters anywhere in *doc*.

    Consumers treat these as prompt-injection markers and fail closed; a
    document carrying them is unusable downstream regardless of schema
    validity. Reports the first few offending paths, not all of them.
    """
    reported = 0

    def walk(value: Any, path: str) -> None:
        nonlocal reported
        if reported >= 5:
            return
        if isinstance(value, str):
            if _INVISIBLE_CHARS_RE.search(value):
                result.add(
                    "contains zero-width/bidi-control character(s) "
                    "(U+200B-U+200D, U+FEFF, U+202A-U+202E)",
                    path=path,
                )
                reported += 1
        elif isinstance(value, list):
            for i, v in enumerate(value):
                walk(v, f"{path}[{i}]")
        elif isinstance(value, dict):
            for k, v in value.items():
                walk(v, f"{path}.{k}" if path else str(k))

    walk(doc, "")


def _check_duplicate_ids(doc: dict[str, Any], result: ValidationResult) -> None:
    """Blocking error for duplicate clause/clause_library/invariant/document ids (issue #70).

    Nothing else enforces uniqueness here — the JSON Schema cannot express
    "unique across siblings" for these ids, and no other normative check
    catches it. A foreign or producer-bugged OPF doc carrying two clauses
    (or two clause_library entries, floor invariants, or corpus documents)
    with the same id makes :mod:`playbook_engine.export_profile`'s
    path-keyed sample/location maps silently collapse: a judge-flagged
    residue rewrite lands on only the LAST duplicate, shipping the first
    duplicate's flagged text unmodified (and the corresponding setter
    rewrites the wrong entry). Engine-generated docs never carry duplicate
    ids (they derive from unique taxonomy keys), so this only fires on
    hand-edited/foreign input — but it must fail loud there rather than
    silently mis-redact.

    Version-agnostic: uses :func:`playbook_clauses`/:func:`playbook_clause_library`
    so it applies to v0.1's top-level ``clauses``/``clause_library`` and
    v0.2/v0.3's ``evidence.clauses``/``evidence.clause_library`` alike, and
    ``floor.invariants``/``corpus.documents`` are identical in shape across
    all versions.
    """
    # Mirror playbook_clauses'/playbook_clause_library's own v0.2-precedence
    # condition exactly, so the reported path always points at the shape the
    # id actually came from (v0.1/hand-authored top-level vs. v0.2/v0.3
    # ``evidence.*``) instead of unconditionally naming the v0.1 shape, which
    # does not exist on a v0.2/v0.3 doc.
    evidence = doc.get("evidence")
    clause_prefix = (
        "evidence.clauses" if isinstance(evidence, dict) and "clauses" in evidence else "clauses"
    )
    clause_library_prefix = (
        "evidence.clause_library"
        if isinstance(evidence, dict) and "clause_library" in evidence
        else "clause_library"
    )

    seen_clause_ids: dict[str, int] = {}
    for i, clause in enumerate(playbook_clauses(doc)):
        if not isinstance(clause, dict):
            continue
        clause_id = clause.get("id")
        if not isinstance(clause_id, str):
            continue
        first_index = seen_clause_ids.get(clause_id)
        if first_index is not None:
            result.add(
                f"duplicate clause id {clause_id!r} (first seen at {clause_prefix}[{first_index}])",
                path=f"{clause_prefix}[{i}].id",
            )
        else:
            seen_clause_ids[clause_id] = i

    seen_concept_ids: dict[str, int] = {}
    for i, concept in enumerate(playbook_clause_library(doc)):
        if not isinstance(concept, dict):
            continue
        concept_id = concept.get("concept_id")
        if not isinstance(concept_id, str):
            continue
        first_index = seen_concept_ids.get(concept_id)
        if first_index is not None:
            result.add(
                f"duplicate clause_library concept_id {concept_id!r} "
                f"(first seen at {clause_library_prefix}[{first_index}])",
                path=f"{clause_library_prefix}[{i}].concept_id",
            )
        else:
            seen_concept_ids[concept_id] = i

    seen_invariant_ids: dict[str, int] = {}
    floor_section = doc.get("floor")
    invariants = floor_section.get("invariants") if isinstance(floor_section, dict) else None
    if not isinstance(invariants, list):
        # Hand-authored YAML can spell `floor:\n  invariants:` (a valueless
        # key, i.e. None) or `floor: {invariants: "x"}` — neither is a list,
        # but both must fall through to the schema check below rather than
        # raise here (issue #70 round 2).
        invariants = []
    for i, invariant in enumerate(invariants):
        if not isinstance(invariant, dict):
            continue
        invariant_id = invariant.get("id")
        if not isinstance(invariant_id, str):
            continue
        first_index = seen_invariant_ids.get(invariant_id)
        if first_index is not None:
            result.add(
                f"duplicate floor invariant id {invariant_id!r} "
                f"(first seen at floor.invariants[{first_index}])",
                path=f"floor.invariants[{i}].id",
            )
        else:
            seen_invariant_ids[invariant_id] = i

    seen_document_ids: dict[str, int] = {}
    corpus_section = doc.get("corpus")
    corpus_documents = corpus_section.get("documents") if isinstance(corpus_section, dict) else None
    if not isinstance(corpus_documents, list):
        # Same discipline as `invariants` above (`corpus: null`, `corpus: []`,
        # or a non-dict document entry) — this is the first check
        # `validate_document` runs, so it must not widen the crash surface
        # `_corpus_docs` already tolerates elsewhere in this module.
        corpus_documents = []
    for i, corpus_doc in enumerate(corpus_documents):
        if not isinstance(corpus_doc, dict):
            continue
        document_id = corpus_doc.get("document_id")
        if not isinstance(document_id, str):
            continue
        first_index = seen_document_ids.get(document_id)
        if first_index is not None:
            result.add(
                f"duplicate corpus document_id {document_id!r} "
                f"(first seen at corpus.documents[{first_index}])",
                path=f"corpus.documents[{i}].document_id",
            )
        else:
            seen_document_ids[document_id] = i


def _check_floor_attribution(doc: dict[str, Any], result: ValidationResult) -> None:
    """Non-blocking SHOULD-warn: name each ``floor.invariants[]`` entry that
    carries no structural attribution marker (issue #127).

    Three producers write into ``floor.invariants``, and each leaves a
    distinct, mechanically-checkable trace — see
    :func:`playbook_engine.floor_candidates.floor_invariant_attribution` for
    the three it recognizes (a hand-signed ``x_signed_by``, a Posture-
    interview Q4 promotion, or an accepted review-feedback candidate).
    Advisory only, same convention as every other SHOULD finding in this
    validator: the schema and the blocking checks above already accept an
    unattributed entry structurally (``floor.invariants[].id``/``statement``
    are the only required properties) — this exists purely so a human
    reading ``validate``'s output sees which invariants lack a traceable
    author, rather than assuming every entry in the Floor was genuinely
    signed off. It does not, by itself, prove an entry IS genuine — see
    that function's docstring for what this can and cannot catch.

    Version-agnostic (``floor.invariants`` is identical in shape across
    v0.1/v0.2/v0.3 — see :func:`_check_duplicate_ids`'s own floor handling),
    so this runs unconditionally rather than being gated to v0.2/v0.3 like
    :func:`_check_posture_floor_conflict_v2`.

    Deferred import (same reason as ``_check_posture_floor_conflict_v2``'s,
    just above) — :mod:`playbook_engine.floor_candidates` imports
    :func:`load_opf_file` from this module at ITS top level, so a top-level
    import here the other way would cycle.
    """
    from playbook_engine.floor_candidates import floor_invariant_attribution  # noqa: PLC0415

    floor_section = doc.get("floor")
    invariants = floor_section.get("invariants") if isinstance(floor_section, dict) else None
    if not isinstance(invariants, list):
        return
    for i, invariant in enumerate(invariants):
        if not isinstance(invariant, dict):
            continue
        if floor_invariant_attribution(invariant) is not None:
            continue
        invariant_id = invariant.get("id")
        label = f"{invariant_id!r} " if isinstance(invariant_id, str) else ""
        result.add(
            f"floor invariant {label}carries no structural attribution — no "
            "x_signed_by, and its rationale doesn't match a Posture-interview "
            "or review-feedback promotion marker; confirm a human actually "
            "authored/signed this hard line.",
            path=f"floor.invariants[{i}]",
            blocking=False,
        )


def _check_digest_v3(doc: dict[str, Any], result: ValidationResult) -> None:
    """OPF 0.3 digest consistency (normative, beyond the schema).

    When a `digest` section is present: its clause ids must exactly match the
    evidence section's clause ids (a digest describing different clauses than
    the document it ships in is worse than no digest), and no digest entry may
    carry `full_text` — the digest's size contract and the drill-down-via-
    example_ref design both depend on that.
    """
    digest = doc.get("digest")
    if digest is None:
        return
    if not isinstance(digest, dict):
        result.add("digest must be an object", path="digest")
        return

    evidence_ids = [c.get("id") for c in doc.get("evidence", {}).get("clauses", [])]
    digest_ids = [c.get("id") for c in digest.get("clauses", [])]
    if sorted(map(str, digest_ids)) != sorted(map(str, evidence_ids)):
        result.add(
            f"digest.clauses ids do not match evidence.clauses ids "
            f"({len(digest_ids)} vs {len(evidence_ids)})",
            path="digest.clauses",
        )

    def walk(value: Any, path: str) -> None:
        if isinstance(value, dict):
            for k, v in value.items():
                if k == "full_text":
                    result.add(
                        "digest must not carry full_text (use example_ref drill-down)",
                        path=path,
                    )
                walk(v, f"{path}.{k}")
        elif isinstance(value, list):
            for i, item in enumerate(value):
                walk(item, f"{path}.{i}")

    walk(digest, "digest")


def validate_document(
    doc: dict[str, Any], *, min_evidence_n: int = MIN_EVIDENCE_N
) -> ValidationResult:
    """Validate *doc* against its OPF schema and normative rules.

    Args:
        doc:            The OPF playbook document (dict, already parsed).
        min_evidence_n: Producer-configurable evidence-depth floor (issue
                        #144, config.provenance.min_evidence_n) enforced by
                        the v0.2 evidence-depth check
                        (``_check_evidence_depth_rule_v2``) — must match
                        whatever threshold the producing compiler used, or
                        this will reject a correctly-compiled document (or
                        worse, silently accept an under-evidenced one).
                        Defaults to ``clause_position_compiler.MIN_EVIDENCE_N``
                        (2), the same default the compiler uses absent an
                        explicit override, so the two stay aligned on one
                        rule when neither side customizes it.
    """
    result = ValidationResult()
    _check_invisible_chars(doc, result)
    _check_duplicate_ids(doc, result)
    _check_floor_attribution(doc, result)
    opf_version = doc.get("opf_version")
    schema = _load_schema(opf_version)

    validator_cls = jsonschema.validators.validator_for(schema)
    validator_cls.check_schema(schema)
    validator = validator_cls(schema)

    for err in sorted(validator.iter_errors(doc), key=lambda e: list(e.absolute_path)):
        path = _path_str(list(err.absolute_path))
        result.add(f"Schema: {err.message}", path=path)

    if _check_opf_version(doc, result):
        # _check_out_of_scope_rationale (§3.6) reads only `corpus`, which is
        # identical in shape across v0.1/v0.2 — one implementation serves both.
        if opf_version in ("0.2", "0.3"):
            _check_provenance_rule_v2(doc, result)
            _check_evidence_depth_rule_v2(doc, result, min_evidence_n)
            _check_out_of_scope_rationale(doc, result)
            _check_citations_v2(doc, result)
            _check_posture_floor_conflict_v2(doc, result)
            _check_posture_interview_provenance_v2(doc, result)
            _check_dynamics_v2(doc, result)
            if opf_version == "0.3":
                _check_digest_v3(doc, result)
        else:
            _check_provenance_rule(doc, result)
            _check_out_of_scope_rationale(doc, result)
            _check_citations(doc, result)

    return result


def load_opf_file(path: Path) -> dict[str, Any]:
    """Load JSON or YAML from path."""
    text = path.read_text(encoding="utf-8")
    loaded = yaml.safe_load(text) if path.suffix.lower() in {".yaml", ".yml"} else json.loads(text)
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected a JSON/YAML object, got {type(loaded).__name__}")
    return loaded
