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

# ``playbook publish`` (issue #188) coarsens ``observed_at`` from an ISO-8601
# date to ``YYYY-Qn`` (exact dates can identify a counterparty). A published
# document must still pass this validator, so the dynamics check below
# accepts either form — a malformed value in neither shape is still rejected.
_QUARTER_DATE_RE = re.compile(r"^\d{4}-Q[1-4]$")

_SCHEMA_PATH_V1 = Path(__file__).parent.parent / "spec" / "playbook.schema.json"
_SCHEMA_PATH_V2 = Path(__file__).parent.parent / "spec" / "playbook.schema-0.2.json"

# Versions this validator's normative checks know how to enforce:
#   "0.1" — clauses/clause_library top-level, rollup.position (§2.2, §3.6, §4)
#   "0.2" — clauses/clause_library under `evidence`, summary.historical_stance
#           (OPF-SPEC.md §2.2, §3.5, §4)
# Anything else (missing, or an unrecognized version) must fail loud rather
# than silently pass with an empty `doc.get("clauses", [])`.
_SUPPORTED_OPF_VERSIONS = {"0.1", "0.2"}

# Public alias — `playbook --version` (cli.py) reports these alongside the
# engine version so bug reports carry both, since engine version and OPF
# version drift independently (issue #176).
SUPPORTED_OPF_VERSIONS = _SUPPORTED_OPF_VERSIONS


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

    "0.2" gets the v0.2 (Evidence/Posture/Floor) schema; everything else
    (including missing/unrecognized versions) is validated against the v0.1
    schema, whose `const: "0.1"` on `opf_version` produces the expected schema
    error for those cases — preserving pre-existing v0.1 behavior byte-for-byte.
    """
    schema_path = _SCHEMA_PATH_V2 if opf_version == "0.2" else _SCHEMA_PATH_V1
    with schema_path.open() as f:
        return json.load(f)  # type: ignore[no-any-return]


def _path_str(path: list[str | int]) -> str:
    return ".".join(str(p) for p in path) if path else "<root>"


def _corpus_docs(doc: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {d["document_id"]: d for d in doc.get("corpus", {}).get("documents", [])}


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

    doc_id = ref.get("document_id", "")
    if not doc_id.strip():
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
            if not std.get("text", "").strip():
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
            if not std.get("text", "").strip():
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
        if opf_version == "0.2":
            _check_provenance_rule_v2(doc, result)
            _check_evidence_depth_rule_v2(doc, result, min_evidence_n)
            _check_out_of_scope_rationale(doc, result)
            _check_citations_v2(doc, result)
            _check_posture_floor_conflict_v2(doc, result)
            _check_dynamics_v2(doc, result)
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
