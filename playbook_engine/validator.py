"""OPF playbook validator.

Validates a playbook JSON/YAML against:
  1. spec/playbook.schema-0.4.json (JSON Schema draft 2020-12)
  2. Normative rules the schema cannot express (OPF §3.5, §3.6, §4)

The engine reads and writes exactly one format, OPF 0.4 (issue #238). A
document claiming any other ``opf_version`` — including the retired 0.1,
0.2 and 0.3 — is rejected with an "unsupported opf_version" error and no
further checks.
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

from playbook_engine.canonicalize import compute_section_digests, content_hash
from playbook_engine.opf_accessors import playbook_clauses

# A date may be coarsened from an ISO-8601 date to ``YYYY-Qn`` (exact dates
# can identify a counterparty). The date checks below accept either form — a
# malformed value in neither shape is still rejected.
_QUARTER_DATE_RE = re.compile(r"^\d{4}-Q[1-4]$")

#: The one OPF version this validator (and the engine) supports.
OPF_VERSION = "0.4"

_SCHEMA_PATH = Path(__file__).parent.parent / "spec" / "playbook.schema-0.4.json"

# Public — `playbook --version` (cli.py) reports these alongside the engine
# version so bug reports carry both, since engine version and OPF version
# drift independently (issue #176).
SUPPORTED_OPF_VERSIONS = frozenset({OPF_VERSION})

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


def _load_schema() -> dict[str, Any]:
    """Load the OPF 0.4 schema."""
    with _SCHEMA_PATH.open() as f:
        return json.load(f)  # type: ignore[no-any-return]


def _path_str(path: list[str | int]) -> str:
    return ".".join(str(p) for p in path) if path else "<root>"


def _corpus_docs(doc: dict[str, Any]) -> dict[str, dict[str, Any]]:
    # A hand-edited/foreign corpus document may omit document_id entirely (or
    # carry a non-string one) — skip it rather than KeyError; the schema pass
    # already reports the missing/invalid field, and a document that can't be
    # keyed can't be cited by document_id anyway (see issue #72).
    corpus = doc.get("corpus")
    documents = corpus.get("documents") if isinstance(corpus, dict) else None
    return {
        d["document_id"]: d
        for d in (documents if isinstance(documents, list) else [])
        if isinstance(d, dict) and isinstance(d.get("document_id"), str)
    }


def _check_opf_version(doc: dict[str, Any], result: ValidationResult) -> bool:
    """Fail loud on an opf_version this validator doesn't support.

    Returns True if the document claims the supported version and the checks
    may proceed; False otherwise — the shape of any other version (including
    the retired 0.1-0.3) is not this validator's to check, so validating it
    against the 0.4 schema would only bury the real problem under schema
    noise.
    """
    version = doc.get("opf_version")
    # isinstance first: a hand-edited list/dict value is unhashable.
    if not isinstance(version, str) or version not in SUPPORTED_OPF_VERSIONS:
        supported = ", ".join(sorted(SUPPORTED_OPF_VERSIONS))
        result.add(
            f"unsupported opf_version {version!r} (supported: {supported}) — the engine "
            "reads and writes only OPF 0.4; the 0.1-0.3 formats were retired "
            "(spec/CHANGELOG.md). No other check was run.",
            path="opf_version",
        )
        return False
    return True


def _check_out_of_scope_rationale(doc: dict[str, Any], result: ValidationResult) -> None:
    """OPF §3.6: out-of-scope docs must carry scope_rationale."""
    corpus = doc.get("corpus")
    documents = corpus.get("documents") if isinstance(corpus, dict) else None
    for i, corpus_doc in enumerate(documents if isinstance(documents, list) else []):
        if not isinstance(corpus_doc, dict):
            continue  # the schema pass reports a non-object entry
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


def _check_posture_floor_conflict(doc: dict[str, Any], result: ValidationResult) -> None:
    """OPF §3.6 rule 3, issue #156: a Posture that softens language
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


def _check_posture_interview_provenance(doc: dict[str, Any], result: ValidationResult) -> None:
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
    ``spec/playbook.schema-0.4.json``'s posture description, and
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


def _check_perspective_present(doc: dict[str, Any], result: ValidationResult) -> None:
    """Non-blocking SHOULD-warn: no top-level ``perspective`` (issue #212).

    OPF-SPEC.md §3.1 marks the field OPTIONAL but states in the same breath
    that "an open-standard OPF instance must say who 'us' is — negotiation
    knowledge is meaningless without it", and the schema repeats that
    sentence as the field's own ``description``. The schema cannot enforce
    it: ``perspective`` is absent from the top-level ``required`` list, and
    the 1.0 stability policy (spec/CHANGELOG.md, issue #113) forbids adding
    a new REQUIRED field in a 1.x release. So the prose MUST has no
    mechanical backing and a perspective-less playbook validates clean.

    That gap is not cosmetic. A consumer that cannot tell which side it
    acts for applies symmetric judgment to a one-sided clause — the
    reported failure is a reviewer redlining a one-sided IP grant *against*
    its own principal, fluently and in the wrong direction. Warning here is
    the additive-only way to make the omission loud, same convention as
    every other SHOULD finding in this validator (see
    :func:`_check_posture_interview_provenance`).

    A *partial* perspective needs no warning: the schema requires ``party``
    and ``counterparty_type`` together with ``additionalProperties: false``,
    so an incomplete block is already a blocking schema error above.
    """
    if "perspective" not in doc:
        result.add(
            "top-level perspective is absent -- an OPF instance MUST say who "
            "'us' is (OPF-SPEC.md §3.1). A consumer cannot tell which party "
            "this playbook reviews as, and will judge one-sided clauses "
            "symmetrically. Set perspective.party and "
            "perspective.counterparty_type in the producing config.",
            path="perspective",
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
    """Blocking error for duplicate clause/invariant/document ids (issue #70).

    Nothing else enforces uniqueness here — the JSON Schema cannot express
    "unique across siblings" for these ids, and no other normative check
    catches it. A foreign or producer-bugged OPF doc carrying two clauses
    (or two floor invariants, or corpus documents) with the same id makes any
    consumer that keys by id silently collapse them. Engine-generated docs
    never carry duplicate ids (they derive from unique taxonomy keys), so this
    only fires on hand-edited/foreign input — but it must fail loud there.
    (Precedent id uniqueness is checked with the rest of the precedent
    record, :func:`_check_precedent`.)
    """
    clause_prefix = "evidence.clauses"

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

    Two producers write into ``floor.invariants``, and each leaves a
    distinct, mechanically-checkable trace — see
    :func:`playbook_engine.floor_candidates.floor_invariant_attribution` for
    the two it recognizes (a hand-signed ``x_signed_by``, or a Posture-
    interview Q4 promotion).
    Advisory only, same convention as every other SHOULD finding in this
    validator: the schema and the blocking checks above already accept an
    unattributed entry structurally (``floor.invariants[].id``/``statement``
    are the only required properties) — this exists purely so a human
    reading ``validate``'s output sees which invariants lack a traceable
    author, rather than assuming every entry in the Floor was genuinely
    signed off. It does not, by itself, prove an entry IS genuine — see
    that function's docstring for what this can and cannot catch.

    Deferred import (same reason as ``_check_posture_floor_conflict``'s,
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
            "promotion marker; confirm a human actually "
            "authored/signed this hard line.",
            path=f"floor.invariants[{i}]",
            blocking=False,
        )


def _check_digest_shape(doc: dict[str, Any], result: ValidationResult) -> None:
    """Digest consistency rules that name the most specific cause.

    When a `digest` section is present: its clause ids must exactly match the
    evidence section's clause ids (a digest describing different clauses than
    the document it ships in is worse than no digest), and no digest entry may
    carry `full_text` — the digest's size contract and the drill-down-via-
    citation design both depend on that.
    """
    digest = doc.get("digest")
    if digest is None:
        return
    if not isinstance(digest, dict):
        result.add("digest must be an object", path="digest")
        return

    evidence_ids = [c.get("id") for c in _evidence_list(doc, "clauses") if isinstance(c, dict)]
    digest_clauses = digest.get("clauses")
    digest_ids = [
        c.get("id")
        for c in (digest_clauses if isinstance(digest_clauses, list) else [])
        if isinstance(c, dict)
    ]
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


# ---------------------------------------------------------------------------
# Precedent-record checks — the verdict-free per-deal precedent record (#223).
# ---------------------------------------------------------------------------


def _evidence_list(doc: dict[str, Any], key: str) -> list[Any]:
    evidence = doc.get("evidence")
    value = evidence.get(key) if isinstance(evidence, dict) else None
    return value if isinstance(value, list) else []


def _check_citations(doc: dict[str, Any], result: ValidationResult) -> None:
    """OPF 0.4 §4: every asserted text carries a citation that resolves.

    Covers ``evidence.clauses[].our_standard.source_ref`` and, per precedent,
    ``signed_text.ref``, ``opening_text.ref`` and ``refused_asks[].ref``.
    The digest's refs are copies of these (enforced by the digest equality
    check), so they need no separate pass.
    """
    corpus_docs = _corpus_docs(doc)
    for i, clause in enumerate(_evidence_list(doc, "clauses")):
        if not isinstance(clause, dict):
            continue
        std = clause.get("our_standard")
        if isinstance(std, dict):
            text = std.get("text") or ""
            if not isinstance(text, str) or not text.strip():
                result.add(
                    "our_standard.text is empty", path=f"evidence.clauses[{i}].our_standard.text"
                )
            _check_citation_ref(
                std.get("source_ref"),
                f"evidence.clauses[{i}].our_standard.source_ref",
                result,
                corpus_docs,
            )
    for i, record in enumerate(_evidence_list(doc, "precedent")):
        if not isinstance(record, dict):
            continue
        for key in ("signed_text", "opening_text"):
            entry = record.get(key)
            if isinstance(entry, dict):
                _check_citation_ref(
                    entry.get("ref"), f"evidence.precedent[{i}].{key}.ref", result, corpus_docs
                )
        for j, ask in enumerate(record.get("refused_asks") or []):
            if isinstance(ask, dict):
                _check_citation_ref(
                    ask.get("ref"),
                    f"evidence.precedent[{i}].refused_asks[{j}].ref",
                    result,
                    corpus_docs,
                )


def _check_precedent(doc: dict[str, Any], result: ValidationResult) -> None:
    """OPF 0.4 precedent cross-checks (normative, beyond the schema).

    - every precedent ``id`` is unique and equals
      ``precedent.precedent_id(agreement_type.id, document_id, taxonomy_id,
      signed_text.text)`` — a stable, recomputable id;
    - at most one precedent per (document_id, taxonomy_id): the deal is the
      unit of precedent (issue #216);
    - every ``taxonomy_id`` names an ``evidence.clauses`` entry;
    - every ``document_id`` resolves to ``corpus.documents`` and ``signed``
      agrees with that document's ``signed_version`` (when recorded);
    - ``standard: true`` requires a ``signed_text``; ``signed_at`` is a real
      date (or a ``YYYY-Qn`` quarter);
    - each clause's ``n_deals``/``n_signed_standard``/``n_variants``/
      ``n_refused`` equal what ``precedent`` implies
      (``precedent.clause_counts``, grouping with the document's own
      ``perspective.party``);
    - paper side (issue #225) — metadata only, so these are honesty checks,
      never gates: see :func:`_check_paper`.
    """
    from playbook_engine.opf_accessors import perspective_party  # noqa: PLC0415
    from playbook_engine.precedent import clause_counts, precedent_id  # noqa: PLC0415

    corpus_docs = _corpus_docs(doc)
    agreement_type = doc.get("agreement_type")
    agreement_type_id = agreement_type.get("id") if isinstance(agreement_type, dict) else None
    clauses = [c for c in _evidence_list(doc, "clauses") if isinstance(c, dict)]
    clause_tids = {c.get("taxonomy_id") for c in clauses}
    precedent = _evidence_list(doc, "precedent")

    seen_ids: dict[str, int] = {}
    seen_pairs: dict[tuple[Any, Any], int] = {}
    for i, record in enumerate(precedent):
        if not isinstance(record, dict):
            continue
        path = f"evidence.precedent[{i}]"
        pid, doc_id, tid = record.get("id"), record.get("document_id"), record.get("taxonomy_id")
        if isinstance(pid, str):
            if pid in seen_ids:
                result.add(
                    f"duplicate precedent id {pid!r} (first seen at evidence.precedent[{seen_ids[pid]}])",
                    path=f"{path}.id",
                )
            else:
                seen_ids[pid] = i
            signed_text = record.get("signed_text")
            text = signed_text.get("text") if isinstance(signed_text, dict) else None
            if (
                isinstance(agreement_type_id, str)
                and isinstance(doc_id, str)
                and isinstance(tid, str)
            ):
                expected = precedent_id(
                    agreement_type_id, doc_id, tid, text if isinstance(text, str) else None
                )
                if pid != expected:
                    result.add(
                        f"precedent id {pid!r} does not match the id recomputed from "
                        f"(agreement_type.id, document_id, taxonomy_id, signed_text) ({expected!r})",
                        path=f"{path}.id",
                    )
        pair = (doc_id, tid)
        if pair in seen_pairs:
            result.add(
                f"more than one precedent for deal {doc_id!r} and clause {tid!r} "
                f"(first at evidence.precedent[{seen_pairs[pair]}]) — the deal is the "
                "unit of precedent",
                path=path,
            )
        else:
            seen_pairs[pair] = i
        if tid not in clause_tids:
            result.add(
                f"precedent taxonomy_id {tid!r} names no evidence.clauses entry",
                path=f"{path}.taxonomy_id",
            )
        corpus_doc = corpus_docs.get(doc_id) if isinstance(doc_id, str) else None
        if corpus_doc is None:
            result.add(
                f"precedent document_id {doc_id!r} is not in corpus.documents — dangling deal",
                path=f"{path}.document_id",
            )
        elif "signed_version" in corpus_doc and isinstance(record.get("signed"), bool):
            has_signed_copy = corpus_doc.get("signed_version") is not None
            if record["signed"] != has_signed_copy:
                result.add(
                    f"precedent signed={record['signed']} but corpus.documents[{doc_id!r}]"
                    f".signed_version={corpus_doc.get('signed_version')!r}",
                    path=f"{path}.signed",
                )
        if record.get("standard") is True and not isinstance(record.get("signed_text"), dict):
            result.add(
                "precedent standard=true but signed_text is null — nothing was signed to "
                "be standard",
                path=f"{path}.standard",
            )
        signed_at = record.get("signed_at")
        if isinstance(signed_at, str) and not _QUARTER_DATE_RE.match(signed_at):
            try:
                datetime.date.fromisoformat(signed_at)
            except ValueError:
                result.add(
                    f"signed_at={signed_at!r} is not an ISO-8601 date", path=f"{path}.signed_at"
                )

    _check_paper(doc, result)

    records = [p for p in precedent if isinstance(p, dict)]
    party = perspective_party(doc)
    for i, clause in enumerate(_evidence_list(doc, "clauses")):
        if not isinstance(clause, dict) or not isinstance(clause.get("taxonomy_id"), str):
            continue
        expected_counts = clause_counts(clause["taxonomy_id"], records, party=party)
        for key, expected_n in expected_counts.items():
            if clause.get(key) != expected_n:
                result.add(
                    f"{key}={clause.get(key)!r} but evidence.precedent implies {expected_n}",
                    path=f"evidence.clauses[{i}].{key}",
                )


def _check_paper(doc: dict[str, Any], result: ValidationResult) -> None:
    """OPF 0.4 paper-side cross-checks (issue #225).

    Paper side is three-valued deal metadata that gates nothing, so these
    only check that the document tells one consistent story about it:

    - a precedent's ``paper`` agrees with its deal's
      ``corpus.documents[].provenance`` (``precedent.paper_of_corpus_document``:
      ``our_paper`` -> ``"ours"``, ``counterparty_paper`` -> ``"theirs"``, an
      ambiguous detection -> ``"unknown"``). A side MUST match; an ambiguous
      detection MUST be ``"unknown"`` (never coerced to a side). ``"unknown"``
      against an unflagged document is accepted — the two-valued corpus field
      cannot say "unknown", so a record may honestly withhold a side;
    - a numeric ``paper_confidence`` equals the document's numeric
      ``provenance_confidence``;
    - every precedent of one deal carries the same ``paper`` /
      ``paper_basis`` / ``paper_confidence`` (it is a fact about the deal);
    - an unknown-paper deal contributes no ``our_standard``;
    - with no canonical template configured (``baseline.has_canonical_template``
      false) an unknown-paper deal is never ``standard`` — so it is excluded
      from ``n_signed_standard``. With a template, ``standard`` is the exact
      match against it and paper side does not enter into it.
    """
    from playbook_engine.precedent import (  # noqa: PLC0415
        PAPER_UNKNOWN,
        paper_of_corpus_document,
    )

    corpus_docs = _corpus_docs(doc)
    baseline = doc.get("baseline")
    has_template = isinstance(baseline, dict) and baseline.get("has_canonical_template") is True
    paper_by_deal: dict[Any, tuple[Any, Any, Any]] = {}
    first_by_deal: dict[Any, int] = {}
    unknown_deals: set[Any] = set()
    for i, record in enumerate(_evidence_list(doc, "precedent")):
        if not isinstance(record, dict):
            continue
        path = f"evidence.precedent[{i}]"
        doc_id = record.get("document_id")
        paper = record.get("paper")
        triple = (paper, record.get("paper_basis"), record.get("paper_confidence"))
        if doc_id in paper_by_deal:
            if paper_by_deal[doc_id] != triple:
                result.add(
                    f"precedent paper/paper_basis/paper_confidence {list(triple)!r} differs "
                    f"from deal {doc_id!r}'s first precedent "
                    f"(evidence.precedent[{first_by_deal[doc_id]}]: "
                    f"{list(paper_by_deal[doc_id])!r}) — paper side is a fact about the deal",
                    path=f"{path}.paper",
                )
        else:
            paper_by_deal[doc_id] = triple
            first_by_deal[doc_id] = i
        corpus_doc = corpus_docs.get(doc_id) if isinstance(doc_id, str) else None
        if paper == PAPER_UNKNOWN or (
            corpus_doc is not None and paper_of_corpus_document(corpus_doc) == PAPER_UNKNOWN
        ):
            unknown_deals.add(doc_id)
        if corpus_doc is not None and isinstance(paper, str):
            expected = paper_of_corpus_document(corpus_doc)
            if paper != expected and paper != PAPER_UNKNOWN:
                result.add(
                    f"precedent paper={paper!r} but corpus.documents[{doc_id!r}] records "
                    f"provenance={corpus_doc.get('provenance')!r}"
                    + (
                        " with provenance_is_ambiguous=true (an ambiguous detection is "
                        "'unknown', never a side)"
                        if corpus_doc.get("provenance_is_ambiguous") is True
                        else f" (paper {expected!r})"
                    ),
                    path=f"{path}.paper",
                )
            doc_conf = corpus_doc.get("provenance_confidence")
            rec_conf = record.get("paper_confidence")
            if (
                isinstance(doc_conf, (int, float))
                and isinstance(rec_conf, (int, float))
                and float(doc_conf) != float(rec_conf)
            ):
                result.add(
                    f"precedent paper_confidence={rec_conf!r} but corpus.documents"
                    f"[{doc_id!r}].provenance_confidence={doc_conf!r}",
                    path=f"{path}.paper_confidence",
                )
        if not has_template and paper == PAPER_UNKNOWN and record.get("standard") is True:
            result.add(
                "precedent standard=true on an unknown-paper deal with no canonical "
                "template configured — with no template there is no standard to match "
                "but our own paper, which this deal is not known to be",
                path=f"{path}.standard",
            )

    for i, clause in enumerate(_evidence_list(doc, "clauses")):
        if not isinstance(clause, dict):
            continue
        std = clause.get("our_standard")
        source = std.get("source_ref") if isinstance(std, dict) else None
        src_doc = source.get("document_id") if isinstance(source, dict) else None
        if not isinstance(src_doc, str) or src_doc == "template":
            continue
        if src_doc in unknown_deals or (
            src_doc in corpus_docs
            and paper_of_corpus_document(corpus_docs[src_doc]) == PAPER_UNKNOWN
        ):
            result.add(
                f"our_standard sourced from deal {src_doc!r}, whose paper side is "
                "unknown — an unknown-paper deal contributes no our_standard",
                path=f"evidence.clauses[{i}].our_standard.source_ref",
            )


def _check_digest(doc: dict[str, Any], result: ValidationResult) -> None:
    """A present ``digest`` MUST equal ``build_digest(document)``.

    The digest is a pure function of the document (issue #223) — an
    embedded digest that differs from a recomputation describes evidence the
    document does not carry. The id-match and no-``full_text`` rules
    (:func:`_check_digest_shape`) run first so a mismatch names its most
    specific cause.
    """
    digest = doc.get("digest")
    if digest is None:
        return
    _check_digest_shape(doc, result)
    if not isinstance(digest, dict):
        return
    from playbook_engine.digest import build_digest  # noqa: PLC0415

    try:
        expected = build_digest(doc)
    except (TypeError, ValueError, KeyError, AttributeError) as exc:
        result.add(f"digest cannot be recomputed from this document: {exc}", path="digest")
        return
    if digest != expected:
        differing = sorted(
            k for k in set(digest) | set(expected) if digest.get(k) != expected.get(k)
        )
        result.add(
            "digest does not equal build_digest(document) — it was edited, or "
            f"built from different evidence (differs in: {', '.join(differing)})",
            path="digest",
        )


def _check_identity_hash(doc: dict[str, Any], result: ValidationResult) -> None:
    """Issue #143 / #178: a present ``identity.content_hash``
    or ``identity.section_digests`` must match what
    :mod:`playbook_engine.canonicalize` recomputes over the document's
    current content.

    ``validate_document`` runs schema + normative checks but, before this
    check, never recomputed identity — so a hand-edited playbook (which the
    skill/spec forbid precisely because it corrupts the hash) or any
    stale-hash producer bug passed ``playbook validate`` exit 0, while a
    downstream consumer verifying ``identity.content_hash`` would reject the
    artifact. This closes that gap directly in the validator, the one local
    command positioned as the integrity gate.

    ``identity`` (and each of its sub-fields) is OPTIONAL — not every
    producer populates it (see the schema's own description) — so this is a
    pure no-op when ``identity``/``content_hash``/``section_digests`` is
    absent or malformed; the schema check above already reports a malformed
    ``identity`` shape. When a value IS present, though, it is asserting a
    verifiable fact about the document's own bytes, so a mismatch is
    blocking, not advisory — an unverifiable identity is worse than none.

    ``section_digests`` is compared key-by-key against what's actually
    present in the document (not the full recomputed dict) because
    ``curation`` is an optional key within ``section_digests`` too (schema:
    only evidence/posture/floor are ``required``) — a document that omits it
    has asserted nothing about the curation digest, so there is nothing to
    contradict.
    """
    identity = doc.get("identity")
    if not isinstance(identity, dict):
        return

    doc_hash = identity.get("content_hash")
    if isinstance(doc_hash, str):
        expected_hash = content_hash(doc)
        if doc_hash != expected_hash:
            result.add(
                f"identity.content_hash {doc_hash!r} does not match the hash recomputed "
                f"over the document's current content ({expected_hash!r}) — the document "
                "was hand-edited or otherwise changed after its identity was stamped "
                "(OPF-SPEC.md §3.10/§8)",
                path="identity.content_hash",
            )

    doc_digests = identity.get("section_digests")
    if isinstance(doc_digests, dict):
        expected_digests = compute_section_digests(doc)
        for name, doc_digest in doc_digests.items():
            if not isinstance(doc_digest, str):
                continue
            expected_digest = expected_digests.get(name)
            if expected_digest is not None and doc_digest != expected_digest:
                result.add(
                    f"identity.section_digests.{name} {doc_digest!r} does not match "
                    f"the digest recomputed over the current {name!r} section "
                    f"({expected_digest!r}) — the section was changed after its "
                    "identity was stamped (OPF-SPEC.md §3.10/§8)",
                    path=f"identity.section_digests.{name}",
                )


def validate_document(doc: dict[str, Any]) -> ValidationResult:
    """Validate *doc* against the OPF 0.4 schema and normative rules.

    A document whose ``opf_version`` is not "0.4" gets one blocking
    "unsupported opf_version" error and no other check (issue #238).

    Args:
        doc: The OPF playbook document (dict, already parsed).
    """
    result = ValidationResult()
    if not _check_opf_version(doc, result):
        return result

    _check_invisible_chars(doc, result)
    _check_duplicate_ids(doc, result)
    _check_floor_attribution(doc, result)

    schema = _load_schema()
    validator_cls = jsonschema.validators.validator_for(schema)
    validator_cls.check_schema(schema)
    validator = validator_cls(schema)
    for err in sorted(validator.iter_errors(doc), key=lambda e: list(e.absolute_path)):
        path = _path_str(list(err.absolute_path))
        result.add(f"Schema: {err.message}", path=path)

    # Paper side never gates anything (owner decision 2026-09-13 (b)) and
    # there is no stance to cap, so these are honesty and integrity checks.
    _check_out_of_scope_rationale(doc, result)
    _check_citations(doc, result)
    _check_precedent(doc, result)
    _check_posture_floor_conflict(doc, result)
    _check_posture_interview_provenance(doc, result)
    _check_perspective_present(doc, result)
    _check_identity_hash(doc, result)
    _check_digest(doc, result)
    return result


def load_opf_file(path: Path) -> dict[str, Any]:
    """Load JSON or YAML from path."""
    text = path.read_text(encoding="utf-8")
    loaded = yaml.safe_load(text) if path.suffix.lower() in {".yaml", ".yml"} else json.loads(text)
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected a JSON/YAML object, got {type(loaded).__name__}")
    return loaded
