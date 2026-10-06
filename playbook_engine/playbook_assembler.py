"""Playbook assembler — final L5 stage.

Assembles the full OPF playbook document — by default OPF 0.4 (issue #223):
the verdict-free per-deal precedent record as ``evidence``
(``playbook_engine/precedent.py``) plus a digest_version 3 digest;
``opf_version="0.3"`` keeps the 0.3 shape (evidence-wrapped clauses with a
descriptive ``summary.historical_stance``, digest v2) for one release. Both
carry empty-but-present ``posture``/``floor``, an embedded ``curation``
overlay of attorney-pinned positions that survives recompile (issue #147),
and an ``identity`` block carrying ``content_hash`` + per-section digests
(issue #143) — and ``write_playbook`` writes ``playbook.opf.json``.  The assembled document is
self-validated via the built-in validator before any data is written to
disk.

API
---
``assemble_playbook()`` — assemble + validate; raises ``AssemblyError`` on
                          blocking validation failures.
``write_playbook()``    — atomic write of a playbook dict to ``.opf.json``.

``generated_at`` (ISO-8601 datetime) is always supplied by the caller so that
the assembler itself remains deterministic and testable without time mocking.
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from playbook_engine.canonicalize import (
    canonicalize,
    compute_section_digests,
    content_hash,
    sha256_hex,
)
from playbook_engine.clause_library_compiler import ClauseConcept
from playbook_engine.clause_position_compiler import (
    MIN_EVIDENCE_N,
    ClausePosition,
    UnclassifiedCoverage,
)
from playbook_engine.curation import merge_curation
from playbook_engine.digest import build_digest
from playbook_engine.observation_builder import Observation, RoundMove
from playbook_engine.opf_accessors import clause_stance, perspective_party
from playbook_engine.precedent import (
    build_precedent_evidence,
    build_x_judgments,
    restamp_evidence,
)
from playbook_engine.validator import validate_document

# OPF 0.4 (issue #223) = the verdict-free per-deal precedent record as the
# evidence shape, plus a digest_version 3 digest. The assembler emits 0.4 by
# default; ``opf_version="0.3"`` keeps the 0.3 shape (0.2 evidence + digest
# v2) for one release. Validation accepts 0.1 through 0.4.
_OPF_VERSION = "0.4"

#: OPF versions ``assemble_playbook`` can emit.
EMITTABLE_OPF_VERSIONS = ("0.3", "0.4")
_COMPILER_NAME = "playbook-engine"

# Zero-width and bidirectional-control characters (ZWSP/ZWNJ/ZWJ/BOM and the
# LRE..RLO embedding range). Extraction preserves them from source documents,
# but downstream consumers treat them as prompt-injection markers — a
# downstream review engine's fail-closed injection scan rejects exactly this
# set — so the assembled document must never carry them (issue:
# pre-derivation QA).
_INVISIBLE_CHARS_RE = re.compile("[\u200b-\u200d\ufeff\u202a-\u202e]")


def _strip_invisible(value: Any) -> Any:
    """Recursively remove invisible/bidi-control characters from all strings."""
    if isinstance(value, str):
        return _INVISIBLE_CHARS_RE.sub("", value)
    if isinstance(value, list):
        return [_strip_invisible(v) for v in value]
    if isinstance(value, dict):
        return {k: _strip_invisible(v) for k, v in value.items()}
    return value


# Keys corpus.documents[].version_ingest[] may carry into the PUBLISHED
# playbook — mirrors spec/playbook.schema-0.3.json's (and -0.2.json's,
# identical here) corpus.documents.items.properties.version_ingest.items.
# properties exactly, whose additionalProperties:false rejects anything
# else. corpus_documents (as read from corpus_manifest.json) can carry
# richer, engine-internal-only keys not in this set — e.g. "reason"
# (extraction.ExtractorLabel.reason, issue #81) or "signature_block_span"
# (issue #217), additive to
# corpus_manifest.json/review.json but never part of the public OPF schema
# — see _sanitize_corpus_documents_for_schema below, which strips down to
# exactly this set before assembly. A test (test_playbook_assembler.py)
# asserts this set stays in sync with the schema's actual property set:
# used as a strip-list, drift in the OTHER direction (a future schema
# addition silently stripped from every published playbook) would
# otherwise fail silently.
_VERSION_INGEST_SCHEMA_KEYS = frozenset({"version", "status", "error", "extractor"})

# Engine-internal corpus document keys that never reach the published
# ``corpus.documents[]`` (additionalProperties:false). "dropped_observations"
# (issue #216) is the per-document count of net-diff rows that produced no
# observation, by reason — assemble_playbook sums it into
# ``corpus.stats.dropped_observations`` instead.
_CORPUS_DOCUMENT_INTERNAL_KEYS = frozenset({"dropped_observations"})


def _dropped_observation_stats(corpus_documents: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Sum each document's ``dropped_observations`` into one stats block.

    Returns ``None`` when nothing was dropped anywhere, so a corpus with no
    dropped rows carries no empty placeholder. The reasons are
    ``observation_builder.DROPPED_SURVIVES_IN_TERMINAL`` (a clause with no
    signed slot whose normalized text still occurs verbatim in the signed
    version, fill-in blanks aside, e.g. a relocation the aligner left
    unpaired — not removed at all; text narrowed or replaced before signing
    never counts here),
    ``DROPPED_ORIGIN_UNDETERMINED`` (a clause removed before signing whose
    origin — our standard or their ask — cannot be determined because there
    is no standard text for it) and ``DROPPED_STANDARD_REMOVED_UNSIGNED``
    (our standard language removed in a deal with no detected executed
    copy — never counted as a concession, issue #83).
    """
    by_reason: dict[str, int] = {}
    by_document: dict[str, int] = {}
    for doc in corpus_documents:
        dropped = doc.get("dropped_observations")
        if not isinstance(dropped, dict):
            continue
        doc_total = 0
        for reason, n in dropped.items():
            if isinstance(n, int) and n > 0:
                by_reason[reason] = by_reason.get(reason, 0) + n
                doc_total += n
        if doc_total:
            doc_id = str(doc.get("document_id"))
            by_document[doc_id] = by_document.get(doc_id, 0) + doc_total
    if not by_reason:
        return None
    return {
        "count": sum(by_reason.values()),
        "by_reason": dict(sorted(by_reason.items())),
        "by_document": dict(sorted(by_document.items())),
    }


def _has_mixed_extractors(version_ingest: list[Any]) -> bool:
    """``True`` when this deal's successfully-ingested versions were produced by
    more than one extractor (issue #218).

    Counts the distinct non-null ``extractor`` labels across ``status == "ok"``
    entries only: a failed version contributed no text, so it cannot skew how
    the surviving versions' clauses align. On the LLM/agent path the labels
    are ``"docling"``/``"legacy"`` (a docling->legacy fallback on some drafts
    is the mechanism behind #122's partition artifacts); on the deterministic
    path they are the file suffix, so a trail mixing DOCX drafts with a PDF
    signed copy is — truthfully — mixed there too, since each format has its
    own structure detector.
    """
    extractors = {
        vi.get("extractor")
        for vi in version_ingest
        if isinstance(vi, dict) and vi.get("status") == "ok" and vi.get("extractor")
    }
    return len(extractors) > 1


def _sanitize_corpus_documents_for_schema(
    corpus_documents: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return *corpus_documents* with each ``version_ingest`` entry stripped to
    :data:`_VERSION_INGEST_SCHEMA_KEYS` — see the constant's docstring.

    Also drops the engine-internal ``_CORPUS_DOCUMENT_INTERNAL_KEYS`` (issue
    #216). Every other field on each document dict (and every other key on
    each ``version_ingest`` entry within ``_VERSION_INGEST_SCHEMA_KEYS``)
    passes through unchanged — only ``version_ingest`` entries are rebuilt,
    and only to drop keys outside the whitelist.

    Two document-level ``x_`` extensions are added, computed from the
    UNSTRIPPED entries, so a consumer can see which trails mixed extractors
    (issue #218) without the per-version ``reason`` that every published
    schema's ``version_ingest.items`` (``additionalProperties: false``, no
    ``x_`` pattern — 0.2, 0.3 and 0.4 alike) cannot carry. Both go under the
    ``^x_`` pattern ``corpus.documents.items`` already sanctions in every
    version, so no published schema changes; a future format version is
    where they would become first-class fields:

    - ``x_mixed_extractors`` (bool, always present when ``version_ingest`` is
      a list): see :func:`_has_mixed_extractors`.
    - ``x_ingest_reason`` (list, present only when at least one version has a
      non-null reason): ``version_ingest[i].reason`` for every *i*, in the
      same order — index-aligned rather than keyed by version label, so it
      carries nothing ``publisher.py``'s version-label scrub would have to
      rewrite. Values are closed enums: the ``ExtractorLabel`` reasons
      (``"env-missing"``, ``"backend-error"``, ``"declared"``) for a mined
      version, the ``ExtractionError`` reasons (``"timeout"``, ``"no-text"``)
      for a failed one, or ``null``.
    """
    sanitized: list[dict[str, Any]] = []
    for doc in corpus_documents:
        if _CORPUS_DOCUMENT_INTERNAL_KEYS & doc.keys():
            doc = {k: v for k, v in doc.items() if k not in _CORPUS_DOCUMENT_INTERNAL_KEYS}
        version_ingest = doc.get("version_ingest")
        if not isinstance(version_ingest, list):
            sanitized.append(doc)
            continue
        new_doc = dict(doc)
        new_doc["x_mixed_extractors"] = _has_mixed_extractors(version_ingest)
        reasons = [vi.get("reason") if isinstance(vi, dict) else None for vi in version_ingest]
        if any(r is not None for r in reasons):
            new_doc["x_ingest_reason"] = reasons
        new_doc["version_ingest"] = [
            (
                {k: v for k, v in vi.items() if k in _VERSION_INGEST_SCHEMA_KEYS}
                if isinstance(vi, dict)
                else vi
            )
            for vi in version_ingest
        ]
        sanitized.append(new_doc)
    return sanitized


# Observation bases meaning no real judge assessed the clause — mirrors
# ``clause_position_compiler._UNJUDGED_BASES``. "stub" (no judge configured
# at all) is the strict case; "needs_review"/"judge_error" additionally
# cover the zero-LLM deviation stub (``_NullDeviationJudge``, pipeline.py)
# that an opt-in ``--with-deviation-judge`` run wires until verdicts land,
# which emits basis="needs_review" rather than "stub" for every changed
# clause since a judge protocol IS wired (just not a real one). Watermarking
# on all three is what makes such a run (no LLM configured anywhere)
# actually watermark its output — see issue #101. The default consumer path
# (issue #220) wires no deviation judge at all: its observations carry
# basis="deterministic", the standard check, which is not an unjudged
# placeholder.
_UNJUDGED_OBSERVATION_BASES = frozenset({"stub", "needs_review", "judge_error"})


def _compiler_version() -> str:
    try:
        return importlib.metadata.version("playbook-engine")
    except importlib.metadata.PackageNotFoundError:
        return "0.1.0"


# ---------------------------------------------------------------------------
# Error type
# ---------------------------------------------------------------------------


@dataclass
class AssemblyError(Exception):
    """Raised when the assembled playbook fails schema or normative validation."""

    blocking_errors: list[str]

    def __str__(self) -> str:
        lines = ["Playbook assembly failed validation:"]
        lines.extend(f"  {e}" for e in self.blocking_errors)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def assemble_playbook(
    agreement_type: dict[str, Any],
    baseline: dict[str, Any],
    taxonomy: dict[str, Any],
    clause_positions: list[ClausePosition],
    clause_library: list[ClauseConcept],
    corpus_documents: list[dict[str, Any]],
    generated_at: str,
    run_id: str | None = None,
    observations: list[Observation] | None = None,
    scope_bases: list[str] | None = None,
    unclassified_coverage: UnclassifiedCoverage | None = None,
    perspective: dict[str, str] | None = None,
    de_minimis: list[str] | None = None,
    playbook_id: str | None = None,
    playbook_version: str | None = None,
    supersedes: str | None = None,
    min_evidence_n: int = MIN_EVIDENCE_N,
    existing_curation: dict[str, Any] | None = None,
    existing_posture: dict[str, Any] | None = None,
    existing_floor: dict[str, Any] | None = None,
    round_moves: list[RoundMove] | None = None,
    opf_version: str = _OPF_VERSION,
) -> dict[str, Any]:
    """Assemble and validate a complete OPF playbook document (0.4 by default).

    Args:
        agreement_type:    Top-level ``{id, name}`` (``description``/``aliases``
                          optional).
        baseline:          ``{has_canonical_template: bool, template_ref?, notes?}``.
        taxonomy:          ``{source: str, entries: [...]}``.
        clause_positions:  Output of ``compile_clause_positions()``.
        clause_library:    Output of ``compile_clause_library()``.
        corpus_documents:  One dict per corpus document, each with at least
                          ``{document_id, provenance, in_scope}``.
                          Out-of-scope docs MUST have ``scope_rationale``.
        generated_at:      ISO-8601 datetime string (supplied by caller).
        run_id:            Optional run identifier for audit purposes.
        observations:      The full L4 observation list this playbook was
                          compiled from (same list passed to
                          ``compile_clause_positions()``). Used only to
                          watermark the ``compiler`` block with
                          ``stub_basis_present`` when any observation's basis
                          is in ``_UNJUDGED_OBSERVATION_BASES`` (``"stub"``,
                          ``"needs_review"``, or ``"judge_error"``) — i.e. no
                          real judge assessed that clause, so this playbook
                          must not be trusted as fully LLM-assessed. ``None``
                          (the default) contributes no watermark signal.
        scope_bases:       The ``ScopeDecision.basis`` value for every
                          document considered at L1b (in-scope or not),
                          e.g. read from ``scope.json``. Also feeds the
                          ``stub_basis_present`` watermark: any entry equal
                          to ``"stub"`` means the scope gate itself ran on
                          the no-LLM default (``_AllInScopeJudge``) for at
                          least one document. ``None`` (the default)
                          contributes no watermark signal.
        unclassified_coverage: Coverage summary (issue #113) for
                          ``taxonomy_id=None`` observations that were
                          excluded from ``clauses``/``clause_library`` —
                          typically ``compile_clause_positions()``'s third
                          return value. Recorded in
                          ``corpus.stats.unclassified`` so a consumer can see
                          omitted-content coverage without cross-referencing
                          the AAR. ``None`` (the default) omits the key
                          entirely.
        perspective:      Optional ``{party, counterparty_type}`` (OPF §3 —
                          "whose perspective this playbook is reviewed
                          'as'"). Not yet derivable from the corpus alone (no
                          config surface supplies it as of issue #140); a
                          future slice wires it from producer config. Passed
                          straight through into the document when supplied by
                          the caller; ``None`` (the default) omits the key
                          entirely, since neither field may be fabricated.
        de_minimis:       Optional list of change categories accepted even if
                          technically novel (negotiation knowledge OPF owns —
                          see OPF-SPEC.md). Passed straight through
                          when supplied; ``None`` (the default) omits the key
                          entirely.
        playbook_id:      Optional producer-assigned playbook identifier
                          (issue #143). Like ``run_id``, this is lineage
                          metadata the engine cannot derive from the corpus —
                          it is recorded in ``identity.id`` when supplied and
                          omitted otherwise, never fabricated.
        playbook_version: Optional producer-assigned version label,
                          recorded in ``identity.version`` when supplied.
        supersedes:       Optional identifier of the playbook this one
                          supersedes, recorded in ``identity.supersedes``
                          when supplied.
        min_evidence_n:   Producer-configurable evidence-depth floor (issue
                          #144, config.provenance.min_evidence_n) — must match
                          whatever value was passed to
                          ``compile_clause_positions()`` for this same run, so
                          the self-validation below (``validate_document()``)
                          enforces the identical threshold the compiler
                          already used to derive ``historical_stance``.
                          Defaults to ``MIN_EVIDENCE_N`` (2).
        existing_curation: The prior compile's ``playbook["curation"]`` dict
                          (issue #147), read by the caller from the previous
                          ``playbook.opf.json`` before it's overwritten. Every
                          pin is preserved across this recompile; its
                          ``conflict`` flag is set/cleared by comparing the
                          freshly recomputed ``historical_stance`` against
                          the pin's ``baseline_stance`` (see
                          ``playbook_engine/curation.py``). ``None`` (the
                          default) means no prior pins to carry forward — a
                          first compile, or a store with no curation history.
        existing_posture: The prior compile's ``playbook["posture"]`` dict
                          (issue #123), read by the caller from the previous
                          ``playbook.opf.json`` before it's overwritten.
                          Carried forward VERBATIM — the engine never
                          authors or edits Posture itself (that's
                          ``playbook posture interview``'s job); a recompile
                          only refreshes Evidence. ``None`` (the default, or
                          an explicit ``{}``) means no prior Posture to carry
                          forward — a first compile, or one that hasn't been
                          authored yet.
        existing_floor:   The prior compile's ``playbook["floor"]`` dict
                          (issue #123), read the same way and carried forward
                          VERBATIM for the same reason — Floor invariants are
                          authored by ``playbook floor sign``/``floor
                          propose``, never fabricated by a recompile. ``None``
                          (the default, or an explicit ``{}``) means no prior
                          Floor to carry forward.
        round_moves:      Per-round clause moves (``round_moves.jsonl``,
                          issue #177). OPF 0.4 derives each precedent's
                          ``rounds``/``moved`` from them; OPF 0.3 carries
                          them already grouped as each ClausePosition's
                          ``negotiation_trail``. ``None`` = no moves recorded.
        opf_version:      ``"0.4"`` (the default — the verdict-free per-deal
                          precedent record, issue #223) or ``"0.3"`` (the
                          0.2 evidence shape plus digest v2, kept for one
                          release).

    Returns:
        A validated playbook dict conforming to OPF v0.2 (evidence-wrapped
        clauses, descriptive ``summary.historical_stance``, empty-but-present
        ``posture``/``floor``, and an ``identity`` block carrying
        ``content_hash``/``section_digests`` — see issue #143).

    Raises:
        AssemblyError: if ``validate_document()`` reports any blocking errors.
    """
    if opf_version not in EMITTABLE_OPF_VERSIONS:
        raise ValueError(
            f"opf_version {opf_version!r} cannot be emitted "
            f"(supported: {', '.join(EMITTABLE_OPF_VERSIONS)})"
        )

    # Strip each version_ingest entry down to the schema-allowed key set
    # (issue #81) BEFORE anything below reads/embeds corpus_documents — the
    # schema's additionalProperties:false on version_ingest.items would
    # otherwise reject a document carrying e.g. "reason"
    # (extraction.ExtractorLabel.reason), which corpus_manifest.json/
    # review.json/the CLI need but the published OPF does not. Reassigning
    # the local name means every use below (stats, the embedded
    # corpus.documents, corpus.snapshot's manifest_triples) sees the
    # sanitized shape automatically.
    dropped_stats = _dropped_observation_stats(corpus_documents)
    corpus_documents = _sanitize_corpus_documents_for_schema(corpus_documents)

    # --- corpus stats (auto-computed) ---
    n_total = len(corpus_documents)
    n_in_scope = sum(1 for d in corpus_documents if d.get("in_scope", True))
    n_versions = sum(d.get("versions", 0) for d in corpus_documents)
    stats: dict[str, Any] = {
        "documents_total": n_total,
        "documents_in_scope": n_in_scope,
        "versions_total": n_versions,
    }
    if unclassified_coverage is not None:
        # Issue #113: surface unclassified (taxonomy_id=None) observation
        # coverage in the playbook itself, not just the AAR.
        stats["unclassified"] = unclassified_coverage.to_dict()
    if dropped_stats is not None:
        # Issue #216: net-diff rows that produced no observation — text with
        # no signed slot that survives in the signed version, text removed
        # before signing whose origin cannot be determined, or our standard
        # removed from a deal with no detected signed copy
        # (removed_standard_no_signed_copy) — are counted, never dropped
        # silently.
        stats["dropped_observations"] = dropped_stats

    # --- compiler metadata ---
    # Watermark (issue #101): True when at least one observation feeding this
    # playbook was never assessed by a real judge (basis in
    # _UNJUDGED_OBSERVATION_BASES — covers both "no judge configured at all"
    # and "a judge protocol IS wired but it's the zero-LLM stub default"), OR
    # when the L1b scope gate itself ran on the no-LLM stub default for at
    # least one document (a "stub" entry in scope_bases). Either signal means
    # a consuming review application should refuse to run redlines against
    # this playbook without human review.
    stub_basis_present = any(
        obs.basis in _UNJUDGED_OBSERVATION_BASES for obs in (observations or [])
    ) or any(b == "stub" for b in (scope_bases or []))
    compiler: dict[str, Any] = {
        "name": _COMPILER_NAME,
        "version": _compiler_version(),
        "generated_at": generated_at,
        "stub_basis_present": stub_basis_present,
    }
    if run_id is not None:
        compiler["run_id"] = run_id

    # --- assemble ---
    # Field order mirrors spec/playbook.schema-0.2.json's property order.
    playbook: dict[str, Any] = {
        "opf_version": opf_version,
        "agreement_type": agreement_type,
        "baseline": baseline,
        "taxonomy": taxonomy,
    }
    if perspective is not None:
        playbook["perspective"] = perspective
    if de_minimis is not None:
        playbook["de_minimis"] = de_minimis
    if opf_version == "0.4":
        # Issue #223: the verdict-free per-deal precedent record. No
        # clause_library, summary, observed_positions or negotiation_trail —
        # every field is a fact read off the store, never a judged verdict.
        playbook["evidence"] = build_precedent_evidence(
            agreement_type_id=str(agreement_type.get("id")),
            clause_positions=clause_positions,
            observations=list(observations or []),
            corpus_documents=corpus_documents,
            round_moves=round_moves,
            party=perspective_party(playbook),
        )
    else:
        playbook["evidence"] = {
            "clauses": [cp.to_dict() for cp in clause_positions],
            "clause_library": [cc.to_dict() for cc in clause_library],
        }
    # Posture/Floor (§3.6/§3.7): empty-but-present by default (#140 scope
    # excludes Floor invariant content — see #145) — the engine must never
    # fabricate negotiation intent or hard lines, so both sections are always
    # structurally present (satisfying every consumer's "the section exists"
    # expectation). But "no content yet" is only true on a FIRST compile:
    # once a human has authored Posture (`playbook posture interview`) or
    # signed a Floor invariant (`playbook floor sign`), those sections live
    # ONLY inside the previously-written playbook.opf.json — nothing else in
    # the out-dir can reconstruct them. A recompile must carry them forward
    # verbatim, exactly like `existing_curation` above, or Route C's "the
    # Posture and Floor you already signed should survive" promise
    # (SKILL.md) is false (issue #123). Unlike curation, there is no
    # per-clause merge/conflict step here — Posture/Floor are not derived
    # from clause_stances, so a straight carry-forward is the whole contract.
    playbook["posture"] = existing_posture if existing_posture is not None else {}
    playbook["floor"] = existing_floor if existing_floor is not None else {}
    playbook["corpus"] = {
        "documents": corpus_documents,
        "stats": stats,
    }
    # Corpus snapshot identity (issue #185, §3.8): one hash naming the exact
    # corpus state this playbook was compiled from — the canonical JSON of
    # every (document_id, version, sha256) triple, sorted. Omitted when no
    # document carries version_files (a pre-#185 store or hand-built corpus):
    # an empty-manifest hash would name "no corpus", not this one.
    manifest_triples = sorted(
        (d["document_id"], vf["version"], vf["sha256"])
        for d in corpus_documents
        for vf in d.get("version_files", [])
    )
    if manifest_triples:
        playbook["corpus"]["snapshot"] = {
            "manifest_hash": sha256_hex(canonicalize([list(t) for t in manifest_triples]))
        }
    playbook["compiler"] = compiler

    # --- curation (issue #147) ---
    # Merge any prior compile's attorney-pinned positions over this compile's
    # freshly recomputed historical_stance, flagging/clearing conflict per
    # clause. Computed before `identity` below so section_digests.curation
    # reflects the merged (not the stale) curation content. Omitted entirely
    # when there's nothing to carry forward (no prior pins) — mirrors
    # perspective/de_minimis's "never fabricate, omit when absent" rule.
    # Read through the same accessor `playbook curate`/`view apply` use to
    # stamp a pin's baseline_stance, so a pin made on this document never
    # conflicts with it on recompile. An OPF 0.4 document carries no stance
    # (issue #223), so every 0.4 clause reads "unknown": a pin carried over
    # from a 0.3 compile is flagged once (its stance is no longer computed),
    # and a pin made on a 0.4 document never is.
    clause_stances = {c["id"]: clause_stance(c) for c in playbook["evidence"]["clauses"]}
    curation = merge_curation(existing_curation, clause_stances, checked_at=generated_at)
    if curation:
        playbook["curation"] = curation

    # Strip zero-width/bidi-control characters carried in from extraction
    # BEFORE the digest is built, so digest._dedupe_rank groups observations
    # by their post-strip text (issue #35) — otherwise an intra-word ZWSP
    # splits what should be one dedupe group into two, and the embedded
    # digest diverges from build_digest() recomputed over the shipped
    # (stripped) playbook, breaking the "digest is a pure function of the
    # evidence section" invariant that content_hash lineage relies on.
    playbook = _strip_invisible(playbook)

    # Judged verdicts from an opt-in judged run (issue #220) are vendor data
    # under x_judgments, keyed by precedent id — never part of
    # evidence.precedent, never read by the digest (issue #223). Emitted only
    # when a real judge assessed at least one precedent's terminal row.
    if opf_version == "0.4":
        restamp_evidence(
            playbook["evidence"],
            str(agreement_type.get("id")),
            party=perspective_party(playbook),
        )
        judgments = build_x_judgments(list(observations or []), playbook["evidence"]["precedent"])
        if judgments:
            playbook["x_judgments"] = _strip_invisible(judgments)

    # --- digest (OPF 0.3: digest v2; OPF 0.4: digest v3) ---
    # The compact model-facing projection of the evidence section. Computed
    # after stripping (above) and before identity so it is covered by
    # content_hash like every other content section (it is a pure function of
    # evidence — two compiles of identical evidence carry identical digests).
    playbook["digest"] = build_digest(playbook)

    # --- identity (issue #143) ---
    # content_hash/section_digests are engine-computed and always populated —
    # they are pure functions of the document's own content, unlike
    # perspective/de_minimis which require input the engine cannot derive.
    # id/version/supersedes are producer-assigned lineage metadata (like
    # run_id above): recorded only when the caller supplies them, never
    # fabricated. Computed last so canonicalize_playbook() sees the fully
    # assembled document (it excludes `identity` and the run-metadata
    # compiler keys itself — see playbook_engine/canonicalize.py).

    identity: dict[str, Any] = {}
    if playbook_id is not None:
        identity["id"] = playbook_id
    if playbook_version is not None:
        identity["version"] = playbook_version
    if supersedes is not None:
        identity["supersedes"] = supersedes
    identity["content_hash"] = content_hash(playbook)
    identity["section_digests"] = compute_section_digests(playbook)
    playbook["identity"] = identity

    # --- validate ---
    result = validate_document(playbook, min_evidence_n=min_evidence_n)
    if not result.ok:
        raise AssemblyError(blocking_errors=[str(e) for e in result.errors if e.blocking])

    return playbook


def write_playbook(playbook: dict[str, Any], path: Path) -> None:
    """Write *playbook* to *path* as pretty-printed JSON, atomically.

    The parent directory is created if it does not exist.  A temp file is
    written first and then renamed via ``os.replace()`` to prevent partial
    writes.

    Args:
        playbook: A validated playbook dict (from ``assemble_playbook()``).
        path:     Destination path (conventionally ``<dir>/playbook.opf.json``).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(playbook, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)
