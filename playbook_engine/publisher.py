"""``playbook publish`` — party-anonymous publication profile (issue #188).

Goal (owner decision 2026-07-12): "publish a playbook without naming any
party" is a first-class, verifiable FORMAT guarantee, not a best-effort
nicety layered on top of :mod:`playbook_engine.export_profile` (issue #146).
Born-safe pseudonymization (:mod:`playbook_engine.entity_registry`, issue
#153) aliases known entities at ingest, and ``export_profile`` runs the
semantic-residue judgment + independent verify pass — but neither strips
the publishing party's real name from ``perspective.party``, coarsens
identifying dates, or removes DMS-structure-leaking source paths. This
module is the six-step deterministic-transform-then-judge pipeline that
does all of that, in order:

  1. ``perspective.party`` -> a role label (default ``"the company"``), and
     every GENERIC free-text mention of "the counterparty" (default label
     ``"the counterparty"``) is normalized to the configured label. Per-deal
     entity-registry aliases (e.g. ``Counterparty-7``) are NOT touched by
     this step — collapsing them would destroy the cross-deal signal of
     which counterparty conceded what (see :func:`_normalize_generic_party_mentions`
     for how a numbered alias is distinguished from a generic mention).
     Records ``x_publication: {profile: "public", published_at: ...}``.
  2. Strips ``corpus.documents[].version_files[].source_uri`` and
     ``baseline.template_ref.source`` — paths/URIs leak DMS structure;
     ``sha256`` hashes are kept (they leak nothing and preserve
     verifiability, per OPF-SPEC.md §4).
  3. Coarsens ``observed_positions[].observed_at`` to ``YYYY-Qn``
     (``keep_dates=True`` opts out) — an exact date can identify a
     counterparty. ``negotiation_trail[]`` carries no date field of its own
     (only a ``round`` ordinal) as of OPF-SPEC.md §3.5.3, so there is
     nothing else to coarsen there.
  4. Deterministic no-known-entity backstop: every string in the doc
     (recursively) is scanned, case/punctuation-normalized, against
     *known_entity_names* (the entity registry's real names — the
     ingest-time holdout map, by construction the same data). ANY hit
     raises :class:`PublishError` — loud, unconditional, no flag suppresses
     it. This is the hard backstop under the best-effort judgment pass
     below.
  5. Runs :func:`playbook_engine.export_profile.export_profile` (now
     full-surface, issue #188) over the doc from step 4. Per the #146/#152
     standard a "leaked" verify finding does not hard-gate *export_profile*
     itself — but *publish* holds itself to a stronger default: a non-empty
     ``leaked`` raises :class:`PublishError` unless the caller passes
     ``accept_residue_risk=True``. The report always lists every finding
     either way.
  6. Recomputes ``identity`` on the transformed document — it is a
     different artifact. ``identity.supersedes`` is set to the PRIVATE
     document's own ``content_hash`` (before any of the above ran).

Every step no-ops cleanly when its input field is absent (e.g. the
negotiation-dynamics fields on a pre-#177 store, or a doc with no
``curation``/``floor``/``posture`` content yet) — there is no ordering
dependency on which OPF sections a given input document happens to carry.

Scope note: this module is verified offline with fake judges (issue #188's
required verification) — same "mechanism only" scope boundary as
``export_profile``, ``floor_judge``, and ``scope_gate``. Publishing a
real derived playbook is a separate, needs-human issue (#189).
"""

from __future__ import annotations

import copy
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from playbook_engine.canonicalize import compute_section_digests, content_hash
from playbook_engine.export_profile import (
    RedactionFinding,
    RedactionJudge,
    VerifyFinding,
    VerifyJudge,
    _apply_rewrites,
    _extract_text_samples,
    export_profile,
)
from playbook_engine.opf_accessors import playbook_clauses

DEFAULT_PARTY_LABEL = "the company"
DEFAULT_COUNTERPARTY_LABEL = "the counterparty"

_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# Matches a GENERIC "(the) counterparty" mention — bare or with a leading
# article — but NOT a per-deal numbered alias (``Counterparty-7``): the
# negative lookahead excludes exactly the "-<digit>" suffix the entity
# registry's alias format uses (``entity_registry._ALIAS_PREFIX = "Counterparty"``,
# aliases are ``f"{prefix}-{n}"``). Case-insensitive so it also catches a
# sentence-initial "Counterparty pushed back on...".
_GENERIC_COUNTERPARTY_RE = re.compile(r"\b(?:the\s+)?counterparty\b(?!-\d)", re.IGNORECASE)

# Normalizes a string for the deterministic no-known-entity backstop scan:
# casefold + punctuation stripped to whitespace + whitespace collapsed, so
# "Acme, Inc." and "acme inc" compare equal. Padding the normalized text with
# leading/trailing spaces before substring-checking a normalized name (see
# ``_entity_backstop_scan``) approximates a whole-word/phrase match without a
# per-name compiled regex.
_PUNCT_RE = re.compile(r"[^\w\s]")
_WS_RE = re.compile(r"\s+")


def _normalize_for_scan(s: str) -> str:
    return _WS_RE.sub(" ", _PUNCT_RE.sub(" ", s.casefold())).strip()


class PublishError(Exception):
    """Raised when :func:`publish_playbook` cannot produce a safe public artifact.

    Two distinct triggers, both loud-fail (never silent), never confused with
    each other in the message:

      - the deterministic no-known-entity backstop (step 4) found a real
        entity name surviving in the transformed output — a hard invariant;
        no flag suppresses this;
      - the independent verify pass (step 5) flagged residual semantic
        residue and the caller did not pass ``accept_residue_risk=True`` —
        best-effort/soft per the #146/#152 standard, but *publish* defaults
        to blocking on it (a stronger default than ``export_profile``
        itself), suppressible by the one flag meant for exactly this.
    """


@dataclass(frozen=True)
class PublishReport:
    """Result of :func:`publish_playbook`.

    Attributes:
        doc:                The published (public-profile) document.
        redaction_findings: Every :class:`~playbook_engine.export_profile.RedactionFinding`
                            from step 5's redaction pass.
        verify_findings:    Every :class:`~playbook_engine.export_profile.VerifyFinding`
                            from step 5's independent verify pass.
        leaked:             The subset of ``verify_findings`` with
                            ``leaked=True``. Non-empty only when the caller
                            passed ``accept_residue_risk=True`` — otherwise
                            :func:`publish_playbook` raises :class:`PublishError`
                            before returning.
        proper_noun_findings: Every proper-noun-like string remaining in the
                            published free text (issue #211). Advisory — the
                            reviewer's checkable "confirm none is a
                            counterparty" list; never gates the publish.
    """

    doc: dict[str, Any]
    redaction_findings: tuple[RedactionFinding, ...]
    verify_findings: tuple[VerifyFinding, ...]
    leaked: tuple[VerifyFinding, ...]
    proper_noun_findings: tuple[ProperNounFinding, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "redaction_findings": [f.to_dict() for f in self.redaction_findings],
            "verify_findings": [f.to_dict() for f in self.verify_findings],
            "leaked": [f.to_dict() for f in self.leaked],
            "proper_noun_findings": [f.to_dict() for f in self.proper_noun_findings],
        }


# ---------------------------------------------------------------------------
# Step 1: party/counterparty role-label transform
# ---------------------------------------------------------------------------


def _normalize_generic_party_mentions(
    doc: dict[str, Any], counterparty_label: str
) -> dict[str, Any]:
    """Return a deep copy of *doc* with generic "(the) counterparty" mentions
    in every free-text sample normalized to *counterparty_label*.

    Reuses ``export_profile``'s own free-text extraction/apply machinery so
    this touches EXACTLY the surfaces the residue judge will later see — no
    separate field list to keep in sync. Per-deal numbered aliases
    (``Counterparty-7``) never match ``_GENERIC_COUNTERPARTY_RE`` and so
    survive unchanged (issue #188 design: "only free-text generic references
    normalize to the role label").
    """
    samples, locations = _extract_text_samples(doc)
    findings = []
    for sample in samples:
        rewritten = _GENERIC_COUNTERPARTY_RE.sub(counterparty_label, sample.text)
        if rewritten != sample.text:
            findings.append(
                RedactionFinding(
                    path=sample.path,
                    has_residue=True,
                    rationale="Deterministic generic-counterparty role-label normalization (issue #188).",
                    rewritten_text=rewritten,
                    basis="stub",
                )
            )
    return _apply_rewrites(doc, findings, locations)


# ---------------------------------------------------------------------------
# Step 2: strip DMS-structure-leaking source paths/URIs
# ---------------------------------------------------------------------------


def _strip_source_paths(doc: dict[str, Any]) -> None:
    """Mutate *doc* in place, deleting source paths/URIs (keeping hashes)."""
    corpus = doc.get("corpus")
    if isinstance(corpus, dict):
        for document in corpus.get("documents", []):
            for version_file in document.get("version_files", []):
                version_file.pop("source_uri", None)

    baseline = doc.get("baseline")
    if isinstance(baseline, dict):
        template_ref = baseline.get("template_ref")
        if isinstance(template_ref, dict):
            template_ref.pop("source", None)


# ---------------------------------------------------------------------------
# Step 3: coarsen identifying dates
# ---------------------------------------------------------------------------


def _coarsen_to_quarter(date_str: str) -> str:
    year, month, _day = date_str.split("-")
    quarter = (int(month) - 1) // 3 + 1
    return f"{year}-Q{quarter}"


def _coarsen_dates(doc: dict[str, Any]) -> None:
    """Mutate *doc* in place, coarsening every ``observed_at`` to ``YYYY-Qn``.

    ``negotiation_trail[]`` entries carry no independent date field of their
    own (only a ``round`` ordinal, per OPF-SPEC.md §3.5.3) — the only actual
    date signal on a clause is ``observed_positions[].observed_at``, which
    this coarsens. No-ops per-observation when ``observed_at`` is absent or
    not a well-formed ISO date (never fabricates, never raises on odd data).
    """
    for clause in playbook_clauses(doc):
        for obs in clause.get("observed_positions", []):
            observed_at = obs.get("observed_at")
            if isinstance(observed_at, str) and _ISO_DATE_RE.match(observed_at):
                obs["observed_at"] = _coarsen_to_quarter(observed_at)


# ---------------------------------------------------------------------------
# Step 4: deterministic no-known-entity backstop
# ---------------------------------------------------------------------------


def _walk_strings(node: Any, path: str = "$") -> list[tuple[str, str]]:
    """Return ``(path, value)`` for every string leaf reachable from *node*."""
    found: list[tuple[str, str]] = []
    if isinstance(node, str):
        found.append((path, node))
    elif isinstance(node, dict):
        for key, value in node.items():
            found.extend(_walk_strings(value, f"{path}.{key}"))
    elif isinstance(node, list):
        for idx, value in enumerate(node):
            found.extend(_walk_strings(value, f"{path}[{idx}]"))
    return found


def _entity_backstop_scan(
    doc: dict[str, Any], known_entity_names: Sequence[str]
) -> list[tuple[str, str]]:
    """Return every ``(path, matched_real_name)`` hit of a known entity name
    surviving anywhere in *doc*, case/punctuation-normalized.

    Scans EVERY string in the document — not just the free-text samples
    ``export_profile`` judges — since a real name could leak through a
    structural field (a title, a document_id) the judgment pass never sees.
    """
    normalized_names = [
        (name, _normalize_for_scan(name)) for name in known_entity_names if name and name.strip()
    ]
    hits: list[tuple[str, str]] = []
    for path, text in _walk_strings(doc):
        if not text:
            continue
        haystack = f" {_normalize_for_scan(text)} "
        for real_name, normalized_name in normalized_names:
            if normalized_name and f" {normalized_name} " in haystack:
                hits.append((path, real_name))
    return hits


# ---------------------------------------------------------------------------
# List-independent proper-noun residue sweep (issue #211)
# ---------------------------------------------------------------------------
#
# The step-4 backstop above can only catch names it was GIVEN
# (``known_entity_names``); an incomplete list lets an unknown real name
# through. This sweep needs NO name list and NO LLM: it flags every
# capitalized-token run remaining in the published free text, so a reviewer
# gets an exhaustive, checkable list of everything name-shaped that survived
# ("here is every proper noun left — confirm none is a counterparty") rather
# than the unfalsifiable "we removed the names we knew about". Advisory, not
# gating — proper nouns are frequently benign (governing-law states), so this
# feeds human review (and the agent-classification skill step) instead of
# hard-failing.

_MONTHS = {
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
}
_WEEKDAYS = {"monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"}
# Common capitalized NON-name tokens: legal-doc structure, boilerplate, and
# frequent sentence-initial words. Kept deliberately small — a benign term
# that slips through is one the reviewer confirms and moves on; a real name
# dropped is a leak. So the sweep errs toward INCLUSION.
_STRUCTURAL_STOPWORDS = {
    # legal-document structure
    "agreement",
    "section",
    "article",
    "exhibit",
    "schedule",
    "appendix",
    "addendum",
    "amendment",
    "attachment",
    "clause",
    "party",
    "parties",
    "provider",
    "institution",
    "company",
    "counterparty",
    "effective",
    "date",
    "term",
    "whereas",
    "therefore",
    "hereto",
    "herein",
    "hereunder",
    "student",
    "students",
    "university",
    "college",
    "school",
    "program",
    "programs",
    "services",
    "service",
    "confidential",
    "governing",
    "law",
    "state",
    "states",
    "united",
    "page",
    "no",
    "playbook",
    "evidence",
    "posture",
    "floor",
    "provenance",
    # frequent contract nouns (never the distinctive token of a party name —
    # institution/company names carry a proper token like "Alpha"/"Regents")
    "contract",
    "coverage",
    "liability",
    "insurance",
    "indemnification",
    "indemnify",
    "obligation",
    "obligations",
    "termination",
    "confidentiality",
    "mutual",
    "commercial",
    "general",
    "professional",
    "limited",
    "standard",
    "initial",
    "final",
    "draft",
    "redline",
    "signed",
    "executed",
    "notice",
    "consent",
    "approval",
    "review",
    "period",
    "annual",
    "monthly",
    "aggregate",
    "occurrence",
    "claim",
    "claims",
    "damages",
    "fees",
    "costs",
    "payment",
    # common discourse / sentence-initial words
    "the",
    "this",
    "that",
    "these",
    "those",
    "it",
    "we",
    "you",
    "they",
    "them",
    "their",
    "our",
    "its",
    "his",
    "her",
    "if",
    "when",
    "where",
    "each",
    "any",
    "all",
    "such",
    "upon",
    "during",
    "notwithstanding",
    "provided",
    "subject",
    "and",
    "or",
    "but",
    "not",
    "never",
    "always",
    "also",
    "both",
    "either",
    "neither",
    "however",
    "moreover",
    "further",
    "additionally",
    "please",
    "note",
    "see",
    "yes",
    "accept",
    "accepted",
    "asked",
    "per",
}
_BASE_STOPWORDS = _MONTHS | _WEEKDAYS | _STRUCTURAL_STOPWORDS

# Lowercase connectors kept INSIDE a multi-word proper noun ("State of New
# York", "Board of Regents") so an org/place name is not split apart. A span
# made up ENTIRELY of connectors + stopwords is dropped.
_CONNECTORS = {"of", "the", "and", "for", "&", "de", "von", "van"}

# A capitalized word (allowing internal apostrophe/hyphen), then any run of
# further capitalized words optionally joined by a single lowercase connector.
# A period is deliberately NOT part of a word, so a match cannot bridge a
# sentence boundary ("...New York. Alpha University" is two names, not one).
_CAP_WORD = r"[A-Z][A-Za-z'’-]*"
_PROPER_NOUN_RE = re.compile(
    rf"{_CAP_WORD}(?:\s+(?:(?:of|the|and|for|&|de|von|van)\s+)?{_CAP_WORD})*"
)

_DEFAULT_MAX_SAMPLE_PATHS = 3


@dataclass(frozen=True)
class ProperNounFinding:
    """One capitalized-token run surviving in the published free text.

    Attributes:
        text:         The proper-noun-like string (e.g. ``"Alpha University"``).
        count:        Total occurrences across all scanned surfaces.
        sample_paths: Up to ``max_sample_paths`` OPF paths where it appears,
                      so a reviewer can locate it.
    """

    text: str
    count: int
    sample_paths: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "count": self.count, "sample_paths": list(self.sample_paths)}


def proper_noun_residue(
    doc: dict[str, Any],
    *,
    party_label: str = DEFAULT_PARTY_LABEL,
    counterparty_label: str = DEFAULT_COUNTERPARTY_LABEL,
    max_sample_paths: int = _DEFAULT_MAX_SAMPLE_PATHS,
) -> tuple[ProperNounFinding, ...]:
    """Return every proper-noun-like string remaining in *doc*'s free text.

    List-independent and LLM-free (issue #211). Scans the same free-text
    surfaces the residue judge sees (:func:`_extract_text_samples`), extracts
    capitalized-token runs, and drops any run made up entirely of connectors
    and stopwords (including the configured party/counterparty role labels).
    Deduplicated case-insensitively, ordered most-frequent first. Advisory
    output — the caller (a reviewer / the classification skill step) decides
    what is a real name vs. a benign place/generic term.
    """
    stopwords = set(_BASE_STOPWORDS)
    for label in (party_label, counterparty_label):
        stopwords.update(_normalize_for_scan(label).split())

    samples, _ = _extract_text_samples(doc)
    agg: dict[str, dict[str, Any]] = {}
    for sample in samples:
        for match in _PROPER_NOUN_RE.finditer(sample.text):
            span = match.group(0).strip(" .,'’-&")
            # A lone capital letter is never a meaningful name — it is almost
            # always a fragment of an abbreviation ("U.S.") or a sentence-start
            # "A"/"I". Drop single-character spans.
            if len(span) < 2:
                continue
            tokens = [t for t in _WS_RE.split(span) if t]
            meaningful = [t for t in tokens if t.casefold() not in _CONNECTORS]
            if not meaningful or all(t.casefold() in stopwords for t in meaningful):
                continue
            key = span.casefold()
            entry = agg.setdefault(key, {"text": span, "count": 0, "paths": []})
            entry["count"] += 1
            if sample.path not in entry["paths"] and len(entry["paths"]) < max_sample_paths:
                entry["paths"].append(sample.path)

    return tuple(
        ProperNounFinding(text=e["text"], count=e["count"], sample_paths=tuple(e["paths"]))
        for e in sorted(agg.values(), key=lambda e: (-e["count"], e["text"].casefold()))
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def publish_playbook(
    doc: dict[str, Any],
    redaction_judge: RedactionJudge,
    verify_judge: VerifyJudge,
    known_entity_names: Sequence[str],
    *,
    published_at: str,
    party_label: str = DEFAULT_PARTY_LABEL,
    counterparty_label: str = DEFAULT_COUNTERPARTY_LABEL,
    keep_dates: bool = False,
    accept_residue_risk: bool = False,
) -> PublishReport:
    """Run the six-step party-anonymous publication transform (issue #188).

    Args:
        doc:                 A compiled OPF playbook (already born-safe per
                              #153 — known entity names are aliases, not raw
                              names, before this ever runs).
        redaction_judge:      Step 5's semantic-residue redaction pass.
        verify_judge:         Step 5's independent verify pass.
        known_entity_names:   Real entity names for step 4's deterministic
                              backstop — the entity registry's canonical
                              names (``EntityRegistry.alias_map().values()``),
                              which is by construction the same data as the
                              ingest-time holdout map. An empty sequence
                              makes step 4 a no-op (nothing to check against).
        published_at:         ISO-8601 datetime, supplied by the caller (like
                              ``playbook_assembler.assemble_playbook``'s
                              ``generated_at``) so this function stays
                              deterministic and testable without time mocking.
        party_label:          Replaces ``perspective.party``. Default
                              ``"the company"``.
        counterparty_label:   Replaces every GENERIC free-text "(the)
                              counterparty" mention. Default
                              ``"the counterparty"``. Per-deal numbered
                              aliases are never touched — see module
                              docstring step 1.
        keep_dates:           Skip step 3's date coarsening when ``True``.
        accept_residue_risk:  When ``True``, a non-empty ``leaked`` from
                              step 5 is returned on the report instead of
                              raising :class:`PublishError`. Never affects
                              step 4's hard backstop.

    Returns:
        :class:`PublishReport`.

    Raises:
        PublishError: step 4 found a known real entity name surviving in the
                      output, OR step 5's verify pass flagged residual
                      semantic residue and ``accept_residue_risk`` is
                      ``False``.
    """
    published = copy.deepcopy(doc)

    # --- step 1: party/counterparty role-label transform ---
    perspective = published.get("perspective")
    if isinstance(perspective, dict) and "party" in perspective:
        perspective["party"] = party_label
    published = _normalize_generic_party_mentions(published, counterparty_label)
    published["x_publication"] = {"profile": "public", "published_at": published_at}

    # --- step 2: strip DMS-structure-leaking source paths/URIs ---
    _strip_source_paths(published)

    # --- step 3: coarsen identifying dates ---
    if not keep_dates:
        _coarsen_dates(published)

    # --- step 4: deterministic no-known-entity backstop (hard, unconditional) ---
    hits = _entity_backstop_scan(published, known_entity_names)
    if hits:
        listing = "; ".join(f"{path} matches {name!r}" for path, name in hits)
        raise PublishError(
            f"publish blocked: {len(hits)} known real-entity-name hit(s) survived the "
            f"deterministic transform: {listing}. This is a hard backstop — no flag "
            "suppresses it (issue #188)."
        )

    # --- step 5: full-surface residue judgment + independent verify pass ---
    export_report = export_profile(
        published, redaction_judge=redaction_judge, verify_judge=verify_judge
    )
    published = export_report.doc
    if export_report.leaked and not accept_residue_risk:
        leaked_paths = ", ".join(f.path for f in export_report.leaked)
        raise PublishError(
            f"publish blocked: independent verify pass flagged "
            f"{len(export_report.leaked)} sample(s) as still leaking semantic "
            f"residue: {leaked_paths}. Pass accept_residue_risk=True "
            "(--accept-residue-risk) to publish anyway."
        )

    # --- step 6: recompute identity — a published doc is a different artifact ---
    private_identity = doc.get("identity")
    private_content_hash = (
        private_identity.get("content_hash") if isinstance(private_identity, dict) else None
    )
    new_identity = {
        k: v
        for k, v in (published.get("identity") or {}).items()
        if k not in ("content_hash", "section_digests")
    }
    if private_content_hash is not None:
        new_identity["supersedes"] = private_content_hash
    new_identity["content_hash"] = content_hash(published)
    new_identity["section_digests"] = compute_section_digests(published)
    published["identity"] = new_identity

    # --- list-independent proper-noun residue sweep (issue #211, advisory) ---
    # Computed on the FINAL published doc so it reflects every transform above.
    proper_nouns = proper_noun_residue(
        published, party_label=party_label, counterparty_label=counterparty_label
    )

    return PublishReport(
        doc=published,
        redaction_findings=export_report.redaction_findings,
        verify_findings=export_report.verify_findings,
        leaked=export_report.leaked,
        proper_noun_findings=proper_nouns,
    )
