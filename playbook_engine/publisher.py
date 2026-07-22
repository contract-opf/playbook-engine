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
  5.5. Final institution-identity gate (:func:`_institution_identity_hits`):
     re-scans the WHOLE surface — every string value AND dict key — for
     high-confidence institution-name shapes ("University of X", "X University",
     "X College", "Regents of ...", "Board of Trustees/Regents") and any postal
     address that survived the scrub. ANY hit raises :class:`PublishError`,
     unconditionally (like step 4). This is the layer the step-4 backstop
     (list-dependent) and the #211 proper-noun sweep (advisory, samples-only)
     both miss: an unregistered counterparty name hiding in a signature block,
     a ``corpus.stats`` dict key, or a filename-derived ``document_id`` slug —
     the exact class of leak that shipped in a public example-playbook publish
     (2026-07-22). Governing-law states and generic descriptors do not match.
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
            # version_ingest[].version carries raw source filename stems —
            # DMS structure (and, when a counterparty acronym is missing from
            # the entity registry, an identity leak — issue #234/#189). The
            # public artifact needs only the ordinal.
            for i, ingest in enumerate(document.get("version_ingest", [])):
                if isinstance(ingest, dict) and "version" in ingest:
                    ingest["version"] = f"v{i + 1}"

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
# Step 3.5: publication-noise scrub + GC redact list (issue #234)
# ---------------------------------------------------------------------------

#: E-signature/CLM audit-trail noise (DocuSign/Adobe Sign): signature pages
#: extract into full_text as event lines, envelope/transaction ids, and IP
#: labels. None of it is contract content, and audit trails carry signatory
#: names and network metadata — strip the LINE, conservatively (a line must
#: match one of these unambiguous markers to be dropped).
_ESIGN_LINE_RE = re.compile(
    r"("
    r"docusign|adobe\s*sign|envelopeid|envelope\s+id|source\s+envelope|"
    r"transaction\s+id|ip\s+address|final\s+audit\s+report|record\s+tracking|"
    r"autonav|time\s+source|timestamp|holder:|signer\s+events|carbon\s+copy\s+events|"
    r"notary\s+events|payment\s+events|envelope\s+(sent|summary|originator)|"
    r"certified\s+delivered|signing\s+complete|envelope\s+stamping|"
    r"^\s*(signed|viewed|sent|delivered|created|completed|resent|read)\b.*"
    r"\b\d{1,2}:\d{2}(:\d{2})?\s*(am|pm)?"
    r")",
    re.IGNORECASE,
)

#: Long opaque tokens (envelope ids, base64-ish audit hashes) anywhere in a
#: line — redacted in place rather than dropping the line. Requires mixed
#: case so lowercase-hex content addresses (sha256) never match; strings
#: beginning with "sha256:" are additionally exempted wholesale in the
#: scrubber.
_OPAQUE_TOKEN_RE = re.compile(
    r"\b(?=[A-Za-z0-9_-]*[A-Z])(?=[A-Za-z0-9_-]*[a-z])[A-Za-z0-9_-]{18,}\b"
)

#: Street-address spans: "123 Rose Garden Lane", "P.O. Box 6186",
#: "Suite 400", and City, ST 12345 tails. Addresses in notice/signature
#: blocks uniquely identify a counterparty even after its name is aliased.
_ADDRESS_SPAN_RE = re.compile(
    r"("
    r"\b\d{1,6}(\s+[A-Za-z][A-Za-z.'-]*){1,5}\s+"
    r"(Road|Rd|Street|St|Avenue|Ave|Boulevard|Blvd|Lane|Ln|Drive|Dr|Parkway|Pkwy|"
    r"Highway|Hwy|Way|Circle|Court|Ct|Place|Pl|Trail|Terrace)\b\.?"
    r"|\bP\.?\s*O\.?\s+Box\s+\d+\b"
    r"|\b(Suite|Ste|Room|Rm|Bldg|Building)\s*#?\s*[A-Za-z0-9-]+\b"
    r"|\b[A-Z][A-Za-z.-]+,?\s+[A-Z]{2}\s+\d{5}(-\d{4})?\b"
    r")"
)

_ADDRESS_LABEL = "[address redacted]"
_REDACT_LABEL = "[redacted]"

#: Full US state names — for the "City, <State> <ZIP>" postal form that the
#: two-letter ``[A-Z]{2}`` rule in ``_ADDRESS_SPAN_RE`` misses (e.g. a notice
#: block written "New York, New York 10017"). Kept separate so the address
#: scrub and the final address backstop share one authority.
_US_STATE_NAMES = (
    "Alabama|Alaska|Arizona|Arkansas|California|Colorado|Connecticut|Delaware|"
    "Florida|Georgia|Hawaii|Idaho|Illinois|Indiana|Iowa|Kansas|Kentucky|"
    "Louisiana|Maine|Maryland|Massachusetts|Michigan|Minnesota|Mississippi|"
    "Missouri|Montana|Nebraska|Nevada|New\\s+Hampshire|New\\s+Jersey|New\\s+Mexico|"
    "New\\s+York|North\\s+Carolina|North\\s+Dakota|Ohio|Oklahoma|Oregon|Pennsylvania|"
    "Rhode\\s+Island|South\\s+Carolina|South\\s+Dakota|Tennessee|Texas|Utah|Vermont|"
    "Virginia|Washington|West\\s+Virginia|Wisconsin|Wyoming"
)
_STATE_ZIP_RE = re.compile(
    r"\b[A-Z][A-Za-z.'-]+(?:\s+[A-Z][A-Za-z.'-]+)*,?\s+"
    rf"(?:{_US_STATE_NAMES})\s+\d{{5}}(?:-\d{{4}})?\b"
)

#: E-mail addresses: NAME@institution-domain is a double identity leak
#: (person + counterparty domain), never contract content.
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")

#: UUIDs (e-sign transaction/document ids) — lowercase hex evades the
#: mixed-case opaque-token rule, so match the canonical shape directly.
_UUID_RE = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)

#: URLs: portal/DMS links (and institution web addresses) are structure, not
#: contract content.
_URL_RE = re.compile(r"\b(?:https?://|www\.)\S+", re.IGNORECASE)


def _transform_all_strings(node: Any, fn: Any, *, include_keys: bool = False) -> Any:
    """Recursively apply *fn* to every string in *node* (returns new tree).

    ``include_keys=True`` also rewrites dict KEYS — needed by the redact-term
    pass, because document_id slugs appear as map keys (``corpus.stats``
    tallies) and a redacted id must transform identically everywhere it
    occurs, key or value, to keep cross-references consistent. The noise
    scrub deliberately leaves keys alone (structural key names are not
    prose).
    """
    if isinstance(node, str):
        return fn(node)
    if isinstance(node, dict):
        return {
            (fn(k) if include_keys and isinstance(k, str) else k): _transform_all_strings(
                v, fn, include_keys=include_keys
            )
            for k, v in node.items()
        }
    if isinstance(node, list):
        return [_transform_all_strings(item, fn, include_keys=include_keys) for item in node]
    return node


def _scrub_publication_noise(doc: dict[str, Any]) -> dict[str, Any]:
    """Strip e-sign audit lines and redact address spans, doc-wide.

    Operates on EVERY string (the same surface ``proper_noun_residue``
    sweeps) so the scrub and the residue report cannot disagree about
    coverage. Deterministic; never raises.
    """

    def scrub(text: str) -> str:
        if text.startswith("sha256:"):
            return text  # content addresses are never publication noise
        lines_out = []
        for line in text.split("\n"):
            if _ESIGN_LINE_RE.search(line):
                continue
            line = _EMAIL_RE.sub(_REDACT_LABEL, line)
            line = _UUID_RE.sub(_REDACT_LABEL, line)
            line = _URL_RE.sub(_REDACT_LABEL, line)
            line = _OPAQUE_TOKEN_RE.sub(_REDACT_LABEL, line)
            line = _ADDRESS_SPAN_RE.sub(_ADDRESS_LABEL, line)
            line = _STATE_ZIP_RE.sub(_ADDRESS_LABEL, line)
            lines_out.append(line)
        return "\n".join(lines_out)

    identity = doc.pop("identity", None)  # recomputed in step 6; never scrub it
    scrubbed = _transform_all_strings(doc, scrub)
    if identity is not None:
        scrubbed["identity"] = identity
    return scrubbed  # type: ignore[no-any-return]


def _apply_redact_terms(doc: dict[str, Any], terms: Sequence[str]) -> dict[str, Any]:
    """Replace every GC-supplied redact term with ``[redacted]``, doc-wide.

    The redaction list is the GC's residue-review output (signatory names,
    institution name fragments, campus towns — whatever the residue report
    surfaced that the entity registry did not know). Matching is
    case-insensitive and whitespace-flexible (extraction doubles spaces);
    the list itself is sensitive and stays out of the repo — pass it via
    ``--redact-terms`` from a local, gitignored file.
    """
    cleaned = [t.strip() for t in terms if t.strip()]
    if not cleaned:
        return doc

    # Terms tokenize on \w+ runs and join on ANY non-word separator run
    # ([\W_]+) — the same normalization class as the step-4 backstop. So
    # "Chapel Hill" also redacts "chapel-hill" inside a document_id slug and
    # "amanda.wynn" inside an e-mail localpart, and a term written with
    # punctuation ("Kansas City, Missouri") matches the text however the
    # punctuation/casing/spacing came out of extraction. Slug redaction is
    # safe: every reference to the same document_id transforms identically,
    # so citations keep resolving.
    def _term_pattern(term: str) -> re.Pattern[str] | None:
        words = re.findall(r"\w+", term)
        if not words:
            return None
        return re.compile(
            r"\b" + r"[\W_]+".join(re.escape(w) for w in words) + r"\b", re.IGNORECASE
        )

    patterns = [p for term in sorted(cleaned, key=len, reverse=True) if (p := _term_pattern(term))]

    def redact(text: str) -> str:
        for pat in patterns:
            text = pat.sub(_REDACT_LABEL, text)
        return text

    identity = doc.pop("identity", None)
    redacted = _transform_all_strings(doc, redact, include_keys=True)
    if identity is not None:
        redacted["identity"] = identity
    return redacted  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Step 4: deterministic no-known-entity backstop
# ---------------------------------------------------------------------------


def _walk_strings(node: Any, path: str = "$") -> list[tuple[str, str]]:
    """Return ``(path, value)`` for every string reachable from *node*.

    Dict KEYS are included (path ``{key}``), not just values: document_id
    slugs appear as map keys in ``corpus.stats``-style tallies, and a
    counterparty fragment hiding in a key must trip the backstop and the
    residue sweep exactly like one in a value (issue #234 follow-up).
    """
    found: list[tuple[str, str]] = []
    if isinstance(node, str):
        found.append((path, node))
    elif isinstance(node, dict):
        for key, value in node.items():
            if isinstance(key, str):
                found.append((f"{path}.{{{key}}}", key))
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
# Step 5.5: final institution-identity gate (deterministic, full-surface)
# ---------------------------------------------------------------------------
#
# The step-4 backstop only catches names it was GIVEN (``known_entity_names``);
# the #211 proper-noun sweep is list-independent but ADVISORY and only sees the
# free-text *samples*. A real counterparty that was never registered — its name
# surviving in a signature block, a notices address, a dict KEY, or a
# filename-derived ``document_id`` slug — slips past both. That is exactly the
# class of leak that shipped in a public example-playbook publish (2026-07-22):
# a real institution name (a public university, a college-of-health, a
# community-college district) survived in extracted text, stats keys, and slugs
# the pseudonymizer never rewrote.
#
# This is a deterministic, list-INDEPENDENT, FULL-SURFACE gate. It walks every
# string — values AND dict keys, via :func:`_walk_strings` — for high-confidence
# institution-name shapes and any postal address that survived the scrub, and
# HARD-FAILS the publish on a hit. The fix path for a real survivor is the same
# as the step-4 backstop: add the name to ``--redact-terms`` (or register it),
# and re-run. Governing-law states ("the laws of the State of New York") and
# generic descriptors ("College of Health Professions", bare "the University")
# deliberately do NOT match, so the gate stays fail-closed without tripping on
# benign content — those remain the advisory proper-noun sweep's job.

# Distinctive-token guard: a token from one of these is a role/qualifier word,
# never the identifying part of an institution name, so a match carrying only
# these is dropped ("the University", "State University", "of the ...").
_INSTITUTION_TOKEN_STOP = frozenset(
    {
        "the",
        "a",
        "an",
        "our",
        "its",
        "their",
        "this",
        "that",
        "these",
        "those",
        "each",
        "any",
        "all",
        "such",
        "state",
        "community",
        "technical",
        "international",
        "public",
        "private",
        "national",
        "american",
        "new",
        "other",
        "same",
        "said",
        "and",
        "or",
        "of",
        "for",
        "by",
        "to",
        "at",
        "in",
        "on",
        "from",
        "counterparty",
        "company",
        "provider",
        "institution",
        "party",
        "parties",
        "educational",
        "academic",
        "affiliated",
        "affiliate",
    }
)

# All matched against the case/punctuation-normalized string (``_normalize_for_scan``),
# so "University of Westmoor", "university-of-westmoor" (a slug), and
# "UNIVERSITY  OF  WESTMOOR" (double-spaced extraction) all reduce to the same
# "university of westmoor" and match identically.
_INST_UNIVERSITY_OF_RE = re.compile(r"\buniversity\s+of\s+(?:the\s+)?([a-z][a-z0-9]*)")
_INST_X_UNIVERSITY_RE = re.compile(
    r"\b([a-z][a-z0-9]*)\s+"
    r"(?:state\s+|community\s+|technical\s+|international\s+|memorial\s+)?university\b"
)
_INST_X_COLLEGE_RE = re.compile(
    # "junior" is deliberately NOT a qualifier: it is itself the distinctive
    # token of "Junior College [District of ...]", so treating it as a skip
    # word would let a leading stopword ("... Institution junior college ...")
    # absorb the match and hide the name.
    r"\b([a-z][a-z0-9]*)\s+"
    r"(?:community\s+|technical\s+|city\s+|state\s+)?college\b"
)
_INST_REGENTS_RE = re.compile(r"\bregents\s+of\b")
_INST_BOARD_RE = re.compile(r"\bboard\s+of\s+(?:trustees|regents)\b")


def _institution_identity_hits(doc: dict[str, Any]) -> list[tuple[str, str, str]]:
    """Return every ``(path, matched_text, kind)`` institution-name or postal-
    address pattern surviving anywhere in *doc* (values AND dict keys).

    ``kind`` is ``"institution"`` or ``"address"``. List-independent and
    LLM-free — see the section header for why this exists.
    """
    hits: list[tuple[str, str, str]] = []
    for path, text in _walk_strings(doc):
        if not text or text.startswith("sha256:"):
            continue
        normalized = _normalize_for_scan(text)
        # Token-bearing rules: flag only when the distinctive token is a real
        # name, not a role/qualifier word (so "the University" never trips).
        for pattern in (_INST_UNIVERSITY_OF_RE, _INST_X_UNIVERSITY_RE, _INST_X_COLLEGE_RE):
            for match in pattern.finditer(normalized):
                token = match.group(1)
                if token and token not in _INSTITUTION_TOKEN_STOP:
                    hits.append((path, match.group(0).strip(), "institution"))
        # Unambiguous org markers: a governing body is always a specific
        # (usually public) institution, whatever token follows.
        for pattern in (_INST_REGENTS_RE, _INST_BOARD_RE):
            marker = pattern.search(normalized)
            if marker:
                hits.append((path, marker.group(0).strip(), "institution"))
        # Postal address that survived the scrub (address regexes are
        # case-sensitive, so match the ORIGINAL text, not the normalized form).
        for pattern in (_ADDRESS_SPAN_RE, _STATE_ZIP_RE):
            for match in pattern.finditer(text):
                hits.append((path, match.group(0).strip(), "address"))
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
    redact_terms: Sequence[str] = (),
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
                      output; OR step 5's verify pass flagged residual semantic
                      residue and ``accept_residue_risk`` is ``False``; OR the
                      final step-5.5 institution-identity gate found an
                      institution-name shape or postal address surviving
                      anywhere (value or dict key) — unconditional, fixed by
                      naming the survivor in ``redact_terms``.
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

    # --- step 3.5: publication-noise scrub + GC redact list (issue #234) ---
    # E-sign audit trails and notice-block street addresses identify parties
    # and people even after every entity name is aliased; the redact list is
    # the GC's residue-review output for anything the registry didn't know.
    published = _scrub_publication_noise(published)
    published = _apply_redact_terms(published, redact_terms)

    # --- step 4: deterministic no-known-entity backstop (hard, unconditional) ---
    # Redact terms join the backstop: a term the GC ordered redacted
    # surviving anywhere is as blocking as a known entity name.
    hits = _entity_backstop_scan(published, [*known_entity_names, *redact_terms])
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

    # --- step 5.5: final institution-identity gate (deterministic, hard) ---
    # After EVERY transform above, re-scan the whole surface (values AND keys)
    # for institution-name shapes and surviving addresses the born-safe
    # pseudonymizer / redact list never covered. Unconditional — no flag
    # suppresses it (like step 4); the fix is to name the survivor in
    # redact_terms. Catches the class of leak that shipped in a public
    # example-playbook publish: a real counterparty name in a signature block,
    # a stats dict key, or a filename-derived document_id slug.
    identity_hits = _institution_identity_hits(published)
    if identity_hits:
        listing = "; ".join(
            f"{path}: {matched!r} ({kind})" for path, matched, kind in identity_hits[:25]
        )
        more = "" if len(identity_hits) <= 25 else f" (+{len(identity_hits) - 25} more)"
        raise PublishError(
            f"publish blocked: {len(identity_hits)} institution-identity / address "
            f"pattern(s) survived every transform — a real counterparty name or postal "
            f"address the born-safe pseudonymizer and redact list did not cover: "
            f"{listing}{more}. Add each offending name to --redact-terms (or register it "
            "in the entity registry) and re-run. Deterministic backstop — no flag "
            "suppresses it (identity-residue backstop; 2026-07-22 publish incident)."
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
