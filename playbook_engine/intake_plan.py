"""Universal corpus intake (deterministic core) — issue #186.

When a source corpus's directory layout encodes none of the three known
shapes (``flat`` / ``clm_nested`` / ``manifest`` — see ``staging.py``),
``detect_layout`` returns ``"unknown"`` and ``stage()`` refuses to guess
(``UnknownLayoutError``). This module assembles the negotiation story —
which files form one deal, in what order, which is signed — from file
*contents and embedded metadata* instead, for corpora where the directory
layout carries no signal at all (loose files in one folder, email-export
trees, ad-hoc naming).

Everything here is deterministic and offline: no LLM calls. LLM arbitration
of low-confidence clusters lives in the companion skill issue (same epic,
#161) — this module's job stops at proposing a plan, never staging directly.

Pipeline:
1. :func:`build_staging_plan` — ingest every supported file in *src_dir*,
   extract per-file evidence (embedded metadata date, signed-copy
   determination via the existing ``signed_detector``, a content fingerprint
   for near-duplicate clustering via the existing ``version_orderer``, and
   capitalized-span party-name candidates), cluster files into candidate
   deals, order each cluster with the existing min-edit-distance chain
   (``version_orderer.order_versions``), and emit a ``staging_plan.json``-
   shaped dict. Every file the clusterer can't confidently place lands in
   ``unassigned`` with a reason — fail-loud, never silently dropped (same
   philosophy as ``scope_gate.py``).
2. :func:`execute_staging_plan` — execute a (possibly human/skill-edited)
   plan into the canonical ``out_dir/<deal_id>/<NN>__<name>`` layout, reusing
   ``staging.py``'s placement (symlink/copy) and ``hints.yaml`` writers so
   the output is indistinguishable from a directly-staged corpus.

Clustering thresholds (validated against ``examples/judge-fixture/corpus``,
see ``tests/test_intake_plan.py``):
  ``SAME_DEAL_DISTANCE``  — pairwise content distance at or below this value
                            means "two versions of the same deal".
  ``UNRELATED_CEILING``   — a file whose nearest-neighbour distance to every
                            other file exceeds this value has no plausible
                            home in this corpus at all (junk/unrelated) and
                            goes to ``unassigned`` rather than becoming its
                            own single-file "deal".  A file *between* the two
                            thresholds becomes a legitimate single-file deal
                            (e.g. a corpus that really does have just one
                            version of some agreement).

Security: no real agreement content is stored here. All corpus content is
accessed read-only from caller-supplied paths at runtime; ``execute_staging_plan``
writes only to its ``out_dir`` argument, same invariant as ``staging.py``.
"""

from __future__ import annotations

import datetime
import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from playbook_engine.clause_tree import ClauseTree
from playbook_engine.pipeline import _ingest_file
from playbook_engine.signed_detector import SignedStatus, detect_signed
from playbook_engine.staging import (
    _STAGING_MARKER,
    _SUPPORTED,
    _looks_signed,
    _place,
    _recreate_out_dir,
    _write_hints,
)
from playbook_engine.version_orderer import (
    VersionInput,
    VersionOrder,
    order_versions,
    pairwise_distances,
)

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

SAME_DEAL_DISTANCE: float = 0.65
"""Pairwise content distance at/below this: same-deal version cluster."""

UNRELATED_CEILING: float = 0.90
"""A file whose nearest-neighbour distance exceeds this has no plausible
home anywhere in the corpus and is reported as unassigned."""

# ---------------------------------------------------------------------------
# Party-name candidate heuristics (advisory evidence only)
# ---------------------------------------------------------------------------

# CamelCase single-token entity names, e.g. "FixtureCorp", "TechCo" — the
# strongest generic signal since ordinary English prose essentially never
# produces an internal-capital token.
_CAMEL_TOKEN_RE = re.compile(r"\b[A-Z][a-z]+(?:[A-Z][a-zA-Z]*)+\b")

# Multi-word Title-Case spans (2-4 words), e.g. "Acme Widgets Inc" — weaker
# signal, filtered below to drop common contract-boilerplate phrases.
_TITLE_SPAN_RE = re.compile(r"\b(?:[A-Z][a-zA-Z]+\s+){1,3}[A-Z][a-zA-Z]+\b")

# First-word stoplist: multi-word spans opening with one of these are almost
# always boilerplate (jurisdiction refs, defined-term headings), not a party
# name — e.g. "State of California", "Effective Date", "Data Privacy".
_SPAN_STOPWORDS = frozenset(
    {
        "State",
        "Effective",
        "This",
        "The",
        "Each",
        "All",
        "Term",
        "Governing",
        "Data",
        "Student",
        "Third",
        "Agreement",
        "Section",
        "Article",
        "Exhibit",
        "Schedule",
    }
)


def _extract_party_candidates(tree: ClauseTree) -> list[str]:
    """Return capitalized-span party-name candidates found in *tree*'s body text.

    Advisory evidence only — see module docstring. Scans body text (not
    headings, which are structural: "Governing Law", "Indemnification", ...).
    Returns candidates ranked by frequency, most-common first, deduplicated.
    """
    body_text = " ".join(node.text for node in tree.all_nodes() if node.text)
    counts: dict[str, int] = {}

    for m in _CAMEL_TOKEN_RE.finditer(body_text):
        counts[m.group(0)] = counts.get(m.group(0), 0) + 1

    for m in _TITLE_SPAN_RE.finditer(body_text):
        span = m.group(0)
        first_word = span.split(" ", 1)[0]
        if first_word in _SPAN_STOPWORDS:
            continue
        # Jurisdiction refs ("...the State of New York", "Commonwealth of
        # Massachusetts") are not party names even though "New York" itself
        # passes the stopword check above — look back a few words for the
        # governing-law idiom before accepting the span.
        prefix = body_text[max(0, m.start() - 20) : m.start()]
        if re.search(r"(?:State|Commonwealth|Province) of\s*$", prefix):
            continue
        counts[span] = counts.get(span, 0) + 1

    return sorted(counts, key=lambda k: (-counts[k], k))


# ---------------------------------------------------------------------------
# Embedded-metadata date extraction
# ---------------------------------------------------------------------------

_PDF_DATE_RE = re.compile(r"D:(\d{4})(\d{2})(\d{2})")


def _extract_metadata_date(path: Path) -> str | None:
    """Return an ISO ``YYYY-MM-DD`` date from *path*'s embedded metadata, if any.

    DOCX: core properties (``created``/``modified``) via ``python-docx`` —
    already a project dependency. PDF: ``/CreationDate`` or ``/ModDate`` via
    ``pdfplumber`` — also already a dependency. RTF carries no equivalent
    embedded-metadata convention this module extracts; callers fall back to
    filesystem mtime (see :func:`_fallback_timestamp`).

    Never raises — a corrupt or unreadable metadata stream just yields
    ``None``, same fail-soft posture as the rest of this evidence-gathering
    pass (a missing date is weak evidence, not a hard error).
    """
    ext = path.suffix.lower()
    try:
        if ext == ".docx":
            from docx import Document  # noqa: PLC0415

            props = Document(str(path)).core_properties
            dt = props.created or props.modified
            if dt is not None:
                return str(dt.date().isoformat())
        elif ext == ".pdf":
            import pdfplumber  # noqa: PLC0415

            with pdfplumber.open(str(path)) as pdf:
                meta = pdf.metadata or {}
                raw = meta.get("CreationDate") or meta.get("ModDate")
                if raw:
                    m = _PDF_DATE_RE.search(raw)
                    if m:
                        year, month, day = m.groups()
                        return f"{year}-{month}-{day}"
    except Exception:  # noqa: BLE001 - metadata extraction must never crash the plan
        return None
    return None


def _fallback_timestamp(path: Path) -> str:
    """Filesystem mtime as an ISO date — the weakest available ordering signal.

    Used only when a file carries no embedded-metadata date (all RTF, or a
    DOCX/PDF whose properties are stripped/absent). Still real evidence: on
    a corpus copied off a live filesystem, mtime plausibly tracks authorship
    order even without content metadata.
    """
    return datetime.date.fromtimestamp(path.stat().st_mtime).isoformat()


# ---------------------------------------------------------------------------
# Per-file evidence
# ---------------------------------------------------------------------------


@dataclass
class _FileEvidence:
    rel_path: str
    tree: ClauseTree
    signed: SignedStatus
    timestamp: str
    timestamp_is_metadata: bool
    party_candidates: list[str]
    filename_signed_cue: bool
    ingest_error: str | None = None


def _gather_evidence(
    src_dir: Path,
    progress: Callable[[str], None] = lambda _: None,
) -> tuple[dict[str, _FileEvidence], list[dict[str, str]]]:
    """Ingest every supported file under *src_dir* and extract per-file evidence.

    Returns ``(evidence_by_relpath, unreadable)`` — files that fail to ingest
    (corrupt/unsupported content) are reported separately with a reason
    rather than silently dropped or crashing the whole run.

    Args:
        progress: Callable receiving one status line per file ingested (issue
                  #44 — this loop can run seconds per file on scanned PDFs, so
                  a large loose corpus otherwise sits silent for minutes with
                  no way to distinguish a hang from work).
    """
    files = sorted(
        p
        for p in src_dir.rglob("*")
        if p.is_file()
        and p.suffix.lower() in _SUPPORTED
        and not any(part.startswith(".") for part in p.relative_to(src_dir).parts)
    )

    evidence: dict[str, _FileEvidence] = {}
    unreadable: list[dict[str, str]] = []

    for i, path in enumerate(files, start=1):
        rel = str(path.relative_to(src_dir))
        progress(f"  [{i}/{len(files)}] {rel}")
        try:
            tree = _ingest_file(path, path.stem, "1")
        except Exception as exc:  # noqa: BLE001 - unreadable file, not a fatal error
            unreadable.append({"path": rel, "reason": f"could not ingest file: {exc}"})
            continue

        signed = detect_signed(tree)
        metadata_date = _extract_metadata_date(path)
        timestamp = metadata_date or _fallback_timestamp(path)

        evidence[rel] = _FileEvidence(
            rel_path=rel,
            tree=tree,
            signed=signed,
            timestamp=timestamp,
            timestamp_is_metadata=metadata_date is not None,
            party_candidates=_extract_party_candidates(tree),
            filename_signed_cue=_looks_signed(path.stem),
        )

    return evidence, unreadable


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------


class _UnionFind:
    """Minimal union-find for deterministic connected-components clustering."""

    def __init__(self, items: list[str]) -> None:
        self._parent = {item: item for item in items}

    def find(self, item: str) -> str:
        while self._parent[item] != item:
            self._parent[item] = self._parent[self._parent[item]]
            item = self._parent[item]
        return item

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # Deterministic root choice so output doesn't depend on call order.
            if rb < ra:
                ra, rb = rb, ra
            self._parent[rb] = ra

    def components(self) -> list[list[str]]:
        groups: dict[str, list[str]] = {}
        for item in self._parent:
            groups.setdefault(self.find(item), []).append(item)
        return [sorted(members) for members in groups.values()]


def _cluster(
    rel_paths: list[str],
    evidence: dict[str, _FileEvidence],
    dist: dict[tuple[str, str], float],
) -> list[list[str]]:
    """Union files into candidate deals: near-duplicate content, or a shared
    party-name candidate corroborated by not-implausible content distance."""
    uf = _UnionFind(rel_paths)
    for i, a in enumerate(rel_paths):
        for b in rel_paths[i + 1 :]:
            d = dist[(a, b)]
            if d <= SAME_DEAL_DISTANCE:
                uf.union(a, b)
                continue
            if d <= UNRELATED_CEILING:
                shared = set(evidence[a].party_candidates) & set(evidence[b].party_candidates)
                if shared:
                    uf.union(a, b)
    return uf.components()


def _nearest_distance(
    rel_path: str, others: list[str], dist: dict[tuple[str, str], float]
) -> float | None:
    """Return the smallest pairwise distance from *rel_path* to any of *others*."""
    candidates = [dist[(rel_path, other)] for other in others if other != rel_path]
    return min(candidates) if candidates else None


# ---------------------------------------------------------------------------
# Deal assembly
# ---------------------------------------------------------------------------


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "deal"


def _build_deal(
    members: list[str],
    evidence: dict[str, _FileEvidence],
    dist: dict[tuple[str, str], float],
    all_paths: list[str],
) -> dict[str, Any]:
    """Order one cluster's members and assemble its plan entry (minus deal_id)."""
    versions = [
        VersionInput(
            version_id=rel,
            tree=evidence[rel].tree,
            signed=evidence[rel].signed,
            timestamp=evidence[rel].timestamp,
        )
        for rel in members
    ]
    order: VersionOrder = order_versions(versions)

    counterparty_guess: str | None = None
    for rel in order.ordered_ids:
        candidates = evidence[rel].party_candidates
        if candidates:
            counterparty_guess = candidates[0]
            break

    if len(members) > 1:
        pair_dists = [dist[(a, b)] for i, a in enumerate(members) for b in members[i + 1 :]]
        avg_dist = sum(pair_dists) / len(pair_dists)
        confidence = round(max(0.0, 1.0 - avg_dist), 2)
    else:
        others = [p for p in all_paths if p not in members]
        nearest = _nearest_distance(members[0], others, dist)
        confidence = round(max(0.0, 1.0 - nearest), 2) if nearest is not None else 1.0

    files: list[dict[str, Any]] = []
    for i, rel in enumerate(order.ordered_ids, start=1):
        ev = evidence[rel]
        file_evidence: list[str] = []
        file_evidence.append(
            f"metadata_date:{ev.timestamp}"
            if ev.timestamp_is_metadata
            else f"fs_mtime:{ev.timestamp}"
        )
        file_evidence.append(f"signed:{ev.signed.basis}")
        if ev.filename_signed_cue:
            file_evidence.append("filename_signed_cue")
        if ev.party_candidates:
            file_evidence.append(f"party_candidate:{ev.party_candidates[0]}")
        mates = [m for m in order.ordered_ids if m != rel]
        if mates:
            file_evidence.append(f"near_dup_of:{','.join(mates)}")

        files.append(
            {
                "path": rel,
                "proposed_version": i,
                "signed": ev.signed.signed,
                "evidence": file_evidence,
            }
        )

    return {
        "counterparty_guess": counterparty_guess,
        "confidence": confidence,
        "files": files,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_staging_plan(
    src_dir: Path,
    progress: Callable[[str], None] = lambda _: None,
) -> dict[str, Any]:
    """Assemble a ``staging_plan.json``-shaped proposal for *src_dir*.

    Never stages anything — the returned dict is a proposal for
    :func:`execute_staging_plan` (optionally hand/skill-edited first).

    Args:
        src_dir: Corpus root with no known layout (see ``staging.detect_layout``).
        progress: Callable receiving status line strings (issue #44) — one
                  per file ingested, plus one line before the O(n^2) pairwise
                  distance computation. Defaults to a no-op.

    Returns:
        A dict matching the schema documented in issue #186: ``layout``,
        ``deals`` (each with ``deal_id``, ``counterparty_guess``,
        ``confidence``, ``files``), ``unassigned``, and ``warnings``.
    """
    evidence, unreadable = _gather_evidence(src_dir, progress)
    rel_paths = sorted(evidence)

    warnings: list[str] = []
    unassigned: list[dict[str, str]] = list(unreadable)

    if not rel_paths:
        return {
            "layout": "unknown",
            "deals": [],
            "unassigned": unassigned,
            "warnings": warnings,
        }

    fingerprint_versions = [
        VersionInput(version_id=rel, tree=evidence[rel].tree, signed=evidence[rel].signed)
        for rel in rel_paths
    ]
    n_files = len(fingerprint_versions)
    progress(f"  computing {n_files * (n_files - 1) // 2} pairwise distances…")
    dist = pairwise_distances(fingerprint_versions)

    clusters = _cluster(rel_paths, evidence, dist)

    deal_entries: list[dict[str, Any]] = []
    for members in clusters:
        if len(members) == 1:
            others = [p for p in rel_paths if p not in members]
            nearest = _nearest_distance(members[0], others, dist)
            if nearest is not None and nearest > UNRELATED_CEILING:
                unassigned.append(
                    {
                        "path": members[0],
                        "reason": (
                            f"no cluster above threshold (nearest content distance "
                            f"{nearest:.2f} to any other file)"
                        ),
                    }
                )
                continue
        deal_entries.append(_build_deal(members, evidence, dist, rel_paths))

    # Deterministic ordering independent of naming: sort by each deal's
    # earliest (sorted) member path before assigning generic slugs.
    deal_entries.sort(key=lambda d: min(f["path"] for f in d["files"]))

    used_slugs: set[str] = set()
    deals: list[dict[str, Any]] = []
    for idx, entry in enumerate(deal_entries, start=1):
        base = (
            _slugify(entry["counterparty_guess"]) if entry["counterparty_guess"] else f"deal-{idx}"
        )
        slug = base
        n = 2
        while slug in used_slugs:
            slug = f"{base}-{n}"
            n += 1
        used_slugs.add(slug)

        deals.append(
            {
                "deal_id": slug,
                "counterparty_guess": entry["counterparty_guess"],
                "confidence": entry["confidence"],
                "files": entry["files"],
            }
        )

    return {
        "layout": "unknown",
        "deals": deals,
        "unassigned": unassigned,
        "warnings": warnings,
    }


def _is_safe_path_component(value: str) -> bool:
    """True if *value* is usable as a single path segment with no traversal risk.

    Rejects an absolute path, either separator (checked literally so this is
    platform-independent — a ``deal_id`` containing a Windows-style ``\\``
    must still be rejected when running on POSIX, and vice versa), and the
    special ``.``/``..`` segments.
    """
    if os.path.isabs(value) or "/" in value or "\\" in value:
        return False
    return value not in (".", "..")


# Filenames `stage --plan` itself writes directly into out_dir (see
# cli.py's stage_cmd: the executed plan record, the scaffolded config, and
# staging.py's own marker). A deal_id equal to one of these shadows that
# file with a same-named deal *directory*, so the later write raises a raw
# `IsADirectoryError` instead of the located, human-readable error this
# validator exists to produce (issue #45 round 2).
_RESERVED_OUT_DIR_NAMES = frozenset({"staging_plan.json", "playbook.config.yaml", _STAGING_MARKER})


def _validate_plan(plan: dict[str, Any], src_dir: Path) -> None:
    """Validate a (possibly hand/skill-edited) staging plan's shape.

    ``execute_staging_plan`` calls this *before* ``_recreate_out_dir`` wipes
    anything (issue #45): the plan file is documented as hand-editable, so a
    malformed ``deals`` list is expected input, not a programming error — it
    must be rejected with a located, human-readable message instead of an
    unhandled ``KeyError``/``TypeError`` fired after ``out_dir`` (and
    possibly the user's own hand-edited plan, if it lives at
    ``out_dir/staging_plan.json``) has already been deleted.

    Only the fields ``execute_staging_plan`` actually consumes — ``deal_id``,
    ``files`` (each with ``path`` and ``proposed_version``) — are checked
    here; ``counterparty_guess``, ``confidence``, ``unassigned``, and
    ``warnings`` are informational and not validated.

    ``deal_id`` is used directly as an ``out_dir`` subdirectory name and
    ``path`` as a ``src_dir``-relative lookup (see ``execute_staging_plan``),
    so both are also checked for path-traversal/absolute-path escapes here —
    the plan is untrusted, hand/skill-edited input, and ``stage``'s
    documented contract is that it writes only inside *out_dir* and reads
    only from inside *src_dir*. ``deal_id`` is also checked against the
    filenames ``stage --plan`` itself writes into ``out_dir`` (see
    ``_RESERVED_OUT_DIR_NAMES``), and ``path`` is checked lexically for
    ``..`` components rather than by resolving symlinks — a corpus file that
    is itself a symlink pointing outside *src_dir* is still a legitimate,
    in-corpus ``path`` (issue #45 round 2).

    Args:
        plan:    The plan dict to validate.
        src_dir: Corpus root ``path`` entries are relative to. Unused for
            the escape check itself (which is purely lexical) but kept for
            signature/documentation symmetry with ``execute_staging_plan``.

    Raises:
        ValueError: the first shape problem found, naming the offending deal
            index (and file index, where applicable) and field.
    """
    if not isinstance(plan, dict):
        raise ValueError(f"plan must be a JSON object, got {type(plan).__name__}")

    deals = plan.get("deals")
    if not isinstance(deals, list):
        raise ValueError("plan.deals must be a list")

    seen_deal_ids: set[str] = set()
    for i, deal in enumerate(deals):
        if not isinstance(deal, dict):
            raise ValueError(f"deal {i}: expected an object, got {type(deal).__name__}")

        deal_id = deal.get("deal_id")
        if not isinstance(deal_id, str) or not deal_id:
            raise ValueError(f"deal {i}: 'deal_id' must be a non-empty string")
        if not _is_safe_path_component(deal_id):
            raise ValueError(
                f"deal {i}: 'deal_id' {deal_id!r} must be a single path segment "
                "(no '/', '\\', absolute path, or '..')"
            )
        if deal_id in _RESERVED_OUT_DIR_NAMES:
            raise ValueError(
                f"deal {i}: 'deal_id' {deal_id!r} collides with a reserved output "
                f"filename ({sorted(_RESERVED_OUT_DIR_NAMES)!r}) that stage --from-plan "
                "writes directly into out_dir"
            )
        if deal_id in seen_deal_ids:
            raise ValueError(f"deal {i}: duplicate deal_id {deal_id!r}")
        seen_deal_ids.add(deal_id)

        files = deal.get("files")
        if not isinstance(files, list) or not files:
            raise ValueError(f"deal {i} ({deal_id}): 'files' must be a non-empty list")

        for j, f in enumerate(files):
            if not isinstance(f, dict):
                raise ValueError(
                    f"deal {i} ({deal_id}): file {j}: expected an object, got {type(f).__name__}"
                )
            path = f.get("path")
            if not isinstance(path, str) or not path:
                raise ValueError(
                    f"deal {i} ({deal_id}): file {j}: 'path' must be a non-empty string"
                )
            if os.path.isabs(path):
                raise ValueError(
                    f"deal {i} ({deal_id}): file {j}: 'path' {path!r} must be relative "
                    "to src_dir, not absolute"
                )
            if ".." in Path(path).parts:
                raise ValueError(f"deal {i} ({deal_id}): file {j}: 'path' {path!r} escapes src_dir")
            proposed_version = f.get("proposed_version")
            # bool is a subclass of int in Python — exclude it explicitly so
            # `"proposed_version": true` doesn't slip through as version 1.
            if not isinstance(proposed_version, int) or isinstance(proposed_version, bool):
                raise ValueError(
                    f"deal {i} ({deal_id}): file {j}: 'proposed_version' must be an int, "
                    f"got {type(proposed_version).__name__}"
                )


def _check_plan_files_exist(plan: dict[str, Any], src_dir: Path) -> None:
    """Raise if any plan file ``path`` doesn't resolve to a real file under *src_dir*.

    Checked before ``_recreate_out_dir`` wipes anything — same ordering
    rationale as :func:`_validate_plan`. Without this, a hand-edited plan
    with a typo'd path (or one run against the wrong ``src_dir``) reaches
    ``_place``, which in symlink mode (the default) happily creates a
    dangling symlink via ``os.symlink``; ``_discover_versions`` then drops
    that version silently (``p.is_file()`` is ``False`` for a dangling
    link), and ``stage --plan`` reports "staged : N version(s) ... OK" as if
    nothing were wrong (issue #52). In ``--copy`` mode the same typo instead
    raises an unhandled ``FileNotFoundError`` from ``shutil.copy2`` — this
    check replaces that traceback with a clean, located error too.

    Args:
        plan:    The plan dict, already shape-validated by :func:`_validate_plan`.
        src_dir: Corpus root ``path`` entries are relative to.

    Raises:
        ValueError: lists every missing ``deal_id``/``path`` pair found (not
            just the first), mirroring ``StagingResult.missing`` for the
            manifest layout so a plan with several typos is fixed in one pass.
    """
    missing: list[str] = []
    for deal in plan.get("deals", []):
        deal_id = deal["deal_id"]
        for f in deal["files"]:
            src = src_dir / f["path"]
            if not src.is_file():
                missing.append(f"{deal_id}/{f['path']}")
    if missing:
        raise ValueError(
            f"{len(missing)} plan file(s) not found under {src_dir}: " + ", ".join(missing)
        )


def execute_staging_plan(
    plan: dict[str, Any],
    src_dir: Path,
    out_dir: Path,
    *,
    copy_files: bool = False,
) -> Any:
    """Execute a ``staging_plan.json``-shaped *plan* into *out_dir*.

    Reuses ``staging.py``'s placement (symlink/copy) and ``hints.yaml``
    writer so the output is indistinguishable from a directly-staged corpus.
    *plan* may have been hand/skill-edited between :func:`build_staging_plan`
    and this call — only ``deal_id``, ``path``, ``proposed_version``, and
    ``signed`` are consumed here, and their shape is validated up front (see
    :func:`_validate_plan`) before *out_dir* is touched.

    Args:
        plan:        A plan dict as produced by :func:`build_staging_plan`.
        src_dir:     Corpus root the plan's ``path`` entries are relative to.
        out_dir:     Destination directory. Recreated on each call (same
                     contract as ``staging.stage``) — refused if it overlaps
                     *src_dir*, or if it exists, is non-empty, and isn't
                     itself a previous staging output (issue #248).
        copy_files:  Write real file copies instead of absolute symlinks
                     (see ``staging._place``).

    Returns:
        A ``staging.StagingResult`` with ``layout="unknown"``.

    Raises:
        ValueError: *plan* is malformed (issue #45) — see
            :func:`_validate_plan` — a plan file path doesn't exist under
            *src_dir* (issue #52) — see :func:`_check_plan_files_exist` — or
            *out_dir* overlaps *src_dir*, or *out_dir* exists, is non-empty,
            and is not itself a previous staging output (see
            ``staging._recreate_out_dir``). Does not touch *out_dir* in any
            of these cases.
    """
    from playbook_engine.staging import StagingResult  # noqa: PLC0415

    _validate_plan(plan, src_dir)
    _check_plan_files_exist(plan, src_dir)
    _recreate_out_dir(src_dir, out_dir)

    staged = 0
    agreement_count = 0

    for deal in plan.get("deals", []):
        deal_id = deal["deal_id"]
        dest = out_dir / deal_id
        dest.mkdir()

        files_sorted = sorted(deal["files"], key=lambda f: f["proposed_version"])
        order: list[str] = []
        signed_name: str | None = None

        for f in files_sorted:
            src = src_dir / f["path"]
            name = f"{f['proposed_version']:02d}__{src.name}"
            _place(src, dest / name, copy_files=copy_files)
            order.append(name)
            if f.get("signed"):
                signed_name = name
            staged += 1

        _write_hints(dest, order, signed_name)
        agreement_count += 1

    return StagingResult(
        out_dir=out_dir,
        layout="unknown",
        staged_count=staged,
        agreement_count=agreement_count,
    )
