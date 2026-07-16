"""Taxonomy inductor — proposes a candidate taxonomy from corpus clause headings.

When no taxonomy exists for a new agreement type, this pre-pass:
1. Collects all clause headings from the corpus clause trees.
2. Clusters similar headings (token-Jaccard similarity, deterministic).
3. Maps each cluster to a reference category where possible — either a
   genuine CUAD v1 category (``cuad_origin`` set) or a curated
   playbook-engine-base supplemental category (``cuad_origin`` stays
   ``None``; see ``_SUPPLEMENTAL``). Only a genuine CUAD v1 match may set
   ``cuad_origin``, so induced entries never carry false CUAD provenance.
4. Assigns status (OPF §3.3):
   - ``active``   — CUAD-mapped cluster present in ≥ representation_threshold docs.
   - ``custom``   — unmapped (novel) cluster present in ≥ representation_threshold docs.
   - ``inactive`` — any cluster below the threshold, or CUAD-mapped+rare cluster; both
                    CUAD-mapped-but-rare (cuad_origin set) and unmapped-rare (cuad_origin
                    null) become inactive so the attorney can decide whether to promote.
5. Returns a ``TaxonomyInductionResult`` with a candidate ``Taxonomy`` plus
   per-entry examples for attorney review.

Induction is fully deterministic (no LLM).  The result is a starting point for
attorney curation, not a finished taxonomy.

Algorithm detail:
  Headings are normalized (lowercase, stripped punctuation, stop-word removal)
  and compared by token-Jaccard similarity.  Groups are sorted by descending
  frequency before clustering, so the most-common heading becomes the cluster
  representative when there are ties — giving stable, reproducible output.
"""

from __future__ import annotations

import os
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from playbook_engine.clause_tree import ClauseTree
from playbook_engine.taxonomy import Taxonomy, TaxonomyEntry, load_taxonomy

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPRESENTATION_THRESHOLD: float = 0.20
"""Fraction of documents a cluster must appear in to earn ``active``/``custom`` status."""

CLUSTER_SIMILARITY_THRESHOLD: float = 0.50
"""Minimum token-overlap-coefficient to merge two normalized headings into one cluster.

Uses overlap coefficient (|A∩B| / min(|A|,|B|)) rather than Jaccard so that a
short heading is absorbed into a longer one that contains it as a subset:
  "Indemnification" + "Indemnification and Hold Harmless"  → merged  (coeff=1.0)
  "Term"            + "Term of Agreement"                 → merged  (coeff=1.0)
  "Indemnification" + "Governing Law"                     → separate (coeff=0.0)
"""

CUAD_MATCH_THRESHOLD: float = 0.60
"""Minimum token-Jaccard similarity to accept a CUAD category match.

Set above 0.50 to avoid half-overlap false positives: a single-token heading
like "Term" shares exactly half its tokens with "Renewal Term" (jaccard=0.50)
but should NOT be mapped to the Renewal Term CUAD category — the threshold of
0.60 correctly blocks this while still accepting single-token exact matches
(e.g. "Indemnification" → 1.0, "Insurance" → 1.0).
"""

_SNIPPET_MAX_CHARS: int = 200
_MAX_EXAMPLES_PER_ENTRY: int = 3

# Stop words excluded from token-Jaccard to avoid false clustering on common words.
_STOP_WORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "in",
        "is",
        "its",
        "of",
        "or",
        "that",
        "the",
        "this",
        "to",
        "with",
    }
)

# ---------------------------------------------------------------------------
# CUAD v1 category reference
# Source: "CUAD: An Expert-Annotated NLP Dataset for Legal Contracts" (2021).
#
# Loaded from the shipped ``spec/taxonomy/cuad-base.yaml`` — that file (not
# this module) is the single source of truth for the reference-category list,
# so the same category set is available both as a standalone
# ``builtin:cuad-base.yaml`` taxonomy for adopters and as the inductor's match
# reference (issue #167). Only a genuine CUAD v1 match may set
# ``cuad_origin``; headings that merely resemble common contract sections but
# are not part of the upstream CUAD v1 taxonomy belong in the separate
# ``general-commercial.yaml`` (loaded below as ``_SUPPLEMENTAL``), so matches
# against them are never mislabeled with CUAD provenance (see
# ``_best_reference_match``).
# ---------------------------------------------------------------------------

_TAXONOMY_DIR: Path = Path(__file__).resolve().parent.parent / "spec" / "taxonomy"
_CUAD_BASE_PATH: Path = _TAXONOMY_DIR / "cuad-base.yaml"
_GENERAL_COMMERCIAL_PATH: Path = _TAXONOMY_DIR / "general-commercial.yaml"


def _load_reference_categories(path: Path) -> tuple[tuple[str, str], ...]:
    """Load an (id, canonical_label) reference-category list from a shipped
    taxonomy YAML (module-relative path, independent of process cwd)."""
    taxonomy = load_taxonomy(path)
    return tuple((entry.id, entry.label) for entry in taxonomy.entries)


_CUAD_V1: tuple[tuple[str, str], ...] = _load_reference_categories(_CUAD_BASE_PATH)
"""Genuine CUAD v1 categories — loaded from ``spec/taxonomy/cuad-base.yaml``."""

# ---------------------------------------------------------------------------
# Supplemental category reference — NOT part of CUAD v1.
#
# Loaded from the shipped ``spec/taxonomy/general-commercial.yaml``. These are
# curated playbook-engine additions for common contract sections that the
# upstream CUAD v1 taxonomy does not cover. A cluster matched to one of these
# must NOT be attributed to CUAD: ``cuad_origin`` stays ``None`` and the
# induced entry's ``source`` is recorded as ``playbook-engine-base``
# (2026-07 dual-repo audit finding: these were previously shipped under the
# same CUAD v1 citation as genuine entries, which is false provenance).
# ---------------------------------------------------------------------------

_SUPPLEMENTAL_SOURCE: str = "playbook-engine-base"
_CUAD_SOURCE: str = "CUAD v1"

_SUPPLEMENTAL: tuple[tuple[str, str], ...] = _load_reference_categories(_GENERAL_COMMERCIAL_PATH)
"""Curated non-CUAD categories — loaded from ``spec/taxonomy/general-commercial.yaml``."""


def _token_map(
    entries: tuple[tuple[str, str], ...],
) -> tuple[tuple[str, str, frozenset[str]], ...]:
    """Pre-build (id, label, token-set) tuples for a reference category list."""
    return tuple(
        (
            cid,
            clabel,
            frozenset(
                w for w in re.sub(r"[^\w\s]", " ", clabel.lower()).split() if w not in _STOP_WORDS
            ),
        )
        for cid, clabel in entries
    )


# Pre-built token maps for reference matching (built once at module load).
_CUAD_TOKEN_MAP: tuple[tuple[str, str, frozenset[str]], ...] = _token_map(_CUAD_V1)
_SUPPLEMENTAL_TOKEN_MAP: tuple[tuple[str, str, frozenset[str]], ...] = _token_map(_SUPPLEMENTAL)

# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClusterExample:
    """One concrete example citation for an induced taxonomy entry."""

    document_id: str
    clause_path: str
    heading: str
    text_snippet: str  # first _SNIPPET_MAX_CHARS chars of the clause body

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "clause_path": self.clause_path,
            "heading": self.heading,
            "text_snippet": self.text_snippet,
        }


@dataclass(frozen=True)
class InducedEntry:
    """One proposed taxonomy entry with provenance metadata."""

    entry: TaxonomyEntry
    document_frequency: float  # fraction of total documents this cluster appears in
    examples: tuple[ClusterExample, ...]
    source: str = ""
    """Where the entry's reference-category match (if any) came from:
    ``"CUAD v1"``, ``"playbook-engine-base"``, or ``""`` for an unmatched
    (novel/custom) cluster. Distinct from ``entry.cuad_origin``, which is
    ``None`` unless ``source == "CUAD v1"`` — this field preserves the
    provenance even when ``cuad_origin`` correctly stays unset."""


@dataclass
class TaxonomyInductionResult:
    """Result of ``induce_taxonomy()``."""

    taxonomy: Taxonomy
    induced_entries: list[InducedEntry]
    total_documents: int


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def induce_taxonomy(
    trees: Sequence[ClauseTree],
    *,
    representation_threshold: float = REPRESENTATION_THRESHOLD,
    cluster_similarity_threshold: float = CLUSTER_SIMILARITY_THRESHOLD,
    cuad_match_threshold: float = CUAD_MATCH_THRESHOLD,
) -> TaxonomyInductionResult:
    """Induce a candidate taxonomy from the clause headings in *trees*.

    Args:
        trees:                       Clause trees from the corpus (one per
                                     representative document).
        representation_threshold:    Minimum document-frequency for
                                     ``active``/``custom`` status.
        cluster_similarity_threshold: Jaccard similarity to merge headings
                                     into the same cluster.
        cuad_match_threshold:        Jaccard similarity to accept a CUAD match.

    Returns:
        ``TaxonomyInductionResult`` with a candidate ``Taxonomy`` and per-entry
        examples for attorney review.
    """
    if not trees:
        return TaxonomyInductionResult(
            taxonomy=Taxonomy(source="induced", entries=[]),
            induced_entries=[],
            total_documents=0,
        )

    total_docs = len(trees)

    # Step 1: collect (doc_id, clause_path, original_heading, text_snippet)
    _Occ = tuple[str, str, str, str]
    occurrences: list[_Occ] = []
    for tree in trees:
        for node in tree.all_nodes():
            heading = (node.heading or "").strip()
            if not heading:
                continue
            snippet = (node.text or "")[:_SNIPPET_MAX_CHARS]
            occurrences.append((tree.document_id, node.clause_path, heading, snippet))

    if not occurrences:
        return TaxonomyInductionResult(
            taxonomy=Taxonomy(source="induced", entries=[]),
            induced_entries=[],
            total_documents=total_docs,
        )

    # Step 2: group by exact normalized heading.
    heading_groups: dict[str, list[_Occ]] = {}
    for occ in occurrences:
        norm = _normalize_heading(occ[2])
        if norm:
            heading_groups.setdefault(norm, []).append(occ)

    # Step 3: sort groups by (desc frequency, asc normalized heading) for determinism.
    sorted_groups = sorted(
        heading_groups.items(),
        key=lambda x: (-len(x[1]), x[0]),
    )

    # Step 4: greedy heading clustering.
    # Each cluster is (canonical_original_heading, list_of_occurrences).
    clusters: list[tuple[str, list[_Occ]]] = []
    cluster_tokens: list[frozenset[str]] = []

    for norm_heading, group_occ in sorted_groups:
        tokens = _heading_tokens(norm_heading)
        if not tokens:
            continue

        best_idx = -1
        best_sim = 0.0
        for i, c_tokens in enumerate(cluster_tokens):
            sim = _overlap_coefficient(tokens, c_tokens)
            if sim >= cluster_similarity_threshold and sim > best_sim:
                best_sim = sim
                best_idx = i

        if best_idx >= 0:
            clusters[best_idx][1].extend(group_occ)
        else:
            # Representative heading: original heading of the first occurrence
            # in the most-frequent group (groups are sorted desc by frequency).
            canonical = group_occ[0][2]
            clusters.append((canonical, list(group_occ)))
            cluster_tokens.append(tokens)

    # Step 5: build InducedEntry for each cluster.
    seen_ids: set[str] = set()
    induced_entries: list[InducedEntry] = []

    for canonical_heading, cluster_occ in clusters:
        doc_ids = {occ[0] for occ in cluster_occ}
        doc_frequency = len(doc_ids) / total_docs

        tokens = _heading_tokens(_normalize_heading(canonical_heading))
        match_id, match_label, match_sim, match_source = _best_reference_match(tokens)
        has_match = match_id is not None and match_sim >= cuad_match_threshold
        is_genuine_cuad = has_match and match_source == _CUAD_SOURCE

        if has_match:
            status = "active" if doc_frequency >= representation_threshold else "inactive"
            # Only a genuine CUAD v1 match may set cuad_origin — a supplemental
            # (playbook-engine-base) match must never claim CUAD provenance.
            cuad_origin: str | None = match_label if is_genuine_cuad else None
            source = match_source or ""
        elif doc_frequency >= representation_threshold:
            status = "custom"
            cuad_origin = None
            source = ""
        else:
            status = "inactive"
            cuad_origin = None
            source = ""

        entry_id = _unique_entry_id(_to_entry_id(canonical_heading), seen_ids)
        seen_ids.add(entry_id)

        entry = TaxonomyEntry(
            id=entry_id,
            label=canonical_heading,
            status=status,
            cuad_origin=cuad_origin,
            description="",
        )

        examples = _pick_examples(cluster_occ)

        induced_entries.append(
            InducedEntry(
                entry=entry,
                document_frequency=doc_frequency,
                examples=examples,
                source=source,
            )
        )

    # Sort: active/custom first (by doc_frequency desc), then inactive.
    induced_entries.sort(
        key=lambda ie: (0 if ie.entry.status in ("active", "custom") else 1, -ie.document_frequency)
    )

    return TaxonomyInductionResult(
        taxonomy=Taxonomy(source="induced", entries=[ie.entry for ie in induced_entries]),
        induced_entries=induced_entries,
        total_documents=total_docs,
    )


def emit_taxonomy_yaml(result: TaxonomyInductionResult, path: Path) -> None:
    """Write the induced taxonomy to *path* as a YAML file (atomic rename).

    The emitted file is loadable by ``load_taxonomy()`` from ``taxonomy.py``.
    Each entry also carries an ``examples`` list and a ``source`` string (the
    actual provenance of any reference-category match — ``"CUAD v1"`` or
    ``"playbook-engine-base"``) for attorney review; ``load_taxonomy`` ignores
    both fields.

    Args:
        result: Result from ``induce_taxonomy()``.
        path:   Destination YAML file path.  Parent directories are created.
    """
    entries_data: list[dict[str, Any]] = []
    for ie in result.induced_entries:
        e = ie.entry
        entry_dict: dict[str, Any] = {
            "id": e.id,
            "label": e.label,
            "status": e.status,
            "cuad_origin": e.cuad_origin,
            "description": e.description,
        }
        if ie.source:
            entry_dict["source"] = ie.source
        if ie.examples:
            entry_dict["examples"] = [ex.to_dict() for ex in ie.examples]
        entries_data.append(entry_dict)

    data: dict[str, Any] = {
        "source": "induced",
        "entries": entries_data,
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".yaml.tmp")
    tmp.write_text(
        yaml.dump(data, allow_unicode=True, default_flow_style=False, sort_keys=False),
        encoding="utf-8",
    )
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _normalize_heading(heading: str) -> str:
    """Lowercase and strip punctuation/extra whitespace."""
    s = heading.lower()
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _heading_tokens(normalized: str) -> frozenset[str]:
    """Return meaningful tokens (stop words excluded)."""
    return frozenset(w for w in normalized.split() if w not in _STOP_WORDS)


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    """Token-Jaccard similarity; 1 = identical token sets.

    Used for reference-category mapping (``_best_reference_match``).
    """
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _overlap_coefficient(a: frozenset[str], b: frozenset[str]) -> float:
    """Token-overlap coefficient: |A∩B| / min(|A|, |B|).

    Used for heading clustering.  A short heading is absorbed into a longer
    one that contains it as a subset (e.g. "Indemnification" into
    "Indemnification and Hold Harmless"), which Jaccard would miss.
    """
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def _best_reference_match(
    tokens: frozenset[str],
) -> tuple[str | None, str | None, float, str | None]:
    """Return (id, label, best_similarity, source) for the best reference match.

    Searches both the genuine CUAD v1 categories and the curated supplemental
    categories (``_SUPPLEMENTAL``). ``source`` is ``"CUAD v1"`` or
    ``"playbook-engine-base"`` depending on which list produced the best
    match, so callers can tell genuine CUAD provenance apart from a curated
    addition that merely looks similar.
    """
    best_id: str | None = None
    best_label: str | None = None
    best_sim: float = 0.0
    best_source: str | None = None
    for c_id, c_label, c_tokens in _CUAD_TOKEN_MAP:
        sim = _jaccard(tokens, c_tokens)
        if sim > best_sim:
            best_sim = sim
            best_id = c_id
            best_label = c_label
            best_source = _CUAD_SOURCE
    for c_id, c_label, c_tokens in _SUPPLEMENTAL_TOKEN_MAP:
        sim = _jaccard(tokens, c_tokens)
        if sim > best_sim:
            best_sim = sim
            best_id = c_id
            best_label = c_label
            best_source = _SUPPLEMENTAL_SOURCE
    return best_id, best_label, best_sim, best_source


def _to_entry_id(heading: str) -> str:
    """Convert an original heading to a snake_case entry ID (max 60 chars)."""
    norm = _normalize_heading(heading)
    entry_id = re.sub(r"\s+", "_", norm)
    entry_id = re.sub(r"[^\w]", "", entry_id)
    entry_id = re.sub(r"_+", "_", entry_id).strip("_")
    return (entry_id or "entry")[:60]


def _unique_entry_id(base: str, seen: set[str]) -> str:
    """Return *base* with a suffix counter if already present in *seen*."""
    if base not in seen:
        return base
    for n in range(2, 1000):
        candidate = f"{base}_{n}"
        if candidate not in seen:
            return candidate
    return f"{base}_{len(seen)}"  # fallback, practically unreachable


def _pick_examples(
    occurrences: list[tuple[str, str, str, str]],
) -> tuple[ClusterExample, ...]:
    """Pick up to _MAX_EXAMPLES_PER_ENTRY examples, one per distinct document."""
    seen_docs: set[str] = set()
    examples: list[ClusterExample] = []
    for doc_id, clause_path, heading, snippet in occurrences:
        if doc_id in seen_docs:
            continue
        examples.append(
            ClusterExample(
                document_id=doc_id,
                clause_path=clause_path,
                heading=heading,
                text_snippet=snippet,
            )
        )
        seen_docs.add(doc_id)
        if len(examples) >= _MAX_EXAMPLES_PER_ENTRY:
            break
    return tuple(examples)
