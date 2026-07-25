"""Citation resolver — the reference implementation of OPF §4 resolution
(issue #185, format half of #117).

Given a playbook, a clause id, an observation index, and a directory holding
the corpus source files, resolve the observation's ``example_ref`` to the
actual cited file and verify its bytes by content address:

1. Read the citation ``(document_id, version, clause_path, char_span)``.
2. Look up ``corpus.documents[document_id].version_files`` and take the
   entry whose ``version`` matches — its ``sha256`` names the exact bytes
   the compiler read.
3. Find a file under ``corpus_dir`` whose sha256 matches (the hash IS the
   key — filenames/layout are the consumer's business): the document's own
   subdirectory ``corpus_dir/<document_id>/`` is searched first, then the
   whole tree.
4. Return the file path plus ``clause_path``/``char_span`` so the consumer
   can open the cited clause; raise ``CitationResolutionError`` on a hash
   mismatch or a missing/unaddressable citation.

Consumers copy this algorithm; ``playbook resolve-citation`` (cli.py) is
its command-line face.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from playbook_engine.canonicalize import file_sha256
from playbook_engine.opf_accessors import playbook_clauses


class CitationResolutionError(ValueError):
    """A citation could not be resolved to verified bytes."""


@dataclass(frozen=True)
class ResolvedCitation:
    """A citation resolved to a verified source file."""

    document_id: str
    version: int | str
    sha256: str
    file_path: Path
    clause_path: str | None
    char_span: tuple[int, int] | None

    def describe(self) -> str:
        span = f" chars [{self.char_span[0]}, {self.char_span[1]}]" if self.char_span else ""
        clause = f" clause {self.clause_path}" if self.clause_path else ""
        return (
            f"{self.document_id} v{self.version} -> {self.file_path}"
            f"{clause}{span} (verified {self.sha256})"
        )


def _find_by_hash(corpus_dir: Path, document_id: str, expected_sha256: str) -> Path | None:
    """Locate a file whose bytes hash to *expected_sha256*.

    The document's own subdirectory is the fast path (the engine's staging
    layout); the whole tree is the fallback, since a consumer may hold the
    corpus in any arrangement — the hash, not the layout, is the address.
    """
    doc_dir = corpus_dir / document_id
    search_roots = [doc_dir] if doc_dir.is_dir() else []
    search_roots.append(corpus_dir)
    seen: set[Path] = set()
    for root in search_roots:
        for candidate in sorted(p for p in root.rglob("*") if p.is_file()):
            if candidate in seen:
                continue
            seen.add(candidate)
            if file_sha256(candidate) == expected_sha256:
                return candidate
    return None


def resolve_citation(
    playbook: dict[str, Any],
    clause_id: str,
    obs_index: int,
    corpus_dir: Path,
) -> ResolvedCitation:
    """Resolve one observation's citation to a hash-verified source file.

    Args:
        playbook:   Parsed playbook document (OPF v0.2).
        clause_id:  ``evidence.clauses[].id`` (e.g. ``"clause.indemnification"``).
        obs_index:  Index into that clause's ``observed_positions``.
        corpus_dir: Directory holding the corpus source files.

    Raises:
        CitationResolutionError: unknown clause/observation, a citation with
            no ``version_files`` content address, no file with the expected
            hash under *corpus_dir* (message contains ``"hash mismatch"``),
            or a missing corpus directory.
    """
    clause = next((c for c in playbook_clauses(playbook) if c.get("id") == clause_id), None)
    if clause is None:
        known = ", ".join(c.get("id", "?") for c in playbook_clauses(playbook))
        raise CitationResolutionError(f"no clause with id {clause_id!r} (known: {known})")

    observations = clause.get("observed_positions", [])
    if not 0 <= obs_index < len(observations):
        raise CitationResolutionError(
            f"observation index {obs_index} out of range for {clause_id!r} "
            f"({len(observations)} observation(s))"
        )
    ref = observations[obs_index].get("example_ref") or {}
    document_id = ref.get("document_id", "")
    version: int | str | None = ref.get("version")
    if not isinstance(version, (int, str)):
        raise CitationResolutionError(
            f"citation on {clause_id!r} obs {obs_index} carries no version — unresolvable"
        )

    if document_id == "template":
        expected = (playbook.get("baseline", {}).get("template_ref") or {}).get("sha256")
        if not expected:
            raise CitationResolutionError(
                "citation cites the template but baseline.template_ref carries no sha256 "
                "content address — unverifiable"
            )
    else:
        corpus_doc = next(
            (
                d
                for d in playbook.get("corpus", {}).get("documents", [])
                if d.get("document_id") == document_id
            ),
            None,
        )
        if corpus_doc is None:
            raise CitationResolutionError(f"citation cites unknown document {document_id!r}")
        entry = next(
            (vf for vf in corpus_doc.get("version_files", []) if vf.get("version") == version),
            None,
        )
        if entry is None:
            raise CitationResolutionError(
                f"{document_id!r} v{version} has no version_files content address — "
                "this playbook predates §4 resolution or the version was never mined"
            )
        expected = entry.get("sha256")
        if not expected:
            # A hand-edited or foreign playbook can carry a version_files
            # entry without the digest; that must surface as the documented
            # error type, not a KeyError escaping the contract.
            raise CitationResolutionError(
                f"{document_id!r} v{version}'s version_files entry carries no sha256 — unverifiable"
            )

    if not corpus_dir.is_dir():
        raise CitationResolutionError(f"corpus directory not found: {corpus_dir}")

    found = _find_by_hash(corpus_dir, document_id, expected)
    if found is None:
        raise CitationResolutionError(
            f"hash mismatch: no file under {corpus_dir} matches {expected} "
            f"for {document_id!r} v{version} — the corpus copy differs from "
            "the one this playbook was compiled from"
        )

    char_span = ref.get("char_span")
    return ResolvedCitation(
        document_id=document_id,
        version=version,
        sha256=expected,
        file_path=found,
        clause_path=ref.get("clause_path"),
        char_span=tuple(char_span) if char_span else None,
    )
