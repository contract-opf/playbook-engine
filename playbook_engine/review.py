"""Checkpoint-review module: artifacts → ReviewFlags (issues #57 and #59).

Reads the pipeline's intermediate artifacts from an output directory and emits
structured review flags — the engine half of the orchestrator's checklist.
Provides :func:`write_review` to persist the flags as a machine-readable
``review.json`` artifact (issue #59).

Pure and deterministic — no LLM calls, no corpus access.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_AMBIGUITY_THRESHOLD: float = 0.70
"""Confidence below this level triggers a provenance/signed-copy warning.

Mirrors ``signed_detector.AMBIGUITY_THRESHOLD`` and
``provenance_detector.AMBIGUITY_THRESHOLD`` — kept local so this module has
no runtime import dependency on either detector.
"""

# ---------------------------------------------------------------------------
# ReviewFlag
# ---------------------------------------------------------------------------

_VALID_SEVERITIES = frozenset({"info", "warn", "block"})


@dataclass
class ReviewFlag:
    """A single review flag emitted by :func:`review_out_dir`.

    Attributes:
        document_id:      Source document identifier, or ``None`` for corpus-level flags.
        stage:            Pipeline stage that produced the issue (e.g. ``"trail"``,
                          ``"scope"``, ``"observation"``).
        kind:             Machine-readable flag kind (slug, e.g. ``"ambiguous_version_chain"``).
        severity:         One of ``"info"``, ``"warn"``, or ``"block"``.
        detail:           Human-readable description of the issue.
        suggested_action: Recommended next step for the reviewer.
    """

    document_id: str | None
    stage: str
    kind: str
    severity: str
    detail: str
    suggested_action: str

    def __post_init__(self) -> None:
        if self.severity not in _VALID_SEVERITIES:
            raise ValueError(
                f"ReviewFlag.severity must be one of {sorted(_VALID_SEVERITIES)!r}; "
                f"got {self.severity!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "stage": self.stage,
            "kind": self.kind,
            "severity": self.severity,
            "detail": self.detail,
            "suggested_action": self.suggested_action,
        }


# ---------------------------------------------------------------------------
# Core review logic
# ---------------------------------------------------------------------------


def review_out_dir(out_dir: Path) -> list[ReviewFlag]:
    """Read pipeline artifacts from *out_dir* and return all review flags.

    Reads:
      - ``scope.json``              — scope-gate decisions.
      - ``trail/<doc_id>.json``     — version-order and provenance signals per document.
      - ``observations.jsonl``      — per-clause observation records.
      - ``corpus_manifest.json``    — per-document metadata, including per-version
        ingest status (issue #89).

    Missing files are silently skipped (the module is lenient so it can be
    called on partial outputs).

    Returns:
        A list of :class:`ReviewFlag` objects, empty when no issues are found.
    """
    flags: list[ReviewFlag] = []
    flags.extend(_check_scope(out_dir))
    flags.extend(_check_trails(out_dir))
    flags.extend(_check_observations(out_dir))
    flags.extend(_check_manifest(out_dir))
    return flags


def write_review(out_dir: Path) -> Path:
    """Run the P3.3 checklist, fold in ``coherence_flags.json``, and write
    ``out_dir/review.json``.

    This is the P3.4 machine-readable artifact.  It round-trips: the flags
    list can be reconstructed from the JSON via :class:`ReviewFlag`.

    Reads:
      - All artifacts consumed by :func:`review_out_dir`.
      - ``coherence_flags.json`` (P3.2 artifact) — CoherenceJudge flags;
        each entry is incorporated as a ``ReviewFlag`` with
        ``stage="coherence"``.  Absent file is silently skipped.

    Args:
        out_dir: Path to the pipeline output directory.

    Returns:
        Path of the written ``review.json`` file.

    Raises:
        FileNotFoundError: If *out_dir* does not exist.
    """
    if not out_dir.exists():
        raise FileNotFoundError(f"Output directory not found: {out_dir}")

    flags: list[ReviewFlag] = review_out_dir(out_dir)
    flags.extend(_check_coherence_flags(out_dir))

    payload: dict[str, Any] = {"flags": [f.to_dict() for f in flags]}
    review_path = out_dir / "review.json"
    tmp = review_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(review_path)
    return review_path


# ---------------------------------------------------------------------------
# Per-artifact checkers
# ---------------------------------------------------------------------------


def _check_scope(out_dir: Path) -> list[ReviewFlag]:
    """Emit ``block`` flags for documents whose scope judge failed."""
    scope_path = out_dir / "scope.json"
    if not scope_path.exists():
        return []

    data: dict[str, Any] = json.loads(scope_path.read_text(encoding="utf-8"))
    flags: list[ReviewFlag] = []

    for doc in data.get("documents", []):
        if doc.get("basis") == "judge_error":
            flags.append(
                ReviewFlag(
                    document_id=doc.get("document_id"),
                    stage="scope",
                    kind="scope_judge_failed",
                    severity="block",
                    detail="Scope judge raised an error; document was retained but its scope is unverified.",
                    suggested_action="Re-run with a working scope judge or manually verify the document's scope.",
                )
            )

    return flags


def _check_trails(out_dir: Path) -> list[ReviewFlag]:
    """Emit ``warn`` flags for version-chain and provenance issues in trail files."""
    trail_dir = out_dir / "trail"
    if not trail_dir.exists():
        return []

    flags: list[ReviewFlag] = []
    for trail_file in sorted(trail_dir.glob("*.json")):
        trail: dict[str, Any] = json.loads(trail_file.read_text(encoding="utf-8"))
        doc_id: str | None = trail.get("document_id")
        flags.extend(_check_single_trail(doc_id, trail))

    return flags


def _check_single_trail(doc_id: str | None, trail: dict[str, Any]) -> list[ReviewFlag]:
    flags: list[ReviewFlag] = []

    basis: str | None = trail.get("basis")
    shape: str | None = trail.get("shape")
    signed_conf: float | None = trail.get("signed_copy_confidence")
    prov_is_ambiguous: bool = bool(trail.get("provenance_is_ambiguous", False))
    prov_conf: float | None = trail.get("provenance_confidence")

    # Rule: ambiguous version chain
    if basis in {"greedy", "llm"}:
        flags.append(
            ReviewFlag(
                document_id=doc_id,
                stage="trail",
                kind="ambiguous_version_chain",
                severity="warn",
                detail=f"Version chain ordering used basis={basis!r}; result may not be reliable.",
                suggested_action="Review version ordering manually or supply explicit ordering hints.",
            )
        )

    # Rule: fork / missing draft
    if shape in {"fork", "gap"}:
        flags.append(
            ReviewFlag(
                document_id=doc_id,
                stage="trail",
                kind="fork_or_missing_draft",
                severity="warn",
                detail=f"Version chain shape={shape!r} suggests a fork or missing intermediate draft.",
                suggested_action="Locate the missing document version(s) and re-compile.",
            )
        )

    # Rule: weak signed anchor
    if signed_conf is not None and signed_conf < _AMBIGUITY_THRESHOLD:
        flags.append(
            ReviewFlag(
                document_id=doc_id,
                stage="trail",
                kind="weak_signed_anchor",
                severity="warn",
                detail=(
                    f"Signed-copy confidence {signed_conf:.2f} is below the "
                    f"{_AMBIGUITY_THRESHOLD:.2f} threshold."
                ),
                suggested_action="Manually verify which version is the signed counterpart.",
            )
        )

    # Rule: unreliable provenance
    if prov_is_ambiguous or (prov_conf is not None and prov_conf < _AMBIGUITY_THRESHOLD):
        flags.append(
            ReviewFlag(
                document_id=doc_id,
                stage="trail",
                kind="unreliable_provenance",
                severity="warn",
                detail=(
                    "Provenance detection is ambiguous"
                    + (f" (confidence={prov_conf:.2f})" if prov_conf is not None else "")
                    + "."
                ),
                suggested_action="Confirm which party drafted this agreement before using it for playbook positioning.",
            )
        )

    return flags


def _check_observations(out_dir: Path) -> list[ReviewFlag]:
    """Emit ``warn`` flags for judge-error or needs-review observations."""
    obs_path = out_dir / "observations.jsonl"
    if not obs_path.exists():
        return []

    flags: list[ReviewFlag] = []
    for line in obs_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obs: dict[str, Any] = json.loads(line)
        doc_id: str | None = obs.get("citation", {}).get("document_id")
        basis: str | None = obs.get("basis")
        deviation: str | None = obs.get("deviation")

        if basis == "judge_error" or deviation == "needs_review":
            flags.append(
                ReviewFlag(
                    document_id=doc_id,
                    stage="observation",
                    kind="deviation_needs_review",
                    severity="warn",
                    detail=(
                        f"Observation {obs.get('observation_id', '?')!r} requires review "
                        f"(basis={basis!r}, deviation={deviation!r})."
                    ),
                    suggested_action="Manually classify this clause deviation before publishing the playbook.",
                )
            )

    return flags


def _check_manifest(out_dir: Path) -> list[ReviewFlag]:
    """Emit ``warn`` flags for versions whose ingest failed (issue #89).

    ``corpus_manifest.json["version_ingest"]`` records "ok"/"failed" for every
    version file a document folder contains — a version whose extraction or
    segmentation raised previously surfaced only as a scrolled-past progress-
    line WARNING and then vanished (a cache hit on a later run didn't even
    re-print that). Turning each failed entry into a durable ``ReviewFlag``
    means a negotiation version that was never actually read shows up in the
    reviewer's checklist, not just stdout.
    """
    manifest_path = out_dir / "corpus_manifest.json"
    if not manifest_path.exists():
        return []

    try:
        docs: list[dict[str, Any]] = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return []

    if not isinstance(docs, list):
        return []

    flags: list[ReviewFlag] = []
    for doc in docs:
        if not isinstance(doc, dict):
            continue
        doc_id = doc.get("document_id")
        for ver in doc.get("version_ingest", []) or []:
            if not isinstance(ver, dict) or ver.get("status") != "failed":
                continue
            version = ver.get("version", "?")
            error = ver.get("error") or "unknown error"
            flags.append(
                ReviewFlag(
                    document_id=doc_id,
                    stage="ingest",
                    kind="version_ingest_failed",
                    severity="warn",
                    detail=f"Version {version!r} failed to ingest and was never mined: {error}",
                    suggested_action=(
                        f"Version {version!r} failed to ingest and was never mined: {error}. "
                        "Inspect the source file and re-run 'playbook mine' with --no-cache."
                    ),
                )
            )

    return flags


def _check_coherence_flags(out_dir: Path) -> list[ReviewFlag]:
    """Convert ``coherence_flags.json`` (P3.2) entries into :class:`ReviewFlag` objects.

    Each CoherenceFlag dict has keys ``clause_id``, ``reason``, and ``severity``.
    They are mapped to :class:`ReviewFlag` with ``stage="coherence"`` and
    ``kind="low_coherence"`` so the orchestrator can treat them uniformly.

    Missing ``coherence_flags.json`` is silently skipped.
    """
    cf_path = out_dir / "coherence_flags.json"
    if not cf_path.exists():
        return []

    try:
        entries: list[dict[str, Any]] = json.loads(cf_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return []

    if not isinstance(entries, list):
        return []

    flags: list[ReviewFlag] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        clause_id: str = entry.get("clause_id", "unknown")
        reason: str = entry.get("reason", "")
        severity: str = entry.get("severity", "warn")
        if severity not in _VALID_SEVERITIES:
            severity = "warn"
        flags.append(
            ReviewFlag(
                document_id=None,
                stage="coherence",
                kind="low_coherence",
                severity=severity,
                detail=f"Clause {clause_id!r} flagged by CoherenceJudge: {reason}",
                suggested_action="Review clause position reliability before publishing the playbook.",
            )
        )
    return flags
