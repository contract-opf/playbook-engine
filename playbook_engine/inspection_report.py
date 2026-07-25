"""Human-readable inspection report for trail/ and observations.jsonl.

Lets a lawyer verify the engine's structural inferences — version ordering,
signed-copy identification, provenance, and per-clause deviations — before
trusting the compiled playbook.

Usage::

    from playbook_engine.inspection_report import build_inspection_report
    report_md = build_inspection_report(out_dir)
    Path("report.md").write_text(report_md, encoding="utf-8")
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from playbook_engine.clause_position_compiler import CoherenceFlag
from playbook_engine.observation_builder import read_observations_jsonl

_log = logging.getLogger(__name__)


def render_review_flags(flags: list[dict[str, object]]) -> str:
    """Render a Markdown "Needs attention" section for review flags loaded from
    ``review.json`` (P3.4 artifact).

    Args:
        flags: List of flag dicts as stored in ``review.json`` (each dict has
               keys ``document_id``, ``severity``, ``kind``, ``detail``, and
               ``suggested_action``).  May be empty.

    Returns:
        Markdown string (empty when *flags* is empty).
    """
    if not flags:
        return ""

    lines: list[str] = []
    lines.append("## Needs attention")
    lines.append("")
    lines.append(
        "> The following issues were detected during automated review.  "
        "Resolve before publishing the playbook."
    )
    lines.append("")
    lines.append("| Document | Severity | Kind | Suggested action |")
    lines.append("|----------|----------|------|-----------------|")
    for flag in flags:
        doc = flag.get("document_id") or "*(corpus)*"
        severity = flag.get("severity", "?")
        severity_str = f"**{severity}**" if severity == "block" else str(severity)
        kind = flag.get("kind", "?")
        action = _md_escape(str(flag.get("suggested_action", "")))
        lines.append(f"| {doc} | {severity_str} | `{kind}` | {action} |")
    lines.append("")
    return "\n".join(lines)


def render_floor_candidates(candidates: list[dict[str, Any]]) -> str:
    """Render a Markdown "Floor candidates" section from ``floor.candidates.json``
    (issue #166).

    These are PROPOSALS only — derived from ``proposed_then_reversed``
    observations and the Posture interview's Q4 answer, never auto-promoted
    into the OPF ``floor.invariants``. A legal owner accepts a candidate by
    editing ``floor.invariants`` directly (or via the curation CLI).

    Args:
        candidates: The ``candidates`` list from ``floor.candidates.json``.
                    May be empty.

    Returns:
        Markdown string (empty when *candidates* is empty).
    """
    if not candidates:
        return ""

    lines: list[str] = []
    lines.append("## Floor candidates (proposed — not yet accepted)")
    lines.append("")
    lines.append(
        "> Derived from reversed proposals and the Posture interview's Q4 answer "
        "(OPF §3.7 rule 4). These are proposals for the legal owner, NOT part of "
        "the signed OPF Floor. Accept a candidate by editing `floor.invariants` "
        "directly (or via the curation CLI)."
    )
    lines.append("")
    lines.append("| ID | Source | Statement | Rationale | Citations |")
    lines.append("|----|--------|-----------|-----------|-----------|")
    for c in candidates:
        citations = c.get("citations") or []
        cite_str = (
            "; ".join(
                f"{cit.get('document_id')} v{cit.get('version')} §{cit.get('clause_path')}"
                for cit in citations
            )
            if citations
            else "*(none)*"
        )
        lines.append(
            f"| `{c.get('id', '?')}` | {c.get('source', '?')} | "
            f"{_md_escape(str(c.get('statement', '')))} | "
            f"{_md_escape(str(c.get('rationale', '')))} | {cite_str} |"
        )
    lines.append("")
    return "\n".join(lines)


def render_coherence_flags(coherence_flags: list[CoherenceFlag]) -> str:
    """Render a Markdown section for CoherenceFlag entries.

    Args:
        coherence_flags: Flags emitted by ``CoherenceJudge`` for unreliable
                         clause positions.  May be empty.

    Returns:
        Markdown string (may be empty if there are no flags).
    """
    if not coherence_flags:
        return ""

    lines: list[str] = []
    lines.append("## Coherence Flags")
    lines.append("")
    lines.append(
        "> The following clauses were flagged by the coherence reviewer as potentially "
        "unreliable.  Review before publishing the playbook."
    )
    lines.append("")
    lines.append("| Clause ID | Severity | Reason |")
    lines.append("|-----------|----------|--------|")
    for flag in coherence_flags:
        severity_marker = "**block**" if flag.severity == "block" else "warn"
        lines.append(f"| `{flag.clause_id}` | {severity_marker} | {_md_escape(flag.reason)} |")
    lines.append("")
    return "\n".join(lines)


def build_inspection_report(
    out_dir: Path,
    coherence_flags: list[CoherenceFlag] | None = None,
) -> str:
    """Build a Markdown inspection report from a compiled output directory.

    Reads ``scope.json``, ``trail/*.json``, and ``observations.jsonl`` and
    returns a Markdown string reviewable without opening the raw JSON.
    Out-of-scope documents are included with their exclusion rationale so a
    reviewer can verify every scope decision.

    Args:
        out_dir:          Path to the ``out/`` directory produced by
                          ``playbook compile``.
        coherence_flags:  Optional list of ``CoherenceFlag`` entries emitted
                          by ``CoherenceJudge``.  When provided and non-empty,
                          a dedicated section is prepended to the report.

    Returns:
        Markdown-formatted report string.

    Raises:
        FileNotFoundError: If the output directory does not exist.
    """
    if not out_dir.exists():
        raise FileNotFoundError(f"Output directory not found: {out_dir}")

    scope = _load_scope(out_dir)
    trails = _load_trails(out_dir)
    observations_by_doc = _load_observations(out_dir)
    manifest = _load_manifest(out_dir)
    # Failed per-version ingests (issue #89) are read straight from
    # corpus_manifest.json — unlike the rest of ``review_flags`` these don't
    # require a separate ``playbook review`` run to have populated
    # review.json first, so a failed version always shows up here. Dedupe
    # against review.json's own version_ingest_failed entries (if a
    # ``playbook review`` run already wrote them) so a failure that appears
    # in both sources renders as one row, not two.
    review_flags = _dedupe_flags(
        _version_ingest_review_flags(manifest) + _load_review_flags(out_dir)
    )

    # B1: include all doc IDs — in-scope (have a trail) AND out-of-scope (scope-only)
    all_doc_ids = sorted(set(trails) | set(scope))

    lines: list[str] = []

    # Header
    lines.append("# Playbook Inspection Report")
    lines.append("")
    lines.append(f"**Output directory:** `{out_dir}`")

    # A2: use scope for total count; trails only cover in-scope docs
    total_docs = len(scope) if scope else len(trails)
    in_scope = sum(1 for d in scope.values() if d.get("in_scope", False))
    lines.append(f"**Documents:** {in_scope} in scope / {total_docs} total")
    total_obs = sum(len(v) for v in observations_by_doc.values())
    lines.append(f"**Observations:** {total_obs} total")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(
        "> Review **Version order**, **Signed copy**, and **Provenance** for each document "
        "before trusting the playbook. If any inference is wrong, add a `hints.yaml` to "
        "the document folder and re-run with `--no-cache`."
    )
    lines.append("")

    # "Needs attention" section — review flags from review.json (P3.4)
    if review_flags:
        needs_attention = render_review_flags(review_flags)
        if needs_attention:
            lines.append(needs_attention)

    # Coherence flags section (prepended before per-document sections)
    if coherence_flags:
        flags_section = render_coherence_flags(coherence_flags)
        if flags_section:
            lines.append(flags_section)

    # Floor candidates section (issue #166) — rendered when 'playbook floor
    # propose' has been run and wrote floor.candidates.json to out_dir.
    floor_candidates = _load_floor_candidates(out_dir)
    if floor_candidates:
        candidates_section = render_floor_candidates(floor_candidates)
        if candidates_section:
            lines.append(candidates_section)

    for doc_id in all_doc_ids:
        trail = trails.get(doc_id, {})
        doc_scope = scope.get(doc_id, {})
        obs_list = observations_by_doc.get(doc_id, [])
        lines.extend(_render_document(doc_id, trail, doc_scope, obs_list))
        lines.append("")

    return "\n".join(lines)


def write_inspection_report(
    out_dir: Path,
    report_path: Path,
    coherence_flags: list[CoherenceFlag] | None = None,
) -> None:
    """Build and write the inspection report to a file.

    Args:
        out_dir:          Pipeline output directory.
        report_path:      Destination path for the Markdown report.
        coherence_flags:  Optional list of ``CoherenceFlag`` entries to render.
    """
    report = build_inspection_report(out_dir, coherence_flags=coherence_flags)
    tmp = report_path.with_suffix(".tmp")
    tmp.write_text(report, encoding="utf-8")
    tmp.replace(report_path)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_review_flags(out_dir: Path) -> list[dict[str, Any]]:
    """Return the list of flag dicts from ``review.json`` (P3.4 artifact).

    Returns an empty list when the file is absent or unparseable (back-compat).
    """
    review_path = out_dir / "review.json"
    if not review_path.exists():
        return []
    try:
        data: dict[str, Any] = json.loads(review_path.read_text(encoding="utf-8"))
        flags = data.get("flags", [])
        return flags if isinstance(flags, list) else []
    except Exception:  # noqa: BLE001
        _log.warning("Could not parse review.json; needs-attention section will be omitted.")
        return []


def _load_floor_candidates(out_dir: Path) -> list[dict[str, Any]]:
    """Return the ``candidates`` list from ``floor.candidates.json`` (issue #166).

    Returns an empty list when the file is absent (no ``playbook floor
    propose`` run yet) or unparseable.
    """
    candidates_path = out_dir / "floor.candidates.json"
    if not candidates_path.exists():
        return []
    try:
        data = json.loads(candidates_path.read_text(encoding="utf-8"))
        candidates = data.get("candidates", [])
        return candidates if isinstance(candidates, list) else []
    except Exception:  # noqa: BLE001
        _log.warning("Could not parse floor.candidates.json; Floor candidates will be omitted.")
        return []


def _load_scope(out_dir: Path) -> dict[str, Any]:
    """Return {doc_id: scope_decision} from scope.json (or empty dict if absent)."""
    scope_path = out_dir / "scope.json"
    if not scope_path.exists():
        return {}
    raw = json.loads(scope_path.read_text(encoding="utf-8"))
    return {d["document_id"]: d for d in raw.get("documents", [])}


def _load_manifest(out_dir: Path) -> dict[str, Any]:
    """Return {doc_id: corpus_doc_dict} from corpus_manifest.json (or {} if absent).

    Used to surface per-version ingest status (``version_ingest`` — issue #89)
    without every caller re-parsing the manifest file itself.
    """
    manifest_path = out_dir / "corpus_manifest.json"
    if not manifest_path.exists():
        return {}
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        _log.warning("Could not parse corpus_manifest.json; version-ingest status will be omitted.")
        return {}
    if not isinstance(raw, list):
        return {}
    return {
        d["document_id"]: d for d in raw if isinstance(d, dict) and d.get("document_id") is not None
    }


def _dedupe_flags(flags: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop exact (document_id, kind, suggested_action) duplicates, preserving order."""
    seen: set[tuple[Any, Any, Any]] = set()
    deduped: list[dict[str, Any]] = []
    for flag in flags:
        key = (flag.get("document_id"), flag.get("kind"), flag.get("suggested_action"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(flag)
    return deduped


def _version_ingest_review_flags(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Build "Needs attention"-shaped flag dicts for failed per-version ingests.

    Mirrors :func:`playbook_engine.review._check_manifest` but reads
    ``corpus_manifest.json`` directly (via *manifest*) so the inspection
    report surfaces a failed version even when ``playbook review`` was never
    run to populate ``review.json`` (issue #89).
    """
    flags: list[dict[str, Any]] = []
    for doc_id in sorted(manifest):
        for ver in manifest[doc_id].get("version_ingest", []) or []:
            if not isinstance(ver, dict) or ver.get("status") != "failed":
                continue
            version = ver.get("version", "?")
            error = ver.get("error") or "unknown error"
            flags.append(
                {
                    "document_id": doc_id,
                    "severity": "warn",
                    "kind": "version_ingest_failed",
                    "suggested_action": (
                        f"Version {version!r} failed to ingest and was never mined: {error}. "
                        "Inspect the source file and re-run 'playbook mine' with --no-cache."
                    ),
                }
            )
    return flags


def _load_trails(out_dir: Path) -> dict[str, Any]:
    """Return {doc_id: trail_dict} from trail/*.json files."""
    trail_dir = out_dir / "trail"
    if not trail_dir.exists():
        return {}
    result: dict[str, Any] = {}
    for p in sorted(trail_dir.glob("*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        doc_id = data.get("document_id", p.stem)
        result[doc_id] = data
    return result


def _load_observations(out_dir: Path) -> dict[str, list[dict[str, Any]]]:
    """Return {doc_id: [observation_dict, ...]} from observations.jsonl.

    Malformed lines are skipped with a warning rather than crashing.
    """
    obs_path = out_dir / "observations.jsonl"
    if not obs_path.exists():
        return {}
    try:
        raw_list = read_observations_jsonl(obs_path)
    except Exception:  # noqa: BLE001
        _log.warning("Could not parse observations.jsonl; observations will be omitted.")
        return {}
    result: dict[str, list[dict[str, Any]]] = {}
    for obs in raw_list:
        citation = obs.get("citation", {}) if isinstance(obs, dict) else {}
        doc_id = citation.get("document_id", "unknown") if isinstance(citation, dict) else "unknown"
        result.setdefault(doc_id, []).append(obs)
    return result


def _render_document(
    doc_id: str,
    trail: dict[str, Any],
    doc_scope: dict[str, Any],
    obs_list: list[dict[str, Any]],
) -> list[str]:
    lines: list[str] = []
    lines.append(f"## {doc_id}")
    lines.append("")

    # Structural inferences table
    ordered = trail.get("ordered_versions", [])
    signed = trail.get("signed_version") or "*(not detected)*"
    provenance = trail.get("provenance", "unknown")
    basis = trail.get("basis", "")
    in_scope = doc_scope.get("in_scope")
    # Confidence fields (issue #59): surface these so reviewers can verify
    # them without opening the raw trail JSON.
    shape: str | None = trail.get("shape")
    prov_conf: float | None = trail.get("provenance_confidence")
    prov_is_ambiguous: bool | None = trail.get("provenance_is_ambiguous")
    signed_conf: float | None = trail.get("signed_copy_confidence")

    scope_marker = "✓ in scope" if in_scope else ("✗ out of scope" if in_scope is False else "—")
    lines.append("| Field | Value |")
    lines.append("|-------|-------|")
    lines.append(f"| Scope | {scope_marker} |")
    if provenance != "unknown" or trail:
        lines.append(f"| Provenance | `{provenance}` |")
    if prov_conf is not None:
        ambig_str = " *(ambiguous)*" if prov_is_ambiguous else ""
        lines.append(f"| Provenance confidence | {prov_conf:.2f}{ambig_str} |")
    elif prov_is_ambiguous:
        lines.append("| Provenance confidence | *(ambiguous)* |")
    if ordered:
        lines.append(f"| Version order | {' → '.join(str(v) for v in ordered)} |")
    if shape:
        lines.append(f"| Chain shape | `{shape}` |")
    if trail.get("signed_version"):
        lines.append(f"| Signed copy | `{signed}` |")
    if signed_conf is not None:
        lines.append(f"| Signed copy confidence | {signed_conf:.2f} |")
    if basis:
        lines.append(f"| Order basis | {basis} |")
    if doc_scope.get("scope_rationale"):
        rationale = doc_scope["scope_rationale"]
        conf = doc_scope.get("scope_confidence", "")
        conf_str = f" *(confidence: {conf})*" if conf != "" else ""
        lines.append(f"| Scope rationale | {rationale}{conf_str} |")

    lines.append("")

    if in_scope is False:
        lines.append("*Document excluded from compilation — no clause observations.*")
        return lines

    if not obs_list:
        lines.append("*No observations for this document.*")
        return lines

    # Group observations by taxonomy_id for readability
    by_tid: dict[str | None, list[dict[str, Any]]] = {}
    for obs in obs_list:
        tid = obs.get("taxonomy_id") if isinstance(obs, dict) else None
        by_tid.setdefault(tid, []).append(obs)

    classified = [(tid, obs) for tid, obs in by_tid.items() if tid is not None]
    unclassified = by_tid.get(None, [])

    total = len(obs_list)
    lines.append(f"### Clause Observations ({total})")
    lines.append("")

    if classified:
        # A4: include Version column for traceability
        lines.append("| Taxonomy ID | Version | Text | Deviation | Risk | Outcome |")
        lines.append("|-------------|---------|------|-----------|------|---------|")
        for tid, obs_group in sorted(classified, key=lambda x: x[0] or ""):
            for obs in obs_group:
                version = obs.get("citation", {}).get("version", "?")
                text = _truncate(obs.get("text_summary", ""), 80)
                deviation = obs.get("deviation", "?")
                risk = obs.get("risk_delta", {}) if isinstance(obs.get("risk_delta"), dict) else {}
                risk_dir = risk.get("direction", "?")
                risk_mag = risk.get("magnitude", "?")
                outcome = obs.get("outcome", "?")
                lines.append(
                    f"| `{tid}` | {version} | {_md_escape(text)} | {deviation} "
                    f"| {risk_dir} / {risk_mag} | {outcome} |"
                )

    if unclassified:
        lines.append("")
        lines.append(f"**Unclassified clauses ({len(unclassified)})** — taxonomy_id not matched")
        lines.append("")
        lines.append("| Version | Text | Deviation | Risk | Outcome |")
        lines.append("|---------|------|-----------|------|---------|")
        for obs in unclassified:
            version = obs.get("citation", {}).get("version", "?") if isinstance(obs, dict) else "?"
            text = _truncate(obs.get("text_summary", ""), 80) if isinstance(obs, dict) else ""
            deviation = obs.get("deviation", "?") if isinstance(obs, dict) else "?"
            risk = (
                obs.get("risk_delta", {})
                if isinstance(obs, dict) and isinstance(obs.get("risk_delta"), dict)
                else {}
            )
            risk_dir = risk.get("direction", "?")
            risk_mag = risk.get("magnitude", "?")
            outcome = obs.get("outcome", "?") if isinstance(obs, dict) else "?"
            lines.append(
                f"| {version} | {_md_escape(text)} | {deviation} | {risk_dir} / {risk_mag} | {outcome} |"
            )

    return lines


def _truncate(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


def _md_escape(text: str) -> str:
    """Escape pipe characters so they don't break Markdown tables."""
    return text.replace("|", "\\|")
