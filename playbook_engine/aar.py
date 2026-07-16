"""After-action report (AAR) — post-compilation playbook health summary.

Renders a strong after-action report from the mined artifacts plus the
compiled playbook, covering:

  1. **Corpus coverage** — agreements/versions, in-scope vs out-of-scope.
  2. **Backbone health** — trail ordering, signed copies, reversals,
     version-order basis.
  3. **Judgment economics** — unique items judged, dedup/cache ratio,
     rough token estimate from ``<out>/judge/*``.
  4. **Semantic coverage** — % clauses classified, deviation distribution,
     provenance distribution, rollup-position histogram.
  5. **Needs attention** — low-confidence/``needs_review``/``judge_error``/
     ambiguous-provenance items, each with its item number.
  6. **Honesty** — blank/defaulted fields enumerated; human-input-required
     v0.2 items (GC-authored Posture, Floor from classified reversals).

Usage::

    from playbook_engine.aar import build_after_action_report
    md = build_after_action_report(out_dir)

    from playbook_engine.aar import write_after_action_report
    write_after_action_report(out_dir, Path("report.md"))  # also writes report.json
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from playbook_engine.inspection_report import (
    _load_manifest,
    _load_observations,
    _load_scope,
    _load_trails,
)
from playbook_engine.opf_accessors import clause_confidence, clause_stance, playbook_clauses

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Token estimate constants
# ---------------------------------------------------------------------------

_AVG_TOKENS_PER_ITEM = 200


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_after_action_data(out_dir: Path) -> dict[str, Any]:
    """Build the structured data dict for the after-action report.

    This is the canonical representation; ``build_after_action_report``
    renders it to Markdown.  Build the dict first so Markdown and JSON never
    drift.

    Args:
        out_dir: Path to the ``out/`` directory produced by ``playbook compile``
                 (or ``playbook mine`` + ``playbook project``).

    Returns:
        Nested dict with sections: corpus_coverage, backbone_health,
        judgment_economics, semantic_coverage, needs_attention, honesty.

    Raises:
        FileNotFoundError: If *out_dir* does not exist.
    """
    if not out_dir.exists():
        raise FileNotFoundError(f"Output directory not found: {out_dir}")

    scope = _load_scope(out_dir)
    trails = _load_trails(out_dir)
    observations_by_doc = _load_observations(out_dir)
    manifest = _load_manifest(out_dir)
    playbook = _load_playbook(out_dir)

    # Flatten all observations for cross-section analysis
    all_obs: list[dict[str, Any]] = [
        obs for obs_list in observations_by_doc.values() for obs in obs_list
    ]

    data: dict[str, Any] = {}

    data["corpus_coverage"] = _build_corpus_coverage(scope, trails)
    data["backbone_health"] = _build_backbone_health(trails)
    data["judgment_economics"] = _build_judgment_economics(out_dir, all_obs)
    data["semantic_coverage"] = _build_semantic_coverage(all_obs, playbook)
    data["needs_attention"] = _build_needs_attention(all_obs, observations_by_doc, manifest)
    data["honesty"] = _build_honesty(all_obs, playbook)

    # Surface generated_at from the playbook compiler block (deterministic; no wall-clock)
    if playbook:
        compiler = playbook.get("compiler", {})
        data["generated_at"] = compiler.get("generated_at", "")
        data["compiler_version"] = compiler.get("version", "")
    else:
        data["generated_at"] = ""
        data["compiler_version"] = ""

    return data


def build_after_action_report(out_dir: Path) -> str:
    """Build a Markdown after-action report from a compiled output directory.

    Calls ``build_after_action_data`` and renders each section to Markdown.

    Args:
        out_dir: Path to the ``out/`` directory produced by the engine.

    Returns:
        Markdown-formatted after-action report string.

    Raises:
        FileNotFoundError: If *out_dir* does not exist.
    """
    data = build_after_action_data(out_dir)
    lines: list[str] = []

    lines.append("# Playbook After-Action Report")
    lines.append("")
    lines.append(f"**Output directory:** `{out_dir}`")
    if data.get("generated_at"):
        lines.append(f"**Compiled at:** {data['generated_at']}")
    lines.append("")
    lines.append("---")
    lines.append("")

    lines.extend(_render_corpus_coverage(data["corpus_coverage"]))
    lines.append("")
    lines.extend(_render_backbone_health(data["backbone_health"]))
    lines.append("")
    lines.extend(_render_judgment_economics(data["judgment_economics"]))
    lines.append("")
    lines.extend(_render_semantic_coverage(data["semantic_coverage"]))
    lines.append("")
    lines.extend(_render_needs_attention(data["needs_attention"]))
    lines.append("")
    lines.extend(_render_honesty(data["honesty"]))
    lines.append("")

    return "\n".join(lines)


def write_after_action_report(out_dir: Path, dest: Path) -> None:
    """Build and write the after-action report Markdown and JSON twin.

    Args:
        out_dir: Pipeline output directory.
        dest:    Destination path for the Markdown report (``*.md``).
                 A JSON twin is written alongside at the same stem
                 (e.g. ``report.json`` next to ``report.md``).

    Raises:
        FileNotFoundError: If *out_dir* does not exist.
    """
    data = build_after_action_data(out_dir)
    md = _render_from_data(out_dir, data)

    # Atomic write of Markdown
    tmp_md = dest.with_suffix(".tmp")
    tmp_md.write_text(md, encoding="utf-8")
    tmp_md.replace(dest)

    # Atomic write of JSON twin
    json_dest = dest.with_suffix(".json")
    tmp_json = json_dest.with_suffix(".tmp")
    tmp_json.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp_json.replace(json_dest)


# ---------------------------------------------------------------------------
# Section builders — return structured dicts
# ---------------------------------------------------------------------------


def _load_playbook(out_dir: Path) -> dict[str, Any]:
    """Return the compiled playbook dict, or {} if absent/unparseable."""
    pb_path = out_dir / "playbook.opf.json"
    if not pb_path.exists():
        return {}
    try:
        data: dict[str, Any] = json.loads(pb_path.read_text(encoding="utf-8"))
        return data
    except Exception:  # noqa: BLE001
        _log.warning("Could not parse playbook.opf.json; playbook sections will be limited.")
        return {}


def _build_corpus_coverage(
    scope: dict[str, Any],
    trails: dict[str, Any],
) -> dict[str, Any]:
    """Corpus coverage: agreements, versions, in/out-of-scope decisions."""
    all_doc_ids = sorted(set(scope) | set(trails))
    total = len(all_doc_ids)
    in_scope_docs: list[dict[str, Any]] = []
    out_of_scope_docs: list[dict[str, Any]] = []

    for doc_id in all_doc_ids:
        doc_scope = scope.get(doc_id, {})
        trail = trails.get(doc_id, {})
        in_scope = doc_scope.get("in_scope")

        versions = trail.get("ordered_versions") or []
        n_versions = len(versions) if versions else 0

        entry: dict[str, Any] = {
            "document_id": doc_id,
            "in_scope": in_scope,
            "versions": n_versions,
            "scope_rationale": doc_scope.get("scope_rationale", ""),
            "scope_confidence": doc_scope.get("scope_confidence"),
        }
        if in_scope is False:
            out_of_scope_docs.append(entry)
        else:
            in_scope_docs.append(entry)

    return {
        "total_documents": total,
        "in_scope_count": len(in_scope_docs),
        "out_of_scope_count": len(out_of_scope_docs),
        "in_scope_documents": in_scope_docs,
        "out_of_scope_documents": out_of_scope_docs,
    }


def _build_backbone_health(trails: dict[str, Any]) -> dict[str, Any]:
    """Backbone health: trails ordered, signed copies found, reversals, basis."""
    ordered_count = 0
    signed_count = 0
    reversal_count = 0
    trail_summaries: list[dict[str, Any]] = []

    for doc_id in sorted(trails):
        trail = trails[doc_id]
        ordered_versions = trail.get("ordered_versions") or []
        signed_version = trail.get("signed_version")
        basis = trail.get("basis", "")
        shape = trail.get("shape", "")

        has_ordering = bool(ordered_versions)
        has_signed = bool(signed_version)

        # A "reversal" in trail context means the trail detected proposed-then-reversed events.
        # The flag may be stored directly or we infer from observations (handled in semantic).
        reversals = trail.get("reversals", [])

        if has_ordering:
            ordered_count += 1
        if has_signed:
            signed_count += 1
        if reversals:
            reversal_count += 1

        trail_summaries.append(
            {
                "document_id": doc_id,
                "ordered_versions": ordered_versions,
                "signed_version": signed_version,
                "basis": basis,
                "shape": shape,
                "provenance": trail.get("provenance", "unknown"),
                "provenance_confidence": trail.get("provenance_confidence"),
                "provenance_is_ambiguous": trail.get("provenance_is_ambiguous"),
                "signed_copy_confidence": trail.get("signed_copy_confidence"),
                "reversals": reversals,
            }
        )

    return {
        "total_trails": len(trails),
        "ordered_count": ordered_count,
        "signed_count": signed_count,
        "reversal_count": reversal_count,
        "trails": trail_summaries,
    }


def _build_judgment_economics(
    out_dir: Path,
    all_obs: list[dict[str, Any]],
) -> dict[str, Any]:
    """Judgment economics: unique items judged, dedup ratio, token estimate."""
    judge_dir = out_dir / "judge"
    verdicts_path = judge_dir / "verdicts.jsonl"
    pending_path = judge_dir / "pending.jsonl"

    verdicts_count = 0
    pending_count = 0
    pending_by_kind: dict[str, int] = {}

    if verdicts_path.exists():
        try:
            lines = [
                line
                for line in verdicts_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            verdicts_count = len(lines)
        except Exception:  # noqa: BLE001
            _log.warning("Could not read verdicts.jsonl")

    if pending_path.exists():
        try:
            pending_lines = [
                line
                for line in pending_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            pending_count = len(pending_lines)
            for line in pending_lines:
                try:
                    rec = json.loads(line)
                    kind = rec.get("kind", "unknown")
                    pending_by_kind[kind] = pending_by_kind.get(kind, 0) + 1
                except Exception:  # noqa: BLE001
                    pass
        except Exception:  # noqa: BLE001
            _log.warning("Could not read pending.jsonl")

    # Needs-review observations count as "not yet judged"
    needs_review_count = sum(
        1
        for obs in all_obs
        if isinstance(obs, dict)
        and (
            obs.get("deviation") == "needs_review"
            or obs.get("taxonomy_id") == "needs_review"
            or (obs.get("citation", {}) or {}).get("basis") == "needs_review"
        )
    )

    total_judged = verdicts_count
    token_estimate = pending_count * _AVG_TOKENS_PER_ITEM

    return {
        "verdicts_in_store": verdicts_count,
        "pending_count": pending_count,
        "pending_by_kind": pending_by_kind,
        "needs_review_obs": needs_review_count,
        "token_estimate": token_estimate,
        "judge_dir_present": judge_dir.exists(),
        "total_judged": total_judged,
    }


def _build_semantic_coverage(
    all_obs: list[dict[str, Any]],
    playbook: dict[str, Any],
) -> dict[str, Any]:
    """Semantic coverage: classification %, deviation dist, provenance dist, position hist."""
    total_obs = len(all_obs)
    classified_count = sum(
        1 for obs in all_obs if isinstance(obs, dict) and obs.get("taxonomy_id") is not None
    )
    unclassified_count = total_obs - classified_count
    classification_pct = (classified_count / total_obs * 100) if total_obs > 0 else 0.0

    # Deviation distribution
    deviation_dist: dict[str, int] = {}
    for obs in all_obs:
        if not isinstance(obs, dict):
            continue
        dev = obs.get("deviation", "unknown")
        deviation_dist[dev] = deviation_dist.get(dev, 0) + 1

    # Provenance distribution
    provenance_dist: dict[str, int] = {}
    for obs in all_obs:
        if not isinstance(obs, dict):
            continue
        prov = obs.get("provenance", "unknown")
        provenance_dist[prov] = provenance_dist.get(prov, 0) + 1

    # Rollup-position histogram from playbook clauses (shape-agnostic —
    # reads v0.2 evidence.clauses / summary.historical_stance or v0.1
    # clauses / rollup.position via playbook_clauses()/clause_stance()).
    position_hist: dict[str, int] = {}
    clauses: list[dict[str, Any]] = []
    if playbook:
        clauses = playbook_clauses(playbook)
        for clause in clauses:
            if not isinstance(clause, dict):
                continue
            position = clause_stance(clause)
            position_hist[position] = position_hist.get(position, 0) + 1

    return {
        "total_observations": total_obs,
        "classified_count": classified_count,
        "unclassified_count": unclassified_count,
        "classification_pct": round(classification_pct, 1),
        "deviation_distribution": deviation_dist,
        "provenance_distribution": provenance_dist,
        "rollup_position_histogram": position_hist,
        "total_clauses_in_playbook": len(clauses),
    }


def _build_needs_attention(
    all_obs: list[dict[str, Any]],
    observations_by_doc: dict[str, list[dict[str, Any]]],
    manifest: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Needs attention: failed version ingests, low-confidence, needs_review,
    judge_error, ambiguous-provenance items.
    """
    items: list[dict[str, Any]] = []
    item_num = 0

    # Failed per-version ingests (issue #89): a version that was never
    # actually mined must show up here, not just as a scrolled-past
    # progress-line WARNING that a cache hit wouldn't even re-print — see
    # corpus_manifest.json["version_ingest"] (written by _compute_doc_result).
    for doc_id in sorted(manifest or {}):
        for ver in (manifest or {})[doc_id].get("version_ingest", []) or []:
            if not isinstance(ver, dict) or ver.get("status") != "failed":
                continue
            item_num += 1
            items.append(
                {
                    "item_number": item_num,
                    "document_id": doc_id,
                    "version": ver.get("version", "?"),
                    "taxonomy_id": None,
                    "reasons": [f"version ingest failed: {ver.get('error') or 'unknown error'}"],
                }
            )

    for obs in all_obs:
        if not isinstance(obs, dict):
            continue

        reasons: list[str] = []

        # needs_review sentinel in deviation or basis
        deviation = obs.get("deviation", "")
        if deviation == "needs_review":
            reasons.append("needs_review deviation")

        # judge_error
        if deviation == "judge_error":
            reasons.append("judge_error")

        # taxonomy_id indicates needs_review
        if obs.get("taxonomy_id") == "needs_review":
            reasons.append("needs_review taxonomy_id")

        # Low confidence (when stored as a field)
        conf = obs.get("confidence")
        if conf is not None and isinstance(conf, (int, float)) and conf < 0.5:
            reasons.append(f"low confidence ({conf:.2f})")

        # Ambiguous provenance via trail context — check basis field
        citation = obs.get("citation") or {}
        if isinstance(citation, dict):
            pass  # provenance ambiguity is captured in trail, not obs

        if reasons:
            item_num += 1
            citation_doc = (
                citation.get("document_id", "unknown") if isinstance(citation, dict) else "unknown"
            )
            citation_ver = citation.get("version", "?") if isinstance(citation, dict) else "?"
            items.append(
                {
                    "item_number": item_num,
                    "document_id": citation_doc,
                    "version": citation_ver,
                    "taxonomy_id": obs.get("taxonomy_id"),
                    "reasons": reasons,
                }
            )

    return items


def _build_honesty(
    all_obs: list[dict[str, Any]],
    playbook: dict[str, Any],
) -> dict[str, Any]:
    """Honesty: blank/defaulted fields; human-input-required v0.2 items."""
    blank_fields: list[dict[str, Any]] = []
    human_required: list[dict[str, Any]] = []

    # Enumerate blank/defaulted OPF clause fields (shape-agnostic — see
    # opf_accessors for the v0.2 evidence.clauses/summary vs v0.1
    # clauses/rollup fallback).
    if playbook:
        clauses: list[dict[str, Any]] = playbook_clauses(playbook)
        # A ZERO-clause playbook is the biggest possible gap and previously
        # sailed through this section unflagged — every check below is
        # per-clause, so an empty list produced "no blank fields detected"
        # on a schema-valid but semantically empty document (issue #208).
        # Name the likely cause when it is derivable from the observations.
        if not clauses:
            outcomes = {obs.get("outcome") for obs in all_obs if isinstance(obs, dict)}
            if outcomes and outcomes == {"unsigned"}:
                reason = (
                    "playbook compiled ZERO clause positions — every observation "
                    "is outcome=unsigned (no signed/executed copy was detected in "
                    "any document; check signature blocks or set signed_version "
                    "in hints.yaml)"
                )
            else:
                reason = (
                    "playbook compiled ZERO clause positions — the compiled "
                    "playbook is schema-valid but semantically empty"
                )
            blank_fields.append({"clause_id": "—", "field": "evidence.clauses", "reason": reason})
        for clause in clauses:
            if not isinstance(clause, dict):
                continue
            clause_id = clause.get("id", "?")
            conf = clause_confidence(clause)

            # Blank our_standard
            if clause.get("our_standard") is None:
                blank_fields.append(
                    {
                        "clause_id": clause_id,
                        "field": "our_standard",
                        "reason": "no template clause found",
                    }
                )

            # Low-confidence rollup
            score = conf.get("score")
            if score is not None and score < 0.5:
                blank_fields.append(
                    {
                        "clause_id": clause_id,
                        "field": "rollup.confidence.score",
                        "reason": f"low score ({score:.2f})",
                    }
                )

            # Under-grounded positions require human review (issue #107): this
            # previously only checked negotiable/hold_firm, but "standard" and
            # "acceptable_variants_exist" built on a handful of our-paper
            # citations are the more dangerous case — they read as settled
            # guidance rather than a live negotiation point. Check ALL
            # positions, not just negotiable/hold_firm.
            position = clause_stance(clause)
            if position and position != "unknown":
                n_our = conf.get("n_our_paper", 0) or 0
                if n_our < 3:
                    human_required.append(
                        {
                            "clause_id": clause_id,
                            "position": position,
                            "reason": f"position={position!r} with n_our_paper={n_our} (< 3 citations)",
                        }
                    )

    # needs_review observations require human input
    needs_review_obs = [
        obs for obs in all_obs if isinstance(obs, dict) and obs.get("deviation") == "needs_review"
    ]

    # reversals that need GC review for Floor derivation
    reversal_obs = [
        obs
        for obs in all_obs
        if isinstance(obs, dict) and obs.get("outcome") == "proposed_then_reversed"
    ]

    return {
        "blank_or_defaulted_fields": blank_fields,
        "human_input_required": human_required,
        "needs_review_observation_count": len(needs_review_obs),
        "reversal_observation_count": len(reversal_obs),
        "honesty_notes": [
            "GC-authored Posture is a v0.2 human-input field not yet generated by the engine.",
            "Floor clauses are derived from classified reversals and require attorney sign-off.",
        ],
    }


# ---------------------------------------------------------------------------
# Markdown renderers — consume structured data, return list[str]
# ---------------------------------------------------------------------------


def _render_corpus_coverage(data: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    lines.append("## Corpus Coverage")
    lines.append("")
    lines.append(
        f"**{data['in_scope_count']} in scope / {data['total_documents']} total** "
        f"({data['out_of_scope_count']} excluded)"
    )
    lines.append("")

    if data["in_scope_documents"]:
        lines.append("### In-scope agreements")
        lines.append("")
        lines.append("| Agreement | Versions | Scope rationale |")
        lines.append("|-----------|----------|-----------------|")
        for doc in data["in_scope_documents"]:
            rationale = _md_escape(doc.get("scope_rationale", "") or "")
            lines.append(f"| `{doc['document_id']}` | {doc['versions']} | {rationale} |")
        lines.append("")

    if data["out_of_scope_documents"]:
        lines.append("### Excluded agreements")
        lines.append("")
        lines.append("| Agreement | Rationale |")
        lines.append("|-----------|-----------|")
        for doc in data["out_of_scope_documents"]:
            rationale = _md_escape(doc.get("scope_rationale", "") or "")
            lines.append(f"| `{doc['document_id']}` | {rationale} |")
        lines.append("")

    return lines


def _render_backbone_health(data: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    lines.append("## Backbone Health")
    lines.append("")
    lines.append(
        f"**{data['ordered_count']}/{data['total_trails']}** trails ordered | "
        f"**{data['signed_count']}/{data['total_trails']}** signed copies found | "
        f"**{data['reversal_count']}** reversal(s) detected"
    )
    lines.append("")

    if data["trails"]:
        lines.append("| Agreement | Version order | Signed | Basis | Provenance |")
        lines.append("|-----------|---------------|--------|-------|------------|")
        for t in data["trails"]:
            ordered = " → ".join(str(v) for v in (t["ordered_versions"] or []))
            signed = str(t["signed_version"]) if t["signed_version"] else "*(none)*"
            basis = t.get("basis", "")
            prov = t.get("provenance", "unknown")
            lines.append(f"| `{t['document_id']}` | {ordered} | {signed} | {basis} | `{prov}` |")
        lines.append("")

    return lines


def _render_judgment_economics(data: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    lines.append("## Judgment Economics")
    lines.append("")

    if not data["judge_dir_present"]:
        lines.append(
            "> No `judge/` directory found — engine ran in stub-judge mode "
            "(no real LLM judgments recorded)."
        )
        lines.append("")
        return lines

    lines.append(f"**Verdicts in store:** {data['verdicts_in_store']}")
    lines.append(f"**Pending (awaiting verdict):** {data['pending_count']}")

    if data["pending_by_kind"]:
        lines.append("")
        lines.append("| Kind | Count |")
        lines.append("|------|-------|")
        for kind, count in sorted(data["pending_by_kind"].items()):
            lines.append(f"| {kind} | {count} |")

    if data["pending_count"] > 0:
        lines.append(f"\n**Rough token estimate:** ~{data['token_estimate']:,} tokens")

    if data["needs_review_obs"] > 0:
        lines.append(
            f"\n> **{data['needs_review_obs']} observation(s)** have `needs_review` "
            "deviation — run `playbook judge` to resolve them."
        )

    lines.append("")
    return lines


def _render_semantic_coverage(data: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    lines.append("## Semantic Coverage")
    lines.append("")
    lines.append(
        f"**{data['classified_count']}/{data['total_observations']} clauses classified "
        f"({data['classification_pct']}%)** | "
        f"{data['unclassified_count']} unclassified"
    )
    lines.append("")

    # Deviation distribution
    if data["deviation_distribution"]:
        lines.append("### Deviation distribution")
        lines.append("")
        lines.append("| Deviation | Count |")
        lines.append("|-----------|-------|")
        for dev, count in sorted(data["deviation_distribution"].items()):
            lines.append(f"| `{dev}` | {count} |")
        lines.append("")

    # Provenance distribution
    if data["provenance_distribution"]:
        lines.append("### Provenance distribution")
        lines.append("")
        lines.append("| Provenance | Count |")
        lines.append("|------------|-------|")
        for prov, count in sorted(data["provenance_distribution"].items()):
            lines.append(f"| `{prov}` | {count} |")
        lines.append("")

    # Rollup position histogram
    if data["rollup_position_histogram"]:
        lines.append("### Rollup-position histogram")
        lines.append("")
        lines.append(f"*(from {data['total_clauses_in_playbook']} clause(s) in playbook)*")
        lines.append("")
        lines.append("| Position | Count |")
        lines.append("|----------|-------|")
        for pos, count in sorted(data["rollup_position_histogram"].items()):
            lines.append(f"| `{pos}` | {count} |")
        lines.append("")

    return lines


def _render_needs_attention(items: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    lines.append("## Needs Attention")
    lines.append("")

    if not items:
        lines.append(
            "> No needs-attention items detected — all observations have clean "
            "deviation and taxonomy_id."
        )
        lines.append("")
        return lines

    lines.append(f"**{len(items)} item(s) require attention before publishing.**")
    lines.append("")
    lines.append("| # | Agreement | Version | Taxonomy ID | Reasons |")
    lines.append("|---|-----------|---------|-------------|---------|")
    for item in items:
        reasons_str = "; ".join(item["reasons"])
        tid = item.get("taxonomy_id") or "*(unclassified)*"
        lines.append(
            f"| {item['item_number']} | `{item['document_id']}` | {item['version']} "
            f"| `{tid}` | {_md_escape(reasons_str)} |"
        )

    lines.append("")
    return lines


def _render_honesty(data: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    lines.append("## Honesty")
    lines.append("")

    lines.append("> This section explicitly lists what the engine does not know.")
    lines.append("")

    for note in data.get("honesty_notes", []):
        lines.append(f"- {note}")
    lines.append("")

    blank = data.get("blank_or_defaulted_fields", [])
    if blank:
        lines.append(f"**{len(blank)} blank/defaulted field(s):**")
        lines.append("")
        lines.append("| Clause | Field | Reason |")
        lines.append("|--------|-------|--------|")
        for entry in blank:
            lines.append(
                f"| `{entry['clause_id']}` | `{entry['field']}` | {_md_escape(entry['reason'])} |"
            )
        lines.append("")
    else:
        lines.append("No blank or defaulted fields detected in the compiled playbook.")
        lines.append("")

    human = data.get("human_input_required", [])
    if human:
        lines.append(f"**{len(human)} clause(s) require human sign-off:**")
        lines.append("")
        lines.append("| Clause | Position | Reason |")
        lines.append("|--------|----------|--------|")
        for entry in human:
            lines.append(
                f"| `{entry['clause_id']}` | `{entry['position']}` | {_md_escape(entry['reason'])} |"
            )
        lines.append("")

    nr = data.get("needs_review_observation_count", 0)
    if nr > 0:
        lines.append(
            f"**{nr} observation(s)** have `needs_review` deviation (human verdict needed)."
        )
        lines.append("")

    rev = data.get("reversal_observation_count", 0)
    if rev > 0:
        lines.append(
            f"**{rev} reversal(s)** detected — Floor derivation requires attorney classification."
        )
        lines.append("")

    return lines


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _render_from_data(out_dir: Path, data: dict[str, Any]) -> str:
    """Render Markdown from a pre-built data dict (used by write_after_action_report)."""
    lines: list[str] = []

    lines.append("# Playbook After-Action Report")
    lines.append("")
    lines.append(f"**Output directory:** `{out_dir}`")
    if data.get("generated_at"):
        lines.append(f"**Compiled at:** {data['generated_at']}")
    lines.append("")
    lines.append("---")
    lines.append("")

    lines.extend(_render_corpus_coverage(data["corpus_coverage"]))
    lines.append("")
    lines.extend(_render_backbone_health(data["backbone_health"]))
    lines.append("")
    lines.extend(_render_judgment_economics(data["judgment_economics"]))
    lines.append("")
    lines.extend(_render_semantic_coverage(data["semantic_coverage"]))
    lines.append("")
    lines.extend(_render_needs_attention(data["needs_attention"]))
    lines.append("")
    lines.extend(_render_honesty(data["honesty"]))
    lines.append("")

    return "\n".join(lines)


def _md_escape(text: str) -> str:
    """Escape pipe characters so they don't break Markdown tables."""
    return text.replace("|", "\\|")
