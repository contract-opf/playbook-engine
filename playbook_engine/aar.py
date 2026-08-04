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
  5. **Needs attention** — quarantined documents (from ``quarantine.json``),
     degenerate deviation distributions (rubber-stamp judging), corpus-level
     provenance ambiguity flips, low-confidence/``needs_review``/
     ``judge_error`` items, each with its item number.
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
from playbook_engine.opf_accessors import (
    clause_confidence,
    clause_stance,
    playbook_clause_library,
    playbook_clauses,
)

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Token estimate constants
# ---------------------------------------------------------------------------

_AVG_TOKENS_PER_ITEM = 200

# ---------------------------------------------------------------------------
# Silent-degradation thresholds
# ---------------------------------------------------------------------------

# A deviation distribution where one value covers more than this share of the
# deviation-judged observations is degenerate — a rubber-stamp judge, not a
# real distribution (a prior real run recorded 1029/1029 verdicts "none" and
# no gate flagged it).
_DEGENERATE_DEVIATION_SHARE = 0.9
# ...but only once there are at least this many judged observations — a tiny
# corpus can legitimately share one deviation value.
_DEGENERATE_DEVIATION_MIN_OBS = 20


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
    quarantine = _load_quarantine(out_dir)

    # Flatten all observations for cross-section analysis
    all_obs: list[dict[str, Any]] = [
        obs for obs_list in observations_by_doc.values() for obs in obs_list
    ]

    data: dict[str, Any] = {}

    data["corpus_coverage"] = _build_corpus_coverage(scope, trails)
    data["backbone_health"] = _build_backbone_health(trails)
    data["judgment_economics"] = _build_judgment_economics(out_dir, all_obs)
    data["semantic_coverage"] = _build_semantic_coverage(all_obs, playbook)
    data["needs_attention"] = _build_needs_attention(
        all_obs,
        observations_by_doc,
        manifest,
        quarantine=quarantine,
        trails=trails,
        playbook=playbook,
    )
    data["honesty"] = _build_honesty(all_obs, playbook)
    data["artifacts"] = _build_artifacts(out_dir, playbook)

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
    # .get: report.json twins written before the artifacts section existed
    # must still render.
    if data.get("artifacts") is not None:
        lines.extend(_render_artifacts(data["artifacts"]))
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
        ValueError: If *dest* already ends in ``.json`` — the JSON twin is
            derived from the Markdown path by replacing its suffix, so a
            ``.json`` *dest* would collide with (and clobber) the twin.
    """
    if dest.suffix.lower() == ".json":
        raise ValueError(
            f"--out must be the Markdown report path, not the JSON twin: {dest}. "
            "The .json twin is derived automatically from the Markdown path "
            "(e.g. --out report.md also writes report.json)."
        )

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


def _load_quarantine(out_dir: Path) -> list[dict[str, Any]]:
    """Return the ``quarantine.json`` entries, or [] if absent/unparseable.

    The pipeline writes quarantine.json for documents it had to set aside
    (fail-loud QA-gate failures, malformed hints.yaml, every version failing
    ingest). Until the report read it back, ``validate`` passed on a silently
    thinner playbook.
    """
    q_path = out_dir / "quarantine.json"
    if not q_path.exists():
        return []
    try:
        raw = json.loads(q_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        _log.warning("Could not parse quarantine.json; quarantined documents will be omitted.")
        return []
    if not isinstance(raw, list):
        return []
    return [q for q in raw if isinstance(q, dict) and q.get("document_id")]


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


def _zero_clause_reason(
    all_obs: list[dict[str, Any]], playbook: dict[str, Any] | None = None
) -> str:
    """Derive the likely-cause explanation for a playbook that compiled ZERO
    clause positions — shared verbatim between the Needs Attention and
    Honesty sections (issue #25) so the two can never drift out of sync.

    Checks are ordered most-specific-first and are NOT mutually exclusive
    with each other in the data — the first match wins:
      1. All observations outcome=unsigned (unchanged from fix round 1 —
         kept first so a corpus that is entirely unsigned keeps reporting
         that as the cause even when the taxonomy also happens to be
         empty, e.g. an all-unsigned fixture built from a minimal/empty
         taxonomy).
      2. Empty clause taxonomy (issue #25 fix round 2): taxonomy.entries
         is schema-legal to be empty (spec/playbook.schema-0.3.json has no
         minItems on it — see playbook_engine/taxonomy.py's loader), and a
         zero-entry taxonomy leaves nothing to classify any observation
         into regardless of signing status.
      3. Generic fallback when neither specific cause is derivable.
    """
    outcomes = {obs.get("outcome") for obs in all_obs if isinstance(obs, dict)}
    if outcomes and outcomes == {"unsigned"}:
        return (
            "playbook compiled ZERO clause positions — every observation "
            "is outcome=unsigned (no signed/executed copy was detected in "
            "any document; check signature blocks or set signed_version "
            "in hints.yaml)"
        )
    if playbook is not None and not (playbook.get("taxonomy") or {}).get("entries"):
        return (
            "playbook compiled ZERO clause positions — the compiled "
            "playbook's clause taxonomy has zero entries (taxonomy.entries "
            "is empty), so no observation could be classified into any "
            "clause taxonomy position"
        )
    return (
        "playbook compiled ZERO clause positions — the compiled "
        "playbook is schema-valid but semantically empty"
    )


def _build_needs_attention(
    all_obs: list[dict[str, Any]],
    observations_by_doc: dict[str, list[dict[str, Any]]],
    manifest: dict[str, Any] | None = None,
    quarantine: list[dict[str, Any]] | None = None,
    trails: dict[str, Any] | None = None,
    playbook: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Needs attention: quarantined documents, a zero-clause playbook,
    degenerate deviation distributions, corpus-level provenance ambiguity
    flips, failed version ingests, low-confidence, needs_review, judge_error
    items.
    """
    items: list[dict[str, Any]] = []
    item_num = 0

    # Quarantined documents: the pipeline writes quarantine.json but until
    # now NOTHING read it back, so a quarantined document silently vanished
    # from the report and `validate` passed on a thinner playbook. Surface
    # the manifest-vs-quarantine count first, then every quarantined
    # document id with its reason.
    quarantine = quarantine or []
    if quarantine:
        # issue #83: a QA-quarantined document can now ALSO carry a partial
        # (zero-observation) entry in corpus_manifest.json — see
        # pipeline._build_quarantine_corpus_doc — appended so its extractor
        # label/QA status survive for audit visibility. manifest_count here
        # must keep meaning "documents that actually contributed playbook
        # content", so exclude any manifest entry whose document_id is ALSO
        # in quarantine.json — otherwise a quarantined document's partial
        # record would double-count as if the compiled playbook covered it
        # too. Both files key on document_id the same way (both are written
        # AFTER the pipeline's born-safe pseudonymization pass — see
        # pipeline.mine_corpus — quarantine.json used to be written before
        # it, which is what made this comparison a no-op until that was
        # fixed), so this holds whether or not known_entities is configured.
        quarantined_ids = {q.get("document_id") for q in quarantine}
        manifest_count = len((manifest or {}).keys() - quarantined_ids)
        item_num += 1
        items.append(
            {
                "item_number": item_num,
                "document_id": "—",
                "version": "—",
                "taxonomy_id": None,
                "reasons": [
                    f"{len(quarantine)} document(s) quarantined vs "
                    f"{manifest_count} in the corpus manifest — the compiled "
                    f"playbook covers {manifest_count} of "
                    f"{manifest_count + len(quarantine)} discovered document(s)"
                ],
            }
        )
        for q in quarantine:
            item_num += 1
            items.append(
                {
                    "item_number": item_num,
                    "document_id": q.get("document_id", "unknown"),
                    "version": "—",
                    "taxonomy_id": None,
                    "reasons": [f"quarantined: {q.get('reason') or 'unknown reason'}"],
                }
            )

    # Zero-clause playbook (issue #25): a schema-valid but semantically
    # empty playbook — evidence.clauses compiled to nothing — is the
    # single biggest gap a reader can miss, and previously only the
    # Honesty section's blank-field list caught it; an empty
    # needs_attention list read as "all observations clean" even though
    # nothing was actually classified into a clause position. Shares its
    # derived likely cause with the Honesty entry via _zero_clause_reason
    # so the two sections never disagree. Guarded on `if playbook:` (not
    # just `if not playbook_clauses(playbook):`) so a report built before
    # `playbook project` has even run — no playbook.opf.json on disk yet —
    # is never mistaken for a compiled-but-empty playbook.
    if playbook and not playbook_clauses(playbook):
        item_num += 1
        items.append(
            {
                "item_number": item_num,
                "document_id": "—",
                "version": "—",
                "taxonomy_id": None,
                "reasons": [_zero_clause_reason(all_obs, playbook)],
            }
        )

    # Degenerate judging: a prior real run recorded 1029/1029 deviation
    # verdicts "none" (a scripted rubber stamp) and no gate flagged it. When
    # one value covers more than _DEGENERATE_DEVIATION_SHARE of the
    # deviation-judged observations the distribution carries no signal — say
    # so instead of letting it read as a clean corpus. Sentinels
    # (needs_review/judge_error) are not judged verdicts, so they don't count.
    judged = [
        obs["deviation"]
        for obs in all_obs
        if isinstance(obs, dict)
        and obs.get("deviation") not in (None, "", "needs_review", "judge_error")
    ]
    if len(judged) >= _DEGENERATE_DEVIATION_MIN_OBS:
        dev_counts: dict[str, int] = {}
        for dev in judged:
            dev_counts[dev] = dev_counts.get(dev, 0) + 1
        top_value, top_count = max(dev_counts.items(), key=lambda kv: kv[1])
        if top_count / len(judged) > _DEGENERATE_DEVIATION_SHARE:
            item_num += 1
            items.append(
                {
                    "item_number": item_num,
                    "document_id": "—",
                    "version": "—",
                    "taxonomy_id": None,
                    "reasons": [
                        f"degenerate deviation distribution: {top_count}/"
                        f"{len(judged)} deviation-judged observation(s) share "
                        f"the value {top_value!r} — judging looks like a "
                        "rubber stamp; review the deviation judge before "
                        "trusting this playbook"
                    ],
                }
            )

    # Corpus-level provenance ambiguity: pipeline flips provenance to
    # "counterparty_paper" whenever the detector is ambiguous (confidence
    # below threshold); on a prior run 40/44 documents were flipped and
    # nothing warned. When more than half the trails carry
    # provenance_is_ambiguous, the corpus provenance split is a default, not
    # a detection — one loud line here.
    trails = trails or {}
    ambiguous_count = sum(
        1
        for t in trails.values()
        if isinstance(t, dict) and t.get("provenance_is_ambiguous") is True
    )
    if trails and ambiguous_count > len(trails) / 2:
        item_num += 1
        items.append(
            {
                "item_number": item_num,
                "document_id": "—",
                "version": "—",
                "taxonomy_id": None,
                "reasons": [
                    f"provenance ambiguity-flipped for {ambiguous_count}/"
                    f"{len(trails)} document(s): detector confidence was below "
                    "threshold so provenance defaulted to 'counterparty_paper' "
                    "— the corpus provenance distribution is a default, not a "
                    "detection"
                ],
            }
        )

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
            reason = _zero_clause_reason(all_obs, playbook)
            blank_fields.append({"clause_id": "—", "field": "evidence.clauses", "reason": reason})

            # Issue #25: evidence.clauses is not the only top-level section
            # that can be empty. When the compiled playbook is this hollow,
            # name every other empty top-level section too, so the reader
            # sees the full extent of "this playbook is empty" rather than
            # just the clause count. Scoped inside `if not clauses:` on
            # purpose — posture/floor are unconditionally empty-but-present
            # on EVERY compiled playbook today (#140: no interview has been
            # run and no invariants have been derived in this engine slice),
            # so flagging them here unconditionally would fire on every
            # healthy, fully-populated playbook too. The unconditional
            # honesty_notes below already disclose that standing fact; this
            # block instead answers "is THIS playbook empty end to end?".
            if not playbook_clause_library(playbook):
                blank_fields.append(
                    {
                        "clause_id": "—",
                        "field": "evidence.clause_library",
                        "reason": (
                            "no clause concepts were extracted — the compiled "
                            "clause-concept library is empty"
                        ),
                    }
                )
            if not playbook.get("posture"):
                blank_fields.append(
                    {
                        "clause_id": "—",
                        "field": "posture",
                        "reason": (
                            "GC-authored Posture is a v0.2 human-input field not yet "
                            "generated by the engine — no interview has been run"
                        ),
                    }
                )
            if not playbook.get("floor"):
                blank_fields.append(
                    {
                        "clause_id": "—",
                        "field": "floor",
                        "reason": (
                            "Floor clauses are derived from classified reversals and "
                            "require attorney sign-off — none have been derived yet"
                        ),
                    }
                )
            if not (playbook.get("corpus") or {}).get("documents"):
                blank_fields.append(
                    {
                        "clause_id": "—",
                        "field": "corpus.documents",
                        "reason": (
                            "no documents are recorded in the compiled playbook's corpus section"
                        ),
                    }
                )
            # Issue #25 fix round 2: taxonomy is a required top-level section
            # (spec/playbook.schema-0.3.json `required`) but taxonomy.entries
            # carries no minItems, so `entries: []` is schema-legal — and a
            # zero-entry taxonomy is itself a likely root cause of the
            # zero-clause playbook this whole block is reporting on (nothing
            # to classify observations into). Same reasoning as the
            # clause_library/posture/floor/corpus.documents checks above:
            # scoped inside `if not clauses:` so a healthy, non-empty
            # taxonomy on a fully-populated playbook never flags here.
            if not (playbook.get("taxonomy") or {}).get("entries"):
                blank_fields.append(
                    {
                        "clause_id": "—",
                        "field": "taxonomy.entries",
                        "reason": (
                            "no taxonomy entries are defined — the compiled "
                            "playbook's clause taxonomy is empty"
                        ),
                    }
                )
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


def _build_artifacts(out_dir: Path, playbook: dict[str, Any] | None) -> dict[str, Any]:
    """Artifacts: what shipped in out_dir, incl. the OPF 0.3 digest/bundle."""
    digest = (playbook or {}).get("digest") or {}
    token_estimate = None
    if digest:
        from playbook_engine.digest import digest_token_estimate  # noqa: PLC0415

        token_estimate = digest_token_estimate(digest)
    return {
        "opf_version": (playbook or {}).get("opf_version"),
        "digest_present": bool(digest),
        "digest_clause_count": digest.get("clause_count"),
        "digest_token_estimate": token_estimate,
        "files": {
            name: (out_dir / name).exists()
            for name in (
                "playbook.opf.json",
                "playbook.digest.json",
                "playbook.review.html",
                "playbook.opf.html",
            )
        },
    }


def _render_artifacts(data: dict[str, Any]) -> list[str]:
    lines = ["## Artifacts"]
    lines.append("")
    opf_version = data.get("opf_version")
    lines.append(f"- **OPF version:** {opf_version or 'unknown (no playbook.opf.json)'}")
    if data.get("digest_present"):
        est = data.get("digest_token_estimate")
        est_str = f"~{est:,} tokens (chars/4)" if est is not None else "size unknown"
        lines.append(
            f"- **Digest:** present — {data.get('digest_clause_count', '?')} clauses, {est_str}"
        )
    else:
        lines.append(
            "- **Digest:** absent (pre-0.3 playbook — run `playbook digest` to derive one)"
        )
    present = [name for name, ok in data.get("files", {}).items() if ok]
    missing = [name for name, ok in data.get("files", {}).items() if not ok]
    if present:
        lines.append(f"- **Present:** {', '.join(f'`{n}`' for n in present)}")
    if missing:
        lines.append(f"- **Not yet rendered:** {', '.join(f'`{n}`' for n in missing)}")
    return lines


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
    # .get: report.json twins written before the artifacts section existed
    # must still render.
    if data.get("artifacts") is not None:
        lines.extend(_render_artifacts(data["artifacts"]))
        lines.append("")

    return "\n".join(lines)


def _md_escape(text: str) -> str:
    """Escape pipe characters so they don't break Markdown tables."""
    return text.replace("|", "\\|")
