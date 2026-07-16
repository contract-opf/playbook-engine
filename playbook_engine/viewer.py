"""Playbook review viewer and feedback loop — issue #68.

Provides a self-contained static HTML review surface for non-engineer
reviewers, plus a feedback-apply path that translates reviewer corrections
into engine-actionable hints and verdict-store entries.

Public API
----------
``render_review_html(out_dir, out_file=None) -> str``
    Read ``playbook.opf.json`` from *out_dir* and return a single self-contained
    HTML string.  If *out_file* is given, also write the file atomically.
    Requires no external network (no CDN, no fetch).

``render_review_html`` accepts an optional ``alias_map`` (issue #146): the
``alias -> real entity name`` held-out map written by
``entity_registry.write_holdout_map`` (``<out_dir>/alias_map.json``) at
corpus-mining time (issue #153). When supplied, every alias occurring in the
rendered HTML (titles, summaries, citations, ...) is resolved to its real
name for readability by an authorized reviewer. Access to the map IS the
authorization gate: the map is a restricted-permission sidecar the caller
must already have read access to (chmod 0600) — this function does no
additional access control of its own. Critically, resolution happens ONLY
in the returned/written HTML: ``playbook.opf.json`` on disk is never read
back and never rewritten, so the stored artifact continues to hold only
aliases regardless of whether a reviewer renders it with or without the map.

``apply_feedback(out_dir, feedback_path) -> ApplyResult``
    Read *feedback_path* (a ``feedback.json`` produced by the HTML viewer) and
    translate corrections into:

    - ``hints.yaml`` per-document entries (provenance / signed / order overrides)
    - ``VerdictStore`` entries (classification corrections via the agent-judge
      bridge from issue #64)
    - ``viewer_notes.md`` for free-text notes and reviewer comments
    - ``curation.pins`` EMBEDDED directly in ``playbook.opf.json`` (issue
      #147) for ``override`` — an attorney-pinned clause position. Unlike
      every other correction key, this one rewrites ``playbook.opf.json``
      itself: the pin records the asserted position, the clause's
      ``historical_stance`` at pin time (``baseline_stance``, so a later
      recompile can tell whether evidence has actually moved), and refreshes
      ``identity.content_hash``/``section_digests`` if the document already
      carries an ``identity`` block. A pipeline recompile
      (``pipeline.project_playbook``) preserves the pin and flags/clears its
      ``conflict`` against freshly recomputed evidence — see
      ``playbook_engine/curation.py``.

    Any correction key ``apply_feedback`` cannot honor is recorded in
    ``ApplyResult.skipped`` instead of being silently dropped, so callers
    never report false success (issue #138).

Feedback JSON schema (produced by the viewer's **Export feedback** button)::

    {
      "C1": {"comment": "...", "override": null},
      "C1.1": {
        "comment": "looks correct",
        "provenance": "counterparty_paper",
        "classification": "governing_law",
        "signed_version": "v3",
        "order": ["v1", "v2", "v3"],
        "note": "free-text note"
      }
    }

Each key is an item number (``Cx`` for a clause, ``Cx.y`` for an observation).
Recognised correction keys:

- ``provenance``    → ``hints.yaml`` for the cited document
- ``signed_version``→ ``hints.yaml`` for the cited document
- ``order``         → ``hints.yaml`` for the cited document
- ``classification``→ VerdictStore entry (``taxonomy_id`` correction)
- ``note``          → appended to ``viewer_notes.md``
- ``comment``       → appended to ``viewer_notes.md`` (same sink as ``note``)
- ``override``      → an embedded ``curation`` pin in ``playbook.opf.json``
                      (issue #147)

Any other key is not applied; it is reported back via ``ApplyResult.skipped``
as a human-readable "not applied" message.
"""

from __future__ import annotations

import datetime
import html as html_lib
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from playbook_engine.agent_judge import VerdictStore
from playbook_engine.canonicalize import compute_section_digests, content_hash
from playbook_engine.clause_tree import ClauseTree
from playbook_engine.curation import CurationPin
from playbook_engine.opf_accessors import clause_confidence, clause_stance, playbook_clauses
from playbook_engine.playbook_assembler import write_playbook

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public result type
# ---------------------------------------------------------------------------


@dataclass
class ApplyResult:
    """Summary of what ``apply_feedback`` wrote.

    Attributes:
        hints_written:    Document IDs for which ``hints.yaml`` was written
                          or updated.
        verdicts_written: Number of VerdictStore entries written.
        notes_written:    ``True`` if ``viewer_notes.md`` was updated.
        pins_written:     Item numbers whose ``override`` was embedded as a
                          ``curation`` pin in ``playbook.opf.json`` (issue
                          #147). Empty if no ``override`` corrections were
                          present.
        skipped:          Item number → list of human-readable "not applied"
                          messages, one per correction key that
                          ``apply_feedback`` recognised as unsupported.
                          Empty if every key was applied.
    """

    hints_written: list[str] = field(default_factory=list)
    verdicts_written: int = 0
    notes_written: bool = False
    pins_written: list[str] = field(default_factory=list)
    skipped: dict[str, list[str]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Item numbering
# ---------------------------------------------------------------------------


def _build_index(doc: dict[str, Any]) -> list[tuple[str, str, dict[str, Any]]]:
    """Return a flat list of (number, kind, payload) for all clauses and observations.

    Clause items are numbered ``C1``, ``C2``, … (ordered by taxonomy_id then id
    for determinism).  Observation items within a clause are numbered
    ``C1.1``, ``C1.2``, … preserving the order they appear in
    ``observed_positions``.

    Args:
        doc: The parsed OPF ``playbook.opf.json`` dict.

    Returns:
        List of ``(item_number, kind, payload)`` where *kind* is ``"clause"``
        or ``"observation"`` and *payload* holds the clause/observation dict
        plus extra keys ``_clause_id``, ``_clause_num``, and (for observations)
        ``_obs_num``.
    """
    clauses = playbook_clauses(doc)
    # Sort clauses deterministically: by taxonomy_id then id
    sorted_clauses = sorted(clauses, key=lambda c: (c.get("taxonomy_id", ""), c.get("id", "")))

    index: list[tuple[str, str, dict[str, Any]]] = []
    for clause_num, clause in enumerate(sorted_clauses, start=1):
        cnum = f"C{clause_num}"
        clause_payload = {**clause, "_clause_id": clause.get("id"), "_clause_num": cnum}
        index.append((cnum, "clause", clause_payload))

        obs_list = clause.get("observed_positions", [])
        for obs_num, obs in enumerate(obs_list, start=1):
            onum = f"{cnum}.{obs_num}"
            obs_payload = {
                **obs,
                "_clause_id": clause.get("id"),
                "_clause_num": cnum,
                "_obs_num": onum,
                "_clause_title": clause.get("title", ""),
            }
            index.append((onum, "observation", obs_payload))

    return index


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------

_POSITION_COLORS = {
    # OPF v0.1 rollup.position (prescriptive) — kept for back-compat.
    "standard": "#2563eb",
    "acceptable_variants_exist": "#059669",
    "negotiable": "#d97706",
    "hold_firm": "#dc2626",
    # OPF v0.2 summary.historical_stance (descriptive) — issue #155.
    "consistently_held": "#2563eb",
    "usually_held": "#059669",
    "mixed": "#d97706",
    "usually_conceded": "#dc2626",
    "no_signal": "#9ca3af",
}

_DEVIATION_BADGES = {
    "none": ("none", "#d1fae5", "#065f46"),
    "reworded_equivalent": ("reworded", "#dbeafe", "#1e40af"),
    "substantive": ("substantive", "#fee2e2", "#991b1b"),
    "needs_review": ("needs review", "#fef3c7", "#92400e"),
}

_RISK_BADGES = {
    ("better", "none"): ("better/none", "#d1fae5", "#065f46"),
    ("better", "minor"): ("better/minor", "#d1fae5", "#065f46"),
    ("better", "material"): ("better/material", "#d1fae5", "#065f46"),
    ("neutral", "none"): ("neutral", "#f3f4f6", "#374151"),
    ("neutral", "minor"): ("neutral/minor", "#f3f4f6", "#374151"),
    ("neutral", "material"): ("neutral/material", "#f3f4f6", "#374151"),
    ("worse", "none"): ("worse/none", "#fef3c7", "#92400e"),
    ("worse", "minor"): ("worse/minor", "#fef3c7", "#92400e"),
    ("worse", "material"): ("worse/material", "#fee2e2", "#991b1b"),
}


# ---------------------------------------------------------------------------
# Feedback correction keys — which are honored vs. reported as skipped
# ---------------------------------------------------------------------------

# Every correction key apply_feedback knows how to act on. Anything else is
# recorded in ApplyResult.skipped instead of being silently dropped — see
# issue #138. "override" (issue #147) is honored as an embedded curation pin.
_RECOGNIZED_FEEDBACK_KEYS = frozenset(
    {"provenance", "signed_version", "order", "classification", "note", "comment", "override"}
)

# Friendlier per-key messages for keys we know about but don't yet honor.
# Falls back to a generic "<key> not yet supported" for anything else.
_UNSUPPORTED_KEY_MESSAGES: dict[str, str] = {}


def _unsupported_message(key: str) -> str:
    return _UNSUPPORTED_KEY_MESSAGES.get(key, f"{key!r} not yet supported")


def _badge(text: str, bg: str, fg: str) -> str:
    return (
        f'<span style="background:{bg};color:{fg};padding:1px 6px;'
        f'border-radius:3px;font-size:0.78em;font-weight:600">{html_lib.escape(text)}</span>'
    )


def _deviation_badge(deviation: str) -> str:
    label, bg, fg = _DEVIATION_BADGES.get(deviation, (deviation, "#f3f4f6", "#374151"))
    return _badge(label, bg, fg)


def _risk_badge(risk_delta: dict[str, str]) -> str:
    direction = risk_delta.get("direction", "neutral")
    magnitude = risk_delta.get("magnitude", "none")
    label, bg, fg = _RISK_BADGES.get(
        (direction, magnitude), (f"{direction}/{magnitude}", "#f3f4f6", "#374151")
    )
    return _badge(label, bg, fg)


def _outcome_badge(outcome: str) -> str:
    if outcome == "signed":
        return _badge("signed", "#d1fae5", "#065f46")
    if outcome == "proposed_then_reversed":
        return _badge("reversed", "#fee2e2", "#991b1b")
    return _badge(outcome, "#f3f4f6", "#374151")


def _citation_str(ref: dict[str, Any] | None) -> str:
    if not ref:
        return ""
    doc_id = ref.get("document_id", "")
    version = ref.get("version")
    clause_path = ref.get("clause_path", "")
    parts = [doc_id]
    if version is not None:
        parts.append(f"v{version}")
    if clause_path:
        parts.append(f"§{clause_path}")
    return " ".join(parts)


def _render_clause_section(
    clause: dict[str, Any],
    cnum: str,
    obs_start: int,
    taxonomy_label: str,
) -> str:
    """Render one clause block with its observations."""
    lines: list[str] = []
    title = html_lib.escape(clause.get("title", ""))
    position = clause_stance(clause)
    pos_color = _POSITION_COLORS.get(position, "#374151")
    confidence = clause_confidence(clause)
    conf_score = confidence.get("score")
    n_our = confidence.get("n_our_paper")
    n_cp = confidence.get("n_counterparty_paper")
    clause_id = clause.get("id", "")

    conf_str = ""
    if conf_score is not None:
        conf_str = f"confidence {conf_score:.0%}"
    if n_our is not None:
        conf_str += f" | n_our={n_our}"
    if n_cp is not None:
        conf_str += f" | n_cp={n_cp}"

    our_standard = clause.get("our_standard") or {}
    our_std_text = our_standard.get("text", "")

    # Needs-review or low-confidence highlight
    needs_review = position in ("needs_review",) or (conf_score is not None and conf_score < 0.5)
    highlight_style = "border-left:4px solid #f59e0b;background:#fffbeb" if needs_review else ""

    lines.append(f'<div class="clause" id="{html_lib.escape(cnum)}" style="{highlight_style}">')
    lines.append(
        f'<div class="clause-header">'
        f'<span class="item-num">{html_lib.escape(cnum)}</span> '
        f'<span class="clause-title">{title}</span> '
        f'<span class="taxonomy-tag">{html_lib.escape(taxonomy_label)}</span> '
        f'<span style="color:{pos_color};font-weight:700">{html_lib.escape(position)}</span>'
        f"</div>"
    )
    if conf_str:
        lines.append(f'<div class="clause-meta">{html_lib.escape(conf_str)}</div>')

    if our_std_text:
        lines.append(
            f'<div class="our-standard">'
            f"<strong>Our standard:</strong> {html_lib.escape(our_std_text)}"
            f"</div>"
        )

    # Comment / override for the clause item
    lines.append(
        f'<div class="feedback-row">'
        f'<label>Comment: <input class="comment-input" data-item="{html_lib.escape(cnum)}" '
        f'data-clause-id="{html_lib.escape(clause_id)}" type="text" '
        f'placeholder="Reviewer note…" style="width:50%"></label> '
        f"<label>Pin position: "
        f'<select class="override-select" data-item="{html_lib.escape(cnum)}" '
        f'data-clause-id="{html_lib.escape(clause_id)}">'
        f'<option value="">—</option>'
        # Values mirror OPF v0.2 summary.historical_stance (issue #155) so a
        # pin is directly comparable to the recomputed stance on recompile
        # (issue #147) rather than a different, incompatible vocabulary.
        f'<option value="consistently_held">consistently_held</option>'
        f'<option value="usually_held">usually_held</option>'
        f'<option value="mixed">mixed</option>'
        f'<option value="usually_conceded">usually_conceded</option>'
        f'<option value="no_signal">no_signal</option>'
        f"</select></label>"
        f"</div>"
    )

    # Observations
    obs_list = clause.get("observed_positions", [])
    if obs_list:
        lines.append('<div class="observations">')
        for i, obs in enumerate(obs_list, start=1):
            onum = f"{cnum}.{i}"
            text_summary = obs.get("text_summary", "")
            deviation = obs.get("deviation", "")
            risk_delta = obs.get("risk_delta", {})
            provenance = obs.get("provenance", "")
            outcome = obs.get("outcome", "")
            example_ref = obs.get("example_ref")
            citation = _citation_str(example_ref)

            lines.append(
                f'<div class="observation" id="{html_lib.escape(onum)}">'
                f'<span class="item-num obs-num">{html_lib.escape(onum)}</span> '
                f"{html_lib.escape(text_summary)} "
                f"{_deviation_badge(deviation)} "
                f"{_risk_badge(risk_delta)} "
                f"{_outcome_badge(outcome)} "
                f'<span class="prov-tag">{html_lib.escape(provenance)}</span>'
            )
            if citation:
                lines.append(
                    f'<span class="citation-tag" title="{html_lib.escape(citation)}">'
                    f"{html_lib.escape(citation)}"
                    f"</span>"
                )
            lines.append(
                f'<div class="feedback-row">'
                f'<label>Comment: <input class="comment-input" data-item="{html_lib.escape(onum)}" '
                f'type="text" placeholder="Note on this observation…" style="width:50%"></label>'
                f"</div>"
            )
            lines.append("</div>")  # .observation
        lines.append("</div>")  # .observations

    lines.append("</div>")  # .clause
    return "\n".join(lines)


_CSS = """
body { font-family: system-ui, sans-serif; margin: 0; padding: 0; background: #f9fafb; color: #111 }
h1 { font-size: 1.4rem; margin: 0 0 0.5rem }
.toolbar { background: #1e293b; color: #f1f5f9; padding: 0.8rem 1.2rem; display: flex; align-items: center; gap: 1rem }
.toolbar button { background: #3b82f6; color: #fff; border: none; border-radius: 4px; padding: 0.4rem 1rem; cursor: pointer; font-weight: 600 }
.toolbar button:hover { background: #2563eb }
#toc { background: #fff; border-right: 1px solid #e5e7eb; padding: 1rem; min-width: 160px; max-width: 220px; overflow-y: auto; position: sticky; top: 0; height: 100vh; flex-shrink: 0 }
#toc a { display: block; padding: 2px 0; color: #374151; text-decoration: none; font-size: 0.85rem }
#toc a:hover { color: #2563eb }
#content { flex: 1; padding: 1.2rem; overflow-y: auto; max-width: 900px }
.layout { display: flex; gap: 0 }
.clause { background: #fff; border: 1px solid #e5e7eb; border-radius: 6px; margin-bottom: 1rem; padding: 1rem 1.2rem }
.clause-header { font-size: 1rem; font-weight: 600; margin-bottom: 0.3rem }
.item-num { background: #1e293b; color: #f1f5f9; border-radius: 3px; padding: 1px 6px; font-size: 0.8rem; font-family: monospace }
.obs-num { background: #6b7280 !important }
.clause-title { font-size: 1.05rem }
.taxonomy-tag { background: #ede9fe; color: #5b21b6; border-radius: 3px; padding: 1px 5px; font-size: 0.78rem; margin-left: 0.3rem }
.clause-meta { color: #6b7280; font-size: 0.83rem; margin-bottom: 0.4rem }
.our-standard { background: #f0fdf4; border-left: 3px solid #86efac; padding: 0.4rem 0.8rem; margin: 0.4rem 0; font-size: 0.9rem }
.observations { margin-top: 0.6rem; padding-left: 1rem; border-left: 2px solid #e5e7eb }
.observation { margin-bottom: 0.5rem; padding: 0.4rem 0.5rem; background: #f9fafb; border-radius: 4px; font-size: 0.9rem }
.prov-tag { background: #dbeafe; color: #1e40af; border-radius: 3px; padding: 1px 5px; font-size: 0.78rem; margin-left: 0.3rem }
.citation-tag { background: #f3f4f6; color: #374151; border-radius: 3px; padding: 1px 5px; font-size: 0.78rem; margin-left: 0.3rem; font-family: monospace }
.feedback-row { margin-top: 0.4rem; font-size: 0.85rem; color: #374151 }
.feedback-row input, .feedback-row select { font-size: 0.85rem; padding: 2px 6px; border: 1px solid #d1d5db; border-radius: 3px }
#toc-taxonomy h4 { font-size: 0.8rem; text-transform: uppercase; color: #9ca3af; margin: 0.6rem 0 0.1rem }
"""

_JS = r"""
function collectFeedback() {
  var fb = {};
  document.querySelectorAll('.comment-input').forEach(function(el) {
    var item = el.getAttribute('data-item');
    if (!fb[item]) fb[item] = {};
    if (el.value.trim()) fb[item].comment = el.value.trim();
  });
  document.querySelectorAll('.override-select').forEach(function(el) {
    var item = el.getAttribute('data-item');
    if (el.value) {
      if (!fb[item]) fb[item] = {};
      fb[item].override = el.value;
    }
  });
  return fb;
}

function exportFeedback() {
  var fb = collectFeedback();
  var blob = new Blob([JSON.stringify(fb, null, 2)], {type: 'application/json'});
  var a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'feedback.json';
  a.click();
}
"""


# ---------------------------------------------------------------------------
# Alias -> real name resolution at render time (issue #146)
# ---------------------------------------------------------------------------


def load_alias_map(path: Path) -> dict[str, str]:
    """Load an ``alias -> real entity name`` held-out map from *path*.

    *path* is typically ``<out_dir>/alias_map.json`` as written by
    ``entity_registry.write_holdout_map``. Raises ``FileNotFoundError`` if
    *path* does not exist — callers should treat that as "no map available",
    not silently render with aliases unresolved (issue #138's silent-success
    concern applies here too).
    """
    raw = path.read_text(encoding="utf-8")
    data: dict[str, str] = json.loads(raw)
    return data


def _resolve_aliases_in_value(value: Any, patterns: list[tuple[re.Pattern[str], str]]) -> Any:
    """Recursively substitute every alias occurrence in *value* with its real name."""
    if isinstance(value, str):
        for pattern, real_name in patterns:
            value = pattern.sub(real_name, value)
        return value
    if isinstance(value, dict):
        return {k: _resolve_aliases_in_value(v, patterns) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_aliases_in_value(v, patterns) for v in value]
    return value


def _resolve_aliases_in_doc(doc: dict[str, Any], alias_map: dict[str, str]) -> dict[str, Any]:
    """Return a deep copy of *doc* with every alias resolved to its real name.

    Longest aliases are matched first (mirrors
    ``entity_registry.pseudonymize_text``'s longest-match-first rule) so a
    shorter alias that happens to prefix a longer one never partially
    shadows it. Matching is whole-word (``\\b``-bounded) — the same
    boundary convention used going the other direction at ingest.
    """
    if not alias_map:
        return doc
    patterns = [
        (re.compile(r"\b" + re.escape(alias) + r"\b"), real_name)
        for alias, real_name in sorted(alias_map.items(), key=lambda kv: len(kv[0]), reverse=True)
    ]
    resolved: dict[str, Any] = _resolve_aliases_in_value(doc, patterns)
    return resolved


def render_review_html(
    out_dir: Path, out_file: Path | None = None, alias_map: dict[str, str] | None = None
) -> str:
    """Render a self-contained HTML review page from ``playbook.opf.json``.

    Groups clauses by taxonomy, numbers them ``C1``, ``C1.1``, etc.  Embeds
    the full playbook JSON in a ``<script>`` tag for drill-down access.  Adds a
    per-item comment box and an **Export feedback** button that produces
    ``feedback.json``.

    Args:
        out_dir:   Path to the directory containing ``playbook.opf.json``.
        out_file:  If given, write the HTML to this file atomically; the HTML
                   string is still returned regardless.
        alias_map: Optional ``alias -> real entity name`` held-out map (issue
                   #146). When given, every alias in the rendered HTML
                   (titles, summaries, citations, the embedded drill-down
                   JSON, ...) is resolved to its real name. ``playbook.opf.json``
                   on disk is unaffected either way — see the module docstring.

    Returns:
        Self-contained HTML string (no external network requests required).

    Raises:
        FileNotFoundError: If ``playbook.opf.json`` does not exist in
                           *out_dir*.
    """
    opf_path = out_dir / "playbook.opf.json"
    if not opf_path.exists():
        raise FileNotFoundError(f"playbook.opf.json not found in {out_dir}")

    raw = opf_path.read_text(encoding="utf-8")
    doc: dict[str, Any] = json.loads(raw)

    # Resolve aliases -> real names for rendering ONLY (issue #146). `doc`
    # (used for every rendering step below, including the embedded
    # drill-down JSON) is swapped for the resolved copy; `raw` — the exact
    # on-disk bytes — is left untouched, and never re-serialized unless a
    # map was supplied.
    render_doc = _resolve_aliases_in_doc(doc, alias_map) if alias_map else doc
    embedded_json = json.dumps(render_doc, indent=2, ensure_ascii=False) if alias_map else raw

    index = _build_index(render_doc)

    # Build taxonomy label map
    tax_map: dict[str, str] = {}
    for entry in render_doc.get("taxonomy", {}).get("entries", []):
        tax_map[entry.get("id", "")] = entry.get("label", entry.get("id", ""))

    # Group by taxonomy
    from collections import defaultdict  # noqa: PLC0415

    grouped: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    for item_num, kind, payload in index:
        if kind == "clause":
            tid = payload.get("taxonomy_id", "")
            grouped[tid].append((item_num, payload))

    # Build TOC HTML
    toc_lines = ['<div id="toc"><h3 style="margin:0 0 0.5rem">Index</h3>']
    for tid, items in grouped.items():
        tax_label = tax_map.get(tid, tid)
        toc_lines.append(f"<h4>{html_lib.escape(tax_label)}</h4>")
        for item_num, clause in items:
            clause_title = html_lib.escape(clause.get("title", ""))
            toc_lines.append(
                f'<a href="#{html_lib.escape(item_num)}">'
                f"{html_lib.escape(item_num)} {clause_title}</a>"
            )
    toc_lines.append("</div>")
    toc_html = "\n".join(toc_lines)

    # Build clause sections
    clause_sections: list[str] = []
    clause_counter = 0
    for item_num, kind, payload in index:
        if kind != "clause":
            continue
        clause_counter += 1
        tid = payload.get("taxonomy_id", "")
        tax_label = tax_map.get(tid, tid)
        section = _render_clause_section(payload, item_num, clause_counter, tax_label)
        clause_sections.append(section)

    clauses_html = "\n".join(clause_sections)

    agreement_type = render_doc.get("agreement_type", {}).get("name", "Playbook Review")
    generated_at = render_doc.get("compiler", {}).get("generated_at", "")

    html_parts: list[str] = [
        "<!DOCTYPE html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="UTF-8">',
        f"<title>{html_lib.escape(agreement_type)} — Playbook Review</title>",
        f"<style>{_CSS}</style>",
        "</head>",
        "<body>",
        '<div class="toolbar">',
        f"<h1>{html_lib.escape(agreement_type)} — Playbook Review</h1>",
        f'<span style="font-size:0.8rem;color:#94a3b8">{html_lib.escape(generated_at)}</span>',
        '<button onclick="exportFeedback()">Export feedback</button>',
        "</div>",
        '<div class="layout">',
        toc_html,
        f'<div id="content">{clauses_html}</div>',
        "</div>",
        # Embedded playbook JSON for drill-down / JS access
        f'<script id="playbook-data" type="application/json">\n{embedded_json}\n</script>',
        f"<script>\n{_JS}\n</script>",
        "</body>",
        "</html>",
    ]

    html_str = "\n".join(html_parts)

    if out_file is not None:
        tmp = out_file.with_suffix(".tmp")
        tmp.write_text(html_str, encoding="utf-8")
        tmp.replace(out_file)

    return html_str


# ---------------------------------------------------------------------------
# Feedback application
# ---------------------------------------------------------------------------


def apply_feedback(out_dir: Path, feedback_path: Path) -> ApplyResult:
    """Translate reviewer feedback into engine-actionable corrections.

    Reads *feedback_path* (``feedback.json`` produced by the HTML viewer)
    and applies corrections to the out-directory:

    - ``provenance`` / ``signed_version`` / ``order`` → per-document
      ``hints.yaml`` entries alongside the corpus document.
    - ``classification`` → ``VerdictStore`` entry at
      ``out_dir/judge/verdicts.jsonl``.
    - ``note`` → appended to ``out_dir/viewer_notes.md``.

    Args:
        out_dir:       Directory containing ``playbook.opf.json`` and corpus
                       staging (to locate ``hints.yaml`` files).
        feedback_path: Path to the ``feedback.json`` file produced by the
                       viewer's Export button.

    Returns:
        ``ApplyResult`` summarising what was written.

    Raises:
        FileNotFoundError: If ``playbook.opf.json`` does not exist.
        ValueError: If *feedback_path* contains invalid JSON.
    """
    opf_path = out_dir / "playbook.opf.json"
    if not opf_path.exists():
        raise FileNotFoundError(f"playbook.opf.json not found in {out_dir}")

    raw = opf_path.read_text(encoding="utf-8")
    doc: dict[str, Any] = json.loads(raw)

    try:
        feedback_raw = feedback_path.read_text(encoding="utf-8")
        feedback: dict[str, Any] = json.loads(feedback_raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {feedback_path}: {exc}") from exc

    index = _build_index(doc)
    # Build item_num → (kind, payload) map
    item_map: dict[str, tuple[str, dict[str, Any]]] = {
        item_num: (kind, payload) for item_num, kind, payload in index
    }

    result = ApplyResult()

    # Accumulate hints by document_id
    hints_by_doc: dict[str, dict[str, Any]] = {}
    # Accumulate VerdictStore entries: list of (payload, verdict) pairs
    verdict_entries: list[tuple[dict[str, Any], dict[str, Any]]] = []
    # Accumulate notes
    notes: list[str] = []
    # Accumulate curation pins by clause_id (issue #147) — a single run of
    # feedback should only ever produce one pin per clause even if somehow
    # cited from multiple item numbers.
    pins_by_clause_id: dict[str, CurationPin] = {}
    pinned_at = datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")

    for item_num, corrections in feedback.items():
        if not isinstance(corrections, dict):
            continue

        kind_payload = item_map.get(item_num)
        if kind_payload is None:
            _log.warning("apply_feedback: unknown item number %s; skipping", item_num)
            continue

        kind, payload = kind_payload

        # --- Unsupported keys: report, don't silently drop (issue #138) ------
        unsupported_keys = [k for k in corrections if k not in _RECOGNIZED_FEEDBACK_KEYS]
        if unsupported_keys:
            result.skipped.setdefault(item_num, []).extend(
                _unsupported_message(k) for k in unsupported_keys
            )

        # --- Document-level hint corrections (from observation level) --------
        # For observation-level items, the cited document is in example_ref.
        # For clause-level items, we look at the first observation's example_ref.
        cited_doc_id: str | None = None
        if kind == "observation":
            example_ref = payload.get("example_ref") or {}
            cited_doc_id = example_ref.get("document_id")
        elif kind == "clause":
            # Use first observation for document reference
            obs_list = payload.get("observed_positions", [])
            if obs_list:
                cited_doc_id = (obs_list[0].get("example_ref") or {}).get("document_id")

        # provenance override
        if "provenance" in corrections and cited_doc_id:
            if cited_doc_id not in hints_by_doc:
                hints_by_doc[cited_doc_id] = {}
            hints_by_doc[cited_doc_id]["provenance"] = corrections["provenance"]

        # signed_version override
        if "signed_version" in corrections and cited_doc_id:
            if cited_doc_id not in hints_by_doc:
                hints_by_doc[cited_doc_id] = {}
            hints_by_doc[cited_doc_id]["signed_version"] = corrections["signed_version"]

        # order override
        if "order" in corrections and cited_doc_id:
            if cited_doc_id not in hints_by_doc:
                hints_by_doc[cited_doc_id] = {}
            hints_by_doc[cited_doc_id]["order"] = corrections["order"]

        # --- Classification correction ----------------------------------------
        # Rebuild the EXACT payload key that StoreBackedClassificationJudge
        # hashes (stage + FULL clause text + node heading + sorted taxonomy ids),
        # sourcing the full text from the normalized clause trees the judge
        # actually classified. An earlier version wrote text="" (the OPF only
        # carries a truncated summary), which produced a key that never matched —
        # so the correction was a silent no-op (issue #70).
        if "classification" in corrections:
            new_tid = corrections["classification"]
            # Resolve the parent clause for this item.
            if kind == "clause":
                clause_obj: dict[str, Any] | None = payload
            else:
                clause_id = payload.get("_clause_id")
                clause_obj = next(
                    (p for _, k, p in index if k == "clause" and p.get("id") == clause_id),
                    None,
                )

            if clause_obj is not None:
                # Collect (document_id, clause_path) for every underlying node:
                # all observed positions, plus our_standard if it cites a corpus
                # document (the template is not written to normalized/).
                cited: set[tuple[str, str]] = set()
                for obs in clause_obj.get("observed_positions", []):
                    ref = obs.get("example_ref") or {}
                    cdoc, cpath = ref.get("document_id"), ref.get("clause_path")
                    if cdoc and cpath:
                        cited.add((cdoc, cpath))
                std_ref = (clause_obj.get("our_standard") or {}).get("source_ref") or {}
                if (
                    std_ref.get("document_id")
                    and std_ref.get("clause_path")
                    and std_ref.get("version") != "template"
                ):
                    cited.add((std_ref["document_id"], std_ref["clause_path"]))

                tax_ids = sorted(
                    e.get("id", "") for e in doc.get("taxonomy", {}).get("entries", [])
                )
                # The same clause_path can appear across versions with different
                # text; override every distinct node so the correction lands
                # whichever version the judge classifies.
                for cdoc, cpath in sorted(cited):
                    norm_dir = out_dir / "normalized" / cdoc
                    for tree_path in sorted(norm_dir.glob("*.clauses.json")):
                        try:
                            tree = ClauseTree.load(tree_path)
                        except Exception:  # noqa: BLE001
                            continue
                        node = tree.resolve_path(cpath)
                        if node is None:
                            continue
                        judge_payload = {
                            "stage": "classify",
                            "text": node.text or "",
                            "heading": node.heading or "",
                            "taxonomy_ids": tax_ids,
                        }
                        verdict_entries.append(
                            (
                                judge_payload,
                                # basis MUST be "judge": classify_tree rejects any
                                # other basis returned by a ClassificationJudge.
                                {"taxonomy_id": new_tid, "confidence": 1.0, "basis": "judge"},
                            )
                        )
                if not cited:
                    _log.warning(
                        "apply_feedback: classification correction for %s has no corpus "
                        "citation; cannot map to a clause node — skipping",
                        item_num,
                    )

        # --- Position override -> embedded curation pin (issue #147) ---------
        # Resolves to the parent clause the same way "classification" does
        # above. baseline_stance records THIS clause's historical_stance
        # right now — what the attorney is overriding FROM — so a later
        # recompile can flag a conflict only when the evidence-driven stance
        # actually moves, not merely because it differs from the pin.
        if "override" in corrections:
            if kind == "clause":
                override_clause_obj: dict[str, Any] | None = payload
            else:
                override_clause_id = payload.get("_clause_id")
                override_clause_obj = next(
                    (p for _, k, p in index if k == "clause" and p.get("id") == override_clause_id),
                    None,
                )
            if override_clause_obj is not None and override_clause_obj.get("id"):
                comment_val = corrections.get("comment") or corrections.get("note")
                pins_by_clause_id[override_clause_obj["id"]] = CurationPin(
                    clause_id=override_clause_obj["id"],
                    item_id=item_num,
                    position=str(corrections["override"]),
                    baseline_stance=clause_stance(override_clause_obj),
                    pinned_at=pinned_at,
                    comment=str(comment_val).strip() if comment_val else None,
                )
            else:
                _log.warning(
                    "apply_feedback: override correction for %s has no resolvable clause; skipping",
                    item_num,
                )

        # --- Free-text note / comment -----------------------------------------
        # Both route to the same viewer_notes.md sink. "comment" is what the
        # HTML viewer's Export feedback button actually produces; "note" is
        # kept for callers that build feedback.json directly (see docstring).
        for text_key in ("comment", "note"):
            text_val = corrections.get(text_key, "")
            if text_val and str(text_val).strip():
                clause_title = payload.get("title") or payload.get("_clause_title", "")
                notes.append(f"**{item_num}** ({clause_title}): {str(text_val).strip()}")

    # Write hints.yaml files
    # Locate document directories: scan out_dir and one level up for doc dirs
    # The hints.yaml lives alongside the corpus document folder.
    # Convention from version_orderer.py: hints live at <corpus_dir>/<doc_id>/hints.yaml
    # We search for a hints.yaml sibling folder relative to out_dir's parent.
    hints_written: list[str] = []
    for doc_id, hint_updates in hints_by_doc.items():
        hints_path = _find_hints_path(out_dir, doc_id)
        if hints_path is None:
            # Write into out_dir/hints/<doc_id>.yaml as a fallback
            hints_path = out_dir / "hints" / f"{doc_id}.yaml"
        hints_path.parent.mkdir(parents=True, exist_ok=True)

        # Merge with existing hints
        existing: dict[str, Any] = {}
        if hints_path.exists():
            try:
                existing = yaml.safe_load(hints_path.read_text(encoding="utf-8")) or {}
            except Exception:  # noqa: BLE001
                existing = {}
        existing.update(hint_updates)

        hints_path.write_text(
            yaml.dump(existing, allow_unicode=True, sort_keys=True), encoding="utf-8"
        )
        hints_written.append(doc_id)

    result.hints_written = hints_written

    # Write VerdictStore entries (deduplicated by content key — the same clause
    # node can be reached via multiple citations / versions).
    if verdict_entries:
        verdicts_path = out_dir / "judge" / "verdicts.jsonl"
        store = VerdictStore(verdicts_path)
        seen_sigs: set[str] = set()
        for judge_payload, verdict in verdict_entries:
            sig = json.dumps(judge_payload, sort_keys=True, ensure_ascii=False)
            if sig in seen_sigs:
                continue
            seen_sigs.add(sig)
            store.put(judge_payload, verdict)
        result.verdicts_written = len(seen_sigs)

    # Write notes
    if notes:
        notes_path = out_dir / "viewer_notes.md"
        existing_notes = ""
        if notes_path.exists():
            existing_notes = notes_path.read_text(encoding="utf-8")
        new_notes = "\n".join(f"- {n}" for n in notes)
        combined = f"{existing_notes}\n{new_notes}\n" if existing_notes else f"{new_notes}\n"
        notes_path.write_text(combined, encoding="utf-8")
        result.notes_written = True

    # Write curation pins — EMBEDDED directly in playbook.opf.json, not a
    # sidecar (issue #147). Merges into any existing curation.pins by
    # clause_id (a later pin on the same clause replaces the earlier one,
    # clearing whatever conflict it may have carried — the attorney is
    # re-asserting the position now).
    if pins_by_clause_id:
        existing_pins = {p["clause_id"]: p for p in (doc.get("curation") or {}).get("pins", [])}
        pins_written: list[str] = []
        for clause_id, pin in pins_by_clause_id.items():
            existing_pins[clause_id] = pin.to_dict()
            pins_written.append(pin.item_id)
        doc["curation"] = {"pins": list(existing_pins.values())}

        # Refresh identity so it never goes stale after a curation-only edit.
        # content_hash itself is unaffected (curation is excluded from it —
        # see canonicalize.py) but section_digests.curation must track the
        # new pin content. Only touched if this document already carries an
        # identity block (v0.1 fixtures / playbooks without one are left as-is).
        if "identity" in doc:
            doc["identity"]["content_hash"] = content_hash(doc)
            doc["identity"]["section_digests"] = compute_section_digests(doc)

        write_playbook(doc, opf_path)
        result.pins_written = pins_written

    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _find_hints_path(out_dir: Path, doc_id: str) -> Path | None:
    """Search for an existing or natural hints.yaml location for *doc_id*.

    Looks in:
    1. ``out_dir/../<doc_id>/hints.yaml`` (standard corpus layout)
    2. ``out_dir/../../<doc_id>/hints.yaml`` (two-level layout)

    Returns the path to use, or ``None`` if no canonical location is found.
    """
    for parent in (out_dir.parent, out_dir.parent.parent):
        candidate = parent / doc_id / "hints.yaml"
        if candidate.parent.exists():
            return candidate
    return None
