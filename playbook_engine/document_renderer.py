"""Presentational HTML rendering of a compiled playbook — ``view bundle``.

Companion to :mod:`playbook_engine.viewer` with a different job: where
``view render`` produces the *review/annotation surface* (numbered items,
comment boxes, feedback export), this module produces the *readable
playbook* — a typographic, print-friendly document a lawyer can read top to
bottom or hand to a stakeholder.

The readable document is no longer an artifact of its own. It ships inside
the single-file bundle ``playbook.opf.html`` (:func:`render_bundle_html`),
which is the one shareable/uploadable HTML: the same document body, plus a
digest summary, plus the canonical OPF JSON and digest embedded as
machine-readable ``<script type="application/json">`` blocks. The bare
``playbook.opf.json`` remains the canonical source of truth on disk.
:func:`render_document_html` stays as the internal document renderer the
bundle composes; it is no longer exposed as its own CLI command (the former
``playbook view document`` deprecated alias was removed — use ``view
bundle``).

Both renderings are built by :func:`_render_document_page` from a parsed OPF
document, with the bundle passing its extra markup through that function's
explicit seams — so the bundle contains the whole document by construction,
not by string-splicing rendered template text.

Sections rendered per clause: historical stance + confidence, the
``acceptable_if`` variations, fallback positions, rejected asks (collapsed),
and accepted exemplar forms from the clause library. Empty Posture/Floor
sections render as an explicit "pending GC interview" note rather than being
omitted — same honesty-first convention as the after-action report.

Alias handling: the document body shows the born-safe ``Counterparty-N``
aliases exactly as stored in ``playbook.opf.json``. The bundle takes no alias
map by design — it embeds the canonical pseudonymized JSON, so resolving real
names would leak and break hash verification. Internal-eyes review with real
names lives in ``view render --alias-map``. The on-disk OPF is never modified.
"""

from __future__ import annotations

import html as html_lib
import json
from pathlib import Path
from typing import Any

from playbook_engine.opf_accessors import (
    clause_confidence,
    clause_stance,
    playbook_clauses,
)
from playbook_engine.viewer import _resolve_aliases_in_doc

_STANCE_STYLES: dict[str, tuple[str, str]] = {
    # stance -> (background, foreground)
    "consistently_held": ("#dcfce7", "#14532d"),
    "usually_held": ("#dcfce7", "#166534"),
    "mixed": ("#fef9c3", "#713f12"),
    "usually_conceded": ("#fee2e2", "#7f1d1d"),
    "no_signal": ("#e5e7eb", "#374151"),
    "unknown": ("#e5e7eb", "#374151"),
}

_RISK_GLYPHS = {"worse": "▲", "better": "▼", "neutral": "–"}


def _chip(text: str, bg: str, fg: str) -> str:
    return (
        f'<span style="background:{bg};color:{fg};border-radius:999px;'
        f"padding:2px 10px;font-size:0.78rem;font-weight:600;"
        f'letter-spacing:0.02em">{html_lib.escape(text)}</span>'
    )


_STANCE_HELP = {
    "consistently_held": "Corpus shows we held our language every time this was contested.",
    "usually_held": "Corpus shows we held our language in most contested deals.",
    "mixed": "Corpus shows BOTH concessions and successful pushbacks for this clause type.",
    "usually_conceded": "Corpus shows we have conceded this clause when contested.",
    "no_signal": "Not enough our-paper evidence (or no standard to measure against) to characterise a stance.",
    "unknown": "Stance could not be derived.",
}


def _stance_chip(stance: str) -> str:
    bg, fg = _STANCE_STYLES.get(stance, _STANCE_STYLES["unknown"])
    help_text = _STANCE_HELP.get(stance, _STANCE_HELP["unknown"])
    return (
        f'<span title="{html_lib.escape(help_text)}" '
        f'style="background:{bg};color:{fg};border-radius:999px;'
        f"padding:2px 10px;font-size:0.78rem;font-weight:600;cursor:help;"
        f'letter-spacing:0.02em">{html_lib.escape(stance.replace("_", " "))}</span>'
    )


def _risk_str(risk: dict[str, Any] | None) -> str:
    if not isinstance(risk, dict):
        return ""
    direction = str(risk.get("direction", ""))
    magnitude = str(risk.get("magnitude", ""))
    glyph = _RISK_GLYPHS.get(direction, "")
    return f"{glyph} {direction}/{magnitude}".strip()


def _cite_str(ref: dict[str, Any] | None) -> str:
    if not isinstance(ref, dict):
        return ""
    doc_id = str(ref.get("document_id", ""))
    version = ref.get("version", "")
    clause_path = str(ref.get("clause_path", ""))
    return f"{doc_id} · v{version} · §{clause_path}"


def _quote_block(text: str, cite: str = "") -> str:
    body = html_lib.escape(text)
    cite_html = f'<div class="cite">{html_lib.escape(cite)}</div>' if cite else ""
    return f"<blockquote>{body}{cite_html}</blockquote>"


def _render_clause(
    clause: dict[str, Any],
    library_by_tid: dict[str, dict[str, Any]],
    tax_labels: dict[str, str],
    number: int,
) -> str:
    tid = str(clause.get("taxonomy_id", ""))
    title = clause.get("title") or tax_labels.get(tid, tid)
    stance = clause_stance(clause)
    confidence = clause_confidence(clause)
    summary = clause.get("summary") if isinstance(clause.get("summary"), dict) else {}
    assert isinstance(summary, dict)

    parts: list[str] = [f'<section class="clause" id="clause-{number}">']
    parts.append(
        f'<h2><span class="cnum">{number}.</span> {html_lib.escape(str(title))} '
        f"{_stance_chip(stance)}</h2>"
    )

    meta_bits: list[str] = [f"taxonomy: {html_lib.escape(tax_labels.get(tid, tid))}"]
    score = confidence.get("score")
    if isinstance(score, (int, float)):
        meta_bits.append(f"confidence {score:.0%}")
    n_our = confidence.get("n_our_paper")
    n_cp = confidence.get("n_counterparty_paper")
    if n_our is not None or n_cp is not None:
        meta_bits.append(f"evidence n_our={n_our} n_counterparty={n_cp}")
    stance_detail = summary.get("stance_detail")
    if isinstance(stance_detail, dict) and stance_detail:
        held = stance_detail.get("held")
        of = stance_detail.get("of")
        if held is not None and of is not None:
            meta_bits.append(f"held {held} of {of}")
    parts.append(
        f'<p class="meta" title="confidence = evidence-depth score; n_our / '
        f"n_counterparty = observations from our paper vs counterparty paper; "
        f"held X of Y = contested deals where our language survived to "
        f'signature">{" · ".join(meta_bits)}</p>'
    )

    our_standard = clause.get("our_standard") or {}
    std_text = our_standard.get("text") if isinstance(our_standard, dict) else None
    if std_text:
        parts.append("<h3>Our standard</h3>")
        parts.append(_quote_block(str(std_text)))

    acceptable = summary.get("acceptable_if") or []
    if acceptable:
        parts.append(
            '<h3 title="Negotiated changes we signed at neutral or equivalent '
            "risk. Precedent says: take these without escalation. (OPF field: "
            'summary.acceptable_if)">Preferred variations</h3>'
        )
        for entry in acceptable:
            if not isinstance(entry, dict):
                continue
            parts.append('<div class="variation">')
            if entry.get("if"):
                parts.append(f'<p class="var-label">From</p>{_quote_block(str(entry["if"]))}')
            if entry.get("to"):
                parts.append(
                    f'<p class="var-label">Acceptable as</p>{_quote_block(str(entry["to"]))}'
                )
            if entry.get("rationale"):
                parts.append(f'<p class="rationale">{html_lib.escape(str(entry["rationale"]))}</p>')
            parts.append("</div>")

    fallbacks = summary.get("fallbacks") or []
    if fallbacks:
        parts.append(
            '<h3 title="Forms we have historically signed even though they moved '
            "risk against us (see the risk marker on each). Concessions you can "
            "live with when pressed — not first asks. (OPF field: "
            'summary.fallbacks)">Acceptable variations — concessions</h3><ul>'
        )
        for fb in fallbacks:
            if isinstance(fb, dict):
                text = fb.get("text_summary") or fb.get("text") or json.dumps(fb)
                risk = _risk_str(fb.get("risk_delta"))
                suffix = (
                    f' <span class="risk" title="Judged risk shift from our '
                    f"perspective: direction (worse = more risk for us) / "
                    f'magnitude (minor or material)">{html_lib.escape(risk)}</span>'
                    if risk
                    else ""
                )
                parts.append(f"<li>{html_lib.escape(str(text))}{suffix}</li>")
            else:
                parts.append(f"<li>{html_lib.escape(str(fb))}</li>")
        parts.append("</ul>")

    rejected = summary.get("rejected") or []
    if rejected:
        parts.append(
            f'<details><summary title="Counterparty asks that appeared in a draft '
            f"and were reversed or removed before signing — historically refused. "
            f'Use as pushback precedent. (OPF field: summary.rejected)">'
            f"Unacceptable variations — rejected/reversed asks ({len(rejected)})</summary><ul>"
        )
        for r in rejected:
            if not isinstance(r, dict):
                continue
            text = r.get("text_summary") or r.get("full_text") or ""
            risk = _risk_str(r.get("risk_delta"))
            suffix = f' <span class="risk">{html_lib.escape(risk)}</span>' if risk else ""
            parts.append(f"<li>{html_lib.escape(str(text)[:400])}{suffix}</li>")
        parts.append("</ul></details>")

    library_entry = library_by_tid.get(tid)
    accepted_forms = (library_entry or {}).get("accepted_forms") or []
    if accepted_forms:
        parts.append(
            f'<details><summary title="The evidence library: every distinct final '
            f"form of this clause across the signed corpus, with citations — "
            f"including forms that were never negotiated. The variation sections "
            f"above are distilled from the negotiated subset of these, so entries "
            f'overlap by design.">All signed forms — evidence library '
            f"({len(accepted_forms)})</summary>"
        )
        for form in accepted_forms:
            if not isinstance(form, dict):
                continue
            parts.append(
                _quote_block(
                    str(form.get("text_summary", "")),
                    _cite_str(form.get("example_ref")),
                )
            )
        parts.append("</details>")

    parts.append("</section>")
    return "\n".join(parts)


def _render_method_panel(doc: dict[str, Any], clauses: list[dict[str, Any]]) -> str:
    """The "Method & provenance" panel: how this document was built, from the
    document's own numbers — so "where did this come from?" is answerable
    without leaving the page. Every figure is computed from the OPF itself
    (no side files), which keeps the panel honest under recompiles.
    """
    corpus = doc.get("corpus", {})
    compiler = doc.get("compiler", {})
    identity = doc.get("identity", {})
    docs = corpus.get("documents", [])
    stats = corpus.get("stats", {})

    n_total = stats.get("documents_total", len(docs))
    n_in_scope = stats.get("documents_in_scope", sum(1 for d in docs if d.get("in_scope")))
    n_versions = stats.get("versions_total", "—")
    n_our = sum(1 for d in docs if d.get("provenance") == "our_paper")
    n_cp = sum(1 for d in docs if d.get("provenance") == "counterparty_paper")
    n_excluded = n_total - n_in_scope if isinstance(n_total, int) else "—"

    # version_ingest is a LIST of per-version records in compiled documents
    # (a dict keyed by version id in some older fixtures) — accept both.
    def _ingest_records(d: dict[str, Any]) -> list[Any]:
        vi = d.get("version_ingest")
        if isinstance(vi, dict):
            return list(vi.values())
        if isinstance(vi, list):
            return vi
        return []

    failed_versions = sum(
        1
        for d in docs
        for v in _ingest_records(d)
        if isinstance(v, dict) and v.get("status") == "failed"
    )
    unclassified = (stats.get("unclassified") or {}).get("count", 0)

    dev_counts: dict[str, int] = {}
    n_obs = 0
    for clause in clauses:
        for obs in clause.get("observed_positions", []):
            n_obs += 1
            key = str(obs.get("deviation", "?"))
            dev_counts[key] = dev_counts.get(key, 0) + 1
    dev_line = ", ".join(
        f"{dev_counts[k]} {k.replace('_', ' ')}"
        for k in ("none", "reworded_equivalent", "substantive")
        if k in dev_counts
    )

    judge_line = (
        "structural stages are deterministic; scope, provenance, and "
        "deviation/risk judgments were made by an LLM judge, each stored with "
        "its rationale in an auditable, content-addressed verdict store"
    )
    if compiler.get("stub_basis_present"):
        judge_line = (
            "CAUTION: carries the stub_basis_present watermark — some clauses "
            "were never assessed by a real judge"
        )

    content_hash = str(identity.get("content_hash", ""))

    return f"""
<section class="clause" id="method">
  <h2>Method &amp; provenance</h2>
  <p>This playbook was <b>compiled from evidence, not authored</b>. Pipeline:
  ingest &rarr; negotiation-trail reconstruction &rarr; clause segmentation
  &rarr; taxonomy classification &rarr; draft-to-draft diffing &rarr; LLM
  judgment (scope / provenance / deviation &amp; risk) &rarr; deterministic
  assembly &rarr; schema + normative validation.</p>
  <ul>
    <li><b>Corpus:</b> {n_in_scope} of {n_total} agreements in scope
      ({n_excluded} excluded with recorded rationale), {n_versions} negotiation
      versions; {failed_versions} version file(s) failed extraction and are
      quarantined, not silently dropped.</li>
    <li><b>Drafting origin:</b> {n_our} agreements on our paper,
      {n_cp} on counterparty paper — judged from each document's recitals and
      form structure.</li>
    <li><b>Judged evidence:</b> {n_obs} observed clause positions
      ({dev_line}); {unclassified} clause instances remain unclassified and are
      counted, not hidden.</li>
    <li><b>Judging:</b> {judge_line}.</li>
    <li><b>Traceability:</b> every position cites the exact document, version,
      and character span it came from; source files are pinned by SHA-256 in
      the machine-readable playbook, and counterparty names are pseudonymized
      at ingestion.</li>
    <li><b>Integrity:</b> document content hash
      <code>{html_lib.escape(content_hash[:16])}&hellip;</code> — any edit to
      the compiled content changes this fingerprint.</li>
  </ul>
</section>
"""


_CSS = """
:root { color-scheme: light; }
body { font-family: Charter, Georgia, 'Times New Roman', serif; margin: 0;
       background: #f8f7f4; color: #1f2937; line-height: 1.55; }
main { max-width: 46rem; margin: 0 auto; padding: 3rem 1.5rem 6rem; }
header.cover { border-bottom: 3px double #9ca3af; margin-bottom: 2.5rem;
               padding-bottom: 1.5rem; }
header.cover h1 { font-size: 2rem; margin: 0 0 0.25rem; letter-spacing: -0.01em; }
header.cover .subtitle { color: #6b7280; font-variant: small-caps;
                         letter-spacing: 0.08em; }
.stats { display: flex; flex-wrap: wrap; gap: 1.5rem; margin-top: 1rem;
         font-size: 0.9rem; color: #374151; }
.stats b { display: block; font-size: 1.3rem; color: #111827; }
nav.toc { background: #fff; border: 1px solid #e5e7eb; border-radius: 8px;
          padding: 1rem 1.5rem; margin-bottom: 2.5rem; font-size: 0.95rem; }
nav.toc a { color: #1d4ed8; text-decoration: none; }
nav.toc li { margin: 0.15rem 0; }
section.clause { background: #fff; border: 1px solid #e5e7eb; border-radius: 8px;
                 padding: 1.5rem 2rem; margin-bottom: 1.75rem;
                 box-shadow: 0 1px 2px rgba(0,0,0,0.04); }
section.clause h2 { font-size: 1.25rem; margin: 0 0 0.25rem; }
section.clause h3 { font-size: 0.85rem; text-transform: uppercase;
                    letter-spacing: 0.08em; color: #6b7280; margin: 1.25rem 0 0.5rem; }
.cnum { color: #9ca3af; font-weight: 400; }
p.meta { color: #6b7280; font-size: 0.85rem; margin: 0 0 0.5rem; }
blockquote { margin: 0.5rem 0; padding: 0.6rem 1rem; background: #f9fafb;
             border-left: 3px solid #d1d5db; font-size: 0.92rem;
             white-space: pre-wrap; }
blockquote .cite { margin-top: 0.4rem; font-size: 0.75rem; color: #9ca3af;
                   font-family: ui-monospace, Menlo, monospace; }
.variation { margin-bottom: 1rem; }
.var-label { font-size: 0.75rem; text-transform: uppercase; color: #9ca3af;
             margin: 0.5rem 0 0.1rem; letter-spacing: 0.06em; }
p.rationale { font-size: 0.88rem; color: #4b5563; font-style: italic; }
.risk { font-size: 0.78rem; color: #92400e; font-family: ui-monospace, Menlo, monospace; }
details { margin: 0.75rem 0; }
details summary { cursor: pointer; color: #1d4ed8; font-size: 0.92rem; }
details ul { font-size: 0.88rem; }
section.pending { background: #fffbeb; border: 1px solid #fde68a;
                  border-radius: 8px; padding: 1rem 1.5rem; margin-bottom: 1.75rem;
                  color: #713f12; }
footer { margin-top: 3rem; padding-top: 1rem; border-top: 1px solid #e5e7eb;
         color: #9ca3af; font-size: 0.8rem; }
@media print {
  body { background: #fff; }
  section.clause { border: none; box-shadow: none; padding: 0 0 1rem;
                   page-break-inside: avoid; }
  details { display: none; }
  nav.toc { display: none; }
}
"""


def _read_opf(out_dir: Path) -> tuple[str, dict[str, Any]]:
    """Read ``out_dir/playbook.opf.json``, returning ``(raw_text, parsed)``.

    Raises:
        FileNotFoundError: ``playbook.opf.json`` missing from *out_dir*.
    """
    opf_path = out_dir / "playbook.opf.json"
    if not opf_path.exists():
        raise FileNotFoundError(f"playbook.opf.json not found in {out_dir}")
    raw = opf_path.read_text(encoding="utf-8")
    return raw, json.loads(raw)


def _write_atomic(out_file: Path, text: str) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_file.with_suffix(out_file.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(out_file)


def _render_document_page(
    doc: dict[str, Any],
    *,
    extra_sections: str = "",
    trailing_blocks: str = "",
) -> str:
    """Build the readable document page from an already-parsed OPF document.

    This is the single place the document body is constructed, and the seam
    the bundle composes through. Both callers get the same body by
    construction rather than by pattern-matching on rendered template text:

        extra_sections:  extra ``<section>`` markup placed after the Method &
                         provenance panel and before the ``<footer>`` — where
                         ``render_bundle_html`` puts its digest summary.
        trailing_blocks: markup placed at the very end of ``<body>`` — where
                         ``render_bundle_html`` puts its machine-readable
                         ``<script type="application/json">`` payloads.

    Both default to empty, which yields the plain document.
    """
    agreement = doc.get("agreement_type", {})
    name = agreement.get("name", agreement.get("id", "Playbook"))
    identity = doc.get("identity", {})
    compiler = doc.get("compiler", {})
    corpus = doc.get("corpus", {})
    stats = corpus.get("stats", {})

    tax_labels: dict[str, str] = {}
    for entry in doc.get("taxonomy", {}).get("entries", []):
        tax_labels[str(entry.get("id", ""))] = str(entry.get("label", entry.get("id", "")))

    library_by_tid: dict[str, dict[str, Any]] = {}
    for concept in doc.get("evidence", {}).get("clause_library", []):
        if isinstance(concept, dict):
            library_by_tid[str(concept.get("taxonomy_id", ""))] = concept

    clauses = playbook_clauses(doc)
    clauses_sorted = sorted(
        clauses, key=lambda c: tax_labels.get(str(c.get("taxonomy_id", "")), "")
    )

    docs_total = stats.get("documents_total", len(corpus.get("documents", [])))
    docs_in_scope = stats.get(
        "documents_in_scope",
        sum(1 for d in corpus.get("documents", []) if d.get("in_scope")),
    )
    versions_total = stats.get("versions_total", "—")
    n_acceptable = sum(len((c.get("summary") or {}).get("acceptable_if") or []) for c in clauses)
    n_rejected = sum(len((c.get("summary") or {}).get("rejected") or []) for c in clauses)

    toc_items = "".join(
        f'<li><a href="#clause-{i}">{html_lib.escape(str(c.get("title") or tax_labels.get(str(c.get("taxonomy_id", "")), "")))}</a>'
        f" {_stance_chip(clause_stance(c))}</li>"
        for i, c in enumerate(clauses_sorted, start=1)
    )

    clause_html = "\n".join(
        _render_clause(c, library_by_tid, tax_labels, i)
        for i, c in enumerate(clauses_sorted, start=1)
    )

    posture = doc.get("posture") or {}
    floor = doc.get("floor") or {}
    posture_html = (
        f'<section class="clause"><h2>Posture</h2><blockquote>'
        f"{html_lib.escape(str(posture.get('system_prompt', '')))}</blockquote></section>"
        if posture.get("system_prompt")
        else (
            '<section class="pending"><strong>Posture:</strong> pending (optional). '
            "Posture is the negotiation-intent brief: short prose telling a reviewer "
            "— human or AI — how to lean where the evidence leaves room (what to "
            "hold, what to trade, tone). It is authored by the General Counsel, "
            "never derived from the corpus, so this playbook ships without one. "
            "A consuming review application works fine without it, running on the "
            "evidence sections alone. <em>To enable:</em> run "
            "<code>playbook posture interview &lt;out_dir&gt;</code> (see "
            "<code>playbook posture questions</code> for the question set), then "
            "re-validate and re-render this bundle — the content hash changes. "
            "<em>Why:</em> reviews gain your intent, not just your history.</section>"
        )
    )
    invariants = floor.get("invariants") or []
    if invariants:
        floor_items = "".join(
            f"<li>{html_lib.escape(str(inv.get('text', inv) if isinstance(inv, dict) else inv))}</li>"
            for inv in invariants
        )
        floor_html = (
            f'<section class="clause"><h2>Floor (non-negotiable)</h2>'
            f"<ul>{floor_items}</ul></section>"
        )
    else:
        floor_html = (
            '<section class="pending"><strong>Floor:</strong> pending (optional). '
            "The Floor is the short list of walk-away invariants — categorical red "
            "lines a review must always flag and can never waive (e.g. an "
            "indemnification cap below your minimum). Invariants require the legal "
            "owner's sign-off and are never auto-promoted from data, so this "
            "playbook ships without any: nothing is treated as non-negotiable "
            "until you say so, and a consuming review application works fine in "
            "that state. <em>To enable:</em> run "
            "<code>playbook floor propose &lt;out_dir&gt;</code> to derive "
            "candidates from observed reversals (written to "
            "<code>floor.candidates.json</code>, a review sidecar), accept the "
            "ones you mean by editing <code>floor.invariants</code>, then "
            "re-validate and re-render this bundle. <em>Why:</em> Floor "
            "violations are flagged on every review, categorically — independent "
            "of model judgment.</section>"
        )

    watermark = ""
    if compiler.get("stub_basis_present"):
        watermark = (
            '<section class="pending"><strong>Caution:</strong> this playbook '
            "carries the <code>stub_basis_present</code> watermark — some clauses "
            "were never assessed by a real judge. Do not rely on it without "
            "review.</section>"
        )

    generated_at = compiler.get("generated_at", "")
    version_line = " · ".join(
        str(x)
        for x in (
            identity.get("id"),
            identity.get("version"),
            f"OPF {doc.get('opf_version', '')}",
            generated_at,
        )
        if x
    )

    method_html = _render_method_panel(doc, clauses)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html_lib.escape(str(name))} — Negotiation Playbook</title>
<style>{_CSS}</style>
</head>
<body>
<main>
<header class="cover">
  <div class="subtitle">Negotiation Playbook</div>
  <h1>{html_lib.escape(str(name))}</h1>
  <p class="meta">{html_lib.escape(version_line)}</p>
  <div class="stats">
    <div><b>{docs_in_scope}/{docs_total}</b> agreements in scope</div>
    <div><b>{versions_total}</b> negotiation versions</div>
    <div><b>{len(clauses)}</b> clause concepts</div>
    <div><b>{n_acceptable}</b> acceptable variations</div>
    <div><b>{n_rejected}</b> rejected asks</div>
  </div>
</header>
{watermark}
<nav class="toc"><strong>Clauses</strong><ol>{toc_items}</ol></nav>
{posture_html}
{floor_html}
{clause_html}
{method_html}
{extra_sections}<footer>
Compiled by {html_lib.escape(str(compiler.get("name", "playbook-engine")))}
{html_lib.escape(str(compiler.get("version", "")))} — evidence-derived; every
position traces to cited corpus text. Confidential work product.
</footer>
</main>
{trailing_blocks}</body>
</html>
"""


def render_document_html(
    out_dir: Path, out_file: Path | None = None, alias_map: dict[str, str] | None = None
) -> str:
    """Render ``playbook.opf.json`` as a readable, print-friendly document.

    Internal rendering entry point: the readable document is no longer a
    user-facing artifact of its own — it ships inside the single-file bundle
    (``render_bundle_html`` / ``playbook view bundle``), which composes this
    same body through ``_render_document_page``. Kept as a function because
    the bundle and the alias-resolving internal path both need it.

    Args:
        out_dir:   Directory containing ``playbook.opf.json``.
        out_file:  If given, write the HTML there atomically (parent dirs
                   created); the HTML string is returned regardless.
        alias_map: Optional held-out ``alias -> real name`` map — same
                   contract as ``view render`` (issue #146): resolution
                   affects the rendered HTML only, never the stored OPF.

    Returns:
        Self-contained HTML string (no scripts, no external requests).

    Raises:
        FileNotFoundError: ``playbook.opf.json`` missing from *out_dir*.
    """
    _, doc = _read_opf(out_dir)
    if alias_map:
        doc = _resolve_aliases_in_doc(doc, alias_map)

    html_out = _render_document_page(doc)

    if out_file is not None:
        _write_atomic(out_file, html_out)

    return html_out


def _escape_json_for_script(json_text: str) -> str:
    """Make a JSON string safe inside a ``<script type="application/json">``.

    Replaces ``</`` with ``<\\/`` so no substring can close the script tag.
    ``JSON.parse``/``json.loads`` restore the original value exactly, so a
    consumer that parses the block and re-canonicalizes still verifies
    ``identity.content_hash`` — only the raw bytes differ, never the value.
    """
    return json_text.replace("</", "<\\/")


def render_bundle_html(out_dir: Path, out_file: Path | None = None) -> str:
    """Render the single-file OPF bundle: ``playbook.opf.html``.

    The full human document (the same body :func:`render_document_html`
    produces, including the Method & provenance panel), plus a digest
    summary section, with the
    CANONICAL OPF JSON and the digest embedded verbatim in
    ``<script type="application/json">`` blocks:

    - ``id="opf-canonical"`` — the on-disk ``playbook.opf.json`` text. The
      bare JSON file remains the canonical artifact; this block contains it,
      never replaces it. A consumer extracts the block, parses it, and
      verifies ``identity.content_hash`` over the canonical serialization
      (``playbook_engine.canonicalize``).
    - ``id="opf-digest"`` — the digest section (built on the fly for a
      pre-0.3 document that carries none).

    Deliberately takes no alias map: the bundle is the shareable artifact and
    embeds the canonical (pseudonymized) JSON; resolving real names into it
    would both leak and break hash verification. Internal-eyes review with
    real names belongs to ``view render --alias-map``.
    """
    from playbook_engine.digest import build_digest, digest_token_estimate  # noqa: PLC0415

    raw, doc = _read_opf(out_dir)
    digest = doc.get("digest") or build_digest(doc)

    d_clauses = digest.get("clauses", [])
    token_est = digest_token_estimate(digest)
    rows = "".join(
        "<tr>"
        f"<td>{html_lib.escape(str(c.get('title') or c.get('taxonomy_id') or ''))}</td>"
        f"<td>{_stance_chip(str(c.get('historical_stance') or 'unknown'))}</td>"
        f"<td>{len(c.get('preferred_variations') or [])}</td>"
        f"<td>{len(c.get('concessions') or [])}</td>"
        f"<td>{len(c.get('unacceptable') or [])}</td>"
        f"<td>{len(c.get('exemplar_forms') or [])}</td>"
        "</tr>"
        for c in d_clauses
    )
    digest_summary = f"""<section class="clause" id="digest">
  <h2>Digest (model-facing projection)</h2>
  <p>This bundle embeds a compact digest of the evidence section — per clause:
  stance, preferred variations verbatim, concession/unacceptable summaries, and
  frequency-annotated exemplar forms (deduped; <code>n</code>-weighted; bands
  often/sometimes/rare). Estimated size: ~{token_est:,} tokens. The machine
  blocks below carry the digest and the canonical OPF JSON; the bare
  <code>playbook.opf.json</code> remains the canonical artifact.</p>
  <table>
    <thead><tr><th>Clause</th><th>Stance</th><th>Preferred</th>
    <th>Concessions</th><th>Unacceptable</th><th>Exemplars</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
</section>
"""

    scripts = (
        "<!-- Machine-readable payloads. Extract a block, JSON-parse it, and verify\n"
        "     identity.content_hash over the canonical serialization (see\n"
        '     playbook_engine/canonicalize.py). "</" is escaped as "<\\/" inside the\n'
        "     blocks; JSON parsing restores the original text exactly. -->\n"
        f'<script id="opf-canonical" type="application/json">\n'
        f"{_escape_json_for_script(raw)}\n</script>\n"
        f'<script id="opf-digest" type="application/json">\n'
        f"{_escape_json_for_script(json.dumps(digest, indent=1, ensure_ascii=False))}\n</script>\n"
    )

    html_out = _render_document_page(doc, extra_sections=digest_summary, trailing_blocks=scripts)

    if out_file is not None:
        _write_atomic(out_file, html_out)

    return html_out
