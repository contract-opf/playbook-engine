"""Presentational HTML rendering of a compiled playbook — ``view bundle``.

Companion to :mod:`playbook_engine.viewer` with a different job: where
``view render`` produces the *review/annotation surface* (numbered items,
comment boxes, feedback export), this module produces the *readable
playbook* — a typographic, print-friendly document a lawyer can read top to
bottom or hand to a stakeholder.

The readable document is no longer an artifact of its own. It ships inside
the single-file bundle ``playbook.opf.html`` (:func:`render_bundle_html`),
which packages the same document body, plus a digest summary, plus the
canonical OPF JSON and digest embedded as machine-readable
``<script type="application/json">`` blocks. This is NOT a guarantee of
pseudonymization: ``known_entities`` matching is best-effort, and a
misconfigured or incomplete list can leave real names in this file (see the
mandatory residue check in the ``playbook-from-corpus`` skill, issue #136) —
run that check before treating the bundle as shareable. The bare
``playbook.opf.json`` remains the canonical source of truth on disk.
:func:`render_document_html` stays as the internal document renderer the
bundle composes; it is no longer exposed as its own CLI command (the former
``playbook view document`` deprecated alias was removed — use ``view
bundle``).

Both renderings are built by :func:`_render_document_page` from a parsed OPF
document, with the bundle passing its extra markup through that function's
explicit seams — so the bundle contains the whole document by construction,
not by string-splicing rendered template text.

Sections rendered per clause: our standard, how many deals signed it, the
non-standard variants signed and the asks refused before signing (collapsed),
each with its distinct-deal count and citation. Empty Posture/Floor
sections render as an explicit "pending GC interview" note rather than being
omitted — same honesty-first convention as the after-action report.

Alias handling: the document body shows the ``Counterparty-N`` aliases
exactly as stored in ``playbook.opf.json`` — best-effort ``known_entities``
matching, not a guarantee no raw name reached that JSON (run the mandatory
residue check before calling it shareable). The bundle takes no alias map by
design — it embeds the canonical JSON, so resolving real names into it would
both leak them and break hash verification. Internal-eyes review with real
names lives in ``view render --alias-map``. The on-disk OPF is never
modified.
"""

from __future__ import annotations

import html as html_lib
import json
from pathlib import Path
from typing import Any

from playbook_engine.opf_accessors import (
    perspective_party,
    playbook_clauses,
    playbook_precedent,
)
from playbook_engine.viewer import _resolve_aliases_in_doc


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
    precedent: list[dict[str, Any]],
    tax_labels: dict[str, str],
    number: int,
    *,
    party: str | None,
) -> str:
    """One clause (issue #223): our standard, how many deals signed it,
    every non-standard variant signed and every refused ask — each with its
    distinct-deal count and citation. No stance chip, no risk marker: the
    document carries no judged verdict."""
    from playbook_engine.digest import clause_precedent_groups  # noqa: PLC0415

    tid = str(clause.get("taxonomy_id", ""))
    title = clause.get("title") or tax_labels.get(tid, tid)
    parts: list[str] = [f'<section class="clause" id="clause-{number}">']
    parts.append(f'<h2><span class="cnum">{number}.</span> {html_lib.escape(str(title))}</h2>')
    meta_bits = [
        f"taxonomy: {html_lib.escape(tax_labels.get(tid, tid))}",
        f"{clause.get('n_deals', 0)} deal(s)",
        f"our standard signed in {clause.get('n_signed_standard', 0)}",
    ]
    parts.append(f'<p class="meta">{" · ".join(meta_bits)}</p>')

    our_standard = clause.get("our_standard") or {}
    std_text = our_standard.get("text") if isinstance(our_standard, dict) else None
    if std_text:
        parts.append("<h3>Our standard</h3>")
        parts.append(_quote_block(str(std_text), _cite_str(our_standard.get("source_ref"))))

    groups = clause_precedent_groups(clause.get("taxonomy_id"), precedent, party=party)
    if groups["signed_variants"]:
        parts.append(
            '<h3 title="Non-standard language signed in at least one deal, grouped by '
            'normalized text. (OPF field: evidence.precedent[].signed_text)">'
            "Signed variants</h3>"
        )
        for v in groups["signed_variants"]:
            parts.append(
                _quote_block(str(v["text"]), f"{_cite_str(v.get('ref'))} · {v['n_deals']} deal(s)")
            )
    if groups["refused_asks"]:
        parts.append(
            f'<details><summary title="Text proposed in a draft and struck before '
            f'signing. (OPF field: evidence.precedent[].refused_asks)">'
            f"Refused asks ({len(groups['refused_asks'])})</summary>"
        )
        for a in groups["refused_asks"]:
            parts.append(
                _quote_block(str(a["text"]), f"{_cite_str(a.get('ref'))} · {a['n_deals']} deal(s)")
            )
        parts.append("</details>")
    parts.append("</section>")
    return "\n".join(parts)


def _render_method_panel(doc: dict[str, Any]) -> str:
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
    # Issue #225: an ambiguous detection's two-valued provenance is not a
    # determined side — count it as undetermined, never as either paper.
    determined = [d for d in docs if d.get("provenance_is_ambiguous") is not True]
    n_our = sum(1 for d in determined if d.get("provenance") == "our_paper")
    n_cp = sum(1 for d in determined if d.get("provenance") == "counterparty_paper")
    n_unknown = len(docs) - len(determined)
    unknown_paper = f", {n_unknown} undetermined" if n_unknown else ""
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

    # One precedent per (deal, clause); the standard check is a
    # deterministic fact, not a judged deviation.
    dev_counts: dict[str, int] = {}
    n_obs = 0
    for record in playbook_precedent(doc):
        n_obs += 1
        key = "standard" if record.get("standard") is True else "non-standard"
        dev_counts[key] = dev_counts.get(key, 0) + 1
    dev_line = ", ".join(
        f"{dev_counts[k]} {k}" for k in ("standard", "non-standard") if k in dev_counts
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
      {n_cp} on counterparty paper{unknown_paper} — judged from each document's
      recitals and form structure.</li>
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
    acceptable_label, rejected_label = "signed variants", "refused asks"
    n_acceptable = sum(c.get("n_variants") or 0 for c in clauses)
    n_rejected = sum(c.get("n_refused") or 0 for c in clauses)

    toc_items = "".join(
        f'<li><a href="#clause-{i}">{html_lib.escape(str(c.get("title") or tax_labels.get(str(c.get("taxonomy_id", "")), "")))}</a>'
        + "</li>"
        for i, c in enumerate(clauses_sorted, start=1)
    )

    precedent = playbook_precedent(doc)
    party = perspective_party(doc)
    clause_html = "\n".join(
        _render_clause(c, precedent, tax_labels, i, party=party)
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

    method_html = _render_method_panel(doc)

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
    <div><b>{n_acceptable}</b> {acceptable_label}</div>
    <div><b>{n_rejected}</b> {rejected_label}</div>
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


def _render_digest_summary(d_clauses: list[dict[str, Any]], token_est: int) -> str:
    """Digest-section summary table (digest_version 3)."""
    rows = "".join(
        "<tr>"
        f"<td>{html_lib.escape(str(c.get('title') or c.get('taxonomy_id') or ''))}</td>"
        f"<td>{c.get('n_signed_standard', 0)} of {c.get('n_deals', 0)}</td>"
        f"<td>{c.get('n_variants_total', 0)}</td>"
        f"<td>{c.get('n_refused_total', 0)}</td>"
        "</tr>"
        for c in d_clauses
    )
    return f"""<section class="clause" id="digest">
  <h2>Digest (model-facing projection)</h2>
  <p>This bundle embeds a compact, verdict-free digest of the precedent record
  (digest_version 3) — per clause: our standard, how many deals signed it, the
  non-standard variants signed and the asks refused before signing, each with
  its distinct-deal count and citation (capped; the totals are always given).
  Estimated size: ~{token_est:,} tokens. The machine blocks below carry the
  digest and the canonical OPF JSON; the bare <code>playbook.opf.json</code>
  remains the canonical artifact.</p>
  <table>
    <thead><tr><th>Clause</th><th>Signed our standard</th><th>Signed variants</th>
    <th>Refused asks</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
</section>
"""


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
    - ``id="opf-digest"`` — the digest section (built on the fly when the
      document carries none).

    Deliberately takes no alias map: the bundle embeds the canonical JSON
    verbatim, so resolving real names into it would both leak them and break
    hash verification. This is NOT a guarantee of pseudonymization —
    ``known_entities`` matching is best-effort; run the mandatory residue
    check (``playbook-from-corpus`` skill, issue #136) before treating the
    bundle as shareable. Internal-eyes review with real names belongs to
    ``view render --alias-map``.
    """
    from playbook_engine.digest import build_digest, digest_token_estimate  # noqa: PLC0415

    raw, doc = _read_opf(out_dir)
    digest = doc.get("digest") or build_digest(doc)

    d_clauses = digest.get("clauses", [])
    token_est = digest_token_estimate(digest)
    digest_summary = _render_digest_summary(d_clauses, token_est)

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
