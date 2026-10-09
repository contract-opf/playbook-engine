"""``index.html`` — the one tabbed page of a compiled playbook (issue #241).

``playbook view bundle OUT`` writes ``OUT/index.html``: ONE self-contained file
(no network, no external script, font or stylesheet) that is both the viewer of
a compiled playbook and an OPTIONAL editor that saves back to the out-dir.

Tabs (ids in :data:`TAB_IDS`):

- **Start here** — brief instructions, the playbook's identity (``content_hash``,
  OPF version, perspective), file paths and the toaster install steps
  (:mod:`playbook_engine.toaster_install`, the same constant the skill prints).
- **Playbook** — Posture, Floor and the digest per clause: what the consuming
  model is shown.
- **Evidence** — the precedent per clause: how each deal's clause opened and
  what was signed, the non-standard variants, the openings not signed as
  proposed, the refused asks, the deal counts and the clause types the corpus
  has no evidence for.
- **Review (optional)** — the shortest useful list of model judgments a person
  may confirm or change (:mod:`playbook_engine.review_rows`).
- **Posture & Floor** — the authored text, with edit fields.

Edits never touch ``playbook.opf.json``: the page writes ``overrides.json``
(:mod:`playbook_engine.overrides`) — straight into the out-dir through the File
System Access API when the person grants the folder (Chrome, Edge), or as a
download where the API is missing — and ``playbook apply-overrides`` (or the
next ``playbook project``) folds it in, recomputing the digest, dossiers and
``identity.content_hash``. Nothing in a run waits on the page.

The canonical OPF JSON and the digest are embedded verbatim in
``<script type="application/json">`` blocks (ids ``opf-canonical``,
``opf-digest``). The bare ``playbook.opf.json`` remains the canonical artifact;
the page contains it, never replaces it. This is NOT a guarantee of
pseudonymization: ``known_entities`` matching is best-effort, so run the
mandatory residue check (see the ``playbook-from-corpus`` skill) before treating
the page as shareable.
"""

from __future__ import annotations

import html as html_lib
import json
from pathlib import Path
from typing import Any

from playbook_engine.digest import (
    build_digest_v4,
    clause_precedent_groups,
    digest_token_estimate,
)
from playbook_engine.index_page_assets import CSS, JS
from playbook_engine.opf_accessors import (
    perspective_party,
    playbook_clauses,
    playbook_precedent,
)
from playbook_engine.overrides import FLOOR_FIELDS, OVERRIDES_FILENAME, floor_edit_refusal
from playbook_engine.review_rows import REVIEW_LABEL_ORDER, ReviewRows, build_review_rows
from playbook_engine.toaster_install import PLAYBOOK_FILE, TOASTER_INSTALL_STEPS

#: The tab ids of ``index.html``, in display order.
TAB_IDS = ("start", "playbook", "evidence", "review", "posture-floor")

INDEX_FILENAME = "index.html"

_TAB_LABELS = {
    "start": "Start here",
    "playbook": "Playbook",
    "evidence": "Evidence",
    "review": "Review (optional)",
    "posture-floor": "Posture & Floor",
}

_LABEL_NAMES = {
    "less_protective": "Less protective",
    "different_concept": "Different concept",
    "more_protective": "More protective",
    "equivalent": "Equivalent",
}


def _esc(value: Any) -> str:
    return html_lib.escape(str(value))


def _cite_str(ref: dict[str, Any] | None) -> str:
    if not isinstance(ref, dict):
        return ""
    doc_id = str(ref.get("document_id", ""))
    version = ref.get("version", "")
    clause_path = str(ref.get("clause_path", ""))
    return f"{doc_id} · v{version} · §{clause_path}"


def _quote_block(text: str, cite: str = "") -> str:
    cite_html = f'<div class="cite">{_esc(cite)}</div>' if cite else ""
    return f"<blockquote>{_esc(text)}{cite_html}</blockquote>"


def _group_cite(group: dict[str, Any], *facts: str) -> str:
    """Citation line of a digest group: its citation, deal count and opening facts."""
    bits = [_cite_str(group.get("ref")), f"{group['n_deals']} deal(s)", *facts]
    return " · ".join(b for b in bits if b)


def _escape_json_for_script(json_text: str) -> str:
    """Make a JSON string safe inside a ``<script type="application/json">``.

    Replaces every ``<`` with the JSON escape ``\\u003c``, so no substring can
    close the script tag, open a comment or start a nested script.
    ``JSON.parse``/``json.loads`` restore the original value exactly, so a
    consumer that parses the block and re-canonicalizes still verifies
    ``identity.content_hash`` — only the raw bytes differ, never the value.
    """
    return json_text.replace("<", "\\u003c")


def _json_block(block_id: str, payload: str) -> str:
    return f'<script id="{block_id}" type="application/json">\n{_escape_json_for_script(payload)}\n</script>\n'


def _tax_labels(doc: dict[str, Any]) -> dict[str, str]:
    labels: dict[str, str] = {}
    for entry in doc.get("taxonomy", {}).get("entries", []):
        labels[str(entry.get("id", ""))] = str(entry.get("label", entry.get("id", "")))
    return labels


def _clause_title(clause: dict[str, Any], tax_labels: dict[str, str]) -> str:
    tid = str(clause.get("taxonomy_id", ""))
    return str(clause.get("title") or tax_labels.get(tid, tid))


# ---------------------------------------------------------------------------
# Start here
# ---------------------------------------------------------------------------


def _stats_html(doc: dict[str, Any], clauses: list[dict[str, Any]]) -> str:
    corpus = doc.get("corpus", {})
    stats = corpus.get("stats", {})
    docs_total = stats.get("documents_total", len(corpus.get("documents", [])))
    docs_in_scope = stats.get(
        "documents_in_scope", sum(1 for d in corpus.get("documents", []) if d.get("in_scope"))
    )
    versions_total = stats.get("versions_total", "—")
    n_variants = sum(c.get("n_variants") or 0 for c in clauses)
    n_refused = sum(c.get("n_refused") or 0 for c in clauses)
    return (
        '<div class="stats">'
        f"<div><b>{_esc(docs_in_scope)}/{_esc(docs_total)}</b> agreements in scope</div>"
        f"<div><b>{_esc(versions_total)}</b> negotiation versions</div>"
        f"<div><b>{len(clauses)}</b> clause concepts</div>"
        f"<div><b>{n_variants}</b> signed variants</div>"
        f"<div><b>{n_refused}</b> refused asks</div>"
        "</div>"
    )


def _render_start(doc: dict[str, Any], review: ReviewRows) -> str:
    identity = doc.get("identity", {})
    compiler = doc.get("compiler", {})
    perspective = doc.get("perspective")
    persp_line = (
        f"{_esc(perspective.get('party', ''))} (counterparty: "
        f"{_esc(perspective.get('counterparty_type', ''))})"
        if isinstance(perspective, dict)
        else "not set"
    )
    steps = "".join(
        "<li>"
        + _esc(step).replace("{file}", '<code class="js-opf-path">' + PLAYBOOK_FILE + "</code>")
        + "</li>"
        for step in TOASTER_INSTALL_STEPS
    )
    n_rows = len(review.rows)
    watermark = ""
    if compiler.get("stub_basis_present"):
        watermark = (
            '<section class="pending"><strong>Caution:</strong> this playbook carries the '
            "<code>stub_basis_present</code> watermark — some clauses were never assessed by a "
            "real judge. Do not rely on it without review.</section>"
        )
    return f"""<section class="card">
  <h2>Your playbook is ready</h2>
  <p>This page is the playbook and an optional editor. <strong>Nothing here has to be
  done</strong> — the model's judgments stand unless you override them.</p>
  <ol class="steps">
    <li>Look through the <a href="#playbook" data-goto="playbook">Playbook</a> and
    <a href="#evidence" data-goto="evidence">Evidence</a> tabs.</li>
    <li>Optional: in <a href="#review" data-goto="review">Review</a>, confirm or change the
    {n_rows} model judgment{"" if n_rows == 1 else "s"} the consuming model is shown, and
    edit your Posture and Floor text in
    <a href="#posture-floor" data-goto="posture-floor">Posture &amp; Floor</a>.</li>
    <li>Install the playbook in the toaster (below).</li>
  </ol>
</section>
{watermark}<section class="card">
  <h2>Install in the toaster</h2>
  <ol class="steps">{steps}</ol>
  <p class="hint">The file the toaster takes is <code>{PLAYBOOK_FILE}</code>, not this page.</p>
</section>
<section class="card">
  <h2>This playbook</h2>
  <div class="table-wrap"><table>
    <tr><th>Agreement type</th><td>{_esc(doc.get("agreement_type", {}).get("name", ""))}</td></tr>
    <tr><th>Perspective</th><td>{persp_line}</td></tr>
    <tr><th>OPF version</th><td>{_esc(doc.get("opf_version", ""))}</td></tr>
    <tr><th>content_hash</th><td><code>{_esc(identity.get("content_hash", ""))}</code></td></tr>
    <tr><th>Compiled</th><td>{_esc(compiler.get("generated_at", ""))} by
      {_esc(compiler.get("name", "playbook-engine"))} {_esc(compiler.get("version", ""))}</td></tr>
  </table></div>
</section>
<section class="card">
  <h2>Files</h2>
  <p class="small muted">Folder: <code class="js-folder">the folder this page is in</code></p>
  <div class="table-wrap"><table>
    <tr><th>File</th><th>What it is</th></tr>
    <tr><td><code class="js-opf-path">{PLAYBOOK_FILE}</code></td>
      <td>The canonical playbook. Upload this to the toaster.</td></tr>
    <tr><td><code class="js-index-path">{INDEX_FILENAME}</code></td>
      <td>This page.</td></tr>
    <tr><td><code>{OVERRIDES_FILENAME}</code></td>
      <td>Your optional edits from the Review and Posture &amp; Floor tabs. The assistant
      (or <code>playbook apply-overrides</code>) folds them into the playbook.</td></tr>
  </table></div>
</section>"""


# ---------------------------------------------------------------------------
# Playbook tab
# ---------------------------------------------------------------------------


def _render_posture_floor_readonly(doc: dict[str, Any]) -> str:
    posture = doc.get("posture") or {}
    floor = doc.get("floor") or {}
    if posture.get("system_prompt"):
        posture_html = (
            f'<section class="clause"><h2>Posture</h2>'
            f"{_quote_block(str(posture['system_prompt']))}</section>"
        )
    else:
        posture_html = (
            '<section class="pending"><strong>Posture:</strong> pending (optional). Posture is '
            "the negotiation-intent brief: short prose telling a reviewer how to lean where the "
            "evidence leaves room. It is authored by the General Counsel, never derived from the "
            "corpus, so this playbook ships without one and works fine on the evidence alone. "
            '<em>To add one:</em> ask the assistant to "author my posture" (or run '
            "<code>playbook posture interview &lt;out_dir&gt;</code>); the page is rebuilt "
            "afterwards.</section>"
        )
    invariants = floor.get("invariants") or []
    if invariants:
        items = "".join(
            f"<li>{_esc(inv.get('statement', inv) if isinstance(inv, dict) else inv)}</li>"
            for inv in invariants
        )
        floor_html = (
            f'<section class="clause"><h2>Floor (non-negotiable)</h2><ul>{items}</ul></section>'
        )
    else:
        floor_html = (
            '<section class="pending"><strong>Floor:</strong> pending (optional). The Floor is '
            "the short list of walk-away invariants a review must always flag. Invariants need "
            "the legal owner's sign-off and are never promoted from data, so this playbook ships "
            'without any. <em>To add one:</em> ask the assistant to "propose and sign hard lines" '
            "(or run <code>playbook floor propose &lt;out_dir&gt;</code>, then "
            "<code>playbook floor sign</code>).</section>"
        )
    return posture_html + floor_html


def _render_playbook_clause(
    clause: dict[str, Any], dclause: dict[str, Any] | None, number: int, title: str
) -> str:
    standard = clause.get("our_standard") or {}
    std_text = standard.get("text") if isinstance(standard, dict) else None
    meta = [
        f"{clause.get('n_deals', 0)} deal(s)",
        f"our standard signed in {clause.get('n_signed_standard', 0)}",
    ]
    if clause.get("n_opened_standard"):
        meta.append(
            f"opened with our standard in {clause['n_opened_standard']}; "
            f"kept it in {clause.get('n_kept_standard', 0)}"
        )
    positions = (dclause or {}).get("positions") or {}
    pos_bits = [f"{positions['standard']} signed our standard"] if positions.get("standard") else []
    for label in REVIEW_LABEL_ORDER:
        if positions.get(label):
            pos_bits.append(f"{positions[label]} {_LABEL_NAMES[label].lower()}")
    if positions.get("unjudged"):
        pos_bits.append(f"{positions['unjudged']} unjudged")
    parts = [
        f'<section class="clause" id="clause-{number}">',
        f"<h2>{number}. {_esc(title)}</h2>",
        f'<p class="meta">{_esc(" · ".join(meta))}</p>',
    ]
    if pos_bits:
        parts.append(f'<p class="meta">Signed positions: {_esc(", ".join(pos_bits))}</p>')
    if std_text:
        text = str(std_text)
        if len(text) > 700:
            parts.append(
                f"<details><summary>Our standard ({len(text):,} characters)</summary>"
                f"{_quote_block(text, _cite_str(standard.get('source_ref')))}</details>"
            )
        else:
            parts.append("<h3>Our standard</h3>")
            parts.append(_quote_block(text, _cite_str(standard.get("source_ref"))))
    parts.append("</section>")
    return "\n".join(parts)


def _render_digest_summary(
    d_clauses: list[dict[str, Any]], token_est: int, uncovered: list[dict[str, Any]] | None
) -> str:
    rows = "".join(
        "<tr>"
        f"<td>{_esc(c.get('title') or c.get('taxonomy_id') or '')}</td>"
        f'<td class="num">{c.get("n_signed_standard", 0)} of {c.get("n_deals", 0)}</td>'
        f'<td class="num">{c.get("n_kept_standard", 0)} of {c.get("n_opened_standard", 0)}</td>'
        f'<td class="num">{c.get("n_variants_total", 0)}</td>'
        f'<td class="num">{c.get("n_changed_openings_total", 0)}</td>'
        f'<td class="num">{c.get("n_refused_total", 0)}</td>'
        "</tr>"
        for c in d_clauses
    )
    names = [
        str(u.get("label") or u.get("taxonomy_id")) for u in uncovered or [] if isinstance(u, dict)
    ]
    uncovered_line = (
        "<p>No evidence in this corpus for: " + _esc(", ".join(names)) + ".</p>" if names else ""
    )
    return f"""<section class="clause" id="digest">
  <h2>Digest (what the consuming model is shown)</h2>
  <p>Per clause: our standard, how many deals signed it and how many opened with it and
  kept it, the non-standard variants signed, the openings not signed as proposed and the asks
  refused before signing, each with its distinct-deal count and citation (capped; the totals are
  always given). Estimated size: ~{token_est:,} tokens. Verdict-free except the one judged
  <code>vs_standard</code> label of each text (see the Review tab).</p>
  <div class="table-wrap"><table>
    <thead><tr><th>Clause</th><th class="num">Signed our standard</th>
    <th class="num">Opened with ours, kept it</th><th class="num">Signed variants</th>
    <th class="num">Not signed as proposed</th><th class="num">Refused asks</th></tr></thead>
    <tbody>{rows}</tbody>
  </table></div>
  {uncovered_line}
</section>"""


def _render_method_panel(doc: dict[str, Any]) -> str:
    """The "Method & provenance" panel, from the document's own numbers."""
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
    for record in playbook_precedent(doc):
        n_obs += 1
        key = "standard" if record.get("standard") is True else "non-standard"
        dev_counts[key] = dev_counts.get(key, 0) + 1
    dev_line = ", ".join(
        f"{dev_counts[k]} {k}" for k in ("standard", "non-standard") if k in dev_counts
    )
    judge_line = (
        "structural stages are deterministic; scope, provenance and the vs_standard label "
        "were judged by a model, each stored with its rationale in an auditable, "
        "content-addressed verdict store"
    )
    if compiler.get("stub_basis_present"):
        judge_line = (
            "CAUTION: carries the stub_basis_present watermark — some clauses "
            "were never assessed by a real judge"
        )
    content_hash = str(identity.get("content_hash", ""))
    return f"""<section class="clause" id="method">
  <h2>Method &amp; provenance</h2>
  <p>This playbook was <b>compiled from evidence, not authored</b>: ingest &rarr;
  negotiation-trail reconstruction &rarr; clause segmentation &rarr; taxonomy classification
  &rarr; draft-to-draft diffing &rarr; model judgment (scope, provenance, vs_standard) &rarr;
  deterministic assembly &rarr; schema + normative validation.</p>
  <ul>
    <li><b>Corpus:</b> {n_in_scope} of {n_total} agreements in scope ({n_excluded} excluded with
      recorded rationale), {n_versions} negotiation versions; {failed_versions} version file(s)
      failed extraction and are quarantined, not silently dropped.</li>
    <li><b>Drafting origin:</b> {n_our} agreements on our paper, {n_cp} on counterparty
      paper{unknown_paper} — judged from each document's recitals and form structure.</li>
    <li><b>Judged evidence:</b> {n_obs} observed clause positions ({dev_line}); {unclassified}
      clause instances remain unclassified and are counted, not hidden.</li>
    <li><b>Judging:</b> {judge_line}.</li>
    <li><b>Traceability:</b> every position cites the exact document, version and character span
      it came from; source files are pinned by SHA-256 in the machine-readable playbook, and
      counterparty names are pseudonymized at ingestion.</li>
    <li><b>Integrity:</b> content hash <code>{_esc(content_hash[:16])}&hellip;</code> — any edit
      to the compiled content changes this fingerprint.</li>
  </ul>
</section>"""


def _render_playbook_tab(
    doc: dict[str, Any],
    digest: dict[str, Any],
    clauses_sorted: list[dict[str, Any]],
    tax_labels: dict[str, str],
) -> str:
    dby = {c.get("taxonomy_id"): c for c in digest.get("clauses", []) if isinstance(c, dict)}
    toc = "".join(
        f'<li><a href="#clause-{i}">{_esc(_clause_title(c, tax_labels))}</a></li>'
        for i, c in enumerate(clauses_sorted, start=1)
    )
    body = "\n".join(
        _render_playbook_clause(c, dby.get(c.get("taxonomy_id")), i, _clause_title(c, tax_labels))
        for i, c in enumerate(clauses_sorted, start=1)
    )
    summary = _render_digest_summary(
        digest.get("clauses", []),
        digest_token_estimate(digest),
        digest.get("uncovered_clause_types"),
    )
    return (
        _render_posture_floor_readonly(doc)
        + f'<nav class="toc card"><strong>Clauses</strong><ol>{toc}</ol></nav>\n'
        + body
        + "\n"
        + summary
        + "\n"
        + _render_method_panel(doc)
    )


# ---------------------------------------------------------------------------
# Evidence tab
# ---------------------------------------------------------------------------


def _outcome(record: dict[str, Any]) -> str:
    if record.get("signed") is not True:
        return "not signed"
    if record.get("signed_text") is None:
        return "struck before signing"
    return "signed our standard" if record.get("standard") is True else "signed a variant"


_OPENED = {
    "standard": "our standard",
    "non_standard": "non-standard text",
    "absent": "absent (added later)",
}


def _render_deal_trail(records: list[dict[str, Any]]) -> str:
    """One row per deal: how its draft opened and what was signed.

    The Signed column is dropped when no deal carries a ``signed_at`` (a corpus
    whose dates were coarsened away would otherwise show a column of dashes).
    """
    dated = any(r.get("signed_at") for r in records)
    rows = "".join(
        "<tr>"
        f"<td>{_esc(r.get('document_id', ''))}</td>"
        + (f"<td>{_esc(r.get('signed_at') or '—')}</td>" if dated else "")
        + f"<td>{_esc(_OPENED.get(str(r.get('opened_with')), '—'))}</td>"
        f"<td>{_esc(_outcome(r))}</td>"
        f'<td class="num">{len(r.get("refused_asks") or [])}</td>'
        f'<td class="num">{r.get("rounds", 0)}</td>'
        "</tr>"
        for r in sorted(records, key=lambda r: (str(r.get("document_id")), str(r.get("id"))))
    )
    return (
        "<h3>Deal by deal: how it opened, what was signed</h3>"
        '<div class="table-wrap"><table><thead><tr><th>Deal</th>'
        + ("<th>Signed</th>" if dated else "")
        + '<th>Opened with</th><th>Result</th><th class="num">Refused asks</th>'
        '<th class="num">Rounds</th></tr></thead>'
        f"<tbody>{rows}</tbody></table></div>"
    )


def _render_evidence_clause(
    clause: dict[str, Any],
    precedent: list[dict[str, Any]],
    tax_labels: dict[str, str],
    number: int,
    *,
    party: str | None,
) -> str:
    tid = clause.get("taxonomy_id")
    title = _clause_title(clause, tax_labels)
    records = [p for p in precedent if p.get("taxonomy_id") == tid]
    groups = clause_precedent_groups(tid, precedent, party=party)
    pills = [
        f"{clause.get('n_deals', 0)} deals",
        f"ours signed {clause.get('n_signed_standard', 0)}",
        f"opened with ours {clause.get('n_opened_standard', 0)}, kept {clause.get('n_kept_standard', 0)}",
        f"{len(groups['signed_variants'])} variants",
        f"{len(groups['changed_openings'])} changed openings",
        f"{len(groups['refused_asks'])} refused",
    ]
    meta = "".join(f'<span class="pill">{_esc(p)}</span>' for p in pills)
    parts = [
        f'<details class="clause card" id="ev-{number}">',
        f'<summary><strong class="ev-title">{number}. {_esc(title)}</strong>'
        f'<span class="pills">{meta}</span></summary>',
        _render_deal_trail(records),
    ]
    if groups["signed_variants"]:
        parts.append(
            '<h3 title="Non-standard language signed in at least one deal, grouped by '
            'normalized text. (OPF field: evidence.precedent[].signed_text)">Signed variants</h3>'
        )
        for v in groups["signed_variants"]:
            facts = []
            if v.get("n_from_standard"):
                facts.append(f"from our standard in {v['n_from_standard']}")
            if v.get("n_unchanged"):
                facts.append(f"signed as proposed in {v['n_unchanged']}")
            parts.append(_quote_block(str(v["text"]), _group_cite(v, *facts)))
    if groups["changed_openings"]:
        parts.append(
            '<h3 title="Non-standard language the clause opened with that was not signed as '
            'proposed. (OPF field: evidence.precedent[].opening_text)">'
            "Openings not signed as proposed</h3>"
        )
        for o in groups["changed_openings"]:
            facts = []
            if o.get("n_to_standard"):
                facts.append(f"signed as our standard in {o['n_to_standard']}")
            if o.get("n_struck"):
                facts.append(f"struck in {o['n_struck']}")
            parts.append(_quote_block(str(o["text"]), _group_cite(o, *facts)))
    if groups["refused_asks"]:
        parts.append(
            '<h3 title="Text proposed in a draft and struck before signing. '
            '(OPF field: evidence.precedent[].refused_asks)">Refused asks</h3>'
        )
        for a in groups["refused_asks"]:
            parts.append(_quote_block(str(a["text"]), _group_cite(a)))
    parts.append("</details>")
    return "\n".join(parts)


def _render_evidence_tab(
    doc: dict[str, Any],
    digest: dict[str, Any],
    clauses_sorted: list[dict[str, Any]],
    tax_labels: dict[str, str],
) -> str:
    precedent = playbook_precedent(doc)
    party = perspective_party(doc)
    body = "\n".join(
        _render_evidence_clause(c, precedent, tax_labels, i, party=party)
        for i, c in enumerate(clauses_sorted, start=1)
    )
    uncovered = [u for u in digest.get("uncovered_clause_types") or [] if isinstance(u, dict)]
    if uncovered:
        names = ", ".join(_esc(u.get("label") or u.get("taxonomy_id")) for u in uncovered)
        uncovered_html = (
            f'<section class="card"><h2>No evidence for</h2><p>{names}.</p>'
            '<p class="hint">Recognised clause types with no precedent in this corpus; nothing '
            "more is claimed.</p></section>"
        )
    else:
        uncovered_html = ""
    return (
        '<section class="card"><h2>Precedent per clause</h2>'
        "<p>Facts only: for each clause, how each deal's draft opened and what was signed, the "
        "non-standard language that was signed, the openings that were not signed as proposed "
        "and the asks refused before signing. Open a clause to see it.</p></section>\n"
        + body
        + "\n"
        + uncovered_html
    )


# ---------------------------------------------------------------------------
# Review tab
# ---------------------------------------------------------------------------


def _render_review_tab(review: ReviewRows) -> str:
    chips = [("all", "All"), ("todo", "Not reviewed yet")] + [
        (label, _LABEL_NAMES[label]) for label in REVIEW_LABEL_ORDER
    ]
    filters = "".join(
        f'<button type="button" data-filter="{f}" data-name="{_esc(name)}" aria-pressed="false">'
        f"{_esc(name)}</button>"
        for f, name in chips
    )
    extra = []
    if review.n_not_shown:
        extra.append(
            f"{review.n_not_shown} other judged text(s) are not in the digest, so they change "
            "nothing the model sees and are not listed."
        )
    if review.n_unjudged:
        extra.append(
            f"{review.n_unjudged} text(s) in the digest have no label yet (run "
            "<code>playbook judge</code> to queue them)."
        )
    return f"""<section class="card">
  <h2>Review (optional)</h2>
  <p>The consuming model is shown one label for each non-standard text — equivalent to our
  standard, more or less protective, or a different concept — and checks it against the cited
  text. These are the labels it is shown, worst first. <strong>Nothing waits on this
  list:</strong> confirm what is right, change what is not, skip the rest. Your word is final
  and is recorded as <code>owner</code>.</p>
  <p class="hint">{" ".join(extra)}</p>
  <div class="savebar">
    <button type="button" class="btn" id="btn-connect" hidden>Connect folder</button>
    <button type="button" class="btn secondary" id="btn-reconnect" hidden>Reconnect folder</button>
    <button type="button" class="btn secondary" id="btn-download">Download edits</button>
    <span id="save-status" role="status" aria-live="polite"></span>
  </div>
  <p class="hint" id="save-note" hidden>This browser cannot save into a folder (Chrome and Edge
  can). Use <strong>Download edits</strong> and put the downloaded <code>{OVERRIDES_FILENAME}</code>
  next to <code>{PLAYBOOK_FILE}</code>.</p>
  <p class="hint">Connect folder: choose the folder this page is in; edits are saved to
  <code>{OVERRIDES_FILENAME}</code> there as you make them.</p>
  <p class="progress" id="review-progress"></p>
</section>
<noscript><section class="pending">The Review tab needs scripting to list the labels. The
labels themselves are in the digest of <code>{PLAYBOOK_FILE}</code>.</section></noscript>
<div class="filters" role="group" aria-label="Filter">{filters}</div>
<div class="bulk">
  <button type="button" class="btn secondary" id="bulk-confirm">Confirm all in this view</button>
  <button type="button" class="btn secondary" id="bulk-undo">Undo all in this view</button>
</div>
<div id="review-list"></div>
<p id="review-empty" hidden class="muted">Nothing to show for this filter.</p>
<p><button type="button" class="btn secondary" id="review-more" hidden>Show more</button></p>"""


# ---------------------------------------------------------------------------
# Posture & Floor tab
# ---------------------------------------------------------------------------


def _edit_box(target: str, ident: str, field: str | None, label: str, text: str) -> str:
    field_attr = f' data-field="{_esc(field)}"' if field else ""
    return (
        f'<div class="edit-box"><label><strong>{_esc(label)}</strong>'
        f'<textarea class="edit" data-target="{target}" data-id="{_esc(ident)}"{field_attr}>'
        f"{_esc(text)}</textarea></label>"
        '<span class="edited-flag" hidden>edited</span> '
        '<button type="button" class="btn secondary revert" hidden>Revert</button></div>'
    )


def _render_posture_floor_tab(doc: dict[str, Any]) -> str:
    posture = doc.get("posture") or {}
    invariants = [
        i for i in (doc.get("floor") or {}).get("invariants") or [] if isinstance(i, dict)
    ]
    if posture.get("system_prompt"):
        gen = posture.get("generation") or {}
        posture_html = (
            '<section class="card"><h2>Posture</h2>'
            f'<p class="meta">version {_esc(posture.get("version", ""))}'
            f"{' · ' + _esc(gen.get('generated_at')) if gen.get('generated_at') else ''}</p>"
            + _edit_box(
                "posture", "system_prompt", None, "Posture text", str(posture["system_prompt"])
            )
            + "</section>"
        )
    else:
        posture_html = (
            '<section class="pending"><strong>Posture:</strong> none yet. It is authored by the '
            "General Counsel from a short interview (<code>playbook posture interview</code>); "
            "this page edits one that exists.</section>"
        )
    if invariants:
        boxes = []
        for inv in invariants:
            ident = str(inv.get("id", ""))
            signer = f" · signed by {_esc(inv['x_signed_by'])}" if inv.get("x_signed_by") else ""
            fields = []
            for field in FLOOR_FIELDS:
                if not isinstance(inv.get(field), str):
                    continue
                refusal = floor_edit_refusal(inv, field)
                if refusal is None:
                    fields.append(_edit_box("floor", ident, field, field.capitalize(), inv[field]))
                else:
                    # Not authored text the page may replace: shown, with why.
                    fields.append(
                        f'<div class="edit-box"><strong>{_esc(field.capitalize())}</strong>'
                        f"<p>{_esc(inv[field])}</p>"
                        f'<p class="meta">Not editable here: {_esc(refusal)}.</p></div>'
                    )
            boxes.append(
                f'<section class="card"><h2>{_esc(ident)}</h2>'
                f'<p class="meta">Floor invariant{signer}</p>' + "".join(fields) + "</section>"
            )
        floor_html = "".join(boxes)
    else:
        floor_html = (
            '<section class="pending"><strong>Floor:</strong> no invariants yet. They need the '
            "legal owner's sign-off (<code>playbook floor propose</code>, then "
            "<code>playbook floor sign</code>); this page edits ones that exist.</section>"
        )
    return (
        '<section class="card"><h2>Posture &amp; Floor</h2>'
        "<p>The authored text, with edit fields. An edit is saved to "
        f"<code>{OVERRIDES_FILENAME}</code> (see the Review tab to connect a folder or download "
        "it) and folded into the playbook by the engine, which recomputes the digest and "
        "<code>content_hash</code>. Optional.</p>"
        "</section>" + posture_html + floor_html
    )


# ---------------------------------------------------------------------------
# The page
# ---------------------------------------------------------------------------


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


def _initial_overrides(out_dir: Path) -> str | None:
    """The text of a well-formed ``overrides.json`` already in *out_dir*, else ``None``."""
    path = out_dir / OVERRIDES_FILENAME
    if not path.is_file():
        return None
    try:
        text = path.read_text(encoding="utf-8")
        return text if isinstance(json.loads(text), dict) else None
    except (OSError, ValueError):
        return None


def render_index_html(out_dir: Path, out_file: Path | None = None) -> str:
    """Render ``index.html`` for the playbook in *out_dir*; write it to *out_file* when given.

    Raises:
        FileNotFoundError: ``playbook.opf.json`` missing from *out_dir*.
    """
    raw, doc = _read_opf(out_dir)
    digest = doc.get("digest") or build_digest_v4(doc)
    agreement = doc.get("agreement_type", {})
    name = agreement.get("name", agreement.get("id", "Playbook"))
    identity = doc.get("identity", {})
    compiler = doc.get("compiler", {})

    tax_labels = _tax_labels(doc)
    clauses = playbook_clauses(doc)
    clauses_sorted = sorted(clauses, key=lambda c: _clause_title(c, tax_labels).lower())
    review = build_review_rows(doc)

    panels = {
        "start": _render_start(doc, review),
        "playbook": _render_playbook_tab(doc, digest, clauses_sorted, tax_labels),
        "evidence": _render_evidence_tab(doc, digest, clauses_sorted, tax_labels),
        "review": _render_review_tab(review),
        "posture-floor": _render_posture_floor_tab(doc),
    }
    tab_buttons = "".join(
        f'<button type="button" role="tab" id="t-{tid}" data-tab="{tid}" '
        f'aria-controls="tab-{tid}" aria-selected="{"true" if tid == "start" else "false"}">'
        f"{_esc(_TAB_LABELS[tid])}"
        + ('<span class="count" id="tab-review-count"></span>' if tid == "review" else "")
        + "</button>"
        for tid in TAB_IDS
    )
    tab_panels = "".join(
        f'<div role="tabpanel" id="tab-{tid}" aria-labelledby="t-{tid}">{panels[tid]}</div>\n'
        for tid in TAB_IDS
    )

    posture = doc.get("posture") or {}
    pf_data = {
        "posture": {"system_prompt": posture.get("system_prompt")}
        if posture.get("system_prompt")
        else None,
        "floor": [
            {f: i.get(f) for f in ("id", *FLOOR_FIELDS)}
            for i in (doc.get("floor") or {}).get("invariants") or []
            if isinstance(i, dict)
        ],
    }
    raw_perspective = doc.get("perspective")
    perspective: dict[str, Any] = raw_perspective if isinstance(raw_perspective, dict) else {}
    # What the page checks a folder's playbook.opf.json against before it reads or
    # writes overrides.json there (the folder must hold THIS playbook).
    meta = {
        "content_hash": identity.get("content_hash"),
        "agreement_type": agreement.get("id"),
        "perspective": {
            "party": perspective.get("party"),
            "counterparty_type": perspective.get("counterparty_type"),
        },
        "overrides_file": OVERRIDES_FILENAME,
    }
    initial = _initial_overrides(out_dir)
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
    scripts = (
        "<!-- Machine-readable payloads. Extract a block, JSON-parse it, and verify\n"
        "     identity.content_hash over the canonical serialization (see\n"
        '     playbook_engine/canonicalize.py). "<" is escaped as "\\u003c" inside the\n'
        "     blocks; JSON parsing restores the original text exactly. -->\n"
        + _json_block("opf-canonical", raw)
        + _json_block("opf-digest", json.dumps(digest, indent=1, ensure_ascii=False))
        + _json_block("page-meta", json.dumps(meta, ensure_ascii=False))
        + _json_block("review-rows", json.dumps(review.rows, ensure_ascii=False))
        + _json_block("pf-data", json.dumps(pf_data, ensure_ascii=False))
        + (_json_block("overrides-initial", initial) if initial is not None else "")
    )

    html_out = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_esc(name)} — Playbook</title>
<link rel="icon" href="data:,">
<style>{CSS}</style>
</head>
<body>
<main>
<header class="cover">
  <div class="subtitle">Negotiation Playbook</div>
  <h1>{_esc(name)}</h1>
  <p class="meta">{_esc(version_line)}</p>
  {_stats_html(doc, clauses)}
</header>
<div role="tablist" aria-label="Playbook">{tab_buttons}</div>
{tab_panels}<footer>
Compiled by {_esc(compiler.get("name", "playbook-engine"))} {_esc(compiler.get("version", ""))} —
evidence-derived; every position traces to cited corpus text. Confidential work product.
Not a guarantee of pseudonymization: run the residue check before sharing.
</footer>
</main>
{scripts}<script>{JS}</script>
</body>
</html>
"""
    if out_file is not None:
        _write_atomic(out_file, html_out)
    return html_out


__all__ = [
    "INDEX_FILENAME",
    "TAB_IDS",
    "render_index_html",
]
