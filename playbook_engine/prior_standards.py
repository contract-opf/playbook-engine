"""Prior standards — our former forms as a factual standard history (issue #249, OPF 0.6).

Past deals mostly opened on OLDER versions of our own form
(``baseline.prior_templates``, issue #246). Without them our own former wording
read as the counterparty's: a deal that opened on last year's indemnity was an
``opened_with: non_standard`` deal and its opening a "changed opening". This
module is the pure half of recording them as ours:

* :class:`PriorForm` — one supplied form after it was segmented and classified
  exactly like the current template (same producer, same front-matter rule):
  its label, its config dates and its template-style observations in document
  order. ``mine`` persists the forms as ``prior_templates.json``; ``project``
  reads them back (no re-ingest).
* :func:`origin_standards` — the reference ``build_observations`` classifies a
  first draft against (after the current standard): per clause type, each
  form's template nodes in config order.
* :func:`prior_standard_entries` — the ``evidence.clauses[].prior_standards[]``
  entries: per clause type and form, the form's clause text as ordered parts
  with stable part ids (the same ``part_id`` rule as ``our_standard``), a
  citation of the clause inside the form (``document_id``
  ``prior_template:<label>``, ``version`` ``prior_template``: resolved through
  the form's own ``baseline.prior_templates[].sha256``, never through
  ``"template"``, which is today's form) and ``vs_standard``, the judgment of
  that text against TODAY's standard, part by part (the issue #243 machinery,
  subject role ``prior_standard``, filled by ``label_evidence``).

Nothing here is judged and nothing here says WHY a form changed; the only
reason ever shown is the owner's note (``evidence.clauses[].why_changed``).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from playbook_engine.observation_builder import Observation, PriorFormNodes
from playbook_engine.opf_accessors import PRIOR_TEMPLATE_VERSION, prior_template_document_id
from playbook_engine.template_standards import template_standards

__all__ = [
    "PRIOR_TEMPLATES_FILENAME",
    "PRIOR_TEMPLATES_VERSION",
    "PriorForm",
    "baseline_prior_templates",
    "origin_standards",
    "prior_standard_entries",
]

#: The persisted forms of a ``mine`` out-dir (read by ``project``). Absent when
#: no prior form is configured.
PRIOR_TEMPLATES_FILENAME = "prior_templates.json"
PRIOR_TEMPLATES_VERSION = 1


@dataclass(frozen=True)
class PriorForm:
    """One older version of our form, segmented and classified like the template.

    ``observations`` are template-style observations (``document_id`` and
    ``version`` ``"template"`` inside the mined store only) in document order,
    classified clauses only, with form front matter flagged
    (``Observation.form_front_matter``). What the playbook cites is the form
    itself (:func:`prior_standard_entries`), not ``"template"``.
    ``in_use_from`` / ``in_use_until`` are the config's own values (``YYYY``,
    ``YYYY-MM`` or ``YYYY-MM-DD``) or ``None``.
    """

    label: str
    observations: tuple[Observation, ...]
    in_use_from: str | None = None
    in_use_until: str | None = None
    sha256: str | None = None

    def nodes_by_tid(self) -> dict[str, list[str]]:
        """Every non-empty node of the form per clause type, document order,
        form front matter INCLUDED — the origin reference, exactly as the
        current template's ``template_std_nodes_by_tid`` (issue #216)."""
        out: dict[str, list[str]] = {}
        for obs in self.observations:
            if obs.taxonomy_id is not None and obs.full_text.strip():
                out.setdefault(obs.taxonomy_id, []).append(obs.full_text)
        return out


def origin_standards(
    forms: Sequence[PriorForm],
) -> dict[str, list[PriorFormNodes]]:
    """``{taxonomy_id: [(form label, nodes, config index), ...]}``.

    The index is the form's position in *forms* (``baseline.prior_templates``
    order), so a deal that reads as two forms alike is credited to the one
    listed first, whichever clause types the forms have in common.

    What ``observation_builder.build_observations`` matches a first draft
    against after the current standard (``prior_standards_by_tid``)."""
    out: dict[str, list[PriorFormNodes]] = {}
    for index, form in enumerate(forms):
        for tid, nodes in form.nodes_by_tid().items():
            out.setdefault(tid, []).append((form.label, nodes, index))
    return out


def baseline_prior_templates(forms: Iterable[PriorForm]) -> list[dict[str, Any]]:
    """``baseline.prior_templates[]`` of the playbook: each supplied form's
    label, config dates and content address (dates and hash only when known)."""
    out: list[dict[str, Any]] = []
    for form in forms:
        entry: dict[str, Any] = {"label": form.label}
        if form.in_use_from is not None:
            entry["in_use_from"] = form.in_use_from
        if form.in_use_until is not None:
            entry["in_use_until"] = form.in_use_until
        if form.sha256 is not None:
            entry["sha256"] = form.sha256
        out.append(entry)
    return out


def prior_standard_entries(
    forms: Sequence[PriorForm], taxonomy_ids: Iterable[str]
) -> dict[str, list[dict[str, Any]]]:
    """``evidence.clauses[].prior_standards`` per clause type, forms in config order.

    One entry per (form, clause type) for each of *taxonomy_ids* the form has a
    standard for — its classified, non-empty, non-front-matter nodes in
    document order (the rule ``our_standard`` follows,
    :func:`~playbook_engine.template_standards.template_standards`). A form
    with no text for a clause type has no entry for it.

    ``source_ref`` cites the clause in the FORM's own bytes (OPF-SPEC §4):
    ``document_id`` ``prior_template:<label>`` and ``version``
    ``prior_template``, resolved through the form's
    ``baseline.prior_templates[].sha256``; ``clause_path`` and ``char_span``
    are the form's own coordinates.

    ``part_id`` and ``order`` of the parts are stamped later by
    ``precedent.restamp_evidence`` (they are functions of the shipped text);
    ``vs_standard`` starts ``None`` and is set by
    ``equivalence.label_evidence``.
    """
    wanted = set(taxonomy_ids)
    out: dict[str, list[dict[str, Any]]] = {tid: [] for tid in sorted(wanted)}
    for form in forms:
        for tid, std in template_standards(form.observations).items():
            if tid not in wanted:
                continue
            first = std.nodes[0].citation
            single = len(std.nodes) == 1
            source_ref: dict[str, Any] = {
                "document_id": prior_template_document_id(form.label),
                "version": PRIOR_TEMPLATE_VERSION,
                "clause_path": (
                    first.clause_path
                    if single
                    else ", ".join(n.citation.clause_path or "?" for n in std.nodes)
                ),
            }
            if single and first.char_span is not None:
                source_ref["char_span"] = list(first.char_span)
            out[tid].append(
                {
                    "form_label": form.label,
                    "text": std.text,
                    "parts": [
                        {"part_id": "", "order": i + 1, "text": n.full_text}
                        for i, n in enumerate(std.nodes)
                    ],
                    "source_ref": source_ref,
                    "vs_standard": None,
                }
            )
    return out
