"""Template standards — what ``our_standard`` is for a clause type (issue #242).

A clause type's standard is the template's COMPLETE operative language for it,
not whichever template node happened to be classified first. Two facts about a
real form decide that, and both used to be wrong:

1. **A clause is often several template nodes.** A lead-in sentence ending
   in a colon sits in one node and the operative limbs it introduces in the
   next ones; a form may also carry one clause type across several
   sections. Every template node carrying the clause type's taxonomy id is
   part of its standard, in document order, joined into one text. (The
   standard check — :func:`playbook_engine.deviation_classifier.is_standard_text`
   — and the origin reference of a struck clause already read every node;
   only ``our_standard`` itself took the first one.)

2. **Form front matter is not a clause.** A cover table of fill-in fields
   (party names, notice address, term dates, each a bracketed placeholder) is
   the order form wrapped around the terms. Taking it as a clause type's
   standard makes every real clause of that type "unrankable" against a
   table of blanks. Front matter (:func:`form_front_matter`) is a node SHAPED
   like a fill-in table (:func:`is_form_front_matter`: blanks in a good share
   of its cells, no sentence anywhere) that comes BEFORE the template's first
   operative clause. It contributes to no standard, whatever taxonomy id the
   segmenter gave it. Both conditions are needed: an operative schedule later
   in the form (a fee table of ``[Amount]`` cells, a notices address table)
   has the shape but not the position, and stays a clause.

Front matter is decided ONCE, where every template node is still visible: the
template-observation producer
(``playbook_engine.pipeline._template_observations_from_classified``) runs
:func:`front_matter_indices` over the full classified template, unclassified
nodes included (an unclassified operative sentence still ends the front
matter), and records the verdict on the observation
(``Observation.form_front_matter``, persisted as ``x_form_front_matter`` in
``template_observations.jsonl``). Unclassified nodes never become template
observations, so deciding it later, from the observations alone, would miss
them; reading the persisted flag is what keeps ``project`` in agreement with
``mine``.

Front matter leaves ``our_standard`` (and the content-similarity exemplars
built from it) only. It stays a template observation and stays in the origin
reference and ``standard`` fact of every deal clause (issue #216): our own
cover-table text struck before signing is our concession, never the
counterparty's refused ask.

Nothing here is specific to one form or one agreement type. The tests in
``tests/test_template_standards.py`` use synthetic text only.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from playbook_engine.observation_builder import Observation

__all__ = [
    "STANDARD_NODE_SEPARATOR",
    "TemplateStandard",
    "form_front_matter",
    "front_matter_indices",
    "is_form_front_matter",
    "standard_observations",
    "template_standards",
]

#: Joins a clause type's template nodes into one ``our_standard.text``. A blank
#: line keeps each node's own heading readable; the standard check and the
#: equivalence key both collapse whitespace, so the choice never changes a
#: comparison.
STANDARD_NODE_SEPARATOR = "\n\n"

#: A fill-in field: ``[Institution Name]``, ``____``, ``{{party}}``, ``<<date>>``.
#: A bracketed span followed by ``(`` is a markdown link, not a blank.
_PLACEHOLDER = re.compile(r"\[[^\[\]\n]{2,80}\](?!\()|_{3,}|\{\{[^{}\n]+\}\}|<<[^<>\n]+>>")

#: Cell separator the extractors render table rows with.
_CELL_SEPARATOR = "|"

#: A cell with this many words that ends in sentence punctuation is language,
#: not a field value.
_SENTENCE_MIN_WORDS = 6
_SENTENCE_END = (".", "!", "?", ";")

#: A fill-in table has a header and a data row of at least two columns.
_MIN_TABLE_CELLS = 4

#: At least one cell in this many holds a fill-in placeholder: in a two-column
#: field/value table (header row included) that is about half its value
#: column. An operative schedule with a blank or two among filled values (an
#: insurance limit, a service-level target) falls short of it.
_BLANK_CELL_SHARE_DENOMINATOR = 4


def _is_sentence(cell: str) -> bool:
    cell = cell.strip()
    return len(cell.split()) >= _SENTENCE_MIN_WORDS and cell.endswith(_SENTENCE_END)


def _lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def _cells(row: str) -> list[str]:
    return [cell.strip() for cell in row.split(_CELL_SEPARATOR) if cell.strip()]


def _has_sentence(text: str) -> bool:
    """Whether any table cell, or any line outside a table, is a sentence."""
    for line in _lines(text):
        parts = _cells(line) if _CELL_SEPARATOR in line else [line]
        if any(_is_sentence(part) for part in parts):
            return True
    return False


def is_form_front_matter(text: str) -> bool:
    """Whether *text* is SHAPED like a form's fill-in table rather than a clause.

    The shape alone never excludes a node: :func:`front_matter_indices` also
    requires it to come before the template's first operative clause.

    Judged on CELLS, not lines, because the extractors disagree on how a table
    reaches us: docling renders one row per line, the legacy DOCX extractor
    joins a whole table into a single ``" | "``-separated line. All of: the
    table lines (those with a cell separator) make up at least half of the
    node's lines and hold at least four cells (two rows of two, however they
    are broken into lines); at least one cell in four holds a fill-in
    placeholder; and no cell, and no line outside the table, is itself a
    sentence. A responsibilities matrix, a schedule of filled values with one
    blank, a table under a prose sentence, or a clause with a bracketed blank
    in a sentence is none of these and stays a clause.
    """
    lines = _lines(text)
    rows = [line for line in lines if _CELL_SEPARATOR in line]
    if not rows or len(rows) * 2 < len(lines):
        return False
    cells = [cell for row in rows for cell in _cells(row)]
    if len(cells) < _MIN_TABLE_CELLS:
        return False
    blanks = sum(1 for cell in cells if _PLACEHOLDER.search(cell))
    if not blanks or blanks * _BLANK_CELL_SHARE_DENOMINATOR < len(cells):
        return False
    return not _has_sentence(text)


def front_matter_indices(texts: Sequence[str]) -> set[int]:
    """The positions in *texts* that are form front matter.

    *texts* is EVERY node of the template in document order, classified or
    not: the template-observation producer calls this before it drops the
    unclassified nodes. A node is front matter when it is shaped like a
    fill-in table (:func:`is_form_front_matter`) and comes before the
    template's first operative clause, which is the first node, classified or
    not, with a sentence in it. Empty nodes, and nodes with neither shape (a
    title line), neither count nor end the front matter.
    """
    front: set[int] = set()
    for i, text in enumerate(texts):
        if not text.strip():
            continue
        if is_form_front_matter(text):
            front.add(i)
            continue
        if _has_sentence(text):
            break  # the first operative clause: no front matter after it
    return front


def form_front_matter(observations: Iterable[Observation]) -> list[Observation]:
    """The template observations the producer marked as form front matter
    (``Observation.form_front_matter``, see :func:`front_matter_indices`), in
    the order given.

    These nodes contribute to no ``our_standard``. They are still template
    observations and part of the origin reference (see the module docstring).
    """
    return [obs for obs in observations if obs.form_front_matter]


def standard_observations(observations: Iterable[Observation]) -> list[Observation]:
    """The template observations that can be (part of) a standard, in order:
    classified, with text, and not form front matter (:func:`form_front_matter`).
    *observations* must be in document order."""
    return [
        obs
        for obs in observations
        if obs.taxonomy_id is not None and obs.full_text.strip() and not obs.form_front_matter
    ]


@dataclass(frozen=True)
class TemplateStandard:
    """One clause type's standard: the joined text and the nodes it came from
    (template observations, document order)."""

    taxonomy_id: str
    text: str
    nodes: tuple[Observation, ...]


def template_standards(observations: Iterable[Observation]) -> dict[str, TemplateStandard]:
    """``{taxonomy_id: TemplateStandard}`` from the template's observations,
    which must be in document order (the order the template was segmented in).

    A clause type whose every node is empty or form front matter has no
    entry: it has no standard, and its ``our_standard`` is null.
    """
    nodes_by_tid: dict[str, list[Observation]] = {}
    for obs in standard_observations(observations):
        assert obs.taxonomy_id is not None  # standard_observations' filter
        nodes_by_tid.setdefault(obs.taxonomy_id, []).append(obs)
    return {
        tid: TemplateStandard(
            taxonomy_id=tid,
            text=STANDARD_NODE_SEPARATOR.join(o.full_text for o in nodes),
            nodes=tuple(nodes),
        )
        for tid, nodes in nodes_by_tid.items()
    }
