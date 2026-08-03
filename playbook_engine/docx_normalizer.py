"""Pre-normalize tracked-changes/commented DOCX files so docling stops
crashing on redlines (issue #84).

docling 2.x's DOCX backend crashes on tracked-changes/comment nodes
(``etree.QName`` on a comment factory) — exactly what redline drafts
contain, the highest-value documents in a negotiation corpus. Routing those
through the legacy per-format DOCX adapter (no heading detection at all)
loses real document structure for the versions that matter most.

:func:`normalize_tracked_docx` produces a temp copy of the document with an
*accepted-changes* view of the body XML: insertions (``w:ins``) are unwrapped
in place (their children promoted to the surrounding content, in document
order), deletions (``w:del``) are removed along with their contents, and
comment markup (``w:commentReference``/``w:commentRangeStart``/
``w:commentRangeEnd``, plus the ``comments.xml`` part itself when present) is
dropped. Feeding docling this clean copy recovers real structure instead of
degrading straight to the legacy fallback — see
:mod:`playbook_engine.extraction`, which retries docling on the normalized
copy before falling back.

Semantic contract: the normalized copy's plain text must equal the
accepted-changes view :func:`playbook_engine.docx_ingester._extract_para_text`
already produces for the ORIGINAL document (insertions included, deletions
excluded) — both paths must agree on document content. See
``tests/test_docx_normalizer.py``.

Traversal shape: unlike :mod:`playbook_engine.docx_ingester`'s read-only text
extraction — which must recurse into a fixed allowlist of wrapper elements
(``w:hyperlink``, ``w:smartTag``, ``w:sdt``/``w:sdtContent``, ``w:fldSimple``)
one paragraph at a time, so it can build an offset stream without
double-counting text — this module only deletes or unwraps elements, so a
single ``element.iter(tag)`` sweep over the whole body finds every
``w:ins``/``w:del``/comment-marker element regardless of how deeply it is
nested or what wraps it (hyperlinks, smart tags, content controls, table
cells — tracked changes inside table cells are normalized too here, even
though :mod:`docx_ingester` does not record them in its ``TrackedChanges``
side-channel; see that module's docstring "known gap"). This is strictly
more thorough than mirroring the allowlisted recursion would be, with less
code, since ``iter()`` cannot miss a wrapper type the allowlist forgot.

Not in scope: capturing tracked-change *metadata* here (author/date
side-channel — a separate ticket, which must read the ORIGINAL file, never
this normalized copy; see issue #84's Out-of-scope section). Separately —
this module's own documented scope decision, not something the ticket
dictates — handling stops at inline run-level ``w:ins``/``w:del``: row- or
paragraph-mark-level tracked insert/delete markers carry no text of their
own and are simply discarded as inert empty elements by the same sweep.

Pure module: no imports beyond python-docx/lxml and the stdlib, so it stays
independently testable without pulling in the rest of the engine.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

from docx import Document
from docx.oxml.ns import qn

# Tags this module mutates. All body-XML, per the ticket's scope — headers/
# footers are separate OPC parts docx_ingester never reads either.
_W_INS = qn("w:ins")
_W_DEL = qn("w:del")
_W_COMMENT_REFERENCE = qn("w:commentReference")
_W_COMMENT_RANGE_START = qn("w:commentRangeStart")
_W_COMMENT_RANGE_END = qn("w:commentRangeEnd")


def normalize_tracked_docx(path: Path) -> Path:
    """Return the path to a fresh temp copy of *path* with an accepted-changes view.

    - ``w:ins`` elements are unwrapped in place: their children are spliced
      into the parent at the position the wrapper occupied, preserving
      document order, then the (now-empty) wrapper is removed. This covers
      ``w:ins`` nested inside ``w:hyperlink``/``w:smartTag``/``w:sdtContent``/
      table cells identically to a top-level paragraph, since the sweep does
      not care what the immediate parent is.
    - ``w:del`` elements are removed entirely, contents and all — deleted
      text must never reach docling.
    - ``w:commentReference``/``w:commentRangeStart``/``w:commentRangeEnd``
      marker elements are removed (they carry no text of their own).
    - The ``comments.xml`` part (and any sibling comment-family part, e.g.
      ``commentsExtended.xml``) is dropped from the saved package by
      removing the document part's relationship to it — see
      :func:`_drop_comments_part`.

    The returned path is a fresh temp file the CALLER owns and must delete
    (e.g. ``finally: normalized_path.unlink(missing_ok=True)``) once done
    with it — this function does not clean up after itself, since the whole
    point of returning a path (rather than, say, bytes) is to hand it to
    another process (docling) after this call returns.
    """
    document = Document(str(path))
    body = document.element.body

    # Deletions first: if a deletion ever contains a nested insertion (not
    # normal Word output, but not disallowed by the schema either), the
    # reject-deletions rule must win — removing the whole w:del subtree
    # before unwrapping w:ins ensures that. See module docstring "Not in
    # scope" for why this ordering is not otherwise load-bearing in practice.
    _remove_all(body, _W_DEL)
    _unwrap_all(body, _W_INS)
    _remove_all(body, _W_COMMENT_REFERENCE)
    _remove_all(body, _W_COMMENT_RANGE_START)
    _remove_all(body, _W_COMMENT_RANGE_END)
    _drop_comments_part(document)

    fd, tmp_name = tempfile.mkstemp(suffix=".docx", prefix="docx-normalized-")
    os.close(fd)
    tmp_path = Path(tmp_name)
    document.save(str(tmp_path))
    return tmp_path


def _remove_all(container: Any, tag: str) -> None:
    """Remove every *tag* element under *container*, contents and all.

    ``list(...)`` materializes the match set before mutating the tree —
    ``lxml``'s ``iter()`` walks live sibling links, and removing elements
    mid-traversal can skip siblings.
    """
    for elem in list(container.iter(tag)):
        parent = elem.getparent()
        if parent is not None:
            parent.remove(elem)


def _unwrap_all(container: Any, tag: str) -> None:
    """Replace every *tag* element under *container* with its own children,
    in place, preserving document order — an "accept this wrapper" splice.

    ``list(...)`` on both the outer search and each wrapper's own children is
    required for the same live-iteration reason as :func:`_remove_all`:
    ``parent.insert(...)`` on a child that is already attached elsewhere in
    an lxml tree MOVES it (detaching it from its current parent first), so
    iterating ``elem`` directly while splicing its own children out from
    under it would skip every other child.
    """
    for elem in list(container.iter(tag)):
        parent = elem.getparent()
        if parent is None:
            continue
        index = parent.index(elem)
        children = list(elem)
        for offset, child in enumerate(children):
            parent.insert(index + offset, child)
        parent.remove(elem)


def _drop_comments_part(document: Any) -> None:
    """Drop the document part's relationship(s) to the comments part(s), if any.

    python-docx only serializes parts still reachable from the package's
    relationship graph at save time (``OpcPackage.iter_parts``/``save``) —
    dropping the relationship is therefore sufficient to exclude
    ``word/comments.xml`` (and any same-family part, e.g.
    ``commentsExtended.xml``, matched via ``"comment" in reltype``) from both
    the saved ``[Content_Types].xml`` and the physical package entirely,
    rather than leaving a dangling reference now that every
    ``w:commentReference``/``w:commentRangeStart``/``w:commentRangeEnd``
    marker pointing into it has been removed (a dangling rel is exactly what
    the ticket's verifier corrections warn can make python-docx choke
    reopening the file — verified clean by
    ``tests/test_docx_normalizer.py``'s reopen check).
    """
    comment_rids = [
        rid for rid, rel in document.part.rels.items() if "comment" in rel.reltype.lower()
    ]
    for rid in comment_rids:
        document.part.drop_rel(rid)
