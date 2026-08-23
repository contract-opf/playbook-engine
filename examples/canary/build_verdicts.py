#!/usr/bin/env python3
"""Regenerate examples/canary/segment-verdicts.jsonl for the canary corpus.

``segmentation.agent: true`` makes segmentation a store-backed, key-free
loop: ``playbook segment`` queues each un-segmented document's ``Block``
stream, an agent partitions those blocks into clause ranges, and
``playbook segment-apply`` loads the result into the segmentation cache so
``playbook mine`` replays it with no API call.

For the canary that partition must be **committed and stable**, not produced
by a live agent — so this script plays the agent's part deterministically:
every block matching ``N. Heading`` opens a clause that runs to the block
before the next heading, and the blocks before the first heading become the
preamble clause. Each clause's ``taxonomy_id`` comes from the fixed
``_TAXONOMY_BY_HEADING`` map below.

The result is gated through the very same
``segmentation_qa.run_gates`` that ``playbook segment-apply`` applies, so a
partition that would be rejected at apply time fails here instead.

IMPORTANT: the segmentation cache keys on the document's **canonical text**,
which is a function of the source bytes AND the extractor environment. So
this script must be re-run after ``build_corpus.py`` changes the corpus, and
must be run under the same pinned ``extraction.extractor`` the canary config
declares (it reads that config, so this happens automatically).

Run from the repo root, after build_corpus.py::

    python examples/canary/build_verdicts.py

Then refresh the committed expectations::

    python examples/canary/build_expected.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from playbook_engine.config import load_config
from playbook_engine.extraction import extract_blocks
from playbook_engine.pipeline import _discover_versions
from playbook_engine.segmentation_grounding import SegNode
from playbook_engine.segmentation_qa import run_gates
from playbook_engine.taxonomy import load_taxonomy

_HERE = Path(__file__).resolve().parent
_CONFIG = _HERE / "config.yaml"
_CORPUS = _HERE / "corpus"
_VERDICTS = _HERE / "segment-verdicts.jsonl"

_HEADING_RE = re.compile(r"^(\d+)\.\s+(.*)$")

# Fixed heading -> taxonomy id map (ids from spec/taxonomy/nda.yaml). Kept
# explicit rather than fuzzy-matched so the committed segmentation is a
# stated decision, not the output of a similarity threshold that could drift.
_TAXONOMY_BY_HEADING: dict[str, str] = {
    "Purpose": "purpose_permitted_use",
    "Definition of Confidential Information": "definition_confidential_info",
    "Exclusions from Confidential Information": "exclusions_from_confidential",
    "Standard of Care": "standard_of_care",
    "Permitted Disclosure to Representatives": "permitted_disclosure_reps",
    "Term and Survival": "survival_period",
    "Return or Destruction": "return_or_destruction",
    "Governing Law": "governing_law",
    "Limitation of Liability": "limitation_of_liability",
}

_PREAMBLE_TAXONOMY_ID = "parties_and_recitals"
_PREAMBLE_HEADING = "Parties and Recitals"

# The unnumbered execution page appended to each negotiation's LAST version.
# Split into its own node rather than swept into the trailing numbered clause,
# so the signature block does not pollute Governing Law / Limitation of
# Liability text. It carries no taxonomy_id — "unclassified" is the honest
# label for a signature page, and the taxonomy gate accepts null.
_EXECUTION_MARKER = "IN WITNESS WHEREOF"
_EXECUTION_HEADING = "Execution"


def _partition(blocks: list) -> list[SegNode]:
    """Partition *blocks* into one contiguous clause per numbered heading."""
    starts: list[int] = [i for i, b in enumerate(blocks) if _HEADING_RE.match(b.text.strip())]
    if not starts:
        raise SystemExit("no numbered headings found — corpus shape changed?")
    execution_at = next(
        (i for i, b in enumerate(blocks) if b.text.strip().startswith(_EXECUTION_MARKER)),
        None,
    )
    body_end = (execution_at - 1) if execution_at is not None else len(blocks) - 1

    ranges: list[tuple[str | None, int, int]] = []
    if starts[0] > 0:
        ranges.append((None, 0, starts[0] - 1))
    for n, first in enumerate(starts):
        last = (starts[n + 1] - 1) if n + 1 < len(starts) else body_end
        ranges.append((blocks[first].text.strip(), first, last))

    if execution_at is not None:
        ranges.append((_EXECUTION_MARKER, execution_at, len(blocks) - 1))

    nodes: list[SegNode] = []
    for order, (raw_heading, first, last) in enumerate(ranges, start=1):
        if raw_heading is None:
            heading, taxonomy_id = _PREAMBLE_HEADING, _PREAMBLE_TAXONOMY_ID
        elif raw_heading == _EXECUTION_MARKER:
            heading, taxonomy_id = _EXECUTION_HEADING, None
        else:
            match = _HEADING_RE.match(raw_heading)
            assert match is not None
            heading = raw_heading
            label = match.group(2).strip()
            if label not in _TAXONOMY_BY_HEADING:
                raise SystemExit(f"unmapped heading {label!r} — add it to _TAXONOMY_BY_HEADING")
            taxonomy_id = _TAXONOMY_BY_HEADING[label]
        nodes.append(
            SegNode(
                node_id=f"c{order}",
                parent_id=None,
                order=order,
                heading=heading,
                taxonomy_id=taxonomy_id,
                start_block_id=blocks[first].block_id,
                end_block_id=blocks[last].block_id,
                start_quote=blocks[first].text[:40],
                end_quote=blocks[last].text[-40:],
            )
        )
    return nodes


def _node_to_dict(node: SegNode) -> dict:
    return {
        "node_id": node.node_id,
        "parent_id": node.parent_id,
        "order": node.order,
        "heading": node.heading,
        "taxonomy_id": node.taxonomy_id,
        "start_block_id": node.start_block_id,
        "end_block_id": node.end_block_id,
        "start_quote": node.start_quote,
        "end_quote": node.end_quote,
    }


def main() -> None:
    cfg = load_config(_CONFIG)
    taxonomy = load_taxonomy(cfg.taxonomy_path)
    taxonomy_ids = [e.id for e in taxonomy.classifier_entries()]

    lines: list[str] = []
    seen: set[str] = set()
    for doc_dir in sorted(d for d in _CORPUS.iterdir() if d.is_dir()):
        for path in _discover_versions(doc_dir):
            canonical_text, blocks, _label = extract_blocks(
                path, extractor=cfg.extraction.extractor
            )
            if canonical_text in seen:
                continue
            seen.add(canonical_text)
            nodes = _partition(blocks)
            # Same gate `playbook segment-apply` runs — fail here, not there.
            run_gates(
                canonical_text,
                blocks,
                nodes,
                taxonomy_ids=taxonomy_ids,
                document_id=doc_dir.name,
                version=path.stem,
            )
            lines.append(
                json.dumps(
                    {
                        "canonical_text": canonical_text,
                        "nodes": [_node_to_dict(n) for n in nodes],
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            print(f"{doc_dir.name}/{path.name}: {len(nodes)} clause(s) over {len(blocks)} block(s)")

    _VERDICTS.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {_VERDICTS} ({len(lines)} segmentation(s))")


if __name__ == "__main__":
    main()
