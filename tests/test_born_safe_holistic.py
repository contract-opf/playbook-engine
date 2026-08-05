"""Holistic born-safe regression test (issue #98).

Four separate leaks of content-derived text into persisted artifacts were
found and fixed REACTIVELY between 2026-08-02 and 2026-08-04 (#81, #83, #96,
#97) — each one because a reviewer happened to trip over it. This module is
the durable guard issue #98 asks for: it plants ONE synthetic
``provenance.known_entities`` name (``ENTITY_NAME`` below) in every plausible
input position SIMULTANEOUSLY, runs the real pipeline end-to-end (mine ->
project -> review -> inspection report -> AAR -> floor propose -> view ->
publish), and asserts the raw name survives in NONE of the written artifact
files — only its alias. A future field that forgets to scrub gets caught
here instead of by a reviewer's luck.

Positions planted, and the mechanism each one exercises:

  1. Document folder name          -> entity_registry.pseudonymize_document_id
  2. hints.yaml content (malformed,
     glued into an undefined YAML
     alias, issue #96's exact shape) -> version_orderer.HintsError /
                                        _yaml_error_detail ("never embed it")
  3. An unreadable version file
     whose path embeds the folder   -> pipeline._compute_doc_result's broad
                                        except-Exception handler (issue #98
                                        fix: version_ingest[].error)
  4. A clause heading + body text   -> entity_registry.pseudonymize_text
     (ordinary whole-word mention)     (the everyday, expected-safe case)
  5. An LLM-assigned taxonomy_id
     carrying the entity name       -> segmentation_qa._safe_taxonomy_id_repr
     (issue #97's established shape)
  6. A raising ScopeJudge whose
     failure message echoes         -> scope_gate.scope_gate's except-Exception
     "model output"                    handler (issue #98 fix: scope_rationale)

Deliberately NOT tested here (known, ACCEPTED residuals per the ticket's Out
of scope / Notes, not something this suite claims to close):
  - A taxonomy_id that IS identifier-shaped (lowercase snake_case) rejects
    into an ECHOED value, not a redacted one (segmentation_qa.
    _safe_taxonomy_id_repr's docstring calls this out explicitly). Position 5
    above uses the established issue #97 fixture shape instead (a
    non-identifier-shaped rejected id, which the taxonomy gate DOES redact to
    a length-only placeholder) — proven safe, not the accepted gap.
  - entity_registry._fuzzy_name_pattern is not hardened (ticket: out of
    scope) — every diagnostic-channel plant below therefore uses the GLUED
    form (no separating word boundary) specifically so the pass/fail proves
    the SOURCE fix ("never embed it"), not this regex, is what's holding.
  - The grounding gate (segmentation_grounding.GroundingError, gate 1 of 5)
    is NOT hardened. Its 8 raise sites (segmentation_grounding.py:158,160,
    162,169,179,181,191,203) interpolate the model's own unconstrained
    node_id/start_block_id/end_block_id/parent_id (no enum/pattern in
    llm_segmenter.py's schema, unlike taxonomy_id); segmentation_qa.py's
    run_gates and segment_verify_repair wrap that verbatim into a
    SegmentationQAError, which pipeline.py's sibling
    except-SegmentationQAError branch persists unbounded into
    version_ingest[].error -> quarantine.json / corpus_manifest.json /
    playbook.opf.json. This is a real, reported-but-unfixed residual (see
    issue #98's rescope comment, which records the verified line list for
    its own follow-up ticket, not yet filed) — NOT exercised by this suite.
    Planting a grounding failure (e.g. an out-of-range start_block_id
    carrying the entity name) would currently FAIL this test, by design:
    this list exists so that failure is expected, not a surprise.

Artifacts DELIBERATELY excluded from the leak sweep, not overlooked:
  - alias_map.json:            the held-out alias->real-name sidecar
                                (entity_registry.write_holdout_map) — its
                                entire purpose is reversing pseudonymization.
                                0600-permissioned, never embedded in the OPF.
                                Checked SEPARATELY below (must contain the
                                raw name, and must be 0600).
  - extraction_cache.jsonl,
    out_dir/.cache/,
    out_dir/normalized/*.json: raw, PRE-pseudonymization working caches.
                                pseudonymize_* is a single corpus-level pass
                                mine_corpus runs AFTER the whole per-document
                                ingest loop (see pipeline.mine_corpus's "Born-
                                safe pseudonymization" comment) — anything
                                written DURING that loop is architecturally
                                out of the born-safe boundary, exactly like
                                the source corpus files themselves — a
                                design-boundary fact, not a residue leak.
                                Only extraction_cache.jsonl is actually
                                PINNED by an assertion in this module (see
                                test_extraction_cache_is_a_pre_boundary_cache_not_a_residue_leak
                                at the bottom of this file: it asserts
                                ENTITY_NAME IS present there). out_dir/.cache/
                                and out_dir/normalized/*.json rest on the
                                SAME design-boundary reasoning above but are
                                NOT separately pinned by any assertion here —
                                nothing in this module would notice if either
                                of those two moved inside the born-safe
                                boundary.
  - viewer_notes.md:           only produced by chat_curate.apply_curate_commands
                                from a human-typed note string plus an
                                ALREADY-pseudonymized clause title -- not
                                exercised by the automatic mine/project/
                                publish flow this test drives end-to-end.

MUTATION-TESTED (see the module-level comment at the bottom of this file for
the exact command): reverting version_orderer._yaml_error_detail to its
pre-#96 ``return str(exc)`` form, with every other fix intact, makes
``test_born_safe_holistic_no_raw_entity_leak`` fail — proving this test is
load-bearing, not a snapshot that would pass regardless.

SECURITY NOTE: every name/institution in this file is synthetic
("Fictional University", "Alpha Corp") — no real counterparty, no EXOS
reference. No live LLM anywhere: every judge is a fake/deterministic
callable, matching test_pipeline_llm_seg.py's / test_publish.py's existing
convention.
"""

from __future__ import annotations

import json
import stat
from pathlib import Path
from typing import Any

import pytest
import yaml

from playbook_engine import publisher as publisher_module
from playbook_engine.aar import write_after_action_report
from playbook_engine.config import load_config
from playbook_engine.document_renderer import render_bundle_html, render_document_html
from playbook_engine.entity_registry import entity_slug
from playbook_engine.export_profile import RedactionFinding, VerifyFinding
from playbook_engine.floor_candidates import write_floor_candidates
from playbook_engine.inspection_report import write_inspection_report
from playbook_engine.pipeline import mine_corpus, project_playbook
from playbook_engine.playbook_assembler import write_playbook
from playbook_engine.review import write_review
from playbook_engine.segmentation_grounding import Block, SegNode
from playbook_engine.taxonomy import load_taxonomy
from playbook_engine.viewer import render_review_html

_TAXONOMY_PATH = Path(__file__).parent.parent / "spec" / "taxonomy" / "affiliation-agreement.yaml"

_RTF_PROLOGUE = (
    r"{\rtf1\ansi\deff0"
    r"{\fonttbl{\f0\froman\fcharset0 Times New Roman;}}"
    r"\f0\fs24 "
)
_RTF_EPILOGUE = r"}"


def _write_rtf(path: Path, body: str) -> None:
    path.write_text(_RTF_PROLOGUE + body + _RTF_EPILOGUE, encoding="utf-8")


# ---------------------------------------------------------------------------
# The one planted entity name, in every representation a leak could take.
# ---------------------------------------------------------------------------

ENTITY_NAME = "Fictional University"
ENTITY_SLUG = entity_slug(ENTITY_NAME)  # "fictional-university" (document_id/path form)
ENTITY_GLUED = ENTITY_NAME.replace(" ", "_")  # "Fictional_University" (defeats \w boundary)

_LEAK_NEEDLES = (ENTITY_NAME, ENTITY_SLUG, ENTITY_GLUED)

# ---------------------------------------------------------------------------
# Clause bodies. Each has a unique dispatch marker (segment_fn's contract has
# no doc_id parameter to key on — see llm_segmenter_batch/pipeline's own
# "_dispatching_segment_fn" convention) except the "main" document, which is
# the dispatch fallback. Two clauses each — scope_gate.MIN_CLAUSE_COUNT (2)
# is a deterministic pre-check that short-circuits BEFORE the (injected,
# raising) judge ever runs on a trivially-short document; these fixtures
# must clear it so _RaisingScopeJudge is actually reached.
# ---------------------------------------------------------------------------

_NOTICES_CLAUSE = (
    r"2. Notices\par "
    r"All notices under this agreement shall be delivered in writing to the "
    r"addresses set forth on the signature page.\par "
)

_HINTS_DOC_BODY = (
    r"1. Term\par "
    r"This clause covers HINTSDOC placement obligations between the parties "
    r"for the duration of the programme.\par " + _NOTICES_CLAUSE
)

_EXTRACT_OK_BODY = (
    r"1. Term\par "
    r"This agreement continues for the EXTRACTDOC duration stated on the "
    r"cover page of this agreement.\par " + _NOTICES_CLAUSE
)

# Heading AND body both carry the entity name as an ordinary whole-word
# mention — the everyday case entity_registry.pseudonymize_text is proven to
# catch (position 4).
_MAIN_DOC_BODY = (
    rf"1. {ENTITY_NAME} Data Rights\par "
    rf"{ENTITY_NAME} shall retain exclusive ownership of all placement data "
    r"submitted by Alpha Corp under this agreement.\par " + _NOTICES_CLAUSE
)

_TAXONOMY_DOC_BODY = (
    r"1. Confidentiality\par "
    r"TAXONOMYDOC obligations apply to all information disclosed under this "
    r"clause.\par " + _NOTICES_CLAUSE
)


def _clean_pairs_segment_fn(taxonomy_id: str) -> Any:
    """A segment_fn pairing consecutive (heading, body) blocks into
    gate-passing nodes — N/2 nodes for N blocks (mirrors
    test_pipeline_llm_seg.py's ``_fake_segment_fn`` pairing convention).
    Every node gets the same *taxonomy_id*; its exact value is irrelevant to
    what this module tests.
    """

    def _fn(canonical_text: str, blocks: list[Block]) -> list[SegNode]:
        del canonical_text
        nodes: list[SegNode] = []
        order = 1
        for i in range(0, len(blocks) - 1, 2):
            heading, body = blocks[i], blocks[i + 1]
            nodes.append(
                SegNode(
                    node_id=f"n{order}",
                    parent_id=None,
                    order=order,
                    heading=heading.text,
                    taxonomy_id=taxonomy_id,
                    start_block_id=heading.block_id,
                    end_block_id=body.block_id,
                    start_quote=heading.text[:10],
                    end_quote=body.text[-10:],
                )
            )
            order += 1
        return nodes

    return _fn


_hints_segment_fn = _clean_pairs_segment_fn("governing_law")
_extract_ok_segment_fn = _clean_pairs_segment_fn("governing_law")
_main_segment_fn = _clean_pairs_segment_fn("indemnification")

#: Position 5 — an out-of-enum taxonomy_id carrying the LLM-"assigned"
#: entity name, mirroring test_segmentation_qa.py's
#: test_taxonomy_gate_failure_redacts_non_identifier_rejected_id EXACTLY
#: (issue #97's established, already-safe shape: NOT identifier-shaped —
#: it has a space and mixed case — so segmentation_qa._safe_taxonomy_id_repr
#: redacts it to a length-only placeholder rather than echoing it). Single
#: node spanning every block passes grounding/coverage/reconstruction/tree
#: trivially and fails ONLY the taxonomy gate, deterministically on every
#: repair attempt.
_TAXONOMY_LEAK_REJECTED_ID = f"{ENTITY_NAME}_2023"


def _taxonomy_leak_segment_fn(canonical_text: str, blocks: list[Block]) -> list[SegNode]:
    del canonical_text
    return [
        SegNode(
            node_id="n1",
            parent_id=None,
            order=1,
            heading=blocks[0].text,
            taxonomy_id=_TAXONOMY_LEAK_REJECTED_ID,
            start_block_id=blocks[0].block_id,
            end_block_id=blocks[-1].block_id,
        )
    ]


def _dispatch_segment_fn(canonical_text: str, blocks: list[Block]) -> list[SegNode]:
    """Routes by content marker — segment_fn's contract has no doc_id."""
    if "HINTSDOC" in canonical_text:
        return _hints_segment_fn(canonical_text, blocks)
    if "EXTRACTDOC" in canonical_text:
        return _extract_ok_segment_fn(canonical_text, blocks)
    if "TAXONOMYDOC" in canonical_text:
        return _taxonomy_leak_segment_fn(canonical_text, blocks)
    return _main_segment_fn(canonical_text, blocks)


class _RaisingScopeJudge:
    """Position 6 — always raises, simulating an LLM call whose OWN failure
    message echoes document content back (e.g. a JSON-decode error quoting a
    fragment of the model's malformed response). Embeds the entity name
    GLUED (no separating word boundary) — same reasoning as the hints.yaml
    and taxonomy-id plants: this must be caught by scope_gate's fix (never
    interpolating str(exc)), not by entity_registry._fuzzy_name_pattern,
    which a glued token defeats.

    Raises unconditionally (for every document that reaches the scope
    gate): basis="judge_error" always sets in_scope=True, so the document is
    retained and the pipeline continues normally afterward — this is not a
    quarantine path.
    """

    def judge(self, tree: Any, agreement_type: Any) -> Any:
        raise RuntimeError(f"upstream model returned malformed JSON near token {ENTITY_GLUED}_2023")


# ---------------------------------------------------------------------------
# Corpus + config
# ---------------------------------------------------------------------------


def _build_corpus(tmp_path: Path) -> tuple[Path, Path, Path]:
    corpus_dir = tmp_path / "corpus"

    # Doc A: folder name embeds the entity (position 1); malformed hints.yaml
    # glues the entity name into an undefined YAML alias (position 2, issue
    # #96's exact regression shape) -> HintsError -> quarantined.
    doc_a = corpus_dir / f"{ENTITY_SLUG}-alpha"
    doc_a.mkdir(parents=True)
    _write_rtf(doc_a / "v1.rtf", _HINTS_DOC_BODY)
    (doc_a / "hints.yaml").write_text(f"order: [*{ENTITY_GLUED}_2023]\n", encoding="utf-8")

    # Doc B: folder name embeds the entity (position 1); v1 is an unreadable
    # .docx (position 3 — its ingest failure message would, pre-#98-fix,
    # embed the absolute source path, i.e. this folder name) while v2
    # ingests cleanly, so the document is NOT quarantined and
    # version_ingest[].error is reachable in corpus_manifest.json /
    # playbook.opf.json / review.json / the inspection report / report.md.
    doc_b = corpus_dir / f"{ENTITY_SLUG}-beta"
    doc_b.mkdir(parents=True)
    (doc_b / "v1.docx").write_bytes(b"not a real docx file")
    _write_rtf(doc_b / "v2.rtf", _EXTRACT_OK_BODY)

    # Doc C: folder name embeds the entity (position 1); heading AND body
    # text both mention the entity whole-word (position 4).
    doc_c = corpus_dir / f"{ENTITY_SLUG}-gamma"
    doc_c.mkdir(parents=True)
    _write_rtf(doc_c / "v1.rtf", _MAIN_DOC_BODY)

    # Doc D: out-of-enum taxonomy_id carries the entity name (position 5) ->
    # taxonomy gate fails -> quarantined (issue #97's established shape).
    doc_d = corpus_dir / f"{ENTITY_SLUG}-delta"
    doc_d.mkdir(parents=True)
    _write_rtf(doc_d / "v1.rtf", _TAXONOMY_DOC_BODY)

    cfg = {
        "agreement_type": {
            "id": "educational-affiliation",
            "name": "Educational Affiliation Agreement",
        },
        "baseline": {"template": None},
        "taxonomy": str(_TAXONOMY_PATH),
        "provenance": {
            "our_party_aliases": ["Alpha Corp"],
            "known_entities": [ENTITY_NAME],
        },
    }
    config_path = tmp_path / "playbook.config.yaml"
    config_path.write_text(yaml.dump(cfg), encoding="utf-8")

    out_dir = tmp_path / "out"
    return corpus_dir, config_path, out_dir


# ---------------------------------------------------------------------------
# Fake publish judges (no LLM, no network — matches test_publish.py)
# ---------------------------------------------------------------------------


class _CleanRedactionJudge:
    def evaluate_batch(self, samples: Any) -> list[RedactionFinding]:
        return [
            RedactionFinding(path=s.path, has_residue=False, rationale="No residue found.")
            for s in samples
        ]


class _CleanVerifyJudge:
    def evaluate_batch(self, samples: Any) -> list[VerifyFinding]:
        return [
            VerifyFinding(path=s.path, leaked=False, rationale="Independently confirmed clean.")
            for s in samples
        ]


# ---------------------------------------------------------------------------
# Leak scanner
# ---------------------------------------------------------------------------


def _assert_no_raw_leak(path: Path, artifact_label: str) -> None:
    """Assert *path* carries no raw form of the planted entity name.

    Reads the WRITTEN FILE on disk (never an in-memory dict/object — see the
    module docstring and the ticket's "verify against the WRITTEN artifact
    files" instruction). Fails naming the artifact, the file, the matched
    representation, and a line + snippet, so the next person gets a pointer,
    not a mystery.
    """
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8", errors="replace")
    lowered = text.lower()
    for needle in _LEAK_NEEDLES:
        idx = lowered.find(needle.lower())
        if idx == -1:
            continue
        line_no = text.count("\n", 0, idx) + 1
        start = max(0, idx - 60)
        end = min(len(text), idx + len(needle) + 60)
        snippet = text[start:end].replace("\n", "\\n")
        pytest.fail(
            f"RAW ENTITY NAME LEAK: artifact={artifact_label!r} field/content "
            f"matched {needle!r} at {path} line {line_no}\n    context: ...{snippet}..."
        )


# ---------------------------------------------------------------------------
# The holistic test
# ---------------------------------------------------------------------------


def test_born_safe_holistic_no_raw_entity_leak(tmp_path: Path) -> None:
    corpus_dir, config_path, out_dir = _build_corpus(tmp_path)
    taxonomy = load_taxonomy(_TAXONOMY_PATH)
    config = load_config(config_path)

    # no_cache=True is required: mine_corpus only wraps scope_judge in
    # BatchedScopeJudge (judgment.py's ALREADY-safe wrapper) when
    # no_cache=False. This test targets scope_gate.py's OWN except-Exception
    # branch (issue #98's fix), which is unreachable through the wrapper —
    # the exact condition `playbook judge` forces in production (see
    # scope_gate.py's fix comment).
    mine_corpus(
        corpus_dir=corpus_dir,
        config=config,
        taxonomy=taxonomy,
        out_dir=out_dir,
        scope_judge=_RaisingScopeJudge(),
        use_llm_segmentation=True,
        llm_segment_fn=_dispatch_segment_fn,
        entity_registry_path=tmp_path / "registry.json",
        no_cache=True,
    )

    # ---- Sanity: the dangerous inputs actually fired the paths they target
    # (not a vacuous pass) ----------------------------------------------
    quarantine = json.loads((out_dir / "quarantine.json").read_text(encoding="utf-8"))
    quarantined_reasons = {q["document_id"]: q["reason"] for q in quarantine}
    assert len(quarantined_reasons) == 2, (
        f"expected doc A (HintsError) and doc D (SegmentationQAError) quarantined, "
        f"got {sorted(quarantined_reasons)}"
    )
    assert any("HintsError" in r for r in quarantined_reasons.values()), (
        "doc A's malformed hints.yaml must have triggered HintsError"
    )
    assert any("SegmentationQAError" in r for r in quarantined_reasons.values()), (
        "doc D's out-of-enum taxonomy_id must have triggered the taxonomy gate"
    )

    # document_id is ALREADY aliased here (that is the fix working) — doc B
    # (the only 2-version document) is found by shape, not by its raw slug.
    manifest = json.loads((out_dir / "corpus_manifest.json").read_text(encoding="utf-8"))
    two_version_docs = [d for d in manifest if len(d.get("version_ingest") or []) == 2]
    assert len(two_version_docs) == 1, (
        f"expected exactly one 2-version document (doc B); got {len(two_version_docs)}"
    )
    beta_doc = two_version_docs[0]
    assert not beta_doc["document_id"].startswith(ENTITY_SLUG), (
        f"doc B's document_id must be ALIASED in corpus_manifest.json, not the raw slug: "
        f"{beta_doc['document_id']!r}"
    )
    beta_versions = {v["version"]: v for v in beta_doc["version_ingest"]}
    assert beta_versions["v1"]["status"] == "failed", (
        "doc B's garbage .docx must have failed ingest — sanity check for "
        "the version_ingest[].error plant"
    )
    assert beta_versions["v1"]["error"] == "ExtractionError", (
        "issue #98 fix: version_ingest[].error must be the exception TYPE "
        f"only, got {beta_versions['v1']['error']!r}"
    )
    assert beta_versions["v2"]["status"] == "ok"

    # scope.json's document_id is ALSO the ALIASED form (pipeline.py aliases
    # scope_log entries too — see the "Scope log keys on document_id too"
    # comment). Doc A's scope decision IS computed (_RaisingScopeJudge runs
    # before Hints.load) but never reaches scope.json — HintsError propagates
    # out of _compute_doc_result before it can return scope_decision to be
    # logged, same as doc D never reaching the scope gate at all (its
    # SegmentationQAError fires earlier still, during L1 segmentation) — so
    # only docs B and C's judge_error decisions are recorded here.
    scope_entries = json.loads((out_dir / "scope.json").read_text(encoding="utf-8"))["documents"]
    judge_error_entries = [sd for sd in scope_entries if sd.get("basis") == "judge_error"]
    assert len(judge_error_entries) == 2, (
        "docs B and C must both show basis='judge_error' from "
        f"_RaisingScopeJudge; got {scope_entries}"
    )
    assert all(
        "RuntimeError" in sd["scope_rationale"] and ENTITY_GLUED not in sd["scope_rationale"]
        for sd in judge_error_entries
    ), (
        "issue #98 fix: scope_rationale must carry the exception TYPE only, "
        f"never the raised message; got {judge_error_entries}"
    )
    assert not any(sd["document_id"].startswith(ENTITY_SLUG) for sd in scope_entries), (
        f"scope.json document_id must be ALIASED, not the raw slug: {scope_entries}"
    )

    # ---- Run the rest of the pipeline end-to-end ------------------------
    playbook = project_playbook(out_dir, config, taxonomy)
    write_playbook(playbook, out_dir / "playbook.opf.json")  # already written by project_playbook;
    # re-affirms the on-disk file is what we scan below, not the in-memory dict.

    write_review(out_dir)
    write_after_action_report(out_dir, out_dir / "report.md")  # also writes report.json
    write_inspection_report(out_dir, out_dir / "inspection_report.md")
    write_floor_candidates(out_dir)

    render_document_html(
        out_dir, out_dir / "playbook.document.html"
    )  # alias_map=None: born-safe path
    render_bundle_html(out_dir, out_dir / "playbook.opf.html")
    render_review_html(out_dir, out_dir / "playbook.review.html")

    known_entity_names = [ENTITY_NAME]
    report = publisher_module.publish_playbook(
        playbook,
        redaction_judge=_CleanRedactionJudge(),
        verify_judge=_CleanVerifyJudge(),
        known_entity_names=known_entity_names,
        published_at="2026-08-04T00:00:00Z",
    )
    write_playbook(report.doc, out_dir / "playbook.public.json")

    # Required by the ticket: the deterministic backstop itself must see
    # zero hits on the published doc.
    backstop_hits = publisher_module._entity_backstop_scan(report.doc, known_entity_names)
    assert backstop_hits == [], f"publisher._entity_backstop_scan found hits: {backstop_hits}"

    # ---- Sweep every persisted artifact this ticket lists (plus scope.json,
    # found during the audit and not in the ticket's original list) --------
    artifacts = [
        ("quarantine.json", out_dir / "quarantine.json"),
        ("corpus_manifest.json", out_dir / "corpus_manifest.json"),
        ("scope.json", out_dir / "scope.json"),
        ("observations.jsonl", out_dir / "observations.jsonl"),
        ("template_observations.jsonl", out_dir / "template_observations.jsonl"),
        ("round_moves.jsonl", out_dir / "round_moves.jsonl"),
        ("playbook.opf.json", out_dir / "playbook.opf.json"),
        ("review.json", out_dir / "review.json"),
        ("coherence_flags.json", out_dir / "coherence_flags.json"),
        ("floor.candidates.json", out_dir / "floor.candidates.json"),
        ("report.md (aar)", out_dir / "report.md"),
        ("report.json (aar)", out_dir / "report.json"),
        ("inspection_report", out_dir / "inspection_report.md"),
        ("playbook.document.html", out_dir / "playbook.document.html"),
        ("playbook.opf.html", out_dir / "playbook.opf.html"),
        ("playbook.review.html", out_dir / "playbook.review.html"),
        ("playbook publish output", out_dir / "playbook.public.json"),
    ]
    for trail_file in sorted((out_dir / "trail").glob("*.json")):
        artifacts.append((f"trail/{trail_file.name}", trail_file))

    for label, path in artifacts:
        assert path.exists(), f"expected artifact missing: {label} ({path})"
        _assert_no_raw_leak(path, label)

    # ---- The two places the raw name IS expected: the two sensitive
    # sidecars (never swept above) ----------------------------------------
    # 1. out_dir/alias_map.json — write_holdout_map's output, the
    #    alias->real-name reverse map for THIS run.
    holdout_path = out_dir / "alias_map.json"
    assert holdout_path.exists()
    assert ENTITY_NAME in holdout_path.read_text(encoding="utf-8"), (
        "alias_map.json is EXPECTED to carry the real name — reversing "
        "pseudonymization is its entire purpose. Its absence here would mean "
        "pseudonymization never ran at all, silently invalidating every "
        "assertion above."
    )
    holdout_mode = stat.S_IMODE(holdout_path.stat().st_mode)
    assert holdout_mode == 0o600, f"alias_map.json must be 0600, got {oct(holdout_mode)}"

    # 2. The corpus-wide entity registry (entity_registry_path) — the
    #    cross-run source alias_map.json is derived from; equally sensitive.
    registry_path = tmp_path / "registry.json"
    assert registry_path.exists()
    assert ENTITY_NAME in registry_path.read_text(encoding="utf-8"), (
        "entity_registry.json (the corpus-wide registry) is EXPECTED to "
        "carry the real name for the same reason as alias_map.json above."
    )
    registry_mode = stat.S_IMODE(registry_path.stat().st_mode)
    assert registry_mode == 0o600, f"entity_registry.json must be 0600, got {oct(registry_mode)}"


# ---------------------------------------------------------------------------
# Pre-pseudonymization working caches: verified once, explicitly, that they
# DO carry the raw name (confirming the design-boundary claim in the module
# docstring above is accurate, not just asserted) — never swept by the loop
# above, and never should be.
# ---------------------------------------------------------------------------


def test_extraction_cache_is_a_pre_boundary_cache_not_a_residue_leak(tmp_path: Path) -> None:
    """extraction_cache.jsonl legitimately holds raw canonical_text/blocks —
    it caches extract_blocks()'s output, which runs BEFORE
    entity_registry.pseudonymize_text is ever applied (mine_corpus's single
    corpus-level pseudonymization pass runs AFTER the whole per-document
    ingest loop — see pipeline.mine_corpus's "Born-safe pseudonymization"
    comment). This is not a residue leak to close; it is the documented
    design boundary. Confirmed here so the boundary claim in this module's
    docstring is verified, not merely asserted.
    """
    corpus_dir, config_path, out_dir = _build_corpus(tmp_path)
    taxonomy = load_taxonomy(_TAXONOMY_PATH)
    config = load_config(config_path)

    from playbook_engine.extraction import ExtractionCache

    cache = ExtractionCache(out_dir / "extraction_cache.jsonl")
    mine_corpus(
        corpus_dir=corpus_dir,
        config=config,
        taxonomy=taxonomy,
        out_dir=out_dir,
        scope_judge=_RaisingScopeJudge(),
        use_llm_segmentation=True,
        llm_segment_fn=_dispatch_segment_fn,
        entity_registry_path=tmp_path / "registry.json",
        extraction_cache=cache,
        no_cache=True,
    )

    cache_path = out_dir / "extraction_cache.jsonl"
    assert cache_path.exists()
    assert ENTITY_NAME in cache_path.read_text(encoding="utf-8"), (
        "extraction_cache.jsonl is expected to carry the raw entity name "
        "(doc C's heading/body) — it is a pre-pseudonymization cache by "
        "design, not a residue-leak surface. If this assertion starts "
        "failing, the design boundary itself has changed and this module's "
        "docstring needs updating, not celebrating."
    )


# ---------------------------------------------------------------------------
# Mutation-testing note (issue #98 acceptance criterion — reported, not
# automated: reverting a fix and asserting failure would leave the fix
# reverted in the tree, which defeats the point).
#
# Verification performed for this ticket:
#   1. `git stash` this test file out, confirm the suite is green.
#   2. Temporarily replace version_orderer._yaml_error_detail's body with
#      `return str(exc)` (its pre-#96 form) — no other change.
#   3. Run:
#      source .venv/bin/activate && pytest tests/test_born_safe_holistic.py -q
#      Result: test_born_safe_holistic_no_raw_entity_leak FAILS — the
#      RAW ENTITY NAME LEAK assertion fires on quarantine.json (doc A's
#      reason embeds "Fictional_University" via ComposerError's
#      "found undefined alias %r" — issue #96 round 2's exact regression).
#   4. Revert the mutation; confirm the suite is green again.
#
# This proves the test is load-bearing against a REAL regression of an
# already-landed fix, not merely a snapshot of current behavior.
# ---------------------------------------------------------------------------
