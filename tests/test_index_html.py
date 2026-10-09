"""``index.html`` — the one tabbed page of a compiled playbook (issue #241).

Structure tests (no browser): the five tabs, the Start-here install text, that
the page loads nothing from the network, the embedded machine blocks, and the
Review list (``review_rows``). The page's behavior in a real browser (Connect
folder, the download fallback, no console errors) is ``tests/test_index_browser.py``.

Fixture provenance (the producer of every shape used here): the page is built
by ``playbook view bundle`` over the committed NDA example playbook (the real
output of ``mine`` -> ``project`` -> ``posture interview``); the synthetic
playbooks of the Review-list tests are assembled by the production
``assemble_playbook`` over ``Observation`` rows of the shapes
``observation_builder`` writes, with verdicts stored through the production
``VerdictStore`` under the production cache key.

SECURITY NOTE: synthetic text only.
"""

from __future__ import annotations

import html as html_lib
import json
import re
import shutil
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from playbook_engine.agent_judge import StoreBackedEquivalenceJudge, VerdictStore
from playbook_engine.canonicalize import content_hash
from playbook_engine.clause_position_compiler import compile_clause_positions
from playbook_engine.cli import cli
from playbook_engine.document_renderer import INDEX_FILENAME, TAB_IDS, render_index_html
from playbook_engine.equivalence import equivalence_key, summarize
from playbook_engine.observation_builder import Observation, ObservationCitation
from playbook_engine.overrides import OVERRIDES_FILENAME
from playbook_engine.playbook_assembler import assemble_playbook
from playbook_engine.review_rows import REVIEW_LABEL_ORDER, build_review_rows
from playbook_engine.toaster_install import TOASTER_INSTALL_STEPS, install_steps

_ROOT = Path(__file__).resolve().parent.parent
_NDA_PLAYBOOK = _ROOT / "examples" / "nda" / "playbook.opf.json"


# ---------------------------------------------------------------------------
# The page
# ---------------------------------------------------------------------------


@pytest.fixture
def page_dir(tmp_path: Path) -> Path:
    out = tmp_path / "out"
    out.mkdir()
    shutil.copy(_NDA_PLAYBOOK, out / "playbook.opf.json")
    return out


def _build(out: Path) -> str:
    result = CliRunner().invoke(cli, ["view", "bundle", str(out)])
    assert result.exit_code == 0, result.output
    return (out / INDEX_FILENAME).read_text(encoding="utf-8")


class _Tags(HTMLParser):
    """Every start tag with its attributes, and the text of inline scripts and styles."""

    def __init__(self) -> None:
        super().__init__()
        self.tags: list[tuple[str, dict[str, str | None]]] = []
        self.scripts: list[tuple[dict[str, str | None], str]] = []
        self._in_script: dict[str, str | None] | None = None
        self._buf: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.tags.append((tag, dict(attrs)))
        if tag == "script":
            self._in_script = dict(attrs)
            self._buf = []

    def handle_endtag(self, tag: str) -> None:
        if tag == "script" and self._in_script is not None:
            self.scripts.append((self._in_script, "".join(self._buf)))
            self._in_script = None

    def handle_data(self, data: str) -> None:
        if self._in_script is not None:
            self._buf.append(data)


def _parse(html: str) -> _Tags:
    parser = _Tags()
    parser.feed(html)
    return parser


def test_the_default_name_is_index_html_and_there_is_no_old_bundle(page_dir: Path) -> None:
    _build(page_dir)
    assert (page_dir / "index.html").is_file()
    assert not (page_dir / "playbook.opf.html").exists()


def test_the_page_has_exactly_the_five_tabs_in_order(page_dir: Path) -> None:
    html = _build(page_dir)
    assert TAB_IDS == ("start", "playbook", "evidence", "review", "posture-floor")
    parsed = _parse(html)
    tabs = [a["data-tab"] for tag, a in parsed.tags if a.get("role") == "tab"]
    panels = [a["id"] for tag, a in parsed.tags if a.get("role") == "tabpanel"]
    assert tabs == list(TAB_IDS)
    assert panels == [f"tab-{t}" for t in TAB_IDS]
    for _tag, attrs in parsed.tags:
        if attrs.get("role") == "tab":
            assert attrs["aria-controls"] == f"tab-{attrs['data-tab']}"
    assert [a["aria-selected"] for tag, a in parsed.tags if a.get("role") == "tab"] == [
        "true",
        "false",
        "false",
        "false",
        "false",
    ]


def _panel(html: str, tab: str) -> str:
    m = re.search(
        rf'<div role="tabpanel" id="tab-{tab}".*?(?=<div role="tabpanel"|<footer)', html, re.S
    )
    assert m, tab
    return m.group(0)


def test_start_here_carries_the_install_steps_and_the_playbooks_identity(page_dir: Path) -> None:
    html = _build(page_dir)
    start = _panel(html, "start")
    doc = json.loads((page_dir / "playbook.opf.json").read_text(encoding="utf-8"))
    # the numbered toaster steps, from the one constant, naming the exact file
    assert len(TOASTER_INSTALL_STEPS) in (2, 3)
    for step in TOASTER_INSTALL_STEPS:
        text = html_lib.escape(step).replace(
            "{file}", '<code class="js-opf-path">playbook.opf.json</code>'
        )
        assert text in start, step
    assert "Upload new playbook" in start and "Approve &amp; activate" in start
    assert "playbook.opf.json" in start and "index.html" in start
    # identity
    assert doc["identity"]["content_hash"] in start
    assert f"<td>{doc['opf_version']}</td>" in start
    assert doc["perspective"]["party"] in start and doc["perspective"]["counterparty_type"] in start
    # review is optional, and says so
    assert "Nothing here has to be" in start


def test_the_skill_closing_prints_the_same_steps_the_page_shows() -> None:
    result = CliRunner().invoke(cli, ["install-steps", "--file", "/Users/me/out/playbook.opf.json"])
    assert result.exit_code == 0, result.output
    assert result.output.splitlines() == install_steps("/Users/me/out/playbook.opf.json")
    for line, step in zip(result.output.splitlines(), TOASTER_INSTALL_STEPS, strict=True):
        assert step.replace("{file}", "/Users/me/out/playbook.opf.json") in line
    assert "Playbooks" in result.output and "Approve & activate" in result.output
    bare = CliRunner().invoke(cli, ["install-steps"])
    assert "playbook.opf.json" in bare.output


_NETWORK_APIS = (
    "fetch(",
    "XMLHttpRequest",
    "WebSocket",
    "EventSource",
    "sendBeacon",
    "importScripts",
    "import(",
    "navigator.serviceWorker",
)


def test_the_page_loads_no_external_resources(page_dir: Path) -> None:
    html = _build(page_dir)
    parsed = _parse(html)
    for tag, attrs in parsed.tags:
        if tag == "link":
            # the one link is an empty data: icon, which stops a browser asking a
            # server for /favicon.ico and fetches nothing
            assert attrs == {"rel": "icon", "href": "data:,"}, attrs
            continue
        assert tag not in {
            "img",
            "iframe",
            "embed",
            "object",
            "audio",
            "video",
            "source",
            "base",
        }, tag
        for name in ("src", "href", "action", "srcset", "poster", "data"):
            value = attrs.get(name)
            if value is None:
                continue
            # in-page anchors only; nothing that leaves the file
            assert value.startswith("#") or tag == "a" and value.startswith("blob:"), (
                tag,
                name,
                value,
            )
    # no stylesheet imports or url() fetches
    for tag, attrs in parsed.tags:
        assert not (tag == "script" and attrs.get("src"))
    style = re.search(r"<style>(.*?)</style>", html, re.S)
    assert style and "@import" not in style.group(1) and "url(" not in style.group(1)
    code = [text for attrs, text in parsed.scripts if attrs.get("type") != "application/json"]
    assert len(code) == 1, "one inline program"
    for api in _NETWORK_APIS:
        assert api not in code[0], f"the page program uses {api}"
    assert not re.search(r"https?://", code[0])


def test_the_machine_blocks_are_embedded_and_the_canonical_one_verifies(page_dir: Path) -> None:
    html = _build(page_dir)
    parsed = _parse(html)
    blocks = {
        attrs["id"]: text
        for attrs, text in parsed.scripts
        if attrs.get("type") == "application/json"
    }
    assert {"opf-canonical", "opf-digest", "page-meta", "review-rows", "pf-data"} <= set(blocks)
    on_disk = json.loads((page_dir / "playbook.opf.json").read_text(encoding="utf-8"))
    canonical = json.loads(blocks["opf-canonical"])
    assert canonical == on_disk
    assert canonical["identity"]["content_hash"] == content_hash(canonical)
    assert json.loads(blocks["opf-digest"]) == on_disk["digest"]
    assert json.loads(blocks["page-meta"])["content_hash"] == on_disk["identity"]["content_hash"]
    assert json.loads(blocks["review-rows"]) == build_review_rows(on_disk).rows
    assert "overrides-initial" not in blocks


def test_a_wellformed_overrides_file_is_embedded_and_a_broken_one_is_not(page_dir: Path) -> None:
    saved = {"overrides_version": 1, "overrides": []}
    (page_dir / OVERRIDES_FILENAME).write_text(json.dumps(saved), encoding="utf-8")
    parsed = _parse(_build(page_dir))
    blocks = {attrs.get("id"): text for attrs, text in parsed.scripts}
    assert json.loads(blocks["overrides-initial"]) == saved
    (page_dir / OVERRIDES_FILENAME).write_text("{broken", encoding="utf-8")
    parsed = _parse(_build(page_dir))
    assert "overrides-initial" not in {attrs.get("id") for attrs, _ in parsed.scripts}


def test_a_hostile_script_closer_in_the_playbook_cannot_break_out(page_dir: Path) -> None:
    pb_path = page_dir / "playbook.opf.json"
    pb = json.loads(pb_path.read_text(encoding="utf-8"))
    pb["agreement_type"]["name"] = 'x</script><script>alert("pwned")</script><!-- y'
    pb_path.write_text(json.dumps(pb, indent=2, ensure_ascii=False), encoding="utf-8")
    html = _build(page_dir)
    parsed = _parse(html)
    programs = [text for attrs, text in parsed.scripts if attrs.get("type") != "application/json"]
    assert len(programs) == 1 and "pwned" not in programs[0]
    blocks = {
        attrs["id"]: text
        for attrs, text in parsed.scripts
        if attrs.get("type") == "application/json"
    }
    assert (
        json.loads(blocks["opf-canonical"])["agreement_type"]["name"]
        == pb["agreement_type"]["name"]
    )
    assert "<script>alert" not in html


def test_the_tabs_say_what_is_optional_and_where_edits_go(page_dir: Path) -> None:
    html = _build(page_dir)
    review = _panel(html, "review")
    flat = " ".join(review.split())
    assert "Review (optional)" in flat and "Nothing waits on this list" in flat
    for control in (
        "btn-connect",
        "btn-download",
        "btn-reconnect",
        "save-status",
        "review-progress",
    ):
        assert f'id="{control}"' in review
    assert OVERRIDES_FILENAME in review
    floor = _panel(html, "posture-floor")
    assert 'data-target="posture"' in floor and 'data-target="floor"' in floor
    # the interview-promoted invariants take a statement edit; no invariant of the example
    # has authored rationale text, and the signed one is not editable at all
    assert 'data-field="statement"' in floor
    assert 'data-field="rationale"' not in floor
    assert floor.count("Not editable here") == 4


def _method_panel_text(html: str) -> str:
    m = re.search(
        r'<section class="clause" id="method">.*?</section>', _panel(html, "playbook"), re.S
    )
    assert m, "the Playbook tab has a Method & provenance panel"
    return " ".join(re.sub(r"<[^>]+>", "", m.group(0)).split())


def test_the_method_panel_states_drafting_origin_and_counts_an_ambiguous_one_undetermined(
    page_dir: Path,
) -> None:
    """The Playbook tab keeps the bundle's "Drafting origin" line, with issue #225's rule:
    an ambiguous provenance detection is not a determined side, so it is counted as
    undetermined, never as either paper."""
    doc = json.loads((page_dir / "playbook.opf.json").read_text(encoding="utf-8"))
    docs = doc["corpus"]["documents"]
    assert sorted(d["provenance"] for d in docs) == ["counterparty_paper"] * 2 + ["our_paper"] * 4
    assert not any(d.get("provenance_is_ambiguous") for d in docs), "premise"
    flat = _method_panel_text(_build(page_dir))
    assert "Drafting origin: 4 agreements on our paper, 2 on counterparty paper — judged" in flat
    assert "undetermined" not in flat
    assert f"Judged evidence: {len(doc['evidence']['precedent'])} observed clause positions" in flat

    next(d for d in docs if d["provenance"] == "our_paper")["provenance_is_ambiguous"] = True
    flat = _method_panel_text(render_index_html_from(doc, page_dir.parent))
    assert (
        "Drafting origin: 3 agreements on our paper, 2 on counterparty paper, 1 undetermined"
        in flat
    )


def test_render_index_html_writes_atomically_and_refuses_a_missing_playbook(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        render_index_html(tmp_path)
    result = CliRunner().invoke(cli, ["view", "bundle", str(tmp_path)])
    assert result.exit_code == 1 and "playbook.opf.json not found" in result.output


# ---------------------------------------------------------------------------
# The Review list
# ---------------------------------------------------------------------------

_PARTY = "Acme Corp"
_STANDARD = "Each party shall keep the other party's information confidential for three years."
_TYPE_ID = "test-agreement"


def _obs(doc_id: str, text: str, *, outcome: str = "signed", standard: bool = False) -> Observation:
    return Observation(
        observation_id=f"{doc_id}/v2/8",
        taxonomy_id="survival_period",
        text_summary=text,
        full_text=text,
        citation=ObservationCitation(
            document_id=doc_id, version="v2", clause_path="8", char_span=None
        ),
        deviation="none" if standard else "substantive",
        risk_delta={"direction": "neutral", "magnitude": "none"},
        provenance="our_paper",
        outcome=outcome,
        basis="deterministic",
        standard=standard,
        opened_with="standard" if standard else "non_standard",
    )


def _doc(doc_id: str) -> dict[str, Any]:
    return {
        "document_id": doc_id,
        "provenance": "our_paper",
        "in_scope": True,
        "versions": 2,
        "signed_version": 2,
        "version_order_basis": "edit_distance_chain",
    }


_TAXONOMY = {
    "source": "test",
    "entries": [
        {
            "id": "survival_period",
            "label": "Survival Period",
            "status": "active",
            "cuad_origin": "x",
            "description": "How long the duty lasts.",
        }
    ],
}


def _playbook(
    tmp_path: Path, variants: list[tuple[str, str | None]], *, copies: dict[str, int] | None = None
) -> dict[str, Any]:
    """One clause, one deal per variant text, the text's label stored under its real key.

    *variants*: ``(text, label or None)``; *copies*: extra deals signing a text again.
    """
    store = VerdictStore(tmp_path / "judge" / "verdicts.jsonl")
    observations = [_obs("deal_std", _STANDARD, standard=True)]
    docs = [_doc("deal_std")]
    n = 0
    for text, label in variants:
        for _ in range(1 + (copies or {}).get(text, 0)):
            n += 1
            observations.append(_obs(f"deal_{n:02d}", text))
            docs.append(_doc(f"deal_{n:02d}"))
        if label is not None:
            store.put_by_key(
                equivalence_key(_TYPE_ID, "survival_period", _PARTY, text, _STANDARD),
                {"label": label, "reason": f"Because {label}.", "basis": "agent"},
            )
    template = [_obs("template", _STANDARD, standard=True)]
    positions, _, _ = compile_clause_positions(observations, template)
    return assemble_playbook(
        agreement_type={"id": _TYPE_ID, "name": "Test Agreement"},
        baseline={"has_canonical_template": True},
        taxonomy=_TAXONOMY,
        clause_positions=positions,
        corpus_documents=docs,
        generated_at="2026-10-09T00:00:00Z",
        observations=observations,
        perspective={"party": _PARTY, "counterparty_type": "vendor"},
        equivalence_judge=StoreBackedEquivalenceJudge(store=store),
    )


def _variant(i: int) -> str:
    words = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india"]
    return f"Each party shall keep the other party's information confidential, subject to {words[i]} terms."


def test_rows_are_ordered_less_protective_then_different_concept_then_more_then_equivalent(
    tmp_path: Path,
) -> None:
    labels = ["equivalent", "more_protective", "different_concept", "less_protective"]
    playbook = _playbook(tmp_path, [(_variant(i), label) for i, label in enumerate(labels)])
    rows = build_review_rows(playbook)
    assert [r["label"] for r in rows.rows] == list(REVIEW_LABEL_ORDER)
    assert REVIEW_LABEL_ORDER == (
        "less_protective",
        "different_concept",
        "more_protective",
        "equivalent",
    )
    assert rows.by_label() == dict.fromkeys(labels, 1)
    assert rows.n_not_shown == 0 and rows.n_unjudged == 0
    for row in rows.rows:
        assert re.fullmatch(r"[0-9a-f]{64}", row["key"])
        assert row["reason"] == f"Because {row['label']}."
        assert row["title"] == "Survival Period" and row["roles"] == ["signed"]


def test_the_list_is_cut_to_what_the_digest_shows_and_counts_the_rest(tmp_path: Path) -> None:
    """Eight judged variants, a digest that keeps five entries per list: the equivalent
    variants collapse into ONE entry which is the sixth, so the cap drops it."""
    labels = [
        "less_protective",
        "less_protective",
        "different_concept",
        "different_concept",
        "more_protective",
        "equivalent",
        "equivalent",
        "equivalent",
    ]
    playbook = _playbook(tmp_path, [(_variant(i), label) for i, label in enumerate(labels)])
    digest_entry_labels = [e["label"] for e in playbook["digest"]["clauses"][0]["signed_variants"]]
    assert len(digest_entry_labels) == 5 and "equivalent" not in digest_entry_labels
    rows = build_review_rows(playbook)
    assert rows.by_label() == {
        "less_protective": 2,
        "different_concept": 2,
        "more_protective": 1,
        "equivalent": 0,
    }
    assert rows.n_not_shown == 3


def test_equivalent_variants_the_digest_collapses_are_listed_when_it_shows_them(
    tmp_path: Path,
) -> None:
    labels = ["less_protective", "equivalent", "equivalent", "equivalent"]
    playbook = _playbook(tmp_path, [(_variant(i), label) for i, label in enumerate(labels)])
    collapsed = [e for e in playbook["digest"]["clauses"][0]["signed_variants"] if e.get("n_texts")]
    assert len(collapsed) == 1 and collapsed[0]["n_texts"] == 3
    rows = build_review_rows(playbook)
    assert rows.by_label()["equivalent"] == 3 and rows.n_not_shown == 0


def test_a_text_signed_in_many_deals_is_one_row_with_its_deal_count(tmp_path: Path) -> None:
    playbook = _playbook(
        tmp_path,
        [(_variant(0), "less_protective"), (_variant(1), "less_protective")],
        copies={_variant(1): 2},
    )
    rows = build_review_rows(playbook).rows
    assert len(rows) == 2
    assert [r["n_deals"] for r in rows] == [3, 1], "the text most deals signed comes first"


def test_an_unjudged_text_is_counted_not_listed(tmp_path: Path) -> None:
    playbook = _playbook(tmp_path, [(_variant(0), "less_protective"), (_variant(1), None)])
    rows = build_review_rows(playbook)
    assert len(rows.rows) == 1 and rows.n_unjudged == 1
    html = render_index_html_from(playbook, tmp_path)
    assert "1 text(s) in the digest have no label yet" in html


def test_a_judged_text_no_digest_group_holds_is_counted_as_not_shown(tmp_path: Path) -> None:
    """The last draft of a deal that never signed is judged but belongs to no digest group:
    it is no row, and ``n_not_shown`` still counts it (the count matches the playbook's judged texts)."""
    store = VerdictStore(tmp_path / "judge" / "verdicts.jsonl")
    for i in (0, 1):
        store.put_by_key(
            equivalence_key(_TYPE_ID, "survival_period", _PARTY, _variant(i), _STANDARD),
            {"label": "less_protective", "reason": "Shorter.", "basis": "agent"},
        )
    observations = [
        _obs("deal_std", _STANDARD, standard=True),
        _obs("deal_a", _variant(0)),
        _obs("deal_u", _variant(1)),
    ]
    docs = [_doc("deal_std"), _doc("deal_a"), {**_doc("deal_u"), "signed_version": None}]
    positions, _, _ = compile_clause_positions(
        observations, [_obs("template", _STANDARD, standard=True)]
    )
    playbook = assemble_playbook(
        agreement_type={"id": _TYPE_ID, "name": "Test Agreement"},
        baseline={"has_canonical_template": True},
        taxonomy=_TAXONOMY,
        clause_positions=positions,
        corpus_documents=docs,
        generated_at="2026-10-09T00:00:00Z",
        observations=observations,
        perspective={"party": _PARTY, "counterparty_type": "vendor"},
        equivalence_judge=StoreBackedEquivalenceJudge(store=store),
    )
    unsigned = next(p for p in playbook["evidence"]["precedent"] if p["document_id"] == "deal_u")
    assert unsigned["signed"] is False and unsigned["signed_text"]["vs_standard"] is not None
    rows = build_review_rows(playbook)
    assert len(rows.rows) == 1 and rows.n_not_shown == 1
    totals = summarize(playbook["evidence"], _TYPE_ID, _PARTY)["totals"]
    assert len(rows.rows) + rows.n_not_shown == totals["drafted"]


def test_a_playbook_with_no_standard_has_no_rows(tmp_path: Path) -> None:
    doc = json.loads(_NDA_PLAYBOOK.read_text(encoding="utf-8"))
    for clause in doc["evidence"]["clauses"]:
        clause["our_standard"] = None
    assert build_review_rows(doc).rows == []


def test_every_row_of_the_nda_example_resolves_to_a_text_in_the_page(page_dir: Path) -> None:
    """The locator the page uses to show full text points at a real record and role."""
    doc = json.loads((page_dir / "playbook.opf.json").read_text(encoding="utf-8"))
    by_id = {p["id"]: p for p in doc["evidence"]["precedent"]}
    rows = build_review_rows(doc).rows
    assert rows
    for row in rows:
        record = by_id[row["loc"]["record"]]
        role = row["loc"]["role"]
        holder = (
            record["signed_text"]
            if role == "signed"
            else record["opening_text"]
            if role == "opening"
            else record["refused_asks"][row["loc"]["i"]]
        )
        assert holder["text"].startswith(row["text"][:20].rstrip("…"))
        assert holder["vs_standard"]["label"] == row["label"]


def render_index_html_from(playbook: dict[str, Any], tmp_path: Path) -> str:
    out = tmp_path / "page"
    out.mkdir(exist_ok=True)
    (out / "playbook.opf.json").write_text(json.dumps(playbook, indent=2), encoding="utf-8")
    return render_index_html(out)
