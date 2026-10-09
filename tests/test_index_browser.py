"""``index.html`` in a real headless browser (issue #241).

The repo carries no Playwright, so this drives a locally installed Chrome,
Chromium or Edge over the DevTools pipe (``tests/chrome_cdp.py``) and SKIPS
when none is installed. What it proves:

- the page opened from a ``file://`` URL renders all five tabs with no console
  errors, in a secure context where the File System Access API exists;
- **Connect folder** + an edit writes ``overrides.json`` into the granted
  folder, saved on each change and restored after a reload;
- **Download edits** (the fallback where the API is missing: Safari, Firefox)
  produces the SAME file;
- ``playbook apply-overrides`` accepts the file the page wrote;
- the page remembers its folder per out-dir and only ever reads or writes
  ``overrides.json`` in a folder holding its OWN playbook: a second out-dir of
  the same agreement type, opened from the same origin, never touches the first.

The browser's native directory picker cannot be driven headless, so the Connect
tests replace ``window.showDirectoryPicker`` with a function returning the
browser's own origin-private file system root, or a subfolder of it (a real
``FileSystemDirectoryHandle``, real ``createWritable`` writes), seeded with the
out-dir's ``playbook.opf.json``. That root is unavailable to ``file://`` pages,
so these tests serve the same pages from ``http://127.0.0.1`` (also a secure
context; several out-dirs under one server share one storage origin exactly as
every ``file://`` page does in Chrome); the picker-less download test and the
load test run on ``file://``.

Fixture provenance: the page is built by ``playbook view bundle`` over the
committed NDA example playbook; the edits are made by clicking the page's own
buttons; the saved file is read back and applied by the production
``playbook apply-overrides``.
"""

from __future__ import annotations

import contextlib
import functools
import http.server
import json
import shutil
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from playbook_engine.cli import cli
from playbook_engine.document_renderer import TAB_IDS
from playbook_engine.overrides import OVERRIDES_FILENAME
from playbook_engine.review_rows import build_review_rows
from tests.chrome_cdp import Browser, find_browser

_ROOT = Path(__file__).resolve().parent.parent
_NDA_PLAYBOOK = _ROOT / "examples" / "nda" / "playbook.opf.json"
_NDA_SIDECAR = _ROOT / "examples" / "nda" / "precedent.jsonl"

_EXE = find_browser()
pytestmark = pytest.mark.skipif(_EXE is None, reason="no Chrome/Chromium/Edge installed")


@pytest.fixture
def out_dir(tmp_path: Path) -> Path:
    out = tmp_path / "out"
    out.mkdir()
    shutil.copy(_NDA_PLAYBOOK, out / "playbook.opf.json")
    shutil.copy(_NDA_SIDECAR, out / "precedent.jsonl")
    result = CliRunner().invoke(cli, ["view", "bundle", str(out)])
    assert result.exit_code == 0, result.output
    return out


@contextlib.contextmanager
def _serve(directory: Path) -> Iterator[str]:
    """Serve *directory* on ``http://127.0.0.1:<port>``; yield that base URL."""
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(directory))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        thread.join(timeout=5)


@pytest.fixture
def served(out_dir: Path) -> Iterator[str]:
    with _serve(out_dir) as base:
        yield f"{base}/index.html"


def _doc(out_dir: Path) -> dict[str, Any]:
    return json.loads((out_dir / "playbook.opf.json").read_text(encoding="utf-8"))


def _click(browser: Browser, selector: str) -> None:
    assert browser.eval(
        f"(() => {{ const el = document.querySelector({json.dumps(selector)}); "
        "if (!el) return false; el.click(); return true; })()"
    ), f"no element {selector}"


def _confirm_first_and_change_second(browser: Browser) -> None:
    """Confirm the first Review row; change the label of the second to something else."""
    browser.eval("document.querySelector('[data-tab=review]').click()")
    _click(browser, "article.rv:nth-of-type(1) .acts button")
    browser.eval(
        "(() => { const sel = document.querySelector('article.rv:nth-of-type(2) select');"
        "sel.value = sel.options[1].value; sel.dispatchEvent(new Event('change')); })()"
    )
    browser.wait_for("Object.keys(window.__indexPage.edits()).length === 2")


def _expected_pair(doc: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], str]:
    rows = build_review_rows(doc).rows
    first, second = rows[0], rows[1]
    changed_to = next(
        label
        for label in ("less_protective", "different_concept", "more_protective", "equivalent")
        if label != second["label"]
    )
    return first, second, changed_to


def _applies_cleanly(out_dir: Path, overrides: dict[str, Any]) -> dict[str, Any]:
    """Run the production ``apply-overrides`` on *overrides* in a copy of the page's out-dir."""
    copy = out_dir.parent / "apply-copy"
    shutil.copytree(out_dir, copy, dirs_exist_ok=True)
    (copy / OVERRIDES_FILENAME).write_text(json.dumps(overrides), encoding="utf-8")
    result = CliRunner().invoke(cli, ["apply-overrides", str(copy)])
    assert result.exit_code == 0, result.output
    assert CliRunner().invoke(cli, ["validate", str(copy / "playbook.opf.json")]).exit_code == 0
    return json.loads((copy / "playbook.opf.json").read_text(encoding="utf-8"))


def test_the_page_renders_every_tab_on_file_url_with_no_console_errors(out_dir: Path) -> None:
    assert _EXE is not None
    with Browser(_EXE) as browser:
        browser.goto((out_dir / "index.html").as_uri())
        facts = browser.eval(
            "JSON.stringify({secure: window.isSecureContext, picker: typeof window.showDirectoryPicker,"
            " tabs: [...document.querySelectorAll('[role=tab]')].map(t => t.dataset.tab),"
            " paint: (performance.getEntriesByName('first-contentful-paint')[0] || {}).startTime})"
        )
        info = json.loads(facts)
        assert info["secure"] is True, "a file:// page is a secure context in current Chrome"
        assert info["picker"] == "function", "the File System Access API is present"
        assert info["tabs"] == list(TAB_IDS)
        assert info["paint"] is not None and info["paint"] < 5000
        for tab in TAB_IDS:
            _click(browser, f"[data-tab={tab}]")
            state = json.loads(
                browser.eval(
                    f"JSON.stringify((() => {{ const p = document.getElementById('tab-{tab}');"
                    "const shown = [...document.querySelectorAll('[role=tabpanel]')]"
                    ".filter(x => !x.hidden).map(x => x.id);"
                    "return {shown, text: p.innerText.trim().length}; })())"
                )
            )
            assert state["shown"] == [f"tab-{tab}"], (tab, state)
            assert state["text"] > 100, f"tab {tab} rendered no content"
        rows = browser.eval("document.querySelectorAll('article.rv').length")
        assert rows == min(25, len(build_review_rows(_doc(out_dir)).rows))
        assert browser.problems() == []


def test_the_download_fallback_writes_the_overrides_file(out_dir: Path, tmp_path: Path) -> None:
    assert _EXE is not None
    downloads = tmp_path / "downloads"
    downloads.mkdir()
    doc = _doc(out_dir)
    first, second, changed_to = _expected_pair(doc)
    with Browser(_EXE, download_dir=downloads) as browser:
        # Safari and Firefox have no showDirectoryPicker: remove it before the page runs.
        browser.add_script_on_load("delete window.showDirectoryPicker;")
        browser.goto((out_dir / "index.html").as_uri())
        assert browser.eval("typeof window.showDirectoryPicker") == "undefined"
        browser.eval("document.querySelector('[data-tab=review]').click()")
        assert browser.eval("document.getElementById('btn-connect').hidden") is True
        assert browser.eval("document.getElementById('save-note').hidden") is False
        _confirm_first_and_change_second(browser)
        assert "not saved yet" in browser.eval("document.getElementById('save-status').textContent")
        _click(browser, "#btn-download")
        saved = _wait_download(downloads)
        assert browser.problems() == []
    _check_saved(saved, doc, first, second, changed_to)
    after = _applies_cleanly(out_dir, saved)
    assert after["identity"]["content_hash"] != doc["identity"]["content_hash"]


def _check_saved(
    saved: dict[str, Any],
    doc: dict[str, Any],
    first: dict[str, Any],
    second: dict[str, Any],
    changed_to: str,
) -> None:
    assert saved["overrides_version"] == 1
    assert saved["based_on"] == doc["identity"]["content_hash"]
    by_id = {e["id"]: e for e in saved["overrides"]}
    assert set(by_id) == {first["key"], second["key"]}
    assert by_id[first["key"]]["value"] == {"label": first["label"]}, "Confirm keeps the label"
    assert by_id[second["key"]]["value"] == {"label": changed_to}
    for entry in saved["overrides"]:
        assert entry["target"] == "vs_standard" and entry["basis"] == "owner"


# Real Chrome refuses showDirectoryPicker() and requestPermission() outside a user
# gesture (SecurityError). The stand-ins do too, and they count EVERY call before that
# check runs: a page that called either on load is counted even though its call is then
# rejected (and its rejection swallowed into a status message), so the on-load
# ``== 0`` assertions fail instead of passing unnoticed. They are installed before the
# page's own scripts run (``add_script_on_load``), in every document.
_GESTURE_GUARD = """
window.__gestureGuard = (name) => {
  if (!navigator.userActivation.isActive) {
    throw new DOMException(name + ' requires a user gesture', 'SecurityError');
  }
};
"""

# The picker returns the origin-private root, or the subfolder named by
# ``window.__pickPath`` ("a/out"): the folder the person would have chosen.
_PICKER_STUB = (
    _GESTURE_GUARD
    + """
window.__activeAtLoad = navigator.userActivation.isActive;
window.__pickerCalls = 0;
window.showDirectoryPicker = async (opts) => {
  window.__pickerCalls += 1;
  window.__gestureGuard('showDirectoryPicker');
  window.__pickerOpts = opts;
  let dir = await navigator.storage.getDirectory();
  for (const part of (window.__pickPath || '').split('/').filter(Boolean)) {
    dir = await dir.getDirectoryHandle(part);
  }
  return dir;
};
"""
)

# The folder must hold the page's own playbook.opf.json (the page refuses any other):
# seed it with the bytes the test server serves next to index.html.
_SEED = """
(async () => {
  const root = await navigator.storage.getDirectory();
  for await (const name of root.keys()) { await root.removeEntry(name, {recursive: true}); }
  const text = await (await fetch('playbook.opf.json')).text();
  const seed = await root.getFileHandle('playbook.opf.json', {create: true});
  const w = await seed.createWritable(); await w.write(text); await w.close();
  return true;
})()
"""

_EMPTY_ROOT = """
(async () => {
  const root = await navigator.storage.getDirectory();
  for await (const n of root.keys()) { await root.removeEntry(n, {recursive: true}); }
  return true;
})()
"""

_READ_SAVED = """
(async () => {
  const root = await navigator.storage.getDirectory();
  try { return await (await (await root.getFileHandle('overrides.json')).getFile()).text(); }
  catch (e) { return null; }
})()
"""

_SAVED_COUNT = """
(async () => {
  const root = await navigator.storage.getDirectory();
  try {
    const text = await (await (await root.getFileHandle('overrides.json')).getFile()).text();
    return JSON.parse(text).overrides.length === %d;
  } catch (e) { return false; }
})()
"""


def _wait_saved(browser: Browser, n: int) -> dict[str, Any]:
    """Wait until overrides.json in the granted folder holds *n* entries; return it."""
    browser.wait_for(_SAVED_COUNT % n)
    return json.loads(browser.eval(_READ_SAVED))


def _wait_download(downloads: Path) -> dict[str, Any]:
    deadline = time.monotonic() + 10
    path = downloads / OVERRIDES_FILENAME
    while time.monotonic() < deadline:
        # Chrome writes <name>.crdownload first and renames when complete
        if path.is_file():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except ValueError:
                pass
        time.sleep(0.05)
    raise AssertionError(f"no {OVERRIDES_FILENAME} was downloaded")


def test_connect_folder_saves_each_edit_to_overrides_json_and_restores_after_reload(
    out_dir: Path, served: str, tmp_path: Path
) -> None:
    assert _EXE is not None
    doc = _doc(out_dir)
    first, second, changed_to = _expected_pair(doc)
    downloads = tmp_path / "downloads"
    downloads.mkdir()
    with Browser(_EXE, download_dir=downloads) as browser:
        browser.add_script_on_load(_PICKER_STUB)
        browser.goto(served)
        assert browser.eval("window.isSecureContext") is True
        assert browser.eval("window.__activeAtLoad") is False
        assert browser.eval("window.__pickerCalls") == 0, "the picker was called on load"
        assert browser.eval(_SEED) is True
        browser.eval("document.querySelector('[data-tab=review]').click()")
        assert browser.eval("document.getElementById('btn-connect').hidden") is False
        _click(browser, "#btn-connect")
        browser.wait_for("document.getElementById('save-status').textContent.includes('saved to')")
        assert browser.eval("window.__pickerOpts.mode") == "readwrite"
        assert browser.eval("window.__pickerCalls") == 1

        # an edit is written to overrides.json in the folder as it is made
        _click(browser, "article.rv:nth-of-type(1) .acts button")
        one = _wait_saved(browser, 1)
        assert [e["id"] for e in one["overrides"]] == [first["key"]]
        browser.wait_for(
            "document.getElementById('save-status').textContent.includes('1 edit saved to overrides.json')"
        )

        _confirm_first_and_change_second(browser)
        connected = _wait_saved(browser, 2)
        _check_saved(connected, doc, first, second, changed_to)

        # the download button produces the same file
        _click(browser, "#btn-download")
        downloaded = _wait_download(downloads)
        assert browser.problems() == []
        assert downloaded == connected

        # reload: the remembered folder reconnects and the saved edits come back
        browser.goto(served)
        browser.wait_for("Object.keys(window.__indexPage.edits()).length === 2")
        assert browser.eval("window.__pickerCalls") == 0, "the picker was called on reload"
        restored = browser.eval(
            "JSON.stringify({edits: Object.keys(window.__indexPage.edits()).length,"
            " status: document.getElementById('save-status').textContent,"
            " progress: document.getElementById('review-progress').textContent})"
        )
        state = json.loads(restored)
        assert state["edits"] == 2, state
        assert "saved to overrides.json" in state["status"], state
        assert state["progress"].startswith("2 of "), state
        assert browser.problems() == []

    after = _applies_cleanly(out_dir, connected)
    assert after["identity"]["content_hash"] != doc["identity"]["content_hash"]


def test_a_folder_without_the_playbook_is_refused_and_nothing_is_written(
    out_dir: Path, served: str
) -> None:
    assert _EXE is not None
    with Browser(_EXE) as browser:
        browser.add_script_on_load(_PICKER_STUB)
        browser.goto(served)
        assert browser.eval(_EMPTY_ROOT) is True
        browser.eval("document.querySelector('[data-tab=review]').click()")
        _click(browser, "#btn-connect")
        browser.wait_for(
            "document.getElementById('save-status').textContent.includes('has no playbook.opf.json')"
        )
        _click(browser, "article.rv:nth-of-type(1) .acts button")
        browser.wait_for("Object.keys(window.__indexPage.edits()).length === 1")
        browser.pump(0.3)
        assert browser.eval(_READ_SAVED) is None


def test_a_remembered_folder_asks_permission_again_after_a_reload(
    out_dir: Path, served: str
) -> None:
    """Chrome forgets a folder grant between visits (``queryPermission`` says ``prompt``):
    the page then offers a Reconnect button, and its click re-requests permission."""
    assert _EXE is not None
    doc = _doc(out_dir)
    first, _second, _changed = _expected_pair(doc)
    with Browser(_EXE) as browser:
        browser.add_script_on_load(_PICKER_STUB)
        browser.goto(served)
        assert browser.eval(_SEED) is True
        browser.eval("document.querySelector('[data-tab=review]').click()")
        _click(browser, "#btn-connect")
        browser.wait_for("document.getElementById('save-status').textContent.includes('saved to')")
        _click(browser, "article.rv:nth-of-type(1) .acts button")
        _wait_saved(browser, 1)

        browser.add_script_on_load(
            "FileSystemDirectoryHandle.prototype.queryPermission = async () => 'prompt';"
            "FileSystemDirectoryHandle.prototype.requestPermission = async () => {"
            "  window.__permissionRequests += 1; window.__gestureGuard('requestPermission');"
            "  return 'granted'; };"
            "window.__permissionRequests = 0;"
        )
        browser.goto(served)
        browser.wait_for("!document.getElementById('btn-reconnect').hidden")
        assert browser.eval("window.__permissionRequests") == 0, "permission asked for on load"
        assert browser.eval("window.__pickerCalls") == 0
        browser.eval("document.querySelector('[data-tab=review]').click()")
        assert browser.eval("document.getElementById('btn-reconnect').hidden") is False
        assert browser.eval("document.getElementById('btn-connect').hidden") is False
        assert browser.eval("Object.keys(window.__indexPage.edits()).length") == 0
        _click(browser, "#btn-reconnect")
        browser.wait_for("document.getElementById('btn-reconnect').hidden")
        assert browser.eval("window.__permissionRequests") == 1
        page_file = json.loads(browser.eval("JSON.stringify(window.__indexPage.fileObject())"))
        assert page_file["overrides"], "the saved edit was loaded into the page on reconnect"
        saved = json.loads(browser.eval(_READ_SAVED))
        assert [e["id"] for e in saved["overrides"]] == [first["key"]]
        assert browser.problems() == []


# Every getFileHandle() the page makes, with the folder it was made on (its path under the
# origin-private root, e.g. "a/out"). The test's own reads and seeding use the original.
_FS_LOG = """
window.__fsLog = [];
window.__rawGetFileHandle = FileSystemDirectoryHandle.prototype.getFileHandle;
FileSystemDirectoryHandle.prototype.getFileHandle = async function (name, opts) {
  const root = await navigator.storage.getDirectory();
  const path = await root.resolve(this);
  window.__fsLog.push({dir: path ? path.join('/') : null, name: name, create: !!(opts && opts.create)});
  return window.__rawGetFileHandle.call(this, name, opts);
};
"""

_SEED_OUT_DIRS = """
(async () => {
  const root = await navigator.storage.getDirectory();
  for await (const n of root.keys()) { await root.removeEntry(n, {recursive: true}); }
  for (const side of %s) {
    const top = await root.getDirectoryHandle(side, {create: true});
    const dir = await top.getDirectoryHandle('out', {create: true});
    const text = await (await fetch('/' + side + '/out/playbook.opf.json')).text();
    const fh = await window.__rawGetFileHandle.call(dir, 'playbook.opf.json', {create: true});
    const w = await fh.createWritable(); await w.write(text); await w.close();
  }
  return true;
})()
"""

_COPY_PLAYBOOK = """
(async () => {
  const root = await navigator.storage.getDirectory();
  const src = await (await root.getDirectoryHandle(%s)).getDirectoryHandle('out');
  const dst = await (await root.getDirectoryHandle(%s)).getDirectoryHandle('out');
  const text = await (await (await window.__rawGetFileHandle.call(src, 'playbook.opf.json')).getFile()).text();
  const fh = await window.__rawGetFileHandle.call(dst, 'playbook.opf.json', {create: true});
  const w = await fh.createWritable(); await w.write(text); await w.close();
  return true;
})()
"""

_READ_OVERRIDES_IN = """
(async () => {
  const root = await navigator.storage.getDirectory();
  const dir = await (await root.getDirectoryHandle(%s)).getDirectoryHandle('out');
  try { return await (await (await window.__rawGetFileHandle.call(dir, 'overrides.json')).getFile()).text(); }
  catch (e) { return null; }
})()
"""

_ONE_EDIT_SAVED = "document.getElementById('save-status').textContent.includes('1 edit saved')"

_PAGE_STATE = (
    "JSON.stringify({edits: Object.values(window.__indexPage.edits()).map(e => e.id),"
    " log: window.__fsLog, picker: window.__pickerCalls,"
    " connect: document.getElementById('btn-connect').hidden,"
    " reconnect: document.getElementById('btn-reconnect').hidden,"
    " status: document.getElementById('save-status').textContent})"
)


def _build_out_dir(path: Path, *, posture: str | None = None) -> dict[str, Any]:
    """An out-dir holding the NDA example and its page; with *posture*, a different build
    of the same playbook (same agreement type and perspective, another content hash): the
    owner's Posture edit folded in by the production ``apply-overrides``."""
    path.mkdir(parents=True)
    shutil.copy(_NDA_PLAYBOOK, path / "playbook.opf.json")
    shutil.copy(_NDA_SIDECAR, path / "precedent.jsonl")
    if posture is not None:
        entry = {"target": "posture", "id": "system_prompt", "value": posture, "basis": "owner"}
        (path / OVERRIDES_FILENAME).write_text(
            json.dumps({"overrides_version": 1, "overrides": [entry]}), encoding="utf-8"
        )
        applied = CliRunner().invoke(cli, ["apply-overrides", str(path)])
        assert applied.exit_code == 0, applied.output
        (path / OVERRIDES_FILENAME).unlink()
    result = CliRunner().invoke(cli, ["view", "bundle", str(path)])
    assert result.exit_code == 0, result.output
    return _doc(path)


def _state(browser: Browser) -> dict[str, Any]:
    return json.loads(browser.eval(_PAGE_STATE))


def test_a_second_out_dir_of_the_same_agreement_type_never_reads_or_writes_the_first(
    tmp_path: Path,
) -> None:
    """Chrome gives every ``file://`` page one storage origin, so a folder remembered by
    page A is visible to page B of another out-dir. Page B (same agreement type and
    perspective, another build) must not adopt A's folder on load, must not offer to
    reconnect to it, and when the person picks A's folder on Connect it only reads A's
    ``playbook.opf.json`` to see it is not its own playbook, then refuses: A's
    ``overrides.json`` is never read into B and never written by B."""
    assert _EXE is not None
    site = tmp_path / "site"
    doc_a = _build_out_dir(site / "a" / "out")
    doc_b = _build_out_dir(site / "b" / "out", posture="Out-dir B: hold the exclusions.")
    assert doc_a["agreement_type"]["id"] == doc_b["agreement_type"]["id"], "premise"
    assert doc_a["perspective"] == doc_b["perspective"], "premise"
    assert doc_a["identity"]["content_hash"] != doc_b["identity"]["content_hash"], "premise"
    rows = build_review_rows(doc_a).rows
    assert [r["key"] for r in build_review_rows(doc_b).rows[:2]] == [r["key"] for r in rows[:2]]
    first_key, second_key = rows[0]["key"], rows[1]["key"]
    only_identity_check = {"dir": "a/out", "name": "playbook.opf.json", "create": False}

    with _serve(site) as base, Browser(_EXE) as browser:
        browser.add_script_on_load(_PICKER_STUB + _FS_LOG)
        page_a, page_b = f"{base}/a/out/index.html", f"{base}/b/out/index.html"

        # page A connects its own folder and saves one edit there
        browser.goto(page_a)
        assert browser.eval(_SEED_OUT_DIRS % json.dumps(["a", "b"])) is True
        browser.eval("window.__pickPath = 'a/out'")
        browser.eval("document.querySelector('[data-tab=review]').click()")
        _click(browser, "#btn-connect")
        browser.wait_for("document.getElementById('save-status').textContent.includes('saved to')")
        _click(browser, "article.rv:nth-of-type(1) .acts button")
        browser.wait_for(_ONE_EDIT_SAVED)
        a_saved = browser.eval(_READ_OVERRIDES_IN % json.dumps("a"))
        assert [e["id"] for e in json.loads(a_saved)["overrides"]] == [first_key]

        # page B on the same origin, no click: nothing of A's folder is touched or offered
        browser.goto(page_b)
        browser.pump(1.0)
        assert _state(browser) == {
            "edits": [],
            "log": [],
            "picker": 0,
            "connect": False,
            "reconnect": True,
            "status": "No edits yet. Connect folder to save them as you go.",
        }

        # B's Connect picks A's folder: refused after reading only A's playbook.opf.json
        browser.eval("window.__pickPath = 'a/out'")
        browser.eval("document.querySelector('[data-tab=review]').click()")
        _click(browser, "#btn-connect")
        browser.wait_for(
            "document.getElementById('save-status').textContent"
            ".includes('different build of this playbook')"
        )
        refused = _state(browser)
        assert refused["edits"] == [], "A's edits were loaded into page B"
        assert refused["log"] == [only_identity_check]
        assert refused["connect"] is False and refused["reconnect"] is True
        assert "No edits were loaded from or written to that folder" in refused["status"]
        browser.pump(0.4)
        assert browser.eval(_READ_OVERRIDES_IN % json.dumps("a")) == a_saved

        # B's own edit goes to B's own folder, and A's file is still byte-identical
        _click(browser, "article.rv:nth-of-type(2) .acts button")
        browser.eval("window.__pickPath = 'b/out'")
        _click(browser, "#btn-connect")
        browser.wait_for(_ONE_EDIT_SAVED)
        b_saved = json.loads(browser.eval(_READ_OVERRIDES_IN % json.dumps("b")))
        assert [e["id"] for e in b_saved["overrides"]] == [second_key]
        assert b_saved["based_on"] == doc_b["identity"]["content_hash"]
        assert browser.eval(_READ_OVERRIDES_IN % json.dumps("a")) == a_saved
        assert [e for e in _state(browser)["log"] if e["dir"] == "a/out"] == [only_identity_check]
        assert browser.problems() == []

        # B reloaded: it reconnects to ITS folder (remembered per out-dir), never to A's
        browser.goto(page_b)
        browser.wait_for(_ONE_EDIT_SAVED)
        reloaded = _state(browser)
        assert reloaded["edits"] == [second_key]
        assert all(e["dir"] == "b/out" for e in reloaded["log"]), reloaded["log"]
        assert browser.problems() == []

        # A reloaded: it reconnects to A's folder with A's one edit, untouched by B
        browser.goto(page_a)
        browser.wait_for(_ONE_EDIT_SAVED)
        state_a = _state(browser)
        assert state_a["edits"] == [first_key]
        assert all(e["dir"] == "a/out" for e in state_a["log"]), state_a["log"]
        assert browser.eval(_READ_OVERRIDES_IN % json.dumps("a")) == a_saved

        # the remembered-folder path checks too: B's folder now holds A's playbook (the
        # playbook changed after page B was built), so B's reload refuses to use it
        assert browser.eval(_COPY_PLAYBOOK % (json.dumps("a"), json.dumps("b"))) is True
        b_before = browser.eval(_READ_OVERRIDES_IN % json.dumps("b"))
        browser.goto(page_b)
        browser.wait_for(
            "document.getElementById('save-status').textContent"
            ".includes('different build of this playbook')"
        )
        stale = _state(browser)
        assert stale["edits"] == []
        assert stale["log"] == [{"dir": "b/out", "name": "playbook.opf.json", "create": False}]
        assert stale["picker"] == 0
        browser.pump(0.4)
        assert browser.eval(_READ_OVERRIDES_IN % json.dumps("b")) == b_before
        assert browser.problems() == []


def test_confirm_all_in_a_view_records_one_entry_per_row_and_undo_clears_them(
    out_dir: Path,
) -> None:
    assert _EXE is not None
    rows = build_review_rows(_doc(out_dir)).rows
    with Browser(_EXE) as browser:
        browser.goto((out_dir / "index.html").as_uri())
        browser.eval("document.querySelector('[data-tab=review]').click()")
        assert rows, "premise: the NDA example has Review rows"
        _click(browser, "[data-filter=less_protective]")
        n_less = sum(1 for r in rows if r["label"] == "less_protective")
        _click(browser, "#bulk-confirm")
        edits = json.loads(
            browser.eval("JSON.stringify(window.__indexPage.fileObject().overrides)")
        )
        assert len(edits) == n_less
        assert {e["id"] for e in edits} == {
            r["key"] for r in rows if r["label"] == "less_protective"
        }
        assert all(e["value"] == {"label": "less_protective"} for e in edits)
        assert browser.eval("document.getElementById('bulk-confirm').disabled") is True
        _click(browser, "[data-filter=all]")
        assert (
            browser.eval("document.getElementById('review-progress').textContent")
            == f"{n_less} of {len(rows)} reviewed (optional)"
        )
        _click(browser, "#bulk-undo")
        assert browser.eval("Object.keys(window.__indexPage.edits()).length") == 0
        assert browser.problems() == []


def test_a_rows_full_text_and_standard_are_resolved_from_the_embedded_playbook(
    out_dir: Path,
) -> None:
    """The locator on a Review row points into the playbook the page embeds: opening the
    row shows the record's full text and the clause's full standard, not the summary."""
    assert _EXE is not None
    doc = _doc(out_dir)
    by_id = {p["id"]: p for p in doc["evidence"]["precedent"]}
    standards = {
        c["taxonomy_id"]: c["our_standard"]["text"]
        for c in doc["evidence"]["clauses"]
        if c.get("our_standard")
    }
    # one row per role the playbook has (signed, opening, refused): fullText() branches on it
    shown_rows = build_review_rows(doc).rows[:25]
    first_of_role: dict[str, int] = {}
    for position, candidate in enumerate(shown_rows, start=1):
        first_of_role.setdefault(candidate["loc"]["role"], position)
    assert set(first_of_role) == {"signed", "opening", "refused"}, first_of_role
    with Browser(_EXE) as browser:
        browser.goto((out_dir / "index.html").as_uri())
        browser.eval("document.querySelector('[data-tab=review]').click()")
        for i in sorted(first_of_role.values()):
            row = shown_rows[i - 1]
            browser.eval(
                f"document.querySelector('article.rv:nth-of-type({i}) details').open = true"
            )
            browser.wait_for(
                f"document.querySelector('article.rv:nth-of-type({i}) details').dataset.filled"
            )
            shown = json.loads(
                browser.eval(
                    f"JSON.stringify([...document.querySelectorAll('article.rv:nth-of-type({i}) details blockquote')]"
                    ".map(b => b.textContent))"
                )
            )
            record = by_id[row["loc"]["record"]]
            role = row["loc"]["role"]
            holder = (
                record["signed_text"]
                if role == "signed"
                else record["opening_text"]
                if role == "opening"
                else record["refused_asks"][row["loc"]["i"]]
            )
            assert shown == [holder["text"], standards[row["taxonomy_id"]]]
        assert browser.problems() == []


def test_editing_the_posture_and_a_floor_statement_writes_entries_the_engine_applies(
    out_dir: Path, tmp_path: Path
) -> None:
    """The Posture & Floor edit boxes write the entry shapes ``apply-overrides`` reads, and
    reverting an edit removes its entry."""
    assert _EXE is not None
    doc = _doc(out_dir)
    invariant = doc["floor"]["invariants"][0]
    downloads = tmp_path / "downloads"
    downloads.mkdir()
    new_posture = "Edited in the page: hold exclusions and survival."
    new_statement = "Edited in the page: never concede this."
    with Browser(_EXE, download_dir=downloads) as browser:
        browser.add_script_on_load("delete window.showDirectoryPicker;")
        browser.goto((out_dir / "index.html").as_uri())
        browser.eval("document.querySelector('[data-tab=posture-floor]').click()")
        type_into = (
            "(() => {{ const ta = document.querySelector({sel}); ta.value = {text};"
            " ta.dispatchEvent(new Event('input', {{bubbles: true}})); }})()"
        )
        browser.eval(
            type_into.format(
                sel=json.dumps("textarea[data-target=posture]"), text=json.dumps(new_posture)
            )
        )
        floor_sel = (
            f'textarea[data-target=floor][data-id="{invariant["id"]}"][data-field=statement]'
        )
        browser.eval(type_into.format(sel=json.dumps(floor_sel), text=json.dumps(new_statement)))
        assert browser.eval("Object.keys(window.__indexPage.edits()).length") == 2
        # editing a field back to its original text drops the entry
        browser.eval(
            type_into.format(sel=json.dumps(floor_sel), text=json.dumps(invariant["statement"]))
        )
        assert browser.eval("Object.keys(window.__indexPage.edits()).length") == 1
        browser.eval(type_into.format(sel=json.dumps(floor_sel), text=json.dumps(new_statement)))
        _click(browser, "[data-tab=review]")
        _click(browser, "#btn-download")
        saved = _wait_download(downloads)
        assert browser.problems() == []
    by_target = {e["target"]: e for e in saved["overrides"]}
    assert by_target["posture"] == {
        "target": "posture",
        "id": "system_prompt",
        "value": new_posture,
        "basis": "owner",
    }
    assert by_target["floor"] == {
        "target": "floor",
        "id": invariant["id"],
        "field": "statement",
        "value": new_statement,
        "basis": "owner",
    }
    after = _applies_cleanly(out_dir, saved)
    assert after["posture"]["system_prompt"] == new_posture
    assert after["posture"]["version"] == doc["posture"]["version"] + 1
    assert after["floor"]["invariants"][0]["statement"] == new_statement
