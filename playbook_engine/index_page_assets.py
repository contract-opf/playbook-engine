"""Static CSS and JavaScript of ``index.html`` (issue #241).

Kept apart from :mod:`playbook_engine.document_renderer` so the renderer reads
as markup and the page behavior reads as a program. Both are inlined into the
one self-contained page: no network, no external font, script or stylesheet.

The script is progressive: with scripting off, every tab panel stays visible
(the page reads top to bottom) and only the editing controls are inert.
"""

from __future__ import annotations

CSS = r"""
:root {
  color-scheme: light dark;
  --bg: #f8f7f4; --panel: #ffffff; --ink: #1f2937; --muted: #6b7280; --faint: #9ca3af;
  --line: #e5e7eb; --accent: #1d4ed8; --accent-ink: #ffffff; --quote: #f9fafb;
  --warn-bg: #fffbeb; --warn-line: #fde68a; --warn-ink: #713f12;
  --ok: #166534; --bad: #991b1b;
  --c-less: #b91c1c; --c-diff: #b45309; --c-more: #166534; --c-equiv: #475569;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #14161a; --panel: #1c1f26; --ink: #e5e7eb; --muted: #9ca3af; --faint: #6b7280;
    --line: #2d323c; --accent: #7aa2ff; --accent-ink: #0b1020; --quote: #232731;
    --warn-bg: #2a2310; --warn-line: #5b4a14; --warn-ink: #f3dc9b;
    --ok: #6ee7a0; --bad: #fca5a5;
    --c-less: #fca5a5; --c-diff: #fcd34d; --c-more: #86efac; --c-equiv: #cbd5e1;
  }
}
* { box-sizing: border-box; }
body { font-family: Charter, Georgia, 'Times New Roman', serif; margin: 0; background: var(--bg);
       color: var(--ink); line-height: 1.55; }
main { max-width: 56rem; margin: 0 auto; padding: 2rem 1rem 5rem; }
h1 { font-size: 1.8rem; margin: 0 0 .15rem; letter-spacing: -0.01em; }
h2 { font-size: 1.2rem; margin: 0 0 .35rem; }
h3 { font-size: .8rem; text-transform: uppercase; letter-spacing: .08em; color: var(--muted);
     margin: 1.1rem 0 .4rem; }
code, .mono { font-family: ui-monospace, Menlo, Consolas, monospace; font-size: .86em; }
header.cover { border-bottom: 3px double var(--faint); margin-bottom: 1rem; padding-bottom: .9rem; }
.subtitle { color: var(--muted); font-variant: small-caps; letter-spacing: .08em; }
.meta, p.meta { color: var(--muted); font-size: .85rem; margin: 0 0 .5rem; }
.stats { display: flex; flex-wrap: wrap; gap: .5rem 1.5rem; margin-top: .75rem; font-size: .9rem; }
.stats b { display: block; font-size: 1.25rem; }
[role=tablist] { display: flex; flex-wrap: wrap; gap: .25rem; border-bottom: 1px solid var(--line);
                 margin: 0 0 1.25rem; position: sticky; top: 0; background: var(--bg); z-index: 3;
                 padding-top: .35rem; }
[role=tab] { font: inherit; font-size: .95rem; background: none; color: var(--muted); border: 0;
             border-bottom: 3px solid transparent; padding: .55rem .8rem; cursor: pointer; }
[role=tab][aria-selected=true] { color: var(--ink); border-bottom-color: var(--accent); font-weight: 600; }
[role=tab]:focus-visible, button:focus-visible, select:focus-visible, textarea:focus-visible,
input:focus-visible, summary:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
[role=tab] .count { font-size: .75rem; color: var(--faint); margin-left: .25rem; }
[role=tabpanel][hidden] { display: none; }
section.card, section.clause { background: var(--panel); border: 1px solid var(--line);
  border-radius: 8px; padding: 1.1rem 1.4rem; margin-bottom: 1.1rem; }
section.pending { background: var(--warn-bg); border: 1px solid var(--warn-line); color: var(--warn-ink);
  border-radius: 8px; padding: .9rem 1.2rem; margin-bottom: 1.1rem; }
blockquote { margin: .45rem 0; padding: .55rem .9rem; background: var(--quote);
  border-left: 3px solid var(--line); font-size: .92rem; white-space: pre-wrap; overflow-wrap: anywhere; }
blockquote .cite { margin-top: .35rem; font-size: .75rem; color: var(--faint);
  font-family: ui-monospace, Menlo, monospace; }
details { margin: .6rem 0; }
details summary { cursor: pointer; color: var(--accent); font-size: .92rem; }
table { border-collapse: collapse; width: 100%; font-size: .88rem; }
th, td { text-align: left; padding: .3rem .5rem; border-bottom: 1px solid var(--line); vertical-align: top; }
th { color: var(--muted); font-weight: 600; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
.table-wrap { overflow-x: auto; }
ol.steps li { margin: .4rem 0; }
.btn { font: inherit; font-size: .9rem; border: 1px solid var(--accent); background: var(--accent);
  color: var(--accent-ink); border-radius: 6px; padding: .4rem .8rem; cursor: pointer; }
.btn.secondary { background: transparent; color: var(--accent); }
.btn[disabled] { opacity: .5; cursor: default; }
.chip { display: inline-block; font-size: .72rem; text-transform: uppercase; letter-spacing: .05em;
  border: 1px solid currentColor; border-radius: 999px; padding: 0 .5rem; font-family: system-ui, sans-serif; }
.chip.less_protective { color: var(--c-less); } .chip.different_concept { color: var(--c-diff); }
.chip.more_protective { color: var(--c-more); } .chip.equivalent { color: var(--c-equiv); }
.chip.owner { color: var(--ok); }
.savebar { display: flex; flex-wrap: wrap; align-items: center; gap: .5rem .8rem; }
#save-status { font-size: .88rem; color: var(--muted); }
#save-status.bad { color: var(--bad); }
.progress { font-size: .9rem; }
.filters { display: flex; flex-wrap: wrap; gap: .4rem; margin: .8rem 0; }
.filters button { font: inherit; font-size: .82rem; border: 1px solid var(--line); background: var(--panel);
  color: var(--ink); border-radius: 999px; padding: .15rem .7rem; cursor: pointer; }
.filters button[aria-pressed=true] { border-color: var(--accent); color: var(--accent); font-weight: 600; }
article.rv { border: 1px solid var(--line); border-radius: 8px; padding: .8rem 1rem; margin: .7rem 0;
  background: var(--panel); }
article.rv.done { border-left: 4px solid var(--ok); }
article.rv header { display: flex; flex-wrap: wrap; gap: .3rem .7rem; align-items: baseline; }
article.rv .reason { margin: .35rem 0; font-size: .92rem; }
article.rv .acts { display: flex; flex-wrap: wrap; gap: .4rem .6rem; align-items: center; margin-top: .5rem; }
article.rv input[type=text], article.rv select, textarea.edit { font: inherit; font-size: .88rem;
  background: var(--panel); color: var(--ink); border: 1px solid var(--line); border-radius: 6px; padding: .25rem .4rem; }
article.rv input[type=text] { min-width: 14rem; flex: 1; }
textarea.edit { width: 100%; min-height: 8rem; font-size: .92rem; }
details.clause > summary { display: flex; flex-wrap: wrap; gap: .25rem .7rem; align-items: baseline; list-style-position: outside; }
.ev-title { font-weight: 600; }
.pills { display: flex; flex-wrap: wrap; gap: .2rem .4rem; }
.pill { font-size: .75rem; color: var(--muted); border: 1px solid var(--line); border-radius: 999px;
  padding: 0 .5rem; font-family: system-ui, sans-serif; white-space: nowrap; }
.bulk { display: flex; flex-wrap: wrap; gap: .5rem; margin: 0 0 .6rem; }
.bulk .btn { font-size: .82rem; padding: .2rem .6rem; }
.edited-flag { font-size: .78rem; color: var(--ok); margin-left: .4rem; }
.muted { color: var(--muted); } .small { font-size: .85rem; }
.toc a { color: var(--accent); text-decoration: none; }
.toc li { margin: .15rem 0; }
.hint { font-size: .85rem; color: var(--muted); }
footer { margin-top: 2.5rem; padding-top: .8rem; border-top: 1px solid var(--line); color: var(--faint); font-size: .8rem; }
@media print {
  body { background: #fff; color: #000; } [role=tablist], .savebar, .acts { display: none; }
  [role=tabpanel][hidden] { display: block; }
  section.card, section.clause { border: none; page-break-inside: avoid; }
}
"""


JS = r"""
(function () {
  'use strict';
  var doc = document;
  function $(s, r) { return (r || doc).querySelector(s); }
  function $$(s, r) { return Array.prototype.slice.call((r || doc).querySelectorAll(s)); }
  function readJSON(id) {
    var el = doc.getElementById(id);
    if (!el) return null;
    try { return JSON.parse(el.textContent); } catch (e) { return null; }
  }
  function h(tag, attrs, kids) {
    var el = doc.createElement(tag);
    Object.keys(attrs || {}).forEach(function (k) {
      if (k === 'text') el.textContent = attrs[k];
      else if (k === 'class') el.className = attrs[k];
      else el.setAttribute(k, attrs[k]);
    });
    (kids || []).forEach(function (c) { if (c) el.appendChild(typeof c === 'string' ? doc.createTextNode(c) : c); });
    return el;
  }

  var META = readJSON('page-meta') || {};
  var ROWS = readJSON('review-rows') || [];
  var PF = readJSON('pf-data') || { posture: null, floor: [] };
  var FILE = META.overrides_file || 'overrides.json';
  var LABEL_NAMES = {
    less_protective: 'Less protective', different_concept: 'Different concept',
    more_protective: 'More protective', equivalent: 'Equivalent'
  };
  var LABELS = ['less_protective', 'different_concept', 'more_protective', 'equivalent'];

  /* ---------- tabs ---------- */
  var tabs = $$('[role=tab]');
  var panels = $$('[role=tabpanel]');
  function showTab(id, focus) {
    tabs.forEach(function (t) {
      var on = t.getAttribute('data-tab') === id;
      t.setAttribute('aria-selected', on ? 'true' : 'false');
      t.tabIndex = on ? 0 : -1;
      if (on && focus) t.focus();
    });
    panels.forEach(function (p) { p.hidden = p.id !== 'tab-' + id; });
    try { history.replaceState(null, '', '#' + id); } catch (e) { /* file:// may refuse */ }
  }
  tabs.forEach(function (t, i) {
    t.addEventListener('click', function () { showTab(t.getAttribute('data-tab'), false); });
    t.addEventListener('keydown', function (ev) {
      var d = ev.key === 'ArrowRight' ? 1 : ev.key === 'ArrowLeft' ? -1 : 0;
      if (d) { ev.preventDefault(); showTab(tabs[(i + d + tabs.length) % tabs.length].getAttribute('data-tab'), true); }
    });
  });
  $$('[data-goto]').forEach(function (a) {
    a.addEventListener('click', function (ev) { ev.preventDefault(); showTab(a.getAttribute('data-goto'), false); });
  });
  var first = (location.hash || '').replace('#', '');
  showTab(tabs.some(function (t) { return t.getAttribute('data-tab') === first; }) ? first : 'start', false);

  /* ---------- where this file lives (file:// only) ---------- */
  var folderPath = '';
  if (location.protocol === 'file:') {
    try {
      folderPath = decodeURIComponent(location.pathname.replace(/[^\/]*$/, ''));
      if (/^\/[A-Za-z]:/.test(folderPath)) folderPath = folderPath.slice(1);
    } catch (e) { folderPath = ''; }
  }
  $$('.js-folder').forEach(function (el) { if (folderPath) el.textContent = folderPath; });
  $$('.js-opf-path').forEach(function (el) { if (folderPath) el.textContent = folderPath + 'playbook.opf.json'; });
  $$('.js-index-path').forEach(function (el) { if (folderPath) el.textContent = folderPath + 'index.html'; });

  /* ---------- the edits ---------- */
  var edits = {};
  var fileExists = false;
  function ekey(e) { return e.target + '|' + e.id + '|' + (e.field || ''); }
  function validEntry(e) {
    return e && typeof e === 'object' && typeof e.id === 'string' &&
      (e.target === 'vs_standard' || e.target === 'posture' || e.target === 'floor');
  }
  function mergeFile(obj) {
    var list = obj && Array.isArray(obj.overrides) ? obj.overrides : [];
    list.forEach(function (e) { if (validEntry(e)) edits[ekey(e)] = e; });
  }
  function fileObject() {
    var list = Object.keys(edits).map(function (k) { return edits[k]; });
    list.sort(function (a, b) { return ekey(a) < ekey(b) ? -1 : ekey(a) > ekey(b) ? 1 : 0; });
    var out = { overrides_version: 1 };
    if (META.content_hash) out.based_on = META.content_hash;
    out.overrides = list;
    return out;
  }
  function fileText() { return JSON.stringify(fileObject(), null, 2) + '\n'; }
  function entry(target, id, field, value, note) {
    var e = { target: target, id: id };
    if (field) e.field = field;
    e.value = value; e.basis = 'owner';
    if (note) e.note = note;
    return e;
  }
  mergeFile(readJSON('overrides-initial'));
  if (readJSON('overrides-initial')) fileExists = true;

  /* ---------- saving ---------- */
  var canPick = typeof window.showDirectoryPicker === 'function' && window.isSecureContext !== false;
  var folder = { handle: null };
  var statusEl = $('#save-status');
  var dirty = false, timer = null;
  function status(text, bad) {
    if (!statusEl) return;
    statusEl.textContent = text;
    statusEl.className = bad ? 'bad' : '';
  }
  function describeState() {
    var n = Object.keys(edits).length;
    if (folder.handle) return n + ' edit' + (n === 1 ? '' : 's') + (dirty ? ' - saving...' : ' saved to ' + FILE + ' in "' + folder.handle.name + '"');
    if (!n) return canPick ? 'No edits yet. Connect folder to save them as you go.' : 'No edits yet.';
    return n + ' edit' + (n === 1 ? '' : 's') + ' not saved yet: ' + (canPick ? 'Connect folder, or ' : '') + 'Download edits and put the file in the playbook folder.';
  }
  async function flush() {
    if (!folder.handle) return;
    if (!Object.keys(edits).length && !fileExists) { dirty = false; status(describeState()); return; }
    try {
      var fh = await folder.handle.getFileHandle(FILE, { create: true });
      var w = await fh.createWritable();
      await w.write(fileText());
      await w.close();
      fileExists = true; dirty = false;
      status(describeState());
    } catch (e) {
      status('Could not save to the folder (' + (e && e.message ? e.message : e) + '). Use Download edits instead.', true);
    }
  }
  var unsaved = false;
  window.addEventListener('beforeunload', function (ev) {
    if (unsaved && !folder.handle) { ev.preventDefault(); ev.returnValue = ''; }
  });
  function changed() {
    unsaved = true;
    dirty = true;
    status(describeState());
    if (folder.handle) { clearTimeout(timer); timer = setTimeout(flush, 200); }
    updateProgress();
  }
  function download() {
    var blob = new Blob([fileText()], { type: 'application/json' });
    var a = h('a', { href: URL.createObjectURL(blob), download: FILE });
    doc.body.appendChild(a); a.click();
    setTimeout(function () { URL.revokeObjectURL(a.href); a.remove(); }, 0);
    unsaved = false;
    status('Downloaded ' + FILE + '. Put it in the playbook folder next to playbook.opf.json.');
  }

  var idb = {
    open: function () {
      return new Promise(function (res, rej) {
        var r = indexedDB.open('playbook-index-page', 1);
        r.onupgradeneeded = function () { r.result.createObjectStore('handles'); };
        r.onsuccess = function () { res(r.result); };
        r.onerror = function () { rej(r.error); };
      });
    },
    get: async function (k) {
      var db = await idb.open();
      return new Promise(function (res, rej) {
        var q = db.transaction('handles').objectStore('handles').get(k);
        q.onsuccess = function () { res(q.result); }; q.onerror = function () { rej(q.error); };
      });
    },
    put: async function (k, v) {
      var db = await idb.open();
      return new Promise(function (res, rej) {
        var t = db.transaction('handles', 'readwrite');
        t.objectStore('handles').put(v, k);
        t.oncomplete = function () { res(); }; t.onerror = function () { rej(t.error); };
      });
    }
  };
  // One remembered folder per page location. Chrome gives every file:// page ONE storage
  // origin, so a key shared by out-dirs (say, the agreement type) would hand this page the
  // folder of another out-dir; the page's own path is stable across rebuilds of its out-dir.
  var handleKey = 'folder:' + location.protocol + '//' + location.host + location.pathname;

  function shortHash(hash) { return hash ? String(hash).slice(0, 19) + '...' : 'none'; }
  // Null when the folder's playbook.opf.json is the playbook this page shows (agreement
  // type, perspective and content hash); otherwise why not. Nothing else in the folder is
  // read before this passes.
  async function whyNotThisPlaybook(handle) {
    var text;
    try { text = await (await (await handle.getFileHandle('playbook.opf.json')).getFile()).text(); }
    catch (e) {
      if (e && e.name === 'NotFoundError') return 'That folder has no playbook.opf.json. Choose the folder this page is in.';
      return 'Could not read playbook.opf.json in that folder (' + (e && e.message ? e.message : e) + ').';
    }
    var opf = null;
    try { opf = JSON.parse(text); } catch (e) { opf = null; }
    if (!opf || typeof opf !== 'object') return 'playbook.opf.json in that folder is not valid JSON. Choose the folder this page is in.';
    var kept = ' No edits were loaded from or written to that folder.';
    var at = (opf.agreement_type || {}).id || null;
    var p = opf.perspective || {}, mp = META.perspective || {};
    if (at !== (META.agreement_type || null) || (p.party || null) !== (mp.party || null) ||
        (p.counterparty_type || null) !== (mp.counterparty_type || null)) {
      return 'That folder holds a different playbook' + (at ? ' (agreement type "' + at + '")' : '') +
        ', not the one this page shows.' + kept + ' Choose the folder this page is in.';
    }
    var hash = (opf.identity || {}).content_hash || null;
    if (hash !== (META.content_hash || null)) {
      return 'That folder holds a different build of this playbook (content hash ' + shortHash(hash) +
        '; this page shows ' + shortHash(META.content_hash) + ').' + kept +
        ' If that is the folder this page is in, its playbook changed after the page was built: rebuild the page (playbook view bundle) and open it again. Otherwise choose the folder this page is in.';
    }
    return null;
  }

  async function adopt(handle) {
    var refusal = await whyNotThisPlaybook(handle);
    if (refusal) { status(refusal, true); return false; }
    var diskText = null;
    try {
      var existing = await (await handle.getFileHandle(FILE)).getFile();
      diskText = await existing.text();
    } catch (e) { if (!e || e.name !== 'NotFoundError') { status('Could not read ' + FILE + ' in that folder: ' + (e && e.message), true); return false; } }
    if (diskText !== null) {
      var parsed = null;
      try { parsed = JSON.parse(diskText); } catch (e) { parsed = undefined; }
      if (parsed === undefined) { status(FILE + ' in that folder is not valid JSON. Fix or delete it, then connect again.', true); return false; }
      mergeFile(parsed); fileExists = true;
    }
    folder.handle = handle;
    try { await idb.put(handleKey, handle); } catch (e) { /* a handle that cannot be stored just will not reconnect */ }
    setSaveUi();
    renderReview(); syncPF();
    dirty = true;
    await flush();
    updateProgress();
    return true;
  }
  async function connect() {
    var handle;
    try { handle = await window.showDirectoryPicker({ mode: 'readwrite' }); }
    catch (e) { if (e && e.name === 'AbortError') return; status('Could not open the folder: ' + (e && e.message), true); return; }
    await adopt(handle);
  }
  async function reconnect(handle) {
    try {
      var perm = await handle.requestPermission({ mode: 'readwrite' });
      if (perm !== 'granted') { status('Permission to the folder was not granted.', true); return; }
    } catch (e) { status('Could not reconnect: ' + (e && e.message), true); return; }
    await adopt(handle);
  }
  var remembered = null;
  function setSaveUi() {
    var c = $('#btn-connect'), r = $('#btn-reconnect'), note = $('#save-note');
    if (c) c.hidden = !canPick || !!folder.handle;
    if (r) { r.hidden = !remembered || !!folder.handle; if (remembered) r.textContent = 'Reconnect "' + remembered.name + '"'; }
    if (note) note.hidden = canPick;
    status(describeState());
  }
  var bc = $('#btn-connect'); if (bc) bc.addEventListener('click', connect);
  var br = $('#btn-reconnect'); if (br) br.addEventListener('click', function () { if (remembered) reconnect(remembered); });
  var bd = $('#btn-download'); if (bd) bd.addEventListener('click', download);
  if (canPick) {
    idb.get(handleKey).then(async function (hd) {
      if (!hd) return;
      remembered = hd;
      try { if ((await hd.queryPermission({ mode: 'readwrite' })) === 'granted') { await adopt(hd); return; } } catch (e) { /* ask the user instead */ }
      setSaveUi();
    }).catch(function () { /* no stored handle */ });
  }

  /* ---------- Review tab ---------- */
  var reviewEl = $('#review-list');
  var filter = 'all', shown = 25;
  var canonical = null, precById = null;
  function loadCanonical() {
    if (canonical) return canonical;
    canonical = readJSON('opf-canonical') || {};
    precById = {};
    ((canonical.evidence || {}).precedent || []).forEach(function (p) { precById[p.id] = p; });
    return canonical;
  }
  function fullText(row) {
    loadCanonical();
    var rec = precById[row.loc.record];
    if (!rec) return null;
    var slot = row.loc.role === 'signed' ? rec.signed_text : row.loc.role === 'opening' ? rec.opening_text : (rec.refused_asks || [])[row.loc.i || 0];
    return slot && slot.text || null;
  }
  function standardText(row) {
    loadCanonical();
    var cl = ((canonical.evidence || {}).clauses || []).filter(function (c) { return c.taxonomy_id === row.taxonomy_id; })[0];
    return cl && cl.our_standard && cl.our_standard.text || null;
  }
  function rowEdit(row) { return edits['vs_standard|' + row.key + '|']; }
  function effLabel(row) { var e = rowEdit(row); return e ? e.value.label : row.label; }
  function matches(row) {
    if (filter === 'all') return true;
    if (filter === 'todo') return !rowEdit(row);
    return row.label === filter;
  }
  function setEdit(row, label, note) {
    edits['vs_standard|' + row.key + '|'] = entry('vs_standard', row.key, null, { label: label }, note);
    changed();
  }
  function clearEdit(row) { delete edits['vs_standard|' + row.key + '|']; changed(); }
  function buildRow(row) {
    var art = h('article', { class: 'rv', 'data-key': row.key });
    function fill() {
      art.textContent = '';
      var e = rowEdit(row);
      art.className = 'rv' + (e ? ' done' : '');
      var head = h('header', {}, [
        h('span', { class: 'chip ' + row.label, text: LABEL_NAMES[row.label] }),
        h('strong', { text: row.title }),
        h('span', { class: 'muted small', text: row.n_deals + ' deal' + (row.n_deals === 1 ? '' : 's') + ' - ' + row.roles.join(', ') + ' - judged by ' + row.basis + (row.check && row.check !== 'owner' ? ' (' + row.check + ')' : '') })
      ]);
      if (e) head.appendChild(h('span', { class: 'chip owner', text: e.value.label === row.label ? 'You confirmed' : 'You set: ' + LABEL_NAMES[e.value.label] }));
      art.appendChild(head);
      if (row.reason) art.appendChild(h('p', { class: 'reason', text: row.reason }));
      art.appendChild(h('blockquote', { text: row.text }, [h('div', { class: 'cite', text: row.cite })]));
      var det = h('details', {}, [h('summary', { text: 'Show the full text and our standard' })]);
      det.addEventListener('toggle', function () {
        if (!det.open || det.getAttribute('data-filled')) return;
        det.setAttribute('data-filled', '1');
        det.appendChild(h('h3', { text: 'Full text' }));
        det.appendChild(h('blockquote', { text: fullText(row) || row.text }));
        det.appendChild(h('h3', { text: 'Our standard' }));
        det.appendChild(h('blockquote', { text: standardText(row) || row.standard }));
      });
      art.appendChild(det);
      var acts = h('div', { class: 'acts' });
      var confirm = h('button', { type: 'button', class: 'btn secondary', text: 'Confirm' });
      confirm.addEventListener('click', function () { setEdit(row, row.label, e && e.note); fill(); });
      var sel = h('select', { 'aria-label': 'Change label' });
      sel.appendChild(h('option', { value: '', text: 'Change label...' }));
      LABELS.forEach(function (l) { if (l !== row.label) sel.appendChild(h('option', { value: l, text: LABEL_NAMES[l] })); });
      sel.addEventListener('change', function () { if (sel.value) { setEdit(row, sel.value, e && e.note); fill(); } });
      acts.appendChild(confirm); acts.appendChild(sel);
      if (e) {
        var note = h('input', { type: 'text', placeholder: 'Note (optional)', 'aria-label': 'Note', value: e.note || '' });
        note.addEventListener('change', function () { setEdit(row, e.value.label, note.value.trim() || null); });
        var undo = h('button', { type: 'button', class: 'btn secondary', text: 'Undo' });
        undo.addEventListener('click', function () { clearEdit(row); fill(); });
        acts.appendChild(note); acts.appendChild(undo);
      }
      art.appendChild(acts);
    }
    fill();
    return art;
  }
  function renderReview() {
    if (!reviewEl) return;
    reviewEl.textContent = '';
    var list = ROWS.filter(matches);
    list.slice(0, shown).forEach(function (r) { reviewEl.appendChild(buildRow(r)); });
    var more = $('#review-more');
    if (more) { more.hidden = list.length <= shown; more.textContent = 'Show ' + Math.min(25, list.length - shown) + ' more (' + (list.length - shown) + ' left)'; }
    var empty = $('#review-empty'); if (empty) empty.hidden = list.length > 0;
    var todo = list.filter(function (r) { return !rowEdit(r); }).length;
    if (bulkConfirm) { bulkConfirm.textContent = 'Confirm all ' + todo + ' unreviewed in this view'; bulkConfirm.disabled = !todo; }
    if (bulkUndo) { bulkUndo.disabled = list.length === todo; }
    updateProgress();
  }
  function updateProgress() {
    var done = ROWS.filter(rowEdit).length;
    var p = $('#review-progress');
    if (p) p.textContent = ROWS.length ? done + ' of ' + ROWS.length + ' reviewed (optional)' : 'Nothing to review.';
    var c = $('#tab-review-count'); if (c) c.textContent = ROWS.length ? ' ' + done + '/' + ROWS.length : '';
    $$('.filters button').forEach(function (b) {
      var f = b.getAttribute('data-filter');
      var n = f === 'all' ? ROWS.length : f === 'todo' ? ROWS.length - done : ROWS.filter(function (r) { return r.label === f; }).length;
      b.textContent = b.getAttribute('data-name') + ' (' + n + ')';
      b.setAttribute('aria-pressed', f === filter ? 'true' : 'false');
    });
  }
  $$('.filters button').forEach(function (b) {
    b.addEventListener('click', function () { filter = b.getAttribute('data-filter'); shown = 25; renderReview(); });
  });
  var more = $('#review-more');
  if (more) more.addEventListener('click', function () { shown += 25; renderReview(); });
  var bulkConfirm = $('#bulk-confirm'), bulkUndo = $('#bulk-undo');
  if (bulkConfirm) bulkConfirm.addEventListener('click', function () {
    ROWS.filter(matches).forEach(function (r) {
      if (!rowEdit(r)) edits['vs_standard|' + r.key + '|'] = entry('vs_standard', r.key, null, { label: r.label });
    });
    changed(); renderReview();
  });
  if (bulkUndo) bulkUndo.addEventListener('click', function () {
    ROWS.filter(matches).forEach(function (r) { delete edits['vs_standard|' + r.key + '|']; });
    changed(); renderReview();
  });

  /* ---------- Posture & Floor tab ---------- */
  var pfFields = [];
  function bindPF(ta, original, target, id, field) {
    var flag = ta.parentNode.querySelector('.edited-flag');
    var revert = ta.parentNode.querySelector('.revert');
    function sync() {
      var isEdit = !!edits[target + '|' + id + '|' + (field || '')];
      if (flag) flag.hidden = !isEdit;
      if (revert) revert.hidden = !isEdit;
    }
    ta.addEventListener('input', function () {
      var k = target + '|' + id + '|' + (field || '');
      if (ta.value === original || !ta.value.trim()) delete edits[k];
      else edits[k] = entry(target, id, field, ta.value, edits[k] && edits[k].note);
      sync(); changed();
    });
    if (revert) revert.addEventListener('click', function () { ta.value = original; delete edits[target + '|' + id + '|' + (field || '')]; sync(); changed(); });
    pfFields.push({ ta: ta, original: original, target: target, id: id, field: field, sync: sync });
    sync();
  }
  function syncPF() {
    pfFields.forEach(function (f) {
      var e = edits[f.target + '|' + f.id + '|' + (f.field || '')];
      f.ta.value = e ? e.value : f.original;
      f.sync();
    });
  }
  $$('textarea.edit').forEach(function (ta) {
    var target = ta.getAttribute('data-target'), id = ta.getAttribute('data-id'), field = ta.getAttribute('data-field') || null;
    var original = target === 'posture' ? (PF.posture && PF.posture.system_prompt) || '' :
      ((PF.floor || []).filter(function (i) { return i.id === id; })[0] || {})[field] || '';
    ta.value = original;
    bindPF(ta, original, target, id, field);
  });
  syncPF();

  setSaveUi();
  renderReview();
  window.__indexPage = { fileObject: fileObject, flush: flush, canPick: canPick, edits: function () { return edits; } };
}());
"""
