# Quick Compile

> **This is a quick local smoke run — stub judges, no LLM calls, output is
> structurally valid but semantically blank.** For the real derivation path
> (LLM-backed judgment producing a playbook with actual semantics), see
> [`.claude/skills/playbook-from-corpus/SKILL.md`](../.claude/skills/playbook-from-corpus/SKILL.md).

Turn a folder of contract files into a negotiation playbook — without writing any code.

---

## What this guide covers

1. **Checks** your folder layout and tells you what to fix.
2. **Runs** the deterministic pipeline stages (ingest, structure, classify).
3. **Stubs** the LLM judgment stages (scope, deviation) — structurally
   valid output, semantically blank (see the banner above).
4. **Emits** a validated `playbook.opf.json` you can review and share.

---

## Before you start: layout your folder

Your corpus must follow this structure. Every agreement gets its own subfolder; put all drafts plus the final signed copy inside it.

```
corpus/
├── university-of-example/
│   ├── v1-draft-we-sent.docx
│   ├── v2-their-redline.docx
│   └── v3-fully-executed.pdf
├── another-school/
│   ├── draft.docx
│   └── signed.pdf
└── ...
```

Rules:
- **One subfolder per agreement.** All versions of the same negotiation go in the same subfolder.
- **Supported formats:** `.docx`, `.pdf`, `.rtf`. Mixed formats in one subfolder are fine.
- **Filenames don't matter.** The engine infers order from content, not names.
- **Include the signed copy** if you have it — it anchors the negotiation trail.
- **Anything goes in.** Off-topic documents are flagged and excluded automatically.

You also need a **config file** (`playbook.config.yaml`) in the same directory as your corpus. Use the example at `examples/affiliation-config/playbook.config.yaml` as a starting point.

---

## Step 1 — Check your layout

```bash
playbook lint-corpus ./corpus --config ./playbook.config.yaml
```

Fix every **ERR** item before proceeding. **WARN** items are advisory.

### Common errors and how to fix them

| Error | What to do |
|-------|-----------|
| `CORPUS_NOT_FOUND` | Check the path — the folder doesn't exist yet. |
| `EMPTY_CORPUS` | Add at least one agreement subfolder with files. |
| `DOC_NO_SUPPORTED_FILES` | Add `.docx`, `.pdf`, or `.rtf` files to the subfolder, or delete it. |
| `CONFIG_NOT_FOUND` | Create a `playbook.config.yaml` (copy the example). |
| `CONFIG_MISSING_TAXONOMY` | Set `taxonomy:` in your config to the taxonomy YAML path. |
| `CONFIG_TAXONOMY_NOT_FOUND` | Check the taxonomy path — the file is missing or misspelled. |
| `CONFIG_TEMPLATE_NOT_FOUND` | Check `baseline.template` — the file is missing. Set to `null` if you don't have a template. |

### Common warnings and what they mean

| Warning | What it means |
|---------|--------------|
| `DOC_SINGLE_VERSION` | Only one file in a subfolder — can't show negotiation history. Consider adding more drafts. |
| `DOC_UNSUPPORTED_FILES` | Stray `.txt`, `.xlsx`, etc. files — harmless, but worth cleaning up. |
| `CONFIG_NO_TEMPLATE` | No canonical template configured — the playbook will be emergent (weaker positions). |

---

## Step 2 — Compile the playbook

`playbook compile` is the one-shot convenience wrapper — `mine` + `project`
in a single command, stub judges throughout. The judged derivation path the
packaged skill drives runs the stages separately (`mine` → `judge` /
`judge-apply` → `project`); see [PLAN-FIRST.md](PLAN-FIRST.md).

```bash
playbook compile ./corpus \
  --config ./playbook.config.yaml \
  --out ./out
```

This runs the full pipeline. Progress is printed to the console. When it finishes:

```
OK  ./out/playbook.opf.json
```

### If compilation fails

- **Config/taxonomy errors:** re-run `lint-corpus` and fix the reported errors.
- **Scope gate errors:** the engine couldn't determine if a document is the right agreement type. Check that your corpus subfolders contain the right documents. Add a `hints.yaml` file if needed (see corpus layout guide).
- **Validation errors:** the assembled playbook failed internal consistency checks. Open an issue with the full error output.

---

## Step 3 — Validate the output

```bash
playbook validate ./out/playbook.opf.json
```

A green `OK` means the playbook is valid. Red lines are blocking errors.

---

## Step 4 — Review the intermediates

Open `./out/` to review what the engine inferred:

| File | What it contains |
|------|-----------------|
| `scope.json` | Which documents were in-scope and why. |
| `trail/*.json` | Inferred version order, signed copy, and provenance for each agreement. |
| `normalized/*/*.clauses.json` | Extracted clause trees — useful for debugging ingestion. |
| `observations.jsonl` | One row per clause observation feeding the playbook. |
| `playbook.opf.json` | The final playbook. |

Review `scope.json` and `trail/` before trusting the playbook. If the engine got the signed copy wrong or misidentified provenance, add a `hints.yaml` to the relevant subfolder:

```yaml
# corpus/university-of-example/hints.yaml
signed_version: v3-fully-executed.pdf
provenance: our_paper
order:
  - v1-draft-we-sent.docx
  - v2-their-redline.docx
  - v3-fully-executed.pdf
```

Then re-run `playbook compile --no-cache` to pick up the hints.

---

## Re-running after changes

```bash
# Default: resumes from saved observations (fast)
playbook compile ./corpus --config ./playbook.config.yaml --out ./out

# Force full re-run (e.g. after adding new documents)
playbook compile ./corpus --config ./playbook.config.yaml --out ./out --no-cache
```

The engine auto-detects corpus changes (new files, modified files) and forces a full re-run even without `--no-cache`.

---

## For LLM judgment stages

The engine ships with **stub judges** that run without LLM access. They produce valid output but do not perform semantic judgment:

- Every document is accepted as in-scope.
- All clause deviations are marked as substantive + neutral risk.
- Clause classification relies on keyword matching (Jaccard similarity) only.

For real semantic judgment, you don't write code: use the packaged
[`playbook-from-corpus` skill](../.claude/skills/playbook-from-corpus/SKILL.md)
(Claude acts as the judge on your plan — no API key) or configure an
`ANTHROPIC_API_KEY` for headless judges; [PLAN-FIRST.md](PLAN-FIRST.md)
compares the two, stage by stage.

---

## Troubleshooting

**"no .docx/.pdf/.rtf files found"** — Check that your files are in a *subfolder*, not directly in the corpus root. Each agreement needs its own subfolder.

**"Only 1 version file"** — The engine can still compile with a single version, but cannot show negotiation history. Positions will carry low confidence (`historical_stance: no_signal`) without a signed-vs-draft comparison.

**Playbook has no clauses** — If all clauses are unclassified (taxonomy_id=None), the compiled clauses list will be empty. This usually means the taxonomy doesn't match the document content. Check that your taxonomy covers the agreement type, or switch to a more appropriate taxonomy.

**"Accepted without LLM judgment"** — This is the stub scope judge. Every document was accepted as in-scope. To get real judgment on scope, run the `playbook judge` / `judge-apply` review loop (see docs/ORCHESTRATION.md).
