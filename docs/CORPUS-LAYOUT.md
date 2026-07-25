# Corpus layout — how to organize your files

The engine reads a **corpus directory**: one folder per agreement, with every version of that agreement inside it. You do not need to label which version is which, or which one was signed — the engine infers order and detects the signed copy. You just need to put all versions of one agreement in one folder.

The layout below is a **fast path**, not a requirement — **any arrangement is supported**. If your export looks different (loose files with no per-agreement folders, an email-export tree, ad-hoc naming), you don't need to restructure anything by hand: run `playbook stage --plan-only` and the `playbook-from-corpus` skill will assemble the story from the file contents and ask you only what it cannot infer (see the skill's "Assemble the corpus story" stage).

## Minimum layout

```
corpus/
├── university-of-example/
│   ├── draft-we-sent.docx
│   ├── their-redline.docx
│   ├── our-counter.docx
│   └── fully-executed.pdf
├── another-school/
│   ├── v1.docx
│   └── signed.pdf
└── ...
```

Rules:
- **One folder per agreement.** All versions of the same negotiation go together. Different agreements go in different folders.
- **Any common format works** — `.docx`, `.pdf`, `.rtf`. Mixed formats in one folder are fine (drafts as Word, signed copy as PDF is typical).
- **Filenames don't matter.** Name them anything. The engine does not trust filenames to decide order, status, or scope.
- **Include the signed copy if you have one.** It is the strongest signal (what actually became binding). A folder with no signed copy still works; it's treated as an in-progress trail.
- **Throw in everything, even off-topic documents — via the judge workflow.** Every document always gets a scope decision logged to `scope.json` with a rationale; a document is never silently dropped. But substantive out-of-scope detection (e.g. flagging a stray data-sharing agreement as not this agreement type) requires a real scope judge: run `playbook judge` (queues an unreviewed document's scope for a verdict rather than assuming one) and `playbook judge-apply` before `playbook mine`/`project` — see `playbook judge --help`. Without that workflow, `playbook compile`'s (and `mine`'s, without a verdict store) built-in scope check is a stub that accepts every non-trivial document as in-scope at confidence 0.5 rather than excluding anything — pre-clean off-topic documents yourself if you're not running the judge workflow.

## Optional: hints

If you happen to know metadata, you can drop a `hints.yaml` in a document folder. Everything in it is optional, but the three fields behave differently:

```yaml
signed_version: fully-executed.pdf   # if you know which is signed
order: [draft-we-sent.docx, their-redline.docx, our-counter.docx, fully-executed.pdf]
provenance: our_paper                # or counterparty_paper
```

- `signed_version` and `provenance` are **hard overrides** — if set, they replace whatever the engine detected outright (confidence 1.0, `basis: hint`), even if the engine's own heuristic was confident and disagreed.
- `order` only **seeds** the version-ordering inference — it is used to help resolve ties, but never overrides strong content evidence when the two conflict.

An unrecognized filename in `signed_version` or `order` (one that doesn't match a discovered version) produces a warning and is ignored rather than silently accepted.

## Config: defining the agreement type

One config file per playbook tells the engine what type it's compiling and where your template is (if you have one):

```yaml
agreement_type:
  id: educational-affiliation
  name: "Educational Affiliation Agreement"
baseline:
  template: ./template/internship-template.rtf   # your standard paper, or null
taxonomy: ./spec/taxonomy/affiliation-agreement.yaml
provenance:
  our_party_aliases: ["FixtureCorp", "FixtureCorp Holdings, LLC"]
perspective:                       # optional — whose "us" this playbook is
  party: "FixtureCorp"                    # reviewed as; an open-standard OPF
  counterparty_type: "Educational Institution"   # instance must say who "us" is
```

If you don't have a canonical template, set `baseline.template: null` — the engine will build an *emergent* playbook from your signed deals (weaker, but still useful).

If you omit `perspective` entirely, `party` defaults from `provenance.our_party_aliases[0]`; `counterparty_type` has no default (never fabricated). Both fields must be set for the playbook to actually carry a `perspective` block.

## What you get back

A single `playbook.opf.json` (plus human-readable intermediates in `out/`). It validates against the [Open Playbook Format](OPF-SPEC.md) and can be read by any OPF-aware review engine.
