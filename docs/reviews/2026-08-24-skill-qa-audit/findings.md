# Skill QA Audit — Findings (2026-08-24)

Comprehensive QA audit of the `playbook-from-corpus` skill: 10 parallel review workstreams plus an adversarial verification pass. 123 candidate findings; 122 CONFIRMED, 1 refuted. Grouped below by area, severity-ordered; duplicate discoveries across workstreams are merged, and each entry links its GitHub issue (label `skill-qa-audit`). Evidence line numbers reference `main` = 500e0aa.

Counterparty-identifying details are scrubbed for this public copy; the raw appendix with per-finding verifier notes is local-only (see README.md here).


## Control ladder & Route C

The skill's central design claim — human-authored Posture and Floor, never derived, never lost — is where the worst defect lives.

### [CRITICAL] Re-running `project` silently wipes authored Posture and signed Floor; Route C's survival promise is false

*Audit finding(s) #44, #77, #98 · doc-vs-code · pipeline-state, control-ladder, failure-modes · CONFIRMED · [issue](https://github.com/contract-opf/playbook-engine/issues/123)*

**What:** SKILL.md Route C promises "The Posture and Floor you already signed should survive" a re-derivation and states "Curation pins and signed Floor invariants survive a recompile by design", but `playbook project` unconditionally resets both sections to {} — only the `curation` section is carried forward from the prior playbook.

**Evidence:** playbook_engine/playbook_assembler.py:344-345 (`playbook["posture"] = {}` / `playbook["floor"] = {}`, with a comment saying content waits for "a later slice"); playbook_engine/pipeline.py:3481-3488 reads the prior playbook ONLY for `prior_playbook.get("curation")`; SKILL.md:63-65 and the Route C row at SKILL.md:90. The spec (docs/OPF-SPEC.md:70, docs/ADOPTING.md:198) only ever promises curation-pin survival, never Floor/Posture. The real production out-dir (eiaa rerun-2026-08-23/out/playbook.opf.json) currently carries posture {system_prompt, version, generation} and 2 floor invariants and no curation section — a Route C recompile of it would lose everything the GC authored.

**Consequence:** A GC following Route C after a corpus change (steps 1→7 include `project`, then "re-run 7a/7b only if the interview answers changed") ends with a playbook whose human-signed Rung 2 hard lines and Rung 1 Posture are gone. `validate` passes (evidence-only is legal Rung 0), so nothing flags the loss; the consumer (toaster's FloorJudge) silently receives a playbook with no floor to enforce. Recovery is manual: re-run the interview from posture-answers.json (if saved) and re-sign every `floor sign` invariant from memory.

**Fix:** Either make `project_playbook` carry forward the prior playbook's `posture` and `floor` sections exactly as it does `curation` (recomputing identity hashes), or change SKILL.md Route C to say the recompile clears them and mandate re-running 7a/7b (interview from the saved answers file) plus re-signing every floor invariant after every `project`. The code fix is strictly better — the skill's promise matches the obvious right behavior.

> Found independently by three workstreams (pipeline-state #44, control-ladder #77, failure-modes #98). #98 reproduced it live on a scratch copy of the July out-dir: posture interview + 2 signed invariants, then `playbook project` → exit 0, green OK, posture {} floor {}. The two production floor invariants exist only inside playbook.opf.json — nothing else in the out-dir can restore a hand-signed Floor after a re-project.

### [HIGH] SKILL's Feedback re-entry sequence destroys the floor accepts `view apply` just promoted, and never re-renders

*Audit finding(s) #99 · doc-vs-code · failure-modes · CONFIRMED · [issue](https://github.com/contract-opf/playbook-engine/issues/124)*

**What:** The Feedback re-entry command block (SKILL.md:1077-1095) runs `view apply` (which promotes accepted floor candidates into playbook.opf.json's floor.invariants) and then ends with mine → project → validate → report — the project step wipes those just-promoted invariants (finding 1), and the sequence also never re-runs `view render`/`view bundle`/`digest`, so the review HTML, shareable bundle, and digest sidecar all stay stale after the correction round.

**Evidence:** SKILL.md:1084-1094 (sequence ends at report; no view render/bundle/digest); viewer.py apply_feedback promotes into floor.invariants and refreshes identity (cli.py:3166-3173, viewer.py:1617 comment); pipeline.py:3481-3488 + playbook_assembler.py:344-345 then discard it at the sequence's project step.

**Consequence:** A reviewer who accepts hard-line candidates in the checklist, exports feedback, and lets the agent run the documented re-entry loop ends with a playbook that contains none of their accepted invariants and reviewer artifacts (playbook.review.html / playbook.opf.html) that still show the pre-correction state — the next review round annotates stale content.

**Fix:** Reorder the re-entry sequence: run view apply's floor decisions AFTER the recompile (or fix finding 1 so promotion survives), and append `view render`, `view bundle`, and `digest` to the end of the Feedback re-entry block so every artifact is regenerated after corrections.

### [HIGH] Sacred-clauses (Q4) answer is signed into the binding Floor without telling the signer at answer time

*Audit finding(s) #111 · usability · lawyer-usability · CONFIRMED · [issue](https://github.com/contract-opf/playbook-engine/issues/125)*

**What:** The one interview answer that becomes hard-binding — Q4 writes directly into signed `floor.invariants` — is collected by a plain prompt ("Which clause types are non-negotiable regardless of deal value?") that carries no disclosure of that weight, and SKILL.md explicitly forbids a second confirmation ("Do not ask the user to 'confirm' Q4 a second time"). Nothing in the guaranteed user-visible flow says, before the answer is given, that this answer is a signature with fail-closed force that a conformant consumer may never override.

**Evidence:** playbook_engine/cli.py:3258-3263 (interactive loop `click.prompt(iq.question)` with no weight disclosure); playbook_engine/posture.py:110-113 (question text); SKILL.md:876-879 ("written directly into signed floor.invariants... Do not ask the user to 'confirm' Q4 a second time"). The weight IS explained in docs/ADOPTING.md:131-142, but only if the user happens to read that doc.

**Consequence:** A GC answers what looks like a survey question and has thereby executed the only binding act in the whole pipeline — the legal equivalent of signing without being shown the signature block. For a lawyer audience this is an informed-consent failure, and a hard line they'd have phrased more carefully (e.g. conditionally) gets templated into "Do not concede on X." unaware.

**Fix:** Put the weight disclosure into the Q4 question itself, in both surfaces: the CLI prompt/posture.py question text ("— note: this answer is signed into your Floor as a binding hard line") and a SKILL.md instruction that the agent MUST state, in the Q4 option round, that this is the one answer with binding force (which also satisfies the no-second-confirm rule).

### [MEDIUM] Posture governed version counter resets across re-derivation

*Audit finding(s) #78 · doc-vs-reality · control-ladder · CONFIRMED · [issue](https://github.com/contract-opf/playbook-engine/issues/126)*

**What:** Re-running the interview after a re-derivation restarts posture.version at 1, because generate_posture derives the bump only from the (freshly wiped or fresh-out-dir) document's own prior posture. The governed counter OPF §3.6 rule 1 defines ('bumped each re-run of the interview') goes backwards across exactly the event it exists to govern.

**Evidence:** Real artifacts: <corpus>/eiaa-playbook-output/full-run.bak-2026-08-23/out/playbook.opf.json has posture.version=2 (generated_at 2026-08-21), while the newer validated rerun-2026-08-23/out/playbook.opf.json — derived from identical posture-answers.json (verified `a == b` True) — has posture.version=1 (generated_at 2026-08-23). Mechanism: posture.py:258-259 (`prior_version + 1 if isinstance(...) else 1`) reads only `doc.get("posture")` of the out-dir's playbook.

**Consequence:** The §8 lineage chain (corpus → OPF posture vN → installed bundle → edited versions) becomes unorderable: the successor playbook carries a LOWER posture version than the one it supersedes, so a consumer's governance/rollback logic (and any human auditing 'which posture is newer') is misled.

**Fix:** Let `playbook posture interview` accept a prior playbook (or `--base-version N`) so a re-derivation continues the counter; or fold into finding 1's carry-forward so the prior posture (and its version) is present when the interview re-runs. Skill: Route C should state the version behavior and how to preserve it.

### [MEDIUM] Floor 'signature' is a free-text convention no tool can verify and the agent itself can perform

*Audit finding(s) #79 · doc-vs-code · control-ladder · CONFIRMED · [issue](https://github.com/contract-opf/playbook-engine/issues/127)*

**What:** OPF §3.7 says the legal owner 'authors & signs' the Floor, but nothing structural distinguishes a human-signed invariant from one the LLM agent fabricated: the schema allows only id/statement/rationale (+x_*), there is no signer/date field, `playbook floor sign` is a plain CLI command the skill instructs the agent to run, and `validate` checks only duplicate ids and the lexical softening SHOULD-warn. The attribution regexes in floor_candidates.py protect against overwrites, not against invented sign-offs.

**Evidence:** spec/playbook.schema-0.3.json floor.invariants items: required [id, statement], properties id/statement/rationale only (+^x_); validator.py:648-667 (duplicate-id check) and 526-543 (softening warn) are the only floor checks; cli.py:3365-3398 `floor sign` takes any --statement/--rationale from whoever runs it. The real playbook's sign-off ('Hand-authored and signed by the legal owner (Marc Mandel, GC), 2026-08-21') is plain rationale text an agent could emit verbatim.

**Consequence:** The control ladder's Rung 2 guarantee is enforced purely by skill prose (SKILL.md:1113-1121). A prompt-injected or sloppy agent session can put a 'signed' hard line into floor.invariants that forces fail-closed outcomes on every future review at the consumer, with an audit trail indistinguishable from a genuine one.

**Fix:** Add a structural attribution: e.g. `x_signed_by`/`x_signed_at` fields that `floor sign` requires (prompting for the human's name, or refusing when absent), plus a `validate` info line listing each invariant's attribution so a human review surfaces unattributed ones. In the skill, require the agent to show the exact statement to the human and get an explicit in-chat confirmation before running `floor sign`, and to record who signed in --rationale.

### [MEDIUM] Step 7a claims Q4 always self-signs, but the skill's own recommended absence-shaped wording is sentence-shaped and silently skipped

*Audit finding(s) #81 · doc-vs-code · control-ladder · CONFIRMED · [issue](https://github.com/contract-opf/playbook-engine/issues/128)*

**What:** SKILL.md Step 7a states unconditionally that sacred_clauses 'Writes straight into signed floor.invariants' (line 806; again lines 872-879), never mentioning the issue #104 carve-out: a sentence-shaped item (>7 words, or containing ' if/unless/must/shall/provided ') is NOT promoted — only a WARN naming `floor sign` is printed. The skill's own most-emphasized advice ('X must be present at all', line 845; example answer 'limitation of liability must always be present', line 838-839) contains ' must ' and would be skipped.

**Evidence:** floor_candidates.py:432,475-478 (`_Q4_SENTENCE_MARKERS` includes ' must '; `len(item.split()) > 7`); posture.py:550-558 emits the warning; SKILL.md:806/845/872-879 contain no mention of the skip. The real production run confirms the path: its interview has NO sacred_clauses answer and both invariants were recorded via `floor sign` instead.

**Consequence:** A user who picks the absence-shaped option — the one the skill calls 'often the most valuable thing in the whole Floor' — believes it is signed; it is not, unless the agent notices one yellow WARN line and runs the follow-up command. A missed WARN leaves the top hard line absent from the Floor while the report shows a populated posture, masking the gap.

**Fix:** In Step 7a's three-rules list, add a fourth rule: sentence-shaped Q4 items (including every absence-shaped 'must be present' line) do not self-promote — the CLI prints the exact `playbook floor sign` command; always run it and confirm the invariant landed (e.g. jq '.floor.invariants[].id').

### [MEDIUM] Step 7a ranking snippet surfaces unclassified and structural reversals the engine deliberately excludes

*Audit finding(s) #80 · usability · control-ladder · CONFIRMED · [issue](https://github.com/contract-opf/playbook-engine/issues/129)*

**What:** The skill's sacred-clauses option-computation snippet ranks 'most REVERSED' over raw observations without the unclassified/structural filters that `floor propose` applies (issues #106), so its top candidates are segmentation debris and boilerplate.

**Evidence:** Ran the SKILL.md:769-798 snippet verbatim against <corpus>/eiaa-playbook-output/rerun-2026-08-23/out: `most REVERSED top3: [(None, 65), ('Parties & Recitals', 48), ('Supervision & Evaluation', 34)]` — None (unclassified) and Parties & Recitals (structural) lead. Contrast floor_candidates.py:276-286, which excludes exactly these ('neither is a proposable hard line'), and its docstring's measurement that these two classes were 'three-quarters of the review checklist... noise'.

**Consequence:** The interview options seeded from this ranking — the input to the ONLY hard-binding answer — lead with non-positions unless the LLM independently knows to skip them; a faithful executor would offer 'Parties & Recitals' as a candidate sacred clause to the GC.

**Fix:** In the snippet, filter `rev` to observations with a truthy taxonomy_id and drop taxonomy entries curated `structural: true` (readable from the config's taxonomy YAML), mirroring _group_reversal_observations; add one sentence telling the agent why.

### [LOW] `playbook curate` is absent from the skill despite its 'update my playbook' trigger and Route C's reliance on pins

*Audit finding(s) #38 · consumer · cli-audit · CONFIRMED · [issue](https://github.com/contract-opf/playbook-engine/issues/130)*

**What:** The skill's description claims coverage of updating an existing playbook, and Route C leans on 'curation pins ... survive a recompile by design', but the command that creates pins outside the HTML round-trip — `playbook curate` with its pin/note grammar and per-run pin-conflict refresh — is never named in SKILL.md or REFERENCE.md.

**Evidence:** `playbook curate --help` (pin/note grammar, --by attribution, conflict refresh); grep 'curate' over .claude/skills/playbook-from-corpus/ returns only an unrelated word at REFERENCE.md:343; SKILL.md:90 (Route C pins claim).

**Consequence:** An agent asked to 'pin governing_law to usually_conceded' has no documented path and may hand-edit playbook.opf.json — exactly what the guardrails forbid for floor and should discourage generally.

**Fix:** Add a short 'Curation (Rung 3)' subsection to SKILL.md naming `playbook curate --command 'pin ... to ...' --by <attorney>` and `view apply` overrides as the two sanctioned pin paths.

### [LOW] SKILL never states the hard minimum of 3 interview answers that the engine enforces

*Audit finding(s) #13 · doc-vs-code · skill-internal · CONFIRMED · [issue](https://github.com/contract-opf/playbook-engine/issues/131)*

**What:** SKILL says "Three answers is enough. ... Skipping questions is legitimate" but never says fewer than three is a hard CLI error; the engine rejects an interview with <3 answers.

**Evidence:** SKILL.md:869-871 vs playbook_engine/posture.py:128 (`_MIN_ANSWERS = 3`) and posture.py:244-247 (raises when `len(answered) < _MIN_ANSWERS`).

**Consequence:** A GC who answers only sacred_clauses + risk_appetite (perfectly plausible under 'skipping is legitimate') gets an error at apply time after the interview effort was spent, and the agent has no documented reason to have pushed for a third answer.

**Fix:** Change the rule to: "Three answers is enough — and also the enforced minimum: fewer than 3 is rejected by `posture interview` (OPF §7); make sure at least three questions get answers."

### [LOW] posture interview bumps posture.version on a byte-identical re-run

*Audit finding(s) #108 · producer · failure-modes · CONFIRMED · [issue](https://github.com/contract-opf/playbook-engine/issues/132)*

**What:** Re-running `posture interview` with an unchanged answers file bumps posture.version (1→2) even though nothing changed, so the governance version number signals a revision that never happened. The Q4 floor invariant is correctly idempotent, but the version counter is not.

**Evidence:** Scratch test: same answers file applied twice → "OK posture.version=1" then "OK posture.version=2", floor.invariants unchanged (one entry). SKILL.md:882-884 says "Re-running is safe. The same statement always resolves to the same id and updates in place".

**Consequence:** Consumers or auditors diffing posture.version across copies infer a posture change that never occurred; repeated idempotent re-runs (which the skill encourages offering) inflate the version.

**Fix:** In apply_posture_interview, compare the newly assembled posture content against the existing block and skip the version bump (no-op message) when identical — mirroring floor sign's no-op behavior.

### [LOW] Spec's 'interview MUST be retained' has no validator or schema enforcement

*Audit finding(s) #85 · doc-vs-code · control-ladder · CONFIRMED · [issue](https://github.com/contract-opf/playbook-engine/issues/133)*

**What:** OPF §3.6 rule 2 makes retaining posture.generation.interview a MUST, but schema-0.3 marks `generation` (and `interview` within it) optional with no conditional requirement when system_prompt is present, and validator.py adds no check — a posture with prose and no provenance validates clean.

**Evidence:** spec/playbook.schema-0.3.json posture: no `required` on the posture object or generation; validator.py's only posture check is _check_posture_floor_conflict_v2 (lines 526-543, 780). OPF-SPEC.md §3.6 rule 2: 'It MUST be retained.'

**Consequence:** A hand-edited or third-party playbook can strip the interview record — the auditability §3.6 rule 2 exists for — and still pass `playbook validate`, so the skill's Step 8 gate cannot catch it.

**Fix:** Add a non-blocking (or blocking, per spec-owner choice) validator check: posture.system_prompt present but generation.interview absent/empty → warn citing §3.6 rule 2.

### [LOW] Q5 auto-rejection is invisible in the skill

*Audit finding(s) #86 · usability · control-ladder · CONFIRMED · [issue](https://github.com/contract-opf/playbook-engine/issues/134)*

**What:** flexible_clauses is described in Step 7a as 'binds nothing' (SKILL.md:810), but a Q5 answer auto-marks matching reversal candidates `decision: rejected` in floor.candidates.json (issue #105), and Step 7b never mentions that reviewers will see pre-rejected 'Recommended reject' rows attributed to their own interview answer.

**Evidence:** floor_candidates.py:600-618, 806-814 (_Q5_REJECTION_COMMENT, slug match, pre-marking); grep of SKILL.md finds no mention of Q5 auto-rejection or the recommended-reject rendering in Step 7b (SKILL.md:890-947).

**Consequence:** A GC reviewing playbook.review.html finds candidates already marked rejected with no explanation from the skill-driven walkthrough; 'binds nothing' is technically true for floor.invariants but understates a real side effect on the review artifact.

**Fix:** One sentence in Step 7b: 'Candidates whose clause type you named as flexible in the interview arrive pre-marked Recommended reject — you can still accept them; an explicit decision clears the recommendation.'

### [LOW] Posture prose is templated labels, not the evidence-grounded draft the spec depicts

*Audit finding(s) #87 · doc-vs-code · control-ladder · CONFIRMED · [issue](https://github.com/contract-opf/playbook-engine/issues/135)*

**What:** OPF §3.6/§7 depict a compiler that 'drafts posture.system_prompt grounded in the Evidence' (the spec's example is fluent strategic prose); the engine deterministically concatenates 'Label: answer.' sentences and records grounded_in as a digest pointer only — a deliberate, documented slice decision (#116/#156), but the spec text was never softened to match.

**Evidence:** posture.py:149-156 (_PROSE_TEMPLATES 'Rounds before escalating: {answer}') and module docstring lines 5-7 ('the prose is templated/assembled from the interview answers, not model-written') vs OPF-SPEC.md §3.6 example system_prompt and §7 'drafts posture.system_prompt grounded in the Evidence'.

**Consequence:** A consumer or auditor comparing spec to artifact sees a genesis Posture that does not resemble what §3.6 promises; the 'grounded_in' digest implies evidence grounding that never influenced a word of the prose.

**Fix:** Add a sentence to §3.6/§7 noting the reference producer assembles the genesis Posture deterministically from answers (grounding is via the recorded interview options, which the skill computes from Evidence), or have the skill/LLM offer a reviewed, richer draft through the governed edit path.


## Confidentiality & pseudonymization

Born-safe pseudonymization is load-bearing for a public-repo project run over real agreements — and it currently leaks along several independent paths.

### [HIGH] Canonical 'shareable' artifacts carry hundreds of real counterparty names despite born-safe pseudonymization

*Audit finding(s) #57 · doc-vs-reality · corpus-reality · CONFIRMED · [issue](https://github.com/contract-opf/playbook-engine/issues/136)*

**What:** SKILL.md (lines 996-1004) calls playbook.opf.html 'THE shareable/uploadable playbook' that 'embeds the canonical, pseudonymized JSON', but in the validated rerun the canonical playbook.opf.json, playbook.opf.html, and the model-facing playbook.digest.json all retain real counterparty names at scale — pseudonymization only replaced entities whose config known_entities entry matched the document text verbatim, and roughly a dozen entries were written in a stopword-stripped form (e.g. 'University <City>' instead of 'University of <City>') that the contiguous word-sequence matcher in pseudonymize_text (playbook_engine/entity_registry.py:195, pattern builder at :150-194) can never match.

**Evidence:** In <corpus>/eiaa-playbook-output/rerun-2026-08-23/out/: distinctive single-token greps of 7 counterparty names return 139-642 occurrences each in playbook.opf.json (e.g. one midwestern-university token: 642; one Ohio-university token: 516) and 1-54 in playbook.digest.json; a real signatory's surname appears 46x in the canonical JSON and 48x in the 'shareable' HTML. Running the engine's own publisher._entity_backstop_scan over this playbook with the run's entity_registry.json returns 65 hits across 14 distinct known-entity names — and that scan itself under-counts because the registry's stopword-stripped canonical names can't match the 'University of X' forms in text. Config at <corpus>/eiaa-playbook-output/staging-full-materialized/playbook.config.yaml lists the stripped forms under provenance.known_entities.

**Consequence:** A GC who follows Step 10 and hands playbook.opf.html to a stakeholder or uploads it to the toaster believes it is pseudonymized ('resolving real names would leak them') while it actually discloses which institutions negotiated, what they asked for, and a signatory's name — the exact incident class already hit once (eiaa-example de-id, 2026-07-22).

**Fix:** In SKILL.md: (a) require known_entities entries to be verbatim recital spellings and add a mandatory post-mine residue check of the canonical artifacts (the publisher backstop scan, or a simple distinctive-token grep against alias_map values) before calling anything shareable; (b) soften/correct the Step 10 claim that the canonical JSON is pseudonymized. In the engine: warn at mine time when a known_entities entry matches zero document text (the warning that already exists for our_party_aliases at SKILL.md:140).

### [HIGH] Production public artifact fails the engine's own 'non-negotiable' publish backstop

*Audit finding(s) #58 · doc-vs-reality · corpus-reality · CONFIRMED · [issue](https://github.com/contract-opf/playbook-engine/issues/137)*

**What:** SKILL.md Step 11 (lines 1035-1038) states the publish hard backstop 'fails loud and writes nothing' if any known entity name survives — 'Non-negotiable.' The public export sitting in the production out-dir contains registry-listed real names, including one counterparty's full legal name with its Office of General Counsel street address, and pseudonym-breaking juxtapositions that map aliases to real identities in the same sentence.

**Evidence:** Running current publisher._entity_backstop_scan over full-run.bak-2026-08-23/out/playbook.public.opf.json with that run's own entity_registry.json returns 10 hits, all one registry name (one public university system; its full legal name together with its Office of General Counsel street address survives in the file — 5 grep occurrences). Grep also shows two pseudonym-breaking juxtapositions where an alias (e.g. 'Counterparty-12', 'Counterparty-30') sits in the same sentence as the real institution's name. residue_report.json in the same dir flags these tokens, but redact_terms.txt (110 terms) contains none of them.

**Consequence:** If this on-disk playbook.public.opf.json is ever treated as release-ready (it is the file Step 11 produces for sign-off), real counterparty identities ship, and the alias-to-real-name juxtapositions defeat the pseudonymization of every other mention of those parties.

**Fix:** Regenerate the public export with the current engine (the backstop now catches it and should refuse); delete or clearly quarantine the stale playbook.public.opf.json; add a SKILL.md note that public exports produced before a backstop/registry change must be re-gated, and that the residue-report UNKNOWN buckets flagged in residue_report.json must land in redact_terms.txt before sign-off.

### [HIGH] Step 11 publish command omits --entity-registry, so the documented run hard-fails (and its 'hard backstop' claim is misleading)

*Audit finding(s) #32 · doc-vs-code · cli-audit · CONFIRMED · [issue](https://github.com/contract-opf/playbook-engine/issues/138)*

**What:** SKILL.md Step 11's Docker publish command passes no --entity-registry. Inside the container the default registry path (~/.cache/playbook-engine/entity_registry.json) is empty, and publish hard-fails on an empty registry unless --allow-empty-registry is given — yet the skill asserts 'Two safety layers run automatically', including the known-entity backstop, for this exact command. The skill itself pinned the registry to $OUT in Step 3 but never threads it back into publish.

**Evidence:** SKILL.md:1030-1039 (command with no --entity-registry; 'Hard backstop' claim) vs playbook_engine/cli.py:862-872 (registry defaults to DEFAULT_REGISTRY_PATH; empty registry without --allow-empty-registry → hard fail) and `playbook publish --help`: 'Defaults to the machine-global ~/.cache/playbook-engine/entity_registry.json — which inside a container is EMPTY, disabling the step-4 backstop... Always pass the run's sidecar here.' SKILL.md:527-528 shows mine was run with --entity-registry /work/out/entity_registry.json.

**Consequence:** The release-gating step fails on first invocation; worse, the natural 'fix' an agent might reach for from the error text is --allow-empty-registry, which silently disables the exact backstop the skill promises the GC is protecting the publication.

**Fix:** Add `--entity-registry /work/out/entity_registry.json` to the Step 11 command, and add a warning that --allow-empty-registry must never be used for a corpus-derived playbook.

### [MEDIUM] normalized/ is written pre-pseudonymization under raw counterparty names, never pruned, and not rewritten on cache hits — contradicting the skill's sensitive-artifact claim and breaking aliased joins

*Audit finding(s) #48 · doc-vs-reality · pipeline-state · CONFIRMED · [issue](https://github.com/contract-opf/playbook-engine/issues/139)*

**What:** Clause trees are written to `out/normalized/<raw_doc_id>/` inside the per-document loop, BEFORE the born-safe pseudonymization pass renames everything else; unlike `trail/` (which is stale-cleared and alias-renamed every run), normalized/ keeps raw counterparty names forever and accumulates stale directories. The skill claims `$OUT/alias_map.json` + `$OUT/entity_registry.json` are "the only durable sensitive artifacts" under $OUT. Additionally, consumers that join on the aliased document_id (viewer sourcing full text from `out/normalized/<cdoc>`) miss the raw-named directory.

**Evidence:** pipeline.py:1859 (`tree.write(out_dir / "normalized" / doc_id / ...)` — raw id, mid-loop), pipeline.py:3205-3216 (stale-clear + alias applied to trail/ only); verified in the real run: eiaa rerun-2026-08-23/out/normalized/ subdirectory names embed real university counterparty names while trail/ files are `affiliation-agreement-counterparty-1-….json` etc.; viewer.py:1437 (`norm_dir = out_dir / "normalized" / cdoc`); SKILL.md:231-238 ("$OUT/alias_map.json + $OUT/entity_registry.json are the only durable sensitive artifacts").

**Consequence:** A user who trusts the skill's claim and shares an out-dir minus the two named files leaks every counterparty name (directory names AND raw clause text inside the trees). Separately, viewer features that read normalized/ by aliased id silently degrade for every pseudonymized document, and renamed/removed corpus documents leave phantom trees indefinitely.

**Fix:** Mirror the trail/ treatment for normalized/: stale-clear each run, write (or rename) under the aliased doc_id after the pseudonymization pass, and rewrite on cache hits or record trees in the stage cache. Until then, correct SKILL.md's sensitive-artifact list to include normalized/ (and raw text in my-verdicts files).

### [MEDIUM] judge rounds write the sensitive alias→real-name entity registry to the machine-global default; no --entity-registry flag exists on judge

*Audit finding(s) #100 · doc-vs-code · failure-modes · CONFIRMED · [issue](https://github.com/contract-opf/playbook-engine/issues/140)*

**What:** SKILL.md (231-238) says passing --entity-registry to `mine` makes "$OUT/alias_map.json + $OUT/entity_registry.json ... the only durable sensitive artifacts — no post-run 'purge the global registry?' step needed". But `playbook judge` (the command run repeatedly in the drain loop, doing the identical mine_corpus work) accepts no --entity-registry option and calls mine_corpus without entity_registry_path, so every judge round on the venv path loads/writes DEFAULT_REGISTRY_PATH = ~/.cache/playbook-engine/entity_registry.json. The skill blesses venv as "equally valid" (line 154-155).

**Evidence:** cli.py:1817-1826 judge_cmd signature (no entity_registry param) and its mine_corpus call at cli.py:2044-2062 (no entity_registry_path kwarg); pipeline.py:3125 `EntityRegistry.load(entity_registry_path or DEFAULT_REGISTRY_PATH)`; entity_registry.py:46. Confirmed on this machine: ~/.cache/playbook-engine/entity_registry.json exists (8,418 bytes, mtime Jul 27) despite the skill's $OUT-only guidance being followed for mine.

**Consequence:** Real counterparty names durably land outside the gitignored $OUT on the operator's machine, contradicting the skill's confidentiality claim; additionally, mine (out-dir registry) and judge (global registry) can assign different aliases for the same entities across alternating runs against the same $OUT, so alias labels in inspection/report artifacts can silently refer to different real entities between rounds.

**Fix:** Add --entity-registry to judge_cmd (mirroring mine) and pass it through to mine_corpus in both plan and normal branches; update SKILL.md Step 6 to include the flag in the drain-loop command lines; optionally have `doctor` report a non-empty machine-global registry.

### [MEDIUM] QUICK-COMPILE invites GCs to paste full error output into a public issue, against the project's own confidentiality rule

*Audit finding(s) #112 · doc-vs-doc · lawyer-usability · CONFIRMED · [issue](https://github.com/contract-opf/playbook-engine/issues/141)*

**What:** QUICK-COMPILE tells the user, for validation errors, to "Open an issue with the full error output" — with no caution that error output from a run over real agreements may carry clause text or counterparty names. The project's own onboarding prompt sets the opposite standing rule: "never quote them into GitHub issues or anything that leaves this machine", and run_manifest deliberately emits a "no corpus content" copy-pasteable block precisely because the repo cares about this.

**Evidence:** docs/QUICK-COMPILE.md:131 ("Open an issue with the full error output.") vs docs/prompts/create-playbook.md:9-11 ("never quote them into GitHub issues"); SKILL.md:612-614 (run manifest prints environment facts "(no corpus content)" for exactly this send-to-maintainers case).

**Consequence:** A naive GC hits a validation failure on their real corpus and pastes the error — potentially containing negotiated clause text or party names — into a public GitHub issue on this public repo. This is the same leak class as the 2026-07-22 de-id incident.

**Fix:** Amend QUICK-COMPILE.md:131 to: "Open an issue with the error output after removing any agreement text or party names — validation errors can quote your documents" (or point at the run-manifest environment block as the safe thing to paste).

### [MEDIUM] Step 11 never mentions --redact-terms or publish --config, the flags built for its own residue workflow

*Audit finding(s) #37 · consumer · cli-audit · CONFIRMED · [issue](https://github.com/contract-opf/playbook-engine/issues/142)*

**What:** The skill's residue procedure says the only remediation for surviving names is 'fix the entity extraction... re-mine, and re-run publish', but `publish --redact-terms FILE` exists precisely for GC residue-review output (signatory names, institution fragments) the registry cannot know, joining the hard backstop without a re-mine. Likewise `publish --config` (restoring an affiliation-flavored corpus's scan leniency) is never mentioned, though the primary real corpus is exactly that domain.

**Evidence:** SKILL.md:1056-1060 (re-mine as the only fix) vs `playbook publish --help` --redact-terms ('the GC's residue-review output ... that the entity registry did not know') and --config ('pass the config an affiliation ... corpus was mined with to restore its prior scan leniency').

**Consequence:** The agent drives the user through multi-hour re-mine cycles for a signatory surname a one-line redact file would handle, and on affiliation corpora the neutral-vocabulary sweep over-flags benign institutional boilerplate.

**Fix:** Add both flags to Step 11: --config $CORPUS config for domain corpora, and --redact-terms as the sanctioned remediation for human-identified residue that isn't a counterparty-entity extraction failure.

### [LOW] Version strings are never pseudonymized, undoing document-id aliasing in every report row

*Audit finding(s) #63 · producer · corpus-reality · CONFIRMED · [issue](https://github.com/contract-opf/playbook-engine/issues/143)*

**What:** Document ids get aliased (pseudonymize_document_id) but the per-version labels keep the raw source filenames, which carry the real counterparty names — so manifest, report, and inspection rows print the alias and the real identity side by side.

**Evidence:** rerun-2026-08-23/out/report.json needs_attention item 1: an aliased document_id ('affiliation-agreement-counterparty-2-…') printed beside a version label of the form '02__<Real-University-Name>-2024-01-23'; same pattern in inspection.md and corpus_manifest.json version_ingest. pseudonymize_document_id (playbook_engine/entity_registry.py:213) handles only the doc slug; no code path touches version labels.

**Consequence:** Every artifact that lists versions leaks the real name AND maps it to its alias, so a single leaked report row de-pseudonymizes that counterparty everywhere.

**Fix:** Run the same slug-token pseudonymization over version labels at ingest, or derive version labels from ordinal + date only.


## Commands that fail or destroy state as written

The skill is executed literally by an LLM; each of these is a documented command that errors out or silently destroys user work.

### [HIGH] Documented Docker stage command crashes: engine rmtree's the bind-mount root

*Audit finding(s) #30 · doc-vs-code · cli-audit · CONFIRMED · [issue](https://github.com/contract-opf/playbook-engine/issues/144)*

**What:** SKILL.md Step 1 and Step 1a tell the agent to run `make docker-run ... ARGS="stage /work/corpus --out /work/out"` (and the --from-plan variant), but `playbook stage` wipes-and-recreates its --out directory via shutil.rmtree, and /work/out is a bind-mount root that cannot be removed from inside the container — the command dies with an unhandled OSError traceback.

**Evidence:** SKILL.md:307-309 and 433-436 (the commands); playbook_engine/staging.py:273 (`shutil.rmtree(out_dir)` in _recreate_out_dir, called from stage() at staging.py:368 and from execute_staging_plan). Reproduced live with a scratch corpus: `docker run --rm -i -v .../fake-corpus:/work/corpus:ro -v .../stage-out:/work/out playbook-engine stage /work/corpus --out /work/out` → `OSError: [Errno 16] Device or resource busy: PosixPath('/work/out')`. Additionally, the Step 1 host mount is OUT=~/.cache/playbook-engine/staging — the SHARED staging root; even where rmtree could succeed, _recreate_out_dir (staging.py:267-271) would refuse it as a non-empty directory lacking the .playbook-staging marker (the marker lives in per-corpus subdirs, not the root).

**Consequence:** An LLM following the skill literally on Route A with a nested corpus hits a raw traceback at the very first pipeline step, in the runtime the skill itself declares primary ('Docker is the engine runtime'). A GC user sees a Python stack dump instead of staging. Staging only works today because operators drop to the venv form.

**Fix:** In SKILL.md, run stage on the host venv (staging needs no docker tooling — it's file copying), OR mount a parent directory and stage into a subpath (`--out /work/out/staged`). Engine-side: _recreate_out_dir could clear the directory's contents instead of removing the directory itself when out_dir is a mount point (or catch EBUSY and fall back to per-entry deletion).

### [HIGH] Step 7b runs `playbook view apply $OUT` without the required FEEDBACK_FILE argument

*Audit finding(s) #0, #31 · doc-vs-code · skill-internal, cli-audit · CONFIRMED · [issue](https://github.com/contract-opf/playbook-engine/issues/145)*

**What:** SKILL.md Step 7b instructs `playbook view apply $OUT` (one argument), but the CLI defines feedback_file as a second REQUIRED positional argument, so the documented command always fails with a usage error. The same file shows the correct two-argument form later.

**Evidence:** SKILL.md:915 (`playbook view apply $OUT`) vs playbook_engine/cli.py:3122-3127 (`@click.argument("out_dir"...) @click.argument("feedback_file", type=click.Path(exists=True...))` — no default) vs SKILL.md:1077 (`view apply /work/out /work/out/feedback.json`).

**Consequence:** An LLM literally executing Step 7b (the Floor accept/reject round-trip — the heart of Rung 2) hits a hard CLI error at the exact moment the human just finished reviewing; the likely improvised recovery is hand-editing floor.invariants, which the same page forbids.

**Fix:** Change SKILL.md:917 to `playbook view apply $OUT $OUT/feedback.json` (matching line 1077).

> Found independently by skill-internal (#0) and cli-audit (#31). The correct two-argument form appears later in the same file (SKILL.md:1077).

### [HIGH] Re-running `stage` rmtrees the staged corpus, destroying the config, template, and hand-edited hints.yaml the skill told the user to put there

*Audit finding(s) #45, #105 · doc-vs-code · pipeline-state, failure-modes · CONFIRMED · [issue](https://github.com/contract-opf/playbook-engine/issues/146)*

**What:** `playbook stage` (both auto and --from-plan paths) wipes and recreates the destination via `_recreate_out_dir` whenever the `.playbook-staging` marker is present — which it always is for a previously staged corpus. The skill instructs users to copy `playbook.config.yaml` and the baseline template INTO the staged dir and to hand-edit per-agreement `hints.yaml` there, then presents re-staging as a routine step (Route C: "1 → 7"), never warning that a re-stage deletes all of that.

**Evidence:** playbook_engine/staging.py:267-275 (`shutil.rmtree(out_dir)` when marker present); cli.py:1735-1736 (auto path) and intake_plan.py:723 (from-plan path) both call `_recreate_out_dir`; SKILL.md:319-321 ("copy the baseline template and playbook.config.yaml into it too"), SKILL.md:585-587 ("add hints.yaml files to the relevant corpus subfolders ... then re-run playbook mine"), Route C row SKILL.md:90 (steps 1→7 begin with Step 1 — Stage).

**Consequence:** A user re-deriving after new agreements arrive re-runs `stage` per Route C and loses: the tuned config (party aliases, known_entities, template path), the template file, and every hints.yaml trail/provenance correction accumulated through Step 4 and feedback rounds — hints are regenerated from scratch by staging heuristics. The loss is silent (staging prints green OK) and only surfaces as regressed trail ordering deep in the next run.

**Fix:** In SKILL.md, add an explicit warning to Step 1 and Route C: re-staging wipes the staged tree — back up playbook.config.yaml, the template, and all hints.yaml before re-staging (or stage into a fresh dir and port them over). Engine-side, `stage` could preserve/merge existing hints.yaml and playbook.config.yaml on re-stage, or at minimum print what it is about to delete.

> Found independently by pipeline-state (#45) and failure-modes (#105).

### [MEDIUM] Route B promises 'no corpus and no config', but Route B's own Step 7b passes --config and silently loses structural exclusion without it

*Audit finding(s) #35, #6, #82 · doc-vs-doc · cli-audit, skill-internal, control-ladder · CONFIRMED · [issue](https://github.com/contract-opf/playbook-engine/issues/147)*

**What:** SKILL.md Step 0 states 'Route B needs no corpus and no config — every command in Steps 7a/7b reads and writes the single out-dir', yet Step 7b's command is `playbook floor propose $OUT --config $CORPUS/playbook.config.yaml` and the adjacent comment warns that omitting --config 'silently turns off' structural exclusion (issue #106). A true Route B user (out-dir only) cannot follow both instructions.

**Evidence:** SKILL.md:92-94 vs SKILL.md:896-901; `playbook floor propose --help` confirms --config is optional but supplies the structural-exclusion taxonomy.

**Consequence:** An agent on Route B either blocks asking for a config the user was told they don't need, or omits it and produces floor candidates polluted with structural clauses (e.g. Parties & Recitals) — the failure mode issue #106 was filed about.

**Fix:** In Route B guidance, say: pass --config when a config is available; when the user genuinely has only an out-dir, omit it but tell the user structural entries may appear as candidates and should be rejected in review.

> Found independently by cli-audit (#35), skill-internal (#6), and control-ladder (#82); `floor sign --clause` additionally hard-requires --config (cli.py:3437-3443).

### [LOW] Route B's own step commands still use `make docker-run CORPUS=./corpus`, the exact form the Route B caveat says will fail

*Audit finding(s) #9 · doc-vs-doc · skill-internal · CONFIRMED · [issue](https://github.com/contract-opf/playbook-engine/issues/148)*

**What:** The Route B caveat (no corpus dir → mount error; use venv or CORPUS=$OUT) is contradicted by the commands Route B users are routed to: Step 7a's docker form omits CORPUS entirely (falling back to the Makefile's ./corpus default) and Steps 8-10 all hardcode `CORPUS=./corpus`.

**Evidence:** SKILL.md:96-100 (caveat) vs SKILL.md:864 (`make docker-run OUT=./out ARGS="posture interview ..."` — no CORPUS override) and SKILL.md:952, 965-966, 985-987 (Steps 8-10 all `CORPUS=./corpus`); Makefile:8 (`CORPUS := $(CURDIR)/corpus`).

**Consequence:** A Route B user on the Docker runtime copy-pastes Step 8's validate and hits the mount failure the skill predicted three hundred lines earlier, with the fix documented only back in Step 0.

**Fix:** Show `CORPUS=$OUT` in the docker forms of every command in Route B's step list (7a, 7b, 8, 9, 10), or add a one-line reminder at Step 8.

### [LOW] SKILL guardrail states the wrong stage default path

*Audit finding(s) #41 · doc-vs-code · cli-audit · CONFIRMED · [issue](https://github.com/contract-opf/playbook-engine/issues/149)*

**What:** The guardrails say `playbook stage` defaults to `~/.cache/playbook-engine/staging`, but the actual default is a per-corpus subdirectory `~/.cache/playbook-engine/staging/<src_dir_name>` — and the distinction matters because Step 1's Docker command mounts the parent root as OUT, feeding the wipe-refusal/rmtree problem in the first finding.

**Evidence:** SKILL.md:1105-1106 vs `playbook stage --help` ('default: ~/.cache/playbook-engine/staging/<src_dir_name>') and cli.py stage_cmd (`DEFAULT_STAGING_ROOT / resolved.name`).

**Consequence:** Minor on its own, but it normalizes pointing --out at the shared staging root, which the engine refuses (or, in Docker, crashes on).

**Fix:** Correct the path in the guardrail to `~/.cache/playbook-engine/staging/<corpus-name>`.


## Judge loop: correctness

The drain loop mostly works — the verdict store replayed 1,444/1,444 inherited verdicts correctly — but its edges are sharp.

### [HIGH] needs_review is write-only: 254 flagged verdicts silently vanish; the promised after-action listing does not exist

*Audit finding(s) #17, #69 · doc-vs-code · judge-prompts, judge-economics · CONFIRMED · [issue](https://github.com/contract-opf/playbook-engine/issues/150)*

**What:** REFERENCE.md instructs the judge to set `needs_review: true` in classify (<0.6), deviation (<0.65) and provenance (<0.7) verdicts, and Guardrail 2 promises 'The after-action report will list these for human follow-up' — but no engine code reads that boolean: ClauseClassification, DeviationResult and ProvenanceResult have no needs_review field, replay reconstruction discards it, and aar.py builds Needs Attention only from deviation=='needs_review' sentinels (which producers are forbidden to emit), taxonomy_id=='needs_review', and obs confidence < 0.5.

**Evidence:** REFERENCE.md:40,109,157,199-206,337-339 vs clause_classifier.py:126-166, deviation_classifier.py:107-147, provenance_detector.py:128-157 (no needs_review field); agent_judge.py:571-578,704-715 (replay drops it); aar.py:687-709 (reason computation). Production proof: rerun-2026-08-23/out/judge/verdicts.jsonl contains 254 verdicts with needs_review:true (252 deviation, conf 0.4–0.62), while report.json's needs_attention reasons breakdown is exactly {1925 'low confidence (0.45)', 310 'low confidence (0.00)', 3 'version ingest failed'} — zero of the 254 flagged doubts surface.

**Consequence:** The GC follows the skill exactly, flags 254 genuinely doubtful deviation calls for human follow-up, and every one of those doubts silently vanishes: the playbook ships with judge-uncertain risk assessments recorded as clean basis='judge' verdicts and no human ever reviews them.

**Fix:** Either surface the flag (have aar.py/report join observations back to judge/verdicts.jsonl and list needs_review:true verdicts under Needs Attention, or propagate the flag through DeviationResult into observations), or rewrite REFERENCE.md's rules/Guardrail 2 to stop promising report surfacing and instead tell the judge the only channel the report reads (deviation-verdict confidence never reaches observations at all, so today there is no channel).

> Found independently by judge-prompts (#17) and judge-economics (#69). The engine even forbids the one sentinel the report DOES count (agent_judge.py:423-427 directs the judge to the boolean nothing reads). Related: REFERENCE's 0.6/0.65 thresholds map to no engine mechanism (#27).

### [HIGH] Classify pending payload offers inactive taxonomy ids as 'allowed'; picking one passes judge-apply then crashes the next run

*Audit finding(s) #18 · producer · judge-prompts · CONFIRMED · [issue](https://github.com/contract-opf/playbook-engine/issues/151)*

**What:** StoreBackedClassificationJudge builds the pending payload's taxonomy_ids from ALL taxonomy entries (sorted(e.id for e in taxonomy.entries)), including status:inactive ones, while REFERENCE.md calls this the 'flat list of allowed ids' and orders 'Never invent a taxonomy_id not in the provided list'. A verdict naming an inactive id passes validate_verdict at judge-apply (no eligibility check) and then raises ValueError inside classify_tree on replay — outside both the judge-call try/except and mine_corpus's quarantine (which catches only SegmentationQAError/NormalizeTrailError/HintsError), so playbook judge/mine aborts with a raw traceback.

**Evidence:** agent_judge.py:538 (all entries) vs cli.py:2572 segment path correctly using taxonomy.classifier_entries(); clause_classifier.py:343-350 (raise); cli.py:2988 (quarantine catch list); reproduced: validate_verdict('classify', {'taxonomy_id':'beta',...}) ACCEPTED, then classify_tree replay RAISED ValueError "Judge returned taxonomy_id='beta' which is not an active/custom entry". The production eiaa taxonomy (staging-full-materialized/affiliation-agreement.yaml) has 10 inactive entries; REFERENCE.md:20,41.

**Consequence:** A judge that follows REFERENCE to the letter can bank a verdict that bricks every subsequent judge/mine run on that out-dir; a GC (non-engineer) sees an unhandled Python traceback with no pointer to the offending verdict line and no way to recover short of hand-editing verdicts.jsonl.

**Fix:** Build the classify payload's taxonomy_ids from taxonomy.classifier_entries() (matching segment_cmd), and/or make validate_verdict reject inactive ids when the pending payload is available at apply time; note in REFERENCE that the list may contain retired entries until fixed. (Cache-key note: filtering changes classify payload keys, so gate it as a rubric/format event.)

### [HIGH] `judge --plan-only` incurs the extraction and LLM-segmentation spend it claims to be estimating

*Audit finding(s) #47, #109 · doc-vs-code · pipeline-state, failure-modes · CONFIRMED · [issue](https://github.com/contract-opf/playbook-engine/issues/152)*

**What:** SKILL.md Step 5 presents `judge --plan-only` as the pre-commit gate ("count pending items and estimate token cost before committing to a full-corpus pass"; the segmentation line "must be part of the go/no-go decision"). In code, plan mode runs the full `mine_corpus` with the LIVE segment functions wired: on a cold cache it performs whole-corpus extraction/OCR and real per-version LLM segmentation calls (sync or Message Batches) during the plan run itself. The `Segmentation: N version(s) not yet cached` line is printed from counters incremented inside the closures as each real API call is made — i.e., after the spend has already happened.

**Evidence:** cli.py:1930-1971 (plan branch calls mine_corpus with **seg_kwargs into a temp out-dir but real caches), cli.py:191-208 and 234-250 (stats incremented at actual call time inside `_llm_segment_fn` / `_segment_documents_batch_fn`), cli.py:309-321 (`_echo_segmentation_cost_line` prints post-hoc); SKILL.md:649-663.

**Consequence:** On the llm-segmentation path, the GC who asked for a cost estimate has already paid the single largest cost of the run (and, if `segmentation.cache` is false in the config, paid it without even banking the result — the identical spend recurs on the real run). The go/no-go decision the skill mandates is fictional for segmentation; only the judgment-token estimate is genuinely predictive.

**Fix:** Skill: state plainly that `--plan-only` performs (and caches) extraction + segmentation, and that only the judge-item token estimate is a forecast — so the estimator script plus the extraction-cache check is the real pre-spend gate; require `segmentation.cache: true` before any plan run. Engine: a true dry-run would count uncached versions without invoking the segment fns (the cache-probe logic in `_segment_documents_batch_fn` already shows how) and reword the line to distinguish 'were segmented this run' from 'would be segmented'.

> Found independently by pipeline-state (#47) and failure-modes (#109).

### [MEDIUM] REFERENCE's wrong-basis warning describes pre-#182 behavior that judge-apply now prevents

*Audit finding(s) #20 · doc-vs-code · judge-prompts · CONFIRMED · [issue](https://github.com/contract-opf/playbook-engine/issues/153)*

**What:** REFERENCE.md warns that using the wrong basis 'doesn't raise where you'd notice it: the verdict hits the store, fails reconstruction … and is silently re-queued … which then fails schema validation much later, at project/validate'. That chain is stale: validate_verdict now rejects wrong bases at judge-apply with a line number (deviation restricted to {'judge'}, classify to {'judge','unclassified'}, provenance constructs ProvenanceResult which rejects 'judge'), and even the residual re-queue path is loudly warned about by judge_cmd, not silent.

**Evidence:** REFERENCE.md:180-186 vs agent_judge.py:379-472 (_DEVIATION_REPLAYABLE_BASES/_CLASSIFY_REPLAYABLE_BASES + validate_verdict) and cli.py:2260-2264 (per-line rejection), cli.py:2105-2132 (explicit WARNING for replay-failure re-queues).

**Consequence:** The judge is taught a wrong mental model of the failure mode (silent late breakage instead of immediate apply-time rejection), which wastes effort on defensive behavior and misdirects debugging when an apply-time rejection actually occurs.

**Fix:** Rewrite the paragraph: wrong basis is rejected at `playbook judge-apply` with the line number; the silent-requeue description applies only to stores written by hand (bypassing judge-apply) or banked before the #182 hardening.

### [MEDIUM] REFERENCE cites a nonexistent `template_hunk` field and gives two conflicting emergent-mode rules

*Audit finding(s) #21, #7 · doc-vs-doc · judge-prompts, skill-internal · CONFIRMED · [issue](https://github.com/contract-opf/playbook-engine/issues/154)*

**What:** REFERENCE.md's deviation rule 'If no template hunk is provided (`template_hunk` is null), deviation assessment is relative to the modal observed position, not a template' names a field that exists nowhere in the engine (the payload carries only hunk/our_standard plus traceability keys), and gives different guidance than the earlier rule for the same situation ('When our_standard is EMPTY, judge the [BEFORE]/[AFTER] change itself').

**Evidence:** REFERENCE.md:107-108 vs grep for template_hunk (only hits: REFERENCE.md itself and a test function name); agent_judge.py:663-667 (payload keys: stage/hunk/our_standard); REFERENCE.md:93 (conflicting emergent-mode instruction). No engine code computes a 'modal observed position' for the judge.

**Consequence:** A judge looking for template_hunk in pending items finds nothing and must guess which of two conflicting rules governs empty-template items; two judges (or two runs) can apply different comparison baselines to identical hunks.

**Fix:** Delete the template_hunk bullet or rewrite it in terms of the real field: 'when our_standard is the empty string, judge the [BEFORE]/[AFTER] change itself' — matching line 93 — and drop the unimplemented 'modal observed position' language.

> Found independently by judge-prompts (#21) and skill-internal (#7).

### [MEDIUM] Migration section omits the pre-#117 caveat: blanket-adopting legacy deviation banks is exactly what the v2 bump exists to prevent

*Audit finding(s) #22 · doc-vs-code · judge-prompts · CONFIRMED · [issue](https://github.com/contract-opf/playbook-engine/issues/155)*

**What:** REFERENCE.md presents `playbook judge-migrate` adoption as unconditionally safe ('banked human work is preserved rather than re-queued'), with no warning that adopting stamps legacy verdicts as CURRENT — including deviation verdicts banked before issues #117/#118/#119 changed hunk assembly, which rubric.py's own v1→v2 rationale calls 'answers to a materially different question'. All three on-disk production stores are 100% unstamped, and the July `full/out` store (1110 verdicts) predates #117.

**Evidence:** REFERENCE.md:317-328 (no caution; contrast cli.py:2361-2367 'Run --dry-run first, and only adopt once you have satisfied yourself the rubric has not moved under them'); rubric.py:117-127 (deviation='v2' bump rationale); measured: rerun-2026-08-23 1690/1690 unstamped, full-run.bak-2026-08-23 1444/1444 unstamped, full 1110/1110 unstamped (jq over judge/verdicts.jsonl).

**Consequence:** An agent following REFERENCE runs judge-migrate on the July out-dir and legitimizes ~1,110 deviation verdicts rendered over pre-#117 hunks as current-v2 answers, permanently defeating the invalidation the rubric bump was shipped to perform.

**Fix:** Add the CLI docstring's caution to REFERENCE and a concrete rule: for stores whose deviation verdicts predate #117 (July 2026 runs), do NOT adopt the deviation kind — scope adoption with --kind to the other kinds and let deviation re-queue.

### [MEDIUM] mine/judge append to segment/pending.jsonl without ever resetting it; the finished production run still holds 9 stale entries

*Audit finding(s) #72, #62, #56 · doc-vs-code · judge-economics, corpus-reality, pipeline-state · CONFIRMED · [issue](https://github.com/contract-opf/playbook-engine/issues/156)*

**What:** The standalone `segment` command resets its pending file each round (cli.py comment: 'fresh queue each round (mirrors judge, issue #182)'), but the agent-segmentation wiring used by `mine`/`judge` appends to the same $OUT/segment/pending.jsonl without ever resetting it, so drained items linger and duplicate across rounds. The validated production run ends with 9 stale lines (5 unique keys, 4 duplicated) in segment/pending.jsonl despite quarantine.json = [] and a complete playbook.

**Evidence:** cli.py:142 PendingQueue(seg_dir / 'pending.jsonl') constructed with no unlink, vs segment_cmd's pending_path.unlink(missing_ok=True) (~cli.py:2567). Artifact: rerun-2026-08-23/out/segment/pending.jsonl → 9 lines, jq -r .key | sort -u | wc -l → 5, duplicates present, mtime 12:53 (mid-run) while segment/cache.jsonl mtime is 12:55; quarantine.json is [].

**Consequence:** An agent following SKILL.md:500 ('Re-run segment to confirm 0 pending') or any later operator reading the leftover file concludes segmentation is incomplete and re-runs the single most expensive stage (SKILL.md:659-662 calls LLM segmentation 'typically the largest spend'), or re-judges already-segmented versions.

**Fix:** Unlink seg_dir/pending.jsonl in _segmentation_kwargs (cli.py:~136-146) before constructing the PendingQueue, mirroring judge's cli.py:2032; until then, add a note to SKILL.md Step 3/6 that segment/pending.jsonl left behind by mine is stale and only the `segment` command's own output is authoritative.

> Found independently by judge-economics (#72), corpus-reality (#62), and pipeline-state (#56).

### [LOW] Step 2a resume rule overclaims: stale segment verdicts are cached WITHOUT QA gating, not "rejected loudly"

*Audit finding(s) #53 · doc-vs-code · pipeline-state · CONFIRMED · [issue](https://github.com/contract-opf/playbook-engine/issues/157)*

**What:** SKILL.md's resume rule says a prior session's segment-verdicts file is safe to re-apply because "a stale/partial file is rejected loudly, never silently cached." In code, `segment-apply` QA-gates only verdicts whose canonical_text key is still in the CURRENT segment/pending.jsonl; any other line is counted as "ungated" and cached anyway with a single WARN.

**Evidence:** cli.py:2736-2738 (payload lookup miss → `ungated += 1`, no gating) and cli.py:2767-2774 (WARN "cached without QA gating"); SKILL.md:483-489.

**Consequence:** If the new session re-ran `segment` before applying (rewriting pending), or the corpus/extraction changed, old verdicts bypass the QA gates that exist to keep a bad partition from wedging a document (the exact wedge the gate-before-cache comment at cli.py:2692-2694 warns about) — with only a warning that the skill has taught the agent to expect as benign.

**Fix:** Skill: order the resume rule strictly as apply-BEFORE-re-running-segment and say that ungated warnings after a segment re-run mean stop and regenerate the queue. Engine: consider refusing (or requiring a flag) to cache verdicts with no matching pending item.

### [LOW] Production verdict store is 100% rubric-unstamped and the skill's update route never says to migrate it

*Audit finding(s) #76 · doc-vs-reality · judge-economics · CONFIRMED · [issue](https://github.com/contract-opf/playbook-engine/issues/158)*

**What:** All 1690 verdicts in the production store carry no rubric stamp (the run finished 13:07-13:11 on 2026-08-23; the rubric-versioning merge 999234a landed 15:14 the same day), so the very next `playbook judge` will print the legacy NOTE about all 1690 on every run until `judge-migrate` is run. REFERENCE.md § Rubric versions documents the migrate commands well, but SKILL.md's update route and drain loop mention judge-migrate only parenthetically (SKILL.md:726-733) and have no 'first run after an engine upgrade: dry-run judge-migrate' step.

**Evidence:** jq 'select(.rubric!=null)' rerun-2026-08-23/out/judge/verdicts.jsonl | wc -l → 0 of 1690; git log: run artifacts stamped 13:07-13:11 vs commit 999234a at 15:14:59 -0400; SKILL.md:726-733; REFERENCE.md:285-329.

**Consequence:** The next operator sees a warning about 1,690 verdicts of unknown validity and has to discover the migration workflow from REFERENCE.md's bottom section; worse, an over-cautious agent might treat the legacy bank as needing re-judgment (multi-hour, thousands of tokens) instead of a one-command adoption.

**Fix:** Add one line to SKILL.md's update route (Step 0 route B) and Step 6: 'On the first judge run after an engine upgrade, run playbook judge-migrate <out> --config … --dry-run; if only legacy items are reported and the rubric has not changed, adopt them with judge-migrate before draining — never re-judge a legacy bank.'

### [LOW] `judge --strict-rubric` is never documented, and the 'pending grows every round' failure symptom is impossible as written

*Audit finding(s) #43, #28 · doc-vs-code · cli-audit, judge-prompts · CONFIRMED · [issue](https://github.com/contract-opf/playbook-engine/issues/159)*

**What:** REFERENCE's 'Rubric versions' table documents the legacy (no-stamp) state as 'replays, reported every run until migrated', but omits the `--strict-rubric` flag that flips legacy verdicts to re-queued — the conservative option an auditor of a pre-versioning store may actually want.

**Evidence:** REFERENCE.md:295-299 (state table) and 317-328 (migration) vs `playbook judge --help` --strict-rubric ('Treat stored verdicts that carry NO rubric version ... as stale and re-queue them').

**Consequence:** An agent asked to fully re-validate an old verdict bank doesn't know the one-flag path and may instead delete the store or hand-filter verdicts.jsonl.

**Fix:** Add one row/sentence to REFERENCE's rubric table: 'to force re-judgement of the legacy bank instead of adopting it, run judge with --strict-rubric'.

> Found independently by cli-audit (#43) and judge-prompts (#28).

### [LOW] judge-apply help text contradicts its own behavior on partial loads

*Audit finding(s) #23 · doc-vs-code · judge-prompts · CONFIRMED · [issue](https://github.com/contract-opf/playbook-engine/issues/160)*

**What:** judge_apply_cmd's docstring says 'Valid lines are appended to the store even if earlier lines fail' immediately before the parenthetical '(partial loads are not performed — all lines are validated first)'; the code validates every line and exits on the first failure, loading nothing.

**Evidence:** cli.py:2155-2157 (docstring) vs cli.py:2212-2264 (validate-all-then-SystemExit(1) before any store.put_by_key).

**Consequence:** An operator whose apply failed mid-file may believe earlier verdicts were banked and skip re-applying them, leaving the drain loop mysteriously stuck one round longer.

**Fix:** Delete the first sentence; keep 'all lines are validated first; nothing is loaded if any line fails'.

### [LOW] String confidence crashes judge-apply with a raw traceback; deviation confidence is never validated at all

*Audit finding(s) #24 · producer · judge-prompts · CONFIRMED · [issue](https://github.com/contract-opf/playbook-engine/issues/161)*

**What:** validate_verdict raises TypeError (not ValueError) for a classify or scope verdict whose confidence is a string (e.g. "0.8"), and judge_apply_cmd catches only ValueError — so the user gets an unhandled traceback with no line number. Deviation verdicts skip confidence validation entirely (DeviationResult does not range- or type-check confidence), so a string or out-of-range confidence loads silently into the store.

**Evidence:** Reproduced: validate_verdict('classify', {...,'confidence':'0.8',...}) → TypeError "'<=' not supported between instances of 'float' and 'str'"; same for scope scope_confidence:'high'; deviation with confidence:'0.7' → accepted. cli.py:2260-2264 catches only ValueError; deviation_classifier.py:128-138 (__post_init__ checks deviation/basis only).

**Consequence:** LLM-produced verdicts commonly stringify numbers; the producing agent (or GC) sees a Python traceback instead of the actionable 'line N (kind): …' error every other malformation gets, and malformed deviation confidences bank silently.

**Fix:** In validate_verdict, coerce/verify confidence-type fields up front and raise ValueError with the field name; catch (ValueError, TypeError) in judge_apply_cmd; add a numeric-range check to DeviationResult.confidence.

### [LOW] Producer-allowed classify basis 'unclassified' is misdocumented as engine-only; provenance apply accepts undocumented bases

*Audit finding(s) #26 · doc-vs-code · judge-prompts · CONFIRMED · [issue](https://github.com/contract-opf/playbook-engine/issues/162)*

**What:** REFERENCE lists 'unclassified' among classify bases 'set by the engine itself, not by you', but _CLASSIFY_REPLAYABLE_BASES deliberately whitelists producer-supplied {'judge','unclassified'}. Conversely, validate_verdict accepts any deterministic provenance basis (template_similarity, alias_first_party, hint, …) from a producer file, none of which REFERENCE mentions as accepted.

**Evidence:** REFERENCE.md:187-189 vs agent_judge.py:381-385 (_CLASSIFY_REPLAYABLE_BASES comment: 'minus the two engine-internal bases'); agent_judge.py:454-461 (provenance path constructs ProvenanceResult with any _BASIS_VALUES basis).

**Consequence:** Silent gaps in both directions: a judge cannot use the sanctioned 'unclassified' signal it is allowed, and a sloppy producer can bank provenance verdicts under heuristic-looking bases that downstream consumers will misread as detector output rather than judge output.

**Fix:** Document 'unclassified' as valid for a no-fit classify verdict (taxonomy_id must be null), and tighten validate_verdict's provenance branch to {'llm'} to match the documented contract.

### [LOW] REFERENCE's confidence thresholds map to no engine mechanism and disagree with the engine's own ambiguity constants

*Audit finding(s) #27 · doc-vs-code · judge-prompts · CONFIRMED · [issue](https://github.com/contract-opf/playbook-engine/issues/163)*

**What:** REFERENCE's flag thresholds (classify <0.6, deviation <0.65) correspond to no engine constant: clause_classifier's ambiguity flag is 0.70, the AAR's low-confidence flag is 0.5, and deviation confidence never reaches observations. Only the provenance 0.70 threshold matches an engine behavior (the pipeline flip REFERENCE correctly documents). Combined with the dropped needs_review flag, the numbers are unenforced prose.

**Evidence:** REFERENCE.md:40,109 vs clause_classifier.py:75 (AMBIGUITY_THRESHOLD=0.70), aar.py:708 (conf < 0.5), observation keys in rerun-2026-08-23/out/observations.jsonl carry classification confidence only (all 0.45/0.0).

**Consequence:** A classify verdict at 0.62 is 'fine' per REFERENCE but ambiguous per the engine (is_ambiguous <0.70); a deviation verdict at 0.55 is 'flag it' per REFERENCE but indistinguishable from a confident one in every downstream artifact.

**Fix:** Align REFERENCE's classify threshold with AMBIGUITY_THRESHOLD (0.70), and state explicitly which thresholds are behavioral (provenance 0.70 flip; AAR 0.5 on classification confidence) versus advisory.

### [LOW] SegNode section says 'one verdict line per document'; queue items are per version

*Audit finding(s) #29 · doc-vs-code · judge-prompts · CONFIRMED · [issue](https://github.com/contract-opf/playbook-engine/issues/164)*

**What:** REFERENCE's SegNode section describes pending items as 'a document's canonical_text' and instructs 'one verdict line per document', but segment_cmd queues one item per VERSION file (payload carries both document_id and version), so a 5-version document needs 5 verdict lines.

**Evidence:** REFERENCE.md:250-253 vs cli.py:2586-2613 (per-version loop: for path in versions → pending.add(segment_payload_key(canonical_text), 'segment', {'document_id': …, 'version': path.stem, …})).

**Consequence:** A judge that emits one line per document leaves the other versions' items undrained; `playbook segment` keeps reporting pending items and the operator hunts a phantom bug.

**Fix:** Change to 'one verdict line per pending item (each item is one version of one document; identical version texts dedup by content hash)'.

### [LOW] Verdict-file naming convention in the drain loop does not survive contact with a real multi-round run

*Audit finding(s) #75 · usability · judge-economics · CONFIRMED · [issue](https://github.com/contract-opf/playbook-engine/issues/165)*

**What:** SKILL.md's drain loop names exactly one file, $OUT/my-verdicts.jsonl, and says only 'Do not accumulate across rounds'. The real rerun produced five differently-named files invented ad hoc (my-verdicts.jsonl, -dev, -prov, -scope, -2026-08-23), and the my-verdicts.jsonl actually present in the finished out-dir is a stale 651-verdict leftover copied in from the July run (mtime 12:41:04, identical to the seeded segment-verdicts.jsonl; all 651 keys already in the inherited store) — exactly the file the skill's literal instruction would overwrite or re-apply.

**Evidence:** stat mtimes: my-verdicts.jsonl and segment-verdicts.jsonl both Aug 23 12:41:04 (out-dir seeding), fresh files 12:59-13:07; comm of my-verdicts.jsonl keys vs full-run.bak store → 651/651 inherited; zero overlap with the fresh my-verdicts-2026-08-23.jsonl.

**Consequence:** On an update run against a copied out-dir, an agent following the skill verbatim either clobbers an audit file or re-applies a stale one; harmless to the store (idempotent put_by_key) but confusing at review time and a trap for 'which file is this round's work?'.

**Fix:** Have SKILL.md Step 6 prescribe per-round names (e.g. my-verdicts-<date>-r<N>.jsonl, matching what the operator actually did), state that judge-apply may be pointed at each round file, and warn that a my-verdicts.jsonl inherited inside a copied out-dir is a previous run's artifact, not a template to append to.


## Judge economics & triage

On the real corpus, triage IS the job: 88% of fresh deviation spend was alignment noise. The documented triage method cannot find it.

### [HIGH] Relocation triage method in REFERENCE.md cannot find most relocations; the method that worked (normalized/ clause trees) is undocumented

*Audit finding(s) #68 · doc-vs-reality · judge-economics · CONFIRMED · [issue](https://github.com/contract-opf/playbook-engine/issues/166)*

**What:** REFERENCE.md:97-106 tells the judge to find relocation pairs by scanning 'each document-pair's full hunk set' in pending.jsonl for near-identical disappear/reappear text. In the real 2026-08-23 rerun this caught only ~9 pairs (18 verdicts with the canned 'clause relocated, text unchanged' rationale) out of 143 alignment-artifact verdicts (~71 pairs); the other 125 were resolved by reading the per-version clause trees in $OUT/normalized/<doc>/NN__*.clauses.json, which neither SKILL.md nor REFERENCE.md ever mentions as a triage resource (SKILL.md:536 lists normalized/ only as a mine output).

**Evidence:** jq -r '.verdict.rationale' rerun-2026-08-23/out/my-verdicts-dev.jsonl | grep -ci 'relocat\|alignment artifact' → 143 of 163 (88%); of those, 18 = 'alignment artifact — clause relocated, text unchanged' (REFERENCE's canned wording), 101 = 'clause's text is present in two or more versions of the same document, so the aligner paired it differently' and 25 = 'segmented at different granularity' — determinations only checkable from the per-version trees. Structural cause is code-verified: the pending deviation payload is {stage, hunk, our_standard, taxonomy_id, clause_path, document_id} (playbook_engine/agent_judge.py:663-667, 741-746) where hunk is a flat '[BEFORE]\n…\n[AFTER]' string with NO version identifiers (see full-run.bak-2026-08-23/out/judge/batches/batch-00.jsonl line 1), and a relocation's unchanged counterpart clause generates no hunk at all, so it is simply absent from pending.jsonl — pair-scanning the pending set is structurally incapable of finding it.

**Consequence:** An agent following the written triage burns judge tokens and floods the queue with needs_review on items that are pure alignment noise (the July drain, my-verdicts.jsonl, shows 89 needs_review verdicts; the tree-based fresh round shows 0), and a GC gets a playbook whose deviation stats are polluted by phantom 'substantive' removals/additions.

**Fix:** Rewrite REFERENCE.md's relocation-triage bullet: (1) group pending deviation items by document_id; (2) for each hunk showing a clause disappearing or appearing, open the document's per-version clause trees at $OUT/normalized/<document_id>/*.clauses.json (each node carries clause_path/heading/text) and search the adjacent versions' node texts for the clause body after normalizing whitespace/quotes/numbering — presence in both versions ⇒ alignment artifact, auto-verdict all echoes none/neutral/none; (3) state explicitly that most relocation counterparts never appear in pending.jsonl, so scanning the pending set alone cannot find them. Add the same pointer to SKILL.md's 'Token efficiency' bullet (SKILL.md:1110-1112) and to the Step 4 inspection checklist ('Do not spend judgment effort until the backbone is healthy', SKILL.md:570-583). Producer side: add version_from/version_to to the pending payload as traceability context excluded from the hash, the exact pattern agent_judge.py:624-632 already uses for taxonomy_id/clause_path/document_id.

### [MEDIUM] 88% of fresh judge spend was alignment noise the engine could pre-filter deterministically

*Audit finding(s) #70 · producer · judge-economics · CONFIRMED · [issue](https://github.com/contract-opf/playbook-engine/issues/167)*

**What:** In the incremental rerun, 143 of 163 fresh deviation items (88%) — and therefore ~58% of all 246 fresh judged items — were alignment artifacts, and at least the 101 'text present in two or more versions' cases are detectable without an LLM: normalized-text containment of the hunk's clause body in the adjacent version's clause tree, the same check the human judge performed by hand.

**Evidence:** Fresh-round per-kind files: my-verdicts-scope.jsonl 43, my-verdicts-prov.jsonl 40, my-verdicts-dev.jsonl 163 (union my-verdicts-2026-08-23.jsonl 246); 143/163 artifact rationales, 101 of them the deterministic-looking 'present in two or more versions of the same document' wording. Dedup key construction already lives in agent_judge.py; the aligner produces the hunks upstream. Marked medium confidence: I did not machine-verify normalized containment for the 101 (would need re-deriving their payloads; the drained pending.jsonl no longer exists).

**Consequence:** On every future incremental corpus update, the dominant token cost of the judge loop is re-judging aligner noise one item at a time; the queue and report are dominated by non-decisions.

**Fix:** In the alignment/deviation-placeholder stage, before queueing a DISAPPEARED/APPEARED hunk, check whether its normalized clause text occurs in the counterpart version's clause tree; if so either suppress the item with a deterministic 'relocation' observation (basis: alignment) or pre-tag the pending payload with relocation_candidate: true so the judge can batch-confirm. Verify feasibility first by scripting the containment check over one document's normalized/ trees against its historical artifact verdicts.

### [LOW] REFERENCE's 'a third or more' relocation share badly understates reality (88% on the incremental rerun); '~92% artifacts' conflates two measures

*Audit finding(s) #64, #73 · doc-vs-reality · corpus-reality, judge-economics · CONFIRMED · [issue](https://github.com/contract-opf/playbook-engine/issues/168)*

**What:** Of the 163 deviation items in the labeled judge round, 150 (92.0%) were judged `none` but only 143 (87.7%) carry the 'alignment artifact' relocation rationale — the ~92% figure equals the none-verdict rate, not strictly the alignment-artifact rate. Meanwhile REFERENCE.md tells the judge to expect 'a third or more' relocation echoes, versus ~88% observed on this real corpus.

**Evidence:** wc -l <corpus>/eiaa-playbook-output/rerun-2026-08-23/out/my-verdicts-dev.jsonl → 163; jq verdict.deviation distribution → 150 none / 1 reworded_equivalent / 12 substantive; grep -c 'alignment artifact' → 143. REFERENCE.md:104-105 ('a third or more of deviation items are relocation echoes').

**Consequence:** A judge budgeting effort from REFERENCE.md expects to hand-judge ~two-thirds of items when the realistic residue after triage can be under 15%; and any prose citing '92% were alignment artifacts' overstates by ~4 points.

**Fix:** Update REFERENCE.md:104 to 'on real corpora one-third to as much as ~90% of deviation items can be relocation echoes (a real 44-agreement run saw 143/163)', and use 88%/143-of-163 when quoting the alignment-artifact share.

> Found independently by corpus-reality (#64) and judge-economics (#73).


## Pipeline state machine & silent failure modes

Atomic writes make interruption corruption-free; what survives is staleness and silence.

### [HIGH] `view apply` hints round-trip is dead under the documented Docker flow and for pseudonymized doc ids — corrections silently no-op while the CLI prints success

*Audit finding(s) #46 · doc-vs-code · pipeline-state · CONFIRMED · [issue](https://github.com/contract-opf/playbook-engine/issues/169)*

**What:** The skill's Feedback re-entry section runs `view apply` inside the container and claims "This writes corrected hints.yaml files ... Then re-judge and re-project." But `_find_hints_path` only probes `out_dir.parent/<doc_id>/hints.yaml` and `out_dir.parent.parent/<doc_id>/hints.yaml`; in the container that is `/work/<doc_id>` (never exists — corpus is at /work/corpus/<doc_id>), and with `known_entities` configured the feedback's document_id is the ALIAS, which matches no raw-named corpus folder anywhere. Both cases fall back to writing `out_dir/hints/<doc_id>.yaml` — a file no engine code ever reads (version ordering reads only `<corpus>/<doc>/hints.yaml`) — yet `hints_written` is still populated and the CLI prints "hints.yaml updated for <doc>".

**Evidence:** playbook_engine/viewer.py:1636-1649 (_find_hints_path probe locations), viewer.py:1538-1540 (dead-file fallback), viewer.py:1555 (appends doc_id to hints_written even on fallback), cli.py:3154-3156 (prints "hints.yaml updated"); grep shows the only hints.yaml readers are version_orderer/artifact_store/corpus_linter, all rooted at the corpus doc dir — nothing reads out/hints/. SKILL.md:1074-1082 (Docker `view apply` + "writes corrected hints.yaml files"); contrast SKILL.md:196-199, which says hints edits must be made "directly onto the host corpus tree with the agent's own file tools" — the skill contradicts itself.

**Consequence:** A reviewer's provenance/signed-version/order corrections exported from the review HTML are acknowledged as applied but never reach the corpus; the next judge/mine round reproduces exactly the trail errors the reviewer fixed, with no signal anywhere. On a pseudonymized corpus this happens even on the host venv with perfect layout.

**Fix:** Engine: give `apply_feedback` an explicit corpus-dir argument (or read it from run provenance), reverse-map aliased document_ids through the entity registry before locating the doc folder, and treat the fallback write as a reported skip (`result.skipped`), not a success. Skill: until then, route hints corrections per SKILL.md:199 (agent edits host corpus hints.yaml directly) and drop the claim that Docker `view apply` writes hints.

### [MEDIUM] The done-criteria read an interrupted judge round as 'done': pending.jsonl is deleted at round start and recreated lazily

*Audit finding(s) #103, #49 · doc-vs-code · failure-modes, pipeline-state · CONFIRMED · [issue](https://github.com/contract-opf/playbook-engine/issues/170)*

**What:** REFERENCE.md's machine-checkable done-criterion 1 accepts an ABSENT out/judge/pending.jsonl as done, but judge_cmd unlinks pending.jsonl at the start of every normal round and PendingQueue only creates the file on its first add — so a judge round killed mid-mining leaves pending.jsonl absent, and all three done-criteria pass against the stale artifacts of the previous round.

**Evidence:** cli.py:2032 `pending_path.unlink(missing_ok=True)` before mine_corpus runs; agent_judge.py:289-315 (PendingQueue stores the path at __init__, writes only in add()); REFERENCE.md:376-381 (`[ ! -f ... ] || [ ! -s ... ]` counts absence as done). Criteria 2 and 3 (validate exit 0, report/review HTML exist) are satisfied by the prior round's outputs.

**Consequence:** An agent resuming a session (or a human checking the done-criteria after an overnight run that died) concludes the derivation is complete and ships a playbook compiled from the previous round's store, with un-judged items silently missing.

**Fix:** Have judge write an explicit empty pending.jsonl (or a `judge/last_run.json` status stamp) on successful completion so absence is distinguishable from interruption; update REFERENCE.md's criterion 1 to require the empty-file form plus a successful exit of the last judge invocation.

> Found independently by failure-modes (#103) and pipeline-state (#49).

### [MEDIUM] Machine-checkable done-criteria never look at quarantine.json — quarantined documents exit the loop silently

*Audit finding(s) #50 · consumer · pipeline-state · CONFIRMED · [issue](https://github.com/contract-opf/playbook-engine/issues/171)*

**What:** A document quarantined during a judge/mine round (SegmentationQAError, HintsError, all-versions-failed-ingest) contributes no pending items, so the drain loop converges, `validate` passes, and REFERENCE.md's three done-criteria (pending empty, validate 0, report files exist) all hold on a playbook that silently lost whole agreements. quarantine.json is rewritten every run and is the canonical record, but neither SKILL.md Step 6's loop invariant nor REFERENCE.md's done-criteria mention checking it.

**Evidence:** pipeline.py:2959-3023 (quarantine paths produce no pending work), pipeline.py:3232 (quarantine.json always rewritten); REFERENCE.md:372-400 (no quarantine check); SKILL.md Step 6 (no quarantine mention in the loop).

**Consequence:** A GC's derivation completes 'done' with e.g. one agreement's whole negotiation history absent from the evidence; the only trace is a line in the AAR's needs-attention section, which the done-criteria treat as non-blocking prose.

**Fix:** Add a fourth machine-checkable criterion: `python3 -c 'import json;q=json.load(open("out/quarantine.json"));exit(1 if q else 0)'` — or require the loop to end by triaging every quarantine.json entry (re-segment, fix hints, or explicitly accept the exclusion in the report).

### [MEDIUM] `judge` runs no corpus lint preflight — only mine and segment are gated, yet judge is the command the drain loop runs most

*Audit finding(s) #101, #52 · doc-vs-code · failure-modes, pipeline-state · CONFIRMED · [issue](https://github.com/contract-opf/playbook-engine/issues/172)*

**What:** _run_corpus_preflight (the mandatory lint-corpus precondition added to stop issue #130-style dangling-symlink derivations) is wired into mine and segment only; judge_cmd runs the identical full mine_corpus work (extraction included) with only the environment-manifest check, no corpus lint.

**Evidence:** grep of cli.py: _run_corpus_preflight called at cli.py:1295 (mine) and cli.py:2515 (segment) only; judge_cmd (cli.py:1817-2136) calls _preflight_environment (1886) but never _run_corpus_preflight. SKILL.md:471-473 tells users "These same checks now run automatically as a preflight inside mine and segment".

**Consequence:** The drain loop (Step 6) re-runs judge over hours or days; a corpus that breaks between rounds (staged tree moved, symlinks now dangling, config template path broken) sails into a judge round and produces the same "finished, wrong-looking-like-right" thin derivation the preflight was built to prevent — per-file extraction failures only warn-and-quarantine.

**Fix:** Call _run_corpus_preflight(corpus_dir, config_path, skip=..., command="playbook judge") in judge_cmd after the environment preflight, with a --skip-preflight flag mirroring mine's.

> Found independently by failure-modes (#101) and pipeline-state (#52).

### [MEDIUM] `segment` neither checks nor stamps run_manifest.json, though it is the first whole-corpus extraction on the agent path

*Audit finding(s) #102, #51 · doc-vs-code · failure-modes, pipeline-state · CONFIRMED · [issue](https://github.com/contract-opf/playbook-engine/issues/173)*

**What:** SKILL.md Step 5 (606-615) says "You do not have to police this by hand" because mine/judge check the stored run manifest. But `segment` — the command that performs the expensive extraction for the key-free route and banks hours of agent segmentation work — has no _preflight_environment call and no _record_run_manifest, so a drifted environment (docling lost, stale image) is only caught later at mine, after segmentation verdicts were keyed to canonical_text extracted under the wrong environment (permanent cache misses once the environment is fixed).

**Evidence:** cli.py:2493-2637 segment_cmd contains neither _preflight_environment nor _record_run_manifest (grep at cli.py shows those only at 1287/1352 for mine and 1886/2071 for judge); segment builds ExtractionCache directly (cli.py:2570) and extracts every version (2589-2591). run_manifest.py:16 itself cites the observations-fell-2400→66 incident this check exists for.

**Consequence:** On the Step 2a route, an operator can burn a full re-extraction plus a full agent segmentation pass under a silently degraded environment; the refusal only comes at the subsequent mine, and the banked segment/cache.jsonl entries never replay after the environment is repaired because the canonical_text hashes changed.

**Fix:** Add the same _preflight_environment / _record_run_manifest pair to segment_cmd (it already loads the config and resolves out_dir), and soften SKILL.md's "you do not have to police this by hand" to name segment as the exception until then.

> Found independently by failure-modes (#102) and pipeline-state (#51).

### [MEDIUM] Stale feedback.json applies to the wrong clauses with no warning — export carries no binding to the render it came from

*Audit finding(s) #104 · producer · failure-modes · CONFIRMED · [issue](https://github.com/contract-opf/playbook-engine/issues/174)*

**What:** The review HTML's Export feedback produces {item_number: corrections} keyed by positional Cx/Cx.y numbers, with no content_hash or generated_at echo; apply_feedback maps those numbers onto the CURRENT playbook.opf.json. If the out-dir was re-projected between render and apply (corpus changed, clause set shifted), still-valid item numbers land corrections — pins, classifications, provenance hints — on different clauses silently; only unknown numbers get a skip warning.

**Evidence:** viewer.py:994-996 (exportFeedback serializes the fb map only); viewer.py:1319-1321 (item_map rebuilt from current doc), 1348-1350 (unknown numbers warned, shifted-but-valid numbers not detectable); no read of identity.content_hash from the feedback file anywhere in apply_feedback (viewer.py:1275-1345).

**Consequence:** In the skill's correction loop a reviewer can annotate an older playbook.review.html (e.g. the stale one the Feedback re-entry sequence leaves behind — see finding 2), and an attorney pin or provenance correction is recorded against the wrong clause/document with green "OK feedback applied" output.

**Fix:** Embed identity.content_hash (and generated_at) in the exported feedback JSON and have view apply refuse — or warn loudly and require --force — when it does not match the current playbook.opf.json.

### [MEDIUM] report.json's artifact checklist is false-by-construction under the skill's step ordering

*Audit finding(s) #61 · doc-vs-reality · corpus-reality · CONFIRMED · [issue](https://github.com/contract-opf/playbook-engine/issues/175)*

**What:** The persisted report claims playbook.digest.json, playbook.review.html, and playbook.opf.html do not exist even though all three are present in the same directory, because SKILL.md orders report (Step 9) before view/digest (Step 10) and nothing regenerates the report afterwards.

**Evidence:** jq '.artifacts' on <corpus>/eiaa-playbook-output/rerun-2026-08-23/out/report.json → files: {"playbook.opf.json": true, "playbook.digest.json": false, "playbook.review.html": false, "playbook.opf.html": false}; ls of the same directory shows all four files present (playbook.review.html 9.6MB, digest 190KB). SKILL.md:962-1009 fixes the order report → view/digest.

**Consequence:** Any consumer (or the toaster, or a resuming agent) that trusts report.json's artifact inventory concludes the render/digest steps never ran; the report also becomes internally inconsistent with REFERENCE.md's done-criterion 3.

**Fix:** Either swap Steps 9 and 10 in SKILL.md, add a final 'regenerate report' command after view/digest, or make the report's files block a note that it reflects report-time state.

### [LOW] REFERENCE done-criteria pass without the bundle and digest that SKILL Step 10 defines as THE deliverables

*Audit finding(s) #15 · doc-vs-doc · skill-internal · CONFIRMED · [issue](https://github.com/contract-opf/playbook-engine/issues/176)*

**What:** SKILL Step 10 declares `playbook.opf.html` "THE shareable/uploadable playbook" and `playbook.digest.json` a required sidecar, but REFERENCE's machine-checkable done-criteria only test report.md, report.json, and playbook.review.html — a run can be 'done' with neither shareable artifact generated.

**Evidence:** SKILL.md:990-1009 (two HTML artifacts + digest) vs REFERENCE.md:390-395 (criterion 3 tests only `report.md`, `report.json`, `playbook.review.html`).

**Consequence:** An agent using the done-criteria as its stop condition ends the session before `view bundle`/`digest`, and the GC receives the internal annotation surface but not the artifact the skill says is the only one to hand a stakeholder.

**Fix:** Extend done-criterion 3 to also test `playbook.opf.html` and `playbook.digest.json` (for the full-derivation route).

### [LOW] report silently tolerates a missing/partial out-dir and prints plausible zeros

*Audit finding(s) #106 · doc-vs-code · failure-modes · CONFIRMED · [issue](https://github.com/contract-opf/playbook-engine/issues/177)*

**What:** `playbook report` on an out-dir whose trail/ (and judge/) directories are missing renders a full report with "Versions: 0" for every agreement instead of refusing — an out-of-order or partially-copied out-dir produces confident-looking wrong numbers.

**Evidence:** Ran `playbook report` against a scratchpad copy of full/out containing only observations/manifest/scope/round_moves/playbook: output shows every agreement with Versions=0 and a fresh "Compiled at" timestamp, exit 0. aar.py:273-276 (`trail = trails.get(doc_id, {})`, `versions = trail.get("ordered_versions") or []`).

**Consequence:** A GC reading the after-action report from an incomplete directory sees a well-formed report whose Backbone Health/coverage numbers are silently wrong, rather than an error naming the missing artifacts.

**Fix:** Have build_after_action_report error (or print a prominent MISSING-ARTIFACT banner in the report) when trail/ or judge/ are absent while observations.jsonl is present.

### [LOW] validate never verifies identity.content_hash / section_digests

*Audit finding(s) #107 · doc-vs-code · failure-modes · CONFIRMED · [issue](https://github.com/contract-opf/playbook-engine/issues/178)*

**What:** validate_document runs schema + normative checks but never recomputes identity.content_hash or section_digests, so a hand-edited playbook (which the skill forbids three separate times precisely because it corrupts the hash) or any stale-hash bug passes `playbook validate` exit 0, while downstream consumers verifying the hash will reject the artifact.

**Evidence:** validator.py:739-789 — check list contains no identity/content_hash verification (grep for `identity`/`content_hash` in validator.py matches nothing in the check path); consumers are told to verify identity.content_hash (SKILL.md:1011-1014).

**Consequence:** The skill's Step 8 gate ("must exit 0") gives false assurance: the one local command positioned as the integrity gate cannot catch the hand-edit failure mode the skill warns about, and the mismatch only surfaces in the consuming application.

**Fix:** Add a non-blocking (or blocking, when identity is present) check to validate_document: recompute content_hash/section_digests over the document and report a mismatch.

### [LOW] run_manifest counts miss the llm-path segmentation cache, and artifact_cache_format is captured but never compared

*Audit finding(s) #54 · producer · pipeline-state · CONFIRMED · [issue](https://github.com/contract-opf/playbook-engine/issues/179)*

**What:** `collect_counts` reads only `out/segment/cache.jsonl` (agent path) for segmentation_cache_entries; the llm path's cache lives at `out/segmentation_cache.jsonl` (cli.py:269-271), so mismatch reports for llm-mode out-dirs understate the work at stake ("The saved clause grouping(s)..." falls back to the vague form). Separately, `RunEnvironment.artifact_cache_format` is recorded and rendered in the paste block but `compare()` has no finding for it, so a stage-cache format bump (silent full L1-L4 recompute) is the one recorded dimension that never produces a note.

**Evidence:** run_manifest.py:420 (`_count_lines(out_dir / "segment" / "cache.jsonl")`); cli.py:269-271 (SegmentationVerdictCache at `out_dir / "segmentation_cache.jsonl"`); run_manifest.py:566-808 (compare() covers resolved_extractor, extraction_cache_format, engine_version, segmentation identity, config_hash, git_sha — artifact_cache_format only at line 893 in the paste block).

**Consequence:** The plain-English consequence lines the mismatch report is built around are wrong (zero/vague counts) for llm-mode users, and an artifact-cache format bump redoes every per-doc stage silently despite the manifest knowing about it.

**Fix:** Sum both cache paths in collect_counts; add a non-blocking (or blocking, matching extraction-cache-format's treatment) compare() finding for artifact_cache_format.

### [LOW] Production backup out-dir is internally inconsistent: aborted-run manifest beside the full production playbook

*Audit finding(s) #66 · doc-vs-reality · corpus-reality · CONFIRMED · [issue](https://github.com/contract-opf/playbook-engine/issues/180)*

**What:** full-run.bak-2026-08-23/out mixes artifacts from two different runs: its corpus_manifest.json describes an aborted state (1 of 44 docs in-scope, 19 of 161 versions mined, 43 docs 'QA-quarantined … queued for agent segmentation') while its playbook.opf.json is the complete production playbook (2,303 observed positions, populated posture, 2 floor invariants). Note also this quarantine's cause is agent-segmentation queueing, not the docling-loss incident SKILL.md:286-288 describes with the same '43 of 44' number — the two stories are easy to conflate.

**Evidence:** jq over <corpus>/eiaa-playbook-output/full-run.bak-2026-08-23/out/corpus_manifest.json → {docs:44, found:161, mined:19, in_scope:1} with scope_rationale 'QA-quarantined during extraction/segmentation…' on 43 docs; jq over the same dir's playbook.opf.json → 24 clauses, 2,303 observed positions, posture populated, floor.invariants length 2.

**Consequence:** Any agent or tool that reads the manifest to characterize the production playbook (the skill's own Route B preflight reads the out-dir) concludes the corpus derivation collapsed, and the 43/44 number invites misattribution to the docling war story.

**Fix:** When backing up an out-dir, snapshot it atomically (or regenerate corpus_manifest.json to match the playbook state); consider having `report`/`inspect` flag manifest-vs-playbook disagreement.


## Report & GC-facing surfaces

The report is the artifact the skill tells the lawyer to read; today it drowns them and contradicts the playbook beside it.

### [MEDIUM] Needs Attention floods to 2,238 rows (91% of observations) by design on an agent-segmented corpus; 3 real ingest failures drown in it

*Audit finding(s) #60, #19, #71 · usability · corpus-reality, judge-prompts, judge-economics · CONFIRMED · [issue](https://github.com/contract-opf/playbook-engine/issues/181)*

**What:** Agent segmentation stamps every taxonomy assignment with flat confidence 0.45 (deliberately below aar's 0.5 threshold), so on a fully agent-segmented corpus every classified observation lands in the report's needs-attention list — 2,238 items in the rerun — which SKILL.md's Step 4 pitch ('first-pass classification for free') never mentions, and which defeats REFERENCE.md's stated goal of not flooding the report with review flags.

**Evidence:** jq '.needs_attention | length' on <corpus>/eiaa-playbook-output/rerun-2026-08-23/out/report.json → 2238; reasons distribution: 1925 'low confidence (0.45)', 310 'low confidence (0.00)', 3 ingest failures. Source of the constant: playbook_engine/pipeline.py:774 (_LLM_SEGMENTER_CONFIDENCE = 0.45, docstring explicitly says every such clause 'always surfaces for human review'). SKILL.md:512-514 and REFERENCE.md:104-106 give no hint of this consequence.

**Consequence:** The GC's 'Needs Attention' section — the report's core human-review contract — is an unactionable 2,238-row table where 3 genuinely broken versions (ExtractionError) drown among 2,235 by-design flags; real problems get missed.

**Fix:** Roll the flat-0.45 cohort into a single aggregate line in the report ('N clauses classified during agent segmentation — spot-check sample'), keep individual rows only for the 0.00 and failure cases, and document the trade-off in SKILL.md Step 4.

> Found independently by corpus-reality (#60), judge-prompts (#19), and judge-economics (#71: report.md is 2,443 lines of which 2,247 are the table; grouping by (doc, taxonomy, reason) alone would collapse it to 676 rows).

### [MEDIUM] Report honesty notes are hardcoded and contradict the populated Posture/Floor in the same run

*Audit finding(s) #59 · producer · corpus-reality · CONFIRMED · [issue](https://github.com/contract-opf/playbook-engine/issues/182)*

**What:** aar.py emits fixed honesty_notes — 'GC-authored Posture is a v0.2 human-input field not yet generated by the engine' and 'Floor clauses are derived from classified reversals and require attorney sign-off' — unconditionally, so the rerun's report denies the existence of the posture and floor that its own playbook contains; the second note also states Floor is 'derived', contradicting the control ladder (SKILL.md:1113 'Posture and Floor are never derived').

**Evidence:** playbook_engine/aar.py:893-897 (unconditional list in the return value); <corpus>/eiaa-playbook-output/rerun-2026-08-23/out/report.md:2403-2404 carries both notes and :2435 says '346 reversal(s) detected — Floor derivation requires attorney classification', while the same dir's playbook.opf.json has posture.version=1 with a 5-answer interview (generated_at 2026-08-23T17:10:01Z) and floor.invariants of length 2.

**Consequence:** A GC reading the report — the artifact the skill tells them to review — is told their signed posture and floor do not exist, and is told floor content is machine-derived, undermining trust in the Rung 1/2 provenance story.

**Fix:** Make the honesty notes conditional on playbook state (posture/floor empty vs populated) in aar.py, and reword the floor note to 'Floor candidates are proposed from classified reversals; only human-signed invariants enter the playbook.'

### [MEDIUM] Floor checklist help text explains signing in spec/engineer language, not legal weight

*Audit finding(s) #114 · usability · lawyer-usability · CONFIRMED · [issue](https://github.com/contract-opf/playbook-engine/issues/183)*

**What:** The user-facing "Proposed hard lines" help paragraph in playbook.review.html — the exact surface where the GC clicks Accept — describes acceptance as promoting "into the signed `floor.invariants`, a categorical, judge-checkable hard line" and cites "OPF-SPEC.md §3.7 rule 4". "judge-checkable" reads to a lawyer as courtroom-judge; `floor.invariants` is a JSON path; the spec citation is unread by this audience. The operative legal meaning — this will block/fail every future review that violates it and cannot be overridden by the consuming reviewer — is stated in ADOPTING.md but not at the point of the click.

**Evidence:** playbook_engine/viewer.py:807-815 (help-text string); contrast docs/ADOPTING.md:181-184 (the plain-language version: "a signed Floor invariant is the one section a conformant consumer may never quietly override").

**Consequence:** The GC either under-weights the click (signs casually) or is scared off by the jargon and stalls at 'undecided' forever; either way the one irreversible-feeling control on the page is the worst-explained one.

**Fix:** Rewrite the help paragraph in viewer.py in ADOPTING's register: "Accepting signs this as a hard line: every future contract review will treat it as non-negotiable and flag any deal that violates it. No reviewer or AI can override it. Rejecting records that you looked and declined." Drop or footnote the spec citation and the word 'judge-checkable'.

### [LOW] Report token estimate still uses the flat per-item average the plan command was fixed to abandon

*Audit finding(s) #74 · doc-vs-code · judge-economics · CONFIRMED · [issue](https://github.com/contract-opf/playbook-engine/issues/184)*

**What:** cli.py's --plan-only estimate was reworked (issue #134) to size the queue from real payload bytes because 'a flat per-item average previously ignored that … payloads can differ by an order of magnitude', but aar.py still computes the report's token_estimate as pending_count * 200 (_AVG_TOKENS_PER_ITEM), so the plan and the report can disagree about the same pending queue.

**Evidence:** playbook_engine/aar.py:55 (_AVG_TOKENS_PER_ITEM = 200) and aar.py:407 (token_estimate = pending_count * _AVG_TOKENS_PER_ITEM) vs cli.py:~2007-2018 (sum of json.dumps(payload) lengths // 4, with the issue #134 comment).

**Consequence:** When a report is generated with a non-empty queue, the GC sees a token estimate that can be off by an order of magnitude versus what `judge --plan-only` just told them.

**Fix:** Have _build_judgment_economics size pending items from their actual payload bytes // 4 (the records are already parsed at aar.py:378-392), or share the estimator with cli.py.

### [LOW] Signed posture embeds a slightly wrong corpus statistic (94% vs measured 95.1%)

*Audit finding(s) #67 · doc-vs-reality · corpus-reality · CONFIRMED · [issue](https://github.com/contract-opf/playbook-engine/issues/185)*

**What:** The rerun's authored posture answer states '94% of observed clauses sit on our paper' (mirroring SKILL.md:821's worked example), but the run's own observations measure 2,345/2,466 = 95.1% our_paper; document-level provenance is 37/44 = 84%. The signed artifact bakes in a number matching neither denominator.

**Evidence:** jq -r '.provenance' <corpus>/eiaa-playbook-output/rerun-2026-08-23/out/observations.jsonl | sort | uniq -c → 2345 our_paper / 121 counterparty_paper; posture text via jq -r '.posture.system_prompt' in the same dir's playbook.opf.json contains '94% of observed clauses sit on our paper'. (The '2 rounds across 40 agreements' claim in the same posture does check out: round_moves.jsonl spans exactly 40 unique document_ids.)

**Consequence:** Small, but the posture is a signed Rung-1 artifact whose stated corpus grounding a diligent reviewer can falsify against the same out-dir — eroding trust in the grounding discipline.

**Fix:** Have the interview step compute the cited percentage from observations.jsonl at authoring time (and name the denominator: observations vs documents), rather than reusing the skill's example figure.

### [LOW] Triage header advertises a "checklist below" that is absent when zero candidates are proposed

*Audit finding(s) #117 · doc-vs-reality · lawyer-usability · CONFIRMED · [issue](https://github.com/contract-opf/playbook-engine/issues/186)*

**What:** viewer.py emits the triage action "~minutes — sign proposed hard lines: checklist below" unconditionally, but the "Proposed hard lines" section renders as empty string when there are no candidate rows — so with 0 proposed, the header points at a section that does not exist on the page.

**Evidence:** playbook_engine/viewer.py:854 (unconditional `<li>` string) vs viewer.py:795-803 (returns "" when no rows). Confirmed in a real artifact: <corpus>/eiaa-playbook-output/rerun-2026-08-23/out/playbook.review.html shows "0 proposed awaiting sign-off" and "sign proposed hard lines: checklist below" while `grep -c 'Proposed hard lines'` on the file returns 0.

**Consequence:** The GC scrolls looking for a checklist that isn't there and concludes the page is broken or that they're missing something.

**Fix:** When pending_count == 0, render the action as "sign proposed hard lines: none proposed right now — run `playbook floor propose` after the next recompile" (or omit the 'below' pointer).


## Docs & onboarding (the lawyer path)

Read as a GC with no engineering background, the doc set has one hard wall and steady friction.

### [HIGH] QUICK-COMPILE assumes an installed `playbook` CLI with no install step or link

*Audit finding(s) #110 · usability · lawyer-usability · CONFIRMED · [issue](https://github.com/contract-opf/playbook-engine/issues/187)*

**What:** docs/QUICK-COMPILE.md — the doc explicitly pitched at non-engineers ("without writing any code", and SKILL.md:1129-1131 calls it the "non-engineer corpus layout guide") — opens Step 0 with `playbook doctor` and never explains where the `playbook` command comes from, that commands are typed in Terminal.app, or how to clone/install anything. `grep -in 'install|README|venv|terminal' docs/QUICK-COMPILE.md` matches only line 62 (a passing mention of the Docker image).

**Evidence:** docs/QUICK-COMPILE.md:55-59 (Step 0 is `playbook doctor`); grep for install/README/terminal across the file returns only line 62. Install instructions exist only in README.md:158-181 (venv + `brew install pandoc` + `make install`, or `make docker-build`), which QUICK-COMPILE never links.

**Consequence:** A GC on a Mac types `playbook doctor` (or doesn't even know where to type it), gets `command not found`, and bounces at the very first command. Every subsequent step in the guide is unreachable.

**Fix:** Add a 'Step -1: Get set up' section (or a prominent link to README § Installation plus a one-line 'open Terminal.app, paste this') at the top of QUICK-COMPILE.md; alternatively state that the intended non-engineer path is opening the repo in Claude Code and letting the skill drive.

### [MEDIUM] SKILL 'Before you start' tells the agent to ask which party is 'us'; the Interactive-setup section forbids exactly that

*Audit finding(s) #4 · doc-vs-doc · skill-internal · CONFIRMED · [issue](https://github.com/contract-opf/playbook-engine/issues/188)*

**What:** Prerequisite item 3 says the designation of which party is "us" should be collected "with one interactive prompt before beginning", while the Interactive setup section says "Do **not** ask 'which party is us' or 'who are your counterparties' — derive those yourself".

**Evidence:** SKILL.md:38-40 ("A designation of which party is 'us' ... Collect these with one interactive prompt") vs SKILL.md:110-111 ("Do **not** ask 'which party is us'").

**Consequence:** Depending on which paragraph the LLM anchors on, the GC either gets the unreliable brand-name prompt the derive-party-names section was specifically written to prevent (risking wrong provenance corpus-wide), or gets no prompt — two contradictory instructions for the same setup moment.

**Fix:** Rewrite prerequisite 3 to: 'a designation of which party is us — derived automatically from recitals (see "Derive party names automatically"); only the display name and optional baseline template are asked interactively.'

### [MEDIUM] ADOPTING promises the skill 'interviews you for' the config; the skill requires a pre-existing playbook.config.yaml

*Audit finding(s) #5 · doc-vs-doc · skill-internal · CONFIRMED · [issue](https://github.com/contract-opf/playbook-engine/issues/189)*

**What:** ADOPTING Stage 1 says "if you use this path, you can skip the config-writing above; the skill interviews you for it", but SKILL.md lists an existing `playbook.config.yaml` referencing a taxonomy as a prerequisite, and its interactive setup asks only two questions (display name, baseline template) — it never walks through creating a config, choosing a taxonomy, or setting agreement_type.

**Evidence:** docs/ADOPTING.md:76-78 vs SKILL.md:36-37 ("You need: ... A `playbook.config.yaml` referencing your taxonomy YAML") and SKILL.md:104-112 (only 2 questions); SKILL.md:126-127 only edits provenance lists into an EXISTING config.

**Consequence:** A first-time adopter who follows ADOPTING's promise and skips config-writing lands in a skill whose Step 2 lint immediately fails CONFIG_NOT_FOUND, with no documented recovery path inside the skill for authoring the config (taxonomy choice, agreement_type) from scratch.

**Fix:** Either add a config-authoring step to SKILL.md's interactive setup (taxonomy default `builtin:cuad-base.yaml`, agreement_type, template) or soften ADOPTING to 'the skill fills in the provenance lists for you; you still write the four-line config'.

### [MEDIUM] Config placement contradicts between QUICK-COMPILE (beside the corpus) and SKILL (inside it, for the Docker mount)

*Audit finding(s) #113, #16 · doc-vs-doc · lawyer-usability, skill-internal · CONFIRMED · [issue](https://github.com/contract-opf/playbook-engine/issues/190)*

**What:** QUICK-COMPILE says the config goes "in the same directory as your corpus" and its examples put it as a sibling of the corpus folder (`playbook lint-corpus ./corpus --config ./playbook.config.yaml`), while SKILL.md requires the config live *inside* the corpus directory so it is visible under the read-only Docker mount. A user who sets up per QUICK-COMPILE and then graduates to the skill's recommended Docker path has their config in a location the container cannot see.

**Evidence:** docs/QUICK-COMPILE.md:45 ("in the same directory as your corpus") and :71 (sibling-path example) vs SKILL.md:189-192 ("must live *inside* the corpus directory... put yours at $CORPUS/playbook.config.yaml") and :211-213.

**Consequence:** GC gets `CONFIG_NOT_FOUND` (or a mount error) on the upgrade path and has no way to connect the failure to a sentence they followed correctly in the other doc.

**Fix:** Make QUICK-COMPILE.md:45 say "inside your corpus folder, next to the agreement subfolders" and update its example commands to `--config ./corpus/playbook.config.yaml`, matching the skill and the Docker mount reality.

> Found independently by lawyer-usability (#113) and skill-internal (#16).

### [MEDIUM] ADOPTING Stage 5 describes publish as running 'an LLM residue pass' — the CLI wires stub judges only

*Audit finding(s) #33 · doc-vs-code · cli-audit · CONFIRMED · [issue](https://github.com/contract-opf/playbook-engine/issues/191)*

**What:** ADOPTING.md says `playbook publish` performs 'an LLM residue pass over every free-text surface with an independent verify pass behind it', but no LLM is wired into publish — it defaults to stub judges (basis="stub"); in the skill route the residue classification is done manually by the agent reading residue_report.json.

**Evidence:** docs/ADOPTING.md:228-231 vs `playbook publish --help`: 'No LLM is wired here — this defaults to stub judges (basis="stub"), same as every other zero-configuration path in this engine. Wiring a real judge is a separate concern (see playbook_engine/export_profile.py).'

**Consequence:** A GC deciding whether a published playbook is safe to share reads ADOPTING and believes a semantic LLM sweep vouched for the artifact; in reality the semantic pass is a stub unless the skill/agent route did the classification by hand.

**Fix:** Reword ADOPTING Stage 5 to match the CLI: deterministic backstop + residue_report.json for review, with the LLM/agent judgment supplied by the skill route (or a wired judge), not by `playbook publish` itself.

### [MEDIUM] The runtime estimator silently misclassifies every scanned PDF as born-digital when pdfplumber is absent — and the skill invokes it with bare `python`

*Audit finding(s) #34, #115 · producer · cli-audit, lawyer-usability · CONFIRMED · [issue](https://github.com/contract-opf/playbook-engine/issues/192)*

**What:** estimate_runtime.py's is_scanned_pdf returns False ('optimistic') on ImportError with no warning printed, so run outside the venv every scanned PDF is costed at ~60s instead of ~330s. SKILL.md invokes it as `python .claude/.../estimate_runtime.py`, which on a machine where pdfplumber lives only in .venv produces a confidently wrong, several-fold-low ETA — defeating the pre-flight's entire purpose ('a multi-hour run is never a surprise').

**Evidence:** .claude/skills/playbook-from-corpus/estimate_runtime.py:200-203 (`except ImportError: return False  # can't tell; treat as born-digital (optimistic)`) — no stderr warning anywhere (grep for 'pdfplumber' shows only lines 6, 201, 205, 211); SKILL.md:253 and 626 use bare `python`.

**Consequence:** The GC approves a '45-minute' run that actually takes 5+ hours on a scanned corpus — the exact surprise the Pre-flight section exists to prevent.

**Fix:** Print a one-line warning when pdfplumber is missing ('scanned-PDF detection disabled — ETA may be several times too low') and/or exit non-zero; change the SKILL invocations to `.venv/bin/python`.

> Found independently by cli-audit (#34) and lawyer-usability (#115). Verified live: system python3 on this machine lacks pdfplumber; scanned PDFs under-cost ~5.5×.

### [MEDIUM] Extraction-cache pre-flight and estimator gate sit at Step 5, after Step 3 already ran the multi-hour extraction

*Audit finding(s) #8 · doc-vs-doc · skill-internal · CONFIRMED · [issue](https://github.com/contract-opf/playbook-engine/issues/193)*

**What:** In the Ordered pipeline, Step 3 (`mine`) performs extraction/OCR, but the guidance to reuse a warm extraction cache, keep $OUT stable, and run the estimator to surface remaining OCR cost before committing lives in Step 5 — two steps after a literal reader has already paid the cost. The standalone pre-Step-1 'Pre-flight' section partially covers this, but the ordered sequence itself is what an LLM executes, and it never says 'run the Step 5 extraction pre-flight before Step 3'.

**Evidence:** SKILL.md:523-529 (Step 3 mine = ingest/extraction) vs SKILL.md:589-647 (Step 5 "Pre-flight — reuse a warm extraction cache; never re-OCR what's done" + estimator with $OUT); the earlier gate at SKILL.md:242-263 covers stage/segment/mine but appears outside the numbered pipeline.

**Consequence:** An agent following Steps 1→11 in order (as the Route A table instructs: "1 → 11 (everything below, in order)") can start a surprise multi-hour OCR at Step 3 with no ETA confirmation — the exact outcome the pre-flight sections exist to prevent.

**Fix:** Move the extraction-cache/estimator pre-flight into (or immediately before) Step 3, or renumber it as Step 2b, leaving Step 5 with only the judge --plan-only / --subset material.

### [MEDIUM] REFERENCE names a config key `known_aliases` that does not exist anywhere in the engine

*Audit finding(s) #3, #25, #39 · doc-vs-code · skill-internal, judge-prompts, cli-audit · CONFIRMED · [issue](https://github.com/contract-opf/playbook-engine/issues/194)*

**What:** Three places in REFERENCE.md (provenance rule, guardrail 3, failure-mode fix "Supply `known_aliases`") reference `known_aliases`; the real config keys are `provenance.our_party_aliases` (us) and `provenance.known_entities` (counterparties). `known_aliases` appears nowhere else in the repo.

**Evidence:** `grep -rn known_aliases playbook_engine/ docs/ .claude/skills/` → only REFERENCE.md:154, :341, :410; real keys at playbook_engine/config.py:203-209 and docs/ADOPTING.md:36-38.

**Consequence:** The failure-mode fix for 'All provenance = counterparty_paper' is unactionable as written, and an agent auditing 'unknown aliases' has no real key to check the name against — it may edit the config with a key the loader ignores, leaving the underlying problem in place.

**Fix:** Replace `known_aliases` with `provenance.our_party_aliases` (for the us-check in provenance rules) and `provenance.known_entities` (for the counterparty/pseudonymization check) in all three spots.

> Found independently by skill-internal (#3), judge-prompts (#25), and cli-audit (#39). REFERENCE's own provenance section uses the correct name, so the file is internally inconsistent.

### [MEDIUM] REFERENCE guardrail 7 mislocates historical_stance in the Posture and inverts its derivability

*Audit finding(s) #2, #84 · doc-vs-doc · skill-internal, control-ladder · CONFIRMED · [issue](https://github.com/contract-opf/playbook-engine/issues/195)*

**What:** REFERENCE.md says "The `historical_stance` (Posture section) ... require the GC interview (OPF-SPEC.md §7) and cannot be derived from the corpus". Per the spec, historical_stance lives in Evidence, is purely DESCRIPTIVE, and is exactly the thing that IS derived from the corpus ("what has the corpus shown we do here?"). QUICK-COMPILE also treats it as compiled output.

**Evidence:** REFERENCE.md:357-359 vs docs/OPF-SPEC.md:84-88 ("historical_stance — a purely descriptive summary of the corpus") and docs/QUICK-COMPILE.md:214 (`historical_stance: no_signal` as compiled output).

**Consequence:** A judge-agent reading the guardrail can conclude a populated historical_stance is a fabrication violation (it 'cannot be derived from the corpus'), or tell the GC the interview is needed to fill a field the compiler already filled — confusing exactly the Rung-0/Rung-1 boundary the skill works hard to explain.

**Fix:** Rewrite guardrail 7 to name the actual interview-only fields: `posture.system_prompt` (Posture) and `floor.invariants` (Floor); drop historical_stance or move it to the Evidence side as corpus-derived.

> Found independently by skill-internal (#2) and control-ladder (#84).

### [MEDIUM] The sacred-clauses question is called both Q1 and Q4 inside the same SKILL step

*Audit finding(s) #1, #83 · doc-vs-doc · skill-internal, control-ladder · CONFIRMED · [issue](https://github.com/contract-opf/playbook-engine/issues/196)*

**What:** Step 7a re-numbers the interview by importance (table at SKILL.md:805-812: order 1 = sacred_clauses, order 4 = rounds), and the compute-options script comment uses that numbering ("seeds Q1/sacred ... rounds-to-settle per document (Q4)"), while the rules three paragraphs later and the Guardrails use spec numbering where Q4 = sacred ("Q4 is different in kind, and signs itself").

**Evidence:** SKILL.md:771 ("reversal candidates (seeds Q1/sacred) ... rounds-to-settle per document (Q4)") vs SKILL.md:872 ("Q4 is different in kind, and signs itself") and SKILL.md:1119 ("the interview's own Q4 answer"); engine and spec fix Q4=sacred_clauses (playbook_engine/posture.py:84, docs/OPF-SPEC.md:625,629).

**Consequence:** An LLM cross-referencing the two passages can conclude the ROUNDS answer is the one that signs directly into floor.invariants — miswiring which answer becomes a hard-binding invariant, the single highest-stakes mapping in the interview.

**Fix:** Use spec ids everywhere: change line 771's comment to "(seeds Q4/sacred) ... (Q1 rounds)", or drop Q-numbers from the script comment and use the question ids (`sacred_clauses`, `rounds`).

> Found independently by skill-internal (#1) and control-ladder (#83).

### [MEDIUM] Three surfaces prescribe `--no-cache` after a hints.yaml edit; the hints hash is already in the cache key, so plain `mine` suffices

*Audit finding(s) #36, #55, #116 · doc-vs-code · cli-audit, pipeline-state, lawyer-usability · CONFIRMED · [issue](https://github.com/contract-opf/playbook-engine/issues/197)*

**What:** QUICK-COMPILE Step 4 (and `playbook inspect --help`) instruct re-running `playbook mine --no-cache` after adding hints.yaml, then in the same paragraph warn that --no-cache forces full re-extraction and 'is not a cheap flag to reach for routinely'. The doc-level cache key already includes the sha256 of hints.yaml, so a plain `playbook mine` re-run picks hints up for exactly the affected documents while replaying the warm extraction cache — the skill's own Step 4 says plain re-run, correctly.

**Evidence:** docs/QUICK-COMPILE.md:169 vs playbook_engine/artifact_store.py:70-75 ('hints.yaml drives version ordering → must be part of the key'; hashed into make_doc_key) and playbook_engine/pipeline.py:2884 (hints_path passed). SKILL.md:585-588 says re-run `playbook mine` with no flag. `playbook inspect --help`: 're-run with --no-cache'.

**Consequence:** A GC correcting one trail on a scanned 44-agreement corpus re-OCRs everything for hours instead of a minutes-long incremental re-mine; the internally contradictory paragraph also erodes trust in the doc.

**Fix:** Change QUICK-COMPILE.md:169 and the inspect command's docstring to plain `playbook mine` for hint pickup, reserving --no-cache for suspect extractions.

> Found independently by cli-audit (#36: QUICK-COMPILE:169), pipeline-state (#55: the inspect docstring), and lawyer-usability (#116: the internal contradiction at QUICK-COMPILE:185, which the verifier settled in code — auto-detect is correct).

### [LOW] ADOPTING claims staging always proposes a reviewable plan ('Nothing moves until you approve'); plan/approve is only the layout-unknown path

*Audit finding(s) #42, #10 · doc-vs-code · cli-audit, skill-internal · CONFIRMED · [issue](https://github.com/contract-opf/playbook-engine/issues/198)*

**What:** ADOPTING Stage 1 says a CLM/DMS export makes stage 'propose a staging_plan.json ... you review the plan, then execute it. Nothing moves until you approve.' In code, the plan flow is only entered for layout `unknown` or with an explicit --plan-only; a recognized clm_nested or manifest layout stages immediately with no plan and no approval step.

**Evidence:** docs/ADOPTING.md:24-28 vs playbook_engine/staging.py:358-368 (stage() raises UnknownLayoutError for unknown, otherwise proceeds straight to _recreate_out_dir and staging) and `playbook stage --help` ('Required first step for a corpus whose layout is unknown').

**Consequence:** A cautious adopter expecting a review gate finds their staging root already written; conversely they may wait for an approval prompt that never comes.

**Fix:** Reword to: recognized layouts stage directly (source never modified); unknown layouts refuse and require the --plan-only → review → --from-plan loop; add --plan-only explicitly for adopters who want the preview on any layout.

> Found independently by cli-audit (#42) and skill-internal (#10).

### [LOW] quarantine.json and coherence_flags.json are real mine outputs no document mentions

*Audit finding(s) #40 · doc-vs-reality · cli-audit · CONFIRMED · [issue](https://github.com/contract-opf/playbook-engine/issues/199)*

**What:** The real run's out-dir contains quarantine.json and coherence_flags.json, but neither appears in SKILL.md, REFERENCE.md, QUICK-COMPILE's Step 4 intermediates table, ADOPTING, or `playbook mine --help`'s output list — even though the skill's own docling anecdote ('quarantined 43 of 44 documents') shows quarantine visibility is exactly what a GC needs to check.

**Evidence:** <corpus>/eiaa-playbook-output/rerun-2026-08-23/out/ listing (quarantine.json, coherence_flags.json present); docs/QUICK-COMPILE.md:149-155 table omits both; grep 'quarantine' over the four docs hits only SKILL.md:288,440 (prose, not the artifact).

**Consequence:** The inspection checkpoint (Step 4) never tells the agent to open quarantine.json, so a partially quarantined corpus can pass the 'backbone healthy' check by looking merely thin.

**Fix:** Add both files to QUICK-COMPILE's intermediates table and to SKILL Step 4's sanity-check list ('read quarantine.json — any quarantined version is a document the playbook silently lost').

### [LOW] Classification prompt says pick from `taxonomy_entries`; the payload field (correctly named 3 lines earlier) is `taxonomy_ids`

*Audit finding(s) #11 · doc-vs-doc · skill-internal · CONFIRMED · [issue](https://github.com/contract-opf/playbook-engine/issues/200)*

**What:** REFERENCE's classification Task line references `taxonomy_entries` although the Input-fields line and the engine payload both use `taxonomy_ids` (a flat id list; labels live in the taxonomy YAML).

**Evidence:** REFERENCE.md:20 ("Assign the best-fit `taxonomy_id` from `taxonomy_entries`") vs REFERENCE.md:16-17 (`taxonomy_ids` (flat list...)) and playbook_engine/agent_judge.py:548-552 (payload key `taxonomy_ids`).

**Consequence:** Minor confusion for the judging agent hunting for a `taxonomy_entries` field carrying labels/descriptions that is not in the payload; it may waste a pass or under-use the taxonomy YAML it was told to read.

**Fix:** Change line 20 to "from `taxonomy_ids`" (and keep the pointer to the taxonomy YAML for labels).

### [LOW] "Two things this changes about the steps below" introduces a five-item list

*Audit finding(s) #12 · doc-vs-doc · skill-internal · CONFIRMED · [issue](https://github.com/contract-opf/playbook-engine/issues/201)*

**What:** The Docker-runtime section's lead-in says two things change, then enumerates five numbered items — items 3-5 (template-mode verification, config/template placement, image-version stamp) were evidently appended without updating the count.

**Evidence:** SKILL.md:185 ("Two things this changes about the steps below:") followed by items 1-5 at SKILL.md:187-221.

**Consequence:** An LLM skimming for 'the two things' may treat items 3-5 as lower-priority commentary and skip the template-mode-degradation check (item 3), which the text itself marks 'do not proceed'.

**Fix:** Change the lead-in to "Five things this changes" or simply "What this changes about the steps below:".

### [LOW] Route C's step notation '8 → 10' is ambiguous next to Route B's explicit '8 → 9 → 10'

*Audit finding(s) #14 · doc-vs-doc · skill-internal · CONFIRMED · [issue](https://github.com/contract-opf/playbook-engine/issues/202)*

**What:** The route table writes Route B's tail as "8 → 9 → 10" but Route C's as "then 8 → 10" — either a range (8,9,10) or a skip of Step 9 (report). Nothing disambiguates, and Route C is precisely the route where the report's needs-attention delta after a re-derivation matters most.

**Evidence:** SKILL.md:89 (Route B: "7a → 7b → 8 → 9 → 10") vs SKILL.md:90 (Route C: "then 8 → 10").

**Consequence:** An LLM parsing arrows as explicit step sequences (as Route B's row teaches it to) skips the Step 9 after-action report on Route C, losing the surface where recompile conflicts with pins/floor are reported.

**Fix:** Write Route C's tail as "8 → 9 → 10".

### [LOW] Engineer jargon in the non-engineer guide: Jaccard, L1-L4/L5, warm cache, symlink-in-container, no_signal

*Audit finding(s) #118 · usability · lawyer-usability · CONFIRMED · [issue](https://github.com/contract-opf/playbook-engine/issues/203)*

**What:** QUICK-COMPILE, the doc pitched at people who don't write code, uses unglossed engineering terms: "keyword matching (Jaccard similarity)" (line 195), "L1–L4"/"L5" pipeline layer names (105-110), "even if `extraction_cache.jsonl` is already warm" (169, 180), "a symlink-staged corpus read inside a container" (87), and the raw spec enum `historical_stance: no_signal` (214).

**Evidence:** docs/QUICK-COMPILE.md:87, 105-110, 169, 180, 195, 214 (grep output confirms each).

**Consequence:** None of these blocks the GC outright, but each is a comprehension speed-bump that erodes the doc's 'this is for you' promise; 'Jaccard similarity' in particular signals 'engineers only' at the exact moment the doc is explaining what the stub output is missing.

**Fix:** Replace with plain equivalents: "simple word-overlap matching", "the automated stages", "already extracted once", "shortcut files whose targets the container can't see", "positions will be marked low-confidence". Keep the technical term in parentheses if desired.

### [LOW] ADOPTING's one-command skill entry assumes Claude Code and shell substitution with no on-ramp

*Audit finding(s) #119 · usability · lawyer-usability · CONFIRMED · [issue](https://github.com/contract-opf/playbook-engine/issues/204)*

**What:** ADOPTING Stage 1's first-class path is `claude "$(cat docs/prompts/create-playbook.md)"` — command substitution syntax, run from an unstated location, assuming the `claude` CLI is installed; no link explains installing Claude Code. The parenthetical alternative ("open the repo in Claude Code and say 'derive a playbook from my corpus'") is the GC-usable version but is presented second and assumes they know what 'open the repo in Claude Code' means.

**Evidence:** docs/ADOPTING.md:74-85 (command at line 81).

**Consequence:** The persona's designated best path (skill-driven, no API key) is gated behind the one syntax form ($(cat ...)) most likely to be mangled by a non-engineer pasting into the wrong place.

**Fix:** Lead with the plain-English version ("install Claude Code [link], open this folder in it, and say: 'derive a playbook from my corpus'"), and demote the shell one-liner to a note for CLI users.

### [LOW] The 'audience' interview question is double-barreled

*Audit finding(s) #120 · usability · lawyer-usability · CONFIRMED · [issue](https://github.com/contract-opf/playbook-engine/issues/205)*

**What:** Q6 asks two unrelated things in one prompt — "Does your posture change above a deal-value threshold? Who reads the output — a GC who wants terse rationale, or a junior reviewer who needs it explained?" — and the answer is templated into a single sentence labeled "Deal-size sensitivity / output audience:", so a user answering only one half produces a Posture sentence that silently drops the other.

**Evidence:** playbook_engine/posture.py:115-120 (question text) and :157 (single combined prose template).

**Consequence:** Muddled answers ("yes" — to which half?) and a Posture sentence that reads oddly; minor, but this is the interview the whole Rung 1 UX rests on.

**Fix:** Split into two question ids (deal_value_threshold, audience) or reword to one question ("Who will read the output, and does anything change above a deal-value threshold?") with an option set that covers the combinations.

### [LOW] User-facing setup question uses 'emergent playbook' and 'canonical standard' unexplained

*Audit finding(s) #121 · usability · lawyer-usability · CONFIRMED · [issue](https://github.com/contract-opf/playbook-engine/issues/206)*

**What:** The verbatim question SKILL.md tells the agent to ask a first-time user — "Do you have a baseline template file I should use as the canonical standard? (optional — I'll derive an emergent playbook if not)" — uses two internal terms ('canonical standard', 'emergent') a GC has never seen; 'emergent' is defined nowhere the user-facing flow guarantees they've read.

**Evidence:** .claude/skills/playbook-from-corpus/SKILL.md:108; also Step 0's suggested route script (SKILL.md:54-65) asks about "a validated playbook.opf.json with Evidence in it" and "out-dir" in similar register.

**Consequence:** The user either answers wrong (handing over a random form) or asks what 'emergent' means, costing the interview its crispness; template-vs-emergent materially changes output quality (SKILL.md:202-206), so a confused answer here has downstream cost.

**Fix:** Reword the scripted question: "Do you have your own standard form/template for this agreement type? If yes I'll measure every deal against it; if not, I'll build the playbook from what your negotiated history shows (a weaker but still valid baseline)."

### [LOW] ADOPTING Stage 0 '1 minute' claim hides the venv install prerequisite

*Audit finding(s) #122 · doc-vs-reality · lawyer-usability · CONFIRMED · [issue](https://github.com/contract-opf/playbook-engine/issues/207)*

**What:** ADOPTING promises "See it work (1 minute, no API key)" but the linked quickstart's Step 0 is creating a Python venv and `make install` — which on a fresh Mac pulls in Xcode command-line tools and minutes of dependency installs, none of which a GC can self-serve from the two commands shown.

**Evidence:** docs/ADOPTING.md:7 vs examples/README.md "### 0. Install (one-time)" (`python3 -m venv .venv && make install`).

**Consequence:** First-contact disappointment: the promised 1-minute wow moment is actually an environment-setup session, and the persona's trust in the doc's other time estimates (~10 min interview) drops accordingly.

**Fix:** Say "1 minute once installed (one-time setup: ~10 min, see README § Installation)" or move the wow-demo behind the Claude Code path where the agent handles setup.


## Producer / consumer seam (Contract Toaster)

The seam genuinely works end-to-end on real data — the rerun artifact loads, verifies, and composes a 3-block review prompt through Toaster's own code — but the contract has unsettled edges.

### [HIGH] Digest 'summaries' are verbatim clause text; consumer's acceptance gate FAILS on the real playbook

*Audit finding(s) #88 · doc-vs-reality · consumer-seam · CONFIRMED · [issue](https://github.com/contract-opf/contract-toaster/issues/13)*

**What:** The engine's digest presents verbatim clause text as 'summaries': text_summary is by design the first <=200 chars of the clause (verbatim), and preferred_variations if/to ship verbatim, so short clauses appear byte-for-byte in the digest. This breaks the consumer's contract on two fronts: Toaster's acceptance harness stage 4 (its 'no wholesale evidence' leak gate) reports FAIL against the engine's real current output, and Toaster's DIGEST_INTRO tells the model the digest is 'SUMMARIES, not the corpus's exact wording — do not present drafted language as quoted from the corpus' when for hundreds of entries it IS the exact wording. The engine's own digest.py docstring claims 'the digest itself never contains full_text', which is false whenever a clause is <=200 chars.

**Evidence:** Command: `.venv/bin/python scripts/opf_acceptance.py --bundle .../rerun-2026-08-23/out/playbook.opf.html` -> '4. REVIEW FAIL ... evidence full_text LEAKED into the prompt' / 'ACCEPTANCE: FAILED — REVIEW'. Measured: 705/2112 observations have text_summary == full_text, 1405 more are a verbatim prefix; digest/clauses[0]/preferred_variations[0]/to is the entire 298-char clause text of an observation. Producer design: playbook_engine/observation_builder.py:45 (_TEXT_SUMMARY_MAX=200) and :~170 ('First <= 200 chars of the clause text'); digest.py:10-11 ('the digest itself never contains full_text') vs digest.py:26-29 ('if/to language ships verbatim'). Consumer: contract-toaster/scripts/opf_acceptance.py:163-177 (leak gate), scripts/opf_prompt.py:278-290 (DIGEST_INTRO).

**Consequence:** The GC running the seam's own 'ONE command that proves the chain end to end' gets ACCEPTANCE: FAILED on a fully valid, freshly validated playbook and cannot tell a real leak from the by-design shape; meanwhile the review model is instructed not to reproduce verbatim signed language exactly where reproducing it (a preferred variation's `to`) is the point.

**Fix:** Decide the contract explicitly and align all three surfaces: either (a) the engine emits true summaries (or renames/marks the fields verbatim) and fixes the digest.py 'never contains full_text' claim, or (b) Toaster's leak gate exempts the digest's by-design verbatim fields (preferred_variations if/to; text_summary for clauses <=200 chars) and DIGEST_INTRO stops claiming the text is never exact corpus wording.

### [MEDIUM] SKILL never tells the user what a consuming application does with a Rung 0 playbook

*Audit finding(s) #90 · usability · consumer-seam · CONFIRMED · [issue](https://github.com/contract-opf/playbook-engine/issues/208)*

**What:** The skill frames evidence-only (Rung 0) as 'a complete OPF document that can ship as-is' and mentions only the engine-side ADVISORY-ONLY render marker, but never states the consumer-side consequences: in contract-toaster an empty posture makes review composition refuse unless the operator passes an explicit accept_empty_posture flag, an empty floor gives floor_judge zero invariants (no fail-closed enforcement at all, no Binding block), and the older OPF 0.2 output (<corpus>/.../full/out, no digest) cannot drive a digest-mode review at all. SKILL.md and REFERENCE.md contain zero mentions of the consumer beyond 'hand playbook.opf.html to a consuming review application'.

**Evidence:** playbook-engine/.claude/skills/playbook-from-corpus/SKILL.md:884-888 ('can ship as-is'), :1113-1121; grep for 'toaster|consumer' over SKILL.md/REFERENCE.md yields only SKILL.md:947 and :997/:1012. Consumer behavior: contract-toaster/scripts/review_knowledge.py:316-323 (accept_empty_posture refusal), scripts/review_spine.py:717-736 (floor_judge only runs `if floor_invariants`), scripts/opf_prompt.py:461-468 (PromptCompositionError for no digest.clauses — verified: full/out is opf_version 0.2 with digest_version None).

**Consequence:** A GC who stops at Rung 0 (the skill's 'legitimate endpoint') and uploads the artifact discovers only at review time that the consumer refuses or that hard-line enforcement is inert — the exact 'starved FloorJudge' coupling this seam has already produced once.

**Fix:** Add a short 'what a consuming review application sees' note at Route B / Step 7a/7b and Step 10: Rung 0 = no Posture block, no Binding block, nothing for the floor judge to enforce, and (in contract-toaster) an explicit accept-empty-posture operator flag before any review runs; 0.2-era outputs without a digest cannot drive a review.

### [MEDIUM] Floor rationale reaches the model in one Toaster path and is deliberately withheld in another; the engine packs signer PII into it

*Audit finding(s) #92 · producer · consumer-seam · CONFIRMED · [issue](https://github.com/contract-opf/playbook-engine/issues/209)*

**What:** floor_judge.py deliberately never sends an invariant's rationale to the model or logs ('may carry confidential contract or playbook substance'), but opf_prompt._binding_block renders the same rationale verbatim into the model-facing Binding block. The engine, having no structured attribution field in the OPF floor shape, writes sign-off attribution — a person's name and date ('Hand-authored and signed by the legal owner (Marc Mandel, GC), 2026-08-21...') — into rationale, so that attribution ships into every review prompt of the consumer.

**Evidence:** contract-toaster/scripts/floor_judge.py:46-53 and :204-206 ('rationale is accepted but never sent to the model or echoed anywhere') vs scripts/opf_prompt.py:599-601 (`line += f" (Rationale: {rationale})"`). Real artifact: rerun-2026-08-23/out/playbook.opf.json floor.invariants[0].rationale begins 'Hand-authored and signed by the legal owner (Marc Mandel, GC), 2026-08-21.'; grep 'signed_by|signed_at|attribution' over spec/playbook.schema-0.3.json: no field.

**Consequence:** Two components of the same consumer apply opposite confidentiality postures to one field, and internal attribution metadata (who signed, when) is exposed to the model and to anything that echoes prompt content — inconsistent with the repo's own no-substance-in-prompt-surfaces discipline for floor text.

**Fix:** Short term: keep attribution out of rationale (engine `floor sign` could record it in posture.generation-style metadata or a x_ extension) or stop rendering rationale in the Binding block. Long term: give floor.invariants structured signed_by/signed_at in the next opf_version and render only the legal rationale.

### [MEDIUM] Engine's conformance vectors — built for this exact seam — are not consumed by any Toaster test

*Audit finding(s) #91 · consumer · consumer-seam · CONFIRMED · [issue](https://github.com/contract-opf/contract-toaster/issues/14)*

**What:** The engine's spec/conformance suite (added 2026-08-22, pushed to origin) explicitly names Contract Toaster's hand-maintained ports as the drift class it defends against, yet no Toaster test loads those vectors; Toaster's canonicalize protection is a self-pinned golden hash of its own fixture, and its only true upstream-drift detector (schema-sync layer 3) covers schema files only, not the canonicalize port.

**Evidence:** playbook-engine/spec/CHANGELOG.md 2026-08-22 entry ('the drift class it defends against is Contract Toaster's hand-maintained source-level ports'); grep 'conformance' over contract-toaster/tests returns no engine-vector consumer; tests/test_opf_canonicalize.py:38 pins GOLDEN_CONTENT_HASH over its own fixture. Verified the vectors pass today: all 13 vectors' content_hash + section_digests match through contract-toaster/scripts/opf_canonicalize.py (0 failures).

**Consequence:** The next canonicalization change (unicode normalization, float formatting, exclusion-list edit) breaks hash verification of every uploaded playbook silently until an upload fails in production, despite a mechanical check now existing upstream.

**Fix:** Add a sibling-checkout/local-only test (same skip-cleanly pattern as test_opf_schema_sync layer 3, honoring $PLAYBOOK_ENGINE_REPO) that runs every spec/conformance vector through opf_canonicalize, and note the vector suite in playbooks/opf/README.md's re-vendoring procedure.

### [MEDIUM] Acceptance leak check counts trivial substrings (3-char full_text)

*Audit finding(s) #89 · consumer · consumer-seam · CONFIRMED · [issue](https://github.com/contract-opf/contract-toaster/issues/15)*

**What:** Toaster's stage-4 leak check treats ANY observation full_text found as a substring of the joined prompt as a leak, with no minimum length, so degenerate fragments like '1 6', 'the facility', and 'indemnification' count as leaked evidence.

**Evidence:** contract-toaster/scripts/opf_acceptance.py:165-170 (`if o.get('full_text') and o['full_text'] in joined`). Measured against rerun-2026-08-23: 464 'leaks' bucketed as {'<25 chars': 131, '25-99': 52, '100-199': 256, '>=200': 25} — the <25 bucket includes full_text '1 6' (3 chars).

**Consequence:** Even after the verbatim-summary contract is settled, the gate will keep failing spuriously on single-word matches, training the operator (a GC, not an engineer) to distrust or ignore a FAIL.

**Fix:** Add a minimum-length threshold (e.g. only count full_text >= ~50 chars) and/or exclude the digest's by-design verbatim fields from the comparison surface.

### [MEDIUM] 173 degenerate observations (page-number and single-word fragments) in the real evidence section

*Audit finding(s) #96 · producer · consumer-seam · CONFIRMED · [issue](https://github.com/contract-opf/playbook-engine/issues/210)*

**What:** The current real playbook carries 173 of 2112 observed_positions whose full_text is under 25 characters — fragments like '1 6', '1 7' (page artifacts), 'the facility', 'indemnification' — recorded as our_paper observations with citations.

**Evidence:** Measured over rerun-2026-08-23/out/playbook.opf.json: 173/2112 observations with len(full_text) < 25; examples ('clause.accreditation_licensure', '1 6', 'our_paper'), ('clause.amendments', 'indemnification', 'our_paper').

**Consequence:** Fragments inflate precedent counts (n/band weighting), surface as meaningless drill-down results through Toaster's lookup_clause_evidence, and are the main source of the leak-gate's <25-char noise; a GC drilling into 'the exact language we signed' can land on '1 6'.

**Fix:** Add a minimum-viable-observation guard in mining/segmentation (or a lint in the skill's checkpoint step) that quarantines sub-sentence fragments instead of admitting them as observations; the AAR's blank-our_standard pattern (aar.py:838) is a precedent for surfacing it.

### [LOW] Digest meets its 40K budget only under the repo's canonical-form heuristic; the on-disk file measures ~19% over

*Audit finding(s) #65 · doc-vs-reality · corpus-reality · CONFIRMED · [issue](https://github.com/contract-opf/playbook-engine/issues/211)*

**What:** The digest's enforced ~40K-token budget is computed as canonicalized-chars/4 (36,763 for the rerun), but the pretty-printed playbook.digest.json on disk is 190,077 chars → 47.5K by the same chars/4 rule, and a real tokenizer on JSON typically lands higher still — so a consumer measuring the artifact it actually loads will see the 'target ~40K' claim exceeded.

**Evidence:** wc -c <corpus>/eiaa-playbook-output/rerun-2026-08-23/out/playbook.digest.json → 190,206 bytes; canonicalize() of the same content → 147,054 chars → 36,763 est tokens (matches report.json artifacts.digest_token_estimate = 36763); budget constant DIGEST_TOKEN_BUDGET = 40_000 at playbook_engine/digest.py:53, estimate at :270-273.

**Consequence:** The toaster (or any consumer) that budgets model context from the file it reads will under-provision by ~25-30% relative to the file's real size; the '~40K' promise silently depends on the consumer re-canonicalizing.

**Fix:** Write the standalone digest sidecar compact (no indentation) so file size ≈ canonical size, or state in SKILL.md Step 10 that the ~40K figure is for the canonical (minified) form.

### [LOW] Toaster's load-bearing comments still say the real playbook ships posture:{} / floor:{} — no longer true

*Audit finding(s) #93 · doc-vs-reality · consumer-seam · CONFIRMED · [issue](https://github.com/contract-opf/contract-toaster/issues/16)*

**What:** Present-tense claims across Toaster ('the real playbook ships BOTH posture == {} and floor == {}', 'that is what EVERY production OPF review composes today', 'the real and public OPF playbooks ship posture: {} on purpose') are stale: both current engine artifacts (rerun-2026-08-23 and the in-production full-run.bak-2026-08-23) ship a populated posture.system_prompt and 2 signed floor invariants, and compose a 3-block prompt with a Binding block through Toaster's own code.

**Evidence:** contract-toaster/scripts/opf_prompt.py:18-19, :252, :540, :563-575; scripts/review_knowledge.py:46, :304; backend/src/pipeline_runner.py:646. Measured: both playbook.opf.json files show posture keys ['system_prompt','version','generation'] and 2 floor invariants; compose_opf_system_blocks returns 3 blocks (Posture + Binding + Digest) with only the no-policy 'guidance' omission recorded.

**Consequence:** Future maintainers (and the #479-DECISION reasoning threaded through these comments) design against an empty-is-the-norm producer state that no longer exists — e.g. treating accept_empty_posture as the routine day-one path rather than the exception.

**Fix:** Sweep these comments to past tense ('shipped until 2026-08', 'may ship') and note that engine-authored Posture/Floor now populate both sections in the current real artifact.

### [LOW] Toaster README points OPF playbook authors at the legacy v1 schema

*Audit finding(s) #94 · consumer · consumer-seam · CONFIRMED · [issue](https://github.com/contract-opf/contract-toaster/issues/17)*

**What:** README.md tells adopters to 'write an OPF-native JSON document against playbooks/schema.json' — but playbooks/schema.json is the legacy v1 'Contract Review Playbook' schema (topics/hard_rejections/decision_rubric), not OPF; the OPF schemas live at playbooks/opf/playbook.schema-0.{2,3}.json.

**Evidence:** contract-toaster/README.md:195 (also :78 pairs 'OPF ... schema' with 'playbooks/schema.json'); `python3 -c` over playbooks/schema.json shows title 'Contract Review Playbook' with top-level props ['playbook','general_principles','decision_rubric','topics','hard_rejections',...].

**Consequence:** An OSS adopter authoring their first playbook validates against the wrong (legacy) schema and produces a document opf_load rejects at upload ('/opf_version: missing or unsupported version').

**Fix:** Point authoring guidance at playbooks/opf/playbook.schema-0.3.json (and opf_load's two accepted upload forms); reserve playbooks/schema.json references for the legacy v1 path.

### [LOW] Consumer reads top-level perspective/de_minimis; producer never emits them in the real artifact

*Audit finding(s) #97 · producer · consumer-seam · CONFIRMED · [issue](https://github.com/contract-opf/playbook-engine/issues/212)*

**What:** Toaster's Context block renders top-level `perspective` and `de_minimis` and its risk rendering states 'direction is from OUR perspective', but the real playbook carries neither key (perspective is optional engine config the eiaa run omits), so the Context block never composes and the model learns whose perspective 'worse' means only from posture prose.

**Evidence:** contract-toaster/scripts/opf_prompt.py:650-661 (_context_block) and :342 ('direction is from OUR perspective'); rerun-2026-08-23/out/playbook.opf.json top-level keys contain no 'perspective'/'de_minimis'; playbook-engine/playbook_engine/config.py:24-35 (perspective optional, omit-and-drop).

**Consequence:** A consumer-side affordance silently never fires on real data; on a playbook shared outside the authoring org (the skill's Step 11 public path), risk_delta direction is ambiguous to the reviewing model.

**Fix:** Have the skill's config template (or Step 3 config authoring) encourage populating `perspective:` for any playbook that will be consumed downstream; alternatively Toaster could record the missing-context case in its omission ledger like the other absent blocks.
