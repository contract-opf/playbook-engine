"""Verification tests for the playbook-from-corpus companion skill.

Parses the SKILL.md frontmatter, asserts required sections exist, and
confirms every referenced subcommand is present in the CLI.

No browser, no network, no real corpus.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent
SKILL_DIR = REPO_ROOT / ".claude" / "skills" / "playbook-from-corpus"
SKILL_MD = SKILL_DIR / "SKILL.md"
REFERENCE_MD = SKILL_DIR / "REFERENCE.md"

# Old, pre-#137 location — must no longer exist (Claude Code only
# auto-discovers project skills under .claude/skills/<name>/SKILL.md).
OLD_SKILL_DIR = REPO_ROOT / ".claude" / "playbook-from-corpus"

# The other top-level workflow doc (issue #133, moved under docs/ by #174):
# docs/QUICK-COMPILE.md is the mine/project no-code guide, and
# .claude/skills/playbook-from-corpus/SKILL.md is the companion skill (cites
# flags across several subcommands, not just mine). Both cite `--flag`s in
# prose/code fences — if any drifts (e.g. citing a flag that was renamed or
# removed, as docs/QUICK-COMPILE.md once did with --no-resume vs. the actual
# --no-cache), this catches it instead of a user hitting "no such option"
# mid-workflow.
ROOT_SKILL_MD = REPO_ROOT / "docs" / "QUICK-COMPILE.md"
# SKILL_MD (.claude/skills/playbook-from-corpus/SKILL.md) is defined above.

_FLAG_PATTERN = re.compile(r"--[a-z][a-z-]*")

# Flags belonging to non-playbook tools cited in verification snippets
# (e.g. `git status --porcelain`) — not part of the playbook CLI surface, so
# excluded from the drift check below rather than misread as a playbook flag.
# `--rm` belongs to `docker run` (issue #149: SKILL.md wraps every engine
# command in the documented `docker run` / `make docker-run` invocation).
_NON_PLAYBOOK_FLAGS = {"--porcelain", "--rm"}

# Subcommands the skill references — each must appear in `playbook --help`.
REQUIRED_SUBCOMMANDS = [
    "stage",
    "lint-corpus",
    "mine",
    "judge",
    "judge-apply",
    "project",
    "validate",
    "view",
    "apply-overrides",
    "install-steps",
    "inspect",
    # Step 7a/7b of SKILL.md — both are command GROUPS, handled like `view`
    # in `_all_subcommand_help_text` below (a group's own --help lists only
    # subcommand names, so flags such as `posture interview --answers-file`
    # only appear in the SUBcommand's help).
    "posture",
    "floor",
]

# Commands retired by issue #239 — the skill must not send anyone to them.
RETIRED_SUBCOMMANDS = [
    "curate",
    "digest",
    "judge-migrate",
    "publish",
    "render-prompt",
    "report",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_frontmatter(path: Path) -> tuple[dict, str]:
    """Parse YAML frontmatter from a Markdown file.

    Returns (frontmatter_dict, body_text).  Raises ValueError if the file
    does not begin with a ``---`` delimiter.
    """
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        raise ValueError(f"{path} does not start with a YAML frontmatter delimiter '---'")
    # Find the closing ---
    end = text.index("---", 3)
    raw_fm = text[3:end].strip()
    body = text[end + 3 :].strip()
    fm = yaml.safe_load(raw_fm)
    return fm, body


def _playbook_help() -> str:
    """Run ``playbook --help`` and return its output."""
    result = subprocess.run(
        [sys.executable, "-m", "playbook_engine.cli", "--help"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not result.stdout:
        # Fall back to invoking the installed entry-point directly
        result = subprocess.run(
            ["playbook", "--help"],
            capture_output=True,
            text=True,
        )
    return result.stdout + result.stderr


def _subcommand_help(subcmd: str) -> str:
    """Run ``playbook <subcmd> --help`` and return its output."""
    result = subprocess.run(
        ["playbook", subcmd, "--help"],
        capture_output=True,
        text=True,
    )
    return result.stdout + result.stderr


def _extract_cli_flags(text: str) -> set[str]:
    """Return the set of ``--flag``-shaped tokens cited anywhere in *text*."""
    return set(_FLAG_PATTERN.findall(text))


def _all_subcommand_help_text() -> str:
    """Concatenated ``--help`` output for every playbook subcommand.

    Some workflow docs (e.g. .claude/skills/playbook-from-corpus/SKILL.md) cite
    flags across several subcommands, not just ``mine`` — checking a
    citation against this combined surface still catches a genuinely
    nonexistent flag (the --no-resume incident) without false-flagging a
    real flag that just belongs to a different subcommand. ``view`` is a
    command GROUP whose own --help lists only subcommand names, so its
    subcommands' help is included too.
    """
    subcmds = list(REQUIRED_SUBCOMMANDS)
    texts = [_subcommand_help(subcmd) for subcmd in subcmds]
    for group, group_subs in (
        ("view", ("bundle",)),
        ("posture", ("questions", "interview")),
        ("floor", ("propose", "sign")),
    ):
        for group_sub in group_subs:
            result = subprocess.run(
                ["playbook", group, group_sub, "--help"],
                capture_output=True,
                text=True,
            )
            texts.append(result.stdout + result.stderr)
    return "\n".join(texts)


# ---------------------------------------------------------------------------
# SKILL.md — file existence
# ---------------------------------------------------------------------------


def test_skill_md_exists() -> None:
    """SKILL.md must exist in .claude/skills/playbook-from-corpus/."""
    assert SKILL_MD.exists(), f"Missing: {SKILL_MD}"


def test_old_skill_location_no_longer_exists() -> None:
    """The pre-#137 path (.claude/playbook-from-corpus/) must be gone.

    Claude Code only auto-discovers project skills under the convention
    .claude/skills/<name>/SKILL.md — the old location was silently never
    offered in a session.
    """
    assert not OLD_SKILL_DIR.exists(), f"Old skill location still present: {OLD_SKILL_DIR}"


# ---------------------------------------------------------------------------
# SKILL.md — frontmatter
# ---------------------------------------------------------------------------


def test_frontmatter_parses_as_valid_yaml() -> None:
    """SKILL.md frontmatter must be valid YAML."""
    fm, _ = _parse_frontmatter(SKILL_MD)
    assert isinstance(fm, dict)


def test_frontmatter_name_is_playbook_from_corpus() -> None:
    """frontmatter `name` must be exactly 'playbook-from-corpus'."""
    fm, _ = _parse_frontmatter(SKILL_MD)
    assert fm.get("name") == "playbook-from-corpus"


def test_frontmatter_description_is_non_empty() -> None:
    """frontmatter `description` must be a non-empty string."""
    fm, _ = _parse_frontmatter(SKILL_MD)
    desc = fm.get("description", "")
    assert isinstance(desc, str) and desc.strip(), "description must be a non-empty string"


def test_frontmatter_description_max_1024_chars() -> None:
    """frontmatter `description` must not exceed 1024 characters."""
    fm, _ = _parse_frontmatter(SKILL_MD)
    desc = fm.get("description", "")
    assert len(desc) <= 1024, f"description is {len(desc)} chars (max 1024)"


def test_frontmatter_description_contains_use_when() -> None:
    """frontmatter `description` must contain the phrase 'Use when'."""
    fm, _ = _parse_frontmatter(SKILL_MD)
    assert "Use when" in fm.get("description", ""), "description must contain 'Use when'"


# ---------------------------------------------------------------------------
# SKILL.md — required body sections
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "section_heading",
    [
        "Ordered pipeline",
        "Stage",
        "Lint",
        "Mine",
        "Checkpoint",
        "Unattended judge-drain loop",
        "Project",
        "Validate",
        "Inspect",
        "View",
        "Guardrails",
    ],
)
def test_skill_body_contains_section(section_heading: str) -> None:
    """SKILL.md body must document each pipeline phase and the guardrails."""
    _, body = _parse_frontmatter(SKILL_MD)
    assert section_heading.lower() in body.lower(), (
        f"SKILL.md body missing section or keyword: '{section_heading}'"
    )


def test_skill_body_references_judge_apply() -> None:
    """SKILL.md body must reference the judge-apply subcommand."""
    _, body = _parse_frontmatter(SKILL_MD)
    assert "judge-apply" in body


def test_skill_body_names_no_retired_command() -> None:
    """SKILL.md must not send anyone to a command issue #239 retired.

    The retired surfaces (curation, the review HTML and its feedback loop,
    the after-action report, render-prompt, publish, the digest sidecar,
    judge-migrate and the deviation judge) left the corpus -> playbook ->
    toaster path; a skill step that still tells the agent to run one is a
    step that fails at the CLI.
    """
    _, body = _parse_frontmatter(SKILL_MD)
    for retired in (
        "playbook curate",
        "playbook digest",
        "playbook publish",
        "playbook report",
        "render-prompt",
        "view render",
        "view apply",
        "judge-migrate",
        "--with-deviation-judge",
        "playbook.review.html",
        "playbook.digest.json",
        "feedback.json",
        "x_judgments",
    ):
        assert retired not in body, f"SKILL.md still cites the retired surface {retired!r}"
    assert "Step 7c" not in body and "## Step 11" not in body and "### Step 10" not in body


def test_skill_step_9_is_the_page_the_residue_check_and_the_closing() -> None:
    """The one human-readable artifact is `view bundle`'s index.html; the mandatory
    residue check scans the canonical OPF and the page (the digest is a section of
    the OPF, not a file); and the run ends open -> paths -> install steps ->
    optional-review line, in that order (issue #241)."""
    _, body = _parse_frontmatter(SKILL_MD)
    step_9 = body.split("### Step 9 —", 1)[1].split("## Guardrails", 1)[0]
    assert 'ARGS="view bundle /work/out"' in step_9
    assert "find_residue" in step_9
    assert "playbook.opf.json" in step_9 and "index.html" in step_9
    assert "playbook.opf.html" not in body
    closing = step_9.split("**Close — always, in this order**", 1)[1]
    positions = [
        closing.index(marker)
        for marker in (
            "# (a) open the page",
            'open "$HOST_OUT/index.html"',
            "xdg-open",
            'start ""',
            "# (b) print the absolute paths",
            "# (c) the toaster install steps",
            'ARGS="install-steps --file $HOST_OUT/playbook.opf.json"',
            "# (d) one line",
            "is optional",
        )
    ]
    assert positions == sorted(positions), (
        "closing must be open, paths, install steps, optional line"
    )
    assert "overrides.json" in step_9 and "apply-overrides" in step_9


def test_skill_opening_is_brief_and_every_choice_is_a_button() -> None:
    """The standard flow: a short opening, multiple-choice questions with the
    recommended option first (AskUserQuestion), and the closing beat (issue #241)."""
    _, body = _parse_frontmatter(SKILL_MD)
    flow = body.split("## The standard flow", 1)[1].split("\n---\n", 1)[0]
    opening = flow.split("**1. Opening", 1)[1].split("**2. Questions are buttons.**", 1)[0]
    quoted = [line for line in opening.splitlines() if line.startswith(">")]
    assert 0 < len(quoted) <= 6, "the opening says what will happen in at most six lines"
    for needed in (
        "folder of agreements",
        "standard form",
        "agreement type",
        'which party is "us"',
    ):
        assert needed in opening
    assert "AskUserQuestion" in flow and "recommended option first" in flow
    for question in ("route", "corpus folder", "agreement type", "perspective party"):
        assert question in flow.lower().replace("perspective party", "perspective party")
    # the route question and the setup questions are choices, not prose prompts
    step_0 = body.split("## Step 0", 1)[1].split("## Interactive setup", 1)[0]
    assert "AskUserQuestion" in step_0 and "recommended" in step_0
    setup = body.split("## Interactive setup", 1)[1].split("## Derive party names", 1)[0]
    assert "AskUserQuestion" in setup
    # a route A run never waits on the posture interview or the floor
    assert "Skip 7a/7b" in step_0
    assert "OPTIONAL" in body.split("### Step 7a", 1)[1].split("\n", 1)[0]


def test_skill_names_the_overrides_and_install_commands_the_engine_has() -> None:
    _, body = _parse_frontmatter(SKILL_MD)
    for command in ("apply-overrides", "install-steps"):
        assert command in body
        assert command in _playbook_help()


def test_skill_body_documents_q5_pre_rejection() -> None:
    """SKILL.md must disclose that Q5 (flexible_clauses) pre-marks candidates.

    Regression test for issue #134 (skill QA audit finding #86):
    ``flexible_clauses`` is described as "binds nothing" in Step 7a's
    question-ordering table, but a Q5 answer auto-marks matching REVERSAL
    candidates ``decision: rejected`` in floor.candidates.json (issue #105,
    ``floor_candidates._Q5_REJECTION_COMMENT``). Step 7a must tell the reader
    this happens so a reviewer isn't surprised by pre-rejected rows attributed
    to their own answer.
    """
    _, body = _parse_frontmatter(SKILL_MD)
    flat = " ".join(body.split())
    assert '"decision": "rejected"' in flat and "flexible_clauses" in flat, (
        "SKILL.md must mention the pre-rejection that Q5 (flexible_clauses) "
        "writes onto matching Floor candidates (issue #134)"
    )
    assert "playbook floor sign" in flat


def test_skill_body_documents_full_drain_invariant() -> None:
    """SKILL.md must document that pending.jsonl must be empty before project."""
    _, body = _parse_frontmatter(SKILL_MD)
    assert "pending" in body.lower() and "empty" in body.lower(), (
        "SKILL.md must document the full-drain invariant (pending.jsonl empty before project)"
    )


def test_skill_step4_sanity_check_mentions_quarantine_json() -> None:
    """SKILL.md's Step 4 checkpoint list must tell the agent to read
    out/quarantine.json (issue #199): it is a real ``mine`` output — a
    document quarantine.json lists is one the playbook silently lost — but
    was previously mentioned only at Step 6, after judgment effort was
    already spent.
    """
    _, body = _parse_frontmatter(SKILL_MD)
    step4 = body.split("### Step 4 —", 1)[1].split("### Step 5 —", 1)[0]
    assert "quarantine.json" in step4, (
        "SKILL.md Step 4 (checkpoint before judging) must mention out/quarantine.json"
    )


def test_skill_step7_mentions_coherence_flags_json() -> None:
    """SKILL.md's Step 7 (``playbook project``) must mention
    out/coherence_flags.json (issue #199) — a real output of ``project``
    (always written, even empty) that no doc previously named.
    """
    _, body = _parse_frontmatter(SKILL_MD)
    step7 = body.split("### Step 7 —", 1)[1].split("### Step 7a —", 1)[0]
    assert "coherence_flags.json" in step7, (
        "SKILL.md Step 7 (project the playbook) must mention out/coherence_flags.json"
    )


def test_quick_compile_intermediates_table_lists_quarantine_and_coherence_flags() -> None:
    """QUICK-COMPILE.md's Step 4 intermediates table must list both
    quarantine.json and coherence_flags.json (issue #199) — both are real
    files in a mine+project out-dir that the table previously omitted.
    """
    content = ROOT_SKILL_MD.read_text(encoding="utf-8")
    table = content.split("## Step 4 — Review the intermediates", 1)[1].split(
        "## Re-running after changes", 1
    )[0]
    assert "`quarantine.json`" in table, (
        "QUICK-COMPILE.md's Step 4 intermediates table must list quarantine.json"
    )
    assert "`coherence_flags.json`" in table, (
        "QUICK-COMPILE.md's Step 4 intermediates table must list coherence_flags.json"
    )


def test_skill_body_references_no_gitignored_path() -> None:
    """SKILL.md must not point readers at a gitignored path (issue #135).

    The skill previously referenced ``ignore/DERIVATION-HANDOFF.md`` in its
    Reference section — ``ignore/`` is gitignored as local, disposable scratch
    (see .gitignore: "Local scratch — disposable artifacts, staging adapters,
    run configs"), so anyone cloning the public repo got a dangling reference
    to a file that only ever existed locally. Checked against the *scratch*
    gitignore patterns specifically (not every gitignored dir — ``out/`` and
    ``normalized/`` are also gitignored but are the engine's own per-run
    output dirs, legitimately referenced throughout this doc).
    """
    _, body = _parse_frontmatter(SKILL_MD)
    gitignore_text = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "ignore/" in gitignore_text, "expected .gitignore to still list 'ignore/' as scratch"

    scratch_dirs = ["ignore/"]
    for scratch in scratch_dirs:
        assert scratch not in body, (
            f"SKILL.md references gitignored scratch path '{scratch}' — anyone cloning "
            "the public repo gets a dangling reference to a file that isn't tracked"
        )


def test_skill_body_does_not_understate_sensitive_artifacts() -> None:
    """SKILL.md must not claim alias_map.json/entity_registry.json are the
    *only* durable sensitive artifacts under $OUT (issue #139, audit finding
    #48).

    ``judge/pending.jsonl`` and ``my-verdicts.jsonl`` are written from the
    RAW, pre-pseudonymization source — they exist for a human to review and
    answer BEFORE the born-safe pseudonymization pass ever runs, so there is
    no aliased form to fall back to — and carry raw counterparty names
    indefinitely; the skill must call those two out as sensitive.
    ``normalized/<document_id>/`` is deliberately NOT on that list: issue
    #139's code fix made pipeline.py stale-clear and rewrite it under the
    ALIASED document_id after the pseudonymization pass (mirroring
    trail/'s treatment), so — unlike the two files above — it is safe.
    """
    _, body = _parse_frontmatter(SKILL_MD)
    assert "are the only durable sensitive artifacts" not in body, (
        "SKILL.md must not claim alias_map.json/entity_registry.json are the "
        "*only* durable sensitive artifacts under $OUT — the judge "
        "pending/my-verdicts files carry raw counterparty names too "
        "(issue #139)"
    )
    assert "pending.jsonl" in body and "my-verdicts.jsonl" in body, (
        "SKILL.md's born-safe sidecar note must call out judge/pending.jsonl "
        "and my-verdicts.jsonl as sensitive artifacts (issue #139)"
    )
    # Whitespace-normalized: the prose wraps at ~100 cols, so a multi-word
    # phrase can straddle a hard newline in the raw markdown.
    flat_body = " ".join(body.split())
    assert "not on this list" in flat_body or "is safe to share" in flat_body, (
        "SKILL.md must affirmatively state that normalized/ is no longer "
        "one of the sensitive artifacts, now that issue #139's code fix "
        "aliases it (otherwise a reader has no way to tell the omission "
        "from an oversight)"
    )


# ---------------------------------------------------------------------------
# CLI — subcommand presence
# ---------------------------------------------------------------------------


def test_playbook_help_exits_successfully() -> None:
    """``playbook --help`` must exit 0."""
    result = subprocess.run(["playbook", "--help"], capture_output=True, text=True)
    assert result.returncode == 0, f"playbook --help exited {result.returncode}: {result.stderr}"


@pytest.mark.parametrize("subcmd", REQUIRED_SUBCOMMANDS)
def test_subcommand_present_in_playbook_help(subcmd: str) -> None:
    """Each subcommand referenced by the skill must appear in ``playbook --help``."""
    help_text = _playbook_help()
    assert subcmd in help_text, f"Subcommand '{subcmd}' not found in `playbook --help` output"


@pytest.mark.parametrize("subcmd", RETIRED_SUBCOMMANDS)
def test_retired_subcommand_is_absent_from_playbook_help(subcmd: str) -> None:
    """Issue #239: each retired command is gone from `playbook --help`."""
    names = {
        line.split()[0]
        for line in _playbook_help().split("Commands:", 1)[1].splitlines()
        if line.strip()
    }
    assert subcmd not in names, f"`playbook {subcmd}` should have been retired"


@pytest.mark.parametrize("subcmd", REQUIRED_SUBCOMMANDS)
def test_subcommand_help_exits_successfully(subcmd: str) -> None:
    """``playbook <subcmd> --help`` must exit 0 for every referenced subcommand."""
    result = subprocess.run(
        ["playbook", subcmd, "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"`playbook {subcmd} --help` exited {result.returncode}:\n{result.stderr}"
    )


# ---------------------------------------------------------------------------
# Workflow docs — cited CLI flags must actually exist (issue #133)
# ---------------------------------------------------------------------------
#
# docs/QUICK-COMPILE.md (formerly root SKILL.md) once instructed
# `playbook compile --no-resume`, a flag that was never wired up (`resume`
# survives only as a deprecated pipeline.py kwarg) — the real flag is
# `--no-cache`. None of these docs had its own test coverage, so the drift
# shipped silently. Checked against the combined --help output of every
# playbook subcommand (not just `mine`), since some docs cite flags
# belonging to other subcommands too.


@pytest.mark.parametrize(
    "doc_path",
    [ROOT_SKILL_MD, SKILL_MD],
    ids=[
        "docs-QUICK-COMPILE.md",
        "claude-playbook-from-corpus-SKILL.md",
    ],
)
def test_doc_cited_flags_exist_in_subcommand_help(doc_path: Path) -> None:
    """Every ``--flag`` cited in a workflow doc must exist on some playbook subcommand."""
    assert doc_path.exists(), f"Missing: {doc_path}"
    cited_flags = _extract_cli_flags(doc_path.read_text(encoding="utf-8")) - _NON_PLAYBOOK_FLAGS
    assert cited_flags, f"Expected at least one --flag citation in {doc_path}"

    help_text = _all_subcommand_help_text()

    missing = sorted(flag for flag in cited_flags if flag not in help_text)
    assert not missing, (
        f"{doc_path.name} cites flag(s) not found in any `playbook <subcmd> --help`: {missing}"
    )


# ---------------------------------------------------------------------------
# SKILL.md — Docker/make wrapping (issue #149)
# ---------------------------------------------------------------------------
#
# Q10 (decided 2026-07-09): Docker is the engine runtime; this skill is the
# orchestration layer and wraps each engine command in the documented
# `docker run`/`make` invocation, with the corpus and out directories as
# shared volumes so the pending/verdicts JSONL loop works across the
# container boundary.


def test_skill_body_references_docker_run_invocation() -> None:
    """SKILL.md body must reference the documented `docker run` invocation."""
    _, body = _parse_frontmatter(SKILL_MD)
    assert "docker run" in body, (
        "SKILL.md must wrap engine commands in the documented `docker run` invocation"
    )


def test_skill_body_references_make_docker_run_invocation() -> None:
    """SKILL.md body must reference the `make docker-run` convenience wrapper."""
    _, body = _parse_frontmatter(SKILL_MD)
    assert "make docker-run" in body, (
        "SKILL.md must wrap engine commands in the documented `make docker-run` invocation"
    )


def test_skill_body_documents_shared_volumes_for_jsonl_loop() -> None:
    """SKILL.md must explain that corpus/out are shared volumes for the JSONL loop.

    The judge/judge-apply round-trip writes pending.jsonl (container) and
    verdicts.jsonl (agent, host-side) — both sides must agree on where those
    files live across the container boundary, or the loop silently breaks.
    """
    _, body = _parse_frontmatter(SKILL_MD)
    assert "$OUT" in body, "SKILL.md must reference the $OUT shared-volume convention"
    assert "verdicts" in body.lower() and "container" in body.lower(), (
        "SKILL.md must explain that verdict/feedback files must be written under the "
        "shared $OUT volume to be visible inside the container"
    )


def test_skill_docker_wrapped_commands_use_container_paths() -> None:
    """Every `make docker-run ... ARGS="<subcmd> ..."` block must use container paths.

    Paths inside ARGS get translated by the -v mounts, so they must reference
    /work/corpus or /work/out, never the bare host-relative ./corpus or ./out
    (that would be a real path the container can't see).
    """
    _, body = _parse_frontmatter(SKILL_MD)
    args_blocks = re.findall(r'ARGS="([^"]+)"', body)
    assert args_blocks, 'expected at least one make docker-run ARGS="..." block in SKILL.md'
    for block in args_blocks:
        assert "./corpus" not in block and "./out" not in block, (
            f"ARGS block uses a host-relative path instead of a container path: {block!r}"
        )


# ---------------------------------------------------------------------------
# REFERENCE.md — existence and required content
# ---------------------------------------------------------------------------


def test_reference_md_exists() -> None:
    """REFERENCE.md must exist alongside SKILL.md."""
    assert REFERENCE_MD.exists(), f"Missing: {REFERENCE_MD}"


@pytest.mark.parametrize(
    "keyword",
    [
        # Judge guidance sections
        "Classification",
        "Deviation",
        "Provenance",
        # Guardrails
        "Guardrail",
        "fabricat",  # "fabrication" or "fabricate"
        "needs_review",
        # Done-criteria
        "done-criteria",
        "pending.jsonl",
        "validate",
    ],
)
def test_reference_md_contains_required_content(keyword: str) -> None:
    """REFERENCE.md must cover judge guidance, guardrails, and done-criteria."""
    content = REFERENCE_MD.read_text(encoding="utf-8")
    assert keyword.lower() in content.lower(), f"REFERENCE.md missing required content: '{keyword}'"


def test_reference_md_done_criteria_mentions_validate() -> None:
    """REFERENCE.md done-criteria must reference `playbook validate`."""
    content = REFERENCE_MD.read_text(encoding="utf-8")
    assert "playbook validate" in content


def test_reference_md_done_criteria_mentions_empty_pending() -> None:
    """REFERENCE.md done-criteria must require pending.jsonl to be empty."""
    content = REFERENCE_MD.read_text(encoding="utf-8")
    assert "pending.jsonl" in content and "empty" in content.lower()


def test_reference_md_done_criterion_3_checks_the_bundle() -> None:
    """Criterion 3's file-existence check must name the page.

    SKILL.md Step 9 declares `index.html` the one human-readable artifact — a
    run that stops after `validate` is not actually done. Regression for issue #176: assert the criterion-3 code
    block's own `test -f` names the artifact, not just that the string appears
    somewhere in the file. The retired report, review HTML and digest sidecar
    (issue #239) must not be required.
    """
    content = REFERENCE_MD.read_text(encoding="utf-8")
    match = re.search(
        r"3\. \*\*The packaged artifact.*?```bash\n(.*?)\n\s*```",
        content,
        re.DOTALL,
    )
    assert match, "could not locate criterion 3's code block in REFERENCE.md"
    criterion_3_block = match.group(1)
    assert "index.html" in criterion_3_block, "criterion 3's test command must check for index.html"
    for retired in ("report.md", "playbook.review.html", "playbook.digest.json"):
        assert retired not in criterion_3_block, f"criterion 3 still requires {retired}"


def test_reference_md_guardrails_flag_unknown_aliases() -> None:
    """REFERENCE.md guardrails must address unknown entity aliases."""
    content = REFERENCE_MD.read_text(encoding="utf-8")
    assert "alias" in content.lower() or "unknown" in content.lower()


def test_reference_md_guardrails_no_fabrication() -> None:
    """REFERENCE.md guardrails must explicitly prohibit fabricating legal content."""
    content = REFERENCE_MD.read_text(encoding="utf-8")
    assert "fabricat" in content.lower()  # fabricate / fabrication


def test_reference_md_verdict_format_documented() -> None:
    """REFERENCE.md must document the verdict JSON format (key + verdict)."""
    content = REFERENCE_MD.read_text(encoding="utf-8")
    assert '"key"' in content or "verdict" in content.lower()


def test_reference_md_classify_threshold_matches_ambiguity_threshold() -> None:
    """REFERENCE.md's classify needs_review threshold must match the engine's
    real behavioral cutoff (issue #163).

    ``ClauseClassification.is_ambiguous`` — the only engine mechanism a
    stored classify confidence actually trips — fires below
    ``clause_classifier.AMBIGUITY_THRESHOLD``. REFERENCE.md used to instruct
    the judge to flag ``needs_review`` below 0.6, a number that corresponds
    to no engine constant at all; a verdict scored at 0.62 would look "fine"
    per the doc while the engine's own is_ambiguous check already disagreed.
    """
    from playbook_engine.clause_classifier import AMBIGUITY_THRESHOLD

    assert AMBIGUITY_THRESHOLD == 0.70, (
        "AMBIGUITY_THRESHOLD moved — update REFERENCE.md's classify rule to match "
        f"the new value ({AMBIGUITY_THRESHOLD}) before touching this test"
    )
    content = REFERENCE_MD.read_text(encoding="utf-8")
    assert "Confidence < 0.70: set `needs_review: true`" in content, (
        "REFERENCE.md's classify rule must cite AMBIGUITY_THRESHOLD (0.70), not an unrelated number"
    )
    assert "Confidence < 0.6:" not in content, (
        "REFERENCE.md still quotes the stale classify threshold (0.6) that matches "
        "no engine constant (clause_classifier.AMBIGUITY_THRESHOLD is 0.70)"
    )


def test_reference_md_has_no_deviation_judge_prompt() -> None:
    """There is no deviation judge (issue #239): REFERENCE.md carries no
    deviation prompt, verdict format, threshold or relocation-triage rule."""
    content = REFERENCE_MD.read_text(encoding="utf-8")
    assert "kind: deviation`)" not in content
    assert "Confidence < 0.65" not in content
    assert "Relocation triage" not in content
    assert "risk_delta" not in content
    assert "--with-deviation-judge" not in content
    assert "after-action" not in content.lower()


# ---------------------------------------------------------------------------
# README.md — Installation section documents both install paths (issue #149)
# ---------------------------------------------------------------------------

README = REPO_ROOT / "README.md"


def test_readme_has_installation_section() -> None:
    """README must have an Installation section (delivery vehicle, issue #149)."""
    content = README.read_text(encoding="utf-8")
    assert re.search(r"^##\s+Installation", content, re.MULTILINE), (
        "README.md must have a top-level '## Installation' section"
    )


def test_readme_documents_docker_install_path() -> None:
    """README Installation section must document the Docker path."""
    content = README.read_text(encoding="utf-8")
    assert "docker run" in content and "docker build" in content


def test_readme_documents_venv_install_path() -> None:
    """README Installation section must document the local venv path."""
    content = README.read_text(encoding="utf-8")
    assert ".venv" in content and "pip" in content.lower()


def test_readme_states_extraction_stack_per_path() -> None:
    """README must state which extraction stack each install path yields.

    Docker yields the full docling+OCR stack; the venv path falls back to
    legacy per-format adapters with no OCR. A legal-ops user choosing an
    install path needs to know this trade-off up front.
    """
    content = README.read_text(encoding="utf-8")
    assert "docling" in content
    assert "OCR" in content
    assert "legacy" in content.lower() or "no OCR" in content


def test_readme_references_playbook_from_corpus_skill() -> None:
    """README should point at the packaged skill as the orchestration layer."""
    content = README.read_text(encoding="utf-8")
    assert "playbook-from-corpus" in content
