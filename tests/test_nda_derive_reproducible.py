"""Reproducibility guard for the NDA worked example's derivation (issue #9).

`tests/test_nda_smoke.py` proves the deterministic no-LLM pipeline runs
end-to-end but deliberately stops at an evidence-only playbook (empty
`posture`/`floor` -- the stub judges never touch either). This test proves
the OTHER half of issue #9's reviewer gate: that the *genuinely judged*,
Posture-and-Floor-populated `examples/nda/playbook.opf.json` committed
alongside the corpus is itself reproducible from committed inputs, entirely
offline:

    lint-corpus -> judge-apply (canned verdicts) -> mine -> project
    -> posture interview (canned answers) -> floor sign -> validate

Every input this replays is committed under `examples/nda/`:
`corpus/`, `config.smoke.yaml`, `canned-verdicts.jsonl`,
`posture-answers.json`. The `floor sign` statement/id/rationale below are
copied verbatim from the same command used to author the committed
playbook (see `examples/nda/README.md`) -- keep the two in sync.

Hermetic, same guarantees as test_nda_smoke.py:
  - No network, no `ANTHROPIC_API_KEY` read (config.smoke.yaml omits
    `segmentation.agent`/`segmentation.llm`).
  - All output written under pytest's tmp_path.

This is deliberately NOT byte-identical reproduction: `compiler.generated_at`
and `posture.generation.generated_at` are wall-clock timestamps, and
`baseline.template_ref.source` is the deriving machine's own absolute
filesystem path, so `identity.content_hash` legitimately differs run to run
(same rule the `our-paper-baseline.v0.2.playbook.json` fixture documents for
its own frozen provenance strings). What must be reproducible is the
*semantic* content:
same clause/observation counts, same reversal, same Floor invariants, same
populated Posture -- asserted below by diffing the reproduced document
against the committed one with the volatile fields excluded.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from click.testing import CliRunner
from click.testing import Result as CliResult

from playbook_engine.cli import cli
from playbook_engine.validator import validate_document

_REPO_ROOT = Path(__file__).resolve().parent.parent
_NDA_DIR = _REPO_ROOT / "examples" / "nda"
_CORPUS_DIR = _NDA_DIR / "corpus"
_SMOKE_CONFIG = _NDA_DIR / "config.smoke.yaml"
_CANNED_VERDICTS = _NDA_DIR / "canned-verdicts.jsonl"
_POSTURE_ANSWERS = _NDA_DIR / "posture-answers.json"
_COMMITTED_PLAYBOOK = _NDA_DIR / "playbook.opf.json"

# Verbatim copy of the `playbook floor sign` invocation documented in
# examples/nda/README.md -- the one hand-authored Floor invariant that
# doesn't come from the Posture interview's Q4 auto-promotion.
_FLOOR_SIGN_ARGS = {
    "statement": (
        "Limitation of liability, if present, must not apply to a breach of the "
        "confidentiality obligations in this Agreement."
    ),
    "id": "limitation-of-liability-confidentiality-carveout",
    "rationale": (
        "A $50,000 liability cap appeared in 3 of 6 deals (epsilon-systems, "
        "theta-logistics, zeta-diagnostics) -- introduced by the counterparty in "
        "theta-logistics and zeta-diagnostics (both counterparty-paper) and carried "
        "into our own epsilon-systems draft; our taxonomy flags "
        "limitation-of-liability as normally absent from a mutual NDA because "
        "capping breach-of-confidence damages guts the agreement's only real "
        "remedy."
    ),
    "clause": "limitation_of_liability",
}


def _invoke(args: list[str]) -> CliResult:
    return CliRunner().invoke(cli, args)


def _strip_volatile(doc: dict[str, Any]) -> dict[str, Any]:
    """Drop wall-clock timestamps and the hashes computed over them, so two
    independently-derived runs of the same inputs compare equal."""
    doc = json.loads(json.dumps(doc))  # deep copy
    doc.get("compiler", {}).pop("generated_at", None)
    doc.get("posture", {}).get("generation", {}).pop("generated_at", None)
    doc.pop("identity", None)
    # baseline.template_ref.source is the derivation machine's own absolute
    # filesystem path to standard-form.rtf -- it differs by checkout root
    # (contributor clone, CI runner, Docker image) even when every other
    # field reproduces exactly. The committed artifact ships with this key
    # scrubbed entirely (issue #9 fix round 1 findings 1/2 -- same
    # treatment `publisher.py` applies before publication); a freshly
    # derived doc still carries the deriving machine's own path, so drop it
    # here too rather than asserting path equality.
    doc.get("baseline", {}).get("template_ref", {}).pop("source", None)
    return doc


def test_nda_derivation_reproducible_from_committed_inputs(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Replays the full judged derivation from examples/nda/'s committed
    corpus + config + canned verdicts + posture answers + floor-sign
    statement, and asserts the result matches the committed
    playbook.opf.json in every way that isn't a wall-clock timestamp."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True)

    lint_result = _invoke(["lint-corpus", str(_CORPUS_DIR), "--config", str(_SMOKE_CONFIG)])
    assert lint_result.exit_code == 0, f"lint-corpus failed:\n{lint_result.output}"

    judge_apply_result = _invoke(["judge-apply", str(out_dir), "--verdicts", str(_CANNED_VERDICTS)])
    assert judge_apply_result.exit_code == 0, f"judge-apply failed:\n{judge_apply_result.output}"

    mine_result = _invoke(
        ["mine", str(_CORPUS_DIR), "--config", str(_SMOKE_CONFIG), "--out", str(out_dir)]
    )
    assert mine_result.exit_code == 0, f"mine failed:\n{mine_result.output}"

    # judge (no --plan-only) confirms the canned verdicts drained the queue
    # completely -- a stale/incomplete canned-verdicts.jsonl would leave
    # needs_review items that fail validation below instead of here.
    judge_result = _invoke(
        ["judge", str(_CORPUS_DIR), "--config", str(_SMOKE_CONFIG), "--out", str(out_dir)]
    )
    assert judge_result.exit_code == 0, f"judge failed:\n{judge_result.output}"
    assert "(0 pending items)" in judge_result.output, (
        f"canned-verdicts.jsonl left pending judge items:\n{judge_result.output}"
    )

    project_result = _invoke(["project", str(out_dir), "--config", str(_SMOKE_CONFIG)])
    assert project_result.exit_code == 0, f"project failed:\n{project_result.output}"

    interview_result = _invoke(
        [
            "posture",
            "interview",
            str(out_dir),
            "--answers-file",
            str(_POSTURE_ANSWERS),
        ]
    )
    assert interview_result.exit_code == 0, f"posture interview failed:\n{interview_result.output}"

    floor_sign_args = [
        "floor",
        "sign",
        str(out_dir),
        "--statement",
        _FLOOR_SIGN_ARGS["statement"],
        "--id",
        _FLOOR_SIGN_ARGS["id"],
        "--rationale",
        _FLOOR_SIGN_ARGS["rationale"],
        "--clause",
        _FLOOR_SIGN_ARGS["clause"],
        "--config",
        str(_SMOKE_CONFIG),
    ]
    floor_sign_result = _invoke(floor_sign_args)
    assert floor_sign_result.exit_code == 0, f"floor sign failed:\n{floor_sign_result.output}"

    playbook_path = out_dir / "playbook.opf.json"
    assert playbook_path.exists(), "project did not write playbook.opf.json"

    validate_result = _invoke(["validate", str(playbook_path)])
    assert validate_result.exit_code == 0, f"validate failed:\n{validate_result.output}"

    reproduced = json.loads(playbook_path.read_text(encoding="utf-8"))
    result = validate_document(reproduced)
    assert result.ok, [str(e) for e in result.errors if e.blocking]

    committed = json.loads(_COMMITTED_PLAYBOOK.read_text(encoding="utf-8"))
    assert _strip_volatile(reproduced) == _strip_volatile(committed), (
        "Reproducing the derivation from examples/nda/'s committed corpus, config, "
        "canned-verdicts.jsonl, posture-answers.json, and the documented `floor sign` "
        "statement did not reproduce the committed playbook.opf.json -- one of those "
        "committed inputs is stale relative to the others. Regenerate "
        "examples/nda/playbook.opf.json from a fresh run of the same commands."
    )
