"""Intake of older versions of our own form (``baseline.prior_templates``, #246).

Covers the three places the setup flow touches: config parsing/validation,
``playbook stage`` (older forms copied beside the current template, in every
staging mode: plain, ``--from-plan``, and refused by ``--plan-only``), and
``lint-corpus`` (validation errors, plus a warning -- never an error -- when
none are supplied for a template-mode run over a real corpus). Also pins that
the setup skill asks for them straight after the current standard form.

SECURITY NOTE: all fixtures are synthetic stubs or the repo's synthetic
judge-fixture corpus; no real agreement content.
"""

from __future__ import annotations

import shutil
from datetime import date
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from playbook_engine.cli import cli
from playbook_engine.config import (
    ConfigError,
    PriorTemplate,
    load_config,
    normalize_in_use_date,
    period_end,
    period_start,
)
from playbook_engine.corpus_linter import PRIOR_TEMPLATES_WARN_MIN_DEALS, lint_corpus
from playbook_engine.intake_plan import build_staging_plan
from playbook_engine.staging import scaffold_config

REPO = Path(__file__).parent.parent
SKILL = REPO / ".claude" / "skills" / "playbook-from-corpus" / "SKILL.md"


def _write_rtf(path: Path, text: str = "stub") -> None:
    path.write_text(f"{{\\rtf1 {text}}}\n", encoding="utf-8")


def _config(tmp_path: Path, baseline: str, *, files: tuple[str, ...] = ()) -> Path:
    for name in files:
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        _write_rtf(tmp_path / name)
    cfg = tmp_path / "playbook.config.yaml"
    cfg.write_text(
        "agreement_type:\n  id: fictional-agreement\n  name: Fictional Agreement\n"
        f"baseline:\n{baseline}\n"
        "taxonomy: builtin:cuad-base.yaml\n"
        "provenance:\n  our_party_aliases: [FixtureCorp]\n",
        encoding="utf-8",
    )
    return cfg


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------


class TestConfigParsing:
    def test_missing_list_means_none_supplied(self, tmp_path: Path) -> None:
        cfg = load_config(_config(tmp_path, "  template: null"))
        assert cfg.baseline.prior_templates == []

    def test_empty_and_null_list_are_valid(self, tmp_path: Path) -> None:
        assert (
            load_config(_config(tmp_path, "  prior_templates: []")).baseline.prior_templates == []
        )
        assert load_config(_config(tmp_path, "  prior_templates:")).baseline.prior_templates == []

    def test_round_trip_resolves_paths_and_normalizes_dates(self, tmp_path: Path) -> None:
        path = _config(
            tmp_path,
            "  template: current.rtf\n"
            "  prior_templates:\n"
            "    - {path: old/a.rtf, label: 2019 form, in_use_from: 2019, in_use_until: 2021-06}\n"
            "    - {path: old/b.rtf, label: Earlier form, in_use_until: 2019-02-14}\n"
            "    - {path: old/c.rtf, label: Undated form}\n"
            '    - {path: old/d.rtf, label: Quoted dates, in_use_from: "2015", in_use_until: "2016-03"}',
            files=("current.rtf", "old/a.rtf", "old/b.rtf", "old/c.rtf", "old/d.rtf"),
        )
        priors = load_config(path).baseline.prior_templates
        assert priors == [
            PriorTemplate((tmp_path / "old/a.rtf").resolve(), "2019 form", "2019", "2021-06"),
            PriorTemplate((tmp_path / "old/b.rtf").resolve(), "Earlier form", None, "2019-02-14"),
            PriorTemplate((tmp_path / "old/c.rtf").resolve(), "Undated form", None, None),
            PriorTemplate((tmp_path / "old/d.rtf").resolve(), "Quoted dates", "2015", "2016-03"),
        ]

    def test_prior_forms_without_a_current_template_are_accepted(self, tmp_path: Path) -> None:
        path = _config(
            tmp_path,
            "  template: null\n  prior_templates:\n    - {path: a.rtf, label: Only}",
            files=("a.rtf",),
        )
        cfg = load_config(path)
        assert not cfg.baseline.has_canonical_template
        assert [p.label for p in cfg.baseline.prior_templates] == ["Only"]

    @pytest.mark.parametrize(
        ("baseline", "needle"),
        [
            ("  prior_templates: nope", "must be a list"),
            ("  prior_templates:\n    - just-a-string", "must be a mapping"),
            ("  prior_templates:\n    - {label: X}", "'path' is missing"),
            ("  prior_templates:\n    - {path: a.rtf}", "'label' is missing"),
            ("  prior_templates:\n    - {path: a.rtf, label: '  '}", "label must be a non-empty"),
            ("  prior_templates:\n    - {path: gone.rtf, label: X}", "path not found"),
            (
                "  prior_templates:\n    - {path: a.rtf, label: X, colour: red}",
                "colour",
            ),
            (
                "  prior_templates:\n    - {path: a.rtf, label: X}\n    - {path: b.rtf, label: x}",
                "labels must be unique",
            ),
            (
                "  prior_templates:\n    - {path: a.rtf, label: X}\n    - {path: ./a.rtf, label: Y}",
                "same file",
            ),
            (
                "  template: a.rtf\n  prior_templates:\n    - {path: a.rtf, label: X}",
                "current baseline.template",
            ),
            (
                "  prior_templates:\n    - {path: a.rtf, label: X, in_use_from: 2021, in_use_until: 2019}",
                "runs backwards",
            ),
            (
                "  prior_templates:\n    - {path: a.rtf, label: X, in_use_from: 2021-07, in_use_until: 2021-06}",
                "runs backwards",
            ),
            (
                "  prior_templates:\n    - {path: a.rtf, label: X, in_use_from: 2021-13}",
                "in_use_from",
            ),
            (
                "  prior_templates:\n    - {path: a.rtf, label: X, in_use_until: soon}",
                "in_use_until",
            ),
            (
                "  prior_templates:\n    - {path: a.rtf, label: X, in_use_from: 1066}",
                "year must be between",
            ),
        ],
    )
    def test_invalid_entries_give_clear_errors(
        self, tmp_path: Path, baseline: str, needle: str
    ) -> None:
        path = _config(tmp_path, baseline, files=("a.rtf", "b.rtf"))
        with pytest.raises(ConfigError, match=needle):
            load_config(path)

    def test_error_names_the_offending_entry(self, tmp_path: Path) -> None:
        path = _config(
            tmp_path,
            "  prior_templates:\n    - {path: a.rtf, label: Fine}\n    - {path: nope.rtf, label: Bad}",
            files=("a.rtf",),
        )
        with pytest.raises(ConfigError, match=r"prior_templates\[1\]"):
            load_config(path)

    def test_same_period_range_is_sane(self, tmp_path: Path) -> None:
        path = _config(
            tmp_path,
            "  prior_templates:\n    - {path: a.rtf, label: X, in_use_from: 2020-03, in_use_until: 2020}",
            files=("a.rtf",),
        )
        assert load_config(path).baseline.prior_templates[0].in_use_until == "2020"


class TestDateHelpers:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (2019, "2019"),
            ("2019", "2019"),
            ("2019-03", "2019-03"),
            ("2019-03-15", "2019-03-15"),
            (date(2019, 3, 15), "2019-03-15"),
            (None, None),
        ],
    )
    def test_normalize(self, value: object, expected: str | None) -> None:
        assert normalize_in_use_date(value, "x") == expected

    @pytest.mark.parametrize("value", [True, 1.5, "2019-02-30", "March 2019", ["2019"]])
    def test_normalize_rejects(self, value: object) -> None:
        with pytest.raises(ConfigError):
            normalize_in_use_date(value, "x")

    def test_period_bounds(self) -> None:
        assert period_start("2020") == date(2020, 1, 1)
        assert period_end("2020") == date(2020, 12, 31)
        assert period_end("2020-02") == date(2020, 2, 29)
        assert period_start("2020-02-10") == period_end("2020-02-10") == date(2020, 2, 10)


# ---------------------------------------------------------------------------
# Staging
# ---------------------------------------------------------------------------


def _src_corpus(tmp_path: Path, *, with_priors: bool = True) -> Path:
    src = tmp_path / "raw"
    (src / "deal-a").mkdir(parents=True)
    _write_rtf(src / "deal-a" / "v1.rtf", "one")
    _write_rtf(src / "deal-a" / "v2.rtf", "two")
    _write_rtf(src / "our-template.rtf", "current")
    if with_priors:
        _write_rtf(src / "prior-template-2019 form.rtf", "old1")
        _write_rtf(src / "Prior_Template_2017.rtf", "old2")
    return src


def _staged_config_as_loadable(out: Path) -> Path:
    """The scaffolded config with its taxonomy placeholder filled in, as a user would."""
    text = (out / "playbook.config.yaml").read_text(encoding="utf-8")
    ready = out / "ready.config.yaml"
    ready.write_text(
        text.replace("FILL_IN_TAXONOMY_PATH", "builtin:cuad-base.yaml"), encoding="utf-8"
    )
    return ready


class TestStaging:
    def test_root_convention_files_are_staged_beside_the_current_template(
        self, tmp_path: Path
    ) -> None:
        src, out = _src_corpus(tmp_path), tmp_path / "staged"
        skeleton = scaffold_config(src, out)
        assert skeleton["baseline"]["template"] == "our-template.rtf"
        assert skeleton["baseline"]["prior_templates"] == [
            {"path": "prior-template--2017.rtf", "label": "2017"},
            {"path": "prior-template--2019-form.rtf", "label": "2019 form"},
        ]
        for name in (
            "our-template.rtf",
            "prior-template--2017.rtf",
            "prior-template--2019-form.rtf",
        ):
            assert (out / name).is_file(), name
        # Plain root files only: a sub-folder would be mined as a deal.
        assert not [d for d in out.iterdir() if d.is_dir()]

    def test_scaffolded_config_loads_and_round_trips(self, tmp_path: Path) -> None:
        src, out = _src_corpus(tmp_path), tmp_path / "staged"
        scaffold_config(src, out)
        cfg = load_config(_staged_config_as_loadable(out))
        assert cfg.baseline.template_path == (out / "our-template.rtf").resolve()
        assert [p.label for p in cfg.baseline.prior_templates] == ["2017", "2019 form"]
        assert all(p.path.parent == out.resolve() for p in cfg.baseline.prior_templates)

    def test_prior_forms_never_hijack_the_current_template_detection(self, tmp_path: Path) -> None:
        src = tmp_path / "raw"
        src.mkdir()
        _write_rtf(src / "prior-template-2015.rtf")  # sorts before, contains "template"
        _write_rtf(src / "zz-our-template.rtf")
        skeleton = scaffold_config(src, tmp_path / "staged")
        assert skeleton["baseline"]["template"] == "zz-our-template.rtf"

    def test_explicit_paths_deduplicate_against_the_convention(self, tmp_path: Path) -> None:
        src, out = _src_corpus(tmp_path), tmp_path / "staged"
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        _write_rtf(elsewhere / "Form 2012.rtf")
        skeleton = scaffold_config(
            src, out, [elsewhere / "Form 2012.rtf", src / "Prior_Template_2017.rtf"]
        )
        labels = [e["label"] for e in skeleton["baseline"]["prior_templates"]]
        assert sorted(labels) == ["2017", "2019 form", "Form 2012"]

    def test_duplicate_labels_are_made_unique(self, tmp_path: Path) -> None:
        src = _src_corpus(tmp_path, with_priors=False)
        a, b = tmp_path / "x" / "Form.rtf", tmp_path / "y" / "Form.rtf"
        for p in (a, b):
            p.parent.mkdir()
            _write_rtf(p)
        skeleton = scaffold_config(src, tmp_path / "staged", [a, b])
        entries = skeleton["baseline"]["prior_templates"]
        assert [e["label"] for e in entries] == ["Form", "Form (2)"]
        assert len({e["path"] for e in entries}) == 2
        load_config(_staged_config_as_loadable(tmp_path / "staged"))  # unique, loads

    def test_none_supplied_leaves_no_list_and_a_reminder_comment(self, tmp_path: Path) -> None:
        src, out = _src_corpus(tmp_path, with_priors=False), tmp_path / "staged"
        skeleton = scaffold_config(src, out)
        assert "prior_templates" not in skeleton["baseline"]
        text = (out / "playbook.config.yaml").read_text(encoding="utf-8")
        assert "# Optional: baseline.prior_templates" in text
        assert yaml.safe_load(text)["baseline"] == {"template": "our-template.rtf"}
        assert load_config(_staged_config_as_loadable(out)).baseline.prior_templates == []

    def test_unsupported_or_missing_explicit_path_is_a_value_error(self, tmp_path: Path) -> None:
        src = _src_corpus(tmp_path, with_priors=False)
        bad = tmp_path / "notes.txt"
        bad.write_text("x")
        with pytest.raises(ValueError, match="unsupported format"):
            scaffold_config(src, tmp_path / "o1", [bad])
        with pytest.raises(ValueError, match="not found"):
            scaffold_config(src, tmp_path / "o2", [tmp_path / "missing.rtf"])

    def test_same_slug_different_label_sources_both_survive(self, tmp_path: Path) -> None:
        src, out = _src_corpus(tmp_path, with_priors=False), tmp_path / "staged"
        a, b = tmp_path / "x" / "2019 form.rtf", tmp_path / "y" / "2019_form.rtf"
        a.parent.mkdir()
        b.parent.mkdir()
        _write_rtf(a, "first-content")
        _write_rtf(b, "second-content")
        skeleton = scaffold_config(src, out, [a, b])
        entries = skeleton["baseline"]["prior_templates"]
        assert len({e["path"] for e in entries}) == 2
        assert len({e["label"].casefold() for e in entries}) == 2
        contents = sorted((out / e["path"]).read_text(encoding="utf-8") for e in entries)
        assert contents == sorted([a.read_text(encoding="utf-8"), b.read_text(encoding="utf-8")])
        cfg = load_config(_staged_config_as_loadable(out))
        assert len(cfg.baseline.prior_templates) == 2

    def test_current_template_as_prior_template_is_rejected(self, tmp_path: Path) -> None:
        src = _src_corpus(tmp_path, with_priors=False)
        with pytest.raises(ValueError, match="current standard form"):
            scaffold_config(src, tmp_path / "o1", [src / "our-template.rtf"])
        result = CliRunner().invoke(
            cli,
            [
                "stage",
                str(src),
                "--out",
                str(tmp_path / "o2"),
                "--prior-template",
                str(src / "our-template.rtf"),
            ],
        )
        assert result.exit_code == 1
        assert "current standard form" in result.output

    def test_cli_stage_flag_and_summary_line(self, tmp_path: Path) -> None:
        src, out = _src_corpus(tmp_path, with_priors=False), tmp_path / "staged"
        old = tmp_path / "Old Form.rtf"
        _write_rtf(old)
        result = CliRunner().invoke(
            cli, ["stage", str(src), "--out", str(out), "--prior-template", str(old)]
        )
        assert result.exit_code == 0, result.output
        assert "1 older version(s) of our form staged" in result.output
        assert (out / "prior-template--old-form.rtf").is_file()
        assert (out / "deal-a").is_dir()

    def test_cli_stage_without_flag_says_none_supplied(self, tmp_path: Path) -> None:
        src, out = _src_corpus(tmp_path, with_priors=False), tmp_path / "staged"
        result = CliRunner().invoke(cli, ["stage", str(src), "--out", str(out)])
        assert result.exit_code == 0, result.output
        assert "no older versions of our form supplied" in result.output

    def test_cli_stage_bad_prior_template_exits_one(self, tmp_path: Path) -> None:
        src, out = _src_corpus(tmp_path, with_priors=False), tmp_path / "staged"
        bad = tmp_path / "notes.txt"
        bad.write_text("x")
        result = CliRunner().invoke(
            cli, ["stage", str(src), "--out", str(out), "--prior-template", str(bad)]
        )
        assert result.exit_code == 1
        assert "unsupported format" in result.output


# A loose corpus (layout `unknown`): real fixture agreements as flat files, plus
# an older version of our form named by the root convention. Its text is a copy
# of a deal document, so without the exclusion it would cluster into a deal.
FIXTURE_CORPUS = REPO / "examples" / "judge-fixture" / "corpus"


def _loose_corpus(tmp_path: Path) -> Path:
    src = tmp_path / "loose"
    src.mkdir()
    for original, name in (
        (FIXTURE_CORPUS / "deal-alpha" / "v1.rtf", "doc_001.rtf"),
        (FIXTURE_CORPUS / "deal-alpha" / "v2.rtf", "doc_002.rtf"),
        (FIXTURE_CORPUS / "deal-beta" / "v1.rtf", "doc_003.rtf"),
        (FIXTURE_CORPUS / "deal-alpha" / "v1.rtf", "prior-template-2019.rtf"),
    ):
        shutil.copy2(original, src / name)
    return src


def _plan_paths(plan: dict) -> set[str]:
    in_deals = {f["path"] for d in plan["deals"] for f in d["files"]}
    return in_deals | {u["path"] for u in plan["unassigned"]}


class TestStagingThroughAPlan:
    def test_plan_only_refuses_prior_template_and_writes_nothing(self, tmp_path: Path) -> None:
        src, out = _loose_corpus(tmp_path), tmp_path / "out"
        old = tmp_path / "Old Form.rtf"
        _write_rtf(old)
        result = CliRunner().invoke(
            cli,
            ["stage", str(src), "--out", str(out), "--plan-only", "--prior-template", str(old)],
        )
        assert result.exit_code == 2, result.output
        assert "--prior-template is not used by --plan-only" in result.output
        assert "--from-plan" in result.output
        assert not (out / "staging_plan.json").exists()

    def test_plan_leaves_root_prior_forms_out_of_deal_assembly(self, tmp_path: Path) -> None:
        plan = build_staging_plan(_loose_corpus(tmp_path))
        paths = _plan_paths(plan)
        assert "prior-template-2019.rtf" not in paths
        assert {"doc_001.rtf", "doc_002.rtf", "doc_003.rtf"} <= paths
        assert any(w.startswith("prior-template-2019.rtf: left out") for w in plan["warnings"])

    def test_next_step_hints_carry_prior_template_to_from_plan(self, tmp_path: Path) -> None:
        src = _loose_corpus(tmp_path)
        unknown = CliRunner().invoke(cli, ["stage", str(src), "--out", str(tmp_path / "o1")])
        assert unknown.exit_code == 1
        assert "could not determine a known layout" in unknown.output
        assert "--prior-template" in unknown.output
        planned = CliRunner().invoke(
            cli, ["stage", str(src), "--out", str(tmp_path / "o2"), "--plan-only"]
        )
        assert planned.exit_code == 0, planned.output
        assert "--from-plan" in planned.output
        assert "--prior-template" in planned.output

    def test_from_plan_stages_every_older_form(self, tmp_path: Path) -> None:
        src, out = _loose_corpus(tmp_path), tmp_path / "out"
        planned = CliRunner().invoke(cli, ["stage", str(src), "--out", str(out), "--plan-only"])
        assert planned.exit_code == 0, planned.output
        plan_copy = tmp_path / "staging_plan.json"
        shutil.copy2(out / "staging_plan.json", plan_copy)
        old = tmp_path / "Old Form.rtf"
        _write_rtf(old, "older")

        result = CliRunner().invoke(
            cli,
            [
                "stage",
                str(src),
                "--out",
                str(out),
                "--from-plan",
                str(plan_copy),
                "--prior-template",
                str(old),
            ],
        )
        assert result.exit_code == 0, result.output
        assert "2 older version(s) of our form staged" in result.output
        skeleton = yaml.safe_load((out / "playbook.config.yaml").read_text(encoding="utf-8"))
        assert skeleton["baseline"]["prior_templates"] == [
            {"path": "prior-template--old-form.rtf", "label": "Old Form"},
            {"path": "prior-template--2019.rtf", "label": "2019"},
        ]
        for entry in skeleton["baseline"]["prior_templates"]:
            assert (out / entry["path"]).is_file()
        # The older form is a root file beside the template, never a deal version.
        staged_versions = [p.name for d in out.iterdir() if d.is_dir() for p in d.iterdir()]
        assert not [n for n in staged_versions if "prior-template" in n]
        cfg = load_config(_staged_config_as_loadable(out))
        assert [p.label for p in cfg.baseline.prior_templates] == ["Old Form", "2019"]


# ---------------------------------------------------------------------------
# lint-corpus
# ---------------------------------------------------------------------------


def _lintable(tmp_path: Path, n_deals: int, baseline: str) -> tuple[Path, Path]:
    corpus = tmp_path / "corpus"
    for i in range(n_deals):
        d = corpus / f"deal-{i:02d}"
        d.mkdir(parents=True)
        _write_rtf(d / "v1.rtf")
    cfg = _config(corpus, baseline, files=("current.rtf", "old-a.rtf"))
    return corpus, cfg


def _codes(report, level: str) -> list[str]:
    return [i.code for i in report.items if i.level == level]


class TestLint:
    def test_warns_when_none_supplied_for_template_run_over_enough_deals(
        self, tmp_path: Path
    ) -> None:
        corpus, cfg = _lintable(tmp_path, PRIOR_TEMPLATES_WARN_MIN_DEALS, "  template: current.rtf")
        report = lint_corpus(corpus, cfg)
        assert "CONFIG_NO_PRIOR_TEMPLATES" in _codes(report, "warning")
        assert not report.has_errors

    def test_empty_list_warns_the_same(self, tmp_path: Path) -> None:
        corpus, cfg = _lintable(
            tmp_path,
            PRIOR_TEMPLATES_WARN_MIN_DEALS,
            "  template: current.rtf\n  prior_templates: []",
        )
        assert "CONFIG_NO_PRIOR_TEMPLATES" in _codes(lint_corpus(corpus, cfg), "warning")

    def test_no_warning_below_the_deal_threshold(self, tmp_path: Path) -> None:
        corpus, cfg = _lintable(
            tmp_path, PRIOR_TEMPLATES_WARN_MIN_DEALS - 1, "  template: current.rtf"
        )
        assert "CONFIG_NO_PRIOR_TEMPLATES" not in _codes(lint_corpus(corpus, cfg), "warning")

    def test_no_warning_in_emergent_mode(self, tmp_path: Path) -> None:
        corpus, cfg = _lintable(tmp_path, PRIOR_TEMPLATES_WARN_MIN_DEALS, "  template: null")
        assert "CONFIG_NO_PRIOR_TEMPLATES" not in _codes(lint_corpus(corpus, cfg), "warning")

    def test_no_warning_when_priors_supplied(self, tmp_path: Path) -> None:
        corpus, cfg = _lintable(
            tmp_path,
            PRIOR_TEMPLATES_WARN_MIN_DEALS,
            "  template: current.rtf\n  prior_templates:\n    - {path: old-a.rtf, label: Old A}",
        )
        report = lint_corpus(corpus, cfg)
        assert "CONFIG_NO_PRIOR_TEMPLATES" not in _codes(report, "warning")
        assert "CONFIG_PRIOR_TEMPLATES" in _codes(report, "ok")
        assert not report.has_errors

    def test_invalid_entry_is_an_error_with_the_loader_message(self, tmp_path: Path) -> None:
        corpus, cfg = _lintable(
            tmp_path,
            2,
            "  template: current.rtf\n  prior_templates:\n    - {path: nope.rtf, label: Old A}",
        )
        report = lint_corpus(corpus, cfg)
        assert "CONFIG_PRIOR_TEMPLATES_INVALID" in _codes(report, "error")
        with pytest.raises(ConfigError, match="path not found"):
            load_config(cfg)  # the loader refuses the very same config

    def test_unknown_entry_key_and_backwards_dates_are_errors(self, tmp_path: Path) -> None:
        corpus, cfg = _lintable(
            tmp_path,
            2,
            "  prior_templates:\n    - {path: old-a.rtf, label: A, in_use_from: 2022, in_use_until: 2020}",
        )
        assert "CONFIG_PRIOR_TEMPLATES_INVALID" in _codes(lint_corpus(corpus, cfg), "error")

    def test_unquoted_impossible_date_is_a_clear_error_not_a_crash(self, tmp_path: Path) -> None:
        corpus, cfg = _lintable(
            tmp_path,
            2,
            "  template: current.rtf\n  prior_templates:\n"
            "    - path: old-a.rtf\n      label: Old A\n      in_use_until: 2021-06-31",
        )
        with pytest.raises(ConfigError, match=r"prior_templates\[0\]"):
            load_config(cfg)
        report = lint_corpus(corpus, cfg)  # must report, not raise
        assert "CONFIG_PRIOR_TEMPLATES_INVALID" in _codes(report, "error")

    def test_unquoted_valid_date_still_loads(self, tmp_path: Path) -> None:
        _, cfg = _lintable(
            tmp_path,
            2,
            "  template: current.rtf\n  prior_templates:\n"
            "    - path: old-a.rtf\n      label: Old A\n      in_use_until: 2021-06-15",
        )
        assert load_config(cfg).baseline.prior_templates[0].in_use_until == "2021-06-15"

    def test_lint_survives_a_missing_taxonomy_value(self, tmp_path: Path) -> None:
        """A config whose taxonomy is empty must still reach the prior-template check."""
        corpus = tmp_path / "corpus"
        (corpus / "deal-a").mkdir(parents=True)
        _write_rtf(corpus / "deal-a" / "v1.rtf")
        cfg = corpus / "playbook.config.yaml"
        cfg.write_text(
            "agreement_type: {id: x, name: X}\nbaseline:\n  prior_templates: [oops]\n",
            encoding="utf-8",
        )
        assert "CONFIG_PRIOR_TEMPLATES_INVALID" in _codes(lint_corpus(corpus, cfg), "error")


# ---------------------------------------------------------------------------
# The setup skill asks for them, right after the current form
# ---------------------------------------------------------------------------


class TestSkillSetup:
    @pytest.fixture
    def skill(self) -> str:
        return SKILL.read_text(encoding="utf-8")

    def test_prior_versions_question_follows_the_standard_form_question(self, skill: str) -> None:
        setup = skill.index("## Interactive setup")
        form_q = skill.index("Do you have your own standard form", setup)
        prior_q = skill.index("ask for the older versions of our form", form_q)
        record = skill.index("Record the answers", form_q)
        party = skill.index("## Derive party names automatically", setup)
        assert form_q < prior_q < record < party

    def test_opening_flow_lists_it_straight_after_the_standard_form_call(self, skill: str) -> None:
        flow = skill.index("## The standard flow")
        second_call = skill.index("- second call (routes A and C)", flow)
        prior = skill.index("older versions of our\n  form", second_call)
        third_call = skill.index("- third call", second_call)
        assert second_call < prior < third_call

    def test_offers_the_three_multiple_choice_answers(self, skill: str) -> None:
        for option in ("I have older versions", "I don't have any", "Not sure"):
            assert f'"{option}"' in skill

    def test_explains_why_and_that_none_is_fine(self, skill: str) -> None:
        assert "recognise our own former wording" in skill
        assert "reconstruct the older forms from the corpus" in skill
        assert "perfectly normal answer" in skill

    def test_wires_the_answer_to_stage_and_config(self, skill: str) -> None:
        assert "--prior-template" in skill
        assert "baseline.prior_templates" in skill

    def test_plan_route_carries_the_older_forms_to_from_plan(self, skill: str) -> None:
        step = skill.index("### Step 1a")
        block = skill[step : skill.index("Hard rules for this stage", step)]
        assert "`--plan-only` refuses it" in block
        assert "Keep them out of deal assembly" in block
        from_plan = block.index("--from-plan ~/.cache")
        assert "--prior-template" in block[from_plan : block.index("```", from_plan)]
