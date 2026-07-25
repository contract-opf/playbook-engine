"""Tests for canonical schema $id URLs and Pages publication (issue #160).

Issue #148 made schema `$id`s repo-relative as an interim measure "until
publication." Issue #126 decided (2026-07-10) to publish now; this issue
(#160) restores the canonical https:// `$id`s and adds the GitHub Pages
workflow that serves them at those URLs.

Verifies:
  - Every spec/*.schema*.json `$id` is the canonical
    https://contract-opf.github.io/playbook-engine/<repo-relative-path> URL
    matching its own repo path.
  - The Pages-publish workflow file exists.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent

CANONICAL_BASE = "https://contract-opf.github.io/playbook-engine"

SCHEMA_FILES = [
    ROOT / "spec" / "playbook.schema.json",
    ROOT / "spec" / "playbook.schema-0.2.json",
    ROOT / "spec" / "clause-tree.schema.json",
]

PAGES_WORKFLOW_PATH = ROOT / ".github" / "workflows" / "pages.yml"


def test_schema_ids_are_canonical_urls() -> None:
    for schema_path in SCHEMA_FILES:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        schema_id = schema.get("$id")
        assert schema_id, f"{schema_path.name} is missing $id"

        repo_relative = schema_path.relative_to(ROOT).as_posix()
        expected = f"{CANONICAL_BASE}/{repo_relative}"
        assert schema_id == expected, (
            f"{schema_path.name} $id is not its canonical URL: "
            f"got {schema_id!r}, expected {expected!r}"
        )


def test_pages_workflow_exists() -> None:
    assert PAGES_WORKFLOW_PATH.is_file(), f"Pages-publish workflow missing: {PAGES_WORKFLOW_PATH}"


def test_pages_workflow_serves_spec_schemas() -> None:
    workflow_text = PAGES_WORKFLOW_PATH.read_text(encoding="utf-8")
    workflow = yaml.safe_load(workflow_text)

    assert "deploy-pages" in workflow_text, (
        "pages.yml does not use actions/deploy-pages to publish to GitHub Pages"
    )
    assert "spec" in workflow_text, "pages.yml does not reference the spec/ directory"

    jobs = workflow.get("jobs", {})
    assert jobs, "pages.yml defines no jobs"
    permissions = workflow.get("permissions", {})
    assert permissions.get("pages") == "write", (
        "pages.yml must grant 'pages: write' to deploy to GitHub Pages"
    )
    assert permissions.get("id-token") == "write", (
        "pages.yml must grant 'id-token: write' for GitHub Pages OIDC deployment"
    )
