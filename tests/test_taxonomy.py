"""Tests for the taxonomy loader and CUAD-merge utility."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from playbook_engine.taxonomy import (
    Taxonomy,
    TaxonomyError,
    load_taxonomy,
    merge_taxonomy,
)

AFFILIATION_TAXONOMY = (
    Path(__file__).parent.parent / "spec" / "taxonomy" / "affiliation-agreement.yaml"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_taxonomy(tmp_path: Path, data: dict) -> Path:  # type: ignore[type-arg]
    path = tmp_path / "taxonomy.yaml"
    path.write_text(yaml.dump(data, allow_unicode=True), encoding="utf-8")
    return path


def _minimal_taxonomy(tmp_path: Path) -> Path:
    return _write_taxonomy(
        tmp_path,
        {
            "source": "test-v1",
            "entries": [
                {
                    "id": "indemnification",
                    "label": "Indemnification",
                    "status": "active",
                    "cuad_origin": "Indemnification",
                },
                {
                    "id": "governing_law",
                    "label": "Governing Law",
                    "status": "active",
                    "cuad_origin": "Governing Law",
                },
                {
                    "id": "exclusivity",
                    "label": "Exclusivity",
                    "status": "inactive",
                    "cuad_origin": "Exclusivity",
                },
                {
                    "id": "internal_notes",
                    "label": "Internal Notes",
                    "status": "custom",
                    "cuad_origin": None,
                },
            ],
        },
    )


# ---------------------------------------------------------------------------
# Happy-path: bundled affiliation taxonomy
# ---------------------------------------------------------------------------


def test_load_affiliation_taxonomy() -> None:
    tax = load_taxonomy(AFFILIATION_TAXONOMY)
    assert isinstance(tax, Taxonomy)
    assert tax.source == "CUAD-v1"
    assert len(tax.entries) > 10


def test_affiliation_has_active_entries() -> None:
    tax = load_taxonomy(AFFILIATION_TAXONOMY)
    active = [e for e in tax.entries if e.status == "active"]
    assert len(active) >= 10, "Affiliation taxonomy should have many active entries"


def test_affiliation_has_inactive_entries() -> None:
    tax = load_taxonomy(AFFILIATION_TAXONOMY)
    inactive = [e for e in tax.entries if e.status == "inactive"]
    assert len(inactive) >= 5, "Affiliation taxonomy should have several inactive entries"


def test_classifier_entries_excludes_inactive() -> None:
    tax = load_taxonomy(AFFILIATION_TAXONOMY)
    classifier = tax.classifier_entries()
    assert all(e.status in ("active", "custom") for e in classifier)
    assert not any(e.status == "inactive" for e in classifier)


def test_no_duplicate_ids() -> None:
    tax = load_taxonomy(AFFILIATION_TAXONOMY)
    ids = [e.id for e in tax.entries]
    assert len(ids) == len(set(ids)), "All taxonomy entry ids must be unique"


def test_get_entry_by_id() -> None:
    tax = load_taxonomy(AFFILIATION_TAXONOMY)
    entry = tax.get("indemnification")
    assert entry is not None
    assert entry.id == "indemnification"
    assert entry.status == "active"


def test_get_unknown_id_returns_none() -> None:
    tax = load_taxonomy(AFFILIATION_TAXONOMY)
    assert tax.get("nonexistent_clause_xyz") is None


# ---------------------------------------------------------------------------
# Minimal taxonomy
# ---------------------------------------------------------------------------


def test_minimal_taxonomy_loads(tmp_path: Path) -> None:
    path = _minimal_taxonomy(tmp_path)
    tax = load_taxonomy(path)
    assert tax.source == "test-v1"
    assert len(tax.entries) == 4


def test_custom_entry_cuad_origin_null(tmp_path: Path) -> None:
    path = _minimal_taxonomy(tmp_path)
    tax = load_taxonomy(path)
    custom = tax.get("internal_notes")
    assert custom is not None
    assert custom.status == "custom"
    assert custom.cuad_origin is None


def test_classifier_eligible_property(tmp_path: Path) -> None:
    path = _minimal_taxonomy(tmp_path)
    tax = load_taxonomy(path)
    assert tax.get("indemnification").is_classifier_eligible  # type: ignore[union-attr]
    assert tax.get("internal_notes").is_classifier_eligible  # type: ignore[union-attr]
    assert not tax.get("exclusivity").is_classifier_eligible  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# CUAD-merge utility (OPF §5)
# ---------------------------------------------------------------------------


def test_merge_adds_new_entries_as_inactive(tmp_path: Path) -> None:
    path = _minimal_taxonomy(tmp_path)
    existing = load_taxonomy(path)

    upstream = [
        {"id": "brand_new_clause", "label": "Brand New Clause", "cuad_origin": "Brand New Clause"},
        {"id": "another_new", "label": "Another New", "cuad_origin": "Another New"},
    ]
    merged = merge_taxonomy(existing, upstream)

    assert len(merged.entries) == len(existing.entries) + 2

    brand_new = merged.get("brand_new_clause")
    assert brand_new is not None
    assert brand_new.status == "inactive", "New entries must enter as inactive (OPF §5)"
    assert brand_new.cuad_origin == "Brand New Clause"


def test_merge_preserves_existing_curation(tmp_path: Path) -> None:
    """Known ids must keep their curated status — not reset to inactive (OPF §5)."""
    path = _minimal_taxonomy(tmp_path)
    existing = load_taxonomy(path)

    # upstream has the same ids but with different status values
    upstream = [
        {
            "id": "indemnification",
            "label": "Indemnification",
            "status": "inactive",
            "cuad_origin": "Indemnification",
        },
        {
            "id": "exclusivity",
            "label": "Exclusivity",
            "status": "active",
            "cuad_origin": "Exclusivity",
        },
        {"id": "brand_new", "label": "Brand New", "cuad_origin": "Brand New"},
    ]
    merged = merge_taxonomy(existing, upstream)

    # existing entries must be unchanged
    assert merged.get("indemnification").status == "active"  # type: ignore[union-attr]
    assert merged.get("exclusivity").status == "inactive"  # type: ignore[union-attr]
    # only the new one is added
    assert len(merged.entries) == len(existing.entries) + 1


def test_merge_preserves_custom_entries(tmp_path: Path) -> None:
    """Custom entries (status: custom) must survive a merge unchanged."""
    path = _minimal_taxonomy(tmp_path)
    existing = load_taxonomy(path)
    merged = merge_taxonomy(existing, [])
    custom = merged.get("internal_notes")
    assert custom is not None
    assert custom.status == "custom"
    assert custom.cuad_origin is None


def test_merge_with_empty_upstream_is_identity(tmp_path: Path) -> None:
    path = _minimal_taxonomy(tmp_path)
    existing = load_taxonomy(path)
    merged = merge_taxonomy(existing, [])
    assert len(merged.entries) == len(existing.entries)
    assert merged.source == existing.source


def test_entry_is_immutable(tmp_path: Path) -> None:
    """TaxonomyEntry must be frozen — mutation raises FrozenInstanceError (NB-A fix)."""
    import dataclasses

    path = _minimal_taxonomy(tmp_path)
    tax = load_taxonomy(path)
    entry = tax.entries[0]
    with pytest.raises(dataclasses.FrozenInstanceError):
        entry.status = "inactive"  # type: ignore[misc]


def test_merge_does_not_mutate_existing(tmp_path: Path) -> None:
    path = _minimal_taxonomy(tmp_path)
    existing = load_taxonomy(path)
    original_count = len(existing.entries)
    merge_taxonomy(existing, [{"id": "new_one", "label": "New One", "cuad_origin": "New One"}])
    assert len(existing.entries) == original_count, "merge_taxonomy must not mutate the original"


def test_merge_ignores_malformed_upstream_entries(tmp_path: Path) -> None:
    """Malformed upstream entries (non-dict, missing id) are silently skipped."""
    path = _minimal_taxonomy(tmp_path)
    existing = load_taxonomy(path)
    upstream = [
        "not a dict",
        {"label": "Missing ID"},
        {"id": "", "label": "Empty ID"},
        {"id": "valid_new", "label": "Valid New", "cuad_origin": "Valid New"},
    ]
    merged = merge_taxonomy(existing, upstream)  # type: ignore[arg-type]
    assert merged.get("valid_new") is not None
    assert len(merged.entries) == len(existing.entries) + 1


# ---------------------------------------------------------------------------
# Simulated CUAD upgrade (acceptance criterion)
# ---------------------------------------------------------------------------


def test_simulated_cuad_upgrade(tmp_path: Path) -> None:
    """Core acceptance criterion: merge a simulated CUAD update against the affiliation taxonomy.

    - New CUAD categories must enter as inactive.
    - Existing curated statuses (active, inactive, custom) must not change.
    """
    existing = load_taxonomy(AFFILIATION_TAXONOMY)
    n_existing = len(existing.entries)
    existing_statuses = {e.id: e.status for e in existing.entries}

    # Simulate a newer CUAD release: includes all existing ids plus new ones
    simulated_new_cuad_entries = [
        # existing ids with "wrong" status (should be ignored)
        {
            "id": "indemnification",
            "label": "Indemnification",
            "status": "inactive",
            "cuad_origin": "Indemnification",
        },
        {
            "id": "most_favored_nation",
            "label": "Most Favored Nation",
            "status": "active",
            "cuad_origin": "Most Favored Nation",
        },
        # genuinely new ids
        {
            "id": "force_majeure",
            "label": "Force Majeure",
            "cuad_origin": "Force Majeure",
            "description": "New in CUAD v2",
        },
        {
            "id": "dispute_resolution",
            "label": "Dispute Resolution",
            "cuad_origin": "Dispute Resolution",
        },
    ]

    merged = merge_taxonomy(existing, simulated_new_cuad_entries)

    # All existing entries are preserved with unchanged status
    for entry_id, original_status in existing_statuses.items():
        merged_entry = merged.get(entry_id)
        assert merged_entry is not None, f"Existing entry {entry_id!r} was lost after merge"
        assert merged_entry.status == original_status, (
            f"Entry {entry_id!r} status changed from {original_status!r} to {merged_entry.status!r}"
        )

    # New entries are added as inactive
    assert merged.get("force_majeure") is not None
    assert merged.get("force_majeure").status == "inactive"  # type: ignore[union-attr]
    assert merged.get("dispute_resolution") is not None
    assert merged.get("dispute_resolution").status == "inactive"  # type: ignore[union-attr]

    # Total count is existing + 2 new (force_majeure, dispute_resolution)
    assert len(merged.entries) == n_existing + 2


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(TaxonomyError, match="not found"):
        load_taxonomy(tmp_path / "ghost.yaml")


def test_invalid_yaml_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("{{invalid", encoding="utf-8")
    with pytest.raises(TaxonomyError, match="not valid YAML"):
        load_taxonomy(bad)


def test_non_mapping_root_raises(tmp_path: Path) -> None:
    bad = tmp_path / "list.yaml"
    bad.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(TaxonomyError, match="must be a YAML mapping"):
        load_taxonomy(bad)


def test_missing_source_raises(tmp_path: Path) -> None:
    path = _write_taxonomy(tmp_path, {"entries": []})
    with pytest.raises(TaxonomyError, match="source"):
        load_taxonomy(path)


def test_invalid_status_raises(tmp_path: Path) -> None:
    path = _write_taxonomy(
        tmp_path,
        {
            "source": "test",
            "entries": [{"id": "bad_entry", "label": "Bad", "status": "unknown"}],
        },
    )
    with pytest.raises(TaxonomyError, match="status"):
        load_taxonomy(path)


def test_duplicate_id_raises(tmp_path: Path) -> None:
    path = _write_taxonomy(
        tmp_path,
        {
            "source": "test",
            "entries": [
                {"id": "dup", "label": "A", "status": "active"},
                {"id": "dup", "label": "B", "status": "inactive"},
            ],
        },
    )
    with pytest.raises(TaxonomyError, match="duplicate id"):
        load_taxonomy(path)


def test_custom_entry_with_cuad_origin_raises(tmp_path: Path) -> None:
    path = _write_taxonomy(
        tmp_path,
        {
            "source": "test",
            "entries": [
                {
                    "id": "my_entry",
                    "label": "My Entry",
                    "status": "custom",
                    "cuad_origin": "Indemnification",
                },
            ],
        },
    )
    with pytest.raises(TaxonomyError, match="cuad_origin: null"):
        load_taxonomy(path)


# ---------------------------------------------------------------------------
# CLI — playbook taxonomy merge
# ---------------------------------------------------------------------------


def test_cli_taxonomy_merge_dry_run(tmp_path: Path) -> None:
    from click.testing import CliRunner

    from playbook_engine.cli import cli

    tax_path = _minimal_taxonomy(tmp_path)
    upstream_path = tmp_path / "upstream.yaml"
    upstream_path.write_text(
        yaml.dump(
            {
                "source": "CUAD-v2",
                "entries": [
                    {
                        "id": "brand_new_cli",
                        "label": "Brand New CLI",
                        "cuad_origin": "Brand New CLI",
                    },
                    {"id": "indemnification", "label": "Indemnification", "status": "inactive"},
                ],
            }
        ),
        encoding="utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(
        cli, ["taxonomy", "merge", str(tax_path), str(upstream_path), "--dry-run"]
    )
    assert result.exit_code == 0, result.output
    assert "brand_new_cli" in result.output
    assert "[NEW]" in result.output
    # The taxonomy file must NOT be modified on disk for dry-run
    reloaded = load_taxonomy(tax_path)
    assert reloaded.get("brand_new_cli") is None


def test_cli_taxonomy_merge_writes_output(tmp_path: Path) -> None:
    from click.testing import CliRunner

    from playbook_engine.cli import cli

    tax_path = _minimal_taxonomy(tmp_path)
    upstream_path = tmp_path / "upstream.yaml"
    upstream_path.write_text(
        yaml.dump(
            {
                "source": "CUAD-v2",
                "entries": [
                    {"id": "new_clause", "label": "New Clause", "cuad_origin": "New Clause"}
                ],
            }
        ),
        encoding="utf-8",
    )
    out_path = tmp_path / "merged.yaml"

    runner = CliRunner()
    result = runner.invoke(
        cli, ["taxonomy", "merge", str(tax_path), str(upstream_path), "--out", str(out_path)]
    )
    assert result.exit_code == 0, result.output
    merged = load_taxonomy(out_path)
    assert merged.get("new_clause") is not None
    assert merged.get("new_clause").status == "inactive"  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# merge_taxonomy — new_source parameter (issue #37 fix)
# ---------------------------------------------------------------------------


def test_merge_source_updated_when_new_entries_added(tmp_path: Path) -> None:
    """new_source is applied when at least one new entry is added."""
    path = _minimal_taxonomy(tmp_path)
    existing = load_taxonomy(path)
    upstream = [{"id": "brand_new", "label": "Brand New", "cuad_origin": "Brand New"}]
    merged = merge_taxonomy(existing, upstream, new_source="CUAD-v2")
    assert merged.source == "CUAD-v2"


def test_merge_source_not_updated_when_no_new_entries(tmp_path: Path) -> None:
    """new_source is ignored when upstream adds no new ids."""
    path = _minimal_taxonomy(tmp_path)
    existing = load_taxonomy(path)
    # All upstream ids already exist — no new entries
    upstream = [
        {"id": "indemnification", "label": "Indemnification", "cuad_origin": "Indemnification"}
    ]
    merged = merge_taxonomy(existing, upstream, new_source="CUAD-v2")
    assert merged.source == existing.source, "source must not change when nothing was added"


def test_merge_source_unchanged_when_new_source_is_none(tmp_path: Path) -> None:
    """Default behavior: new_source=None never updates the source."""
    path = _minimal_taxonomy(tmp_path)
    existing = load_taxonomy(path)
    upstream = [{"id": "brand_new", "label": "Brand New", "cuad_origin": "Brand New"}]
    merged = merge_taxonomy(existing, upstream)
    assert merged.source == existing.source


def test_merge_source_unchanged_with_empty_upstream(tmp_path: Path) -> None:
    """new_source with empty upstream produces no change (no entries added)."""
    path = _minimal_taxonomy(tmp_path)
    existing = load_taxonomy(path)
    merged = merge_taxonomy(existing, [], new_source="CUAD-v99")
    assert merged.source == existing.source


def test_merge_empty_string_new_source_falls_back_to_existing(tmp_path: Path) -> None:
    """new_source='' is treated the same as None — existing source is preserved.

    This prevents writing a taxonomy with source: '' which load_taxonomy rejects.
    """
    path = _minimal_taxonomy(tmp_path)
    existing = load_taxonomy(path)
    upstream = [{"id": "brand_new", "label": "Brand New", "cuad_origin": "Brand New"}]
    merged = merge_taxonomy(existing, upstream, new_source="")
    assert merged.source == existing.source, (
        "empty new_source must never produce source:'' — load_taxonomy would reject it"
    )


# ---------------------------------------------------------------------------
# CLI — --new-source flag (issue #37 fix)
# ---------------------------------------------------------------------------


def test_cli_taxonomy_merge_new_source_updates_written_file(tmp_path: Path) -> None:
    """--new-source updates the source field in the output file when entries are added."""
    from click.testing import CliRunner

    from playbook_engine.cli import cli

    tax_path = _minimal_taxonomy(tmp_path)
    upstream_path = tmp_path / "upstream.yaml"
    upstream_path.write_text(
        yaml.dump(
            {
                "source": "CUAD-v2",
                "entries": [
                    {
                        "id": "force_majeure",
                        "label": "Force Majeure",
                        "cuad_origin": "Force Majeure",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    out_path = tmp_path / "merged.yaml"

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "taxonomy",
            "merge",
            str(tax_path),
            str(upstream_path),
            "--out",
            str(out_path),
            "--new-source",
            "CUAD-v2",
        ],
    )
    assert result.exit_code == 0, result.output
    merged = load_taxonomy(out_path)
    assert merged.source == "CUAD-v2"


def test_cli_taxonomy_merge_new_source_ignored_when_no_new_entries(tmp_path: Path) -> None:
    """--new-source has no effect when all upstream ids already exist."""
    from click.testing import CliRunner

    from playbook_engine.cli import cli

    tax_path = _minimal_taxonomy(tmp_path)
    existing = load_taxonomy(tax_path)
    upstream_path = tmp_path / "upstream.yaml"
    # Upstream only has entries that already exist
    upstream_path.write_text(
        yaml.dump(
            {
                "source": "CUAD-v2",
                "entries": [
                    {
                        "id": "indemnification",
                        "label": "Indemnification",
                        "cuad_origin": "Indemnification",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    out_path = tmp_path / "merged.yaml"

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "taxonomy",
            "merge",
            str(tax_path),
            str(upstream_path),
            "--out",
            str(out_path),
            "--new-source",
            "CUAD-v2",
        ],
    )
    assert result.exit_code == 0, result.output
    merged = load_taxonomy(out_path)
    assert merged.source == existing.source


def test_cli_taxonomy_merge_empty_new_source_preserves_source(tmp_path: Path) -> None:
    """--new-source '' must not write source: '' to the output file."""
    from click.testing import CliRunner

    from playbook_engine.cli import cli

    tax_path = _minimal_taxonomy(tmp_path)
    existing = load_taxonomy(tax_path)
    upstream_path = tmp_path / "upstream.yaml"
    upstream_path.write_text(
        yaml.dump(
            {
                "source": "CUAD-v2",
                "entries": [{"id": "new_entry", "label": "New Entry", "cuad_origin": "New Entry"}],
            }
        ),
        encoding="utf-8",
    )
    out_path = tmp_path / "merged.yaml"

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "taxonomy",
            "merge",
            str(tax_path),
            str(upstream_path),
            "--out",
            str(out_path),
            "--new-source",
            "",
        ],
    )
    assert result.exit_code == 0, result.output
    merged = load_taxonomy(out_path)
    assert merged.source == existing.source


def test_cli_taxonomy_merge_no_new_source_flag_preserves_source(tmp_path: Path) -> None:
    """Without --new-source the source field is always preserved."""
    from click.testing import CliRunner

    from playbook_engine.cli import cli

    tax_path = _minimal_taxonomy(tmp_path)
    existing = load_taxonomy(tax_path)
    upstream_path = tmp_path / "upstream.yaml"
    upstream_path.write_text(
        yaml.dump(
            {
                "source": "CUAD-v2",
                "entries": [{"id": "new_entry", "label": "New Entry", "cuad_origin": "New Entry"}],
            }
        ),
        encoding="utf-8",
    )
    out_path = tmp_path / "merged.yaml"

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["taxonomy", "merge", str(tax_path), str(upstream_path), "--out", str(out_path)],
    )
    assert result.exit_code == 0, result.output
    merged = load_taxonomy(out_path)
    assert merged.source == existing.source
