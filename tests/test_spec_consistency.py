"""Spec/schema consistency guards (issues #162, #163).

The v0.2 spec prose must describe the Floor that actually ships: judged
NL invariants (schema `floor.invariants[] = {id, statement, rationale}`,
`floor_judge.py`), not the superseded lexical-detector design decided out
on 2026-07-09 (#145). These are lasting guards, not a one-time grep — the
lexical vocabulary must never reappear, and the spec's own floor example
must validate against the shipped schema.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import jsonschema
import pytest

ROOT = Path(__file__).parent.parent
SCHEMA_PATH = ROOT / "spec" / "playbook.schema-0.2.json"

# After #172 renames the draft, whatever OPF-SPEC*.md exists is in scope.
SPEC_PATHS = sorted((ROOT / "docs").glob("OPF-SPEC*.md"))

_LEXICAL_FLOOR_VOCABULARY = [
    "trigger_terms",
    "hard_rejections",
    "on_insert",
    "on_remove_or_alter",
    "exempt_terms",
    "must_preserve",
]


def _spec_texts() -> list[tuple[Path, str]]:
    assert SPEC_PATHS, "no docs/OPF-SPEC*.md found"
    return [(p, p.read_text(encoding="utf-8")) for p in SPEC_PATHS]


def test_spec_has_no_lexical_floor_vocabulary() -> None:
    """§3.7/§3.7.1/§5 must describe the judged-invariant Floor — none of the
    superseded detector grammar's vocabulary may appear anywhere in the spec."""
    for path, text in _spec_texts():
        for term in _LEXICAL_FLOOR_VOCABULARY:
            assert term not in text, (
                f"{path.name} still contains lexical-Floor vocabulary {term!r} — "
                "the shipped Floor is judged NL invariants (#145/#162)"
            )


def _floor_fences(text: str) -> list[str]:
    """Return jsonc code fences that define a floor block (the judged-invariant
    Floor's defining key is `invariants`; fences that merely mention a `floor`
    digest string, e.g. §3.10's identity example, are not floor examples)."""
    fences = re.findall(r"```jsonc?\n(.*?)```", text, flags=re.DOTALL)
    return [f for f in fences if '"invariants"' in f]


def _strip_jsonc(fence: str) -> str:
    """Make the spec's jsonc examples parseable: strip //-comments and a
    leading `"floor": ` label, wrap bare objects."""
    no_comments = re.sub(r"//[^\n]*", "", fence)
    body = no_comments.strip()
    if body.startswith('"floor"'):
        body = "{" + body.rstrip(",") + "}"
    return body


def test_spec_floor_example_validates() -> None:
    """The spec's own floor example must validate against the shipped schema's
    floor definition."""
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    # The floor definition is self-contained (invariants are inline, no $refs),
    # so it validates standalone.
    floor_schema = schema["properties"]["floor"]

    validated = 0
    for path, text in _spec_texts():
        for fence in _floor_fences(text):
            try:
                doc = json.loads(_strip_jsonc(fence))
            except json.JSONDecodeError:
                continue  # fence is a fragment of a larger example, not a floor doc
            floor = doc.get("floor", doc)
            errors = list(
                jsonschema.validators.validator_for(schema)(floor_schema).iter_errors(floor)
            )
            assert not errors, (
                f"{path.name} floor example does not validate against the shipped "
                f"schema: {[e.message for e in errors]}"
            )
            validated += 1
    if not validated:
        pytest.fail("no parseable floor example found in the spec — §3.7 must carry one")


# ---------------------------------------------------------------------------
# #163 — spec/schema editorial reconciliation guards
# ---------------------------------------------------------------------------


def _draft_spec_text() -> str:
    draft = ROOT / "docs" / "OPF-SPEC-v0.2-DRAFT.md"
    if draft.exists():
        return draft.read_text(encoding="utf-8")
    # #172 promoted the v0.2 draft to the canonical docs/OPF-SPEC.md filename
    # (the v0.1 predecessor was archived to docs/OPF-SPEC-v0.1.md, which also
    # matches the OPF-SPEC*.md glob — so prefer the canonical name explicitly
    # rather than falling back to a concatenation that could pick up v0.1's
    # same-numbered section headers first).
    canonical = ROOT / "docs" / "OPF-SPEC.md"
    if canonical.exists():
        return canonical.read_text(encoding="utf-8")
    return "\n".join(text for _, text in _spec_texts())


def test_every_schema_toplevel_property_documented() -> None:
    """A spec reader must be able to learn every top-level field exists: each
    key in the schema's top-level `properties` appears (backticked or as a
    JSON key) in the spec's §3 document-structure section."""
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    text = _draft_spec_text()
    start = text.index("## 3. Document structure")
    end = text.index("## 4.")
    section = text[start:end]
    for key in schema["properties"]:
        assert f"`{key}`" in section or f'"{key}"' in section, (
            f"schema top-level property {key!r} is not documented in the spec's "
            "§3 document-structure section (#163)"
        )


def test_conformance_section_names_v02_schema() -> None:
    """§10 must point at the real v0.2 schema file, not the v0.1 one."""
    text = _draft_spec_text()
    start = text.index("## 10. Conformance")
    end = text.index("## 11.")
    section = text[start:end]
    assert "playbook.schema-0.2.json" in section
    assert "playbook.schema.json` (to be updated" not in section


def test_content_hash_description_lists_curation() -> None:
    """The schema's content_hash description must state the curation exclusion
    (spec §3.10 and canonicalize.py both exclude it)."""
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    description = schema["properties"]["identity"]["properties"]["content_hash"]["description"]
    assert "curation" in description


# ---------------------------------------------------------------------------
# #172 — docs coherence for launch: README authority chain, supersession
# ---------------------------------------------------------------------------

README_PATH = ROOT / "README.md"
_MD_RELATIVE_LINK_RE = re.compile(r"\]\(((?:docs|spec)/[^)#]+)(?:#[^)]*)?\)")


def test_readme_links_resolve() -> None:
    """Every docs/... or spec/... relative link in README must exist on disk —
    a first-time reader should never hit a dead keystone pointer."""
    text = README_PATH.read_text(encoding="utf-8")
    links = _MD_RELATIVE_LINK_RE.findall(text)
    assert links, "expected at least one docs/spec relative link in README"
    for link in links:
        target = ROOT / link
        assert target.exists(), f"README links to {link!r}, which does not exist on disk"


def test_readme_claims_v02() -> None:
    """README's Status section must state OPF is at 0.2, not the stale v0.1
    draft claim."""
    text = README_PATH.read_text(encoding="utf-8")
    start = text.index("## Status")
    end = text.index("## License")
    section = text[start:end]
    assert "0.2" in section
    assert "OPF is at `v0.1`" not in section


def test_superseded_banner() -> None:
    """docs/OPF-SPEC-v0.1.md must carry a supersession banner near the top so
    a reader who lands on it directly knows not to implement against it."""
    v01_path = ROOT / "docs" / "OPF-SPEC-v0.1.md"
    first_five_lines = "\n".join(v01_path.read_text(encoding="utf-8").splitlines()[:5])
    assert "SUPERSEDED" in first_five_lines
