"""Config loader — per-agreement-type configuration.

Config schema (YAML):
  agreement_type:
    id: educational-affiliation       # slug pattern ^[a-z0-9-]+$; self-assigned,
                                       # optionally namespaced (e.g. "fixturecorp-eiaa")
                                       # — see docs/OPF-SPEC.md §3.1
    name: "Educational Affiliation Agreement"
    description: "..."               # optional
    aliases: ["eiaa"]                 # optional; other names/keys this same
                                       # agreement type is known by elsewhere
                                       # (e.g. a consuming app's own dial key)
  baseline:
    template: ./template/file.rtf    # path relative to config file, or null
  taxonomy: ./spec/taxonomy/file.yaml
    # or: builtin:file.yaml          # resolves against the engine's own
    #                                  bundled spec/taxonomy/ dir (see
    #                                  ``_BUILTIN_TAXONOMY_DIR`` below),
    #                                  regardless of where this config file
    #                                  lives — use this instead of a
    #                                  repo-relative ``../../spec/...`` path
    #                                  so the config stays valid after being
    #                                  copied next to a corpus (issue #130).
  perspective:                       # optional; whose "us" this playbook is
                                      # reviewed as (issue #165) — an
                                      # open-standard OPF instance must say
                                      # who "us" is (spec/playbook.schema-0.2.json
                                      # perspective). Omit entirely and
                                      # ``party`` defaults from
                                      # provenance.our_party_aliases[0] below;
                                      # ``counterparty_type`` has no default
                                      # (never fabricated) and stays unset
                                      # until you supply it. Both fields must
                                      # be present for the projected playbook
                                      # to actually carry a ``perspective``
                                      # block — a party-only default is
                                      # config-level only, since the schema
                                      # requires the whole object or nothing.
    party: "FixtureCorp"
    counterparty_type: "Educational Institution"
  provenance:
    our_party_aliases: ["FixtureCorp", "FixtureCorp Holdings, LLC"]
    known_entities: ["State University"]
      # Known counterparty entity names to pseudonymize at ingest (issue #153)
      # — human-curated from the corpus manifest/folder names, same workflow
      # our_party_aliases already uses. Every occurrence in stored clause
      # text/summaries/document_ids/citations is deterministically replaced
      # with a stable alias (playbook_engine.entity_registry) before the
      # observation store is written, so playbook.opf.json never carries a
      # raw name. Defaults to an empty list (no pseudonymization) when omitted.
    min_evidence_n: 2         # optional; minimum distinct our-paper observations
      # required before a clause may carry a position/historical_stance
      # stronger than "negotiable"/"mixed" (issue #144, OPF §2.2). Defaults to
      # clause_position_compiler.MIN_EVIDENCE_N (2) when omitted. Must be a
      # positive integer. The compiler and validator both enforce this same
      # threshold — see playbook_engine.clause_position_compiler._derive_rollup
      # and playbook_engine.validator._check_evidence_depth_rule_v2.
  segmentation:                      # optional; omit entirely to keep today's
                                      # deterministic-only behavior unchanged
    llm: true                        # opt in to LLM-first segmentation; default false
    batch: true                      # use segment_documents_batch (Message Batches)
    cache: true                      # cache LLM segmentation verdicts AND extracted
                                      # blocks/clause trees on disk (issue #132)
    normalize_trail: false           # opt in to LLM-based cross-version trail normalization
    model: claude-opus-4-8           # optional; model id override (issue #131). Defaults to
                                      # llm_segmenter.DEFAULT_MODEL — data, not a code change,
                                      # so switching models (e.g. to a Bedrock-hosted model id)
                                      # never requires editing the engine itself.
  classification:                    # optional; all keys optional — omit entirely to
                                      # keep today's clause_classifier.py constants
                                      # (issue #168 — "type knowledge lives in config")
    ambiguity_threshold: 0.70         # below this Jaccard similarity a clause heading
                                      # is auto-unclassified rather than escalated to
                                      # the classification judge. Defaults to
                                      # clause_classifier.AMBIGUITY_THRESHOLD (0.70).
    auto_classify_threshold: 0.85     # at or above this Jaccard similarity a clause
                                      # heading is auto-classified without the judge.
                                      # Defaults to clause_classifier.AUTO_CLASSIFY_THRESHOLD
                                      # (0.85). Must satisfy
                                      # 0 < ambiguity_threshold <= auto_classify_threshold <= 1.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from playbook_engine.clause_classifier import AMBIGUITY_THRESHOLD, AUTO_CLASSIFY_THRESHOLD
from playbook_engine.clause_position_compiler import MIN_EVIDENCE_N
from playbook_engine.llm_segmenter import DEFAULT_MODEL

# Bundled taxonomies shipped with the engine itself (as opposed to a
# corpus-specific taxonomy authored by a user). Resolved relative to the
# *package's* own location — same pattern as ``validator.py``'s
# ``_SCHEMA_PATH`` — so it is independent of the config file's location or
# the process's current working directory.
_BUILTIN_TAXONOMY_DIR = Path(__file__).resolve().parent.parent / "spec" / "taxonomy"

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class AgreementType:
    id: str
    name: str
    description: str = ""
    aliases: list[str] = field(default_factory=list)


@dataclass
class BaselineConfig:
    has_canonical_template: bool
    template_path: Path | None  # absolute; None when has_canonical_template is False


@dataclass
class PerspectiveConfig:
    """Issue #165: whose "us" this playbook is reviewed as.

    ``party`` defaults from ``provenance.our_party_aliases[0]`` when the
    config's ``perspective:`` block (or the block itself) omits it —
    ``None`` only when neither is supplied. ``counterparty_type`` has no
    derivable default (never fabricated) and stays ``None`` until a config
    sets it explicitly.

    A ``PerspectiveConfig`` with only ``party`` set is a valid, useful
    config-level value (e.g. for future callers), but is NOT enough on its
    own to populate the assembled playbook's top-level ``perspective`` key:
    ``spec/playbook.schema-0.2.json`` requires ``party`` AND
    ``counterparty_type`` together or neither — see
    ``pipeline.project_playbook``.
    """

    party: str | None = None
    counterparty_type: str | None = None


@dataclass
class ProvenanceConfig:
    our_party_aliases: list[str] = field(default_factory=list)
    known_entities: list[str] = field(default_factory=list)
    # Issue #144: producer-configurable evidence-depth floor. Defaults to
    # clause_position_compiler.MIN_EVIDENCE_N so an existing config with no
    # ``min_evidence_n`` key keeps enforcing the same threshold it always has.
    min_evidence_n: int = MIN_EVIDENCE_N


@dataclass
class SegmentationConfig:
    """Opt-in LLM-first segmentation settings for ``playbook mine``.

    The boolean fields default to ``False`` so a config file with no
    ``segmentation:`` section (every existing fixture) preserves today's
    deterministic-only behavior exactly. ``model`` defaults to
    :data:`~playbook_engine.llm_segmenter.DEFAULT_MODEL` so an existing
    ``segmentation:`` block with no ``model:`` key keeps calling the same
    model it always has (issue #131 — this field makes the model *data*,
    letting a corpus override it, e.g. to a Bedrock-hosted model id, without
    an engine code change).
    """

    llm: bool = False
    batch: bool = False
    cache: bool = False
    normalize_trail: bool = False
    model: str = DEFAULT_MODEL
    # Agent-as-segmenter (issue #191): the AGENT segments key-free via the
    # `segment`/`segment-apply` store-backed loop instead of a live LLM call.
    # Implies llm-first + cache; no ANTHROPIC_API_KEY is used. Defaults False
    # so existing configs are unchanged.
    agent: bool = False


@dataclass
class ClassificationConfig:
    """Producer-configurable clause-classifier confidence bands (issue #168).

    Mirrors the ``provenance.min_evidence_n`` pattern: both fields default to
    ``clause_classifier.py``'s module constants, so a config with no
    ``classification:`` block (every existing fixture) preserves today's
    classification banding exactly.
    """

    ambiguity_threshold: float = AMBIGUITY_THRESHOLD
    auto_classify_threshold: float = AUTO_CLASSIFY_THRESHOLD


@dataclass
class EngineConfig:
    agreement_type: AgreementType
    baseline: BaselineConfig
    taxonomy_path: Path  # absolute
    provenance: ProvenanceConfig
    perspective: PerspectiveConfig
    segmentation: SegmentationConfig
    classification: ClassificationConfig
    config_path: Path  # absolute path to the config file itself


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


class ConfigError(ValueError):
    """Friendly configuration error."""


def _require(mapping: dict[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise ConfigError(f"{context}: required field '{key}' is missing")
    return mapping[key]


_BUILTIN_SCHEME = "builtin:"


def _resolve_taxonomy_path(tax_val: str, base_dir: Path) -> Path:
    """Resolve ``config.taxonomy`` to an absolute path.

    Two forms are accepted:

    - ``builtin:<name>`` — resolved against the engine's own bundled
      ``spec/taxonomy/`` directory, independent of *base_dir*. Use this for
      taxonomies that ship with the engine, so a config referencing one
      keeps working after being copied anywhere (next to a corpus, into a
      container mount, etc.) — see issue #130.
    - anything else — a path relative to *base_dir* (the config file's own
      parent directory), as before.
    """
    if tax_val.startswith(_BUILTIN_SCHEME):
        name = tax_val[len(_BUILTIN_SCHEME) :].strip()
        if not name:
            raise ConfigError(
                "config.taxonomy: 'builtin:' scheme requires a name, "
                "e.g. 'builtin:affiliation-agreement.yaml'"
            )
        candidate = _BUILTIN_TAXONOMY_DIR / name
        if not candidate.is_file():
            available = (
                sorted(p.name for p in _BUILTIN_TAXONOMY_DIR.glob("*.yaml"))
                if _BUILTIN_TAXONOMY_DIR.is_dir()
                else []
            )
            raise ConfigError(
                f"builtin taxonomy {name!r} not found in {_BUILTIN_TAXONOMY_DIR} "
                f"(available: {', '.join(available) or 'none'})"
            )
        return candidate.resolve()

    tax_path = (base_dir / tax_val).resolve()
    if not tax_path.is_file():
        raise ConfigError(f"taxonomy file not found (or is not a file): {tax_path}")
    return tax_path


def resolve_taxonomy_path(tax_val: str, base_dir: Path) -> Path:
    """Public wrapper for taxonomy resolution (``builtin:`` scheme + relative
    paths). Lets other modules (e.g. ``corpus_linter``) resolve a config's
    taxonomy the SAME way ``load_config`` does — so a ``builtin:`` value that
    the loader accepts is not rejected as a literal path elsewhere (issue #182).
    Raises :class:`ConfigError` if the taxonomy cannot be resolved.
    """
    return _resolve_taxonomy_path(tax_val, base_dir)


def load_config(path: Path) -> EngineConfig:
    """Load and validate an engine config YAML file.

    All relative paths in the config are resolved relative to the config file's
    parent directory.
    """
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")

    raw_text = path.read_text(encoding="utf-8")
    try:
        raw: dict[str, Any] = yaml.safe_load(raw_text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Config file is not valid YAML: {exc}") from exc

    if not isinstance(raw, dict):
        raise ConfigError(f"Config file must be a YAML mapping, got {type(raw).__name__}")

    base_dir = path.parent

    # --- agreement_type ---
    at_raw = _require(raw, "agreement_type", "config")
    if not isinstance(at_raw, dict):
        raise ConfigError("config.agreement_type must be a mapping")
    at_id = _require(at_raw, "id", "agreement_type")
    at_name = _require(at_raw, "name", "agreement_type")
    if not isinstance(at_id, str) or not at_id:
        raise ConfigError("agreement_type.id must be a non-empty string")
    if not isinstance(at_name, str) or not at_name:
        raise ConfigError("agreement_type.name must be a non-empty string")
    if not re.match(r"^[a-z0-9-]+$", at_id):
        raise ConfigError(f"agreement_type.id {at_id!r} must match ^[a-z0-9-]+$ (lowercase slug)")
    at_aliases_raw = at_raw.get("aliases", [])
    if not isinstance(at_aliases_raw, list):
        raise ConfigError("agreement_type.aliases must be a list")
    at_aliases = [str(a) for a in at_aliases_raw]
    agreement_type = AgreementType(
        id=at_id,
        name=at_name,
        description=str(at_raw.get("description", "")),
        aliases=at_aliases,
    )

    # --- baseline ---
    bl_raw = _require(raw, "baseline", "config")
    if not isinstance(bl_raw, dict):
        raise ConfigError("config.baseline must be a mapping")
    template_val = bl_raw.get("template")
    if template_val is None:
        baseline = BaselineConfig(has_canonical_template=False, template_path=None)
    else:
        tpl_path = (base_dir / template_val).resolve()
        if not tpl_path.is_file():
            raise ConfigError(f"baseline.template not found (or is not a file): {tpl_path}")
        baseline = BaselineConfig(has_canonical_template=True, template_path=tpl_path)

    # --- taxonomy ---
    tax_val = _require(raw, "taxonomy", "config")
    if not isinstance(tax_val, str) or not tax_val:
        raise ConfigError("config.taxonomy must be a non-empty path string")
    tax_path = _resolve_taxonomy_path(tax_val, base_dir)

    # --- provenance ---
    prov_raw = raw.get("provenance", {})
    if not isinstance(prov_raw, dict):
        raise ConfigError("config.provenance must be a mapping")
    aliases_raw = prov_raw.get("our_party_aliases", [])
    if not isinstance(aliases_raw, list):
        raise ConfigError("provenance.our_party_aliases must be a list")
    aliases = [str(a) for a in aliases_raw]

    known_entities_raw = prov_raw.get("known_entities", [])
    if not isinstance(known_entities_raw, list):
        raise ConfigError("provenance.known_entities must be a list")
    known_entities = [str(e) for e in known_entities_raw]

    min_evidence_n_raw = prov_raw.get("min_evidence_n", MIN_EVIDENCE_N)
    if isinstance(min_evidence_n_raw, bool) or not isinstance(min_evidence_n_raw, int):
        raise ConfigError("provenance.min_evidence_n must be a positive integer")
    if min_evidence_n_raw < 1:
        raise ConfigError("provenance.min_evidence_n must be a positive integer")

    # --- perspective (issue #165) ---
    persp_raw = raw.get("perspective", {})
    if not isinstance(persp_raw, dict):
        raise ConfigError("config.perspective must be a mapping")
    persp_party_raw = persp_raw.get("party")
    if persp_party_raw is not None and (
        not isinstance(persp_party_raw, str) or not persp_party_raw
    ):
        raise ConfigError("perspective.party must be a non-empty string")
    persp_counterparty_raw = persp_raw.get("counterparty_type")
    if persp_counterparty_raw is not None and (
        not isinstance(persp_counterparty_raw, str) or not persp_counterparty_raw
    ):
        raise ConfigError("perspective.counterparty_type must be a non-empty string")
    # Default party from provenance.our_party_aliases[0] when unset — a config
    # that already declares "who we are" for pseudonymization shouldn't have
    # to repeat it to also answer "who is 'us'" for the playbook.
    # counterparty_type has no derivable default: never fabricated.
    resolved_party = (
        persp_party_raw if persp_party_raw is not None else (aliases[0] if aliases else None)
    )
    perspective = PerspectiveConfig(party=resolved_party, counterparty_type=persp_counterparty_raw)

    # --- segmentation ---
    seg_raw = raw.get("segmentation", {})
    if not isinstance(seg_raw, dict):
        raise ConfigError("config.segmentation must be a mapping")
    model_val = seg_raw.get("model", DEFAULT_MODEL)
    if not isinstance(model_val, str) or not model_val:
        raise ConfigError("segmentation.model must be a non-empty string")

    agent_seg = bool(seg_raw.get("agent", False))
    if agent_seg:
        # Agent-as-segmenter (issue #191) makes no LLM call, so the model field
        # is meaningless as an Anthropic id — but it is still the segmentation
        # cache key, and `segment`/`segment-apply`/`mine` must all agree on it.
        # Force the sentinel (mirrors agent_segmenter.AGENT_SEGMENTER_MODEL) so
        # agent-produced cache entries never collide with a real model's, and
        # so it also implies llm-first + cache (the store-backed replay path).
        model_val = "store-backed-agent"

    segmentation = SegmentationConfig(
        llm=bool(seg_raw.get("llm", False)) or agent_seg,
        batch=bool(seg_raw.get("batch", False)),
        cache=bool(seg_raw.get("cache", False)) or agent_seg,
        normalize_trail=bool(seg_raw.get("normalize_trail", False)),
        model=model_val,
        agent=agent_seg,
    )

    # --- classification (issue #168) ---
    cls_raw = raw.get("classification", {})
    if not isinstance(cls_raw, dict):
        raise ConfigError("config.classification must be a mapping")

    def _threshold(raw_val: Any, key: str) -> float:
        if isinstance(raw_val, bool) or not isinstance(raw_val, (int, float)):
            raise ConfigError(f"classification.{key} must be a number")
        return float(raw_val)

    ambiguity_threshold_raw = _threshold(
        cls_raw.get("ambiguity_threshold", AMBIGUITY_THRESHOLD), "ambiguity_threshold"
    )
    auto_classify_threshold_raw = _threshold(
        cls_raw.get("auto_classify_threshold", AUTO_CLASSIFY_THRESHOLD),
        "auto_classify_threshold",
    )
    if not (0 < ambiguity_threshold_raw <= auto_classify_threshold_raw <= 1):
        raise ConfigError(
            "classification.ambiguity_threshold and classification.auto_classify_threshold "
            "must satisfy 0 < ambiguity_threshold <= auto_classify_threshold <= 1 "
            f"(got ambiguity_threshold={ambiguity_threshold_raw}, "
            f"auto_classify_threshold={auto_classify_threshold_raw})"
        )

    classification = ClassificationConfig(
        ambiguity_threshold=ambiguity_threshold_raw,
        auto_classify_threshold=auto_classify_threshold_raw,
    )

    return EngineConfig(
        agreement_type=agreement_type,
        baseline=baseline,
        taxonomy_path=tax_path,
        provenance=ProvenanceConfig(
            our_party_aliases=aliases,
            known_entities=known_entities,
            min_evidence_n=min_evidence_n_raw,
        ),
        perspective=perspective,
        segmentation=segmentation,
        classification=classification,
        config_path=path.resolve(),
    )
