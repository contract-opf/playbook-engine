"""Model-facing digest of an OPF playbook — the OPF 0.3 `digest` section.

The full OPF document carries every observation's ``full_text`` and measures
in the millions of characters on a real corpus — far beyond what a consuming
review application can put in a model's context. The digest is the compact
projection designed for exactly that use: per clause, the stance, the
preferred/concession/unacceptable variation summaries, and a deduplicated,
frequency-annotated sample of exemplar forms, each carrying an
``example_ref`` citation that resolves into the full playbook for on-demand
drill-down (a consumer fetches ``full_text`` from the full OPF when needed —
the digest itself never contains ``full_text``).

Emitted by ``assemble_playbook`` as the top-level ``digest`` section of an
OPF 0.3 document, and extractable standalone via ``playbook digest``. The
digest is a pure function of the evidence section, so it participates in
``identity.content_hash`` like any other content section.

Size discipline: the budget is ~40K tokens (chars/4 rule of thumb — this
codebase has no tokenizer dependency) and is ENFORCED by construction, not
aspirational: every list — preferred variations, concessions, unacceptable
variations, exemplar forms — is deduplicated by normalized text and capped
at the top-N by evidentiary weight (precedent-count-weighted ``n``) plus
every material-risk group; if the digest still exceeds the budget,
``build_digest`` tightens the cap stepwise (5 → 4 → 3) until it fits.
Surviving entries are never truncated or paraphrased — a preferred
variation's ``if``/``to`` language ships verbatim; only the compiler-
generated ``rationale`` narration is left to the full OPF (reachable via
``observation_ref``).
"""

from __future__ import annotations

import re
from typing import Any

from playbook_engine.canonicalize import canonicalize
from playbook_engine.opf_accessors import clause_stance, playbook_clauses

#: Schema version of the digest section itself — bump on any shape change so
#: consumers can dispatch (the digest is consumed outside this repo).
#: v2: preferred_variations deduped/ranked/capped like the other lists; digest
#: entries carry {if, to, observation_ref, n, band} (rationale stays in the
#: full OPF).
DIGEST_VERSION = "2"

#: List selection (all four lists): keep the top N deduplicated entries by
#: observation count, plus every entry carrying material risk regardless of
#: rank. This is the default/loosest cap; build_digest tightens it to fit
#: the token budget.
EXEMPLAR_TOP_N = 5

#: The hard size budget build_digest enforces (chars/4 rule of thumb).
DIGEST_TOKEN_BUDGET = 40_000

#: build_digest never tightens the per-list cap below this.
_MIN_TOP_N = 3

#: Frequency bands for exemplar forms — coarse language a reviewing model can
#: use directly ("often signed as…") without re-deriving statistics.
_BAND_OFTEN_MIN = 10
_BAND_SOMETIMES_MIN = 2


def _normalize_text(text: str) -> str:
    """Normalization used to dedupe exemplar forms (case/punct/ws-insensitive)."""
    s = text.lower()
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _band(n: int) -> str:
    if n >= _BAND_OFTEN_MIN:
        return "often"
    if n >= _BAND_SOMETIMES_MIN:
        return "sometimes"
    return "rare"


def _is_material(obs: dict[str, Any]) -> bool:
    risk = obs.get("risk_delta") or {}
    return isinstance(risk, dict) and risk.get("magnitude") == "material"


def _dedupe_rank(
    observations: list[dict[str, Any]], *, include_deviation: bool, top_n: int = EXEMPLAR_TOP_N
) -> list[dict[str, Any]]:
    """Dedupe observations by normalized text and rank by frequency.

    The digest's one size discipline, applied uniformly to exemplar forms,
    concessions, and unacceptable variations: group by the normalized
    ``full_text`` (falling back to ``text_summary``); ``n`` sums each
    member's ``precedent_count`` (default 1 — the compiler already folds
    exact-duplicate observations into one position with a count); keep the
    top ``EXEMPLAR_TOP_N`` groups by ``n`` plus every group containing
    material risk, in rank order. Output entries carry ``text_summary``
    ONLY — never ``full_text``; ``example_ref`` is the drill-down path.
    """
    groups: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for obs in observations:
        key = _normalize_text(str(obs.get("full_text") or obs.get("text_summary") or ""))
        if not key:
            continue
        if key not in groups:
            groups[key] = {"n": 0, "rep": obs, "material": False}
            order.append(key)
        g = groups[key]
        g["n"] += int(obs.get("precedent_count") or 1)
        if _is_material(obs):
            g["material"] = True
            # A material observation is the most informative representative.
            g["rep"] = obs

    first_seen = {k: i for i, k in enumerate(order)}
    ranked = sorted(order, key=lambda k: (-groups[k]["n"], first_seen[k]))
    keep = set(ranked[:top_n]) | {k for k in ranked if groups[k]["material"]}

    forms: list[dict[str, Any]] = []
    for key in ranked:
        if key not in keep:
            continue
        g = groups[key]
        rep = g["rep"]
        form: dict[str, Any] = {
            "text_summary": rep.get("text_summary", ""),
            "n": g["n"],
            "band": _band(g["n"]),
        }
        if include_deviation and rep.get("deviation") is not None:
            form["deviation"] = rep["deviation"]
        if rep.get("risk_delta") is not None:
            form["risk_delta"] = rep["risk_delta"]
        if rep.get("example_ref") is not None:
            form["example_ref"] = rep["example_ref"]
        forms.append(form)
    return forms


def _exemplar_forms(
    observed_positions: list[dict[str, Any]], top_n: int = EXEMPLAR_TOP_N
) -> list[dict[str, Any]]:
    return _dedupe_rank(observed_positions, include_deviation=True, top_n=top_n)


def _preferred_variations(clause: dict[str, Any], top_n: int) -> list[Any]:
    """Project acceptable_if entries to the digest: dedupe, rank, cap.

    Same discipline as the other three lists. Grouping key: the normalized
    ``if``+``to`` text (or the whole entry for legacy bare strings). Rank
    weight ``n``: the ``precedent_count`` of the underlying observation,
    resolved by matching ``observation_ref`` against the clause's own
    ``observed_positions`` (1 when unresolvable). Surviving dict entries ship
    ``if``/``to`` VERBATIM plus ``observation_ref``, ``n``, and ``band`` —
    the compiler-generated ``rationale`` narration stays in the full OPF.
    Legacy bare-string entries pass through as strings.
    """
    entries = (clause.get("summary") or {}).get("acceptable_if") or []
    if not entries:
        return []

    obs_by_ref: dict[tuple[Any, Any, Any], dict[str, Any]] = {}
    for pos in clause.get("observed_positions") or []:
        pos_ref = pos.get("example_ref") or {}
        obs_by_ref[
            (pos_ref.get("document_id"), pos_ref.get("version"), pos_ref.get("clause_path"))
        ] = pos

    groups: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for entry in entries:
        obs: dict[str, Any] = {}
        if isinstance(entry, str):
            key = _normalize_text(entry)
        else:
            key = _normalize_text(f"{entry.get('if', '')} {entry.get('to', '')}")
            ref = entry.get("observation_ref") or {}
            obs = obs_by_ref.get(
                (ref.get("document_id"), ref.get("version"), ref.get("clause_path")), {}
            )
        if not key:
            continue
        n = int(obs.get("precedent_count") or 1)
        if key not in groups:
            groups[key] = {"n": 0, "rep": entry, "rep_n": -1, "material": False}
            order.append(key)
        g = groups[key]
        g["n"] += n
        if _is_material(obs):
            g["material"] = True
        if n > g["rep_n"]:
            g["rep"], g["rep_n"] = entry, n

    first_seen = {k: i for i, k in enumerate(order)}
    ranked = sorted(order, key=lambda k: (-groups[k]["n"], first_seen[k]))
    keep = set(ranked[:top_n]) | {k for k in ranked if groups[k]["material"]}

    out: list[Any] = []
    for key in ranked:
        if key not in keep:
            continue
        g = groups[key]
        rep = g["rep"]
        if isinstance(rep, str):
            out.append(rep)
            continue
        projected: dict[str, Any] = {"if": rep.get("if"), "to": rep.get("to")}
        if rep.get("observation_ref") is not None:
            projected["observation_ref"] = rep["observation_ref"]
        projected["n"] = g["n"]
        projected["band"] = _band(g["n"])
        out.append(projected)
    return out


def _build_digest_at(playbook: dict[str, Any], top_n: int) -> dict[str, Any]:
    """Build the digest with a fixed per-list cap of *top_n*."""
    digest_clauses: list[dict[str, Any]] = []
    for clause in playbook_clauses(playbook):
        summary = clause.get("summary") or {}
        our_standard = clause.get("our_standard")
        entry: dict[str, Any] = {
            "id": clause.get("id"),
            "taxonomy_id": clause.get("taxonomy_id"),
            "title": clause.get("title"),
            "historical_stance": clause_stance(clause),
            "stance_detail": summary.get("stance_detail"),
            "our_standard": our_standard if isinstance(our_standard, dict) else None,
            "preferred_variations": _preferred_variations(clause, top_n),
            "concessions": _dedupe_rank(
                summary.get("fallbacks") or [], include_deviation=False, top_n=top_n
            ),
            "unacceptable": _dedupe_rank(
                summary.get("rejected") or [], include_deviation=False, top_n=top_n
            ),
            "exemplar_forms": _exemplar_forms(clause.get("observed_positions") or [], top_n),
        }
        digest_clauses.append(entry)

    return {
        "digest_version": DIGEST_VERSION,
        "clause_count": len(digest_clauses),
        "clauses": digest_clauses,
    }


def build_digest(
    playbook: dict[str, Any], *, token_budget: int | None = DIGEST_TOKEN_BUDGET
) -> dict[str, Any]:
    """Build the digest section from an assembled playbook's evidence.

    Works on any document ``playbook_clauses`` understands (0.2 or 0.3), so
    the CLI can also derive a digest from a pre-0.3 artifact.

    Enforces *token_budget* by construction: starts at the default per-list
    cap (``EXEMPLAR_TOP_N``) and tightens it stepwise down to ``_MIN_TOP_N``
    until the digest fits. Pass ``token_budget=None`` for the loosest cap
    unconditionally. On an extreme corpus even the tightest cap can exceed
    the budget (material-risk entries are never dropped) — the CLI warns in
    that case.
    """
    digest = _build_digest_at(playbook, EXEMPLAR_TOP_N)
    if token_budget is None:
        return digest
    for top_n in range(EXEMPLAR_TOP_N - 1, _MIN_TOP_N - 1, -1):
        if digest_token_estimate(digest) <= token_budget:
            break
        digest = _build_digest_at(playbook, top_n)
    return digest


def digest_token_estimate(digest: dict[str, Any]) -> int:
    """Rough token estimate — canonical chars / 4, the repo-wide rule of thumb."""
    return len(canonicalize(digest)) // 4
