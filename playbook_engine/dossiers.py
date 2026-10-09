"""Hard-rule manifest, critic dossiers and provenance index of an OPF playbook.

OPF 0.5 (issue #228, epic #236): three top-level sections the engine derives
from the document, alongside ``digest``. Like the digest, each is a pure
function of the document, the validator recomputes them, and they are covered
by ``identity.content_hash``. None of them holds a judged verdict: the model
still judges, these supply bounded facts.

``manifest``
    ``{hard_rules: [...]}``, one record per ``floor.invariants[]`` entry:
    ``{rule_id, clause_id, taxonomy_id, statement, required_presence,
    condition, permissible_proof, fallback_language}``. A rule comes ONLY from
    the signed Floor and its presence conditions, never from precedent counts.
    ``fallback_language`` is the clause's ``our_standard.text`` (``null`` when
    there is none). ``condition`` is a deterministic predicate spec the
    consumer can evaluate outside any model (:func:`validate_condition`) or the
    string ``"judged"`` for a rule that is not machine-evaluable.

``dossiers``
    ``{clause_id: dossier}``, one bounded dossier per ``evidence.clauses[]``
    entry, with at most :data:`MAX_EXCERPTS` precedent excerpts. NO TEXT IS
    EVER CUT: every excerpt text (and our standard, and each listed Floor
    rule) is the record's own text whole, because a cut-off clause can lose
    the carve-out or cap that makes it mean what it means and so plant a false
    fact. The size bound is kept by dropping WHOLE parts instead: listed
    Floor rules, last first, only as far as the first excerpt needs, then
    excerpts, last selected first, keeping the first (:func:`build_dossier`,
    budget :func:`dossier_budget`). The dossier names the excerpts it dropped
    (``n_omitted``, ``omitted_precedent_ids``) and the provenance index
    indexes them, so the critic can fetch them. The excerpts are chosen by a
    fixed order (:func:`select_excerpts`), so adding precedents can change
    WHICH excerpts appear, never add to a dossier: it carries no count, so a
    precedent that leaves the selection unchanged leaves the dossier
    byte-identical.

``provenance_index``
    Not sent to a model: the compiler, the corpus snapshot hash, the
    documents behind the selected excerpts, kept or dropped (ids, signing
    dates, source file hashes) and which excerpts each dossier selected, each
    marked ``omitted`` when it was dropped for size.

Size is chars/4 of the canonical JSON, the repo-wide rule of thumb
(``digest.digest_token_estimate``, ``playbook scorecard``).
"""

from __future__ import annotations

import re
from typing import Any

from playbook_engine.canonicalize import canonicalize
from playbook_engine.digest import (
    _signed_at_key,
    arrange_variant_slots,
    clause_precedent_groups_with_members,
    playbook_precedent_records,
)
from playbook_engine.opf_accessors import perspective_party, playbook_clauses

#: A dossier's budget is never below this many tokens (chars/4 of its
#: canonical JSON); see :func:`dossier_budget`.
DOSSIER_MIN_BUDGET = 1_000

#: A clause whose our-standard text is long gets this multiple of that text's
#: tokens as its budget, when that exceeds :data:`DOSSIER_MIN_BUDGET`: a long
#: standard is the clause's own size, not padding to be dropped.
DOSSIER_STANDARD_MULTIPLE = 3

#: At most this many precedent excerpts per dossier.
MAX_EXCERPTS = 2

#: At most this many Floor rules are listed in one dossier (``n_floor_rules``
#: always reports the total; fewer are listed when rules are dropped for size,
#: :func:`build_dossier`).
MAX_FLOOR_RULES = 3

#: ``condition`` value of a rule that is not machine-evaluable.
JUDGED = "judged"

#: ``excerpt.outcome`` vocabulary.
OUTCOME_SIGNED = "signed"
OUTCOME_STRUCK = "struck_before_signing"
OUTCOME_ASK_REFUSED = "ask_refused"

#: ``excerpt.kind`` vocabulary: which digest list the excerpt came from.
KIND_SIGNED_VARIANT = "signed_variant"
KIND_CHANGED_OPENING = "changed_opening"
KIND_REFUSED_ASK = "refused_ask"


# ---------------------------------------------------------------------------
# Predicate specs (manifest ``condition``)
# ---------------------------------------------------------------------------

#: ``condition.type`` vocabulary.
CONDITION_REQUIRED_PHRASES = "required_phrases"
CONDITION_NUMERIC_BOUND = "numeric_bound"
CONDITION_CROSS_REFERENCE = "cross_reference"
CONDITION_TYPES = (CONDITION_REQUIRED_PHRASES, CONDITION_NUMERIC_BOUND, CONDITION_CROSS_REFERENCE)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def validate_condition(spec: Any) -> str | None:
    """``None`` when *spec* is a well-formed ``condition``, else why it is not.

    A condition is the string ``"judged"`` or one of three predicate specs
    (objects with no other keys):

    - ``{"type": "required_phrases", "phrases": [str, ...], "match": "all"|"any"}``
      (``match`` optional, default ``all``): the clause text must contain every
      (or any) phrase, compared case-insensitively on normalized whitespace.
    - ``{"type": "numeric_bound", "pattern": "<regex, one capture group>",
      "min"?: number, "max"?: number, "unit"?: str}``: the number the pattern
      captures must be at least ``min`` and at most ``max`` (one of them at
      least).
    - ``{"type": "cross_reference", "clause_id": str}``: the clause text must
      cross-reference that clause.
    """
    if spec == JUDGED:
        return None
    if not isinstance(spec, dict):
        return f'condition must be "{JUDGED}" or a predicate spec object'
    kind = spec.get("type")
    if kind == CONDITION_REQUIRED_PHRASES:
        extra = set(spec) - {"type", "phrases", "match"}
        phrases = spec.get("phrases")
        if extra:
            return f"required_phrases has unknown key(s): {', '.join(sorted(extra))}"
        if (
            not isinstance(phrases, list)
            or not phrases
            or not all(isinstance(p, str) and p.strip() for p in phrases)
        ):
            return "required_phrases needs a non-empty list of non-blank strings in phrases"
        if spec.get("match", "all") not in ("all", "any"):
            return 'required_phrases match must be "all" or "any"'
        return None
    if kind == CONDITION_NUMERIC_BOUND:
        extra = set(spec) - {"type", "pattern", "min", "max", "unit"}
        if extra:
            return f"numeric_bound has unknown key(s): {', '.join(sorted(extra))}"
        pattern = spec.get("pattern")
        if not isinstance(pattern, str):
            return "numeric_bound needs a regex string in pattern"
        try:
            compiled = re.compile(pattern)
        except re.error as exc:
            return f"numeric_bound pattern does not compile: {exc}"
        if compiled.groups != 1:
            return "numeric_bound pattern must have exactly one capture group"
        bounds = [spec[k] for k in ("min", "max") if k in spec]
        if not bounds or not all(_is_number(b) for b in bounds):
            return "numeric_bound needs a numeric min and/or max"
        if "min" in spec and "max" in spec and spec["min"] > spec["max"]:
            return "numeric_bound min is greater than max"
        if "unit" in spec and not isinstance(spec["unit"], str):
            return "numeric_bound unit must be a string"
        return None
    if kind == CONDITION_CROSS_REFERENCE:
        extra = set(spec) - {"type", "clause_id"}
        if extra:
            return f"cross_reference has unknown key(s): {', '.join(sorted(extra))}"
        target = spec.get("clause_id")
        if not isinstance(target, str) or not target.strip():
            return "cross_reference needs a non-blank clause_id"
        return None
    return f"condition type must be one of {', '.join(CONDITION_TYPES)}"


def floor_rule_errors(invariant: dict[str, Any]) -> list[str]:
    """What is malformed in the manifest extension keys of a ``floor.invariants[]`` entry.

    The Floor stays free-form natural language; a signer MAY add
    ``x_required_presence`` (bool), ``x_condition`` (:func:`validate_condition`)
    and ``x_permissible_proof`` (list of strings) so the manifest can state a
    machine-evaluable rule.

    A rule that says a clause must be present (``x_required_presence`` true),
    or carries a predicate over the clause text (an ``x_condition`` other than
    ``"judged"``), MUST name that clause with ``x_taxonomy_id``: a rule with no
    clause has no text to check and no clause whose absence it could reject,
    so the consumer could not enforce it outside both models.
    """
    errors: list[str] = []
    if "x_required_presence" in invariant and not isinstance(
        invariant["x_required_presence"], bool
    ):
        errors.append("x_required_presence must be a boolean")
    if "x_condition" in invariant:
        why = validate_condition(invariant["x_condition"])
        if why:
            errors.append(f"x_condition: {why}")
    proof = invariant.get("x_permissible_proof")
    if "x_permissible_proof" in invariant and (
        not isinstance(proof, list) or not all(isinstance(p, str) and p.strip() for p in proof)
    ):
        errors.append("x_permissible_proof must be a list of non-blank strings")
    tid = invariant.get("x_taxonomy_id")
    if not (isinstance(tid, str) and tid.strip()):
        if invariant.get("x_required_presence") is True:
            errors.append(
                "x_required_presence true needs x_taxonomy_id: a presence rule must name "
                "the clause that has to be present"
            )
        condition = invariant.get("x_condition", JUDGED)
        if condition != JUDGED and not validate_condition(condition):
            errors.append(
                'x_condition other than "judged" needs x_taxonomy_id: a predicate is '
                "checked against the text of the clause it names"
            )
    return errors


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def _floor_invariants(playbook: dict[str, Any]) -> list[dict[str, Any]]:
    floor = playbook.get("floor")
    invariants = floor.get("invariants") if isinstance(floor, dict) else None
    return [
        i
        for i in (invariants if isinstance(invariants, list) else [])
        if isinstance(i, dict)
        and isinstance(i.get("id"), str)
        and isinstance(i.get("statement"), str)
    ]


def _clause_by_taxonomy(playbook: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for clause in playbook_clauses(playbook):
        tid = clause.get("taxonomy_id")
        if isinstance(tid, str):
            out.setdefault(tid, clause)
    return out


def build_manifest(playbook: dict[str, Any]) -> dict[str, Any]:
    """The hard-rule manifest of *playbook*: ``{hard_rules: [...]}``.

    One rule per ``floor.invariants[]`` entry, in Floor order (so an
    invariant added later never reorders the others):

    - ``rule_id``: the invariant's ``id``.
    - ``clause_id`` / ``taxonomy_id``: the evidence clause the invariant's
      ``x_taxonomy_id`` names (``null`` when it names none, or the corpus has
      no evidence for it; ``taxonomy_id`` still echoes the name).
    - ``statement``: the invariant, verbatim.
    - ``required_presence``: ``x_required_presence`` when the signer set it;
      ``false`` otherwise (an invariant that does not say the clause must
      appear never demands its presence).
    - ``condition``: ``x_condition`` when it is a valid predicate spec,
      ``"judged"`` otherwise.
    - ``permissible_proof``: ``x_permissible_proof``, else ``[]``.
    - ``fallback_language``: the clause's ``our_standard.text``, else ``null``.
    """
    by_taxonomy = _clause_by_taxonomy(playbook)
    rules: list[dict[str, Any]] = []
    for inv in _floor_invariants(playbook):
        tid = inv.get("x_taxonomy_id")
        taxonomy_id = tid if isinstance(tid, str) and tid.strip() else None
        clause = by_taxonomy.get(taxonomy_id) if taxonomy_id is not None else None
        our_standard = clause.get("our_standard") if clause is not None else None
        text = our_standard.get("text") if isinstance(our_standard, dict) else None
        valid = not floor_rule_errors(inv)
        presence = inv.get("x_required_presence") if valid else None
        condition = inv.get("x_condition") if valid and "x_condition" in inv else JUDGED
        proof = inv.get("x_permissible_proof") if valid else None
        rules.append(
            {
                "rule_id": inv["id"],
                "clause_id": clause.get("id") if clause is not None else None,
                "taxonomy_id": taxonomy_id,
                "statement": inv["statement"],
                "required_presence": presence if isinstance(presence, bool) else False,
                "condition": condition,
                "permissible_proof": list(proof) if isinstance(proof, list) else [],
                "fallback_language": text if isinstance(text, str) else None,
            }
        )
    return {"hard_rules": rules}


# ---------------------------------------------------------------------------
# Excerpt selection
# ---------------------------------------------------------------------------


def _excerpt_order_key(record: dict[str, Any]) -> tuple[bool, tuple[int, ...], str]:
    """Which record of a group an excerpt shows (OPF-SPEC §3.12.3).

    Latest ``signed_at`` first (unknown last), ties broken on the lowest
    ``precedent_id``. Not the digest's representative (which breaks ties on
    ``document_id``): the reference compiler records no ``signed_at``, so the
    lowest ``precedent_id`` decides.
    """
    k = _signed_at_key(record.get("signed_at"))
    return (k is None, tuple(-x for x in (k or (0, 0, 0))), str(record.get("id")))


def _ask_round(ask: dict[str, Any]) -> int:
    value = ask.get("round")
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def select_excerpts(
    taxonomy_id: Any, precedent: list[dict[str, Any]], *, party: str | None
) -> list[dict[str, Any]]:
    """The at most :data:`MAX_EXCERPTS` precedent excerpts of one clause, in order.

    Each excerpt is ``{precedent_id, kind, opening, signed, outcome}`` over ONE
    precedent record, verbatim and never re-summarised: ``opening`` is the
    record's ``opening_text`` and ``signed`` its ``signed_text`` (``null``
    when struck before signing, outcome ``struck_before_signing``); a record
    without an ``opening_text`` is its signed text alone (``opening`` null). A
    refused-ask excerpt carries the ask as ``opening`` and what the deal
    signed instead as ``signed`` (outcome ``ask_refused``).

    Candidates, in this order (digest 4's group order, ``digest.py``), each
    contributing ONE record of its group, skipping a group or a record
    already chosen:

    1. the first ``signed_variants`` group with ``n_from_standard`` > 0 (a
       concession on record), its record taken among the members whose
       clause opened with our standard (``opened_with`` ``"standard"``), so
       the excerpt is the concession itself;
    2. the first ``changed_openings`` group (language that did not survive as
       proposed);
    3. the remaining ``signed_variants`` groups, then the ``refused_asks``
       groups.

    The collapsed ``equivalent`` entry of the digest stands for its groups in
    group order: rule 1 takes the first of them with ``n_from_standard`` > 0,
    rule 3 the first of them. A group's record is its (eligible) member with
    the latest ``signed_at``, ties broken on the lowest ``precedent_id``
    (:func:`_excerpt_order_key`); a refused-ask group's record is chosen the
    same way among the deals that made the ask, showing that record's
    earliest-round ask in the group.
    """
    groups = clause_precedent_groups_with_members(taxonomy_id, precedent, party=party)
    members_of = {id(entry): members for entry, members in groups["signed_variants"]}
    slots = arrange_variant_slots([entry for entry, _ in groups["signed_variants"]])

    # (group, kind, record, ask): the group is (digest list, id of its entry).
    candidates: list[tuple[tuple[str, int], str, dict[str, Any], Any]] = []
    for slot in slots:
        hit = next((e for e in slot if e.get("n_from_standard", 0) > 0), None)
        if hit is not None:
            conceded = [m for m in members_of[id(hit)] if m.get("opened_with") == "standard"]
            record = min(conceded, key=_excerpt_order_key)
            candidates.append((("signed_variants", id(hit)), KIND_SIGNED_VARIANT, record, None))
            break
    if groups["changed_openings"]:
        entry, members = groups["changed_openings"][0]
        record = min(members, key=_excerpt_order_key)
        candidates.append((("changed_openings", id(entry)), KIND_CHANGED_OPENING, record, None))
    for slot in slots:
        record = min(members_of[id(slot[0])], key=_excerpt_order_key)
        candidates.append((("signed_variants", id(slot[0])), KIND_SIGNED_VARIANT, record, None))
    for entry, pairs in groups["refused_asks"]:
        ask, record = min(pairs, key=lambda ap: (_excerpt_order_key(ap[1]), _ask_round(ap[0])))
        candidates.append((("refused_asks", id(entry)), KIND_REFUSED_ASK, record, ask))

    excerpts: list[dict[str, Any]] = []
    seen_groups: set[tuple[str, int]] = set()
    seen_records: set[str] = set()
    for group, kind, record, ask in candidates:
        if len(excerpts) >= MAX_EXCERPTS:
            break
        pid = str(record.get("id"))
        if group in seen_groups or pid in seen_records:
            continue
        seen_groups.add(group)
        seen_records.add(pid)
        excerpts.append(_excerpt(kind, record, ask))
    return excerpts


def _text_of(ref: Any) -> str | None:
    text = ref.get("text") if isinstance(ref, dict) else None
    return text if isinstance(text, str) else None


def _excerpt(kind: str, record: dict[str, Any], ask: Any) -> dict[str, Any]:
    signed = _text_of(record.get("signed_text"))
    if kind == KIND_REFUSED_ASK:
        opening, outcome = _text_of(ask), OUTCOME_ASK_REFUSED
    else:
        opening = _text_of(record.get("opening_text"))
        outcome = OUTCOME_SIGNED if signed is not None else OUTCOME_STRUCK
    return {
        "precedent_id": str(record.get("id")),
        "kind": kind,
        "opening": opening,
        "signed": signed,
        "outcome": outcome,
    }


# ---------------------------------------------------------------------------
# Dossiers
# ---------------------------------------------------------------------------


def dossier_tokens(dossier: Any) -> int:
    """Size of one dossier in tokens: canonical JSON chars / 4."""
    return len(canonicalize(dossier)) // 4


def dossier_budget(standard_text: Any) -> int:
    """The token budget of one clause's dossier.

    ``max(DOSSIER_MIN_BUDGET, DOSSIER_STANDARD_MULTIPLE * tokens(our standard))``
    where ``tokens`` is the text's characters / 4, or :data:`DOSSIER_MIN_BUDGET`
    alone when the clause has no our-standard text. The budget scales with the
    clause so that a clause whose own standard is long is not forced to drop the
    evidence beside it; a dossier is brought within this budget by dropping
    whole listed Floor rules and WHOLE excerpts (:func:`build_dossier`), never
    by cutting a text.
    """
    if not isinstance(standard_text, str):
        return DOSSIER_MIN_BUDGET
    return max(DOSSIER_MIN_BUDGET, DOSSIER_STANDARD_MULTIPLE * (len(standard_text) // 4))


def clause_standard_text(clause: dict[str, Any]) -> str | None:
    """The our-standard text of an evidence clause (``null`` when it has none)."""
    return _text_of(clause.get("our_standard"))


def _dossier_of(
    clause: dict[str, Any],
    n_rules: int,
    rules: list[dict[str, Any]],
    excerpts: list[dict[str, Any]],
    omitted: list[dict[str, Any]],
) -> dict[str, Any]:
    """The dossier of *clause* listing *rules* and *excerpts*, every text whole.

    *n_rules* is the clause's total of Floor rules (listed or not); *omitted*
    the selected excerpts left out to fit the budget.
    """
    standard_text = clause_standard_text(clause)
    floor_rules = []
    for rule in rules:
        entry: dict[str, Any] = {"rule_id": rule["id"], "statement": rule["statement"]}
        rationale = rule.get("rationale")
        if isinstance(rationale, str) and rationale.strip():
            entry["rationale"] = rationale
        floor_rules.append(entry)
    omitted_ids = sorted(ex["precedent_id"] for ex in omitted)
    return {
        "clause_id": clause.get("id"),
        "taxonomy_id": clause.get("taxonomy_id"),
        "title": clause.get("title"),
        "our_standard": {"text": standard_text} if standard_text is not None else None,
        "n_floor_rules": n_rules,
        "floor_rules": floor_rules,
        "excerpts": list(excerpts),
        "n_omitted": len(omitted_ids),
        "omitted_precedent_ids": omitted_ids,
    }


def build_dossier(
    clause: dict[str, Any],
    precedent: list[dict[str, Any]],
    invariants: list[dict[str, Any]],
    *,
    party: str | None,
) -> dict[str, Any]:
    """The critic dossier of one ``evidence.clauses[]`` entry.

    ``{clause_id, taxonomy_id, title, our_standard: {text}|null, n_floor_rules,
    floor_rules: [{rule_id, statement, rationale?}], excerpts, n_omitted,
    omitted_precedent_ids}``. The rationale is the signed Floor's own words for
    this clause (its ``x_taxonomy_id`` invariants, at most
    :data:`MAX_FLOOR_RULES` listed; ``n_floor_rules`` counts them all); the
    excerpts are :func:`select_excerpts`.

    No text is ever cut. When the dossier exceeds :func:`dossier_budget`, WHOLE
    parts are dropped, in this order:

    1. the listed Floor rules, last first, while the dossier holding only its
       first excerpt (none, when the clause has none) is still over budget:
       a rule gives way only to the parts that are never dropped, our standard,
       the identifiers and that first excerpt (``n_floor_rules`` still counts
       every rule, and the manifest states each one verbatim);
    2. then excerpts, in reverse selection order, keeping the first, until it
       fits: a dossier always keeps at least one complete excerpt (when the
       clause has any).

    So a Floor rule is listed in preference to a second excerpt, and a second
    excerpt is never dropped when it fits beside the rules left listed. Our
    standard and the identifiers are never
    dropped (the budget is at least three times our standard's tokens). The
    only dossier that may exceed its budget is one holding its single kept
    excerpt, every listed Floor rule dropped: keeping that excerpt whole puts
    it over. ``playbook scorecard`` counts every such dossier
    (``over_budget_single_excerpt``). A dossier with no excerpt that is still
    over its budget (its standard and identifiers alone exceed it) is invalid:
    the validator refuses it, so assembly fails rather than emit it. The
    dropped excerpts are named in ``omitted_precedent_ids`` (sorted) and
    counted in ``n_omitted``.
    """
    tid = clause.get("taxonomy_id")
    rules = [i for i in invariants if i.get("x_taxonomy_id") == tid]
    excerpts = select_excerpts(tid, precedent, party=party)
    budget = dossier_budget(clause_standard_text(clause))

    def tokens_with(n_listed: int, keep: int) -> int:
        return dossier_tokens(
            _dossier_of(clause, len(rules), rules[:n_listed], excerpts[:keep], excerpts[keep:])
        )

    n_listed = min(len(rules), MAX_FLOOR_RULES)
    first = min(1, len(excerpts))
    while n_listed > 0 and tokens_with(n_listed, first) > budget:
        n_listed -= 1
    keep = len(excerpts)
    while keep > 1 and tokens_with(n_listed, keep) > budget:
        keep -= 1
    return _dossier_of(clause, len(rules), rules[:n_listed], excerpts[:keep], excerpts[keep:])


def build_dossiers(playbook: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """``{clause id: dossier}`` for every ``evidence.clauses[]`` entry, in clause order."""
    precedent = playbook_precedent_records(playbook)
    party = perspective_party(playbook)
    invariants = _floor_invariants(playbook)
    out: dict[str, dict[str, Any]] = {}
    for clause in playbook_clauses(playbook):
        cid = clause.get("id")
        if isinstance(cid, str):
            out[cid] = build_dossier(clause, precedent, invariants, party=party)
    return out


# ---------------------------------------------------------------------------
# Provenance index
# ---------------------------------------------------------------------------


def build_provenance_index(
    playbook: dict[str, Any], dossiers: dict[str, dict[str, Any]] | None = None
) -> dict[str, Any]:
    """The provenance index: where every dossier excerpt came from. Never sent to a model.

    ``{compiler: {name, version}, corpus_manifest_hash, documents,
    dossiers}``: ``dossiers`` maps a clause id to the ``{precedent_id,
    document_id, kind, omitted}`` of each excerpt it selected, in selection
    order: the kept excerpts (``omitted`` false, in dossier order), then the
    ones dropped whole to fit the budget (``omitted`` true), so every id in a
    dossier's ``omitted_precedent_ids`` resolves here to its deal and kind.
    ``documents`` holds one ``{document_id, signed_at, version_files}`` per
    deal a selected excerpt was taken from, kept or omitted (sorted by id;
    ``version_files`` the ``{version, sha256}`` of its source files, empty
    when the corpus recorded none).
    """
    dossiers = build_dossiers(playbook) if dossiers is None else dossiers
    precedent = playbook_precedent_records(playbook)
    by_id = {str(p.get("id")): p for p in precedent}
    party = perspective_party(playbook)
    corpus = playbook.get("corpus")
    corpus_docs = {
        d.get("document_id"): d
        for d in (corpus.get("documents") if isinstance(corpus, dict) else None) or []
        if isinstance(d, dict)
    }
    snapshot = corpus.get("snapshot") if isinstance(corpus, dict) else None
    compiler = playbook.get("compiler")
    selected: dict[str, list[dict[str, Any]]] = {}
    signed_at_of: dict[str, Any] = {}
    for clause_id, dossier in dossiers.items():
        omitted_ids = set(dossier.get("omitted_precedent_ids") or [])
        # The dropped excerpts are the tail of the selection: recompute it for
        # their kinds, in selection order.
        dropped = (
            [
                ex
                for ex in select_excerpts(dossier.get("taxonomy_id"), precedent, party=party)
                if ex["precedent_id"] in omitted_ids
            ]
            if omitted_ids
            else []
        )
        rows = []
        for ex, omitted in [(ex, False) for ex in dossier.get("excerpts", [])] + [
            (ex, True) for ex in dropped
        ]:
            record = by_id.get(ex["precedent_id"], {})
            doc_id = str(record.get("document_id"))
            signed_at_of.setdefault(doc_id, record.get("signed_at"))
            rows.append(
                {
                    "precedent_id": ex["precedent_id"],
                    "document_id": doc_id,
                    "kind": ex["kind"],
                    "omitted": omitted,
                }
            )
        selected[clause_id] = rows
    documents = []
    for doc_id in sorted(signed_at_of):
        files = corpus_docs.get(doc_id, {}).get("version_files")
        documents.append(
            {
                "document_id": doc_id,
                "signed_at": signed_at_of[doc_id],
                "version_files": [
                    {"version": f.get("version"), "sha256": f.get("sha256")}
                    for f in (files if isinstance(files, list) else [])
                    if isinstance(f, dict)
                ],
            }
        )
    return {
        "compiler": {
            k: compiler.get(k) if isinstance(compiler, dict) else None for k in ("name", "version")
        },
        "corpus_manifest_hash": snapshot.get("manifest_hash")
        if isinstance(snapshot, dict)
        else None,
        "documents": documents,
        "dossiers": selected,
    }


# ---------------------------------------------------------------------------
# Refresh
# ---------------------------------------------------------------------------

#: The top-level sections this module derives.
DERIVED_SECTIONS = ("manifest", "dossiers", "provenance_index")


def build_derived_sections(playbook: dict[str, Any]) -> dict[str, Any]:
    """``{manifest, dossiers, provenance_index}`` for *playbook*."""
    dossiers = build_dossiers(playbook)
    return {
        "manifest": build_manifest(playbook),
        "dossiers": dossiers,
        "provenance_index": build_provenance_index(playbook, dossiers),
    }


def refresh_derived_sections(playbook: dict[str, Any]) -> bool:
    """Recompute the sections *playbook* carries; ``True`` when one changed.

    A writer that changes the Floor after assembly (``playbook floor sign``,
    the Posture interview's Q4 promotion) calls this before it restamps
    ``identity``: the manifest and the dossiers read the Floor, so they would
    otherwise disagree with the document. A document carrying none of the
    sections is left alone.
    """
    if not any(name in playbook for name in DERIVED_SECTIONS):
        return False
    fresh = build_derived_sections(playbook)
    changed = False
    for name in DERIVED_SECTIONS:
        if name in playbook and playbook[name] != fresh[name]:
            playbook[name] = fresh[name]
            changed = True
    return changed
