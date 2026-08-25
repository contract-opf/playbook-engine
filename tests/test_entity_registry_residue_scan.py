"""Regression tests for the post-hoc residue scan — issue #136 review finding.

The autonomous build loop's first attempt at #136 added a "mandatory residue
check" to the ``playbook-from-corpus`` skill that grepped canonical output
for the LITERAL VALUES stored in ``alias_map.json``. An independent review
pass (see issue #136 comments) proved that check does not detect the exact
leak class the ticket is about: ``EntityRegistry.alias_for`` is called
unconditionally for every configured ``known_entities`` name regardless of
whether it actually matched anything, so a misconfigured (e.g.
stopword-stripped) entry's ``alias_map`` value is the WRONG spelling, not the
real text that leaked — grepping output for the wrong spelling naturally
finds zero hits even when the real, differently-spelled name is present
throughout.

These tests reproduce the reviewer's own repro case against
:func:`entity_registry.find_residue` — the token-level replacement for that
naive literal-substring check — and confirm it actually catches the leak.

SECURITY NOTE: All fixtures are synthetic, fictional names constructed for
this test. "Example Institute of Fictional City" is not a real institution.
"""

from __future__ import annotations

from playbook_engine.entity_registry import find_residue, residue_tokens


def test_stopword_stripped_registration_still_flags_the_real_spelling() -> None:
    """The reviewer's exact repro: registering a stopword-stripped name
    ("Example Institute Fictional City") must still flag the corpus's real,
    differently spelled text ("Example Institute of Fictional City") — the
    leak class issue #136 is about. A literal full-phrase check of the
    alias_map value against this text finds nothing (the wrong spelling never
    appears); find_residue must still catch it via the shared distinctive
    token."""
    alias_map = {"Counterparty-1": "Example Institute Fictional City"}
    canonical_output = (
        "This Affiliation Agreement is between Alpha Corp and "
        "Example Institute of Fictional City, a nonprofit educational "
        "institution located in Fictional City."
    )
    # Prove the naive literal-substring check the loop's first attempt used
    # really does miss this (documents the bug this test guards against).
    assert alias_map["Counterparty-1"] not in canonical_output

    hits = find_residue(alias_map, {"playbook.opf.json": canonical_output})
    assert hits, "find_residue failed to catch a stopword-stripped registration's real spelling"
    assert any(h[1] == "Example Institute Fictional City" for h in hits)


def test_verbatim_registration_with_no_leak_is_clean() -> None:
    """A correctly registered name that is fully substituted everywhere (no
    leak) must not be flagged — the healthy, nothing-to-see-here case."""
    alias_map = {"Counterparty-1": "Example Institute of Fictional City"}
    pseudonymized_output = "This Affiliation Agreement is between Alpha Corp and Counterparty-1."
    hits = find_residue(alias_map, {"playbook.opf.json": pseudonymized_output})
    assert hits == []


def test_unregistered_alias_value_is_never_flagged() -> None:
    """An empty/falsy alias_map value must be skipped, not crash or false-flag."""
    alias_map = {"Counterparty-1": ""}
    hits = find_residue(alias_map, {"playbook.opf.json": "Counterparty-1 anonymous filler text"})
    assert hits == []


def test_generic_institutional_words_alone_do_not_flag() -> None:
    """A name composed only of generic/short words (e.g. "State University")
    yields no distinctive tokens and cannot be checked — documented
    limitation, not a false positive on every mention of "State" or
    "University" elsewhere in the document."""
    alias_map = {"Counterparty-1": "State University"}
    text = "This State University policy governs the state university system broadly."
    assert residue_tokens("State University") == []
    hits = find_residue(alias_map, {"playbook.opf.json": text})
    assert hits == []


def test_hit_names_the_offending_label_and_token() -> None:
    """A hit's (label, real_name, token) triple must point at the actual
    scanned text and the actual distinctive token that matched, so a reader
    of the check's output can go verify it in context."""
    alias_map = {"Counterparty-1": "Example Institute Fictional City"}
    hits = find_residue(
        alias_map,
        {"playbook.opf.json": "Example Institute of Fictional City signed the agreement."},
    )
    assert hits
    label, real_name, token = hits[0]
    assert label == "playbook.opf.json"
    assert real_name == "Example Institute Fictional City"
    assert token in {"Example", "Fictional", "City"}


def test_residue_tokens_filters_short_and_stopword_terms() -> None:
    """residue_tokens drops short (<4 char) words and common institutional/
    legal stopwords, keeping only the distinctive proper-noun-shaped tokens."""
    assert residue_tokens("University of Meridian") == ["Meridian"]
    assert residue_tokens("Bob Inc") == []  # "Bob" < 4 chars, "Inc" is a stopword
    assert residue_tokens("Winston-Salem Regional Group") == ["Winston", "Salem", "Regional"]
