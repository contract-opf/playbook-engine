#!/usr/bin/env python3
"""Regenerate the canary corpus's DOCX files, byte-deterministically.

The canary corpus (``examples/canary/corpus/``) is committed as ``.docx``
binaries because the extraction cache keys on the source file's **sha256**
(``extraction._extraction_cache_payload``) — a corpus generated at test time
would change its key on every run and could never be warm.

Everything here is **wholly synthetic**: fictional parties (Vantage Orbital
Systems / Halcyon Freight / Meridian Assay Group), fictional plain
boilerplate, invented reviewer names. Nothing is copied, adapted, or derived
from any real agreement or any ``*-corpus/`` directory.

Determinism: the ``.docx`` files are written by a hand-rolled minimal OOXML
zip writer with a fixed ZipInfo timestamp and fixed compression, so re-running
this script reproduces byte-identical files (verify with ``git status``).
python-docx is never used to *write* — it is only ever used by the engine to
*read* these files back.

Run from the repo root::

    python examples/canary/build_corpus.py

Then regenerate the segmentation verdicts and expectations::

    python examples/canary/build_verdicts.py

See examples/canary/README.md.
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

_HERE = Path(__file__).resolve().parent
_CORPUS = _HERE / "corpus"

# Fixed (1980-01-01 00:00:00) so the zip's local headers carry no wall clock.
_ZIP_DATE = (1980, 1, 1, 0, 0, 0)

_W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"

_CONTENT_TYPES = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>
"""

_ROOT_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>
"""

_DOC_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"/>
"""


# ---------------------------------------------------------------------------
# Paragraph model
#
# A paragraph is a list of runs. A run is one of:
#   ("t",   text)                  -- plain text
#   ("ins", text, author, date)    -- tracked insertion (w:ins)
#   ("del", text, author, date)    -- tracked deletion  (w:del/w:delText)
#
# The engine's accepted-changes view (docx_ingester._extract_para_text, which
# extraction._extract_docx_lines reuses) includes "t" + "ins" and excludes
# "del" -- so the canonical text of a redline is its post-markup state.
# ---------------------------------------------------------------------------

Run = tuple[str, ...]
Para = list[Run]


def _t(text: str) -> Run:
    return ("t", text)


def _ins(text: str, author: str, date: str) -> Run:
    return ("ins", text, author, date)


def _del(text: str, author: str, date: str) -> Run:
    return ("del", text, author, date)


def _run_xml(text: str) -> str:
    return f'<w:r><w:t xml:space="preserve">{escape(text)}</w:t></w:r>'


def _para_xml(para: Para, ids: list[int]) -> str:
    parts: list[str] = ["<w:p>"]
    for run in para:
        kind = run[0]
        if kind == "t":
            parts.append(_run_xml(run[1]))
        elif kind == "ins":
            _, text, author, date = run
            ids[0] += 1
            parts.append(
                f'<w:ins w:id="{ids[0]}" w:author="{escape(author)}" '
                f'w:date="{escape(date)}">{_run_xml(text)}</w:ins>'
            )
        elif kind == "del":
            _, text, author, date = run
            ids[0] += 1
            parts.append(
                f'<w:del w:id="{ids[0]}" w:author="{escape(author)}" w:date="{escape(date)}">'
                f'<w:r><w:delText xml:space="preserve">{escape(text)}</w:delText></w:r></w:del>'
            )
        else:  # pragma: no cover - developer error
            raise ValueError(f"unknown run kind: {kind!r}")
    parts.append("</w:p>")
    return "".join(parts)


def _document_xml(paras: list[Para]) -> str:
    ids = [1000]
    body = "".join(_para_xml(p, ids) for p in paras)
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<w:document xmlns:w="{_W_NS}"><w:body>{body}</w:body></w:document>'
    )


def write_docx(path: Path, paras: list[Para]) -> None:
    """Write a minimal, byte-deterministic DOCX containing *paras*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    members = [
        ("[Content_Types].xml", _CONTENT_TYPES),
        ("_rels/.rels", _ROOT_RELS),
        ("word/_rels/document.xml.rels", _DOC_RELS),
        ("word/document.xml", _document_xml(paras)),
    ]
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, text in members:
            info = zipfile.ZipInfo(name, date_time=_ZIP_DATE)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            zf.writestr(info, text.encode("utf-8"))


# ---------------------------------------------------------------------------
# The synthetic agreements
# ---------------------------------------------------------------------------

OUR_PARTY = "Vantage Orbital Systems, Inc."
OUR_AUTHOR = "A. Whitfield"
HALCYON = "Halcyon Freight Limited"
HALCYON_AUTHOR = "R. Okonkwo"
MERIDIAN = "Meridian Assay Group LLC"

_D1 = "2026-03-04T10:00:00Z"
_D2 = "2026-03-11T09:30:00Z"


def _plain(text: str) -> Para:
    return [_t(text)]


def _base_body(counterparty: str, survival_years: str, notice_days: str) -> list[Para]:
    """The shared plain-boilerplate skeleton both agreements start from."""
    return [
        _plain("MUTUAL NON-DISCLOSURE AGREEMENT"),
        _plain(
            f"This Mutual Non-Disclosure Agreement is entered into by "
            f"{OUR_PARTY} and {counterparty}. Each party may act as "
            f"Discloser and as Recipient."
        ),
        _plain("1. Purpose"),
        _plain(
            "The parties wish to evaluate a potential commercial relationship "
            "and may exchange confidential information for that sole purpose. "
            "Recipient shall not use Confidential Information for any other "
            "purpose."
        ),
        _plain("2. Definition of Confidential Information"),
        _plain(
            "Confidential Information means non-public information disclosed "
            "by Discloser that is marked confidential or that a reasonable "
            f"person would understand to be confidential. Oral disclosures "
            f"must be confirmed in writing within {notice_days} days."
        ),
        _plain("3. Exclusions from Confidential Information"),
        _plain(
            "Confidential Information does not include information that is "
            "already known to Recipient without a duty of confidence, is "
            "independently developed by Recipient without reference to "
            "Discloser's information, is publicly available through no fault "
            "of Recipient, or is rightfully received from a third party."
        ),
        _plain("4. Standard of Care"),
        _plain(
            "Recipient shall protect Confidential Information using the same "
            "degree of care it uses for its own confidential information, and "
            "in no event less than reasonable care."
        ),
        _plain("5. Permitted Disclosure to Representatives"),
        _plain(
            "Recipient may disclose Confidential Information to its "
            "employees and professional advisors who need to know it for the "
            "Purpose and who are bound by obligations no less protective "
            "than these. Recipient remains responsible for their compliance."
        ),
        _plain("6. Term and Survival"),
        _plain(
            f"This Agreement runs for two years from its effective date. The "
            f"confidentiality obligations survive for {survival_years} years "
            f"after expiry or termination."
        ),
        _plain("7. Return or Destruction"),
        _plain(
            "On written request, Recipient shall return or destroy "
            "Confidential Information, except for one archival copy retained "
            "for compliance purposes."
        ),
        _plain("8. Governing Law"),
        _plain(
            "This Agreement is governed by the laws of the State of Delaware, "
            "without regard to its conflict-of-laws rules."
        ),
    ]


def _execution_block(counterparty: str, their_signatory: str) -> list[Para]:
    """An executed signature page, appended to each negotiation's LAST version.

    Load-bearing, not decoration: ``clause_position_compiler`` only admits
    observations whose ``outcome`` is in ``_OPF_OUTCOMES``
    (``signed``/``proposed_then_reversed``). Without a version
    ``signed_detector`` recognises as executed, every observation is
    ``unsigned``, the compiled playbook carries ZERO evidence clauses, and
    the canary's "clauses" expectation would be a vacuous 0 that no
    derivation regression could ever move.
    """
    return [
        _plain(
            "IN WITNESS WHEREOF, the parties have executed this Agreement as "
            "of the date last written below."
        ),
        _plain(OUR_PARTY),
        _plain(f"By: /s/ {OUR_AUTHOR}"),
        _plain(counterparty),
        _plain(f"By: /s/ {their_signatory}"),
    ]


def halcyon_v1() -> list[Para]:
    """Our paper, first draft out. Clean (no tracked changes), unexecuted."""
    return _base_body(HALCYON, "five", "thirty")


def halcyon_v2() -> list[Para]:
    """Counterparty's REDLINE back — tracked changes by Halcyon's counsel.

    This is the tracked-changes/redline document the canary exists to keep on
    a supported path (issue #84 / commit bdbdc5b): a DOCX carrying real
    ``w:ins``/``w:del`` markup, whose accepted-changes view is what the
    engine must extract, and whose per-change authorship is what
    ``observation_builder.build_round_moves`` attributes.
    """
    paras = _base_body(HALCYON, "five", "thirty")
    # Clause 3 body: Halcyon strikes the independent-development carve-out.
    paras[7] = [
        _t(
            "Confidential Information does not include information that is "
            "already known to Recipient without a duty of confidence, "
        ),
        _del(
            "is independently developed by Recipient without reference to "
            "Discloser's information, ",
            HALCYON_AUTHOR,
            _D1,
        ),
        _t(
            "is publicly available through no fault of Recipient, or is "
            "rightfully received from a third party."
        ),
    ]
    # Clause 6 body: Halcyon shortens survival from five years to two.
    paras[13] = [
        _t("This Agreement runs for two years from its effective date. The "),
        _t("confidentiality obligations survive for "),
        _del("five", HALCYON_AUTHOR, _D1),
        _ins("two", HALCYON_AUTHOR, _D1),
        _t(" years after expiry or termination."),
    ]
    # Clause 8: Halcyon proposes its own governing law.
    #
    # Every change in this file is authored by HALCYON_AUTHOR, deliberately:
    # a round whose side channel carries two or more distinct authors makes
    # ``tracked_changes_overlay.round_level_fallback_attribution`` refuse
    # outright, so the single-author shape is the one that actually exercises
    # the attribution path end to end. The resulting ``moved_by`` is
    # ``"unknown"`` (never ``"counterparty"``) — ``observation_builder.
    # party_side_for_author`` returns "us" or "unknown" and never guesses the
    # other side (OPF §3.5.3). meridian-assay/v2.docx is the mirror case,
    # authored by OUR_AUTHOR, and pins the ``"us"`` half of that contract, so
    # the canary's expected {us: 2, unknown: 3} split bites in both
    # directions.
    paras[17] = [
        _t("This Agreement is governed by the laws of the State of "),
        _del("Delaware", HALCYON_AUTHOR, _D1),
        _ins("New York", HALCYON_AUTHOR, _D1),
        _t(", without regard to its conflict-of-laws rules."),
    ]
    return paras + _execution_block(HALCYON, HALCYON_AUTHOR)


def meridian_v1() -> list[Para]:
    """Counterparty paper, first draft in. Clean, and narrower than ours."""
    paras = _base_body(MERIDIAN, "three", "ten")
    # Their form carries a liability cap — normally absent from a mutual NDA.
    paras.append(_plain("9. Limitation of Liability"))
    paras.append(
        _plain(
            "Each party's aggregate liability under this Agreement is limited "
            "to fifty thousand dollars."
        )
    )
    return paras


def meridian_v2() -> list[Para]:
    """Our markup back on their paper — tracked changes authored by us."""
    paras = meridian_v1()
    # We restore the five-year survival period.
    paras[13] = [
        _t("This Agreement runs for two years from its effective date. The "),
        _t("confidentiality obligations survive for "),
        _del("three", OUR_AUTHOR, _D2),
        _ins("five", OUR_AUTHOR, _D2),
        _t(" years after expiry or termination."),
    ]
    # We carve confidentiality breaches out of the liability cap.
    paras[19] = [
        _t(
            "Each party's aggregate liability under this Agreement is limited "
            "to fifty thousand dollars."
        ),
        _ins(
            " This limitation does not apply to a breach of the "
            "confidentiality obligations in this Agreement.",
            OUR_AUTHOR,
            _D2,
        ),
    ]
    return paras + _execution_block(MERIDIAN, "L. Farrow")


DOCUMENTS: dict[str, list[Para]] = {
    "halcyon-freight/v1.docx": halcyon_v1(),
    "halcyon-freight/v2.docx": halcyon_v2(),
    "meridian-assay/v1.docx": meridian_v1(),
    "meridian-assay/v2.docx": meridian_v2(),
}


def main() -> None:
    for rel, paras in DOCUMENTS.items():
        path = _CORPUS / rel
        write_docx(path, paras)
        print(f"wrote {path.relative_to(_HERE.parent.parent)} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
