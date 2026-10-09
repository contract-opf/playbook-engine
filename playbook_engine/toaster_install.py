"""How to install a compiled playbook in the contract-toaster (issue #241).

ONE constant, used by both the skill's closing message (``playbook
install-steps``) and the "Start here" tab of ``index.html``, so the two never
drift. The path is the admin screen of contract-opf/contract-toaster ``main``
(``frontend/src/AdminPlaybooks.tsx``): the **Playbooks** tab of an admin
session, whose "Upload new playbook" and "Version history" > "Upload version"
forms take an OPF document (``.opf.json``) (the version form also requires a typed
"Version identifier") and whose version row carries one
"Approve & activate" control.

What is uploaded is ``playbook.opf.json`` — the canonical playbook — never
``index.html``, which is a viewer.
"""

from __future__ import annotations

#: The playbook file the toaster takes. A name the toaster's upload form accepts (``*.opf.json``).
PLAYBOOK_FILE = "playbook.opf.json"

#: Placeholder in :data:`TOASTER_INSTALL_STEPS` for the file to upload.
FILE_PLACEHOLDER = "{file}"

#: The numbered steps, one string each (no numbering in the text).
TOASTER_INSTALL_STEPS: tuple[str, ...] = (
    "Sign in to the contract-toaster as an administrator and open the Playbooks tab.",
    "New agreement type: choose Upload new playbook and drop {file}. "
    "A new version of a playbook the toaster already has: choose Version history on "
    "that playbook, then Upload version, type a Version identifier (the form requires "
    "one; for example a date or a short label) and drop {file}.",
    "On the uploaded version's row, choose Approve & activate.",
)


def install_steps(file: str = PLAYBOOK_FILE) -> list[str]:
    """The numbered install steps, naming *file* (an absolute path when the caller has one)."""
    return [
        f"{i}. {step.replace(FILE_PLACEHOLDER, file)}"
        for i, step in enumerate(TOASTER_INSTALL_STEPS, start=1)
    ]
