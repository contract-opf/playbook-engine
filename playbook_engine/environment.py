"""What the engine needs from the machine it runs on — declared, and checkable.

The engine's Python dependencies are declared in ``pyproject.toml`` and
installed by pip. Its *external* dependencies are not: ``docling`` and
``pandoc`` are discovered at runtime with :func:`shutil.which` deep inside
``extraction.py``, and until now nothing anywhere stated the set. That is how
an environment drifts without anyone noticing — ``docling`` disappeared from a
host venv between two runs on the same machine, the next derivation silently
fell back to the legacy extractor, and 43 of 44 documents were quarantined with
no announcement at any layer (issue #121).

This module is the declaration. :data:`EXTERNAL_TOOLS` names every external
binary the pipeline can reach, what it is for, and what happens without it;
:func:`probe_environment` answers what is actually present. ``playbook doctor``
renders it, and ``corpus_linter`` continues to make the config-scoped judgment
calls (a missing ``pandoc`` only matters for a corpus containing ``.rtf``).

Nothing here reads a corpus or a config. It describes the machine, so it can be
run before there is anything else to run.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
from dataclasses import dataclass
from pathlib import Path

from playbook_engine import __version__

# Written into the Docker image at build time (see the Dockerfile's
# ENGINE_VERSION/ENGINE_GIT_SHA stamp). Its presence is how a process knows it
# is running inside the project's own image rather than on a host.
IMAGE_STAMP_PATH = Path("/etc/playbook-engine-image.json")


@dataclass(frozen=True)
class ExternalTool:
    """One external binary the pipeline may shell out to.

    Attributes:
        name:        Executable name, as looked up on PATH.
        purpose:     What the engine uses it for, in the user's terms.
        consequence: What happens when it is absent — the important field.
                     Every one of these is a *silent* degradation somewhere in
                     the pipeline, which is the whole reason this table exists.
        install:     Shortest honest way to get it.
        expected_in_image: True when the project's Docker image installs it, so
                     its absence *inside* the image is a broken image rather
                     than a user's missing optional dependency.
    """

    name: str
    purpose: str
    consequence: str
    install: str
    expected_in_image: bool


#: Every external binary the pipeline can reach. Kept in one place so the
#: answer to "what does this need installed?" is a file, not a grep.
EXTERNAL_TOOLS: tuple[ExternalTool, ...] = (
    ExternalTool(
        name="docling",
        purpose=(
            "primary document extractor — layout, table structure, heading "
            "hierarchy, and OCR for scanned PDFs"
        ),
        consequence=(
            "extraction falls back to the legacy adapters (python-docx / "
            "pdfplumber / pandoc). Born-digital DOCX still works; scanned PDFs "
            "yield little or no text, and clause structure is coarser"
        ),
        install="run inside the project image (`make docker-build`), or `pip install docling`",
        expected_in_image=True,
    ),
    ExternalTool(
        name="pandoc",
        purpose="the legacy .rtf extractor's converter",
        consequence=(
            "any .rtf version fails to extract at mine/segment time when the "
            "legacy path is in use (docling absent, or extractor: legacy)"
        ),
        install="`brew install pandoc` or `apt-get install pandoc`",
        expected_in_image=True,
    ),
    ExternalTool(
        name="soffice",
        purpose="converting legacy binary .doc files to .docx before staging",
        consequence=(
            "the engine cannot read .doc at all, so early .doc drafts are "
            "excluded and a negotiation trail can appear to start later than "
            "it did"
        ),
        install="install LibreOffice",
        expected_in_image=False,
    ),
)


@dataclass(frozen=True)
class ToolStatus:
    """Whether one :class:`ExternalTool` was found, and where."""

    tool: ExternalTool
    path: str | None

    @property
    def present(self) -> bool:
        return self.path is not None


@dataclass(frozen=True)
class ImageStamp:
    """The engine version/commit baked into the running container image."""

    engine_version: str
    git_sha: str


@dataclass(frozen=True)
class EnvironmentReport:
    """Everything :func:`probe_environment` could determine about this machine."""

    engine_version: str
    python_version: str
    platform_name: str
    tools: tuple[ToolStatus, ...]
    image: ImageStamp | None
    anthropic_key_set: bool

    @property
    def in_project_image(self) -> bool:
        """True when running inside the project's own Docker image."""
        return self.image is not None

    def missing(self, *, expected_in_image_only: bool = False) -> list[ToolStatus]:
        """Tools that were not found.

        Args:
            expected_in_image_only: restrict to tools the project image is
                supposed to install — the set whose absence means the image
                itself is wrong.
        """
        return [
            s
            for s in self.tools
            if not s.present and (s.tool.expected_in_image or not expected_in_image_only)
        ]

    def image_matches_engine(self) -> bool | None:
        """Whether the image stamp agrees with the engine actually imported.

        ``None`` when not running in the project image (nothing to compare).
        A ``False`` here means the stamp and the installed package disagree —
        e.g. an editable checkout bind-mounted over the installed one.
        """
        if self.image is None:
            return None
        return self.image.engine_version == self.engine_version


def read_image_stamp(path: Path = IMAGE_STAMP_PATH) -> ImageStamp | None:
    """Read the Docker image's build stamp, or ``None`` if there isn't one.

    Absence is the normal case on a host. A malformed stamp is treated as
    absent rather than raising: this is diagnostic code, and it must not be the
    thing that breaks a run.
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(raw, dict):
        return None
    return ImageStamp(
        engine_version=str(raw.get("engine_version", "unknown")),
        git_sha=str(raw.get("git_sha", "unknown")),
    )


def probe_environment(stamp_path: Path = IMAGE_STAMP_PATH) -> EnvironmentReport:
    """Determine what this machine actually provides.

    Args:
        stamp_path: Override for the image stamp location (tests).

    Returns:
        An :class:`EnvironmentReport`. Never raises.
    """
    return EnvironmentReport(
        engine_version=__version__,
        python_version=platform.python_version(),
        platform_name=f"{platform.system()} {platform.machine()}",
        tools=tuple(ToolStatus(tool=t, path=shutil.which(t.name)) for t in EXTERNAL_TOOLS),
        image=read_image_stamp(stamp_path),
        # Presence only — the value is never read here, printed, or logged.
        anthropic_key_set=bool(os.environ.get("ANTHROPIC_API_KEY")),
    )
