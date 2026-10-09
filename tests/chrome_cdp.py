"""A tiny stdlib-only Chrome DevTools driver for the ``index.html`` browser test (issue #241).

The repo carries no Playwright, so the browser test drives a locally installed
Chrome/Chromium/Edge headless over ``--remote-debugging-pipe`` (JSON messages,
NUL-terminated, on fds 3 and 4 of the browser process). Only what the test
needs: open a page, evaluate JavaScript, collect console errors and uncaught
exceptions, and route downloads to a directory. ``find_browser()`` returns
``None`` when no browser is installed and the test then skips.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import select
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

_MAC_APPS = (
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
)
_NAMES = (
    "google-chrome",
    "google-chrome-stable",
    "chromium",
    "chromium-browser",
    "chrome",
    "microsoft-edge",
)


def find_browser() -> str | None:
    """A Chromium-family browser executable, or ``None``."""
    env = os.environ.get("PLAYBOOK_TEST_BROWSER")
    if env and Path(env).exists():
        return env
    for name in _NAMES:
        found = shutil.which(name)
        if found:
            return found
    for path in _MAC_APPS:
        if Path(path).exists():
            return path
    return None


class BrowserError(RuntimeError):
    pass


class Browser:
    """One headless browser with one page. Use as a context manager."""

    def __init__(self, exe: str, *, download_dir: Path | None = None) -> None:
        if sys.platform == "win32":  # pragma: no cover - the pipe transport is POSIX only
            raise BrowserError("the pipe transport needs POSIX file descriptors")
        self._profile = tempfile.mkdtemp(prefix="opf-index-chrome-")
        # High descriptor numbers, so dup2 onto 3 and 4 in the child never collides.
        to_browser_r, self._to_browser = (
            fcntl.fcntl(fd, fcntl.F_DUPFD_CLOEXEC, 10) for fd in os.pipe()
        )
        self._from_browser, from_browser_w = (
            fcntl.fcntl(fd, fcntl.F_DUPFD_CLOEXEC, 10) for fd in os.pipe()
        )

        def _wire() -> None:  # runs in the child between fork and exec
            os.dup2(to_browser_r, 3)
            os.dup2(from_browser_w, 4)

        self._proc = subprocess.Popen(
            [
                exe,
                "--headless=new",
                "--remote-debugging-pipe",
                "--no-first-run",
                "--no-default-browser-check",
                "--disable-gpu",
                "--disable-extensions",
                "--no-sandbox",
                f"--user-data-dir={self._profile}",
                "about:blank",
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=False,
            preexec_fn=_wire,  # noqa: PLW1509
        )
        os.close(to_browser_r)
        os.close(from_browser_w)
        self._buf = b""
        self._next_id = 0
        self.events: list[dict[str, Any]] = []
        self._session: str | None = None
        self._download_dir = download_dir

    # -- transport ---------------------------------------------------------

    def _send(self, method: str, params: dict[str, Any] | None = None, session: bool = True) -> int:
        self._next_id += 1
        msg: dict[str, Any] = {"id": self._next_id, "method": method, "params": params or {}}
        if session and self._session:
            msg["sessionId"] = self._session
        os.write(self._to_browser, json.dumps(msg).encode() + b"\0")
        return self._next_id

    def _read_message(self, timeout: float) -> dict[str, Any] | None:
        deadline = time.monotonic() + timeout
        while b"\0" not in self._buf:
            left = deadline - time.monotonic()
            if left <= 0:
                return None
            ready, _, _ = select.select([self._from_browser], [], [], left)
            if not ready:
                return None
            chunk = os.read(self._from_browser, 1 << 20)
            if not chunk:
                raise BrowserError("the browser closed the DevTools pipe")
            self._buf += chunk
        raw, self._buf = self._buf.split(b"\0", 1)
        return json.loads(raw)

    def call(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        session: bool = True,
        timeout: float = 30.0,
    ) -> dict[str, Any]:
        want = self._send(method, params, session)
        deadline = time.monotonic() + timeout
        while True:
            msg = self._read_message(max(0.0, deadline - time.monotonic()))
            if msg is None:
                raise BrowserError(f"timed out waiting for {method}")
            if msg.get("id") == want:
                if "error" in msg:
                    raise BrowserError(f"{method}: {msg['error']}")
                return msg.get("result", {})
            if "method" in msg:
                self.events.append(msg)

    def pump(self, seconds: float) -> None:
        """Collect events for *seconds*."""
        deadline = time.monotonic() + seconds
        while True:
            msg = self._read_message(max(0.0, deadline - time.monotonic()))
            if msg is None:
                return
            if "method" in msg:
                self.events.append(msg)

    # -- lifecycle ---------------------------------------------------------

    def __enter__(self) -> Browser:
        target = self.call("Target.createTarget", {"url": "about:blank"}, session=False)
        attached = self.call(
            "Target.attachToTarget",
            {"targetId": target["targetId"], "flatten": True},
            session=False,
        )
        self._session = attached["sessionId"]
        for domain in ("Page", "Runtime", "Log"):
            self.call(f"{domain}.enable")
        if self._download_dir is not None:
            self.call(
                "Browser.setDownloadBehavior",
                {"behavior": "allow", "downloadPath": str(self._download_dir)},
                session=False,
            )
        return self

    def __exit__(self, *exc: object) -> None:
        with contextlib.suppress(Exception):
            self.call("Browser.close", session=False, timeout=5)
        try:
            self._proc.wait(timeout=10)
        except subprocess.TimeoutExpired:  # pragma: no cover
            self._proc.kill()
        for fd in (self._to_browser, self._from_browser):
            with contextlib.suppress(OSError):
                os.close(fd)
        shutil.rmtree(self._profile, ignore_errors=True)

    # -- page --------------------------------------------------------------

    def add_script_on_load(self, source: str) -> None:
        self.call("Page.addScriptToEvaluateOnNewDocument", {"source": source})

    def goto(self, url: str) -> None:
        self.events.clear()
        self.call("Page.navigate", {"url": url})
        self.wait_for_event("Page.loadEventFired", timeout=60)
        self.pump(0.3)

    def wait_for_event(self, method: str, timeout: float = 30.0) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        seen = 0
        while True:
            for ev in self.events[seen:]:
                if ev.get("method") == method:
                    return ev
            seen = len(self.events)
            msg = self._read_message(max(0.0, deadline - time.monotonic()))
            if msg is None:
                raise BrowserError(f"timed out waiting for {method}")
            if "method" in msg:
                self.events.append(msg)

    def eval(self, expression: str, *, timeout: float = 30.0) -> Any:
        """Evaluate *expression* (awaiting a returned promise); return its JSON value."""
        result = self.call(
            "Runtime.evaluate",
            {
                "expression": expression,
                "awaitPromise": True,
                "returnByValue": True,
                "userGesture": True,
            },
            timeout=timeout,
        )
        if "exceptionDetails" in result:
            raise BrowserError(f"page exception: {result['exceptionDetails']}")
        return result["result"].get("value")

    def wait_for(self, expression: str, *, timeout: float = 10.0, interval: float = 0.05) -> Any:
        """Poll *expression* until it is truthy and return its value (``BrowserError`` on timeout)."""
        deadline = time.monotonic() + timeout
        value: Any = None
        while time.monotonic() < deadline:
            value = self.eval(expression)
            if value:
                return value
            self.pump(interval)
        raise BrowserError(f"timed out waiting for {expression!r}; last value {value!r}")

    def problems(self) -> list[str]:
        """Console errors, uncaught exceptions and failed loads seen since the last ``goto``."""
        found: list[str] = []
        for ev in self.events:
            method, params = ev.get("method"), ev.get("params", {})
            if method == "Runtime.exceptionThrown":
                details = params.get("exceptionDetails", {})
                found.append(
                    f"exception: {details.get('text')} {details.get('exception', {}).get('description', '')}"
                )
            elif method == "Runtime.consoleAPICalled" and params.get("type") == "error":
                found.append(
                    "console.error: "
                    + " ".join(
                        str(a.get("value", a.get("description", "")))
                        for a in params.get("args", [])
                    )
                )
            elif method == "Log.entryAdded" and params.get("entry", {}).get("level") == "error":
                entry = params["entry"]
                found.append(f"log error: {entry.get('text')} {entry.get('url', '')}")
        return found
