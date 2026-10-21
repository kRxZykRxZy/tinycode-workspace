"""Optional Cloudflare quick tunnel for the dashboard.

The tunnel is entirely optional: the builder never requires it and never
blocks on it.  ``cloudflared`` must already be installed; the builder
only starts the process and reads the temporary URL from its output.
"""

from __future__ import annotations

import re
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Any

__all__ = ["TunnelHandle", "CloudflareTunnel", "is_cloudflared_available"]

_URL_PATTERN = re.compile(r"https://[a-z0-9\-]+\.trycloudflare\.com")


def is_cloudflared_available() -> bool:
    """Return whether the ``cloudflared`` binary can be found."""

    try:
        result = subprocess.run(
            ["cloudflared", "--version"],
            capture_output=True,
            timeout=10,
            check=False,
        )
        return result.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


@dataclass
class TunnelHandle:
    """State of a running tunnel."""

    url: str | None = None
    process: Any = None
    started: bool = False
    error: str = ""
    output: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "started": self.started,
            "error": self.error,
            "output": list(self.output[-10:]),
        }


class CloudflareTunnel:
    """Starts a ``cloudflared tunnel --url`` process for the dashboard."""

    def __init__(self, port: int, host: str = "127.0.0.1") -> None:
        self.port = port
        self.host = host
        self.handle = TunnelHandle()
        self._thread: threading.Thread | None = None

    def start(self, timeout: float = 30.0) -> TunnelHandle:
        """Start the tunnel and wait for the temporary URL."""

        if not is_cloudflared_available():
            self.handle.error = "cloudflared is not installed; the tunnel was not started."
            return self.handle

        try:
            process = subprocess.Popen(
                ["cloudflared", "tunnel", "--url", f"http://{self.host}:{self.port}"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except OSError as exc:
            self.handle.error = f"Failed to start cloudflared: {exc}"
            return self.handle

        self.handle.process = process
        self.handle.started = True

        def _reader() -> None:
            try:
                assert process.stdout is not None
                for line in process.stdout:
                    self.handle.output.append(line.rstrip())
                    match = _URL_PATTERN.search(line)
                    if match and not self.handle.url:
                        self.handle.url = match.group(0)
            except Exception as exc:  # pragma: no cover - process died
                self.handle.error = str(exc)

        self._thread = threading.Thread(target=_reader, name="tinycode-tunnel", daemon=True)
        self._thread.start()

        deadline = time.time() + timeout
        while time.time() < deadline and not self.handle.url:
            time.sleep(0.25)
        if not self.handle.url and not self.handle.error:
            self.handle.error = "Timed out waiting for the Cloudflare quick tunnel URL."
        return self.handle

    def stop(self) -> None:
        """Terminate the tunnel process."""

        process = self.handle.process
        if process is not None and process.poll() is None:
            try:
                process.terminate()
                process.wait(timeout=10)
            except Exception:  # pragma: no cover
                process.kill()
        self.handle.started = False
