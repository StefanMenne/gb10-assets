"""Shutting the host down from the dashboard: on request or after a time without jobs.

The server runs in a container without rights on the host: it only creates <jobs_dir>/.shutdown. A systemd path
unit on the host (`scripts/docker.sh install-shutdown`) watches that file, deletes it and runs
`systemctl poweroff`; it also creates <jobs_dir>/.shutdown-helper, without it the server offers no shutdown.
The idle setting is kept in <jobs_dir>/shutdown.json.
"""
from __future__ import annotations

import threading
import time
from typing import Callable

from pydantic import BaseModel, Field

from .config import Settings

CHECK_S = 2.0
REQUEST_FILE = ".shutdown"
HELPER_FILE = ".shutdown-helper"  # created by `scripts/docker.sh install-shutdown`
SETTINGS_FILE = "shutdown.json"


class IdleShutdown(BaseModel):
    enabled: bool = False
    minutes: int = Field(60, ge=1, le=7 * 24 * 60)


class Power:
    def __init__(self, settings: Settings, queue, clock: Callable[[], float] = time.time):
        self.queue = queue
        self.clock = clock
        self.request_file = settings.jobs_dir / REQUEST_FILE
        self.helper_file = settings.jobs_dir / HELPER_FILE
        self.settings_file = settings.jobs_dir / SETTINGS_FILE
        try:
            self.idle = IdleShutdown.model_validate_json(self.settings_file.read_text())
        except (OSError, ValueError):
            self.idle = IdleShutdown()
        # a request the host did not take before the server stopped (helper not installed) is dropped, so that
        # installing the helper later does not shut down at once
        self.request_file.unlink(missing_ok=True)
        self.idle_since = clock()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, name="power", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=10)

    def _loop(self) -> None:
        while not self._stop.wait(CHECK_S):
            try:
                self.check()
            except Exception:  # noqa: BLE001 - a failed check must not end the thread
                pass

    def installed(self) -> bool:
        return self.helper_file.exists()

    def check(self) -> None:
        """Shut down when idle shutdown is on and no job ran or waited for the set time."""
        status = self.queue.status()
        with self._lock:
            if status["running"] or status["queued"]:
                self.idle_since = self.clock()
                return
            due = self.idle.enabled and self.clock() - self.idle_since >= self.idle.minutes * 60
            minutes = self.idle.minutes
        if due and self.installed() and not self.request_file.exists():
            self.request(f"no job for {minutes} min")

    def request(self, reason: str) -> None:
        self.request_file.write_text(reason + "\n")
        self.queue.events.add("shutdown", reason=reason)

    def set_idle(self, idle: IdleShutdown) -> None:
        with self._lock:
            self.idle = idle
            self.idle_since = self.clock()  # the time counts from the change
            self.settings_file.write_text(idle.model_dump_json())

    def state(self) -> dict:
        try:
            requested = self.request_file.stat().st_mtime
        except OSError:
            requested = None
        busy = self.queue.status()
        with self._lock:
            return {"installed": self.installed(), **self.idle.model_dump(), "requested": requested,
                    "idle_since": None if busy["running"] or busy["queued"] else self.idle_since}
