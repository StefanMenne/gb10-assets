"""Worker processes: one tools/<tool>/run.py per tool, started in that tool's own uv environment.

GPU memory is released completely because the worker process exits; see tools/_worker.py for the protocol.
Tools are installed by scripts/install.sh into <install_dir>/<tool>/ (env/, weights/); a worker only starts
for a complete installation that matches the current tool definition (tool_hash).
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import IO

from .config import Settings


@dataclass(frozen=True)
class ToolSpec:
    name: str
    min_free_gb: int  # MemAvailable needed before the worker starts (unified memory, measured peak + margin)


TOOLS: dict[str, ToolSpec] = {t.name: t for t in (
    ToolSpec("flux2-klein", min_free_gb=30),
    ToolSpec("qwen-image-edit", min_free_gb=70),  # measured peak ~63 GB (bf16 weights + LoRA, 1 MP)
    ToolSpec("qwen-tts", min_free_gb=20),  # VoiceDesign + Base, 2 x 4.5 GB bf16; estimate, not measured yet
    ToolSpec("confucius-tts", min_free_gb=16),  # T2S 2.6 GB + w2v-BERT 2.3 GB (fp32) + S2A + BigVGAN; estimate
    ToolSpec("ltx-video", min_free_gb=85),  # measured peak ~77 GB (all weights resident, 1536x1024, 5 s retake)
    ToolSpec("pixal3d", min_free_gb=24),  # measured peak ~14 GB (int8 DiT, resolution 1536, 4096 textures)
    ToolSpec("blender", min_free_gb=8),  # bpy itself ~1 GB; scripts on big meshes or renders need more
)}


def tool_hash(tool_dir: Path) -> str:
    """Fingerprint of what an installation depends on, see tool_hash in scripts/lib/config.sh (same result):
    sha256 over the `sha256sum` lines of every file in tools/<tool> except the Python code."""
    files = sorted((p for p in tool_dir.iterdir() if p.is_file() and p.suffix != ".py"), key=lambda p: p.name)
    lines = "".join(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}\n" for p in files)
    return hashlib.sha256(lines.encode()).hexdigest()[:16]


def install_problem(settings: Settings, tool: str) -> str | None:
    """Why the tool cannot run on this server, or None if it is installed and up to date."""
    try:
        installed = (settings.install_dir / tool / ".installed").read_text().strip()
    except FileNotFoundError:
        return f"{tool} is not installed on the server (run there: scripts/docker.sh install {tool})"
    if installed != tool_hash(settings.tools_dir / tool):
        return (f"{tool} was installed for another version of the server, reinstall it "
                "(run there: scripts/docker.sh update)")
    return None


class WorkerError(RuntimeError):
    """The worker process is unusable (load failure, crash, kill); the queue discards it."""


class JobError(WorkerError):
    """A single job failed ("error" event); the worker keeps running with its model loaded."""


def mem_available_gb() -> float:
    """MemAvailable from /proc/meminfo. On GB10, nvidia-smi reports no total; unified memory is what counts."""
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024**2
    raise RuntimeError("MemAvailable not found in /proc/meminfo")


class Worker:
    """A running tools/<tool>/run.py process. Not thread-safe: used by the queue thread only (kill() excepted)."""

    def __init__(self, spec: ToolSpec, settings: Settings):
        self.spec = spec
        self.tool = spec.name
        self.idle_since = time.monotonic()
        self._log: IO[str] | None = None
        self._log_lock = threading.Lock()
        tool_dir = settings.tools_dir / spec.name
        installed = settings.install_dir / spec.name
        if not installed.is_dir():
            raise WorkerError(install_problem(settings, spec.name))
        # shared lock for the worker's lifetime: scripts/install.sh and `docker.sh uninstall` wait for it
        self._lock = open(installed / ".lock", "a")
        try:
            try:
                fcntl.flock(self._lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
            except BlockingIOError:
                raise WorkerError(f"{spec.name} is being installed on the server, retry when that is done") from None
            if problem := install_problem(settings, spec.name):
                raise WorkerError(problem)
            env = dict(os.environ, WEIGHTS_DIR=str(installed / "weights"), PYTHONUNBUFFERED="1")
            env.pop("GB10_ASSETS_TOKEN", None)
            self.proc = subprocess.Popen(
                [str(installed / "env" / "bin" / "python"), str(tool_dir / "run.py")], cwd=tool_dir, env=env,
                text=True, bufsize=1, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                start_new_session=True)
        except BaseException:
            self._lock.close()
            raise
        threading.Thread(target=self._pump_stderr, daemon=True).start()

    def set_log(self, path: Path | None) -> None:
        with self._log_lock:
            if self._log:
                self._log.close()
            self._log = open(path, "a", buffering=1) if path else None

    def _pump_stderr(self) -> None:
        assert self.proc.stderr
        for line in self.proc.stderr:
            with self._log_lock:
                if self._log:
                    self._log.write(line)

    def _event(self) -> dict:
        assert self.proc.stdout
        line = self.proc.stdout.readline()
        if not line:
            code = self.proc.wait()
            raise WorkerError(f"{self.tool} worker exited with code {code}")
        return json.loads(line)

    def wait_ready(self, timeout: float) -> None:
        timer = threading.Timer(timeout, self.kill)
        timer.start()
        try:
            ev = self._event()
        except WorkerError as e:
            raise WorkerError(f"{e} while loading (timeout {timeout:.0f}s)") from None
        finally:
            timer.cancel()
        if ev.get("event") != "ready":
            raise WorkerError(ev.get("message", f"unexpected event {ev}"))

    def run(self, job_dir: Path, params: dict) -> list[str]:
        assert self.proc.stdin
        self.proc.stdin.write(json.dumps({"job_dir": str(job_dir), "params": params}) + "\n")
        self.proc.stdin.flush()
        ev = self._event()
        self.idle_since = time.monotonic()
        if ev.get("event") == "done":
            return list(ev["outputs"])
        if ev.get("event") == "error":
            raise JobError(ev.get("message", "job failed"))
        raise WorkerError(ev.get("message", f"unexpected event {ev}"))

    @property
    def pid(self) -> int:
        return self.proc.pid

    def alive(self) -> bool:
        return self.proc.poll() is None

    def stop(self, timeout: float = 30) -> None:
        """Close stdin so the worker exits on its own; kill it if it does not."""
        try:
            if self.proc.stdin:
                self.proc.stdin.close()
            self.proc.wait(timeout)
        except (subprocess.TimeoutExpired, OSError):
            self.kill()
            self.proc.wait()
        self.set_log(None)
        self._lock.close()

    def kill(self) -> None:
        try:
            os.killpg(self.proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
