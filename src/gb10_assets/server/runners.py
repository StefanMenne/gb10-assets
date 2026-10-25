"""Worker processes: one tools/<tool>/run.py per tool, started in that tool's own uv environment.

GPU memory is released completely because the worker process exits; see tools/_worker.py for the protocol.
"""
from __future__ import annotations

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
)}


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
        envs = settings.tool_envs_dir / spec.name if settings.tool_envs_dir else tool_dir
        python = envs / ".venv" / "bin" / "python"
        if not python.exists():
            raise WorkerError(f"environment for {spec.name} missing: {python} (run `uv sync` in {tool_dir})")
        env = dict(os.environ, MODELS_DIR=str(settings.models_dir), PYTHONUNBUFFERED="1")
        env.pop("GB10_ASSETS_TOKEN", None)
        self.proc = subprocess.Popen(
            [str(python), str(tool_dir / "run.py")], cwd=tool_dir, env=env, text=True, bufsize=1,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
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
        self.set_log(None)

    def kill(self) -> None:
        try:
            os.killpg(self.proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
