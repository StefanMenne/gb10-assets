"""Dashboard data: an event log (jobs, model loading and unloading) and system metrics.

Events are appended to <jobs_dir>/events.jsonl (rotated at EVENTS_FILE_BYTES to events.jsonl.1); the last
EVENTS_KEPT stay in memory and are reloaded after a restart. A sampler thread reads GPU (NVML), memory, CPU and
the worker processes every SAMPLE_S seconds and keeps HISTORY_S of history in memory. Every CACHE_S seconds it
checks how much of each tool's weights is in the page cache (mincore): after a worker exits, its weights often
stay there, so loading the same tool again is fast.

On GB10, CUDA memory does not show in a process's RSS and NVML reports no device memory total; the GPU memory
of a process comes from NVML's process list, the system memory from /proc/meminfo (unified memory).
"""
from __future__ import annotations

import ctypes
import json
import os
import shutil
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

from .config import Settings
from .runners import TOOLS, install_problem

EVENTS_KEPT = 2000
EVENTS_SHOWN = 300
EVENTS_FILE_BYTES = 10 * 1024**2
SAMPLE_S = 2.0
HISTORY_S = 3600
CACHE_S = 30.0
TEXT_CHARS = 160   # job parameters in events: longer strings are cut
LIST_ITEMS = 8     # ... and lists after this many items
GB = 1024**3


# ---------------------------------------------------------------------------------------------- event log

def summarize_params(params: dict) -> dict:
    """Job parameters for the event log: no own code, long texts and lists cut, empty values dropped."""
    def short(v):
        if isinstance(v, str):
            return v if len(v) <= TEXT_CHARS else v[:TEXT_CHARS] + "…"
        if isinstance(v, list):
            items = [short(x) for x in v[:LIST_ITEMS]]
            return items + [f"+{len(v) - LIST_ITEMS} more"] if len(v) > LIST_ITEMS else items
        return v

    out = {}
    for k, v in params.items():
        if v is None or v == []:
            continue
        out[k] = f"<{len(v)} characters of code>" if k == "code" and isinstance(v, str) else short(v)
    return out


class EventLog:
    def __init__(self, path: Path | None):
        self.path = path
        self._events: deque[dict] = deque(maxlen=EVENTS_KEPT)
        self._lock = threading.Lock()
        if path:
            self._load()

    def add(self, kind: str, **fields) -> None:
        event = {"time": datetime.now(timezone.utc).isoformat(timespec="seconds"), "kind": kind,
                 **{k: v for k, v in fields.items() if v is not None}}
        with self._lock:
            self._events.append(event)
            if self.path:
                try:
                    if self.path.exists() and self.path.stat().st_size > EVENTS_FILE_BYTES:
                        self.path.replace(self.path.with_name(self.path.name + ".1"))
                    with open(self.path, "a") as f:
                        f.write(json.dumps(event) + "\n")
                except OSError:
                    pass  # the event log must never fail a job

    def recent(self) -> list[dict]:
        with self._lock:
            return list(self._events)

    def _load(self) -> None:
        for path in (self.path.with_name(self.path.name + ".1"), self.path):
            try:
                lines = path.read_text(errors="replace").splitlines()[-EVENTS_KEPT:]
            except OSError:
                continue
            for line in lines:
                try:
                    self._events.append(json.loads(line))
                except ValueError:
                    pass


def weight_models(install_dir: Path, tool: str) -> list[str]:
    """Entries of installed/<tool>/weights: usually one directory per model."""
    try:
        return sorted(p.name for p in (install_dir / tool / "weights").iterdir() if not p.name.startswith("."))
    except OSError:
        return []


# ---------------------------------------------------------------------------------------------- page cache

class _PageCache:
    """Bytes of a file in the page cache, via mmap + mincore (no read, no change to the cache)."""

    CHUNK = 1 << 30

    def __init__(self):
        self.libc = ctypes.CDLL(None, use_errno=True)
        self.libc.mmap.restype = ctypes.c_void_p
        self.libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                   ctypes.c_long]
        self.libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        self.libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
        self.page = os.sysconf("SC_PAGE_SIZE")
        self.vec = (ctypes.c_ubyte * (self.CHUNK // self.page))()

    def cached(self, path: str, size: int) -> int:
        fd = os.open(path, os.O_RDONLY)
        try:
            addr = self.libc.mmap(None, size, 1, 1, fd, 0)  # PROT_READ, MAP_SHARED
            if addr in (None, ctypes.c_void_p(-1).value):
                return 0
            try:
                pages = 0
                for off in range(0, size, self.CHUNK):
                    length = min(self.CHUNK, size - off)
                    n = -(-length // self.page)
                    if self.libc.mincore(addr + off, length, self.vec) == 0:
                        pages += n - bytes(memoryview(self.vec)[:n]).count(0)
                return min(pages * self.page, size)
            finally:
                self.libc.munmap(addr, size)
        finally:
            os.close(fd)

    def scan(self, path: Path) -> tuple[int, int]:
        """(size, cached bytes) of all files below path."""
        size = cached = 0
        seen = set()
        for root, _, files in os.walk(path):
            for name in files:
                p = os.path.join(root, name)
                try:
                    st = os.stat(p)
                    if not st.st_size or (st.st_dev, st.st_ino) in seen:
                        continue
                    seen.add((st.st_dev, st.st_ino))
                    size += st.st_size
                    cached += self.cached(p, st.st_size)
                except OSError:
                    pass
        return size, cached


# ---------------------------------------------------------------------------------------------- system readers

def read_meminfo() -> dict[str, int]:
    """/proc/meminfo in bytes."""
    out = {}
    with open("/proc/meminfo") as f:
        for line in f:
            key, _, rest = line.partition(":")
            out[key] = int(rest.split()[0]) * 1024
    return out


def read_cpu_times() -> tuple[int, int]:
    """(busy, total) jiffies over all cores."""
    with open("/proc/stat") as f:
        values = [int(v) for v in f.readline().split()[1:]]
    idle = values[3] + values[4]  # idle + iowait
    return sum(values) - idle, sum(values)


def process_tree_memory(session: int) -> tuple[list[int], int]:
    """Pids of a session (a worker is started with start_new_session, ComfyUI etc. are its children) and their
    summed anonymous and shared resident memory in bytes; file pages (mmapped weights) are page cache."""
    pids, rss = [], 0
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/stat") as f:
                fields = f.read().rpartition(")")[2].split()
            if int(fields[3]) != session:
                continue
            with open(f"/proc/{entry}/status") as f:
                rss += sum(int(line.split()[1]) * 1024 for line in f if line.startswith(("RssAnon:", "RssShmem:")))
            pids.append(int(entry))
        except (OSError, ValueError, IndexError):
            pass
    return pids, rss


class _Gpu:
    """NVML; every value is None where the driver does not support it (or there is no NVML at all)."""

    def __init__(self):
        try:
            import pynvml
            pynvml.nvmlInit()
            self.nv, self.handle = pynvml, pynvml.nvmlDeviceGetHandleByIndex(0)
            self.name = self._try(pynvml.nvmlDeviceGetName)
        except Exception:  # noqa: BLE001 - no driver, no library: the dashboard shows no GPU values
            self.nv = self.handle = self.name = None

    def _try(self, fn, *args):
        try:
            return fn(self.handle, *args)
        except Exception:  # noqa: BLE001 - pynvml.NVMLError_NotSupported and friends
            return None

    def sample(self) -> dict:
        if not self.nv:
            return {"util": None, "temp": None, "power": None, "clock": None, "processes": {}}
        nv = self.nv
        util = self._try(nv.nvmlDeviceGetUtilizationRates)
        power = self._try(nv.nvmlDeviceGetPowerUsage)
        procs = self._try(nv.nvmlDeviceGetComputeRunningProcesses) or []
        return {"util": util.gpu if util else None,
                "temp": self._try(nv.nvmlDeviceGetTemperature, nv.NVML_TEMPERATURE_GPU),
                "power": power / 1000 if power is not None else None,
                "clock": self._try(nv.nvmlDeviceGetClockInfo, nv.NVML_CLOCK_SM),
                "processes": {p.pid: p.usedGpuMemory or 0 for p in procs}}


# ---------------------------------------------------------------------------------------------- monitor

class Monitor:
    def __init__(self, settings: Settings, queue):
        self.s = settings
        self.queue = queue
        self.started = time.time()
        self._history: deque[dict] = deque(maxlen=int(HISTORY_S / SAMPLE_S))
        self._now: dict = {}
        self._tools: dict[str, dict] = {}
        self._disks: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._cpu_prev: tuple[int, int] | None = None
        self._gpu: _Gpu | None = None
        self._cache: _PageCache | None = None
        self._last_cache = 0.0

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, name="monitor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=10)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.sample()
            except Exception:  # noqa: BLE001 - a broken reading must not end the sampler
                pass
            self._stop.wait(SAMPLE_S)

    def sample(self) -> None:
        """One reading; the page cache and installation state only every CACHE_S seconds."""
        if self._gpu is None:
            self._gpu = _Gpu()
        if time.monotonic() - self._last_cache > CACHE_S:
            self._last_cache = time.monotonic()
            self._sample_tools()
        gpu = self._gpu.sample()
        mem = read_meminfo()
        busy, total = read_cpu_times()
        prev, self._cpu_prev = self._cpu_prev, (busy, total)
        cpu = 100 * (busy - prev[0]) / (total - prev[1]) if prev and total > prev[1] else None

        worker = self.queue.overview()["worker"]
        pids: list[int] = []
        if worker and worker.get("pid"):
            pids, rss = process_tree_memory(worker["pid"])
            worker["gpu_mem"] = sum(gpu["processes"].get(p, 0) for p in pids)
            worker["mem"] = rss + worker["gpu_mem"]

        used = mem["MemTotal"] - mem["MemAvailable"]
        now = {
            "time": time.time(),
            "gpu": {k: v for k, v in gpu.items() if k != "processes"} | {"name": self._gpu.name},
            "cpu": cpu, "cores": os.cpu_count(),
            "mem": {"total": mem["MemTotal"], "used": used, "free": mem["MemFree"],
                    "cache": max(0, mem["MemAvailable"] - mem["MemFree"]),
                    "worker": worker.get("mem", 0) if worker else 0,
                    "swap_total": mem.get("SwapTotal", 0),
                    "swap_used": mem.get("SwapTotal", 0) - mem.get("SwapFree", 0)},
            # GPU memory of processes outside the worker: in this container only, others show no pid here
            "gpu_other": sum(m for p, m in gpu["processes"].items() if p not in pids),
            "worker": worker,
        }
        with self._lock:
            self._now = now
            self._history.append({"t": round(now["time"], 1), "gpu": gpu["util"], "cpu": cpu, "used": used,
                                  "worker": now["mem"]["worker"], "cache": now["mem"]["cache"],
                                  "tool": worker["tool"] if worker else None})

    def _sample_tools(self) -> None:
        if self._cache is None:
            self._cache = _PageCache()
        tools = {}
        for tool in TOOLS:
            models = []
            for name in weight_models(self.s.install_dir, tool):
                size, cached = self._cache.scan(self.s.install_dir / tool / "weights" / name)
                models.append({"name": name, "size": size, "cached": cached})
            problem = install_problem(self.s, tool)
            tools[tool] = {"installed": problem is None, "problem": problem, "models": models}
        disks = {}
        for name, path in (("installed", self.s.install_dir), ("jobs", self.s.jobs_dir)):
            try:
                u = shutil.disk_usage(path)
                disks[name] = {"total": u.total, "free": u.free}
            except OSError:
                pass
        with self._lock:
            self._tools = tools
            self._disks = disks

    def dashboard(self, since: float = 0.0) -> dict:
        with self._lock:
            now, tools = dict(self._now), dict(self._tools)
            history = [h for h in self._history if h["t"] > since]
            disks = dict(self._disks)
        events = self.queue.events.recent()
        overview = self.queue.overview()
        return {
            "server": {"started": self.started, "details": self.s.dashboard_details},
            "now": now, "disks": disks, "history": history,
            "tools": [self._tool_row(t, tools.get(t, {}), now.get("worker"), overview, events) for t in TOOLS],
            "events": events[-EVENTS_SHOWN:],
        }

    @staticmethod
    def _tool_row(tool: str, info: dict, worker: dict | None, overview: dict, events: list[dict]) -> dict:
        worker = worker if worker and worker["tool"] == tool else None
        running = overview["running"]
        day_ago = datetime.now(timezone.utc).timestamp() - 86400
        row = {"tool": tool, "installed": info.get("installed"), "problem": info.get("problem"),
               "models": info.get("models", []), "worker": worker,
               "queued": overview["queued"].get(tool, 0),
               "running": running if running and running["tool"] == tool else None,
               "jobs_24h": {}, "last_job": None, "last_load": None}
        for ev in events:
            if ev.get("tool") != tool:
                continue
            if ev["kind"] == "job_end":
                row["last_job"] = ev
                if datetime.fromisoformat(ev["time"]).timestamp() > day_ago:
                    row["jobs_24h"][ev["state"]] = row["jobs_24h"].get(ev["state"], 0) + 1
            elif ev["kind"] == "load_end":
                row["last_load"] = ev
        return row
