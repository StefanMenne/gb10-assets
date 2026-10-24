"""Job queue: exactly one GPU job at a time, run by a single background thread.

Sticky worker: after a job the worker process stays alive if the next queued job uses the same tool, or for
``worker_idle_s`` seconds waiting for one; otherwise it exits and frees the GPU memory. Before a worker starts,
the queue waits (up to ``memory_wait_s``) for enough free memory. Processes it does not own, such as an
LLM server, are never touched.
"""
from __future__ import annotations

import json
import shutil
import threading
import time
import uuid
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

from ..shared.schemas import JobInfo, JobState
from .config import Settings
from .runners import TOOLS, JobError, Worker, WorkerError, mem_available_gb

LOG_TAIL = 20


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Job:
    def __init__(self, info: JobInfo, params: dict, path: Path):
        self.info = info
        self.params = params
        self.path = path

    @property
    def log_path(self) -> Path:
        return self.path / "log.txt"

    def save(self) -> None:
        data = {"info": self.info.model_dump(mode="json"), "params": self.params}
        tmp = self.path / "job.json.tmp"
        tmp.write_text(json.dumps(data, indent=1))
        tmp.replace(self.path / "job.json")

    def log(self, msg: str) -> None:
        with open(self.log_path, "a") as f:
            f.write(f"[queue] {msg}\n")

    def tail(self) -> list[str]:
        try:
            return self.log_path.read_text(errors="replace").splitlines()[-LOG_TAIL:]
        except FileNotFoundError:
            return []


class JobQueue:
    def __init__(self, settings: Settings, worker_factory: Callable[..., Worker] = Worker,
                 mem_gb: Callable[[], float] = mem_available_gb):
        self.s = settings
        self._worker_factory = worker_factory
        self._mem_gb = mem_gb
        self._jobs: dict[str, Job] = {}
        self._pending: deque[str] = deque()
        self._running: str | None = None
        self._worker: Worker | None = None
        self._cond = threading.Condition()
        self._stopping = False
        self._thread: threading.Thread | None = None
        self._last_cleanup = 0.0
        self.s.jobs_dir.mkdir(parents=True, exist_ok=True)
        self._restore()

    # ------------------------------------------------------------------ public API (any thread)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, name="gpu-queue", daemon=True)
        self._thread.start()

    def shutdown(self) -> None:
        with self._cond:
            self._stopping = True
            self._cond.notify_all()
            worker = self._worker
        if worker:
            worker.kill()
        if self._thread:
            self._thread.join(timeout=30)

    def create(self, tool: str, params: dict) -> Job:
        """Create the job directory; the caller writes inputs to job.path / "in" and then calls enqueue()."""
        job_id = uuid.uuid4().hex[:12]
        path = self.s.jobs_dir / job_id
        (path / "in").mkdir(parents=True)
        (path / "out").mkdir()
        return Job(JobInfo(id=job_id, tool=tool, state=JobState.queued, created_at=_now()), params, path)

    def enqueue(self, job: Job) -> JobInfo:
        with self._cond:
            job.save()
            self._jobs[job.info.id] = job
            self._pending.append(job.info.id)
            self._cond.notify_all()
            return self._info(job)

    def get(self, job_id: str) -> JobInfo | None:
        with self._cond:
            job = self._jobs.get(job_id)
            return self._info(job) if job else None

    def path(self, job_id: str) -> Path | None:
        with self._cond:
            job = self._jobs.get(job_id)
            return job.path if job else None

    def list(self) -> list[JobInfo]:
        with self._cond:
            return [self._info(j) for j in self._jobs.values()]

    def cancel(self, job_id: str) -> JobInfo | None:
        with self._cond:
            job = self._jobs.get(job_id)
            if not job:
                return None
            if job.info.state == JobState.queued:
                self._pending.remove(job_id)
                self._finish(job, JobState.cancelled, error="cancelled")
            elif job.info.state == JobState.running:
                job.info.error = "cancelled"   # the queue thread sees the dying worker and marks the job
                if self._worker:
                    self._worker.kill()
            return self._info(job)

    def delete(self, job_id: str) -> bool:
        """Cancel if needed and remove the job with all its files."""
        info = self.cancel(job_id)
        if info is None:
            return False
        deadline = time.monotonic() + 30
        while info and not info.state.finished and time.monotonic() < deadline:
            time.sleep(0.1)
            info = self.get(job_id)
        with self._cond:
            job = self._jobs.pop(job_id, None)
        if job:
            shutil.rmtree(job.path, ignore_errors=True)
        return True

    def status(self) -> dict:
        with self._cond:
            return {"running": self._running, "queued": len(self._pending),
                    "worker": self._worker.tool if self._worker else None}

    # ------------------------------------------------------------------ queue thread

    def _info(self, job: Job) -> JobInfo:
        info = job.info.model_copy()
        if info.state == JobState.queued and job.info.id in self._pending:
            info.queue_position = self._pending.index(job.info.id)
        if info.state == JobState.failed:
            info.log_tail = job.tail()
        return info

    def _finish(self, job: Job, state: JobState, outputs: list[str] | None = None, error: str | None = None):
        job.info.state = state
        job.info.finished_at = _now()
        job.info.outputs = outputs or []
        job.info.error = error
        job.save()

    def _loop(self) -> None:
        while True:
            with self._cond:
                while not self._pending and not self._stopping:
                    self._expire_idle_worker()
                    self._cleanup()
                    self._cond.wait(timeout=5)
                if self._stopping:
                    break
                job = self._jobs[self._pending.popleft()]
                job.info.state = JobState.running
                job.info.started_at = _now()
                self._running = job.info.id
                job.save()
            try:
                outputs = self._run(job)
                result = (JobState.done, outputs, None)
            except WorkerError as e:
                cancelled = job.info.error == "cancelled"
                result = (JobState.cancelled, None, "cancelled") if cancelled else (JobState.failed, None, str(e))
                job.log(f"{result[0].value}: {result[2]}")
                if not isinstance(e, JobError):  # after a job error the worker is still up with its model loaded
                    self._stop_worker()
            with self._cond:
                self._finish(job, *result)
                self._running = None
                if self.s.worker_idle_s <= 0 and not self._next_uses(job.info.tool):
                    self._stop_worker()
        self._stop_worker()

    def _run(self, job: Job) -> list[str]:
        worker = self._worker
        if worker and (worker.tool != job.info.tool or not worker.alive()):
            self._stop_worker()
            worker = None
        if worker is None:
            worker = self._start_worker(job)
        worker.set_log(job.log_path)
        try:
            return worker.run(job.path, job.params)
        finally:
            worker.set_log(None)

    def _start_worker(self, job: Job) -> Worker:
        spec = TOOLS[job.info.tool]
        deadline = time.monotonic() + self.s.memory_wait_s
        logged = None
        while (free := self._mem_gb()) < spec.min_free_gb:
            if time.monotonic() > deadline or job.info.error == "cancelled":
                raise WorkerError(
                    f"only {free:.0f} GB memory available, {spec.name} needs ~{spec.min_free_gb} GB. Another "
                    f"process (e.g. an LLM server) holds the memory; free it or retry later.")
            if logged != round(free):
                job.log(f"waiting for memory: {free:.0f} GB available, {spec.min_free_gb} GB needed")
                logged = round(free)
            time.sleep(min(10.0, max(0.05, deadline - time.monotonic())))
        job.log(f"starting {spec.name} worker ({free:.0f} GB available)")
        worker = self._worker_factory(spec, self.s)
        with self._cond:
            self._worker = worker
            if job.info.error == "cancelled":  # cancelled before there was a worker to kill
                raise WorkerError("cancelled")
        worker.set_log(job.log_path)
        t = time.monotonic()
        worker.wait_ready(self.s.worker_ready_s)
        job.log(f"model loaded in {time.monotonic() - t:.0f}s")
        if job.info.error == "cancelled":
            raise WorkerError("cancelled")
        return worker

    def _stop_worker(self) -> None:
        with self._cond:
            worker, self._worker = self._worker, None
        if worker:
            worker.stop()

    def _next_uses(self, tool: str) -> bool:
        return bool(self._pending) and self._jobs[self._pending[0]].info.tool == tool

    def _expire_idle_worker(self) -> None:
        w = self._worker
        if w and (not w.alive() or time.monotonic() - w.idle_since > self.s.worker_idle_s):
            self._worker = None
            threading.Thread(target=w.stop, daemon=True).start()

    def _cleanup(self) -> None:
        if time.monotonic() - self._last_cleanup < 600:
            return
        self._last_cleanup = time.monotonic()
        limit = _now() - timedelta(hours=self.s.job_ttl_h)
        for job_id, job in list(self._jobs.items()):
            if job.info.state.finished and job.info.finished_at and job.info.finished_at < limit:
                del self._jobs[job_id]
                shutil.rmtree(job.path, ignore_errors=True)

    def _restore(self) -> None:
        """Reload jobs from a previous run; jobs that were queued or running are marked failed."""
        for f in sorted(self.s.jobs_dir.glob("*/job.json")):
            try:
                data = json.loads(f.read_text())
                job = Job(JobInfo.model_validate(data["info"]), data["params"], f.parent)
            except (ValueError, KeyError):
                continue
            if not job.info.state.finished:
                self._finish(job, JobState.failed, error="server restarted while the job was queued or running")
            self._jobs[job.info.id] = job
