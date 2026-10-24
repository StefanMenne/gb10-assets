import threading
import time

import pytest

from gb10_assets.server.config import Settings
from gb10_assets.server.runners import WorkerError

TOKEN = "test-token-0123456789"


class FakeWorker:
    """In-process stand-in for runners.Worker: writes one output file per job."""

    started: list[str] = []

    def __init__(self, spec, settings, fail_load=False, job_seconds=0.0):
        self.spec, self.tool = spec, spec.name
        self.idle_since = time.monotonic()
        self.fail_load, self.job_seconds = fail_load, job_seconds
        self.killed = threading.Event()
        self.stopped = False
        FakeWorker.started.append(spec.name)

    def set_log(self, path):
        pass

    def wait_ready(self, timeout):
        if self.fail_load:
            raise WorkerError("loading failed: boom")

    def run(self, job_dir, params):
        if self.killed.wait(self.job_seconds):
            raise WorkerError("worker exited with code -9")
        if params.get("prompt") == "fail":
            raise WorkerError("RuntimeError('bad prompt')")
        (job_dir / "out" / "image_1.png").write_bytes(b"png:" + params["prompt"].encode())
        self.idle_since = time.monotonic()
        return ["image_1.png"]

    def alive(self):
        return not (self.stopped or self.killed.is_set())

    def stop(self, timeout=30):
        self.stopped = True

    def kill(self):
        self.killed.set()


@pytest.fixture(autouse=True)
def _reset_fake():
    FakeWorker.started = []


@pytest.fixture
def settings(tmp_path):
    return Settings(token=TOKEN, jobs_dir=tmp_path / "jobs", models_dir=tmp_path / "models",
                    memory_wait_s=0.2, worker_ready_s=10)


def wait_for(cond, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return True
        time.sleep(0.02)
    raise AssertionError("condition not reached")
