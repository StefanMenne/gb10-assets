"""Real worker subprocess with a fake run.py: exercises tools/_worker.py and runners.Worker without a GPU."""
import shutil
import sys
from pathlib import Path

import pytest

from gb10_assets.server.config import REPO_ROOT
from gb10_assets.server.runners import JobError, ToolSpec, Worker, WorkerError

FAKE_RUN = '''
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _worker import serve, seeds

def load():
    print("noise on stdout must not break the protocol")
    if Path("fail_load").exists():
        raise RuntimeError("weights missing")
    return "model"

def run(model, params, job_dir):
    if params["prompt"] == "fail":
        raise ValueError("bad prompt")
    names = []
    for s in seeds(params):
        (job_dir / "out" / f"r_{s}.txt").write_text(params["prompt"])
        names.append(f"r_{s}.txt")
    print("also noise")
    return names

serve(load, run)
'''


@pytest.fixture
def tool_settings(settings, tmp_path):
    tools = tmp_path / "tools"
    (tools / "fake").mkdir(parents=True)
    shutil.copy(REPO_ROOT / "tools" / "_worker.py", tools)
    (tools / "fake" / "run.py").write_text(FAKE_RUN)
    venv_bin = tools / "fake" / ".venv" / "bin"
    venv_bin.mkdir(parents=True)
    (venv_bin / "python").symlink_to(sys.executable)
    settings.tools_dir = tools
    return settings


def test_worker_serves_several_jobs(tool_settings, tmp_path):
    w = Worker(ToolSpec("fake", 1), tool_settings)
    try:
        log = tmp_path / "log.txt"
        w.set_log(log)
        w.wait_ready(30)
        job = tmp_path / "job1"
        assert w.run(job, {"prompt": "hi", "seed": 5, "count": 2}) == ["r_5.txt", "r_6.txt"]
        assert (job / "out" / "r_6.txt").read_text() == "hi"
        with pytest.raises(JobError, match="bad prompt"):
            w.run(tmp_path / "job2", {"prompt": "fail"})
        assert w.run(tmp_path / "job3", {"prompt": "again", "seed": 1}) == ["r_1.txt"]  # still alive
        w.set_log(None)
        assert "noise on stdout" in log.read_text() and "ValueError" in log.read_text()
    finally:
        w.stop()
    assert not w.alive()


def test_worker_load_failure(tool_settings):
    (Path(tool_settings.tools_dir) / "fake" / "fail_load").touch()
    w = Worker(ToolSpec("fake", 1), tool_settings)
    with pytest.raises(WorkerError, match="weights missing"):
        w.wait_ready(30)
    w.stop()


def test_missing_environment(settings, tmp_path):
    settings.tools_dir = tmp_path
    with pytest.raises(WorkerError, match="environment for nope missing"):
        Worker(ToolSpec("nope", 1), settings)
