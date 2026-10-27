"""Real worker subprocess with a fake run.py: exercises tools/_worker.py and runners.Worker without a GPU."""
import fcntl
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from gb10_assets.server.config import REPO_ROOT
from gb10_assets.server.runners import JobError, ToolSpec, Worker, WorkerError, install_problem, tool_hash

FAKE_RUN = '''
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _worker import serve, seeds

def load():
    print("noise on stdout must not break the protocol")
    if Path("fail_load.py").exists():
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
    """A tool "fake" installed like scripts/install.sh does it, with the test's Python as its environment."""
    tools = tmp_path / "tools"
    (tools / "fake").mkdir(parents=True)
    shutil.copy(REPO_ROOT / "tools" / "_worker.py", tools)
    (tools / "fake" / "run.py").write_text(FAKE_RUN)
    (tools / "fake" / "uv.lock").write_text("locked\n")
    venv_bin = settings.install_dir / "fake" / "env" / "bin"
    venv_bin.mkdir(parents=True)
    (venv_bin / "python").symlink_to(sys.executable)
    settings.tools_dir = tools
    (settings.install_dir / "fake" / ".installed").write_text(tool_hash(tools / "fake") + "\n")
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


def test_last_log_line_before_event_reaches_the_job_log(tool_settings, tmp_path):
    w = Worker(ToolSpec("fake", 1), tool_settings)
    try:
        w.wait_ready(30)
        for i in range(20):  # the queue closes the log right after the event; formerly the last line raced it
            log = tmp_path / f"log{i}.txt"
            w.set_log(log)
            w.run(tmp_path / f"job{i}", {"prompt": "hi", "seed": i})
            w.set_log(None)
            text = log.read_text()
            assert "job done in" in text and "sync" not in text
    finally:
        w.stop()


def test_worker_load_failure(tool_settings):
    (Path(tool_settings.tools_dir) / "fake" / "fail_load.py").touch()  # .py: does not change the tool_hash
    w = Worker(ToolSpec("fake", 1), tool_settings)
    with pytest.raises(WorkerError, match="weights missing"):
        w.wait_ready(30)
    w.stop()


def test_not_installed(settings, tmp_path):
    settings.tools_dir = tmp_path
    with pytest.raises(WorkerError, match="nope is not installed on the server"):
        Worker(ToolSpec("nope", 1), settings)


def test_outdated_installation(tool_settings):
    assert install_problem(tool_settings, "fake") is None
    (Path(tool_settings.tools_dir) / "fake" / "run.py").write_text(FAKE_RUN + "\n")  # code: still up to date
    assert install_problem(tool_settings, "fake") is None
    (Path(tool_settings.tools_dir) / "fake" / "uv.lock").write_text("changed\n")
    with pytest.raises(WorkerError, match="reinstall it"):
        Worker(ToolSpec("fake", 1), tool_settings)


def test_no_worker_during_installation(tool_settings):
    with open(tool_settings.install_dir / "fake" / ".lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)  # held by scripts/install.sh
        with pytest.raises(WorkerError, match="being installed"):
            Worker(ToolSpec("fake", 1), tool_settings)


def test_worker_holds_shared_lock(tool_settings):
    w = Worker(ToolSpec("fake", 1), tool_settings)
    lock = tool_settings.install_dir / "fake" / ".lock"
    try:
        w.wait_ready(30)
        assert subprocess.run(["flock", "-n", "-x", str(lock), "true"]).returncode != 0
    finally:
        w.stop()
    assert subprocess.run(["flock", "-n", "-x", str(lock), "true"]).returncode == 0


def test_tool_hash_matches_shell(tmp_path):
    """runners.tool_hash and tool_hash in scripts/lib/config.sh must agree, install.sh writes the latter."""
    for tool in ("flux2-klein", "ltx-video"):
        shell = subprocess.run(["bash", "-c", f"source scripts/lib/config.sh && tool_hash {tool}"],
                               cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()
        assert shell == tool_hash(REPO_ROOT / "tools" / tool)
