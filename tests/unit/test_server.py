import json
from functools import partial

import pytest
from conftest import TOKEN, FakeWorker, wait_for
from fastapi.testclient import TestClient

from gb10_assets.server import runners
from gb10_assets.server.app import create_app
from gb10_assets.server.config import Settings
from gb10_assets.server.queue import JobQueue
from gb10_assets.shared.schemas import API_VERSION, JobState

AUTH = {"Authorization": f"Bearer {TOKEN}"}


def make_queue(settings, mem=200.0, **worker_kw):
    return JobQueue(settings, worker_factory=partial(FakeWorker, **worker_kw), mem_gb=lambda: mem)


@pytest.fixture
def api(settings):
    queue = make_queue(settings)
    with TestClient(create_app(settings, queue)) as client:
        yield client


def submit(api, params, files=(), tool="flux2-klein"):
    body = {"request": (None, json.dumps({"tool": tool, "params": params}))}
    return api.post("/jobs", headers=AUTH, files=list(body.items()) + [("files", f) for f in files])


def test_health_without_auth(api):
    h = api.get("/health").json()
    assert h["api_version"] == API_VERSION and h["status"] == "ok"


@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer wrong"}, {"Authorization": TOKEN}])
def test_auth_required(api, headers):
    assert api.get("/jobs", headers=headers).status_code == 401


@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer anything"}])
def test_auth_disabled(settings, headers):
    settings.token = None
    with TestClient(create_app(settings, make_queue(settings))) as client:
        assert client.get("/jobs", headers=headers).status_code == 200


@pytest.mark.parametrize("value", ["none", "NONE"])
def test_token_none_disables_auth(monkeypatch, value):
    monkeypatch.setenv("GB10_ASSETS_TOKEN", value)
    assert Settings.from_env().token is None


@pytest.mark.parametrize("value", ["", "short"])
def test_token_too_short(monkeypatch, value):
    monkeypatch.setenv("GB10_ASSETS_TOKEN", value)
    with pytest.raises(SystemExit, match="none"):
        Settings.from_env()


def test_job_roundtrip(api):
    r = submit(api, {"prompt": "fox", "reference_images": ["ref.png"]}, [("ref.png", b"img")])
    assert r.status_code == 201, r.text
    job_id = r.json()["id"]
    wait_for(lambda: api.get(f"/jobs/{job_id}", headers=AUTH).json()["state"] == "done")
    info = api.get(f"/jobs/{job_id}", headers=AUTH).json()
    assert info["outputs"] == ["image_1.png"]
    assert api.get(f"/jobs/{job_id}/files/image_1.png", headers=AUTH).content == b"png:fox"
    assert api.get(f"/jobs/{job_id}/files/log.txt", headers=AUTH).status_code == 404  # only listed outputs
    assert api.delete(f"/jobs/{job_id}", headers=AUTH).status_code == 204
    assert api.get(f"/jobs/{job_id}", headers=AUTH).status_code == 404


def test_tools_lists_schemas(api):
    tools = api.get("/tools", headers=AUTH).json()
    assert "prompt" in tools["flux2-klein"]["properties"]


@pytest.mark.parametrize("params,files,tool,needle", [
    ({"prompt": "x"}, [], "nope", "unknown tool"),
    ({"prompt": "x", "width": 1001}, [], "flux2-klein", "multiple of 16"),
    ({"prompt": "x", "reference_images": ["a.png"]}, [], "flux2-klein", "not uploaded"),
    ({"prompt": "x"}, [("../evil.png", b"x")], "flux2-klein", "invalid file name"),
])
def test_rejects_bad_requests(api, params, files, tool, needle):
    r = submit(api, params, files, tool)
    assert r.status_code == 422 and needle in r.text


def test_failed_job_reports_error(api):
    job_id = submit(api, {"prompt": "fail"}).json()["id"]
    wait_for(lambda: api.get(f"/jobs/{job_id}", headers=AUTH).json()["state"] == "failed")
    assert "bad prompt" in api.get(f"/jobs/{job_id}", headers=AUTH).json()["error"]


# ---------------------------------------------------------------- queue behaviour


@pytest.fixture
def two_tools(monkeypatch):
    monkeypatch.setitem(runners.TOOLS, "other", runners.ToolSpec("other", min_free_gb=1))


def enqueue(q, tool="flux2-klein", prompt="p"):
    return q.enqueue(q.create(tool, {"prompt": prompt})).id


def run_queue(q, ids):
    q.start()
    try:
        wait_for(lambda: all(q.get(i).state.finished for i in ids))
    finally:
        q.shutdown()
    return [q.get(i) for i in ids]


def test_sticky_worker_reused_for_same_tool(settings, two_tools):
    q = make_queue(settings)
    ids = [enqueue(q), enqueue(q), enqueue(q, "other"), enqueue(q)]
    assert [i.state for i in run_queue(q, ids)] == [JobState.done] * 4
    assert FakeWorker.started == ["flux2-klein", "other", "flux2-klein"]


def test_idle_worker_kept_for_later_job(settings):
    settings.worker_idle_s = 60
    q = make_queue(settings)
    q.start()
    try:
        first = enqueue(q)
        wait_for(lambda: q.get(first).state.finished)
        assert q.status()["worker"] == "flux2-klein"
        second = enqueue(q)
        wait_for(lambda: q.get(second).state.finished)
    finally:
        q.shutdown()
    assert FakeWorker.started == ["flux2-klein"]


def test_job_error_keeps_worker(settings):
    q = make_queue(settings)
    ids = [enqueue(q), enqueue(q, prompt="fail"), enqueue(q)]
    assert [i.state for i in run_queue(q, ids)] == [JobState.done, JobState.failed, JobState.done]
    assert FakeWorker.started == ["flux2-klein"]


def test_worker_released_when_queue_empty(settings):
    q = make_queue(settings)
    run_queue(q, [enqueue(q)])
    assert q.status()["worker"] is None


def test_not_enough_memory_fails_without_killing(settings):
    q = make_queue(settings, mem=10.0)
    [info] = run_queue(q, [enqueue(q)])
    assert info.state == JobState.failed and "only 10 GB" in info.error
    assert FakeWorker.started == []


def test_load_failure(settings):
    q = make_queue(settings, fail_load=True)
    [info] = run_queue(q, [enqueue(q)])
    assert info.state == JobState.failed and "boom" in info.error


def test_cancel_queued_and_running(settings):
    q = make_queue(settings, job_seconds=30)
    running, queued = enqueue(q), enqueue(q)
    q.start()
    try:
        wait_for(lambda: q.get(running).state == JobState.running)
        assert q.get(queued).queue_position == 0
        assert q.cancel(queued).state == JobState.cancelled
        q.cancel(running)
        wait_for(lambda: q.get(running).state == JobState.cancelled)
    finally:
        q.shutdown()


def test_restore_after_restart(settings):
    q = make_queue(settings)
    done = enqueue(q)
    run_queue(q, [done])
    lost = enqueue(q)  # queued, never run
    q2 = make_queue(settings)
    assert q2.get(done).state == JobState.done and q2.get(done).outputs == ["image_1.png"]
    assert q2.get(lost).state == JobState.failed and "restarted" in q2.get(lost).error
