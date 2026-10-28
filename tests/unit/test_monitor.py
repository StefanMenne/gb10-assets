import json

import pytest
from conftest import FakeWorker, wait_for
from fastapi.testclient import TestClient
from test_server import AUTH, make_queue, submit

from gb10_assets.server.app import create_app
from gb10_assets.server.monitor import EVENTS_KEPT, EventLog, Monitor, _PageCache, summarize_params


@pytest.fixture
def api(settings):
    settings.dashboard_details = True
    with TestClient(create_app(settings, make_queue(settings))) as client:
        yield client


def kinds(events, tool="flux2-klein"):
    return [e["kind"] for e in events if e.get("tool") == tool]


def test_summarize_params_cuts_texts_and_hides_code():
    s = summarize_params({"prompt": "x" * 500, "texts": [f"t{i}" for i in range(20)], "code": "print(1)\n" * 99,
                          "seed": None, "images": [], "width": 512})
    assert s["prompt"] == "x" * 160 + "…"
    assert s["texts"][:2] == ["t0", "t1"] and s["texts"][-1] == "+12 more"
    assert s["code"] == "<891 characters of code>"
    assert "seed" not in s and "images" not in s and s["width"] == 512


def test_job_events(api, settings):
    job_id = submit(api, {"prompt": "fox " * 100}).json()["id"]
    wait_for(lambda: "unload" in kinds(api.app.state.queue.events.recent()))
    events = [e for e in api.app.state.queue.events.recent() if e.get("tool") == "flux2-klein"]
    assert kinds(events) == ["job_start", "load_start", "load_end", "job_end", "unload"]
    start, end = events[0], events[3]
    assert start["job"] == job_id and start["params"]["prompt"].endswith("…")
    assert end["state"] == "done" and end["outputs"] == ["image_1.png"] and end["seconds"] >= 0
    on_disk = [json.loads(line) for line in (settings.jobs_dir / "events.jsonl").read_text().splitlines()]
    assert [e["kind"] for e in on_disk] == kinds(events)


def test_params_hidden_without_details(settings):
    settings.dashboard_details = False
    with TestClient(create_app(settings, make_queue(settings))) as client:
        submit(client, {"prompt": "secret"})
        wait_for(lambda: "job_end" in kinds(client.app.state.queue.events.recent()))
        start = next(e for e in client.app.state.queue.events.recent() if e["kind"] == "job_start")
        assert "params" not in start
        assert client.get("/dashboard/data").json()["server"]["details"] is False


def test_failed_load_is_logged(settings):
    with TestClient(create_app(settings, make_queue(settings, fail_load=True))) as client:
        submit(client, {"prompt": "fox"})
        wait_for(lambda: "job_end" in kinds(client.app.state.queue.events.recent()))
        events = client.app.state.queue.events.recent()
        assert next(e for e in events if e["kind"] == "load_end")["error"] == "loading failed: boom"
        assert next(e for e in events if e["kind"] == "unload")["reason"] == "worker failed"


def test_event_log_survives_restart(tmp_path):
    log = EventLog(tmp_path / "events.jsonl")
    for i in range(EVENTS_KEPT + 5):
        log.add("job_end", tool="x", n=i)
    reloaded = EventLog(tmp_path / "events.jsonl").recent()
    assert len(reloaded) == EVENTS_KEPT and reloaded[-1]["n"] == EVENTS_KEPT + 4


def test_dashboard_without_auth(api):
    assert "gb10-assets" in api.get("/dashboard").text
    assert api.get("/jobs").status_code == 401  # the rest still needs the token
    submit(api, {"prompt": "fox"})
    wait_for(lambda: "job_end" in kinds(api.app.state.queue.events.recent()))
    data = api.get("/dashboard/data").json()
    row = next(t for t in data["tools"] if t["tool"] == "flux2-klein")
    assert row["jobs_24h"] == {"done": 1} and row["last_job"]["state"] == "done"
    assert row["last_load"]["kind"] == "load_end"
    assert data["now"]["mem"]["total"] > 0 and data["history"]
    assert api.get(f"/dashboard/data?since={data['history'][-1]['t']}").json()["history"] == []


def test_dashboard_shows_loaded_worker(settings):
    queue = make_queue(settings, job_seconds=2)
    with TestClient(create_app(settings, queue)) as client:
        job_id = submit(client, {"prompt": "fox"}).json()["id"]
        # a job is running before its worker exists and has loaded: wait for the loaded worker
        wait_for(lambda: (queue.overview()["worker"] or {}).get("loaded") is not None)
        Monitor(settings, queue).sample()
        overview = queue.overview()
        assert overview["running"]["id"] == job_id and overview["worker"]["tool"] == "flux2-klein"
        assert overview["worker"]["loaded"] is not None
        client.delete(f"/jobs/{job_id}", headers=AUTH)
    assert FakeWorker.started == ["flux2-klein"]


def test_page_cache_scan(tmp_path):
    f = tmp_path / "w" / "model.safetensors"
    f.parent.mkdir()
    f.write_bytes(b"x" * 300_000)
    f.read_bytes()
    size, cached = _PageCache().scan(tmp_path / "w")
    assert size == 300_000 and 0 <= cached <= size
