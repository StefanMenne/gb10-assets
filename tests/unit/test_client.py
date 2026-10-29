import asyncio
import json
from contextlib import contextmanager
from functools import partial
from types import SimpleNamespace

import httpx
import pytest
from conftest import TOKEN, FakeWorker
from fastapi.testclient import TestClient

from gb10_assets.client import http_client, mcp_server
from gb10_assets.client.http_client import ApiError, Client, VersionMismatch, free_path, upload_name
from gb10_assets.server.app import create_app
from gb10_assets.server.queue import JobQueue


@pytest.fixture
def client(settings):
    queue = JobQueue(settings, worker_factory=FakeWorker, mem_gb=lambda: 200.0)
    with TestClient(create_app(settings, queue), headers={"Authorization": f"Bearer {TOKEN}"}) as tc:
        c = Client("http://testserver", TOKEN)
        c.http = tc
        yield c


def test_submit_uploads_and_fetch_writes_locally(client, tmp_path):
    ref = tmp_path / "my ref (1).png"
    ref.write_bytes(b"img")
    info = client.submit("flux2-klein", {"prompt": "fox", "reference_images": [str(ref)]})
    out = tmp_path / "out"
    (out).mkdir()
    (out / "image_1.png").write_bytes(b"old")
    info, paths = client.fetch(info.id, out, wait=5)
    assert [p.name for p in paths] == ["image_1_1.png"]          # existing file kept
    assert (out / "image_1.png").read_bytes() == b"old"
    assert paths[0].read_bytes() == b"png:fox"
    with pytest.raises(ApiError, match="404"):                   # deleted on the server after fetch
        client.status(info.id)


def test_interrupted_download_leaves_no_part_file(client, monkeypatch, tmp_path):
    info = client.wait(client.submit("flux2-klein", {"prompt": "fox"}).id, 5)

    def chunks():
        yield b"partial"
        raise httpx.ReadError("connection lost")

    @contextmanager
    def stream(method, url):
        yield SimpleNamespace(status_code=200, iter_bytes=chunks)

    monkeypatch.setattr(client.http, "stream", stream)
    out = tmp_path / "out"
    with pytest.raises(httpx.ReadError):
        client.fetch(info.id, out)
    assert list(out.iterdir()) == []


def test_submit_validates_locally(client, tmp_path):
    with pytest.raises(ApiError, match="not found"):
        client.submit("flux2-klein", {"prompt": "x", "reference_images": [str(tmp_path / "missing.png")]})
    with pytest.raises(ApiError, match="invalid parameters"):
        client.submit("flux2-klein", {"prompt": "x", "width": 1001})


def test_failed_job_raises_with_log(client):
    info = client.submit("flux2-klein", {"prompt": "fail"})
    with pytest.raises(ApiError, match="bad prompt"):
        client.fetch(info.id, ".", wait=5)


def test_version_mismatch_and_unreachable():
    def handler(request):
        return httpx.Response(200, json={"status": "ok", "api_version": 999, "version": "9.9.9"})

    c = Client("http://x", TOKEN, transport=httpx.MockTransport(handler))
    with pytest.raises(VersionMismatch, match="v999"):
        c.check_version()

    def down(request):
        raise httpx.ConnectError("refused")

    with pytest.raises(ApiError, match="not reachable"):
        Client("http://x", TOKEN, transport=httpx.MockTransport(down)).health()


def test_get_retried_after_dropped_connection():
    calls = []

    def flaky(request):
        calls.append(request.method)
        if len(calls) < 3:
            raise httpx.RemoteProtocolError("Server disconnected without sending a response.")
        return httpx.Response(200, json={"status": "ok", "api_version": 1, "version": "0.1.0"})

    assert Client("http://x", TOKEN, transport=httpx.MockTransport(flaky)).health().status == "ok"
    assert calls == ["GET"] * 3

    def post_fails(request):
        calls.append(request.method)
        raise httpx.RemoteProtocolError("dropped")

    calls.clear()
    with pytest.raises(ApiError, match="not reachable"):
        Client("http://x", TOKEN, transport=httpx.MockTransport(post_fails))._call("POST", "/jobs")
    assert calls == ["POST"]  # never repeated: it could submit the job twice


def test_missing_config(monkeypatch):
    monkeypatch.delenv("GB10_ASSETS_URL", raising=False)
    with pytest.raises(ApiError, match="GB10_ASSETS_URL"):
        Client()


@pytest.mark.parametrize("token", [None, "none"])
def test_no_token_sends_no_auth_header(monkeypatch, token):
    monkeypatch.delenv("GB10_ASSETS_TOKEN", raising=False)
    seen = []

    def handler(request):
        seen.append(request.headers.get("authorization"))
        return httpx.Response(200, json={"status": "ok", "api_version": 1, "version": "0.1.0"})

    Client("http://x", token, transport=httpx.MockTransport(handler)).health()
    assert seen == [None]


def test_upload_names_unique_and_safe(tmp_path):
    taken = set()
    assert upload_name(tmp_path / "a b.png", taken) == "a_b.png"
    assert upload_name(tmp_path / "a b.png", taken) == "a_b_1.png"
    assert upload_name(tmp_path / "..png", taken) == "input.png"
    (tmp_path / "x.png").write_bytes(b"")
    assert free_path(tmp_path / "x.png").name == "x_1.png"


# ---------------------------------------------------------------- MCP adapter


def call(server, name, args):
    return asyncio.run(server.call_tool(name, args))


def test_mcp_tools_end_to_end(client, tmp_path):
    adapter = mcp_server.Adapter(client)
    server = mcp_server.build(adapter)
    names = {t.name for t in asyncio.run(server.list_tools())}
    assert {"generate_image", "job_status", "fetch_result", "cancel_job"} <= names
    sub = json.loads(call(server, "generate_image", {"prompt": "fox", "count": 1}).content[0].text)
    assert sub["state"] == "queued"
    res = call(server, "fetch_result", {"job_id": sub["job_id"], "target_dir": str(tmp_path), "wait": 5})
    assert json.loads(res.content[0].text)["files"] == [str((tmp_path / "image_1.png").resolve())]
    assert (tmp_path / "image_1.png").read_bytes() == b"png:fox"


def test_mcp_edit_image_uploads_inputs(client, tmp_path):
    src = tmp_path / "in put.png"
    src.write_bytes(b"img")
    server = mcp_server.build(mcp_server.Adapter(client))
    sub = json.loads(call(server, "edit_image", {"prompt": "night", "images": [str(src)]}).content[0].text)
    info = client.wait(sub["job_id"], 5)
    assert info.state.value == "done" and info.tool == "qwen-image-edit"
    with pytest.raises(ApiError, match="input file not found"):
        client.submit("qwen-image-edit", {"prompt": "x", "images": [str(tmp_path / "missing.png")]})


def test_mcp_speech_tools(client, tmp_path):
    ref = tmp_path / "stimme 1.wav"
    ref.write_bytes(b"wav")
    server = mcp_server.build(mcp_server.Adapter(client))
    calls = [("design_voice", {"voice_description": "warm", "text": "Hallo", "language": "de"}, "qwen-tts"),
             ("speak", {"texts": ["a", "b"], "reference_audio": str(ref), "reference_text": "Hallo"}, "qwen-tts"),
             ("speak_multilingual", {"texts": ["x"], "language": "th", "reference_audio": str(ref)}, "confucius-tts")]
    for name, args, tool in calls:
        sub = json.loads(call(server, name, args).content[0].text)
        info = client.wait(sub["job_id"], 5)
        assert info.state.value == "done" and info.tool == tool
    with pytest.raises(Exception) as exc:  # Thai is not a Qwen3-TTS language
        call(server, "speak", {"texts": ["x"], "reference_audio": str(ref), "language": "th"})
    assert "invalid parameters" in str(exc.value.__cause__)


def test_mcp_video_tools(client, tmp_path):
    for name in ("first.png", "song.wav", "clip.mp4"):
        (tmp_path / name).write_bytes(b"x")
    server = mcp_server.build(mcp_server.Adapter(client))
    calls = [("generate_video", {"prompt": "fox", "images": [str(tmp_path / "first.png")], "duration_s": 4}),
             ("audio_to_video", {"prompt": "singer", "audio": str(tmp_path / "song.wav")}),
             ("retake_video", {"video": str(tmp_path / "clip.mp4"), "prompt": "a cat", "start_s": 1, "end_s": 2})]
    for name, args in calls:
        sub = json.loads(call(server, name, args).content[0].text)
        info = client.wait(sub["job_id"], 5)
        assert info.state.value == "done" and info.tool == "ltx-video"
    with pytest.raises(Exception) as exc:
        call(server, "retake_video", {"video": str(tmp_path / "clip.mp4"), "prompt": "x", "start_s": 2, "end_s": 1})
    assert "start_s < end_s" in str(exc.value.__cause__)


def test_mcp_image_to_3d(client, tmp_path):
    (tmp_path / "chair.png").write_bytes(b"x")
    server = mcp_server.build(mcp_server.Adapter(client))
    sub = json.loads(call(server, "image_to_3d", {"image": str(tmp_path / "chair.png"), "resolution": 1024}).content[0].text)
    info = client.wait(sub["job_id"], 5)
    assert info.state.value == "done" and info.tool == "pixal3d"


def test_mcp_blender_maps_input_paths_in_args(client, tmp_path):
    (tmp_path / "my fox.glb").write_bytes(b"x")
    local = str(tmp_path / "my fox.glb")
    server = mcp_server.build(mcp_server.Adapter(client))
    sub = json.loads(call(server, "blender", {"files": [local], "script": "rig_hop",
                                              "args": [local, "fox_rig.glb", "--height", "1"]}).content[0].text)
    res = json.loads(call(server, "fetch_result", {"job_id": sub["job_id"], "target_dir": str(tmp_path / "out"),
                                                   "wait": 5}).content[0].text)
    assert FakeWorker.last_params["args"] == ["my_fox.glb", "fox_rig.glb", "--height", "1"]
    assert FakeWorker.last_params["files"] == ["my_fox.glb"]
    assert res["state"] == "done" and len(res["files"]) == 1
    assert any("starting blender worker" in line for line in res["log"])
    with pytest.raises(ApiError, match="404"):                   # deleted after the log was read
        client.status(sub["job_id"])


def test_mcp_refuses_to_start_on_mismatch(monkeypatch, capsys):
    def handler(request):
        return httpx.Response(200, json={"status": "ok", "api_version": 999, "version": "9.9.9"})

    fake = Client("http://x", TOKEN, transport=httpx.MockTransport(handler))
    monkeypatch.setattr(mcp_server, "Adapter", partial(mcp_server.Adapter, fake))
    with pytest.raises(SystemExit) as e:
        mcp_server.run()
    assert e.value.code == 1 and "version mismatch" in capsys.readouterr().err


def test_mcp_starts_when_server_unreachable(monkeypatch, capsys):
    def down(request):
        raise httpx.ConnectError("refused")

    adapter = mcp_server.Adapter(Client("http://x", TOKEN, transport=httpx.MockTransport(down)))
    adapter.startup_check()  # warning only
    assert "warning" in capsys.readouterr().err


def test_mcp_startup_check_is_one_quick_attempt(capsys):
    calls = []

    def down(request):
        calls.append(request.extensions["timeout"]["connect"])
        raise httpx.ConnectTimeout("timed out")

    adapter = mcp_server.Adapter(Client("http://x", TOKEN, transport=httpx.MockTransport(down)))
    adapter.startup_check()
    assert calls == [http_client.QUICK_TIMEOUT_S]                # no retries, short timeout
    assert "warning" in capsys.readouterr().err


def test_wait_polls_through_an_outage(monkeypatch):
    monkeypatch.setattr(http_client, "RETRY_DELAY_S", 0)
    job = {"id": "abc", "tool": "flux2-klein", "state": "running", "created_at": "2026-10-29T09:00:00Z"}
    answers = iter(["running", "down", "down", "down", "down", "down", "failed"])

    def server(request):
        state = next(answers)
        if state == "down":
            raise httpx.ConnectError("no route to host")
        error = "server restarted while the job was queued or running" if state == "failed" else None
        return httpx.Response(200, json={**job, "state": state, "error": error})

    c = Client("http://x", TOKEN, transport=httpx.MockTransport(server))
    info = c.wait("abc", timeout=10, poll=0)
    assert info.state.value == "failed" and "restarted" in info.error


def test_wait_raises_when_still_unreachable_at_the_deadline(monkeypatch):
    monkeypatch.setattr(http_client, "RETRY_DELAY_S", 0)

    def down(request):
        raise httpx.ConnectError("no route to host")

    with pytest.raises(http_client.Unreachable):
        Client("http://x", TOKEN, transport=httpx.MockTransport(down)).wait("abc", timeout=0.05, poll=0.01)
