import asyncio
import json
from contextlib import contextmanager
from functools import partial
from types import SimpleNamespace

import httpx
import pytest
from conftest import TOKEN, FakeWorker
from fastapi.testclient import TestClient

from gb10_assets.client import mcp_server
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
