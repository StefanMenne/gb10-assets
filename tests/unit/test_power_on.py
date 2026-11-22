"""Switching the host on: the server hands out POWER_ON, clients keep it and run it while the host is off."""
import httpx
import pytest
from conftest import TOKEN, FakeWorker
from fastapi.testclient import TestClient

from gb10_assets.client import power
from gb10_assets.client.http_client import ApiError, Client
from gb10_assets.server.app import create_app
from gb10_assets.server.config import Settings
from gb10_assets.server.queue import JobQueue
from gb10_assets.shared.schemas import PowerOn


@pytest.fixture(autouse=True)
def _config_home(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))


@pytest.mark.parametrize("spec, kind, url, switch", [
    ("shelly://shelly1", "shelly", "http://shelly1", 0),
    ("shelly://192.168.178.30/1", "shelly", "http://192.168.178.30", 1),
    ("http://plug/on?x=1", "http", "http://plug/on?x=1", 0),
])
def test_parse(spec, kind, url, switch):
    assert PowerOn.parse(spec) == PowerOn(kind=kind, url=url, switch=switch)


@pytest.mark.parametrize("spec", ["shelly://", "ftp://x", "shelly1", "shelly://a/b"])
def test_parse_rejects(spec):
    with pytest.raises(ValueError):
        PowerOn.parse(spec)


def test_bad_setting_stops_the_server(monkeypatch):
    monkeypatch.setenv("GB10_ASSETS_TOKEN", TOKEN)
    monkeypatch.setenv("POWER_ON", "shelly1")
    with pytest.raises(SystemExit, match="POWER_ON"):
        Settings.from_env()
    monkeypatch.setenv("POWER_ON", " shelly://shelly1 ")
    assert Settings.from_env().power_on == "shelly://shelly1"


def _client(settings):
    queue = JobQueue(settings, worker_factory=FakeWorker, mem_gb=lambda: 200.0)
    tc = TestClient(create_app(settings, queue), headers={"Authorization": f"Bearer {TOKEN}"})
    c = Client("http://testserver", TOKEN)
    c.http = tc
    return tc, c


def test_client_remembers_and_forgets(settings):
    settings.power_on = "shelly://shelly1"
    with _client(settings)[0] as tc:
        assert tc.get("/power-on", headers={"Authorization": "Bearer wrong"}).status_code == 401
        c = Client("http://testserver", TOKEN)
        c.http = tc
        assert power.remember(c) == PowerOn(kind="shelly", url="http://shelly1")
        assert power.stored("http://testserver/") == PowerOn(kind="shelly", url="http://shelly1")
    settings.power_on = None
    with _client(settings)[0] as tc:
        c.http = tc
        assert tc.get("/power-on").status_code == 404
        assert power.remember(c) is None and power.stored("http://testserver") is None


def test_remember_keeps_the_command_when_the_server_is_off(settings):
    settings.power_on = "shelly://shelly1"
    with _client(settings)[0] as tc:
        c = Client("http://testserver", TOKEN)
        c.http = tc
        power.remember(c)
    c.http = httpx.Client(transport=httpx.MockTransport(_down), base_url="http://testserver")
    assert power.remember(c) == PowerOn(kind="shelly", url="http://shelly1")


def _down(request):
    raise httpx.ConnectError("connection refused")


class FakeShelly:
    """A Gen2+ Shelly plug (RPC) in front of a host that boots when the power returns."""

    def __init__(self, on, watts, gen=3):
        self.on, self.watts, self.gen, self.calls = on, watts, gen, []

    def __call__(self, request):
        path, q = request.url.path, request.url.params
        self.calls.append((path, dict(q)))
        if path == "/shelly":
            return httpx.Response(200, json={"gen": self.gen, "model": "S3PL-30110EU"} if self.gen >= 2
                                  else {"type": "SHPLG-S"})
        if path == "/rpc/Switch.GetStatus":
            return httpx.Response(200, json={"id": 0, "output": self.on, "apower": self.watts})
        if path == "/rpc/Switch.Set":
            self.on = q["on"] == "true"
            return httpx.Response(200, json={"was_on": not self.on})
        if path == "/relay/0":
            if "turn" in q:
                self.on = q["turn"] == "on"
            return httpx.Response(200, json={"ison": self.on})
        if path == "/status":
            return httpx.Response(200, json={"meters": [{"power": self.watts}]})
        return httpx.Response(404)


def _sets(plug):
    return [q.get("on", q.get("turn")) for p, q in plug.calls if p in ("/rpc/Switch.Set", "/relay/0") and q]


CMD = PowerOn(kind="shelly", url="http://shelly1")


@pytest.mark.parametrize("gen", [1, 3])
def test_plug_off_is_switched_on(gen):
    plug = FakeShelly(on=False, watts=0.0, gen=gen)
    done = power.switch_on(CMD, httpx.Client(transport=httpx.MockTransport(plug)), sleep=lambda s: None)
    assert plug.on and _sets(plug) in (["true"], ["on"]) and "switched the plug" in done


def test_plug_on_with_host_off_is_cycled():
    """Within the plug's auto-off delay after a shutdown, or without auto-off: on alone would change nothing."""
    plug, slept = FakeShelly(on=True, watts=1.2), []
    done = power.switch_on(CMD, httpx.Client(transport=httpx.MockTransport(plug)), sleep=slept.append)
    assert _sets(plug) == ["false", "true"] and slept == [power.CYCLE_OFF_S] and "off and on" in done


def test_running_host_is_left_alone():
    plug = FakeShelly(on=True, watts=29.0)
    power.switch_on(CMD, httpx.Client(transport=httpx.MockTransport(plug)), sleep=lambda s: None)
    assert _sets(plug) == []


def test_http_command():
    seen = []
    http = httpx.Client(transport=httpx.MockTransport(lambda r: seen.append(str(r.url)) or httpx.Response(200)))
    power.switch_on(PowerOn(kind="http", url="http://plug/on"), http)
    assert seen == ["http://plug/on"]


def test_power_on_waits_for_the_server(settings, monkeypatch):
    settings.power_on = "shelly://shelly1"
    tc, c = _client(settings)
    with tc:
        power.remember(c)
        up = {"after": 3}

        def server(request):   # answers from the 3rd health check on, as a host that is booting
            up["after"] -= 1
            if up["after"] > 0:
                raise httpx.ConnectError("refused")
            r = tc.get(request.url.path)
            return httpx.Response(r.status_code, content=r.content, headers={"content-type": "application/json"})

        c.http = httpx.Client(transport=httpx.MockTransport(server), base_url="http://testserver")
        plug = FakeShelly(on=False, watts=0.0)
        msg = power.power_on(c, wait=60, http=httpx.Client(transport=httpx.MockTransport(plug)),
                             sleep=lambda s: None)
    assert plug.on and "the server is up" in msg


def test_power_on_without_a_kept_command():
    c = Client("http://testserver", TOKEN)
    c.http = httpx.Client(transport=httpx.MockTransport(_down), base_url="http://testserver")
    with pytest.raises(ApiError, match="no power-on command"):
        power.power_on(c, wait=0)


def test_power_on_does_nothing_when_the_server_is_up(settings):
    tc, c = _client(settings)
    with tc:
        assert "already on" in power.power_on(c, wait=0)
