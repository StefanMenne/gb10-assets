"""Switching the server's host on from the client, with the command the server hands out while it is on.

The server tells how its host is switched on (GET /power-on, set up with `scripts/docker.sh power-on`): usually
a Shelly smart plug in front of a host that boots when its power returns. Every client that reaches the server
keeps that command in ~/.config/gb10-assets/power-on.json (per server URL), so it can still use it when the host
is off and the server cannot answer.

A Shelly plug may still be on while the host is already off: after a shutdown until a plug rule (such as a
script that switches off below 2 W) cuts the power, or for good without such a rule. Switching it "on" would
then change nothing, so a plug that is on while the host draws almost no power is switched off and on again.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Callable

import httpx

from ..shared.schemas import PowerOn
from .http_client import ApiError, Client, Unreachable

HOST_OFF_W = 10.0      # a plug on, the host drawing less: the host is off (GB10 idles at ~30 W)
CYCLE_OFF_S = 10.0     # power off this long before switching on again, so the host sees the power return
DEVICE_TIMEOUT_S = 5.0
WAIT_S = 300.0         # boot and server start take ~1-2 minutes
POLL_S = 5.0


def store_path() -> Path:
    base = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    return base / "gb10-assets" / "power-on.json"


def _load() -> dict:
    try:
        data = json.loads(store_path().read_text())
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _save(data: dict) -> None:
    path = store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(path)


def stored(server_url: str) -> PowerOn | None:
    raw = _load().get(server_url.rstrip("/"))
    try:
        return PowerOn.model_validate(raw) if raw else None
    except ValueError:
        return None


def remember(client: Client) -> PowerOn | None:
    """Ask the server for its power-on command and keep it (forget it when the server has none). Never raises:
    a client that cannot learn the command simply keeps the one it had."""
    try:
        cmd: PowerOn | None = PowerOn.model_validate(client._call("GET", "/power-on", quick=True).json())
    except ApiError as e:
        if "HTTP 404" not in str(e):   # 404: not set up (or a server without the endpoint)
            return stored(client.url)
        cmd = None
    except ValueError:
        return stored(client.url)
    data = _load()
    key = client.url.rstrip("/")
    new = cmd.model_dump() if cmd else None
    if data.get(key) != new:
        if new is None:
            data.pop(key, None)
        else:
            data[key] = new
        try:
            _save(data)
        except OSError:
            pass
    return cmd


class Shelly:
    """A Shelly plug or relay: Gen1 HTTP API (/relay/N) or Gen2+ RPC (/rpc/Switch.*), detected from /shelly."""

    def __init__(self, cmd: PowerOn, http: httpx.Client):
        self.base, self.id, self.http = cmd.url.rstrip("/"), cmd.switch, http
        info = self._get("/shelly")
        self.gen = int(info.get("gen", 1))

    def _get(self, path: str, **params) -> dict:
        try:
            r = self.http.get(self.base + path, params=params, timeout=DEVICE_TIMEOUT_S)
            r.raise_for_status()
            return r.json()
        except (httpx.HTTPError, ValueError) as e:
            raise ApiError(f"power switch {self.base} not reachable: {e}") from None

    def status(self) -> tuple[bool, float | None]:
        """(on, watts; None if the device does not measure)."""
        if self.gen >= 2:
            s = self._get("/rpc/Switch.GetStatus", id=self.id)
            return bool(s["output"]), s.get("apower")
        s = self._get(f"/relay/{self.id}")
        try:
            watts = self._get("/status")["meters"][self.id]["power"]
        except (ApiError, KeyError, IndexError, TypeError):
            watts = None
        return bool(s["ison"]), watts

    def set(self, on: bool) -> None:
        if self.gen >= 2:
            self._get("/rpc/Switch.Set", id=self.id, on="true" if on else "false")
        else:
            self._get(f"/relay/{self.id}", turn="on" if on else "off")


def switch_on(cmd: PowerOn, http: httpx.Client | None = None, sleep: Callable[[float], None] = time.sleep,
              log: Callable[[str], None] = lambda m: None) -> str:
    """Run the power-on command once; returns what was done."""
    http = http or httpx.Client()
    if cmd.kind == "http":
        try:
            http.get(cmd.url, timeout=DEVICE_TIMEOUT_S).raise_for_status()
        except httpx.HTTPError as e:
            raise ApiError(f"power-on URL {cmd.url} failed: {e}") from None
        return f"called {cmd.url}"
    plug = Shelly(cmd, http)
    on, watts = plug.status()
    if not on:
        plug.set(True)
        return f"switched the plug {cmd.url} on"
    if watts is not None and watts < HOST_OFF_W:
        log(f"plug {cmd.url} is on, but the host draws {watts:.1f} W (off): switching the power off and on")
        plug.set(False)
        sleep(CYCLE_OFF_S)
        plug.set(True)
        return f"switched the plug {cmd.url} off and on again (it was on, the host off at {watts:.1f} W)"
    return f"the plug {cmd.url} is on and the host draws {watts if watts is None else round(watts, 1)} W: " \
           "it is running or starting"


def power_on(client: Client, wait: float = WAIT_S, http: httpx.Client | None = None,
             sleep: Callable[[float], None] = time.sleep, log: Callable[[str], None] = lambda m: None) -> str:
    """Switch the server's host on with the stored command and wait up to `wait` seconds for the server."""
    try:
        h = client.health(quick=True)
        remember(client)
        return f"the server {client.url} is already on (version {h.version})"
    except Unreachable:
        pass
    cmd = stored(client.url)
    if cmd is None:
        raise ApiError(f"no power-on command known for {client.url}: the server has to be reached once while "
                       f"it is on (and set up with `scripts/docker.sh power-on` on the host)")
    done = switch_on(cmd, http, sleep, log)
    log(done)
    if wait <= 0:
        return done
    start = time.monotonic()
    deadline = start + wait
    while time.monotonic() < deadline:
        sleep(POLL_S)
        try:
            h = client.health(quick=True)
            return f"{done}; the server is up after {time.monotonic() - start:.0f} s (version {h.version})"
        except Unreachable:
            continue
    raise ApiError(f"{done}, but the server did not answer within {wait:.0f} s. Is the host set to power on when "
                   "its power returns (BIOS: restore on AC power loss)? Check the host's power LED.")
