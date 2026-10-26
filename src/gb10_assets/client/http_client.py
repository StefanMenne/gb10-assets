"""HTTP client for the gb10-assets server, shared by the MCP adapter and the CLI.

Configuration: GB10_ASSETS_URL (e.g. http://<gb10-host>:8750) and GB10_ASSETS_TOKEN (unset or "none" when the
server runs without authentication).
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import httpx
from pydantic import ValidationError

from ..shared.schemas import API_VERSION, REQUESTS, Health, JobInfo, JobState


GET_RETRIES, RETRY_DELAY_S = 3, 1.0


class ApiError(RuntimeError):
    pass


class VersionMismatch(ApiError):
    pass


def upload_name(path: Path, taken: set[str]) -> str:
    """A flat, safe, unique file name for an uploaded input."""
    stem = re.sub(r"[^A-Za-z0-9._-]", "_", path.stem).strip("._-")[:80] or "input"
    suffix = re.sub(r"[^A-Za-z0-9.]", "", path.suffix)[:10]
    name, k = f"{stem}{suffix}", 1
    while name in taken:
        name, k = f"{stem}_{k}{suffix}", k + 1
    taken.add(name)
    return name


def free_path(target: Path) -> Path:
    """target, or target with _1, _2 ... appended if a file of that name already exists."""
    k, path = 1, target
    while path.exists():
        path = target.with_name(f"{target.stem}_{k}{target.suffix}")
        k += 1
    return path


class Client:
    def __init__(self, url: str | None = None, token: str | None = None, *,
                 transport: httpx.BaseTransport | None = None, timeout: float = 60):
        url = url or os.environ.get("GB10_ASSETS_URL")
        token = token or os.environ.get("GB10_ASSETS_TOKEN")
        if not url:
            raise ApiError("GB10_ASSETS_URL must be set (server address, e.g. http://<gb10-host>:8750)")
        self.url = url.rstrip("/")
        headers = {"Authorization": f"Bearer {token}"} if token and token.lower() != "none" else {}
        self.http = httpx.Client(base_url=self.url, headers=headers, timeout=timeout, transport=transport)

    def _call(self, method: str, path: str, **kw) -> httpx.Response:
        # GETs are retried: a keep-alive connection the server closes just as the next poll reuses it fails with
        # "Server disconnected without sending a response" although the server is fine
        attempts = GET_RETRIES + 1 if method == "GET" else 1
        for attempt in range(attempts):
            try:
                r = self.http.request(method, path, **kw)
                break
            except httpx.TransportError as e:
                if attempt + 1 == attempts:
                    raise ApiError(f"server {self.url} not reachable: {e}") from None
                time.sleep(RETRY_DELAY_S)
            except httpx.HTTPError as e:
                raise ApiError(f"server {self.url} not reachable: {e}") from None
        if r.status_code >= 400:
            try:
                detail = r.json().get("detail", r.text)
            except ValueError:
                detail = r.text
            raise ApiError(f"{method} {path} -> HTTP {r.status_code}: {detail}")
        return r

    def health(self) -> Health:
        return Health.model_validate(self._call("GET", "/health").json())

    def check_version(self) -> Health:
        h = self.health()
        if h.api_version != API_VERSION:
            raise VersionMismatch(
                f"API version mismatch: server {self.url} speaks v{h.api_version} (gb10-assets {h.version}), "
                f"this client speaks v{API_VERSION}. Update the plugin (`/plugin update gb10-assets`) or the "
                f"server (`git pull && docker compose up -d --build`) so both run the same release.")
        return h

    def submit(self, tool: str, params: dict) -> JobInfo:
        """Submit a job. Input fields of the tool's request model hold local paths; they are uploaded. Script
        arguments (``args``, blender) equal to one of these paths are replaced by its uploaded file name."""
        model = REQUESTS.get(tool)
        if model is None:
            raise ApiError(f"unknown tool {tool!r}")
        params, uploads, taken, renamed = dict(params), [], set(), {}
        for field in model.input_fields:
            value = params.get(field)
            if value is None:
                continue
            given = value if isinstance(value, list) else [value]
            names = []
            for raw in given:
                p = Path(raw).expanduser()
                if not p.is_file():
                    raise ApiError(f"input file not found: {p}")
                names.append(upload_name(p, taken))
                uploads.append((p, names[-1]))
                renamed[str(raw)] = names[-1]
            params[field] = names if isinstance(value, list) else names[0]
        if isinstance(params.get("args"), list):
            params["args"] = [renamed.get(arg, arg) for arg in params["args"]]
        try:
            req = model.model_validate(params)
        except ValidationError as e:
            raise ApiError(f"invalid parameters for {tool}: {e}") from None
        body = {"request": (None, json.dumps({"tool": tool, "params": req.model_dump(mode="json")}), "application/json")}
        handles = [open(p, "rb") for p, _ in uploads]
        try:
            files = list(body.items()) + [("files", (name, fh)) for (_, name), fh in zip(uploads, handles)]
            return JobInfo.model_validate(self._call("POST", "/jobs", files=files).json())
        finally:
            for fh in handles:
                fh.close()

    def status(self, job_id: str) -> JobInfo:
        return JobInfo.model_validate(self._call("GET", f"/jobs/{job_id}").json())

    def wait(self, job_id: str, timeout: float, poll: float = 2.0) -> JobInfo:
        """Poll until the job is finished or timeout seconds have passed; returns the latest state."""
        deadline = time.monotonic() + timeout
        info = self.status(job_id)
        while not info.state.finished and time.monotonic() < deadline:
            time.sleep(min(poll, max(0.0, deadline - time.monotonic())))
            info = self.status(job_id)
        return info

    def fetch(self, job_id: str, target_dir: str | Path, wait: float = 0,
              delete: bool = True) -> tuple[JobInfo, list[Path]]:
        """Download the results into target_dir once the job is done, then delete the job on the server.

        Returns (info, []) while the job is still queued or running. Raises ApiError if it failed.
        """
        info = self.wait(job_id, wait) if wait > 0 else self.status(job_id)
        if info.state in (JobState.failed, JobState.cancelled):
            tail = "\n".join(info.log_tail[-10:])
            raise ApiError(f"job {job_id} {info.state.value}: {info.error}" + (f"\n--- log ---\n{tail}" if tail else ""))
        if info.state != JobState.done:
            return info, []
        target = Path(target_dir).expanduser()
        target.mkdir(parents=True, exist_ok=True)
        paths = []
        for name in info.outputs:
            dest = free_path(target / name)
            tmp = dest.with_name(dest.name + ".part")
            try:
                with self.http.stream("GET", f"/jobs/{job_id}/files/{name}") as r:
                    if r.status_code >= 400:
                        raise ApiError(f"download of {name} failed: HTTP {r.status_code}")
                    with open(tmp, "wb") as f:
                        for chunk in r.iter_bytes():
                            f.write(chunk)
                tmp.replace(dest)
            finally:
                tmp.unlink(missing_ok=True)
            paths.append(dest)
        if delete:
            self.delete(job_id)
        return info, paths

    def log(self, job_id: str) -> str:
        return self._call("GET", f"/jobs/{job_id}/log").text

    def delete(self, job_id: str) -> None:
        self._call("DELETE", f"/jobs/{job_id}")
