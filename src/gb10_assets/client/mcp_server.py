"""MCP server over stdio, started by Claude Code as a child process (see .mcp.json).

Long jobs follow submit -> job_id -> job_status -> fetch_result. Generating tools return immediately with a
job id; fetch_result blocks only as long as its ``wait`` argument allows.
"""
from __future__ import annotations

import sys
from typing import Annotated

from mcp.server.mcpserver import MCPServer
from pydantic import Field

from .. import __version__
from ..shared.schemas import JobInfo
from .http_client import ApiError, Client, VersionMismatch

MAX_WAIT_S = 600

INSTRUCTIONS = """Generates media assets on a self-hosted NVIDIA GB10 server.
Generating tools only queue a job and return a job_id. Then call fetch_result(job_id, target_dir, wait=...)
to download the results; with wait=0 it returns the current state without blocking. The server runs one GPU
job at a time and loads the model per job, so request all variants in one call (count) instead of many calls.
Look at generated images with the Read tool before using them."""


class Adapter:
    def __init__(self, client: Client | None = None):
        self._client = client
        self._checked = False

    @property
    def client(self) -> Client:
        if self._client is None:
            self._client = Client()
        if not self._checked:
            self._client.check_version()
            self._checked = True
        return self._client

    def startup_check(self) -> None:
        """Fail hard on missing configuration or a version mismatch; an unreachable server is only a warning
        (the check is retried on the first tool call)."""
        if self._client is None:
            self._client = Client()
        try:
            h = self._client.check_version()
            self._checked = True
            print(f"[gb10-assets] connected to {self._client.url}, server {h.version}, API v{h.api_version}",
                  file=sys.stderr)
        except VersionMismatch:
            raise
        except ApiError as e:
            print(f"[gb10-assets] warning: {e}", file=sys.stderr)


def submitted(info: JobInfo) -> dict:
    return {"job_id": info.id, "state": info.state.value, "queue_position": info.queue_position,
            "next": f"fetch_result(job_id='{info.id}', target_dir=..., wait=...)"}


def build(adapter: Adapter | None = None) -> MCPServer:
    adapter = adapter or Adapter()
    mcp = MCPServer("gb10-assets", instructions=INSTRUCTIONS, version=__version__)

    @mcp.tool()
    def generate_image(
        prompt: Annotated[str, Field(description="English prompt: subject, style, view, lighting, background")],
        width: Annotated[int, Field(description="Multiple of 16, 256-2048")] = 1024,
        height: Annotated[int, Field(description="Multiple of 16, 256-2048")] = 1024,
        seed: Annotated[int | None, Field(description="Start seed; variants use seed..seed+count-1")] = None,
        count: Annotated[int, Field(description="Number of variants in this job (1-16); a sheet.png is added if > 1")] = 1,
        reference_images: Annotated[list[str] | None, Field(
            description="Local image paths to edit or recolor with the prompt (max 4)")] = None,
    ) -> dict:
        """Create images from a text prompt with FLUX.2 [klein] 4B, optionally guided by reference images
        (fast recoloring and restyling that keeps the pose). Returns a job_id; fetch with fetch_result."""
        params = {"prompt": prompt, "width": width, "height": height, "seed": seed, "count": count,
                  "reference_images": reference_images or []}
        return submitted(adapter.client.submit("flux2-klein", params))

    @mcp.tool()
    def edit_image(
        prompt: Annotated[str, Field(description="English edit instruction, e.g. 'replace the background with a "
                                                 "beach', 'the character from image 1 sits on the chair from image 2'")],
        images: Annotated[list[str], Field(description="Local paths of the input images (1-3)")],
        seed: Annotated[int | None, Field(description="Start seed; variants use seed..seed+count-1")] = None,
        count: Annotated[int, Field(description="Number of variants in this job (1-16); a sheet.png is added if > 1")] = 1,
        width: Annotated[int | None, Field(description="Multiple of 16; default keeps the first image's aspect ratio")] = None,
        height: Annotated[int | None, Field(description="Multiple of 16; default keeps the first image's aspect ratio")] = None,
    ) -> dict:
        """Edit images with a text instruction using Qwen-Image-Edit-2511: change objects, text, style or
        background, keep a character consistent, or combine up to three images (4-step Lightning version).
        Returns a job_id; fetch with fetch_result."""
        params = {"prompt": prompt, "images": images, "seed": seed, "count": count, "width": width,
                  "height": height}
        return submitted(adapter.client.submit("qwen-image-edit", params))

    @mcp.tool()
    def job_status(job_id: str) -> dict:
        """Current state of a job: queued (with queue_position), running, done, failed or cancelled."""
        return adapter.client.status(job_id).model_dump(mode="json", exclude_none=True)

    @mcp.tool()
    def fetch_result(
        job_id: str,
        target_dir: Annotated[str, Field(description="Local directory for the result files (created if missing)")],
        wait: Annotated[int, Field(description=f"Seconds to wait for the job to finish (0-{MAX_WAIT_S}); "
                                               "0 returns the current state immediately")] = 0,
    ) -> dict:
        """Download the results of a finished job into target_dir and delete the job on the server.
        Existing files are not overwritten (a suffix _1, _2 ... is added). If the job is not done yet, returns
        its state; call again later."""
        info, paths = adapter.client.fetch(job_id, target_dir, wait=max(0, min(wait, MAX_WAIT_S)))
        if not paths:
            return {"job_id": job_id, "state": info.state.value, "queue_position": info.queue_position,
                    "files": [], "hint": "not finished yet; call fetch_result again with wait > 0"}
        return {"job_id": job_id, "state": info.state.value, "files": [str(p.resolve()) for p in paths]}

    @mcp.tool()
    def cancel_job(job_id: str) -> dict:
        """Cancel a queued or running job and delete it with all its files on the server."""
        adapter.client.delete(job_id)
        return {"job_id": job_id, "state": "deleted"}

    return mcp


def run() -> None:
    adapter = Adapter()
    try:
        adapter.startup_check()
    except ApiError as e:  # version mismatch or missing configuration
        print(f"[gb10-assets] {e}", file=sys.stderr)
        raise SystemExit(1) from None
    build(adapter).run("stdio")

