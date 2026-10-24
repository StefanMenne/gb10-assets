"""Job schemas and API version: the single source of truth shared by server and client.

A job is submitted as ``{"tool": <name>, "params": {...}}`` plus uploaded input files. ``params`` is validated
against the request model registered for the tool in ``REQUESTS``. Fields listed in ``input_fields`` hold
input file names: the client replaces local paths with the uploaded file names, the server checks that every
referenced file was uploaded.

Bump ``API_VERSION`` on every incompatible change to these models or the HTTP endpoints.
"""
from __future__ import annotations

import re
from datetime import datetime
from enum import Enum
from typing import Annotated, ClassVar

from pydantic import AfterValidator, BaseModel, ConfigDict, Field

API_VERSION = 1

_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def check_file_name(name: str) -> str:
    """Input and output files are flat names inside the job directory, never paths."""
    if not _SAFE_NAME.match(name) or ".." in name:
        raise ValueError(f"invalid file name {name!r}")
    return name


FileName = Annotated[str, AfterValidator(check_file_name)]


def _multiple_of_16(v: int) -> int:
    if v % 16:
        raise ValueError("must be a multiple of 16")
    return v


Side = Annotated[int, Field(ge=256, le=2048), AfterValidator(_multiple_of_16)]
Seed = Annotated[int, Field(ge=0, lt=2**32)]
Count = Annotated[int, Field(ge=1, le=16, description="Number of variants (seeds seed..seed+count-1) in one job; loading the model is the expensive part.")]


class JobRequest(BaseModel):
    """Base class for all tool requests."""

    model_config = ConfigDict(extra="forbid")

    tool: ClassVar[str]
    input_fields: ClassVar[tuple[str, ...]] = ()

    def input_files(self) -> list[str]:
        files: list[str] = []
        for name in self.input_fields:
            value = getattr(self, name)
            if value is None:
                continue
            files.extend(value if isinstance(value, list) else [value])
        return files


class GenerateImage(JobRequest):
    """Text to image (or image + prompt to image) with FLUX.2 [klein] 4B."""

    tool: ClassVar[str] = "flux2-klein"
    input_fields: ClassVar[tuple[str, ...]] = ("reference_images",)

    prompt: str = Field(min_length=1, max_length=4000)
    width: Side = 1024
    height: Side = 1024
    seed: Seed | None = None
    count: Count = 1
    reference_images: list[FileName] = Field(default_factory=list, max_length=4)


REQUESTS: dict[str, type[JobRequest]] = {m.tool: m for m in (GenerateImage,)}


class SubmitJob(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool: str
    params: dict


class JobState(str, Enum):
    queued = "queued"
    running = "running"
    done = "done"
    failed = "failed"
    cancelled = "cancelled"

    @property
    def finished(self) -> bool:
        return self in (JobState.done, JobState.failed, JobState.cancelled)


class JobInfo(BaseModel):
    id: str
    tool: str
    state: JobState
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    queue_position: int | None = Field(None, description="0 = next to run; only while queued")
    outputs: list[str] = Field(default_factory=list, description="Result file names, available when done")
    error: str | None = None
    log_tail: list[str] = Field(default_factory=list)


class Health(BaseModel):
    status: str = "ok"
    api_version: int = API_VERSION
    version: str
    running: str | None = Field(None, description="Id of the running job")
    queued: int = 0
    worker: str | None = Field(None, description="Tool whose worker process is currently alive")
