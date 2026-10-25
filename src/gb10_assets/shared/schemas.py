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
from typing import Annotated, ClassVar, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator

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


class EditImage(JobRequest):
    """Edit one to three images with a text instruction (Qwen-Image-Edit-2511 with the 4-step Lightning LoRA)."""

    tool: ClassVar[str] = "qwen-image-edit"
    input_fields: ClassVar[tuple[str, ...]] = ("images",)

    prompt: str = Field(min_length=1, max_length=4000)
    images: list[FileName] = Field(min_length=1, max_length=3)
    width: Side | None = Field(None, description="Default: aspect ratio of the first image at about 1 megapixel")
    height: Side | None = None
    seed: Seed | None = None
    count: Count = 1


# ISO 639-1 codes; each worker maps them to its model's own language names
QwenLanguage = Literal["auto", "de", "en", "es", "fr", "it", "ja", "ko", "pt", "ru", "zh"]
ConfuciusLanguage = Literal["de", "en", "es", "fr", "id", "it", "ja", "ko", "ms", "pt", "ru", "th", "vi", "zh"]
Texts = Annotated[list[Annotated[str, Field(min_length=1, max_length=2000)]], Field(
    min_length=1, max_length=64, description="One output file per text; the model is loaded once for all of them")]
SpeechCount = Annotated[int, Field(ge=1, le=16, description="Variants per text (seeds seed..seed+count-1)")]


class Speak(JobRequest):
    """Text to speech with Qwen3-TTS 1.7B: either a new voice from a description (VoiceDesign model) or a cloned
    voice from a reference recording (Base model)."""

    tool: ClassVar[str] = "qwen-tts"
    input_fields: ClassVar[tuple[str, ...]] = ("reference_audio",)

    texts: Texts
    language: QwenLanguage = "auto"
    voice_description: str | None = Field(None, min_length=1, max_length=2000, description=(
        "VoiceDesign: natural-language description of the voice (gender, age, pitch, timbre, emotion, pace)"))
    reference_audio: FileName | None = Field(None, description="Voice cloning: recording of the voice, 3-20 s")
    reference_text: str | None = Field(None, min_length=1, max_length=2000, description=(
        "Transcript of reference_audio; without it only the speaker embedding is used (lower similarity)"))
    seed: Seed | None = None
    count: SpeechCount = 1

    @model_validator(mode="after")
    def _one_voice(self):
        if (self.voice_description is None) == (self.reference_audio is None):
            raise ValueError("give either voice_description or reference_audio")
        if self.reference_text is not None and self.reference_audio is None:
            raise ValueError("reference_text needs reference_audio")
        return self


class SpeakMultilingual(JobRequest):
    """Cross-lingual voice cloning with Confucius4-TTS: the reference may speak another language than the text."""

    tool: ClassVar[str] = "confucius-tts"
    input_fields: ClassVar[tuple[str, ...]] = ("reference_audio",)

    texts: Texts
    language: ConfuciusLanguage
    reference_audio: FileName = Field(description="Recording of the voice in any language, 3-20 s")
    seed: Seed | None = None
    count: SpeechCount = 1


def _multiple_of_64(v: int) -> int:
    if v % 64:
        raise ValueError("must be a multiple of 64")
    return v


VideoSide = Annotated[int, Field(ge=256, le=2048), AfterValidator(_multiple_of_64)]


class GenerateVideo(JobRequest):
    """LTX-2.5 (22B distilled, NVFP4): video with sound from text, keyframe images or an audio track, or a retake
    of a time range of an existing video. The mode follows from the inputs: ``video`` -> retake, ``audio`` -> audio
    to video (with an optional first frame in ``images``), otherwise text/images to video."""

    tool: ClassVar[str] = "ltx-video"
    input_fields: ClassVar[tuple[str, ...]] = ("images", "audio", "video")

    prompt: str = Field(min_length=1, max_length=8000)
    images: list[FileName] = Field(default_factory=list, max_length=4, description=(
        "Keyframes spread evenly over the clip: one image is the first frame, two are first and last frame"))
    audio: FileName | None = Field(None, description="Audio to video: the soundtrack the video is generated for")
    video: FileName | None = Field(None, description="Retake: video whose time range start_s..end_s is regenerated")
    start_s: float | None = Field(None, ge=0)
    end_s: float | None = Field(None, gt=0)
    regenerate: Literal["both", "video", "audio"] = Field("both", description="Retake: which stream to regenerate")
    duration_s: float | None = Field(None, ge=1, le=20, description=(
        "Default: predicted from the prompt (one image or none), 5 s (several images), length of audio (max 20 s)"))
    width: VideoSide | None = Field(None, description="Default 1536x1024, or the first image's aspect ratio")
    height: VideoSide | None = None
    fps: int = Field(24, ge=12, le=50, description="Frame rate; a retake keeps the frame rate of the video")
    enhance_prompt: bool = Field(False, description="Let the Gemma text encoder expand a short prompt first")
    seed: Seed | None = None
    count: Annotated[int, Field(ge=1, le=8, description="Number of variants (seeds seed..seed+count-1)")] = 1

    @model_validator(mode="after")
    def _one_mode(self):
        if self.video is not None:
            if self.images or self.audio is not None:
                raise ValueError("a retake takes only video, no images or audio")
            if self.start_s is None or self.end_s is None or self.start_s >= self.end_s:
                raise ValueError("a retake needs start_s < end_s")
        elif self.start_s is not None or self.end_s is not None or self.regenerate != "both":
            raise ValueError("start_s, end_s and regenerate need video")
        if self.audio is not None and len(self.images) > 1:
            raise ValueError("audio to video takes at most one image (the first frame)")
        if (self.width is None) != (self.height is None):
            raise ValueError("give both width and height or neither")
        return self


REQUESTS: dict[str, type[JobRequest]] = {
    m.tool: m for m in (GenerateImage, EditImage, Speak, SpeakMultilingual, GenerateVideo)}


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
