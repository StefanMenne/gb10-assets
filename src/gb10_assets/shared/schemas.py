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

# LTX-2.5 memory on GB10, the same model as memory_gb in tools/ltx-video/run.py (see there; tests compare them):
# 1536x1024 fits 6 s, 1280x832 9 s, 1024x704 13 s, 1024x576 16 s, 768x512 20 s at 24 fps
LTX_MAX_GB = 98.0
LTX_KEYFRAMES_S = 5.0


def ltx_frames(seconds: float, fps: int) -> int:
    return min(481, round(seconds * fps / 8) * 8 + 1)


def ltx_memory_gb(width: int, height: int, frames: int, fps: int) -> float:
    return 51 + 0.172 * width * height * frames / 1e6 + 0.39 * frames / fps


def ltx_max_seconds(width: int, height: int, fps: int) -> float:
    """Longest video of this size under LTX_MAX_GB."""
    frames = 481
    while frames > 9 and ltx_memory_gb(width, height, frames, fps) > LTX_MAX_GB:
        frames -= 8
    return (frames - 1) / fps


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
        "Default: predicted from the prompt (one image or none, at most what fits the size), 5 s (several images), "
        "length of audio (max 20 s). Memory limits the length per size: 1536x1024 6 s, 1280x832 9 s, "
        "1024x704 13 s, 768x512 20 s (at 24 fps)"))
    width: VideoSide | None = Field(None, description=(
        "Default 1536x1024, or the first image's aspect ratio; for longer clips it shrinks to fit the memory"))
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
        seconds = self.duration_s or (LTX_KEYFRAMES_S if len(self.images) > 1 and self.audio is None else None)
        if self.video is None and self.width is not None and seconds:
            need = ltx_memory_gb(self.width, self.height, ltx_frames(seconds, self.fps), self.fps)
            if need > LTX_MAX_GB:
                raise ValueError(
                    f"{self.width}x{self.height} for {seconds:g} s at {self.fps} fps needs ~{need:.0f} GB, more than "
                    f"the {LTX_MAX_GB:.0f} GB a video may use: at this size at most "
                    f"{ltx_max_seconds(self.width, self.height, self.fps):.1f} s, or choose a smaller size "
                    "(768x512 fits 20 s)")
        return self


class ImageTo3D(JobRequest):
    """Pixal3D (on TRELLIS.2, run by ComfyUI): one image to a textured 3D model (GLB with PBR materials). The
    shape follows the image pixel-aligned, in the camera perspective of the photo."""

    tool: ClassVar[str] = "pixal3d"
    input_fields: ClassVar[tuple[str, ...]] = ("image",)

    image: FileName = Field(description="One object, ideally on a plain background")
    remove_background: bool = Field(True, description="Cut the object out first (BiRefNet); off for images that "
                                                      "already have a transparent or clean black background")
    resolution: Literal[1024, 1536] = Field(1536, description="Voxel resolution of the shape: 1536 more detail, "
                                                              "1024 about twice as fast")
    max_faces: int = Field(700_000, ge=10_000, le=2_000_000, description="Faces of the exported mesh")
    texture_size: Literal[1024, 2048, 4096] = Field(4096, description="Side of the PBR texture maps")
    seed: Seed | None = None
    count: Annotated[int, Field(ge=1, le=4, description="Number of variants (seeds seed..seed+count-1)")] = 1


# tools/blender/scripts/<name>.py (tests check that both agree); usage and options in each script's docstring
BLENDER_SCRIPTS = {
    "glb_post": "in.glb out.glb [--height M] [--emission 0.5] [--tex 1024]: stand a generated model on the "
                "ground, centre and scale it, brighten it for unlit viewers, shrink its textures",
    "preview": "model.glb out.png [--views 4] [--action Walk --frame 6]: render the model from several sides "
               "(EEVEE) into one image",
    "rig_quadruped": "in.glb out.glb --landmarks giraffe|pony|sheep|FILE.json: four-legged rig with animations "
                     "Idle, Nod, Talk, Walk",
    "rig_biped": "in.glb out.glb --landmarks elephant|rabbit|FILE.json [--mode walk|hop]: two-legged rig (IK "
                 "legs) with animations Idle, Walk",
    "rig_hop": "in.glb out.glb [--yaw 0] [--height 0]: 3-bone hop rig for any figure, animations Idle, Walk",
    "eye_geo": "in.glb out.glb --eyes eyes.json [--apart R] [--drop R] [--look R]: insert eyes as geometry",
    "tex_eyes": "in.glb out.glb --style black|white [--save-eyes eyes.json]: find painted eyes in the texture "
                "and paint them anew",
    "tex_eyecover": "in.glb out.glb --eyes eyes.json: paint over painted eyes with the surrounding color "
                    "(before eye_geo)",
    "tex_adjust": "in.glb out.glb [--recolor R,G,B>R,G,B@radius [--region box]] [--whiten box] [--metal 0]: "
                  "recolor the texture, also only in a 3D region, remove metal",
    "tex_hair": "in.glb out.glb --color R,G,B@radius|--dark L [--region box]: paint hair strands into the texture",
    "talk_video": "rig.blend voice.wav out.mp4 [--mode fast|slow]: video of a rig_quadruped figure speaking "
                  "the recording (jaw follows the loudness)",
}
ScriptArg = Annotated[str, Field(max_length=4000)]


class BlenderScript(JobRequest):
    """Run a Python script in Blender 5.2 (bpy) on input files: one of the bundled scripts or own code. The
    script runs in a directory with the input files, reads them by name and writes its results there under new
    names; every new file is returned, print() output goes to the job log."""

    tool: ClassVar[str] = "blender"
    input_fields: ClassVar[tuple[str, ...]] = ("files",)

    files: list[FileName] = Field(default_factory=list, max_length=32, description="Input files")
    script: Literal[tuple(BLENDER_SCRIPTS)] | None = Field(None, description="A bundled script")  # type: ignore[valid-type]
    code: str | None = Field(None, min_length=1, max_length=200_000, description=(
        "Own Python code instead of a bundled script; the server may disable this (BLENDER_CODE=0)"))
    args: list[ScriptArg] = Field(default_factory=list, max_length=256, description=(
        "Command line arguments of the script (sys.argv[1:]); input files by their file name"))

    @model_validator(mode="after")
    def _script_or_code(self):
        if (self.script is None) == (self.code is None):
            raise ValueError("give either script or code")
        return self


REQUESTS: dict[str, type[JobRequest]] = {
    m.tool: m for m in (GenerateImage, EditImage, Speak, SpeakMultilingual, GenerateVideo, ImageTo3D,
                        BlenderScript)}


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
