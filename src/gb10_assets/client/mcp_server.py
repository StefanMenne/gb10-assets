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
Look at generated images with the Read tool before using them.
Speech: design_voice creates a voice from a description, speak clones a voice for de/en and 8 more languages,
speak_multilingual clones a voice into languages such as Thai, even from a reference in another language.
Video (with sound): generate_video from text and keyframe images, audio_to_video for a given soundtrack,
retake_video to regenerate a time range of a video. A video job takes minutes; look at the .jpg frames.
3D: image_to_3d turns one image of an object into a textured GLB (several minutes); look at the .png render."""


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
    def design_voice(
        voice_description: Annotated[str, Field(description=(
            "Description of the voice, e.g. 'warm female voice, about 40, low pitch, calm and friendly, slow pace'"))],
        text: Annotated[str, Field(description="What the voice says; 5-15 s of speech makes a good clone reference")],
        language: Annotated[str, Field(description="auto, de, en, es, fr, it, ja, ko, pt, ru or zh")] = "auto",
        count: Annotated[int, Field(description="Number of voice variants to choose from (1-16)")] = 4,
        seed: Annotated[int | None, Field(description="Start seed; variants use seed..seed+count-1")] = None,
    ) -> dict:
        """Create a new voice from a text description with Qwen3-TTS VoiceDesign. Each variant is a
        speech_01_<seed>.wav plus a .txt with its transcript; use a chosen pair as reference_audio and
        reference_text for speak, or the wav alone for speak_multilingual. Returns a job_id; fetch with
        fetch_result."""
        params = {"texts": [text], "voice_description": voice_description, "language": language, "count": count,
                  "seed": seed}
        return submitted(adapter.client.submit("qwen-tts", params))

    @mcp.tool()
    def speak(
        texts: Annotated[list[str], Field(description="Texts to speak, one wav per text (max 64)")],
        reference_audio: Annotated[str, Field(description="Local path of a recording of the voice (3-20 s)")],
        reference_text: Annotated[str | None, Field(description=(
            "Exact transcript of reference_audio; strongly recommended, without it the voice is less similar"))] = None,
        language: Annotated[str, Field(description="auto, de, en, es, fr, it, ja, ko, pt, ru or zh")] = "auto",
        seed: Annotated[int | None, Field(description="Seed for reproducible results")] = None,
    ) -> dict:
        """Speak texts with a cloned voice using Qwen3-TTS Base (10 languages, best quality for these).
        Outputs speech_<NN>_<seed>.wav in the order of texts. For other languages such as Thai use
        speak_multilingual. Returns a job_id; fetch with fetch_result."""
        params = {"texts": texts, "reference_audio": reference_audio, "reference_text": reference_text,
                  "language": language, "seed": seed}
        return submitted(adapter.client.submit("qwen-tts", params))

    @mcp.tool()
    def speak_multilingual(
        texts: Annotated[list[str], Field(description=(
            "Texts to speak, one wav per text (max 64); outside en/zh/ja write numbers as words"))],
        language: Annotated[str, Field(description="de, en, es, fr, id, it, ja, ko, ms, pt, ru, th, vi or zh")],
        reference_audio: Annotated[str, Field(description=(
            "Local path of a recording of the voice (3-20 s), in any language, e.g. a German voice for Thai text"))],
        seed: Annotated[int | None, Field(description="Seed for reproducible results")] = None,
    ) -> dict:
        """Speak texts with a cloned voice using Confucius4-TTS, also across languages: the reference may speak
        another language than the texts. Outputs speech_<NN>_<seed>.wav in the order of texts. Returns a
        job_id; fetch with fetch_result."""
        params = {"texts": texts, "language": language, "reference_audio": reference_audio, "seed": seed}
        return submitted(adapter.client.submit("confucius-tts", params))

    @mcp.tool()
    def generate_video(
        prompt: Annotated[str, Field(description=(
            "English prompt in flowing sentences: subject, action, camera movement, lighting, sound and spoken "
            "dialogue in quotes; the video comes with matching audio"))],
        images: Annotated[list[str] | None, Field(description=(
            "Local paths of keyframe images (max 4), spread evenly: one image is the first frame (image to "
            "video), two are first and last frame"))] = None,
        duration_s: Annotated[float | None, Field(description=(
            "Length in seconds (1-20); default: predicted from the prompt, 5 s with several images"))] = None,
        width: Annotated[int | None, Field(description="Multiple of 64, 256-2048; default 1536x1024 or the "
                                                       "first image's aspect ratio")] = None,
        height: Annotated[int | None, Field(description="Multiple of 64, 256-2048")] = None,
        fps: Annotated[int, Field(description="Frame rate, 12-50")] = 24,
        count: Annotated[int, Field(description="Number of variants in this job (1-8)")] = 1,
        seed: Annotated[int | None, Field(description="Start seed; variants use seed..seed+count-1")] = None,
        enhance_prompt: Annotated[bool, Field(description="Expand a short prompt with the text encoder first")] = False,
    ) -> dict:
        """Create a video with sound from a text prompt with LTX-2.5, optionally starting from an image or
        interpolating between keyframe images. Each variant is a video_<seed>.mp4 plus video_<seed>.jpg with four
        frames to look at. Takes minutes; returns a job_id, fetch with fetch_result."""
        params = {"prompt": prompt, "images": images or [], "duration_s": duration_s, "width": width,
                  "height": height, "fps": fps, "count": count, "seed": seed, "enhance_prompt": enhance_prompt}
        return submitted(adapter.client.submit("ltx-video", params))

    @mcp.tool()
    def audio_to_video(
        prompt: Annotated[str, Field(description="English prompt describing the scene, people and camera")],
        audio: Annotated[str, Field(description="Local path of the speech, song or sound to make the video for")],
        image: Annotated[str | None, Field(description="Local path of an image to use as the first frame")] = None,
        duration_s: Annotated[float | None, Field(description="Seconds; default: the audio, max 20")] = None,
        width: Annotated[int | None, Field(description="Multiple of 64; default 1536x1024 or as the image")] = None,
        height: Annotated[int | None, Field(description="Multiple of 64")] = None,
        count: Annotated[int, Field(description="Number of variants in this job (1-8)")] = 1,
        seed: Annotated[int | None, Field(description="Start seed; variants use seed..seed+count-1")] = None,
    ) -> dict:
        """Create a video for a given audio track with LTX-2.5, e.g. a talking or singing person in sync with
        the audio; the audio is kept unchanged. Outputs video_<seed>.mp4 and .jpg. Returns a job_id."""
        params = {"prompt": prompt, "audio": audio, "images": [image] if image else [], "duration_s": duration_s,
                  "width": width, "height": height, "count": count, "seed": seed}
        return submitted(adapter.client.submit("ltx-video", params))

    @mcp.tool()
    def retake_video(
        video: Annotated[str, Field(description="Local path of the video to edit (e.g. a generated one)")],
        prompt: Annotated[str, Field(description=(
            "English prompt describing the whole scene as it should be in the time range (setting, people, the new "
            "action and sound), not only the change; a prompt about the change alone barely alters the video"))],
        start_s: Annotated[float, Field(description="Start of the time range to regenerate, in seconds")],
        end_s: Annotated[float, Field(description="End of the time range to regenerate, in seconds")],
        regenerate: Annotated[str, Field(description="both, video (keeps audio) or audio (keeps picture)")] = "both",
        count: Annotated[int, Field(description="Number of variants in this job (1-8)")] = 1,
        seed: Annotated[int | None, Field(description="Start seed; variants use seed..seed+count-1")] = None,
    ) -> dict:
        """Regenerate a time range of a video with LTX-2.5 while the rest stays unchanged: fix a bad segment,
        change what happens or what is said. Videos are cut to 8k+1 frames (max 20 s) and sides to multiples
        of 32. Outputs video_<seed>.mp4 and .jpg. Returns a job_id."""
        params = {"video": video, "prompt": prompt, "start_s": start_s, "end_s": end_s, "regenerate": regenerate,
                  "count": count, "seed": seed}
        return submitted(adapter.client.submit("ltx-video", params))

    @mcp.tool()
    def image_to_3d(
        image: Annotated[str, Field(description=(
            "Local path of an image of one object, ideally on a plain background; generate one first if needed"))],
        remove_background: Annotated[bool, Field(description="Cut the object out first; False only for an image "
                                                             "with a clean black or transparent background")] = True,
        resolution: Annotated[int, Field(description="Shape detail: 1536 (default) or 1024 (about twice as fast)")] = 1536,
        max_faces: Annotated[int, Field(description="Faces of the mesh (10000-2000000); lower for game assets")] = 700_000,
        texture_size: Annotated[int, Field(description="PBR texture size: 1024, 2048 or 4096")] = 4096,
        count: Annotated[int, Field(description="Number of variants in this job (1-4)")] = 1,
        seed: Annotated[int | None, Field(description="Start seed; variants use seed..seed+count-1")] = None,
    ) -> dict:
        """Create a textured 3D model from one image with Pixal3D (TRELLIS.2 based): the shape follows the image,
        in the photo's camera perspective. Outputs model_<seed>.glb (PBR: base color, metallic, roughness, normal,
        occlusion) and model_<seed>.png, a render of it. Takes 3-7 minutes. Returns a job_id."""
        params = {"image": image, "remove_background": remove_background, "resolution": resolution,
                  "max_faces": max_faces, "texture_size": texture_size, "count": count, "seed": seed}
        return submitted(adapter.client.submit("pixal3d", params))

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

