"""LTX-2.5 worker: video with sound from text, keyframe images or an audio track, and retakes of existing videos.

Weights: $WEIGHTS_DIR/LTX-2.5 (download.sh): the 22B distilled transformer in NVFP4, run through the
NVFP4 GEMM of ltx-kernels (post_install.sh), plus the bf16 Gemma 4 text encoder, VAEs, duration head and spatial
upscaler. All modes share one ModelRegistry, so every component is read from disk once and stays in memory
(~50 GB) for the next jobs; the pipelines themselves only build modules around the cached weights.

Modes, chosen by the inputs (see GenerateVideo in gb10_assets/shared/schemas.py):
  generate        DistilledPipeline: half resolution, 2x latent upscale, refine (8 + 3 steps, no CFG). Images
                  are keyframes spread evenly over the clip (first frame replaces, later ones guide).
  audio to video  A2VidPipelineTwoStage on the distilled transformer with the distilled sigmas and no guidance;
                  the audio is kept and the video generated for it (lip sync, music).
  retake          RetakePipeline (distilled): regenerates start_s..end_s of the video and/or its audio. The input
                  is conformed first: 8k+1 frames, sides cropped to multiples of 32.
Outputs: video_<seed>.mp4 (H.264 + AAC) per variant and video_<seed>.jpg with four frames of it, so a client
that cannot play videos can still look at the result.
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _worker import log, seeds, serve  # noqa: E402

ROOT = Path(os.environ.get("WEIGHTS_DIR", "weights")) / "LTX-2.5"
TRANSFORMER = ROOT / "diffusion_models" / "ltx-2.5-22b-distilled-transformer-nvfp4.safetensors"
TEXT_ENCODER = ROOT / "text_encoders" / "gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"
VIDEO_VAE = ROOT / "vae" / "ltx-2.5-video-vae-bf16.safetensors"
AUDIO_VAE = ROOT / "vae" / "ltx-2.5-audio-vae-bf16.safetensors"
DURATION_HEAD = ROOT / "model_patches" / "ltx-2.5-duration-head-bf16.safetensors"
UPSAMPLER = ROOT / "latent_upscale_models" / "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"
DEFAULT_SIZE = (1536, 1024)
KEYFRAMES_DURATION_S = 5.0
MAX_FRAMES = 481  # 20 s at 24 fps


def unified_mem_get_info(device=None):
    """torch.cuda.mem_get_info for GB10: the driver reports MemFree of the unified memory as free, which leaves out
    the page cache (after loading the weights tens of GB), so LTX sized its VAE decode tiles for 0 bytes.
    MemAvailable counts the reclaimable cache, like the server's memory check."""
    info = dict(line.split()[:2] for line in Path("/proc/meminfo").read_text().splitlines())
    return int(info["MemAvailable:"]) * 1024, int(info["MemTotal:"]) * 1024


def load():
    import torch

    torch.cuda.mem_get_info = unified_mem_get_info
    from ltx_core.allocator_trim_strategy import AllocatorTrimStrategy
    from ltx_core.loader.registry import ModelRegistry
    from ltx_kernels.nvfp4 import unavailable_reason
    from ltx_pipelines.a2vid_two_stage import A2VidPipelineTwoStage
    from ltx_pipelines.distilled import DistilledPipeline
    from ltx_pipelines.retake import RetakePipeline
    from ltx_pipelines.utils.model_paths import ModelPaths
    from ltx_pipelines.utils.quantization_factory import QuantizationKind

    for path in (TRANSFORMER, TEXT_ENCODER, VIDEO_VAE, AUDIO_VAE, DURATION_HEAD, UPSAMPLER):
        if not path.exists():
            raise FileNotFoundError(f"{path} missing, install the tool on the GB10 system: "
                                    "scripts/docker.sh install ltx-video")
    if reason := unavailable_reason():
        raise RuntimeError(f"NVFP4 kernels unusable ({reason}); reinstall the tool on the GB10 system: "
                           "scripts/docker.sh install ltx-video")
    paths = ModelPaths.from_split(transformer_path=str(TRANSFORMER), text_encoder_path=str(TEXT_ENCODER),
                                  video_vae_path=str(VIDEO_VAE), audio_vae_path=str(AUDIO_VAE),
                                  duration_head_path=str(DURATION_HEAD))
    # one registry: weights stay on the GPU between stages, pipelines and jobs; DEFER skips the allocator trim
    kw = {"device": torch.device("cuda"), "registry": ModelRegistry(),
          "quantization": QuantizationKind.NVFP4_PREQUANT.to_policy(str(TRANSFORMER)),
          "alloc_trim_strategy": AllocatorTrimStrategy.DEFER}
    return {
        "generate": DistilledPipeline(paths, spatial_upsampler_path=str(UPSAMPLER), loras=(), **kw),
        "audio": A2VidPipelineTwoStage(paths, distilled_lora=(), spatial_upsampler_path=str(UPSAMPLER), loras=(),
                                       **kw),
        "retake": RetakePipeline(paths, loras=(), distilled=True, **kw),
    }


def size_for(params, image):
    """Requested size, else the first image's aspect ratio at about DEFAULT_SIZE pixels, sides multiples of 64."""
    if params.get("width"):
        return params["width"], params["height"]
    if image is None:
        return DEFAULT_SIZE
    from PIL import Image

    w, h = Image.open(image).size
    area = DEFAULT_SIZE[0] * DEFAULT_SIZE[1]
    scale = (area / (w * h)) ** 0.5
    return (max(256, min(2048, round(w * scale / 64) * 64)),
            max(256, min(2048, round(h * scale / 64) * 64)))


def frames_for(seconds, fps):
    """Frame count on the 8k+1 grid of the video VAE."""
    return min(MAX_FRAMES, round(seconds * fps / 8) * 8 + 1)


def conform_video(src, dst):
    """Copy of src that RetakePipeline accepts: 8k+1 frames (at most MAX_FRAMES), sides cropped to multiples of
    32. Returns src itself if it already fits."""
    import torch
    from ltx_core.types import Audio
    from ltx_pipelines.utils.media_io import encode_video, get_videostream_metadata
    from ltx_pipelines.utils.media_io.decode import decode_audio_from_file, decode_video_by_frame

    meta = get_videostream_metadata(str(src))
    frames = min(MAX_FRAMES, (meta.frames - 1) // 8 * 8 + 1)
    h, w = meta.height // 32 * 32, meta.width // 32 * 32
    if frames == meta.frames and (h, w) == (meta.height, meta.width):
        return src
    if frames < 9 or h < 64 or w < 64:
        raise ValueError(f"video too small for a retake: {meta.width}x{meta.height}, {meta.frames} frames")
    top, left = (meta.height - h) // 2, (meta.width - w) // 2
    video = torch.cat([f[:, top:top + h, left:left + w] for f in decode_video_by_frame(str(src), "cpu", 0, frames)])
    audio = decode_audio_from_file(str(src), torch.device("cpu"), 0.0, frames / meta.fps)
    if audio is not None:  # (1, channels, samples) -> stereo (2, samples) for the AAC muxer
        audio = Audio(waveform=audio.waveform[0, :2].expand(2, -1), sampling_rate=audio.sampling_rate)
    encode_video(video=video.float() / 255, fps=round(meta.fps), audio=audio, output_path=str(dst),
                 video_chunks_number=1, crf=12)
    log(f"conformed {src.name}: {meta.width}x{meta.height}, {meta.frames} frames -> {w}x{h}, {frames} frames")
    return dst


def preview(video_path, count=4, size=480):
    """Four frames of the video side by side with their time stamps, as <stem>.jpg."""
    import av
    from PIL import Image, ImageDraw

    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate)
        frames = [f.to_image() for f in container.decode(stream)]
    picks = [round(k * (len(frames) - 1) / (count - 1)) for k in range(count)]
    w, h = frames[0].size
    tw, th = size, round(size * h / w)
    sheet = Image.new("RGB", (tw * count, th + 24), "white")
    draw = ImageDraw.Draw(sheet)
    for i, k in enumerate(picks):
        sheet.paste(frames[k].resize((tw, th)), (tw * i, 24))
        draw.text((tw * i + 6, 6), f"{k / fps:.1f}s", fill="black")
    out = video_path.with_suffix(".jpg")
    sheet.save(out, quality=88)
    return out.name


def run(pipes, params, job_dir):
    import torch
    from ltx_core.components.guiders import MultiModalGuiderParams
    from ltx_core.model.video_vae import get_video_chunks_number
    from ltx_pipelines.utils.constants import DISTILLED_SIGMAS
    from ltx_pipelines.utils.media_io import encode_video, get_videostream_metadata
    from ltx_pipelines.utils.types import ImageConditioningInput

    inp = job_dir / "in"
    images = [str(inp / name) for name in params.get("images", [])]
    prompt, fps, enhance = params["prompt"], params.get("fps", 24), params.get("enhance_prompt", False)
    duration = params.get("duration_s")

    if params.get("video"):
        video = conform_video(inp / params["video"], job_dir / "conformed.mp4")
        fps = round(get_videostream_metadata(str(video)).fps)
        regenerate = params.get("regenerate", "both")

        def generate(seed):
            return pipes["retake"](video_path=str(video), prompt=prompt, start_time=params["start_s"],
                                   end_time=params["end_s"], seed=seed, enhance_prompt=enhance,
                                   regenerate_video=regenerate != "audio", regenerate_audio=regenerate != "video")
    else:
        width, height = size_for(params, images[0] if images else None)
        if params.get("audio"):
            def generate(seed):
                return pipes["audio"](
                    prompt=prompt, negative_prompt="", seed=seed, height=height, width=width,
                    num_frames=None, frame_rate=fps,
                    num_inference_steps=len(DISTILLED_SIGMAS) - 1, video_guider_params=MultiModalGuiderParams(),
                    images=[ImageConditioningInput.at(p, 0, 1.0) for p in images],
                    audio_path=str(inp / params["audio"]), audio_max_duration=duration or MAX_FRAMES / fps,
                    enhance_prompt=enhance, stage_1_sigmas=DISTILLED_SIGMAS)
        else:
            kw = {}
            if len(images) > 1 or duration:
                kw["num_frames"] = frames_for(duration or KEYFRAMES_DURATION_S, fps)
            last = kw.get("num_frames", 1) - 1
            # with a single image only frame 0 is used, so the predicted duration still works
            keyframes = [ImageConditioningInput.at(p, round(i * last / max(1, len(images) - 1)), 1.0)
                         for i, p in enumerate(images)]

            def generate(seed):
                return pipes["generate"](prompt=prompt, seed=seed, height=height, width=width, frame_rate=fps,
                                         images=keyframes, enhance_prompt=enhance, **kw)

    names = []
    for seed in seeds(params):
        t = time.time()
        with torch.inference_mode():
            result = generate(seed)
            path = job_dir / "out" / f"video_{seed}.mp4"
            encode_video(video=result.video, fps=fps, audio=result.audio, output_path=str(path),
                         video_chunks_number=get_video_chunks_number(result.num_frames, result.tiling_config))
        names += [path.name, preview(path)]
        log(f"{path.name}: {result.num_frames} frames ({result.num_frames / fps:.1f}s) in {time.time() - t:.0f}s")
    return names


if __name__ == "__main__":
    serve(load, run)
