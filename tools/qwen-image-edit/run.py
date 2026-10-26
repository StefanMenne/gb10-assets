"""Qwen-Image-Edit-2511 worker: edit one to three input images with a text instruction.

Weights: $WEIGHTS_DIR/Qwen-Image-Edit-2511 and the 4-step Lightning LoRA in
$WEIGHTS_DIR/Qwen-Image-Edit-2511-Lightning (download.sh). The LoRA is fused into the
transformer at load time; settings follow the Qwen-Image-Lightning reference script: 4 steps, CFG 1.0 and a
scheduler with shift 3 (CFG 1.0: no negative prompt). Without width/height the output keeps the aspect ratio of the first image at about one
megapixel.
Outputs: image_<seed>.png per variant, plus sheet.png when count > 1.
"""
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _worker import contact_sheet, log, seeds, serve  # noqa: E402

ROOT = Path(os.environ.get("WEIGHTS_DIR", "weights"))
MODEL = ROOT / "Qwen-Image-Edit-2511"
LORA = ROOT / "Qwen-Image-Edit-2511-Lightning" / "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"
STEPS, CFG = 4, 1.0
# https://github.com/ModelTC/Qwen-Image-Lightning/blob/main/generate_with_diffusers.py
SCHEDULER = {
    "base_image_seq_len": 256,
    "base_shift": math.log(3),  # shift 3, as in the distillation
    "invert_sigmas": False,
    "max_image_seq_len": 8192,
    "max_shift": math.log(3),
    "num_train_timesteps": 1000,
    "shift": 1.0,
    "shift_terminal": None,
    "stochastic_sampling": False,
    "time_shift_type": "exponential",
    "use_beta_sigmas": False,
    "use_dynamic_shifting": True,
    "use_exponential_sigmas": False,
    "use_karras_sigmas": False,
}


def read_text_encoder(path, dtype):
    """Qwen2.5-VL text encoder read without mmap. On GB10, moving mmap-loaded weights to the GPU takes ~105 s for
    these 16.6 GB; reading the files at once and building the model from the state dict takes ~12 s."""
    from safetensors.torch import load
    from transformers import AutoConfig, Qwen2_5_VLForConditionalGeneration

    state = {}
    for f in sorted(path.glob("*.safetensors")):
        state.update(load(f.read_bytes()))
    return Qwen2_5_VLForConditionalGeneration.from_pretrained(
        None, config=AutoConfig.from_pretrained(path), state_dict=state, dtype=dtype)


def load():
    import torch
    from diffusers import FlowMatchEulerDiscreteScheduler, QwenImageEditPlusPipeline

    for path in (MODEL / "model_index.json", LORA):
        if not path.exists():
            raise FileNotFoundError(f"{path} missing, install the tool on the GB10 system: "
                                    "scripts/docker.sh install qwen-image-edit")
    # disable_mmap: the same slowdown for the diffusers components (transformer 41 GB: ~45 s instead of minutes)
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        MODEL, scheduler=FlowMatchEulerDiscreteScheduler.from_config(SCHEDULER),
        text_encoder=read_text_encoder(MODEL / "text_encoder", torch.bfloat16), dtype=torch.bfloat16,
        disable_mmap=True).to("cuda")
    # fuse on the GPU: 1 s instead of 26 s on the CPU
    pipe.load_lora_weights(LORA.parent, weight_name=LORA.name)
    pipe.fuse_lora()
    pipe.unload_lora_weights()
    return pipe


def run(pipe, params, job_dir):
    import torch
    from PIL import Image

    images = [Image.open(job_dir / "in" / name).convert("RGB") for name in params["images"]]
    kw = {k: params[k] for k in ("width", "height") if params.get(k)}
    paths = []
    for seed in seeds(params):
        t = time.time()
        image = pipe(image=images, prompt=params["prompt"], true_cfg_scale=CFG,
                     num_inference_steps=STEPS, generator=torch.Generator("cuda").manual_seed(seed), **kw).images[0]
        path = job_dir / "out" / f"image_{seed}.png"
        image.save(path)
        log(f"{path.name} in {time.time() - t:.1f}s")
        paths.append(path)
    names = [p.name for p in paths]
    if len(paths) > 1:
        names.append(contact_sheet(paths, job_dir / "out" / "sheet.png"))
    return names


if __name__ == "__main__":
    serve(load, run)
