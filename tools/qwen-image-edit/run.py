"""Qwen-Image-Edit-2511 worker: edit one to three input images with a text instruction.

Weights: $MODELS_DIR/qwen-image-edit/Qwen-Image-Edit-2511 and the 4-step Lightning LoRA in
$MODELS_DIR/qwen-image-edit/Qwen-Image-Edit-2511-Lightning (download.sh). The LoRA is fused into the
transformer at load time; settings follow the Qwen-Image-Lightning reference script: 4 steps, CFG 1.0 and a
scheduler with shift 3. Without width/height the output keeps the aspect ratio of the first image at about one
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

ROOT = Path(os.environ.get("MODELS_DIR", "models")) / "qwen-image-edit"
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


def load():
    import torch
    from diffusers import FlowMatchEulerDiscreteScheduler, QwenImageEditPlusPipeline

    for path in (MODEL / "model_index.json", LORA):
        if not path.exists():
            raise FileNotFoundError(f"{path} missing, download it on the GB10 system: "
                                    "scripts/docker.sh download qwen-image-edit")
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        MODEL, scheduler=FlowMatchEulerDiscreteScheduler.from_config(SCHEDULER), torch_dtype=torch.bfloat16)
    pipe.load_lora_weights(LORA.parent, weight_name=LORA.name)
    pipe.fuse_lora()
    pipe.unload_lora_weights()
    return pipe.to("cuda")


def run(pipe, params, job_dir):
    import torch
    from PIL import Image

    images = [Image.open(job_dir / "in" / name).convert("RGB") for name in params["images"]]
    kw = {k: params[k] for k in ("width", "height") if params.get(k)}
    paths = []
    for seed in seeds(params):
        t = time.time()
        image = pipe(image=images, prompt=params["prompt"], negative_prompt=" ", true_cfg_scale=CFG,
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
