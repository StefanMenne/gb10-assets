"""Qwen-Image-2512 worker: text to image, strong at rendering text (signs, posters, UI) and photorealism.

Weights: $WEIGHTS_DIR/Qwen-Image-2512 and the 4-step Lightning LoRA in $WEIGHTS_DIR/Qwen-Image-2512-Lightning
(download.sh). The LoRA is fused into the transformer at load time; settings follow the Qwen-Image-Lightning
reference script: 4 steps, CFG 1.0 and a scheduler with shift 3 (CFG 1.0: no negative prompt).
Outputs: image_<seed>.png per variant, plus sheet.png when count > 1.
"""
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _worker import contact_sheet, fast_cuda_copies, log, seeds, serve  # noqa: E402

ROOT = Path(os.environ.get("WEIGHTS_DIR", "weights"))
MODEL = ROOT / "Qwen-Image-2512"
LORA = ROOT / "Qwen-Image-2512-Lightning" / "Qwen-Image-2512-Lightning-4steps-V1.0-bf16.safetensors"
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
    from diffusers import FlowMatchEulerDiscreteScheduler, QwenImagePipeline

    for path in (MODEL / "model_index.json", LORA):
        if not path.exists():
            raise FileNotFoundError(f"{path} missing, install the tool on the GB10 system: "
                                    "scripts/docker.sh install qwen-image")
    with fast_cuda_copies():
        pipe = QwenImagePipeline.from_pretrained(
            MODEL, scheduler=FlowMatchEulerDiscreteScheduler.from_config(SCHEDULER), dtype=torch.bfloat16).to("cuda")
    # fuse on the GPU, as in qwen-image-edit
    pipe.load_lora_weights(LORA.parent, weight_name=LORA.name)
    pipe.fuse_lora()
    pipe.unload_lora_weights()
    return pipe


def run(pipe, params, job_dir):
    import torch

    paths = []
    for seed in seeds(params):
        t = time.time()
        image = pipe(prompt=params["prompt"], width=params.get("width", 1328), height=params.get("height", 1328),
                     true_cfg_scale=CFG, num_inference_steps=STEPS,
                     generator=torch.Generator("cuda").manual_seed(seed)).images[0]
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
