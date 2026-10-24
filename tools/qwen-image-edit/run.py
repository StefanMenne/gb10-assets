"""Qwen-Image-Edit-2511 worker: edit one to three input images with a text instruction.

Weights: $MODELS_DIR/qwen-image-edit/Qwen-Image-Edit-2511 (download.sh). Settings follow the model card: 40
steps, true CFG 4.0 with an empty negative prompt. Without width/height the output keeps the aspect ratio of
the first image at about one megapixel.
Outputs: image_<seed>.png per variant, plus sheet.png when count > 1.
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _worker import contact_sheet, log, seeds, serve  # noqa: E402

MODEL = Path(os.environ.get("MODELS_DIR", "models")) / "qwen-image-edit" / "Qwen-Image-Edit-2511"


def load():
    import torch
    from diffusers import QwenImageEditPlusPipeline

    if not (MODEL / "model_index.json").exists():
        raise FileNotFoundError(f"{MODEL} missing, download it on the GB10 system: "
                                "scripts/docker.sh download qwen-image-edit")
    return QwenImageEditPlusPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to("cuda")


def run(pipe, params, job_dir):
    import torch
    from PIL import Image

    images = [Image.open(job_dir / "in" / name).convert("RGB") for name in params["images"]]
    kw = {k: params[k] for k in ("width", "height") if params.get(k)}
    paths = []
    for seed in seeds(params):
        t = time.time()
        image = pipe(image=images, prompt=params["prompt"], negative_prompt=params.get("negative_prompt") or " ",
                     true_cfg_scale=params.get("true_cfg_scale", 4.0), guidance_scale=1.0,
                     num_inference_steps=params.get("steps", 40),
                     generator=torch.Generator("cuda").manual_seed(seed), **kw).images[0]
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
