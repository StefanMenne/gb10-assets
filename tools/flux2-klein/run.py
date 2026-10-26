"""FLUX.2 [klein] 4B worker: text to image, or reference image(s) + prompt to image.

Weights: $WEIGHTS_DIR/FLUX.2-klein-4B (download.sh). The model is distilled to 4 steps at
guidance 1.0; more steps or higher guidance do not improve it.
Outputs: image_<seed>.png per variant, plus sheet.png when count > 1.
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _worker import contact_sheet, log, seeds, serve  # noqa: E402

MODEL = Path(os.environ.get("WEIGHTS_DIR", "weights")) / "FLUX.2-klein-4B"
STEPS, GUIDANCE = 4, 1.0


def load():
    import torch
    from diffusers import Flux2KleinPipeline

    if not (MODEL / "model_index.json").exists():
        raise FileNotFoundError(f"{MODEL} missing, install the tool on the GB10 system: "
                                "scripts/docker.sh install flux2-klein")
    return Flux2KleinPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to("cuda")


def run(pipe, params, job_dir):
    import torch
    from PIL import Image

    kw = {}
    refs = [Image.open(job_dir / "in" / name).convert("RGB") for name in params.get("reference_images", [])]
    if refs:
        kw["image"] = refs if len(refs) > 1 else refs[0]
    paths = []
    for seed in seeds(params):
        t = time.time()
        image = pipe(prompt=params["prompt"], width=params.get("width", 1024), height=params.get("height", 1024),
                     num_inference_steps=STEPS, guidance_scale=GUIDANCE,
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
