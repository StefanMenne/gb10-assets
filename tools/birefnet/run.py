"""BiRefNet-matting worker: removes the background of images with ComfyUI's native background removal nodes.

ComfyUI (GPL-3.0, fetched by post_install.sh into <install>/build/ComfyUI) runs headless next to the worker
(tools/_comfyui.py), which sends it one workflow per image. BiRefNet sees the image at 1024x1024; its mask is
scaled back to the image's own size, so the cutout keeps the full resolution.
Weights: $WEIGHTS_DIR in ComfyUI's model folder layout (download.sh).
Outputs per input image <stem>.<ext>: <stem>_cutout.png (RGBA, background transparent), with `mask` also
<stem>_mask.png (white = object).
"""
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _comfyui import start  # noqa: E402
from _worker import log, serve  # noqa: E402

ROOT = Path(os.environ.get("WEIGHTS_DIR", "weights"))
COMFYUI = ROOT.parent / "build" / "ComfyUI"   # post_install.sh
FILES = {"background_removal": "model.safetensors"}   # BiRefNet-matting (download.sh)


def load():
    return start(COMFYUI, ROOT, FILES, "birefnet")


def workflow(image, mask):
    """API format; the cutout is saved by node 10, the mask (if wanted) by node 12."""
    n = {
        "1": {"class_type": "LoadImage", "inputs": {"image": image}},
        "2": {"class_type": "LoadBackgroundRemovalModel", "inputs": {"bg_removal_name": FILES["background_removal"]}},
        "3": {"class_type": "RemoveBackground", "inputs": {"bg_removal_model": ["2", 0], "image": ["1", 0]}},
        # RemoveBackground: 1 = object; JoinImageWithAlpha takes ComfyUI's mask convention, 1 = transparent
        "4": {"class_type": "InvertMask", "inputs": {"mask": ["3", 0]}},
        "5": {"class_type": "JoinImageWithAlpha", "inputs": {"image": ["1", 0], "alpha": ["4", 0]}},
        "10": {"class_type": "SaveImage", "inputs": {"images": ["5", 0], "filename_prefix": "cutout"}},
    }
    if mask:
        n["11"] = {"class_type": "MaskToImage", "inputs": {"mask": ["3", 0]}}
        n["12"] = {"class_type": "SaveImage", "inputs": {"images": ["11", 0], "filename_prefix": "mask"}}
    return n


def output_stems(images):
    """<stem> per input; a.png and a.jpg become a and a_2."""
    stems, seen = [], set()
    for name in images:
        stem, k = Path(name).stem, 1
        while (candidate := stem if k == 1 else f"{stem}_{k}") in seen:
            k += 1
        seen.add(candidate)
        stems.append(candidate)
    return stems


def run(comfy, params, job_dir):
    mask = params.get("mask", False)
    names = []
    for name, stem in zip(params["images"], output_stems(params["images"])):
        t = time.time()
        image = f"{job_dir.name}_{name}"
        shutil.copy(job_dir / "in" / name, comfy.input_dir / image)
        try:
            outputs = comfy.run(workflow(image, mask), poll_s=0.1)
        finally:
            (comfy.input_dir / image).unlink(missing_ok=True)
        for node, suffix in (("10", "cutout"), ("12", "mask"))[:2 if mask else 1]:
            out = f"{stem}_{suffix}.png"
            shutil.move(comfy.output_file(outputs[node]["images"][0]), job_dir / "out" / out)
            names.append(out)
        log(f"{name} in {time.time() - t:.1f}s")
    return names


if __name__ == "__main__":
    serve(load, run)
