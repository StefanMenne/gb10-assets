"""Pixal3D worker: one image to a textured 3D model (GLB), with the image-to-model workflow of ComfyUI.

ComfyUI (GPL-3.0, fetched by post_install.sh into <install>/build/ComfyUI) runs headless next to the worker
(tools/_comfyui.py), which sends it the workflow per variant.
Weights: $WEIGHTS_DIR in ComfyUI's model folder layout (download.sh).

The workflow is Comfy-Org's template "Pixal3D & TRELLIS.2: Image to Model" itself (workflow_templates, MIT:
workflow.LICENSE), workflow.json as ComfyUI's editor saves it (every widget as a named input); the worker converts
it to API format and changes only the inputs below, by the template's node ids. Everything else is the template's: background
removal and crop, MoGe field of view, Pixal3D (or TRELLIS.2) conditioning, sparse structure, shape at 512 and
upsampled, texture, remesh, decimate, UV unwrap and baking of base color, metallic, roughness, normal and ambient
occlusion maps, with its samplers, step counts and int8 DiTs. To take over a newer template: open it in ComfyUI,
save it and copy it here (the tests check the node ids used below).
Each sampler keeps its seed offset of the template (56, 42, 42, 43): seed 42 gives the template's result.
Outputs: model_<seed>.glb per variant and model_<seed>.png, a front render of it.
"""
import json
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _comfyui import api_format, prune, start  # noqa: E402
from _worker import log, seeds, serve  # noqa: E402

ROOT = Path(os.environ.get("WEIGHTS_DIR", "weights"))
COMFYUI = ROOT.parent / "build" / "ComfyUI"   # post_install.sh
TEMPLATE = Path(__file__).with_name("workflow.json")
FILES = {   # the models workflow.json loads (download.sh)
    "clip_vision": "dino_v3_L_naf_fp32.safetensors",
    "diffusion_models": ("pixal3d_int8_convrot.safetensors", "trellis_2_int8_convrot.safetensors"),
    "vae": ("trellis_2_shape_vae_bf16.safetensors", "trellis_2_texture_vae_bf16.safetensors"),
    "background_removal": "birefnet.safetensors",
    "geometry_estimation": "moge_2_vitl_normal_fp16.safetensors",
}
SAMPLERS = (3, 18, 23, 12)   # sparse structure, shape at 512, upsampled shape, texture
TEMPLATE_SEED = 42
SAVE, RENDER = "322", "401"   # the template's Save 3D node; the front render added here


def load():
    return start(COMFYUI, ROOT, FILES, "pixal3d")


def workflow(image, seed, p, template=None):
    """workflow.json in API format with the request's inputs."""
    wf = api_format(template or json.loads(TEMPLATE.read_text()), [SAVE])

    def put(nid, **inputs):
        wf[str(nid)]["inputs"].update(inputs)

    put(122, image=image)
    put(248, switch=p.get("remove_background", True))   # off: the image's own alpha channel is the mask
    put(316, value=p.get("model", "pixal3d") == "trellis2")
    for nid in SAMPLERS:
        put(nid, seed=seed + wf[str(nid)]["inputs"]["seed"] - TEMPLATE_SEED)
    if p.get("resolution"):
        put(94, target_resolution=p["resolution"])
    if p.get("max_faces"):
        put(186, target_face_count=p["max_faces"])
    if p.get("texture_size"):
        put(288, value=p["texture_size"])
    if p.get("texture_guidance"):
        put(12, cfg=p["texture_guidance"])
    put(SAVE, filename_prefix="model")
    mesh = wf[wf[SAVE]["inputs"]["model_3d"][0]]["inputs"]["mesh"]   # the textured mesh the template saves
    wf[RENDER] = {"class_type": "RenderMesh",
                  "inputs": {"mesh": mesh, "mode": "auto", "width": 1024, "height": 1024, "background": "#ffffff"}}
    wf["402"] = {"class_type": "SaveImage", "inputs": {"images": [RENDER, 0], "filename_prefix": "model"}}
    return prune(wf, [SAVE, "402"])


def run(comfy, params, job_dir):
    inputs = comfy.input_dir
    image = f"{job_dir.name}_{params['image']}"
    shutil.copy(job_dir / "in" / params["image"], inputs / image)
    names = []
    try:
        for seed in seeds(params):
            t = time.time()
            outputs = comfy.run(workflow(image, seed, params))
            for node, key, ext in ((SAVE, "3d", "glb"), ("402", "images", "png")):
                name = f"model_{seed}.{ext}"
                shutil.move(comfy.output_file(outputs[node][key][0]), job_dir / "out" / name)
                names.append(name)
            log(f"model_{seed}.glb in {time.time() - t:.0f}s")
    finally:
        (inputs / image).unlink(missing_ok=True)
    return names


if __name__ == "__main__":
    serve(load, run)
