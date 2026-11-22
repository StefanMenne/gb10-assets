"""Retexture worker: paints a new base color texture for a GLB on its own UVs, from a text prompt.

ComfyUI (GPL-3.0, fetched by post_install.sh into <install>/build/ComfyUI) runs headless next to the worker
(tools/_comfyui.py); Blender (the blender tool's environment, bpy) runs as a subprocess for the geometry.

1. blender_views.py renders the model from a ring of cameras (and one from above): per pixel the UV and the
   depth it sees, plus per texel its 3D position and normal (texproj.Geometry).
2. Per view, in order: what earlier views painted is rendered into the view and kept, the rest is inpainted
   by Qwen-Image-2512 (4-step Lightning LoRA) with the depth of the view as control (Qwen-Image-2512 Fun
   ControlNet Union). So the views agree with each other instead of being generated independently.
3. The painted lighting is removed per view (delight: a spherical-harmonics fit of the luminance over the
   known normals), optionally SeedVR2 upscales the view 2x, then its pixels are baked into the texels it sees
   best (texproj.Texture.paint). Texels no view sees are filled from their neighbourhood.
4. glb.py swaps the base color image in the GLB; everything else (mesh, UVs, normal map, rig, animations)
   stays. With reset_pbr (default) the old metallic-roughness map is dropped: it belongs to the old paint.

Weights: $WEIGHTS_DIR in ComfyUI's model folder layout (download.sh).
Outputs per seed: model_<seed>.glb, model_<seed>.png (4 views of the result), views_<seed>.png (the
generated views), texture_<seed>.png (the base color texture).
"""
import io
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _comfyui import start  # noqa: E402
from _worker import log, seeds, serve  # noqa: E402

ROOT = Path(os.environ.get("WEIGHTS_DIR", "weights"))
COMFYUI = ROOT.parent / "build" / "ComfyUI"   # post_install.sh
BLENDER = ROOT.parent.parent / "blender"      # the blender tool's installation
HERE = Path(__file__).resolve().parent
FILES = {
    "diffusion_models": ("qwen_image_2512_bf16.safetensors", "seedvr2_3b_fp16.safetensors"),
    "text_encoders": "qwen_2.5_vl_7b.safetensors",
    "vae": ("qwen_image_vae.safetensors", "seedvr2_ema_vae_fp16.safetensors"),
    "loras": "Qwen-Image-2512-Lightning-4steps-V1.0-bf16.safetensors",
    "controlnet": "Qwen-Image-2512-Fun-Controlnet-Union-2602.safetensors",
}
VIEW_PIXELS = 1152 * 1152
ELEVATION, TOP = 10.0, 65.0
STEPS, SHIFT = 4, 3.0   # the Lightning LoRA's settings, as tools/qwen-image
STYLE = ("{view}. The whole object, evenly lit by soft diffuse studio light, no cast shadows, no highlights, "
         "on a plain neutral grey background.")


def ring(n):
    """Azimuths of n ring views, alternating to both sides of the front so neighbours are painted in turn."""
    step = 360 / n
    out = [0.0]
    k = 1
    while len(out) < n:
        out.append(k * step)
        if len(out) < n:
            out.append(-k * step)
        k += 1
    return [((a + 180) % 360) - 180 for a in out]


def view_name(az, el):
    if el >= 45:
        return "View from above"
    a = abs(az)
    side = "left" if az > 0 else "right"
    if a < 30:
        return "Front view"
    if a < 75:
        return f"Three-quarter view from the front {side}"
    if a < 105:
        return f"Side view from the {side}"
    if a < 150:
        return f"Three-quarter view from the back {side}"
    return "Back view"


def load():
    blender_py = BLENDER / "env" / "bin" / "python"
    if not (BLENDER / ".installed").exists():
        raise FileNotFoundError(f"retexture needs the blender tool: scripts/docker.sh install blender "
                                f"({BLENDER} missing)")
    comfy = start(COMFYUI, ROOT, FILES, "retexture")
    comfy.blender = blender_py
    return comfy


def paint_workflow(image, depth, prompt, seed, size, strength):
    """Qwen-Image-2512 inpainting with the depth ControlNet; image: RGBA, transparent = to be painted."""
    w, h = size
    n = {}

    def add(nid, cls, **inputs):
        n[str(nid)] = {"class_type": cls, "inputs": inputs}

    add(1, "UNETLoader", unet_name=FILES["diffusion_models"][0], weight_dtype="default")
    add(2, "LoraLoaderModelOnly", model=["1", 0], lora_name=FILES["loras"], strength_model=1.0)
    add(3, "ModelSamplingAuraFlow", model=["2", 0], shift=SHIFT)
    add(4, "CLIPLoader", clip_name=FILES["text_encoders"], type="qwen_image", device="default")
    add(5, "VAELoader", vae_name=FILES["vae"][0])
    add(6, "CLIPTextEncode", clip=["4", 0], text=prompt)
    add(7, "ConditioningZeroOut", conditioning=["6", 0])   # CFG 1: the negative is not used
    add(8, "ControlNetLoader", control_net_name=FILES["controlnet"])
    add(9, "LoadImage", image=depth)
    add(10, "ControlNetApplyAdvanced", positive=["6", 0], negative=["7", 0], control_net=["8", 0],
        image=["9", 0], strength=strength, start_percent=0.0, end_percent=1.0, vae=["5", 0])
    add(11, "LoadImage", image=image)
    add(12, "ImageScale", image=["11", 0], upscale_method="lanczos", width=w, height=h, crop="disabled")
    add(13, "VAEEncode", pixels=["12", 0], vae=["5", 0])
    add(14, "SetLatentNoiseMask", samples=["13", 0], mask=["11", 1])
    add(15, "KSampler", model=["3", 0], seed=seed, steps=STEPS, cfg=1.0, sampler_name="euler", scheduler="simple",
        positive=["10", 0], negative=["10", 1], latent_image=["14", 0], denoise=1.0)
    add(16, "VAEDecode", samples=["15", 0], vae=["5", 0])
    add(17, "SaveImage", images=["16", 0], filename_prefix="view")
    return n


def upscale_workflow(image, seed):
    """SeedVR2 3B on an image that is already resized to the target size (Comfy-Org's image upscale template)."""
    tiles = dict(tile_size=512, overlap=128, temporal_size=4096, temporal_overlap=8)
    return {
        "1": {"class_type": "LoadImage", "inputs": {"image": image}},
        "2": {"class_type": "SeedVR2Preprocess", "inputs": {"resized_images": ["1", 0]}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": FILES["vae"][1]}},
        "4": {"class_type": "VAEEncodeTiled", "inputs": {"pixels": ["2", 0], "vae": ["3", 0], **tiles}},
        "5": {"class_type": "UNETLoader", "inputs": {"unet_name": FILES["diffusion_models"][1],
                                                     "weight_dtype": "default"}},
        "6": {"class_type": "SeedVR2Conditioning", "inputs": {"model": ["5", 0], "vae_conditioning": ["4", 0]}},
        "7": {"class_type": "KSampler", "inputs": {"model": ["5", 0], "seed": seed, "steps": 1, "cfg": 1.0,
                                                   "sampler_name": "euler", "scheduler": "simple",
                                                   "positive": ["6", 0], "negative": ["6", 1],
                                                   "latent_image": ["4", 0], "denoise": 1.0}},
        "8": {"class_type": "VAEDecodeTiled", "inputs": {"samples": ["7", 0], "vae": ["3", 0], **tiles}},
        "9": {"class_type": "SeedVR2PostProcessing", "inputs": {"images": ["8", 0], "original_resized_images": ["1", 0],
                                                                "color_correction_method": "lab"}},
        "10": {"class_type": "SaveImage", "inputs": {"images": ["9", 0], "filename_prefix": "up"}},
    }


def blender(comfy, script, *args, cwd):
    t = time.time()
    r = subprocess.run([str(comfy.blender), str(script), *map(str, args)], cwd=cwd, stdout=sys.stderr,
                       stderr=sys.stderr)
    if r.returncode:
        raise RuntimeError(f"{Path(script).name} failed with code {r.returncode}, see the log")
    log(f"{Path(script).name} in {time.time() - t:.0f}s")


def comfy_image(comfy, workflow, node):
    import texproj as tp

    outputs = comfy.run(workflow)
    path = comfy.output_file(outputs[node]["images"][0])
    img = tp.load(path)
    path.unlink()
    return img


def run(comfy, params, job_dir):
    import numpy as np
    from PIL import Image, ImageFilter

    import glb
    import texproj as tp

    work = job_dir / "work"
    work.mkdir(exist_ok=True)
    model = job_dir / "in" / params["model"]
    T = params.get("texture_size", 2048)
    views = [(az, ELEVATION) for az in ring(params.get("views", 6))]
    if params.get("top", True):
        views.append((0.0, TOP))
    blender(comfy, HERE / "blender_views.py", model, work, "--texture", T, "--pixels", VIEW_PIXELS,
            "--views", ";".join(f"{az:g},{el:g}" for az, el in views), cwd=work)
    info = json.loads((work / "views.json").read_text())
    geo = tp.Geometry(np.load(work / "texels.npz"), np.load(work / "views.npz"), info)
    size = (geo.W, geo.H)
    upscale = params.get("upscale", True)
    strength = params.get("control_strength", 0.9)
    grey = np.full(3, 0.5)
    names = []
    tag = job_dir.name
    for seed in seeds(params):
        t0 = time.time()
        tex = tp.Texture(geo)
        sheet = []
        for i, (az, el) in enumerate(views):
            t = time.time()
            img, known, obj, nrm, cos = tex.view_state(i)
            # context for the model: known paint, grey background; transparent = paint here (a few pixels
            # into the known area, so the new paint joins it smoothly)
            todo = obj & ~known
            if todo.sum() < 0.002 * obj.sum():
                log(f"view {i} ({view_name(az, el)}): nothing left to paint")
                continue
            grow = np.asarray(Image.fromarray(tp.to_uint8(todo.astype(float))).filter(ImageFilter.MaxFilter(9))) > 127
            ctx = np.where(obj[..., None], img, grey)
            rgba = np.concatenate([tp.to_uint8(ctx), np.where(grow, 0, 255).astype(np.uint8)[..., None]], 2)
            image, depth = f"{tag}_ctx{i}.png", f"{tag}_depth{i}.png"
            Image.fromarray(rgba, "RGBA").save(comfy.input_dir / image)
            tp.save(tp.depth_control(geo.depth[i]), comfy.input_dir / depth)
            prompt = f"{params['prompt'].strip()}\n\n{STYLE.format(view=view_name(az, el))}"
            try:
                gen = comfy_image(comfy, paint_workflow(image, depth, prompt, seed + 1000 * i, size, strength), "17")
            finally:
                (comfy.input_dir / image).unlink(missing_ok=True)
                (comfy.input_dir / depth).unlink(missing_ok=True)
            if params.get("delight", 1.0) > 0:
                gen, _ = tp.delight(gen, nrm, todo, params.get("delight", 1.0))
            # the ring of known paint the mask reached into: both images show the same surface there
            gen = tp.match_colors(gen, img, grow & known & ~geo.edges(i), todo)
            final = np.where(known[..., None], img, gen)
            final = np.where(obj[..., None], final, grey)
            sheet.append(final)
            if upscale:
                big = f"{tag}_up{i}.png"
                Image.fromarray(tp.to_uint8(final)).resize((2 * geo.W, 2 * geo.H), Image.LANCZOS).save(
                    comfy.input_dir / big)
                try:
                    final = comfy_image(comfy, upscale_workflow(big, seed), "10")
                finally:
                    (comfy.input_dir / big).unlink(missing_ok=True)
            n = tex.paint(i, final)
            log(f"view {i} ({view_name(az, el)}): {todo.sum() / obj.sum() * 100:.0f}% of it new, "
                f"{n} texels in {time.time() - t:.0f}s")
        painted = (tex.weight > 0).mean()
        texture = tex.image()
        png = io.BytesIO()
        Image.fromarray(tp.to_uint8(texture)).save(png, "PNG")
        tiny = io.BytesIO()
        Image.new("RGB", (1, 1), (0, 255, 0)).save(tiny, "PNG")
        out = job_dir / "out"
        glb.retexture(model, out / f"model_{seed}.glb", png.getvalue(), reset_pbr=params.get("reset_pbr", True),
                      tiny_png=tiny.getvalue())
        (out / f"texture_{seed}.png").write_bytes(png.getvalue())
        tp.save(contact(sheet), out / f"views_{seed}.png")
        blender(comfy, HERE.parent / "blender" / "scripts" / "preview.py", out / f"model_{seed}.glb",
                out / f"model_{seed}.png", "--views", 4, "--size", 400, cwd=work)
        names += [f"model_{seed}.glb", f"model_{seed}.png", f"views_{seed}.png", f"texture_{seed}.png"]
        log(f"model_{seed}.glb in {time.time() - t0:.0f}s, {painted * 100:.0f}% of the texels painted by a view")
    shutil.rmtree(work, ignore_errors=True)
    return names


def contact(images, height=512):
    import numpy as np
    from PIL import Image

    tiles = []
    for im in images:
        pil = Image.fromarray((np.clip(im, 0, 1) * 255).astype(np.uint8))
        pil = pil.resize((round(pil.width * height / pil.height), height), Image.LANCZOS)
        tiles.append(np.asarray(pil, float) / 255)
    return np.concatenate(tiles, 1) if tiles else np.zeros((height, height, 3))


if __name__ == "__main__":
    serve(load, run)
