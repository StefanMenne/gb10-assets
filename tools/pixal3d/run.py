"""Pixal3D worker: one image to a textured 3D model (GLB), with the image-to-model workflow of ComfyUI.

ComfyUI (GPL-3.0, fetched by post_install.sh into <install>/build/ComfyUI) runs as a separate headless process on
a local port; this worker sends it the workflow per variant over its HTTP API and collects the files. Models stay
loaded in ComfyUI between jobs and are freed when the worker exits (ComfyUI exits with it).
Weights: $WEIGHTS_DIR in ComfyUI's model folder layout (download.sh).

The workflow is Comfy-Org's template "Pixal3D & TRELLIS.2: Image to Model" (workflow_templates,
3d_pixal3d_trellis2_image_to_model.json) on its Pixal3D path, with the same samplers and post-processing:
background removal and crop, MoGe field of view, Pixal3D conditioning, sparse structure (12 steps), shape at
512 (20 steps) and upsampled to `resolution` (12 steps), texture (12 steps), remesh, decimate, UV unwrap and
baking of base color, metallic, roughness, normal and ambient occlusion maps.
Outputs: model_<seed>.glb per variant and model_<seed>.png, a front render of it.
"""
import atexit
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _worker import log, seeds, serve  # noqa: E402

ROOT = Path(os.environ.get("WEIGHTS_DIR", "weights"))
COMFYUI = ROOT.parent / "build" / "ComfyUI"   # post_install.sh
FILES = {
    "clip_vision": "dino_v3_L_naf_fp32.safetensors",
    "diffusion_models": "pixal3d_int8_convrot.safetensors",
    "vae": ("trellis_2_shape_vae_bf16.safetensors", "trellis_2_texture_vae_bf16.safetensors"),
    "background_removal": "birefnet.safetensors",
    "geometry_estimation": "moge_2_vitl_normal_fp16.safetensors",
}
READY_S = 300


class ComfyUI:
    def __init__(self):
        self.base = Path(tempfile.mkdtemp(prefix="gb10-comfyui-"))
        (self.base / "frontend").mkdir()   # headless: no web frontend installed
        (self.base / "custom_nodes").mkdir()   # stays empty, ComfyUI only lists it
        paths = self.base / "extra_model_paths.yaml"
        paths.write_text("gb10:\n  base_path: " + json.dumps(str(ROOT)) + "\n"
                         + "".join(f"  {folder}: {folder}/\n" for folder in FILES))
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            self.port = s.getsockname()[1]
        self.url = f"http://127.0.0.1:{self.port}"
        # stdout too goes to stderr (the job log); same process group, so killing the worker kills ComfyUI
        self.proc = subprocess.Popen(
            [sys.executable, "main.py", "--listen", "127.0.0.1", "--port", str(self.port), "--disable-auto-launch",
             "--base-directory", str(self.base), "--extra-model-paths-config", str(paths),
             "--front-end-root", str(self.base / "frontend")],
            cwd=COMFYUI, stdout=sys.stderr, stderr=sys.stderr)
        atexit.register(self.close)

    def close(self):
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        shutil.rmtree(self.base, ignore_errors=True)

    def call(self, path, body=None):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.url + path, data, {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.load(r)

    def check_alive(self):
        if self.proc.poll() is not None:
            # leave the worker: the server starts a new one (and ComfyUI with it) for the next job
            raise SystemExit(f"ComfyUI exited with code {self.proc.returncode}")

    def wait_ready(self):
        deadline = time.time() + READY_S
        while True:
            self.check_alive()
            try:
                return self.call("/system_stats")
            except (urllib.error.URLError, ConnectionError):
                if time.time() > deadline:
                    raise RuntimeError(f"ComfyUI not ready after {READY_S}s") from None
                time.sleep(1)

    def run(self, prompt):
        """Queue a workflow in API format and wait for it; returns its outputs per node id."""
        try:
            pid = self.call("/prompt", {"prompt": prompt})["prompt_id"]
        except urllib.error.HTTPError as e:   # validation errors come as 400 with details
            raise RuntimeError(f"ComfyUI rejected the workflow: {e.read().decode()[:2000]}") from None
        while True:
            self.check_alive()
            entry = self.call(f"/history/{pid}").get(pid)
            if entry and entry["status"].get("completed") is not None:
                break
            time.sleep(1)
        if entry["status"].get("status_str") != "success":
            errors = [m[1] for m in entry["status"].get("messages", []) if m[0] == "execution_error"]
            detail = "; ".join(f"{e.get('node_type')}: {e.get('exception_message', '').strip()}" for e in errors)
            raise RuntimeError(f"workflow failed: {detail or entry['status']}")
        return entry["outputs"]

    def output_file(self, item):
        return self.base / "output" / item.get("subfolder", "") / item["filename"]


def load():
    missing = [f"{folder}/{name}" for folder, names in FILES.items()
               for name in ([names] if isinstance(names, str) else names) if not (ROOT / folder / name).exists()]
    if missing:
        raise FileNotFoundError(f"{', '.join(missing)} missing in {ROOT}, install the tool on the GB10 system: "
                                "scripts/docker.sh install pixal3d")
    if not (COMFYUI / "main.py").exists():
        raise FileNotFoundError(f"ComfyUI missing in {COMFYUI}: scripts/docker.sh install pixal3d")
    comfy = ComfyUI()
    stats = comfy.wait_ready()
    log(f"ComfyUI {stats['system'].get('comfyui_version')} on port {comfy.port}")
    return comfy


def workflow(image, seed, p):
    """The template's Pixal3D path in API format; node ids are the template's, 400-402 are the outputs here."""
    n = {}

    def add(nid, cls, **inputs):
        n[str(nid)] = {"class_type": cls, "inputs": inputs}

    def ref(nid, slot=0):
        return [str(nid), slot]

    add(122, "LoadImage", image=image)
    if p.get("remove_background", True):
        add(193, "LoadBackgroundRemovalModel", bg_removal_name=FILES["background_removal"])
        add(192, "RemoveBackground", bg_removal_model=ref(193), image=ref(122))
        mask = ref(192)
    else:
        mask = ref(122, 1)   # the image's own alpha channel
    add(312, "ImageCropToMask", images=ref(122), masks=mask, width=1024, height=1024, pad_factor=1.1, grow_mask=0,
        background="#000000")
    add(15, "CLIPVisionLoader", clip_name=FILES["clip_vision"])
    add(55, "LoadMoGeModel", model_name=FILES["geometry_estimation"])
    add(56, "MoGeInference", moge_model=ref(55), image=ref(312), resolution_level=9, fov_x_degrees=0.0,
        batch_size=4, force_projection=True, apply_mask=True, refine_steps=3)
    add(242, "MoGeGeometryToFOV", moge_geometry=ref(56), axis="horizontal", unit="degrees")
    add(298, "Pixal3DConditioning", clip_vision_model=ref(15), image=ref(312), camera_angle_x=ref(242))
    pos, neg = ref(298, 0), ref(298, 1)
    add(319, "UNETLoader", unet_name=FILES["diffusion_models"], weight_dtype="default")
    add(117, "VAELoader", vae_name=FILES["vae"][0])
    add(118, "VAELoader", vae_name=FILES["vae"][1])
    # sparse structure; CFG override and rescale reproduce the original pipeline's guidance interval
    add(199, "CFGOverride", model=ref(319), cfg=1.0, start_percent=0.667, end_percent=1.0)
    add(125, "RescaleCFG", model=ref(199), multiplier=0.7)
    add(108, "ModelSamplingSD3", model=ref(125), shift=5.0)
    add(87, "EmptyTrellis2LatentStructure", batch_size=1)
    add(3, "KSampler", model=ref(108), seed=seed, steps=12, cfg=7.5, sampler_name="euler", scheduler="normal",
        positive=pos, negative=neg, latent_image=ref(87), denoise=1.0)
    add(119, "VaeDecodeStructureTrellis2", samples=ref(3), vae=ref(117), resolution="32")
    # shape at 512, then upsampled
    add(91, "Trellis2ShapeStage", positive=pos, negative=neg, voxel=ref(119))
    add(279, "CFGOverride", model=ref(319), cfg=1.0, start_percent=0.769, end_percent=1.0)
    add(126, "RescaleCFG", model=ref(279), multiplier=0.5)
    add(18, "KSampler", model=ref(126), seed=seed, steps=20, cfg=7.5, sampler_name="euler", scheduler="normal",
        positive=ref(91, 0), negative=ref(91, 1), latent_image=ref(91, 2), denoise=1.0)
    add(94, "Trellis2UpsampleStage", positive=ref(91, 0), negative=ref(91, 1), shape_latent=ref(18), vae=ref(117),
        target_resolution=p.get("resolution", 1536))
    add(23, "KSampler", model=ref(126), seed=seed, steps=12, cfg=7.5, sampler_name="euler", scheduler="simple",
        positive=ref(94, 0), negative=ref(94, 1), latent_image=ref(94, 2), denoise=1.0)
    add(92, "VaeDecodeShapeTrellis", samples=ref(23), vae=ref(117))
    # texture
    add(98, "Trellis2TextureStage", positive=ref(94, 0), negative=ref(94, 1), shape_latent=ref(23))
    add(12, "KSampler", model=ref(319), seed=seed, steps=12, cfg=1.0, sampler_name="euler", scheduler="normal",
        positive=ref(98, 0), negative=ref(98, 1), latent_image=ref(98, 2), denoise=1.0)
    add(93, "VaeDecodeTextureTrellis", samples=ref(12), vae=ref(118), shape_subdivides=ref(92, 1))
    # mesh post-processing and baking
    tex = p.get("texture_size", 4096)
    add(241, "RemeshMesh", mesh=ref(92), resolution=768, sign_mode="solid",
        **{"sign_mode.fill": 20.0, "sign_mode.qef": False}, band=1.0, project_back=0.0, fix_poles=False,
        smooth_iters=20, drop_small_components=0.01, precluster_max_verts=20_000_000)
    add(186, "DecimateMesh", mesh=ref(241), target_face_count=p.get("max_faces", 700_000), placement_mode="midpoint")
    add(238, "MeshSmoothNormals", mesh=ref(186), crease_angle=180.0)
    add(196, "UnwrapMesh", mesh=ref(238), segmenter="pec", resolution=tex, padding=1, weld_distance=0.0002)
    add(147, "BakeTextureFromVoxel", mesh=ref(196), voxel_colors=ref(93), texture_size=tex, reference_mesh=ref(92))
    add(224, "BakeNormalMapFromMesh", low_poly=ref(196), high_poly=ref(241), resolution=min(tex, 2048),
        cage_distance=0.05, ignore_backfaces=True)
    add(233, "BakeAmbientOcclusion", low_poly=ref(196), high_poly=ref(241), resolution=min(tex, 1024), samples=64,
        max_distance=0.71, strength=1.0, bias=0.01)
    add(210, "ApplyTextureToMesh", mesh=ref(196), base_color=ref(147, 0), metallic=ref(147, 1),
        roughness=ref(147, 2), occlusion=ref(233), normal_map=ref(224))
    add(260, "MeshSmoothNormals", mesh=ref(210), crease_angle=180.0)
    add(400, "SaveGLB", mesh=ref(260), filename_prefix="model")
    add(401, "RenderMesh", mesh=ref(260), mode="auto", width=1024, height=1024, background="#ffffff")
    add(402, "SaveImage", images=ref(401), filename_prefix="model")
    return n


def run(comfy, params, job_dir):
    inputs = comfy.base / "input"
    inputs.mkdir(exist_ok=True)
    image = f"{job_dir.name}_{params['image']}"
    shutil.copy(job_dir / "in" / params["image"], inputs / image)
    names = []
    try:
        for seed in seeds(params):
            t = time.time()
            outputs = comfy.run(workflow(image, seed, params))
            for node, key, ext in (("400", "3d", "glb"), ("402", "images", "png")):
                name = f"model_{seed}.{ext}"
                shutil.move(comfy.output_file(outputs[node][key][0]), job_dir / "out" / name)
                names.append(name)
            log(f"model_{seed}.glb in {time.time() - t:.0f}s")
    finally:
        (inputs / image).unlink(missing_ok=True)
    return names


if __name__ == "__main__":
    serve(load, run)
