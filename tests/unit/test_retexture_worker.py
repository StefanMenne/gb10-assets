"""tools/retexture without GPU, ComfyUI or Blender: its workflows, the projection maths of texproj.py on a synthetic
plane, and the texture swap of glb.py on a synthetic GLB."""
import importlib.util
import json
import re
import struct
import sys
import tomllib

import numpy as np
import pytest

from gb10_assets.server.config import REPO_ROOT

TOOLS = REPO_ROOT / "tools"


def _module(name):
    sys.path.insert(0, str(TOOLS / "retexture"))
    sys.path.insert(0, str(TOOLS))
    spec = importlib.util.spec_from_file_location(f"retexture_{name}", TOOLS / "retexture" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def run_py():
    return _module("run")


@pytest.fixture(scope="module")
def tp():
    return _module("texproj")


@pytest.fixture(scope="module")
def glb():
    return _module("glb")


def _links_resolve(wf):
    for node in wf.values():
        for value in node["inputs"].values():
            if isinstance(value, list):
                assert value[0] in wf, f"{node['class_type']} links to missing node {value[0]}"


def test_workflows(run_py):
    wf = run_py.paint_workflow("ctx.png", "depth.png", "a red box", 7, (816, 1296), 0.9)
    _links_resolve(wf)
    assert wf["15"]["inputs"]["steps"] == 4 and wf["15"]["inputs"]["cfg"] == 1.0
    assert wf["10"]["inputs"]["strength"] == 0.9 and wf["10"]["inputs"]["image"] == ["9", 0]
    assert wf["14"]["inputs"]["mask"] == ["11", 1]   # the alpha of the context image: transparent = paint
    assert (wf["12"]["inputs"]["width"], wf["12"]["inputs"]["height"]) == (816, 1296)
    up = run_py.upscale_workflow("big.png", 7)
    _links_resolve(up)
    assert up["5"]["inputs"]["unet_name"] in run_py.FILES["diffusion_models"]
    assert up["3"]["inputs"]["vae_name"] in run_py.FILES["vae"]


def test_ring_alternates_from_the_front(run_py):
    assert run_py.ring(4) == [0.0, 90.0, -90.0, -180.0]
    assert run_py.ring(6) == [0.0, 60.0, -60.0, 120.0, -120.0, -180.0]
    assert len(set(run_py.ring(8))) == 8
    assert run_py.view_name(0, 10) == "Front view" and run_py.view_name(-180, 10) == "Back view"
    assert run_py.view_name(0, 65) == "View from above" and "left" in run_py.view_name(90, 10)


def test_same_environment_as_pixal3d():
    """Same dependencies, lock and ComfyUI revision: the environments share their files in the uv cache."""
    def deps(tool):
        project = tomllib.loads((TOOLS / tool / "pyproject.toml").read_text())
        return project["project"]["dependencies"], project["tool"]

    def lock(tool):
        return (TOOLS / tool / "uv.lock").read_text().replace(f"gb10-tool-{tool}", "TOOL")

    def rev(tool):
        return re.search(r"^COMFYUI_REV=(\w+)", (TOOLS / tool / "post_install.sh").read_text(), re.M).group(1)

    assert deps("retexture") == deps("pixal3d")
    assert lock("retexture") == lock("pixal3d")
    assert rev("retexture") == rev("pixal3d")


def _plane(T=64, W=96, H=96):
    """A unit square in the x-z plane facing -y (Blender front), UV = (x, 1 - z), seen by a camera in front of it;
    the arrays blender_views.py would write."""
    yy, xx = np.mgrid[0:T, 0:T]
    u, v = (xx + 0.5) / T, (yy + 0.5) / T
    pos = np.stack([u - 0.5, np.zeros_like(u), 0.5 - v], -1)
    nrm = np.broadcast_to([0.0, -1.0, 0.0], pos.shape)
    texels = {"pos": pos, "nrm": nrm, "cover": np.ones((T, T), bool)}
    f, dist = 100.0, 2.0
    eye = np.array([0.0, -dist, 0.0])
    R = np.array([[1.0, 0, 0], [0, 0, 1], [0, -1, 0]])   # world -> camera: x right, y up (= z), looks along +y
    py, px = np.mgrid[0:H, 0:W]
    X = (px + 0.5 - W / 2) / f * dist
    Z = -(py + 0.5 - H / 2) / f * dist
    hit = (np.abs(X) < 0.5) & (np.abs(Z) < 0.5)
    uv = np.where(hit[..., None], np.stack([X + 0.5, 0.5 - Z], -1), -1)
    views = {"uv": uv[None], "depth": np.where(hit, dist, 0.0)[None], "R": R[None], "eye": eye[None],
             "f": np.float64(f)}
    info = {"views": [[0, 0]], "bmin": [-0.5, 0, -0.5], "bmax": [0.5, 0, 0.5]}
    return texels, views, info, hit


def test_paint_round_trip(tp):
    """A view rendered from a texture and painted back gives the texture again."""
    texels, views, info, hit = _plane()
    geo = tp.Geometry(texels, views, info)
    yy, xx = np.mgrid[0:geo.T, 0:geo.T] / geo.T   # smooth: the view has fewer pixels than the texture texels
    truth = np.stack([xx, yy, 0.5 + 0.4 * np.sin(6 * xx) * np.cos(4 * yy)], -1)
    src = tp.Texture(geo)
    src.color[:] = truth.reshape(-1, 3)[geo.idx]
    src.weight[:] = 1
    img, known, obj, nrm, cos = src.view_state(0)
    inner = obj & ~geo.edges(0)
    assert (obj == hit).all() and known[obj].all() and (cos[inner] > 0.9).all() and (cos[obj & ~inner] < 0.3).all()
    tex = tp.Texture(geo)
    assert not tex.view_state(0)[1].any()
    painted = tex.paint(0, img)
    assert painted > 0.9 * len(geo.idx)
    p = tex.weight > 0
    assert np.abs(tex.color[p] - src.color[p]).mean() < 0.02
    # painted texels are known now: a second paint changes nothing
    assert tex.paint(0, 1 - img) == 0
    assert tex.view_state(0)[1][obj].all()


def test_occluded_texels_are_not_painted(tp):
    texels, views, info, _ = _plane()
    views["depth"] = views["depth"] * 0.5   # something nearer than the plane covers it
    geo = tp.Geometry(texels, views, info)
    assert tp.Texture(geo).paint(0, np.ones((96, 96, 3))) == 0


def test_push_pull_fills_holes(tp):
    img = np.zeros((16, 16, 3))
    have = np.zeros((16, 16))
    img[:, :8], have[:, :8] = 0.2, 1
    img[:, 12:], have[:, 12:] = 0.8, 1
    out = tp.push_pull(img, have)
    assert np.allclose(out[have > 0], img[have > 0])
    assert (out[:, 8:12] > 0.15).all() and (out[:, 8:12] < 0.85).all()


def test_delight_removes_directional_light(tp):
    rng = np.random.default_rng(1)
    n = rng.normal(size=(4000, 3))
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    albedo = 0.5
    shade = np.exp(0.4 * n[:, 2])   # light from above
    lin = albedo * shade / np.exp(0.4 * n[:, 2]).mean()
    img = tp.linear_to_srgb(np.repeat(lin[:, None], 3, 1))[None]
    out, f = tp.delight(img, n[None], np.ones((1, 4000), bool))
    flat = tp.srgb_to_linear(out[0, :, 0])
    assert flat.std() < 0.1 * lin.std()


def _glb(doc, binary):
    js = json.dumps(doc).encode()
    js += b" " * (-len(js) % 4)
    binary += b"\0" * (-len(binary) % 4)
    return (struct.pack("<III", 0x46546C67, 2, 28 + len(js) + len(binary)) + struct.pack("<II", len(js), 0x4E4F534A)
            + js + struct.pack("<II", len(binary), 0x004E4942) + binary)


def test_glb_swaps_base_color_only(glb, tmp_path):
    positions = struct.pack("<9f", 0, 0, 0, 1, 0, 0, 0, 1, 0)
    color, mr = b"OLDCOLOR" * 5, b"OLDMR" * 7
    binary = positions + color + b"\0" * (-len(color) % 4) + mr
    doc = {
        "asset": {"version": "2.0"},
        "buffers": [{"byteLength": len(binary)}],
        "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": 36},
                        {"buffer": 0, "byteOffset": 36, "byteLength": len(color)},
                        {"buffer": 0, "byteOffset": 36 + len(color) + (-len(color) % 4), "byteLength": len(mr)}],
        "accessors": [{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"}],
        "images": [{"bufferView": 1, "mimeType": "image/jpeg"}, {"bufferView": 2, "mimeType": "image/png"}],
        "textures": [{"source": 0}, {"source": 1}],
        "materials": [{"pbrMetallicRoughness": {"baseColorTexture": {"index": 0},
                                                "metallicRoughnessTexture": {"index": 1}, "metallicFactor": 1}}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "material": 0}]}],
    }
    src, dst = tmp_path / "in.glb", tmp_path / "out.glb"
    src.write_bytes(_glb(doc, binary))
    glb.retexture(src, dst, b"NEWPNG" * 11, tiny_png=b"TINY")
    d, b = glb.read(dst)
    assert glb.image_bytes(d, b, 0) == b"NEWPNG" * 11 and d["images"][0]["mimeType"] == "image/png"
    assert glb.image_bytes(d, b, 1) == b"TINY"   # the dropped metallic-roughness image shrinks
    v = d["bufferViews"][0]
    assert b[v["byteOffset"]:v["byteOffset"] + 36] == positions
    pbr = d["materials"][0]["pbrMetallicRoughness"]
    assert "metallicRoughnessTexture" not in pbr and pbr["metallicFactor"] == 0 and pbr["roughnessFactor"] == 0.8
    assert d["buffers"][0]["byteLength"] == len(b.rstrip(b"\0")) or d["buffers"][0]["byteLength"] <= len(b)
    glb.retexture(src, dst, b"NEW", reset_pbr=False)
    d, b = glb.read(dst)
    assert glb.image_bytes(d, b, 1) == mr and "metallicRoughnessTexture" in d["materials"][0]["pbrMetallicRoughness"]


def test_match_colors_takes_on_the_known_tint(tp):
    rng = np.random.default_rng(2)
    ref = rng.uniform(0.2, 0.8, (40, 40, 3))
    gen = tp.linear_to_srgb(tp.srgb_to_linear(ref) * np.array([1.2, 1.0, 0.85]))   # a warmer, brighter view
    overlap = np.zeros((40, 40), bool)
    overlap[:, :20] = True
    out = tp.match_colors(gen, ref, overlap, ~overlap)
    assert np.abs(out[:, 20:] - ref[:, 20:]).mean() < 0.01 < np.abs(gen[:, 20:] - ref[:, 20:]).mean()
    assert (out[:, :20] == gen[:, :20]).all()
