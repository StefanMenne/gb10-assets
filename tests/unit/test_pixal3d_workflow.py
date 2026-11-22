"""tools/pixal3d: workflow.json (Comfy-Org's template) in API format, built without ComfyUI."""
import importlib.util
import json
import sys

import pytest

from gb10_assets.server.config import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "tools"))
from _comfyui import api_format, prune  # noqa: E402

TEMPLATE = json.loads((REPO_ROOT / "tools" / "pixal3d" / "workflow.json").read_text())


@pytest.fixture(scope="module")
def run_py():
    spec = importlib.util.spec_from_file_location("pixal3d_run", REPO_ROOT / "tools" / "pixal3d" / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_api_format_widgets_and_links():
    wf = api_format(TEMPLATE, ["322"])
    # the "control after generate" value after seed is no input
    assert wf["3"]["inputs"] == {"seed": 56, "steps": 12, "cfg": 7.5, "sampler_name": "euler", "scheduler": "normal",
                                 "denoise": 1, "model": ["108", 0], "positive": ["314", 0], "negative": ["315", 0],
                                 "latent_image": ["87", 0]}
    assert wf["288"]["inputs"] == {"value": 4096}
    # a linked widget takes the link; sub-widgets of dynamic combos keep their dotted names
    assert wf["196"]["inputs"]["resolution"] == ["288", 0]
    assert wf["241"]["inputs"]["sign_mode.fill"] == 20 and wf["241"]["inputs"]["resolution"] == 768
    # notes and the previews nothing depends on are gone, every link resolves
    types = {n["class_type"] for n in wf.values()}
    assert not types & {"Note", "MarkdownNote", "Preview3DAdvanced", "RenderUVAtlas", "PaintMesh"}
    for node in wf.values():
        for value in node["inputs"].values():
            if isinstance(value, list):
                assert value[0] in wf


def test_api_format_rejects_bypassed_nodes():
    ui = json.loads(json.dumps(TEMPLATE))
    next(n for n in ui["nodes"] if n["id"] == 241)["mode"] = 4
    with pytest.raises(ValueError, match="bypassed"):
        api_format(ui, ["322"])


def test_prune():
    api = {"1": {"inputs": {}}, "2": {"inputs": {"a": ["1", 0]}}, "3": {"inputs": {"a": ["1", 0]}}}
    assert set(prune(api, ["2"])) == {"1", "2"}


def test_template_defaults(run_py):
    wf = run_py.workflow("in.png", 42, {})
    template = api_format(TEMPLATE, ["322"])
    # seed 42 is the template's own run; only the image and the file name differ
    for nid, node in template.items():
        changed = {k for k, v in node["inputs"].items() if wf[nid]["inputs"][k] != v}
        assert changed == {"122": {"image"}, "322": {"filename_prefix"}}.get(nid, set()), nid
    assert wf["122"]["inputs"]["image"] == "in.png"
    assert {wf[n]["class_type"] for n in ("322", "401", "402")} == {"Save3DAdvanced", "RenderMesh", "SaveImage"}
    assert wf["401"]["inputs"]["mesh"] == ["260", 0]


def test_request_inputs(run_py):
    wf = run_py.workflow("in.png", 7, {"remove_background": False, "model": "trellis2", "resolution": 2048,
                                       "max_faces": 50_000, "texture_size": 2048, "texture_guidance": 2.5})
    assert [wf[str(n)]["inputs"]["seed"] for n in run_py.SAMPLERS] == [21, 7, 7, 8]
    assert wf["248"]["inputs"]["switch"] is False and wf["316"]["inputs"]["value"] is True
    assert wf["94"]["inputs"]["target_resolution"] == 2048 and wf["186"]["inputs"]["target_face_count"] == 50_000
    assert wf["288"]["inputs"]["value"] == 2048 and wf["12"]["inputs"]["cfg"] == 2.5


def test_files_are_the_templates_models(run_py):
    loaders = {"VAELoader": "vae_name", "CLIPVisionLoader": "clip_name", "UNETLoader": "unet_name",
               "LoadMoGeModel": "model_name", "LoadBackgroundRemovalModel": "bg_removal_name"}
    used = {n["inputs"][loaders[n["class_type"]]] for n in api_format(TEMPLATE, ["322"]).values()
            if n["class_type"] in loaders}
    files = {f for names in run_py.FILES.values() for f in ([names] if isinstance(names, str) else names)}
    assert used == files
