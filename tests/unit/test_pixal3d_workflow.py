"""The ComfyUI workflow of tools/pixal3d/run.py, built without ComfyUI: every link points to an existing node."""
import importlib.util

import pytest

from gb10_assets.server.config import REPO_ROOT


@pytest.fixture(scope="module")
def run_py():
    spec = importlib.util.spec_from_file_location("pixal3d_run", REPO_ROOT / "tools" / "pixal3d" / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("remove_background", [True, False])
def test_links_resolve(run_py, remove_background):
    wf = run_py.workflow("in.png", 7, {"remove_background": remove_background, "resolution": 1024,
                                       "max_faces": 50_000, "texture_size": 2048})
    for node in wf.values():
        for value in node["inputs"].values():
            if isinstance(value, list):
                assert value[0] in wf, f"{node['class_type']} links to missing node {value[0]}"
    assert ("RemoveBackground" in {n["class_type"] for n in wf.values()}) == remove_background
    assert wf["94"]["inputs"]["target_resolution"] == 1024 and wf["186"]["inputs"]["target_face_count"] == 50_000
    assert {wf[n]["class_type"] for n in ("400", "402")} == {"SaveGLB", "SaveImage"}
