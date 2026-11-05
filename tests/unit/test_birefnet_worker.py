"""tools/birefnet: its ComfyUI workflow (built without ComfyUI) and its environment, the same as tools/pixal3d."""
import importlib.util
import re
import tomllib

import pytest

from gb10_assets.server.config import REPO_ROOT

TOOLS = REPO_ROOT / "tools"


@pytest.fixture(scope="module")
def run_py():
    spec = importlib.util.spec_from_file_location("birefnet_run", TOOLS / "birefnet" / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("mask", [True, False])
def test_links_resolve(run_py, mask):
    wf = run_py.workflow("in.png", mask)
    for node in wf.values():
        for value in node["inputs"].values():
            if isinstance(value, list):
                assert value[0] in wf, f"{node['class_type']} links to missing node {value[0]}"
    assert wf["10"]["class_type"] == "SaveImage" and ("12" in wf) == mask
    # the alpha is the inverted foreground mask (JoinImageWithAlpha inverts it once more)
    assert wf["5"]["inputs"]["alpha"] == ["4", 0] and wf["4"]["class_type"] == "InvertMask"


def test_output_stems(run_py):
    assert run_py.output_stems(["a.png", "b.jpg", "a.jpg", "a_2.png"]) == ["a", "b", "a_2", "a_2_2"]


def test_same_environment_as_pixal3d():
    """Same dependencies, lock and ComfyUI revision: the environments share their files in the uv cache."""
    def deps(tool):
        project = tomllib.loads((TOOLS / tool / "pyproject.toml").read_text())
        return project["project"]["dependencies"], project["tool"]

    def lock(tool):
        return (TOOLS / tool / "uv.lock").read_text().replace(f"gb10-tool-{tool}", "TOOL")

    def rev(tool):
        return re.search(r"^COMFYUI_REV=(\w+)", (TOOLS / tool / "post_install.sh").read_text(), re.M).group(1)

    assert deps("birefnet") == deps("pixal3d")
    assert lock("birefnet") == lock("pixal3d")
    assert rev("birefnet") == rev("pixal3d")
