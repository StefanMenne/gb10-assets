"""tools/tapnet: coordinate conversion of its worker (without torch) and the client side of a tracking job."""
import importlib.util
import json

import numpy as np
import pytest

from gb10_assets.client.http_client import ApiError, track_params
from gb10_assets.server.config import REPO_ROOT


@pytest.fixture(scope="module")
def run_py():
    spec = importlib.util.spec_from_file_location("tapnet_run", REPO_ROOT / "tools" / "tapnet" / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_model_coordinates_round_trip(run_py):
    # [frame, x, y] in a 1536x640 frame -> [t, y, x] at 512x512 and back as x, y
    assert run_py.to_model([[3, 768, 320]], (1536, 640), 512) == [[3, 256, 256]]
    tracks = np.array([[[256.0, 256.0], [512.0, 0.0]]])
    assert run_py.from_model(tracks, (1536, 640), 512).tolist() == [[[768, 320], [1536, 0]]]


def test_track_params(tmp_path):
    (tmp_path / "frames").mkdir()
    for name in ("frame_0002.png", "frame_0001.png", "notes.txt"):
        (tmp_path / "frames" / name).write_bytes(b"")
    (tmp_path / "q.json").write_text(json.dumps({"kinds": ["foot"], "queries": [[0, 1.5, 2.5]]}))
    p = track_params([str(tmp_path / "frames")], str(tmp_path / "q.json"), 256)
    assert [n.rsplit("/", 1)[1] for n in p["frames"]] == ["frame_0001.png", "frame_0002.png"]
    assert p["queries"] == [[0, 1.5, 2.5]] and p["resolution"] == 256
    (tmp_path / "list.json").write_text("[[1, 2, 3]]")
    assert track_params([str(tmp_path / "frames")], str(tmp_path / "list.json"))["queries"] == [[1, 2, 3]]
    with pytest.raises(ApiError):
        track_params([], str(tmp_path / "missing.json"))
