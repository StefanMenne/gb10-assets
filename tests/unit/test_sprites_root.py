"""plugin/skills/sprite-animation/scripts/sprites.py: root motion from point tracks of a synthetic walker."""
import importlib.util
import json
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from gb10_assets.server.config import REPO_ROOT

SPRITES = REPO_ROOT / "plugin" / "skills" / "sprite-animation" / "scripts" / "sprites.py"
HEIGHT = 600   # figure height in px
CYCLE = 24     # frames per cycle (two steps)
STANCE = 0.6   # share of the cycle a foot stands on the ground


@pytest.fixture(scope="module")
def sprites():
    spec = importlib.util.spec_from_file_location("sprites", SPRITES)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def foot_offset(t, speed, phase):
    """Foot position relative to a walker moving at `speed` (px/frame along x): during the stance it moves
    back with the ground, during the swing it is lifted and moves forward again (cosine profile)."""
    u = ((t / CYCLE) + phase) % 1
    stride = speed * STANCE * CYCLE
    if u < STANCE:
        return stride / 2 - stride * u / STANCE, 0.0
    s = (u - STANCE) / (1 - STANCE)
    return -stride / 2 + stride * (1 - np.cos(np.pi * s)) / 2, -40 * np.sin(np.pi * s)


def walker(speed=8.0, across=False, slide=0.0, frames=73, seed=0):
    """Tracks (points, frames, 3) and kinds: 2 feet x 3 points, 3 body points, 0.3 px tracking noise.
    across: the body travels over a static ground; slide: the standing foot glides by that share of speed."""
    rng = np.random.default_rng(seed)
    tracks, kinds = [], []
    for t0, kind in [(0, "foot"), (0.5, "foot")] * 3 + [(None, "body")] * 3:
        dx, dy = rng.normal(0, 6, 2)
        pts = []
        for t in range(frames):
            travel = speed * t if across else 0.0
            if kind == "foot":
                fx, fy = foot_offset(t, speed, t0)
                fx -= slide * speed * t * (1 if across else 0)
                x, y = 500 + travel + fx + dx, 900 + fy + dy
            else:
                x, y = 500 + travel + dx, 600 + 4 * np.sin(4 * np.pi * t / CYCLE) + dy
            pts.append([x + rng.normal(0, 0.3), y + rng.normal(0, 0.3), float(rng.random() > 0.05)])
        tracks.append(pts)
        kinds.append(kind)
    return np.array(tracks), kinds


@pytest.mark.parametrize("across", [False, True])
def test_root_motion_in_place_and_across(sprites, across):
    tracks, kinds = walker(across=across)
    r = sprites.root_motion(tracks, kinds, HEIGHT)
    assert r["velocity"] == pytest.approx([8.0, 0.0], abs=0.4)
    assert r["walks_across"] == across
    assert r["coverage"] > 0.9 and r["unsteady"] < 0.3
    per_frame = np.array(r["per_frame"])
    assert per_frame.shape == (72, 2) and np.median(per_frame[:, 0]) == pytest.approx(8.0, abs=0.4)


def test_marching_on_the_spot_has_no_root_motion(sprites):
    tracks, kinds = walker(speed=0.0)   # the feet are still lifted in turn, but put down at the same place
    assert np.ptp(tracks[0, :, 1]) > 30
    assert sprites.root_motion(tracks, kinds, HEIGHT)["speed"] < 0.008 * HEIGHT


def test_sliding_foot_is_measured(sprites):
    tracks, kinds = walker(across=True, slide=0.3)
    r = sprites.root_motion(tracks, kinds, HEIGHT)
    assert np.linalg.norm(r["ground_velocity"]) == pytest.approx(0.3 * 8, abs=0.5)


def test_root_and_export_files(sprites, tmp_path):
    tracks, kinds = walker()
    frames = [f"frame_{t:04d}.png" for t in range(tracks.shape[1])]
    (tmp_path / "q.json").write_text(json.dumps({"frames": frames, "kinds": kinds, "figure_height": HEIGHT,
                                                 "queries": [[0, 1, 1]] * len(kinds), "size": [1024, 1024]}))
    (tmp_path / "t.json").write_text(json.dumps({"frames": frames, "tracks": tracks.tolist()}))
    sprites.cmd_root(SimpleNamespace(tracks=tmp_path / "t.json", queries=tmp_path / "q.json",
                                     out=tmp_path / "root.json", fps=24))
    root = json.loads((tmp_path / "root.json").read_text())
    assert root["warnings"] == [] and not root["walks_across"]

    # a cycle unpacked with --range 24-47 --step 3, scaled by 0.25, played at 8 fps
    files = [tmp_path / f"frame_{t:04d}.png" for t in range(24, 48, 3)]
    moves = sprites.root_per_sprite_frame(root, files, 0.25)
    assert moves.shape == (8, 2)
    assert moves[:, 0].mean() == pytest.approx(8 * 3 * 0.25, abs=0.2)   # px per sprite frame


def test_stabilize_removes_travel(sprites, tmp_path):
    """A disc moving 5 px per frame to the right stands still after stabilize."""
    names = [f"frame_{t:04d}.png" for t in range(9)]
    (tmp_path / "in").mkdir()
    for t, name in enumerate(names):
        yy, xx = np.mgrid[:128, :256]
        alpha = ((xx - 60 - 5 * t) ** 2 + (yy - 64) ** 2 < 30 ** 2).astype(np.uint8) * 255
        Image.fromarray(np.dstack([alpha, alpha * 0, alpha * 0, alpha])).save(tmp_path / "in" / name)
    (tmp_path / "root.json").write_text(json.dumps({"frames": names, "body_velocity": [5.0, 0.0]}))
    sprites.cmd_stabilize(SimpleNamespace(inputs=[tmp_path / "in"], root=tmp_path / "root.json",
                                          out=tmp_path / "out", max_scale=0.1))
    centers = []
    for name in names:
        a = np.asarray(Image.open(tmp_path / "out" / name))[..., 3].astype(float)
        centers.append((a * np.arange(256)).sum() / a.sum())
    assert max(centers) - min(centers) < 0.5
    assert centers[4] == pytest.approx(80, abs=0.5)   # the middle frame stays where it was


def test_foot_root_with_standing_phase(sprites, tmp_path):
    """Standing 20 frames, then walking in place: root --foot gives 0 while standing and the cycle's speed after."""
    walk, kinds = walker(frames=73)
    stand = np.repeat(walk[:, :1], 20, axis=1)
    stand[..., :2] += np.random.default_rng(1).normal(0, 0.3, stand[..., :2].shape)
    stand[..., 2] = 1.0
    tracks = np.concatenate([stand, walk], axis=1)
    n = tracks.shape[1]
    frames = [f"frame_{t:04d}.png" for t in range(n)]
    queries = [[t, 1, 1] for t in range(0, n, 2) for _ in kinds]
    all_kinds = [k for _ in range(0, n, 2) for k in kinds]
    tracks = np.concatenate([tracks] * len(range(0, n, 2)))   # every query frame tracks the same points again
    (tmp_path / "q.json").write_text(json.dumps({"frames": frames, "kinds": all_kinds, "figure_height": HEIGHT,
                                                 "queries": queries, "size": [1024, 1024]}))
    (tmp_path / "t.json").write_text(json.dumps({"frames": frames, "tracks": tracks.tolist()}))
    # the cycle: only the walking frames
    cyc = tracks[:, 20:]
    keep = [i for i, q in enumerate(queries) if q[0] >= 20]
    (tmp_path / "qc.json").write_text(json.dumps({
        "frames": frames[20:], "kinds": [all_kinds[i] for i in keep], "figure_height": HEIGHT,
        "queries": [[queries[i][0] - 20, 1, 1] for i in keep], "size": [1024, 1024]}))
    (tmp_path / "tc.json").write_text(json.dumps({"frames": frames[20:], "tracks": cyc[keep].tolist()}))
    sprites.cmd_root(SimpleNamespace(tracks=tmp_path / "tc.json", queries=tmp_path / "qc.json",
                                     out=tmp_path / "cycle.json", fps=24, foot=False))
    cycle = json.loads((tmp_path / "cycle.json").read_text())
    assert cycle["speed"] == pytest.approx(8.0, abs=0.4) and cycle["foot_speed"] > 0

    sprites.cmd_root(SimpleNamespace(tracks=tmp_path / "t.json", queries=tmp_path / "q.json",
                                     out=tmp_path / "seg.json", fps=24, foot=True, like=tmp_path / "cycle.json"))
    seg = json.loads((tmp_path / "seg.json").read_text())
    per_frame = np.array(seg["per_frame"])
    assert per_frame.shape == (n - 1, 2)
    assert np.abs(per_frame[:16]).max() < 0.15 * 8           # standing: only tracking noise
    assert np.median(per_frame[30:, 0]) == pytest.approx(8.0, abs=0.8)  # walking, scaled to the cycle


def test_place_feet_and_scale_box(sprites, tmp_path):
    """place puts the feet on the given line; scale --box gives the same anchor for differently cropped figures."""
    cut = np.zeros((300, 200, 4), np.uint8)
    cut[50:250, 80:120] = (200, 30, 30, 255)
    Image.fromarray(cut).save(tmp_path / "cut.png")
    sprites.cmd_place(SimpleNamespace(cutout=tmp_path / "cut.png", out=tmp_path / "ref.png", color="00FF00",
                                      size="400x600", at="center", height=400.0, bottom=0.06, feet=520))
    ref = np.asarray(Image.open(tmp_path / "ref.png"))
    rows = np.nonzero((ref[..., 0] > 100).any(1))[0]
    assert (rows.min(), rows.max()) == (120, 519)

    for name, (y0, y1) in {"a": (100, 300), "b": (200, 300)}.items():   # tall and short figure, same feet
        frame = np.zeros((400, 300, 4), np.uint8)
        frame[y0:y1, 140:160] = 255
        (tmp_path / name).mkdir()
        Image.fromarray(frame).save(tmp_path / name / "frame_0000.png")
        sprites.cmd_scale(SimpleNamespace(inputs=[tmp_path / name], out=tmp_path / f"{name}_s", factor=0.5,
                                          height=None, pad=4, box="-20,0,320,400", anchor=None))
        out = np.asarray(Image.open(tmp_path / f"{name}_s" / "frame_0000.png"))
        assert out.shape[:2] == (200, 170)
        assert np.nonzero((out[..., 3] > 128).any(1))[0].max() == 149
