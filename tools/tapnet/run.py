"""TAPNet worker: tracks query points through a sequence of frames with BootsTAPIR (offline TAPIR, PyTorch).

Code: tapnet_torch/ is tapnet/torch from https://github.com/google-deepmind/tapnet (Apache-2.0, see
tapnet_torch/LICENSE), vendored because the tapnet package depends on JAX; only its imports are changed.
Weights: $WEIGHTS_DIR/bootstapir_checkpoint_v2.pt (download.sh).

The frames are resized to resolution x resolution (the model sees the whole sequence at once, so a point may be
queried in any frame and is tracked forwards and backwards). Coordinates are pixels of the original frames with
(0, 0) the top-left corner of the top-left pixel, so the centre of pixel (i, j) is (i + 0.5, j + 0.5).
Outputs: tracks.json {"size": [w, h], "frames": [names], "queries": [[frame, x, y], ...],
"tracks": per query per frame [x, y, visibility 0..1]} and tracks.png, the tracks drawn on a few frames.
"""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _worker import log, serve  # noqa: E402

ROOT = Path(os.environ.get("WEIGHTS_DIR", "weights"))
CHECKPOINT = ROOT / "bootstapir_checkpoint_v2.pt"
CHUNK = 256  # query points per forward pass: the cost volumes grow with points x frames


def load():
    import torch
    from tapnet_torch import tapir_model

    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"{CHECKPOINT} missing, install the tool on the GB10 system: "
                                "scripts/docker.sh install tapnet")
    model = tapir_model.TAPIR(pyramid_level=1)
    model.load_state_dict(torch.load(CHECKPOINT, map_location="cpu", weights_only=True))
    torch.set_grad_enabled(False)
    return model.to("cuda").eval()


def read_frames(paths):
    """RGB uint8 arrays; transparent pixels (keyed frames) become mid gray."""
    import numpy as np
    from PIL import Image

    frames = []
    for p in paths:
        im = Image.open(p)
        if im.mode in ("RGBA", "LA", "PA") or "transparency" in im.info:
            bg = Image.new("RGBA", im.size, (128, 128, 128, 255))
            im = Image.alpha_composite(bg, im.convert("RGBA"))
        frames.append(np.asarray(im.convert("RGB")))
    sizes = {f.shape[:2] for f in frames}
    if len(sizes) > 1:
        raise ValueError(f"frames differ in size: {sorted(sizes)}")
    return frames


def to_model(queries, size, res):
    """[frame, x, y] in frame pixels -> [t, y, x] in model pixels (as the tapnet demos: scaled corner coordinates)."""
    w, h = size
    return [[t, y * res / h, x * res / w] for t, x, y in queries]


def from_model(tracks, size, res):
    w, h = size
    out = tracks.copy()
    out[..., 0] *= w / res
    out[..., 1] *= h / res
    return out


def track(model, frames, queries, res):
    """frames: list of HxWx3 uint8; queries: [[frame, x, y]]. Returns (N, T, 2) positions and (N, T) visibility."""
    import numpy as np
    import torch
    import torch.nn.functional as F

    h, w = frames[0].shape[:2]
    video = torch.from_numpy(np.stack(frames)).to("cuda").permute(0, 3, 1, 2).float()
    video = F.interpolate(video, size=(res, res), mode="bilinear", align_corners=False, antialias=True)
    video = (video / 255 * 2 - 1).permute(0, 2, 3, 1)[None]  # 1 T H W 3, [-1, 1]
    q = torch.tensor(to_model(queries, (w, h), res), dtype=torch.float32, device="cuda")
    positions, visibility = [], []
    for start in range(0, len(q), CHUNK):
        out = model(video, q[None, start:start + CHUNK])
        vis = (1 - torch.sigmoid(out["occlusion"][0])) * (1 - torch.sigmoid(out["expected_dist"][0]))
        positions.append(out["tracks"][0].cpu().numpy())
        visibility.append(vis.cpu().numpy())
    return from_model(np.concatenate(positions), (w, h), res), np.concatenate(visibility)


def draw(frames, positions, visibility, out, count=6, height=420, trail=6):
    """A few frames side by side with the tracks: dot = visible, ring = occluded, line = the last frames. Each
    frame is cropped to the region of its points (a figure walking across a wide frame stays large)."""
    import colorsys

    import numpy as np
    from PIL import Image, ImageDraw

    n_frames = len(frames)
    picks = sorted({round(i * (n_frames - 1) / max(1, count - 1)) for i in range(count)})
    h, w = frames[0].shape[:2]
    colors = [tuple(round(255 * c) for c in colorsys.hsv_to_rgb(k * 0.618 % 1, 0.9, 1)) for k in range(len(positions))]
    tiles = []
    for t in picks:
        pts = positions[:, t]
        (x0, y0), (x1, y1) = pts.min(0), pts.max(0)
        side = max(x1 - x0, y1 - y0) * 1.3 + 40
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        left, top = np.clip(cx - side / 2, 0, max(0, w - side)), np.clip(cy - side / 2, 0, max(0, h - side))
        box = (round(left), round(top), round(min(w, left + side)), round(min(h, top + side)))
        s = height / (box[3] - box[1])
        im = Image.fromarray(frames[t]).crop(box)
        im = im.resize((round(im.width * s), height))
        fd = ImageDraw.Draw(im)
        for k, (pos, vis) in enumerate(zip(positions, visibility)):
            seg = [((x - box[0]) * s, (y - box[1]) * s) for x, y in pos[max(0, t - trail):t + 1]]
            if len(seg) > 1:
                fd.line(seg, fill=colors[k], width=1)
            x, y = seg[-1]
            dot = (x - 3, y - 3, x + 3, y + 3)
            fd.ellipse(dot, fill=colors[k]) if vis[t] > 0.5 else fd.ellipse(dot, outline=colors[k])
        tiles.append((t, im))
    sheet = Image.new("RGB", (sum(im.width for _, im in tiles), height + 24), "white")
    d = ImageDraw.Draw(sheet)
    x = 0
    for t, im in tiles:
        sheet.paste(im, (x, 24))
        d.text((x + 6, 6), f"frame {t}", fill="black")
        x += im.width
    sheet.save(out)


def run(model, params, job_dir):
    import numpy as np

    names = params["frames"]
    frames = read_frames([job_dir / "in" / n for n in names])
    h, w = frames[0].shape[:2]
    queries = params["queries"]
    for t, x, y in queries:
        if not (0 <= t < len(frames) and 0 <= x <= w and 0 <= y <= h):
            raise ValueError(f"query {[t, x, y]} outside {len(frames)} frames of {w}x{h}")
    res = params.get("resolution", 512)
    t0 = time.time()
    positions, visibility = track(model, frames, queries, res)
    log(f"{len(queries)} points through {len(frames)} frames ({w}x{h} -> {res}x{res}) in {time.time() - t0:.1f}s")
    tracks = np.concatenate([positions, visibility[..., None]], axis=-1).round(3).tolist()
    out = job_dir / "out"
    (out / "tracks.json").write_text(json.dumps(
        {"size": [w, h], "frames": names, "queries": queries, "tracks": tracks}, separators=(",", ":")))
    draw(frames, positions, visibility, out / "tracks.png")
    return ["tracks.json", "tracks.png"]


if __name__ == "__main__":
    serve(load, run)
