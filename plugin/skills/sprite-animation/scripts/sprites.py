# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy>=1.26", "pillow>=10", "scipy>=1.11"]
# ///
"""Local steps of the sprite-animation skill: chroma key, cleanup, scaling, quality check, Godot export.

Run with `uv run sprites.py COMMAND ...` (uv installs the dependencies above). Images are PNGs; the chroma color
is green (00FF00) or magenta (FF00FF).

    flatten CUTOUT.png OUT.png --color 00FF00     put a cutout (remove_background) on an exactly flat chroma color
    place CUTOUT.png OUT.png --size 1536x640 --at left   cutout on a chroma canvas at a given height and feet line
    shift IN.png OUT.png [--mirror] [--dx 0 --dy 0] [--extend-bottom 64]   lossless mirror / shift / margin
    unpack FRAMES.zip DIR [--range 12-35 --step 2]  extract frames of generate_video(frames=True)
    queries DIR --out QUERIES.json --color 00FF00 points on the feet and the body to track (track_points)
    root TRACKS.json --out ROOT.json              root motion from the tracks: ground speed, drift, foot sliding
         [--foot --like CYCLE_ROOT.json]          per frame from the feet, for start, stop and standing phases
    stabilize DIR --root ROOT.json --out DIR      remove the travel of a figure walking across: walks in place
    sheet DIR... --out SHEET.png [--every 2]      numbered contact sheet of frames, to choose a cycle
    cycle DIR [--min 8 --max 48]                  loop lengths where the motion repeats best
    check-bg DIR --color 00FF00                   is the background even and separate in every frame?
    key DIR --color 00FF00 --out DIR [--ml DIR]   chroma key + despill, largest region, 1 px erosion
    scale DIR --out DIR (--height PX | --factor F) [--pad 4]   common crop, premultiplied resize
          [--box x0,y0,x1,y1 --anchor x,y]        fixed box for all animations of a character, anchor in sprite px
    qc DIR --out QC.png [--scene SCENE.png]       frames on black, white, the other chroma color and the scene
    export --anim NAME=DIR [--anim ...] --out DIR [--fps 8] [--res res://sprites/] [--root NAME=ROOT.json --factor F]
                                                  sheets + SpriteFrames .tres (+ root motion per animation)

Every command that takes DIR also takes PNG files. Frames are processed in name order.

Root motion: while a foot stands on the ground it moves with the ground, so in every frame the standing feet share
one velocity; the swinging foot moves differently and keeps changing. `root` finds that shared velocity as the
mode of all foot velocities (ground velocity g) and the body velocity b as the slope of the body points. The
figure moves over the ground by b - g per frame: walking in place b ~ 0 and the root motion is -g; walking across
the frame with a static camera g ~ 0 (otherwise the feet slide) and it is b. Vectors are screen pixels of the
video (x right, y down), so a diagonal or depth walk gives a diagonal vector.

`root --foot` is for segments that are not a steady cycle (standing, starting, stopping): per frame the feet's
movement relative to the body, against the walking direction of the cycle; the standing foot moves back with the
ground fastest, so a high percentile of it is the ground speed of that frame (0 while standing). It is scaled so
that over the cycle it matches the cycle's root motion.
"""
import argparse
import json
import math
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

CHROMA = {"00FF00": (0, 255, 0), "FF00FF": (255, 0, 255)}


# ------------------------------------------------------------------------------------------------- helpers
def pngs(paths):
    files = []
    for p in map(Path, paths):
        files += sorted(p.glob("*.png")) if p.is_dir() else [p]
    if not files:
        sys.exit(f"no PNG files in {' '.join(map(str, paths))}")
    return files


def rgb(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255


def rgba(path):
    return np.asarray(Image.open(path).convert("RGBA"), dtype=np.float32) / 255


def save(arr, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.clip(arr, 0, 1) * 255 + 0.5).astype(np.uint8)).save(path)


def chroma(name):
    name = name.upper().lstrip("#")
    if name not in CHROMA:
        sys.exit(f"--color must be one of {', '.join(CHROMA)}")
    return name


def key_strength(img, color):
    """How much a pixel looks like the chroma color: green g - max(r, b), magenta min(r, b) - g."""
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    return g - np.maximum(r, b) if color == "00FF00" else np.minimum(r, b) - g


def border(img, frac=0.03):
    h, w = img.shape[:2]
    m = max(2, round(min(h, w) * frac))
    return np.concatenate([img[:m].reshape(-1, 3), img[-m:].reshape(-1, 3),
                           img[:, :m].reshape(-1, 3), img[:, -m:].reshape(-1, 3)])


def composite(fg, bg):
    a = fg[..., 3:4]
    return fg[..., :3] * a + bg * (1 - a)


# ------------------------------------------------------------------------------------------------ commands
def cmd_flatten(a):
    color = chroma(a.color)
    cut = rgba(a.cutout)
    bg = np.broadcast_to(np.array(CHROMA[color], np.float32) / 255, cut.shape[:2] + (3,))
    save(composite(cut, bg), a.out)
    print(a.out)


def cmd_place(a):
    color = chroma(a.color)
    cut = rgba(a.cutout)
    ys, xs = np.nonzero(cut[..., 3] > 0.01)
    if not len(ys):
        sys.exit(f"{a.cutout} is empty")
    cut = cut[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    W, H = map(int, a.size.lower().split("x"))
    factor = (a.height if a.height > 1 else a.height * H) / cut.shape[0]
    w, h = round(cut.shape[1] * factor), round(cut.shape[0] * factor)
    if w > W:
        sys.exit(f"the figure is {w} px wide at --height {a.height}, wider than the canvas")
    fig = resize_premultiplied(cut, (w, h))
    x = int(np.clip(round({"left": W / 4, "center": W / 2, "right": 3 * W / 4}[a.at] - w / 2), 0, W - w))
    feet = a.feet if a.feet is not None else H - round(a.bottom if a.bottom >= 1 else a.bottom * H)
    y = feet - h
    if y < 0 or feet > H:
        sys.exit(f"the figure ({h} px high, feet at y={feet}) does not fit into the canvas")
    canvas = np.zeros((H, W, 4), np.float32)
    canvas[y:y + h, x:x + w] = fig
    bg = np.broadcast_to(np.array(CHROMA[color], np.float32) / 255, (H, W, 3))
    save(composite(canvas, bg), a.out)
    print(f"{a.out}: {W}x{H}, figure {w}x{h} at x={x}, y={y}, feet at y={feet}")


def cmd_shift(a):
    img = np.asarray(Image.open(a.inp).convert("RGB"))
    if a.mirror:
        img = img[:, ::-1]
    h, w = img.shape[:2]
    fill = CHROMA[chroma(a.color)]
    out = np.empty_like(img)
    out[:] = fill
    ys, xs = slice(max(0, -a.dy), min(h, h - a.dy)), slice(max(0, -a.dx), min(w, w - a.dx))
    out[ys.start + a.dy:ys.stop + a.dy, xs.start + a.dx:xs.stop + a.dx] = img[ys, xs]
    if a.extend_bottom:
        ext = np.empty((a.extend_bottom, w, 3), np.uint8)
        ext[:] = fill
        out = np.concatenate([out, ext])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out).save(a.out)
    print(f"{a.out}: {out.shape[1]}x{out.shape[0]}")


def cmd_unpack(a):
    lo, hi = map(int, a.range.split("-")) if a.range else (0, 10**9)
    out = Path(a.dir)
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(a.zip) as z:
        names = [n for n in sorted(z.namelist()) if re.fullmatch(r"frame_\d+\.png", n)
                 and lo <= int(n[6:-4]) <= hi and (int(n[6:-4]) - lo) % a.step == 0]
        for n in names:
            (out / n).write_bytes(z.read(n))
    print(f"{len(names)} frames -> {out}")


def cmd_sheet(a):
    files = pngs(a.inputs)[::a.every]
    cols = min(len(files), a.columns)
    rows = math.ceil(len(files) / cols)
    first = Image.open(files[0])
    tw, th = a.size, round(a.size * first.height / first.width)
    sheet = Image.new("RGB", (cols * tw, rows * (th + 18)), "white")
    draw = ImageDraw.Draw(sheet)
    for i, f in enumerate(files):
        im = Image.open(f).convert("RGBA").resize((tw, th))
        x, y = i % cols * tw, i // cols * (th + 18)
        tile = Image.new("RGBA", (tw, th), (128, 128, 128, 255))
        tile.alpha_composite(im)
        sheet.paste(tile.convert("RGB"), (x, y + 18))
        draw.text((x + 4, y + 3), f.stem, fill="black")
    sheet.save(a.out)
    print(f"{a.out}: {len(files)} frames")


def cmd_cycle(a):
    files = pngs(a.inputs)
    small = np.stack([np.asarray(Image.open(f).convert("L").resize((64, 64)), np.float32) / 255 for f in files])
    results = []
    for length in range(a.min, min(a.max, len(files) - 1) + 1):
        diffs = [np.abs(small[i] - small[i + length]).mean() for i in range(len(files) - length)]
        start = int(np.argmin(diffs))
        results.append((diffs[start], length, start))
    if not results:
        sys.exit(f"too few frames ({len(files)}) for a cycle of at least {a.min}")
    print("difference  length  start  steps that divide the cycle into 8-12 frames (no stutter at the loop)")
    for diff, length, start in sorted(results)[:10]:
        steps = [f"--step {k} -> {length // k} frames" for k in range(1, 7) if length % k == 0 and 8 <= length // k <= 12]
        print(f"{diff:10.4f}  {length:6d}  {start:5d}  "
              f"{'; '.join(steps) or 'none'}  (unpack --range {start}-{start + length - 1})")
    print("prefer a cycle away from the first and last frames (the keyframes)")


def cmd_check_bg(a):
    color = chroma(a.color)
    target = np.array(CHROMA[color], np.float32) / 255
    bad = 0
    print("frame                 bg color      off target  spread  corners  figure at edge")
    for f in pngs(a.inputs):
        img = rgb(f)
        edge = border(img)
        med = np.median(edge, axis=0)
        spread = np.percentile(np.linalg.norm(edge - med, axis=1), 95) * 255
        off = np.linalg.norm(med - target) * 255
        h, w = img.shape[:2]
        m = max(4, min(h, w) // 16)
        corners = [np.median(c.reshape(-1, 3), axis=0) for c in
                   (img[:m, :m], img[:m, -m:], img[-m:, :m], img[-m:, -m:])]
        corner = max(np.linalg.norm(c - d) for c in corners for d in corners) * 255
        # pixels at the image border that do not look like the background: the figure touches the edge
        strength = key_strength(edge, color)
        at_edge = float((strength < 0.5 * key_strength(med[None], color)[0]).mean())
        problems = [p for p, cond in (("not the chroma color", off > 60), ("uneven", spread > 30),
                                      ("gradient", corner > 25), ("cut off", at_edge > 0.002)) if cond]
        bad += bool(problems)
        print(f"{f.name:20}  {'#%02X%02X%02X' % tuple(int(v * 255) for v in med)}  {off:10.0f}  {spread:6.0f}  "
              f"{corner:7.0f}  {at_edge:13.1%}  {', '.join(problems)}")
    print(f"{bad} frame(s) with problems" if bad else "background ok in all frames")
    sys.exit(1 if bad else 0)


def chroma_key(img, color):
    """Alpha from the key strength relative to the measured background, colors unmixed from it and despilled."""
    bg = np.median(border(img), axis=0)
    bg_strength = max(float(key_strength(bg[None], color)[0]), 0.1)
    s = key_strength(img, color)
    alpha = 1 - np.clip((s - 0.15 * bg_strength) / (0.6 * bg_strength), 0, 1)
    a = alpha[..., None]
    fg = np.where(a > 0.01, (img - (1 - a) * bg) / np.maximum(a, 0.01), 0)
    fg = np.clip(fg, 0, 1)
    r, g, b = fg[..., 0], fg[..., 1], fg[..., 2]
    if color == "00FF00":
        g = np.minimum(g, np.maximum(r, b))
    else:
        spill = np.clip(np.minimum(r, b) - g, 0, None)
        r, b = r - spill, b - spill
    return np.dstack([r, g, b]), alpha


def cleanup(alpha):
    """Only the largest connected region (with its soft edge), then 1 px erosion."""
    labels, n = ndimage.label(alpha > 0.02, structure=np.ones((3, 3)))
    if n > 1:
        sizes = ndimage.sum(np.ones_like(alpha), labels, range(1, n + 1))
        alpha = np.where(labels == 1 + int(np.argmax(sizes)), alpha, 0)
    return ndimage.grey_erosion(alpha, size=(3, 3))


def cmd_key(a):
    color = chroma(a.color)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    print("frame                 unsure   (soft alpha inside the figure, not at its edge; hair, motion blur)")
    for f in pngs(a.inputs):
        img = rgb(f)
        fg, alpha = chroma_key(img, color)
        hard = alpha > 0.5
        band = ndimage.binary_dilation(hard, iterations=2) & ~ndimage.binary_erosion(hard, iterations=2)
        unsure = (alpha > 0.05) & (alpha < 0.95) & ~band
        note = ""
        if a.ml:
            mask_file = Path(a.ml) / f"{f.stem}_mask.png"
            if mask_file.exists():
                ml = np.asarray(Image.open(mask_file).convert("L"), np.float32) / 255
                zone = ndimage.binary_dilation(unsure | band, iterations=2)
                alpha = np.where(zone, ml, alpha)
                bg = np.median(border(img), axis=0)
                m = alpha[..., None]
                fg = np.where(zone[..., None] & (m > 0.01), np.clip((img - (1 - m) * bg) / np.maximum(m, 0.01), 0, 1), fg)
                note = "ml mask combined"
        alpha = cleanup(alpha)
        share = unsure.sum() / max(1, hard.sum())
        flag = "  -> remove_background(mask=True) for this frame" if share > 0.01 and not note else ""
        print(f"{f.name:20}  {share:6.1%}  {note}{flag}")
        save(np.dstack([fg, alpha]), out / f.name)


def resize_premultiplied(img, size):
    """RGBA float image resized with premultiplied alpha (no dark or colored fringes)."""
    alpha = img[..., 3]
    planes = [img[..., c] * alpha for c in range(3)] + [alpha]
    res = [np.asarray(Image.fromarray(p.astype(np.float32), "F").resize(size, Image.LANCZOS)) for p in planes]
    a = np.clip(res[3], 0, 1)
    color = np.dstack([np.where(a > 1e-4, res[c] / np.maximum(a, 1e-4), 0) for c in range(3)])
    return np.dstack([np.clip(color, 0, 1), a])


def cmd_scale(a):
    files = pngs(a.inputs)
    frames = [rgba(f) for f in files]
    if a.box:
        if not a.factor:
            sys.exit("--box needs --factor (the same for all animations of the character)")
        left, top, right, bottom = map(int, a.box.split(","))
        pad = 0
    else:
        ys, xs = np.nonzero(np.max([fr[..., 3] for fr in frames], axis=0) > 0.01)
        if not len(ys):
            sys.exit("all frames are empty")
        top, bottom, left, right = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        pad = a.pad
    factor = a.factor or a.height / (bottom - top)
    w, h = round((right - left) * factor), round((bottom - top) * factor)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for f, fr in zip(files, frames):
        crop = np.zeros((bottom - top, right - left, 4), np.float32)  # the box may reach beyond the frame
        H, W = fr.shape[:2]
        y0, y1, x0, x1 = max(top, 0), min(bottom, H), max(left, 0), min(right, W)
        crop[y0 - top:y1 - top, x0 - left:x1 - left] = fr[y0:y1, x0:x1]
        canvas = np.zeros((h + 2 * pad, w + 2 * pad, 4), np.float32)
        canvas[pad:pad + h, pad:pad + w] = resize_premultiplied(crop, (w, h))
        save(canvas, out / f.name)
    print(f"{len(files)} frames {w + 2 * pad}x{h + 2 * pad} -> {out} (factor {factor:.5f}: "
          "pass --factor to scale the other animations of this character alike)")
    if a.anchor:
        ax, ay = map(float, a.anchor.split(","))
        print(f"anchor ({(ax - left) * factor + pad:.1f}, {(ay - top) * factor + pad:.1f}) in sprite pixels")


def cmd_qc(a):
    files = pngs(a.inputs)
    frames = [Image.open(f).convert("RGBA") for f in files]
    w, h = frames[0].size
    backgrounds = [("black", Image.new("RGBA", (w, h), (0, 0, 0, 255))),
                   ("white", Image.new("RGBA", (w, h), (255, 255, 255, 255))),
                   ("magenta", Image.new("RGBA", (w, h), (255, 0, 255, 255))),
                   ("green", Image.new("RGBA", (w, h), (0, 255, 0, 255)))]
    if a.scene:
        scene = Image.open(a.scene).convert("RGBA")
        x, y = (map(int, a.at.split(",")) if a.at else ((scene.width - w) // 2, (scene.height - h) // 2))
        backgrounds.append(("scene", scene.crop((x, y, x + w, y + h))))   # 1:1, as in the game
    zoom = max(1, 160 // max(w, h))
    tw, th = w * zoom, h * zoom
    sheet = Image.new("RGB", (60 + len(frames) * tw, len(backgrounds) * th + 18), "white")
    draw = ImageDraw.Draw(sheet)
    for i, f in enumerate(files):
        draw.text((60 + i * tw + 2, 3), f.stem[-4:], fill="black")
    for row, (name, bg) in enumerate(backgrounds):
        draw.text((4, 18 + row * th + th // 2), name, fill="black")
        for i, fr in enumerate(frames):
            tile = bg.copy()
            tile.alpha_composite(fr)
            sheet.paste(tile.convert("RGB").resize((tw, th), Image.NEAREST), (60 + i * tw, 18 + row * th))
    sheet.save(a.out)
    print(f"{a.out}: {len(frames)} frames on {', '.join(n for n, _ in backgrounds)}")


# --------------------------------------------------------------------------------------------- root motion
def frame_no(name):
    """frame_0039.png -> 39 (names from unpack, kept by key, scale and stabilize)."""
    digits = re.findall(r"\d+", Path(name).stem)
    if not digits:
        sys.exit(f"{name}: no frame number in the name")
    return int(digits[-1])


def figure_alpha(path, color):
    """Alpha of a keyed frame, or the chroma key's alpha of a frame on the chroma color."""
    img = rgba(path)
    return img[..., 3] if img[..., 3].min() < 1 else chroma_key(img[..., :3], color)[1]


def spread_points(xs, ys, n):
    """n of the points, far apart from each other (farthest point sampling, starting at the lowest)."""
    pts = np.stack([xs, ys], 1).astype(np.float64)
    if len(pts) <= n:
        return pts
    chosen = [int(np.argmax(pts[:, 1]))]
    dist = np.linalg.norm(pts - pts[chosen[0]], axis=1)
    for _ in range(n - 1):
        chosen.append(int(np.argmax(dist)))
        dist = np.minimum(dist, np.linalg.norm(pts - pts[chosen[-1]], axis=1))
    return pts[chosen]


def cmd_queries(a):
    color = chroma(a.color)
    files = pngs(a.inputs)
    queries, kinds, heights = [], [], []
    for t in range(0, len(files), a.every):
        mask = ndimage.binary_erosion(figure_alpha(files[t], color) > 0.5, iterations=3)
        ys, xs = np.nonzero(mask)
        if not len(ys):
            continue
        top, bottom = ys.min(), ys.max()
        h = bottom - top
        heights.append(h)
        middle = (ys > top + 0.35 * h) & (ys < top + 0.6 * h)
        # feet: the lowest part of the figure; body: the hips and belly, not the arms at the sides
        groups = {"foot": ys >= bottom - a.foot * h,
                  "body": middle & (np.abs(xs - np.median(xs[middle])) < 0.12 * h)}
        for kind, n in (("foot", a.feet), ("body", a.body)):
            sel = groups[kind]
            for x, y in spread_points(xs[sel], ys[sel], n):
                queries.append([t, round(float(x) + 0.5, 1), round(float(y) + 0.5, 1)])
                kinds.append(kind)
    if not queries:
        sys.exit("no figure found in the frames (wrong --color?)")
    w, h = Image.open(files[0]).size
    Path(a.out).write_text(json.dumps({"frames": [f.name for f in files], "size": [w, h],
                                       "figure_height": float(np.median(heights)), "kinds": kinds,
                                       "queries": queries}))
    print(f"{a.out}: {kinds.count('foot')} foot and {kinds.count('body')} body points in {len(files)} frames "
          f"(figure about {np.median(heights):.0f} px high); track them with track_points")


def densest(samples, radius):
    """The 2D point where the samples are densest: histogram peak, refined by mean shift within radius."""
    cell = radius / 2
    lo = samples.min(0)
    idx = np.floor((samples - lo) / cell).astype(int)
    hist = np.zeros(idx.max(0) + 1)
    np.add.at(hist, (idx[:, 0], idx[:, 1]), 1)
    hist = ndimage.uniform_filter(hist, size=3, mode="constant")
    center = lo + (np.array(np.unravel_index(np.argmax(hist), hist.shape)) + 0.5) * cell
    for _ in range(20):
        near = np.linalg.norm(samples - center, axis=1) < radius
        center = samples[near].mean(0)
    return center


def root_motion(tracks, kinds, height, tolerance=0.006):
    """Ground velocity g (mode of the foot velocities), body velocity b (slope of the body points) and the root
    motion b - g per frame transition. tracks: (points, frames, 3) x, y, visibility; height: figure height in px."""
    kinds = np.asarray(kinds)
    pos, vis = tracks[..., :2], tracks[..., 2] > 0.5
    vel = pos[:, 1:] - pos[:, :-1]
    ok = vis[:, 1:] & vis[:, :-1]
    n = vel.shape[1]

    def per_frame_median(sel, valid):
        return np.array([np.median(vel[sel, t][valid[:, t]], axis=0) if valid[:, t].any() else (np.nan, np.nan)
                         for t in range(n)])

    body = kinds == "body"
    b_t = per_frame_median(body, ok[body])
    if np.isnan(b_t).all():
        sys.exit("no body point is visible in two frames in a row")
    b_t = np.where(np.isnan(b_t), np.nanmedian(b_t, axis=0), b_t)
    path = np.vstack([[0, 0], np.cumsum(b_t, axis=0)])
    b = np.polyfit(np.arange(n + 1), path, 1)[0]

    foot = kinds == "foot"
    fv, fok = vel[foot], ok[foot]
    if not fok.any():
        sys.exit("no foot point is visible in two frames in a row")
    radius = max(1.0, tolerance * height)
    g = densest(fv[fok], radius)
    stance = fok & (np.linalg.norm(fv - g, axis=-1) < radius)
    g_t = per_frame_median(foot, stance)
    measured = ~np.isnan(g_t[:, 0])
    g_t[~measured] = g
    root_t = b - g_t
    speed = float(np.linalg.norm(b - g))
    return {"velocity": (b - g).tolist(), "body_velocity": b.tolist(), "ground_velocity": g.tolist(),
            "per_frame": root_t.round(3).tolist(), "speed": speed, "coverage": float(measured.mean()),
            "unsteady": float(np.percentile(np.linalg.norm(g_t[measured] - g, axis=1), 75) / max(speed, 1e-6))
            if measured.any() else 1.0,
            "walks_across": bool(np.linalg.norm(b) > np.linalg.norm(g))}


def foot_root(tracks, kinds, query_frames, direction, near=6, pct=90):
    """Ground speed per frame transition along `direction` (the walking direction): the feet's velocity relative
    to the median body velocity, projected against the walking direction, its `pct` percentile, never negative.
    Only tracks whose query lies at most `near` frames away (fresh tracks are more exact)."""
    kinds, qf = np.asarray(kinds), np.asarray(query_frames)
    d = np.asarray(direction, np.float64)
    d = d / max(np.linalg.norm(d), 1e-9)
    vis = tracks[..., 2] > 0.5
    speed = np.zeros(tracks.shape[1] - 1)
    for t in range(len(speed)):
        ok = vis[:, t] & vis[:, t + 1] & (np.abs(qf - t) <= near)
        v = tracks[:, t + 1, :2] - tracks[:, t, :2]
        body, foot = ok & (kinds == "body"), ok & (kinds == "foot")
        if foot.any():
            vb = np.median(v[body], 0) if body.any() else np.zeros(2)
            speed[t] = max(0.0, float(np.percentile(-((v[foot] - vb) @ d), pct)))
    return speed


def cmd_root(a):
    tr = json.loads(Path(a.tracks).read_text())
    q = json.loads(Path(a.queries).read_text())
    if len(q["kinds"]) != len(tr["tracks"]) or q["frames"] != tr["frames"]:
        sys.exit(f"{a.tracks} does not belong to {a.queries} (other points or frames)")
    height = q["figure_height"]
    tracks = np.array(tr["tracks"], np.float64)
    query_frames = [p[0] for p in q["queries"]]
    if getattr(a, "foot", False):
        return cmd_root_foot(a, tr, tracks, q, query_frames)
    r = root_motion(tracks, q["kinds"], height)
    vx, vy = r["velocity"]
    speed = r["speed"]
    g, b = np.linalg.norm(r["ground_velocity"]), np.linalg.norm(r["body_velocity"])
    warnings = []
    if speed < 0.008 * height:
        warnings.append("the feet hardly push back: the figure marches on the spot instead of walking; moved in "
                        "the game its feet would slide. Try another seed")
    if r["coverage"] < 0.8:
        warnings.append(f"only in {r['coverage']:.0%} of the frames a foot moves with the ground: the feet glide "
                        "or the tracks are lost (look at tracks.png)")
    if r["unsteady"] > 0.3:
        warnings.append(f"the ground speed varies by {r['unsteady']:.0%} between the frames: uneven steps or "
                        "sliding feet")
    if r["walks_across"] and g > 0.15 * speed:
        warnings.append(f"the standing foot slides over the ground by {g / speed:.0%} of the walking speed")
    r.update(frames=tr["frames"], figure_height=height, warnings=warnings,
             foot_speed=float(foot_root(tracks, q["kinds"], query_frames, r["velocity"]).mean()))
    Path(a.out).write_text(json.dumps(r))
    kind = ("walks across the frame: run stabilize before cycle and scale" if r["walks_across"] else
            f"walks in place (body drifts {b / max(speed, 1e-6):.0%} of the walking speed)")
    print(f"{a.out}: root motion ({vx:+.2f}, {vy:+.2f}) px per frame = {speed * a.fps / height:.2f} figure "
          f"heights per second at {a.fps:g} fps; {kind}")
    print(f"ground velocity ({r['ground_velocity'][0]:+.2f}, {r['ground_velocity'][1]:+.2f}), body velocity "
          f"({r['body_velocity'][0]:+.2f}, {r['body_velocity'][1]:+.2f}) px per frame, a standing foot found in "
          f"{r['coverage']:.0%} of the frames")
    for w in warnings:
        print(f"WARNING: {w}")


def cmd_root_foot(a, tr, tracks, q, query_frames):
    if not a.like:
        sys.exit("--foot needs --like CYCLE_ROOT.json: the root motion of this direction's walk cycle")
    like = json.loads(Path(a.like).read_text())
    if "foot_speed" not in like:
        sys.exit(f"{a.like} has no foot_speed: run root on the cycle again with this version")
    d = np.array(like["velocity"], np.float64)
    d /= max(np.linalg.norm(d), 1e-9)
    speed = foot_root(tracks, q["kinds"], query_frames, d)
    speed = ndimage.median_filter(speed, size=5, mode="nearest") * like["speed"] / max(like["foot_speed"], 1e-6)
    per_frame = speed[:, None] * d
    body = np.asarray(q["kinds"]) == "body"
    vel = tracks[body, 1:, :2] - tracks[body, :-1, :2]
    r = {"velocity": per_frame.mean(0).tolist(), "body_velocity": np.median(vel, axis=(0, 1)).tolist(),
         "ground_velocity": (-per_frame.mean(0)).tolist(), "per_frame": per_frame.round(3).tolist(),
         "speed": float(speed.mean()), "walks_across": False, "frames": tr["frames"],
         "figure_height": q["figure_height"], "warnings": [], "from_feet": True}
    Path(a.out).write_text(json.dumps(r))
    steps = " ".join(f"{s:.1f}" for s in speed)
    print(f"{a.out}: root motion from the feet, mean {r['speed']:.2f} px per frame (cycle {like['speed']:.2f}), "
          f"per frame: {steps}")


def cmd_stabilize(a):
    root = json.loads(Path(a.root).read_text())
    index = {frame_no(n): i for i, n in enumerate(root["frames"])}
    files = pngs(a.inputs)
    missing = [f.name for f in files if frame_no(f.name) not in index]
    if missing:
        sys.exit(f"{', '.join(missing[:3])} not among the tracked frames of {a.root}")
    t = np.array([index[frame_no(f.name)] for f in files], np.float64)
    frames = [rgba(f) for f in files]
    boxes = []
    for fr in frames:
        ys, xs = np.nonzero(fr[..., 3] > 0.5)
        if not len(ys):
            sys.exit("a frame is empty: run stabilize on keyed frames")
        boxes.append((ys.max() + 1, (xs.min() + xs.max() + 1) / 2, ys.max() + 1 - ys.min()))
    boxes = np.array(boxes)
    ref = float(np.median(t))
    b = np.array(root["body_velocity"])
    # walking towards or away from the camera changes the size: the trend of the height (steps make it wobble)
    slope = np.polyfit(t, np.log(boxes[:, 2]), 1)[0] if len(t) > 2 else 0.0
    change = math.exp(abs(slope) * (t.max() - t.min())) - 1
    if change > a.max_scale:
        sys.exit(f"the figure changes its size by {change:.0%} over the frames: it walks towards or away from the "
                 "camera. Use a video of walking in place for this direction")
    scale = change > 0.02
    shift = (t - ref)[:, None] * b                       # x, y travel relative to the reference frame
    pivot = np.median(boxes[:, [1, 0]] - shift, axis=0)  # bottom centre (x, y) of the figure at the reference
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for f, fr, ti, (dx, dy) in zip(files, frames, t, shift):
        k = math.exp(slope * (ti - ref)) if scale else 1.0
        # output pixel o shows input pixel k * (o - pivot) + pivot + shift, as (row, column)
        c = np.array([pivot[1], pivot[0]])
        offset = c + np.array([dy, dx]) - k * c
        alpha = fr[..., 3]
        planes = [fr[..., i] * alpha for i in range(3)] + [alpha]
        moved = [ndimage.affine_transform(p, [k, k], offset=offset, order=1, mode="constant") for p in planes]
        a_ = np.clip(moved[3], 0, 1)
        color = np.dstack([np.where(a_ > 1e-4, moved[i] / np.maximum(a_, 1e-4), 0) for i in range(3)])
        save(np.dstack([np.clip(color, 0, 1), a_]), out / f.name)
    print(f"{len(files)} frames -> {out}: travel of ({b[0]:+.2f}, {b[1]:+.2f}) px per frame removed"
          + (f", size change of {change:.0%} evened out" if scale else ""))


def root_per_sprite_frame(root, files, factor):
    """Movement in sprite pixels while each exported frame is shown (the last one: back to the first)."""
    nums = [frame_no(f.name) for f in files]
    tracked = [frame_no(n) for n in root["frames"]]
    per_frame = np.array(root["per_frame"])
    velocity = {}  # per video frame number: movement to the next video frame
    for j in range(len(tracked) - 1):
        for k in range(tracked[j], tracked[j + 1]):
            velocity[k] = per_frame[j] / (tracked[j + 1] - tracked[j])
    mean = np.array(root["velocity"])
    step = int(np.median(np.diff(nums))) if len(nums) > 1 else 1
    moves = []
    for i, n in enumerate(nums):
        nxt = nums[i + 1] if i + 1 < len(nums) else n + step
        moves.append(sum((velocity.get(k, mean) for k in range(n, nxt)), np.zeros(2)) * factor)
    return np.array(moves)


def cmd_export(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res = a.res if a.res.endswith("/") else a.res + "/"
    roots = {}
    for spec in a.root or []:
        name, _, path = spec.partition("=")
        if not name or not path:
            sys.exit(f"--root NAME=ROOT.json, not {spec!r}")
        roots[name] = json.loads(Path(path).read_text())
    if roots and not a.factor:
        sys.exit("--root needs --factor, the factor printed by scale")
    ext, subs, anims = [], [], []
    for spec in a.anim:
        name, _, src = spec.partition("=")
        if not name or not src:
            sys.exit(f"--anim NAME=DIR, not {spec!r}")
        files = pngs([src])
        frames = [Image.open(f).convert("RGBA") for f in files]
        if name in roots:
            moves = root_per_sprite_frame(roots[name], files, a.factor)
            v = moves.mean(0) * a.fps
            (out / f"{name}_root.json").write_text(json.dumps({
                "fps": a.fps, "velocity_px_s": v.round(2).tolist(), "per_frame_px": moves.round(3).tolist(),
                "note": "sprite pixels, x right, y down; per_frame_px[i]: movement while frame i is shown; "
                        "negate x for flip_h"}, indent=1))
            print(f"{out / name}_root.json: {np.linalg.norm(v):.1f} px/s ({v[0]:+.1f}, {v[1]:+.1f})")
        w, h = frames[0].size
        if any(fr.size != (w, h) for fr in frames):
            sys.exit(f"{name}: frames differ in size, run scale first")
        cols = min(len(frames), a.columns)
        sheet = Image.new("RGBA", (cols * w, math.ceil(len(frames) / cols) * h), (0, 0, 0, 0))
        regions = []
        for i, fr in enumerate(frames):
            x, y = i % cols * w, i // cols * h
            sheet.paste(fr, (x, y))
            regions.append((x, y))
        sheet.save(out / f"{name}.png")
        (out / name).mkdir(exist_ok=True)
        for i, fr in enumerate(frames):
            fr.save(out / name / f"{name}_{i:02d}.png")
        ext.append(f'[ext_resource type="Texture2D" path="{res}{name}.png" id="{len(ext) + 1}"]')
        ids = []
        for x, y in regions:
            sid = f"AtlasTexture_{name}_{len(ids)}"
            subs.append(f'[sub_resource type="AtlasTexture" id="{sid}"]\natlas = ExtResource("{len(ext)}")\n'
                        f"region = Rect2({x}, {y}, {w}, {h})")
            ids.append(sid)
        frame_list = ", ".join(f'{{\n"duration": 1.0,\n"texture": SubResource("{sid}")\n}}' for sid in ids)
        anims.append(f'{{\n"frames": [{frame_list}],\n"loop": true,\n"name": &"{name}",\n"speed": {a.fps:.1f}\n}}')
        print(f"{out / name}.png: {len(frames)} frames {w}x{h}, single frames in {out / name}/")
    tres = (f'[gd_resource type="SpriteFrames" load_steps={len(ext) + len(subs) + 1} format=3]\n\n'
            + "\n".join(ext) + "\n\n" + "\n\n".join(subs) + "\n\n[resource]\nanimations = [" + ", ".join(anims) + "]\n")
    (out / f"{a.name}.tres").write_text(tres)
    print(f"{out / a.name}.tres: SpriteFrames with {', '.join(s.partition('=')[0] for s in a.anim)} "
          f"(textures expected at {res})")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("flatten"); s.add_argument("cutout"); s.add_argument("out"); s.add_argument("--color", default="00FF00")
    s = sub.add_parser("place"); s.add_argument("cutout"); s.add_argument("out"); s.add_argument("--color", default="00FF00")
    s.add_argument("--size", default="1536x640", help="WxH of the video"); s.add_argument("--at", choices=("left", "center", "right"), default="left")
    s.add_argument("--height", type=float, default=0.8, help="figure height: fraction of the canvas height or px")
    s.add_argument("--bottom", type=float, default=0.06, help="margin below the feet, fraction of the height or px")
    s.add_argument("--feet", type=int, help="y of the feet line in px (instead of --bottom)")
    s = sub.add_parser("shift"); s.add_argument("inp"); s.add_argument("out"); s.add_argument("--color", default="00FF00")
    s.add_argument("--mirror", action="store_true", help="mirror horizontally (before shifting)")
    s.add_argument("--dx", type=int, default=0); s.add_argument("--dy", type=int, default=0)
    s.add_argument("--extend-bottom", type=int, default=0, help="add this many rows of the chroma color below")
    s = sub.add_parser("queries"); s.add_argument("inputs", nargs="+"); s.add_argument("--out", required=True)
    s.add_argument("--color", default="00FF00"); s.add_argument("--every", type=int, default=4, help="query every n-th frame")
    s.add_argument("--feet", type=int, default=6); s.add_argument("--body", type=int, default=3)
    s.add_argument("--foot", type=float, default=0.1, help="height of the foot band, fraction of the figure height")
    s = sub.add_parser("root"); s.add_argument("tracks"); s.add_argument("--queries", required=True)
    s.add_argument("--out", required=True); s.add_argument("--fps", type=float, default=24, help="of the video")
    s.add_argument("--foot", action="store_true", help="per frame from the feet (start, stop, standing phases)")
    s.add_argument("--like", help="ROOT.json of the walk cycle: direction and scale for --foot")
    s = sub.add_parser("stabilize"); s.add_argument("inputs", nargs="+"); s.add_argument("--root", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--max-scale", type=float, default=0.10, help="largest size change that is evened out")
    s = sub.add_parser("unpack"); s.add_argument("zip"); s.add_argument("dir"); s.add_argument("--range")
    s.add_argument("--step", type=int, default=1, help="every n-th frame of the range")
    s = sub.add_parser("sheet"); s.add_argument("inputs", nargs="+"); s.add_argument("--out", required=True)
    s.add_argument("--every", type=int, default=1); s.add_argument("--columns", type=int, default=12)
    s.add_argument("--size", type=int, default=160)
    s = sub.add_parser("cycle"); s.add_argument("inputs", nargs="+")
    s.add_argument("--min", type=int, default=8); s.add_argument("--max", type=int, default=48)
    s = sub.add_parser("check-bg"); s.add_argument("inputs", nargs="+"); s.add_argument("--color", default="00FF00")
    s = sub.add_parser("key"); s.add_argument("inputs", nargs="+"); s.add_argument("--color", default="00FF00")
    s.add_argument("--out", required=True); s.add_argument("--ml", help="directory with <frame>_mask.png")
    s = sub.add_parser("scale"); s.add_argument("inputs", nargs="+"); s.add_argument("--out", required=True)
    g = s.add_mutually_exclusive_group(required=True)
    g.add_argument("--height", type=int, help="height of the figure in pixels"); g.add_argument("--factor", type=float)
    s.add_argument("--pad", type=int, default=4, help="not with --box")
    s.add_argument("--box", help="x0,y0,x1,y1: fixed crop in the video frame (may exceed it), needs --factor")
    s.add_argument("--anchor", help="x,y in the video frame (the feet point): printed in sprite pixels")
    s = sub.add_parser("qc"); s.add_argument("inputs", nargs="+"); s.add_argument("--out", required=True)
    s.add_argument("--scene", help="a real scene background of the game"); s.add_argument("--at", help="x,y in the scene")
    s = sub.add_parser("export"); s.add_argument("--anim", action="append", required=True, help="NAME=DIR")
    s.add_argument("--out", required=True); s.add_argument("--name", default="sprite_frames")
    s.add_argument("--fps", type=float, default=8); s.add_argument("--res", default="res://")
    s.add_argument("--columns", type=int, default=8)
    s.add_argument("--root", action="append", help="NAME=ROOT.json: root motion of that animation")
    s.add_argument("--factor", type=float, help="the factor scale printed (video px -> sprite px)")
    a = p.parse_args()
    globals()["cmd_" + a.cmd.replace("-", "_")](a)


if __name__ == "__main__":
    main()
