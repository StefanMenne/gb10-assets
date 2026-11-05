# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy>=1.26", "pillow>=10", "scipy>=1.11"]
# ///
"""Local steps of the sprite-animation skill: chroma key, cleanup, scaling, quality check, Godot export.

Run with `uv run sprites.py COMMAND ...` (uv installs the dependencies above). Images are PNGs; the chroma color
is green (00FF00) or magenta (FF00FF).

    flatten CUTOUT.png OUT.png --color 00FF00     put a cutout (remove_background) on an exactly flat chroma color
    unpack FRAMES.zip DIR [--range 12-35 --step 2]  extract frames of generate_video(frames=True)
    sheet DIR... --out SHEET.png [--every 2]      numbered contact sheet of frames, to choose a cycle
    cycle DIR [--min 8 --max 48]                  loop lengths where the motion repeats best
    check-bg DIR --color 00FF00                   is the background even and separate in every frame?
    key DIR --color 00FF00 --out DIR [--ml DIR]   chroma key + despill, largest region, 1 px erosion
    scale DIR --out DIR (--height PX | --factor F) [--pad 4]   common crop, premultiplied resize
    qc DIR --out QC.png [--scene SCENE.png]       frames on black, white, the other chroma color and the scene
    export --anim NAME=DIR [--anim ...] --out DIR [--fps 8] [--res res://sprites/]   sheets + SpriteFrames .tres

Every command that takes DIR also takes PNG files. Frames are processed in name order.
"""
import argparse
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
    ys, xs = np.nonzero(np.max([fr[..., 3] for fr in frames], axis=0) > 0.01)
    if not len(ys):
        sys.exit("all frames are empty")
    top, bottom, left, right = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    factor = a.factor or a.height / (bottom - top)
    w, h = round((right - left) * factor), round((bottom - top) * factor)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for f, fr in zip(files, frames):
        small = resize_premultiplied(fr[top:bottom, left:right], (w, h))
        canvas = np.zeros((h + 2 * a.pad, w + 2 * a.pad, 4), np.float32)
        canvas[a.pad:a.pad + h, a.pad:a.pad + w] = small
        save(canvas, out / f.name)
    print(f"{len(files)} frames {w + 2 * a.pad}x{h + 2 * a.pad} -> {out} (factor {factor:.5f}: "
          "pass --factor to scale the other animations of this character alike)")


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


def cmd_export(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res = a.res if a.res.endswith("/") else a.res + "/"
    ext, subs, anims = [], [], []
    for spec in a.anim:
        name, _, src = spec.partition("=")
        if not name or not src:
            sys.exit(f"--anim NAME=DIR, not {spec!r}")
        frames = [Image.open(f).convert("RGBA") for f in pngs([src])]
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
    s.add_argument("--pad", type=int, default=4)
    s = sub.add_parser("qc"); s.add_argument("inputs", nargs="+"); s.add_argument("--out", required=True)
    s.add_argument("--scene", help="a real scene background of the game"); s.add_argument("--at", help="x,y in the scene")
    s = sub.add_parser("export"); s.add_argument("--anim", action="append", required=True, help="NAME=DIR")
    s.add_argument("--out", required=True); s.add_argument("--name", default="sprite_frames")
    s.add_argument("--fps", type=float, default=8); s.add_argument("--res", default="res://")
    s.add_argument("--columns", type=int, default=8)
    a = p.parse_args()
    globals()["cmd_" + a.cmd.replace("-", "_")](a)


if __name__ == "__main__":
    main()
