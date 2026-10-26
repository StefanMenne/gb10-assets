"""Remove painted eyes from the texture (before eye_geo): fill the area around each eye with the surrounding
fur/skin color (Laplace inpainting in the texture, boundary values = texels outside the mask).

  tex_eyecover in.glb out.glb --eyes eyes.json [--cover 1.9] [--size 1.15] [--drop 0] [--apart 0]

eyes.json as for eye_geo (local mesh coordinates). Mask: distance < cover x detected radius around the detected
eye centre and around the centre where eye_geo puts the eye (--size/--drop/--apart as there), so that eyes moved
relative to each other are covered in white and black textures. Unknown options (e.g. --look of eye_geo) are
ignored, so the same options fit both.
"""
import argparse
import json

import numpy as np
from texlib import Tex

p = argparse.ArgumentParser(prog="tex_eyecover")
p.add_argument("src")
p.add_argument("dst")
p.add_argument("--eyes", required=True)
p.add_argument("--cover", type=float, default=1.9)
p.add_argument("--size", type=float, default=1.15)
p.add_argument("--drop", type=float, default=0.0)
p.add_argument("--apart", type=float, default=0.0)
p.add_argument("--iter", type=int, default=400)
p.add_argument("--keep-above", type=float, default=None,
               help="never paint over texels higher than the eye centre + this x radius (protects eyebrows)")
p.add_argument("--only-dark", type=float, default=0.0,
               help="paint over only texels darker than this fraction of the surrounding brightness (e.g. 0.8; "
                    "light figures: lashes/lid lines go, skin and brows outside stay untouched)")
a, _ = p.parse_known_args()

t = Tex(a.src)
P = t.P
valid = P[..., 0] >= 0
W3 = t.world(np.where(valid[..., None], P, 0))
with open(a.eyes) as f:
    eyes = json.load(f)
C = [np.array(e["c"]) for e in eyes]
mid = sum(C) / len(C)
R = np.mean([e["r"] for e in eyes]) * a.size
mask = np.zeros(valid.shape, bool)
for c, e in zip(C, eyes):
    lat = c - mid
    lat[1] = 0
    lat /= np.linalg.norm(lat) + 1e-9
    c2 = c - np.array([0, a.drop * R, 0]) + lat * a.apart * R  # glTF: y up
    for cc in (c, c2):
        m = valid & (np.linalg.norm(W3 - cc, axis=2) < a.cover * e["r"])
        if a.keep_above is not None:
            m &= W3[..., 1] < cc[1] + a.keep_above * e["r"]
        mask |= m

# start value: median of the texels just outside the mask, then Laplace smoothing inside the mask only
ring = valid & ~mask
for _ in range(3):
    grow = mask.copy()
    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        grow |= np.roll(mask, (dy, dx), axis=(0, 1))
    border = grow & ring
    if border.sum() > 50:
        break
rgb = t.rgb
if a.only_dark:
    lum = rgb[..., :3].max(axis=2)  # brightest channel: pink/red does not count as dark
    dark = mask & (lum < a.only_dark * np.median(lum[border]))
    for _ in range(2):  # take the soft edges of the lines along
        grow = dark.copy()
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            grow |= np.roll(dark, (dy, dx), axis=(0, 1))
        dark = grow & mask
    mask = dark
    ring = valid & ~mask
    grow = mask.copy()
    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        grow |= np.roll(mask, (dy, dx), axis=(0, 1))
    border = grow & ring
rgb[mask] = np.median(rgb[border], axis=0)
w = valid.astype(float)
for _ in range(a.iter):
    acc = np.zeros_like(rgb)
    cnt = np.zeros(w.shape)
    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        acc += np.roll(rgb * w[..., None], (dy, dx), axis=(0, 1))
        cnt += np.roll(w, (dy, dx), axis=(0, 1))
    upd = mask & (cnt > 0)
    rgb[upd] = acc[upd] / cnt[upd][:, None]
if t.mr is not None:  # the same roughness as the surroundings
    t.mr[mask] = np.median(t.mr[border], axis=0)
t.save(a.dst)
print(f"[tex_eyecover] {a.dst}: {mask.sum()} texels painted over")
