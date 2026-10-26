"""Paint hair structure (strands) into mane/tail/fur: 3D noise, stretched along the hair direction.

  tex_hair in.glb out.glb --color R,G,B[@radius] | --dark L  [--region box] [--exclude R,G,B@radius]
           [--dir y] [--freq 220] [--strength 0.25] [--seed 1] [--mask-from other.glb]

Mask: texels near --color (RGB distance) or darker than --dark (brightness), optionally only in --region (box in
fractions of the bounding box, glTF y up / z front), without colors near --exclude. Hair direction --dir (x|y|z,
default y = hair falls/stands vertically). The strands lie in 3D space, so they run across UV seams. The
roughness is modulated too (grooves duller, strands glossier). Too strong or too coarse looks like wood grain
(good: --freq 220 --strength 0.25).
"""
import argparse

import numpy as np
from texlib import Tex, ramp, region_weight, value_noise3

p = argparse.ArgumentParser(prog="tex_hair")
p.add_argument("src")
p.add_argument("dst")
p.add_argument("--color", action="append", default=[])
p.add_argument("--dark", type=float, default=None)
p.add_argument("--region", default=None)
p.add_argument("--exclude", action="append", default=[])
p.add_argument("--dir", default="y", choices=["x", "y", "z"])
p.add_argument("--freq", type=float, default=220.0, help="strands per bounding box width (across)")
p.add_argument("--strength", type=float, default=0.25)
p.add_argument("--seed", type=int, default=1)
p.add_argument("--mask-from", default=None,
               help="determine the color mask on another GLB with the same UVs (e.g. white texture for black)")
a = p.parse_args()

t = Tex(a.src)
rgb = t.rgb
mrgb = rgb
if a.mask_from:
    mrgb = Tex(a.mask_from).rgb
    assert mrgb.shape == rgb.shape, "--mask-from needs the same texture size"
w = np.zeros((t.H, t.W))
for spec in a.color:
    c, rad = (spec.split("@") + ["60"])[:2]
    c = np.array([float(v) for v in c.split(",")])
    rad = float(rad)
    w = np.maximum(w, 1 - ramp(np.linalg.norm(mrgb - c, axis=2), rad * 0.6, rad))
if a.dark is not None:
    w = np.maximum(w, 1 - ramp(mrgb.mean(2), a.dark - 10, a.dark + 10))
for spec in a.exclude:
    c, rad = (spec.split("@") + ["40"])[:2]
    c = np.array([float(v) for v in c.split(",")])
    rad = float(rad)
    w *= ramp(np.linalg.norm(mrgb - c, axis=2), rad * 0.6, rad)
P = t.P
w *= (P[..., 0] >= 0)
if a.region:
    w *= region_weight(P, a.region)

idx = np.nonzero(w > 0.01)
Pw = t.world(P[idx])
size = (t.bmax - t.bmin).max()
k = a.freq / size
scale = np.array([k, k, k])
scale["xyz".index(a.dir)] = k / 5  # hardly any variation along the hair
q = Pw * scale
n = (0.55 * value_noise3(q, a.seed) + 0.3 * value_noise3(q * 2.1, a.seed + 7)
     + 0.15 * value_noise3(q * 4.3, a.seed + 13))
n = (n - n.mean()) / (n.std() + 1e-6)                      # ~N(0,1)
groove = np.clip(-n - 0.9, 0, None)                         # dark grooves between the strands
mod = 1 + a.strength * (0.5 * np.tanh(n) - 0.6 * groove)
ww = w[idx]
rgb[idx] = rgb[idx] * (1 + ww * (mod - 1))[:, None]
if t.mr is not None:
    t.mr[..., 1][idx] = np.clip(t.mr[..., 1][idx] * (1 + ww * (0.25 * groove - 0.12 * np.tanh(n))), 0, 255)
t.save(a.dst)
print(f"[tex_hair] {a.dst}: hair structure on {np.mean(w > 0.5) * 100:.1f}% of the texels")
