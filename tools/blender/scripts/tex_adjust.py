"""Adjust colors/material in the texture of a GLB; geometry and UVs stay unchanged.

  tex_adjust in.glb out.glb [options]

  --factor F --lum A,B --sat S   grey tones (max-min < S, brightness A..B, soft border) x F
  --red-light X                  mix strong red by fraction X towards light red (240,80,70)
  --recolor "R,G,B>R,G,B[@radius]"   shift colors near the source color (RGB distance < radius, default 60,
                                 soft border) by (target - source); gradations stay. Repeatable.
  --region "x0,x1,y0,y1,z0,z1"   --recolor only on texels whose 3D point lies in this box (fractions of the
                                 bounding box, glTF: y = up, z = front). Repeated = applies to the --recolor
                                 at the same position.
  --whiten "x0,x1,y0,y1,z0,z1[@R,G,B]"  in this 3D region set all colored texels (saturation > 25) to the
                                 base color (default 250,248,246) - against invented spots on the back.
                                 Repeatable.
  --metal S                      multiply the metallic channel by S (0 = no metal)
  --metal-keep "x0,x1,y0,y1,z0,z1"  keep metal only on gold colored texels in this 3D region (e.g. a crown),
                                 0 everywhere else. Repeatable.

Generated models often oversaturate colors and invent backs (the color of cheeks or ears ends up on the back of
the head), and some come out as metal entirely (grey, mirroring): check the metallic channel.
"""
import argparse

import numpy as np
from texlib import Tex, ramp, region_weight

p = argparse.ArgumentParser(prog="tex_adjust")
p.add_argument("src")
p.add_argument("dst")
p.add_argument("--factor", type=float, default=1.0)
p.add_argument("--lum", default="60,140")
p.add_argument("--sat", type=float, default=30)
p.add_argument("--red-light", type=float, default=0.0)
p.add_argument("--recolor", action="append", default=[])
p.add_argument("--region", action="append", default=[])
p.add_argument("--whiten", action="append", default=[])
p.add_argument("--metal", type=float, default=1.0)
p.add_argument("--metal-keep", action="append", default=[])
a = p.parse_args()

t = Tex(a.src)
rgb = t.rgb
msg = []

# ---------------------------------------------------------------- grey tones
if a.factor != 1.0:
    lo, hi = (float(x) for x in a.lum.split(","))
    L = rgb.mean(2)
    S = rgb.max(2) - rgb.min(2)
    w = ramp(L, lo - 20, lo) * (1 - ramp(L, hi, hi + 25)) * (1 - ramp(S, a.sat - 10, a.sat + 10))
    rgb *= (1 - w * (1 - a.factor))[..., None]
    msg.append(f"grey x{a.factor} on {np.mean(w > 0.5) * 100:.0f}%")

# ---------------------------------------------------------------- lighten red
if a.red_light > 0:
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    wr = ramp(r / (np.maximum(g, b) + 8), 1.5, 2.2) * ramp(r, 60, 100)
    rgb += (wr * a.red_light)[..., None] * (np.array([240.0, 80.0, 70.0]) - rgb)
    msg.append(f"red lightened on {np.mean(wr > 0.5) * 100:.0f}%")

# ---------------------------------------------------------------- replace colors
if a.recolor:
    regions = a.region + [None] * (len(a.recolor) - len(a.region))
    for spec, reg in zip(a.recolor, regions):
        rad = 60.0
        if "@" in spec:
            spec, rad = spec.split("@")
            rad = float(rad)
        src, dst = (np.array([float(v) for v in s.split(",")]) for s in spec.split(">"))
        dist = np.linalg.norm(rgb - src, axis=2)
        w = 1 - ramp(dist, rad * 0.5, rad)
        if reg:
            w *= region_weight(t.P, reg)
        rgb += w[..., None] * (dst - src)
        msg.append(f"{tuple(int(v) for v in src)}->{tuple(int(v) for v in dst)}{' in ' + reg if reg else ''}: "
                   f"{np.mean(w > 0.5) * 100:.1f}%")

for spec in a.whiten:
    reg, col = (spec.split("@") + ["250,248,246"])[:2]
    col = np.array([float(v) for v in col.split(",")])
    S = rgb.max(2) - rgb.min(2)
    w = region_weight(t.P, reg) * ramp(S, 15, 35)
    rgb += w[..., None] * (col - rgb)
    msg.append(f"white in {reg}: {np.mean(w > 0.5) * 100:.1f}%")
np.clip(rgb, 0, 255, out=rgb)

# ---------------------------------------------------------------- metal (blue channel of metallicRoughness)
if a.metal_keep and t.mr is not None:
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    gold = ramp(r - b, 40, 70) * ramp(r, 110, 140) * ramp(g - b, 20, 40)
    keep = np.clip(sum(region_weight(t.P, reg) for reg in a.metal_keep), 0, 1) * gold
    t.mr[..., 2] *= keep
    msg.append(f"metal only in {' + '.join(a.metal_keep)} (gold): {np.mean(t.mr[..., 2] > 128) * 100:.1f}%")
if a.metal != 1.0 and t.mr is not None:
    t.mr[..., 2] *= a.metal
    msg.append(f"metal x{a.metal}")

t.save(a.dst)
print(f"[tex_adjust] {a.dst}: " + ", ".join(msg))
