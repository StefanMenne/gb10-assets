"""Find eyes and paint them again: dark, mirror-symmetric spots on the head.

  tex_eyes in.glb out.glb --style black|white [--pupil 0.55] [--glint 0.28] [--dark 40] [--debug]
           [--eyes "x,y,z;x,y,z"] [--save-eyes eyes.json] [--load-eyes eyes.json]

--style black: white eye, black pupil (radius = pupil x eye radius), small glint.
--style white: the eye stays black, plus a white "pupil" (a glint, radius = glint x eye radius).
Painting works on 3D distances (position map), whatever the eyes' layout in the UVs.
The eyes are found automatically: connected dark areas (brightness < dark) in the upper half, of which the pair
that is the most mirror-symmetric (x or z mirror plane, the head may be turned) and the same size. Generated
eyes are domes: the centre is the outermost point in the eye's gaze direction, not the centroid (that is inside).
A second texture on the same mesh takes over the eyes with --save-eyes/--load-eyes, unless the texture comes
from another source (a new mesh instead of a new texture); special cases (eyes merged with the mouth): --eyes.
"""
import argparse
import json
from collections import deque

import numpy as np
from texlib import Tex, ramp

p = argparse.ArgumentParser(prog="tex_eyes")
p.add_argument("src")
p.add_argument("dst")
p.add_argument("--style", required=True, choices=["black", "white"])
p.add_argument("--pupil", type=float, default=0.55)
p.add_argument("--glint", type=float, default=0.28)
p.add_argument("--dark", type=float, default=40)
p.add_argument("--grid", type=float, default=0.012, help="voxel size for the spot search (fraction of the bbox)")
p.add_argument("--eyes", default=None,
               help="eye centres by hand 'x,y,z;x,y,z' (fractions of the bbox), the nearest spot is taken")
p.add_argument("--save-eyes", default=None, help="save the eyes found (centre, radius) as JSON")
p.add_argument("--load-eyes", default=None, help="take the eyes from JSON (e.g. white -> black, same mesh)")
p.add_argument("--debug", action="store_true")
a = p.parse_args()

t = Tex(a.src)
P = t.P
valid = P[..., 0] >= 0
L = t.rgb.mean(2)
dark = valid & (L < a.dark) & (P[..., 1] > 0.35)
ys, xs = np.nonzero(dark)
pts = P[ys, xs]

# ---------------------------------------------------------------- spots: voxel connectivity
vox = np.floor(pts / a.grid).astype(int)
keys = {}
for i, v in enumerate(map(tuple, vox)):
    keys.setdefault(v, []).append(i)
seen, comps = set(), []
for v in keys:
    if v in seen:
        continue
    comp, dq = [], deque([v])
    seen.add(v)
    while dq:
        c = dq.popleft()
        comp.extend(keys[c])
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    nb = (c[0] + dx, c[1] + dy, c[2] + dz)
                    if nb in keys and nb not in seen:
                        seen.add(nb)
                        dq.append(nb)
    comps.append(np.array(comp))

blobs = []
for c in comps:
    if len(c) < 200:
        continue
    q = t.world(pts[c])
    cen = q.mean(0)
    ext = np.percentile(np.linalg.norm(q - cen, axis=1), 90)
    blobs.append(dict(idx=c, cen=cen, cen_n=pts[c].mean(0), r=ext, n=len(c)))
size = (t.bmax - t.bmin).max()
blobs = [b for b in blobs if b["r"] < 0.12 * size]
if a.debug:
    for b in blobs:
        print(f"  spot n={b['n']:6d} centre(norm)={np.round(b['cen_n'], 2)} r={b['r'] / size:.3f}")

# ---------------------------------------------------------------- best mirror pair
# The mirror plane is unknown (the head may be turned): same height, same depth along the other axis, similar
# size; bigger spots preferred (eyes > nostrils).
best, score = None, 1e9
if a.eyes:
    want = [np.array([float(v) for v in e.split(",")]) for e in a.eyes.split(";")]
    best = tuple(min(blobs, key=lambda b: np.linalg.norm(b["cen_n"] - w)) for w in want)
    score = 0.0
elif not a.load_eyes:
    for i in range(len(blobs)):
        for j in range(i + 1, len(blobs)):
            A, B = blobs[i]["cen_n"], blobs[j]["cen_n"]
            for ax in (0, 2):
                other = 2 - ax
                if abs(A[ax] - B[ax]) < 0.05:  # same side: not a pair of eyes
                    continue
                s = 0.05 * abs(A[ax] + B[ax] - 1) + abs(A[1] - B[1]) + abs(A[other] - B[other])
                s += 0.5 * abs(np.log(blobs[i]["n"] / blobs[j]["n"]))
                rank = s - 0.05 * np.log(min(blobs[i]["n"], blobs[j]["n"]))
                if rank < score:
                    best, score, raw = (blobs[i], blobs[j]), rank, s
    if best is None or raw > 0.35:
        raise SystemExit("[tex_eyes] no pair of eyes found; give --eyes (--debug lists the spots)")
    score = raw
if best:
    print(f"[tex_eyes] eyes: {np.round(best[0]['cen_n'], 2)} / {np.round(best[1]['cen_n'], 2)}  "
          f"r={best[0]['r'] / size:.3f}/{best[1]['r'] / size:.3f}  (symmetry {score:.2f})")

# ---------------------------------------------------------------- paint
W3 = t.world(np.where(valid[..., None], P, 0))
if a.load_eyes:
    with open(a.load_eyes) as f:
        eyes = [(np.array(e["c"]), e["r"]) for e in json.load(f)]
else:
    eyes = []
    ctr = (t.bmin + t.bmax) / 2
    for e in best:
        # eyes are domes: the centre is the outermost point in the eye's gaze direction (away from the head,
        # horizontal), near the axis through the centroid. The centroid itself lies inside.
        q = t.world(pts[e["idx"]])
        out = e["cen"] - ctr
        out[1] = 0
        out /= np.linalg.norm(out) + 1e-9
        rel = q - e["cen"]
        along = rel @ out
        lat = np.linalg.norm(rel - along[:, None] * out, axis=1)
        near_axis = lat < 0.5 * e["r"]
        k = np.argmax(np.where(near_axis, along, -1e9))
        eyes.append((q[k], e["r"]))
    rmin = min(r for _, r in eyes)
    eyes = [(c, min(r, 1.25 * rmin)) for c, r in eyes]  # both eyes the same size (cap merged spots)
if a.save_eyes:
    with open(a.save_eyes, "w") as f:
        json.dump([dict(c=[float(v) for v in c], r=float(r)) for c, r in eyes], f)
up = np.array([0, 1, 0.0])
for c, r in eyes:
    d = np.linalg.norm(W3 - c, axis=2)
    d[~valid] = 1e9
    near = d < 1.6 * r
    side = np.cross(up, c - (t.bmin + t.bmax) / 2)
    side /= np.linalg.norm(side) + 1e-9
    g = c + up * 0.38 * r + side * 0.25 * r                          # glint top side
    dg = np.linalg.norm(W3 - g, axis=2)
    if a.style == "black":
        # paint dark remains of the old eye outside the new circle in skin color
        ring = (d > 1.15 * r) & (d < 2.0 * r) & (L > a.dark + 10)
        if ring.any():
            skin = np.median(t.rgb[ring], axis=0)
            rest = (d > 0.95 * r) & (d < 1.6 * r) & (L < a.dark + 5)
            t.rgb[rest] = skin
        sclera = (1 - ramp(d, 0.92 * r, 1.05 * r)) * near
        t.rgb += sclera[..., None] * (np.array([246, 245, 242.0]) - t.rgb)
        pup = (1 - ramp(d, a.pupil * r * 0.9, a.pupil * r * 1.05)) * near
        t.rgb += pup[..., None] * (np.array([14, 12, 14.0]) - t.rgb)
        gl = (1 - ramp(dg, 0.12 * r, 0.17 * r)) * near
    else:
        gl = (1 - ramp(dg, a.glint * r * 0.85, a.glint * r)) * near
        gl2 = (1 - ramp(np.linalg.norm(W3 - (c - up * 0.3 * r - side * 0.3 * r), axis=2), 0.08 * r, 0.12 * r)) * near
        gl = np.maximum(gl, gl2)
    t.rgb += gl[..., None] * (np.array([255, 255, 255.0]) - t.rgb)
    if t.mr is not None:  # glossy eyes
        t.mr[..., 1] -= (d < r) * 0.6 * t.mr[..., 1]
t.save(a.dst)
print(f"[tex_eyes] {a.dst}: style {a.style}")
