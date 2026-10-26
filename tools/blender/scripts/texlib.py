"""Shared helpers of the texture tools (tex_adjust, tex_eyes, tex_eyecover, tex_hair): load/save a GLB, texel ->
3D position (position map), soft ramps. Geometry and UVs are never changed. Plain Python (trimesh, Pillow),
no Blender."""
import numpy as np
import trimesh
from PIL import Image


def ramp(x, a0, a1):
    return np.clip((x - a0) / (a1 - a0), 0, 1)


class Tex:
    """A GLB with exactly one mesh; edit .rgb (float HxWx3) and .mr (metallic/roughness, float), then .save()."""

    def __init__(self, path):
        self.scene = trimesh.load(path, process=False)
        self.g = list(self.scene.geometry.values())[0]
        self.m = self.g.visual.material
        rgba = np.asarray(self.m.baseColorTexture.convert("RGBA"), float)
        self.rgb, self.alpha = rgba[..., :3].copy(), rgba[..., 3:]
        self.H, self.W = self.rgb.shape[:2]
        mr = self.m.metallicRoughnessTexture
        self.mr = np.asarray(mr.convert("RGB"), float) if mr is not None else None
        V = self.g.vertices
        self.bmin, self.bmax = V.min(0), V.max(0)
        self._P = None

    @property
    def P(self):
        """Position map: per texel the 3D point normalised to the bounding box (0..1, glTF: y up, z front),
        -1 = empty."""
        if self._P is None:
            self._P = position_map(self.g, self.W, self.H, self.bmin, self.bmax)
        return self._P

    def world(self, Pn):
        """Normalised coordinates -> model coordinates (for isotropic distances)."""
        return self.bmin + Pn * (self.bmax - self.bmin)

    def save(self, path):
        rgba = np.concatenate([np.clip(self.rgb, 0, 255), self.alpha], 2).astype(np.uint8)
        self.m.baseColorTexture = Image.fromarray(rgba, "RGBA")
        if self.mr is not None:
            self.m.metallicRoughnessTexture = Image.fromarray(np.clip(self.mr, 0, 255).astype(np.uint8), "RGB")
        self.scene.export(path)


def position_map(g, W, H, bmin, bmax):
    Vn = (g.vertices - bmin) / (bmax - bmin)
    uv = g.visual.uv.copy()
    uv[:, 0] *= W
    uv[:, 1] = (1 - uv[:, 1]) * H
    pos = np.full((H, W, 3), np.nan)
    for f in g.faces:
        t = uv[f]
        x0, y0 = np.floor(t.min(0)).astype(int)
        x1, y1 = np.ceil(t.max(0)).astype(int)
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, W - 1), min(y1, H - 1)
        if x1 < x0 or y1 < y0:
            continue
        xs, ys = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
        (ax, ay), (bx, by), (cx, cy) = t
        d = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        if abs(d) < 1e-12:
            continue
        l1 = ((by - cy) * (xs - cx) + (cx - bx) * (ys - cy)) / d
        l2 = ((cy - ay) * (xs - cx) + (ax - cx) * (ys - cy)) / d
        l3 = 1 - l1 - l2
        ins = (l1 >= -0.01) & (l2 >= -0.01) & (l3 >= -0.01)
        if not ins.any():
            continue
        Pt = l1[..., None] * Vn[f[0]] + l2[..., None] * Vn[f[1]] + l3[..., None] * Vn[f[2]]
        sub = pos[y0:y1 + 1, x0:x1 + 1]
        sub[ins] = Pt[ins]
    for _ in range(12):  # fill the borders (inpainting area) by 12 px, so that seams get recolored too
        nan = np.isnan(pos[..., 0])
        if not nan.any():
            break
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            sh = np.roll(pos, (dy, dx), axis=(0, 1))
            fill = nan & ~np.isnan(sh[..., 0])
            pos[fill] = sh[fill]
            nan = np.isnan(pos[..., 0])
    pos[np.isnan(pos)] = -1
    return pos


def region_weight(P, reg):
    """Soft box 'x0,x1,y0,y1,z0,z1' (fractions of the bounding box)."""
    b = [float(v) for v in reg.split(",")]
    w = np.ones(P.shape[:2])
    for k in range(3):
        w *= ramp(P[..., k], b[2 * k] - 0.03, b[2 * k]) * (1 - ramp(P[..., k], b[2 * k + 1], b[2 * k + 1] + 0.03))
    return w


def value_noise3(p, seed=0):
    """Smooth 3D value noise for points p (N x 3), values 0..1."""
    i = np.floor(p).astype(np.int64)
    f = p - i
    f = f * f * (3 - 2 * f)

    def h(ix, iy, iz):
        n = (ix * 374761393 + iy * 668265263 + iz * 2147483647 + seed * 974634583) & 0xFFFFFFFF
        n = (n ^ (n >> 13)) * 1274126177 & 0xFFFFFFFF
        return (n & 0xFFFF) / 65535.0

    out = 0
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                w = ((f[:, 0] if dx else 1 - f[:, 0]) * (f[:, 1] if dy else 1 - f[:, 1])
                     * (f[:, 2] if dz else 1 - f[:, 2]))
                out = out + w * h(i[:, 0] + dx, i[:, 1] + dy, i[:, 2] + dz)
    return out
