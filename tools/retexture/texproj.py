"""Texture projection for retexturing (numpy and Pillow only): from generated views onto the model's own UVs.

Inputs are the arrays of blender_views.py. The texture is kept per covered texel: color (sRGB 0..1) and weight,
the cosine between the texel's normal and the direction to the camera of the view it came from (0 = not painted
yet). A later view paints only texels that no earlier view has seen well (Paint3D/TEXTure style): what is known
is shown to the image model as fixed context, the rest is inpainted, so the views stay consistent.
"""
import numpy as np
from PIL import Image

KNOWN = 0.5        # a texel painted at this cosine or better counts as known in later views
MIN_COS = 0.12     # grazing views paint nothing
EDGE_PX, EDGE_WEIGHT = 2, 0.3   # pixels this close to the silhouette or a depth jump count less (colour bleeds)


class Geometry:
    def __init__(self, texels, views, info):
        cover = texels["cover"]
        self.T = cover.shape[0]
        self.idx = np.flatnonzero(cover)                         # covered texels, flat index into (T,T)
        self.P = texels["pos"].reshape(-1, 3)[self.idx].astype(np.float64)
        self.N = texels["nrm"].reshape(-1, 3)[self.idx].astype(np.float64)
        self.uv, self.depth = views["uv"], views["depth"]
        self.R, self.eye, self.f = views["R"], views["eye"], float(views["f"])
        self.H, self.W = self.depth.shape[1:]
        self.views = [tuple(v) for v in info["views"]]
        self.extent = float(np.linalg.norm(np.subtract(info["bmax"], info["bmin"])))
        # texel id per (T,T) cell, -1 where no face is
        self.cell = np.full(self.T * self.T, -1, np.int64)
        self.cell[self.idx] = np.arange(len(self.idx))
        self._edges = {}

    def edges(self, i):
        """(H,W) pixels of view i within EDGE_PX of the background or of a jump in depth."""
        if i not in self._edges:
            d = self.depth[i]
            jump = d <= 0
            step = 0.02 * self.extent
            for ax in (0, 1):
                g = np.abs(np.diff(d, axis=ax)) > step
                pad = [(0, 0), (0, 0)]
                pad[ax] = (0, 1)
                jump |= np.pad(g, pad)
                pad[ax] = (1, 0)
                jump |= np.pad(g, pad)
            e = jump.copy()
            for _ in range(EDGE_PX):
                e[1:] |= e[:-1].copy()
                e[:-1] |= e[1:].copy()
                e[:, 1:] |= e[:, :-1].copy()
                e[:, :-1] |= e[:, 1:].copy()
            self._edges[i] = e & (d > 0)
        return self._edges[i]

    # --- texels seen from view i
    def project(self, i):
        """Pixel coordinates (x, y; pixel centres at integers), depth and cosine of every texel in view i."""
        d = self.P - self.eye[i]
        c = d @ self.R[i].T
        z = -c[:, 2]
        zs = np.maximum(z, 1e-9)
        x = self.W / 2 + self.f * c[:, 0] / zs - 0.5
        y = self.H / 2 - self.f * c[:, 1] / zs - 0.5
        cos = -(self.N * d).sum(1) / np.maximum(np.linalg.norm(d, axis=1), 1e-12)
        return x, y, z, cos

    def visible(self, i, x, y, z, cos):
        """Texels whose surface is what view i shows at their pixel (depth test against the render)."""
        xi, yi = np.rint(x).astype(np.int64), np.rint(y).astype(np.int64)
        inside = (xi >= 0) & (xi < self.W) & (yi >= 0) & (yi < self.H) & (z > 0)
        D = np.zeros_like(z)
        D[inside] = self.depth[i][yi[inside], xi[inside]]
        # tolerance: a little of the model size plus the depth change across ~1.5 pixels at this slant
        tol = 0.004 * self.extent + 1.5 * z / self.f / np.maximum(cos, 0.15)
        return inside & (D > 0) & (np.abs(z - D) < tol) & (cos > MIN_COS)

    # --- per pixel of view i: which texel it shows
    def pixel_texels(self, i):
        """(H,W) texel id (-1: background or a texel outside the coverage) for each pixel of view i."""
        uv = self.uv[i]
        hit = uv[..., 0] >= 0
        col = np.clip((uv[..., 0] % 1.0) * self.T, 0, self.T - 1).astype(np.int64)
        row = np.clip((uv[..., 1] % 1.0) * self.T, 0, self.T - 1).astype(np.int64)
        tid = np.where(hit, self.cell[row * self.T + col], -1)
        # a pixel centre may land on a margin texel next to its face: take a covered neighbour
        for dr, dc in ((0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)):
            miss = hit & (tid < 0)
            if not miss.any():
                break
            r2, c2 = np.clip(row + dr, 0, self.T - 1), np.clip(col + dc, 0, self.T - 1)
            tid[miss] = self.cell[r2[miss] * self.T + c2[miss]]
        return tid


class Texture:
    def __init__(self, geo):
        self.geo = geo
        self.color = np.zeros((len(geo.idx), 3))
        self.weight = np.zeros(len(geo.idx))

    def view_state(self, i):
        """What view i shows of the current texture: image (H,W,3), known (H,W) bool, object (H,W) bool,
        normals (H,W,3) and cosines (H,W) of the pixels."""
        g = self.geo
        tid = g.pixel_texels(i)
        obj = tid >= 0
        t = tid[obj]
        img = np.zeros((g.H, g.W, 3))
        img[obj] = self.color[t]
        P, N = g.P[t], g.N[t]
        d = P - g.eye[i]
        cos = np.zeros((g.H, g.W))
        cos[obj] = -(N * d).sum(1) / np.maximum(np.linalg.norm(d, axis=1), 1e-12)
        cos = np.where(g.edges(i), cos * EDGE_WEIGHT, cos)   # as paint() rates these pixels
        w = np.zeros((g.H, g.W))
        w[obj] = self.weight[t]
        known = obj & (w > 0) & (w >= np.minimum(KNOWN, cos))
        nrm = np.zeros((g.H, g.W, 3))
        nrm[obj] = N
        return img, known, obj, nrm, cos

    def paint(self, i, image):
        """Bake view i's image (H',W',3 float 0..1, any multiple of the view size) into the texels it sees
        better than any view before, except the known ones. Returns the number of texels painted."""
        g = self.geo
        x, y, z, cos = g.project(i)
        vis = g.visible(i, x, y, z, cos)
        xi = np.clip(np.rint(x).astype(np.int64), 0, g.W - 1)
        yi = np.clip(np.rint(y).astype(np.int64), 0, g.H - 1)
        cos = np.where(g.edges(i)[yi, xi], cos * EDGE_WEIGHT, cos)
        known = (self.weight > 0) & (self.weight >= np.minimum(KNOWN, cos))
        sel = vis & ~known & (cos > self.weight)
        s = image.shape[0] / g.H
        self.color[sel] = bilinear(image, (x[sel] + 0.5) * s - 0.5, (y[sel] + 0.5) * s - 0.5)
        self.weight[sel] = cos[sel]
        return int(sel.sum())

    def image(self, fill=True):
        """The texture as (T,T,3) float, unpainted texels and the margins around UV islands filled from their
        neighbourhood (push-pull), so that seams and texture filtering pick up no black."""
        g = self.geo
        tex = np.zeros((g.T * g.T, 3))
        have = np.zeros(g.T * g.T)
        tex[g.idx] = self.color
        have[g.idx] = self.weight > 0
        tex, have = tex.reshape(g.T, g.T, 3), have.reshape(g.T, g.T)
        return push_pull(tex, have) if fill else tex


def bilinear(img, x, y):
    H, W = img.shape[:2]
    x, y = np.clip(x, 0, W - 1), np.clip(y, 0, H - 1)
    x0, y0 = np.minimum(np.floor(x).astype(np.int64), W - 2), np.minimum(np.floor(y).astype(np.int64), H - 2)
    fx, fy = (x - x0)[:, None], (y - y0)[:, None]
    return ((img[y0, x0] * (1 - fx) + img[y0, x0 + 1] * fx) * (1 - fy)
            + (img[y0 + 1, x0] * (1 - fx) + img[y0 + 1, x0 + 1] * fx) * fy)


def push_pull(img, have):
    """Fill the pixels without a value (have == 0) from the pixels around them, at every scale."""
    if have.all() or not have.any():
        return img
    levels = [(img * have[..., None], have.astype(float))]
    while min(levels[-1][1].shape) > 1:
        c, w = levels[-1]
        h2, w2 = (c.shape[0] + 1) // 2 * 2, (c.shape[1] + 1) // 2 * 2
        c = np.pad(c, ((0, h2 - c.shape[0]), (0, w2 - c.shape[1]), (0, 0)))
        w = np.pad(w, ((0, h2 - w.shape[0]), (0, w2 - w.shape[1])))
        c = c.reshape(h2 // 2, 2, w2 // 2, 2, 3).sum((1, 3))
        w = w.reshape(h2 // 2, 2, w2 // 2, 2).sum((1, 3))
        levels.append((c, w))
    c, w = levels[-1]
    filled = c / np.maximum(w, 1e-12)[..., None]
    for c, w in reversed(levels[:-1]):
        up = np.repeat(np.repeat(filled, 2, 0), 2, 1)[:c.shape[0], :c.shape[1]]
        own = c / np.maximum(w, 1e-12)[..., None]
        a = np.clip(w, 0, 1)[..., None]
        filled = own * a + up * (1 - a)
    return np.where(have[..., None] > 0, img, filled)


def srgb_to_linear(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c):
    c = np.clip(c, 0, None)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


def sh_basis(n):
    x, y, z = n[:, 0], n[:, 1], n[:, 2]
    return np.stack([np.ones_like(x), x, y, z, x * y, y * z, 3 * z * z - 1, x * z, x * x - y * y], 1)


def delight(img, nrm, mask, strength=1.0, order=2):
    """Remove the lighting the image model painted in: fit log luminance over the masked pixels as a
    spherical-harmonics function of their normals (low-frequency light) and divide it out. Returns the image
    and the shading factor per pixel (1 outside the mask)."""
    shading = np.ones(img.shape[:2])
    lin = srgb_to_linear(img[mask])
    lum = lin @ np.array([0.2126, 0.7152, 0.0722])
    ok = lum > 0.01
    if ok.sum() < 2000:
        return img, shading
    k = 4 if order == 1 else 9
    A = sh_basis(nrm[mask])[:, :k]
    b = np.log(lum[ok])
    coef = np.linalg.lstsq(A[ok], b, rcond=None)[0]
    s = (A[:, 1:] @ coef[1:]) * strength
    s -= np.average(s[ok], weights=lum[ok])          # keep the mean brightness
    f = np.clip(np.exp(s), 0.5, 2.0)
    out = img.copy()
    out[mask] = linear_to_srgb(lin / f[:, None])
    shading[mask] = f
    return np.clip(out, 0, 1), shading


def match_colors(gen, ref, overlap, apply):
    """Fit gen to ref over the overlap pixels (both paint the same surface there) per channel, gain and offset in
    linear light, and apply it to the apply pixels: a new view takes on the exposure and tint of the paint it
    continues, so no seam shows where views meet."""
    if overlap.sum() < 500:
        return gen
    g, r = srgb_to_linear(gen[overlap]), srgb_to_linear(ref[overlap])
    out = gen.copy()
    lin = srgb_to_linear(gen[apply])
    for c in range(3):
        gain = np.clip(r[:, c].std() / max(g[:, c].std(), 1e-6), 0.7, 1.4)
        off = np.clip(r[:, c].mean() - gain * g[:, c].mean(), -0.15, 0.15)
        lin[:, c] = lin[:, c] * gain + off
    out[apply] = linear_to_srgb(lin)
    return np.clip(out, 0, 1)


def depth_control(depth, near=1.0, far=0.15):
    """Depth as the ControlNet expects it: near white, far dark, background black; the object spans
    near..far (uint8 HxWx3)."""
    m = depth > 0
    out = np.zeros(depth.shape)
    if m.any():
        lo, hi = depth[m].min(), depth[m].max()
        out[m] = near + (depth[m] - lo) / max(hi - lo, 1e-9) * (far - near)
    return np.repeat((out * 255).round().astype(np.uint8)[..., None], 3, 2)


def to_uint8(img):
    return (np.clip(img, 0, 1) * 255).round().astype(np.uint8)


def save(img, path):
    Image.fromarray(img if img.dtype == np.uint8 else to_uint8(img)).save(path)


def load(path):
    return np.asarray(Image.open(path).convert("RGB"), float) / 255
