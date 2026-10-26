"""Insert eyes as geometry: white eyeball, glossy black pupil, two glints.

  eye_geo in.glb out.glb --eyes eyes.json [--size 1.15] [--pupil 0.72] [--look 0.15] [--apart 0] [--back 0]

eyes.json: [{"c": [x,y,z], "r": r}, ...] in the local mesh coordinates of the GLB (glTF, without node
transform), as tex_eyes --save-eyes writes them. The domes sit on top of the painted eyes and cover them.
Gaze direction: estimated from the eyes' positions relative to each other (forward, turned outwards by --spread
degrees), then averaged with the matching face normals around the eye centre (generated normals partly point
inwards); --look moves the pupil towards the figure's front.
Moving: --drop (down), --apart (apart), --back (towards the back of the head, along the head's gaze direction),
--right (both to the right as seen by the figure), all in eyeball radii; the new centre is put onto the head's
surface. A handy scale: 10 units = 0.5 R.
Every part is a half sphere (the back half would be inside the head anyway), ~330 triangles per eye.
The eyes stay separate objects named "Eye*" (the rigs attach them rigidly to the head; rig_hop joins all meshes).
"""
import argparse
import json

import bmesh
import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

p = argparse.ArgumentParser(prog="eye_geo")
p.add_argument("src")
p.add_argument("dst")
p.add_argument("--eyes", required=True)
p.add_argument("--size", type=float, default=1.15, help="eyeball radius x detected radius")
p.add_argument("--flat", type=float, default=0.45, help="depth of the dome (fraction of the radius)")
p.add_argument("--pupil", type=float, default=0.72, help="pupil radius x eyeball radius")
p.add_argument("--spread", type=float, default=30, help="eyes turned outwards (degrees)")
p.add_argument("--drop", type=float, default=0.0, help="move the eyes down (x radius)")
p.add_argument("--sink", type=float, default=0.25, help="sink the eyeball this deep into the head (x radius)")
p.add_argument("--apart", type=float, default=0.0, help="move the eyes apart (+) / together (-) (x radius)")
p.add_argument("--back", type=float, default=0.0, help="move the eyes back (+) / forward (-) on the head (x radius)")
p.add_argument("--right", type=float, default=0.0,
               help="both eyes to the right (+) / left (-) as seen by the figure (x radius)")
p.add_argument("--level", action="store_true", help="both eyes at the same height (mean)")
p.add_argument("--look", type=float, default=0.15, help="move the pupil towards the front (x radius)")
a = p.parse_args()

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=a.src)
body = [o for o in bpy.context.scene.objects if o.type == "MESH"][0]
bpy.context.view_layer.update()
M = body.matrix_world
V = np.array([M @ v.co for v in body.data.vertices])
F = [(M.to_3x3() @ f.normal).normalized() for f in body.data.polygons]
FC = np.array([M @ f.center for f in body.data.polygons])
FA = np.array([f.area for f in body.data.polygons])
bvh = BVHTree.FromPolygons([Vector(v) for v in V], [tuple(f.vertices) for f in body.data.polygons])
front = Vector((0, -1, 0))  # glTF +Z = Blender -Y


def mat(name, rgb, rough, emit=0.0):
    m = bpy.data.materials.new(name)
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*rgb, 1)
    b.inputs["Roughness"].default_value = rough
    if emit:
        b.inputs["Emission Color"].default_value = (*rgb, 1)
        b.inputs["Emission Strength"].default_value = emit
    return m


m_white = mat("Eye_white", (0.92, 0.92, 0.90), 0.3)
m_black = mat("Eye_pupil", (0.005, 0.005, 0.006), 0.2)  # smoother: scene lights reflect as dots
m_glint = mat("Eye_glint", (1, 1, 1), 0.2, emit=1.0)


def dome(name, center, n, radius, depth, material, seg):
    """Half sphere (local z >= 0) with seg segments and seg/2 rings over the full sphere."""
    me = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=seg, v_segments=seg // 2, radius=1)
    bmesh.ops.delete(bm, geom=[v for v in bm.verts if v.co.z < -1e-4], context="VERTS")
    bm.to_mesh(me)
    bm.free()
    me.uv_layers.new(name="UVMap")  # the same UV layer as the body, otherwise it is lost when they are joined
    o = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(o)
    bpy.context.view_layer.objects.active = o
    o.select_set(True)
    o.scale = (radius, radius, depth)
    o.rotation_mode = "QUATERNION"
    o.rotation_quaternion = n.to_track_quat("Z", "Y")
    o.location = center
    bpy.ops.object.shade_smooth()
    o.select_set(False)
    o.data.materials.append(material)
    return o


with open(a.eyes) as f:
    eyes = json.load(f)
C = [M @ Vector((e["c"][0], -e["c"][2], e["c"][1])) for e in eyes]  # glTF local -> Blender world
mid = sum(C, Vector()) / len(C)
if a.level:
    for v in C:
        v.z = mid.z
R = np.mean([e["r"] for e in eyes]) * a.size  # both eyes the same size
for k, e in enumerate(eyes):
    lat = C[k] - mid
    lat.z = 0
    lat.normalize()
    fwd = Vector((0, 0, 1)).cross(lat)
    if fwd.dot(front) < 0:
        fwd = -fwd
    guess = (fwd * np.cos(np.radians(a.spread)) + lat * np.sin(np.radians(a.spread))).normalized()

    def onto(p, d):  # project a point along direction d onto the head's surface
        hit = bvh.ray_cast(p + d * (3 * R), -d, 6 * R)[0]
        return hit if hit is not None else bvh.find_nearest(p)[0]

    c = C[k]
    if a.drop or a.apart or a.right or a.level:
        # sideways/up: project in the front view, so that the move is exact as seen from the front
        c = onto(c - Vector((0, 0, a.drop * R)) + lat * (a.apart * R)
                 + fwd.cross(Vector((0, 0, 1))).normalized() * (a.right * R), fwd)
    if a.back:  # towards the back of the head: back along the eye's gaze, onto the surface again
        c = onto(c - fwd * (a.back * R), guess)
    d = np.linalg.norm(FC - np.array(c), axis=1)
    ok = [i for i in np.nonzero(d < 1.2 * e["r"])[0] if F[i].dot(guess) > 0.3]
    n = (sum((F[i] * FA[i] for i in ok), Vector()).normalized() + guess).normalized() if ok else guess
    # eyeball: flat dome, its centre sunk a little into the head, so that no gap to the fur remains
    base = c - n * (a.sink * R)
    dome(f"Eye{k}_white", base, n, R, a.flat * R, m_white, 20)
    # pupil: moved towards the front, sits just on the eyeball
    t = front - n * front.dot(n)
    t = t.normalized() if t.length > 1e-6 else Vector()
    off = t * (a.look * R)
    s = off.length / R
    h = a.flat * R * (1 - s * s) ** 0.5
    pr = a.pupil * R
    pc = base + off + n * (h - 0.42 * a.flat * pr)
    dome(f"Eye{k}_pupil", pc, n, pr, 0.5 * a.flat * pr, m_black, 16)
    # glints top side (mirrored towards the head's centre, so that both eyes look alike)
    up = Vector((0, 0, 1))
    up = (up - n * up.dot(n)).normalized()
    side = n.cross(up) * (1 if k == 0 else -1)
    for frac, gr in ((0.38, 0.17), (0.3, 0.08)):
        sgn = 1 if gr > 0.1 else -1
        go = up * (sgn * frac * pr) + side * (sgn * 0.3 * pr)
        gc = pc + go + n * (0.5 * a.flat * pr * (1 - (go.length / pr) ** 2) ** 0.5)
        dome(f"Eye{k}_glint", gc, n, gr * pr, 0.35 * gr * pr, m_glint, 8)
    print(f"[eye_geo] eye {k}: centre {tuple(round(v, 3) for v in c)} R={R:.3f} "
          f"normal {tuple(round(v, 2) for v in n)}")

faces = sum(len(o.data.polygons) for o in bpy.data.objects if o.name.startswith("Eye"))
print(f"[eye_geo] eyes: {faces} faces (triangles + quads) for both eyes")
bpy.ops.export_scene.gltf(filepath=a.dst, export_format="GLB", export_image_format="JPEG",
                          export_image_quality=90, export_yup=True)
print(f"[eye_geo] OUT {a.dst}")
