"""Biped rig + animations Idle and Walk for a static GLB.

  rig_biped in.glb out.glb --landmarks elephant|rabbit|FILE.json [--mode walk|hop] [--yaw 0] [--height 0]
            [--walk 0.6] [--stride 0.5] [--hop 0.22] [--fps 24] [--save-blend x.blend] [--dump-landmarks FILE.json]

Expects a model like from glb_post: Z up in Blender, looking towards -Y (glTF +Z). --yaw/--height as in rig_hop
(turn, scale to a height, bottom centred at z=0). The bone positions are fractions of the bounding box (x: -1..1
half width, y: -1..1 half length, z: 0..1 height), the left side (+x) is mirrored: a built-in profile from
LANDMARKS or a JSON file with the same keys (--dump-landmarks writes a profile as a template). Eyes from eye_geo
(objects "Eye*") are attached rigidly to the head.

Legs: upper leg, lower leg, foot. The knee angles are computed per frame from the foot position (analytic
2-bone IK in the side plane), so that the feet stay on the ground when the legs bend; the export has plain
rotation keys.
  walk: walking on the spot (the standing foot glides back, the swinging leg lifts), pelvis bobs, arms swing
        against the legs, head/ears/trunk sway. Moving across a board is the game's job.
  hop:  one rabbit hop per cycle: crouch, push off, pull the legs in the air, cushion the landing, ears follow.
Animations (glTF, loops): Walk (--walk s), Idle (4 s); the rest pose is unanimated.
"""
import argparse
import json
import math

import bmesh
import bpy
import numpy as np
from mathutils import Matrix, Vector

p = argparse.ArgumentParser(prog="rig_biped")
p.add_argument("src")
p.add_argument("dst")
p.add_argument("--landmarks", required=True, help="profile from LANDMARKS (elephant, rabbit) or a JSON file")
p.add_argument("--mode", choices=("walk", "hop"), default="walk")
p.add_argument("--yaw", type=float, default=0.0, help="turn about the vertical axis in degrees (before rigging)")
p.add_argument("--height", type=float, default=0.0, help="scale to this height (0 = unchanged)")
p.add_argument("--walk", type=float, default=0.6, help="length of a Walk cycle (two steps or one hop) in s")
p.add_argument("--stride", type=float, default=0.5, help="step length as a fraction of the leg length (walk)")
p.add_argument("--hop", type=float, default=0.22, help="hop height as a fraction of the figure's height (hop)")
p.add_argument("--fps", type=int, default=24)
p.add_argument("--save-blend", default=None)
p.add_argument("--dump-landmarks", default=None, help="write the profile as JSON (template for a new figure)")
a = p.parse_args()

# ------------------------------------------------------------------ landmarks (fractions of the bounding box)
# Per bone (head, tail). Arms/legs/ears/feet only on the left (+x), the right side is mirrored.
LANDMARKS = {
    # elephant in a cloak (a chess king): short legs, arms at the sides, trunk pointing forward and down
    "elephant": dict(
        hips=((0, 0.05, 0.16), (0, 0.05, 0.34)),
        chest=((0, 0.05, 0.34), (0, 0.05, 0.50)),
        head=((0, 0.05, 0.50), (0, 0.05, 0.88)),
        trunk=[((0, -0.35, 0.55), (0, -0.70, 0.48)), ((0, -0.70, 0.48), (0, -0.98, 0.53))],
        ear_L=((0.35, 0.10, 0.66), (0.90, 0.10, 0.65)),
        arm_up_L=((0.33, -0.05, 0.42), (0.50, -0.10, 0.30)),
        arm_lo_L=((0.50, -0.10, 0.30), (0.58, -0.15, 0.20)),
        leg_up_L=((0.15, -0.08, 0.18), (0.15, -0.10, 0.10)),
        leg_lo_L=((0.15, -0.10, 0.10), (0.15, -0.10, 0.03)),
        foot_L=((0.15, -0.10, 0.03), (0.15, -0.32, 0.01)),
        tail=((0, 0.55, 0.22), (0, 0.92, 0.35)),
        leg_top=0.20, belly=0.12,
    ),
    # rabbit (a chess pawn): upright, big feet pointing forward, holds a carrot with both paws, tail on the ground
    "rabbit": dict(
        hips=((0, -0.10, 0.12), (0, -0.15, 0.35)),
        chest=((0, -0.15, 0.35), (0, -0.25, 0.48)),
        head=((0, -0.25, 0.48), (0, -0.25, 0.80)),
        ear_L=((0.35, -0.25, 0.80), (0.65, -0.30, 0.98)),
        arm_up_L=((0.40, -0.45, 0.40), (0.35, -0.70, 0.32)),
        arm_lo_L=((0.35, -0.70, 0.32), (0.15, -0.80, 0.28)),
        leg_up_L=((0.40, -0.10, 0.16), (0.40, -0.32, 0.10)),
        leg_lo_L=((0.40, -0.32, 0.10), (0.40, -0.25, 0.03)),
        foot_L=((0.40, -0.25, 0.03), (0.40, -0.62, 0.01)),
        tail=((0, 0.35, 0.10), (0, 0.85, 0.08)),
        leg_top=0.18, belly=0.10,
    ),
}
if a.landmarks.endswith(".json"):
    with open(a.landmarks) as f:
        L = json.load(f)
else:
    L = LANDMARKS[a.landmarks]
if a.dump_landmarks:
    with open(a.dump_landmarks, "w") as f:
        json.dump(L, f, indent=1)
    print(f"[rig] landmarks written: {a.dump_landmarks}")
    raise SystemExit(0)

# ------------------------------------------------------------------ import
EYES = ("Eye", "Auge")  # eye_geo objects; "Auge": models from before the port
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=a.src)
meshes = [o for o in bpy.context.scene.objects if o.type == "MESH" and not o.name.startswith(EYES)]
eyes = [o for o in bpy.context.scene.objects if o.type == "MESH" and o.name.startswith(EYES)]
bpy.ops.object.select_all(action="DESELECT")
for o in meshes + eyes:
    o.select_set(True)
bpy.context.view_layer.objects.active = meshes[0]
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
for o in meshes + eyes:
    o.parent = None
if len(meshes) > 1:
    bpy.ops.object.select_all(action="DESELECT")
    for o in meshes:
        o.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    bpy.ops.object.join()
mesh = meshes[0]
# glTF splits the mesh at UV seams -> bone heat fails. Merge them (the UVs are stored per loop and survive).
bm = bmesh.new()
bm.from_mesh(mesh.data)
n0 = len(bm.verts)
bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-5)
bm.to_mesh(mesh.data)
bm.free()
print(f"[rig] vertices merged: {n0} -> {len(mesh.data.vertices)}")

# turn, scale to the height, bottom centred at z=0 (eyes too)
V = np.array([v.co[:] for v in mesh.data.vertices])
mn, mx = V.min(0), V.max(0)
k = a.height / (mx[2] - mn[2]) if a.height > 0 else 1.0
T = (Matrix.Rotation(math.radians(a.yaw), 4, "Z") @ Matrix.Scale(k, 4)
     @ Matrix.Translation((-(mn[0] + mx[0]) / 2, -(mn[1] + mx[1]) / 2, -mn[2])))
for o in [mesh] + eyes:
    o.data.transform(T)
    o.data.update()
V = np.array([v.co[:] for v in mesh.data.vertices])
mn, mx = V.min(0), V.max(0)
half = (mx - mn) / 2
ctr = (mn + mx) / 2
H = mx[2] - mn[2]
print(f"[rig] height {H:.3f}, footprint {mx[0]-mn[0]:.3f} (x) x {mx[1]-mn[1]:.3f} (depth)")


def P(f, sx=1):
    """Landmark (fractions) -> world coordinate; sx=-1 mirrors to the right side."""
    return Vector((ctr[0] + sx * f[0] * half[0], ctr[1] + f[1] * half[1], mn[2] + f[2] * H))


# ------------------------------------------------------------------ armature
arm_data = bpy.data.armatures.new("Rig")
arm = bpy.data.objects.new("Rig", arm_data)
bpy.context.scene.collection.objects.link(arm)
bpy.context.view_layer.objects.active = arm
bpy.ops.object.mode_set(mode="EDIT")
eb = arm_data.edit_bones


def bone(name, ht, parent=None, deform=True, sx=1, connect=False):
    b = eb.new(name)
    b.head, b.tail = P(ht[0], sx), P(ht[1], sx)
    b.roll = 0
    b.use_deform = deform
    if parent:
        b.parent = eb[parent]
        b.use_connect = connect
    return b


b = eb.new("root")
b.head, b.tail, b.use_deform = Vector((ctr[0], ctr[1], 0)), Vector((ctr[0], ctr[1], 0.1 * H)), False
bone("hips", L["hips"], "root")
bone("chest", L["chest"], "hips", connect=True)
bone("head", L["head"], "chest", connect=True)
for i, ht in enumerate(L.get("trunk", [])):
    bone(f"trunk{i + 1}", ht, "head" if i == 0 else f"trunk{i}", connect=i > 0)
if "tail" in L:
    bone("tail", L["tail"], "hips")
SIDES = (("L", 1), ("R", -1))
for side, sx in SIDES:
    if "ear_L" in L:
        bone(f"ear.{side}", L["ear_L"], "head", sx=sx)
    bone(f"arm_up.{side}", L["arm_up_L"], "chest", sx=sx)
    bone(f"arm_lo.{side}", L["arm_lo_L"], f"arm_up.{side}", sx=sx, connect=True)
    bone(f"leg_up.{side}", L["leg_up_L"], "hips", sx=sx)
    bone(f"leg_lo.{side}", L["leg_lo_L"], f"leg_up.{side}", sx=sx, connect=True)
    bone(f"foot.{side}", L["foot_L"], f"leg_lo.{side}", sx=sx, connect=True)
bpy.ops.object.mode_set(mode="OBJECT")

# ------------------------------------------------------------------ skinning
bpy.ops.object.select_all(action="DESELECT")
mesh.select_set(True)
arm.select_set(True)
bpy.context.view_layer.objects.active = arm
bpy.ops.object.parent_set(type="ARMATURE_AUTO")
vg = mesh.vertex_groups
used = {g.group for v in mesh.data.vertices for g in v.groups if g.weight > 0}
empty = [g.name for g in vg if g.index not in used]
if len(vg) == 0 or len(empty) > 3:
    print(f"[rig] automatic weights incomplete ({empty}) -> envelope")
    for g in list(vg):
        vg.remove(g)
    bpy.ops.object.parent_set(type="ARMATURE_ENVELOPE")
elif empty:
    print(f"[rig] bones without weights: {empty}")

# upper legs: bone heat gives them parts of belly/cloak too -> fade out towards the top (the rest goes to hips)
z_body = mn[2] + L["leg_top"] * H
z_belly = mn[2] + L["belly"] * H
g_hips = vg["hips"]
for side, _ in SIDES:
    g_leg = vg[f"leg_up.{side}"]
    for v in mesh.data.vertices:
        for g in v.groups:
            if g.group == g_leg.index and g.weight > 0:
                k = float(np.clip((z_body - v.co.z) / max(z_body - z_belly, 1e-6), 0, 1))  # 1 low, 0 high
                moved = g.weight * (1 - k)
                if moved > 0:
                    g.weight -= moved
                    g_hips.add([v.index], moved, "ADD")

# eyes: rigidly on the head, joined after the skinning (loose half spheres disturb bone heat)
if eyes:
    for o in eyes:
        o.vertex_groups.new(name="head").add(list(range(len(o.data.vertices))), 1.0, "REPLACE")
    bpy.ops.object.select_all(action="DESELECT")
    for o in eyes:
        o.select_set(True)
    mesh.select_set(True)
    bpy.context.view_layer.objects.active = mesh
    bpy.ops.object.join()
    print(f"[rig] eyes on the head: {len(eyes)} parts")

# ------------------------------------------------------------------ animation helpers
scene = bpy.context.scene
scene.render.fps = a.fps
pb = arm.pose.bones
for bb in pb:
    bb.rotation_mode = "XYZ"
B = arm.data.bones
FWD, UP = Vector((0, -1, 0)), Vector((0, 0, 1))


def sign_toward(bname, world_dir):
    """Sign of a rotation about the local X axis that moves the bone's tail towards world_dir."""
    z = B[bname].matrix_local.to_3x3().col[2]
    return 1.0 if z.dot(Vector(world_dir)) >= 0 else -1.0


def xsign(bname):
    """Sign with which a rotation about world X enters as a rotation about the local X axis."""
    return 1.0 if B[bname].matrix_local.to_3x3().col[0].x >= 0 else -1.0


def to_local(bname, w):
    """World offset -> pose.location of the bone (rest pose space)."""
    return B[bname].matrix_local.to_3x3().inverted() @ Vector(w)


def new_action(name, frames):
    act = bpy.data.actions.new(name)
    act.use_fake_user = True
    arm.animation_data_create()
    arm.animation_data.action = act
    for bb in pb:
        bb.rotation_euler = (0, 0, 0)
        bb.location = (0, 0, 0)
    act.frame_range = (0, frames)
    act.use_frame_range = True
    act.use_cyclic = True
    return act


def key_rot(bname, f, rx=0.0, ry=0.0, rz=0.0):
    if bname not in pb:
        return
    bb = pb[bname]
    bb.rotation_euler = (rx, ry, rz)
    bb.keyframe_insert("rotation_euler", frame=f)


def key_loc(bname, f, w):
    bb = pb[bname]
    bb.location = to_local(bname, w)
    bb.keyframe_insert("location", frame=f)


def ang(v):
    """Angle in the side plane (y, z); a + rotation about world X increases it."""
    return math.atan2(v.z, v.y)


# leg geometry in the rest pose (armature space = world, the armature object is at the origin)
LEG = {}
for side, _ in SIDES:
    hip, knee = B[f"leg_up.{side}"].head_local, B[f"leg_lo.{side}"].head_local
    ankle = B[f"foot.{side}"].head_local
    LEG[side] = dict(hip=hip.copy(), ankle=ankle.copy(), l1=(knee - hip).length, l2=(ankle - knee).length,
                     a1=ang(knee - hip), a2=ang(ankle - knee))
LEGLEN = sum(LEG["L"][k] for k in ("l1", "l2"))


def key_leg(side, f, hip_off, foot_off, toe=0.0):
    """Bend the leg so that the ankle is at rest + foot_off while the hip is moved by hip_off (both world,
    relative to root). toe: tilt the foot in addition (rad, + = toes down)."""
    g = LEG[side]
    hp = g["hip"] + Vector(hip_off)
    tg = g["ankle"] + Vector(foot_off)
    d = Vector((0, tg.y - hp.y, tg.z - hp.z))
    l1, l2 = g["l1"], g["l2"]
    dist = min(max(d.length, abs(l1 - l2) + 1e-4), l1 + l2 - 1e-4)
    c = (l1 * l1 + dist * dist - l2 * l2) / (2 * l1 * dist)
    A = math.acos(max(-1.0, min(1.0, c)))
    base = ang(d)
    best = None
    for s in (1, -1):  # choose the knee pointing forward (-y)
        a1 = base + s * A
        ky = hp.y + l1 * math.cos(a1)
        if best is None or ky < best[1]:
            best = (a1, ky)
    a1 = best[0]
    # target out of reach -> pull it to the reachable distance in its direction
    ty, tz = hp.y + dist * math.cos(base), hp.z + dist * math.sin(base)
    a2 = ang(Vector((0, ty - (hp.y + l1 * math.cos(a1)), tz - (hp.z + l1 * math.sin(a1)))))
    d1 = a1 - g["a1"]
    d2 = (a2 - g["a2"]) - d1
    key_rot(f"leg_up.{side}", f, rx=xsign(f"leg_up.{side}") * d1)
    key_rot(f"leg_lo.{side}", f, rx=xsign(f"leg_lo.{side}") * d2)
    key_rot(f"foot.{side}", f, rx=xsign(f"foot.{side}") * (-(d1 + d2) - toe))


def smooth(t):
    return t * t * (3 - 2 * t)


def interp(keys, u):
    """keys: [(u, value...)] -> values at u, interpolated smoothly (smoothstep)."""
    for (u0, *v0), (u1, *v1) in zip(keys, keys[1:]):
        if u0 <= u <= u1:
            t = smooth((u - u0) / (u1 - u0)) if u1 > u0 else 0.0
            return [x0 + (x1 - x0) * t for x0, x1 in zip(v0, v1)]
    return list(keys[-1][1:])


F = a.fps
sg = {n: sign_toward(n, FWD) for n in pb.keys() if n.startswith(("arm_up", "arm_lo", "chest", "head", "trunk", "hips"))}
ear_up = {s: sign_toward(f"ear.{s}", UP) for s, _ in SIDES} if "ear.L" in pb else {}

# ------------------------------------------------------------------ Walk
N = max(8, round(a.walk * F))
walk = new_action("Walk", N)
if a.mode == "walk":
    stride = a.stride * LEGLEN
    lift = 0.22 * LEGLEN
    crouch = 0.05 * LEGLEN   # knees always slightly bent
    bob = 0.04 * LEGLEN
    beta = 0.6               # share of the stance phase
    for i in range(N + 1):
        t = i / N
        dz = -crouch - bob * (0.5 + 0.5 * math.cos(4 * math.pi * t))  # low when a foot lands
        key_loc("hips", i, (0, 0, dz))
        for side, ph in (("L", 0.0), ("R", 0.5)):
            u = (t + ph) % 1.0
            if u < beta:   # stance: the foot glides from front to back
                fy, fz, toe = -stride / 2 + stride * u / beta, 0.0, 0.0
            else:          # swing: lift and move forward
                v = (u - beta) / (1 - beta)
                fy, fz = stride / 2 - stride * smooth(v), lift * math.sin(math.pi * v)
                toe = 0.25 * math.sin(math.pi * v)
            key_leg(side, i, (0, 0, dz), (0, fy, fz), toe)
        s = math.sin(2 * math.pi * t)   # +: left leg in front
        key_rot("hips", i, rz=math.radians(4) * s)
        key_rot("chest", i, rz=-math.radians(7) * s, ry=math.radians(2.5) * math.sin(2 * math.pi * t))
        key_rot("head", i, rz=math.radians(3) * s, rx=sg["head"] * math.radians(2) * math.cos(4 * math.pi * t))
        for side, sx in SIDES:   # arms against the leg of the same side
            key_rot(f"arm_up.{side}", i, rx=sg[f"arm_up.{side}"] * math.radians(22) * (-s * sx))
            key_rot(f"arm_lo.{side}", i, rx=sg[f"arm_lo.{side}"] * math.radians(10 + 8 * (-s * sx)))
            if ear_up:
                key_rot(f"ear.{side}", i, rx=ear_up[side] * math.radians(8) * math.sin(4 * math.pi * t + sx))
        if "trunk1" in pb:
            key_rot("trunk1", i, rz=math.radians(12) * math.sin(2 * math.pi * t - 0.8))
            key_rot("trunk2", i, rz=math.radians(18) * math.sin(2 * math.pi * t - 1.4),
                    rx=sg["trunk2"] * math.radians(6) * math.sin(4 * math.pi * t))
        if "tail" in pb:
            key_rot("tail", i, rz=math.radians(15) * s)
else:
    # (u, hip dz [leg lengths], hop height [figure heights], foot dy [leg l.], foot dz [leg l.], lean [deg],
    #  ears [deg], toes [rad])
    K = [(0.00, 0.00, 0.00, 0.00, 0.00, 0, 0, 0.0),
         (0.16, -0.28, 0.00, 0.00, 0.00, 10, 6, 0.0),      # crouch
         (0.30, 0.00, 0.12, 0.12, -0.04, 14, -18, 0.5),   # push off, legs stretched backwards
         (0.50, -0.14, 1.00, -0.10, 0.18, 2, -14, 0.1),   # top, legs pulled in
         (0.68, -0.04, 0.45, -0.14, 0.06, -6, 4, -0.2),   # falling, legs forward
         (0.82, -0.30, 0.00, 0.00, 0.00, 8, 16, 0.0),      # cushion the landing, ears swing on
         (1.00, 0.00, 0.00, 0.00, 0.00, 0, 0, 0.0)]
    for i in range(N + 1):
        u = i / N
        hz, lz, fy, fz, lean, ear, toe = interp(K, u)
        dz = hz * LEGLEN
        key_loc("root", i, (0, 0, lz * a.hop * H))
        key_loc("hips", i, (0, 0, dz))
        for side, _ in SIDES:
            key_leg(side, i, (0, 0, dz), (0, fy * LEGLEN, fz * LEGLEN), toe)
        key_rot("hips", i, rx=sg["hips"] * math.radians(lean * 0.4))
        key_rot("chest", i, rx=sg["chest"] * math.radians(lean * 0.6))
        key_rot("head", i, rx=sg["head"] * -math.radians(lean * 0.5))
        for side, sx in SIDES:
            if ear_up:
                key_rot(f"ear.{side}", i, rx=-ear_up[side] * math.radians(ear))
            key_rot(f"arm_up.{side}", i, rx=sg[f"arm_up.{side}"] * math.radians(lean * 0.5))
        if "tail" in pb:
            key_rot("tail", i, rx=math.radians(ear * 0.8))

# ------------------------------------------------------------------ Idle (4 s, breathing, looking around)
NI = 4 * F
idle = new_action("Idle", NI)
for i in range(NI + 1):
    t = i / NI
    br = math.sin(4 * math.pi * t)   # two breaths
    key_rot("chest", i, rx=sg["chest"] * -math.radians(1.5) * br)
    look = interp([(0, 0), (0.2, 0), (0.32, 1), (0.48, 1), (0.6, -1), (0.76, -1), (0.88, 0), (1, 0)], t)[0]
    key_rot("head", i, rz=math.radians(12) * look, rx=sg["head"] * math.radians(2) * br)
    key_rot("hips", i, rz=math.radians(2) * math.sin(2 * math.pi * t))
    for side, sx in SIDES:
        key_rot(f"arm_up.{side}", i, rx=sg[f"arm_up.{side}"] * math.radians(3) * br)
        key_leg(side, i, (0, 0, -0.03 * LEGLEN * (0.5 + 0.5 * br)), (0, 0, 0))
        if ear_up:
            tw = math.exp(-((t - (0.25 if side == "L" else 0.7)) / 0.02) ** 2)  # short ear twitch
            key_rot(f"ear.{side}", i, rx=ear_up[side] * math.radians(14) * tw + ear_up[side] * math.radians(3) * br)
    key_loc("hips", i, (0, 0, -0.03 * LEGLEN * (0.5 + 0.5 * br)))
    if "trunk1" in pb:
        key_rot("trunk1", i, rz=math.radians(8) * math.sin(2 * math.pi * t))
        key_rot("trunk2", i, rz=math.radians(12) * math.sin(2 * math.pi * t - 0.7),
                rx=sg["trunk2"] * math.radians(10) * math.sin(4 * math.pi * t))
    if "tail" in pb:
        key_rot("tail", i, rz=math.radians(10) * math.sin(4 * math.pi * t))

# rest pose: no active action (viewers show the figure unmoved)
arm.animation_data.action = None
for bb in pb:
    bb.location, bb.rotation_euler = (0, 0, 0), (0, 0, 0)
scene.frame_start, scene.frame_end = 0, NI

if a.save_blend:
    bpy.ops.wm.save_as_mainfile(filepath=a.save_blend)

bpy.ops.object.select_all(action="SELECT")
bpy.ops.export_scene.gltf(
    filepath=a.dst,
    export_format="GLB",
    export_image_format="JPEG",
    export_image_quality=90,
    export_yup=True,
    export_skins=True,
    export_animations=True,
    export_animation_mode="ACTIONS",
    export_force_sampling=True,
    export_frame_range=False,
)
print(f"[rig] OUT {a.dst}  bones: {len(arm.data.bones)}  animations: {[x.name for x in bpy.data.actions]}")
