"""Quadruped rig + animations (Walk, Nod, Talk, Idle) for a static GLB.

  rig_quadruped in.glb out.glb --landmarks giraffe|pony|sheep|FILE.json [--fps 24] [--yaw 0] [--height 0]
                [--walk 1.0] [--stride 18] [--knee 35] [--save-blend x.blend] [--dump-landmarks FILE.json]

Expects a model like from glb_post: Z up in Blender (glTF Y up), bottom at z=0, looking towards -Y (glTF +Z).
The bone positions are fractions of the bounding box (x: -1..1 = half width, y: -1..1 = half length, z: 0..1 =
height; head at the front = -y): a built-in profile from LANDMARKS or a JSON file with the same keys
(--dump-landmarks writes a profile as a template). Measure a new animal with a slice analysis of its vertices
(e.g. vertices in z slices; legs = 4 clusters at z < 0.08).

The jaw gets its own weights (region below the mouth line at the snout), since generated models have no open
mouth: "talking" moves the lower snout up and down.

Eyes from eye_geo (objects "Eye*") are attached rigidly to the head. --yaw/--height turn the model about the
vertical axis and scale it to a height before rigging (bottom centred at z=0).
Animations (glTF, loops): Idle (4 s, index 0), Nod (2 s), Talk (3 s), Walk (--walk s, on the spot, diagonal gait).
"""
import argparse
import json
import math

import bmesh
import bpy
import numpy as np
from mathutils import Matrix, Vector

p = argparse.ArgumentParser(prog="rig_quadruped")
p.add_argument("src")
p.add_argument("dst")
p.add_argument("--landmarks", required=True, help="profile from LANDMARKS (giraffe, pony, sheep) or a JSON file")
p.add_argument("--fps", type=int, default=24)
p.add_argument("--yaw", type=float, default=0.0, help="turn about the vertical axis in degrees (before rigging)")
p.add_argument("--height", type=float, default=0.0, help="scale to this height (0 = unchanged)")
p.add_argument("--walk", type=float, default=1.0, help="length of a step cycle in seconds")
p.add_argument("--stride", type=float, default=18.0, help="swing of the upper legs in degrees")
p.add_argument("--knee", type=float, default=35.0, help="bend of the lower legs when lifted, in degrees")
p.add_argument("--save-blend", default=None)
p.add_argument("--dump-landmarks", default=None, help="write the profile as JSON (template for a new animal)")
a = p.parse_args()

# ------------------------------------------------------------------ landmarks (fractions of the bounding box)
# Head at the front = -y. legs: (x, y) of the foot centre; hip/shoulder height "leg_top", knee "knee".
LANDMARKS = {
    # toy giraffe (a chess queen): short body, upright neck, head with a crown, looking ahead
    "giraffe": dict(
        hips=((0, 0.36, 0.36), (0, 0.0, 0.37)),
        chest=((0, 0.0, 0.37), (0, -0.40, 0.40)),
        neck1=((0, -0.17, 0.44), (0, -0.20, 0.56)),
        neck2=((0, -0.20, 0.56), (0, -0.24, 0.67)),
        head=((0, -0.24, 0.67), (0, -0.24, 0.95)),
        jaw=((0, -0.45, 0.70), (0, -0.98, 0.68)),
        ear_L=((0.45, -0.15, 0.86), (0.97, -0.15, 0.87)),
        tail1=((0, 0.55, 0.39), (0, 0.78, 0.39)),
        tail2=((0, 0.78, 0.39), (0, 0.97, 0.41)),
        legs=dict(FL=(0.33, -0.52), BL=(0.33, 0.24)),
        leg_top=0.33, knee=0.14, belly=0.27,
        jaw_region=dict(y_max=-0.50, z_mouth=0.70, blend=0.04),
    ),
    # pony (a chess knight): head front-up, tail hangs back
    "pony": dict(
        hips=((0, 0.40, 0.40), (0, 0.0, 0.40)),
        chest=((0, 0.0, 0.40), (0, -0.45, 0.43)),
        neck1=((0, -0.40, 0.45), (0, -0.38, 0.55)),
        neck2=((0, -0.38, 0.55), (0, -0.36, 0.65)),
        head=((0, -0.36, 0.65), (0, -0.36, 0.95)),
        jaw=((0, -0.60, 0.58), (0, -0.98, 0.55)),
        ear_L=((0.30, -0.25, 0.88), (0.55, -0.25, 0.99)),
        tail1=((0, 0.45, 0.42), (0, 0.75, 0.30)),
        tail2=((0, 0.75, 0.30), (0, 0.95, 0.15)),
        legs=dict(FL=(0.45, -0.52), BL=(0.45, 0.28)),
        leg_top=0.36, knee=0.14, belly=0.27,
        jaw_region=dict(y_max=-0.60, z_mouth=0.58, blend=0.04),
    ),
    # sheep with a mitre (a chess bishop): thick woolly body, short legs, head (with mitre) at the front
    "sheep": dict(
        hips=((0, 0.55, 0.28), (0, 0.10, 0.28)),
        chest=((0, 0.10, 0.28), (0, -0.45, 0.32)),
        neck1=((0, -0.45, 0.38), (0, -0.48, 0.45)),
        neck2=((0, -0.48, 0.45), (0, -0.50, 0.52)),
        head=((0, -0.50, 0.52), (0, -0.50, 0.98)),
        jaw=((0, -0.75, 0.50), (0, -1.0, 0.48)),
        ear_L=((0.55, -0.35, 0.63), (1.0, -0.35, 0.61)),
        tail1=((0, 0.85, 0.30), (0, 0.95, 0.32)),
        tail2=((0, 0.95, 0.32), (0, 1.0, 0.33)),
        legs=dict(FL=(0.28, -0.35), BL=(0.28, 0.48)),
        leg_top=0.20, knee=0.07, belly=0.12,
        jaw_region=dict(y_max=-0.75, z_mouth=0.51, blend=0.03),
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
bpy.ops.object.select_all(action="DESELECT")
mesh.select_set(True)
bpy.context.view_layer.objects.active = mesh
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
# glTF splits the mesh at UV seams (loose islands) -> bone heat fails. Merge them; the UVs are stored per
# loop and survive, the export splits the seams again.
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


def P(f):
    """Landmark (fractions) -> world coordinate."""
    return Vector((ctr[0] + f[0] * half[0], ctr[1] + f[1] * half[1], mn[2] + f[2] * (mx[2] - mn[2])))


# ------------------------------------------------------------------ armature
arm_data = bpy.data.armatures.new("Rig")
arm = bpy.data.objects.new("Rig", arm_data)
bpy.context.scene.collection.objects.link(arm)
bpy.context.view_layer.objects.active = arm
bpy.ops.object.mode_set(mode="EDIT")
eb = arm_data.edit_bones


def bone(name, head, tail, parent=None, deform=True):
    b = eb.new(name)
    b.head, b.tail = head, tail
    b.roll = 0
    b.use_deform = deform
    if parent:
        b.parent = eb[parent]
    return b


bone("root", Vector((ctr[0], ctr[1], 0)), Vector((ctr[0], ctr[1] - 0.15 * half[1], 0)), deform=False)
bone("hips", P(L["hips"][0]), P(L["hips"][1]), "root")
bone("chest", P(L["chest"][0]), P(L["chest"][1]), "hips").use_connect = True
bone("neck1", P(L["neck1"][0]), P(L["neck1"][1]), "chest")
bone("neck2", P(L["neck2"][0]), P(L["neck2"][1]), "neck1").use_connect = True
bone("head", P(L["head"][0]), P(L["head"][1]), "neck2").use_connect = True
bone("jaw", P(L["jaw"][0]), P(L["jaw"][1]), "head")
for side, sx in (("L", 1), ("R", -1)):
    h, t = L["ear_L"]
    bone(f"ear.{side}", P((h[0] * sx, h[1], h[2])), P((t[0] * sx, t[1], t[2])), "head")
bone("tail1", P(L["tail1"][0]), P(L["tail1"][1]), "hips")
bone("tail2", P(L["tail2"][0]), P(L["tail2"][1]), "tail1").use_connect = True
LEGS = {}
for key, (lx, ly) in L["legs"].items():
    for side, sx in (("L", 1), ("R", -1)):
        n = f"{key[0]}{side}"  # FL, FR, BL, BR
        parent = "chest" if key[0] == "F" else "hips"
        x, y = lx * sx, ly
        bone(f"leg_up.{n}", P((x, y, L["leg_top"])), P((x, y, L["knee"])), parent)
        bone(f"leg_lo.{n}", P((x, y, L["knee"])), P((x, y, 0.0)), f"leg_up.{n}").use_connect = True
        LEGS[n] = key[0]
bpy.ops.object.mode_set(mode="OBJECT")

# ------------------------------------------------------------------ skinning
bpy.ops.object.select_all(action="DESELECT")
mesh.select_set(True)
arm.select_set(True)
bpy.context.view_layer.objects.active = arm
bpy.ops.object.parent_set(type="ARMATURE_AUTO")
vg = mesh.vertex_groups
empty = [g.name for g in vg if not any(g.index in [x.group for x in v.groups] for v in mesh.data.vertices)]
if len(vg) == 0 or len(empty) > 3:
    print(f"[rig] automatic weights incomplete ({empty}) -> envelope")
    for g in list(vg):
        vg.remove(g)
    bpy.ops.object.parent_set(type="ARMATURE_ENVELOPE")

# Legs: bone heat gives the leg bones parts of chest/belly too -> the belly skin moves along when they swing.
# Fade the upper legs' weight out towards the top (the rest goes to hips/chest).
z_body = mn[2] + L["leg_top"] * (mx[2] - mn[2])          # hip/shoulder height
z_belly = mn[2] + L.get("belly", 0.15) * (mx[2] - mn[2])  # underside of the body
for n, fb in LEGS.items():
    g_leg = vg[f"leg_up.{n}"]
    g_body = vg["chest" if fb == "F" else "hips"]
    for v in mesh.data.vertices:
        for g in v.groups:
            if g.group == g_leg.index and g.weight > 0:
                k = float(np.clip((z_body - v.co.z) / max(z_body - z_belly, 1e-6), 0, 1))  # 1 low, 0 high
                moved = g.weight * (1 - k)
                if moved > 0:
                    g.weight -= moved
                    g_body.add([v.index], moved, "ADD")

# jaw: its own weights; what the jaw gets is taken from the head
jr = L["jaw_region"]
y_max = ctr[1] + jr["y_max"] * half[1]
z_mouth = mn[2] + jr["z_mouth"] * (mx[2] - mn[2])
bl = jr["blend"] * (mx[2] - mn[2])
g_jaw = vg.get("jaw") or vg.new(name="jaw")
g_head = vg["head"]
n_jaw = 0
for v in mesh.data.vertices:
    wz = np.clip((z_mouth - v.co.z) / bl + 0.5, 0, 1)
    wy = np.clip((y_max - v.co.y) / bl + 0.5, 0, 1)
    w = float(wz * wy)
    # only around the head (below the mouth line there is the neck too)
    if v.co.z < z_mouth - 3 * bl:
        w = 0.0
    if w > 0:
        n_jaw += 1
        for g in v.groups:
            if g.group != g_jaw.index:
                g.weight *= (1 - w)
        g_jaw.add([v.index], w, "REPLACE")
print(f"[rig] jaw vertices: {n_jaw}")

# eyes: rigidly on the head, joined after the skinning (loose half spheres disturb bone heat)
if eyes:
    for o in eyes:
        o.vertex_groups.new(name="head").add(list(range(len(o.data.vertices))), 1.0, "REPLACE")
    bpy.ops.object.mode_set(mode="OBJECT")
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
for b in pb:
    b.rotation_mode = "XYZ"


def sign_toward(bname, world_dir):
    """Sign of a rotation about the local X axis that moves the bone's tail towards world_dir
    (+rotation about X turns local Y towards local Z)."""
    z = arm.data.bones[bname].matrix_local.to_3x3().col[2]
    return 1.0 if z.dot(Vector(world_dir)) >= 0 else -1.0


FWD, BACK, DOWN, UP = (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)


def new_action(name, frames):
    act = bpy.data.actions.new(name)
    act.use_fake_user = True
    arm.animation_data_create()
    arm.animation_data.action = act
    for b in pb:
        b.rotation_euler = (0, 0, 0)
        b.location = (0, 0, 0)
    act.frame_range = (0, frames)
    act.use_frame_range = True
    act.use_cyclic = True
    return act


def key(bname, frame, rx=0.0, ry=0.0, rz=0.0, loc=None):
    b = pb[bname]
    b.rotation_euler = (math.radians(rx), math.radians(ry), math.radians(rz))
    b.keyframe_insert("rotation_euler", frame=frame)
    if loc is not None:
        b.location = loc
        b.keyframe_insert("location", frame=frame)


def wave(bname, frames, amp_x=0.0, amp_y=0.0, amp_z=0.0, phase=0.0, steps=8, harmonic=1, offset=(0, 0, 0)):
    """Closed sine loop over the action's length (amplitudes in degrees)."""
    for i in range(steps + 1):
        f = frames * i / steps
        s = math.sin(2 * math.pi * (harmonic * i / steps + phase))
        key(bname, f, offset[0] + amp_x * s, offset[1] + amp_y * s, offset[2] + amp_z * s)


# ------------------------------------------------------------------ Walk (--walk s, diagonal gait)
F = a.fps
W = round(a.walk * F)
act = new_action("Walk", W)
for n, fb in LEGS.items():
    # diagonal: FL+BR in phase, FR+BL in opposite phase
    ph = 0.0 if n in ("FL", "BR") else 0.5
    sw = sign_toward(f"leg_up.{n}", FWD)      # swing the leg forwards
    kn = sign_toward(f"leg_lo.{n}", BACK)     # lower leg backwards when lifted
    for i in range(17):
        t = i / 16
        f = W * t
        s = math.sin(2 * math.pi * (t + ph))
        lift = max(0.0, math.sin(2 * math.pi * (t + ph + 0.25)))  # lift during the forward swing
        key(f"leg_up.{n}", f, rx=sw * a.stride * s)
        key(f"leg_lo.{n}", f, rx=kn * a.knee * lift)
# body: slight sway, head against it, tail swings
wave("hips", W, amp_y=2.5, harmonic=1)
wave("chest", W, amp_x=sign_toward("chest", UP) * 1.5, harmonic=2)
wave("neck1", W, amp_x=sign_toward("neck1", FWD) * 4, harmonic=2, phase=0.15)
wave("head", W, amp_x=sign_toward("head", FWD) * -3, harmonic=2, phase=0.3)
wave("tail1", W, amp_z=14, harmonic=1, phase=0.2)
wave("tail2", W, amp_z=18, harmonic=1, phase=0.35)
wave("ear.L", W, amp_x=4, harmonic=2)
wave("ear.R", W, amp_x=4, harmonic=2, phase=0.5)

# ------------------------------------------------------------------ Nod (2 s, "yes" twice)
act = new_action("Nod", 2 * F)
sh = sign_toward("head", FWD)
sn = sign_toward("neck2", FWD)
for f, hd, nk in [(0, 0, 0), (0.25, 22, 6), (0.5, -4, -1), (0.75, 20, 5), (1.0, -3, 0), (1.3, 0, 0), (2.0, 0, 0)]:
    key("head", f * F, rx=sh * hd)
    key("neck2", f * F, rx=sn * nk)
for e, ph in (("ear.L", 0), ("ear.R", 0.1)):
    for f, d in [(0, 0), (0.3, 8), (0.8, 8), (1.2, 0), (2.0, 0)]:
        key(e, (f + ph) * F if f < 2 else f * F, rx=d)
wave("tail1", 2 * F, amp_z=6)

# ------------------------------------------------------------------ Talk (3 s, syllable rhythm)
act = new_action("Talk", 3 * F)
sj = sign_toward("jaw", DOWN)
# jaw opening in degrees: irregular syllables, short pauses
syll = [0, 14, 4, 18, 2, 12, 0, 0, 16, 6, 20, 3, 10, 0, 15, 5, 19, 2, 0, 0, 13, 4, 17, 0]
step = 3 * F / len(syll)
for i, d in enumerate(syll + [0]):
    key("jaw", i * step, rx=sj * d)
for f, hx, hz in [(0, 0, 0), (0.6, 4, 5), (1.2, -2, -3), (1.9, 5, 2), (2.5, 1, -5), (3.0, 0, 0)]:
    key("head", f * F, rx=sh * hx, rz=hz)
wave("neck2", 3 * F, amp_z=3, harmonic=1)
wave("ear.L", 3 * F, amp_x=5, harmonic=3)
wave("ear.R", 3 * F, amp_x=5, harmonic=3, phase=0.4)
wave("tail1", 3 * F, amp_z=8, harmonic=2)

# ------------------------------------------------------------------ Idle (4 s, breathing, looking around)
act = new_action("Idle", 4 * F)
wave("chest", 4 * F, amp_x=sign_toward("chest", UP) * 1.2, harmonic=2)
for f, hz, hx in [(0, 0, 0), (1.0, 10, 2), (1.8, 10, 2), (2.6, -6, -1), (3.4, -6, -1), (4.0, 0, 0)]:
    key("neck2", f * F, rz=hz * 0.5)
    key("head", f * F, rz=hz, rx=sh * hx)
wave("tail1", 4 * F, amp_z=10, harmonic=2)
wave("tail2", 4 * F, amp_z=12, harmonic=2, phase=0.2)
for e, ph in (("ear.L", 0.0), ("ear.R", 0.55)):
    for f, d in [(0, 0), (1.5, 0), (1.6, 15), (1.75, 0), (4.0, 0)]:
        key(e, ((f + ph) % 4.0) * F if 0 < f < 4 else f * F, rx=d)
idle = act

# keep Idle active, so that a viewer without animation shows the figure normally
arm.animation_data.action = idle
scene.frame_start, scene.frame_end = 0, 4 * F

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
names = [x.name for x in bpy.data.actions]
print(f"[rig] OUT {a.dst}  bones: {len(arm.data.bones)}  animations: {names}")
