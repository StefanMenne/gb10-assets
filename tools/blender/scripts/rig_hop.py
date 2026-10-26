"""Simple hop rig for any figure (e.g. chess pieces) + animations Idle and Walk.

  rig_hop in.glb out.glb [--yaw 0] [--height 0] [--fps 24] [--hop 0.22] [--split 0.5] [--save-blend x.blend]

Works whatever the anatomy (four, two or no legs): instead of legs the whole figure moves. 3 bones, weights by
height (no bone heat):
  root  (bottom, not deforming)        -> hop height
  body  (ground to --split)            -> squash & stretch, pivot on the ground
  top   (--split to the top)           -> leaning/swaying of upper body, head, hat
Animations (glTF, loops): Idle (2 s, breathing + gentle swaying), Walk (0.6 s, one hop on the spot; moving
across a board is the game's job). The rest pose is unanimated.

Expects a model from glb_post (bottom at z=0). --yaw turns the figure about the vertical axis first, so that it
looks forward (glTF +Z, Blender -Y); --height scales it to that height.
"""
import argparse
import math

import bpy
import numpy as np
from mathutils import Matrix, Vector

p = argparse.ArgumentParser(prog="rig_hop")
p.add_argument("src")
p.add_argument("dst")
p.add_argument("--yaw", type=float, default=0.0, help="turn about the vertical axis in degrees (before rigging)")
p.add_argument("--height", type=float, default=0.0, help="scale to this height (0 = unchanged)")
p.add_argument("--fps", type=int, default=24)
p.add_argument("--hop", type=float, default=0.22, help="hop height as a fraction of the figure's height")
p.add_argument("--split", type=float, default=0.5, help="height (fraction) of the body/top border")
p.add_argument("--save-blend", default=None)
a = p.parse_args()

# ------------------------------------------------------------------ import
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
bpy.ops.import_scene.gltf(filepath=a.src)
meshes = [o for o in scene.objects if o.type == "MESH"]
bpy.ops.object.select_all(action="DESELECT")
for o in meshes:
    o.select_set(True)
bpy.context.view_layer.objects.active = max(meshes, key=lambda o: len(o.data.vertices))  # the body, not props
if len(meshes) > 1:
    bpy.ops.object.join()
mesh = bpy.context.view_layer.objects.active
mesh.rotation_mode = "XYZ"
mesh.rotation_euler[2] += math.radians(a.yaw)
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

V = np.array([v.co[:] for v in mesh.data.vertices])
mn, mx = V.min(0), V.max(0)
H = mx[2] - mn[2]
cx, cy = (mn[0] + mx[0]) / 2, (mn[1] + mx[1]) / 2
# bottom exactly at z=0 and centred, so that the figure stands on its square
k = a.height / H if a.height > 0 else 1.0
mesh.data.transform(Matrix.Scale(k, 4) @ Matrix.Translation((-cx, -cy, -mn[2])))
H, mn, mx = H * k, mn * k, mx * k
mesh.data.update()
V = np.array([v.co[:] for v in mesh.data.vertices])
zs = V[:, 2]
print(f"[rig] {len(V)} vertices, height {H:.3f}, footprint {mx[0]-mn[0]:.3f} (x) x {mx[1]-mn[1]:.3f} (depth)")

# ------------------------------------------------------------------ armature
arm_data = bpy.data.armatures.new("Rig")
arm = bpy.data.objects.new("Rig", arm_data)
scene.collection.objects.link(arm)
bpy.context.view_layer.objects.active = arm
bpy.ops.object.mode_set(mode="EDIT")
eb = arm_data.edit_bones


def bone(name, z0, z1, parent=None, deform=True):
    b = eb.new(name)
    b.head, b.tail = Vector((0, 0, z0)), Vector((0, 0, z1))
    b.roll = 0
    b.use_deform = deform
    if parent:
        b.parent = eb[parent]
    return b


zs_split = a.split * H
bone("root", 0, 0.1 * H, deform=False)
bone("body", 0, zs_split, "root")
bone("top", zs_split, H, "body").use_connect = True
bpy.ops.object.mode_set(mode="OBJECT")

# weights by height: soft transition around the border (±15 % of the height)
mesh.parent = arm
mod = mesh.modifiers.new("Armature", "ARMATURE")
mod.object = arm
g_body = mesh.vertex_groups.new(name="body")
g_top = mesh.vertex_groups.new(name="top")
band = 0.15 * H
for i, z in enumerate(zs):
    t = min(max((z - (zs_split - band)) / (2 * band), 0.0), 1.0)
    t = t * t * (3 - 2 * t)  # smoothstep
    if t < 1:
        g_body.add([i], 1 - t, "REPLACE")
    if t > 0:
        g_top.add([i], t, "REPLACE")

scene.render.fps = a.fps
pb = arm.pose.bones
for b in pb:
    b.rotation_mode = "XYZ"
# bone axes: Y = bone direction (world +Z), X = world +X. A +X rotation tilts the tip towards -Y = the front.
fwd = 1.0 if arm.data.bones["top"].matrix_local.to_3x3().col[2].dot(Vector((0, -1, 0))) >= 0 else -1.0


def new_action(name, frames):
    act = bpy.data.actions.new(name)
    act.use_fake_user = True
    arm.animation_data_create()
    arm.animation_data.action = act
    act.frame_range = (0, frames)
    act.use_frame_range = True
    act.use_cyclic = True
    return act


def key(f, body_s=1.0, lean=0.0, top_lean=0.0, sway=0.0, top_sway=0.0, lift=0.0):
    """body_s: stretch (1 = neutral, the volume stays about the same), lean/sway in degrees, lift in figure
    heights."""
    b = pb["body"]
    side = 1 / math.sqrt(body_s)
    b.scale = (side, body_s, side)
    b.rotation_euler = (math.radians(fwd * lean), 0, math.radians(sway))
    b.keyframe_insert("scale", frame=f)
    b.keyframe_insert("rotation_euler", frame=f)
    t = pb["top"]
    t.rotation_euler = (math.radians(fwd * top_lean), 0, math.radians(top_sway))
    t.keyframe_insert("rotation_euler", frame=f)
    r = pb["root"]
    r.location = (0, lift * H, 0)  # the root bone points up: local Y = world Z
    r.keyframe_insert("location", frame=f)


F = a.fps

# ------------------------------------------------------------------ Idle (2 s): breathing + swaying
act_idle = new_action("Idle", 2 * F)
n = 16
for i in range(n + 1):
    ph = 2 * math.pi * i / n
    key(2 * F * i / n,
        body_s=1 + 0.025 * math.sin(2 * ph),          # 2 breaths per cycle
        top_sway=4.0 * math.sin(ph),                   # head sways left/right
        sway=1.0 * math.sin(ph),
        top_lean=1.5 * math.sin(2 * ph + 1.0))

# ------------------------------------------------------------------ Walk (0.6 s): one hop
act_walk = new_action("Walk", round(0.6 * F))
L = round(0.6 * F)
h = a.hop
# (fraction, stretch, lean body, lean top, height)
for u, s, ln, tl, z in [
    (0.00, 1.00, 0, 0, 0.0),        # contact
    (0.15, 0.84, 3, 5, 0.0),        # crouch (run-up)
    (0.30, 1.12, 6, 3, 0.35 * h),   # take-off, stretched
    (0.50, 1.04, 4, -2, 1.00 * h),  # top
    (0.70, 1.08, 0, -4, 0.45 * h),  # falling
    (0.82, 0.86, -2, 4, 0.0),       # landing, cushioning
    (1.00, 1.00, 0, 0, 0.0),        # contact
]:
    key(u * L, body_s=s, lean=ln, top_lean=tl, lift=z)

# soft curves, landing/take-off without overshooting below the ground
for act in (act_idle, act_walk):
    for fc in act.fcurves if hasattr(act, "fcurves") else []:
        for kp in fc.keyframe_points:
            kp.interpolation = "BEZIER"
            kp.handle_left_type = kp.handle_right_type = "AUTO_CLAMPED"

# rest pose: no active action (viewers show the figure unmoved)
arm.animation_data.action = None
for b in pb:
    b.location, b.rotation_euler, b.scale = (0, 0, 0), (0, 0, 0), (1, 1, 1)
scene.frame_start, scene.frame_end = 0, 2 * F

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
print(f"[rig] OUT {a.dst}  animations: {[x.name for x in bpy.data.actions]}")
