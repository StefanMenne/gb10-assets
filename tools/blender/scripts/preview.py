"""Preview of a GLB: renders it from several sides (EEVEE on the GPU) into one PNG, views side by side.

  preview model.glb out.png [--size 400] [--views 4] [--elevation 10] [--action Walk --frame 6]
                            [--engine BLENDER_EEVEE|CYCLES]

The camera circles the model, starting front-left (the front is glTF +Z, Blender -Y), at --elevation degrees,
framed on its bounding box. Neutral studio light: grey world plus a key and a rim light. --action/--frame pose
an animated model (a frame of that action) instead of the rest pose.
"""
import argparse
import math
import os
import tempfile

import bpy
from mathutils import Vector
from PIL import Image

p = argparse.ArgumentParser(prog="preview")
p.add_argument("glb")
p.add_argument("out")
p.add_argument("--size", type=int, default=400, help="width of one view in px (height x1.25)")
p.add_argument("--views", type=int, default=4)
p.add_argument("--elevation", type=float, default=10.0, help="camera height angle in degrees")
p.add_argument("--action", default=None, help="pose the model with this animation")
p.add_argument("--frame", type=float, default=0.0, help="frame of --action")
p.add_argument("--engine", default="BLENDER_EEVEE", choices=["BLENDER_EEVEE", "CYCLES"])
a = p.parse_args()

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
bpy.ops.import_scene.gltf(filepath=a.glb)
if a.action:
    act = bpy.data.actions.get(a.action)
    if act is None:
        raise SystemExit(f"[preview] no action {a.action!r}, available: {[x.name for x in bpy.data.actions]}")
    for o in scene.objects:
        if o.type == "ARMATURE":
            o.animation_data_create()
            o.animation_data.action = act
    scene.frame_set(int(a.frame), subframe=a.frame % 1)
bpy.context.view_layer.update()

# visible only: the glTF importer adds a hidden "Icosphere" (radius 1) as bone shape to rigged models
meshes = [o for o in scene.objects if o.type == "MESH" and o.visible_get()]
pts =[o.matrix_world @ Vector(c) for o in meshes for c in o.bound_box]
mn = Vector([min(p[i] for p in pts) for i in range(3)])
mx = Vector([max(p[i] for p in pts) for i in range(3)])
center, radius = (mn + mx) / 2, (mx - mn).length / 2

world = bpy.data.worlds.new("Studio")
world.use_nodes = True
world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.55, 0.57, 0.6, 1)
world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.8
scene.world = world
for name, energy, rot in (("Key", 3.0, (50, 0, -35)), ("Rim", 1.5, (60, 0, 150))):
    light = bpy.data.objects.new(name, bpy.data.lights.new(name, "SUN"))
    light.data.energy = energy
    light.rotation_euler = [math.radians(v) for v in rot]
    scene.collection.objects.link(light)

cam = bpy.data.objects.new("Cam", bpy.data.cameras.new("Cam"))
cam.data.lens = 50
scene.collection.objects.link(cam)
scene.camera = cam
w, h = a.size, int(a.size * 1.25)
scene.render.engine = a.engine
scene.render.resolution_x, scene.render.resolution_y = w, h
scene.render.film_transparent = False
scene.view_settings.view_transform = "Standard"
if a.engine == "CYCLES":
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "CUDA"
    prefs.get_devices()
    for d in prefs.devices:
        d.use = d.type == "CUDA"
    scene.cycles.device = "GPU"
    scene.cycles.samples = 32
    scene.cycles.use_denoising = True

# distance so that the bounding sphere fits the narrower (horizontal) field of view
fov = 2 * math.atan(cam.data.sensor_width / 2 / cam.data.lens)
dist = radius / math.sin(fov / 2) * 1.05
el = math.radians(a.elevation)
sheet = Image.new("RGB", (w * a.views, h))
with tempfile.TemporaryDirectory() as tmp:
    for i in range(a.views):
        az = math.radians(-40 + 360 * i / a.views)   # 0 = front (-Y), positive = to the figure's left
        d = Vector((math.sin(az) * math.cos(el), -math.cos(az) * math.cos(el), math.sin(el)))
        cam.location = center + d * dist
        cam.rotation_euler = (-d).to_track_quat("-Z", "Y").to_euler()
        scene.render.filepath = os.path.join(tmp, f"{i}.png")
        bpy.ops.render.render(write_still=True)
        sheet.paste(Image.open(scene.render.filepath).convert("RGB"), (w * i, 0))
sheet.save(a.out)
print(f"[preview] OUT {a.out}  ({a.views} views, {a.engine})")
