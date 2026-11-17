"""Depth video of an animated figure, as the control video for generate_video (LTX Union Control IC-LoRA).

  depth_video rigged.glb depth.mp4 [--action Walk] [--seconds 5] [--fps 24] [--res 1024x1024] [--fill 0.78]
              [--turn 0] [--elevation 4] [--color first.png]

The camera looks at the front of the figure (Blender -Y, as rig_* expects), framed on its bounding box in the
first frame: --fill is the figure's height as a fraction of the picture, so it can be matched to the image that
becomes the first frame of the video. The action loops for the whole clip; --turn rotates the figure by that many
degrees (positive: to its left) over the clip, smoothly.

Depth as the annotators of the IC-LoRA training deliver it (Video Depth Anything): near = white, far = black,
background black. near/far come from the deformed vertices over the whole clip, so the figure spans the grey range
and the values mean the same in every frame. Frames: 8k+1 (what LTX needs), H.264 near lossless.
--color writes frame 0 lit in color (to compare framing with the source image, or as the first frame).
"""
import argparse
import math
import os
import shutil
import subprocess
import tempfile

import bpy
import imageio_ffmpeg
from mathutils import Vector

p = argparse.ArgumentParser(prog="depth_video")
p.add_argument("glb")
p.add_argument("out")
p.add_argument("--action", default="Walk")
p.add_argument("--seconds", type=float, default=5.0)
p.add_argument("--fps", type=int, default=24)
p.add_argument("--res", default="1024x1024", help="WxH, multiples of 64 (the video size)")
p.add_argument("--fill", type=float, default=0.78, help="figure height as a fraction of the picture height")
p.add_argument("--turn", type=float, default=0.0, help="rotation in degrees over the clip (positive: to its left)")
p.add_argument("--elevation", type=float, default=4.0, help="camera height angle in degrees")
p.add_argument("--color", default=None, help="also write frame 0 lit in color to this PNG")
a = p.parse_args()
a.out = os.path.abspath(a.out)
rx, ry = (int(v) for v in a.res.lower().split("x"))
if rx % 64 or ry % 64:
    raise SystemExit(f"[depth_video] --res {a.res}: sides must be multiples of 64")
frames = min(481, round(a.seconds * a.fps / 8) * 8 + 1)

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
bpy.ops.import_scene.gltf(filepath=os.path.abspath(a.glb))
act = bpy.data.actions.get(a.action)
if act is None:
    raise SystemExit(f"[depth_video] no action {a.action!r}, available: {[x.name for x in bpy.data.actions]}")
arm = next(o for o in scene.objects if o.type == "ARMATURE")
arm.animation_data_create()
arm.animation_data.action = act
f0, f1 = (int(round(x)) for x in act.frame_range)
cycle = max(1, f1 - f0)
pivot = bpy.data.objects.new("Pivot", None)
scene.collection.objects.link(pivot)
for o in [o for o in scene.objects if o.parent is None and o is not pivot]:
    o.parent = pivot


def pose(i):
    """Frame i of the clip: the action looped, the figure turned by its share of --turn."""
    scene.frame_set(f0 + i % cycle)
    t = i / max(1, frames - 1)
    pivot.rotation_euler = (0, 0, math.radians(a.turn) * (0.5 - 0.5 * math.cos(math.pi * t)))
    bpy.context.view_layer.update()


# visible only: the glTF importer adds a hidden "Icosphere" as bone shape to rigged models
meshes = [o for o in scene.objects if o.type == "MESH" and o.visible_get()]
pose(0)
pts = [o.matrix_world @ Vector(c) for o in meshes for c in o.bound_box]
mn = Vector([min(q[i] for q in pts) for i in range(3)])
mx = Vector([max(q[i] for q in pts) for i in range(3)])
center = (mn + mx) / 2

cam = bpy.data.objects.new("Cam", bpy.data.cameras.new("Cam"))
cam.data.lens = 50
cam.data.sensor_fit = "VERTICAL"
scene.collection.objects.link(cam)
scene.camera = cam
fov = 2 * math.atan(cam.data.sensor_height / 2 / cam.data.lens)
dist = (mx.z - mn.z) / a.fill / 2 / math.tan(fov / 2)
el = math.radians(a.elevation)
view = Vector((0, -math.cos(el), math.sin(el)))   # from the figure towards the camera
cam.location = center + view * dist
cam.rotation_euler = (-view).to_track_quat("-Z", "Y").to_euler()

scene.render.engine = "BLENDER_EEVEE"
scene.render.resolution_x, scene.render.resolution_y = rx, ry
scene.render.image_settings.file_format = "PNG"
scene.view_settings.view_transform = "Standard"
world = bpy.data.worlds.new("World")
world.use_nodes = True
background = world.node_tree.nodes["Background"].inputs["Color"]
scene.world = world

if a.color:
    background.default_value = (0.6, 0.6, 0.62, 1)
    lights = []
    for name, energy, rot in (("Key", 3.0, (50, 0, -35)), ("Rim", 1.5, (60, 0, 150))):
        light = bpy.data.objects.new(name, bpy.data.lights.new(name, "SUN"))
        light.data.energy = energy
        light.rotation_euler = [math.radians(v) for v in rot]
        scene.collection.objects.link(light)
        lights.append(light)
    scene.render.filepath = os.path.abspath(a.color)
    bpy.ops.render.render(write_still=True)
    for light in lights:
        bpy.data.objects.remove(light)

# near/far over the clip, from every 7th vertex of the deformed meshes in every 8th frame
depths = []
for i in range(0, frames, 8):
    pose(i)
    dg = bpy.context.evaluated_depsgraph_get()
    for o in meshes:
        ev = o.evaluated_get(dg)
        mesh = ev.to_mesh()
        depths += [(o.matrix_world @ v.co - cam.location).dot(-view)
                   for k, v in enumerate(mesh.vertices) if k % 7 == 0]
        ev.to_mesh_clear()
span = max(depths) - min(depths)
near, far = min(depths) - 0.02 * span, max(depths) + 0.15 * span

# depth as emission: camera Z depth mapped near..far -> 1..0, values written as they are (no view transform)
mat = bpy.data.materials.new("Depth")
mat.use_nodes = True
nt = mat.node_tree
nt.nodes.clear()
camera_data = nt.nodes.new("ShaderNodeCameraData")
ramp = nt.nodes.new("ShaderNodeMapRange")
ramp.clamp = True
for name, value in (("From Min", near), ("From Max", far), ("To Min", 1.0), ("To Max", 0.0)):
    ramp.inputs[name].default_value = value
emission = nt.nodes.new("ShaderNodeEmission")
output = nt.nodes.new("ShaderNodeOutputMaterial")
nt.links.new(camera_data.outputs["View Z Depth"], ramp.inputs["Value"])
nt.links.new(ramp.outputs["Result"], emission.inputs["Color"])
nt.links.new(emission.outputs["Emission"], output.inputs["Surface"])
scene.view_layers[0].material_override = mat
background.default_value = (0, 0, 0, 1)
scene.view_settings.view_transform = "Raw"

tmp = tempfile.mkdtemp(prefix="depth_")
try:
    for i in range(frames):
        pose(i)
        scene.render.filepath = os.path.join(tmp, f"f_{i:04d}.png")
        bpy.ops.render.render(write_still=True)
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-framerate", str(a.fps),
                    "-i", os.path.join(tmp, "f_%04d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "10",
                    a.out], check=True)
finally:
    shutil.rmtree(tmp)
print(f"[depth_video] OUT {a.out}  ({frames} frames, {frames / a.fps:.2f}s, {rx}x{ry}@{a.fps}, "
      f"action {a.action} cycle {cycle} frames, near {near:.3f} far {far:.3f})")
