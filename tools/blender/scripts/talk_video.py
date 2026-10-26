"""Talking figure as a video: the jaw follows the loudness of a speech recording, plus head and body language.

  talk_video rig.blend voice.wav out.mp4 [--mode fast|slow] [--res 1280x720] [--fps 24] [--samples N]
             [--engine BLENDER_EEVEE|CYCLES] [--nod 1.4] [--tilt 8.0] [--preview-frame N] [--max-frames N]

rig.blend: from  rig_quadruped in.glb out.glb --landmarks ... --save-blend rig.blend  (its bone names).
Modes: fast (default) = EEVEE, 16 samples, about 0.3 s per frame on the GB10 (1.5 min for a 10 s video).
       slow           = Cycles path tracing on the GPU, 32 samples + denoising: softer shadows, more realistic
                        light, about 2.5 s per frame.
--nod/--tilt: times in s (from the start of speech, comma separated) for a nod or a questioning head tilt.
The voice is a WAV file (8/16/32 bit PCM). Renders PNG frames and puts them together with the sound into an
H.264/AAC MP4 (ffmpeg from imageio-ffmpeg). --preview-frame N renders only frame N into out (a PNG; for setting
up camera and light), --max-frames N renders the first N frames and prints the time per frame, without video.
Stage: sky gradient, green ground, sun, camera with a slow push-in onto the figure.
"""
import argparse
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import wave

import bpy
import imageio_ffmpeg
import numpy as np
from mathutils import Vector

p = argparse.ArgumentParser(prog="talk_video")
p.add_argument("blend")
p.add_argument("audio")
p.add_argument("out")
p.add_argument("--res", default="1280x720")
p.add_argument("--fps", type=int, default=24)
p.add_argument("--mode", default="fast", choices=["fast", "slow"],
               help="fast = EEVEE, 16 samples (default); slow = Cycles path tracing, 32 samples + denoising")
p.add_argument("--samples", type=int, default=None, help="overrides the samples of the mode")
p.add_argument("--engine", default=None, choices=["BLENDER_EEVEE", "CYCLES"], help="overrides the engine of the mode")
p.add_argument("--lead", type=float, default=0.5, help="silence before speaking in s")
p.add_argument("--tail", type=float, default=0.8, help="time after speaking in s")
p.add_argument("--jaw-max", type=float, default=20.0, help="max. jaw opening in degrees")
p.add_argument("--nod", default="", help="times (s, from the start of speech) for nods")
p.add_argument("--tilt", default="", help="times (s) for a questioning head tilt")
p.add_argument("--emission", type=float, default=0.1, help="emission in the video (the scene is lit)")
p.add_argument("--preview-frame", type=int, default=None)
p.add_argument("--max-frames", type=int, default=None,
               help="measuring run: render only the first N frames, print the time per frame, no video")
a = p.parse_args()
a.out, a.audio, a.blend = (os.path.abspath(v) for v in (a.out, a.audio, a.blend))
MODES = {"fast": ("BLENDER_EEVEE", 16), "slow": ("CYCLES", 32)}
a.engine = a.engine or MODES[a.mode][0]
a.samples = a.samples or MODES[a.mode][1]
print(f"[talk] mode {a.mode}: {a.engine}, {a.samples} samples")
FPS = a.fps

# ------------------------------------------------------------------ audio -> envelope per frame
with wave.open(a.audio) as w:
    sr, n, ch, sw = w.getframerate(), w.getnframes(), w.getnchannels(), w.getsampwidth()
    raw = w.readframes(n)
dt = {1: np.int8, 2: np.int16, 4: np.int32}[sw]
x = np.frombuffer(raw, dtype=dt).astype(np.float32) / np.iinfo(dt).max
if ch > 1:
    x = x.reshape(-1, ch).mean(1)
dur = len(x) / sr
hop = sr / FPS
env = np.array([np.sqrt(np.mean(x[int(i * hop):int((i + 1) * hop)] ** 2) + 1e-12)
                for i in range(int(math.ceil(dur * FPS)))])
db = 20 * np.log10(env)
# -45 dB = closed, peak level (95th percentile) = fully open; smooth a little (fast attack, slower release)
lo, hi = -45.0, np.percentile(db, 95)
open_ = np.clip((db - lo) / (hi - lo), 0, 1) ** 1.3
sm = np.zeros_like(open_)
for i, v in enumerate(open_):
    prev = sm[i - 1] if i else 0
    sm[i] = prev + (v - prev) * (0.75 if v > prev else 0.45)
LEAD = int(round(a.lead * FPS))
total = LEAD + len(sm) + int(round(a.tail * FPS))
print(f"[talk] audio {dur:.2f}s, {len(sm)} frames of speech, video {total / FPS:.2f}s")

# ------------------------------------------------------------------ scene
bpy.ops.wm.open_mainfile(filepath=a.blend)
scene = bpy.context.scene
arm = next(o for o in scene.objects if o.type == "ARMATURE")
mesh = next(o for o in scene.objects if o.type == "MESH")
pb = arm.pose.bones
for b in pb:
    b.rotation_mode = "XYZ"


def sign_toward(bname, world_dir):
    z = arm.data.bones[bname].matrix_local.to_3x3().col[2]
    return 1.0 if z.dot(Vector(world_dir)) >= 0 else -1.0


FWD, DOWN, UP = (0, -1, 0), (0, 0, -1), (0, 0, 1)
act = bpy.data.actions.new("Speech")
arm.animation_data_create()
arm.animation_data.action = act
for b in pb:
    b.rotation_euler = (0, 0, 0)


def key(bname, frame, rx=0.0, ry=0.0, rz=0.0):
    b = pb[bname]
    b.rotation_euler = (math.radians(rx), math.radians(ry), math.radians(rz))
    b.keyframe_insert("rotation_euler", frame=frame)


def bump(t, centers, width):
    """Sum of smooth bells (0..1) around the times centers (s)."""
    return sum(math.exp(-((t - c) / width) ** 2) for c in centers)


sj, sh, sn = sign_toward("jaw", DOWN), sign_toward("head", FWD), sign_toward("neck2", FWD)
sc = sign_toward("chest", UP)
nods = [float(v) for v in a.nod.split(",") if v]
tilts = [float(v) for v in a.tilt.split(",") if v]
for f in range(total + 1):
    t_all = f / FPS
    t = t_all - a.lead  # time from the start of speech
    i = f - LEAD
    jaw = sm[i] if 0 <= i < len(sm) else 0.0
    talking = 1.0 if 0 <= i < len(sm) else 0.0
    # head: calm swaying + nodding with the syllable energy, accent nods, questioning tilt
    hx = 2.5 * math.sin(2 * math.pi * 0.35 * t_all) + 4 * jaw * talking + 14 * bump(t, nods, 0.18)
    # the head bone points up: local Y = vertical axis (turning), local Z = gaze axis (tilting)
    turn = 5 * math.sin(2 * math.pi * 0.21 * t_all + 1.0)
    tilt = 14 * bump(t, tilts, 0.45)
    key("jaw", f, rx=sj * a.jaw_max * jaw)
    key("head", f, rx=sh * hx, ry=turn, rz=tilt)
    key("neck2", f, rx=sn * (1.5 * math.sin(2 * math.pi * 0.35 * t_all - 0.6) + 4 * bump(t, nods, 0.25)),
        ry=2 * math.sin(2 * math.pi * 0.21 * t_all))
    key("chest", f, rx=sc * 1.0 * math.sin(2 * math.pi * 0.3 * t_all))
    key("ear.L", f, rx=5 * math.sin(2 * math.pi * 0.5 * t_all) + 10 * bump(t, tilts, 0.3))
    key("ear.R", f, rx=5 * math.sin(2 * math.pi * 0.5 * t_all + 1.3) + 10 * bump(t, tilts, 0.3))
    key("tail1", f, rz=10 * math.sin(2 * math.pi * 0.6 * t_all))
    key("tail2", f, rz=14 * math.sin(2 * math.pi * 0.6 * t_all - 0.7))

# the emission from glb_post is meant for unlit viewers -> reduce it here
for slot in mesh.material_slots:
    for nd in slot.material.node_tree.nodes:
        if nd.type == "BSDF_PRINCIPLED":
            nd.inputs["Emission Strength"].default_value = a.emission

# ------------------------------------------------------------------ stage: sky, ground, light, camera
scene.render.engine = a.engine
if a.engine == "CYCLES":
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "CUDA"
    prefs.get_devices()
    for d in prefs.devices:
        d.use = d.type == "CUDA"
    gpu = any(d.use for d in prefs.devices)
    scene.cycles.device = "GPU" if gpu else "CPU"
    scene.cycles.samples = a.samples
    scene.cycles.use_denoising = True
    print(f"[talk] Cycles on the {'GPU' if gpu else 'CPU'}")
else:
    scene.eevee.taa_render_samples = a.samples
rx, ry = (int(v) for v in a.res.split("x"))
scene.render.resolution_x, scene.render.resolution_y = rx, ry
scene.render.resolution_percentage = 100
scene.render.fps = FPS
scene.frame_start, scene.frame_end = 0, total
scene.view_settings.view_transform = "Standard"

world = bpy.data.worlds.new("Sky")
world.use_nodes = True
nt = world.node_tree
bg = nt.nodes["Background"]
coord = nt.nodes.new("ShaderNodeTexCoord")
sep = nt.nodes.new("ShaderNodeSeparateXYZ")
ramp = nt.nodes.new("ShaderNodeValToRGB")
nt.links.new(coord.outputs["Generated"], sep.inputs["Vector"])   # world: view direction
nt.links.new(sep.outputs["Z"], ramp.inputs["Fac"])               # height above the horizon
nt.links.new(ramp.outputs["Color"], bg.inputs["Color"])
ramp.color_ramp.elements[0].position = 0.0
ramp.color_ramp.elements[0].color = (0.95, 0.90, 0.78, 1)   # warm at the horizon
ramp.color_ramp.elements[1].position = 0.15
ramp.color_ramp.elements[1].color = (0.35, 0.62, 0.92, 1)   # sky blue above
bg.inputs["Strength"].default_value = 0.9
scene.world = world

bpy.ops.mesh.primitive_circle_add(vertices=64, radius=40, fill_type="NGON", location=(0, 0, 0))
ground = bpy.context.active_object
gm = bpy.data.materials.new("Ground")
gm.use_nodes = True
gb = gm.node_tree.nodes["Principled BSDF"]
gb.inputs["Base Color"].default_value = (0.42, 0.70, 0.28, 1)
gb.inputs["Roughness"].default_value = 0.9
ground.data.materials.append(gm)

bpy.ops.object.light_add(type="SUN", location=(2, -2, 4))
sun = bpy.context.active_object
sun.data.energy = 2.2
sun.data.angle = math.radians(8)
sun.rotation_euler = (math.radians(50), math.radians(15), math.radians(35))

cam_data = bpy.data.cameras.new("Cam")
cam_data.lens = 50
cam = bpy.data.objects.new("Cam", cam_data)
scene.collection.objects.link(cam)
scene.camera = cam
# bounding box of the figure
bb = [mesh.matrix_world @ Vector(c) for c in mesh.bound_box]
zmax = max(v.z for v in bb)
target = Vector((0, -0.05, 0.48 * zmax))
cam_start = Vector((1.35, -2.9, 0.62 * zmax))
cam_end = Vector((1.1, -2.45, 0.66 * zmax))      # slow push-in
track = cam.constraints.new("TRACK_TO")
tgt = bpy.data.objects.new("CamTarget", None)
scene.collection.objects.link(tgt)
tgt.location = target
track.target = tgt
track.track_axis, track.up_axis = "TRACK_NEGATIVE_Z", "UP_Y"
for f, pos, tz in ((0, cam_start, 0.48), (total, cam_end, 0.55)):
    cam.location = pos
    cam.keyframe_insert("location", frame=f)
    tgt.location = Vector((0, -0.05, tz * zmax))
    tgt.keyframe_insert("location", frame=f)

# ------------------------------------------------------------------ render
if a.preview_frame is not None:
    scene.frame_set(a.preview_frame)
    scene.render.filepath = a.out
    bpy.ops.render.render(write_still=True)
    print(f"[talk] preview {a.out}")
    sys.exit(0)

if a.max_frames:
    stamps = []
    bpy.app.handlers.render_post.append(lambda *_: stamps.append(time.time()))
    scene.frame_end = a.max_frames - 1
    tmp = tempfile.mkdtemp(prefix="talk_")
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = os.path.join(tmp, "f_")
    t0 = time.time()
    bpy.ops.render.render(animation=True)
    bpy.app.handlers.render_post.clear()
    shutil.rmtree(tmp)
    d = np.diff([t0] + stamps)
    print(f"[talk] MEASURED {a.engine} {rx}x{ry} {a.samples} samples: first frame {d[0]:.1f}s, "
          f"then {d[1:].mean():.2f}s per frame ({len(d)} frames)")
    sys.exit(0)

tmp = tempfile.mkdtemp(prefix="talk_")
scene.render.image_settings.file_format = "PNG"
scene.render.filepath = os.path.join(tmp, "f_")
bpy.ops.render.render(animation=True)
cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-framerate", str(FPS),
       "-start_number", "0", "-i", os.path.join(tmp, "f_%04d.png"),
       "-itsoffset", f"{a.lead}", "-i", a.audio,
       "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
       "-c:a", "aac", "-b:a", "160k", "-af", f"apad=pad_dur={a.tail}", "-shortest", a.out]
subprocess.run(cmd, check=True)
shutil.rmtree(tmp)
print(f"[talk] OUT {a.out}  ({total / FPS:.2f}s, {rx}x{ry}@{FPS})")
