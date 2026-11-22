"""Geometry for retexturing, run by run.py in the blender tool's environment (bpy, Cycles on the GPU):

  python blender_views.py model.glb OUT_DIR --texture 2048 --views 0,10;60,10;... --pixels 1048576

Writes into OUT_DIR:
  texels.npz   pos (T,T,3) and nrm (T,T,3): 3D position and unit normal per texel, cover (T,T): texels the mesh
               covers; rows top-down like the PNG texture (glTF UV)
  views.npz    per view uv (N,H,W,2) (glTF UV, u right, v down), depth (N,H,W) (distance along the view axis, 0 =
               background), and the cameras: R (N,3,3) world -> camera rotation, eye (N,3), f (pixels)
  views.json   width, height, the views' azimuth/elevation, bounding box
The UV and depth passes are Cycles renders of an emission shader with one sample at the pixel centre (no filter),
so each pixel holds the exact values of the surface it sees. World space is Blender's (Z up, the model's front
toward -Y, as the glTF importer places a glTF model facing +Z). Azimuth 0 looks at the front, positive azimuth
moves the camera to the model's left; elevation in degrees above the horizon.
"""
import argparse
import json
import math
import os
import tempfile

import bpy
import numpy as np
from mathutils import Matrix, Vector

p = argparse.ArgumentParser(prog="blender_views")
p.add_argument("glb")
p.add_argument("out")
p.add_argument("--texture", type=int, default=2048, help="side of the texel maps")
p.add_argument("--views", default="0,10", help="azimuth,elevation;... in degrees")
p.add_argument("--pixels", type=int, default=1024 * 1024, help="pixels per view (about)")
p.add_argument("--fov", type=float, default=30.0, help="vertical field of view in degrees")
a = p.parse_args()
views = [tuple(float(v) for v in s.split(",")) for s in a.views.split(";")]
os.makedirs(a.out, exist_ok=True)

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
bpy.ops.import_scene.gltf(filepath=os.path.abspath(a.glb))
# visible only: the glTF importer adds a hidden "Icosphere" as bone shape to rigged models
meshes = [o for o in scene.objects if o.type == "MESH" and o.visible_get()]
if len(meshes) != 1:
    raise SystemExit(f"[blender_views] the model needs exactly one mesh, it has {len(meshes)}: "
                     f"{[o.name for o in meshes]}")
obj = meshes[0]
if not obj.data.uv_layers:
    raise SystemExit("[blender_views] the mesh has no UV map")
# rest pose: a rigged model keeps its armature, but the bind pose is what the texture is painted on
for mod in obj.modifiers:
    mod.show_render = mod.show_viewport = False
bpy.context.view_layer.update()

V = np.empty(len(obj.data.vertices) * 3)
obj.data.vertices.foreach_get("co", V)
M = np.array(obj.matrix_world)
V = V.reshape(-1, 3) @ M[:3, :3].T + M[:3, 3]
bmin, bmax = V.min(0), V.max(0)
center = (bmin + bmax) / 2
sample = V[np.random.default_rng(0).choice(len(V), min(len(V), 50_000), replace=False)]

# one material for every pass: emission of a color chosen per pass, the bake target as active image node
mat = bpy.data.materials.new("Pass")
nt = mat.node_tree
nt.nodes.clear()
out = nt.nodes.new("ShaderNodeOutputMaterial")
emit = nt.nodes.new("ShaderNodeEmission")
nt.links.new(emit.outputs[0], out.inputs["Surface"])
uvmap = nt.nodes.new("ShaderNodeUVMap")
uvmap.uv_map = obj.data.uv_layers.active.name
geo = nt.nodes.new("ShaderNodeNewGeometry")
camdata = nt.nodes.new("ShaderNodeCameraData")
comb = nt.nodes.new("ShaderNodeCombineXYZ")
sep = nt.nodes.new("ShaderNodeSeparateXYZ")
nt.links.new(uvmap.outputs["UV"], sep.inputs[0])
nt.links.new(sep.outputs[0], comb.inputs[0])
nt.links.new(sep.outputs[1], comb.inputs[1])
nt.links.new(camdata.outputs["View Z Depth"], comb.inputs[2])
target = nt.nodes.new("ShaderNodeTexImage")
nt.nodes.active = target
obj.data.materials.clear()
obj.data.materials.append(mat)
for poly in obj.data.polygons:
    poly.material_index = 0

scene.render.engine = "CYCLES"
prefs = bpy.context.preferences.addons["cycles"].preferences
prefs.compute_device_type = "CUDA"
prefs.get_devices()
for d in prefs.devices:
    d.use = d.type == "CUDA"
scene.cycles.device = "GPU"
scene.cycles.samples = 1
scene.cycles.use_denoising = False
scene.cycles.use_adaptive_sampling = False
scene.cycles.max_bounces = 0
scene.cycles.pixel_filter_type = "BOX"
scene.cycles.filter_width = 0.01   # one sample at the pixel centre
scene.cycles.seed = 0
scene.render.film_transparent = True
scene.view_settings.view_transform = "Raw"
scene.display_settings.display_device = "sRGB"
world = bpy.data.worlds.new("Black")
world.use_nodes = True
world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.0
scene.world = world


def pixels(img):
    """Image -> float array (H,W,4), rows top-down."""
    w, h = img.size
    buf = np.empty(w * h * 4, np.float32)
    img.pixels.foreach_get(buf)
    return buf.reshape(h, w, 4)[::-1]


# --- texel maps: bake position and normal in UV space
T = a.texture
bpy.ops.object.select_all(action="DESELECT")
obj.select_set(True)
bpy.context.view_layer.objects.active = obj
bake = {}
one = nt.nodes.new("ShaderNodeRGB")
one.outputs[0].default_value = (1, 1, 1, 1)
for name, socket in (("pos", geo.outputs["Position"]), ("nrm", geo.outputs["Normal"]), ("cover", one.outputs[0])):
    img = bpy.data.images.new(name, T, T, alpha=False, float_buffer=True)
    target.image = img
    nt.links.new(socket, emit.inputs["Color"])
    bpy.ops.object.bake(type="EMIT", margin=0, use_clear=True)   # clear: black where no face is
    bake[name] = pixels(img)
cover = bake["cover"][..., 0] > 0.5
nrm = bake["nrm"][..., :3]
nrm /= np.maximum(np.linalg.norm(nrm, axis=-1, keepdims=True), 1e-8)
np.savez(os.path.join(a.out, "texels.npz"), pos=bake["pos"][..., :3], nrm=nrm.astype(np.float32), cover=cover)
print(f"[blender_views] texel maps {T}x{T}: {cover.mean() * 100:.0f}% covered")

# --- views: size from the model's proportions, distance so that the model fills ~90% of the picture
ext_h = math.hypot(*(bmax - bmin)[:2])   # widest horizontal extent over all azimuths
ext_v = bmax[2] - bmin[2]
aspect = min(1.6, max(0.625, ext_h / max(ext_v, 1e-6)))
H = int(round(math.sqrt(a.pixels / aspect) / 16)) * 16
W = int(round(H * aspect / 16)) * 16
fy = (H / 2) / math.tan(math.radians(a.fov) / 2)
fx = fy
scene.render.resolution_x, scene.render.resolution_y = W, H
scene.render.resolution_percentage = 100
cam = bpy.data.objects.new("Cam", bpy.data.cameras.new("Cam"))
cam.data.sensor_fit = "VERTICAL"
cam.data.angle_y = math.radians(a.fov)
cam.data.clip_start = 1e-3
scene.collection.objects.link(cam)
scene.camera = cam
nt.links.new(comb.outputs[0], emit.inputs["Color"])
scene.render.image_settings.file_format = "OPEN_EXR"
scene.render.image_settings.color_depth = "32"
scene.render.image_settings.exr_codec = "ZIP"

uvs, depths, Rs, eyes = [], [], [], []
tmp = tempfile.mkdtemp(prefix="retex-")
for i, (az, el) in enumerate(views):
    az_r, el_r = math.radians(az), math.radians(el)
    # direction from the model to the camera: azimuth 0 = front (-Y), 90 = the model's left (+X)
    d = np.array([math.sin(az_r) * math.cos(el_r), -math.cos(az_r) * math.cos(el_r), math.sin(el_r)])
    fwd = -d
    up0 = np.array([0.0, 0.0, 1.0]) if abs(el) < 89 else np.array([0.0, 1.0, 0.0])
    right = np.cross(fwd, up0)
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    # distance: every sample point inside 90% of the frame (perspective)
    rel = sample - center
    px, py, pf = rel @ right, rel @ up, rel @ fwd
    dist = float(np.max(np.concatenate([pf + np.abs(px) * fx / (0.9 * W / 2), pf + np.abs(py) * fy / (0.9 * H / 2)])))
    eye = center - fwd * dist
    R = np.stack([right, up, -fwd])   # world -> camera (Blender camera: x right, y up, looks along -z)
    cam.matrix_world = Matrix.Translation(Vector(eye)) @ Matrix(R.T.tolist()).to_4x4()
    scene.render.filepath = os.path.join(tmp, f"view{i}.exr")
    bpy.ops.render.render(write_still=True)
    img = bpy.data.images.load(scene.render.filepath)
    px_ = pixels(img)
    bpy.data.images.remove(img)
    hit = px_[..., 3] > 0.5
    uv = np.stack([px_[..., 0], 1 - px_[..., 1]], -1)   # Blender v up -> glTF v down
    uvs.append(np.where(hit[..., None], uv, -1).astype(np.float32))
    depths.append(np.where(hit, px_[..., 2], 0).astype(np.float32))
    Rs.append(R)
    eyes.append(eye)
    print(f"[blender_views] view {i}: azimuth {az:g}, elevation {el:g}, {hit.mean() * 100:.0f}% of the picture")

np.savez(os.path.join(a.out, "views.npz"), uv=np.stack(uvs), depth=np.stack(depths), R=np.stack(Rs),
         eye=np.stack(eyes), f=np.float64(fy))
with open(os.path.join(a.out, "views.json"), "w") as fh:
    json.dump({"width": W, "height": H, "views": views, "bmin": bmin.tolist(), "bmax": bmax.tolist()}, fh)
print(f"[blender_views] {len(views)} views {W}x{H}")
