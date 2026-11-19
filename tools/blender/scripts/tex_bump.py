"""Bump map from the fine contours of a model's texture, independent of the material set when rendering.

  tex_bump model.glb bump.png [--detail 1.0] [--invert] [--glb out.glb [--depth 0.004]]
           [--preview out.png [--elevation 30]]

Generated textures carry contours that are too fine for the geometry: eyes, ornaments, hair strands, seams. When
the model is rendered with its own material (wood, stone, clay, one color) instead of the texture, those
contours are lost. This script keeps them as a bump map: the light/dark contrast of the base color, high-pass
filtered so that only fine patterns remain, not the large color areas. Grey 0.5 = flat, dark = deep.

Outputs:
- bump.png: the bump (height) map in the model's UV layout, at the texture's size. In Blender: Image Texture
  (Non-Color, UV of the model) -> Bump node (Height; Distance ~ 0.004 x model height; Normal from the model's
  Normal Map node) -> Normal input of the material. Other renderers: as bump or height map.
- --glb out.glb: the model with the bump added into its normal map (baked, the size of the normal map; colors and
  materials unchanged), for viewers and game engines without a bump input. glTF stores no bump map itself.
- --preview out.png: Cycles render from 4 sides at --elevation degrees with a plain grey material, model normal
  map plus the bump at --depth: shows only the relief.

Options:
  --detail  size of the patterns that become bump, relative (2 = also coarser ones, 0.5 = only the finest)
  --invert  light = deep instead of dark = deep (dark figures with light patterns)
  --depth   bump depth for --glb and --preview, a fraction of the model height (0.004 = 4 mm on a 1 m figure)

Contrast of the texture decides how strong the bump is: an image_to_3d model made with texture_guidance 2-3 has
crisper contours. Every mesh with a base color texture gets a map (bump.png, bump_1.png, ...), rigged models too.
"""
import argparse
import math
import os
import tempfile

import bpy
import numpy as np
from mathutils import Vector
from PIL import Image, ImageDraw
from scipy import ndimage

p = argparse.ArgumentParser(prog="tex_bump")
p.add_argument("glb")
p.add_argument("out", help="output PNG of the bump map")
p.add_argument("--detail", type=float, default=1.0)
p.add_argument("--invert", action="store_true")
p.add_argument("--glb", dest="out_glb")
p.add_argument("--depth", type=float, default=0.004)
p.add_argument("--preview")
p.add_argument("--elevation", type=float, default=30.0)
a = p.parse_args()

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
bpy.ops.import_scene.gltf(filepath=a.glb)
bpy.context.view_layer.update()
meshes = [o for o in scene.objects if o.type == "MESH" and o.visible_get()]
pts = [o.matrix_world @ Vector(c) for o in meshes for c in o.bound_box]
bmin = Vector([min(q[i] for q in pts) for i in range(3)])
bmax = Vector([max(q[i] for q in pts) for i in range(3)])
height = bmax.z - bmin.z


def upstream(node, socket_name, node_type, seen=None):
    """The nearest node of node_type feeding node.inputs[socket_name], through any other nodes."""
    seen = seen or set()
    sock = node.inputs.get(socket_name)
    if sock is None or not sock.is_linked:
        return None
    src = sock.links[0].from_node
    if src.type == node_type:
        return src
    if src.name in seen:
        return None
    seen.add(src.name)
    for inp in src.inputs:
        if inp.is_linked:
            found = upstream(src, inp.name, node_type, seen)
            if found:
                return found
    return None


def uv_coverage(obj, w, h):
    """Mask of the texels covered by the object's UV triangles (rows from the bottom, like Blender pixels)."""
    me = obj.data
    me.calc_loop_triangles()
    uv = np.empty(len(me.loops) * 2, np.float32)
    me.uv_layers.active.data.foreach_get("uv", uv)
    uv = uv.reshape(-1, 2) * (w, h)
    uv[:, 1] = h - uv[:, 1]
    tri = np.empty(len(me.loop_triangles) * 3, np.int32)
    me.loop_triangles.foreach_get("loops", tri)
    img = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(img)
    for t in uv[tri].reshape(-1, 3, 2):
        draw.polygon([tuple(v) for v in t], fill=255, outline=255)
    return np.asarray(img)[::-1] > 0


def height_map(image, mask):
    """Bump height 0..1 (0.5 = flat) from the fine light/dark contrast of the image."""
    w, h = image.size
    px = np.empty(w * h * 4, np.float32)
    image.pixels.foreach_get(px)
    lum = px.reshape(h, w, 4)[..., :3] @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    # the outermost texels of a UV island are unreliable (baked half on, half off the surface): drop them,
    # they are filled from inside below, otherwise every UV seam becomes a ridge
    mask = ndimage.binary_erosion(mask, iterations=max(2, w // 1024))
    m = mask.astype(np.float32)
    sigma = max(1.0, a.detail * w / 256)

    def blur(x, s):   # normalised convolution: empty texels between UV islands do not count
        return ndimage.gaussian_filter(x * m, s) / np.maximum(ndimage.gaussian_filter(m, s), 1e-4)

    fine = blur(lum, 0.6) - blur(lum, sigma)
    std = float(np.std(fine[mask])) or 1.0
    hgt = 0.5 + 0.5 * np.tanh(fine / (2.5 * std))
    if a.invert:
        hgt = 1 - hgt
    idx = ndimage.distance_transform_edt(~mask, return_distances=False, return_indices=True)
    return hgt[idx[0], idx[1]], mask.mean()


def data_image(name, arr):
    """Non-color image from a HxW array (rows from the bottom)."""
    h, w = arr.shape
    img = bpy.data.images.new(name, w, h, is_data=True)
    rgba = np.ones((h, w, 4), np.float32)
    rgba[..., :3] = arr[..., None]
    img.pixels.foreach_set(rgba.ravel())
    return img


def add_bump(mat, bump_img, uv_name):
    """Bump node between the material's normal (map) and its BSDF; returns the nodes added and the BSDF's
    previous normal link source (to undo)."""
    nt = mat.node_tree
    bsdf = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
    prev = bsdf.inputs["Normal"].links[0].from_socket if bsdf.inputs["Normal"].is_linked else None
    uv = nt.nodes.new("ShaderNodeUVMap")
    uv.uv_map = uv_name
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = bump_img
    tex.interpolation = "Cubic"
    nt.links.new(uv.outputs["UV"], tex.inputs["Vector"])
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Distance"].default_value = a.depth * height
    nt.links.new(tex.outputs["Color"], bump.inputs["Height"])
    if prev is not None:
        nt.links.new(prev, bump.inputs["Normal"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    return [uv, tex, bump], prev


stem, ext = os.path.splitext(a.out)
jobs = []   # (object, material, bump image)
for obj in meshes:
    mat = obj.data.materials[0] if obj.data.materials else None
    nodes = mat.node_tree.nodes if mat and mat.node_tree else []
    bsdf = next((n for n in nodes if n.type == "BSDF_PRINCIPLED"), None)
    base = upstream(bsdf, "Base Color", "TEX_IMAGE") if bsdf else None
    if base is None or not obj.data.uv_layers:
        print(f"[tex_bump] {obj.name}: no base color texture, skipped")
        continue
    w, h = base.image.size
    hgt, covered = height_map(base.image, uv_coverage(obj, w, h))
    path = f"{stem}{'' if not jobs else f'_{len(jobs)}'}{ext or '.png'}"
    Image.fromarray((hgt[::-1] * 255 + 0.5).astype(np.uint8), "L").save(path)
    jobs.append((obj, mat, data_image(f"{obj.name}_bump", hgt)))
    print(f"[tex_bump] {obj.name}: OUT {path} ({w}x{h}, bump on {covered * 100:.0f}% of the atlas)")
if not jobs:
    raise SystemExit("[tex_bump] no mesh with a base color texture")

if a.out_glb or a.preview:
    scene.render.engine = "CYCLES"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "CUDA"
    prefs.get_devices()
    for d in prefs.devices:
        d.use = d.type == "CUDA"
    scene.cycles.device = "GPU"

if a.out_glb:
    tmp = tempfile.mkdtemp()
    for obj, mat, bump_img in jobs:
        nt = mat.node_tree
        bsdf = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
        nmap = upstream(bsdf, "Normal", "NORMAL_MAP")
        ntex = upstream(nmap, "Color", "TEX_IMAGE") if nmap else None
        size = ntex.image.size[:] if ntex else bump_img.size[:]
        added, prev = add_bump(mat, bump_img, obj.data.uv_layers.active.name)
        baked = bpy.data.images.new(f"{obj.name}_normal", *size, is_data=True)
        target = nt.nodes.new("ShaderNodeTexImage")
        target.image = baked
        nt.nodes.active = target
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        scene.cycles.samples = 4
        bpy.ops.object.bake(type="NORMAL", normal_space="TANGENT", margin=8, use_clear=True)
        path = os.path.join(tmp, f"{obj.name}_normal.png")
        baked.filepath_raw, baked.file_format = path, "PNG"
        baked.save()
        for n in added + [target]:
            nt.nodes.remove(n)
        img = bpy.data.images.load(path)
        img.colorspace_settings.is_data = True
        if ntex is None:   # model without a normal map: add one
            nmap = nt.nodes.new("ShaderNodeNormalMap")
            nmap.uv_map = obj.data.uv_layers.active.name
            ntex = nt.nodes.new("ShaderNodeTexImage")
            nt.links.new(ntex.outputs["Color"], nmap.inputs["Color"])
        ntex.image = img
        nt.links.new(nmap.outputs["Normal"], bsdf.inputs["Normal"])
    bpy.ops.export_scene.gltf(filepath=a.out_glb, export_format="GLB")
    print(f"[tex_bump] OUT {a.out_glb} (bump in the normal map, depth {a.depth})")

if a.preview:
    grey = {}
    for obj, mat, bump_img in jobs:   # plain grey with the model's normal map and the bump: only the relief
        g = mat.copy()
        bsdf = next(n for n in g.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
        for name in ("Base Color", "Metallic", "Roughness"):
            for link in list(bsdf.inputs[name].links):
                g.node_tree.links.remove(link)
        bsdf.inputs["Base Color"].default_value = (0.6, 0.6, 0.6, 1)
        bsdf.inputs["Metallic"].default_value = 0.0
        bsdf.inputs["Roughness"].default_value = 0.6
        if not a.out_glb:   # with --glb the normal map already holds the bump
            add_bump(g, bump_img, obj.data.uv_layers.active.name)
        grey[obj.name] = g
        obj.data.materials[0] = g
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
    cam = bpy.data.objects.new("PreviewCam", bpy.data.cameras.new("PreviewCam"))
    scene.collection.objects.link(cam)
    scene.camera = cam
    size = 512
    scene.render.resolution_x, scene.render.resolution_y = size, int(size * 1.25)
    scene.view_settings.view_transform = "Standard"
    scene.cycles.samples = 64
    scene.cycles.use_denoising = True
    center, radius = (bmin + bmax) / 2, (bmax - bmin).length / 2
    fov = 2 * math.atan(cam.data.sensor_width / 2 / cam.data.lens)
    dist = radius / math.sin(fov / 2) * 1.05
    el = math.radians(a.elevation)
    sheet = Image.new("RGB", (size * 4, int(size * 1.25)))
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(4):
            az = math.radians(-40 + 90 * i)
            d = Vector((math.sin(az) * math.cos(el), -math.cos(az) * math.cos(el), math.sin(el)))
            cam.location = center + d * dist
            cam.rotation_euler = (-d).to_track_quat("-Z", "Y").to_euler()
            scene.render.filepath = os.path.join(tmp, f"{i}.png")
            bpy.ops.render.render(write_still=True)
            sheet.paste(Image.open(scene.render.filepath).convert("RGB"), (size * i, 0))
    sheet.save(a.preview)
    print(f"[tex_bump] OUT {a.preview}")
