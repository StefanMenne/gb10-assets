"""Give a generated model a carved-wood look: procedural 3D wood plus a bump map from its texture.

  wood_material model.glb out.blend [--wood oak|walnut|pine|beech] [--depth 0.004] [--detail 1.0] [--invert]
                [--stain 0] [--roughness 0.55] [--glb out.glb [--tex 2048]] [--preview out.png [--elevation 30]]

Generated textures carry fine detail (ornaments, seams, eyes, hair strands) that is too fine for the geometry.
This script turns the texture into a bump map: the light/dark contrast of the base color, high-pass filtered so
that only fine patterns remain, not the large color areas. The color itself is replaced by procedural wood in
object space: the grain runs through the whole figure as if it were carved from one upright block, with no UV
seams. The model's normal map (geometry detail) is kept underneath the bump.

Outputs:
- out.blend: the model with this material, images packed; append or link it into a render scene. glTF cannot
  store a bump map or a procedural shader, so this is the file for rendering in Blender.
- <out>_bump.png: the bump (height) map in the model's UV layout, grey 0.5 = flat, for other renderers.
- --glb out.glb: everything baked into a self-contained GLB (base color with the wood, normal map with the bump
  added) at --tex, for viewers and game engines; the grain is then fixed in the texture.
- --preview out.png: Cycles render from 4 sides at --elevation degrees.

Options:
  --wood      wood preset (colors of the light and dark grain)
  --depth     bump depth as a fraction of the model height (0.004 = 4 mm on a 1 m figure); 0 = no bump
  --detail    size of the patterns that become bump, relative (2 = also coarser ones, 0.5 = only the finest)
  --invert    light = deep instead of dark = deep (for dark figures with light patterns)
  --stain S   also darken the wood where the texture is dark, 0-1: painted patterns appear as stain
  --roughness surface roughness of the wood

Texture contrast decides how strong the bump is: an image_to_3d model made with texture_guidance 2-3 has
crisper patterns. Works on rigged models too (every mesh with an image texture gets the material).
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

WOODS = {   # sRGB colors of the light and dark grain
    "oak": ((0.66, 0.47, 0.27), (0.42, 0.27, 0.13)),
    "walnut": ((0.40, 0.26, 0.16), (0.20, 0.12, 0.07)),
    "pine": ((0.86, 0.70, 0.47), (0.68, 0.49, 0.28)),
    "beech": ((0.80, 0.59, 0.41), (0.63, 0.43, 0.28)),
}

p = argparse.ArgumentParser(prog="wood_material")
p.add_argument("glb")
p.add_argument("out", help="output .blend")
p.add_argument("--wood", default="oak", choices=list(WOODS))
p.add_argument("--depth", type=float, default=0.004)
p.add_argument("--detail", type=float, default=1.0)
p.add_argument("--invert", action="store_true")
p.add_argument("--stain", type=float, default=0.0)
p.add_argument("--roughness", type=float, default=0.55)
p.add_argument("--glb", dest="out_glb")
p.add_argument("--tex", type=int, default=2048)
p.add_argument("--preview")
p.add_argument("--elevation", type=float, default=30.0)
a = p.parse_args()
if not a.out.endswith(".blend"):
    raise SystemExit("[wood_material] out must be a .blend file")

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
bpy.ops.import_scene.gltf(filepath=a.glb)
bpy.context.view_layer.update()
meshes = [o for o in scene.objects if o.type == "MESH" and o.visible_get()]
pts = [o.matrix_world @ Vector(c) for o in meshes for c in o.bound_box]
bmin = Vector([min(q[i] for q in pts) for i in range(3)])
bmax = Vector([max(q[i] for q in pts) for i in range(3)])
height = bmax.z - bmin.z


def upstream_image(node, socket_name, seen=None):
    """The image texture node feeding node.inputs[socket_name], through any other nodes."""
    seen = seen or set()
    sock = node.inputs.get(socket_name)
    if sock is None or not sock.is_linked:
        return None
    src = sock.links[0].from_node
    if src.type == "TEX_IMAGE":
        return src
    if src.name in seen:
        return None
    seen.add(src.name)
    for inp in src.inputs:
        if inp.is_linked:
            found = upstream_image(src, inp.name, seen)
            if found:
                return found
    return None


def uv_coverage(obj, w, h):
    """Mask of the texels covered by the object's UV triangles."""
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
    return np.asarray(img)[::-1] > 0   # Blender pixel rows start at the bottom


def height_map(image, mask):
    """Bump height 0..1 (0.5 = flat) from the fine light/dark contrast of the image."""
    w, h = image.size
    px = np.empty(w * h * 4, np.float32)
    image.pixels.foreach_get(px)
    rgb = px.reshape(h, w, 4)[..., :3]
    lum = rgb @ np.array([0.2126, 0.7152, 0.0722], np.float32)
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
    # fill the empty texels with the nearest covered one: no steps at UV seams
    idx = ndimage.distance_transform_edt(~mask, return_distances=False, return_indices=True)
    hgt = hgt[idx[0], idx[1]]
    return hgt, lum


def new_image(name, arr):
    """Non-color float image from a HxW array (rows from the bottom)."""
    h, w = arr.shape
    img = bpy.data.images.new(name, w, h, float_buffer=False, is_data=True)
    rgba = np.ones((h, w, 4), np.float32)
    rgba[..., :3] = arr[..., None]
    img.pixels.foreach_set(rgba.ravel())
    img.colorspace_settings.is_data = True
    return img


def wood_material(name, bump_img, normal_img, stain_img, uv_name):
    light, dark = ([c ** 2.2 for c in col] + [1.0] for col in WOODS[a.wood])
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nodes, links = nt.nodes, nt.links
    bsdf = nodes["Principled BSDF"]
    bsdf.inputs["Metallic"].default_value = 0.0
    bsdf.inputs["Roughness"].default_value = a.roughness

    # wood: rings around the vertical axis, distorted by noise, plus fine fibres along the grain
    coord = nodes.new("ShaderNodeTexCoord")
    mapping = nodes.new("ShaderNodeMapping")
    mapping.inputs["Location"].default_value = -(bmin + bmax) / 2 / height
    mapping.inputs["Scale"].default_value = (1 / height,) * 3
    links.new(coord.outputs["Object"], mapping.inputs["Vector"])
    warp = nodes.new("ShaderNodeTexNoise")
    warp.inputs["Scale"].default_value = 2.0
    warp.inputs["Detail"].default_value = 3.0
    warped = nodes.new("ShaderNodeVectorMath")
    warped.operation = "MULTIPLY_ADD"
    warped.inputs[1].default_value = (0.1, 0.1, 0.3)
    links.new(mapping.outputs["Vector"], warp.inputs["Vector"])
    links.new(warp.outputs["Color"], warped.inputs[0])
    links.new(mapping.outputs["Vector"], warped.inputs[2])
    rings = nodes.new("ShaderNodeTexWave")
    rings.wave_type = "RINGS"
    rings.rings_direction = "Z"
    rings.inputs["Scale"].default_value = 24.0
    rings.inputs["Distortion"].default_value = 3.0
    rings.inputs["Detail"].default_value = 2.0
    links.new(warped.outputs["Vector"], rings.inputs["Vector"])
    fibres_map = nodes.new("ShaderNodeMapping")
    fibres_map.inputs["Scale"].default_value = (60.0, 60.0, 2.0)
    links.new(mapping.outputs["Vector"], fibres_map.inputs["Vector"])
    fibres = nodes.new("ShaderNodeTexNoise")
    fibres.inputs["Scale"].default_value = 4.0
    fibres.inputs["Detail"].default_value = 6.0
    links.new(fibres_map.outputs["Vector"], fibres.inputs["Vector"])
    grain = nodes.new("ShaderNodeMath")
    grain.operation = "MULTIPLY_ADD"
    grain.inputs[1].default_value = 0.25
    links.new(fibres.outputs["Fac"], grain.inputs[0])
    links.new(rings.outputs["Fac"], grain.inputs[2])
    ramp = nodes.new("ShaderNodeValToRGB")
    # thin dark late-wood lines on light early wood
    ramp.color_ramp.elements[0].position, ramp.color_ramp.elements[0].color = 0.05, dark
    ramp.color_ramp.elements[1].position, ramp.color_ramp.elements[1].color = 0.45, light
    links.new(grain.outputs["Value"], ramp.inputs["Fac"])
    color = ramp.outputs["Color"]

    uv = nodes.new("ShaderNodeUVMap")
    uv.uv_map = uv_name
    if stain_img is not None and a.stain > 0:
        stain = nodes.new("ShaderNodeTexImage")
        stain.image = stain_img
        stain.interpolation = "Cubic"
        links.new(uv.outputs["UV"], stain.inputs["Vector"])
        darken = nodes.new("ShaderNodeMix")
        darken.data_type = "RGBA"
        darken.blend_type = "MULTIPLY"
        darken.inputs["Factor"].default_value = a.stain
        links.new(color, darken.inputs["A"])
        links.new(stain.outputs["Color"], darken.inputs["B"])
        color = darken.outputs["Result"]
    links.new(color, bsdf.inputs["Base Color"])

    normal = None
    if normal_img is not None:
        nmap_tex = nodes.new("ShaderNodeTexImage")
        nmap_tex.image = normal_img
        links.new(uv.outputs["UV"], nmap_tex.inputs["Vector"])
        nmap = nodes.new("ShaderNodeNormalMap")
        nmap.uv_map = uv_name
        links.new(nmap_tex.outputs["Color"], nmap.inputs["Color"])
        normal = nmap.outputs["Normal"]
    if bump_img is not None and a.depth > 0:
        bump_tex = nodes.new("ShaderNodeTexImage")
        bump_tex.image = bump_img
        bump_tex.interpolation = "Cubic"
        links.new(uv.outputs["UV"], bump_tex.inputs["Vector"])
        bump = nodes.new("ShaderNodeBump")
        bump.inputs["Strength"].default_value = 1.0
        bump.inputs["Distance"].default_value = a.depth * height
        links.new(bump_tex.outputs["Color"], bump.inputs["Height"])
        if normal is not None:
            links.new(normal, bump.inputs["Normal"])
        normal = bump.outputs["Normal"]
    if normal is not None:
        links.new(normal, bsdf.inputs["Normal"])
    return mat


stem = os.path.splitext(a.out)[0]
done = 0
for obj in meshes:
    if not obj.data.materials or not obj.data.uv_layers:
        continue
    old = obj.data.materials[0]
    bsdf = next((n for n in old.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None) if old.node_tree else None
    base = upstream_image(bsdf, "Base Color") if bsdf else None
    if base is None:
        print(f"[wood_material] {obj.name}: no base color texture, skipped")
        continue
    nrm = upstream_image(bsdf, "Normal")
    w, h = base.image.size
    mask = uv_coverage(obj, w, h)
    hgt, lum = height_map(base.image, mask)
    suffix = "" if done == 0 else f"_{done}"
    bump_img = new_image(f"{obj.name}_bump", hgt)
    Image.fromarray((hgt[::-1] * 255 + 0.5).astype(np.uint8), "L").save(f"{stem}_bump{suffix}.png")
    stain_img = None
    if a.stain > 0:   # dark texture -> darker wood; normalised so that the median texel leaves the wood as it is
        rel = np.clip(lum / max(float(np.median(lum[mask])), 1e-3), 0, 1)
        stain_img = new_image(f"{obj.name}_stain", rel)
    mat = wood_material(f"Wood_{obj.name}", bump_img, nrm.image if nrm else None, stain_img,
                        obj.data.uv_layers.active.name)
    obj.data.materials.clear()
    obj.data.materials.append(mat)
    done += 1
    print(f"[wood_material] {obj.name}: texture {w}x{h}, bump on {mask.mean() * 100:.0f}% of the atlas, "
          f"depth {a.depth * height:.4f} (model height {height:.3f})")
if not done:
    raise SystemExit("[wood_material] no mesh with a base color texture")

# ---------------------------------------------------------------- Cycles on the GPU (bake and preview)
scene.render.engine = "CYCLES"
prefs = bpy.context.preferences.addons["cycles"].preferences
prefs.compute_device_type = "CUDA"
prefs.get_devices()
for d in prefs.devices:
    d.use = d.type == "CUDA"
scene.cycles.device = "GPU"

if a.out_glb:
    tmp = tempfile.mkdtemp()
    for obj in meshes:
        mat = obj.data.materials[0] if obj.data.materials else None
        if mat is None or not mat.name.startswith("Wood_"):
            continue
        nodes = mat.node_tree.nodes
        baked = {}
        for kind, bake_type, is_data in (("color", "DIFFUSE", False), ("normal", "NORMAL", True)):
            img = bpy.data.images.new(f"{obj.name}_{kind}", a.tex, a.tex)
            img.colorspace_settings.is_data = is_data
            target = nodes.new("ShaderNodeTexImage")
            target.image = img
            nodes.active = target
            bpy.ops.object.select_all(action="DESELECT")
            obj.select_set(True)
            bpy.context.view_layer.objects.active = obj
            scene.cycles.samples = 4
            bpy.ops.object.bake(type=bake_type, pass_filter={"COLOR"}, normal_space="TANGENT", margin=8,
                                use_clear=True)
            path = os.path.join(tmp, f"{obj.name}_{kind}.png")
            img.filepath_raw, img.file_format = path, "PNG"
            img.save()
            nodes.remove(target)
            baked[kind] = bpy.data.images.load(path)
            baked[kind].colorspace_settings.is_data = is_data
        out = bpy.data.materials.new(f"WoodBaked_{obj.name}")
        out.use_nodes = True
        bsdf = out.node_tree.nodes["Principled BSDF"]
        bsdf.inputs["Metallic"].default_value = 0.0
        bsdf.inputs["Roughness"].default_value = a.roughness
        col = out.node_tree.nodes.new("ShaderNodeTexImage")
        col.image = baked["color"]
        out.node_tree.links.new(col.outputs["Color"], bsdf.inputs["Base Color"])
        ntex = out.node_tree.nodes.new("ShaderNodeTexImage")
        ntex.image = baked["normal"]
        nmap = out.node_tree.nodes.new("ShaderNodeNormalMap")
        out.node_tree.links.new(ntex.outputs["Color"], nmap.inputs["Color"])
        out.node_tree.links.new(nmap.outputs["Normal"], bsdf.inputs["Normal"])
        obj["wood_material"] = mat.name
        obj.data.materials[0] = out
    bpy.ops.export_scene.gltf(filepath=a.out_glb, export_format="GLB")
    for obj in meshes:   # back to the procedural material for the .blend
        if "wood_material" in obj:
            obj.data.materials[0] = bpy.data.materials[obj["wood_material"]]
            del obj["wood_material"]
    print(f"[wood_material] OUT {a.out_glb} (baked {a.tex}x{a.tex})")

if a.preview:
    world = bpy.data.worlds.new("Studio")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.55, 0.57, 0.6, 1)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.8
    scene.world = world
    lights = []
    for name, energy, rot in (("Key", 3.0, (50, 0, -35)), ("Rim", 1.5, (60, 0, 150))):
        light = bpy.data.objects.new(name, bpy.data.lights.new(name, "SUN"))
        light.data.energy = energy
        light.rotation_euler = [math.radians(v) for v in rot]
        scene.collection.objects.link(light)
        lights.append(light)
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
    for obj in lights + [cam]:   # keep the .blend free of preview helpers
        bpy.data.objects.remove(obj)
    scene.world = None
    print(f"[wood_material] OUT {a.preview}")

for img in bpy.data.images:   # generated ones (bump, stain) first: pack_all only packs images with a file
    if img.packed_file is None and img.source == "GENERATED":
        img.pack()
bpy.ops.file.pack_all()
bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(a.out))
print(f"[wood_material] OUT {a.out}, {stem}_bump.png ({a.wood}, depth {a.depth}, stain {a.stain})")
