"""Post-process a generated GLB (Pixal3D, TRELLIS): stand it on the ground, brighten, shrink its textures.

  glb_post in.glb out.glb [--emission 0.5] [--tex 1024] [--format JPEG|PNG|WEBP] [--height 1.0] [--roughness -1]

- stands on the ground (bottom centre at the origin), optionally scaled to --height metres
- --emission: base color also as emission (0 = off). Brightens cartoon figures in viewers without
  environment light (F3D, Windows 3D Viewer, ...) without changing their colors
- --tex/--format: scale textures down and store them as JPEG -> much smaller file
- --roughness: a fixed roughness instead of the texture, metallic 0 (-1 = keep the textures)
"""
import argparse

import bpy
from mathutils import Matrix, Vector

p = argparse.ArgumentParser(prog="glb_post")
p.add_argument("src")
p.add_argument("dst")
p.add_argument("--emission", type=float, default=0.5)
p.add_argument("--tex", type=int, default=1024)
p.add_argument("--format", default="JPEG", choices=["JPEG", "PNG", "WEBP"])
p.add_argument("--quality", type=int, default=90, help="JPEG/WEBP quality")
p.add_argument("--height", type=float, default=0.0, help="target height in m (0 = unchanged)")
p.add_argument("--roughness", type=float, default=-1.0)
a = p.parse_args()

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=a.src)
meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]

# transform: onto the ground, centred, optionally scaled
bpy.context.view_layer.update()
pts = [o.matrix_world @ Vector(c) for o in meshes for c in o.bound_box]
mn = Vector([min(p[i] for p in pts) for i in range(3)])
mx = Vector([max(p[i] for p in pts) for i in range(3)])
scale = a.height / (mx.z - mn.z) if a.height > 0 else 1.0
shift = Vector(((mn.x + mx.x) / 2, (mn.y + mx.y) / 2, mn.z))
for o in meshes:
    o.matrix_world.translation -= shift
    o.matrix_world = Matrix.Scale(scale, 4) @ o.matrix_world
print(f"[glb_post] height {(mx.z - mn.z) * scale:.3f} m")

done = set()
for o in meshes:
    for slot in o.material_slots:
        m = slot.material
        if not m or m.name in done or not m.use_nodes:
            continue
        done.add(m.name)
        nt = m.node_tree
        bsdf = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
        base_link = bsdf.inputs["Base Color"].links
        base_img_node = base_link[0].from_node if base_link else None

        if a.emission > 0 and base_img_node is not None:
            nt.links.new(base_img_node.outputs["Color"], bsdf.inputs["Emission Color"])
            bsdf.inputs["Emission Strength"].default_value = a.emission

        if a.roughness >= 0:
            for link in list(bsdf.inputs["Roughness"].links) + list(bsdf.inputs["Metallic"].links):
                nt.links.remove(link)
            bsdf.inputs["Roughness"].default_value = a.roughness
            bsdf.inputs["Metallic"].default_value = 0.0

# scale textures down
for img in bpy.data.images:
    if img.size[0] > a.tex:
        img.scale(a.tex, a.tex)

bpy.ops.export_scene.gltf(
    filepath=a.dst,
    export_format="GLB",
    export_image_format=a.format,
    export_image_quality=a.quality,
    export_yup=True,
)
tris = sum(len(p.vertices) - 2 for o in meshes for p in o.data.polygons)
print(f"[glb_post] OUT {a.dst}  ({tris} triangles, texture {a.tex}px {a.format}, emission {a.emission})")
