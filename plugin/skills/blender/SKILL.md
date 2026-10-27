---
name: blender
description: Run Blender 5.2 (bpy, headless, GPU) on the gb10-assets server with the blender tool, with a bundled script or own Python code, on GLB, FBX, OBJ, blend or image files. Use when the user wants to inspect, measure, convert, repair, retexture, rig, animate or render a 3D model, or anything else that needs Blender.
---

# Blender on the server

`blender(files, script | code, args)` returns a job id; `fetch_result(job_id, target_dir, wait=600)` downloads
every NEW file the job wrote plus `log` (the end of the print output). The job runs in a directory holding the
input files under their file names; refer to them by name in `args` and write results under new names.

## Bundled scripts

The list with the usage of every script is part of the `blender` tool description, so no extra call is
needed; on the command line `gb10-assets blender --list` prints it. A script fails with its usage text on wrong
arguments, so a first try with the obvious arguments is fine.

| Need | Script |
|---|---|
| put a generated model on the ground, centre, scale, smaller textures | `glb_post` |
| look at a model or an animation frame | `preview` (`--views 4`, `--action Walk --frame 6`) |
| rig and animate | `rig_quadruped`, `rig_biped`, `rig_hop` (see the game-figure skill) |
| texture: recolor, remove metal, whiten an invented back | `tex_adjust` |
| eyes | `tex_eyes`, `tex_eyecover`, `eye_geo` |
| hair strands | `tex_hair` |
| talking figure video from a voice recording | `talk_video` (needs the `--save-blend` of `rig_quadruped`) |

After every step that changes the look, render a `preview` and look at it with the Read tool.

## Own code

Instead of a script, pass Python directly as `code` (a string; on the command line `gb10-assets blender
in.glb --code my_script.py -o DIR -- ARGS`) when no script fits: measuring, format conversion (glTF, FBX, OBJ, USD, STL),
decimation, UV or material fixes, procedural models, renders with a custom camera.

- Blender starts from an empty scene: import first (`bpy.ops.import_scene.gltf(filepath="in.glb")`), export at
  the end (`bpy.ops.export_scene.gltf(filepath="out.glb")`)
- arguments arrive in `sys.argv[1:]` (from `args`)
- available: bpy, bmesh, mathutils, numpy, scipy, trimesh, PIL and `import texlib` (texture helpers of the
  bundled scripts)
- `print()` what you need to know (sizes, counts, bounding boxes); it comes back in `log`
- Cycles on the GPU: `prefs = bpy.context.preferences.addons["cycles"].preferences`,
  `prefs.compute_device_type = "CUDA"`, `prefs.get_devices()`, enable all `prefs.devices`, then
  `scene.cycles.device = "GPU"`; EEVEE needs nothing
- conventions: Blender Z up, front = -Y; glTF Y up, front = +Z

Start with a small measuring script before a large change: one cheap job that prints the bounding box, the
objects, materials and vertex counts saves several failed ones.

A `403` means the server allows only bundled scripts (`BLENDER_CODE=0`); say so and use a script instead.
