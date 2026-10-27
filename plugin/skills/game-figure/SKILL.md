---
name: game-figure
description: Turn a description or image of a character, animal or object into a textured, rigged and animated GLB for a game with the gb10-assets tools (generate_image, image_to_3d, blender). Use when the user wants a 3D game figure, an animated model, a rigged GLB or a talking 3D character.
---

# Animated game figure

Each step is a job on the GB10 server: submit it, then `fetch_result(job_id, target_dir, wait=600)`. Keep all
files of one figure in one directory and look at every image with the Read tool before the next step.

## 1. Image

Skip if the user gives one. `generate_image` with `count` 4, one call:

- the whole figure, standing, uncropped, in a view that shows its shape, plain light background, soft even light
- no props that touch the body, legs apart and visible, mouth closed
- name the style ("stylized low-poly", "plush toy", "clay figure") so the texture fits the game

Show the sheet to the user and let them pick; fix details with `edit_image` instead of generating anew.

## 2. Image to 3D

`image_to_3d` with `max_faces` 10000-50000 and `texture_size` 2048 for game assets (`resolution` 1024 halves the
time for a first try). Check the `.png` render: is the back plausible, are legs separate?

## 3. Clean up

`blender` with `script` `glb_post`, args `[model.glb, figure.glb]` (add `--height` in metres if the size
matters). Then, as needed:

- `tex_adjust` with `--metal 0` when the render looks metallic, `--whiten box` for an invented back
- `tex_eyes` (painted eyes) or `tex_eyecover` + `eye_geo` (eyes as geometry) when the eyes look wrong

## 4. Rig and animate

| Figure | Script | Animations |
|---|---|---|
| four legs (giraffe, pony, sheep, dog) | `rig_quadruped --landmarks <preset or FILE.json>` | Idle, Nod, Talk, Walk |
| two legs (rabbit, elephant standing, humanoid) | `rig_biped --landmarks <preset or FILE.json>` | Idle, Walk |
| anything else, or when the others fail | `rig_hop` | Idle, Walk (hopping) |

Without a fitting preset: measure the model with own `code` (vertices in z/y slices: legs are clusters near the
ground, the head is at the front), write a template with `--dump-landmarks out.json`, adapt it and pass it as an
input file.

## 5. Check

`preview` with `--action Walk --frame N` for two or three frames. Look at each: feet on the ground, no limbs
bending through the body, the figure walks forward (-Y in Blender). Fix landmarks or `--yaw` and rig again.

For a talking figure: `rig_quadruped` with `--save-blend`, then `talk_video` with the `.blend` and a voice
recording (see the voice-lines skill).
