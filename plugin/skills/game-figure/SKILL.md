---
name: game-figure
description: Turn a description or image of a character, animal or object into a textured, rigged and animated GLB for a game with the gb10-assets tools (generate_image, image_to_3d, blender), and into videos where the figure from the image moves like its animation (control_video). Use when the user wants a 3D game figure, an animated model, a rigged GLB, a talking 3D character, or a filmed-looking video of a figure with a given motion.
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

`image_to_3d` with `max_faces` 10000-50000 and `texture_size` 2048 for game assets (~2-4
minutes). Check the `.png` render: is the back plausible, are legs separate?

## 3. Clean up

`blender` with `script` `glb_post`, args `[model.glb, figure.glb]` (add `--height` in metres if the size
matters). Then, as needed:

- `tex_adjust` with `--metal 0` when the render looks metallic, `--whiten box` for an invented back
- `retexture_model` when the texture as a whole is wrong (blurry or invented back, baked-in light and shadows,
  wrong colors) or the user wants another look for the same shape: paints a new, consistent texture from a
  prompt on the model's UVs (a few minutes); describe colors and materials per part, not the lighting. Check
  `views_<seed>.png` and the render; it also works on a rigged GLB
- `tex_eyes` (painted eyes) or `tex_eyecover` + `eye_geo` (eyes as geometry) when the eyes look wrong
- `tex_bump` when the figure is rendered with its own material instead of the texture (wood, stone, clay):
  turns the fine contours of the texture (eyes, ornaments, hair strands) into a bump map PNG for the material's
  bump input, so they stay visible as relief; `--glb out.glb` adds them into the model's normal map instead
  (glTF has no bump map). Contours come out crisper from an `image_to_3d` with `texture_guidance` 2-3 (the
  colors then oversaturate, which does not matter here)

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

## 6. Video with the figure's look and the rig's motion

The rig gives exact motion, the source image the look: render the animation as a depth video and let
`control_video` paint the figure from the image over it.

1. `blender` with `script` `depth_video`, args `[rigged.glb, depth.mp4, --action Walk, --seconds 5,
   --res 1024x1024, --fill 0.78, --color first.png]`. The camera looks at the front; `--fill` is the figure's
   height in the picture: match it to the source image (compare `first.png` with it). `--turn 35` turns the
   figure over the clip. `--res` is the video size, `--seconds` its length (memory table of the video skill).
2. `control_video` with `control_video` depth.mp4, `image` the source image (same aspect ratio as `--res`), a
   prompt about the look and the setting, not the motion.

Small motions stay small: for a clear walk raise `rig_biped --stride` (0.9) and `--walk` (0.8 s per cycle).
The frame is set on the first frame, so a hop (`--mode hop`) leaves it at the top: lower `--fill` (and the
figure in the image) for jumps.

