---
name: images
description: Create and edit images for games on the gb10-assets server (generate_image with FLUX.2 klein, edit_image with Qwen-Image-Edit): backgrounds and scenes, characters, items, sprites, icons, UI elements, textures, concept art, variants and edits of existing pictures. Use when the user wants any picture or several variants, a changed or combined image, or a start image for video or 3D.
---

# Images for games

Request all variants in one call (`count` up to 16; with more than one a `sheet.png` is added) and look at the
results, or the sheet, with the Read tool before using them. Then `fetch_result(job_id, target_dir, wait=300)`.

## Pick the tool

| Goal | Tool |
|---|---|
| new image from text | `generate_image` |
| quick recolor or restyle that keeps the composition | `generate_image` with `reference_images` (max 4) |
| change something specific: object, text, background, light, time of day, expression | `edit_image` with 1 image |
| combine: the character from image 1 in the scene of image 2 | `edit_image` with 2-3 images |
| the same character, item or place in a new situation | `edit_image` with the existing image as input |

## One style for the whole game

Agree on a style line with the user once (e.g. "hand-painted 2D adventure art, muted autumn colors, soft
light") and put it unchanged into every prompt of the project; keep it in a file of the project if there is
none yet. Reuse an existing asset as `reference_images` or as input of `edit_image` to match its look.

## Prompts by asset type

Prompts are English: content first, then style, then composition and light. Choose view and background for
what the asset is used for; there is no default view.

| Asset | What to say | Size |
|---|---|---|
| background, scene, adventure location | place, objects, mood, time of day, camera ("wide shot, eye level", "side view for a 2D platformer"); leave room where characters walk; no characters unless wanted | game resolution, e.g. 1920x1088 or 2048x1152, crop afterwards |
| character, creature, item to cut out | the whole subject, uncropped, the view the game needs, plain contrasting background | square or upright, 1024 |
| sprite, icon, UI element | single centered object, clear silhouette, flat or plain background, readable when small | square 1024, scale down afterwards |
| texture, material | "seamless tileable texture of ...", top-down, even light, no perspective; seams are not guaranteed, check them | square 1024-2048 |
| concept art, key art, cover | free composition, as for an illustration | as needed |

`edit_image`: one clear instruction and name the inputs by number ("the cat from image 1 sits on the sofa from
image 2"); unmentioned parts stay as they are. It keeps the first image's aspect ratio by default.

Width and height are multiples of 16 (256-2048). Keep the `seed` of a good variant to vary the prompt alone.

## For later steps

- 3D (`image_to_3d`): exactly one object, whole and uncropped, plain background, no floor shadows; the model
  follows the camera perspective of the image (see the game-figure skill)
- video start frame or keyframes: the video's aspect ratio (default 1536x1024), the same style line and seed for
  every keyframe, change only what moves (see the video skill)
