---
name: sprite-animation
description: "Turn a character into animated 2D sprites for a game with the gb10-assets tools (generate_image, remove_background, generate_video with frames) and the bundled sprites.py: concept image on a chroma background, image-to-video walking in place, lossless frames, cycle selection, chroma key with despill, cleanup, scaling, quality check and export as sprite sheet and SpriteFrames for Godot. Use when the user wants a walk cycle, idle or other animation of a 2D character, animated sprites or a sprite sheet."
---

# Animated 2D sprites

Server steps are jobs: submit, then `fetch_result(job_id, target_dir, wait=600)`. The local steps run
`scripts/sprites.py` from this skill's directory with `uv run <skill dir>/scripts/sprites.py COMMAND ...` (uv
installs numpy, Pillow and SciPy itself). Keep all files of one character in one directory and look at every
image and contact sheet with the Read tool before the next step.

## 1. Concept image

`generate_image` (or `edit_image` from an existing picture of the character), `count` 4:

- three-quarter view, camera height matching the perspective of the game's scenes, even light, no floor shadow
- the whole figure, uncropped, with some margin
- background: one flat color that does not occur in the figure, above all not at its edges. Default green
  `#00FF00`; magenta `#FF00FF` only if the figure contains green (also olive or khaki clothes). In the prompt:
  "solid flat pure green (#00FF00) background, no gradient, no shadow"

View and camera height are the same for every animation of a character: keep the chosen concept image and
reuse it.

## 2. Fix the background if needed

FLUX rarely paints the exact color and often adds a floor shadow or a gradient. Check with
`sprites.py check-bg concept.png --color 00FF00`. If it reports a problem: `remove_background` on the concept
image, check the cutout by eye, then `sprites.py flatten concept_cutout.png reference.png --color 00FF00`. This
RGB image on the exactly flat chroma color is the reference for every video of the character.

## 3. Video

`generate_video` (see the video skill) with `images` = [reference, reference] for a loop (same first and last
frame), `frames=True`, the size of the reference (e.g. 1024x1024), 3 s and the default 24 fps (73 frames, about
2 minutes, a zip of about 70 MB). The prompt must keep the character in place, otherwise it walks out of the
frame:

"The [character] walks in place with a steady walking cycle, as on a treadmill, legs and arms swinging
alternately; it stays in the center of the frame and keeps facing [front-right]. Static camera, three-quarter
view, solid flat pure [green (#00FF00)] background, no shadow, no other objects."

Use the chroma color of the reference in the prompt. If the video has no clean cycle (the character turns,
drifts or changes), try another seed before changing the prompt.

## 4. Frames

`frames=True` returns `video_<seed>_frames.zip`: every frame lossless as PNG, as it left the VAE. Never take
frames from the MP4 (H.264 smears the edges and the chroma color).
`sprites.py unpack video_<seed>_frames.zip frames`, then `sprites.py sheet frames --every 3 --out sheet.png`
to look at them.

## 5. Select frames

One clean cycle, 8-12 frames for a walk cycle. `sprites.py cycle frames` lists loop lengths where the motion
repeats and steps that divide them evenly (otherwise the loop stutters). Take a cycle from the middle: the
frames near the first and last keyframe often have ghosting. Then
`sprites.py unpack video_<seed>_frames.zip walk --range 39-65 --step 3` and look at `sheet walk`. Check feet
and legs closely: they overlap in the three-quarter view (a leg missing, doubled or sliding).

## 6. Check the background

`sprites.py check-bg walk --color 00FF00`: background even, no gradient, figure not cut off at the edge, in
every selected frame. The video's background is usually a few values off the exact color; the key measures it
per frame. If a frame has a problem, report it to the user instead of keying it anyway.

## 7.-8. Cut out and clean up

`sprites.py key walk --color 00FF00 --out keyed`: chroma key with despill (colors at soft edges are unmixed from
the background), then only the largest connected region and 1 px alpha erosion. Its "unsure" column marks frames
with soft alpha inside the figure (hair, motion blur): only for those run `remove_background` with `mask=True`,
put the `<frame>_mask.png` files into their own directory and key again with `--ml masks`; the masks then
replace the key's alpha where it is unsure.

## 9. Scale

`sprites.py scale keyed --out scaled --height 192`: one crop box for all frames (the figure does not jump) and
premultiplied-alpha resizing. It prints the factor; pass it as `--factor` for the other animations and
directions of the character so that all have the same scale.

## 10. Quality check

`sprites.py qc scaled --out qc.png --scene scene.png [--at x,y]`: every frame on black, white, magenta, green and
a 1:1 crop of a real scene of the game. Show the contact sheet to the user; look for colored fringes, holes and
missing feet.

## 11. Export for Godot

`sprites.py export --anim walk_front=scaled --anim walk_back=scaled_back --out godot --name fox --fps 8
--res res://sprites/fox/`: per animation a sprite sheet `<name>.png` and the single frames, plus `fox.tres`, a
SpriteFrames resource with all animations for `AnimatedSprite2D`. Copy the files to the `--res` folder of the
project.

`--fps` is the video's frame rate divided by the step of step 5 (24 fps, every 3rd frame: 8); a higher value
plays the walk faster than it was generated. The character walks in place: the game moves the node, at a speed
matched to the stride so that the feet do not slide over the ground.

## Directions

Movement in depth needs at least two directions: front three-quarter and back three-quarter, each its own
video from its own reference (same character, same camera height). Left and right come from mirroring
(`flip_h` of `AnimatedSprite2D`), as long as the character has no asymmetric details (a sword in one hand, an eye
patch, a logo); with asymmetric details ask the user.
