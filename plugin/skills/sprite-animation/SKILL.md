---
name: sprite-animation
description: "Turn a character into animated 2D sprites for a game with the gb10-assets tools (generate_image, remove_background, generate_video with frames, track_points) and the bundled sprites.py: concept image on a chroma background, image-to-video walking in place or across the frame, lossless frames, root motion (walking speed without sliding feet), cycle selection, chroma key with despill, cleanup, scaling, quality check and export as sprite sheet and SpriteFrames for Godot. Use when the user wants a walk cycle, idle or other animation of a 2D character, animated sprites, a sprite sheet or root motion for a walk."
---

# Animated 2D sprites

Server steps are jobs: submit, then `fetch_result(job_id, target_dir, wait=600)`. The local steps run
`scripts/sprites.py` from this skill's directory with `uv run <skill dir>/scripts/sprites.py COMMAND ...` (uv
installs numpy, Pillow and SciPy itself). Keep all files of one character in one directory and look at every
image and contact sheet with the Read tool before the next step.

## Two ways to film a walk

| The figure moves in the game | Video | Why |
|---|---|---|
| sideways, parallel to the screen (side view, platformer) | **walking across**: the figure really walks through the frame, static camera | natural gait, real root motion, no treadmill artefacts (drifting, marching on the spot) |
| into the depth or diagonally (three-quarter front or back, away from or towards the camera) | **walking in place**, as on a treadmill | walking away the figure gets smaller and is seen more from above; the game moves and scales the sprite instead |

Both: same camera height and tilt as the game's scenes, static camera (no pan, zoom or dolly), so that the
measured speed in screen pixels fits the game. Idle and other animations without travel: walking in place
steps only, without root motion.

## 1. Concept image

`generate_image` (or `edit_image` from an existing picture of the character), `count` 4:

- the view of the animation: profile for walking across, three-quarter for walking in place; camera height
  matching the perspective of the game's scenes, even light, no floor shadow
- the whole figure, uncropped, with some margin
- background: one flat color that does not occur in the figure, above all not at its edges. Default green
  `#00FF00`; magenta `#FF00FF` only if the figure contains green (also olive or khaki clothes). In the prompt:
  "solid flat pure green (#00FF00) background, no gradient, no shadow"

View and camera height are the same for every animation of a character: keep the chosen concept image and
reuse it.

## 2. Fix the background if needed

FLUX rarely paints the exact color and often adds a floor shadow or a gradient (or green clothes on a green
background: then switch to magenta). Check with `sprites.py check-bg concept.png --color 00FF00`. If it reports
a problem: `remove_background` on the concept image, check the cutout by eye, then
`sprites.py flatten concept_cutout.png reference.png --color 00FF00`. This RGB image on the exactly flat chroma
color is the reference for every video of the character.

Walking across needs room in the walking direction: `remove_background`, then
`sprites.py place concept_cutout.png reference_wide.png --size 1536x640 --at left --height 0.55 --color 00FF00`
(`--at right` for walking to the left). A figure of about half the frame height crosses about 2/3 of the width
in 3 s.

## 3. Video

`generate_video` (see the video skill) with `frames=True`, the size of the reference, 3 s and the default 24 fps
(73 frames, about 2 minutes), `count` 2. Use the chroma color of the reference in the prompt. If the video has no
clean cycle (the character turns, drifts or changes), try another seed before changing the prompt.

Walking in place: `images` = [reference, reference] (same first and last frame, 1024x1024):

"The [character] walks in place with a steady walking cycle, as on a treadmill, legs and arms swinging
alternately; it stays in the center of the frame and keeps facing [front-right]. Static camera, three-quarter
view, solid flat pure [green (#00FF00)] background, no shadow, no other objects."

Walking across: `images` = [reference_wide] only (the end is free), 1536x640:

"The [character] walks from left to right across the frame with a steady natural walking cycle, legs and arms
swinging alternately, side view in profile, it keeps facing right. Static locked-off camera shot from far away
with a long telephoto lens, no camera movement, no zoom. Solid flat pure [green (#00FF00)] background, no shadow,
no floor, no other objects."

The telephoto lens keeps the perspective flat; the figure must walk exactly across the view, at least one whole
cycle fully inside the frame.

## 4. Frames

`frames=True` returns `video_<seed>_frames.zip`: every frame lossless as PNG, as it left the VAE. Never take
frames from the MP4 (H.264 smears the edges and the chroma color).
`sprites.py unpack video_<seed>_frames.zip frames` (all frames), then `sprites.py sheet frames --every 3
--out sheet.png` to look at them.

## 5. Root motion

How far the figure moves over the ground per frame, measured on the standing foot (it moves with the ground),
so that the game moves the sprite exactly as fast as its feet push and nothing slides.

1. `sprites.py queries frames --color 00FF00 --out queries.json`: points on the feet and the body.
2. `track_points(frames=<frames dir>, queries=queries.json)` -> `tracks.json` and `tracks.png` (a few seconds).
   Look at `tracks.png`: the dots must stay on the feet and the body.
3. `sprites.py root tracks.json --queries queries.json --out root.json`: prints the root motion in px per frame
   and whether the figure walks in place or across.

Warnings and what to do: "marches on the spot" (the feet are lifted but do not push back) or "the standing foot
slides" over 15% (common in generated videos): another seed. "the ground speed varies": look at the steps, often
another seed. Compare both seeds of a job and take the one without warnings.

## 6. Select frames

One clean cycle, 8-12 frames for a walk cycle. Walking across: first `sprites.py key frames --color 00FF00 --out
keyed` and `sprites.py stabilize keyed --root root.json --out stable` (the figure then walks in place), and look
for the cycle in `stable`. `sprites.py cycle frames` (or `stable`) lists loop lengths where the motion repeats and
steps that divide them evenly (otherwise the loop stutters). Take a cycle from the middle: the frames near the
first and last keyframe often have ghosting. Then `sprites.py unpack video_<seed>_frames.zip walk --range 39-65
--step 3` and look at `sheet walk`. Check feet and legs closely: they overlap in the three-quarter view (a leg
missing, doubled or sliding).

## 7. Check the background

`sprites.py check-bg walk --color 00FF00`: background even, no gradient, figure not cut off at the edge, in
every selected frame. The video's background is usually a few values off the exact color; the key measures it
per frame. If a frame has a problem, report it to the user instead of keying it anyway.

## 8.-9. Cut out and clean up

`sprites.py key walk --color 00FF00 --out keyed`: chroma key with despill (colors at soft edges are unmixed from
the background), then only the largest connected region and 1 px alpha erosion. Its "unsure" column marks frames
with soft alpha inside the figure (hair, motion blur): only for those run `remove_background` with `mask=True`,
put the `<frame>_mask.png` files into their own directory and key again with `--ml masks`; the masks then
replace the key's alpha where it is unsure.

Walking across: then `sprites.py stabilize keyed --root root.json --out stable` and continue with `stable`. It
also evens out a small change of size (walking slightly towards or away from the camera) and refuses more than
10%: then film this direction walking in place.

## 10. Scale

`sprites.py scale keyed --out scaled --height 192`: one crop box for all frames (the figure does not jump) and
premultiplied-alpha resizing. It prints the factor; pass it as `--factor` for the other animations and
directions of the character so that all have the same scale, and to `export --factor` for the root motion.

## 11. Quality check

`sprites.py qc scaled --out qc.png --scene scene.png [--at x,y]`: every frame on black, white, magenta, green and
a 1:1 crop of a real scene of the game. Show the contact sheet to the user; look for colored fringes, holes and
missing feet.

## 12. Export for Godot

`sprites.py export --anim walk_front=scaled --anim walk_side=scaled_side --out godot --name fox --fps 8
--res res://sprites/fox/ --root walk_front=root.json --root walk_side=root_side.json --factor <scale factor>`:
per animation a sprite sheet `<name>.png` and the single frames, plus `fox.tres`, a SpriteFrames resource with
all animations for `AnimatedSprite2D`, and with `--root` a `<name>_root.json`: `velocity_px_s` (x right, y down,
in sprite pixels) and `per_frame_px` (the movement while each frame is shown). Copy the files to the `--res`
folder of the project.

`--fps` is the video's frame rate divided by the step of step 6 (24 fps, every 3rd frame: 8); a higher value
plays the walk faster than it was generated, then scale the velocity alike. The game moves the node by
`velocity_px_s` (or by `per_frame_px` on each frame change for an exact match); with `flip_h` negate x. For a
walk into the depth the vector is the screen movement at the size of the sprite; if the game scales the sprite
with the depth, scale the movement alike.

## Directions

Movement in depth needs at least two directions: front three-quarter and back three-quarter, each its own
video from its own reference (same character, same camera height), walking in place. Sideways movement: walking
across in profile. Left and right come from mirroring (`flip_h` of `AnimatedSprite2D`), as long as the character
has no asymmetric details (a sword in one hand, an eye patch, a logo); with asymmetric details ask the user.
