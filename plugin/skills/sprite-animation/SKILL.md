---
name: sprite-animation
description: "Turn a character into animated 2D sprites for a game with the gb10-assets tools (generate_image_qwen, edit_image, remove_background, generate_video with frames, track_points) and the bundled sprites.py: reference views for several directions, image-to-video walking in place (with starting and stopping) or across the frame, turning on the spot, idle, talking and gestures, lossless frames, root motion (walking speed without sliding feet), cycle selection, chroma key with despill, cleanup, scaling, quality check and export as sprite sheet and SpriteFrames for Godot. Use when the user wants a walk cycle, idle, turn or other animation of a 2D character, animated sprites, a sprite sheet or root motion for a walk."
---

# Animated 2D sprites

Server steps are jobs: submit, then `fetch_result(job_id, target_dir, wait=600)`. The local steps run
`scripts/sprites.py` from this skill's directory with `uv run <skill dir>/scripts/sprites.py COMMAND ...` (uv
installs numpy, Pillow and SciPy itself). Keep all files of one character in one directory and look at every
image and contact sheet with the Read tool before the next step. Proven prompts for views, walks, turns, idle,
talking and gestures: [references/prompts.md](references/prompts.md).

Stop for the user's choice after the concept image, after each view and after each video, with the variants
side by side and your recommendation. For videos, first make one direction (front-right) and only then the
others. The same seed gives pixel-identical results, also with `count` 1 instead of 4: note the chosen seed.

## Two ways to film a walk

| The figure moves in the game | Video | Why |
|---|---|---|
| sideways, parallel to the screen (side view, platformer) | **walking across**: the figure really walks through the frame, static camera | natural gait, real root motion, no treadmill artefacts (drifting, marching on the spot) |
| into the depth or diagonally (three-quarter front or back, away from or towards the camera) | **walking in place**, as on a treadmill | walking away the figure gets smaller and is seen more from above; the game moves and scales the sprite instead |

Both: same camera height and tilt as the game's scenes, static camera (no pan, zoom or dolly), so that the
measured speed in screen pixels fits the game. Idle and other animations without travel: no root motion.

## 1. Concept image and reference

`generate_image_qwen` (sharper and more realistic than `generate_image`; or `edit_image` from an existing
picture of the character), `count` 4, 3:4 (1104x1472) for walking in place:

- the view of the animation: profile for walking across, three-quarter front-right (FR) for walking in place;
  camera height matching the perspective of the game's scenes, even light, no floor shadow
- the whole figure, uncropped, with some margin
- background: one flat color that does not occur in the figure, above all not at its edges. Default green
  `#00FF00`; magenta `#FF00FF` only if the figure contains green (also olive or khaki clothes). In the prompt:
  "solid flat pure green (#00FF00) background, no gradient, no shadow"

The models rarely paint the exact color, add shadows or gradients and put the feet near the edge. So always
`remove_background`, check the cutout by eye, then place it at a fixed size on the exactly flat color:

- in place: `sprites.py place cutout.png reference.png --size 1152x1536 --at center --height 0.75 --bottom 0.1
  --color 00FF00` (`--height`, `--bottom` as fractions or in px, or `--feet Y`). Leave this margin below the feet:
  feet stepping or turning towards the camera reach lower and would be cut off.
- across: `--size 1536x640 --at left --height 0.55` (`--at right` for walking to the left). A figure of about
  half the frame height crosses about 2/3 of the width in 3 s.

This RGB image is the reference for every video of this view. View and camera height are the same for every
animation of a character.

## 2. Views for several directions

Movement into the depth needs at least front-right (FR) and back-right (BR), often also the back view (B, for
turns). Make each new view with `edit_image` from the **nearest view already chosen**, never all from the first:
BR from FR, then B from BR. The back is not visible in FR; generated twice it comes out different (vent, hair,
shoes), generated once in BR, B inherits it. Let the user choose BR before making B.

- Back views: "heels toward the viewer, toes of both shoes pointing slightly outward", otherwise the feet often
  point inward. B: "perfectly symmetrical pose"; take the most symmetrical variant.
- Pass `width`/`height` (the reference size), then `remove_background` and `place` with exactly the same
  parameters as FR: all views share scale and feet line, so no animation needs its own scaling.
- Left views: `sprites.py shift fr.png fl.png --mirror` (lossless). Asymmetric details (a sword in one hand, a
  button placket, a logo) switch sides: ask the user. In the game left animations are `flip_h`.
- Another outfit of the same character: `edit_image` with two images, [original view, outfit view already
  chosen], e.g. BR of the outfit from [br.png, outfit_fr.png]; for B say "change only the clothes", otherwise it
  takes the pose of image 2.

## 3. Video

`generate_video` (see the video skill) with `frames=True`, the size of the reference, default 24 fps, `count`
2-4. Use the chroma color of the reference in the prompt. If the video has no clean cycle (the character turns,
drifts or changes), try another seed before changing the prompt.

Walking in place, 1152x1536: resolution helps the feet more than anything else (at 1024x1024 or 768x1024 the
swinging foot smears; slow motion, 48 fps or "crisp shoes" prompts do not help). Preferred: one 5 s video per
direction with `images` = [reference, reference] that **stands, starts walking, walks, stops and stands** again
(prompt in the references). Start and stop then fit the cycle seamlessly, without extra videos. Just the cycle:
3 s with the treadmill prompt. Back views tend to drift to profile or straight back: insist in the prompt
("stays turned three-quarters toward the right ... never turns his back straight toward the viewer"), new seed.

Walking across: `images` = [reference_wide] only (the end is free), 1536x640, 3 s:

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
--out sheet.png` to look at them. The first and last frames often have ghosting: do not use them.

A video from elsewhere that exists only as MP4: `ffmpeg -i in.mp4 -start_number 0 frames/frame_%04d.png`. Its
background color may drift and jump in the last frames (drop them); `key` measures the color per frame.

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
another seed. Compare the seeds of a job and take the one without warnings. Walking away from the camera (back
views) the steps are foreshortened: a lower speed and "feet hardly push back" are normal there.

A video that stands, starts, walks and stops: track each segment on its own (step 6 splits it; `queries --every
2`). The cycle with `root` as above; start and stop with `sprites.py root tracks_start.json --queries
queries_start.json --foot --like root_cycle.json --out root_start.json`: per frame from the feet (0 while
standing), in the cycle's direction and scale. Look at the numbers: a frame where a foot is lost may need a
correction by hand. The speed of generated walks varies a lot between directions and seeds; the game can stretch
the root motion of a path a little (and the frame rate alike) so that whole cycles fit.

## 6. Select frames

One clean cycle, 8-12 frames for a walk cycle. Walking across: first `sprites.py key frames --color 00FF00 --out
keyed` and `sprites.py stabilize keyed --root root.json --out stable` (the figure then walks in place), and look
for the cycle in `stable`. `sprites.py cycle frames` (or `stable`) lists loop lengths where the motion repeats and
steps that divide them evenly (otherwise the loop stutters). Take a cycle from the middle. Then `sprites.py unpack
video_<seed>_frames.zip walk --range 39-65 --step 3` and look at `sheet walk`. Check feet and legs closely: they
overlap in the three-quarter view (a leg missing, doubled or sliding).

Splitting a stand-start-walk-stop video: the cycle from the middle of the walking part; start = from frame 0
(which is also the standing frame of this direction) up to about 3 frames before the cycle; stop = not right
after the cycle but from the next frame in phase with the cycle start (compare silhouettes), so it fits every
cycle end and is shorter. Always check frame numbers on the keyed frames, not from notes.

## 7. Check the background

`sprites.py check-bg walk --color 00FF00`: background even, no gradient, figure not cut off at the edge, in
every selected frame (a foot stepping towards the camera close to the edge is also reported: check by eye). The
video's background is usually a few values off the exact color; the key measures it per frame. If a frame has a
problem, report it to the user instead of keying it anyway.

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

All animations of a character from references placed alike (step 1-2): one fixed box and factor for all of them,
otherwise the feet point and the sprite size differ between animations and the figure jumps when they change:
`sprites.py scale keyed --out scaled --box 0,0,1152,1536 --factor 0.25 --anchor 576,1382` (the whole frame, the
anchor at the feet line of `place`). It prints the anchor in sprite pixels: the offset of the sprite in the game.
Choose the factor from the wanted figure height (figure px in the reference x factor).

A single animation, or walking across: `sprites.py scale keyed --out scaled --height 192` crops to the figure
(one box for all its frames) and prints the factor; pass it as `--factor` for the other animations.
Pass the factor to `export --factor` for the root motion.

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

## Turning on the spot

`images` = [from view, to view], 2 s, the prompt names the way ("turns away from the viewer toward the right" or
"turns to face the viewer ... never shows us his back") and "feet stay on the same spot, one or two small steps".
Six videos cover all 20 turns between FR, FL, BR, BL and B: FR→BR, FR→B, FR→FL, FR→BL, BR→B, BR→BL; the opposite
direction is the same frames played backwards, the left variants are mirrored. LTX turns FR→FL always via the
back: make a frontal view F (`edit_image` from FR) and use it as middle keyframe, [fr, f, fl]. Turning while
walking is better done in the game as stop, turn on the spot, start.

## Idle, talking, gestures

Loops (idle, talking): `images` = [stand, stand], 4 s; the last frame equals the first, so drop it. Gestures
(take, use, point): 3 s, use all of it. The idle prompt must ask for almost no movement, otherwise it is far too
restless. Talking only in views that show the face; elsewhere use the idle. Watch for objects appearing in empty
hands. Every 2nd frame (12 fps) is usually enough.
