---
name: video
description: "Create videos with sound on the gb10-assets server with LTX-2.5 (generate_video, audio_to_video, control_video, retake_video): from text, from a start image, between keyframes, for a given voice or song, with the motion of a control video (depth, edges, pose), or by regenerating a segment of a video. Use when the user wants a video clip, an animated scene, a cutscene, a trailer shot, a talking or singing person, a figure that moves like a given video or animation, or wants to fix part of a video."
---

# Video with sound (LTX-2.5)

Every video job takes minutes and loads ~50 GB of weights once: put all variants into one call (`count`, max 8)
instead of several calls, then `fetch_result(job_id, target_dir, wait=600)`. Each variant is a
`video_<seed>.mp4` (H.264 + AAC) plus `video_<seed>.jpg` with four frames; look at the jpg with the Read tool,
you cannot play the mp4.

## Pick the tool

| Goal | Tool | Inputs |
|---|---|---|
| scene from text | `generate_video` | `prompt` |
| animate a picture | `generate_video` | `prompt` + `images` = [first frame] |
| from A to B | `generate_video` | `images` = [first, last] (up to 4 keyframes, spread evenly) |
| person speaks or sings a recording | `audio_to_video` | `audio` (+ `image` as first frame) |
| figure from an image moves exactly like a guide | `control_video` | `control_video` (depth, canny or pose) + `image` |
| fix or change 2-4 s of a clip | `retake_video` | `video`, `start_s`, `end_s`, `prompt` |

## Prompts

English, flowing sentences, in this order: subject and look, action over time, camera ("slow dolly in",
"handheld, follows the fox"), light and mood, then sound: ambience, effects and dialogue in quotes
("she says: \"We have to go.\""). The audio is generated from the prompt, so describe it.
With keyframe images, describe the motion between them, not the images. `enhance_prompt` expands a short
prompt.

`control_video`: describe the look (subject, style, setting, light, sound), not the motion; the control video
gives it.

`retake_video` needs the whole scene in the time range (setting, people, the new action and sound), not only
the change; a prompt about the change alone barely alters the video. `regenerate` `video` keeps the audio,
`audio` keeps the picture.

## Size and length

Memory limits the length per size (measured on the GB10; 1536x1024 for 10 s runs out of memory):

| Size | Longest clip at 24 fps |
|---|---|
| 1536x1024 or 1024x1536 (default) | 6 s |
| 1280x832 | 9 s |
| 1024x704 | 13 s |
| 1024x576 | 16 s |
| 768x512 | 20 s |

- `duration_s` 1-20; default predicted from the prompt, capped to what fits the size (5 s with keyframes, the
  audio length for `audio_to_video`)
- default 1536x1024 or the aspect ratio of the first image; for a longer `duration_s` without a size the server
  picks a smaller size that fits. An explicit size with a too long `duration_s` is rejected with the limit, so
  pick a pair from the table. Sides are multiples of 64. For a quick draft use 768x512, then the final size
- higher `fps` means more frames: at 48 fps half the length fits
- `count` does not add memory, the variants run one after another
- `fps` 24 (12-50); keep the `seed` of a good variant to vary only the prompt

## Motion from a control video

`control_video` runs LTX-2.5 with the Union Control IC-LoRA: the output follows the motion and layout of the
control video and takes its size (sides cropped to multiples of 64), length (8k+1 frames) and frame rate; the
memory limits of the table apply to it. The control video is not a normal video but a control signal:

- depth: near white, far black, background black (what Video Depth Anything delivers); best for figures,
  animals and fur. For a game figure render it from its animation: `depth_video` in `blender` (see the
  game-figure skill), no pose detection needed
- canny edges (white lines on black): architecture, products, hard outlines
- pose skeletons (DWPose/OpenPose colors): humans only

`image` sets the look: it becomes the first frame, so the figure in it should stand where and as large as in the
first control frame (`depth_video --color first.png` renders frame 0 for comparison; `--fill` sets the figure's
height). The image also sets the scene: a figure on a studio background stays there, whatever the prompt says, so put
the figure into the wanted setting first (`edit_image`). `control_strength` 1 (default) follows the motion
closely; at 0.7 a walk on the spot already shrinks to barely moving feet, so lower it only for a loose guide.
Motion the control video only hints at stays a hint in the output: make the animation clear (bigger steps,
turns).

## Building blocks from the other tools

- start frame or keyframes: `generate_image` / `edit_image` with the same style words (see the images skill)
- a voice for `audio_to_video`: `design_voice` and `speak` (see the voice-lines skill)
- a talking 3D figure instead of a filmed look: `talk_video` in `blender` (see the game-figure skill)
- frames to process further (keying, sprites): `frames=True` adds `video_<seed>_frames.zip` with every frame
  lossless as PNG; never take frames from the MP4. Animated 2D sprites, walk cycles with root motion
  (`track_points`): see the sprite-animation skill
