---
name: video
description: Create videos with sound on the gb10-assets server with LTX-2.5 (generate_video, audio_to_video, retake_video): from text, from a start image, between keyframes, for a given voice or song, or by regenerating a segment of a video. Use when the user wants a video clip, an animated scene, a cutscene, a trailer shot, a talking or singing person, or wants to fix part of a video.
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
| fix or change 2-4 s of a clip | `retake_video` | `video`, `start_s`, `end_s`, `prompt` |

## Prompts

English, flowing sentences, in this order: subject and look, action over time, camera ("slow dolly in",
"handheld, follows the fox"), light and mood, then sound: ambience, effects and dialogue in quotes
("she says: \"We have to go.\""). The audio is generated from the prompt, so describe it.
With keyframe images, describe the motion between them, not the images. `enhance_prompt` expands a short
prompt.

`retake_video` needs the whole scene in the time range (setting, people, the new action and sound), not only
the change; a prompt about the change alone barely alters the video. `regenerate` `video` keeps the audio,
`audio` keeps the picture.

## Size and length

- `duration_s` 1-20; default predicted from the prompt (5 s with keyframes, the audio length for
  `audio_to_video`)
- default 1536x1024 or the aspect ratio of the first image; sides multiples of 64. For a quick draft use
  768x512 and short clips, then the full size for the chosen idea
- `fps` 24 (12-50); keep the `seed` of a good variant to vary only the prompt

## Building blocks from the other tools

- start frame or keyframes: `generate_image` / `edit_image` with the same style words (see the images skill)
- a voice for `audio_to_video`: `design_voice` and `speak` (see the voice-lines skill)
- a talking 3D figure instead of a filmed look: `talk_video` in `blender` (see the game-figure skill)
