---
name: voice-lines
description: Create a consistent character voice and speak many lines with it using the gb10-assets tools (design_voice, speak, speak_multilingual). Use when the user wants voice-over, dialogue lines, narration or NPC speech for a game or video, in one or several languages.
---

# Voice lines for a character

## 1. Design the voice

Skip if the user has a recording (3-20 s) and its transcript. Otherwise `design_voice`:

- `voice_description`: gender, age, pitch, tempo, mood, accent, e.g. "gravelly old man, about 70, slow,
  warm, slightly amused"
- `text`: one sentence of 5-15 s in the target language, typical for the character
- `count` 4, `language` set explicitly

Each variant is a `speech_01_<seed>.wav` plus `.txt`. Let the user listen and choose; keep the chosen pair as
the voice reference of this character (e.g. `voices/<name>.wav` and `.txt`).

## 2. Speak the lines

All lines of one character go into one call (max 64 texts): the model loads once per job.

- de, en, es, fr, it, ja, ko, pt, ru, zh: `speak` with `reference_audio`, `reference_text` (the transcript,
  it makes the voice much more similar) and `language`
- other languages (th, vi, id, ms ...) or another language than the reference: `speak_multilingual`; write
  numbers as words outside en/zh/ja

Outputs are numbered in the order of the texts (`speech_<NN>_<seed>.wav`); rename them to the line ids of the
game. Fix a bad line by speaking it again with another `seed`.

## 3. Use it

- talking person in a video: `audio_to_video` with the wav and an image of the character
- talking 3D figure: `talk_video` in `blender` (see the game-figure skill)
