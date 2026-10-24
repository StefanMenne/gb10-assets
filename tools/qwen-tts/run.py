"""Qwen3-TTS 1.7B worker: text to speech with a designed voice (VoiceDesign) or a cloned voice (Base).

Weights: $MODELS_DIR/qwen-tts/Qwen3-TTS-12Hz-1.7B-VoiceDesign and .../Qwen3-TTS-12Hz-1.7B-Base (download.sh),
both loaded at start (2 x 4.5 GB bf16), because designing a voice and then cloning it is the usual workflow.
VoiceDesign takes voice_description; cloning takes reference_audio plus optionally reference_text (without the
transcript only the speaker embedding is used). The clone prompt is built once per job.
Outputs: speech_<NN>_<seed>.wav per text (NN = 01, 02 ... in input order) and variant; with voice_description also
speech_<NN>_<seed>.txt holding the spoken text, so a designed voice can serve directly as a clone reference.
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _worker import log, seeds, serve  # noqa: E402

ROOT = Path(os.environ.get("MODELS_DIR", "models")) / "qwen-tts"
DESIGN = ROOT / "Qwen3-TTS-12Hz-1.7B-VoiceDesign"
BASE = ROOT / "Qwen3-TTS-12Hz-1.7B-Base"
# ISO 639-1 codes of the request -> Qwen3-TTS language names
LANGUAGES = {"auto": "Auto", "de": "German", "en": "English", "es": "Spanish", "fr": "French", "it": "Italian",
             "ja": "Japanese", "ko": "Korean", "pt": "Portuguese", "ru": "Russian", "zh": "Chinese"}


def load():
    import torch
    from qwen_tts import Qwen3TTSModel

    for path in (DESIGN, BASE):
        if not (path / "model.safetensors").exists():
            raise FileNotFoundError(f"{path} missing, download it on the GB10 system: "
                                    "scripts/docker.sh download qwen-tts")
    # sdpa: flash-attn has no aarch64 wheels
    kw = {"device_map": "cuda", "dtype": torch.bfloat16, "attn_implementation": "sdpa"}
    return {"design": Qwen3TTSModel.from_pretrained(str(DESIGN), **kw),
            "base": Qwen3TTSModel.from_pretrained(str(BASE), **kw)}


def run(models, params, job_dir):
    import soundfile as sf
    import torch

    language = LANGUAGES[params.get("language", "auto")]
    design = params.get("voice_description")
    if design:
        def speak(text):
            return models["design"].generate_voice_design(text=text, language=language, instruct=design)
    else:
        ref_text = params.get("reference_text")
        prompt = models["base"].create_voice_clone_prompt(
            ref_audio=str(job_dir / "in" / params["reference_audio"]), ref_text=ref_text,
            x_vector_only_mode=ref_text is None)

        def speak(text):
            return models["base"].generate_voice_clone(text=text, language=language, voice_clone_prompt=prompt)

    names, variants = [], seeds(params)
    for i, text in enumerate(params["texts"], 1):
        for seed in variants:
            t = time.time()
            torch.manual_seed(seed)
            wavs, sr = speak(text)
            path = job_dir / "out" / f"speech_{i:02d}_{seed}.wav"
            sf.write(path, wavs[0], sr)
            names.append(path.name)
            if design:
                path.with_suffix(".txt").write_text(text + "\n")
                names.append(path.with_suffix(".txt").name)
            log(f"{path.name}: {len(wavs[0]) / sr:.1f}s audio in {time.time() - t:.1f}s")
    return names


if __name__ == "__main__":
    serve(load, run)
