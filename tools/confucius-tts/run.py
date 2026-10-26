"""Confucius4-TTS worker: cross-lingual zero-shot voice cloning (e.g. a German reference voice speaking Thai).

Weights (download.sh), all in $WEIGHTS_DIR:
  Confucius4-TTS/                  T2S and S2A checkpoints, tokenizer, w2v-BERT feature statistics
  w2v-bert-2.0/                    semantic features of the reference (facebook/w2v-bert-2.0)
  campplus/campplus_cn_common.bin  speaker style encoder (funasr/campplus)
  bigvgan_v2_22khz_80band_256x/    vocoder (nvidia)
Upstream ConfuciusTTS downloads these from the Hub at load time and reads a config file relative to the working
directory; here the config is built in code and the downloads are redirected to the local copies.
Upstream normalizes numbers etc. only for zh and ja, every other language as English ("12 Äpfel" -> "twelve
Äpfel"), so only en/zh/ja texts are normalized; write numbers as words in the other languages.
Outputs: speech_<NN>_<seed>.wav per text (NN = 01, 02 ... in input order) and variant, 22.05 kHz.
"""
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _worker import log, seeds, serve  # noqa: E402

ROOT = Path(os.environ.get("WEIGHTS_DIR", "weights"))
LOCAL = {  # Hub repo -> local directory
    "netease-youdao/Confucius4-TTS": ROOT / "Confucius4-TTS",
    "facebook/w2v-bert-2.0": ROOT / "w2v-bert-2.0",
    "funasr/campplus": ROOT / "campplus",
    "nvidia/bigvgan_v2_22khz_80band_256x": ROOT / "bigvgan_v2_22khz_80band_256x",
}
MODEL = LOCAL["netease-youdao/Confucius4-TTS"]
NORMALIZED = {"en", "zh", "ja"}
# config/inference_config.yaml of the pinned upstream commit, paths pointing to the local copies
CONFIG = {
    "paths": {
        "tokenizer_path": str(MODEL),
        "w2v_bert_path": str(LOCAL["facebook/w2v-bert-2.0"]),
        "w2v_stat": str(MODEL / "wav2vec2bert_stats.pt"),
        "style_encoder": {"target": "external.campplus.CAMPPlus", "init_args": {"feat_dim": 80, "embedding_size": 192},
                          "checkpoint": "campplus_cn_common.bin"},
        "vocoder_path": str(LOCAL["nvidia/bigvgan_v2_22khz_80band_256x"]),
        "t2s_checkpoint": "t2s_model.safetensors",
        "s2a_checkpoint": "s2a_model.pt",
    },
    "t2s_model": {"num_layers": 24, "model_dim": 1280, "num_heads": 20, "max_text_seq_lens": 520,
                  "max_semantic_seq_lens": 1520, "vocab_size": 32000, "semantic_vocab_size": 8194,
                  "text_embedding_dim": 4096, "speaker_embedding_dim": 1024, "start_semantic_token": 8192,
                  "stop_semantic_token": 8193},
    "s2a_model": {"input_size": 512, "output_size": 80, "spk_embed_dim": 192, "semantic_embed_dim": 1024,
                  "lm_latent_dim": 1280, "estimator_mlp_ratio": 3.0},
    "audio": {"target_sample_rate": 22050, "prompt_sample_rate": 16000, "n_fft": 1024, "hop_length": 256,
              "win_length": 1024, "n_mels": 80, "fmin": 0, "fmax": None},
}
REQUIRED = [MODEL / "t2s_model.safetensors", MODEL / "s2a_model.pt", MODEL / "tokenizer.json",
            LOCAL["facebook/w2v-bert-2.0"] / "model.safetensors", LOCAL["funasr/campplus"] / "campplus_cn_common.bin",
            LOCAL["nvidia/bigvgan_v2_22khz_80band_256x"] / "bigvgan_generator.pt"]


def local_file(repo_id, filename, **_):
    path = LOCAL[repo_id] / filename
    if not path.exists():
        raise FileNotFoundError(path)
    return str(path)


def load_prompt(self, prompt_wav):
    """Replaces ConfuciusTTS._load_prompt: torchaudio.load needs torchcodec since torchaudio 2.9; librosa reads
    wav, flac, ogg and mp3. Returns mono (1, T) tensors at 16 kHz and at the output rate."""
    import librosa
    import torch

    wav, sr = librosa.load(prompt_wav, sr=None, mono=True)
    return tuple(torch.from_numpy(librosa.resample(wav, orig_sr=sr, target_sr=rate) if rate != sr else wav)[None]
                 for rate in (16000, self.sample_rate))


def load():
    import yaml
    from confuciustts.cli import inference

    missing = [str(p) for p in REQUIRED if not p.exists()]
    if missing:
        raise FileNotFoundError(f"{', '.join(missing)} missing, install the tool on the GB10 system: "
                                "scripts/docker.sh install confucius-tts")
    inference.hf_hub_download = local_file
    inference.ConfuciusTTS._load_prompt = load_prompt
    with tempfile.NamedTemporaryFile("w", suffix=".yaml") as f:
        yaml.safe_dump(CONFIG, f)
        f.flush()
        return inference.ConfuciusTTS(config_path=f.name, device="cuda")


def run(model, params, job_dir):
    import soundfile as sf
    import torch

    lang = params["language"]
    ref = str(job_dir / "in" / params["reference_audio"])
    names, variants = [], seeds(params)
    for i, text in enumerate(params["texts"], 1):
        for seed in variants:
            t = time.time()
            torch.manual_seed(seed)
            audio = model.generate(text, lang, ref, raw=lang not in NORMALIZED)
            path = job_dir / "out" / f"speech_{i:02d}_{seed}.wav"
            sf.write(path, audio.squeeze(0).float().cpu().numpy(), model.sample_rate)
            names.append(path.name)
            log(f"{path.name}: {audio.shape[-1] / model.sample_rate:.1f}s audio in {time.time() - t:.1f}s")
    return names


if __name__ == "__main__":
    serve(load, run)
