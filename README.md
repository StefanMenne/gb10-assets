# gb10-assets

Self-hosted generative asset pipeline (image, 3D, video, speech, music) for NVIDIA DGX Spark, ASUS Ascent GX10 and other GB10 systems.

Work in progress.

## Development

```sh
uv run --frozen --group dev pytest
```

## Model weights

```sh
scripts/download-all.sh            # all tools, or name them: scripts/download-all.sh flux2-klein
```

Weights go to `<repo>/models` unless `MODELS_DIR` is set; the server uses the same default. The download
scripts need [uv](https://docs.astral.sh/uv/) on the host. Gated models need `HF_TOKEN`.
