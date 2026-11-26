# comfyui-magi2-native

MAGI-2-preview inside ComfyUI's own process, on one 96 GB GPU. No Docker, no HTTP
service. The nodes import the MAGI-2 fork's `inference` package and run its pipeline:
text encoder, preview denoiser (NVFP4 experts, Triton MoE kernel), optional 1080p
refiner, VAE and audio decode.

For two GPUs (about twice as fast) use the service-based pack, `comfyui-magi2-api`.

## What you need

- One GPU with 96 GB (tested: RTX PRO 6000 Blackwell). Peak use is about 82 GiB
  allocated, 86 to 88 GiB reserved; a card that also drives a desktop (3.5 GiB) works.
- About 64 GB of free host RAM (the 52 GB text encoder stays in RAM and runs on the CPU).
- Weights: the pre-quantised NVFP4 checkpoint (67 GiB) and, from the original release,
  `text_encoder/` (52 GB), `vae/`, `turbo_vae/`, `stable-audio-open-1.0/` (9 GB together)
  and, for 1080p, `refiner/` (13 GB). The original 213 GB `preview/` directory is not
  needed.
- The MAGI-2 fork (branch `integration`) at `magi2_fork/` inside this pack (a symlink is
  fine), or anywhere else with `MAGI2_FORK_DIR` pointing at it.
- Python packages: only what ComfyUI already has, plus `triton` and `transformers` 5.x.
  `unfoldNd`, `diffusers`, `flash_attn` and MagiCompiler are **not** needed; the fork
  carries fallbacks for them.

## Nodes (category MAGI-2)

- **MAGI-2 Load (native, 1 GPU)**: `ckpt_root`, `nvfp4_ckpt` (empty = the
  `MAGI2_CKPT_ROOT` / `MAGI2_NVFP4_CKPT` environment variables), `mode` (`540p`, or `1080p`
  which also loads the refiner). Loads in about 11 s from a warm disk cache. It first
  unloads ComfyUI's own models from the card.
- **MAGI-2 Generate (native)**: prompt (plain text or the structured-caption JSON), seed,
  steps, resolution, optional first-frame image, guidance (`-1` = model default: video
  5.0, audio 7.0), `keep_loaded`. Outputs `frames`, `audio`, `fps`; wire them into core
  **Create Video** and **Save Video**.
- **MAGI-2 Unload (native)**: frees the card.

Example workflow (API format): `workflows/magi2_native_api.json`.

## Measured (this host, real weights, fox prompt, seed 42)

| Run | Time | Peak |
|---|---|---|
| 540p, 100 steps, GPU 1 (desktop card) | 36 min 21 s (21.8 s/step) | 81.8 GiB allocated, 87.5 reserved, 90.9 GiB on the card |
| 1080p, 3 preview + 5 refiner steps, GPU 0 | 16 min 49 s | 81.8 GiB allocated, 87.0 GiB on the card |

Both ran through ComfyUI's `/prompt` API in an isolated ComfyUI 0.46.0 instance
(`tools/native_e2e.sh` in the parent directory).

## Limits

- The MAGI-2 models are not ComfyUI model patchers. ComfyUI cannot see their memory, evict
  them, or apply its samplers, schedulers or LoRA nodes to them. Use **Unload** (or
  `keep_loaded` off) before running other large models.
- The fork is imported as the top-level package `inference`; another custom node with a
  package of that name would clash.
- The nodes set `MAGI2_*` environment variables in the ComfyUI process (existing values
  win) and switch fp32 matmuls to TF32 only while MAGI-2 code runs, to match the reference
  Docker image.
- Start ComfyUI with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`; the tests used it,
  and without it fragmentation may cost several GiB.
- Clips are 10 seconds; that is the only duration the model supports.
