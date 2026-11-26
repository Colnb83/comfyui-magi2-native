# comfyui-magi2-native

MAGI-2-preview inside ComfyUI's own process, on one 96 GB GPU. No Docker, no HTTP
service. Two ways to use it, from the same pack:

1. **As a ComfyUI model (recommended).** The pack teaches ComfyUI the MAGI-2 model type, so
   the stock **Load Diffusion Model** node loads it as a normal `MODEL`: ComfyUI's model
   management loads and unloads it, stock guiders, samplers and the stock Wan2.2 VAE work
   on it. See "Model workflow" below.
2. **As one pipeline node** ("MAGI-2 Generate (native)"): the fork's whole pipeline behind
   a single node. Simpler graph.

For two GPUs (about twice as fast) use the service-based pack, `comfyui-magi2-api`.

## Model workflow

Files (list them with `--extra-model-paths-config` or put them under `models/`):

| File | Folder | Size | Made by |
|---|---|---|---|
| `magi2_preview_nvfp4_comfy.safetensors` | `diffusion_models` | 66.9 GiB | fork `scripts/convert_comfy_checkpoint.py` from the NVFP4 checkpoint |
| `magi2_refiner.safetensors` (1080p only) | `diffusion_models` | 12.6 GiB | fork `scripts/merge_safetensors.py` from the release's `refiner/` shards |
| `Wan2.2_VAE.pth` | `vae` | 2.8 GB | the release, unchanged |

The text encoder (`text_encoder/`, 52 GB) and the audio VAE (`stable-audio-open-1.0/`) are read
from the release's own directories by the MAGI-2 nodes: set `MAGI2_CKPT_ROOT` to the directory
holding them, or fill in the nodes' `ckpt_root`.

Each workflow comes in two forms: `<name>.json` opens in the ComfyUI editor (generated from the
tested API file by `tools/make_ui_workflow.py`; node layout is automatic and it has not been
opened in a browser yet), and `<name>_api.json` is the API format the tests submit.

540p graph (`workflows/magi2_model.json`, `magi2_model_api.json`):

```
Load Diffusion Model (magi2_preview_nvfp4_comfy) ─┐
MAGI-2 Text Encode (prompt) ──────────────────────┤
MAGI-2 Text Encode (default negative prompt) ─────┼─> MAGI-2 Guider (video 5, audio 7) ─┐
Empty MAGI-2 Latent (896x512) ───────────────────────────────────────────────────────────┤
MAGI-2 Sigmas (100 steps, shift 7) ──────────────────────────────────────────────────────┼─> SamplerCustomAdvanced
MAGI-2 Sampler (UniPC) ──────────────────────────────────────────────────────────────────┤
RandomNoise ─────────────────────────────────────────────────────────────────────────────┘
   latent ─> VAE Decode (Load VAE: Wan2.2_VAE.pth) ─> Create Video (fps from Empty MAGI-2 Latent) ─> Save Video
   latent ─> MAGI-2 Audio Decode ───────────────────────┘
```

1080p (`workflows/magi2_model_1080p_api.json`) adds: a second Load Diffusion Model
(`magi2_refiner`), **MAGI-2 Refiner Latent** (1920x1088) on the sampled latent, stock
**CFGGuider** (cfg 2.0), **MAGI-2 Sigmas** (5 steps), **DisableNoise**, and a second
SamplerCustomAdvanced; VAE Decode takes the refined latent, audio still comes from the first.

Notes:

- Stock samplers work too (tested: `euler` through KSamplerSelect). The MAGI-2 sampler and
  sigma nodes reproduce MAGI-2's own UniPC solver and schedule exactly.
- The stock **LTXVDualCFGGuider** can replace MAGI-2 Guider; MAGI-2 Guider adds `cfg_rescale`.
- The latent is ComfyUI's joint audio-video latent, in raw VAE space like every other model.
- Both MAGI-2 models must be entirely on the GPU to run. They load and unload as a whole;
  unloading does not copy the weights into RAM (they are re-read from the file).
- **Image-to-video** (`workflows/magi2_model_i2v_api.json`): connect an image and the Wan2.2
  VAE to the prompt's MAGI-2 Text Encode node and use MAGI-2 Guider (it shows the image to
  the negative branch too, as MAGI-2 does). The image conditions the clip as a reference
  the prompt points at; it is letterboxed, not cropped.
- Not in the model path: ComfyUI's own text-encoder loader (the text encoder runs through
  the MAGI-2 node on the CPU, as in MAGI-2's pipeline). LoRA nodes have no key map for this
  model.

Measured (isolated ComfyUI, `/prompt`, real weights, fox prompt, GPU 1 beside a desktop):

| Run | Time | Peak on the card |
|---|---|---|
| 540p, 100 steps | 32 min 47 s | 90.2 GiB |
| 1080p, 3 preview + 5 refiner steps | 18 min 31 s | 87.9 GiB |
| the same twice in one process (preview reloaded after the refiner displaced it) | 17 min 39 s, then 16 min 41 s | 90.8 GiB |
| 272p, 8 steps, stock euler; second prompt in the same process | 58 s; 36 s | 80.6 GiB |

The denoiser gives the same velocity as MAGI-2's pipeline on identical inputs, to within the
pipeline's own run-to-run difference (8.4e-3 against 8.2e-3 relative, video). The refiner is
bit-identical. Image-to-video: text features and the image's prompt embedding are bit-equal to
the pipeline's, the image latent differs by 6.3e-3 (ComfyUI's VAE against MAGI-2's copy), and
the denoiser output agrees to run-to-run noise (9.3e-3 against 8.6e-3); without the image the
output changes by 38%.

# The pipeline nodes

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

## Pipeline nodes (category MAGI-2)

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
