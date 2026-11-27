# Training adapters (LoRA) for MAGI-2-preview on one 96 GB GPU

**Experimental.** The pipeline runs end to end and the numbers below are measured. One
character LoRA has been trained with it so far (see "First result"); the step count,
learning rate and data mix are first guesses, not tuned values.

What it does: trains low-rank adapters on the non-expert weights (attention and MLP linears)
and, optionally, native parameter groups such as the expert router, with the 114 B expert
bank frozen in its 4-bit NVFP4 form. Video, stills and audio are all supported; a clip
with sound trains picture and sound together.

## Files

| File | Purpose |
|---|---|
| `train_precompute.py` | clips / stills + captions -> cached latents and text features |
| `train.py` | the trainer |
| `train_loss_probe.py` | forward-only check that cached samples give a sane loss |
| `train_probe.py` | time and memory of a training step on random inputs of a given size |
| `../tools/adapter_to_comfy_lora.py` | trained adapter -> LoRA for ComfyUI's stock loader |
| `../vendor/magi2/inference/train/` | expert-kernel backward, LoRA, loss |

The scripts import the MAGI-2 code from `../vendor/magi2` (or `MAGI2_FORK_DIR`).

## What you need

- One 96 GB GPU. Peak use depends on the longest sample (see the table below).
- The NVFP4 checkpoint in its **shard** form (`preview/` on the Hugging Face repo, 67 GiB), not
  the single ComfyUI file, plus the original release's `text_encoder/`, `vae/` and
  `stable-audio-open-1.0/`.
- `ffmpeg` / `ffprobe` and `scipy` for data preparation.
- Tested inside the release's Docker image (`sandai/magi-2-preview`) with this code mounted.
  Running in a plain Python environment has not been tested.

Environment for every command below:

```
MAGI2_CKPT_ROOT=<release dir>   MAGI2_NVFP4_CKPT=<nvfp4 shard dir>
MAGI2_NO_DIST=1  MAGI2_NO_MAGI_COMPILER=1  MAGI2_MOE_FP8=1  MAGI2_MOE_QUANT=nvfp4
MAGI2_MOE_AUTOGRAD=1  MAGI2_LAYER_CHECKPOINT=1
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

## 1. Prepare data

A directory of media with a caption beside each file (`name.mp4` + `name.txt`, or
`name.json` for MAGI-2's structured captions). Videos up to 10 s, with or without sound;
stills (`.png/.jpg/.webp`) teach appearance only.

```
python training/train_precompute.py --data <clips dir>  --out cache/vid    --size 448x256
python training/train_precompute.py --data <stills dir> --out cache/stills --size 896x512
```

Text features are the slow part (15 to 40 s per caption on the CPU); `--text-device cuda`
puts the 52 GB text encoder on the GPU.

## 2. Train

```
python training/train.py --cache cache/vid cache/stills --out runs/my_lora \
    --schedule video:2500,image:1000,video:2500 \
    --lora attn+mlp --rank 16 --lr 1e-4 --save-every 250 --last-every 50
```

- `--schedule kind:steps,...` with kinds `video` (clips), `image` (stills), `mixed`.
- `--lora attn | attn+mlp | all | none`, `--rank`.
- `--native router,norms,sinks,mhc,adapters` trains those groups in place
  (`--native-lr`, e.g. `1e-5`). The router decides which experts each token uses.
- `--resume runs/my_lora/adapter_last.safetensors` continues, with optimizer state.
- On a card that also drives a display: `--gpu-mem-fraction 0.86 --max-latent-frames 16`
  caps the process and skips clips longer than 16 latent frames (5 s) so it stays under the cap.

Measured (rank 16 on all attention and MLP linears, 50.1 M trainable parameters):

| Sample | Forward + backward | Peak allocated |
|---|---|---|
| still, 896x512 | 0.55 + 2.7 s | 75.1 GiB |
| 2.5 s clip, 448x256 | 0.72 + 4.0 s | 77.3 GiB |
| 4.2 s clip, 448x256 | 1.01 + 5.8 s | 80.1 GiB |
| 10 s clip, 448x256 | 2.06 + 11.5 s | 88.9 GiB |

A 10 s clip at 896x512 does not fit. On a set of 21 clips (1 to 10 s) and 183 stills, a
step averaged 5.5 s on clips and 3.2 s on stills, so 6,000 steps take about 9 hours.

## First result

One dataset: 21 clips with sound (1 to 10 s, 448x256) and 183 stills (896x512) of one
character, plain-text captions with a trigger word. Schedule `video:2500,image:1000,video:2500`,
rank 16 on attention + MLP, lr 1e-4, stopped at step 5,750 (about 9 hours on one card).

- The dataset's owner judged the character learned, at step 3,000 and at step 5,750.
- The same adapter rendered through ComfyUI's stock LoRA loader (converted, strength 1.0) and
  un-merged through MAGI-2's pipeline: both judged working.
- A second run that also trained the router (`--native router --native-lr 1e-5`, clips
  limited to 5 s) gave blurry video. LoRA only is the recommended starting point.
- Smoothed loss, start -> end: video 0.52 -> about 0.2, audio 0.98 -> about 0.5.

## 3. Use the adapter

- **ComfyUI:** convert, then load with the stock **LoraLoaderModelOnly** node:

  ```
  python tools/adapter_to_comfy_lora.py runs/my_lora/adapter_006000.safetensors my_lora.safetensors \
      --base <magi2_preview_nvfp4_comfy.safetensors>
  ```

  `--base` is only needed when native groups were trained (they become full differences).
  ComfyUI merges the LoRA into bf16 weights, which rounds small deltas (see the main README).
  A trained adapter has been run both this way and un-merged (below); both worked.
- **MAGI-2's own pipeline:** `MAGI2_ADAPTER=<adapter file>` attaches it un-merged.

## How it works, briefly

- The inference expert kernel is forward-only. `moe_autograd.py` adds a hand-written backward
  for the kernel's input and the routing probabilities (checked against autograd: 1.4e-16 on
  the CPU in float64). Expert weights are constants; each layer's bank is dequantised again in
  the backward pass instead of being kept.
- Each of the 40 layers is gradient-checkpointed, so only one layer's activations are alive.
- Objective: flow matching on video and audio latents with one shared noise level; captions
  are dropped with probability `--dropout` so guidance keeps working.

## Limits

- One GPU, batch size 1.
- The expert weights themselves do not train.
- Whether the voice is learned as well as the look has not been established.
- Training the router made results worse in the one attempt so far.
