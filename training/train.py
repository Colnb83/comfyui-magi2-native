#!/usr/bin/env python3
# Copyright (c) 2026 SandAI. All Rights Reserved.
# Apache-2.0.
"""Train an adapter for MAGI-2-preview on one GPU, experts frozen in NVFP4.

    MAGI2_MOE_AUTOGRAD=1 MAGI2_LAYER_CHECKPOINT=1 MAGI2_NO_DIST=1 MAGI2_NO_MAGI_COMPILER=1 \
    MAGI2_MOE_QUANT=nvfp4 python training/train.py --cache <samples dir> --out <run dir>

Samples come from training/train_precompute.py. What is trained:

* ``--lora attn | attn+mlp | all | none`` at ``--rank``: low-rank adapters on the
  non-expert linears (style, motion, character, voice);
* ``--native router,norms,sinks,mhc,adapters``: native parameter groups trained in
  place; ``router`` changes which experts each token uses, which is the lever for
  teaching the model something it does not already represent.

Objective: flow matching on video and audio latents with one shared noise level,
sigma = shift * u / (1 + (shift - 1) * u), u uniform or logit-normal; the caption is
dropped (empty text) with probability ``--dropout`` so guidance keeps working.
Clips without sound train the video only.

The adapter is written to ``<out>/adapter_<step>.safetensors`` and
``<out>/adapter_last.safetensors``; ``--resume`` continues from one.
"""

import argparse
import glob
import json
import math
import os
import random
import sys
import time

import torch

REPO = os.environ.get("MAGI2_FORK_DIR") or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vendor", "magi2")
sys.path.insert(0, REPO)
DEFAULT_CONFIG = os.path.join(REPO, "configs", "local", "1gpu_preview.json")


def sample_sigma(kind: str, shift: float) -> float:
    if kind == "logit_normal":
        u = 1.0 / (1.0 + math.exp(-random.gauss(0.0, 1.0)))
    else:
        u = random.random()
    u = min(max(u, 1e-3), 0.999)
    return shift * u / (1.0 + (shift - 1.0) * u)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, nargs="+", help="one or more directories of samples from train_precompute.py")
    ap.add_argument("--schedule", default="", help="phases as kind:steps, e.g. video:600,image:200,video:600 "
                    "(kinds: video = clips, image = stills, mixed = both). Overrides --steps.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--size", default="", help="only samples of this WxH (default: all sizes in the cache)")
    ap.add_argument("--lora", default="attn+mlp")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=None, help="default: equal to the rank")
    ap.add_argument("--layers", default="", help="first:last layer range for the LoRA (default all)")
    ap.add_argument("--native", default="")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--native-lr", type=float, default=1e-5)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--steps", type=int, default=1000, help="optimizer steps")
    ap.add_argument("--accum", type=int, default=1, help="samples per optimizer step")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--sigma", default="uniform", choices=["uniform", "logit_normal"])
    ap.add_argument("--shift", type=float, default=7.0)
    ap.add_argument("--audio-weight", type=float, default=1.0)
    ap.add_argument("--save-every", type=int, default=250, help="numbered adapter checkpoints")
    ap.add_argument("--last-every", type=int, default=50, help="adapter_last + optimizer state, for resuming after an interruption")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--resume", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-latent-frames", type=int, default=0, help="skip clips longer than this many latent frames (32 = 10 s); 0 = no limit")
    ap.add_argument("--gpu-mem-fraction", type=float, default=0.0,
                    help="hard cap on this process's GPU memory as a fraction of the card (e.g. 0.88 on a card that also drives a display); 0 = no cap")
    args = ap.parse_args()

    from inference.infra.checkpoint.load_checkpoint import load_magi2_model
    from inference.pipeline.entry import build_config
    from inference.pipeline.preview_data_proxy import Magi2DataProxy
    from inference.train import lora
    from inference.train.batch import flow_matching_loss

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    files = sorted(f for c in args.cache for f in glob.glob(os.path.join(c, f"*_{args.size}.pt" if args.size else "*.pt")))
    if not files:
        sys.exit(f"no samples in {args.cache}" + (f" of size {args.size}" if args.size else ""))
    samples = [torch.load(f, map_location="cpu") for f in files]
    if args.max_latent_frames > 0:
        kept = [s for s in samples if s["video"].shape[1] <= args.max_latent_frames]
        print(f"[train] --max-latent-frames {args.max_latent_frames}: {len(samples) - len(kept)} longer clip(s) skipped", flush=True)
        samples = kept
    pools = {"mixed": samples,
             "image": [s for s in samples if s["video"].shape[1] == 1],
             "video": [s for s in samples if s["video"].shape[1] > 1]}
    print(f"[train] {len(samples)} samples ({len(pools['video'])} clips, {len(pools['image'])} stills): "
          + ", ".join(sorted({'%dx%d' % tuple(s['size']) for s in samples})), flush=True)
    # Phases: which pool each optimizer step draws from.
    phases = []
    if args.schedule:
        for part in args.schedule.split(","):
            kind, _, n = part.strip().partition(":")
            if kind not in pools or not n.isdigit():
                sys.exit(f"bad --schedule entry {part!r}: use kind:steps with kind in video, image, mixed")
            if not pools[kind]:
                sys.exit(f"--schedule asks for '{kind}' samples but the cache has none")
            phases.append((kind, int(n)))
        args.steps = sum(n for _, n in phases)
    else:
        phases = [("mixed", args.steps)]
    bounds, total = [], 0
    for kind, n in phases:
        total += n
        bounds.append((total, kind))
    print("[train] schedule: " + " -> ".join(f"{k} x{n}" for k, n in phases) + f" = {args.steps} steps", flush=True)

    def pool_for(step):
        return pools[next(kind for end, kind in bounds if step < end)]

    torch.cuda.set_device(0)
    dev = torch.device("cuda:0")
    if args.gpu_mem_fraction > 0:
        # PyTorch then fails with its own out-of-memory error instead of taking memory other
        # users of the card (a desktop) need.
        torch.cuda.set_per_process_memory_fraction(args.gpu_mem_fraction, 0)
        print(f"[train] GPU memory capped at {args.gpu_mem_fraction:.2f} of the card "
              f"({args.gpu_mem_fraction * torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GiB)", flush=True)
    config = build_config(args.config, seed=args.seed)
    model = load_magi2_model(config)
    model.requires_grad_(False)
    proxy = Magi2DataProxy(config.evaluation_config.data_proxy_config)
    torch.backends.cuda.matmul.allow_tf32 = True

    start_step = 0
    native_groups = [g for g in args.native.split(",") if g]
    if args.resume:
        meta = lora.load_adapter(model, args.resume)
        from torch.nn.utils import parametrize

        adapters = {n: m.parametrizations.weight[0] for n, m in model.named_modules()
                    if parametrize.is_parametrized(m, "weight")}
        native = lora.enable_native(model, json.loads(meta.get("native", "[]"))) if meta.get("native") else {}
        start_step = int(json.loads(meta.get("step", "0")))
        print(f"[train] resumed {args.resume} at step {start_step}", flush=True)
    else:
        layers = None
        if args.layers:
            a, b = args.layers.split(":")
            layers = range(int(a), int(b))
        adapters = {} if args.lora == "none" else lora.add_lora(model, args.lora, rank=args.rank, alpha=args.alpha, layers=layers)
        native = lora.enable_native(model, native_groups) if native_groups else {}
    lora_params = [p for a_ in adapters.values() for p in (a_.lora_A, a_.lora_B)]
    for p in lora_params:
        p.requires_grad_(True)
    native_params = list(native.values())
    if not lora_params and not native_params:
        sys.exit("nothing to train: --lora none and no --native groups")
    groups = []
    if lora_params:
        groups.append({"params": lora_params, "lr": args.lr})
    if native_params:
        groups.append({"params": native_params, "lr": args.native_lr})
    optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.99), weight_decay=args.weight_decay)
    base_lrs = [g["lr"] for g in groups]
    all_params = lora_params + native_params
    print(f"[train] LoRA {args.lora} rank {args.rank} on {len(adapters)} modules ({sum(p.numel() for p in lora_params) / 1e6:.2f} M), "
          f"native {native_groups} ({sum(p.numel() for p in native_params) / 1e6:.2f} M); "
          f"{torch.cuda.memory_allocated() / 2**30:.2f} GiB allocated", flush=True)

    meta = {"rank": args.rank, "alpha": args.rank if args.alpha is None else args.alpha, "lora": args.lora, "native": native_groups,
            "shift": args.shift, "sigma": args.sigma, "dropout": args.dropout, "samples": len(samples), "schedule": args.schedule or "mixed",
            "base": "magi-2-preview nvfp4"}

    state_path = os.path.join(args.out, "train_state.pt")
    if args.resume and os.path.isfile(state_path):
        # Optimizer moments and RNG, so a resumed run continues instead of restarting cold.
        state = torch.load(state_path, map_location="cpu")
        if int(state.get("step", -1)) == start_step:
            optimizer.load_state_dict(state["optimizer"])
            random.setstate(state["python_rng"])
            torch.set_rng_state(state["torch_rng"])
            print(f"[train] optimizer state restored from {state_path}", flush=True)
        else:
            print(f"[train] {state_path} is for step {state.get('step')}, not {start_step}: optimizer starts fresh", flush=True)

    def save(step: int, name: str):
        path = os.path.join(args.out, name)
        lora.save_adapter(path + ".tmp", adapters, native, {**meta, "step": step})
        os.replace(path + ".tmp", path)
        if name == "adapter_last.safetensors":
            torch.save({"step": step, "optimizer": optimizer.state_dict(), "python_rng": random.getstate(),
                        "torch_rng": torch.get_rng_state()}, state_path + ".tmp")
            os.replace(state_path + ".tmp", state_path)

    empty = torch.zeros(1, 0, 5120, dtype=torch.bfloat16, device=dev)
    ema_v = ema_a = None
    t_last, n_last = time.time(), 0
    torch.cuda.reset_peak_memory_stats()
    for step in range(start_step, args.steps):
        scale = min(1.0, (step + 1) / max(1, args.warmup))
        for g, lr in zip(optimizer.param_groups, base_lrs):
            g["lr"] = lr * scale
        optimizer.zero_grad(set_to_none=True)
        step_v = step_a = 0.0
        n_audio = 0
        for _ in range(args.accum):
            s = random.choice(pool_for(step))
            video0 = s["video"].unsqueeze(0).to(dev)
            audio0 = s["audio"].unsqueeze(0).to(dev)
            context = empty if random.random() < args.dropout else s["text"].unsqueeze(0).to(dev)
            sigma = torch.tensor(sample_sigma(args.sigma, args.shift))
            has_audio = bool(s.get("has_audio", True)) and s["audio"].shape[0] > 0
            loss, vm, am = flow_matching_loss(model, proxy, video0, audio0, context, sigma,
                                              audio_weight=args.audio_weight if has_audio else 0.0)
            (loss / args.accum).backward()
            step_v += float(vm) / args.accum
            if has_audio:
                step_a += float(am)
                n_audio += 1
        grad_norm = torch.nn.utils.clip_grad_norm_(all_params, args.clip if args.clip > 0 else float("inf"))
        if not torch.isfinite(grad_norm):
            print(f"[train] step {step + 1}: non-finite gradient, step skipped", flush=True)
            continue
        optimizer.step()
        ema_v = step_v if ema_v is None else 0.95 * ema_v + 0.05 * step_v
        if n_audio:
            a = step_a / n_audio
            ema_a = a if ema_a is None else 0.95 * ema_a + 0.05 * a
        n_last += 1
        if (step + 1) % args.log_every == 0 or step == start_step:
            dt = (time.time() - t_last) / n_last
            t_last, n_last = time.time(), 0
            kind_now = next(kind for end, kind in bounds if step < end)
            print(f"[train] step {step + 1}/{args.steps} [{kind_now}] video mse {step_v:.4f} (ema {ema_v:.4f}) audio mse "
                  f"{(step_a / n_audio) if n_audio else float('nan'):.4f} (ema {ema_a if ema_a is not None else float('nan'):.4f}) "
                  f"grad {float(grad_norm):.3e} lr {optimizer.param_groups[0]['lr']:.2e} {dt:.1f} s/step "
                  f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
        if (step + 1) % args.save_every == 0:
            save(step + 1, f"adapter_{step + 1:06d}.safetensors")
        if (step + 1) % args.last_every == 0 or (step + 1) % args.save_every == 0:
            save(step + 1, "adapter_last.safetensors")
    save(args.steps, "adapter_last.safetensors")
    print(f"[train] finished {args.steps} steps -> {os.path.join(args.out, 'adapter_last.safetensors')}", flush=True)


if __name__ == "__main__":
    main()
