#!/usr/bin/env python3
# Copyright (c) 2026 SandAI. All Rights Reserved.
# Apache-2.0.
"""Does a training step run, and what does it cost?  One process, one GPU, real weights.

    MAGI2_MOE_AUTOGRAD=1 MAGI2_LAYER_CHECKPOINT=1 MAGI2_NO_DIST=1 MAGI2_NO_MAGI_COMPILER=1 \
    MAGI2_MOE_QUANT=nvfp4 python training/train_probe.py --frames 32 --height 16 --width 28

Random latents and text features of the given size (latent units), a LoRA and/or
native parameter groups, then: a no-gradient forward (reference time and memory)
and ``--steps`` forward+backward passes. Prints peak memory, time and gradient norms.
"""

import argparse
import os
import time

import torch

import sys

REPO = os.environ.get("MAGI2_FORK_DIR") or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vendor", "magi2")
sys.path.insert(0, REPO)
DEFAULT_CONFIG = os.path.join(REPO, "configs", "local", "1gpu_preview.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--frames", type=int, default=32, help="video latent frames (32 = 10 s)")
    ap.add_argument("--height", type=int, default=16, help="latent rows (16 = 256 px)")
    ap.add_argument("--width", type=int, default=28, help="latent columns (28 = 448 px)")
    ap.add_argument("--audio", type=int, default=250, help="audio latents (250 = 10 s)")
    ap.add_argument("--text", type=int, default=456, help="text tokens")
    ap.add_argument("--lora", default="attn+mlp", help="LoRA preset, or 'none'")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--layers", default="", help="first:last layer range for the LoRA, e.g. 38:40 (default all)")
    ap.add_argument("--native", default="", help="comma list of native groups: router,sinks,norms,mhc,adapters")
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--sigma", type=float, default=0.7)
    args = ap.parse_args()

    from inference.infra.checkpoint.load_checkpoint import load_magi2_model
    from inference.pipeline.entry import build_config
    from inference.pipeline.preview_data_proxy import Magi2DataProxy
    from inference.train import lora
    from inference.train.batch import flow_matching_loss, predict_velocity

    torch.cuda.set_device(0)
    dev = torch.device("cuda:0")
    config = build_config(args.config, seed=0)
    t0 = time.time()
    model = load_magi2_model(config)
    model.requires_grad_(False)
    proxy = Magi2DataProxy(config.evaluation_config.data_proxy_config)
    gib = lambda b: b / 2**30
    print(f"[probe] loaded in {time.time() - t0:.1f}s, {gib(torch.cuda.memory_allocated()):.2f} GiB allocated", flush=True)

    layers = None
    if args.layers:
        a, b = args.layers.split(":")
        layers = range(int(a), int(b))
    adapters = {} if args.lora == "none" else lora.add_lora(model, args.lora, rank=args.rank, layers=layers)
    native = lora.enable_native(model, [g for g in args.native.split(",") if g]) if args.native else {}
    params = [p for a_ in adapters.values() for p in (a_.lora_A, a_.lora_B)] + list(native.values())
    n_train = sum(p.numel() for p in params)
    print(f"[probe] LoRA '{args.lora}' rank {args.rank}: {len(adapters)} modules; native {sorted(native)[:2]}... {len(native)} tensors; "
          f"{n_train / 1e6:.2f} M trainable parameters; {gib(torch.cuda.memory_allocated()):.2f} GiB allocated", flush=True)

    g = torch.Generator(device="cpu").manual_seed(0)
    video0 = torch.randn(1, 48, args.frames, args.height, args.width, generator=g).to(dev)
    audio0 = torch.randn(1, args.audio, 64, generator=g).to(dev)
    context = (torch.randn(1, args.text, 5120, generator=g) * 2).to(dev, torch.bfloat16)
    sigma = torch.tensor(args.sigma)
    tokens = args.frames * args.height * args.width + args.audio + args.text
    print(f"[probe] tokens: video {args.frames * args.height * args.width} + audio {args.audio} + text {args.text} = {tokens}", flush=True)

    torch.backends.cuda.matmul.allow_tf32 = True
    for i in range(2):
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t0 = time.time()
        with torch.no_grad():
            v, a = predict_velocity(model, proxy, video0, audio0, context, sigma)
        torch.cuda.synchronize()
        print(f"[probe] no-grad forward {i}: {time.time() - t0:.2f}s peak {gib(torch.cuda.max_memory_allocated()):.2f} GiB "
              f"finite={bool(torch.isfinite(v).all() and torch.isfinite(a).all())}", flush=True)
    base = torch.cuda.memory_allocated()
    for i in range(args.steps):
        for p in params:
            p.grad = None
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t0 = time.time()
        loss, vm, am = flow_matching_loss(model, proxy, video0, audio0, context, sigma)
        torch.cuda.synchronize(); t1 = time.time(); peak_fwd = torch.cuda.max_memory_allocated()
        loss.backward()
        torch.cuda.synchronize(); t2 = time.time(); peak = torch.cuda.max_memory_allocated()
        gn = torch.sqrt(sum((p.grad.float() ** 2).sum() for p in params if p.grad is not None))
        missing = sum(1 for p in params if p.grad is None)
        nz = sum(1 for p in params if p.grad is not None and float(p.grad.abs().max()) > 0)
        print(f"[probe] step {i}: loss {float(loss):.4f} (video {float(vm):.4f} audio {float(am):.4f}) forward {t1 - t0:.2f}s backward {t2 - t1:.2f}s "
              f"peak fwd {gib(peak_fwd):.2f} GiB peak {gib(peak):.2f} GiB (+{gib(peak - base):.2f} over idle) "
              f"grad norm {float(gn):.4e} finite={bool(torch.isfinite(gn))} tensors with grad {len(params) - missing}/{len(params)} nonzero {nz}", flush=True)
    print(f"[probe] reserved {gib(torch.cuda.max_memory_reserved()):.2f} GiB; MB per token over idle: {(peak - base) / tokens / 1e6:.3f}", flush=True)


if __name__ == "__main__":
    main()
