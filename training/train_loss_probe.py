#!/usr/bin/env python3
# Copyright (c) 2026 SandAI. All Rights Reserved.
# Apache-2.0.
"""Forward-only check of the training data conventions, before any training.

    python training/train_loss_probe.py --cache <dir with *.pt from train_precompute.py>

For every cached sample and a few noise levels it measures the flow-matching loss of
the UNMODIFIED model with (a) the sample's own caption, (b) another sample's caption,
(c) no caption. If the latents are in the space the model expects and the caption
features are right, (a) is clearly the lowest and well below the loss of predicting
zero velocity (which is mean((noise - x0)^2), about 1 + var(x0)).
"""

import argparse
import glob
import os
import sys

import torch

REPO = os.environ.get("MAGI2_FORK_DIR") or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vendor", "magi2")
sys.path.insert(0, REPO)
DEFAULT_CONFIG = os.path.join(REPO, "configs", "local", "1gpu_preview.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--sigmas", default="0.3,0.6,0.9")
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from inference.infra.checkpoint.load_checkpoint import load_magi2_model
    from inference.pipeline.entry import build_config
    from inference.pipeline.preview_data_proxy import Magi2DataProxy
    from inference.train.batch import predict_velocity

    files = sorted(glob.glob(os.path.join(args.cache, "*.pt")))[: args.limit]
    if len(files) < 2:
        sys.exit("need at least two cached samples")
    samples = [torch.load(f, map_location="cpu") for f in files]
    torch.cuda.set_device(0)
    dev = torch.device("cuda:0")
    config = build_config(args.config, seed=0)
    model = load_magi2_model(config)
    model.requires_grad_(False)
    proxy = Magi2DataProxy(config.evaluation_config.data_proxy_config)
    torch.backends.cuda.matmul.allow_tf32 = True
    sigmas = [float(s) for s in args.sigmas.split(",")]
    empty = torch.zeros(1, 0, 5120, dtype=torch.bfloat16, device=dev)

    totals = {}
    for i, s in enumerate(samples):
        name = os.path.basename(files[i])[:-3]
        video0 = s["video"].unsqueeze(0).to(dev)
        audio0 = s["audio"].unsqueeze(0).to(dev)
        own = s["text"].unsqueeze(0).to(dev)
        other = samples[(i + 1) % len(samples)]["text"].unsqueeze(0).to(dev)
        print(f"[probe] {name}: video latent mean {float(video0.mean()):+.3f} std {float(video0.std()):.3f} | "
              f"audio latent mean {float(audio0.mean()):+.3f} std {float(audio0.std()):.3f} | text tokens {own.shape[1]}", flush=True)
        for sigma in sigmas:
            g = torch.Generator(device="cpu").manual_seed(args.seed + i)
            nv = torch.randn(video0.shape, generator=g).to(dev)
            na = torch.randn(audio0.shape, generator=g).to(dev)
            st = torch.tensor(sigma, device=dev)
            vt, at = (1 - st) * video0 + st * nv, (1 - st) * audio0 + st * na
            row = {}
            with torch.no_grad():
                for label, ctx in (("own", own), ("other", other), ("none", empty)):
                    pv, pa = predict_velocity(model, proxy, vt, at, ctx, st)
                    row[label] = (float(torch.mean((pv - (nv - video0)) ** 2)), float(torch.mean((pa - (na - audio0)) ** 2)))
            # Ablations with the sample's own caption: the audio reversed in time, and
            # rolled by half its length. If audio and video are aligned the way the model
            # expects, both should be clearly worse than "own".
            if audio0.shape[1] > 1:
                with torch.no_grad():
                    for label, a_alt in (("a-rev", audio0.flip(1)), ("a-roll", audio0.roll(audio0.shape[1] // 2, 1))):
                        at_alt = (1 - st) * a_alt + st * na
                        pv, pa = predict_velocity(model, proxy, vt, at_alt, own, st)
                        row[label] = (float(torch.mean((pv - (nv - video0)) ** 2)), float(torch.mean((pa - (na - a_alt)) ** 2)))
            zero = (float(torch.mean((nv - video0) ** 2)), float(torch.mean((na - audio0) ** 2)))
            print(f"[probe]   sigma {sigma:.2f}  video mse: own {row['own'][0]:.4f} other {row['other'][0]:.4f} none {row['none'][0]:.4f} zero-pred {zero[0]:.4f}"
                  f"  | audio mse: own {row['own'][1]:.4f} other {row['other'][1]:.4f} none {row['none'][1]:.4f} zero-pred {zero[1]:.4f}", flush=True)
            for label in row:
                t = totals.setdefault(label, [0.0, 0.0, 0])
                t[0] += row[label][0]; t[1] += row[label][1]; t[2] += 1
    for label, (v, a, n) in totals.items():
        print(f"[probe] mean over {n}: caption={label:5s} video mse {v / n:.4f} audio mse {a / n:.4f}", flush=True)


if __name__ == "__main__":
    main()
