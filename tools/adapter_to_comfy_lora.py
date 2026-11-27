#!/usr/bin/env python3
"""Convert a MAGI-2 adapter (from training/train.py) into a LoRA file that
ComfyUI's stock LoRA loader applies to the MAGI-2 model.

    python tools/adapter_to_comfy_lora.py adapter.safetensors out_lora.safetensors [--base magi2_preview_nvfp4_comfy.safetensors]

* Low-rank adapters: the trainer keeps one pair per modality group, ``A [G, r, in]`` and
  ``B [G, out, r]``, on a weight stored as ``[G * out, in]``. ComfyUI wants one pair per
  weight, so the groups are stacked: ``lora_A [G*r, in]``, ``lora_B [G*out, G*r]`` (block
  diagonal), ``alpha = alpha * G``. The resulting weight delta is identical.
* Native parameter groups (router, norms, ...): written as full differences
  ``<key>.diff = trained - base``, which needs ``--base`` to read the original values.
  Without ``--base`` they are skipped and listed.

Needs torch and safetensors only.
"""
import argparse
import json
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def stack_groups(A, B):
    G, r, in_f = A.shape
    out_f = B.shape[1]
    down = A.reshape(G * r, in_f).contiguous()
    up = torch.zeros(G * out_f, G * r, dtype=B.dtype)
    for g in range(G):
        up[g * out_f:(g + 1) * out_f, g * r:(g + 1) * r] = B[g]
    return down, up


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("adapter")
    ap.add_argument("out")
    ap.add_argument("--base", default="", help="the ComfyUI model file, to convert native parameter groups")
    args = ap.parse_args()
    with safe_open(args.adapter, framework="pt") as f:
        meta = f.metadata() or {}
        tensors = {k: f.get_tensor(k) for k in f.keys()}
    rank, alpha = int(json.loads(meta["rank"])), float(json.loads(meta["alpha"]))
    out, skipped = {}, []
    names = sorted({k[len("lora."):-2] for k in tensors if k.startswith("lora.")})
    for name in names:
        A, B = tensors[f"lora.{name}.A"].float(), tensors[f"lora.{name}.B"].float()
        down, up = stack_groups(A, B)
        key = f"diffusion_model.{name}"
        out[f"{key}.lora_A.weight"] = down
        out[f"{key}.lora_B.weight"] = up
        out[f"{key}.alpha"] = torch.tensor(alpha * A.shape[0])
    native = sorted(k for k in tensors if k.startswith("native."))
    if native and args.base:
        with safe_open(args.base, framework="pt") as base:
            for k in native:
                name = k[len("native."):]
                out[f"diffusion_model.{name}.diff"] = tensors[k].float() - base.get_tensor(name).float()
    else:
        skipped = native
    save_file(out, args.out, metadata={"format": "pt", "magi2.source_adapter_step": str(meta.get("step", "")), "magi2.rank": str(rank)})
    print(f"wrote {args.out}: {len(names)} low-rank adapters, {0 if skipped else len(native)} parameter differences")
    if skipped:
        print(f"skipped {len(skipped)} native parameters (pass --base to include them), e.g. {skipped[0][len('native.'):]}")


if __name__ == "__main__":
    main()
