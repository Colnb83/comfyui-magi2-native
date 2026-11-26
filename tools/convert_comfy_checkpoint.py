#!/usr/bin/env python3
# Copyright (c) 2026 SandAI. All Rights Reserved.
# Apache-2.0.
"""NVFP4 preview checkpoint (sharded) -> ONE safetensors file for the ComfyUI model class.

    MAGI2_MOE_QUANT=nvfp4 python scripts/convert_comfy_checkpoint.py <nvfp4_ckpt_dir> <out.safetensors>

The output holds exactly the denoiser's ``state_dict`` with ``MAGI2_MOE_QUANT=nvfp4``:
every key, shape and dtype is what ``Transformer`` constructs, so ComfyUI can load it
with ``load_state_dict(strict=True, assign=True)`` straight from a memory map, with no
conversion at load time.

* Non-expert tensors: copied byte for byte.
* Expert tensors ``W_gate / W_up / W_down``: the checkpoint's ComfyUI bank layout
  (``.weight`` / ``.weight_scale`` / ``.weight_scale_2``) is rearranged losslessly into
  the kernel's device layout: ``<K>`` (uint8 codes), ``<K>_bscale`` (uint8 E4M3 block
  scales), ``<K>_scale`` (fp32 per expert). No re-quantisation.
* ``router.expert_bias`` is filled from ``router.expert_bias_ema``, as the loader does
  (``MAGI2_ROUTER_BIAS_SOURCE=main`` keeps the raw bias).

Runs on the GPU when one is visible, else on the CPU. Written to ``<out>.partial`` and
renamed when complete.
"""

import argparse
import json
import os
import struct
import sys
import time

os.environ.setdefault("MAGI2_MOE_QUANT", "nvfp4")
os.environ.setdefault("MAGI2_MOE_FP8", "1")
os.environ.setdefault("MAGI2_NO_DIST", "1")
os.environ.setdefault("MAGI2_CKPT_ROOT", "/unused")

import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

ST_DTYPE = {
    torch.float32: "F32", torch.bfloat16: "BF16", torch.float16: "F16", torch.uint8: "U8",
    torch.int64: "I64", torch.int32: "I32", torch.bool: "BOOL", torch.float64: "F64",
}
EXPERTS = ("W_gate", "W_up", "W_down")
COMFY_STATE_BUFFERS = tuple(f"{n}{s}" for n in EXPERTS for s in ("_bscale", "_scale"))


def comfy_state_dict_spec(model):
    """Key -> (shape, dtype) of the denoiser including the expert scale buffers,
    which the model registers as non-persistent."""
    spec = {k: (tuple(v.shape), v.dtype) for k, v in model.state_dict().items()}
    for name, buf in model.named_buffers():
        if name.rsplit(".", 1)[-1] in COMFY_STATE_BUFFERS:
            spec[name] = (tuple(buf.shape), buf.dtype)
    return spec


def raw_bytes(t: torch.Tensor) -> bytes:
    t = t.detach().cpu().contiguous()
    if t.ndim == 0:
        t = t.reshape(1)
    return t.view(torch.uint8).numpy().tobytes()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint_dir", help="directory with the NVFP4 shards (holding or being preview/)")
    ap.add_argument("out")
    ap.add_argument("--config", default=os.path.join(REPO, "configs/local/1gpu_preview.json"))
    args = ap.parse_args()
    if os.path.exists(args.out):
        sys.exit(f"{args.out} exists; refusing to overwrite")
    ckpt = args.checkpoint_dir
    if os.path.isdir(os.path.join(ckpt, "preview")):
        ckpt = os.path.join(ckpt, "preview")

    from inference.common.magi2_config import load_config
    from inference.flash_mh_moe.nvfp4 import device_layout_from_file
    from inference.model.magi2_preview import Transformer

    with torch.device("meta"):
        model = Transformer(load_config(args.config).arch_config, ep_size=1)
    spec = comfy_state_dict_spec(model)
    del model

    file_map = json.load(open(os.path.join(ckpt, "model.safetensors.index.json")))["weight_map"]
    use_ema = (os.environ.get("MAGI2_ROUTER_BIAS_SOURCE") or "ema").strip().lower() != "main"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    header, offset = {}, 0
    for key, (shape, dtype) in spec.items():
        n = dtype.itemsize
        for s in shape:
            n *= s
        header[key] = {"dtype": ST_DTYPE[dtype], "shape": list(shape), "data_offsets": [offset, offset + n]}
        offset += n
    meta = {
        "format": "pt",
        "modelspec.architecture": "magi-2-preview",
        "magi2.layout": "comfy-state-dict-v1",
        "magi2.experts": "nvfp4-device-layout",
        "magi2.router_bias": "ema" if use_ema else "main",
        "magi2.source": os.path.basename(os.path.dirname(os.path.abspath(ckpt))) + "/" + os.path.basename(ckpt),
    }
    blob = json.dumps({"__metadata__": meta, **header}, separators=(",", ":")).encode("utf-8")
    blob += b" " * (-len(blob) % 8)

    handles = {}

    def get(key):
        path = os.path.join(ckpt, file_map[key])
        if path not in handles:
            handles[path] = safe_open(path, framework="pt")
        return handles[path].get_tensor(key)

    converted = {}  # expert key -> (codes, bscale, scale), produced once, consumed by its three keys

    def expert_parts(base):
        if base not in converted:
            weight, wscale, wscale2 = get(base + ".weight"), get(base + ".weight_scale"), get(base + ".weight_scale_2")
            codes, bscales = [], []
            for e0 in range(0, weight.shape[0], 256):
                c, b = device_layout_from_file(weight[e0:e0 + 256].to(device), wscale[e0:e0 + 256].to(device))
                codes.append(c.cpu())
                bscales.append(b.cpu())
            converted[base] = [torch.cat(codes), torch.cat(bscales), wscale2.to(torch.float32).reshape(-1)]
        return converted[base]

    tmp = args.out + ".partial"
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    t0, last, n_expert = time.time(), 0.0, 0
    with open(tmp, "wb") as out:
        out.write(struct.pack("<Q", len(blob)))
        out.write(blob)
        for key, (shape, dtype) in spec.items():
            leaf = key.rsplit(".", 1)[-1]
            if leaf in EXPERTS or leaf in COMFY_STATE_BUFFERS:
                base = key
                for suffix, idx in (("_bscale", 1), ("_scale", 2)):
                    if leaf.endswith(suffix):
                        base, which = key[: -len(suffix)], idx
                        break
                else:
                    which = 0
                parts = expert_parts(base)
                tensor = parts[which]
                parts[which] = None
                if all(p is None for p in parts):
                    del converted[base]
                n_expert += 1
            elif use_ema and key.endswith("moe_mlp.router.expert_bias"):
                tensor = get(key + "_ema")
            else:
                tensor = get(key)
            if tuple(tensor.shape) != shape or tensor.dtype != dtype:
                sys.exit(f"{key}: checkpoint gives {tuple(tensor.shape)} {tensor.dtype}, model expects {shape} {dtype}")
            data = raw_bytes(tensor)
            begin, end = header[key]["data_offsets"]
            if len(data) != end - begin or out.tell() != 8 + len(blob) + begin:
                sys.exit(f"{key}: size/offset mismatch")
            out.write(data)
            del tensor, data
            if time.time() - last > 15:
                last = time.time()
                print(f"[convert] {out.tell() / 2**30:.1f} / {offset / 2**30:.1f} GiB", flush=True)
        out.flush()
        os.fsync(out.fileno())
    if converted:
        sys.exit(f"internal error: unconsumed expert tensors {list(converted)[:3]}")
    os.rename(tmp, args.out)
    print(f"[convert] wrote {args.out}: {len(spec)} tensors ({n_expert} expert tensors rearranged on {device}), "
          f"{offset / 2**30:.2f} GiB, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
