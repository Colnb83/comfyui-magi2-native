#!/usr/bin/env python3
"""Write a pre-quantised NVFP4 copy of the MAGI-2-preview checkpoint.

The 108 MoE expert tensors (``block.layers.<L>.mlp.moe_mlp.W_{gate,up,down}``,
bf16 ``[E, in, out]`` in the source) are stored as ComfyUI / comfy-kitchen
``MoEExperts`` NVFP4 banks; every other tensor is copied unchanged (same key,
same dtype, same bytes).  For an expert tensor with source key ``K``:

  K.weight          uint8    [E, out, in/2]   two E2M1 codes per byte, element 2i of a row
                                               in the HIGH nibble; rows are [out, in], i.e. the
                                               source tensor transposed (permute(0, 2, 1))
  K.weight_scale    f8_e4m3  [E, out, in/16]  block scale per 16 weights along ``in``, each
                                               expert in comfy-kitchen's (cuBLAS) tiled order
  K.weight_scale_2  float32  [E]              per-expert scale, amax / (448 * 6)
  K.comfy_quant     uint8    [n]              UTF-8 JSON {"format": "nvfp4", "num_experts": E}

  value[e, out, in] = E2M1(code) * weight_scale[e][out, in // 16] * weight_scale_2[e]

Output: ``model-000NN-of-00039.safetensors`` (1-2: the non-expert tensors of source shards
1 and 2; 3: router tensors; 4-39: one MoE layer each), ``model.safetensors.index.json`` and
``nvfp4_manifest.json`` (written last; the launcher only uses the directory once it exists).

The fork loads this directory with ``MAGI2_MOE_QUANT=nvfp4`` and ``MAGI2_NVFP4_CKPT`` set
(scripts/run_local_2gpu.sh does both); see ``stream_load_magi2_model_``.

Resumable: existing shards are skipped, so the expert stage can be run a few layers at a
time (``--layers``) to stay inside a GPU time slot:

  python scripts/quantize_nvfp4.py --src /ckpt/preview --dst /ckpt-nvfp4/preview --stage others
  python scripts/quantize_nvfp4.py --src ... --dst ... --stage experts --layers 2,3,4,5,6,7,8,9,10,11,12,13
  python scripts/quantize_nvfp4.py --src ... --dst ... --stage finalize
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import struct
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

EXPERT_NAMES = ("W_gate", "W_up", "W_down")
INDEX_NAME = "model.safetensors.index.json"
MANIFEST_NAME = "nvfp4_manifest.json"
FORMAT_VERSION = 1


def _expert_match(key: str):
    return re.fullmatch(r"block\.layers\.(\d+)\.mlp\.moe_mlp\.(W_gate|W_up|W_down)", key)


def _header(path: str) -> dict:
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        h = json.loads(fh.read(n))
    h.pop("__metadata__", None)
    return h


def _plan(src: str) -> dict:
    wmap = json.load(open(os.path.join(src, INDEX_NAME)))["weight_map"]
    layers = sorted({int(_expert_match(k).group(1)) for k in wmap if _expert_match(k)})
    n_shards = 3 + len(layers)
    name = lambda i: f"model-{i:05d}-of-{n_shards:05d}.safetensors"  # noqa: E731
    src_shards = sorted(set(wmap.values()))
    other_groups: dict[str, list[str]] = {name(1): [], name(2): [], name(3): []}
    for key, shard in wmap.items():
        if _expert_match(key):
            continue
        if ".moe_mlp." in key:
            other_groups[name(3)].append(key)
        else:
            other_groups[name(1) if shard == src_shards[0] else name(2)].append(key)
    return {"wmap": wmap, "layers": layers, "n_shards": n_shards, "other_groups": other_groups,
            "layer_shard": {layer: name(4 + i) for i, layer in enumerate(layers)}}


def _save(tensors: dict, path: str, metadata: dict) -> None:
    tmp = path + ".tmp"
    save_file(tensors, tmp, metadata={k: str(v) for k, v in metadata.items()})
    os.chmod(tmp, 0o644)
    os.replace(tmp, path)


def _readback_equal(tensors: dict, path: str) -> bool:
    ok = True
    with safe_open(path, framework="pt") as h:
        ok &= set(h.keys()) == set(tensors)
        for key, t in tensors.items():
            r = h.get_tensor(key)
            ok &= r.dtype == t.dtype and r.shape == t.shape
            ok &= bool(torch.equal(r.view(torch.uint8) if r.dtype == torch.float8_e4m3fn else r,
                                   t.view(torch.uint8) if t.dtype == torch.float8_e4m3fn else t))
    return bool(ok)


def stage_others(args, plan) -> None:
    for shard, keys in plan["other_groups"].items():
        dst = os.path.join(args.dst, shard)
        if os.path.exists(dst):
            print(f"[others] {shard}: exists, skipped")
            continue
        t0 = time.time()
        by_src: dict[str, list[str]] = {}
        for k in keys:
            by_src.setdefault(plan["wmap"][k], []).append(k)
        tensors = {}
        for src_shard, ks in by_src.items():
            with safe_open(os.path.join(args.src, src_shard), framework="pt") as h:
                for k in ks:
                    tensors[k] = h.get_tensor(k)
        _save(tensors, dst, {"format": "pt", "magi2_nvfp4": "unchanged copy of non-expert tensors"})
        ok = _readback_equal(tensors, dst)
        print(f"[others] {shard}: {len(tensors)} tensors, {os.path.getsize(dst):,} B, readback_equal={ok}, {time.time() - t0:.1f}s")
        if not ok:
            raise RuntimeError(f"read-back mismatch in {dst}")


def _quantile_summary(v: torch.Tensor) -> dict:
    v = v.double().flatten().cpu()
    q = torch.quantile(v, torch.tensor([0.0, 0.01, 0.5, 0.99, 1.0], dtype=torch.float64))
    return dict(zip(["min", "p01", "p50", "p99", "max"], [round(float(x), 6) for x in q]))


def stage_experts(args, plan) -> None:
    from inference.flash_mh_moe.nvfp4 import nvfp4_quantize_experts, pack_file_layout

    dev = torch.device(args.device)
    layers = [int(v) for v in args.layers.split(",")] if args.layers else plan["layers"]
    for layer in layers:
        shard = plan["layer_shard"][layer]
        dst = os.path.join(args.dst, shard)
        stats_path = os.path.join(args.dst, f".stats_layer{layer:02d}.json")
        if os.path.exists(dst) and os.path.exists(stats_path):
            print(f"[experts] layer {layer}: {shard} exists, skipped")
            continue
        t0 = time.time()
        tensors, stats = {}, {"layer": layer, "shard": shard, "block_scale": args.block_scale, "tensors": {}}
        t_quant = 0.0
        for name in EXPERT_NAMES:
            key = f"block.layers.{layer}.mlp.moe_mlp.{name}"
            with safe_open(os.path.join(args.src, plan["wmap"][key]), framework="pt") as h:
                w = h.get_tensor(key)
            assert w.dtype == torch.bfloat16 and w.dim() == 3, (key, w.dtype, w.shape)
            E, K, N = w.shape
            weight = torch.empty((E, N, K // 2), dtype=torch.uint8)
            n_blk_rows, n_blk_cols = -(-N // 128) * 128, -(-(K // 16) // 4) * 4
            wscale = torch.empty((E, n_blk_rows, n_blk_cols), dtype=torch.uint8)
            scale2 = torch.empty(E, dtype=torch.float32)
            num = torch.empty(E, dtype=torch.float64)
            den = torch.empty(E, dtype=torch.float64)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            tq = time.time()
            for e0 in range(0, E, args.chunk):
                codes, bcode, s, n_, d_ = nvfp4_quantize_experts(
                    w[e0 : e0 + args.chunk].to(dev), block_scale=args.block_scale, return_error=True)
                packed, tiled = pack_file_layout(codes, bcode)
                weight[e0 : e0 + args.chunk] = packed.cpu()
                wscale[e0 : e0 + args.chunk] = tiled.cpu()
                scale2[e0 : e0 + args.chunk] = s.cpu()
                num[e0 : e0 + args.chunk] = n_.cpu()
                den[e0 : e0 + args.chunk] = d_.cpu()
            t_quant += time.time() - tq
            quant = {"format": "nvfp4", "num_experts": E}
            tensors[key + ".weight"] = weight
            tensors[key + ".weight_scale"] = wscale.view(torch.float8_e4m3fn)
            tensors[key + ".weight_scale_2"] = scale2
            tensors[key + ".comfy_quant"] = torch.tensor(list(json.dumps(quant).encode("utf-8")), dtype=torch.uint8)
            stats["tensors"][key] = {
                "source_shape_E_in_out": [E, K, N], "source_dtype": "BF16", "source_bytes": w.numel() * 2,
                "nvfp4_bytes": weight.numel() + wscale.numel() + scale2.numel() * 4,
                "rel_fro_error": round(float((num.sum() / den.sum()).sqrt()), 6),
                "per_expert_rel_fro_error": _quantile_summary((num / den.clamp_min(1e-300)).sqrt()),
                "weight_scale_2": _quantile_summary(scale2),
            }
            del w
        _save(tensors, dst, {"format": "pt", "magi2_nvfp4": f"layer {layer} experts, block_scale={args.block_scale}"})
        t_write = time.time()
        ok = _readback_equal(tensors, dst)
        stats.update({"file_bytes": os.path.getsize(dst), "readback_equal": ok, "seconds_total": round(time.time() - t0, 2),
                      "seconds_quantise": round(t_quant, 2), "seconds_readback": round(time.time() - t_write, 2)})
        if not ok:
            os.remove(dst)
            raise RuntimeError(f"read-back mismatch in {dst}")
        json.dump(stats, open(stats_path, "w"), indent=1)
        errs = " ".join(f"{v['rel_fro_error']:.4f}" for v in stats["tensors"].values())
        print(f"[experts] layer {layer}: {shard} {stats['file_bytes']:,} B  rel err {errs}  readback_equal={ok}  "
              f"{stats['seconds_total']:.1f}s (quantise {t_quant:.1f}s)", flush=True)
        del tensors


def stage_finalize(args, plan) -> None:
    shards = list(plan["other_groups"]) + [plan["layer_shard"][layer] for layer in plan["layers"]]
    missing = [s for s in shards if not os.path.exists(os.path.join(args.dst, s))]
    if missing:
        raise SystemExit(f"finalize: {len(missing)} shards missing, e.g. {missing[:3]}")
    weight_map, total, per_dtype = {}, 0, {}
    for s in shards:
        for key, v in _header(os.path.join(args.dst, s)).items():
            assert key not in weight_map, key
            weight_map[key] = s
            nbytes = v["data_offsets"][1] - v["data_offsets"][0]
            total += nbytes
            per_dtype[v["dtype"]] = per_dtype.get(v["dtype"], 0) + nbytes
    # every source tensor is present: unchanged keys as-is, expert keys as the four bank keys
    src_headers = {}
    for s in sorted(set(plan["wmap"].values())):
        src_headers.update(_header(os.path.join(args.src, s)))
    problems = []
    for key, v in src_headers.items():
        if _expert_match(key):
            E, K, N = v["shape"]
            want = {".weight": ("U8", [E, N, K // 2]), ".weight_scale": ("F8_E4M3", [E, N, K // 16]),
                    ".weight_scale_2": ("F32", [E])}
            for suffix, (dt, shape) in want.items():
                h = _header(os.path.join(args.dst, weight_map[key + suffix]))[key + suffix] if key + suffix in weight_map else None
                if h is None or h["dtype"] != dt or h["shape"] != shape:
                    problems.append(key + suffix)
            if key + ".comfy_quant" not in weight_map:
                problems.append(key + ".comfy_quant")
        else:
            h = _header(os.path.join(args.dst, weight_map[key]))[key] if key in weight_map else None
            if h is None or h["dtype"] != v["dtype"] or h["shape"] != v["shape"]:
                problems.append(key)
    if problems:
        raise SystemExit(f"finalize: {len(problems)} tensors missing or with the wrong dtype/shape, e.g. {problems[:5]}")
    index = {"metadata": {"total_size": total}, "weight_map": dict(sorted(weight_map.items()))}
    json.dump(index, open(os.path.join(args.dst, INDEX_NAME), "w"), indent=2)

    layer_stats = [json.load(open(os.path.join(args.dst, f".stats_layer{layer:02d}.json"))) for layer in plan["layers"]]
    rules = sorted({s["block_scale"] for s in layer_stats})
    errs = [t["rel_fro_error"] for s in layer_stats for t in s["tensors"].values()]
    src_index = os.path.join(args.src, INDEX_NAME)
    manifest = {
        "format": "magi2-preview-nvfp4",
        "format_version": FORMAT_VERSION,
        "created_unix": int(time.time()),
        "source": {
            "path": args.src_label or args.src,
            "index_sha256": hashlib.sha256(open(src_index, "rb").read()).hexdigest(),
            "total_size": json.load(open(src_index))["metadata"]["total_size"],
            "tensors": len(src_headers),
        },
        "writer": {"script": "scripts/quantize_nvfp4.py", "git_revision": args.git_rev, "torch": torch.__version__},
        "quantiser": {
            "scheme": "NVFP4: E2M1 codes, E4M3 scale per 16 weights along the input dim, fp32 scale per expert",
            "per_expert_scale": "amax(expert) / (448 * 6), divided in bf16 then cast to fp32 (comfy-kitchen default)",
            "block_scale_rule": rules[0] if len(rules) == 1 else rules,
            "block_scale_rules": {
                "amax": "e4m3(block_amax / 6 / scale), comfy-kitchen's rule",
                "mse": "per block, the E4M3 code within +-6 codes of the amax code minimising the block's squared error",
            },
            "rounding": "E2M1 round to nearest, ties to even code; sign bit kept for -0",
        },
        "key_mapping": {
            "unchanged": "every non-expert tensor keeps its key, dtype and bytes",
            "experts": {
                "source": "block.layers.<L>.mlp.moe_mlp.{W_gate,W_up,W_down}  BF16 [E, in, out]",
                "<key>.weight": "U8 [E, out, in/2]: source permute(0, 2, 1), two E2M1 codes per byte, element 2i in the high nibble",
                "<key>.weight_scale": "F8_E4M3 [E, out, in/16]: block scales, each expert in comfy-kitchen (cuBLAS) tiled order",
                "<key>.weight_scale_2": "F32 [E]: per-expert scale",
                "<key>.comfy_quant": 'U8 UTF-8 JSON {"format": "nvfp4", "num_experts": E}',
                "dequantise": "value[e, out, in] = E2M1(code) * weight_scale[e][out, in // 16] * weight_scale_2[e]",
            },
        },
        "sizes": {
            "total_tensor_bytes": total,
            "bytes_by_dtype": per_dtype,
            "file_bytes": sum(os.path.getsize(os.path.join(args.dst, s)) for s in shards),
            "shards": len(shards),
            "tensors": len(weight_map),
        },
        "error_summary": {
            "expert_tensors": len(errs),
            "rel_fro_error_min": min(errs), "rel_fro_error_max": max(errs),
            "rel_fro_error_mean": round(sum(errs) / len(errs), 6),
            "all_shards_readback_equal": all(s["readback_equal"] for s in layer_stats),
            "seconds_quantise_total": round(sum(s["seconds_quantise"] for s in layer_stats), 1),
            "seconds_total": round(sum(s["seconds_total"] for s in layer_stats), 1),
        },
        "layers": layer_stats,
    }
    json.dump(manifest, open(os.path.join(args.dst, MANIFEST_NAME), "w"), indent=1)
    print(f"[finalize] {len(shards)} shards, {len(weight_map)} tensors, {total:,} B of tensor data; "
          f"expert rel err {min(errs):.4f}-{max(errs):.4f}; wrote {INDEX_NAME} and {MANIFEST_NAME}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="source preview checkpoint dir (read-only)")
    ap.add_argument("--dst", required=True, help="output dir, e.g. .../ckpt-nvfp4/preview")
    ap.add_argument("--stage", choices=("all", "others", "experts", "finalize"), default="all")
    ap.add_argument("--block-scale", choices=("mse", "amax"), default="mse")
    ap.add_argument("--layers", help="comma list of MoE layers for the experts stage (default: all)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--chunk", type=int, default=128, help="experts quantised at a time")
    ap.add_argument("--git-rev", default="", help="recorded in the manifest")
    ap.add_argument("--src-label", default="", help="source path to record in the manifest (host path)")
    args = ap.parse_args(argv)
    if os.path.realpath(args.src) == os.path.realpath(args.dst):
        raise SystemExit("--dst must differ from --src")
    os.makedirs(args.dst, exist_ok=True)
    plan = _plan(args.src)
    if args.stage in ("all", "others"):
        stage_others(args, plan)
    if args.stage in ("all", "experts"):
        stage_experts(args, plan)
    if args.stage in ("all", "finalize"):
        stage_finalize(args, plan)
    return 0


if __name__ == "__main__":
    sys.exit(main())
