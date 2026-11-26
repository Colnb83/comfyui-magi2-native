#!/usr/bin/env python3
# Copyright (c) 2026 SandAI. All Rights Reserved.
# Apache-2.0.
"""Merge a sharded safetensors checkpoint into ONE safetensors file.

    scripts/merge_safetensors.py <checkpoint_dir> <out.safetensors> [--meta key=value ...]

Every tensor keeps its key, dtype, shape and bytes; only the container changes.
Tensor data is streamed shard by shard (constant memory). The output is written
to ``<out>.partial`` and renamed when complete. Used to turn the NVFP4 preview
checkpoint (39 shards) into a single file a ComfyUI diffusion-model loader can
list. Stdlib only.
"""

import argparse
import json
import os
import struct
import sys
import time

CHUNK = 64 << 20


def read_header(path):
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
    return header, 8 + n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint_dir")
    ap.add_argument("out")
    ap.add_argument("--meta", action="append", default=[], help="extra __metadata__ entry, key=value")
    args = ap.parse_args()

    shards = sorted(f for f in os.listdir(args.checkpoint_dir) if f.endswith(".safetensors"))
    if not shards:
        sys.exit(f"no .safetensors files in {args.checkpoint_dir}")
    if os.path.exists(args.out):
        sys.exit(f"{args.out} exists; refusing to overwrite")

    merged, plan, offset = {}, [], 0
    for shard in shards:
        path = os.path.join(args.checkpoint_dir, shard)
        header, data_start = read_header(path)
        header.pop("__metadata__", None)
        for key, info in sorted(header.items(), key=lambda kv: kv[1]["data_offsets"][0]):
            if key in merged:
                sys.exit(f"duplicate tensor {key} (again in {shard})")
            begin, end = info["data_offsets"]
            merged[key] = {"dtype": info["dtype"], "shape": info["shape"], "data_offsets": [offset, offset + end - begin]}
            plan.append((path, data_start + begin, end - begin))
            offset += end - begin

    meta = {"format": "pt", "merged_from": os.path.basename(os.path.abspath(args.checkpoint_dir)), "merged_shards": str(len(shards))}
    for item in args.meta:
        key, _, value = item.partition("=")
        meta[key] = value
    blob = json.dumps({"__metadata__": meta, **merged}, separators=(",", ":")).encode("utf-8")
    blob += b" " * (-len(blob) % 8)

    tmp = args.out + ".partial"
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    t0, done, last = time.time(), 0, 0.0
    with open(tmp, "wb") as out:
        out.write(struct.pack("<Q", len(blob)))
        out.write(blob)
        current, handle = None, None
        for path, start, length in plan:
            if path != current:
                if handle:
                    handle.close()
                handle, current = open(path, "rb"), path
            handle.seek(start)
            left = length
            while left:
                buf = handle.read(min(CHUNK, left))
                if not buf:
                    sys.exit(f"short read in {path}")
                out.write(buf)
                left -= len(buf)
                done += len(buf)
                if time.time() - last > 15:
                    last = time.time()
                    print(f"[merge] {done / 2**30:.1f} / {offset / 2**30:.1f} GiB", flush=True)
        if handle:
            handle.close()
        out.flush()
        os.fsync(out.fileno())
    os.rename(tmp, args.out)
    print(f"[merge] wrote {args.out}: {len(merged)} tensors, {offset / 2**30:.2f} GiB of data, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
