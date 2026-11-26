# Copyright (c) 2025 SandAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Range-masked attention via PyTorch flex_attention (non-Hopper fallback).

MagiAttention's ``flex_flash_attn_func`` takes a list of (q_range, k_range)
pairs: tokens in each q_range attend to the tokens in the paired k_range, and
a q token attends to the union of all k_ranges paired with ranges covering it.
Its kernels are sm_90-only. The refiner's original fallback looped over every
pair with one FA2 call each (~390k calls per layer at 1080p), which is
unusably slow.

Here the pairs are compiled once into a flex_attention ``BlockMask``:

1. All range endpoints cut the sequence into *segments*; every q_range and
   k_range is a contiguous run of segments.
2. An ``allowed[seg_q, seg_k]`` bool table records which segments interact,
   and ``mask_mod`` looks it up per element, so masking is exact even though
   ranges are not aligned to the 128-token kernel tile.
3. 128x128 tiles that touch no allowed segment pair are skipped entirely.

The mask depends only on the ranges, so it is cached and reused across all
layers and denoising steps of a clip.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import torch
from torch.nn.attention.flex_attention import BlockMask, flex_attention

_TILE = 128
_cache: dict = {}
_compiled_flex = None


def _flex():
    global _compiled_flex
    if _compiled_flex is None:
        _compiled_flex = torch.compile(flex_attention, dynamic=False)
    return _compiled_flex


def _kernel_options() -> dict:
    # sm_120 has ~99KB of shared memory per block; the default flex tiles
    # target sm_90/sm_100 and can exceed it at head_dim=128.
    opts = os.environ.get("MAGI2_FLEX_KERNEL_OPTIONS")  # e.g. "64,64,2,4"
    bm, bn, stages, warps = (int(v) for v in opts.split(",")) if opts else (64, 64, 2, 4)
    return {"BLOCK_M": bm, "BLOCK_N": bn, "num_stages": stages, "num_warps": warps}


@torch.no_grad()
def build_range_block_mask(
    q_ranges: torch.Tensor, k_ranges: torch.Tensor, seqlen_q: int, seqlen_k: int
) -> BlockMask:
    device = q_ranges.device
    q_ranges = q_ranges.to(device=device, dtype=torch.int64)
    k_ranges = k_ranges.to(device=device, dtype=torch.int64)

    # 1. Segments: cut both sequences at every range endpoint (shared index
    #    space; q and k are the same token sequence for self-attention).
    seqlen = max(seqlen_q, seqlen_k)
    bounds = torch.unique(
        torch.cat([q_ranges.flatten(), k_ranges.flatten(),
                   torch.tensor([0, seqlen], device=device)])
    )
    num_seg = bounds.numel() - 1
    token_seg = (
        torch.searchsorted(bounds, torch.arange(seqlen, device=device), right=True) - 1
    ).to(torch.int32)

    # 2. allowed[seg_q, seg_k]. Each range is a run of segments [lo, hi).
    q_lo = torch.searchsorted(bounds, q_ranges[:, 0])
    q_hi = torch.searchsorted(bounds, q_ranges[:, 1])
    k_lo = torch.searchsorted(bounds, k_ranges[:, 0])
    k_hi = torch.searchsorted(bounds, k_ranges[:, 1])
    allowed = torch.zeros(num_seg, num_seg, dtype=torch.bool, device=device)
    single = ((q_hi - q_lo) == 1) & ((k_hi - k_lo) == 1)
    allowed[q_lo[single], k_lo[single]] = True
    for i in torch.nonzero(~single).flatten().tolist():  # few wide ranges
        allowed[q_lo[i]:q_hi[i], k_lo[i]:k_hi[i]] = True

    # 3. Tile-level sparsity: tile t touches segments token_seg[t*T : (t+1)*T].
    def tile_seg_incidence(n_tokens: int) -> torch.Tensor:
        n_tiles = (n_tokens + _TILE - 1) // _TILE
        first = token_seg[torch.arange(n_tiles, device=device) * _TILE].long()
        last = token_seg[(torch.arange(n_tiles, device=device) * _TILE + _TILE - 1)
                         .clamp_max(n_tokens - 1)].long()
        inc = torch.zeros(n_tiles, num_seg, dtype=torch.float32, device=device)
        span = (last - first + 1).max().item()
        for off in range(span):  # a tile spans only a few segments
            seg = first + off
            ok = seg <= last
            inc[torch.nonzero(ok).flatten(), seg[ok]] = 1.0
        return inc

    inc_q = tile_seg_incidence(seqlen_q)
    inc_k = tile_seg_incidence(seqlen_k)
    tile_allowed = (inc_q @ allowed.float() @ inc_k.T) > 0  # [nq_tiles, nk_tiles]
    del inc_q, inc_k

    kv_num_blocks = tile_allowed.sum(dim=1, dtype=torch.int32)
    # Allowed tile columns first (stable order), padded with arbitrary indices.
    kv_indices = torch.argsort(~tile_allowed, dim=1, stable=True).to(torch.int32)

    def mask_mod(b, h, q_idx, kv_idx):
        return allowed[token_seg[q_idx], token_seg[kv_idx]]

    return BlockMask.from_kv_blocks(
        kv_num_blocks[None, None],
        kv_indices[None, None],
        BLOCK_SIZE=_TILE,
        mask_mod=mask_mod,
        seq_lengths=(seqlen_q, seqlen_k),
    )


def _get_block_mask(q_ranges, k_ranges, seqlen_q, seqlen_k) -> BlockMask:
    key = (
        seqlen_q, seqlen_k, q_ranges.shape[0],
        int(q_ranges.long().sum()), int(k_ranges.long().sum()),
        int((q_ranges[:, 0].long() * k_ranges[:, 1].long()).sum() % (2**61)),
    )
    mask = _cache.get(key)
    if mask is None:
        _cache.clear()  # one live mask per clip/resolution is enough
        mask = build_range_block_mask(q_ranges, k_ranges, seqlen_q, seqlen_k)
        _cache[key] = mask
    return mask


def range_flex_attn(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    q_ranges: torch.Tensor,
    k_ranges: torch.Tensor,
    softmax_scale: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """query [Sq, Hq, D], key/value [Sk, Hkv, D] -> (out [Sq, Hq, D], lse [Sq, Hq] fp32)."""
    sq, hq, _ = query.shape
    sk, hkv, _ = key.shape
    mask = _get_block_mask(q_ranges, k_ranges, sq, sk)
    out, lse = _flex()(
        query.transpose(0, 1).unsqueeze(0),
        key.transpose(0, 1).unsqueeze(0),
        value.transpose(0, 1).unsqueeze(0),
        block_mask=mask,
        scale=softmax_scale,
        enable_gqa=hq != hkv,
        return_lse=True,
        kernel_options=_kernel_options(),
    )
    out = out.squeeze(0).transpose(0, 1).contiguous()
    lse = lse.squeeze(0).transpose(0, 1).float().contiguous()
    return out, lse
