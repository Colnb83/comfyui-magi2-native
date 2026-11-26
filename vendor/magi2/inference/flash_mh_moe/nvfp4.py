# Copyright (c) 2026 SandAI. All Rights Reserved.
# Apache-2.0.
"""NVFP4 storage for the MoE expert weights (storage only; compute stays bf16).

An expert tensor ``W`` is ``[E, K, N]`` (``K`` = input dim, the GEMM reduction
axis; ``N`` = output dim).  NVFP4 keeps, per expert matrix:

* 4-bit E2M1 codes (values 0, .5, 1, 1.5, 2, 3, 4, 6 and a sign bit),
* one FP8 E4M3 scale per block of 16 weights along ``K``,
* one fp32 scale,

so that ``W[e, k, n] = E2M1(code) * block_scale[e, n, k // 16] * scale[e]``:
4.5 bits per weight.  The numbers are the ones ComfyUI / comfy-kitchen produce
for the ``[N, K]`` matrix of each expert (``TensorCoreNVFP4Layout.quantize``:
scale = amax / (448 * 6), block scale = e4m3(block_amax / 6 / scale), codes
rounded to nearest even); ``scripts/nvfp4_probe.py crosscheck`` verifies that
bit for bit.

Two layouts are used:

* the *file* layout (ComfyUI's ``MoEExperts`` bank): codes ``[E, N, K/2]`` with
  element ``2i`` of a row in the HIGH nibble, block scales ``[E, N, K/16]`` in
  cuBLAS tiled order, scale ``[E]``;
* the *device* layout held by the model: codes ``[E, K, N/2]`` (pairs taken
  along ``N``, element ``2j`` in the high nibble), block scales
  ``[E, K/16, N]``, scale ``[E]``.  It is the same set of codes and scales,
  only transposed, so that a kernel reading it front to back writes the
  ``[E, K, N]`` bf16 tensor the MoE kernel consumes front to back.

The idea of dequantising NVFP4 experts with a small Triton kernel in front of
the bf16 grouped GEMM follows upstream PR #24 (``nvfp4_pack.py``, Apache-2.0);
no code is taken from it (its packing differs: low nibble first, ties rounded
toward zero, plain scale order).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

__all__ = [
    "BLOCK",
    "nvfp4_quantize_experts",
    "nvfp4_dequant_experts",
    "nvfp4_dequant_reference",
    "device_layout_from_file",
    "file_layout_from_device",
    "swizzle_block_scales",
    "unswizzle_block_scales",
]

BLOCK = 16
_F4_MAX = 6.0
_F8_MAX = 448.0


# ═══════════════════════════════════════════════════════════════════════════
#  Quantiser (plain PyTorch, any device)
# ═══════════════════════════════════════════════════════════════════════════


def _e4m3_values(device) -> torch.Tensor:
    """The 127 non-negative finite E4M3 values; index == uint8 code."""
    code = torch.arange(127, device=device)
    e, m = code >> 3, (code & 7).float()
    return torch.where(e == 0, m / 8.0 * 2.0**-6, (1.0 + m / 8.0) * torch.exp2(e.float() - 7.0))


def _e2m1_encode(a: torch.Tensor) -> torch.Tensor:
    """fp32 (already divided by the total scale) -> E2M1 code 0..15, round to
    nearest with ties to the even code, saturating at +-6, sign bit kept."""
    m = a.abs().clamp_(max=_F4_MAX)
    code = torch.where(
        m < 2.0,
        torch.round(m * 2.0),
        torch.where(m < 4.0, torch.round(m) + 2.0, torch.round(m * 0.5) + 4.0),
    ).to(torch.uint8)
    return code | (torch.signbit(a).to(torch.uint8) << 3)


def _e2m1_values(device) -> torch.Tensor:
    v = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    return torch.tensor(v + [-x for x in v], dtype=torch.float32, device=device)


def _codes_for(xb: torch.Tensor, total: torch.Tensor) -> torch.Tensor:
    zero = total == 0
    data = xb / torch.where(zero, torch.ones_like(total), total).unsqueeze(-1)
    data = torch.where(zero.unsqueeze(-1), torch.zeros_like(data), data)
    return _e2m1_encode(data)


@torch.no_grad()
def nvfp4_quantize_experts(
    w: torch.Tensor, block_scale: str = "amax", mse_span: int = 6, return_error: bool = False
):
    """Quantise ``w`` ``[g, K, N]`` (bf16 or fp32) to NVFP4, one scale per expert.

    Returns ``(codes [g, N, K] uint8 unpacked, bscale [g, N, K/16] uint8 E4M3
    codes, scale [g] fp32)`` and, with ``return_error``, the per-expert squared
    error and squared norm (float64).  ``block_scale``: ``"amax"`` is
    comfy-kitchen's rule; ``"mse"`` picks, per block, the E4M3 code within
    ``+-mse_span`` of the amax code that minimises the block's squared error.
    """
    g, K, N = w.shape
    assert K % BLOCK == 0 and N % 2 == 0, (K, N)
    x = w.transpose(1, 2).float().contiguous()  # [g, N, K]
    xb = x.view(g, N, K // BLOCK, BLOCK)
    amax = xb.abs().amax(dim=(1, 2, 3))
    # comfy divides in the tensor's own dtype (bf16 here) and then casts to fp32
    scale = (amax.to(torch.bfloat16) / (_F8_MAX * _F4_MAX)).to(torch.float32).clamp_min(1e-30)
    s = scale.view(g, 1, 1)
    rel = (xb.abs().amax(dim=-1) / _F4_MAX) / s
    bcode = rel.clamp(max=_F8_MAX).to(torch.float8_e4m3fn).view(torch.uint8).long()
    table = _e4m3_values(w.device)
    if block_scale == "mse":
        lut = _e2m1_values(w.device)
        best_err, best = None, bcode
        for d in range(-mse_span, mse_span + 1):
            cand = (bcode + d).clamp_(1, 126)
            total = s * table[cand]
            err = (lut[_codes_for(xb, total).long()] * total.unsqueeze(-1) - xb).square_().sum(dim=-1)
            if best_err is None:
                best_err, best = err, cand
            else:
                better = err < best_err
                best_err = torch.where(better, err, best_err)
                best = torch.where(better, cand, best)
        bcode = torch.where(bcode == 0, bcode, best)
    elif block_scale != "amax":
        raise ValueError(f"block_scale must be 'amax' or 'mse', got {block_scale!r}")
    total = s * table[bcode]
    codes = _codes_for(xb, total)
    out = (codes.view(g, N, K), bcode.to(torch.uint8), scale)
    if return_error:
        q = _e2m1_values(w.device)[codes.long()] * total.unsqueeze(-1)
        num = (q - xb).double().square_().sum(dim=(1, 2, 3))
        den = xb.double().square_().sum(dim=(1, 2, 3))
        out = out + (num, den)
    return out


def pack_device_layout(codes: torch.Tensor, bcode: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Unpacked codes ``[g, N, K]`` + block codes ``[g, N, K/16]`` -> device layout
    (codes ``[g, K, N/2]``, element ``2j`` along N in the high nibble; scales ``[g, K/16, N]``)."""
    ct = codes.transpose(1, 2)  # [g, K, N]
    packed = (ct[..., 0::2] << 4) | ct[..., 1::2]
    return packed.contiguous(), bcode.transpose(1, 2).contiguous()


def pack_file_layout(codes: torch.Tensor, bcode: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Unpacked codes ``[g, N, K]`` + block codes -> ComfyUI bank layout (codes
    ``[g, N, K/2]`` high nibble first; scales tiled per expert, uint8 E4M3 codes)."""
    packed = (codes[..., 0::2] << 4) | codes[..., 1::2]
    return packed.contiguous(), swizzle_block_scales(bcode)


def swizzle_block_scales(s: torch.Tensor) -> torch.Tensor:
    """``[..., R, C]`` -> cuBLAS tiled order ``[..., roundup(R,128), roundup(C,4)]``
    (comfy_kitchen.float_utils.to_blocked applied to every leading index)."""
    *lead, R, C = s.shape
    pr, pc = -(-R // 128) * 128, -(-C // 4) * 4
    if (pr, pc) != (R, C):
        p = torch.zeros((*lead, pr, pc), dtype=s.dtype, device=s.device)
        p[..., :R, :C] = s
        s = p
    t = s.reshape(-1, pr // 128, 4, 32, pc // 4, 4)
    return t.permute(0, 1, 4, 3, 2, 5).reshape(*lead, pr, pc).contiguous()


def unswizzle_block_scales(b: torch.Tensor, R: int, C: int) -> torch.Tensor:
    *lead, pr, pc = b.shape
    assert (pr, pc) == (-(-R // 128) * 128, -(-C // 4) * 4), (b.shape, R, C)
    t = b.reshape(-1, pr // 128, pc // 4, 32, 4, 4)
    return t.permute(0, 1, 4, 3, 2, 5).reshape(*lead, pr, pc)[..., :R, :C].contiguous()


def _unpack_hi_first(packed: torch.Tensor) -> torch.Tensor:
    return torch.stack((packed >> 4, packed & 0x0F), dim=-1).flatten(-2)


def device_layout_from_file(weight: torch.Tensor, weight_scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """ComfyUI bank tensors (``weight`` ``[g, N, K/2]`` uint8, ``weight_scale``
    tiled ``[g, N', K/16']`` uint8/float8) -> device layout.  Lossless."""
    g, N, half = weight.shape
    K = half * 2
    bs = weight_scale.view(torch.uint8) if weight_scale.dtype != torch.uint8 else weight_scale
    return pack_device_layout(_unpack_hi_first(weight), unswizzle_block_scales(bs, N, K // BLOCK))


def file_layout_from_device(codes_t: torch.Tensor, bscale_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Inverse of :func:`device_layout_from_file`."""
    codes = _unpack_hi_first(codes_t).transpose(1, 2)  # [g, N, K]
    return pack_file_layout(codes, bscale_t.transpose(1, 2))


def nvfp4_dequant_reference(codes_t: torch.Tensor, bscale_t: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Device layout -> bf16 ``[g, K, N]`` in plain PyTorch: fp32 product, one cast."""
    g, K, half = codes_t.shape
    N = half * 2
    vals = _e2m1_values(codes_t.device)[_unpack_hi_first(codes_t).long()]  # [g, K, N]
    total = _e4m3_values(codes_t.device)[bscale_t.long()] * scale.view(g, 1, 1)  # [g, K/16, N]
    return (vals.view(g, K // BLOCK, BLOCK, N) * total.unsqueeze(2)).view(g, K, N).to(torch.bfloat16)


# ═══════════════════════════════════════════════════════════════════════════
#  Dequantisation kernel: device layout -> bf16 [E, K, N]
# ═══════════════════════════════════════════════════════════════════════════


@triton.jit
def _e2m1_decode(c):
    e = (c >> 1) & 3
    m = (c & 1).to(tl.float32)
    v = tl.where(e == 0, 0.5 * m, (1.0 + 0.5 * m) * (1 << e).to(tl.float32) * 0.5)
    return tl.where((c & 8) != 0, -v, v)


@triton.jit
def _nvfp4_dequant_kernel(
    q_ptr,
    bs_ptr,
    s_ptr,
    out_ptr,
    n_bytes,
    HALF_N: tl.constexpr,
    K: tl.constexpr,
    TILE: tl.constexpr,
):
    # q: uint8 [E, K, N/2]; bs: e4m3 [E, K/16, N]; s: fp32 [E]; out: bf16 [E, K, N]
    offs = tl.program_id(0).to(tl.int64) * TILE + tl.arange(0, TILE)
    mask = offs < n_bytes
    b = tl.load(q_ptr + offs, mask=mask, other=0).to(tl.int32)
    row = offs // HALF_N  # e * K + k
    j = offs % HALF_N
    e = row // K
    # scale row index: e * (K/16) + k // 16  ==  row // 16  (K is a multiple of 16)
    bs_offs = (row // 16) * (2 * HALF_N) + 2 * j
    s_hi = tl.load(bs_ptr + bs_offs, mask=mask, other=0.0).to(tl.float32)
    s_lo = tl.load(bs_ptr + bs_offs + 1, mask=mask, other=0.0).to(tl.float32)
    scale = tl.load(s_ptr + e, mask=mask, other=0.0)
    v_hi = _e2m1_decode(b >> 4) * (s_hi * scale)
    v_lo = _e2m1_decode(b & 15) * (s_lo * scale)
    tl.store(out_ptr + 2 * offs, v_hi.to(tl.bfloat16), mask=mask)
    tl.store(out_ptr + 2 * offs + 1, v_lo.to(tl.bfloat16), mask=mask)


def nvfp4_dequant_experts(
    codes_t: torch.Tensor, bscale_t: torch.Tensor, scale: torch.Tensor, tile: int = 1024
) -> torch.Tensor:
    """Device layout (codes ``[E, K, N/2]`` uint8, block scales ``[E, K/16, N]``
    uint8 E4M3 codes, scale ``[E]`` fp32) -> bf16 ``[E, K, N]``, contiguous.
    Equal to ``round_bf16(E2M1 * (block_scale * scale))`` computed in fp32."""
    E, K, half = codes_t.shape
    assert K % BLOCK == 0 and bscale_t.shape == (E, K // BLOCK, 2 * half) and scale.shape == (E,)
    out = torch.empty((E, K, 2 * half), dtype=torch.bfloat16, device=codes_t.device)
    n_bytes = codes_t.numel()
    if n_bytes == 0:
        return out
    _nvfp4_dequant_kernel[(triton.cdiv(n_bytes, tile),)](
        codes_t,
        bscale_t.view(torch.float8_e4m3fn),
        scale,
        out,
        n_bytes,
        HALF_N=half,
        K=K,
        TILE=tile,
    )
    return out
