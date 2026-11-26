# Copyright (c) 2026 SandAI. All Rights Reserved.
# Apache-2.0.
"""Memory-bounded refiner forward (``MAGI2_REFINER_LOWMEM=1``).

The stock forward treats the whole clip as one sequence and materialises every
intermediate at full length: at 1080p (514,080 video tokens, hidden 4096) that
is 7.85 GiB per fp32 activation, 11.8 GiB for the fp32 qkv projection and
41.8 GiB for the fp32 MLP up-projection, which does not fit one 96 GB card.

Nothing here changes the maths. The same modules and the same attention op are
called; only the *extent* of each call changes:

* Per-token work (adapters, RMS norms, qkv / gate / proj linears, q/k norm,
  RoPE, the MLP, the final norm + projection) runs over slices of the
  modality-sorted sequence and writes into preallocated full-length buffers.
* Attention runs over groups of query heads with their matching KV heads
  (heads are independent; the q:kv group ratio is kept). Each group's output
  overwrites that group's (now dead) queries.
* The residual stream is updated in place, in the dtype the stock code holds
  (bf16 until the first MLP add promotes it to fp32, then fp32).

Full-length tensors alive at the peak (1080p): residual fp32 7.85 GiB, q/out
bf16 3.93 GiB, k and v bf16 0.98 GiB each, gate fp32 0.06 GiB, RoPE table.

Tunables:
  MAGI2_REFINER_LOWMEM_TOKENS  tokens per slice for per-token work (default 16384)
  MAGI2_REFINER_LOWMEM_HEADS   query heads per attention call (default 8; 0 = all)
  MAGI2_REFINER_LOWMEM_LOG     1 = print peak memory and time per forward
"""

from __future__ import annotations

import os
import time
from typing import Iterator, List, Tuple

import torch


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    return int(raw) if raw else default


class _ChunkDispatcher:
    """Stands in for ModalityDispatcher on a slice that holds a single modality.

    ``dispatch`` hands the slice to its own expert and empty tensors to the
    others, so the unmodified MoE modules (norms, linears, LoRA) do exactly
    their per-expert computation on it.
    """

    def __init__(self, expert: int, num_modalities: int):
        self.expert = expert
        self.num_modalities = num_modalities

    def dispatch(self, x: torch.Tensor) -> List[torch.Tensor]:
        groups = [x[:0]] * self.num_modalities
        groups[self.expert] = x
        return groups

    def undispatch(self, *groups: torch.Tensor) -> torch.Tensor:
        return groups[self.expert]


def _slices(group_sizes: List[int], tokens: int) -> Iterator[Tuple[int, int, int]]:
    """(modality, start, end) over the modality-sorted sequence."""
    offset = 0
    for modality, size in enumerate(group_sizes):
        for a in range(offset, offset + size, tokens):
            yield modality, a, min(a + tokens, offset + size)
        offset += size


def _residual_add(hidden, promoted, a: int, b: int, update):
    """hidden[a:b] + update with the stock dtype promotion.

    Same dtype: in place. Otherwise (bf16 stream + fp32 update, once, at the
    first MLP) the sum goes into a new stream of the promoted dtype.
    """
    out_dtype = torch.promote_types(hidden.dtype, update.dtype)
    if out_dtype == hidden.dtype:
        hidden[a:b].add_(update)
        return promoted
    if promoted is None:
        promoted = torch.empty(hidden.shape, dtype=out_dtype, device=hidden.device)
    torch.add(hidden[a:b], update, out=promoted[a:b])
    return promoted


def _layer_forward(layer, hidden, cos_emb, sin_emb, perm, group_sizes, local_attn_handler,
                   cp_split_sizes, tokens: int, heads: int):
    from inference.model import magi2_refiner as R

    attn = layer.attention
    cfg = attn.config
    n_tok = hidden.shape[0]
    hq, hkv, hd = cfg.num_heads_q, cfg.num_heads_kv, cfg.head_dim
    dev = hidden.device
    bf16 = torch.bfloat16

    # Head-major so that a head group is one contiguous block for the kernel.
    q_all = torch.empty(hq, n_tok, hd, dtype=bf16, device=dev)
    k_all = torch.empty(hkv, n_tok, hd, dtype=bf16, device=dev)
    v_all = torch.empty(hkv, n_tok, hd, dtype=bf16, device=dev)
    g_all = (
        torch.empty(n_tok, hq, dtype=torch.float32, device=dev)
        if cfg.enable_attn_gating else None
    )

    # 1) pre-norm, qkv (+gate) projection, q/k norm, RoPE: per token.
    for modality, a, b in _slices(group_sizes, tokens):
        d = _ChunkDispatcher(modality, len(group_sizes))
        idx = perm[a:b]
        hn = attn.pre_norm(hidden[a:b], modality_dispatcher=d).to(bf16)
        qkv = attn.linear_qkv(hn, modality_dispatcher=d).to(torch.float32)
        q, k, v = torch.split(qkv, [attn.q_size, attn.kv_size, attn.kv_size], dim=1)
        q = q.view(-1, hq, hd)
        k = k.view(-1, hkv, hd)
        v = v.view(-1, hkv, hd)
        if g_all is not None:
            g_all[a:b] = attn.linear_g(hn, modality_dispatcher=d).to(torch.float32)
        del hn
        q = attn.q_norm(q, modality_dispatcher=d)
        k = attn.k_norm(k, modality_dispatcher=d)
        cos, sin = cos_emb[idx], sin_emb[idx]
        q = R.apply_rotary_emb(q.unsqueeze(0), cos, sin).squeeze(0)
        k = R.apply_rotary_emb(k.unsqueeze(0), cos, sin).squeeze(0)
        # The stock path un-permutes, then casts q/k/v to bf16 for the kernel.
        q_all[:, idx] = q.transpose(0, 1).to(bf16)
        k_all[:, idx] = k.transpose(0, 1).to(bf16)
        v_all[:, idx] = v.transpose(0, 1).to(bf16)
        del qkv, q, k, v, cos, sin

    # 2) attention, one group of query heads (and its KV heads) at a time.
    ratio = hq // hkv
    step = hq if heads <= 0 else max(ratio, (min(heads, hq) // ratio) * ratio)
    for h0 in range(0, hq, step):
        h1 = min(h0 + step, hq)
        qg = q_all[h0:h1].transpose(0, 1)
        kg = k_all[h0 // ratio:h1 // ratio].transpose(0, 1)
        vg = v_all[h0 // ratio:h1 // ratio].transpose(0, 1)
        if cfg.use_local_attn:
            out, _ = R._magi2_op("flex_flash_attn_func")(
                qg, kg, vg,
                local_attn_handler.q_ranges,
                local_attn_handler.k_ranges,
                local_attn_handler.attn_type_map,
                local_attn_handler.max_seqlen_q,
                bool(getattr(local_attn_handler, "auto_range_merge", False)),
            )
        else:
            out = R._magi2_op("flash_attn_func")(
                qg.unsqueeze(0), kg.unsqueeze(0), vg.unsqueeze(0)
            ).squeeze(0)
        q_all[h0:h1] = out.transpose(0, 1)  # these queries are dead: reuse as output
        del out, qg, kg, vg
    del k_all, v_all

    # 3) gate, output projection, residual add, then the MLP: per token.
    promoted_attn = None
    promoted_mlp = None
    mlp_pending = []  # only used if the attention add itself promotes (never in practice)
    for modality, a, b in _slices(group_sizes, tokens):
        d = _ChunkDispatcher(modality, len(group_sizes))
        idx = perm[a:b]
        o = q_all[:, idx].transpose(0, 1)
        if g_all is not None:
            o = o * torch.sigmoid(g_all[a:b].unsqueeze(-1))
        o = o.reshape(-1, hq * hd).to(bf16)
        o = attn.linear_proj(o, modality_dispatcher=d)
        if layer.post_norm:
            o = layer.attn_post_norm(o, modality_dispatcher=d)
        promoted_attn = _residual_add(hidden, promoted_attn, a, b, o)
        del o
        if promoted_attn is not None:
            mlp_pending.append((modality, a, b))
            continue
        m = layer.mlp(hidden[a:b], d)
        if layer.post_norm:
            m = layer.mlp_post_norm(m, modality_dispatcher=d)
        promoted_mlp = _residual_add(hidden, promoted_mlp, a, b, m)
        del m
    del q_all, g_all

    if promoted_attn is not None:
        hidden = promoted_attn
        for modality, a, b in mlp_pending:
            d = _ChunkDispatcher(modality, len(group_sizes))
            m = layer.mlp(hidden[a:b], d)
            if layer.post_norm:
                m = layer.mlp_post_norm(m, modality_dispatcher=d)
            promoted_mlp = _residual_add(hidden, promoted_mlp, a, b, m)
            del m
    if promoted_mlp is not None:
        hidden = promoted_mlp
    return hidden


def lowmem_forward(self, x, coords_mapping, modality_mapping, varlen_handler, local_attn_handler):
    """Drop-in for ``Transformer.forward``; eager, slice by slice."""
    from inference.infra.distributed import psm
    from inference.infra.parallelism.context_parallel import ulysses_scheduler
    from inference.model import magi2_refiner as R

    if psm.get_world_size("cp") > 1:
        # Context-parallel runs already hold 1/cp of the sequence per rank.
        return self._magi2_stock_forward(
            x, coords_mapping, modality_mapping, varlen_handler, local_attn_handler
        )

    tokens = max(1, _int_env("MAGI2_REFINER_LOWMEM_TOKENS", 16384))
    heads = _int_env("MAGI2_REFINER_LOWMEM_HEADS", 8)
    log = os.environ.get("MAGI2_REFINER_LOWMEM_LOG", "0") == "1"
    if log:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()

    x = ulysses_scheduler().dispatch(x)
    coords_mapping = ulysses_scheduler().dispatch(coords_mapping)
    modality_mapping = ulysses_scheduler().dispatch(modality_mapping)
    cp_split_sizes = ulysses_scheduler().cp_split_sizes

    dispatcher = R.ModalityDispatcher(modality_mapping, 3)
    perm = dispatcher.permute_mapping
    group_sizes = [int(s) for s in dispatcher.group_size.tolist()]
    n_tok = x.shape[0]
    cfg = self.config
    pre = self.pre_adapter
    post = self.post_adapter

    probe = self.block.layers[0].attention.linear_qkv.weight
    if probe.device != x.device:
        raise RuntimeError(
            "MAGI2_REFINER_LOWMEM=1 needs the refiner weights on the GPU "
            f"(found {probe.device}); run with MAGI_COMPILE_COMPILE_MODE=NONE"
        )

    rope = pre.rope(coords_mapping)
    sin_emb, cos_emb = rope.tensor_split(2, -1)

    # Pre-adapter: embed per modality, straight into the sorted bf16 stream.
    embedders = (
        (pre.video_embedder, pre.config.video_in_channels),
        (pre.audio_embedder, pre.config.audio_in_channels),
        (pre.text_embedder, pre.config.text_in_channels),
    )
    hidden = torch.empty(n_tok, pre.config.hidden_size, dtype=cfg.params_dtype, device=x.device)
    for modality, a, b in _slices(group_sizes, tokens):
        embedder, channels = embedders[modality]
        # stock: zeros(dtype=x.dtype)[mask] = embedder(...), then .to(params_dtype)
        hidden[a:b] = embedder(x[perm[a:b], :channels]).to(x.dtype)

    for layer in self.block.layers:
        hidden = _layer_forward(
            layer, hidden, cos_emb, sin_emb, perm, group_sizes, local_attn_handler,
            cp_split_sizes, tokens, heads,
        )

    # Post-adapter: final norm + projection per modality, written un-permuted.
    x_out = torch.zeros(
        n_tok,
        max(post.config.video_in_channels, post.config.audio_in_channels),
        device=hidden.device,
        dtype=hidden.dtype,
    )
    heads_out = (
        (post.final_norm_video, post.final_linear_video, post.config.video_in_channels),
        (post.final_norm_audio, post.final_linear_audio, post.config.audio_in_channels),
    )
    for modality, a, b in _slices(group_sizes, tokens):
        if modality >= len(heads_out):
            continue
        norm, linear, channels = heads_out[modality]
        y = linear(norm(hidden[a:b].to(norm.weight.dtype)))
        x_out[perm[a:b], :channels] = y.to(x_out.dtype)
        del y
    del hidden

    x_out = ulysses_scheduler().undispatch(x_out)
    if log:
        torch.cuda.synchronize()
        print(
            f"[magi2 refiner lowmem] forward tokens={n_tok} slice={tokens} heads={heads} "
            f"time={time.perf_counter() - t0:.2f}s "
            f"peak_alloc={torch.cuda.max_memory_allocated() / 2**30:.2f}GiB "
            f"peak_reserved={torch.cuda.max_memory_reserved() / 2**30:.2f}GiB",
            flush=True,
        )
    return x_out


def install(transformer_cls) -> None:
    """Route ``Transformer.forward`` to the low-memory path (stock one kept)."""
    if getattr(transformer_cls, "_magi2_stock_forward", None) is None:
        transformer_cls._magi2_stock_forward = transformer_cls.forward
        transformer_cls.forward = lowmem_forward
