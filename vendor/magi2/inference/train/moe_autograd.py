# Copyright (c) 2026 SandAI. All Rights Reserved.
# Apache-2.0.
"""Backward pass for the multi-head MoE expert kernel.

The inference kernel (``flash_mh_moe_fwd``) is forward only. Training needs the
gradient of its output with respect to

* its input ``x`` (every adapter in front of an MoE layer needs this), and
* the routing probabilities ``probs`` (training the router, or anything that
  changes which experts are weighted how).

It does NOT need gradients for the expert weights themselves while the experts
stay frozen (NVFP4 storage, dequantised to bf16 constants).

The maths of one routed (token, head, expert e) pair, as in the kernel::

    g = clamp(x @ W_gate[e], max=7)          u = clamp(x @ W_up[e], -7, 7)
    hidden = g * sigmoid(1.702 * g) * (u + 1)
    y[token, head] += prob * (hidden @ W_down[e])

``moe_forward_reference`` is that maths in plain PyTorch (differentiable by
autograd; used on the CPU and as the test oracle). ``moe_backward`` is the
hand-written backward, which recomputes ``g`` and ``u`` instead of storing them:
stored, they would be 72 pairs x 1280 x 2 tensors per token and layer.

Pairs arrive sorted by expert (CSR ``expert_offsets``). Experts are processed in
buckets of similar group size with a padded ``bmm``, so cost and memory follow
the real number of pairs, not the largest group.

``MoEFunction`` ties the two together: forward = the Triton kernel on CUDA
(reference maths elsewhere), backward = ``moe_backward``.
"""

from __future__ import annotations

import os

import torch

ALPHA = 1.702
LIMIT = 7.0


def _bucket_budget() -> int:
    """Elements of the widest intermediate (pairs x expert width) per bucket."""
    return int(float(os.environ.get("MAGI2_MOE_BWD_BUCKET_GIB", "0.02")) * 2**30 / 4)


def _buckets(sizes: torch.Tensor, width: int):
    """Yield index tensors into ``sizes`` (experts of one head), largest groups
    first, each bucket's padded size bounded by the budget."""
    order = torch.argsort(sizes, descending=True)
    sizes_sorted = sizes[order].tolist()
    budget = _bucket_budget()
    start, total = 0, len(sizes_sorted)
    while start < total and sizes_sorted[start] > 0:
        n = sizes_sorted[start]
        count = max(1, min(total - start, budget // max(1, n * width)))
        end = start + count
        # drop trailing empty groups from the bucket
        while end > start + 1 and sizes_sorted[end - 1] == 0:
            end -= 1
        yield order[start:end], n
        start += count


def _swiglu(gate_raw: torch.Tensor, up_raw: torch.Tensor):
    gate = gate_raw.clamp(max=LIMIT)
    up = up_raw.clamp(min=-LIMIT, max=LIMIT)
    sig = torch.sigmoid(ALPHA * gate)
    glu = gate * sig
    return gate, up, sig, glu


def _walk(x, gather_ids, probs, expert_offsets, W_gate, visit):
    """Call ``visit(head, rows, pair_idx, mask, experts)`` for every bucket.

    rows: token index per padded pair ``[B, n]``; pair_idx: index into the sorted
    pair arrays ``[B, n]`` (clamped where padded); mask: validity ``[B, n]``;
    experts: global expert ids ``[B]``.
    """
    S, H, D = x.shape
    total_experts = expert_offsets.numel() - 1
    E = total_experts // H
    width = W_gate.shape[-1]
    offsets = expert_offsets.to(torch.long)
    sizes_all = offsets[1:] - offsets[:-1]
    last = max(int(offsets[-1].item()) - 1, 0)
    for h in range(H):
        sizes = sizes_all[h * E : (h + 1) * E]
        if int(sizes.sum().item()) == 0:
            continue
        for local, n in _buckets(sizes, width):
            experts = local + h * E
            ar = torch.arange(n, device=x.device)
            pair_idx = offsets[experts].unsqueeze(1) + ar.unsqueeze(0)
            mask = ar.unsqueeze(0) < sizes[local].unsqueeze(1)
            pair_idx = torch.where(mask, pair_idx, torch.full_like(pair_idx, last)).clamp_(max=last)
            rows = gather_ids[pair_idx].to(torch.long)
            visit(h, rows, pair_idx, mask, experts)


def moe_forward_reference(x, gather_ids, probs, expert_offsets, W_gate, W_up, W_down):
    """The kernel's maths in plain PyTorch. ``x`` ``[S, H, D]``; weights ``[H*E, D, W]``
    (gate, up) and ``[H*E, W, D]`` (down). Returns ``[S, H, D]`` in ``x.dtype``.
    Differentiable by autograd in ``x``, ``probs`` and the weights."""
    S, H, D = x.shape
    out = [torch.zeros(S, D, device=x.device, dtype=torch.float32 if x.dtype != torch.float64 else x.dtype) for _ in range(H)]
    wdtype = W_gate.dtype

    def visit(h, rows, pair_idx, mask, experts):
        xe = x[rows, h].to(wdtype)
        gate_raw = torch.bmm(xe, W_gate[experts]).to(out[h].dtype)
        up_raw = torch.bmm(xe, W_up[experts]).to(out[h].dtype)
        _, up, _, glu = _swiglu(gate_raw, up_raw)
        hidden = glu * (up + 1.0)
        y = torch.bmm(hidden.to(wdtype), W_down[experts]).to(out[h].dtype)
        y = y * (probs[pair_idx].to(y.dtype) * mask).unsqueeze(-1)
        out[h] = out[h].index_add(0, rows.reshape(-1), y.reshape(-1, D))

    _walk(x, gather_ids, probs, expert_offsets, W_gate, visit)
    return torch.stack(out, dim=1).to(x.dtype)


@torch.no_grad()
def moe_backward(x, grad_y, gather_ids, probs, expert_offsets, W_gate, W_up, W_down, need_x=True, need_probs=True):
    """Gradients of the MoE output with respect to ``x`` (``[S, H, D]``) and the
    sorted ``probs`` (``[T]``), given ``grad_y`` (``[S, H, D]``). Either may be
    skipped. Recomputes the gate/up pre-activations; stores nothing per pair
    beyond the current bucket."""
    S, H, D = x.shape
    acc = torch.float64 if x.dtype == torch.float64 else torch.float32
    wdtype = W_gate.dtype
    grad_x = [torch.zeros(S, D, device=x.device, dtype=acc) for _ in range(H)] if need_x else None
    grad_p = torch.zeros(probs.shape, device=x.device, dtype=acc) if need_probs else None

    def visit(h, rows, pair_idx, mask, experts):
        Wg, Wu, Wd = W_gate[experts], W_up[experts], W_down[experts]
        xe = x[rows, h].to(wdtype)
        gate_raw = torch.bmm(xe, Wg).to(acc)
        up_raw = torch.bmm(xe, Wu).to(acc)
        del xe
        gate, up, sig, glu = _swiglu(gate_raw, up_raw)
        ge = grad_y[rows, h].to(acc) * mask.unsqueeze(-1)
        if need_probs:
            hidden = glu * (up + 1.0)
            y = torch.bmm(hidden.to(wdtype), Wd).to(acc)
            del hidden
            contrib = (ge * y).sum(-1)
            del y
            # padded slots have ge == 0 and point at a real pair: adding 0 is harmless
            grad_p.index_add_(0, pair_idx.reshape(-1), contrib.reshape(-1))
        if need_x:
            g_out = ge * probs[pair_idx].to(acc).unsqueeze(-1)
            g_hidden = torch.bmm(g_out.to(wdtype), Wd.transpose(1, 2)).to(acc)
            del g_out
            d_gate = g_hidden * (up + 1.0) * (sig + ALPHA * gate * sig * (1.0 - sig)) * (gate_raw <= LIMIT)
            d_up = g_hidden * glu * ((up_raw >= -LIMIT) & (up_raw <= LIMIT))
            del g_hidden
            gxe = torch.bmm(d_gate.to(wdtype), Wg.transpose(1, 2)).to(acc)
            gxe += torch.bmm(d_up.to(wdtype), Wu.transpose(1, 2)).to(acc)
            grad_x[h].index_add_(0, rows.reshape(-1), gxe.reshape(-1, D))

    _walk(x, gather_ids, probs, expert_offsets, W_gate, visit)
    gx = torch.stack(grad_x, dim=1).to(x.dtype) if need_x else None
    gp = grad_p.to(probs.dtype) if need_probs else None
    return gx, gp


class MoEFunction(torch.autograd.Function):
    """Expert kernel with a backward for ``x`` and ``probs`` (weights are constants).

    The weights are passed as a callable returning ``(W_gate, W_up, W_down)`` and are
    produced again in backward. A whole layer of dequantised experts is 5.6 GiB:
    keeping them on the graph node would pin every layer's copy until backward
    (gradient checkpointing does not drop Python attributes of a node)."""

    @staticmethod
    def forward(ctx, x, probs, gather_ids, expert_offsets, weight_fn):
        W_gate, W_up, W_down = weight_fn()
        if x.is_cuda and x.dtype == torch.bfloat16:
            from inference.flash_mh_moe.fwd import flash_mh_moe_fwd

            y = flash_mh_moe_fwd(
                x=x, gather_ids=gather_ids, probs=probs, expert_offsets=expert_offsets,
                W_gate=W_gate, W_up=W_up, W_down=W_down,
            )
        else:
            y = moe_forward_reference(x, gather_ids, probs, expert_offsets, W_gate, W_up, W_down)
        ctx.save_for_backward(x, probs, gather_ids, expert_offsets)
        ctx.weight_fn = weight_fn
        return y

    @staticmethod
    def backward(ctx, grad_y):
        x, probs, gather_ids, expert_offsets = ctx.saved_tensors
        W_gate, W_up, W_down = ctx.weight_fn()
        need_x, need_probs = ctx.needs_input_grad[0], ctx.needs_input_grad[1]
        gx, gp = moe_backward(
            x, grad_y, gather_ids, probs, expert_offsets, W_gate, W_up, W_down,
            need_x=need_x, need_probs=need_probs,
        )
        return gx, gp, None, None, None


def moe_apply(x, gather_ids, probs, expert_offsets, weight_fn):
    """Differentiable stand-in for ``flash_mh_moe_fwd``. ``weight_fn()`` returns the
    three (constant) expert weight tensors."""
    return MoEFunction.apply(x, probs, gather_ids, expert_offsets, weight_fn)
