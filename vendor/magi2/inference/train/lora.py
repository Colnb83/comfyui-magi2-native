# Copyright (c) 2026 SandAI. All Rights Reserved.
# Apache-2.0.
"""Low-rank adapters for the denoiser's non-expert linears, plus a registry of the
small native parameter groups that can be trained directly.

Every non-expert linear in the model is a ``GroupedLinearBase``: one weight
``[groups * out, in]`` holding ``groups`` matrices (1, or 3 in the multi-modality
layers: one per modality). A LoRA here is one pair ``A [groups, r, in]``,
``B [groups, out, r]`` per module and the effective weight is

    W + (alpha / r) * B @ A

applied as a ``torch.nn.utils.parametrize`` parametrisation, so the model code
reads ``module.weight`` as before and nothing in the forward changes. The base
weights stay frozen in bf16 and are never modified: the adapter is kept un-merged.

Native parameter groups (``NATIVE_GROUPS``) are trained in place, in fp32 copies
where the stored tensor is bf16. They matter for teaching the model things an
attention LoRA cannot: the router ``gate`` decides which experts a token uses.
"""

from __future__ import annotations

import json
import math
import re

import torch
import torch.nn as nn
from torch.nn.utils import parametrize

# LoRA target presets: regexes on the module path inside the Transformer.
LORA_PRESETS = {
    "attn": [r"\.attention\.linear_(qkv|proj)$"],
    "attn+mlp": [
        r"\.attention\.linear_(qkv|proj)$",
        r"\.mlp\.(split_linear|merge_linear)$",
        r"\.mlp\.(shared_expert|modality_specific_shared_expert)_fc[12]$",
        r"\.mlp\.(up_gate_proj|down_proj)$",
    ],
    "all": [r"\.attention\.linear_\w+$", r"\.mlp\.\w*(linear|fc)\w*$"],
}

# Native parameter groups: regexes on parameter names.
NATIVE_GROUPS = {
    "router": [r"\.moe_mlp\.gate$"],  # [heads * experts, head_dim]: which experts a token goes to
    "sinks": [r"\.attention\.sinks$"],
    "norms": [r"\.(pre_norm|q_norm|k_norm|mhc_norm)\.weight$", r"post_adapter\.final_norm_\w+\.weight$"],
    "mhc": [r"\.mhc_(alpha|bias|phi_fused)_\w+$"],
    "adapters": [r"^(pre_adapter|post_adapter)\.\w+\.(weight|bias)$"],
}


class LoRAParametrization(nn.Module):
    def __init__(self, groups: int, out_features: int, in_features: int, rank: int, alpha: float):
        super().__init__()
        self.groups, self.out_features, self.in_features = groups, out_features, in_features
        self.rank = rank
        self.scale = alpha / rank
        self.lora_A = nn.Parameter(torch.empty(groups, rank, in_features, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(groups, out_features, rank, dtype=torch.float32))
        for g in range(groups):
            nn.init.kaiming_uniform_(self.lora_A[g], a=math.sqrt(5))
        self.enabled = True

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return weight
        delta = torch.bmm(self.lora_B, self.lora_A) * self.scale  # [groups, out, in], fp32
        return weight + delta.reshape(weight.shape).to(weight.dtype)


def _matches(name: str, patterns) -> bool:
    return any(re.search(p, name) for p in patterns)


def add_lora(model: nn.Module, preset: str = "attn", rank: int = 16, alpha: float | None = None,
             layers: range | list[int] | None = None) -> dict[str, LoRAParametrization]:
    """Attach LoRA to every ``GroupedLinearBase`` whose path matches the preset.
    ``layers`` restricts it to those transformer layer indices. Returns path -> adapter."""
    from inference.model.magi2_preview import GroupedLinearBase

    patterns = LORA_PRESETS[preset] if preset in LORA_PRESETS else [preset]
    alpha = float(rank) if alpha is None else float(alpha)
    wanted = None if layers is None else set(layers)
    adapters = {}
    for name, module in list(model.named_modules()):
        if not isinstance(module, GroupedLinearBase) or not _matches("." + name, patterns):
            continue
        if wanted is not None:
            m = re.search(r"layers\.(\d+)\.", name)
            if m is None or int(m.group(1)) not in wanted:
                continue
        adapter = LoRAParametrization(module.num_experts, module.out_features, module.in_features, rank, alpha)
        adapter.to(module.weight.device)
        parametrize.register_parametrization(module, "weight", adapter)
        module.parametrizations.weight.original.requires_grad_(False)
        adapters[name] = adapter
    return adapters


def enable_native(model: nn.Module, groups: list[str]) -> dict[str, nn.Parameter]:
    """Make the named native parameter groups trainable. bf16 tensors are promoted
    to fp32 so small updates are not rounded away. Returns name -> parameter."""
    patterns = [p for g in groups for p in NATIVE_GROUPS[g]]
    out = {}
    for name, param in model.named_parameters():
        if "parametrizations" in name or not _matches("." + name if not name.startswith(("pre_", "post_")) else name, patterns):
            continue
        if not param.is_floating_point():
            continue
        if param.dtype != torch.float32:
            param.data = param.data.float()
        param.requires_grad_(True)
        out[name] = param
    return out


def trainable_state(adapters: dict[str, LoRAParametrization], native: dict[str, nn.Parameter]) -> dict[str, torch.Tensor]:
    state = {}
    for name, adapter in adapters.items():
        state[f"lora.{name}.A"] = adapter.lora_A.detach().cpu()
        state[f"lora.{name}.B"] = adapter.lora_B.detach().cpu()
    for name, param in native.items():
        state[f"native.{name}"] = param.detach().cpu()
    return state


def save_adapter(path: str, adapters, native, meta: dict) -> None:
    from safetensors.torch import save_file

    header = {k: (v if isinstance(v, str) else json.dumps(v)) for k, v in meta.items()}
    header["format"] = "magi2-adapter-v1"
    save_file(trainable_state(adapters, native), path, metadata=header)


def load_adapter(model: nn.Module, path: str) -> dict:
    """Attach a saved adapter to a model for inference or to resume training."""
    from safetensors import safe_open

    with safe_open(path, framework="pt") as f:
        meta = f.metadata() or {}
        tensors = {k: f.get_tensor(k) for k in f.keys()}
    rank, alpha = int(json.loads(meta["rank"])), float(json.loads(meta["alpha"]))
    modules = dict(model.named_modules())
    params = dict(model.named_parameters())
    names = sorted({k[len("lora."):-2] for k in tensors if k.startswith("lora.")})
    for name in names:
        module = modules[name]
        adapter = LoRAParametrization(module.num_experts, module.out_features, module.in_features, rank, alpha)
        adapter.lora_A.data.copy_(tensors[f"lora.{name}.A"])
        adapter.lora_B.data.copy_(tensors[f"lora.{name}.B"])
        adapter.to(module.weight.device)
        parametrize.register_parametrization(module, "weight", adapter)
    for key, value in tensors.items():
        if key.startswith("native."):
            param = params[key[len("native."):]]
            param.data = value.to(device=param.device, dtype=torch.float32)
    return meta
