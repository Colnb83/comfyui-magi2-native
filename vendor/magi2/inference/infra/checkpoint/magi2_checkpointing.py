# Copyright (c) 2026 SandAI. All Rights Reserved.
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

"""Checkpoint loading for MAGI-2.

Weights ship as a single HuggingFace-style safetensors directory holding the
full, unsharded expert stack. Every expert-parallel rank reads that same
directory but pulls only the slice of each MoE tensor it owns, so one copy on
disk serves any ``ep_size``.
"""

import json
import os
from collections import defaultdict
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import torch
from safetensors import safe_open
from tqdm.auto import tqdm

from inference.common.magi2_config import EngineConfig
from inference.infra.distributed import psm
from inference.utils import print_rank_0

_INDEX_NAME = "model.safetensors.index.json"
_SINGLE_NAME = "model.safetensors"

# MoE tensors are concatenated over (head, expert) along dim 0 and split across
# expert-parallel ranks; everything else is replicated.
_EP_SHARDED_SUFFIXES = (
    "moe_mlp.gate",
    "moe_mlp.W_gate",
    "moe_mlp.W_up",
    "moe_mlp.W_down",
    "moe_mlp.router.expert_bias",
    "moe_mlp.router.expert_bias_ema",
)


def read_safetensors_weight_map(checkpoint_dir: str) -> dict[str, str]:
    """Map every tensor name in a HuggingFace-style directory to its file."""
    index_path = os.path.join(checkpoint_dir, _INDEX_NAME)
    if os.path.exists(index_path):
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]
        return {key: os.path.join(checkpoint_dir, shard) for key, shard in weight_map.items()}

    single_path = os.path.join(checkpoint_dir, _SINGLE_NAME)
    if not os.path.exists(single_path):
        raise FileNotFoundError(f"No {_INDEX_NAME} or {_SINGLE_NAME} under {checkpoint_dir}")
    with safe_open(single_path, framework="pt") as f:
        return {key: single_path for key in f.keys()}


def load_safetensors_dir(checkpoint_dir: str, desc: str = "Loading shards") -> dict[str, torch.Tensor]:
    """Load every tensor from a HuggingFace-style safetensors directory."""
    return _load_by_shard(read_safetensors_weight_map(checkpoint_dir), reader=_read_whole, desc=desc)


def _is_ep_sharded_key(key: str) -> bool:
    return ".moe_mlp." in key and key.endswith(_EP_SHARDED_SUFFIXES)


def _read_whole(handle: Any, key: str, _target: torch.Tensor | None) -> torch.Tensor:
    return handle.get_tensor(key)


def _read_ep_slice(handle: Any, key: str, target: torch.Tensor | None) -> torch.Tensor:
    """Read only the (head, expert) rows this expert-parallel rank owns.

    ``num_heads`` is padded up to a multiple of ``ep_size``, so the trailing
    ranks may fall wholly or partly past the real experts and get zeros.
    """
    if target is None or not _is_ep_sharded_key(key):
        return _read_whole(handle, key, target)

    ep_size = psm.get_world_size("ep")
    view = handle.get_slice(key)
    real_rows = view.get_shape()[0]
    shard_rows = target.shape[0]
    if ep_size <= 1 or shard_rows == real_rows:
        return handle.get_tensor(key)

    start = psm.get_local_rank("ep") * shard_rows
    if start >= real_rows:
        return torch.zeros_like(target, device="cpu")

    chunk = view[start : min(start + shard_rows, real_rows)]
    if chunk.shape[0] == shard_rows:
        return chunk
    padding = torch.zeros((shard_rows - chunk.shape[0], *chunk.shape[1:]), dtype=chunk.dtype)
    return torch.cat([chunk, padding], dim=0)


def _read_ep_rows(handle: Any, key: str, shard_rows: int) -> torch.Tensor:
    """Rows of ``key`` owned by this expert-parallel rank, zero-padded past the real experts."""
    ep_size = psm.get_world_size("ep")
    view = handle.get_slice(key)
    real_rows = view.get_shape()[0]
    if ep_size <= 1 or shard_rows == real_rows:
        return handle.get_tensor(key)
    start = psm.get_local_rank("ep") * shard_rows
    chunk = view[min(start, real_rows) : min(start + shard_rows, real_rows)]
    if chunk.shape[0] == shard_rows:
        return chunk
    padding = torch.zeros((shard_rows - chunk.shape[0], *chunk.shape[1:]), dtype=chunk.dtype)
    return torch.cat([chunk, padding], dim=0)


def resolve_nvfp4_checkpoint_dir() -> str | None:
    """Directory of a pre-quantized NVFP4 preview checkpoint, if one is configured.

    ``MAGI2_NVFP4_CKPT`` may name the checkpoint root (holding ``preview/``) or
    the preview directory itself. Only used with ``MAGI2_MOE_QUANT=nvfp4``.
    """
    if (os.environ.get("MAGI2_MOE_QUANT") or "").strip().lower() != "nvfp4":
        return None
    root = (os.environ.get("MAGI2_NVFP4_CKPT") or "").strip()
    if not root:
        return None
    for candidate in (os.path.join(root, "preview"), root):
        if os.path.exists(os.path.join(candidate, _INDEX_NAME)):
            return candidate
    raise FileNotFoundError(f"MAGI2_NVFP4_CKPT={root}: no {_INDEX_NAME} in it or in its preview/ subdirectory")


def _load_by_shard(
    weight_map: Mapping[str, str],
    reader,
    desc: str,
    targets: Mapping[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Read tensors grouped by file so each shard is opened exactly once."""
    by_shard: dict[str, list[str]] = defaultdict(list)
    for key, path in weight_map.items():
        by_shard[path].append(key)

    def read_shard(item: tuple[str, list[str]]) -> dict[str, torch.Tensor]:
        path, keys = item
        with safe_open(path, framework="pt") as handle:
            return {key: reader(handle, key, None if targets is None else targets.get(key)) for key in keys}

    state_dict: dict[str, torch.Tensor] = {}
    with ThreadPoolExecutor() as executor:
        futures = list(executor.map(read_shard, by_shard.items()))
        for shard_state in tqdm(futures, desc=desc, total=len(futures), disable=psm.get_global_rank() != 0):
            state_dict.update(shard_state)
    return state_dict


def _apply_router_bias_ema(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Pair the router bias buffer with the EMA-smoothed model weights.

    The checkpoint stores both ``expert_bias`` (latest training value) and
    ``expert_bias_ema`` (EMA-smoothed value). Inference uses the EMA model
    weights, so the router must use the paired EMA bias; otherwise routing
    biases mismatch the weights and quality degrades. Set the env var
    ``MAGI2_ROUTER_BIAS_SOURCE=main`` to keep the raw ``expert_bias`` instead.
    """
    if (os.environ.get("MAGI2_ROUTER_BIAS_SOURCE") or "ema").strip().lower() == "main":
        print_rank_0("[magi2] Router bias source: expert_bias (raw, MAGI2_ROUTER_BIAS_SOURCE=main)")
        return state_dict

    copied = 0
    for key, value in list(state_dict.items()):
        if not key.endswith("moe_mlp.router.expert_bias_ema"):
            continue
        target_key = key[: -len("_ema")]
        target = state_dict.get(target_key)
        if target is None or tuple(target.shape) != tuple(value.shape):
            continue
        state_dict[target_key] = value.clone()
        copied += 1

    print_rank_0(f"[magi2] Router bias: expert_bias_ema -> expert_bias ({copied} tensor(s))")
    return state_dict


_EXPERT_WEIGHT_SUFFIXES = ("moe_mlp.W_gate", "moe_mlp.W_up", "moe_mlp.W_down")


@torch.no_grad()
def release_preview_experts_(model: torch.nn.Module) -> int:
    """Free every MoE layer's expert weights on GPU; returns bytes freed."""
    from inference.model.magi2_preview import CoreMultiHeadMoE

    return sum(m.release_expert_storage_() for m in model.modules() if isinstance(m, CoreMultiHeadMoE))


@torch.no_grad()
def reload_preview_experts_(model: torch.nn.Module, engine_config: EngineConfig) -> bool:
    """Re-materialize expert weights freed by ``release_preview_experts_``.

    Returns False (no-op) if nothing was released.
    """
    from inference.model.magi2_preview import CoreMultiHeadMoE
    from inference.utils import env_is_true

    moes = [m for m in model.modules() if isinstance(m, CoreMultiHeadMoE)]
    if not any(m.expert_storage_released() for m in moes):
        return False
    try:
        for m in moes:
            m.restore_expert_storage_()
        if env_is_true("SKIP_LOAD_MODEL"):
            for m in moes:
                for name in m._EXPERT_PARAM_NAMES:
                    getattr(m, name).zero_()
        else:
            stream_load_magi2_model_(model, engine_config, only_suffixes=_EXPERT_WEIGHT_SUFFIXES)
    except BaseException:
        # A partial reload must not look complete: ``expert_storage_released`` only
        # tests the storage size, so a long-lived caller (the service) would skip
        # the reload on its next job and run with uninitialised experts. Release
        # everything again so the next attempt reloads from scratch, then re-raise.
        for m in moes:
            try:
                m.release_expert_storage_()
            except Exception:  # noqa: BLE001 - best effort; the original error matters
                pass
        raise
    return True


@torch.no_grad()
def stream_load_magi2_model_(
    model: torch.nn.Module,
    engine_config: EngineConfig,
    only_suffixes: tuple[str, ...] | None = None,
) -> None:
    """Load weights tensor-by-tensor straight into an already-on-GPU model.

    Unlike :func:`load_magi2_model_state_dict`, nothing is accumulated in host
    RAM, so two EP ranks can load a checkpoint larger than system memory. MoE
    expert tensors are routed through ``CoreMultiHeadMoE.load_expert_weight_``
    so they can be quantized on the GPU when MAGI2_MOE_FP8=1.
    """
    from inference.model.magi2_preview import CoreMultiHeadMoE

    checkpoint_dir = engine_config.load
    if not checkpoint_dir:
        raise ValueError("engine_config.load must point at a safetensors checkpoint directory")
    nvfp4_dir = resolve_nvfp4_checkpoint_dir()
    if nvfp4_dir is not None:
        checkpoint_dir = nvfp4_dir
        print_rank_0(f"[magi2] Pre-quantized NVFP4 checkpoint: {nvfp4_dir} (experts are not re-quantized)")
    print_rank_0(f"[magi2] Streaming preview weights from {checkpoint_dir} to GPU")

    targets = model.state_dict()
    if only_suffixes is not None:
        targets = {k: v for k, v in targets.items() if k.endswith(only_suffixes)}
    file_map = read_safetensors_weight_map(checkpoint_dir)
    # A pre-quantized checkpoint stores expert tensor K as the ComfyUI bank
    # K.weight / K.weight_scale / K.weight_scale_2 (+ K.comfy_quant); they are
    # read together when K comes up.
    nvfp4_keys = {
        k for k in targets if k.endswith(_EXPERT_WEIGHT_SUFFIXES) and k not in file_map and f"{k}.weight" in file_map
    }
    weight_map = {k: p for k, p in file_map.items() if k in targets}
    weight_map.update({k: file_map[f"{k}.weight"] for k in nvfp4_keys})
    missing = sorted(set(targets) - set(weight_map))
    if missing:
        raise KeyError(f"{len(missing)} model tensors missing from checkpoint, e.g. {missing[:5]}")

    use_ema_bias = (os.environ.get("MAGI2_ROUTER_BIAS_SOURCE") or "ema").strip().lower() != "main"
    modules = dict(model.named_modules())
    device = next(model.parameters()).device

    by_shard: dict[str, list[str]] = defaultdict(list)
    for key, path in weight_map.items():
        by_shard[path].append(key)

    copied_bias = 0
    for path, keys in tqdm(sorted(by_shard.items()), desc="Streaming shards", disable=psm.get_global_rank() != 0):
        with safe_open(path, framework="pt") as handle:
            for key in keys:
                if use_ema_bias and key.endswith("moe_mlp.router.expert_bias"):
                    continue  # filled from expert_bias_ema below
                mod_name, _, attr = key.rpartition(".")
                module = modules.get(mod_name)
                if key in nvfp4_keys:
                    rows = targets[key].shape[0]
                    parts = []
                    for suffix in (".weight", ".weight_scale", ".weight_scale_2"):
                        part_path = file_map[key + suffix]
                        if part_path == path:
                            parts.append(_read_ep_rows(handle, key + suffix, rows))
                        else:
                            with safe_open(part_path, framework="pt") as other:
                                parts.append(_read_ep_rows(other, key + suffix, rows))
                    module.load_expert_nvfp4_(attr, *parts)
                    del parts
                    continue
                value = _read_ep_slice(handle, key, targets[key]).to(device, non_blocking=True)
                if isinstance(module, CoreMultiHeadMoE) and attr in ("W_gate", "W_up", "W_down"):
                    module.load_expert_weight_(attr, value)
                else:
                    targets[key].copy_(value)
                if use_ema_bias and key.endswith("moe_mlp.router.expert_bias_ema"):
                    targets[key[: -len("_ema")]].copy_(value)
                    copied_bias += 1
                del value
    torch.cuda.synchronize(device)
    print_rank_0(f"[magi2] Streamed {len(weight_map)} tensors; router bias from EMA: {copied_bias}")


def load_magi2_model_state_dict(model: torch.nn.Module, engine_config: EngineConfig) -> Mapping[str, Any]:
    checkpoint_dir = engine_config.load
    if not checkpoint_dir:
        raise ValueError("engine_config.load must point at a safetensors checkpoint directory")

    print_rank_0(f"[magi2] Loading preview weights from {checkpoint_dir}")
    targets = model.state_dict()
    weight_map = {key: path for key, path in read_safetensors_weight_map(checkpoint_dir).items() if key in targets}

    state_dict = _load_by_shard(weight_map, reader=_read_ep_slice, desc="Loading shards", targets=targets)
    for key, value in state_dict.items():
        target = targets[key]
        if value.dtype != target.dtype and value.is_floating_point() and target.is_floating_point():
            state_dict[key] = value.to(dtype=target.dtype)
    return _apply_router_bias_ema(state_dict)
