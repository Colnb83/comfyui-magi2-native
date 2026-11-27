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

"""Checkpoint loading for MAGI-2 models."""

import gc
import os

import torch

from inference.common.magi2_config import Magi2Config
from inference.infra.checkpoint.magi2_checkpointing import (
    load_magi2_model_state_dict,
    load_safetensors_dir,
)
from inference.utils import env_is_true, print_rank_0


def load_magi2_model(config: Magi2Config) -> torch.nn.Module:
    """Load the MAGI-2 DiT model from checkpoint."""
    from inference.model.magi2_preview import Transformer

    arch = config.arch_config
    print_rank_0(
        f"[magi2] Building model: layers={arch.num_layers}, "
        f"hidden={arch.hidden_size}, MoE experts={arch.moe_config.num_experts}"
    )

    # MAGI2_MOE_FP8 implies building on the GPU and streaming weights in, since
    # each rank's bf16 host copy would not fit in system RAM.
    # MAGI2_MOE_QUANT=nvfp4 quantizes on the GPU while streaming, same constraint.
    build_on_gpu = (
        env_is_true("MAGI2_MOE_FP8")
        or env_is_true("MAGI2_LOAD_TO_GPU")
        or (os.environ.get("MAGI2_MOE_QUANT") or "").strip().lower() == "nvfp4"
    )
    if build_on_gpu:
        with torch.device(f"cuda:{torch.cuda.current_device()}"):
            model = Transformer(arch, ep_size=config.engine_config.ep_size)
    else:
        model = Transformer(arch, ep_size=config.engine_config.ep_size)
    print_rank_0(f"[magi2] Model built, params_dtype={arch.params_dtype}, on_gpu={build_on_gpu}")
    # Kept so experts freed for the refiner stage can be re-streamed later.
    model._magi2_engine_config = config.engine_config

    if build_on_gpu and not env_is_true("SKIP_LOAD_MODEL"):
        from inference.infra.checkpoint.magi2_checkpointing import stream_load_magi2_model_

        stream_load_magi2_model_(model, config.engine_config)
    elif env_is_true("SKIP_LOAD_MODEL"):
        # Transformer.__init__ already filled in synthetic weights. Shapes,
        # compilation and memory footprint stay real; only the read is skipped.
        print_rank_0("[magi2] SKIP_LOAD_MODEL=1: synthetic weights, no checkpoint read")
    else:
        state_dict = load_magi2_model_state_dict(model, config.engine_config)
        model.load_state_dict(state_dict, strict=True)
        print_rank_0("[magi2] Checkpoint loaded successfully")

        del state_dict
        gc.collect()

    model = model.to(f"cuda:{torch.cuda.current_device()}")
    model.eval()
    adapter_path = (os.environ.get("MAGI2_ADAPTER") or "").strip()
    if adapter_path:
        # MAGI2_ADAPTER=<file from scripts/train.py>: attach a trained adapter, un-merged.
        from inference.train.lora import load_adapter

        meta = load_adapter(model, adapter_path)
        print_rank_0(f"[magi2] adapter {adapter_path}: step {meta.get('step')}, lora {meta.get('lora')} rank {meta.get('rank')}, native {meta.get('native')}")
    gc.collect()
    torch.cuda.empty_cache()
    return model


def load_magi2_refiner(config: Magi2Config) -> torch.nn.Module:
    """Load the super-resolution model."""
    from inference.model.magi2_refiner import Transformer as Magi2RefinerModel

    refiner_config = config.magi2_refiner_arch_config
    print_rank_0(
        f"[magi2] Building magi2_refiner: layers={refiner_config.num_layers}, "
        f"hidden={refiner_config.hidden_size}"
    )

    model = Magi2RefinerModel(refiner_config)
    if env_is_true("SKIP_LOAD_MODEL"):
        print_rank_0("[magi2] SKIP_LOAD_MODEL=1: synthetic refiner weights, no checkpoint read")
        model.eval()
        return model

    model_path = config.evaluation_config.magi2_refiner_model_path
    print_rank_0(f"[magi2] Loading magi2_refiner checkpoint from {model_path}")

    state_dict = load_safetensors_dir(model_path, desc="Loading magi2_refiner shards")

    model.load_state_dict(state_dict, strict=True)
    print_rank_0("[magi2] magi2_refiner checkpoint loaded successfully")

    del state_dict
    gc.collect()
    model.eval()
    return model
