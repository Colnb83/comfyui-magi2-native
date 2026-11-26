"""MAGI-2-preview as a ComfyUI model type.

Importing this module registers the model with ComfyUI core, without editing core:

* ``comfy.model_detection.detect_unet_config`` is wrapped so a state dict with the
  MAGI-2 key signature is recognised (everything else goes to the original function);
* a model config is added to ``comfy.supported_models.models``.

After that the stock **Load Diffusion Model** node returns a normal ``MODEL`` for
``magi2_preview_nvfp4_comfy.safetensors`` (made by the fork's
``scripts/convert_comfy_checkpoint.py``), ComfyUI's model management loads and unloads
it, and stock guiders and samplers drive it.

The denoiser is the MAGI-2 fork's own ``Transformer`` (NVFP4 expert storage, Triton
MoE kernel). It is built on the ``meta`` device and the checkpoint is assigned into
it, so nothing is copied or converted at load time. The latent is ComfyUI's joint
audio-video ``NestedTensor``: video ``[B, 48, T, H/16, W/16]`` in the raw Wan2.2 VAE
space and audio ``[B, 64, L]`` in the raw Stable Audio Open VAE space.

The fork's modules are plain ``torch.nn`` modules: the whole model must be on the GPU
to run. Low-VRAM partial loading is not supported.
"""

import logging
import math
import os

import torch

import comfy.conds
import comfy.latent_formats
import comfy.model_base
import comfy.model_detection
import comfy.model_management
import comfy.nested_tensor
import comfy.supported_models
import comfy.supported_models_base
import comfy.utils

from .fork_env import image_numerics, prepare_fork_import

IMAGE_MODEL = "magi2_preview"
SIGNATURE_KEYS = (
    "block.layers.0.mhc_norm.weight",
    "pre_adapter.video_embedder.weight",
    "pre_adapter.audio_embedder.weight",
    "post_adapter.final_linear_audio.weight",
)
LAYOUT_SUFFIX = "moe_mlp.W_gate_bscale"  # present only in the converted (device-layout) file
EXPERT_BUFFERS = tuple(f"{n}{s}" for n in ("W_gate", "W_up", "W_down") for s in ("_bscale", "_scale"))

# Inference memory beyond the weights, measured at 540p (57.9k tokens per forward):
# 81.8 GiB peak with 67.5 GiB of weights. One layer's dequantised experts are 5.6 GiB.
BASE_INFERENCE_BYTES = 6.5 * 2**30
BYTES_PER_TOKEN = 150e3


# -----------------------------------------------------------------------------
# Denoiser: the fork's Transformer behind ComfyUI's calling convention
# -----------------------------------------------------------------------------

_DENOISER_CLASS = None


def _denoiser_class():
    """The wrapper class, created on first use so that ComfyUI start-up does not
    import the fork (Triton, kernels)."""
    global _DENOISER_CLASS
    if _DENOISER_CLASS is not None:
        return _DENOISER_CLASS

    fork = prepare_fork_import()
    from inference.common.magi2_config import load_config
    from inference.model.magi2_preview import Transformer
    from inference.pipeline.preview_data_proxy import Magi2DataProxy, ModelInput

    class MAGI2Denoiser(Transformer):
        def __init__(self, dtype=None, device=None, operations=None, **kwargs):
            config = load_config(os.path.join(fork, "configs", "local", "1gpu_preview.json"))
            # No storage is allocated here; load_state_dict(assign=True) supplies it.
            with torch.device("meta"):
                super().__init__(config.arch_config, ep_size=1)
            # The fork keeps the expert scales out of its state dict; ComfyUI should
            # see, load, move and account for every tensor.
            for module in self.modules():
                for name in EXPERT_BUFFERS:
                    if name in module._buffers:
                        module._non_persistent_buffers_set.discard(name)
            self.data_proxy = Magi2DataProxy(config.evaluation_config.data_proxy_config)
            self.dtype = torch.bfloat16
            # MAGI-2's own pipeline gives the model the integer timestep (trunc(sigma*1000)).
            self.integer_timesteps = os.environ.get("MAGI2_COMFY_EXACT_TIMESTEP", "0") != "1"

        def _single_input(self, video, audio, context, t_norm):
            _, _, frames, height, width = video.shape
            length = audio.shape[1]
            device = video.device
            return ModelInput(
                x_t=video,
                audio_x_t=audio,
                audio_feat_len=torch.tensor([length], device=device),
                txt_feat=context,
                txt_feat_len=torch.tensor([context.shape[1]], device=device),
                t=t_norm,
                per_token_video_t=t_norm.view(1, 1, 1, 1, 1).expand(1, 1, frames, height, width).clone(),
                per_token_audio_t=t_norm.view(1, 1, 1).expand(1, length, 1).clone(),
                ref_audio_feat=torch.zeros(1, 0, audio.shape[2], dtype=audio.dtype, device=device),
                ref_audio_feat_len=torch.tensor([0], device=device),
                ref_video_feat=torch.zeros_like(video),
                ref_video_feat_len=torch.tensor([0], device=device),
                ref_image_feat=None,
                ref_image_feat_len=None,
                ref_image_special_token_embedding=None,
            )

        def forward(self, x, timestep, context=None, transformer_options={}, **kwargs):
            if not isinstance(x, (list, tuple)) or len(x) != 2:
                raise RuntimeError(
                    "MAGI-2 needs a joint video+audio latent (use the Empty MAGI-2 Latent node); "
                    "got a single-stream latent"
                )
            if context is None:
                raise RuntimeError("MAGI-2 needs text conditioning (use the MAGI-2 Text Encode node)")
            video, audio = x
            device = video.device
            probe = self.pre_adapter.video_embedder.weight
            if probe.device != device or probe.device.type != "cuda":
                raise RuntimeError(
                    f"MAGI-2 must be fully loaded on the GPU to run (weights are on {probe.device}, "
                    f"latents on {device}). It needs about 82 GiB of VRAM; low-VRAM partial loading "
                    "is not supported."
                )
            videos, audios = [], []
            with image_numerics():
                for i in range(video.shape[0]):
                    t = timestep[i].float()
                    if self.integer_timesteps:
                        t = torch.floor(t)
                    model_input = self._single_input(
                        video[i : i + 1].float(),
                        audio[i : i + 1].float().transpose(1, 2),  # ComfyUI [B, 64, L] -> MAGI-2 [B, L, 64]
                        context[i : i + 1].to(torch.bfloat16),
                        (t / 1000.0).reshape(1),
                    )
                    packed = self.data_proxy.process_input(model_input)
                    output = Transformer.forward(self, *packed)
                    del packed
                    out_video, out_audio = self.data_proxy.process_output(output)
                    del output
                    videos.append(out_video.float())
                    audios.append(out_audio.float().transpose(1, 2))
            return [torch.cat(videos, dim=0), torch.cat(audios, dim=0)]

    _DENOISER_CLASS = MAGI2Denoiser
    return MAGI2Denoiser


def _make_denoiser(**kwargs):
    kwargs.pop("image_model", None)
    return _denoiser_class()(**kwargs)


# -----------------------------------------------------------------------------
# ComfyUI model class, latent format and config
# -----------------------------------------------------------------------------


class MAGI2Latent(comfy.latent_formats.Wan22):
    """LATENT dicts hold raw VAE-space latents, like every other ComfyUI model.
    The Wan2.2 mean/std normalisation applies to the video stream only, so it is
    done in the model class (the sampler hands over one packed tensor); here it
    is an identity. The preview colour factors are inherited."""

    def process_in(self, latent):
        return latent

    def process_out(self, latent):
        return latent


class MAGI2Preview(comfy.model_base.BaseModel):
    def __init__(self, model_config, model_type=comfy.model_base.ModelType.FLOW, device=None):
        super().__init__(model_config, model_type, device=device, unet_model=_make_denoiser)
        self._wan22 = comfy.latent_formats.Wan22()

    def load_model_weights(self, sd, unet_prefix="", assign=False):
        to_load = {}
        for key in list(sd.keys()):
            if key.startswith(unet_prefix):
                to_load[key[len(unet_prefix) :]] = sd.pop(key)
        # Always assign: the model was built without storage. Tensors stay backed by
        # the file's memory map until ComfyUI moves them to the GPU.
        missing, unexpected = self.diffusion_model.load_state_dict(to_load, strict=False, assign=True)
        if missing or unexpected:
            raise RuntimeError(
                f"MAGI-2 checkpoint does not match the model: {len(missing)} missing "
                f"(e.g. {missing[:3]}), {len(unexpected)} unexpected (e.g. {unexpected[:3]})"
            )
        leftover = [n for n, t in self.diffusion_model.state_dict().items() if t.device.type == "meta"]
        if leftover:
            raise RuntimeError(f"MAGI-2: {len(leftover)} tensors have no data after loading, e.g. {leftover[:3]}")
        return self

    # --- per-stream latent normalisation -------------------------------------

    def _map_video(self, latent, fn):
        if getattr(latent, "is_nested", False):
            streams = latent.unbind()
            return comfy.nested_tensor.NestedTensor([fn(streams[0])] + list(streams[1:]))
        shapes = self.latent_shapes
        if shapes is not None and len(shapes) > 1:
            streams = comfy.utils.unpack_latents(latent, shapes)
            streams[0] = fn(streams[0])
            return comfy.utils.pack_latents(streams)[0]
        if latent.ndim == 5:
            return fn(latent)
        return latent

    def process_latent_in(self, latent):
        return self._map_video(latent, self._wan22.process_in)

    def process_latent_out(self, latent):
        return self._map_video(latent, self._wan22.process_out)

    # --- conditioning ---------------------------------------------------------

    def extra_conds(self, **kwargs):
        out = super().extra_conds(**kwargs)
        cross_attn = kwargs.get("cross_attn", None)
        if cross_attn is not None:
            # CONDRegular: conds of different token counts are run separately, never
            # padded into one batch (MAGI-2 has no text attention mask here).
            out["c_crossattn"] = comfy.conds.CONDRegular(cross_attn)
        latent_shapes = kwargs.get("latent_shapes", None)
        if latent_shapes is not None and len(latent_shapes) > 1:
            out["latent_shapes"] = comfy.conds.CONDConstant(latent_shapes)
        return out

    def memory_required(self, input_shape, cond_shapes={}):
        # The forward runs one batch item at a time, so batch size does not scale it.
        tokens = math.prod(input_shape[1:]) / 48.0 + 1024
        return BASE_INFERENCE_BYTES + tokens * BYTES_PER_TOKEN


class MAGI2PreviewConfig(comfy.supported_models_base.BASE):
    unet_config = {"image_model": IMAGE_MODEL}
    unet_extra_config = {}
    sampling_settings = {"shift": 7.0, "multiplier": 1000}
    latent_format = MAGI2Latent
    supported_inference_dtypes = [torch.bfloat16]
    memory_usage_factor = 1.0
    vae_key_prefix = ["vae."]
    text_encoder_key_prefix = ["text_encoders."]

    def get_model(self, state_dict, prefix="", device=None):
        return MAGI2Preview(self, device=device)

    def clip_target(self, state_dict={}):
        return None


# -----------------------------------------------------------------------------
# Registration
# -----------------------------------------------------------------------------


def _is_magi2(state_dict, key_prefix):
    return all(f"{key_prefix}{k}" in state_dict for k in SIGNATURE_KEYS)


def register():
    """Idempotent: safe if the pack is imported twice."""
    current = comfy.model_detection.detect_unet_config
    if not getattr(current, "_magi2_wrapped", False):

        def detect_unet_config(state_dict, key_prefix, *args, **kwargs):
            if _is_magi2(state_dict, key_prefix):
                if not any(k.endswith(LAYOUT_SUFFIX) for k in state_dict.keys()):
                    raise RuntimeError(
                        "This is a MAGI-2 preview checkpoint, but not in the layout the ComfyUI model "
                        "class loads. Convert the NVFP4 checkpoint with the fork's "
                        "scripts/convert_comfy_checkpoint.py and load the file it writes."
                    )
                return {"image_model": IMAGE_MODEL}
            return current(state_dict, key_prefix, *args, **kwargs)

        detect_unet_config._magi2_wrapped = True
        comfy.model_detection.detect_unet_config = detect_unet_config
    if MAGI2PreviewConfig not in comfy.supported_models.models:
        comfy.supported_models.models.append(MAGI2PreviewConfig)
    logging.info("[MAGI-2] registered model type '%s' with ComfyUI", IMAGE_MODEL)
