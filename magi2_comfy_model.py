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
import comfy.model_patcher
import comfy.nested_tensor
import comfy.sd
import comfy.supported_models
import comfy.supported_models_base
import comfy.utils

from .fork_env import image_numerics, prepare_fork_import

IMAGE_MODEL = "magi2_preview"
REFINER_IMAGE_MODEL = "magi2_refiner"
REFINER_SIGNATURE_KEYS = (
    "block.layers.0.mlp.up_gate_proj.weight",
    "pre_adapter.video_embedder.weight",
    "pre_adapter.audio_embedder.weight",
    "post_adapter.final_linear_video.weight",
)
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
# Loading and unloading: all on the GPU or all on disk
# -----------------------------------------------------------------------------


class FileBackedMixin:
    """For a module whose weights were assigned from a checkpoint's memory map.

    Moving it to the GPU copies the weights there as usual. Moving it back to
    the CPU does not copy 67 GiB into host RAM: the tensors are simply pointed
    at the memory map again."""

    _magi2_source = None

    def _apply(self, fn, recurse=True):
        source = self._magi2_source
        if source is not None:
            target = fn(torch.empty(0)).device
            current = next(iter(self.parameters())).device
            if target.type == "cpu" and current.type != "cpu":
                self.load_state_dict(source, strict=False, assign=True)
                return self
        return super()._apply(fn, recurse)


class MAGI2ModelPatcher(comfy.model_patcher.ModelPatcher):
    """MAGI-2's modules are plain torch modules running custom kernels: they work
    fully on the GPU or not at all. This patcher never loads or unloads a part."""

    def load(self, device_to=None, lowvram_model_memory=0, force_patch_weights=False, full_load=False):
        # The weights need most of the card in one go. Hand PyTorch's cached blocks
        # (left by a previous load or by sampling) back first, or the allocator can
        # fail to fit them although enough memory is free in total.
        comfy.model_management.soft_empty_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return super().load(device_to, lowvram_model_memory=0, force_patch_weights=force_patch_weights, full_load=True)

    def partially_unload(self, device_to, memory_to_free=0, force_patch_weights=False):
        return 0  # nothing freed in part; the caller then unloads the whole model

    def add_patches(self, patches, strength_patch=1.0, strength_model=1.0):
        # LoRA and weight-diff patches apply to the floating-point weights (attention,
        # MLP linears, router, norms). The NVFP4 expert tensors are packed 4-bit codes:
        # a delta cannot be added to them, so such patches are dropped with a warning.
        state = self.model.state_dict()
        usable, skipped = {}, []
        for key, value in patches.items():
            name = key[0] if isinstance(key, tuple) else key
            target = state.get(name)
            if target is not None and not target.is_floating_point():
                skipped.append(name)
            else:
                usable[key] = value
        if skipped:
            logging.warning(
                "[MAGI-2] %d patch(es) target NVFP4 expert tensors and were not applied, e.g. %s",
                len(skipped), skipped[0],
            )
        return super().add_patches(usable, strength_patch, strength_model)


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

    class MAGI2Denoiser(FileBackedMixin, Transformer):
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

        def _single_input(self, video, audio, context, t_norm, ref=None):
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
                # Conditioning image(s): clean VAE latents packed as extra tokens, each
                # prefixed by the text embedding pooled over its <Figure N> mention.
                ref_image_feat=None if ref is None else ref["feat"].to(device=device, dtype=torch.float32),
                ref_image_feat_len=None if ref is None else ref["feat_len"].to(device=device),
                ref_image_special_token_embedding=None if ref is None else ref["special"].to(device=device, dtype=torch.bfloat16),
            )

        def forward(self, x, timestep, context=None, transformer_options={}, magi2_ref=None, **kwargs):
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
                        ref=magi2_ref,
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
# Refiner (the 1080p stage): video-only latent, no time input
# -----------------------------------------------------------------------------

_REFINER_CLASS = None


def _refiner_class():
    global _REFINER_CLASS
    if _REFINER_CLASS is not None:
        return _REFINER_CLASS

    fork = prepare_fork_import()
    from inference.common.magi2_config import load_config
    from inference.model.magi2_refiner import Transformer as RefinerTransformer
    from inference.pipeline.inference_engine import EvalInput
    from inference.pipeline.refiner_data_proxy import Magi2RefinerDataProxy

    class MAGI2RefinerDenoiser(FileBackedMixin, RefinerTransformer):
        def __init__(self, dtype=None, device=None, operations=None, **kwargs):
            config = load_config(os.path.join(fork, "configs", "local", "1gpu_refiner.json"))
            with torch.device("meta"):
                super().__init__(config.magi2_refiner_arch_config)
            self.data_proxy = Magi2RefinerDataProxy(config.evaluation_config.magi2_refiner_data_proxy_config)
            self.dtype = torch.bfloat16

        def forward(self, x, timestep, context=None, transformer_options={}, **kwargs):
            if isinstance(x, (list, tuple)):
                raise RuntimeError(
                    "The MAGI-2 refiner takes a video-only latent (use the MAGI-2 Refiner Latent node), "
                    "not the joint video+audio latent"
                )
            if context is None:
                raise RuntimeError("MAGI-2 refiner needs text conditioning (use the MAGI-2 Text Encode node)")
            device = x.device
            probe = self.pre_adapter.video_embedder.weight
            if probe.device != device or probe.device.type != "cuda":
                raise RuntimeError(
                    f"The MAGI-2 refiner must be fully loaded on the GPU to run (weights are on {probe.device}, "
                    f"latents on {device}). At 1080p it needs about 55 GiB of VRAM."
                )
            videos = []
            with image_numerics():
                for i in range(x.shape[0]):
                    video = x[i : i + 1].float()
                    text = context[i : i + 1].float()
                    # No audio tokens: the shipping configuration refines the video only.
                    empty_audio = torch.zeros(1, 0, 64, dtype=torch.float32, device=device)
                    eval_input = EvalInput(
                        x_t=video,
                        audio_x_t=empty_audio,
                        audio_feat_len=torch.tensor([0]),
                        txt_feat=text,
                        txt_feat_len=torch.tensor([text.shape[1]]),
                        ref_audio_feat=empty_audio,
                        ref_audio_feat_len=torch.tensor([0]),
                        ref_video_feat=torch.empty_like(video),
                        ref_video_feat_len=torch.tensor([0]),
                    )
                    packed = self.data_proxy.process_input(eval_input)
                    output = RefinerTransformer.forward(self, *packed)
                    del packed
                    out_video, _ = self.data_proxy.process_output(output)
                    del output
                    videos.append(out_video.float().reshape(video.shape))
            return torch.cat(videos, dim=0)

    _REFINER_CLASS = MAGI2RefinerDenoiser
    return MAGI2RefinerDenoiser


def _make_refiner(**kwargs):
    kwargs.pop("image_model", None)
    return _refiner_class()(**kwargs)


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
        self.diffusion_model._magi2_source = to_load  # file-backed tensors, for unloading without a RAM copy
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
        ref = kwargs.get("magi2_ref_image", None)
        if ref is not None:
            out["magi2_ref"] = comfy.conds.CONDConstant(ref)  # a dict: left alone by the dtype cast
        return out

    def memory_required(self, input_shape, cond_shapes={}):
        # The forward runs one batch item at a time, so batch size does not scale it.
        tokens = math.prod(input_shape[1:]) / 48.0 + 1024
        return BASE_INFERENCE_BYTES + tokens * BYTES_PER_TOKEN


class MAGI2Refiner(comfy.model_base.BaseModel):
    """The 1080p refiner. MAGI-2 starts it from the upscaled 540p latent mixed with
    noise by its own rule (see the MAGI-2 Refiner Latent node) and then runs a full
    flow schedule over it; the model has no time input. So the sampler must not
    re-noise or rescale the latent it is given."""

    def __init__(self, model_config, model_type=comfy.model_base.ModelType.FLOW, device=None):
        super().__init__(model_config, model_type, device=device, unet_model=_make_refiner)
        sampling = self.model_sampling

        def noise_scaling(sigma, noise, latent_image, max_denoise=False):
            return latent_image

        def inverse_noise_scaling(sigma, latent):
            return latent

        sampling.noise_scaling = noise_scaling
        sampling.inverse_noise_scaling = inverse_noise_scaling

    load_model_weights = MAGI2Preview.load_model_weights

    def extra_conds(self, **kwargs):
        out = super().extra_conds(**kwargs)
        cross_attn = kwargs.get("cross_attn", None)
        if cross_attn is not None:
            out["c_crossattn"] = comfy.conds.CONDRegular(cross_attn)
        return out

    def memory_required(self, input_shape, cond_shapes={}):
        # Low-memory forward, measured: 54.8 GiB peak with 12.6 GiB of weights at
        # 514k tokens (1920x1088, 63 latent frames).
        tokens = math.prod(input_shape[2:]) + 1024
        return 2 * 2**30 + tokens * 80e3


class MAGI2RefinerConfig(comfy.supported_models_base.BASE):
    unet_config = {"image_model": REFINER_IMAGE_MODEL}
    unet_extra_config = {}
    sampling_settings = {"shift": 7.0, "multiplier": 1000}
    latent_format = comfy.latent_formats.Wan22
    supported_inference_dtypes = [torch.bfloat16]
    memory_usage_factor = 1.0
    vae_key_prefix = ["vae."]
    text_encoder_key_prefix = ["text_encoders."]

    def get_model(self, state_dict, prefix="", device=None):
        return MAGI2Refiner(self, device=device)

    def clip_target(self, state_dict={}):
        return None


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
            if all(f"{key_prefix}{k}" in state_dict for k in REFINER_SIGNATURE_KEYS):
                return {"image_model": REFINER_IMAGE_MODEL}
            return current(state_dict, key_prefix, *args, **kwargs)

        detect_unet_config._magi2_wrapped = True
        comfy.model_detection.detect_unet_config = detect_unet_config
    loader = comfy.sd.load_diffusion_model_state_dict
    if not getattr(loader, "_magi2_wrapped", False):

        def load_diffusion_model_state_dict(sd, *args, **kwargs):
            magi = _is_magi2(sd, "") or all(k in sd for k in REFINER_SIGNATURE_KEYS)
            if magi:
                # ComfyUI's dynamic-VRAM patcher assumes other models give memory back
                # on demand; these cannot, so they use the classic load/unload path.
                if len(args) >= 3:
                    args = args[:2] + (True,) + args[3:]
                else:
                    kwargs["disable_dynamic"] = True
            patcher = loader(sd, *args, **kwargs)
            if magi and patcher is not None and isinstance(patcher.model, (MAGI2Preview, MAGI2Refiner)):
                patcher.__class__ = MAGI2ModelPatcher
            return patcher

        load_diffusion_model_state_dict._magi2_wrapped = True
        comfy.sd.load_diffusion_model_state_dict = load_diffusion_model_state_dict
    for config in (MAGI2PreviewConfig, MAGI2RefinerConfig):
        if config not in comfy.supported_models.models:
            comfy.supported_models.models.append(config)
    logging.info("[MAGI-2] registered model types '%s' and '%s' with ComfyUI", IMAGE_MODEL, REFINER_IMAGE_MODEL)
