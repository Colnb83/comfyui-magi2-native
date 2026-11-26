"""Nodes that go with the MAGI-2 ComfyUI model type (see magi2_comfy_model.py).

The denoiser is loaded by the stock Load Diffusion Model node and sampled by stock
nodes (SamplerCustomAdvanced, LTXVDualCFGGuider or CFGGuider). These nodes supply
what core has no equivalent for: the joint empty latent, text conditioning from
MAGI-2's own Qwen3.5 encoder, MAGI-2's sigma schedule and UniPC solver, and audio
decoding with MAGI-2's resampling step.
"""

import logging
import os
import time

import numpy as np
import torch
from tqdm.auto import trange

import comfy.model_management
import comfy.nested_tensor
import comfy.samplers

from .fork_env import prepare_fork_import

CLIP_SECONDS = 10.0
VIDEO_FPS = 12.5  # the 540p stage; frames = (latent frames - 1) * 4 + 1
AUDIO_LATENT_FPS = 25.0
SHIFT = 7.0
SIGMA_MAX = float(np.float32(0.999))


class MagiModelNodeError(RuntimeError):
    pass


def _ckpt_dir(value, sub):
    root = (value or "").strip() or os.environ.get("MAGI2_CKPT_ROOT", "")
    if not root or not os.path.isdir(root):
        raise MagiModelNodeError("ckpt_root is empty (or not a directory) and MAGI2_CKPT_ROOT is not set")
    path = root if os.path.basename(os.path.normpath(root)) == sub else os.path.join(root, sub)
    if not os.path.isdir(path):
        raise MagiModelNodeError(f"{root} has no {sub}/ directory")
    return os.path.realpath(path)


# -----------------------------------------------------------------------------
# Latent
# -----------------------------------------------------------------------------


class EmptyMAGI2Latent:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 896, "min": 256, "max": 4096, "step": 16}),
                "height": ("INT", {"default": 512, "min": 256, "max": 4096, "step": 16}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 16}),
            }
        }

    RETURN_TYPES = ("LATENT", "FLOAT")
    RETURN_NAMES = ("latent", "fps")
    FUNCTION = "generate"
    CATEGORY = "MAGI-2/model"
    DESCRIPTION = "Joint video+audio latent for a 10 second MAGI-2 clip (the only duration the model supports)."

    def generate(self, width, height, batch_size):
        frames = int(round(CLIP_SECONDS * VIDEO_FPS * 2))  # 250, counted at 25 fps
        video_frames = (frames - 1) // 8 + 1  # 32 latent frames -> 125 decoded frames
        audio_frames = int(round(CLIP_SECONDS * AUDIO_LATENT_FPS))  # 250
        device = comfy.model_management.intermediate_device()
        video = torch.zeros([batch_size, 48, video_frames, height // 16, width // 16], device=device)
        audio = torch.zeros([batch_size, 64, audio_frames], device=device)
        return ({"samples": comfy.nested_tensor.NestedTensor((video, audio))}, VIDEO_FPS)


# -----------------------------------------------------------------------------
# Text conditioning (MAGI-2's own encoder wrapper; runs on the CPU like the pipeline)
# -----------------------------------------------------------------------------

_TEXT_ENCODER = {"path": None, "encoder": None}


def _text_encoder(path):
    if _TEXT_ENCODER["encoder"] is None or _TEXT_ENCODER["path"] != path:
        prepare_fork_import()
        from inference.model.qwen35 import Qwen35TextEncoder

        _TEXT_ENCODER["encoder"] = None
        t0 = time.time()
        # skip_layer=2: hidden_states[-3], what MAGI-2 was trained on.
        _TEXT_ENCODER["encoder"] = Qwen35TextEncoder(model_path=path, device="cpu", precision=torch.bfloat16, skip_layer=2)
        _TEXT_ENCODER["path"] = path
        logging.info("[MAGI-2] text encoder loaded from %s in %.1f s (CPU)", path, time.time() - t0)
    return _TEXT_ENCODER["encoder"]


class MAGI2TextEncode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": "", "tooltip": "Plain text or MAGI-2's structured-caption JSON"}),
                "kind": (["prompt", "default negative prompt"], {"tooltip": "'default negative prompt' ignores the text and encodes MAGI-2's built-in negative prompt"}),
                "ckpt_root": ("STRING", {"default": "", "tooltip": "Directory holding text_encoder/ (or that directory itself). Empty = $MAGI2_CKPT_ROOT"}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"
    CATEGORY = "MAGI-2/model"

    def encode(self, text, kind, ckpt_root):
        encoder = _text_encoder(_ckpt_dir(ckpt_root, "text_encoder"))
        if kind == "default negative prompt":
            from inference.pipeline.inference_engine import NEGATIVE_PROMPT

            text = NEGATIVE_PROMPT
        elif not text.strip():
            raise MagiModelNodeError("text is empty")
        embedding = encoder.encode(text)
        return ([[embedding.float().cpu(), {}]],)


class MAGI2UnloadTextEncoder:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}, "optional": {"after": ("*", {})}}

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    RETURN_TYPES = ()
    FUNCTION = "unload"
    CATEGORY = "MAGI-2/model"
    OUTPUT_NODE = True
    DESCRIPTION = "Frees the 52 GB text encoder from host RAM."

    def unload(self, after=None):
        _TEXT_ENCODER["encoder"] = None
        _TEXT_ENCODER["path"] = None
        return ()


# -----------------------------------------------------------------------------
# Sigmas and sampler: MAGI-2's own schedule and UniPC solver
# -----------------------------------------------------------------------------


def magi2_sigmas(steps, shift):
    """FlowUniPCMultistepScheduler.set_timesteps: linspace(sigma_max, 0, steps + 1)[:-1],
    shifted, then a final zero. sigma_max is the scheduler's own float32 0.999."""
    sigmas = np.linspace(SIGMA_MAX, 0.0, steps + 1)[:-1]
    sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
    return torch.from_numpy(np.concatenate([sigmas, [0.0]]).astype(np.float32))


class MAGI2Sigmas:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "steps": ("INT", {"default": 100, "min": 1, "max": 1000}),
                "shift": ("FLOAT", {"default": SHIFT, "min": 0.1, "max": 100.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "get_sigmas"
    CATEGORY = "MAGI-2/model"

    def get_sigmas(self, steps, shift):
        return (magi2_sigmas(steps, shift),)


@torch.no_grad()
def sample_magi2_unipc(model, x, sigmas, extra_args=None, callback=None, disable=None):
    """MAGI-2's FlowUniPC multistep solver (the fork's scheduler class), fed the
    velocity recovered from ComfyUI's denoised prediction."""
    prepare_fork_import()
    from inference.pipeline.sampler import FlowUniPCMultistepScheduler

    extra_args = {} if extra_args is None else extra_args
    if float(sigmas[-1]) != 0.0:
        raise MagiModelNodeError("the MAGI-2 UniPC sampler needs a sigma schedule that ends at 0")
    scheduler = FlowUniPCMultistepScheduler()
    # shift=1.0: the sigmas passed in are final, do not shift them again.
    scheduler.set_timesteps(sigmas=sigmas[:-1].detach().cpu().double().numpy(), device=x.device, shift=1.0)
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        sigma = sigmas[i]
        denoised = model(x, sigma * s_in, **extra_args)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigma, "sigma_hat": sigma, "denoised": denoised})
        velocity = (x - denoised) / sigma
        x = scheduler.step(velocity, scheduler.timesteps[i], x, return_dict=False)[0]
    return x


class MAGI2SamplerUniPC:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ("SAMPLER",)
    FUNCTION = "get_sampler"
    CATEGORY = "MAGI-2/model"

    def get_sampler(self):
        return (comfy.samplers.KSAMPLER(sample_magi2_unipc),)


# -----------------------------------------------------------------------------
# Audio decode: Stable Audio Open VAE + MAGI-2's 441/512 resample
# -----------------------------------------------------------------------------

_AUDIO_VAE = {"path": None, "vae": None}


class MAGI2AudioDecode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT", {"tooltip": "The joint MAGI-2 latent (or its audio stream)"}),
                "ckpt_root": ("STRING", {"default": "", "tooltip": "Directory holding stable-audio-open-1.0/. Empty = $MAGI2_CKPT_ROOT"}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "decode"
    CATEGORY = "MAGI-2/model"

    def decode(self, samples, ckpt_root):
        latent = samples["samples"]
        if getattr(latent, "is_nested", False):
            latent = latent.unbind()[-1]
        if latent.ndim != 3 or latent.shape[1] != 64:
            raise MagiModelNodeError(f"expected an audio latent [B, 64, L], got {tuple(latent.shape)}")
        path = _ckpt_dir(ckpt_root, "stable-audio-open-1.0")
        prepare_fork_import()
        from inference.pipeline.audio_decoder import SAAudioFeatureExtractor, resample_audio_sinc

        if _AUDIO_VAE["vae"] is None or _AUDIO_VAE["path"] != path:
            _AUDIO_VAE["vae"] = SAAudioFeatureExtractor(path)
            _AUDIO_VAE["path"] = path
        vae = _AUDIO_VAE["vae"]
        waves = []
        for i in range(latent.shape[0]):
            # Same steps as the pipeline's decode_audio: decode, then stretch by 441/512.
            waveform = vae.decode(latent[i : i + 1].to(device=vae.device, dtype=torch.float32))
            audio = waveform.squeeze(0).T.float().cpu().numpy()  # (samples, channels)
            audio = resample_audio_sinc(audio, 441 / 512)
            waves.append(torch.from_numpy(np.ascontiguousarray(audio.T.astype(np.float32))))
        return ({"waveform": torch.stack(waves, dim=0), "sample_rate": int(vae.sample_rate)},)


NODE_CLASS_MAPPINGS = {
    "EmptyMAGI2Latent": EmptyMAGI2Latent,
    "MAGI2TextEncode": MAGI2TextEncode,
    "MAGI2UnloadTextEncoder": MAGI2UnloadTextEncoder,
    "MAGI2Sigmas": MAGI2Sigmas,
    "MAGI2SamplerUniPC": MAGI2SamplerUniPC,
    "MAGI2AudioDecode": MAGI2AudioDecode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EmptyMAGI2Latent": "Empty MAGI-2 Latent (video+audio)",
    "MAGI2TextEncode": "MAGI-2 Text Encode",
    "MAGI2UnloadTextEncoder": "MAGI-2 Unload Text Encoder",
    "MAGI2Sigmas": "MAGI-2 Sigmas",
    "MAGI2SamplerUniPC": "MAGI-2 Sampler (UniPC)",
    "MAGI2AudioDecode": "MAGI-2 Audio Decode",
}
