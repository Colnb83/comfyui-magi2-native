"""Nodes that go with the MAGI-2 ComfyUI model type (see magi2_comfy_model.py).

The denoiser is loaded by the stock Load Diffusion Model node and sampled by stock
nodes (SamplerCustomAdvanced, LTXVDualCFGGuider or CFGGuider). These nodes supply
what core has no equivalent for: the joint empty latent, text conditioning from
MAGI-2's own Qwen3.5 encoder, MAGI-2's sigma schedule and UniPC solver, and audio
decoding with MAGI-2's resampling step.
"""

import logging
import math
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


class MAGI2RefinerLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT", {"tooltip": "The sampled 540p MAGI-2 latent (joint, or its video stream)"}),
                "width": ("INT", {"default": 1920, "min": 256, "max": 4096, "step": 16}),
                "height": ("INT", {"default": 1088, "min": 256, "max": 4096, "step": 16}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "noise_level": ("INT", {"default": 220, "min": 0, "max": 999, "tooltip": "MAGI-2's refiner noise index (220 in the released configuration); 0 = no noise"}),
            }
        }

    RETURN_TYPES = ("LATENT", "FLOAT")
    RETURN_NAMES = ("latent", "fps")
    FUNCTION = "prepare"
    CATEGORY = "MAGI-2/model"
    DESCRIPTION = ("Start latent for the MAGI-2 refiner: the 540p video latent upscaled to the target size with "
                   "twice the frames, mixed with noise by MAGI-2's own rule. Sample it with the refiner model, "
                   "noise disabled.")

    def prepare(self, samples, width, height, seed, noise_level):
        import comfy.latent_formats

        latent = samples["samples"]
        if getattr(latent, "is_nested", False):
            latent = latent.unbind()[0]
        if latent.ndim != 5 or latent.shape[1] != 48:
            raise MagiModelNodeError(f"expected a video latent [B, 48, T, H, W], got {tuple(latent.shape)}")
        wan = comfy.latent_formats.Wan22()
        video = wan.process_in(latent.float())  # the model's normalised space, where the pipeline does this
        video = torch.nn.functional.interpolate(
            video, size=(video.shape[2] * 2 - 1, height // 16, width // 16), mode="trilinear", align_corners=True
        )
        if noise_level != 0:
            prepare_fork_import()
            from inference.pipeline.inference_engine import ZeroSNRDDPMDiscretization

            sigma = ZeroSNRDDPMDiscretization()(1000, do_append_zero=False, flip=True)[noise_level].to(video)
            noise = torch.randn(video.shape, generator=torch.Generator(device="cpu").manual_seed(seed), dtype=torch.float32).to(video)
            video = video * sigma + noise * (1 - sigma**2) ** 0.5
        return ({"samples": wan.process_out(video)}, VIDEO_FPS * 2)


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
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "Image-to-video: the image the clip starts from. Needs vae."}),
                "vae": ("VAE", {"tooltip": "The Wan2.2 VAE, to encode the image"}),
                "width": ("INT", {"default": 896, "min": 256, "max": 4096, "step": 16, "tooltip": "Clip size, used to scale the image"}),
                "height": ("INT", {"default": 512, "min": 256, "max": 4096, "step": 16}),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"
    CATEGORY = "MAGI-2/model"

    @staticmethod
    def _image_latent(image, vae, width, height):
        """MAGI-2's image preparation: long edge scaled to the clip's long edge, sides
        floored to a multiple of 32, then the VAE latent in the model's normalised space."""
        import comfy.latent_formats
        from PIL import Image

        from inference.pipeline.inference_engine import resizepad

        pixels = image[0, ..., :3].detach().clamp(0, 1).mul(255).round().to(device="cpu", dtype=torch.uint8).numpy()
        pil = Image.fromarray(pixels, mode="RGB")
        long_edge = max(height, width)
        if pil.width > pil.height:
            target_w, target_h = long_edge, int(pil.height * long_edge / pil.width)
        else:
            target_h, target_w = long_edge, int(pil.width * long_edge / pil.height)
        pil = resizepad(pil, target_h, target_w)
        target_w, target_h = target_w - target_w % 32, target_h - target_h % 32
        pil = pil.resize((target_w, target_h), resample=Image.LANCZOS)
        frame = torch.from_numpy(np.array(pil).astype(np.float32) / 255.0).unsqueeze(0)  # [1, H, W, 3]
        raw = vae.encode(frame).float()  # [1, 48, 1, h, w], raw VAE space
        feat = comfy.latent_formats.Wan22().process_in(raw).cpu()
        feat_len = torch.tensor([[[feat.shape[3], feat.shape[4]]]], dtype=torch.long)
        return feat.unsqueeze(1), feat_len  # [1, 1, 48, 1, h, w]

    def encode(self, text, kind, ckpt_root, image=None, vae=None, width=896, height=512):
        encoder = _text_encoder(_ckpt_dir(ckpt_root, "text_encoder"))
        if kind == "default negative prompt":
            from inference.pipeline.inference_engine import NEGATIVE_PROMPT

            return ([[encoder.encode(NEGATIVE_PROMPT).float().cpu(), {}]],)
        if not text.strip():
            raise MagiModelNodeError("text is empty")
        if image is None:
            return ([[encoder.encode(text).float().cpu(), {}]],)
        if vae is None:
            raise MagiModelNodeError("image-to-video needs the Wan2.2 VAE connected to 'vae'")
        from inference.pipeline.inference_engine import Magi2InferenceEngine

        # The prompt must mention the image; its embedding over that mention labels the image tokens.
        text = Magi2InferenceEngine._ensure_figure_tokens(text, [1])
        embedding = encoder.encode(text)
        special = encoder.get_special_token(text, ["<Figure 1>"], embedding)  # [1, 5120]
        feat, feat_len = self._image_latent(image, vae, width, height)
        ref = {"feat": feat, "feat_len": feat_len, "special": special.float().cpu().unsqueeze(0)}
        return ([[embedding.float().cpu(), {"magi2_ref_image": ref}]],)


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
# Guider: separate video and audio guidance, with MAGI-2's optional CFG rescale
# -----------------------------------------------------------------------------


class Guider_MAGI2(comfy.samplers.CFGGuider):
    """Video and audio guidance scales as in MAGI-2's sampler, plus its ``cfg_rescale``:
    the guided prediction of each stream is scaled so its standard deviation moves
    towards that of the conditional prediction (0 = off, 1 = fully matched)."""

    def set_scales(self, video_cfg, audio_cfg, cfg_rescale):
        self.video_cfg, self.audio_cfg, self.cfg_rescale = video_cfg, audio_cfg, cfg_rescale
        self.cfg = max(video_cfg, audio_cfg)

    def sample(self, noise, latent_image, *args, **kwargs):
        self._shapes = None
        if getattr(latent_image, "is_nested", False):
            self._shapes = [tuple(p.shape) for p in latent_image.unbind()]
        return super().sample(noise, latent_image, *args, **kwargs)

    @staticmethod
    def _rescale(cond, guided, dims, amount):
        cond_std = cond.float().std(dim=dims, keepdim=True, unbiased=False)
        guided_std = guided.float().std(dim=dims, keepdim=True, unbiased=False).clamp_min(1e-6)
        return guided * (amount * (cond_std / guided_std) + (1 - amount)).to(guided.dtype)

    def predict_noise(self, x, timestep, model_options={}, seed=None):
        shapes = getattr(self, "_shapes", None)
        video_cfg, audio_cfg, amount = self.video_cfg, self.audio_cfg, self.cfg_rescale

        def guide(args):
            # cond / uncond are x - prediction (sigma * velocity): the same linear
            # combination and the same std ratios as in velocity space.
            cond, uncond = args["cond"], args["uncond"]
            if shapes is None or len(shapes) < 2:  # single stream (e.g. the refiner)
                out = uncond + (cond - uncond) * video_cfg
                if amount > 0 and out.ndim >= 4:
                    out = self._rescale(cond, out, tuple(range(2, out.ndim)), amount)
                return out
            batch = cond.shape[0]
            v = math.prod(shapes[0][1:])
            out = uncond + (cond - uncond) * video_cfg
            out[..., v:] = uncond[..., v:] + (cond[..., v:] - uncond[..., v:]) * audio_cfg
            if amount > 0:
                vshape, ashape = (batch,) + shapes[0][1:], (batch,) + shapes[1][1:]
                video = self._rescale(cond[..., :v].reshape(vshape), out[..., :v].reshape(vshape), (2, 3, 4), amount)
                # audio is [B, 64, L] here; MAGI-2 takes the std over time per channel
                audio = self._rescale(cond[..., v:].reshape(ashape), out[..., v:].reshape(ashape), (2,), amount)
                out = torch.cat([video.reshape(batch, 1, -1), audio.reshape(batch, 1, -1)], dim=-1)
            return out

        model_options = {**model_options, "sampler_cfg_function": guide, "disable_cfg1_optimization": True}
        return super().predict_noise(x, timestep, model_options, seed)


class MAGI2Guider:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "video_cfg": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "audio_cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "cfg_rescale": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "0 = off (MAGI-2's default)"}),
            }
        }

    RETURN_TYPES = ("GUIDER",)
    FUNCTION = "get_guider"
    CATEGORY = "MAGI-2/model"

    def get_guider(self, model, positive, negative, video_cfg, audio_cfg, cfg_rescale):
        # Image-to-video: MAGI-2 shows the image to the unconditional branch too, so
        # guidance steers the text alone.
        ref = next((c[1]["magi2_ref_image"] for c in positive if "magi2_ref_image" in c[1]), None)
        if ref is not None:
            negative = [[c[0], {**c[1], "magi2_ref_image": c[1].get("magi2_ref_image", ref)}] for c in negative]
        guider = Guider_MAGI2(model)
        guider.set_conds(positive, negative)
        guider.set_scales(video_cfg, audio_cfg, cfg_rescale)
        return (guider,)


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
    "MAGI2RefinerLatent": MAGI2RefinerLatent,
    "MAGI2TextEncode": MAGI2TextEncode,
    "MAGI2UnloadTextEncoder": MAGI2UnloadTextEncoder,
    "MAGI2Guider": MAGI2Guider,
    "MAGI2Sigmas": MAGI2Sigmas,
    "MAGI2SamplerUniPC": MAGI2SamplerUniPC,
    "MAGI2AudioDecode": MAGI2AudioDecode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EmptyMAGI2Latent": "Empty MAGI-2 Latent (video+audio)",
    "MAGI2RefinerLatent": "MAGI-2 Refiner Latent",
    "MAGI2TextEncode": "MAGI-2 Text Encode",
    "MAGI2UnloadTextEncoder": "MAGI-2 Unload Text Encoder",
    "MAGI2Guider": "MAGI-2 Guider (video/audio CFG, rescale)",
    "MAGI2Sigmas": "MAGI-2 Sigmas",
    "MAGI2SamplerUniPC": "MAGI-2 Sampler (UniPC)",
    "MAGI2AudioDecode": "MAGI-2 Audio Decode",
}
