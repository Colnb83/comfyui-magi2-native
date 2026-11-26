"""MAGI-2 inside ComfyUI's own process, on one GPU.

The nodes import the MAGI-2 fork's ``inference`` package (``magi2_fork/`` next to
this file, or ``MAGI2_FORK_DIR``) and run its own pipeline: text encoder, preview
denoiser (NVFP4 experts, Triton MoE kernel), optional refiner, VAE and audio
decode. No Docker, no HTTP service, no torch.distributed.

The fork reads its switches from the environment at import time, so they are
set here before the first import. A value already present in the environment
wins, which lets a launcher override any of them.
"""

import gc
import logging
import os
import sys
import time

import numpy as np
import torch

import comfy.model_management
import comfy.utils

from .fork_env import ENV_DEFAULTS, fork_dir as _fork_dir, image_numerics as _image_numerics

CLIP_SECONDS = 10.0

MODES = {
    # mode -> (config relative to the fork, resolutions it can run)
    "540p": ("configs/local/1gpu_preview.json", ["540p", "272p"]),
    "1080p": ("configs/local/1gpu_refiner.json", ["1080p", "540p", "272p"]),
}

_STATE = {"pipeline": None, "key": None}


class MagiNativeError(RuntimeError):
    pass


def _resolve_dirs(ckpt_root, nvfp4_ckpt):
    ckpt_root = (ckpt_root or "").strip() or os.environ.get("MAGI2_CKPT_ROOT", "")
    nvfp4_ckpt = (nvfp4_ckpt or "").strip() or os.environ.get("MAGI2_NVFP4_CKPT", "")
    if not ckpt_root:
        raise MagiNativeError("ckpt_root is empty and MAGI2_CKPT_ROOT is not set")
    for sub in ("text_encoder", "vae", "turbo_vae", "stable-audio-open-1.0"):
        if not os.path.isdir(os.path.join(ckpt_root, sub)):
            raise MagiNativeError(f"ckpt_root {ckpt_root} has no {sub}/ directory")
    if not nvfp4_ckpt:
        raise MagiNativeError("nvfp4_ckpt is empty and MAGI2_NVFP4_CKPT is not set")
    if not (
        os.path.isfile(os.path.join(nvfp4_ckpt, "preview", "nvfp4_manifest.json"))
        or os.path.isfile(os.path.join(nvfp4_ckpt, "nvfp4_manifest.json"))
    ):
        raise MagiNativeError(
            f"nvfp4_ckpt {nvfp4_ckpt} is not a pre-quantised NVFP4 checkpoint (no nvfp4_manifest.json); "
            "the FP8 experts do not fit on one 96 GB card"
        )
    return os.path.realpath(ckpt_root), os.path.realpath(nvfp4_ckpt)


def _free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def unload_pipeline():
    had = _STATE["pipeline"] is not None
    _STATE["pipeline"] = None
    _STATE["key"] = None
    _free_gpu()
    return had


def load_pipeline(ckpt_root, nvfp4_ckpt, mode):
    fork = _fork_dir()
    ckpt_root, nvfp4_ckpt = _resolve_dirs(ckpt_root, nvfp4_ckpt)
    key = (fork, ckpt_root, nvfp4_ckpt, mode)
    if _STATE["pipeline"] is not None and _STATE["key"] == key:
        return _STATE["pipeline"]
    unload_pipeline()

    device = comfy.model_management.get_torch_device()
    if device.type != "cuda":
        raise MagiNativeError(f"MAGI-2 needs a CUDA device, ComfyUI is using {device}")
    # The MAGI-2 models are not ComfyUI ModelPatchers, so ComfyUI cannot evict
    # them or make room for them: clear its own models off the card first.
    comfy.model_management.unload_all_models()
    comfy.model_management.soft_empty_cache()

    for name, value in ENV_DEFAULTS.items():
        os.environ.setdefault(name, value)
    os.environ["MAGI2_CKPT_ROOT"] = ckpt_root
    os.environ["MAGI2_NVFP4_CKPT"] = nvfp4_ckpt
    if fork not in sys.path:
        sys.path.insert(0, fork)

    t0 = time.time()
    with _image_numerics():
        torch.cuda.set_device(device.index or 0)
        from inference.pipeline.entry import build_config
        from inference.pipeline.pipeline import Magi2Pipeline

        config = build_config(os.path.join(fork, MODES[mode][0]), seed=0)
        if config.engine_config.cp_size != 1 or config.engine_config.ep_size != 1:
            raise MagiNativeError("the native node needs a cp=1 / ep=1 config")
        pipeline = Magi2Pipeline(config)
    pipeline._magi2_native_mode = mode
    _STATE["pipeline"] = pipeline
    _STATE["key"] = key
    logging.info(
        "[MAGI-2 native] loaded (%s) in %.1f s, %.1f GiB allocated on %s",
        mode, time.time() - t0, torch.cuda.memory_allocated(device) / 2**30, device,
    )
    return pipeline


def _to_pil(image):
    if image is None or len(image) == 0:
        return None
    from PIL import Image

    pixels = image[0, ..., :3].detach().clamp(0, 1).mul(255).round().to(device="cpu", dtype=torch.uint8).numpy()
    return Image.fromarray(pixels, mode="RGB")


def _audio_output(audio_np, sample_rate):
    if audio_np is None:
        logging.warning("[MAGI-2 native] no audio decoded; returning one second of silence")
        return {"waveform": torch.zeros((1, 1, 44100), dtype=torch.float32), "sample_rate": 44100}
    array = np.asarray(audio_np, dtype=np.float32)
    if array.ndim == 1:
        array = array[None, :]
    elif array.ndim == 2 and array.shape[0] > array.shape[1]:
        array = array.T  # (samples, channels), as soundfile wants it -> (channels, samples)
    elif array.ndim != 2:
        raise MagiNativeError(f"unexpected decoded audio shape {array.shape}")
    return {"waveform": torch.from_numpy(np.ascontiguousarray(array)).unsqueeze(0), "sample_rate": int(sample_rate)}


def generate(pipeline, prompt, seed, steps, resolution, refiner_steps, image, video_guidance, audio_guidance, cfg_rescale=0.0):
    from inference.pipeline.entry import RESOLUTION_PRESETS
    from inference.prompt_enhancement import enhance_prompt

    mode = getattr(pipeline, "_magi2_native_mode", "540p")
    if resolution not in MODES[mode][1]:
        raise MagiNativeError(f"resolution {resolution} needs the loader in 1080p mode (it is in {mode} mode)")
    if not prompt or not prompt.strip():
        raise MagiNativeError("prompt is empty")
    preset = RESOLUTION_PRESETS[resolution]
    engine = pipeline.evaluator.inference_engine
    config = pipeline.evaluator.config
    refiner_width, refiner_height = preset.get("rw"), preset.get("rh")
    if refiner_width:
        refiner_steps = refiner_steps if refiner_steps and refiner_steps > 0 else config.magi2_refiner_num_inference_steps
    else:
        refiner_steps = None

    bar = comfy.utils.ProgressBar(steps)
    peak = {"alloc": 0, "reserved": 0}

    def sample_peak():
        # The engine resets the CUDA peak counters between stages, so keep our own maximum.
        peak["alloc"] = max(peak["alloc"], torch.cuda.max_memory_allocated())
        peak["reserved"] = max(peak["reserved"], torch.cuda.max_memory_reserved())

    def on_progress(stage, step, total):
        sample_peak()
        comfy.model_management.throw_exception_if_processing_interrupted()
        if stage == "denoise" and total:
            bar.update_absolute(step, total=total)

    cfg = engine.cfg_config
    saved = (cfg.video_txt_guidance_scale, cfg.audio_txt_guidance_scale)
    saved_rescale = cfg.cfg_rescale
    engine.progress_callback = on_progress
    pil_image = _to_pil(image)
    t0 = time.time()
    try:
        if video_guidance is not None and video_guidance >= 0:
            cfg.video_txt_guidance_scale = float(video_guidance)
        if audio_guidance is not None and audio_guidance >= 0:
            cfg.audio_txt_guidance_scale = float(audio_guidance)
        cfg.cfg_rescale = float(cfg_rescale or 0.0)
        with _image_numerics():
            torch.cuda.reset_peak_memory_stats()
            seed32 = int(seed) % (2**63)
            torch.manual_seed(seed32)
            torch.cuda.manual_seed_all(seed32)
            # Same order as the fork's own entry point: optional prompt
            # enhancement (a no-op without API_KEY), then the engine.
            prompt = enhance_prompt(prompt, pil_image)
            video_np, audio_np = engine.evaluate(
                prompt=prompt,
                image=pil_image,
                seconds=CLIP_SECONDS,
                preview_width=preset["pw"],
                preview_height=preset["ph"],
                br_num_inference_steps=int(steps),
                refiner_width=refiner_width,
                refiner_height=refiner_height,
                magi2_refiner_num_inference_steps=refiner_steps,
            )
    finally:
        sample_peak()
        engine.progress_callback = None
        cfg.video_txt_guidance_scale, cfg.audio_txt_guidance_scale = saved
        cfg.cfg_rescale = saved_rescale
    if video_np is None:
        raise MagiNativeError("MAGI-2 pipeline returned no video")
    if video_np.dtype != np.uint8 or video_np.ndim != 4 or video_np.shape[3] != 3:
        raise MagiNativeError(f"unexpected decoded video {video_np.dtype} {video_np.shape}")
    fps = float(config.fps) * (2 if refiner_width else 1)
    frames = torch.from_numpy(np.ascontiguousarray(video_np)).float().div_(255.0)
    audio = _audio_output(audio_np, engine.audio_vae.sample_rate if audio_np is not None else None)
    logging.info(
        "[MAGI-2 native] %s, %d steps: %.1f s, peak %.2f GiB allocated / %.2f GiB reserved, %d frames %dx%d @ %.1f fps",
        resolution, steps, time.time() - t0, peak["alloc"] / 2**30,
        peak["reserved"] / 2**30, frames.shape[0], frames.shape[2], frames.shape[1], fps,
    )
    return frames, audio, fps


class MAGI2NativeLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_root": ("STRING", {"default": "", "tooltip": "Directory holding text_encoder/, vae/, turbo_vae/, stable-audio-open-1.0/ (and refiner/). Empty = $MAGI2_CKPT_ROOT"}),
                "nvfp4_ckpt": ("STRING", {"default": "", "tooltip": "Pre-quantised NVFP4 checkpoint directory (holding preview/). Empty = $MAGI2_NVFP4_CKPT"}),
                "mode": (list(MODES), {"tooltip": "1080p also loads the refiner"}),
            }
        }

    RETURN_TYPES = ("MAGI2_PIPELINE",)
    RETURN_NAMES = ("pipeline",)
    FUNCTION = "load"
    CATEGORY = "MAGI-2"

    @classmethod
    def IS_CHANGED(cls, ckpt_root, nvfp4_ckpt, mode):
        # Re-run after an unload so the pipeline object is never stale.
        return float("nan") if _STATE["pipeline"] is None else repr(_STATE["key"])

    def load(self, ckpt_root, nvfp4_ckpt, mode):
        return (load_pipeline(ckpt_root, nvfp4_ckpt, mode),)


class MAGI2NativeGenerate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("MAGI2_PIPELINE",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps": ("INT", {"default": 100, "min": 1, "max": 1000}),
                "resolution": (["540p", "1080p", "272p"],),
            },
            "optional": {
                "image": ("IMAGE",),
                "video_guidance": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 100.0, "step": 0.1, "tooltip": "-1 = model default (5.0)"}),
                "audio_guidance": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 100.0, "step": 0.1, "tooltip": "-1 = model default (7.0)"}),
                "refiner_steps": ("INT", {"default": 0, "min": 0, "max": 1000, "tooltip": "1080p only; 0 = config default"}),
                "cfg_rescale": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "0 = off (MAGI-2's default)"}),
                "keep_loaded": ("BOOLEAN", {"default": True, "tooltip": "Off: free the GPU after this clip"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "FLOAT")
    RETURN_NAMES = ("frames", "audio", "fps")
    FUNCTION = "generate"
    CATEGORY = "MAGI-2"

    def generate(self, pipeline, prompt, seed, steps, resolution, image=None, video_guidance=None,
                 audio_guidance=None, refiner_steps=0, cfg_rescale=0.0, keep_loaded=True):
        if pipeline is not _STATE["pipeline"]:
            raise MagiNativeError("this MAGI-2 pipeline was unloaded; run the loader node again")
        try:
            return generate(pipeline, prompt, seed, steps, resolution, refiner_steps, image, video_guidance, audio_guidance, cfg_rescale)
        finally:
            if keep_loaded:
                _free_gpu()
            else:
                del pipeline
                unload_pipeline()


class MAGI2NativeUnload:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}, "optional": {"after": ("*", {"tooltip": "Connect anything to order this node after it"})}}

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "unload"
    CATEGORY = "MAGI-2"
    OUTPUT_NODE = True

    def unload(self, after=None):
        had = unload_pipeline()
        free = torch.cuda.mem_get_info()[0] / 2**30 if torch.cuda.is_available() else 0.0
        return (f"{'unloaded' if had else 'nothing loaded'}; {free:.1f} GiB free on the GPU",)
