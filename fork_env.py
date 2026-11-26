"""Locating and importing the MAGI-2 fork, shared by both node families."""

import contextlib
import logging
import os
import sys

import torch

PACK_DIR = os.path.dirname(os.path.realpath(__file__))

# Single process, no MagiCompiler, PyTorch's flash kernel, NVFP4 experts,
# one guidance branch per forward (halves the activations alive at once).
ENV_DEFAULTS = {
    "MAGI2_NO_DIST": "1",
    "MAGI2_NO_MAGI_COMPILER": "1",
    "MAGI2_ATTN_BACKEND": "sdpa",
    "MAGI2_MOE_FP8": "1",
    "MAGI2_MOE_QUANT": "nvfp4",
    "MAGI2_CFG_SEQUENTIAL": "1",
    "MAGI2_PREVIEW_OFFLOAD_MODE": "gpu",
    "MAGI2_TEXT_ENC_OFFLOAD_MODE": "cpu",
    "MAGI2_VAE_OFFLOAD_MODE": "roundtrip",  # VAEs wait in host RAM while denoising (2.8 GiB)
    "MAGI2_REFINER_OFFLOAD_MODE": "roundtrip",
    # 1080p: drop the preview experts while the refiner runs (re-streamed for the
    # next clip) and use the memory-bounded refiner forward.
    "MAGI2_PREVIEW_RELEASE_EXPERTS": "1",
    "MAGI2_REFINER_LOWMEM": "1",
}


class MagiForkError(RuntimeError):
    pass


def fork_dir():
    # MAGI2_FORK_DIR, then a development checkout at magi2_fork/, then the vendored copy.
    path = os.environ.get("MAGI2_FORK_DIR")
    if not path:
        path = os.path.join(PACK_DIR, "magi2_fork")
        if not os.path.exists(path):
            path = os.path.join(PACK_DIR, "vendor", "magi2")
    path = os.path.realpath(path)
    if not os.path.isfile(os.path.join(path, "inference", "pipeline", "entry.py")):
        raise MagiForkError(
            f"MAGI-2 fork not found at {path}: put the fork (the directory holding inference/ and "
            "configs/) at magi2_fork/ inside this node pack, or set MAGI2_FORK_DIR"
        )
    return path


def prepare_fork_import():
    """Set the fork's switches (existing environment values win) and make its
    ``inference`` package importable. Returns the fork directory."""
    path = fork_dir()
    for name, value in ENV_DEFAULTS.items():
        os.environ.setdefault(name, value)
    # Config files name the weights directory through this variable; the model
    # class itself never reads from it.
    os.environ.setdefault("MAGI2_CKPT_ROOT", "/MAGI2_CKPT_ROOT-not-set")
    if path not in sys.path:
        sys.path.insert(0, path)
        logging.info("[MAGI-2] using the MAGI-2 inference code at %s", path)
    return path


@contextlib.contextmanager
def image_numerics():
    """The reference Docker image runs fp32 matmuls as TF32; match it while
    MAGI-2 code runs and put ComfyUI's setting back afterwards."""
    saved = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = saved
