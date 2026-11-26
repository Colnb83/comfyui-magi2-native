# Copyright (c) 2026 SandAI. All Rights Reserved.
# Apache-2.0.
"""The few ``diffusers`` names this package uses, with a built-in fallback.

Default: everything is re-exported from the installed ``diffusers`` (unchanged
behaviour).

If ``diffusers`` cannot be imported (absent, or an old release broken by a newer
``huggingface_hub``, as in ComfyUI's Python on the development host), minimal
stand-ins are used instead. They cover exactly what this package calls:

* ``ConfigMixin`` / ``register_to_config``: record ``__init__`` arguments in
  ``self.config`` (attribute and item access), ``from_config(dict)``.
* ``SchedulerMixin``, ``SchedulerOutput``, ``KarrasDiffusionSchedulers``.
* ``ModelMixin``: an ``nn.Module`` with ``device`` / ``dtype`` properties.
* ``deprecate``, ``randn_tensor``, ``load_image``.
* ``VideoProcessor.preprocess`` for PIL images with the default settings
  (Lanczos resize to a multiple of ``vae_scale_factor``, scale to [-1, 1]).

``MAGI2_DIFFUSERS_COMPAT=1`` forces the stand-ins even when ``diffusers`` works
(used to test them against the real package).
"""

import functools
import inspect
import os
import warnings
from dataclasses import dataclass
from enum import Enum

import numpy as np
import torch
import torch.nn as nn

USING_DIFFUSERS = False
if os.environ.get("MAGI2_DIFFUSERS_COMPAT", "0") != "1":
    try:
        from diffusers.configuration_utils import ConfigMixin, register_to_config  # noqa: F401
        from diffusers.models.modeling_utils import ModelMixin  # noqa: F401
        from diffusers.schedulers.scheduling_utils import (  # noqa: F401
            KarrasDiffusionSchedulers,
            SchedulerMixin,
            SchedulerOutput,
        )
        from diffusers.utils import deprecate, load_image  # noqa: F401
        from diffusers.utils.torch_utils import randn_tensor  # noqa: F401
        from diffusers.video_processor import VideoProcessor  # noqa: F401

        USING_DIFFUSERS = True
    except Exception:  # ImportError, or a broken install raising at import time
        USING_DIFFUSERS = False

if not USING_DIFFUSERS:

    class _FrozenConfig(dict):
        """Init arguments, readable as ``cfg.name`` or ``cfg["name"]``."""

        def __getattr__(self, name):
            try:
                return self[name]
            except KeyError:
                raise AttributeError(name) from None

    class ConfigMixin:
        def register_to_config(self, **kwargs):
            kwargs.pop("kwargs", None)
            merged = dict(getattr(self, "_internal_dict", {}))
            merged.update(kwargs)
            object.__setattr__(self, "_internal_dict", _FrozenConfig(merged))

        @property
        def config(self):
            return self._internal_dict

        @classmethod
        def from_config(cls, config, **kwargs):
            params = inspect.signature(cls.__init__).parameters
            accepted = {
                k: v
                for k, v in {**dict(config), **kwargs}.items()
                if k in params and k != "self" and not k.startswith("_")
            }
            return cls(**accepted)

    def register_to_config(init):
        @functools.wraps(init)
        def inner_init(self, *args, **kwargs):
            init_kwargs = {k: v for k, v in kwargs.items() if not k.startswith("_")}
            signature = inspect.signature(init)
            parameters = {
                name: p.default
                for i, (name, p) in enumerate(signature.parameters.items())
                if i > 0 and p.kind is not inspect.Parameter.VAR_KEYWORD
            }
            new_kwargs = {}
            for arg, name in zip(args, parameters.keys()):
                new_kwargs[name] = arg
            new_kwargs.update(
                {k: init_kwargs.get(k, default) for k, default in parameters.items() if k not in new_kwargs}
            )
            self.register_to_config(**new_kwargs)
            init(self, *args, **init_kwargs)

        return inner_init

    class KarrasDiffusionSchedulers(Enum):
        """Only iterated for ``_compatibles``; no compatible classes here."""

    class SchedulerMixin:
        pass

    @dataclass
    class SchedulerOutput:
        prev_sample: torch.Tensor

    class ModelMixin(nn.Module):
        @property
        def device(self) -> torch.device:
            return next(self.parameters()).device

        @property
        def dtype(self) -> torch.dtype:
            return next(self.parameters()).dtype

    def deprecate(name, version, message, **kwargs):
        warnings.warn(f"`{name}` is deprecated. {message}", FutureWarning, stacklevel=3)

    def randn_tensor(shape, generator=None, device=None, dtype=None, layout=None):
        layout = layout or torch.strided
        device = torch.device(device) if device is not None else torch.device("cpu")
        rand_device = device
        if generator is not None and generator.device.type != device.type and generator.device.type == "cpu":
            rand_device = torch.device("cpu")
        return torch.randn(
            tuple(shape), generator=generator, device=rand_device, dtype=dtype, layout=layout
        ).to(device)

    def load_image(image):
        import PIL.Image
        import PIL.ImageOps

        if isinstance(image, str):
            if image.startswith(("http://", "https://")):
                raise ValueError("load_image stand-in: URLs are not supported, pass a file path")
            image = PIL.Image.open(image)
        elif not isinstance(image, PIL.Image.Image):
            raise ValueError("load_image expects a file path or a PIL image")
        image = PIL.ImageOps.exif_transpose(image)
        return image.convert("RGB")

    class VideoProcessor:
        def __init__(self, vae_scale_factor: int = 8):
            self.vae_scale_factor = int(vae_scale_factor)

        def preprocess(self, image, height=None, width=None):
            import PIL.Image

            if not isinstance(image, PIL.Image.Image):
                raise ValueError("VideoProcessor stand-in: only a single PIL image is supported")
            height = image.height if height is None else height
            width = image.width if width is None else width
            width, height = (x - x % self.vae_scale_factor for x in (width, height))
            image = image.resize((width, height), resample=PIL.Image.LANCZOS)
            array = np.array(image).astype(np.float32) / 255.0
            if array.ndim == 2:
                array = array[..., None]
            tensor = torch.from_numpy(array[None].transpose(0, 3, 1, 2))
            return 2.0 * tensor - 1.0


__all__ = [
    "USING_DIFFUSERS",
    "ConfigMixin",
    "KarrasDiffusionSchedulers",
    "ModelMixin",
    "SchedulerMixin",
    "SchedulerOutput",
    "VideoProcessor",
    "deprecate",
    "load_image",
    "randn_tensor",
    "register_to_config",
]
