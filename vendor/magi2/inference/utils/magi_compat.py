# Copyright (c) 2026 SandAI. All Rights Reserved.
# Apache-2.0.
"""MagiCompiler entry points, with an opt-in no-MagiCompiler fallback.

Default: re-export ``magi_compile`` / ``magi_register_custom_op`` from the
installed ``magi_compiler`` package (unchanged behaviour).

``MAGI2_NO_MAGI_COMPILER=1``: do not import ``magi_compiler`` at all.
``magi_compile`` returns its target unchanged (what the real package does for
``MAGI_COMPILE_COMPILE_MODE=NONE``) and ``magi_register_custom_op`` leaves the
function a plain Python callable, so the model runs as eager PyTorch in an
environment that does not have MagiCompiler installed.
"""

import functools
import os

NO_MAGI_COMPILER = os.environ.get("MAGI2_NO_MAGI_COMPILER", "0") == "1"

if NO_MAGI_COMPILER:

    def magi_compile(obj=None, **kwargs):
        if obj is None:
            return functools.partial(magi_compile, **kwargs)
        return obj

    def magi_register_custom_op(name=None, **kwargs):
        def decorator(fn):
            return fn

        return decorator

else:
    from magi_compiler import magi_compile, magi_register_custom_op  # noqa: F401

__all__ = ["NO_MAGI_COMPILER", "magi_compile", "magi_register_custom_op"]
