from . import magi2_comfy_model, nodes_model
from .nodes import MAGI2NativeGenerate, MAGI2NativeLoader, MAGI2NativeUnload

# Teach ComfyUI core the MAGI-2 model type (stock Load Diffusion Model then works).
magi2_comfy_model.register()

NODE_CLASS_MAPPINGS = {
    "MAGI2NativeLoader": MAGI2NativeLoader,
    "MAGI2NativeGenerate": MAGI2NativeGenerate,
    "MAGI2NativeUnload": MAGI2NativeUnload,
    **nodes_model.NODE_CLASS_MAPPINGS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MAGI2NativeLoader": "MAGI-2 Load (native, 1 GPU)",
    "MAGI2NativeGenerate": "MAGI-2 Generate (native)",
    "MAGI2NativeUnload": "MAGI-2 Unload (native)",
    **nodes_model.NODE_DISPLAY_NAME_MAPPINGS,
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
