from .nodes import MAGI2NativeGenerate, MAGI2NativeLoader, MAGI2NativeUnload

NODE_CLASS_MAPPINGS = {
    "MAGI2NativeLoader": MAGI2NativeLoader,
    "MAGI2NativeGenerate": MAGI2NativeGenerate,
    "MAGI2NativeUnload": MAGI2NativeUnload,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MAGI2NativeLoader": "MAGI-2 Load (native, 1 GPU)",
    "MAGI2NativeGenerate": "MAGI-2 Generate (native)",
    "MAGI2NativeUnload": "MAGI-2 Unload (native)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
