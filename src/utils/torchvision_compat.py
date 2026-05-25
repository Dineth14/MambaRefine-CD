"""Compatibility shim for older Torch with newer TorchVision/Timm imports."""
from __future__ import annotations


def patch_register_fake() -> None:
    try:
        import torch
    except Exception:
        return
    library = getattr(torch, "library", None)
    if library is None or hasattr(library, "register_fake"):
        return

    def register_fake(*args, **kwargs):
        def decorator(fn):
            return fn
        return decorator

    library.register_fake = register_fake
