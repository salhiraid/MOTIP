# Compatibility shim for vendored DEIM code (Apache-2.0, see ./LICENSE).
# Upstream DEIM uses a YAML registry (`engine.core.register`) and its own dist utils; MOTIP2 builds
# the modules explicitly (see builder.py), so the registry decorator is a no-op here.
import torch
import torch.distributed
import torch.nn as nn


def register(name=None):
    """No-op replacement of DEIM's `@register()` (works both as `@register` and `@register()`)."""
    if callable(name):
        return name

    def _wrap(cls):
        return cls
    return _wrap


def is_dist_available_and_initialized() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def get_world_size() -> int:
    return torch.distributed.get_world_size() if is_dist_available_and_initialized() else 1


def de_parallel(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model
