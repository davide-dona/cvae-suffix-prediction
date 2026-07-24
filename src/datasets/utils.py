from __future__ import annotations
import torch


def move_to_device(x, device: torch.device):
    """Recursively move every tensor in `x` to `device`, preserving its container shape.

    Handles the nested structures used throughout `src/training/`: a bare `Tensor`
    (e.g. `mean`/`std`), or the `(attributes, activities, timestamps, resources)` tuple
    where `attributes` is a `dict[str, Tensor]`. Anything else is returned unchanged.
    """
    if isinstance(x, torch.Tensor):
        return x.to(device)
    if isinstance(x, dict):
        return {key: move_to_device(value, device) for key, value in x.items()}
    if isinstance(x, tuple):
        return tuple(move_to_device(value, device) for value in x)
    if isinstance(x, list):
        return [move_to_device(value, device) for value in x]
    return x
