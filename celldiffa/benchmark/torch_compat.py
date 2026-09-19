"""Numerically equivalent compatibility shims for the pinned upstream runtime."""

from __future__ import annotations


def extract_into_tensor(arr, timesteps, broadcast_shape):
    """Convert float64 schedules before transfer; MPS has no float64 storage.

    Upstream also returns float32, but casts only after device transfer. Moving
    that cast earlier does not alter the selected float32 coefficients.
    """
    import torch

    result = torch.as_tensor(arr, dtype=torch.float32, device=timesteps.device)[timesteps]
    while result.ndim < len(broadcast_shape):
        result = result[..., None]
    return result.expand(broadcast_shape)


def install_perturbdiff_mps_compat(device: str) -> None:
    if str(device) != "mps":
        return
    import importlib

    # The upstream modules import the function by name, so replace all four
    # bound references without changing files in the official checkout.
    for name in (
        "src.common.utils",
        "src.models.diffusion.diffusion_core",
        "src.models.diffusion.diffusion_sampling",
        "src.models.diffusion.diffusion_training",
    ):
        module = importlib.import_module(name)
        if hasattr(module, "_extract_into_tensor"):
            module._extract_into_tensor = extract_into_tensor
