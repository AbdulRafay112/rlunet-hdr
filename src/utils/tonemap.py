"""Tone-mapping utility functions for HDR reconstruction."""

from typing import Union
import numpy as np
import torch


def mu_law_tonemap(
    x: Union[torch.Tensor, np.ndarray],
    mu: float = 5000.0,
) -> Union[torch.Tensor, np.ndarray]:
    """Apply mu-law tonemapping T(x) = log(1 + mu * x) / log(1 + mu).

    Compresses high dynamic range radiance values into [0, 1] while preserving
    contrast in dark and mid-tone regions.

    Args:
        x: Input non-negative tensor or numpy array.
        mu: Tone-mapping parameter (default: 5000.0).

    Returns:
        Tone-mapped values in [0, 1], same type as input.
    """
    if isinstance(x, torch.Tensor):
        orig_dtype = x.dtype
        if x.dtype in (torch.float16, torch.bfloat16):
            x_f32 = x.float()
            x_clamped = torch.clamp(x_f32, min=0.0)
            denom = float(np.log(1.0 + mu))
            res = torch.log1p(mu * x_clamped) / denom
            return res.to(orig_dtype)
        else:
            x_clamped = torch.clamp(x, min=0.0)
            denom = float(np.log(1.0 + mu))
            return torch.log1p(mu * x_clamped) / denom
    elif isinstance(x, np.ndarray):
        x_clamped = np.maximum(x, 0.0)
        denom = float(np.log(1.0 + mu))
        return np.log1p(mu * x_clamped) / denom
    else:
        raise TypeError(f"Unsupported input type: {type(x)}. Expected torch.Tensor or np.ndarray.")


def inv_mu_law_tonemap(
    t: Union[torch.Tensor, np.ndarray],
    mu: float = 5000.0,
) -> Union[torch.Tensor, np.ndarray]:
    """Inverse mu-law tonemapping: x = ((1 + mu)^t - 1) / mu.

    Inverts tone-mapped values in [0, 1] back to linear radiance values.

    Args:
        t: Tone-mapped values in [0, 1].
        mu: Tone-mapping parameter (default: 5000.0).

    Returns:
        Linear radiance values, same type as input.
    """
    if isinstance(t, torch.Tensor):
        t_clamped = torch.clamp(t, min=0.0)
        return (torch.pow(1.0 + mu, t_clamped) - 1.0) / mu
    elif isinstance(t, np.ndarray):
        t_clamped = np.maximum(t, 0.0)
        return (np.power(1.0 + mu, t_clamped) - 1.0) / mu
    else:
        raise TypeError(f"Unsupported input type: {type(t)}. Expected torch.Tensor or np.ndarray.")
