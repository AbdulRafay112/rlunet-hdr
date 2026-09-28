"""Evaluation metrics for HDR image reconstruction: PSNR, SSIM, and PSNR-mu."""

from typing import Union
import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim_fn

from src.utils.tonemap import mu_law_tonemap


def to_numpy_hwc(img: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """Convert a PyTorch tensor (C, H, W) or numpy array to a (H, W, C) numpy array."""
    if isinstance(img, torch.Tensor):
        if img.ndim == 4:
            img = img.squeeze(0)
        img_np = img.detach().cpu().numpy()
        if img_np.ndim == 3 and img_np.shape[0] in (1, 3):
            img_np = np.transpose(img_np, (1, 2, 0))
        return img_np
    elif isinstance(img, np.ndarray):
        if img.ndim == 3 and img.shape[0] in (1, 3) and img.shape[2] not in (1, 3):
            return np.transpose(img, (1, 2, 0))
        return img
    else:
        raise TypeError(f"Expected torch.Tensor or np.ndarray, got {type(img)}")


def compute_psnr(
    pred: Union[torch.Tensor, np.ndarray],
    target: Union[torch.Tensor, np.ndarray],
    data_range: float = 1.0,
) -> float:
    """Compute Peak Signal-to-Noise Ratio (PSNR) between prediction and ground truth.

    Args:
        pred: Predicted image tensor or array.
        target: Ground truth image tensor or array.
        data_range: Dynamic range of input images (default: 1.0).

    Returns:
        float: PSNR in decibels (dB).
    """
    pred_arr = to_numpy_hwc(pred).astype(np.float64)
    target_arr = to_numpy_hwc(target).astype(np.float64)

    mse = np.mean((pred_arr - target_arr) ** 2)
    if mse <= 1e-12:
        return 100.0  # Cap perfect match at 100 dB

    psnr = 10.0 * np.log10((data_range ** 2) / mse)
    return float(psnr)


def compute_ssim(
    pred: Union[torch.Tensor, np.ndarray],
    target: Union[torch.Tensor, np.ndarray],
    data_range: float = 1.0,
) -> float:
    """Compute Structural Similarity Index Measure (SSIM).

    Args:
        pred: Predicted image tensor or array.
        target: Ground truth image tensor or array.
        data_range: Dynamic range of input images (default: 1.0).

    Returns:
        float: SSIM score in [-1.0, 1.0].
    """
    pred_arr = to_numpy_hwc(pred).astype(np.float64)
    target_arr = to_numpy_hwc(target).astype(np.float64)

    channel_axis = 2 if pred_arr.ndim == 3 and pred_arr.shape[2] > 1 else None
    val = ssim_fn(
        target_arr,
        pred_arr,
        data_range=data_range,
        channel_axis=channel_axis,
    )
    return float(val)


def compute_psnr_mu(
    pred: Union[torch.Tensor, np.ndarray],
    target: Union[torch.Tensor, np.ndarray],
    mu: float = 5000.0,
) -> float:
    """Compute PSNR-mu on mu-law tone-mapped outputs with data_range=1.0.

    Args:
        pred: Predicted non-negative HDR image.
        target: Ground truth non-negative HDR image.
        mu: Mu-law parameter (default: 5000.0).

    Returns:
        float: PSNR-mu in dB.
    """
    pred_tm = mu_law_tonemap(pred, mu=mu)
    target_tm = mu_law_tonemap(target, mu=mu)
    return compute_psnr(pred_tm, target_tm, data_range=1.0)


def compute_ssim_mu(
    pred: Union[torch.Tensor, np.ndarray],
    target: Union[torch.Tensor, np.ndarray],
    mu: float = 5000.0,
) -> float:
    """Compute SSIM on mu-law tone-mapped outputs with data_range=1.0.

    Args:
        pred: Predicted non-negative HDR image.
        target: Ground truth non-negative HDR image.
        mu: Mu-law parameter (default: 5000.0).

    Returns:
        float: SSIM score on tone-mapped domain.
    """
    pred_tm = mu_law_tonemap(pred, mu=mu)
    target_tm = mu_law_tonemap(target, mu=mu)
    return compute_ssim(pred_tm, target_tm, data_range=1.0)
