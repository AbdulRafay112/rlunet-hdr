"""Evaluation script for single-image HDR reconstruction models.

Computes PSNR-mu and SSIM metrics on tone-mapped normalized outputs over the
validation split and saves side-by-side comparison PNGs:
  [Input LDR | Tone-mapped Prediction | Tone-mapped Ground Truth]
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Ensure project root is in sys.path
project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from src.data.dataset import HDRDataset
from src.training.train import build_model, get_train_val_split, load_config, set_seed
from src.utils.metrics import compute_psnr, compute_ssim, to_numpy_hwc
from src.utils.tonemap import mu_law_tonemap


def parse_args() -> argparse.Namespace:
    """Parse evaluation command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate HDR reconstruction model with PSNR-mu and SSIM metrics."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to trained model checkpoint (.pth).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="rlunet",
        choices=["baseline", "rlunet"],
        help="Model architecture (default: 'rlunet').",
    )
    parser.add_argument(
        "--base_channels",
        type=int,
        default=None,
        help="Base channels for model (default: 32 for RLUNet, 64 for BaselineUNet).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/configs/default_config.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Path to dataset root folder.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/figures/validation_eval",
        help="Directory to save side-by-side visual comparisons (default: results/figures/validation_eval).",
    )
    parser.add_argument(
        "--metrics_dir",
        type=str,
        default="results/metrics",
        help="Directory to save numerical evaluation metrics (default: results/metrics).",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Maximum number of visual comparison PNGs to save (default: 10).",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.10,
        help="Validation split ratio (default: 0.10).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for consistent validation splitting (default: 42).",
    )
    parser.add_argument(
        "--mu",
        type=float,
        default=5000.0,
        help="Mu-law parameter for tone mapping (default: 5000.0).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use ('cuda', 'cpu').",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run self-test with synthetic dummy data.",
    )
    return parser.parse_args()


def create_side_by_side_image(
    ldr_np: np.ndarray,
    pred_tm_np: np.ndarray,
    gt_tm_np: np.ndarray,
    psnr_val: float,
    ssim_val: float,
) -> np.ndarray:
    """Create a side-by-side comparison image [Input LDR | Tone-mapped Pred | Tone-mapped GT].

    Args:
        ldr_np: (H, W, 3) in [0, 1].
        pred_tm_np: (H, W, 3) tone-mapped in [0, 1].
        gt_tm_np: (H, W, 3) tone-mapped in [0, 1].
        psnr_val: PSNR-mu score for this sample.
        ssim_val: SSIM score for this sample.

    Returns:
        np.ndarray: (H, W*3, 3) uint8 RGB image with annotations.
    """
    # Convert to uint8 RGB
    ldr_uint8 = np.clip(ldr_np * 255.0, 0, 255).astype(np.uint8)
    pred_uint8 = np.clip(pred_tm_np * 255.0, 0, 255).astype(np.uint8)
    gt_uint8 = np.clip(gt_tm_np * 255.0, 0, 255).astype(np.uint8)

    # Concatenate side by side: [LDR | Pred | GT]
    composite = np.concatenate([ldr_uint8, pred_uint8, gt_uint8], axis=1)

    # Add text banner at the top
    h, w, _ = ldr_uint8.shape
    banner_height = 36
    banner = np.zeros((banner_height, composite.shape[1], 3), dtype=np.uint8)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    color = (255, 255, 255)
    thickness = 1

    cv2.putText(banner, "LDR Input (8-bit)", (10, 24), font, font_scale, color, thickness)
    cv2.putText(
        banner,
        f"Pred Tone-mapped (PSNR: {psnr_val:.2f}dB, SSIM: {ssim_val:.4f})",
        (w + 10, 24),
        font,
        font_scale,
        color,
        thickness,
    )
    cv2.putText(banner, "Ground Truth Tone-mapped", (2 * w + 10, 24), font, font_scale, color, thickness)

    return np.vstack([banner, composite])


def evaluate(
    checkpoint_path: Optional[str] = None,
    model_name: str = "rlunet",
    config_path: Optional[str] = None,
    data_dir: Optional[str] = None,
    output_dir: str = "results/figures/validation_eval",
    metrics_dir: str = "results/metrics",
    num_samples: int = 10,
    val_split: float = 0.10,
    seed: int = 42,
    mu: float = 5000.0,
    device: Optional[str] = None,
    base_channels: Optional[int] = None,
    synthetic_test: bool = False,
) -> Dict[str, float]:
    """Run evaluation and save metrics and visual comparisons."""
    cfg = load_config(config_path)
    set_seed(seed)

    if device is not None:
        target_device = torch.device(device)
    else:
        target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Resolve model
    resolved_model = model_name.strip().lower()
    if base_channels is None:
        base_channels = 32 if resolved_model == "rlunet" else 64

    model = build_model(
        model_name=resolved_model,
        in_channels=3,
        out_channels=3,
        base_channels=base_channels,
    ).to(target_device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())

    print("=" * 65)
    print(f"RLUNet Project - Model Evaluation")
    print("=" * 65)
    print(f"Model:           {model_name.upper()} ({resolved_model})")
    print(f"Total Params:    {total_params:,}")
    print(f"Device:          {target_device}")
    print(f"Mu parameter:    {mu}")
    print(f"Visual Dir:      {output_dir}")
    print(f"Metrics Dir:     {metrics_dir}")
    print("=" * 65)

    # Load checkpoint if provided
    if checkpoint_path and os.path.isfile(checkpoint_path):
        print(f"--> Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=target_device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"--> Successfully loaded checkpoint from epoch {checkpoint.get('epoch', 'N/A')}")
    elif checkpoint_path:
        print(f"[Warning] Checkpoint file '{checkpoint_path}' not found. Using initialized weights.")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    # Handle synthetic test mode or missing real dataset
    data_dir = (
        data_dir
        or cfg.get("data", {}).get("data_root")
        or "data/HDR-Real/HDR-Real"
    )

    is_dummy_run = synthetic_test
    if not is_dummy_run:
        if not os.path.isdir(data_dir):
            is_dummy_run = True
        else:
            _, val_set = get_train_val_split(data_dir=data_dir, val_split=val_split, seed=seed)
            if len(val_set) == 0:
                is_dummy_run = True

    if is_dummy_run:
        print("[Notice] Running evaluation test with synthetic dummy validation data...")
        num_eval = min(num_samples, 4)
        psnr_list: List[float] = []
        ssim_list: List[float] = []

        for idx in range(num_eval):
            # Synthetic 256x256 test sample
            torch.manual_seed(seed + idx)
            dummy_ldr = torch.rand(1, 3, 256, 256, device=target_device)
            dummy_hdr = torch.rand(1, 3, 256, 256, device=target_device)  # Normalized [0, 1]

            with torch.no_grad():
                pred = model(dummy_ldr)
                pred_clamped = torch.clamp(pred, min=0.0)

            # Mu-law tone mapping
            pred_tm = mu_law_tonemap(pred_clamped, mu=mu)
            gt_tm = mu_law_tonemap(dummy_hdr, mu=mu)

            pred_tm_np = to_numpy_hwc(pred_tm)
            gt_tm_np = to_numpy_hwc(gt_tm)
            ldr_np = to_numpy_hwc(dummy_ldr)

            p_val = compute_psnr(pred_tm_np, gt_tm_np, data_range=1.0)
            s_val = compute_ssim(pred_tm_np, gt_tm_np, data_range=1.0)

            psnr_list.append(p_val)
            ssim_list.append(s_val)

            # Save comparison PNG
            vis_img = create_side_by_side_image(ldr_np, pred_tm_np, gt_tm_np, p_val, s_val)
            out_file = os.path.join(output_dir, f"val_sample_{idx + 1:04d}.png")
            cv2.imwrite(out_file, cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))

        mean_psnr = float(np.mean(psnr_list))
        mean_ssim = float(np.mean(ssim_list))

        results = {
            "model": model_name,
            "total_params": total_params,
            "mean_psnr_mu": mean_psnr,
            "mean_ssim": mean_ssim,
            "num_evaluated": num_eval,
            "synthetic": True,
        }

        metrics_file = os.path.join(metrics_dir, "eval_metrics.json")
        with open(metrics_file, "w") as f:
            json.dump(results, f, indent=2)

        print("\n" + "=" * 65)
        print("Evaluation Results Summary (Synthetic Test):")
        print(f"  Mean PSNR-mu:    {mean_psnr:.2f} dB")
        print(f"  Mean SSIM:       {mean_ssim:.4f}")
        print(f"  Visualizations:  Saved to '{output_dir}'")
        print(f"  Metrics JSON:    Saved to '{metrics_file}'")
        print("=" * 65)
        return results

    # Full real validation evaluation
    _, val_set = get_train_val_split(data_dir=data_dir, val_split=val_split, seed=seed)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=0)

    print(f"Evaluating on {len(val_set)} validation sample(s)...")
    psnr_list = []
    ssim_list = []

    for idx, batch in enumerate(val_loader):
        if len(batch) == 3:
            ldr_img, hdr_gt, _ = batch
        else:
            ldr_img, hdr_gt = batch

        ldr_img = ldr_img.to(target_device)
        hdr_gt = hdr_gt.to(target_device)

        with torch.no_grad():
            pred = model(ldr_img)
            pred_clamped = torch.clamp(pred, min=0.0)

        # Tone mapping
        pred_tm = mu_law_tonemap(pred_clamped, mu=mu)
        gt_tm = mu_law_tonemap(hdr_gt, mu=mu)

        pred_tm_np = to_numpy_hwc(pred_tm)
        gt_tm_np = to_numpy_hwc(gt_tm)
        ldr_np = to_numpy_hwc(ldr_img)

        p_val = compute_psnr(pred_tm_np, gt_tm_np, data_range=1.0)
        s_val = compute_ssim(pred_tm_np, gt_tm_np, data_range=1.0)

        psnr_list.append(p_val)
        ssim_list.append(s_val)

        # Save side-by-side visualization for the first `num_samples`
        if idx < num_samples:
            vis_img = create_side_by_side_image(ldr_np, pred_tm_np, gt_tm_np, p_val, s_val)
            out_file = os.path.join(output_dir, f"val_sample_{idx + 1:04d}.png")
            cv2.imwrite(out_file, cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))

    mean_psnr = float(np.mean(psnr_list))
    mean_ssim = float(np.mean(ssim_list))

    results = {
        "model": model_name,
        "total_params": total_params,
        "mean_psnr_mu": mean_psnr,
        "mean_ssim": mean_ssim,
        "num_evaluated": len(val_set),
        "synthetic": False,
    }

    metrics_file = os.path.join(metrics_dir, "eval_metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 65)
    print("Evaluation Results Summary:")
    print(f"  Mean PSNR-mu:    {mean_psnr:.2f} dB")
    print(f"  Mean SSIM:       {mean_ssim:.4f}")
    print(f"  Visualizations:  Saved {min(len(val_set), num_samples)} images to '{output_dir}'")
    print(f"  Metrics JSON:    Saved to '{metrics_file}'")
    print("=" * 65)
    return results


def main():
    args = parse_args()
    evaluate(
        checkpoint_path=args.checkpoint,
        model_name=args.model,
        config_path=args.config,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        metrics_dir=args.metrics_dir,
        num_samples=args.num_samples,
        val_split=args.val_split,
        seed=args.seed,
        mu=args.mu,
        device=args.device,
        base_channels=args.base_channels,
        synthetic_test=args.test,
    )


if __name__ == "__main__":
    main()
