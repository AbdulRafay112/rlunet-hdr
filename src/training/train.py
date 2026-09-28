"""Training script for RLUNet and BaselineUNet on HDRDataset."""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

# Ensure project root is in sys.path for direct script execution
project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import yaml

from src.data.dataset import HDRDataset, get_train_val_split
from src.models.baseline_unet import BaselineUNet
from src.models.rlunet import RLUNet
from src.utils.tonemap import mu_law_tonemap


class MuLawLoss(nn.Module):
    """L1 loss in the mu-law tone-mapped domain: T(x) = log(1 + mu*x) / log(1 + mu)."""

    def __init__(self, mu: float = 5000.0, max_val: float = 16.0):
        super().__init__()
        self.mu = mu
        self.max_val = max_val

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Cast to float32 to compute loss outside autocast in high precision
        pred_f32 = pred.float()
        target_f32 = target.float()

        # Guard: if prediction or target contains inf/nan, produce non-finite loss for batch skipping
        if not torch.isfinite(pred_f32).all() or not torch.isfinite(target_f32).all():
            return torch.tensor(float("inf"), device=pred.device, dtype=torch.float32, requires_grad=True)

        # Clamp predictions to [0, 16] before the mu-law transform (normalized targets are in [0, 1])
        pred_clamped = torch.clamp(pred_f32, min=0.0, max=self.max_val)
        target_clamped = torch.clamp(target_f32, min=0.0)
        pred_tm = mu_law_tonemap(pred_clamped, mu=self.mu)
        target_tm = mu_law_tonemap(target_clamped, mu=self.mu)
        return F.l1_loss(pred_tm, target_tm)


def load_config(config_path: Optional[str]) -> Dict[str, Any]:
    """Load configuration from a YAML file if provided and exists."""
    if not config_path or not os.path.isfile(config_path):
        return {}
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config or {}


def parse_args() -> argparse.Namespace:
    """Parse command line arguments with fallback to YAML config defaults."""
    parser = argparse.ArgumentParser(
        description="Train RLUNet or BaselineUNet for single-image HDR reconstruction."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="rlunet",
        choices=["baseline", "rlunet"],
        help="Model architecture to train (choices: 'baseline', 'rlunet'; default: 'rlunet').",
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="mulaw",
        choices=["mulaw", "l1"],
        help="Loss function type: 'mulaw' for tone-mapped L1 (mu=5000), 'l1' for raw L1 (default: 'mulaw').",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Enable automatic mixed precision (AMP) using torch.cuda.amp.",
    )
    parser.add_argument(
        "--base_channels",
        type=int,
        default=None,
        help="Base feature channels for the model (defaults to 32 for RLUNet [~7.79M], 64 for BaselineUNet [~31.04M]).",
    )
    parser.add_argument(
        "--crop_size",
        type=int,
        default=256,
        help="Random crop size for training images (default: 256).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/configs/default_config.yaml",
        help="Path to YAML configuration file (default: experiments/configs/default_config.yaml).",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Path to root dataset folder containing numbered sample subfolders.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Directory to save model checkpoints (default: models/checkpoints).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Total number of training epochs (overrides config train.epochs).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size for training and validation (overrides config data.batch_size).",
    )
    parser.add_argument(
        "--lr",
        "--learning_rate",
        dest="lr",
        type=float,
        default=None,
        help="Learning rate for Adam optimizer (overrides config train.learning_rate).",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=None,
        help="Weight decay for Adam optimizer (overrides config train.weight_decay).",
    )
    parser.add_argument(
        "--save_freq",
        "--save_freq_epochs",
        dest="save_freq",
        type=int,
        default=None,
        help="Frequency (in epochs) to save periodic checkpoints (overrides config logging.save_freq_epochs).",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint file (.pth) to resume training from, or 'latest' to auto-detect.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Number of workers for DataLoader (overrides config data.num_workers).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use for training ('cuda', 'cpu', 'cuda:0', etc.).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (overrides config seed).",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.10,
        help="Fraction of dataset to use for validation when layout='folders' (default: 0.10 for 90/10 split).",
    )
    parser.add_argument(
        "--val_count",
        type=int,
        default=300,
        help="Number of held-out validation images for 'flat' layout (default: 300).",
    )
    parser.add_argument(
        "--layout",
        type=str,
        default="folders",
        choices=["folders", "flat"],
        help="Dataset directory layout: 'folders' (<root>/NNNNN/input.jpg) or 'flat' (<root>/LDR_in/*.jpg, <root>/HDR_gt/*.hdr) (default: 'folders').",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Run 1 training and 1 validation step for sanity check, then exit.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def save_checkpoint(
    state: Dict[str, Any],
    checkpoint_dir: str,
    filename: str,
    is_best: bool = False,
) -> str:
    """Save training checkpoint to disk."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)

    # Save latest checkpoint pointer
    latest_path = os.path.join(checkpoint_dir, "latest_checkpoint.pth")
    torch.save(state, latest_path)

    if is_best:
        best_path = os.path.join(checkpoint_dir, "best_checkpoint.pth")
        torch.save(state, best_path)

    return filepath


def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Find the latest checkpoint in checkpoint_dir."""
    latest_path = os.path.join(checkpoint_dir, "latest_checkpoint.pth")
    if os.path.isfile(latest_path):
        return latest_path

    # Check for periodic checkpoint files
    if os.path.isdir(checkpoint_dir):
        checkpoints = [
            os.path.join(checkpoint_dir, f)
            for f in os.listdir(checkpoint_dir)
            if f.endswith(".pth") and f.startswith("checkpoint_epoch_")
        ]
        if checkpoints:
            checkpoints.sort(key=os.path.getmtime)
            return checkpoints[-1]
    return None


def unpack_batch(
    batch: Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Unpack batch supporting either (ldr, hdr) or (ldr, hdr, scale_factor)."""
    if len(batch) == 3:
        ldr, hdr, scale = batch
        return ldr.to(device, non_blocking=True), hdr.to(device, non_blocking=True), scale.to(device, non_blocking=True)
    else:
        ldr, hdr = batch
        return ldr.to(device, non_blocking=True), hdr.to(device, non_blocking=True), None


def get_autocast_context(device: torch.device, enabled: bool):
    """Return device-appropriate autocast context manager."""
    device_type = "cuda" if device.type == "cuda" else "cpu"
    if hasattr(torch, "autocast"):
        return torch.autocast(device_type=device_type, enabled=enabled)
    return torch.cuda.amp.autocast(enabled=enabled and device.type == "cuda")


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler],
    device: torch.device,
    use_amp: bool = False,
    dry_run: bool = False,
) -> Tuple[float, int]:
    """Run one epoch of training and return (average_loss, skipped_batches)."""
    model.train()
    running_loss = 0.0
    total_samples = 0
    skipped_batches = 0
    amp_enabled = use_amp and (device.type == "cuda")

    for batch_idx, batch in enumerate(dataloader):
        ldr_imgs, hdr_gts, _ = unpack_batch(batch, device)

        optimizer.zero_grad()

        # 1. Forward pass under autocast (if AMP enabled)
        with get_autocast_context(device, amp_enabled):
            predictions = model(ldr_imgs)

        # 1. Compute loss in float32 outside autocast
        predictions_f32 = predictions.float()
        hdr_gts_f32 = hdr_gts.float()
        loss = criterion(predictions_f32, hdr_gts_f32)

        # 3. Guard: if loss is not finite, skip batch (no backward, no optimizer step)
        if not torch.isfinite(loss):
            skipped_batches += 1
            continue

        if amp_enabled and scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # Report average loss only over finite batches
        running_loss += loss.item() * ldr_imgs.size(0)
        total_samples += ldr_imgs.size(0)

        if dry_run:
            break

    avg_loss = running_loss / total_samples if total_samples > 0 else float("nan")
    return avg_loss, skipped_batches


def validate(
    model: nn.Module,
    dataloader: Optional[DataLoader],
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool = False,
    dry_run: bool = False,
) -> Tuple[float, int]:
    """Run evaluation on the validation set and return (average_loss, skipped_batches)."""
    if dataloader is None or len(dataloader) == 0:
        return 0.0, 0

    model.eval()
    running_loss = 0.0
    total_samples = 0
    skipped_batches = 0
    amp_enabled = use_amp and (device.type == "cuda")

    with torch.no_grad():
        for batch in dataloader:
            ldr_imgs, hdr_gts, _ = unpack_batch(batch, device)

            with get_autocast_context(device, amp_enabled):
                predictions = model(ldr_imgs)

            # 4. Validation uses the same float32 loss computation outside autocast
            predictions_f32 = predictions.float()
            hdr_gts_f32 = hdr_gts.float()
            loss = criterion(predictions_f32, hdr_gts_f32)

            # Skip non-finite batches to keep metrics finite
            if not torch.isfinite(loss):
                skipped_batches += 1
                continue

            running_loss += loss.item() * ldr_imgs.size(0)
            total_samples += ldr_imgs.size(0)

            if dry_run:
                break

    avg_loss = running_loss / total_samples if total_samples > 0 else float("nan")
    return avg_loss, skipped_batches


def build_model(
    model_name: str = "rlunet",
    in_channels: int = 3,
    out_channels: int = 3,
    base_channels: Optional[int] = None,
) -> nn.Module:
    """Build and return either RLUNet or BaselineUNet with correct architectural defaults.

    Args:
        model_name: 'rlunet' or 'baseline' (case-insensitive).
        in_channels: Number of input color channels (default: 3).
        out_channels: Number of output HDR channels (default: 3).
        base_channels: Base feature channels.
                       Defaults to 32 for RLUNet (~7.79M params)
                       and 64 for BaselineUNet (~31.04M params).

    Returns:
        nn.Module: Instantiated model.
    """
    model_type = model_name.strip().lower()
    if model_type in ("rlunet", "rl_unet", "imagingpipelinemodule", "ipm"):
        channels = base_channels if base_channels is not None else 32
        return RLUNet(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=channels,
        )
    elif model_type in ("baseline", "baselineunet", "baseline_unet"):
        channels = base_channels if base_channels is not None else 64
        return BaselineUNet(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=channels,
        )
    else:
        raise ValueError(
            f"Unsupported model type '{model_name}'. Expected 'baseline' or 'rlunet'."
        )


def train(
    config_path: Optional[str] = None,
    model_name: Optional[str] = None,
    data_dir: Optional[str] = None,
    checkpoint_dir: Optional[str] = None,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    lr: Optional[float] = None,
    weight_decay: Optional[float] = None,
    save_freq: Optional[int] = None,
    resume: Optional[str] = None,
    num_workers: Optional[int] = None,
    device: Optional[str] = None,
    seed: Optional[int] = None,
    val_split: float = 0.10,
    val_count: Optional[int] = None,
    layout: str = "folders",
    dry_run: bool = False,
    base_channels: Optional[int] = None,
    loss_type: Optional[str] = None,
    use_amp: bool = False,
    crop_size: int = 256,
) -> Dict[str, Any]:
    """Main training routine."""
    cfg = load_config(config_path)

    # Resolve settings (CLI arguments take precedence over YAML config)
    seed = seed if seed is not None else cfg.get("seed", 42)
    set_seed(seed)

    # Resolve model architecture ("rlunet" or "baseline", default: "rlunet")
    raw_model = (
        model_name
        or cfg.get("model", {}).get("name")
        or "rlunet"
    )
    model_type = raw_model.strip().lower()
    if model_type in ("baseline", "baselineunet", "baseline_unet"):
        model_type = "baseline"
    elif model_type in ("rlunet", "rl_unet", "imagingpipelinemodule", "ipm"):
        model_type = "rlunet"
    else:
        raise ValueError(
            f"Unsupported model type '{raw_model}'. Expected 'baseline' or 'rlunet'."
        )

    # Resolve loss type
    raw_loss = (
        loss_type
        or cfg.get("train", {}).get("loss_type")
        or "mulaw"
    )
    resolved_loss_type = raw_loss.strip().lower()
    if "mulaw" in resolved_loss_type or "mu_law" in resolved_loss_type:
        criterion = MuLawLoss(mu=5000.0)
        loss_label = "MuLawLoss (mu=5000)"
    else:
        criterion = nn.L1Loss()
        loss_label = "Raw L1Loss"

    data_dir = (
        data_dir
        or cfg.get("data", {}).get("data_root")
        or "data/HDR-Real/HDR-Real"
    )
    checkpoint_dir = (
        checkpoint_dir
        or cfg.get("logging", {}).get("save_dir")
        or "models/checkpoints"
    )
    epochs = (
        epochs
        if epochs is not None
        else cfg.get("train", {}).get("epochs", 100)
    )
    batch_size = (
        batch_size
        if batch_size is not None
        else cfg.get("data", {}).get("batch_size", 8)
    )
    lr = (
        lr
        if lr is not None
        else cfg.get("train", {}).get("learning_rate", 0.0002)
    )
    weight_decay = (
        weight_decay
        if weight_decay is not None
        else cfg.get("train", {}).get("weight_decay", 0.0001)
    )
    save_freq = (
        save_freq
        if save_freq is not None
        else cfg.get("logging", {}).get("save_freq_epochs", 5)
    )

    # Determine num_workers
    default_workers = (
        0 if sys.platform.startswith("win") else 2
    )
    num_workers = (
        num_workers
        if num_workers is not None
        else cfg.get("data", {}).get("num_workers", default_workers)
    )

    # Determine device
    if device is not None:
        target_device = torch.device(device)
    else:
        target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_display_name = "RLUNet" if model_type == "rlunet" else "BaselineUNet"

    # Resolve base_channels
    if base_channels is not None:
        model_base_channels = base_channels
    else:
        cfg_model_name = str(cfg.get("model", {}).get("name", "")).strip().lower()
        if model_type == "rlunet":
            if cfg_model_name in ("rlunet", "rl_unet", "imagingpipelinemodule", "ipm"):
                cfg_channels = cfg.get("model", {}).get("base_channels")
                model_base_channels = cfg_channels if cfg_channels is not None else 32
            else:
                model_base_channels = cfg.get("model", {}).get("rlunet_base_channels", 32)
        else:
            if cfg_model_name in ("baseline", "baselineunet", "baseline_unet"):
                model_base_channels = cfg.get("model", {}).get("base_channels", 64)
            else:
                model_base_channels = 64

    # Resolve layout and validation parameters
    resolved_layout = (
        layout
        or cfg.get("data", {}).get("layout")
        or "folders"
    ).lower().strip()
    resolved_val_count = (
        val_count
        if val_count is not None
        else cfg.get("data", {}).get("val_count", 300)
    )

    # Mixed precision setup
    amp_active = use_amp and (target_device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_active) if target_device.type == "cuda" else None

    print("=" * 65)
    print(f"RLUNet Project - {model_display_name} Training")
    print("=" * 65)
    print(f"Model:            {model_display_name} ({model_type})")
    print(f"Loss Function:    {loss_label}")
    print(f"Mixed Precision:  {'Enabled (torch.cuda.amp)' if amp_active else ('Disabled (CPU)' if use_amp else 'Disabled')}")
    print(f"Device:           {target_device}")
    print(f"Data Directory:   {data_dir}")
    print(f"Data Layout:      {resolved_layout}")
    if resolved_layout == "flat":
        print(f"Validation Set:   Fixed {resolved_val_count} held-out images")
    else:
        print(f"Validation Ratio: {val_split * 100:.1f}%")
    print(f"Checkpoints:      {checkpoint_dir}")
    print(f"Total Epochs:     {epochs}")
    print(f"Batch Size:       {batch_size}")
    print(f"Learning Rate:    {lr}")
    print(f"Weight Decay:     {weight_decay}")
    print(f"Save Frequency:   Every {save_freq} epoch(s)")
    print(f"Num Workers:      {num_workers}")
    print(f"Random Seed:      {seed}")
    print("=" * 65)

    # 1. Instantiate Model
    model = build_model(
        model_name=model_type,
        in_channels=3,
        out_channels=3,
        base_channels=model_base_channels,
    ).to(target_device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Parameters: {total_params:,} (Trainable: {trainable_params:,})")

    # 2. Prepare Train and Val Datasets via seeded Subset
    train_set, val_set = get_train_val_split(
        data_dir=data_dir,
        layout=resolved_layout,
        val_split=val_split,
        val_count=resolved_val_count,
        seed=seed,
        crop_size=crop_size,
        normalize_hdr=True,
    )
    total_samples = len(train_set) + len(val_set)
    print(f"Loaded dataset ({resolved_layout}): {len(train_set)} train (with 256x256 crop & augmentations), {len(val_set)} validation (full image).")

    if total_samples == 0:
        print("[Warning] No sample folders found in dataset directory.")
        print(f"--> Running dummy verification test for {model_display_name}...")
        test_b = min(batch_size, 2)
        dummy_in = torch.rand(test_b, 3, crop_size, crop_size, device=target_device)
        dummy_gt = torch.rand(test_b, 3, crop_size, crop_size, device=target_device)
        with torch.no_grad():
            dummy_out = model(dummy_in)
            dummy_loss = criterion(dummy_out.float(), dummy_gt.float())
        print(f"  Dummy Input:    {tuple(dummy_in.shape)}")
        print(f"  Dummy Output:   {tuple(dummy_out.shape)}")
        print(f"  Dummy Loss:     {dummy_loss.item():.6f} ({loss_label})")
        print(f"  Model Params:   {total_params:,}")
        print(f"[SUCCESS] {model_display_name} verification passed ({total_params:,} parameters)!")
        return {"model": model, "total_params": total_params, "verified": True}

    pin_memory = target_device.type == "cuda"
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # 3. Optimizer & Scheduler
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    lr_decay_step = cfg.get("train", {}).get("lr_decay_step", 40)
    lr_gamma = cfg.get("train", {}).get("lr_gamma", 0.5)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=lr_decay_step, gamma=lr_gamma
    )

    # 4. Resume from Checkpoint if requested
    start_epoch = 1
    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": [], "lr": []}

    resume_path = None
    if resume:
        if resume.lower() in ("latest", "auto"):
            resume_path = find_latest_checkpoint(checkpoint_dir)
            if not resume_path:
                print(f"[Warning] No existing checkpoint found in '{checkpoint_dir}' to resume from.")
        else:
            resume_path = resume

    if resume_path:
        if not os.path.isfile(resume_path):
            raise FileNotFoundError(f"Checkpoint file not found: '{resume_path}'")

        print(f"--> Resuming training from checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=target_device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        if "scheduler_state_dict" in checkpoint and scheduler is not None and checkpoint["scheduler_state_dict"] is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        # Gracefully load scaler state if available (does not crash on old checkpoints)
        if "scaler_state_dict" in checkpoint and scaler is not None and checkpoint["scaler_state_dict"] is not None:
            try:
                scaler.load_state_dict(checkpoint["scaler_state_dict"])
            except Exception as e:
                print(f"[Warning] Could not load scaler state dict: {e}")

        if "history" in checkpoint and checkpoint["history"] is not None:
            history = checkpoint["history"]
        print(f"--> Resumed at Epoch {start_epoch} (Best Val Loss so far: {best_val_loss:.6f})")

    # 5. Training Loop
    os.makedirs(checkpoint_dir, exist_ok=True)
    history_file = os.path.join(checkpoint_dir, "loss_history.json")

    print("\nStarting training loop...")
    for epoch in range(start_epoch, epochs + 1):
        epoch_start = time.time()

        # Training phase
        train_loss, train_skipped = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=target_device,
            use_amp=amp_active,
            dry_run=dry_run,
        )

        # Validation phase
        val_loss, val_skipped = validate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=target_device,
            use_amp=amp_active,
            dry_run=dry_run,
        )

        scheduler.step()
        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        # Track history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(current_lr)
        history.setdefault("skipped_batches", []).append(train_skipped)
        history.setdefault("val_skipped_batches", []).append(val_skipped)

        # Check if best model (ensure finite val_loss)
        is_best = (val_loss < best_val_loss) and not np.isnan(val_loss) and not np.isinf(val_loss)
        if is_best:
            best_val_loss = val_loss

        # Console logging
        best_marker = " (*Best)" if is_best else ""
        val_skip_str = f" (val_skipped:{val_skipped})" if val_skipped > 0 else ""
        print(
            f"Epoch [{epoch:03d}/{epochs:03d}] | "
            f"Train Loss ({loss_label}): {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f}{best_marker} | "
            f"skipped:{train_skipped}{val_skip_str} | "
            f"LR: {current_lr:.6f} | "
            f"Time: {epoch_time:.2f}s"
        )

        # Checkpoint saving
        should_save_periodic = (epoch % save_freq == 0) or (epoch == epochs)
        if should_save_periodic or is_best:
            checkpoint_state = {
                "epoch": epoch,
                "model_name": model_type,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict() if (scaler is not None and amp_active) else None,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_skipped": train_skipped,
                "val_skipped": val_skipped,
                "best_val_loss": best_val_loss,
                "history": history,
                "config": cfg,
                "loss_type": resolved_loss_type,
                "amp": amp_active,
            }
            if should_save_periodic:
                filename = f"checkpoint_epoch_{epoch:04d}.pth"
                save_checkpoint(
                    checkpoint_state,
                    checkpoint_dir=checkpoint_dir,
                    filename=filename,
                    is_best=is_best,
                )
                print(f"  [Checkpoint saved: {filename}]")
            elif is_best:
                # Save just best if not periodic
                save_checkpoint(
                    checkpoint_state,
                    checkpoint_dir=checkpoint_dir,
                    filename="latest_checkpoint.pth",
                    is_best=True,
                )
                print(f"  [New best checkpoint saved: best_checkpoint.pth]")

        # Persist history
        with open(history_file, "w") as f:
            json.dump(history, f, indent=2)

        if dry_run:
            print("\n[Dry Run] Single-batch step completed successfully.")
            break

    print("\nTraining completed.")
    print(f"Best Validation Loss: {best_val_loss:.6f}")
    print(f"Checkpoints directory: {checkpoint_dir}")
    return {"best_val_loss": best_val_loss, "history": history}


def main():
    args = parse_args()
    train(
        config_path=args.config,
        model_name=args.model,
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        save_freq=args.save_freq,
        resume=args.resume,
        num_workers=args.num_workers,
        device=args.device,
        seed=args.seed,
        val_split=args.val_split,
        val_count=args.val_count,
        layout=args.layout,
        dry_run=args.dry_run,
        base_channels=args.base_channels,
        loss_type=args.loss,
        use_amp=args.amp,
        crop_size=args.crop_size,
    )


if __name__ == "__main__":
    main()
