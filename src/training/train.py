"""Training script for BaselineUNet on HDRDataset."""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# Ensure project root is in sys.path for direct script execution
project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import yaml

from src.data.dataset import HDRDataset
from src.models.baseline_unet import BaselineUNet


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
        description="Train BaselineUNet for single-image HDR reconstruction."
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
        help="Fraction of dataset to use for validation (default: 0.10 for 90/10 split).",
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


def get_train_val_split(
    dataset: HDRDataset, val_split: float, seed: int
) -> Tuple[torch.utils.data.Dataset, Optional[torch.utils.data.Dataset]]:
    """Split dataset into 90% training and 10% validation subsets."""
    total_len = len(dataset)
    if total_len == 0:
        raise ValueError("Cannot split an empty dataset.")

    if total_len == 1:
        # Single sample edge-case: use for both or warn
        return dataset, dataset

    val_len = int(round(total_len * val_split))
    val_len = max(1, val_len)  # At least 1 validation sample if total_len > 1
    train_len = total_len - val_len

    if train_len == 0:
        train_len = 1
        val_len = total_len - 1

    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(dataset, [train_len, val_len], generator=generator)
    return train_set, val_set


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


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    dry_run: bool = False,
) -> float:
    """Run one epoch of training and return the average loss."""
    model.train()
    running_loss = 0.0
    total_samples = 0

    for batch_idx, (ldr_imgs, hdr_gts) in enumerate(dataloader):
        ldr_imgs = ldr_imgs.to(device, non_blocking=True)
        hdr_gts = hdr_gts.to(device, non_blocking=True)

        optimizer.zero_grad()
        predictions = model(ldr_imgs)
        loss = criterion(predictions, hdr_gts)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * ldr_imgs.size(0)
        total_samples += ldr_imgs.size(0)

        if dry_run:
            break

    return running_loss / max(1, total_samples)


def validate(
    model: nn.Module,
    dataloader: Optional[DataLoader],
    criterion: nn.Module,
    device: torch.device,
    dry_run: bool = False,
) -> float:
    """Run evaluation on the validation set and return average loss."""
    if dataloader is None or len(dataloader) == 0:
        return 0.0

    model.eval()
    running_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for ldr_imgs, hdr_gts in dataloader:
            ldr_imgs = ldr_imgs.to(device, non_blocking=True)
            hdr_gts = hdr_gts.to(device, non_blocking=True)

            predictions = model(ldr_imgs)
            loss = criterion(predictions, hdr_gts)

            running_loss += loss.item() * ldr_imgs.size(0)
            total_samples += ldr_imgs.size(0)

            if dry_run:
                break

    return running_loss / max(1, total_samples)


def train(
    config_path: Optional[str] = None,
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
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Main training routine."""
    cfg = load_config(config_path)

    # Resolve settings (CLI arguments take precedence over YAML config)
    seed = seed if seed is not None else cfg.get("seed", 42)
    set_seed(seed)

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

    print("=" * 65)
    print("RLUNet Project - BaselineUNet Training")
    print("=" * 65)
    print(f"Device:           {target_device}")
    print(f"Data Directory:   {data_dir}")
    print(f"Checkpoints:      {checkpoint_dir}")
    print(f"Total Epochs:     {epochs}")
    print(f"Batch Size:       {batch_size}")
    print(f"Learning Rate:    {lr}")
    print(f"Weight Decay:     {weight_decay}")
    print(f"Save Frequency:   Every {save_freq} epoch(s)")
    print(f"Num Workers:      {num_workers}")
    print(f"Random Seed:      {seed}")
    print("=" * 65)

    # 1. Prepare Dataset and 90/10 Train/Val Split
    full_dataset = HDRDataset(root_dir=data_dir)
    total_samples = len(full_dataset)
    print(f"Loaded dataset with {total_samples} total sample(s).")

    train_set, val_set = get_train_val_split(full_dataset, val_split=val_split, seed=seed)
    print(f"Dataset split (90/10): {len(train_set)} train, {len(val_set)} validation.")

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

    # 2. Instantiate Model (3-channel LDR in -> 3-channel HDR out, linear)
    base_channels = cfg.get("model", {}).get("base_channels", 64)
    model = BaselineUNet(
        in_channels=3,
        out_channels=3,
        base_channels=base_channels,
    ).to(target_device)

    # 3. Loss function & Optimizer
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    # Optional learning rate scheduler
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
        if "scheduler_state_dict" in checkpoint and scheduler is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if "history" in checkpoint:
            history = checkpoint["history"]
        print(f"--> Resumed at Epoch {start_epoch} (Best Val Loss so far: {best_val_loss:.6f})")

    # 5. Training Loop
    os.makedirs(checkpoint_dir, exist_ok=True)
    history_file = os.path.join(checkpoint_dir, "loss_history.json")

    print("\nStarting training loop...")
    for epoch in range(start_epoch, epochs + 1):
        epoch_start = time.time()

        # Training phase
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=target_device,
            dry_run=dry_run,
        )

        # Validation phase
        val_loss = validate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=target_device,
            dry_run=dry_run,
        )

        scheduler.step()
        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        # Track history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(current_lr)

        # Check if best model
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        # Console logging
        best_marker = " (*Best)" if is_best else ""
        print(
            f"Epoch [{epoch:03d}/{epochs:03d}] | "
            f"Train Loss (L1): {train_loss:.6f} | "
            f"Val Loss (L1): {val_loss:.6f}{best_marker} | "
            f"LR: {current_lr:.6f} | "
            f"Time: {epoch_time:.2f}s"
        )

        # Checkpoint saving
        should_save_periodic = (epoch % save_freq == 0) or (epoch == epochs)
        if should_save_periodic or is_best:
            checkpoint_state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "best_val_loss": best_val_loss,
                "history": history,
                "config": cfg,
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
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
