"""PyTorch Dataset implementation for RLUNet HDR image reconstruction."""

import os
import warnings
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class HDRDataset(Dataset):
    """PyTorch Dataset for paired LDR and HDR reconstruction (e.g. HDR-Real).

    Each sample directory under `root_dir` is expected to be a numbered folder
    (e.g., '00001', '00205') containing:
      - `input.jpg`: Low Dynamic Range (LDR) input (uint8, [0, 255])
      - `gt.hdr`: High Dynamic Range (HDR) ground truth (float32, [0, ~10000+])

    Features:
      - Split-aware mode: 'train' applies joint augmentations (random 256x256 crops,
        random horizontal/vertical flips, and random 90-degree rotations);
        'val' / 'test' evaluates on the full uncropped image without spatial distortion.
      - Per-image HDR max normalization: scales HDR ground truth by its max radiance
        into [0.0, 1.0], returning the scale factor for unscaled reconstruction.

    Returns:
      Tuple[torch.Tensor, torch.Tensor] or Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        - `ldr_tensor`: FloatTensor of shape (3, H, W) normalized to [0.0, 1.0].
        - `hdr_tensor`: FloatTensor of shape (3, H, W) (normalized if normalize_hdr=True).
        - (Optional) `scale_factor`: FloatTensor scalar if return_scale=True.
    """

    def __init__(
        self,
        root_dir: Union[str, Path],
        split: str = "train",
        crop_size: int = 256,
        normalize_hdr: bool = True,
        return_scale: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        joint_transform: Optional[Callable] = None,
        ldr_filename: str = "input.jpg",
        hdr_filename: str = "gt.hdr",
        check_missing: bool = False,
    ):
        """Initialize the HDR dataset.

        Args:
            root_dir: Root path containing numbered sample subfolders (e.g., 'data/HDR-Real/HDR-Real').
            split: Dataset split ('train', 'val', or 'test'). Training enables augmentations.
            crop_size: Size of random square crop for training (default: 256).
            normalize_hdr: If True, normalizes HDR image by its per-image maximum (default: True).
            return_scale: If True, returns (ldr, hdr, scale_factor). If False, (ldr, hdr) (default: True).
            transform: Optional transform callable applied to the LDR tensor.
            target_transform: Optional transform callable applied to the HDR tensor.
            joint_transform: Optional callable applied jointly to (ldr_tensor, hdr_tensor).
            ldr_filename: Filename of the LDR image in each folder (default: 'input.jpg').
            hdr_filename: Filename of the HDR image in each folder (default: 'gt.hdr').
            check_missing: If True, filters out subfolders missing input/gt files at init.
        """
        super().__init__()
        self.root_dir = Path(root_dir)
        self.split = split.lower().strip()
        self.crop_size = crop_size
        self.normalize_hdr = normalize_hdr
        self.return_scale = return_scale
        self.transform = transform
        self.target_transform = target_transform
        self.joint_transform = joint_transform
        self.ldr_filename = ldr_filename
        self.hdr_filename = hdr_filename

        if not self.root_dir.exists():
            raise FileNotFoundError(f"Root directory does not exist: '{self.root_dir}'")
        if not self.root_dir.is_dir():
            raise NotADirectoryError(f"Root path is not a directory: '{self.root_dir}'")

        # Scan for numbered sample subfolders
        self.sample_dirs = self._scan_sample_folders()

        if len(self.sample_dirs) == 0:
            warnings.warn(
                f"No numbered sample subfolders found in root directory '{self.root_dir}'."
            )

        if check_missing:
            valid_dirs = []
            for s_dir in self.sample_dirs:
                ldr_p = s_dir / self.ldr_filename
                hdr_p = s_dir / self.hdr_filename
                if ldr_p.is_file() and hdr_p.is_file():
                    valid_dirs.append(s_dir)
                else:
                    warnings.warn(
                        f"Skipping sample directory '{s_dir}': missing required file(s) "
                        f"('{self.ldr_filename}' or '{self.hdr_filename}')."
                    )
            self.sample_dirs = valid_dirs

    def _scan_sample_folders(self) -> List[Path]:
        """Scan root_dir for numbered subfolders and sort them numerically."""
        subfolders = [
            d for d in self.root_dir.iterdir() if d.is_dir() and d.name.isdigit()
        ]

        if not subfolders:
            subfolders = [
                d for d in self.root_dir.rglob("*") if d.is_dir() and d.name.isdigit()
            ]

        subfolders.sort(
            key=lambda p: (int(p.name) if p.name.isdigit() else p.name, str(p))
        )
        return subfolders

    @property
    def samples(self) -> List[str]:
        """Return list of sample directory paths as strings."""
        return [str(p) for p in self.sample_dirs]

    def __len__(self) -> int:
        return len(self.sample_dirs)

    def get_sample_path(self, index: int) -> Path:
        """Get the directory path of the sample at the given index."""
        if index < 0 or index >= len(self.sample_dirs):
            raise IndexError(
                f"Index {index} out of range for dataset with {len(self.sample_dirs)} samples."
            )
        return self.sample_dirs[index]

    def _apply_joint_augmentations(
        self, ldr: torch.Tensor, hdr: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply random crop, horizontal/vertical flips, and 90-degree rotations jointly."""
        _, h, w = ldr.shape

        # 1. Random Crop to (crop_size, crop_size)
        if self.crop_size is not None and self.crop_size > 0:
            pad_h = max(0, self.crop_size - h)
            pad_w = max(0, self.crop_size - w)
            if pad_h > 0 or pad_w > 0:
                # Pad to at least crop_size with reflection (or replicate if too small)
                pad_mode = "reflect" if (pad_h < h and pad_w < w) else "replicate"
                padding = [0, pad_w, 0, pad_h]
                ldr = F.pad(ldr.unsqueeze(0), padding, mode=pad_mode).squeeze(0)
                hdr = F.pad(hdr.unsqueeze(0), padding, mode=pad_mode).squeeze(0)
                _, h, w = ldr.shape

            top = int(torch.randint(0, h - self.crop_size + 1, (1,)).item())
            left = int(torch.randint(0, w - self.crop_size + 1, (1,)).item())
            ldr = ldr[:, top : top + self.crop_size, left : left + self.crop_size]
            hdr = hdr[:, top : top + self.crop_size, left : left + self.crop_size]

        # 2. Random Horizontal Flip (50% prob)
        if torch.rand(1).item() > 0.5:
            ldr = torch.flip(ldr, dims=[2])
            hdr = torch.flip(hdr, dims=[2])

        # 3. Random Vertical Flip (50% prob)
        if torch.rand(1).item() > 0.5:
            ldr = torch.flip(ldr, dims=[1])
            hdr = torch.flip(hdr, dims=[1])

        # 4. Random 90-degree rotation (0, 90, 180, or 270 degrees)
        rot_k = int(torch.randint(0, 4, (1,)).item())
        if rot_k > 0:
            ldr = torch.rot90(ldr, rot_k, dims=[1, 2])
            hdr = torch.rot90(hdr, rot_k, dims=[1, 2])

        return ldr, hdr

    def __getitem__(self, index: int) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Load and return an LDR/HDR pair with optional scale factor.

        Args:
            index: Index of the sample to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor] (if return_scale=True)
            Tuple[torch.Tensor, torch.Tensor] (if return_scale=False)
        """
        sample_dir = self.get_sample_path(index)
        ldr_path = sample_dir / self.ldr_filename
        hdr_path = sample_dir / self.hdr_filename

        if not ldr_path.is_file():
            raise FileNotFoundError(
                f"LDR image file '{self.ldr_filename}' not found in: '{sample_dir}'"
            )
        if not hdr_path.is_file():
            raise FileNotFoundError(
                f"HDR ground truth file '{self.hdr_filename}' not found in: '{sample_dir}'"
            )

        # Load LDR image (uint8, 0-255)
        ldr_bgr = cv2.imread(str(ldr_path), cv2.IMREAD_COLOR)
        if ldr_bgr is None:
            raise IOError(f"Failed to read LDR image at '{ldr_path}'.")
        ldr_rgb = cv2.cvtColor(ldr_bgr, cv2.COLOR_BGR2RGB)

        # Load HDR image (float32, 0 to ~10000+)
        hdr_img = cv2.imread(str(hdr_path), cv2.IMREAD_ANYDEPTH)
        if hdr_img is None or hdr_img.ndim == 2:
            hdr_color = cv2.imread(str(hdr_path), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
            if hdr_color is not None:
                hdr_img = hdr_color

        if hdr_img is None:
            raise IOError(f"Failed to read HDR ground truth at '{hdr_path}'.")

        if hdr_img.ndim == 2:
            hdr_img = np.stack([hdr_img] * 3, axis=-1)
        elif hdr_img.ndim == 3 and hdr_img.shape[2] == 3:
            hdr_img = cv2.cvtColor(hdr_img, cv2.COLOR_BGR2RGB)

        # Convert to float32 tensors with shape (3, H, W)
        ldr_tensor = (
            torch.from_numpy(ldr_rgb.astype(np.float32) / 255.0)
            .permute(2, 0, 1)
            .contiguous()
        )
        hdr_tensor = (
            torch.from_numpy(hdr_img.astype(np.float32))
            .permute(2, 0, 1)
            .contiguous()
        )

        # Apply training augmentations if in training split
        if self.split == "train":
            ldr_tensor, hdr_tensor = self._apply_joint_augmentations(ldr_tensor, hdr_tensor)

        # Per-image HDR max normalization
        if self.normalize_hdr:
            max_val = hdr_tensor.max().item()
            scale = max(float(max_val), 1e-6)
            hdr_tensor = hdr_tensor / scale
            scale_tensor = torch.tensor(scale, dtype=torch.float32)
        else:
            scale_tensor = torch.tensor(1.0, dtype=torch.float32)

        # Apply optional custom transforms
        if self.transform is not None:
            ldr_tensor = self.transform(ldr_tensor)
        if self.target_transform is not None:
            hdr_tensor = self.target_transform(hdr_tensor)
        if self.joint_transform is not None:
            ldr_tensor, hdr_tensor = self.joint_transform(ldr_tensor, hdr_tensor)

        if self.return_scale:
            return ldr_tensor, hdr_tensor, scale_tensor
        return ldr_tensor, hdr_tensor


# Alias for convenience
RLUNetDataset = HDRDataset


if __name__ == "__main__":
    from torch.utils.data import DataLoader, Subset

    sample_root = Path("data/HDR-Real/HDR-Real")
    if not sample_root.exists():
        sample_root = Path(__file__).resolve().parents[2] / "data" / "HDR-Real" / "HDR-Real"

    print("--- Testing HDRDataset Implementation ---")
    print(f"Sample root path: {sample_root}")
    if sample_root.exists() and len(list(sample_root.iterdir())) > 0:
        ds_train = HDRDataset(sample_root, split="train", crop_size=256, normalize_hdr=True)
        ds_val = HDRDataset(sample_root, split="val", normalize_hdr=True)
        print(f"Samples found: {len(ds_train)}")
        ldr_tr, hdr_tr, scale_tr = ds_train[0]
        print(f"Train sample 0 -> LDR: {ldr_tr.shape}, HDR: {hdr_tr.shape}, Scale: {scale_tr.item():.4f}")
        ldr_val, hdr_val, scale_val = ds_val[0]
        print(f"Val sample 0   -> LDR: {ldr_val.shape}, HDR: {hdr_val.shape}, Scale: {scale_val.item():.4f}")
    else:
        print("[Notice] Real dataset not found on disk; unit tests will verify with synthetic data.")
