"""PyTorch Dataset implementation for RLUNet HDR image reconstruction."""

import os
import warnings
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class HDRDataset(Dataset):
    """PyTorch Dataset for paired LDR and HDR reconstruction (e.g. HDR-Real).

    Each sample directory under `root_dir` is expected to be a numbered folder
    (e.g., '00001', '00205') containing:
      - `input.jpg`: Low Dynamic Range (LDR) input (uint8, [0, 255])
      - `gt.hdr`: High Dynamic Range (HDR) ground truth (float32, [0, ~10000+])

    Returns:
      tuple: (ldr_tensor, hdr_tensor)
        - `ldr_tensor`: FloatTensor of shape (3, H, W) normalized to [0.0, 1.0].
        - `hdr_tensor`: FloatTensor of shape (3, H, W) with original physical radiance.
    """

    def __init__(
        self,
        root_dir: Union[str, Path],
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
            transform: Optional transform callable applied to the LDR tensor.
            target_transform: Optional transform callable applied to the HDR tensor.
            joint_transform: Optional callable applied jointly to (ldr_tensor, hdr_tensor).
            ldr_filename: Filename of the LDR image in each folder (default: 'input.jpg').
            hdr_filename: Filename of the HDR image in each folder (default: 'gt.hdr').
            check_missing: If True, filters out subfolders missing input/gt files at init.
                           If False, verifies files on demand during __getitem__.
        """
        super().__init__()
        self.root_dir = Path(root_dir)
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
        # Check direct subfolders first
        subfolders = [
            d for d in self.root_dir.iterdir() if d.is_dir() and d.name.isdigit()
        ]

        # If no direct numbered folders found, search recursively
        if not subfolders:
            subfolders = [
                d for d in self.root_dir.rglob("*") if d.is_dir() and d.name.isdigit()
            ]

        # Sort numerically by folder name, with fallback to alphabetical string sort
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

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load and return an (ldr_tensor, hdr_tensor) pair.

        Args:
            index: Index of the sample to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - ldr_tensor: torch.FloatTensor of shape (3, H, W) normalized to [0, 1].
                - hdr_tensor: torch.FloatTensor of shape (3, H, W) in float32 radiance values.

        Raises:
            FileNotFoundError: If input.jpg or gt.hdr is missing in the sample folder.
            IOError: If OpenCV fails to read/decode either image.
        """
        sample_dir = self.get_sample_path(index)
        ldr_path = sample_dir / self.ldr_filename
        hdr_path = sample_dir / self.hdr_filename

        # Basic error handling for missing files
        if not ldr_path.is_file():
            raise FileNotFoundError(
                f"LDR image file '{self.ldr_filename}' not found in sample directory: '{sample_dir}'"
            )
        if not hdr_path.is_file():
            raise FileNotFoundError(
                f"HDR ground truth file '{self.hdr_filename}' not found in sample directory: '{sample_dir}'"
            )

        # Load LDR image (uint8, 0-255)
        ldr_bgr = cv2.imread(str(ldr_path), cv2.IMREAD_COLOR)
        if ldr_bgr is None:
            raise IOError(
                f"Failed to read LDR image at '{ldr_path}'. The file may be corrupted or unreadable."
            )
        # Convert BGR to RGB
        ldr_rgb = cv2.cvtColor(ldr_bgr, cv2.COLOR_BGR2RGB)

        # Load HDR image (float32, 0 to ~10000+)
        # cv2.IMREAD_ANYDEPTH preserves 32-bit floating-point depth.
        # Combined with IMREAD_COLOR to ensure 3 color channels are read.
        hdr_img = cv2.imread(str(hdr_path), cv2.IMREAD_ANYDEPTH)
        if hdr_img is None or hdr_img.ndim == 2:
            hdr_color = cv2.imread(
                str(hdr_path), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR
            )
            if hdr_color is not None:
                hdr_img = hdr_color

        if hdr_img is None:
            raise IOError(
                f"Failed to read HDR ground truth at '{hdr_path}'. The file may be corrupted or unreadable."
            )

        # Ensure 3 channels and convert BGR to RGB
        if hdr_img.ndim == 2:
            hdr_img = np.stack([hdr_img] * 3, axis=-1)
        elif hdr_img.ndim == 3 and hdr_img.shape[2] == 3:
            hdr_img = cv2.cvtColor(hdr_img, cv2.COLOR_BGR2RGB)

        # Convert LDR to float32 tensor normalized to [0, 1] with shape (3, H, W)
        ldr_tensor = (
            torch.from_numpy(ldr_rgb.astype(np.float32) / 255.0)
            .permute(2, 0, 1)
            .contiguous()
        )

        # Convert HDR to float32 tensor with shape (3, H, W)
        hdr_tensor = (
            torch.from_numpy(hdr_img.astype(np.float32))
            .permute(2, 0, 1)
            .contiguous()
        )

        # Apply transformations if provided
        if self.transform is not None:
            ldr_tensor = self.transform(ldr_tensor)
        if self.target_transform is not None:
            hdr_tensor = self.target_transform(hdr_tensor)
        if self.joint_transform is not None:
            ldr_tensor, hdr_tensor = self.joint_transform(ldr_tensor, hdr_tensor)

        return ldr_tensor, hdr_tensor


# Alias for convenience and project naming convention
RLUNetDataset = HDRDataset

if __name__ == "__main__":
    from torch.utils.data import DataLoader

    sample_root = Path("data/HDR-Real/HDR-Real")
    if not sample_root.exists():
        sample_root = Path(__file__).resolve().parents[2] / "data" / "HDR-Real" / "HDR-Real"

    print(f"--- Running Usage Example for HDRDataset ---")
    print(f"Loading HDRDataset from: {sample_root}")
    dataset = HDRDataset(root_dir=sample_root)

    print(f"Total samples found: {len(dataset)}")
    for idx in range(len(dataset)):
        print(f"  Sample {idx}: {dataset.get_sample_path(idx)}")

    ldr_tensor, hdr_tensor = dataset[0]
    print(
        f"\nSample [0] Details:"
        f"\n  LDR Tensor -> shape: {ldr_tensor.shape}, dtype: {ldr_tensor.dtype}, range: [{ldr_tensor.min().item():.4f}, {ldr_tensor.max().item():.4f}]"
        f"\n  HDR Tensor -> shape: {hdr_tensor.shape}, dtype: {hdr_tensor.dtype}, range: [{hdr_tensor.min().item():.4f}, {hdr_tensor.max().item():.4f}]"
    )

    # DataLoader test
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
    batch_ldr, batch_hdr = next(iter(dataloader))
    print(
        f"\nDataLoader (batch_size=2):"
        f"\n  Batch LDR Tensor -> shape: {batch_ldr.shape}, dtype: {batch_ldr.dtype}"
        f"\n  Batch HDR Tensor -> shape: {batch_hdr.shape}, dtype: {batch_hdr.dtype}"
    )
    print("\n[SUCCESS] HDRDataset usage example executed successfully!")

