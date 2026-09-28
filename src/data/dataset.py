"""PyTorch Dataset implementation for RLUNet HDR image reconstruction."""

import os
import warnings
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset


class HDRDataset(Dataset):
    """PyTorch Dataset for paired LDR and HDR reconstruction (e.g. HDR-Real).

    Supports two dataset layouts:
      1. 'folders' (default):
         Numbered subdirectories under `root_dir` (e.g., '00001/input.jpg' and '00001/gt.hdr').
      2. 'flat':
         Two parallel folders: '<root_dir>/LDR_in/<stem>.jpg' and '<root_dir>/HDR_gt/<stem>.hdr'.
         Pairs files by stem and ignores unpaired files.

    Features:
      - Split-aware mode:
        - 'train': applies joint augmentations (random square crop >= 256 padding if smaller,
          random horizontal/vertical flips, and random 90-degree rotations).
        - 'val' / 'test': evaluates on the full uncropped image without spatial distortion.
      - Per-image HDR max normalization:
        Scales HDR ground truth by its max radiance into [0.0, 1.0], returning the scale factor.

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
        layout: str = "folders",
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
            root_dir: Root path containing dataset files or folders.
            split: Dataset split ('train', 'val', or 'test'). Training enables augmentations.
            layout: Directory layout: 'folders' (default) or 'flat'.
            crop_size: Size of random square crop for training (default: 256).
            normalize_hdr: If True, normalizes HDR image by its per-image maximum (default: True).
            return_scale: If True, returns (ldr, hdr, scale_factor). If False, (ldr, hdr) (default: True).
            transform: Optional transform callable applied to the LDR tensor.
            target_transform: Optional transform callable applied to the HDR tensor.
            joint_transform: Optional callable applied jointly to (ldr_tensor, hdr_tensor).
            ldr_filename: Filename of the LDR image in subfolders for 'folders' layout (default: 'input.jpg').
            hdr_filename: Filename of the HDR image in subfolders for 'folders' layout (default: 'gt.hdr').
            check_missing: If True, filters out missing pairs at initialization.
        """
        super().__init__()
        self.root_dir = Path(root_dir)
        self.split = split.lower().strip()
        self.layout = layout.lower().strip()
        self.crop_size = crop_size
        self.normalize_hdr = normalize_hdr
        self.return_scale = return_scale
        self.transform = transform
        self.target_transform = target_transform
        self.joint_transform = joint_transform
        self.ldr_filename = ldr_filename
        self.hdr_filename = hdr_filename
        self.check_missing = check_missing

        if self.layout not in ("folders", "flat"):
            raise ValueError(
                f"Unsupported layout '{self.layout}'. Expected 'folders' or 'flat'."
            )

        if not self.root_dir.exists():
            raise FileNotFoundError(f"Root directory does not exist: '{self.root_dir}'")
        if not self.root_dir.is_dir():
            raise NotADirectoryError(f"Root path is not a directory: '{self.root_dir}'")

        # Scan for paired files according to chosen layout
        if self.layout == "flat":
            self.samples: List[Tuple[Path, Path]] = self._scan_flat_files()
        else:
            self.samples = self._scan_folder_samples()

        if len(self.samples) == 0:
            warnings.warn(
                f"No matching image pairs found in '{self.root_dir}' with layout='{self.layout}'."
            )

    def _find_subfolder(self, candidates: List[str]) -> Optional[Path]:
        """Find a subfolder matching one of the candidate names (case-insensitive)."""
        for cand in candidates:
            d = self.root_dir / cand
            if d.is_dir():
                return d
        if self.root_dir.is_dir():
            cand_lower = {c.lower(): c for c in candidates}
            for sub in self.root_dir.iterdir():
                if sub.is_dir() and sub.name.lower() in cand_lower:
                    return sub
        return None

    def _scan_flat_files(self) -> List[Tuple[Path, Path]]:
        """Scan flat directory layout: <root>/LDR_in/<stem>.jpg and <root>/HDR_gt/<stem>.hdr."""
        ldr_dir = self._find_subfolder(["LDR_in", "ldr_in", "LDR", "ldr", "input", "inputs"])
        hdr_dir = self._find_subfolder(["HDR_gt", "hdr_gt", "HDR", "hdr", "gt", "ground_truth"])

        if ldr_dir is None or not ldr_dir.is_dir():
            warnings.warn(f"Flat layout: LDR_in directory not found under '{self.root_dir}'.")
            return []
        if hdr_dir is None or not hdr_dir.is_dir():
            warnings.warn(f"Flat layout: HDR_gt directory not found under '{self.root_dir}'.")
            return []

        ldr_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        ldr_dict = {
            f.stem: f
            for f in ldr_dir.iterdir()
            if f.is_file() and f.suffix.lower() in ldr_exts
        }

        hdr_exts = {".hdr", ".exr", ".tif", ".tiff"}
        hdr_dict = {
            f.stem: f
            for f in hdr_dir.iterdir()
            if f.is_file() and f.suffix.lower() in hdr_exts
        }

        # Pair files by matching stem and ignore unpaired ones
        common_stems = sorted(
            list(set(ldr_dict.keys()) & set(hdr_dict.keys())),
            key=lambda s: (int(s) if s.isdigit() else float("inf"), s),
        )

        unpaired_ldr = len(ldr_dict) - len(common_stems)
        unpaired_hdr = len(hdr_dict) - len(common_stems)
        if unpaired_ldr > 0 or unpaired_hdr > 0:
            warnings.warn(
                f"Flat layout paired {len(common_stems)} matching stems. "
                f"Ignored {unpaired_ldr} unpaired LDR file(s) and {unpaired_hdr} unpaired HDR file(s)."
            )

        return [(ldr_dict[s], hdr_dict[s]) for s in common_stems]

    def _scan_folder_samples(self) -> List[Tuple[Path, Path]]:
        """Scan numbered subfolder layout: <root>/<folder>/input.jpg and gt.hdr."""
        subfolders = [
            d for d in self.root_dir.iterdir() if d.is_dir() and d.name.isdigit()
        ]

        if not subfolders:
            subfolders = [
                d for d in self.root_dir.rglob("*") if d.is_dir() and d.name.isdigit()
            ]

        subfolders.sort(
            key=lambda p: (int(p.name) if p.name.isdigit() else float("inf"), str(p))
        )

        pairs = []
        for s_dir in subfolders:
            ldr_p = s_dir / self.ldr_filename
            hdr_p = s_dir / self.hdr_filename
            if self.check_missing:
                if ldr_p.is_file() and hdr_p.is_file():
                    pairs.append((ldr_p, hdr_p))
                else:
                    warnings.warn(
                        f"Skipping sample directory '{s_dir}': missing required file(s) "
                        f"('{self.ldr_filename}' or '{self.hdr_filename}')."
                    )
            else:
                pairs.append((ldr_p, hdr_p))
        return pairs

    def __len__(self) -> int:
        return len(self.samples)

    def get_sample_path(self, index: int) -> Tuple[Path, Path]:
        """Get the (ldr_path, hdr_path) of the sample at the given index."""
        if index < 0 or index >= len(self.samples):
            raise IndexError(
                f"Index {index} out of range for dataset with {len(self.samples)} samples."
            )
        return self.samples[index]

    def _apply_joint_augmentations(
        self, ldr: torch.Tensor, hdr: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply random crop (padding if smaller), horizontal/vertical flips, and 90-degree rotations."""
        _, h, w = ldr.shape

        # 1. Random Crop to (crop_size, crop_size), padding if smaller
        if self.crop_size is not None and self.crop_size > 0:
            if h < self.crop_size or w < self.crop_size:
                pad_h = max(0, self.crop_size - h)
                pad_w = max(0, self.crop_size - w)
                pad_top = pad_h // 2
                pad_bottom = pad_h - pad_top
                pad_left = pad_w // 2
                pad_right = pad_w - pad_left
                # Replicate edge padding to prevent zero artifacts in radiance
                ldr = F.pad(
                    ldr.unsqueeze(0),
                    (pad_left, pad_right, pad_top, pad_bottom),
                    mode="replicate",
                ).squeeze(0)
                hdr = F.pad(
                    hdr.unsqueeze(0),
                    (pad_left, pad_right, pad_top, pad_bottom),
                    mode="replicate",
                ).squeeze(0)
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

    def __getitem__(
        self, index: int
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Load and return an LDR/HDR pair with optional scale factor.

        Args:
            index: Index of the sample to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor] (if return_scale=True)
            Tuple[torch.Tensor, torch.Tensor] (if return_scale=False)
        """
        ldr_path, hdr_path = self.get_sample_path(index)

        if not ldr_path.is_file():
            raise FileNotFoundError(f"LDR image file not found: '{ldr_path}'")
        if not hdr_path.is_file():
            raise FileNotFoundError(f"HDR ground truth file not found: '{hdr_path}'")

        # Load LDR image (uint8, 0-255)
        ldr_bgr = cv2.imread(str(ldr_path), cv2.IMREAD_COLOR)
        if ldr_bgr is None:
            raise IOError(f"Failed to read LDR image at '{ldr_path}'.")
        ldr_rgb = cv2.cvtColor(ldr_bgr, cv2.COLOR_BGR2RGB)

        # Load HDR image (float32, 0 to ~10000+)
        hdr_img = cv2.imread(str(hdr_path), cv2.IMREAD_ANYDEPTH)
        if hdr_img is None or hdr_img.ndim == 2:
            hdr_color = cv2.imread(
                str(hdr_path), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR
            )
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


def get_train_val_split(
    data_dir: Union[str, Path],
    layout: str = "folders",
    val_split: float = 0.10,
    val_count: Optional[int] = None,
    seed: int = 42,
    crop_size: int = 256,
    normalize_hdr: bool = True,
    return_scale: bool = True,
) -> Tuple[Subset, Optional[Subset]]:
    """Create two separate HDRDataset instances (train and val) and split with seeded indices.

    For 'flat' layout: uses a fixed seeded held-out validation set of val_count images
    (default: 300); the rest is used for training.
    For 'folders' layout: uses val_split fraction (default: 0.10) unless val_count is specified.

    Args:
        data_dir: Path to dataset root directory.
        layout: 'folders' or 'flat' layout.
        val_split: Fraction for validation when layout='folders' (default: 0.10).
        val_count: Exact number of held-out validation images for flat layout (default: 300).
        seed: Random seed for reproducible splitting (default: 42).
        crop_size: Random crop size for training (default: 256).
        normalize_hdr: Whether to normalize HDR by max value (default: True).
        return_scale: Whether dataset returns scale factors (default: True).

    Returns:
        Tuple[Subset, Subset]: (train_subset, val_subset).
    """
    train_dataset = HDRDataset(
        root_dir=data_dir,
        split="train",
        layout=layout,
        crop_size=crop_size,
        normalize_hdr=normalize_hdr,
        return_scale=return_scale,
    )
    val_dataset = HDRDataset(
        root_dir=data_dir,
        split="val",
        layout=layout,
        normalize_hdr=normalize_hdr,
        return_scale=return_scale,
    )

    total_len = len(train_dataset)
    if total_len == 0:
        return Subset(train_dataset, []), Subset(val_dataset, [])

    if total_len == 1:
        return Subset(train_dataset, [0]), Subset(val_dataset, [0])

    # Determine validation size
    if layout.lower().strip() == "flat":
        target_val = val_count if val_count is not None else 300
        val_len = min(target_val, total_len - 1)
        val_len = max(1, val_len)
    else:
        if val_count is not None and val_count > 0:
            val_len = min(val_count, total_len - 1)
            val_len = max(1, val_len)
        else:
            val_len = int(round(total_len * val_split))
            val_len = max(1, min(val_len, total_len - 1))

    train_len = total_len - val_len

    # Fixed seeded index permutation
    generator = torch.Generator().manual_seed(seed)
    shuffled_indices = torch.randperm(total_len, generator=generator).tolist()

    train_indices = shuffled_indices[:train_len]
    val_indices = shuffled_indices[train_len:]

    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)

    return train_subset, val_subset


if __name__ == "__main__":
    sample_root = Path("data/HDR-Real/HDR-Real")
    print("--- Testing HDRDataset Implementation ---")
    print(f"Sample root path: {sample_root}")
    if sample_root.exists() and len(list(sample_root.iterdir())) > 0:
        ds_train = HDRDataset(sample_root, split="train", layout="folders", crop_size=256)
        ds_val = HDRDataset(sample_root, split="val", layout="folders")
        print(f"Samples found: {len(ds_train)}")
        ldr_tr, hdr_tr, scale_tr = ds_train[0]
        print(f"Train item [0] -> LDR: {ldr_tr.shape}, HDR: {hdr_tr.shape}, Scale: {scale_tr.item():.4f}")
        ldr_val, hdr_val, scale_val = ds_val[0]
        print(f"Val item [0]   -> LDR: {ldr_val.shape}, HDR: {hdr_val.shape}, Scale: {scale_val.item():.4f}")
    else:
        print("[Notice] Real dataset not found on disk; unit tests will verify with synthetic data.")
