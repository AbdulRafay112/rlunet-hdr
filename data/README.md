# Dataset Directory

Store datasets used for training and evaluating RLUNet here.

## Recommended Structure
```text
data/
├── raw/                      # Downloaded benchmark datasets (unmodified)
│   ├── Kalantari17/
│   │   ├── Training/
│   │   └── Test/
│   └── SICE/
└── processed/                # Pre-cropped patches, normalized numpy arrays / tensors
    ├── train_patches/
    └── val_patches/
```

> **Note:** Raw and processed dataset files (e.g., `.hdr`, `.exr`, `.npy`, `.h5`) are ignored by version control.
