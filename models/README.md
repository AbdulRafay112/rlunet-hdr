# Models & Checkpoints

This directory is reserved for storing trained model weights, checkpoints, and exported artifacts (ONNX/TorchScript).

## Directory Structure
```text
models/
└── checkpoints/              # Model weights (.pth, .pt)
    ├── best_model.pth
    └── checkpoint_epoch_*.pth
```

> **Note:** `.pth`, `.pt`, and `.ckpt` files are ignored by git in `.gitignore`.
