# RLUNet: HDR Image Reconstruction

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A Deep Learning research framework for High Dynamic Range (HDR) image reconstruction based on RLUNet architecture.

---

## 📌 Project Overview
- **Objective:** High Dynamic Range (HDR) image synthesis and reconstruction from multi-exposure low-dynamic-range (LDR) inputs or single-image HDR reconstruction.
- **Model Architecture:** RLUNet (placeholder for architecture summary, attention mechanisms, and feature aggregation details).
- **Domain:** Computational Photography, Deep Learning, Computer Vision.

---

## 📁 Repository Structure

```text
spm_project/
├── data/                  # Dataset storage and preparation scripts
│   ├── raw/               # Original, unprocessed datasets (e.g., multi-exposure LDRs)
│   └── processed/         # Preprocessed patches, aligned images, or tensors
├── experiments/           # Experiment configuration files and logs
│   └── configs/           # YAML / JSON training & evaluation hyperparameter configs
├── models/                # Saved weights, model checkpoints, and exported artifacts
│   └── checkpoints/       # Best and periodic checkpoint files (.pth, .pt)
├── notebooks/             # Jupyter notebooks for EDA, visual inspections, and prototyping
├── results/               # Generated evaluation outputs, figures, and benchmark logs
│   ├── figures/           # Visual comparison plots and sample outputs
│   └── metrics/           # Quantitative scores (PSNR, SSIM, HDR-VDP-2)
├── src/                   # Core Python package source code
│   ├── data/              # Dataset loaders, pipelines, and augmentation logic
│   ├── models/            # RLUNet model definition, layers, and loss functions
│   ├── training/          # Training loops, validation routines, and optimizers
│   └── utils/             # Metric calculations, HDR I/O (.hdr, .exr), and visualization
├── .gitignore             # Git ignore patterns for DL projects
├── README.md              # Project documentation and guide
└── requirements.txt       # Environment dependencies
```

---

## ⚙️ Installation & Setup

### Prerequisites
- Python >= 3.9
- CUDA-compatible GPU (recommended) & cuDNN

### Environment Setup
```bash
# Clone the repository
git clone https://github.com/your-username/RLUNet.git
cd RLUNet

# Create and activate virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
# source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 📊 Dataset Preparation

Organize your HDR datasets under the `data/` folder:

```text
data/
├── raw/
│   ├── Kalantari17/
│   └── SICE/
└── processed/
```

- Refer to [data/README.md](file:///C:/spm_project/data/README.md) for details on downloading benchmarks and preprocessing inputs.

---

## 🚀 Training

To train the RLUNet model using default configuration:

```bash
# Example training command placeholder
python -m src.training.train --config experiments/configs/default_config.yaml
```

---

## 🧪 Evaluation & Inference

Evaluate trained checkpoints on test benchmarks:

```bash
# Example evaluation command placeholder
python -m src.training.evaluate --checkpoint models/checkpoints/best_model.pth --config experiments/configs/eval_config.yaml
```

---

## 📈 Benchmarks & Results

| Dataset | PSNR-L (dB) | PSNR-μ (dB) | SSIM-L | SSIM-μ | HDR-VDP-2 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| *Benchmark 1 (e.g., Kalantari)* | TBD | TBD | TBD | TBD | TBD |
| *Benchmark 2 (e.g., SICE)* | TBD | TBD | TBD | TBD | TBD |

---

## 📜 Citation & License

- **License:** [MIT License](LICENSE) (or specify chosen license).
- **Citation:**
```bibtex
@article{rlunet_hdr,
  title={RLUNet: Deep Learning for High Dynamic Range Image Reconstruction},
  author={Author Name},
  year={2026}
}
```
