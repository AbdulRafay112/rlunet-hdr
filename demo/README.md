# 🌅 RLUNet Single-Image HDR Reconstruction - Interactive Demo

This directory contains an interactive web application built with [Gradio](https://gradio.app/) for single-image high-dynamic-range (HDR) reconstruction using the **RLUNet** architecture.

> **Proof of Concept Notice:**  
> *This model is a proof of concept trained for a limited number of epochs on the HDR-Real dataset, so results vary by scene.*

---

## 🛠️ Overview of the Pipeline

1. **Upload & Preprocessing**:
   - The user uploads any standard photo (JPEG, PNG, etc.).
   - The image is resized to $512 \times 512$ pixels and normalized to float32 in $[0.0, 1.0]$.
2. **RLUNet Inference**:
   - The input tensor $(1, 3, 512, 512)$ is processed by `RLUNet` on CPU under `torch.no_grad()`.
   - Pretrained weights are loaded from `models/checkpoints/rlunet_final.pth` (`model_state_dict`).
   - Predicted raw HDR values are clamped to $[0.0, 16.0]$ and cached in memory.
3. **Selectable Display Tone-Mapping Modes**:
   - **`Natural (gamma)` (Default)**: Normalizes the predicted HDR by the 99th percentile of its per-pixel luminance ($Y = 0.2126 R + 0.7152 G + 0.0722 B$), clamps to $[0.0, 1.0]$, and applies standard display gamma correction ($\gamma = 1/2.2$).
   - **`Mu-law (as in evaluation)`**: Normalizes the predicted HDR by its maximum radiance, then applies $\mu$-law compression ($\mu = 5000$):
     $$T(x) = \frac{\ln(1 + \mu x)}{\ln(1 + \mu)}$$
   - *Note: Switching the display mode only alters how the HDR radiance is tone-mapped and shown on standard SDR displays, without re-running the neural network inference.*

---

## 🚀 Installation & Setup

1. **Activate the virtual environment**:
   - **Windows**:
     ```powershell
     .\venv\Scripts\activate
     ```
   - **Linux / macOS**:
     ```bash
     source venv/bin/activate
     ```

2. **Install requirements**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify Checkpoint**:
   Ensure `models/checkpoints/rlunet_final.pth` is present.

---

## 🖥️ Exact Run Commands

### 1. Launch the Gradio Web Server
Run the application on the local server (default: `http://127.0.0.1:7860`):
```bash
python demo/app.py
```

### 2. Custom Port or Public Shareable Link
To change the port or generate a temporary public Gradio link:
```bash
python demo/app.py --port 7860 --share
```

### 3. Non-Blocking Terminal Test (Headless)
To verify that model loading, inference, and both display modes run without starting the web server:
```bash
python demo/app.py --test
```
