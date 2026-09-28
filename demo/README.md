# 🌅 RLUNet Single-Image HDR Reconstruction - Interactive Demo

This directory contains a web demo built with [Gradio](https://gradio.app/) for interactive single-image high-dynamic-range (HDR) reconstruction using the **RLUNet** architecture.

> **Proof of Concept Notice:**  
> *This is a proof of concept trained for a limited number of epochs on the HDR-Real dataset, so results vary by scene.*

---

## 🛠️ Overview of the Pipeline

1. **Upload & Preprocessing**:
   - The user uploads any standard photo (JPEG, PNG, etc.).
   - The image is resized to $512 \times 512$ pixels and normalized to float32 in $[0.0, 1.0]$.
2. **RLUNet Inference**:
   - The tensor $(1, 3, 512, 512)$ is passed through `RLUNet` on CPU under `torch.no_grad()`.
   - Pretrained weights are loaded from `models/checkpoints/rlunet_final.pth` (`model_state_dict`).
3. **Postprocessing & Tone Mapping**:
   - Predictions are clamped to $[0.0, 16.0]$.
   - The output is normalized by its own maximum radiance: $\hat{y}_{\text{norm}} = \frac{\hat{y}}{\max(\hat{y})}$.
   - The normalized output is tone-mapped with $\mu$-law compression ($\mu = 5000$):
     $$T(x) = \frac{\ln(1 + \mu x)}{\ln(1 + \mu)}$$
   - Both the $512 \times 512$ input and the tone-mapped reconstructed HDR output are displayed side-by-side.

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
To change the port or generate a temporary public Gradio URL:
```bash
python demo/app.py --port 7860 --share
```

### 3. Non-Blocking Terminal Test (Headless)
To verify that the model loads and the inference pipeline works without launching the web server:
```bash
python demo/app.py --test
```
