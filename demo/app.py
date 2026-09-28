"""RLUNet Single-Image HDR Reconstruction - Interactive Gradio Demo.

Loads trained RLUNet weights from models/checkpoints/rlunet_final.pth on CPU,
accepts an uploaded photo, resizes it to 512x512, normalizes to [0, 1], runs
RLUNet inference under torch.no_grad, clamps output to [0, 16], and displays the
reconstructed HDR image using selectable display tone-mapping modes:
  1. 'Natural (gamma)' (default): normalized by 99th percentile of per-pixel luminance,
     clamped to [0, 1], and gamma-corrected (1/2.2).
  2. 'Mu-law (as in evaluation)': normalized by max radiance, tone-mapped with mu=5000.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Tuple, Union

# Ensure project root is in sys.path
project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import cv2
import gradio as gr
import numpy as np
import torch

from src.models.rlunet import RLUNet
from src.utils.tonemap import mu_law_tonemap

DEFAULT_CHECKPOINT = os.path.join(project_root, "models", "checkpoints", "rlunet_final.pth")
MODE_NATURAL = "Natural (gamma)"
MODE_MULAW = "Mu-law (as in evaluation)"


def load_rlunet_model(checkpoint_path: str = DEFAULT_CHECKPOINT) -> RLUNet:
    """Load RLUNet model on CPU with trained checkpoint weights."""
    model = RLUNet(in_channels=3, out_channels=3, base_channels=32)
    if os.path.isfile(checkpoint_path):
        print(f"Loading checkpoint weights from '{checkpoint_path}' on CPU...")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = (
            checkpoint["model_state_dict"]
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
            else checkpoint
        )
        model.load_state_dict(state_dict)
        print("Model weights loaded successfully.")
    else:
        print(f"[Warning] Checkpoint not found at '{checkpoint_path}'. Initializing random weights.")
    model.eval()
    return model


# Global model instance on CPU
model = load_rlunet_model()


def apply_display_mode(
    raw_hdr: np.ndarray,
    mode: str = MODE_NATURAL,
) -> Tuple[np.ndarray, str]:
    """Convert raw predicted HDR radiance [0, 16] to a displayable uint8 RGB image.

    Modes:
      - 'Natural (gamma)':
          Normalize by 99th percentile of per-pixel luminance, clamp to [0, 1],
          and apply gamma correction (1/2.2).
      - 'Mu-law (as in evaluation)':
          Normalize by max radiance, apply mu-law tonemapping (mu=5000).

    Note:
      This function only alters how the HDR radiance is tone-mapped and displayed,
      not the underlying raw predicted HDR values.
    """
    if raw_hdr is None:
        return None, ""

    if mode == MODE_NATURAL:
        # Per-pixel luminance using Rec.709 coefficients
        lum = (
            0.2126 * raw_hdr[..., 0]
            + 0.7152 * raw_hdr[..., 1]
            + 0.0722 * raw_hdr[..., 2]
        )
        p99 = float(np.percentile(lum, 99))
        scale = max(p99, 1e-6)
        norm = np.clip(raw_hdr / scale, 0.0, 1.0)
        gamma = np.power(norm, 1.0 / 2.2)
        display_uint8 = np.clip(gamma * 255.0, 0, 255).astype(np.uint8)
        details = (
            f"Display: Natural (gamma 1/2.2) | 99th% Luminance Scale: {scale:.4f} | "
            f"Max Radiance: {float(raw_hdr.max()):.2f}"
        )
    else:  # "Mu-law (as in evaluation)"
        max_val = float(raw_hdr.max())
        scale = max(max_val, 1e-6)
        norm = raw_hdr / scale
        tm = mu_law_tonemap(norm, mu=5000.0)
        display_uint8 = np.clip(tm * 255.0, 0, 255).astype(np.uint8)
        details = (
            f"Display: Mu-law (mu=5000, normalized by max) | "
            f"Max Radiance: {scale:.2f}"
        )

    return display_uint8, details


def reconstruct_hdr(
    input_image: Optional[np.ndarray],
    display_mode: str = MODE_NATURAL,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str, Optional[np.ndarray]]:
    """Reconstruct HDR image from uploaded LDR photo and apply selected display mode.

    Returns:
      (resized_input, display_output, info_text, raw_hdr_array)
    """
    if input_image is None:
        return None, None, "Please upload an image to begin reconstruction.", None

    # Ensure numpy array
    if not isinstance(input_image, np.ndarray):
        input_image = np.array(input_image)

    # Handle grayscale or RGBA channels
    if input_image.ndim == 2:
        input_image = np.stack([input_image] * 3, axis=-1)
    elif input_image.ndim == 3 and input_image.shape[2] == 4:
        input_image = input_image[:, :, :3]

    # Handle float inputs
    if input_image.dtype != np.uint8:
        input_image = np.clip(input_image * 255.0, 0, 255).astype(np.uint8)

    orig_h, orig_w, _ = input_image.shape

    # 1. Resize to 512x512
    interp = cv2.INTER_AREA if (orig_h > 512 or orig_w > 512) else cv2.INTER_LINEAR
    resized_input = cv2.resize(input_image, (512, 512), interpolation=interp)

    # 2. Convert to float [0, 1]
    img_float = resized_input.astype(np.float32) / 255.0

    # 3. Create torch tensor: shape (1, 3, 512, 512)
    tensor_in = torch.from_numpy(img_float).permute(2, 0, 1).unsqueeze(0)

    # 4. Run RLUNet in eval mode under no_grad
    with torch.no_grad():
        output = model(tensor_in)

    # 5. Clamp output to [0, 16]
    output_clamped = torch.clamp(output.float(), min=0.0, max=16.0)

    # Store raw HDR predictions in numpy array: shape (512, 512, 3)
    raw_hdr = output_clamped.squeeze(0).permute(1, 2, 0).cpu().numpy()

    # 6. Apply display tone mapping
    display_img, mode_details = apply_display_mode(raw_hdr, mode=display_mode)

    info_text = (
        f"Input: {orig_w}x{orig_h} (processed at 512x512) | {mode_details}"
    )
    return resized_input, display_img, info_text, raw_hdr


def on_display_mode_change(
    raw_hdr: Optional[np.ndarray],
    display_mode: str,
) -> Tuple[Optional[np.ndarray], str]:
    """Switch display mode on cached raw HDR predictions without re-running the model."""
    if raw_hdr is None:
        return None, "Upload and reconstruct an image first."

    display_img, mode_details = apply_display_mode(raw_hdr, mode=display_mode)
    info_text = f"Display mode updated (cached prediction) | {mode_details}"
    return display_img, info_text


def create_demo() -> gr.Blocks:
    """Build the Gradio interface."""
    with gr.Blocks(title="RLUNet HDR Reconstruction Demo") as demo:
        gr.Markdown(
            """
            # 🌅 RLUNet: Single-Image HDR Reconstruction
            Upload any standard photo (LDR) to estimate its high-dynamic-range (HDR) radiance map.
            
            > **Proof of Concept Notice:**  
            > *This is a proof of concept trained for a limited number of epochs on HDR-Real, so results vary by scene.*
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(
                    label="Upload Photo (LDR)",
                    type="numpy",
                )
                display_mode = gr.Radio(
                    choices=[MODE_NATURAL, MODE_MULAW],
                    value=MODE_NATURAL,
                    label="Display Mode",
                    info="Note: Display mode only changes how the output is shown, not what the model predicts.",
                )
                reconstruct_btn = gr.Button("Reconstruct HDR", variant="primary", size="lg")
                info_output = gr.Textbox(label="Inference & Display Details", interactive=False)

            with gr.Column(scale=2):
                with gr.Row():
                    preview_input = gr.Image(
                        label="Input Image (512x512)",
                        interactive=False,
                    )
                    preview_output = gr.Image(
                        label="Reconstructed HDR",
                        interactive=False,
                    )

        # Gradio state to hold raw predicted HDR radiance without re-running inference
        raw_hdr_state = gr.State(None)

        # Reconstruct on click or upload
        reconstruct_btn.click(
            fn=reconstruct_hdr,
            inputs=[input_image, display_mode],
            outputs=[preview_input, preview_output, info_output, raw_hdr_state],
        )
        input_image.upload(
            fn=reconstruct_hdr,
            inputs=[input_image, display_mode],
            outputs=[preview_input, preview_output, info_output, raw_hdr_state],
        )

        # Instant display mode switch without re-running the model
        display_mode.change(
            fn=on_display_mode_change,
            inputs=[raw_hdr_state, display_mode],
            outputs=[preview_output, info_output],
        )

    return demo


def main():
    parser = argparse.ArgumentParser(description="RLUNet Single-Image HDR Reconstruction Gradio Demo")
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run terminal test with synthetic dummy image and exit without launching web server.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to launch the Gradio server on (default: 7860).",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a publicly shareable Gradio link.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host interface to bind to (default: 127.0.0.1).",
    )
    args = parser.parse_args()

    if args.test:
        print("\n--- Running CLI Terminal Test (Dummy Image) ---")
        dummy_img = np.random.randint(0, 256, (400, 600, 3), dtype=np.uint8)
        print(f"Input dummy shape: {dummy_img.shape}")

        # Test Natural (gamma) default mode
        in_preview, out_nat, details_nat, raw_hdr = reconstruct_hdr(dummy_img, display_mode=MODE_NATURAL)
        print(f"Natural mode output shape: {out_nat.shape}, dtype: {out_nat.dtype}")
        print(f"Details: {details_nat}")
        assert in_preview.shape == (512, 512, 3), f"Expected (512, 512, 3), got {in_preview.shape}"
        assert out_nat.shape == (512, 512, 3), f"Expected (512, 512, 3), got {out_nat.shape}"
        assert out_nat.dtype == np.uint8, f"Expected uint8, got {out_nat.dtype}"
        assert raw_hdr.shape == (512, 512, 3), f"Expected raw HDR shape (512, 512, 3), got {raw_hdr.shape}"

        # Test switching to Mu-law mode using cached raw_hdr (no re-running model)
        out_mu, details_mu = on_display_mode_change(raw_hdr, display_mode=MODE_MULAW)
        print(f"Mu-law mode output shape: {out_mu.shape}, dtype: {out_mu.dtype}")
        print(f"Details: {details_mu}")
        assert out_mu.shape == (512, 512, 3), f"Expected (512, 512, 3), got {out_mu.shape}"
        assert out_mu.dtype == np.uint8, f"Expected uint8, got {out_mu.dtype}"

        # Verify that switching back to Natural produces the same output
        out_nat2, _ = on_display_mode_change(raw_hdr, display_mode=MODE_NATURAL)
        assert np.array_equal(out_nat, out_nat2), "Display mode switch should be deterministic!"

        print("[SUCCESS] Terminal test verified both display modes and instant mode switching!\n")
    else:
        demo = create_demo()
        print(f"\nLaunching Gradio Demo on http://{args.host}:{args.port}...")
        demo.launch(
            server_name=args.host,
            server_port=args.port,
            share=args.share,
        )


if __name__ == "__main__":
    main()
