"""RLUNet Single-Image HDR Reconstruction - Interactive Gradio Demo.

Loads trained RLUNet weights from models/checkpoints/rlunet_final.pth on CPU,
accepts an uploaded photo, resizes it to 512x512, normalizes to [0, 1], runs
RLUNet inference under torch.no_grad, clamps output to [0, 16], and displays the
mu-law tone-mapped result alongside the input image.
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


def reconstruct_hdr(
    input_image: Optional[np.ndarray],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
    """Reconstruct HDR image from uploaded LDR photo.

    Steps:
      1. Resize image to 512x512 RGB.
      2. Convert to float32 in [0, 1] tensor of shape (1, 3, 512, 512).
      3. Run RLUNet in eval mode under torch.no_grad().
      4. Clamp predicted output to [0, 16].
      5. Normalize output by its own max value, then tone-map with mu-law (mu=5000).
      6. Return (512x512 input, tone-mapped output, metadata string).
    """
    if input_image is None:
        return None, None, "Please upload an image to begin reconstruction."

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

    # 6. Normalize by output's own max radiance and apply mu-law tonemapping (mu=5000)
    max_val = output_clamped.max().item()
    scale = max(float(max_val), 1e-6)
    output_norm = output_clamped / scale
    output_tm = mu_law_tonemap(output_norm, mu=5000.0)

    # 7. Convert tone-mapped tensor back to uint8 RGB (512, 512, 3)
    tm_np = output_tm.squeeze(0).permute(1, 2, 0).cpu().numpy()
    tm_uint8 = np.clip(tm_np * 255.0, 0, 255).astype(np.uint8)

    info_text = (
        f"Input: {orig_w}x{orig_h} (processed at 512x512) | "
        f"Estimated Max Radiance: {scale:.2f} | "
        f"Tone-mapping: mu-law (mu=5000, normalized by max)"
    )
    return resized_input, tm_uint8, info_text


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
                reconstruct_btn = gr.Button("Reconstruct HDR", variant="primary", size="lg")
                info_output = gr.Textbox(label="Inference Details", interactive=False)

            with gr.Column(scale=2):
                with gr.Row():
                    preview_input = gr.Image(
                        label="Input Image (512x512)",
                        interactive=False,
                    )
                    preview_output = gr.Image(
                        label="Reconstructed HDR (Tone-Mapped)",
                        interactive=False,
                    )

        reconstruct_btn.click(
            fn=reconstruct_hdr,
            inputs=[input_image],
            outputs=[preview_input, preview_output, info_output],
        )
        input_image.upload(
            fn=reconstruct_hdr,
            inputs=[input_image],
            outputs=[preview_input, preview_output, info_output],
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
        in_preview, out_tm, details = reconstruct_hdr(dummy_img)
        print(f"Resized input shape: {in_preview.shape}")
        print(f"Reconstructed tone-mapped output shape: {out_tm.shape}")
        print(f"Details: {details}")
        assert in_preview.shape == (512, 512, 3), f"Expected (512, 512, 3), got {in_preview.shape}"
        assert out_tm.shape == (512, 512, 3), f"Expected (512, 512, 3), got {out_tm.shape}"
        assert out_tm.dtype == np.uint8, f"Expected uint8, got {out_tm.dtype}"
        print("[SUCCESS] Terminal inference test passed successfully!\n")
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
