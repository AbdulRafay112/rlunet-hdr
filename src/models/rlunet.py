"""RLUNet model architecture for single-image HDR reconstruction.

Reverses the camera Low Dynamic Range (LDR) imaging pipeline using four sequential modules:
  1. Dequantization Module (DQ):
     Removes 8-bit quantization artifacts (banding, contouring) and outputs a refined
     LDR estimate in [0, 1].
  2. Linearization / Reverse Lossy Network (RLN):
     Reverses the camera's non-linear tone-mapping curve (CRF) and compression to recover
     a linear-domain radiance image (non-negative, >= 0).
  3. Luminance Guidance Module (LGM):
     Lightweight sub-network estimating a luminance/exposure guidance map from the linear image,
     helping subsequent stages focus on over- and under-exposed regions.
  4. Texture Filling Module (TFM):
     A U-Net-style hallucination network (sharing structural components with BaselineUNet)
     that takes the linear image concatenated with the LGM guidance map (4 channels), and
     predicts the final HDR output, recovering lost textures and highlights.
  5. Imaging Pipeline Module (IPM) / RLUNet:
     Wraps all stages into a unified forward() pass returning unbounded HDR predictions directly.
"""

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse modular building blocks from baseline_unet
try:
    from .baseline_unet import DoubleConv, Down, Up
except (ImportError, ValueError):
    # Fallback for direct script execution
    project_root = str(Path(__file__).resolve().parents[2])
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from src.models.baseline_unet import DoubleConv, Down, Up


class DequantizationModule(nn.Module):
    """Dequantization Module (DQ).

    Removes 8-bit quantization artifacts (banding, stepping, false contours)
    from normalized LDR input images in [0, 1].

    Uses a lightweight residual convolutional network:
        delta = ConvStack(x)
        refined_x = clamp(x + delta, 0.0, 1.0)

    Args:
        in_channels: Number of input color channels (default: 3).
        mid_channels: Number of intermediate feature channels (default: 32).
        num_layers: Total number of convolution layers (default: 3).
    """

    def __init__(
        self,
        in_channels: int = 3,
        mid_channels: int = 32,
        num_layers: int = 3,
    ):
        super().__init__()
        if num_layers < 2:
            raise ValueError(f"num_layers must be at least 2, got {num_layers}")

        layers = [
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        ]
        for _ in range(num_layers - 2):
            layers.append(
                nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1)
            )
            layers.append(nn.ReLU(inplace=True))
        layers.append(
            nn.Conv2d(mid_channels, in_channels, kernel_size=3, padding=1)
        )

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for dequantization.

        Args:
            x: Input LDR tensor of shape (B, C, H, W) normalized to [0, 1].

        Returns:
            Refined estimate in [0, 1] of shape (B, C, H, W).
        """
        residual = self.net(x)
        return torch.clamp(x + residual, 0.0, 1.0)


class LinearizationNetwork(nn.Module):
    """Linearization / Reverse Lossy Network (RLN).

    Reverses the camera's non-linear tone-mapping curve (CRF) and lossy compression
    to recover a linear-domain radiance image from the dequantized LDR input.

    Ensures non-negative linear radiance values (>= 0) using ReLU activation.

    Args:
        in_channels: Number of color channels (default: 3).
        mid_channels: Number of intermediate feature channels (default: 32).
        num_layers: Total number of convolution layers (default: 3).
    """

    def __init__(
        self,
        in_channels: int = 3,
        mid_channels: int = 32,
        num_layers: int = 3,
    ):
        super().__init__()
        if num_layers < 2:
            raise ValueError(f"num_layers must be at least 2, got {num_layers}")

        layers = [
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        ]
        for _ in range(num_layers - 2):
            layers.append(
                nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1)
            )
            layers.append(nn.ReLU(inplace=True))
        layers.append(
            nn.Conv2d(mid_channels, in_channels, kernel_size=3, padding=1)
        )

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for linearization.

        Args:
            x: Dequantized LDR tensor of shape (B, C, H, W) in [0, 1].

        Returns:
            Linear-domain tensor of shape (B, C, H, W) with values >= 0.
        """
        delta = self.net(x)
        return F.relu(x + delta)


class LuminanceGuidanceModule(nn.Module):
    """Luminance Guidance Module (LGM).

    A lightweight sub-network that estimates a luminance/exposure guidance map
    from the linear-domain image. This guidance map highlights saturated/overexposed
    and dark/underexposed regions, guiding the downstream Texture Filling Module.

    Args:
        in_channels: Number of input color channels (default: 3).
        mid_channels: Number of intermediate feature channels (default: 16).
        out_channels: Number of guidance map channels (default: 1).
    """

    def __init__(
        self,
        in_channels: int = 3,
        mid_channels: int = 16,
        out_channels: int = 1,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for luminance guidance.

        Args:
            x: Linear-domain image tensor of shape (B, C, H, W).

        Returns:
            Normalized guidance map of shape (B, 1, H, W) in [0, 1].
        """
        return self.net(x)


class TextureFillingModule(nn.Module):
    """Texture Filling Module (TFM).

    A U-Net-style hallucination network (sharing architecture with BaselineUNet)
    that takes the linear image concatenated with the LGM guidance map (3 + 1 = 4 channels),
    and predicts the final HDR output. It focuses on hallucinating plausible high-frequency
    textures and physical radiance levels in overexposed/clipped regions.

    Features:
      - 4 downsampling levels and 4 upsampling levels with skip connections.
      - Linear 1x1 convolution output with NO final activation function,
        preserving unbounded HDR radiance values.

    Args:
        in_channels: Number of input channels (default: 4 for 3 RGB + 1 Guidance).
        out_channels: Number of output channels (default: 3 for HDR RGB).
        base_channels: Number of feature maps in the first level (default: 32).
        bilinear: If True, uses bilinear interpolation instead of ConvTranspose2d (default: False).
        use_batchnorm: Whether to include BatchNorm2d layers in DoubleConv blocks (default: True).
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 3,
        base_channels: int = 32,
        bilinear: bool = False,
        use_batchnorm: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.bilinear = bilinear

        factor = 2 if bilinear else 1

        # Initial convolution (level 0)
        self.inc = DoubleConv(
            in_channels, base_channels, use_batchnorm=use_batchnorm
        )

        # 4 Downsampling stages
        self.down1 = Down(
            base_channels, base_channels * 2, use_batchnorm=use_batchnorm
        )
        self.down2 = Down(
            base_channels * 2, base_channels * 4, use_batchnorm=use_batchnorm
        )
        self.down3 = Down(
            base_channels * 4, base_channels * 8, use_batchnorm=use_batchnorm
        )
        self.down4 = Down(
            base_channels * 8,
            (base_channels * 16) // factor,
            use_batchnorm=use_batchnorm,
        )

        # 4 Upsampling stages with skip connections
        self.up1 = Up(
            base_channels * 16,
            (base_channels * 8) // factor,
            bilinear=bilinear,
            use_batchnorm=use_batchnorm,
        )
        self.up2 = Up(
            base_channels * 8,
            (base_channels * 4) // factor,
            bilinear=bilinear,
            use_batchnorm=use_batchnorm,
        )
        self.up3 = Up(
            base_channels * 4,
            (base_channels * 2) // factor,
            bilinear=bilinear,
            use_batchnorm=use_batchnorm,
        )
        self.up4 = Up(
            base_channels * 2,
            base_channels,
            bilinear=bilinear,
            use_batchnorm=use_batchnorm,
        )

        # Final 1x1 convolution - NO activation function (unbounded output)
        self.outc = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(
        self,
        x: torch.Tensor,
        guidance: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for texture filling.

        Args:
            x: Linear image tensor of shape (B, 3, H, W) or concatenated (B, 4, H, W).
            guidance: Optional guidance map of shape (B, 1, H, W). If provided, concatenated
                      along channel dimension to form (B, 4, H, W).

        Returns:
            Predicted HDR tensor of shape (B, 3, H, W) with unbounded radiance values.
        """
        if guidance is not None:
            feat = torch.cat([x, guidance], dim=1)
        else:
            feat = x

        # Encoder
        x1 = self.inc(feat)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder with skip connections
        d = self.up1(x5, x4)
        d = self.up2(d, x3)
        d = self.up3(d, x2)
        d = self.up4(d, x1)

        # Final linear projection
        hdr_pred = self.outc(d)
        return hdr_pred


class ImagingPipelineModule(nn.Module):
    """Imaging Pipeline Module (IPM) / Full RLUNet Model.

    Reverses the LDR imaging pipeline for single-image HDR reconstruction by executing:
      1. Dequantization Module (DQ): Removes 8-bit quantization artifacts -> [0, 1].
      2. Linearization Network (RLN): Inverts non-linear camera tone curve/CRF -> Linear domain.
      3. Luminance Guidance Module (LGM): Estimates exposure/luminance mask -> [0, 1].
      4. Texture Filling Module (TFM): Hallucinates saturated details via U-Net -> Final HDR.

    RLUNet(ldr_image) directly returns the final HDR prediction with unbounded values
    (no final activation function).

    Args:
        in_channels: Number of input color channels (default: 3).
        out_channels: Number of output HDR channels (default: 3).
        base_channels: Base feature channels for the TFM U-Net (default: 32).
        guidance_channels: Number of guidance map channels (default: 1).
        dq_channels: Intermediate channels for Dequantization Module (default: 32).
        rln_channels: Intermediate channels for Linearization Network (default: 32).
        lgm_channels: Intermediate channels for Luminance Guidance Module (default: 16).
        bilinear: If True, uses bilinear interpolation for TFM upsampling (default: False).
        use_batchnorm: Whether to use BatchNorm2d in TFM (default: True).
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 32,
        guidance_channels: int = 1,
        dq_channels: int = 32,
        rln_channels: int = 32,
        lgm_channels: int = 16,
        bilinear: bool = False,
        use_batchnorm: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.guidance_channels = guidance_channels

        # 1. Dequantization Module
        self.dq = DequantizationModule(
            in_channels=in_channels,
            mid_channels=dq_channels,
            num_layers=3,
        )

        # 2. Linearization / Reverse Lossy Network
        self.rln = LinearizationNetwork(
            in_channels=in_channels,
            mid_channels=rln_channels,
            num_layers=3,
        )

        # 3. Luminance Guidance Module
        self.lgm = LuminanceGuidanceModule(
            in_channels=in_channels,
            mid_channels=lgm_channels,
            out_channels=guidance_channels,
        )

        # 4. Texture Filling Module (U-Net)
        self.tfm = TextureFillingModule(
            in_channels=in_channels + guidance_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            bilinear=bilinear,
            use_batchnorm=use_batchnorm,
        )

    def forward(
        self,
        x: torch.Tensor,
        return_intermediates: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Reconstruct HDR image from an 8-bit normalized LDR input.

        Args:
            x: Input LDR image tensor of shape (B, 3, H, W) normalized to [0, 1].
            return_intermediates: If True, returns a tuple of (hdr_prediction, intermediate_dict).
                                  If False (default), returns hdr_prediction directly.

        Returns:
            torch.Tensor: Final reconstructed HDR image of shape (B, 3, H, W),
                          with unbounded radiance values (no final activation).
            (Optional) Dict[str, torch.Tensor]: Dictionary containing intermediate tensors:
              - 'refined_ldr': Dequantized estimate in [0, 1].
              - 'linear_img': Linear-domain radiance estimate (>= 0).
              - 'guidance_map': Luminance/exposure guidance map in [0, 1].
        """
        # Step 1: Dequantization
        refined_ldr = self.dq(x)

        # Step 2: Linearization (inverting camera response curve)
        linear_img = self.rln(refined_ldr)

        # Step 3: Luminance guidance map estimation
        guidance_map = self.lgm(linear_img)

        # Step 4: Texture filling / HDR hallucination
        hdr_pred = self.tfm(linear_img, guidance_map)

        if return_intermediates:
            intermediates = {
                "refined_ldr": refined_ldr,
                "linear_img": linear_img,
                "guidance_map": guidance_map,
            }
            return hdr_pred, intermediates

        return hdr_pred


# Aliases for convenience and acronym naming conventions
RLUNet = ImagingPipelineModule
IPM = ImagingPipelineModule
DQ = DequantizationModule
RLN = LinearizationNetwork
LGM = LuminanceGuidanceModule
TFM = TextureFillingModule

__all__ = [
    "DequantizationModule",
    "LinearizationNetwork",
    "LuminanceGuidanceModule",
    "TextureFillingModule",
    "ImagingPipelineModule",
    "RLUNet",
    "DQ",
    "RLN",
    "LGM",
    "TFM",
    "IPM",
]


if __name__ == "__main__":
    print("=" * 65)
    print("--- Testing RLUNet (ImagingPipelineModule) Model ---")
    print("=" * 65)

    batch_size = 2
    in_channels = 3
    out_channels = 3
    height = 512
    width = 512

    # Instantiate model with base_channels=32 for a lightweight, trainable setup
    model = RLUNet(
        in_channels=in_channels,
        out_channels=out_channels,
        base_channels=32,
    )
    model.eval()

    # Parameter counting breakdown
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    dq_params = sum(p.numel() for p in model.dq.parameters())
    rln_params = sum(p.numel() for p in model.rln.parameters())
    lgm_params = sum(p.numel() for p in model.lgm.parameters())
    tfm_params = sum(p.numel() for p in model.tfm.parameters())

    print("Model Parameter Breakdown:")
    print(f"  1. Dequantization Module (DQ):         {dq_params:>10,} params")
    print(f"  2. Linearization Network (RLN):         {rln_params:>10,} params")
    print(f"  3. Luminance Guidance Module (LGM):     {lgm_params:>10,} params")
    print(f"  4. Texture Filling Module (TFM U-Net):  {tfm_params:>10,} params")
    print("  " + "-" * 50)
    print(f"  Total Model Parameters:                {total_params:>10,} params")
    print(f"  Trainable Parameters:                  {trainable_params:>10,} params")
    print("=" * 65)

    # Create dummy 3-channel LDR input (B, C, H, W)
    dummy_input = torch.rand(
        batch_size, in_channels, height, width, dtype=torch.float32
    )
    print(
        f"Input Tensor Shape:  {tuple(dummy_input.shape)} | Range: [{dummy_input.min().item():.3f}, {dummy_input.max().item():.3f}]"
    )

    with torch.no_grad():
        # Standard forward pass
        output = model(dummy_input)

        # Forward pass returning intermediate outputs
        output_check, intermediates = model(dummy_input, return_intermediates=True)

    print(f"Output Tensor Shape: {tuple(output.shape)}")

    # 1. Shape verification
    expected_shape = (batch_size, out_channels, height, width)
    assert (
        output.shape == expected_shape
    ), f"Shape mismatch: expected {expected_shape}, got {output.shape}"
    print(f"Shape Verification: PASSED (Matches expected {expected_shape})")

    # 2. Output layer activation verification (unbounded output)
    assert isinstance(
        model.tfm.outc, nn.Conv2d
    ), "Final layer must be raw Conv2d without activation for unbounded HDR output."
    print("Output Layer Activation Check: PASSED (Linear / No final activation)")

    # 3. Intermediate stages verification
    refined_ldr = intermediates["refined_ldr"]
    linear_img = intermediates["linear_img"]
    guidance_map = intermediates["guidance_map"]

    assert refined_ldr.shape == (batch_size, 3, height, width), "DQ shape mismatch"
    assert linear_img.shape == (batch_size, 3, height, width), "RLN shape mismatch"
    assert guidance_map.shape == (batch_size, 1, height, width), "LGM shape mismatch"

    assert (
        0.0 <= refined_ldr.min().item() and refined_ldr.max().item() <= 1.0
    ), "Refined LDR should be bounded in [0, 1]."
    assert (
        linear_img.min().item() >= 0.0
    ), "Linear radiance must be non-negative (>= 0)."
    assert (
        0.0 <= guidance_map.min().item() and guidance_map.max().item() <= 1.0
    ), "Guidance map should be normalized in [0, 1]."

    print("Intermediate Stages Verification: PASSED")
    print(
        f"  - Refined LDR (DQ) range:    [{refined_ldr.min().item():.3f}, {refined_ldr.max().item():.3f}]"
    )
    print(
        f"  - Linear Image (RLN) range:   [{linear_img.min().item():.3f}, {linear_img.max().item():.3f}]"
    )
    print(
        f"  - Guidance Map (LGM) range:   [{guidance_map.min().item():.3f}, {guidance_map.max().item():.3f}]"
    )
    print("=" * 65)
    print("[SUCCESS] RLUNet sanity check completed successfully!")
