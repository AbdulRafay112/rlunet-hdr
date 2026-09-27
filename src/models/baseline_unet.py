"""Baseline U-Net architecture for single-image HDR reconstruction."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Two successive Convolution-BatchNorm-ReLU blocks."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: Optional[int] = None,
        use_batchnorm: bool = True,
    ):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels

        layers = [
            nn.Conv2d(
                in_channels,
                mid_channels,
                kernel_size=3,
                padding=1,
                bias=not use_batchnorm,
            )
        ]
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(mid_channels))
        layers.append(nn.ReLU(inplace=True))

        layers.append(
            nn.Conv2d(
                mid_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=not use_batchnorm,
            )
        )
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    """Downscaling block: MaxPool2d(2) followed by DoubleConv."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_batchnorm: bool = True,
    ):
        super().__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels, use_batchnorm=use_batchnorm),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(x)


class Up(nn.Module):
    """Upscaling block: ConvTranspose2d (or Bilinear Upsample), skip concatenation, and DoubleConv."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bilinear: bool = False,
        use_batchnorm: bool = True,
    ):
        super().__init__()
        self.bilinear = bilinear

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(
                in_channels, out_channels, in_channels // 2, use_batchnorm=use_batchnorm
            )
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(
                in_channels, out_channels, use_batchnorm=use_batchnorm
            )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Forward pass with skip connection.

        Args:
            x1: Feature map from previous decoder level (needs upsampling).
            x2: Skip-connection feature map from encoder.
        """
        x1 = self.up(x1)

        # Handle potential padding/shape discrepancies if input dimensions are odd
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        if diff_y != 0 or diff_x != 0:
            x1 = F.pad(
                x1,
                [
                    diff_x // 2,
                    diff_x - diff_x // 2,
                    diff_y // 2,
                    diff_y - diff_y // 2,
                ],
            )

        # Concatenate along channel dimension
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class BaselineUNet(nn.Module):
    """Standard 4-level Encoder-Decoder U-Net baseline for HDR reconstruction.

    Takes a 3-channel LDR image as input and predicts a 3-channel HDR output
    with the exact same spatial dimensions.

    Features:
      - 4 downsampling levels and 4 upsampling levels with skip connections.
      - Linear 1x1 convolution output with NO final activation function,
        preserving unbounded HDR radiance values.

    Args:
        in_channels: Number of input channels (default: 3 for LDR RGB).
        out_channels: Number of output channels (default: 3 for HDR RGB).
        base_channels: Number of feature maps in the first layer (default: 64).
        bilinear: If True, uses bilinear interpolation for upsampling instead of ConvTranspose2d (default: False).
        use_batchnorm: Whether to include BatchNorm2d layers in DoubleConv blocks (default: True).
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 64,
        bilinear: bool = False,
        use_batchnorm: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.bilinear = bilinear

        factor = 2 if bilinear else 1

        # Initial convolution (level 0: 512x512)
        self.inc = DoubleConv(
            in_channels, base_channels, use_batchnorm=use_batchnorm
        )

        # 4 Downsampling levels
        # Level 1 down: 512 -> 256
        self.down1 = Down(
            base_channels, base_channels * 2, use_batchnorm=use_batchnorm
        )
        # Level 2 down: 256 -> 128
        self.down2 = Down(
            base_channels * 2, base_channels * 4, use_batchnorm=use_batchnorm
        )
        # Level 3 down: 128 -> 64
        self.down3 = Down(
            base_channels * 4, base_channels * 8, use_batchnorm=use_batchnorm
        )
        # Level 4 down: 64 -> 32 (bottleneck)
        self.down4 = Down(
            base_channels * 8,
            (base_channels * 16) // factor,
            use_batchnorm=use_batchnorm,
        )

        # 4 Upsampling levels
        # Level 4 up: 32 -> 64
        self.up1 = Up(
            base_channels * 16,
            (base_channels * 8) // factor,
            bilinear=bilinear,
            use_batchnorm=use_batchnorm,
        )
        # Level 3 up: 64 -> 128
        self.up2 = Up(
            base_channels * 8,
            (base_channels * 4) // factor,
            bilinear=bilinear,
            use_batchnorm=use_batchnorm,
        )
        # Level 2 up: 128 -> 256
        self.up3 = Up(
            base_channels * 4,
            (base_channels * 2) // factor,
            bilinear=bilinear,
            use_batchnorm=use_batchnorm,
        )
        # Level 1 up: 256 -> 512
        self.up4 = Up(
            base_channels * 2,
            base_channels,
            bilinear=bilinear,
            use_batchnorm=use_batchnorm,
        )

        # Final 1x1 convolution output layer - NO activation function
        self.outc = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x1 = self.inc(x)         # (B, 64, 512, 512)
        x2 = self.down1(x1)       # (B, 128, 256, 256)
        x3 = self.down2(x2)       # (B, 256, 128, 128)
        x4 = self.down3(x3)       # (B, 512, 64, 64)
        x5 = self.down4(x4)       # (B, 1024, 32, 32) (bottleneck)

        # Decoder with skip connections
        x = self.up1(x5, x4)      # (B, 512, 64, 64)
        x = self.up2(x, x3)       # (B, 256, 128, 128)
        x = self.up3(x, x2)       # (B, 128, 256, 256)
        x = self.up4(x, x1)       # (B, 64, 512, 512)

        # Linear output layer (unbounded HDR values)
        logits = self.outc(x)     # (B, 3, 512, 512)
        return logits


if __name__ == "__main__":
    print("--- Testing BaselineUNet Model ---")
    batch_size = 2
    in_channels = 3
    out_channels = 3
    height = 512
    width = 512

    model = BaselineUNet(
        in_channels=in_channels,
        out_channels=out_channels,
        base_channels=64,
        bilinear=False,
    )
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")

    # Create dummy 3-channel LDR input (B, C, H, W)
    dummy_input = torch.rand(batch_size, in_channels, height, width, dtype=torch.float32)
    print(f"Input Tensor Shape: {tuple(dummy_input.shape)}")

    with torch.no_grad():
        output = model(dummy_input)

    print(f"Output Tensor Shape: {tuple(output.shape)}")

    # Verifications
    expected_shape = (batch_size, out_channels, height, width)
    assert (
        output.shape == expected_shape
    ), f"Shape mismatch: expected {expected_shape}, got {output.shape}"
    print(f"Shape Verification: PASSED (Matches expected {expected_shape})")

    # Verify no final activation was applied (model.outc is pure Conv2d)
    assert isinstance(
        model.outc, nn.Conv2d
    ), "Final layer should be a raw Conv2d layer without activation."
    print("Output Layer Activation Check: PASSED (Linear / No activation)")

    print("[SUCCESS] BaselineUNet sanity check completed successfully!")
