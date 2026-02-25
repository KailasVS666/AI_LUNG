from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class Denoise25DUNetSmall(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 1, base_channels: int = 32) -> None:
        super().__init__()

        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.out_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        b = self.bottleneck(p2)

        u2 = self.up2(b)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))

        u1 = self.up1(d2)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))

        out = self.out_conv(d1)
        return out


class ChannelAttention3D(nn.Module):
    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.mlp = nn.Sequential(
            nn.Conv3d(channels, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.mlp(self.avg_pool(x))
        return x * w


class ConvBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Recon3DAttentionUNetSmall(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1, base_channels: int = 16) -> None:
        super().__init__()
        self.enc1 = ConvBlock3D(in_channels, base_channels)
        self.attn1 = ChannelAttention3D(base_channels)
        self.pool1 = nn.MaxPool3d(2)

        self.enc2 = ConvBlock3D(base_channels, base_channels * 2)
        self.attn2 = ChannelAttention3D(base_channels * 2)
        self.pool2 = nn.MaxPool3d(2)

        self.bottleneck = ConvBlock3D(base_channels * 2, base_channels * 4)
        self.bottleneck_attn = ChannelAttention3D(base_channels * 4)

        self.up2 = nn.ConvTranspose3d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock3D(base_channels * 4, base_channels * 2)

        self.up1 = nn.ConvTranspose3d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = ConvBlock3D(base_channels * 2, base_channels)

        self.out_conv = nn.Conv3d(base_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.attn1(self.enc1(x))
        p1 = self.pool1(e1)

        e2 = self.attn2(self.enc2(p1))
        p2 = self.pool2(e2)

        b = self.bottleneck_attn(self.bottleneck(p2))

        u2 = self.up2(b)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))

        u1 = self.up1(d2)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))

        out = self.out_conv(d1)
        return torch.clamp(out, 0.0, 1.0)


class PhysicsGuidedReconLoss(nn.Module):
    def __init__(self, l1_weight: float = 1.0, grad_weight: float = 0.2, range_weight: float = 0.05) -> None:
        super().__init__()
        self.l1_weight = l1_weight
        self.grad_weight = grad_weight
        self.range_weight = range_weight
        self.l1 = nn.L1Loss()

    @staticmethod
    def _grad3d(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dz = x[:, :, 1:, :, :] - x[:, :, :-1, :, :]
        dy = x[:, :, :, 1:, :] - x[:, :, :, :-1, :]
        dx = x[:, :, :, :, 1:] - x[:, :, :, :, :-1]
        return dz, dy, dx

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        recon = self.l1(pred, target)

        p_dz, p_dy, p_dx = self._grad3d(pred)
        t_dz, t_dy, t_dx = self._grad3d(target)
        grad = (
            self.l1(p_dz, t_dz)
            + self.l1(p_dy, t_dy)
            + self.l1(p_dx, t_dx)
        ) / 3.0

        below_zero = F.relu(-pred)
        above_one = F.relu(pred - 1.0)
        range_penalty = (below_zero.mean() + above_one.mean())

        total = self.l1_weight * recon + self.grad_weight * grad + self.range_weight * range_penalty
        metrics = {
            "loss_recon": float(recon.detach().item()),
            "loss_grad": float(grad.detach().item()),
            "loss_range": float(range_penalty.detach().item()),
            "loss_total": float(total.detach().item()),
        }
        return total, metrics


class NoduleClassifier3D(nn.Module):
    """
    Lightweight 3D CNN classifier for nodule detection from 3D patches.
    Uses strided convolutions for downsampling, global average pooling, and binary classification head.
    """
    def __init__(self, in_channels: int = 1, base_channels: int = 16, num_classes: int = 2) -> None:
        super().__init__()
        
        # Feature extraction blocks
        self.features = nn.Sequential(
            # Block 1: (1, 32, 64, 64) -> (16, 16, 32, 32)
            nn.Conv3d(in_channels, base_channels, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm3d(base_channels),
            nn.LeakyReLU(0.1, inplace=True),
            
            # Block 2: (16, 16, 32, 32) -> (32, 8, 16, 16)
            nn.Conv3d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm3d(base_channels * 2),
            nn.LeakyReLU(0.1, inplace=True),
            
            # Block 3: (32, 8, 16, 16) -> (64, 4, 8, 8)
            nn.Conv3d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm3d(base_channels * 4),
            nn.LeakyReLU(0.1, inplace=True),
            
            # Block 4: (64, 4, 8, 8) -> (128, 2, 4, 4)
            nn.Conv3d(base_channels * 4, base_channels * 8, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm3d(base_channels * 8),
            nn.LeakyReLU(0.1, inplace=True),
        )
        
        # Global average pooling + classifier
        self.gap = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(base_channels * 8, num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, D, H, W) input patches
            
        Returns:
            (B, num_classes) logits
        """
        features = self.features(x)  # (B, 128, 2, 4, 4)
        pooled = self.gap(features)  # (B, 128, 1, 1, 1)
        flattened = pooled.view(pooled.size(0), -1)  # (B, 128)
        logits = self.classifier(flattened)  # (B, 2)
        return logits

