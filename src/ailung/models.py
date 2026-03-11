"""
AI-LUNG Neural Network Models
==============================
Stage 1: Denoise25DUNet  — 2.5D U-Net (EfficientNet-B5 encoder + CBAM attention)
Stage 2: Recon3DUNet     — Lightweight 3D U-Net with 3D-CBAM attention
Stage 3: NoduleDetector3D — 3D CNN with Dice + Cross-Entropy loss
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Shared Utility: 2D CBAM Attention
# ---------------------------------------------------------------------------

class CBAM2D(nn.Module):
    """Convolutional Block Attention Module for 2D feature maps."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 4)

        # Channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )

        # Spatial attention
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Channel attention
        ca = torch.sigmoid(self.mlp(self.avg_pool(x)) + self.mlp(self.max_pool(x)))
        x = x * ca

        # Spatial attention
        avg_s = x.mean(dim=1, keepdim=True)
        max_s, _ = x.max(dim=1, keepdim=True)
        sa = torch.sigmoid(self.spatial_conv(torch.cat([avg_s, max_s], dim=1)))
        return x * sa


# ---------------------------------------------------------------------------
# Stage 1: 2.5D U-Net with EfficientNet-B5 encoder + CBAM
# ---------------------------------------------------------------------------

class _EfficientNetB5Encoder(nn.Module):
    """
    Simplified EfficientNet-B5–style encoder built from scratch with
    depthwise-separable convolutions and compound scaling.
    Accepts arbitrary in_channels (we patch the first conv).
    Produces skip connections at 4 scales.
    """

    def __init__(self, in_channels: int) -> None:
        super().__init__()

        def dw_sep(c_in: int, c_out: int, stride: int = 1) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(c_in, c_in, 3, stride=stride, padding=1, groups=c_in, bias=False),
                nn.BatchNorm2d(c_in),
                nn.SiLU(inplace=True),
                nn.Conv2d(c_in, c_out, 1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.SiLU(inplace=True),
            )

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 48, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.SiLU(inplace=True),
        )           # → (B, 48, H/2, W/2)

        # Stage 1 — no stride
        self.s1 = nn.Sequential(dw_sep(48, 24), dw_sep(24, 24))       # (B, 24, H/2, W/2)
        # Stage 2 — stride 2
        self.s2 = nn.Sequential(dw_sep(24, 40, stride=2), dw_sep(40, 40))  # (B, 40, H/4, W/4)
        # Stage 3 — stride 2
        self.s3 = nn.Sequential(dw_sep(40, 80, stride=2), dw_sep(80, 80))  # (B, 80, H/8, W/8)
        # Stage 4 — stride 2
        self.s4 = nn.Sequential(dw_sep(80, 192, stride=2), dw_sep(192, 192))  # (B, 192, H/16, W/16)

        # Bottleneck projection
        self.bottleneck_proj = nn.Sequential(
            dw_sep(192, 320),
            CBAM2D(320),
        )  # (B, 320, H/16, W/16)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        s0 = self.stem(x)    # (B,  48, H/2,  W/2 )
        s1 = self.s1(s0)     # (B,  24, H/2,  W/2 )
        s2 = self.s2(s1)     # (B,  40, H/4,  W/4 )
        s3 = self.s3(s2)     # (B,  80, H/8,  W/8 )
        s4 = self.s4(s3)     # (B, 192, H/16, W/16)
        b  = self.bottleneck_proj(s4)  # (B, 320, H/16, W/16)
        return s1, s2, s3, s4, b


class _DecoderBlock2D(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )
        self.cbam = CBAM2D(out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.cbam(self.conv(x))


class Denoise25DUNet(nn.Module):
    """
    Stage 1: 2.5D Denoising U-Net.

    Input:  (B, context_slices*2+1, H, W)  — 9 consecutive low-dose CT slices
    Output: (B, 1, H, W)                   — denoised central slice
    Loss:   L1 + (1 - SSIM) + Gradient
    """

    def __init__(self, in_channels: int = 9, out_channels: int = 1) -> None:
        super().__init__()
        self.encoder = _EfficientNetB5Encoder(in_channels)

        # Decoder: bottleneck(320) → s4(192) → s3(80) → s2(40) → s1(24)
        self.dec4 = _DecoderBlock2D(320, 192, 192)
        self.dec3 = _DecoderBlock2D(192, 80, 80)
        self.dec2 = _DecoderBlock2D(80, 40, 40)
        self.dec1 = _DecoderBlock2D(40, 24, 32)

        # Final upsampling to restore original resolution (×2 from stem stride)
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.SiLU(inplace=True),
        )
        self.out_conv = nn.Conv2d(16, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1, s2, s3, s4, b = self.encoder(x)
        d4 = self.dec4(b, s4)
        d3 = self.dec3(d4, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)
        out = self.out_conv(self.final_up(d1))
        return torch.sigmoid(out) # Ensure output is [0, 1] for stability


# ---------------------------------------------------------------------------
# Stage 1 Loss: L1 + (1 - SSIM) + Gradient
# ---------------------------------------------------------------------------

def _ssim2d(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    C1: float = 0.01 ** 2,
    C2: float = 0.03 ** 2,
) -> torch.Tensor:
    # Force High Precision for SSIM math (prevents NaNs in AMP)
    pred_32   = pred.to(torch.float32)
    target_32 = target.to(torch.float32)

    # Gaussian kernel (stays float32)
    coords = torch.arange(window_size, dtype=torch.float32, device=pred.device)
    coords -= window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * 1.5 ** 2))
    g = g / g.sum()
    kernel = g.outer(g).unsqueeze(0).unsqueeze(0)  # (1,1,ws,ws)

    pad = window_size // 2

    def conv(t: torch.Tensor) -> torch.Tensor:
        return F.conv2d(t, kernel, padding=pad, groups=1)

    mu1 = conv(pred_32)
    mu2 = conv(target_32)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = torch.clamp(conv(pred_32 * pred_32) - mu1_sq, min=1e-4)
    sigma2_sq = torch.clamp(conv(target_32 * target_32) - mu2_sq, min=1e-4)
    sigma12   = conv(pred_32 * target_32) - mu1_mu2

    numerator   = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    # 1e-4 epsilon for high numerical stability in float16/AMP
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2) + 1e-4

    score = numerator / denominator
    return score.mean()


class DenoiseLoss(nn.Module):
    """
    Combined loss for Stage 1 denoising.
    Total = λ_l1 * L1 + λ_ssim * (1 − SSIM) + λ_grad * GradLoss
    """

    def __init__(
        self,
        l1_weight: float = 1.0,
        ssim_weight: float = 0.5,
        grad_weight: float = 0.2,
    ) -> None:
        super().__init__()
        self.l1_weight   = l1_weight
        self.ssim_weight = ssim_weight
        self.grad_weight = grad_weight
        self.l1 = nn.L1Loss()

    @staticmethod
    def _grad_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        gx_p = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        gy_p = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        gx_t = target[:, :, :, 1:] - target[:, :, :, :-1]
        gy_t = target[:, :, 1:, :] - target[:, :, :-1, :]
        return (F.l1_loss(gx_p, gx_t) + F.l1_loss(gy_p, gy_t)) / 2.0

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        l1   = self.l1(pred, target)
        ssim_val = _ssim2d(pred, target)
        ssim_loss = 1.0 - ssim_val
        grad = self._grad_loss(pred, target)

        total = self.l1_weight * l1 + self.ssim_weight * ssim_loss + self.grad_weight * grad
        metrics = {
            "loss_l1":   float(l1.detach().item()),
            "loss_ssim": float(ssim_loss.detach().item()),
            "loss_grad": float(grad.detach().item()),
            "loss_total": float(total.detach().item()),
        }
        return total, metrics


# ---------------------------------------------------------------------------
# Shared Utility: 3D CBAM Attention
# ---------------------------------------------------------------------------

class CBAM3D(nn.Module):
    """3D Convolutional Block Attention Module (channel + spatial)."""

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(channels // reduction, 4)

        # Channel attention
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.mlp = nn.Sequential(
            nn.Conv3d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden, channels, 1, bias=False),
        )

        # Spatial attention
        self.spatial_conv = nn.Conv3d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ca = torch.sigmoid(self.mlp(self.avg_pool(x)) + self.mlp(self.max_pool(x)))
        x = x * ca
        avg_s = x.mean(dim=1, keepdim=True)
        max_s, _ = x.max(dim=1, keepdim=True)
        sa = torch.sigmoid(self.spatial_conv(torch.cat([avg_s, max_s], dim=1)))
        return x * sa


class ConvBlock3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ---------------------------------------------------------------------------
# Stage 2: Lightweight 3D U-Net with 3D-CBAM
# ---------------------------------------------------------------------------

class Recon3DUNet(nn.Module):
    """
    Stage 2: 3D Reconstruction U-Net.

    Input:  (B, 1, D, H, W)  — stacked denoised slices as 3D patches (64×64×64)
    Output: (B, 1, D, H, W)  — refined 3D lung volume
    Loss:   L1 + SSIM3D + Gradient3D + Projection consistency
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1, base_channels: int = 16) -> None:
        super().__init__()

        c1, c2, c3, c4 = base_channels, base_channels * 2, base_channels * 4, base_channels * 8

        # Encoder
        self.enc1 = ConvBlock3D(in_channels, c1)
        self.attn1 = CBAM3D(c1)
        self.pool1 = nn.MaxPool3d(2)

        self.enc2 = ConvBlock3D(c1, c2)
        self.attn2 = CBAM3D(c2)
        self.pool2 = nn.MaxPool3d(2)

        self.enc3 = ConvBlock3D(c2, c3)
        self.attn3 = CBAM3D(c3)
        self.pool3 = nn.MaxPool3d(2)

        # Bottleneck
        self.bottleneck = ConvBlock3D(c3, c4)
        self.bottleneck_attn = CBAM3D(c4)

        # Decoder
        self.up3 = nn.ConvTranspose3d(c4, c3, kernel_size=2, stride=2)
        self.dec3 = ConvBlock3D(c3 * 2, c3)

        self.up2 = nn.ConvTranspose3d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock3D(c2 * 2, c2)

        self.up1 = nn.ConvTranspose3d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = ConvBlock3D(c1 * 2, c1)

        self.out_conv = nn.Conv3d(c1, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.attn1(self.enc1(x));  p1 = self.pool1(e1)
        e2 = self.attn2(self.enc2(p1)); p2 = self.pool2(e2)
        e3 = self.attn3(self.enc3(p2)); p3 = self.pool3(e3)

        b = self.bottleneck_attn(self.bottleneck(p3))

        u3 = self.up3(b)
        d3 = self.dec3(torch.cat([u3, e3], dim=1))

        u2 = self.up2(d3)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))

        u1 = self.up1(d2)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))

        out = self.out_conv(d1)
        return torch.clamp(out, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Stage 2 Loss: L1 + SSIM3D + Gradient3D + Projection Consistency
# ---------------------------------------------------------------------------

def _ssim3d_patch(pred: torch.Tensor, target: torch.Tensor, C1: float = 1e-4, C2: float = 9e-4) -> torch.Tensor:
    """Fast patch-level SSIM3D using global statistics."""
    mu1 = pred.mean()
    mu2 = target.mean()
    sigma1_sq = pred.var()
    sigma2_sq = target.var()
    sigma12 = ((pred - mu1) * (target - mu2)).mean()

    num = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    den = (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2)
    return num / den


class Recon3DLoss(nn.Module):
    """
    Combined loss for Stage 2 3D reconstruction.
    Total = λ_l1 * L1 + λ_ssim * (1 − SSIM3D) + λ_grad * Grad3D + λ_proj * ProjConsistency
    """

    def __init__(
        self,
        l1_weight:   float = 1.0,
        ssim_weight: float = 0.5,
        grad_weight: float = 0.2,
        proj_weight: float = 0.1,
    ) -> None:
        super().__init__()
        self.l1_weight   = l1_weight
        self.ssim_weight = ssim_weight
        self.grad_weight = grad_weight
        self.proj_weight = proj_weight
        self.l1 = nn.L1Loss()

    @staticmethod
    def _grad3d(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dz = x[:, :, 1:, :, :] - x[:, :, :-1, :, :]
        dy = x[:, :, :, 1:, :] - x[:, :, :, :-1, :]
        dx = x[:, :, :, :, 1:] - x[:, :, :, :, :-1]
        return dz, dy, dx

    @staticmethod
    def _projection_consistency(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Max-intensity projection along each axis, compare pred vs target."""
        # Axial (z), Coronal (y), Sagittal (x) MIP
        loss = 0.0
        for dim in (2, 3, 4):
            proj_pred   = pred.max(dim=dim).values
            proj_target = target.max(dim=dim).values
            loss = loss + F.l1_loss(proj_pred, proj_target)
        return loss / 3.0

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        l1   = self.l1(pred, target)
        ssim_val  = _ssim3d_patch(pred, target)
        ssim_loss = 1.0 - ssim_val

        p_dz, p_dy, p_dx = self._grad3d(pred)
        t_dz, t_dy, t_dx = self._grad3d(target)
        grad = (self.l1(p_dz, t_dz) + self.l1(p_dy, t_dy) + self.l1(p_dx, t_dx)) / 3.0

        proj = self._projection_consistency(pred, target)

        # HU range penalty (keep predictions in [0, 1])
        range_penalty = (F.relu(-pred).mean() + F.relu(pred - 1.0).mean())

        total = (
            self.l1_weight   * l1
            + self.ssim_weight * ssim_loss
            + self.grad_weight * grad
            + self.proj_weight * proj
            + 0.05            * range_penalty
        )
        metrics = {
            "loss_l1":    float(l1.detach().item()),
            "loss_ssim":  float(ssim_loss.detach().item()),
            "loss_grad":  float(grad.detach().item()),
            "loss_proj":  float(proj.detach().item()),
            "loss_total": float(total.detach().item()),
        }
        return total, metrics


# ---------------------------------------------------------------------------
# Stage 3: 3D CNN Nodule Detector with Dice + Cross-Entropy Loss
# ---------------------------------------------------------------------------

class NoduleDetector3D(nn.Module):
    """
    Stage 3: 3D CNN for nodule detection and classification.

    Input:  (B, 1, D, H, W)  — 3D patch from reconstructed volume  (e.g. 32×64×64)
    Output: (B, num_classes) — logits (benign=0, malignant=1)

    Detects nodules as small as 3–5 mm.
    """

    def __init__(self, in_channels: int = 1, base_channels: int = 32, num_classes: int = 2) -> None:
        super().__init__()

        c = base_channels
        self.features = nn.Sequential(
            # Block 1
            nn.Conv3d(in_channels, c, 3, stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(c),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(c, c, 3, stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(c),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool3d(2),       # → c × D/2 × H/2 × W/2

            # Block 2
            nn.Conv3d(c, c * 2, 3, stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(c * 2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(c * 2, c * 2, 3, stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(c * 2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool3d(2),       # → 2c × D/4 × H/4 × W/4

            # Block 3
            nn.Conv3d(c * 2, c * 4, 3, stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(c * 4),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(c * 4, c * 4, 3, stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(c * 4),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool3d(2),       # → 4c × D/8 × H/8 × W/8

            # Block 4
            nn.Conv3d(c * 4, c * 8, 3, stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(c * 8),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.cbam = CBAM3D(c * 8)
        self.gap  = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(c * 8, c * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(c * 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.cbam(self.features(x))
        pooled = self.gap(feat).view(feat.size(0), -1)
        return self.classifier(pooled)


class NoduleDetectionLoss(nn.Module):
    """
    Combined Dice + Cross-Entropy loss for Stage 3 nodule detection.
    Works on class logits (B, C) and integer labels (B,).
    """

    def __init__(self, dice_weight: float = 1.0, ce_weight: float = 1.0, smooth: float = 1.0) -> None:
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight   = ce_weight
        self.smooth      = smooth
        self.ce = nn.CrossEntropyLoss()

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        ce_loss = self.ce(logits, targets)

        # Soft Dice on probabilities
        probs   = torch.softmax(logits, dim=1)          # (B, C)
        one_hot = F.one_hot(targets, num_classes=probs.shape[1]).float()  # (B, C)
        inter   = (probs * one_hot).sum(0)              # (C,)
        denom   = probs.sum(0) + one_hot.sum(0)         # (C,)
        dice    = 1.0 - (2.0 * inter + self.smooth) / (denom + self.smooth)
        dice_loss = dice.mean()

        total = self.dice_weight * dice_loss + self.ce_weight * ce_loss
        metrics = {
            "loss_ce":    float(ce_loss.detach().item()),
            "loss_dice":  float(dice_loss.detach().item()),
            "loss_total": float(total.detach().item()),
        }
        return total, metrics


# ---------------------------------------------------------------------------
# Legacy compatibility aliases (so old imports don't break)
# ---------------------------------------------------------------------------
Denoise25DUNetSmall        = Denoise25DUNet          # Stage 1 (upgraded)
Recon3DAttentionUNetSmall  = Recon3DUNet             # Stage 2 (upgraded)
PhysicsGuidedReconLoss     = Recon3DLoss             # Stage 2 loss (upgraded)
NoduleClassifier3D         = NoduleDetector3D        # Stage 3 (upgraded)
