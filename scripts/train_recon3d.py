"""
Stage 2 Training: 3D Reconstruction
======================================
Model : Recon3DUNet   (Lightweight 3D U-Net with 3D-CBAM attention)
Loss  : Recon3DLoss   (L1 + SSIM3D + Gradient3D + Projection Consistency)
Input : 64×64×64 patches from DENOISED volumes (Stage 1 output .npy files)
Output: Refined 3D lung volume patches

Prerequisites:
    python scripts/export_denoised.py --config configs/baseline_colab.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
import yaml
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from ailung.models import Recon3DUNet, Recon3DLoss
from ailung.splits import load_split
from ailung.torch_dataset import LIDCRecon3DPatchDataset


def _resolve_device(device_cfg: str) -> torch.device:
    if device_cfg == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _compute_psnr_ssim(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    y_true = np.clip(y_true.astype(np.float32), 0.0, 1.0)
    y_pred = np.clip(y_pred.astype(np.float32), 0.0, 1.0)
    psnr = float(peak_signal_noise_ratio(y_true, y_pred, data_range=1.0))
    ssim = float(structural_similarity(y_true, y_pred, data_range=1.0))
    return psnr, ssim


def _save_preview(
    epoch: int, output_dir: Path, x: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray
) -> None:
    preview_dir = output_dir / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    mid = x.shape[0] // 2

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(x[mid], cmap="gray");     axes[0].set_title("Denoised Input (mid-z)"); axes[0].axis("off")
    axes[1].imshow(y_true[mid], cmap="gray"); axes[1].set_title("Target clean (mid-z)");  axes[1].axis("off")
    axes[2].imshow(y_pred[mid], cmap="gray"); axes[2].set_title("Reconstruction (mid-z)"); axes[2].axis("off")
    fig.tight_layout()
    fig.savefig(preview_dir / f"epoch_{epoch+1:03d}.png", dpi=150)
    plt.close(fig)


def _build_loader(split_entries: list[dict], cfg: dict, split_name: str) -> DataLoader:
    ds = LIDCRecon3DPatchDataset(
        split_entries=split_entries,
        hu_min=cfg["data"]["hu_min"],
        hu_max=cfg["data"]["hu_max"],
        patch_size=tuple(cfg["data"]["patch_size"]),
        patches_per_volume=cfg["data"]["patches_per_volume"][split_name],
        max_cases=cfg["data"]["max_cases_per_split"][split_name],
        seed=int(cfg["seed"]),
        denoised_vol_dir=cfg["data"].get("denoised_vol_dir", None),
        noise_std=float(cfg["data"].get("noise_std", 0.03)),
    )
    return DataLoader(
        ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=(split_name == "train"),
        num_workers=cfg["train"]["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Stage 2 — 3D Reconstruction")
    parser.add_argument("--config", type=str, default="configs/recon3d.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(int(cfg["seed"]))

    split = load_split(cfg["data"]["splits_path"])
    train_loader = _build_loader(split["train"], cfg, "train")
    val_loader   = _build_loader(split["val"],   cfg, "val")

    model = Recon3DUNet(base_channels=int(cfg["model"]["base_channels"]))

    criterion = Recon3DLoss(
        l1_weight=float(cfg["model"].get("l1_weight", 1.0)),
        ssim_weight=float(cfg["model"].get("ssim_weight", 0.5)),
        grad_weight=float(cfg["model"].get("grad_weight", 0.2)),
        proj_weight=float(cfg["model"].get("proj_weight", 0.1)),
    )

    device = _resolve_device(cfg["runtime"]["device"])
    model  = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["learning_rate"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )

    use_amp = cfg["runtime"].get("mixed_precision", False) and torch.cuda.is_available()
    scaler  = GradScaler() if use_amp else None

    output_dir = Path(cfg["train"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    history = {
        "train_loss": [], "val_loss": [], "val_psnr": [], "val_ssim": [],
        "val_loss_l1": [], "val_loss_ssim": [], "val_loss_grad": [], "val_loss_proj": [],
    }
    best_psnr   = -float("inf")
    start_epoch = 0

    # Resume support
    resume_path = output_dir / "recon3d_last.pt"
    hist_path   = output_dir / "history.json"
    if resume_path.exists():
        print(f"Resuming from {resume_path}", flush=True)
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0)
        if hist_path.exists():
            with hist_path.open() as f:
                history = json.load(f)
            best_psnr = max(history["val_psnr"]) if history["val_psnr"] else best_psnr
        print(f"Resumed from epoch {start_epoch}. Best PSNR: {best_psnr:.4f}", flush=True)

    epochs = int(cfg["train"]["epochs"])
    for epoch in range(start_epoch, epochs):
        # ---- TRAIN ----
        model.train()
        train_loss_sum = 0.0
        train_steps    = 0

        for batch in train_loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)

            optimizer.zero_grad()
            if use_amp:
                with autocast():
                    pred = model(x)
                    loss, _ = criterion(pred, y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                pred = model(x)
                loss, _ = criterion(pred, y)
                loss.backward()
                optimizer.step()

            train_loss_sum += float(loss.item())
            train_steps    += 1

        train_loss = train_loss_sum / max(train_steps, 1)

        # ---- VALIDATE ----
        model.eval()
        val_loss_sum   = 0.0; val_steps     = 0
        val_psnr_sum   = 0.0; val_ssim_sum  = 0.0; val_img_count = 0
        val_l1_sum     = 0.0; val_ssim_l_sum = 0.0
        val_grad_sum   = 0.0; val_proj_sum  = 0.0
        preview_saved  = False

        with torch.no_grad():
            for batch in val_loader:
                x = batch["x"].to(device)
                y = batch["y"].to(device)
                pred = model(x)
                loss, metrics = criterion(pred, y)
                val_loss_sum    += float(loss.item()); val_steps += 1
                val_l1_sum      += metrics["loss_l1"]
                val_ssim_l_sum  += metrics["loss_ssim"]
                val_grad_sum    += metrics["loss_grad"]
                val_proj_sum    += metrics["loss_proj"]

                y_np    = y.detach().cpu().numpy()[:, 0]
                pred_np = pred.detach().cpu().numpy()[:, 0]
                x_np    = x.detach().cpu().numpy()[:, 0]

                for i in range(y_np.shape[0]):
                    mid = y_np[i].shape[0] // 2
                    psnr, ssim = _compute_psnr_ssim(y_np[i][mid], pred_np[i][mid])
                    val_psnr_sum += psnr; val_ssim_sum += ssim; val_img_count += 1

                if not preview_saved and y_np.shape[0] > 0:
                    _save_preview(epoch, output_dir, x_np[0], y_np[0], pred_np[0])
                    preview_saved = True

        val_loss = val_loss_sum / max(val_steps, 1)
        val_psnr = val_psnr_sum / max(val_img_count, 1)
        val_ssim = val_ssim_sum / max(val_img_count, 1)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_psnr"].append(val_psnr)
        history["val_ssim"].append(val_ssim)
        history["val_loss_l1"].append(val_l1_sum / max(val_steps, 1))
        history["val_loss_ssim"].append(val_ssim_l_sum / max(val_steps, 1))
        history["val_loss_grad"].append(val_grad_sum / max(val_steps, 1))
        history["val_loss_proj"].append(val_proj_sum / max(val_steps, 1))

        print(
            f"Epoch {epoch+1}/{epochs} - train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f} val_psnr={val_psnr:.4f} val_ssim={val_ssim:.4f}",
            flush=True,
        )

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": cfg,
            },
            output_dir / "recon3d_last.pt",
        )

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save(
                {"epoch": epoch + 1, "model_state_dict": model.state_dict(), "config": cfg},
                output_dir / "recon3d_best.pt",
            )
            print(f"  -> New best PSNR: {best_psnr:.4f} — saved recon3d_best.pt", flush=True)

        with hist_path.open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    print("Training complete.", flush=True)


if __name__ == "__main__":
    main()
