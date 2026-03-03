from __future__ import annotations

import argparse
from pathlib import Path
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import yaml
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from ailung.models import Denoise25DUNetSmall
from ailung.splits import load_split
from ailung.torch_dataset import LIDCDenoise25DDataset


def _resolve_device(device_cfg: str) -> torch.device:
    if device_cfg == "cpu":
        return torch.device("cpu")
    if device_cfg == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_loader(split_entries: list[dict], cfg: dict, split_name: str) -> DataLoader:
    max_cases = cfg["data"]["max_cases_per_split"].get(split_name)
    ds = LIDCDenoise25DDataset(
        split_entries=split_entries,
        hu_min=cfg["data"]["hu_min"],
        hu_max=cfg["data"]["hu_max"],
        context_slices=cfg["data"]["context_slices"],
        max_cases=max_cases,
    )
    loader = DataLoader(
        ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=(split_name == "train"),
        num_workers=cfg["train"]["num_workers"],
        pin_memory=False,
    )
    return loader


def _compute_psnr_ssim(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    y_true = np.clip(y_true.astype(np.float32), 0.0, 1.0)
    y_pred = np.clip(y_pred.astype(np.float32), 0.0, 1.0)
    psnr = float(peak_signal_noise_ratio(y_true, y_pred, data_range=1.0))
    ssim = float(structural_similarity(y_true, y_pred, data_range=1.0))
    return psnr, ssim


def _save_preview(epoch: int, output_dir: Path, x: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    preview_dir = output_dir / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(x, cmap="gray")
    axes[0].set_title("Input (center slice)")
    axes[0].axis("off")

    axes[1].imshow(y_true, cmap="gray")
    axes[1].set_title("Target")
    axes[1].axis("off")

    axes[2].imshow(y_pred, cmap="gray")
    axes[2].set_title("Prediction")
    axes[2].axis("off")

    fig.tight_layout()
    fig.savefig(preview_dir / f"epoch_{epoch + 1:03d}.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train 2.5D denoiser baseline")
    parser.add_argument("--config", type=str, default="d:/AI_LUNG/configs/baseline.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(cfg["seed"])

    split = load_split(cfg["data"]["splits_path"])
    train_loader = _build_loader(split["train"], cfg, "train")
    val_loader = _build_loader(split["val"], cfg, "val")

    in_channels = cfg["data"]["context_slices"] * 2 + 1
    model = Denoise25DUNetSmall(in_channels=in_channels, out_channels=1)

    device = _resolve_device(cfg["runtime"]["device"])
    model = model.to(device)

    criterion = nn.L1Loss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["learning_rate"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )

    output_dir = Path(cfg["train"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    history = {"train_loss": [], "val_loss": [], "val_psnr": [], "val_ssim": []}
    best_psnr = -float("inf")
    start_epoch = 0

    # --- Resume from checkpoint if available ---
    resume_path = output_dir / "denoiser_last.pt"
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
        print(f"Resumed from epoch {start_epoch}. Best PSNR so far: {best_psnr:.4f}", flush=True)

    for epoch in range(start_epoch, cfg["train"]["epochs"]):
        model.train()
        train_loss_sum = 0.0
        train_steps = 0

        for batch_idx, batch in enumerate(train_loader):
            x = batch["x"].to(device)
            y = batch["y"].to(device)

            pred = model(x)
            loss = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += float(loss.item())
            train_steps += 1

            # Print batch progress every 50 steps
            if (batch_idx + 1) % 50 == 0:
                print(
                    f"  Epoch {epoch+1}/{cfg['train']['epochs']} "
                    f"batch {batch_idx+1}/{len(train_loader)} "
                    f"loss={train_loss_sum/train_steps:.6f}",
                    flush=True,
                )

        train_loss = train_loss_sum / max(train_steps, 1)

        model.eval()
        val_loss_sum = 0.0
        val_steps = 0
        val_psnr_sum = 0.0
        val_ssim_sum = 0.0
        val_image_count = 0
        preview_saved = False
        with torch.no_grad():
            for batch in val_loader:
                x = batch["x"].to(device)
                y = batch["y"].to(device)
                pred = model(x)
                loss = criterion(pred, y)
                val_loss_sum += float(loss.item())
                val_steps += 1

                y_np = y.detach().cpu().numpy()[:, 0, :, :]
                pred_np = pred.detach().cpu().numpy()[:, 0, :, :]
                x_center_np = x.detach().cpu().numpy()[:, x.shape[1] // 2, :, :]

                for i in range(y_np.shape[0]):
                    psnr, ssim = _compute_psnr_ssim(y_np[i], pred_np[i])
                    val_psnr_sum += psnr
                    val_ssim_sum += ssim
                    val_image_count += 1

                if not preview_saved and y_np.shape[0] > 0:
                    _save_preview(epoch, output_dir, x_center_np[0], y_np[0], pred_np[0])
                    preview_saved = True

        val_loss = val_loss_sum / max(val_steps, 1)
        val_psnr = val_psnr_sum / max(val_image_count, 1)
        val_ssim = val_ssim_sum / max(val_image_count, 1)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_psnr"].append(val_psnr)
        history["val_ssim"].append(val_ssim)

        print(
            f"Epoch {epoch + 1}/{cfg['train']['epochs']} — "
            f"train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
            f"val_psnr={val_psnr:.4f} val_ssim={val_ssim:.4f}",
            flush=True,
        )

        # Save latest checkpoint every epoch (resume-safe)
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": cfg,
        }, output_dir / "denoiser_last.pt")

        # Save best checkpoint
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "config": cfg,
            }, output_dir / "denoiser_best.pt")
            print(f"  -> New best PSNR: {best_psnr:.4f} — saved denoiser_best.pt", flush=True)

        # Save history every epoch (persists progress)
        with hist_path.open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    print("Training complete.", flush=True)


if __name__ == "__main__":
    main()
