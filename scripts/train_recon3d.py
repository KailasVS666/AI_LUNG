from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from ailung.models import PhysicsGuidedReconLoss, Recon3DAttentionUNetSmall
from ailung.splits import load_split
from ailung.torch_dataset import LIDCRecon3DPatchDataset


def _resolve_device(device_cfg: str) -> torch.device:
    if device_cfg == "cpu":
        return torch.device("cpu")
    if device_cfg == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _compute_psnr_ssim(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    y_true = np.clip(y_true.astype(np.float32), 0.0, 1.0)
    y_pred = np.clip(y_pred.astype(np.float32), 0.0, 1.0)
    psnr = float(peak_signal_noise_ratio(y_true, y_pred, data_range=1.0))
    ssim = float(structural_similarity(y_true, y_pred, data_range=1.0))
    return psnr, ssim


def _save_preview(epoch: int, output_dir: Path, x: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    preview_dir = output_dir / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)

    mid = x.shape[0] // 2

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(x[mid], cmap="gray")
    axes[0].set_title("Input noisy (mid-z)")
    axes[0].axis("off")

    axes[1].imshow(y_true[mid], cmap="gray")
    axes[1].set_title("Target clean (mid-z)")
    axes[1].axis("off")

    axes[2].imshow(y_pred[mid], cmap="gray")
    axes[2].set_title("Prediction (mid-z)")
    axes[2].axis("off")

    fig.tight_layout()
    fig.savefig(preview_dir / f"epoch_{epoch + 1:03d}.png", dpi=150)
    plt.close(fig)


def _build_loader(split_entries: list[dict], cfg: dict, split_name: str) -> DataLoader:
    ds = LIDCRecon3DPatchDataset(
        split_entries=split_entries,
        hu_min=cfg["data"]["hu_min"],
        hu_max=cfg["data"]["hu_max"],
        patch_size=tuple(cfg["data"]["patch_size"]),
        patches_per_volume=cfg["data"]["patches_per_volume"][split_name],
        noise_std=float(cfg["data"]["noise_std"]),
        max_cases=cfg["data"]["max_cases_per_split"][split_name],
        seed=int(cfg["seed"]),
    )
    return DataLoader(
        ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=(split_name == "train"),
        num_workers=cfg["train"]["num_workers"],
        pin_memory=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train 3D reconstruction model with physics-guided loss")
    parser.add_argument("--config", type=str, default="d:/AI_LUNG/configs/recon3d.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(int(cfg["seed"]))

    split = load_split(cfg["data"]["splits_path"])
    train_loader = _build_loader(split["train"], cfg, "train")
    val_loader = _build_loader(split["val"], cfg, "val")

    model = Recon3DAttentionUNetSmall(base_channels=int(cfg["model"]["base_channels"]))
    criterion = PhysicsGuidedReconLoss(
        l1_weight=float(cfg["model"]["l1_weight"]),
        grad_weight=float(cfg["model"]["grad_weight"]),
        range_weight=float(cfg["model"]["range_weight"]),
    )

    device = _resolve_device(cfg["runtime"]["device"])
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["learning_rate"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )

    output_dir = Path(cfg["train"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_psnr": [],
        "val_ssim": [],
        "val_loss_recon": [],
        "val_loss_grad": [],
        "val_loss_range": [],
    }

    epochs = int(cfg["train"]["epochs"])
    for epoch in range(epochs):
        model.train()
        train_loss_sum = 0.0
        train_steps = 0

        for batch in train_loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)

            pred = model(x)
            loss, _ = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += float(loss.item())
            train_steps += 1

        train_loss = train_loss_sum / max(train_steps, 1)

        model.eval()
        val_loss_sum = 0.0
        val_steps = 0
        val_psnr_sum = 0.0
        val_ssim_sum = 0.0
        val_img_count = 0
        val_recon_sum = 0.0
        val_grad_sum = 0.0
        val_range_sum = 0.0
        preview_saved = False

        with torch.no_grad():
            for batch in val_loader:
                x = batch["x"].to(device)
                y = batch["y"].to(device)
                pred = model(x)

                loss, metrics = criterion(pred, y)
                val_loss_sum += float(loss.item())
                val_steps += 1

                val_recon_sum += metrics["loss_recon"]
                val_grad_sum += metrics["loss_grad"]
                val_range_sum += metrics["loss_range"]

                y_np = y.detach().cpu().numpy()[:, 0, :, :, :]
                pred_np = pred.detach().cpu().numpy()[:, 0, :, :, :]
                x_np = x.detach().cpu().numpy()[:, 0, :, :, :]

                for i in range(y_np.shape[0]):
                    mid = y_np[i].shape[0] // 2
                    psnr, ssim = _compute_psnr_ssim(y_np[i][mid], pred_np[i][mid])
                    val_psnr_sum += psnr
                    val_ssim_sum += ssim
                    val_img_count += 1

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
        history["val_loss_recon"].append(val_recon_sum / max(val_steps, 1))
        history["val_loss_grad"].append(val_grad_sum / max(val_steps, 1))
        history["val_loss_range"].append(val_range_sum / max(val_steps, 1))

        print(
            f"Epoch {epoch + 1}/{epochs} - train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
            f"val_psnr={val_psnr:.4f} val_ssim={val_ssim:.4f}"
        )

    ckpt_path = output_dir / "recon3d_last.pt"
    torch.save({"model_state_dict": model.state_dict(), "config": cfg}, ckpt_path)

    hist_path = output_dir / "history.json"
    with hist_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print("Saved checkpoint:", ckpt_path)
    print("Saved history:", hist_path)


if __name__ == "__main__":
    main()
