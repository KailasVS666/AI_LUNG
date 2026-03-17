"""
Stage 1 Training: 2.5D Denoising
==================================
Model : Denoise25DUNet  (EfficientNet-B5 encoder + CBAM attention)
Loss  : DenoiseLoss     (L1 + (1 − SSIM) + Gradient)
Input : 9 consecutive low-dose simulated slices  (Radon + Poisson + FBP)
Output: 1 denoised normal-dose central slice

After training, run:
    python scripts/export_denoised.py --config configs/baseline_colab.yaml
to produce the denoised .npy volumes that Stage 2 expects.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

# Add src to sys.path to ensure 'ailung' module is found
repo_root = Path(__file__).resolve().parent.parent
sys.path.append(str(repo_root / "src"))
import json
import warnings
warnings.filterwarnings("ignore", category=UserWarning)   # suppress Radon warning

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from ailung.models import Denoise25DUNet, DenoiseLoss
from ailung.splits import load_split
from ailung.torch_dataset import LIDCDenoise25DDataset, GroupedSeriesSampler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_device(device_cfg: str) -> torch.device:
    if device_cfg == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_npy_mapping(cfg: dict) -> dict[str, str]:
    """Load path→npy mapping from preprocess_to_npy.py output, if it exists."""
    npy_dir = cfg["data"].get("preprocessed_npy_dir")
    if not npy_dir:
        npy_dir = str(Path(cfg["train"]["output_dir"]).parent / "preprocessed_npy")
    mapping_path = Path(npy_dir) / "path_to_npy.json"
    if mapping_path.exists():
        with open(mapping_path) as f:
            mapping = json.load(f)
        return mapping
    return {}


def _sync_to_local_disk(cfg: dict, split_entries: list[dict]):
    """Best-effort sync: Copy files to local NVMe until disk is 80% full.
    Prevents VM crashes while maximizing speed."""
    drive_npy = cfg["data"].get("preprocessed_npy_dir")
    local_npy = cfg["data"].get("local_npy_cache")
    if not local_npy or not drive_npy: return None

    npy_mapping = _load_npy_mapping(cfg)
    needed_files = [Path(npy_mapping[e["file_location"]]).name for e in split_entries if e["file_location"] in npy_mapping]
    if not needed_files: return None

    import shutil
    import subprocess
    os.makedirs(local_npy, exist_ok=True)
    
    # Shuffle so we get a diverse set of patients on local disk if we hit the limit
    random.shuffle(needed_files)

    print(f"\n🚀 Turbo-charging: Best-effort sync to local NVMe...", flush=True)
    count = 0
    # Copy in batches to check disk space frequently
    batch_size = 20
    for i in range(0, len(needed_files), batch_size):
        # Safety Check: Stop at 80% disk usage to leave room for checkpoints/OS
        total, used, free = shutil.disk_usage("/content")
        percent_used = (used / total) * 100
        if percent_used > 80:
            print(f"⚠️ Disk 80% full ({percent_used:.1f}%). Stopping sync to keep session stable.", flush=True)
            break
            
        batch = needed_files[i : i + batch_size]
        # Use rsync for the batch (very fast)
        list_file = Path("/tmp/sync_batch.txt")
        with open(list_file, "w") as f:
            for fname in batch: f.write(f"{fname}\n")
        
        cmd = ["rsync", "-rLt", "--size-only", "--files-from=" + str(list_file), str(drive_npy) + "/", str(local_npy) + "/"]
        # Remove capture_output=True so we don't buffer the progress
        subprocess.run(cmd, check=False) 
        count += len(batch)
        print(f"  Synced {count}/{len(needed_files)} series... ({percent_used:.1f}% disk used)", flush=True)

    print(f"\n✅ Sync complete ({count} series cached locally). Training starts now!", flush=True)
    return local_npy


def _build_loader(split_entries: list[dict], cfg: dict, split_name: str,
                  npy_mapping: dict | None = None, local_cache: str | None = None,
                  start_batch: int = 0) -> tuple[DataLoader, LIDCDenoise25DDataset]:
    max_cases = cfg["data"]["max_cases_per_split"].get(split_name)
    ds = LIDCDenoise25DDataset(
        split_entries=split_entries,
        hu_min=cfg["data"]["hu_min"],
        hu_max=cfg["data"]["hu_max"],
        context_slices=cfg["data"]["context_slices"],
        max_cases=max_cases,
        apply_clahe_flag=cfg["data"].get("apply_clahe", True),
        low_dose_i0=float(cfg["data"].get("low_dose_i0", 1e5)),
        seed=int(cfg["seed"]),
        fast_mode=bool(cfg["data"].get("fast_mode", False)),
        fast_mode_noise_std=float(cfg["data"].get("fast_mode_noise_std", 0.05)),
        max_samples_per_series=cfg["data"].get("max_samples_per_series", 64),
        npy_mapping=npy_mapping,
        local_cache_path=local_cache,
    )
    is_train = (split_name == "train")
    sampler  = GroupedSeriesSampler(ds, batch_size=cfg["train"]["batch_size"], 
                                    shuffle=is_train, seed=int(cfg["seed"]), 
                                    start_offset=(start_batch if is_train else 0))
    loader   = DataLoader(
        ds,
        batch_size=cfg["train"]["batch_size"],
        sampler=sampler,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )
    return loader, ds


def _compute_psnr_ssim(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    y_true = np.clip(y_true.astype(np.float32), 0.0, 1.0)
    y_pred = np.clip(y_pred.astype(np.float32), 0.0, 1.0)
    psnr = float(peak_signal_noise_ratio(y_true, y_pred, data_range=1.0))
    ssim = float(structural_similarity(y_true, y_pred, data_range=1.0))
    return psnr, ssim


def _save_preview(epoch: int, output_dir: Path,
                  x_ld: np.ndarray, y_nd: np.ndarray, y_pred: np.ndarray) -> None:
    preview_dir = output_dir / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    center = x_ld.shape[0] // 2
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(x_ld[center], cmap="gray"); axes[0].set_title("Low-Dose Input"); axes[0].axis("off")
    axes[1].imshow(y_nd,         cmap="gray"); axes[1].set_title("Normal-Dose Target"); axes[1].axis("off")
    axes[2].imshow(y_pred,       cmap="gray"); axes[2].set_title("Denoised Output"); axes[2].axis("off")
    fig.tight_layout()
    fig.savefig(preview_dir / f"epoch_{epoch + 1:03d}.png", dpi=120)
    plt.close(fig)


def _plot_history(history: dict, output_dir: Path) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(epochs, history["train_loss"], label="Train"); axes[0].plot(epochs, history["val_loss"], label="Val")
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(True)
    axes[1].plot(epochs, history["val_psnr"], "g-o"); axes[1].set_title("Val PSNR (dB)"); axes[1].grid(True)
    axes[2].plot(epochs, history["val_ssim"], "b-o"); axes[2].set_title("Val SSIM"); axes[2].grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "training_curves.png", dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train Stage 1 — 2.5D Denoising")
    parser.add_argument("--config", type=str, default="configs/baseline.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(cfg["seed"])

    # Early stopping config (with sensible defaults)
    early_stop_patience = int(cfg["train"].get("early_stop_patience", 7))
    early_stop_min_delta = float(cfg["train"].get("early_stop_min_delta", 0.01))

    # --- 1. Runtime Config & Globals ---
    use_amp    = cfg["runtime"].get("mixed_precision", False) and torch.cuda.is_available()
    scaler     = torch.amp.GradScaler("cuda") if use_amp else None
    output_dir = Path(cfg["train"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    device = _resolve_device(cfg["runtime"]["device"])
    in_channels = cfg["data"]["context_slices"] * 2 + 1
    model = Denoise25DUNet(in_channels=in_channels, out_channels=1).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["learning_rate"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    history     = {"train_loss": [], "val_loss": [], "val_psnr": [], "val_ssim": []}
    best_psnr   = -float("inf")
    start_epoch = 0
    start_batch = -1
    no_improve_count = 0 

    # --- 2. RESUME Logic ---
    resume_path = output_dir / "denoiser_last.pt"
    hist_path   = output_dir / "history.json"
    if resume_path.exists():
        print(f"Resuming from {resume_path}", flush=True)
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch      = ckpt.get("epoch", 0)
        start_batch      = ckpt.get("batch_idx", -1)
        no_improve_count = ckpt.get("no_improve_count", 0)

        if hist_path.exists():
            with open(hist_path, "r", encoding="utf-8") as f:
                history = json.load(f)
            if history["val_psnr"]:
                best_psnr = max(history["val_psnr"])
        
        batch_msg = f"Batch: {start_batch}" if start_batch > 0 else "Start of Epoch"
        print(f"  ✓ Resumed Epoch: {start_epoch} | {batch_msg} | Best PSNR: {best_psnr:.2f} dB", flush=True)

    # --- 3. Load Data & Sync ---
    npy_mapping = _load_npy_mapping(cfg)
    split       = load_split(cfg["data"]["splits_path"])
    if npy_mapping:
        print(f"  [npy] Found {len(npy_mapping)} preprocessed series.", flush=True)

    needed_for_sync = split["train"] + split["val"]
    local_cache     = _sync_to_local_disk(cfg, needed_for_sync)
    
    train_loader, train_ds = _build_loader(split["train"], cfg, "train", npy_mapping, local_cache, 
                                           start_batch=(start_batch + 1 if start_batch >= 0 else 0))
    val_loader,   val_ds   = _build_loader(split["val"],   cfg, "val",   npy_mapping, local_cache)
    
    # Calculate effective train batches remaining
    total_train_batches = len(train_loader)
    print(f"  Train: {total_train_batches} batches remaining | Val: {len(val_loader)} batches", flush=True)

    criterion = DenoiseLoss(
        l1_weight=float(cfg.get("loss", {}).get("l1_weight", 1.0)),
        ssim_weight=float(cfg.get("loss", {}).get("ssim_weight", 0.5)),
        grad_weight=float(cfg.get("loss", {}).get("grad_weight", 0.2)),
    )

    total_epochs = cfg["train"]["epochs"]

    total_epochs = cfg["train"]["epochs"]

    for epoch in range(start_epoch, total_epochs):
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"\n{'='*65}", flush=True)
        print(f"  Epoch {epoch+1}/{total_epochs}  |  LR: {current_lr:.2e}  |  No-improve: {no_improve_count}/{early_stop_patience}", flush=True)
        print(f"{'='*65}", flush=True)

        # Update grouped sampler epoch (for shuffling)
        train_loader.sampler.set_epoch(epoch)

        model.train()
        train_loss_sum = 0.0
        train_steps    = 0

        # Calculate global total for progress bar clarity
        total_global_batches = len(train_loader) + (start_batch + 1 if start_batch >= 0 else 0)

        train_bar = tqdm(
            train_loader, 
            desc="  Train", 
            unit="batch",
            dynamic_ncols=True, 
            leave=True,
            initial=(start_batch + 1 if start_batch >= 0 else 0),
            total=total_global_batches
        )

        for batch_idx, batch in enumerate(train_bar):
            # The sampler now handles the skipping at the index level
            # batch_idx here will start at 0, but the data will be from (start_batch + 1)
            
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            optimizer.zero_grad()

            if use_amp:
                with autocast(device_type="cuda"):
                    pred = model(x)
                    loss, _ = criterion(pred, y)
                scaler.scale(loss).backward()
                # Gradient Clipping for Stability (Peak Performance)
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                pred = model(x)
                loss, _ = criterion(pred, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            # --- NaN Detection & Protection ---
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"⚠️ NaN detected at batch {batch_idx}! Reducing LR and skipping.", flush=True)
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.5 # Halve LR
                continue

            train_loss_sum += float(loss.item())
            train_steps    += 1
            running_avg     = train_loss_sum / train_steps

            train_bar.set_postfix(loss=f"{running_avg:.4f}")

            # --- RESILIENCE: Mid-epoch checkpoint (Every 200 batches for Colab) ---
            if (batch_idx + 1) % 200 == 0:
                current_global_batch = batch_idx + (start_batch + 1 if start_batch >= 0 else 0)
                ckpt_path = output_dir / "denoiser_last.pt"
                tmp_path  = output_dir / "denoiser_last.tmp"
                
                torch.save(
                    {
                        "epoch": epoch, 
                        "batch_idx": current_global_batch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "no_improve_count": no_improve_count,
                        "config": cfg,
                    },
                    tmp_path,
                )
                
                # Hard rename and sync for Google Drive stability
                try:
                    import os
                    os.replace(tmp_path, ckpt_path)
                    
                    # Force a sync for Google Drive cloud stability
                    if os.name != "nt": # On Linux/Colab, we can flush the directory
                        try:
                            fd = os.open(str(output_dir), os.O_RDONLY)
                            os.fsync(fd)
                            os.close(fd)
                        except: pass 
                    print(f"\n🚀 [HARD-SYNC] DRIVE UPDATED: Global Batch {current_global_batch} is now locked in!", flush=True)
                except Exception as e:
                    print(f"\n⚠️ Sync Warning: {e}", flush=True)

        train_loss = train_loss_sum / max(train_steps, 1)
        
        # Clear start_batch and reset sampler offset after first resumed epoch finishes
        start_batch = -1
        if hasattr(train_loader.sampler, 'start_offset'):
            train_loader.sampler.start_offset = 0

        # ----------------------------------------------------------------
        # VALIDATE
        # ----------------------------------------------------------------
        model.eval()
        val_loss_sum    = 0.0
        val_steps       = 0
        val_psnr_sum    = 0.0
        val_ssim_sum    = 0.0
        val_image_count = 0
        preview_saved   = False

        val_bar = tqdm(val_loader, desc="  Val  ", unit="batch",
                       dynamic_ncols=True, leave=True)

        with torch.no_grad():
            for batch in val_bar:
                x = batch["x"].to(device)
                y = batch["y"].to(device)
                pred = model(x)
                loss, _ = criterion(pred, y)
                val_loss_sum += float(loss.item())
                val_steps    += 1

                y_np    = y.detach().cpu().numpy()[:, 0]
                pred_np = pred.detach().cpu().numpy()[:, 0]
                x_np    = x.detach().cpu().numpy()

                for i in range(y_np.shape[0]):
                    psnr, ssim = _compute_psnr_ssim(y_np[i], pred_np[i])
                    val_psnr_sum    += psnr
                    val_ssim_sum    += ssim
                    val_image_count += 1

                val_bar.set_postfix(
                    loss=f"{val_loss_sum/val_steps:.4f}",
                    psnr=f"{val_psnr_sum/max(val_image_count,1):.2f}",
                )

                if not preview_saved and y_np.shape[0] > 0:
                    _save_preview(epoch, output_dir, x_np[0], y_np[0], pred_np[0])
                    preview_saved = True

        val_loss = val_loss_sum  / max(val_steps, 1)
        val_psnr = val_psnr_sum  / max(val_image_count, 1)
        val_ssim = val_ssim_sum  / max(val_image_count, 1)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_psnr"].append(val_psnr)
        history["val_ssim"].append(val_ssim)

        # ----------------------------------------------------------------
        # Summary line
        # ----------------------------------------------------------------
        print(
            f"\n  Summary  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"val_psnr={val_psnr:.2f} dB  val_ssim={val_ssim:.4f}",
            flush=True,
        )

        # ----------------------------------------------------------------
        # LR Scheduler
        # ----------------------------------------------------------------
        scheduler.step(val_psnr)

        # ----------------------------------------------------------------
        # Checkpointing
        # ----------------------------------------------------------------
        # Always save last (resume-safe)
        torch.save(
            {
                "epoch": epoch + 1,
                "batch_idx": -1, # Reset batch idx for full resume
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "no_improve_count": no_improve_count,
                "config": cfg,
            },
            output_dir / "denoiser_last.pt",
        )

        # Save best
        improvement = val_psnr - best_psnr
        if improvement > early_stop_min_delta:
            best_psnr = val_psnr
            no_improve_count = 0
            torch.save(
                {"epoch": epoch + 1, "model_state_dict": model.state_dict(), "config": cfg},
                output_dir / "denoiser_best.pt",
            )
            print(f"  ✓ New best PSNR: {best_psnr:.2f} dB — saved denoiser_best.pt", flush=True)
        else:
            no_improve_count += 1
            print(
                f"  ✗ No improvement ({improvement:+.4f} dB).  "
                f"Patience: {no_improve_count}/{early_stop_patience}",
                flush=True,
            )

        # Save history + curves
        with hist_path.open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        _plot_history(history, output_dir)

        # ----------------------------------------------------------------
        # Early stopping
        # ----------------------------------------------------------------
        if no_improve_count >= early_stop_patience:
            print(
                f"\n  *** EARLY STOPPING triggered after {epoch+1} epochs ***\n"
                f"  Val PSNR did not improve by >{early_stop_min_delta} dB "
                f"for {early_stop_patience} consecutive epochs.\n"
                f"  Best PSNR: {best_psnr:.2f} dB",
                flush=True,
            )
            break

    print("\nTraining complete.", flush=True)
    print(f"Best Val PSNR: {best_psnr:.2f} dB", flush=True)
    print(f"Checkpoints saved to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
