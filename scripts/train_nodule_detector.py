"""
Stage 3 Training: Nodule Detection & Classification
======================================================
Model : NoduleDetector3D   (3D CNN with CBAM attention)
Loss  : NoduleDetectionLoss (Dice + Cross-Entropy)
Input : 32×64×64 patches from RECONSTRUCTED 3D volumes (Stage 2 output)
Detects nodules as small as 3–5 mm; classifies as benign (0) / malignant (1)

Prerequisites:
    python scripts/export_denoised.py --config configs/baseline_colab.yaml
    python scripts/train_recon3d.py   --config configs/recon3d_colab.yaml
    (optionally: python scripts/export_recon3d.py for pre-saved recon volumes)
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
import json

import yaml
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

from ailung.splits import load_split
from ailung.torch_dataset import NoduleDetectionDataset
from ailung.models import NoduleDetector3D, NoduleDetectionLoss


def compute_metrics(y_true: np.ndarray, y_pred_proba: np.ndarray, threshold: float = 0.5, num_classes: int = 2) -> dict:
    if num_classes == 6:
        pred_class = np.argmax(y_pred_proba, axis=1)
        correct = (pred_class == y_true).sum()
        accuracy = correct / len(y_true)
        
        # Detection metrics (group 0-4 as Nodule, 5 as Background)
        y_true_binary = (y_true < 5).astype(int)
        y_pred_proba_binary = 1.0 - y_pred_proba[:, 5]
        y_pred_binary = (y_pred_proba_binary >= threshold).astype(int)
        
        cm = confusion_matrix(y_true_binary, y_pred_binary, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        try:
            auc = roc_auc_score(y_true_binary, y_pred_proba_binary)
        except Exception:
            auc = 0.0
            
        return {
            "auc": float(auc),
            "sensitivity": float(sensitivity),
            "specificity": float(specificity),
            "accuracy": float(accuracy),
            "detection_accuracy": float((tp + tn) / (tp + tn + fp + fn)),
            "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        }
    else:
        y_pred = (y_pred_proba >= threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        try:
            auc = roc_auc_score(y_true, y_pred_proba)
        except Exception:
            auc = 0.0
        return {
            "auc": float(auc),
            "sensitivity": float(sensitivity),
            "specificity": float(specificity),
            "accuracy": float((tp + tn) / (tp + tn + fp + fn)),
            "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        }


def validate_epoch(model, loader, device, threshold: float = 0.5, num_classes: int = 2) -> dict:
    model.eval()
    all_labels, all_probs = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            logits = model(x)
            if num_classes == 6:
                probs = torch.softmax(logits, dim=1)
                all_probs.extend(probs.cpu().numpy())
            else:
                probs = torch.softmax(logits, dim=1)[:, 1]
                all_probs.extend(probs.cpu().numpy())
            all_labels.extend(batch["y"].numpy())
            
    target_metrics = compute_metrics(
        np.array(all_labels), 
        np.array(all_probs), 
        threshold=threshold, 
        num_classes=num_classes
    )
    target_metrics["all_labels"] = np.array(all_labels)
    target_metrics["all_probs"] = np.array(all_probs)
    return target_metrics


def train_one_epoch(model, loader, criterion, optimizer, device, use_amp, scaler) -> float:
    model.train()
    total_loss = 0.0; count = 0
    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        optimizer.zero_grad()
        if use_amp:
            with autocast():
                logits = model(x)
                loss, _ = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss, _ = criterion(logits, y)
            loss.backward()
            optimizer.step()
        total_loss += loss.item(); count += 1
    return total_loss / count if count > 0 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Stage 3 — Nodule Detection")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    xml_dir    = Path(config["xml_dir"])
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    batch_size            = config.get("batch_size", 4)
    epochs                = config.get("epochs", 30)
    lr                    = config.get("learning_rate", 1e-4)
    patch_size            = tuple(config.get("patch_size", [32, 64, 64]))
    min_malignancy        = config.get("min_malignancy", 1)
    negatives_per_positive = config.get("negatives_per_positive", 2)
    max_cases_train       = config.get("max_cases_train", None)
    max_cases_val         = config.get("max_cases_val", None)
    hu_min                = config.get("hu_min", -1000)
    hu_max                = config.get("hu_max", 400)
    reconstructed_vol_dir = config.get("reconstructed_vol_dir", None)
    use_amp               = config.get("mixed_precision", False) and torch.cuda.is_available()
    pos_weight            = config.get("pos_weight", None)
    decision_threshold    = config.get("decision_threshold", 0.5)
    num_classes           = config.get("num_classes", 6)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    splits = load_split(config["split_json"])
    train_entries = splits["train"]
    val_entries   = splits["val"]
    print(f"Train series: {len(train_entries)}, Val series: {len(val_entries)}", flush=True)

    print("Building train dataset...", flush=True)
    train_ds = NoduleDetectionDataset(
        train_entries, xml_dir=xml_dir, hu_min=hu_min, hu_max=hu_max,
        patch_size=patch_size, min_malignancy=min_malignancy,
        negatives_per_positive=negatives_per_positive,
        max_cases=max_cases_train, seed=42,
        reconstructed_vol_dir=reconstructed_vol_dir,
    )
    pos = sum(1 for s in train_ds.samples if s["label"] == 1)
    print(f"Train: {len(train_ds)} samples (pos={pos}, neg={len(train_ds)-pos})", flush=True)

    print("Building val dataset...", flush=True)
    val_ds = NoduleDetectionDataset(
        val_entries, xml_dir=xml_dir, hu_min=hu_min, hu_max=hu_max,
        patch_size=patch_size, min_malignancy=min_malignancy,
        negatives_per_positive=negatives_per_positive,
        max_cases=max_cases_val, seed=123,
        reconstructed_vol_dir=reconstructed_vol_dir,
    )
    pos_v = sum(1 for s in val_ds.samples if s["label"] == 1)
    print(f"Val: {len(val_ds)} samples (pos={pos_v}, neg={len(val_ds)-pos_v})", flush=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2)

    model     = NoduleDetector3D(in_channels=1, base_channels=16, num_classes=num_classes).to(device)
    criterion = NoduleDetectionLoss(dice_weight=1.0, ce_weight=1.0, pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scaler    = GradScaler() if use_amp else None

    history     = {"train_loss": [], "val_metrics": []}
    best_auc    = 0.0
    start_epoch = 1

    # --- Resume Logic ---
    resume_path = output_dir / "nodule_detector_last.pt"
    if not resume_path.exists():
        resume_path = output_dir / "nodule_detector_best.pt"
    
    hist_path = output_dir / "history.json"
    if resume_path.exists():
        print(f"Resuming Stage 3 from {resume_path}", flush=True)
        ckpt = torch.load(resume_path, map_location=device)
        
        # Check for model state dict keys matching
        is_dict_ckpt = isinstance(ckpt, dict) and "model_state_dict" in ckpt
        state_dict_to_check = ckpt["model_state_dict"] if is_dict_ckpt else ckpt
        
        model_keys = set(model.state_dict().keys())
        ckpt_keys = set(state_dict_to_check.keys())
        
        # Verify both key set equality and size matching for first convolution
        keys_match = (ckpt_keys == model_keys)
        if keys_match:
            try:
                first_key = "features.0.weight"
                if first_key in state_dict_to_check:
                    keys_match = (state_dict_to_check[first_key].shape == model.state_dict()[first_key].shape)
            except Exception:
                keys_match = False

        if keys_match:
            if is_dict_ckpt:
                model.load_state_dict(ckpt["model_state_dict"])
                if "optimizer_state_dict" in ckpt:
                    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                start_epoch = ckpt.get("epoch", 1)
            else:
                model.load_state_dict(ckpt)
                if hist_path.exists():
                    with open(hist_path, "r") as f:
                        history = json.load(f)
                    start_epoch = len(history["train_loss"]) + 1
            
            if hist_path.exists():
                with open(hist_path, "r") as f:
                    history = json.load(f)
                if history.get("val_metrics"):
                    best_auc = max(m["auc"] for m in history["val_metrics"])
            print(f"  * Resumed Epoch: {start_epoch} | Best AUC: {best_auc:.4f}", flush=True)
        else:
            print("Warning: Checkpoint architecture/key mismatch. Skipping resume, training from scratch.", flush=True)

    for epoch in range(start_epoch, epochs + 1):
        start = time.time()
        train_loss  = train_one_epoch(model, train_loader, criterion, optimizer, device, use_amp, scaler)
        val_metrics = validate_epoch(model, val_loader, device, threshold=decision_threshold, num_classes=num_classes)
        elapsed = time.time() - start

        labels_arr = val_metrics.pop("all_labels")
        probs_arr = val_metrics.pop("all_probs")

        print(
            f"Epoch {epoch}/{epochs} ({elapsed:.1f}s) — "
            f"train_loss={train_loss:.4f}, "
            f"val_auc={val_metrics['auc']:.4f}, "
            f"val_sens={val_metrics['sensitivity']:.4f}, "
            f"val_spec={val_metrics['specificity']:.4f}",
            flush=True,
        )

        if len(np.unique(labels_arr)) >= 2:
            alt_metrics = []
            for t in [0.3, 0.4, 0.5]:
                m = compute_metrics(labels_arr, probs_arr, threshold=t, num_classes=num_classes)
                alt_metrics.append(f"T={t:.1f}: sens={m['sensitivity']:.3f}, spec={m['specificity']:.3f}")
            print(f"  - Threshold Sweep -> " + " | ".join(alt_metrics), flush=True)

        history["train_loss"].append(train_loss)
        history["val_metrics"].append(val_metrics)

        if val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            torch.save(model.state_dict(), output_dir / "nodule_detector_best.pt")
            print(f"  -> New best AUC: {best_auc:.4f} — saved nodule_detector_best.pt", flush=True)

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            output_dir / "nodule_detector_last.pt",
        )

        with open(output_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

    # Plots
    aucs = [m["auc"] for m in history["val_metrics"]]
    sens = [m["sensitivity"] for m in history["val_metrics"]]
    spec = [m["specificity"] for m in history["val_metrics"]]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["train_loss"], "r-o"); axes[0].set_title("Train Loss (Dice+CE)"); axes[0].grid(True)
    axes[1].plot(aucs, label=f"AUC (best {max(aucs):.4f})", marker="o")
    axes[1].plot(sens, label="Sensitivity", marker="s")
    axes[1].plot(spec, label="Specificity", marker="^")
    axes[1].set_title("Validation Metrics"); axes[1].legend(); axes[1].grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "training_curves.png", dpi=150, bbox_inches="tight")
    print(f"\nTraining complete. Best val AUC: {best_auc:.4f}", flush=True)


if __name__ == "__main__":
    main()
