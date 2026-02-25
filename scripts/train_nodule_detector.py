#!/usr/bin/env python
"""
Training script for nodule detection classifier.
Evaluates with AUC, sensitivity, and specificity.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
import json

import yaml
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

from ailung.dataset import discover_ct_series
from ailung.splits import load_split
from ailung.torch_dataset import NoduleDetectionDataset
from ailung.models import NoduleClassifier3D


def compute_metrics(y_true: np.ndarray, y_pred_proba: np.ndarray, threshold: float = 0.5) -> dict:
    """Compute detection metrics: AUC, sensitivity, specificity."""
    auc = roc_auc_score(y_true, y_pred_proba)
    
    y_pred = (y_pred_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    return {
        "auc": float(auc),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def validate_epoch(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    """Run validation and compute detection metrics."""
    model.eval()
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[:, 1]  # Probability of class 1 (nodule)
            
            all_labels.extend(y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    metrics = compute_metrics(all_labels, all_probs)
    return metrics


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Train for one epoch and return average loss."""
    model.train()
    total_loss = 0.0
    count = 0
    
    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        count += 1
    
    return total_loss / count if count > 0 else 0.0


def save_checkpoint(model: nn.Module, path: Path) -> None:
    """Save model state dict."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Saved checkpoint: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train nodule detection classifier")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Paths
    dataset_root = config["dataset_root"]
    metadata_csv = config["metadata_csv"]
    xml_dir = Path(config["xml_dir"])
    split_json = config["split_json"]
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Hyperparameters
    batch_size = config.get("batch_size", 4)
    epochs = config.get("epochs", 5)
    lr = config.get("learning_rate", 1e-4)
    patch_size = tuple(config.get("patch_size", [32, 64, 64]))
    min_malignancy = config.get("min_malignancy", 3)
    negatives_per_positive = config.get("negatives_per_positive", 2)
    max_cases_train = config.get("max_cases_train", None)
    max_cases_val = config.get("max_cases_val", None)
    hu_min = config.get("hu_min", -1000)
    hu_max = config.get("hu_max", 400)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load splits
    all_series = discover_ct_series(dataset_root, metadata_csv)
    splits = load_split(split_json)
    
    train_entries = splits["train"]
    val_entries = splits["val"]
    
    print(f"Train series: {len(train_entries)}, Val series: {len(val_entries)}")

    # Create datasets
    print("Building train dataset...")
    train_ds = NoduleDetectionDataset(
        train_entries,
        xml_dir=xml_dir,
        hu_min=hu_min,
        hu_max=hu_max,
        patch_size=patch_size,
        min_malignancy=min_malignancy,
        negatives_per_positive=negatives_per_positive,
        max_cases=max_cases_train,
        seed=42,
    )
    train_positives = sum(1 for s in train_ds.samples if s["label"] == 1)
    train_negatives = len(train_ds) - train_positives
    print(f"Train samples: {len(train_ds)} (pos={train_positives}, neg={train_negatives})")
    
    print("Building val dataset...")
    val_ds = NoduleDetectionDataset(
        val_entries,
        xml_dir=xml_dir,
        hu_min=hu_min,
        hu_max=hu_max,
        patch_size=patch_size,
        min_malignancy=min_malignancy,
        negatives_per_positive=negatives_per_positive,
        max_cases=max_cases_val,
        seed=123,
    )
    val_positives = sum(1 for s in val_ds.samples if s["label"] == 1)
    val_negatives = len(val_ds) - val_positives
    print(f"Val samples: {len(val_ds)} (pos={val_positives}, neg={val_negatives})")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # Model
    model = NoduleClassifier3D(in_channels=1, base_channels=16, num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Training loop
    history = {"train_loss": [], "val_metrics": []}
    best_auc = 0.0

    for epoch in range(1, epochs + 1):
        start = time.time()
        
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = validate_epoch(model, val_loader, device)
        
        elapsed = time.time() - start
        
        print(
            f"Epoch {epoch}/{epochs} ({elapsed:.1f}s) - "
            f"train_loss={train_loss:.4f}, "
            f"val_auc={val_metrics['auc']:.4f}, "
            f"val_sens={val_metrics['sensitivity']:.4f}, "
            f"val_spec={val_metrics['specificity']:.4f}"
        )
        
        history["train_loss"].append(train_loss)
        history["val_metrics"].append(val_metrics)
        
        # Save best model
        if val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            save_checkpoint(model, output_dir / "nodule_detector_best.pt")
    
    # Save final checkpoint
    save_checkpoint(model, output_dir / "nodule_detector_last.pt")
    
    # Save history
    history_path = output_dir / "history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Saved history: {history_path}")
    
    # Plot metrics
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss curve
    axes[0].plot(history["train_loss"], label="Train Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss")
    axes[0].legend()
    axes[0].grid(True)
    
    # Metrics curve
    aucs = [m["auc"] for m in history["val_metrics"]]
    sens = [m["sensitivity"] for m in history["val_metrics"]]
    spec = [m["specificity"] for m in history["val_metrics"]]
    
    axes[1].plot(aucs, label="AUC", marker="o")
    axes[1].plot(sens, label="Sensitivity", marker="s")
    axes[1].plot(spec, label="Specificity", marker="^")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Metric")
    axes[1].set_title("Validation Metrics")
    axes[1].legend()
    axes[1].grid(True)
    
    plot_path = output_dir / "training_curves.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot: {plot_path}")
    
    print(f"\nTraining complete. Best val AUC: {best_auc:.4f}")


if __name__ == "__main__":
    main()
