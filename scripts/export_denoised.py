"""
Stage 1 → Stage 2 Bridge: Export Denoised Volumes
====================================================
After Stage 1 training is complete, run this script to:
  1. Load the best Stage 1 checkpoint (denoiser_best.pt)
  2. Run all training+validation series through the denoiser
  3. Save each denoised volume as <series_uid>_denoised.npy to Drive

The .npy files are then loaded by LIDCRecon3DPatchDataset (Stage 2).

Usage:
    python scripts/export_denoised.py --config configs/baseline_colab.yaml

Output (in config denoised_out_dir or output_dir/denoised_vols/):
    <series_uid>_denoised.npy  — float32 (Z, H, W) in [0, 1]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml

from ailung.models import Denoise25DUNet
from ailung.preprocess import build_volume, hu_clip_normalize, apply_clahe, simulate_low_dose_volume
from ailung.splits import load_split


def _resolve_device(device_cfg: str) -> torch.device:
    if device_cfg == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def export_series(
    model: torch.nn.Module,
    device: torch.device,
    item: dict,
    cfg: dict,
    out_dir: Path,
) -> None:
    series_path = str(item["file_location"])
    series_uid  = item.get("series_uid", "unknown")

    out_path = out_dir / f"{series_uid}_denoised.npy"
    if out_path.exists():
        print(f"  [SKIP] {series_uid} — already exported", flush=True)
        return

    # Preprocessing (same as training)
    volume_hu, _ = build_volume(series_path)
    volume_nd = hu_clip_normalize(
        volume_hu,
        hu_min=cfg["data"]["hu_min"],
        hu_max=cfg["data"]["hu_max"],
    )
    if cfg["data"].get("apply_clahe", True):
        volume_nd = apply_clahe(volume_nd)

    low_dose_i0 = float(cfg["data"].get("low_dose_i0", 1e5))
    volume_ld = simulate_low_dose_volume(volume_nd, i0=low_dose_i0, seed=0)

    context = cfg["data"]["context_slices"]  # 4
    z_count = volume_ld.shape[0]
    denoised = np.zeros_like(volume_nd)

    model.eval()
    with torch.no_grad():
        for z in range(context, z_count - context):
            stack = volume_ld[z - context : z + context + 1]  # (9, H, W)
            x = torch.from_numpy(stack).float().unsqueeze(0).to(device)  # (1, 9, H, W)
            pred = model(x)                                                # (1, 1, H, W)
            denoised[z] = pred.squeeze().cpu().numpy()

    # Fill boundary slices with the LD values (they won't be used as patch centres)
    denoised[:context] = volume_ld[:context]
    denoised[-context:] = volume_ld[-context:]

    np.save(str(out_path), denoised)
    print(f"  [DONE] {series_uid} → {out_path.name}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Stage 1 denoised volumes for Stage 2")
    parser.add_argument("--config", type=str, default="configs/baseline.yaml",
                        help="Stage 1 config YAML file")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = _resolve_device(cfg["runtime"]["device"])

    # --- Load Stage 1 model ---
    ckpt_path = Path(cfg["train"]["output_dir"]) / "denoiser_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Stage 1 checkpoint not found: {ckpt_path}\n"
            "Run train_denoiser_baseline.py first."
        )

    in_channels = cfg["data"]["context_slices"] * 2 + 1
    model = Denoise25DUNet(in_channels=in_channels, out_channels=1)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    print(f"Loaded Stage 1 checkpoint from {ckpt_path}", flush=True)

    # --- Output directory ---
    out_dir = Path(
        cfg.get("denoised_out_dir",
                str(Path(cfg["train"]["output_dir"]).parent / "denoised_vols"))
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Exporting denoised volumes to: {out_dir}", flush=True)

    # --- Export all train + val series ---
    split = load_split(cfg["data"]["splits_path"])

    for split_name in ("train", "val", "test"):
        entries = split.get(split_name, [])
        max_cases = cfg["data"]["max_cases_per_split"].get(split_name)
        if max_cases is not None:
            entries = entries[:max_cases]
        print(f"\n=== Exporting {split_name} ({len(entries)} series) ===", flush=True)
        for item in entries:
            try:
                export_series(model, device, item, cfg, out_dir)
            except Exception as exc:
                print(f"  [ERROR] {item.get('series_uid','?')}: {exc}", flush=True)

    print("\nExport complete.", flush=True)


if __name__ == "__main__":
    main()
