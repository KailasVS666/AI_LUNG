"""
One-time preprocessing: Convert all DICOM series to float16 .npy
=================================================================
Run this ONCE before training. After it finishes, training will be
10-15x faster because .npy loads ~2s vs DICOM ~30s from Drive.

Usage (in Colab):
    %cd /content/AI_LUNG
    !python scripts/preprocess_to_npy.py --config configs/baseline_colab.yaml

Output: AI_LUNG_DATA/outputs/preprocessed_npy/<series_uid>.npy
  - float16, shape (Z, H, W), values in [0, 1]
  - HU clipping + normalization + CLAHE already applied
  - ~30-80 MB per file, ~55 GB total for all 1019 series

Time: ~1-2 hours on Colab (CPU-bound DICOM reading + CLAHE)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import yaml
from tqdm import tqdm

from ailung.preprocess import build_volume, hu_clip_normalize, apply_clahe
from ailung.splits import load_split


def get_series_uid(item: dict) -> str:
    """Derive a unique filename from the series path."""
    uid = item.get("series_uid", "")
    if uid:
        return uid
    # Fallback: use last two path parts (patient_id/series_id)
    parts = Path(str(item["file_location"])).parts
    return "_".join(parts[-2:]).replace(" ", "_")


def preprocess_series(
    item: dict,
    out_dir: Path,
    hu_min: int,
    hu_max: int,
    apply_clahe_flag: bool,
) -> tuple[str, str]:
    """
    Process one CT series and save as float16 .npy.
    Returns (series_uid, status) where status is 'done', 'skip', or 'error: ...'
    """
    uid      = get_series_uid(item)
    out_path = out_dir / f"{uid}.npy"

    if out_path.exists():
        return uid, "skip"

    try:
        volume_hu, _ = build_volume(str(item["file_location"]))
        volume_nd    = hu_clip_normalize(volume_hu, hu_min=hu_min, hu_max=hu_max)
        if apply_clahe_flag:
            volume_nd = apply_clahe(volume_nd)
        np.save(str(out_path), volume_nd.astype(np.float16))
        return uid, "done"
    except Exception as e:
        return uid, f"error: {e}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-convert DICOM series to .npy")
    parser.add_argument("--config", type=str, default="configs/baseline_colab.yaml")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"],
                        help="Which splits to process")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    hu_min           = cfg["data"]["hu_min"]
    hu_max           = cfg["data"]["hu_max"]
    apply_clahe_flag = cfg["data"].get("apply_clahe", True)
    splits_path      = cfg["data"]["splits_path"]

    # Output directory
    base_out = Path(cfg["data"].get(
        "preprocessed_npy_dir",
        str(Path(cfg["train"]["output_dir"]).parent / "preprocessed_npy")
    ))
    base_out.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {base_out}", flush=True)

    split = load_split(splits_path)

    all_items = []
    for s in args.splits:
        entries = split.get(s, [])
        all_items.extend(entries)

    # De-duplicate (some series appear in multiple splits? unlikely but safe)
    seen_paths = set()
    unique_items = []
    for item in all_items:
        p = str(item["file_location"])
        if p not in seen_paths:
            seen_paths.add(p)
            unique_items.append(item)

    print(f"\nTotal unique series to process: {len(unique_items)}", flush=True)
    print(f"Estimated time: {len(unique_items) * 8 / 60:.0f}–{len(unique_items) * 15 / 60:.0f} minutes\n", flush=True)

    done = skipped = errors = 0
    error_log = []
    t0 = time.time()

    bar = tqdm(unique_items, unit="series", dynamic_ncols=True)
    for item in bar:
        uid, status = preprocess_series(item, base_out, hu_min, hu_max, apply_clahe_flag)

        if status == "done":
            done += 1
        elif status == "skip":
            skipped += 1
        else:
            errors += 1
            error_log.append(f"{uid}: {status}")

        elapsed = time.time() - t0
        rate    = (done + skipped) / max(elapsed, 1)
        remaining = (len(unique_items) - done - skipped - errors) / max(rate, 1e-6)

        bar.set_postfix(
            done=done, skip=skipped, err=errors,
            eta=f"{remaining/60:.1f}m"
        )

    print(f"\n{'='*55}", flush=True)
    print(f"  Done:    {done}", flush=True)
    print(f"  Skipped: {skipped} (already existed)", flush=True)
    print(f"  Errors:  {errors}", flush=True)
    print(f"  Total time: {(time.time()-t0)/60:.1f} min", flush=True)
    print(f"  Saved to: {base_out}", flush=True)
    if error_log:
        print(f"\nFailed series:", flush=True)
        for e in error_log:
            print(f"  {e}", flush=True)

    # Save a mapping file so the dataset can find .npy files by series path
    mapping = {}
    for item in unique_items:
        uid = get_series_uid(item)
        npy_path = base_out / f"{uid}.npy"
        if npy_path.exists():
            mapping[str(item["file_location"])] = str(npy_path)

    mapping_path = base_out / "path_to_npy.json"
    with open(mapping_path, "w") as f:
        json.dump(mapping, f, indent=2)
    print(f"\nMapping saved: {mapping_path}", flush=True)
    print(f"Total .npy files: {len(mapping)}", flush=True)


if __name__ == "__main__":
    main()
