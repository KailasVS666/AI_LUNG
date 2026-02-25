from __future__ import annotations

import argparse
from pathlib import Path

from ailung.splits import build_patient_split, save_split


def main() -> None:
    parser = argparse.ArgumentParser(description="Build patient-wise train/val/test split for LIDC CT series")
    parser.add_argument("--dataset-root", type=str, default="d:/AI_LUNG/manifest-1600709154662")
    parser.add_argument("--metadata-csv", type=str, default="d:/AI_LUNG/manifest-1600709154662/metadata.csv")
    parser.add_argument("--out", type=str, default="d:/AI_LUNG/outputs/splits/patient_split.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    args = parser.parse_args()

    split = build_patient_split(
        dataset_root=args.dataset_root,
        metadata_csv=args.metadata_csv,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    out = save_split(split, args.out)

    print("Split file:", out)
    print("Meta:")
    for k, v in split["meta"].items():
        print(f"- {k}: {v}")


if __name__ == "__main__":
    main()
