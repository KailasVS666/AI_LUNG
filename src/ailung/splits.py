from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
import json
import random

from .dataset import LIDCSeries, discover_ct_series


def _group_by_subject(series_list: list[LIDCSeries]) -> dict[str, list[LIDCSeries]]:
    grouped: dict[str, list[LIDCSeries]] = defaultdict(list)
    for item in series_list:
        grouped[item.subject_id].append(item)
    return grouped


def build_patient_split(
    dataset_root: str | Path,
    metadata_csv: str | Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> dict:
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-8:
        raise ValueError("train/val/test ratios must sum to 1.0")

    series_list = discover_ct_series(dataset_root, metadata_csv)
    grouped = _group_by_subject(series_list)

    subject_ids = sorted(grouped.keys())
    rng = random.Random(seed)
    rng.shuffle(subject_ids)

    n = len(subject_ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_subjects = set(subject_ids[:n_train])
    val_subjects = set(subject_ids[n_train : n_train + n_val])
    test_subjects = set(subject_ids[n_train + n_val :])

    split = {"train": [], "val": [], "test": []}

    for sid, items in grouped.items():
        bucket = "test"
        if sid in train_subjects:
            bucket = "train"
        elif sid in val_subjects:
            bucket = "val"

        split[bucket].extend(
            [
                {
                    "subject_id": item.subject_id,
                    "series_uid": item.series_uid,
                    "file_location": str(item.file_location),
                    "modality": item.modality,
                    "number_of_images": item.number_of_images,
                }
                for item in items
            ]
        )

    split["meta"] = {
        "seed": seed,
        "subjects_total": n,
        "series_total": len(series_list),
        "subjects_train": len(train_subjects),
        "subjects_val": len(val_subjects),
        "subjects_test": len(test_subjects),
        "series_train": len(split["train"]),
        "series_val": len(split["val"]),
        "series_test": len(split["test"]),
    }
    return split


def save_split(split: dict, output_path: str | Path) -> Path:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(split, f, indent=2)
    return out


def load_split(path: str | Path) -> dict:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)
