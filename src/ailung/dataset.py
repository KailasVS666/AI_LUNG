from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class LIDCSeries:
    subject_id: str
    series_uid: str
    file_location: Path
    modality: str
    number_of_images: int


def discover_ct_series(dataset_root: str | Path, metadata_csv: str | Path) -> list[LIDCSeries]:
    root = Path(dataset_root)
    metadata_path = Path(metadata_csv)

    metadata = pd.read_csv(metadata_path)
    ct_rows = metadata[metadata["Modality"] == "CT"].copy()

    result: list[LIDCSeries] = []
    for _, row in ct_rows.iterrows():
        rel = str(row["File Location"])
        # Normalise Windows-style separators and leading .\ or ./
        rel = rel.replace("\\", "/").replace("./", "").lstrip("/")
        abs_path = (root / rel).resolve()
        if not abs_path.exists():
            continue

        result.append(
            LIDCSeries(
                subject_id=str(row["Subject ID"]),
                series_uid=str(row["Series UID"]),
                file_location=abs_path,
                modality=str(row["Modality"]),
                number_of_images=int(row["Number of Images"]),
            )
        )

    return result


def sample_series(series_list: Iterable[LIDCSeries], n: int) -> list[LIDCSeries]:
    sampled: list[LIDCSeries] = []
    for index, item in enumerate(series_list):
        if index >= n:
            break
        sampled.append(item)
    return sampled
