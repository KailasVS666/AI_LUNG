from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .preprocess import build_volume, hu_clip_normalize


class LIDCDenoise25DDataset(Dataset):
    def __init__(
        self,
        split_entries: list[dict[str, Any]],
        hu_min: int = -1000,
        hu_max: int = 400,
        context_slices: int = 1,
        max_cases: int | None = None,
    ) -> None:
        self.hu_min = hu_min
        self.hu_max = hu_max
        self.context_slices = context_slices

        selected = split_entries[:max_cases] if max_cases is not None else split_entries

        self.samples: list[tuple[str, int]] = []
        self.volumes: dict[str, np.ndarray] = {}

        for item in selected:
            series_path = str(item["file_location"])
            volume_hu, _ = build_volume(series_path)
            volume_norm = hu_clip_normalize(volume_hu, hu_min=self.hu_min, hu_max=self.hu_max)
            self.volumes[series_path] = volume_norm

            z_count = volume_norm.shape[0]
            for z in range(self.context_slices, z_count - self.context_slices):
                self.samples.append((series_path, z))

        if not self.samples:
            raise RuntimeError("No training samples created. Check split entries and context_slices.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        series_path, z = self.samples[index]
        vol = self.volumes[series_path]

        start = z - self.context_slices
        end = z + self.context_slices + 1
        x = vol[start:end]
        y = vol[z]

        x_t = torch.from_numpy(x).float()
        y_t = torch.from_numpy(y).float().unsqueeze(0)

        return {
            "x": x_t,
            "y": y_t,
        }
