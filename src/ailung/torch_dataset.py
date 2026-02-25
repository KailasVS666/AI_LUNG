from __future__ import annotations

from pathlib import Path
from typing import Any
import random

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


class LIDCRecon3DPatchDataset(Dataset):
    def __init__(
        self,
        split_entries: list[dict[str, Any]],
        hu_min: int = -1000,
        hu_max: int = 400,
        patch_size: tuple[int, int, int] = (32, 96, 96),
        patches_per_volume: int = 16,
        noise_std: float = 0.03,
        max_cases: int | None = None,
        seed: int = 42,
    ) -> None:
        self.hu_min = hu_min
        self.hu_max = hu_max
        self.patch_size = patch_size
        self.patches_per_volume = patches_per_volume
        self.noise_std = noise_std
        self.rng = random.Random(seed)

        selected = split_entries[:max_cases] if max_cases is not None else split_entries

        self.volumes: dict[str, np.ndarray] = {}
        self.samples: list[tuple[str, int, int, int]] = []

        pz, py, px = self.patch_size

        for item in selected:
            series_path = str(item["file_location"])
            volume_hu, _ = build_volume(series_path)
            volume_norm = hu_clip_normalize(volume_hu, hu_min=self.hu_min, hu_max=self.hu_max)

            if volume_norm.shape[0] < pz or volume_norm.shape[1] < py or volume_norm.shape[2] < px:
                continue

            self.volumes[series_path] = volume_norm

            z_max = volume_norm.shape[0] - pz
            y_max = volume_norm.shape[1] - py
            x_max = volume_norm.shape[2] - px

            for _ in range(self.patches_per_volume):
                sz = self.rng.randint(0, z_max)
                sy = self.rng.randint(0, y_max)
                sx = self.rng.randint(0, x_max)
                self.samples.append((series_path, sz, sy, sx))

        if not self.samples:
            raise RuntimeError("No 3D patch samples created. Reduce patch size or increase max_cases.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        series_path, sz, sy, sx = self.samples[index]
        vol = self.volumes[series_path]

        pz, py, px = self.patch_size
        target = vol[sz : sz + pz, sy : sy + py, sx : sx + px].astype(np.float32)

        noise = np.random.normal(loc=0.0, scale=self.noise_std, size=target.shape).astype(np.float32)
        source = np.clip(target + noise, 0.0, 1.0)

        x_t = torch.from_numpy(source).unsqueeze(0)
        y_t = torch.from_numpy(target).unsqueeze(0)

        return {
            "x": x_t,
            "y": y_t,
        }
