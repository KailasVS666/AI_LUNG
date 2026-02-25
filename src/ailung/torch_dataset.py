from __future__ import annotations

from pathlib import Path
from typing import Any
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from .preprocess import build_volume, hu_clip_normalize
from .annotations import build_nodule_candidates, build_series_to_xml_mapping


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


class NoduleDetectionDataset(Dataset):
    """
    Balanced dataset for nodule detection: positive patches centered on XML nodules
    (malignancy >= min_malignancy) and negative patches from random background regions.
    """
    def __init__(
        self,
        split_entries: list[dict[str, Any]],
        xml_dir: str | Path,
        hu_min: int = -1000,
        hu_max: int = 400,
        patch_size: tuple[int, int, int] = (32, 64, 64),
        min_malignancy: int = 3,
        negatives_per_positive: int = 2,
        max_cases: int | None = None,
        seed: int = 42,
    ) -> None:
        """
        Args:
            split_entries: List of series dictionaries from splits.json
            xml_dir: Directory containing LIDC XML annotation files
            hu_min: HU window minimum
            hu_max: HU window maximum
            patch_size: (z, y, x) patch dimensions in voxels
            min_malignancy: Minimum malignancy score (1-5) for positive samples
            negatives_per_positive: Number of negative samples to generate per positive
            max_cases: Maximum number of cases to load (None = all)
            seed: Random seed for reproducibility
        """
        self.hu_min = hu_min
        self.hu_max = hu_max
        self.patch_size = patch_size
        self.negatives_per_positive = negatives_per_positive
        self.xml_dir = Path(xml_dir)
        self.rng = random.Random(seed)
        
        # Build series UID -> XML path mapping once
        print("Building series UID -> XML mapping...")
        self.series_to_xml = build_series_to_xml_mapping(xml_dir)
        print(f"Found {len(self.series_to_xml)} XML files with series UIDs")
        
        selected = split_entries[:max_cases] if max_cases is not None else split_entries
        
        self.samples: list[dict] = []  # Each: {"series_path": ..., "center_mm": ..., "label": 0 or 1}
        self.volumes: dict[str, np.ndarray] = {}
        self.spacings: dict[str, tuple[float, float, float]] = {}
        
        pz, py, px = self.patch_size
        
        for item in selected:
            series_path = str(item["file_location"])
            series_uid = item["series_uid"]
            
            # Load volume
            volume_hu, spacing = build_volume(series_path)
            volume_norm = hu_clip_normalize(volume_hu, hu_min=self.hu_min, hu_max=self.hu_max)
            
            if volume_norm.shape[0] < pz or volume_norm.shape[1] < py or volume_norm.shape[2] < px:
                continue
            
            self.volumes[series_path] = volume_norm
            self.spacings[series_path] = spacing
            
            # Find corresponding XML file using the mapping
            xml_path = self.series_to_xml.get(series_uid)
            if xml_path is None:
                # No annotations, only sample negatives
                num_negatives = self.negatives_per_positive * 3  # arbitrary baseline
                self._add_negative_samples(series_path, num_negatives)
                continue
            
            nodule_candidates = build_nodule_candidates(xml_path, spacing, min_malignancy=min_malignancy)
            
            # Add positive samples (centered on nodules)
            for candidate in nodule_candidates:
                self.samples.append({
                    "series_path": series_path,
                    "center_mm": candidate["centroid_3d"],  # (z_mm, y_mm, x_mm)
                    "label": 1,
                })
            
            # Add negative samples (random background positions)
            num_positives = len(nodule_candidates)
            num_negatives = max(num_positives * self.negatives_per_positive, 1)
            self._add_negative_samples(series_path, num_negatives)
        
        if not self.samples:
            raise RuntimeError("No detection samples created. Check XML paths and min_malignancy.")
    
    def _add_negative_samples(self, series_path: str, count: int) -> None:
        """Add random negative patches that avoid nodule regions (simplified: fully random)."""
        vol = self.volumes[series_path]
        spacing = self.spacings[series_path]
        
        pz, py, px = self.patch_size
        z_max_voxel = vol.shape[0] - pz
        y_max_voxel = vol.shape[1] - py
        x_max_voxel = vol.shape[2] - px
        
        for _ in range(count):
            sz = self.rng.randint(0, z_max_voxel)
            sy = self.rng.randint(0, y_max_voxel)
            sx = self.rng.randint(0, x_max_voxel)
            
            # Convert to physical coordinates (center of patch)
            z_mm = (sz + pz / 2) * spacing[0]
            y_mm = (sy + py / 2) * spacing[1]
            x_mm = (sx + px / 2) * spacing[2]
            
            self.samples.append({
                "series_path": series_path,
                "center_mm": (z_mm, y_mm, x_mm),
                "label": 0,
            })
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.samples[index]
        series_path = sample["series_path"]
        center_mm = sample["center_mm"]
        label = sample["label"]
        
        vol = self.volumes[series_path]
        spacing = self.spacings[series_path]
        
        # Convert center from mm to voxel index
        z_voxel = int(center_mm[0] / spacing[0])
        y_voxel = int(center_mm[1] / spacing[1])
        x_voxel = int(center_mm[2] / spacing[2])
        
        pz, py, px = self.patch_size
        
        # Extract patch centered on voxel
        sz = max(0, min(z_voxel - pz // 2, vol.shape[0] - pz))
        sy = max(0, min(y_voxel - py // 2, vol.shape[1] - py))
        sx = max(0, min(x_voxel - px // 2, vol.shape[2] - px))
        
        patch = vol[sz : sz + pz, sy : sy + py, sx : sx + px].astype(np.float32)
        
        # Handle edge cases where patch might be smaller than requested
        if patch.shape != (pz, py, px):
            padded = np.zeros((pz, py, px), dtype=np.float32)
            padded[: patch.shape[0], : patch.shape[1], : patch.shape[2]] = patch
            patch = padded
        
        x_t = torch.from_numpy(patch).unsqueeze(0)  # (1, pz, py, px)
        y_t = torch.tensor(label, dtype=torch.long)
        
        return {
            "x": x_t,
            "y": y_t,
        }

