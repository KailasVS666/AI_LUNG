"""
AI-LUNG PyTorch Datasets
=========================

Stage 1 — LIDCDenoise25DDataset:
    Input:  9 consecutive LOW-DOSE simulated CT slices  (Radon+Poisson+FBP)
    Target: 1 clean NORMAL-DOSE central slice
    Returns: {"ld": Tensor(9,H,W), "nd": Tensor(1,H,W)}

Stage 2 — LIDCRecon3DPatchDataset:
    Input:  (1, 64, 64, 64) patch of DENOISED slices  (output of Stage 1)
    Target: (1, 64, 64, 64) patch of NORMAL-DOSE volume
    Returns: {"x": Tensor(1,D,H,W), "y": Tensor(1,D,H,W)}

Stage 3 — NoduleDetectionDataset:
    Input:  (1, pz, py, px) patch from RECONSTRUCTED 3D volume
    Target: integer label (0 = benign / background, 1 = malignant)
    Returns: {"x": Tensor(1,pz,py,px), "y": LongTensor()}
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from .preprocess import (
    build_volume,
    hu_clip_normalize,
    apply_clahe,
    simulate_low_dose_volume,
)
from .annotations import build_nodule_candidates, build_series_to_xml_mapping


# ---------------------------------------------------------------------------
# Stage 1 Dataset
# ---------------------------------------------------------------------------

class LIDCDenoise25DDataset(Dataset):
    """
    2.5D Denoising Dataset.

    For each CT series it builds:
      - `volume_nd` : normal-dose volume (ground truth) via CLAHE-enhanced HU
      - `volume_ld` : simulated low-dose version (Radon + Poisson + FBP)

    Each sample is a windows of `context_slices*2+1` consecutive slices.
    Default context_slices=4 → 9-slice input.
    """

    def __init__(
        self,
        split_entries: list[dict[str, Any]],
        hu_min: int = -1000,
        hu_max: int = 400,
        context_slices: int = 4,        # 9-slice input (4 + center + 4)
        max_cases: int | None = None,
        apply_clahe_flag: bool = True,
        low_dose_i0: float = 1e5,
        seed: int = 42,
    ) -> None:
        self.hu_min = hu_min
        self.hu_max = hu_max
        self.context_slices = context_slices

        selected = split_entries[:max_cases] if max_cases is not None else split_entries

        self.samples: list[tuple[str, int]] = []
        self.volumes_nd: dict[str, np.ndarray] = {}   # normal-dose (clean)
        self.volumes_ld: dict[str, np.ndarray] = {}   # simulated low-dose

        for idx, item in enumerate(selected):
            series_path = str(item["file_location"])
            volume_hu, _ = build_volume(series_path)
            volume_nd = hu_clip_normalize(volume_hu, hu_min=self.hu_min, hu_max=self.hu_max)

            if apply_clahe_flag:
                volume_nd = apply_clahe(volume_nd)

            # Per-series low-dose simulation (deterministic via seed+idx)
            volume_ld = simulate_low_dose_volume(volume_nd, i0=low_dose_i0, seed=seed + idx * 1000)

            self.volumes_nd[series_path] = volume_nd
            self.volumes_ld[series_path] = volume_ld

            z_count = volume_nd.shape[0]
            for z in range(self.context_slices, z_count - self.context_slices):
                self.samples.append((series_path, z))

        if not self.samples:
            raise RuntimeError(
                "No training samples created. Check split entries and context_slices."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        series_path, z = self.samples[index]
        vol_nd = self.volumes_nd[series_path]
        vol_ld = self.volumes_ld[series_path]

        start = z - self.context_slices
        end   = z + self.context_slices + 1

        # 9-slice low-dose input, 1-slice normal-dose target
        ld_stack = vol_ld[start:end]   # (9, H, W)
        nd_slice  = vol_nd[z]           # (H, W)

        return {
            "ld": torch.from_numpy(ld_stack).float(),          # (9, H, W)
            "nd": torch.from_numpy(nd_slice).float().unsqueeze(0),  # (1, H, W)
            # Legacy key aliases so old training scripts still work
            "x":  torch.from_numpy(ld_stack).float(),
            "y":  torch.from_numpy(nd_slice).float().unsqueeze(0),
        }


# ---------------------------------------------------------------------------
# Stage 2 Dataset
# ---------------------------------------------------------------------------

class LIDCRecon3DPatchDataset(Dataset):
    """
    3D Reconstruction Dataset.

    Loads DENOISED volumes (either from pre-saved numpy files produced by
    `scripts/export_denoised.py`, or falls back to the raw normal-dose volume
    with Gaussian noise if denoised_vol_dir is None).

    Each sample is a random 3D patch of size `patch_size` (default 64×64×64).

    Args:
        denoised_vol_dir : path to folder of <series_uid>_denoised.npy files
                           produced by the Stage 1 export step.
                           If None, uses noisy simulation instead.
    """

    def __init__(
        self,
        split_entries: list[dict[str, Any]],
        hu_min: int = -1000,
        hu_max: int = 400,
        patch_size: tuple[int, int, int] = (64, 64, 64),
        patches_per_volume: int = 16,
        max_cases: int | None = None,
        seed: int = 42,
        denoised_vol_dir: str | Path | None = None,
        noise_std: float = 0.03,   # fallback noise when denoised_vol_dir=None
    ) -> None:
        self.hu_min             = hu_min
        self.hu_max             = hu_max
        self.patch_size         = patch_size
        self.patches_per_volume = patches_per_volume
        self.noise_std          = noise_std
        self.rng                = random.Random(seed)
        self.denoised_vol_dir   = Path(denoised_vol_dir) if denoised_vol_dir else None

        selected = split_entries[:max_cases] if max_cases is not None else split_entries

        self.volumes_nd: dict[str, np.ndarray] = {}   # normal-dose (ground truth)
        self.volumes_in: dict[str, np.ndarray] = {}   # denoised input (or noisy fallback)
        self.samples: list[tuple[str, int, int, int]] = []

        pz, py, px = self.patch_size

        for item in selected:
            series_path = str(item["file_location"])
            series_uid  = item.get("series_uid", "")

            volume_hu, _ = build_volume(series_path)
            volume_nd    = hu_clip_normalize(volume_hu, hu_min=self.hu_min, hu_max=self.hu_max)

            if volume_nd.shape[0] < pz or volume_nd.shape[1] < py or volume_nd.shape[2] < px:
                continue

            self.volumes_nd[series_path] = volume_nd

            # Try to load pre-exported denoised volume
            if self.denoised_vol_dir is not None:
                den_path = self.denoised_vol_dir / f"{series_uid}_denoised.npy"
                if den_path.exists():
                    volume_in = np.load(str(den_path))
                else:
                    # Fallback: noisy simulation
                    noise = np.random.normal(0.0, self.noise_std, volume_nd.shape).astype(np.float32)
                    volume_in = np.clip(volume_nd + noise, 0.0, 1.0)
            else:
                # No denoised dir provided — use Gaussian noisy volume
                noise = np.random.normal(0.0, self.noise_std, volume_nd.shape).astype(np.float32)
                volume_in = np.clip(volume_nd + noise, 0.0, 1.0)

            self.volumes_in[series_path] = volume_in

            z_max = volume_nd.shape[0] - pz
            y_max = volume_nd.shape[1] - py
            x_max = volume_nd.shape[2] - px

            for _ in range(self.patches_per_volume):
                sz = self.rng.randint(0, z_max)
                sy = self.rng.randint(0, y_max)
                sx = self.rng.randint(0, x_max)
                self.samples.append((series_path, sz, sy, sx))

        if not self.samples:
            raise RuntimeError(
                "No 3D patch samples created. Reduce patch_size or increase max_cases."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        series_path, sz, sy, sx = self.samples[index]
        pz, py, px = self.patch_size

        target = self.volumes_nd[series_path][sz:sz+pz, sy:sy+py, sx:sx+px].astype(np.float32)
        source = self.volumes_in[series_path][sz:sz+pz, sy:sy+py, sx:sx+px].astype(np.float32)

        return {
            "x": torch.from_numpy(source).unsqueeze(0),   # (1, D, H, W)  denoised input
            "y": torch.from_numpy(target).unsqueeze(0),   # (1, D, H, W)  normal-dose target
        }


# ---------------------------------------------------------------------------
# Stage 3 Dataset
# ---------------------------------------------------------------------------

class NoduleDetectionDataset(Dataset):
    """
    Balanced nodule detection dataset.

    Positive patches: centered on annotated LIDC nodules (malignancy >= min_malignancy)
    Negative patches: random background regions

    Patches are extracted from the RECONSTRUCTED 3D volume (output of Stage 2).
    Falls back to normal-dose volume if reconstructed_vol_dir is None.
    """

    def __init__(
        self,
        split_entries: list[dict[str, Any]],
        xml_dir: str | Path,
        hu_min: int = -1000,
        hu_max: int = 400,
        patch_size: tuple[int, int, int] = (32, 64, 64),
        min_malignancy: int = 1,
        negatives_per_positive: int = 2,
        max_cases: int | None = None,
        seed: int = 42,
        reconstructed_vol_dir: str | Path | None = None,
    ) -> None:
        self.hu_min = hu_min
        self.hu_max = hu_max
        self.patch_size = patch_size
        self.negatives_per_positive = negatives_per_positive
        self.xml_dir = Path(xml_dir)
        self.rng = random.Random(seed)
        self.reconstructed_vol_dir = (
            Path(reconstructed_vol_dir) if reconstructed_vol_dir else None
        )

        print("Building series UID → XML mapping...")
        self.series_to_xml = build_series_to_xml_mapping(xml_dir)
        print(f"Found {len(self.series_to_xml)} XML files with series UIDs")

        selected = split_entries[:max_cases] if max_cases is not None else split_entries

        self.samples: list[dict] = []
        self.volumes: dict[str, np.ndarray] = {}
        self.spacings: dict[str, tuple[float, float, float]] = {}

        pz, py, px = self.patch_size

        for item in selected:
            series_path = str(item["file_location"])
            series_uid  = item["series_uid"]

            volume_hu, spacing = build_volume(series_path)
            volume_nd = hu_clip_normalize(volume_hu, hu_min=self.hu_min, hu_max=self.hu_max)

            if volume_nd.shape[0] < pz or volume_nd.shape[1] < py or volume_nd.shape[2] < px:
                continue

            # Try to load reconstructed 3D volume; fall back to normal-dose
            if self.reconstructed_vol_dir is not None:
                rec_path = self.reconstructed_vol_dir / f"{series_uid}_recon3d.npy"
                vol = np.load(str(rec_path)) if rec_path.exists() else volume_nd
            else:
                vol = volume_nd

            self.volumes[series_path]  = vol
            self.spacings[series_path] = spacing

            xml_path = self.series_to_xml.get(series_uid)
            if xml_path is None:
                self._add_negative_samples(series_path, self.negatives_per_positive * 3)
                continue

            nodule_candidates = build_nodule_candidates(
                xml_path, spacing, min_malignancy=min_malignancy
            )

            for candidate in nodule_candidates:
                self.samples.append({
                    "series_path": series_path,
                    "center_mm":   candidate["centroid_3d"],
                    "label": 1,
                })

            num_negatives = max(len(nodule_candidates) * self.negatives_per_positive, 1)
            self._add_negative_samples(series_path, num_negatives)

        if not self.samples:
            raise RuntimeError(
                "No detection samples created. Check XML paths and min_malignancy."
            )

    def _add_negative_samples(self, series_path: str, count: int) -> None:
        vol     = self.volumes[series_path]
        spacing = self.spacings[series_path]
        pz, py, px = self.patch_size

        z_max = vol.shape[0] - pz
        y_max = vol.shape[1] - py
        x_max = vol.shape[2] - px

        for _ in range(count):
            sz = self.rng.randint(0, z_max)
            sy = self.rng.randint(0, y_max)
            sx = self.rng.randint(0, x_max)

            z_mm = (sz + pz / 2) * spacing[0]
            y_mm = (sy + py / 2) * spacing[1]
            x_mm = (sx + px / 2) * spacing[2]

            self.samples.append({
                "series_path": series_path,
                "center_mm":   (z_mm, y_mm, x_mm),
                "label": 0,
            })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample      = self.samples[index]
        series_path = sample["series_path"]
        center_mm   = sample["center_mm"]
        label       = sample["label"]

        vol     = self.volumes[series_path]
        spacing = self.spacings[series_path]
        pz, py, px = self.patch_size

        z_voxel = int(center_mm[0] / spacing[0])
        y_voxel = int(center_mm[1] / spacing[1])
        x_voxel = int(center_mm[2] / spacing[2])

        sz = max(0, min(z_voxel - pz // 2, vol.shape[0] - pz))
        sy = max(0, min(y_voxel - py // 2, vol.shape[1] - py))
        sx = max(0, min(x_voxel - px // 2, vol.shape[2] - px))

        patch = vol[sz:sz+pz, sy:sy+py, sx:sx+px].astype(np.float32)

        if patch.shape != (pz, py, px):
            padded = np.zeros((pz, py, px), dtype=np.float32)
            padded[:patch.shape[0], :patch.shape[1], :patch.shape[2]] = patch
            patch = padded

        return {
            "x": torch.from_numpy(patch).unsqueeze(0),   # (1, pz, py, px)
            "y": torch.tensor(label, dtype=torch.long),
        }
