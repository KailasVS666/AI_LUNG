"""
AI-LUNG PyTorch Datasets
=========================

Stage 1 — LIDCDenoise25DDataset:
    LAZY LOADING: volumes are loaded & simulated on-demand per __getitem__,
    with an LRU cache to avoid re-loading the same series.
    Training starts in seconds — no upfront pre-computation.

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

from collections import OrderedDict
from pathlib import Path
from typing import Any
import random
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import torch
from torch.utils.data import Dataset

from .preprocess import (
    build_volume,
    hu_clip_normalize,
    apply_clahe,
    simulate_low_dose_volume,
    simulate_low_dose_fast,
)
from .annotations import build_nodule_candidates, build_series_to_xml_mapping


# ---------------------------------------------------------------------------
# LRU Volume Cache (shared across dataset instances in the same process)
# ---------------------------------------------------------------------------

class _LRUCache:
    """Simple thread-unsafe LRU cache for numpy arrays."""
    def __init__(self, maxsize: int = 8):
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self._maxsize = maxsize

    def get(self, key: str) -> Any | None:
        if key not in self._cache:
            return None
        self._cache.move_to_end(key)
        return self._cache[key]

    def put(self, key: str, value: Any) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = value
        if len(self._cache) > self._maxsize:
            self._cache.popitem(last=False)


# ---------------------------------------------------------------------------
# Stage 1 Dataset — Lazy Loading
# ---------------------------------------------------------------------------

class LIDCDenoise25DDataset(Dataset):
    """
    2.5D Denoising Dataset — LAZY LOADING with LRU cache.

    __init__  : fast scan of series paths only (no DICOM loading)
    __getitem__: loads + simulates on demand, caches last `cache_size` volumes

    Each sample = 9 low-dose slices → 1 normal-dose central slice
    """

    def __init__(
        self,
        split_entries: list[dict[str, Any]],
        hu_min: int = -1000,
        hu_max: int = 400,
        context_slices: int = 4,
        max_cases: int | None = None,
        apply_clahe_flag: bool = True,
        low_dose_i0: float = 1e5,
        seed: int = 42,
        fast_mode: bool = False,
        fast_mode_noise_std: float = 0.05,
        cache_size: int = 6,
        npy_mapping: dict[str, str] | None = None,  # path→npy_file, from preprocess_to_npy.py
    ) -> None:
        self.hu_min             = hu_min
        self.hu_max             = hu_max
        self.context_slices     = context_slices
        self.apply_clahe_flag   = apply_clahe_flag
        self.low_dose_i0        = low_dose_i0
        self.seed               = seed
        self.fast_mode          = fast_mode
        self.fast_mode_noise_std = fast_mode_noise_std

        # LRU cache: key = series_path, value = vol_nd ONLY
        # LD simulation happens per __getitem__ on 9 slices (microseconds)
        self._cache = _LRUCache(maxsize=cache_size)
        self._series_idx: dict[str, int] = {}  # O(1) lookup
        # .npy fast-path: series_path -> npy file path
        self._npy_map: dict[str, str] = npy_mapping or {}
        if self._npy_map:
            print(f"  [npy mode] {len(self._npy_map)} series have pre-computed .npy files.", flush=True)

        selected = split_entries[:max_cases] if max_cases is not None else split_entries

        # Fast scan: record (series_path, series_idx, z) without loading DICOM
        print(f"Scanning {len(selected)} series paths...", flush=True)
        self.series_list: list[tuple[str, int]] = []   # (path, series_idx)
        self.samples: list[tuple[str, int]] = []        # (series_path, z)

        for idx, item in enumerate(selected):
            series_path = str(item["file_location"])
            self.series_list.append((series_path, idx))
            self._series_idx[series_path] = idx

        # We need z-ranges — load shapes only (read one DICOM header per series)
        # This is ~0.1s per series vs ~5s for full Radon load
        print("Reading volume shapes (lightweight scan)...", flush=True)
        self._shape_cache: dict[str, tuple[int, int, int]] = {}
        from pathlib import Path as _Path
        import pydicom

        for series_path, idx in self.series_list:
            # If npy_mode is on, only process series that were successfully converted
            if self._npy_map and series_path not in self._npy_map:
                continue

            try:
                dcm_files = sorted(_Path(series_path).glob("*.dcm"))
                if not dcm_files:
                    continue
                sample_dcm = pydicom.dcmread(str(dcm_files[0]))
                rows = int(sample_dcm.Rows)
                cols = int(sample_dcm.Columns)
                z    = len(dcm_files)
                self._shape_cache[series_path] = (z, rows, cols)
                for zi in range(self.context_slices, z - self.context_slices):
                    self.samples.append((series_path, zi))
            except Exception as e:
                print(f"  [WARN] Skipping {series_path}: {e}", flush=True)

        print(
            f"Dataset ready: {len(self.series_list)} series, "
            f"{len(self.samples)} samples. Training starts now!",
            flush=True,
        )

        if not self.samples:
            raise RuntimeError("No samples found. Check splits path and DICOM files.")

    def _load_nd_volume(self, series_path: str) -> np.ndarray:
        """Load normal-dose volume — from .npy if available (fast), else DICOM."""
        cached = self._cache.get(series_path)
        if cached is not None:
            return cached

        npy_path = self._npy_map.get(series_path)
        if npy_path and Path(npy_path).exists():
            # Fast path: load pre-processed float16 .npy (~2s from Drive)
            volume_nd = np.load(npy_path).astype(np.float32)
        else:
            # Slow path: load raw DICOM + preprocess (~30s from Drive)
            volume_hu, _ = build_volume(series_path)
            volume_nd = hu_clip_normalize(volume_hu, hu_min=self.hu_min, hu_max=self.hu_max)
            if self.apply_clahe_flag:
                volume_nd = apply_clahe(volume_nd)

        self._cache.put(series_path, volume_nd)
        return volume_nd

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        series_path, z  = self.samples[index]
        series_idx      = self._series_idx.get(series_path, 0)

        # Load (or fetch from cache) the CLEAN normal-dose volume
        vol_nd = self._load_nd_volume(series_path)

        start = z - self.context_slices
        end   = z + self.context_slices + 1
        nd_window = vol_nd[start:end]    # (9, H, W) — normal-dose window
        nd_slice  = vol_nd[z]             # (H, W)   — central target slice

        # Simulate low-dose on ONLY the 9 needed slices — microseconds
        if self.fast_mode:
            rng      = np.random.default_rng(self.seed + series_idx * 1000 + z)
            noise    = rng.normal(0.0, self.fast_mode_noise_std,
                                  nd_window.shape).astype(np.float32)
            ld_stack = np.clip(nd_window + noise, 0.0, 1.0)
        else:
            # Fast pixel-space Beer-Lambert + Poisson (replaces Radon for online training)
            ld_stack = simulate_low_dose_fast(
                nd_window,
                i0=self.low_dose_i0,
                seed=self.seed + series_idx * 1000 + z,
            )

        return {
            "ld": torch.from_numpy(ld_stack.astype(np.float32)),
            "nd": torch.from_numpy(nd_slice.astype(np.float32)).unsqueeze(0),
            "x":  torch.from_numpy(ld_stack.astype(np.float32)),
            "y":  torch.from_numpy(nd_slice.astype(np.float32)).unsqueeze(0),
        }


# ---------------------------------------------------------------------------
# Grouped Series Sampler — maximizes LRU cache hits
# ---------------------------------------------------------------------------

class GroupedSeriesSampler(torch.utils.data.Sampler):
    """
    Groups all slice indices from the same series together into consecutive
    batches. This ensures each series is loaded from Drive ONLY ONCE per epoch,
    making the LRU cache effectively 100% efficient.

    Shuffle=True shuffles the ORDER of series across epochs, and shuffles slice
    order within each series, so the model still sees diverse training signal.
    """
    def __init__(
        self,
        samples: list[tuple[str, int]],
        shuffle: bool = True,
        seed: int = 42,
    ) -> None:
        self.samples = samples
        self.shuffle = shuffle
        self.epoch   = 0
        self.seed    = seed

        # Group indices by series_path
        self._groups: dict[str, list[int]] = {}
        for i, (series_path, _) in enumerate(samples):
            self._groups.setdefault(series_path, []).append(i)

    def set_epoch(self, epoch: int) -> None:
        """Call before each epoch for reproducible shuffling."""
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        series_keys = list(self._groups.keys())

        if self.shuffle:
            rng.shuffle(series_keys)

        for key in series_keys:
            indices = self._groups[key].copy()
            if self.shuffle:
                rng.shuffle(indices)
            yield from indices


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
