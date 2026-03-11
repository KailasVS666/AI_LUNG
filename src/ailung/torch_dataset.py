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
        max_samples_per_series: int | None = 64,  # Limit slices per volume for speed
        npy_mapping: dict[str, str] | None = None,
        local_cache_path: str | Path | None = None,
    ) -> None:
        self.hu_min             = hu_min
        self.hu_max             = hu_max
        self.context_slices     = context_slices
        self.apply_clahe_flag   = apply_clahe_flag
        self.low_dose_i0        = low_dose_i0
        self.seed               = seed
        self.fast_mode          = fast_mode
        self.fast_mode_noise_std = fast_mode_noise_std
        self.max_samples_per_series = max_samples_per_series
        self.local_cache_path = Path(local_cache_path) if local_cache_path else None

        # LRU cache: key = series_path, value = vol_nd ONLY
        # LD simulation happens per __getitem__ on 9 slices (microseconds)
        self._cache = _LRUCache(maxsize=cache_size)
        self._series_idx: dict[str, int] = {}  # O(1) lookup
        self._shape_cache: dict[str, tuple[int, int, int]] = {} # metadata cache
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

        # Threaded scan for shapes — reduces 10-minute wait to ~15 seconds on Google Drive
        from concurrent.futures import ThreadPoolExecutor
        from tqdm import tqdm as _tqdm
        import pydicom
        from pathlib import Path as _Path

        # Fast path: retrieves volume shapes from .npy headers if available, else DICOM
        def _get_shape_fast(path_idx):
            path, _ = path_idx
            
            # 1. Try NPY fast path (reads ONLY the header, very fast)
            # Check local cache first, then Drive mapping
            potential_npy = None
            if self.local_cache_path and self._npy_map.get(path):
                fname = Path(self._npy_map[path]).name
                potential_npy = self.local_cache_path / fname
            elif self._npy_map.get(path):
                potential_npy = Path(self._npy_map[path])
                
            if potential_npy and potential_npy.exists():
                try:
                    res = np.load(str(potential_npy), mmap_mode='r')
                    return path, res.shape
                except: pass
            
            # 2. Try DICOM fallback
            try:
                dcm_files = sorted(_Path(path).glob("*.dcm"))
                if dcm_files:
                    d = pydicom.dcmread(str(dcm_files[0]), stop_before_pixels=True)
                    return path, (len(dcm_files), int(d.Rows), int(d.Columns))
            except: pass
            
            return None

        print(f"  Scanning volume shapes for {len(self.series_list)} series (threaded)...", flush=True)
        with ThreadPoolExecutor(max_workers=12) as executor:
            # tqdm(executor.map) gives us a real-time progress bar for the scan
            shapes = list(_tqdm(executor.map(_get_shape_fast, self.series_list), 
                               total=len(self.series_list), desc="  Scan progress"))

        for res in shapes:
            if res:
                p, (z, r, c) = res
                self._shape_cache[p] = (z, r, c)
                
                # Determine which slices to train on
                all_possible_zi = list(range(self.context_slices, z - self.context_slices))
                
                if self.max_samples_per_series and len(all_possible_zi) > self.max_samples_per_series:
                    # Deterministic sampling based on series index for reproducibility
                    rng = np.random.default_rng(self.seed + self._series_idx[p])
                    sampled_zi = rng.choice(all_possible_zi, size=self.max_samples_per_series, replace=False)
                    for zi in sampled_zi:
                        self.samples.append((p, int(zi)))
                else:
                    for zi in all_possible_zi:
                        self.samples.append((p, zi))

        print(
            f"Dataset ready: {len(self._shape_cache)} series, "
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

        # 1. Try Local NVMe Cache (Super Fast)
        npy_filename = None
        if self._npy_map.get(series_path):
            npy_filename = Path(self._npy_map[series_path]).name
            
        if self.local_cache_path and npy_filename:
            local_path = self.local_cache_path / npy_filename
            if local_path.exists():
                try:
                    vol_nd = np.load(local_path).astype(np.float32)
                    self._cache.put(series_path, vol_nd)
                    return vol_nd
                except: pass

        # 2. Fallback to Drive/Original NPY Path
        npy_path = self._npy_map.get(series_path)
        if npy_path and Path(npy_path).exists():
            vol_nd = np.load(npy_path).astype(np.float32)
        else:
            vol_hu, _ = build_volume(series_path)
            vol_nd = hu_clip_normalize(vol_hu, hu_min=self.hu_min, hu_max=self.hu_max)
            if self.apply_clahe_flag:
                vol_nd = apply_clahe(vol_nd)

        self._cache.put(series_path, vol_nd)
        return vol_nd

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
    def __init__(self, dataset, batch_size, shuffle=True, seed=42, start_offset=0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.start_offset = start_offset # Skip this many BATCHES

        # Group indices by series_path
        self.series_groups: dict[str, list[int]] = {}
        for i, (series_path, _) in enumerate(dataset.samples):
            self.series_groups.setdefault(series_path, []).append(i)

    def set_epoch(self, epoch: int) -> None:
        """Call before each epoch for reproducible shuffling."""
        self.epoch = epoch

    def __len__(self) -> int:
        total_samples = len(self.dataset.samples)
        if self.start_offset <= 0:
            return total_samples
        # Return only the remaining samples
        remaining = total_samples - (self.start_offset * self.batch_size)
        return max(remaining, 0)

    def __iter__(self):
        # Deterministic shuffle based on epoch
        rng = np.random.default_rng(self.seed + self.epoch)
        
        series_indices = list(self.series_groups.keys())
        if self.shuffle:
            rng.shuffle(series_indices)

        all_indices = []
        for s_idx in series_indices:
            group = list(self.series_groups[s_idx])
            if self.shuffle:
                rng.shuffle(group)
            all_indices.extend(group)

        # Apply start_offset for instant resume (Peak Performance)
        if self.start_offset > 0:
            # We skip by BATCHES, so we need to know the batch size.
            # But the sampler works on SAMPLES. 
            # We convert our batch offset into a sample offset.
            sample_offset = self.start_offset * self.batch_size
            all_indices = all_indices[sample_offset:]

        return iter(all_indices)


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
        noise_std: float = 0.03,
        cache_size: int = 6,
        npy_mapping: dict[str, str] | None = None,
    ) -> None:
        self.hu_min             = hu_min
        self.hu_max             = hu_max
        self.patch_size         = patch_size
        self.patches_per_volume = patches_per_volume
        self.noise_std          = noise_std
        self.rng                = random.Random(seed)
        self.denoised_vol_dir   = Path(denoised_vol_dir) if denoised_vol_dir else None
        self._npy_map           = npy_mapping or {}
        self._cache             = _LRUCache(maxsize=cache_size)

        selected = split_entries[:max_cases] if max_cases is not None else split_entries

        # We store just the metadata; volumes load lazily in __getitem__
        self.samples: list[dict] = []
        pz, py, px = self.patch_size

        print(f"Initializing Stage 2 Dataset ({len(selected)} volumes)...", flush=True)

        for item in selected:
            series_path = str(item["file_location"])
            series_uid  = item.get("series_uid", "")

            # If we have an npy map, use it for z-count check to stay fast
            if self._npy_map and series_path in self._npy_map:
                try:
                    # mmap_mode='r' only reads the header
                    shape = np.load(self._npy_map[series_path], mmap_mode='r').shape
                    z_dim, y_dim, x_dim = shape
                except: continue
            else:
                # Fallback: lightweight DICOM shape check
                try:
                    dcm_files = list(Path(series_path).glob("*.dcm"))
                    z_dim, y_dim, x_dim = len(dcm_files), 512, 512
                except: continue

            if z_dim < pz: continue

            for _ in range(self.patches_per_volume):
                sz = self.rng.randint(0, z_dim - pz)
                sy = self.rng.randint(0, y_dim - py)
                sx = self.rng.randint(0, x_dim - px)
                self.samples.append({
                    "series_path": series_path, 
                    "series_uid": series_uid,
                    "coords": (sz, sy, sx)
                })

        if not self.samples:
            raise RuntimeError("No 3D patch samples created.")

    def _load_volumes(self, series_path: str, series_uid: str) -> tuple[np.ndarray, np.ndarray]:
        """Lazy load ND volume and Denoised/Input volume."""
        cached = self._cache.get(series_path)
        if cached is not None:
            return cached

        # 1. Load Ground Truth (Normal Dose)
        npy_path = self._npy_map.get(series_path)
        if npy_path and Path(npy_path).exists():
            vol_nd = np.load(npy_path).astype(np.float32)
        else:
            vol_hu, _ = build_volume(series_path)
            vol_nd = hu_clip_normalize(vol_hu, self.hu_min, self.hu_max)

        # 2. Load Input (Denoised)
        if self.denoised_vol_dir is not None:
            den_path = self.denoised_vol_dir / f"{series_uid}_denoised.npy"
            if den_path.exists():
                vol_in = np.load(str(den_path)).astype(np.float32)
            else:
                noise = np.random.normal(0.0, self.noise_std, vol_nd.shape).astype(np.float32)
                vol_in = np.clip(vol_nd + noise, 0.0, 1.0)
        else:
            noise = np.random.normal(0.0, self.noise_std, vol_nd.shape).astype(np.float32)
            vol_in = np.clip(vol_nd + noise, 0.0, 1.0)

        self._cache.put(series_path, (vol_nd, vol_in))
        return vol_nd, vol_in

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.samples[index]
        series_path, series_uid = sample["series_path"], sample["series_uid"]
        sz, sy, sx = sample["coords"]
        pz, py, px = self.patch_size

        vol_nd, vol_in = self._load_volumes(series_path, series_uid)

        target = vol_nd[sz:sz+pz, sy:sy+py, sx:sx+px]
        source = vol_in[sz:sz+pz, sy:sy+py, sx:sx+px]

        return {
            "x": torch.from_numpy(source).unsqueeze(0),
            "y": torch.from_numpy(target).unsqueeze(0),
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
        cache_size: int = 6,
        npy_mapping: dict[str, str] | None = None,
    ) -> None:
        self.hu_min = hu_min
        self.hu_max = hu_max
        self.patch_size = patch_size
        self.negatives_per_positive = negatives_per_positive
        self.xml_dir = Path(xml_dir)
        self.rng = random.Random(seed)
        self.reconstructed_vol_dir = Path(reconstructed_vol_dir) if reconstructed_vol_dir else None
        self._npy_map = npy_mapping or {}
        self._cache = _LRUCache(maxsize=cache_size)

        print("Building series UID → XML mapping...")
        self.series_to_xml = build_series_to_xml_mapping(xml_dir)

        selected = split_entries[:max_cases] if max_cases is not None else split_entries
        self.samples: list[dict] = []
        self._metadata: dict[str, tuple] = {} # series_path -> (volume_shape, spacing)

        pz, py, px = self.patch_size

        for item in selected:
            series_path = str(item["file_location"])
            series_uid  = item["series_uid"]

            # Lightweight metadata load
            try:
                if self._npy_map and series_path in self._npy_map:
                    v_shape = np.load(self._npy_map[series_path], mmap_mode='r').shape
                    v_spacing = (1.0, 1.0, 1.0) # Assume standardized for Stage 3
                else:
                    dcm_files = list(Path(series_path).glob("*.dcm"))
                    v_shape = (len(dcm_files), 512, 512)
                    v_spacing = (1.0, 1.0, 1.0) # placeholders
                
                self._metadata[series_path] = (v_shape, v_spacing)
            except: continue

            xml_path = self.series_to_xml.get(series_uid)
            if xml_path is None:
                self._add_negative_samples(series_path, self.negatives_per_positive * 3)
                continue

            nodule_candidates = build_nodule_candidates(xml_path, v_spacing, min_malignancy=min_malignancy)
            for candidate in nodule_candidates:
                self.samples.append({"series_path": series_path, "series_uid": series_uid, "center_mm": candidate["centroid_3d"], "label": 1})

            self._add_negative_samples(series_path, max(len(nodule_candidates) * self.negatives_per_positive, 1))

    def _get_volume(self, series_path: str, series_uid: str) -> np.ndarray:
        cached = self._cache.get(series_path)
        if cached is not None: return cached

        # Try stage-specific reconstructed volume first
        if self.reconstructed_vol_dir is not None:
            rec_path = self.reconstructed_vol_dir / f"{series_uid}_recon3d.npy"
            if rec_path.exists():
                vol = np.load(str(rec_path)).astype(np.float32)
                self._cache.put(series_path, vol)
                return vol

        # Fallback to pre-processed ND volume
        npy_path = self._npy_map.get(series_path)
        if npy_path and Path(npy_path).exists():
            vol = np.load(npy_path).astype(np.float32)
        else:
            hu, _ = build_volume(series_path)
            vol = hu_clip_normalize(hu, self.hu_min, self.hu_max)
        
        self._cache.put(series_path, vol)
        return vol

    def _add_negative_samples(self, series_path: str, count: int) -> None:
        v_shape, spacing = self._metadata[series_path]
        pz, py, px = self.patch_size
        for _ in range(count):
            sz = self.rng.randint(0, v_shape[0] - pz)
            sy = self.rng.randint(0, v_shape[1] - py)
            sx = self.rng.randint(0, v_shape[2] - px)
            self.samples.append({
                "series_path": series_path, 
                "series_uid": "", # only needed for recon path
                "center_mm": ((sz+pz/2)*spacing[0], (sy+py/2)*spacing[1], (sx+px/2)*spacing[2]),
                "label": 0
            })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        s = self.samples[index]
        vol = self._get_volume(s["series_path"], s.get("series_uid", ""))
        v_shape, spacing = self._metadata[s["series_path"]]
        pz, py, px = self.patch_size

        zv, yv, xv = int(s["center_mm"][0]/spacing[0]), int(s["center_mm"][1]/spacing[1]), int(s["center_mm"][2]/spacing[2])
        sz, sy, sx = max(0, min(zv-pz//2, v_shape[0]-pz)), max(0, min(yv-py//2, v_shape[1]-py)), max(0, min(xv-px//2, v_shape[2]-px))
        patch = vol[sz:sz+pz, sy:sy+py, sx:sx+px].astype(np.float32)

        return {"x": torch.from_numpy(patch).unsqueeze(0), "y": torch.tensor(s["label"], dtype=torch.long)}
