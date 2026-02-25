from __future__ import annotations

from pathlib import Path

import numpy as np
import pydicom
from scipy.ndimage import zoom


def _safe_float(value: object, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _load_slices(series_path: str | Path) -> list[pydicom.dataset.FileDataset]:
    series_dir = Path(series_path)
    slices = []
    for dcm_path in sorted(series_dir.glob("*.dcm")):
        ds = pydicom.dcmread(str(dcm_path), force=True)
        slices.append(ds)

    if not slices:
        raise FileNotFoundError(f"No DICOM slices found in: {series_dir}")

    slices.sort(
        key=lambda x: _safe_float(getattr(x, "ImagePositionPatient", [0, 0, 0])[2], _safe_float(getattr(x, "SliceLocation", 0.0), 0.0))
    )
    return slices


def build_volume(series_path: str | Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    slices = _load_slices(series_path)

    image_stack = np.stack([s.pixel_array.astype(np.int16) for s in slices], axis=0)

    first = slices[0]
    slope = _safe_float(getattr(first, "RescaleSlope", 1.0), 1.0)
    intercept = _safe_float(getattr(first, "RescaleIntercept", 0.0), 0.0)

    volume_hu = image_stack.astype(np.float32) * slope + intercept

    pixel_spacing = getattr(first, "PixelSpacing", [1.0, 1.0])
    spacing_y = _safe_float(pixel_spacing[0], 1.0)
    spacing_x = _safe_float(pixel_spacing[1], 1.0)

    if len(slices) > 1:
        z0 = _safe_float(getattr(slices[0], "ImagePositionPatient", [0, 0, 0])[2], 0.0)
        z1 = _safe_float(getattr(slices[1], "ImagePositionPatient", [0, 0, 0])[2], z0 + 1.0)
        spacing_z = abs(z1 - z0) if abs(z1 - z0) > 0 else _safe_float(getattr(first, "SliceThickness", 1.0), 1.0)
    else:
        spacing_z = _safe_float(getattr(first, "SliceThickness", 1.0), 1.0)

    spacing = (spacing_z, spacing_y, spacing_x)
    return volume_hu, spacing


def hu_clip_normalize(volume_hu: np.ndarray, hu_min: int = -1000, hu_max: int = 400) -> np.ndarray:
    clipped = np.clip(volume_hu, hu_min, hu_max)
    normalized = (clipped - hu_min) / (hu_max - hu_min)
    return normalized.astype(np.float32)


def resample_isotropic(volume: np.ndarray, spacing: tuple[float, float, float], target_spacing: float = 1.0) -> tuple[np.ndarray, tuple[float, float, float]]:
    zoom_factors = tuple(s / target_spacing for s in spacing)
    resampled = zoom(volume, zoom=zoom_factors, order=1)
    return resampled.astype(np.float32), (target_spacing, target_spacing, target_spacing)
