"""
AI-LUNG Preprocessing Pipeline
================================
Steps (in order):
  1. Load DICOM files
  2. Convert to Hounsfield Units (HU)
  3. Clip HU to [-1000, 400] lung window
  4. Normalize to [0, 1]
  5. Apply CLAHE contrast enhancement
  6. Simulate low-dose (Radon → Poisson noise → FBP reconstruction)
  7. Utility: resample to isotropic 1 mm spacing
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pydicom
from scipy.ndimage import zoom


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_float(value: object, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


# ---------------------------------------------------------------------------
# Step 1 + 2: Load DICOM → HU volume
# ---------------------------------------------------------------------------

def _load_slices(series_path: str | Path) -> list[pydicom.dataset.FileDataset]:
    series_dir = Path(series_path)
    slices = [
        pydicom.dcmread(str(p), force=True)
        for p in sorted(series_dir.glob("*.dcm"))
    ]
    if not slices:
        raise FileNotFoundError(f"No DICOM slices found in: {series_dir}")

    slices.sort(
        key=lambda x: _safe_float(
            getattr(x, "ImagePositionPatient", [0, 0, 0])[2],
            _safe_float(getattr(x, "SliceLocation", 0.0), 0.0),
        )
    )
    return slices


def build_volume(
    series_path: str | Path,
) -> tuple[np.ndarray, tuple[float, float, float]]:
    """
    Load a DICOM series and return:
      - volume_hu : float32 array (Z, Y, X) in Hounsfield Units
      - spacing   : (z_mm, y_mm, x_mm) voxel spacing
    """
    slices = _load_slices(series_path)

    image_stack = np.stack(
        [s.pixel_array.astype(np.int16) for s in slices], axis=0
    )

    first   = slices[0]
    slope   = _safe_float(getattr(first, "RescaleSlope", 1.0), 1.0)
    intercept = _safe_float(getattr(first, "RescaleIntercept", 0.0), 0.0)
    volume_hu = image_stack.astype(np.float32) * slope + intercept

    pixel_spacing = getattr(first, "PixelSpacing", [1.0, 1.0])
    spacing_y = _safe_float(pixel_spacing[0], 1.0)
    spacing_x = _safe_float(pixel_spacing[1], 1.0)

    if len(slices) > 1:
        z0 = _safe_float(getattr(slices[0], "ImagePositionPatient", [0, 0, 0])[2], 0.0)
        z1 = _safe_float(getattr(slices[1], "ImagePositionPatient", [0, 0, 0])[2], z0 + 1.0)
        spacing_z = abs(z1 - z0) if abs(z1 - z0) > 0 else _safe_float(
            getattr(first, "SliceThickness", 1.0), 1.0
        )
    else:
        spacing_z = _safe_float(getattr(first, "SliceThickness", 1.0), 1.0)

    return volume_hu, (spacing_z, spacing_y, spacing_x)


# ---------------------------------------------------------------------------
# Step 3 + 4: Clip HU + Normalize to [0, 1]
# ---------------------------------------------------------------------------

def hu_clip_normalize(
    volume_hu: np.ndarray,
    hu_min: int = -1000,
    hu_max: int = 400,
) -> np.ndarray:
    """Clip to lung window [-1000, 400] HU and normalize to [0, 1]."""
    clipped = np.clip(volume_hu, hu_min, hu_max)
    normalized = (clipped - hu_min) / (hu_max - hu_min)
    return normalized.astype(np.float32)


# ---------------------------------------------------------------------------
# Step 5: CLAHE contrast enhancement (slice-by-slice)
# ---------------------------------------------------------------------------

def apply_clahe(
    volume: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to each
    axial slice of a normalized [0, 1] float32 volume.

    Returns the enhanced volume in [0, 1].
    """
    try:
        import cv2  # OpenCV — optional; falls back gracefully if not installed
    except ImportError:
        # Fallback: no-op if cv2 unavailable (e.g. local dev without it)
        return volume

    out = np.empty_like(volume)
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit, tileGridSize=tile_grid_size
    )
    for z in range(volume.shape[0]):
        slice_u8 = (volume[z] * 255).astype(np.uint8)
        enhanced = clahe.apply(slice_u8)
        out[z] = enhanced.astype(np.float32) / 255.0
    return out


# ---------------------------------------------------------------------------
# Step 6: Low-Dose Simulation (Radon → Poisson noise → FBP)
# ---------------------------------------------------------------------------

def simulate_low_dose(
    slice_2d: np.ndarray,
    i0: float = 1e5,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Simulate a low-dose CT slice from a normal-dose 2D slice using:
      1. Radon transform  (forward projection)
      2. Poisson noise injection  (I0 * exp(-sinogram))
      3. Filtered Back-Projection  (FBP / iradon)

    Args:
        slice_2d : 2D float32 array in [0, 1]  (one axial CT slice)
        i0       : Photon count (lower = more noise). Default 1e5 mimics low-dose.
        seed     : Optional random seed for reproducibility.

    Returns:
        Low-dose reconstruction as float32 in [0, 1].

    Note:
        Requires scikit-image >= 0.19.
        This function is called per-slice during dataset construction.
    """
    from skimage.transform import radon, iradon

    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    # Radon angles: equally spaced over 180°
    theta = np.linspace(0.0, 180.0, max(slice_2d.shape[0], 36), endpoint=False)

    # Forward project — sinogram is in [0, 1] × path length units
    sinogram = radon(slice_2d, theta=theta, circle=True)

    # Convert to attenuation * path length scale for photon model
    # (clip so exp doesn't overflow)
    sinogram_scaled = np.clip(sinogram * slice_2d.max() + 1e-8, 0.0, None)

    # Poisson noise model: I_measured = Poisson(I0 * exp(-sinogram))
    expected_counts = i0 * np.exp(-sinogram_scaled)
    noisy_counts = rng.poisson(expected_counts).astype(np.float64)
    noisy_counts = np.clip(noisy_counts, 1, None)  # avoid log(0)

    # Back to attenuation
    noisy_sinogram = -np.log(noisy_counts / i0)

    # FBP reconstruction
    reconstruction = iradon(noisy_sinogram, theta=theta, circle=True, filter_name="ramp")

    # Re-normalize to [0, 1]
    r_min, r_max = reconstruction.min(), reconstruction.max()
    if r_max > r_min:
        reconstruction = (reconstruction - r_min) / (r_max - r_min)
    else:
        reconstruction = np.zeros_like(reconstruction)

    return reconstruction.astype(np.float32)


def simulate_low_dose_volume(
    volume: np.ndarray,
    i0: float = 1e5,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Apply simulate_low_dose to every axial slice of a 3D volume.

    Args:
        volume : (Z, Y, X) float32 array in [0, 1]
        i0     : Photon count (lower = noisier)
        seed   : Optional base seed (each slice gets seed+z for reproducibility)

    Returns:
        (Z, Y, X) float32 low-dose volume.
    """
    out = np.empty_like(volume)
    for z in range(volume.shape[0]):
        slice_seed = None if seed is None else seed + z
        out[z] = simulate_low_dose(volume[z], i0=i0, seed=slice_seed)
    return out


# ---------------------------------------------------------------------------
# Utility: Isotropic resampling
# ---------------------------------------------------------------------------

def resample_isotropic(
    volume: np.ndarray,
    spacing: tuple[float, float, float],
    target_spacing: float = 1.0,
) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Resample volume to isotropic voxel spacing (default 1 mm)."""
    zoom_factors = tuple(s / target_spacing for s in spacing)
    resampled = zoom(volume, zoom=zoom_factors, order=1)
    return resampled.astype(np.float32), (target_spacing, target_spacing, target_spacing)
