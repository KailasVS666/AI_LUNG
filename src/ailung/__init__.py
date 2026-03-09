from .dataset import LIDCSeries, discover_ct_series
from .preprocess import build_volume, hu_clip_normalize, resample_isotropic, apply_clahe, simulate_low_dose_volume
from .annotations import parse_lidc_xml
from .splits import build_patient_split, save_split, load_split
from .models import (
    Denoise25DUNet,
    DenoiseLoss,
    Recon3DUNet,
    Recon3DLoss,
    NoduleDetector3D,
    NoduleDetectionLoss,
    # Legacy aliases
    Denoise25DUNetSmall,
    Recon3DAttentionUNetSmall,
    PhysicsGuidedReconLoss,
    NoduleClassifier3D,
)
from .torch_dataset import LIDCDenoise25DDataset, LIDCRecon3DPatchDataset, NoduleDetectionDataset

__all__ = [
    # Dataset utilities
    "LIDCSeries",
    "discover_ct_series",
    # Preprocessing
    "build_volume",
    "hu_clip_normalize",
    "resample_isotropic",
    "apply_clahe",
    "simulate_low_dose_volume",
    # Annotations
    "parse_lidc_xml",
    # Splits
    "build_patient_split",
    "save_split",
    "load_split",
    # Stage 1
    "Denoise25DUNet",
    "DenoiseLoss",
    # Stage 2
    "Recon3DUNet",
    "Recon3DLoss",
    # Stage 3
    "NoduleDetector3D",
    "NoduleDetectionLoss",
    # Legacy aliases
    "Denoise25DUNetSmall",
    "Recon3DAttentionUNetSmall",
    "PhysicsGuidedReconLoss",
    "NoduleClassifier3D",
    # PyTorch Datasets
    "LIDCDenoise25DDataset",
    "LIDCRecon3DPatchDataset",
    "NoduleDetectionDataset",
]
