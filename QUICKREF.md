# AI_LUNG Quick Reference Card

## Essential Commands

### Setup (One-time)
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

### Data Preparation
```bash
# Generate splits
python scripts/build_splits.py \
  --dataset_root manifest-1600709154662 \
  --metadata_csv manifest-1600709154662/metadata.csv \
  --output outputs/splits/patient_split.json

# Validate single patient
python scripts/sanity_check.py \
  --dataset_root manifest-1600709154662 \
  --metadata_csv manifest-1600709154662/metadata.csv \
  --xml_dir LIDC-XML-only/tcia-lidc-xml \
  --patient_id LIDC-IDRI-0001
```

### Training Commands
```bash
# 2.5D Denoising
python scripts/train_denoiser_baseline.py --config configs/baseline.yaml

# 3D Reconstruction
python scripts/train_recon3d.py --config configs/recon3d.yaml

# Nodule Detection
python scripts/train_nodule_detector.py --config configs/nodule_detection.yaml
```

## Key Files & Locations

### Source Code
| File | Purpose |
|------|---------|
| `src/ailung/dataset.py` | LIDC series discovery from metadata.csv |
| `src/ailung/preprocess.py` | DICOM→HU conversion, normalization, resampling |
| `src/ailung/annotations.py` | XML parsing, nodule centroid extraction |
| `src/ailung/splits.py` | Patient-wise train/val/test splitting |
| `src/ailung/torch_dataset.py` | PyTorch datasets for all 3 modules |
| `src/ailung/models.py` | Neural network architectures + losses |

### Training Scripts
| Script | Module | Output Dir |
|--------|--------|------------|
| `scripts/train_denoiser_baseline.py` | 2.5D Denoising | `outputs/train_runs/baseline_denoiser/` |
| `scripts/train_recon3d.py` | 3D Reconstruction | `outputs/train_runs/recon3d/` |
| `scripts/train_nodule_detector.py` | Nodule Detection | `outputs/train_runs/nodule_detection/` |

### Configuration Files
| Config | Module | Key Parameters |
|--------|--------|----------------|
| `configs/baseline.yaml` | 2.5D Denoising | `context_slices: 1`, `batch_size: 4` |
| `configs/recon3d.yaml` | 3D Reconstruction | `patch_size: [32,96,96]`, `noise_std: 0.03` |
| `configs/nodule_detection.yaml` | Detection | `min_malignancy: 1`, `negatives_per_positive: 2` |

### Output Artifacts
```
outputs/
├── splits/
│   └── patient_split.json              # 713 train / 153 val / 153 test
│
└── train_runs/
    ├── baseline_denoiser/
    │   ├── denoiser_last.pt            # Final checkpoint
    │   ├── history.json                # Metrics per epoch
    │   └── epoch_*.png                 # Visual previews
    │
    ├── recon3d/
    │   ├── recon3d_last.pt
    │   ├── recon3d_best.pt             # Best PSNR checkpoint
    │   ├── history.json
    │   └── epoch_*_preview.png
    │
    └── nodule_detection/
        ├── nodule_detector_last.pt
        ├── nodule_detector_best.pt     # Best AUC checkpoint
        ├── history.json
        └── training_curves.png
```

## Model Architectures

### Stage 1 — `Denoise25DUNet`
- **Input**: `(B, 9, H, W)` — 9 consecutive LDCT slices (4 context above + target + 4 context below)
- **Encoder**: EfficientNet-B5-style depthwise-separable convolutions → 48→24→40→80→192→320 channels
- **Attention**: CBAM2D at every skip connection and bottleneck
- **Loss**: L1 + (1 − SSIM) + Gradient
- **Output**: `(B, 1, H, W)` — denoised central slice (sigmoid activated)

### Stage 2 — `Recon3DUNet`
- **Input**: `(B, 1, 64, 64, 64)` — 3D patch from stacked denoised slices
- **Architecture**: 4-level 3D U-Net with InstanceNorm3d + LeakyReLU, CBAM3D at each encoder stage
- **Channels**: 16 → 32 → 64 → 128 (base_channels=16)
- **Loss**: L1 + SSIM3D + Gradient3D + Projection Consistency + Range Penalty
- **Output**: `(B, 1, 64, 64, 64)` — refined 3D lung volume (clamped to [0, 1])

### Stage 3 — `NoduleDetector3D`
- **Input**: `(B, 1, 32, 64, 64)` — 3D patch from reconstructed volume
- **Architecture**: 4× Conv3D blocks (32→64→128→256) + CBAM3D + GlobalAvgPool + 2-layer FC classifier
- **Dropout**: 0.4 → 0.2
- **Loss**: Soft Dice + Cross-Entropy
- **Output**: `(B, 2)` — logits for [benign, malignant]

> **Note**: `NoduleClassifier3D`, `Recon3DAttentionUNetSmall`, `Denoise25DUNetSmall` are legacy aliases pointing to the above classes.

## Common Config Tweaks

### For Faster Iteration
```yaml
max_cases_train: 10    # Default: 20
max_cases_val: 3       # Default: 5
epochs: 1              # Default: 2-3
```

### For Production
```yaml
max_cases_train: null  # Use all 713 series
max_cases_val: null    # Use all 153 series
epochs: 30             # Default: 2-3
batch_size: 8          # If GPU available (default: 4)
```

### For Memory Issues
```yaml
batch_size: 2          # Reduce from 4
patch_size: [24, 64, 64]  # For 3D modules (reduce first dim)
```

## Validation Metrics

### What to Expect (Local Small-Scale)
| Module | Metric | Small Run | Full Expected |
|--------|--------|-----------|---------------|
| Denoising | PSNR | 28-30 dB | 32-35 dB |
| | SSIM | 0.85-0.90 | 0.92-0.96 |
| Reconstruction | PSNR gain | +2 dB | +3-5 dB |
| | SSIM gain | +0.10 | +0.15-0.25 |
| Detection | AUC | 0.90-0.95 | 0.96-0.98 |
| | Sensitivity@90%Spec | 0.80-0.90 | 0.90-0.95 |

## Git Workflow

```bash
# Check status
git status

# Stage changes
git add src/ scripts/ configs/

# Commit
git commit -m "feat: Your description"

# Push to GitHub
git push origin main

# View history
git log --oneline -5
```

## Dataset Info

### LIDC-IDRI Statistics
- **Patients**: 1,018
- **CT Series**: 1,018
- **DICOM Slices**: ~244,000+
- **XML Files**: 1,018 (series UIDs embedded in `<SeriesInstanceUid>` tag)
- **Nodules**: ~2,600 annotated by 4 radiologists
- **Malignancy Range**: 1 (benign) to 5 (highly suspicious)
- **Split**: 713 train / 153 val / 153 test (patient-wise, no leakage)

### Preprocessing Pipeline
```
DICOM folder → pydicom.dcmread()
            → Sort by ImagePositionPatient[2] (z-axis)
            → Stack to 3D: (Z, Y, X)
            → Apply HU: pixel_array × RescaleSlope + RescaleIntercept
            → Clip HU [-1000, 400]
            → Normalize [0, 1]
            → Resample to isotropic 1mm³ spacing
```

### XML Matching
- XMLs named by numeric ID (158.xml, 159.xml...)
- Contains `<SeriesInstanceUid>` tag inside
- `build_series_to_xml_mapping()` scans 1,018 XMLs → creates UID lookup

## Troubleshooting Quick Fixes

| Error | Quick Fix |
|-------|-----------|
| `FileNotFoundError` | Check paths in config match your system |
| `CUDA out of memory` | Reduce `batch_size` or `patch_size` |
| `No training samples` | Verify dataset paths with `sanity_check.py` |
| `ROC AUC not defined` | Lower `min_malignancy` or increase `max_cases_val` |
| Import errors | `pip install -e .` and restart terminal |
| Slow training | Use GPU (Colab) or reduce dataset size |

## Useful Python Snippets

### Load a Checkpoint
```python
from ailung.models import NoduleClassifier3D
import torch

model = NoduleClassifier3D()
model.load_state_dict(torch.load("outputs/train_runs/nodule_detection/nodule_detector_best.pt"))
model.eval()
print("Model loaded successfully")
```

### Inspect Training History
```python
import json

with open("outputs/train_runs/recon3d/history.json") as f:
    history = json.load(f)

print(f"Final val PSNR: {history['val_psnr'][-1]:.4f}")
print(f"Best val PSNR: {max(history['val_psnr']):.4f}")
```

### Quick Dataset Stats
```python
from ailung.dataset import discover_ct_series
from ailung.splits import load_split

series = discover_ct_series("manifest-1600709154662", "manifest-1600709154662/metadata.csv")
splits = load_split("outputs/splits/patient_split.json")

print(f"Total series: {len(series)}")
print(f"Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")
```

---

**See [README.md](README.md) for full project overview**  
**See [TRAINING_GUIDE.md](TRAINING_GUIDE.md) for detailed training instructions**  
**See [DOCUMENTATION.md](DOCUMENTATION.md) for full academic documentation (Abstract → References)**
