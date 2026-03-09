# AI_LUNG: AI-Powered 3D Lung Imaging for Early Cancer Detection

A deep learning pipeline for processing low-dose CT (LDCT) scans to enable affordable early lung cancer detection in low- and middle-income countries (LMICs).

## Overview

This project implements a **3-stage sequential AI pipeline**:

| Stage | Task | Model | Loss |
|-------|------|-------|------|
| 1 | **2.5D Denoising** | EfficientNet-B5 encoder + CBAM U-Net | L1 + (1−SSIM) + Gradient |
| 2 | **3D Reconstruction** | Lightweight 3D U-Net + 3D-CBAM | L1 + SSIM3D + Gradient3D + Projection |
| 3 | **Nodule Detection** | 3D CNN + CBAM + classifier | Dice + Cross-Entropy |

### Why Low-Dose CT?
LDCT scans are cheaper and safer (less radiation) but produce noisy images. Stage 1 enhances them to match high-dose quality. Stage 2 reconstructs the full 3D lung volume. Stage 3 flags suspicious nodules for radiologists.

### Low-Dose Simulation
Since LIDC-IDRI contains normal-dose scans, we simulate low-dose by:
1. Radon transform (forward projection)
2. Poisson noise injection into sinogram
3. Filtered Back-Projection (FBP) reconstruction
→ Gives low-dose input + normal-dose ground truth pairs

## Dataset

**LIDC-IDRI** (Lung Image Database Consortium)
- **Size**: 1,018 patients, CT series in DICOM format
- **Annotations**: 4-radiologist consensus XML files with nodule malignancy ratings
- **Source**: [TCIA LIDC-IDRI Collection](https://www.cancerimagingarchive.net/collection/lidc-idri/)

### Dataset Structure
```
manifest-1600709154662/
├── metadata.csv
└── LIDC-IDRI/
    └── LIDC-IDRI-0001/
        └── <study>/
            └── <series>/   ← DICOM files

LIDC-XML-only/
└── tcia-lidc-xml/
    └── */*.xml             ← Nodule annotations
```

## Installation

### Prerequisites
- Python 3.12+
- 16 GB+ RAM (for full training)
- GPU recommended for training

### Setup

```bash
git clone https://github.com/KailasVS666/AI_LUNG.git
cd AI_LUNG
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # Linux/macOS

pip install -r requirements.txt
pip install -e .
```

### Dependencies
| Package | Purpose |
|---------|---------|
| `torch >= 2.2` | Deep learning |
| `monai >= 1.3` | Medical imaging utilities |
| `pydicom >= 2.4` | DICOM file reading |
| `scikit-image >= 0.22` | Radon/FBP transform |
| `opencv-python >= 4.9` | CLAHE enhancement |
| `scipy >= 1.11` | Resampling |
| `scikit-learn >= 1.3` | AUC metrics |
| `numpy, pandas, matplotlib, PyYAML` | Utilities |

## Repository Structure

```
AI_LUNG/
├── src/ailung/                       # Core library
│   ├── dataset.py                    # LIDC series discovery
│   ├── preprocess.py                 # DICOM→HU, CLAHE, low-dose simulation
│   ├── annotations.py                # XML parsing, nodule extraction
│   ├── splits.py                     # Patient-wise splits
│   ├── torch_dataset.py              # PyTorch datasets (all 3 stages)
│   └── models.py                     # Neural network architectures + losses
│
├── scripts/
│   ├── build_splits.py               # Generate data splits
│   ├── sanity_check.py               # Single-patient validation
│   ├── train_denoiser_baseline.py    # Stage 1 training
│   ├── export_denoised.py            # Stage 1→2 bridge (export denoised .npy)
│   ├── train_recon3d.py              # Stage 2 training
│   └── train_nodule_detector.py      # Stage 3 training
│
├── configs/
│   ├── baseline.yaml / baseline_colab.yaml          # Stage 1
│   ├── recon3d.yaml / recon3d_colab.yaml            # Stage 2
│   └── nodule_detection.yaml / nodule_detection_colab.yaml  # Stage 3
│
├── notebooks/
│   └── train_colab.ipynb             # Full end-to-end Colab notebook
│
├── outputs/                          # Generated files (not in git)
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Training Pipeline

### (Quick Reference — Full instructions in `TRAINING_GUIDE.md`)

#### 1. Build Data Splits
```bash
python scripts/build_splits.py \
  --dataset-root manifest-1600709154662 \
  --metadata-csv manifest-1600709154662/metadata.csv \
  --out outputs/splits/patient_split.json
```

#### 2. Stage 1 — Denoiser (local test)
```bash
python scripts/train_denoiser_baseline.py --config configs/baseline.yaml
```
- **Input**: 9 low-dose slices → **Output**: 1 denoised slice
- **Metrics**: PSNR, SSIM

#### 3. Export Denoised Volumes (Stage 1 → Stage 2 bridge)
```bash
python scripts/export_denoised.py --config configs/baseline.yaml
```
Saves `<series_uid>_denoised.npy` for each series.

#### 4. Stage 2 — 3D Reconstruction (local test)
```bash
python scripts/train_recon3d.py --config configs/recon3d.yaml
```
- **Input**: 64×64×64 patch of denoised volume → **Output**: refined 3D patch
- **Metrics**: PSNR, SSIM

#### 5. Stage 3 — Nodule Detection (local test)
```bash
python scripts/train_nodule_detector.py --config configs/nodule_detection.yaml
```
- **Input**: 32×64×64 patch → **Output**: benign/malignant classification
- **Metrics**: AUC, Sensitivity, Specificity

## Preprocessing Pipeline

```
Load DICOM
  → Convert to Hounsfield Units (HU)
  → Clip to [-1000, 400] lung window
  → Normalize to [0, 1]
  → Apply CLAHE (contrast enhancement, slice-by-slice)
  → Simulate low-dose (Radon → Poisson noise → FBP)  ← Stage 1 input
  → Extract overlapping 3D patches (64×64×64)
  → Parse nodule annotations from XML
```

## Google Colab Training

Use `notebooks/train_colab.ipynb`. The notebook covers all stages sequentially:

1. GPU check → Mount Drive
2. Unzip dataset → Verify layout
3. Clone repo → Install deps
4. Generate splits (or restore from Drive)
5. **Stage 1**: Train denoiser
6. **Bridge**: Export denoised volumes
7. **Stage 2**: Train 3D reconstruction
8. **Stage 3**: Train nodule detector
9. Final results summary + checkpoint verification

All checkpoints are saved directly to Drive under `MyDrive/AI_LUNG_DATA/outputs/`.

## Project Status

- ✅ Dataset validation and exploration
- ✅ Full preprocessing pipeline (DICOM, HU, CLAHE, Radon/Poisson/FBP)
- ✅ Patient-wise data splitting (no leakage)
- ✅ Stage 1: EfficientNet-B5 + CBAM 2.5D U-Net with L1+SSIM+Grad loss
- ✅ Stage 1→2 bridge: export_denoised.py
- ✅ Stage 2: 3D U-Net + 3D-CBAM with L1+SSIM3D+Grad+Projection loss
- ✅ Stage 3: 3D CNN + CBAM with Dice+CE loss
- ✅ Full Colab notebook (sequential pipeline)
- ⏳ Full-scale Colab training
- ⏳ Ablation studies
- ⏳ Final evaluation and reporting

## Citation

If you use LIDC-IDRI dataset, please cite:

```bibtex
@article{armato2011lidc,
  title={The lung image database consortium (LIDC) and image database resource initiative (IDRI): a completed reference database of lung nodules on CT scans},
  author={Armato III, Samuel G and McLennan, Geoffrey and others},
  journal={Medical physics},
  volume={38},
  number={2},
  pages={915--931},
  year={2011},
  publisher={Wiley Online Library}
}
```

## Contact

**GitHub**: https://github.com/KailasVS666/AI_LUNG

---

**Last Updated**: March 2026  
**Status**: Full architecture implemented, ready for Colab training
