# AI_LUNG: AI-Powered 3D Lung Imaging for Early Cancer Detection

A deep learning pipeline for processing low-dose CT (LDCT) scans to enable affordable early lung cancer detection in low- and middle-income countries (LMICs).

## Overview

This project implements three core AI modules:

1. **2.5D Denoising**: Reduces noise in LDCT scans using multi-slice context U-Net
2. **3D Reconstruction**: Enhances volumetric quality with attention-based 3D U-Net and physics-guided loss
3. **Nodule Detection**: Identifies and classifies lung nodules using 3D CNN classifier with ROC metrics

## Dataset

**LIDC-IDRI** (Lung Image Database Consortium - Image Database Resource Initiative)
- **Size**: 1,011 patients, 1,019 CT series, 244,527 DICOM slices
- **Annotations**: 1,180 XML files with 4-radiologist consensus
- **Labels**: Nodule characteristics (malignancy 1-5, subtlety, spiculation, etc.)
- **Source**: [TCIA LIDC-IDRI Collection](https://www.cancerimagingarchive.net/)

### Dataset Structure
```
manifest-1600709154662/
├── metadata.csv                    # Series metadata
└── LIDC-IDRI/
    ├── LIDC-IDRI-0001/            # Patient folders
    │   └── 01-01-2000-NA-NA-*/    # Study folders
    │       └── */                  # DICOM series folders
    └── ...

LIDC-XML-only/
└── tcia-lidc-xml/
    └── */*.xml                     # Annotation files
```

## Installation

### Prerequisites
- Python 3.12+
- Windows/Linux/macOS
- 16GB+ RAM recommended for full training

### Setup

```bash
# Clone repository
git clone https://github.com/KailasVS666/AI_LUNG.git
cd AI_LUNG

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/macOS

# Install dependencies
pip install -r requirements.txt

# Install package in editable mode
pip install -e .
```

### Dependencies
- **Deep Learning**: PyTorch 2.2+
- **Medical Imaging**: pydicom 2.4+, scipy 1.11+, scikit-image 0.22+
- **Data Processing**: numpy 1.26+, pandas 2.2+
- **Metrics**: scikit-learn 1.3+
- **Visualization**: matplotlib 3.8+
- **Configuration**: PyYAML 6.0+

## Repository Structure

```
AI_LUNG/
├── src/ailung/                     # Core library
│   ├── dataset.py                  # LIDC series discovery
│   ├── preprocess.py               # DICOM processing, HU normalization
│   ├── annotations.py              # XML parsing, nodule extraction
│   ├── splits.py                   # Patient-wise train/val/test splits
│   ├── torch_dataset.py            # PyTorch datasets (3 modules)
│   └── models.py                   # Neural network architectures
│
├── scripts/                        # Training executables
│   ├── sanity_check.py             # Single-patient validation
│   ├── build_splits.py             # Generate data splits
│   ├── train_denoiser_baseline.py  # 2.5D denoising training
│   ├── train_recon3d.py            # 3D reconstruction training
│   └── train_nodule_detector.py    # Nodule detection training
│
├── configs/                        # YAML hyperparameters
│   ├── baseline.yaml               # 2.5D denoiser config
│   ├── recon3d.yaml                # 3D reconstruction config
│   └── nodule_detection.yaml       # Detection config
│
├── outputs/                        # Generated files (not in git)
│   ├── splits/                     # Train/val/test splits
│   └── train_runs/                 # Checkpoints, logs, plots
│
├── requirements.txt                # Python dependencies
├── pyproject.toml                  # Package metadata
├── README.md                       # This file
└── TRAINING_GUIDE.md               # Detailed training instructions
```

## Quick Start

### 1. Generate Data Splits

```bash
python scripts/build_splits.py \
  --dataset_root manifest-1600709154662 \
  --metadata_csv manifest-1600709154662/metadata.csv \
  --output outputs/splits/patient_split.json \
  --train_frac 0.70 \
  --val_frac 0.15
```

**Output**: `outputs/splits/patient_split.json` (713 train / 153 val / 153 test series)

### 2. Training Modules

#### 2.5D Denoising
```bash
python scripts/train_denoiser_baseline.py --config configs/baseline.yaml
```

**Architecture**: 2-level 2D U-Net, 3-slice input → 1-slice output  
**Metrics**: PSNR, SSIM  
**Output**: `outputs/train_runs/baseline_denoiser/`

#### 3D Reconstruction
```bash
python scripts/train_recon3d.py --config configs/recon3d.yaml
```

**Architecture**: 3D U-Net with channel attention gates  
**Loss**: Physics-guided (L1 + 3D gradient consistency + HU range penalty)  
**Metrics**: PSNR, SSIM  
**Output**: `outputs/train_runs/recon3d/`

#### Nodule Detection
```bash
python scripts/train_nodule_detector.py --config configs/nodule_detection.yaml
```

**Architecture**: 4-level 3D CNN with global average pooling  
**Metrics**: ROC AUC, sensitivity, specificity  
**Output**: `outputs/train_runs/nodule_detection/`

### 3. Sanity Check (Single Patient)

```bash
python scripts/sanity_check.py \
  --dataset_root manifest-1600709154662 \
  --metadata_csv manifest-1600709154662/metadata.csv \
  --xml_dir LIDC-XML-only/tcia-lidc-xml \
  --patient_id LIDC-IDRI-0001
```

## Validation Results

Small-scale local validation (15-20 patients, 2-3 epochs):

| Module | Metric | Result |
|--------|--------|--------|
| 2.5D Denoising | PSNR | 28.5 dB |
| | SSIM | 0.89 |
| 3D Reconstruction | PSNR | 17.8 → 19.9 dB |
| | SSIM | 0.42 → 0.52 |
| Nodule Detection | ROC AUC | **0.9402** |
| | Sensitivity | 92.7% |
| | Specificity | 84.5% |

*Note: Full-scale training with complete dataset expected to improve metrics by 20-40%*

## Documentation

- **[README.md](README.md)** (this file) - Project overview and quick start
- **[TRAINING_GUIDE.md](TRAINING_GUIDE.md)** - Detailed training instructions, hyperparameter tuning, troubleshooting

## Project Status

- ✅ Dataset validation and exploration
- ✅ Preprocessing pipeline (DICOM, HU, resampling, XML parsing)
- ✅ Patient-wise data splitting (no leakage)
- ✅ 2.5D denoising baseline
- ✅ 3D reconstruction with attention + physics loss
- ✅ Nodule detection with clinical metrics
- ⏳ Full-scale Colab training
- ⏳ Ablation studies
- ⏳ Final evaluation and reporting

## Citation

If you use LIDC-IDRI dataset, please cite:

```bibtex
@article{armato2011lidc,
  title={The lung image database consortium (LIDC) and image database resource initiative (IDRI): a completed reference database of lung nodules on CT scans},
  author={Armato III, Samuel G and McLennan, Geoffrey and Bidaut, Luc and McNitt-Gray, Michael F and Meyer, Charles R and Reeves, Anthony P and Zhao, Binsheng and Aberle, Denise R and Henschke, Claudia I and Hoffman, Eric A and others},
  journal={Medical physics},
  volume={38},
  number={2},
  pages={915--931},
  year={2011},
  publisher={Wiley Online Library}
}
```

## License

This project uses the LIDC-IDRI dataset which is publicly available under the Creative Commons Attribution 3.0 Unported License.

## Contact

**GitHub**: https://github.com/KailasVS666/AI_LUNG

For questions, issues, or collaboration inquiries, please open a GitHub issue.

---

**Last Updated**: February 2026  
**Status**: Core modules implemented, ready for full-scale training
