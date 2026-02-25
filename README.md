# AI-LUNG (Local VS Code + Colab Workflow)

This repository contains local preprocessing utilities for LIDC-IDRI that are lightweight enough to run in VS Code on a small sample and portable to Google Colab for full training.

## Quick start (local)

1. Create and activate a Python env.
2. Install deps:

```bash
pip install -r requirements.txt
pip install -e .
```

3. Run a quick sanity check on one CT series:

```bash
python scripts/sanity_check.py
```

## Recommended workflow

- Local VS Code:
  - Write/debug code
  - Parse DICOM/XML
  - Run sanity checks on 1-20 cases
- Colab:
  - Full preprocessing on all scans
  - Training/fine-tuning
  - Evaluation and checkpointing

## Build patient-wise splits

```bash
python scripts/build_splits.py
```

This writes `d:/AI_LUNG/outputs/splits/patient_split.json`.

## Train 2.5D denoiser baseline

```bash
python scripts/train_denoiser_baseline.py --config d:/AI_LUNG/configs/baseline.yaml
```

Outputs are saved under `d:/AI_LUNG/outputs/train_runs/baseline_25d`.

Training now records:
- `train_loss`, `val_loss`, `val_psnr`, `val_ssim` in `history.json`
- Per-epoch preview PNGs under `previews/` (input center slice, target, prediction)

## Train 3D reconstruction model (attention + physics-guided loss)

```bash
python scripts/train_recon3d.py --config d:/AI_LUNG/configs/recon3d.yaml
```

Outputs are saved under `d:/AI_LUNG/outputs/train_runs/recon3d`.

Training records:
- `train_loss`, `val_loss`, `val_psnr`, `val_ssim`
- Physics loss components: `val_loss_recon`, `val_loss_grad`, `val_loss_range`
- Per-epoch 3-panel preview PNGs under `previews/`

## Current modules

- `ailung.dataset`: discovers CT series from `metadata.csv`
- `ailung.preprocess`: DICOM -> HU volume -> normalize -> isotropic resample
- `ailung.annotations`: parses LIDC XML nodules and ROI contour points
- `ailung.splits`: patient-wise train/val/test split creation and loading
- `ailung.torch_dataset`: baseline 2.5D denoiser dataset class
- `ailung.models`: small U-Net style 2.5D denoiser baseline

Additional reconstruction components:
- `ailung.models.Recon3DAttentionUNetSmall`
- `ailung.models.PhysicsGuidedReconLoss`
- `ailung.torch_dataset.LIDCRecon3DPatchDataset`
