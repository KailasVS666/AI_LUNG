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

## Current modules

- `ailung.dataset`: discovers CT series from `metadata.csv`
- `ailung.preprocess`: DICOM -> HU volume -> normalize -> isotropic resample
- `ailung.annotations`: parses LIDC XML nodules and ROI contour points
