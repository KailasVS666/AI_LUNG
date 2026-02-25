# Training Guide

Detailed instructions for training all three AI_LUNG modules locally and on Google Colab.

## Prerequisites

- ✅ Dataset downloaded (LIDC-IDRI manifest + XML annotations)
- ✅ Environment set up (`pip install -e .`)
- ✅ Data splits generated (`python scripts/build_splits.py`)

## Local Training Workflow

### 1. Start Small (Smoke Test)

Always validate on a small subset first:

```bash
# Edit config to limit cases
# configs/*.yaml:
#   max_cases_train: 20
#   max_cases_val: 5
#   epochs: 2

# Run training
python scripts/train_denoiser_baseline.py --config configs/baseline.yaml
python scripts/train_recon3d.py --config configs/recon3d.yaml
python scripts/train_nodule_detector.py --config configs/nodule_detection.yaml
```

**Expected time**: 1-3 minutes per module on CPU

### 2. Monitor Training

Each script saves outputs to `outputs/train_runs/<module_name>/`:

- `*_last.pt` - Final checkpoint
- `*_best.pt` - Best validation checkpoint
- `history.json` - Metrics per epoch
- `training_curves.png` - Loss/metric plots
- `epoch_*.png` - Visual preview images (denoising/recon only)

**Watch for**:
- Decreasing training loss
- Stable/improving validation metrics
- Visual previews showing qualitative improvement

### 3. Debug Common Issues

#### Out of Memory (OOM)
```yaml
# Reduce batch size
batch_size: 2  # Default: 4

# Reduce patch size (3D modules)
patch_size: [24, 64, 64]  # Default: [32, 96, 96]
```

#### Slow Training
```python
# Increase DataLoader workers (if you have multiple CPU cores)
train_loader = DataLoader(..., num_workers=4)  # Default: 0
```

#### NaN Loss
- Check learning rate (too high → divergence)
- Verify HU normalization and data preprocessing
- Inspect a few samples manually with `scripts/sanity_check.py`

#### Imbalanced Detection Classes
```yaml
# Adjust nodule malignancy threshold
min_malignancy: 1  # Include all nodules (1-5)

# Adjust negative sampling ratio
negatives_per_positive: 3  # Increase for more negatives
```

## Google Colab Training

### Setup Colab Notebook

```python
# 1. Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. Clone repository
!git clone https://github.com/KailasVS666/AI_LUNG.git
%cd AI_LUNG

# 3. Install dependencies
!pip install -q -r requirements.txt
!pip install -q -e .

# 4. Create symlinks to dataset in Drive
!ln -s /content/drive/MyDrive/LIDC-IDRI/manifest-1600709154662 manifest-1600709154662
!ln -s /content/drive/MyDrive/LIDC-IDRI/LIDC-XML-only LIDC-XML-only
```

### Update Configs for Production

Create production-scale configs in Colab:

```python
# Edit configs programmatically
import yaml

for config_name in ['baseline.yaml', 'recon3d.yaml', 'nodule_detection.yaml']:
    with open(f'configs/{config_name}', 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Remove case limits
    cfg['max_cases_train'] = None
    cfg['max_cases_val'] = None
    
    # Increase epochs
    cfg['epochs'] = 30
    
    # Increase batch size for GPU
    cfg['batch_size'] = 8  # Adjust based on GPU memory
    
    with open(f'configs/{config_name}_colab.yaml', 'w') as f:
        yaml.dump(cfg, f)
```

### Run Training

```python
# Verify GPU availability
import torch
print(f"GPU available: {torch.cuda.is_available()}")
print(f"GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

# Run training scripts
!python scripts/train_denoiser_baseline.py --config configs/baseline_colab.yaml
!python scripts/train_recon3d.py --config configs/recon3d_colab.yaml
!python scripts/train_nodule_detector.py --config configs/nodule_detection_colab.yaml
```

### Monitor Long Training

For multi-hour training, add checkpointing:

```python
# Already implemented in scripts! Just watch outputs:
!ls -lh outputs/train_runs/*/
```

To resume interrupted training, load checkpoint manually:

```python
# In training script, add before training loop:
checkpoint_path = Path(output_dir) / "recon3d_last.pt"
if checkpoint_path.exists():
    model.load_state_dict(torch.load(checkpoint_path))
    print(f"Resumed from {checkpoint_path}")
```

## Training Schedule

### Phase 1: Quick Iteration (Local, 1-2 hours)
- 20 train / 5 val patients
- 2-3 epochs
- Goal: Verify code works end-to-end

### Phase 2: Medium Scale (Colab, 4-8 hours)
- 200 train / 50 val patients
- 10-15 epochs
- Goal: Tune hyperparameters, assess overfitting

### Phase 3: Full Training (Colab, 12-24 hours)
- All 713 train / 153 val patients
- 30-50 epochs
- Goal: Production-ready models

## Hyperparameter Tuning

### Learning Rate
```yaml
# Try these values (one at a time)
learning_rate: 0.0003  # Default
learning_rate: 0.0001  # Conservative
learning_rate: 0.00005 # Very conservative
```

**Signs of bad LR**:
- Too high: Loss fluctuates wildly, NaN values
- Too low: Loss decreases very slowly

### Batch Size
```yaml
# Larger = more stable gradients but more memory
batch_size: 4   # CPU-safe
batch_size: 8   # T4 GPU
batch_size: 16  # A100 GPU
```

### Regularization

**Dropout** (detection only):
```python
# In models.py NoduleClassifier3D
nn.Dropout(0.3)  # Default
nn.Dropout(0.5)  # Stronger
```

**Weight Decay**:
```python
# In training scripts
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
```

## Evaluation Best Practices

### 1. Train/Val/Test Separation

Current splits are **patient-wise** (prevents data leakage):
- Train: 713 series, 707 unique patients
- Val: 153 series, 151 unique patients
- Test: 153 series, 152 unique patients

**Never** train on val or test sets. Only use test for final evaluation.

### 2. Validation Metrics

**Denoising**:
- PSNR ≥ 28 dB (good), ≥ 32 dB (excellent)
- SSIM ≥ 0.85 (good), ≥ 0.92 (excellent)

**Reconstruction**:
- PSNR improvement ≥ 2 dB
- SSIM improvement ≥ 0.1
- Visual: Edges sharper, noise reduced

**Detection**:
- ROC AUC ≥ 0.90 (clinical baseline)
- Sensitivity ≥ 0.85 at specificity 0.90
- False positives per scan < 4

### 3. Test Set Evaluation

After finalizing hyperparameters on val set:

```python
# Modify training script to use test split
splits = load_split(split_json)
test_entries = splits["test"]

# Create test dataset (no training!)
test_ds = NoduleDetectionDataset(test_entries, ...)
test_loader = DataLoader(test_ds, batch_size=8, shuffle=False)

# Evaluate
model.load_state_dict(torch.load("nodule_detector_best.pt"))
test_metrics = validate_epoch(model, test_loader, device)
print(f"Test AUC: {test_metrics['auc']:.4f}")
```

## Saving and Sharing Checkpoints

### Export Final Models

```bash
# Copy best checkpoints to organized folder
mkdir -p final_models
cp outputs/train_runs/baseline_denoiser/denoiser_best.pt final_models/
cp outputs/train_runs/recon3d/recon3d_best.pt final_models/
cp outputs/train_runs/nodule_detection/nodule_detector_best.pt final_models/
```

### Model Card Template

Create `final_models/MODEL_CARD.md`:

```markdown
# AI_LUNG Model Card

## Model Details
- **Date**: February 2026
- **Version**: 1.0
- **Training Data**: LIDC-IDRI (713 train, 153 val, 153 test)
- **Framework**: PyTorch 2.2+

## Performance

| Module | Metric | Train | Val | Test |
|--------|--------|-------|-----|------|
| Denoising | PSNR | XX.X | XX.X | XX.X |
| Reconstruction | AUC | X.XXX | X.XXX | X.XXX |
| Detection | Sensitivity | X.XX | X.XX | X.XX |

## Usage
```python
from ailung.models import NoduleClassifier3D
model = NoduleClassifier3D()
model.load_state_dict(torch.load("nodule_detector_best.pt"))
model.eval()
```

## Limitations
- Trained only on LIDC-IDRI (North American population)
- Performance may degrade on non-CT modalities
- Not validated in clinical settings
```

## Troubleshooting

### "RuntimeError: CUDA out of memory"
```yaml
# Reduce batch size
batch_size: 2

# Reduce patch size
patch_size: [24, 48, 48]
```

### "FileNotFoundError: No such file"
```bash
# Verify paths in config match your system
# Use absolute paths if relative paths fail
dataset_root: "D:/AI_LUNG/manifest-1600709154662"  # Windows
dataset_root: "/content/AI_LUNG/manifest-1600709154662"  # Colab
```

### "ValueError: No training samples created"
```bash
# Check dataset paths
ls manifest-1600709154662/LIDC-IDRI/LIDC-IDRI-0001

# Run sanity check
python scripts/sanity_check.py --patient_id LIDC-IDRI-0001
```

### "UndefinedMetricWarning: ROC AUC not defined"
- Validation set has only one class (all positives or all negatives)
- Lower `min_malignancy` threshold to get more positive samples
- Increase `max_cases_val` to get more diverse samples

## Get Help

1. Check [README.md](README.md) for setup
2. Review [outputs/train_runs/*/history.json](outputs/train_runs/) for metrics
3. Inspect code in [src/ailung/](src/ailung/)
4. Open GitHub issue with:
   - Command you ran
   - Full error traceback
   - Config file contents
   - System info (OS, Python version, GPU)

---

**Happy Training!** 🚀
