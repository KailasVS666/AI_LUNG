import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to pythonpath
sys.path.insert(0, "d:/AI_LUNG/src")
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

from ailung.models import Denoise25DUNet, Recon3DUNet, NoduleDetector3D
from ailung.preprocess import build_volume, hu_clip_normalize, apply_clahe, simulate_low_dose_fast

class GradCAM3D:
    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        self.forward_hook = self.target_layer.register_forward_hook(self._save_activation)
        self.backward_hook = self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, x: torch.Tensor, class_idx: int | None = None) -> torch.Tensor:
        self.model.zero_grad()
        logits = self.model(x)
        if class_idx is None:
            class_idx = torch.argmax(logits, dim=1).item()
        
        loss = logits[0, class_idx]
        loss.backward()
        
        # Calculate weights: average gradients across spatial dimensions (2, 3, 4)
        weights = torch.mean(self.gradients, dim=(2, 3, 4), keepdim=True) # (1, channels, 1, 1, 1)
        
        # Weighted combination of activations
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True) # (1, 1, D_feat, H_feat, W_feat)
        
        # Relu to only show positive activations
        cam = F.relu(cam)
        
        # Upscale to original spatial dimensions
        cam = F.interpolate(cam, size=x.shape[2:], mode="trilinear", align_corners=False) # (1, 1, D, H, W)
        
        # Normalize
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)
            
        return cam.squeeze() # (D, H, W)

    def release(self):
        self.forward_hook.remove()
        self.backward_hook.remove()

def main():
    device = torch.device("cpu")
    print("Loading Stage 3 model and checkpoints...", flush=True)

    # 1. Load pipeline models to get reconstructed nodule
    s1_path = Path("d:/AI_LUNG/outputs/train_runs/denoiser_25d/denoiser_best.pt")
    model_s1 = Denoise25DUNet(in_channels=9, out_channels=1).to(device)
    ckpt_s1 = torch.load(s1_path, map_location=device)
    model_s1.load_state_dict(ckpt_s1["model_state_dict"])
    model_s1.eval()

    s2_path = Path("d:/AI_LUNG/outputs/train_runs/recon3d/recon3d_best.pt")
    model_s2 = Recon3DUNet(in_channels=1, base_channels=16, out_channels=1).to(device)
    ckpt_s2 = torch.load(s2_path, map_location=device)
    model_s2.load_state_dict(ckpt_s2["model_state_dict"])
    model_s2.eval()

    s3_path = Path("d:/AI_LUNG/outputs/train_runs/nodule_detection/nodule_detector_best.pt")
    model_s3 = NoduleDetector3D(in_channels=1, base_channels=16, num_classes=2).to(device)
    ckpt_s3 = torch.load(s3_path, map_location=device)
    model_s3.load_state_dict(ckpt_s3)
    model_s3.eval()

    # 2. Extract a reconstructed nodule patch
    patient_dir = "d:/AI_LUNG/manifest-1600709154662/LIDC-IDRI/LIDC-IDRI-0001"
    dicom_dirs = [d for d in Path(patient_dir).rglob("*") if d.is_dir() and list(d.glob("*.dcm"))]
    series_path = str(dicom_dirs[0])
    
    volume_hu, _ = build_volume(series_path)
    volume_nd = hu_clip_normalize(volume_hu, hu_min=-1000, hu_max=400)
    volume_nd = apply_clahe(volume_nd)
    
    # Simulate low dose scan (noisy input) and run pipeline
    volume_ld = simulate_low_dose_fast(volume_nd, i0=1e5, seed=42)
    center_z = volume_ld.shape[0] // 2
    
    # Stage 1 Denoise
    context = 4
    denoised_volume = np.zeros_like(volume_nd)
    with torch.no_grad():
        for z in range(context, volume_ld.shape[0] - context):
            stack = volume_ld[z - context : z + context + 1]
            x = torch.from_numpy(stack).float().unsqueeze(0).to(device)
            pred = model_s1(x)
            denoised_volume[z] = pred.squeeze().numpy()
            
    # Stage 2 Reconstruction
    sub_volume_ld = denoised_volume[center_z - 16 : center_z + 16]
    h, w = sub_volume_ld.shape[1], sub_volume_ld.shape[2]
    crop_h = slice(h//2 - 64, h//2 + 64)
    crop_w = slice(w//2 - 64, w//2 + 64)
    sub_volume_ld_cropped = sub_volume_ld[:, crop_h, crop_w]
    
    x2 = torch.from_numpy(sub_volume_ld_cropped).float().unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        reconstructed_volume = model_s2(x2).squeeze().numpy()

    # Stage 3 Crop Patch
    patch = reconstructed_volume[:, 32:96, 32:96] # (32, 64, 64)
    x3 = torch.from_numpy(patch).float().unsqueeze(0).unsqueeze(0).to(device)
    x3.requires_grad = True

    # 3. Setup and run 3D Grad-CAM
    # Hook the final CBAM attention module of NoduleDetector3D
    cam_extractor = GradCAM3D(model_s3, model_s3.cbam)
    
    # Target class: 1 (Malignant) to see what clinical malignancy indicators it spotted
    print("Extracting 3D Grad-CAM activations for Malignant class...", flush=True)
    heatmap = cam_extractor(x3, class_idx=1)
    cam_extractor.release()
    
    heatmap_np = heatmap.cpu().numpy() # (32, 64, 64)
    patch_np = patch # (32, 64, 64)

    # 4. Save visualization slices (Axial, Coronal, Sagittal through center)
    mid_z, mid_y, mid_x = 16, 32, 32
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    
    # Row 1: Reconstruction
    axes[0, 0].imshow(patch_np[mid_z], cmap="gray"); axes[0, 0].set_title("Axial Slice (Reconstruction)"); axes[0, 0].axis("off")
    axes[0, 1].imshow(patch_np[:, mid_y], cmap="gray"); axes[0, 1].set_title("Coronal Slice (Reconstruction)"); axes[0, 1].axis("off")
    axes[0, 2].imshow(patch_np[:, :, mid_x], cmap="gray"); axes[0, 2].set_title("Sagittal Slice (Reconstruction)"); axes[0, 2].axis("off")
    
    # Row 2: Overlays
    def show_overlay(ax, base, heat, title):
        ax.imshow(base, cmap="gray")
        # Color overlay with transparency
        im = ax.imshow(heat, cmap="jet", alpha=0.45)
        ax.set_title(title)
        ax.axis("off")
        return im

    show_overlay(axes[1, 0], patch_np[mid_z], heatmap_np[mid_z], "Axial (3D Grad-CAM Overlay)")
    show_overlay(axes[1, 1], patch_np[:, mid_y], heatmap_np[:, mid_y], "Coronal (3D Grad-CAM Overlay)")
    im = show_overlay(axes[1, 2], patch_np[:, :, mid_x], heatmap_np[:, :, mid_x], "Sagittal (3D Grad-CAM Overlay)")
    
    # Add colorbar
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="Attention Activation Strength")
    
    plt.suptitle("Clinical 3D Grad-CAM Visual Explainability\nTarget Class: Malignant", fontsize=16, fontweight="bold", y=0.96)
    
    out_path = Path("C:/Users/sharj/.gemini/antigravity-ide/brain/3cda39e9-8af5-4519-b054-9286b0221fc4/clinical_gradcam.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Success! Explainability plot saved to: {out_path}", flush=True)

if __name__ == "__main__":
    main()
