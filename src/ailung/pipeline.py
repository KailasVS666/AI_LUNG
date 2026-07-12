import os
import torch
import numpy as np
from pathlib import Path
import scipy.ndimage as ndimage
import torch.nn.functional as F

from ailung.models import Denoise25DUNet, Recon3DUNet, NoduleDetector3D
from ailung.preprocess import build_volume, hu_clip_normalize, apply_clahe, resample_isotropic
from ailung.annotations import parse_lidc_xml, build_nodule_candidates

class AILungPipeline:
    def __init__(self, s1_ckpt_path: str, s2_ckpt_path: str, s3_ckpt_path: str, device: str = "auto") -> None:
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"Initializing AILungPipeline on device: {self.device}", flush=True)

        # 1. Load Stage 1 Denoising Model
        self.model_s1 = Denoise25DUNet(in_channels=9, out_channels=1).to(self.device)
        s1_ckpt = torch.load(s1_ckpt_path, map_location=self.device)
        self.model_s1.load_state_dict(s1_ckpt["model_state_dict"])
        self.model_s1.eval()
        print("  * Loaded Stage 1 denoiser.", flush=True)

        # 2. Load Stage 2 Reconstruction Model
        self.model_s2 = Recon3DUNet(in_channels=1, base_channels=16, out_channels=1).to(self.device)
        s2_ckpt = torch.load(s2_ckpt_path, map_location=self.device)
        self.model_s2.load_state_dict(s2_ckpt["model_state_dict"])
        self.model_s2.eval()
        print("  * Loaded Stage 2 reconstructor.", flush=True)

        # 3. Load Stage 3 Nodule Detection Model
        s3_ckpt = torch.load(s3_ckpt_path, map_location=self.device)
        state_dict_s3 = s3_ckpt["model_state_dict"] if isinstance(s3_ckpt, dict) and "model_state_dict" in s3_ckpt else s3_ckpt
        
        # Dynamically determine classes from checkpoint shape
        num_classes = state_dict_s3["classifier.4.weight"].shape[0]
        self.model_s3 = NoduleDetector3D(in_channels=1, base_channels=16, num_classes=num_classes).to(self.device)
        self.model_s3.load_state_dict(state_dict_s3)
        self.model_s3.eval()
        print(f"  * Loaded Stage 3 classifier ({num_classes} classes).", flush=True)

    def predict_volume(self, dicom_dir: str, xml_path: str | None = None) -> dict:
        """
        Run the end-to-end pipeline:
        1. Load & normalize DICOM slices.
        2. Denoise slice-by-slice (Stage 1).
        3. Reconstruct isotropic 3D volume (Stage 2).
        4. Detect or parse candidate nodule centroids.
        5. Crop candidate patches and classify malignancy risk (Stage 3).
        """
        print(f"Loading DICOM slices from: {dicom_dir}", flush=True)
        volume_hu, spacing = build_volume(dicom_dir)
        print(f"Raw volume shape: {volume_hu.shape}, Spacing: {spacing}", flush=True)

        # Normalization and enhancement
        volume_nd = hu_clip_normalize(volume_hu)
        volume_nd = apply_clahe(volume_nd)

        # Stage 1: Denoise slice-by-slice
        print("Running Stage 1 (Denoising)...", flush=True)
        context = 4
        z_dim, h_dim, w_dim = volume_nd.shape
        denoised = np.zeros_like(volume_nd)
        
        with torch.no_grad():
            for z in range(context, z_dim - context):
                stack = volume_nd[z - context : z + context + 1]
                x = torch.from_numpy(stack).float().unsqueeze(0).to(self.device)
                pred = self.model_s1(x)
                denoised[z] = pred.squeeze(0).squeeze(0).cpu().numpy()
        
        # Fill boundaries
        denoised[:context] = volume_nd[:context]
        denoised[-context:] = volume_nd[-context:]

        # Resample to 1.0mm isotropic resolution
        print("Resampling denoised volume to 1.0mm isotropic resolution...", flush=True)
        volume_iso, iso_spacing = resample_isotropic(denoised, spacing)
        print(f"Resampled volume shape: {volume_iso.shape}", flush=True)

        # Stage 2: Volumetric Reconstruction in blocks to avoid out-of-memory errors
        print("Running Stage 2 (3D Reconstruction)...", flush=True)
        reconstructed = np.zeros_like(volume_iso)
        z_iso, h_iso, w_iso = volume_iso.shape
        
        # Process in chunks of 32 slices along the Z axis
        chunk_size = 32
        stride = 24  # overlapping boundary stride
        
        with torch.no_grad():
            for z in range(0, z_iso, stride):
                z_start = min(z, max(0, z_iso - chunk_size))
                z_end = z_start + chunk_size
                
                # Crop block (we crop the center H x W to avoid memory overload)
                # Recon3DUNet requires inputs multiple of 16. The isotropic slice shape (e.g. 360x360) is padded or processed in crops.
                # Let's crop into 128x128 windows to process safely on CPU/low-memory GPU
                block = volume_iso[z_start:z_end] # (32, H, W)
                
                h_crop = 128
                w_crop = 128
                
                for y in range(0, h_iso, 96):
                    y_start = min(y, max(0, h_iso - h_crop))
                    y_end = y_start + h_crop
                    
                    for x in range(0, w_iso, 96):
                        x_start = min(x, max(0, w_iso - w_crop))
                        x_end = x_start + w_crop
                        
                        patch_3d = block[:, y_start:y_end, x_start:x_end]
                        x_in = torch.from_numpy(patch_3d).float().unsqueeze(0).unsqueeze(0).to(self.device)
                        
                        pred_3d = self.model_s2(x_in).squeeze(0).squeeze(0).cpu().numpy()
                        
                        # Accumulate reconstruction output (use max to merge overlaps)
                        reconstructed[z_start:z_end, y_start:y_end, x_start:x_end] = np.maximum(
                            reconstructed[z_start:z_end, y_start:y_end, x_start:x_end],
                            pred_3d
                        )

        # Load raw slices Z positions to map absolute coordinates
        z_positions = []
        dcm_files = sorted(Path(dicom_dir).glob("*.dcm"))
        if dcm_files:
            import pydicom
            for f in dcm_files:
                try:
                    meta = pydicom.dcmread(str(f), stop_before_pixels=True)
                    z_pos = float(getattr(meta, "ImagePositionPatient", [0, 0, 0])[2])
                    z_positions.append(z_pos)
                except: pass
            z_positions.sort()

        # Stage 3: Candidate nodule coordinates search
        candidates = []
        if xml_path and os.path.exists(xml_path):
            print(f"Parsing guide XML annotations from: {xml_path}", flush=True)
            nodule_candidates = build_nodule_candidates(xml_path, spacing)
            for cand in nodule_candidates:
                z_mm, y_mm, x_mm = cand["centroid_3d"]
                
                # 1. Convert physical coordinates to raw voxel indices
                if z_positions:
                    z_raw = min(range(len(z_positions)), key=lambda i: abs(z_positions[i] - z_mm))
                else:
                    z_raw = int(z_mm)
                y_raw = int(y_mm / spacing[1])
                x_raw = int(x_mm / spacing[2])
                
                # 2. Convert raw indices to isotropic scale index coordinates
                coords_iso = (
                    int(z_raw * spacing[0]),
                    int(y_raw * spacing[1]),
                    int(x_raw * spacing[2])
                )
                candidates.append({
                    "id": cand["nodule_id"],
                    "centroid_iso": coords_iso,
                    "source": "XML"
                })
        else:
            print("No XML guide provided. Running automated connected-component blob detection...", flush=True)
            # Threshold density matching soft tissue nodule range in lung window
            mask = reconstructed > 0.65
            labeled_mask, num_features = ndimage.label(mask)
            print(f"  Found {num_features} connected volumetric regions.", flush=True)
            
            # Filter regions by typical nodule volume size (diameter 3mm to 30mm)
            # at 1mm spacing, 3mm diameter sphere = ~14 voxels, 30mm diameter = ~14,000 voxels
            min_size = 14
            max_size = 14000
            
            sizes = ndimage.sum(mask, labeled_mask, range(1, num_features + 1))
            for idx, size in enumerate(sizes):
                if min_size <= size <= max_size:
                    centroid = ndimage.center_of_mass(mask, labeled_mask, idx + 1)
                    # Convert to integer coordinates
                    centroid_iso = (int(centroid[0]), int(centroid[1]), int(centroid[2]))
                    candidates.append({
                        "id": f"auto_{len(candidates)+1}",
                        "centroid_iso": centroid_iso,
                        "source": "detection",
                        "size_voxels": int(size)
                    })

        print(f"Total candidates to evaluate: {len(candidates)}", flush=True)

        # Stage 4: Patch classification
        predictions = []
        for cand in candidates:
            z_c, y_c, x_c = cand["centroid_iso"]
            
            # Ensure crop box stays inside isotropic reconstructed volume boundaries
            z_start = max(0, min(z_iso - 32, z_c - 16))
            y_start = max(0, min(h_iso - 64, y_c - 32))
            x_start = max(0, min(w_iso - 64, x_c - 32))
            
            patch = reconstructed[z_start:z_start+32, y_start:y_start+64, x_start:x_start+64]
            
            x_in = torch.from_numpy(patch).float().unsqueeze(0).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.model_s3(x_in)
                probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
                
            pred_class = int(np.argmax(probs))
            
            if self.model_s3.classifier[4].out_features == 6:
                # 6-class mode
                if pred_class == 5:
                    classification = "Background"
                    malignancy_level = 0
                    malignancy_score = 0.0
                else:
                    classification = "Nodule"
                    malignancy_level = pred_class + 1
                    malignancy_score = float(1.0 - probs[5]) # probability of not background
                
                predictions.append({
                    "nodule_id": cand["id"],
                    "source": cand["source"],
                    "coordinates_iso": {
                        "z": z_c,
                        "y": y_c,
                        "x": x_c
                    },
                    "classification": classification,
                    "malignancy_level": malignancy_level,
                    "malignancy_score": malignancy_score,
                    "class_probabilities": [float(p) for p in probs],
                    "estimated_volume_voxels": cand.get("size_voxels", None)
                })
            else:
                # Binary mode
                malignancy_score = float(probs[1])
                predictions.append({
                    "nodule_id": cand["id"],
                    "source": cand["source"],
                    "coordinates_iso": {
                        "z": z_c,
                        "y": y_c,
                        "x": x_c
                    },
                    "malignancy_score": malignancy_score,
                    "benign_score": float(probs[0]),
                    "classification": "Malignant" if pred_class == 1 else "Benign",
                    "estimated_volume_voxels": cand.get("size_voxels", None)
                })

        print("Inference pipeline complete.", flush=True)
        return {
            "series_uid": os.path.basename(dicom_dir),
            "dimensions_iso": [z_iso, h_iso, w_iso],
            "total_nodules_found": len(predictions),
            "predictions": predictions
        }
