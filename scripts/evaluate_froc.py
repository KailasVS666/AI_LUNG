import os
import sys
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial.distance import euclidean

# Add src to pythonpath
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ailung.pipeline import AILungPipeline
from ailung.annotations import build_nodule_candidates, build_series_to_xml_mapping
from ailung.preprocess import build_volume
from ailung.splits import load_split

def main():
    parser = argparse.ArgumentParser(description="AI-LUNG Whole-Lung FROC Validation")
    parser.add_argument("--config", type=str, required=True, help="Path to config file containing splits/metadata")
    parser.add_argument("--max-cases", type=int, default=5, help="Number of patient cases to evaluate")
    parser.add_argument("--dist-tol", type=float, default=15.0, help="Matching distance tolerance in mm")
    parser.add_argument("--device", type=str, default="auto", help="Device to execute inference (auto/cpu/cuda)")
    parser.add_argument("--output-img", type=str, default="outputs/froc_curve.png", help="Path to save FROC curve image")
    args = parser.parse_args()

    import yaml
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 1. Initialize Pipeline
    project_root = Path(__file__).parent.parent
    s1_ckpt = project_root / "outputs/train_runs/denoiser_25d/denoiser_best.pt"
    s2_ckpt = project_root / "outputs/train_runs/recon3d/recon3d_best.pt"
    # Load 2-class or 6-class classifier
    s3_ckpt = project_root / "outputs/train_runs/nodule_detection/nodule_detector_best.pt"

    pipeline = AILungPipeline(
        s1_ckpt_path=str(s1_ckpt),
        s2_ckpt_path=str(s2_ckpt),
        s3_ckpt_path=str(s3_ckpt),
        device=args.device
    )

    # 2. Get list of evaluation series from val split
    splits = load_split(cfg["split_json"] if "split_json" in cfg else cfg["data"]["splits_path"])
    val_entries = splits["val"][:args.max_cases]
    xml_dir = Path(cfg["xml_dir"])
    
    print("Building series UID -> XML mapping...")
    series_to_xml = build_series_to_xml_mapping(xml_dir)

    all_true_nodules = [] # List of True nodules: {"centroid_3d": (z, y, x), "series_uid": str, "detected": False}
    all_detections = []   # List of detected candidates: {"centroid_3d": (z, y, x), "score": float, "series_uid": str}
    
    evaluated_scans = 0

    for idx, item in enumerate(val_entries):
        dicom_dir = str(item["file_location"])
        series_uid = item["series_uid"]
        
        xml_path = series_to_xml.get(series_uid)
        if not xml_path or not xml_path.exists():
            print(f"Skipping series {series_uid}: XML annotations not found.")
            continue

        print(f"\nEvaluating Scan {idx+1}/{len(val_entries)}: {series_uid}")
        evaluated_scans += 1

        # A. Read spacing and raw slice Z coordinates
        _, spacing = build_volume(dicom_dir)
        dcm_files = sorted(Path(dicom_dir).glob("*.dcm"))
        z_positions = []
        if dcm_files:
            import pydicom
            for f in dcm_files:
                try:
                    meta = pydicom.dcmread(str(f), stop_before_pixels=True)
                    z_pos = float(getattr(meta, "ImagePositionPatient", [0, 0, 0])[2])
                    z_positions.append(z_pos)
                except: pass
            z_positions.sort()

        # B. Parse true nodules in absolute mm coordinates
        true_candidates = build_nodule_candidates(xml_path, spacing, min_malignancy=1)
        for tc in true_candidates:
            all_true_nodules.append({
                "centroid_3d": tc["centroid_3d"], # (z_mm, y_mm, x_mm)
                "series_uid": series_uid,
                "detected": False
            })
            print(f"  True Nodule Centroid (mm): {tc['centroid_3d']}")

        # C. Run automated detector on full volume
        results = pipeline.predict_volume(dicom_dir)
        for pred in results["predictions"]:
            z_det, y_det, x_det = pred["coordinates_iso"]["z"], pred["coordinates_iso"]["y"], pred["coordinates_iso"]["x"]
            
            # Map detected isotropic index coordinates back to absolute physical coordinates (mm)
            z_raw_det = int(z_det / spacing[0])
            if z_positions and z_raw_det < len(z_positions):
                z_mm_det = z_positions[z_raw_det]
            else:
                z_mm_det = z_raw_det * spacing[0]
                
            y_mm_det = int(y_det / spacing[1]) * spacing[1]
            x_mm_det = int(x_det / spacing[2]) * spacing[2]
            
            # Note: malignancy_score is 1.0 - probs[5] if 6-class, else probs[1]
            # Since predict_volume handles this internally, we can read malignancy_score directly
            score = pred["malignancy_score"]
            
            all_detections.append({
                "centroid_3d": (z_mm_det, y_mm_det, x_mm_det),
                "score": score,
                "series_uid": series_uid
            })

    print(f"\nEvaluation summary:")
    print(f"  Scans evaluated: {evaluated_scans}")
    print(f"  Total true nodules: {len(all_true_nodules)}")
    print(f"  Total automated proposals: {len(all_detections)}")

    if not all_true_nodules or not all_detections:
        print("Error: No nodules or proposals found. Cannot compute FROC curve.")
        return

    # 3. Sweep thresholds to compute FROC curve
    thresholds = sorted(list(set([d["score"] for d in all_detections])), reverse=True)
    # Add boundary values
    thresholds = [1.05] + thresholds + [0.0]

    sensitivities = []
    fps_per_scan = []

    for th in thresholds:
        # Filter detections by score threshold
        active_dets = [d for d in all_detections if d["score"] >= th]
        
        # Reset detected flags
        for tn in all_true_nodules:
            tn["detected"] = False

        tp_count = 0
        fp_count = 0

        # Group detections by series to match within same scan
        for det in active_dets:
            matched = False
            for tn in all_true_nodules:
                if tn["series_uid"] == det["series_uid"]:
                    dist = euclidean(det["centroid_3d"], tn["centroid_3d"])
                    if dist <= args.dist_tol:
                        if not tn["detected"]:
                            tn["detected"] = True
                            tp_count += 1
                        matched = True
            
            if not matched:
                fp_count += 1

        # Calculate metrics
        sens = sum(1 for tn in all_true_nodules if tn["detected"]) / len(all_true_nodules)
        fp_rate = fp_count / evaluated_scans

        sensitivities.append(sens)
        fps_per_scan.append(fp_rate)

    # 4. Plot FROC Curve
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fps_per_scan, sensitivities, "b-", linewidth=2.5, label="AI-LUNG Detector")
    
    # Highlight standard clinical FP points: 0.25, 0.5, 1.0, 2.0, 4.0, 8.0
    fp_points = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
    sens_at_points = []
    for fp_pt in fp_points:
        # Interpolate sensitivity at specific FP rate
        idx = np.argmin(np.abs(np.array(fps_per_scan) - fp_pt))
        sens_at_points.append(sensitivities[idx])
        ax.plot(fps_per_scan[idx], sensitivities[idx], "ro")
        ax.text(fps_per_scan[idx] + 0.1, sensitivities[idx] - 0.03, f"{sensitivities[idx]*100:.1f}%", fontsize=9)

    mean_froc = np.mean(sens_at_points)

    ax.set_title(f"Whole-Lung FROC Curve (Mean Clinical Sens: {mean_froc*100:.1f}%)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Average False Positives per Scan", fontsize=12)
    ax.set_ylabel("Sensitivity (Fraction of Nodules Detected)", fontsize=12)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1.05)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="lower right")

    # Save visualization
    output_path = Path(args.output_img)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"\n=======================================================")
    print(f"Success! FROC evaluation complete.")
    print(f"FROC Curve Image saved to: {output_path}")
    print(f"Mean clinical sensitivity (0.25 to 8.0 FPs): {mean_froc*100:.2f}%")
    print(f"=======================================================")

if __name__ == "__main__":
    main()
