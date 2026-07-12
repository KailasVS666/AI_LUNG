import os
import sys
import json
import argparse
from pathlib import Path

# Add src to pythonpath
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ailung.pipeline import AILungPipeline

def main():
    parser = argparse.ArgumentParser(description="AI-LUNG End-to-End Inference PACS Wrapper")
    parser.add_argument("--dicom", type=str, required=True, help="Path to raw DICOM series directory")
    parser.add_argument("--xml", type=str, default=None, help="Optional path to XML guide annotation file")
    parser.add_argument("--output", type=str, default="outputs/predictions.json", help="Path to write prediction JSON results")
    parser.add_argument("--device", type=str, default="auto", help="Device to execute inference (auto/cpu/cuda)")
    args = parser.parse_args()

    # Paths to best checkpoints
    project_root = Path(__file__).parent.parent
    s1_ckpt = project_root / "outputs/train_runs/denoiser_25d/denoiser_best.pt"
    s2_ckpt = project_root / "outputs/train_runs/recon3d/recon3d_best.pt"
    s3_ckpt = project_root / "outputs/train_runs/nodule_detection/nodule_detector_best.pt"

    for path, name in [(s1_ckpt, "Stage 1 Denoiser"), (s2_ckpt, "Stage 2 Reconstructor"), (s3_ckpt, "Stage 3 Classifier")]:
        if not path.exists():
            raise FileNotFoundError(f"Error: {name} checkpoint not found at: {path}. Run training or restore outputs first.")

    # Initialize pipeline
    pipeline = AILungPipeline(
        s1_ckpt_path=str(s1_ckpt),
        s2_ckpt_path=str(s2_ckpt),
        s3_ckpt_path=str(s3_ckpt),
        device=args.device
    )

    # Execute end-to-end prediction
    results = pipeline.predict_volume(args.dicom, xml_path=args.xml)

    # Write output to JSON
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n=======================================================")
    print(f"Success! Predictions successfully saved to: {output_path}")
    print(f"Total Nodules Evaluated: {results['total_nodules_found']}")
    print(f"=======================================================")

if __name__ == "__main__":
    main()
