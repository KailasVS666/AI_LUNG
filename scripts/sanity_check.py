from __future__ import annotations

from pathlib import Path
import argparse

from ailung.annotations import parse_lidc_xml
from ailung.dataset import discover_ct_series
from ailung.preprocess import build_volume, hu_clip_normalize, resample_isotropic


def main() -> None:
    parser = argparse.ArgumentParser(description="AI-LUNG local sanity check")
    parser.add_argument("--dataset-root", type=str, default="d:/AI_LUNG/manifest-1600709154662")
    parser.add_argument("--metadata-csv", type=str, default="d:/AI_LUNG/manifest-1600709154662/metadata.csv")
    parser.add_argument("--sample-index", type=int, default=0)
    args = parser.parse_args()

    series = discover_ct_series(args.dataset_root, args.metadata_csv)
    if not series:
        raise RuntimeError("No CT series found from metadata")

    selected = series[args.sample_index]
    volume_hu, spacing = build_volume(selected.file_location)
    volume_norm = hu_clip_normalize(volume_hu)
    volume_iso, iso_spacing = resample_isotropic(volume_norm, spacing)

    xml_files = sorted(Path(selected.file_location).glob("*.xml"))
    annotations = parse_lidc_xml(xml_files[0]) if xml_files else {"nodules": []}

    print("=== AI-LUNG Sanity Check ===")
    print(f"Subject: {selected.subject_id}")
    print(f"Series UID: {selected.series_uid}")
    print(f"Series path: {selected.file_location}")
    print(f"Raw volume shape (z,y,x): {volume_hu.shape}")
    print(f"Raw spacing (z,y,x): {spacing}")
    print(f"Normalized range: min={volume_norm.min():.4f}, max={volume_norm.max():.4f}")
    print(f"Resampled shape (z,y,x): {volume_iso.shape}")
    print(f"Isotropic spacing (z,y,x): {iso_spacing}")
    print(f"Nodule annotation entries: {len(annotations.get('nodules', []))}")


if __name__ == "__main__":
    main()
