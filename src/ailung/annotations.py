from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np


def build_series_to_xml_mapping(xml_dir: str | Path) -> dict[str, Path]:
    """
    Scan XML directory and build mapping from SeriesInstanceUid to XML file path.
    This is needed because LIDC XML files are named by numeric IDs, not series UIDs.
    
    Returns:
        Dictionary mapping series_uid (str) -> xml_path (Path)
    """
    xml_dir = Path(xml_dir)
    mapping = {}
    
    for xml_file in xml_dir.rglob("*.xml"):
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            series_uid = root.findtext(".//{*}SeriesInstanceUid")
            if series_uid:
                mapping[series_uid] = xml_file
        except Exception:
            # Skip malformed XML files
            continue
    
    return mapping


def parse_lidc_xml(xml_path: str | Path) -> dict[str, list[dict]]:
    path = Path(xml_path)
    tree = ET.parse(path)
    root = tree.getroot()

    data = defaultdict(list)

    for rs in root.iterfind(".//{*}readingSession"):
        radiologist_id = rs.findtext("{*}servicingRadiologistID", default="unknown")

        for nodule in rs.findall("{*}unblindedReadNodule"):
            nodule_id = nodule.findtext("{*}noduleID", default="")

            characteristics = {}
            chars = nodule.find("{*}characteristics")
            if chars is not None:
                for child in list(chars):
                    tag = child.tag.split("}")[-1]
                    characteristics[tag] = child.text

            rois = []
            for roi in nodule.findall("{*}roi"):
                z = roi.findtext("{*}imageZposition", default="")
                sop_uid = roi.findtext("{*}imageSOP_UID", default="")
                inclusion = roi.findtext("{*}inclusion", default="")

                edges = []
                for em in roi.findall("{*}edgeMap"):
                    x = em.findtext("{*}xCoord", default="")
                    y = em.findtext("{*}yCoord", default="")
                    edges.append({"x": x, "y": y})

                rois.append(
                    {
                        "image_z_position": z,
                        "image_sop_uid": sop_uid,
                        "inclusion": inclusion,
                        "edge_map": edges,
                    }
                )

            data["nodules"].append(
                {
                    "radiologist_id": radiologist_id,
                    "nodule_id": nodule_id,
                    "characteristics": characteristics,
                    "rois": rois,
                }
            )

    return dict(data)


def compute_nodule_centroid_3d(
    nodule: dict, volume_spacing: tuple[float, float, float]
) -> tuple[float, float, float] | None:
    """
    Compute the 3D centroid of a nodule from edge_map annotations.
    
    Args:
        nodule: Single nodule dictionary from parse_lidc_xml
        volume_spacing: (z_spacing, y_spacing, x_spacing) in mm
        
    Returns:
        (z_mm, y_mm, x_mm) centroid in physical coordinates, or None if invalid
    """
    rois = nodule.get("rois", [])
    if not rois:
        return None
    
    all_x = []
    all_y = []
    all_z = []
    
    for roi in rois:
        z_str = roi.get("image_z_position", "")
        edges = roi.get("edge_map", [])
        
        try:
            z_val = float(z_str)
        except ValueError:
            continue
        
        for edge in edges:
            try:
                x_val = float(edge["x"])
                y_val = float(edge["y"])
                all_x.append(x_val)
                all_y.append(y_val)
                all_z.append(z_val)
            except (ValueError, KeyError):
                continue
    
    if not all_x:
        return None
    
    # Compute mean position
    z_mm = np.mean(all_z)
    y_mm = np.mean(all_y) * volume_spacing[1]
    x_mm = np.mean(all_x) * volume_spacing[2]
    
    return (z_mm, y_mm, x_mm)


def get_malignancy_score(nodule: dict) -> int | None:
    """
    Extract malignancy score (1-5) from nodule characteristics.
    Returns None if not found or unparseable.
    """
    chars = nodule.get("characteristics", {})
    mal_str = chars.get("malignancy", "")
    try:
        return int(mal_str)
    except ValueError:
        return None


def build_nodule_candidates(
    xml_path: str | Path, 
    volume_spacing: tuple[float, float, float],
    min_malignancy: int = 3
) -> list[dict]:
    """
    Build a list of nodule candidates suitable for detection training.
    
    Args:
        xml_path: Path to LIDC XML annotation file
        volume_spacing: (z_spacing, y_spacing, x_spacing) in mm
        min_malignancy: Minimum malignancy score (1-5) to include
        
    Returns:
        List of candidate nodules with centroid_3d, malignancy, radiologist_id
    """
    parsed = parse_lidc_xml(xml_path)
    nodules = parsed.get("nodules", [])
    
    candidates = []
    for nodule in nodules:
        malignancy = get_malignancy_score(nodule)
        if malignancy is None or malignancy < min_malignancy:
            continue
        
        centroid = compute_nodule_centroid_3d(nodule, volume_spacing)
        if centroid is None:
            continue
        
        candidates.append({
            "centroid_3d": centroid,  # (z_mm, y_mm, x_mm)
            "malignancy": malignancy,
            "radiologist_id": nodule.get("radiologist_id", "unknown"),
            "nodule_id": nodule.get("nodule_id", "")
        })
    
    return candidates

