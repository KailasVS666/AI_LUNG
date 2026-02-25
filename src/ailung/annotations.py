from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import xml.etree.ElementTree as ET


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
