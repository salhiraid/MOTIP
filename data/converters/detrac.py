"""UA-DETRAC (60 train + 40 test sequences, 960x540, 25 fps) -> unified layout.  Official train -> `train`, official
test -> `val` (both have public annotations).  Vehicle types: car, bus, van, others.

Expected input:
    <root>/DETRAC-train-data/MVI_20011/img00001.jpg ...    <root>/DETRAC-Train-Annotations-XML/MVI_20011.xml
    <root>/DETRAC-test-data/MVI_39031/img00001.jpg ...     <root>/DETRAC-Test-Annotations-XML/MVI_39031.xml
(directory names are matched case-insensitively; a leading 'Insight-' on the image folder is accepted.)
XML: <sequence name><ignored_region><box left top width height/></ignored_region>
     <frame num density><target_list><target id><box left top width height/><attribute vehicle_type=.../></target>
"""
import os
import xml.etree.ElementTree as ET
from collections import Counter

from .common import Sequence, map_boxes, sorted_listdir
from data.class_maps import DISTRACTOR, IGNORE

NAME, DIR, FPS = "detrac", "UA_DETRAC", 25.0
LICENSE = "UA-DETRAC: research use; see the dataset page (detrac-db.rit.albany.edu) - licence not verified here."
# ASSUMPTION: `others` (misc. trucks/vehicles) is not mapped to a class: it becomes an evaluation distractor.
DEFAULT_MAP = {"car": "car", "van": "car", "bus": "bus", "others": DISTRACTOR}
IGNORE_LABEL = "__ignored_region__"
DEFAULT_MAP[IGNORE_LABEL] = IGNORE


def add_args(ap):
    ap.add_argument("--root", required=True)


def _find(root, *cands):
    names = {n.lower(): n for n in os.listdir(root)}
    for c in cands:
        if c.lower() in names:
            return os.path.join(root, names[c.lower()])
    return None


def _seq_dir(img_root, name):
    for c in (name, f"Insight-{name}", f"{name}_img"):
        p = os.path.join(img_root, c)
        if os.path.isdir(p):
            return p
    return None


def sequences(a, class_map, class_set):
    unknown = Counter()
    for split, img_name, ann_name in (("train", "DETRAC-train-data", "DETRAC-Train-Annotations-XML"),
                                      ("val", "DETRAC-test-data", "DETRAC-Test-Annotations-XML")):
        img_root, ann_root = _find(a.root, img_name), _find(a.root, ann_name)
        if not img_root or not ann_root:
            print(f"[detrac] split {split}: missing {img_name!r} or {ann_name!r} under {a.root}")
            continue
        for fn in sorted(f for f in os.listdir(ann_root) if f.endswith(".xml")):
            name = fn[:-4]
            sdir = _seq_dir(img_root, name)
            if sdir is None:
                print(f"[detrac] no image folder for {name}")
                continue
            paths = [os.path.join(sdir, f) for f in sorted_listdir(sdir, (".jpg", ".png"))]
            seq = Sequence(name=name, split=split, images=paths, fps=FPS)
            root = ET.parse(os.path.join(ann_root, fn)).getroot()
            n_frames = len(paths)
            for b in root.findall("./ignored_region/box"):
                for fr in range(1, n_frames + 1):
                    map_boxes(class_map, class_set, IGNORE_LABEL, 0, fr, float(b.get("left")), float(b.get("top")),
                              float(b.get("width")), float(b.get("height")), 1.0, seq, unknown)
            for fe in root.findall("frame"):
                fr = int(fe.get("num"))
                for t in fe.findall("./target_list/target"):
                    b, at = t.find("box"), t.find("attribute")
                    map_boxes(class_map, class_set, (at.get("vehicle_type") if at is not None else "car"), int(t.get("id")),
                              fr, float(b.get("left")), float(b.get("top")), float(b.get("width")), float(b.get("height")),
                              1.0, seq, unknown)
            if unknown:
                seq.meta["warnings"] = [f"unmapped labels dropped: {dict(unknown)}"]
            yield seq
