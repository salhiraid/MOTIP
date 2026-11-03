"""Generic 3D-box importer: any dataset with 3D cuboids + camera projection matrices (e.g. I-24 3D) -> 2D sequences.

I did NOT implement a dedicated I-24 3D reader: its file format could not be verified offline.  Instead, write one small
JSON per scene in the neutral format below (a few lines of Python from the dataset's own files) and run this converter:

    <root>/<anything>.json
    {"name": "scene1", "split": "train", "fps": 30,
     "cameras": {"cam01": {"P": [[...3x4 world->pixel projection...]], "width": 3840, "height": 2160,
                           "frames_dir": "/abs/or/relative/path/to/frames", "ext": ".jpg"}},
     "boxes": [{"frame": 0, "id": 7, "class": "car", "center": [x, y, z], "size": [length, width, height], "yaw": 0.3}]}

* frames in `frames_dir` are sorted by name; box "frame" is the 0-based index into that sorted list.
* "class" is looked up in the class map (default maps the unified names to themselves; add your own labels with
  --class-map).  yaw is the rotation about the vertical (z) axis, in radians, of the box's length axis.
* One sequence per (scene, camera).  Boxes with a corner behind the camera are dropped.
"""
import json
import os
from collections import Counter

import numpy as np

from .common import Sequence, IdMap, map_boxes, sorted_listdir
from .project3d import box_corners, yaw_to_rot, project_matrix_box
from data.class_maps import CLASS_SETS

NAME, DIR, FPS = "boxes3d", "Generic3D", 30.0
LICENSE = "Depends on the source dataset (I-24 3D: check the licence of the release you use)."
ANNOTATED = "data"     # see prepare.py: [] = coarse/class-less labels (no class supervision), "data" = classes present
DEFAULT_MAP = {n: n for names in CLASS_SETS.values() for n in names}


def add_args(ap):
    ap.add_argument("--root", required=True, help="folder with the scene JSON files")
    ap.add_argument("--min-in-image", type=float, default=0.4)


def sequences(a, class_map, class_set):
    unknown = Counter()
    for fn in sorted_listdir(a.root, (".json",)):
        sc = json.load(open(os.path.join(a.root, fn)))
        for cam, c in sc["cameras"].items():
            fdir = c["frames_dir"] if os.path.isabs(c["frames_dir"]) else os.path.join(a.root, c["frames_dir"])
            paths = [os.path.join(fdir, f) for f in sorted_listdir(fdir, (c.get("ext", ".jpg"),))]
            seq = Sequence(name=f"{sc['name']}_{cam}", split=sc.get("split", "train"), images=paths,
                           fps=float(sc.get("fps", FPS)), width=int(c["width"]), height=int(c["height"]))
            ids = IdMap()
            for b in sc["boxes"]:
                l, w, h = b["size"]
                corners = box_corners(b["center"], l, w, h, yaw_to_rot(b.get("yaw", 0.0)))
                pb = project_matrix_box(corners, c["P"], c["width"], c["height"])
                if pb is None or pb[4] < a.min_in_image:
                    continue
                x1, y1, x2, y2, _ = pb
                map_boxes(class_map, class_set, b["class"], ids(b["id"]), int(b["frame"]) + 1, x1, y1, x2 - x1, y2 - y1,
                          1.0, seq, unknown)
            if unknown:
                seq.meta["warnings"] = [f"unmapped classes dropped: {dict(unknown)}"]
            yield seq
