"""BDD100K MOT (2020) -> unified layout.  Official splits: train (1400 videos) / val (200).  5 fps JPEG frames.

Expected input (as downloaded from bdd-data.berkeley.edu, "MOT 2020 Images" + "MOT 2020 Labels"):
    <root>/images/track/{train,val}/<video>/<video>-0000001.jpg ...
    <root>/labels/box_track_20/{train,val}/<video>.json          (one JSON list of frames per video)
Label frame: {"name": "<video>-0000001.jpg", "frameIndex": 0, "labels": [{"id": "...", "category": "car",
              "attributes": {"crowd": false, ...}, "box2d": {"x1","y1","x2","y2"}}]}
"""
import json
import os
from collections import Counter

from .common import Sequence, IdMap, map_boxes
from data.class_maps import IGNORE, DISTRACTOR

NAME, DIR, FPS = "bdd100k", "BDD100K_MOT", 5.0
LICENSE = "BDD100K: custom licence (Berkeley DeepDrive) - read it; commonly treated as non-commercial research use."
DEFAULT_MAP = {
    "car": "car", "truck": "truck", "bus": "bus", "motorcycle": "motorcycle", "bicycle": "bicycle",
    "pedestrian": "person", "other person": "person", "rider": "person",
    "train": DISTRACTOR, "trailer": DISTRACTOR, "other vehicle": DISTRACTOR,
}


def add_args(ap):
    ap.add_argument("--root", required=True, help="BDD100K root containing images/track and labels/box_track_20")
    ap.add_argument("--images", default=None, help="default <root>/images/track")
    ap.add_argument("--labels", default=None, help="default <root>/labels/box_track_20")


def sequences(a, class_map, class_set):
    images = a.images or os.path.join(a.root, "images", "track")
    labels = a.labels or os.path.join(a.root, "labels", "box_track_20")
    unknown = Counter()
    for split in ("train", "val"):
        ldir = os.path.join(labels, split)
        if not os.path.isdir(ldir):
            print(f"[bdd100k] no labels for split {split} at {ldir}")
            continue
        for fn in sorted(os.listdir(ldir)):
            if not fn.endswith(".json"):
                continue
            video = fn[:-5]
            frames = json.load(open(os.path.join(ldir, fn)))
            frames.sort(key=lambda f: f.get("frameIndex", 0))
            paths = [os.path.join(images, split, video, f["name"]) for f in frames]
            seq = Sequence(name=video, split=split, images=paths, fps=FPS)
            ids = IdMap()
            for i, f in enumerate(frames, 1):
                for lb in f.get("labels") or []:
                    b = lb.get("box2d")
                    if not b:
                        continue
                    x, y, w, h = b["x1"], b["y1"], b["x2"] - b["x1"], b["y2"] - b["y1"]
                    label = IGNORE_LABEL if (lb.get("attributes") or {}).get("crowd") else lb["category"]
                    map_boxes(class_map, class_set, label, ids(lb["id"]), i, x, y, w, h, 1.0, seq, unknown)
            if unknown:
                seq.meta["warnings"] = [f"unmapped labels dropped: {dict(unknown)}"]
            yield seq


IGNORE_LABEL = "__crowd__"
DEFAULT_MAP[IGNORE_LABEL] = IGNORE
