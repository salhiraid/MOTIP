"""COCO-style still images with boxes (+ keypoints) -> unified layout, as independent 1-frame "static" sequences.

This is how the user's mmdet vehicle keypoint data (images + COCO json, 31 keypoints per vehicle) can train the MOTIP2 detector:
each image becomes a one-frame sequence flagged `isStatic=1`, which the sampler uses like CrowdHuman images (a 1-frame
"clip" repeated T times, no identities across frames). Keypoint supervision then acts on those images; tracking clips of
other datasets (no keypoint labels) contribute no keypoint loss.

    python tools/prepare_dataset.py cocokp --images /data/imgs --train-json train.json --val-json val.json \
        --out ./datasets --dir-name VehicleKeypoints --image-mode jpeg --resize-short 800 --max-long 1440

Annotation fields used: bbox [x, y, w, h], category_id (-> categories[].name -> the class map), keypoints [x, y, v] * K (any K,
all annotations of a file must agree), iscrowd (-> ignore region). v is kept as is: set KEYPOINT_LABELED_VALUES /
KEYPOINT_VISIBLE_VALUES to your convention (COCO: 1 occluded, 2 visible; binary: 1). Annotations with `num_keypoints == 0` or
all v == 0 keep their box and are simply unlabelled for keypoints.
"""
import json
import os
from collections import Counter

from .common import Sequence, map_boxes
from data.class_maps import IGNORE, DISTRACTOR

NAME, DIR, FPS = "cocokp", "COCOKeypoints", 1.0
MIN_FRAMES = 1
LICENSE = "Depends on the source of your images/labels."
# COCO category names; add yours with --class-map (e.g. {"vehicle": "car"}). Unknown names are dropped and counted.
DEFAULT_MAP = {"car": "car", "bus": "bus", "truck": "truck", "motorcycle": "motorcycle", "bicycle": "bicycle",
               "person": "person", "van": "car", "vehicle": "car", "__crowd__": IGNORE}


def add_args(ap):
    ap.add_argument("--images", required=True, help="folder the json `file_name`s are relative to")
    ap.add_argument("--train-json", required=True)
    ap.add_argument("--val-json", default=None)
    ap.add_argument("--root", default=None, help="unused (kept so every converter has the same options)")


def sequences(a, class_map, class_set):
    unknown = Counter()
    for split, path in (("train", a.train_json), ("val", a.val_json)):
        if not path:
            continue
        j = json.load(open(path))
        cats = {c["id"]: c["name"] for c in j["categories"]}
        by_img = {}
        for an in j["annotations"]:
            by_img.setdefault(an["image_id"], []).append(an)
        for im in j["images"]:
            seq = Sequence(name=f"img{im['id']}", split=split, images=[os.path.join(a.images, im["file_name"])], fps=FPS,
                           width=int(im["width"]), height=int(im["height"]), static=True)
            for k, an in enumerate(by_img.get(im["id"], []), 1):
                x, y, w, h = an["bbox"]
                label = "__crowd__" if an.get("iscrowd") else cats.get(an["category_id"], str(an["category_id"]))
                map_boxes(class_map, class_set, label, k, 1, x, y, w, h, 1.0, seq, unknown)
                if an.get("keypoints") and any(v > 0 for v in an["keypoints"][2::3]):
                    seq.keypoints.append((1, k, [float(v) for v in an["keypoints"]]))
            if unknown:
                seq.meta["warnings"] = [f"unmapped categories dropped: {dict(unknown)}"]
            yield seq
