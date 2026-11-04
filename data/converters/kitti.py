"""KITTI tracking (2D boxes of the 21 labelled training sequences, 10 fps, 1242x375 PNG) -> unified layout.

Expected input:  <root>/training/image_02/<0000>/<000000>.png    <root>/training/label_02/<0000>.txt
label row: frame track_id type truncated occluded alpha x1 y1 x2 y2 h w l x y z ry
Split: the CenterTrack validation sequences (1 6 8 10 12 13 14 15 16 18 19) -> val, the rest -> train; override with
--val-seqs. KITTI's test sequences have no public labels.  PNG frames: prefer `--image-mode jpeg` (faster loading).
"""
import os
from collections import Counter

from .common import Sequence, map_boxes, sorted_listdir
from data.class_maps import DISTRACTOR, IGNORE

NAME, DIR, FPS = "kitti", "KITTI_Tracking", 10.0
LICENSE = "KITTI: CC BY-NC-SA 3.0 (non-commercial)."
DEFAULT_MAP = {"Car": "car", "Van": "car", "Truck": "truck", "Cyclist": "bicycle", "Pedestrian": "person",
               "Person_sitting": "person", "Tram": DISTRACTOR, "Misc": DISTRACTOR, "DontCare": IGNORE}
VAL_SEQS = [1, 6, 8, 10, 12, 13, 14, 15, 16, 18, 19]


def add_args(ap):
    ap.add_argument("--root", required=True)
    ap.add_argument("--val-seqs", type=int, nargs="*", default=None)


def sequences(a, class_map, class_set):
    unknown = Counter()
    val = set(a.val_seqs if a.val_seqs is not None else VAL_SEQS)
    imgs, labs = os.path.join(a.root, "training", "image_02"), os.path.join(a.root, "training", "label_02")
    for sid in sorted_listdir(imgs):
        sdir = os.path.join(imgs, sid)
        if not os.path.isdir(sdir) or not os.path.exists(os.path.join(labs, sid + ".txt")):
            continue
        paths = [os.path.join(sdir, f) for f in sorted_listdir(sdir, (".png", ".jpg"))]
        seq = Sequence(name=sid, split="val" if int(sid) in val else "train", images=paths, fps=FPS)
        with open(os.path.join(labs, sid + ".txt")) as f:
            for line in f:
                p = line.split()
                if len(p) < 10:
                    continue
                x1, y1, x2, y2 = map(float, p[6:10])
                occ = int(float(p[4]))
                map_boxes(class_map, class_set, p[2], int(p[1]), int(p[0]) + 1, x1, y1, x2 - x1, y2 - y1,
                          {0: 1.0, 1: 0.7, 2: 0.4}.get(occ, 0.5), seq, unknown)
        if unknown:
            seq.meta["warnings"] = [f"unmapped labels dropped: {dict(unknown)}"]
        yield seq
