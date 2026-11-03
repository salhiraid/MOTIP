"""VisDrone2019-MOT (drone footage) -> unified layout.  train (56 seq) -> train, val (7 seq) -> val.

Expected input (VisDrone2019-MOT-train / -val as downloaded):
    <root>/VisDrone2019-MOT-train/sequences/<seq>/0000001.jpg ...
    <root>/VisDrone2019-MOT-train/annotations/<seq>.txt
Annotation row: frame,target_id,left,top,width,height,score,object_category,truncation,occlusion
  score = 0 marks an entry the benchmark does not consider (-> ignore region);  category 0 = ignored region.
Categories: 1 pedestrian, 2 people, 3 bicycle, 4 car, 5 van, 6 truck, 7 tricycle, 8 awning-tricycle, 9 bus, 10 motor, 11 others
"""
import os
from collections import Counter

from .common import Sequence, map_boxes, sorted_listdir
from data.class_maps import DISTRACTOR, IGNORE

NAME, DIR = "visdrone", "VisDrone_MOT"
LICENSE = "VisDrone: research use, see github.com/VisDrone/VisDrone-Dataset - licence not verified here."
# ASSUMPTION (override with --class-map): tricycles are not motorcycles -> distractors; vans are cars.
DEFAULT_MAP = {1: "person", 2: "person", 3: "bicycle", 4: "car", 5: "car", 6: "truck", 7: DISTRACTOR, 8: DISTRACTOR,
               9: "bus", 10: "motorcycle", 11: DISTRACTOR, 0: IGNORE}
SCORE0 = "__not_considered__"
DEFAULT_MAP[SCORE0] = IGNORE


def add_args(ap):
    ap.add_argument("--root", required=True)
    ap.add_argument("--fps", type=float, default=30.0, help="frame rate to record (VisDrone does not ship it)")


def sequences(a, class_map, class_set):
    unknown = Counter()
    # JSON class maps have string keys
    class_map = {(int(k) if isinstance(k, str) and k.lstrip("-").isdigit() else k): v for k, v in class_map.items()}
    for split, folder in (("train", "VisDrone2019-MOT-train"), ("val", "VisDrone2019-MOT-val")):
        base = os.path.join(a.root, folder)
        if not os.path.isdir(base):
            print(f"[visdrone] missing {base}")
            continue
        for seq_name in sorted_listdir(os.path.join(base, "sequences")):
            sdir = os.path.join(base, "sequences", seq_name)
            paths = [os.path.join(sdir, f) for f in sorted_listdir(sdir, (".jpg",))]
            seq = Sequence(name=seq_name, split=split, images=paths, fps=a.fps)
            with open(os.path.join(base, "annotations", seq_name + ".txt")) as f:
                for line in f:
                    p = line.strip().split(",")
                    if len(p) < 8:
                        continue
                    fr, tid = int(p[0]), int(p[1])
                    x, y, w, h = map(float, p[2:6])
                    label = SCORE0 if int(float(p[6])) == 0 else int(p[7])
                    map_boxes(class_map, class_set, label, tid, fr, x, y, w, h, 1.0, seq, unknown)
            if unknown:
                seq.meta["warnings"] = [f"unmapped labels dropped: {dict(unknown)}"]
            yield seq
