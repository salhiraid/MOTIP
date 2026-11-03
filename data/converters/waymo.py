"""Waymo Open Dataset (perception, v2 parquet) 2D camera boxes with per-camera tracking ids -> unified layout.

Expected input (Waymo v2.0.x modular parquet, https://waymo.com/open):
    <root>/{training,validation}/camera_image/<segment>.parquet     key.camera_name, key.frame_timestamp_micros, image bytes
    <root>/{training,validation}/camera_box/<segment>.parquet       key.camera_object_id, box.center.x/y, box.size.x/y, type
Column names are matched by SUFFIX ("...box.center.x"), so both the flattened `[CameraBoxComponent].box.center.x` and
`CameraBoxComponent.box.center.x` spellings work; if a column is missing the error lists what the file has.
One sequence per (segment, camera) -- 10 Hz, ~200 frames.  Needs pandas + pyarrow.  The v1 TFRecord format is NOT read.

CAVEATS: (1) the camera labels only distinguish vehicle / pedestrian / cyclist / sign: every `vehicle` becomes `car`
(buses, trucks included) -> do not rely on it for the bus/truck classes.  (2) NON-COMMERCIAL licence.
(3) NOT tested on a real Waymo file here (no access); tested on files written in this schema.
"""
import os
from collections import Counter

from .common import Sequence, IdMap, map_boxes, sorted_listdir

NAME, DIR, FPS = "waymo", "Waymo", 10.0
LICENSE = "Waymo Open Dataset: custom NON-COMMERCIAL licence (waymo.com/open/terms)."
# type ids: 1 vehicle, 2 pedestrian, 3 sign, 4 cyclist.  ASSUMPTION: vehicle -> car, cyclist -> bicycle.
ANNOTATED = []     # see prepare.py: [] = coarse/class-less labels (no class supervision), "data" = classes present
DEFAULT_MAP = {1: "car", 2: "person", 3: None, 4: "bicycle"}
CAMERA_NAMES = {1: "FRONT", 2: "FRONT_LEFT", 3: "FRONT_RIGHT", 4: "SIDE_LEFT", 5: "SIDE_RIGHT"}


def add_args(ap):
    ap.add_argument("--root", required=True)
    ap.add_argument("--cameras", nargs="*", type=int, default=[1], help="camera ids (1 FRONT, 2 FL, 3 FR, 4 SL, 5 SR)")
    ap.add_argument("--max-segments", type=int, default=0, help="debug: only the first N segments per split")


def _col(df, suffix):
    hits = [c for c in df.columns if c.endswith(suffix)]
    if not hits:
        raise KeyError(f"no column ending in {suffix!r}; columns: {list(df.columns)}")
    return hits[0]


def sequences(a, class_map, class_set):
    import pandas as pd
    unknown = Counter()
    class_map = {(int(k) if isinstance(k, str) and k.isdigit() else k): v for k, v in class_map.items()}
    for folder, split in (("training", "train"), ("validation", "val")):
        img_dir, box_dir = (os.path.join(a.root, folder, "camera_image"), os.path.join(a.root, folder, "camera_box"))
        if not os.path.isdir(img_dir) or not os.path.isdir(box_dir):
            print(f"[waymo] split {folder}: missing camera_image/ or camera_box/ under {a.root}")
            continue
        files = [f for f in sorted_listdir(box_dir, (".parquet",))]
        for n, fn in enumerate(files):
            if a.max_segments and n >= a.max_segments:
                break
            if not os.path.exists(os.path.join(img_dir, fn)):
                continue
            imgs, boxes = pd.read_parquet(os.path.join(img_dir, fn)), pd.read_parquet(os.path.join(box_dir, fn))
            i_cam, i_ts, i_img = _col(imgs, "camera_name"), _col(imgs, "frame_timestamp_micros"), _col(imgs, "image")
            b_cam, b_ts, b_id = _col(boxes, "camera_name"), _col(boxes, "frame_timestamp_micros"), _col(boxes, "camera_object_id")
            cx, cy, sx, sy, ty = (_col(boxes, s) for s in ("box.center.x", "box.center.y", "box.size.x", "box.size.y", "type"))
            for cam in a.cameras:
                im = imgs[imgs[i_cam] == cam].sort_values(i_ts)
                if len(im) == 0:
                    continue
                ts_index = {t: k for k, t in enumerate(im[i_ts].tolist(), 1)}
                seq = Sequence(name=f"{fn[:-8]}_{CAMERA_NAMES.get(cam, cam)}", split=split,
                               images=[bytes(b) for b in im[i_img].tolist()], fps=FPS)
                ids = IdMap()
                for row in boxes[boxes[b_cam] == cam].to_dict("records"):
                    k = ts_index.get(row[b_ts])
                    if k is None:
                        continue
                    w, h = float(row[sx]), float(row[sy])
                    map_boxes(class_map, class_set, int(row[ty]), ids(row[b_id]), k, float(row[cx]) - w / 2,
                              float(row[cy]) - h / 2, w, h, 1.0, seq, unknown)
                if unknown:
                    seq.meta["warnings"] = [f"unmapped types dropped: {dict(unknown)}"]
                yield seq
