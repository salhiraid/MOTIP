"""Argoverse 2 Sensor dataset (3D cuboids, 7 ring cameras, 10 Hz annotations) -> 2D tracking sequences by projection.

Expected input (https://www.argoverse.org, sensor dataset):
    <root>/{train,val}/<log_id>/annotations.feather                 timestamp_ns, track_uuid, category, length_m, width_m,
                                                                      height_m, qw,qx,qy,qz, tx_m,ty_m,tz_m, num_interior_pts
    <root>/<split>/<log_id>/calibration/{intrinsics,egovehicle_SE3_sensor}.feather
    <root>/<split>/<log_id>/city_SE3_egovehicle.feather             timestamp_ns, qw,qx,qy,qz, tx_m,ty_m,tz_m
    <root>/<split>/<log_id>/sensors/cameras/<cam>/<timestamp_ns>.jpg
Cuboids live in the ego frame at the lidar timestamp; they are moved to the ego pose of the nearest camera frame
(via the city-frame poses) and projected with the pinhole intrinsics (distortion ignored, like the devkit's default).
One sequence per (log, camera), one frame per annotated timestamp (10 Hz).  Needs pandas + pyarrow.
CAVEATS: no occlusion information -> occluded objects stay labelled (see --min-pts); boxes are looser than 2D boxes.
"""
import os
from collections import Counter

import numpy as np

from .common import Sequence, IdMap, map_boxes, sorted_listdir
from .project3d import quat_to_rot, box_corners, to_frame, project_box
from data.class_maps import DISTRACTOR

NAME, DIR, FPS = "argoverse2", "Argoverse2", 10.0
LICENSE = "Argoverse 2: CC BY-NC-SA 4.0 (non-commercial)."
DEFAULT_MAP = {
    "REGULAR_VEHICLE": "car", "LARGE_VEHICLE": "truck", "TRUCK": "truck", "BOX_TRUCK": "truck", "TRUCK_CAB": "truck",
    "BUS": "bus", "SCHOOL_BUS": "bus", "ARTICULATED_BUS": "bus", "MOTORCYCLE": "motorcycle", "BICYCLE": "bicycle",
    "PEDESTRIAN": "person", "MOTORCYCLIST": DISTRACTOR, "BICYCLIST": DISTRACTOR, "VEHICULAR_TRAILER": DISTRACTOR,
    "WHEELED_RIDER": DISTRACTOR, "WHEELED_DEVICE": DISTRACTOR,
}
CAMERAS = ["ring_front_center", "ring_front_left", "ring_front_right", "ring_rear_left", "ring_rear_right",
           "ring_side_left", "ring_side_right"]


def add_args(ap):
    ap.add_argument("--root", required=True)
    ap.add_argument("--cameras", nargs="*", default=CAMERAS)
    ap.add_argument("--min-pts", type=int, default=1, help="min lidar points inside the cuboid")
    ap.add_argument("--min-in-image", type=float, default=0.4)
    ap.add_argument("--max-sync-ms", type=float, default=50.0, help="max |camera - lidar| timestamp difference")


def _pose(row):
    return quat_to_rot([row.qw, row.qx, row.qy, row.qz]), np.array([row.tx_m, row.ty_m, row.tz_m])


def _nearest(sorted_ts, t):
    i = int(np.searchsorted(sorted_ts, t))
    cands = [j for j in (i - 1, i) if 0 <= j < len(sorted_ts)]
    return min(cands, key=lambda j: abs(int(sorted_ts[j]) - int(t)))


def sequences(a, class_map, class_set):
    import pandas as pd
    unknown = Counter()
    for split in ("train", "val"):
        base = os.path.join(a.root, split)
        if not os.path.isdir(base):
            continue
        for log in sorted_listdir(base):
            ld = os.path.join(base, log)
            if not os.path.exists(os.path.join(ld, "annotations.feather")):
                continue
            ann = pd.read_feather(os.path.join(ld, "annotations.feather"))
            intr = pd.read_feather(os.path.join(ld, "calibration", "intrinsics.feather")).set_index("sensor_name")
            ext = pd.read_feather(os.path.join(ld, "calibration", "egovehicle_SE3_sensor.feather")).set_index("sensor_name")
            city = pd.read_feather(os.path.join(ld, "city_SE3_egovehicle.feather")).sort_values("timestamp_ns")
            city_ts = city["timestamp_ns"].to_numpy()
            city_rows = list(city.itertuples())
            ann_ts = sorted(ann["timestamp_ns"].unique())
            by_ts = {t: g for t, g in ann.groupby("timestamp_ns")}
            for cam in a.cameras:
                cdir = os.path.join(ld, "sensors", "cameras", cam)
                if not os.path.isdir(cdir):
                    continue
                cam_ts = np.array(sorted(int(f[:-4]) for f in os.listdir(cdir) if f.endswith(".jpg")), dtype=np.int64)
                if len(cam_ts) == 0:
                    continue
                K = np.array([[intr.loc[cam, "fx_px"], 0, intr.loc[cam, "cx_px"]],
                              [0, intr.loc[cam, "fy_px"], intr.loc[cam, "cy_px"]], [0, 0, 1.0]])
                W, H = int(intr.loc[cam, "width_px"]), int(intr.loc[cam, "height_px"])
                R_se, t_se = quat_to_rot([ext.loc[cam, c] for c in ("qw", "qx", "qy", "qz")]), \
                    np.array([ext.loc[cam, c] for c in ("tx_m", "ty_m", "tz_m")])
                frames = []
                for t in ann_ts:
                    j = _nearest(cam_ts, t)
                    if abs(int(cam_ts[j]) - int(t)) <= a.max_sync_ms * 1e6:
                        frames.append((t, int(cam_ts[j])))
                paths = [os.path.join(cdir, f"{tc}.jpg") for _, tc in frames]
                seq = Sequence(name=f"{log}_{cam}", split=split, images=paths, fps=FPS, width=W, height=H)
                ids = IdMap()
                for i, (t, tc) in enumerate(frames, 1):
                    R_a, t_a = _pose(city_rows[_nearest(city_ts, t)])          # ego(t_lidar) -> city
                    R_c, t_c = _pose(city_rows[_nearest(city_ts, tc)])         # ego(t_cam)   -> city
                    for r in by_ts[t].itertuples():
                        if r.num_interior_pts < a.min_pts:
                            continue
                        corners = box_corners([r.tx_m, r.ty_m, r.tz_m], r.length_m, r.width_m, r.height_m,
                                              quat_to_rot([r.qw, r.qx, r.qy, r.qz]))
                        city_pts = corners @ R_a.T + t_a
                        cam_pts = to_frame(to_frame(city_pts, R_c, t_c), R_se, t_se)
                        pb = project_box(cam_pts, K, W, H)
                        if pb is None or pb[4] < a.min_in_image:
                            continue
                        x1, y1, x2, y2, _ = pb
                        map_boxes(class_map, class_set, r.category, ids(r.track_uuid), i, x1, y1, x2 - x1, y2 - y1,
                                  1.0, seq, unknown)
                if unknown:
                    seq.meta["warnings"] = [f"unmapped categories dropped: {dict(unknown)}"]
                yield seq
