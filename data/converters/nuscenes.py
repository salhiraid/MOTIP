"""nuScenes (3D cuboids, 6 cameras) -> 2D tracking sequences by projecting the cuboids into each camera.

Reads the devkit's JSON tables directly (no devkit needed): <root>/<version>/{scene,sample,sample_data,sensor,
calibrated_sensor,ego_pose,sample_annotation,instance,category,log,visibility}.json and the images under <root>/samples/.
One sequence per (scene, camera); frames = the ANNOTATED KEYFRAMES (2 Hz), so a 20 s scene gives ~40 frames.

CAVEATS: (1) 2 Hz is far slower than the other datasets: the config generator caps its frame interval at 1.
(2) 2D boxes are the bounding box of the projected cuboid -> looser than hand-drawn 2D boxes, and nuScenes has no
per-camera occlusion, so occluded objects stay labelled: filter with --min-visibility / --min-pts / --min-in-image.
(3) Official train/val scene lists come from the devkit (`pip install nuscenes-devkit`); without it the split is by LOG
(whole drives), deterministic with --seed.
"""
import json
import os
import random
from collections import Counter, defaultdict

import numpy as np

from .common import Sequence, IdMap, map_boxes
from .project3d import quat_to_rot, box_corners, to_frame, project_box
from data.class_maps import DISTRACTOR

NAME, DIR, FPS = "nuscenes", "NuScenes", 2.0
LICENSE = "nuScenes: CC BY-NC-SA 4.0 (non-commercial) - commercial use needs a separate licence from Motional."
DEFAULT_MAP = {
    "vehicle.car": "car", "vehicle.truck": "truck", "vehicle.bus.bendy": "bus", "vehicle.bus.rigid": "bus",
    "vehicle.motorcycle": "motorcycle", "vehicle.bicycle": "bicycle",
    "human.pedestrian.adult": "person", "human.pedestrian.child": "person", "human.pedestrian.construction_worker": "person",
    "human.pedestrian.police_officer": "person", "human.pedestrian.personal_mobility": "person",
    "vehicle.trailer": DISTRACTOR, "vehicle.construction": DISTRACTOR, "vehicle.emergency.ambulance": DISTRACTOR,
    "vehicle.emergency.police": DISTRACTOR,
}
CAMERAS = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]


def add_args(ap):
    ap.add_argument("--root", required=True, help="nuScenes root (contains samples/ and <version>/)")
    ap.add_argument("--version", default="v1.0-trainval")
    ap.add_argument("--cameras", nargs="*", default=CAMERAS)
    ap.add_argument("--min-visibility", type=int, default=2, help="nuScenes visibility level 1..4 (1 = 0-40%% visible)")
    ap.add_argument("--min-pts", type=int, default=1, help="min lidar+radar points in the cuboid")
    ap.add_argument("--min-in-image", type=float, default=0.4, help="min fraction of the projected box inside the image")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-frac", type=float, default=0.15, help="fallback split (no devkit): fraction of logs")


def _load(d, name):
    return json.load(open(os.path.join(d, name + ".json")))


def _split_scenes(scenes, logs, a):
    try:
        from nuscenes.utils.splits import create_splits_scenes
        sp = create_splits_scenes()
        train, val = set(sp["train"]), set(sp["val"])
        out = {s["token"]: ("train" if s["name"] in train else "val" if s["name"] in val else None) for s in scenes}
        if any(out.values()):
            return out, "devkit train/val lists"
    except Exception:                                      # noqa: BLE001 - devkit not installed / mini version
        pass
    log_tokens = sorted({s["log_token"] for s in scenes})
    random.Random(a.seed).shuffle(log_tokens)
    val_logs = set(log_tokens[:max(1, int(round(a.val_frac * len(log_tokens))))]) if len(log_tokens) > 1 else set()
    return {s["token"]: ("val" if s["log_token"] in val_logs else "train") for s in scenes}, \
        f"fallback split by log ({len(val_logs)} of {len(log_tokens)} logs in val); install nuscenes-devkit for the official lists"


def sequences(a, class_map, class_set):
    d = os.path.join(a.root, a.version)
    scenes, samples, sdata = _load(d, "scene"), _load(d, "sample"), _load(d, "sample_data")
    sensors = {s["token"]: s for s in _load(d, "sensor")}
    calib = {c["token"]: c for c in _load(d, "calibrated_sensor")}
    ego = {e["token"]: e for e in _load(d, "ego_pose")}
    anns = _load(d, "sample_annotation")
    inst = {i["token"]: i for i in _load(d, "instance")}
    cats = {c["token"]: c["name"] for c in _load(d, "category")}
    vis_level = {v["token"]: int(v["token"]) for v in _load(d, "visibility")}
    sample_by_token = {s["token"]: s for s in samples}
    split_of, how = _split_scenes(scenes, None, a)
    print(f"[nuscenes] {len(scenes)} scenes, split: {how}")
    cam_data = {}                                           # (sample_token, channel) -> sample_data (keyframes)
    for sd in sdata:
        if sd["is_key_frame"]:
            ch = sensors[calib[sd["calibrated_sensor_token"]]["sensor_token"]]["channel"]
            cam_data[(sd["sample_token"], ch)] = sd
    anns_by_sample = defaultdict(list)
    for an in anns:
        anns_by_sample[an["sample_token"]].append(an)
    unknown = Counter()
    for sc in scenes:
        split = split_of[sc["token"]]
        if split is None:
            continue
        chain, tok = [], sc["first_sample_token"]
        while tok:
            chain.append(sample_by_token[tok])
            tok = sample_by_token[tok]["next"]
        for ch in a.cameras:
            frames = [cam_data.get((s["token"], ch)) for s in chain]
            if any(f is None for f in frames):
                continue
            paths = [os.path.join(a.root, f["filename"]) for f in frames]
            seq = Sequence(name=f"{sc['name']}_{ch}", split=split, images=paths, fps=FPS)
            ids = IdMap()
            for i, (s, sd) in enumerate(zip(chain, frames), 1):
                cs, ep = calib[sd["calibrated_sensor_token"]], ego[sd["ego_pose_token"]]
                R_ego, t_ego = quat_to_rot(ep["rotation"]), np.array(ep["translation"])
                R_cs, t_cs = quat_to_rot(cs["rotation"]), np.array(cs["translation"])
                K = np.array(cs["camera_intrinsic"])
                W, H = sd.get("width") or 1600, sd.get("height") or 900
                seq.width, seq.height = int(W), int(H)
                for an in anns_by_sample[s["token"]]:
                    if vis_level.get(an.get("visibility_token"), 4) < a.min_visibility:
                        continue
                    if an.get("num_lidar_pts", 1) + an.get("num_radar_pts", 0) < a.min_pts:
                        continue
                    w_, l_, h_ = an["size"]
                    corners = box_corners(an["translation"], l_, w_, h_, quat_to_rot(an["rotation"]))
                    cam = to_frame(to_frame(corners, R_ego, t_ego), R_cs, t_cs)
                    pb = project_box(cam, K, W, H)
                    if pb is None or pb[4] < a.min_in_image:
                        continue
                    x1, y1, x2, y2, _ = pb
                    label = cats[inst[an["instance_token"]]["category_token"]]
                    map_boxes(class_map, class_set, label, ids(an["instance_token"]), i, x1, y1, x2 - x1, y2 - y1,
                              1.0, seq, unknown)
            if unknown:
                seq.meta["warnings"] = [f"unmapped categories dropped: {dict(unknown)}"]
            yield seq
