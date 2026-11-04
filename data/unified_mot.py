# Generic loader for every dataset converted by data/converters/* (and for the user's own `Vehicles` dataset).
#
# Layout (MOT-Challenge style, DanceTrack-compatible):
#   <DATA_ROOT>/<DirName>/{train,val,valmini,...}/<seq>/img1/00000001.<ext>     1-indexed frames, ext from seqinfo.ini imExt
#                                                    /gt/gt.txt          TrackEval GT: class column 1 = every target (class-agnostic
#                                                                        HOTA), 8 = distractor / ignore regions
#                                                    /gt/gt_classes.txt  same targets WITHOUT distractors, class column = unified
#                                                                        class index (data/class_maps.py) -> detector training
#                                                    /seqinfo.ini        incl. frameRate, imExt
#   <DATA_ROOT>/<DirName>/<split>_seqmap.txt
# `valmini` may be a folder of symlinks to val sequences.
#
# Any dataset name that is not registered below is loaded from `<DATA_ROOT>/<name>` with this same loader, so a custom
# dataset only needs to be written in this layout (see docs/DATASETS.md, "Adding your own dataset").
import json
import os

import torch
from configparser import ConfigParser

from .dancetrack import DanceTrack
from .util import is_legal, append_annotation

# registry key -> directory name under DATA_ROOT
UNIFIED_DATASETS = {
    "Vehicles": "Vehicles",                 # user's own CVAT-video dataset (tools/vehicles/)
    "BDD100K_MOT": "BDD100K_MOT",
    "UA_DETRAC": "UA_DETRAC",
    "VisDrone_MOT": "VisDrone_MOT",
    "UAVDT": "UAVDT",
    "FastTracker": "FastTracker",
    "CityFlow": "CityFlow",
    "KITTI_Tracking": "KITTI_Tracking",
    "Waymo": "Waymo",
    "NuScenes": "NuScenes",
    "Argoverse2": "Argoverse2",
    "COCOKeypoints": "COCOKeypoints",       # still images with keypoint labels (tools/prepare_dataset.py cocokp)
    "Generic3D": "Generic3D",               # 3D-box datasets projected by data/converters/boxes3d_generic.py (e.g. I-24 3D)
}


class UnifiedMOT(DanceTrack):
    SUB_DIR = "Vehicles"

    def __init__(self, data_root: str = "./datasets/", sub_dir: str | None = None, split: str = "train",
                 load_annotation: bool = True):
        sub_dir = sub_dir or self.SUB_DIR
        path = os.path.join(data_root, sub_dir, split)
        if not os.path.isdir(path):
            raise FileNotFoundError(
                f"dataset '{sub_dir}' split '{split}' not found at {path}. Registered datasets: "
                f"{sorted(UNIFIED_DATASETS)}; custom datasets must use the layout described in data/unified_mot.py.")
        super().__init__(data_root=data_root, sub_dir=sub_dir, split=split, load_annotation=load_annotation)

    def annotated_classes(self):
        """Indices of the unified classes this dataset annotates (written by the converter as annotated_classes.json),
        or None = all of them (own datasets without the file).  Used for per-dataset classification-loss masking."""
        path = os.path.join(self.data_dir, "annotated_classes.json")
        if not os.path.exists(path):
            return None
        return [int(i) for i in json.load(open(path))["annotated_indices"]]

    def _get_sequence_names(self):
        return sorted(d for d in os.listdir(os.path.join(self.data_dir, self.split))
                      if os.path.isdir(os.path.join(self.data_dir, self.split, d)))

    def _get_sequence_infos(self):
        infos = super()._get_sequence_infos()
        for name, info in infos.items():
            ini = ConfigParser()
            ini.read(os.path.join(self._get_sequence_dir(self.data_dir, self.split, name), "seqinfo.ini"))
            info["ext"] = ini["Sequence"].get("imExt", ".jpg") or ".jpg"
            info["fps"] = float(ini["Sequence"].get("frameRate", 0) or 0)
            info["is_static"] = ini["Sequence"].get("isStatic", "0").strip() in ("1", "true", "True")   # independent still image
        return infos

    def _get_image_paths(self):
        image_paths = {}
        for name in self._get_sequence_names():
            seq_dir = self._get_sequence_dir(self.data_dir, self.split, name)
            ext = self.sequence_infos[name]["ext"]
            image_paths[name] = [os.path.join(seq_dir, "img1", f"{i + 1:08d}{ext}")
                                 for i in range(self.sequence_infos[name]["length"])]
        return image_paths

    def _get_annotations(self):
        sequence_names = self._get_sequence_names()
        annotations = self._init_annotations(sequence_names)
        for name in sequence_names:
            gt_dir = os.path.join(self._get_sequence_dir(self.data_dir, self.split, name), "gt")
            path = os.path.join(gt_dir, "gt_classes.txt")
            if not os.path.exists(path):                  # plain MOT layout: single class
                path = os.path.join(gt_dir, "gt.txt")
            n_frames = self.sequence_infos[name]["length"]
            with open(path, "r") as f:
                for line in f:
                    p = line.strip().split(",")
                    if len(p) < 6:
                        continue
                    frame_id, obj_id = int(p[0]), int(float(p[1]))
                    x, y, w, h = map(float, p[2:6])
                    category = int(float(p[7])) if len(p) > 7 and path.endswith("gt_classes.txt") else 0
                    if not (1 <= frame_id <= n_frames) or w <= 0 or h <= 0:
                        continue
                    annotations[name][frame_id - 1] = append_annotation(
                        annotation=annotations[name][frame_id - 1],
                        obj_id=obj_id, category=category, bbox=[x, y, w, h], visibility=1.0)
            for i in range(n_frames):
                annotations[name][i]["is_legal"] = is_legal(annotations[name][i])
            self._load_keypoints(annotations[name], os.path.join(gt_dir, "gt_keypoints.txt"), n_frames)
        return annotations

    @staticmethod
    def _load_keypoints(frames: list, path: str, n_frames: int):
        """Optional `gt/gt_keypoints.txt`: one row per (frame, id): `frame,id,x1,y1,v1,x2,y2,v2,...` (pixels, v = GT
        visibility value, 0 = not labelled). Adds `keypoints` (N,K,2) and `keypoints_vis` (N,K) aligned with the boxes of
        each frame; objects without a row keep v = 0."""
        if not os.path.exists(path):
            return
        rows, K = {}, None
        with open(path) as f:
            for line in f:
                p = line.strip().split(",")
                if len(p) < 5 or (len(p) - 2) % 3:
                    continue
                k = (len(p) - 2) // 3
                if K is None:
                    K = k
                assert k == K, f"{path}: rows with {k} and {K} keypoints"
                vals = torch.tensor([float(x) for x in p[2:]]).reshape(K, 3)
                rows[(int(p[0]), int(float(p[1])))] = vals
        if K is None:
            return
        for i in range(n_frames):
            ann = frames[i]
            n = len(ann["id"])
            kp, vis = torch.zeros(n, K, 2), torch.zeros(n, K)
            for j, oid in enumerate(ann["id"].tolist()):
                v = rows.get((i + 1, oid))
                if v is not None:
                    kp[j], vis[j] = v[:, :2], v[:, 2]
            ann["keypoints"], ann["keypoints_vis"] = kp, vis


def make_unified_class(name: str, sub_dir: str):
    return type(name, (UnifiedMOT,), {"SUB_DIR": sub_dir, "__doc__": f"Unified-layout loader for {sub_dir}."})


class DatasetRegistry(dict):
    """dict of loader classes; unknown names fall back to a UnifiedMOT reading `<DATA_ROOT>/<name>`."""

    def __missing__(self, name):
        cls = make_unified_class(name, name)
        return cls


# Facts used by tools/gen_mix_config.py (frame rate -> frame-interval cap, licence warnings, label granularity).
#   fps:      native frame rate of the converted frames (None = varies per sequence, read from seqinfo.ini)
#   licence:  "own" | "non-commercial" | "OpenRAIL-M" | "unverified" (research licence not checked by me)
#   classes:  "full" = the unified vehicle classes are annotated, "partial" = a subset, "coarse" = class-less/one vehicle label
DATASET_INFO = {
    "Vehicles":       dict(fps=None, licence="own", classes="full"),
    "BDD100K_MOT":    dict(fps=5, licence="unverified (BDD100K licence)", classes="full"),
    "UA_DETRAC":      dict(fps=25, licence="unverified (research)", classes="partial"),
    "VisDrone_MOT":   dict(fps=30, licence="unverified (research)", classes="full"),
    "UAVDT":          dict(fps=30, licence="unverified (research)", classes="partial"),
    "FastTracker":    dict(fps=None, licence="OpenRAIL-M", classes="partial"),
    "CityFlow":       dict(fps=10, licence="unverified (AI City data licence)", classes="coarse"),
    "KITTI_Tracking": dict(fps=10, licence="non-commercial", classes="partial"),
    "Waymo":          dict(fps=10, licence="non-commercial", classes="coarse"),
    "NuScenes":       dict(fps=2, licence="non-commercial", classes="full"),
    "Argoverse2":     dict(fps=10, licence="non-commercial", classes="full"),
    "COCOKeypoints":  dict(fps=None, licence="depends on the source", classes="partial"),
    "Generic3D":      dict(fps=None, licence="depends on the source", classes="partial"),
}
