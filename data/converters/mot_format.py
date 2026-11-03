"""Any dataset already in MOT-Challenge layout (frames + gt.txt with a class column) -> unified layout.  Used for the
FastTracker benchmark (12 CCTV videos, 9 classes, MOT format) and for any other MOT-format data you own.

Accepted input layouts (auto-detected):
    <root>/<split>/<seq>/{img1/*.jpg, gt/gt.txt, seqinfo.ini}     split folders train/val/test(/...) present, or
    <root>/<seq>/{img1/*.jpg, gt/gt.txt, seqinfo.ini}             flat: whole sequences are assigned to train/val
gt.txt row: frame,id,x,y,w,h,conf,class,visibility  (conf == 0 rows are skipped, as in MOT).

The class column must be mapped with --class-names: a JSON {"<class id>": "<unified class name>" | "distractor" |
"ignore" | null}.  The converter prints the class-id histogram with --dry-run so you can write that file.  For the
FastTracker benchmark I could NOT verify the 9 class ids offline: read the dataset card, then write the JSON.
"""
import json
import os
import random
from collections import Counter
from configparser import ConfigParser

from .common import Sequence, map_boxes, sorted_listdir

NAME, DIR, FPS = "fasttracker", "FastTracker", 25.0
LICENSE = "FastTracker benchmark: OpenRAIL-M (use-based restrictions, see the licence text)."
ANNOTATED = "data"     # see prepare.py: [] = coarse/class-less labels (no class supervision), "data" = classes present
DEFAULT_MAP = {}


def add_args(ap):
    ap.add_argument("--root", required=True)
    ap.add_argument("--class-names", default=None, help="JSON mapping class id -> unified class name (required)")
    ap.add_argument("--single-class", default=None, help="map every object to this class name (class-less data)")
    ap.add_argument("--val-seqs", nargs="*", default=None, help="flat layout: sequences used as val")
    ap.add_argument("--val-frac", type=float, default=0.25, help="flat layout without --val-seqs: fraction of whole sequences")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fps", type=float, default=0, help="override frameRate (0 = seqinfo.ini or default)")
    ap.add_argument("--split-map", nargs="*", default=[], metavar="SRC=DST", help="rename split folders, e.g. test=val")


def _find_sequences(root):
    """-> list of (split or None, seq_name, seq_dir)"""
    out = []
    for d in sorted_listdir(root):
        p = os.path.join(root, d)
        if not os.path.isdir(p):
            continue
        if os.path.isdir(os.path.join(p, "img1")):
            out.append((None, d, p))
        else:
            for s in sorted_listdir(p):
                q = os.path.join(p, s)
                if os.path.isdir(os.path.join(q, "img1")):
                    out.append((d, s, q))
    return out


def histogram(root):
    c = Counter()
    for _, _, d in _find_sequences(root):
        with open(os.path.join(d, "gt", "gt.txt")) as f:
            for line in f:
                p = line.strip().split(",")
                if len(p) > 7:
                    c[p[7].strip()] += 1
    return dict(c)


def sequences(a, class_map, class_set):
    unknown = Counter()
    if a.class_names:
        class_map = {str(k): v for k, v in json.load(open(a.class_names)).items()}
    elif a.single_class:
        class_map = None
    else:
        raise SystemExit(f"--class-names (JSON class id -> name) or --single-class is required. "
                         f"Class ids found in the gt files (id: #rows): {histogram(a.root)}")
    found = _find_sequences(a.root)
    renames = dict(s.split("=") for s in a.split_map)
    flat = [f for f in found if f[0] is None]
    if flat and a.val_seqs is None:
        rng = random.Random(a.seed)
        names = sorted(n for _, n, _ in flat)
        rng.shuffle(names)
        val = set(names[:max(1, int(round(a.val_frac * len(names))))]) if len(names) > 1 else set()
    else:
        val = set(a.val_seqs or [])
    for split, name, d in found:
        ini = ConfigParser()
        ini.read(os.path.join(d, "seqinfo.ini"))
        sec = ini["Sequence"] if "Sequence" in ini else {}
        img_dir = os.path.join(d, "img1")
        paths = [os.path.join(img_dir, f) for f in sorted_listdir(img_dir, (".jpg", ".jpeg", ".png"))]
        fps = a.fps or float(sec.get("frameRate", 0) or 0) or FPS
        sp = renames.get(split, split) if split else ("val" if name in val else "train")
        seq = Sequence(name=name, split=sp, images=paths, fps=fps)
        with open(os.path.join(d, "gt", "gt.txt")) as f:
            for line in f:
                p = line.strip().split(",")
                if len(p) < 6:
                    continue
                if len(p) > 6 and float(p[6]) == 0:                 # MOT "ignore" rows
                    continue
                x, y, w, h = map(float, p[2:6])
                vis = float(p[8]) if len(p) > 8 else 1.0
                if a.single_class:
                    seq.boxes.append((int(p[0]), int(float(p[1])), _cls_index(a.single_class, class_set), x, y, w, h, vis))
                else:
                    map_boxes(class_map, class_set, p[7].strip() if len(p) > 7 else "0", int(float(p[1])), int(p[0]),
                              x, y, w, h, vis, seq, unknown)
        if unknown:
            seq.meta["warnings"] = [f"unmapped class ids dropped: {dict(unknown)}"]
        yield seq


def _cls_index(name, class_set):
    from data.class_maps import class_names
    names = class_names(class_set)
    if name not in names:
        raise SystemExit(f"--single-class {name!r} is not in class set {class_set} {names}")
    return names.index(name)
