"""CityFlow (AI City Challenge, multi-camera vehicle tracking, 10 fps, 1080p videos) -> unified layout.

Expected input (CityFlow v1 track-1/3 download):
    <root>/{train,validation}/S0x/c0yy/{vdo.avi, gt/gt.txt}      gt row: frame,ID,left,top,width,height,1,-1,-1,-1
One sequence per camera; frames are extracted with ffmpeg into <out>/CityFlow/_frames and linked/copied from there.
Split by SCENE (site-disjoint): train = S01 S03 S04, val = S02 S05 (the official validation scenes).

CAVEATS (read before using it for detector training):
  * CityFlow labels are class-less: every object is mapped to --class (default `car`), buses/trucks included.
  * Only vehicles that appear in >= 2 cameras are annotated; single-camera vehicles are unlabelled -> missing positives.
Hence it is fine for the ID-decoder side but mixing it into DETECTOR training with a high weight will hurt recall.
"""
import os
import subprocess
from collections import Counter

from .common import Sequence, sorted_listdir
from data.class_maps import class_names

NAME, DIR, FPS = "cityflow", "CityFlow", 10.0
LICENSE = "CityFlow / AI City Challenge: research use under the AI City data licence (registration required)."
ANNOTATED = []     # see prepare.py: [] = coarse/class-less labels (no class supervision), "data" = classes present
DEFAULT_MAP = {}


def add_args(ap):
    ap.add_argument("--root", required=True)
    ap.add_argument("--class", dest="cls", default="car", help="unified class given to every CityFlow vehicle")
    ap.add_argument("--train-scenes", nargs="*", default=["S01", "S03", "S04"])
    ap.add_argument("--val-scenes", nargs="*", default=["S02", "S05"])
    ap.add_argument("--video", default="vdo.avi")


def extract(video, out_dir, short=0):
    os.makedirs(out_dir, exist_ok=True)
    if not [f for f in os.listdir(out_dir) if f.endswith(".jpg")]:
        subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", video, "-q:v", "2", "-start_number", "1",
                        os.path.join(out_dir, "%08d.jpg")], check=True)
    return [os.path.join(out_dir, f) for f in sorted_listdir(out_dir, (".jpg",))]


def sequences(a, class_map, class_set):
    names = class_names(class_set)
    if a.cls not in names:
        raise SystemExit(f"--class {a.cls!r} not in class set {names}")
    cls = names.index(a.cls)
    for folder in ("train", "validation"):
        base = os.path.join(a.root, folder)
        if not os.path.isdir(base):
            continue
        for scene in sorted_listdir(base):
            split = "train" if scene in a.train_scenes else "val" if scene in a.val_scenes else None
            if split is None:
                continue
            for cam in sorted_listdir(os.path.join(base, scene)):
                d = os.path.join(base, scene, cam)
                if not os.path.exists(os.path.join(d, a.video)):
                    continue
                frames_dir = os.path.join(a.out, DIR, "_frames", scene, cam)
                paths = extract(os.path.join(d, a.video), frames_dir) if not a.dry_run else []
                seq = Sequence(name=f"{scene}_{cam}", split=split, images=paths, fps=FPS)
                if a.dry_run:                      # no frame extraction: only count frames from the labels
                    n = 0
                with open(os.path.join(d, "gt", "gt.txt")) as f:
                    for line in f:
                        p = line.strip().split(",")
                        if len(p) >= 6:
                            seq.boxes.append((int(p[0]), int(p[1]), cls, *map(float, p[2:6]), 1.0))
                if a.dry_run:
                    seq.images = [None] * max((b[0] for b in seq.boxes), default=0)
                    seq.width, seq.height = 1920, 1080
                yield seq
