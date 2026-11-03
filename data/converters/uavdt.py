"""UAVDT (drone, car/truck/bus) MOT benchmark -> unified layout.  50 sequences: the benchmark's 30 train / 20 test
sequences become `train` / `val` (the test sequences have public annotations).

Expected input:
    <root>/UAV-benchmark-M/<M0101>/img000001.jpg ...
    <root>/UAV-benchmark-MOTD_v1.0/GT/M0101_gt_whole.txt  [+ M0101_gt_ignore.txt]
gt_whole row: frame,target_id,left,top,width,height,out_of_view,occlusion,object_category   (1 car, 2 truck, 3 bus)
"""
import os
from collections import Counter

from .common import Sequence, map_boxes, sorted_listdir
from data.class_maps import IGNORE

NAME, DIR, FPS = "uavdt", "UAVDT", 30.0
LICENSE = "UAVDT: research use (sites.google.com/view/grli-uavdt) - licence not verified here."
DEFAULT_MAP = {1: "car", 2: "truck", 3: "bus", "__ignore__": IGNORE}
# ASSUMPTION (from memory of the benchmark; override with --test-seqs FILE, one sequence name per line): the official
# MOT test split. The converter prints how many sequences it assigned to val.
TEST_SEQS = ("M0203 M0205 M0208 M0209 M0403 M0601 M0602 M0606 M0701 M0801 M0802 M1001 M1004 M1007 M1009 M1101 "
             "M1301 M1302 M1303 M1401").split()


def add_args(ap):
    ap.add_argument("--root", required=True)
    ap.add_argument("--test-seqs", default=None, help="file with the sequence names to use as val")


def sequences(a, class_map, class_set):
    unknown = Counter()
    class_map = {(int(k) if isinstance(k, str) and k.isdigit() else k): v for k, v in class_map.items()}
    test = set(open(a.test_seqs).read().split()) if a.test_seqs else set(TEST_SEQS)
    img_root = os.path.join(a.root, "UAV-benchmark-M")
    gt_root = os.path.join(a.root, "UAV-benchmark-MOTD_v1.0", "GT")
    names = [n for n in sorted_listdir(img_root) if os.path.isdir(os.path.join(img_root, n))]
    print(f"[uavdt] {len(names)} sequences, {sum(n in test for n in names)} assigned to val")
    for name in names:
        sdir = os.path.join(img_root, name)
        paths = [os.path.join(sdir, f) for f in sorted_listdir(sdir, (".jpg",))]
        seq = Sequence(name=name, split="val" if name in test else "train", images=paths, fps=FPS)
        with open(os.path.join(gt_root, f"{name}_gt_whole.txt")) as f:
            for line in f:
                p = line.strip().split(",")
                if len(p) < 9:
                    continue
                x, y, w, h = map(float, p[2:6])
                map_boxes(class_map, class_set, int(p[8]), int(p[1]), int(p[0]), x, y, w, h, 1.0, seq, unknown)
        ign = os.path.join(gt_root, f"{name}_gt_ignore.txt")
        if os.path.exists(ign):
            with open(ign) as f:
                for line in f:
                    p = line.strip().split(",")
                    if len(p) >= 6:
                        x, y, w, h = map(float, p[2:6])
                        map_boxes(class_map, class_set, "__ignore__", 0, int(p[0]), x, y, w, h, 1.0, seq, unknown)
        if unknown:
            seq.meta["warnings"] = [f"unmapped labels dropped: {dict(unknown)}"]
        yield seq
