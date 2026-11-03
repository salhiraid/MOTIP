#!/usr/bin/env python
"""Draw the converted ground truth on a few frames to verify frame numbering, box scaling, ids and class mapping BEFORE
training:   python tools/vehicles/visualize_gt.py --seq-dir datasets/Vehicles/train/v1001 --frames 1 40 80 --out /tmp/vis
Colour = class, label = `<id>:<class>`; ignore regions (class 8 in gt.txt) are drawn as grey boxes."""
import argparse
import os

from PIL import Image, ImageDraw

CLASSES = ("bicycle", "car", "motorcycle", "bus", "truck")
COLORS = [(255, 160, 0), (60, 200, 60), (230, 60, 230), (60, 140, 255), (255, 70, 70)]


def render(seq_dir, frame, out_path):
    img = Image.open(os.path.join(seq_dir, "img1", f"{frame:08d}.jpg")).convert("RGB")
    d = ImageDraw.Draw(img)
    for line in open(os.path.join(seq_dir, "gt", "gt.txt")):
        f = line.strip().split(",")
        if int(f[0]) == frame and f[7] == "8":
            x, y, w, h = map(float, f[2:6])
            d.rectangle([x, y, x + w, y + h], outline=(150, 150, 150), width=2)
    for line in open(os.path.join(seq_dir, "gt", "gt_classes.txt")):
        f = line.strip().split(",")
        if int(f[0]) == frame:
            x, y, w, h = map(float, f[2:6])
            c = int(f[7])
            d.rectangle([x, y, x + w, y + h], outline=COLORS[c], width=2)
            d.text((x + 2, y + 2), f"{f[1]}:{CLASSES[c]}", fill=COLORS[c])
    img.save(out_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-dir", required=True)
    ap.add_argument("--frames", type=int, nargs="+", default=[1])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for fr in a.frames:
        p = os.path.join(a.out, f"{os.path.basename(a.seq_dir.rstrip('/'))}_{fr:05d}.png")
        render(a.seq_dir, fr, p)
        print(p)
