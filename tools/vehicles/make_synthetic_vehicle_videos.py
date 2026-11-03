#!/usr/bin/env python
"""Synthetic stand-in for the real data: mp4 videos + CVAT-video XML + index CSV, in the formats of the vehicle dataset.
Only for testing tools/vehicles/prepare_vehicle_dataset.py and the whole training pipeline on a machine without data.
    python tools/vehicles/make_synthetic_vehicle_videos.py --out /tmp/synth_vehicles
"""
import argparse
import csv
import os
import random
import shutil
import subprocess
import tempfile

import numpy as np
from PIL import Image

LABELS = ["bicycle", "car", "car_with_trailer", "van", "light_truck", "heavy_truck", "bus", "motorcycle"]
SITES = [("FR", "paris_a"), ("FR", "paris_b"), ("DE", "berlin_a"), ("DE", "berlin_b"), ("DE", "munich"), ("ES", "madrid")]


def xml_text(width, height, n, tracks, ignore_box=None):
    out = ['<?xml version="1.0" encoding="utf-8"?>', "<annotations>", "  <version>1.1</version>", "  <meta>", "    <task>",
           f"      <size>{n}</size>", "      <mode>interpolation</mode>", "      <start_frame>0</start_frame>",
           f"      <stop_frame>{n - 1}</stop_frame>", "      <frame_filter></frame_filter>",
           f"      <original_size><width>{width}</width><height>{height}</height></original_size>",
           "    </task>", "  </meta>"]
    for tid, (label, boxes) in enumerate(tracks):
        out.append(f'  <track id="{tid}" label="{label}" source="manual">')
        for f, x1, y1, x2, y2, outside, key in boxes:
            out.append(f'    <box frame="{f}" outside="{outside}" occluded="0" keyframe="{key}" xtl="{x1:.2f}" ytl="{y1:.2f}" '
                       f'xbr="{x2:.2f}" ybr="{y2:.2f}" z_order="0"></box>')
        out.append("  </track>")
    if ignore_box:
        tid = len(tracks)
        out.append(f'  <track id="{tid}" label="ignore_region" source="manual">')
        for f in range(n):
            x1, y1, x2, y2 = ignore_box
            out.append(f'    <box frame="{f}" outside="0" occluded="0" keyframe="{int(f == 0)}" xtl="{x1}" ytl="{y1}" xbr="{x2}" ybr="{y2}" z_order="0"></box>')
        out.append("  </track>")
    out.append("</annotations>")
    return "\n".join(out)


def make_video(vid, out_dir, rng, n_frames, w, h, n_obj, keyframes_only):
    tmp = tempfile.mkdtemp()
    tracks = []
    palette = {}
    objs = []
    for _ in range(n_obj):
        label = rng.choice(LABELS)
        bw, bh = rng.integers(int(0.15 * w), int(0.3 * w)), rng.integers(int(0.12 * h), int(0.25 * h))
        a = int(rng.integers(0, n_frames // 4))
        b = int(rng.integers(3 * n_frames // 4, n_frames))
        x0, y0 = rng.uniform(0, w - bw), rng.uniform(0, h - bh)
        vx, vy = rng.uniform(-2.5, 2.5), rng.uniform(-1.0, 1.0)
        col = tuple(int(c) for c in rng.integers(70, 255, 3))
        objs.append((label, bw, bh, a, b, x0, y0, vx, vy, col))
    ignore_box = (0, int(0.85 * h), int(0.4 * w), h)
    boxes_per_obj = [[] for _ in objs]
    for f in range(n_frames):
        img = np.full((h, w, 3), 45, dtype=np.uint8)
        img[ignore_box[1]:ignore_box[3], ignore_box[0]:ignore_box[2]] = 120
        for i, (label, bw, bh, a, b, x0, y0, vx, vy, col) in enumerate(objs):
            if a <= f <= b:
                x = float(np.clip(x0 + vx * (f - a), 0, w - bw)); y = float(np.clip(y0 + vy * (f - a), 0, h - bh))
                img[int(y):int(y + bh), int(x):int(x + bw)] = col
                img[int(y):int(y + bh) // 3 + int(y), int(x):int(x + bw)] = tuple(255 - c for c in col)
                key = int((f - a) % 6 == 0 or f == b)
                boxes_per_obj[i].append((f, x, y, x + bw, y + bh, 0, key))
        Image.fromarray(img).save(os.path.join(tmp, f"{f:05d}.png"))
    for (label, bw, bh, a, b, *_), boxes in zip(objs, boxes_per_obj):
        if keyframes_only:                      # CVAT interpolation export with only the keyframes present
            boxes = [bx for bx in boxes if bx[6]]
        boxes = boxes + [(b + 1, boxes[-1][1], boxes[-1][2], boxes[-1][3], boxes[-1][4], 1, 1)]   # "outside" marker
        tracks.append((label, boxes))
    mp4 = os.path.join(out_dir, f"{vid}.mp4")
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-framerate", "25", "-i", os.path.join(tmp, "%05d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "14", mp4], check=True)
    shutil.rmtree(tmp)
    with open(os.path.join(out_dir, f"{vid}.xml"), "w") as f:
        f.write(xml_text(w, h, n_frames, tracks, ignore_box))
    return objs


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--videos-per-site", type=int, default=2)
    ap.add_argument("--frames", type=int, default=64)
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--height", type=int, default=192)
    ap.add_argument("--objects", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    rng = np.random.default_rng(a.seed)
    os.makedirs(a.out, exist_ok=True)
    rows, vid = [], 1000
    for country, site in SITES:
        for k in range(a.videos_per_site):
            vid += 1
            make_video(vid, a.out, rng, a.frames, a.width, a.height, a.objects, keyframes_only=(vid % 3 == 0))
            rows.append(dict(id=vid, Video_file=f"{vid}.mp4", Anno_file=f"{vid}.xml", Time="day", Country=country,
                             Site=site, Orient="front", Calibration="no", Pov="high"))
    with open(os.path.join(a.out, "dataset.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"{len(rows)} videos in {a.out}")


if __name__ == "__main__":
    main()
