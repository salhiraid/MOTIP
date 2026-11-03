# Synthetic DanceTrack-format dataset (coloured rectangles that cross each other) for CPU smoke tests of the
# whole MOTIP2 pipeline (data -> training loop -> tracker -> TrackEval). NOT a benchmark: metrics on it are
# meaningless; it only proves the plumbing runs.
#   python tools/make_synthetic_dancetrack.py --out ./datasets_synth --train-seqs 3 --val-seqs 1 --frames 80
import argparse
import os
import random

import numpy as np
from PIL import Image


def make_sequence(seq_dir, n_frames, h, w, n_obj, rng):
    os.makedirs(os.path.join(seq_dir, "img1"), exist_ok=True)
    os.makedirs(os.path.join(seq_dir, "gt"), exist_ok=True)
    colors = [tuple(int(c) for c in rng.integers(60, 255, 3)) for _ in range(n_obj)]
    size = rng.integers(int(0.12 * h), int(0.28 * h), size=(n_obj, 2))          # (bh, bw)
    pos = np.stack([rng.uniform(0, h - size[:, 0]), rng.uniform(0, w - size[:, 1])], 1)
    vel = rng.uniform(-6, 6, size=(n_obj, 2))
    rows = []
    for f in range(n_frames):
        img = np.full((h, w, 3), 40, dtype=np.uint8)
        for i in range(n_obj):
            y, x = pos[i]
            bh, bw = size[i]
            y0, x0 = int(y), int(x)
            img[y0:y0 + bh, x0:x0 + bw] = colors[i]
            img[y0:y0 + bh // 4, x0:x0 + bw] = (255 - colors[i][0], 255 - colors[i][1], 255 - colors[i][2])  # texture
            rows.append(f"{f + 1},{i + 1},{x0},{y0},{bw},{bh},1,1,1")
        Image.fromarray(img).save(os.path.join(seq_dir, "img1", f"{f + 1:08d}.jpg"), quality=92)
        pos += vel
        for i in range(n_obj):                              # bounce on borders
            for d, lim in ((0, h - size[i, 0]), (1, w - size[i, 1])):
                if pos[i, d] < 0 or pos[i, d] > lim:
                    vel[i, d] *= -1
                    pos[i, d] = np.clip(pos[i, d], 0, lim)
    with open(os.path.join(seq_dir, "gt", "gt.txt"), "w") as f:
        f.write("\n".join(rows) + "\n")
    with open(os.path.join(seq_dir, "seqinfo.ini"), "w") as f:
        f.write(f"[Sequence]\nname={os.path.basename(seq_dir)}\nimDir=img1\nframeRate=20\n"
                f"seqLength={n_frames}\nimWidth={w}\nimHeight={h}\nimExt=.jpg\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--train-seqs", type=int, default=3)
    ap.add_argument("--val-seqs", type=int, default=1)
    ap.add_argument("--frames", type=int, default=80)
    ap.add_argument("--height", type=int, default=192)
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--objects", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    root = os.path.join(a.out, "DanceTrack")
    for split, n in (("train", a.train_seqs), ("val", a.val_seqs)):
        names = [f"synth{split}{i:04d}" for i in range(n)]
        for name in names:
            make_sequence(os.path.join(root, split, name), a.frames, a.height, a.width, a.objects, rng)
        with open(os.path.join(root, f"{split}_seqmap.txt"), "w") as f:
            f.write("name\n" + "\n".join(names) + "\n")
    print("written to", root)
