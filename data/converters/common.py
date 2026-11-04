"""Shared machinery of the dataset converters: one `Sequence` description in -> one unified MOT sequence folder out
(layout: data/unified_mot.py), plus split files, statistics and the dictionary-size recommendation.

A converter only has to (1) read its dataset's native annotation format, (2) map native labels to the unified classes
(`data/class_maps.py`), (3) yield `Sequence` objects.  Everything else (image linking/re-encoding/resizing, box
clipping, de-duplication, gt/gt_classes/seqinfo/seqmap files, valmini, report) lives here so all datasets behave the same.
"""
import argparse
import json
import math
import os
import shutil
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from data.class_maps import CLASS_SETS, class_names, resolve_label, IGNORE, DISTRACTOR  # noqa: E402

DISTRACTOR_CLASS = 8          # TrackEval MOT-Challenge distractor class
IGNORE_ID_BASE = 1_000_000    # ids of ignore regions (never collide with target ids)


@dataclass
class Sequence:
    """One video clip. All coordinates are in SOURCE pixels; frames are 1-indexed positions in `images`.

    images:      list of image file paths (or JPEG/PNG `bytes`), in temporal order, contiguous frames.
    boxes:       [(frame, track_id, cls_index, x, y, w, h, visibility)]  targets (cls_index in the chosen class set)
    distractors: [(frame, track_id, x, y, w, h)]  objects of a class outside the set: evaluation ignores matches
    ignore:      [(frame, x, y, w, h)]            crowd / don't-care regions
    """
    name: str
    split: str
    images: list
    fps: float
    boxes: list = field(default_factory=list)
    distractors: list = field(default_factory=list)
    ignore: list = field(default_factory=list)
    width: int = 0
    height: int = 0
    meta: dict = field(default_factory=dict)
    keypoints: list = field(default_factory=list)   # [(frame, track_id, [x, y, v, x, y, v, ...])]  source pixels, v = GT visibility value
    static: bool = False                            # independent still image (seqinfo isStatic=1): sampled as a 1-frame "clip"


class IdMap:
    """native (str/int) track id -> small positive int, stable per sequence."""

    def __init__(self):
        self.m = {}

    def __call__(self, key):
        if key not in self.m:
            self.m[key] = len(self.m) + 1
        return self.m[key]


def add_common_args(ap: argparse.ArgumentParser):
    ap.add_argument("--out", required=True, help="datasets root; creates <out>/<DatasetDir>")
    ap.add_argument("--class-set", default="vehicles5", choices=sorted(CLASS_SETS),
                    help="unified detector classes (data/class_maps.py). Labels outside the set become evaluation "
                         "distractors (never training targets)")
    ap.add_argument("--class-map", default=None,
                    help="JSON {native_label: unified class name | 'ignore' | 'distractor' | null} overriding the defaults")
    ap.add_argument("--image-mode", default="link", choices=["link", "copy", "jpeg"],
                    help="link = symlink the original frames (no disk use, original resolution); copy = copy them; "
                         "jpeg = decode, optionally resize, re-encode as JPEG (use for huge frames / PNG / video bytes)")
    ap.add_argument("--resize-short", type=int, default=0, help="jpeg mode: shorter side (0 = keep); boxes are rescaled")
    ap.add_argument("--max-long", type=int, default=0, help="jpeg mode: longer side cap (0 = none)")
    ap.add_argument("--jpeg-quality", type=int, default=92)
    ap.add_argument("--min-frames", type=int, default=10, help="sequences shorter than this are skipped")
    ap.add_argument("--min-box", type=float, default=2.0, help="drop boxes thinner than this (output pixels)")
    ap.add_argument("--valmini", type=int, default=10, help="#val sequences in the small in-training eval split (0 = none)")
    ap.add_argument("--window", type=int, default=30, help="T used for the dictionary-size recommendation")
    ap.add_argument("--workers", type=int, default=4, help="threads for image copying / re-encoding")
    ap.add_argument("--limit", type=int, default=0, help="debug: only the first N sequences per split")
    ap.add_argument("--dry-run", action="store_true", help="read annotations only, print statistics, write nothing")
    ap.add_argument("--overwrite", action="store_true")
    return ap


def load_class_map(defaults: dict, path: str | None) -> dict:
    cm = dict(defaults)
    if path:
        cm.update(json.load(open(path)))
    return cm


def out_size(w, h, short, long_):
    if not short and not long_:
        return w, h
    s = min(short / min(w, h) if short else 1e9, long_ / max(w, h) if long_ else 1e9)
    return max(2, int(round(w * s))), max(2, int(round(h * s)))


def _image_size(img):
    from PIL import Image
    import io
    im = Image.open(io.BytesIO(img)) if isinstance(img, (bytes, bytearray)) else Image.open(img)
    return im.size


def _emit_image(job):
    src, dst, mode, size, quality = job
    from PIL import Image
    import io
    if mode in ("link", "copy") and not isinstance(src, (bytes, bytearray)):
        if os.path.lexists(dst):
            os.remove(dst)
        if mode == "link":
            os.symlink(os.path.abspath(src), dst)
        else:
            shutil.copyfile(src, dst)
        return
    im = Image.open(io.BytesIO(src)) if isinstance(src, (bytes, bytearray)) else Image.open(src)
    if mode == "copy" and isinstance(src, (bytes, bytearray)):          # keep original bytes (assumed JPEG)
        open(dst, "wb").write(src)
        return
    im = im.convert("RGB")
    if size is not None and size != im.size:
        im = im.resize(size, Image.BILINEAR if size[0] > im.size[0] else Image.LANCZOS)
    im.save(dst, quality=quality)


def _ext_of(images, mode):
    if mode == "jpeg":
        return ".jpg"
    first = images[0]
    if isinstance(first, (bytes, bytearray)):
        return ".jpg"
    return os.path.splitext(first)[1].lower() or ".jpg"


def _clip(x, y, w, h, W, H):
    x1, y1, x2, y2 = max(0.0, x), max(0.0, y), min(W, x + w), min(H, y + h)
    return x1, y1, x2 - x1, y2 - y1


def build_output_rows(seq: Sequence, scale, out_wh, min_box):
    """Scale + clip + de-duplicate. Returns (rows, distractor_rows, ignore_rows) in output pixels."""
    sx, sy = scale
    W, H = out_wh
    n = len(seq.images)
    rows, seen = [], set()
    for fr, tid, cls, x, y, w, h, vis in sorted(seq.boxes, key=lambda r: (r[0], r[1])):
        if not (1 <= fr <= n) or (fr, tid) in seen:
            continue
        x, y, w, h = _clip(x * sx, y * sy, w * sx, h * sy, W, H)
        if w < min_box or h < min_box:
            continue
        seen.add((fr, tid))
        rows.append((fr, tid, cls, x, y, w, h, vis))
    drows = []
    for fr, tid, x, y, w, h in seq.distractors:
        if not (1 <= fr <= n) or (fr, tid) in seen:
            continue
        x, y, w, h = _clip(x * sx, y * sy, w * sx, h * sy, W, H)
        if w >= min_box and h >= min_box:
            drows.append((fr, tid, x, y, w, h))
    kept = {(r[0], r[1]) for r in rows}
    krows = []
    for fr, tid, flat in seq.keypoints:
        if (fr, tid) not in kept:
            continue
        pts = []
        for j in range(0, len(flat), 3):
            x, y, v = flat[j] * sx, flat[j + 1] * sy, flat[j + 2]
            if not (0.0 <= x <= W and 0.0 <= y <= H):
                v = 0.0                                  # outside the image after scaling/cropping -> not labelled
            pts += [x, y, v]
        krows.append((fr, tid, pts))
    irows = []
    for i, (fr, x, y, w, h) in enumerate(seq.ignore):
        if not (1 <= fr <= n):
            continue
        x, y, w, h = _clip(x * sx, y * sy, w * sx, h * sy, W, H)
        if w >= min_box and h >= min_box:
            irows.append((fr, IGNORE_ID_BASE + i, x, y, w, h))
    return rows, drows, irows, krows


def write_gt(seq_dir, rows, drows, irows, n_frames, out_wh, fps, ext, name, krows=(), static=False):
    os.makedirs(os.path.join(seq_dir, "gt"), exist_ok=True)
    W, H = out_wh
    f = lambda v: f"{v:.2f}"
    with open(os.path.join(seq_dir, "gt", "gt.txt"), "w") as g, \
            open(os.path.join(seq_dir, "gt", "gt_classes.txt"), "w") as gc:
        for fr, tid, cls, x, y, w, h, vis in rows:
            g.write(f"{fr},{tid},{f(x)},{f(y)},{f(w)},{f(h)},1,1,{vis}\n")           # evaluation: class-agnostic
            gc.write(f"{fr},{tid},{f(x)},{f(y)},{f(w)},{f(h)},1,{cls},{vis}\n")      # training: real class index
        for fr, tid, x, y, w, h in list(drows) + list(irows):                        # distractors / ignore -> class 8
            g.write(f"{fr},{tid},{f(x)},{f(y)},{f(w)},{f(h)},1,{DISTRACTOR_CLASS},1\n")
    if krows:
        with open(os.path.join(seq_dir, "gt", "gt_keypoints.txt"), "w") as gk:
            for fr, tid, pts in krows:
                gk.write(f"{fr},{tid}," + ",".join(f"{v:.2f}" if j % 3 != 2 else f"{v:g}" for j, v in enumerate(pts)) + "\n")
    with open(os.path.join(seq_dir, "seqinfo.ini"), "w") as f_:
        f_.write(f"[Sequence]\nname={name}\nimDir=img1\nframeRate={int(round(fps)) or 1}\nseqLength={n_frames}\n"
                 f"imWidth={W}\nimHeight={H}\nimExt={ext}\n" + ("isStatic=1\n" if static else ""))


def convert_sequence(seq: Sequence, root: str, a, pool: ThreadPoolExecutor | None):
    """Write one sequence. Returns a record for the report (never raises: errors become warnings)."""
    rec = {"name": seq.name, "split": seq.split, "ok": False, "warnings": list(seq.meta.get("warnings", []))}
    try:
        n = len(seq.images)
        if n < a.min_frames:
            rec["warnings"].append(f"{n} frames (< {a.min_frames}) -> skipped")
            return rec
        mode = a.image_mode
        if (a.resize_short or a.max_long) and mode != "jpeg":
            rec["warnings"].append("resize requested -> image-mode forced to jpeg")
            mode = "jpeg"
        if isinstance(seq.images[0], (bytes, bytearray)) and mode == "link":
            mode = "copy"
        W, H = (seq.width, seq.height) if seq.width else _image_size(seq.images[0])
        if not a.dry_run and seq.width:                        # verify a declared size against the first image
            W0, H0 = _image_size(seq.images[0])
            if (W0, H0) != (W, H):
                rec["warnings"].append(f"declared size {W}x{H} != image {W0}x{H0}; using the image size")
                W, H = W0, H0
        out_wh = out_size(W, H, a.resize_short, a.max_long) if mode == "jpeg" else (W, H)
        scale = (out_wh[0] / W, out_wh[1] / H)
        rows, drows, irows, krows = build_output_rows(seq, scale, out_wh, a.min_box)
        if not rows:
            rec["warnings"].append("no target boxes -> skipped")
            return rec
        per_frame = defaultdict(set)
        for r in rows:
            per_frame[r[0]].add(r[1])
        rec.update(frames=n, boxes=len(rows), distractors=len(drows), ignore=len(irows), fps=seq.fps, out_wh=out_wh,
                   classes=dict(Counter(r[2] for r in rows)), per_frame_ids=[sorted(per_frame.get(i, ())) for i in range(1, n + 1)])
        if not a.dry_run:
            seq_dir = os.path.join(root, seq.split, seq.name)
            marker = os.path.join(seq_dir, ".done")
            if os.path.exists(marker) and not a.overwrite:
                rec["warnings"].append("already converted (kept)")
            else:
                if os.path.isdir(seq_dir):
                    shutil.rmtree(seq_dir)
                img_dir = os.path.join(seq_dir, "img1")
                os.makedirs(img_dir)
                ext = _ext_of(seq.images, mode)
                size = out_wh if mode == "jpeg" else None
                jobs = [(src, os.path.join(img_dir, f"{i + 1:08d}{ext}"), mode, size, a.jpeg_quality)
                        for i, src in enumerate(seq.images)]
                list(pool.map(_emit_image, jobs)) if pool else [ _emit_image(j) for j in jobs]
                write_gt(seq_dir, rows, drows, irows, n, out_wh, seq.fps, ext, seq.name, krows, seq.static)
                open(marker, "w").write("ok\n")
        rec["ok"] = True
    except Exception as e:                                     # noqa: BLE001 - report, keep going
        rec["warnings"].append(f"ERROR {type(e).__name__}: {e}")
    return rec


# ----------------------------------------------------------------------------------------------------------------------
def window_unique_counts(per_frame_ids, T):
    step = max(1, T // 2)
    return [len(set().union(*per_frame_ids[i:i + T])) for i in range(0, max(1, len(per_frame_ids) - T + 1), step)]


def pct(a, p):
    a = sorted(a)
    return a[min(len(a) - 1, int(math.ceil(p / 100 * len(a))) - 1)] if a else 0


def summarize(recs, names, T):
    out = {}
    ok = [r for r in recs if r.get("ok")]
    for split in sorted({r["split"] for r in ok}):
        rs = [r for r in ok if r["split"] == split]
        per_frame = [len(ids) for r in rs for ids in r["per_frame_ids"]]
        windows = [c for r in rs for c in window_unique_counts(r["per_frame_ids"], T)]
        cls = Counter()
        for r in rs:
            cls.update({names[int(k)] if int(k) < len(names) else str(k): v for k, v in r["classes"].items()})
        out[split] = {"sequences": len(rs), "frames": sum(r["frames"] for r in rs), "boxes": sum(r["boxes"] for r in rs),
                      "objects_per_frame": {"mean": round(sum(per_frame) / max(1, len(per_frame)), 2),
                                            "p99": pct(per_frame, 99), "max": max(per_frame, default=0)},
                      f"unique_ids_per_{T}_frame_window": {"p99": pct(windows, 99), "max": max(windows, default=0)},
                      "classes": dict(cls), "fps": sorted({round(r["fps"]) for r in rs}),
                      "out_sizes": sorted({"x".join(map(str, r["out_wh"])) for r in rs})[:6]}
    tr = out.get("train", {}).get(f"unique_ids_per_{T}_frame_window", {})
    out["recommended_NUM_ID_VOCABULARY"] = max(50, int(math.ceil(2.5 * tr.get("p99", 0) / 50.0)) * 50)
    return out


def run_conversion(dir_name: str, sequences, a, class_set: str, notes: list[str] | None = None, annotated=None):
    """sequences: iterable of Sequence (any split). Writes <out>/<dir_name>/..., prints the report, returns it.
    annotated: names of the unified classes this dataset ANNOTATES (list), "data" = the classes actually present, or None
    = all.  Written to annotated_classes.json, used to mask the classification loss of unannotated classes."""
    names = class_names(class_set)
    root = os.path.join(a.out, dir_name)
    pool = ThreadPoolExecutor(max(1, a.workers)) if (a.workers > 1 and not a.dry_run) else None
    recs, per_split = [], Counter()
    for seq in sequences:
        if a.limit and per_split[seq.split] >= a.limit:
            continue
        per_split[seq.split] += 1
        recs.append(convert_sequence(seq, root, a, pool))
        if len(recs) % 20 == 0:
            print(f"  converted {len(recs)} sequences", flush=True)
    stats = summarize(recs, names, a.window)
    report = {"dataset": dir_name, "class_set": list(names), "stats": stats, "notes": notes or [], "args": vars(a),
              "warnings": {f"{r['split']}/{r['name']}": r["warnings"] for r in recs if r["warnings"]},
              "skipped": [f"{r['split']}/{r['name']}" for r in recs if not r.get("ok")]}
    print(json.dumps(stats, indent=1))
    print(f"{len(recs)} sequences seen, {sum(r.get('ok', False) for r in recs)} converted, "
          f"warnings in {len(report['warnings'])}, skipped {len(report['skipped'])}")
    for n in notes or []:
        print("NOTE:", n)
    print(f"==> NUM_ID_VOCABULARY (K) >= {stats['recommended_NUM_ID_VOCABULARY']} for this dataset "
          f"(p99 unique ids per {a.window}-frame window x 2.5, multiple of 50); frame rates {sorted({round(r['fps']) for r in recs if r.get('ok')})}")
    if a.dry_run:
        return report
    os.makedirs(root, exist_ok=True)
    by_split = defaultdict(list)
    for r in recs:
        if r.get("ok"):
            by_split[r["split"]].append(r["name"])
    for sp, lst in by_split.items():
        with open(os.path.join(root, f"{sp}_seqmap.txt"), "w") as f:
            f.write("name\n" + "\n".join(sorted(lst)) + "\n")
    if a.valmini and by_split.get("val"):
        mini = sorted(by_split["val"])
        step = max(1, len(mini) // a.valmini)          # evenly spread over the (sorted) val sequences
        mini = mini[::step][:a.valmini]
        mini_dir = os.path.join(root, "valmini")
        shutil.rmtree(mini_dir, ignore_errors=True)
        os.makedirs(mini_dir)
        for s in mini:
            os.symlink(os.path.join("..", "val", s), os.path.join(mini_dir, s))
        with open(os.path.join(root, "valmini_seqmap.txt"), "w") as f:
            f.write("name\n" + "\n".join(mini) + "\n")
    if annotated == "data":
        annotated = sorted({c for sp in stats.values() if isinstance(sp, dict) for c in sp.get("classes", {})}, key=names.index)
    if annotated is not None:
        with open(os.path.join(root, "annotated_classes.json"), "w") as f:
            json.dump({"class_set": list(names), "annotated": list(annotated),
                       "annotated_indices": [names.index(n) for n in annotated]}, f, indent=1)
        missing = [n for n in names if n not in annotated]
        print(f"annotated classes: {list(annotated)}; NOT annotated (masked in the classification loss): {missing}")
    with open(os.path.join(root, "prepare_report.json"), "w") as f:
        json.dump(report, f, indent=1, default=str)
    print(f"written: {root}  ({', '.join(f'{k} {len(v)}' for k, v in sorted(by_split.items()))} sequences)")
    return report


def map_boxes(label_map, class_set, label, tid, fr, x, y, w, h, vis, seq: Sequence, unknown: Counter):
    """Route one native annotation into seq.boxes / seq.distractors / seq.ignore according to the label map."""
    if label not in label_map:
        unknown[label] += 1
        return
    r = resolve_label(label_map, label, class_set)
    if r is None:
        return
    if r == IGNORE:
        seq.ignore.append((fr, x, y, w, h))
    elif r == DISTRACTOR:
        seq.distractors.append((fr, tid, x, y, w, h))
    else:
        seq.boxes.append((fr, tid, r, x, y, w, h, vis))


def sorted_listdir(path, exts=None):
    names = sorted(os.listdir(path))
    return [n for n in names if not exts or n.lower().endswith(tuple(exts))]
