#!/usr/bin/env python
"""Convert a CSV-indexed collection of annotated vehicle VIDEOS (CVAT "video 1.1" XML tracks) into the MOT-style frame
dataset used by MOTIP2 (`data/vehicles.py`), with a train/val split that never puts the same site in both splits.

    python tools/vehicles/prepare_vehicle_dataset.py \
        --csv dataset.csv --videos-dir /data/videos --anno-dir /data/videos --out ./datasets \
        --val-frac 0.12 --resize-short 800 --max-long 1440 --workers 16

CSV columns: id,Video_file,Anno_file,Time,Country,Site,Orient,Calibration,Pov.  A video is `<videos-dir>/<Video_file>`
(or `<videos-dir>/<id>.<ext>` if that column is empty / not found); the annotation likewise in `--anno-dir`.

Output (`<out>/Vehicles/`):
    {train,val,valmini}/<seq>/img1/00000001.jpg ...  gt/gt.txt  gt/gt_classes.txt  seqinfo.ini
    {train,val,valmini}_seqmap.txt   split.csv   prepare_report.json
See data/vehicles.py for the exact file semantics.  Run with --dry-run first: it only reads CSV + XML, prints the split,
class histogram, #frames/GB estimate and the recommended ID-dictionary size, and writes nothing.
"""
import argparse
import csv
import json
import math
import os
import random
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from multiprocessing import Pool

CLASSES = ("bicycle", "car", "motorcycle", "bus", "truck")      # detector classes (index = position)
IGNORE = "ignore"
# CVAT label -> class index | IGNORE | None (drop). ASSUMPTION (confirm!): vans and cars-with-trailer count as `car`,
# every kind of truck as `truck`, buses (with/without trailer) as `bus`.
DEFAULT_CLASS_MAP = {
    "bicycle": 0,
    "car": 1, "car_with_trailer": 1, "van": 1, "van_with_trailer": 1,
    "motorcycle": 2,
    "bus": 3, "bus_with_trailer": 3,
    "light_truck": 4, "heavy_truck": 4, "truck_without_trailer": 4,
    "ignore_region": IGNORE,
}
VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".mpg", ".mpeg", ".webm", ".m4v", ".ts")
MB_PER_FRAME = 0.12            # rough JPEG size at 800x1422, only for the printed disk estimate


# ----------------------------------------------------------------------------------------------------------------------
# CSV / files
# ----------------------------------------------------------------------------------------------------------------------
def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = [{(k or "").strip(): (v or "").strip() for k, v in r.items()} for r in csv.DictReader(f)]
    need = {"id", "Video_file", "Anno_file", "Country", "Site"}
    missing = need - set(rows[0].keys()) if rows else need
    if missing:
        raise SystemExit(f"CSV is missing columns {sorted(missing)}; found {sorted(rows[0].keys()) if rows else '[]'}")
    return rows


def resolve(base_dir, value, vid, exts):
    """`<base>/<value>` if it exists, else `<base>/<id><ext>` for the known extensions, else None."""
    cands = []
    if value:
        cands += [value if os.path.isabs(value) else os.path.join(base_dir, value),
                  os.path.join(base_dir, os.path.basename(value))]
    cands += [os.path.join(base_dir, f"{vid}{e}") for e in exts]
    for c in cands:
        if os.path.isfile(c):
            return c
    return None


def site_key(row):
    """Videos of one physical site must stay together. (Country, Site) because site names may repeat across countries."""
    return (row.get("Country", "").lower(), row.get("Site", "").lower())


def make_split(rows, val_frac, seed):
    """Site-disjoint split, stratified by country: inside each country whole sites are moved to validation until
    ~val_frac of its videos are covered. Countries with a single site stay in train (cannot be split)."""
    rng = random.Random(seed)
    by_country = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_country[r.get("Country", "").lower()][site_key(r)].append(r["id"])
    split = {}
    for country, sites in sorted(by_country.items()):
        keys = sorted(sites)
        rng.shuffle(keys)
        n_videos = sum(len(v) for v in sites.values())
        target = val_frac * n_videos
        got = 0
        val_sites = set()
        if len(keys) >= 2:
            for k in keys:
                if got >= target or len(val_sites) >= len(keys) - 1:      # always keep >= 1 site in train
                    break
                # take a site only if it brings us closer to the target (avoid overshooting with huge sites)
                if abs(got + len(sites[k]) - target) <= abs(got - target) or got == 0:
                    val_sites.add(k)
                    got += len(sites[k])
        for k in keys:
            for vid in sites[k]:
                split[vid] = "val" if k in val_sites else "train"
    check_site_disjoint(rows, split)
    return split


def check_site_disjoint(rows, split):
    seen = defaultdict(set)
    for r in rows:
        seen[site_key(r)].add(split[r["id"]])
    bad = {k: v for k, v in seen.items() if len(v) > 1}
    assert not bad, f"site(s) present in both splits: {bad}"


def pick_valmini(rows, split, n):
    """`n` validation videos spread over as many different sites as possible (round robin over sites)."""
    per_site = defaultdict(list)
    for r in sorted(rows, key=lambda r: r["id"]):
        if split.get(r["id"]) == "val":
            per_site[site_key(r)].append(r["id"])
    out, i = [], 0
    sites = sorted(per_site)
    while len(out) < n and any(per_site.values()):
        s = sites[i % len(sites)]
        if per_site[s]:
            out.append(per_site[s].pop(0))
        i += 1
    return set(out)


# ----------------------------------------------------------------------------------------------------------------------
# CVAT XML
# ----------------------------------------------------------------------------------------------------------------------
def parse_cvat_xml(path, class_map):
    """Returns dict(width, height, start, stop, tracks=[{id,label,cls,boxes=[(frame,x1,y1,x2,y2,occluded,outside)]}],
    warnings=[...]). Only <track><box> shapes are used (CVAT video export); other shape types are counted and ignored."""
    root = ET.parse(path).getroot()
    warnings = []
    size = root.find("./meta/task/original_size")
    width = int(float(size.findtext("width"))) if size is not None else None
    height = int(float(size.findtext("height"))) if size is not None else None
    start = int(root.findtext("./meta/task/start_frame") or 0)
    stop = root.findtext("./meta/task/stop_frame")
    stop = int(stop) if stop is not None else None
    if (root.findtext("./meta/task/frame_filter") or "").strip():
        warnings.append(f"frame_filter={root.findtext('./meta/task/frame_filter')!r}: frame numbers may not match the video")
    tracks, unknown = [], Counter()
    for t in root.findall("track"):
        label = t.get("label")
        if label not in class_map:
            unknown[label] += 1
            continue
        cls = class_map[label]
        if cls is None:
            continue
        boxes = []
        for b in t.findall("box"):
            boxes.append((int(b.get("frame")), float(b.get("xtl")), float(b.get("ytl")), float(b.get("xbr")),
                          float(b.get("ybr")), int(b.get("occluded", "0")), int(b.get("outside", "0"))))
        boxes.sort()
        tracks.append({"id": int(t.get("id")), "label": label, "cls": cls, "boxes": boxes})
    for tag in ("polygon", "polyline", "points", "mask"):
        n = len(root.findall(f".//{tag}"))
        if n:
            warnings.append(f"{n} <{tag}> shapes ignored")
    if unknown:
        warnings.append(f"labels not in class map were dropped: {dict(unknown)}")
    return {"width": width, "height": height, "start": start, "stop": stop, "tracks": tracks, "warnings": warnings}


def expand_track(boxes):
    """Rows per frame for one track, outside=1 boxes removed, gaps between two visible boxes linearly interpolated
    (CVAT interpolation semantics; a no-op for full exports). Returns (frame -> (x1,y1,x2,y2,occ)), n_interpolated."""
    out, filled = {}, 0
    for i, (f, x1, y1, x2, y2, occ, outside) in enumerate(boxes):
        if outside:
            continue
        out[f] = (x1, y1, x2, y2, occ)
        if i + 1 < len(boxes) and not boxes[i + 1][6] and boxes[i + 1][0] - f > 1:
            g, a, b = boxes[i + 1][0], (x1, y1, x2, y2), boxes[i + 1][1:5]
            for m in range(f + 1, g):
                w = (m - f) / (g - f)
                out[m] = tuple(a[j] + w * (b[j] - a[j]) for j in range(4)) + (occ,)
                filled += 1
    return out, filled


def build_rows(parsed, stride, scale_xy, out_wh, min_box=2.0):
    """Annotated frame range + MOT rows. Frames are re-numbered 1..N after trimming to the annotated range."""
    per_track, filled = [], 0
    for tr in parsed["tracks"]:
        fr, n = expand_track(tr["boxes"])
        filled += n
        per_track.append((tr, fr))
    frames = [f for _, fr in per_track for f in fr]
    if not frames:
        return None
    first, last = min(frames), max(frames)
    sx, sy = scale_xy
    W, H = out_wh
    rows, n_degenerate = [], 0
    for tr, fr in per_track:
        for f, (x1, y1, x2, y2, occ) in fr.items():
            if (f - first) % stride:
                continue
            x1, x2 = max(0.0, min(W, x1 * sx)), max(0.0, min(W, x2 * sx))
            y1, y2 = max(0.0, min(H, y1 * sy)), max(0.0, min(H, y2 * sy))
            if x2 - x1 < min_box or y2 - y1 < min_box:
                n_degenerate += 1
                continue
            rows.append(((f - first) // stride + 1, tr["id"] + 1, tr["cls"], x1, y1, x2 - x1, y2 - y1, occ))
    rows.sort(key=lambda r: (r[0], r[1]))
    seen, dedup = set(), []
    for r in rows:                                 # one box per (frame, id)
        if (r[0], r[1]) not in seen:
            seen.add((r[0], r[1]))
            dedup.append(r)
    return {"first": first, "last": last, "rows": dedup, "filled": filled, "degenerate": n_degenerate}


def out_size(w, h, short, long_):
    if not short:
        return w, h
    s = min(short / min(w, h), long_ / max(w, h)) if long_ else short / min(w, h)
    return max(2, int(round(w * s))), max(2, int(round(h * s)))


# ----------------------------------------------------------------------------------------------------------------------
# video
# ----------------------------------------------------------------------------------------------------------------------
def probe_video(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                          "-show_entries", "stream=width,height,nb_frames,r_frame_rate", "-of", "json", path],
                         capture_output=True, text=True, check=True).stdout
    s = json.loads(out)["streams"][0]
    num, den = (s.get("r_frame_rate", "25/1").split("/") + ["1"])[:2]
    return {"width": int(s["width"]), "height": int(s["height"]), "frames": int(s.get("nb_frames") or 0),
            "fps": float(num) / max(1.0, float(den))}


def extract_frames(video, img_dir, first, last, stride, size, q):
    os.makedirs(img_dir, exist_ok=True)
    vf = f"select='between(n,{first},{last})*not(mod(n-{first},{stride}))',scale={size[0]}:{size[1]}:flags=area"
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", video, "-vf", vf,
                    "-fps_mode", "passthrough", "-start_number", "1", "-q:v", str(q), os.path.join(img_dir, "%08d.jpg")],
                   check=True)
    return len([f for f in os.listdir(img_dir) if f.endswith(".jpg")])


def write_sequence(seq_dir, rows, n_frames, size, fps, ignore_rows):
    os.makedirs(os.path.join(seq_dir, "gt"), exist_ok=True)
    W, H = size
    f = lambda v: f"{v:.2f}"
    with open(os.path.join(seq_dir, "gt", "gt.txt"), "w") as g, \
            open(os.path.join(seq_dir, "gt", "gt_classes.txt"), "w") as gc:
        for fr, oid, cls, x, y, w, h, occ in rows:
            vis = 0.5 if occ else 1.0
            g.write(f"{fr},{oid},{f(x)},{f(y)},{f(w)},{f(h)},1,1,{vis}\n")          # evaluation: class-agnostic
            gc.write(f"{fr},{oid},{f(x)},{f(y)},{f(w)},{f(h)},1,{cls},{vis}\n")     # training: real class
        for fr, oid, _c, x, y, w, h, _o in ignore_rows:                             # ignore regions -> distractor (8)
            g.write(f"{fr},{oid},{f(x)},{f(y)},{f(w)},{f(h)},1,8,1\n")
    with open(os.path.join(seq_dir, "seqinfo.ini"), "w") as f_:
        f_.write(f"[Sequence]\nname={os.path.basename(seq_dir)}\nimDir=img1\nframeRate={int(round(fps))}\n"
                 f"seqLength={n_frames}\nimWidth={W}\nimHeight={H}\nimExt=.jpg\n")


def process_video(job):
    """Worker: one video -> one sequence folder. Returns a record (never raises, errors are reported)."""
    rec = {"id": job["id"], "seq": job["seq"], "split": job["split"], "ok": False, "warnings": []}
    try:
        class_map = job["class_map"]
        parsed = parse_cvat_xml(job["anno"], class_map)
        rec["warnings"] += parsed["warnings"]
        veh = {"tracks": [t for t in parsed["tracks"] if t["cls"] != IGNORE]}
        ign = {"tracks": [t for t in parsed["tracks"] if t["cls"] == IGNORE]}
        if job["dry_run"]:
            vw, vh, fps = parsed["width"] or 1920, parsed["height"] or 1080, 25.0
        else:
            info = probe_video(job["video"])
            vw, vh, fps = info["width"], info["height"], info["fps"]
        xw, xh = parsed["width"] or vw, parsed["height"] or vh
        if (xw, xh) != (vw, vh):
            rec["warnings"].append(f"XML original_size {xw}x{xh} != video {vw}x{vh}; boxes are mapped XML->output size")
        out_wh = out_size(vw, vh, job["resize_short"], job["max_long"])
        scale = (out_wh[0] / xw, out_wh[1] / xh)
        built = build_rows(veh, job["stride"], scale, out_wh)
        if built is None:
            rec["warnings"].append("no vehicle boxes -> sequence skipped")
            return rec
        # ignore regions use the same frame trimming/renumbering
        ign_rows = []
        for t in ign["tracks"]:
            fr, _ = expand_track(t["boxes"])
            for f, (x1, y1, x2, y2, occ) in fr.items():
                if f < built["first"] or f > built["last"] or (f - built["first"]) % job["stride"]:
                    continue
                X1, X2 = max(0.0, min(out_wh[0], x1 * scale[0])), max(0.0, min(out_wh[0], x2 * scale[0]))
                Y1, Y2 = max(0.0, min(out_wh[1], y1 * scale[1])), max(0.0, min(out_wh[1], y2 * scale[1]))
                if X2 - X1 >= 2 and Y2 - Y1 >= 2:
                    ign_rows.append(((f - built["first"]) // job["stride"] + 1, 100000 + t["id"], 8, X1, Y1, X2 - X1, Y2 - Y1, 0))
        n_frames = (built["last"] - built["first"]) // job["stride"] + 1
        rec.update(frames=n_frames, objects=len(built["rows"]), fps=fps, out_wh=out_wh, filled=built["filled"],
                   classes=dict(Counter(r[2] for r in built["rows"])), rows=None)
        if n_frames < job["min_frames"]:
            rec["warnings"].append(f"only {n_frames} annotated frames (< {job['min_frames']}) -> skipped")
            rec["rows"] = None
            return rec
        # per-frame object counts / ids for the dictionary-size recommendation
        per_frame = defaultdict(set)
        for r in built["rows"]:
            per_frame[r[0]].add(r[1])
        rec["per_frame_ids"] = [sorted(per_frame.get(i, ())) for i in range(1, n_frames + 1)]
        if not job["dry_run"]:
            seq_dir = job["seq_dir"]
            marker = os.path.join(seq_dir, ".done")
            if os.path.exists(marker) and not job["overwrite"]:
                rec["warnings"].append("already extracted (kept)")
            else:
                if os.path.isdir(seq_dir):
                    shutil.rmtree(seq_dir)
                got = 0 if job["skip_extract"] else extract_frames(
                    job["video"], os.path.join(seq_dir, "img1"), built["first"], built["last"], job["stride"],
                    out_wh, job["jpeg_q"])
                if not job["skip_extract"] and got != n_frames:
                    rec["warnings"].append(f"extracted {got} frames but annotations span {n_frames} (video shorter?)")
                    n_frames = min(n_frames, got) if got else n_frames
                os.makedirs(seq_dir, exist_ok=True)
                write_sequence(seq_dir, [r for r in built["rows"] if r[0] <= n_frames], n_frames, out_wh, fps,
                               [r for r in ign_rows if r[0] <= n_frames])
                open(marker, "w").write("ok\n")
        rec["ok"] = True
    except Exception as e:                         # noqa: BLE001 - report, keep going
        rec["warnings"].append(f"ERROR {type(e).__name__}: {e}")
    return rec


# ----------------------------------------------------------------------------------------------------------------------
# statistics / recommendation
# ----------------------------------------------------------------------------------------------------------------------
def window_unique_counts(per_frame_ids, T):
    step = max(1, T // 2)
    return [len(set().union(*per_frame_ids[i:i + T])) for i in range(0, max(1, len(per_frame_ids) - T + 1), step)]


def pct(a, p):
    a = sorted(a)
    return a[min(len(a) - 1, int(math.ceil(p / 100 * len(a))) - 1)] if a else 0


def summarize(recs, T):
    ok = [r for r in recs if r.get("ok") and r.get("frames")]
    out = {}
    for split in ("train", "val"):
        rs = [r for r in ok if r["split"] == split]
        per_frame = [len(ids) for r in rs for ids in r["per_frame_ids"]]
        windows = [c for r in rs for c in window_unique_counts(r["per_frame_ids"], T)]
        cls = Counter()
        for r in rs:
            cls.update({CLASSES[int(k)]: v for k, v in r["classes"].items()})
        out[split] = {"videos": len(rs), "frames": sum(r["frames"] for r in rs), "boxes": sum(r["objects"] for r in rs),
                      "objects_per_frame": {"mean": round(sum(per_frame) / max(1, len(per_frame)), 2),
                                            "p99": pct(per_frame, 99), "max": max(per_frame, default=0)},
                      f"unique_ids_per_{T}_frame_window": {"p99": pct(windows, 99), "max": max(windows, default=0)},
                      "classes": dict(cls), "fps": sorted({round(r["fps"]) for r in rs})}
    w = out["train"].get(f"unique_ids_per_{T}_frame_window", {})
    out["recommended_NUM_ID_VOCABULARY"] = max(50, int(math.ceil(2.5 * w.get("p99", 0) / 50.0)) * 50)
    return out


# ----------------------------------------------------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--videos-dir", required=True)
    ap.add_argument("--anno-dir", default=None, help="default: same as --videos-dir")
    ap.add_argument("--out", required=True, help="datasets root; creates <out>/Vehicles")
    ap.add_argument("--val-frac", type=float, default=0.12, help="fraction of VIDEOS going to val (whole sites)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--valmini", type=int, default=20, help="#val videos in the small in-training eval split")
    ap.add_argument("--resize-short", type=int, default=800, help="0 = keep original resolution")
    ap.add_argument("--max-long", type=int, default=1440)
    ap.add_argument("--stride", type=int, default=1, help="keep every k-th frame (storage); 1 = all frames")
    ap.add_argument("--jpeg-q", type=int, default=3, help="ffmpeg -q:v (2 = best, 31 = worst)")
    ap.add_argument("--min-frames", type=int, default=30)
    ap.add_argument("--class-map", default=None, help="JSON {cvat_label: class_index | 'ignore' | null}")
    ap.add_argument("--window", type=int, default=30, help="T used for the dictionary-size recommendation")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--limit", type=int, default=0, help="debug: only the first N videos")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-extract", action="store_true", help="write annotations only (frames already extracted)")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)

    class_map = dict(DEFAULT_CLASS_MAP)
    if a.class_map:
        class_map.update(json.load(open(a.class_map)))
    anno_dir = a.anno_dir or a.videos_dir
    rows = read_csv(a.csv)
    if a.limit:
        rows = rows[:a.limit]
    missing, jobs_rows = [], []
    for r in rows:
        v = resolve(a.videos_dir, r.get("Video_file", ""), r["id"], VIDEO_EXTS)
        x = resolve(anno_dir, r.get("Anno_file", ""), r["id"], (".xml",))
        if x is None or (v is None and not a.dry_run):
            missing.append({"id": r["id"], "video": v, "anno": x})
            continue
        jobs_rows.append((r, v, x))
    rows = [r for r, _, _ in jobs_rows]
    if not rows:
        raise SystemExit(f"no usable rows; missing={missing[:5]}")
    split = make_split(rows, a.val_frac, a.seed)
    mini = pick_valmini(rows, split, a.valmini)
    root = os.path.join(a.out, "Vehicles")
    jobs = []
    for r, v, x in jobs_rows:
        seq = f"v{r['id']}"
        jobs.append(dict(id=r["id"], seq=seq, split=split[r["id"]], video=v, anno=x, class_map=class_map,
                         seq_dir=os.path.join(root, split[r["id"]], seq), stride=a.stride, resize_short=a.resize_short,
                         max_long=a.max_long, min_frames=a.min_frames, jpeg_q=a.jpeg_q, dry_run=a.dry_run,
                         skip_extract=a.skip_extract, overwrite=a.overwrite))
    print(f"{len(jobs)} videos usable ({len(missing)} missing) -> "
          f"train {sum(j['split']=='train' for j in jobs)} / val {sum(j['split']=='val' for j in jobs)} "
          f"(valmini {len(mini)}); workers={a.workers}{' [DRY RUN]' if a.dry_run else ''}", flush=True)
    if a.workers > 1 and len(jobs) > 1:
        with Pool(a.workers) as pool:
            recs = []
            for i, rec in enumerate(pool.imap_unordered(process_video, jobs), 1):
                recs.append(rec)
                if i % 20 == 0 or i == len(jobs):
                    print(f"  processed {i}/{len(jobs)}", flush=True)
    else:
        recs = [process_video(j) for j in jobs]
    recs.sort(key=lambda r: r["id"])

    by_id = {r["id"]: r for r in rows}
    stats = summarize(recs, a.window)
    stats["site_overlap"] = False
    for sp in ("train", "val"):
        stats[sp]["sites"] = len({site_key(by_id[r["id"]]) for r in recs if r["split"] == sp and r.get("ok")})
        stats[sp]["countries"] = dict(Counter(by_id[r["id"]].get("Country", "") for r in recs if r["split"] == sp and r.get("ok")))
    frames = stats["train"]["frames"] + stats["val"]["frames"]
    stats["estimated_disk_GB"] = round(frames * MB_PER_FRAME / 1024, 1)
    report = {"stats": stats, "missing": missing, "class_map": class_map, "args": vars(a),
              "warnings": {r["id"]: r["warnings"] for r in recs if r["warnings"]},
              "skipped": [r["id"] for r in recs if not r.get("ok") or not r.get("frames")]}
    print(json.dumps(stats, indent=1))
    print(f"warnings in {len(report['warnings'])} videos, skipped {len(report['skipped'])}, missing {len(missing)}")
    print(f"==> set NUM_ID_VOCABULARY (K) to >= {stats['recommended_NUM_ID_VOCABULARY']} in the config "
          f"(p99 unique ids per {a.window}-frame window x 2.5, multiple of 50)")
    if a.dry_run:
        return report

    # split bookkeeping, seqmaps, valmini symlinks
    good = {r["id"] for r in recs if r.get("ok") and r.get("frames") and r["frames"] >= a.min_frames}
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, "split.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "seq", "split", "country", "site", "frames", "boxes"])
        for r in recs:
            if r["id"] in good:
                rr = by_id[r["id"]]
                w.writerow([r["id"], r["seq"], r["split"], rr.get("Country", ""), rr.get("Site", ""), r["frames"], r["objects"]])
    seqs = {sp: sorted(r["seq"] for r in recs if r["id"] in good and r["split"] == sp) for sp in ("train", "val")}
    mini_seqs = sorted(r["seq"] for r in recs if r["id"] in good and r["id"] in mini)
    for name, lst in (("train", seqs["train"]), ("val", seqs["val"]), ("valmini", mini_seqs)):
        with open(os.path.join(root, f"{name}_seqmap.txt"), "w") as f:
            f.write("name\n" + "\n".join(lst) + "\n")
    mini_dir = os.path.join(root, "valmini")
    shutil.rmtree(mini_dir, ignore_errors=True)
    os.makedirs(mini_dir)
    for s in mini_seqs:
        os.symlink(os.path.join("..", "val", s), os.path.join(mini_dir, s))
    with open(os.path.join(root, "prepare_report.json"), "w") as f:
        json.dump(report, f, indent=1, default=str)
    print(f"written: {root}  (train {len(seqs['train'])} / val {len(seqs['val'])} / valmini {len(mini_seqs)} sequences)")
    return report


if __name__ == "__main__":
    main()
