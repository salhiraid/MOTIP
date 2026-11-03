# Vehicle dataset preparation: CSV -> site-disjoint split, CVAT XML parsing, ffmpeg frame extraction, MOT-style output.
import os
import shutil
import subprocess
import sys
import textwrap

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools", "vehicles"))

import prepare_vehicle_dataset as P
import make_synthetic_vehicle_videos as G

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")

# the layout of the user's CVAT export (labels block, original_size, heavy_truck track with a few boxes)
CVAT_SNIPPET = textwrap.dedent("""\
    <?xml version="1.0" encoding="utf-8"?>
    <annotations>
      <version>1.1</version>
      <meta>
        <task>
          <id>1041</id><name>Video annotation</name><size>271</size><mode>interpolation</mode><overlap>5</overlap>
          <start_frame>0</start_frame><stop_frame>270</stop_frame><frame_filter></frame_filter>
          <labels>
            <label><name>heavy_truck</name><color>#b25050</color><attributes></attributes></label>
            <label><name>ignore_region</name><color>#b83df5</color><attributes></attributes></label>
          </labels>
          <original_size><width>1920</width><height>1080</height></original_size>
        </task>
        <dumped>2021-12-29 20:23:25.812592+00:00</dumped>
        <source>xxxxxxxx-truck.mp4</source>
      </meta>
      <track id="0" label="heavy_truck" source="manual">
        <box frame="43" outside="0" occluded="0" keyframe="1" xtl="0.00" ytl="326.13" xbr="509.52" ybr="1080.00" z_order="0"></box>
        <box frame="44" outside="0" occluded="0" keyframe="0" xtl="0.00" ytl="312.40" xbr="547.87" ybr="1080.00" z_order="0"></box>
        <box frame="47" outside="0" occluded="1" keyframe="1" xtl="0.00" ytl="271.20" xbr="662.90" ybr="1080.00" z_order="0"></box>
        <box frame="48" outside="1" occluded="0" keyframe="1" xtl="0.00" ytl="265.55" xbr="696.00" ybr="1080.00" z_order="0"></box>
        <box frame="60" outside="0" occluded="0" keyframe="1" xtl="10.00" ytl="200.00" xbr="700.00" ybr="1000.00" z_order="0"></box>
      </track>
      <track id="1" label="ignore_region" source="manual">
        <box frame="43" outside="0" occluded="0" keyframe="1" xtl="0" ytl="900" xbr="400" ybr="1080" z_order="0"></box>
      </track>
      <track id="2" label="weird_label" source="manual">
        <box frame="43" outside="0" occluded="0" keyframe="1" xtl="0" ytl="0" xbr="10" ybr="10" z_order="0"></box>
      </track>
    </annotations>
    """)


def test_cvat_snippet_in_the_users_format(tmp_path):
    x = tmp_path / "a.xml"
    x.write_text(CVAT_SNIPPET)
    parsed = P.parse_cvat_xml(str(x), P.DEFAULT_CLASS_MAP)
    assert (parsed["width"], parsed["height"], parsed["start"], parsed["stop"]) == (1920, 1080, 0, 270)
    assert [t["label"] for t in parsed["tracks"]] == ["heavy_truck", "ignore_region"]      # unknown label dropped
    assert any("weird_label" in w for w in parsed["warnings"])
    truck = parsed["tracks"][0]
    assert truck["cls"] == P.CLASSES.index("truck")
    frames, filled = P.expand_track(truck["boxes"])
    # 43,44 visible; 45,46 interpolated between 44 and 47; 47 visible; 48 is the `outside` marker (removed);
    # 49..59 NOT interpolated because the previous box is outside; 60 visible
    assert sorted(frames) == [43, 44, 45, 46, 47, 60] and filled == 2
    assert frames[45][:4] == pytest.approx((0.0, 312.40 + (271.20 - 312.40) / 3, 547.87 + (662.90 - 547.87) / 3, 1080.0), rel=1e-3) \
        or frames[45][1] == pytest.approx(312.40 + (271.20 - 312.40) / 3, rel=1e-3)
    assert frames[47][4] == 1                                                              # occluded flag kept
    built = P.build_rows({"tracks": [truck]}, stride=1, scale_xy=(0.5, 0.5), out_wh=(960, 540))
    assert built["first"] == 43 and built["last"] == 60
    assert [r[0] for r in built["rows"]] == [1, 2, 3, 4, 5, 18]                            # re-numbered from 1
    assert built["rows"][0][1] == 1 and built["rows"][0][2] == 4                           # id = track id + 1, class = truck
    fr, oid, cls, x1, y1, w, h, occ = built["rows"][0]
    assert (x1, y1, w, h) == pytest.approx((0.0, 163.065, 254.76, 540.0 - 163.065))        # scaled by 0.5


def test_stride_and_output_size():
    assert P.out_size(1920, 1080, 800, 1440) == (1422, 800)
    assert P.out_size(1080, 1920, 800, 1440) == (800, 1422)
    assert P.out_size(4000, 1000, 800, 1440) == (1440, 360)                                # long side capped
    assert P.out_size(1920, 1080, 0, 0) == (1920, 1080)


def test_site_disjoint_split_stratified_by_country():
    import random
    rows, vid = [], 0
    rng = random.Random(1)
    for country, n_sites in [("FR", 8), ("DE", 6), ("ES", 1), ("IT", 2)]:
        for s in range(n_sites):
            for _ in range(rng.randint(2, 9)):
                vid += 1
                rows.append({"id": str(vid), "Country": country, "Site": f"{country}_{s}"})
    split = P.make_split(rows, val_frac=0.2, seed=3)
    P.check_site_disjoint(rows, split)                                                     # no site in both splits
    assert split == P.make_split(rows, val_frac=0.2, seed=3)                               # deterministic
    assert split != P.make_split(rows, val_frac=0.2, seed=4)
    es = {split[r["id"]] for r in rows if r["Country"] == "ES"}
    assert es == {"train"}                                                                 # single-site country: stays in train
    for c in ("FR", "DE", "IT"):
        assert "train" in {split[r["id"]] for r in rows if r["Country"] == c}              # never the whole country in val
    frac = sum(v == "val" for v in split.values()) / len(split)
    assert 0.08 < frac < 0.35, frac


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    d = tmp_path_factory.mktemp("veh")
    src, out = d / "src", d / "out"
    G.main(["--out", str(src), "--videos-per-site", "2", "--frames", "60"])
    rep = P.main(["--csv", str(src / "dataset.csv"), "--videos-dir", str(src), "--out", str(out), "--val-frac", "0.3",
                  "--resize-short", "96", "--max-long", "160", "--valmini", "2", "--workers", "2", "--min-frames", "20"])
    return src, out / "Vehicles", rep


def test_prepared_dataset_structure_and_split(prepared):
    src, root, rep = prepared
    import csv
    rows = {r["id"]: r for r in csv.DictReader(open(src / "dataset.csv"))}
    split = {r["seq"]: r for r in csv.DictReader(open(root / "split.csv"))}
    sites = {s: {(rows[r["id"]]["Country"], rows[r["id"]]["Site"]) for r in split.values() if r["split"] == s} for s in ("train", "val")}
    assert sites["train"] and sites["val"] and not (sites["train"] & sites["val"])         # site-disjoint
    for sp in ("train", "val", "valmini"):
        names = open(root / f"{sp}_seqmap.txt").read().split()[1:]
        assert names and all((root / sp / n / "seqinfo.ini").exists() for n in names)
    assert set(open(root / "valmini_seqmap.txt").read().split()[1:]) <= set(open(root / "val_seqmap.txt").read().split()[1:])
    assert all(os.path.islink(root / "valmini" / n) for n in os.listdir(root / "valmini"))
    assert rep["stats"]["site_overlap"] is False and rep["stats"]["recommended_NUM_ID_VOCABULARY"] >= 50


def test_prepared_sequences_frames_and_gt_are_consistent(prepared):
    from PIL import Image
    _, root, _ = prepared
    for sp in ("train", "val"):
        for seq in os.listdir(root / sp):
            d = root / sp / seq
            info = dict(l.strip().split("=") for l in open(d / "seqinfo.ini") if "=" in l)
            n, W, H = int(info["seqLength"]), int(info["imWidth"]), int(info["imHeight"])
            jpgs = sorted(os.listdir(d / "img1"))
            assert len(jpgs) == n and jpgs[0] == "00000001.jpg" and jpgs[-1] == f"{n:08d}.jpg"
            assert Image.open(d / "img1" / jpgs[0]).size == (W, H) and min(W, H) == 96 and max(W, H) <= 160
            gt = [l.strip().split(",") for l in open(d / "gt" / "gt.txt")]
            gc = [l.strip().split(",") for l in open(d / "gt" / "gt_classes.txt")]
            veh = [r for r in gt if r[7] == "1"]
            assert len(veh) == len(gc) and all(r[7] == "8" for r in gt if r[7] != "1")      # eval class=1, ignore regions=8
            assert {int(r[7]) for r in gc} <= set(range(5))
            frames = {int(r[0]) for r in gc}
            assert frames == set(range(1, n + 1))                                           # every frame annotated (interpolation)
            for r in gc:                                                                    # boxes inside the image
                x, y, w, h = map(float, r[2:6])
                assert -1e-6 <= x and x + w <= W + 1e-6 and -1e-6 <= y and y + h <= H + 1e-6 and w >= 2 and h >= 2
            ids_per_frame = {}
            for r in gc:
                ids_per_frame.setdefault(int(r[0]), []).append(r[1])
            assert all(len(v) == len(set(v)) for v in ids_per_frame.values())


def test_boxes_follow_the_moving_objects(prepared):
    """Consecutive-frame boxes of one id overlap strongly (they come from smoothly moving objects) -> frame numbering
    and box scaling are consistent with the decoded video."""
    _, root, _ = prepared
    d = root / "train" / sorted(os.listdir(root / "train"))[0]
    by = {}
    for l in open(d / "gt" / "gt_classes.txt"):
        f = l.strip().split(",")
        by.setdefault(f[1], {})[int(f[0])] = tuple(map(float, f[2:6]))
    def iou(a, b):
        ax, ay, aw, ah = a; bx, by_, bw, bh = b
        iw = max(0, min(ax + aw, bx + bw) - max(ax, bx)); ih = max(0, min(ay + ah, by_ + bh) - max(ay, by_))
        return iw * ih / (aw * ah + bw * bh - iw * ih)
    pairs = [iou(t[f], t[f + 1]) for t in by.values() for f in t if f + 1 in t]
    assert pairs and min(pairs) > 0.5


def test_rendered_object_is_where_the_box_says(prepared):
    """Pixel check: the (distinctly coloured) object is inside its box in the decoded JPEG."""
    import numpy as np
    from PIL import Image
    _, root, _ = prepared
    d = root / "val" / sorted(os.listdir(root / "val"))[0]
    row = next(l.strip().split(",") for l in open(d / "gt" / "gt_classes.txt") if l.startswith("10,"))
    x, y, w, h = map(float, row[2:6])
    img = np.asarray(Image.open(d / "img1" / "00000010.jpg").convert("RGB")).astype(float)
    def fg(region):                                    # fraction of pixels that differ from the grey background (45)
        return float((np.abs(region - 45.0).max(-1) > 35).mean()) if region.size else 0.0
    inside = img[int(y + 3):int(y + h - 3), int(x + 3):int(x + w - 3)]
    assert fg(inside) > 0.9                            # the object fills its box ...
    # ... and the box is tight: a same-size window shifted far away from the box is mostly background
    H_, W_ = img.shape[:2]
    cx = int(x + w / 2)
    far = img[int(y + h) + 4:int(y + h) + 4 + int(h * 0.5), max(0, cx - 3):cx + 3] if y + h + 4 + h * 0.5 < H_ else img[:0]
    assert fg(far) < 0.8 or far.size == 0


def test_vehicles_dataset_class_loads_classes_and_legality(prepared):
    from data.joint_dataset import dataset_classes, JointDataset
    _, root, _ = prepared
    ds = dataset_classes["Vehicles"](data_root=str(root.parent), split="train")
    anns = ds.get_annotations()
    cats = {int(c) for seq in anns.values() for a in seq for c in a["category"]}
    assert cats <= set(range(5)) and len(cats) >= 3
    assert all(a["is_legal"] for seq in anns.values() for a in seq)
    jd = JointDataset(data_root=str(root.parent), datasets=["Vehicles"], splits=["train"])
    assert jd.statistics()[0].startswith("Vehicles.train")


def test_trackeval_scores_the_ground_truth_as_a_perfect_tracker(prepared, tmp_path):
    _, root, _ = prepared
    tr = tmp_path / "tr"
    tr.mkdir()
    for seq in os.listdir(root / "val"):
        out = [",".join(f[:6] + ["1", "-1", "-1", "-1"]) for f in (l.strip().split(",") for l in open(root / "val" / seq / "gt" / "gt.txt")) if f[7] == "1"]
        (tr / f"{seq}.txt").write_text("\n".join(out) + "\n")
    cmd = [sys.executable, "TrackEval/scripts/run_mot_challenge.py", "--SPLIT_TO_EVAL", "val", "--METRICS", "HOTA", "CLEAR", "Identity",
           "--GT_FOLDER", str(root / "val"), "--SEQMAP_FILE", str(root / "val_seqmap.txt"), "--SKIP_SPLIT_FOL", "True",
           "--TRACKERS_TO_EVAL", "", "--TRACKER_SUB_FOLDER", "", "--USE_PARALLEL", "False", "--PLOT_CURVES", "False",
           "--TRACKERS_FOLDER", str(tr)]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.join(os.path.dirname(__file__), ".."))
    assert r.returncode == 0, r.stderr[-500:]
    summary = (tr / "pedestrian_summary.txt").read_text().split("\n")
    d = dict(zip(summary[0].split(), summary[1].split()))
    assert float(d["HOTA"]) == pytest.approx(100.0) and float(d["IDF1"]) == pytest.approx(100.0) and float(d["IDSW"]) == 0


def test_dry_run_writes_nothing_and_reports(tmp_path):
    src = tmp_path / "src"
    G.main(["--out", str(src), "--videos-per-site", "1", "--frames", "40"])
    rep = P.main(["--csv", str(src / "dataset.csv"), "--videos-dir", str(src), "--out", str(tmp_path / "o"), "--dry-run",
                  "--workers", "1", "--min-frames", "20"])
    assert not (tmp_path / "o").exists() and rep["stats"]["train"]["videos"] > 0
