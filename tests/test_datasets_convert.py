# Converters of the public tracking datasets: each is fed a tiny dataset written in the dataset's NATIVE format
# (tests/helpers/fake_datasets.py), then the converted output is loaded with the real loader (data/unified_mot.py) and
# compared with the known ground truth: boxes, unified class indices, id consistency, splits, distractors/ignore regions.
import json
import os
import shutil
import sys

import pytest
import torch

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(__file__))

from helpers import fake_datasets as F                        # noqa: E402
from data.converters import prepare                            # noqa: E402
from data.joint_dataset import dataset_classes                # noqa: E402

CAR, BUS, TRUCK = 1, 3, 4                                     # vehicles5 = (bicycle, car, motorcycle, bus, truck)


def _load(out, name, split):
    ds = dataset_classes[name](data_root=out, split=split, load_annotation=True)
    return ds.get_sequence_infos(), ds.get_image_paths(), ds.get_annotations()


def _class8_per_frame(out, dirname, split, seq):
    rows = [l.split(",") for l in open(os.path.join(out, dirname, split, seq, "gt", "gt.txt")).read().split()]
    n8, n1 = {}, {}
    for r in rows:
        tgt = n8 if int(r[7]) == 8 else n1
        tgt[int(r[0])] = tgt.get(int(r[0]), 0) + 1
    return n1, n8


def check(tmp, name, dirname, seqs, targets, n_class8, tol=0.02, ext=".jpg", shape=None):
    """seqs: {split: [seq names]}; targets: {class index: box fn(frame)}; n_class8: distractor+ignore rows per frame."""
    out = str(tmp / "out")
    for split, names in seqs.items():
        info, paths, ann = _load(out, name, split)
        assert sorted(info) == sorted(names), (split, sorted(info))
        for seq in names:
            assert info[seq]["length"] == F.N
            assert (info[seq]["width"], info[seq]["height"]) == (shape or (F.W, F.H))
            assert len(paths[seq]) == F.N and all(os.path.exists(p) and p.endswith(ext) for p in paths[seq])
            by_cls = {}
            for f, a in enumerate(ann[seq], 1):
                assert len(a["id"]) == len(targets), (seq, f, a["id"])
                for i in range(len(a["id"])):
                    by_cls.setdefault(int(a["category"][i]), []).append((f, int(a["id"][i]), a["bbox"][i].tolist()))
            assert set(by_cls) == set(targets)
            for cls, fn in targets.items():
                rows = by_cls[cls]
                assert len({r[1] for r in rows}) == 1, f"class {cls}: identity not constant {set(r[1] for r in rows)}"
                assert len(rows) == F.N
                for f, _, b in rows:
                    exp = fn(f)
                    assert all(abs(b[k] - exp[k]) < tol for k in range(4)), (seq, cls, f, b, exp)
            assert len({r[1] for rows in by_cls.values() for r in rows}) == len(targets)    # distinct ids per object
            n1, n8 = _class8_per_frame(out, dirname, split, seq)
            assert set(n1.values()) == {len(targets)}
            assert (set(n8.values()) if n_class8 else set(n8)) == ({n_class8} if n_class8 else set()), n8
            gc = open(os.path.join(out, dirname, split, seq, "gt", "gt_classes.txt")).read()
            assert all(int(l.split(",")[7]) in targets for l in gc.split())                  # no distractor in training GT
        assert os.path.exists(os.path.join(out, dirname, f"{split}_seqmap.txt"))


@pytest.fixture
def tmp(tmp_path):
    return tmp_path


def run(tmp, *argv):
    return prepare.main([argv[0], "--root", str(tmp / "src"), "--out", str(tmp / "out"), "--workers", "1", *argv[1:]])


def test_bdd100k(tmp):
    F.bdd100k(str(tmp / "src"))
    run(tmp, "bdd100k", "--valmini", "1")
    check(tmp, "BDD100K_MOT", "BDD100K_MOT", {"train": ["trainvid"], "val": ["valvid"]},
          {CAR: F.box_a, TRUCK: F.box_b}, n_class8=2)         # pedestrian (distractor) + crowd region (ignore)
    assert os.path.isdir(str(tmp / "out" / "BDD100K_MOT" / "valmini" / "valvid"))


def test_bdd100k_person_class_set_turns_pedestrians_into_targets(tmp):
    F.bdd100k(str(tmp / "src"))
    run(tmp, "bdd100k", "--class-set", "vehicles5_person")
    check(tmp, "BDD100K_MOT", "BDD100K_MOT", {"train": ["trainvid"], "val": ["valvid"]},
          {CAR: F.box_a, TRUCK: F.box_b, 5: F.box_c}, n_class8=1)       # only the crowd region is left as class 8


def test_detrac(tmp):
    F.detrac(str(tmp / "src"))
    run(tmp, "detrac")
    check(tmp, "UA_DETRAC", "UA_DETRAC", {"train": ["MVI_20011"], "val": ["MVI_39031"]},
          {CAR: F.box_a, BUS: F.box_b}, n_class8=2)           # `others` = distractor + the ignored region


def test_visdrone(tmp):
    F.visdrone(str(tmp / "src"))
    run(tmp, "visdrone")
    check(tmp, "VisDrone_MOT", "VisDrone_MOT", {"train": ["uav0000013_00000_v"], "val": ["uav0000086_00000_v"]},
          {CAR: F.box_a, TRUCK: F.box_b}, n_class8=3)         # pedestrian + category-0 region + score-0 entry


def test_uavdt(tmp):
    F.uavdt(str(tmp / "src"))
    run(tmp, "uavdt")
    check(tmp, "UAVDT", "UAVDT", {"train": ["M0101"], "val": ["M0203"]}, {CAR: F.box_a, TRUCK: F.box_b}, n_class8=1)


def test_kitti_png(tmp):
    F.kitti(str(tmp / "src"))
    run(tmp, "kitti")
    check(tmp, "KITTI_Tracking", "KITTI_Tracking", {"train": ["0000"], "val": ["0001"]},
          {CAR: F.box_a, TRUCK: F.box_b}, n_class8=2, ext=".png")       # pedestrian + DontCare


def test_kitti_jpeg_mode_and_resize_rescales_boxes(tmp):
    F.kitti(str(tmp / "src"))
    run(tmp, "kitti", "--image-mode", "jpeg", "--resize-short", "32")           # 96x64 -> 48x32 (x0.5)
    half = lambda fn: (lambda f: tuple(v / 2 for v in fn(f)))
    check(tmp, "KITTI_Tracking", "KITTI_Tracking", {"train": ["0000"], "val": ["0001"]},
          {CAR: half(F.box_a), TRUCK: half(F.box_b)}, n_class8=2, shape=(48, 32))


def test_mot_format_fasttracker(tmp):
    F.mot_format(str(tmp / "src"))
    json.dump({"3": "car", "7": "truck", "1": "person", "9": "ignore"}, open(tmp / "names.json", "w"))
    run(tmp, "fasttracker", "--class-names", str(tmp / "names.json"), "--val-seqs", "cctv4")
    check(tmp, "FastTracker", "FastTracker", {"train": ["cctv1", "cctv2", "cctv3"], "val": ["cctv4"]},
          {CAR: F.box_a, TRUCK: F.box_b}, n_class8=2)         # person (distractor) + ignore
    assert "frameRate=15" in open(tmp / "out" / "FastTracker" / "val" / "cctv4" / "seqinfo.ini").read()


def test_mot_format_requires_class_names_and_lists_ids(tmp, capsys):
    F.mot_format(str(tmp / "src"))
    with pytest.raises(SystemExit) as e:
        run(tmp, "fasttracker")
    assert "'3'" in str(e.value) and "'7'" in str(e.value)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_cityflow_extracts_frames_and_splits_by_scene(tmp):
    F.cityflow(str(tmp / "src"))
    run(tmp, "cityflow")
    out = str(tmp / "out")
    for split, seq in (("train", "S01_c001"), ("val", "S02_c001")):
        info, paths, ann = _load(out, "CityFlow", split)
        assert info[seq]["length"] == F.N and (info[seq]["width"], info[seq]["height"]) == (F.W, F.H)
        assert all(int(c) == CAR for a in ann[seq] for c in a["category"])          # class-less data -> `car`
        assert sorted(set(int(i) for a in ann[seq] for i in a["id"])) == [101, 202]
        b = ann[seq][4]["bbox"][0].tolist()
        assert max(abs(b[k] - F.box_a(5)[k]) for k in range(4)) < 0.02


def test_waymo_v2_parquet(tmp):
    pytest.importorskip("pyarrow")
    F.waymo(str(tmp / "src"))
    run(tmp, "waymo", "--cameras", "1")
    out = str(tmp / "out")
    info, paths, ann = _load(out, "Waymo", "train")
    seq = "segA_FRONT"
    assert info[seq]["length"] == F.N
    # vehicles a and b are both `car` (the camera labels cannot tell more); pedestrian = distractor; sign dropped
    assert all(int(c) == CAR for a in ann[seq] for c in a["category"]) and all(len(a["id"]) == 2 for a in ann[seq])
    got = sorted(a["bbox"][i].tolist() for a in ann[seq][2:3] for i in range(2))
    exp = sorted([list(F.box_a(3)), list(F.box_b(3))])
    assert all(abs(g[k] - e[k]) < 0.02 for g, e in zip(got, exp) for k in range(4))
    n1, n8 = _class8_per_frame(out, "Waymo", "train", seq)
    assert set(n8.values()) == {1}


def test_nuscenes_projection_matches_independent_projection(tmp):
    F.nuscenes(str(tmp / "src"))
    run(tmp, "nuscenes", "--version", "v1.0-mini", "--cameras", "CAM_FRONT")
    out = str(tmp / "out")
    found = {}
    for split in ("train", "val"):
        if os.path.isdir(os.path.join(out, "NuScenes", split)):
            info, paths, ann = _load(out, "NuScenes", split)
            found.update({s: ann[s] for s in info})
    assert sorted(found) == ["scene-0000_CAM_FRONT", "scene-0001_CAM_FRONT"]     # translated ego / translated+yawed ego
    assert len({sp for sp in ("train", "val") if os.path.isdir(os.path.join(out, "NuScenes", sp))}) == 2
    for seq, ann in found.items():
        for f, a in enumerate(ann, 1):
            assert sorted(int(c) for c in a["category"]) == [CAR, TRUCK]          # pedestrian = distractor
            for i in range(2):
                exp = F.project_expected(*(F.ego_a(f) if int(a["category"][i]) == CAR else F.ego_b(f)))
                got = a["bbox"][i].tolist()
                assert max(abs(got[k] - exp[k]) for k in range(4)) < 0.05, (seq, f, got, exp)


def test_nuscenes_visibility_and_lidar_filters(tmp):
    F.nuscenes(str(tmp / "src"))
    p = tmp / "src" / "v1.0-mini" / "sample_annotation.json"
    anns = json.load(open(p))
    for a in anns:
        if a["instance_token"] == "iB":
            a["num_lidar_pts"] = 0                      # no points -> dropped by --min-pts 1
        if a["instance_token"] == "iA":
            a["visibility_token"] = "1"                 # < --min-visibility 2 -> dropped
    json.dump(anns, open(p, "w"))
    (tmp / "src" / "v1.0-mini" / "visibility.json").write_text(json.dumps([{"token": "1", "level": "v0-40"}, {"token": "4", "level": "v80-100"}]))
    run(tmp, "nuscenes", "--version", "v1.0-mini", "--cameras", "CAM_FRONT")
    # nothing left to track -> every sequence is skipped, nothing is written as an empty dataset
    assert not any(os.path.isdir(str(tmp / "out" / "NuScenes" / s)) and os.listdir(str(tmp / "out" / "NuScenes" / s))
                   for s in ("train", "val"))


def test_argoverse2_ego_motion_compensation(tmp):
    pytest.importorskip("pyarrow")
    F.argoverse2(str(tmp / "src"))
    run(tmp, "argoverse2", "--cameras", "ring_front_center")
    out = str(tmp / "out")
    for split, seq in (("train", "logT_ring_front_center"), ("val", "logV_ring_front_center")):
        info, paths, ann = _load(out, "Argoverse2", split)
        assert info[seq]["length"] == F.N
        for f, a in enumerate(ann[seq], 1):
            assert sorted(int(c) for c in a["category"]) == [CAR, TRUCK]
            for i in range(2):
                fn = F.ego_a if int(a["category"][i]) == CAR else F.ego_b
                exp = F.project_expected(*F.ego_moved(*fn(f)))                # the camera frame is 0.5 m further along
                got = a["bbox"][i].tolist()
                assert max(abs(got[k] - exp[k]) for k in range(4)) < 0.05, (seq, f, got, exp)
        # the compensation matters: without it the boxes would be visibly different
        raw = F.project_expected(*F.ego_a(5))
        comp = F.project_expected(*F.ego_moved(*F.ego_a(5)))
        assert max(abs(raw[k] - comp[k]) for k in range(4)) > 0.2


def test_generic_3d_boxes_with_projection_matrix(tmp):
    F.boxes3d(str(tmp / "src"))
    run(tmp, "boxes3d", "--min-frames", "5")
    check(tmp, "Generic3D", "Generic3D", {"train": ["i24like_camA"]}, {CAR: lambda f: F.project_expected(*F.ego_a(f)),
                                                                     TRUCK: lambda f: F.project_expected(*F.ego_b(f))},
          n_class8=0, tol=0.05)


def test_dry_run_writes_nothing_and_reports_statistics(tmp, capsys):
    F.detrac(str(tmp / "src"))
    rep = run(tmp, "detrac", "--dry-run")
    assert not (tmp / "out").exists()
    assert rep["stats"]["train"]["classes"] == {"car": 12, "bus": 12}
    assert rep["stats"]["recommended_NUM_ID_VOCABULARY"] >= 50


def test_class_map_override_changes_the_mapping(tmp):
    F.detrac(str(tmp / "src"))
    json.dump({"others": "truck"}, open(tmp / "cm.json", "w"))
    run(tmp, "detrac", "--class-map", str(tmp / "cm.json"))
    check(tmp, "UA_DETRAC", "UA_DETRAC", {"train": ["MVI_20011"], "val": ["MVI_39031"]},
          {CAR: F.box_a, BUS: F.box_b, TRUCK: F.box_c}, n_class8=1)


def test_unknown_dataset_name_falls_back_to_the_unified_loader_with_a_clear_error(tmp):
    with pytest.raises(FileNotFoundError, match="not found"):
        dataset_classes["MyCustomData"](data_root=str(tmp), split="train", load_annotation=True)
    F.mot_format(str(tmp / "src"))                                  # any dir in the unified layout is loadable by name
    json.dump({"3": "car", "7": "truck", "1": "person", "9": "ignore"}, open(tmp / "n.json", "w"))
    prepare.main(["mot", "--root", str(tmp / "src"), "--out", str(tmp / "out"), "--dir-name", "MyCustomData",
                  "--class-names", str(tmp / "n.json"), "--workers", "1"])
    info, _, ann = _load(str(tmp / "out"), "MyCustomData", "train")
    assert len(info) >= 1
