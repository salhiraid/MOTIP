# Training on one or several datasets: mixed sampling, per-dataset frame-interval caps, per-dataset class masking in the
# detection loss, the config generator, and an end-to-end CPU training run on two converted datasets.
import json
import os
import re
import subprocess
import sys

import pytest
import torch
import yaml

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(__file__))

from helpers import fake_datasets as F                                 # noqa: E402
from data.converters import prepare                                     # noqa: E402
from data.joint_dataset import JointDataset                            # noqa: E402
from data.naive_sampler import NaiveSampler                            # noqa: E402


def convert(tmp, which):
    """Convert fake datasets (native formats) into tmp/data. Returns the data root."""
    src, out = tmp / "src", tmp / "data"
    for name in which:
        d = src / name
        getattr(F, name)(str(d))
        prepare.main([name, "--root", str(d), "--out", str(out), "--workers", "1", "--valmini", "1"])
    return str(out)


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("multi")
    return convert(tmp, ["detrac", "bdd100k", "uavdt"]), tmp


def sampler(root, datasets, mix=None, caps=None, n=300, length=4, interval=4):
    ds = JointDataset(data_root=root, datasets=datasets, splits=["train"] * len(datasets))
    per = lambda v: None if v is None else {d: {"train": x} for d, x in zip(datasets, v)}
    s = NaiveSampler(ds, [0], [length], [interval], None, samples_per_epoch=n, dataset_mix=per(mix), max_intervals=per(caps))
    return ds, s


def test_mix_gives_exact_dataset_shares_and_is_deterministic(data):
    root, _ = data
    names = ["UA_DETRAC", "BDD100K_MOT", "UAVDT"]
    _, s = sampler(root, names, mix=[6, 3, 1], n=1000)
    s.prepare_for_epoch(0)
    c = {n: sum(i["dataset"] == n for i in s.sample_infos) for n in names}
    assert c == {"UA_DETRAC": 600, "BDD100K_MOT": 300, "UAVDT": 100} and len(s) == 1000
    a = [(i["dataset"], i["sequence"], tuple(i["frame_idxs"])) for i in s.sample_infos]
    s.prepare_for_epoch(0)
    assert a == [(i["dataset"], i["sequence"], tuple(i["frame_idxs"])) for i in s.sample_infos]       # same epoch, same draw
    s.prepare_for_epoch(1)
    assert a != [(i["dataset"], i["sequence"], tuple(i["frame_idxs"])) for i in s.sample_infos]       # new epoch, new draw


def test_mix_rounding_sums_exactly_and_zero_share_excludes_a_dataset(data):
    root, _ = data
    names = ["UA_DETRAC", "BDD100K_MOT", "UAVDT"]
    _, s = sampler(root, names, mix=[1, 1, 1], n=100)
    s.prepare_for_epoch(0)
    counts = sorted(sum(i["dataset"] == n for i in s.sample_infos) for n in names)
    assert counts == [33, 33, 34]
    _, s = sampler(root, names, mix=[1, 0, 1], n=100)
    s.prepare_for_epoch(0)
    assert {i["dataset"] for i in s.sample_infos} == {"UA_DETRAC", "UAVDT"} and len(s) == 100


def test_small_pool_is_cycled_and_every_clip_is_used(data):
    root, _ = data
    ds, s = sampler(root, ["UA_DETRAC", "UAVDT"], mix=[1, 1], n=400)
    s.prepare_for_epoch(0)
    ua = {(i["sequence"], i["frame_idxs"][0]) for i in s.sample_infos if i["dataset"] == "UA_DETRAC"}
    assert len(ua) == 12 - 4 + 1            # a 12-frame sequence has 9 start positions of a 4-frame clip: all drawn


def test_per_dataset_interval_cap(data):
    root, _ = data
    names = ["UA_DETRAC", "BDD100K_MOT"]
    _, s = sampler(root, names, mix=[1, 1], caps=[1, 4], n=600)
    s.prepare_for_epoch(0)
    steps = {n: set() for n in names}
    for i in s.sample_infos:
        steps[i["dataset"]].add(i["frame_idxs"][1] - i["frame_idxs"][0])
    assert steps["UA_DETRAC"] == {1}
    assert steps["BDD100K_MOT"] == {1, 2, 3} or max(steps["BDD100K_MOT"]) > 1      # 12 frames: max interval 3 for T=4


def test_default_sampling_is_unchanged_without_mix(data):
    root, _ = data
    ds = JointDataset(data_root=root, datasets=["UA_DETRAC"], splits=["train"])
    a = NaiveSampler(ds, [0], [4], [4], None, samples_per_epoch=50)
    b = NaiveSampler(ds, [0], [4], [4], None, samples_per_epoch=50, dataset_mix=None, max_intervals=None)
    a.prepare_for_epoch(3); b.prepare_for_epoch(3)
    assert a.sample_infos == b.sample_infos


def test_mix_without_samples_per_epoch_is_rejected(data):
    root, _ = data
    ds = JointDataset(data_root=root, datasets=["UA_DETRAC"], splits=["train"])
    with pytest.raises(AssertionError):
        NaiveSampler(ds, [0], [4], [4], None, dataset_mix={"UA_DETRAC": {"train": 1.0}})


# ---------------------------------------------------------------------------------------------- class masking
def test_annotated_classes_come_from_the_converter_and_become_masks(data):
    root, _ = data
    ds = JointDataset(data_root=root, datasets=["UA_DETRAC", "BDD100K_MOT"], splits=["train", "train"], class_masking=True,
                      num_classes=5)
    # UA-DETRAC annotates car + bus only; BDD100K car, truck, bus, motorcycle, bicycle (all five) -> no mask needed
    assert ds.class_masks["UA_DETRAC"]["train"].tolist() == [0, 1, 0, 1, 0]
    assert ds.class_masks["BDD100K_MOT"]["train"] is None
    images, annotations, metas = ds[{"dataset": "UA_DETRAC", "split": "train", "sequence": "MVI_20011", "frame_idxs": [0, 1]}]
    assert all(a["class_mask"].tolist() == [0, 1, 0, 1, 0] for a in annotations)
    images, annotations, metas = ds[{"dataset": "BDD100K_MOT", "split": "train", "sequence": "trainvid", "frame_idxs": [0, 1]}]
    assert all("class_mask" not in a for a in annotations)
    off = JointDataset(data_root=root, datasets=["UA_DETRAC"], splits=["train"], class_masking=False, num_classes=5)
    assert off.class_masks["UA_DETRAC"]["train"] is None


def test_coarse_datasets_annotate_no_class(tmp_path):
    root = convert(tmp_path, ["waymo"])
    j = json.load(open(os.path.join(root, "Waymo", "annotated_classes.json")))
    assert j["annotated"] == [] and j["annotated_indices"] == []
    ds = JointDataset(data_root=root, datasets=["Waymo"], splits=["train"], class_masking=True, num_classes=5)
    assert ds.class_masks["Waymo"]["train"].tolist() == [0, 0, 0, 0, 0]


def _loss_grad(mask_rows):
    from models.deim.builder import build_detection_criterion
    inner = build_detection_criterion(5).deim
    torch.manual_seed(0)
    logits = torch.randn(2, 6, 5, requires_grad=True)
    boxes = torch.rand(2, 6, 4) * 0.5 + 0.2
    targets = [{"labels": torch.tensor([1, 3]), "boxes": torch.tensor([[.4, .4, .2, .2], [.6, .6, .2, .2]])} for _ in range(2)]
    for t, m in zip(targets, mask_rows):
        if m is not None:
            t["class_mask"] = torch.tensor(m, dtype=torch.float32)
    idx = [(torch.tensor([0, 1]), torch.tensor([0, 1]))] * 2
    out = inner.loss_labels_mal({"pred_logits": logits, "pred_boxes": boxes}, targets, idx, 4.0)["loss_mal"]
    out.backward()
    return out.item(), logits.grad.clone()


def test_class_mask_removes_the_loss_and_gradient_of_unannotated_classes_only():
    l0, g0 = _loss_grad([None, None])
    l1, g1 = _loss_grad([[0, 1, 0, 1, 0], None])           # image 0 annotates only classes 1 and 3
    assert l1 < l0
    assert torch.equal(g1[0][:, [0, 2, 4]], torch.zeros(6, 3))                  # masked columns: exactly zero gradient
    assert torch.allclose(g1[0][:, [1, 3]], g0[0][:, [1, 3]])                   # annotated columns: unchanged
    assert torch.equal(g1[1], g0[1])                                            # the unmasked image is unchanged
    l2, g2 = _loss_grad([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])                     # coarse datasets: no classification loss
    assert l2 == 0.0 and g2.abs().sum() == 0


def test_mask_over_a_class_agnostic_encoder_head_uses_any_annotated_class():
    from models.deim.deim_criterion import DEIMCriterion
    logits = torch.zeros(2, 3, 1)
    t = [{"class_mask": torch.tensor([0., 1., 0.])}, {"class_mask": torch.tensor([0., 0., 0.])}]
    m = DEIMCriterion._class_mask(t, logits)
    assert m.shape == (2, 1, 1) and m.flatten().tolist() == [1.0, 0.0]


# ---------------------------------------------------------------------------------------------- config generator
def test_gen_mix_config_writes_consistent_configs_and_warns(tmp_path, capsys):
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import gen_mix_config as G
    cfgs, warns = G.main(["--name", "t", "--size", "S", "--datasets", "Vehicles", "BDD100K_MOT:train", "KITTI_Tracking", "Waymo",
                          "--mix", "4", "2", "1", "1", "--out-dir", str(tmp_path), "--no-check-data"])
    s2 = yaml.safe_load(open(tmp_path / "t.yaml"))
    s1 = yaml.safe_load(open(tmp_path / "pretrain_t.yaml"))
    assert s2["DATASETS"] == ["Vehicles", "BDD100K_MOT", "KITTI_Tracking", "Waymo"] and s2["DATASET_SPLITS"] == ["train"] * 4
    assert s2["DATASET_MIX"] == [4.0, 2.0, 1.0, 1.0] and s2["DATASET_MAX_INTERVALS"] == [4, 1, 2, 2]     # 5 fps -> 1, 10 fps -> 2
    assert s2["CLASS_MASKING"] is True and s2["NUM_CLASSES"] == 5 and s1["ONLY_DETR"] is True
    assert s2["INFERENCE_DATASET"] == "Vehicles" and s2["DETR_PRETRAIN"].endswith("_t_stage1.pth")
    assert s1["DETR_PRETRAIN"].endswith("deim_s_vehicles_converted.pth")
    assert any("KITTI_Tracking" in w and "NON-COMMERCIAL" in w for w in warns) and any("Waymo" in w and "coarse" in w for w in warns)
    cfgs, _ = G.main(["--name", "c", "--datasets", "UA_DETRAC", "--from-coco", "--class-set", "vehicles5_person", "--out-dir",
                      str(tmp_path), "--no-check-data"])
    assert cfgs[1]["DETR_CLASS_INDICES"] == [1, 2, 3, 5, 7, 0] and cfgs[1]["NUM_CLASSES"] == 6
    assert cfgs[1]["NUM_KEYPOINTS"] == 0
    cfgs, _ = G.main(["--name", "k", "--datasets", "Vehicles", "COCOKeypoints", "--num-keypoints", "31", "--out-dir", str(tmp_path),
                      "--no-check-data"])
    assert cfgs[2]["NUM_KEYPOINTS"] == 31


def test_gen_mix_config_rejects_bad_mix(tmp_path):
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import gen_mix_config as G
    with pytest.raises(SystemExit):
        G.main(["--name", "t", "--datasets", "Vehicles", "UAVDT", "--mix", "1", "--out-dir", str(tmp_path), "--no-check-data"])


def test_eval_datasets_prints_one_command_per_dataset(capsys):
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import eval_datasets as E
    E.main(["--config", "c.yaml", "--model", "outputs/x/checkpoint_9.pth", "--datasets", "Vehicles:val", "UA_DETRAC", "--outputs-dir", "o"])
    lines = [l for l in capsys.readouterr().out.splitlines() if l.startswith("accelerate")]
    assert len(lines) == 2 and "--inference-dataset UA_DETRAC --inference-split val" in lines[1]


# ---------------------------------------------------------------------------------------------- end to end
@pytest.mark.slow
def test_end_to_end_cpu_training_and_evaluation_on_two_datasets(data, tmp_path):
    root, _ = data
    from models.deim.builder import build_deim
    torch.manual_seed(0)
    ck = tmp_path / "det.pth"
    torch.save({"model": build_deim("N", num_classes=5).state_dict()}, ck)
    cmd = [sys.executable, "train.py", "--config-path", "configs/motip2/smoke_vehicles_stage2.yaml", "--data-root", root,
           "--detr-pretrain", str(ck), "--exp-name", "mix", "--outputs-dir", str(tmp_path / "mix"),
           "--datasets", "UA_DETRAC", "BDD100K_MOT", "--dataset-splits", "train", "train", "--dataset-mix", "0.5", "0.5",
           "--dataset-max-intervals", "4", "1", "--class-masking", "True", "--samples-per-epoch", "8", "--epochs", "1",
           "--sample-lengths", "6", "--rel-pe-length", "6", "--miss-tolerance", "6",
           "--inference-dataset", "UA_DETRAC", "--inference-split", "val"]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, timeout=1500)
    out = re.sub(r"\x1b\[[0-9;]*m", "", r.stdout + r.stderr)
    assert r.returncode == 0, out[-2000:]
    line = [l for l in out.splitlines() if "Finish epoch" in l][-1]
    vals = {k: float(v) for k, v in re.findall(r"(\w+) = (-?[0-9.]+)", line)}
    assert all(map(lambda v: v == v and abs(v) < 1e6, vals.values())) and vals["detr_loss"] > 0
    assert "Eval epoch" in out


def test_class_mask_survives_the_real_augmentation_pipeline_and_reaches_the_detection_targets(data):
    root, _ = data
    from data import build_dataset
    from data.util import collate_fn
    from train import annotations_to_flatten_detr_targets
    from utils.misc import yaml_to_dict
    from configs.util import load_super_config
    cfg = yaml_to_dict(os.path.join(ROOT, "configs/motip2/smoke_vehicles_stage2.yaml"))
    cfg = load_super_config(cfg, cfg["SUPER_CONFIG_PATH"])
    cfg.update(DATA_ROOT=root, DATASETS=["UA_DETRAC", "BDD100K_MOT"], DATASET_SPLITS=["train", "train"], CLASS_MASKING=True,
               SAMPLE_LENGTHS=[4])
    ds = build_dataset(cfg)
    batch = [ds[{"dataset": d, "split": "train", "sequence": s, "frame_idxs": [0, 1, 2, 3]}]
             for d, s in (("UA_DETRAC", "MVI_20011"), ("BDD100K_MOT", "trainvid"))]
    targets = annotations_to_flatten_detr_targets(collate_fn(batch)["annotations"], "cpu")
    assert len(targets) == 8
    assert all(t["class_mask"].tolist() == [0, 1, 0, 1, 0] for t in targets[:4])      # UA-DETRAC frames
    assert all("class_mask" not in t for t in targets[4:])                             # BDD100K annotates every class
