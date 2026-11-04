# Per-detection keypoints: decoding inside the box, losses (analytic OKS, label semantics independent of the batch), behaviour
# (the head can learn), augmentation consistency, loader, converter of the mmdet branches, tracker output.
import math
import os
import sys

import pytest
import torch

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools", "vehicles"))

from models.deim.keypoints import KeypointHead, KeypointLoss, decode_inside_box, keypoint_pixel_error   # noqa: E402
from data import transforms as TR                                                                          # noqa: E402


# ------------------------------------------------------------------------------------------------ decoding
def test_decoded_points_lie_inside_the_box_and_zero_logits_give_the_centre():
    boxes = torch.tensor([[[0.5, 0.5, 0.2, 0.4], [0.9, 0.1, 0.4, 0.4]]])             # second box sticks out of the image
    xy = decode_inside_box(torch.zeros(1, 2, 6), boxes)                               # K = 3
    assert xy.shape == (1, 2, 3, 2)
    assert torch.allclose(xy[0, 0], torch.tensor([[0.5, 0.5]] * 3))
    # clamped box [0.7, 1.0] x [0.0, 0.3] -> centre (0.85, 0.15)
    assert torch.allclose(xy[0, 1, 0], torch.tensor([0.85, 0.15]))
    big = decode_inside_box(torch.randn(1, 2, 6) * 10, boxes)
    x1, x2 = torch.tensor([0.4, 0.7]), torch.tensor([0.6, 1.0])
    assert ((big[0, :, :, 0] >= x1[:, None] - 1e-6) & (big[0, :, :, 0] <= x2[:, None] + 1e-6)).all()


def test_keypoint_losses_cannot_move_the_box():
    head = KeypointHead(16, 4, 2)
    boxes = torch.tensor([[[0.5, 0.5, 0.3, 0.3]]], requires_grad=True)
    xy, vis = head(torch.randn(1, 1, 16), boxes)
    (xy.sum() + vis.sum()).backward()
    assert boxes.grad is None or boxes.grad.abs().sum() == 0
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in head.xy_branch.parameters())


# ------------------------------------------------------------------------------------------------ losses
def make_batch(K=3, wh=(200.0, 100.0)):
    """One image, one query matched to one GT. Returns outputs, targets, indices."""
    gt = torch.tensor([[[0.40, 0.50], [0.50, 0.55], [0.45, 0.60]]])                  # (1, K, 2) normalised
    targets = [{"boxes": torch.tensor([[0.5, 0.5, 0.4, 0.4]]), "keypoints": gt, "keypoints_vis": torch.tensor([[2.0, 2.0, 2.0]]),
                "size_wh": torch.tensor(wh)}]
    outputs = {"pred_keypoints": gt.clone()[None].expand(1, 2, K, 2).clone(), "pred_keypoint_vis": torch.zeros(1, 2, K)}
    return outputs, targets, [(torch.tensor([1]), torch.tensor([0]))]


def test_perfect_prediction_has_zero_xy_and_oks_loss():
    out, tg, idx = make_batch()
    out["pred_keypoint_vis"] = torch.full((1, 2, 3), 20.0)
    l = KeypointLoss(3)(out, tg, idx)
    assert l["loss_kp_xy"].item() == 0.0 and l["loss_kp_oks"].item() < 1e-6 and l["loss_kp_vis"].item() < 1e-6


def test_oks_matches_the_coco_formula_in_pixels():
    out, tg, idx = make_batch(wh=(200.0, 100.0))
    out["pred_keypoints"][0, 1, 0] += torch.tensor([0.05, 0.0])                        # keypoint 0 off by 0.05*200 = 10 px
    sig = [0.04, 0.05, 0.06]
    L = KeypointLoss(3, weight_xy=0, weight_vis=0, weight_oks=1.0, sigmas=sig)
    got = L(out, tg, idx)["loss_kp_oks"].item()
    area = 0.4 * 0.4 * 200 * 100                                                       # GT box area in px^2
    e0 = math.exp(-(10.0 ** 2) / (2 * area * (2 * 0.04) ** 2))
    expected = 1 - (e0 + 1 + 1) / 3
    assert abs(got - expected) < 1e-6, (got, expected)


def test_label_semantics_do_not_depend_on_the_batch():
    """The user's mmdet loss decides 'v==2 is visible' only if the BATCH contains a 2 (else v>0). Here it is explicit."""
    out, tg, idx = make_batch()
    tg[0]["keypoints_vis"] = torch.tensor([[1.0, 1.0, 1.0]])                           # occluded-but-labelled everywhere
    L = KeypointLoss(3, weight_xy=0, weight_vis=1.0, weight_oks=0)
    out["pred_keypoint_vis"] = torch.full((1, 2, 3), 20.0)                              # model says 'visible'
    wrong = L(out, tg, idx)["loss_kp_vis"].item()
    assert wrong > 10                                                                  # v=1 is NOT visible (target 0) -> large loss
    # same keypoints in a batch that also holds a v=2 instance elsewhere: identical result for this instance
    tg2 = tg + [{"boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]]), "keypoints": torch.zeros(1, 3, 2),
                 "keypoints_vis": torch.tensor([[2.0, 2.0, 2.0]]), "size_wh": torch.tensor([200.0, 100.0])}]
    out2 = {k: torch.cat([v, v[:1] * 0], 0) for k, v in out.items()}
    only_first = KeypointLoss(3, weight_xy=0, weight_vis=1.0, weight_oks=0)(out2, tg2, idx + [(torch.tensor([]).long(), torch.tensor([]).long())])
    assert abs(only_first["loss_kp_vis"].item() - wrong) < 1e-6
    # binary datasets: say v=1 means visible
    binary = KeypointLoss(3, weight_xy=0, weight_vis=1.0, weight_oks=0, labeled_values=(1,), visible_values=(1,))
    assert binary(out, tg, idx)["loss_kp_vis"].item() < 1e-6


def test_unlabelled_points_are_not_supervised_and_unlabelled_objects_are_skipped():
    out, tg, idx = make_batch()
    out["pred_keypoints"] = out["pred_keypoints"].clone().requires_grad_(True)
    tg[0]["keypoints_vis"] = torch.tensor([[2.0, 0.0, 2.0]])                           # middle point not labelled
    out["pred_keypoints"].data[0, 1, 1] += 0.3                                         # ...and wildly wrong
    l = KeypointLoss(3)(out, tg, idx)
    assert l["loss_kp_xy"].item() == 0.0
    (l["loss_kp_xy"] + l["loss_kp_oks"]).backward()
    assert out["pred_keypoints"].grad[0, 1, 1].abs().sum() == 0
    tg[0]["keypoints_vis"] = torch.zeros(1, 3)                                         # nothing labelled -> zero loss, graph kept
    z = KeypointLoss(3)(out, tg, idx)
    assert all(v.item() == 0.0 and v.requires_grad for v in z.values())


def test_head_learns_keypoints_and_the_control_does_not():
    torch.manual_seed(0)
    n, K, D = 12, 5, 32
    emb = torch.randn(n, D)
    gt = torch.rand(n, K, 2) * 0.8 + 0.1
    boxes = torch.tensor([0.5, 0.5, 1.0, 1.0]).expand(n, 4)                             # full-image boxes: keypoint = (u, v)
    tg = [{"boxes": boxes, "keypoints": gt, "keypoints_vis": torch.full((n, K), 2.0), "size_wh": torch.tensor([100.0, 100.0])}]
    idx = [(torch.arange(n), torch.arange(n))]
    errs = {}
    for lr in (0.0, 3e-3):
        torch.manual_seed(1)
        head = KeypointHead(D, K, 2)
        opt = torch.optim.Adam(head.parameters(), lr=max(lr, 1e-12))
        loss_fn = KeypointLoss(K)
        for _ in range(400):
            xy, vis = head(emb[None], boxes[None])
            loss = sum(loss_fn({"pred_keypoints": xy, "pred_keypoint_vis": vis}, tg, idx).values())
            opt.zero_grad(); loss.backward()
            if lr:
                opt.step()
        errs[lr] = keypoint_pixel_error(xy[0].detach(), gt, torch.full((n, K), 2.0), (100.0, 100.0)).item()
    assert errs[3e-3] < 0.35 * errs[0.0], errs
    assert errs[3e-3] < 6.0                                                             # px on a 100 px image


# ------------------------------------------------------------------------------------------------ detector wrapper
def test_wrapper_outputs_keypoints_inside_the_final_boxes_and_lightweight_skips_them():
    from models.deim.builder import build_deim
    torch.manual_seed(0)
    m = build_deim("N", num_classes=5, num_queries=40, num_denoising=0, num_keypoints=7).eval()
    x = torch.rand(1, 3, 128, 192)
    with torch.no_grad():
        o = m(x)
        light = m(x, lightweight=True)
    assert o["pred_keypoints"].shape == (1, 40, 7, 2) and o["pred_keypoint_vis"].shape == (1, 40, 7)
    assert "pred_keypoints" not in light
    b = o["pred_boxes"][0]
    x1, x2 = (b[:, 0] - b[:, 2] / 2).clamp(0, 1), (b[:, 0] + b[:, 2] / 2).clamp(0, 1)
    k = o["pred_keypoints"][0]
    assert ((k[..., 0] >= x1[:, None] - 1e-5) & (k[..., 0] <= x2[:, None] + 1e-5)).all()
    plain = build_deim("N", num_classes=5, num_queries=40, num_denoising=0).eval()
    with torch.no_grad():
        assert "pred_keypoints" not in plain(x)


def test_pretrain_without_keypoint_head_loads_with_a_warning():
    from models.deim.builder import build_deim
    from models.deim.wrapper import load_deim_pretrain
    plain = build_deim("N", num_classes=5, num_queries=40, num_denoising=0)
    kp = build_deim("N", num_classes=5, num_queries=40, num_denoising=0, num_keypoints=4)
    with pytest.warns(UserWarning, match="keypoint_head"):
        load_deim_pretrain(kp, state=plain.state_dict())


def test_criterion_adds_keypoint_losses_on_the_matched_queries_end_to_end():
    from models.deim.builder import build_deim, build_detection_criterion
    torch.manual_seed(0)
    K = 4
    m = build_deim("N", num_classes=5, num_queries=40, num_denoising=20, num_keypoints=K).train()
    crit = build_detection_criterion(5, num_keypoints=K)
    x = torch.rand(2, 3, 128, 192)
    box = torch.tensor([[0.5, 0.5, 0.3, 0.4]])
    targets = [{"labels": torch.tensor([1]), "boxes": box, "keypoints": torch.full((1, K, 2), 0.5),
                "keypoints_vis": torch.full((1, K), 2.0), "size_wh": torch.tensor([192.0, 128.0])},
               {"labels": torch.tensor([3]), "boxes": box, "keypoints": torch.zeros(1, K, 2), "keypoints_vis": torch.zeros(1, K),
                "size_wh": torch.tensor([192.0, 128.0])}]                              # image 2: no keypoint labels
    out = m(x, targets=targets)
    losses, idx = crit(out, targets)
    assert {"loss_kp_xy", "loss_kp_vis", "loss_kp_oks"} <= set(losses) and all(torch.isfinite(v) for v in losses.values())
    sum(losses.values()).backward()
    head_grads = [p.grad for p in m.keypoint_head.parameters()]
    assert all(g is not None for g in head_grads) and sum(g.abs().sum() for g in head_grads) > 0


# ------------------------------------------------------------------------------------------------ augmentation
def ann(kp, vis, box=(10.0, 10.0, 50.0, 40.0)):
    return {"bbox": torch.tensor([box]), "category": torch.tensor([1]), "id": torch.tensor([7]), "visibility": torch.tensor([1.0]),
            "keypoints": torch.tensor([kp], dtype=torch.float32), "keypoints_vis": torch.tensor([vis], dtype=torch.float32),
            "is_legal": True}


def img(w=100, h=80):
    from PIL import Image
    return Image.new("RGB", (w, h))


def test_flip_mirrors_points_and_swaps_pairs_and_refuses_without_pairs():
    a = ann([[20, 20], [30, 25], [40, 30]], [2, 1, 0])
    t = TR.MultiRandomHorizontalFlip(p=1.0, keypoint_flip_pairs=[[0, 1]])
    _, (b,), _ = t([img()], [a], [{}])
    assert b["keypoints"][0].tolist() == [[100 - 30, 25], [100 - 20, 20], [100 - 40, 30]]        # swapped 0<->1, mirrored x
    assert b["keypoints_vis"][0].tolist() == [1, 2, 0]
    assert b["bbox"][0].tolist() == [50, 10, 90, 40]                                              # (10,10,50,40) mirrored in w=100
    with pytest.raises(ValueError, match="KEYPOINT_FLIP_PAIRS"):
        TR.MultiRandomHorizontalFlip(p=1.0)([img()], [ann([[1, 1]], [2])], [{}])
    assert TR.flip_pair_permutation(4, [0, 3, 1, 2]) == [3, 2, 1, 0]                             # flat pair list (CLI form)


def test_builder_drops_the_flip_when_keypoints_have_no_pairs():
    with pytest.warns(UserWarning, match="DISABLED"):
        assert TR._flip_transforms({"NUM_KEYPOINTS": 31}) == []
    assert len(TR._flip_transforms({"NUM_KEYPOINTS": 31, "KEYPOINT_FLIP_PAIRS": [[0, 1]]})) == 1
    assert len(TR._flip_transforms({})) == 1


def test_resize_zoomout_crop_and_normalize_keep_points_on_the_object():
    a = ann([[20, 20], [60, 50]], [2, 2])
    (im,), (r,), _ = TR.MultiRandomResize(sizes=[160])([img()], [a], [{}])                       # 100x80 -> 200x160 (x2)
    assert r["keypoints"][0].tolist() == [[40, 40], [120, 100]] and r["bbox"][0].tolist() == [20, 20, 100, 80]
    import random
    random.seed(0)
    z = TR.MultiRandomZoomOut(p=1.0, scale_range=(2.0, 2.0))
    ims, (zz,), _ = z([img()], [ann([[20, 20]], [2])], [{}])
    shift = zz["bbox"][0, 0].item() - 10.0
    assert zz["keypoints"][0, 0].tolist() == [20 + shift, 20 + (zz["bbox"][0, 1].item() - 10.0)]
    # IoU crop: a point outside the crop becomes unlabelled, a point inside is shifted with the box
    c = TR.MultiRandomIoUCrop._crop_annotation(ann([[15, 15], [90, 70]], [2, 2], box=(10, 10, 95, 75)), 5, 5, 50, 40)
    assert c["keypoints"][0].tolist() == [[10.0, 10.0], [85.0, 65.0]] and c["keypoints_vis"][0].tolist() == [2, 0]
    n = TR.MultiNormalizeBoundingBoxes()
    _, (nn_,), _ = n(torch.zeros(1, 3, 80, 100), [ann([[50, 40]], [2])], [{}])
    assert nn_["keypoints"][0, 0].tolist() == [0.5, 0.5]


# ------------------------------------------------------------------------------------------------ loader
def test_loader_reads_gt_keypoints_and_zero_fills_other_datasets(tmp_path):
    import shutil
    sys.path.insert(0, os.path.join(ROOT, "tests"))
    from helpers import fake_datasets as F
    from data.converters import prepare
    from data.joint_dataset import JointDataset
    F.uavdt(str(tmp_path / "src"))
    prepare.main(["uavdt", "--root", str(tmp_path / "src"), "--out", str(tmp_path / "data"), "--workers", "1"])
    seq = tmp_path / "data" / "UAVDT" / "train" / "M0101"
    K = 3
    with open(seq / "gt" / "gt_keypoints.txt", "w") as f:
        for fr in range(1, F.N + 1):
            bx = F.box_a(fr)
            row = [fr, 1]                                                                   # object id 1 = the car
            for u, v in ((0.0, 0.0), (0.5, 0.5), (1.0, 1.0)):
                row += [bx[0] + u * bx[2], bx[1] + v * bx[3], 2]
            f.write(",".join(str(x) for x in row) + "\n")
    ds = JointDataset(data_root=str(tmp_path / "data"), datasets=["UAVDT"], splits=["train"], num_keypoints=K)
    a = ds.annotations["UAVDT"]["train"]["M0101"][4]
    ids = a["id"].tolist()
    car = ids.index(1)
    bx = F.box_a(5)
    assert torch.allclose(a["keypoints"][car][1], torch.tensor([bx[0] + bx[2] / 2, bx[1] + bx[3] / 2]), atol=0.02)
    assert a["keypoints_vis"][ids.index(2)].tolist() == [0, 0, 0]                           # the truck has no row -> unlabelled
    # a dataset without the file gets zeros lazily, with the right shapes
    ds2 = JointDataset(data_root=str(tmp_path / "data"), datasets=["UAVDT"], splits=["val"], num_keypoints=K)
    _, anns, _ = ds2[{"dataset": "UAVDT", "split": "val", "sequence": "M0203", "frame_idxs": [0, 1]}]
    assert anns[0]["keypoints"].shape == (2, K, 2) and anns[0]["keypoints_vis"].sum() == 0
    with pytest.raises(ValueError, match="NUM_KEYPOINTS"):
        JointDataset(data_root=str(tmp_path / "data"), datasets=["UAVDT"], splits=["train"], num_keypoints=5)[
            {"dataset": "UAVDT", "split": "train", "sequence": "M0101", "frame_idxs": [0, 1]}]


# ------------------------------------------------------------------------------------------------ converter
def test_converter_maps_the_mmdet_keypoint_branches_exactly():
    import convert_mmdet_deim as C
    from models.deim.builder import build_deim
    from models.deim.wrapper import load_deim_pretrain
    sys.path.insert(0, os.path.join(ROOT, "tests"))
    from test_convert_mmdet import make_target, to_mmdet_style
    K, L = 7, 2
    base = to_mmdet_style(make_target(0).state_dict())
    D = 256
    g = torch.Generator().manual_seed(3)
    src = dict(base)
    ref = {}
    for branch, name, out in (("xy_branch", "kps_xy_branches", 2 * K), ("vis_branch", "kps_vis_branches", K)):
        for i in range(2 * L + 1):
            if i % 2 == 0:                                                                    # Linear layers at 0, 2, ..., 2L
                last = i == 2 * L
                w, b = torch.randn(out if last else D, D, generator=g), torch.randn(out if last else D, generator=g)
                src[f"module.bbox_head.{name}.0.{i}.weight"], src[f"module.bbox_head.{name}.0.{i}.bias"] = w, b
                ref[f"keypoint_head.{branch}.{i}.weight"], ref[f"keypoint_head.{branch}.{i}.bias"] = w, b
    rest, kp, k, l = C.split_keypoint_tensors({kk.replace("module.", ""): v for kk, v in src.items()})
    assert (k, l) == (K, L) and set(kp) == set(ref) and not any("kps_" in x for x in rest)
    assert all(torch.equal(kp[x], ref[x]) for x in ref)
    # inconsistent branches are rejected
    bad = dict(src)
    bad["module.bbox_head.kps_vis_branches.0.4.weight"] = torch.randn(K + 1, D)
    with pytest.raises(AssertionError):
        C.split_keypoint_tensors({kk.replace("module.", ""): v for kk, v in bad.items()})
    # no keypoint branches -> unchanged behaviour
    assert C.split_keypoint_tensors({kk.replace("module.", ""): v for kk, v in base.items()})[1:] == ({}, 0, 0)


# ------------------------------------------------------------------------------------------------ end to end (CPU)
@pytest.mark.slow
def test_train_and_track_with_keypoints_end_to_end(tmp_path):
    import re
    import subprocess
    sys.path.insert(0, os.path.join(ROOT, "tests"))
    from helpers import fake_datasets as F
    from data.converters import prepare
    from models.deim.builder import build_deim
    F.detrac(str(tmp_path / "src"))
    prepare.main(["detrac", "--root", str(tmp_path / "src"), "--out", str(tmp_path / "data"), "--workers", "1"])
    K = 3
    for split, seq in (("train", "MVI_20011"), ("val", "MVI_39031")):
        with open(tmp_path / "data" / "UA_DETRAC" / split / seq / "gt" / "gt_keypoints.txt", "w") as f:
            for fr in range(1, F.N + 1):
                for oid, fn in ((1, F.box_a), (2, F.box_b)):
                    bx = fn(fr)
                    row = [fr, oid]
                    for u, v in ((0.2, 0.2), (0.5, 0.5), (0.8, 0.8)):
                        row += [bx[0] + u * bx[2], bx[1] + v * bx[3], 2]
                    f.write(",".join(str(x) for x in row) + "\n")
    torch.manual_seed(0)
    ck = tmp_path / "det.pth"
    torch.save({"model": build_deim("N", num_classes=5).state_dict()}, ck)                   # no keypoint head in the pretrain
    common = ["--data-root", str(tmp_path / "data"), "--num-keypoints", str(K)]
    cmd = [sys.executable, "train.py", "--config-path", "configs/motip2/smoke_vehicles_stage2.yaml", *common, "--detr-pretrain", str(ck),
           "--exp-name", "kp", "--outputs-dir", str(tmp_path / "kp"), "--datasets", "UA_DETRAC", "--dataset-splits", "train",
           "--samples-per-epoch", "8", "--epochs", "1", "--sample-lengths", "6", "--rel-pe-length", "6", "--miss-tolerance", "6",
           "--inference-dataset", "UA_DETRAC", "--inference-split", "val"]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, timeout=1500)
    out = re.sub(r"\x1b\[[0-9;]*m", "", r.stdout + r.stderr)
    assert r.returncode == 0, out[-2500:]
    line = [l for l in out.splitlines() if "Finish epoch" in l][-1]
    vals = {k: float(v) for k, v in re.findall(r"(\w+) = (-?[0-9.]+)", line)}
    assert {"loss_kp_xy", "loss_kp_vis", "loss_kp_oks"} <= set(vals) and vals["loss_kp_oks"] > 0 and vals["loss_kp_xy"] > 0
    ev = [sys.executable, "submit_and_evaluate.py", "--config-path", "configs/motip2/smoke_vehicles_stage2.yaml", *common,
          "--inference-mode", "submit", "--inference-model", str(tmp_path / "kp" / "checkpoint_0.pth"),
          "--inference-dataset", "UA_DETRAC", "--inference-split", "val", "--outputs-dir", str(tmp_path / "sub"), "--n-init", "1"]
    r = subprocess.run(ev, capture_output=True, text=True, cwd=ROOT, timeout=1500)
    assert r.returncode == 0, (r.stdout + r.stderr)[-2500:]
    base = tmp_path / "sub" / "submit" / "default" / "UA_DETRAC" / "val" / "checkpoint_0"
    kp_files = list((base / "keypoints").glob("*.txt"))
    assert kp_files
    rows = [l.split(",") for l in kp_files[0].read_text().splitlines()]
    assert rows and all(len(r_) == 4 + 3 * K for r_ in rows)                           # frame,id,class,score + (x,y,p) * K
    tracker_rows = sum(len(f.read_text().splitlines()) for f in (base / "tracker").glob("*.txt"))
    assert len(rows) == tracker_rows                                                   # one keypoint row per output track box
    xs = [float(r_[4]) for r_ in rows]
    assert min(xs) >= -1 and max(xs) <= F.W * 4                                        # frame pixels of the (resized) sequence


# ------------------------------------------------------------------------------------------------ COCO keypoint images
def test_coco_keypoint_images_become_static_sequences_with_scaled_keypoints(tmp_path):
    import json
    from PIL import Image
    from data.converters import prepare
    from data.joint_dataset import JointDataset
    from data.naive_sampler import NaiveSampler
    imgs = tmp_path / "imgs"
    imgs.mkdir()
    Image.new("RGB", (200, 100), (10, 20, 30)).save(imgs / "a.jpg")
    Image.new("RGB", (200, 100), (30, 20, 10)).save(imgs / "b.jpg")
    kp1 = [60, 30, 2, 80, 40, 1, 190, 90, 0, 0, 0, 0]                      # K = 4; third unlabelled, fourth empty
    coco = {"images": [{"id": 1, "file_name": "a.jpg", "width": 200, "height": 100}, {"id": 2, "file_name": "b.jpg", "width": 200, "height": 100}],
            "categories": [{"id": 1, "name": "car"}, {"id": 2, "name": "truck"}, {"id": 3, "name": "weirdo"}],
            "annotations": [{"image_id": 1, "category_id": 1, "bbox": [50, 20, 60, 40], "keypoints": kp1},
                            {"image_id": 1, "category_id": 3, "bbox": [0, 0, 20, 20], "keypoints": [0] * 12},
                            {"image_id": 2, "category_id": 2, "bbox": [10, 10, 40, 30], "keypoints": [20, 20, 2] * 4, "num_keypoints": 4}]}
    (tmp_path / "train.json").write_text(json.dumps(coco))
    prepare.main(["cocokp", "--images", str(imgs), "--train-json", str(tmp_path / "train.json"), "--out", str(tmp_path / "data"),
                  "--image-mode", "jpeg", "--resize-short", "50", "--workers", "1"])               # 200x100 -> 100x50 (x0.5)
    root = tmp_path / "data" / "COCOKeypoints"
    assert "isStatic=1" in (root / "train" / "img1" / "seqinfo.ini").read_text()
    ds = JointDataset(data_root=str(tmp_path / "data"), datasets=["COCOKeypoints"], splits=["train"], num_keypoints=4)
    a = ds.annotations["COCOKeypoints"]["train"]["img1"][0]
    assert a["category"].tolist() == [1] and a["bbox"][0].tolist() == [25, 10, 30, 20]                # `weirdo` dropped, box x0.5
    assert a["keypoints"][0, :2].tolist() == [[30, 15], [40, 20]] and a["keypoints_vis"][0].tolist() == [2, 1, 0, 0]
    assert ds.sequence_infos["COCOKeypoints"]["train"]["img1"]["is_static"] is True
    s = NaiveSampler(ds, [0], [5], [4], None, samples_per_epoch=6)
    s.prepare_for_epoch(0)
    assert all(i["frame_idxs"] == [0] * 5 for i in s.sample_infos)                                     # still image repeated T times
    ann_cls = json.load(open(root / "annotated_classes.json"))
    assert ann_cls["annotated"] == ["bicycle", "car", "motorcycle", "bus", "truck"]                  # taxonomy of the class map
