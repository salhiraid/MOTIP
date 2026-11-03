# Gate 8 (unit, CPU): data pipeline and training-loop plumbing that cannot be checked by the loss/decoder tests.
import copy
import os
import random
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn
import yaml
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.transforms import _build_motip2_transforms
from data.seq_dataset import SeqDataset
from data.util import generate_scales, batch_multiscale, collate_fn
from utils.nested_tensor import NestedTensor
from train import deim_detr_forward, get_param_groups_motip2


def cfg():
    with open(os.path.join(os.path.dirname(__file__), "..", "configs", "motip2", "s_dancetrack.yaml")) as f:
        c = yaml.safe_load(f)
    c.update(AUG_BASE_SIZE=96, AUG_MAX_SIZE=176, NUM_ID_VOCABULARY=20, NUM_TRAINING_IDS=20)
    return c


def clip(T=6, n=5, static=False, h=120, w=200, seed=0):
    rng = np.random.default_rng(seed)
    images = [Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8)) for _ in range(T)]
    anns = []
    base = np.stack([rng.uniform(5, w - 60, n), rng.uniform(5, h - 50, n), rng.uniform(25, 50, n), rng.uniform(30, 45, n)], 1)
    for t in range(T):
        b = base if static else base + np.array([2 * t, t, 0, 0])
        anns.append({"id": torch.arange(1, n + 1), "category": torch.zeros(n, dtype=torch.int64),
                     "bbox": torch.tensor(b, dtype=torch.float32), "visibility": torch.ones(n), "is_legal": True})
    metas = [{"is_static": static, "is_begin": t == 0, "size_divisibility": 0} for t in range(T)]
    return images, anns, metas


@pytest.mark.parametrize("static", [False, True])
def test_motip2_augmentation_pipeline_keeps_boxes_valid(static):
    tf = _build_motip2_transforms(cfg())
    for seed in range(60):
        random.seed(seed); torch.manual_seed(seed)
        images, anns, metas = tf(*clip(static=static, seed=seed))
        assert images.shape[0] == 6 and images.shape[1] == 3 and images.dtype == torch.uint8
        H, W = images.shape[-2:]
        assert max(H, W) <= 176 * 2.0 + 1                      # zoom-out / crop never explode the size
        for a in anns:
            b = a["bbox"]
            assert b.shape[1] == 4 and (b[:, 2:] > 0).all()
            xyxy = torch.cat([b[:, :2] - b[:, 2:] / 2, b[:, :2] + b[:, 2:] / 2], 1)
            assert (xyxy >= -1e-3).all() and (xyxy <= 1 + 1e-3).all()      # normalised cxcywh inside the image
            assert a["trajectory_ann_idxs"].shape == a["unknown_ann_idxs"].shape
            assert a["unknown_real_id_labels"].shape == a["unknown_id_labels"].shape
            # every present slot points to a real annotation
            idx = a["trajectory_ann_idxs"][~a["trajectory_id_masks"]]
            assert (idx >= 0).all() and (idx < len(a["bbox"])).all()


def test_zoom_out_and_iou_crop_apply_one_transform_to_the_whole_clip():
    from data.transforms import MultiRandomZoomOut, MultiRandomIoUCrop
    for tf in (MultiRandomZoomOut(p=1.0), MultiRandomIoUCrop(p=1.0)):
        for seed in range(20):
            random.seed(seed); torch.manual_seed(seed)
            images, anns, _ = clip(static=True, seed=seed)
            for a in anns:                                    # xywh -> xyxy like the real pipeline
                a["bbox"] = torch.cat([a["bbox"][:, :2], a["bbox"][:, :2] + a["bbox"][:, 2:]], 1)
            images2, anns2, _ = tf(images, anns, [{}] * 6)
            assert len({im.size for im in images2}) == 1                # same canvas/crop for every frame
            for a in anns2[1:]:                                         # static scene -> identical boxes in all frames
                assert torch.equal(a["bbox"], anns2[0]["bbox"])


def test_generate_scales_matches_deim_semantics():
    sc = generate_scales(800, 20)
    assert min(sc) == 576 and max(sc) == 992 and sc.count(800) >= 20 and all(s % 32 == 0 for s in sc)


def test_batch_multiscale_gives_one_common_unpadded_size():
    x = torch.rand(2, 3, 3, 96, 160)
    m = torch.zeros(2, 3, 96, 160, dtype=torch.bool)
    m[1, :, :, 128:] = True; x[1, ..., 128:] = 0                 # clip 1 is narrower (padded on the right)
    out = batch_multiscale(NestedTensor(x, m), scale_factor=1.25)
    assert out.tensors.shape == (2, 3, 3, 128, 192) and not out.mask.any()
    assert out.tensors.shape[-2] % 32 == 0 and out.tensors.shape[-1] % 32 == 0


def test_seq_dataset_inference_normalisation_is_optional(tmp_path):
    Image.fromarray(np.full((40, 64, 3), 30, dtype=np.uint8)).save(tmp_path / "a.jpg")
    kw = dict(seq_info={"width": 64, "height": 40, "length": 1}, image_paths=[str(tmp_path / "a.jpg")],
              max_shorter=32, max_longer=64)
    raw = SeqDataset(normalize=False, **kw)[0][0].tensors
    norm = SeqDataset(normalize=True, **kw)[0][0].tensors
    assert raw.min() >= 0 and raw.max() <= 1 and abs(raw.mean().item() - 30 / 255) < 0.01     # DEIM: [0,1] inputs
    assert norm.min() < 0                                                                      # legacy: ImageNet-normalised
    # the processed size follows Paper §4.1: shorter side 32 here (800 in the real config), aspect preserved
    assert raw.shape[-2:] == (32, 51) or raw.shape[-2] == 32


class _StubModel(nn.Module):
    def forward(self, frames=None, part=None, targets=None, lightweight=False):
        fid = frames.tensors[:, 0, 0, 0]                      # each frame is filled with its flat (b, t) id
        z = lambda d: fid[:, None, None].expand(-1, 2, d).clone()
        return {"pred_logits": z(1), "pred_boxes": z(4), "outputs": z(3), "pos_embeds": z(3)}


class _StubCriterion:
    def __call__(self, outputs, targets):
        return {"loss": outputs["pred_logits"].sum() * 0 + 1}, self.match(outputs, targets)

    def match(self, outputs, targets):
        return [(torch.tensor([int(v)]), torch.tensor([0])) for v in outputs["pred_logits"][:, 0, 0]]


@pytest.mark.parametrize("n_train", [0, 1, 2, 4, 100])
def test_deim_detr_forward_returns_all_frames_in_b_t_order(n_train):
    B, T = 2, 4
    x = torch.arange(B * T, dtype=torch.float32).view(B, T, 1, 1, 1).expand(B, T, 3, 8, 8).contiguous()
    images = NestedTensor(x, torch.zeros(B, T, 8, 8, dtype=torch.bool))
    anns = [[{"bbox": torch.rand(1, 4), "category": torch.zeros(1, dtype=torch.int64)} for _ in range(T)] for _ in range(B)]
    for seed in range(5):
        torch.manual_seed(seed)
        out, losses, idx = deim_detr_forward(_StubModel(), _StubCriterion(), images, anns, torch.device("cpu"),
                                             num_train_frames=n_train, num_no_grad_chunk=3)
        for i in range(B * T):
            assert out["pred_logits"][i, 0, 0] == i and out["outputs"][i, 0, 0] == i
            assert int(idx[i][0][0]) == i                       # the matching of frame i belongs to frame i
        assert ("loss" in losses) == (min(n_train, T) > 0)      # detection loss only if some frame has gradient


def test_param_groups_follow_paper_lrs_and_no_weight_decay_on_norm_bias_embed():
    from models.motip import build as build_motip
    c = cfg()
    model, _ = build_motip(c)
    groups = get_param_groups_motip2(model, c)
    ids = {id(p): g for g in groups for p in g["params"]}
    assert sum(len(g["params"]) for g in groups) == sum(p.requires_grad for p in model.parameters())
    for n, p in model.named_parameters():
        if not p.requires_grad:                                  # e.g. DEIM's fixed `up` / `reg_scale`
            assert id(p) not in ids
            continue
        g = ids[id(p)]
        scale = c["LR_BACKBONE_SCALE"] if "detr.backbone" in n else (2.0 if n.startswith(("id_decoder", "trajectory")) else 1.0)
        assert g["lr"] == pytest.approx(c["LR"] * scale) and g["lr_scale"] == scale
        if p.ndim <= 1 or n.endswith(".bias") or "norm" in n or "embed" in n:
            assert g["weight_decay"] == 0.0, n
        else:
            assert g["weight_decay"] == c["WEIGHT_DECAY"], n


def test_aggregate_results_mean_std_over_seeds(tmp_path):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
    from aggregate_results import parse, aggregate
    md = tmp_path / "R.md"
    md.write_text("| Date | Git commit | Config | Seed | Benchmark | Split | HOTA | DetA | AssA | MOTA | IDF1 | IDSW | Latency (ms) | Checkpoint | Notes |\n"
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n"
                  + "".join(f"| d | c | x.yaml | {s} | DanceTrack | val | {h} | 74.0 | 56.5 | 82.9 | 68.8 | 2489 | — | — | variant=full |\n"
                            for s, h in [(1, 64.0), (2, 65.0), (3, 64.5)]))
    r = aggregate(parse(str(md)))[("DanceTrack", "val", "full")]
    assert r["n_seeds"] == 3 and r["stats"]["HOTA"][0] == pytest.approx(64.5) and r["stats"]["HOTA"][1] == pytest.approx(0.5)
