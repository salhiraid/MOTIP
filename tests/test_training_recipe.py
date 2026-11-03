# Paper App. A recipe items, asserted directly (not only through smoke runs).
import math
import os
import sys

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools", "vehicles"))

from models.deim.builder import build_detection_criterion
from models.deim.ema import ModelEMA
from train import lr_warmup_by_iters


def test_detection_loss_and_matcher_use_the_paper_weights():
    """App. A: MAL 1.0, L1 5.0, GIoU 2.0, FGL 0.15, DDF 1.5; focal alpha .75 gamma 1.5; matcher costs class 2, bbox 5, GIoU 2."""
    c = build_detection_criterion(1).deim
    assert c.weight_dict == {"loss_mal": 1, "loss_bbox": 5, "loss_giou": 2, "loss_fgl": 0.15, "loss_ddf": 1.5}
    assert list(c.losses) == ["mal", "boxes", "local"]
    assert (c.alpha, c.gamma) == (0.75, 1.5)
    m = c.matcher
    assert (m.cost_class, m.cost_bbox, m.cost_giou) == (2, 5, 2)


def test_ema_follows_the_paper_decay_and_warmup():
    """App. A: EMA decay 0.9995; warm-up 1000 steps in joint training, none in detector pretraining."""
    for warm, expect in ((1000, lambda k: 0.9995 * (1 - math.exp(-k / 1000))), (0, lambda k: 0.9995)):
        model = nn.Linear(3, 2, bias=False)
        ema = ModelEMA(model, decay=0.9995, warmups=warm)
        w0 = ema.module.weight.clone()
        target = torch.ones_like(w0)
        with torch.no_grad():
            model.weight.copy_(target)
        e = w0.clone()
        for k in range(1, 6):
            ema.update(model)
            d = expect(k)
            e = d * e + (1 - d) * target
            assert torch.allclose(ema.module.weight, e, atol=1e-6), (warm, k)
        assert ema.decay == 0.9995 and not any(p.requires_grad for p in ema.module.parameters())


def test_linear_lr_warmup_over_500_iterations_respects_group_multipliers():
    p = [nn.Parameter(torch.zeros(1)) for _ in range(3)]
    opt = torch.optim.AdamW([{"params": [p[0]], "lr_scale": 0.5}, {"params": [p[1]], "lr_scale": 2.0}, {"params": [p[2]]}], lr=2e-4)
    base = 2e-4
    lr_warmup_by_iters(opt, curr_iter=0, tgt_lr=base, warmup_iters=500)
    assert [g["lr"] for g in opt.param_groups] == pytest.approx([base / 500 * 0.5, base / 500 * 2.0, base / 500])
    lr_warmup_by_iters(opt, curr_iter=249, tgt_lr=base, warmup_iters=500)
    assert opt.param_groups[2]["lr"] == pytest.approx(base * 0.5)
    lr_warmup_by_iters(opt, curr_iter=499, tgt_lr=base, warmup_iters=500)     # last warm-up step reaches the full LR
    assert [g["lr"] for g in opt.param_groups] == pytest.approx([base * 0.5, base * 2.0, base])


def test_optimizer_choices_in_the_config_match_the_paper():
    import yaml
    c = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "..", "configs", "motip2", "s_dancetrack.yaml")))
    assert c["MAX_CLIP_NORM"] == 0.1 and c["LR_SCHEDULER"] == "constant" and c["ACCUMULATE_STEPS"] == 1   # "no gradient accumulation"
    src = open(os.path.join(os.path.dirname(__file__), "..", "train.py")).read()
    assert "AdamW(" in src and "betas=(0.9, 0.999)" in src


@pytest.fixture(scope="module")
def vehicle_root(tmp_path_factory):
    import shutil
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg missing")
    import make_synthetic_vehicle_videos as G
    import prepare_vehicle_dataset as P
    d = tmp_path_factory.mktemp("sampler")
    G.main(["--out", str(d / "src"), "--videos-per-site", "1", "--frames", "200"])
    P.main(["--csv", str(d / "src" / "dataset.csv"), "--videos-dir", str(d / "src"), "--out", str(d / "out"), "--val-frac", "0.3",
            "--resize-short", "96", "--max-long", "160", "--workers", "2", "--min-frames", "30"])
    return str(d / "out")


def test_clip_sampling_follows_the_paper(vehicle_root):
    """App. A: clips of T=30 frames with ONE temporal interval per clip drawn uniformly from {1,...,4}; 40k samples per epoch."""
    from collections import Counter
    from data.joint_dataset import JointDataset
    from data.naive_sampler import NaiveSampler
    ds = JointDataset(data_root=vehicle_root, datasets=["Vehicles"], splits=["train"])
    sampler = NaiveSampler(ds, [0], [30], [4], None, samples_per_epoch=500)
    sampler.prepare_for_epoch(0)
    assert len(sampler) == 500                                              # fixed epoch length (cycled if the data is smaller)
    intervals = Counter()
    epoch0 = [i["frame_idxs"] for i in sampler]
    for fr in epoch0:
        assert len(fr) == 30
        diffs = {b - a for a, b in zip(fr, fr[1:])}
        assert len(diffs) == 1 and 1 <= next(iter(diffs)) <= 4             # a single interval per clip, in {1..4}
        intervals[next(iter(diffs))] += 1
    assert set(intervals) == {1, 2, 3, 4}                                   # all four intervals occur ...
    assert all(v >= 0.08 * 500 for v in intervals.values()), intervals      # ... roughly uniformly (4th is limited by clip ends)
    sampler.prepare_for_epoch(1)                                            # a fresh draw every epoch
    assert [i["frame_idxs"] for i in sampler] != epoch0
    sampler.prepare_for_epoch(0)                                            # ... but reproducible for the same epoch
    assert [i["frame_idxs"] for i in sampler] == epoch0
