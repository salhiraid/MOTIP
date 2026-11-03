# BEHAVIOURAL tests: not "does it run / are the shapes right" but "does each mechanism do what the paper claims".
# A toy association task on detection-feature level (tests/helpers/toy_task.py) isolates one mechanism per test and
# has a CONTROL where the mechanism is removed and the model must fail. Seeds are fixed; each model trains in seconds on CPU.
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from helpers.toy_task import make_batch, make_decoder, train_eval
import models.motip.rope as rope_module

T = 7


def acc_for(content, anchor, steps=250, **dk):
    dec = make_decoder(spatial_anchor=anchor, **dk)
    mk = lambda B, s: make_batch(B, T=T, content=content, seed=s, with_pos=anchor)
    return train_eval(dec, mk, mk, steps=steps)[0]


# ---- Spatial Anchor (Paper §3.4) --------------------------------------------------------------------------------------
def test_spatial_anchor_is_what_lets_the_decoder_separate_identically_looking_objects():
    """Objects with IDENTICAL appearance can only be told apart by where they are. With p_i added to the detection tokens the
    decoder learns the association; without it, the very same model/data/optimiser stays at chance (1/4)."""
    with_anchor = acc_for("identical", True)
    without = acc_for("identical", False)
    assert with_anchor > 0.9, with_anchor
    assert without < 0.40, without                      # chance = 0.25
    assert acc_for("distinct", False) > 0.97            # sanity: with distinguishable appearance no anchor is needed


# ---- temporal encoding (Paper §3.1) -------------------------------------------------------------------------------------
def stale_acc(pos_enc, steps, seed=0):
    dec = make_decoder(pos_enc=pos_enc, seed=seed)
    mk = lambda B, s: make_batch(B, T=T, content="distinct", stale_labels=True, seed=s, with_pos=False)
    return train_eval(dec, mk, mk, steps=steps, query_from=T - 1, seed=seed)[0]


def test_rope_learns_to_trust_the_most_recent_frame_and_the_control_without_it_cannot():
    """History older than one frame carries consistently WRONG labels (majority vote over the context is wrong); only a model
    that knows which context frame is one step back can answer. RoPE learns it from every initialisation tried (measured
    95-98 % over 5 inits); with the rotation removed the same architecture is at chance."""
    for seed in (0, 1):
        assert stale_acc("rope", 400, seed) > 0.9
    original = rope_module.apply_rope
    rope_module.apply_rope = lambda x, positions, base=10000.0: x          # CONTROL: no temporal encoding
    try:
        assert stale_acc("rope", 400, 0) < 0.4
    finally:
        rope_module.apply_rope = original


def test_legacy_learned_relative_bias_path_is_functional():
    """The MOTIP baseline path (POS_ENC=learned_rel) must stay functional (it is the 'Rel. PE' ablation of Tab. 6). On this
    task it is initialisation-sensitive (measured: learns from 2 of 5 inits in 600 steps, 4 of 5 in 1200, whereas RoPE learns
    from 5 of 5 in 400 -- the direction of the paper's Tab. 6), so the assertion is that it CAN learn (init seed 1)."""
    assert stale_acc("learned_rel", 600, seed=1) > 0.9


# ---- additive fusion (Paper §3.1) ----------------------------------------------------------------------------------------
def test_additive_fusion_with_half_the_width_is_as_good_as_concatenation_on_appearance_matching():
    add = make_decoder(id_fusion="add", pos_enc="rope")
    cat = make_decoder(id_fusion="concat", pos_enc="learned_rel", D=48)
    mk = lambda B, s: make_batch(B, T=T, content="distinct", seed=s, with_pos=False)
    a, c = train_eval(add, mk, mk, steps=250)[0], train_eval(cat, mk, mk, steps=250)[0]
    assert add.model_dim * 2 == cat.model_dim                      # half the hidden width
    assert a > 0.97 and a >= c - 0.03, (a, c)


# ---- Spatial ID Loss (Paper §3.3) -----------------------------------------------------------------------------------------
def spatial_loss_run(spatial, noise=0.06, seed=0, steps=500):
    from helpers.toy_spatial import batch, D, K
    from models.motip.id_criterion import IDCriterion, giou_elementwise
    torch.manual_seed(seed)
    dec = make_decoder(D=D, K=K, spatial_anchor=True)
    crit = IDCriterion(1.0, False, spatial=spatial, iou_weight=False)
    opt = torch.optim.AdamW(dec.parameters(), lr=2e-3)
    dec.train()
    for it in range(steps):
        si, extra = batch(24, 1000 + it, noise)
        lg, lb, mk = dec(si, False)
        loss = crit(lg, lb, mk, extra)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(dec.parameters(), 1.0); opt.step()
    dec.eval()
    with torch.no_grad():
        si, extra = batch(512, 777, noise)
        lg, _, _ = dec(si, False)
        lg, lab = lg[:, :, 1:], si["unknown_id_labels"][:, :, 1:]
        pred = lg.argmax(-1)
        gt, real = extra["gt_boxes"][:, :, 1:], extra["real_id_labels"][:, :, 1:]
        slot = (real[..., None, :] == pred[..., None]).float().argmax(-1)
        pb = torch.gather(gt, 3, slot[..., None].expand(-1, -1, -1, -1, 4))
        wrong_dist = (1 - giou_elementwise(pb, gt))[pred != lab].mean().item()
        ex = {k: v[:, :, 1:] for k, v in extra.items()}
        _, e_cost = IDCriterion(1.0, False, spatial=True)._spatial_terms(lg, torch.zeros_like(lab, dtype=torch.bool), ex)
        return (pred == lab).float().mean().item(), wrong_dist, e_cost.mean().item()


@pytest.mark.parametrize("seed", [0, 1])
def test_spatial_id_loss_makes_the_decoders_uncertainty_geometrically_local(seed):
    """With noisy positions the decoder is sometimes unsure. Adding the spatial terms lowers the expected localisation cost
    sum_c p_c(1-GIoU) and moves the wrong predictions onto nearer objects, while the decoder still identifies most objects.
    (At heavier noise this costs accuracy -- the newborn class has zero localisation cost -- see RESULTS.md.)"""
    ce_acc, ce_dist, ce_cost = spatial_loss_run(False, seed=seed)
    sp_acc, sp_dist, sp_cost = spatial_loss_run(True, seed=seed)
    assert ce_acc > 0.9 and sp_acc > 0.9
    assert sp_cost < 0.95 * ce_cost, (sp_cost, ce_cost)
    assert sp_dist < ce_dist, (sp_dist, ce_dist)


# ---- detector (Paper §3.1 / App. A detection loss) --------------------------------------------------------------------------
@pytest.mark.slow
def test_detector_with_the_deim_losses_overfits_a_tiny_batch():
    """DEIM-N from random initialisation, the real criterion (matching + MAL/L1/GIoU/FGL/DDF + denoising), 300 steps on 4 fixed
    images of coloured rectangles: the best-matching prediction of every ground-truth box must become accurate AND confident.
    (Measured: IoU 0.215 -> 0.978, score 0.017 -> 0.957. A from-scratch detector cannot be trained to convergence on this CPU,
    so this checks the learning signal, not final accuracy.)"""
    import numpy as np
    from torchvision.ops import box_iou
    from models.deim.builder import build_deim, build_detection_criterion
    from utils.box_ops import box_cxcywh_to_xyxy

    rng = np.random.default_rng(0)
    H, W = 192, 320
    imgs, tg = [], []
    for _ in range(4):
        im = np.full((H, W, 3), 45, dtype=np.uint8)
        boxes = []
        slots = [(10, 10), (170, 10), (10, 100), (170, 100)]
        for k in range(int(rng.integers(3, 5))):
            w, h = int(rng.integers(50, 90)), int(rng.integers(30, 60))
            x0, y0 = slots[k][0] + int(rng.integers(0, 40)), slots[k][1] + int(rng.integers(0, 25))
            im[y0:y0 + h, x0:x0 + w] = rng.integers(120, 235, 3)
            boxes.append([(x0 + w / 2) / W, (y0 + h / 2) / H, w / W, h / H])
        imgs.append(torch.from_numpy(im).permute(2, 0, 1).float() / 255)
        tg.append({"labels": torch.from_numpy(rng.integers(0, 5, len(boxes))).long(), "boxes": torch.tensor(boxes, dtype=torch.float32)})
    x = torch.stack(imgs)
    torch.manual_seed(0)
    model = build_deim("N", num_classes=5, num_queries=100).train()
    crit = build_detection_criterion(5)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    def quality():
        model.eval()
        with torch.no_grad():
            o = model(x)
        model.train()
        ious, scores = [], []
        for i, t in enumerate(tg):
            iou = box_iou(box_cxcywh_to_xyxy(t["boxes"]), box_cxcywh_to_xyxy(o["pred_boxes"][i]))
            best, idx = iou.max(1)
            ious.append(best)
            scores.append(o["pred_logits"][i].sigmoid().max(-1)[0][idx])
        return torch.cat(ious).mean().item(), torch.cat(scores).mean().item()

    iou0, sc0 = quality()
    assert iou0 < 0.4 and sc0 < 0.1
    for _ in range(300):
        out = model(x, targets=tg)
        losses, _ = crit(out, tg)
        loss = crit.total(losses)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1); opt.step()
    iou1, sc1 = quality()
    assert iou1 > 0.85 and sc1 > 0.7, (iou1, sc1)
