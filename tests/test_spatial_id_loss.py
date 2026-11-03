# Gate 6 (unit, CPU): Spatial ID Loss. PLAN.md §6 Gate 6 (Paper Sec. 3.3, App. A).
import os
import sys

import pytest
import torch
import torch.nn.functional as F
import torchvision

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.motip.id_criterion import IDCriterion, giou_elementwise, box_iou_elementwise
from utils.box_ops import box_cxcywh_to_xyxy

K = 6


def make_batch(B=1, G=1, T=3, N=3, seed=0):
    """3 objects per frame with distinct identity labels; object 0 and 1 are close, object 2 is far away."""
    torch.manual_seed(seed)
    gt = torch.tensor([[0.30, 0.50, 0.10, 0.20], [0.34, 0.50, 0.10, 0.20], [0.80, 0.50, 0.10, 0.20]])
    gt = gt[None, None, None].repeat(B, G, T, 1, 1)
    pred = gt + 0.005 * torch.randn_like(gt)                                    # detections ~ GT
    real = torch.tensor([2, 4, 5])[None, None, None, :].repeat(B, G, T, 1)       # identity label of each slot
    unknown = real.clone()                                                      # (no newborn tokens by default)
    present = torch.ones(B, G, T, N, dtype=torch.bool)
    masks = torch.zeros(B, G, T, N, dtype=torch.bool)
    logits = torch.randn(B, G, T, N, K + 1)
    extra = {"pred_boxes": pred, "gt_boxes": gt, "gt_present": present, "real_id_labels": real}
    return logits, unknown, masks, extra


def crit(**kw):
    return IDCriterion(weight=1.0, use_focal_loss=False, **kw)


def test_renormalised_probabilities_sum_to_one_over_C_i():
    logits, _, _, extra = make_batch()
    extra["gt_present"][..., 2] = False                                          # one identity absent from every frame
    p_slot, p_new, z, present = IDCriterion.candidate_probs(logits, extra)
    # absent identity contributes nothing; the renormalised distribution over C_i sums to exactly 1
    assert torch.all(p_slot[..., 2] == 0)
    assert torch.allclose(((p_slot.sum(-1) + p_new) / z), torch.ones_like(z), atol=1e-6)
    assert not present[..., 2].any()
    # ...and it is a *re-normalisation of the full-dictionary softmax*, not a new softmax
    probs = logits.softmax(-1)
    expected_z = probs[..., 2] + probs[..., 4] + probs[..., K]
    assert torch.allclose(z, expected_z, atol=1e-6)


def test_newborn_class_contributes_only_its_own_localisation_error():
    logits, unknown, masks, extra = make_batch()
    logits = torch.full_like(logits, -30.0)
    logits[..., K] = 30.0                                                        # all mass on the newborn class
    c = crit(spatial=True, iou_weight=False)
    c(logits, unknown, masks, extra)
    pred, gt = extra["pred_boxes"][:, :, 1:], extra["gt_boxes"][:, :, 1:]
    own_l1 = (pred - gt).abs().sum(-1).mean().item()
    own_giou = (1 - giou_elementwise(pred, gt)).mean().item()
    t = c.last_terms
    num = masks[:, :, 1:].numel()
    # last_terms are already divided by #ids -> mean over tokens
    assert t["id_l1"] == pytest.approx(own_l1, rel=1e-4)
    assert t["id_giou"] == pytest.approx(own_giou, rel=1e-4)
    assert num == 6


def test_confident_wrong_far_costs_more_than_wrong_near():
    logits, unknown, masks, extra = make_batch()
    def terms(identity_slot):
        lg = torch.full_like(logits, -30.0)
        lg[..., [2, 4, 5][identity_slot]] = 30.0
        c = crit(spatial=True)
        c(lg, unknown, masks, extra)
        return c.last_terms
    # token slot 0 is the nearest to slot 1 (wrong-but-near) and far from slot 2 (wrong-but-far)
    near, far = terms(1), terms(2)
    assert near["id_giou"] < far["id_giou"] and near["id_l1"] < far["id_l1"]
    # cross-entropy alone cannot tell them apart (this is the gap the paper motivates)
    def ce_for(identity_slot):
        lg = torch.full_like(logits, -30.0); lg[..., [2, 4, 5][identity_slot]] = 30.0
        return F.cross_entropy(lg[0, 0, 1, 0:1], torch.tensor([2])).item()
    assert ce_for(1) == pytest.approx(ce_for(2), rel=1e-3)


def test_predicted_box_is_detached_no_gradient_to_detection_head():
    logits, unknown, masks, extra = make_batch()
    extra["pred_boxes"] = extra["pred_boxes"].clone().requires_grad_(True)
    logits = logits.clone().requires_grad_(True)
    c = crit(spatial=True, iou_weight=True)
    c(logits, unknown, masks, extra).backward()
    assert extra["pred_boxes"].grad is None                                      # neither the spatial terms nor the IoU weight
    assert logits.grad is not None and logits.grad.abs().sum() > 0               # ...shape only the ID logits


def test_total_is_weighted_sum_of_the_three_terms_with_paper_lambdas():
    logits, unknown, masks, extra = make_batch(B=2, G=2)
    c = crit(spatial=True, iou_weight=False, lambda_ce=1.0, lambda_giou=2.0, lambda_l1=5.0)
    total = c(logits, unknown, masks, extra).item()
    t = c.last_terms
    assert total == pytest.approx(1.0 * t["id_ce"] + 2.0 * t["id_giou"] + 5.0 * t["id_l1"], rel=1e-5)


def test_spatial_off_equals_legacy_cross_entropy():
    logits, unknown, masks, extra = make_batch(B=2, G=2)
    masks[0, 0, 1, 1] = True                                                     # an occluded (unsupervised) token
    loss = crit(spatial=False, iou_weight=False)(logits, unknown, masks)
    lg, lb, mk = logits[:, :, 1:], unknown[:, :, 1:], masks[:, :, 1:]
    ref = F.cross_entropy(lg[~mk], lb[~mk], reduction="sum") / (~mk).sum()
    assert loss.item() == pytest.approx(ref.item(), rel=1e-6)


def test_iou_weighting_scales_each_sample_by_its_detached_iou():
    logits, unknown, masks, extra = make_batch()
    extra["pred_boxes"] = extra["gt_boxes"].clone()
    extra["pred_boxes"][..., 0, 0] += 0.05                                       # slot 0 detection is shifted -> IoU < 1
    w = box_iou_elementwise(extra["pred_boxes"], extra["gt_boxes"])
    assert (w[..., 0] < 1).all() and torch.allclose(w[..., 1:], torch.ones_like(w[..., 1:]), atol=1e-6)
    loss = crit(spatial=False, iou_weight=True)(logits, unknown, masks, extra)
    ce = F.cross_entropy(logits[:, :, 1:].reshape(-1, K + 1), unknown[:, :, 1:].reshape(-1), reduction="none")
    ref = (ce * w[:, :, 1:].reshape(-1)).sum() / ce.numel()
    assert loss.item() == pytest.approx(ref.item(), rel=1e-5)


def test_box_format_matches_detection_loss_and_torchvision():
    torch.manual_seed(0)
    a, b = torch.rand(10, 4) * 0.4 + 0.2, torch.rand(10, 4) * 0.4 + 0.2          # normalised cxcywh
    ref = torch.diag(torchvision.ops.generalized_box_iou(box_cxcywh_to_xyxy(a), box_cxcywh_to_xyxy(b)))
    assert torch.allclose(giou_elementwise(a, b), ref, atol=1e-5)
    ref_iou = torch.diag(torchvision.ops.box_iou(box_cxcywh_to_xyxy(a), box_cxcywh_to_xyxy(b)))
    assert torch.allclose(box_iou_elementwise(a, b), ref_iou, atol=1e-5)


def test_absent_and_padded_tokens_do_not_produce_nan():
    logits, unknown, masks, extra = make_batch()
    extra["gt_present"][..., 2] = False
    extra["gt_boxes"][..., 2, :] = 0.0                                           # degenerate (absent) GT
    masks[..., 2] = True
    unknown[..., 2] = -1
    extra["real_id_labels"][..., 2] = -1
    loss = crit(spatial=True, iou_weight=True)(logits, unknown, masks, extra)
    assert torch.isfinite(loss)
