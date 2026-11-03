# MOTIP2 Sec. 3.2 "Spatial ID Switches" (data stage, training only).
#
# MOTIP perturbs the training context by selecting a random subset S of the objects of a frame and permuting their
# identities uniformly. MOTIP2 keeps the selection and changes only how the identities are reassigned:
#   G_t  = pairwise GIoU of the boxes of S in the frame                      (G_ii = 1)
#   A_t  = G_t + E,   E_ij ~ i.i.d. N(0, sigma^2)                            (noise on every cell, diagonal included)
#   pi   = argmax_pi sum_i A_t[i, pi(i)]                                     (Hungarian, maximisation)
# sigma -> infinity recovers the uniform permutation; sigma = 0.5 is shared across benchmarks.
import torch
import numpy as np
from scipy.optimize import linear_sum_assignment

from utils.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


def pairwise_giou(boxes_cxcywh: torch.Tensor) -> torch.Tensor:
    """GIoU matrix of normalised cxcywh boxes with G_ii = 1 (and NaN -> -1 for degenerate boxes)."""
    xyxy = box_cxcywh_to_xyxy(boxes_cxcywh.to(torch.float32))
    g = generalized_box_iou(xyxy, xyxy)
    g = torch.nan_to_num(g, nan=-1.0)
    g.fill_diagonal_(1.0)
    return g


def spatial_permutation(giou: torch.Tensor, sigma: float) -> torch.Tensor:
    """pi for one frame. Returns `perm` with perm[i] = j: identity i takes the object currently labelled j."""
    # MOTIP2 Sec. 3.2: A = G + E, E_ij ~ N(0, sigma^2); Hungarian algorithm maximising the assignment.
    affinity = giou.to(torch.float64) + sigma * torch.randn(giou.shape, dtype=torch.float64)
    rows, cols = linear_sum_assignment(affinity.numpy(), maximize=True)
    assert (rows == np.arange(len(rows))).all()
    return torch.as_tensor(cols, dtype=torch.int64)


class SwitchStats:
    """Tab. 8 statistics: Swap% (objects whose identity changed / objects present), LR% (fraction of swapped pairs that
    are long-range, GIoU <= 0), mean GIoU of swapped pairs."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.n_obj = self.n_swapped = self.n_long_range = 0
        self.giou_sum = 0.0

    def add_frame(self, n_present: int):
        self.n_obj += int(n_present)

    def add_swaps(self, giou_of_swapped_pairs: torch.Tensor):
        g = giou_of_swapped_pairs.reshape(-1)
        self.n_swapped += len(g)
        self.n_long_range += int((g <= 0).sum())
        self.giou_sum += float(g.sum())

    def summary(self) -> dict:
        return {
            "Swap%": 100.0 * self.n_swapped / max(1, self.n_obj),
            "LR%": 100.0 * self.n_long_range / max(1, self.n_swapped),
            "meanGIoU": self.giou_sum / max(1, self.n_swapped),
            "n_obj": self.n_obj, "n_swapped": self.n_swapped,
        }


def apply_id_switch(
        ann_idxs: torch.Tensor,         # (G, T, N) annotation index each identity slot points to (-1 = none)
        id_masks: torch.Tensor,         # (G, T, N) True = slot absent / masked
        gt_boxes,                       # list over T of (n_obj_t, 4) normalised cxcywh GT boxes
        mode: str = "spatial",          # none | uniform | spatial
        sigma: float = 0.5,
        reference: str = "per_frame",   # per_frame | last_frame (PLAN decision #6)
        subset_prob: float = 0.5,       # MOTIP's per-object inclusion probability (PLAN decision #7)
        absent: str = "skip",           # skip | neg_one (PLAN N2); only used by spatial
        stats: SwitchStats = None,
):
    """Return switched (ann_idxs, id_masks). Switching ann_idxs / masks of a slot is how MOTIP permutes identities:
    slot n keeps its identity label but now carries the appearance (and box) of another object."""
    if mode == "none":
        return ann_idxs, id_masks
    G, T, N = ann_idxs.shape
    ann_idxs, id_masks = ann_idxs.clone(), id_masks.clone()

    if mode == "uniform":
        # MOTIP's original augmentation, unchanged (per (group, frame), uniform permutation of the subset).
        a = ann_idxs.reshape(G * T, N)
        m = id_masks.reshape(G * T, N)
        for g_t in range(G * T):
            switch_map = torch.bernoulli(torch.ones((N,)) * subset_prob)
            switch_idxs = torch.nonzero(switch_map)[:, 0]
            present = ~m[g_t] & (a[g_t] >= 0)
            if stats is not None:
                stats.add_frame(present.sum())
            if len(switch_idxs) > 1:
                shuffled = switch_idxs[torch.randperm(len(switch_idxs))]
                if stats is not None:
                    _record_swaps(stats, a[g_t], m[g_t], switch_idxs, shuffled, gt_boxes[g_t % T])
                a[g_t, switch_idxs] = a[g_t, shuffled]
                m[g_t, switch_idxs] = m[g_t, shuffled]
        return a.reshape(G, T, N), m.reshape(G, T, N)

    assert mode == "spatial", mode
    for g in range(G):
        if reference == "last_frame":
            t_ref = T - 1
            S = _select_subset(ann_idxs[g, t_ref], id_masks[g, t_ref], subset_prob, absent)
            if stats is not None:
                stats.add_frame((~id_masks[g, t_ref] & (ann_idxs[g, t_ref] >= 0)).sum())
            if len(S) < 2:
                continue
            perm = spatial_permutation(_frame_giou(gt_boxes[t_ref], ann_idxs[g, t_ref], id_masks[g, t_ref], S), sigma)
            if stats is not None:
                _record_swaps(stats, ann_idxs[g, t_ref], id_masks[g, t_ref], S, S[perm], gt_boxes[t_ref])
            for t in range(T):      # apply pi to every occurrence of the permuted identities in the context
                ann_idxs[g, t, S] = ann_idxs[g, t, S[perm]].clone()
                id_masks[g, t, S] = id_masks[g, t, S[perm]].clone()
        else:
            for t in range(T):
                S = _select_subset(ann_idxs[g, t], id_masks[g, t], subset_prob, absent)
                if stats is not None:
                    stats.add_frame((~id_masks[g, t] & (ann_idxs[g, t] >= 0)).sum())
                if len(S) < 2:
                    continue
                perm = spatial_permutation(_frame_giou(gt_boxes[t], ann_idxs[g, t], id_masks[g, t], S), sigma)
                if stats is not None:
                    _record_swaps(stats, ann_idxs[g, t], id_masks[g, t], S, S[perm], gt_boxes[t])
                ann_idxs[g, t, S] = ann_idxs[g, t, S[perm]].clone()
                id_masks[g, t, S] = id_masks[g, t, S[perm]].clone()
    return ann_idxs, id_masks


def _select_subset(ann_idxs_t, id_masks_t, subset_prob: float, absent: str) -> torch.Tensor:
    """S ~ Bernoulli(subset_prob) per object (MOTIP). PLAN N2: slots without a box in this frame are skipped
    (`skip`) because GIoU is undefined for them, or kept (`neg_one`, treated as GIoU = -1 with everything)."""
    N = len(ann_idxs_t)
    chosen = torch.bernoulli(torch.ones((N,)) * subset_prob).bool()
    if absent == "skip":
        chosen &= (~id_masks_t) & (ann_idxs_t >= 0)
    return torch.nonzero(chosen)[:, 0]


def _frame_giou(boxes_t, ann_idxs_t, id_masks_t, S) -> torch.Tensor:
    present = (~id_masks_t[S]) & (ann_idxs_t[S] >= 0)
    g = torch.full((len(S), len(S)), -1.0)                  # absent slots: GIoU = -1 with everything (neg_one mode)
    if present.any():
        sub = pairwise_giou(boxes_t[ann_idxs_t[S][present]])
        pi = torch.nonzero(present)[:, 0]
        g[pi[:, None], pi[None, :]] = sub
    g.fill_diagonal_(1.0)
    return g


def _record_swaps(stats: SwitchStats, ann_idxs_t, id_masks_t, S, S_new, boxes_t):
    """Stats over slot pairs (i takes the object of slot j != i) where both objects are present in this frame."""
    moved = S != S_new
    if not moved.any():
        return
    i, j = S[moved], S_new[moved]
    ok = (~id_masks_t[i]) & (~id_masks_t[j]) & (ann_idxs_t[i] >= 0) & (ann_idxs_t[j] >= 0)
    if not ok.any():
        return
    bi, bj = boxes_t[ann_idxs_t[i][ok]], boxes_t[ann_idxs_t[j][ok]]
    xi, xj = box_cxcywh_to_xyxy(bi.float()), box_cxcywh_to_xyxy(bj.float())
    g = torch.diag(torch.nan_to_num(generalized_box_iou(xi, xj), nan=-1.0))
    stats.add_swaps(g)
