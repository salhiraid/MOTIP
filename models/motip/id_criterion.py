# Copyright (c) Ruopeng Gao. All Rights Reserved.

import torch
import einops
import torch.nn as nn
import torch.nn.functional as F

from utils.misc import is_distributed, distributed_world_size, labels_to_one_hot
from utils.box_ops import box_cxcywh_to_xyxy


class IDCriterion(nn.Module):
    """ID loss. MOTIP: cross-entropy. MOTIP2 (Paper Sec. 3.3, App. A):

        L_id = w * ( lambda_CE * L_CE + lambda_GIoU * L_id-giou + lambda_L1 * L_id-l1 )

    * L_CE      cross-entropy over the full ID softmax, weighted per sample by IoU(b_hat, b_gt) (detached).
    * L_id-l1   sum_c p_c * || b_hat - b_c^gt ||_1
    * L_id-giou sum_c p_c * ( 1 - GIoU(b_hat, b_c^gt) )
      where c runs over the identities present in the current frame plus the newborn class, p_c is the ID softmax
      renormalised over that set, b_c^gt is the GT box of the object holding identity c (for the newborn class: the
      detection's own GT box) and b_hat is the predicted box of the detection, DETACHED (the terms shape only the ID
      logits; the detection head is untouched).
    The overall weight w (ID_LOSS_WEIGHT) is applied by the training loop.
    """

    def __init__(
            self,
            weight: float,
            use_focal_loss: bool,
            spatial: bool = False,
            iou_weight: bool = False,
            lambda_ce: float = 1.0,
            lambda_giou: float = 2.0,
            lambda_l1: float = 5.0,
    ):
        super().__init__()
        self.weight = weight
        self.use_focal_loss = use_focal_loss
        self.spatial = spatial
        self.iou_weight = iou_weight
        self.lambda_ce, self.lambda_giou, self.lambda_l1 = lambda_ce, lambda_giou, lambda_l1
        self.last_terms = {}

        if not self.use_focal_loss:
            self.ce_loss = nn.CrossEntropyLoss(reduction="none")
        return

    def forward(self, id_logits, id_labels, id_masks, extra: dict | None = None):
        # _B, _G, _T, _N = id_logits.shape
        # Remove the first T for supervision:
        id_logits = id_logits[:, :, 1:, :, :]
        id_labels = id_labels[:, :, 1:, :]
        id_masks = id_masks[:, :, 1:, :]
        pass

        need_boxes = (self.spatial or self.iou_weight) and not self.use_focal_loss
        if need_boxes:
            assert extra is not None, "ID_LOSS_SPATIAL / ID_CE_IOU_WEIGHT need pred/GT boxes (pass `extra`)"
            extra = {k: self._tile_like(v, id_logits)[:, :, 1:] for k, v in extra.items()}

        # Flatten:
        id_logits_flatten = einops.rearrange(id_logits, "b g t n c -> (b g t n) c")
        id_labels_flatten = einops.rearrange(id_labels, "b g t n -> (b g t n)")
        id_masks_flatten = einops.rearrange(id_masks, "b g t n -> (b g t n)")
        # Filter out the invalid id labels:
        id_logits_flatten = id_logits_flatten[~id_masks_flatten]
        id_labels_flatten = id_labels_flatten[~id_masks_flatten]
        # Calculate the loss:
        if self.use_focal_loss:
            id_labels_flatten_one_hot = labels_to_one_hot(id_labels_flatten, class_num=id_logits_flatten.shape[-1])
            id_labels_flatten_one_hot = torch.tensor(id_labels_flatten_one_hot, device=id_logits.device)
            loss = sigmoid_focal_loss(inputs=id_logits_flatten, targets=id_labels_flatten_one_hot).sum()
            num_ids = torch.as_tensor([len(id_logits_flatten)], dtype=torch.float, device=id_logits.device)
            if is_distributed():
                torch.distributed.all_reduce(num_ids)
            num_ids = torch.clamp(num_ids / distributed_world_size(), min=1).item()
            return loss / num_ids

        ce = self.ce_loss(id_logits_flatten, id_labels_flatten)                          # (M,)
        if need_boxes:
            valid = ~einops.rearrange(id_masks, "b g t n -> (b g t n)")
            pred = einops.rearrange(extra["pred_boxes"], "b g t n d -> (b g t n) d")[valid].detach()
            own_gt = einops.rearrange(extra["gt_boxes"], "b g t n d -> (b g t n) d")[valid]
            if self.iou_weight:
                # App. A: "the ID cross-entropy is weighted per-sample by the predicted-to-ground-truth IoU"
                ce = ce * box_iou_elementwise(pred, own_gt).detach()
        loss_ce = ce.sum()
        loss_l1 = loss_giou = torch.zeros((), device=id_logits.device)
        if self.spatial:
            l1, giou_term = self._spatial_terms(id_logits, id_masks, extra)             # each (B,G,T',N)
            valid_full = ~id_masks
            loss_l1, loss_giou = l1[valid_full].sum(), giou_term[valid_full].sum()
        num_ids = torch.as_tensor([len(id_logits_flatten)], dtype=torch.float, device=id_logits.device)

        if is_distributed():
            torch.distributed.all_reduce(num_ids)
        num_ids = torch.clamp(num_ids / distributed_world_size(), min=1).item()

        total = (self.lambda_ce * loss_ce + self.lambda_giou * loss_giou + self.lambda_l1 * loss_l1) / num_ids
        self.last_terms = {"id_ce": (loss_ce / num_ids).item(),
                           "id_giou": (loss_giou / num_ids).item(), "id_l1": (loss_l1 / num_ids).item()}
        return total

    @staticmethod
    def _tile_like(v, logits):
        """Auxiliary decoder layers stack their logits along the batch dimension; tile per-token extras to match."""
        if v.shape[0] != logits.shape[0]:
            v = v.repeat(logits.shape[0] // v.shape[0], *([1] * (v.dim() - 1)))
        return v

    @staticmethod
    def candidate_probs(id_logits, extra):
        """p_ic renormalised over C_i = {identities present in the frame} + {newborn}. Returns
        (p_slot (...,N,N) masked probabilities of the identity held by slot m for token i, p_newborn (...,N), Z)."""
        K = id_logits.shape[-1] - 1
        probs = id_logits.softmax(dim=-1)                                              # (B,G,T,N,K+1)
        real = extra["real_id_labels"]                                                 # (B,G,T,N)
        present = extra["gt_present"] & (real >= 0)                                    # identities present in the frame
        N = real.shape[-1]
        idx = real.clamp(min=0)[..., None, :].expand(*real.shape[:-1], N, N)           # [..., i, m] = label of slot m
        p_slot = probs.gather(-1, idx) * present[..., None, :]
        p_new = probs[..., K]
        z = p_slot.sum(-1) + p_new
        return p_slot, p_new, z.clamp(min=1e-12), present

    def _spatial_terms(self, id_logits, id_masks, extra):
        p_slot, p_new, z, present = self.candidate_probs(id_logits, extra)
        pred = extra["pred_boxes"].detach()                                            # b_hat: stop-gradient (Sec. 3.3)
        gt = extra["gt_boxes"]                                                         # (B,G,T,N,4) GT of slot m
        # normalised cxcywh, the same format as the detection loss (PLAN decision #10)
        l1 = (pred[..., :, None, :] - gt[..., None, :, :]).abs().sum(-1)               # [..., i, m]
        giou_cost = 1.0 - giou_elementwise(pred[..., :, None, :], gt[..., None, :, :])
        # newborn class: no associated object -> the detection's own GT box (diagonal)
        l1_new, giou_new = l1.diagonal(dim1=-2, dim2=-1), giou_cost.diagonal(dim1=-2, dim2=-1)
        l_l1 = ((p_slot * l1).sum(-1) + p_new * l1_new) / z
        l_giou = ((p_slot * giou_cost).sum(-1) + p_new * giou_new) / z
        return l_l1, l_giou


def box_iou_elementwise(a, b, eps: float = 1e-7):
    """IoU of matching rows of normalised cxcywh boxes a, b (..., 4)."""
    a, b = box_cxcywh_to_xyxy(a), box_cxcywh_to_xyxy(b)
    inter = (torch.min(a[..., 2:], b[..., 2:]) - torch.max(a[..., :2], b[..., :2])).clamp(min=0).prod(-1)
    union = (a[..., 2:] - a[..., :2]).clamp(min=0).prod(-1) + (b[..., 2:] - b[..., :2]).clamp(min=0).prod(-1) - inter
    return inter / union.clamp(min=eps)


def giou_elementwise(a, b, eps: float = 1e-7):
    """Generalised IoU (Rezatofighi et al.) of broadcastable normalised cxcywh boxes (..., 4)."""
    a, b = box_cxcywh_to_xyxy(a), box_cxcywh_to_xyxy(b)
    inter = (torch.min(a[..., 2:], b[..., 2:]) - torch.max(a[..., :2], b[..., :2])).clamp(min=0).prod(-1)
    union = (a[..., 2:] - a[..., :2]).clamp(min=0).prod(-1) + (b[..., 2:] - b[..., :2]).clamp(min=0).prod(-1) - inter
    iou = inter / union.clamp(min=eps)
    hull = (torch.max(a[..., 2:], b[..., 2:]) - torch.min(a[..., :2], b[..., :2])).clamp(min=0).prod(-1)
    return iou - (hull - union) / hull.clamp(min=eps)


def sigmoid_focal_loss(inputs, targets, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum()


def build(config: dict):
    return IDCriterion(
        weight=config["ID_LOSS_WEIGHT"],
        use_focal_loss=config["USE_FOCAL_LOSS"],
        spatial=bool(config.get("ID_LOSS_SPATIAL", False)),
        iou_weight=bool(config.get("ID_CE_IOU_WEIGHT", False)),
        lambda_ce=config.get("ID_LAMBDA_CE", 1.0),
        lambda_giou=config.get("ID_LAMBDA_GIOU", 2.0),
        lambda_l1=config.get("ID_LAMBDA_L1", 5.0),
    )