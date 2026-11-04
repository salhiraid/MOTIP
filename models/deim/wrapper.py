# MOTIP2 PLAN Phase 2: DEIM as the detector of MOTIP.
# Paper Sec. 3.1: "We replace MOTIP's Deformable DETR detector with the real-time detector DEIM".
# Output contract (superset of MOTIP's Deformable-DETR wrapper):
#   pred_logits (B,Q,C), pred_boxes (B,Q,4 cxcywh normalised), outputs = e_i (B,Q,D), pos_embeds = p_i (B,Q,D)
import torch
import torch.nn as nn

from utils.nested_tensor import NestedTensor

_LIGHT_KEYS = ("pred_logits", "pred_boxes", "outputs", "pos_embeds")
_KP_KEYS = ("pred_keypoints", "pred_keypoint_vis")


class DEIMDetector(nn.Module):
    # Attribute names `backbone`, `encoder`, `decoder` match upstream DEIM so COCO checkpoints load by key.
    def __init__(self, backbone, encoder, decoder, anchor_source: str = "layer_input", keypoint_head=None):
        super().__init__()
        assert anchor_source in ("layer_input", "final_box")
        self.backbone = backbone
        self.encoder = encoder
        self.decoder = decoder
        self.anchor_source = anchor_source
        self.hidden_dim = decoder.hidden_dim
        # Optional per-detection keypoints (models/deim/keypoints.py); None = plain DEIM.
        self.keypoint_head = keypoint_head

    def forward(self, samples: NestedTensor, targets=None, lightweight: bool = False):
        """
        Args:
            samples: NestedTensor; DEIM has no padding-mask support, so batches must be padding-free
                     (PLAN N5: all frames of a batch share one scale). The mask is ignored.
            targets: list of {"labels","boxes"} used only for contrastive denoising during training.
            lightweight: skip denoising and auxiliary outputs (frames without detection loss).
        """
        x = samples.tensors if isinstance(samples, NestedTensor) else samples
        x = self.backbone(x)
        x = self.encoder(x)
        num_dn = self.decoder.num_denoising
        if lightweight:
            self.decoder.num_denoising = 0
        try:
            out = self.decoder(x, None if lightweight else targets)
        finally:
            self.decoder.num_denoising = num_dn
        if self.anchor_source == "final_box":
            # N1(b): embed the final box itself with the detector's own positional MLP.
            out["pos_embeds"] = self.decoder.box_to_pos_embed(out["pred_boxes"].detach())
        if self.keypoint_head is not None and not lightweight:
            # keypoints decoded inside the SAME final box of each query (box detached), from the same e_i the ID decoder reads
            out["pred_keypoints"], out["pred_keypoint_vis"] = self.keypoint_head(out["outputs"], out["pred_boxes"])
        if lightweight:
            out = {k: out[k] for k in _LIGHT_KEYS}
        return out


class DetectionCriterion(nn.Module):
    """Wraps DEIMCriterion; returns (loss_dict, final_layer_indices) like MOTIP's SetCriterion.
    DEIMCriterion already applies its weight_dict, so `total()` just sums."""

    def __init__(self, deim_criterion, keypoint_loss=None):
        super().__init__()
        self.deim = deim_criterion
        self.keypoint_loss = keypoint_loss          # models.deim.keypoints.KeypointLoss | None

    def forward(self, outputs, targets):
        losses = self.deim(outputs, targets)
        if self.keypoint_loss is not None and "pred_keypoints" in outputs:
            # supervised on the queries the detection matching selected for the last layer (no second matching)
            losses.update(self.keypoint_loss(outputs, targets, self.deim.last_indices))
        return losses, self.deim.last_indices

    @torch.no_grad()
    def match(self, outputs, targets):
        """Final-layer Hungarian matching only (used for frames without detection loss)."""
        sub = {k: outputs[k] for k in ("pred_logits", "pred_boxes")}
        return self.deim.matcher(sub, targets)["indices"]

    @staticmethod
    def total(loss_dict: dict):
        return sum(loss_dict.values())


def load_deim_pretrain(model: nn.Module, path: str = None, default_class_idx: int = 0, num_classes: int = None,
                       state: dict = None, class_indices=None):
    """Load a DEIM checkpoint (official COCO, a converted custom one, or a MOTIP2 stage-1 one) into `DEIMDetector`
    (or a MOTIP model's `.detr`).

    Class-dependent tensors are `*score_head*` (num_classes x hidden) and `denoising_class_embed.weight`
    (num_classes + 1 padding row). If their size differs from the model's they are re-built by selecting source classes:
      * `class_indices` (list of source class rows, in the order of the target classes), e.g. COCO-80 -> the vehicle
        classes ('bicycle','car','motorcycle','bus','truck') = [1, 2, 3, 5, 7];
      * default: single class `default_class_idx` (COCO person = 0), as before.
    Identical sizes load as they are (stage-1 checkpoints, converted vehicle checkpoints)."""
    if state is None:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if "ema" in ckpt and isinstance(ckpt["ema"], dict) and "module" in ckpt["ema"]:
            state = ckpt["ema"]["module"]
        elif "model" in ckpt:
            state = ckpt["model"]
        else:
            state = ckpt
    model_state = model.state_dict()
    if class_indices is None:
        class_indices = [default_class_idx]
    class_indices = list(class_indices)
    new_state = {}
    for k, v in state.items():
        if k in model_state and v.shape != model_state[k].shape:
            if "score_head" in k:                                    # dec_score_head.*, enc_score_head
                assert len(class_indices) == model_state[k].shape[0], \
                    f"{k}: cannot map {v.shape[0]} source classes to {model_state[k].shape[0]} with indices {class_indices}"
                v = v[class_indices]
            elif k.endswith("denoising_class_embed.weight"):         # (num_classes + 1 padding row, hidden)
                assert len(class_indices) + 1 == model_state[k].shape[0], k
                v = torch.cat([v[class_indices], v[-1:]], dim=0)
        new_state[k] = v
    dropped = [k for k in new_state if k not in model_state]
    # Buffers that only exist upstream when `eval_spatial_size` is fixed (we use dynamic sizes).
    allowed = ("anchors", "valid_mask", "pos_embed")
    bad = [k for k in dropped if not any(a in k for a in allowed)]
    assert not bad, f"Unexpected checkpoint keys: {bad[:5]}"
    for k in dropped:
        new_state.pop(k)
    missing = [k for k in model_state if k not in new_state]
    # a detector without keypoints (COCO / older checkpoints) can initialise a model that has a keypoint head: the head
    # then starts from random weights (it is trained by the keypoint losses, or converted from an mmdet checkpoint)
    kp_missing = [k for k in missing if k.startswith("keypoint_head.")]
    if kp_missing:
        import warnings
        warnings.warn(f"{len(kp_missing)} keypoint_head tensors not in the checkpoint: keypoint head is randomly initialised")
        for k in kp_missing:
            new_state[k] = model_state[k]
        missing = [k for k in missing if k not in kp_missing]
    assert not missing, f"Missing keys in checkpoint: {missing[:5]}"
    model.load_state_dict(new_state, strict=True)
    return dropped
