# MOTIP2 PLAN Phase 2: DEIM as the detector of MOTIP.
# Paper Sec. 3.1: "We replace MOTIP's Deformable DETR detector with the real-time detector DEIM".
# Output contract (superset of MOTIP's Deformable-DETR wrapper):
#   pred_logits (B,Q,C), pred_boxes (B,Q,4 cxcywh normalised), outputs = e_i (B,Q,D), pos_embeds = p_i (B,Q,D)
import torch
import torch.nn as nn

from utils.nested_tensor import NestedTensor

_LIGHT_KEYS = ("pred_logits", "pred_boxes", "outputs", "pos_embeds")


class DEIMDetector(nn.Module):
    # Attribute names `backbone`, `encoder`, `decoder` match upstream DEIM so COCO checkpoints load by key.
    def __init__(self, backbone, encoder, decoder, anchor_source: str = "layer_input"):
        super().__init__()
        assert anchor_source in ("layer_input", "final_box")
        self.backbone = backbone
        self.encoder = encoder
        self.decoder = decoder
        self.anchor_source = anchor_source
        self.hidden_dim = decoder.hidden_dim

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
        if lightweight:
            out = {k: out[k] for k in _LIGHT_KEYS}
        return out


class DetectionCriterion(nn.Module):
    """Wraps DEIMCriterion; returns (loss_dict, final_layer_indices) like MOTIP's SetCriterion.
    DEIMCriterion already applies its weight_dict, so `total()` just sums."""

    def __init__(self, deim_criterion):
        super().__init__()
        self.deim = deim_criterion

    def forward(self, outputs, targets):
        losses = self.deim(outputs, targets)
        return losses, self.deim.last_indices

    @torch.no_grad()
    def match(self, outputs, targets):
        """Final-layer Hungarian matching only (used for frames without detection loss)."""
        sub = {k: outputs[k] for k in ("pred_logits", "pred_boxes")}
        return self.deim.matcher(sub, targets)["indices"]

    @staticmethod
    def total(loss_dict: dict):
        return sum(loss_dict.values())


def load_deim_pretrain(model: nn.Module, path: str = None, default_class_idx: int = 0, num_classes: int = 1,
                       state: dict = None):
    """Load an official DEIM COCO checkpoint into `DEIMDetector` (or a MOTIP model's `.detr`).
    COCO head has 80 remapped classes (person = 0, `remap_mscoco_category: True`); slice to `num_classes`=1."""
    if state is None:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if "ema" in ckpt and isinstance(ckpt["ema"], dict) and "module" in ckpt["ema"]:
            state = ckpt["ema"]["module"]
        elif "model" in ckpt:
            state = ckpt["model"]
        else:
            state = ckpt
    assert num_classes == 1, "only the single-class (person) head is supported"
    model_state = model.state_dict()
    new_state = {}
    for k, v in state.items():
        # Slice the 80-class COCO head to the single person class only when shapes differ, so that stage-1
        # (already single-class) checkpoints load through the same function.
        if k in model_state and v.shape != model_state[k].shape:
            if "score_head" in k:                                    # dec_score_head.*, enc_score_head
                v = v[default_class_idx:default_class_idx + 1]
            elif k.endswith("denoising_class_embed.weight"):         # (num_classes + 1 padding row, hidden)
                v = torch.cat([v[default_class_idx:default_class_idx + 1], v[-1:]], dim=0)
        new_state[k] = v
    dropped = [k for k in new_state if k not in model_state]
    # Buffers that only exist upstream when `eval_spatial_size` is fixed (we use dynamic sizes).
    allowed = ("anchors", "valid_mask", "pos_embed")
    bad = [k for k in dropped if not any(a in k for a in allowed)]
    assert not bad, f"Unexpected checkpoint keys: {bad[:5]}"
    for k in dropped:
        new_state.pop(k)
    missing = [k for k in model_state if k not in new_state]
    assert not missing, f"Missing keys in checkpoint: {missing[:5]}"
    model.load_state_dict(new_state, strict=True)
    return dropped
