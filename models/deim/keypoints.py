"""Per-detection keypoints for the DEIM detector (31 vehicle keypoints in the user's mmdet setup, any K here).

Design = the user's mmdet D-FINE keypoint branch, ported to the MOTIP2 DEIM wrapper:

* every query (detection token `e_i`, the last decoder layer's output) predicts, with two small MLPs,
    - K local coordinates (u, v) = sigmoid(.) in [0, 1] and
    - K visibility logits.
  The absolute (normalised image) position is DECODED INSIDE THE SAME QUERY'S FINAL BOX:
      x = x1 + u (x2 - x1),   y = y1 + v (y2 - y1)
  with the box detached, so keypoint losses cannot move the box branch.  No new matching: a keypoint is supervised
  on the query that the detection Hungarian matching already assigned to the object (`last_indices`).
* losses (weights default to the user's config: 10 / 3 / 35) on matched queries whose GT has labelled keypoints:
    - L1 on the labelled points (normalised coordinates), mean;
    - BCE-with-logits on visibility, mean;
    - OKS loss  1 - OKS  (COCO definition, per-keypoint sigmas) in PIXELS.
  Which GT values count as "labelled" (supervise xy) and as "visible" (visibility target 1) is explicit
  (`labeled_values`, `visible_values`) -- it does not depend on the batch contents.
Layout of the module names (`xy_branch`, `vis_branch`) is what tools/vehicles/convert_mmdet_deim.py maps the mmdet
`kps_xy_branches.0` / `kps_vis_branches.0` onto.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _mlp(in_dim: int, hidden_layers: int, out_dim: int) -> nn.Sequential:
    layers = []
    for _ in range(hidden_layers):
        layers += [nn.Linear(in_dim, in_dim), nn.ReLU()]
    layers.append(nn.Linear(in_dim, out_dim))
    return nn.Sequential(*layers)


def decode_inside_box(local_logits: torch.Tensor, boxes_cxcywh: torch.Tensor) -> torch.Tensor:
    """local_logits (..., K*2), boxes (..., 4) normalised cxcywh -> absolute normalised xy (..., K, 2)."""
    uv = local_logits.sigmoid().reshape(*local_logits.shape[:-1], -1, 2)
    b = boxes_cxcywh.detach()
    cx, cy, w, h = b[..., 0], b[..., 1], b[..., 2].clamp(min=0.0), b[..., 3].clamp(min=0.0)
    x1, y1 = (cx - 0.5 * w).clamp(0.0, 1.0), (cy - 0.5 * h).clamp(0.0, 1.0)
    x2, y2 = (cx + 0.5 * w).clamp(0.0, 1.0), (cy + 0.5 * h).clamp(0.0, 1.0)
    left, right = torch.minimum(x1, x2), torch.maximum(x1, x2)
    top, bottom = torch.minimum(y1, y2), torch.maximum(y1, y2)
    x = left.unsqueeze(-1) + uv[..., 0] * (right - left).unsqueeze(-1)
    y = top.unsqueeze(-1) + uv[..., 1] * (bottom - top).unsqueeze(-1)
    return torch.stack((x, y), dim=-1)


class KeypointHead(nn.Module):
    def __init__(self, hidden_dim: int, num_keypoints: int, num_hidden_layers: int = 4):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.xy_branch = _mlp(hidden_dim, num_hidden_layers, 2 * num_keypoints)
        self.vis_branch = _mlp(hidden_dim, num_hidden_layers, num_keypoints)

    def forward(self, embeds: torch.Tensor, boxes_cxcywh: torch.Tensor):
        """embeds (B,Q,D), boxes (B,Q,4) -> keypoints (B,Q,K,2) in [0,1], visibility logits (B,Q,K)."""
        return decode_inside_box(self.xy_branch(embeds), boxes_cxcywh), self.vis_branch(embeds)


class KeypointLoss(nn.Module):
    def __init__(self, num_keypoints: int, weight_xy: float = 10.0, weight_vis: float = 3.0, weight_oks: float = 35.0,
                 sigmas=None, labeled_values=(1, 2), visible_values=(2,)):
        super().__init__()
        self.K = num_keypoints
        self.w = dict(xy=weight_xy, vis=weight_vis, oks=weight_oks)
        s = torch.full((num_keypoints,), 0.05) if sigmas is None else torch.as_tensor(sigmas, dtype=torch.float32)
        assert s.numel() == num_keypoints, f"{s.numel()} sigmas for {num_keypoints} keypoints"
        self.register_buffer("sigmas", s, persistent=False)
        self.labeled_values = tuple(float(v) for v in labeled_values)
        self.visible_values = tuple(float(v) for v in visible_values)

    @staticmethod
    def _isin(x: torch.Tensor, values) -> torch.Tensor:
        out = torch.zeros_like(x, dtype=torch.bool)
        for v in values:
            out |= x == v
        return out

    def matched(self, outputs, targets, indices):
        """Gather the matched predictions / GT. Returns None when no matched GT has labelled keypoints."""
        px, pv, gx, gv, area, scale = [], [], [], [], [], []
        for b, (src, tgt) in enumerate(indices):
            t = targets[b]
            if "keypoints" not in t or src.numel() == 0:
                continue
            gv_b = t["keypoints_vis"][tgt.to(t["keypoints_vis"].device)]
            ok = self._isin(gv_b, self.labeled_values).any(-1)
            if not ok.any():
                continue
            src, tgt = src.to(outputs["pred_keypoints"].device)[ok], tgt.to(t["keypoints"].device)[ok]
            wh = t.get("size_wh", None)
            wh = torch.ones(2, device=t["keypoints"].device) if wh is None else wh.to(t["keypoints"].device).float()
            px.append(outputs["pred_keypoints"][b, src])
            pv.append(outputs["pred_keypoint_vis"][b, src])
            gx.append(t["keypoints"][tgt].float())
            gv.append(gv_b[ok])
            boxes = t["boxes"][tgt].float()                                   # normalised cxcywh
            area.append(boxes[:, 2] * boxes[:, 3] * wh[0] * wh[1])            # pixels^2
            scale.append(wh.expand(int(ok.sum()), 2))
        if not px:
            return None
        return (torch.cat(px).float(), torch.cat(pv).float(), torch.cat(gx), torch.cat(gv), torch.cat(area),
                torch.cat(scale))

    def oks(self, pred_px, gt_px, labeled, area):
        """COCO OKS per instance: sum_k exp(-d_k^2 / (2 s^2 (2 sigma_k)^2)) [labeled] / sum_k [labeled]."""
        d2 = ((pred_px - gt_px) ** 2).sum(-1)
        var = (2 * self.sigmas.to(d2.device)) ** 2
        e = torch.exp(-d2 / (2 * area.clamp(min=1.0)[:, None] * var[None]))
        lab = labeled.float()
        return (e * lab).sum(-1) / lab.sum(-1).clamp(min=1.0)

    def forward(self, outputs, targets, indices):
        zero = outputs["pred_keypoints"].sum() * 0.0 + outputs["pred_keypoint_vis"].sum() * 0.0
        m = self.matched(outputs, targets, indices)
        if m is None:
            return {"loss_kp_xy": zero, "loss_kp_vis": zero, "loss_kp_oks": zero}
        pred, vis, gt, gv, area, wh = m
        labeled = self._isin(gv, self.labeled_values)
        l_xy = F.l1_loss(pred[labeled], gt[labeled], reduction="mean")
        target_vis = self._isin(gv, self.visible_values).float()
        l_vis = F.binary_cross_entropy_with_logits(vis, target_vis, reduction="mean")
        oks = self.oks(pred * wh[:, None, :], gt * wh[:, None, :], labeled, area)
        l_oks = (1.0 - oks).mean()
        return {"loss_kp_xy": self.w["xy"] * l_xy, "loss_kp_vis": self.w["vis"] * l_vis, "loss_kp_oks": self.w["oks"] * l_oks}


@torch.no_grad()
def keypoint_pixel_error(pred_norm, gt_norm, gv, wh, labeled_values=(1, 2)):
    """Mean pixel error over labelled keypoints (diagnostic / tests)."""
    lab = KeypointLoss._isin(gv, labeled_values)
    d = ((pred_norm - gt_norm) * torch.as_tensor(wh, dtype=pred_norm.dtype)).norm(dim=-1)
    return (d * lab).sum() / lab.sum().clamp(min=1)
