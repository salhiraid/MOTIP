# MOTIP2 PLAN decision #1: official DEIM HGNetV2 variants N/S/M/L/X, built explicitly (no YAML registry).
# Values transcribed from DEIM configs (configs/base/dfine_hgnetv2.yml, configs/base/deim.yml,
# configs/deim_dfine/dfine_hgnetv2_{n,s,m,l,x}_coco.yml @ 09d35d5).
import copy

from .backbone.hgnetv2 import HGNetv2
from .hybrid_encoder import HybridEncoder
from .dfine_decoder import DFINETransformer
from .matcher import HungarianMatcher
from .deim_criterion import DEIMCriterion

_BASE = dict(
    backbone=dict(name="B4", use_lab=False, return_idx=[1, 2, 3], freeze_stem_only=True, freeze_at=0, freeze_norm=True),
    encoder=dict(in_channels=[512, 1024, 2048], feat_strides=[8, 16, 32], hidden_dim=256, use_encoder_idx=[2],
                 num_encoder_layers=1, nhead=8, dim_feedforward=1024, dropout=0.0, enc_act="gelu",
                 expansion=1.0, depth_mult=1.0, act="silu"),
    decoder=dict(feat_channels=[256, 256, 256], feat_strides=[8, 16, 32], hidden_dim=256, dim_feedforward=1024,
                 num_levels=3, num_layers=6, eval_idx=-1, num_queries=300, num_denoising=100,
                 label_noise_ratio=0.5, box_noise_scale=1.0, reg_max=32, reg_scale=4, layer_scale=1,
                 num_points=[3, 6, 3], cross_attn_method="default", query_select_method="default",
                 activation="silu", mlp_act="silu"),
)

_NSM_BACKBONE = dict(use_lab=True, return_idx=[1, 2, 3], freeze_stem_only=True, freeze_at=-1, freeze_norm=False)

_OVERRIDES = {
    "N": dict(
        backbone=dict(_NSM_BACKBONE, name="B0", return_idx=[2, 3]),
        encoder=dict(in_channels=[512, 1024], feat_strides=[16, 32], hidden_dim=128, use_encoder_idx=[1],
                     dim_feedforward=512, expansion=0.34, depth_mult=0.5),
        decoder=dict(feat_channels=[128, 128], feat_strides=[16, 32], hidden_dim=128, dim_feedforward=512,
                     num_levels=2, num_layers=3, num_points=[6, 6]),
    ),
    "S": dict(
        backbone=dict(_NSM_BACKBONE, name="B0"),
        encoder=dict(in_channels=[256, 512, 1024], hidden_dim=256, depth_mult=0.34, expansion=0.5),
        decoder=dict(num_layers=3),
    ),
    "M": dict(
        backbone=dict(_NSM_BACKBONE, name="B2"),
        encoder=dict(in_channels=[384, 768, 1536], hidden_dim=256, depth_mult=0.67),
        decoder=dict(num_layers=4),
    ),
    "L": dict(backbone=dict(name="B4"), encoder=dict(), decoder=dict()),
    "X": dict(
        backbone=dict(name="B5"),
        encoder=dict(hidden_dim=384, dim_feedforward=2048),
        decoder=dict(feat_channels=[384, 384, 384], reg_scale=8),   # decoder hidden_dim stays 256
    ),
}


def deim_config(size: str) -> dict:
    size = size.upper()
    assert size in _OVERRIDES, f"Unknown MODEL_SIZE {size}"
    cfg = copy.deepcopy(_BASE)
    for part, ov in _OVERRIDES[size].items():
        cfg[part].update(copy.deepcopy(ov))
    return cfg


def deim_hidden_dim(size: str) -> int:
    """Hidden dim of the DEIM *decoder*; the ID decoder mirrors it (Paper §4.1 'Network')."""
    return deim_config(size)["decoder"]["hidden_dim"]


def deim_num_layers(size: str) -> int:
    return deim_config(size)["decoder"]["num_layers"]


def build_deim(size: str, num_classes: int = 1, num_queries: int = 300, num_denoising: int = 100,
               anchor_source: str = "layer_input", pretrained_backbone: bool = False):
    from .wrapper import DEIMDetector
    cfg = deim_config(size)
    cfg["decoder"]["num_queries"] = num_queries
    cfg["decoder"]["num_denoising"] = num_denoising
    # `eval_spatial_size=None`: anchors / positional grids are generated for the actual input size
    # (DEIM defaults to a fixed 640x640 grid; we run 800x1440) -- PLAN §1.2 item 4.
    backbone = HGNetv2(pretrained=pretrained_backbone, **cfg["backbone"])
    encoder = HybridEncoder(eval_spatial_size=None, **cfg["encoder"])
    decoder = DFINETransformer(num_classes=num_classes, eval_spatial_size=None, **cfg["decoder"])
    return DEIMDetector(backbone, encoder, decoder, anchor_source=anchor_source)


def build_detection_criterion(num_classes: int = 1):
    """DEIM losses exactly as Paper App. A: MAL 1.0, L1 5.0, GIoU 2.0, FGL 0.15, DDF 1.5; alpha 0.75, gamma 1.5;
    matcher costs class 2.0 / bbox 5.0 / giou 2.0."""
    from .wrapper import DetectionCriterion
    matcher = HungarianMatcher(weight_dict=dict(cost_class=2, cost_bbox=5, cost_giou=2),
                               use_focal_loss=True, alpha=0.25, gamma=2.0)
    deim = DEIMCriterion(matcher=matcher,
                         weight_dict=dict(loss_mal=1, loss_bbox=5, loss_giou=2, loss_fgl=0.15, loss_ddf=1.5),
                         losses=["mal", "boxes", "local"], alpha=0.75, gamma=1.5,
                         num_classes=num_classes, reg_max=32)
    return DetectionCriterion(deim)
