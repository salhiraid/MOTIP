# Copyright (c) Ruopeng Gao. All Rights Reserved.

from .motip import MOTIP
from structures.args import Args
from models.motip.trajectory_modeling import TrajectoryModeling
from models.motip.id_decoder import IDDecoder


def build_legacy_detr(config: dict):
    # Generate DETR args:
    detr_args = Args()
    # 1. backbone:
    detr_args.backbone = config["BACKBONE"]
    detr_args.lr_backbone = config["LR"] * config["LR_BACKBONE_SCALE"]
    detr_args.dilation = config["DILATION"]
    # 2. transformer:
    detr_args.num_classes = config["NUM_CLASSES"]
    detr_args.device = config["DEVICE"]
    detr_args.num_queries = config["DETR_NUM_QUERIES"]
    detr_args.num_feature_levels = config["DETR_NUM_FEATURE_LEVELS"]
    detr_args.aux_loss = config["DETR_AUX_LOSS"]
    detr_args.with_box_refine = config["DETR_WITH_BOX_REFINE"]
    detr_args.two_stage = config["DETR_TWO_STAGE"]
    detr_args.hidden_dim = config["DETR_HIDDEN_DIM"]
    detr_args.masks = config["DETR_MASKS"]
    detr_args.position_embedding = config["DETR_POSITION_EMBEDDING"]
    detr_args.nheads = config["DETR_NUM_HEADS"]
    detr_args.enc_layers = config["DETR_ENC_LAYERS"]
    detr_args.dec_layers = config["DETR_DEC_LAYERS"]
    detr_args.dim_feedforward = config["DETR_DIM_FEEDFORWARD"]
    detr_args.dropout = config["DETR_DROPOUT"]
    detr_args.dec_n_points = config["DETR_DEC_N_POINTS"]
    detr_args.enc_n_points = config["DETR_ENC_N_POINTS"]
    detr_args.cls_loss_coef = config["DETR_CLS_LOSS_COEF"]
    detr_args.bbox_loss_coef = config["DETR_BBOX_LOSS_COEF"]
    detr_args.giou_loss_coef = config["DETR_GIOU_LOSS_COEF"]
    detr_args.focal_alpha = config["DETR_FOCAL_ALPHA"]
    detr_args.set_cost_class = config["DETR_SET_COST_CLASS"]
    detr_args.set_cost_bbox = config["DETR_SET_COST_BBOX"]
    detr_args.set_cost_giou = config["DETR_SET_COST_GIOU"]
    from models.deformable_detr.deformable_detr import build as build_deformable_detr
    detr, detr_criterion, _ = build_deformable_detr(args=detr_args)
    return detr, detr_criterion


def build(config: dict):
    # MOTIP2 flag `DETECTOR: deformable_detr | deim` (falls back to the legacy DETR_FRAMEWORK key).
    detr_framework = str(config.get("DETECTOR", config.get("DETR_FRAMEWORK", "deformable_detr"))).lower()
    match detr_framework:
        case "deformable_detr":
            detr, detr_criterion = build_legacy_detr(config)
        case "deim":
            from models.deim.builder import build_deim, build_detection_criterion
            detr = build_deim(
                size=config["MODEL_SIZE"], num_classes=config["NUM_CLASSES"],
                num_queries=config["DETR_NUM_QUERIES"], num_denoising=config.get("DEIM_NUM_DENOISING", 100),
                anchor_source=config.get("SPATIAL_ANCHOR_SOURCE", "layer_input"),
                num_keypoints=int(config.get("NUM_KEYPOINTS", 0) or 0),
                keypoint_hidden_layers=int(config.get("KEYPOINT_HIDDEN_LAYERS", 4)),
            )
            detr_criterion = build_detection_criterion(
                num_classes=config["NUM_CLASSES"], num_keypoints=int(config.get("NUM_KEYPOINTS", 0) or 0),
                keypoint_loss_weights=tuple(config.get("KEYPOINT_LOSS_WEIGHTS") or (10.0, 3.0, 35.0)),
                keypoint_sigmas=config.get("KEYPOINT_SIGMAS"),
                keypoint_labeled_values=tuple(config.get("KEYPOINT_LABELED_VALUES") or (1, 2)),
                keypoint_visible_values=tuple(config.get("KEYPOINT_VISIBLE_VALUES") or (2,)),
            )
        case _:
            raise NotImplementedError(f"DETR framework {detr_framework} is not supported.")

    # Build each component:
    # 1. trajectory modeling (currently, only FFNs are used; PLAN N4: kept in MOTIP2, flag TRAJECTORY_MODELING):
    _trajectory_modeling = TrajectoryModeling(
        detr_dim=config["DETR_HIDDEN_DIM"],
        ffn_dim_ratio=config["FFN_DIM_RATIO"],
        feature_dim=config["FEATURE_DIM"],
    ) if (config["ONLY_DETR"] is False and config.get("TRAJECTORY_MODELING", True)) else None
    # 2. ID decoder (MOTIP2 flags ID_FUSION / POS_ENC / SPATIAL_ANCHOR; defaults reproduce MOTIP):
    _id_decoder = IDDecoder(
        feature_dim=config["FEATURE_DIM"],
        id_dim=config["ID_DIM"],
        ffn_dim_ratio=config.get("ID_FFN_DIM_RATIO", config["FFN_DIM_RATIO"]),
        num_layers=config["NUM_ID_DECODER_LAYERS"],
        head_dim=config["HEAD_DIM"],
        num_id_vocabulary=config["NUM_ID_VOCABULARY"],
        rel_pe_length=config["REL_PE_LENGTH"],
        use_aux_loss=config["USE_AUX_LOSS"],
        use_shared_aux_head=config["USE_SHARED_AUX_HEAD"],
        id_fusion=config.get("ID_FUSION", "concat"),
        pos_enc=config.get("POS_ENC", "learned_rel"),
        spatial_anchor=bool(config.get("SPATIAL_ANCHOR", False)),
    ) if config["ONLY_DETR"] is False else None

    # Construct MOTIP model:
    motip_model = MOTIP(
        detr=detr,
        detr_framework=detr_framework,
        only_detr=config["ONLY_DETR"],
        trajectory_modeling=_trajectory_modeling,
        id_decoder=_id_decoder,
    )

    return motip_model, detr_criterion
