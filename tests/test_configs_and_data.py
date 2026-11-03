# Gate 8 (unit, CPU): generated configs carry the paper's numbers; configs build; PersonPath22 loader; CLI flags.
import glob
import os
import sys

import pytest
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))

from models.deim.builder import deim_hidden_dim, deim_num_layers
from models.motip import build as build_motip

CFG = os.path.join(os.path.dirname(__file__), "..", "configs", "motip2")


def load(name):
    with open(os.path.join(CFG, name)) as f:
        return yaml.safe_load(f)


# Paper App. A / §4.1: base LR, backbone multiplier, ID-decoder LR x2
LR = {"N": 8e-4, "S": 2e-4, "M": 2e-4, "L": 2.5e-4, "X": 2.5e-4}
BB = {"N": 0.5, "S": 0.5, "M": 0.1, "L": 0.05, "X": 0.01}
BENCH = {"dancetrack": dict(T=30, K=50, det=0.3, newborn=0.7), "sportsmot": dict(T=60, K=50, det=0.3, newborn=0.7),
         "personpath22": dict(T=30, K=300, det=0.4, newborn=0.6)}


@pytest.mark.parametrize("size", "NSMLX")
@pytest.mark.parametrize("bench", list(BENCH))
def test_joint_config_matches_paper(size, bench):
    c, b = load(f"{size.lower()}_{bench}.yaml"), BENCH[bench]
    assert (c["LR"], c["LR_BACKBONE_SCALE"], c["LR_ID_DECODER_SCALE"]) == (LR[size], BB[size], 2.0)
    assert (c["MISS_TOLERANCE"], c["SAMPLE_LENGTHS"], c["NUM_ID_VOCABULARY"]) == (b["T"], [b["T"]], b["K"])
    assert (c["DET_THRESH"], c["NEWBORN_THRESH"]) == (b["det"], b["newborn"])
    # shared hyper-parameters (§1.3): sigma 0.5, N_init 2, lambda_ID 0.1, alpha 2.0, shorter 800 / longer 1440
    assert (c["ID_SWITCH_SIGMA"], c["N_INIT"], c["ID_THRESH"], c["ROPE_ALPHA"]) == (0.5, 2, 0.1, 2.0)
    assert (c["INFERENCE_MAX_SHORTER"], c["INFERENCE_MAX_LONGER"]) == (800, 1440)
    # the six toggles default to MOTIP2
    assert (c["ID_SWITCH"], c["ID_LOSS_SPATIAL"], c["SPATIAL_ANCHOR"], c["POS_ENC"], c["DETECTOR"]) == \
           ("spatial", True, True, "rope", "deim")
    # ID decoder mirrors the DEIM decoder (hidden dim and number of layers)
    assert c["FEATURE_DIM"] == c["ID_DIM"] == c["DETR_HIDDEN_DIM"] == deim_hidden_dim(size)
    assert c["NUM_ID_DECODER_LAYERS"] == deim_num_layers(size)
    # losses / optimiser / schedule (App. A)
    assert (c["ID_LOSS_WEIGHT"], c["ID_LAMBDA_CE"], c["ID_LAMBDA_GIOU"], c["ID_LAMBDA_L1"]) == (1.0, 1.0, 2.0, 5.0)
    assert (c["MAX_CLIP_NORM"], c["EMA_DECAY"], c["EMA_WARMUPS"], c["LR_WARMUP_ITERS"]) == (0.1, 0.9995, 1000, 500)
    assert c["LR_SCHEDULER"] == "constant" and c["SAMPLES_PER_EPOCH"] == 40000 and c["SAMPLE_INTERVALS"] == [4]
    assert c["DETR_NUM_TRAIN_FRAMES"] == (8 if size == "N" else 4)
    assert c["BATCH_SIZE"] == (4 if size == "N" else 2)                      # clips per GPU
    assert c["AUG_NUM_GROUPS"] == 6 and c["AUG_TRAJECTORY_OCCLUSION_PROB"] == 0.5
    assert c["IMAGE_NORMALIZE"] is False and c["ID_FFN_DIM_RATIO"] == 4
    assert c["WEIGHT_DECAY"] == (1.25e-4 if size in "LX" else 1e-4)


@pytest.mark.parametrize("size", "NSMLX")
def test_pretrain_config_stage1(size):
    c = load(f"pretrain_{size.lower()}_dancetrack.yaml")
    assert c["ONLY_DETR"] is True and c["SAMPLE_LENGTHS"] == [1] and c["LR_WARMUP_ITERS"] == 0 and c["EMA_WARMUPS"] == 0
    assert c["BATCH_SIZE"] == (32 if size == "N" else 8)                     # images per GPU


@pytest.mark.parametrize("size", ["N", "S"])
def test_config_builds_the_model_and_toggles_reach_the_modules(size):
    c = load(f"{size.lower()}_dancetrack.yaml")
    model, crit = build_motip(c)
    dec = model.id_decoder
    assert (dec.id_fusion, dec.pos_enc, dec.spatial_anchor) == ("add", "rope", True)
    assert dec.model_dim == deim_hidden_dim(size) and dec.num_layers == deim_num_layers(size)
    for flag, check in [("POS_ENC", "learned_rel"), ("SPATIAL_ANCHOR", False), ("ID_FUSION", "concat")]:
        c2 = dict(c); c2[flag] = check
        if flag == "ID_FUSION":
            c2["SPATIAL_ANCHOR"] = False; c2["POS_ENC"] = "learned_rel"
            c2["ID_DIM"] = c["FEATURE_DIM"]
        m2, _ = build_motip(c2)
        assert {"POS_ENC": m2.id_decoder.pos_enc, "SPATIAL_ANCHOR": m2.id_decoder.spatial_anchor,
                "ID_FUSION": m2.id_decoder.id_fusion}[flag] == check


def test_every_motip2_cli_flag_has_a_config_key():
    src = open(os.path.join(os.path.dirname(__file__), "..", "runtime_option.py")).read()
    flags = {l.split('"--')[1].split('"')[0] for l in src.splitlines() if 'add_argument("--' in l}
    cfg = load("s_dancetrack.yaml")
    motip2_flags = ["id-switch", "id-loss-spatial", "spatial-anchor", "pos-enc", "n-init", "detector", "id-fusion",
                    "model-size", "id-switch-sigma", "id-switch-reference", "id-switch-subset-prob", "id-switch-absent",
                    "id-lambda-ce", "id-lambda-giou", "id-lambda-l1", "id-ce-iou-weight", "spatial-anchor-source",
                    "spatial-anchor-detach", "rope-alpha", "rope-alpha-mode", "trajectory-modeling", "samples-per-epoch",
                    "lr-warmup-iters", "use-ema", "ema-decay", "ema-warmups", "inference-max-shorter", "inference-only-detr"]
    for f in motip2_flags:
        assert f in flags, f"--{f} missing in runtime_option.py"
        assert f.replace("-", "_").upper() in cfg, f"{f} missing in generated configs"


def test_personpath22_loader_and_registry(tmp_path):
    from make_synthetic_dancetrack import make_sequence
    import numpy as np
    from data.joint_dataset import dataset_classes, JointDataset
    assert {"PersonPath22", "PersonPath22_Inference"} <= set(dataset_classes)
    rng = np.random.default_rng(0)
    seq = tmp_path / "PersonPath22" / "train" / "vid0001"
    make_sequence(str(seq), 12, 96, 160, 3, rng)
    with open(seq / "gt" / "gt.txt", "a") as f:
        f.write("1,99,5,5,10,10,0,1,1\n")                    # conf == 0 -> ignore row
    ds = dataset_classes["PersonPath22"](data_root=str(tmp_path), split="train")
    ann = ds.get_annotations()["vid0001"]
    assert ds.get_sequence_infos()["vid0001"]["length"] == 12
    assert len(ann[0]["id"]) == 3 and 99 not in ann[0]["id"].tolist() and ann[0]["is_legal"]
    jd = JointDataset(data_root=str(tmp_path), datasets=["PersonPath22"], splits=["train"])
    assert jd.statistics()[0].startswith("PersonPath22.train, 1 sequences, 12 frames")
    # inference view
    inf = tmp_path / "PersonPath22_Inference" / "test" / "vid0002"
    make_sequence(str(inf), 12, 96, 160, 2, rng)
    di = dataset_classes["PersonPath22_Inference"](data_root=str(tmp_path), split="test", load_annotation=False)
    assert list(di.get_sequence_infos()) == ["vid0002"] and len(di.get_image_paths()["vid0002"]) == 12
