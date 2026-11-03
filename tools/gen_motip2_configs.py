# MOTIP2 PLAN Phase 8: generate configs/motip2/*.yaml from the paper's numbers (skill §1.2-§1.4, §10).
# MOTIP's config chaining is linear (SUPER_CONFIG_PATH), so SIZE x BENCH x STAGE combinations are composed here
# and written as flat YAML files. Re-run after changing anything below:  python tools/gen_motip2_configs.py
import os

import yaml

OUT = os.path.join(os.path.dirname(__file__), "..", "configs", "motip2")

# ---- Paper §3 / §4.1 / App. A (shared across benchmarks unless noted) ----------------------------------------------
BASE = {
    "SUPER_CONFIG_PATH": None,
    "SEED": 42, "NUM_WORKERS": 6, "PREFETCH_FACTOR": 6, "DEVICE": "cuda", "DATA_ROOT": "./datasets/",
    # --- the six ablation toggles (skill §0.2) ---
    "ID_SWITCH": "spatial",            # none | uniform | spatial
    "ID_LOSS_SPATIAL": True,
    "SPATIAL_ANCHOR": True,
    "POS_ENC": "rope",                 # learned_rel | rope
    "N_INIT": 2,
    "DETECTOR": "deim",                # deformable_detr | deim
    # --- additional switches (PLAN §3.2) ---
    "ID_FUSION": "add",                # add | concat
    "ID_SWITCH_SIGMA": 0.5,            # sigma (Paper §4.1)
    "ID_SWITCH_REFERENCE": "per_frame",   # per_frame | last_frame (PLAN decision #6)
    "ID_SWITCH_SUBSET_PROB": 0.5,      # MOTIP's per-object inclusion probability (PLAN decision #7)
    "ID_SWITCH_ABSENT": "skip",        # skip | neg_one (PLAN N2)
    "ID_LOSS_WEIGHT": 1.0,             # overall ID loss weight (App. A)
    "ID_LAMBDA_CE": 1.0, "ID_LAMBDA_GIOU": 2.0, "ID_LAMBDA_L1": 5.0,   # Sec. 4.1
    "ID_CE_IOU_WEIGHT": True,          # App. A: CE weighted per sample by predicted-to-GT IoU
    "SPATIAL_ANCHOR_SOURCE": "layer_input",   # layer_input | final_box (PLAN N1)
    "SPATIAL_ANCHOR_DETACH": False,    # PLAN N8
    "ROPE_ALPHA": 2.0,                 # alpha (Paper §4.1)
    "ROPE_ALPHA_MODE": "multiply",     # multiply | divide (PLAN decision #8)
    "TRAJECTORY_MODELING": True,       # PLAN N4
    "N_INIT_BACKFILL": False,          # PLAN N6
    # --- model ---
    "ONLY_DETR": False, "NUM_CLASSES": 1, "DETR_NUM_QUERIES": 300, "DEIM_NUM_DENOISING": 100,
    "HEAD_DIM": 32, "ID_FFN_DIM_RATIO": 4, "FFN_DIM_RATIO": 4,
    "USE_AUX_LOSS": False, "USE_SHARED_AUX_HEAD": True, "USE_FOCAL_LOSS": False,
    # --- data / augmentation (App. A) ---
    "SAMPLE_STEPS": [0], "SAMPLE_INTERVALS": [4], "LENGTH_PER_ITERATION": None, "SAMPLES_PER_EPOCH": 40000,
    "AUG_PIPELINE": "motip2", "AUG_BASE_SIZE": 800, "AUG_MAX_SIZE": 1440, "AUG_MULTISCALE": True,
    "AUG_MAX_SHIFT_RATIO": 0.06, "AUG_OVERFLOW_BBOX": False,
    "AUG_NUM_GROUPS": 6, "AUG_TRAJECTORY_OCCLUSION_PROB": 0.5, "AUG_TRAJECTORY_SWITCH_PROB": 0.5,
    "IMAGE_NORMALIZE": False,          # DEIM expects [0,1] inputs
    "SIZE_DIVISIBILITY": 32,
    # --- optimisation (App. A) ---
    "LR_SCHEDULER": "constant", "LR_ID_DECODER_SCALE": 2.0, "LR_WARMUP_EPOCHS": 0,
    "SCHEDULER_MILESTONES": [6, 9], "SCHEDULER_GAMMA": 0.1,
    "MAX_CLIP_NORM": 0.1, "SEPARATE_CLIP_NORM": True, "USE_ACCELERATE_CLIP_NORM": True,
    "USE_EMA": True, "EMA_DECAY": 0.9995,
    "DETR_NUM_CHECKPOINT_FRAMES": 5, "DETR_CRITERION_BATCH_LEN": 10, "USE_DECODER_CHECKPOINT": False,
    "ACCUMULATE_STEPS": 1,
    "RESUME_MODEL": None, "RESUME_OPTIMIZER": True, "RESUME_SCHEDULER": True, "RESUME_STATES": True,
    # --- inference (§1.3, §1.5) ---
    "INFERENCE_MODEL": None, "INFERENCE_MODE": "evaluate", "INFERENCE_GROUP": "default",
    "INFERENCE_MAX_SHORTER": 800, "INFERENCE_MAX_LONGER": 1440, "INFERENCE_DTYPE": "FP32",
    "ASSIGNMENT_PROTOCOL": "object-max",   # MOTIP's greedy readout
    "ID_THRESH": 0.1,                      # lambda_ID
    "AREA_THRESH": 0, "INFERENCE_ONLY_DETR": None,
    # --- outputs ---
    "OUTPUTS_DIR": None, "USE_WANDB": False, "EXP_OWNER": "hellorpg", "EXP_PROJECT": "MOTIP2",
    "EXP_GROUP": None, "EXP_NAME": None, "GIT_VERSION": None,
    "SAVE_CHECKPOINT_PER_EPOCH": 1, "MULTI_LAST_CHECKPOINTS": 0, "USE_PREVIOUS_CHECKPOINT": False,
}

# ---- per size (App. A: LR 8e-4 N / 2e-4 S,M / 2.5e-4 L,X; backbone x0.5/0.5/0.1/0.05/0.01; WD = DEIM's per size) --
# hidden / layers = DEIM *decoder* (ID decoder mirrors it). `repeat` = DEIM base_size_repeat (N: ASSUMPTION = S's).
SIZES = {
    #      lr      bb_scale  wd        hidden layers repeat  imgs/GPU clips/GPU det_frames
    "N": (8e-4,   0.5,      1e-4,     128,   3,     20,     32,       4,        8),
    "S": (2e-4,   0.5,      1e-4,     256,   3,     20,     8,        2,        4),
    "M": (2e-4,   0.1,      1e-4,     256,   4,     6,      8,        2,        4),
    "L": (2.5e-4, 0.05,     1.25e-4,  256,   6,     4,      8,        2,        4),
    "X": (2.5e-4, 0.01,     1.25e-4,  256,   6,     3,      8,        2,        4),
}

# ---- per benchmark (§1.3) -------------------------------------------------------------------------------------------
BENCHES = {
    "dancetrack": dict(DATASETS=["DanceTrack"], DATASET_SPLITS=["train"], T=30, K=50, DET=0.3, NEWBORN=0.7,
                       INFERENCE_DATASET="DanceTrack", INFERENCE_SPLIT="val"),
    "sportsmot": dict(DATASETS=["SportsMOT"], DATASET_SPLITS=["train"], T=60, K=50, DET=0.3, NEWBORN=0.7,
                      INFERENCE_DATASET="SportsMOT", INFERENCE_SPLIT="val"),
    # PersonPath22 co-trained with CrowdHuman (MOTIP recipe, PLAN decision #13). Loader: data/personpath22.py.
    "personpath22": dict(DATASETS=["PersonPath22", "CrowdHuman"], DATASET_SPLITS=["train", "train"], T=30, K=300,
                         DET=0.4, NEWBORN=0.6, INFERENCE_DATASET="PersonPath22_Inference", INFERENCE_SPLIT="test",
                         NUM_TRAINING_IDS=100),
}


def compose(size: str, bench: str, stage: int) -> dict:
    lr, bb, wd, hidden, layers, repeat, imgs, clips, det_frames = SIZES[size]
    b = BENCHES[bench]
    cfg = dict(BASE)
    cfg.update({
        "MODEL_SIZE": size, "DETR_HIDDEN_DIM": hidden, "FEATURE_DIM": hidden, "ID_DIM": hidden,
        "NUM_ID_DECODER_LAYERS": layers, "NUM_ID_VOCABULARY": b["K"], "REL_PE_LENGTH": b["T"],
        "NUM_TRAINING_IDS": b.get("NUM_TRAINING_IDS", b["K"]),
        "DATASETS": b["DATASETS"], "DATASET_SPLITS": b["DATASET_SPLITS"],
        "MISS_TOLERANCE": b["T"], "DET_THRESH": b["DET"], "NEWBORN_THRESH": b["NEWBORN"],
        "INFERENCE_DATASET": b["INFERENCE_DATASET"], "INFERENCE_SPLIT": b["INFERENCE_SPLIT"],
        "LR": lr, "LR_BACKBONE_SCALE": bb, "WEIGHT_DECAY": wd, "BASE_SIZE_REPEAT": repeat,
    })
    if stage == 1:     # detector pretraining on the tracking dataset (single frames), no warmup, EMA warmup 0
        cfg.update({
            "ONLY_DETR": True, "SAMPLE_LENGTHS": [1], "DETR_NUM_TRAIN_FRAMES": 100, "BATCH_SIZE": imgs,
            "EPOCHS": 20,                      # PLAN decision #4 (MOTIP's pretrain recipe length)
            "LR_WARMUP_ITERS": 0, "EMA_WARMUPS": 0, "INFERENCE_ONLY_DETR": True,
            "DETR_PRETRAIN": f"./pretrains/deim_hgnetv2_{size.lower()}_coco.pth",
            "SAVE_CHECKPOINT_PER_EPOCH": 5,
        })
    else:              # joint training of detector + ID decoder
        cfg.update({
            "SAMPLE_LENGTHS": [b["T"]], "DETR_NUM_TRAIN_FRAMES": det_frames, "BATCH_SIZE": clips,
            "EPOCHS": 10,                      # PLAN N3 (Paper Fig. 5: epochs 0-9)
            "LR_WARMUP_ITERS": 500, "EMA_WARMUPS": 1000,
            "DETR_PRETRAIN": f"./pretrains/deim_hgnetv2_{size.lower()}_{bench}_stage1.pth",
        })
    return cfg


def write(name: str, cfg: dict):
    path = os.path.join(OUT, name)
    with open(path, "w") as f:
        f.write("# GENERATED by tools/gen_motip2_configs.py -- edit the generator, not this file.\n")
        yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=None)
    return path


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    n = 0
    for size in SIZES:
        for bench in BENCHES:
            write(f"{size.lower()}_{bench}.yaml", compose(size, bench, 2)); n += 1
            write(f"pretrain_{size.lower()}_{bench}.yaml", compose(size, bench, 1)); n += 1
    print(f"wrote {n} configs to {os.path.abspath(OUT)}")
