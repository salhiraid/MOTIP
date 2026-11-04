#!/usr/bin/env python
"""Generate MOTIP2 configs (stage 1 detector pre-training + stage 2 joint training) for training on ONE OR SEVERAL datasets
converted by tools/prepare_dataset.py (or your own `Vehicles` set), all sharing one class set.

    python tools/gen_mix_config.py --name mix_a --size S \
        --datasets Vehicles BDD100K_MOT UA_DETRAC --mix 0.5 0.3 0.2 --data-root ./datasets

* --mix = share of every epoch's samples taken from each dataset (normalised).  Default: equal shares.  A dataset that is
  small is cycled, a large one is sub-sampled, so BDD100K's 2000 videos do not drown a 12-video dataset.
* --datasets accepts `Name` or `Name:split` (default split `train`).
* The frame-interval cap of every dataset follows its frame rate (5 fps -> 1 ... 25+ fps -> 4; data/class_maps.py).
* CLASS_MASKING is on: a dataset gets no classification loss for classes it does not annotate (UA-DETRAC has no
  motorcycle/bicycle/truck labels: its unlabelled ones must not be trained as background).  Class-less datasets
  (CityFlow, Waymo) therefore only teach box regression and identities.
* Prints the launch commands and warns about licences / coarse labels / missing data.
"""
import argparse
import json
import os
import sys

import yaml

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gen_motip2_configs import compose, SIZES   # noqa: E402
from data.class_maps import CLASS_SETS, coco_rows, default_max_interval   # noqa: E402
from data.unified_mot import UNIFIED_DATASETS, DATASET_INFO               # noqa: E402


def parse_datasets(items):
    out = []
    for it in items:
        name, _, split = it.partition(":")
        out.append((name, split or "train"))
    return out


def build(a):
    ds = parse_datasets(a.datasets)
    mix = a.mix or [1.0] * len(ds)
    if len(mix) != len(ds):
        raise SystemExit(f"--mix has {len(mix)} values for {len(ds)} datasets")
    if any(m < 0 for m in mix) or sum(mix) <= 0:
        raise SystemExit("--mix must be >= 0 and not all zero")
    warns = []
    k_rec = []
    for (name, split) in ds:
        info = DATASET_INFO.get(name)
        if info is None:
            warns.append(f"{name}: not a registered dataset - loaded from <data-root>/{name} with the unified loader")
        else:
            if "non-commercial" in info["licence"]:
                warns.append(f"{name}: NON-COMMERCIAL licence - a model trained on it may not be usable commercially")
            elif info["licence"].startswith("unverified"):
                warns.append(f"{name}: licence not verified by me ({info['licence']}) - check before commercial use")
            if info["classes"] == "coarse":
                warns.append(f"{name}: class-less/coarse labels -> classification loss masked; it teaches boxes + identities only")
        d = os.path.join(a.data_root, UNIFIED_DATASETS.get(name, name))
        if a.check_data:
            if not os.path.isdir(os.path.join(d, split)):
                warns.append(f"{name}: {os.path.join(d, split)} not found (run tools/prepare_dataset.py first)")
            else:
                ac = os.path.join(d, "annotated_classes.json")
                if os.path.exists(ac):
                    j = json.load(open(ac))
                    if j["class_set"] != list(CLASS_SETS[a.class_set]):
                        warns.append(f"{name}: converted with class set {j['class_set']} != {list(CLASS_SETS[a.class_set])}")
                rp = os.path.join(d, "prepare_report.json")
                if os.path.exists(rp):
                    k_rec.append(json.load(open(rp))["stats"].get("recommended_NUM_ID_VOCABULARY", 0))
    K = a.num_id_vocabulary or max([100] + k_rec)
    caps = []
    for (name, _) in ds:
        info = DATASET_INFO.get(name, {})
        caps.append(a.max_interval if a.max_interval else default_max_interval(info.get("fps") or 25))
    bench = dict(DATASETS=[n for n, _ in ds], DATASET_SPLITS=[s for _, s in ds], T=a.window, K=K, DET=a.det_thresh,
                 NEWBORN=a.newborn_thresh, NUM_TRAINING_IDS=a.num_training_ids or K,
                 INFERENCE_DATASET=a.eval_dataset or ds[0][0], INFERENCE_SPLIT=a.eval_split, NUM_CLASSES=len(CLASS_SETS[a.class_set]))
    out = {}
    for stage in (1, 2):
        cfg = compose(a.size, bench, stage)
        cfg["DATASET_MIX"] = [float(m) for m in mix]
        cfg["DATASET_MAX_INTERVALS"] = caps
        cfg["CLASS_MASKING"] = True
        cfg["NUM_KEYPOINTS"] = a.num_keypoints
        cfg["DATA_ROOT"] = a.data_root
        if stage == 1:
            if a.from_coco:
                cfg["DETR_PRETRAIN"] = a.detr_pretrain or f"./pretrains/deim_hgnetv2_{a.size.lower()}_coco.pth"
                cfg["DETR_CLASS_INDICES"] = coco_rows(a.class_set)
            else:
                cfg["DETR_PRETRAIN"] = a.detr_pretrain or f"./pretrains/deim_{a.size.lower()}_vehicles_converted.pth"
        else:
            cfg["DETR_PRETRAIN"] = f"./pretrains/deim_hgnetv2_{a.size.lower()}_{a.name}_stage1.pth"
        out[stage] = cfg
    return out, warns, K, caps


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="config/experiment name, e.g. mix_a")
    ap.add_argument("--size", default="S", choices=sorted(SIZES))
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--mix", nargs="*", type=float, default=None)
    ap.add_argument("--class-set", default="vehicles5", choices=sorted(CLASS_SETS))
    ap.add_argument("--data-root", default="./datasets/")
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "configs", "motip2", "mixes"))
    ap.add_argument("--window", type=int, default=30, help="clip length T (= miss tolerance = RoPE window)")
    ap.add_argument("--num-id-vocabulary", type=int, default=0, help="K; default max(100, recommendations of the converted datasets)")
    ap.add_argument("--num-training-ids", type=int, default=0)
    ap.add_argument("--max-interval", type=int, default=0, help="force the same frame-interval cap for every dataset")
    ap.add_argument("--det-thresh", type=float, default=0.3)
    ap.add_argument("--newborn-thresh", type=float, default=0.7)
    ap.add_argument("--eval-dataset", default=None, help="in-training evaluation dataset (default: the first one)")
    ap.add_argument("--eval-split", default="valmini")
    ap.add_argument("--num-keypoints", type=int, default=0, help="per-detection keypoints (e.g. 31); needs labelled data, see docs/KEYPOINTS.md")
    ap.add_argument("--from-coco", action="store_true", help="stage 1 starts from official COCO DEIM weights (class rows sliced)")
    ap.add_argument("--detr-pretrain", default=None)
    ap.add_argument("--no-check-data", dest="check_data", action="store_false")
    a = ap.parse_args(argv)
    cfgs, warns, K, caps = build(a)
    os.makedirs(a.out_dir, exist_ok=True)
    paths = {}
    for stage, cfg in cfgs.items():
        paths[stage] = os.path.join(a.out_dir, f"{'pretrain_' if stage == 1 else ''}{a.name}.yaml")
        with open(paths[stage], "w") as f:
            f.write("# GENERATED by tools/gen_mix_config.py -- re-run it instead of editing.\n")
            yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=None)
    rel = lambda p: os.path.relpath(p, os.getcwd())
    print(f"datasets {a.datasets}  mix {a.mix or 'equal'}  frame-interval caps {caps}  K={K}  T={a.window}")
    for w in warns:
        print("WARNING:", w)
    print(f"wrote {rel(paths[1])} and {rel(paths[2])}\n")
    print(f"""# stage 1 (detector on the mixture), then stage 2 (joint)
accelerate launch --num_processes=4 train.py --config-path {rel(paths[1])} --exp-name {a.name}_stage1 --epochs 5
cp outputs/{a.name}_stage1/checkpoint_4.pth pretrains/deim_hgnetv2_{a.size.lower()}_{a.name}_stage1.pth
accelerate launch --num_processes=4 train.py --config-path {rel(paths[2])} --exp-name {a.name}
# per-dataset evaluation of the final checkpoint:
python tools/eval_datasets.py --config {rel(paths[2])} --model outputs/{a.name}/checkpoint_9.pth \\
    --datasets {' '.join(f'{n}:val' for n, _ in parse_datasets(a.datasets))} --outputs-dir outputs/{a.name} --run""")
    return cfgs, warns


if __name__ == "__main__":
    main()
