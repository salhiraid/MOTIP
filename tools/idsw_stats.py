# MOTIP2 PLAN Gate 5 (statistics): realised Spatial-ID-Switch statistics over one epoch (Paper App. C, Tab. 8):
# Swap% (objects whose identity changed), LR% (swapped pairs with GIoU <= 0), mean GIoU of swapped pairs.
# CPU only, no model, no image decoding.
#   python tools/idsw_stats.py --config configs/motip2/s_dancetrack.yaml --data-root ./datasets --mode spatial
#   (targets for DanceTrack: spatial 10% / 26% / 0.20, uniform 37% / 85% / -0.40; PersonPath22: 10/31/0.15, 43/94/-0.65)
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from configs.util import load_super_config
from data.id_switch import SwitchStats
from data.joint_dataset import JointDataset
from data.naive_sampler import NaiveSampler
from data.transforms import build_transforms, TurnIntoTrajectoryAndUnknown
from utils.misc import yaml_to_dict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--mode", default=None, choices=["none", "uniform", "spatial"])
    ap.add_argument("--reference", default=None, choices=["per_frame", "last_frame"])
    ap.add_argument("--absent", default=None, choices=["skip", "neg_one"])
    ap.add_argument("--sigma", type=float, default=None)
    ap.add_argument("--epoch", type=int, default=0)
    ap.add_argument("--max-samples", type=int, default=1000, help="subsample the epoch (0 = full epoch)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    cfg = yaml_to_dict(a.config)
    cfg = load_super_config(cfg, cfg.get("SUPER_CONFIG_PATH"))
    for k, v in (("ID_SWITCH", a.mode), ("ID_SWITCH_REFERENCE", a.reference), ("ID_SWITCH_ABSENT", a.absent),
                 ("ID_SWITCH_SIGMA", a.sigma)):
        if v is not None:
            cfg[k] = v
    cfg["DATA_ROOT"] = a.data_root
    torch.manual_seed(a.seed)

    transforms = build_transforms(cfg)
    stats = SwitchStats()
    for t in transforms.transforms:
        if isinstance(t, TurnIntoTrajectoryAndUnknown):
            t.stats = stats
    dataset = JointDataset(data_root=a.data_root, datasets=cfg["DATASETS"], splits=cfg["DATASET_SPLITS"],
                           transforms=transforms, size_divisibility=cfg.get("SIZE_DIVISIBILITY", 0), fake_images=True)
    sampler = NaiveSampler(dataset, cfg["SAMPLE_STEPS"], cfg["SAMPLE_LENGTHS"], cfg["SAMPLE_INTERVALS"],
                           cfg["LENGTH_PER_ITERATION"], samples_per_epoch=a.max_samples or cfg.get("SAMPLES_PER_EPOCH"))
    sampler.prepare_for_epoch(a.epoch)
    for info in sampler:
        dataset[info]
    r = stats.summary()
    print(f"mode={cfg['ID_SWITCH']} reference={cfg['ID_SWITCH_REFERENCE']} absent={cfg['ID_SWITCH_ABSENT']} "
          f"sigma={cfg['ID_SWITCH_SIGMA']} samples={len(sampler)}")
    print(f"Swap% = {r['Swap%']:.1f}   LR% = {r['LR%']:.1f}   mean GIoU = {r['meanGIoU']:.2f}   "
          f"(objects {r['n_obj']}, swapped {r['n_swapped']})")


if __name__ == "__main__":
    main()
