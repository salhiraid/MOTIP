# MOTIP2 on top of this MOTIP codebase

Implementation of *MOTIP2: Spatial Priors for End-to-End Multi-Object Tracking* (arXiv 2610.10391, BMVC 2026) following
`PLAN.md` (decisions, gates, targets) and `RESULTS.md` (every run and gate result). Status of each gate is in `RESULTS.md`.

## What was added
| Paper | Where | Flag (config key / CLI) |
|---|---|---|
| DEIM detector (§3.1) | `models/deim/` (vendored, Apache-2.0) | `DETECTOR=deim`, `MODEL_SIZE=N/S/M/L/X` |
| Additive fusion, FFN ×2 (§3.1) | `models/motip/id_decoder.py` | `ID_FUSION=add` |
| RoPE + inference timescale α (§3.1) | `models/motip/rope.py` | `POS_ENC=rope`, `ROPE_ALPHA`, `ROPE_ALPHA_MODE` |
| Spatial ID Switches (§3.2) | `data/id_switch.py` | `ID_SWITCH=spatial`, `ID_SWITCH_SIGMA`, `ID_SWITCH_REFERENCE`, `ID_SWITCH_ABSENT` |
| Spatial ID Loss (§3.3) | `models/motip/id_criterion.py` | `ID_LOSS_SPATIAL`, `ID_LAMBDA_*`, `ID_CE_IOU_WEIGHT` |
| Spatial Anchor (§3.4) | `id_decoder.py`, `models/deim/dfine_decoder.py` | `SPATIAL_ANCHOR`, `SPATIAL_ANCHOR_SOURCE` |
| Tracklet confirmation (§3.1) | `models/runtime_tracker.py` | `N_INIT` |
| Two-stage training, EMA, constant LR (App. A) | `train.py` | `ONLY_DETR`, `USE_EMA`, `LR_SCHEDULER`, `LR_WARMUP_ITERS` |

`ID_SWITCH=uniform, ID_LOSS_SPATIAL=False, ID_CE_IOU_WEIGHT=False, SPATIAL_ANCHOR=False, POS_ENC=learned_rel,
ID_FUSION=concat, N_INIT=1` reproduces MOTIP's decoder/augmentation/tracker exactly (bit-identical in tests).

## Setup
```bash
pip install torch torchvision einops scipy pyyaml pillow accelerate wandb pytest   # DEIM needs no CUDA extension
python tools/gen_motip2_configs.py           # regenerates configs/motip2/*.yaml from the paper's numbers
python -m pytest tests -q -m "not slow"        # CPU tests (drop -m to include the end-to-end/DDP/learning ones)
```
DEIM COCO checkpoints (official, Google Drive links in the DEIM README) go to `./pretrains/deim_hgnetv2_{n,s,m,l,x}_coco.pth`.
Datasets: DanceTrack / SportsMOT / CrowdHuman as in `docs/DATASET.md`; PersonPath22 as MOT-style folders, see `data/personpath22.py`.

## Run
```bash
# stage 1: detector pretraining on the tracking dataset (paper: 4 GPUs, 8 images/GPU)
accelerate launch --num_processes=4 train.py --data-root ./datasets --config-path configs/motip2/pretrain_s_dancetrack.yaml \
    --exp-name deim_s_dancetrack_stage1
cp outputs/deim_s_dancetrack_stage1/checkpoint_19.pth pretrains/deim_hgnetv2_s_dancetrack_stage1.pth
# stage 2: joint training (paper: 4 GPUs, 2 clips/GPU)
accelerate launch --num_processes=4 train.py --data-root ./datasets --config-path configs/motip2/s_dancetrack.yaml --exp-name s_dancetrack
# evaluation (DanceTrack val; EMA weights are used automatically)
accelerate launch --num_processes=4 submit_and_evaluate.py --data-root ./datasets --inference-mode evaluate \
    --config-path configs/motip2/s_dancetrack.yaml --inference-model outputs/s_dancetrack/checkpoint_9.pth \
    --inference-dataset DanceTrack --inference-split val --outputs-dir outputs/s_dancetrack
python tools/log_result.py --summary <.../tracker/pedestrian_summary.txt> --config configs/motip2/s_dancetrack.yaml \
    --seed 42 --benchmark DanceTrack --split val --checkpoint outputs/s_dancetrack/checkpoint_9.pth --notes "variant=full"
```
Ablations / sweeps: `DRY_RUN=1 SIZE=S BENCH=dancetrack bash tools/run_ablations.sh` prints every command (Tab. 5, 6, 9).
Tab. 8 statistics: `python tools/idsw_stats.py --config configs/motip2/s_dancetrack.yaml --data-root ./datasets --mode spatial`.
Latency (Tab. 4 protocol): `python tools/benchmark_latency.py --config configs/motip2/s_dancetrack.yaml --baseline-config configs/r50_deformable_detr_motip_dancetrack.yaml`.
Mean ± std over seeds: `python tools/aggregate_results.py`.

## Smoke test without data/GPU
`python tools/make_synthetic_dancetrack.py --out /tmp/synth` then
`python train.py --config-path configs/motip2/smoke_stage1.yaml --data-root /tmp/synth --exp-name s1 --outputs-dir /tmp/o1`
(set `DETR_PRETRAIN` in the smoke configs to any DEIM-N checkpoint). Proves the plumbing only; metrics are meaningless.

## Things to know before the first GPU run
* Not run on a GPU or on real data by the author of these changes: Gate 1 and all metric gates are open (see `RESULTS.md`).
* If DDP complains about unused parameters set `DDP_FIND_UNUSED_PARAMETERS: true`.
* Inference never applies ImageNet normalisation for DEIM (`IMAGE_NORMALIZE: false`); legacy configs are unchanged.
* `ROPE_ALPHA_MODE` default is `multiply` (skill default); run the Gate-3 A/B against `divide` (PLAN decision #8).
