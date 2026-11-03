# MOTIP2 on top of the MOTIP codebase

This branch (`motip2`) turns the existing MOTIP (CVPR 2025) codebase into **MOTIP2** as described in
*"MOTIP2: Spatial Priors for End-to-End Multi-Object Tracking"* (arXiv 2610.10391, BMVC 2026). The original MOTIP README
is [README.md](./README.md); everything below is about what was added on top of it.

> **Honest status.** All code is written and tested (169 CPU tests incl. behavioural, integration, 2-process DDP and ONNX-parity tests; see
> [AUDIT.md](./AUDIT.md)) and the full pipeline (data → training → tracker → TrackEval) runs end to end on a *synthetic* dataset. It has **not** been run on a GPU, on real
> DanceTrack/SportsMOT/PersonPath22 data, or with real DEIM weights, so **no number from the paper has been reproduced yet**.
> Gate-by-gate status is kept in [RESULTS.md](./RESULTS.md); the full plan, every design decision and the target numbers
> are in [PLAN.md](./PLAN.md).

---

## 1. What MOTIP2 changes (paper → this code)

| Paper | What it does | Where in the code | Switch (config key / CLI flag) |
|---|---|---|---|
| **DEIM detector** (§3.1) | Replaces MOTIP's Deformable-DETR with the real-time DEIM detector (N/S/M/L/X) | `models/deim/` (vendored from DEIM, Apache-2.0), `models/motip/__init__.py` | `DETECTOR` = `deim` \| `deformable_detr`, `MODEL_SIZE` = `N/S/M/L/X` |
| **Additive fusion** (§3.1) | ID decoder input is `e_i + id_embedding` instead of concatenation → half the hidden dim, FFN ratio ×2 | `models/motip/id_decoder.py` | `ID_FUSION` = `add` \| `concat` |
| **RoPE** (§3.1) | Rotary embedding inside the query–key product replaces MOTIP's learned temporal bias; inference timescale rescaled by α | `models/motip/rope.py`, `id_decoder.py` | `POS_ENC` = `rope` \| `learned_rel`, `ROPE_ALPHA`, `ROPE_ALPHA_MODE` |
| **Spatial ID Switches** (§3.2) | ID-switch augmentation draws permutations from `GIoU + N(0,σ²)` via the Hungarian algorithm instead of uniformly | `data/id_switch.py`, `data/transforms.py` | `ID_SWITCH` = `none` \| `uniform` \| `spatial`, `ID_SWITCH_SIGMA` (0.5) |
| **Spatial ID Loss** (§3.3) | CE plus `Σ_c p_c‖b̂−b_c‖₁` and `Σ_c p_c(1−GIoU)` over the identities present in the frame + newborn, box detached | `models/motip/id_criterion.py` | `ID_LOSS_SPATIAL`, `ID_LAMBDA_CE/GIOU/L1` (1/2/5), `ID_CE_IOU_WEIGHT` |
| **Spatial Anchor** (§3.4) | The detector's positional encoding `p_i` is added to every detection token entering the ID decoder (no new parameters) | `models/deim/dfine_decoder.py` (exposes `p_i`), `id_decoder.py`, `train.py`, `runtime_tracker.py` | `SPATIAL_ANCHOR`, `SPATIAL_ANCHOR_SOURCE` |
| **Tracklet confirmation** (§3.1) | A new track is output only after `N_init`=2 consecutive consistent assignments; unconfirmed tracks are discarded and their slot freed | `models/runtime_tracker.py` | `N_INIT` |
| **Training recipe** (App. A) | 2-stage (DEIM pretrain → joint), AdamW, EMA 0.9995, constant LR, 500-iter warmup, per-size LRs, ID decoder LR ×2 | `train.py`, `configs/motip2/` | see §6 |

Setting `ID_FUSION=concat POS_ENC=learned_rel SPATIAL_ANCHOR=False ID_SWITCH=uniform ID_LOSS_SPATIAL=False
ID_CE_IOU_WEIGHT=False N_INIT=1` reproduces MOTIP's decoder, augmentation, loss and tracker **exactly** — this is
asserted bit-for-bit in the tests against frozen copies of the original code (`tests/legacy_*.py`).

---

## 2. What was done, exactly

Work followed the phased plan in `PLAN.md`; each phase is one commit (`git log --oneline`).

### Phase 0 — recon and plan (`PLAN.md`, `RESULTS.md`)
* Mapped the MOTIP repo (detector wrapper, ID decoder, sampler, ID-switch augmentation, loss, train loop, tracker, eval, configs) and the DEIM repo.
* Found where DEIM produces the positional encoding `p_i`: `DFINETransformer.query_pos_head` (`dfine_decoder.py`), applied at the top of each decoder layer to the box that **enters** that layer. At the last layer that is layer L−1's refined box, **not** the final output box. Both readings are implemented (`SPATIAL_ANCHOR_SOURCE=layer_input` default, `final_box` available).
* Wrote the file-by-file change list, config flags, decisions for every ambiguity, gates with exact commands, and target numbers.

### Phase 1 — eval tooling (`phase-1`)
* `submit_and_evaluate.py` now also records IDSW / FP / FN; `tools/log_result.py` appends a run row to `RESULTS.md`.
* **Gate 1 (reproduce MOTIP 64.0 HOTA on DanceTrack val) was not run** — it needs a GPU, data and the released checkpoint.

### Phase 2 — DEIM detector (`phase-2`)
* Vendored DEIM (`models/deim/`), patched to also return the content embedding `e_i` and `p_i`, and to keep the final-layer Hungarian indices (used to supervise the ID decoder). Wrapper `DEIMDetector`, `DetectionCriterion` (MAL 1 / L1 5 / GIoU 2 / FGL 0.15 / DDF 1.5, α=.75, γ=1.5, matcher costs 2/5/2), COCO-checkpoint loader that slices the 80-class head to "person".
* `train.py`: detection loss on 4 random frames/clip (8 for N) with the remaining frames run without gradient in a lightweight mode; EMA (0.9995, warmup 0 in stage 1 / 1000 in stage 2); constant LR with 500-iteration linear warmup; param groups (backbone ×0.5/0.5/0.1/0.05/0.01, ID decoder ×2, no weight decay on norm/bias/embeddings); batch-level multi-scale (75–125 % of 800, multiples of 32); no ImageNet normalisation for DEIM; `SAMPLES_PER_EPOCH=40000`.
* Image augmentation of App. A: flip, zoom-out [1,2], IoU-aware crop (p .8, scale [.3,1], aspect [1.5,2]), photometric.

### Phases 3, 4, 7 — ID decoder, Spatial Anchor, tracker (`phase-3/4/7`)
* `id_decoder.py`: `add` fusion, RoPE cross-attention (`RoPECrossAttention`, same `add_zero_attn` semantics as MOTIP), Spatial Anchor, α applied **only at inference**. ID decoder width/depth mirror the DEIM decoder (N 128×3, S 256×3, M 256×4, L/X 256×6).
* `runtime_tracker.py`: keeps `p_i` in the context buffer, applies α, implements `N_INIT` confirmation (tentative tracks stay in the context so they can be re-identified; unconfirmed + missed tracks are removed and their slot returns to the front of the free queue; no back-filling of the withheld first frame), reads the feature dim from the model (was hard-coded 256).

### Phases 5, 6 — Spatial ID Switches and Spatial ID Loss (`phase-5/6`)
* `data/id_switch.py`: per-frame (default) or last-frame reference, GIoU matrix with `G_ii=1`, Gaussian noise on every cell, `linear_sum_assignment(..., maximize=True)`; absent slots are skipped (or treated as GIoU=−1); `uniform` mode is MOTIP's original code unchanged; `SwitchStats` reproduces the Table-8 statistics (Swap %, LR %, mean GIoU).
* `id_criterion.py`: exactly the three terms of §3.3 / App. A. Probabilities are re-normalised over `C_i`; the newborn class uses the detection's own GT box; `b̂` is detached; CE is weighted per sample by the detached IoU.

### Phase 8/9 — configs, data, tools, docs (`phase-8/9`)
* `tools/gen_motip2_configs.py` composes **30 configs** (N/S/M/L/X × DanceTrack/SportsMOT/PersonPath22 × stage 1/2) from the paper's numbers. `tests/test_configs_and_data.py` asserts each number.
* `data/personpath22.py` (the repo had no PersonPath22 loader), `tools/benchmark_latency.py`, `tools/idsw_stats.py`, `tools/run_ablations.sh`, `tools/aggregate_results.py`, `tools/make_synthetic_dancetrack.py`.

### Bugs in the existing code found and fixed on the way
* Inference always applied ImageNet normalisation → wrong for DEIM. Now controlled by `IMAGE_NORMALIZE`.
* Original tracker crashed when #newborn detections > #free ID slots while some detections were not newborn.
* `data/transforms.py` imported `triton` (unused) and used torchvision private APIs that were renamed (colour jitter / photometric distort were broken on current torchvision).
* `.cuda()` / `torch.cuda.synchronize()` assumptions; `INFERENCE_MAX_SHORTER` (800) was not configurable.

### Decisions that deviate from the skill's defaults (you approved these)
| # | Decision |
|---|---|
| 3 | Weight decay = DEIM's per-size values (1e-4 N/S/M, 1.25e-4 L/X), not MOTIP's 5e-4 |
| 4 | Stage-1 length = 20 epochs (MOTIP's own pretrain recipe), not DEIM's COCO-scale schedule |
| 6 | Spatial ID Switch uses each frame's own boxes (`ID_SWITCH_REFERENCE=per_frame`); the skill's last-frame variant is a flag |
| 8 | α default stays `multiply` (skill) but `divide` is implemented; decide by an inference-only A/B (Gate 3) |
| N1 | `p_i` taken from inside the last decoder layer (box entering it); `final_box` is an ablation flag |

Everything else (all other ambiguities) is documented in `PLAN.md` §5.

---

## 2b. Vehicle tracking on your own videos (added later)

A complete workflow for a CSV-indexed collection of annotated videos (CVAT video XML) and a detector trained in mmdet is in
**[docs/VEHICLES.md](./docs/VEHICLES.md)**. What was added for it:

| Piece | File |
|---|---|
| CSV + CVAT-XML + video → frames, MOT-style GT, **site-disjoint train/val split**, `valmini`, statistics, K recommendation | `tools/vehicles/prepare_vehicle_dataset.py` |
| Ground-truth overlay to check boxes/ids/classes | `tools/vehicles/visualize_gt.py` |
| mmdet DEIM checkpoint → official-DEIM layout (tensor alignment, report, refuses incomplete conversions) | `tools/vehicles/convert_mmdet_deim.py` |
| Multi-class (5 vehicle classes) detector head loading, e.g. COCO-80 rows `1 2 3 5 7` | `models/deim/wrapper.py` (`--detr-class-indices`) |
| `Vehicles` dataset class (+ evaluation registration) | `data/vehicles.py` |
| Configs | `configs/motip2/{pretrain_,}{n,s,m,l,x}_vehicles.yaml` |
| Synthetic videos + CVAT XML for testing | `tools/vehicles/make_synthetic_vehicle_videos.py` |

Quick start: prepare data (`--dry-run` first) → `visualize_gt.py` → convert detector → detection-only check → short stage 1 → stage 2 →
evaluate on `val`; every command is in the guide. Not run on your real data/GPU; see its status box.

## 2c. Training on one or several public tracking datasets (added later)

BDD100K MOT, UA-DETRAC, VisDrone-MOT, UAVDT, KITTI tracking, CityFlow, the FastTracker benchmark (any MOT-format data), Waymo (v2 2D
camera boxes), nuScenes / Argoverse 2 (3D boxes projected to 2D) and generic 3D-box data (e.g. I-24 3D) are converted to **one unified
layout and one class space** (default: your 5 classes), then mixed in a single run. Guide, per-dataset input layouts, label mappings,
licence warnings and the verification status: **[docs/DATASETS.md](./docs/DATASETS.md)**.

| Piece | File |
|---|---|
| Unified loader (any folder in the layout is loadable by name) + dataset facts | `data/unified_mot.py` |
| Class sets, label → class resolution, frame-interval rule | `data/class_maps.py` |
| One converter per dataset + shared writer (`--dry-run`, image link/copy/jpeg+resize, statistics, K recommendation) | `data/converters/*`, CLI `tools/prepare_dataset.py` |
| 3D → 2D cuboid projection (near-plane clipping) | `data/converters/project3d.py` |
| Exact per-dataset shares of every epoch (`DATASET_MIX`), per-dataset frame-interval cap (`DATASET_MAX_INTERVALS`) | `data/naive_sampler.py`, `train.py` |
| No classification loss for classes a dataset does not annotate (`CLASS_MASKING`) | `models/deim/deim_criterion.py`, `data/joint_dataset.py` |
| Config generator for any mixture (stage 1 + 2, warnings, commands) | `tools/gen_mix_config.py` |
| Same checkpoint on several datasets → one table | `tools/eval_datasets.py` |

```bash
python tools/prepare_dataset.py detrac --root /data/ua-detrac --out ./datasets --dry-run
python tools/gen_mix_config.py --name mix_a --size S --datasets Vehicles UA_DETRAC --mix 0.7 0.3
```
Status: tested on synthetic files in each dataset's native format and on a CPU training run; **not** run on the real downloads (no access
here), and no accuracy claim is made for any mixture.

## 3. Installation

```bash
pip install torch torchvision einops scipy pyyaml pillow accelerate wandb pytest pandas pyarrow     # DEIM needs no CUDA extension
# only for the ORIGINAL MOTIP baseline (Deformable DETR): compile its CUDA op
(cd models/ops && sh make.sh)
```
Original MOTIP install notes are in [docs/INSTALL.md](./docs/INSTALL.md).

## 4. Data and weights

* **Datasets** (layout in [docs/DATASET.md](./docs/DATASET.md)): `datasets/DanceTrack/{train,val,test}` + `*_seqmap.txt`, `SportsMOT`, `CrowdHuman` (after `tools/gen_crowdhuman_gts.py`).
* **PersonPath22** (new loader, MOT-style folders — *unverified against the real data, adapt if yours differs*):
  ```
  datasets/PersonPath22/{train,val}/<seq>/{img1/00000001.jpg, gt/gt.txt, seqinfo.ini}
  datasets/PersonPath22_Inference/test/<seq>/{img1/..., seqinfo.ini}
  datasets/PersonPath22_Inference/gts/person_path_22-test/<seq>/{gt/gt.txt, seqinfo.ini}
  datasets/PersonPath22_Inference/gts/seqmaps/person_path_22-test.txt
  ```
* **DEIM COCO checkpoints** (official, Google-Drive links in the DEIM README) → `pretrains/deim_hgnetv2_{n,s,m,l,x}_coco.pth`.
* The released MOTIP checkpoint for the baseline (Gate 1): `pretrains/r50_deformable_detr_motip_dancetrack.pth` (see [docs/MODEL_ZOO.md](./docs/MODEL_ZOO.md)).

## 5. Quick checks (no GPU, no data needed)

```bash
python -m pytest tests -q -m "not slow"           # fast subset, ~5 min on CPU (drop -m to include the ~10 min of end-to-end tests)
python tools/make_synthetic_dancetrack.py --out /tmp/synth        # tiny fake DanceTrack-format dataset
# end-to-end smoke: stage 1 (detector only) ...
python train.py --config-path configs/motip2/smoke_stage1.yaml --data-root /tmp/synth \
    --detr-pretrain <any DEIM-N checkpoint> --exp-name s1 --outputs-dir /tmp/o1
# ... then stage 2 (all priors on) using the stage-1 checkpoint
python train.py --config-path configs/motip2/smoke_stage2.yaml --data-root /tmp/synth \
    --detr-pretrain /tmp/o1/checkpoint_0.pth --exp-name s2 --outputs-dir /tmp/o2
```
The smoke run only proves the plumbing; its metrics are meaningless. To produce a throw-away DEIM-N checkpoint:
`python -c "import torch; from models.deim.builder import build_deim; torch.save({'model': build_deim('N', num_classes=80).state_dict()}, '/tmp/deim_n_fake.pth')"`.

## 6. Training and evaluation (paper settings: 4 GPUs)

```bash
# Stage 1 — pretrain DEIM alone on the tracking dataset (8 images/GPU, 20 epochs x 40k samples, no warmup)
accelerate launch --num_processes=4 train.py --data-root ./datasets \
    --config-path configs/motip2/pretrain_s_dancetrack.yaml --exp-name deim_s_dancetrack_stage1
cp outputs/deim_s_dancetrack_stage1/checkpoint_19.pth pretrains/deim_hgnetv2_s_dancetrack_stage1.pth

# Stage 2 — joint training of detector + ID decoder (2 clips/GPU of 30 frames, 10 epochs, EMA, 500-iter warmup)
accelerate launch --num_processes=4 train.py --data-root ./datasets \
    --config-path configs/motip2/s_dancetrack.yaml --exp-name s_dancetrack

# Evaluate on DanceTrack val (EMA weights are used automatically when present in the checkpoint)
accelerate launch --num_processes=4 submit_and_evaluate.py --data-root ./datasets --inference-mode evaluate \
    --config-path configs/motip2/s_dancetrack.yaml --inference-model outputs/s_dancetrack/checkpoint_9.pth \
    --inference-dataset DanceTrack --inference-split val --outputs-dir outputs/s_dancetrack

# Log the run in RESULTS.md (tag variant=... so tools/aggregate_results.py can group seeds)
python tools/log_result.py --summary outputs/s_dancetrack/evaluate/default/DanceTrack/val/checkpoint_9/tracker/pedestrian_summary.txt \
    --config configs/motip2/s_dancetrack.yaml --seed 42 --benchmark DanceTrack --split val \
    --checkpoint outputs/s_dancetrack/checkpoint_9.pth --notes "variant=full"
```
Use `--inference-split test` (DanceTrack/SportsMOT) to write submission files with `--inference-mode submit`.
Config names: `{n,s,m,l,x}_{dancetrack,sportsmot,personpath22}.yaml` (stage 2) and `pretrain_{...}.yaml` (stage 1).
Per-GPU batch sizes in the configs assume the paper's 4 GPUs; with other hardware keep the *global* batch (32 images / 8 clips; N: 128 / 16), e.g. via `--batch-size` and more processes.

### Frequently used flags (all can be given in YAML or on the command line)
| Flag | Default | Meaning |
|---|---|---|
| `--id-switch` | `spatial` | `none` / `uniform` / `spatial` |
| `--id-loss-spatial` | `True` | add the GIoU/L1 terms |
| `--spatial-anchor` | `True` | add `p_i` to detection tokens |
| `--pos-enc` | `rope` | `rope` / `learned_rel` |
| `--n-init` | `2` | tracklet confirmation length (1 = off; forced to 1 in detector-only evaluation) |
| `--n-init-ttl` | `0` | missed frames an *unconfirmed* track survives (0 = discard at first miss, Paper; >0 = skill §9.4 "after the miss tolerance") |
| `--detector` | `deim` | `deim` / `deformable_detr` |
| `--rope-alpha`, `--rope-alpha-mode` | `2.0`, `multiply` | inference timescale rescale (`multiply` / `divide`) |
| `--id-switch-reference` | `per_frame` | or `last_frame` |
| `--id-switch-absent` | `skip` | or `neg_one` |
| `--spatial-anchor-source` | `layer_input` | or `final_box` |
| `--model-size` | per config | `N S M L X` |
| `--samples-per-epoch`, `--epochs`, `--lr`, `--seed`, `--batch-size` | see config | usual |

Fixed paper hyper-parameters (do not change without a reason): σ=0.5, N_init=2, λ_ID (`ID_THRESH`)=0.1, α=2.0, inference
shorter side 800 / longer ≤1440, T=W=30 (60 SportsMOT), K=50 (300 PersonPath22), `(λ_det, λ_newborn)` = (0.3, 0.7) or (0.4, 0.6) for PersonPath22.

## 7. Ablations and sweeps

```bash
DRY_RUN=1 SIZE=S BENCH=dancetrack SEED=42 NG=4 DATA=./datasets bash tools/run_ablations.sh    # prints every command
DRY_RUN=0 ... bash tools/run_ablations.sh                                                      # runs them
```
It covers (a) full, (b) w/o ID switch, (c) uniform ID switch, (d) w/o Spatial ID Loss, (e) w/o Spatial Anchor, (f) all priors
off, (g) learned relative PE (all **retrained**, sharing one stage-1 detector), and inference-only on the full model:
(h) `N_init` ∈ {1,2,3}, (i) α ∈ {1.0, 1.5, 2.0, 2.5} × {multiply, divide}. Run ≥3 seeds before claiming a gain under 1 HOTA;
then `python tools/aggregate_results.py` prints mean ± std per variant.

## 8. Tools

| Tool | Purpose |
|---|---|
| `tools/gen_motip2_configs.py` | (Re)generate `configs/motip2/*.yaml` from the paper's numbers |
| `tools/idsw_stats.py` | Table-8 statistics (Swap %, LR %, mean GIoU) of the ID-switch augmentation over an epoch; CPU only, no images decoded. Targets: DanceTrack spatial 10 % / 26 % / 0.20 vs uniform 37 % / 85 % / −0.40 |
| `tools/benchmark_latency.py` | Detector / ID decoder / total latency in the Table-4 protocol; `--baseline-config` prints the speed-up vs MOTIP (paper: 3.2×) |
| `tools/log_result.py`, `tools/aggregate_results.py` | Maintain `RESULTS.md` and aggregate seeds |
| `tools/make_synthetic_dancetrack.py` | Synthetic dataset for smoke tests |

## 8b. Export to ONNX (Phase 10)

```bash
python tools/export_onnx.py --config configs/motip2/s_dancetrack.yaml --ckpt outputs/s_dancetrack/checkpoint_9.pth \
    --out-dir export/s --height 800 --width 1440 --verify
```
Writes `detector.onnx` (fixed resolution) and `id_decoder.onnx` (**dynamic** number of context frames, tracks and detections; RoPE α baked in),
plus `export_meta.json`. `--verify` compares both with PyTorch through onnxruntime (ID decoder at 5 different shapes, detector set-wise).
`tools/onnx_runtime_model.py` runs the existing `RuntimeTracker` on the two graphs (greedy readout / confirmation / context buffer stay in
Python, as in the paper) and the tests check it tracks identically to PyTorch. **TensorRT engines are not built** (no TensorRT/GPU in the
development environment); the ONNX graphs are the input for `trtexec`.

## 8c. Distributed testing without GPUs

`torchrun --nproc_per_node 2 --master_addr 127.0.0.1 tools/ddp_cpu_launch.py train.py <usual flags>` runs the real multi-process (gloo) code
paths on CPU: sharded sampling, DDP gradient sync, EMA under DDP, metric gathering, sharded evaluation.

## 9. Repository map (new / changed)

```
PLAN.md  RESULTS.md  README_MOTIP2.md  docs/MOTIP2.md          plan, run/gate log, this file
models/deim/            vendored DEIM + wrapper.py, builder.py (N/S/M/L/X), ema.py
models/motip/           id_decoder.py (fusion, RoPE, anchor), rope.py, id_criterion.py (Spatial ID Loss)
models/runtime_tracker.py   p_i buffer, N_init confirmation, RoPE alpha
data/id_switch.py       Spatial ID Switches + statistics       data/personpath22.py   PP22 loaders
data/transforms.py      zoom-out, IoU-crop, MOTIP2 pipeline    data/util.py           batch multi-scale
train.py                two-stage loop, EMA, param groups, DEIM step
configs/motip2/         30 generated configs + 2 smoke configs
tests/                  169 tests (+ frozen copies of the original decoder/tracker used as references)
```

## 10. Verification status (details and the full requirement → test map: [AUDIT.md](./AUDIT.md))

**169 automated tests** (CPU, `python -m pytest tests -q`; `-m "not slow"` skips the ~10 minutes of end-to-end/learning tests).
They are graded: **EXACT** (against the spec or the original MOTIP code, bit-for-bit), **BEHAVIOUR** (a trained model must show the
claimed effect, with a control that removes the mechanism and must fail) and **WIRING** (the flag reaches the module).

| Claim | Evidence |
|---|---|
| Decoder with `ID_FUSION=concat POS_ENC=learned_rel` and tracker with `N_INIT=1` ≡ original MOTIP | EXACT (frozen copies of the original code) |
| Spatial Anchor works | BEHAVIOUR: identically-looking objects are separated **98 %** with `p_i` vs **24 %** (chance 25 %) without |
| RoPE gives the decoder a usable temporal notion | BEHAVIOUR: **95–98 %** (5/5 inits) vs **25 %** with the rotation removed; the legacy learned bias is init-sensitive on the same task |
| Spatial ID Loss does what Eq. (1)–(2) say | EXACT values + BEHAVIOUR: expected localisation cost −10 %, errors land on nearer objects (costs accuracy at heavy noise) |
| Spatial ID Switches are implemented as in §3.2 | EXACT (permutations, σ→∞ ≡ uniform, uniform ≡ original code, live Tab. 8 statistics). **Their benefit is not demonstrable on a toy** — needs the real Tab. 5 ablation |
| Detection loss pipeline learns | BEHAVIOUR: DEIM-N overfits 4 images: best-match IoU **0.22 → 0.98**, confidence 0.02 → 0.96 |
| Tracking stack (buffer, ids, readout, confirmation, RoPE times, anchor) is train/inference consistent | BEHAVIOUR with a *trained* decoder + oracle detections, real `RuntimeTracker`, TrackEval: **HOTA 100** (n_init=1), 97.5 (n_init=2), 96.3 with 3-frame occlusions, 0 identity switches; untrained control **HOTA 71.6, 49 IDSW** |
| ONNX export | EXACT parity 2e-6 (ID decoder, dynamic T/N/M) and detector; exported graphs track identically inside the real tracker |
| Distributed + resume | real 2-process gloo DDP training + sharded eval + EMA + resume (found and fixed a deadlock and a CUDA-only reduction) |
| Vehicle data tool | converter GT equals generator truth to 0.01 px; strict pixel test (box filled and tight); TrackEval scores GT as a perfect tracker (HOTA 100) |
| Docs | every command in the docs parses against the real parsers (negative controls included) |

**Not verified (needs a GPU / real data / TensorRT):** MOTIP baseline reproduction (Gate 1), every paper number (Tab. 1–10), latency
(Tab. 4), TensorRT FP16 engines, detector quality on real benchmarks, the benefit of Spatial ID Switches, end-to-end accuracy
(an attempt to train from random weights on CPU was inconclusive: box loss fell but no detections reached the threshold, and ID
cross-entropy was still near chance — documented in `RESULTS.md`).

## 11. Known limitations / things to check first on a GPU

* No metric has been measured; treat the architecture as "faithful to the paper's text and unit-tested", not "reproduced".
* PersonPath22 folder layout is an assumption (see §4).
* Run a short GPU smoke first (`--epochs 1 --samples-per-epoch 800`) to get memory and iteration time before budgeting full runs.
* If DDP reports unused parameters: `DDP_FIND_UNUSED_PARAMETERS: true` in the config.
* The α mode (`multiply` vs `divide`) and the `p_i` source (`layer_input` vs `final_box`) are the two open modelling choices; both are flags and inexpensive to test.
* With `ID_LOSS_SPATIAL=False` the IoU-weighted CE (App. A) is kept; set `ID_CE_IOU_WEIGHT=False` for a pure-CE ablation.
