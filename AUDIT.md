# AUDIT.md — is everything in the specification implemented, and how do we know?

Every requirement of the skill file (`motip2-implementation`, §0–§15) and the paper (arXiv 2610.10391) is listed with where it is
implemented and **which test proves it**. "Proves" is graded honestly:

* **EXACT** — the test compares against the specification/original code bit-for-bit or against an analytic value.
* **BEHAVIOUR** — a trained model must show the claimed effect, with a control that removes the mechanism and must fail.
* **WIRING** — the flag/number reaches the module (a config or plumbing check). Necessary, not sufficient.
* **NOT VERIFIED** — needs real data, a GPU or TensorRT; nothing here claims it works.

Run everything: `python -m pytest tests -q` (209 tests, ~14 min on 4 CPU cores). `-m "not slow"` skips the detector-overfit and end-to-end/DDP tests (~5 min saved).

## A. Requirements → code → proof

| # | Requirement (source) | Implementation | Proof |
|---|---|---|---|
| 1 | DEIM detector N/S/M/L/X replaces Deformable DETR (§1.1) | `models/deim/*`, `models/motip/__init__.py` | EXACT param counts vs DEIM README (N 4M … X 62M); decoder dims/layers; `test_deim_wrapper.py` |
| 2 | Detector gives `e_i`, final boxes, `p_i` from its own positional MLP (§1.1) | `models/deim/dfine_decoder.py` (`out['outputs']`, `out['pos_embeds']`) | EXACT `p_i == query_pos_head(box entering last layer)`; lightweight == full forward; dynamic sizes |
| 3 | Fusion by addition, hidden dim halved, FFN ratio ×2 (§1.1) | `IDDecoder(id_fusion='add')` | EXACT input = `e_i + id_emb`; `model_dim` halved; BEHAVIOUR add ≥ concat on appearance task |
| 4 | RoPE replaces learned temporal bias; α at inference only (§1.1, §1.3) | `models/motip/rope.py`, `id_decoder.py` | EXACT relative-offset invariance; α inactive in training; BEHAVIOUR RoPE learns "trust the previous frame" from 5/5 inits, control (no rotation) at chance |
| 5 | `pos_enc=learned_rel` reproduces MOTIP (§5) | `id_decoder.py` legacy branch | EXACT bit-identical to the frozen original decoder (train and eval); BEHAVIOUR it can learn the same task |
| 6 | ID dictionary K+1, decoder mirrors DEIM width/depth (§1.1) | configs | WIRING `test_configs_and_data.py` (all N/S/M/L/X × 3 benchmarks + vehicles) |
| 7 | Detection loss weights, focal α/γ, matcher costs (§1.2) | `models/deim/builder.py` | EXACT `test_training_recipe.py::test_detection_loss_and_matcher_use_the_paper_weights` |
| 8 | ID loss = CE (IoU-weighted) + GIoU + L1 over `C_i`, `b̂` detached, newborn = own GT (§1.2, §8) | `models/motip/id_criterion.py` | EXACT (Σp=1 over `C_i`, newborn cost, detach, weights 1/2/5, torchvision GIoU); BEHAVIOUR spatial terms lower expected localisation cost and move errors to nearer objects (2 seeds) |
| 9 | σ=0.5, N_init=2, λ_ID=0.1, α=2, 800/1440 (§1.3) | configs | WIRING + doc-lint |
| 10 | Benchmark-dependent T=W, K, thresholds (§1.3) | `tools/gen_motip2_configs.py` | WIRING all 3 benchmarks |
| 11 | Two stages, AdamW, clip 0.1, EMA 0.9995 (+1000 warm-up joint, none in stage 1), constant LR, 500-iter warm-up (§1.4) | `train.py` | EXACT EMA formula incl. warm-up; EXACT warm-up values per param group; WIRING optimiser/clip |
| 12 | Per-size LR, backbone mult., ID-decoder ×2, no WD on norm/bias/embeddings (§1.4) | `get_param_groups_motip2` | EXACT every parameter's group (`test_pipeline.py`) |
| 13 | Clips of 30/60 frames, ONE interval ∈{1..4} per clip, 40k samples/epoch (§1.4) | `data/naive_sampler.py` | EXACT `test_clip_sampling_follows_the_paper` |
| 14 | Detection loss on 4 (8 for N) random frames, ID decoder gets whole clip (§1.4) | `deim_detr_forward` | EXACT frame/index ordering of all (b,t) frames for 0…all grad frames |
| 15 | Image augmentation (flip, zoom-out, IoU-crop, photometric, multi-scale 75–125 %) (§1.4) | `data/transforms.py`, `data/util.py` | BEHAVIOUR-lite: 120 random draws keep boxes valid/legal; crop/zoom shared by the clip; scales = DEIM `generate_scales` |
| 16 | Spatial ID Switches: GIoU + N(0,σ²) → Hungarian, p=1, σ→∞ ≡ uniform (§7) | `data/id_switch.py` | EXACT valid permutations; σ→∞ rate ≡ uniform; uniform bit-identical to original code; far pairs ≈ never swap; live Tab. 8 statistics in the training log. **Benefit NOT demonstrable on a toy** (see §C) |
| 17 | Random occlusion masking p=0.5, 6 groups × 100 IDs (§1.4) | unchanged MOTIP code + `NUM_TRAINING_IDS` | WIRING |
| 18 | Spatial Anchor: `p_i` added to every detection token, no new parameters (§6) | `id_decoder.py`, `train.py`, `runtime_tracker.py` | EXACT off == Phase-3 input, on == +`p_i`, no new params; BEHAVIOUR identical-looking objects: 98 % with anchor vs 24 % (chance 25 %) without |
| 19 | Online FIFO context of length T (§9.1) | `runtime_tracker.py` | EXACT buffer never exceeds T and is full in steady state |
| 20 | Greedy readout with λ_ID, unique identities (§9.2) | `object-max` protocol | EXACT uniqueness / known identities / threshold for object-max, id-max, Hungarian over 900 random score matrices |
| 21 | Tracklet confirmation N_init=2, unconfirmed discarded and slot freed (§9.4) | `runtime_tracker.py` (+ `N_INIT_TTL`) | EXACT 17 scenarios/properties incl. blip, occlusion, TTL, readout; **N_INIT=1 ≡ original tracker** on 6 random scenes; forced off in detector-only evaluation (bug found by the vehicle smoke test) |
| 22 | Latency tool, per-component breakdown, ratio vs MOTIP (§12) | `tools/benchmark_latency.py` | runs on CPU; paper numbers need a V100 — NOT VERIFIED |
| 23 | Per-variant configs (§10) | `configs/motip2/*` (40) | WIRING |
| 24 | Ablation list (a)–(i) as flag flips (§11) | `tools/run_ablations.sh` | doc-lint: every emitted command parses against the real parser and every flag maps to a config key |
| 25 | Seed aggregation, RESULTS.md logging (§11, §15) | `tools/aggregate_results.py`, `tools/log_result.py` | EXACT mean/std test; row insertion |
| 26 | Export: detector and ID decoder as separate ONNX; ID decoder dynamic track/detection counts (§13) | `tools/export_onnx.py`, `tools/onnx_runtime_model.py` | EXACT parity vs PyTorch (2e-6, 5 shapes incl. 1-frame context); exported graphs track identically inside the real tracker. **TensorRT FP16 NOT BUILT** |
| 27 | Distributed training/eval | `tools/ddp_cpu_launch.py` | 2-process gloo DDP stage 1 and stage 2 with all priors, EMA, sharded eval, resume (found + fixed a CUDA-only reduction) |
| 28 | Resume with EMA state | `train.py`, `models/misc.py` | epoch/step/EMA-update counters verified across a resume |
| 29 | Evaluation pipeline reproduces MOTIP (Gate 1, 64.0 HOTA) | `submit_and_evaluate.py` | **NOT VERIFIED** (needs GPU, data, checkpoint) |
| 30 | Paper numbers (Tab. 1–10) | — | **NOT VERIFIED**: no run on real benchmarks exists |
| 31 | Train on one or several tracking datasets (BDD100K, UA-DETRAC, VisDrone, UAVDT, FastTracker, CityFlow, KITTI, Waymo, nuScenes, AV2, 3D-box data) | `data/converters/*`, `data/unified_mot.py`, `data/class_maps.py` | EXACT each converter on a tiny dataset in the dataset's *native format* (boxes, classes, ids, splits, distractors/ignore, jpeg-resize rescaling); EXACT 3D projections vs an independent corner projection (translated/yawed ego, AV2 ego-motion); mutation-checked (dropping ego-motion, ego translation, near-plane clipping makes tests fail). **Real downloads NOT tested** (docs/DATASETS.md §8) |
| 32 | Exact dataset shares per epoch, per-dataset frame-interval cap | `data/naive_sampler.py` | EXACT shares (largest remainder), determinism, cycling, caps; default sampling bit-identical without the options |
| 33 | No classification loss for classes a dataset does not annotate | `models/deim/deim_criterion.py`, `data/joint_dataset.py` | EXACT zero gradient on masked columns / unchanged elsewhere; mask survives the real augmentation + collate; CPU end-to-end run on two converted datasets. **Benefit NOT measured** |

## B. Defects found by auditing/testing (all fixed, each with a regression test)

1. Detector-only evaluation produced **no output** with the MOTIP2 default `N_INIT=2` (would have broken Gate 2).
2. Inference applied ImageNet normalisation to a detector trained on [0,1] inputs.
3. Original tracker crashed when newborn detections exceeded free dictionary slots.
4. Logging used `device="cuda"` and `ReduceOp.AVG` (NCCL-only) — broke any CPU/gloo run; metrics first appearing after step 0 were never synced (also stale epoch averages).
5. torchvision private API renames broke colour jitter/photometric augmentation; a stray `triton` import broke CPU installs.
6. Tracker hard-coded 256-dim features (DEIM-N is 128).
7. Test-suite weaknesses removed: vacuous assertions (`or True`), a pixel check that could pass for any colour, a learning test whose outcome depended on an unseeded initialisation.

## C. What could NOT be demonstrated, and why

* **Spatial ID Switches improve robustness.** Two toy designs showed no benefit (none ≈ uniform ≈ spatial): even a decoder trained on clean
  histories pools all context frames and is already robust to the single-frame label swaps I could construct. Making the toy favourable would
  be fitting the test to the answer. The augmentation is verified exactly (see row 16); its benefit is the paper's Tab. 5 ablation on real data.
* **Spatial ID Loss costs accuracy when appearance cannot separate the identities** (toy, heavy noise: 89 % → 74 % at σ_pos=0.08): the newborn class has
  zero localisation cost, so with weights (1, 2, 5) the spatial terms can pull mass toward "newborn". Faithful to the paper's definition; on real data
  with informative embeddings it is reported as +1 HOTA. Watch `id_ce` vs `id_giou/id_l1` in the log.
* **RoPE vs learned bias:** on the toy RoPE learned from 5/5 initialisations in 400 steps, the learned bias from 2/5 in 600 and 4/5 in 1200 —
  the direction of the paper's Tab. 6, but a toy, not a replication.
* Everything needing a GPU, real data or TensorRT (rows 22, 29, 30).
