# RESULTS.md — MOTIP2 reproduction log

Protocol (skill §15): every run gets a row. Never claim a match to the paper without a run; report seed, config, checkpoint. Tolerances (§12): baseline ±0.5 HOTA; main ±1.0 HOTA/IDF1; ablations direction-only. When a gate fails: record the failing test and measured value here, list ≤ 2 hypotheses, then ask before changing code or shared defaults.

Paper reference numbers live in `PLAN.md` §7. Latency: report the ratio S/MOTIP when not on a V100.

## 1. Run log

| Date | Git commit | Config | Seed | Benchmark | Split | HOTA | DetA | AssA | MOTA | IDF1 | IDSW | Latency (ms) | Checkpoint | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| — | — | — | — | — | — | — | — | — | — | — | — | — | — | No runs yet (Phase 0 only). |

## 2. Gate status

| Gate | Phase | Status | Evidence / measured value | Date |
|---|---|---|---|---|
| 0 | Recon + PLAN.md | **Approved by user** ("start implementing everything, you have my validation"); ⚠ decisions taken as recommended, α default `multiply` + A/B at Gate 3 | `PLAN.md` incl. skill-§1 traceability matrix | 2026-11-03 |
| 1 | Baseline MOTIP (DanceTrack val, target 64.0 ±0.5) | **DEFERRED (cannot run here)** | Tooling done (IDSW capture, `tools/log_result.py`); eval needs GPU + datasets + released checkpoint. Work proceeds on user instruction ("start implementing everything"); documented deviation from §0.3 — must still be run before any metric claim. | 2026-11-03 |
| 2 | DEIM detector (DetA recorded, latency recorded) | **unit PASS / metric pending** | `tests/test_deim_wrapper.py` 12/12 (params N/S/M/L/X within 20% of DEIM README 4/10/19/31/62M; p_i = query_pos_head of the box entering the last layer; lightweight==full; dynamic sizes; COCO-head slicing). CPU smoke on synthetic data: stage-1 + stage-2 train, EMA save/load, tracker, TrackEval all run. DetA on real DanceTrack val + latency need GPU/data. | 2026-11-03 |
| 3 | ID decoder (unit + α sweep) | **unit PASS / metric pending** | `tests/test_rope.py`, `tests/test_id_decoder.py` (RoPE relative-offset invariance; additive fusion; ID_FUSION=concat+POS_ENC=learned_rel bit-identical to the frozen original MOTIP decoder in train and eval mode; masks/causality; α inference-only). α sweep (both modes) needs a trained PP22/DanceTrack MOTIP2-S + GPU. | 2026-11-03 |
| 4 | Spatial Anchor | **unit PASS / metric pending** | `tests/test_spatial_anchor.py`: anchor off == Phase-3 input bit-for-bit; on == Phase-3 input + p_i exactly (trajectory and unknown tokens); no new parameters; gradient reaches p_i; p_i width == decoder width for N/S. Metric (64.5 vs 64.0 DanceTrack val) pending. | 2026-11-03 |
| 5 | Spatial ID Switches (unit + Tab. 8 stats) | **unit PASS / Tab. 8 stats pending real data** | `tests/test_spatial_idsw.py` (15): valid permutation; sigma=0 identity; sigma=1e6 swap rate == uniform (40.2% vs 40.6%, meanGIoU 0.05 vs 0.05 on a synthetic scene with 10 objects); sigma=0.5 swaps 19% with LR 0.8% / meanGIoU 0.70; far-apart pairs swap 0.1% (uniform 40.6%); uniform mode bit-identical to MOTIP's original code; `skip` never touches absent slots; `last_frame` applies one pi to the whole history. `tools/idsw_stats.py` ready; Tab. 8 targets (DanceTrack 10/26/0.20 vs 37/85/-0.40) need the real dataset. | 2026-11-03 |
| 6 | Spatial ID Loss | **unit PASS / metric pending** | `tests/test_spatial_id_loss.py` (9): probabilities sum to 1 over C_i (and equal a renormalisation of the full softmax); newborn class costs only its own localisation error; wrong-far > wrong-near while CE cannot tell them apart; b_hat detached (no grad reaches pred boxes, also for the IoU weight); total = 1*CE + 2*GIoU + 5*L1; spatial off == original CE; box format == torchvision GIoU/IoU; no NaN with absent/padded tokens. Smoke: id_loss 5.29 = 0.705 + 2*1.305 + 5*0.396. Metric (63.5 vs 64.5) pending. | 2026-11-03 |
| 7 | Inference / N_init | **unit PASS / metric pending** | `tests/test_tracker_confirmation.py` (10 tests): N_init=2 withholds the first frame, drops blips and frees slots/context, confirmed tracks survive occlusion; **N_INIT=1 reproduces the frozen original tracker exactly on 6 random scripted scenes**. Tab. 9 IDSW sweep needs a trained model. | 2026-11-03 |
| 8 | Configs + smoke | **CPU part PASS / GPU smoke pending** | `tools/gen_motip2_configs.py` writes 30 configs (N/S/M/L/X × DanceTrack/SportsMOT/PP22 × stage 1/2); `tests/test_configs_and_data.py` asserts every §1.3/§1.4 number (LR, backbone x, ID-dec x2, WD, T=W, K, thresholds, sigma, N_init, lambda_ID, alpha, EMA, warmups, batch, groups, DEIM-mirrored decoder dims) and that each of the six toggles reaches its module; `tests/test_pipeline.py` (augmentation validity, batch multi-scale, SeqDataset normalisation, **frame/index ordering of the DEIM step**, param groups). CPU smoke on a synthetic DanceTrack-format set: stage-1 and stage-2 (all priors on) train, EMA ckpt, tracker (anchor + N_init + alpha), TrackEval run; id_loss decomposes as 1*CE+2*GIoU+5*L1. **Still needed:** a GPU smoke (loss decreasing over ~100 its, memory, iteration time -> compute budget estimate). Full suite: 106 passed. | 2026-11-03 |
| 9 | Eval + ablations | not started — **needs your compute-budget confirmation** | `tools/run_ablations.sh` (dry-run prints every command for ablations (a)-(i)), `tools/aggregate_results.py` (mean±std over seeds). No full training run has been started. | — |

## 3. Environment (Phase 0 snapshot)

- Container: Linux, 4 CPU, 15 GB RAM, **no GPU**, no `torch`, no `datasets/`, no `pretrains/`; `pip` reachable.
- MOTIP base commit `e23387c`; DEIM reference `Intellindust-AI-Lab/DEIM@09d35d5` (read-only).
- Not yet known: user's GPU type/count, dataset locations, DEIM COCO checkpoint availability.
- Dev environment for the unit tests: CPU torch 2.15 installed from PyPI; 106 tests pass (`python -m pytest tests -q`).

## 4. Deviations, open decisions, and gaps (append-only)

| # | Date | Item | Status |
|---|---|---|---|
| D1 | 2026-11-03 | Skill ambiguity #6 default (IDSW reference = last frame) conflicts with MOTIP code (`data/transforms.py:519-529`, per-frame) and Paper §3.2 ("in each frame"). Plan uses per-frame; last-frame kept as flag. | Awaiting approval |
| D2 | 2026-11-03 | Skill ambiguity #8 default (positions × α at inference) may be inverted; plan implements both modes and decides by an inference-only A/B at Gate 3. | Awaiting approval |
| D3 | 2026-11-03 | DEIM's `query_pos_head` embeds the box *entering* the last decoder layer, not the final output box; plan N1 picks `layer_input` with `final_box` as ablation. | Awaiting approval |
| D4 | 2026-11-03 | Weight decay per size taken from DEIM (1e-4 / 1.25e-4) instead of skill default (MOTIP 5e-4); stage-1 length = 20 epochs instead of "DEIM default" (COCO-scale). | Awaiting approval |
| D5 | 2026-11-03 | Repo has no PersonPath22 or MOT17 training loader (`data/joint_dataset.py:17-22`); PP22 gates blocked until `data/personpath22.py` exists. | Planned (Phase 8) |
| D6 | 2026-11-03 | MOTIP's `AUG_TRAJECTORY_SWITCH_PROB=0.5` is the per-object inclusion probability, not the augmentation probability; "p = 1.0" in the paper read as "always on". | Clarification, plan decision #7 |
| D7 | 2026-11-03 | DEIM expects un-normalised [0,1] inputs and static `eval_spatial_size`; MOTIP normalises with ImageNet stats. Must be handled in Phase 2 (N5). | Planned |
| D9 | 2026-11-03 | PLAN correction: DEIM-X *decoder* hidden dim is 256 (only its encoder is 384), so ID-decoder dims are N=128, S/M/L/X=256. | Fixed in PLAN |
| D10 | 2026-11-03 | Fixed latent portability bugs found by smoke tests: stray `triton` import in data/transforms.py; torchvision renamed `_get_params/_transform` (colour jitter / photometric were broken on current torchvision); `.cuda()` / `cuda.synchronize` assumptions; tracker hard-coded 256-dim features; `INFERENCE_MAX_SHORTER` was not configurable. | Done |
| D11 | 2026-11-03 | Latent bug in MOTIP's original tracker: when #newborn detections > #free dictionary slots and not all detections are newborn, `newborn_keep_idxs` was sized by all detections -> RuntimeError. Fixed (sized by #newborn); only changes a previously-crashing case. | Done |
| D12 | 2026-11-03 | Found while reviewing the inference path: `SeqDataset` always applied ImageNet normalisation; DEIM expects [0,1] inputs. Now `IMAGE_NORMALIZE` controls both training and inference (legacy default unchanged). | Done |
| D13 | 2026-11-03 | PersonPath22 loaders assume MOT-style folders (see `data/personpath22.py`); the real dataset layout/conversion is unverified. | Open (needs your data) |
| D14 | 2026-11-03 | ID_LOSS_SPATIAL=False keeps the IoU-weighted CE (App. A) and drops only the two spatial terms (skill §8 says "only L_CE"; the IoU weight is a CE detail, toggle with `ID_CE_IOU_WEIGHT`). | Noted |
| D15 | 2026-11-03 | Bug found by the vehicle smoke test: in detector-only evaluation (Gate 2) every detection is newborn each frame, so `N_INIT=2` suppressed ALL output. Tracker now forces `n_init=1` when `only_detr`; test added. | Fixed |
| D16 | 2026-11-03 | Vehicle workflow added (docs/VEHICLES.md): CVAT video XML preprocessing with site-disjoint split, 5-class DEIM head, mmdet-checkpoint converter. Verified only on synthetic videos (125 tests + CPU end-to-end smoke); real data / checkpoint / GPU not yet exercised. Class mapping (van, car_with_trailer -> car; all trucks -> truck) is an assumption to confirm. | Open (needs your data) |
| D8 | 2026-11-03 | Gate 1's ±0.5 HOTA is tighter than MOTIP's documented ~1.5 retrain instability (`docs/GET_STARTED.md:61`); applied to the released checkpoint only. | Noted |
| P1 | 2026-11-03 | Paper inconsistency: §4.2 text says disabling priors moves DetA "by at most 0.2" but Tab. 1 shows MOTIP2-X DetA 84.3 → 83.0 (w/o priors). Not a blocker; use the table values. | Noted |

## 5. Verification campaign (synthetic data, CPU) — what was measured

Everything here ran on this CPU-only machine with synthetic data: it demonstrates that **mechanisms work and code paths are consistent**,
not benchmark accuracy. Method details and the full requirement→test map: `AUDIT.md`.

| Experiment | Result |
|---|---|
| Spatial Anchor, objects with IDENTICAL appearance (toy, trained ID decoder) | **98.1 %** with anchor vs **24.2 %** without (chance 25 %); distinct appearance without anchor 99.8 % |
| Temporal encoding, older history carries consistently wrong labels (toy) | RoPE **95–98 %** (5/5 inits, 400 steps); no-rotation control **25 %** (chance); legacy learned bias 99 % but only 2/5 inits in 600 steps (4/5 in 1200) |
| Additive fusion at half width vs concatenation (toy) | both > 97 %, additive ≥ concat − 3 pts |
| Spatial ID Loss (toy, noisy positions, 2 seeds) | expected localisation cost −10 %, wrong predictions land on nearer objects; at heavy noise accuracy drops 89→74 % (newborn class has zero spatial cost) |
| Spatial ID Switches benefit (toy) | **not demonstrable**: none ≈ uniform ≈ spatial in two toy designs (clean-trained decoder already pools all context frames). Verified exactly instead (permutations, σ→∞ ≡ uniform, uniform ≡ original code, live Tab. 8 statistics) |
| Detector, DEIM-N from random init, 300 steps on 4 images, real criterion | best-match IoU **0.215 → 0.978**, matched-query score **0.017 → 0.957** |
| Tracking stack with a TRAINED decoder + oracle detections, real `RuntimeTracker`, TrackEval | **HOTA 100 / IDF1 100 / IDSW 0** (n_init=1); **HOTA 97.5** (n_init=2, first frame of each track withheld); **HOTA 96.3, IDSW 0** with two objects undetected for 3 frames; control with UNTRAINED decoder: **HOTA 71.6, AssA 51.3, IDSW 49** |
| ONNX export (S/N size, trained checkpoint) | detector: 100 % of boxes matched, max diff 3e-6; ID decoder: 2e-6 at shapes (1×3×2)…(20×30×12) incl. 1-frame context; exported graphs inside the real tracker give the same boxes/identities as PyTorch |
| 2-process CPU DDP (gloo), stage 1 and stage 2 with all priors, EMA, sharded eval, resume | passes (found/fixed a CUDA-only reduction in the logger) |
| End-to-end learning from RANDOM weights on synthetic videos (500 detector + 160 joint iterations) | **inconclusive, and reported as such**: box loss fell 3.6→0.5 but scores never reached the 0.3 threshold (0 detections), ID cross-entropy 2.5 vs chance 3.9. Too little compute for a from-scratch detector; pretrained weights could not be downloaded here |

### Defects found by this campaign (all fixed + regression tests)
D15 detector-only eval emitted nothing with `N_INIT=2` · D17 logger hard-coded `cuda` / NCCL-only reduce, metrics first seen after step 0 never synced ·
D18 synthetic-data bugs found by strict tests (colour contrast, stripe precedence, nonlinear motion) — converter itself was correct (GT equals generator truth to 0.01 px).

## 6. Multi-dataset training (public tracking datasets) — what was measured
Added: unified layout/loader (`data/unified_mot.py`), 11 converters (`data/converters/`, CLI `tools/prepare_dataset.py`), exact per-dataset epoch
shares + per-dataset frame-interval caps (`NaiveSampler`), per-dataset classification-loss masking (`CLASS_MASKING`), `tools/gen_mix_config.py`,
`tools/eval_datasets.py`, `docs/DATASETS.md`. Measured on SYNTHETIC data only (no accuracy numbers exist for any mixture):

| Check | Result |
|---|---|
| Every converter on a tiny dataset in the dataset's native format → loader | boxes within 0.02 px, unified classes, constant ids, splits, distractor/ignore rows |
| 3D projections (nuScenes, AV2, generic) vs independent corner projection | max error < 0.05 px; translated + yawed ego; AV2 moving ego between lidar and camera time |
| Mutation checks (drop AV2 ego-motion / nuScenes ego translation / near-plane clipping) | each makes the tests fail |
| Mixed sampling | exact shares (600/300/100 of 1000; 33/33/34 of 100), deterministic per epoch, small pools fully cycled, interval caps respected |
| Class masking | zero gradient on masked columns, unchanged on annotated ones, bit-identical without masks; survives augmentation + collate |
| CPU end-to-end (train + in-loop eval on 2 converted datasets) | runs, finite losses |
Open: converters on real downloads; FastTracker class ids; UAVDT test list; Waymo v2 column names; I-24 3D reader (none); effect of any mixture on HOTA.

## 7. Keypoints and Orin export — what was measured (synthetic / CPU only)
| Check | Result |
|---|---|
| Keypoints decoded inside the same query's box; box gets no keypoint gradient | exact |
| OKS loss vs the COCO formula (analytic, pixel units, per-keypoint sigmas) | exact (1e-6) |
| Visibility target independent of the batch contents (the mmdet code's is not) | tested |
| Toy learning: head trained with the keypoint loss vs lr = 0 control | error 0.35x lower or better (and < 6 px on 100 px) |
| Real detector + criterion: keypoint losses on matched queries, gradients reach the head | yes |
| Augmentation (flip + pairs, resize, zoom-out, IoU crop with points leaving the crop, normalise) | exact geometry |
| CPU train (keypoint images + tracking) and tracking: one keypoint row per output box | yes |
| mmdet `kps_xy_branches.0` / `kps_vis_branches.0` converter | exact tensor mapping; inconsistent branches rejected |
| ONNX S and M with 31 keypoints, detector + ID decoder | parity < 1e-3; opset in file = 18 (requested 17 is raised by the exporter, now reported); ORT-in-tracker == PyTorch |
Open: keypoint accuracy and its effect on HOTA on real data; TensorRT FP16/INT8 engines and latency on the Orin; the user's metafile values (sigmas, flip pairs).
