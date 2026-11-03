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
| 8 | Configs + smoke | not started | — | — |
| 9 | Eval + ablations | not started | — | — |

## 3. Environment (Phase 0 snapshot)

- Container: Linux, 4 CPU, 15 GB RAM, **no GPU**, no `torch`, no `datasets/`, no `pretrains/`; `pip` reachable.
- MOTIP base commit `e23387c`; DEIM reference `Intellindust-AI-Lab/DEIM@09d35d5` (read-only).
- Not yet known: user's GPU type/count, dataset locations, DEIM COCO checkpoint availability.

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
| D8 | 2026-11-03 | Gate 1's ±0.5 HOTA is tighter than MOTIP's documented ~1.5 retrain instability (`docs/GET_STARTED.md:61`); applied to the released checkpoint only. | Noted |
| P1 | 2026-11-03 | Paper inconsistency: §4.2 text says disabling priors moves DetA "by at most 0.2" but Tab. 1 shows MOTIP2-X DetA 84.3 → 83.0 (w/o priors). Not a blocker; use the table values. | Noted |
