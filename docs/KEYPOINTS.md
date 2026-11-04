# Per-detection keypoints in MOTIP2 (your mmdet D-FINE keypoint branch, ported) + review of that branch

## 1. What was added

The DEIM detector can predict, for every query, **K keypoints and K visibility logits** (your 31 vehicle keypoints; any K works).
Design = the one in your mmdet code (`DFINETransformerDecoder._decode_keypoints_inside_bbox`, `DFINEHead` keypoint losses):

* two MLPs on the final decoder query (the same detection token `e_i` the ID decoder reads): `xy_branch` → K local `(u, v)` = sigmoid(·),
  `vis_branch` → K logits. Depth = `KEYPOINT_HIDDEN_LAYERS` (4 = your `num_reg_fcs*2`; the converter reads it from your checkpoint).
* absolute position = `box_x1 + u·box_w`, `box_y1 + v·box_h` using **the same query's final box, detached** (clamped to the image), so
  keypoint losses cannot move boxes and ONNX/TensorRT need nothing special.
* **no second matching**: keypoints are supervised on the query that the detection Hungarian matching assigned to the object (final
  layer only, like your `loss_by_feat`). Losses (weights = yours by default, `KEYPOINT_LOSS_WEIGHTS: [10, 3, 35]`): L1 on labelled
  points, BCE on visibility, OKS (`1 − OKS`) in pixels. Instances without keypoint labels contribute exactly zero (graph kept for DDP).
* the tracker returns `keypoints` (n, K, 3) = x, y in the original frame's pixels + visibility probability for every confirmed track;
  `submit_and_evaluate.py` writes them to `<outputs>/keypoints/<seq>.txt`: `frame,id,class,score,x1,y1,p1,x2,y2,p2,…`.
  The ID decoder does not use keypoints (they are a per-frame output; no temporal modelling of them).
* ONNX: `detector.onnx` gets two more outputs `kpts` (1,Q,K,2) and `kpt_vis` (1,Q,K) — see docs/DEPLOY_ORIN.md.

Files: `models/deim/keypoints.py` (head + loss), `models/deim/wrapper.py`, `data/unified_mot.py` (`gt/gt_keypoints.txt`),
`data/transforms.py` (augmentation), `data/converters/coco_keypoints.py`, `tools/vehicles/convert_mmdet_deim.py`
(maps `kps_xy_branches.0` / `kps_vis_branches.0`), configs `configs/motip2/*_vehicles_kp.yaml` (`NUM_KEYPOINTS: 31`).

## 2. Data

Keypoint labels are needed to train the head. Two sources:

**(a) your COCO-style image data** (images + json with `keypoints`, as in your mmdet datasets) → still-image sequences. These behave like
CrowdHuman in MOTIP: one image = a 1-frame sequence repeated T times, no identities across frames.
```bash
python tools/prepare_dataset.py cocokp --images /data/vehicle_kp/images --train-json /data/vehicle_kp/train.json \
    --val-json /data/vehicle_kp/val.json --out ./datasets --image-mode jpeg --resize-short 800 --max-long 1440 --dry-run
python tools/prepare_dataset.py cocokp --images /data/vehicle_kp/images --train-json /data/vehicle_kp/train.json \
    --val-json /data/vehicle_kp/val.json --out ./datasets --image-mode jpeg --resize-short 800 --max-long 1440
```
→ `datasets/COCOKeypoints/{train,val}/img<id>/…`. Category names are mapped with `--class-map` (default maps car, bus, truck, motorcycle,
bicycle, person, van, vehicle). `iscrowd` → ignore region. Keypoint values `v` are kept **as in your file**.

**(b) tracking videos with keypoints**: add `gt/gt_keypoints.txt` to a sequence of the unified layout, one row per (frame, id):
`frame,id,x1,y1,v1,x2,y2,v2,…` in the sequence's pixels. Objects without a row are unlabelled. Datasets without the file need nothing.

Tell the loss what your `v` means (it must not depend on the batch, see §4):

| your labels | `KEYPOINT_LABELED_VALUES` (xy supervised) | `KEYPOINT_VISIBLE_VALUES` (visibility target 1) |
|---|---|---|
| COCO: 0 absent, 1 occluded, 2 visible (default) | `[1, 2]` | `[2]` |
| binary: 0 absent, 1 visible | `[1]` | `[1]` |

Also set `KEYPOINT_SIGMAS` (list of K OKS sigmas — copy them from your metafile `vehicle_31_kp_with_vis_different_joint_weights.py`; default 0.05 each) and, to use
horizontal-flip augmentation, `KEYPOINT_FLIP_PAIRS` (`[[left, right], …]`, from the same metafile). **Without the pairs the flip
augmentation is disabled** whenever `NUM_KEYPOINTS > 0` (a flip without swapping left/right keypoints would mislabel them); the
code refuses to flip keypoints without pairs.

## 3. Training and converting your detector

```bash
# your mmdet checkpoint -> MOTIP2 detector (keypoint branches are detected and converted; the report lists everything unmatched)
python tools/vehicles/convert_mmdet_deim.py --ckpt epoch_140.pth --size S --num-classes 5 --out pretrains/deim_s_vehicles_converted.pth
# one run on tracking videos + keypoint images (class masking, shares, frame-interval caps as in docs/DATASETS.md)
python tools/gen_mix_config.py --name veh_kp --size S --datasets Vehicles COCOKeypoints --mix 0.7 0.3 --num-keypoints 31
```
Then `accelerate launch … train.py --config-path configs/motip2/mixes/pretrain_veh_kp.yaml …` (stage 1), then stage 2 — exactly as in
docs/VEHICLES.md / docs/DATASETS.md. For `{s,m}_vehicles_kp.yaml` (tracking videos only) the keypoint losses are zero unless the
sequences have `gt_keypoints.txt`. **Consequence you should plan for:** if stage 2 sees no keypoint labels, the head is not
updated while the shared detection token `e_i` keeps changing (the ID loss and detection loss move it), so keypoint quality can
drift. Keep a share of keypoint-labelled images in the mix, or freeze nothing and check keypoint error on a labelled set after stage 2.
Whether keypoint training helps or hurts HOTA (shared `e_i`) is **not measured**.

## 4. Review of the mmdet code you pasted

I read it as pasted; I did not run it and I do not have `mmpose.OKSLoss`, `FillKeypoints`, your metafile or your RandomFlip. Findings,
most important first:

1. **Visibility target depends on the batch** (`loss_bioqual_and_keypoints`): `max_vis = labels_keypoints_vis.max(); if max_vis > 1:
   target = (v == 2) else: target = (v > 0)`. With COCO-style labels, a batch that contains any `v == 2` treats `v == 1` (occluded) as
   *not visible*, a batch without a `2` treats it as *visible*. The same keypoint gets different targets in different batches. Make the
   convention explicit (done here: `KEYPOINT_VISIBLE_VALUES`; tested in `tests/test_keypoints.py::test_label_semantics_do_not_depend_on_the_batch`).
2. **Horizontal flip and left/right keypoints**: `train_pipeline_aug` uses `RandomFlip(prob=0.5)`. If that is mmdet's stock `RandomFlip`, it
   mirrors keypoint coordinates but does **not** swap symmetric indices (left/right headlight, wheel, mirror). Then every flipped sample
   teaches the wrong side. I cannot see whether your custom pipeline swaps them (your metafile probably holds the flip indices) — please check.
3. **Reachability inside the predicted box**: keypoints are decoded inside the *predicted, image-clamped* box, but supervised against GT
   positions. A GT keypoint outside the predicted box (inaccurate box early in training, truncated vehicles, extrapolated occluded points) cannot
   be reached; the loss then pushes `u, v` to saturate at 0/1. Fine for points that are inside the object box by construction; worth a check
   of the fraction of your GT keypoints outside their GT box (if it is not ~0 you pay for it).
4. **"Not annotated" is trained as "not visible"**: the visibility BCE runs over all K points of every pose-labelled instance, so padded /
   unlabelled points (`FillKeypoints` to 31) get target 0. Reasonable if absent ≈ invisible; just be aware that the head cannot distinguish them.
   (Ported unchanged.)
5. **Auxiliary decoder layers have no keypoint loss** (only the last layer; the code comments call it debug isolation) although the
   keypoint branch is evaluated at every layer in training — wasted compute, not a bug. Ported as "last layer only".
6. **OKS**: I implemented the COCO definition `exp(−d² / (2 s² (2σ)²))`, `s²` = GT box area in pixels, mean over labelled points, loss `1 − OKS`. Your
   `mmpose.OKSLoss` may normalise differently (it also takes `areas`/`target_weight`), so the numeric scale of the OKS term can differ from
   your runs — re-tune `KEYPOINT_LOSS_WEIGHTS` (35 was tuned for your loss) if the OKS term dominates or vanishes.
7. Detection weights in your config (`loss_iou 10, loss_cls 3`) differ from the MOTIP2 paper's DEIM weights used here (GIoU 2, MAL 1);
   the keypoint weights (10 / 3 / 35) were kept as in your config, so their *relative* strength differs.

What is verified (tests/test_keypoints.py, test_export_onnx.py): decoding inside the box; the box receives no keypoint gradient; analytic OKS;
unlabelled points get no gradient; the head learns on a toy task and the control (lr 0) does not; criterion on the real detector end to end;
augmentation (flip with/without pairs, resize, zoom-out, IoU-crop with points leaving the crop, normalisation); loader and
COCO-image conversion; the mmdet-branch converter (exact tensor mapping); CPU training + tracking with keypoints (one keypoint row per
output box); ONNX with parity for S and M inside the tracker. **Not verified:** keypoint accuracy on real data, training stability
with the real 31-keypoint labels, your real checkpoint's key names (the converter matches `kps_xy_branches.0.<i>` / `kps_vis_branches.0.<i>`
as in the code you pasted; anything else is reported, not guessed), TensorRT.
