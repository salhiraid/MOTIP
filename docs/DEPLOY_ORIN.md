# MOTIP2 on a Jetson Orin: ONNX export (S and M) → TensorRT FP16 / INT8

What this repository gives you: the **ONNX files** (detector and ID decoder, separately), verified against PyTorch with onnxruntime
and inside the real tracker loop. What it does **not** give you, because there is no TensorRT, no Jetson and no GPU in the
development environment: TensorRT engines, INT8 calibration, latency, or accuracy after quantisation. Everything under "On the Orin"
below is therefore a procedure, not a result. Quantise on the Orin as you planned and check it with the HOTA procedure of §6.

## 1. Export

You need a trained **stage-2** checkpoint per size (it contains the EMA weights, the ID decoder and, with `*_vehicles_kp.yaml`, the
keypoint head). Configs: `configs/motip2/{s,m}_vehicles_kp.yaml` (31 keypoints; use `{s,m}_vehicles.yaml` for no keypoints).

```bash
CKPT_S=outputs/s_vehicles_kp/checkpoint_9.pth CKPT_M=outputs/m_vehicles_kp/checkpoint_9.pth HEIGHT=800 WIDTH=1440 \
    bash tools/export_orin.sh
```
which runs, per size, the single-size command
```bash
python tools/export_onnx.py --config configs/motip2/s_vehicles_kp.yaml --ckpt outputs/s_vehicles_kp/checkpoint_9.pth \
    --out-dir export/orin/s --height 800 --width 1440 --verify
```
Output per size: `detector.onnx`, `id_decoder.onnx`, `export_meta.json` (input/output names and shapes, thresholds, RoPE α,
the **operator list** of both graphs, the opset the file really declares, verification numbers). `--verify` compares onnxruntime
with PyTorch (detector set-wise, ID decoder at five shapes); the same checks pass in `tests/test_export_onnx.py` for S and M.
Use `--num-keypoints 0` to export a keypoint checkpoint without the keypoint outputs, `--only detector|id_decoder` for one graph.

The other sizes work the same way (`SIZES="N L X"`; N/L/X were not exported with real weights either).

## 2. Choosing HEIGHT × WIDTH (important)

The detector graph has a **fixed** input size (anchors and positional grids depend on it). The network was trained on frames that
are resized (shorter side `INFERENCE_MAX_SHORTER`=800, longer side ≤ `INFERENCE_MAX_LONGER`=1440, keep aspect) and then
**zero-padded to a multiple of 32**; box and keypoint outputs are normalised by the resized (unpadded) size and the tracker multiplies
them by the original frame size. So `HEIGHT × WIDTH` must be *your resized frame padded to a multiple of 32*, not an arbitrary canvas:

| camera | resize to (shorter 800) | pad → `--height --width` |
|---|---|---|
| 1920×1080 | 1422×800 | **800 × 1440** |
| 1280×720 | 1422×800 (up-scaled) or shorter 640 → 1138×640 | 800 × 1440, or **640 × 1152** |

A smaller resolution is much cheaper on the Orin, but the model was trained with multi-scale 75–125 % of 800 (600–1000 px
shorter side). Below that range (e.g. 480) accuracy was **not measured** — if you want it, train with `--aug-base-size` and
`--inference-max-shorter/--inference-max-longer` set to that size (the configs expose all of them) and export with the matching `HEIGHT/WIDTH`.
Frame pre-processing for the engine = exactly `data/seq_dataset.py`: RGB, `float32`, `x/255` in [0,1], **no mean/std normalisation**
(`IMAGE_NORMALIZE: false`).

## 3. What the two graphs contain and what stays outside

| | inputs | outputs |
|---|---|---|
| `detector.onnx` | `x` (1,3,H,W) | `logits` (1,300,C), `boxes` (1,300,4 cxcywh, normalised), `emb` (1,300,D) = detection token e_i, `pos` (1,300,D) = p_i (Spatial Anchor); with keypoints also `kpts` (1,300,K,2) normalised xy **inside the box of the same query**, `kpt_vis` (1,300,K) visibility **logits** |
| `id_decoder.onnx` | `traj_feat`(T,N,D) `traj_pos`(T,N,D) `traj_ids`(T,N) int64 `traj_times`(T,N) int64 `traj_masks`(T,N) bool `unk_feat`(M,D) `unk_pos`(M,D) `unk_masks`(M) bool `unk_times`(M) int64 | `logits` (M, K+1) (K+1 = newborn). RoPE α is baked in. Trajectory adapter included. |

D = 256 for S and M (128 for N), C = 5 classes, K(ID vocabulary) = `NUM_ID_VOCABULARY` of the config, keypoints = 31.
**Outside the engines** (a few lines each; reference = `models/runtime_tracker.py`, which already runs the ONNX graphs through
`tools/onnx_runtime_model.py`): score = max sigmoid(logits), class = argmax, `DET_THRESH` and area filters; the FIFO context buffer
of length T=`MISS_TOLERANCE`; building the ID-decoder inputs; the greedy `object-max` readout with `ID_THRESH`; newborn handling with
`NEWBORN_THRESH`; tracklet confirmation `N_INIT`; keypoints of a track = `kpts` of its query × the frame size, visibility = sigmoid(`kpt_vis`).
The thresholds are in `export_meta.json["inference"]`.

## 4. On the Orin: TensorRT (procedure, not verified here)

**Version.** Both files declare ONNX **opset 18** (the PyTorch exporter raises it when an operator needs it; asking for 17 still gives 18)
and use `GridSample`, `TopK`, `LayerNormalization`, `Resize`, `GatherElements`, `Softmax`, … (full list in `export_meta.json`). Use a
TensorRT whose ONNX parser supports opset 18 and those operators (JetPack 6.x); on older JetPacks `trtexec` will tell you
which node it rejects. Always run the parser first: `trtexec --onnx=detector.onnx --skipInference`.

**FP16**
```bash
trtexec --onnx=export/orin/s/detector.onnx --fp16 --saveEngine=detector_s_fp16.engine
trtexec --onnx=export/orin/s/id_decoder.onnx --fp16 --saveEngine=id_decoder_s_fp16.engine \
  --minShapes=traj_feat:1x1x256,traj_pos:1x1x256,traj_ids:1x1,traj_times:1x1,traj_masks:1x1,unk_feat:1x256,unk_pos:1x256,unk_masks:1,unk_times:1 \
  --optShapes=traj_feat:30x30x256,traj_pos:30x30x256,traj_ids:30x30,traj_times:30x30,traj_masks:30x30,unk_feat:20x256,unk_pos:20x256,unk_masks:20,unk_times:20 \
  --maxShapes=traj_feat:64x300x256,traj_pos:64x300x256,traj_ids:64x300,traj_times:64x300,traj_masks:64x300,unk_feat:300x256,unk_pos:300x256,unk_masks:300,unk_times:300
```
The context length is 1 on the second frame of a sequence (onnxruntime accepts T=1; if your TensorRT build refuses a
minimum of 1, use 2 and treat that single frame specially). `int64`/`bool` inputs: TensorRT narrows `int64` to `int32` (a warning) and
supports `bool`; if your version rejects either, cast in the runtime and tell me — a cast wrapper is a two-line change in `tools/export_onnx.py`.

**INT8** — not run, so only what is certain and what is advice:
* Calibrate with **real frames from the target cameras** (several hundred), pre-processed exactly as in §2. The ID decoder needs realistic
  *tracks* (e_i, p_i, ids) as calibration input, not random tensors: record them from a run of the FP16 tracker.
* Advice (typical for transformer detectors, **not measured on this model**): keep the parts that are sensitive to precision in FP16 —
  the deformable-attention sampling and softmax, the box-distribution decoding (`integral`, LQE) and the whole ID decoder — and quantise
  the backbone/encoder convolutions first; use per-layer precision constraints in the builder. Quantise the keypoint MLPs last.
* The only valid acceptance test is §6: engines inside the tracker on your `val` split vs the FP32 ONNX numbers.

Quickest path while you evaluate: `OnnxMOTIP("export/orin/s", providers=("TensorrtExecutionProvider", "CUDAExecutionProvider"))`
(`tools/onnx_runtime_model.py`) runs the same graphs through onnxruntime's TensorRT provider (FP16/INT8 via its provider options) —
convenient for accuracy checks, not necessarily the fastest deployment.

## 5. Sizes

S and M differ only in the backbone (HGNetV2-B0 / B2), encoder width/depth and 3 vs 4 decoder layers; the ID decoder mirrors the DEIM
decoder (D=256, 3 / 4 layers). Parameter counts follow the DEIM README (S ≈ 10 M, M ≈ 19 M, detector only); latency on the Orin is
**not measured** (`tools/benchmark_latency.py` measures PyTorch components on whatever device you run it on).

## 6. Accuracy check after any conversion

```bash
python tools/eval_datasets.py --config configs/motip2/s_vehicles_kp.yaml --model outputs/s_vehicles_kp/checkpoint_9.pth \
    --datasets Vehicles:val --outputs-dir outputs/s_vehicles_kp
```
Then run the same `val` sequences through the exported/quantised pipeline, write the tracker files in MOT format and score them with
TrackEval (`TrackEval/scripts/run_mot_challenge.py`, `--GT_FOLDER datasets/Vehicles/val`). Compare HOTA / DetA / AssA / IDSW. For the
keypoints compare the mean pixel error on a labelled set between PyTorch and the engines (the output file format is in docs/KEYPOINTS.md).
