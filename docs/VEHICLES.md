# Vehicle tracking with MOTIP2 — your dataset, step by step

Goal: train and evaluate MOTIP2 for **vehicle tracking** using
* your **1200 annotated videos** (CSV index + CVAT *video* XML tracks), and
* your **DEIM-S detector** trained in the mmdet framework (initialisation of the detector).

Classes of the detector: `('bicycle', 'car', 'motorcycle', 'bus', 'truck')`. Identity (tracking) is **class-agnostic**,
as in MOTIP; HOTA/IDF1/MOTA are computed class-agnostically over all vehicles.

> **Status.** Everything below was implemented and tested on CPU with *synthetic* videos + CVAT XML (real `mp4` decoding with ffmpeg,
> real TrackEval, both pretrain routes, full training loop). It has **not** been run on your real data, on a GPU, or on your real
> checkpoint. Steps 3 and 5 exist precisely to catch problems on your side early.

```
videos + CVAT xml + csv ──(1,2) prepare──▶ datasets/Vehicles/{train,val,valmini}/<seq>/img1,gt
mmdet DEIM-S checkpoint ──(3) convert───▶ pretrains/deim_s_vehicles_converted.pth ──(5) check──▶ (6) stage 1 ──▶ (7) stage 2 ──▶ (8) eval
```

---

## 0. Requirements

```bash
pip install torch torchvision einops scipy pyyaml pillow accelerate wandb pytest pandas
sudo apt-get install ffmpeg            # used for decoding; no OpenCV needed
python -m pytest tests -q              # 125 tests, CPU, ~1 min  (sanity of your install)
```

## 1. Preprocess: dry run (reads only the CSV + XML, writes nothing)

```bash
python tools/vehicles/prepare_vehicle_dataset.py \
    --csv /data/dataset.csv --videos-dir /data/videos --anno-dir /data/annotations \
    --out ./datasets --dry-run --workers 16
```
* The video of row `id` is `--videos-dir/<Video_file>`; if that column does not resolve, `--videos-dir/<id>.mp4` (also `.avi/.mov/.mkv...`).
  The annotation is `--anno-dir/<Anno_file>` or `<id>.xml`. If everything is in one folder, omit `--anno-dir`.
* It prints: videos found/missing, **train/val counts, sites per split, countries**, boxes per class, objects per frame, **#frames and
  disk estimate**, and the recommended **ID dictionary size K**. Read it before the real run.

### How the split works (`--val-frac 0.12`, default)
* Videos are grouped by **(Country, Site)**. A site is moved to validation **as a whole**, so *no site is ever in both splits*
  (asserted, and written to `split.csv`). The split is stratified by country (each country contributes ≈12 % of its videos to
  validation when it has ≥ 2 sites; a country with a single site stays in train, and a country is never entirely in validation).
  `--seed` changes the draw. Change `--val-frac` for a bigger/smaller validation set.
* `valmini` = `--valmini 20` validation videos spread over as many sites as possible (symlinks into `val/`); used for the *cheap evaluation
  after every training epoch*. Final numbers always use the full `val`.
* Different sites of the same camera/intersection spelled differently in the CSV would leak — if you know of such cases, merge their
  `Site` strings in the CSV before running.

### Class mapping (please confirm — it is an assumption)
| CVAT label | → class |
|---|---|
| `bicycle` | bicycle |
| `car`, `car_with_trailer`, `van`, `van_with_trailer` | car |
| `motorcycle` | motorcycle |
| `bus`, `bus_with_trailer` | bus |
| `light_truck`, `heavy_truck`, `truck_without_trailer` | truck |
| `ignore_region` | ignore region |

If your detector pretraining used another grouping (e.g. `van` → truck), pass `--class-map map.json` with only the changed entries,
e.g. `{"van": 4, "van_with_trailer": 4}` (indices: bicycle 0, car 1, motorcycle 2, bus 3, truck 4). Labels not in the map are dropped
with a warning. **Ignore regions** are written as TrackEval "distractor" rows: tracker boxes inside them are not counted as false
positives; they are not used for training.

### What the XML conversion does
* Uses `<track>/<box>`; boxes with `outside="1"` end a track; the **annotated frame range** of each video is trimmed (first→last box) and
  frames are renumbered from 1. If an export contains only *keyframes*, gaps between two visible boxes are linearly interpolated
  (CVAT semantics, a no-op for full exports). `occluded` boxes are kept (visibility 0.5).
* Track id → object id (`id+1`). Boxes are scaled to the stored resolution and clipped to the image; boxes < 2 px are dropped.
* Videos with < `--min-frames` (30) annotated frames are skipped.

## 2. Preprocess: real run

```bash
python tools/vehicles/prepare_vehicle_dataset.py --csv /data/dataset.csv --videos-dir /data/videos --anno-dir /data/annotations \
    --out ./datasets --val-frac 0.12 --resize-short 800 --max-long 1440 --workers 16
```
* Frames are **stored already resized** (shorter side 800, longer ≤ 1440 = the paper's inference resolution; 1920×1080 → 1422×800),
  so training/inference do not re-decode full-HD video. `--resize-short 0` keeps the original size. If your own detector works better at a
  different scale use e.g. `--resize-short 896 --max-long 1536` **and** the matching flags in step 7 (`--aug-base-size 896 --aug-max-size 1536
  --inference-max-shorter 896 --inference-max-longer 1536`).
* Disk: roughly 0.12 MB/frame at 800×1422 (1200 videos × ~270 frames ≈ 40 GB). `--stride 2` halves it but changes the time scale (the
  context window then covers twice as much real time) — keep 1 unless you must.
* It is resumable (`.done` marker per sequence); `--overwrite` redoes everything; `--skip-extract` only rewrites annotations.
* Output: `datasets/Vehicles/{train,val,valmini}/<seq>/{img1,gt/gt.txt,gt/gt_classes.txt,seqinfo.ini}`, `*_seqmap.txt`, `split.csv`,
  `prepare_report.json` (warnings per video, skipped videos, statistics). **Read `warnings` and `skipped` in the report.**
  Sequence name = `v<id>`.
* `gt.txt` (for TrackEval) has class 1 for every vehicle; `gt_classes.txt` (for detector training) has the real class 0..4.

### Check it visually (do this!)
```bash
python tools/vehicles/visualize_gt.py --seq-dir datasets/Vehicles/train/v1001 --frames 1 60 120 --out /tmp/vis
```
Boxes must sit on the vehicles, ids must be stable over time, classes plausible. Also check `prepare_report.json` → `stats`:
`recommended_NUM_ID_VOCABULARY` (K), objects per frame, fps.

## 3. Convert your mmdet DEIM-S checkpoint

The mmdet port names modules differently from official DEIM, and I could not see your checkpoint, so the converter **aligns tensors**
(same shape bucket + name similarity + order, solved with the Hungarian algorithm, with a second pass that learns which source module
prefix corresponds to which target module) and **refuses to write** the result unless *every* target parameter is matched.

```bash
# 3a. dump both key lists (keep this file; send it to me if anything below is refused)
python tools/vehicles/convert_mmdet_deim.py --ckpt work_dirs/xxx/epoch_140.pth --size S --num-classes 5 --inspect inspect.txt
# 3b. convert
python tools/vehicles/convert_mmdet_deim.py --ckpt work_dirs/xxx/epoch_140.pth --size S --num-classes 5 \
    --out pretrains/deim_s_vehicles_converted.pth
```
* Which weights: the default reads `state_dict`. mmengine's EMA hook may store the EMA weights in `state_dict` and the raw ones in
  `ema_state_dict` (or the opposite depending on the version) — the tool prints the sections it finds; choose with `--which ema_state_dict`.
* **Class order**: the converter assumes the source head has 5 classes in the order `('bicycle','car','motorcycle','bus','truck')`.
  If the head has more classes: `--src-num-classes 8 --class-indices 1 2 3 5 7` (source rows → our classes).
* Keypoint heads (31 keypoints, OKS) and other tensors that do not exist in the official model are dropped (listed in the report).
* Output: `pretrains/deim_s_vehicles_converted.pth` + `…convert_report.txt` (full mapping, ambiguous matches with their score margins).
  **Ambiguous matches > 0 is normal** (many identically-shaped tensors, e.g. LayerNorm weights of several layers); that is what step 5 verifies.
* The model must be the same architecture as official DEIM-S: HGNetV2-B0 (`use_lab`), 3 decoder layers, hidden 256, `num_points [3,6,3]`,
  `reg_max 32`, 300 queries, `ChannelMapper` ≙ `input_proj`. Your config matches. If your head layout differs (e.g. separate q/k/v
  projections), the converter reports unmatched tensors and stops.

## 4. Configs (generated; edit by flags)

`configs/motip2/pretrain_s_vehicles.yaml` (stage 1) and `configs/motip2/s_vehicles.yaml` (stage 2) were generated by
`tools/gen_motip2_configs.py` from the paper's numbers. Things **specific to your data** to decide (flags work on both train & eval):

| What | Flag | Default | How to choose |
|---|---|---|---|
| ID dictionary size K | `--num-id-vocabulary` | 50 | ≥ the `recommended_NUM_ID_VOCABULARY` printed by the preprocessing (p99 unique ids per 30-frame window × 2.5). Busy junctions → 100/150/300 |
| Context window T=W | `--sample-lengths 30 --rel-pe-length 30 --miss-tolerance 30` | 30 | frames (not seconds): 30 ≙ 1.2 s at 25 fps. Use the three flags together; RoPE lets you test a longer `--miss-tolerance` at inference |
| Detection / newborn thresholds | `--det-thresh --newborn-thresh` | 0.3 / 0.7 | tune on `valmini` (inference-only, cheap) |
| Resolution | see step 2 | 800/1440 | training: `--aug-base-size --aug-max-size`; inference: `--inference-max-shorter --inference-max-longer` |
| Clips per GPU | `--batch-size` | 2 | paper global batch = 8 clips × 30 frames; lower per-GPU batch if out of memory, add GPUs to keep 8 |
| Epochs / epoch length | `--epochs --samples-per-epoch` | 10 / 40000 | |
| Detection frames per clip | `--detr-num-train-frames` | 4 | frames with detection loss; memory knob |

## 5. Verify the converted detector (detection-only evaluation) — mandatory

This evaluates only the detector on the **val** split (every box above `NEWBORN_THRESH` is a "track" of length 1; DetA is the number to read):

```bash
accelerate launch --num_processes=1 submit_and_evaluate.py --data-root ./datasets \
    --config-path configs/motip2/pretrain_s_vehicles.yaml --inference-mode evaluate \
    --inference-model pretrains/deim_s_vehicles_converted.pth \
    --inference-dataset Vehicles --inference-split val --outputs-dir outputs/check_converted
```
How to read it: DetA should be high for a detector trained on vehicles (and clearly above the COCO-initialised route in step 6b, which
is only a lower bound). **If DetA is near 0 / far below your mmdet model's quality on the same videos, the conversion is wrong** — send me
`inspect.txt` and the `.convert_report.txt`. (Boxes are normalised cxcywh, images are fed un-normalised [0,1] RGB, exactly like DEIM;
your mmdet `DetDataPreprocessor` used `mean 0, std 255, bgr_to_rgb` which is the same convention.)

## 6. Stage 1 — adapt the detector to this dataset (recommended, short)

The paper pretrains the detector alone on the tracking dataset first. Your detector is already vehicle-specialised, so a **short** stage 1
(5 epochs) is enough; you can also skip it (6c).

```bash
# 6a. from YOUR converted detector (recommended)
accelerate launch --num_processes=4 train.py --data-root ./datasets \
    --config-path configs/motip2/pretrain_s_vehicles.yaml --exp-name deim_s_vehicles_stage1 --epochs 5
cp outputs/deim_s_vehicles_stage1/checkpoint_4.pth pretrains/deim_hgnetv2_s_vehicles_stage1.pth

# 6b. (optional baseline / fallback) from official COCO weights: COCO-80 rows of bicycle, car, motorcycle, bus, truck = 1 2 3 5 7
accelerate launch --num_processes=4 train.py --data-root ./datasets \
    --config-path configs/motip2/pretrain_s_vehicles.yaml --exp-name deim_s_vehicles_stage1_coco \
    --detr-pretrain pretrains/deim_hgnetv2_s_coco.pth --detr-class-indices 1 2 3 5 7 --epochs 20

# 6c. skip stage 1: use the converted detector directly as the stage-2 initialisation
cp pretrains/deim_s_vehicles_converted.pth pretrains/deim_hgnetv2_s_vehicles_stage1.pth
```
Stage 1 trains single frames (8 images/GPU, EMA 0.9995, no warm-up, constant LR) and evaluates the detector on `valmini` every 5 epochs
(`checkpoint_4.pth` = EMA weights after 5 epochs).

## 7. Stage 2 — joint training of detector + ID decoder (all MOTIP2 priors on)

```bash
accelerate launch --num_processes=4 train.py --data-root ./datasets \
    --config-path configs/motip2/s_vehicles.yaml --exp-name s_vehicles \
    --num-id-vocabulary 50          # <- the value from step 2
```
* 30-frame clips (frame interval random 1–4), 4 frames with detection loss per clip, 2 clips per GPU, 10 epochs × 40k samples, EMA,
  500-iteration warm-up, constant LR (S: 2e-4, backbone ×0.5, ID decoder ×2), Spatial ID Switches (σ=0.5) + Spatial ID Loss + Spatial
  Anchor + RoPE, `N_init=2`.
* Logged: `detr_loss`, `id_loss` and its parts `id_ce / id_giou / id_l1`, grad norms. After each epoch the EMA model is evaluated on
  **valmini** (HOTA, DetA, AssA, MOTA, IDF1, IDSW in the log). Checkpoints: `outputs/s_vehicles/checkpoint_<epoch>.pth` (contain EMA weights).
* **Before the full run**: a short GPU smoke to get iteration time and memory (and your compute budget):
  `... --epochs 1 --samples-per-epoch 800 --exp-name smoke`. Loss must decrease, no NaN.
* ID-switch augmentation statistics on your data (what the training will see): 
  `python tools/idsw_stats.py --config configs/motip2/s_vehicles.yaml --data-root ./datasets --mode spatial` (and `--mode uniform`).
  Expect far fewer, mostly-overlapping swaps for `spatial` (paper, DanceTrack: 10 % swap / 26 % long-range vs 37 % / 85 %).

## 8. Final evaluation on the full validation split

```bash
accelerate launch --num_processes=4 submit_and_evaluate.py --data-root ./datasets \
    --config-path configs/motip2/s_vehicles.yaml --inference-mode evaluate \
    --inference-model outputs/s_vehicles/checkpoint_9.pth \
    --inference-dataset Vehicles --inference-split val --outputs-dir outputs/s_vehicles
python tools/log_result.py --summary outputs/s_vehicles/evaluate/default/Vehicles/val/checkpoint_9/tracker/pedestrian_summary.txt \
    --config configs/motip2/s_vehicles.yaml --seed 42 --benchmark Vehicles --split val \
    --checkpoint outputs/s_vehicles/checkpoint_9.pth --notes "variant=full"
```
Results are in `…/tracker/pedestrian_summary.txt` (HOTA, DetA, AssA, MOTA, IDF1, IDSW …; the "pedestrian" label is TrackEval's name for
the single evaluated class = all vehicles) and the tracks in `…/tracker/<seq>.txt` (MOT format: `frame,id,x,y,w,h,1,-1,-1,-1`, pixel
coordinates of the stored frames).

Cheap inference-only experiments on the same checkpoint: `--n-init 1|2|3`, `--rope-alpha 1.0|1.5|2.0|2.5 --rope-alpha-mode multiply|divide`,
`--det-thresh`, `--newborn-thresh`, `--miss-tolerance`. Ablations that need retraining: `DRY_RUN=1 SIZE=S BENCH=vehicles bash tools/run_ablations.sh`
prints the commands (set `INFER_DATASET=Vehicles` there; ≥ 3 seeds before claiming a gain below 1 HOTA).

## 9. What to expect / limitations

* No number is promised: the paper's results are for DanceTrack/SportsMOT/PersonPath22. Compare your **own baseline** — the same data with the
  priors switched off (`--id-switch uniform --id-loss-spatial False --spatial-anchor False --pos-enc learned_rel --n-init 1`) — to see what the
  priors give on vehicles.
* Class-agnostic tracking metrics only; there is no per-class HOTA. The tracker output files do not carry the class (it is available in the
  model output if you need it later).
* Mapping several CVAT labels to 5 classes loses `van`/`car_with_trailer`/... distinctions by design.
* All 5 vehicle classes share one ID dictionary; ids are unique per video, not across videos.
* Sites that are really the same location under different names in the CSV would leak between train and val (see step 1).

## 10. Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `missing` videos/annotations in the preprocessing report | file naming: check `Video_file`/`Anno_file`, or put files in `--videos-dir/<id>.mp4` / `<id>.xml` |
| `extracted N frames but annotations span M` warning | video shorter than the XML says, or `frame_filter` was used in CVAT; the sequence is truncated to the decoded frames |
| `XML original_size != video` warning | boxes are mapped from the XML size to the output size; check with `visualize_gt.py` that they still fit |
| Converter exits with code 2 | required target tensors unmatched → send `inspect.txt` + `.convert_report.txt` |
| Detection-only DetA ≈ 0 after conversion | wrong weights section (`--which`) or mis-routed tensors; try `--which ema_state_dict`, else send the files |
| Stage-1/2 log shows `selected index k out of range` | frames too small for 300 queries (only with tiny test images) |
| DDP "unused parameters" error | set `DDP_FIND_UNUSED_PARAMETERS: true` in the config |
| No tracker output at all | `N_INIT=2` needs the ID decoder to re-identify a new track in the next frame: an untrained decoder outputs nothing (use `--n-init 1` only for debugging) |
