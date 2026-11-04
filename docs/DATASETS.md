# Training MOTIP2 on one or several tracking datasets

Everything here is **implemented and tested on synthetic data only** (no real dataset, no GPU is available in the development
sandbox). Read the status box in §8 before trusting a converter on a real download.

All datasets go through the same two steps: **convert once** to one common on-disk layout, then **mix** them in one training run.

```
download  ──►  tools/prepare_dataset.py <name>  ──►  datasets/<Name>/{train,val,valmini}/<seq>/...   (unified layout)
                                                                │
                      tools/gen_mix_config.py (shares, frame-interval caps, class masking) ──► configs ──► train.py
```

## 1. The unified layout (what every converter writes, what the loader reads)

```
datasets/<Name>/{train,val,valmini}/<seq>/img1/00000001.<ext>     frames, 1-indexed (ext from seqinfo.ini imExt: .jpg/.png)
                                         /gt/gt.txt               TrackEval GT. class column 1 = every target (class-agnostic HOTA), 8 = ignore/distractor
                                         /gt/gt_classes.txt       the same targets (no class-8 rows), class column = UNIFIED class index -> detector training
                                         /seqinfo.ini             frameRate, seqLength, imWidth, imHeight, imExt
datasets/<Name>/{train,val,valmini}_seqmap.txt   annotated_classes.json   prepare_report.json
```
`valmini` is a folder of symlinks to ~10 evenly spread `val` sequences (fast in-training evaluation). `annotated_classes.json` lists the
classes the dataset really labels (§4). Any folder in this layout is loadable **by name** (`--datasets MyData`), also your own.
The existing `Vehicles` dataset (tools/vehicles/) already uses it.

## 2. One class space for all datasets

Class sets (`data/class_maps.py`; the **index is the position**, do not reorder):

| set | classes |
|---|---|
| `vehicles5` (default, your list) | 0 bicycle, 1 car, 2 motorcycle, 3 bus, 4 truck |
| `vehicles5_person` | the same + 5 person |

Native labels are mapped to these (defaults in each `data/converters/<name>.py`, override with `--class-map file.json`
`{"native label": "car" | "ignore" | "distractor" | null}`). A label that exists but is **not in the chosen set** (e.g. `person` with
`vehicles5`) becomes a **distractor**: never a training target, but a tracker output matched to it is not counted as a false
positive in the evaluation (TrackEval class 8). `ignore` = crowd / don't-care regions, same treatment. Mappings I had to **assume**:

| dataset | mapping | assumption to confirm |
|---|---|---|
| BDD100K MOT | car, truck, bus, motorcycle, bicycle → same; pedestrian/rider/other person → `person`; train, trailer, other vehicle → distractor; `crowd` attribute → ignore | – |
| UA-DETRAC | car, van → `car`; bus → `bus`; **others → distractor** | `others` has no clean class |
| VisDrone-MOT | bicycle, car, van→`car`, truck, bus, motor→`motorcycle`; pedestrian/people → `person`; **tricycles, others → distractor**; category 0 and score-0 rows → ignore | tricycles |
| UAVDT | 1 car, 2 truck, 3 bus | – |
| KITTI | Car, Van→`car`; Truck; **Cyclist→`bicycle`** (box includes rider); Pedestrian/Person_sitting→`person`; Tram, Misc → distractor; DontCare → ignore | cyclist |
| nuScenes | vehicle.car/truck/bus.*/motorcycle/bicycle; human.pedestrian.* → `person`; trailer, construction, emergency → distractor | – |
| Argoverse 2 | REGULAR_VEHICLE→`car`; LARGE_VEHICLE, TRUCK, BOX_TRUCK, TRUCK_CAB→`truck`; BUS, SCHOOL_BUS, ARTICULATED_BUS→`bus`; MOTORCYCLE; BICYCLE; PEDESTRIAN→`person`; riders, trailers → distractor | – |
| Waymo | **vehicle → `car`** (labels cannot tell more), cyclist → `bicycle`, pedestrian → `person`, sign dropped | coarse |
| CityFlow | **everything → one class** (`--class car`) | coarse, see §3 |
| FastTracker (MOT format) | you supply `--class-names` (class id → name) | I could not read the dataset card (§8) |

## 3. Converting each dataset

Always `--dry-run` first: it reads the annotations only, prints classes, sizes, frame rates, objects per frame and the
**recommended `NUM_ID_VOCABULARY`**, and writes nothing. Common options: `--image-mode link|copy|jpeg` (link = symlink the original
frames, no extra disk; **jpeg** = re-encode, needed to resize or to avoid slow PNG decoding), `--resize-short/--max-long`
(boxes are rescaled), `--class-set`, `--class-map`, `--valmini N`, `--min-frames`, `--limit N` (debug).

```bash
python tools/prepare_dataset.py bdd100k --root /data/bdd100k --out ./datasets --dry-run
python tools/prepare_dataset.py bdd100k --root /data/bdd100k --out ./datasets
python tools/prepare_dataset.py detrac --root /data/ua-detrac --out ./datasets
python tools/prepare_dataset.py visdrone --root /data/visdrone-mot --out ./datasets --image-mode jpeg --resize-short 800 --max-long 1440
python tools/prepare_dataset.py uavdt --root /data/uavdt --out ./datasets
python tools/prepare_dataset.py kitti --root /data/kitti-tracking --out ./datasets --image-mode jpeg
python tools/prepare_dataset.py cityflow --root /data/cityflow --out ./datasets
python tools/prepare_dataset.py fasttracker --root /data/fasttracker-mot --out ./datasets --dry-run
python tools/prepare_dataset.py waymo --root /data/waymo_v2 --out ./datasets --cameras 1 --image-mode jpeg --resize-short 800 --max-long 1440
python tools/prepare_dataset.py nuscenes --root /data/nuscenes --out ./datasets --version v1.0-trainval --cameras CAM_FRONT CAM_BACK
python tools/prepare_dataset.py argoverse2 --root /data/av2/sensor --out ./datasets --cameras ring_front_center --image-mode jpeg
python tools/prepare_dataset.py boxes3d --root /data/i24_scenes_json --out ./datasets --min-frames 30
```
Check the result visually before training (frame numbering, scaling, ids, classes; grey = ignore/distractor):
```bash
python tools/vehicles/visualize_gt.py --seq-dir datasets/UA_DETRAC/train/MVI_20011 --frames 1 40 80 --out /tmp/vis
```

| dataset (`<name>`) | input expected (summary; the module docstring has the exact paths) | splits made | fps | licence | notes |
|---|---|---|---|---|---|
| BDD100K MOT (`bdd100k`) | `images/track/<split>/<video>/*.jpg`, `labels/box_track_20/<split>/<video>.json` | official train / val | 5 | BDD100K licence – **check** | dashcam: ego-motion; frame interval capped at 1 |
| UA-DETRAC (`detrac`) | `DETRAC-train-data`, `DETRAC-Train-Annotations-XML`, `DETRAC-test-data`, `DETRAC-Test-Annotations-XML` | official train → train, official test → val | 25 | check | fixed elevated cameras (closest to CCTV); lacks truck/motorcycle/bicycle labels |
| VisDrone-MOT (`visdrone`) | `VisDrone2019-MOT-{train,val}/{sequences,annotations}` | official train / val | not shipped (`--fps 30`) | check | drone views, very dense: raise K |
| UAVDT (`uavdt`) | `UAV-benchmark-M/<seq>`, `UAV-benchmark-MOTD_v1.0/GT/*_gt_whole.txt` | benchmark MOT train(30) / test(20) → train / val | 30 | check | test list is from memory: override with `--test-seqs` |
| KITTI tracking (`kitti`) | `training/image_02`, `training/label_02` | CenterTrack val sequences → val (override `--val-seqs`) | 10 | **CC BY-NC-SA 3.0** (non-commercial) | 1242×375 PNG; use `--image-mode jpeg` |
| CityFlow (`cityflow`) | `{train,validation}/S0x/c0yy/{vdo.avi,gt/gt.txt}` | by scene: S01 S03 S04 → train, S02 S05 → val | 10 | check | **class-less + only vehicles seen in ≥2 cameras are labelled**; frames extracted with ffmpeg into `<out>/CityFlow/_frames` (kept: sequences link to them) |
| FastTracker (`fasttracker`, also generic `mot`) | MOT-challenge folders (`img1/`, `gt/gt.txt` with a class column) | `--val-seqs` or whole-sequence random `--val-frac` | from `seqinfo.ini` | **OpenRAIL-M** | needs `--class-names` JSON; sequences are different scenes, so a sequence-level split is site-disjoint if they are different cameras |
| Waymo Open v2 (`waymo`) | `{training,validation}/{camera_image,camera_box}/*.parquet` | training / validation | 10 | **non-commercial** | needs pandas+pyarrow; v1 TFRecords not read; coarse vehicle class |
| nuScenes (`nuscenes`) | `<root>/<version>/*.json` + `samples/` (no devkit needed) | official lists if `nuscenes-devkit` is installed, else a deterministic split **by log** | 2 (keyframes only) | **non-commercial** | 3D→2D projection, see below |
| Argoverse 2 sensor (`argoverse2`) | `<split>/<log>/{annotations,city_SE3_egovehicle}.feather`, `calibration/`, `sensors/cameras/` | train / val | 10 | **non-commercial** | 3D→2D projection with ego-motion compensation; needs pandas+pyarrow |
| I-24 3D & other 3D-box data (`boxes3d`) | one JSON per scene in the documented neutral format (module docstring) | `split` field of the JSON | per scene | check | **no dedicated I-24 reader** (§8) |

**3D datasets (nuScenes, Argoverse 2, I-24 via `boxes3d`).** 2D boxes are the axis-aligned bounds of the projected cuboid
(`data/converters/project3d.py`, near-plane clipping, image clipping). Consequences you must know: boxes are looser than
hand-drawn 2D boxes; there is **no per-camera occlusion**, so occluded objects stay labelled (filters: nuScenes `--min-visibility`,
`--min-pts`, `--min-in-image`; AV2 `--min-pts`, `--min-in-image`); lens distortion is ignored; nuScenes is annotated at 2 Hz only.
Use them as extra *identity/appearance* data rather than as the main source of detector quality.

## 4. Mixing datasets: shares, frame rates, missing labels

```bash
python tools/gen_mix_config.py --name mix_a --size S --datasets Vehicles BDD100K_MOT UA_DETRAC \
    --mix 0.6 0.2 0.2 --data-root ./datasets
```
This writes `configs/motip2/mixes/pretrain_mix_a.yaml` (stage 1, detector only) and `mix_a.yaml` (stage 2, joint), warns about
licences / coarse labels / missing data, and prints the exact launch commands. What it sets, and why:

* **`DATASET_MIX`** – the share of every epoch's `SAMPLES_PER_EPOCH` (40 000) drawn from each dataset, normalised, exact (largest
  remainder). A small dataset is cycled, a big one sub-sampled, so 2000 BDD videos do not drown a handful of your own. (The original
  mechanism, integer `DATASET_WEIGHTS`, multiplies the number of clips and is unchanged.)
* **`DATASET_MAX_INTERVALS`** – cap of the random frame interval per dataset, from its frame rate: `clip(round(fps/5), 1, 4)` →
  5 fps: 1, 10 fps: 2, ≥25 fps: 4 (MOTIP2 samples one interval ∈ {1..4} per clip at 20–30 fps; skipping frames of a 5 fps video would
  make motion unrealistic). Override with `--max-interval`.
* **`CLASS_MASKING`** – a frame's classification loss ignores the classes its dataset does **not annotate** (`annotated_classes.json`):
  UA-DETRAC has no truck/motorcycle/bicycle labels, so an unlabelled truck there must not be trained as background. Coarse datasets
  (CityFlow, Waymo) annotate *no* class, so they teach box regression and identities only. Implemented in the DEIM criterion
  (mask over the class columns, also for the denoising/auxiliary/encoder terms); unmasked datasets are bit-identical to before.
* **`NUM_CLASSES`** = size of the class set; **`K` (`NUM_ID_VOCABULARY`)** = max(100, recommendations in the converted datasets'
  `prepare_report.json`) – dense datasets (VisDrone) need more; a clip with more objects than `NUM_TRAINING_IDS` is randomly sub-sampled.
* **`T`** (`--window`, default 30) is the clip length = miss tolerance = RoPE window, identical for all datasets.
* Stage 1 starts from your converted detector (`--detr-pretrain`) or from COCO (`--from-coco`, class rows sliced to the set).

Equivalent overrides on an existing config (e.g. the vehicles one) without generating anything:
```bash
accelerate launch --num_processes=4 train.py --config-path configs/motip2/s_vehicles.yaml --exp-name s_mix \
    --datasets Vehicles BDD100K_MOT --dataset-splits train train --dataset-mix 0.7 0.3 --dataset-max-intervals 4 1 --class-masking True
```
Training on **one** dataset is the same thing with a single entry (`--datasets UA_DETRAC`); no mix needed.

**Which datasets help your vehicle task?** Judgement, not measured: your data are fixed roadside/CCTV views with five classes. UA-DETRAC,
CityFlow and the FastTracker benchmark are the closest in viewpoint; BDD100K/KITTI/nuScenes/AV2/Waymo are ego-vehicle views (moving
camera: useful for appearance/identity diversity, a domain shift for a static-camera deployment); VisDrone/UAVDT are aerial. A
reasonable first experiment is *own data only* vs *own + one close dataset at 20–30 %*, compared on **your** `val` split.
Generic advice that I did not test: put your own dataset first and the largest share on it.

## 5. Evaluation, per dataset

```bash
python tools/eval_datasets.py --config configs/motip2/mixes/<name>.yaml --model outputs/<name>/checkpoint_9.pth \
    --datasets Vehicles:val BDD100K_MOT:val UA_DETRAC:val --outputs-dir outputs/<name> --run
```
Without `--run` it only prints the commands (one `submit_and_evaluate.py` per dataset). It prints a HOTA/DetA/AssA/MOTA/IDF1/IDSW
table. Notes: HOTA is **class-agnostic** (all targets are class 1 in `gt.txt`); distractors/ignore regions follow §2; each dataset
is scored on its own sequences – do not average across datasets of very different size without weighting; the in-training
evaluation uses one dataset (`INFERENCE_DATASET`, default the first one's `valmini`).

## 6. Your own dataset

Write sequences in the layout of §1 (a `gt_classes.txt` with unified class indices; `gt.txt` with class 1 for targets, 8 for
ignore) and optionally `annotated_classes.json` (`{"annotated_indices": [1, 4]}`; absent = all classes). Then
`--datasets MyData` works with no code change. For a dataset in MOT-Challenge layout with a class column use
`python tools/prepare_dataset.py mot --root ... --dir-name MyData --class-names names.json`.

## 7. Licences

KITTI (CC BY-NC-SA 3.0), nuScenes (CC BY-NC-SA 4.0), Argoverse 2 (CC BY-NC-SA 4.0) and Waymo Open (own non-commercial licence) are
**non-commercial**; FastTracker is OpenRAIL-M (use restrictions). For BDD100K, UA-DETRAC, VisDrone, UAVDT and CityFlow I did **not**
verify the terms. `gen_mix_config.py` prints a warning for each. Whether a model trained on such data may be deployed commercially is
a legal question I cannot answer; check each licence.

## 8. Status – what is and is not verified

* **Verified (tests/test_datasets_convert.py, test_project3d.py, test_multi_dataset.py):** each converter on a tiny dataset written
  in that dataset's native format (as I know it from its documentation) – boxes, unified class indices, id consistency, splits,
  distractor/ignore rows, image modes/resizing; the 3D projections against an independent corner-by-corner projection (including
  translated and yawed ego poses and, for AV2, a moving ego between lidar and camera time); mixed sampling (exact shares, caps,
  determinism), class masking (zero gradient on masked columns, unchanged elsewhere, through the real augmentation pipeline), the
  config generator, and a CPU training + evaluation run on two converted datasets.
* **NOT verified:** any converter on the **real** download. Formats come from the dataset documentation as I remember it; the sandbox
  has no access to the datasets and the sites that could confirm details (arXiv, Hugging Face) were blocked. In particular:
  FastTracker's 9 class ids and layout, the UAVDT test-sequence list, the Waymo v2 column names (matched by suffix, error lists the
  columns found), the AV2 / nuScenes file layouts, BDD100K label field names, the VisDrone `score=0` convention. **I-24 3D has no
  dedicated reader** (its file format could not be checked): convert it to the neutral JSON of `boxes3d`. Run `--dry-run` and
  `visualize_gt.py` on every new dataset and fix `--class-map` / paths if the output looks wrong.
* The nuScenes *official* train/val lists need `nuscenes-devkit`, which is not installed here: that branch is untested (the by-log
  fallback is tested).
* **NOT measured:** whether mixing any of these datasets improves HOTA on your vehicle data. That needs GPU runs, and the full
  training still waits for your confirmation of the compute budget.
