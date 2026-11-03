# PLAN.md — MOTIP → MOTIP2 (arXiv 2610.10391, BMVC 2026)

Status: **Phase 0 approved by the user; Phases 1-8 implemented and unit-tested on CPU (see RESULTS.md for gate status). Gate 1 and all metric gates need a GPU + real data; no full training run has been started.**
Branch: `motip2` (from `e23387c`). Skill: `motip2-implementation` (section numbers `§n` refer to it; `Paper §n` / `Tab. n` to the PDF).
DEIM reference checkout inspected: `Intellindust-AI-Lab/DEIM @ 09d35d5` (Apache-2.0), read-only at `/home/user/intellindust-ai-lab/deim`.

---

## 0. Environment facts that shape this plan

| Fact | Consequence |
|---|---|
| This container has **no GPU**, no `torch`, 4 CPU / 15 GB RAM, no `datasets/`, no `pretrains/`. `pip` works (torch 2.15 visible on the index). | Unit gates (3–7 unit parts) can run on CPU after `pip install torch --index-url …/cpu`. Every metric gate and every smoke/full run needs a GPU box with datasets and weights. |
| `models/ops` (MOTIP's Deformable-DETR CUDA op) must be compiled with `nvcc`. | Needed only for Phase 1 (baseline) and `detector: deformable_detr`. DEIM is pure PyTorch (no custom op). |
| `data/joint_dataset.py:17-22` registers only DanceTrack, SportsMOT, CrowdHuman, BFT. **No PersonPath22 and no MOT17 loader exists**, although `submit_and_evaluate.py:223,268,288` already references `PersonPath22_Inference`. | Any PersonPath22 gate (Gate 3 metric, Tab. 5/6 PP22 columns) needs a new `data/personpath22.py` (train + inference classes). Listed in §2, Phase 8. |
| DEIM COCO checkpoints (N/S/M/L/X) are distributed via Google Drive links in the DEIM README. | Must be supplied by the user (or fetched if the proxy allows). Blocker for Phase 2. |
| DanceTrack **test** GT is not public. | All local gates use DanceTrack **val**. Test numbers (Tab. 1) need the CodaLab server and are out of scope for local gating. |

---

## 1. Phase 0 map of the existing MOTIP repo

### 1.1 Where each concern lives today

| Concern | File:lines | What it does now | MOTIP2 action |
|---|---|---|---|
| Detector wrapper | `models/motip/motip.py:7-50`; build in `models/motip/__init__.py:10-79` | `MOTIP.forward(part=…)` dispatches `detr` / `trajectory_modeling` / `id_decoder`. Builder hard-wires Deformable DETR (`:44-48`). | Add `detector: deim` branch + `models/deim/` wrapper. |
| Deformable DETR forward | `models/deformable_detr/deformable_detr.py:118-195` | Returns `pred_logits`, `pred_boxes` (`:184`), aux outputs, and **`out["outputs"] = hs[-1]`** (`:194`) = detection embedding `e_i`. | DEIM wrapper must return the same keys **plus `pos_embeds` (p_i)**. |
| Detection loss + matcher | `models/deformable_detr/deformable_detr.py:365-445` (`SetCriterion.forward` → `(losses, indices)`), `matcher.py:20-96` | Focal-loss class cost, α=0.25 γ=2; returns Hungarian `indices` used later for ID supervision. | DEIM criterion (MAL/FGL/DDF) **returns losses only** (`engine/deim/deim_criterion.py:266-353`); wrap to also return final-layer `indices`. |
| ID decoder input construction | `models/motip/id_decoder.py:108-112` | Trajectory tokens = `cat[feature(256), id_embed(256)]`; unknown tokens = `cat[feature, empty-id-embed(K)]`. Dim 512. | Replace by `feature + id_embed` (+ `p_i` for detections), dim = DEIM hidden dim. |
| Temporal bias | `id_decoder.py:49-56` (learned `rel_pos_embeds[layers, REL_PE_LENGTH, heads]`), `:121-130` (index pairs), `:214-215` (added to attn mask) | Learned discrete bias per layer/head, clipped to `REL_PE_LENGTH=30`. | `pos_enc: rope` rotates q/k on head dim; keep `learned_rel` path untouched. |
| Self-attn among unknowns / cross-attn | `id_decoder.py:200-228` | Layer 0: cross-attn only; layers ≥1: self-attn over same-frame unknowns then cross-attn to trajectory. `add_zero_attn=True`. | Keep structure; swap dim/FFN ratio/PE. |
| Trajectory feature adapter | `models/motip/trajectory_modeling.py:8-46` | Two residual FFN+LayerNorm blocks applied to **trajectory features only** (`train.py:421`, `runtime_tracker.py:205`). | Kept (see N4). |
| ID switch augmentation | `data/transforms.py:511-534` (inside `TurnIntoTrajectoryAndUnknown`, `:452-575`) | For every (group, frame) independently: Bernoulli(`AUG_TRAJECTORY_SWITCH_PROB=0.5`) per object picks subset S; **uniform `randperm`** over S permutes `trajectory_ann_idxs` and masks. Unknown tokens never switched. | New `spatial` mode: Hungarian over `GIoU + N(0,σ²)` using that frame's boxes. |
| Occlusion masking | `transforms.py:487-509` | Per (group, object): p=0.5, random span masked in both trajectory and unknown masks. | Unchanged (p=0.5 matches Paper App. A). |
| ID-label generation | `transforms.py:375-449` (`GenerateIDLabels`), groups `AUG_NUM_GROUPS=6`, `NUM_TRAINING_IDS` | Random `randperm(K)[:N]` labels per group. | Unchanged; see N7 for "6×100 IDs". |
| Newborn labels | `transforms.py:540-562` | Overwrites first-appearance unknown label with `K`. | Must also export un-overwritten labels for Spatial ID Loss (`C_i`). |
| Dataset / sampler | `data/joint_dataset.py:127-171`, `data/naive_sampler.py:46-131` | Epoch = every legal clip start; `interval = min(randint(1, SAMPLE_INTERVALS), max_interval)` (`:102`) — matches Paper (uniform 1..4). No "samples per epoch" knob (only `LENGTH_PER_ITERATION`). | Add `samples_per_epoch` (Paper: 40k). |
| Collate / padding | `data/util.py:56-115` | NestedTensor with masks; pads ID tensors to max N. | DEIM ignores padding masks → see N5. |
| Image augmentation | `transforms.py:578-621` | Random shift sim., flip, resize/crop, torchvision ColorJitter(0.4,0.4,0.4,0.2). | Replace by Paper App. A pipeline (flip, zoom-out, IoU-crop, photometric, DEIM-style multi-scale). |
| ID loss | `models/motip/id_criterion.py:25-53` | Plain CE summed / #valid ids; first frame of clip excluded (`:28-30`). | Add IoU-weighted CE + spatial GIoU/L1 terms. |
| Training loop | `train.py:32-257` (epoch loop), `:260-522` (`train_one_epoch`) | `MultiStepLR` (`:122`, milestones [6,9]); warmup in epochs (`:565-576`); ImageNet mean/std normalisation on GPU (`:313-319`); 4 grad frames + rest `no_grad` (`:331-409`); detr criterion over **all** frames (`:412`); clip-norm separately for `detr` vs rest (`:453-454`); **no EMA**. | Constant LR, 500-iter warmup, EMA(0.9995, 1000 warmup), per-group LR/WD per Paper App. A, no ImageNet normalisation for DEIM, new loss plumbing. |
| Param groups | `train.py:525-562` | backbone ×0.1, linear-proj ×0.05, dictionary ×1.0, rest ×1.0. | New groups: backbone (size-dependent), ID decoder ×2, rest ×1; WD off for norm/bias/embeddings. |
| `prepare_for_motip` | `train.py:657-710` | Builds trajectory/unknown tensors from matched queries: features + **predicted boxes** (already carried, unused by the decoder today). | Also carry `p_i`, GT boxes, un-overwritten labels, presence masks. |
| Inference tracker | `models/runtime_tracker.py:14-405` | Buffer cut `[-miss_tolerance+2:]` (`:261`), times re-indexed `arange` each frame (`:313-316`), `id_queue` OrderedSet of free slots, protocols hungarian/object-max/id-max (`:328-405`), hard-coded 256-dim (`:72,281,293`). Unconfirmed-track logic: **none** (newborns are emitted immediately). | Add N_init confirmation, RoPE α, dims from config, p_i in buffer. |
| Default assignment | `configs/r50_…_dancetrack.yaml:124-125`: `ASSIGNMENT_PROTOCOL: object-max`, `ID_THRESH: 0.2`; docs/TUTORIAL.md:8 ("simpler … only selects the highest confidence") | MOTIP's "greedy" readout = `object-max`. | Keep `object-max`; set `ID_THRESH=0.1` (λ_ID, §1.3). |
| Eval | `submit_and_evaluate.py:115-342` → `TrackEval/scripts/run_mot_challenge.py`; metrics read from `pedestrian_summary.txt` (`:325-336`, **IDSW not captured**) | HOTA/DetA/AssA/MOTA/IDF1. `RuntimeTracker` per sequence; FPS measured from frame 10 (`:360-362`). | Capture IDSW; add latency tool; add `--n-init`, `--rope-alpha` pass-through. |
| Configs | `configs/*.yaml`, `configs/util.py`, `runtime_option.py:16-111` | YAML + `SUPER_CONFIG_PATH` chaining; CLI flags only for keys listed in `runtime_option.py`. | New `configs/motip2/` tree; new CLI flags for every new key. |
| Checkpoint IO | `models/misc.py:87-173` | `load_detr_pretrain` slices COCO 91-class head to 1 class (Deformable-DETR specific). | DEIM loader (80-class remapped head → slice person=0 for `dec_score_head`, `enc_score_head`, `denoising_class_embed`). |

### 1.2 Divergences from the skill/paper discovered in the existing code (affect decisions below)

1. MOTIP's switch is **per-frame** (independent for every `(group, frame)`), using Bernoulli(0.5) *per object* to choose S. `AUG_TRAJECTORY_SWITCH_PROB=0.5` is therefore not "probability of applying the augmentation"; Paper App. A's "applied with p=1.0" must mean "always applied, with MOTIP's subset sampling unchanged".
2. Absent/occluded objects take part in MOTIP's uniform permutation (their mask is swapped with a present object's). A GIoU affinity is undefined for them → N2.
3. DEIM expects **un-normalised** `[0,1]` input (`ConvertPILImage(scale=True)`), MOTIP normalises with ImageNet mean/std (`train.py:313-319`, `data/seq_dataset.py:29`). Feeding COCO-pretrained DEIM ImageNet-normalised images would silently hurt.
4. DEIM and its decoder anchors use a **static** `eval_spatial_size` (`[640,640]`) unless set to `None` (`dfine_decoder.py:639`, `hybrid_encoder.py:405`). We run 800×1440 → must be `None`.
5. DEIM has **no padding-mask support** in the encoder/decoder (`memory_mask=None`), while MOTIP batches NestedTensors of unequal size.

---

## 2. DEIM check — where `p_i` is produced (Task 2)

**Location.** `engine/deim/dfine_decoder.py`
- Module: `DFINETransformer.query_pos_head = MLP(4, 2*hidden_dim, hidden_dim, 2)` (`:481`) — this is the "positional MLP"; no other positional MLP exists in the decoder.
- Use: `TransformerDecoder.forward` (`:327-396`), at the top of each layer loop: `query_pos_embed = query_pos_head(ref_points_detach).clamp(min=-10, max=10)` (`:359`), consumed by `layer(output, ref_points_input, value, spatial_shapes, attn_mask, query_pos_embed)` (`:368`).
- Box that is embedded: `ref_points_detach` = `sigmoid(ref_points_unact)` for layer 0, and **`inter_ref_bbox.detach()` of the previous layer** thereafter (`:393`). Boxes are normalised cxcywh.
- Final-layer outputs: `inter_ref_bbox = distance2bbox(…)` (`:378`) appended at `i == eval_idx` (`:380-387`) → `out['pred_boxes']`; layer output `output` (`:368`) is the content embedding `e_i` (**not exposed today**; MOTIP's analogue is `out["outputs"]`).

**Finding that the skill's Phase 2/4 wording glosses over.** `query_pos_head` at the *last* decoder layer embeds the box **entering** that layer (= layer L−1's refined box), not the final output box `b̂_i`. So "p_i = positional MLP output of the final-layer box" has two readings:

- **(a) `layer_input`** — capture `query_pos_embed` at layer `eval_idx` inside the decoder. Zero extra compute, literally "reuse what the detector already computes", matches Paper Fig. 2 right panel (the MLP at layer *l* consumes `(x,y,w,h)_{l-1}`) and "no new parameters". p_i is one refinement step stale w.r.t. `b̂_i`.
- **(b) `final_box`** — `query_pos_head(b̂_i.detach()).clamp(-10,10)`: one extra tiny MLP call, same parameters, embeds exactly the box the ID loss/outputs use.

Decision recorded as **N1** (default (a), (b) behind a flag).

**Other DEIM facts needed by the plan**
- Variant configs (`configs/deim_dfine/dfine_hgnetv2_{n,s,m,l,x}_coco.yml`): encoder hidden dim N=128, S/M/L=256, X=384 (**decoder** hidden dim N=128, S/M/L/X=256 — X's decoder keeps the base 256; corrected during Phase 2); decoder layers N=3, **S=3**, M=4, L=6, X=6 (`eval_idx: -1` ⇒ last layer). Backbones B0 (N,S), B2 (M), B4 (L), B5 (X). `num_queries=300`, `num_denoising=100`. → ID decoder (hidden, layers) = (128,3), (256,3), (256,4), (256,6), (256,6) per §1.1 "mirrors the DEIM decoder".
- Losses (`configs/base/deim.yml`, `dfine_hgnetv2.yml`): `loss_mal:1, loss_bbox:5, loss_giou:2, loss_fgl:0.15, loss_ddf:1.5`; α=0.75, γ=1.5; matcher costs class 2 / bbox 5 / giou 2 → **exactly Paper App. A**.
- Per-size optimiser: DEIM LR N 8e-4, S/M 4e-4, L/X 5e-4; backbone LR = 0.5×(N,S), 0.1×(M), 0.05×(L), 0.01×(X); weight decay 1e-4 (N/S/M), 1.25e-4 (L/X); WD off for norm/bn/bias. **Paper's LRs are DEIM's halved for S/M/L/X (N unchanged); backbone multipliers are identical to DEIM's.** (evidence for decision #3)
- Multi-scale: `generate_scales(base_size, base_size_repeat)` in `engine/data/dataloader.py:86-91` = 75 %…125 % of base, multiples of 32, `base_size` repeated for bias → this *is* the Paper's "tunable bias toward base"; applied **per batch** in the collate fn. Resolves how "all frames in a clip share the scale".
- EMA: `engine/optim/ema.py:23-66`, `decay_fn = decay·(1−exp(−updates/warmups))` ⇒ Paper's "0.9995, 1000-step warmup" maps 1:1 (`warmups=1000` joint, `warmups=0` pretrain).
- Decoder training forward needs `targets` for contrastive denoising (`dfine_decoder.py:705-722`); MOTIP's `model(frames=…, part="detr")` passes none.
- Criterion returns losses only; indices (`:276`) are internal.

---

## 3. Config flags (§0 + additions)

All keys are YAML keys (UPPER_SNAKE in files, `--kebab-case` on the CLI via `runtime_option.py`). Defaults are the MOTIP2 defaults; `legacy` row shows what reproduces the unmodified MOTIP.

### 3.1 The six mandatory ablation toggles (§0.2)

| Key | Values | MOTIP2 default | Legacy-MOTIP value |
|---|---|---|---|
| `ID_SWITCH` | `none` \| `uniform` \| `spatial` | `spatial` | `uniform` |
| `ID_LOSS_SPATIAL` | bool | `True` | `False` |
| `SPATIAL_ANCHOR` | bool | `True` | `False` |
| `POS_ENC` | `learned_rel` \| `rope` | `rope` | `learned_rel` |
| `N_INIT` | int ≥ 1 | `2` | `1` (= no confirmation) |
| `DETECTOR` | `deformable_detr` \| `deim` | `deim` | `deformable_detr` |

### 3.2 Additional keys (needed so every sub-choice in §14 is switchable and testable)

| Key | Default | Purpose |
|---|---|---|
| `MODEL_SIZE` | `S` (`N/S/M/L/X`) | Selects DEIM config, ID-decoder (dim, layers), LR/WD/backbone multiplier. |
| `DEIM_CONFIG` | path to vendored YAML | Frozen copy of the DEIM per-size config (decision #1). |
| `ID_FUSION` | `add` \| `concat` | `concat` only to keep Phase-1/legacy path alive. |
| `ID_FFN_DIM_RATIO` | `4` (MOTIP: 2 at 2× width) | FFN width = 4×256 = 1024 = MOTIP's 2×512. |
| `ID_SWITCH_SIGMA` | `0.5` | σ (§1.3). |
| `ID_SWITCH_REFERENCE` | `per_frame` \| `last_frame` | Decision #6. |
| `ID_SWITCH_SUBSET_PROB` | `0.5` | MOTIP's per-object inclusion prob (decision #7); replaces the meaning of `AUG_TRAJECTORY_SWITCH_PROB`. |
| `ID_SWITCH_ABSENT` | `skip` \| `neg_one` | N2. |
| `ID_LOSS_WEIGHT` | `1.0` | Overall (App. A). |
| `ID_LAMBDA_CE`, `ID_LAMBDA_GIOU`, `ID_LAMBDA_L1` | `1.0, 2.0, 5.0` | Sec. 4.1. |
| `ID_CE_IOU_WEIGHT` | `True` | App. A per-sample IoU weighting. |
| `SPATIAL_ANCHOR_SOURCE` | `layer_input` \| `final_box` | N1. |
| `SPATIAL_ANCHOR_DETACH` | `False` | N8. |
| `ROPE_ALPHA` | `2.0` | α (§1.3). Inference-only. |
| `ROPE_ALPHA_MODE` | `multiply` \| `divide` | Decision #8 (skill default `multiply`; open question). |
| `ID_THRESH` | `0.1` | λ_ID (MOTIP configs use 0.2). |
| `DET_THRESH`, `NEWBORN_THRESH` | `(0.3,0.7)` Dance/Sports, `(0.4,0.6)` PP22 | §1.3. |
| `MISS_TOLERANCE`, `SAMPLE_LENGTHS` | `30` (60 SportsMOT) | `T = W` (§1.3). `REL_PE_LENGTH` only used for `learned_rel`. |
| `NUM_ID_VOCABULARY` | `50` (300 PP22) | K. |
| `NUM_TRAINING_IDS` | `min(100, K)` | N7. |
| `USE_EMA`, `EMA_DECAY`, `EMA_WARMUPS` | `True, 0.9995, 0/1000` (stage 1 / 2) | App. A. |
| `LR_SCHEDULER` | `constant` | MOTIP uses `MultiStepLR`. |
| `LR_WARMUP_ITERS` | `0` (stage 1) / `500` (stage 2) | Replaces `LR_WARMUP_EPOCHS`. |
| `SAMPLES_PER_EPOCH` | `40000` | N3. |
| `TRAJECTORY_MODELING` | `True` | N4. |
| `NUM_ID_DECODER_LAYERS`, `ID_DIM`, `FEATURE_DIM` | derived from `MODEL_SIZE` | Decisions #2. |
| `IMAGE_NORMALIZE` | `False` for DEIM / `True` legacy | §1.2.3. |
| `BASE_SIZE_REPEAT` | DEIM default | N5. |
| `N_INIT_BACKFILL` | `False` | N6. |

---

## 4. File-by-file change list

Legend: **N** new, **M** modified, **K** kept as-is. "Flag" = toggle that disables the change.

### Phase 1 — Baseline reproduction (no model edits)
| File | Change |
|---|---|
| `submit_and_evaluate.py:325-336` (M) | Also parse `IDSW` (and `CLR_FP/FN`) from `pedestrian_summary.txt` so RESULTS.md rows are complete. Pure logging; no behaviour change. |
| `tools/log_result.py` (N) | Appends one RESULTS.md row (date, commit, config, seed, benchmark, split, HOTA, DetA, AssA, MOTA, IDF1, IDSW, latency, ckpt). |
| `tools/benchmark_latency.py` (N) | Detector-only / ID-decoder-only / total latency, FP32, 800×1440, T=30, 50 tracks, 50 dets (Tab. 4 protocol). |

### Phase 2 — DEIM detector
| File | Change |
|---|---|
| `models/deim/` (N) | Vendored from DEIM (Apache-2.0, headers kept): `hybrid_encoder.py`, `dfine_decoder.py`, `dfine_utils.py`, `denoising.py`, `matcher.py`, `deim_criterion.py`, `box_ops.py`, `utils.py`, HGNetV2 backbone, `ema.py`. Drop DEIM's YAML-registry/solver; replace with a small explicit builder. |
| `models/deim/dfine_decoder.py` (M vs upstream) | `TransformerDecoder.forward` additionally records, at `i == eval_idx`: `output` (→ `outputs`, e_i) and `query_pos_embed` (→ `pos_embeds`, p_i, N1(a)). `DFINETransformer.forward` splits them with `dn_num_split` like logits/boxes and returns `out['outputs']`, `out['pos_embeds']` (eval path included). |
| `models/deim/wrapper.py` (N) | `DEIMDetector(nn.Module)`: `forward(samples: NestedTensor) → {pred_logits, pred_boxes, outputs, pos_embeds, [train: aux/dn/pre/enc outputs, pred_corners, ref_points, up, reg_scale, dn_meta]}`. Takes `targets` for denoising when training. Lightweight forward (final layer only) for `no_grad` frames. `eval_spatial_size=None`. |
| `models/deim/criterion.py` (N) | `DEIMCriterion` wrapper returning `(loss_dict, final_layer_indices)`; `match_only()` for the 26 no-grad frames (avoids full aux/DN loss there). |
| `models/motip/__init__.py:10-79` (M) | Dispatch on `DETECTOR`; build DEIM + criterion; map `MODEL_SIZE`→(hidden, layers). |
| `models/motip/motip.py:30-50` (M) | `part="detr"` passes `targets`/`lightweight` through. |
| `models/misc.py:87-139` (M) | `load_deim_pretrain()`: COCO-80 head → 1 class (person = index 0 under `remap_mscoco_category`), slice `dec_score_head.*`, `enc_score_head`, `denoising_class_embed`; strict load for the rest. EMA-aware `save/load_checkpoint`. |
| `train.py` (M) | Detector-agnostic detr step: grad frames full forward + loss; no-grad frames lightweight forward + `match_only`; replace `tensor_dict_cat` on aux/dn dicts (it raises on `None`/ints, `train.py:635`); skip ImageNet normalisation when `IMAGE_NORMALIZE=False`; EMA; constant LR; iteration warmup; new param groups; WD masks; `samples_per_epoch`. |
| `data/transforms.py:578-621` (M) | New `build_transforms` branch for Paper App. A image pipeline: flip 0.5, zoom-out 0.5 scale [1,2], IoU-crop 0.8 (scale [0.3,1], aspect [1.5,2]), photometric 0.5 (brightness [0.875,1.125], contrast/sat [0.5,1.5], hue ±0.05). Multi-scale moved to batch level (`generate_scales`). |
| `data/util.py:56-115` (M) | Batch-level scale selection; pad only within equal-scale batches (N5). |
| `data/naive_sampler.py` (M) | `samples_per_epoch` (random subset of `sample_infos`, seeded per epoch). |
| `configs/motip2/` (N) | `pretrain_{s}_{dancetrack}.yaml` etc. (see Phase 8). |
| `runtime_option.py` (M) | CLI flags for every new key (§3). |
| `tests/test_deim_wrapper.py` (N) | Shapes; `outputs`/`pos_embeds` present; **p_i == `query_pos_head(inter_ref_bbox_{L-1})`** (N1a) and ≠ embedding of final box; train/eval parity of final-layer outputs. |

### Phase 3 — ID decoder structure
| File | Change |
|---|---|
| `models/motip/id_decoder.py` (M) | `ID_FUSION=add`: `x = e + word_to_embed(id)`; hidden dim = DEIM hidden (no concat); `n_heads = hidden // HEAD_DIM`; FFN ratio 4; layers from `MODEL_SIZE`. Heads read `unknown_embeds` (full dim) instead of `[..., -id_dim:]` (`:171`). `POS_ENC=rope`: RoPE applied to q/k inside cross-attn (and none in same-frame self-attn); requires replacing `nn.MultiheadAttention` by an explicit q/k/v attention for the cross-attn so q,k can be rotated; masks (`key_padding`, causal-in-time `trajectory_times >= unknown_times`) preserved; `learned_rel` path left bit-identical. |
| `models/motip/rope.py` (N) | `apply_rope(x, positions, base=10000)`; `rope_positions(times, alpha, mode, training)`. Eq. as comment `# MOTIP2 Sec. 3.1 (RoPE) …`. |
| `models/motip/trajectory_modeling.py` (M) | Dim from `DETR_HIDDEN_DIM`; skippable via `TRAJECTORY_MODELING` (N4). |
| `models/runtime_tracker.py` (M) | Replace hard-coded `256` (`:72,281,293`) with config dim; positions = `times · α` or `times / α` per `ROPE_ALPHA_MODE` at inference only. |
| `tests/test_rope.py`, `tests/test_id_decoder.py` (N) | See Gate 3. |

### Phase 4 — Spatial Anchor
| File | Change |
|---|---|
| `train.py:657-710` `prepare_for_motip` (M) | Also gather `pos_embeds[matched queries]` → `trajectory_pos`, `unknown_pos`. |
| `models/runtime_tracker.py` (M) | Keep `p_i` in the trajectory buffer alongside features; feed current-frame `p_i`. |
| `models/motip/id_decoder.py` (M) | If `SPATIAL_ANCHOR`: `x = e + id_embed (+ trajectory_modeling) + p_i`, added **after** fusion, before layer 0, for **every detection token (trajectory and unknown)** — "so every track carries its frame position" (Paper §3.4). No new parameters. `p_i` dim == hidden dim (asserted). |
| `tests/test_spatial_anchor.py` (N) | Gate 4. |

### Phase 5 — Spatial ID Switches
| File | Change |
|---|---|
| `data/transforms.py:511-534` (M) | `ID_SWITCH=spatial`: per (group, frame) [or last-frame, #6] subset S ~ Bernoulli(`ID_SWITCH_SUBSET_PROB`) per object (unchanged), `G=GIoU(boxes_S)`, `A=G+N(0,σ²)`, `π=linear_sum_assignment(A, maximize=True)`, apply π to `trajectory_ann_idxs` / masks of S. `uniform` = existing code; `none` = skip. Needs the per-frame boxes → `GenerateIDLabels`/annotations already carry `bbox` (cxcywh, normalised after `MultiNormalizeBoundingBoxes`). |
| `data/switch_stats.py` (N) | Counters for Swap%, LR% (GIoU ≤ 0), mean GIoU of swapped pairs (Tab. 8 definitions). Opt-in, cheap. |
| `tools/idsw_stats.py` (N) | Runs the transform pipeline over one epoch (CPU, no model) and prints Tab. 8 columns for `none/uniform/spatial`. |
| `tests/test_spatial_idsw.py` (N) | Gate 5 unit. |

### Phase 6 — Spatial ID Loss
| File | Change |
|---|---|
| `data/transforms.py:540-575` (M) | Export per-token `unknown_real_id_labels` (pre-newborn overwrite) and `unknown_present_masks` (GT presence before occlusion masking, N3b) so `C_i` can be formed. |
| `data/util.py:56-115` (M) | Pad the new fields. |
| `train.py:657-710` (M) | Carry `unknown_gt_boxes` (GT box of the slot's annotation, from `annotations[b][t]["bbox"][ann_idx]`), detached predicted boxes already carried. |
| `models/motip/id_criterion.py:25-53` (M) | `L_CE` (per-sample weight `w_i = IoU(b̂_i, b_gt_i).detach()`), `L_id-l1`, `L_id-giou` over softmax renormalised on `C_i` (present identities of the same group/frame + newborn), `b̂` detached, GT box of newborn class = token's own GT. `L_id = ID_LOSS_WEIGHT·(λ_CE L_CE + λ_GIoU L_giou + λ_L1 L_l1)`. Same cxcywh-normalised box format as the detection loss (#10). Logs the three terms separately. |
| `tests/test_spatial_id_loss.py` (N) | Gate 6 unit. |

### Phase 7 — Inference
| File | Change |
|---|---|
| `models/runtime_tracker.py` (M) | Tentative-track state machine for `N_INIT` (N6): newborn → tentative (counts as hit 1, embedding enters buffer so it can be re-identified), confirmed after `N_INIT` consecutive frames assigned to the same slot with prob ≥ `ID_THRESH`; unconfirmed + missed → removed from buffer, slot returned to `id_queue`; only confirmed tracks written to results (no backfill). `N_INIT=1` ≡ legacy behaviour (asserted by test). Assignment protocol stays `object-max`; `ID_THRESH=0.1`. |
| `submit_and_evaluate.py` (M) | Pass `n_init`, `rope_alpha`, `rope_alpha_mode`, `det/newborn/id` thresholds; inference size: shorter 800, longer ≤ 1440 (`image_max_longer`), `size_divisibility=32`. |
| `tests/test_tracker_confirmation.py` (N) | Synthetic 3-frame scenarios (blip detection, consistent detection, slot freed). |

### Phase 8 — Full configs / data
| File | Change |
|---|---|
| `configs/motip2/base.yaml` (N) | All shared defaults (§1.3, §1.2 losses, §1.4 aug/optim). |
| `configs/motip2/size_{n,s,m,l,x}.yaml` (N) | Per-size: DEIM config id, ID-decoder dim/layers/heads, LR (8e-4 / 2e-4 / 2e-4 / 2.5e-4 / 2.5e-4), ID-decoder LR ×2, backbone ×(0.5, 0.5, 0.1, 0.05, 0.01), WD (#3), batch (images/clips per GPU), `DETR_NUM_TRAIN_FRAMES` (8 for N, else 4). |
| `configs/motip2/bench_{dancetrack,sportsmot,personpath22}.yaml` (N) | `T=W`, K, thresholds, datasets, sample lengths, inference dataset/split. |
| `configs/motip2/pretrain_*.yaml` (N) | Stage 1: `ONLY_DETR=True`, `SAMPLE_LENGTHS=[1]`, no warmup, EMA warmups 0. |
| `data/personpath22.py` (N), `data/joint_dataset.py:17-22` (M), `submit_and_evaluate.py` dataset tables (M) | PP22 train/inference loaders + CrowdHuman co-training weights (follow MOTIP's recipe, #13). `is_legal` already tolerates duplicate ids (`data/util.py:20`) — verify. |
| `docs/MOTIP2.md` (N) | How to run each phase. |

### Phase 9 — Ablations
| File | Change |
|---|---|
| `tools/run_ablations.sh` (N) | Emits the exact flag sets for (a)–(i) in §11. |
| `tools/aggregate_results.py` (N) | Mean ± std over seeds from RESULTS.md rows. |

---

## 5. Ambiguity decisions

Items 1–14 are §14 of the skill. **N1–N9** are additional ambiguities I found while mapping the code. "Δ skill" = differs from the skill's stated default. **Bold ⚠** = I need your approval before Phase 1 because it can move results materially.

| # | Topic | Decision | Δ skill | Approve? |
|---|---|---|---|---|
| 1 | DEIM configs N/S/M/L/X | Vendor the official `deim_hgnetv2_{n,s,m,l,x}_coco.yml` + `dfine_hgnetv2_*` (verified above: hidden/layers/backbone). Verify Tab. 4 latency ordering N<S<M<L<X and param counts in Gate 2. | no | no |
| 2 | ID decoder hidden dim / FFN | hidden = DEIM *decoder* hidden (128/256/256/256/256), layers = DEIM layers (3/3/4/6/6), `FFN ratio = 4` (= MOTIP's 2×512 = 1024 width at S, so "double the ratio at half the width" preserves FFN capacity), head dim 32. | no | no |
| 3 | Weight decay | **DEIM per-size values**: 1e-4 (N/S/M), 1.25e-4 (L/X); 0 for norm/bias/embeddings (`word_to_embed`, `embed_to_word`, `rel_pos_embeds`). Evidence: App. A says WD is "set per model size"; every other optimiser number in App. A is DEIM's (backbone multipliers identical; LRs = DEIM's ÷2 for S–X); MOTIP has a single 5e-4. | **yes** | ⚠ |
| 4 | Stage-1 DEIM pretraining length | **20 epochs × 40k samples, constant LR, no warmup, EMA decay 0.9995/warmups 0** (= MOTIP's own `pretrain_r50_deformable_detr_dancetrack.yaml`: 20 epochs). DEIM's "default schedule" (S = 132 COCO epochs) is COCO-scale and not applicable to a 40k-sample/epoch tracking set. Record the epoch at which Gate 2 DetA plateaus. | **yes** | ⚠ |
| 5 | EMA vs warmup | Both, as written: LR warmup 500 iters (stage 2 only; stage 1 none), EMA 0.9995 with `warmups=1000` in stage 2 and `0` in stage 1. Eval/ckpt use EMA weights. | no | no |
| 6 | IDSW reference frame | **Per-frame**: for every `(group, frame t)` the subset S and GIoU matrix come from *that frame's* boxes, and π permutes that frame's tokens — this is what MOTIP's code does (`transforms.py:519-529`) and what Paper §3.2 states ("selects a random subset of objects **in each frame**… boxes in frame *t*"); it is also consistent with Tab. 8's per-epoch swap rate (I simulated MOTIP's per-frame Bernoulli(0.5) + uniform permutation: 33.5 / 37.6 / 40.1 / 41.7 % of objects change identity for N = 6 / 8 / 10 / 12 objects per frame, vs. 37 % reported for DanceTrack). The skill's default (most-recent frame for the whole history) is behind `ID_SWITCH_REFERENCE=last_frame` and run as an ablation if Gate 5 statistics are off. | **yes** | ⚠ |
| 7 | Subset-size distribution / "p = 1.0" | Keep MOTIP's per-object Bernoulli(0.5) inclusion unchanged (`ID_SWITCH_SUBSET_PROB`). "Applied with probability 1.0" = augmentation always on (`AUG_TRAJECTORY_SWITCH_PROB` previously conflated both). | no (clarification) | no |
| 8 | RoPE α mechanics | Implement **both** `ROPE_ALPHA_MODE=multiply` (skill default: `pos·α`) and `divide` (`pos/α`, i.e. timescale×α ⇒ angular frequency÷α). **Default = `multiply` per the skill, but I think it is probably backwards:** training positions are clip indices, whose adjacent step averages (1+2+3+4)/4 = **2.5** real frames; at inference one step = 1 real frame ≈ 0.4 training steps. Dividing by α∈[2,2.5] gives 0.4–0.5 (matches the Tab. 6 plateau at α=2.0–2.5), multiplying gives 2.0–2.5 (moves *away* from the training timescale). It is inference-only, so Gate 3 metric runs both modes on one checkpoint; the winner replaces the default and is logged in RESULTS.md as a documented deviation. α value stays 2.0 (§1.3). | open | ⚠ |
| 9 | Spatial Anchor fusion order | `x = e_i + id_embed(+traj. adapter) + p_i`, `p_i` added after fusion, before layer 0, to trajectory **and** unknown detection tokens. Gate 4 bit-exactness test with `SPATIAL_ANCHOR=False`. | no | no |
| 10 | Box units in spatial loss | Normalised cxcywh (same as DEIM/MOTIP loss); GIoU via `utils/box_ops.generalized_box_iou` on xyxy-converted boxes, L1 on cxcywh. Asserted in test against the detection-loss format. | no | no |
| 11 | Loss weight convention | Total ID multiplier 1.0 × (λ_CE=1, λ_GIoU=2, λ_L1=5) as in the skill. | no | no |
| 12 | Newborn in renormalised softmax | Included in `C_i`; newborn's `b_gt` = token's own GT box. | no | no |
| 13 | PP22 + CrowdHuman co-training | Follow MOTIP's recipe (`DATASETS: [PersonPath22, CrowdHuman]`, `DATASET_WEIGHTS` integer repeat; MOTIP's CrowdHuman loader exists, PP22 loader is new). | no | no |
| 14 | Tracklet confirmation × context | See N6. | no | no |
| N1 | **Source of p_i** (§2) | `SPATIAL_ANCHOR_SOURCE=layer_input` (a): capture `query_pos_embed` at layer `eval_idx`. `final_box` (b) available as ablation. Rationale: Paper Fig. 2, "no new parameters/reuses an encoding the detector already computes", "Reinjected". Risk: p_i lags `b̂_i` by one refinement. | open in skill | ⚠ |
| N2 | Absent objects in GIoU affinity | `ID_SWITCH_ABSENT=skip`: slots with no box in frame t (not in image or occluded by the occlusion mask) are excluded from S (they keep their identity) because GIoU is undefined. `neg_one` (G=−1 vs everything) available. This slightly lowers the swap rate vs uniform on the same S; Gate 5 will show whether Tab. 8 numbers still hold. | n/a | no |
| N3 | "40k samples / epoch" and epoch counts | `SAMPLES_PER_EPOCH=40000` clips (stage 2) / images (stage 1) via sampler subsample. Stage 2 = **10 epochs** (Paper Fig. 5 shows epochs 0–9; MOTIP DanceTrack also 10). Per-epoch iterations: 40000/8 = 5000 (stage 2, 8 clips global), 40000/32 = 1250 (stage 1, 32 images global). N3b: `C_i` uses GT presence **before** random occlusion masking (the object is physically in the frame). | n/a | no |
| N4 | Keep `TrajectoryModeling` adapter | Keep (Paper §3.1: "everything else is inherited unchanged"), on trajectory features only, dim = DEIM hidden. Flag `TRAJECTORY_MODELING` lets us test removal later. | n/a | no |
| N5 | Input pipeline for DEIM | (i) No ImageNet normalisation; (ii) `eval_spatial_size=None` (dynamic anchors/pos-embed); (iii) aspect-preserving resize, shorter side from `generate_scales(base=800, repeat=…)`, longer ≤ 1440, all frames and both clips of a per-GPU batch share the scale (so no padding mixes distributions, since DEIM ignores masks); inference shorter 800 / longer ≤ 1440, size_divisibility 32. | n/a | no |
| N6 | N_init semantics | Hit count starts at 1 on the newborn frame; confirmed at `N_INIT` consecutive hits; the tentative frame(s) are **not** back-filled into the output file (online-faithful; `N_INIT_BACKFILL` flag for completeness); tentative embeddings *do* enter the context so they can be re-identified; unconfirmed + missed ⇒ evicted + slot freed. `N_INIT=1` must equal legacy output exactly (regression test). Verified later against Tab. 9 trend. | n/a | no |
| N7 | "6 groups of 100 IDs each" (App. A) vs K=50 | `NUM_TRAINING_IDS = min(100, K)`; 6 groups unchanged (`AUG_NUM_GROUPS=6`). Matters only for PP22 (K=300). | n/a | no |
| N8 | Gradient through p_i | Not detached (the ID loss can train `query_pos_head`); matches how `e_i` already receives ID-loss gradient in joint training. `SPATIAL_ANCHOR_DETACH=True` available. | n/a | no |
| N9 | Greedy protocol & CE normalisation | Greedy readout = MOTIP's `object-max` (already the repo default); IoU-weighted CE normalised by number of valid ids as today (not by Σw). | n/a | no |

Not an ambiguity but a **risk to the Gate-1 tolerance**: MOTIP's own docs report ≈1.5 HOTA run-to-run instability for *retrained* models (`docs/GET_STARTED.md:61`). The ±0.5 HOTA gate is therefore applied only to the **released checkpoint** evaluated with our pipeline; a short retrain is informational only.

---

## 6. Phase gates and exact commands

Placeholders: `NG` = number of GPUs, `$D` = datasets dir (`./datasets/`), `$C` = checkpoint. All GPU commands are run **by you / on your GPU box**; I will not launch any multi-hour run without your confirmation of the compute budget (see §8).

### Gate 0 — Plan (this document)
```bash
test -f PLAN.md && test -f RESULTS.md && git log --oneline -1
```
Pass: every §1 item of the skill is mapped to a file **and** a flag — traceability below; user approval of §4/§5.

| Skill §1 item | File(s) | Flag |
|---|---|---|
| 1.1 DEIM detector, e_i / b̂_i / p_i | `models/deim/*`, `models/motip/__init__.py` | `DETECTOR`, `MODEL_SIZE` |
| 1.1 Fusion by addition, FFN×2 | `models/motip/id_decoder.py` | `ID_FUSION`, `ID_FFN_DIM_RATIO` |
| 1.1 RoPE replaces learned bias | `models/motip/rope.py`, `id_decoder.py` | `POS_ENC`, `ROPE_ALPHA`, `ROPE_ALPHA_MODE` |
| 1.1 Dictionary K+1, decoder mirrors DEIM layers | `id_decoder.py`, `configs/motip2/size_*.yaml` | `NUM_ID_VOCABULARY`, `MODEL_SIZE` |
| 1.2 Detection loss (MAL/L1/GIoU/FGL/DDF) | `models/deim/criterion.py` | (config constants) |
| 1.2 ID loss (CE + GIoU + L1, IoU-weighted CE) | `models/motip/id_criterion.py` | `ID_LOSS_SPATIAL`, `ID_LAMBDA_*`, `ID_CE_IOU_WEIGHT` |
| 1.3 σ, N_init, λ_ID, α, resolution | `transforms.py`, `runtime_tracker.py`, `submit_and_evaluate.py` | `ID_SWITCH_SIGMA`, `N_INIT`, `ID_THRESH`, `ROPE_ALPHA`, `INFERENCE_MAX_LONGER` |
| 1.3 T=W, K, (λ_det, λ_newborn) | `configs/motip2/bench_*.yaml` | `SAMPLE_LENGTHS`, `MISS_TOLERANCE`, `NUM_ID_VOCABULARY`, `DET_THRESH`, `NEWBORN_THRESH` |
| 1.4 Two-stage schedule, AdamW, clip 0.1, EMA | `train.py`, `configs/motip2/pretrain_*.yaml` | `ONLY_DETR`, `USE_EMA`, `LR_SCHEDULER`, `LR_WARMUP_ITERS` |
| 1.4 LR/size, ID-dec ×2, backbone mult., WD | `train.py:get_param_groups`, `size_*.yaml` | `MODEL_SIZE` |
| 1.4 Clips 30/60, interval 1–4, 4 (8) det frames, batches | `naive_sampler.py`, `size_*.yaml` | `SAMPLE_LENGTHS`, `SAMPLE_INTERVALS`, `DETR_NUM_TRAIN_FRAMES`, `BATCH_SIZE` |
| 1.4 Image/photometric/multi-scale aug | `transforms.py`, `util.py` | `AUG_*`, `BASE_SIZE_REPEAT` |
| 1.4 Spatial ID Switch p=1, occlusion 0.5, 6×100 | `transforms.py` | `ID_SWITCH`, `AUG_TRAJECTORY_OCCLUSION_PROB`, `AUG_NUM_GROUPS`, `NUM_TRAINING_IDS` |
| 1.5 FIFO T, greedy, confirmation, α | `runtime_tracker.py` | `MISS_TOLERANCE`, `ASSIGNMENT_PROTOCOL`, `N_INIT`, `ROPE_ALPHA` |
| §7 Spatial ID Switches | `transforms.py` | `ID_SWITCH`, `ID_SWITCH_REFERENCE`, `ID_SWITCH_ABSENT` |
| §8 Spatial ID Loss | `id_criterion.py` | `ID_LOSS_SPATIAL` |
| §6 Spatial Anchor | `id_decoder.py`, `train.py`, `runtime_tracker.py` | `SPATIAL_ANCHOR`, `SPATIAL_ANCHOR_SOURCE` |

### Gate 1 — Baseline reproduction (Phase 1)
```bash
pip install -r requirements.txt            # per docs/INSTALL.md
(cd models/ops && sh make.sh)              # needs nvcc
mkdir -p pretrains && curl -L -o pretrains/r50_deformable_detr_motip_dancetrack.pth \
  https://github.com/MCG-NJU/MOTIP/releases/download/v0.1/r50_deformable_detr_motip_dancetrack.pth
accelerate launch --num_processes=$NG submit_and_evaluate.py --data-root $D \
  --inference-mode evaluate --config-path ./configs/r50_deformable_detr_motip_dancetrack.yaml \
  --inference-model ./pretrains/r50_deformable_detr_motip_dancetrack.pth \
  --outputs-dir ./outputs/baseline_motip/ --inference-dataset DanceTrack --inference-split val
python tools/log_result.py --run baseline_motip --benchmark DanceTrack --split val
```
Pass: `|HOTA − 64.0| ≤ 0.5` (Tab. 4, DanceTrack-val). DetA ≈ 75.4, AssA ≈ 54.7 recorded for reference. Latency: `python tools/benchmark_latency.py --config ./configs/r50_deformable_detr_motip_dancetrack.yaml --ckpt … --height 800 --width 1440 --dtype FP32` → expect ≈ 86.9 ms on a V100 (Tab. 4); on other GPUs report the **ratio** only.

### Gate 2 — DEIM detector (Phase 2)
```bash
python -m pytest tests/test_deim_wrapper.py -q                      # CPU
accelerate launch --num_processes=$NG train.py --data-root $D --exp-name deim_s_pretrain_dancetrack \
  --config-path ./configs/motip2/pretrain_s_dancetrack.yaml        # stage 1, needs budget OK
accelerate launch --num_processes=$NG submit_and_evaluate.py --data-root $D --inference-mode evaluate \
  --config-path ./configs/motip2/pretrain_s_dancetrack.yaml --inference-model $C \
  --inference-only-detr True --outputs-dir ./outputs/deim_s_pretrain_dancetrack/ \
  --inference-dataset DanceTrack --inference-split val
python tools/benchmark_latency.py --config ./configs/motip2/pretrain_s_dancetrack.yaml --ckpt $C \
  --part detector --height 800 --width 1440 --dtype FP32
```
Pass: unit tests green; DetA recorded (reference: full MOTIP2-S 74.0, MOTIP 75.4; **heuristic** flag if < 70, not from the paper); detector latency recorded (Tab. 4 V100 FP32: N 14.9 / S 21.8 / M 32.5 / L 45.0 / X 71.2 ms).

### Gate 3 — ID decoder structure (Phase 3)
Unit (CPU):
```bash
python -m pytest tests/test_rope.py tests/test_id_decoder.py -q
```
Checks: output shape `(…, num_tracks+num_dets, hidden)`; RoPE relative-offset invariance (score(q_i,k_j) unchanged when both positions shift by Δ, atol 1e-5); `POS_ENC=learned_rel, ID_FUSION=concat` bit-identical to the legacy module output for the same seed/weights; masks (causality, padding) preserved.
Metric (needs a trained MOTIP2-S on PP22+CrowdHuman, budget confirmation required; inference-only sweeps reuse one checkpoint):
```bash
for A in 1.0 1.5 2.0 2.5; do for M in multiply divide; do
  accelerate launch --num_processes=$NG submit_and_evaluate.py --data-root $D --inference-mode evaluate \
   --config-path ./configs/motip2/s_personpath22.yaml --inference-model $C \
   --rope-alpha $A --rope-alpha-mode $M --inference-dataset PersonPath22_Inference --inference-split test \
   --outputs-dir ./outputs/s_pp22_alpha${A}_${M}/
done; done
```
Pass: ordering α=1.0 < α=2.0 (Tab. 6: 60.4 → 61.6 HOTA) in the better of the two modes; `POS_ENC=learned_rel` run ≈ 61.3.

### Gate 4 — Spatial Anchor (Phase 4)
```bash
python -m pytest tests/test_spatial_anchor.py -q
```
Checks: `SPATIAL_ANCHOR=False` ⇒ decoder input identical bit-for-bit to Phase 3 (same seed/weights); with `True`, input = Phase-3 input + p_i exactly (`torch.equal`); p_i dim == hidden; no new parameters (`sum(p.numel())` equal).
Metric: DanceTrack-val S with/without anchor: 64.5 vs 64.0 HOTA (direction only, §12).

### Gate 5 — Spatial ID Switches (Phase 5)
```bash
python -m pytest tests/test_spatial_idsw.py -q
python tools/idsw_stats.py --data-root $D --config ./configs/motip2/bench_dancetrack.yaml --epoch 0 --mode spatial
python tools/idsw_stats.py --data-root $D --config ./configs/motip2/bench_dancetrack.yaml --epoch 0 --mode uniform
# same for PersonPath22 once its loader exists
```
Unit pass: π valid permutation of S; σ=1e6 swap rate within sampling noise of uniform over 10k draws; GIoU ≤ 0 pairs rarely swap at σ=0.5; `none` is identity; absent slots never gain/lose identity (N2).
Stats pass (Tab. 8, "within a few points"):

| Benchmark | Swap% spatial | LR% spatial | mean GIoU spatial | Swap% uniform | LR% uniform |
|---|---|---|---|---|---|
| DanceTrack | 10 | 26 | 0.20 | 37 | 85 |
| PersonPath22 | 10 | 31 | 0.15 | 43 | 94 |

If off: try `ID_SWITCH_REFERENCE=last_frame` and `ID_SWITCH_ABSENT=neg_one` (at most these two hypotheses) before editing code.

### Gate 6 — Spatial ID Loss (Phase 6)
```bash
python -m pytest tests/test_spatial_id_loss.py -q
```
Checks: Σ_c p_ic = 1 over `C_i`; newborn class contributes only localisation error to its own GT; `.grad is None` on detector params after `backward()` from the two spatial terms only (b̂ detached); IoU weight carries no grad; box format identical to detection loss; `ID_LOSS_SPATIAL=False` ⇒ loss == legacy CE.
Metric: DanceTrack-val S, `--id-loss-spatial False`: expect 63.5 HOTA / 55.5 AssA vs 64.5 / 56.5 (needs full-model run, budget confirmation).

### Gate 7 — Inference (Phase 7)
```bash
python -m pytest tests/test_tracker_confirmation.py -q
for N in 1 2 3; do accelerate launch --num_processes=$NG submit_and_evaluate.py --data-root $D \
  --inference-mode evaluate --config-path ./configs/motip2/s_dancetrack.yaml --inference-model $C \
  --n-init $N --inference-dataset DanceTrack --inference-split val --outputs-dir ./outputs/s_dt_ninit$N/; done
```
Pass (Tab. 9, DanceTrack val): IDSW 3287 → 2489 → 2356 for N_init 1→2→3, HOTA flat (64.3 / 64.5 / 64.5). `N_INIT=1` equals legacy tracker output byte-for-byte on a fixed checkpoint (regression).

### Gate 8 — Full configs + smoke (Phase 8)
```bash
accelerate launch --num_processes=$NG train.py --data-root $D --exp-name smoke_s_dt \
  --config-path ./configs/motip2/s_dancetrack.yaml --epochs 1 --samples-per-epoch 800
```
Pass: loss decreases over ~100 iters, no NaN/Inf, per-GPU peak memory fits, Gate-5 statistics reproduced inside the live loop, iteration time recorded → **compute estimate for you** before any full run.

### Gate 9 — Evaluation and ablations (Phase 9)
Ablation list (a)–(i) in §11: retrain set = (a) full, (b) `--id-switch none`, (c) `--id-switch uniform`, (d) `--id-loss-spatial False`, (e) `--spatial-anchor False`, (f) all off (`uniform` on DanceTrack / `none` on PP22 per Tab. 5 footnote), (g) `--pos-enc learned_rel`; inference-only: (h) `--n-init 1/2/3`, (i) `--rope-alpha 1.0/1.5/2.0/2.5`. Stage-1 detector is **shared** across variants of one benchmark. Gains < 1 HOTA need ≥ 3 seeds (mean ± std via `tools/aggregate_results.py`). Pass criteria = §12 tolerances.

---

## 7. Target numbers (§11; verified against the PDF)

**Main (no extra data)** — tolerance ±1.0 HOTA / IDF1
| Benchmark | Model | HOTA | DetA | AssA | MOTA | IDF1 | Other |
|---|---|---|---|---|---|---|---|
| DanceTrack test | MOTIP2-L | 73.4 | 83.1 | 64.9 | 91.6 | 77.0 | — |
| DanceTrack test | MOTIP2-X | 74.8 | 84.3 | 66.5 | 92.5 | 78.7 | — |
| DanceTrack test | L / X w/o priors | 71.5 / 72.7 | 83.0 / 83.0 | 61.7 / 63.0 | 91.6 / 92.4 | 73.8 / 75.8 | Tab. 1 (note: the paper's text says DetA "moves by at most 0.2", but X 84.3 → 83.0 in the table) |
| SportsMOT test | MOTIP2-L | 76.0 | 85.6 | 67.5 | 93.8 | 78.9 | — |
| PersonPath22 test | MOTIP2-L | 62.8 | 66.1 | 60.4 | 75.8 | 71.1 | IDSW 5045, FP 18K, FN 40K |
| MOT17 test (App. B) | L / X | 60.4 / 60.7 | 63.2 / 63.9 | 58.0 / 58.0 | 76.3 / 77.3 | 72.7 / 72.6 | not required |

DanceTrack val, MOTIP2 sizes (Tab. 4/10): HOTA / DetA / AssA — N 59.2/69.3/50.9, **S 64.5/74.0/56.5**, M 67.3/76.4/59.7, L 68.0/77.3/60.0, X 70.1/80.7/61.1; MOTIP 64.0/75.4/54.7. S full-model also IDF1 68.8, MOTA 82.9, IDSW 2489 (Tab. 9).

**Ablations, MOTIP2-S (Tab. 5)** — direction must match; magnitudes may differ
| Variant | PP22 HOTA / AssA | DanceTrack val HOTA / AssA |
|---|---|---|
| Full | 61.6 / 60.3 | 64.5 / 56.5 |
| uniform IDSW | 56.1 / 54.4 | 63.7 / 55.8 |
| w/o IDSW | 59.5 / 55.8 | 60.1 / 49.7 |
| w/o Spatial ID Loss | 61.4 / 59.4 | 63.5 / 55.5 |
| w/o Spatial Anchor | 61.6 / 59.9 | 64.0 / 56.1 |
| w/o all priors† | 58.7 / 54.5 | 63.0 / 54.0 |

† data-stage prior removed (`none`) on PP22, uniform on DanceTrack (Tab. 5 footnote).

**Decoder/inference**: Tab. 6 (PP22 test) Rel-PE 61.3; RoPE α=1.0 / 1.5 / 2.0 / 2.5 → 60.4 / 61.0 / 61.6 / 61.6. Tab. 9 N_init 1/2/3 — PP22 IDSW 5612/5069/4911, DanceTrack-val 3287/2489/2356. Tab. 8 in Gate 5.

**Speed**: V100 PyTorch FP32 800×1440 — MOTIP 86.9 ms; MOTIP2 N 18.8, S 27.1, M 39.3, L 54.8, X 81.1; breakdown S = 21.8 det + 5.3 ID dec. T4 TensorRT FP16 — S 10.1 (9.4 + 0.7), L 23.2, X 40.1. Report S/MOTIP ratio (paper 3.2×).

**Tolerances (§12)**: baseline ±0.5 HOTA; main ±1.0 HOTA/IDF1; ablations direction-only; misses are logged, never tuned away silently.

---

## 8. Compute, ordering, and what I need from you

Run tiers (nothing above Tier A is started without your explicit go):
- **A — CPU, free, now**: unit tests for Gates 2–7 after `pip install torch` (CPU).
- **B — 1 GPU, minutes**: Gate 1 baseline eval; Gate 8 smoke (800 samples) ⇒ iteration time ⇒ budget estimate for you.
- **C — S-size DanceTrack-val**: stage 1 once + stage 2 × (full + b,c,d,e,f,g) = 1 + 7 trainings per seed; inference-only (h),(i) reuse checkpoints.
- **D — PersonPath22+CrowdHuman S**: same set; requires the new PP22 loader and data.
- **E — L/X, SportsMOT, PP22 main results, ≥3 seeds** for sub-1-HOTA claims.

Paper hardware: 4× V100; global batch 32 images (stage 1) / 8 clips (stage 2) (N: 128 / 16). I will size per-GPU batch to whatever you have and keep the global batch identical.

**Questions for you (also summarised in my hand-off message)**
1. Approve/redirect the six ⚠ decisions: #3 weight decay, #4 stage-1 length, #6 IDSW reference frame, #8 α mechanics, N1 p_i source.
2. Hardware/budget for Tier B–C (GPU type and count; datasets and DEIM COCO weights available?).
3. Is PersonPath22 in scope for the first pass, or DanceTrack-val only until Gates 1–7 are green?
