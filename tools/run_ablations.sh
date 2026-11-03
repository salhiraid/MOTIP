#!/usr/bin/env bash
# MOTIP2 PLAN Gate 9 / skill §11: emits (DRY_RUN=1, default) or runs (DRY_RUN=0) the ablation list in order.
#   SIZE=S BENCH=dancetrack SEED=42 NG=4 DATA=./datasets bash tools/run_ablations.sh
# (a) full  (b) w/o IDSW  (c) uniform IDSW  (d) w/o Spatial ID Loss  (e) w/o Spatial Anchor  (f) all priors off
# (g) learned relative PE instead of RoPE  -> retrains;  (h) N_init sweep  (i) alpha sweep (x2 modes) -> inference only.
# The stage-1 detector (pretrain_${size}_${bench}) is SHARED by all variants; train it once and point DETR_PRETRAIN to it.
set -euo pipefail
SIZE=${SIZE:-S}; BENCH=${BENCH:-dancetrack}; SEED=${SEED:-42}; NG=${NG:-4}; DATA=${DATA:-./datasets}; DRY_RUN=${DRY_RUN:-1}
s=$(echo "$SIZE" | tr '[:upper:]' '[:lower:]')
CFG=./configs/motip2/${s}_${BENCH}.yaml
case "$BENCH" in personpath22) IDS=PersonPath22_Inference; SPLIT=test; ALLOFF_IDSW=none;; *) IDS=${INFER_DATASET:-DanceTrack}; SPLIT=val; ALLOFF_IDSW=uniform;; esac
run() { echo "$*"; if [ "$DRY_RUN" = "0" ]; then eval "$@"; fi; }
train() {  # name, extra flags
  run "accelerate launch --num_processes=$NG train.py --data-root $DATA --config-path $CFG --seed $SEED --exp-name ${s}_${BENCH}_$1_seed$SEED --outputs-dir ./outputs/${s}_${BENCH}_$1_seed$SEED ${*:2}"
}
evalc() {  # name, ckpt, extra flags
  run "accelerate launch --num_processes=$NG submit_and_evaluate.py --data-root $DATA --config-path $CFG --inference-mode evaluate --inference-model $2 --inference-dataset $IDS --inference-split $SPLIT --outputs-dir ./outputs/${s}_${BENCH}_$1 ${*:3}"
}
echo "# --- stage 1 (once per size/benchmark) ---"
run "accelerate launch --num_processes=$NG train.py --data-root $DATA --config-path ./configs/motip2/pretrain_${s}_${BENCH}.yaml --exp-name deim_${s}_${BENCH}_stage1"
echo "# --- retrained variants (Tab. 5 / Tab. 6) ---"
train full
train noidsw        --id-switch none
train uniformidsw   --id-switch uniform
train nospatialloss --id-loss-spatial False
train noanchor      --spatial-anchor False
train alloff        --id-switch $ALLOFF_IDSW --id-loss-spatial False --spatial-anchor False
train learnedrelpe  --pos-enc learned_rel
echo "# --- inference-only sweeps on the FULL model checkpoint (Tab. 9 / Tab. 6) ---"
FULL=./outputs/${s}_${BENCH}_full_seed$SEED/checkpoint_9.pth
for n in 1 2 3; do evalc ninit$n $FULL --n-init $n; done
for a in 1.0 1.5 2.0 2.5; do for m in multiply divide; do evalc alpha${a}_$m $FULL --rope-alpha $a --rope-alpha-mode $m; done; done
