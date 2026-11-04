#!/usr/bin/env bash
# Export MOTIP2 detector + ID decoder as ONNX for several model sizes (default S and M) -- the files you copy to the Jetson Orin
# and turn into TensorRT FP16 / INT8 engines there (docs/DEPLOY_ORIN.md).
#
#   CKPT_S=outputs/s_vehicles_kp/checkpoint_9.pth CKPT_M=outputs/m_vehicles_kp/checkpoint_9.pth \
#       bash tools/export_orin.sh              # -> export/orin/{s,m}/{detector.onnx,id_decoder.onnx,export_meta.json}
#
# Environment: SIZES="S M"   HEIGHT=800 WIDTH=1440   VARIANT=vehicles_kp (or vehicles = no keypoints)   OUT=export/orin
#              VERIFY=1 (onnxruntime parity check, default on)   DRY_RUN=1 (only print the commands)
# HEIGHT x WIDTH must be the size of your resized frame padded to a multiple of 32 (see docs/DEPLOY_ORIN.md, section 2).
set -euo pipefail
SIZES=${SIZES:-"S M"}; HEIGHT=${HEIGHT:-800}; WIDTH=${WIDTH:-1440}; VARIANT=${VARIANT:-vehicles_kp}; OUT=${OUT:-export/orin}
VERIFY=${VERIFY:-1}
for s in $SIZES; do
  lower=$(echo "$s" | tr '[:upper:]' '[:lower:]')
  ck_var="CKPT_${s}"; ck=${!ck_var:-}
  cmd=(python tools/export_onnx.py --config "configs/motip2/${lower}_${VARIANT}.yaml" --out-dir "${OUT}/${lower}" --height "$HEIGHT" --width "$WIDTH")
  if [ -n "$ck" ]; then cmd+=(--ckpt "$ck"); else echo "# WARNING: ${ck_var} is not set -> RANDOM weights (smoke export only)" >&2; fi
  [ "$VERIFY" = "1" ] && cmd+=(--verify)
  echo "${cmd[*]}"
  [ "${DRY_RUN:-0}" = "1" ] || "${cmd[@]}"
done
