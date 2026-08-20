#!/usr/bin/env bash
# B2: systematic RVOT↔3VT YOLO slot swap, then gated ALVG (same recipe as main).
#
#   GPUS=0,1,4,5 bash experiments/run_tertiary_yolo_swap_rvot_vvt.sh
#   SEEDS=42 GPUS=0 bash experiments/run_tertiary_yolo_swap_rvot_vvt.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p outputs logs_parallel

GPUS="${GPUS:-0}"
SEEDS="${SEEDS:-42,43,44,45,46,47,48,49,50,51}"
FORCE="${FORCE:-0}"
OUT_ROOT="${OUT_ROOT:-$ROOT/outputs/tertiary_yolo_swap_rvot_vvt}"
SWAP_VIEWS="${SWAP_VIEWS:-rvot,vvt}"

IFS=',' read -ra GPU_ARR <<< "$GPUS"
IFS=',' read -ra SEED_ARR <<< "$SEEDS"
log() { echo "[$(date '+%F %T')] [yolo_swap] $*"; }

idx=0
for seed in "${SEED_ARR[@]}"; do
  seed="$(echo "$seed" | tr -d '[:space:]')"
  [[ -n "$seed" ]] || continue
  gpu="${GPU_ARR[$((idx % ${#GPU_ARR[@]}))]}"
  idx=$((idx + 1))
  log "ALVG seed=$seed gpu=$gpu swap=$SWAP_VIEWS"
  CUDA_VISIBLE_DEVICES="$gpu" DEVICE=0 \
    FEAT_MODEL=m1 M0_FEATURES=none EXTRA_FEATS=view_anat MVP=1 \
    FORCE="$FORCE" INCLUDE_EMPTY_VIEW_PATIENTS=0 \
    SWAP_VIEWS="$SWAP_VIEWS" OUT_ROOT="$OUT_ROOT" \
    JOBS="anatomy_graph:${seed}" \
    bash experiments/run_tertiary_fusion_worker.sh &
done
wait
log "DONE → $OUT_ROOT"
log "Compare F1 to main ALVG 0.847±0.018"
