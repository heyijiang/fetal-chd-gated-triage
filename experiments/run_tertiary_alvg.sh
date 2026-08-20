#!/usr/bin/env bash
# ALVG (Anatomy-Linked View Graph) — main fusion recipe for tertiary review revision.
#
# Config: fusion=anatomy_graph, m0=none (CLIP-only), extra_feats=view_anat, MVP on.
# Seeds 42–51 by default; shard across GPUs via GPUS=0,1,4,5.
#
#   GPUS=0,1,4,5 bash experiments/run_tertiary_alvg.sh
#   GPUS=0 SEEDS=42 bash experiments/run_tertiary_alvg.sh   # smoke test
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

GPUS="${GPUS:-0}"
SEEDS="${SEEDS:-42,43,44,45,46,47,48,49,50,51}"
FORCE="${FORCE:-0}"
INCLUDE_EMPTY="${INCLUDE_EMPTY_VIEW_PATIENTS:-0}"

IFS=',' read -ra GPU_ARR <<< "$GPUS"
IFS=',' read -ra SEED_ARR <<< "$SEEDS"

log() { echo "[$(date '+%F %T')] [alvg] $*"; }

idx=0
for seed in "${SEED_ARR[@]}"; do
  seed="$(echo "$seed" | tr -d '[:space:]')"
  [[ -n "$seed" ]] || continue
  gpu="${GPU_ARR[$((idx % ${#GPU_ARR[@]}))]}"
  idx=$((idx + 1))
  log "launch seed=$seed gpu=$gpu"
  CUDA_VISIBLE_DEVICES="$gpu" DEVICE=0 \
    FEAT_MODEL=m1 \
    M0_FEATURES=none \
    EXTRA_FEATS=view_anat \
    MVP=1 \
    LAMBDA_MVP="${LAMBDA_MVP:-0.1}" \
    FORCE="$FORCE" \
    INCLUDE_EMPTY_VIEW_PATIENTS="$INCLUDE_EMPTY" \
    JOBS="anatomy_graph:${seed}" \
    bash experiments/run_tertiary_fusion_worker.sh &
done
wait
log "ALVG seeds done"
