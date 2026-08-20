#!/usr/bin/env bash
# Graph-structure ablation for BSPC (same CLIP+view_anat+MVP recipe as ALVG).
#
# Arms (fusion=anatomy_graph unless noted):
#   anatomy  — anatomical adjacency (main ALVG; skip if dumps already exist)
#   random   — permute anatomy node labels (seeded; same degree, shuffled pairing)
#   full     — fully connected among present slots
#   mean     — no graph attention: mean-pool present slots, then Transformer
#   view_mean — fusion=view_mean (slot mean-pool, no imputer / no Transformer)
#
# Protocol matches experiments/run_tertiary_alvg.sh:
#   FEAT_MODEL=m1  M0_FEATURES=none  EXTRA_FEATS=view_anat  MVP=1  EMBED_LOAD_ONLY=1
#
# Usage (on the GPU box):
#   GPUS=0,1,4,5 bash experiments/run_tertiary_graph_ablation.sh
#   # smoke
#   GPUS=0 SEEDS=42 MODES=random,full,mean bash experiments/run_tertiary_graph_ablation.sh
#   # skip anatomy if 10-seed ALVG dumps already exist
#   SKIP_ANATOMY=1 GPUS=0,1,4,5 bash experiments/run_tertiary_graph_ablation.sh
#   python -u experiments/summarize_tertiary_graph_ablation.py
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

GPUS="${GPUS:-0}"
SEEDS="${SEEDS:-42,43,44,45,46,47,48,49,50,51}"
MODES="${MODES:-anatomy,random,full,mean,view_mean}"
FORCE="${FORCE:-0}"
SKIP_ANATOMY="${SKIP_ANATOMY:-0}"
INCLUDE_EMPTY="${INCLUDE_EMPTY_VIEW_PATIENTS:-0}"
OUT_ROOT="${OUT_ROOT:-$ROOT/outputs/tertiary_graph_ablation}"

IFS=',' read -ra GPU_ARR <<< "$GPUS"
IFS=',' read -ra SEED_ARR <<< "$SEEDS"
IFS=',' read -ra MODE_ARR <<< "$MODES"

log() { echo "[$(date '+%F %T')] [graph-abl] $*"; }

idx=0
for mode in "${MODE_ARR[@]}"; do
  mode="$(echo "$mode" | tr -d '[:space:]')"
  [[ -n "$mode" ]] || continue
  if [[ "$mode" == "anatomy" && "$SKIP_ANATOMY" == "1" ]]; then
    log "SKIP anatomy arm (SKIP_ANATOMY=1)"
    continue
  fi
  for seed in "${SEED_ARR[@]}"; do
    seed="$(echo "$seed" | tr -d '[:space:]')"
    [[ -n "$seed" ]] || continue
    gpu="${GPU_ARR[$((idx % ${#GPU_ARR[@]}))]}"
    idx=$((idx + 1))

    if [[ "$mode" == "view_mean" ]]; then
      fusion="view_mean"
      suffix="__adj_view_mean"
      graph_adj=""
      mvp_flag=0
      no_mvp_flag=1
    else
      fusion="anatomy_graph"
      suffix="__adj_${mode}"
      graph_adj="$mode"
      mvp_flag=1
      no_mvp_flag=0
    fi

    log "launch mode=$mode seed=$seed gpu=$gpu fusion=$fusion"
    CUDA_VISIBLE_DEVICES="$gpu" DEVICE=0 \
      FEAT_MODEL=m1 \
      M0_FEATURES=none \
      EXTRA_FEATS=view_anat \
      MVP="$mvp_flag" \
      NO_MVP="$no_mvp_flag" \
      LAMBDA_MVP="${LAMBDA_MVP:-0.1}" \
      FORCE="$FORCE" \
      INCLUDE_EMPTY_VIEW_PATIENTS="$INCLUDE_EMPTY" \
      NO_VIEW_DECOMP="${NO_VIEW_DECOMP:-1}" \
      GRAPH_ADJ="$graph_adj" \
      RUN_TAG_SUFFIX="$suffix" \
      OUT_ROOT="$OUT_ROOT" \
      JOBS="${fusion}:${seed}" \
      bash experiments/run_tertiary_fusion_worker.sh &
  done
done
wait
log "graph ablation jobs done → $OUT_ROOT"
log "summarize: python -u experiments/summarize_tertiary_graph_ablation.py --root $OUT_ROOT"
