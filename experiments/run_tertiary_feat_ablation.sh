#!/usr/bin/env bash
# Feature ablation: does core3 matter? Fair head comparison under ALVG default channels.
#
# Recipes (gated tertiary, n_te≈888):
#   mil_clip          attention_mil + m0=none + extra=none
#   mil_clip_anat     attention_mil + m0=none + extra=view_anat   ← fair vs ALVG
#   gnn_clip_anat     graph_transformer + m0=none + extra=view_anat
#   ft_clip_anat      feature_transformer + m0=none + extra=view_anat
#
# Existing refs (do not re-run unless FORCE=1):
#   attention_mil + core3  → fusion_raw (F1≈0.829)
#   anatomy_graph + none+view_anat → fusion_raw (F1≈0.847)
#
#   GPUS=0,1,4,5 bash experiments/run_tertiary_feat_ablation.sh
#   SEEDS=42,43,44 GPUS=0 bash experiments/run_tertiary_feat_ablation.sh  # smoke
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

GPUS="${GPUS:-0,1,4,5}"
SEEDS="${SEEDS:-42,43,44,45,46,47,48,49,50,51}"
FORCE="${FORCE:-0}"
NO_VIEW_DECOMP="${NO_VIEW_DECOMP:-1}"
OUT_ROOT="${OUT_ROOT:-$ROOT/outputs/tertiary_feat_ablation}"

DEFAULT_RECIPES=(
  "mil_clip:attention_mil:none:none:0"
  "mil_clip_anat:attention_mil:none:view_anat:0"
  "gnn_clip_anat:graph_transformer:none:view_anat:1"
  "ft_clip_anat:feature_transformer:none:view_anat:0"
)

RECIPES_CSV="${RECIPES:-}"
if [[ -n "$RECIPES_CSV" ]]; then
  IFS=',' read -ra WANT <<< "$RECIPES_CSV"
  SELECTED=()
  for want in "${WANT[@]}"; do
    want="$(echo "$want" | tr -d '[:space:]')"
    for rec in "${DEFAULT_RECIPES[@]}"; do
      [[ "${rec%%:*}" == "$want" ]] && SELECTED+=("$rec") && break
    done
  done
  RECIPE_ARR=("${SELECTED[@]}")
else
  RECIPE_ARR=("${DEFAULT_RECIPES[@]}")
fi

IFS=',' read -ra GPU_ARR <<< "$GPUS"
IFS=',' read -ra SEED_ARR <<< "$SEEDS"

log() { echo "[$(date '+%F %T')] [feat_abl] $*" >&2; }
mkdir -p "$OUT_ROOT" logs_parallel

# Wait that works even if job was started under nohup / reparented.
wait_pid() {
  local pid="$1"
  [[ -n "$pid" ]] || return 0
  if wait "$pid" 2>/dev/null; then
    return 0
  fi
  while kill -0 "$pid" 2>/dev/null; do
    sleep 20
  done
}

# NOTE: do NOT call via $(launch ...) — that runs in a subshell and breaks wait.
launch() {
  local rec="$1" gpu="$2"
  local rid fusion m0 extra mvp
  IFS=':' read -r rid fusion m0 extra mvp <<< "$rec"
  local jobs=() seed
  for seed in "${SEED_ARR[@]}"; do
    seed="$(echo "$seed" | tr -d '[:space:]')"
    [[ -n "$seed" ]] || continue
    jobs+=("${fusion}:${seed}")
  done
  local jobs_csv; jobs_csv=$(IFS=','; echo "${jobs[*]}")
  local envx=(
    OUT_ROOT="$OUT_ROOT" FORCE="$FORCE" NO_VIEW_DECOMP="$NO_VIEW_DECOMP"
    INCLUDE_EMPTY_VIEW_PATIENTS=0 FEAT_MODEL=m1
    M0_FEATURES="$m0" EXTRA_FEATS="$extra"
    RUN_TAG_SUFFIX="__r-${rid}"
  )
  if [[ "$mvp" == "1" ]]; then
    envx+=(MVP=1)
  else
    envx+=(NO_MVP=1)
  fi
  log "GPU$gpu recipe=$rid fusion=$fusion m0=$m0 extra=$extra"
  nohup env CUDA_VISIBLE_DEVICES="$gpu" DEVICE=0 JOBS="$jobs_csv" "${envx[@]}" \
    bash experiments/run_tertiary_fusion_worker.sh \
    >"logs_parallel/feat_abl_${rid}_gpu${gpu}.log" 2>&1 &
  LAST_PID=$!
}

log "=== feature ablation (core3 death check) ==="
PIDS=()
idx=0
for rec in "${RECIPE_ARR[@]}"; do
  gpu="${GPU_ARR[$((idx % ${#GPU_ARR[@]}))]}"
  idx=$((idx + 1))
  launch "$rec" "$gpu"
  [[ "$LAST_PID" =~ ^[0-9]+$ ]] || { log "bad pid $LAST_PID"; exit 1; }
  PIDS+=("$LAST_PID")
  log "launched pid=$LAST_PID"
done
for pid in "${PIDS[@]}"; do
  log "waiting $pid"
  wait_pid "$pid"
done
log "DONE → $OUT_ROOT"
log "summarize: python -u experiments/summarize_fusion_vs_mil.py --fusion-root $OUT_ROOT --also-root outputs/tertiary_20241125_vs_chd_fusion_raw"
