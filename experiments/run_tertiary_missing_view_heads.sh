#!/usr/bin/env bash
# Missing-view heads under matched CLIP + view_anat (Major M4).
#
# ALVG already logs view_decomp.keep_k in fusion_raw — we still launch it with
# NO_VIEW_DECOMP=0 so FORCE=1 can refresh. Fair FT/GNN must be re-run because
# feat-ablation used NO_VIEW_DECOMP=1.
#
#   GPUS=0,1,4,5 bash experiments/run_tertiary_missing_view_heads.sh
#   SEEDS=42,43,44 GPUS=0 bash experiments/run_tertiary_missing_view_heads.sh  # smoke
#
# Skip ALVG (use existing fusion_raw keep_k):
#   SKIP_ALVG=1 GPUS=0,1 bash experiments/run_tertiary_missing_view_heads.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

GPUS="${GPUS:-0,1,4,5}"
SEEDS="${SEEDS:-42,43,44,45,46,47,48,49,50,51}"
FORCE="${FORCE:-0}"
OUT_ROOT="${OUT_ROOT:-$ROOT/outputs/tertiary_missing_view_heads}"
SKIP_ALVG="${SKIP_ALVG:-0}"
SKIP_FT="${SKIP_FT:-0}"
SKIP_GNN="${SKIP_GNN:-0}"

DEFAULT_RECIPES=()
[[ "$SKIP_ALVG" != "1" ]] && DEFAULT_RECIPES+=("alvg:anatomy_graph:none:view_anat:1")
[[ "$SKIP_FT" != "1" ]] && DEFAULT_RECIPES+=("ft_clip_anat:feature_transformer:none:view_anat:0")
[[ "$SKIP_GNN" != "1" ]] && DEFAULT_RECIPES+=("gnn_clip_anat:graph_transformer:none:view_anat:1")

IFS=',' read -ra GPU_ARR <<< "$GPUS"
IFS=',' read -ra SEED_ARR <<< "$SEEDS"

log() { echo "[$(date '+%F %T')] [miss_view] $*" >&2; }
mkdir -p "$OUT_ROOT" logs_parallel

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
    OUT_ROOT="$OUT_ROOT" FORCE="$FORCE"
    NO_VIEW_DECOMP=0
    INCLUDE_EMPTY_VIEW_PATIENTS=0 FEAT_MODEL=m1
    M0_FEATURES="$m0" EXTRA_FEATS="$extra"
    RUN_TAG_SUFFIX="__r-${rid}"
  )
  if [[ "$mvp" == "1" ]]; then
    envx+=(MVP=1)
  else
    envx+=(NO_MVP=1)
  fi
  log "GPU$gpu recipe=$rid fusion=$fusion (view_decomp ON)"
  nohup env CUDA_VISIBLE_DEVICES="$gpu" DEVICE=0 JOBS="$jobs_csv" "${envx[@]}" \
    bash experiments/run_tertiary_fusion_worker.sh \
    >"logs_parallel/miss_view_${rid}_gpu${gpu}.log" 2>&1 &
  LAST_PID=$!
}

if [[ ${#DEFAULT_RECIPES[@]} -eq 0 ]]; then
  log "nothing to run (all SKIP_*=1)"
  exit 0
fi

log "=== missing-view heads → $OUT_ROOT ==="
PIDS=()
idx=0
for rec in "${DEFAULT_RECIPES[@]}"; do
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

log "summarize"
python -u experiments/summarize_alvg_missing_views.py \
  --roots \
    "$OUT_ROOT" \
    "$ROOT/outputs/tertiary_20241125_vs_chd_fusion_raw" \
  --latex \
  --out-dir "$ROOT/outputs/tertiary_missing_view_summary"

log "DONE → $OUT_ROOT + outputs/tertiary_missing_view_summary/"
