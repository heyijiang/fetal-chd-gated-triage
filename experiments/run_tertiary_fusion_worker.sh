#!/usr/bin/env bash
# Single-GPU fusion worker for tertiary **raw** appearance protocol.
# Caches are not in this repo. Override paths:
#   MANIFEST TAGS FEAT_CACHE EMBED_CACHE OUT_ROOT
# YOLO weights are not required for --embed-load-only (default).
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 \
#   JOBS="feature_transformer:42,feature_transformer:43" \
#   bash experiments/run_tertiary_fusion_worker.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
DATA_ROOT="${DATA_ROOT:-$ROOT/data}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
DEVICE="${DEVICE:-0}"
JOBS="${JOBS:?set JOBS=fusion:seed,fusion:seed,...}"

MANIFEST="${MANIFEST:-$DATA_ROOT/study_screening/manifest_tertiary_20241125_vs_chd_era2020.jsonl}"
TAGS="${TAGS:-$DATA_ROOT/study_screening/yolo_image_tags_tertiary_20241125.jsonl}"
FEAT_CACHE="${FEAT_CACHE:-$DATA_ROOT/study_screening/masvf_m0_tertiary_20241125_era2020_feature_cache.jsonl}"
EMBED_CACHE="${EMBED_CACHE:-$DATA_ROOT/study_screening/tertiary_20241125_vs_chd_raw_base_embeddings.jsonl}"
OUT_ROOT="${OUT_ROOT:-$ROOT/outputs/tertiary_20241125_vs_chd_fusion_raw}"

WITHIN_VIEW_POOL="${WITHIN_VIEW_POOL:-frame_attn}"
MISSING_FILL="${MISSING_FILL:-zero}"
POOL_MASK="${POOL_MASK:-present}"
GRAPH_LAYERS="${GRAPH_LAYERS:-1}"
LAMBDA_MVP="${LAMBDA_MVP:-0.1}"
EPOCHS="${EPOCHS:-40}"
PATIENCE="${PATIENCE:-10}"
FEAT_MODEL="${FEAT_MODEL:-m1}"
M0_FEATURES="${M0_FEATURES:-true_ratios_core3}"
EXTRA_FEATS="${EXTRA_FEATS:-none}"
MAX_PATIENTS="${MAX_PATIENTS:-0}"
FORCE="${FORCE:-0}"
NO_MVP="${NO_MVP:-0}"
NO_ANATOMY_ENCODER_MASK="${NO_ANATOMY_ENCODER_MASK:-0}"
GRAPH_ADJ="${GRAPH_ADJ:-}"

log() { echo "[$(date '+%F %T')] [gpu${CUDA_VISIBLE_DEVICES}] $*"; }

mkdir -p "$OUT_ROOT"
[[ -f "$MANIFEST" ]] || { log "ERROR: missing $MANIFEST"; exit 1; }
[[ -f "$TAGS" ]] || { log "ERROR: missing $TAGS"; exit 1; }
[[ -f "$FEAT_CACHE" ]] || { log "ERROR: missing M0 cache $FEAT_CACHE (run m0 phase first)"; exit 1; }
# Embed cache must exist for load-only; when building (EMBED_LOAD_ONLY=0) it may be created.
if [[ "${EMBED_LOAD_ONLY:-1}" == "1" ]]; then
  [[ -f "$EMBED_CACHE" ]] || { log "ERROR: missing $EMBED_CACHE"; exit 1; }
else
  mkdir -p "$(dirname "$EMBED_CACHE")"
  log "EMBED_LOAD_ONLY=0: will build/refresh embeds at $EMBED_CACHE"
fi

IFS=',' read -ra JOB_ARR <<< "$JOBS"
for job in "${JOB_ARR[@]}"; do
  job="$(echo "$job" | tr -d '[:space:]')"
  [[ -n "$job" ]] || continue
  fusion="${job%%:*}"
  seed="${job##*:}"
  tag="${fusion}_seed${seed}"
  # optional suffix for anatomy ablations
  if [[ -n "${RUN_TAG_SUFFIX:-}" ]]; then
    tag="${tag}${RUN_TAG_SUFFIX}"
  fi
  out="$OUT_ROOT/$tag"
  mkdir -p "$out"

  if [[ "$FORCE" != "1" ]] && ls "$out"/view_token_results_*.json >/dev/null 2>&1; then
    log "SKIP $tag"
    continue
  fi

  log "TRAIN $tag  feat=$FEAT_MODEL m0=$M0_FEATURES extra=$EXTRA_FEATS pool=$POOL_MASK gl=$GRAPH_LAYERS"
  extra=()
  if [[ "$fusion" == "graph_transformer" || "$fusion" == "anatomy_graph" ]]; then
    extra+=(--graph-layers "$GRAPH_LAYERS")
  fi
  if [[ "$NO_MVP" != "1" && ( "$fusion" == "graph_transformer" || "$fusion" == "anatomy_graph" || "${MVP:-0}" == "1" ) ]]; then
    extra+=(--mvp --lambda-mvp "$LAMBDA_MVP")
  fi
  if [[ "$NO_MVP" == "1" ]]; then
    extra+=(--no-mvp)
  fi
  if [[ "$fusion" == "anatomy_graph" && "$NO_ANATOMY_ENCODER_MASK" == "1" ]]; then
    extra+=(--no-anatomy-encoder-mask)
  fi
  if [[ -n "$GRAPH_ADJ" ]]; then
    extra+=(--graph-adj "$GRAPH_ADJ")
    log "  GRAPH_ADJ=$GRAPH_ADJ"
  fi
  if [[ "${NO_VIEW_DECOMP:-0}" == "1" ]]; then
    extra+=(--no-view-decomp)
  fi
  if [[ "${INCLUDE_EMPTY_VIEW_PATIENTS:-0}" == "1" ]]; then
    extra+=(--include-empty-view-patients)
  fi
  if [[ -n "${SWAP_VIEWS:-}" ]]; then
    extra+=(--swap-views "$SWAP_VIEWS")
    log "  SWAP_VIEWS=$SWAP_VIEWS"
  fi
  # T3 leave-one / only-view is computed inside fusion by default (view_decomp)

  # Optional FetalCLIP LoRA probe (BSPC M5): pass adapter + use LoRA embed cache.
  if [[ -n "${LORA_ADAPTER:-}" && -f "${LORA_ADAPTER}" ]]; then
    extra+=(--lora-adapter "$LORA_ADAPTER")
    log "  LoRA adapter: $LORA_ADAPTER"
  fi

  load_only_flag=(--embed-load-only)
  if [[ "${EMBED_LOAD_ONLY:-1}" == "0" ]]; then
    load_only_flag=()
    log "  embed-load-only OFF (will build/refresh embeds)"
  fi

  python -u experiments/masvf_view_token_fusion.py \
    --cohort private \
    --train-protocol private \
    --manifest "$MANIFEST" \
    --image-tags "$TAGS" \
    --feature-cache "$FEAT_CACHE" \
    --fetalclip-cache "$EMBED_CACHE" \
    --fusion "$fusion" \
    --feat-model "$FEAT_MODEL" \
    --m0-features "$M0_FEATURES" \
    --extra-feats "$EXTRA_FEATS" \
    --within-view-pool "$WITHIN_VIEW_POOL" \
    --missing-fill "$MISSING_FILL" \
    --pool-mask "$POOL_MASK" \
    --fallback drop \
    --clip-norm hybrid \
    "${load_only_flag[@]}" \
    --data-root "$ROOT/data" \
    --output-dir "$out" \
    --device "$DEVICE" \
    --seed "$seed" \
    --epochs "$EPOCHS" \
    --patience "$PATIENCE" \
    --max-patients "$MAX_PATIENTS" \
    "${extra[@]+"${extra[@]}"}" \
    2>&1 | tee "$out/run.log"
done

log "worker done"
