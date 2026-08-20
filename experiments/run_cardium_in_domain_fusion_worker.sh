#!/usr/bin/env bash
# Single-GPU CARDIUM **in-domain** fusion worker (train+test on CARDIUM folds).
# NOT external transfer from tertiary.
#
#   CUDA_VISIBLE_DEVICES=0 DEVICE=0 \
#   JOBS="anatomy_graph:42,anatomy_graph:43" FOLDS=1,2,3 \
#   bash experiments/run_cardium_in_domain_fusion_worker.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
DEVICE="${DEVICE:-0}"
JOBS="${JOBS:?set JOBS=fusion:seed,...}"
FOLDS="${FOLDS:-1,2,3}"

CARD_TAGS="${CARD_TAGS:-$ROOT/data/study_screening/yolo_image_tags_cardium.jsonl}"
CARD_FEAT="${CARD_FEAT:-$ROOT/data/study_screening/masvf_m0_cardium_feature_cache.jsonl}"
CARD_EMBED="${CARD_EMBED:-$ROOT/data/study_screening/cardium_raw_base_embeddings.jsonl}"
CARDIUM_PROCESSED="${CARDIUM_PROCESSED:-$ROOT/CARDIUM dataset/processed}"
OUT_ROOT="${OUT_ROOT:-$ROOT/outputs/cardium_in_domain_fusion}"

WITHIN_VIEW_POOL="${WITHIN_VIEW_POOL:-frame_attn}"
MISSING_FILL="${MISSING_FILL:-zero}"
POOL_MASK="${POOL_MASK:-present}"
GRAPH_LAYERS="${GRAPH_LAYERS:-1}"
FEAT_MODEL="${FEAT_MODEL:-m1}"
M0_FEATURES="${M0_FEATURES:-none}"
EXTRA_FEATS="${EXTRA_FEATS:-view_anat}"
EPOCHS="${EPOCHS:-40}"
PATIENCE="${PATIENCE:-10}"
FORCE="${FORCE:-0}"
NO_MVP="${NO_MVP:-0}"
NO_ANATOMY_ENCODER_MASK="${NO_ANATOMY_ENCODER_MASK:-0}"
NO_VIEW_DECOMP="${NO_VIEW_DECOMP:-1}"

log() { echo "[$(date '+%F %T')] [cardium_id gpu${CUDA_VISIBLE_DEVICES}] $*"; }

mkdir -p "$OUT_ROOT"
[[ -f "$CARD_TAGS" ]] || { log "ERROR missing $CARD_TAGS"; exit 1; }
[[ -f "$CARD_EMBED" ]] || { log "ERROR missing $CARD_EMBED"; exit 1; }
[[ -f "$CARD_FEAT" ]] || { log "ERROR missing $CARD_FEAT"; exit 1; }
[[ -d "$CARDIUM_PROCESSED" ]] || { log "ERROR missing $CARDIUM_PROCESSED"; exit 1; }

IFS=',' read -ra JOB_ARR <<< "$JOBS"
for job in "${JOB_ARR[@]}"; do
  job="$(echo "$job" | tr -d '[:space:]')"
  [[ -n "$job" ]] || continue
  fusion="${job%%:*}"
  seed="${job##*:}"
  tag="${fusion}_seed${seed}"
  [[ -n "${RUN_TAG_SUFFIX:-}" ]] && tag="${tag}${RUN_TAG_SUFFIX}"
  out="$OUT_ROOT/$tag"
  mkdir -p "$out"

  if [[ "$FORCE" != "1" ]] && ls "$out"/view_token_results_*.json >/dev/null 2>&1; then
    log "SKIP $tag"
    continue
  fi

  log "TRAIN $tag m0=$M0_FEATURES extra=$EXTRA_FEATS folds=$FOLDS"
  extra=()
  if [[ "$fusion" == "graph_transformer" || "$fusion" == "anatomy_graph" ]]; then
    extra+=(--graph-layers "$GRAPH_LAYERS")
  fi
  if [[ "$NO_MVP" != "1" && ( "$fusion" == "graph_transformer" || "$fusion" == "anatomy_graph" || "${MVP:-0}" == "1" ) ]]; then
    extra+=(--mvp --lambda-mvp "${LAMBDA_MVP:-0.1}")
  fi
  if [[ "$NO_MVP" == "1" ]]; then
    extra+=(--no-mvp)
  fi
  if [[ "$fusion" == "anatomy_graph" && "$NO_ANATOMY_ENCODER_MASK" == "1" ]]; then
    extra+=(--no-anatomy-encoder-mask)
  fi
  if [[ "$NO_VIEW_DECOMP" == "1" ]]; then
    extra+=(--no-view-decomp)
  fi

  python -u experiments/masvf_view_token_fusion.py \
    --cohort cardium \
    --cardium-processed "$CARDIUM_PROCESSED" \
    --folds "$FOLDS" \
    --image-tags "$CARD_TAGS" \
    --feature-cache "$CARD_FEAT" \
    --fetalclip-cache "$CARD_EMBED" \
    --fusion "$fusion" \
    --feat-model "$FEAT_MODEL" \
    --m0-features "$M0_FEATURES" \
    --extra-feats "$EXTRA_FEATS" \
    --within-view-pool "$WITHIN_VIEW_POOL" \
    --missing-fill "$MISSING_FILL" \
    --pool-mask "$POOL_MASK" \
    --fallback drop \
    --clip-norm hybrid \
    --embed-load-only \
    --data-root "$ROOT/data" \
    --output-dir "$out" \
    --device "$DEVICE" \
    --seed "$seed" \
    --epochs "$EPOCHS" \
    --patience "$PATIENCE" \
    "${extra[@]+"${extra[@]}"}" \
    2>&1 | tee "$out/run.log"
done
log "worker done"
